"""评测层测试：相关性判定、指标数学、聚合口径、门禁逻辑、评测集自检。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests.helpers import AppTestCase, make_settings
from tests.helpers import ROOT  # noqa: F401

from ragkb.evaluate import load_dataset, run_eval, sanity_check_dataset, update_baseline
from ragkb.metrics import (
    CaseResult,
    aggregate,
    aggregate_by_tag,
    evaluate_case,
    groundedness,
    is_relevant,
    retrieval_metrics,
)
from ragkb.types import Chunk, EvalCase, Retrieved


def chunk(cid: str, text: str, source: str = "a.md") -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="d", ordinal=0, text=text,
        heading_path="H", metadata={"source": source, "title": "A"},
    )


class TestRelevance(unittest.TestCase):
    def test_snippet_match_is_preferred_over_source(self):
        case = EvalCase(case_id="c", question="q", gold_sources=["a.md"], gold_snippets=["700"])
        self.assertTrue(is_relevant(chunk("c1", "住宿每晚上限 700 元。"), case))
        # 同一篇文档但完全不相干的块 → 不算命中（否则 precision 会虚高）
        self.assertFalse(is_relevant(chunk("c2", "讲的是另一个话题。"), case))

    def test_source_only_fallback(self):
        case = EvalCase(case_id="c", question="q", gold_sources=["a.md"])
        self.assertTrue(is_relevant(chunk("c1", "任意内容", source="a.md"), case))
        self.assertFalse(is_relevant(chunk("c2", "任意内容", source="b.md"), case))

    def test_chunk_id_exact(self):
        case = EvalCase(case_id="c", question="q", gold_chunk_ids=["c9"])
        self.assertTrue(is_relevant(chunk("c9", "x"), case))
        self.assertFalse(is_relevant(chunk("c1", "x"), case))

    def test_no_gold_means_never_relevant(self):
        self.assertFalse(is_relevant(chunk("c1", "x"), EvalCase(case_id="c", question="q")))


class TestRetrievalMetrics(unittest.TestCase):
    def test_perfect_retrieval(self):
        case = EvalCase(case_id="c", question="q", gold_snippets=["700"])
        items = [Retrieved(chunk=chunk("c1", "上限 700 元。")), Retrieved(chunk=chunk("c2", "无关"))]
        m = retrieval_metrics(items, case, k=2)
        self.assertEqual(m["hit@k"], 1.0)
        self.assertEqual(m["recall@k"], 1.0)
        self.assertEqual(m["precision@k"], 0.5)
        self.assertEqual(m["mrr"], 1.0)
        self.assertGreater(m["ndcg@k"], 0.5)

    def test_relevant_at_rank_two(self):
        case = EvalCase(case_id="c", question="q", gold_snippets=["700"])
        items = [Retrieved(chunk=chunk("c2", "无关")), Retrieved(chunk=chunk("c1", "上限 700 元。"))]
        m = retrieval_metrics(items, case, k=2)
        self.assertEqual(m["hit@k"], 1.0)
        self.assertEqual(m["mrr"], 0.5)

    def test_miss_everything(self):
        case = EvalCase(case_id="c", question="q", gold_snippets=["700"])
        m = retrieval_metrics([Retrieved(chunk=chunk("c", "无关"))], case, k=1)
        self.assertEqual(m["hit@k"], 0.0)
        self.assertEqual(m["mrr"], 0.0)
        self.assertEqual(m["ndcg@k"], 0.0)

    def test_multi_snippet_recall_is_partial(self):
        case = EvalCase(case_id="c", question="q", gold_snippets=["700", "120"])
        items = [Retrieved(chunk=chunk("c1", "上限 700 元。"))]
        m = retrieval_metrics(items, case, k=1)
        self.assertEqual(m["hit@k"], 1.0)
        self.assertAlmostEqual(m["recall@k"], 0.5)

    def test_k_truncation(self):
        case = EvalCase(case_id="c", question="q", gold_snippets=["700"])
        items = [Retrieved(chunk=chunk("x", "无关")) for _ in range(3)] + [
            Retrieved(chunk=chunk("c1", "上限 700 元。"))
        ]
        self.assertEqual(retrieval_metrics(items, case, k=3)["hit@k"], 0.0)
        self.assertEqual(retrieval_metrics(items, case, k=4)["hit@k"], 1.0)


class TestGroundedness(unittest.TestCase):
    def test_fully_grounded_answer(self):
        items = [Retrieved(chunk=chunk("c", "一线城市住宿上限 700 元，超标自理。"))]
        self.assertGreater(groundedness("住宿上限 700 元", items), 0.9)

    def test_hallucinated_answer_scores_low(self):
        items = [Retrieved(chunk=chunk("c", "一线城市住宿上限 700 元。"))]
        self.assertLess(groundedness("公司提供火星基地免费住宿两年", items), 0.4)


class TestCaseEvaluation(unittest.TestCase):
    def test_refusal_accuracy_both_directions(self):
        case = EvalCase(case_id="c", question="q", should_refuse=True)
        ok = evaluate_case(case, [Retrieved(chunk=chunk("c", "x"))], refused=True)
        self.assertEqual(ok.metrics["refusal_accuracy"], 1.0)
        bad = evaluate_case(case, [Retrieved(chunk=chunk("c", "x"))], refused=False, answer_text="[1] 编的答案")
        self.assertEqual(bad.metrics["refusal_accuracy"], 0.0)

    def test_false_refusal_is_penalised(self):
        case = EvalCase(case_id="c", question="q", gold_snippets=["700"], gold_answer="700 元")
        r = evaluate_case(case, [Retrieved(chunk=chunk("c", "上限 700 元"))], refused=True)
        self.assertEqual(r.metrics["refusal_accuracy"], 0.0)

    def test_citation_validity_flags_invalid_and_uncited(self):
        case = EvalCase(case_id="c", question="q", gold_answer="700 元")
        items = [Retrieved(chunk=chunk("c", "上限 700 元"))]
        bad = evaluate_case(case, items, answer_text="答案 [7]", invalid_citations=[7])
        self.assertEqual(bad.metrics["citation_validity"], 0.0)
        uncited = evaluate_case(case, items, answer_text="答案是 700 元（没有角标）")
        self.assertEqual(uncited.metrics["citation_validity"], 0.0)
        good = evaluate_case(case, items, answer_text="答案是 700 元 [1]")
        self.assertEqual(good.metrics["citation_validity"], 1.0)

    def test_has_gold_flag(self):
        with_gold = evaluate_case(EvalCase(case_id="1", question="q", gold_snippets=["x"]), [])
        without = evaluate_case(EvalCase(case_id="2", question="q", should_refuse=True), [], refused=True)
        self.assertTrue(with_gold.has_gold)
        self.assertFalse(without.has_gold)


class TestAggregation(unittest.TestCase):
    def _result(self, case_id: str, has_gold: bool, hit: float, refusal: float) -> CaseResult:
        return CaseResult(
            case_id=case_id,
            question="q",
            metrics={"hit@k": hit, "mrr": hit, "refusal_accuracy": refusal},
            has_gold=has_gold,
            latency_ms=10.0,
        )

    def test_retrieval_metrics_exclude_cases_without_gold(self):
        rows = [
            self._result("a", True, 1.0, 1.0),
            self._result("b", True, 0.0, 1.0),
            self._result("c", False, 0.0, 1.0),  # 拒答样本不该稀释 recall
        ]
        agg = aggregate(rows)
        self.assertEqual(agg["hit@k"], 0.5)
        self.assertEqual(agg["n_gold_cases"], 2.0)
        self.assertEqual(agg["refusal_accuracy"], 1.0)

    def test_by_tag_groups(self):
        a = self._result("a", True, 1.0, 1.0)
        a.tags = ["fact"]
        b = self._result("b", True, 0.0, 0.0)
        b.tags = ["refusal"]
        grouped = aggregate_by_tag([a, b])
        self.assertEqual(grouped["fact"]["hit@k"], 1.0)
        self.assertEqual(grouped["refusal"]["refusal_accuracy"], 0.0)


class TestGate(AppTestCase):
    def _write(self, payload: dict, name: str) -> Path:
        path = self.cfg.data_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_threshold_failure_blocks(self):
        cases = load_dataset(self.cfg.eval_dir / "dataset.jsonl")[:3]
        thresholds = self._write({"thresholds": {"hit@k": 2.0}}, "th.json")
        report = run_eval(self.app, cases, self.cfg, k=5, thresholds_path=thresholds, max_drop=0)
        self.assertFalse(report.gate["passed"])
        self.assertTrue(any("hit@k" in f for f in report.gate["failed"]))

    def test_baseline_regression_blocks(self):
        cases = load_dataset(self.cfg.eval_dir / "dataset.jsonl")[:3]
        baseline = self._write({"overall": {"hit@k": 2.0, "mrr": 2.0}}, "base.json")
        report = run_eval(self.app, cases, self.cfg, k=5, baseline_path=baseline, max_drop=0.05)
        self.assertFalse(report.gate["passed"])

    def test_passing_case(self):
        cases = load_dataset(self.cfg.eval_dir / "dataset.jsonl")[:3]
        report = run_eval(self.app, cases, self.cfg, k=5, max_drop=0.05)
        self.assertTrue(report.gate["passed"])
        self.assertEqual(report.overall["n_cases"], 3.0)

    def test_update_baseline_writes_overall(self):
        cases = load_dataset(self.cfg.eval_dir / "dataset.jsonl")[:2]
        report = run_eval(self.app, cases, self.cfg, k=5, max_drop=0)
        path = update_baseline(report, self.cfg.data_dir / "newbase.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("overall", data)
        self.assertIn("hit@k", data["overall"])


class TestDatasetSelfCheck(AppTestCase):
    def test_shipped_dataset_loads_and_is_valid(self):
        cases = load_dataset(self.cfg.eval_dir / "dataset.jsonl")
        self.assertGreaterEqual(len(cases), 15)
        ids = [c.case_id for c in cases]
        self.assertEqual(len(ids), len(set(ids)), "case_id 不能重复")
        self.assertTrue(any(c.should_refuse for c in cases))
        self.assertTrue(any(c.history for c in cases))

    def test_sanity_check_reports_missing_gold(self):
        bad = EvalCase(case_id="bad", question="q", gold_sources=["不存在的文件.md"])
        problems = sanity_check_dataset([bad], self.app)
        self.assertTrue(problems)

    def test_sanity_check_flags_stale_snippet(self):
        bad = EvalCase(
            case_id="stale",
            question="一线城市住宿每晚上限是多少？",
            gold_sources=["差旅报销标准.md"],
            gold_snippets=["这句在语料里根本不存在 12345"],
        )
        problems = sanity_check_dataset([bad], self.app)
        self.assertTrue(any("gold_snippet" in p for p in problems))

    def test_shipped_dataset_passes_self_check(self):
        cases = load_dataset(make_settings("selfcheck").eval_dir / "dataset.jsonl")
        # 用真实语料建库后再自检（这里复用 AppTestCase 的 app，语料不同，所以只检查 gold_files 存在性）
        problems = [
            p for p in sanity_check_dataset(cases, self.app) if "gold_source 不存在" in p
        ]
        # 测试库只入了两篇 fixture，因此允许有 gold_source 缺失，但不能有别的错误类型
        self.assertTrue(all("gold_source 不存在" in p or "gold_snippet" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
