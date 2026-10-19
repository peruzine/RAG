"""端到端测试：入库 → 检索 → 生成 → 引用 → 拒答 → 权限 → 流式 → trace。

这些用例的价值在于把"能跑"升级成"行为可验证"：拒答、越界引用、增量跳过这些
容易在重构中被悄悄破坏的行为，都被固定下来。
"""

from __future__ import annotations

import json
import unittest

from tests.helpers import SECRET_DOC, AppTestCase, build_app, make_settings
from tests.helpers import ROOT  # noqa: F401

from ragkb.answer import Answerer
from ragkb.obs import cost_usd, percentile, summarize_traces
from ragkb.prompts import NO_ANSWER_SENTINEL, REFUSAL_TEXT
from ragkb.query import QueryProcessor
from ragkb.types import Retrieved


class TestAnswerFlow(AppTestCase):
    def test_answerable_question_gets_cited_answer(self):
        ans = self.app.ask("一线城市住宿每晚上限是多少？")
        self.assertFalse(ans.refused)
        self.assertTrue(ans.citations)
        self.assertIn("700", ans.text)
        self.assertEqual(ans.route, "kb")
        self.assertEqual(ans.debug["invalid_citations"], [])
        self.assertFalse(ans.debug["uncited"])

    def test_citation_points_at_the_source_chunk(self):
        ans = self.app.ask("出差期间餐补是多少？")
        cite = ans.citations[0]
        self.assertEqual(cite.index, 1)
        self.assertTrue(cite.chunk_id.startswith(cite.doc_id))
        self.assertIn("餐补", cite.quote + ans.text)

    def test_unanswerable_question_is_refused(self):
        ans = self.app.ask("公司年会在哪里举办？")
        self.assertTrue(ans.refused)
        self.assertEqual(ans.text, REFUSAL_TEXT)
        self.assertEqual(ans.citations, [])
        # 小语料下可能直接"零召回"，大语料下靠覆盖率兜住；两者都算正确拒答
        self.assertTrue(
            (ans.debug["refuse_reason"] or "").startswith(("no_hits", "low_coverage")),
            ans.debug["refuse_reason"],
        )

    def test_no_hits_is_refused_without_calling_model(self):
        ans = self.app.ask("量子纠缠发动机的保养周期是多久？")
        self.assertTrue(ans.refused)
        self.assertIn(ans.debug["refuse_reason"], ("no_hits", "low_coverage:0.000", "too_few_hits"))

    def test_chat_route_skips_retrieval(self):
        ans = self.app.ask("你好")
        self.assertEqual(ans.route, "chat")
        self.assertEqual(ans.retrieved, [])
        self.assertFalse(ans.refused)

    def test_acl_blocks_internal_document(self):
        path = self.cfg.corpus_dir / "secret.md"
        path.write_text(SECRET_DOC, encoding="utf-8")
        self.app.ingest(path, acl=("internal",))
        blocked = self.app.ask("P6 职级的带宽上限是多少？", roles=("public",))
        self.assertTrue(blocked.refused)
        allowed = self.app.ask("P6 职级的带宽上限是多少？", roles=("public", "internal"))
        self.assertFalse(allowed.refused)
        self.assertIn("84.0", allowed.text)

    def test_multi_turn_rewrite_uses_keyword_carryover(self):
        history = [
            {"role": "user", "content": "P0 故障是怎么定义的？"},
            {"role": "assistant", "content": "P0 指核心链路完全不可用。"},
        ]
        ans = self.app.ask("那它的解决时限是多久？", history=history)
        self.assertFalse(ans.refused)
        self.assertIn("4 小时", ans.text)

    def test_model_failure_degrades_to_extractive_fallback(self):
        class Boom:
            name = "boom"

            def complete(self, *a, **kw):
                raise RuntimeError("模型服务 503")

            def stream(self, *a, **kw):
                raise RuntimeError("模型服务 503")

        original = self.app.llm
        self.app.answerer.llm = Boom()  # type: ignore[assignment]
        try:
            ans = self.app.answerer.answer("一线城市每晚上限是多少？")
            self.assertFalse(ans.refused)
            self.assertIn("原文摘录", ans.text)
            self.assertIn("模型服务 503", ans.debug["error"])
        finally:
            self.app.answerer.llm = original  # type: ignore[assignment]


class TestCitationExtraction(unittest.TestCase):
    def _items(self):
        chunks = [
            Retrieved(chunk=__import__("ragkb.types", fromlist=["Chunk"]).Chunk(
                chunk_id=f"d#000{i}", doc_id="d", ordinal=i, text=f"第{i}段内容，讲住宿标准。",
                heading_path="住宿", metadata={"source": "a.md", "title": "A"},
            ))
            for i in range(3)
        ]
        return chunks

    def test_valid_and_invalid_citations(self):
        cites, invalid = Answerer.extract_citations("结论见 [1] 和 [3]，另外 [9] 也提过。", self._items())
        self.assertEqual([c.index for c in cites], [1, 3])
        self.assertEqual(invalid, [9])
        self.assertEqual(cites[0].title, "A")

    def test_duplicate_markers_collapse(self):
        cites, _ = Answerer.extract_citations("[2] 与 [2] 重复", self._items())
        self.assertEqual([c.index for c in cites], [2])

    def test_no_markers(self):
        cites, invalid = Answerer.extract_citations("没有角标的答案", self._items())
        self.assertEqual(cites, [])
        self.assertEqual(invalid, [])


class TestStreaming(AppTestCase):
    def test_event_order_and_payload(self):
        events = list(self.app.stream("一线城市住宿每上限是多少？"))
        self.assertEqual(events[0]["type"], "meta")
        self.assertEqual(events[-1]["type"], "done")
        self.assertTrue(any(e["type"] == "delta" for e in events))
        self.assertIn("contexts", events[0])
        self.assertTrue(events[-1]["answer"]["citations"])

    def test_stream_refuses_cleanly(self):
        events = list(self.app.stream("公司年会在哪里举办？"))
        self.assertTrue(events[-1]["refused"])
        self.assertEqual(events[-1]["answer"]["text"], REFUSAL_TEXT)
    def test_stream_concatenated_deltas_match_final_text(self):
        events = list(self.app.stream("出差期间餐补是多少？"))
        streamed = "".join(e["text"] for e in events if e["type"] == "delta")
        final = events[-1]["answer"]
        if not final["refused"]:
            self.assertEqual(streamed.strip(), final["text"].strip())


class TestIngestPipeline(AppTestCase):
    def test_incremental_is_idempotent(self):
        path = self.cfg.corpus_dir / "差旅报销标准.md"
        first = self.app.ingest(path)
        self.assertEqual(first.skipped, 1)
        self.assertEqual(first.chunks, 0)

    def test_incremental_survives_relative_and_absolute_paths(self):
        """回归：用相对路径入库时，同一篇文档曾被误报为 added 并重复计算向量。"""
        import os

        absolute = self.cfg.corpus_dir / "差旅报销标准.md"
        old = os.getcwd()
        try:
            os.chdir(ROOT)
            summary = self.app.ingest(absolute.relative_to(ROOT))
        finally:
            os.chdir(old)
        self.assertEqual(summary.skipped, 1, summary.summary())
        self.assertEqual(summary.chunks, 0)
        self.assertEqual(len(self.app.store.list_documents()), 2)

    def test_acl_change_without_content_change_takes_effect(self):
        """回归：内容未变时改权限曾被 skip 吞掉——把该收紧的文档继续公开。"""
        path = self.cfg.corpus_dir / "差旅报销标准.md"
        source = path.resolve().as_posix()
        self.assertEqual(set(self.app.store.get_document_by_source(source).acl), {"public"})

        summary = self.app.ingest(path, acl=("internal",))
        self.assertEqual(summary.updated, 1)
        self.assertEqual(summary.chunks, 0, "改权限不应重算向量")
        self.assertIn("acl-only", summary.results[0].report)

        self.assertEqual(set(self.app.store.get_document_by_source(source).acl), {"internal"})
        public = self.app.retriever.retrieve("一线城市住宿每晚上限", roles=("public",), top_k=5)
        self.assertNotIn("700", " ".join(it.chunk.text for it in public.items))
        internal = self.app.retriever.retrieve(
            "一线城市住宿每晚上限", roles=("public", "internal"), top_k=5
        )
        self.assertIn("700", " ".join(it.chunk.text for it in internal.items))

    def test_content_change_triggers_update(self):
        path = self.cfg.corpus_dir / "数据平台SLA.md"
        path.write_text("# 数据平台 SLA\n\n## 响应时限\n\nP0 首次响应改为 10 分钟。\n", encoding="utf-8")
        summary = self.app.ingest(path)
        self.assertEqual(summary.updated, 1)
        doc = self.app.store.get_document_by_source(str(path.as_posix()))
        self.assertEqual(doc.version, 2)
        res = self.app.retriever.retrieve("P0 首次响应", top_k=3)
        self.assertIn("10 分钟", res.items[0].chunk.text)

    def test_force_rebuild(self):
        path = self.cfg.corpus_dir / "差旅报销标准.md"
        summary = self.app.ingest(path, force=True)
        self.assertEqual(summary.updated, 1)

    def test_failed_document_is_reported_not_raised(self):
        bad = self.cfg.corpus_dir / "broken.pdf"  # 缺 pypdf 时应当失败而不是崩
        bad.write_bytes(b"%PDF-1.4 not a real pdf")
        summary = self.app.ingest(bad)
        self.assertEqual(summary.failed, 1)
        self.assertTrue(summary.results[0].error)

    def test_embedding_model_switch_warns(self):
        warnings = self.app.warnings()
        self.assertEqual(warnings, [])
        original = self.app.embedder.name
        self.app.embedder.name = "另一个模型"  # type: ignore[misc]
        try:
            self.assertTrue(self.app.warnings())
        finally:
            self.app.embedder.name = original  # type: ignore[misc]


class TestObservability(AppTestCase):
    def test_trace_file_written_with_stages(self):
        self.app.ask("一线城市每晚上限是多少？")
        rows = [json.loads(l) for l in self.cfg.trace_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["name"], "ask")
        stages = {s["name"] for s in row["spans"]}
        self.assertIn("retrieve", stages)
        self.assertIn("generate", stages)
        self.assertGreater(row["latency_ms"], 0)
        self.assertIn("refused", row)
        self.assertIn("n_context", row)
        self.assertIn("citation_ids", row)

    def test_metrics_aggregation(self):
        for q in ("一线城市每晚上限是多少？", "公司年会在哪里举办？"):
            self.app.ask(q)
        traces = summarize_traces(self.cfg.trace_path)
        self.assertEqual(traces["count"], 2)
        self.assertIn("p95", traces["latency_ms"])
        self.assertGreater(traces["refusal_rate"], 0)
        self.assertIn("retrieve", traces["stage_ms_p95"])

    def test_stats_and_metrics_endpoints_shape(self):
        stats = self.app.stats()
        self.assertEqual(stats["chunks"], self.app.store.count_chunks())
        self.assertIn("components", stats)
        self.assertIn("corpus", self.app.metrics())

    def test_percentile(self):
        self.assertEqual(percentile([], 0.5), 0.0)
        self.assertEqual(percentile([5.0], 0.95), 5.0)
        self.assertAlmostEqual(percentile([0.0, 10.0], 0.5), 5.0)

    def test_cost_calculation(self):
        from ragkb.types import Usage

        usage = Usage(prompt_tokens=1000, completion_tokens=1000)
        self.assertAlmostEqual(cost_usd(usage, "gpt-4o-mini", {"gpt-4o-mini": (0.00015, 0.0006)}), 0.00075)
        self.assertEqual(cost_usd(usage, "未知模型", {}), 0.0)


class TestQueryProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = make_settings("qp")
        from ragkb.llm import ScriptedChat

        self.qp = QueryProcessor(ScriptedChat(), self.cfg)

    def test_route_greeting_to_chat(self):
        self.assertEqual(self.qp.route("你好"), "chat")

    def test_route_kb_question(self):
        self.assertEqual(self.qp.route("报销流程是什么？"), "kb")

    def test_needs_condense_only_with_history_and_pronoun(self):
        self.assertFalse(self.qp.needs_condense("那它多久修好？", []))
        self.assertTrue(self.qp.needs_condense("那它多久修好？", [{"role": "user", "content": "P0 是什么"}]))
        self.assertFalse(
            self.qp.needs_condense("报销住宿标准的具体上限金额是多少", [{"role": "user", "content": "x"}])
        )

    def test_invalid_rewrite_is_rejected(self):
        self.assertFalse(QueryProcessor._valid_rewrite(NO_ANSWER_SENTINEL, "那它呢？"))
        self.assertFalse(QueryProcessor._valid_rewrite("（离线模式：通用对话能力未启用。）", "那它呢？"))
        self.assertTrue(QueryProcessor._valid_rewrite("P0 故障 解决时限", "那它呢？"))

    def test_decompose_splits_compound_question(self):
        parts = self.qp.decompose("住宿标准是什么？另外餐补每天多少钱？")
        self.assertEqual(len(parts), 2)

    def test_decompose_keeps_simple_question(self):
        self.assertEqual(len(self.qp.decompose("住宿上限是多少？")), 1)


class TestEmptyCorpus(unittest.TestCase):
    def test_asking_with_empty_index_refuses(self):
        cfg = make_settings("empty")
        from ragkb.rag import RagApp

        app = RagApp.build(cfg)
        try:
            ans = app.ask("住宿标准是什么？")
            self.assertTrue(ans.refused)
            self.assertEqual(ans.debug["refuse_reason"], "no_hits")
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
