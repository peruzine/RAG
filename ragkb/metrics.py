"""评测指标：检索质量 + 答案质量 + 拒答/引用正确性。

指标选择的理由
--------------
* **检索侧**用 ``recall@k`` / ``MRR`` / ``nDCG@k`` / ``hit@k``。只看"最终答对没答对"
  无法定位问题；这三个指标能分开回答"召回得到吗""排得靠前吗"。
* **生成侧**用 ``char_f1`` / ``contains`` 做参考比对，用 ``groundedness``（答案内容词
  落在检索上下文中的比例）做**无标注**的忠实度代理，再乘上 ``citation_validity``
  （引用编号是否越界）——幻觉往往先表现为"引用了不存在的来源"。
* **拒答**单独统计 ``refusal_accuracy``：语料里没有的问题敢不敢说"不知道"，
  这是企业知识库最容易被投诉的点（比答错更严重的是编造）。

注意：``groundedness`` 是**代理指标**，不等于事实正确性。生产上应该叠加
LLM-as-judge（本项目留了接口：``judge`` 参数），但在 CI 里跑规则指标更稳定、更便宜。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .text import char_f1, containment, content_tokens, fold
from .types import Chunk, EvalCase, Retrieved

# 需要聚合展示的指标名（顺序即报告顺序）
RETRIEVAL_METRICS = ("hit@k", "recall@k", "precision@k", "mrr", "ndcg@k")
ANSWER_METRICS = (
    "refusal_accuracy",
    "answer_f1",
    "answer_contains",
    "groundedness",
    "citation_validity",
)


# --------------------------------------------------------------------------
# 相关性判定
# --------------------------------------------------------------------------
def _source_matches(chunk: Chunk, gold_sources: Sequence[str]) -> bool:
    src = (chunk.metadata.get("source") or "").replace("\\", "/")
    name = src.rsplit("/", 1)[-1]
    return any(g in (src, name) or src.endswith(g) for g in gold_sources)


def is_relevant(chunk: Chunk, case: EvalCase) -> bool:
    """判断一个 chunk 是否算"命中了 gold"。

    优先级刻意是**从严格到宽松**：chunk_id > gold 片段 > 来源文件。
    如果先按文件判相关，那么"检索到了同一篇文档但完全不相干的那一段"也会算命中，
    precision@k 会虚高——这类指标看起来漂亮，实际掩盖了排序问题。
    """
    if case.gold_chunk_ids and chunk.chunk_id in case.gold_chunk_ids:
        return True
    if case.gold_snippets:
        return any(containment(s, chunk.text) for s in case.gold_snippets)
    if case.gold_sources:
        return _source_matches(chunk, case.gold_sources)
    return False


# --------------------------------------------------------------------------
# 检索指标
# --------------------------------------------------------------------------
def retrieval_metrics(items: Sequence[Retrieved], case: EvalCase, k: int = 5) -> dict[str, float]:
    top = list(items)[:k]
    flags = [1 if is_relevant(it.chunk, case) else 0 for it in top]
    n_rel = sum(flags)

    # hit@k：只要有一条相关就算命中（对"能不能答"最直观）
    hit = 1.0 if n_rel > 0 else 0.0

    # recall@k：gold 片段有多少比例被覆盖（比 hit 更严格）
    if case.gold_snippets:
        covered = 0
        for snippet in case.gold_snippets:
            if any(containment(snippet, it.chunk.text) for it in top):
                covered += 1
        recall = covered / len(case.gold_snippets)
    else:
        recall = hit

    precision = n_rel / len(top) if top else 0.0

    # MRR：第一个相关结果的倒数排名
    mrr = 0.0
    for i, f in enumerate(flags, start=1):
        if f:
            mrr = 1.0 / i
            break

    # nDCG@k：二值相关性
    dcg = sum(f / math.log2(i + 1) for i, f in enumerate(flags, start=1))
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(flags), len(case.gold_snippets) or n_rel) + 1))
    ndcg = dcg / ideal if ideal > 0 else 0.0

    return {
        "hit@k": hit,
        "recall@k": recall,
        "precision@k": precision,
        "mrr": mrr,
        "ndcg@k": ndcg,
        "n_relevant": float(n_rel),
    }


# --------------------------------------------------------------------------
# 答案指标
# --------------------------------------------------------------------------
def groundedness(answer_text: str, items: Sequence[Retrieved], top_n: int = 5) -> float:
    """答案内容词中，出现在检索上下文里的比例（0~1）。

    这是"答案是否有据可依"的下界估计：值低说明模型引入了上下文里没有的内容。
    对抽取式答案是 1.0；对生成式模型，中文写作常见的同义改写会让它偏低，
    所以它是**相对指标**——比较不同版本 prompt / 模型之间的变化，而不是绝对值。
    """
    toks = content_tokens(answer_text)
    if not toks:
        return 1.0
    context = " ".join(it.chunk.text for it in items[:top_n]).lower()
    counts = Counter(toks)
    hit = 0
    total = 0
    for tok, n in counts.items():
        total += n
        if tok in context:
            hit += n
    return hit / total if total else 1.0


@dataclass
class CaseResult:
    case_id: str
    question: str
    tags: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    answer_text: str = ""
    refused: bool = False
    latency_ms: float = 0.0
    n_retrieved: int = 0
    top_chunk: str = ""
    invalid_citations: list[int] = field(default_factory=list)
    has_gold: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "tags": self.tags,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "refused": self.refused,
            "latency_ms": round(self.latency_ms, 1),
            "n_retrieved": self.n_retrieved,
            "top_chunk": self.top_chunk,
            "invalid_citations": self.invalid_citations,
            "has_gold": self.has_gold,
            "answer_preview": self.answer_text[:200],
            "note": self.note,
        }


def evaluate_case(
    case: EvalCase,
    items: Sequence[Retrieved],
    *,
    answer_text: str = "",
    refused: bool = False,
    invalid_citations: Sequence[int] = (),
    latency_ms: float = 0.0,
    k: int = 5,
) -> CaseResult:
    metrics = retrieval_metrics(items, case, k=k)

    # 拒答正确性：应当拒答的答了=错；不该拒答却拒了=也错
    metrics["refusal_accuracy"] = 1.0 if bool(refused) == bool(case.should_refuse) else 0.0

    if case.gold_answer and answer_text:
        metrics["answer_f1"] = char_f1(answer_text, case.gold_answer)
        metrics["answer_contains"] = 1.0 if containment(case.gold_answer, answer_text) else 0.0
    else:
        metrics["answer_f1"] = 1.0 if (case.should_refuse and refused) else 0.0
        metrics["answer_contains"] = metrics["answer_f1"]

    metrics["groundedness"] = (
        1.0 if refused else groundedness(answer_text, items)
    )
    # 引用有效性：越界引用一票否决；未拒答却没有任何引用也算不合格
    metrics["citation_validity"] = (
        1.0
        if (refused or (not invalid_citations and "[" in answer_text))
        else 0.0
    )

    return CaseResult(
        case_id=case.case_id,
        question=case.question,
        tags=list(case.tags),
        metrics=metrics,
        answer_text=answer_text,
        refused=bool(refused),
        latency_ms=latency_ms,
        n_retrieved=len(items),
        top_chunk=(items[0].chunk.heading_path or items[0].chunk.doc_id) if items else "",
        invalid_citations=list(invalid_citations),
        has_gold=bool(case.gold_chunk_ids or case.gold_sources or case.gold_snippets),
        note=case.note,
    )


# --------------------------------------------------------------------------
# 聚合
# --------------------------------------------------------------------------
def _mean(values: Iterable[float]) -> float:
    vals = [v for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def aggregate(results: Sequence[CaseResult]) -> dict[str, float]:
    """整体均值。

    检索类指标只在**有 gold 标注**的样本上平均：拒答样本（语料里本来就没有）
    没有可召回的目标，把它们算进 recall 只会稀释指标、掩盖真实问题。
    答案与拒答类指标对所有样本平均。
    """
    out: dict[str, float] = {"n_cases": float(len(results))}
    with_gold = [r for r in results if r.has_gold]
    out["n_gold_cases"] = float(len(with_gold))
    for key in RETRIEVAL_METRICS:
        vals = [r.metrics[key] for r in with_gold if key in r.metrics]
        if vals:
            out[key] = _mean(vals)
    for key in ANSWER_METRICS:
        vals = [r.metrics[key] for r in results if key in r.metrics]
        if vals:
            out[key] = _mean(vals)
    out["latency_ms_avg"] = _mean([r.latency_ms for r in results])
    out["refusal_rate"] = _mean([1.0 if r.refused else 0.0 for r in results])
    return out


def aggregate_by_tag(results: Sequence[CaseResult]) -> dict[str, dict[str, float]]:
    tags: dict[str, list[CaseResult]] = {}
    for r in results:
        for t in r.tags or ["untagged"]:
            tags.setdefault(t, []).append(r)
    return {tag: aggregate(rs) for tag, rs in sorted(tags.items())}


def format_table(report: dict[str, Any]) -> str:
    """把报告渲染成终端可读的对齐表格（用于 CI 日志/评审截图）。"""
    keys = list(RETRIEVAL_METRICS) + list(ANSWER_METRICS) + ["latency_ms_avg", "refusal_rate"]
    lines = [
        f"{'指标':<20}{'本版':>10}{'基线':>10}{'阈值':>10}   结果",
        "-" * 62,
    ]
    overall = report.get("overall", {})
    baseline = report.get("baseline", {}) or {}
    thresholds = report.get("thresholds", {}) or {}
    gate = report.get("gate", {})
    failed = set(gate.get("failed_keys") or [])
    for key in keys:
        if key not in overall:
            continue
        cur = overall[key]
        base = baseline.get(key)
        thr = thresholds.get(key)
        base_s = f"{base:.3f}" if isinstance(base, (int, float)) else "-"
        thr_s = f"{thr:.3f}" if isinstance(thr, (int, float)) else "-"
        status = "FAIL" if key in failed else "pass"
        lines.append(f"{key:<20}{cur:>10.3f}{base_s:>10}{thr_s:>10}   {status}")
    return "\n".join(lines)


def fold_text(text: str) -> str:
    """小工具（供外部复用）。"""
    return fold(text)
