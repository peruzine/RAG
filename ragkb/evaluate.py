"""评测执行器 + CI 门禁。

用法::

    python -m ragkb.cli eval                          # 跑全量，与基线对比，不达标退出码 1
    python -m ragkb.cli eval --retrieval-only         # 只测检索（不调用生成模型，快且免费）
    python -m ragkb.cli eval --update-baseline        # 确认变好后更新基线
    python -m ragkb.cli eval --tags acl,refusal       # 只跑某几类

为什么要有"门禁"
----------------
RAG 项目的改动（改 prompt、换 chunk_size、换 embedding、调 top_k）大多是**隐性的**：
体感上"好像更好了"，但某些 case 变差。没有基线对比，这类回归会一路带到线上。
这里用两个约束卡住：① 绝对阈值（``eval/thresholds.json``）；
② 相对基线最大跌幅（``--max-drop``，默认 0.03）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import Settings
from .metrics import (
    CaseResult,
    aggregate,
    aggregate_by_tag,
    evaluate_case,
    format_table,
)
from .rag import RagApp
from .types import EvalCase


def load_dataset(path: str | Path) -> list[EvalCase]:
    """读取 JSONL 评测集。一行一条，字段与 :class:`EvalCase` 对齐。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"评测集不存在：{p}")
    cases: list[EvalCase] = []
    with open(p, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno} JSON 解析失败：{e}") from e
            roles = raw.get("roles", ())
            cases.append(
                EvalCase(
                    case_id=str(raw.get("id") or f"case-{lineno:03d}"),
                    question=raw["question"],
                    gold_sources=list(raw.get("gold_sources", [])),
                    gold_doc_ids=list(raw.get("gold_doc_ids", [])),
                    gold_chunk_ids=list(raw.get("gold_chunk_ids", [])),
                    gold_snippets=list(raw.get("gold_snippets", [])),
                    gold_answer=raw.get("gold_answer", ""),
                    should_refuse=bool(raw.get("should_refuse", False)),
                    tags=list(raw.get("tags", [])),
                    roles=None if roles is None else tuple(roles),
                    history=list(raw.get("history", [])),
                    note=raw.get("note", ""),
                )
            )
    return cases


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


@dataclass
class EvalReport:
    overall: dict[str, float]
    by_tag: dict[str, dict[str, float]]
    cases: list[CaseResult]
    baseline: dict[str, float]
    thresholds: dict[str, float]
    gate: dict[str, Any]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": {k: round(v, 4) for k, v in self.overall.items()},
            "by_tag": {
                t: {k: round(v, 4) for k, v in m.items()} for t, m in self.by_tag.items()
            },
            "cases": [c.to_dict() for c in self.cases],
            "baseline": self.baseline,
            "thresholds": self.thresholds,
            "gate": self.gate,
            "meta": self.meta,
        }

    def render(self) -> str:
        lines = [format_table(self.to_dict())]
        if self.by_tag:
            lines.append("\n分标签（看哪类问题在拖后腿）：")
            keys = ["hit@k", "recall@k", "mrr", "refusal_accuracy", "citation_validity"]
            head = f"{'tag':<18}" + "".join(f"{k:>18}" for k in keys)
            lines.append(head)
            lines.append("-" * len(head))
            for tag, m in self.by_tag.items():
                lines.append(f"{tag:<18}" + "".join(f"{m.get(k, 0):>18.3f}" for k in keys))
        if self.gate.get("failed"):
            lines.append("\n❌ 门禁未通过：")
            for item in self.gate["failed"]:
                lines.append(f"   - {item}")
        else:
            lines.append("\n✅ 门禁通过")
        return "\n".join(lines)


def run_eval(
    app: RagApp,
    cases: Sequence[EvalCase],
    cfg: Settings,
    *,
    k: int = 5,
    retrieval_only: bool = False,
    baseline_path: Path | None = None,
    thresholds_path: Path | None = None,
    max_drop: float = 0.03,
) -> EvalReport:
    results: list[CaseResult] = []
    for case in cases:
        t0 = time.perf_counter()
        if retrieval_only:
            res = app.retrieve(case.question, top_k=k, roles=case.roles)
            results.append(
                evaluate_case(case, res.items, refused=False, k=k, latency_ms=(time.perf_counter() - t0) * 1000)
            )
            continue
        ans = app.ask(
            case.question, roles=case.roles, history=case.history or None, top_k=k
        )
        results.append(
            evaluate_case(
                case,
                ans.retrieved,
                answer_text=ans.text,
                refused=ans.refused,
                invalid_citations=ans.debug.get("invalid_citations", []),
                latency_ms=ans.latency_ms,
                k=k,
            )
        )

    overall = aggregate(results)
    by_tag = aggregate_by_tag(results)
    baseline = load_json(baseline_path, {}) if baseline_path else {}
    baseline = baseline.get("overall", baseline) if isinstance(baseline, dict) else {}
    thresholds = load_json(thresholds_path, {}) if thresholds_path else {}
    thresholds = thresholds.get("thresholds", thresholds) if isinstance(thresholds, dict) else {}
    file_max_drop = 0.0
    if thresholds_path:
        raw_cfg = load_json(thresholds_path, {}) or {}
        file_max_drop = float(raw_cfg.get("max_drop") or 0.0)
    effective_drop = max_drop if max_drop else file_max_drop

    failed: list[str] = []
    failed_keys: list[str] = []
    for key, floor in (thresholds or {}).items():
        if key in ("max_drop", "notes", "description"):
            continue
        cur = overall.get(key)
        if cur is None:
            continue
        if cur < float(floor):
            failed.append(f"{key}={cur:.3f} < 阈值 {float(floor):.3f}")
            failed_keys.append(key)
    if effective_drop > 0:
        for key, base_val in (baseline or {}).items():
            if key in ("n_cases", "latency_ms_avg", "refusal_rate"):
                continue
            cur = overall.get(key)
            if cur is None or not isinstance(base_val, (int, float)):
                continue
            if cur < float(base_val) - effective_drop:
                failed.append(
                    f"{key}={cur:.3f} 比基线 {float(base_val):.3f} 下跌超过 {effective_drop}"
                )
                failed_keys.append(key)

    gate = {
        "failed": failed,
        # 单独记下"哪个指标没过"：渲染表格要按指标名标状态，
        # 拿失败消息文本去匹配指标名永远匹配不上（实测踩过：整表都显示 pass）。
        "failed_keys": sorted(set(failed_keys)),
        "passed": not failed,
        "max_drop": effective_drop,
    }
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_cases": len(cases),
        "k": k,
        "retrieval_only": retrieval_only,
        "config": cfg.describe(),
        "components": {
            "embedder": app.embedder.name,
            "llm": app.llm.name,
            "reranker": app.reranker.name,
        },
    }
    return EvalReport(overall, by_tag, results, baseline, thresholds, gate, meta)


def save_report(report: EvalReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def update_baseline(report: EvalReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "overall": {k: round(v, 4) for k, v in report.overall.items()},
        "meta": report.meta,
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def sanity_check_dataset(cases: Sequence[EvalCase], app: RagApp) -> list[str]:
    """评测集自检：gold 标注是否真的能在库里找到。

    这一步能挡掉最常见的一类"假失败"——改过文档之后 gold 片段过期了，
    指标突然掉 20 个点，其实是评测集本身错了。
    """
    problems: list[str] = []
    docs = {d["source"]: d for d in app.store.list_documents()}
    names = {k.replace("\\", "/").rsplit("/", 1)[-1] for k in docs}
    for case in cases:
        for src in case.gold_sources:
            if src not in names and not any(n == src for n in names):
                problems.append(f"{case.case_id}: gold_source 不存在于库中 → {src}")
        if case.gold_snippets:
            res = app.retrieve(case.question, top_k=20, roles=case.roles)
            joined = "\n".join(it.chunk.text for it in res.items)
            for snip in case.gold_snippets:
                from .text import containment

                if not containment(snip, joined):
                    problems.append(
                        f"{case.case_id}: gold_snippet 在 top-20 里都找不到（标注可能过期）→ {snip[:40]}"
                    )
    return problems
