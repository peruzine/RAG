"""可观测性：结构化 trace + 成本统计 + 线上指标聚合。

RAG 线上出问题时，"答得不对"至少有四种原因：检索没召回、召回但排序靠后、
上下文被截断、模型没按上下文作答。**没有一个 trace 就只能靠猜**。所以每次问答
都落一条 JSONL，包含每个阶段的耗时、每路召回命中数、token 用量与折算成本。

对外只依赖标准库文件 IO：不装 Langfuse 也能用；接了 Langfuse/LangSmith 时，
只要在 ``Trace.finish`` 里加一个导出分支即可（``RAGKB_TRACE_EXPORT``）。
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .config import Settings
from .types import Usage


# --------------------------------------------------------------------------
# 成本
# --------------------------------------------------------------------------
def cost_usd(usage: Usage, model: str, price_table: dict[str, tuple[float, float]]) -> float:
    """按 USD/1K tokens 折算出本次调用成本。"""
    inp, out = (0.0, 0.0)
    if model in price_table:
        inp, out = price_table[model]
    else:
        for key, price in price_table.items():
            if model.startswith(key):
                inp, out = price
                break
    return usage.prompt_tokens / 1000.0 * inp + usage.completion_tokens / 1000.0 * out


# --------------------------------------------------------------------------
# Trace
# --------------------------------------------------------------------------
@dataclass
class Span:
    name: str
    duration_ms: float = 0.0
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ms": round(self.duration_ms, 3), **self.attrs}


class Trace:
    """一次问答的追踪上下文。``with trace.span("retrieve") as attrs: ...``"""

    def __init__(self, name: str, tracer: "Tracer") -> None:
        self.trace_id = uuid.uuid4().hex[:16]
        self.name = name
        self.tracer = tracer
        self.started = time.perf_counter()
        self.spans: list[Span] = []
        self.attrs: dict[str, Any] = {}
        self.usage = Usage()
        self.cost = 0.0
        self.model = ""

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        t0 = time.perf_counter()
        holder: dict[str, Any] = dict(attrs)
        try:
            yield holder
        finally:
            self.spans.append(
                Span(name=name, duration_ms=(time.perf_counter() - t0) * 1000.0, attrs=holder)
            )

    def add_usage(self, usage: Usage, model: str) -> float:
        self.usage.prompt_tokens += usage.prompt_tokens
        self.usage.completion_tokens += usage.completion_tokens
        self.model = model or self.model
        self.cost += cost_usd(usage, model, self.tracer.price_table)
        return self.cost

    def finish(self, **extra: Any) -> dict[str, Any]:
        record = {
            "trace_id": self.trace_id,
            "name": self.name,
            "ts": time.time(),
            "latency_ms": round((time.perf_counter() - self.started) * 1000.0, 3),
            "spans": [s.to_dict() for s in self.spans],
            "usage": self.usage.to_dict(),
            "model": self.model,
            "cost_usd": round(self.cost, 8),
            **self.attrs,
            **extra,
        }
        self.tracer.write(record)
        return record


class Tracer:
    """写入 JSONL 的极简 tracer（线程安全）。"""

    def __init__(
        self,
        enabled: bool = True,
        path: str | Path | None = None,
        price_table: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.enabled = enabled
        self.path = Path(path) if path else None
        self.price_table = price_table or {}
        self._lock = threading.Lock()
        if self.enabled and self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def new_trace(self, name: str) -> Trace:
        return Trace(name=name, tracer=self)

    def write(self, record: dict[str, Any]) -> None:
        if not self.enabled or not self.path:
            return
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def build_tracer(cfg: Settings) -> Tracer:
    return Tracer(
        enabled=cfg.trace_enabled, path=cfg.trace_path, price_table=cfg.price_table
    )


# --------------------------------------------------------------------------
# 指标聚合
# --------------------------------------------------------------------------
def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def read_jsonl(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:] if limit else rows


def summarize_traces(path: str | Path, limit: int = 2000) -> dict[str, Any]:
    """把 trace 汇总成线上看板需要的那几个数。"""
    rows = [r for r in read_jsonl(path, limit) if r.get("name") in ("ask", "ask_stream", None)]
    if not rows:
        return {"count": 0}
    lat = [float(r.get("latency_ms") or 0.0) for r in rows]
    cost = [float(r.get("cost_usd") or 0.0) for r in rows]
    refused = sum(1 for r in rows if r.get("refused"))
    stage: dict[str, list[float]] = {}
    for r in rows:
        for span in r.get("spans") or []:
            stage.setdefault(span["name"], []).append(float(span.get("ms") or 0.0))
    return {
        "count": len(rows),
        "latency_ms": {
            "p50": round(percentile(lat, 0.5), 1),
            "p95": round(percentile(lat, 0.95), 1),
            "max": round(max(lat), 1),
        },
        "stage_ms_p95": {k: round(percentile(v, 0.95), 1) for k, v in sorted(stage.items())},
        "tokens": {
            "avg_prompt": round(
                sum(float((r.get("usage") or {}).get("prompt_tokens") or 0) for r in rows) / len(rows), 1
            ),
            "avg_completion": round(
                sum(float((r.get("usage") or {}).get("completion_tokens") or 0) for r in rows)
                / len(rows),
                1,
            ),
        },
        "cost_usd": {
            "total": round(sum(cost), 6),
            "avg_per_query": round(sum(cost) / len(rows), 8),
        },
        "refusal_rate": round(refused / len(rows), 4),
        "models": sorted({r.get("model") or "unknown" for r in rows}),
    }


def append_feedback(path: str | Path, record: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.time(), **record}, ensure_ascii=False) + "\n")


def summarize_feedback(path: str | Path, limit: int = 5000) -> dict[str, Any]:
    rows = read_jsonl(path, limit)
    if not rows:
        return {"count": 0}
    ratings = [float(r.get("rating") or 0) for r in rows if r.get("rating") is not None]
    return {
        "count": len(rows),
        "avg_rating": round(sum(ratings) / len(ratings), 3) if ratings else None,
        "good": sum(1 for r in rows if r.get("rating") == 1),
        "bad": sum(1 for r in rows if r.get("rating") == -1),
        "recent_comments": [r.get("comment") for r in rows[-5:] if r.get("comment")],
    }


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}
