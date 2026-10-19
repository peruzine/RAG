"""重排序（rerank）。

第一阶段（召回）追求**高召回**，用便宜且快的 BM25/向量；第二阶段（rerank）
追求**高精度**，用更贵的交叉打分把真正相关的顶上来。这是 RAG 检索质量提升
性价比最高的一个环节。

实现：
* ``LexicalReranker`` —— 本地词法交叉打分：查询词覆盖率（按词长加权）+ 邻近距离
  + 标题/章节命中加成。零依赖、可解释、每个分数都能归因。
* ``HttpReranker`` —— Cohere / Jina 风格的 ``POST /rerank``，失败时**自动降级**
  为词法重排（线上"宁可用差点儿的排序，也不能整个请求失败"）。

重排只改变顺序，不丢结果：未参与重排的条目按原顺序接到末尾，保证上层
``top_k`` 截断的行为稳定、可复现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from .config import Settings
from .http import auth_headers, post_json
from .text import normalize, query_terms
from .types import Retrieved


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class Reranker(ABC):
    name = "noop"

    @abstractmethod
    def rerank(self, query: str, items: list[Retrieved], top_n: int) -> list[Retrieved]:
        ...

    def describe(self) -> dict[str, Any]:
        return {"provider": type(self).__name__, "name": self.name}


class NoopReranker(Reranker):
    """关闭重排：直接按融合分排序。"""

    name = "none"

    def rerank(self, query: str, items: list[Retrieved], top_n: int) -> list[Retrieved]:
        ordered = sorted(items, key=lambda r: -r.score)
        return ordered[:top_n] if top_n else ordered


class LexicalReranker(Reranker):
    """本地词法交叉打分。"""

    name = "lexical"

    def __init__(self, weight: float = 0.65, window: int = 120) -> None:
        self.weight = min(1.0, max(0.0, weight))
        self.window = window

    @staticmethod
    def _term_weight(term: str) -> float:
        return 0.35 if len(term) == 1 else (1.0 if len(term) == 2 else 1.2)

    def _score_one(self, terms: Sequence[str], text: str, heading: str) -> float:
        low = text.lower()
        heading_low = heading.lower()
        total = sum(self._term_weight(t) for t in terms) or 1.0

        # 1) 覆盖率：命中查询词的加权占比
        hit_weight = sum(self._term_weight(t) for t in terms if t in low)
        coverage = hit_weight / total

        # 2) 邻近距离：所有命中词若挤在一个窗口里，说明是"同一句话在讲这件事"
        positions = [low.find(t) for t in terms if t in low]
        proximity = 0.0
        if len(positions) >= 2:
            span = max(positions) - min(positions)
            proximity = max(0.0, 1.0 - span / (self.window * 2))
        elif positions:
            proximity = 0.35

        # 3) 标题/章节命中：层级信息命中通常意味着整块都在讲这个主题
        heading_hits = sum(1 for t in terms if t in heading_low)
        heading_bonus = min(1.0, heading_hits / max(1, len(terms)) * 2.0)

        return 0.62 * coverage + 0.26 * proximity + 0.12 * heading_bonus

    def rerank(self, query: str, items: list[Retrieved], top_n: int) -> list[Retrieved]:
        if not items:
            return []
        terms = query_terms(query)
        if not terms:
            return NoopReranker().rerank(query, items, top_n)

        lexical = [
            self._score_one(terms, it.chunk.text, it.chunk.heading_path or "") for it in items
        ]
        fused_norm = _minmax([it.score for it in items])

        for it, lex, fus in zip(items, lexical, fused_norm):
            it.rerank_score = self.weight * lex + (1 - self.weight) * fus
            it.channel_scores["lexical_rerank"] = round(lex, 6)
            it.channel_scores["fused_norm"] = round(fus, 6)

        ordered = sorted(items, key=lambda r: (-(r.rerank_score or 0.0), r.chunk_id))
        return ordered[:top_n] if top_n else ordered


class HttpReranker(Reranker):
    """Cohere / Jina 风格 ``/rerank``，失败自动降级到词法重排。"""

    name = "http"

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str = "",
        timeout: float = 30.0,
        fallback: Reranker | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.fallback = fallback or LexicalReranker()
        self.last_error: str | None = None

    def rerank(self, query: str, items: list[Retrieved], top_n: int) -> list[Retrieved]:
        if not items:
            return []
        ordered = sorted(items, key=lambda r: -r.score)[:top_n] if top_n else sorted(
            items, key=lambda r: -r.score
        )
        docs = [
            it.chunk.embedding_text(it.chunk.metadata.get("title", ""))[:2000] for it in ordered
        ]
        try:
            data = post_json(
                f"{self.base_url}/rerank",
                {"model": self.model, "query": query, "documents": docs, "top_n": len(docs)},
                headers=auth_headers(self.api_key),
                timeout=self.timeout,
                retries=1,
            )
            results = data.get("results") or data.get("data") or []
            if not results:
                raise RuntimeError(f"rerank 返回为空: {str(data)[:200]}")
            out: list[Retrieved] = []
            for rank, row in enumerate(results):
                idx = int(row.get("index", rank))
                if not (0 <= idx < len(ordered)):
                    continue
                it = ordered[idx]
                it.rerank_score = float(
                    row.get("relevance_score", row.get("score", 1.0 / (rank + 1)))
                )
                out.append(it)
            seen = {id(x) for x in out}
            out.extend(x for x in ordered if id(x) not in seen)
            self.last_error = None
            return out
        except Exception as e:  # 降级而不是让整个请求 500
            self.last_error = f"{type(e).__name__}: {e}"
            return self.fallback.rerank(query, items, top_n)


def build_reranker(cfg: Settings) -> Reranker:
    provider = (cfg.rerank_provider or "lexical").lower()
    if provider in ("none", "off", "noop", ""):
        return NoopReranker()
    if provider in ("lexical", "local"):
        return LexicalReranker(weight=cfg.lexical_rerank_weight)
    if provider in ("http", "cohere", "jina", "api"):
        if not cfg.rerank_base_url:
            raise RuntimeError("rerank_provider=http 但未设置 RAGKB_RERANK_BASE_URL")
        return HttpReranker(
            cfg.rerank_model or "rerank",
            base_url=cfg.rerank_base_url,
            api_key=cfg.rerank_api_key,
            fallback=LexicalReranker(weight=cfg.lexical_rerank_weight),
        )
    raise ValueError(f"未知 rerank_provider: {cfg.rerank_provider}")


def normalize_query(text: str) -> str:
    """对外暴露的小工具：重排前的查询清洗。"""
    return normalize(text)
