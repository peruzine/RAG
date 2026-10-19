"""混合检索 + RRF 融合 + 重排。

流程
----
::

    查询 ─┬─► BM25 词法召回 ─┐
          └─► 向量语义召回 ─┴─► RRF 融合 ─► 取回 chunk ─► rerank ─► top_k

为什么用 RRF（Reciprocal Rank Fusion）而不是加权求和
----------------------------------------------------
BM25 分数与余弦相似度量纲完全不同（前者无上界、后者 0~1），直接加权需要归一化，
而归一化区间随查询变化，非常脆弱。RRF 只用**排名**：``score = Σ 1/(k + rank)``，
对量纲免疫、无需调参、多路召回天然可扩展——所以"子问题拆解"和"以后加 SQL 检索"
都能直接并进来，不用重新调权重。

排查能力
--------
每个结果都记录 ``channels``（被哪些通道召回、各排第几）与 ``channel_scores``。
只有 BM25 召回 → 多半是术语/编号/数字类问题；只有向量召回 → 多半是同义改写提问。
评测时应分别统计这两类，而不是只看一个总分。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import Settings
from .embedding import Embedder
from .rerank import Reranker
from .store import SqliteStore
from .types import Retrieved

CH_LEXICAL = "bm25"
CH_VECTOR = "vector"


@dataclass(slots=True)
class RetrievalDebug:
    query: str
    lexical_hits: int = 0
    vector_hits: int = 0
    fused: int = 0
    returned: int = 0
    best_score: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)
    reranker: str = ""
    reranker_error: str | None = None
    channels_only: dict[str, int] = field(default_factory=dict)
    sub_queries: list[str] = field(default_factory=list)
    coverage: float = 0.0
    coverage_detail: dict[str, float] = field(default_factory=dict)
    best_vector_similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "sub_queries": self.sub_queries,
            "lexical_hits": self.lexical_hits,
            "vector_hits": self.vector_hits,
            "fused": self.fused,
            "returned": self.returned,
            "best_score": round(self.best_score, 6),
            "coverage": self.coverage,
            "coverage_detail": self.coverage_detail,
            "best_vector_similarity": self.best_vector_similarity,
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "reranker": self.reranker,
            "reranker_error": self.reranker_error,
            "channels_only": self.channels_only,
        }


@dataclass(slots=True)
class RetrievalResult:
    items: list[Retrieved]
    debug: RetrievalDebug

    @property
    def best_score(self) -> float:
        if not self.items:
            return 0.0
        top = self.items[0]
        return top.rerank_score if top.rerank_score is not None else top.score

    @property
    def chunk_ids(self) -> list[str]:
        return [it.chunk_id for it in self.items]


class HybridRetriever:
    """检索门面：把 store / embedder / reranker 组合成一条可解释的链路。"""

    def __init__(
        self,
        store: SqliteStore,
        embedder: Embedder,
        reranker: Reranker,
        cfg: Settings,
        *,
        lexical_weight: float = 1.0,
        vector_weight: float = 1.0,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.cfg = cfg
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight

    # ------------------------------------------------------------------
    def _embed_query(self, query: str) -> list[float] | None:
        try:
            return self.embedder.embed_query(query)
        except Exception:
            # 向量通道挂了也要能用 BM25 兜住：可用性优先
            return None

    def fuse_rrf(
        self,
        channels: dict[str, list[tuple[str, float]]],
        top_k: int,
        k: int = 60,
    ) -> tuple[list[tuple[str, float]], dict[str, dict[str, int]]]:
        """RRF 融合。返回 ``[(chunk_id, fused_score)]`` 与各通道排名表。

        通道名形如 ``bm25`` / ``bm25#1``（子查询），权重按前缀取。
        每个通道只取前 ``fuse_top_k`` 名参与融合：排名靠后的候选贡献的
        ``1/(k+rank)`` 与第一名同量级，不截断的话长尾噪声会把真正相关的挤下去。
        """
        scores: dict[str, float] = {}
        ranks: dict[str, dict[str, int]] = {}
        weights = {CH_LEXICAL: self.lexical_weight, CH_VECTOR: self.vector_weight}
        for name, hits in channels.items():
            w = weights.get(name.split("#")[0], 1.0)
            for rank, (cid, _score) in enumerate(hits[: max(1, self.cfg.fuse_top_k)], start=1):
                scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
                ranks.setdefault(cid, {})[name] = rank
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ordered[:top_k], ranks

    # ------------------------------------------------------------------
    def _channels_for(self, query: str) -> dict[str, list[tuple[str, float]]]:
        lexical = self.store.lexical_search(query, self.cfg.retrieve_candidates)
        qvec = self._embed_query(query)
        vector: list[tuple[str, float]] = []
        if qvec is not None:
            try:
                vector = self.store.vector_search(
                    qvec,
                    self.cfg.retrieve_candidates,
                    model=self.embedder.name,
                    min_similarity=self.cfg.vector_min_similarity,
                )
            except Exception:
                vector = []
        return {CH_LEXICAL: lexical, CH_VECTOR: vector}

    @staticmethod
    def _raw_scores(channels: dict[str, list[tuple[str, float]]]) -> dict[str, dict[str, float]]:
        """把各通道的原始分数按 chunk 汇总，供排查与语义拒答门禁使用。"""
        out: dict[str, dict[str, float]] = {}
        for name, hits in channels.items():
            for cid, score in hits:
                out.setdefault(cid, {})[name] = score
        return out

    def _assemble(
        self,
        fused: list[tuple[str, float]],
        ranks: dict[str, dict[str, int]],
        rerank_query: str,
        debug: RetrievalDebug,
        top_k: int,
        use_rerank: bool,
        raw_scores: dict[str, dict[str, float]] | None = None,
    ) -> RetrievalResult:
        debug.fused = len(fused)
        if not fused:
            debug.returned = 0
            return RetrievalResult(items=[], debug=debug)

        only = {"lexical_only": 0, "vector_only": 0, "both": 0}
        for cid, _ in fused:
            r = {k.split("#")[0] for k in ranks.get(cid, {})}
            if len(r) == 2:
                only["both"] += 1
            elif CH_LEXICAL in r:
                only["lexical_only"] += 1
            else:
                only["vector_only"] += 1
        debug.channels_only = only

        t0 = time.perf_counter()
        chunks = {c.chunk_id: c for c in self.store.get_chunks([cid for cid, _ in fused])}
        items: list[Retrieved] = []
        for cid, score in fused:
            ch = chunks.get(cid)
            if ch is None:
                continue
            items.append(
                Retrieved(
                    chunk=ch,
                    score=score,
                    channels=ranks.get(cid, {}),
                    # 保留每个通道的**原始分数**（不只是排名）：排查"谁召回的、分多高"
                    # 要靠它，拒答门禁的语义信号也从这里取。
                    channel_scores=dict(raw_scores.get(cid, {})) if raw_scores else {},
                )
            )
        debug.timings_ms["fetch"] = (time.perf_counter() - t0) * 1000

        if use_rerank and items:
            t0 = time.perf_counter()
            items = self.reranker.rerank(rerank_query, items, len(items))
            debug.timings_ms["rerank"] = (time.perf_counter() - t0) * 1000
            debug.reranker = self.reranker.name
            debug.reranker_error = getattr(self.reranker, "last_error", None)

        items = items[:top_k]
        debug.returned = len(items)
        debug.best_score = (
            (items[0].rerank_score if items[0].rerank_score is not None else items[0].score)
            if items
            else 0.0
        )
        # 原始向量相似度在这里就算好：排查与拒答门禁都要用，
        # 不能只依赖"门禁恰好被调用过"（否则纯 retieve 调试时永远是 0）。
        debug.best_vector_similarity = round(
            max((it.channel_scores.get("vector", 0.0) for it in items), default=0.0), 4
        )
        return RetrievalResult(items=items, debug=debug)

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        roles: Sequence[str] | None = (),
        doc_ids: Sequence[str] | None = None,
        use_rerank: bool = True,
    ) -> RetrievalResult:
        """单查询混合检索。

        ``roles=()`` 表示走 ACL 过滤（默认只有 public 可见）；``roles=None`` 不过滤。
        """
        t_all = time.perf_counter()
        top_k = top_k or self.cfg.final_top_k
        debug = RetrievalDebug(query=query)

        t0 = time.perf_counter()
        lexical = self.store.lexical_search(
            query, self.cfg.retrieve_candidates, roles=roles, doc_ids=doc_ids
        )
        debug.timings_ms["lexical"] = (time.perf_counter() - t0) * 1000
        debug.lexical_hits = len(lexical)

        t0 = time.perf_counter()
        qvec = self._embed_query(query)
        vector: list[tuple[str, float]] = []
        if qvec is not None:
            try:
                vector = self.store.vector_search(
                    qvec,
                    self.cfg.retrieve_candidates,
                    roles=roles,
                    doc_ids=doc_ids,
                    model=self.embedder.name,
                    min_similarity=self.cfg.vector_min_similarity,
                )
            except Exception as e:  # 维度不匹配等：降级为纯词法，别让整个请求挂掉
                debug.reranker_error = f"vector: {type(e).__name__}: {e}"
        debug.timings_ms["vector"] = (time.perf_counter() - t0) * 1000
        debug.vector_hits = len(vector)

        t0 = time.perf_counter()
        channels = {CH_LEXICAL: lexical, CH_VECTOR: vector}
        fused, ranks = self.fuse_rrf(channels, self.cfg.fuse_top_k, k=self.cfg.rrf_k)
        debug.timings_ms["fuse"] = (time.perf_counter() - t0) * 1000

        result = self._assemble(
            fused, ranks, query, debug, top_k, use_rerank, self._raw_scores(channels)
        )
        debug.timings_ms["total"] = (time.perf_counter() - t_all) * 1000
        return result

    def retrieve_multi(
        self,
        queries: Sequence[str],
        *,
        top_k: int | None = None,
        roles: Sequence[str] | None = (),
        doc_ids: Sequence[str] | None = None,
        use_rerank: bool = True,
    ) -> RetrievalResult:
        """多子查询检索：各路结果统一进 RRF，再用原始问题做一次重排。

        子问题拆解能显著改善"一句话里问了两件事"的召回，代价是 N 倍检索开销
        （本地检索很便宜，所以只在检索侧做，不额外调模型）。
        """
        t_all = time.perf_counter()
        top_k = top_k or self.cfg.final_top_k
        cleaned = [q.strip() for q in queries if q and q.strip()] or [""]
        debug = RetrievalDebug(query=" | ".join(cleaned), sub_queries=cleaned)

        channels: dict[str, list[tuple[str, float]]] = {}
        for i, q in enumerate(cleaned):
            t0 = time.perf_counter()
            lexical = self.store.lexical_search(
                q, self.cfg.retrieve_candidates, roles=roles, doc_ids=doc_ids
            )
            debug.timings_ms[f"lexical#{i}"] = (time.perf_counter() - t0) * 1000
            debug.lexical_hits += len(lexical)
            channels[f"{CH_LEXICAL}#{i}"] = lexical

            t0 = time.perf_counter()
            qvec = self._embed_query(q)
            vector: list[tuple[str, float]] = []
            if qvec is not None:
                try:
                    vector = self.store.vector_search(
                        qvec,
                        self.cfg.retrieve_candidates,
                        roles=roles,
                        doc_ids=doc_ids,
                        model=self.embedder.name,
                        min_similarity=self.cfg.vector_min_similarity,
                    )
                except Exception:
                    vector = []
            debug.timings_ms[f"vector#{i}"] = (time.perf_counter() - t0) * 1000
            debug.vector_hits += len(vector)
            channels[f"{CH_VECTOR}#{i}"] = vector

        t0 = time.perf_counter()
        fused, ranks = self.fuse_rrf(channels, self.cfg.fuse_top_k, k=self.cfg.rrf_k)
        debug.timings_ms["fuse"] = (time.perf_counter() - t0) * 1000

        result = self._assemble(
            fused, ranks, " ".join(cleaned), debug, top_k, use_rerank, self._raw_scores(channels)
        )
        debug.timings_ms["total"] = (time.perf_counter() - t_all) * 1000
        return result
