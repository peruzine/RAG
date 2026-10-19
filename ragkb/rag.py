"""端到端装配：把 store / embedder / reranker / LLM / 检索 / 生成 组成一个对象。

CLI、HTTP 服务、评测脚本、单元测试**都只依赖这里**，保证"生产跑的"和
"测试测的"是同一条链路——这是 RAG 项目最容易走样的地方（demo 一套、线上另一套）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Sequence

from .answer import Answerer
from .config import Settings
from .embedding import Embedder, build_embedder
from .ingest import IngestSummary, Ingestor
from .llm import ChatModel, build_chat_model
from .obs import Tracer, build_tracer, summarize_feedback, summarize_traces
from .query import QueryProcessor
from .rerank import Reranker, build_reranker
from .retrieve import HybridRetriever, RetrievalResult
from .store import SqliteStore
from .types import Answer


class RagApp:
    """一个有状态的 RAG 应用实例（长驻进程复用它，别每次请求都重建）。"""

    def __init__(
        self,
        cfg: Settings,
        *,
        store: SqliteStore | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        llm: ChatModel | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.cfg = cfg
        self.store = store or SqliteStore(cfg.db_path)
        self.embedder = embedder or build_embedder(cfg)
        self.reranker = reranker or build_reranker(cfg)
        self.llm = llm or build_chat_model(cfg)
        self.tracer = tracer or build_tracer(cfg)
        self.retriever = HybridRetriever(self.store, self.embedder, self.reranker, cfg)
        self.query_processor = QueryProcessor(self.llm, cfg, store=self.store)
        self.answerer = Answerer(
            self.retriever, self.llm, cfg, self.tracer, self.query_processor
        )
        self.ingestor = Ingestor(self.store, self.embedder, cfg)

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, cfg: Settings | None = None, **overrides: Any) -> "RagApp":
        return cls(cfg or Settings.load(), **overrides)

    # ------------------------------------------------------------------
    def ingest(
        self,
        path: str | Path,
        *,
        acl: Sequence[str] = (),
        force: bool = False,
        recursive: bool = True,
    ) -> IngestSummary:
        return self.ingestor.ingest_path(path, acl=acl, force=force, recursive=recursive)

    def ask(
        self,
        question: str,
        *,
        roles: Sequence[str] | None = (),
        doc_ids: Sequence[str] | None = None,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        use_rerank: bool = True,
        route: str | None = None,
    ) -> Answer:
        return self.answerer.answer(
            question,
            roles=roles,
            doc_ids=doc_ids,
            history=history,
            top_k=top_k,
            use_rerank=use_rerank,
            route=route,
        )

    def stream(
        self,
        question: str,
        *,
        roles: Sequence[str] | None = (),
        doc_ids: Sequence[str] | None = None,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        use_rerank: bool = True,
    ) -> Iterator[dict[str, Any]]:
        return self.answerer.stream(
            question,
            roles=roles,
            doc_ids=doc_ids,
            history=history,
            top_k=top_k,
            use_rerank=use_rerank,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        roles: Sequence[str] | None = (),
        **kw: Any,
    ) -> RetrievalResult:
        return self.retriever.retrieve(query, top_k=top_k, roles=roles, **kw)

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        data = self.store.stats()
        data["config"] = self.cfg.describe()
        data["components"] = {
            "embedder": self.embedder.describe(),
            "reranker": self.reranker.describe(),
            "llm": self.llm.name,
            "tracer_enabled": self.tracer.enabled,
        }
        return data

    def metrics(self) -> dict[str, Any]:
        return {
            "traces": summarize_traces(self.cfg.trace_path),
            "feedback": summarize_feedback(self.cfg.feedback_path),
            "corpus": self.store.stats(),
        }

    def warnings(self) -> list[str]:
        return self.ingestor.check_consistency()

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "RagApp":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
