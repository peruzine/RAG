"""核心数据结构。

全部使用标准库 ``dataclass``，不引入 pydantic，因此核心层可以零依赖运行。
HTTP 层如果需要再做轻量转换。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """一篇入库文档。``doc_id`` 默认由来源路径/标题派生，保证幂等重入。"""

    doc_id: str
    source: str
    title: str
    text: str = ""
    acl: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    added_at: float = 0.0
    updated_at: float = 0.0
    version: int = 1
    n_chunks: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["acl"] = list(self.acl)
        return d


@dataclass(slots=True)
class Chunk:
    """文档切块，检索的最小单位。"""

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    heading_path: str = ""
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    embed_model: str = ""

    @property
    def char_len(self) -> int:
        return len(self.text)

    def embedding_text(self, title: str = "") -> str:
        """用于向量化的文本：注入标题与章节路径（contextual retrieval 的基础做法）。

        检索时只返回 ``text`` 原文，但向量里带上层级信息，可以显著缓解
        "块太小导致指代不明" 的问题。
        """
        prefix = " > ".join(p for p in (title, self.heading_path) if p)
        return f"{prefix}\n{self.text}" if prefix else self.text


@dataclass(slots=True)
class Retrieved:
    """一条检索结果，携带每个召回通道里的排名，便于排查"是谁召回的"。"""

    chunk: Chunk
    score: float = 0.0
    channels: dict[str, int] = field(default_factory=dict)
    channel_scores: dict[str, float] = field(default_factory=dict)
    rerank_score: float | None = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass(slots=True)
class Citation:
    """答案里的引用（对应 ``[1]`` ``[2]`` 角标）。"""

    index: int
    chunk_id: str
    doc_id: str
    source: str
    title: str
    heading_path: str
    quote: str
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class Answer:
    """一次问答的完整结果（含可观测信息）。"""

    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    trace_id: str = ""
    retrieved: list[Retrieved] = field(default_factory=list)
    rewritten_query: str = ""
    route: str = "kb"
    cost_usd: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "refused": self.refused,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "latency_ms": round(self.latency_ms, 2),
            "trace_id": self.trace_id,
            "rewritten_query": self.rewritten_query,
            "route": self.route,
            "cost_usd": round(self.cost_usd, 8),
            "citations": [c.to_dict() for c in self.citations],
            # debug 必须一并返回：线上排查"为什么答错了"靠的就是这些字段
            # （召回明细、覆盖率、拒答原因、越界引用、是否引用了但没给角标）
            "debug": self.debug,
            "retrieved": [
                {
                    "chunk_id": r.chunk_id,
                    "doc_id": r.chunk.doc_id,
                    "ordinal": r.chunk.ordinal,
                    "heading_path": r.chunk.heading_path,
                    "score": round(r.score, 6),
                    "rerank_score": None
                    if r.rerank_score is None
                    else round(r.rerank_score, 6),
                    "channels": r.channels,
                    "snippet": r.chunk.text[:180],
                }
                for r in self.retrieved
            ],
        }


@dataclass(slots=True)
class EvalCase:
    """评测集单条样本。

    gold 标注优先用 ``gold_sources`` + ``gold_snippets``（对切块参数变化免疫），
    只有在需要精确到块时才用 ``gold_chunk_ids``。
    """

    case_id: str
    question: str
    gold_sources: list[str] = field(default_factory=list)
    gold_doc_ids: list[str] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_snippets: list[str] = field(default_factory=list)
    gold_answer: str = ""
    should_refuse: bool = False
    tags: list[str] = field(default_factory=list)
    roles: tuple[str, ...] | None = ()
    history: list[dict[str, str]] = field(default_factory=list)
    note: str = ""
