"""入库管线：加载 → 结构化切块 → 向量化 → 落库（增量、幂等）。

增量策略
--------
以 **来源路径 + 内容 hash** 判断是否需要重做：
* hash 相同 → ``skipped``（省下最贵的 embedding 调用，这是成本大头）
* hash 不同 → 删旧 chunk、写新 chunk、版本号 +1（``updated``）
* 新文件 → ``added``

注意一个真实工程细节：换 embedding 模型后，旧向量与新向量**不在同一空间**，
混用会得到"看起来能跑但排序全乱"的结果。所以向量检索按 ``embed_model`` 过滤，
入库时也会提示库中已有的模型与当前配置不一致。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .chunking import chunk_document
from .config import Settings
from .embedding import Embedder
from .loaders import load_path
from .store import SqliteStore
from .types import Document


@dataclass
class IngestResult:
    source: str
    doc_id: str = ""
    status: str = "added"  # added | updated | skipped | failed
    n_chunks: int = 0
    title: str = ""
    error: str = ""
    report: str = ""
    ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "doc_id": self.doc_id,
            "status": self.status,
            "title": self.title,
            "n_chunks": self.n_chunks,
            "error": self.error,
            "report": self.report,
            "ms": round(self.ms, 1),
        }


@dataclass
class IngestSummary:
    results: list[IngestResult] = field(default_factory=list)

    @property
    def added(self) -> int:
        return sum(1 for r in self.results if r.status == "added")

    @property
    def updated(self) -> int:
        return sum(1 for r in self.results if r.status == "updated")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    @property
    def chunks(self) -> int:
        """本次**实际写入**的切块数（skipped 不算，便于在 CI/日志里直接看增量效果）。"""
        return sum(r.n_chunks for r in self.results if r.status in ("added", "updated"))

    def to_dict(self) -> dict:
        return {
            "added": self.added,
            "updated": self.updated,
            "skipped": self.skipped,
            "failed": self.failed,
            "chunks": self.chunks,
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        return (
            f"added={self.added} updated={self.updated} skipped={self.skipped} "
            f"failed={self.failed} chunks={self.chunks}"
        )


class Ingestor:
    def __init__(self, store: SqliteStore, embedder: Embedder, cfg: Settings) -> None:
        self.store = store
        self.embedder = embedder
        self.cfg = cfg

    # ------------------------------------------------------------------
    def embed_chunks(self, chunks, title: str = "") -> int:
        """按批向量化；每批失败只影响该批，避免一次网络抖动毁掉整篇文档。"""
        if not chunks:
            return 0
        done = 0
        batch_size = max(1, self.cfg.embed_batch_size)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.embedding_text(title) for c in batch]
            vectors = self.embedder.embed(texts)
            for ch, vec in zip(batch, vectors):
                ch.embedding = vec
                ch.embed_model = self.embedder.name
                done += 1
        return done

    def ingest_document(self, doc: Document, *, force: bool = False) -> IngestResult:
        t0 = time.perf_counter()
        # 双保险：先按来源路径找，再按 doc_id 找。两者本应一致，但路径写法变化
        # （相对/绝对、目录搬家）会让按来源的查找落空；这时按 doc_id 兜住，
        # 避免把同一篇文档误报成 added，并白算一次向量。
        existing = self.store.get_document_by_source(doc.source) or self.store.get_document(
            doc.doc_id
        )
        if existing and existing.content_hash == doc.content_hash and not force:
            # 内容没变，但权限标签变了：必须单独更新，否则权限改动会被静默丢弃
            # （把该收紧的文档继续公开，是这类系统里最危险的一类 bug）。
            if set(existing.acl) != set(doc.acl):
                touched = self.store.update_document_acl(existing.doc_id, doc.acl)
                return IngestResult(
                    source=doc.source,
                    doc_id=existing.doc_id,
                    status="updated",
                    # n_chunks 表示"重新写入的切块数"，权限变更没有重写任何切块，
                    # 所以是 0；受影响的切块数放在报告里，避免把"改了权限"误读成"重算了向量"。
                    n_chunks=0,
                    title=existing.title,
                    report=(
                        f"acl-only {sorted(existing.acl)} -> {sorted(doc.acl)}"
                        f"（{touched} 个切块，未重算向量）"
                    ),
                    ms=(time.perf_counter() - t0) * 1000,
                )
            return IngestResult(
                source=doc.source,
                doc_id=existing.doc_id,
                status="skipped",
                n_chunks=existing.n_chunks,
                title=existing.title,
                ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            chunks, report = chunk_document(doc, self.cfg)
            if not chunks:
                return IngestResult(
                    source=doc.source,
                    doc_id=doc.doc_id,
                    status="failed",
                    error="切块结果为空（文档内容过短或格式异常）",
                )
            self.embed_chunks(chunks, doc.title)
            self.store.upsert_document(doc, chunks)
        except Exception as e:
            return IngestResult(
                source=doc.source,
                doc_id=doc.doc_id,
                status="failed",
                error=f"{type(e).__name__}: {e}",
                ms=(time.perf_counter() - t0) * 1000,
            )

        return IngestResult(
            source=doc.source,
            doc_id=doc.doc_id,
            status="updated" if existing else "added",
            n_chunks=len(chunks),
            title=doc.title,
            report=report.summary(),
            ms=(time.perf_counter() - t0) * 1000,
        )

    def ingest_path(
        self,
        path: str | Path,
        *,
        acl: Sequence[str] = (),
        force: bool = False,
        recursive: bool = True,
    ) -> IngestSummary:
        summary = IngestSummary()
        batch = load_path(path, acl=tuple(acl), recursive=recursive)
        for err in batch.errors:
            summary.results.append(IngestResult(source=err, status="failed", error=err))
        for doc in batch.documents:
            summary.results.append(self.ingest_document(doc, force=force))
        return summary

    # ------------------------------------------------------------------
    def check_consistency(self) -> list[str]:
        """返回"库里已有向量 vs 当前配置"的不一致告警。"""
        warnings: list[str] = []
        models = {m["model"] for m in self.store.stats().get("embedding_models", [])}
        stale = {m for m in models if m and m != self.embedder.name}
        if stale:
            warnings.append(
                f"库中已有来自 {sorted(stale)} 的向量，与当前 embedder({self.embedder.name}) "
                "不一致；这些向量不会被检索到，建议重新入库（ingest --force）。"
            )
        return warnings
