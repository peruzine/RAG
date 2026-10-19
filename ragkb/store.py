"""持久化层：单文件 SQLite（零依赖）+ 真正的倒排索引 + 向量检索。

为什么是 SQLite
---------------
* 标准库自带，`pip install` 都不需要，单文件就能交付/备份；
* 事务、并发（WAL）、索引能力齐全，完全够中小规模（十万级 chunk）使用；
* 倒排索引直接落库，**进程重启不用重建**——这是很多"内存版 RAG demo"缺的一环。

表结构
------
``documents``     文档元信息 + 内容 hash（增量入库用）
``chunks``        切块正文 + embedding(float32 blob) + 所属 embedding 模型
``postings``      (term, chunk_id, tf) —— BM25 的倒排表，`WITHOUT ROWID` 省空间
``term_stats``    (term, df) —— BM25 的 IDF 需要文档频率
``chunk_acl``     (chunk_id, role) —— 行级权限过滤
``meta``          语料统计与版本号（缓存失效用）

规模上限与演进路径
------------------
单机 SQLite + Python 侧打分在 ~10 万 chunk 内表现良好（检索 p95 < 100ms）。
再往上应该换 OpenSearch / Milvus，本项目保留 ``lexical`` / ``vector`` 两个通道的
接口抽象，替换时不影响上层。
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
from array import array
from pathlib import Path
from typing import Any, Iterable, Sequence  # noqa: F401  (Path 供外部类型标注)

from .text import index_terms, query_terms
from .types import Chunk, Document

try:  # numpy 只是加速项，没有也能跑
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    source       TEXT,
    title        TEXT,
    text         TEXT,
    metadata     TEXT,
    content_hash TEXT,
    added_at     REAL,
    updated_at   REAL,
    version      INTEGER DEFAULT 1,
    n_chunks     INTEGER DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT,
    ordinal      INTEGER,
    text         TEXT,
    heading_path TEXT,
    token_count  INTEGER,
    metadata     TEXT,
    embedding    BLOB,
    embed_model  TEXT,
    dim          INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS postings (
    term     TEXT,
    chunk_id TEXT,
    tf       INTEGER,
    PRIMARY KEY (term, chunk_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_postings_chunk ON postings(chunk_id);
CREATE TABLE IF NOT EXISTS term_stats (
    term TEXT PRIMARY KEY,
    df   INTEGER
);
CREATE TABLE IF NOT EXISTS chunk_acl (
    chunk_id TEXT,
    role     TEXT,
    PRIMARY KEY (chunk_id, role)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_chunk_acl_role ON chunk_acl(role);
"""


def pack_vector(vec: Sequence[float]) -> bytes:
    return array("f", [float(x) for x in vec]).tobytes()


def unpack_vector(blob: bytes) -> list[float]:
    arr = array("f")
    arr.frombytes(blob)
    return list(arr)


class SqliteStore:
    """线程安全的存储门面。

    每个线程持有独立连接（SQLite 连接不能跨线程共享），写操作加锁串行化。
    """

    def __init__(self, path: str | Path = ":memory:", *, k1: float = 1.5, b: float = 0.75) -> None:
        self.path = str(path)
        self.k1 = k1
        self.b = b
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._vec_cache: dict[str, tuple[list[str], Any]] = {}
        self._acl_cache: dict[Any, set[str]] = {}
        self._cache_version: str = ""
        # 内存库必须用 shared cache：否则每个线程（每个连接）会各自拿到一个
        # 空数据库，服务端多线程场景下会出现"刚写入就查不到"的诡异现象。
        # 名字必须**每个实例唯一**：否则同一进程里多个 SqliteStore(":memory:")
        # 会共享同一个库，测试之间互相污染（实测踩过）。
        if self.path in (":memory:", ""):
            self._dsn = f"file:ragkb-mem-{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._uri = True
        else:
            self._dsn, self._uri = self.path, False
        self._init_schema()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self._dsn, timeout=30.0, isolation_level=None, uri=self._uri)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys=ON")
            self._local.conn = c
        return c

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    def _init_schema(self) -> None:
        with self._write_lock:
            self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------------
    # meta
    # ------------------------------------------------------------------
    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def invalidate_cache(self) -> None:
        self._vec_cache.clear()
        self._acl_cache.clear()

    # ------------------------------------------------------------------
    # 跨进程缓存一致性
    # ------------------------------------------------------------------
    def touch_version(self) -> None:
        """标记语料已变更。它的作用是让**别的进程**知道自己的缓存过期了。

        单进程内 ``invalidate_cache()`` 就够了；但真实部署里常见"CLI 入库 + 常驻服务
        读库"两个进程并存，此时服务进程内存里缓存的全量向量矩阵与 ACL 集合不会自动
        失效——权限收紧后会继续按旧权限放行（实测踩过：CLI 把一篇文档改成 internal，
        服务进程仍然把它当 public 检索出来）。
        """
        self.set_meta("corpus_version", f"{time.time():.6f}")

    def _ensure_cache_fresh(self) -> None:
        """读一次 meta 里的版本号（主键查询，微秒级），版本变了就丢弃本地缓存。"""
        version = self.get_meta("corpus_version")
        if version != self._cache_version:
            self._cache_version = version
            self._vec_cache.clear()
            self._acl_cache.clear()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def upsert_document(self, doc: Document, chunks: list[Chunk]) -> dict[str, int]:
        """整篇替换写入：文档 + 切块 + 倒排 + 权限，单事务。"""
        now = time.time()
        doc.added_at = doc.added_at or now
        doc.updated_at = now
        doc.n_chunks = len(chunks)

        postings: list[tuple[str, str, int]] = []
        acl_rows: list[tuple[str, str]] = []
        for ch in chunks:
            tf: dict[str, int] = {}
            for term in index_terms(ch.text):
                tf[term] = tf.get(term, 0) + 1
            postings.extend((term, ch.chunk_id, n) for term, n in tf.items())
            roles = ch.metadata.get("acl") or ["public"]
            acl_rows.extend((ch.chunk_id, r) for r in roles)

        chunk_rows = [
            (
                ch.chunk_id,
                ch.doc_id,
                ch.ordinal,
                ch.text,
                ch.heading_path,
                ch.token_count,
                json.dumps(ch.metadata, ensure_ascii=False),
                pack_vector(ch.embedding) if ch.embedding else None,
                ch.embed_model,
                len(ch.embedding) if ch.embedding else 0,
            )
            for ch in chunks
        ]

        existing = self.conn.execute(
            "SELECT version FROM documents WHERE doc_id=?", (doc.doc_id,)
        ).fetchone()
        doc.version = (existing["version"] + 1) if existing else 1

        with self._write_lock:
            c = self.conn
            c.execute("BEGIN IMMEDIATE")
            try:
                self._delete_chunks_unlocked(doc.doc_id)
                c.execute(
                    """INSERT INTO documents
                       (doc_id,source,title,text,metadata,content_hash,added_at,updated_at,version,n_chunks)
                       VALUES(?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(doc_id) DO UPDATE SET
                         source=excluded.source, title=excluded.title, text=excluded.text,
                         metadata=excluded.metadata, content_hash=excluded.content_hash,
                         updated_at=excluded.updated_at, version=excluded.version,
                         n_chunks=excluded.n_chunks""",
                    (
                        doc.doc_id,
                        doc.source,
                        doc.title,
                        doc.text,
                        json.dumps(doc.metadata, ensure_ascii=False),
                        doc.content_hash,
                        doc.added_at,
                        doc.updated_at,
                        doc.version,
                        doc.n_chunks,
                    ),
                )
                c.executemany(
                    """INSERT INTO chunks
                       (chunk_id,doc_id,ordinal,text,heading_path,token_count,metadata,embedding,embed_model,dim)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    chunk_rows,
                )
                c.executemany(
                    "INSERT OR REPLACE INTO postings(term,chunk_id,tf) VALUES(?,?,?)", postings
                )
                c.executemany(
                    "INSERT OR REPLACE INTO chunk_acl(chunk_id,role) VALUES(?,?)", acl_rows
                )
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise

        self._refresh_term_stats(postings)
        self.invalidate_cache()
        self.touch_version()
        return {"chunks": len(chunks), "postings": len(postings), "version": doc.version}

    def update_document_acl(self, doc_id: str, roles: Sequence[str]) -> int:
        """只改权限标签，不重算向量，返回受影响的 chunk 数。

        为什么单独开一条路径：内容没变、只是权限变了（比如某篇文档从 public 收紧到
        internal）是**很常见也最危险**的一类变更——如果走"内容 hash 相同就 skip"的
        逻辑，权限改动会被静默丢弃，等于把不该公开的内容继续公开。而为了改几个标签
        去重算全篇向量又太浪费，所以这里只更新权限表与元数据。
        """
        roles = list(dict.fromkeys(roles)) or ["public"]
        ids = [
            r["chunk_id"]
            for r in self.conn.execute("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,))
        ]
        if not ids:
            return 0
        with self._write_lock:
            c = self.conn
            c.execute("BEGIN IMMEDIATE")
            try:
                for i in range(0, len(ids), 500):
                    batch = ids[i : i + 500]
                    q = ",".join("?" * len(batch))
                    c.execute(f"DELETE FROM chunk_acl WHERE chunk_id IN ({q})", batch)
                c.executemany(
                    "INSERT OR REPLACE INTO chunk_acl(chunk_id,role) VALUES(?,?)",
                    [(cid, r) for cid in ids for r in roles],
                )
                # 同步元数据里的 acl 副本，避免"列表显示 public、实际检索已收紧"这类不一致
                for cid in ids:
                    row = c.execute("SELECT metadata FROM chunks WHERE chunk_id=?", (cid,)).fetchone()
                    meta = json.loads(row["metadata"] or "{}")
                    meta["acl"] = roles
                    c.execute(
                        "UPDATE chunks SET metadata=? WHERE chunk_id=?",
                        (json.dumps(meta, ensure_ascii=False), cid),
                    )
                row = c.execute("SELECT metadata FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
                if row:
                    meta = json.loads(row["metadata"] or "{}")
                    meta["acl"] = roles
                    c.execute(
                        "UPDATE documents SET metadata=?, updated_at=?, version=version+1 WHERE doc_id=?",
                        (json.dumps(meta, ensure_ascii=False), time.time(), doc_id),
                    )
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise
        self.invalidate_cache()
        self.touch_version()
        return len(ids)

    def _delete_chunks_unlocked(self, doc_id: str) -> int:
        ids = [r["chunk_id"] for r in self.conn.execute(
            "SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,)
        )]
        if not ids:
            return 0
        c = self.conn
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            q = ",".join("?" * len(batch))
            c.execute(f"DELETE FROM postings WHERE chunk_id IN ({q})", batch)
            c.execute(f"DELETE FROM chunk_acl WHERE chunk_id IN ({q})", batch)
        c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        return len(ids)

    def _refresh_term_stats(self, postings: Iterable[tuple[str, str, int]]) -> None:
        """增量维护 df（文档频率）。删文档时会全量重算，见 ``recompute_stats``。"""
        counts: dict[str, int] = {}
        for term, _cid, _tf in postings:
            counts[term] = counts.get(term, 0) + 1
        with self._write_lock:
            self.conn.executemany(
                "INSERT INTO term_stats(term,df) VALUES(?,?) "
                "ON CONFLICT(term) DO UPDATE SET df=excluded.df",
                list(counts.items()),
            )

    def delete_document(self, doc_id: str) -> bool:
        with self._write_lock:
            c = self.conn
            c.execute("BEGIN IMMEDIATE")
            try:
                removed = self._delete_chunks_unlocked(doc_id)
                cur = c.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise
        if removed:
            self.recompute_stats()
        self.invalidate_cache()
        self.touch_version()
        return bool(cur.rowcount)

    def recompute_stats(self) -> int:
        """全量重算 df（删除文档后必须做，否则 IDF 会偏大）。"""
        with self._write_lock:
            c = self.conn
            c.execute("DELETE FROM term_stats")
            c.execute(
                "INSERT INTO term_stats(term,df) SELECT term, COUNT(*) FROM postings GROUP BY term"
            )
            return c.execute("SELECT COUNT(*) AS n FROM term_stats").fetchone()["n"]

    def prune_postings(self, max_df_ratio: float = 0.5) -> int:
        """按统计剔除高频词（IDF≈0 的词对打分毫无贡献，纯占空间）。"""
        n = max(1, self.count_chunks())
        limit = max(2, int(n * max_df_ratio))
        with self._write_lock:
            c = self.conn
            victims = [r["term"] for r in c.execute("SELECT term FROM term_stats WHERE df>?", (limit,))]
            for i in range(0, len(victims), 500):
                batch = victims[i : i + 500]
                q = ",".join("?" * len(batch))
                c.execute(f"DELETE FROM postings WHERE term IN ({q})", batch)
                c.execute(f"DELETE FROM term_stats WHERE term IN ({q})", batch)
        self.invalidate_cache()
        self.touch_version()
        return len(victims)

    def clear(self) -> None:
        with self._write_lock:
            c = self.conn
            for t in ("postings", "term_stats", "chunk_acl", "chunks", "documents", "meta"):
                c.execute(f"DELETE FROM {t}")
        self.invalidate_cache()
        self.touch_version()

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def _doc_from_row(self, row: sqlite3.Row) -> Document:
        return Document(
            doc_id=row["doc_id"],
            source=row["source"] or "",
            title=row["title"] or "",
            text=row["text"] or "",
            acl=tuple(r["role"] for r in self.conn.execute(
                "SELECT DISTINCT role FROM chunk_acl WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE doc_id=?)",
                (row["doc_id"],),
            )),
            metadata=json.loads(row["metadata"] or "{}"),
            content_hash=row["content_hash"] or "",
            added_at=row["added_at"] or 0.0,
            updated_at=row["updated_at"] or 0.0,
            version=row["version"] or 1,
            n_chunks=row["n_chunks"] or 0,
        )

    def get_document(self, doc_id: str) -> Document | None:
        row = self.conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return self._doc_from_row(row) if row else None

    def get_document_by_source(self, source: str) -> Document | None:
        row = self.conn.execute("SELECT * FROM documents WHERE source=?", (source,)).fetchone()
        return self._doc_from_row(row) if row else None

    def list_documents(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT doc_id,source,title,content_hash,updated_at,version,n_chunks,metadata "
            "FROM documents ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            meta = json.loads(r["metadata"] or "{}")
            # ACL 以 chunk_acl 为准：它是权限过滤真正读的地方，文档级 metadata
            # 里的 acl 只是冗余展示位，两者不一致时必须信 chunk_acl。
            roles = [
                row["role"]
                for row in self.conn.execute(
                    "SELECT DISTINCT role FROM chunk_acl WHERE chunk_id IN "
                    "(SELECT chunk_id FROM chunks WHERE doc_id=?) ORDER BY role",
                    (r["doc_id"],),
                )
            ]
            out.append(
                {
                    "doc_id": r["doc_id"],
                    "source": r["source"],
                    "title": r["title"],
                    "n_chunks": r["n_chunks"],
                    "updated_at": r["updated_at"],
                    "version": r["version"],
                    "acl": roles or meta.get("acl", ["public"]),
                    "content_hash": r["content_hash"],
                }
            )
        return out

    def _chunk_from_row(self, row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            ordinal=row["ordinal"] or 0,
            text=row["text"] or "",
            heading_path=row["heading_path"] or "",
            token_count=row["token_count"] or 0,
            metadata=json.loads(row["metadata"] or "{}"),
            embed_model=row["embed_model"] or "",
        )

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        out: list[Chunk] = []
        ids = list(chunk_ids)
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            q = ",".join("?" * len(batch))
            rows = self.conn.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({q})", batch).fetchall()
            out.extend(self._chunk_from_row(r) for r in rows)
        order = {cid: i for i, cid in enumerate(ids)}
        out.sort(key=lambda ch: order.get(ch.chunk_id, 1 << 30))
        return out

    def iter_chunks(self, doc_id: str | None = None) -> Iterable[Chunk]:
        if doc_id:
            rows = self.conn.execute(
                "SELECT * FROM chunks WHERE doc_id=? ORDER BY ordinal", (doc_id,)
            )
        else:
            rows = self.conn.execute("SELECT * FROM chunks ORDER BY doc_id, ordinal")
        for r in rows:
            yield self._chunk_from_row(r)

    def count_chunks(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]

    def count_documents(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]

    # ------------------------------------------------------------------
    # ACL
    # ------------------------------------------------------------------
    def allowed_chunk_ids(
        self, roles: Sequence[str] | None, doc_ids: Sequence[str] | None = None
    ) -> set[str] | None:
        """返回可见 chunk 集合；``roles is None`` 表示不做权限过滤（管理员）。"""
        if roles is None and not doc_ids:
            return None
        self._ensure_cache_fresh()
        key = (tuple(roles) if roles is not None else None, tuple(doc_ids or ()))
        cached = self._acl_cache.get(key)
        if cached is not None:
            return cached

        clauses: list[str] = []
        params: list[Any] = []
        if roles is not None:
            roles = tuple(roles) or ("public",)
            q = ",".join("?" * len(roles))
            # 没打 ACL 标记的 chunk 默认对所有人可见
            clauses.append(
                f"(c.chunk_id IN (SELECT chunk_id FROM chunk_acl WHERE role IN ({q})) "
                "OR NOT EXISTS (SELECT 1 FROM chunk_acl a2 WHERE a2.chunk_id=c.chunk_id))"
            )
            params.extend(roles)
        if doc_ids:
            q = ",".join("?" * len(doc_ids))
            clauses.append(f"c.doc_id IN ({q})")
            params.extend(doc_ids)

        sql = "SELECT c.chunk_id FROM chunks c"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        ids = {r["chunk_id"] for r in self.conn.execute(sql, params)}
        if len(self._acl_cache) > 64:
            self._acl_cache.clear()
        self._acl_cache[key] = ids
        return ids

    # ------------------------------------------------------------------
    # 词法检索（BM25）
    # ------------------------------------------------------------------
    def lexical_search(
        self,
        query: str,
        top_k: int = 50,
        *,
        roles: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        terms = query_terms(query)
        if not terms:
            return []
        allowed = self.allowed_chunk_ids(roles, doc_ids)
        if allowed is not None and not allowed:
            return []

        n_docs = max(1, self.count_chunks())
        avgdl = float(self.get_meta("avgdl", "0") or 0.0)
        if avgdl <= 0:
            avgdl = self._compute_avgdl()

        placeholders = ",".join("?" * len(terms))
        df_map = {
            r["term"]: r["df"]
            for r in self.conn.execute(
                f"SELECT term,df FROM term_stats WHERE term IN ({placeholders})", terms
            )
        }
        rows = self.conn.execute(
            f"""SELECT p.term AS term, p.chunk_id AS chunk_id, p.tf AS tf, c.token_count AS dl
                FROM postings p JOIN chunks c ON c.chunk_id = p.chunk_id
                WHERE p.term IN ({placeholders})""",
            terms,
        )

        scores: dict[str, float] = {}
        k1, b = self.k1, self.b
        for r in rows:
            cid = r["chunk_id"]
            if allowed is not None and cid not in allowed:
                continue
            df = df_map.get(r["term"], 1)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            dl = r["dl"] or 1
            denom = r["tf"] + k1 * (1 - b + b * dl / avgdl)
            scores[cid] = scores.get(cid, 0.0) + idf * r["tf"] * (k1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]

    def _compute_avgdl(self) -> float:
        row = self.conn.execute(
            "SELECT AVG(token_count) AS a, COUNT(*) AS n FROM chunks"
        ).fetchone()
        avg = float(row["a"] or 0.0) or 300.0
        self.set_meta("avgdl", f"{avg:.4f}")
        return avg

    def term_df(self, terms: Sequence[str]) -> dict[str, int]:
        """查询词的文档频率（语料里出现过多少次 chunk）。未出现为 0。"""
        terms = list(dict.fromkeys(terms))
        if not terms:
            return {}
        placeholders = ",".join("?" * len(terms))
        found = {
            r["term"]: r["df"]
            for r in self.conn.execute(
                f"SELECT term,df FROM term_stats WHERE term IN ({placeholders})", terms
            )
        }
        return {t: int(found.get(t, 0)) for t in terms}

    def term_idf(self, terms: Sequence[str]) -> dict[str, float]:
        """查询词的 IDF（稀疏度权重），用于"问题里的实体在语料里到底有没有"的判断。

        ``df=0``（语料里完全没出现过）的词给最大权重：这既可能是"用户问的是语料
        之外的东西"（我们要靠它拒答），也可能是中文二元切分跨词边界的产物，
        两种情况下"没匹配上"都是有效信号，所以不特殊处理，直接按最稀有算。
        """
        terms = list(dict.fromkeys(terms))
        if not terms:
            return {}
        n = max(1, self.count_chunks())
        df_map = self.term_df(terms)
        return {t: math.log(1.0 + n / (1.0 + df_map.get(t, 0))) for t in terms}

    # ------------------------------------------------------------------
    # 向量检索
    # ------------------------------------------------------------------
    def _vector_matrix(self, model: str | None) -> tuple[list[str], Any]:
        self._ensure_cache_fresh()
        key = model or "*"
        cached = self._vec_cache.get(key)
        if cached is not None:
            return cached
        where = "embedding IS NOT NULL"
        params: list[Any] = []
        if model:
            where += " AND embed_model=?"
            params.append(model)
        rows = self.conn.execute(
            f"SELECT chunk_id, embedding, dim FROM chunks WHERE {where}", params
        ).fetchall()
        ids = [r["chunk_id"] for r in rows]
        blobs = [r["embedding"] for r in rows]
        if not blobs:
            self._vec_cache[key] = (ids, None)
            return ids, None
        if _np is not None:
            dim = len(blobs[0]) // 4
            mat = _np.frombuffer(b"".join(blobs), dtype=_np.float32).reshape(len(blobs), dim)
            self._vec_cache[key] = (ids, mat)
        else:  # pragma: no cover - 有 numpy 时不会走到
            self._vec_cache[key] = (ids, [unpack_vector(b) for b in blobs])
        return self._vec_cache[key]

    def vector_search(
        self,
        vector: Sequence[float],
        top_k: int = 50,
        *,
        roles: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
        model: str | None = None,
        min_similarity: float = 0.0,
    ) -> list[tuple[str, float]]:
        """余弦相似度暴力检索。向量已 L2 归一化时点积即余弦，这里仍做归一化兜底。

        ``min_similarity`` 用来剪掉"相似度接近 0 的纯噪声"。注意它依赖具体
        embedding 模型的分布：本地 hashing 兜底向量整体相似度偏高（共享大量
        汉字二元组），线上 BGE/OpenAI 类模型 0.15–0.25 是常用区间，所以默认不启用。
        """
        ids, mat = self._vector_matrix(model)
        if not ids or mat is None:
            return []
        allowed = self.allowed_chunk_ids(roles, doc_ids)

        if _np is not None:
            q = _np.asarray(vector, dtype=_np.float32)
            if q.shape[0] != mat.shape[1]:
                raise ValueError(
                    f"向量维度不匹配：查询 {q.shape[0]} vs 索引 {mat.shape[1]}（换过 embedding 模型？）"
                )
            qn = float(_np.linalg.norm(q)) or 1.0
            norms = _np.linalg.norm(mat, axis=1)
            norms[norms == 0] = 1.0
            sims = (mat @ q) / (norms * qn)
            idx = _np.argsort(-sims)
            out: list[tuple[str, float]] = []
            for i in idx:
                score = float(sims[int(i)])
                if score < min_similarity:
                    break
                cid = ids[int(i)]
                if allowed is not None and cid not in allowed:
                    continue
                out.append((cid, score))
                if len(out) >= top_k:
                    break
            return out

        qn = math.sqrt(sum(float(x) * float(x) for x in vector)) or 1.0  # pragma: no cover
        scored: list[tuple[str, float]] = []
        for cid, vec in zip(ids, mat):
            dot = sum(float(a) * float(b) for a, b in zip(vec, vector))
            vn = math.sqrt(sum(float(x) * float(x) for x in vec)) or 1.0
            if allowed is not None and cid not in allowed:
                continue
            scored.append((cid, dot / (vn * qn)))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:top_k]

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        n_chunks = self.count_chunks()
        n_docs = self.count_documents()
        row = self.conn.execute(
            "SELECT AVG(LENGTH(text)) AS a, MAX(LENGTH(text)) AS m FROM chunks"
        ).fetchone()
        models = [
            {"model": r["embed_model"], "n": r["n"], "dim": r["dim"]}
            for r in self.conn.execute(
                "SELECT embed_model, dim, COUNT(*) AS n FROM chunks "
                "WHERE embedding IS NOT NULL GROUP BY embed_model, dim"
            )
        ]
        return {
            "documents": n_docs,
            "chunks": n_chunks,
            "terms": self.conn.execute("SELECT COUNT(*) AS n FROM term_stats").fetchone()["n"],
            "postings": self.conn.execute("SELECT COUNT(*) AS n FROM postings").fetchone()["n"],
            "avg_chunk_chars": round(float(row["a"] or 0.0), 1),
            "max_chunk_chars": int(row["m"] or 0),
            "embedding_models": models,
            "acl_roles": [
                r["role"]
                for r in self.conn.execute("SELECT DISTINCT role FROM chunk_acl ORDER BY role")
            ],
            "db_path": self.path,
        }
