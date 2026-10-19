"""结构化切块（structure-aware chunking）。

朴素做法是"按固定字数硬切"，会把表格、代码块、列表拦腰截断，也会让一个块
横跨多个章节导致语义混杂。这里做三件事：

1. **先按 Markdown 结构解析**：标题栈 → ``heading_path``；代码块 / 表格 /
   列表 / 段落各自成为不可分割的 block（除非单块本身就超长）。
2. **再按语义块装箱**：贪心累积到 ``chunk_size``；跨越章节时若当前块已过半就
   提前收尾，避免一个块横跨两节。
3. **块间加句子级 overlap**：从上一块**结尾的完整句子**里取，而不是按字符硬切，
   这样 overlap 部分不会出现半截句子。

每个 chunk 记录 ``heading_path``，向量化时把它拼进前缀（见
``Chunk.embedding_text``），相当于免费拿到了"层级上下文"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Settings
from .text import estimate_tokens, normalize, split_sentences, tail_sentences
from .types import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_TABLE_RE = re.compile(r"^\s*\|")


@dataclass(slots=True)
class Block:
    """切块的最小语义单元。"""

    kind: str  # paragraph | code | table | list
    text: str
    heading_path: str = ""


@dataclass(slots=True)
class ChunkingReport:
    """切块质量指标——入库时打印，用来判断参数是否合理。"""

    n_chunks: int = 0
    n_blocks: int = 0
    avg_chars: float = 0.0
    min_chars: int = 0
    max_chars: int = 0
    n_oversized: int = 0
    n_tiny: int = 0
    kinds: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"chunks={self.n_chunks} blocks={self.n_blocks} "
            f"avg={self.avg_chars:.0f} min={self.min_chars} max={self.max_chars} "
            f"oversized={self.n_oversized} tiny={self.n_tiny} kinds={self.kinds}"
        )


# --------------------------------------------------------------------------
# Markdown → blocks
# --------------------------------------------------------------------------
def parse_blocks(text: str) -> list[Block]:
    """把 Markdown 文本解析成语义块，并维护标题路径。"""
    lines = normalize(text).split("\n")
    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    buf: list[str] = []
    kind = "paragraph"
    in_code = False

    def heading_path() -> str:
        return " > ".join(title for _, title in stack)

    def flush() -> None:
        nonlocal buf, kind
        if not buf:
            return
        body = "\n".join(buf).strip()
        buf = []
        if body:
            blocks.append(Block(kind=kind, text=body, heading_path=heading_path()))
        kind = "paragraph"

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                buf.append(line)
                flush()
                kind = "paragraph"
                in_code = False
            else:
                flush()
                in_code = True
                kind = "code"
                buf.append(line)
            continue

        if in_code:
            buf.append(line)
            continue

        m = _HEADING_RE.match(stripped)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue

        if not stripped:
            flush()
            continue

        if _TABLE_RE.match(line):
            if kind != "table":
                flush()
                kind = "table"
            buf.append(line)
            continue

        if _LIST_RE.match(line):
            if kind != "list":
                flush()
                kind = "list"
            buf.append(line)
            continue

        if kind in ("table", "list"):
            # 表格/列表后的普通行：表格续行（无 | 也算结束）则收尾
            flush()
        buf.append(line)

    flush()
    return [b for b in blocks if b.text]


# --------------------------------------------------------------------------
# blocks → chunks
# --------------------------------------------------------------------------
def _split_oversized(block: Block, size: int, overlap: int) -> list[Block]:
    """超长 block 按句子切分（代码块按行切），并保留 overlap。"""
    if block.kind == "code":
        units = block.text.split("\n")
        joiner = "\n"
    else:
        units = split_sentences(block.text)
        joiner = ""

    out: list[Block] = []
    cur: list[str] = []
    cur_len = 0
    for unit in units:
        ulen = len(unit) + len(joiner)
        if cur and cur_len + ulen > size:
            out.append(Block(block.kind, joiner.join(cur).strip(), block.heading_path))
            carry = tail_sentences(joiner.join(cur), overlap) if joiner == "" else ""
            cur = [carry] if carry else []
            cur_len = len(carry)
        cur.append(unit)
        cur_len += ulen
    if cur:
        body = joiner.join(cur).strip()
        if body:
            out.append(Block(block.kind, body, block.heading_path))
    return out


def pack_blocks(blocks: list[Block], cfg: Settings) -> list[Chunk]:
    """把语义块装箱成 chunk 列表（不含 embedding）。"""
    size = max(200, cfg.chunk_size)
    overlap = max(0, min(cfg.chunk_overlap, size // 2))
    min_chars = max(0, cfg.chunk_min_chars)

    normalized: list[Block] = []
    for blk in blocks:
        if len(blk.text) > size * 1.6:
            normalized.extend(_split_oversized(blk, size, overlap))
        else:
            normalized.append(blk)

    chunks: list[Chunk] = []
    cur: list[Block] = []
    cur_len = 0
    cur_heading = ""

    def emit() -> None:
        nonlocal cur, cur_len, cur_heading
        if not cur:
            return
        body = "\n\n".join(b.text for b in cur).strip()
        if body:
            chunks.append(
                Chunk(
                    chunk_id="",  # 由调用方按 ordinal 赋值
                    doc_id="",
                    ordinal=len(chunks),
                    text=body,
                    heading_path=cur_heading,
                    token_count=estimate_tokens(body),
                )
            )
        cur = []
        cur_len = 0

    for blk in normalized:
        blen = len(blk.text) + 2

        # 跨章节且当前块已经过半 → 提前收尾，避免块横跨两节
        if cur and blk.heading_path != cur_heading and cur_len > size * 0.5:
            emit()

        if cur and cur_len + blen > size:
            prev_text = "\n\n".join(b.text for b in cur)
            emit()
            carry = tail_sentences(prev_text, overlap)
            if carry:
                cur = [Block("paragraph", carry, blk.heading_path)]
                cur_len = len(carry) + 2

        if not cur:
            cur_heading = blk.heading_path
        cur.append(blk)
        cur_len += blen

    emit()

    # 合并过短的尾块（避免出现"只有一行"的碎块污染检索）
    if min_chars and len(chunks) >= 2 and len(chunks[-1].text) < min_chars:
        tail = chunks.pop()
        merged = chunks[-1].text + "\n\n" + tail.text
        if len(merged) <= size * 1.5:
            chunks[-1] = Chunk(
                chunk_id="",
                doc_id="",
                ordinal=chunks[-1].ordinal,
                text=merged,
                heading_path=chunks[-1].heading_path,
                token_count=estimate_tokens(merged),
            )
        else:
            chunks.append(tail)

    for i, ch in enumerate(chunks):
        ch.ordinal = i
    return chunks


def chunk_document(doc: Document, cfg: Settings) -> tuple[list[Chunk], ChunkingReport]:
    """文档 → 切块，并回填 doc_id / chunk_id / metadata。"""
    blocks = parse_blocks(doc.text)
    chunks = pack_blocks(blocks, cfg)

    report = ChunkingReport(n_blocks=len(blocks), n_chunks=len(chunks))
    for ch in chunks:
        ch.doc_id = doc.doc_id
        ch.chunk_id = f"{doc.doc_id}#{ch.ordinal:04d}"
        ch.metadata = {
            "source": doc.source,
            "title": doc.title,
            "acl": list(doc.acl),
            "kind": doc.metadata.get("kind", "doc"),
        }
        report.kinds[ch.metadata["kind"]] = report.kinds.get(ch.metadata["kind"], 0) + 1
        if ch.char_len > cfg.chunk_size * 1.5:
            report.n_oversized += 1
        if ch.char_len < cfg.chunk_min_chars:
            report.n_tiny += 1

    lengths = [c.char_len for c in chunks] or [0]
    report.avg_chars = sum(lengths) / len(lengths)
    report.min_chars = min(lengths)
    report.max_chars = max(lengths)
    return chunks, report
