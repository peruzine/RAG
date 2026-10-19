"""文档加载：把各种格式统一成 Markdown-ish 文本。

支持的格式按"依赖是否可用"分层：
* **零依赖**：``.md .markdown .txt .log .csv``（csv 渲染成 Markdown 表格）
* **零依赖**：``.html .htm`` —— 用标准库 ``html.parser`` 抽正文，保留块级换行
* **可选依赖**：``.docx``（python-docx）、``.xlsx``（openpyxl）、``.pdf``（pypdf）

统一转成 Markdown 有一个实际好处：下游切块器只认 Markdown 结构（标题/表格/代码块），
所以"Word 里的标题层级"和"Excel 里的表格"都能享受结构化切块的收益。

解析失败**不抛异常到顶层**：返回 ``LoadResult(ok=False, error=...)``，
由入库管线汇总成报告，一篇坏文件不会中断整批入库。
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator

from .text import normalize, stable_hash
from .types import Document

TEXT_SUFFIXES = {".md", ".markdown", ".mdx", ".txt", ".text", ".log", ".rst", ".yaml", ".yml", ".json"}
HTML_SUFFIXES = {".html", ".htm", ".xhtml"}
CSV_SUFFIXES = {".csv", ".tsv"}
DOCX_SUFFIXES = {".docx"}
XLSX_SUFFIXES = {".xlsx", ".xlsm"}
PDF_SUFFIXES = {".pdf"}

SUPPORTED_SUFFIXES = (
    TEXT_SUFFIXES | HTML_SUFFIXES | CSV_SUFFIXES | DOCX_SUFFIXES | XLSX_SUFFIXES | PDF_SUFFIXES
)

_BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "table", "ul", "ol", "blockquote", "pre",
}


@dataclass
class LoadResult:
    document: Document | None = None
    ok: bool = True
    error: str = ""

    @staticmethod
    def fail(path: str, error: str) -> "LoadResult":
        return LoadResult(document=None, ok=False, error=f"{path}: {error}")


class _HtmlTextExtractor(HTMLParser):
    """把 HTML 抽成带块级换行的纯文本（不引入 bs4）。"""

    SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._heading_level = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if re.fullmatch(r"h[1-6]", tag):
            self._heading_level = int(tag[1])
            self.parts.append("#" * self._heading_level + " ")
        if tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self.parts.append(text + " ")

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r" *\n *", "\n", raw)
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def _html_to_markdown(raw: str) -> str:
    parser = _HtmlTextExtractor()
    parser.feed(raw)
    return parser.text()


def _rows_to_markdown(rows: list[list[str]]) -> str:
    rows = [[("" if c is None else str(c)).replace("|", "\\|").strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    head = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join([head, sep] + body)


def _read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", "replace")


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        m = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line.strip())
        if m:
            return m.group(1).strip()
    return ""


def derive_doc_id(source: str, title: str = "") -> str:
    """doc_id 由来源路径决定 → 同一个文件重复入库幂等，不会产生副本。"""
    return f"doc-{stable_hash(source)}"


def load_file(path: str | Path, *, acl: tuple[str, ...] = (), title: str = "") -> LoadResult:
    """读取单个文件并转成 :class:`Document`。"""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return LoadResult.fail(str(p), "文件不存在")
    suffix = p.suffix.lower()
    try:
        if suffix in TEXT_SUFFIXES or suffix == "":
            body = _read_text(p)
            kind = "text"
        elif suffix in HTML_SUFFIXES:
            body = _html_to_markdown(_read_text(p))
            kind = "html"
        elif suffix in CSV_SUFFIXES:
            delim = "\t" if suffix == ".tsv" else ","
            text = _read_text(p)
            rows = list(csv.reader(io.StringIO(text), delimiter=delim))
            body = f"# {p.stem}\n\n" + _rows_to_markdown(rows)
            kind = "table"
        elif suffix in DOCX_SUFFIXES:
            body, kind = _load_docx(p)
        elif suffix in XLSX_SUFFIXES:
            body, kind = _load_xlsx(p)
        elif suffix in PDF_SUFFIXES:
            body, kind = _load_pdf(p)
        else:
            return LoadResult.fail(str(p), f"不支持的扩展名 {suffix}")
    except _MissingDependency as e:
        return LoadResult.fail(str(p), str(e))
    except Exception as e:  # 解析异常不阻断整批入库
        return LoadResult.fail(str(p), f"{type(e).__name__}: {e}")

    body = normalize(body)
    if not body.strip():
        return LoadResult.fail(str(p), "解析后内容为空")

    doc = Document(
        # source 必须用**绝对路径**：增量入库靠它认同一篇文档。如果保留用户输入的原样
        # 路径（可能是相对的），`ingest ./data/corpus` 与 `ingest data/corpus` 就会被当成
        # 两篇文档，幂等性静默失效、content_hash 比对永远不命中。实测踩过。
        doc_id=derive_doc_id(str(p.resolve())),
        source=p.resolve().as_posix(),
        title=title or _first_heading(body) or p.stem,
        text=body,
        acl=tuple(acl) or ("public",),
        metadata={"kind": kind, "suffix": suffix, "bytes": p.stat().st_size, "name": p.name},
        content_hash=stable_hash(body),
    )
    return LoadResult(document=doc)


class _MissingDependency(RuntimeError):
    pass


def _import_optional(module: str, hint: str):
    try:
        return __import__(module)
    except ImportError as e:  # pragma: no cover - 取决于环境
        raise _MissingDependency(f"缺少可选依赖 `{module}`（{hint}）") from e


def _load_docx(p: Path) -> tuple[str, str]:
    docx = _import_optional("docx", "pip install python-docx")
    d = docx.Document(str(p))
    parts: list[str] = [f"# {p.stem}"]
    for para in d.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue
        style = (getattr(para.style, "name", "") or "").lower()
        m = re.search(r"heading (\d)", style)
        if m:
            parts.append("#" * (int(m.group(1)) + 1) + " " + text)
        elif style.startswith("list"):
            parts.append(f"- {text}")
        else:
            parts.append(text)
    for ti, table in enumerate(d.tables, start=1):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        md = _rows_to_markdown(rows)
        if md:
            parts.append(f"## 表格 {ti}\n\n{md}")
    return "\n\n".join(parts), "docx"


def _load_xlsx(p: Path) -> tuple[str, str]:
    openpyxl = _import_optional("openpyxl", "pip install openpyxl")
    wb = openpyxl.load_workbook(str(p), data_only=True, read_only=True)
    parts: list[str] = [f"# {p.stem}"]
    for ws in wb.worksheets:
        rows = [
            ["" if c is None else str(c) for c in row]
            for row in ws.iter_rows(values_only=True)
        ]
        md = _rows_to_markdown(rows)
        if md:
            parts.append(f"## {ws.title}\n\n{md}")
    wb.close()
    return "\n\n".join(parts), "xlsx"


def _load_pdf(p: Path) -> tuple[str, str]:
    pypdf = _import_optional("pypdf", "pip install pypdf")
    reader = pypdf.PdfReader(str(p))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"## 第 {i} 页\n\n{text}")
    return f"# {p.stem}\n\n" + "\n\n".join(pages), "pdf"


def iter_input_files(path: str | Path, recursive: bool = True) -> Iterator[Path]:
    p = Path(path)
    if p.is_file():
        yield p
        return
    pattern = "**/*" if recursive else "*"
    for child in sorted(p.glob(pattern)):
        if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
            yield child


@dataclass
class LoadBatch:
    documents: list[Document] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped_unsupported: int = 0

    def summary(self) -> str:
        return (
            f"loaded={len(self.documents)} errors={len(self.errors)} "
            f"unsupported={self.skipped_unsupported}"
        )


def load_path(
    path: str | Path, *, acl: tuple[str, ...] = (), recursive: bool = True
) -> LoadBatch:
    """读取文件或目录。"""
    batch = LoadBatch()
    for f in iter_input_files(path, recursive=recursive):
        res = load_file(f, acl=acl)
        if res.ok and res.document:
            batch.documents.append(res.document)
        else:
            batch.errors.append(res.error)
    return batch
