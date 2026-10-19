"""切块测试：结构识别、表格/代码块原子性、overlap、碎块合并。"""

from __future__ import annotations

import unittest

from tests.helpers import ROOT  # noqa: F401

from ragkb.chunking import chunk_document, pack_blocks, parse_blocks
from ragkb.config import Settings
from ragkb.types import Document


def cfg(**kw) -> Settings:
    base = Settings.load(env={})
    base.chunk_size = kw.pop("chunk_size", 300)
    base.chunk_overlap = kw.pop("chunk_overlap", 60)
    base.chunk_min_chars = kw.pop("chunk_min_chars", 20)
    for k, v in kw.items():
        setattr(base, k, v)
    return base


class TestParseBlocks(unittest.TestCase):
    def test_heading_path_is_tracked(self):
        text = "# 一级\n\n内容A\n\n## 二级\n\n内容B\n"
        blocks = parse_blocks(text)
        self.assertEqual(blocks[0].heading_path, "一级")
        self.assertEqual(blocks[1].heading_path, "一级 > 二级")

    def test_table_stays_one_block(self):
        text = "# T\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].kind, "table")
        self.assertIn("| 1 | 2 |", blocks[0].text)

    def test_fenced_code_is_atomic(self):
        text = "# C\n\n```python\nprint(1)\n\nprint(2)\n```\n"
        blocks = parse_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].kind, "code")
        self.assertIn("print(2)", blocks[0].text)

    def test_list_items_grouped(self):
        text = "# L\n\n- 甲\n- 乙\n- 丙\n"
        blocks = parse_blocks(text)
        self.assertEqual(blocks[0].kind, "list")

    def test_blank_lines_separate_paragraphs(self):
        blocks = parse_blocks("# H\n\n第一段\n\n第二段\n")
        self.assertEqual(len(blocks), 2)


class TestPackBlocks(unittest.TestCase):
    def test_chunks_respect_size(self):
        text = "# H\n\n" + "\n\n".join(f"这是第{i}段内容。" * 8 for i in range(12))
        doc = Document(doc_id="d1", source="s.md", title="T", text=text)
        chunks, report = chunk_document(doc, cfg(chunk_size=300))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.char_len <= 300 * 1.6 for c in chunks))
        self.assertEqual(report.n_chunks, len(chunks))

    def test_heading_change_splits_when_half_full(self):
        long_a = "甲" * 200
        text = f"# A\n\n{long_a}\n\n# B\n\n乙" * 1
        chunks, _ = chunk_document(
            Document(doc_id="d", source="s", title="t", text=text), cfg(chunk_size=300, chunk_min_chars=0)
        )
        paths = {c.heading_path for c in chunks}
        self.assertIn("A", paths)
        self.assertIn("B", paths)

    def test_chunk_ids_and_doc_ids_are_backfilled(self):
        doc = Document(doc_id="doc-abc", source="s.md", title="T", text="# H\n\n" + "内容。" * 100)
        chunks, _ = chunk_document(doc, cfg(chunk_size=200))
        self.assertTrue(all(c.doc_id == "doc-abc" for c in chunks))
        self.assertTrue(all(c.chunk_id.startswith("doc-abc#") for c in chunks))
        self.assertEqual([c.ordinal for c in chunks], list(range(len(chunks))))

    def test_tiny_tail_chunk_gets_merged(self):
        text = "# H\n\n" + "甲" * 250 + "\n\n短"
        chunks, _ = chunk_document(
            Document(doc_id="d", source="s", title="t", text=text),
            cfg(chunk_size=300, chunk_overlap=0, chunk_min_chars=50),
        )
        self.assertTrue(all(c.char_len >= 50 for c in chunks))

    def test_metadata_carries_source_title_acl(self):
        doc = Document(
            doc_id="d2", source="a/b.md", title="标题", text="# H\n\n正文内容够长" * 5, acl=("internal",)
        )
        chunks, _ = chunk_document(doc, cfg())
        self.assertEqual(chunks[0].metadata["acl"], ["internal"])
        self.assertEqual(chunks[0].metadata["source"], "a/b.md")

    def test_embedding_text_includes_heading(self):
        doc = Document(doc_id="d", source="s", title="手册", text="# 安装 > X\n\n内容" * 20)
        chunks, _ = chunk_document(doc, cfg())
        self.assertIn("手册", chunks[0].embedding_text("手册"))

    def test_empty_document_yields_no_chunks(self):
        chunks, report = chunk_document(Document(doc_id="d", source="s", title="t", text="   \n  "), cfg())
        self.assertEqual(chunks, [])
        self.assertEqual(report.n_chunks, 0)


class TestOversizedBlock(unittest.TestCase):
    def test_oversized_paragraph_is_split_with_overlap(self):
        body = "。".join(f"第{i}句内容" for i in range(200)) + "。"
        chunks, _ = chunk_document(
            Document(doc_id="d", source="s", title="t", text="# H\n\n" + body),
            cfg(chunk_size=200, chunk_overlap=40, chunk_min_chars=0),
        )
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(c.char_len <= 200 * 1.6 for c in chunks))
        # overlap 生效：相邻块应有共同文本（取上一块结尾）
        self.assertTrue(any(chunks[i].text[-10:] in chunks[i + 1].text for i in range(len(chunks) - 1)))


if __name__ == "__main__":
    unittest.main()
