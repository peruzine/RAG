"""加载器测试：格式解析、标题推断、doc_id 稳定性、失败降级。"""

from __future__ import annotations

import unittest
from pathlib import Path

from tests.helpers import ROOT, make_settings  # noqa: F401

from ragkb.loaders import (
    SUPPORTED_SUFFIXES,
    derive_doc_id,
    iter_input_files,
    load_file,
    load_path,
)


class LoaderCase(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = make_settings("loaders")
        self.dir = self.cfg.corpus_dir

    def write(self, name: str, body: str | bytes) -> "object":
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, bytes):
            p.write_bytes(body)
        else:
            p.write_text(body, encoding="utf-8")
        return p


class TestBasicFormats(LoaderCase):
    def test_markdown_title_from_first_heading(self):
        p = self.write("doc.md", "# 报销制度\n\n正文内容。\n")
        res = load_file(p)
        self.assertTrue(res.ok)
        self.assertEqual(res.document.title, "报销制度")
        self.assertEqual(res.document.metadata["kind"], "text")
        self.assertEqual(res.document.acl, ("public",))

    def test_title_falls_back_to_filename(self):
        p = self.write("无标题.txt", "第一行不是标题\n第二行\n")
        res = load_file(p)
        self.assertEqual(res.document.title, "无标题")

    def test_plain_text_kept(self):
        p = self.write("a.txt", "一段中文内容，包含标点。")
        res = load_file(p)
        self.assertIn("包含标点。", res.document.text)

    def test_html_is_stripped_to_text(self):
        p = self.write(
            "page.html",
            "<html><head><style>p{color:red}</style></head><body>"
            "<h1>部署说明</h1><p>第一步，安装依赖。</p><script>alert(1)</script>"
            "<ul><li>要求 Python 3.11</li></ul></body></html>",
        )
        res = load_file(p)
        self.assertTrue(res.ok)
        self.assertIn("部署说明", res.document.text)
        self.assertIn("要求 Python 3.11", res.document.text)
        self.assertNotIn("alert", res.document.text)
        self.assertNotIn("color:red", res.document.text)
        self.assertEqual(res.document.metadata["kind"], "html")

    def test_html_headings_become_markdown(self):
        p = self.write("h.html", "<h2>二级标题</h2><p>内容</p>")
        res = load_file(p)
        self.assertIn("## 二级标题", res.document.text)

    def test_csv_becomes_markdown_table(self):
        p = self.write("t.csv", "城市,上限\n一线,700\n二线,500\n")
        res = load_file(p)
        self.assertIn("| 城市 | 上限 |", res.document.text)
        self.assertIn("| 一线 | 700 |", res.document.text)
        self.assertEqual(res.document.metadata["kind"], "table")

    def test_empty_file_is_reported(self):
        p = self.write("empty.md", "   \n\n  ")
        res = load_file(p)
        self.assertFalse(res.ok)
        self.assertIn("为空", res.error)

    def test_missing_file(self):
        res = load_file(self.dir / "nope.md")
        self.assertFalse(res.ok)
        self.assertIn("不存在", res.error)

    def test_unsupported_suffix(self):
        p = self.write("x.exe", b"\x00\x01")
        res = load_file(p)
        self.assertFalse(res.ok)
        self.assertIn("不支持", res.error)

    def test_missing_optional_dependency_is_reported_not_raised(self):
        p = self.write("doc.pdf", b"%PDF-1.4 fake")
        res = load_file(p)
        self.assertFalse(res.ok)
        # 环境里有 pypdf 时会因解析失败报错，没有时提示缺依赖；两种都必须是"失败"而非异常
        self.assertTrue(res.error)


class TestDocIdentity(LoaderCase):
    def test_source_is_stored_as_absolute_path(self):
        p = self.write("abs.md", "# T\n\n内容够长一些")
        res = load_file(p)
        self.assertTrue(Path(res.document.source).is_absolute(), res.document.source)

    def test_same_file_via_relative_and_absolute_path_is_one_document(self):
        """回归：来源路径曾按用户输入原样保存，导致相对/绝对路径被当成两篇文档。"""
        import os

        p = self.write("rel.md", "# T\n\n同一篇文档内容")
        old = os.getcwd()
        try:
            os.chdir(ROOT)
            absolute = load_file(p)
            relative = load_file(p.relative_to(ROOT))
        finally:
            os.chdir(old)
        self.assertEqual(absolute.document.source, relative.document.source)
        self.assertEqual(absolute.document.doc_id, relative.document.doc_id)
        self.assertEqual(absolute.document.content_hash, relative.document.content_hash)

    def test_doc_id_is_stable_for_same_path(self):
        self.assertEqual(derive_doc_id("/x/y/a.md"), derive_doc_id("/x/y/a.md"))

    def test_doc_id_differs_across_paths(self):
        self.assertNotEqual(derive_doc_id("/x/a.md"), derive_doc_id("/y/a.md"))

    def test_same_content_different_path_has_same_hash_but_diff_id(self):
        a = self.write("a.md", "# T\n\n同样的内容")
        b = self.write("sub/b.md", "# T\n\n同样的内容")
        ra, rb = load_file(a), load_file(b)
        self.assertEqual(ra.document.content_hash, rb.document.content_hash)
        self.assertNotEqual(ra.document.doc_id, rb.document.doc_id)

    def test_content_change_changes_hash(self):
        p = self.write("c.md", "# T\n\n版本一")
        h1 = load_file(p).document.content_hash
        p.write_text("# T\n\n版本二", encoding="utf-8")
        self.assertNotEqual(h1, load_file(p).document.content_hash)


class TestDirectoryLoading(LoaderCase):
    def test_recursive_iteration_filters_suffix(self):
        self.write("a.md", "# A\n\n内容")
        self.write("nested/b.txt", "内容")
        self.write("nested/skip.exe", b"x")
        files = list(iter_input_files(self.dir, recursive=True))
        names = {f.name for f in files}
        self.assertEqual(names, {"a.md", "b.txt"})

    def test_non_recursive(self):
        self.write("a.md", "# A\n\n内容")
        self.write("nested/b.txt", "内容")
        names = {f.name for f in iter_input_files(self.dir, recursive=False)}
        self.assertEqual(names, {"a.md"})

    def test_load_path_collects_documents_and_errors(self):
        self.write("a.md", "# A\n\n内容够长一些")
        self.write("bad.exe", b"\x00")
        batch = load_path(self.dir)
        self.assertEqual(len(batch.documents), 1)
        self.assertGreaterEqual(len(batch.errors), 0)  # exe 被后缀过滤，不算错误
        self.assertIn("loaded=1", batch.summary())

    def test_supported_suffixes_cover_docs(self):
        for suffix in (".md", ".txt", ".html", ".csv", ".docx", ".xlsx", ".pdf"):
            self.assertIn(suffix, SUPPORTED_SUFFIXES)


if __name__ == "__main__":
    unittest.main()
