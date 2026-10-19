"""存储层测试：读写、幂等版本、BM25、向量、ACL、删除、统计。"""

from __future__ import annotations

import unittest

from tests.helpers import ROOT  # noqa: F401

from ragkb.store import SqliteStore, pack_vector, unpack_vector
from ragkb.types import Chunk, Document


def make_chunks(doc_id: str, texts: list[str], acl=("public",), vec=None) -> list[Chunk]:
    out = []
    for i, text in enumerate(texts):
        out.append(
            Chunk(
                chunk_id=f"{doc_id}#{i:04d}",
                doc_id=doc_id,
                ordinal=i,
                text=text,
                heading_path="章节",
                token_count=len(text) // 2,
                metadata={"acl": list(acl), "source": f"{doc_id}.md", "title": doc_id},
                embedding=vec[i] if vec else None,
                embed_model="test-embed" if vec else "",
            )
        )
    return out


class StoreCase(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SqliteStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()


class TestDocuments(StoreCase):
    def _doc(self, doc_id="doc-1", text="差旅住宿标准", h="h1") -> Document:
        return Document(doc_id=doc_id, source=f"{doc_id}.md", title=doc_id, text=text, content_hash=h)

    def test_upsert_and_get(self):
        doc = self._doc()
        chunks = make_chunks(doc.doc_id, ["一线城市每晚上限 700 元。", "餐补 120 元每天。"])
        stats = self.store.upsert_document(doc, chunks)
        self.assertEqual(stats["chunks"], 2)
        got = self.store.get_document("doc-1")
        self.assertIsNotNone(got)
        self.assertEqual(got.n_chunks, 2)
        self.assertEqual(got.version, 1)

    def test_reingest_bumps_version_and_replaces_chunks(self):
        doc = self._doc()
        self.store.upsert_document(doc, make_chunks("doc-1", ["旧内容一。", "旧内容二。"]))
        doc2 = self._doc(h="h2")
        self.store.upsert_document(doc2, make_chunks("doc-1", ["新内容只有一条。"]))
        self.assertEqual(self.store.get_document("doc-1").version, 2)
        self.assertEqual(self.store.count_chunks(), 1)
        self.assertIn("新内容", next(iter(self.store.iter_chunks())).text)

    def test_delete_removes_chunks_postings_and_acl(self):
        doc = self._doc()
        self.store.upsert_document(doc, make_chunks("doc-1", ["独一无二的术语 阿尔法。"]))
        self.assertTrue(self.store.delete_document("doc-1"))
        self.assertEqual(self.store.count_chunks(), 0)
        self.assertEqual(self.store.count_documents(), 0)
        self.assertEqual(self.store.lexical_search("阿尔法"), [])
        self.assertFalse(self.store.delete_document("doc-1"))

    def test_list_documents_reports_acl_and_hash(self):
        self.store.upsert_document(self._doc(), make_chunks("doc-1", ["内容"], acl=("internal",)))
        docs = self.store.list_documents()
        self.assertEqual(docs[0]["acl"], ["internal"])
        self.assertEqual(docs[0]["content_hash"], "h1")


class TestLexical(StoreCase):
    def setUp(self) -> None:
        super().setUp()
        self.store.upsert_document(
            Document(doc_id="a", source="a.md", title="a", text="x"),
            make_chunks("a", ["一线城市住宿每晚上限 700 元。", "出差餐补为 120 元每天。"]),
        )
        self.store.upsert_document(
            Document(doc_id="b", source="b.md", title="b", text="y"),
            make_chunks("b", ["P0 故障首次响应 15 分钟。"]),
        )

    def test_finds_chunk_by_chinese_term(self):
        hits = self.store.lexical_search("住宿上限")
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], "a#0000")

    def test_absent_term_returns_nothing(self):
        self.assertEqual(self.store.lexical_search("量子纠缠发动机"), [])

    def test_scores_are_positive_and_sorted(self):
        hits = self.store.lexical_search("餐补 出差")
        scores = [s for _, s in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(s > 0 for s in scores))

    def test_acl_filters_before_scoring(self):
        self.store.upsert_document(
            Document(doc_id="s", source="s.md", title="s", text="z"),
            make_chunks("s", ["薪酬带宽上限 84 万元。"], acl=("internal",)),
        )
        self.assertEqual(self.store.lexical_search("薪酬带宽", roles=("public",)), [])
        self.assertTrue(self.store.lexical_search("薪酬带宽", roles=("public", "internal")))
        self.assertTrue(self.store.lexical_search("薪酬带宽", roles=None))

    def test_prune_removes_high_frequency_terms(self):
        # 单独建库：4 个 chunk 都含"内容"，1 个不含 → df=4/5，超过 0.5 阈值
        store = SqliteStore(":memory:")
        try:
            store.upsert_document(
                Document(doc_id="c", source="c.md", title="c", text="q"),
                make_chunks("c", ["内容 一", "内容 二", "内容 三", "内容 四"]),
            )
            store.upsert_document(
                Document(doc_id="d", source="d.md", title="d", text="w"),
                make_chunks("d", ["别的词。"]),
            )
            self.assertEqual(store.term_df(["内容"])["内容"], 4)
            removed = store.prune_postings(max_df_ratio=0.5)
            self.assertGreater(removed, 0)
            self.assertEqual(store.lexical_search("内容"), [])
        finally:
            store.close()

    def test_term_df_and_idf(self):
        df = self.store.term_df(["住宿", "不存在词"])
        self.assertGreaterEqual(df["住宿"], 1)
        self.assertEqual(df["不存在词"], 0)
        idf = self.store.term_idf(["住宿", "不存在词"])
        self.assertGreater(idf["不存在词"], idf["住宿"])


class TestVector(StoreCase):
    def setUp(self) -> None:
        super().setUp()
        vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        self.store.upsert_document(
            Document(doc_id="v", source="v.md", title="v", text="v"),
            make_chunks("v", ["甲文档内容。", "乙文档内容。"], vec=vecs),
        )

    def test_cosine_search_returns_nearest(self):
        hits = self.store.vector_search([0.9, 0.1, 0.0], top_k=2, model="test-embed")
        self.assertEqual(hits[0][0], "v#0000")
        self.assertGreater(hits[0][1], hits[1][1])

    def test_model_filter_prevents_cross_model_comparison(self):
        self.assertEqual(self.store.vector_search([1.0, 0.0, 0.0], model="other-model"), [])

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.store.vector_search([1.0, 0.0], model="test-embed")

    def test_min_similarity_cuts_noise(self):
        hits = self.store.vector_search([0.0, 1.0, 0.0], min_similarity=0.5, model="test-embed")
        self.assertEqual(len(hits), 1)

    def test_pack_unpack_roundtrip(self):
        self.assertAlmostEqual(unpack_vector(pack_vector([1.5, -2.25]))[1], -2.25, places=5)

    def test_acl_applies_to_vector_search(self):
        self.store.upsert_document(
            Document(doc_id="w", source="w.md", title="w", text="w"),
            make_chunks("w", ["机密内容。"], acl=("internal",), vec=[[0.0, 0.0, 1.0]]),
        )
        # v 的两个 chunk 是 public，应当可见；internal 的 w 必须被挡掉
        hits = self.store.vector_search([0.0, 0.0, 1.0], roles=("public",), model="test-embed")
        ids = [h[0] for h in hits]
        self.assertNotIn("w#0000", ids)
        self.assertIn("v#0000", ids)
        hits2 = self.store.vector_search(
            [0.0, 0.0, 1.0], roles=("public", "internal"), model="test-embed"
        )
        self.assertEqual(hits2[0][0], "w#0000")


class TestCrossProcessCache(unittest.TestCase):
    """模拟"CLI 入库 + 常驻服务读库"两个进程并存：服务侧的缓存必须自动失效。

    这是实测踩过的坑：CLI 把一篇文档收紧成 internal，服务进程因为内存里缓存了旧的
    可见集合，仍把它当 public 检索出来——权限变更等于没生效。
    """

    def test_acl_change_from_another_instance_is_picked_up(self):
        from tests.helpers import make_settings

        cfg = make_settings("xproc")
        service = SqliteStore(cfg.db_path)  # 常驻服务进程
        cli = SqliteStore(cfg.db_path)  # 命令行进程
        try:
            service.upsert_document(
                Document(
                    doc_id="d1", source="d1.md", title="T", text="x",
                    acl=("public",), content_hash="h1",
                ),
                make_chunks("d1", ["薪酬带宽上限 84 万元。"]),
            )
            # 服务侧先查一次，把 public 可见集合缓存进内存
            self.assertIn("d1#0000", service.allowed_chunk_ids(("public",)))

            # 另一个进程收紧权限
            cli.update_document_acl("d1", ["internal"])

            # 服务侧必须立刻看到变更
            self.assertNotIn("d1#0000", service.allowed_chunk_ids(("public",)))
            self.assertIn("d1#0000", service.allowed_chunk_ids(("public", "internal")))
        finally:
            service.close()
            cli.close()

    def test_new_document_from_another_instance_is_visible(self):
        from tests.helpers import make_settings

        cfg = make_settings("xproc2")
        service = SqliteStore(cfg.db_path)
        cli = SqliteStore(cfg.db_path)
        try:
            self.assertEqual(service.lexical_search("阿尔法"), [])
            cli.upsert_document(
                Document(doc_id="d2", source="d2.md", title="T", text="y"),
                make_chunks("d2", ["阿尔法专用流程。"]),
            )
            self.assertTrue(service.lexical_search("阿尔法"))
        finally:
            service.close()
            cli.close()


class TestStats(StoreCase):
    def test_stats_shape(self):
        self.store.upsert_document(
            Document(doc_id="a", source="a.md", title="a", text="x"),
            make_chunks("a", ["内容甲。"], vec=[[1.0, 0.0]]),
        )
        s = self.store.stats()
        for key in ("documents", "chunks", "terms", "postings", "avg_chunk_chars", "embedding_models"):
            self.assertIn(key, s)
        self.assertEqual(s["embedding_models"][0]["model"], "test-embed")
        self.assertEqual(s["acl_roles"], ["public"])

    def test_clear_empties_everything(self):
        self.store.upsert_document(
            Document(doc_id="a", source="a.md", title="a", text="x"), make_chunks("a", ["内容甲。"])
        )
        self.store.clear()
        self.assertEqual(self.store.count_chunks(), 0)
        self.assertEqual(self.store.lexical_search("内容"), [])


if __name__ == "__main__":
    unittest.main()
