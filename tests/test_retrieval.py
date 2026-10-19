"""检索层测试：RRF 融合、混合召回、重排、多子查询、权限过滤。"""

from __future__ import annotations

import unittest

from tests.helpers import SECRET_DOC, AppTestCase, build_app
from tests.helpers import ROOT  # noqa: F401

from ragkb.rerank import LexicalReranker, NoopReranker, build_reranker
from ragkb.retrieve import CH_LEXICAL, CH_VECTOR, HybridRetriever
from ragkb.types import Chunk, Retrieved


class TestRRF(AppTestCase):
    def test_single_channel_ranking(self):
        retr: HybridRetriever = self.app.retriever
        fused, ranks = retr.fuse_rrf({CH_LEXICAL: [("a", 9.0), ("b", 5.0)]}, top_k=10, k=60)
        self.assertEqual([cid for cid, _ in fused], ["a", "b"])
        self.assertAlmostEqual(fused[0][1], 1 / 61)
        self.assertEqual(ranks["a"], {CH_LEXICAL: 1})

    def test_two_channels_reward_agreement(self):
        retr = self.app.retriever
        fused, ranks = retr.fuse_rrf(
            {CH_LEXICAL: [("a", 1.0), ("b", 1.0)], CH_VECTOR: [("b", 1.0), ("a", 1.0)]},
            top_k=10,
            k=60,
        )
        # a 与 b 各在一个通道排 1、另一个排 2 → 分数相同，按 id 稳定排序
        self.assertEqual([cid for cid, _ in fused], ["a", "b"])
        self.assertAlmostEqual(fused[0][1], 1 / 61 + 1 / 62)
        self.assertEqual(ranks["b"], {CH_LEXICAL: 2, CH_VECTOR: 1})

    def test_per_channel_truncation_limits_long_tail(self):
        retr = self.app.retriever
        retr.cfg.fuse_top_k = 2
        long_channel = [(f"c{i}", 1.0) for i in range(10)]
        fused, _ = retr.fuse_rrf({CH_LEXICAL: long_channel}, top_k=10, k=60)
        self.assertEqual(len(fused), 2)
        retr.cfg.fuse_top_k = 20


class TestHybridRetrieve(AppTestCase):
    def test_returns_relevant_chunk_first(self):
        res = self.app.retriever.retrieve("一线城市住宿每晚上限是多少？", top_k=3)
        self.assertTrue(res.items)
        self.assertIn("700", res.items[0].chunk.text)

    def test_debug_records_channels_and_timings(self):
        res = self.app.retriever.retrieve("P0 故障响应时间", top_k=3)
        d = res.debug.to_dict()
        self.assertGreaterEqual(d["lexical_hits"], 1)
        self.assertIn("lexical", d["timings_ms"])
        self.assertIn("rerank", d["timings_ms"])
        self.assertGreater(d["best_score"], 0)
        # 每个结果都能说清"被哪些通道召回、排第几"
        self.assertTrue(all(it.channels for it in res.items))

    def test_absent_topic_has_no_lexical_hits(self):
        res = self.app.retriever.retrieve("量子纠缠发动机的维护手册", top_k=3, use_rerank=False)
        # 词法通道必须严格为零；向量通道用的本地词法兜底向量会给低分噪声，
        # 这正是要靠覆盖率门禁兜住的部分（见 test_rag 的拒答用例）。
        self.assertEqual(res.debug.lexical_hits, 0)
        self.assertLess(res.best_score, 0.15)
        self.assertTrue(self.app.ask("量子纠缠发动机的维护手册").refused)

    def test_acl_hides_internal_documents(self):
        path = self.cfg.corpus_dir / "secret.md"
        path.write_text(SECRET_DOC, encoding="utf-8")
        self.app.ingest(path, acl=("internal",))
        public = self.app.retriever.retrieve("P6 职级带宽上限", roles=("public",), top_k=5)
        self.assertNotIn("84.0", " ".join(it.chunk.text for it in public.items))
        internal = self.app.retriever.retrieve("P6 职级带宽上限", roles=("public", "internal"), top_k=5)
        self.assertIn("84.0", " ".join(it.chunk.text for it in internal.items))
        admin = self.app.retriever.retrieve("P6 职级带宽上限", roles=None, top_k=5)
        self.assertIn("84.0", " ".join(it.chunk.text for it in admin.items))

    def test_doc_ids_filter(self):
        docs = self.app.store.list_documents()
        target = docs[0]["doc_id"]
        res = self.app.retriever.retrieve("标准", doc_ids=[target], top_k=5)
        self.assertTrue(all(it.chunk.doc_id == target for it in res.items))

    def test_retrieve_multi_merges_subqueries(self):
        res = self.app.retriever.retrieve_multi(
            ["住宿上限是多少？", "P0 故障响应时间"], top_k=5
        )
        joined = " ".join(it.chunk.text for it in res.items)
        self.assertIn("700", joined)
        self.assertIn("P0", joined)
        self.assertEqual(len(res.debug.sub_queries), 2)

    def test_vector_channel_failure_degrades_to_lexical(self):
        class Boom:
            name = "boom"

            def embed_query(self, text):
                raise RuntimeError("模拟向量服务故障")

            def embed(self, texts):
                raise RuntimeError("模拟向量服务故障")

        original = self.app.retriever.embedder
        self.app.retriever.embedder = Boom()  # type: ignore[assignment]
        try:
            res = self.app.retriever.retrieve("住宿上限", top_k=3)
            self.assertTrue(res.items)  # BM25 兜住了
            self.assertEqual(res.debug.vector_hits, 0)
        finally:
            self.app.retriever.embedder = original  # type: ignore[assignment]


class TestRerankers(unittest.TestCase):
    def _items(self) -> list[Retrieved]:
        def chunk(cid, text, heading=""):
            return Chunk(chunk_id=cid, doc_id="d", ordinal=0, text=text, heading_path=heading)

        return [
            Retrieved(chunk=chunk("c1", "无关的一句话，讲的是别的事情。"), score=0.9),
            Retrieved(chunk=chunk("c2", "一线城市住宿每晚上限 700 元。"), score=0.5),
        ]

    def test_noop_keeps_fused_order(self):
        out = NoopReranker().rerank("住宿上限", self._items(), 10)
        self.assertEqual([r.chunk_id for r in out], ["c1", "c2"])

    def test_lexical_rerank_promotes_relevant_chunk(self):
        out = LexicalReranker().rerank("住宿上限", self._items(), 10)
        self.assertEqual(out[0].chunk_id, "c2")
        self.assertIsNotNone(out[0].rerank_score)
        self.assertIn("lexical_rerank", out[0].channel_scores)

    def test_rerank_never_drops_items(self):
        out = LexicalReranker().rerank("住宿", self._items(), 10)
        self.assertEqual(len(out), 2)

    def test_heading_hit_gets_bonus(self):
        items = [
            Retrieved(chunk=Chunk("a", "d", 0, "正文提到住宿一次。", heading_path=""), score=0.5),
            Retrieved(chunk=Chunk("b", "d", 0, "正文也提到住宿一次。", heading_path="住宿标准"), score=0.5),
        ]
        out = LexicalReranker().rerank("住宿", items, 10)
        self.assertEqual(out[0].chunk_id, "b")

    def test_empty_input(self):
        self.assertEqual(LexicalReranker().rerank("x", [], 5), [])

    def test_factory_switches_provider(self):
        from tests.helpers import make_settings

        cfg = make_settings("rerank-factory")
        cfg.rerank_provider = "none"
        self.assertIsInstance(build_reranker(cfg), NoopReranker)
        cfg.rerank_provider = "lexical"
        self.assertIsInstance(build_reranker(cfg), LexicalReranker)


if __name__ == "__main__":
    unittest.main()
