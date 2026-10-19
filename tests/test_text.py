"""文本层测试：分词/二元切分/停用词/折叠/相似度。"""

from __future__ import annotations

import unittest

from tests.helpers import ROOT  # noqa: F401  (确保 sys.path 已就绪)

from ragkb import text as T


class TestTokenizer(unittest.TestCase):
    def test_cjk_bigrams_for_query(self):
        self.assertEqual(T.query_terms("知识库"), ["知识", "识库"])

    def test_query_terms_dedupes_and_keeps_order(self):
        terms = T.query_terms("试用期试用期")
        self.assertEqual(terms, ["试用", "用期", "期试"])

    def test_single_char_query_falls_back_to_unigram(self):
        self.assertEqual(T.query_terms("库"), ["库"])

    def test_latin_words_are_kept_and_lowercased(self):
        self.assertIn("ragkb", T.query_terms("RAGKB 怎么部署"))

    def test_index_terms_include_unigrams_and_duplicates(self):
        terms = T.index_terms("知识库知识库")
        self.assertIn("知", terms)  # 一元：为了单字查询也能命中
        self.assertIn("知识", terms)
        self.assertGreater(terms.count("知识"), 1)  # 保留重复 → 可算词频

    def test_stopwords_are_removed_consistently(self):
        self.assertNotIn("的", T.index_terms("我的文档"))
        self.assertNotIn("什么", T.query_terms("什么是 SLA"))

    def test_punctuation_separates_runs(self):
        self.assertEqual(T.query_terms("住宿，标准"), ["住宿", "标准"])


class TestNormalization(unittest.TestCase):
    def test_fold_converts_fullwidth_digits(self):
        self.assertEqual(T.fold("１２３"), "123")

    def test_display_normalize_keeps_chinese_punctuation(self):
        # 存库文本必须保留中文标点，否则答案和引用里全是半角逗号
        self.assertEqual(T.normalize("住宿标准，含税。"), "住宿标准，含税。")

    def test_fold_is_stable_for_matching(self):
        # 全角/半角写法不同，但折叠后必须能互相匹配
        self.assertEqual(T.content_chars("（北上广深）"), T.content_chars("(北上广深)"))

    def test_whitespace_collapsed(self):
        self.assertEqual(T.normalize("a   \t b\n\n\n\nc"), "a b\n\nc")


class TestSentences(unittest.TestCase):
    def test_split_on_chinese_punctuation(self):
        sents = T.split_sentences("第一句。第二句！第三句？")
        self.assertEqual(len(sents), 3)

    def test_tail_sentences_respects_budget(self):
        tail = T.tail_sentences("甲甲甲甲甲。乙乙乙乙乙。丙丙丙丙丙。", 12)
        self.assertTrue(tail.endswith("丙丙丙丙丙。"))
        self.assertLessEqual(len(tail), 12)

    def test_truncate_prefers_sentence_boundary(self):
        out = T.truncate("这是一个很长很长的句子。后面还有很多内容", 14)
        self.assertTrue(out.endswith("…"))


class TestSimilarity(unittest.TestCase):
    def test_char_f1_identical(self):
        self.assertAlmostEqual(T.char_f1("出差餐补 120 元", "出差餐补120元"), 1.0)

    def test_char_f1_disjoint(self):
        self.assertEqual(T.char_f1("完全无关", "另一个东西"), 0.0)

    def test_containment_ignores_punctuation_and_width(self):
        self.assertTrue(T.containment("含税）", "住宿上限 700 元（含税）"))

    def test_estimate_tokens_counts_cjk_per_char(self):
        self.assertGreater(T.estimate_tokens("中文字符测试"), 5)
        self.assertLess(T.estimate_tokens("hello world"), T.estimate_tokens("你好世界你好世界"))

    def test_highlight_picks_sentence_with_most_query_terms(self):
        body = "无关的一句话。住宿上限是 700 元。另一句无关的话。"
        picked = T.highlight_terms(body, T.query_terms("住宿上限"))
        self.assertIn("700", picked)


if __name__ == "__main__":
    unittest.main()
