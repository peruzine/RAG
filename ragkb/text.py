"""中英文混合文本处理：归一化、分词、句子切分、token 估算。

为什么要自己写分词
------------------
中文没有空格，检索必须先解决切词。这里采用工业界常用的**字级二元组（bigram）**
方案：不引入 jieba 这类词典依赖，但对中文检索的效果接近分词，且对未登录词、
专业术语（"pgvector""混合检索"）非常稳。

约定
----
* ``index_terms``：建索引用，中文产出 **一元 + 二元**（保留重复，用于算词频）。
  保留一元是为了让"库"这种单字查询也能命中。
* ``query_terms``：查询用，中文**只产出二元**（长度 ≥2 的连续汉字串），
  因为一元噪声大、会稀释 BM25 的区分度；只有单字串才回退到一元。
* 两者共用同一份停用词表，保证词汇表一致。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# --------------------------------------------------------------------------
# 字符分类
# --------------------------------------------------------------------------
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),  # CJK 扩展 A
    (0x4E00, 0x9FFF),  # CJK 基本区
    (0xF900, 0xFAFF),  # 兼容表意文字
    (0x3040, 0x30FF),  # 日文假名（顺带支持）
    (0x20000, 0x2FA1F),  # CJK 扩展 B+
)

_WS_RE = re.compile(r"[ \t\u00a0\u3000]+")
_NL_RE = re.compile(r"\n{3,}")

# 停用词：只放真正的功能词，避免误伤领域词
STOPWORDS: frozenset[str] = frozenset(
    """
    的 了 是 在 和 与 及 或 我 你 他 她 它 们 这 那 有 无 不 也 就 都 而 并 为 以 于
    之 其 上 下 中 对 从 到 把 被 让 使 请 吗 呢 吧 啊 着 过 很 更 最 会 能 要 可
    什么 怎么 如何 哪些 哪个 哪里 是否 能否 可以 一下 一些 这些 那些 我们 你们 他们
    因为 所以 但是 如果 那么 请问 需要 进行 通过 以及 并且 或者 关于 对于 由于 根据
    the a an and or of to in on for is are was were be been with by as at from that this
    these those it its if then than so such can could should would will shall do does did
    """.split()
)

_CJK_PUNCT = set("。，、；：！？《》〈〉「」『』（）【】〔〕—…·～｜")


def normalize(text: str) -> str:
    """**展示用**归一化：统一换行、压缩空白。

    刻意不做全角转半角：NFKC 会把中文标点（，。：（））改成 ASCII，
    存进库里以后答案和引用里全是半角标点，中文阅读体验很差。
    检索侧的字符折叠交给 :func:`fold`。
    """
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = _WS_RE.sub(" ", t)
    t = _NL_RE.sub("\n\n", t)
    return t.strip()


def fold(text: str) -> str:
    """**检索用**归一化：NFKC（全角→半角、兼容字符统一）+ 空白压缩。

    索引与查询必须走同一个函数，否则会出现"全角数字建索引、半角数字查询不到"
    这类非常难查的问题。
    """
    return normalize(unicodedata.normalize("NFKC", text or ""))


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def _runs(text: str):
    """把文本切成连续同类片段：``('cjk', '知识库')`` / ``('word', 'rag')``。"""
    buf: list[str] = []
    kind = ""
    for ch in text:
        if is_cjk(ch):
            cur = "cjk"
        elif ch.isalnum():
            cur = "word"
        else:
            cur = ""
        if cur != kind:
            if buf:
                yield kind, "".join(buf)
            buf, kind = [], cur
        if cur:
            buf.append(ch)
    if buf:
        yield kind, "".join(buf)


def ngrams(seq: str, n: int) -> list[str]:
    if len(seq) < n:
        return []
    return [seq[i : i + n] for i in range(len(seq) - n + 1)]


def index_terms(text: str) -> list[str]:
    """建索引用的词条（含重复，供词频统计）。"""
    out: list[str] = []
    for kind, run in _runs(fold(text)):
        if kind == "cjk":
            out.extend(run)  # 一元
            out.extend(ngrams(run, 2))  # 二元
        elif kind == "word" and len(run) > 1:
            out.append(run.lower())
    return [t for t in out if t not in STOPWORDS]


def query_terms(text: str) -> list[str]:
    """查询用词条（去重、保序）。"""
    out: list[str] = []
    seen: set[str] = set()

    def push(token: str) -> None:
        if token in STOPWORDS or token in seen:
            return
        seen.add(token)
        out.append(token)

    for kind, run in _runs(fold(text)):
        if kind == "cjk":
            if len(run) >= 2:
                for g in ngrams(run, 2):
                    push(g)
            else:
                push(run)
        elif kind == "word" and len(run) > 1:
            push(run.lower())
    return out


def content_tokens(text: str) -> list[str]:
    """内容词条（一元+二元，去停用词，保留重复）——用于 F1 / 覆盖度计算。"""
    return index_terms(text)


def hashes_ngrams(text: str, n: int = 3) -> set[str]:
    """字符 n-gram 集合，用于近似去重（SimHash 的输入）。"""
    t = re.sub(r"\s+", "", fold(text))
    return set(ngrams(t, n)) or {t}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# 句子切分
# --------------------------------------------------------------------------
_SENT_END = "。！？!?；;"
_SENT_TAIL = "”』」》）】\"')"


def split_sentences(text: str, max_len: int = 0) -> list[str]:
    """按中英文标点切句；``max_len>0`` 时对超长句再按硬长度切分。"""
    sentences: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch == "\n":
            sentences.append("".join(buf))
            buf = []
        elif ch in _SENT_END:
            # 把随后紧跟的右引号/右括号并进同一句
            sentences.append("".join(buf))
            buf = []
        elif ch in _SENT_TAIL and buf and len(buf) > 1:
            continue
    if buf:
        sentences.append("".join(buf))

    out: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if max_len and len(s) > max_len:
            for i in range(0, len(s), max_len):
                out.append(s[i : i + max_len])
        else:
            out.append(s)
    return out


def tail_sentences(text: str, max_chars: int) -> str:
    """取结尾若干完整句子，总长不超过 ``max_chars``（用作 chunk overlap）。"""
    if max_chars <= 0 or not text:
        return ""
    picked: list[str] = []
    total = 0
    for sent in reversed(split_sentences(text)):
        if total + len(sent) > max_chars and picked:
            break
        picked.append(sent)
        total += len(sent)
        if total >= max_chars:
            break
    return "".join(reversed(picked))


def truncate(text: str, max_chars: int, suffix: str = "…") -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[: max_chars - len(suffix)]
    for sep in ("。", "！", "？", "\n", "，", "；"):
        idx = cut.rfind(sep)
        if idx > max_chars * 0.6:
            return cut[: idx + 1] + suffix
    return cut + suffix


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：CJK 约 1 token/字，其余约 3.6 字符/token。

    用于成本预估与上下文预算控制（真实用量仍以 API 返回为准）。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if is_cjk(ch))
    other = len(text) - cjk
    return int(cjk + other / 3.6) + 1


def highlight_terms(text: str, terms: list[str], max_terms: int = 6) -> str:
    """从段落里挑出包含最多查询词的句子，用作引用摘录。"""
    if not terms:
        return truncate(text, 160)
    scored: list[tuple[int, str]] = []
    for sent in split_sentences(text):
        if len(sent) < 6:
            continue
        low = sent.lower()
        hit = sum(1 for t in terms if t in low)
        if hit:
            scored.append((hit, sent))
    if not scored:
        return truncate(text, 160)
    scored.sort(key=lambda x: (-x[0], -len(x[1])))
    return truncate(scored[0][1], 200)


# --------------------------------------------------------------------------
# 文本相似度（评测用）
# --------------------------------------------------------------------------
# 折叠后中文标点已变成 ASCII，所以用"非内容字符"正则统一剔除标点与空白
_NON_CONTENT_RE = re.compile(r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+")


def content_chars(text: str) -> str:
    """只保留字母/数字/汉字，去掉所有标点空白（用于答案比对）。"""
    return _NON_CONTENT_RE.sub("", fold(text))


def char_f1(pred: str, gold: str) -> float:
    """字符级 F1，对中文答案比对很实用（类似 SQuAD 的 token F1）。"""
    p = list(content_chars(pred))
    g = list(content_chars(gold))
    if not p or not g:
        return 0.0
    from collections import Counter

    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def containment(needle: str, haystack: str) -> bool:
    """子串包含判断（先折叠、去标点与空白）。用于评测里匹配 gold 片段。"""
    n, h = content_chars(needle), content_chars(haystack)
    return bool(n) and n in h
