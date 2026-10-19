"""查询理解：多轮改写（condense）、路由（KB / CHAT）、子问题拆解。

工程上最容易被忽略、又最影响体验的一环。这里刻意做了**成本控制**：
改写只在"确实需要"时才调用 LLM（有历史 + 出现指代/省略），否则直接用原问题，
省掉一次模型调用（线上 QPS 高时这一步能省 30%+ 成本）。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Sequence

from .config import Settings
from .llm import ChatModel
from .prompts import NO_ANSWER_SENTINEL, build_condense_messages, build_router_messages
from .text import estimate_tokens, query_terms

if TYPE_CHECKING:  # 只用于类型标注，避免循环导入
    from .store import SqliteStore

# 出现这些词说明这句话不自足，需要结合历史改写
_DEPENDENT_MARKERS = (
    "它", "他", "她", "这个", "那个", "这些", "那些", "该", "此", "上述", "前面", "刚才",
    "那么", "还有", "继续", "再多", "再详细", "为什么", "然后呢", "it", "this", "that", "they",
)
_GREETING_RE = re.compile(r"^\s*(你好|您好|hi|hello|嗨|在吗|谢谢|thanks|thank you|你是谁|你叫什么)[\s!！。.~]*$", re.I)

_KB_HINTS = (
    "制度", "流程", "规范", "文档", "手册", "怎么", "如何", "配置", "安装", "部署", "接口", "api",
    "报错", "故障", "政策", "报销", "请假", "权限", "报价", "价格", "产品", "版本", "sla", "支持",
    "参数", "步骤", "要求", "条件", "标准", "术语", "是指", "什么是", "定义",
)
_CHAT_HINTS = ("讲个笑话", "写首诗", "你是谁", "今天天气", "翻译一下这段话")


class QueryProcessor:
    def __init__(self, llm: ChatModel, cfg: Settings, store: "SqliteStore | None" = None) -> None:
        self.llm = llm
        self.cfg = cfg
        self.store = store
        self.last_condense_cost = 0
        self.last_condense_mode = "none"

    # ------------------------------------------------------------------
    def needs_condense(self, question: str, history: list[dict[str, str]]) -> bool:
        """规则判断是否需要改写：省一次模型调用。"""
        if not history:
            return False
        q = question.strip()
        if len(q) <= 12:
            return True
        low = q.lower()
        return any(m in low for m in _DEPENDENT_MARKERS)

    def condense(self, question: str, history: list[dict[str, str]]) -> tuple[str, bool]:
        """返回 (检索用查询, 是否真的做了改写)。失败时回退到关键词承接。"""
        self.last_condense_mode = "none"
        if not self.needs_condense(question, history):
            return question, False
        try:
            text, usage = self.llm.complete(
                build_condense_messages(history, question), temperature=0.0, max_tokens=120
            )
            self.last_condense_cost = estimate_tokens(text)
            cleaned = text.strip().strip("\"'“”").splitlines()[0].strip()
            if self._valid_rewrite(cleaned, question):
                self.last_condense_mode = "llm"
                return cleaned, True
        except Exception:
            pass
        carried = self._keyword_carryover(question, history)
        if carried:
            self.last_condense_mode = "carryover"
            return carried, True
        return question, False

    @staticmethod
    def _valid_rewrite(candidate: str, question: str) -> bool:
        """校验模型返回的改写结果"像不像一个查询"。

        模型（或兜底实现）把"回答"或"拒绝话术"塞进改写结果，是 RAG 里很隐蔽的一类
        故障：检索用一句垃圾去查，召回全错，而用户看到的是"答非所问"。
        所以这里宁可退回规则兜底，也不接受可疑的改写。
        """
        if not (2 <= len(candidate) <= 200):
            return False
        if NO_ANSWER_SENTINEL in candidate:
            return False
        bad_markers = ("离线", "未启用", "无法", "抱歉", "抱歉", "作为AI", "作为 AI", "请提供")
        if any(m in candidate for m in bad_markers):
            return False
        # 改写通常比原句略长，但不应膨胀成一段话
        return len(candidate) <= max(40, len(question) * 4)

    def _keyword_carryover(self, question: str, history: list[dict[str, str]]) -> str:
        """无模型改写时的兜底：把上一轮里**语料中真实存在且最稀有**的词条接到问句前。

        为什么不能简单地把上一轮整句拼进来：拼接会产生大量新词条，把检索查询"稀释"
        （覆盖率下降、BM25 分数被摊薄），反而比不改写更差。挑稀有词条才是"保留话题、
        不引入噪声"。

        为什么要求 ``df>0``：中文二元切分跨词边界会产生"障是""是怎"这类伪词条，
        它们 df=0、IDF 最高，若按 IDF 排序会优先选中垃圾。
        """
        prev_user = next(
            (t["content"] for t in reversed(history) if t.get("role") == "user" and t.get("content")),
            "",
        )
        if not prev_user:
            return ""
        terms = [t for t in query_terms(prev_user) if len(t) > 1]
        if not terms:
            return ""
        if self.store is None:
            picked = terms[:2]
        else:
            df = self.store.term_df(terms)
            known = [t for t in terms if df.get(t, 0) > 0]
            if not known:
                return ""
            idf = self.store.term_idf(known)
            picked = sorted(known, key=lambda t: (-idf[t], t))[:3]
        if not picked:
            return ""
        return f"{' '.join(picked)} {question}"

    # ------------------------------------------------------------------
    def route(self, question: str) -> str:
        """返回 ``kb`` 或 ``chat``。先用规则（免费、可预测），复杂场景再上模型。"""
        q = question.strip()
        if not q:
            return "chat"
        if _GREETING_RE.match(q):
            return "chat"
        low = q.lower()
        if any(h in low for h in _CHAT_HINTS):
            return "chat"
        if any(h in low for h in _KB_HINTS):
            return "kb"
        # 兜底：短问句且无知识库线索 → 闲聊；其余默认查知识库（召回为空会自然拒答）
        if len(q) <= 8 and not re.search(r"[?？]", q):
            return "chat"
        return "kb"

    def route_llm(self, question: str) -> str:
        """用模型做路由（留作对比实验）。"""
        try:
            text, _ = self.llm.complete(build_router_messages(question), temperature=0.0, max_tokens=8)
            return "chat" if "CHAT" in text.upper() else "kb"
        except Exception:
            return self.route(question)

    # ------------------------------------------------------------------
    def decompose(self, question: str, max_parts: int = 3) -> list[str]:
        """把复合问题拆成子查询（"A 怎么配？另外 B 的限额是多少？"）。

        只在拆出的部分都足够长时才生效，避免把一句话拆碎反而降低召回。
        """
        parts = re.split(r"[？?；;]\s*|\s+(?:以及|还有|另外|同时)\s*|，\s*(?:以及|还有|另外)\s*", question)
        cleaned = [p.strip(" ，,。.") for p in parts if p and len(p.strip()) >= 6]
        if len(cleaned) < 2:
            return [question]
        return cleaned[:max_parts]
