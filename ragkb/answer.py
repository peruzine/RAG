"""答案生成：引用抽取、拒答判定、流式输出。

三条硬规则
----------
1. **引用必须可验证**：模型输出的 ``[n]`` 会被逐一对齐到检索结果；越界的编号
   （比如只给了 5 条上下文却引用 ``[7]``）会被剔除并计入 ``invalid_citations``，
   这个数字是评测幻觉的重要代理指标。
2. **拒答要有两层**：便宜的规则前置（没召回、最高分过低）先挡掉明显无依据的问题，
   模型侧的 ``NO_ANSWER`` 哨兵再挡语义层面的"资料里有但答不了"。
3. **失败要降级不要 500**：模型调用异常时，退化为"给出最相关的原文摘录 + 说明"，
   用户至少能自助，同时 trace 里记录异常，方便告警。
"""

from __future__ import annotations

import re
import time
from typing import Any, Iterator, Sequence

from .config import Settings
from .llm import ChatModel
from .obs import Tracer, cost_usd
from .prompts import NO_ANSWER_SENTINEL, REFUSAL_TEXT, build_answer_messages
from .query import QueryProcessor
from .retrieve import HybridRetriever, RetrievalResult
from .text import highlight_terms, query_terms
from .types import Answer, Citation, Retrieved, Usage

_CITE_RE = re.compile(r"\[(\d{1,2})\]")


class Answerer:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: ChatModel,
        cfg: Settings,
        tracer: Tracer,
        query_processor: QueryProcessor | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.cfg = cfg
        self.tracer = tracer
        self.qp = query_processor or QueryProcessor(llm, cfg)

    # ------------------------------------------------------------------
    # 引用
    # ------------------------------------------------------------------
    @staticmethod
    def extract_citations(
        text: str, items: Sequence[Retrieved], question: str = ""
    ) -> tuple[list[Citation], list[int]]:
        """把 ``[n]`` 映射成结构化引用，返回 (合法引用, 越界编号)。"""
        used: list[int] = []
        for m in _CITE_RE.finditer(text or ""):
            n = int(m.group(1))
            if n not in used:
                used.append(n)

        terms = query_terms(question)
        citations: list[Citation] = []
        invalid: list[int] = []
        for n in used:
            if not (1 <= n <= len(items)):
                invalid.append(n)
                continue
            ch = items[n - 1].chunk
            citations.append(
                Citation(
                    index=n,
                    chunk_id=ch.chunk_id,
                    doc_id=ch.doc_id,
                    source=ch.metadata.get("source", ""),
                    title=ch.metadata.get("title") or ch.doc_id,
                    heading_path=ch.heading_path,
                    quote=highlight_terms(ch.text, terms, 6),
                    score=items[n - 1].rerank_score
                    if items[n - 1].rerank_score is not None
                    else items[n - 1].score,
                )
            )
        citations.sort(key=lambda c: c.index)
        return citations, invalid

    def term_coverage(self, question: str, items: Sequence[Retrieved], top_n: int = 3) -> float:
        """查询词在前若干条上下文中的 **IDF 加权** 覆盖率（0~1）。

        为什么不用相似度分数：不同 embedding 的相似度分布差异极大，阈值无法跨模型
        复用。为什么不用平均覆盖率：中文二元切分会产生大量"公司/数据"这类高频词，
        它们命中与否几乎不含信息量；按 IDF 加权后，"问题里的关键实体到底在不在
        语料里"才是可跨语料复用的硬信号。
        """
        terms = query_terms(question)
        if not terms:
            return 1.0
        idf = self.retriever.store.term_idf(terms)
        total = sum(idf.values())
        if total <= 0:
            return 1.0
        # 把 heading_path 也算进"上下文可见文本"：章节标题会一并送进 prompt
        # （"来源：手册 › 住宿标准"），只算正文会把"按章节提问"误判成语料里没有。
        text = " ".join(
            f"{it.chunk.heading_path} {it.chunk.text}" for it in items[:top_n]
        ).lower()
        hit = sum(idf[t] for t in terms if t in text)
        return hit / total

    def effective_coverage(
        self, search_query: str, items: Sequence[Retrieved], original_query: str = ""
    ) -> tuple[float, float, float]:
        """返回 (取大后的覆盖率, 改写查询覆盖率, 原始问句覆盖率)。

        取**最大值**而不是只算改写后的查询：改写本身可能失败（本地兜底是把上一轮
        问题拼在当前问题前面），拼接出来的长查询会掺进大量不存在的词条，把覆盖率
        压低到误拒。门禁的职责是"这个主题语料里到底有没有"，那么两种措辞里**任一种**
        能被上下文覆盖，就不该拒答。
        """
        cov_search = self.term_coverage(search_query, items)
        if original_query and original_query != search_query:
            cov_orig = self.term_coverage(original_query, items)
            return max(cov_search, cov_orig), cov_search, cov_orig
        return cov_search, cov_search, cov_search

    def _should_refuse_pre_llm(
        self, query: str, result: RetrievalResult, original_query: str = ""
    ) -> str | None:
        """便宜的规则前置拒答，返回拒答原因（None 表示交给模型判断）。

        传入的 ``query`` 是**改写后的检索查询**、``original_query`` 是用户原话：
        多轮场景下原始问句里没有实体（"它多久能修好"），只拿它算覆盖率会必然误拒。
        """
        if not result.items:
            return "no_hits"
        if len(result.items) < self.cfg.refuse_min_hits:
            return "too_few_hits"
        if result.best_score < self.cfg.refuse_score_floor:
            return "low_score"

        # 语义豁免：向量通道认为"确实相关"时，不要用词法覆盖率把问题误拒。
        # 这是换成真实 embedding 之后必须补的一环——否则检索侧的收益会被门禁吃掉：
        # 语义模型能把"未达标"和文档里的"未达到"对上，而二元切分的覆盖率对不上。
        best_vector = max(
            (it.channel_scores.get("vector", 0.0) for it in result.items), default=0.0
        )
        result.debug.best_vector_similarity = round(best_vector, 4)
        if (
            self.cfg.refuse_vector_similarity_floor > 0
            and best_vector >= self.cfg.refuse_vector_similarity_floor
        ):
            result.debug.coverage_detail = {"semantic_exempt": round(best_vector, 4)}
            return None

        if self.cfg.refuse_coverage_floor > 0:
            coverage, cov_search, cov_orig = self.effective_coverage(
                query, result.items, original_query
            )
            result.debug.coverage = round(coverage, 4)
            result.debug.coverage_detail = {
                "rewritten": round(cov_search, 4),
                "original": round(cov_orig, 4),
            }
            if coverage < self.cfg.refuse_coverage_floor:
                return f"low_coverage:{coverage:.3f}"
        return None

    @staticmethod
    def _fallback_from_context(question: str, items: Sequence[Retrieved]) -> str:
        """模型不可用时的抽取式兜底，保证"永远有答案可看"。"""
        if not items:
            return REFUSAL_TEXT
        terms = query_terms(question)
        snippets = []
        for i, it in enumerate(items[:3], start=1):
            snippets.append(f"[{i}] {highlight_terms(it.chunk.text, terms)}")
        return "（模型服务暂时不可用，以下是知识库中最相关的原文摘录）\n" + "\n".join(snippets)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def prepare(
        self,
        question: str,
        *,
        roles: Sequence[str] | None = (),
        doc_ids: Sequence[str] | None = None,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        use_rerank: bool = True,
    ) -> dict[str, Any]:
        """检索 + 组装 messages（流式与非流式共用）。"""
        trace = self.tracer.new_trace("ask")
        history = history or []

        with trace.span("condense") as attrs:
            search_query, rewritten = self.qp.condense(question, history)
            attrs["rewritten"] = rewritten
            attrs["query"] = search_query

        with trace.span("retrieve") as attrs:
            queries = self.qp.decompose(search_query) if self.cfg.final_top_k > 1 else [search_query]
            if len(queries) > 1:
                result = self.retriever.retrieve_multi(
                    queries, top_k=top_k, roles=roles, doc_ids=doc_ids, use_rerank=use_rerank
                )
            else:
                result = self.retriever.retrieve(
                    search_query, top_k=top_k, roles=roles, doc_ids=doc_ids, use_rerank=use_rerank
                )
            attrs.update(
                {
                    "n_hits": len(result.items),
                    "best_score": round(result.best_score, 6),
                    "channels_only": result.debug.channels_only,
                    "stage_ms": result.debug.timings_ms,
                }
            )

        messages = build_answer_messages(search_query, result.items, history)
        return {
            "trace": trace,
            "question": question,
            "search_query": search_query,
            "rewritten": rewritten,
            "result": result,
            "messages": messages,
        }

    def answer(
        self,
        question: str,
        *,
        roles: Sequence[str] | None = (),
        doc_ids: Sequence[str] | None = None,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        use_rerank: bool = True,
        route: str | None = None,
    ) -> Answer:
        t0 = time.perf_counter()
        route = route or self.qp.route(question)

        if route == "chat":
            # 闲聊直接走模型，不检索：省一次检索 + 避免无关文档污染回答
            trace = self.tracer.new_trace("ask")
            try:
                with trace.span("chat"):
                    text, usage = self.llm.complete(
                        [{"role": "user", "content": question}], temperature=0.3, max_tokens=300
                    )
            except Exception as e:
                text, usage = f"（模型不可用：{type(e).__name__}）", Usage()
            trace.add_usage(usage, self.llm.name)
            ans = Answer(
                text=text,
                model=self.llm.name,
                usage=usage,
                latency_ms=(time.perf_counter() - t0) * 1000,
                trace_id=trace.trace_id,
                route="chat",
            )
            trace.attrs.update({"route": "chat", "refused": False})
            trace.finish(route="chat", refused=False, question=question)
            return ans

        ctx = self.prepare(
            question,
            roles=roles,
            doc_ids=doc_ids,
            history=history,
            top_k=top_k,
            use_rerank=use_rerank,
        )
        trace = ctx["trace"]
        result: RetrievalResult = ctx["result"]
        items = result.items

        refuse_reason = self._should_refuse_pre_llm(ctx["search_query"], result, question)
        usage = Usage()
        text = ""
        error: str | None = None

        if refuse_reason:
            text = REFUSAL_TEXT
        else:
            try:
                with trace.span("generate"):
                    text, usage = self.llm.complete(
                        ctx["messages"],
                        temperature=self.cfg.llm_temperature,
                        max_tokens=self.cfg.llm_max_tokens,
                    )
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                text = self._fallback_from_context(question, items)

        refused = NO_ANSWER_SENTINEL in (text or "") or refuse_reason is not None
        citations: list[Citation] = []
        invalid: list[int] = []
        if refused:
            if refuse_reason or error is None:
                text = REFUSAL_TEXT
        else:
            citations, invalid = self.extract_citations(text, items, question)
            # 有上下文却一个引用都没给：记录为 uncited（评测/告警信号），不擅自改答案

        trace.add_usage(usage, self.llm.name)
        total_ms = (time.perf_counter() - t0) * 1000

        ans = Answer(
            text=text,
            citations=citations,
            refused=refused,
            model=self.llm.name,
            usage=usage,
            latency_ms=total_ms,
            trace_id=trace.trace_id,
            retrieved=list(items),
            rewritten_query=ctx["search_query"] if ctx["rewritten"] else "",
            route="kb",
            cost_usd=trace.cost,
            debug={
                "retrieval": result.debug.to_dict(),
                "refuse_reason": refuse_reason,
                "error": error,
                "invalid_citations": invalid,
                "uncited": (not refused and not citations),
                "n_context": len(items),
            },
        )
        trace.attrs.update(
            {
                "route": "kb",
                "question": question,
                "search_query": ctx["search_query"],
                "n_context": len(items),
                "citation_ids": [c.index for c in citations],
                "invalid_citations": invalid,
                "refused": refused,
                "refuse_reason": refuse_reason,
                "error": error,
            }
        )
        trace.finish()
        ans.trace_id = trace.trace_id
        return ans

    # ------------------------------------------------------------------
    def stream(
        self,
        question: str,
        *,
        roles: Sequence[str] | None = (),
        doc_ids: Sequence[str] | None = None,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        use_rerank: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """流式输出事件：``meta`` → ``delta``* → ``done``。"""
        t0 = time.perf_counter()
        ctx = self.prepare(
            question,
            roles=roles,
            doc_ids=doc_ids,
            history=history,
            top_k=top_k,
            use_rerank=use_rerank,
        )
        trace = ctx["trace"]
        result: RetrievalResult = ctx["result"]
        items = result.items

        # 先判定拒答，再发 meta：否则前端看到的 coverage / refuse_reason 永远是初值，
        # 排查"为什么拒答"时会被误导（实测踩过）。
        refuse_reason = self._should_refuse_pre_llm(ctx["search_query"], result, question)

        yield {
            "type": "meta",
            "trace_id": trace.trace_id,
            "search_query": ctx["search_query"],
            "rewritten": ctx["rewritten"],
            "refuse_reason": refuse_reason,
            "retrieval": result.debug.to_dict(),
            "contexts": [
                {
                    "index": i,
                    "chunk_id": it.chunk_id,
                    "title": it.chunk.metadata.get("title") or it.chunk.doc_id,
                    "heading_path": it.chunk.heading_path,
                    "source": it.chunk.metadata.get("source", ""),
                    "score": round(
                        it.rerank_score if it.rerank_score is not None else it.score, 6
                    ),
                    "snippet": it.chunk.text[:200],
                }
                for i, it in enumerate(items, start=1)
            ],
        }

        buffered: list[str] = []
        usage = Usage()
        error: str | None = None

        if refuse_reason:
            buffered.append(REFUSAL_TEXT)
            yield {"type": "delta", "text": REFUSAL_TEXT}
        else:
            try:
                for delta in self.llm.stream(
                    ctx["messages"],
                    temperature=self.cfg.llm_temperature,
                    max_tokens=self.cfg.llm_max_tokens,
                ):
                    buffered.append(delta)
                    yield {"type": "delta", "text": delta}
                usage = getattr(self.llm, "last_usage", Usage())
                if not usage.prompt_tokens:
                    from .text import estimate_tokens

                    usage = Usage(
                        prompt_tokens=sum(estimate_tokens(m["content"]) for m in ctx["messages"]),
                        completion_tokens=estimate_tokens("".join(buffered)),
                    )
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                fallback = self._fallback_from_context(question, items)
                buffered = [fallback]
                yield {"type": "delta", "text": fallback}

        text = "".join(buffered)
        refused = NO_ANSWER_SENTINEL in text or refuse_reason is not None
        citations: list[Citation] = []
        invalid: list[int] = []
        if refused:
            final_text = REFUSAL_TEXT if (refuse_reason or error is None) else text
            # 已把哨兵流式发出去时，前端会用 done 里的最终文本覆盖
        else:
            final_text = text
            citations, invalid = self.extract_citations(text, items, question)

        trace.add_usage(usage, self.llm.name)
        trace.attrs.update(
            {
                "route": "kb",
                "question": question,
                "search_query": ctx["search_query"],
                "n_context": len(items),
                "citation_ids": [c.index for c in citations],
                "invalid_citations": invalid,
                "refused": refused,
                "refuse_reason": refuse_reason,
                "error": error,
            }
        )
        record = trace.finish()

        ans = Answer(
            text=final_text,
            citations=citations,
            refused=refused,
            model=self.llm.name,
            usage=usage,
            latency_ms=(time.perf_counter() - t0) * 1000,
            trace_id=trace.trace_id,
            retrieved=list(items),
            rewritten_query=ctx["search_query"] if ctx["rewritten"] else "",
            route="kb",
            cost_usd=cost_usd(usage, self.llm.name, self.cfg.price_table),
            debug={
                "retrieval": result.debug.to_dict(),
                "refuse_reason": refuse_reason,
                "error": error,
                "invalid_citations": invalid,
                "uncited": (not refused and not citations),
                "n_context": len(items),
                "trace": {"latency_ms": record["latency_ms"], "cost_usd": record["cost_usd"]},
            },
        )
        yield {"type": "done", "answer": ans.to_dict(), "refused": refused}
