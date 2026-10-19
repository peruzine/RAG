"""LLM Provider 抽象。

* ``ScriptedChat`` —— 本地**抽取式**基线：从 prompt 里的上下文块中挑出与问题
  词重叠最高的句子作为答案，并带上引用角标；找不到依据就输出拒答标记。
  它同时承担三个角色：离线可跑的默认实现、单元测试的确定性桩、以及
  "抽取式 vs 生成式" 的对比基线（评测时可以量化生成式模型到底带来多少增益）。
* ``OpenAICompatChat`` —— ``/chat/completions``，支持流式与用量回传。

统一约定：模型无法从上下文回答时，必须输出 ``NO_ANSWER`` 哨兵，由上层翻译成
统一的拒答话术。这样"拒答"就是可评测的，而不是靠感觉。
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import Any, Iterator, Sequence

from .config import Settings
from .http import auth_headers, post_json, stream_sse
from .prompts import NO_ANSWER_SENTINEL
from .text import estimate_tokens, query_terms, split_sentences
from .types import Usage

Message = dict[str, str]

_CONTEXT_RE = re.compile(r"^\[(\d+)\][^\n]*\n(.*?)(?=\n\[\d+\]|\n用户问题[：:]|\Z)", re.S | re.M)
_QUESTION_RE = re.compile(r"用户问题[：:]\s*(.+?)\s*$", re.M)


class ChatModel(ABC):
    name: str = "base"

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, Usage]:
        """一次性返回 (文本, 用量)。"""

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """默认实现：退化为一次性返回（子类可覆写为真流式）。"""
        text, _ = self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        yield text


# --------------------------------------------------------------------------
class ScriptedChat(ChatModel):
    """确定性抽取式生成器（离线默认）。"""

    def __init__(self, name: str = "scripted-extractive", min_hit_weight: float = 1.0) -> None:
        self.name = name
        self.min_hit_weight = min_hit_weight

    # -- 解析 prompt（这样它和真实模型的输入完全一致，可互换） --------------
    @staticmethod
    def _parse(messages: Sequence[Message]) -> tuple[str, list[tuple[int, str]]]:
        user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user = msg.get("content", "")
                break
        contexts = [(int(m.group(1)), m.group(2).strip()) for m in _CONTEXT_RE.finditer(user)]
        qm = None
        for m in _QUESTION_RE.finditer(user):
            qm = m
        question = qm.group(1) if qm else user.strip().splitlines()[-1] if user else ""
        return question, contexts

    @staticmethod
    def _table_candidates(body: str) -> list[str]:
        """把 Markdown 表格的「表头 + 数据行」拼成候选答案片段。

        抽取式基线常犯的一个错：表头（`| 等级 | 名称 | 首次响应 | 解决时限 |`）比
        数据行（`| P0 | 致命 | 15 分钟 | 4 小时 |`）命中更多查询词，于是模型把表头
        当答案输出——用户看到的是"表头"而不是答案。把表头和每一行拼起来参与打分，
        就能让真正含答案的那一行胜出；这也顺带保留了列名（15 分钟对应"首次响应"）。
        """
        lines = [l for l in body.split("\n") if l.strip().startswith("|")]
        if len(lines) < 3:
            return []
        header, rows = lines[0], lines[1:]
        out: list[str] = []
        for row in rows:
            # 跳过分隔行 | --- | --- |
            if not set(row.replace("|", "").strip()) - set("-: "):
                continue
            out.append(f"{header}\n{row}")
        return out

    def _answer(self, messages: Sequence[Message]) -> str:
        question, contexts = self._parse(messages)
        if not contexts:
            # 没有参考资料 = 不是知识库问答任务。离线实现要**如实说明做不到**，
            # 而不是返回一段看似合理的话术——否则会被上层当成有效结果
            # （真实踩过的坑：对话式回复被当成"改写后的查询"送进检索）。
            last = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
            )
            if "改写后的检索查询" in last:
                return NO_ANSWER_SENTINEL  # 声明"我不会改写"，让上层走关键词承接兜底
            if "KB" in last or "CHAT" in last or "输出一个词" in last:
                return "KB"
            if not question:
                return "（离线模式：未收到问题）"
            return (
                "（离线模式：通用对话能力未启用。本默认实现是**抽取式**基线，"
                "只在提供参考资料时给出有依据的回答。配置 RAGKB_LLM_PROVIDER=openai "
                "与 RAGKB_LLM_API_KEY 即可切换到真实模型。）"
            )
        if not question:
            return NO_ANSWER_SENTINEL

        terms = query_terms(question)
        if not terms:
            return NO_ANSWER_SENTINEL

        def weight(term: str) -> float:
            return 0.35 if len(term) == 1 else (1.0 if len(term) == 2 else 1.2)

        best: list[tuple[float, int, str]] = []
        # 只要求"至少命中一个查询词"：中文二元切分会跨词边界产生伪词条，
        # 要求命中多个会把"报销款什么时候到账"这类改写提问全挡掉。
        # "语料里根本没有"的情况由上层的 IDF 覆盖率门禁负责，不在这里判。
        min_hits = 1
        for idx, body in contexts:
            low = body.lower()
            candidates: list[tuple[str, float]] = []
            for sent in split_sentences(body):
                if len(sent) >= 8:
                    candidates.append((sent.strip(), 1.0))
            # 表格行给一点加成：它比表头更可能是"答案本身"
            for pair in self._table_candidates(body):
                candidates.append((pair, 1.15))

            for sent, bonus in candidates:
                low_sent = sent.lower()
                hits = [t for t in terms if t in low_sent]
                if len(hits) < min_hits:
                    continue
                # 命中词加权 + 覆盖率加成，避免"长句偶然含一个词"胜出
                score = (
                    sum(weight(t) for t in hits)
                    * (1.0 + 0.25 * len(hits) / max(1, len(terms)))
                    * bonus
                )
                best.append((score, idx, sent))

        if not best:
            return NO_ANSWER_SENTINEL
        best.sort(key=lambda x: (-x[0], x[1], -len(x[2])))
        score, idx, sentence = best[0]
        if score < self.min_hit_weight:
            return NO_ANSWER_SENTINEL

        # 再补一句不同来源的支撑（如果有），体现"多源交叉"
        extra = ""
        for s2, idx2, sent2 in best[1:]:
            if idx2 != idx and sent2 != sentence and s2 >= score * 0.5:
                extra = f" 此外，[{idx2}] 也提到：{sent2}"
                break
        return f"根据知识库内容：[{idx}] {sentence}{extra}"

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, Usage]:
        text = self._answer(messages)
        prompt_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        return text, Usage(prompt_tokens=prompt_tokens, completion_tokens=estimate_tokens(text))

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        text = self._answer(messages)
        # 按标点/长度切片，模拟真实模型的流式输出节奏
        buf = ""
        for ch in text:
            buf += ch
            if len(buf) >= 12 or ch in "。！？；\n":
                yield buf
                buf = ""
        if buf:
            yield buf


# --------------------------------------------------------------------------
class OpenAICompatChat(ChatModel):
    """OpenAI 兼容 ``/chat/completions``。"""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 900,
        timeout: float = 90.0,
        send_usage_in_stream: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.name = model
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.send_usage_in_stream = send_usage_in_stream
        self.extra_headers = extra_headers or {}

    def _headers(self) -> dict[str, str]:
        return {**auth_headers(self.api_key), **self.extra_headers}

    def _payload(self, messages: Sequence[Message], temperature, max_tokens, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }
        if stream and self.send_usage_in_stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, Usage]:
        data = post_json(
            f"{self.base_url}/chat/completions",
            self._payload(messages, temperature, max_tokens, stream=False),
            headers=self._headers(),
            timeout=self.timeout,
        )
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        u = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            completion_tokens=int(u.get("completion_tokens") or 0),
        )
        if not usage.prompt_tokens:
            usage.prompt_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
            usage.completion_tokens = usage.completion_tokens or estimate_tokens(text)
        return text.strip(), usage

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        usage = Usage()
        for chunk in stream_sse(
            f"{self.base_url}/chat/completions",
            self._payload(messages, temperature, max_tokens, stream=True),
            headers=self._headers(),
            timeout=self.timeout,
        ):
            if chunk.get("usage"):
                usage.prompt_tokens = int(chunk["usage"].get("prompt_tokens") or usage.prompt_tokens)
                usage.completion_tokens = int(
                    chunk["usage"].get("completion_tokens") or usage.completion_tokens
                )
            for choice in chunk.get("choices") or []:
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    yield delta
        self.last_usage = usage


# --------------------------------------------------------------------------
def build_chat_model(cfg: Settings) -> ChatModel:
    provider = (cfg.llm_provider or "scripted").lower()
    if provider in ("scripted", "local", "extractive", "fake"):
        return ScriptedChat()
    if provider in ("openai", "openai-compatible", "compat", "deepseek", "qwen"):
        if not cfg.llm_api_key:
            raise RuntimeError("llm_provider=openai 但未设置 RAGKB_LLM_API_KEY")
        return OpenAICompatChat(
            cfg.llm_model,
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            timeout=cfg.llm_timeout_s,
        )
    raise ValueError(f"未知 llm_provider: {cfg.llm_provider}")


def timed(fn, *args, **kwargs):
    """小工具：返回 (结果, 毫秒)。trace 里到处要用。"""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000.0
