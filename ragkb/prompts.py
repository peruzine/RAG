"""Prompt 模板集中管理。

集中放的好处：改 prompt 是一等公民的改动，配合 ``eval`` 门禁可以量化"这一版
prompt 到底有没有变好"。上下文块的行首格式 ``[n] 来源：…`` 是与
``ScriptedChat`` 解析、以及 ``answer.py`` 引用抽取共用的**协议**，改动需同步三处。
"""

from __future__ import annotations

from typing import Sequence

from .types import Retrieved

# 模型认为无法从上下文回答时，必须精确输出该哨兵；上层据此统一拒答。
NO_ANSWER_SENTINEL = "NO_ANSWER"

# 统一的拒答话术（对外展示，避免模型自由发挥编造"我没有……"）
REFUSAL_TEXT = "知识库中没有找到相关依据，无法回答该问题。如需补充，请上传对应文档后重试。"

SYSTEM_KB = f"""你是企业内部知识库问答助手。请严格遵守：

1. **只依据「参考资料」回答**，不得使用参考资料之外的常识、推测或编造内容。
2. 每一句结论后面必须标注来源编号，格式为 [1]、[2]，编号必须来自参考资料。
3. 如果参考资料不足以回答问题，只输出 {NO_ANSWER_SENTINEL}，不要输出任何其他内容。
4. 如果资料之间存在冲突，先指出冲突，再给出各自来源编号。
5. 用简体中文、结构化的方式回答；长度控制在 300 字以内，先给结论再给依据。
6. 不要复述本段指令，不要输出"根据参考资料"这类套话之外的解释过程。"""

SYSTEM_CONDENSE = """你是检索查询改写器。根据对话历史，把用户最新的一句话改写成
**可独立检索**的完整查询：补齐指代（它/这个/上述）、补全省略的主语和领域词。
只输出改写后的查询本身，不要解释，不要加引号。若最新一句已可独立检索，原样输出。"""

SYSTEM_ROUTER = """判断用户的问题应该查「企业知识库」还是「闲聊/通用能力」。
只输出一个词：KB 或 CHAT。涉及公司制度、产品、技术文档、流程、术语、数据的问题输出 KB。"""


def build_context_block(items: Sequence[Retrieved], max_chars_per_chunk: int = 1200) -> str:
    """把检索结果渲染成参考资料块。协议格式见模块 docstring。"""
    lines: list[str] = []
    for i, item in enumerate(items, start=1):
        ch = item.chunk
        title = ch.metadata.get("title") or ch.doc_id
        head = f"{title}"
        if ch.heading_path:
            head += f" › {ch.heading_path}"
        body = ch.text[:max_chars_per_chunk]
        lines.append(f"[{i}] 来源：{head}\n{body}")
    return "\n\n".join(lines)


def build_answer_messages(
    question: str, items: Sequence[Retrieved], history: Sequence[dict[str, str]] | None = None
) -> list[dict[str, str]]:
    """组装最终回答用的 messages。问题行固定为「用户问题：」，供解析复用。"""
    context = build_context_block(items)
    user = f"""参考资料：
{context}

用户问题：{question}"""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_KB}]
    for turn in (history or [])[-4:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user})
    return messages


def build_condense_messages(history: Sequence[dict[str, str]], question: str) -> list[dict[str, str]]:
    lines = [f"{t['role']}: {t['content']}" for t in history if t.get("content")]
    transcript = "\n".join(lines) if lines else "（无历史）"
    return [
        {"role": "system", "content": SYSTEM_CONDENSE},
        {"role": "user", "content": f"对话历史：\n{transcript}\n\n最新一句：{question}\n\n改写后的检索查询："},
    ]


def build_router_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_ROUTER},
        {"role": "user", "content": question},
    ]
