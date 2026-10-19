"""配置：环境变量 / ``.env`` 文件，零依赖实现。

所有变量以 ``RAGKB_`` 前缀读取，例如 ``RAGKB_LLM_PROVIDER=openai``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PREFIX = "RAGKB_"


def load_dotenv(path: Path) -> dict[str, str]:
    """极简 ``.env`` 解析：支持 ``#`` 注释、``export`` 前缀、成对引号。"""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_tuple(value: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in value.split(",") if p.strip())


@dataclass
class Settings:
    """运行期配置。字段名 ``foo_bar`` 对应环境变量 ``RAGKB_FOO_BAR``。"""

    # ---- 目录与存储 ----
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    db_path: Path = PROJECT_ROOT / "data" / "ragkb.sqlite3"
    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    eval_dir: Path = PROJECT_ROOT / "eval"
    trace_path: Path = PROJECT_ROOT / "data" / "traces.jsonl"
    feedback_path: Path = PROJECT_ROOT / "data" / "feedback.jsonl"

    # ---- embedding ----
    # hashing: 本地确定性词法向量（兜底/测试）；onnx: 本地语义模型（ONNX Runtime，无需 torch）；
    # openai: 任何 OpenAI 兼容 /embeddings
    embed_provider: str = "hashing"
    embed_model: str = "hashing-512"
    embed_dim: int = 512
    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key: str = ""
    embed_batch_size: int = 32
    embed_timeout_s: float = 60.0
    # 本地 ONNX embedding（embed_provider=onnx）
    onnx_model_dir: Path = PROJECT_ROOT / "models" / "bge-small-zh-v1.5"
    onnx_max_length: int = 512
    onnx_batch_size: int = 16
    onnx_pooling: str = "cls"  # cls（BGE 默认）| mean
    onnx_threads: int = 0  # 0 = 交给 onnxruntime 自己决定

    # ---- LLM ----
    # scripted: 本地抽取式生成（离线/测试用）；openai: 任何 OpenAI 兼容 /chat/completions
    llm_provider: str = "scripted"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 900
    llm_timeout_s: float = 90.0

    # ---- rerank ----
    # none | lexical（本地词法交叉打分）| http（Cohere / Jina 风格 /rerank）
    rerank_provider: str = "lexical"
    rerank_model: str = ""
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_top_n: int = 30
    lexical_rerank_weight: float = 0.65

    # ---- 切块 ----
    chunk_size: int = 800
    chunk_overlap: int = 120
    chunk_min_chars: int = 120

    # ---- 检索 ----
    retrieve_candidates: int = 50
    fuse_top_k: int = 20
    final_top_k: int = 6
    rrf_k: int = 60
    # 向量通道的最低相似度（0 = 不启用）。线上真实 embedding 建议 0.15~0.25，
    # 本地 hashing 兜底向量整体相似度偏高，保持 0 即可。
    vector_min_similarity: float = 0.0
    # 拒答门槛：
    # * refuse_min_hits —— 检索结果条数下限
    # * refuse_score_floor —— 最高分下限
    # * refuse_coverage_floor —— 查询词在上下文中的覆盖率下限（0 = 关闭）。
    #   这是最有效的一条：问题里的关键实体在语料里根本没出现时，
    #   检索仍会返回"最不相关的一批"，覆盖率能把它挡在模型调用之前。
    refuse_min_hits: int = 1
    refuse_score_floor: float = 0.02
    refuse_coverage_floor: float = 0.25
    # 语义豁免：当向量通道的最高相似度达到该值时，跳过词法覆盖率门禁。
    # 纯词法门禁会把"用户措辞与文档差得远"的正确问题误拒（实测 5.6%）；
    # 有了真实语义 embedding 后，模型判断"确实相关"就不该被词法挡住。
    # 0 = 关闭（hashing 兜底向量的绝对值不可比，所以默认关闭）；
    # 本地 bge-small-zh 建议 0.58~0.62，云端 bge-m3 建议 0.45~0.55（分布不同，要按自己语料扫）。
    refuse_vector_similarity_floor: float = 0.0

    # ---- 服务 ----
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_key: str = ""
    default_roles: tuple[str, ...] = ("public",)
    max_upload_mb: float = 20.0

    # ---- 可观测 ----
    trace_enabled: bool = True

    # ---- 价格表（USD / 1K tokens），用于成本统计 ----
    price_table: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "gpt-4o-mini": (0.00015, 0.0006),
            "gpt-4o": (0.0025, 0.01),
            "deepseek-chat": (0.00027, 0.0011),
            "qwen-plus": (0.0004, 0.0012),
        }
    )

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, env_file: Path | None = None, env: dict[str, str] | None = None) -> "Settings":
        """从 ``.env`` + 进程环境构建配置（进程环境优先）。"""
        env_file = env_file or (PROJECT_ROOT / ".env")
        merged: dict[str, str] = dict(load_dotenv(env_file))
        merged.update(env if env is not None else dict(os.environ))

        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name == "price_table":
                continue
            raw = merged.get(ENV_PREFIX + f.name.upper())
            if raw is None or raw == "":
                continue
            kwargs[f.name] = _coerce(f.type, raw)

        settings = cls(**kwargs)
        # 目录派生：如果只改了 data_dir，其余路径跟着走
        if "data_dir" in kwargs:
            base = Path(kwargs["data_dir"])
            settings.db_path = base / "ragkb.sqlite3"
            settings.corpus_dir = base / "corpus"
            settings.trace_path = base / "traces.jsonl"
            settings.feedback_path = base / "feedback.jsonl"
        settings.ensure_dirs()
        return settings

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.corpus_dir):
            Path(p).mkdir(parents=True, exist_ok=True)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def price_for(self, model: str) -> tuple[float, float]:
        """返回 (输入单价, 输出单价)，单位 USD / 1K tokens。"""
        if model in self.price_table:
            return self.price_table[model]
        for key, price in self.price_table.items():
            if model.startswith(key):
                return price
        return (0.0, 0.0)

    def describe(self) -> dict[str, Any]:
        """打印/接口用摘要（不含密钥）。"""
        return {
            "embed": f"{self.embed_provider}:{self.embed_model}",
            "llm": f"{self.llm_provider}:{self.llm_model}",
            "rerank": self.rerank_provider,
            "chunk": f"{self.chunk_size}/{self.chunk_overlap}",
            "retrieve": f"cand={self.retrieve_candidates} fuse={self.fuse_top_k} final={self.final_top_k}",
            "db": str(self.db_path),
        }


def _coerce(annotation: Any, raw: str) -> Any:
    """按 dataclass 字段注解把字符串转成目标类型。"""
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    text = str(text)
    if "bool" in text:
        return _as_bool(raw)
    if "int" in text:
        return int(float(raw))
    if "float" in text:
        return float(raw)
    if "tuple" in text:
        return _as_tuple(raw)
    if "Path" in text:
        return Path(raw).expanduser()
    return raw
