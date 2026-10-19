"""Embedding Provider 抽象。

两个实现：

* ``HashingEmbedder`` —— 本地、确定性、零依赖的**词法向量**（signed hashing）。
  它不是"假 embedding"：用同一套中英文词条做特征哈希，配合长度加权近似 IDF，
  再用余弦相似度检索，本质是一个可持久化的稀疏→稠密投影。用于离线开发、
  单元测试、以及"没有外网/没有预算"时的可用兜底。语义泛化能力当然不如真模型，
  README 里明确标注了适用边界。
* ``OpenAICompatEmbedder`` —— 任何 OpenAI 兼容 ``/embeddings`` 网关
  （OpenAI、DeepSeek、Qwen/DashScope、vLLM、Ollama、Xinference…）。

两者输出都做 L2 归一化，因此点积 = 余弦，向量库侧可以直接用矩阵乘法。
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from .config import Settings
from .http import auth_headers, post_json
from .text import index_terms

Vector = list[float]


def l2_normalize(vec: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm == 0:
        return [0.0] * len(vec)
    return [float(x) / norm for x in vec]


class Embedder(ABC):
    """Embedding 接口。``name`` 会写进 chunk，用于防止跨模型比较向量。"""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """批量向量化（顺序与入参一致）。"""

    def embed_query(self, text: str) -> Vector:
        return self.embed([text])[0]

    def describe(self) -> dict[str, object]:
        return {"provider": type(self).__name__, "name": self.name, "dim": self.dim}


# --------------------------------------------------------------------------
class HashingEmbedder(Embedder):
    """signed feature hashing + 长度加权，纯本地。

    权重设计：``(1 + log(tf)) * w(len)``，其中 ``w`` 随词条长度增长，
    用来近似 IDF——一元汉字（"的""系"）几乎无区分度，二元/英文词更有信息量。
    """

    def __init__(self, dim: int = 512, name: str = "hashing") -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正")
        self.dim = dim
        self.name = f"{name}-{dim}"

    @staticmethod
    def _term_weight(term: str) -> float:
        if len(term) == 1:
            return 0.35
        if len(term) == 2:
            return 1.0
        return 1.2

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        out: list[Vector] = []
        for text in texts:
            vec = [0.0] * self.dim
            counts: dict[str, int] = {}
            for term in index_terms(text):
                counts[term] = counts.get(term, 0) + 1
            for term, tf in counts.items():
                digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
                h = int.from_bytes(digest, "big")
                bucket = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                weight = (1.0 + math.log(tf)) * self._term_weight(term)
                vec[bucket] += sign * weight
            out.append(l2_normalize(vec))
        return out


# --------------------------------------------------------------------------
class OpenAICompatEmbedder(Embedder):
    """通过 ``POST {base_url}/embeddings`` 调用任意 OpenAI 兼容网关。"""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str = "",
        batch_size: int = 32,
        timeout: float = 60.0,
        dim: int = 0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.dim = dim
        self.name = model
        self.extra_headers = extra_headers or {}

    def _headers(self) -> dict[str, str]:
        return {**auth_headers(self.api_key), **self.extra_headers}

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        items = [t if t.strip() else " " for t in texts]
        out: list[Vector] = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            data = post_json(
                f"{self.base_url}/embeddings",
                {"model": self.model, "input": batch, "encoding_format": "float"},
                headers=self._headers(),
                timeout=self.timeout,
            )
            rows = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
            if len(rows) != len(batch):
                raise RuntimeError(
                    f"embedding 返回条数不匹配：期望 {len(batch)}，实际 {len(rows)}"
                )
            for row in rows:
                vec = l2_normalize(row["embedding"])
                if self.dim and len(vec) != self.dim:
                    raise RuntimeError(f"维度不符：配置 {self.dim}，返回 {len(vec)}")
                out.append(vec)
        if out and not self.dim:
            self.dim = len(out[0])
        return out


# --------------------------------------------------------------------------
class OnnxEmbedder(Embedder):
    """本地**语义** embedding：ONNX Runtime + 中文 BGE 模型，不需要 torch。

    为什么单独做这条路
    ------------------
    真实 embedding 是检索质量最大的变量，但 `pip install sentence-transformers` 会拖进
    ~2.5GB 的 torch。ONNX Runtime + 量化/小模型只有几十 MB，CPU 上单条几毫秒，
    中小规模知识库完全够用，也不会把部署镜像撑爆。

    实现要点
    --------
    * 分词用 `tokenizers` 直接读模型的 ``tokenizer.json``——BERT 系的 normalization
      （中文按字切、英文 WordPiece、[CLS]/[SEP]）细节很多，自己实现容易出错。
    * 池化默认 **CLS**：BGE 系列的 sentence-transformers 配置就是 cls pooling，
      用 mean pooling 会明显掉点。
    * **动态 padding**：按批内最长序列补齐，而不是每批都补到 512，中文短句能快好几倍。
    * 输出 L2 归一化，与其它 provider 一致（点积即余弦）。
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        name: str = "bge-small-zh-v1.5",
        max_length: int = 512,
        batch_size: int = 16,
        pooling: str = "cls",
        threads: int = 0,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.name = name
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.pooling = (pooling or "cls").lower()
        self.threads = threads
        self.dim = 0
        self._session = None
        self._tokenizer = None
        self._input_names: set[str] = set()
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.model_dir.exists():
            raise RuntimeError(
                f"ONNX 模型目录不存在：{self.model_dir}\n"
                "先运行：python scripts/fetch_model.py"
            )
        try:
            import onnxruntime as ort  # noqa: PLC0415
            from tokenizers import Tokenizer  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - 取决于环境
            raise RuntimeError(
                "缺少可选依赖 onnxruntime / tokenizers：\n"
                "  pip install onnxruntime tokenizers\n"
                "（或改用 RAGKB_EMBED_PROVIDER=openai 调云端 embedding）"
            ) from e

        onnx_file = self._pick_onnx_file()
        so = ort.SessionOptions()
        if self.threads:
            so.intra_op_num_threads = self.threads
        self._session = ort.InferenceSession(
            str(onnx_file), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        tokenizer_file = self.model_dir / "tokenizer.json"
        if not tokenizer_file.exists():
            raise RuntimeError(f"缺少 tokenizer.json：{tokenizer_file}")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self._tokenizer.enable_truncation(max_length=self.max_length)
        self._tokenizer.enable_padding()  # 动态 padding：补齐到批内最长

        out_shape = self._session.get_outputs()[0].shape
        if len(out_shape) == 3 and isinstance(out_shape[-1], int):
            self.dim = int(out_shape[-1])

    def _pick_onnx_file(self) -> Path:
        for rel in ("onnx/model_quantized.onnx", "onnx/model_fp16.onnx", "onnx/model.onnx",
                    "model_quantized.onnx", "model_fp16.onnx", "model.onnx"):
            p = self.model_dir / rel
            if p.exists():
                return p
        raise RuntimeError(
            f"在 {self.model_dir} 下找不到 ONNX 模型文件；"
            "期望 onnx/model.onnx 或 model_quantized.onnx"
        )

    # ------------------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        import numpy as np  # onnxruntime 的依赖，必然存在

        out: list[Vector] = []
        items = [t if (t or "").strip() else " " for t in texts]
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            encodings = self._tokenizer.encode_batch(batch)
            feed: dict[str, Any] = {}
            if "input_ids" in self._input_names:
                feed["input_ids"] = np.array([e.ids for e in encodings], dtype=np.int64)
            if "attention_mask" in self._input_names:
                feed["attention_mask"] = np.array(
                    [e.attention_mask for e in encodings], dtype=np.int64
                )
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.array(
                    [e.type_ids for e in encodings], dtype=np.int64
                )
            hidden = self._session.run(None, feed)[0]  # (B, T, H)

            if self.pooling == "mean":
                mask = feed["attention_mask"][..., None].astype("float32")
                vecs = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            else:  # CLS：BGE 系列的默认池化
                vecs = hidden[:, 0, :]

            norm = np.linalg.norm(vecs, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            vecs = vecs / norm
            if not self.dim:
                self.dim = int(vecs.shape[1])
            out.extend(v.tolist() for v in vecs)
        return out


# --------------------------------------------------------------------------
def build_embedder(cfg: Settings) -> Embedder:
    provider = (cfg.embed_provider or "hashing").lower()
    if provider in ("hashing", "lexical-fallback"):
        return HashingEmbedder(dim=cfg.embed_dim or 512)
    if provider in ("onnx", "local", "bge", "local-model"):
        return OnnxEmbedder(
            cfg.onnx_model_dir,
            name=cfg.embed_model if cfg.embed_model != "hashing-512" else "bge-small-zh-v1.5",
            max_length=cfg.onnx_max_length,
            batch_size=max(cfg.embed_batch_size, cfg.onnx_batch_size),
            pooling=cfg.onnx_pooling,
            threads=cfg.onnx_threads,
        )
    if provider in ("openai", "openai-compatible", "compat"):
        if not cfg.embed_api_key:
            raise RuntimeError("embed_provider=openai 但未设置 RAGKB_EMBED_API_KEY")
        return OpenAICompatEmbedder(
            cfg.embed_model,
            base_url=cfg.embed_base_url,
            api_key=cfg.embed_api_key,
            batch_size=cfg.embed_batch_size,
            timeout=cfg.embed_timeout_s,
            dim=cfg.embed_dim if cfg.embed_dim != 512 else 0,
        )
    raise ValueError(f"未知 embed_provider: {cfg.embed_provider}")
