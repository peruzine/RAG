"""测试公共工具。

两个关键约束：
1. **不联网**：默认用 hashing embedder + scripted LLM，任何环境都能跑，单测秒级完成。
2. **不用系统临时目录，也不用 ``tempfile.mkdtemp``**：容器/沙箱里
   ``tempfile.gettempdir()`` 常常不可写；而 ``mkdtemp`` 创建的目录在 Windows 受限
   令牌（沙箱）下**缺少可写 ACE**，往里写文件会直接 ``WinError 5``。
   所以这里用 ``Path.mkdir`` 自己建目录，让权限从父目录正常继承。
"""

from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragkb.config import Settings  # noqa: E402
from ragkb.rag import RagApp  # noqa: E402

TEST_TMP_ROOT = ROOT / "data" / "_tests"

FIXTURE_DOC = """# 差旅报销标准

## 住宿标准

一线城市每晚上限 700 元，新一线城市 500 元，其他城市 350 元。
超标部分由个人承担。

## 餐补标准

出差期间餐补为 120 元/天，当日往返不享受餐补。

## 交通标准

单程飞行时间超过 5 小时的红眼航班可申请公务舱，需要副总裁审批。

| 舱位 | 适用职级 |
| --- | --- |
| 经济舱 | 全体 |
| 公务舱 | 总监及以上 |
"""

FIXTURE_DOC2 = """# 数据平台 SLA

## 响应时限

P0 故障首次响应 15 分钟，解决时限 4 小时，状态每 30 分钟更新一次。
P1 故障首次响应 30 分钟，解决时限 1 个工作日。
"""

SECRET_DOC = """# 薪酬带宽（内部）

## 带宽

P6 职级中位值 70 万元/年，带宽下限 56.0 万元，带宽上限 84.0 万元。
"""


def make_settings(name: str, **overrides) -> Settings:
    """在项目内创建一个隔离的临时工作目录（沙箱友好，见模块 docstring）。"""
    base = TEST_TMP_ROOT / f"{name}-{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    cfg = Settings.load(env={})
    cfg.data_dir = base
    cfg.db_path = base / "test.sqlite3"
    cfg.corpus_dir = base / "corpus"
    cfg.trace_path = base / "traces.jsonl"
    cfg.feedback_path = base / "feedback.jsonl"
    cfg.eval_dir = ROOT / "eval"
    cfg.embed_provider = "hashing"
    cfg.llm_provider = "scripted"
    cfg.rerank_provider = "lexical"
    cfg.ensure_dirs()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def build_app(name: str, *, docs: dict[str, str] | None = None, **overrides) -> tuple[RagApp, Settings]:
    """建一个已入库固定语料的应用实例。"""
    cfg = make_settings(name, **overrides)
    app = RagApp.build(cfg)
    payload = docs if docs is not None else {"差旅报销标准.md": FIXTURE_DOC, "数据平台SLA.md": FIXTURE_DOC2}
    for filename, body in payload.items():
        path = cfg.corpus_dir / filename
        path.write_text(body, encoding="utf-8")
        app.ingest(path, acl=("public",))
    return app, cfg


class AppTestCase(unittest.TestCase):
    """带自动清理的基类。"""

    app: RagApp
    cfg: Settings

    def setUp(self) -> None:
        self.app, self.cfg = build_app(type(self).__name__)

    def tearDown(self) -> None:
        try:
            self.app.close()
        finally:
            shutil.rmtree(self.cfg.data_dir, ignore_errors=True)


def cleanup_root() -> None:
    shutil.rmtree(TEST_TMP_ROOT, ignore_errors=True)
