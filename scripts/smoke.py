"""端到端冒烟脚本：不联网、不配密钥即可完整跑通检索→生成→引用。

用法::

    python scripts/smoke.py            # 用临时数据库，不会污染 data/ragkb.sqlite3
    python scripts/smoke.py --real     # 用 .env 配置的真实模型与正式库

它同时是一份"可执行的架构说明"：每一步打印的中间结果，就是排查线上问题时要看的东西。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragkb.config import Settings  # noqa: E402
from ragkb.ingest import IngestSummary  # noqa: E402
from ragkb.rag import RagApp  # noqa: E402

QUESTIONS = [
    "试用期最长可以多久？",
    "P0 故障的首次响应时间和解决时限是多少？",
    "出差住宿一线城市每晚上限是多少？",
    "部署知识库平台对 Python 版本有什么要求？",
    "公司年会在哪里举办？",  # 语料里没有 → 应该拒答
    "它多久能修好？",  # 指代不清，用于观察多轮改写
]


def main() -> int:
    ap = argparse.ArgumentParser(description="ragkb 冒烟测试")
    ap.add_argument("--real", action="store_true", help="使用 .env 中的真实模型配置")
    ap.add_argument("--keep", action="store_true", help="保留临时数据库")
    args = ap.parse_args()

    cfg = Settings.load()
    sandbox: Path | None = None
    if not args.real:
        # 用项目内的临时目录而不是系统 temp：容器/沙箱里系统 temp 常常不可写
        sandbox = cfg.project_root / "data" / "_smoke"
        if sandbox.exists():
            shutil.rmtree(sandbox, ignore_errors=True)
        sandbox.mkdir(parents=True, exist_ok=True)
        cfg.db_path = sandbox / "smoke.sqlite3"
        cfg.trace_path = sandbox / "traces.jsonl"
        cfg.feedback_path = sandbox / "feedback.jsonl"
        cfg.ensure_dirs()

    print("=" * 78)
    print("配置：", cfg.describe())
    print("=" * 78)

    app = RagApp.build(cfg)
    try:
        for warn in app.warnings():
            print("⚠ ", warn)

        public = [p for p in cfg.corpus_dir.rglob("*.md") if "restricted" not in p.parts]
        restricted = [p for p in (cfg.corpus_dir / "restricted").rglob("*.md")]

        summary = IngestSummary()
        for path in public:
            summary.results.extend(app.ingest(path, acl=("public",)).results)
        # restricted 目录单独用 internal 标签入库，演示行级权限
        for path in restricted:
            summary.results.extend(app.ingest(path, acl=("internal",)).results)
        print(f"[入库] {summary.summary()}（public={len(public)} restricted={len(restricted)}）")
        for r in summary.results:
            flag = {"added": "+", "updated": "~", "skipped": "=", "failed": "!"}.get(r.status, "?")
            print(f"   {flag} {r.status:8s} chunks={r.n_chunks:<4} {Path(r.source).name} {r.error}")

        # 再次入库应全部 skipped，验证增量幂等
        again = IngestSummary()
        for path in public + restricted:
            again.results.extend(app.ingest(path, acl=("public",)).results)
        print(f"[增量] 重复入库 → {again.summary()}（期望全部 skipped，chunks=0）")

        print("\n" + "=" * 78)
        print("检索+生成")
        print("=" * 78)
        for q in QUESTIONS:
            ans = app.ask(q)
            d = ans.debug["retrieval"]
            print(f"\nQ: {q}")
            print(
                f"   route={ans.route} refused={ans.refused} "
                f"latency={ans.latency_ms:.0f}ms cost=${ans.cost_usd:.6f}"
            )
            print(
                f"   召回: bm25={d['lexical_hits']} vector={d['vector_hits']} "
                f"fused={d['fused']} returned={d['returned']} "
                f"only={d['channels_only']} best={d['best_score']}"
            )
            print(f"   阶段耗时(ms): {d['timings_ms']}")
            print(f"   A: {ans.text[:200]}")
            for c in ans.citations[:3]:
                print(f"      [{c.index}] {c.title} › {c.heading_path}  «{c.quote[:60]}»")
            if ans.debug["invalid_citations"]:
                print(f"      ⚠ 越界引用: {ans.debug['invalid_citations']}")

        print("\n" + "=" * 78)
        print("权限过滤（ACL）")
        print("=" * 78)
        q = "P6 职级的薪酬带宽是多少？"
        for roles in [("public",), ("public", "internal"), None]:
            ans = app.ask(q, roles=roles)
            label = "不过滤" if roles is None else f"roles={roles}"
            print(f"  {label:24s} → refused={ans.refused} 命中={len(ans.retrieved)}  {ans.text[:70]}")

        print("\n" + "=" * 78)
        print("指标")
        print("=" * 78)
        import json

        print(json.dumps(app.metrics()["traces"], ensure_ascii=False, indent=2))
        print(json.dumps(app.stats(), ensure_ascii=False, indent=2))
    finally:
        app.close()
        if sandbox is not None and not args.keep:
            shutil.rmtree(sandbox, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
