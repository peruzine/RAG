"""逐案检索诊断：回答"这个 case 到底为什么召回/排序不对"。

这是 RAG 排查的标准动作：不要只看总分，要能看到**每个通道把 gold 排在了第几、
相似度是多少、最终是谁把它挤下去的**。

用法::

    python scripts/diagnose_case.py --config onnx --case expense-payment
    python scripts/diagnose_case.py --config baseline --all --k 5
    python scripts/diagnose_case.py --config onnx --question "报销款什么时候到账？"
"""

from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragkb.console import enable_utf8_console  # noqa: E402
from ragkb.evaluate import load_dataset  # noqa: E402
from ragkb.metrics import is_relevant  # noqa: E402
from ragkb.rag import RagApp  # noqa: E402

# 与 compare_configs.py 的配置保持一致（这里只保留最常用的几个）
CONFIGS = {
    "baseline": dict(embed_provider="hashing", rerank_provider="lexical"),
    "onnx": dict(
        embed_provider="onnx",
        embed_model="bge-small-zh-v1.5",
        embed_dim=512,
        onnx_model_dir=ROOT / "models" / "bge-small-zh-v1.5",
        rerank_provider="lexical",
    ),
    "onnx-norerank": dict(
        embed_provider="onnx",
        embed_model="bge-small-zh-v1.5",
        embed_dim=512,
        onnx_model_dir=ROOT / "models" / "bge-small-zh-v1.5",
        rerank_provider="none",
    ),
}


def build(config: str) -> RagApp:
    from ragkb.config import Settings

    cfg = Settings.load()
    base = ROOT / "data" / "_diag" / config
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    cfg.data_dir = base
    cfg.db_path = base / "ragkb.sqlite3"
    cfg.corpus_dir = ROOT / "data" / "corpus"
    cfg.trace_path = base / "traces.jsonl"
    cfg.eval_dir = ROOT / "eval"
    cfg.ensure_fresh = True  # 占位，避免误用
    cfg.ensure_dirs()
    for k, v in CONFIGS[config].items():
        setattr(cfg, k, v)
    app = RagApp.build(cfg)
    for path in [p for p in cfg.corpus_dir.rglob("*.md") if "restricted" not in p.parts]:
        app.ingest(path, acl=("public",), force=True)
    for path in (cfg.corpus_dir / "restricted").rglob("*.md"):
        app.ingest(path, acl=("internal",), force=True)
    return app


def show(app: RagApp, case, k: int) -> None:
    res = app.retriever.retrieve(case.question, top_k=k, roles=case.roles)
    print(f"\n── {case.case_id}：{case.question}")
    if case.gold_snippets:
        print(f"   gold 片段：{case.gold_snippets}")
    print(f"   候选：bm25={res.debug.lexical_hits} vector={res.debug.vector_hits} "
          f"融合={res.debug.fused} 返回={res.debug.returned} "
          f"向量最高相似度={res.debug.best_vector_similarity}")
    for i, it in enumerate(res.items, start=1):
        rel = "★" if is_relevant(it.chunk, case) else " "
        vec = it.channel_scores.get("vector")
        bm = it.channel_scores.get("bm25")
        rr = it.rerank_score
        print(
            f"   {rel} [{i}] bm25_rank={it.channels.get('bm25', '-'):<3} "
            f"vec_rank={it.channels.get('vector', '-'):<3} "
            f"bm25={bm if bm is None else round(bm, 3):<8} "
            f"cos={vec if vec is None else round(vec, 3):<7} "
            f"rerank={rr if rr is None else round(rr, 3):<7} "
            f"{it.chunk.heading_path[:28]}"
        )
    hit = any(is_relevant(it.chunk, case) for it in res.items)
    print(f"   → 命中：{'是' if hit else '否（gold 不在 top-%d 内）' % k}")


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="onnx", choices=list(CONFIGS))
    ap.add_argument("--case", default="", help="case_id，逗号分隔")
    ap.add_argument("--question", default="", help="直接给一个问题")
    ap.add_argument("--all", action="store_true", help="跑评测集里所有 case")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    app = build(args.config)
    try:
        if args.question:
            from ragkb.types import EvalCase

            res = app.retriever.retrieve(args.question, top_k=args.k)
            print(f"\n── {args.question}")
            print(f"   候选：bm25={res.debug.lexical_hits} vector={res.debug.vector_hits} "
                  f"向量最高相似度={res.debug.best_vector_similarity}")
            for i, it in enumerate(res.items, start=1):
                vec = it.channel_scores.get("vector")
                bm = it.channel_scores.get("bm25")
                print(
                    f"   [{i}] bm25_rank={it.channels.get('bm25', '-'):<3} "
                    f"vec_rank={it.channels.get('vector', '-'):<3} "
                    f"bm25={bm if bm is None else round(bm, 3):<8} "
                    f"cos={vec if vec is None else round(vec, 3):<7} "
                    f"rerank={None if it.rerank_score is None else round(it.rerank_score, 3)} "
                    f"{it.chunk.heading_path[:30]}"
                )
            return 0

        cases = load_dataset(ROOT / "eval" / "dataset.jsonl")
        wanted = {c.strip() for c in args.case.split(",") if c.strip()}
        if args.all:
            selected = cases
        elif wanted:
            selected = [c for c in cases if c.case_id in wanted]
        else:
            # 默认：只展示"没命中"或"被拒"的 case（真正需要排查的那些）
            selected = cases[:0]
            print("未指定 --case / --all / --question，展示评测集中召回失败的 case：")
            for c in cases:
                res = app.retriever.retrieve(c.question, top_k=args.k, roles=c.roles)
                if c.gold_snippets and not any(is_relevant(it.chunk, c) for it in res.items):
                    selected.append(c)
            if not selected:
                print("（top-%d 内全部命中，没有需要诊断的 case）" % args.k)
                return 0
        for case in selected:
            show(app, case, args.k)
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
