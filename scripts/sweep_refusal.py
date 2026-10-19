"""拒答门禁阈值扫描。

``RAGKB_REFUSE_COVERAGE_FLOOR`` 直接决定"误答"与"误拒"的平衡：调高更保守
（宁可说不知道），调低更激进（尽量回答）。企业知识库里误答的代价通常远大于误拒，
所以这个阈值应该用数据定，而不是拍脑袋。

::

    python scripts/sweep_refusal.py
    python scripts/sweep_refusal.py --values 0,0.05,0.1,0.15,0.2

实现要点：检索与生成**只跑一次**，然后对每个阈值只重新判定覆盖率门禁，
并区分"是门禁拒的"还是"模型自己说 NO_ANSWER 的"——后者不应该随阈值变化。
这样一次扫描的耗时等于一次评测，而不是 N 次。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragkb.config import Settings  # noqa: E402
from ragkb.evaluate import load_dataset  # noqa: E402
from ragkb.metrics import evaluate_case  # noqa: E402
from ragkb.rag import RagApp  # noqa: E402

# 与覆盖率无关的门禁原因：这些在任何阈值下都保持拒答
HARD_REASONS = {"no_hits", "too_few_hits", "low_score"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", default="0,0.05,0.08,0.1,0.12,0.15,0.2,0.25,0.3")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    cfg = Settings.load()
    cases = load_dataset(cfg.eval_dir / "dataset.jsonl")
    floors = [float(v) for v in args.values.split(",") if v.strip()]

    app = RagApp.build(cfg)
    try:
        # 一次跑完，记录：检索结果、查询、模型是否自己拒答、硬门禁原因
        runs = []
        for case in cases:
            ans = app.ask(case.question, roles=case.roles, history=case.history or None, top_k=args.k)
            reason = ans.debug.get("refuse_reason")
            model_refused = ans.refused and reason is None
            runs.append(
                {
                    "case": case,
                    "ans": ans,
                    "query": ans.debug["retrieval"]["query"],
                    "model_refused": model_refused,
                    "hard": reason in HARD_REASONS,
                }
            )

        print(
            f"{'floor':>7}{'拒答准确率':>12}{'误拒率':>10}{'误答率':>10}"
            f"{'recall@k':>11}{'mrr':>8}  被误拒的样本"
        )
        print("-" * 104)
        for floor in floors:
            rows = []
            for run in runs:
                case, ans = run["case"], run["ans"]
                gate_refused = run["hard"]
                if floor > 0 and ans.retrieved:
                    cov, _, _ = app.answerer.effective_coverage(
                        run["query"], ans.retrieved, run["case"].question
                    )
                    if cov < floor:
                        gate_refused = True
                refused = gate_refused or run["model_refused"]
                rows.append(
                    evaluate_case(
                        case,
                        ans.retrieved,
                        answer_text=ans.text,
                        refused=refused,
                        invalid_citations=ans.debug.get("invalid_citations", []),
                        latency_ms=ans.latency_ms,
                        k=args.k,
                    )
                )

            answerable = [r for r, run in zip(rows, runs) if not run["case"].should_refuse]
            negatives = [r for r, run in zip(rows, runs) if run["case"].should_refuse]
            frr = sum(1 for r in answerable if r.refused) / max(1, len(answerable))
            far = sum(1 for r in negatives if not r.refused) / max(1, len(negatives))
            acc = sum(r.metrics["refusal_accuracy"] for r in rows) / len(rows)
            gold = [r for r in rows if r.has_gold]
            recall = sum(r.metrics["recall@k"] for r in gold) / max(1, len(gold))
            mrr = sum(r.metrics["mrr"] for r in gold) / max(1, len(gold))
            missed = ",".join(r.case_id for r in answerable if r.refused)[:40]
            print(
                f"{floor:>7.2f}{acc:>12.3f}{frr:>10.3f}{far:>10.3f}"
                f"{recall:>11.3f}{mrr:>8.3f}  {missed}"
            )
        print(
            "\n提示：误答率（把语料里没有的东西编出来）比误拒率更伤信任，"
            "建议取误答率为 0 的前提下误拒率最低的阈值。"
        )
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
