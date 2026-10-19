"""配置对比评测：一次跑多个 RAG 配置，输出可对比的指标表。

这是"要不要换 embedding / 要不要开 rerank / 接大模型值不值"这类决策的**唯一可靠依据**。
体感判断在 RAG 上极不可靠：换个模型常常是"某些 case 变好、另一些变差"，只看几个例子必然误判。

用法::

    python scripts/compare_configs.py                    # 跑所有可用配置
    python scripts/compare_configs.py --only baseline,onnx
    python scripts/compare_configs.py --list
    python scripts/compare_configs.py --k 5 --out eval/reports

设计要点
--------
* **每个配置用独立的数据库与目录**：换 embedding 必须重建索引，混在同一个库里
  新旧向量不在同一空间，指标会毫无意义。
* **需要 key 的配置自动跳过**并给出说明，而不是报错中断——本地能跑的先跑完。
* 报告同时给**整体指标**和**拒答口径**（误拒率/误答率）：企业知识库里这两者
  的取舍比单一准确率重要得多。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragkb.config import Settings  # noqa: E402
from ragkb.evaluate import load_dataset, run_eval  # noqa: E402
from ragkb.metrics import CaseResult  # noqa: E402
from ragkb.rag import RagApp  # noqa: E402

ONNX_DIR = ROOT / "models" / "bge-small-zh-v1.5"


@dataclass
class Profile:
    key: str
    title: str
    overrides: dict = field(default_factory=dict)
    requires: str = ""  # "onnx" 或 "llm_key"，用于跳过不可用配置

    def available(self) -> tuple[bool, str]:
        if self.requires == "onnx":
            if not ONNX_DIR.exists():
                return False, f"模型目录不存在（先跑 scripts/fetch_model.py）：{ONNX_DIR}"
            try:
                import onnxruntime  # noqa: F401
            except Exception as e:
                return False, f"onnxruntime 不可用（pip install onnxruntime tokenizers）：{e}"
        if self.requires == "llm_key":
            cfg = Settings.load()
            if not cfg.llm_api_key:
                return False, "未配置 RAGKB_LLM_API_KEY"
        return True, ""


PROFILES: list[Profile] = [
    Profile(
        "baseline",
        "离线基线：hashing 向量 + 抽取式生成 + 词法重排",
        dict(embed_provider="hashing", llm_provider="scripted", rerank_provider="lexical"),
    ),
    Profile(
        "baseline-norerank",
        "离线基线（关掉重排）——用来看重排贡献了多少",
        dict(embed_provider="hashing", llm_provider="scripted", rerank_provider="none"),
    ),
    Profile(
        "onnx",
        "本地语义 embedding（bge-small-zh）+ 词法重排，门禁仍为纯词法",
        dict(
            embed_provider="onnx",
            embed_model="bge-small-zh-v1.5",
            embed_dim=512,
            onnx_model_dir=ONNX_DIR,
            llm_provider="scripted",
            rerank_provider="lexical",
        ),
        requires="onnx",
    ),
    Profile(
        "onnx-semantic-gate",
        "本地语义 embedding + **语义拒答豁免**（推荐组合）",
        dict(
            embed_provider="onnx",
            embed_model="bge-small-zh-v1.5",
            embed_dim=512,
            onnx_model_dir=ONNX_DIR,
            llm_provider="scripted",
            rerank_provider="lexical",
            refuse_vector_similarity_floor=0.58,
        ),
        requires="onnx",
    ),
    Profile(
        "onnx-norerank",
        "本地语义 embedding，关掉词法重排——用于排查「词法重排是否在和语义召回打架」",
        dict(
            embed_provider="onnx",
            embed_model="bge-small-zh-v1.5",
            embed_dim=512,
            onnx_model_dir=ONNX_DIR,
            llm_provider="scripted",
            rerank_provider="none",
            refuse_vector_similarity_floor=0.58,
        ),
        requires="onnx",
    ),
    Profile(
        "onnx-llm",
        "本地语义 embedding + 真实生成模型（需 API Key）",
        dict(
            embed_provider="onnx",
            embed_model="bge-small-zh-v1.5",
            embed_dim=512,
            onnx_model_dir=ONNX_DIR,
            llm_provider="openai",
            rerank_provider="lexical",
            refuse_vector_similarity_floor=0.58,
        ),
        requires="llm_key",
    ),
]


def build_settings(profile: Profile, workdir: Path) -> Settings:
    cfg = Settings.load()
    base = workdir / profile.key
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    cfg.data_dir = base
    cfg.db_path = base / "ragkb.sqlite3"
    cfg.corpus_dir = ROOT / "data" / "corpus"
    cfg.trace_path = base / "traces.jsonl"
    cfg.feedback_path = base / "feedback.jsonl"
    cfg.eval_dir = ROOT / "eval"
    cfg.ensure_dirs()
    for k, v in profile.overrides.items():
        setattr(cfg, k, v)
    return cfg


def refusal_breakdown(results: list[CaseResult], cases_by_id: dict) -> dict[str, float]:
    """误拒率 / 误答率——比单一的 refusal_accuracy 更能说明取舍。"""
    answerable = [r for r in results if not cases_by_id[r.case_id].should_refuse]
    negative = [r for r in results if cases_by_id[r.case_id].should_refuse]
    false_refuse = sum(1 for r in answerable if r.refused) / max(1, len(answerable))
    false_answer = sum(1 for r in negative if not r.refused) / max(1, len(negative))
    return {
        "误拒率": false_refuse,
        "误答率": false_answer,
        "可答样本": float(len(answerable)),
        "应拒样本": float(len(negative)),
    }


def run_profile(profile: Profile, workdir: Path, k: int) -> dict:
    cfg = build_settings(profile, workdir)
    app = RagApp.build(cfg)
    try:
        # 入库：public 全量 + restricted 单独打 internal 标签（与真实用法一致）
        docs = [p for p in cfg.corpus_dir.rglob("*.md") if "restricted" not in p.parts]
        restricted = list((cfg.corpus_dir / "restricted").rglob("*.md"))
        chunks = 0
        t0 = time.perf_counter()
        for path in docs:
            chunks += app.ingest(path, acl=("public",), force=True).chunks
        for path in restricted:
            chunks += app.ingest(path, acl=("internal",), force=True).chunks
        ingest_s = time.perf_counter() - t0

        cases = load_dataset(cfg.eval_dir / "dataset.jsonl")
        report = run_eval(app, cases, cfg, k=k, max_drop=0)
        by_id = {c.case_id: c for c in cases}
        breakdown = refusal_breakdown(report.cases, by_id)

        # 检索阶段 p95：来自 trace，比整体耗时更能反映"换 embedding 的代价"
        stage = {}
        try:
            from ragkb.obs import read_jsonl, percentile

            rows = read_jsonl(cfg.trace_path)
            stage = {
                "retrieve_p95_ms": round(
                    percentile(
                        [
                            next((s["ms"] for s in r.get("spans", []) if s["name"] == "retrieve"), 0.0)
                            for r in rows
                            if r.get("name") == "ask"
                        ],
                        0.95,
                    ),
                    1,
                ),
                "generate_p95_ms": round(
                    percentile(
                        [
                            next((s["ms"] for s in r.get("spans", []) if s["name"] == "generate"), 0.0)
                            for r in rows
                            if r.get("name") == "ask"
                        ],
                        0.95,
                    ),
                    1,
                ),
            }
        except Exception:
            pass
        try:
            cost = report.overall.get("cost_usd_avg")
        except Exception:
            cost = None

        return {
            "key": profile.key,
            "title": profile.title,
            "overall": report.overall,
            "by_tag": {t: dict(m) for t, m in report.by_tag.items()},
            "breakdown": breakdown,
            "stage": stage,
            "chunks": chunks,
            "ingest_s": round(ingest_s, 2),
            "embedder": app.embedder.name,
            "llm": app.llm.name,
            "reranker": app.reranker.name,
            "cost_avg": cost,
            "failed_cases": [r.case_id for r in report.cases if r.refused and not by_id[r.case_id].should_refuse],
            "gate": report.gate,
        }
    finally:
        app.close()


METRIC_COLS = [
    ("hit@k", "hit@5", "{:.3f}"),
    ("recall@k", "recall@5", "{:.3f}"),
    ("mrr", "MRR", "{:.3f}"),
    ("ndcg@k", "nDCG@5", "{:.3f}"),
    ("refusal_accuracy", "拒答准确", "{:.3f}"),
    ("answer_f1", "答案F1", "{:.3f}"),
    ("groundedness", "有据率", "{:.3f}"),
    ("citation_validity", "引用有效", "{:.3f}"),
]


def render_markdown(results: list[dict], skipped: list[tuple[str, str]], k: int) -> str:
    lines: list[str] = []
    lines.append("# RAG 配置对比评测\n")
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　评测集：`eval/dataset.jsonl`　top_k={k}\n")
    header = "| 配置 | " + " | ".join(label for _, label, _ in METRIC_COLS)
    header += " | 误拒率 | 误答率 | 检索p95 | 入库耗时 | 切块 |"
    sep = "|" + "---|" * (len(METRIC_COLS) + 7)
    lines.append(header)
    lines.append(sep)
    for r in results:
        cells = []
        for key, _, fmt in METRIC_COLS:
            v = r["overall"].get(key)
            cells.append(fmt.format(v) if isinstance(v, (int, float)) else "-")
        row = f"| **{r['key']}** | " + " | ".join(cells)
        row += f" | {r['breakdown']['误拒率']:.3f} | {r['breakdown']['误答率']:.3f}"
        row += f" | {r['stage'].get('retrieve_p95_ms', '-')}ms"
        row += f" | {r['ingest_s']}s | {r['chunks']} |"
        lines.append(row)

    # 改写提问单独一张表：这是语义 embedding 唯一能体现价值、词法检索必然吃亏的场景。
    # 只看整体均值会把这个问题掩盖掉（整体里大多数样本的问句都贴近文档原文）。
    if any("paraphrase" in (r.get("by_tag") or {}) for r in results):
        lines.append("\n## 分标签：`paraphrase`（用户按自己的说法提问，不照抄文档）\n")
        lines.append("| 配置 | hit@k | recall@k | MRR | nDCG@k | 拒答准确 |")
        lines.append("|---|---|---|---|---|---|")
        for r in results:
            m = (r.get("by_tag") or {}).get("paraphrase")
            if not m:
                lines.append(f"| **{r['key']}** | - | - | - | - | - |")
                continue
            lines.append(
                f"| **{r['key']}** | {m.get('hit@k', 0):.3f} | {m.get('recall@k', 0):.3f} | "
                f"{m.get('mrr', 0):.3f} | {m.get('ndcg@k', 0):.3f} | "
                f"{m.get('refusal_accuracy', 0):.3f} |"
            )
        lines.append(
            "\n> 改写提问下，纯词法召回会明显掉点：问句与文档几乎不共享词条，"
            "BM25 只能靠偶然重合的字，而语义模型能靠意思对上。这就是「要不要上真实 embedding」的决策依据。"
        )

    lines.append("\n## 配置说明\n")
    for r in results:
        lines.append(f"- **{r['key']}**：{r['title']}")
        lines.append(f"  - embedder=`{r['embedder']}`　llm=`{r['llm']}`　reranker=`{r['reranker']}`")
        if r["failed_cases"]:
            lines.append(f"  - 被误拒的可答样本（{len(r['failed_cases'])}）：`{'`, `'.join(r['failed_cases'])}`")
    if skipped:
        lines.append("\n## 未执行的配置\n")
        for key, reason in skipped:
            lines.append(f"- `{key}`：{reason}")

    lines.append("\n## 怎么读这张表\n")
    lines.append(
        "- **误答率**（语料里没有却作答）比误拒率更伤信任，优先把它压到 0；\n"
        "- `hit@5/recall@5` 看「找得到吗」，`MRR/nDCG` 看「排得靠前吗」，两者要一起看；\n"
        "- 换 embedding 后必须重建索引（本脚本每个配置独立建库，所以数据可比）；\n"
        "- `检索p95` 是本地检索与编码的开销，不含生成模型延迟。"
    )
    return "\n".join(lines)


def main() -> int:
    from ragkb.console import enable_utf8_console

    enable_utf8_console()  # GBK 控制台画不出 ✓，不切会直接崩
    ap = argparse.ArgumentParser(description="多配置对比评测")
    ap.add_argument("--only", default="", help="只跑指定配置，逗号分隔")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--list", action="store_true", help="列出配置与可用性")
    ap.add_argument("--out", default=str(ROOT / "eval" / "reports"))
    ap.add_argument("--keep", action="store_true", help="保留各配置的数据库")
    args = ap.parse_args()

    selected = [
        p for p in PROFILES if not args.only or p.key in {x.strip() for x in args.only.split(",")}
    ]

    if args.list:
        print(f"{'配置':<22}{'可用':<8}说明")
        for p in selected:
            ok, why = p.available()
            print(f"{p.key:<22}{'✓' if ok else '✗':<8}{p.title}")
            if not ok:
                print(f"{'':<30}→ {why}")
        return 0

    workdir = ROOT / "data" / "_compare"
    results: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for p in selected:
        ok, why = p.available()
        if not ok:
            skipped.append((p.key, why))
            print(f"⏭  跳过 {p.key}：{why}")
            continue
        print(f"\n{'='*72}\n▶ 运行配置：{p.key} — {p.title}\n{'='*72}")
        try:
            r = run_profile(p, workdir, args.k)
        except Exception as e:
            skipped.append((p.key, f"{type(e).__name__}: {e}"))
            print(f"✗  {p.key} 失败：{type(e).__name__}: {e}")
            continue
        results.append(r)
        o = r["overall"]
        print(
            f"   hit@5={o.get('hit@k', 0):.3f}  recall@5={o.get('recall@k', 0):.3f}  "
            f"MRR={o.get('mrr', 0):.3f}  拒答准确={o.get('refusal_accuracy', 0):.3f}  "
            f"误拒={r['breakdown']['误拒率']:.3f}  误答={r['breakdown']['误答率']:.3f}"
        )
        if r["failed_cases"]:
            print(f"   被误拒的可答样本：{', '.join(r['failed_cases'])}")

    if not results:
        print("\n没有可执行的配置。先跑 scripts/fetch_model.py 或配置 RAGKB_LLM_API_KEY。")
        return 2

    md = render_markdown(results, skipped, args.k)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (outdir / f"compare-{stamp}.md").write_text(md, encoding="utf-8")
    (outdir / f"compare-{stamp}.json").write_text(
        json.dumps({"results": results, "skipped": skipped, "k": args.k}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (outdir / "compare-latest.md").write_text(md, encoding="utf-8")
    print("\n" + "=" * 72)
    print(md.split("## 配置说明")[0])
    print(f"报告：{outdir / f'compare-{stamp}.md'}")

    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
