"""命令行入口：``python -m ragkb.cli <command>``。

命令设计围绕"一个人能不能独立把 RAG 系统跑起来并运营起来"：
``demo/ingest`` 建库 → ``ask`` 验证 → ``check`` 体检 → ``eval`` 度量 → ``serve`` 上线
→ ``stats`` 看线上表现 → ``docs/prune`` 运维。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import Settings
from .console import enable_utf8_console
from .evaluate import (
    load_dataset,
    run_eval,
    sanity_check_dataset,
    save_report,
    update_baseline,
)
from .rag import RagApp
from .text import truncate


def _print(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _roles(raw: str | None, cfg: Settings) -> tuple[str, ...] | None:
    if raw is None:
        return tuple(cfg.default_roles)
    if raw.strip().lower() in ("none", "all", "*"):
        return None
    return tuple(p.strip() for p in raw.split(",") if p.strip())


# --------------------------------------------------------------------------
def cmd_ingest(args: argparse.Namespace, cfg: Settings) -> int:
    app = RagApp.build(cfg)
    try:
        for warn in app.warnings():
            print(f"⚠  {warn}", file=sys.stderr)
        summary = app.ingest(
            args.path, acl=tuple(args.acl.split(",")) if args.acl else (), force=args.force
        )
        for r in summary.results:
            mark = {"added": "+", "updated": "~", "skipped": "=", "failed": "!"}.get(r.status, "?")
            line = f"{mark} {r.status:8s} chunks={r.n_chunks:<4} {r.source}"
            if r.report:
                line += f"  [{r.report}]"
            if r.error:
                line += f"  ERROR: {r.error}"
            print(line)
        print(f"\n{summary.summary()}")
        if args.prune:
            removed = app.store.prune_postings(args.prune_ratio)
            print(f"prune: 移除 {removed} 个高频词（占索引空间但不贡献区分度）")
        print(json.dumps(app.store.stats(), ensure_ascii=False, indent=2))
        return 1 if summary.failed and not summary.added and not summary.updated else 0
    finally:
        app.close()


def cmd_ask(args: argparse.Namespace, cfg: Settings) -> int:
    app = RagApp.build(cfg)
    try:
        roles = _roles(args.roles, cfg)
        if args.stream:
            final: dict[str, Any] = {}
            for event in app.stream(args.question, roles=roles, top_k=args.top_k):
                if event["type"] == "meta":
                    if args.show_context:
                        print("── 检索到的上下文 ──")
                        for c in event["contexts"]:
                            print(f"  [{c['index']}] score={c['score']:.4f} {c['title']} › {c['heading_path']}")
                            print(f"      {truncate(c['snippet'], 100)}")
                        print(f"  检索耗时(ms): {event['retrieval']['timings_ms']}")
                        print("── 回答 ──")
                elif event["type"] == "delta":
                    sys.stdout.write(event["text"])
                    sys.stdout.flush()
                else:
                    final = event["answer"]
            print()
            if final:
                print(
                    f"\n[refused={final['refused']} latency={final['latency_ms']}ms "
                    f"cost=${final['cost_usd']:.6f} trace={final['trace_id']}]"
                )
            return 0

        ans = app.ask(args.question, roles=roles, top_k=args.top_k)
        if args.json:
            _print(ans.to_dict())
            return 0

        print(ans.text)
        if ans.citations:
            print("\n引用：")
            for c in ans.citations:
                print(f"  [{c.index}] {c.title} › {c.heading_path}  «{truncate(c.quote, 80)}»")
        if args.show_context:
            print("\n召回明细（排查问题时先看这里）：")
            for i, r in enumerate(ans.retrieved, start=1):
                print(
                    f"  [{i}] score={r.score:.4f} rerank={r.rerank_score} "
                    f"channels={r.channels} {r.chunk.doc_id}#{r.chunk.ordinal} "
                    f"{r.chunk.heading_path}"
                )
            print(f"  检索调试: {json.dumps(ans.debug['retrieval'], ensure_ascii=False)}")
        print(
            f"\n[route={ans.route} refused={ans.refused} latency={ans.latency_ms:.0f}ms "
            f"tokens={ans.usage.total_tokens} cost=${ans.cost_usd:.6f} trace={ans.trace_id}]"
        )
        return 0
    finally:
        app.close()


def cmd_serve(args: argparse.Namespace, cfg: Settings) -> int:
    from .server import serve

    return serve(cfg, host=args.host, port=args.port)


def cmd_stats(args: argparse.Namespace, cfg: Settings) -> int:
    app = RagApp.build(cfg)
    try:
        data = app.metrics() if args.online else {"corpus": app.stats()}
        if args.json:
            _print(data)
        else:
            print("── 语料 ──")
            _print(data["corpus"])
            if args.online:
                print("── 线上指标（来自 trace/feedback）──")
                _print(data["traces"])
                _print(data["feedback"])
        return 0
    finally:
        app.close()


def cmd_docs(args: argparse.Namespace, cfg: Settings) -> int:
    app = RagApp.build(cfg)
    try:
        if args.action == "list":
            docs = app.store.list_documents()
            if args.json:
                _print(docs)
            else:
                print(f"{'doc_id':<22}{'chunks':>7}  {'acl':<18}title")
                for d in docs:
                    print(
                        f"{d['doc_id']:<22}{d['n_chunks']:>7}  "
                        f"{','.join(d['acl']):<18}{d['title']}"
                    )
            return 0
        if args.action == "delete":
            if not args.doc_id:
                print("需要 --doc-id（或用 --source 指定来源路径）", file=sys.stderr)
                return 2
            ok = app.store.delete_document(args.doc_id)
            print("已删除" if ok else "未找到该文档")
            return 0 if ok else 1
        print(f"未知操作：{args.action}", file=sys.stderr)
        return 2
    finally:
        app.close()


def cmd_eval(args: argparse.Namespace, cfg: Settings) -> int:
    dataset = Path(args.dataset) if args.dataset else cfg.eval_dir / "dataset.jsonl"
    cases = load_dataset(dataset)
    if args.tags:
        wanted = {t.strip() for t in args.tags.split(",") if t.strip()}
        cases = [c for c in cases if wanted & set(c.tags)]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("没有可执行的评测样本", file=sys.stderr)
        return 2

    app = RagApp.build(cfg)
    try:
        warnings = app.warnings()
        for w in warnings:
            print(f"⚠  {w}", file=sys.stderr)

        if args.check:
            problems = sanity_check_dataset(cases, app)
            if problems:
                print("评测集自检发现问题：", file=sys.stderr)
                for p in problems:
                    print(f"  - {p}", file=sys.stderr)
                return 3
            print(f"评测集自检通过（{len(cases)} 条）")

        report = run_eval(
            app,
            cases,
            cfg,
            k=args.k,
            retrieval_only=args.retrieval_only,
            baseline_path=Path(args.baseline) if args.baseline else cfg.eval_dir / "baseline.json",
            thresholds_path=Path(args.thresholds) if args.thresholds else cfg.eval_dir / "thresholds.json",
            max_drop=args.max_drop,
        )
        print(report.render())

        out = Path(args.out) if args.out else cfg.eval_dir / "reports" / "latest.json"
        saved = save_report(report, out)
        print(f"\n报告已写入：{saved}")

        if args.update_baseline:
            bpath = update_baseline(
                report, Path(args.baseline) if args.baseline else cfg.eval_dir / "baseline.json"
            )
            print(f"基线已更新：{bpath}")
            return 0

        if args.verbose:
            print("\n逐条明细：")
            for r in report.cases:
                d = r.to_dict()
                print(
                    f"  {'✓' if not d['invalid_citations'] else '⚠'} {d['case_id']:<24} "
                    f"hit={d['metrics'].get('hit@k')} mrr={d['metrics'].get('mrr')} "
                    f"refused={d['refused']} retrieval={d['n_retrieved']}"
                )
                print(f"      top: {d['top_chunk']}  |  {truncate(d['answer_preview'], 90)}")
        return 0 if report.gate["passed"] else 1
    finally:
        app.close()


def cmd_demo(args: argparse.Namespace, cfg: Settings) -> int:
    """一条命令把示例语料入库（含 internal 权限演示），供快速体验。"""
    app = RagApp.build(cfg)
    try:
        corpus = Path(args.corpus) if args.corpus else cfg.corpus_dir
        public = [p for p in corpus.rglob("*") if p.is_file() and "restricted" not in p.parts]
        restricted = [p for p in corpus.rglob("*") if p.is_file() and "restricted" in p.parts]
        total = 0
        for path in public:
            s = app.ingest(path, acl=("public",), force=args.force)
            total += s.chunks
            print(f"  public     {Path(path).name}: {s.summary()}")
        for path in restricted:
            s = app.ingest(path, acl=("internal",), force=args.force)
            total += s.chunks
            print(f"  internal   {Path(path).name}: {s.summary()}")
        print(f"\n示例语料入库完成，共 {total} 个切块。下一步：")
        print('  python -m ragkb.cli ask "试用期最长可以多久？" --show-context')
        print("  python -m ragkb.cli serve")
        return 0
    finally:
        app.close()


def cmd_check(args: argparse.Namespace, cfg: Settings) -> int:
    """体检：配置一致性 + 评测集标注是否过期 + 库内统计。"""
    app = RagApp.build(cfg)
    problems = 0
    try:
        for w in app.warnings():
            print(f"⚠  {w}")
            problems += 1
        stats = app.store.stats()
        if stats["chunks"] == 0:
            print("⚠  语料为空，先执行 demo 或 ingest")
            problems += 1
        if not stats["embedding_models"]:
            print("⚠  没有任何向量，向量检索通道会失效（只剩 BM25）")
            problems += 1
        dataset = cfg.eval_dir / "dataset.jsonl"
        if dataset.exists():
            cases = load_dataset(dataset)
            issues = sanity_check_dataset(cases, app) if stats["chunks"] else []
            for issue in issues:
                print(f"⚠  {issue}")
            problems += len(issues)
            print(f"评测集：{len(cases)} 条，自检问题 {len(issues)} 个")
        _print(app.metrics()["traces"])
        print("\n体检完成：" + ("未发现问题 ✅" if problems == 0 else f"发现 {problems} 个问题 ⚠"))
        return 0 if problems == 0 else 1
    finally:
        app.close()


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ragkb",
        description="企业知识库问答（RAG）—— 零第三方依赖的参考实现",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "典型流程：\n"
            "  python -m ragkb.cli demo                       # 入库示例语料\n"
            '  python -m ragkb.cli ask "试用期最长可以多久？"   # 提问\n'
            "  python -m ragkb.cli eval --check               # 评测 + 门禁\n"
            "  python -m ragkb.cli serve                      # 起服务（含 Web UI）\n"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="入库文件或目录（增量、幂等）")
    s.add_argument("path", help="文件或目录路径")
    s.add_argument("--acl", default="", help="权限标签，逗号分隔，如 public,internal")
    s.add_argument("--force", action="store_true", help="内容未变也强制重建")
    s.add_argument("--prune", action="store_true", help="入库后清理高频词")
    s.add_argument("--prune-ratio", type=float, default=0.5, help="高频词阈值（df 占比）")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("ask", help="提问")
    s.add_argument("question")
    s.add_argument("--roles", default=None, help="请求角色，逗号分隔；all 表示不过滤")
    s.add_argument("--top-k", type=int, default=None, help="送入模型的上下文条数")
    s.add_argument("--json", action="store_true", help="输出完整 JSON（含 trace）")
    s.add_argument("--stream", action="store_true", help="流式输出")
    s.add_argument("--show-context", action="store_true", help="打印召回明细与阶段耗时")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("serve", help="启动 HTTP 服务（含内置 Web UI）")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("stats", help="语料统计 / 线上指标")
    s.add_argument("--online", action="store_true", help="同时输出 trace 与反馈聚合")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("docs", help="文档管理")
    s.add_argument("action", choices=["list", "delete"])
    s.add_argument("--doc-id", default="")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_docs)

    s = sub.add_parser("eval", help="跑评测并执行 CI 门禁")
    s.add_argument("--dataset", default="")
    s.add_argument("--k", type=int, default=5, help="评测用的 top-k")
    s.add_argument("--tags", default="", help="只跑指定标签，逗号分隔")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--retrieval-only", action="store_true", help="只测检索，不调用生成模型")
    s.add_argument("--baseline", default="")
    s.add_argument("--thresholds", default="")
    s.add_argument("--max-drop", type=float, default=0.0, help="相对基线允许的最大跌幅（0=用阈值文件里的设置）")
    s.add_argument("--out", default="")
    s.add_argument("--update-baseline", action="store_true")
    s.add_argument("--check", action="store_true", help="先做评测集自检")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("demo", help="入库示例语料（快速体验）")
    s.add_argument("--corpus", default="")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("check", help="体检：配置一致性/评测集标注/语料统计")
    s.set_defaults(func=cmd_check)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    # 中文 Windows 的控制台是 GBK，画不出 ✓ ⚠ ✅ 这类符号，会直接抛
    # UnicodeEncodeError（退出码 1）——必须在打印任何东西之前切到 UTF-8。
    enable_utf8_console()
    args = build_parser().parse_args(argv)
    cfg = Settings.load()
    try:
        return int(args.func(args, cfg))
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"错误：{type(e).__name__}: {e}", file=sys.stderr)
        if "--debug" in (argv or sys.argv):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
