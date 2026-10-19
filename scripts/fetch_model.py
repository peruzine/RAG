"""下载本地语义 embedding 模型（ONNX），供 ``RAGKB_EMBED_PROVIDER=onnx`` 使用。

为什么走 ONNX 而不是 sentence-transformers
-----------------------------------------
`sentence-transformers` 会拖进 ~2.5GB 的 torch；ONNX Runtime 只要几十 MB，
CPU 上单条几毫秒。对中小规模知识库，这是"能用真实语义检索"和"装不动"之间的关键取舍。

用法::

    python scripts/fetch_model.py                       # 默认 BAAI/bge-small-zh-v1.5（约 95MB）
    python scripts/fetch_model.py --repo BAAI/bge-base-zh-v1.5
    python scripts/fetch_model.py --list                # 只看仓库里有哪些文件

镜像：默认 ``hf-mirror.com``（HuggingFace 在国内的镜像）；huggingface.co 直连超时会自动回退。
下载支持断点续传（``.part`` 临时文件），中断后重跑即可。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 候选顺序：量化版最小最快，其次 fp16，最后 fp32
ONNX_CANDIDATES = (
    "onnx/model_quantized.onnx",
    "onnx/model_fp16.onnx",
    "onnx/model.onnx",
    "model_quantized.onnx",
    "model.onnx",
)
MIRRORS = ("https://hf-mirror.com", "https://huggingface.co")


def _open(url: str, timeout: float = 30.0):
    req = urllib.request.Request(url, headers={"User-Agent": "ragkb/0.1"})
    return urllib.request.urlopen(req, timeout=timeout)


def api_files(repo: str) -> tuple[str, list[str]]:
    """返回 (可用的镜像, 文件列表)。"""
    last_error: Exception | None = None
    for mirror in MIRRORS:
        try:
            with _open(f"{mirror}/api/models/{repo}", timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            names = [s["rfilename"] for s in data.get("siblings", [])]
            if names:
                return mirror, names
        except Exception as e:  # 换镜像
            last_error = e
            print(f"  {mirror} 不可用：{type(e).__name__}: {e}", file=sys.stderr)
    raise RuntimeError(f"无法获取 {repo} 的文件列表：{last_error}")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(mirror: str, repo: str, remote: str, target: Path) -> None:
    """带进度与断点续传的单文件下载。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{mirror}/{repo}/resolve/main/{remote}"
    part = target.with_suffix(target.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0

    headers = {"User-Agent": "ragkb/0.1"}
    if done:
        headers["Range"] = f"bytes={done}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # 已经下完
            part.replace(target)
            print(f"  {remote} 已完整")
            return
        raise

    total = int(resp.headers.get("Content-Length") or 0) + done
    mode = "ab" if done and resp.status == 206 else "wb"
    if mode == "wb":
        done = 0
    with open(part, mode) as fh:
        chunk = 1 << 16
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            if total:
                pct = done * 100 / total
                bar = "█" * int(pct / 4) + "·" * (25 - int(pct / 4))
                print(f"\r  {remote:<28} [{bar}] {pct:5.1f}%  {human(done)}/{human(total)}", end="")
            else:
                print(f"\r  {remote:<28} {human(done)}", end="")
    print()
    part.replace(target)


def main() -> int:
    ap = argparse.ArgumentParser(description="下载本地 ONNX embedding 模型")
    ap.add_argument("--repo", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--out", default="")
    ap.add_argument("--list", action="store_true", help="只列出仓库文件，不下载")
    args = ap.parse_args()

    target_dir = Path(args.out) if args.out else ROOT / "models" / args.repo.split("/")[-1]
    print(f"仓库：{args.repo}")
    print(f"目标：{target_dir}")

    mirror, names = api_files(args.repo)
    print(f"镜像：{mirror}")
    if args.list:
        for n in names:
            print("  " + n)
        return 0

    # tokenizer.json 是必需的；ONNX 权重按候选顺序挑第一个存在的
    wanted: list[str] = []
    if "tokenizer.json" in names:
        wanted.append("tokenizer.json")
    else:
        print("⚠ 仓库里没有 tokenizer.json，无法离线分词", file=sys.stderr)
    onnx_remote = next((c for c in ONNX_CANDIDATES if c in names), None)
    if not onnx_remote:
        print(f"⚠ 仓库里没有可用的 ONNX 权重。可用文件：\n  " + "\n  ".join(names), file=sys.stderr)
        return 2
    wanted.append(onnx_remote)
    for extra in ("tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "config.json"):
        if extra in names:
            wanted.append(extra)

    print(f"计划下载 {len(wanted)} 个文件：{', '.join(wanted)}")
    for remote in wanted:
        target = target_dir / remote
        if target.exists():
            print(f"  已存在，跳过：{remote}（{human(target.stat().st_size)}）")
            continue
        print(f"  下载 {remote}")
        download(mirror, args.repo, remote, target)

    print(f"\n完成。启用方式：")
    print(f"  RAGKB_EMBED_PROVIDER=onnx")
    print(f"  RAGKB_EMBED_MODEL={args.repo.split('/')[-1]}")
    print(f"  RAGKB_ONNX_MODEL_DIR={target_dir}")
    print("\n注意：换 embedding 模型后必须重建索引（新旧向量不在同一空间）：")
    print("  python -m ragkb.cli ingest ./data/corpus --force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
