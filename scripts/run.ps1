# ragkb 常用开发命令（Windows / PowerShell）
#
#   .\scripts\run.ps1 demo            # 入库示例语料
#   .\scripts\run.ps1 ask "试用期最长可以多久？"
#   .\scripts\run.ps1 eval            # 评测 + CI 门禁
#   .\scripts\run.ps1 test            # 单元测试
#   .\scripts\run.ps1 serve           # 起服务（含 Web UI）
#
# 如果提示"禁止运行脚本"（Windows 默认执行策略），改用同目录下的 .cmd 包装器：
#   .\scripts\ragkb.cmd demo
# 或者只放行当前窗口：
#   Set-ExecutionPolicy -Scope Process Bypass

param(
  [Parameter(Position = 0)][string]$Command = "help",
  [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:PYTHONIOENCODING = "utf-8"

# ---------------------------------------------------------------------------
# 解释器探测
#   Windows 上 `python` 经常是 Microsoft Store 的占位程序：`--version` 看着正常，
#   但 `-m pip` 报 9009、`-c` 不输出。所以这里不是"看命令存不存在"，而是**实际执行**
#   一次版本检查。结果拆成 (可执行文件, 前置参数) 两段，因为 `py -3` 需要额外参数。
# ---------------------------------------------------------------------------
$script:PyExe = $null
$script:PyPre = @()

function Resolve-Python {
  $candidates = @(
    @{ exe = (Join-Path $root ".venv\Scripts\python.exe"); pre = @() },
    @{ exe = "py";      pre = @("-3") },
    @{ exe = "python3"; pre = @() },
    @{ exe = "python";  pre = @() }
  )
  foreach ($c in $candidates) {
    if ($c.exe -match "[\\/]") {
      if (-not (Test-Path -LiteralPath $c.exe)) { continue }
    } elseif (-not (Get-Command $c.exe -ErrorAction SilentlyContinue)) {
      continue
    }
    try {
      & $c.exe @($c.pre) -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>$null
      if ($LASTEXITCODE -eq 0) {
        $script:PyExe = $c.exe
        $script:PyPre = $c.pre
        return
      }
    } catch { }
  }
  throw @"
未找到可用的 Python 3.11+。
本项目核心零第三方依赖，只需要一个 3.11 以上的解释器：
  * 已装 Python 启动器 → 试试 `py -3`
  * 或建虚拟环境       → py -3 -m venv .venv
  * 注意 Windows 上的 `python` 可能是 Microsoft Store 占位程序（`python -m pip` 报 9009）
"@
}

Resolve-Python
function Invoke-Py { & $script:PyExe @script:PyPre @args }

switch ($Command) {
  "demo" { Invoke-Py -m ragkb.cli demo @Rest }
  "ingest" { Invoke-Py -m ragkb.cli ingest @Rest }
  "ask" { Invoke-Py -m ragkb.cli ask @Rest }
  "eval" { Invoke-Py -m ragkb.cli eval @Rest }
  "serve" { Invoke-Py -m ragkb.cli serve @Rest }
  "stats" { Invoke-Py -m ragkb.cli stats --online @Rest }
  "check" { Invoke-Py -m ragkb.cli check @Rest }
  "docs" { Invoke-Py -m ragkb.cli docs list @Rest }
  "test" { Invoke-Py -m unittest discover -s tests -t . @Rest }
  "smoke" { Invoke-Py scripts/smoke.py @Rest }
  "sweep" { Invoke-Py scripts/sweep_refusal.py @Rest }
  "lint" { Invoke-Py -m ruff check ragkb tests scripts @Rest }
  default {
    Write-Host @"
ragkb 开发命令（当前解释器：$script:PyExe $($script:PyPre -join ' ')）

  demo              入库示例语料（含 internal 权限演示）
  ingest <path>     入库文件/目录（增量、幂等）  例：.\scripts\run.ps1 ingest .\data\corpus --acl public
  ask "<问题>"      提问（可加 --stream --show-context --json）
  eval              跑评测 + CI 门禁（--retrieval-only 只测检索）
  serve             启动 HTTP 服务与 Web UI
  stats             语料 + 线上指标
  check             体检：配置一致性 / 评测集标注 / 语料统计
  docs              文档列表
  test              运行单元测试
  smoke             端到端冒烟（离线、临时库）
  sweep             拒答阈值扫描
  lint              ruff 静态检查（需 pip install ruff）

示例：
  .\scripts\run.ps1 demo
  .\scripts\run.ps1 ask "P0 故障的首次响应时间是多少？" --show-context
  .\scripts\run.ps1 eval --verbose
  .\scripts\run.ps1 serve
"@
    exit 0
  }
}

exit $LASTEXITCODE
