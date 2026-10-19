"""控制台输出兼容：让中文 Windows（GBK 代码页）也能正常打印符号。

问题
----
中文 Windows 的控制台默认代码页是 GBK，而 GBK **不包含** ``✓ ✗ ⚠ ✅ → █`` 这些字符。
一旦代码里 print 了它们，Python 会抛 ``UnicodeEncodeError`` 并带着退出码 1 结束——
看起来像功能坏了，其实只是"画不出这个字符"。项目里到处都在用这类符号（进度条、
体检结论、启动横幅），所以这个问题必须在入口处统一解决，而不是逐个换掉符号。

做法
----
把 stdout/stderr 重新配置为 UTF-8 并且 ``errors="replace"``：
* 支持 UTF-8 的终端（Windows Terminal、VS Code、绝大多数 Linux/macOS）显示正常；
* 老式 GBK 控制台不会崩，最多把画不出的字符显示成替代符；
* 输出被重定向到文件时同样不会崩。

放在 CLI 与各脚本的 ``main()`` 最前面调用即可。
"""

from __future__ import annotations

import sys


def enable_utf8_console() -> None:
    """尽最大努力把标准输出/错误切成 UTF-8，失败也不影响功能。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
