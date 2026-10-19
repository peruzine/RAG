@echo off
REM ---------------------------------------------------------------------------
REM ragkb cmd wrapper: bypasses the default Windows script execution policy.
REM
REM   scripts\ragkb.cmd demo
REM   scripts\ragkb.cmd ask "P0 fault resolution time?" --show-context
REM   scripts\ragkb.cmd serve
REM
REM Same as run.ps1, but you do not have to change the execution policy first.
REM ASCII-only on purpose: cmd.exe reads batch files in the OEM code page.
REM ---------------------------------------------------------------------------
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %ERRORLEVEL%