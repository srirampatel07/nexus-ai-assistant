@echo off
rem NEXUS one-click cloud AI launcher (Windows).
rem Double-click this file: no manual venv activation, no local servers.
rem Text + vision use the NVIDIA hosted API directly (no llama-server.exe,
rem no ports 8080/8081, no GGUF files required).
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv\Scripts\python.exe not found.
    echo Run the installer steps in README.md first, then double-click again.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "app\launcher\menu.py" %*
set EXITCODE=%ERRORLEVEL%
if not "%EXITCODE%"=="0" (
    echo.
    echo NEXUS launcher exited with code %EXITCODE%.
    pause
)
endlocal
