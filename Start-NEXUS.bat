@echo off
rem NEXUS one-click local AI launcher (Windows).
rem Double-click this file: no manual venv activation, no manual servers.
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
