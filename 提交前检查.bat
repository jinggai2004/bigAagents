@echo off

chcp 65001 >nul

title bigAagents pre-commit checks

setlocal enabledelayedexpansion

rem ====== Interpreter: BIGA_PYTHON > local default > PATH ======
rem Body is pure ASCII on purpose: Chinese text + chcp 65001 breaks cmd decoding.
set "PY=%BIGA_PYTHON%"
if defined PY if not exist "%PY%" (
    echo [WARN] BIGA_PYTHON path not found, ignored: %PY%
    set "PY="
)
if not defined PY set "PY=C://Users//OMEN//AppData//Local//Programs//Python//Python312//python.exe"
if not exist "%PY%" set "PY=python"

cd /d "%~dp0"

echo.
echo === pre-commit checks (fast set, about 10s) ===
echo.

"%PY%" scripts\pre_commit_checks.py %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" echo [OK]   clean - safe to commit
if "%RC%"=="1" echo [FAIL] blocked - fix the items above before committing
if "%RC%"=="2" echo [WARN] passed with warnings - review them, then commit

pause
exit /b %RC%
