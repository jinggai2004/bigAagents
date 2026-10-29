@echo off

chcp 65001 >nul

title bigAagents tests

setlocal enabledelayedexpansion



rem ====== Interpreter: BIGA_PYTHON > local default > py launcher > PATH ======
rem Other machines / CI: just `set BIGA_PYTHON=...`, never edit this versioned file.
rem This project must use the interpreter that HAS pytest (system Python 3.12):
rem the managed 3.13 has no pytest -> "command runs but collection fails".
set "PY=%BIGA_PYTHON%"
if defined PY if not exist "%PY%" (
    echo [WARN] BIGA_PYTHON path not found, ignored: %PY%
    set "PY="
)
if not defined PY set "PY=C:\Users\OMEN\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY="
rem Falling back to the py launcher MUST resolve to its real exe path:
rem   PY="py -3" would expand into "py -3" and cmd would look for a FILE named
rem   `py -3` and fail (the argument gets treated as part of the filename).
if not defined PY for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do if not defined PY set "PY=%%i"
if not defined PY for /f "delims=" %%i in ('where python 2^>nul') do if not defined PY set "PY=%%i"
if not defined PY (
    echo [ERROR] No usable Python interpreter found.
    echo         Set BIGA_PYTHON to python.exe ^(pytest required^).
    pause
    exit /b 1
)
rem =====================================================

rem =====================================================



cd /d "%~dp0"



rem Three env vars are all required (see .workbuddy\memory\MEMORY.md):
rem   CODEBUDDY_SAFE_DELETE_ENABLED=0 - disable the sandbox hook on os.remove.
rem     Without it: every "delete file" test fails in a FULL run (passes alone),
rem     and a SystemExit inside a background thread makes streamlit AppTest
rem     runner.join() never return -> pytest hangs on one cell, no error.
rem   PYTHONIOENCODING=utf-8 - otherwise Chinese/emoji prints use GBK and break collection.
rem   a FRESH --basetemp - reusing an old dir leaves file handles -> random failures.
set CODEBUDDY_SAFE_DELETE_ENABLED=0

set PYTHONIOENCODING=utf-8



rem Fresh timestamped basetemp every run (old dirs accumulate stale handles).
rem wmic was removed in new Win11 (measured 2026-10-17: DT empty -> garbage path ->
rem pytest INTERNALERROR) -> use PowerShell, fall back to random digits.
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddHHmmss"') do set DT=%%I

if not defined DT set "DT=%RANDOM%%RANDOM%"

set BT=temp\pytest_%DT%



rem Clean old basetemp dirs/logs, keep the last 3 runs. Never fails the script.

"%PY%" "scripts\clean_test_artifacts.py" --keep 3 --max-seconds 45



rem ====== Mode ======

rem   (no arg)             -> quick set (daily, excludes UI smoke)
rem   (arg "full")         -> full set (before commit, 4 parallel workers)

set MODE=%1

if "%MODE%"=="" (set MODE=quick)

rem R9: unknown mode must NOT silently fall through to quick (a typo would run the wrong suite).
if /i not "%MODE%"=="quick" if /i not "%MODE%"=="full" (
    echo [ERROR] unknown mode: %MODE%  ^(accepted: quick ^| full^)
    exit /b 2
)



if /i "%MODE%"=="full" goto :FULL



:QUICK

echo ============================================

echo  bigAagents quick tests ^(pass "full" before commit^)

echo  excluded: UI page smoke

echo  basetemp: %BT%

echo ============================================

echo.

"%PY%" -m pytest tests -q --basetemp="%BT%" -n 4 --ignore=tests/test_ui_pages_smoke.py %2 %3 %4

set RC=%errorlevel%

goto :REPORT



:FULL

echo ============================================

echo  bigAagents FULL tests ^(4 workers^)

echo  basetemp: %BT%

echo ============================================

echo.

"%PY%" -m pytest tests -q --basetemp="%BT%" -n 4 %2 %3 %4

set RC=%errorlevel%



:REPORT

echo.

if %RC%==0 (

    echo [PASS] all tests passed.

) else (

    echo [FAIL] exit code %RC%

    echo   - if failures are all "delete file" tests, check CODEBUDDY_SAFE_DELETE_ENABLED=0

    echo   - if failures happen only in parallel mode, rerun serially: "%PY%" -m pytest tests -q --basetemp=temp\pytest_xxx

)

rem Pause only for interactive (double-click) runs; automation sets BIGA_NO_PAUSE=1.
if not "%BIGA_NO_PAUSE%"=="1" pause

exit /b %RC%
