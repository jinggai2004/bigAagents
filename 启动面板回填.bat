@echo off

chcp 65001 >nul

title bigAagents 面板回填

setlocal enabledelayedexpansion

rem ====== 解释器定位：与 运行测试.bat 完全一致 ======
set "PY=%BIGA_PYTHON%"
if defined PY if not exist "%PY%" ( echo [警告] BIGA_PYTHON 指向的路径不存在，已忽略: %PY% & set "PY=" )
if not defined PY set "PY=C://Users//OMEN//AppData//Local//Programs//Python//Python312//python.exe"
if not exist "%PY%" set "PY="
if not defined PY for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do if not defined PY set "PY=%%i"
if not defined PY for /f "delims=" %%i in ('where python 2^>nul') do if not defined PY set "PY=%%i"
if not defined PY (
    echo [错误] 找不到可用的 Python 解释器。
    pause
    exit /b 1
)
rem =====================================================

cd /d "%~dp0"

set PYTHONIOENCODING=utf-8

echo ============================================
echo  bigAagents 面板回填（价量历史 → data/panel/）
echo  ⚠️ 可随时关闭窗口：重开会自动断点续跑
echo  ⚠️ 若提示 trade_cal 频率超限(1次/小时)：等下一个整点再双击
echo ============================================
echo.

"%PY%" scripts\backfill_panel_baostock.py --apply --source tushare --start 2016-01-01 %*

echo.
echo [结束] 面板状态见上方输出；数据在 data\panel\parts\
pause
