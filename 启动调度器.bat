@echo off
chcp 65001 >nul
title bigAagents 调度器

cd /d "%~dp0"

rem ====== 解释器定位：BIGA_PYTHON > 本机默认路径 > py 启动器 > PATH 中的 python ======
rem 与 启动bigAagents.bat 同一套规则，换机器只需 set BIGA_PYTHON=...
set "PY=%BIGA_PYTHON%"
if defined PY if not exist "%PY%" ( echo [警告] BIGA_PYTHON 指向的路径不存在，已忽略: %PY% & set "PY=" )
if not defined PY set "PY=C:\Users\OMEN\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY="
rem 回退 py 启动器时必须取**真实 exe 路径**：写成 "py -3" 会让 cmd 找一个叫 `py -3` 的文件
if not defined PY for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do if not defined PY set "PY=%%i"
if not defined PY for /f "delims=" %%i in ('where python 2^>nul') do if not defined PY set "PY=%%i"
if not defined PY (
    echo [错误] 找不到可用的 Python 解释器。
    echo        请设置环境变量 BIGA_PYTHON 指向 python.exe。
    pause
    exit /b 1
)

rem 清掉 停止bigAagents.bat 留下的禁用标记，否则守护进程不会把它拉回来
if exist "temp\.scheduler_disabled" del /q "temp\.scheduler_disabled"

echo [1/2] 检查调度器是否已在运行...
rem 判据用 **心跳里的 PID 是否还活着**（不是 check_health 的退出码：
rem 那个 1 表示"有警告"，心跳陈旧也会返回 1，会把"在跑"误判成"没跑"）。
"%PY%" -c "import sys;sys.path.insert(0,'.');from scripts.launch_exe import _scheduler_alive;sys.exit(0 if _scheduler_alive()[0] else 1)"
if not errorlevel 1 (
    echo       已在运行，无需重复启动。
    ping -n 4 127.0.0.1 >nul
    exit /b 0
)

echo [2/2] 启动调度器与守护进程...
start "bigAagents-调度器" "%PY%" "scripts\run_scheduler.py"
start "bigAagents-守护" "%PY%" "scripts\watchdog.py" --loop

echo.
echo ============================================
echo  调度器已启动（18 个任务）
echo    盘前选股 09:25 / 午间复核 11:30 / 盘后全流程 15:40
echo.
echo  两个窗口请保持开启（关闭窗口 = 停止对应服务）
echo  一键停止：运行 停止bigAagents.bat
echo ============================================
ping -n 6 127.0.0.1 >nul
exit /b 0
