@echo off

chcp 65001 >nul

title bigAagents 启动器



rem ====== 解释器定位：BIGA_PYTHON > 本机默认路径 > py 启动器 > PATH 中的 python ======
rem 换机器/CI 只需 set BIGA_PYTHON=...，不必修改这个受版本控制的文件。
rem 注意：本项目必须用装了 pytest 的解释器（系统 Python 3.12），
rem       托管版 3.13 没有 pytest，被选中会出现"命令能跑但测试收集失败"。
set "PY=%BIGA_PYTHON%"
if defined PY if not exist "%PY%" ( echo [警告] BIGA_PYTHON 指向的路径不存在，已忽略: %PY% & set "PY=" )
if not defined PY set "PY=C:\Users\OMEN\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY="
rem ⚠️ 回退到 py 启动器时必须取它的**真实 exe 路径**：
rem   若写成 PY="py -3"，后面 "%PY%" 会展开成 "py -3"，cmd 会去找一个
rem   名叫 `py -3` 的文件而报"找不到命令"（参数被当成文件名的一部分）。
if not defined PY for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do if not defined PY set "PY=%%i"
if not defined PY for /f "delims=" %%i in ('where python 2^>nul') do if not defined PY set "PY=%%i"
if not defined PY (
    echo [错误] 找不到可用的 Python 解释器。
    echo        请设置环境变量 BIGA_PYTHON 指向 python.exe（需已安装 pytest）。
    pause
    exit /b 1
)
rem =====================================================

rem 工作目录 = 本bat所在目录（自动解析，无需改动）

rem =====================================================



cd /d "%~dp0"

rem Clear the disable marker written by 停止bigAagents.bat, otherwise the
rem watchdog (scripts/watchdog.py) would never bring the scheduler back.
if exist "temp\.scheduler_disabled" del /q "temp\.scheduler_disabled"



echo [0/4] 单实例检查：已有调度器在跑就不再启动（避免重复执行选股/回测）...

"%PY%" "scripts\check_health.py" --json >nul 2>&1

if %errorlevel%==0 (

    echo       检测到调度器已运行，跳过重复启动。

    set SKIP_SCHED=1

) else (

    set SKIP_SCHED=0

)



echo [1/4] 启动定时调度器（选股/回测/复盘/新闻采集）...

if "%SKIP_SCHED%"=="0" start "bigAagents-调度器" "%PY%" "scripts\run_scheduler.py"

rem Scheduler watchdog: restarts the scheduler when its heartbeat goes stale
rem (a real outage on 2026-10-20: process stopped at 17:05, the 18:10 analysis
rem  never ran, nobody noticed for 2h47m). It honors temp\.scheduler_disabled,
rem so the stop script still stops everything.
if "%SKIP_SCHED%"=="0" start "bigAagents-守护" "%PY%" "scripts\watchdog.py" --loop

if "%SKIP_SCHED%"=="1" echo       已跳过（调度器已在运行）



echo [2/4] 启动 Web 界面...

start "bigAagents-Web" "%PY%" -m streamlit run ui\app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false



echo [3/4] 等待服务就绪后打开浏览器...

ping -n 13 127.0.0.1 >nul

start http://localhost:8501



echo.

echo ============================================

echo  bigAagents 已启动

echo    Web 界面 : http://localhost:8501

echo    调度任务 : 选股/回测/复盘/新闻（自动运行）

echo.

echo  注意：两个命令行窗口请保持开启

echo        （关闭窗口 = 停止对应服务）

echo  一键停止：运行 停止bigAagents.bat

echo ============================================

ping -n 21 127.0.0.1 >nul

