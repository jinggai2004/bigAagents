@echo off

rem 本文件是 UTF-8 编码，不切代码页会让下面的中文全变成乱码

chcp 65001 >nul

title bigAagents 调度器重启器

rem 用途：只重启定时调度器，不影响已运行的 Web 界面（streamlit）

rem 场景：改了 core/scheduler.py 等调度相关代码后，必须重启才能生效



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

cd /d "%~dp0"



echo [1/3] 停止旧调度器（按命令行精确匹配，不影响其他 Python 程序）...

powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run_scheduler.py*' -and $_.Name -like '*python*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"



rem 心跳文件不删会被单实例检查误判为"还在跑"

if exist "temp\scheduler_heartbeat.json" del /q "temp\scheduler_heartbeat.json"

if exist "temp\scheduler_heartbeat.json.tmp" del /q "temp\scheduler_heartbeat.json.tmp"



echo [2/3] 启动调度器...

start "bigAagents-Scheduler" "%PY%" "scripts\run_scheduler.py"



echo [3/3] 等待心跳写入后自检...

ping -n 12 127.0.0.1 >nul

"%PY%" "scripts\check_health.py"



echo.

echo 完成。上面应显示"调度器: ✅ 运行中"，内置任务数为 10（自定义任务另计）。

pause

