@echo off
chcp 65001 >nul
title bigAagents 停止器

echo 正在停止 bigAagents 全部服务...

rem 0) Write the disable marker FIRST: the watchdog (scripts/watchdog.py)
rem    will not restart the scheduler while this file exists. Order matters -
rem    killing first leaves a window where the watchdog could pull it back up.
if not exist "temp" mkdir "temp"
echo stopped-by-user>"temp\.scheduler_disabled"

rem 1) 按窗口标题停止（启动器开的两个窗口）
taskkill /FI "WINDOWTITLE eq bigAagents-调度器*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq bigAagents-Web*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq bigAagents-守护*" /T /F >nul 2>&1

rem 2) 兜底：按命令行清理残留进程（不影响其他 Python 程序）
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run_scheduler.py*' -and $_.Name -like '*python*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*streamlit run ui*' -and $_.Name -like '*python*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*watchdog.py*' -and $_.Name -like '*python*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
rem    —— 面板自调用起的调度器（2026-10-26）：它是 `bigaagents.exe --scheduler`，
rem       命令行里没有 run_scheduler.py，上面按 python.exe 的判据抓不到它。
rem       不加这条 ⇒ 点了"停止"后调度器照旧在跑，比不装自动拉起还糟。
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*--scheduler*' -and $_.Name -like 'bigaagents*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

rem 3) 清理心跳文件：否则下次启动会被单实例检查误判为"调度器还在跑"
if exist "temp\scheduler_heartbeat.json" del /q "temp\scheduler_heartbeat.json"
rem 守护的 pid 文件也一并清（进程已杀，残留会让下次启动等 pid_alive 判定）
if exist "temp\.watchdog.pid" del /q "temp\.watchdog.pid"
if exist "temp\scheduler_heartbeat.json.tmp" del /q "temp\scheduler_heartbeat.json.tmp"

echo.
echo 已停止。（如个别窗口仍开着，直接手动关闭即可）
echo 可用 `python scripts/check_health.py` 复核停止状态。
ping -n 4 127.0.0.1 >nul
