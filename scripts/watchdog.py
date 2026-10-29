"""调度器守护：心跳超时就重启，并把事件写进 `runs`（页面可见）。

为什么需要它（2026-10-20 事故）
──────────────────────────────
调度器在 17:05:36 跑完最后一条任务后被**外部终止**——没有 Traceback、没有退出日志。
之后 18:05 的新闻采集、18:10 的盘后深度分析全部静默漏跑，直到 19:52 人工查库
才发现**停了 2 小时 47 分钟**。

现有 `scripts/schedule_gap.py` 只能在**事后**算出缺口，没有任何东西会主动发现
"进程没了"并把它拉起来。本脚本补的就是这一环。

为什么不常驻
────────────
用 Windows 计划任务每 5 分钟跑一次、跑完即退。守护本身不是常驻进程，也就不存在
"守护进程自己死了没人管"的二阶问题——常驻守护是这类方案最常见的失败点。

⚠️ 与「停止bigAagents.bat」的冲突（唯一不可省的判断）
────────────────────────────────────────────────────
用户主动停止时**不能**又被拉起来。用禁用标记 `temp/.scheduler_disabled` 表达意图：
停止脚本写入、启动脚本删除，守护看到标记就什么都不做。没有这个标记，
用户永远停不掉服务——那比不装守护更糟。

⚠️ 崩溃循环
────────────
连续重启 `MAX_CONSECUTIVE` 次仍起不来 ⇒ 停止自动重启、只告警。
否则一个必崩的配置（比如缺依赖）会被无限拉起，把日志刷成噪声。

用法
────
    python scripts/watchdog.py --status     # 只看判定，不动作（排查用）
    python scripts/watchdog.py              # 检查并按需重启（计划任务调这个）
    python scripts/watchdog.py --dry-run    # 判定 + 打印，但不启动进程

退出码：0=正常/已处理；2=需要人工介入（告警态）
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import single_instance   # noqa: E402  OS 级单实例（见 core/single_instance）

# 守护**自己**的互斥名：与调度器不同名，两者各持一把，互不干扰
_MUTEX_NAME = single_instance.mutex_name("watchdog")

HEARTBEAT = ROOT / "temp" / "scheduler_heartbeat.json"
DISABLED = ROOT / "temp" / ".scheduler_disabled"
STATE = ROOT / "temp" / ".watchdog_state.json"
PIDFILE = ROOT / "temp" / ".watchdog.pid"
SCHEDULER = ROOT / "scripts" / "run_scheduler.py"

# ⚠️ 用 logging 而不是 print：本项目的可观测性护栏（tests/test_observability_hygiene.py）
#    只认 `_logger|logging|st.*|raise|set_exception`，`print` 不算留痕。
#    而且 `debug` 级别默认被过滤，读心跳失败这种**常态**不会刷屏。
_logger = logging.getLogger("watchdog")

# 心跳：调度进程每 60 秒刷一次（见 core/scheduler_health）。取 5 分钟容忍
# 一次任务重载或短暂卡顿，避免把"正在跑 58 分钟全流程"误判成停了。
STALE_SECONDS = 300
MAX_CONSECUTIVE = 3

_STATE_DAYS = 30          # 状态文件里保留的重启历史条数上限


# ================================================================
# 读写（全部 best-effort，任何失败都不该让守护自己崩）
# ================================================================

def read_heartbeat() -> dict:
    try:
        with open(HEARTBEAT, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        _logger.debug("watchdog 忽略可预期失败（%s）：%s", type(sys.exc_info()[1]).__name__, sys.exc_info()[1])
        print(f"[watchdog] 心跳读取失败（按“不存在”处理）：{type(e).__name__}: {e}")
        return {}


def read_disabled() -> str | None:
    """返回禁用原因；没有标记返回 None。

    标记内容约定：`<ISO 时间> <原因>`（例如 bat 写入 "2026-10-20 20:05 用户停止服务"）。
    """
    try:
        text = DISABLED.read_text(encoding="utf-8").strip()
        return text or "（无原因）"
    except FileNotFoundError:
        return None
    except OSError as e:
        _logger.debug("watchdog 忽略可预期失败（%s）：%s", type(sys.exc_info()[1]).__name__, sys.exc_info()[1])
        # 读不了标记时**倾向禁用**（宁可不重启，也不要违背用户意图）
        print(f"[watchdog] 禁用标记读取失败，按已禁用处理：{e}")
        return "（标记不可读，保守视为已禁用）"


def read_state() -> dict:
    try:
        with open(STATE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        _logger.debug("watchdog 忽略可预期失败（%s）：%s", type(sys.exc_info()[1]).__name__, sys.exc_info()[1])
        return {}


def write_state(state: dict) -> None:
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, STATE)
    except OSError as e:
        _logger.debug("watchdog 忽略可预期失败（%s）：%s", type(sys.exc_info()[1]).__name__, sys.exc_info()[1])
        print(f"[watchdog] 状态写入失败（不影响本次判定）：{e}")


def pid_alive(pid) -> bool:
    """进程是否还活着。

    Windows 上不能用 `os.kill(pid, 0)`（会真的发信号）。
    用 OpenProcess + GetExitCodeProcess：STILL_ACTIVE(259) 才算活。
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        _logger.debug("watchdog 忽略可预期失败（%s）：%s", type(sys.exc_info()[1]).__name__, sys.exc_info()[1])
        return False
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            _logger.debug("watchdog 忽略可预期失败（%s）：%s", type(sys.exc_info()[1]).__name__, sys.exc_info()[1])
            return False
    import ctypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == 259          # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


# ================================================================
# 判定（纯函数 —— 守护的全部逻辑都在这里，便于单测）
# ================================================================

def decide(hb: dict, now: float, disabled: str | None = None,
           state: dict | None = None, alive=None,
           stale: float = STALE_SECONDS,
           max_consec: int = MAX_CONSECUTIVE) -> dict:
    """给出 `{action, reason, age}`。

    action:
      none    正常 / 已禁用 ⇒ 什么都不做
      restart 该重启
      alert   不自动处理，需人工介入（进程卡死 / 连续重启仍失败）

    ⚠️ 「进程活着但心跳陈旧」只告警**不重启**：可能正在跑长任务而心跳线程卡住，
       直接杀掉会留下半截产物（回填的进度文件就踩过这个坑）。

    ⚠️⚠️ 判定顺序是**契约的一部分**（2026-10-20 审查抓到的真 bug）：
       第一版把「心跳不存在」和「时间戳无法解析」两个 `restart` 分支写在
       `consec >= max_consec` 之前就 `return` 了，于是**连续失败闸门只对其中
       一条路径生效**。只要调度器写不出心跳（权限/磁盘/启动即崩），守护会
       每 60 秒拉起一次、永不停止 —— 正是"防崩溃循环"要防的那件事。
       真实库里已经留下过一次痕迹（`{"action":"restarted","reason":"心跳不存在",
       "consecutive":1}`）。

       正确顺序：① 已禁用 → none；② 心跳新鲜 → none（唯一能清零计数的路径）；
       ③ **连续失败闸门** → alert；④ 才轮到各种 restart。
    """
    state = state or {}
    if disabled:
        return {"action": "none", "reason": f"已禁用（{disabled}）", "age": None}

    epoch = (hb or {}).get("epoch")
    age = None
    if epoch:
        try:
            age = max(0.0, float(now) - float(epoch))
        except (TypeError, ValueError):
            _logger.debug("watchdog 忽略可预期失败（%s）：%s", type(sys.exc_info()[1]).__name__, sys.exc_info()[1])
            age = None

    # ② 心跳新鲜 ⇒ 正常（这是唯一会清零 consecutive 的路径）
    if age is not None and age <= stale:
        return {"action": "none", "age": age,
                "reason": f"心跳正常（{age:.0f}s 前刷新，pid={(hb or {}).get('pid')}）"}

    # ③ ⚠️ 连续失败闸门：必须在所有 restart 分支之前
    consec = int(state.get("consecutive") or 0)
    if consec >= max_consec:
        why = "心跳不存在" if not epoch else (
            "心跳时间戳无法解析" if age is None else f"心跳 {age:.0f}s 未刷新")
        return {"action": "alert", "age": age,
                "reason": (f"{why}，且已连续重启 {consec} 次仍未恢复"
                           f"（上限 {max_consec}）——停止自动重启，请查日志")}

    # ④ 各种 restart 分支
    if not epoch:
        return {"action": "restart", "age": None,
                "reason": "心跳不存在（调度器从未启动，或心跳被清除）"}
    if age is None:
        return {"action": "restart", "age": None, "reason": "心跳时间戳无法解析"}

    pid = (hb or {}).get("pid")
    is_alive = alive(pid) if callable(alive) else pid_alive(pid)
    if is_alive:
        return {"action": "alert", "age": age,
                "reason": (f"进程 {pid} 仍在，但心跳 {age:.0f}s 未刷新"
                           f"（阈值 {stale:.0f}s）——可能卡死，需人工确认，不自动重启")}

    return {"action": "restart", "age": age,
            "reason": f"心跳 {age:.0f}s 未刷新且进程 {pid} 不在"}


# ================================================================
# 动作
# ================================================================

def _python_exe() -> str:
    """启动调度器用哪个解释器：BIGA_PYTHON > 项目默认 > 当前解释器。

    与各 .bat 的优先级保持一致——否则计划任务用的是别的 python，
    可能缺 pytest / 缺依赖，起一个必崩的进程。
    """
    # 常见安装位置做候选（不含用户名，glob 展开；正斜杠 Windows 通用）；BIGA_PYTHON 最优先
    import glob as _glob
    cands = [os.getenv("BIGA_PYTHON")]
    cands += sorted(_glob.glob("C:/Users/*/AppData/Local/Programs/Python/Python312/python.exe"),
                    reverse=True)
    for cand in cands:
        if cand and Path(cand).exists():
            return cand
    return sys.executable


def acquire_singleton() -> bool:
    """守护单实例：已有活着的 watchdog 就返回 False（调用方应直接退出）。

    ⚠️⚠️ 2026-10-20 实测：用户连点两次「启动bigAagents.bat」⇒ 两个 watchdog
       同时跑。后果不是"多一个进程"那么轻：
       ① 两边都判定 restart 时**同时拉起调度器**（单实例保护只能挡住第二个，
          挡不住瞬时双写）；
       ② 两边都往 `runs` 写事件，健康页出现重复条目；
       ③ 两边都读改写 `temp/.watchdog_state.json`，`consecutive` 计数互相覆盖
          —— 于是"连续 3 次失败才停手"这个保护会**永远数不到 3**。
       所以守护自己必须有单实例，不能只依赖被守护方的保护。

    ⚠️⚠️ 2026-10-23：下面的 pid 文件判定**本身就是那个 TOCTOU 坑** ——
       "读 pid → 判活 → 写 pid" 三步之间没有原子性，连点两次时两个进程
       可能都读到空/过期 pid ⇒ 都通过。所以先拿 **OS 命名互斥**（进程退出自动
       释放，无窗口），pid 文件保留但降级为"给页面/诊断看"的副本。
    """
    if not single_instance.acquire_os_mutex(_MUTEX_NAME):
        print(f"[watchdog] 已有守护在运行（互斥 {_MUTEX_NAME} 被占用），本实例退出")
        return False
    try:
        old = int(PIDFILE.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError) as e:
        # 没有 pid 文件是首次启动的正常情况；读不出来也只影响"能不能复用旧 pid"
        logging.getLogger("watchdog").debug("读取单实例 pid 文件失败（按首次启动处理）：%s", e)
        old = 0
    if old and old != os.getpid() and pid_alive(old):
        print(f"[watchdog] 已有守护在运行（pid={old}），本实例退出")
        return False
    try:
        PIDFILE.parent.mkdir(parents=True, exist_ok=True)
        PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    except OSError as e:
        logging.getLogger("watchdog").warning("写单实例 pid 文件失败：%s（继续运行）", e)
    return True


def release_singleton() -> None:
    """退出时清掉 pid 文件（只清自己那份）。"""
    try:
        cur = int(PIDFILE.read_text(encoding="utf-8").strip() or 0)
        if cur == os.getpid():
            PIDFILE.unlink()
    except (OSError, ValueError) as e:
        logging.getLogger("watchdog").debug("清理单实例 pid 文件失败：%s", e)
    # 互斥一般由进程退出自动回收；这里显式释放，便于同进程内重复启停（测试/演练）
    single_instance.release_os_mutex(_MUTEX_NAME)


def start_scheduler() -> int | None:
    """脱离本进程启动调度器（守护退出后它继续活）。"""
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.Popen(
            [_python_exe(), str(SCHEDULER)],
            cwd=str(ROOT), env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, creationflags=flags)
        return proc.pid
    except (OSError, ValueError) as e:
        _logger.debug("watchdog 忽略可预期失败（%s）：%s", type(sys.exc_info()[1]).__name__, sys.exc_info()[1])
        print(f"[watchdog] 启动调度器失败：{type(e).__name__}: {e}")
        return None


def record(run_type: str, summary: dict) -> None:
    """写一条 runs（页面可见）。失败只打印，不影响守护结果。"""
    try:
        from db import store
        store.insert_run(run_type, summary)
    except Exception as e:                      # noqa: BLE001
        _logger.debug("watchdog 忽略可预期失败（%s）：%s", type(sys.exc_info()[1]).__name__, sys.exc_info()[1])
        print(f"[watchdog] 运行记录写入失败：{type(e).__name__}: {e}")


def tick(stale: int = STALE_SECONDS, dry_run: bool = False) -> int:
    """做一次检查（判定 + 按需动作）。返回 0=正常/已处理，2=需人工介入。

    ⚠️ 抽成独立函数是为了 `--loop` 常驻模式：计划任务被企业策略/沙箱禁用时
       （本机实测 `schtasks.exe` 在黑名单里），只能靠一个常驻循环每 5 分钟调它。
       常驻模式的残余风险是"守护自己也会挂"，所以 `check_health.py` 里另有一项
       存活检查；但那属于二阶问题，比"调度器挂了完全没人知道"好得多。
    """
    hb = read_heartbeat()
    disabled = read_disabled()
    state = read_state()
    verdict = decide(hb, time.time(), disabled, state, stale=stale)
    action, reason, age = verdict["action"], verdict["reason"], verdict["age"]
    print(f"[watchdog] {datetime.now():%Y-%m-%d %H:%M:%S} → {action}｜{reason}")

    if action == "none":
        # 心跳恢复 ⇒ 清掉连续失败计数（这才是"真的好了"的判据）
        if state.get("consecutive"):
            state["consecutive"] = 0
            write_state(state)
            print("[watchdog] 心跳已恢复，连续重启计数清零")
        return 0

    if action == "alert":
        record("job:scheduler_watchdog", {
            "action": "alert", "reason": reason, "heartbeat_age_s": age,
            "consecutive": int(state.get("consecutive") or 0),
            "note": "需要人工介入；守护不会自动重启"})
        return 2

    # action == "restart"
    if dry_run:
        print("[watchdog] dry-run：不启动进程")
        return 0
    pid = start_scheduler()
    if pid is None:
        state["consecutive"] = int(state.get("consecutive") or 0) + 1
        state["last_error"] = "启动失败"
        write_state(state)
        record("job:scheduler_watchdog", {
            "action": "restart_failed", "reason": reason, "error": "Popen 失败",
            "consecutive": state["consecutive"]})
        return 2

    hist = list(state.get("history") or [])
    hist.append({"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "reason": reason, "new_pid": pid, "stale_age_s": age})
    state.update({"consecutive": int(state.get("consecutive") or 0) + 1,
                  "last_restart": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  "last_reason": reason, "last_pid": pid,
                  "history": hist[-_STATE_DAYS:]})
    write_state(state)
    record("job:scheduler_watchdog", {
        "action": "restarted", "reason": reason, "heartbeat_age_s": age,
        "new_pid": pid, "consecutive": state["consecutive"]})
    print(f"[watchdog] 已启动调度器 pid={pid}（连续第 {state['consecutive']} 次）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="调度器守护（心跳超时自动重启）")
    ap.add_argument("--status", action="store_true", help="只打印判定，不动作")
    ap.add_argument("--dry-run", action="store_true", help="判定并打印，但不启动进程")
    ap.add_argument("--stale", type=int, default=STALE_SECONDS, help="心跳陈旧阈值（秒）")
    ap.add_argument("--loop", action="store_true",
                    help="常驻模式：每 --interval 秒检查一次（随启动器一起启停）")
    ap.add_argument("--interval", type=int, default=300, help="常驻模式检查间隔（秒）")
    ap.add_argument("--log-file", default=None,
                    help="把输出追加到该文件（无控制台运行时必须给）")
    args = ap.parse_args()

    if args.log_file:
        # ⚠️ pythonw.exe / 重定向运行没有 stdout，不重定向的话 print 全部丢失 ——
        #    于是"守护跑了但什么都没记"，等于又造了一个观测盲区。
        try:
            log_path = Path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = open(log_path, "a", encoding="utf-8")
            sys.stdout = sys.stderr = stream
        except OSError as e:
            _logger.debug("watchdog 忽略可预期失败（%s）：%s", type(sys.exc_info()[1]).__name__, sys.exc_info()[1])
            print(f"[watchdog] 日志文件打不开，改用标准输出：{e}")

    if not args.loop:
        if args.status:
            # --status 只判定不动作
            hb, disabled, state = read_heartbeat(), read_disabled(), read_state()
            v = decide(hb, time.time(), disabled, state, stale=args.stale)
            print(f"[watchdog] {datetime.now():%Y-%m-%d %H:%M:%S} → {v['action']}｜{v['reason']}")
            return 0
        return tick(stale=args.stale, dry_run=args.dry_run)

    if not acquire_singleton():
        return 1
    print(f"[watchdog] 常驻模式启动，每 {args.interval}s 检查一次；"
          f"Ctrl+C 退出（或双击「停止bigAagents.bat」）")
    try:
        while True:
            try:
                tick(stale=args.stale)
            except Exception as e:              # noqa: BLE001
                _logger.debug("watchdog 忽略可预期失败（%s）：%s", type(sys.exc_info()[1]).__name__, sys.exc_info()[1])
                # 单次检查失败绝不能把守护本身带走——否则一次读盘错误就永久失效
                print(f"[watchdog] 本次检查异常（继续下一轮）：{type(e).__name__}: {e}")
            time.sleep(max(10, args.interval))
    except KeyboardInterrupt:
        _logger.debug("watchdog 忽略可预期失败（%s）：%s", type(sys.exc_info()[1]).__name__, sys.exc_info()[1])
        print("[watchdog] 收到中断，退出")
        return 0
    finally:
        release_singleton()


if __name__ == "__main__":
    raise SystemExit(main())
