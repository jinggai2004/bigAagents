"""调度器健康检查：回答"它到底在不在跑、数据在不在长"。

为什么需要
----------
批次 16 之前的真实状态是：调度器**从未成功启动过**（入口脚本在 `start()` 之前
调用 `next_run_times()` → AttributeError 直接崩溃），而**没有任何地方能看出来**——
日志文件照样生成、UI 照常打开、"共 N 个任务"日志也打出来了，
只是所有定时任务一个都没执行。`analyses` 长期 1 行、`screen_picks` 0 行，
直到要跑"胜率对照实验"才被发现。

结论：**"服务在跑"必须是可查询的状态，不能靠人去翻日志**。本模块提供：
- 心跳文件：调度进程每 60 秒刷新一次（`scripts/run_scheduler.py` 的心跳线程）
- `check_health()`：一次调用给出进程/心跳/数据新鲜度三层结论

数据新鲜度是最后一道防线：**心跳正常但数据不增长，也说明链路有问题**
（比如任务在跑但每次都异常退出）。
"""

import json
import logging
import os
from datetime import datetime

_logger = logging.getLogger(__name__)

# 数据根必须**运行时**解析：打包后 `__file__` 位于只读包 `_internal/` 里，
# 用它推导会指向空目录 ⇒ 读到空数据（2026-10-25 实测：双击 exe 后
# 选股/复盘/研究全 missing、交易台变空账户、系统页看不到调度器）。
# 规则见 `config/paths.data_root()`；源码运行结果与原先一致。
from config.paths import data_root as _paths_data_root  # noqa: E402
_PROJECT_ROOT = _paths_data_root()
HEARTBEAT_PATH = os.path.join(_PROJECT_ROOT, "temp", "scheduler_heartbeat.json")

# 心跳超过这个秒数没刷新 → 判定调度器已停（进程被杀/崩溃/没启动）
HEARTBEAT_STALE_SECONDS = 180


def write_heartbeat(pid: int = None, jobs: int = None, job_ids: list = None,
                    custom_job_ids: list = None, started_at: float = None):
    """写心跳（调度进程调用；原子替换避免读到半截 JSON）

    job_ids 是**正在运行的这份代码**的**内置**任务清单，用于自检"进程跑的是不是
    最新代码"：只写 jobs（任务个数）看不出少的是哪一个，也看不出是新增还是改名。

    ⚠️ 自定义定时任务**必须走 custom_job_ids**，不能混进 job_ids：
    它们本来就不在 `JOB_IDS` 里，混进去会让 `_code_freshness` 每次都比对出
    "多出来的任务"并告警，健康检查就从守护者退化成狼来了。
    两者分开的意义：内置清单严格比对（缺一个=旧代码），自定义只报告不告警。

    ⚠️ 判断"有没有传"用 `is not None`，**不能**用真假值：
    空列表 `[]` 是**有效信息**（"这个进程知道有自定义任务这回事，当前一个都没有"），
    而 `None` 才是"旧版进程根本没这个字段"。用 `if custom_job_ids` 会让
    `[]` 退化成 `None`，于是没有自定义任务的新进程被误报成"跑的是旧版代码"。
    """
    ids = list(job_ids) if job_ids is not None else None
    custom = list(custom_job_ids) if custom_job_ids is not None else None
    if jobs is None and (ids is not None or custom is not None):
        jobs = len(ids or []) + len(custom or [])
    # `started_at`（2026-10-23 加）：**进程启动时刻**，与每分钟刷新的 `epoch` 不同。
    # 用途：判断"文件改了但进程里还是旧代码"——拿心跳文件的 mtime 或 `epoch` 比
    # 永远比文件新，那个判断会恒为空（实测踩过）。缺省时保留上一次的值，
    # 避免老调用方每拍把它刷成"现在"。
    if started_at is None:
        try:
            with open(HEARTBEAT_PATH, encoding="utf-8") as _f:
                started_at = json.load(_f).get("started_at")
        except (OSError, ValueError):
            started_at = None
    payload = {
        "pid": pid or os.getpid(),
        "jobs": jobs,
        "job_ids": ids,
        "custom_job_ids": custom,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": datetime.now().timestamp(),
        "started_at": started_at,
    }
    os.makedirs(os.path.dirname(HEARTBEAT_PATH), exist_ok=True)
    tmp = HEARTBEAT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    # Windows 上 os.replace 无法覆盖正被读取打开的目标文件（本项目已踩过两次），
    # 但心跳文件读取是毫秒级，冲突概率极低——仍加退避重试以防万一。
    import time
    for attempt in range(4):
        try:
            os.replace(tmp, HEARTBEAT_PATH)
            return True
        except OSError:
            if attempt == 3:
                _logger.debug("[健康检查] 心跳写入被占用，跳过本次")
                return False
            time.sleep(0.02 * (attempt + 1))


def clear_heartbeat():
    """进程正常退出时清除心跳（避免"已停"被误判为"在跑"）"""
    try:
        os.remove(HEARTBEAT_PATH)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def read_heartbeat() -> dict:
    """读心跳，读不到/损坏返回 {}"""
    try:
        with open(HEARTBEAT_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _win_pid_alive(pid: int):
    """Windows 原生判活：`OpenProcess` + `GetExitCodeProcess`。

    返回 **True / False / None**（None = 本机无法判定，交给调用方降级）。

    ⚠️⚠️ 为什么不用 `tasklist`（2026-10-29 实测事故，用户报"不定期跳窗口"）：
      `web/templates/system.html` 每 **10 秒**轮询 `/api/system/health`
      → `system.status()` → `health.load_health()` → `TTL_HEALTH` 也正好是 **10 秒**
      （缓存几乎每次都过期）→ 每 10 秒调一次本函数。
      而 `tasklist.exe` 是**控制台程序**，从 `--noconsole` 打包的 GUI 进程里
      `subprocess.run` 又没传 `CREATE_NO_WINDOW` ⇒ **每 10 秒闪一个黑框**。
      `OpenProcess` 是纯 API 调用：不开进程、不开控制台、微秒级。
    """
    try:
        import ctypes
        from ctypes import wintypes
    except Exception as e:                      # noqa: BLE001
        _logger.debug("[health] ctypes 不可用，降级 tasklist: %s: %s",
                      type(e).__name__, e)
        return None

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    ERROR_ACCESS_DENIED = 5
    ERROR_INVALID_PARAMETER = 87

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        handle = open_process(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            code = wintypes.DWORD()
            try:
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    # 句柄能打开 ≠ 还没退出（已终止但句柄未释放时也能打开）
                    return code.value == STILL_ACTIVE
                return None
            finally:
                close_handle(handle)

        err = ctypes.get_last_error()
        if err == ERROR_INVALID_PARAMETER:       # 没有这个 pid
            return False
        if err == ERROR_ACCESS_DENIED:           # 存在但受保护（系统进程）
            return True
        _logger.debug("[health] OpenProcess(pid=%s) 失败 err=%s，降级 tasklist", pid, err)
        return None
    except Exception as e:                       # noqa: BLE001
        _logger.debug("[health] 原生判活异常(pid=%s): %s: %s", pid, type(e).__name__, e)
        return None


def _tasklist_alive(pid: int) -> bool:
    """降级路径：`tasklist`（**必须 `CREATE_NO_WINDOW`**，否则闪黑框）。

    ⚠️ `tasklist` 在中文 Windows 上输出 GBK，用 text=True 按 utf-8 解码会
    UnicodeDecodeError（实测在子进程读取线程里抛，报错栈与真实原因完全不搭）。
    所以拿 bytes 自己解码，且**先判 pid 是否真的在输出里**再解码——
    输出里没有该 pid 时内容无所谓，避免为一个"不存在"的进程做无用解码。
    """
    import subprocess
    kwargs = {"capture_output": True, "timeout": 10}
    if os.name == "nt":
        # 缺这一行就是"每 10 秒闪一个黑框"的成因
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], **kwargs)
        raw = out.stdout or b""
        return str(pid).encode() in raw
    except Exception as e:                       # noqa: BLE001
        _logger.debug("[health] tasklist 不可用(pid=%s): %s: %s", pid, type(e).__name__, e)
        # 退回 os.kill 探测。⚠️ Windows 下 os.kill(pid, 0) 实测是**探测**
        # （Python 3.12.4：不存在抛 OSError WinError 87，存在则静默返回，不会杀进程），
        # 与 POSIX 语义一致，可以安全用作兜底。
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False


def _process_alive(pid) -> bool:
    """进程是否还在（跨平台、不引第三方依赖）"""
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if os.name == "nt":
        alive = _win_pid_alive(pid)
        if alive is not None:
            return alive
        return _tasklist_alive(pid)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def expected_job_ids() -> list:
    """当前代码期望的任务清单

    延迟 import：core.scheduler 会拉起 pipeline 等一大串依赖，健康检查在
    各种上下文（脚本/UI/测试）里被调用，不能在模块级把它拖进来。
    """
    try:
        from core.scheduler import JOB_IDS
        return list(JOB_IDS)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[健康检查] 读取任务清单失败: %s", e)
        return []


def _code_freshness(hb: dict) -> dict:
    """比对"心跳里的任务清单"与"当前代码的任务清单"

    这是批次 22 结束时那次事故的自动化版本：
    进程活着、心跳新鲜、任务数也对不上（旧进程 8 个 vs 新代码 9 个）——
    只有把清单本身拿出来比，才能发现"跑的是旧代码"。
    """
    expected = expected_job_ids()
    running = hb.get("job_ids")
    if not expected:
        return {"checked": False, "reason": "读不到当前代码的任务清单"}
    if running is None:
        return {"checked": False, "reason": "心跳不含任务清单（旧版进程）",
                "expected": expected, "missing": sorted(set(expected))}
    missing = [j for j in expected if j not in running]
    extra = [j for j in running if j not in expected]
    return {"checked": True, "expected": expected, "running": list(running),
            "missing": missing, "extra": extra, "stale": bool(missing)}


def _enabled_custom_job_ids() -> list | None:
    """库里标记为启用的自定义任务 id。读不到返回 None（=判不了，不猜）。"""
    try:
        from db import store as db_store
        return [j["job_id"] for j in db_store.list_scheduled_jobs(only_enabled=True)]
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[健康检查] 读取自定义定时任务失败: %s", e)
        return None


def _custom_jobs_report(hb: dict) -> dict:
    """自定义定时任务：**只报告，不参与"旧代码"告警**。

    但有一条例外值得告警：库里启用了自定义任务、而运行中的进程上没挂
    （心跳里没有 `custom_job_ids` 字段 = 那个进程是加这一层之前的代码），
    这些任务就永远不会执行——正是本项目"调度器五个月没跑过一次"的同型故障。
    """
    desired = _enabled_custom_job_ids()
    if desired is None:
        return {"checked": False, "reason": "读不到自定义任务表"}
    running = hb.get("custom_job_ids")
    if running is None:
        if not desired:
            return {"checked": True, "running": [], "desired": [], "missing": []}
        # 库里有启用的自定义任务，但心跳没有这个字段 → 进程跑的是旧代码
        return {"checked": True, "running": [], "desired": desired,
                "missing": desired, "stale": True,
                "reason": "心跳不含自定义任务清单（旧版进程）"}
    missing = [j for j in desired if j not in running]
    return {"checked": True, "running": list(running), "desired": desired,
            "missing": missing, "stale": bool(missing),
            "extra": [j for j in running if j not in desired]}


def _data_freshness() -> dict:
    """各关键表的最新时间 + 行数（数据不增长 = 链路有问题）"""
    out = {}
    try:
        from db import store as db_store
        stats = db_store.store_stats()
        for table in ("analyses", "screen_picks", "news_items", "daily_features",
                      "chat_logs"):
            info = stats.get(table) or {}
            out[table] = {"rows": info.get("rows", 0), "latest": info.get("latest")}
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[健康检查] 读取数据统计失败: %s", e)
    return out


def check_health(check_processes: bool = True,
                 check_schedule_gap: bool = True) -> dict:
    """一次性健康报告。

    Args:
        check_processes: 是否检查「长驻进程跑的是不是旧代码」（读**本机真实进程**）。
            默认 True（这才是它的价值：把"改了代码没重启"变成可查的事实）。
            ⚠️ 测试必须传 False —— 否则结果取决于**跑测试那台机器上有没有陈旧进程**，
            用例会在"本机恰好有旧进程"时红（实测 2026-10-09 就是这么红的）。
            该检查逻辑自身的测试见 `tests/test_process_freshness.py`（注入假进程，不依赖本机）。
        check_schedule_gap: 是否扫描**当日调度日志**、找出"按 cron 该跑却没跑"的缺口。
            默认 True —— 这是唯一能发现"任务静默漏跑"的通道（见 core/schedule_gap）：
            `misfire_grace_time=3600 + coalesce=True` 的语义是"迟到的能补跑、超时的被静默丢弃"，
            所以漏跑时**日志里连一行都没有**，只能靠"数数量"发现。
            ⚠️ 测试同样必须传 False（否则结果取决于跑测试那天的日志）。
            该逻辑自身的测试见 `tests/test_schedule_gap.py`（用**构造的日志文件**，不依赖本机）。

    Returns:
        {
          "scheduler": {"running": bool, "reason": str, "pid":…, "ts":…, "idle_seconds":…},
          "code": {"stale": bool, "missing": [...], "extra": [...]},   # 跑的是不是最新代码
          "custom_jobs": {"checked": bool, "running": [...], "desired": [...],
                          "missing": [...]},   # 自定义定时任务（只报告，不参与旧代码判定）
          "long_running": {"checked": bool, "processes": [...], "stale": [...]},
          "schedule_gap": {"available": bool, "max_gap_minutes": int, "jobs": {...}},
          "data": {表名: {"rows", "latest"}},
          "warnings": [...],        # 需要人处理的问题（可直接展示）
        }
    """
    hb = read_heartbeat()
    warnings: list[str] = []
    code: dict = {}

    if not hb:
        sched = {"running": False, "reason": "没有心跳文件——调度器从未启动或已被清理"}
        warnings.append("调度器未运行：定时选股/回测/复盘/回填都不会执行，"
                        "analyses 与 screen_picks 不会增长。请运行 "
                        "`python scripts/run_scheduler.py`（或双击 启动bigAagents.bat）。")
    else:
        idle = None
        try:
            idle = round(datetime.now().timestamp() - float(hb.get("epoch") or 0), 1)
        except (TypeError, ValueError):
            pass
        alive = _process_alive(hb.get("pid"))
        fresh = idle is not None and idle <= HEARTBEAT_STALE_SECONDS
        if alive and fresh:
            sched = {"running": True, "reason": "心跳正常",
                     "pid": hb.get("pid"), "ts": hb.get("ts"),
                     "idle_seconds": idle}
        elif alive and not fresh:
            sched = {"running": False, "reason": f"进程在但心跳已停 {idle}s",
                     "pid": hb.get("pid"), "ts": hb.get("ts"), "idle_seconds": idle}
            warnings.append(f"调度进程 {hb.get('pid')} 存在但心跳停止 {idle} 秒，"
                            "可能在执行超长任务或已卡死。")
        elif not alive:
            sched = {"running": False, "reason": "心跳文件里的进程已不存在",
                     "pid": hb.get("pid"), "ts": hb.get("ts"),
                     "idle_seconds": idle}
            warnings.append(f"调度进程 {hb.get('pid')} 已退出（心跳残留），"
                            "定时任务已停。")
        else:
            sched = {"running": False, "reason": "心跳状态未知", "idle_seconds": idle}

        # 进程在跑 ≠ 跑的是最新代码
        code = _code_freshness(hb)
        if code.get("checked") and code.get("missing"):
            warnings.append(
                "调度进程跑的是旧代码：缺少任务 "
                f"{code['missing']}（当前代码应有 {len(code['expected'])} 个任务）。"
                "改完调度相关代码必须重启调度进程才会生效。")
        if code.get("checked") and code.get("extra"):
            warnings.append(
                f"心跳里有当前代码不存在的任务 {code['extra']}："
                "心跳可能来自另一个分支/旧版本进程。")
        if hb and not code.get("checked"):
            warnings.append(
                f"无法判断代码新旧（{code.get('reason')}）：重启调度进程后本项即可自检。")

    # 自定义定时任务：单独一路，**不参与"旧代码"严格比对**（它们合法地不属于 JOB_IDS）。
    # 但"库里启用了、进程上没挂"会让任务永远不执行，且不报错——必须说出来。
    custom = _custom_jobs_report(hb)
    if sched.get("running") and custom.get("missing"):
        warnings.append(
            f"有 {len(custom['missing'])} 个自定义定时任务未挂到运行中的调度进程上"
            f"（{custom['missing']}）：该进程可能是加自定义任务之前的代码，"
            "重启一次调度器即可；若重启后仍在，检查任务的 cron 是否合法（见调度日志）。")

    data = _data_freshness()

    # 长驻进程的「代码陈旧」：UI 与调度器各自比对"进程启动时间 vs 源码 mtime"。
    # 独立于上面的 `code` 段（那段比的是**心跳里的任务清单**，只覆盖调度器，
    # 且"只改函数体"时清单不变 → 查不出来）。见 core/process_freshness 的模块说明。
    long_running = (_process_freshness_report() if check_processes
                    else {"checked": False, "processes": [], "stale": [],
                          "reason": "本次未检查（check_processes=False）"})
    if long_running.get("checked") and long_running.get("stale"):
        warnings.append(
            "以下长驻进程跑的是**旧代码**："
            + "、".join(long_running["stale"])
            + "。启动之后源码有改动 → 改动的逻辑不会生效，且不会有任何报错。"
            "调度器双击 `重启调度器.bat`；UI 重跑 `streamlit run ui/app.py`。")

    # 数据新鲜度：心跳正常但关键表长期不增长 → 链路有问题（不是"没到时间"就是"一直失败"）
    if sched.get("running"):
        for table, label in (("analyses", "深度分析"), ("screen_picks", "选股候选")):
            info = data.get(table) or {}
            if not info.get("rows"):
                warnings.append(f"{table} 表仍为空（{label}闭环没有样本）："
                                "若调度已运行多日，检查对应任务是否每次异常退出。")

    # 当日调度缺口：唯一能发现"任务静默漏跑"的通道（详见 core/schedule_gap 模块说明）
    gap = (_schedule_gap_report() if check_schedule_gap
           else {"available": False, "reason": "本次未检查（check_schedule_gap=False）",
                 "jobs": {}, "max_gap_minutes": 0, "warnings": []})
    warnings.extend(gap.get("warnings") or [])

    return {"scheduler": sched, "code": code, "custom_jobs": custom, "data": data,
            "long_running": long_running, "schedule_gap": gap,
            "warnings": warnings,
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def _schedule_gap_report() -> dict:
    """当日「按 cron 该跑却没跑」的缺口（best-effort：失败不该拖垮整个健康检查）"""
    try:
        from core import schedule_gap
        return schedule_gap.analyse()
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[健康检查] 调度缺口检测失败: %s", e)
        return {"available": False, "reason": f"{type(e).__name__}: {e}",
                "jobs": {}, "max_gap_minutes": 0, "warnings": []}


def _process_freshness_report() -> dict:
    """长驻进程的代码陈旧诊断（best-effort：它自己的失败不该拖垮整个健康检查）"""
    try:
        from core import process_freshness
        return process_freshness.check()
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[健康检查] 进程陈旧诊断失败: %s", e)
        return {"checked": False, "reason": f"{type(e).__name__}: {e}",
                "processes": [], "stale": []}


def format_report(health: dict = None) -> str:
    """把健康报告转成一段可直接展示/打印的文本"""
    h = health or check_health()
    s = h["scheduler"]
    lines = [f"调度器: {'✅ 运行中' if s['running'] else '❌ 未运行'} — {s['reason']}"]
    if s.get("pid"):
        lines.append(f"  进程 PID {s['pid']}｜心跳时间 {s.get('ts') or '—'}"
                     + (f"｜{s['idle_seconds']}s 前" if s.get("idle_seconds") is not None
                        else ""))
    code = h.get("code") or {}
    if code:
        if not code.get("checked"):
            lines.append(f"代码新旧: ⚠️ 无法判定 — {code.get('reason')}")
        elif code.get("missing") or code.get("extra"):
            lines.append(f"代码新旧: ❌ 旧代码 — 缺 {code.get('missing') or '无'}"
                         f"｜多 {code.get('extra') or '无'}"
                         f"（应有 {len(code.get('expected') or [])} 个任务）")
        else:
            lines.append(f"代码新旧: ✅ 与当前代码一致"
                         f"（{len(code.get('running') or [])} 个任务）")
    custom = h.get("custom_jobs") or {}
    if custom.get("running"):
        lines.append(f"自定义任务: {len(custom['running'])} 个 — "
                     + "、".join(custom["running"]))
    elif custom.get("desired"):
        lines.append(f"自定义任务: ⚠️ 库里有 {len(custom['desired'])} 个启用，"
                     f"但运行中的进程上没挂（{custom.get('reason') or '未同步'}）")
    lr = h.get("long_running") or {}
    if lr:
        lines.append("长驻进程（代码是否已生效）:")
        try:
            from core.process_freshness import describe
            lines.extend(f"  {x}" for x in describe(lr))
        except Exception:                       # noqa: BLE001
            lines.append(f"  ⚠️ 无法判断：{lr.get('reason')}")
    lines.append("数据新鲜度:")
    for table, info in h.get("data", {}).items():
        lines.append(f"  {table:<16} {info['rows']:>6} 行｜最新 {info.get('latest') or '—'}")
    gap = h.get("schedule_gap") or {}
    if gap:
        try:
            from core.schedule_gap import describe as _gap_describe
            lines.extend(_gap_describe(gap))
        except Exception:                       # noqa: BLE001
            lines.append(f"调度缺口: ⚠️ 无法判断 — {gap.get('reason')}")
    if h.get("warnings"):
        lines.append("需处理:")
        lines.extend(f"  ⚠️ {w}" for w in h["warnings"])
    return "\n".join(lines)
