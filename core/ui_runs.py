"""UI 长任务的结果落盘与运行标记（P1 修复）

问题背景：
  选股、深度分析这类 2~4 分钟的任务原先在 Streamlit 的脚本运行里同步执行，结果只存在
  session_state。浏览器一刷新就新建会话 → 回到初始界面；而后台线程仍会把任务跑完
  （Streamlit 的停止是在脚本下一次调用 st.* 时抛异常展开调用栈，无法中断一个正在阻塞的
  调用），结果却发不出去，成为孤儿任务。反复刷新还会叠加多个孤儿任务、重复打数据源。

这里提供两件事：
  1. **结果落盘 + 加载时回填**：刷新后仍能看到上一次结果，并标明生成时间；
  2. **运行标记**：页面可以显示"上一次任务仍在后台运行（已用 X）"，并据此阻止重复启动。

设计约束：
  - 目录 output/ui_runs/（已 gitignore）；
  - 所有读写都是 best-effort：UI 侧的持久化失败绝不能让主流程失败；
  - 写入用"临时文件 + os.replace"，避免刷新/崩溃时读到半个 JSON。
"""

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime

_logger = logging.getLogger(__name__)

# 数据根必须**运行时**解析：打包后 `__file__` 位于只读包 `_internal/` 里，
# 用它推导会指向空目录 ⇒ 读到空数据（2026-10-25 实测：双击 exe 后
# 选股/复盘/研究全 missing、交易台变空账户、系统页看不到调度器）。
# 规则见 `config/paths.data_root()`；源码运行结果与原先一致。
from config.paths import data_root as _paths_data_root  # noqa: E402
_PROJECT_ROOT = _paths_data_root()
UI_RUNS_DIR = os.path.join(_PROJECT_ROOT, "output", "ui_runs")
RUNNING_FILE = "running.json"

# 超过该时长仍标记为"运行中"的，视为孤儿任务（进程被杀/崩溃），允许重跑
DEFAULT_STALE_SECONDS = 900

# 自从任务改到后台线程执行，同一份 JSON 会被多个线程读写（工作线程刷心跳、
# 页面线程读进度/清标记）。Windows 上两个线程同时 os.replace 同一个临时文件会
# 直接报 [WinError 5] 拒绝访问，所以：临时文件名带线程号 + 全模块串行化。
_FILE_LOCK = threading.RLock()


# ----------------------------------------------------------------
# 基础读写
# ----------------------------------------------------------------

def _ensure_dir() -> bool:
    try:
        os.makedirs(UI_RUNS_DIR, exist_ok=True)
        return True
    except OSError as e:
        _logger.warning("[UI任务] 目录创建失败: %s", e)
        return False


def _atomic_write_json(filename: str, payload, retries: int = 4) -> bool:
    if not _ensure_dir():
        return False
    path = os.path.join(UI_RUNS_DIR, filename)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with _FILE_LOCK:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            # Windows 特有问题：os.replace **无法覆盖正被打开读取的目标文件**
            # （POSIX 允许覆盖，Windows 抛 WinError 5 拒绝访问）。读取方是毫秒级
            # 瞬时操作（页面每 2 秒轮询一次进度），因此瞬时锁冲突退避重试即可，
            # 不必为此放弃原子性。不重试时全量 pytest 会随机红 1 例（不同轮次
            # 命中不同用例），且真实使用中进度条会偶发停在旧状态。
            for attempt in range(retries):
                try:
                    os.replace(tmp, path)
                    return True
                except OSError:
                    if attempt == retries - 1:
                        raise
                    time.sleep(0.02 * (attempt + 1))
    except (OSError, TypeError, ValueError) as e:
        _logger.warning("[UI任务] 写入 %s 失败: %s", filename, e)
        try:
            os.path.exists(tmp) and os.remove(tmp)
        except OSError:
            pass
        return False


def _read_json(filename: str):
    path = os.path.join(UI_RUNS_DIR, filename)
    try:
        with _FILE_LOCK:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------
# 供其他模块复用的文件读写（都限定在 UI 运行目录内）
# ----------------------------------------------------------------

def write_json(filename: str, payload) -> bool:
    """原子写入 UI 运行目录下的一个 JSON（core/ui_tasks 用它写进度）"""
    return _atomic_write_json(filename, payload)


def read_json(filename: str):
    return _read_json(filename)


def remove_file(filename: str) -> bool:
    try:
        os.remove(os.path.join(UI_RUNS_DIR, filename))
        return True
    except OSError:
        return False


# ----------------------------------------------------------------
# 结果落盘
# ----------------------------------------------------------------

def result_filename(task: str) -> str:
    return f"{task}.json"


def save_result(task: str, payload: dict) -> bool:
    """保存一次任务结果（附带生成时间）。失败返回 False，不抛异常。"""
    if payload is None:
        return False
    return _atomic_write_json(result_filename(task), {
        "task": task,
        "saved_ts": time.time(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "payload": payload,
    })


def load_result(task: str) -> dict | None:
    """读取上次结果，返回 {"saved_ts","generated_at","payload"} 或 None"""
    record = _read_json(result_filename(task))
    return record if isinstance(record, dict) and "payload" in record else None


def clear_result(task: str) -> bool:
    path = os.path.join(UI_RUNS_DIR, result_filename(task))
    try:
        os.remove(path)
        return True
    except OSError:
        return False


# ----------------------------------------------------------------
# 运行标记
#
# 设计要点（P1 修复 v2）：**只有"有心跳"才算在运行**。
#
# 最初实现只记 started_ts，于是它实际退化成一个计时器：浏览器刷新后
#   - Streamlit 的停止请求会在脚本下一次调用 st.* 时抛 StopException 生效，
#     而我们每次进度更新都要画进度条（那是 st.* 调用）→ 选股在几秒内就被打断，
#     根本没有"在后台继续跑"；
#   - 标记的清除写在 finally 里、且排在 progress_bar.empty() 之后 → 再次抛异常，
#     清除语句被跳过 → 标记一直留到 15 分钟"过期"为止，页面持续谎报"运行中"。
# 现在：心跳由 **core/ui_tasks 的独立看门狗线程**按时刷新（不依赖业务代码何时
#       回调进度），心跳超时（默认 180 秒）即视为已中断；
#       退出路径用 task_guard 保证"先清标记"，不依赖 Streamlit 运行时。
#
# ⚠️ 不要退回"只在进度回调里刷心跳"的写法：自然语言选股的「问财初筛」整段没有
#    回调，超过 180 秒就会把**仍在后台运行**的任务判成崩溃，还会删掉它要用的
#    进度文件——用户看到的是"提示上一次选股没有正常结束，但后台还在跑"。
# --------------------------------------------------------------------------------

HEARTBEAT_TIMEOUT_SECONDS = 180      # 超过该时长没有心跳 → 判定为已中断
HEARTBEAT_MAX_AGE_SECONDS = 3600     # 兜底上限（即使心跳异常频繁也不能无限挂）


def _load_running_raw() -> dict:
    data = _read_json(RUNNING_FILE)
    return data if isinstance(data, dict) else {}


def mark_running(task: str, meta: dict = None, run_id: str = None) -> bool:
    now = time.time()
    # 读-改-写必须**整体**串行化：自从有了心跳看门狗，标记文件有多个写者
    # （工作线程刷心跳、看门狗定时刷心跳、页面清标记）。只在写入那一步加锁
    # 不够——两个写者可以各自读到同一份旧数据，后写的把先写的改动覆盖掉；
    # 更坏的情况是 clear 之后 watchdog 拿着旧数据回写，把标记"复活"成运行中。
    with _FILE_LOCK:
        data = _load_running_raw()
        data[task] = {
            "started_ts": now,
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "heartbeat_ts": now,
            "pid": os.getpid(),
            "run_id": run_id or "",
            "meta": dict(meta or {}),
        }
        return _atomic_write_json(RUNNING_FILE, data)


def heartbeat(task: str) -> bool:
    """刷新心跳（工作线程的进度回调 + 独立看门狗都会调用）。

    任务未登记时返回 False 且**不做任何写入**——这一点很关键：看门狗依赖它
    判断"任务已经结束了，我该退出"，也保证不会把刚清掉的标记又写回来。
    只写文件、不碰 Streamlit 运行时，因此可以在任何线程安全调用。
    """
    with _FILE_LOCK:
        data = _load_running_raw()
        record = data.get(task)
        if not isinstance(record, dict):
            return False
        record["heartbeat_ts"] = time.time()
        return _atomic_write_json(RUNNING_FILE, data)


def clear_running(task: str, run_id: str = None) -> bool:
    """清除运行标记；给了 run_id 时只清"自己那一轮"的标记。

    为什么需要 run_id：任务跑在后台线程里，上一轮线程的 finally 可能恰好落在
    下一轮 start() 之后——如果不加区分，旧线程会把新一轮的标记清掉，
    于是新一轮被判成"已中断"、重复启动的闸门也失效了。
    """
    with _FILE_LOCK:
        data = _load_running_raw()
        record = data.get(task)
        if record is None:
            return True
        if run_id is not None and isinstance(record, dict) and record.get("run_id") \
                and record["run_id"] != run_id:
            return False                      # 这是新一轮的标记，不能动
        data.pop(task, None)
        return _atomic_write_json(RUNNING_FILE, data)


def last_heartbeat_age(record) -> float:
    """距最近一次心跳的秒数（没有心跳字段时退回到 started_ts）"""
    if not isinstance(record, dict):
        return float("inf")
    ts = record.get("heartbeat_ts") or record.get("started_ts")
    if not isinstance(ts, (int, float)):
        return float("inf")
    return max(0.0, time.time() - ts)


def _hb_timeout(value=None) -> float:
    """解析心跳超时：显式传值优先，否则**运行时**读模块常量。

    ⚠️⚠️ 为什么不能写 `def f(..., heartbeat_timeout=HEARTBEAT_TIMEOUT_SECONDS)`：
    Python 的默认参数在**函数定义时**求值一次并固化，之后再改模块常量
    （`monkeypatch.setattr(ui_runs, "HEARTBEAT_TIMEOUT_SECONDS", 0)`）**完全无效**。
    实测后果（2026-10-23）：`test_stale_marker_does_not_block_forever` 想验证
    "心跳停了的旧标记不该永远挡住新任务"，但超时从未被改小 ⇒ 用例恒红，
    而且**真正想看的行为一次都没被验证过**（测试红是噪音，测试绿是假象）。
    ⇒ 统一走本函数做**运行时解析**，让配置与测试都能真正生效。

    ⚠️ 返回 **float**，不要 `int()`：
    测试为了在 1 秒内验证看门狗，会把超时压到 0.6 秒；`int(0.6)` == 0
    ⇒ 任何正心跳年龄都"超时" ⇒ 任务刚启动就被判成已中断（我自己踩过一次）。
    """
    if value is not None:
        return float(value)
    return float(HEARTBEAT_TIMEOUT_SECONDS)


def is_alive(record, heartbeat_timeout: int = None) -> bool:
    """是否仍然活着：近期有心跳，且未超过兜底上限"""
    if not isinstance(record, dict):
        return False
    if last_heartbeat_age(record) > _hb_timeout(heartbeat_timeout):
        return False
    return age_seconds(record) <= HEARTBEAT_MAX_AGE_SECONDS


def get_running(task: str, heartbeat_timeout: int = None) -> dict | None:
    """返回**仍然活着**的运行标记；不存在或心跳已停（视为中断）返回 None"""
    record = _load_running_raw().get(task)
    return record if is_alive(record, heartbeat_timeout) else None


def get_running_any(task: str) -> dict | None:
    """不做存活判断，原样返回标记（供 UI 区分"在跑"与"已中断"）"""
    record = _load_running_raw().get(task)
    return record if isinstance(record, dict) else None


def running_tasks(heartbeat_timeout: int = None) -> dict:
    return {t: r for t, r in _load_running_raw().items()
            if is_alive(r, heartbeat_timeout)}


def clear_dead_running(heartbeat_timeout: int = None) -> list:
    """清掉已中断（心跳停止）的标记，返回被清理的任务名列表"""
    with _FILE_LOCK:
        data = _load_running_raw()
        dead = [t for t, r in data.items() if not is_alive(r, heartbeat_timeout)]
        for t in dead:
            data.pop(t, None)
        if dead:
            _atomic_write_json(RUNNING_FILE, data)
        return dead


def clear_stale_running(stale_after_seconds: int = DEFAULT_STALE_SECONDS) -> int:
    """兼容旧接口：清掉超时的运行标记（孤儿），返回清理条数"""
    with _FILE_LOCK:
        data = _load_running_raw()
        stale = [t for t, r in data.items()
                 if not isinstance(r, dict) or age_seconds(r) > stale_after_seconds]
        for t in stale:
            data.pop(t, None)
        if stale:
            _atomic_write_json(RUNNING_FILE, data)
        return len(stale)


@contextmanager
def task_guard(task: str, meta: dict = None):
    """任务运行标记上下文：进入时登记，退出时**先清标记**。

    退出路径只做文件写入、不调用任何 Streamlit API，因此在被 StopException
    打断（页面刷新/关闭导致脚本运行被停止）时也能清掉标记。
    这正是早期实现的漏洞：`finally` 里先调 `progress_bar.empty()`，它同样会抛异常，
    清除标记的语句被跳过，标记一直留到"过期"为止，页面持续谎报"仍在运行"。
    """
    mark_running(task, meta)
    try:
        yield
    finally:
        clear_running(task)


# ----------------------------------------------------------------
# 时间显示
# ----------------------------------------------------------------

def age_seconds(record) -> float:
    """记录（running 或 result）距今秒数；无法判断时返回一个很大值"""
    if isinstance(record, (int, float)):
        return float(record)
    if isinstance(record, dict):
        ts = record.get("started_ts") or record.get("saved_ts")
        if isinstance(ts, (int, float)):
            return max(0.0, time.time() - ts)
    return float("inf")


def age_text(record) -> str:
    """把秒数/记录格式化成可读时长，例如 "2 分 31 秒" """
    seconds = age_seconds(record)
    if seconds == float("inf"):
        return "未知"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} 秒"
    if seconds < 3600:
        return f"{seconds // 60} 分 {seconds % 60} 秒"
    return f"{seconds // 3600} 小时 {(seconds % 3600) // 60} 分"
