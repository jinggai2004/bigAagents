"""UI 长任务的后台执行（P1 v3：真正的"刷新不中断"）

前两版的教训：
  v1 把选股放在 Streamlit 的**脚本运行里同步执行**——进度回调要调用 st.*，
     而 Streamlit 的停止请求正是在脚本下一次调用 st.* 时抛 StopException 生效，
     于是"刷新页面"必然打断任务，根本没有后台继续跑；
  v2 加了运行标记，但清标记的语句排在 st.* 调用之后，被异常跳过 →
     标记留到"过期"为止，页面持续谎报"运行中"（退化成计时器）。

本模块把任务搬到普通线程里：
  - 线程内**不调用任何 Streamlit API**，只写文件（进度 / 心跳 / 结果），
    所以浏览器断开、刷新、换标签页都不影响它跑完；
  - 页面只当读者：轮询进度文件画进度条，刷新后重新连上就接着显示；
  - 启动互斥依赖 ui_runs 的心跳存活判定（谁都不能靠一个开始时间戳猜）；
  - 支持协作式取消：写一个取消标记，工作线程在下一个进度点退出；
  - **心跳由独立看门狗线程按时刷新**，不依赖业务代码何时回调进度——
    否则"心跳停了"会被当成"进程没了"，把正在跑的任务误报成崩溃（见
    `_heartbeat_watchdog` 的注释，这是一次真实故障的修复）。
"""

import logging
import os
import threading
import time
from datetime import datetime

from core import ui_runs
from utils.task_control import TaskCancelled  # noqa: F401  （re-export，保持历史 import 路径可用）

_logger = logging.getLogger(__name__)


# 心跳看门狗的刷新间隔（秒）。必须**显著小于** ui_runs.HEARTBEAT_TIMEOUT_SECONDS
# （180 秒），给调度抖动和文件 IO 留足余量：单次漏敲不能导致误判。
HEARTBEAT_INTERVAL_SECONDS = 30

_RUNNERS: dict = {}
_BEAT_STOPS: dict = {}
_RUNNERS_LOCK = threading.Lock()


def progress_filename(task: str) -> str:
    return f"progress_{task}.json"


def cancel_filename(task: str) -> str:
    return f"cancel_{task}.json"


# ----------------------------------------------------------------
# 进度
# ----------------------------------------------------------------

def write_progress(task: str, payload: dict) -> bool:
    data = dict(payload or {})
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return ui_runs.write_json(progress_filename(task), data)


def read_progress(task: str) -> dict | None:
    data = ui_runs.read_json(progress_filename(task))
    return data if isinstance(data, dict) else None


def clear_progress(task: str) -> bool:
    return ui_runs.remove_file(progress_filename(task))


# ----------------------------------------------------------------
# 取消
# ----------------------------------------------------------------

def request_cancel(task: str) -> bool:
    return ui_runs.write_json(cancel_filename(task), {"requested_at": time.time()})


def is_cancelled(task: str) -> bool:
    return ui_runs.read_json(cancel_filename(task)) is not None


def clear_cancel(task: str) -> bool:
    return ui_runs.remove_file(cancel_filename(task))


# ----------------------------------------------------------------
# 运行与互斥
# ----------------------------------------------------------------

def is_running(task: str) -> bool:
    """是否有活着的后台任务（心跳判定，见 core/ui_runs）"""
    return ui_runs.get_running(
        task, heartbeat_timeout=ui_runs.HEARTBEAT_TIMEOUT_SECONDS) is not None


def start(task: str, fn, meta: dict = None) -> bool:
    """在后台线程里执行 `fn(progress)`；已有任务在跑时返回 False。

    `fn` 里**绝对不能调用任何 Streamlit API**（线程里没有 ScriptRunContext），
    需要进度就调用传入的 `progress(payload)`。
    """
    with _RUNNERS_LOCK:
        existing = _RUNNERS.get(task)
        if existing is not None and existing.is_alive():
            return False
        if is_running(task):
            return False
        # 先登记标记（同步），页面立刻就能显示"正在运行"，也缩小并发窗口
        run_id = f"{time.time_ns()}-{os.getpid()}"
        ui_runs.mark_running(task, meta, run_id=run_id)
        clear_cancel(task)
        write_progress(task, {"status": "starting", "percent": 0.0, "message": "准备中…",
                              "elapsed": 0.0})
        thread = threading.Thread(target=_run, args=(task, fn, run_id, meta), daemon=True,
                                  name=f"ui-task-{task}")
        _RUNNERS[task] = thread
        # 心跳看门狗：与业务进度回调**解耦**，保证"心跳停了"真等价于"进程没了"。
        # 顺序：先起工作线程再起看门狗，看门狗靠"工作线程是否还活着"决定何时退出，
        # 若反过来，看门狗可能在 is_alive() 还是 False 时就自行返回。
        stop_beat = threading.Event()
        _BEAT_STOPS[task] = stop_beat
        thread.start()
        threading.Thread(target=_heartbeat_watchdog, args=(task, stop_beat, thread),
                         daemon=True, name=f"ui-beat-{task}").start()
        return True


def _heartbeat_watchdog(task: str, stop: threading.Event, worker: threading.Thread,
                        interval: float = None):
    """独立刷新任务心跳 —— 与业务进度回调解耦（真实故障的修复）。

    为什么必须有它：心跳原先**只在 `progress()` 里刷新**，而 `progress()` 由选股/
    分析流程自己决定何时调用。自然语言选股在「问财初筛」这一段只在查询前后各报
    一次进度，中间整段问财请求（会重试、会限频）没有任何回调；即便进入逐只评分，
    数据源降级期**单只也能超过 180 秒**（实测约 8 分钟/只）。只要超过
    `ui_runs.HEARTBEAT_TIMEOUT_SECONDS`（180 秒），页面就会把**仍在后台正常运行**
    的任务判成"进程被关闭或崩溃"，并顺手把它还要用的进度文件删掉——
    用户看到的正是"提示上一次选股没有正常结束，但后台还在跑"。

    看门狗只做一件事：任务还活着就按时敲心跳。于是「心跳停了」才真正等价于
    「进程没了」（守护线程随进程一起消失），存活判定的语义才成立。

    退出条件用"工作线程是否还活着"而不是"标记是否存在"：标记可能被别的会话误清，
    而工作线程还在跑——此时看门狗必须继续值守，等 `progress()` 把标记自愈回来之后
    接着刷心跳。注意 `ui_runs.heartbeat()` 对不存在的标记**不写文件**，
    所以本看门狗不会把已清掉的标记复活。
    """
    if interval is None:
        interval = HEARTBEAT_INTERVAL_SECONDS
    while not stop.wait(interval):
        if not worker.is_alive():
            return
        ui_runs.heartbeat(task)


def _run(task: str, fn, run_id: str, meta: dict = None):
    started = time.time()

    def progress(payload: dict):
        if is_cancelled(task):
            raise TaskCancelled()
        data = dict(payload or {})
        data["status"] = "running"
        data["elapsed"] = round(time.time() - started, 1)
        write_progress(task, data)
        # 自愈：标记若已丢失（例如被误判成崩溃的页面 clear_dead_running 清掉、
        # 或被另一个会话清理），由**正在执行的自己**重新登记——我们是这条任务
        # 是否活着的权威：线程还在跑，就说明进程和任务都还在。
        # 带上自己的 run_id，避免后续收尾时误清新一轮的标记。
        if not ui_runs.heartbeat(task):
            ui_runs.mark_running(task, meta=meta, run_id=run_id)

    try:
        fn(progress)
        write_progress(task, {"status": "done", "percent": 100.0, "message": "完成",
                              "elapsed": round(time.time() - started, 1)})
    except TaskCancelled:
        _logger.info("[UI任务] %s 已取消", task)
        write_progress(task, {"status": "cancelled", "message": "已取消",
                              "elapsed": round(time.time() - started, 1)})
    except Exception as e:                      # noqa: BLE001 - 线程里必须兜住所有异常
        _logger.exception("[UI任务] %s 失败", task)
        write_progress(task, {"status": "error",
                              "message": f"{type(e).__name__}: {e}",
                              "elapsed": round(time.time() - started, 1)})
    finally:
        # 顺序要紧：**先停看门狗，再清标记**。反过来的话，看门狗可能刚读完
        # 标记、还没写完心跳，就把已经清掉的标记又刷回去（phantom running）。
        # 真正的兜底在 ui_runs：mark/heartbeat/clear 的"读-改-写"整体串行化，
        # 所以即使两者交错，也只会有一个结果。
        beat = _BEAT_STOPS.pop(task, None)
        if beat is not None:
            beat.set()
        clear_cancel(task)
        # 只写文件、不碰 Streamlit：无论正常结束还是被取消都能清掉标记。
        # 带上 run_id：万一新一轮任务已经启动，旧线程不会把它的标记误清。
        ui_runs.clear_running(task, run_id=run_id)


def reset_for_tests():
    """清空线程登记（测试用；不碰磁盘状态）"""
    with _RUNNERS_LOCK:
        _RUNNERS.clear()
        for ev in _BEAT_STOPS.values():
            ev.set()
        _BEAT_STOPS.clear()
