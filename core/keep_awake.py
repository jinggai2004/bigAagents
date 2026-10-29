"""现代待机保活：让调度器所在机器别在盘中睡过去。

## 为什么需要这个模块

本机是 **S0 低电量待机（新式待机）** 的笔记本。2026-10-13 曾把
`powercfg /change standby-timeout-ac 0`（"永不待机"）设为 0 并**实测生效**，
但 2026-10-14 盘中**仍然睡了过去**：

| 证据 | 值 |
|---|---|
| `powercfg /q SCHEME_CURRENT SUB_SLEEP STANDBYIDLE` | AC/DC 均 `0x0`（已生效） |
| Kernel-Power **506**（进入新式待机） | `2026-10-14 13:32:54` |
| Kernel-Power **507**（退出新式待机） | `2026-10-14 15:03:14` |
| 同期调度日志 | `盘中异动预警` 13:34 → 15:04 **缺口 90 分钟**，完全吻合 |

⇒ **"待机超时 = 0" 只关闭了"空闲超时"这一条入口**。新式待机的机器上
**熄屏本身就是入待机路径**（本机显示器超时 AC 600s / DC 300s），
而 `SUB_BUTTONS` 子组里**没有 LIDACTION**（合盖项未向系统暴露）
⇒ 合盖/熄屏这两条入口 `powercfg` 都管不住。

⇒ 结论：**不要指望靠改系统设置把机器钉醒**。正确做法是让**进程自己持有电源请求**
（`PowerRequestSystemRequired` 的 Win32 等价物），这在应用层可行、可回滚、换机也不丢。

## 实现

`SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`：
- `ES_CONTINUOUS`：本次设置持续生效，直到下一次调用（不必反复刷新）
- `ES_SYSTEM_REQUIRED`：等价于 **system-required 电源请求** ⇒ 系统不进入新式待机
- ⚠️ **刻意不加 `ES_DISPLAY_REQUIRED`**：那会让**屏幕常亮**，对一台用户白天还要用的
  笔记本是骚扰。允许熄屏、但不允许入待机 —— 这正是我们要的。

Win32 的 `SetThreadExecutionState` **按线程生效**（不是按进程），且**线程退出即自动释放**
电源请求。所以这里起一个**守护线程**长期持有：线程活着 = 请求有效；
进程被 kill ⇒ 线程消失 ⇒ 请求自动释放（不会"泄漏"成机器永不睡眠）。

## 边界（必须如实说，别过度承诺）

- 本模块**只能阻止"因空闲/熄屏而进入新式待机"**。
- **合盖**在部分机型上由固件直接强制入待机，软件请求拦不住（本机未暴露 LIDACTION，
  无法验证属哪一种）。⇒ 若合盖仍导致漏跑，只能改合盖动作或换一台不休眠的机器。
- **非 Windows** 或调用失败 ⇒ `acquire()` 返回 False 并给出原因，**绝不抛异常**：
  保活失败不能变成调度器起不来的新理由（本项目最大的事故正是"调度器没启动而无人察觉"）。
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading

_logger = logging.getLogger(__name__)

# Win32 常量
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_CONTINUOUS = 0x80000000

# 守护线程重刷间隔（秒）。ES_CONTINUOUS 理论上一次就够，但部分机型/驱动会重置请求，
# 定期重刷成本可忽略（一次系统调用/分钟），换来更强的稳健性。
REFRESH_SECONDS = 60

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop = threading.Event()
_state: dict = {"supported": None, "active": False, "reason": "未启用", "calls": 0,
                "last_return": None}


def supported() -> bool:
    """本机是否支持（只在 Windows 上有实现）。"""
    return sys.platform == "win32"


def enabled_by_env() -> bool:
    """是否允许启用：默认启用；`BIGA_KEEP_AWAKE=0` 可关闭。

    留开关是因为它**会让机器不睡**（耗电增加）——用户可能出于续航考虑主动关掉，
    那时他接受的代价就是"关机/睡眠时段的定时任务会漏"。这个取舍必须交给他。
    """
    return os.environ.get("BIGA_KEEP_AWAKE", "1").strip().lower() not in ("0", "false", "no", "off")


def _set_state(flags: int) -> tuple[bool, int | None]:
    """调一次 SetThreadExecutionState。

    Returns:
        `(是否发生异常, 原始返回值)`。

    ⚠️ **不能把返回 0 当失败**：该 API 的返回值是"**上一个**执行状态"，
    而"失败"**也**返回 0 —— 两者**同码**。一个从未调用过它的线程，首次调用成功时
    返回的就是 0。若按"0 = 失败"处理，首刷就会被误判为失败 → **保活被静默关掉**，
    而日志只留一句 warning。**误判成功只是少了保护信息；误判失败会真的关掉保护**，
    两者代价不对称 ⇒ 这里只在**抛异常**时才算失败，0 与"上次为 0"的歧义交给
    `last_return` 如实记录，由独立的「调度缺口检测」提供地面真值。
    """
    try:
        kernel32 = ctypes.WinDLL("kernel32")       # noqa: F821 —— 仅 win32 可达
        kernel32.SetThreadExecutionState.restype = ctypes.c_uint
        kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
        prev = kernel32.SetThreadExecutionState(flags)
        return False, int(prev)
    except Exception as e:                          # noqa: BLE001
        _logger.warning("[保活] SetThreadExecutionState(%s) 失败: %s", hex(flags), e)
        return True, None


def _worker() -> None:
    """守护线程：持有 system-required 电源请求，直到 release()。"""
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    first = True
    while not _stop.is_set():
        failed, prev = _set_state(flags)
        if failed:
            with _lock:
                _state["active"] = False
                _state["reason"] = "系统调用异常（见日志），保活未生效"
            return
        with _lock:
            _state["last_return"] = prev
            _state["calls"] += 1
            _state["active"] = True
            _state["reason"] = "已持有 system-required 电源请求（允许熄屏、阻止入待机）"
        if first:
            first = False
            _logger.info("[保活] 已请求阻止系统进入新式待机"
                         "（SetThreadExecutionState=0x%08X，返回 0x%08X）。"
                         "注意：合盖若被固件强制待机，软件层拦不住。",
                         flags, prev or 0)
        # 用 Event.wait 替代 sleep：release() 时能立刻退出
        _stop.wait(REFRESH_SECONDS)


def acquire() -> bool:
    """开始保活。成功返回 True；不支持/被禁用返回 False（均不抛）。"""
    global _thread
    if not enabled_by_env():
        with _lock:
            _state.update(supported=supported(), active=False,
                          reason="已由 BIGA_KEEP_AWAKE=0 关闭")
        return False
    if not supported():
        with _lock:
            _state.update(supported=False, active=False,
                          reason="非 Windows 平台：无 SetThreadExecutionState，跳过")
        return False
    with _lock:
        if _thread is not None and _thread.is_alive():
            return True
        _stop.clear()
        _thread = threading.Thread(target=_worker, name="biga-keep-awake", daemon=True)
        _thread.start()
    return True


def release() -> None:
    """释放请求（线程退出 ⇒ 电源请求自动失效）。允许重复调用。"""
    global _thread
    _stop.set()
    t = _thread
    if t is not None and t.is_alive():
        t.join(timeout=2)
    _thread = None
    with _lock:
        _state["active"] = False
        _state["reason"] = "已释放"


def is_active() -> bool:
    with _lock:
        return bool(_state["active"])


def describe() -> dict:
    """供健康检查 / UI 展示的结构化状态（只读快照）。"""
    with _lock:
        return {"supported": supported() if _state["supported"] is None else _state["supported"],
                "enabled": enabled_by_env(),
                "active": bool(_state["active"]),
                "reason": _state["reason"],
                "refreshes": _state["calls"],
                "last_return": _state["last_return"]}


class keep_awake:  # noqa: N801 —— 用法上更像上下文管理器，故意小写
    """`with keep_awake(): ...` —— 退出时自动释放。异常安全。"""

    def __enter__(self):
        self._ok = acquire()
        return self._ok

    def __exit__(self, *exc):
        release()
        return False
