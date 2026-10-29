# -*- coding: utf-8 -*-
"""进程级单实例闸门（OS 命名互斥）。

## 为什么不是"读 pid 文件 / 读心跳"
两个都属同一类错误：**「检查-后-启动」之间有一个窗口**，窗口里两个进程都会通过。

2026-10-21 实测事故：心跳是"启动成功后"才写的，三个进程在同一秒内启动时谁都没写心跳
⇒ `read_heartbeat()` 返空 ⇒ 原来那句 `return True` 让**三个实例全部通过**；
当天 09:19:36.875 有三份「已注册任务」、09:25:00.003 三份 `Running job "盘前选股"`，
Stage 1 独占 3211 秒 × 3 并发互抢网络与 SQLite。

2026-10-20 同类事故：连点两次启动 bat ⇒ 两个 watchdog，`consecutive` 计数互相覆盖，
"连续 3 次失败才停手"永远数不到 3。

⇒ 换内核级对象：**进程一退出自动释放**，不存在窗口。

## 语义
- `acquire_os_mutex("Global\\bigaagents_scheduler_mutex")` 拿到 = 本进程是唯一实例；
- **同进程内重复调用返回 True**（否则页面按钮/多入口会互相踢掉自己）；
- 拿不到 ≠ 报错，只是"别人已经在跑"，调用方自己决定要不要退出；
- 创建失败（权限/API 不可用）**退回放行** —— 单实例是护栏，不该让程序起不来。

⚠️ Windows 上 `Local\\` 是**会话内**、`Global\\` 才是**全机**。桌面双击与计划任务
分属不同会话，只有 `Global\\` 挡得住双开。
"""
from __future__ import annotations

import logging
import os
import tempfile

_logger = logging.getLogger(__name__)

# 进程存活期间一直持有；退出由 OS 回收。键 = mutex 名
_MUTEX_HANDLES: dict = {}

_PREFIX = "bigAagents"


def mutex_name(app: str, scope: str = "Global") -> str:
    """拼一个本项目专用的互斥名（`Global\\bigAagents_<app>_mutex`）。"""
    return f"{scope}\\{_PREFIX}_{app}_mutex"


def acquire_os_mutex(name: str) -> bool:
    """尝试成为 `name` 的唯一持有者；拿到/已持有 ⇒ True，已被占用 ⇒ False。"""
    if not name:
        # 空名会让 CreateMutexW 建一个**无名**互斥，等于没有保护 ⇒ 宁可显式拒绝
        _logger.warning("[单实例] 互斥名为空，拒绝启动（这一定是调用方写错了）")
        return False
    if name in _MUTEX_HANDLES:
        return True                     # 本进程已持有（重复调用不该误判）

    if os.name == "nt":
        import ctypes
        # use_last_error=True：否则 GetLastError 会被 Python 内部调用覆盖
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.CreateMutexW(None, False, name)
        err = ctypes.get_last_error()
        if not handle:
            _logger.warning("[单实例] CreateMutexW 失败（err=%s），退回放行以免起不来", err)
            return True
        if err == 183:                  # ERROR_ALREADY_EXISTS
            kernel32.CloseHandle(handle)
            return False
        _MUTEX_HANDLES[name] = handle
        return True

    # 非 Windows：O_EXCL 锁文件（进程退出不自动清，但本项目只在 Windows 跑）
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    lock = os.path.join(tempfile.gettempdir(), f"{safe}.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as e:
        # 这是**设计分支**（O_EXCL 撞上已有锁 = 另一个实例在跑），不是故障；
        # 但按项目铁律"新增 except 必须留痕"，否则"为什么起不来"只能靠猜。
        _logger.debug("[单实例] 锁文件已被占用，判定为已有实例在运行: %s", e)
        return False
    except OSError as e:
        _logger.warning("[单实例] 创建锁文件失败（%s），退回放行以免起不来", e)
        return True
    try:
        os.write(fd, str(os.getpid()).encode("ascii"))
    except OSError as e:
        _logger.debug("[单实例] 写锁文件内容失败（不影响互斥语义）: %s", e)
    finally:
        os.close(fd)
    _MUTEX_HANDLES[name] = -1           # 占位：表示本进程已持有
    return True


def release_os_mutex(name: str) -> bool:
    """主动释放（一般不需要：进程退出由 OS 回收）。返回是否释放了东西。"""
    handle = _MUTEX_HANDLES.pop(name, None)
    if handle is None:
        return False
    if handle != -1 and os.name == "nt":
        import ctypes
        try:
            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)
        except OSError as e:            # pragma: no cover - 只在句柄异常时触发
            _logger.debug("[单实例] CloseHandle 失败（忽略）: %s", e)
    return True
