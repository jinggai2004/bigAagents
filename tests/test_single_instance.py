# -*- coding: utf-8 -*-
"""`core/single_instance` 的互斥语义（P2-9，2026-10-23）。

## 为什么这个必须真测，不能只测"函数返回 True"
这条护栏挡的是**跨进程**双开：
- 2026-10-21 三个调度器并发跑三遍选股（各 3211s）；
- 2026-10-20 连点两次启动 bat ⇒ 两个 watchdog，`consecutive` 计数互相覆盖。

这两个事故的共同点是**单进程内测不出来**：同进程重复调用只会命中"已持有"分支，
永远绿。所以这里必须有**真子进程**的用例 —— 父进程持锁、子进程去抢，必须失败。
"""
from __future__ import annotations

import os
import subprocess
import sys
import uuid

import pytest

from core import single_instance as S

pytestmark = pytest.mark.skipif(os.name != "nt", reason="命名 mutex 只在 Windows 生效")


def _unique(tag: str = "t") -> str:
    return S.mutex_name(f"{tag}_{uuid.uuid4().hex[:8]}")


def _child_try(name: str) -> bool:
    """在**独立进程**里抢同一把锁，返回它是否抢到。"""
    code = (
        "import sys;"
        f"sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r});"
        "from core import single_instance as S;"
        f"print('GOT' if S.acquire_os_mutex({name!r}) else 'BUSY')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    assert r.returncode == 0, r.stderr
    return "GOT" in r.stdout


class TestNaming:
    def test_default_is_global_scope(self):
        """`Local\\` 只是会话内 —— 桌面双击和计划任务分属不同会话，挡不住双开。"""
        assert S.mutex_name("scheduler").startswith("Global\\")
        assert "scheduler" in S.mutex_name("scheduler")

    def test_different_apps_do_not_collide(self):
        assert S.mutex_name("scheduler") != S.mutex_name("watchdog")


class TestAcquire:
    def test_empty_name_is_refused(self):
        """空名会让 CreateMutexW 建无名互斥 = 没有保护，宁可直接拒绝。"""
        assert S.acquire_os_mutex("") is False

    def test_same_process_reentry_is_allowed(self):
        """同进程重复调用必须 True：否则页面按钮/多入口会互相踢掉自己。"""
        name = _unique("reentry")
        try:
            assert S.acquire_os_mutex(name) is True
            assert S.acquire_os_mutex(name) is True
        finally:
            S.release_os_mutex(name)

    def test_second_process_is_blocked(self):
        """⭐ 核心：父进程持锁时，**另一个进程**必须抢不到。"""
        name = _unique("xproc")
        assert S.acquire_os_mutex(name) is True
        try:
            assert _child_try(name) is False, "另一个进程竟然也抢到了 ⇒ 单实例保护失效"
        finally:
            S.release_os_mutex(name)

    def test_released_lock_can_be_taken_again(self):
        """释放后必须能重新拿到（否则"重启一次就再也起不来"）。"""
        name = _unique("rel")
        assert S.acquire_os_mutex(name) is True
        assert S.release_os_mutex(name) is True
        assert S.acquire_os_mutex(name) is True
        try:
            # 释放后子进程仍抢不到，说明是"父进程重新持有了"，不是锁坏了
            assert _child_try(name) is False
        finally:
            S.release_os_mutex(name)
        # 真正放手后，子进程应该能拿到
        assert _child_try(name) is True

    def test_release_unknown_name_is_false(self):
        assert S.release_os_mutex(_unique("never")) is False


class TestSchedulerAndWatchdogWiring:
    """接线守护：两个入口必须**真的**用不同的锁，且都走公共实现。"""

    def test_scheduler_uses_shared_module(self):
        import scripts.run_scheduler as R
        assert R._MUTEX_NAME == S.mutex_name("scheduler")
        assert R._MUTEX_HANDLES is S._MUTEX_HANDLES

    def test_watchdog_uses_its_own_lock(self):
        import scripts.watchdog as W
        assert W._MUTEX_NAME == S.mutex_name("watchdog")
        assert W._MUTEX_NAME != S.mutex_name("scheduler")

    def test_watchdog_acquire_singleton_takes_mutex_first(self, monkeypatch, tmp_path):
        """守护拿不到互斥时必须直接退出，**且不写 pid 文件**（不覆盖别人的 pid）。"""
        import scripts.watchdog as W
        monkeypatch.setattr(W, "PIDFILE", tmp_path / ".watchdog.pid")
        monkeypatch.setattr(W.single_instance, "acquire_os_mutex", lambda name: False)
        assert W.acquire_singleton() is False
        assert not (tmp_path / ".watchdog.pid").exists()
