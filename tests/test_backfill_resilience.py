# -*- coding: utf-8 -*-
"""财务回填的**抗僵死**测试。

## 为什么有这个文件（2026-10-31 事故）

`--sample 1500` 那轮挂了 **16 小时 55 分**：日志停在 `600/1494`、落盘文件停在 00:38，
而进程还活着并占着 8 条 BaoStock 连接。两个原因叠加：

1. **BaoStock 建的 socket 没有超时**（内部 `socket.socket()` 不设 `settimeout`）⇒
   限频时 worker 卡在 `recv()` 上永不返回；
2. **主进程用 `as_completed` 等结果**，没有看门狗 ⇒ 永远等下去；
3. 更糟的是它最终报了 **exit 0**，看起来像"正常跑完"，于是没人发现只做了 40%。

这个文件守三件事：超时真的设上了、卡死能被判定、卡死**必须返回非 0**。
"""
import socket
import sys
import types

import pytest

from scripts import backfill_financials as bf


def test_worker_init_sets_socket_timeout(monkeypatch):
    """`bs.login()` 之前必须设好 socket 默认超时。

    不设的后果不是"慢"，是**永久挂住**：BaoStock 的 `recv()` 没有超时。
    """
    fake = types.SimpleNamespace(
        login=lambda: types.SimpleNamespace(error_code="0", error_msg=""))
    monkeypatch.setitem(sys.modules, "baostock", fake)

    old = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(None)
        bf._worker_init()
        assert socket.getdefaulttimeout() == bf.SOCK_TIMEOUT
    finally:
        socket.setdefaulttimeout(old)


def test_worker_init_retries_login_and_gives_up(monkeypatch):
    """login 连续失败 ⇒ 不抛异常炸掉整个池，只报错（该 worker 取不到数，
    但其它 worker 还能干活；抛异常会让 PoolExecutor 反复重启 initializer）。"""
    calls = {"n": 0}

    def _boom():
        calls["n"] += 1
        raise RuntimeError("login refused")

    monkeypatch.setitem(sys.modules, "baostock",
                        types.SimpleNamespace(login=_boom))
    old = socket.getdefaulttimeout()
    try:
        bf._worker_init()                       # 不应抛
        assert calls["n"] == bf.LOGIN_RETRIES
    finally:
        socket.setdefaulttimeout(old)


def test_stalled_run_returns_nonzero(monkeypatch, tmp_path):
    """**核心守卫**：worker 永不返回结果 ⇒ 中止并返回非 0。

    上一轮僵死 16 小时后报 exit 0，这是最坏的失效形态——脚本说"成功"，
    实际上只做了 40%，而调用方（人或调度）无法分辨。
    """
    from concurrent.futures import Future

    class _FakeEx:
        """submit 返回永不完成的 Future，模拟 worker 全部卡在 socket 上。"""
        _processes = {}

        def __init__(self, *a, **k):
            pass

        def submit(self, *a, **k):
            return Future()                     # 永远 pending

        def shutdown(self, **k):
            pass

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _FakeEx)
    monkeypatch.setattr(bf, "WATCHDOG_TICK", 0.05)
    monkeypatch.setattr(bf, "STALL_LIMIT", 0.2)
    monkeypatch.setattr(bf, "_table_dir", lambda table: tmp_path)

    from data import bars
    monkeypatch.setattr(bars, "available_codes", lambda: ["600000", "600001"])

    rc = bf.run(limit=None, years=[2024], force=False, table="profit")
    assert rc == 3, f"卡死必须返回 3（非 0），实际 {rc}"


def test_watchdog_constants_are_sane():
    """看门狗参数不能退化成"永不触发"或"频繁误报"。"""
    assert bf.WATCHDOG_TICK <= bf.STALL_LIMIT / 2, "检查间隔应明显小于卡死阈值"
    assert 60 <= bf.STALL_LIMIT <= 1800, "卡死阈值应在 1~30 分钟量级"
    assert 10 <= bf.SOCK_TIMEOUT <= 120, "socket 超时应短于卡死阈值的一半量级"
