# -*- coding: utf-8 -*-
"""`core/keep_awake` 的测试（防新式待机保活）。

⚠️ 测试**不依赖本机是否 Windows、也不真去改电源状态**：
只验证"该调的都调了 / 该拒绝的都拒绝了 / 绝不抛"，
以及一条**结构性守护**（见 `test_zero_return_must_not_be_treated_as_failure`）。
"""
import sys

import pytest

from core import keep_awake as KA


@pytest.fixture(autouse=True)
def _cleanup():
    """每个用例后释放，避免守护线程串场。"""
    yield
    KA.release()


def test_describe_shape():
    d = KA.describe()
    for k in ("supported", "enabled", "active", "reason", "refreshes"):
        assert k in d, k
    assert isinstance(KA.supported(), bool)
    assert isinstance(KA.enabled_by_env(), bool)


def test_can_be_disabled_by_env(monkeypatch):
    """`BIGA_KEEP_AWAKE=0` → 不启用，且如实说明（用户可能为续航主动关掉）"""
    monkeypatch.setenv("BIGA_KEEP_AWAKE", "0")
    assert KA.acquire() is False
    d = KA.describe()
    assert d["active"] is False and "BIGA_KEEP_AWAKE" in d["reason"]


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", ""])
def test_env_defaults_to_enabled(val, monkeypatch):
    monkeypatch.setenv("BIGA_KEEP_AWAKE", val)
    assert KA.enabled_by_env() is True


def test_non_windows_returns_false_without_raising(monkeypatch):
    """非 Windows：返回 False + 说明原因，**绝不抛**（保活失败不能变成调度器起不来的理由）"""
    monkeypatch.setattr(KA.sys, "platform", "linux")
    assert KA.acquire() is False
    assert KA.describe()["active"] is False
    assert "非 Windows" in KA.describe()["reason"]


def test_release_is_idempotent():
    KA.release()
    KA.release()
    assert KA.is_active() is False


def test_context_manager_releases():
    with KA.keep_awake() as ok:
        assert isinstance(ok, bool)
    assert KA.is_active() is False


def test_worker_survives_call_exception(monkeypatch):
    """系统调用抛异常 → 标记未生效并退出线程，不冒泡到调用方。"""
    monkeypatch.setattr(KA, "_set_state", lambda flags: (True, None))
    assert KA.acquire() is True              # 起线程本身成功
    import time
    for _ in range(50):                      # 等线程处理完首次调用
        if KA.describe()["active"] is False and "异常" in KA.describe()["reason"]:
            break
        time.sleep(0.02)
    assert KA.describe()["active"] is False
    assert "异常" in KA.describe()["reason"]


def test_zero_return_must_not_be_treated_as_failure():
    """⚠️ **结构性守护**：`SetThreadExecutionState` 返回 0 时**不得**判失败。

    该 API 的返回值是"**上一个**执行状态"；**失败也返回 0** —— 两者同码。
    一个从未调用过它的线程，首次调用**成功**时返回的就是 0。
    若按"0 = 失败"处理，首刷就会被误判成失败 → **保活被静默关掉**，
    而唯一痕迹是一行 warning。

    代价不对称：误判成功只是少了保护信息；**误判失败会真的关掉保护**。
    所以这里用源码级断言把这条锁死。
    """
    import inspect
    src = inspect.getsource(KA._set_state)
    assert "if prev" not in src, "又出现了把返回值当布尔判定的写法（0 会被误判为失败）"
    assert "if not prev" not in src
    # 失败只能由"抛异常"来判定
    assert "_state[\"active\"] = False" not in src


def test_failure_is_only_detected_by_exception(monkeypatch):
    """行为面复验：让系统调用"成功但返回 0"，active 必须仍为 True。"""
    monkeypatch.setattr(KA, "_set_state", lambda flags: (False, 0))
    KA.acquire()
    import time
    for _ in range(50):
        if KA.is_active():
            break
        time.sleep(0.02)
    assert KA.is_active() is True, "返回 0 被误判成了失败 → 保护被静默关掉"
    assert KA.describe()["last_return"] == 0
