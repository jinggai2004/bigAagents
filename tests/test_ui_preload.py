# -*- coding: utf-8 -*-
"""UI 进程预热（`ui/_preload.py`）的测试。

⭐ 三条必须钉住的语义：
  ① **测试环境默认关闭**（否则每个用例都会去 import LangGraph/akshare，慢且不确定）；
  ② **只起一个线程**（app.py 每次交互重跑脚本，模块级 flag + 锁是唯一可靠的去重手段）；
  ③ **导入失败必须被忽略**（预热是加速手段，不是依赖 —— 失败绝不能把页面搞崩）。
"""
import threading

import pytest

from ui import _preload as P


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    """把模块级状态复位，并清掉可能影响判定的环境变量。

    ⚠️ 不清 `_started` 的话，第一个用例起过线程后，后续用例会全部误判成"已启动"。
    """
    monkeypatch.setattr(P, "_started", False)
    for k in ("BIGA_UI_PRELOAD", "BIGA_UI_OFFLINE"):
        monkeypatch.delenv(k, raising=False)


class TestEnabled:
    def test_disabled_in_offline_test_env(self, monkeypatch):
        monkeypatch.setenv("BIGA_UI_OFFLINE", "1")
        assert P.preload_enabled() is False

    def test_env_forces_on_even_offline(self, monkeypatch):
        monkeypatch.setenv("BIGA_UI_OFFLINE", "1")
        monkeypatch.setenv("BIGA_UI_PRELOAD", "1")
        assert P.preload_enabled() is True

    def test_env_forces_off(self, monkeypatch):
        monkeypatch.setenv("BIGA_UI_PRELOAD", "0")
        assert P.preload_enabled() is False

    def test_default_on_outside_tests(self):
        assert P.preload_enabled() is True


class TestEnsurePreloaded:
    def test_starts_only_one_thread(self, monkeypatch):
        """⭐ 幂等：重复调用只起一个线程（app.py 每次 rerun 都会调它）。"""
        entered, release = [], threading.Event()

        def fake_run():
            entered.append(1)
            release.wait(3)

        monkeypatch.setattr(P, "_run", fake_run)
        monkeypatch.setenv("BIGA_UI_PRELOAD", "1")
        assert P.ensure_preloaded() is True
        assert P.ensure_preloaded() is False
        assert P.ensure_preloaded() is False
        release.set()

    def test_disabled_returns_false_without_thread(self, monkeypatch):
        monkeypatch.setenv("BIGA_UI_PRELOAD", "0")
        monkeypatch.setattr(P, "_run",
                            lambda: pytest.fail("禁用时不得启动线程"))
        assert P.ensure_preloaded() is False


class TestRunIsSafe:
    def test_import_failure_is_swallowed(self, monkeypatch):
        """⭐ 预热失败只记日志：坏一个模块不能让整个预热链断掉。"""
        monkeypatch.setattr(P, "PRELOAD_TARGETS",
                            ("definitely_not_a_real_module_12345", "os"))
        P._run()                                    # 不抛异常即通过

    def test_targets_are_unique_and_sorted_heavy_first(self):
        assert len(set(P.PRELOAD_TARGETS)) == len(P.PRELOAD_TARGETS)
        # 最贵的两条必须排在最前（否则预热窗口被便宜的模块占掉）
        assert P.PRELOAD_TARGETS[:2] == ("core.graph_builder", "core.agents")
