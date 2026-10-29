# -*- coding: utf-8 -*-
"""离线闸门 `data/offline_gate.py`（D4，2026-10-18）。

锁住的是一条**能力声明**而不是一个函数：

    「`BIGA_OFFLINE=1` ⇒ 一个外呼都不发」

原来的形态是约定（每个消费方自己写 `if bars.offline_mode()`），实测只有 4 处
K 线兜底遵守，打分仍会打网络（腾讯 K 线、tushare moneyflow）⇒ 单只 3.7s。
所以这里断言的对象是 **`gate.blocked() == 0`**（零外呼），而不是"没报错"。
"""

import pytest

import data.offline_gate as gate


@pytest.fixture(autouse=True)
def _clean_gate(monkeypatch):
    """每个用例前后都还原，避免把拦截器泄漏到别的测试里。"""
    monkeypatch.delenv("BIGA_OFFLINE", raising=False)
    gate.uninstall()
    gate.reset()
    yield
    gate.uninstall()
    gate.reset()


# ================================================================
# 1. 开关语义
# ================================================================

class TestSwitch:
    @pytest.mark.parametrize("val,expect", [
        ("1", True), ("true", True), ("YES", True), ("on", True),
        ("", False), ("0", False), ("false", False),
    ])
    def test_env_values(self, monkeypatch, val, expect):
        monkeypatch.setenv("BIGA_OFFLINE", val)
        assert gate.is_offline() is expect

    def test_online_does_not_install(self, monkeypatch):
        """在线时零副作用：不 patch、不 import requests。"""
        monkeypatch.delenv("BIGA_OFFLINE", raising=False)
        assert gate.ensure() is False
        assert gate.stats()["installed"] is False


# ================================================================
# 2. 拦截与留痕
# ================================================================

class TestBlocking:
    def test_requests_is_blocked_and_counted(self, monkeypatch):
        monkeypatch.setenv("BIGA_OFFLINE", "1")
        gate.install()
        import requests
        with pytest.raises(gate.OfflineViolation) as ei:
            requests.get("https://example.com/quote")
        assert "BIGA_OFFLINE" in str(ei.value)
        assert gate.blocked() == 1
        s = gate.stats()["samples"][0]
        assert "example.com" in s["url"]
        # 违规必须能定位到"谁在偷偷联网"
        assert "test_offline_gate" in s["at"], s["at"]

    def test_all_verbs_blocked(self, monkeypatch):
        """post/put/delete 与 get 同等对待（否则"只拦 GET"就是假闸门）。"""
        monkeypatch.setenv("BIGA_OFFLINE", "1")
        gate.install()
        import requests
        for verb in ("get", "post", "put", "delete"):
            with pytest.raises(gate.OfflineViolation):
                getattr(requests, verb)("https://example.com/x")
        assert gate.blocked() == 4

    def test_urlopen_is_blocked(self, monkeypatch):
        monkeypatch.setenv("BIGA_OFFLINE", "1")
        gate.install()
        import urllib.request
        with pytest.raises(gate.OfflineViolation):
            urllib.request.urlopen("http://example.com/")
        assert gate.blocked() == 1

    def test_uninstall_restores_original(self, monkeypatch):
        monkeypatch.setenv("BIGA_OFFLINE", "1")
        gate.install()
        import requests.sessions as rs
        gate.uninstall()
        # 还原后不得再是我们那个函数（不真发请求，只判身份）
        assert getattr(rs.Session.request, "__name__", "") != "_patched_request"

    def test_bare_socket_is_counted_not_blocked(self, monkeypatch):
        """⭐ R4（2026-10-18 审查）：裸 TCP（如 baostock）**拦不到，但必须计数**。

        不阻断是刻意的：Windows 上本地 IPC（multiprocessing/streamlit）也走
        AF_INET，一刀切会弄挂它们。但"闸门有没有被绕过"必须可审计 ——
        所以离线自检的判据是 `blocked()==0 and socket_attempts()==0`。
        """
        monkeypatch.setenv("BIGA_OFFLINE", "1")
        gate.install()
        import socket
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", 1), timeout=0.2)
        assert gate.socket_attempts() == 1
        assert gate.blocked() == 0, "裸 socket 不算被拦截（它没被拦）"
        assert "socket.connect" in gate.stats()["socket_samples"][0]["kind"]

    def test_install_reports_failure_when_nothing_can_be_patched(self, monkeypatch):
        """⭐ R8（2026-10-18 审查）：三处都装不上时**不许**宣称"已启用"。"""
        monkeypatch.setenv("BIGA_OFFLINE", "1")
        gate.uninstall()
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
            else __builtins__.__import__

        def _blocked_import(name, *a, **k):
            if str(name).split(".")[0] in ("requests", "urllib", "socket"):
                raise ImportError(f"blocked for test: {name}")
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", _blocked_import)
        assert gate.install() is False
        assert gate.stats()["installed"] is False


# ================================================================
# 3. 接线：调用 offline_mode() 就等于装上闸门
# ================================================================

class TestWiring:
    def test_offline_mode_arms_the_gate(self, monkeypatch):
        """⭐ 关键接线：消费方只需问 `offline_mode()`，闸门就该已就位。

        没有这条，闸门就只是"另一个可选模块"——而 D4 的教训正是"约定不可靠"。
        """
        monkeypatch.setenv("BIGA_OFFLINE", "1")
        from data import bars
        assert bars.offline_mode() is True
        assert gate.stats()["installed"] is True

    def test_screening_missing_code_never_touches_network(self, monkeypatch):
        """⭐⭐ 验收判据：离线模式下取一只本地没有的票 ⇒ **零外呼**（不是"快速失败"）。

        原实现会走"联网兜底 ⇒ 多数据源 ×3 重试"，单只几十秒；
        这条断言把"离线可用"变成可自动检查的能力。
        """
        monkeypatch.setenv("BIGA_OFFLINE", "1")
        gate.ensure()
        gate.reset()
        from core.screener import StockScreener
        s = StockScreener()
        assert s._get_kline_data("999999") is None
        assert gate.blocked() == 0, (
            "离线模式下仍有外呼尝试：" + str(gate.stats()["samples"][:3]))
