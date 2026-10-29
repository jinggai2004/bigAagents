# -*- coding: utf-8 -*-
"""持仓取价的**批量注入**（2026-10-15 性能修复的语义守护）。

## 背景（实测）

`PortfolioManager` 的 `load()` / `check_stop_loss()` / `check_trailing_stops()` /
`get_exposure()` **各自都会逐只联网取价**，而 `_get_current_price()` 走的是
`StockDataFetcher.get_stock_info()` → akshare 东财接口：取数受限时要**重试 3 次 + 指数退避**
（实测单只 ≈30s 才失败）才回退腾讯。持仓页一次渲染实测触发 **≈9 次取价** ⇒
**1 只持仓 63 秒**，且**随持仓数线性放大**。调度器的 `run_risk_job` 同款。

而调用方（持仓页 / 调度）**本来就已经批量取过行情**（`realtime.get_realtime_quotes`，
腾讯直连、批量、快）⇒ 于是引入 `prices` 透传。

## 本文件锁住的四条语义

1. **传了 `prices` ⇒ 一次网络都不发**（用"让 `_get_current_price` 抛异常"当探针）；
2. **不传 ⇒ 完全保持原行为**（回落自行取价）—— 这条不能少：既有调用方（AI 工具链等）
   依赖默认行为，改了会静默改变它们的取价口径；
3. **`active_codes()` 不取价**（它存在的唯一理由就是解"先有代码才能批量取价"的鸡生蛋）；
4. **价格字典的两种形状都吃**（`{code: float}` 与页面的 `{code: {"price": float}}`），
   两个都不匹配时**回落取价**而不是当成 0（"不可评"≠"0 分"）。
"""
import json

import pytest
from unittest.mock import MagicMock

from tests.conftest import use_template_db

DAY = "2026-10-13"


@pytest.fixture
def pm(tmp_path, monkeypatch):
    """隔离库。把 `_get_current_price` 换成**会抛异常的探针** —— 一旦被调用即测试失败。"""
    use_template_db(tmp_path, monkeypatch, "price_inject.db")
    from core.portfolio import PortfolioManager
    m = PortfolioManager()
    m._stock_fetcher = MagicMock()
    return m


def _forbid_network(m, monkeypatch):
    """把"自己取价"这条路封死，用来证明传入的 prices 真的被用上了。"""
    def _boom(code):
        raise AssertionError(f"不该联网取价，但为 {code} 调用了 _get_current_price")

    monkeypatch.setattr(m, "_get_current_price", _boom)


def _open(m, code="600028", price=10.0, qty=1000, stop=None, take=None):
    m.open_position(code=code, price=price, quantity=qty,
                    stop_loss=stop, take_profit=take)


class TestPriceInjection:
    def test_load_uses_injected_prices_without_network(self, pm, monkeypatch):
        _open(pm)
        _forbid_network(pm, monkeypatch)
        pf = pm.load(prices={"600028": 12.0})
        assert pf.total_value > pf.cash, "总资产应含按传入价估的持仓市值"
        assert pf.total_value == pytest.approx(pf.cash + 12.0 * 1000, rel=1e-6)

    def test_check_stop_loss_uses_injected_price(self, pm, monkeypatch):
        """⭐ 传入价跌破止损 ⇒ 必须触发（且全程未联网）。"""
        _open(pm, stop=10.5)
        _forbid_network(pm, monkeypatch)
        alerts = pm.check_stop_loss(prices={"600028": 10.0})
        assert alerts and alerts[0]["type"] == "止损", alerts
        assert alerts[0]["current_price"] == 10.0

    def test_check_stop_loss_uses_injected_price_for_take_profit(self, pm, monkeypatch):
        _open(pm, take=12.0)
        _forbid_network(pm, monkeypatch)
        alerts = pm.check_stop_loss(prices={"600028": 12.5})
        assert alerts and alerts[0]["type"] == "止盈", alerts

    def test_get_exposure_uses_injected_prices(self, pm, monkeypatch):
        _open(pm)
        _forbid_network(pm, monkeypatch)
        exposure = pm.get_exposure(prices={"600028": 12.0})
        assert exposure and sum(exposure.values()) > 0

    def test_dict_shaped_quotes_are_accepted(self, pm, monkeypatch):
        """页面的行情字典（`{code: {"price": ...}}`）应能直接透传。"""
        _open(pm, stop=10.5)
        _forbid_network(pm, monkeypatch)
        alerts = pm.check_stop_loss(prices={"600028": {"price": 10.0, "name": "中国石化"}})
        assert alerts and alerts[0]["current_price"] == 10.0

    def test_missing_code_falls_back_not_zero(self, pm, monkeypatch):
        """⭐ 传入的 prices 里**没有**该 code ⇒ 回落取价，**不能当成 0**
        （0 会被读成"价格是 0"，进而误判成"跌破止损"）。"""
        _open(pm, stop=10.5)
        calls = []

        def _stub(code):
            calls.append(code)
            return 30.0

        monkeypatch.setattr(pm, "_get_current_price", _stub)
        alerts = pm.check_stop_loss(prices={"其他票": 1.0})     # 缺 600028
        # ⚠️ 断言"所有回落都指向 600028"而不是"恰好一次"：`get_active_positions()` 内部
        #    也会调 `load()` → 同样会为缺价的 code 回落一次（同一只票被问两次是正常的）。
        assert calls and set(calls) == {"600028"}, "缺价时必须回落到自行取价"
        assert alerts == [], "取到 30.0 不应触发止损（若当成 0 就会误报）"

    def test_default_none_keeps_old_behaviour(self, pm, monkeypatch):
        """⭐ 不传 prices ⇒ 回落自行取价（既有调用方的语义不能被悄悄改掉）。"""
        _open(pm, stop=10.5)
        calls = []

        def _stub(code):
            calls.append(code)
            return 10.0

        monkeypatch.setattr(pm, "_get_current_price", _stub)
        alerts = pm.check_stop_loss()
        assert calls and alerts and alerts[0]["type"] == "止损"


class TestActiveCodes:
    def test_active_codes_does_not_fetch_prices(self, pm, monkeypatch):
        """⭐ `active_codes()` 必须是纯 DB 查询：它存在的理由就是**不取价**。"""
        _open(pm, code="600028")
        _open(pm, code="600519", price=1200.0, qty=100)
        _forbid_network(pm, monkeypatch)
        assert pm.active_codes() == ["600028", "600519"]

    def test_active_codes_dedups_and_skips_closed(self, pm, monkeypatch):
        _open(pm, code="600028")
        _forbid_network(pm, monkeypatch)
        codes = pm.active_codes()
        assert len(codes) == len(set(codes))

    def test_active_codes_empty_when_no_positions(self, pm):
        assert pm.active_codes() == []


class TestWatchPageUsesBatchQuotes:
    """持仓页：**一次渲染只批量取一次行情**（其余调用全部透传）。"""

    def test_page_fetches_quotes_once_per_render(self, tmp_path, monkeypatch):
        import os
        import importlib

        from streamlit.testing.v1 import AppTest

        use_template_db(tmp_path, monkeypatch, "watch_page.db")
        monkeypatch.setenv("BIGA_UI_OFFLINE", "1")

        import streamlit as st
        st.cache_data.clear()

        from core import portfolio
        from data import realtime

        pm = portfolio.PortfolioManager()
        pm._stock_fetcher = MagicMock()
        pm.open_position(code="600028", price=10.0, quantity=1000, stop_loss=10.5)
        monkeypatch.setattr(portfolio, "get_portfolio_manager", lambda: pm)
        monkeypatch.setattr(portfolio, "PortfolioManager", lambda *a, **k: pm)

        calls = {"batch": 0}

        def _batch(codes, force_refresh=False):
            calls["batch"] += 1
            return {c: {"code": c, "name": "中国石化", "price": 12.0} for c in codes}

        monkeypatch.setattr(realtime, "get_realtime_quotes", _batch)

        # 任何"自己取价"都必须失败 —— 否则说明透传没生效、又在偷偷联网
        def _boom(code):
            raise AssertionError(f"页面仍在自行取价：{code}")

        monkeypatch.setattr(pm, "_get_current_price", _boom)

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        at = AppTest.from_file(os.path.join(root, "ui", "pages", "6_💼_持仓.py"),
                               default_timeout=120)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert calls["batch"] == 1, f"批量取价应恰好 1 次（实际 {calls['batch']} 次）"
        st.cache_data.clear()
