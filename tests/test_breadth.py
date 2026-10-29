# -*- coding: utf-8 -*-
"""市场宽度（`core.breadth`）：涨跌家数必须自报分子分母。

存在理由（一句话）：首页那一格原来是**写死的演示数**（3,124 / 1,689 + 炸板率 21%），
挂着 `data-demo` 标注但没人读 —— 这是最坏的一种假数据。改成真实聚合后，最容易复发的
三个错误是：
1. **只给家数不给分母** ⇒ 5,123 被读成"全市场"（实际是 5,572 里的 5,123）；
2. **把 NULL 当平盘** ⇒ 停牌股让"平盘家数"虚高十倍（旧快照 186 只无报价就是这么来的）；
3. **拿上一交易日的宽度冒充今天** ⇒ 必须 `stale=True` 并如实报日期。
"""
import pytest

from core import breadth


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "test_breadth.db")
    yield


def _snap(date, rows):
    """rows = [(code, change_pct)]；写进 market_snapshot"""
    from db import store
    store.save_market_snapshot(date, [
        {"stock_code": c, "stock_name": c, "price": 10.0, "change_pct": p}
        for c, p in rows])


def test_counts_and_denominator(temp_store, monkeypatch):
    _snap("2026-10-26", [("600519", 1.2), ("000001", -0.5), ("300750", 0.0)])
    monkeypatch.setattr(breadth, "_universe_size", lambda: 4)
    monkeypatch.setattr(breadth, "_expected_date", lambda: "2026-10-26")
    b = breadth.market_breadth()
    assert b["available"] is True
    assert (b["up"], b["down"], b["flat"]) == (1, 1, 1)
    assert b["covered"] == 3 and b["universe"] == 4
    assert b["ratio"] == pytest.approx(0.75)
    assert b["source"] == "market_snapshot"
    assert b["stale"] is False


def test_null_change_not_counted_as_flat(temp_store, monkeypatch):
    """`change_pct IS NULL`（停牌）既不算涨也不算跌，**更不算平盘**"""
    from db import store
    store.save_market_snapshot("2026-10-26", [
        {"stock_code": "600519", "price": 10.0, "change_pct": 1.0},
        {"stock_code": "000001", "price": None, "change_pct": None},
        {"stock_code": "300750", "price": None, "change_pct": None},
    ])
    monkeypatch.setattr(breadth, "_universe_size", lambda: 3)
    monkeypatch.setattr(breadth, "_expected_date", lambda: "2026-10-26")
    b = breadth.market_breadth()
    assert b["covered"] == 1 and b["up"] == 1 and b["flat"] == 0


def test_flags_stale_when_older_than_expected(temp_store, monkeypatch):
    _snap("2026-10-23", [("600519", 1.0)])
    monkeypatch.setattr(breadth, "_universe_size", lambda: 1)
    monkeypatch.setattr(breadth, "_expected_date", lambda: "2026-10-26")
    b = breadth.market_breadth()
    assert b["date"] == "2026-10-23"
    assert b["stale"] is True
    assert any("2026-10-26" in x for x in b["degraded"]), "滞后原因必须写出来"


def test_high_flat_share_is_degraded(temp_store, monkeypatch):
    """平盘占比 > 35% ⇒ 大概率是"无报价被当平盘"，必须标注"""
    _snap("2026-10-26", [("600519", 0.0), ("000001", 0.0), ("300750", 1.0)])
    monkeypatch.setattr(breadth, "_universe_size", lambda: 3)
    monkeypatch.setattr(breadth, "_expected_date", lambda: "2026-10-26")
    b = breadth.market_breadth()
    assert any("平盘" in x for x in b["degraded"])


def test_no_data_returns_unavailable_not_zero(temp_store, monkeypatch):
    """没有任何截面 ⇒ `available:False`，**绝不返 0**（0 会被读成"全市场平盘"）"""
    monkeypatch.setattr(breadth, "_universe_size", lambda: 0)
    monkeypatch.setattr(breadth, "_expected_date", lambda: "2026-10-26")
    b = breadth.market_breadth()
    assert b["available"] is False
    assert "up" not in b and b["reason"]


def test_prefers_higher_coverage(temp_store, monkeypatch):
    """两个源都有 ⇒ 取覆盖更多的那个（同一天 3 只 vs 2 只）"""
    _snap("2026-10-26", [("600519", 1.0), ("000001", -1.0), ("300750", 0.5)])
    from db import store
    store.save_fund_flow_snapshot("2026-10-26", [
        {"stock_code": "600519", "change_pct": 1.0, "net_today": 1.0},
        {"stock_code": "000001", "change_pct": -1.0, "net_today": 1.0},
    ])
    monkeypatch.setattr(breadth, "_universe_size", lambda: 3)
    monkeypatch.setattr(breadth, "_expected_date", lambda: "2026-10-26")
    b = breadth.market_breadth()
    assert b["source"] == "market_snapshot" and b["covered"] == 3
