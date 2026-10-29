# -*- coding: utf-8 -*-
"""交易级盈亏分解测试（pnl_attribution）：临时库 + 注入面板价，零网络。

覆盖：
1. rebuild_decision_prices —— F6 自愈：decision_price NULL/假零 → 面板收盘重建，
   滑点按 filled−decision 重算，经 save_paper_state 回写持久化；
2. decompose_round_trips —— 恒等式数学：entry_gap / exit_gap / holding /
   counterfactual 的数值锁死；FIFO 配对与部分数量消耗；
3. degraded 语义 —— 决策价取不到的回合**不进合计**（缺就是缺，不冒充 0）。
"""
import pytest

from core import pnl_attribution as pa
from db import store as db_store


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "test_pnl.db")
    yield


def _order(order_id, code, direction, qty, submit, fill_price, fill_date,
           decision_price=None, slippage=None, status="filled"):
    return {"order_id": order_id, "code": code, "direction": direction,
            "quantity": qty, "submit_date": submit, "status": status,
            "reason": "测试", "filled_quantity": qty if status == "filled" else 0,
            "filled_price": fill_price if status == "filled" else 0.0,
            "filled_date": fill_date if status == "filled" else "",
            "limit_price": None, "decision_price": decision_price,
            "slippage": slippage}


def _fill(code, direction, date, qty, price, commission=0.0, stamp_tax=0.0):
    return {"date": date, "code": code, "direction": direction,
            "quantity": qty, "actual_price": price, "signal_price": price,
            "commission": commission, "stamp_tax": stamp_tax,
            "total": qty * price, "cash_delta": 0.0, "cash_after": 0.0}


def _seed_account(name="t1"):
    """买入决策价缺失（走面板重建=10.0）；卖出决策价显式 11 / 成交 10.9"""
    state = {"name": name, "initial_cash": 1_000_000.0, "cash": 1_000_000.0,
             "positions": {}, "nav": [],
             "orders": [
                 _order(1, "600000", "买入", 1000, "2026-10-19", 10.1, "2026-10-20"),
                 _order(2, "600000", "卖出", 1000, "2026-10-26", 10.9, "2026-10-27",
                        decision_price=11.0),
             ],
             "fills": [
                 _fill("600000", "买入", "2026-10-20", 1000, 10.1, commission=25.0),
                 _fill("600000", "卖出", "2026-10-27", 1000, 10.9, commission=25.0,
                       stamp_tax=10.0),
             ]}
    db_store.save_paper_state(name, state)


def test_rebuild_persists_decision_price_and_slippage(temp_store, monkeypatch):
    """F6 自愈：NULL 决策价 → 面板收盘 10.0；假零滑点 → filled − decision 重算回写。
    买入 NULL 走重建；卖出决策价已显式（11.0）不重建，但滑点缺失仍重算（−0.1）"""
    _seed_account()
    monkeypatch.setattr(pa, "_panel_close", lambda code, date: 10.0)
    out = pa.rebuild_decision_prices("t1")
    assert out["rebuilt_price"] == 1 and out["rebuilt_slippage"] == 2
    assert out["persisted"] is True
    state = db_store.load_paper_state("t1")
    by_id = {o["order_id"]: o for o in state["orders"]}
    assert by_id[1]["decision_price"] == pytest.approx(10.0)
    assert by_id[1]["slippage"] == pytest.approx(0.1)      # 10.1 − 10.0
    assert by_id[2]["decision_price"] == pytest.approx(11.0)   # 显式值不被覆盖
    assert by_id[2]["slippage"] == pytest.approx(-0.1)     # 10.9 − 11.0（卖便宜了）


def test_rebuild_without_price_marks_skip_not_zero(temp_store, monkeypatch):
    """面板取不到 ⇒ 保持 NULL（不冒充 0），计 skipped_no_price；显式值不受影响"""
    _seed_account()
    monkeypatch.setattr(pa, "_panel_close", lambda code, date: 0.0)
    out = pa.rebuild_decision_prices("t1")
    assert out["rebuilt_price"] == 0 and out["skipped_no_price"] == 1
    state = db_store.load_paper_state("t1")
    by_id = {o["order_id"]: o for o in state["orders"]}
    assert by_id[1]["decision_price"] is None
    assert by_id[2]["decision_price"] == pytest.approx(11.0)


def test_decompose_round_trip_math(temp_store, monkeypatch):
    """恒等式数值锁死：
    entry_gap = 10.1/10−1 = +1%；exit_gap = 1−10.9/11 = +0.9091%；
    holding = 10.9/10.1−1 = +7.9208%；counterfactual = 11/10−1 = +10%；
    fees = 25 + 25 + 10 = 60。四个量对不上的实现直接红。"""
    _seed_account()
    monkeypatch.setattr(pa, "_panel_close", lambda code, date: 10.0)
    res = pa.decompose_round_trips("t1")
    assert res["n_trips"] == 1 and res["degraded"] == 0
    t = res["trips"][0]
    assert t["entry_gap_pct"] == pytest.approx(1.0, abs=1e-3)
    assert t["exit_gap_pct"] == pytest.approx(0.9091, abs=1e-3)
    assert t["holding_pct"] == pytest.approx(7.9208, abs=1e-3)
    assert t["counterfactual_pct"] == pytest.approx(10.0, abs=1e-3)
    assert t["fees"] == pytest.approx(60.0, abs=1e-6)
    s = res["summary"]
    assert s["holding_avg_pct"] == pytest.approx(7.9208, abs=1e-3)


def test_decompose_fifo_partial_quantity(temp_store, monkeypatch):
    """买 2000 卖 1000 ⇒ FIFO 消费一半 lot，回合数量 1000；剩余 lot 不产生假回合"""
    state = {"name": "t2", "initial_cash": 1_000_000.0, "cash": 1_000_000.0,
             "positions": {}, "nav": [],
             "orders": [
                 _order(1, "600000", "买入", 2000, "2026-10-19", 10.0, "2026-10-20",
                        decision_price=10.0, slippage=0.0),
                 _order(2, "600000", "卖出", 1000, "2026-10-26", 11.0, "2026-10-27",
                        decision_price=11.0, slippage=0.0),
             ],
             "fills": [
                 _fill("600000", "买入", "2026-10-20", 2000, 10.0, commission=50.0),
                 _fill("600000", "卖出", "2026-10-27", 1000, 11.0, commission=25.0),
             ]}
    db_store.save_paper_state("t2", state)
    res = pa.decompose_round_trips("t2")
    assert res["n_trips"] == 1
    assert res["trips"][0]["qty"] == 1000
    assert res["trips"][0]["holding_pct"] == pytest.approx(10.0, abs=1e-3)
    # 买入手续费按配对数量分摊：50 × (1000/2000) = 25，加卖出 25 ⇒ 50
    assert res["trips"][0]["fees"] == pytest.approx(50.0, abs=1e-6)
    assert res["degraded"] == 0


def test_decompose_excludes_degraded_from_summary(temp_store, monkeypatch):
    """决策价取不到 ⇒ 回合 degraded 且**不进合计**（不冒充 0 参与）"""
    _seed_account()
    monkeypatch.setattr(pa, "_panel_close", lambda code, date: 0.0)
    res = pa.decompose_round_trips("t1")
    assert res["n_trips"] == 0 and res["degraded"] >= 1
    assert res["summary"]["holding_avg_pct"] is None


def test_decompose_multi_account_scoped(temp_store, monkeypatch):
    """name 参数限定账户范围，不串账"""
    _seed_account("a1")
    monkeypatch.setattr(pa, "_panel_close", lambda code, date: 10.0)
    res = pa.decompose_round_trips("a1")
    assert res["accounts"] == ["a1"] and res["n_trips"] == 1
