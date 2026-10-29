"""模拟账户持久化测试（Phase 2）

重点：
  - 状态往返（现金/持仓/订单/成交/净值）后能完整恢复；
  - **T+1 跨进程可恢复**：当日买入的持仓落库再加载后，次日仍能解冻卖出
    （早期实现用内存队列记录待解冻，落库即丢，持仓会被永久锁死）；
  - 幂等：重复保存不累积重复成交/净值；持仓表是快照语义（清仓后不留残行）。
"""

import pytest

from core.paper_account import PaperAccount


def _temp_store(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    db_path = use_template_db(tmp_path, monkeypatch, "paper.db")
    from db import store
    return store


def _build_account() -> PaperAccount:
    acct = PaperAccount(initial_cash=100_000.0, name="test")
    acct.apply_trade({"date": "2026-01-05", "code": "000001", "direction": "买入",
                      "price": 10.0, "quantity": 1000})
    acct.apply_trade({"date": "2026-01-06", "code": "000001", "direction": "卖出",
                      "price": 11.0, "quantity": 400})
    acct.mark_to_market("2026-01-05", {"000001": 10.0})
    acct.mark_to_market("2026-01-06", {"000001": 11.0})
    acct.submit_order("600519", "买入", 100, "2026-01-06")
    return acct


def _count(store, table: str, name: str) -> int:
    conn = store.get_connection()
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE name = ?", (name,)).fetchone()[0]
    finally:
        conn.close()


class TestRoundTrip:
    def test_save_and_load_restores_state(self, tmp_path, monkeypatch):
        store = _temp_store(tmp_path, monkeypatch)
        acct = _build_account()
        res = store.save_paper_state(acct.name, acct.to_state())
        assert res["saved"] is True and res["positions"] == 1

        restored = PaperAccount.from_state(store.load_paper_state(acct.name))
        assert restored.cash == pytest.approx(acct.cash, abs=0.01)
        assert set(restored.positions) == set(acct.positions)
        assert restored.position("000001").quantity == acct.position("000001").quantity
        assert restored.position("000001").avg_cost == pytest.approx(
            acct.position("000001").avg_cost, abs=1e-6)
        assert restored.nav[-1]["total_value"] == pytest.approx(
            acct.nav[-1]["total_value"], abs=0.01)
        assert len(restored.orders) == 1 and restored.orders[0].code == "600519"
        assert len(restored.fills) == len(acct.fills)

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        store = _temp_store(tmp_path, monkeypatch)
        assert store.load_paper_state("nope") is None

    def test_t1_survives_round_trip(self, tmp_path, monkeypatch):
        """关键回归：当日买入的持仓落库再加载后，次日必须能解冻卖出"""
        store = _temp_store(tmp_path, monkeypatch)
        acct = PaperAccount(initial_cash=100_000.0, name="t1")
        acct.apply_trade({"date": "2026-01-05", "code": "000001", "direction": "买入",
                          "price": 10.0, "quantity": 1000})
        assert acct.position("000001").available == 0
        store.save_paper_state("t1", acct.to_state())

        restored = PaperAccount.from_state(store.load_paper_state("t1"))
        assert restored.position("000001").available == 0, "当日仍不可卖"

        out = restored.apply_trade({"date": "2026-01-06", "code": "000001",
                                    "direction": "卖出", "price": 10.5, "quantity": 1000})
        assert out.get("status") != "rejected", "T+1 解冻不能因重启而丢失"
        assert "000001" not in restored.positions


class TestIdempotency:
    def test_double_save_does_not_duplicate(self, tmp_path, monkeypatch):
        store = _temp_store(tmp_path, monkeypatch)
        acct = _build_account()
        store.save_paper_state(acct.name, acct.to_state())
        store.save_paper_state(acct.name, acct.to_state())

        assert _count(store, "paper_fills", acct.name) == len(acct.fills)
        assert _count(store, "paper_nav", acct.name) == len(acct.nav)
        assert _count(store, "paper_orders", acct.name) == len(acct.orders)

    def test_position_snapshot_replaces_previous(self, tmp_path, monkeypatch):
        """清仓/换仓后再保存，持仓表不能残留旧行"""
        store = _temp_store(tmp_path, monkeypatch)
        acct = _build_account()
        store.save_paper_state("a", acct.to_state())

        acct.apply_trade({"date": "2026-01-07", "code": "000001", "direction": "卖出",
                          "price": 11.0, "quantity": 600})
        acct.apply_trade({"date": "2026-01-07", "code": "600519", "direction": "买入",
                          "price": 20.0, "quantity": 100})
        store.save_paper_state("a", acct.to_state())

        state = store.load_paper_state("a")
        assert set(state["positions"]) == {"600519"}

    def test_order_status_progresses_after_settle(self, tmp_path, monkeypatch):
        store = _temp_store(tmp_path, monkeypatch)
        acct = PaperAccount(100_000.0, name="b")
        acct.submit_order("000001", "买入", 1000, "2026-01-05")
        store.save_paper_state("b", acct.to_state())

        acct.settle_open("2026-01-06", {"000001": {"open": 10.0, "close": 10.0,
                                                   "prev_close": 10.0}})
        store.save_paper_state("b", acct.to_state())

        order = store.load_paper_state("b")["orders"][0]
        assert order["status"] == "filled"
        assert order["filled_quantity"] == 1000
        assert _count(store, "paper_orders", "b") == 1

    def test_limit_price_survives_round_trip(self, tmp_path, monkeypatch):
        """限价意图必须能跨进程恢复。

        ⚠️ 这是"未成交就重启"的场景：订单还 pending，若 `limit_price` 读不回来
        （变成 0 = 市价单），明天开盘就会**无条件成交**——静默地把"等价位"
        变成"追价"，而日志里看不出任何异常。
        """
        store = _temp_store(tmp_path, monkeypatch)
        acct = PaperAccount(100_000.0, name="lim")
        acct.submit_order("000001", "买入", 1000, "2026-01-05",
                          limit_price=9.80, decision_price=9.75)
        store.save_paper_state("lim", acct.to_state())

        row = store.load_paper_state("lim")["orders"][0]
        assert row["limit_price"] == pytest.approx(9.80)
        assert row["decision_price"] == pytest.approx(9.75)

        # 反序列化回对象后仍然是限价单
        restored = PaperAccount.from_state(store.load_paper_state("lim"))
        assert restored.orders[0].has_limit is True
        assert restored.orders[0].limit_price == pytest.approx(9.80)

    def test_market_order_stores_null_not_zero(self, tmp_path, monkeypatch):
        """市价单的 limit_price 落 **NULL**，不是 0.0。

        两种"无限价"同时存在时，下游判"是否限价单"要写两遍；
        漏写一遍就会把市价单当成"限价 0 元"，从而永远撤单。
        """
        store = _temp_store(tmp_path, monkeypatch)
        acct = PaperAccount(100_000.0, name="mkt")
        acct.submit_order("000001", "买入", 1000, "2026-01-05")
        store.save_paper_state("mkt", acct.to_state())

        conn = store.get_connection()
        try:
            raw = conn.execute(
                "SELECT limit_price, decision_price, slippage FROM paper_orders "
                "WHERE name = 'mkt'").fetchone()
        finally:
            conn.close()
        assert raw["limit_price"] is None and raw["decision_price"] is None

    def test_slippage_survives_round_trip(self, tmp_path, monkeypatch):
        """滑点要落库——它是判断"挂单阈值准不准"的唯一硬数据。"""
        store = _temp_store(tmp_path, monkeypatch)
        acct = PaperAccount(100_000.0, name="slp")
        acct.submit_order("000001", "买入", 1000, "2026-01-05",
                          limit_price=10.2, decision_price=10.0)
        acct.settle_open("2026-01-06", {"000001": {"open": 10.1, "close": 10.1,
                                                   "prev_close": 10.0}})
        store.save_paper_state("slp", acct.to_state())

        row = store.load_paper_state("slp")["orders"][0]
        assert row["status"] == "filled"
        assert row["slippage"] == pytest.approx(round(row["filled_price"] - 10.0, 4))
        assert row["slippage"] > 0, "开盘 10.1 买入、决策价 10.0 ⇒ 追高，滑点应为正"


class TestAccountAdmin:
    def test_list_and_delete(self, tmp_path, monkeypatch):
        store = _temp_store(tmp_path, monkeypatch)
        for name in ("a", "b"):
            acct = PaperAccount(100_000.0, name=name)
            acct.mark_to_market("2026-01-05", {})
            store.save_paper_state(name, acct.to_state())

        assert store.list_paper_accounts() == ["a", "b"]
        assert store.delete_paper_account("a") > 0
        assert store.list_paper_accounts() == ["b"]
        assert store.load_paper_state("a") is None
        assert _count(store, "paper_nav", "a") == 0

    def test_empty_state_rejected(self, tmp_path, monkeypatch):
        store = _temp_store(tmp_path, monkeypatch)
        assert store.save_paper_state("x", {})["saved"] is False

    def test_two_accounts_are_isolated(self, tmp_path, monkeypatch):
        store = _temp_store(tmp_path, monkeypatch)
        a = PaperAccount(100_000.0, name="a")
        a.apply_trade({"date": "2026-01-05", "code": "000001", "direction": "买入",
                       "price": 10.0, "quantity": 1000})
        b = PaperAccount(200_000.0, name="b")
        store.save_paper_state("a", a.to_state())
        store.save_paper_state("b", b.to_state())

        assert set(store.load_paper_state("a")["positions"]) == {"000001"}
        assert store.load_paper_state("b")["positions"] == {}
        assert store.load_paper_state("b")["initial_cash"] == 200_000.0
