# -*- coding: utf-8 -*-
"""收盘价确认持仓（`core/ai_league.settle_at_close`）的测试。

⭐ 核心验收（都是"踩过才知道要有"的守护）：
  ① **按当日收盘价成交**、且**允许当日决策当日确认**（与 T+1 开盘口径区分）；
  ② **幂等**：回填 order_id + 订单置 filled ⇒ 次日 `settle_daily` **不会重复撮合**同一笔；
  ③ **`dry_run` 零写入**（回归：首版演练把 order_id 写进了库、账户却没保存，数据自相矛盾）；
  ④ **不伪造**：取不到当日收盘价 ⇒ 跳过，绝不用开盘价/成本价充数。

⚠️ 全离线：`daily_bars` 被 mock；库走 use_template_db。
"""
import pytest

from core import ai_league as L, ai_personas
from db import store
from tests.conftest import use_template_db


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    use_template_db(tmp_path, monkeypatch, "ai_close.db")
    monkeypatch.setattr(ai_personas, "PERSONAS_PATH", tmp_path / "ai_personas.json")
    ai_personas.load_personas()
    yield
    ai_personas.invalidate()


# 收盘价 5.65；开盘价刻意给一个明显不同的值（4.00）——用来证明**用的是收盘价**。
BARS = {"600028": {"open": 4.00, "close": 5.65, "prev_close": 5.50}}


def _persona(name="ai_value"):
    return ai_personas.get_persona(name)


def _seed_buy(persona="ai_value", date="2026-10-14", code="600028", size=0.2):
    store.save_ai_decision({"persona": persona, "signal_date": date, "code": code,
                            "advice_status": "可执行", "direction": "看多",
                            "position_size": size, "model": "qwen-max",
                            "order_reason": "测试买入"})


class TestSettleAtClose:
    def test_fills_same_day_decision_with_close_price(self, monkeypatch):
        """⭐ 当日决策也能确认（`signal_date == trade_date`），且成交价取自**收盘价**。"""
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {
            c: BARS[c] for c in codes if c in BARS})
        _seed_buy()
        out = L.settle_at_close("2026-10-14", personas=[_persona()])
        r = out["results"][0]
        assert out["mode"] == "close_confirm"
        assert r["filled"] == 1, r
        # 成交价应≈收盘价（含滑点/取整），但**绝不能**等于开盘价 4.00
        fp = r["items"][0]["filled_price"]
        assert abs(fp - 5.65) < 0.1, fp
        assert abs(fp - 4.00) > 1.0, "用了开盘价 ⇒ 口径写错"
        # 持仓已落库、订单号已回填
        assert store.load_paper_state("ai_value")["positions"]["600028"]["quantity"] > 0
        rows = store.list_ai_decisions(persona="ai_value", signal_date="2026-10-14")
        assert rows[0]["order_id"], "order_id 必须回填（幂等锚点）"

    def test_idempotent_second_run_fills_nothing(self, monkeypatch):
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {
            c: BARS[c] for c in codes if c in BARS})
        _seed_buy()
        assert L.settle_at_close("2026-10-14", personas=[_persona()])["results"][0]["filled"] == 1
        assert L.settle_at_close("2026-10-14", personas=[_persona()])["results"][0]["filled"] == 0

    def test_next_day_settle_daily_does_not_double_buy(self, monkeypatch):
        """⭐ 最关键的守护：收盘价确认后，次日 T+1 开盘结算**不得**再买一次。"""
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {
            c: BARS[c] for c in codes if c in BARS})
        _seed_buy()
        L.settle_at_close("2026-10-14", personas=[_persona()])
        n_before = store.load_paper_state("ai_value")["positions"]["600028"]["quantity"]
        out = L.settle_daily("2026-10-15", personas=[_persona()])
        assert out["results"][0]["submitted"] == 0, out
        n_after = store.load_paper_state("ai_value")["positions"]["600028"]["quantity"]
        assert n_after == n_before, "重复买入！幂等锚点失效"

    def test_dry_run_writes_nothing(self, monkeypatch):
        """⭐ 回归：dry_run 必须**零写入**（订单号/账户都不得落库）。"""
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {
            c: BARS[c] for c in codes if c in BARS})
        _seed_buy()
        out = L.settle_at_close("2026-10-14", personas=[_persona()], dry_run=True)
        assert out["results"][0]["filled"] == 1          # 演练内计算照做
        rows = store.list_ai_decisions(persona="ai_value", signal_date="2026-10-14")
        assert not rows[0]["order_id"], "dry_run 泄漏：把 order_id 写进了库"
        st = store.load_paper_state("ai_value")
        assert not (st or {}).get("positions"), "dry_run 泄漏：账户被写入"

    def test_missing_close_is_skipped_not_faked(self, monkeypatch):
        """取不到当日收盘价 ⇒ 跳过并如实记录，不伪造持仓。"""
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {})
        _seed_buy()
        out = L.settle_at_close("2026-10-14", personas=[_persona()])
        r = out["results"][0]
        assert r["filled"] == 0
        assert r["items"] and r["items"][0]["status"] == "skipped"
        st = store.load_paper_state("ai_value")
        assert not (st or {}).get("positions")
        rows = store.list_ai_decisions(persona="ai_value", signal_date="2026-10-14")
        assert not rows[0]["order_id"], "跳过时不得回填 order_id"


class TestSettleAtCloseSells:
    """⭐ 加速口径同样必须结算**卖出**（与 `settle_daily` 对称，2026-10-16 补）。

    正式口径补上卖出后如果这一步不补，"盘后立刻看到持仓"与"次日真实成交"会**给出不同结果**
    （页面数字对不上）——比两边都不做更糟。
    """

    def _seed_holding(self, persona="ai_value", code="600028", qty=1000, price=5.0):
        from core.paper_account import PaperAccount
        acct = PaperAccount(1_000_000, name=persona)
        acct.apply_trade({"date": "2026-10-13", "code": code, "direction": "买入",
                          "price": price, "quantity": qty})
        store.save_paper_state(persona, acct.to_state())

    def _seed_sell(self, code="600028", date="2026-10-14"):
        store.save_ai_decision({"persona": "ai_value", "signal_date": date,
                                "code": code, "advice_status": "卖出",
                                "direction": "看空", "position_size": 0.0,
                                "order_reason": "测试卖出"})

    def test_sell_is_filled_at_close_price(self, monkeypatch):
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {
            c: BARS[c] for c in codes if c in BARS})
        self._seed_holding()
        self._seed_sell()
        out = L.settle_at_close("2026-10-15", personas=[_persona()])
        r = {x["persona"]: x for x in out["results"]}["ai_value"]
        sells = [i for i in r["items"] if i.get("direction") == "卖出"]
        assert sells and sells[0]["status"] == "filled", r["items"]
        assert abs(sells[0]["close"] - 5.65) < 0.01, "卖出也必须用**收盘价**（不是开盘价）"
        rows = store.list_ai_decisions(persona="ai_value", signal_date="2026-10-14")
        sell_row = [x for x in rows if x["advice_status"] == "卖出"][0]
        assert sell_row["order_id"], "order_id 必须回填（幂等锚点）"

    def test_dry_run_sell_writes_nothing(self, monkeypatch):
        """演练同样不得回填卖出的 order_id。"""
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {
            c: BARS[c] for c in codes if c in BARS})
        self._seed_holding()
        self._seed_sell()
        L.settle_at_close("2026-10-15", personas=[_persona()], dry_run=True)
        rows = store.list_ai_decisions(persona="ai_value", signal_date="2026-10-14")
        assert not [x for x in rows if x["advice_status"] == "卖出"][0]["order_id"], \
            "dry_run 泄漏：把卖出的 order_id 写进了库"

    def test_random_persona_also_confirmed(self, monkeypatch):
        """随机对照组与 LLM 组走同一确认口径（可比性）。"""
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {
            c: BARS[c] for c in codes if c in BARS})
        _seed_buy(persona="ai_random_1", size=0.05)
        out = L.settle_at_close("2026-10-14", personas=[_persona("ai_random_1")])
        assert out["results"][0]["filled"] == 1
