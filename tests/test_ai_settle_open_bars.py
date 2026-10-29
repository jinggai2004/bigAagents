# -*- coding: utf-8 -*-
"""T+1 开盘结算的取价与"零写入"（2026-10-16 修复的回归守护）。

背景：`settle_daily` 原用 `daily_bars(code, 当日)` 取开盘价，而 **09:32 时当日日线尚未生成**
⇒ 每笔都因"无当日行情"被静默跳过，日志只报"提交 0 笔"（无鉴别力）
⇒ 联赛自 10-15 起所有决策从未成交。

修法：取价改用 `open_bars`（**实时行情优先**、日线仅兜底），并**显式记录**跳过原因。
⚠️ 实时行情的三条校验缺一不可 —— 宁可"当日不成交"，也不能用错价成交（那是**静默错账**）。
"""
import pytest

from core import ai_league as L
from db import store
from tests.conftest import use_template_db

DAY = "2026-10-16"
PREV = "2026-10-15"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    use_template_db(tmp_path, monkeypatch, "ai_settle.db")
    from core import ai_personas
    monkeypatch.setattr(ai_personas, "PERSONAS_PATH", tmp_path / "ai_personas.json")
    # 默认挡掉实时源（用例可用 `_patch_quotes` 覆盖）—— 保证整套测试零联网
    monkeypatch.setattr("data.realtime.get_realtime_quotes", lambda codes, *a, **k: {})
    ai_personas.load_personas()
    yield
    ai_personas.invalidate()


def _quote(open_px=10.0, prev=9.8, price=10.2, date=DAY, **kw):
    row = {"code": "600028", "name": "测试标的", "open": open_px, "price": price,
           "prev_close": prev, "quote_date": date, "stale": False, "data_lagging": False}
    row.update(kw)
    return row


def _patch_quotes(monkeypatch, rows):
    """把 `data.realtime.get_realtime_quotes` 换成固定返回（`open_bars` 内部按需导入）。"""
    monkeypatch.setattr(
        "data.realtime.get_realtime_quotes",
        lambda codes, *a, **k: {c: rows[c] for c in codes if c in rows})


def _patch_daily_bars(monkeypatch, bars):
    monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {
        c: bars[c] for c in codes if c in bars})


class TestOpenBars:
    """`open_bars` = 开盘撮合价的**唯一实现**（提交价与撮合 bar 共用）。"""

    def test_uses_realtime_open(self, monkeypatch):
        _patch_quotes(monkeypatch, {"600028": _quote()})
        bars = L.open_bars(["600028"], DAY)
        assert bars["600028"]["open"] == 10.0
        assert bars["600028"]["prev_close"] == 9.8, "prev_close 参与涨跌停判定，必须如实传递"

    def test_rejects_stale_quote(self, monkeypatch):
        """⚠️ 过期缓存不是开盘价 —— 用它成交就是错账。"""
        _patch_quotes(monkeypatch, {"600028": _quote(stale=True)})
        _patch_daily_bars(monkeypatch, {})
        assert L.open_bars(["600028"], DAY) == {}

    def test_rejects_data_lagging(self, monkeypatch):
        _patch_quotes(monkeypatch, {"600028": _quote(data_lagging=True)})
        _patch_daily_bars(monkeypatch, {})
        assert L.open_bars(["600028"], DAY) == {}

    def test_rejects_wrong_quote_date(self, monkeypatch):
        """⭐ 前一日快照当今日开盘价 ⇒ 拒绝（宁可不成交，也要显式失败）。"""
        _patch_quotes(monkeypatch, {"600028": _quote(date=PREV)})
        _patch_daily_bars(monkeypatch, {})
        assert L.open_bars(["600028"], DAY) == {}

    def test_rejects_missing_quote_date(self, monkeypatch):
        """缺日期字段同样拒绝：无法证明它是**今天**的价。"""
        _patch_quotes(monkeypatch, {"600028": _quote(date="")})
        _patch_daily_bars(monkeypatch, {})
        assert L.open_bars(["600028"], DAY) == {}

    def test_falls_back_to_daily_bars(self, monkeypatch):
        """实时拿不到 ⇒ 日线兜底（盘后补跑 / 手动重算场景）。"""
        _patch_quotes(monkeypatch, {})
        _patch_daily_bars(monkeypatch, {"600028": {"open": 11.0, "close": 11.5,
                                                  "prev_close": 10.0}})
        assert L.open_bars(["600028"], DAY)["600028"]["open"] == 11.0

    def test_realtime_wins_over_daily(self, monkeypatch):
        """两个来源都在时**实时优先**（09:32 的日线可能是前一日残留）。"""
        _patch_quotes(monkeypatch, {"600028": _quote(open_px=10.0)})
        _patch_daily_bars(monkeypatch, {"600028": {"open": 99.0, "close": 99.0,
                                                  "prev_close": 98.0}})
        assert L.open_bars(["600028"], DAY)["600028"]["open"] == 10.0

    def test_realtime_source_is_actually_consulted(self, monkeypatch):
        """⭐ 防"注入遮住默认实现"：必须**真的**去问实时行情。

        若哪天有人把实时那一段删掉、只留日线兜底，这条会红 ——
        而 09:32 恰恰是"当日日线尚未生成"的时刻，删掉它等于把事故原样恢复。
        """
        calls: list[list] = []

        def _spy(codes, *a, **k):
            calls.append(list(codes))
            return {"600028": _quote()}

        monkeypatch.setattr("data.realtime.get_realtime_quotes", _spy)
        _patch_daily_bars(monkeypatch, {})                 # 日线故意为空
        bars = L.open_bars(["600028"], DAY)
        assert calls == [["600028"]], "必须把代码清单交给实时源"
        assert bars["600028"]["open"] == 10.0


class TestSettleDaily:
    def _seed_pending(self, persona="ai_random_1", code="600028", size=0.1):
        store.save_ai_decision({"persona": persona, "signal_date": PREV, "code": code,
                                "advice_status": "可执行", "direction": "看多",
                                "position_size": size, "order_reason": "测试"})

    def test_submits_and_fills_with_realtime_open(self, monkeypatch):
        """⭐ 修复核心：仅靠实时行情也能提交并**成交**（原实现此处为 0 笔）。"""
        self._seed_pending()
        _patch_quotes(monkeypatch, {"600028": _quote()})
        out = L.settle_daily(DAY, dry_run=True)
        row = {r["persona"]: r for r in out["results"]}["ai_random_1"]
        assert row["submitted"] == 1
        assert row["settled"] == 1
        assert row["orders"][0]["status"] == "filled", row["orders"]

    def test_missing_price_records_explicit_reason(self, monkeypatch):
        """⭐ 取不到价必须**显式留因** —— 不能又是无鉴别力的"提交 0 笔"。"""
        self._seed_pending()
        _patch_quotes(monkeypatch, {})
        _patch_daily_bars(monkeypatch, {})
        out = L.settle_daily(DAY, dry_run=True)
        row = {r["persona"]: r for r in out["results"]}["ai_random_1"]
        assert row["submitted"] == 0
        assert row["items"] and "取不到" in row["items"][0]["skipped"], row["items"]

    def test_dry_run_writes_nothing(self, monkeypatch):
        """⭐ 演练零写入 —— 原实现漏了 order_id 回填这一步 ⇒ 会污染真实库。"""
        self._seed_pending()
        _patch_quotes(monkeypatch, {"600028": _quote()})
        L.settle_daily(DAY, dry_run=True)
        got = store.list_ai_decisions(persona="ai_random_1", signal_date=PREV)
        assert got and got[0]["order_id"] is None, "演练不得回填 order_id（幂等锚点）"

    def test_apply_backfills_order_id(self, monkeypatch):
        """对照组：真跑必须回填（否则幂等锚点失效 ⇒ 次日重复成交）。"""
        self._seed_pending()
        _patch_quotes(monkeypatch, {"600028": _quote()})
        L.settle_daily(DAY, dry_run=False)
        got = store.list_ai_decisions(persona="ai_random_1", signal_date=PREV)
        assert got and got[0]["order_id"] is not None


class TestPendingWindow:
    """⭐ 结算窗口必须**严格等于前一交易日**（2026-10-16 修订）。

    原条件 `signal_date < trade_date` 会把积压多日的决策一并捞出来，
    而撮合用的是**当日**开盘价 ⇒ 本该 T+1 成交的单被按 T+N 的价成交（错价且无痕）。
    """

    def _decide(self, signal_date, code="600028", size=0.1):
        store.save_ai_decision({"persona": "ai_random_1", "signal_date": signal_date,
                                "code": code, "advice_status": "可执行",
                                "direction": "看多", "position_size": size,
                                "order_reason": "测试"})

    def test_only_previous_trading_day_is_pending(self, monkeypatch):
        monkeypatch.setattr(L, "_prev_trading_day", lambda d: PREV)
        self._decide(PREV)                               # 前一交易日 ⇒ 待结算
        self._decide("2026-10-12", code="600519")        # 积压 ⇒ 不得待结算
        got = L._pending_buys(DAY, "ai_random_1")
        assert [r["code"] for r in got] == ["600028"]

    def test_stale_buy_is_expired_not_settled(self, monkeypatch):
        """⭐ 积压决策**不得被成交**（不能拿今天的价替它成交），必须显式作废。"""
        monkeypatch.setattr(L, "_prev_trading_day", lambda d: PREV)
        _patch_quotes(monkeypatch, {"600519": _quote(open_px=1200.0)})
        self._decide("2026-10-12", code="600519")
        out = L.settle_daily(DAY, dry_run=True)
        row = {r["persona"]: r for r in out["results"]}["ai_random_1"]
        assert row["submitted"] == 0, "积压单不得被成交"
        assert any("错过 T+1 窗口" in (i.get("skipped") or "") for i in row["items"]), row["items"]

    def test_expire_marks_decision_row(self, monkeypatch):
        """真跑时要把"作废"写回决策行 —— 用户必须看得见"这单没成交"。"""
        monkeypatch.setattr(L, "_prev_trading_day", lambda d: PREV)
        self._decide("2026-10-12", code="600519")
        L.settle_daily(DAY, dry_run=False)
        got = store.list_ai_decisions(persona="ai_random_1", signal_date="2026-10-12")
        assert got and "作废" in (got[0]["order_status"] or "")

    def test_unavailable_calendar_settles_nothing(self, monkeypatch):
        """日历不可用 ⇒ **什么都不结算**（宁可当日不成交，也不用错日期的价成交）。"""
        monkeypatch.setattr(L, "_prev_trading_day", lambda d: None)
        self._decide(PREV)
        out = L.settle_daily(DAY, dry_run=True)
        row = {r["persona"]: r for r in out["results"]}["ai_random_1"]
        assert row["submitted"] == 0 and row["prev_trading_day"] is None


class TestSellSettlement:
    """⭐ 卖出决策也必须被结算（2026-10-16 补齐"买卖不对称"）。

    原先**全部**结算筛选都是 `advice_status == "可执行"`（= 买入），
    而 `persist_round` 把 `action="sell"` 写成 `advice_status="卖出"`
    ⇒ **盘后决策里的卖出永远不成交**（`settle_intraday_wakes` 只管唤醒轮）。
    """

    def _seed_holding(self, persona="ai_random_1", code="600887", qty=1600,
                      price=28.14, buy_date="2026-10-14"):
        from core.paper_account import PaperAccount
        acct = PaperAccount(1_000_000, name=persona)
        acct.apply_trade({"date": buy_date, "code": code, "direction": "买入",
                          "price": price, "quantity": qty})
        store.save_paper_state(persona, acct.to_state())

    def _seed_sell(self, signal_date=PREV, code="600887", persona="ai_random_1"):
        store.save_ai_decision({"persona": persona, "signal_date": signal_date,
                                "code": code, "advice_status": "卖出",
                                "direction": "看空", "position_size": 0.0,
                                "order_reason": "测试卖出"})

    def test_sell_decision_is_settled_and_filled(self, monkeypatch):
        """⭐ 核心：卖出决策会被提交并**成交**（原实现完全不结算卖出）。"""
        self._seed_holding()
        self._seed_sell()
        _patch_quotes(monkeypatch, {"600887": _quote(open_px=28.0, prev=28.14, price=28.05)})
        out = L.settle_daily(DAY, dry_run=True)
        row = {r["persona"]: r for r in out["results"]}["ai_random_1"]
        assert row["submitted"] == 1, row["items"]
        sells = [o for o in row["orders"] if o["direction"] == "卖出"]
        assert sells and sells[0]["status"] == "filled", row["orders"]

    def test_sell_of_same_day_holding_is_blocked_by_t1(self, monkeypatch):
        """⚠️ 买入日 = 结算日（T+1 未解锁）⇒ 必须跳过，不能凭空把仓位卖掉。"""
        self._seed_holding(buy_date=DAY)                  # 当日买入，尚不可卖
        self._seed_sell()
        _patch_quotes(monkeypatch, {"600887": _quote()})
        out = L.settle_daily(DAY, dry_run=True)
        row = {r["persona"]: r for r in out["results"]}["ai_random_1"]
        assert row["submitted"] == 0, "T+1 未解锁的持仓不得被卖出"
        assert any("无可卖持仓" in (i.get("skipped") or "") for i in row["items"]), row["items"]

    def test_sell_without_holding_is_skipped_not_faked(self, monkeypatch):
        """无持仓 ⇒ 显式跳过（不伪造一笔成交）。"""
        self._seed_sell()
        _patch_quotes(monkeypatch, {"600887": _quote()})
        out = L.settle_daily(DAY, dry_run=True)
        row = {r["persona"]: r for r in out["results"]}["ai_random_1"]
        assert row["submitted"] == 0
        assert any("无可卖持仓" in (i.get("skipped") or "") for i in row["items"])

    def test_stale_sell_is_expired(self, monkeypatch):
        """积压的卖出决策同样**作废**（与买入同口径，不拿今天的价替它成交）。"""
        self._seed_holding()
        self._seed_sell(signal_date="2026-10-12")
        _patch_quotes(monkeypatch, {"600887": _quote()})
        out = L.settle_daily(DAY, dry_run=True)
        row = {r["persona"]: r for r in out["results"]}["ai_random_1"]
        assert row["submitted"] == 0
        assert any("错过 T+1 窗口" in (i.get("skipped") or "") for i in row["items"]), row["items"]

    def test_sell_without_price_is_not_submitted(self, monkeypatch):
        """⭐⭐ 缺价时**不得提交**卖单 —— 与买入**对称**（2026-10-16 真机演练发现）。

        原先卖出不问价格就 `submit_order` ⇒ 缺价时生成一张 `@0.0 pending` 的挂单，
        而 `order_id` 已被回填（幂等锚点用掉）⇒ 不会再重试 ⇒ 那张挂单只能等**下一个**
        交易日被 `settle_open` 撮合 ⇒ 「今天该成交」变成「明天开盘成交」，**口径错位且无痕**。
        """
        self._seed_holding()
        self._seed_sell()
        _patch_quotes(monkeypatch, {})          # 实时行情取不到
        _patch_daily_bars(monkeypatch, {})      # 日线也取不到（如早盘 09:32）
        out = L.settle_daily(DAY, dry_run=True)
        row = {r["persona"]: r for r in out["results"]}["ai_random_1"]
        assert row["submitted"] == 0, f"缺价却提交了卖单：{row['items']}"
        assert not [o for o in row["orders"] if o["direction"] == "卖出"], row["orders"]
        assert any("取不到" in (i.get("skipped") or "") and i.get("direction") == "卖出"
                   for i in row["items"]), row["items"]

    def test_buy_without_price_is_not_submitted(self, monkeypatch):
        """对照组：买入早就有该检查（保证两条路径**对称**，不再一边有一边没有）。"""
        store.save_ai_decision({"persona": "ai_random_1", "signal_date": PREV,
                                "code": "600900", "advice_status": "可执行",
                                "direction": "看多", "position_size": 0.17,
                                "order_reason": "测试买入"})
        _patch_quotes(monkeypatch, {})
        _patch_daily_bars(monkeypatch, {})
        out = L.settle_daily(DAY, dry_run=True)
        row = {r["persona"]: r for r in out["results"]}["ai_random_1"]
        assert row["submitted"] == 0
        assert any("取不到" in (i.get("skipped") or "") and i.get("direction") == "买入"
                   for i in row["items"]), row["items"]

    def test_buy_and_sell_same_day_both_submitted(self, monkeypatch):
        """买卖同一天都该被结算（卖出先于买入提交 —— 回笼现金后才能买）。"""
        self._seed_holding()
        self._seed_sell()
        store.save_ai_decision({"persona": "ai_random_1", "signal_date": PREV,
                                "code": "600028", "advice_status": "可执行",
                                "direction": "看多", "position_size": 0.1,
                                "order_reason": "测试买入"})
        _patch_quotes(monkeypatch, {"600887": _quote(open_px=28.0, prev=28.14),
                                    "600028": _quote(open_px=5.0, prev=4.95, price=5.1)})
        out = L.settle_daily(DAY, dry_run=True)
        row = {r["persona"]: r for r in out["results"]}["ai_random_1"]
        assert row["submitted"] == 2, row["items"]
        dirs = [o["direction"] for o in row["orders"]]
        assert dirs == ["卖出", "买入"], f"卖出必须先提交（回笼现金）：{dirs}"
