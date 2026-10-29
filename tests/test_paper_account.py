"""模拟交易账户测试（Phase 2）

覆盖：T+1 可卖数量、涨跌停/停牌对挂单的处理、顺延上限、下单校验、净值与缺价兜底，
以及**与回测引擎的一致性**（同信号同价格下逐项对齐——这是 Phase 2 的验收条件）。
"""

import pandas as pd
import pytest

from core import matching
from core.backtest import BacktestEngine
from core.paper_account import MAX_DEFER_DAYS, PaperAccount, Position

DATES = ["2026-01-05", "2026-01-06", "2026-01-07"]
CLOSES = {
    "000001": {"2026-01-05": 10.0, "2026-01-06": 10.5, "2026-01-07": 11.0},
    "600519": {"2026-01-05": 20.0, "2026-01-06": 19.0, "2026-01-07": 21.0},
}
TRADES = [
    {"date": "2026-01-05", "code": "000001", "direction": "买入", "price": 10.0, "quantity": 1000},
    {"date": "2026-01-06", "code": "600519", "direction": "买入", "price": 19.0, "quantity": 500},
    {"date": "2026-01-07", "code": "000001", "direction": "卖出", "price": 11.0, "quantity": 1000},
]


def _kline() -> dict:
    return {code: pd.DataFrame({"date": pd.to_datetime(list(dates)),
                                "close": list(dates.values())})
            for code, dates in CLOSES.items()}


def _prices(date: str) -> dict:
    return {code: dates[date] for code, dates in CLOSES.items()}


def _bar(open_price, prev_close):
    return {"open": open_price, "close": open_price, "prev_close": prev_close}


class TestT1Rule:
    def test_bought_today_not_sellable_today(self):
        acct = PaperAccount(100_000)
        acct.apply_trade({"date": "2026-01-05", "code": "000001", "direction": "买入",
                          "price": 10.0, "quantity": 1000})
        pos = acct.position("000001")
        assert pos.quantity == 1000
        assert pos.available == 0, "当日买入不可卖（T+1）"

        out = acct.apply_trade({"date": "2026-01-05", "code": "000001", "direction": "卖出",
                                "price": 10.5, "quantity": 1000})
        assert out["status"] == "rejected" and "T+1" in out["reason"]
        assert acct.position("000001").quantity == 1000

    def test_sellable_next_day(self):
        acct = PaperAccount(100_000)
        acct.apply_trade({"date": "2026-01-05", "code": "000001", "direction": "买入",
                          "price": 10.0, "quantity": 1000})
        out = acct.apply_trade({"date": "2026-01-06", "code": "000001", "direction": "卖出",
                                "price": 10.5, "quantity": 1000})
        assert out.get("status") != "rejected"
        assert "000001" not in acct.positions

    def test_partial_sell_keeps_avg_cost(self):
        acct = PaperAccount(100_000)
        acct.apply_trade({"date": "2026-01-05", "code": "000001", "direction": "买入",
                          "price": 10.0, "quantity": 1000})
        avg0 = acct.position("000001").avg_cost
        acct.apply_trade({"date": "2026-01-06", "code": "000001", "direction": "卖出",
                          "price": 12.0, "quantity": 400})
        pos = acct.position("000001")
        assert pos.quantity == 600
        assert pos.avg_cost == pytest.approx(avg0, abs=1e-6)


class TestT1AcrossMidnight:
    """T+1 解冻必须"跨日"生效——即要的是**日期切换**，不是"必须发生成交"。

    实例（2026-10-10 用户首次尝试卖出 002040）：持仓 `last_buy_date=2026-09-25`，
    卖出时若 `available=0` 就会被拒，提示 "T+1 未解锁或无可用持仓"。
    ⇒ 该提示必须**只在真的当日买入**时才出现（下方第一条用例钉住），
    否则用户会把"没人帮我解冻"误读成"T+1 在拦我"。
    """

    def test_available_when_sell_date_equals_last_buy_date(self):
        """同日买卖判据：last_buy_date >= 卖出日 视为当日买入 → 不可卖（保守但明确）"""
        acct = PaperAccount(100_000)
        acct.positions["002040"] = Position(
            code="002040", quantity=20000, available=0,
            total_cost=181736.22, last_buy_date="2026-10-10")
        out = acct.apply_trade({"date": "2026-10-10", "code": "002040",
                                "direction": "卖出", "price": 9.5, "quantity": 20000})
        assert out["status"] == "rejected" and "T+1" in out["reason"]

    def test_unlocked_on_any_later_date(self):
        """跨到任何一个更晚的日期都必须解冻（含周末/节假日的"自然日"字符串）"""
        acct = PaperAccount(100_000)
        acct.positions["002040"] = Position(
            code="002040", quantity=20000, available=0,
            total_cost=181736.22, last_buy_date="2026-10-10")
        out = acct.apply_trade({"date": "2026-10-13", "code": "002040",
                                "direction": "卖出", "price": 9.5, "quantity": 20000})
        assert out.get("status") != "rejected", "跨日后应解冻可卖"
        assert "002040" not in acct.positions

    def test_state_reload_keeps_last_buy_date(self):
        """落库→重载后 last_buy_date 必须保留，否则 T+1 判定失去依据"""
        acct = PaperAccount(100_000)
        acct.positions["002040"] = Position(
            code="002040", quantity=20000, available=0,
            total_cost=181736.22, last_buy_date="2026-10-10")
        reloaded = PaperAccount.from_state(acct.to_state())
        pos = reloaded.position("002040")
        assert pos.last_buy_date == "2026-10-10"
        assert pos.available == 0, "重载后当日买入仍不可卖"


class TestUnlockT1PublicEntry:
    """`unlock_t1()` 这个**只读路径也能调**的公开入口（2026-10-10 新增）。

    ⚠️ 这类测试的价值在于**鉴别力**：如果只断言"调完之后可卖"，那 `_unlock_t1`
    时代也测不出差别（因为那时也能通过 `apply_trade` 触发）。所以这里刻意
    **完全不做任何成交调用**，只调 `unlock_t1`，把"解冻是否需要一笔成交"
    这个差异直接钉住——旧的实现下这些用例必红（AttributeError: unlock_t1）。
    """

    def _acct_with_locked_position(self) -> PaperAccount:
        """构造"跨日但未解冻"的账户——模拟周五买入、周末打开的现场"""
        acct = PaperAccount(100_000)
        acct.positions["002040"] = Position(
            code="002040", quantity=20000, available=0,
            total_cost=181736.22, last_buy_date="2026-10-09")
        return acct

    def test_unlock_without_any_trade_call(self):
        """核心：**不发生任何成交**，仅调 unlock_t1 就能解冻"""
        acct = self._acct_with_locked_position()
        assert acct.position("002040").available == 0, "前置：解冻前不可卖"

        unlocked = acct.unlock_t1("2026-10-12")     # 下一个交易日（周一）

        assert unlocked == 1, "应解冻 1 只"
        assert acct.position("002040").available == 20000, "解冻后全部可卖"

    def test_unlock_uses_natural_day_not_trading_day(self):
        """⚠️ 口径说明：解冻判据是**自然日**（`last_buy_date < date`），不含交易日历。

        实证（2026-10-09 周五买入）：
            unlock_t1("2026-10-10") 周六 → **返回 1（解冻）**
            unlock_t1("2026-10-12") 周一 → 返回 0（已解冻）
        严格 A 股 T+1 应等**下一个交易日**（10-12），自然日口径把这提前了。

        为什么保留自然日：① 它是**放宽**（更早可卖）不是收紧，不会把用户仓位锁死；
        ② 加日历会把 `core.paper_account` 与 `data.trading_calendar` 耦合，
        而解冻只影响**模拟盘可卖性**、不涉真实资金。若将来要严格化，
        应在此处引入 `is_trading_day`，并同步 `_unlock_t1`。
        """
        acct = self._acct_with_locked_position()
        assert acct.unlock_t1("2026-10-10") == 1, "自然日口径：周六即算跨日"
        assert acct.unlock_t1("2026-10-12") == 0, "周一已解冻，不重复计数"

    def test_unlock_returns_count_and_is_idempotent(self):
        """返回值语义 + 幂等：第二次调不应重复计数，也不应改变结果"""
        acct = self._acct_with_locked_position()
        assert acct.unlock_t1("2026-10-12") == 1
        assert acct.unlock_t1("2026-10-12") == 0, "已解冻的不应重复计数（幂等）"
        assert acct.position("002040").available == 20000

    def test_does_not_unlock_same_day_buy(self):
        """边界：**当日买入**不能被解冻（T+1 的本意）"""
        acct = PaperAccount(100_000)
        acct.positions["002040"] = Position(
            code="002040", quantity=20000, available=0,
            total_cost=181736.22, last_buy_date="2026-10-10")
        assert acct.unlock_t1("2026-10-10") == 0, "同日不应解冻"
        assert acct.position("002040").available == 0

    def test_empty_last_buy_date_is_ignored(self):
        """缺少 last_buy_date 的持仓不由这里处理（`from_state` 已按解冻处理）"""
        acct = PaperAccount(100_000)
        acct.positions["002040"] = Position(
            code="002040", quantity=20000, available=0,
            total_cost=181736.22, last_buy_date="")
        assert acct.unlock_t1("2026-10-12") == 0

    def test_partially_available_position_is_topped_up(self):
        """部分可卖（如多次买入、部分已解冻）应被**补足**到全量"""
        acct = PaperAccount(100_000)
        acct.positions["002040"] = Position(
            code="002040", quantity=20000, available=5000,
            total_cost=181736.22, last_buy_date="2026-10-09")
        assert acct.unlock_t1("2026-10-12") == 1
        assert acct.position("002040").available == 20000

    def test_no_positions_returns_zero(self):
        """空仓不报错"""
        assert PaperAccount(100_000).unlock_t1("2026-10-12") == 0

    def test_survives_store_roundtrip(self):
        """落库→重载后再解冻仍生效（避免"解冻只在内存里"）"""
        acct = self._acct_with_locked_position()
        reloaded = PaperAccount.from_state(acct.to_state())
        assert reloaded.unlock_t1("2026-10-12") == 1
        assert reloaded.position("002040").available == 20000


class TestFillContract:
    """撮合返回字段的**契约**守护。

    背景（2026-10-10 实际报错）：`ui/pages/8_💰_模拟账户.py` 的确认弹窗写了
    `est['price']`，而 `matching.buy_fill/sell_fill` 返回的是 `actual_price`
    （已含滑点的成交价）—— **没有 `price` 键** → 用户一点"提交下单"就
    `KeyError: 'price'`，弹窗必崩。买入分支和卖出分支各一处，共 3 处（含盈亏行）。

    为什么必须用**契约测试**而不是"页面点了不崩就算"：
      - 页面级测试要真跑 `st.dialog`，成本高、还得模拟点击；
      - 这个 bug 的本质是 **UI 猜的键名 ≠ 撮合真实返回**，所以断言"页面引用的键
        确实存在"最直接、也最不容易漏。
    """

    REQUIRED_KEYS = {"direction", "signal_price", "actual_price", "quantity", "total", "cash_delta"}

    def test_buy_fill_and_sell_fill_expose_required_keys(self):
        for fill in (matching.buy_fill(8.94, 20000), matching.sell_fill(8.94, 20000)):
            missing = self.REQUIRED_KEYS - set(fill)
            assert not missing, f"{fill['direction']}成交明细缺字段 {missing}"

    def test_fill_has_no_bare_price_key(self):
        """反向钉死：**不允许**出现 `price` 键。

        说明为什么这是"结构性"守护而非"行为性"：如果哪天有人给 fill 加了 `price`
        并让页面继续用 `est['price']`，两处会长期各说各话（页面以为它是成交价，
        撮合侧却可能指信号价）。直接不允许该键存在，就把选错的空间关掉了。
        页面若需要成交价，唯一正确的来源是 `actual_price`。
        """
        for fill in (matching.buy_fill(10.0, 100), matching.sell_fill(10.0, 100)):
            assert "price" not in fill, (
                f"{fill['direction']}成交明细不应有 `price` 键（页面曾误用它）；"
                "成交价请用 actual_price")

    def test_actual_price_includes_slippage(self):
        """actual_price 才带滑点 —— 页面展示"预估成交价"必须用这个值

        ⚠️ 用 100.0 而不是 10.0：滑点是 0.0005，10.0×1.0005=10.005 经
        `round(actual_price, 2)` 后回到 10.0，低价股上"滑点被舍入吃掉"是**正常**的，
        用它当断言会误判成实现有问题。测试数据必须落在能体现该特性的量级上。
        """
        buy = matching.buy_fill(100.0, 100)
        sell = matching.sell_fill(100.0, 100)
        assert buy["actual_price"] > buy["signal_price"], "买入成交价应高于信号价（滑点向上）"
        assert sell["actual_price"] < sell["signal_price"], "卖出成交价应低于信号价（滑点向下）"
        assert buy["actual_price"] == 100.05 and sell["actual_price"] == 99.95

    def test_ui_page_does_not_reference_fill_price_key(self):
        """扫描页面源码：确认没有 `est['price']` 这类误用（防止改回旧写法）"""
        import pathlib
        page = (pathlib.Path(__file__).resolve().parents[1]
                / "ui" / "pages" / "8_💰_模拟账户.py")
        src = page.read_text(encoding="utf-8")
        assert "['price']" not in src.replace('fresh["price"]', "").replace("quote['price']", ""), (
            "模拟账户页出现对撮合明细 `['price']` 的取键——"
            "buy_fill/sell_fill 无该键，应用 `['actual_price']`")

    def test_confirm_dialog_renders_without_keyerror(self):
        """端到端：把弹窗依赖的那几行**原样**跑一遍，确保不再 KeyError。

        只取弹窗里真正做过数值格式化的表达式（`{est['actual_price']}`、
        `{est['total']:,.2f}`、盈亏行），不复制整段 Streamlit 代码。
        """
        price, qty = 8.94, 20000
        pos_avg_cost = 9.0868

        est = matching.buy_fill(price, qty)
        # 与页面逐字一致的两处格式化
        assert isinstance(est["actual_price"], float)
        _ = f"预估成交价 {est['actual_price']}　｜　预估金额（含佣金）**{est['total']:,.2f} 元**"

        est = matching.sell_fill(price, qty)
        _ = f"预估成交价 {est['actual_price']}　｜　预估金额（扣印花税后）**{est['total']:,.2f} 元**"
        _ = f"预计盈亏：{(est['actual_price'] - pos_avg_cost) * qty:+,.2f} 元（成本 {pos_avg_cost:.3f}）"


class TestOrderValidation:
    def test_round_lot_enforced_at_submit(self):
        acct = PaperAccount(100_000)
        assert acct.submit_order("000001", "买入", 50, "2026-01-05").status == "cancelled"
        assert acct.submit_order("000001", "买入", 150, "2026-01-05").quantity == 100
    def test_sell_without_position_cancelled(self):
        acct = PaperAccount(100_000)
        order = acct.submit_order("000001", "卖出", 100, "2026-01-05")
        assert order.status == "cancelled" and "无持仓" in order.reason

    def test_unknown_direction_cancelled(self):
        acct = PaperAccount(100_000)
        assert acct.submit_order("000001", "做多", 100, "2026-01-05").status == "cancelled"


class TestLimitOrders:
    """限价单（2026-10-27）

    背景：`submit_order` 原来只有市价单、次日开盘**无条件**成交，于是
    "跌到 15.80 才买"这个意图表达不出来——写进去会退化成"次日开盘追价"，
    恰好是想避免的行为。
    """

    def test_market_order_unchanged(self):
        """不传 limit_price ⇒ 与改动前完全一致（老调用方的行为契约）。"""
        acct = PaperAccount(100_000)
        acct.submit_order("000001", "买入", 1000, "2026-01-05")
        res = acct.settle_open("2026-01-06", {"000001": _bar(10.0, 10.0)})
        assert res[0]["status"] == "filled"
        assert res[0]["limit_price"] == 0.0
        assert res[0]["slippage"] == 0.0, "没有 decision_price 就不该算滑点"

    def test_buy_fills_when_open_below_limit(self):
        acct = PaperAccount(100_000)
        acct.submit_order("000001", "买入", 1000, "2026-01-05", limit_price=10.5)
        res = acct.settle_open("2026-01-06", {"000001": _bar(10.0, 10.0)})
        assert res[0]["status"] == "filled"

    def test_buy_cancelled_when_open_above_limit(self):
        """开盘高于限价 ⇒ 撤单。这是"不追价"的机械实现。"""
        acct = PaperAccount(100_000)
        acct.submit_order("000001", "买入", 1000, "2026-01-05", limit_price=10.0)
        res = acct.settle_open("2026-01-06", {"000001": _bar(10.5, 10.0)})
        assert res[0]["status"] == "cancelled"
        assert "高于限价" in res[0]["reason"]
        assert acct.cash == pytest.approx(100_000), "撤单不得动现金"

    def test_cancel_is_not_deferred(self):
        """⚠️ 撤单**不是顺延**：盘中理由基于当天信息，隔夜后已失效。

        如果这里顺延，明天开盘更低时会成交——等于用一个过期理由买入，
        与"只下跌买"的初衷正好相反。
        """
        acct = PaperAccount(100_000)
        acct.submit_order("000001", "买入", 1000, "2026-01-05", limit_price=10.0)
        acct.settle_open("2026-01-06", {"000001": _bar(10.5, 10.0)})
        res2 = acct.settle_open("2026-01-07", {"000001": _bar(9.5, 10.5)})
        assert res2 == [], "已撤单不应再被处理"
        assert acct.positions == {}

    def test_sell_limit_floor(self):
        """卖出限价：开盘低于限价 ⇒ 不割在更差的价。"""
        acct = PaperAccount(100_000)
        acct.apply_trade({"date": "2026-01-05", "code": "000001",
                          "direction": "买入", "price": 10.0, "quantity": 1000})
        acct.unlock_t1("2026-01-06")
        acct.submit_order("000001", "卖出", 1000, "2026-01-05", limit_price=9.8)
        res = acct.settle_open("2026-01-06", {"000001": _bar(9.5, 10.0)})
        assert res[0]["status"] == "cancelled"
        assert "低于限价" in res[0]["reason"]

    def test_slippage_recorded(self):
        """滑点 = 成交价 − 决策价。买入为正 = 追高吃亏。"""
        acct = PaperAccount(100_000)
        acct.submit_order("000001", "买入", 1000, "2026-01-05",
                          limit_price=10.5, decision_price=10.0)
        got = acct.settle_open("2026-01-06", {"000001": _bar(10.2, 10.0)})[0]
        assert got["status"] == "filled"
        expect = round(got["filled_price"] - 10.0, 4)
        assert got["slippage"] == pytest.approx(expect)
        # fills 里也要有：paper_orders 会被下一次下单覆盖，成交是不可变历史
        assert acct.fills[-1]["slippage"] == pytest.approx(expect)
        assert acct.fills[-1]["decision_price"] == pytest.approx(10.0)

    def test_limit_up_reason_wins_over_limit_price(self):
        """开盘涨停 + 也高于限价：报"涨停"，不报"高于限价"。

        顺序错了会让复盘误判成"阈值设太紧"——而实际是根本买不到，
        调阈值没有任何用。
        """
        acct = PaperAccount(100_000)
        acct.submit_order("000001", "买入", 1000, "2026-01-05", limit_price=10.0)
        res = acct.settle_open("2026-01-06", {"000001": _bar(11.0, 10.0)})
        assert res[0]["status"] == "cancelled"
        assert "涨停" in res[0]["reason"]
        assert "高于限价" not in res[0]["reason"]

    def test_zero_and_none_mean_market_order(self):
        """0 / None / 负数 / 非数字 一律视为市价单，不能当成"限价 0 元"。"""
        for lp in (0, None, -1, "abc"):
            acct = PaperAccount(100_000)
            order = acct.submit_order("000001", "买入", 1000, "2026-01-05", limit_price=lp)
            assert order.has_limit is False, lp
            assert order.limit_price == 0.0


class TestSettleOpenLimits:
    def test_buy_cancelled_on_limit_up_open(self):
        """开盘即涨停买不到：订单直接取消（不追高）"""
        acct = PaperAccount(100_000)
        acct.submit_order("000001", "买入", 1000, "2026-01-05")
        res = acct.settle_open("2026-01-06", {"000001": _bar(11.0, 10.0)})   # +10%
        assert res[0]["status"] == "cancelled"
        assert "涨停" in res[0]["reason"]
        assert acct.cash == pytest.approx(100_000)

    def test_growth_board_uses_20pct_limit(self):
        acct = PaperAccount(100_000)
        acct.submit_order("300750", "买入", 100, "2026-01-05")
        # 涨 10% 不算涨停 → 成交
        res = acct.settle_open("2026-01-06", {"300750": _bar(110.0, 100.0)})
        assert res[0]["status"] == "filled"

        acct2 = PaperAccount(100_000)
        acct2.submit_order("300750", "买入", 100, "2026-01-05")
        res2 = acct2.settle_open("2026-01-06", {"300750": _bar(120.0, 100.0)})  # 涨 20%
        assert res2[0]["status"] == "cancelled"

    def test_buy_can_fill_when_order_of_submit_day_only(self):
        """T 日提交的订单最早 T+1 成交"""
        acct = PaperAccount(100_000)
        acct.submit_order("000001", "买入", 1000, "2026-01-06")
        res = acct.settle_open("2026-01-06", {"000001": _bar(10.0, 10.0)})
        assert res == []
        assert acct.position("000001").quantity == 0

    def test_sell_deferred_on_limit_down_then_filled(self):
        acct = PaperAccount(100_000)
        acct.apply_trade({"date": "2026-01-05", "code": "000001", "direction": "买入",
                          "price": 10.0, "quantity": 1000})
        order = acct.submit_order("000001", "卖出", 1000, "2026-01-05")
        # 次日开盘跌停（-10%）→ 顺延
        res = acct.settle_open("2026-01-06", {"000001": _bar(9.0, 10.0)})
        assert res[0]["status"] == "pending" and "顺延" in res[0]["reason"]
        # 再下一日正常 → 成交
        res2 = acct.settle_open("2026-01-07", {"000001": _bar(10.5, 9.0)})
        assert res2[0]["status"] == "filled"
        assert "000001" not in acct.positions
        assert order.filled_quantity == 1000

    def test_sell_deferral_has_upper_bound(self):
        acct = PaperAccount(100_000)
        acct.apply_trade({"date": "2026-01-05", "code": "000001", "direction": "买入",
                          "price": 10.0, "quantity": 1000})
        order = acct.submit_order("000001", "卖出", 1000, "2026-01-05")
        day = 6
        for _ in range(MAX_DEFER_DAYS + 1):
            acct.settle_open(f"2026-01-{day:02d}", {"000001": _bar(9.0, 10.0)})
            day += 1
        assert order.status == "cancelled"
        assert "取消" in order.reason

    def test_suspended_stock_cancels_buy_defers_sell(self):
        acct = PaperAccount(100_000)
        acct.submit_order("000001", "买入", 1000, "2026-01-05")
        res = acct.settle_open("2026-01-06", {})          # 停牌
        assert res[0]["status"] == "cancelled"

        acct2 = PaperAccount(100_000)
        acct2.apply_trade({"date": "2026-01-05", "code": "000001", "direction": "买入",
                           "price": 10.0, "quantity": 1000})
        order = acct2.submit_order("000001", "卖出", 1000, "2026-01-05")
        res2 = acct2.settle_open("2026-01-06", {})
        assert res2[0]["status"] == "pending" and order.status == "pending"

    def test_sell_proceeds_available_for_same_day_buy(self):
        """同日先卖后买：卖出回笼的现金当日可用（与回测引擎顺序一致）"""
        acct = PaperAccount(20_000)
        acct.apply_trade({"date": "2026-01-05", "code": "000001", "direction": "买入",
                          "price": 10.0, "quantity": 1000})     # 花费约 1 万
        assert acct.cash < 10_500
        acct.submit_order("000001", "卖出", 1000, "2026-01-05")
        acct.submit_order("600519", "买入", 500, "2026-01-05")
        res = acct.settle_open("2026-01-06", {
            "000001": _bar(11.0, 10.0), "600519": _bar(10.0, 10.0)})
        statuses = {r["code"]: r["status"] for r in res}
        assert statuses["000001"] == "filled" and statuses["600519"] == "filled"


class TestValuation:
    def test_missing_price_falls_back_to_avg_cost(self):
        acct = PaperAccount(100_000)
        acct.apply_trade({"date": "2026-01-05", "code": "000001", "direction": "买入",
                          "price": 10.0, "quantity": 1000})
        pos = acct.position("000001")
        assert acct.holdings_value({}) == pytest.approx(pos.avg_cost * 1000, abs=0.01)

    def test_mark_to_market_records_return(self):
        acct = PaperAccount(100_000)
        acct.apply_trade({"date": "2026-01-05", "code": "000001", "direction": "买入",
                          "price": 10.0, "quantity": 1000})
        rec = acct.mark_to_market("2026-01-05", _prices("2026-01-05"))
        assert rec["date"] == "2026-01-05"
        assert rec["total_value"] == pytest.approx(acct.cash + 1000 * 10.0, abs=0.01)
        assert rec["return_pct"] == pytest.approx(
            (rec["total_value"] / 100_000 - 1) * 100, abs=0.01)

    def test_position_to_dict_exposes_avg_cost(self):
        pos = Position(code="000001", quantity=1000, available=0, total_cost=10010.0)
        assert pos.to_dict()["avg_cost"] == pytest.approx(10.01)


class TestConsistencyWithBacktest:
    """Phase 2 验收：模拟盘与回测在相同信号下结果必须一致"""

    def test_cash_nav_and_positions_match_engine(self):
        engine = BacktestEngine()
        engine.inject_kline_cache(_kline())
        bt = engine.run([dict(t) for t in TRADES])

        acct = PaperAccount(initial_cash=1_000_000.0)
        # 必须与引擎同构：逐日"先成交、再记净值"，否则净值序列的时点对不上
        for date in DATES:
            acct.apply_trades([dict(t) for t in TRADES if t["date"] == date])
            acct.mark_to_market(date, _prices(date))

        assert acct.cash == pytest.approx(bt.nav_curve[-1]["cash"], abs=0.01)
        assert acct.nav[-1]["total_value"] == pytest.approx(
            bt.nav_curve[-1]["total_value"], abs=0.01)
        for a, b in zip(acct.nav, bt.nav_curve):
            assert a["date"] == b["date"]
            assert a["total_value"] == pytest.approx(b["total_value"], abs=0.01)

        # 持仓数量：由引擎的成交明细反推，与账户持仓逐只比对。
        # ⚠️ 必须**排除 rejected 明细**（v3 / P1-4 之后 trade_history 里会有被拒订单，
        #    它们数量非 0 但不产生持仓，不排除就会算错）
        engine_qty: dict[str, int] = {}
        for t in bt.trade_history:
            if t.get("status") == "rejected":
                continue
            sign = 1 if t["direction"] == "买入" else -1
            engine_qty[t["code"]] = engine_qty.get(t["code"], 0) + sign * t["quantity"]
        account_qty = {c: p.quantity for c, p in acct.positions.items()}
        assert account_qty == {c: q for c, q in engine_qty.items() if q > 0}

    def test_cash_shortage_rejected_like_engine(self):
        """现金不足时两边**都不成交**，而且**都要留痕**。

        v3（P1-4）：引擎原先"静默跳过"——策略以为买了 5 只、实际只成交 3 只，
        结果与策略意图不符且无从察觉（本用例原来正是在断言这个静默行为）。
        现在引擎也写一条 `status="rejected"` 的明细并计入 `rejected_orders`，
        与账户侧的显式拒单口径对齐。**两边都必须"不成交"这一点没有变。**
        """
        engine = BacktestEngine()
        engine.inject_kline_cache(_kline())
        trades = [{"date": "2026-01-05", "code": "000001", "direction": "买入",
                   "price": 10.0, "quantity": 1_000_000}]     # 远超 100 万本金
        bt = engine.run(trades)

        # 不得有任何**成交**的买单
        assert all(t.get("status") != "filled" for t in bt.trade_history)
        # 且必须留痕：被拒订单数与明细
        assert bt.rejected_orders == 1
        rejected = [t for t in bt.trade_history if t.get("status") == "rejected"]
        assert rejected and rejected[0]["reason"] == "现金不足"
        # 现金没动、没有持仓
        assert bt.nav_curve[-1]["cash"] == pytest.approx(1_000_000.0, abs=0.01)
        assert bt.nav_curve[-1]["position_value"] == 0

        acct = PaperAccount(initial_cash=1_000_000.0)
        out = acct.apply_trades(trades)
        assert out[0]["status"] == "rejected"
        assert acct.cash == pytest.approx(1_000_000.0)

    def test_engine_partial_sell_uses_reduced_cost_basis(self):
        """引擎与账户都按均价扣减成本基数（部分卖出后均价不变）"""
        engine = BacktestEngine()
        engine.inject_kline_cache({
            "000001": pd.DataFrame({"date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
                                    "close": [10.0, 12.0]})})
        bt = engine.run([
            {"date": "2026-01-05", "code": "000001", "direction": "买入",
             "price": 10.0, "quantity": 1000},
            {"date": "2026-01-06", "code": "000001", "direction": "卖出",
             "price": 12.0, "quantity": 400},
            {"date": "2026-01-07", "code": "000001", "direction": "卖出",
             "price": 12.0, "quantity": 600},
        ])
        buy = bt.trade_history[0]
        avg_cost = buy["cost"] + buy["commission"]
        avg_cost /= buy["quantity"]
        sell1, sell2 = bt.trade_history[1], bt.trade_history[2]
        for sell in (sell1, sell2):
            exp = matching.pnl_of({"actual_price": sell["actual_price"],
                                   "quantity": sell["quantity"],
                                   "commission": sell["commission"],
                                   "stamp_tax": sell["stamp_tax"]}, avg_cost)
            assert sell["pnl"] == pytest.approx(exp, abs=0.02)
