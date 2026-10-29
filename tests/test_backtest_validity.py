"""回测引擎的**有效性**守护（v3 / P0-1、P1-4、P1-5、P1-6、P2-5）

这些用例针对的是"数字看起来正常、其实是错的"这一类缺陷——比崩溃危险得多：
调参、写报告、答辩，全都会拿这些数字当依据。

对应 `早期文档/回测模块修改建议报告.md` 附录 A 的验收清单：
  - test_max_drawdown_captures_unrealized_loss      （P0-1 核心验收）
  - test_nav_curve_covers_all_trading_days          （P0-1）
  - test_buy_rejected_when_cash_insufficient_is_recorded （P1-4）
  - test_missing_kline_warns_instead_of_silently_using_cost （P1-6）
  - test_period_covering_scales_with_span           （P1-6）
  - test_avg_position_pct_is_reported               （P1-5）
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import pytest

from core.backtest import BacktestEngine, _period_covering
from data import trading_calendar as tc


def _seq(start: str, n: int) -> list:
    """从 start 起的 n 个连续交易日（用项目自带日历，避免手写日期踩节假日）"""
    end = (pd.to_datetime(start) + timedelta(days=n * 2 + 20)).strftime("%Y-%m-%d")
    return [str(d) for d in tc.trading_days(start, end)][:n]


def _kline(rows) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime([r[0] for r in rows]),
                         "close": [float(r[1]) for r in rows]})


def _engine_with(cache: dict) -> BacktestEngine:
    engine = BacktestEngine()
    engine.inject_kline_cache(cache)
    return engine


# ══════════════════════════════════════════════════════════
# P0-1：逐日估值 —— 核心验收
# ══════════════════════════════════════════════════════════

def test_nav_curve_covers_all_trading_days():
    """净值点数应等于**交易日数**，而不是成交笔数（原实现只在有交易的日子记账）"""
    days = _seq("2025-01-10", 25)
    engine = _engine_with({"000001": _kline([(d, 10.0) for d in days])})
    trades = [
        {"date": days[0], "code": "000001", "direction": "买入", "price": 10.0, "quantity": 1000},
        {"date": days[-1], "code": "000001", "direction": "卖出", "price": 10.0, "quantity": 1000},
    ]
    result = engine.run(trades)

    assert len(result.nav_curve) == len(days) == 25
    assert [n["date"] for n in result.nav_curve] == days


def test_max_drawdown_captures_unrealized_loss():
    """**P0-1 的核心验收**：持仓期的浮亏必须计入回撤。

    构造：T 日买入 50% 仓位 → 中间 20 个交易日阴跌 25%（全程无交易）→ T+21 卖出。

    修改前：nav_curve 只有"买入日"和"卖出日"两个点 → `max_drawdown` **恒为 0**，
            而 calmar = annual_return / |max_drawdown| 还会因分母趋零被高估。
    修改后：逐日 mark-to-market → 回撤约 −12.5%（= 25% 跌幅 × 50% 仓位权重）。
    """
    days = _seq("2025-01-10", 22)
    # 第 0 天 10.00 元，线性跌到第 20 天的 7.50 元（−25%），第 21 天平仓
    closes = [10.0 - 2.5 * min(i, 20) / 20 for i in range(22)]
    engine = _engine_with({"000001": _kline(list(zip(days, closes)))})
    trades = [
        {"date": days[0], "code": "000001", "direction": "买入", "price": 10.0, "quantity": 50_000},
        {"date": days[21], "code": "000001", "direction": "卖出", "price": closes[21], "quantity": 50_000},
    ]
    result = engine.run(trades)

    assert len(result.nav_curve) == 22
    # 关键判据：不是 0。修好之前这个断言必红。
    assert result.max_drawdown < 0, "持仓期浮亏必须反映到回撤上（原实现恒为 0）"
    assert result.max_drawdown == pytest.approx(-12.5, abs=0.6), result.max_drawdown

    # 净值曲线中途确实记录了浮亏，而不是首尾两点拉平
    navs = [n["total_value"] for n in result.nav_curve]
    assert min(navs) < navs[0] * 0.9, "中途必须能看到浮亏，否则又是稀疏净值"
    mid = navs[1:-1]
    assert min(mid) < navs[0] * 0.95, "亏损必须出现在持仓途中，而不是只在平仓那天结算"


def test_trading_day_without_any_trade_still_gets_a_nav_point():
    """买入后长期不动、区间内也没有别的交易 → 仍然每天一个净值点。

    区间末端由 `end_date` 显式给出——引擎只从成交记录推不出"样本到哪天为止"。
    """
    days = _seq("2025-03-03", 12)
    engine = _engine_with({"600519": _kline([(d, 100.0 + i) for i, d in enumerate(days)])})
    result = engine.run(
        [{"date": days[0], "code": "600519", "direction": "买入", "price": 100.0, "quantity": 1000}],
        end_date=days[-1])

    assert len(result.nav_curve) == len(days)
    # 单调上涨的行情下净值应逐日抬升（说明每天真的重新估值了）
    navs = [n["total_value"] for n in result.nav_curve]
    assert navs[-1] > navs[0]
    assert len(set(navs)) > 1


def test_sharpe_uses_a_share_trading_calendar_not_calendar_business_days():
    """P2-5：夏普展开用真实交易日历，节假日不再被 ffill 成"零收益日"。

    判据：把净值日期定在**跨越春节**的区间，日历展开的天数应等于区间内
    真实交易日数，而不是 `freq="B"` 的自然工作日数（后者更大）。
    """
    start, end = "2025-01-20", "2025-02-20"
    days = [str(d) for d in tc.trading_days(start, end)]
    engine = BacktestEngine()
    nav = [{"date": d, "total_value": 1_000_000 + i * 100} for i, d in enumerate(days)]
    result = engine._compute_metrics(nav, [], 1_000_000)

    cal = engine._sharpe_calendar(days)
    business = pd.date_range(start, end, freq="B")
    assert len(cal) == len(days)
    assert len(cal) < len(business), "春节假期必须被排除，否则零收益日被凭空造出来"
    assert result.sharpe_ratio != 0


# ══════════════════════════════════════════════════════════
# P1-4：现金不足必须记录，而不是静默丢弃
# ══════════════════════════════════════════════════════════

def test_buy_rejected_when_cash_insufficient_is_recorded():
    """买不起就**记录**：策略以为买了 N 只、实际成交更少，这件事必须可观测"""
    days = _seq("2025-01-06", 3)
    engine = _engine_with({"600519": _kline([(d, 100.0) for d in days])})
    result = engine.run([
        # 100,000 股 × 100 元 = 1000 万 >> 100 万本金 → 必然被拒
        {"date": days[0], "code": "600519", "direction": "买入", "price": 100.0, "quantity": 100_000},
    ])

    assert result.rejected_orders == 1
    rejected = [t for t in result.trade_history if t.get("status") == "rejected"]
    assert rejected, "被拒订单必须写进交易明细，否则无法察觉"
    assert rejected[0]["reason"] == "现金不足"
    assert rejected[0]["code"] == "600519"
    assert rejected[0]["required"] > 1_000_000
    # 被拒之后不能凭空多出持仓
    assert all(n["position_value"] == 0 for n in result.nav_curve)


def test_rejected_order_does_not_block_later_affordable_buy():
    """同一天先被拒一笔、再成交一笔：被拒那笔不得污染现金与持仓"""
    days = _seq("2025-01-06", 3)
    engine = _engine_with({"600519": _kline([(d, 100.0) for d in days]),
                           "000001": _kline([(d, 10.0) for d in days])})
    result = engine.run([
        {"date": days[0], "code": "600519", "direction": "买入", "price": 100.0, "quantity": 100_000},
        {"date": days[0], "code": "000001", "direction": "买入", "price": 10.0, "quantity": 1_000},
    ])

    assert result.rejected_orders == 1
    filled = [t for t in result.trade_history if t.get("status") == "filled"]
    assert len(filled) == 1 and filled[0]["code"] == "000001"
    assert result.nav_curve[-1]["position_value"] == pytest.approx(10_000.0, abs=1.0)


# ══════════════════════════════════════════════════════════
# P1-6：兜底价不得静默退到成本价
# ══════════════════════════════════════════════════════════

def test_fallback_price_is_carried_forward_not_cost_price():
    """行情只到第 5 天，之后必须**顺延最近收盘价**，而不是退回成本价。

    退成本价 = 浮亏被整个抹掉，净值曲线变成一条假的水平线。
    """
    days = _seq("2025-01-06", 12)
    engine = _engine_with({"000001": _kline([(d, c) for d, c in
                                             zip(days[:6], [10.0, 9.0, 8.0, 7.0, 6.0, 5.0])])})
    result = engine.run(
        [{"date": days[0], "code": "000001", "direction": "买入", "price": 10.0, "quantity": 10_000}],
        end_date=days[-1])

    tail = result.nav_curve[-1]["position_value"]
    # 成本价口径会给 100,000（把 −50% 的浮亏抹掉）；顺延口径给 50,000
    assert tail == pytest.approx(50_000.0, abs=50.0), "必须顺延最近收盘价，不得退到成本价"
    assert result.max_drawdown < -4.0


def test_missing_kline_warns_instead_of_silently_using_cost(caplog):
    """完全取不到行情时，退到成本价是**最后手段**，且必须告警"""
    engine = BacktestEngine()
    engine._stock_fetcher = type("F", (), {"get_stock_data": staticmethod(lambda *a, **k: None)})()
    with caplog.at_level(logging.WARNING, logger="core.backtest"):
        result = engine.run([
            {"date": "2025-01-06", "code": "999999", "direction": "买入",
             "price": 10.0, "quantity": 1000},
        ])

    assert result.nav_curve, "无行情也不能崩，净值曲线仍需产出"
    assert any("成本价" in r.message for r in caplog.records), \
        "静默用成本价正是 P1-6 的病根，必须留下告警"


@pytest.mark.parametrize("delta_days,expect", [
    (-30, "3mo"), (-150, "6mo"),
])
def test_period_covering_short_spans(delta_days, expect):
    start = (datetime.now() + timedelta(days=delta_days)).strftime("%Y-%m-%d")
    assert _period_covering(start) == expect


def test_period_covering_long_span_scales_up():
    """回测起点在 2 年前时不能再要"1y"——否则历史日期一条都命不中（P1-6）"""
    start = (datetime.now() - timedelta(days=800)).strftime("%Y-%m-%d")
    period = _period_covering(start)
    assert period.endswith("y") and int(period[:-1]) >= 3, period


def test_period_covering_future_start_does_not_crash():
    start = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    assert _period_covering(start) == "3mo"
    assert _period_covering("") == "1y"
    assert _period_covering("不是日期") == "1y"


# ══════════════════════════════════════════════════════════
# P1-5：半仓口径必须可观测
# ══════════════════════════════════════════════════════════

def test_avg_position_pct_is_reported():
    """平均仓位要如实输出：5 只 × 10% = 50% 上限，与满仓基准对比时口径必须说明"""
    days = _seq("2025-01-06", 10)
    engine = _engine_with({"000001": _kline([(d, 10.0) for d in days])})
    result = engine.run([
        {"date": days[0], "code": "000001", "direction": "买入", "price": 10.0, "quantity": 10_000},
    ])

    assert 0 < result.avg_position_pct < 20, result.avg_position_pct  # 10 万 / 100 万 ≈ 10%
    assert result.avg_position_pct == pytest.approx(10.0, abs=1.0)


def test_report_and_dict_include_new_fields():
    """新增字段必须同时出现在 format_report 与 to_dict —— 否则 UI/看板读不到"""
    days = _seq("2025-01-06", 5)
    engine = _engine_with({"000001": _kline([(d, 10.0) for d in days])})
    result = engine.run([
        {"date": days[0], "code": "000001", "direction": "买入", "price": 10.0, "quantity": 1000},
    ])

    text = engine.format_report(result)
    assert "平均仓位" in text and "被拒订单" in text
    data = result.to_dict()
    assert "avg_position_pct" in data and "rejected_orders" in data
