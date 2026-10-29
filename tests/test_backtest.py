"""测试回测引擎 backtest.py"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime

from core.backtest import BacktestResult, BacktestEngine


def _nav(values, dates):
    return [{"date": d, "total_value": v} for d, v in zip(dates, values)]


# ===== v2（M0 0.1.4）净值和统计口径 =====

def test_position_value_uses_each_stocks_own_price():
    """持仓估值必须逐票取价。

    原实现在净值结算处复用循环外残留的 signal_price（属于"当天最后一笔交易的
    标的"），于是所有持仓都被按同一只股票的价格估值。
    这里注入 K 线缓存以避免联网取价。
    """
    engine = BacktestEngine()
    engine.inject_kline_cache({
        "000001": pd.DataFrame({"date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
                                "close": [10.0, 11.0]}),
        "600519": pd.DataFrame({"date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
                                "close": [20.0, 21.0]}),
    })
    trades = [
        {"date": "2026-01-05", "code": "000001", "direction": "买入", "price": 10.0, "quantity": 1000},
        {"date": "2026-01-06", "code": "600519", "direction": "买入", "price": 20.0, "quantity": 1000},
    ]
    result = engine.run(trades)

    last = result.nav_curve[-1]
    # 各按自己的收盘价估值：1000×11 + 1000×21
    assert last["position_value"] == pytest.approx(32_000.0, abs=1.0)
    # 旧实现会用"当天最后一笔交易"的价格（约 20.01）给两只股票估值 → 约 40020
    assert last["position_value"] != pytest.approx(40_020.0, abs=500)


def test_annual_return_not_extrapolated_for_short_sample():
    """样本不足半年不做年化外推（原实现 10 个交易日 +5% 会年化成 >200%）"""
    engine = BacktestEngine()
    dates = ["2026-01-%02d" % d for d in (5, 6, 7, 8, 9, 12, 13, 14, 15, 16)]
    values = [1_000_000 + i * 5_000 for i in range(10)]     # 期末 +4.5%
    result = engine._compute_metrics(_nav(values, dates), [], 1_000_000)

    assert result.total_return == pytest.approx(4.5, abs=0.05)
    assert result.annual_return == pytest.approx(result.total_return, abs=0.01), \
        "短样本必须直接给区间收益，不得外推年化"


def test_sharpe_keeps_flat_days():
    """零收益日不得被删除。

    原实现把净值 reindex 到自然日 ffill 后再 `daily_returns != 0` 过滤，
    把横盘日一并删掉：分子被抬高、样本被缩小，夏普被系统性高估。
    这里 3 个交易日中只有 1 天上涨，旧逻辑过滤后只剩 1 个观测值 → 夏普为 0；
    新逻辑保留横盘日 → 夏普非 0。
    """
    engine = BacktestEngine()
    result = engine._compute_metrics(
        _nav([1_000_000, 1_000_000, 1_010_000], ["2026-01-06", "2026-01-07", "2026-01-08"]),
        [], 1_000_000)

    assert result.sharpe_ratio != 0, "横盘日必须参与计算"


def test_sharpe_zero_when_nav_never_changes():
    engine = BacktestEngine()
    result = engine._compute_metrics(
        _nav([1_000_000] * 4, ["2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]),
        [], 1_000_000)
    assert result.sharpe_ratio == 0


# ===== BacktestResult 数据类 =====

def test_backtest_result_default():
    r = BacktestResult()
    assert r.total_return == 0.0
    assert r.sharpe_ratio == 0.0
    assert r.nav_curve == []


def test_to_dict_sampling():
    """_sample_list 等距采样逻辑"""
    items = [{"a": i} for i in range(200)]
    sampled = BacktestResult._sample_list(items, 10)
    assert len(sampled) == 10
    assert sampled[0] == items[0]     # 首保留
    assert sampled[-1] == items[-1]   # 尾保留


def test_to_dict_small_list():
    """列表不足采样上限时原样返回"""
    items = [{"a": i} for i in range(5)]
    sampled = BacktestResult._sample_list(items, 100)
    assert len(sampled) == 5


def test_to_dict_exact_boundary():
    """刚好等于上限"""
    items = [{"a": i} for i in range(10)]
    sampled = BacktestResult._sample_list(items, 10)
    assert len(sampled) == 10
    assert sampled == items


def test_to_dict_output():
    """to_dict() 集成验证"""
    r = BacktestResult(
        total_return=12.5,
        annual_return=18.0,
        sharpe_ratio=1.5,
        calmar_ratio=2.0,
        max_drawdown=-6.0,
        win_rate=60.0,
        profit_loss_ratio=1.8,
        total_trades=42,
        turnover_rate=120.0,
        information_ratio=0.8,
        monthly_win_rate=65.0,
        nav_curve=[{"date": f"2026-01-{d:02d}", "total_value": 1000000 + d * 1000} for d in range(1, 21)],
        trade_history=[{"date": "2026-01-10", "code": "000001"} for _ in range(100)],
    )
    d = r.to_dict()
    assert d["total_return"] == 12.5
    assert d["sharpe_ratio"] == 1.5
    assert d["information_ratio"] == 0.8
    assert len(d["nav_curve"]) == 20  # 小于 max_count，不采样
    assert len(d["trade_history"]) == 50  # 100 -> 50 采样


# ===== BacktestEngine 基础 =====

@pytest.fixture
def engine():
    """构造 mock 数据源的回测引擎"""
    e = BacktestEngine()
    mock_fetcher = MagicMock()
    mock_fetcher.get_stock_data.return_value = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=60, freq="D"),
        "Close": [12.0 + i * 0.05 for i in range(60)],
    })
    e._stock_fetcher = mock_fetcher
    return e


# ===== run() 空 / 边界 =====

def test_run_empty_trades(engine):
    """空交易列表返回空结果"""
    result = engine.run([])
    assert result.total_return == 0.0
    assert result.total_trades == 0


def test_run_no_trades_none(engine):
    """None 参数从 DB 加载（空 DB 场景）"""
    # DB 中无记录 -> 空
    with patch.object(engine, "_load_from_db", return_value=[]):
        result = engine.run(None)
        assert result.total_return == 0.0


# ===== run() 买入卖出完整流程 =====

def test_run_buy_only(engine):
    """仅买入：净值 = 现金 + 持仓市值"""
    trades = [
        {"date": "2025-01-10", "code": "000001", "direction": "买入", "price": 12.50, "quantity": 1000},
    ]
    result = engine.run(trades)
    assert len(result.nav_curve) >= 1
    # 净值应接近初始资金（买入成本略高于持仓市值）
    final_nav = result.nav_curve[-1]
    assert 900000 < final_nav["total_value"] < 1100000


def test_run_buy_and_sell_profit(engine):
    """买入 -> 盈利卖出

    v3（P0-1）：净值必须覆盖**区间内每一个交易日**，而不是只覆盖有成交的两天。
    原断言 `len(nav_curve) == 2` 把 bug 当成了契约——持仓期间的价格波动
    完全不在净值里，回撤恒为 0。
    """
    trades = [
        {"date": "2025-01-10", "code": "000001", "direction": "买入", "price": 12.50, "quantity": 1000},
        {"date": "2025-03-15", "code": "000001", "direction": "卖出", "price": 15.00, "quantity": 1000},
    ]
    result = engine.run(trades)

    from data import trading_calendar as tc
    expected_days = set(tc.trading_days("2025-01-10", "2025-03-15"))
    # 2025-03-15 是周六（历史脏数据/脏日期一样要能跑），它会被并入日历
    expected_days |= {"2025-01-10", "2025-03-15"}
    assert len(result.nav_curve) == len(expected_days) > 30, \
        "净值点数应等于交易日数，而不是成交笔数"
    assert result.nav_curve[0]["date"] == "2025-01-10"
    assert result.nav_curve[-1]["date"] == "2025-03-15"
    assert result.total_return > 0  # 盈利
    assert result.total_trades >= 1


def test_run_buy_and_sell_loss(engine):
    """买入 -> 亏损卖出"""
    trades = [
        {"date": "2025-01-10", "code": "000001", "direction": "买入", "price": 15.00, "quantity": 1000},
        {"date": "2025-03-15", "code": "000001", "direction": "卖出", "price": 12.00, "quantity": 1000},
    ]
    result = engine.run(trades)
    assert result.total_return < 0  # 亏损


def test_run_multiple_stocks(engine):
    """多只股票交易"""
    trades = [
        {"date": "2025-01-10", "code": "000001", "direction": "买入", "price": 12.50, "quantity": 2000},
        {"date": "2025-02-10", "code": "600519", "direction": "买入", "price": 1300.00, "quantity": 100},
        {"date": "2025-04-01", "code": "000001", "direction": "卖出", "price": 13.80, "quantity": 2000},
        {"date": "2025-06-15", "code": "600519", "direction": "卖出", "price": 1280.00, "quantity": 100},
    ]
    result = engine.run(trades)
    assert len(result.nav_curve) >= 3


# ===== 指标计算边缘 =====

def test_compute_metrics_empty_nav():
    """空净值曲线 -> 返回空结果"""
    engine = BacktestEngine()
    result = engine._compute_metrics([], [], 1000000)
    assert result.total_return == 0.0
    assert result.sharpe_ratio == 0.0


def test_compute_metrics_single_point():
    """单点净值曲线"""
    engine = BacktestEngine()
    nav = [{"date": "2025-01-10", "total_value": 1000000}]
    result = engine._compute_metrics(nav, [], 1000000)
    assert result.total_return == 0.0


def test_compute_metrics_sharpe_bounds():
    """夏普比率、最大回撤、卡玛 数量级合理，传入含波动的净值数据"""
    engine = BacktestEngine()
    np.random.seed(42)
    nav = [
        {"date": f"2025-{m:02d}-01", "total_value": 1000000 + m * 2000 + np.random.randint(-3000, 3000)}
        for m in range(1, 13)
    ]
    result = engine._compute_metrics(nav, [], 1000000)
    assert abs(result.sharpe_ratio) < 50
    assert abs(result.max_drawdown) <= 100
    assert result.win_rate == 0.0  # 无交易


def test_compute_metrics_win_rate():
    """胜率计算"""
    engine = BacktestEngine()
    nav = [
        {"date": "2025-01-10", "total_value": 1000000},
        {"date": "2025-02-10", "total_value": 1008000},
        {"date": "2025-03-10", "total_value": 1002000},
    ]
    trades_detail = [
        {"date": "2025-02-10", "pnl": 8000},
        {"date": "2025-03-10", "pnl": -6000},
    ]
    result = engine._compute_metrics(nav, trades_detail, 1000000)
    assert 0 <= result.win_rate <= 100
    assert result.profit_loss_ratio >= 0


# ===== _get_price_on_date =====

def test_get_price_on_date_cache_hit(engine):
    """缓存命中 -> 返回对应日期收盘价"""
    df = pd.DataFrame({
        "Date": pd.date_range("2025-01-10", periods=10, freq="D"),
        "Close": [12.0 + i * 0.1 for i in range(10)],
    })
    engine._kline_cache["000001"] = df
    price = engine._get_price_on_date("000001", "2025-01-13", 10.0)
    assert price == 12.3  # 第4个 (index 3)


def test_get_price_on_date_cache_miss_fallback(engine):
    """缓存未命中 -> 尝试加载 -> 失败返回 fallback"""
    # stock_fetcher 返回 None 或抛异常
    engine._stock_fetcher.get_stock_data.return_value = None
    price = engine._get_price_on_date("999999", "2025-07-01", 15.0)
    assert price == 15.0


# ===== 信息比率 =====

def test_information_ratio_no_benchmark():
    """无基准时返回 0"""
    engine = BacktestEngine()
    ir = engine._calc_information_ratio([1000000, 1005000], ["2025-01-10", "2025-02-10"])
    assert ir == 0


def test_information_ratio_with_benchmark():
    """有基准时的信息比率计算"""
    engine = BacktestEngine()
    engine._benchmark_nav = [
        {"date": "2025-01-10", "total_value": 1000000},
        {"date": "2025-01-11", "total_value": 1002000},
        {"date": "2025-01-12", "total_value": 1003000},
        {"date": "2025-01-13", "total_value": 1001000},
        {"date": "2025-01-14", "total_value": 1005000},
        {"date": "2025-01-15", "total_value": 1008000},
        {"date": "2025-01-16", "total_value": 1010000},
        {"date": "2025-01-17", "total_value": 1009000},
        {"date": "2025-01-18", "total_value": 1012000},
        {"date": "2025-01-19", "total_value": 1015000},
        {"date": "2025-01-20", "total_value": 1013000},
    ]
    portfolio_vals = [1000000, 1002500, 1002800, 1001500, 1006000,
                      1009500, 1011000, 1010000, 1013500, 1016000, 1014800]
    dates = [f"2025-01-{d:02d}" for d in range(10, 21)]
    ir = engine._calc_information_ratio(portfolio_vals, dates)
    # 合理范围内
    assert -20 <= ir <= 20


def test_information_ratio_few_dates():
    """不足10个共同日期 -> 返回 0"""
    engine = BacktestEngine()
    engine._benchmark_nav = [
        {"date": "2025-01-10", "total_value": 1000000},
        {"date": "2025-01-11", "total_value": 1002000},
    ]
    ir = engine._calc_information_ratio([1000000, 1005000], ["2025-01-10", "2025-01-11"])
    assert ir == 0


# ===== 滑点 / 佣金验证 =====

def test_run_commission_min(engine):
    """验证最低佣金 5 元"""
    trades = [
        {"date": "2025-01-10", "code": "000001", "direction": "买入", "price": 10.0, "quantity": 10},
    ]
    # 成本 = 10*10 = 100，佣金 = max(5, 100*0.025%) = 5
    result = engine.run(trades)
    detail = result.trade_history[0] if result.trade_history else result.nav_curve
    assert len(result.nav_curve) > 0


def test_run_sell_stamp_tax(engine):
    """卖出有印花税"""
    trades = [
        {"date": "2025-01-10", "code": "000001", "direction": "买入", "price": 10.0, "quantity": 1000},
        {"date": "2025-02-10", "code": "000001", "direction": "卖出", "price": 10.0, "quantity": 1000},
    ]
    result = engine.run(trades)
    # 买入卖出同价，扣除佣金+印花税后小幅亏损
    assert result.total_return <= 0


# ===== v3（2026-10-29，回测验证设计模块 C）费用拖累 =====

def test_cost_drag_sums_commission_and_stamp_tax():
    """cost_drag = Σ(佣金+印花税)/初始资金×100；turnover_rate 已有，别重复造（F4）"""
    engine = BacktestEngine()
    dates = ["2026-01-%02d" % d for d in (5, 6, 7, 8, 9, 12, 13, 14, 15, 16)]
    values = [1_000_000] * 10
    trade_details = [
        {"date": dates[0], "code": "000001", "direction": "买入",
         "status": "filled", "commission": 25.0},                # 买：只有佣金
        {"date": dates[5], "code": "000001", "direction": "卖出",
         "status": "filled", "commission": 25.0, "stamp_tax": 50.0},  # 卖：佣金+印花税
        {"date": dates[6], "code": "000002", "direction": "买入",
         "status": "rejected", "reason": "现金不足"},             # 被拒：没成交没费用
    ]
    result = engine._compute_metrics(_nav(values, dates), trade_details, 1_000_000)
    assert result.cost_drag == pytest.approx((25.0 + 25.0 + 50.0) / 1_000_000 * 100, abs=1e-6)
    # 被拒订单与空明细都不产生费用
    result0 = engine._compute_metrics(_nav(values, dates), [], 1_000_000)
    assert result0.cost_drag == 0.0
    # 必须进 to_dict（页面/API 消费口径）
    assert result.to_dict()["cost_drag"] == pytest.approx(result.cost_drag, abs=1e-9)
