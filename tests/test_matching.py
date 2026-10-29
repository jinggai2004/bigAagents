"""撮合与费用测试（Phase 2）

两条主线：
  1. 费用口径本身（佣金最低 5 元、滑点方向、印花税只在卖出端、整手、现金校验、盈亏）；
  2. **唯一实现约束**：回测引擎的成交金额必须与 core/matching 完全一致——
     Phase 2 的模拟盘也用它，两套撮合必然分叉。
"""

import pandas as pd
import pytest

from core import matching
from core.backtest import BacktestEngine


class TestCommission:
    def test_minimum_applies_to_small_trades(self):
        # 1000 元 × 0.025% = 0.25 元 < 5 元 → 取最低 5 元
        assert matching.commission_of(1000) == 5.0

    def test_rate_applies_to_large_trades(self):
        assert matching.commission_of(1_000_000) == pytest.approx(250.0)


class TestBuyFill:
    def test_slippage_and_no_stamp_tax(self):
        f = matching.buy_fill(10.0, 1000)
        assert f["direction"] == "买入"
        assert f["commission"] == 5.0          # 2.5 元低于最低佣金
        assert f["stamp_tax"] == 0.0           # A 股买入不收印花税
        assert f["cost"] == pytest.approx(10005.0, abs=0.01)
        assert f["total"] == pytest.approx(10010.0, abs=0.01)
        assert f["cash_delta"] == -f["total"]


class TestSellFill:
    def test_slippage_lowers_price_and_charges_stamp_tax(self):
        f = matching.sell_fill(10.0, 10000)
        assert f["direction"] == "卖出"
        assert f["revenue"] == pytest.approx(99950.0, abs=0.1)
        assert f["commission"] == pytest.approx(99950 * matching.COMMISSION_RATE, abs=0.01)
        assert f["stamp_tax"] == pytest.approx(99.95, abs=0.01)
        assert f["total"] == pytest.approx(
            f["revenue"] - f["commission"] - f["stamp_tax"], abs=0.02)
        assert f["cash_delta"] == f["total"]


class TestPriceRounding:
    """已知口径：成交价四舍五入到分，子分级滑点会被抹掉。

    对 10 元股票，0.05% 滑点 = 0.005 元 → 展示价被舍入到 10.00；
    而成本/收入仍按**未舍入**价计算，因此 actual_price × quantity 与 cost 之间
    会有小数级别的差异。这是历史回测口径，本次抽取保持行为不变；
    若要提高精度只需改 core/matching.py 一处（回测与模拟盘同时生效）。
    """

    def test_buy_display_price_rounded_but_cost_unrounded(self):
        f = matching.buy_fill(10.0, 1000)
        assert f["actual_price"] == 10.0
        assert f["cost"] == pytest.approx(10.005 * 1000, abs=0.01)

    def test_sell_display_price_rounded_but_revenue_unrounded(self):
        f = matching.sell_fill(10.0, 10000)
        assert f["actual_price"] == 10.0
        assert f["revenue"] == pytest.approx(9.995 * 10000, abs=0.01)


class TestRoundLot:
    @pytest.mark.parametrize("qty, expected", [
        (0, 0), (99, 0), (100, 100), (150, 100), (1050, 1000), (-5, 0), (None, 0),
    ])
    def test_rounding(self, qty, expected):
        assert matching.round_lot(qty) == expected


class TestCanAfford:
    def test_boundary(self):
        f = matching.buy_fill(10.0, 1000)
        assert matching.can_afford(f["total"], f) is True
        assert matching.can_afford(f["total"] - 0.01, f) is False


class TestPnl:
    def test_win_and_loss(self):
        assert matching.pnl_of(matching.sell_fill(12.0, 1000), 10.0) > 0
        assert matching.pnl_of(matching.sell_fill(8.0, 1000), 10.0) < 0

    def test_fees_are_deducted(self):
        f = matching.sell_fill(12.0, 1000)
        gross = (f["actual_price"] - 10.0) * 1000
        assert matching.pnl_of(f, 10.0) == pytest.approx(
            gross - f["commission"] - f["stamp_tax"], abs=0.02)


class TestSingleImplementation:
    """Phase 2 硬约束：只有一套撮合。回测与模拟盘都必须用 core/matching。"""

    def test_backtest_constants_are_matching_constants(self):
        from core import backtest
        assert backtest.SLIPPAGE_RATE == matching.SLIPPAGE_RATE
        assert backtest.COMMISSION_RATE == matching.COMMISSION_RATE
        assert backtest.STAMP_TAX_RATE == matching.STAMP_TAX_RATE
        assert backtest.MIN_COMMISSION == matching.MIN_COMMISSION

    def test_signal_generator_limit_logic_delegates_to_matching(self):
        """回测信号生成器的涨跌停口径必须来自 core/matching（模拟盘用同一套）"""
        from scripts.backtest_rule_based import RuleSignalGenerator
        for code in ("000001", "300750", "301536", "688981"):
            assert RuleSignalGenerator._limit_pct(code) == matching.limit_pct(code)

        up = {"open": 11.0, "close": 11.0, "prev_close": 10.0}
        assert (RuleSignalGenerator._blocked_by_price_limit("买入", "000001", up)
                == matching.blocked_by_price_limit("买入", "000001", 11.0, 10.0) != "")
        down = {"open": 9.0, "close": 9.0, "prev_close": 10.0}
        assert (RuleSignalGenerator._blocked_by_price_limit("卖出", "000001", down)
                == matching.blocked_by_price_limit("卖出", "000001", 9.0, 10.0) != "")
        normal = {"open": 10.3, "close": 10.3, "prev_close": 10.0}
        assert RuleSignalGenerator._blocked_by_price_limit("买入", "000001", normal) == ""

    def test_engine_fills_match_matching_module(self):
        engine = BacktestEngine()
        engine.inject_kline_cache({
            "000001": pd.DataFrame({"date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
                                    "close": [10.0, 12.0]}),
        })
        result = engine.run([
            {"date": "2026-01-05", "code": "000001", "direction": "买入",
             "price": 10.0, "quantity": 1000},
            {"date": "2026-01-06", "code": "000001", "direction": "卖出",
             "price": 12.0, "quantity": 1000},
        ])
        buy, sell = result.trade_history[0], result.trade_history[1]

        exp_buy = matching.buy_fill(10.0, 1000)
        assert buy["actual_price"] == exp_buy["actual_price"]
        assert buy["cost"] == exp_buy["cost"]
        assert buy["commission"] == exp_buy["commission"]
        assert buy["cash_after"] == pytest.approx(1_000_000 - exp_buy["total"], abs=0.01)

        exp_sell = matching.sell_fill(12.0, 1000)
        assert sell["actual_price"] == exp_sell["actual_price"]
        assert sell["revenue"] == exp_sell["revenue"]
        assert sell["commission"] == exp_sell["commission"]
        assert sell["stamp_tax"] == exp_sell["stamp_tax"]

        # 引擎的持仓均价含买入佣金，盈亏口径必须与此一致
        avg_cost = exp_buy["total"] / 1000
        assert sell["pnl"] == pytest.approx(matching.pnl_of(exp_sell, avg_cost), abs=0.02)
