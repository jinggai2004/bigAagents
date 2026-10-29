# -*- coding: utf-8 -*-
"""技术指标唯一实现 `core/indicators.py`（P1-1）的语义边界。

锁住三件事：
  1. **单位**：模块内一律**小数**（0.05 = +5%）；百分数只由调用方在调用点 ×100；
  2. **现有实现零漂移**：与 `screener._annualized_vol` / `_momentum20` 逐值一致
     （接线后选股打分不能因为"换了个实现"而变）；
  3. **缺就是缺**：长度不足 / 含 NaN ⇒ None，绝不返回 0（0 会被当成最差表现参与排序）。
"""
import numpy as np
import pandas as pd
import pytest

from core import indicators as I


def _series(vals):
    return pd.Series([float(v) for v in vals])


class TestMomentum:
    def test_exact_value(self):
        # 21 个点：首 10.0、末 12.1 ⇒ 12.1/10.0-1 = 0.21
        s = _series([10.0] * 20 + [12.1])
        assert I.momentum(s, 20) == pytest.approx(0.21, abs=1e-9)

    def test_needs_n_plus_one(self):
        assert I.momentum(_series([10.0] * 20), 20) is None, "20 个点算不出 20 日动量"
        assert I.momentum(_series([10.0] * 21), 20) == pytest.approx(0.0)

    def test_nan_returns_none_not_zero(self):
        v = [10.0] * 20 + [np.nan]
        assert I.momentum(_series(v), 20) is None

    def test_accepts_numpy_and_list(self):
        arr = np.array([10.0] * 20 + [11.0])
        assert I.momentum(arr, 20) == pytest.approx(0.1)
        assert I.momentum(list(arr), 20) == pytest.approx(0.1)

    def test_rejects_dataframe(self):
        """传错对象（DataFrame）应返回 None，而不是悄悄取一列。"""
        assert I.momentum(pd.DataFrame({"close": [10.0] * 21}), 20) is None


class TestVol:
    def test_matches_pandas_and_is_decimal(self):
        rng = np.random.default_rng(7)
        vals = 10 * np.cumprod(1 + rng.normal(0, 0.02, 40))
        s = _series(vals)
        expect = float(s.pct_change().dropna().tail(20).std() * np.sqrt(252))
        got = I.annualized_vol(s, 20)
        assert got == pytest.approx(expect, rel=1e-9)
        assert abs(got) < 3, "必须是小数（0.3 = 30%）；出现 30 说明忘了除 100"

    def test_constant_price_is_zero_vol(self):
        assert I.annualized_vol(_series([10.0] * 21), 20) == pytest.approx(0.0)

    def test_insufficient_returns_none(self):
        assert I.annualized_vol(_series([10.0] * 20), 20) is None


class TestMa:
    def test_ma_deviation_sign(self):
        # 最近 20 个点 = [10]*19 + [11]（滑窗含最新价）⇒ 均值 10.05
        s = _series([10.0] * 20 + [11.0])
        assert I.ma_deviation(s, 20) == pytest.approx(11.0 / np.mean([10.0] * 19 + [11.0]) - 1)

    def test_alignment(self):
        bull = _series([1, 2, 3, 4, 5] * 4 + list(np.linspace(6, 12, 5)))
        assert I.ma_alignment(_series(list(np.linspace(1, 100, 60)))) == "bull"
        assert I.ma_alignment(_series(list(np.linspace(100, 1, 60)))) == "below20"
        assert I.ma_alignment(_series([1.0] * 10)) is None

    def test_distance_to_high(self):
        s = _series([10.0, 12.0, 9.0] + [9.0] * 17 + [9.0])
        assert I.distance_to_high(s, 20) == pytest.approx(9.0 / 12.0 - 1)

    def test_volume_ratio(self):
        # 滑窗含最新量 ⇒ 均值 (19*100+200)/20 = 105
        v = _series([100.0] * 20 + [200.0])
        assert I.volume_ratio(v, 20) == pytest.approx(200.0 / np.mean([100.0] * 19 + [200.0]))


# ================================================================
# 与既有实现的一致性（接线不许改变数值）
# ================================================================

class TestParityWithScreener:
    """⭐⭐ 统一实现后，选股打分的数值必须与旧实现逐值一致。

    这是"重构不许改行为"的硬判据：新函数不是"另一套算法"，只是搬了位置。
    """

    def _df(self, n=60, seed=3):
        rng = np.random.default_rng(seed)
        closes = 10 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
        idx = pd.date_range("2026-01-05", periods=n, freq="B")
        return pd.DataFrame({"Close": closes}, index=idx)

    def test_vol_parity(self):
        from core.screener import StockScreener
        s = StockScreener()
        df = self._df()
        old = s._annualized_vol(df)
        new = I.annualized_vol(df["Close"], 20) * 100.0      # 旧实现是百分数
        assert new == pytest.approx(old, rel=1e-12)

    def test_momentum_parity(self):
        from core.screener import StockScreener
        s = StockScreener()
        df = self._df()
        old = s._momentum20(df)
        new = I.momentum(df["Close"], 20)
        assert new == pytest.approx(old, rel=1e-12)

    def test_panel_vectorized_matches_scalar(self):
        """⭐⭐ 面板级**向量化**实现与标量实现必须逐值相等。

        `factor_analysis._enrich` / `similar_days` 必须走向量化（全市场 ×N 年，
        逐点调标量函数会慢几百倍）——若两者口径漂移，同一只股票在
        "选股打分"与"因子 IC"里会是两个数，整套结论失效。
        """
        from core.condition_backtest import ConditionSignalGenerator
        from core.factor_analysis import _enrich
        df = self._df(n=80, seed=11)
        raw = pd.DataFrame({
            "date": df.index, "close": df["Close"].to_numpy(),
            "open": df["Close"].to_numpy(), "high": df["Close"].to_numpy(),
            "low": df["Close"].to_numpy(), "volume": 1e6,
        })
        enriched = _enrich(raw)
        # 最后一行（与标量函数同输入窗口）
        scalar_mom = I.momentum(enriched["close"], 20)
        scalar_vol = I.annualized_vol(enriched["close"], 20)
        assert enriched["momentum20"].iloc[-1] == pytest.approx(scalar_mom, rel=1e-12)
        assert enriched["volatility"].iloc[-1] == pytest.approx(scalar_vol, rel=1e-12)
