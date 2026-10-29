"""测试择时引擎 timing.py（mock 指数数据）"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from core.timing import TimingEngine, TimingSignal


# ===== 夹具 =====

def make_index_df(close_seq, prefix_date="2026-01-01", vol_mult=1.0, high_low=None):
    """构造模拟指数日线 DataFrame"""
    dates = pd.date_range(start=prefix_date, periods=len(close_seq), freq="D")
    closes = np.array(close_seq, dtype=float)
    if high_low is None:
        highs = closes * 1.02
        lows = closes * 0.98
    else:
        highs, lows = high_low
    volumes = (closes * 1000000 * vol_mult).astype(int)
    return pd.DataFrame({
        "date": dates,
        "open": closes * 0.99,
        "close": closes,
        "high": highs,
        "low": lows,
        "volume": volumes,
    })


@pytest.fixture
def engine():
    return TimingEngine()


@pytest.fixture
def bull_data():
    """模拟上涨趋势数据（MA20 > MA20_prev, MA60 > MA60_prev, 价格在上方）"""
    # 温和上涨 120 个交易日，从 3000 -> 3500
    t = np.linspace(0, 1, 120)
    closes = 3000 + 500 * t + np.random.RandomState(42).normal(0, 10, 120)
    return make_index_df(closes, "2026-01-01")


@pytest.fixture
def bear_data():
    """模拟下跌趋势"""
    t = np.linspace(0, 1, 120)
    closes = 3500 - 600 * t + np.random.RandomState(42).normal(0, 15, 120)
    return make_index_df(closes, "2026-01-01")


@pytest.fixture
def range_data():
    """模拟震荡行情"""
    np.random.seed(42)
    base = 3200
    noise = np.random.normal(0, 30, 120)
    closes = base + np.cumsum(noise * 0.3)  # 均值回复型
    closes = np.clip(closes, base - 150, base + 150)
    return make_index_df(closes, "2026-01-01")


# ===== TimingSignal 基础 =====

def test_timing_signal_defaults():
    s = TimingSignal(regime="bull", position_limit=0.8, total_score=70,
                     breakdown={"上证": 70}, summary="偏多")
    assert s.regime == "bull"
    assert s.position_limit == 0.8
    assert s.total_score == 70
    assert s.summary == "偏多"


def test_timing_signal_to_dict():
    s = TimingSignal(regime="range", position_limit=0.5, total_score=50,
                     breakdown={"上证": 50}, summary="震荡")
    d = s.to_dict()
    assert d["regime"] == "range"
    assert d["position_limit"] == 0.5
    assert d["total_score"] == 50


# ===== _score_trend =====

def test_trend_bull(engine, bull_data):
    """上涨趋势应得高评分"""
    score = engine._score_trend(bull_data)
    assert 60 <= score <= 100, f"上涨趋势评分应 >=60，实际 {score}"


def test_trend_bear(engine, bear_data):
    """下跌趋势应得低评分"""
    score = engine._score_trend(bear_data)
    assert score < 60, f"下跌趋势评分应 <60，实际 {score}"


def test_trend_range(engine, range_data):
    """震荡行情趋势评分在中间区间"""
    score = engine._score_trend(range_data)
    assert 30 <= score <= 80, f"震荡评分应在 30~80，实际 {score}"


def test_trend_short_data(engine):
    """不足60个交易日时返回中性评分 50"""
    df = make_index_df([3000, 3010, 3020, 3015, 3005], "2026-07-01")
    score = engine._score_trend(df)
    assert score == 50


# ===== _score_volume =====

def test_volume_high(engine, bull_data):
    """放量上涨应得高分"""
    score = engine._score_volume(bull_data)
    # 上涨趋势数据量能应不低
    assert score >= 20


def test_volume_no_volume_column(engine):
    """缺 volume 列返回中性 50"""
    df = pd.DataFrame({"close": [3000, 3010, 3020]})
    score = engine._score_volume(df)
    assert score == 50


def test_volume_short_data(engine):
    """不足20行返回中性 50"""
    df = make_index_df([3000, 3010, 3020], "2026-07-01")
    score = engine._score_volume(df)
    assert score == 50


# ===== _score_momentum =====

def test_momentum_bull(engine, bull_data):
    """上涨趋势动量评分较高"""
    score = engine._score_momentum(bull_data)
    assert score >= 40, f"上涨动量应 >=40，实际 {score}"


def test_momentum_bear(engine, bear_data):
    """下跌趋势动量评分较低"""
    score = engine._score_momentum(bear_data)
    assert score < 60, f"下跌动量应 <60，实际 {score}"


def test_momentum_short(engine):
    """不足20行返回中性 50"""
    df = make_index_df([3000, 3010], "2026-07-01")
    score = engine._score_momentum(df)
    assert score == 50


# ===== _score_breadth_fallback =====

def test_breadth_bull_5d(engine, bull_data):
    """近5日上涨应得高分"""
    score = engine._score_breadth_fallback(bull_data)
    assert score >= 40


def test_breadth_bear_5d(engine, bear_data):
    """近5日下跌应得低分"""
    score = engine._score_breadth_fallback(bear_data)
    assert score < 60


def test_breadth_short(engine):
    """不足5行返回中性 50"""
    df = make_index_df([3000, 3010, 3020], "2026-07-01")
    score = engine._score_breadth_fallback(df)
    assert score == 50


# ===== _fetch_index & detect（mock 网络） =====

@patch("core.timing.ak.stock_zh_index_daily")
def test_fetch_index_success(mock_ak, engine):
    """网络正常时获取指数数据"""
    mock_ak.return_value = pd.DataFrame({
        "date": pd.date_range("2026-07-01", periods=50, freq="D"),
        "open": np.random.uniform(3000, 3100, 50),
        "close": np.random.uniform(3000, 3100, 50),
        "high": np.random.uniform(3050, 3150, 50),
        "low": np.random.uniform(2950, 3050, 50),
        "volume": np.random.randint(100000, 500000, 50),
    })
    df = engine._fetch_index("sh000001")
    assert df is not None
    assert len(df) == 50


@patch("core.timing.ak.stock_zh_index_daily")
def test_fetch_index_api_fail(mock_ak, engine):
    """网络异常返回 None"""
    mock_ak.side_effect = Exception("API error")
    df = engine._fetch_index("sh000001")
    assert df is None


@patch("core.timing.ak.stock_zh_index_daily")
def test_detect_bear_market(mock_ak, engine):
    """用熊市数据验证 detect() 输出 bear regime"""
    # 模拟三大指数均下跌 120 日
    bear_closes = [3500.0 - 5.0 * i for i in range(120)]
    mock_ak.return_value = make_index_df(bear_closes, "2026-01-01")
    signal = engine.detect()
    assert isinstance(signal, TimingSignal)
    assert signal.regime == "bear"
    assert signal.position_limit < 0.5


@patch("core.timing.ak.stock_zh_index_daily")
def test_detect_all_fail(mock_ak, engine):
    """所有指数都无法获取时，返回中性兜底"""
    mock_ak.side_effect = Exception("Network down")
    signal = engine.detect()
    assert signal.regime == "range"
    assert signal.position_limit >= 0.3


# ===== 边界值 =====

def test_trend_macd_corner_cases(engine):
    """MACD 计算边缘情况: 接近零值"""
    closes = np.array([100.0] * 80)  # 完全平盘
    df = make_index_df(closes, "2026-01-01")
    score = engine._score_trend(df)
    # 平盘评分应在中低水平
    assert 10 <= score <= 70


def test_ema_static(engine):
    """静态方法 _ema 基础功能"""
    arr = np.array([10.0, 11.0, 12.0, 11.5, 10.5])
    result = engine._ema(arr, 3)
    # _ema 返回的是标量（序列的最后一个 EMA 值）
    assert isinstance(result, (np.floating, float))
    assert 10.0 <= float(result) <= 12.0


def test_volume_ratio_extremes(engine):
    """量比极值: 极度放量"""
    closes = [3000 + i * 2 for i in range(30)]
    volumes = [1000] * 25 + [50000] * 5  # 最后5天极度放量
    df = pd.DataFrame({
        "date": pd.date_range("2026-07-01", periods=30, freq="D"),
        "close": closes,
        "volume": volumes,
    })
    score = engine._score_volume(df)
    assert score >= 60, f"极度放量应得高分，实际 {score}"
