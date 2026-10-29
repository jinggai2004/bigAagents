"""测试风控引擎 risk.py"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from core.risk import RiskEngine, RiskReport, PositionPlan
from core.timing import TimingSignal


# ===== 夹具 =====

@pytest.fixture
def engine():
    """构造 mock 了数据源的 RiskEngine"""
    e = RiskEngine()
    mock_fetcher = MagicMock()
    mock_fetcher.get_stock_info.return_value = {"current_price": 25.0}
    mock_fetcher.get_stock_data.return_value = pd.DataFrame({
        "High": [25.5, 25.8, 26.0, 25.3, 25.7] * 6,
        "Low":  [24.5, 24.8, 25.0, 24.3, 24.7] * 6,
        "Close": [25.0, 25.5, 25.3, 25.0, 25.5] * 6,
    })
    e._stock_fetcher = mock_fetcher
    return e


@pytest.fixture
def bull_timing():
    return TimingSignal(
        regime="bull", position_limit=0.8, total_score=70,
        breakdown={"上证": 70}, summary="偏多",
    )


@pytest.fixture
def bear_timing():
    return TimingSignal(
        regime="bear", position_limit=0.25, total_score=25,
        breakdown={"上证": 25}, summary="偏空",
    )


@pytest.fixture
def range_timing():
    return TimingSignal(
        regime="range", position_limit=0.5, total_score=50,
        breakdown={"上证": 50}, summary="震荡",
    )


@pytest.fixture
def sample_candidates():
    return [
        {"股票代码": "000001", "score": 85, "sector": "金融", "股票名称": "平安银行"},
        {"股票代码": "600519", "score": 80, "sector": "消费", "股票名称": "贵州茅台"},
        {"股票代码": "300750", "score": 75, "sector": "新能源", "股票名称": "宁德时代"},
        {"股票代码": "000858", "score": 70, "sector": "消费", "股票名称": "五粮液"},
        {"股票代码": "300124", "score": 65, "sector": "科技", "股票名称": "汇川技术"},
    ]


# ===== PositionPlan / RiskReport =====

def test_position_plan_basic():
    p = PositionPlan("000001", 0.2, 22.0, 28.0, 25.0, "评分85", 15)
    assert p.code == "000001"
    assert p.weight == 0.2
    assert p.stop_loss == 22.0


def test_risk_report_empty():
    r = RiskReport(overall_risk="")
    assert r.overall_risk == ""
    assert len(r.position_plans) == 0
    assert len(r.warnings) == 0


def test_risk_report_default():
    """RiskReport 默认构造"""
    r = RiskReport()
    assert r.overall_risk == "medium"
    assert r.max_portfolio_drawdown == 0.0


def test_risk_report_to_dict(engine):
    r = RiskReport(
        overall_risk="medium",
        position_plans=[PositionPlan("000001", 0.25, 22.0, 28.0, 25.0)],
        warnings=["测试警告"],
    )
    d = r.to_dict()
    assert d["overall_risk"] == "medium"
    assert len(d["position_plans"]) == 1
    assert d["position_plans"][0]["code"] == "000001"
    assert d["warnings"] == ["测试警告"]


# ===== _kelly_position =====

def test_kelly_positive():
    """高胜率高盈亏比 -> 正仓位"""
    f = RiskEngine._kelly_position(0.6, 0.05, 0.03)
    assert 0 < f <= 0.25


def test_kelly_low_winrate():
    """低胜率 -> 零或极小仓位"""
    f = RiskEngine._kelly_position(0.2, 0.05, 0.03)
    assert f == 0.0


def test_kelly_zero_loss():
    """零亏损幅度 -> 默认 0.15"""
    f = RiskEngine._kelly_position(0.5, 0.05, 0.0)
    assert f == 0.15


def test_kelly_capped():
    """凯利公式上限 0.25"""
    f = RiskEngine._kelly_position(0.9, 0.2, 0.01)
    assert f == 0.25


# ===== _calc_atr =====

def test_calc_atr(engine):
    df = pd.DataFrame({
        "High":  [25.5, 25.8, 26.0, 25.3, 25.7] * 6,
        "Low":   [24.5, 24.8, 25.0, 24.3, 24.7] * 6,
        "Close": [25.0, 25.5, 25.3, 25.0, 25.5] * 6,
    })
    atr = RiskEngine._calc_atr(df)
    assert atr > 0


def test_calc_atr_empty(engine):
    df = pd.DataFrame({"High": [], "Low": [], "Close": []})
    atr = RiskEngine._calc_atr(df)
    assert np.isnan(atr) or atr == 0.0


def test_calc_atr_short(engine):
    """短数据也能正常返回"""
    df = pd.DataFrame({
        "High":  [25.5, 25.8, 26.0, 25.3, 25.7],
        "Low":   [24.5, 24.8, 25.0, 24.3, 24.7],
        "Close": [25.0, 25.5, 25.3, 25.0, 25.5],
    })
    atr = RiskEngine._calc_atr(df)
    assert atr > 0


# ===== calc_trailing_stop =====

def test_trailing_stop_basic():
    """未触发追踪条件 -> 保持基础止损"""
    stop = RiskEngine.calc_trailing_stop(25.0, 25.5, 1.0, 2.5)
    assert stop == 22.5  # 25 - 1.0 * 2.5


def test_trailing_stop_2x_atr():
    """浮盈 > 2x ATR -> 止损上移到成本价"""
    stop = RiskEngine.calc_trailing_stop(25.0, 27.5, 1.0, 2.5)
    assert stop == 25.0


def test_trailing_stop_3x_atr():
    """浮盈 > 3x ATR -> 止损上移到 entry + 1x ATR"""
    stop = RiskEngine.calc_trailing_stop(25.0, 29.0, 1.0, 2.5)
    assert stop == 26.0  # 25 + 1.0 * 1.0


# ===== assess() 熊市熔断 =====

def test_assess_bear_meltdown(engine, bear_timing, sample_candidates):
    """熊市 -> 仓位限制 30%，风险评级 high"""
    report = engine.assess(sample_candidates, bear_timing)
    assert report.overall_risk == "high"
    assert any("熊市" in w or "30%" in w for w in report.warnings)


# ===== assess() 正常风控 =====

def test_assess_bull_normal(engine, bull_timing, sample_candidates):
    """牛市 -> 等权分配，风险评级 low"""
    report = engine.assess(sample_candidates, bull_timing)
    assert report.overall_risk == "low"
    assert len(report.position_plans) > 0
    # 等权验证：总仓位 / 股票数
    expected_weight = round(bull_timing.position_limit / len(report.position_plans), 4)
    for p in report.position_plans:
        assert p.weight == expected_weight


def test_assess_range_normal(engine, range_timing, sample_candidates):
    """震荡市 -> 风险评级 medium"""
    report = engine.assess(sample_candidates, range_timing)
    assert report.overall_risk == "medium"
    assert len(report.position_plans) > 0


# ===== 边界值 =====

def test_assess_empty_candidates(engine, bull_timing):
    """空的候选列表"""
    report = engine.assess([], bull_timing)
    assert report.overall_risk == "high"
    assert "无有效候选" in "".join(report.warnings)


def test_assess_price_zero(engine, bull_timing):
    """候选股价格为 0 或负"""
    with patch.object(engine, "_get_current_price", return_value=0.0):
        report = engine.assess(
            [{"股票代码": "000001", "score": 80, "sector": "default"}],
            bull_timing,
        )
    assert report.overall_risk == "high"


def test_assess_single_stock_cap(engine, bull_timing):
    """单票超过 25% 上限应被截断"""
    # position_limit=0.8, 只有1只股票, equal_weight=0.8 > 0.25
    candidates = [{"股票代码": "000001", "score": 85, "sector": "金融", "股票名称": "平安"}]
    report = engine.assess(candidates, bull_timing)
    assert len(report.position_plans) == 1
    assert report.position_plans[0].weight == 0.25  # 截断到上限
    assert any("超过上限" in w for w in report.warnings)


def test_assess_position_limit_very_low(engine):
    """择时仓位低于 0.1 -> 拒绝"""
    empty_timing = TimingSignal(regime="bear", position_limit=0.08, total_score=10,
                                breakdown={}, summary="")
    report = engine.assess(
        [{"股票代码": "000001", "score": 80, "sector": "default"}],
        empty_timing,
    )
    assert report.overall_risk == "high"
    assert any("空仓" in w for w in report.warnings)
