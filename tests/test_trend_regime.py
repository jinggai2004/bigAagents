# -*- coding: utf-8 -*-
"""P3-15 A+B 灰度：趋势因子在弱市的两端修正（`core/screener.py::_score_trend`）。

钉死的边界
----------
1. **开关默认关**，且任何无法识别的环境变量值都回退为关 —— 线上行为必须逐字不变。
2. **强市完全不动**：补验里强市 `3~10%` 的证据自相冲突，不允许顺手改。
3. **弱市只改两端**：`ret20>=15%` 降档、`ret20<=-10%` 不再 0 分；中间档一分不动。
4. **缺数据不猜**：指数面板读不出 ⇒ regime=None ⇒ 按现状打分，绝不默认强市或弱市。
"""
from __future__ import annotations

import pandas as pd
import pytest

from core.screener import StockScreener
from core.market_regime import ma20_regime


def _kline_with_ret(ret20: float, bars: int = 30) -> pd.DataFrame:
    """构造 20 日累计涨幅为 ret20 的收盘价序列（与 tests/test_screener.py 同法）。"""
    start, end = 10.0, 10.0 * (1 + ret20)
    path = [start + (end - start) * i / 20 for i in range(21)]
    return pd.DataFrame({"Close": [10.0] * (bars - 21) + path})


@pytest.fixture
def scorer():
    return StockScreener.__new__(StockScreener)


@pytest.fixture
def regime_off(monkeypatch):
    monkeypatch.delenv("BIGA_TREND_REGIME", raising=False)


@pytest.fixture
def regime_on(monkeypatch):
    monkeypatch.setenv("BIGA_TREND_REGIME", "1")


# 现状分档（max_score=15）：任何"不该变"的场景都拿这张表逐条对
LEGACY = [
    (0.50, 3),     # >30%  过热
    (0.20, 11),    # >15%  偏强
    (0.05, 15),    # >3%   稳健上行
    (0.00, 7),     # >-3%  横盘
    (-0.05, 3),    # >-10% 偏弱
    (-0.15, 0),    # ≤-10% 下行
]


@pytest.mark.parametrize("ret20, expected", LEGACY)
def test_flag_off_keeps_legacy_scores(scorer, regime_off, ret20, expected):
    """开关关闭时，即使显式传入 weak，也必须与现状完全一致（回退开关的有效性）。"""
    out = scorer._score_trend(_kline_with_ret(ret20), market_state="weak")
    assert out["score"] == expected
    assert "弱市A+B" not in out["detail"]


@pytest.mark.parametrize("ret20, expected", LEGACY)
def test_strong_market_untouched_when_enabled(scorer, regime_on, ret20, expected):
    """强市完全不动 —— 包括 ret20≥15% 与 ≤-10% 两端。"""
    out = scorer._score_trend(_kline_with_ret(ret20), market_state="strong")
    assert out["score"] == expected


def test_weak_veto_downgrades_chasing(scorer, regime_on):
    """A：弱市 ret20=+20% 由 11 分（偏强）降到 3 分。"""
    out = scorer._score_trend(_kline_with_ret(0.20), market_state="weak")
    assert out["score"] == 15 // 4 == 3
    assert "弱市追高降档" in out["detail"]
    assert "弱市A+B" in out["detail"]


def test_weak_oversold_is_no_longer_zero(scorer, regime_on):
    """B：弱市 ret20=-15% 由 0 分改为 7 分（不再一票否决）。"""
    out = scorer._score_trend(_kline_with_ret(-0.15), market_state="weak")
    assert out["score"] == 15 // 2 == 7
    assert "弱市超跌" in out["detail"]


@pytest.mark.parametrize("ret20, expected", [(0.05, 15), (0.00, 7), (-0.05, 3)])
def test_weak_middle_bands_untouched(scorer, regime_on, ret20, expected):
    """A+B 只动两端：3~10%、横盘、偏弱中间档在弱市也保持原分。"""
    out = scorer._score_trend(_kline_with_ret(ret20), market_state="weak")
    assert out["score"] == expected


@pytest.mark.parametrize("bad", ["0", "garbage", "2", "off", ""])
def test_unrecognised_flag_values_fall_back_to_off(scorer, monkeypatch, bad):
    """坏配置退回默认：任何非 1/true/on/yes 的值都不得意外开启新行为。"""
    monkeypatch.setenv("BIGA_TREND_REGIME", bad)
    out = scorer._score_trend(_kline_with_ret(-0.15), market_state="weak")
    assert out["score"] == 0


@pytest.mark.parametrize("ret20, expected", LEGACY)
def test_missing_regime_data_keeps_legacy(scorer, regime_on, monkeypatch, ret20, expected):
    """开关开了但指数数据读不出 ⇒ regime=None ⇒ 一分不改（缺就是缺）。"""
    monkeypatch.setattr(StockScreener, "_weak_regime_info", staticmethod(lambda: None))
    out = scorer._score_trend(_kline_with_ret(ret20), market_state=None)
    assert out["score"] == expected


def test_regime_info_is_used_when_market_state_not_given(scorer, regime_on, monkeypatch):
    """未显式传 market_state 时走本地指数判定，并把指数数据日写进 detail 便于追溯。"""
    monkeypatch.setattr(StockScreener, "_weak_regime_info",
                        staticmethod(lambda: {"weak": True, "as_of": "2026-10-16"}))
    out = scorer._score_trend(_kline_with_ret(-0.15))
    assert out["score"] == 7
    assert "2026-10-16" in out["detail"]


# ===== core/market_regime.py：与补验脚本同口径 =====

def _write_index(path, closes):
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D").strftime("%Y-%m-%d")
    df = pd.DataFrame({"date": dates, "close": closes})
    df.to_parquet(path / "sh000852_test.parquet", index=False)


def test_regime_weak_and_strong(tmp_path):
    _write_index(tmp_path, list(range(100, 60, -1)))     # 一路下行 ⇒ 末值 < MA20
    assert ma20_regime(index_dir=str(tmp_path))["weak"] is True
    _write_index(tmp_path, list(range(60, 100)))         # 一路上行 ⇒ 末值 > MA20
    info = ma20_regime(index_dir=str(tmp_path))
    assert info["weak"] is False
    assert info["as_of"] == "2026-02-09"                 # 40 根日线的最后一根
    assert info["ma"] == pytest.approx(sum(range(80, 100)) / 20)


def test_regime_returns_none_when_data_insufficient(tmp_path):
    """不足 window 根 ⇒ None；目录不存在 ⇒ None。绝不默认强/弱。"""
    _write_index(tmp_path, [10.0] * 5)
    assert ma20_regime(index_dir=str(tmp_path)) is None
    assert ma20_regime(index_dir=str(tmp_path / "nope")) is None


def test_regime_ignores_nan_close(tmp_path):
    """末根 close 为 NaN 时，应取**最后一根有效收盘**对应的日期，不能错位。"""
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "close": [10.0, 9.0, None],
    })
    df.to_parquet(tmp_path / "sh000852_test.parquet", index=False)
    info = ma20_regime(window=2, index_dir=str(tmp_path))
    assert info["as_of"] == "2026-01-02"
    assert info["close"] == 9.0
    assert info["weak"] is True        # 9.0 ≤ mean(10.0, 9.0)=9.5
