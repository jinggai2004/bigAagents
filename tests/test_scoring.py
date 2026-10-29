# -*- coding: utf-8 -*-
"""评分唯一实现 `core/scoring.py`（P0-2）的契约。

锁住四件事：

1. **回测口径逐值不变**：`score_fixed` 与旧 `_score_stock_impl`（本文件内的
   reference 复刻）在同一输入上**分数完全相同** —— 重构不许改历史回测结论；
2. **选股口径逐值不变**：`score_weighted` 与旧归一化段一致；
3. **因子值只有一处计算**：`compute_factors` 与 screener 的 `_momentum20` /
   `_annualized_vol` 逐值相等；
4. **阈值表唯一来源**：`scripts/backtest_rule_based.py` 里**不许**再出现
   `ma_score = 35` / `vol_score = 25` 这类内联分档（否则"单实现"名存实亡）。
"""
import io
import os
import re

import numpy as np
import pandas as pd
import pytest

from core import scoring as S

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _series(n=60, seed=5, drift=0.001, vol=0.02):
    rng = np.random.default_rng(seed)
    closes = 10 * np.cumprod(1 + rng.normal(drift, vol, n))
    return pd.Series(closes)


# ================================================================
# 1. 回测口径 parity（旧实现复刻）
# ================================================================

def _old_score_stock_impl(closes: np.ndarray, sentiment_score: float) -> float:
    """旧 `_score_stock_impl` 的逐行复刻（2026-10-18 重构前版本）。"""
    current = closes[-1]
    ma5 = np.mean(closes[-5:])
    ma10 = np.mean(closes[-10:])
    ma20 = np.mean(closes[-20:])
    ma20_prev = np.mean(closes[-40:-20]) if len(closes) >= 40 else ma20
    if ma5 > ma10 > ma20 and ma20 > ma20_prev:
        ma_score = 35
    elif ma5 > ma10 > ma20:
        ma_score = 28
    elif ma5 > ma10:
        ma_score = 20
    elif current > ma20:
        ma_score = 10
    else:
        ma_score = 0

    returns = pd.Series(closes).pct_change().dropna().tail(20)
    vol = returns.std() * np.sqrt(252) * 100
    if vol < 20:
        vol_score = 25
    elif vol < 30:
        vol_score = 20
    elif vol < 40:
        vol_score = 15
    elif vol < 55:
        vol_score = 10
    else:
        vol_score = 0

    ret_10d = (current - closes[-10]) / closes[-10] * 100
    momentum_score = 0
    if ret_10d > 5:
        momentum_score += 20
    elif ret_10d > 2:
        momentum_score += 15
    elif ret_10d > 0:
        momentum_score += 10
    elif ret_10d > -3:
        momentum_score += 5
    if current > ma20 and ma20 > ma20_prev:
        momentum_score += 10
    elif current > ma20:
        momentum_score += 5

    total = ma_score + vol_score + min(momentum_score, 30) + sentiment_score * 0.10
    return round(min(100, total), 1)


class TestFixedParityWithLegacy:
    @pytest.mark.parametrize("seed", range(12))
    @pytest.mark.parametrize("sentiment", [0.0, 37.5, 100.0])
    def test_matches_old_impl(self, seed, sentiment):
        """⭐⭐ 12 组随机行情 × 3 档情绪，分数必须**完全相等**。"""
        closes = _series(60, seed=seed).to_numpy()
        expect = _old_score_stock_impl(closes, sentiment)
        got = S.score_fixed(S.compute_factors(pd.Series(closes)), sentiment)
        assert got == expect, f"seed={seed} 情绪={sentiment}: {got} != {expect}"

    def test_rising_series_hits_full_marks(self):
        """单调上涨 ⇒ 均线多头 + 低波 + 强动量 ⇒ 高分（不是恒 0/恒 100）。"""
        closes = pd.Series(np.linspace(10, 20, 60))
        sc = S.score_fixed(S.compute_factors(closes), 0.0)
        assert sc >= 70, f"单边上涨只给 {sc} 分，阈值表可能写反了"

    def test_falling_series_is_low(self):
        """单边下跌 ⇒ 无均线/动量加分，只剩波动率分。

        ⚠️ **已知口径缺陷（旧实现固有，非本次重构引入）**：波动率因子只看
        "波动大小"不看方向 —— 线性下跌的波动率为 0 ⇒ 该因子反而满分 25。
        所以这里的上界是 25（不是 0）。改进（让波动率带方向惩罚）属 P0-2b，
        必须先把回测改到横截面口径，否则会与历史结论不可比。
        """
        closes = pd.Series(np.linspace(20, 10, 60))
        sc = S.score_fixed(S.compute_factors(closes), 0.0)
        assert sc <= S.FIXED_WEIGHTS["波动率"], \
            f"单边下跌给了 {sc} 分，超过波动率单项满分（方向分不该有）"

    def test_nan_in_window_does_not_fake_a_score(self):
        """⚠️ 已知口径缺陷（与旧实现一致，P0-2b 才修）：含 NaN 时回测按 0 计。"""
        closes = _series(60, seed=3).copy()
        closes.iloc[-5] = np.nan
        assert S.score_fixed(S.compute_factors(closes), 50.0) >= 0.0  # 不抛异常


# ================================================================
# 2. 选股口径 parity
# ================================================================

class TestWeightedParity:
    def _old(self, breakdown, weights, min_cov=0.7):
        available = weighted = 0.0
        for name, res in breakdown.items():
            w = weights.get(name, 0.0)
            res["weight"] = round(w, 4)
            res.setdefault("status", "ok")
            if res["status"] == "ok" and res.get("max", 0) > 0:
                available += w
                weighted += (res["score"] / res["max"]) * w * 100
        if available < min_cov:
            return None, round(available, 4)
        return round(weighted / available, 1), round(available, 4)

    def test_all_ok(self):
        bd = {"主力资金": {"score": 20, "max": 25},
              "PE估值": {"score": 12, "max": 15},
              "趋势": {"score": 10, "max": 15}}
        w = {"主力资金": 0.2353, "PE估值": 0.1765, "趋势": 0.1765}
        got = S.score_weighted({k: dict(v) for k, v in bd.items()}, w)
        exp_total, exp_cov = self._old({k: dict(v) for k, v in bd.items()}, w)
        assert got["total"] == exp_total and got["coverage"] == exp_cov

    def test_unknown_factor_excluded_from_denominator(self):
        bd = {"主力资金": {"score": 20, "max": 25},
              "趋势": {"score": None, "max": 15, "status": "unknown"}}
        w = {"主力资金": 0.5, "趋势": 0.5}
        got = S.score_weighted(bd, w)
        assert got["status"] == "insufficient_data", "有效权重 0.5 < 0.7 ⇒ 必须判数据不足"
        assert got["coverage"] == pytest.approx(0.5)

    def test_insufficient_threshold_is_the_single_source(self):
        """⭐ 阈值只有一处：`scoring.MIN_WEIGHT_COVERAGE`，screener **引用**它。

        D6（2026-10-18 审查）：原来 `screener.MIN_WEIGHT_COVERAGE = 0.7` 与
        `scoring.MIN_WEIGHT_COVERAGE = 0.70` 是两份字面量 —— 数值相同但形态错误
        （改一处忘另一处 = 静默漂移），正是 P0-2 要消灭的东西。
        """
        from core.screener import StockScreener
        assert StockScreener.MIN_WEIGHT_COVERAGE == S.MIN_WEIGHT_COVERAGE

    def test_no_second_literal_in_screener(self):
        """结构性守护：screener 里**不许**再出现 `MIN_WEIGHT_COVERAGE = 0.7` 字面量。"""
        src = io.open(os.path.join(_ROOT, "core", "screener.py"),
                      encoding="utf-8").read()
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith("#"))
        bad = re.findall(r"MIN_WEIGHT_COVERAGE\s*=\s*0?\.\d+", code)
        assert bad == [], f"screener 里仍有字面量副本：{bad}（应 import scoring 的）"


# ================================================================
# 3. 因子值单实现
# ================================================================

class TestFactorsSingleSource:
    def test_momentum20_matches_screener(self):
        from core.screener import StockScreener
        df = pd.DataFrame({"Close": _series(60, seed=9)})
        f = S.compute_factors(df["Close"])
        assert f["momentum20"] == pytest.approx(StockScreener._momentum20(df), rel=1e-12)

    def test_volatility_matches_screener(self):
        from core.screener import StockScreener
        df = pd.DataFrame({"Close": _series(60, seed=9)})
        f = S.compute_factors(df["Close"])
        # screener 返回百分数，compute_factors 返回小数
        assert f["volatility"] * 100 == pytest.approx(
            StockScreener._annualized_vol(df), rel=1e-12)

    def test_missing_data_returns_none_not_zero(self):
        f = S.compute_factors(pd.Series([10.0] * 5))
        assert f["momentum20"] is None and f["volatility"] is None
        assert f["n"] == 5


# ================================================================
# 4. 结构性守护：阈值表不许再内联回消费方
# ================================================================

class TestNoInlineTiers:
    def test_backtest_script_has_no_inline_score_tiers(self):
        """⭐⭐ "单实现"的结构性判据：回测脚本里不许出现内联分档赋值。

        只看 `_score_stock_impl` 所在文件的**赋值语句**（`ma_score = 35` 等），
        避免误伤注释/文档里的说明。
        """
        path = os.path.join(_ROOT, "scripts", "backtest_rule_based.py")
        src = io.open(path, encoding="utf-8").read()
        # 去掉注释行再匹配（注释里提到 35/25 是允许的）
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith("#"))
        bad = re.findall(r"\b(?:ma_score|vol_score|momentum_score)\s*=\s*\d+", code)
        assert bad == [], f"阈值仍在回测脚本内联：{bad}（应改 core/scoring.FIXED_TIERS）"

    def test_sentiment_scale_is_declared(self):
        assert S.SENTIMENT_SCALE == pytest.approx(0.10)
        assert sum(S.FIXED_WEIGHTS.values()) == 100, "4 因子满分合计必须 100"
