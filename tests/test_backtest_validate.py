# -*- coding: utf-8 -*-
"""回测多重检验校正测试（backtest_validate）：DSR / PBO / 正态工具。

⚠️ 测试方法论（2026-10-29 评审修正）：
"白噪声 DSR 必须 p>0.05" 的**单次抽样断言天然有 5% 误红率**——随机种子换一个
就可能挂。校准类断言一律用 **m 个固定种子的"率"断言**（p<0.05 的比例 ≈5%），
固定种子 ⇒ 完全确定，无 CI 波动。PBO 的 ≈0.5 同理用区间断言 + 固定种子。

F1 回归：DSR 的 hurdle（SR*）里的 γ 是 **Euler–Mascheroni 常数**，不是收益偏度。
偏度误入 SR* 会把负偏（A 股常态）的"运气"判成"显著"——评审实测同一序列
p 从 0.46 翻成 0.0009。本文件把它锁死成数值回归。
"""
import math

import numpy as np
import pytest

from core.backtest_validate import (
    DSR_ALPHA, EULER_GAMMA, PBO_WARN,
    deflated_sharpe, expected_max_sharpe, norm_cdf, norm_ppf, pbo,
)

RNG = np.random.default_rng  # 统一走固定种子，测试完全确定


# ================================================================
# 正态工具
# ================================================================

class TestNormTools:

    def test_cdf_matches_erf_definition(self):
        assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
        assert norm_cdf(1.959964) == pytest.approx(0.975, abs=1e-4)
        assert norm_cdf(-1.959964) == pytest.approx(0.025, abs=1e-4)

    def test_ppf_roundtrip(self):
        for p in (0.025, 0.158655, 0.5, 0.841345, 0.975, 0.9583333, 0.9846718):
            assert norm_ppf(p) == pytest.approx(_bracket_ppf(p), abs=1e-6)

    def test_ppf_rejects_open_interval_violation(self):
        with pytest.raises(ValueError):
            norm_ppf(0.0)
        with pytest.raises(ValueError):
            norm_ppf(1.0)


def _bracket_ppf(p: float) -> float:
    """独立实现：二分法求逆 CDF（与 Acklam 互为印证，防"同一个错写两遍"）。"""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ================================================================
# F1 回归：SR* 用 Euler–Mascheroni 常数，与收益偏度无关
# ================================================================

class TestExpectedMaxSharpe:

    def test_f1_euler_value_pinned(self):
        """评审数值锚点（修正版）：N=24、trials=±0.1（样本方差 ddof=1）⇒ SR*≈0.2022。

        评审手算的 0.1980 与初版文档的 0.1976 都忘了 ddof=1（24/23·0.01）。
        这里用**独立路径**（bisection 逆 CDF + numpy 样本方差）重算期望值并与
        实现对齐，同时双向锁死 F1：偏度误入会算出 ≈0.13，落在锁死区间外。
        """
        trials = [0.1] * 12 + [-0.1] * 12
        expected = math.sqrt(float(np.var(trials, ddof=1))) * (
            (1 - EULER_GAMMA) * _bracket_ppf(1 - 1 / 24)
            + EULER_GAMMA * _bracket_ppf(1 - 1 / (24 * math.e)))
        assert expected == pytest.approx(0.20225, abs=5e-4)   # 独立路径自身合理
        sr_star = expected_max_sharpe(trials)
        assert sr_star == pytest.approx(expected, abs=1e-9)   # Acklam == bisection
        assert 0.19 < sr_star < 0.21
        assert abs(sr_star - 0.13) > 0.05, \
            "SR* 落到 0.13 附近说明 γ 被写成了偏度（F1 回归）"

    def test_monotonic_in_n_and_variance(self):
        t_small = expected_max_sharpe([0.05, -0.05] * 4)       # N=8, std=0.05
        t_big_n = expected_max_sharpe([0.05, -0.05] * 50)      # N=100, std=0.05
        t_big_v = expected_max_sharpe([0.1, -0.1] * 4)         # N=8, std=0.10
        assert 0 < t_small < t_big_n
        assert t_small < t_big_v

    def test_degenerate_inputs_return_zero(self):
        assert expected_max_sharpe([]) == 0.0
        assert expected_max_sharpe([0.2]) == 0.0               # 单试验无方差信息
        assert expected_max_sharpe([0.2, 0.2, 0.2]) == 0.0     # 零方差
        assert expected_max_sharpe([None, float("nan"), 0.1]) == 0.0


# ================================================================
# DSR
# ================================================================

def _noise_trials(seed: int, n_trials: int = 24, t_days: int = 500,
                  vol: float = 0.01):
    """生成 n_trials 条白噪声日收益 + 各自日频夏普。"""
    rng = RNG(seed)
    mat = rng.normal(0.0, vol, size=(t_days, n_trials))
    srs = [float(mat[:, c].mean() / mat[:, c].std(ddof=1)) for c in range(n_trials)]
    return mat, srs


class TestDeflatedSharpe:

    def test_rate_calibration_white_noise(self):
        """校准断言（率式，固定 300 个种子 ⇒ 确定）——两个位置分开测：

        ① 单策略 PSR（SR*=0）：SR_hat~N(0, denom/T) 成立 ⇒ p 值近似均匀
           ⇒ p<0.05 比例 ≈5%（评审的 5% 期望在这里才成立）；
        ② 选最优后的 DSR：极值统计量方差 < 单条估计方差 ⇒ **结构性保守**
           （实测 ≈0.3%，见模块 docstring），只断言无系统性假阳性（≤10%）。
           F1 式假阳性（率 30%+）依然会被这条拦住。
        """
        # ① 单策略 PSR 校准
        hits_psr = 0
        m = 300
        for seed in range(m):
            mat, _ = _noise_trials(seed)
            res = deflated_sharpe(mat[:, 0])          # 无 trials ⇒ SR*=0
            assert res["ok"] and res["hurdle_source"] == "none"
            if res["p_value"] < DSR_ALPHA:
                hits_psr += 1
        rate_psr = hits_psr / m
        assert 0.02 <= rate_psr <= 0.09, f"PSR 校准失真：{rate_psr:.3f}"

        # ② 选最优后的 DSR：只拦系统性假阳性
        hits_dsr = 0
        for seed in range(m):
            mat, srs = _noise_trials(seed)
            best = int(np.argmax(srs))
            res = deflated_sharpe(mat[:, best], sr_trials=srs)
            assert res["ok"] and res["hurdle_source"] == "trials"
            if res["p_value"] < DSR_ALPHA:
                hits_dsr += 1
        rate_dsr = hits_dsr / m
        assert rate_dsr <= 0.10, f"DSR 白噪声假阳性过高：{rate_dsr:.3f}"

    def test_drift_is_significant(self):
        """注入已知漂移 ⇒ 必须显著（反向断言：防"恒不显著"的假绿）。"""
        mat, srs = _noise_trials(seed=7)
        best = int(np.argmax(srs))
        drifted = mat[:, best] + 0.002               # 日漂移 0.2%（SR≈0.2）
        res = deflated_sharpe(drifted, sr_trials=srs)
        assert res["ok"] and res["significant"]
        assert res["p_value"] < 0.01
        assert res["hurdle_source"] == "trials"

    def test_f1_skew_never_touches_hurdle(self):
        """同 trials 下，偏度不同的两条序列 SR* 必须完全一致（偏度只在分母）。"""
        mat, srs = _noise_trials(seed=11)
        base = mat[:, int(np.argmax(srs))]
        skewed = base.copy()
        skewed[-1] -= 0.30                            # 制造重负偏（崩盘尾部）
        r1 = deflated_sharpe(base, sr_trials=srs)
        r2 = deflated_sharpe(skewed, sr_trials=srs)
        assert r1["skew"] != pytest.approx(r2["skew"], abs=0.01)   # 偏度确实变了
        assert r1["sr_star"] == r2["sr_star"]                        # 但 hurdle 不变

    def test_no_trials_degrades_to_psr(self):
        mat, _ = _noise_trials(seed=3)
        res = deflated_sharpe(mat[:, 0])              # 无试验信息
        assert res["ok"] and res["degraded"]
        assert res["hurdle_source"] == "none" and res["sr_star"] == 0.0

    def test_explicit_star_marks_explicit(self):
        mat, _ = _noise_trials(seed=5)
        res = deflated_sharpe(mat[:, 0], sr_star=0.1)
        assert res["hurdle_source"] == "explicit" and not res["degraded"]
        assert res["sr_star"] == pytest.approx(0.1, abs=1e-9)

    def test_insufficient_sample_rejected(self):
        res = deflated_sharpe([0.001] * 15)
        assert not res["ok"] and "样本不足" in res["reason"]

    def test_zero_variance_rejected(self):
        res = deflated_sharpe([0.0] * 100)
        assert not res["ok"] and "零方差" in res["reason"]


# ================================================================
# PBO / CSCV
# ================================================================

class TestPBO:

    def test_true_signal_low_pbo(self):
        """一条真信号 + 五条噪声 ⇒ IS 最优大概率是真信号 ⇒ OOS 排名靠前 ⇒ PBO 低。"""
        rng = RNG(42)
        t = 640
        mat = rng.normal(0.0, 0.01, size=(t, 6))
        mat[:, 0] += 0.0012                            # 真信号：日漂移
        res = pbo(mat, s_blocks=16)
        assert res["ok"] and res["n_combos"] == 12870
        assert res["pbo"] < 0.15, f"真信号 PBO 过高：{res['pbo']}"

    def test_noise_pbo_near_half(self):
        """全噪声 ⇒ IS 最优在 OOS 的排名近似均匀 ⇒ PBO≈0.5（区间断言 + 固定种子）。"""
        rng = RNG(43)
        mat = rng.normal(0.0, 0.01, size=(640, 6))
        res = pbo(mat, s_blocks=16)
        assert res["ok"]
        assert 0.30 <= res["pbo"] <= 0.70, f"噪声 PBO 失真：{res['pbo']}"

    def test_warn_threshold_is_conservative_choice(self):
        """PBO_WARN=0.3 是有意比文献(0.5)严的选择——锁住常量，防被当标准改掉。"""
        assert PBO_WARN == 0.3
        assert PBO_WARN < 0.5

    def test_needs_two_configs(self):
        res = pbo(np.random.default_rng(1).normal(size=(100, 1)))
        assert not res["ok"]

    def test_odd_blocks_coerced_even(self):
        rng = RNG(2)
        res = pbo(rng.normal(size=(800, 4)), s_blocks=15)
        assert res["ok"] and res["s_blocks"] == 16

    def test_too_few_days_rejected(self):
        res = pbo(np.random.default_rng(3).normal(size=(20, 4)), s_blocks=16)
        assert not res["ok"] and "切" in res["reason"]

    def test_few_configs_flagged_degraded(self):
        rng = RNG(4)
        res = pbo(rng.normal(size=(640, 5)), s_blocks=16)
        assert res["ok"] and res["degraded"]           # C=5 < 8 ⇒ 结论仅供参考
