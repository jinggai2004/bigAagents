# -*- coding: utf-8 -*-
"""`core/factor_multiple_testing.py` 的行为契约。

守的是"校正真的在校正"这件事本身 —— 这个模块最容易出的错不是崩溃，
而是**退化成恒真或恒假**（比如 hurdle 恒为 0、或者方向没对齐导致最强因子被判死）。
所以每个正向用例都配一条对照/回归哨兵。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import factor_multiple_testing as fmt
from core.backtest_validate import DSR_ALPHA


def _series(values) -> pd.Series:
    idx = pd.period_range("2019-03", periods=len(values), freq="M")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def _noise(n: int = 90, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return _series(rng.normal(0.0, 0.05, n))


def _rows(names) -> list[dict]:
    return [{"name": n, "verdict": "有效", "t_stat": 3.0} for n in names]


# ── 1. 校准：纯噪音不应通过 ──────────────────────────────────

def test_pure_noise_does_not_pass():
    """59 个纯噪音因子 ⇒ **一个都不该显著**（否则校正就是恒真）。"""
    names = [f"n{i}" for i in range(59)]
    smap = {n: _noise(seed=i) for i, n in enumerate(names)}
    rows = _rows(names)
    out = fmt.adjust_by_multiple_testing(rows, smap)
    assert out["ok"] is True
    assert out["n_trials"] == 59
    passed = [r for r in rows if (r.get("dsr") or {}).get("significant")]
    assert passed == [], f"噪音里有 {len(passed)} 个被判显著"


# ── 2. 有真信号时必须能通过（防"宁可错杀"过头变成恒假）──────────

def test_strong_signal_passes():
    """一个持续稳定的因子 + 58 个噪音 ⇒ 它必须显著。"""
    names = [f"n{i}" for i in range(58)] + ["strong"]
    smap = {n: _noise(seed=i) for i, n in enumerate(names[:-1])}
    smap["strong"] = _series(np.full(90, 0.05))          # 零方差但均值稳定
    smap["strong"] = _series(0.05 + np.random.default_rng(99).normal(0, 0.01, 90))
    rows = _rows(names)
    fmt.adjust_by_multiple_testing(rows, smap)
    strong = next(r for r in rows if r["name"] == "strong")
    assert strong["dsr"]["significant"] is True, strong["dsr"]
    assert strong["dsr"]["p_value"] < DSR_ALPHA
    assert strong["verdict"] == "有效（过 DSR）"


# ── 3. 方向对齐：负 IC 因子不能被判死 ────────────────────────

def test_negative_series_is_direction_aligned():
    """持续**负**的多空收益 = "反过来做就赚" ⇒ 必须被判显著，方向记 −1。

    这是 2026-10-30 首版的真实 bug：不翻正时 SR_hat=−0.69，DSR p=1.0，
    全场最强的 lndvol20 被判成"无效"。
    """
    names = [f"n{i}" for i in range(58)] + ["neg"]
    smap = {n: _noise(seed=i) for i, n in enumerate(names[:-1])}
    smap["neg"] = _series(-0.05 + np.random.default_rng(7).normal(0, 0.01, 90))
    rows = _rows(names)
    fmt.adjust_by_multiple_testing(rows, smap)
    neg = next(r for r in rows if r["name"] == "neg")
    assert neg["ls_direction"] == -1
    assert neg["dsr"]["significant"] is True, neg["dsr"]
    # sr_monthly 是**翻正后**的（1.9+），不是 −1.9
    assert neg["sr_monthly"] > 0


def test_without_alignment_the_same_series_would_fail():
    """回归哨兵：同一条负序列**不翻正**送进 DSR ⇒ p≈1（证明对齐不是多余动作）。"""
    from core.backtest_validate import deflated_sharpe

    neg = _series(-0.05 + np.random.default_rng(7).normal(0, 0.01, 90))
    trials = [0.1, 0.2, 0.3, 0.4]
    raw = deflated_sharpe(neg.to_numpy(), sr_trials=trials)
    flipped = deflated_sharpe((-neg).to_numpy(), sr_trials=trials)
    assert raw["p_value"] > 0.9, raw
    assert flipped["p_value"] < DSR_ALPHA, flipped


# ── 4. hurdle 用 |SR|（否则负 SR 会把门槛拉低）─────────────────

def test_hurdle_uses_absolute_sharpe():
    """混入一个极强的**负** SR 因子时，SR* 应变高而不是被负值拉低。"""
    base = {"a": _noise(seed=1), "b": _noise(seed=2), "c": _noise(seed=3),
            "d": _noise(seed=4), "e": _noise(seed=5)}
    without = fmt.adjust_by_multiple_testing([],
                                             dict(zip(base, base.values())))["sr_star_monthly"]
    with_neg = fmt.adjust_by_multiple_testing(
        [], {**base,
             "neg": _series(-0.5 + np.random.default_rng(11).normal(0, 0.02, 90))}
    )["sr_star_monthly"]
    assert with_neg > without, (with_neg, without)


# ── 5. verdict 覆写 ──────────────────────────────────────────

def test_verdict_downgraded_when_not_significant():
    names = [f"n{i}" for i in range(59)]
    smap = {n: _noise(seed=i) for i, n in enumerate(names)}
    rows = _rows(names)                       # 全部预置 "有效"
    fmt.adjust_by_multiple_testing(rows, smap)
    assert all(r["verdict"] == "未过多重检验校正" for r in rows)


def test_rows_without_tstat_are_untouched():
    """'样本不足' 行没有 t_stat ⇒ 不该被塞 dsr（否则 JSON 里多出半截字段）。"""
    rows = [{"name": "x", "verdict": "样本不足", "n_months": 3}]
    fmt.adjust_by_multiple_testing(rows, {"x": _noise(seed=1)})
    assert "dsr" not in rows[0]


# ── 6. PBO 的边界必须显式退化，不能静默返回 ok ─────────────────

def test_pbo_degraded_when_too_few_factors():
    names = [f"n{i}" for i in range(5)]        # < PBO_MIN_FACTORS(8)
    smap = {n: _noise(seed=i) for i, n in enumerate(names)}
    out = fmt.adjust_by_multiple_testing(_rows(names), smap)
    assert out["pbo"]["ok"] is False
    assert "不足" in out["pbo"]["reason"]


def test_pbo_runs_with_enough_factors():
    names = [f"n{i}" for i in range(12)]
    smap = {n: _noise(seed=i) for i, n in enumerate(names)}
    out = fmt.adjust_by_multiple_testing(_rows(names), smap)
    p = out["pbo"]
    assert p["ok"] is True, p
    assert p["n_configs"] == 12 and p["months_used"] == 90
    assert 0.0 <= p["pbo"] <= 1.0


# ── 7. 空输入不崩 ────────────────────────────────────────────

@pytest.mark.parametrize("smap", [{}, {"short": _series([0.01] * 5)}])
def test_empty_or_short_input_returns_not_ok(smap):
    out = fmt.adjust_by_multiple_testing([], smap)
    assert out["ok"] is False and out["reason"]
