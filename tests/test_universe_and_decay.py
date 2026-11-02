# -*- coding: utf-8 -*-
"""`core/universe_guard.py` 与 `core/alpha_decay.py` 的行为契约。

两个模块都是为了防"看起来很正常的假结果"：
- 守卫防的是**伪数据/不可比宇宙**（2026-10-31：北交所新三板历史映射造出 +142% 月收益）；
- 衰减监控防的是**判据在边界上反转**（历史中位为负时 `ratio<0.5` 会恒真）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import alpha_decay as ad
from core import universe_guard as ug


# ── universe_guard ───────────────────────────────────────────

def _listing(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["code", "first_date", "last_date", "rows"])


def test_is_bse_prefixes():
    for c in ("920001", "830799", "871981", "430047", "820001"):
        assert ug.is_bse(c), c
    for c in ("600000", "000001", "300750", "688111"):
        assert not ug.is_bse(c), c


def test_min_history_boundary():
    """上市未满 min_history_days 的票不纳入；边界两侧必须分开。"""
    lst = _listing([["600000", pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-01"), 900]])
    d = pd.DataFrame({"code": ["600000", "600000"],
                      "month_ts": [pd.Timestamp("2020-12-30"), pd.Timestamp("2021-01-01")]})
    ok = ug.eligible(d, listing=lst, min_history_days=365)
    assert list(ok) == [False, True]


def test_bse_only_after_open():
    """北交所 2021-11-15 开市：之前不纳入，当天起纳入。"""
    lst = _listing([["920001", pd.Timestamp("2016-01-04"), pd.Timestamp("2024-01-01"), 900],
                    ["600000", pd.Timestamp("2016-01-04"), pd.Timestamp("2024-01-01"), 900]])
    d = pd.DataFrame({"code": ["920001", "920001", "600000", "600000"],
                      "month_ts": [pd.Timestamp("2021-11-14"), pd.Timestamp("2021-11-15"),
                                   pd.Timestamp("2021-11-14"), pd.Timestamp("2021-11-15")]})
    ok = ug.eligible(d, listing=lst, min_history_days=0)
    assert list(ok) == [False, True, True, True]


def test_unknown_code_is_excluded_not_passed():
    """清单里没有的票 ⇒ False（宁可少算，不拿来源不明的历史去排名）。"""
    lst = _listing([["600000", pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-01"), 900]])
    d = pd.DataFrame({"code": ["999999"], "month_ts": [pd.Timestamp("2023-01-01")]})
    assert not ug.eligible(d, listing=lst).iloc[0]


def test_guard_has_discriminance():
    """守卫必须有鉴别力：同一批数据，加/不加守卫结果不同（防恒真）。"""
    lst = _listing([["600000", pd.Timestamp("2016-01-04"), pd.Timestamp("2024-01-01"), 900],
                    ["920001", pd.Timestamp("2016-01-04"), pd.Timestamp("2024-01-01"), 900]])
    d = pd.DataFrame({"code": ["600000", "920001"],
                      "month_ts": [pd.Timestamp("2020-06-30"), pd.Timestamp("2020-06-30")]})
    with_bse = ug.eligible(d, listing=lst, min_history_days=0, exclude_bse=False)
    without = ug.eligible(d, listing=lst, min_history_days=0, exclude_bse=True)
    assert with_bse.all() and list(without) == [True, False]


# ── alpha_decay ──────────────────────────────────────────────

def _series(vals, start="2020-01") -> pd.Series:
    idx = pd.period_range(start, periods=len(vals), freq="M").to_timestamp("M")
    return pd.Series(np.asarray(vals, dtype=float), index=idx)


def test_short_history_is_unknown_not_ok():
    """历史不够 ⇒ unknown。**不能返 ok**（"没数据"不等于"没问题"）。"""
    rep = ad.decay_report(_series([0.01] * 10), _series([0.0] * 10))
    assert rep["level"] == ad.LEVEL_UNKNOWN and rep["reason"]


def test_recent_negative_excess_alerts():
    r = _series([0.02] * 40 + [-0.01] * 6)
    b = _series([0.0] * 46)
    rep = ad.decay_report(r, b, recent=6)
    assert rep["level"] == ad.LEVEL_ALERT, rep


def test_positive_recent_with_negative_median_is_ok():
    """回归哨兵（2026-10-31 踩到）：历史中位为负时 `latest/med` 是负数，
    `ratio < 0.5` 会恒真 ⇒ 把"比历史好"误判成"腰斩"。"""
    # 历史多数月份跑输（超额负），最近 6 个月明显跑赢
    r = _series([0.0] * 36 + [0.05] * 6)
    b = _series([0.02] * 42)
    rep = ad.decay_report(r, b, recent=6)
    assert rep["level"] == ad.LEVEL_OK, rep
    assert rep["ratio"] is None            # 中位 ≤0 时不报比值（避免误读）


def test_halved_but_positive_excess_watches():
    r = _series([0.05] * 36 + [0.005] * 6)
    b = _series([0.0] * 42)
    rep = ad.decay_report(r, b, recent=6)
    assert rep["level"] == ad.LEVEL_WATCH, rep


def test_bench_mismatch_is_unknown():
    """基准索引对不上 ⇒ unknown，不能当"零基准"算（否则超额=绝对收益）。"""
    rep = ad.decay_report(_series([0.01] * 30), _series([0.0] * 30, start="2030-01"))
    assert rep["level"] == ad.LEVEL_UNKNOWN


def test_factor_ic_decay_detects_sign_flip():
    ic = _series([-0.05] * 30 + [0.02] * 6)
    rep = ad.factor_ic_decay(ic, recent=6)
    assert rep["level"] == ad.LEVEL_ALERT, rep


def test_factor_ic_decay_stable_is_ok():
    ic = _series([-0.05] * 36)
    rep = ad.factor_ic_decay(ic, recent=6)
    assert rep["level"] == ad.LEVEL_OK, rep


@pytest.mark.parametrize("window", [3, 6, 12])
def test_rolling_excess_length(window):
    r, b = _series([0.01] * 30), _series([0.0] * 30)
    ex = ad.rolling_excess(r, b, window)
    assert ex.notna().sum() == 30 - window + 1
