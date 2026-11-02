# -*- coding: utf-8 -*-
"""`core/factor_composite.py` 的行为契约。

最容易错的两处：
1. **方向翻正**——负 IC 因子必须 `1-rank`，否则"分数越高越买"会买到反向；
2. **单因子缺失不丢整行**——丢了等于按数据完整度选股（新股/停牌期被系统性排除）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import factor_composite as fc


def _df(vals: list[float], month: str = "2024-01") -> pd.DataFrame:
    return pd.DataFrame({"code": [f"S{i}" for i in range(len(vals))],
                         "month": [month] * len(vals), "f": vals})


def test_positive_direction_rewards_high_value():
    d = _df([1.0, 2.0, 3.0, 4.0])
    s = fc.composite_score(d, ["f"], {"f": +1})
    assert s.iloc[0] < s.iloc[-1]                 # 低值低分、高值高分


def test_negative_direction_rewards_low_value():
    """方向 −1 ⇒ 低值拿高分（这是"IC 负"因子的正确用法）。"""
    d = _df([1.0, 2.0, 3.0, 4.0])
    s = fc.composite_score(d, ["f"], {"f": -1})
    assert s.iloc[0] > s.iloc[-1]


def test_rank_is_within_month_not_global():
    """两个月的水平完全不同，但各自的百分位应各自归一。"""
    d = pd.concat([_df([1.0, 2.0], "2024-01"), _df([100.0, 200.0], "2024-02")],
                  ignore_index=True)
    s = fc.composite_score(d, ["f"], {"f": +1})
    assert np.allclose(s.to_numpy(), [0.5, 1.0, 0.5, 1.0])


def test_missing_factor_does_not_drop_row():
    """单因子 NaN ⇒ 该行用**其余因子**的均值，不能变 NaN（否则整行落选）。"""
    d = pd.DataFrame({"code": ["A", "B", "C"], "month": ["2024-01"] * 3,
                      "f1": [1.0, 2.0, 3.0], "f2": [1.0, np.nan, 3.0]})
    s = fc.composite_score(d, ["f1", "f2"], {"f1": +1, "f2": +1})
    assert s.notna().all(), "有 NaN 说明单因子缺失把整行丢了"
    # B 行只有 f1=2.0 ⇒ 在 [1,2,3] 里百分位 2/3，均值就是它自己
    assert s.iloc[1] == pytest.approx(2.0 / 3.0)


def test_all_nan_row_stays_nan():
    d = pd.DataFrame({"code": ["A", "B"], "month": ["2024-01"] * 2,
                      "f1": [1.0, np.nan], "f2": [1.0, np.nan]})
    s = fc.composite_score(d, ["f1", "f2"], {"f1": +1, "f2": +1})
    assert s.notna().iloc[0] and pd.isna(s.iloc[1])


def test_no_usable_column_raises():
    d = pd.DataFrame({"code": ["A"], "month": ["2024-01"], "other": [1.0]})
    with pytest.raises(ValueError):
        fc.composite_score(d, ["f"])


def test_duplicate_index_does_not_explode():
    """截面表来自 concat，索引可能重复；用 Series.add 对齐会炸成笛卡尔积。"""
    d = pd.DataFrame({"code": ["A", "B", "C"], "month": ["2024-01"] * 3,
                      "f1": [1.0, 2.0, 3.0], "f2": [3.0, 2.0, 1.0]},
                     index=[0, 0, 1])
    s = fc.composite_score(d, ["f1", "f2"], {"f1": +1, "f2": -1})
    assert len(s) == 3


def test_promoted_factors_cover_six_with_signs():
    """晋身因子的方向必须与 `factor_analysis.FACTORS` 一致（防两处走样）。"""
    from core import factor_analysis as fa
    assert set(fc.PROMOTED_FACTORS) == {
        "lndvol20", "upshd20", "turn_std20", "amihud20", "mom1m", "pv_corr20"}
    for k, d in fc.PROMOTED_FACTORS.items():
        assert fa.FACTORS[k]["direction"] == d, f"{k} 方向与 factor_analysis 不一致"
