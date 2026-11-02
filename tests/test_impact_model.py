# -*- coding: utf-8 -*-
"""平方根冲击模型的接口测试。

这个模块是"容量测算"和"执行折扣"的**唯一换算入口**，所以守的是三条会静默出错的契约：

1. `impact_bp` 与 `capacity_for` **严格互逆**——两个方向各改一处，结论会错位而不报错；
2. 金额是**平方根**敏感（不是线性）——把 sqrt 写成线性，容量会差一个数量级，
   而输出仍然"看起来像个数"；
3. 非法输入给 **NaN 而不是 0**——0 会被读成"没有冲击"。
"""
import numpy as np
import pandas as pd
import pytest

from core import impact_model as im


def test_roundtrip_impact_then_capacity():
    """capacity_for(impact_bp(Q)) == Q —— 反过来解必须回到原金额。"""
    v = pd.Series([1e8, 5e7, 3e9])       # 日均成交额
    s = pd.Series([0.015, 0.03, 0.01])   # 日波动
    q = 200_000.0
    bp = im.impact_bp(v, s, q, y=1.0)
    back = im.capacity_for(v, s, bp.iloc[0], y=1.0)
    # bp 逐行不同 ⇒ 只能逐行验证
    for i in range(len(v)):
        got = im.capacity_for(pd.Series([v.iloc[i]]), pd.Series([s.iloc[i]]),
                              float(bp.iloc[i]), y=1.0).iloc[0]
        assert got == pytest.approx(q, rel=1e-9), i


def test_impact_is_square_root_in_notional():
    """金额 ×4 ⇒ 冲击 ×2（若有人把 sqrt 改成线性，这条会红）。"""
    v, s = 1e8, 0.02
    bp1 = float(im.impact_bp(v, s, 100_000).iloc[0])
    bp4 = float(im.impact_bp(v, s, 400_000).iloc[0])
    assert bp4 == pytest.approx(bp1 * 2.0, rel=1e-9)


def test_capacity_is_square_in_target():
    """目标冲击减半 ⇒ 容量降到 1/4（不是 1/2）。"""
    v, s = 1e8, 0.02
    c20 = float(im.capacity_for(v, s, 20.0).iloc[0])
    c10 = float(im.capacity_for(v, s, 10.0).iloc[0])
    assert c10 == pytest.approx(c20 / 4.0, rel=1e-9)
    # 且 20bp 的容量确实等于"按这个金额下单得到 20bp"
    assert float(im.impact_bp(v, s, c20).iloc[0]) == pytest.approx(20.0, rel=1e-9)


def test_series_notional_supported():
    """逐行金额：组合每月持仓数不同、分摊金额也不同，不能只吃标量。"""
    v = pd.Series([1e8, 1e8])
    s = pd.Series([0.02, 0.02])
    out = im.impact_bp(v, s, pd.Series([100_000.0, 400_000.0]))
    assert out.iloc[1] == pytest.approx(out.iloc[0] * 2.0, rel=1e-9)


def test_bad_input_is_nan_not_zero():
    """非法金额/零成交额给 NaN —— 返回 0 会被读成"零冲击"。"""
    v = pd.Series([1e8, 0.0])
    s = pd.Series([0.02, 0.02])
    out = im.impact_bp(v, s, 100_000.0)
    assert np.isnan(out.iloc[1]), "零成交额必须是 NaN"
    assert np.isnan(float(im.impact_bp(1e8, 0.02, 0.0).iloc[0])), "零金额必须是 NaN"
    assert np.isnan(float(im.impact_bp(1e8, 0.02, -5.0).iloc[0])), "负金额必须是 NaN"


def test_smaller_cap_buys_less():
    """容量随成交额单调：成交额越大，能买的越多。这条防"分子分母写反"。"""
    s = 0.02
    c_small = float(im.capacity_for(1e7, s, 20.0).iloc[0])
    c_big = float(im.capacity_for(1e9, s, 20.0).iloc[0])
    assert c_big > c_small * 50


def test_execution_discount_uses_same_model():
    """`execution_discount._impact_map` 必须与共享模型逐个相等。

    它原来自己写了一份公式（2026-10-31 抽出来），两处漂移的后果是
    "容量测算说 14bp、回测按 50bp 算"，而这种矛盾没人看得出来。
    """
    from scripts.execution_discount import _impact_map

    df = pd.DataFrame({
        "month_ts": pd.to_datetime(["2025-06-30", "2025-06-30"]),
        "code": ["600000", "000001"],
        "lndvol20": [np.log(1e8), np.log(2e8)],
        "vol20": [0.02, 0.015],
    })
    got = _impact_map(df, 200_000.0)
    want0 = float(im.impact_bp(1e8, 0.02, 200_000.0).iloc[0])
    want1 = float(im.impact_bp(2e8, 0.015, 200_000.0).iloc[0])
    assert got[(df["month_ts"].iloc[0], "600000")] == pytest.approx(want0, rel=1e-9)
    assert got[(df["month_ts"].iloc[1], "000001")] == pytest.approx(want1, rel=1e-9)
