"""择时信号检验的口径测试。

重点钉两条**踩过的坑**：

1. **无前视**：`signal_frame` 的标准化与分位必须只用当期及以前的数据。
   首版用了全样本 `mean()/std()` + `rank(pct=True)` —— 未来数据会改变历史仓位，
   回测漂亮但实盘做不出来。这类"技术指标择时"最经典的作弊点，必须有测试挡。
2. **规则语义**：`x.where(x > 0, 0.0)` 不是"最低档空仓"，而是"原样返回"
   （因为 x 本来就是 0）。首版三条规则里两条因此成了第一条的副本，
   实测三行数字一模一样才发现。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.timing_signal_check as T  # noqa: E402


def _toy_months(n: int = 60, seed: int = 7) -> pd.DataFrame:
    """合成月度信号帧（index=PeriodIndex，列与真实信号一致）。"""
    idx = pd.period_range("2015-01", periods=n, freq="M")
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "date": idx.to_timestamp(how="end"),
        "ma20_dev": rng.normal(scale=0.03, size=n),
        "macd_hist": rng.normal(scale=0.01, size=n),
        "mom20": rng.normal(scale=0.05, size=n),
        "vol20": np.abs(rng.normal(loc=0.2, scale=0.05, size=n)),
        "amt_ratio": rng.normal(scale=0.15, size=n),
        "fwd": rng.normal(loc=0.005, scale=0.05, size=n),
    }, index=idx)
    return d


# ---------------------------------------------------------------- 无前视
def test_signal_frame_has_no_lookahead():
    """追加未来数据，**历史**的 score / 仓位必须一字不变。

    这条是核心守卫：只要有人把 `expanding()` 改回 `mean()/std()` 或
    `rank(pct=True)`，未来的极端值就会改写过去的仓位，这个断言立刻红。
    """
    # ⚠️ 必须用足够长的样本：预热 = 24 期（扩容 z-score）+ 23 期（扩容分位累积）= 47 期。
    #    用 40~60 期会得到空表或十几期，用例会在"没数据"上假失败（踩过）。
    base = _toy_months(120)
    s1 = T.signal_frame(base)

    # 把后 6 个月换成极端值（模拟"未来行情剧烈变化"）
    bumped = base.copy()
    tail = bumped.index[-6:]
    bumped.loc[tail, "ma20_dev"] = 0.5
    bumped.loc[tail, "mom20"] = 0.8
    bumped.loc[tail, "vol20"] = 0.9
    s2 = T.signal_frame(bumped)

    # ⚠️ 必须把被改写的 tail 从比较里剔掉——否则比较的是"改过之后的那几期"，
    #    那不是前视，是我自己改的值（首次写这条用例时就在这里假失败）。
    common = s1.index.intersection(s2.index).difference(tail)
    assert len(common) >= 24
    assert np.allclose(s1.loc[common, "_score"], s2.loc[common, "_score"],
                       equal_nan=False), "历史 score 被未来数据改写了（前视）"
    assert (s1.loc[common, "_limit"] == s2.loc[common, "_limit"]).all()


def test_signal_frame_drops_warmup_window():
    """前 24 个月没有历史分布 ⇒ 必须剔除，不能给一个假的 40% 仓位。"""
    s = T.signal_frame(_toy_months(120))
    # 预热 = 24 期（扩容 z-score）+ 23 期（扩容分位累积）= 47 期（实测稳定值）
    assert len(s) == 120 - 47
    warm = _toy_months(120).index[:47]
    assert not any(i in s.index for i in warm)


def test_signal_frame_score_in_range():
    s = T.signal_frame(_toy_months(120))
    assert s["_score"].between(0, 100).all()
    assert set(s["_limit"].unique()) <= {0.0, 0.2, 0.4, 0.6, 0.8}


# ---------------------------------------------------------------- 档位映射
@pytest.mark.parametrize("score,expect", [
    (95, 0.8), (80, 0.8), (79.9, 0.6), (60, 0.6), (59.9, 0.4),
    (40, 0.4), (39.9, 0.2), (20, 0.2), (19.9, 0.0), (0, 0.0),
])
def test_band_limit_matches_detector(score, expect):
    """5 档映射必须与 `core.market_detector.detect()` 的阈值完全一致。"""
    assert T._band_limit(score) == expect


# ---------------------------------------------------------------- 规则语义
def test_overlay_rules_have_distinct_semantics():
    """三条规则必须是**三种不同**的仓位截面，且"最低档空仓"的均仓位最高。

    首版 `x.where(x > 0, 0.0)` 因为 x 本来就是 0 ⇒ 原样返回，与"全档映射"完全相同。
    这里用均仓位把它们区分开：全档映射 < 最低档半仓 < 最低档空仓 ≤ 1。
    """
    m = _toy_months(120)
    idx = pd.period_range("2015-01", periods=120, freq="M")
    rng = np.random.default_rng(11)
    strat = pd.Series(rng.normal(loc=0.01, scale=0.05, size=80), index=idx[:80])

    ov = T.overlay_on_strategy(m, strat)
    assert ov["ok"] is True
    r = ov["rules"]
    full = r["全档映射(仓位=上限)"]["avg_position"]
    half = r["最低档半仓_其余满仓"]["avg_position"]
    out = r["最低档空仓_其余满仓"]["avg_position"]

    # 顺序：全档映射（低档=0%）< 最低档空仓（低档 0%、其余 100%）
    #        < 最低档半仓（低档 50%、其余 100%）
    assert full < out < half, f"均仓位顺序应为 全档<空仓<半仓，实际 {full}/{out}/{half}"
    assert out <= 1.0
    # 全档映射不能与另外两条撞成同一个数（这正是当初的 bug）
    assert len({round(full, 4), round(half, 4), round(out, 4)}) == 3
    # "其余满仓"必然比"按上限"仓位高
    assert out > full and half > out


def test_overlay_requires_enough_overlap():
    m = _toy_months(80)
    idx = pd.period_range("2015-01", periods=10, freq="M")     # 只重叠 10 个月
    strat = pd.Series(0.01, index=idx)
    ov = T.overlay_on_strategy(m, strat)
    assert ov["ok"] is False and "重叠月份不足" in ov["reason"]


def test_overlay_full_mapping_never_exceeds_one():
    """仓位系数是"上限"，任何规则都不许超过 1（不许悄悄加杠杆）。"""
    m = _toy_months(120)
    idx = pd.period_range("2015-01", periods=120, freq="M")
    rng = np.random.default_rng(3)
    strat = pd.Series(rng.normal(loc=0.01, scale=0.05, size=80), index=idx[40:])
    ov = T.overlay_on_strategy(m, strat)
    for tag, mm in ov["rules"].items():
        assert 0 <= mm["avg_position"] <= 1.0, f"{tag} 仓位越界"
