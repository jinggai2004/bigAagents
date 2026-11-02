# -*- coding: utf-8 -*-
"""因子合成分数（direction 调整后的截面百分位等权均值）。

为什么单独成模块
────────────────
这个公式被**至少两处**用：`scripts/portfolio_ab_test.py`（A/B 实验）与
`scripts/execution_discount.py`（执行折扣）。各写一遍必然走样——项目已有教训
"页面与 prompt 说同一件事必须同源（共用同一个函数）"。

口径（**必须与 `core/screener` 的机制一致**，否则实验证明的不是线上行为）
──────────────────────────────────────────────────────────────────────
1. 先按 `direction` 把每个因子转成"越高越好"：`+1` 用原百分位，`−1` 用 `1−百分位`；
2. 百分位是**当月截面内**的 `rank(pct=True)`，不是全样本；
3. 多因子**等权**平均（真实权重里 3 个因子无历史数据，见 A/B 实验报告说明）。

⚠️ 负 IC 因子必须翻正后再合成
─────────────────────────────
2026-10-30 的教训：把单因子「IC 全负」直接推成"合成分数越高越不该买"是错的。
反向臂（买分数最低）开发段年化 −40.7%、t=−7.37 ⇒ 方向是对的。
"IC 负"说的是单因子，`1−rank` 之后它已经被翻正，两者不是一回事。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 已晋身 `core/factor_analysis.FACTORS` 的 6 个（方向逐字一致）
# −1 = 低值好（负 IC），+1 = 高值好
PROMOTED_FACTORS: dict[str, int] = {
    "lndvol20": -1,
    "upshd20": -1,
    "turn_std20": -1,
    "amihud20": +1,
    "mom1m": -1,
    "pv_corr20": -1,
}


def cross_sectional_rank(df: pd.DataFrame, col: str,
                         month_col: str = "month") -> pd.Series:
    """当月截面百分位（0~1），NaN 保持 NaN。"""
    v = pd.to_numeric(df[col], errors="coerce")
    return df.assign(_v=v).groupby(month_col)["_v"].transform(lambda s: s.rank(pct=True))


def composite_score(df: pd.DataFrame, cols: list[str],
                    directions: dict[str, int] | None = None,
                    month_col: str = "month") -> pd.Series:
    """direction 调整后的等权合成分数（0~1，越高越好）。

    Args:
        df: 截面长表（含 `month_col` 与各因子列）。
        cols: 参与合成的因子列名。
        directions: {因子: +1/−1}，缺省用 `PROMOTED_FACTORS`；不在表里的列按 +1。
        month_col: 截面列（月度分组）。

    Returns:
        Series（与 df 同索引）。**单因子缺失不剔除整行**，只让该因子这一格的
        百分位为 NaN，均值按该行实际可用的因子数算（`skipna`）——否则新股/
        数据不全的票会被整行丢掉，等于按数据完整度选股。
    """
    directions = directions or PROMOTED_FACTORS
    used = [c for c in cols if c in df.columns]
    if not used:
        raise ValueError(f"没有一列可用：{cols}")
    # ⚠️ 用 numpy 按行求和再除有效个数，而不是 `Series.add`：
    #    ① add 遇 NaN 会传染整行（与"单因子缺失不剔除整行"的语义相反）；
    #    ② 索引若重复（截面表来自 concat），Series 对齐会炸成笛卡尔积。
    parts = []
    for c in used:
        r = cross_sectional_rank(df, c, month_col).to_numpy(dtype="float64")
        parts.append(r if directions.get(c, +1) > 0 else (1.0 - r))
    arr = np.column_stack(parts)
    valid = ~np.isnan(arr)
    cnt = valid.sum(axis=1)
    total = np.where(valid, arr, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)
    return pd.Series(out, index=df.index)
