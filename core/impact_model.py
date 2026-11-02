# -*- coding: utf-8 -*-
"""平方根冲击模型：**冲击成本 ↔ 可买金额**的唯一换算入口。

    impact_bp = Y · σ_daily · sqrt(Q / V) · 1e4

- `Q` 我们的下单金额（元），`V` 标的**日均成交额**（元），`σ_daily` 日收益标准差（小数）；
- `Y` 市场相关经验系数。**本项目没有自己的实证校准**——文献与业界实践多落在
  0.5~1.5，所以对外给结论时必须带敏感性，不能只报一个数。

为什么抽成模块（2026-10-31）
────────────────────────────
`scripts/capacity_estimate.py` 用「反解」问"能买多少"，`scripts/execution_discount.py`
用「正解」问"买了要付多少 bp"。两处各写一遍公式必然漂移——而这一漂移**不会报错**，
只会让"容量测算说 14bp、回测用 50bp"这种自相矛盾长期存在。

⚠️ 模型的边界（引用结论时必须一起说）：
  1. 只估**冲击**，不含佣金/印花税/买卖价差（那部分在 `core/matching` 里单算）；
  2. 假设订单在**一个交易日内**完成；拆单到多日能放大容量，也放大暴露时间；
  3. `V` 用 20 日均额：大行情日会放大（利好执行），缩量日会缩小；
  4. 事件驱动的跳空不在此列（`σ` 是历史日波动）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Y_DEFAULT = 1.0
Y_RANGE = (0.5, 1.0, 1.5)


def impact_bp(amount_daily, sigma_daily, notional,
              y: float = Y_DEFAULT) -> pd.Series | np.ndarray:
    """正解：按金额下单，冲击多少 bp。

    Args:
        amount_daily: 日均成交额（元），标量或 Series。
        sigma_daily: **日**收益标准差（小数，不是年化）。
        notional: 单笔下单金额（元）。组合层面传"每只分摊到的金额"。
            **可以是标量，也可以是与入参等长的 Series**——组合里每月持仓数不同，
            分摊金额逐行不同，只吃标量会逼调用方自己写循环（那就是又一处公式副本）。

    Returns:
        与入参同形的 bp 值（10000 = 价格被推高 100%）。非法输入给 NaN，
        **不给 0**——0 会被读成"没有冲击"，那是截然不同的结论。
    """
    v = pd.to_numeric(pd.Series(amount_daily), errors="coerce")
    s = pd.to_numeric(pd.Series(sigma_daily), errors="coerce")
    if np.isscalar(notional):
        q = pd.Series(float(notional), index=v.index)
    else:
        q = pd.Series(np.asarray(notional, dtype=float), index=v.index)
    q = q.where(np.isfinite(q) & (q > 0))       # 非法金额 ⇒ NaN，不是 0
    out = y * s * np.sqrt(q / v.replace(0.0, np.nan)) * 1e4
    return out.replace([np.inf, -np.inf], np.nan)


def capacity_for(amount_daily, sigma_daily, target_bp: float,
                 y: float = Y_DEFAULT) -> pd.Series | np.ndarray:
    """反解：要控制在 `target_bp` 冲击以内，最多能买多少钱（元）。

    与 `impact_bp` 严格互逆：`capacity_for(impact_bp(Q)) == Q`。
    （这条互逆性有单测钉住——否则两个方向各改一处，结论会静默错位。）

    ⚠️ 结果对 `target_bp` 是**平方**敏感：目标冲击减半 ⇒ 容量降到 1/4。
       所以"5bp 还是 20bp"不是细节，是四倍容量差。
    """
    v = pd.to_numeric(pd.Series(amount_daily), errors="coerce")
    s = pd.to_numeric(pd.Series(sigma_daily), errors="coerce")
    bp = float(target_bp)
    if not np.isfinite(bp) or bp <= 0:
        return pd.Series(np.nan, index=v.index)
    scale = (bp / 1e4) / (y * s.replace(0.0, np.nan))
    out = v * scale ** 2
    return out.replace([np.inf, -np.inf], np.nan)
