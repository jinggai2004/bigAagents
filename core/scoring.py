# -*- coding: utf-8 -*-
"""评分**唯一实现**（P0-2，2026-10-18）。

## 为什么要这个模块

审计发现同一件事有 2~3 套实现，靠注释"刻意保持一致"维护：

| 位置 | 因子集 | 归一化方式 | 结果 |
|---|---|---|---|
| `core/screener.py::_score_single` | 6 因子（资金/PE/趋势/均线/波动/新闻） | **横截面分位** + 有效权重归一化 | 0~100 + coverage |
| `scripts/backtest_rule_based.py::_score_stock_impl` | 4 因子（均线/波动/动量/情绪） | **固定阈值分档**（35/25/30/10） | 0~100 |
| `core/condition_backtest.py::ConditionSignalGenerator` | 布尔条件（不是打分） | 全有全无 | True/False |

后果：**回测的"夏普/edge"不能迁移到选股结论上** —— 因为两者算的不是同一个东西。
文档只提过"风控/择时未接入回测"，从未提"评分不是同一套"。

## 本模块的分工（现状 + 迁移路径）

- `compute_factors(close, ...)` —— **因子原始值唯一计算处**。screener 与回测都调它，
  不再各写一份动量/波动率/均线（那些已归 `core/indicators`）。
- `score_weighted(breakdown, weights)` —— 选股口径（A）：加权 + 缺失因子剔除 + 有效权重归一化。
- `score_fixed(factors, sentiment)` —— 回测口径（B）：固定阈值分档，阈值表只在
  `FIXED_TIERS` 一处定义（原来 35/25/30/10 与十几条 `if` 内联在函数体里）。
- ⚠️ **A 与 B 不可互相替代，这是刻意的**：
  - A 问的是"在**当日候选**里排第几"（PE 最便宜的 20% 得满分）；
  - B 问的是"绝对值落在哪一档"（波动率 <20% 得满分）。
  - 迁移路径（P0-2b）：回测改为在 **T 日全市场截面**上调用 A（需要批量预计算
    截面因子，`data/bars.py` + 面板已具备条件，但属于下一批）。
  在迁移完成前，**禁止**用"回测的分数"论证"选股逻辑有效"。

## 口径差异清单（写死在常量里，供审计对照）

- 动量窗口：选股用 **20 日**（`momentum20`），回测用 **10 日**（`momentum10`）；
- 波动率：两者都是 20 日年化，但选股走横截面分位、回测走绝对分档；
- 情绪：选股用"新闻正面率"（需联网），回测用 ATR/涨跌停比（`_score_sentiment`，需 code/date）。
"""
from __future__ import annotations

from typing import Optional

from core import indicators as _ind

# ================================================================
# 回测口径（B）的**唯一**阈值表 —— 原来内联在 _score_stock_impl 里
# ================================================================

# 满分构成（合计 100）
FIXED_WEIGHTS = {"均线排列": 35, "波动率": 25, "趋势": 30, "情绪": 10}

# 均线排列档位：形态 → 得分
MA_TIERS = {
    "bull_rising": 35,   # MA5>MA10>MA20 且 MA20 上行
    "bull": 28,          # MA5>MA10>MA20
    "partial": 20,       # MA5>MA10
    "above20": 10,       # 收盘在 MA20 上方
    "below20": 0,
}

# 波动率档位（年化百分数上限 → 得分，越低越好）
VOL_TIERS = ((20.0, 25), (30.0, 20), (40.0, 15), (55.0, 10))
VOL_FLOOR_SCORE = 0

# 10 日动量档位（百分数下限 → 得分，从高到低匹配）
MOMENTUM_TIERS = ((5.0, 20), (2.0, 15), (0.0, 10), (-3.0, 5))
MOMENTUM_FLOOR_SCORE = 0
MOMENTUM_ABOVE_MA20_BONUS = {"rising": 10, "flat": 5, "below": 0}
MOMENTUM_CAP = 30

# 情绪分转换系数（`_score_sentiment` 返回 0~100，满分权重 10 ⇒ ×0.10）
SENTIMENT_SCALE = 0.10


# ================================================================
# 因子原始值（唯一计算处）
# ================================================================

def compute_factors(close, high=None, low=None) -> dict:
    """单票因子原始值 —— screener 与回测**都调这里**。

    Returns:
        {
          "momentum10": 小数 | None,     # 回测口径（10 日）
          "momentum20": 小数 | None,     # 选股口径（20 日）
          "volatility": 小数 | None,     # 20 日年化（0.28 = 28%）
          "ma_state": "bull_rising"|"bull"|"partial"|"above20"|"below20"|None,
          "above_ma20": bool | None,
          "ma20_rising": bool | None,    # MA20 是否高于前 20 日均值
          "n": 可用点数,
        }

    ⚠️ 缺数据一律 None（**不返回 0** —— 0 会被当成"最差表现"参与排序）。
    """
    out: dict = {"momentum10": None, "momentum20": None, "volatility": None,
                 "ma_state": None, "above_ma20": None, "ma20_rising": None,
                 "n": 0}
    arr = _ind._close_array(close)          # noqa: SLF001 —— 同包内复用归一化
    if arr is None:
        return out
    out["n"] = int(len(arr))
    # ⚠️ 回测口径的"10 日动量"是 `closes[-1]/closes[-10]-1`（**跨度 9 个交易日**，
    #    因为 `closes[-10]` 与 `closes[-1]` 之间只隔 9 个间隔）——不是标准 10 日。
    #    这里逐值复刻旧实现（`_score_stock_impl` 第 491 行），不做"顺手修正"：
    #    一旦改了，历史回测结论就与现在的不可比。
    if len(arr) >= 10 and arr[-10] != 0:
        out["momentum10"] = float(arr[-1] / arr[-10] - 1.0)
    out["momentum20"] = _ind.momentum(arr, 20)
    out["volatility"] = _ind.annualized_vol(arr, 20)
    out["ma_state"] = _ind.ma_alignment(arr)
    if len(arr) >= 20:
        import numpy as np
        ma20 = float(np.mean(arr[-20:]))
        out["above_ma20"] = bool(arr[-1] > ma20)
        if len(arr) >= 40:
            ma20_prev = float(np.mean(arr[-40:-20]))
            out["ma20_rising"] = bool(ma20 > ma20_prev)
        else:
            out["ma20_rising"] = None       # 不足 40 点无法判断"上行"
    return out


# ================================================================
# 口径 B：固定阈值打分（回测）
# ================================================================

def score_fixed(factors: dict, sentiment_score: float = 0.0) -> float:
    """回测口径评分（0~100）—— 4 因子固定阈值。

    Args:
        factors: `compute_factors()` 的返回。
        sentiment_score: `_score_sentiment()` 的原始分（0~100），×0.10 计入。

    Returns: 0~100（因子不可得的部分按 0 计 —— 与旧实现一致，**这是已知口径缺陷**：
        回测里"缺数据"与"表现最差"同分。要修需先让回测也用"有效权重归一化"，
        与 A 合并，属 P0-2b）。
    """
    if not factors or not factors.get("n") or factors["n"] < 20:
        return 0.0

    # ① 均线排列
    state = factors.get("ma_state") or "below20"
    if state == "bull" and factors.get("ma20_rising"):
        state = "bull_rising"
    ma_score = MA_TIERS.get(state, 0)

    # ② 波动率（绝对分档；volatility 是小数 ⇒ ×100）
    vol = factors.get("volatility")
    if vol is None:
        vol_score = VOL_FLOOR_SCORE
    else:
        vol_pct = vol * 100.0
        vol_score = VOL_FLOOR_SCORE
        for cap, sc in VOL_TIERS:
            if vol_pct < cap:
                vol_score = sc
                break

    # ③ 趋势/动量（10 日）
    mom = factors.get("momentum10")
    momentum_score = MOMENTUM_FLOOR_SCORE
    if mom is not None:
        mom_pct = mom * 100.0
        for floor, sc in MOMENTUM_TIERS:
            if mom_pct > floor:
                momentum_score = sc
                break
    if factors.get("above_ma20"):
        momentum_score += (MOMENTUM_ABOVE_MA20_BONUS["rising"]
                           if factors.get("ma20_rising")
                           else MOMENTUM_ABOVE_MA20_BONUS["flat"])
    momentum_score = min(momentum_score, MOMENTUM_CAP)

    # ④ 情绪
    sentiment_contrib = float(sentiment_score or 0.0) * SENTIMENT_SCALE

    return round(min(100.0, ma_score + vol_score + momentum_score + sentiment_contrib), 1)


# ================================================================
# 口径 A：加权 + 有效权重归一化（选股）
# ================================================================

MIN_WEIGHT_COVERAGE = 0.70


def score_weighted(breakdown: dict, weights: dict,
                   min_coverage: float = MIN_WEIGHT_COVERAGE) -> dict:
    """选股口径评分：`Σ(score/max·w) / Σ(w_有效) × 100`。

    Args:
        breakdown: `{因子名: {"score","max","status"}}`（status="unknown" 表不可评）。
        weights: `{因子名: 权重}`（唯一来源 `core/sector.FACTOR_WEIGHTS`）。
        min_coverage: 有效权重占比下限，低于此值判"数据不足"、不给分。

    Returns: `{"total": float|None, "coverage": float, "status": "ok"|"insufficient_data"}`

    ⚠️ 从 `screener._score_single` 抽出（行为逐值不变，有 parity 测试）。
    ⚠️ "不可评"的因子**同时**从分子与分母剔除（v2 语义），绝不给中位分。
    """
    available = 0.0
    weighted = 0.0
    for name, res in (breakdown or {}).items():
        w = float(weights.get(name, 0.0) or 0.0)
        res["weight"] = round(w, 4)
        res.setdefault("status", "ok")
        if res["status"] == "ok" and res.get("max", 0) > 0:
            available += w
            weighted += (res["score"] / res["max"]) * w * 100.0
    if available < min_coverage:
        return {"total": None, "coverage": round(available, 4),
                "status": "insufficient_data",
                "detail": (f"有效因子权重占比 {available:.0%}，低于阈值 "
                           f"{min_coverage:.0%}，不给评分")}
    return {"total": round(weighted / available, 1),
            "coverage": round(available, 4), "status": "ok"}


__all__ = [
    "FIXED_WEIGHTS", "MA_TIERS", "VOL_TIERS", "MOMENTUM_TIERS",
    "MOMENTUM_CAP", "SENTIMENT_SCALE", "MIN_WEIGHT_COVERAGE",
    "compute_factors", "score_fixed", "score_weighted",
]
