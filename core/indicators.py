# -*- coding: utf-8 -*-
"""技术指标**唯一实现**（P1-1 / P0-2 地基，2026-10-18）。

为什么要抽这一层：
  20 日动量与年化波动率原来各写了 2~3 遍，且**单位不统一**：
    - `core/screener.py::_annualized_vol`  → **百分数**（×100）
    - `core/factor_analysis.py::_enrich`   → **小数**（不 ×100）
    - `data/similar_days.py`               → **百分数**（ret_20d/volat_20/ma_dev 都是 %）
  本项目已经因为"同列名不同单位"栽过三次（新浪金额亿元/元、换手率小数/百分数、
  turnover 与 turnoverratio）—— 这里是第四次的风险点。

铁律（写代码时照做）：
  1. **本模块内部一律输出小数**（0.05 = +5%，0.28 = 28% 波动率）；
  2. 需要百分数的调用方在**调用点**显式 `* 100`，函数名/变量名带 `_pct`；
  3. 输入统一是**收盘价序列**（`pd.Series` / `np.ndarray`），不认列名 ——
     `df["Close"]`（screener）与 `df["close"]`（factor/similar_days）的差异
     由调用方自己取列，避免"大小写不同 = 两套实现"再次发生；
  4. 长度不足 / 全 NaN ⇒ 返回 **None**（缺就是缺，绝不返回 0 —— 0 会被当成
     "真实的最差表现"参与排序与打分）。
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# 年化交易日数（A 股惯例 252）
TRADING_DAYS = 252


def _close_array(close) -> Optional[np.ndarray]:
    """归一成 float 数组；不可用 ⇒ None（不抛异常）。"""
    if close is None:
        return None
    try:
        if isinstance(close, pd.Series):
            arr = pd.to_numeric(close, errors="coerce").to_numpy(dtype="float64")
        elif isinstance(close, pd.DataFrame):
            return None                                    # 明确拒绝：调用方该自己取列
        else:
            arr = pd.to_numeric(pd.Series(list(close)), errors="coerce").to_numpy(
                dtype="float64")
    except (TypeError, ValueError) as e:
        # 契约：取不到就返回 None（不抛异常）——但要留痕，否则"指标恒 None"无从排查
        _logger.debug("[indicators] 收盘价序列不可用（%s: %s）", type(e).__name__, e)
        return None
    return arr if arr.size else None


def _tail(arr: np.ndarray, n: int) -> Optional[np.ndarray]:
    if arr is None or len(arr) < n:
        return None
    seg = arr[-n:]
    return seg if np.isfinite(seg).all() else None


# ================================================================
# 收益 / 动量
# ================================================================

def momentum(close, n: int = 20) -> Optional[float]:
    """N 日动量（小数）：`close[-1] / close[-1-n] - 1`。

    ⚠️ 需要 **n+1** 个点（含基准日）；不足 ⇒ None。
    """
    arr = _close_array(close)
    seg = _tail(arr, n + 1) if arr is not None else None
    if seg is None or seg[0] == 0:
        return None
    return float(seg[-1] / seg[0] - 1.0)


def returns_1d(close) -> Optional[np.ndarray]:
    """逐日简单收益（小数数组，首元素为 NaN）。长度不足 2 ⇒ None。"""
    arr = _close_array(close)
    if arr is None or len(arr) < 2:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(arr) / arr[:-1]
    return np.concatenate([[np.nan], r])


# ================================================================
# 波动率
# ================================================================

def annualized_vol(close, n: int = 20) -> Optional[float]:
    """N 日年化波动率（**小数**，0.28 = 28%）。

    口径：最近 n 个日收益的样本标准差 × √252（ddof=1，与 pandas `.std()` 一致）。
    ⚠️ 需要 **n+1** 个收盘价（n 个收益）；不足或含 NaN ⇒ None。
    """
    arr = _close_array(close)
    seg = _tail(arr, n + 1) if arr is not None else None
    if seg is None or len(seg) < 2:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(seg) / seg[:-1]
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return None
    sd = float(np.std(r, ddof=1))
    if not math.isfinite(sd):
        return None
    return sd * math.sqrt(TRADING_DAYS)


def residual_vol(close, market_close, n: int = 20) -> Optional[float]:
    """N 日**市场模型残差波动率**（IVOL，小数，0.28 = 28%）。

    为什么单独一个函数而不是调 `annualized_vol`
    ------------------------------------------
    总波动率扣的是**自身均值**，IVOL 还要再扣**市场因子**：`var(e) = var(r) - β²·var(r_m)`。
    两者算出的排序不一样 —— 实测（2016-2026 / 2,618 交易日 × 5,815 只 / 前瞻 5 日）：

        · ivol20  IC = **−0.0826**（t −29.6，11/11 年为负）
        · sd20    IC = −0.0684（11/11 年为负）      ← 本项目原口径
        · 对 log(成交额20) + 总波动率做正交后，ivol20 仍 **−0.0382**（11/11）

    ⇒ IVOL 强 21% 且不是「换了个名字的规模因子」。证据与分年表见
      `output/文献选股因子可实践性_2026-10-29.md`、`scripts/validate_literature_factors.py`。

    口径（与验证脚本 `rolling_stats()` **逐式一致**，改动必须两边同时改）
    --------------------------------------------------------------
    1. 两条序列按**位置**对齐（调用方负责日期对齐；停牌日两侧都应是 NaN）；
    2. 只取最后 `n` 个位置，有效点须 ≥ `ceil(0.8n)`（缺 4 天以上的不评）；
    3. `beta = cov(r_i, r_m) / var(r_m)`，方差用**总体矩**（除以 k，不是 k−1）；
    4. `ivar = var(r_i) - beta²·var(r_m)`（代数上等于 OLS 残差方差），负值截 0；
    5. 年化 `sqrt(ivar)·sqrt(252)`。

    ⚠️ `market_close` 传的是**市场指数点位**（等权全市场，`core.market_factor`），
       不是收益率 —— 这里两侧都走"收盘价→收益"，单位才不会一半对一半错。
    ⚠️ 有效点不足 / 市场方差 ≤0 / 长度不等 ⇒ **None**（缺就是缺，不返 0）。
    """
    a = _close_array(close)
    b = _close_array(market_close)
    if a is None or b is None or len(a) != len(b) or n < 2:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        rx = np.diff(a) / a[:-1]
        rm = np.diff(b) / b[:-1]
    if len(rx) < n:
        return None
    x, m = rx[-n:], rm[-n:]
    ok = np.isfinite(x) & np.isfinite(m)
    k = int(ok.sum())
    if k < max(2, int(math.ceil(n * 0.8))):
        return None
    x, m = x[ok], m[ok]
    kf = float(k)
    mx, mm = float(x.sum()) / kf, float(m.sum()) / kf
    var_x = float((x * x).sum()) / kf - mx * mx
    var_m = float((m * m).sum()) / kf - mm * mm
    if var_m <= 0:
        return None                                    # 市场零波动：β 不可识别
    cov = float((x * m).sum()) / kf - mx * mm
    ivar = var_x - (cov / var_m) ** 2 * var_m          # = var_x - cov²/var_m
    if not math.isfinite(ivar):
        return None
    return math.sqrt(max(ivar, 0.0)) * math.sqrt(TRADING_DAYS)


# ================================================================
# 均线
# ================================================================

def moving_average(close, n: int) -> Optional[float]:
    """最新 n 日均值。不足 n 个点 ⇒ None。"""
    seg = _tail(_close_array(close), n)
    return None if seg is None else float(np.mean(seg))


def ma_deviation(close, n: int) -> Optional[float]:
    """收盘价偏离 n 日均线的幅度（**小数**，正=在上方）。"""
    ma = moving_average(close, n)
    arr = _close_array(close)
    if ma is None or arr is None or ma == 0:
        return None
    return float(arr[-1] / ma - 1.0)


def ma_alignment(close) -> Optional[str]:
    """均线排列形态：`bull`(MA5>MA10>MA20) / `partial`(MA5>MA10) / `above20` / `below20`。

    与 `screener._score_ma_alignment`、`backtest_rule_based._score_stock_impl`
    的分支一致 —— 统一到这里，避免"两处 MA 计算不同导致档位不同"。
    """
    arr = _close_array(close)
    if arr is None or len(arr) < 20:
        return None
    ma5 = float(np.mean(arr[-5:]))
    ma10 = float(np.mean(arr[-10:]))
    ma20 = float(np.mean(arr[-20:]))
    cur = float(arr[-1])
    if ma5 > ma10 > ma20:
        return "bull"
    if ma5 > ma10:
        return "partial"
    return "above20" if cur > ma20 else "below20"


# ================================================================
# 位置类
# ================================================================

def distance_to_high(close, n: int = 20) -> Optional[float]:
    """距 n 日最高收盘的距离（**小数**，-0.05 = 比高点低 5%）。"""
    seg = _tail(_close_array(close), n)
    if seg is None or seg.max() == 0:
        return None
    return float(seg[-1] / seg.max() - 1.0)


def volume_ratio(volume, n: int = 20) -> Optional[float]:
    """量比：最新成交量 / 最近 n 日均量。分母 <=0 ⇒ None。"""
    arr = _close_array(volume)
    seg = _tail(arr, n) if arr is not None else None
    if seg is None:
        return None
    avg = float(np.mean(seg))
    return None if avg <= 0 else float(arr[-1] / avg)


__all__ = [
    "TRADING_DAYS", "momentum", "returns_1d", "annualized_vol", "residual_vol",
    "moving_average", "ma_deviation", "ma_alignment",
    "distance_to_high", "volume_ratio",
]
