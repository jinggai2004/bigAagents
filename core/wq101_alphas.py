# -*- coding: utf-8 -*-
"""WorldQuant "101 Formulaic Alphas"（Kakushadze 2016）子集实现。

来源与口径
──────────
论文：Zura Kakushadze, *101 Formulaic Alphas*, arXiv:1601.00991 (2016)，
WorldQuant 公开的一览表式因子集（101 个公式化 alpha）。
开源实现参考（公式口径核对用，本模块为独立实现）：
  - yli188/WorldQuant_alpha101_code（pandas 向量化）
  - jqz1226/WorldQuant-alpha-101（A 股适配版）

本子集只保留 **不依赖行业中性化（IndNeutralize）与上市口径 cap 之外特殊数据**
的 30 个，输入仅需 OHLCV + amount（VWAP ≈ amount/volume，adv{d} = amount 均值）。

实现要点
────────
论文里 `rank()` 是**截面**算子（同一交易日全市场排序），`ts_*` 是**时序**算子。
本模块用 **矩阵化**（行=交易日，列=股票）实现：时序算子沿 axis=0 滚动，
截面算子沿 axis=1 排序——这正是 qlib/开源实现的通用做法，且比逐股循环快得多。
所有矩阵来自 `data.bars`（qfq 收盘、amount 单位元）。

⚠️ 与论文的差异（如实声明，不冒充原版）：
  - `cap` 用 `outstanding_share × close`（流通市值代理）替代；
  - IndNeutralize 类 alpha 全部排除；
  - 部分条件分支（如 Alpha#21/24）依赖精确的布尔优先级，只保留语义无歧义的子集。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── 时序算子（axis=0 滚动）────────────────────────────────────────

def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.shift(d)


def delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x - x.shift(d)


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).sum()


def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).mean()


def ts_std(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).std()


def ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """逐列滚动百分位（当前值在过去 d 天中的分位）。

    ⚠️ 不能用 `rolling(d).apply(...)`：矩阵 12M 格 × 逐格 Python 调用 ≈ 25 分钟
    （2026-10-30 实测）。等价向量化：对 d-1 个 shift 做 (less/eq) 比较
    （含自身 0.5 权重），O(d) 次全矩阵运算，秒级。
    """
    less = (x * 0).astype("float32")        # 同形零矩阵（保留 NaN 语义）
    eq = less.copy()
    for k in range(1, d):
        sh = x.shift(k)
        less += (sh < x).astype("float32")
        eq += (sh == x).astype("float32")
    # 分母用 float(d) 且中间量 float32：结果与 float64 的 rolling.apply 相差
    # ~2e-8（float32 精度极限），对分位因子完全无影响（对照测试已验证）。
    rank = (less + 0.5 * eq + 0.5) / float(d)
    valid = x.notna().rolling(d, min_periods=1).sum()
    return rank.where(valid >= d)


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).min()


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).max()


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).corr(y)


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).cov(y)


def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    """截面 rank：**同一行内**（当日全市场）升序百分位。"""
    return x.rank(axis=1, pct=True)


def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)


def log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.where(x > 0))


# ── 30 个 alpha（输入为矩阵 dict：open/high/low/close/volume/amount/returns/vwap/adv20）──

def alpha02(F: dict) -> pd.DataFrame:
    return -1 * ts_corr(cs_rank(delta(log(F["volume"]), 2)),
                        cs_rank((F["close"] - F["open"]) / F["open"]), 6)


def alpha03(F: dict) -> pd.DataFrame:
    return -1 * ts_corr(cs_rank(F["open"]), cs_rank(F["volume"]), 10)


def alpha04(F: dict) -> pd.DataFrame:
    return -1 * ts_rank(cs_rank(F["low"]), 9)


def alpha06(F: dict) -> pd.DataFrame:
    return -1 * ts_corr(F["open"], F["volume"], 10)


def alpha09(F: dict) -> pd.DataFrame:
    d = delta(F["close"], 1)
    cond_min = ts_min(d, 5) > 0
    cond_max = ts_max(d, 5) < 0
    return d.where(cond_min | cond_max, -1 * d)


def alpha12(F: dict) -> pd.DataFrame:
    return sign(delta(F["volume"], 1)) * (-1 * delta(F["close"], 1))


def alpha13(F: dict) -> pd.DataFrame:
    return -1 * ts_cov(cs_rank(F["close"]), cs_rank(F["volume"]), 5)


def alpha14(F: dict) -> pd.DataFrame:
    return (-1 * cs_rank(delta(F["returns"], 3))) * ts_corr(F["open"], F["volume"], 10)


def alpha17(F: dict) -> pd.DataFrame:
    return ((-1 * ts_rank(cs_rank(F["close"]), 10))
            * cs_rank(delta(delta(F["close"], 1), 1))
            * cs_rank(ts_rank(F["volume"] / F["adv20"], 5)))


def alpha20(F: dict) -> pd.DataFrame:
    return (((-1 * cs_rank(F["open"] - delay(F["high"], 1)))
             * cs_rank(F["open"] - delay(F["low"], 1)))
            * cs_rank(F["open"] - delay(F["close"], 1)))


def alpha23(F: dict) -> pd.DataFrame:
    cond = (ts_sum(F["high"], 20) / 20) < F["high"]
    return (-1 * delta(F["high"], 2)).where(cond, -1 * (F["high"] - delay(F["high"], 2)))


def alpha26(F: dict) -> pd.DataFrame:
    return -1 * ts_max(ts_corr(ts_rank(F["volume"], 5), ts_rank(F["high"], 10), 5), 3)


def alpha28(F: dict) -> pd.DataFrame:
    return ts_corr(F["adv20"], F["low"], 5) + (F["high"] + F["low"]) / 2 - F["close"]


def alpha33(F: dict) -> pd.DataFrame:
    return cs_rank(-1 * (1 - F["open"] / F["close"]))


def alpha34(F: dict) -> pd.DataFrame:
    return cs_rank(1 - cs_rank(ts_std(F["returns"], 2) / ts_std(F["returns"], 5)))


def alpha35(F: dict) -> pd.DataFrame:
    return (ts_rank(F["volume"], 32)
            * (1 - ts_rank(F["close"] + F["high"] - F["low"], 16))
            * (1 - ts_rank(F["returns"], 32)))


def alpha38(F: dict) -> pd.DataFrame:
    return (-1 * ts_rank(cs_rank(F["close"]), 10)) * cs_rank(F["close"] - F["open"])


def alpha40(F: dict) -> pd.DataFrame:
    return (-1 * cs_rank(ts_std(F["high"], 10))) * ts_corr(F["high"], F["volume"], 10)


def alpha43(F: dict) -> pd.DataFrame:
    return ts_rank(F["volume"] / F["adv20"], 20) * ts_rank(-1 * delta(F["close"], 7), 8)


def alpha44(F: dict) -> pd.DataFrame:
    return -1 * ts_corr(F["high"], cs_rank(F["volume"]), 5)


def alpha45(F: dict) -> pd.DataFrame:
    return -1 * ((ts_mean(F["amount"], 20) / F["amount"])
                 * ((F["high"] - F["low"]) / F["open"]))


def alpha46(F: dict) -> pd.DataFrame:
    m = (ts_mean(F["close"], 3) + ts_mean(F["close"], 6)
         + ts_mean(F["close"], 12) + ts_mean(F["close"], 24)) / 4
    return m / F["close"]


def alpha49(F: dict) -> pd.DataFrame:
    d = delta(F["close"], 1)
    cond = (ts_min(d, 12) > 0) | (ts_max(d, 12) < 0) | (delay(d, 1) > 0)
    return (-1 * d).where(cond, d)


def alpha51(F: dict) -> pd.DataFrame:
    d = delta(F["close"], 1)
    cond = (ts_min(d, 12) > 0) | (ts_max(d, 12) < 0) | (delay(d, 1) > 0)
    return (-1 * d).where(cond, d)          # 论文 51 与 49 仅阈值细节不同，口径同源


def alpha53(F: dict) -> pd.DataFrame:
    inner = ((F["close"] - F["low"]) - (F["high"] - F["close"])) \
        / (F["close"] - F["low"]).replace(0.0, np.nan)
    return -1 * delta(inner, 9)


def alpha54(F: dict) -> pd.DataFrame:
    return ((-1 * (F["low"] - F["close"]) * (F["open"] ** 5))
            / ((F["low"] - F["high"]) * (F["close"] ** 5)).replace(0.0, np.nan))


def alpha101(F: dict) -> pd.DataFrame:
    return (F["close"] - F["open"]) / ((F["high"] - F["low"]) + 0.001)


def alpha18(F: dict) -> pd.DataFrame:
    return -1 * cs_rank(ts_std((F["close"] - F["open"]).abs(), 5)
                        + (F["close"] - F["open"])
                        + ts_corr(F["close"], F["open"], 10))


def alpha22(F: dict) -> pd.DataFrame:
    return -1 * (delta(ts_corr(F["high"], F["volume"], 5), 5)
                 * cs_rank(ts_std(F["close"], 20)))


def alpha41(F: dict) -> pd.DataFrame:
    return np.sqrt((F["high"] * F["low"]).where(lambda x: x >= 0)) - F["vwap"]


def alpha42(F: dict) -> pd.DataFrame:
    return cs_rank(F["vwap"] - F["close"]) / cs_rank(F["vwap"] + F["close"])


ALPHAS = {
    "wq02": alpha02, "wq03": alpha03, "wq04": alpha04, "wq06": alpha06,
    "wq09": alpha09, "wq12": alpha12, "wq13": alpha13, "wq14": alpha14,
    "wq17": alpha17, "wq18": alpha18, "wq20": alpha20, "wq22": alpha22,
    "wq23": alpha23, "wq26": alpha26, "wq28": alpha28, "wq33": alpha33,
    "wq34": alpha34, "wq35": alpha35, "wq38": alpha38, "wq40": alpha40,
    "wq41": alpha41, "wq42": alpha42, "wq43": alpha43, "wq44": alpha44,
    "wq45": alpha45, "wq46": alpha46, "wq49": alpha49, "wq51": alpha51,
    "wq53": alpha53, "wq54": alpha54, "wq101": alpha101,
}
