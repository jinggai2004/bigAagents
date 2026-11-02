# -*- coding: utf-8 -*-
"""组合构建：闸门（gate）+ 缓冲区调仓（TopK + Drop）+ 权重。

为什么要这一层
──────────────
2026-10-30 的 61 因子全市场回测（`scripts/factor_paper_scan.py`）给出两个事实：

1. 量价因子 IC **全为负**（`lndvol20` −0.109、`upshd20` −0.095、`turn_std20` −0.090）
   ⇒ "打分越高越买"这个方向本身是错的，加权求和取 Top N 等于系统性地买错方向。
2. 61 次试验做 DSR/PBO 校正后**只剩 2 个通过**（`lndvol20` p=0.0028、`amihud20` p=0.0045），
   而这两条的多空收益**相关系数 −0.94（翻正后 +0.94）** ⇒ 本质是**同一个赌注**：
   做多低流动性。PBO=0.0435（低=好）说明因子排序本身稳定，但"稳定"不等于"有这么多因子"。

所以正确用法不是"加权求和取最高分"，而是：
- **闸门**：负 IC 因子的极端分位只做**剔除**（减法），不做加分；
- **缓冲区**：跌出 Top(k×buffer) 才卖，单次换手受控——月频全换会把超额被成本吃光。

⚠️ 与现有流动性下限的张力（**必须显式记录，不能悄悄推翻**）
──────────────────────────────────────────────────────────
`core/screener.py` 现有闸门是"按板块剔除成交额后 30% 分位"（容量保护）；
而因子结论是"低流动性端收益最高"。两者方向相反，都有道理：
- 因子层面：低流动性确实有超额（两条存活因子年化 SR ≈ 2.3~2.4）；
- 但 `amihud20` 的 Q5−Q1=+4.07%/月 全部来自"冲击成本高"的票，**大资金进不去**。

本模块因此把流动性做成**双侧闸门**：
- `min_amount_yuan` 保容量（不买进不去的票）；
- `max_amount_pct` 可选，剔除**极端高流动性**端（那一端是确定的负 alpha）。
两者谁紧谁松由回测决定（`scripts/portfolio_ab_test.py`），**不由本模块拍板**。

设计约束
────────
- **不静默**：缺列就记进 `missing_inputs` 并跳过该道闸门，绝不假装通过；
  每道闸门拒绝多少只都计数，供归因。
- 涨跌停一律走 `core.zt_calendar.limit_ratio`（板块 + 制度分界日），**不准写死 10%**。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.zt_calendar import limit_ratio

# 拒绝原因码（落库/归因用，改字符串会破坏历史统计）
REJ_SUSPENDED = "停牌/无成交"
REJ_LIMIT_UP = "涨停不可买"
REJ_BELOW_PRICE = "价格低于面值线"
REJ_MIN_AMOUNT = "成交额低于门槛"
REJ_MAX_AMOUNT = "成交额高于上沿"
REJ_FACTOR_HIGH = "因子高分位剔除"
REJ_FACTOR_LOW = "因子低分位剔除"

# 涨停判定的价格容差：分价位是 0.01 元，直接用 == 会因浮点误差漏判
_PRICE_TOL = 1e-6


@dataclass(frozen=True)
class GateConfig:
    """闸门配置。全部默认"不启用"以保持与旧行为可比，由调用方显式打开。"""

    min_amount_yuan: float = 0.0
    max_amount_pct: float | None = None          # 例如 0.95 = 剔除成交额最高的 5%
    min_price: float = 0.0                       # 例如 1.0 元（面值退市风险）
    exclude_limit_up: bool = True
    exclude_suspended: bool = True
    # ((列名, "high"|"low", 分位), ...) 例如 (("upshd20", "high", 0.90),)
    # high ⇒ 剔除该列**最大**的 (1-分位) 部分；low ⇒ 剔除最小的那部分
    factor_gates: tuple[tuple[str, str, float], ...] = ()


def _validate_config(config: GateConfig) -> None:
    """参数错误必须在**碰数据之前**就抛。

    ⚠️ 不能等到"存活池 ≥20 只"才算分位、才顺便校验 side：那样小样本下
    非法配置会被静默跳过（实测踩到：side="middle" 在小池子上不报错）。
    """
    if config.max_amount_pct is not None and not (0.0 < config.max_amount_pct < 1.0):
        raise ValueError(f"max_amount_pct 必须在 (0,1)，收到 {config.max_amount_pct}")
    if config.min_amount_yuan < 0 or config.min_price < 0:
        raise ValueError("min_amount_yuan / min_price 不能为负")
    for col, side, q in config.factor_gates:
        if side not in ("high", "low"):
            raise ValueError(f"factor_gates 的 side 只能是 high/low，收到 {side!r}")
        if not (0.0 < q < 1.0):
            raise ValueError(f"factor_gates 的分位必须在 (0,1)，收到 {q!r}（列 {col}）")


def apply_gates(df: pd.DataFrame, config: GateConfig, *,
                trade_date=None,
                col_price: str = "price",
                col_prev_close: str = "prev_close",
                col_amount: str = "amount",
                col_volume: str = "volume",
                col_is_st: str = "is_st") -> tuple[pd.DataFrame, dict]:
    """按顺序过闸门，返回 (存活子集, 统计)。

    Args:
        df: 截面候选，一行一只。列名见 `col_*` 参数。
        config: `GateConfig`。
        trade_date: 涨跌停判定的日期（板块制度分界日按它算）。

    Returns:
        (kept_df, stats)：
        - `stats["rejected"]` = {原因码: 只数}（**只统计第一次被拒的原因**，
          所以各原因之和 + kept == total，不会重复计数）
        - `stats["missing_inputs"]` = [(闸门名, 缺的列), ...]，缺列时该闸跳过
        - `stats["factor_nan"]` = {列名: 该列为 NaN 的只数}（NaN 视为**通过**，
          见下方 ⚠️）

    ⚠️ 因子闸门遇到 NaN 走"通过"：因子缺失不等于"它是极端值"，把缺数据当极端
    会静默砍掉新上市/数据不全的票。但这意味着因子闸门的实际覆盖率要单独看
    `factor_nan` ——调用方必须把这些数字展示出来，不能默认 100% 生效。
    """
    _validate_config(config)                 # 先校验，再看数据量
    if df is None or len(df) == 0:
        return df, {"total": 0, "kept": 0, "rejected": {}, "missing_inputs": [],
                    "factor_nan": {}}

    total = int(len(df))
    keep = pd.Series(True, index=df.index)
    rejected: Counter = Counter()
    missing: list[tuple[str, str]] = []
    factor_nan: dict[str, int] = {}

    def _reject(mask: pd.Series, reason: str) -> None:
        """只记"第一次被拒"，保证 reasons + kept == total。"""
        nonlocal keep                      # ⚠️ `keep &= ~m` 是增量赋值，没有这行会变成局部变量
        m = mask.fillna(False) & keep
        n = int(m.sum())
        if n:
            rejected[reason] += n
            keep &= ~m

    if config.exclude_suspended:
        if col_volume in df.columns:
            vol = pd.to_numeric(df[col_volume], errors="coerce")
            _reject(vol.isna() | (vol <= 0), REJ_SUSPENDED)
        else:
            missing.append(("停牌", col_volume))

    if config.exclude_limit_up:
        need = (col_price, col_prev_close)
        if all(c in df.columns for c in need):
            px = pd.to_numeric(df[col_price], errors="coerce")
            pc = pd.to_numeric(df[col_prev_close], errors="coerce")
            st = df[col_is_st] if col_is_st in df.columns else pd.Series(False, index=df.index)
            codes = df["code"].astype(str) if "code" in df.columns else pd.Series("", index=df.index)
            # 逐只算幅（板块+日期+ST），NaN 视为"无法判定"⇒ 不拒（记进 missing）
            reasons = [limit_ratio(c, s, trade_date) for c, s in zip(codes, st)]
            lim = pd.Series(reasons, index=df.index, dtype="float64")
            unjudged = int((px.notna() & pc.notna() & lim.isna()).sum())
            if unjudged:
                missing.append(("涨停", f"{unjudged} 只幅度未知"))
            hit = (px >= pc * (1.0 + lim) - _PRICE_TOL) & px.notna() & pc.notna()
            _reject(hit, REJ_LIMIT_UP)
        else:
            lacking = [c for c in need if c not in df.columns]
            if lacking:
                missing.append(("涨停", ",".join(lacking)))

    if config.min_price > 0:
        if col_price in df.columns:
            px = pd.to_numeric(df[col_price], errors="coerce")
            _reject(px < config.min_price, REJ_BELOW_PRICE)
        else:
            missing.append(("价格下限", col_price))

    if config.min_amount_yuan > 0:
        if col_amount in df.columns:
            amt = pd.to_numeric(df[col_amount], errors="coerce")
            _reject(amt < config.min_amount_yuan, REJ_MIN_AMOUNT)
        else:
            missing.append(("成交额下限", col_amount))

    if config.max_amount_pct is not None:
        if col_amount in df.columns:
            amt = pd.to_numeric(df[col_amount], errors="coerce")
            live = amt[keep]
            if len(live) >= 20:                       # 分位在**存活池内**算
                q = float(config.max_amount_pct)
                thresh = float(live.quantile(q))
                _reject(amt > thresh, REJ_MAX_AMOUNT)
            else:
                missing.append(("成交额上沿", f"存活仅 {len(live)} 只，分位不可估"))
        else:
            missing.append(("成交额上沿", col_amount))

    for col, side, q in config.factor_gates:
        if col not in df.columns:
            missing.append((f"因子闸门 {col}", col))
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        nan_n = int((v.isna() & keep).sum())
        if nan_n:
            factor_nan[col] = nan_n
        live = v[keep & v.notna()]
        if len(live) < 20:
            missing.append((f"因子闸门 {col}", f"有效仅 {len(live)} 只，分位不可估"))
            continue
        q = float(q)
        if side == "high":
            thresh = float(live.quantile(q))
            _reject(v > thresh, f"{REJ_FACTOR_HIGH}:{col}")
        elif side == "low":
            thresh = float(live.quantile(1.0 - q))
            _reject(v < thresh, f"{REJ_FACTOR_LOW}:{col}")
        else:                                  # 入口已校验，这里只是兜底
            raise ValueError(f"factor_gates 的 side 只能是 high/low，收到 {side!r}")

    kept = df[keep]
    stats = {"total": total, "kept": int(len(kept)),
             "rejected": dict(rejected), "missing_inputs": missing,
             "factor_nan": factor_nan}
    return kept, stats


def select_with_buffer(ranked: list[str], prev: list[str], k: int, *,
                       buffer_mult: float = 1.5) -> dict:
    """TopK + Drop：**跌出 Top(k×buffer) 才卖**，把换手压下来。

    规则（顺序重要）：
    1. 旧持仓只要仍在 `ranked[:buffer]` 内 ⇒ 留住（不因为排名略降就卖）；
    2. 剩余名额按排名从 `ranked[:buffer]` 里补未持有的；
    3. 仍不足 k 只（存活池太小）⇒ 继续往下补到 k，并置 `relaxed=True`（**不静默**）；
    4. 不在 `ranked` 里的旧持仓 ⇒ 卖（说明它今天没过闸门）。

    Args:
        ranked: 已过滤后的候选，**按分数降序**（只用来排序，不用分值）。
        prev: 当前持仓代码。
        k: 目标只数。
        buffer_mult: 缓冲倍数，1.0 = 严格 TopK（换手最高）；1.5~2.0 常见。

    Returns:
        {codes, kept, bought, sold, turnover, buffer, relaxed}
        - `turnover` = 买入只数 / k（单边换手率，0~1）。**这是成本估算的输入**，
          不是"卖了又买了"的总额。

    ⚠️ `ranked` 必须已经过闸门。把闸门放在这里会导致"排名靠前但涨停"的票被买进。
    """
    if k <= 0:
        raise ValueError("k 必须 ≥1")
    if buffer_mult < 1.0:
        raise ValueError("buffer_mult 必须 ≥1（<1 会让缓冲区比目标持仓还小）")
    ranked = [str(c) for c in ranked]
    prev = [str(c) for c in prev]
    buffer = max(k, int(round(k * buffer_mult)))
    head = ranked[:buffer]
    head_set = set(head)

    kept = [c for c in prev if c in head_set]
    kept_set = set(kept)
    bought = [c for c in head if c not in kept_set and c not in set(prev)][:k - len(kept)]
    codes = kept + bought

    relaxed = False
    if len(codes) < k:                      # 存活池不够 ⇒ 继续往下补，并留痕
        relaxed = True
        chosen = set(codes) | set(prev)
        for c in ranked:
            if len(codes) >= k:
                break
            if c not in chosen:
                codes.append(c)
                chosen.add(c)

    sold = [c for c in prev if c not in set(codes)]
    return {"codes": codes,
            "kept": kept,
            "bought": bought,
            "sold": sold,
            "turnover": round(len(bought) / k, 4) if k else 0.0,
            "buffer": buffer,
            "relaxed": relaxed}


def equal_weights(codes: list[str], total: float = 1.0) -> dict[str, float]:
    """等权。空列表返回空 dict（不返 0 权重，那会让下游误以为"有仓位但为 0"）。"""
    n = len(codes)
    if n == 0:
        return {}
    return {str(c): total / n for c in codes}


def inverse_vol_weights(codes: list[str], vols: dict[str, float], *,
                        total: float = 1.0, cap: float | None = None) -> dict[str, float]:
    """波动率倒数权重（低波多配），可选单票上限后归一。

    Args:
        vols: {code: 年化波动率}，**必须为正**。缺失或非正的票给中位数波动率
            （等权倾向），不剔除——剔除会让"因数据缺失而低配"变成隐性选股。
        cap: 单票权重上限（如 0.15）。
    """
    codes = [str(c) for c in codes]
    if not codes:
        return {}
    vals = [vols.get(c) for c in codes]
    pos = [v for v in vals if v is not None and v > 0 and np.isfinite(v)]
    fallback = float(np.median(pos)) if pos else 1.0
    clean = {c: (v if (v is not None and v > 0 and np.isfinite(v)) else fallback)
             for c, v in zip(codes, vals)}
    inv = {c: 1.0 / clean[c] for c in codes}
    s = sum(inv.values())
    w = {c: total * inv[c] / s for c in codes}
    if cap is not None and 0 < cap < 1:
        # 迭代削峰：超上限的钉在 cap，剩余再按原比例分（2 轮足够收敛到常见情形）
        for _ in range(8):
            over = {c: v for c, v in w.items() if v > cap + 1e-12}
            if not over:
                break
            for c in over:
                w[c] = cap
            rest = [c for c in w if w[c] < cap - 1e-12]
            free = total - cap * len(over)
            base = sum(inv[c] for c in rest)
            if not rest or base <= 0 or free <= 0:
                break
            for c in rest:
                w[c] = free * inv[c] / base
    return {c: round(v, 6) for c, v in w.items()}
