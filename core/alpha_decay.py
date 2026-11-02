# -*- coding: utf-8 -*-
"""超额衰减监控：策略的月度超额一旦转负/腰斩，要在**季度内**发现，而不是等年报。

为什么需要
──────────
2026-10-31 实测：同一套选股在开发段/验证段/押注段的**扣成本超额**分别是
+2.06% / −0.38% / +2.12%（年化），波动幅度远大于水平本身。也就是说
"这套策略还行"这个判断，**随时可能因为几个月的数据就翻转**。
靠"季度复盘时人工看净值"太慢，需要机械判据。

判据（三层，由松到紧）
──────────────────────
| 层 | 条件 | 含义 |
|---|---|---|
| `alert` | 最近 `recent` 个月累计超额 ≤ 0 | 策略当前**不赚钱**，该停手复盘 |
| `watch` | 最近窗口超额 < 历史同窗口超额中位数的 `HALVE`(0.5) 倍 | 衰减到一半以下 |
| `ok` | 其余 | 正常波动范围 |

⚠️ 三条防误判（都是本项目踩过的坑）
────────────────────────────────────
1. **窗口不能太短**：月频策略 3 个月超额的标准误极大，`recent` 默认 6 个月；
2. **必须用超额而不是绝对收益**：牛市里策略亏钱也可能"年化为正"，判据会失效；
3. **没有足够历史就不下结论**：少于 `MIN_MONTHS` 个月返回 `unknown`，
   不返 `ok`（"没数据"不等于"没问题"）。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MIN_MONTHS = 24          # 少于两年不下结论
DEFAULT_RECENT = 6       # 最近窗口（月）
HALVE = 0.5              # 衰减到历史中位数的这个比例 ⇒ watch

LEVEL_OK = "ok"
LEVEL_WATCH = "watch"
LEVEL_ALERT = "alert"
LEVEL_UNKNOWN = "unknown"


def _compound(x: pd.Series) -> float:
    if len(x) == 0:
        return float("nan")
    return float((1.0 + x).prod() - 1.0)


def rolling_excess(returns: pd.Series, bench: pd.Series,
                   window: int = DEFAULT_RECENT) -> pd.Series:
    """滚动 `window` 个月的**累计超额**（策略复利 − 基准复利）。

    ⚠️ 用"复利差"而不是"月度差的均值"：月度差之和在小样本下会低估回撤期的影响，
       而投资者感受的是净值差。
    """
    r = pd.Series(returns).dropna()
    b = pd.Series(bench).reindex(r.index)
    ex = pd.Series(np.nan, index=r.index, dtype="float64")
    if len(r) < window:
        return ex
    for i in range(window - 1, len(r)):
        sl = slice(i - window + 1, i + 1)
        ex.iloc[i] = _compound(r.iloc[sl]) - _compound(b.iloc[sl])
    return ex


def decay_report(returns: pd.Series, bench: pd.Series, *,
                 recent: int = DEFAULT_RECENT,
                 halve: float = HALVE,
                 min_months: int = MIN_MONTHS) -> dict:
    """算超额衰减状态。

    Args:
        returns: 策略**月频**收益（已扣成本）。
        bench: 同口径基准月频收益（同一宇宙！否则超额是假的）。
        recent: 最近窗口月数。
        halve: 历史中位数的多少倍算"腰斩"。
        min_months: 历史至少多少个月才下结论。

    Returns:
        {level, months, recent_excess, hist_median, ratio, rolling,
         last_month, reason}
        - `level=unknown` 时 `reason` 说明为什么不下结论（不静默当 ok）。
    """
    r = pd.Series(returns).dropna().sort_index()
    if len(r) < min_months:
        return {"level": LEVEL_UNKNOWN, "months": int(len(r)),
                "reason": f"历史仅 {len(r)} 个月（需 ≥{min_months}）"}
    b = pd.Series(bench).reindex(r.index)
    if b.isna().all():
        return {"level": LEVEL_UNKNOWN, "months": int(len(r)),
                "reason": "基准序列对不上（检查是否同一宇宙/同一月份索引）"}

    ex = rolling_excess(r, b, recent).dropna()
    if len(ex) < 3:
        return {"level": LEVEL_UNKNOWN, "months": int(len(r)),
                "reason": f"滚动窗口仅 {len(ex)} 个，无法比较历史"}

    latest = float(ex.iloc[-1])
    hist = ex.iloc[:-1]                      # 不含当前窗口，避免自己比自己
    med = float(hist.median())
    med_ok = math.isfinite(med) and med > 0
    ratio = (latest / med) if med_ok else None

    if latest <= 0:
        level, reason = LEVEL_ALERT, f"最近 {recent} 个月累计超额 {latest:+.2%} ≤ 0"
    elif med_ok and latest < med * halve:
        level, reason = LEVEL_WATCH, (f"最近 {recent} 个月超额 {latest:+.2%}，"
                                       f"仅为历史中位数 {med:+.2%} 的 {ratio:.0%}")
    else:
        # ⚠️ 历史中位数 ≤0（策略一半以上时间跑输基准）时**不能**用 ratio 判衰减：
        #    latest/med 会变成负数，`ratio < 0.5` 恒成立 ⇒ 把"比历史好"误判成"腰斩"。
        #    实测踩到（A 臂历史中位 −1.84%、最近 +6.71% 被判 WATCH）。
        #    这种情况下"最近为正"本身就是**改善**，判 ok 并由 reason 说清。
        tail = "（历史中位数 ≤0，说明该策略多数时间跑输基准；当前为正属改善）" \
            if not med_ok else ""
        level, reason = LEVEL_OK, (f"最近 {recent} 个月超额 {latest:+.2%}"
                                   f"（历史中位 {med:+.2%}）{tail}")

    return {"level": level, "months": int(len(r)),
            "recent_excess": round(latest, 4),
            "hist_median": round(med, 4) if math.isfinite(med) else None,
            "ratio": round(ratio, 3) if ratio is not None else None,
            "last_month": str(pd.Timestamp(r.index[-1]).date()),
            "reason": reason,
            # 全序列留给画图/落库（不参与判定）
            "rolling": {str(pd.Timestamp(k).date()): round(float(v), 4)
                        for k, v in ex.items()}}


def factor_ic_decay(ic_by_month: pd.Series, *, recent: int = 12) -> dict:
    """因子 IC 的衰减（滚动均值 vs 全样本均值）。

    与 `decay_report` 分开：IC 是**因子层**，超额是**组合层**。
    因子衰减但组合没衰减 = 组合里的其他因子在顶；两者都衰减才说明策略真的老了。
    """
    ic = pd.Series(ic_by_month).dropna().sort_index()
    if len(ic) < MIN_MONTHS:
        return {"level": LEVEL_UNKNOWN, "months": int(len(ic)),
                "reason": f"IC 序列仅 {len(ic)} 个月（需 ≥{MIN_MONTHS}）"}
    full = float(ic.mean())
    roll = ic.rolling(recent, min_periods=max(3, recent // 2)).mean()
    latest = float(roll.iloc[-1])
    sign = 1.0 if full >= 0 else -1.0
    # 用"同号幅度"比较：IC 可以为负（低分位好），衰减看幅度而不是符号
    ratio = (sign * latest / (sign * full)) if full != 0 else None
    if full != 0 and sign * latest <= 0:
        level, reason = LEVEL_ALERT, f"最近 IC {latest:+.3f} 已反向（全样本 {full:+.3f}）"
    elif ratio is not None and ratio < HALVE:
        level, reason = LEVEL_WATCH, f"最近 IC {latest:+.3f} 仅全样本的 {ratio:.0%}"
    else:
        level, reason = LEVEL_OK, f"最近 IC {latest:+.3f}（全样本 {full:+.3f}）"
    return {"level": level, "months": int(len(ic)), "ic_full": round(full, 4),
            "ic_recent": round(latest, 4),
            "ratio": round(ratio, 3) if ratio is not None else None, "reason": reason}
