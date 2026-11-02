# -*- coding: utf-8 -*-
"""因子截面统计的唯一实现：月度 IC 表 + 月频多空序列。

为什么要单独成模块
──────────────────
`ic_table` / `ls_series` 原本写在 `scripts/financial_factor_ic.py` 里。现在
**因子实验室**（用户自定义因子）要用同一套口径——各写一遍必然走样，本项目
已有教训「页面与 prompt 说同一件事必须同源（共用同一个函数）」。

本模块的函数**逐字来自** 2026-10-31 那份验证过的脚本（`ocf_to_np` 过
DSR p=0.015 的结论就建立在它上面），搬动时不许改数值逻辑；改口径必须重跑
那份脚本并比对 JSON 里的每一个数字。

口径（三条不可动）
──────────────────
1. **IC = 截面 Spearman**：每月把因子与 `fwd20` 各转百分位后求相关，再对月求均值。
   不用 Pearson——极值会主导，而本项目的因子（amt/市值/波动）全是长尾。
2. **单月截面少于 `min_stocks` 只 ⇒ 该月不计**，避免小样本月份把 IC 拖动。
3. **多空序列 = 前 20% − 后 20% 的 `fwd20` 均值差**，不做方向对齐；方向由
   `factor_multiple_testing` 按样本内均值符号统一翻正（模块 docstring 第 2 条）。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MIN_STOCKS_PER_MONTH = 100      # 单月截面少于这个数，IC 不可信
MIN_MONTHS = 24                 # 少于两年不做结论


def ls_series(panel: pd.DataFrame, col: str, top: float = 0.2) -> pd.Series | None:
    """月频多空序列：按因子截面前 20% − 后 20% 的 fwd20 均值差。

    供 `adjust_by_multiple_testing` 做 DSR/PBO（试验分布 = |月频 SR|）。
    方向在函数内**不做**对齐——由校正函数统一按样本内均值符号翻正。
    """
    d = panel[["month_ts", "code", col, "fwd20"]].dropna()
    if d.empty:
        return None

    def _spread(g: pd.DataFrame) -> float:
        if len(g) < 25:
            return np.nan
        k = max(1, int(len(g) * top))
        g = g.sort_values(col)
        return float(g["fwd20"].tail(k).mean() - g["fwd20"].head(k).mean())

    per = d.groupby("month_ts")[[col, "fwd20"]].apply(_spread)
    per = per.dropna()
    return per if len(per) >= 12 else None


def ic_table(panel: pd.DataFrame, name: str, col: str,
             min_stocks: int = MIN_STOCKS_PER_MONTH) -> dict:
    """单个因子的月度截面 IC 统计。

    Returns 里带 `verdict`；**只看 verdict 是不够的**——`factor_multiple_testing`
    会把"有效"覆写成"有效（过 DSR）/未过多重检验校正"，那才是最终结论。
    """
    d = panel[["month_ts", "code", col, "fwd20"]].dropna()
    if d.empty:
        return {"name": name, "verdict": "无数据"}
    d["rf"] = d.groupby("month_ts")[col].rank(pct=True)
    d["ff"] = d.groupby("month_ts")["fwd20"].rank(pct=True)
    # ⚠️ 用 apply 传自定义函数而不是先 groupby(...).corr()：避免 pandas 2.x 的
    #    "apply operated on the grouping columns" 弃用告警淹没输出。
    def _corr(g: pd.DataFrame) -> float:
        return float(g["rf"].corr(g["ff"])) if len(g) >= min_stocks else np.nan

    per = d.groupby("month_ts")[["rf", "ff"]].apply(_corr)
    ic = per.dropna()
    if len(ic) < MIN_MONTHS:
        return {"name": name, "verdict": f"月份不足（{len(ic)}）", "n_months": int(len(ic))}
    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    t = mean / std * math.sqrt(len(ic)) if std > 0 else 0.0
    by_year = {int(y): round(float(v), 4) for y, v in ic.groupby(ic.index.year).mean().items()}
    d["q"] = d.groupby("month_ts")[col].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) if len(x) >= 25 else np.nan)
    qm = d.dropna(subset=["q"]).groupby(["month_ts", "q"])["fwd20"].mean().unstack()
    spread = float((qm[4] - qm[0]).mean()) if (4 in qm and 0 in qm) else None
    return {"name": name, "ic_mean": round(mean, 4),
            "ic_ir": round(mean / std, 3) if std > 0 else None,
            "t_stat": round(t, 2), "n_months": int(len(ic)),
            "pos_pct": round(float((ic > 0).mean()), 3),
            "q5_q1_spread": round(spread, 4) if spread is not None else None,
            "by_year": by_year,
            "verdict": ("有效" if abs(t) >= 2.0 and abs(mean) >= 0.015
                        else ("方向可用" if abs(t) >= 1.5 else "无效"))}


def year_consistency(by_year: dict, direction: int) -> dict:
    """分年同号率——**比全样本 t 值更能识破"靠某一年"的因子**。

    `direction=+1` 表示"分年 IC 应为正"。只有 1~2 年的样本不给结论（返回 None），
    因为 2 个点同号有 50% 概率纯属偶然。
    """
    if not by_year:
        return {"years": 0, "same_sign": 0, "ratio": None, "consistent": None}
    vals = [v for v in by_year.values() if v is not None]
    if len(vals) < 3:
        return {"years": len(vals), "same_sign": sum(
            1 for v in vals if v * direction > 0), "ratio": None, "consistent": None}
    same = sum(1 for v in vals if v * direction > 0)
    ratio = same / len(vals)
    return {"years": len(vals), "same_sign": same,
            "ratio": round(ratio, 3), "consistent": ratio >= 0.7}
