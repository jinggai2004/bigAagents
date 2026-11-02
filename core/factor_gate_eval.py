# -*- coding: utf-8 -*-
"""因子「当闸门」的效果评估：剔掉最差 X% 之后，组合的前瞻收益到底变好多少。

为什么需要这个模块
──────────────────
本项目对"因子怎么用"已经有一条硬结论：
- 把弱 IC 因子**混进等权合成**是稀释（AF 臂押注段 −9.47% vs 现状 −0.12%，否决）；
- 把它**当闸门**（只剔最差 10%、排序不动）则有效（AFG 臂 +2.26%）。

但那条结论只验了 `ocf_to_np` 一个因子。要回答"因子研究怎么实际应用"，需要的是
**对所有因子、所有剔除比例**都能量出效果的量具——否则每次都得跑一遍完整 A/B
（十几分钟），根本没法批量筛选。

口径（必须与 A/B 一致，否则量出来的不是线上行为）
────────────────────────────────────────────────
1. 每月截面，按因子排序，剔掉 `pct` 比例的**最差端**（方向由全样本 IC 符号决定）；
2. 闸门收益 = 剩余池**等权** `fwd20` 均值 − 全池等权 `fwd20` 均值；
3. 对月求均值与 t（月数即样本量）；
4. 分年同号率 —— 只有一两年好看的不算。

⚠️ 为什么是"等权剩余池"而不是"原排序前 N"：原排序用的是主观六因子权重（无实证），
   把它混进来就分不清"改善来自闸门"还是"来自那套权重"。等权是最干净的中性对照。

⚠️ 为什么方向由**全样本** IC 符号决定：这引入了 1 bit 的样本内信息，与
   `factor_multiple_testing` 的做法一致（它也是按样本内均值符号翻正）。
   所以结论要过 DSR 才算数，本模块只给"值得进一步验证"的排序。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.factor_stats import MIN_STOCKS_PER_MONTH

DEFAULT_PCTS = (0.05, 0.10, 0.20, 0.30)
MIN_MONTHS_GATE = 24          # 少于两年不下结论（与 factor_stats 一致）
MIN_POS_RATE = 0.6            # 月度胜率下限


def _monthly_ic(panel: pd.DataFrame, col: str,
                min_stocks: int = MIN_STOCKS_PER_MONTH) -> tuple[float, int]:
    """全样本月度截面 Spearman 均值（只用来定方向，不做显著性判定）。"""
    d = panel[["month_ts", col, "fwd20"]].dropna()
    if d.empty:
        return 0.0, 0
    d = d.assign(rf=d.groupby("month_ts")[col].rank(pct=True),
                 ff=d.groupby("month_ts")["fwd20"].rank(pct=True))
    per = d.groupby("month_ts")[["rf", "ff"]].apply(
        lambda g: g["rf"].corr(g["ff"]) if len(g) >= min_stocks else np.nan).dropna()
    return (float(per.mean()) if len(per) else 0.0), int(len(per))


def _gate_gain(panel: pd.DataFrame, col: str, pct: float, direction: int,
               min_stocks: int) -> dict | None:
    """固定剔除方向的闸门效果。`direction=+1` 剔低端（认为高值好） / `-1` 剔高端。"""
    d = panel[["month_ts", col, "fwd20"]].dropna()
    if d.empty:
        return None
    rows = []
    for m, g in d.groupby("month_ts"):
        if len(g) < max(min_stocks, 40):
            continue
        base = float(g["fwd20"].mean())
        k = max(1, int(round(len(g) * pct)))
        g = g.sort_values(col, ascending=(direction > 0))
        kept = g.iloc[k:]
        if len(kept) < 20:
            continue
        rows.append({"month": m, "gain": float(kept["fwd20"].mean()) - base,
                     "n": len(g), "kept": len(kept)})
    if len(rows) < MIN_MONTHS_GATE:
        return {"ok": False, "months": len(rows)}
    r = pd.DataFrame(rows).set_index("month")
    v = r["gain"]
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    t = mean / sd * math.sqrt(len(v)) if sd > 0 else 0.0
    by_year = {int(y): round(float(x) * 100, 3)
               for y, x in v.groupby(v.index.year).mean().items()}
    return {"ok": True, "direction": direction,
            "gain_pp": round(mean * 100, 3), "t_stat": round(t, 2),
            "months": int(len(v)), "pos_rate": round(float((v > 0).mean()), 3),
            "avg_kept": int(r["kept"].mean()), "by_year": by_year}


def evaluate_gate(panel: pd.DataFrame, col: str, pct: float = 0.10,
                  min_stocks: int = MIN_STOCKS_PER_MONTH,
                  direction: int | None = None) -> dict:
    """单因子单比例的闸门效果。

    ⚠️⚠️ **方向必须双向实测，不能照 IC 符号定**。
    实测反例：`max20` / `ret_max5` 的 IC 为负（高值→低收益），按 IC 方向"剔高值端"
    的结果是 **−0.35pp（变差）**——因为 IC 是全截面的秩相关，而闸门只动**极端尾部**，
    尾部可以和外全体反号。本项目"分组不单调 ⇒ 只能分位筛不能多空"是同一件事。
    ⇒ 默认 `direction=None` 会**两个方向都跑**，取效果更好的那个，并记下
      `direction_vs_ic`：若与 IC 符号不一致，页面上要提示"方向与 IC 相反"。
    """
    ic, _ = _monthly_ic(panel, col, min_stocks)
    ic_dir = 1 if ic >= 0 else -1
    if direction is not None:
        res = _gate_gain(panel, col, pct, direction, min_stocks)
        if not res:
            return {"col": col, "pct": pct, "ok": False, "reason": "无有效数据"}
        return {**res, "col": col, "pct": pct, "ic": round(ic, 4), **res}

    both = {}
    for d in (1, -1):
        r = _gate_gain(panel, col, pct, d, min_stocks)
        if r and r.get("ok"):
            both[d] = r
    if not both:
        return {"col": col, "pct": pct, "ok": False, "reason": "可用月份不足",
                "ic": round(ic, 4)}
    best_dir = max(both, key=lambda d: both[d]["gain_pp"])
    best = both[best_dir]
    other = both.get(-best_dir)
    return {
        "col": col, "pct": pct, "ok": True, "ic": round(ic, 4),
        "direction": best_dir,
        "direction_vs_ic": "一致" if best_dir == ic_dir else "相反",
        "gain_pp": best["gain_pp"], "t_stat": best["t_stat"],
        "months": best["months"], "pos_rate": best["pos_rate"],
        "avg_kept": best["avg_kept"], "by_year": best["by_year"],
        "year_same_sign": int(sum(1 for x in best["by_year"].values() if x > 0)),
        "opposite_gain_pp": (other or {}).get("gain_pp"),
        "opposite_t": (other or {}).get("t_stat"),
        "verdict": ("值得验证" if (best["t_stat"] >= 2.0 and best["gain_pp"] > 0
                                and best["pos_rate"] >= MIN_POS_RATE)
                    else ("方向可用" if best["t_stat"] >= 1.5 and best["gain_pp"] > 0
                          else "无效")),
    }


def sweep(panel: pd.DataFrame, cols=None, pcts=DEFAULT_PCTS) -> list[dict]:
    """对一批因子 × 一批剔除比例做扫描，按 `gain_pp` 降序。

    ⚠️ 这是一次**批量试验**：真要认真用，必须拿这些结果去过 DSR（试验次数 = 行数）。
       本模块只负责把"哪些值得进一步验证"排出来，不负责宣布"它有效"。
    """
    cols = cols or [c for c in panel.columns
                    if c not in ("month_ts", "month", "code", "fwd20")]
    out = []
    for c in cols:
        for p in pcts:
            out.append(evaluate_gate(panel, c, p))
    out = [x for x in out if x.get("ok")]
    out.sort(key=lambda x: -x["gain_pp"])
    return out


def direction_mismatch(rows: list[dict]) -> list[dict]:
    """筛出"最佳剔除方向与 IC 符号相反"的因子——这类因子最容易被误用。"""
    return [r for r in rows if r.get("direction_vs_ic") == "相反"]


def evaluate_stack(panel: pd.DataFrame, cols, pct: float = 0.10,
                   min_stocks: int = MIN_STOCKS_PER_MONTH) -> dict:
    """多闸门**依次叠加**：剔完一个再剔下一个。

    这是把因子研究直接落地的形态——不回答"哪个因子权重多少"（那需要主观假设），
    只回答"依次剔掉这几类明显差的票，剩下的是不是更好"。
    ⚠️ 叠加会放大剔除量（n 个闸门各剔 pct ⇒ 理论剩 (1-pct)^n），
       所以必须同时报 `avg_kept`，否则会出现"收益变好但其实只剩 40 只"。
    """
    # 方向用 evaluate_gate 的双向实测结论（不是 IC 符号——尾部和全截面可以反号）
    dirs = {}
    for c in cols:
        g = evaluate_gate(panel, c, pct, min_stocks)
        dirs[c] = int(g.get("direction") or 1) if g.get("ok") else 1
    rows = []
    for m, g in panel.groupby("month_ts"):
        # ⚠️⚠️ 基准池必须与闸门**作用**的池同口径：先用所有用到的因子 dropna 定池，
        #    再在这个池上算 base。首版是用 `dropna(fwd20)` 的**超集**算 base，
        #    结果"因因子缺失而不参与筛选"的票被算进基准，把 `lndvol20` 从 +0.19
        #    翻成 −0.59（结论直接反号，且看不出哪里错）。单因子版本没这个问题，
        #    所以是"两个函数各写一遍基准"造成的口径分叉——同类事故在项目里出现过多次。
        sub = g.dropna(subset=["fwd20", *cols])
        if len(sub) < max(min_stocks, 40):
            continue
        base = float(sub["fwd20"].mean())
        keep = sub
        for c in cols:
            keep = keep.dropna(subset=[c])
            if len(keep) < 40:
                break
            k = max(1, int(round(len(keep) * pct)))
            keep = keep.sort_values(c, ascending=(dirs[c] > 0)).iloc[k:]
        if len(keep) < 20:
            continue
        rows.append({"month": m, "gain": float(keep["fwd20"].mean()) - base,
                     "kept": len(keep), "n": len(sub)})
    if len(rows) < MIN_MONTHS_GATE:
        return {"ok": False, "reason": f"可用月份不足（{len(rows)}）", "cols": list(cols)}
    r = pd.DataFrame(rows).set_index("month")
    v = r["gain"]
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    t = mean / sd * math.sqrt(len(v)) if sd > 0 else 0.0
    return {"ok": True, "cols": list(cols), "pct": pct,
            "gain_pp": round(mean * 100, 3), "t_stat": round(t, 2),
            "months": int(len(v)), "pos_rate": round(float((v > 0).mean()), 3),
            "avg_kept": int(r["kept"].mean()), "avg_pool": int(r["n"].mean()),
            "keep_ratio": round(float((r["kept"] / r["n"]).mean()), 3),
            "by_year": {int(y): round(float(x) * 100, 3)
                        for y, x in v.groupby(v.index.year).mean().items()}}


# ══════════════════════════════════════════════════════════════════
# Top-N 命中率：面板级超额 ≠ 组合级收益（2026-11-01 实测教训）
# ══════════════════════════════════════════════════════════════════
# 反面教材（本项目自己的）：`evaluate_stack` 报三道闸门 +0.76pp/月 t=5.00 胜率 76%，
# 看起来该上线；真去跑组合级 A/B，G3 臂与现状 A 臂在开发段/验证段的收益**逐位相同**
# （+11.23% / +43.07%）。根因：9 年 1,800 个 Top20 席位里，闸门只剔到 **1 个**（0.06%）
# ——闸门剔的极端尾部（lndvol20 高端 / amihud20 低端 / upshd20 高端）与 composite
# 的低分区高度共线，Top20 本来就不含它们。闸门在做排序已经在做的事。
#
# ⇒ 任何新闸门接进选股链之前，必须先问：**它剔掉的票，有多少本来就在 TopN 里？**
#    命中率 < 2% 就是"面板有效、组合无效"，别浪费一轮 A/B（更别上线）。
TOPN_MIN_HIT = 0.02           # TopN 命中率低于这个 ⇒ 判"对组合无增量"


def topn_impact(panel: pd.DataFrame, gate_cols, dirs: dict,
                pct: float = 0.20, topn: int = 20,
                score: pd.Series | None = None) -> dict:
    """量化闸门对 **TopN 组合**的实际影响（而不是对整个池子的等权影响）。

    Args:
        panel: 月度面板（需 `month_ts`/`code`/`fwd20` + 闸门列）
        gate_cols: 闸门因子列名
        dirs: `{列名: ±1}`，−1=剔高端 / +1=剔低端
        pct: 每道剔多少
        topn: 组合持仓数
        score: 与 panel 对齐的打分（越高越好）。None ⇒ 用 `PROMOTED_FACTORS` 现算。
            ⚠️ 必须与实盘同一个打分池，否则算出来的命中率不是线上那个组合的命中率。

    Returns: 命中率、被剔票的 composite 分位、分位段剔除率、以及面板级对照。
    """
    d = panel
    if score is None:
        from core.factor_composite import PROMOTED_FACTORS, composite_score
        cols = [c for c in PROMOTED_FACTORS if c in d.columns]
        if not cols:
            return {"ok": False, "reason": "面板缺 PROMOTED_FACTORS 列，且未传 score"}
        d = d.assign(_score=composite_score(d, cols, PROMOTED_FACTORS))
    else:
        d = d.assign(_score=pd.to_numeric(score, errors="coerce").reindex(d.index))

    need = ["month_ts", "code", "fwd20", "_score", *gate_cols]
    d = d.dropna(subset=[c for c in need if c in d.columns])
    if d.empty:
        return {"ok": False, "reason": "面板无有效行"}

    slots = hits = dropped = 0
    quantiles, buckets = [], {"前0~5%": [0, 0], "前5~10%": [0, 0],
                              "前10~20%": [0, 0], "前20~40%": [0, 0],
                              "前40%以后": [0, 0]}
    kept_fwd, dropped_fwd = [], []
    months = 0
    for _, g in d.groupby("month_ts", sort=True):
        if len(g) < max(50, topn * 3):
            continue
        months += 1
        g = g.assign(_rank=g["_score"].rank(ascending=False, method="first"))
        top = g.nsmallest(topn, "_rank")
        slots += len(top)

        keep = pd.Series(True, index=g.index)
        for c in gate_cols:
            v = pd.to_numeric(g[c], errors="coerce")
            if v.notna().sum() < 50:
                continue
            dir_i = int(dirs.get(c, -1))
            bad = (v > v.quantile(1 - pct)) if dir_i < 0 else (v < v.quantile(pct))
            keep &= ~(bad & v.notna())         # NaN 不剔
        hit_mask = ~keep
        hits += int((hit_mask & g["_rank"].le(topn)).sum())
        dropped += int(hit_mask.sum())

        if hit_mask.any():
            r = g.loc[hit_mask, "_rank"] / len(g)
            quantiles.append(float(r.median()))
            kept_fwd.append(float(g.loc[keep, "fwd20"].mean()))
            dropped_fwd.append(float(g.loc[hit_mask, "fwd20"].mean()))
        for key, (lo, hi) in (("前0~5%", (0, .05)), ("前5~10%", (.05, .10)),
                              ("前10~20%", (.10, .20)), ("前20~40%", (.20, .40)),
                              ("前40%以后", (.40, 1.01))):
            r = g["_rank"] / len(g)
            msk = (r > lo) & (r <= hi)
            buckets[key][0] += int(msk.sum())
            buckets[key][1] += int((msk & hit_mask).sum())

    if months < MIN_MONTHS_GATE:
        return {"ok": False, "reason": f"可用月份不足（{months}）"}

    hit_rate = hits / max(slots, 1)
    return {
        "ok": True, "months": months, "topn": topn,
        "topn_slots": slots, "hit": hits,
        "hit_rate": round(hit_rate, 4),
        "dropped_total": dropped,
        "dropped_share": round(dropped / max(len(d), 1), 3),
        "dropped_composite_quantile_median": (round(float(np.median(quantiles)), 3)
                                              if quantiles else None),
        "drop_rate_by_bucket": {k: round(v[1] / max(v[0], 1), 3)
                                for k, v in buckets.items()},
        # 面板级对照：证明"闸门本身是对的，只是对 TopN 无用"
        "panel_kept_fwd_pp": round(float(np.mean(kept_fwd)) * 100, 3) if kept_fwd else None,
        "panel_dropped_fwd_pp": (round(float(np.mean(dropped_fwd)) * 100, 3)
                                 if dropped_fwd else None),
        "verdict": ("对组合无增量（与打分共线，TopN 本就不含被剔票）"
                    if hit_rate < TOPN_MIN_HIT else "可能有效（TopN 确实被剔到）"),
        "has_increment": hit_rate >= TOPN_MIN_HIT,
    }
