# -*- coding: utf-8 -*-
"""绩效归因：Brinson 分解（配置/选股/交互）+ Carino 多期链接（设计 3.2）。

回答"超额收益是选对了股（选股），还是押对了行业（配置）"。

口径
────
单期、行业 i（算术 Brinson，行业组合收益按**组合内权重**加权）：
    配置效应 alloc_i = (wp_i − wb_i) · rb_i
    选股效应 select_i = wb_i · (rp_i − rb_i)
    交互效应 inter_i = (wp_i − wb_i) · (rp_i − rb_i)
    合计 Σ_i = rp − rb（恒等式，测试锁死）
其中 wp_i = Σ_{c∈i} w_p,c，rp_i = Σ w_p,c·r_c / wp_i（wp_i=0 ⇒ rp_i=rb_i）。

多期：算术效应跨期不可直接相加。Carino（1992）链接：
    k_t = ln[(1+r_t)/(1+b_t)] / (r_t − b_t)，r_t=b_t 时取极限 1/(1+r_t)
    K   = ln[(1+R)/(1+B)] / (R − B)
    链接后效应 A_{t,i} = k_t/K · a_{t,i}，Σ A = R − B（算术总超额，恒等式）。
    ⚠️ k_t 是"对数差 ÷ 算术差"，不是它的倒数——写反了恒等式闭合不上
    （2026-10-29 测试实测 0.0432 ≠ 0.0448 才暴露）。

【前视偏差标记（评审提醒，必须有）】行业映射来自**今天的**申万成分快照
（`data/industry_swa.py`）——用它解释历史 = 假设行业归属十年不变。
所有输出带 `industry_snapshot` 字段；这是"快照归因"，不是精确归因。

行业缺失的票进 `__未分类__` 桶单独参与分解，**绝不静默丢弃**
（缺就是缺；未分类桶的存在本身就是数据质量的可见信号）。
"""

from __future__ import annotations

import math


def brinson_single_period(port: dict, bench: dict, industry: dict,
                          rets: dict) -> dict:
    """单期 Brinson 分解。

    Args:
        port: {code: 权重}（组合，权重和应为 1；不做强制归一——口径由调用方保证）
        bench: {code: 权重}（基准）
        industry: {code: 行业名}（缺的票进 `__未分类__`）
        rets: {code: 当期收益（小数）}

    Returns:
        {"industries": {行业: {alloc, select, inter, total, wp, wb, rp, rb}},
         "portfolio_return", "benchmark_return", "excess",
         "unclassified_codes": [...]}
        交互效应单列（不并进选股/配置——三效应各自可复核）。
    """
    UNCLASSIFIED = "__未分类__"

    def _agg(weights: dict) -> dict:
        """{行业: (Σ权重, Σ权重×收益)}；收益缺失的票按 0 收益参与（权重照算），
        并单独留痕——比静默剔除诚实（少了一只票的行业收益是有偏的）。
        行业映射缺失的票进 `__未分类__` 桶并留痕（unclassified）。"""
        wp: dict[str, float] = {}
        wr: dict[str, float] = {}
        missing: list[str] = []
        unclassified: list[str] = []
        for code, w in weights.items():
            i = industry.get(code)
            if i is None:
                i = UNCLASSIFIED
                unclassified.append(code)
            wp[i] = wp.get(i, 0.0) + float(w)
            r = rets.get(code)
            if r is None:
                missing.append(code)
                r = 0.0
            wr[i] = wr.get(i, 0.0) + float(w) * float(r)
        return wp, wr, missing, unclassified

    wp, wpr, miss_p, uncl_p = _agg(port)
    wb, wbr, miss_b, uncl_b = _agg(bench)
    industries: dict[str, dict] = {}
    for i in set(wp) | set(wb):
        wp_i, wb_i = wp.get(i, 0.0), wb.get(i, 0.0)
        rp_i = wpr.get(i, 0.0) / wp_i if wp_i > 0 else 0.0
        rb_i = wbr.get(i, 0.0) / wb_i if wb_i > 0 else 0.0
        alloc = (wp_i - wb_i) * rb_i
        select = wb_i * (rp_i - rb_i)
        inter = (wp_i - wb_i) * (rp_i - rb_i)
        industries[i] = {"alloc": round(alloc, 8), "select": round(select, 8),
                         "inter": round(inter, 8),
                         "total": round(alloc + select + inter, 8),
                         "wp": round(wp_i, 8), "wb": round(wb_i, 8),
                         "rp": round(rp_i, 8), "rb": round(rb_i, 8)}
    rp = sum(wpr.values())
    rb = sum(wbr.values())
    return {"industries": industries,
            "portfolio_return": round(rp, 8), "benchmark_return": round(rb, 8),
            "excess": round(rp - rb, 8),
            "unclassified_codes": sorted(set(uncl_p) | set(uncl_b)),
            "codes_without_return": sorted(set(miss_p) | set(miss_b))}


def _carino_k(r: float, b: float) -> float:
    """Carino 期系数 k_t = ln[(1+r)/(1+b)] / (r−b)（把对数超额换算成算术超额）。

    r=b 时取极限 1/(1+r)（= d/dr ln(1+r)）。
    ⚠️ 2026-10-29 修正：第一版写成了倒数 (r−b)/lnratio——Carino 恒等式
    Σ(k_t/K)·a_t = R−B 闭合不上（测试算出 0.0432 ≠ 0.0448 才暴露）。
    r 或 b ≤ −1（单期亏光）时对数无定义 ⇒ 退化取 1.0 并交由调用方的
    degraded 标记提示（这类期在 A 股组合里不该出现）。
    """
    if r <= -1.0 or b <= -1.0:
        return 1.0
    if abs(r - b) < 1e-12:
        return 1.0 / (1.0 + r)
    return (math.log1p(r) - math.log1p(b)) / (r - b)


def carino_link(period_effects: list[dict], portfolio_returns: list[float],
                benchmark_returns: list[float]) -> dict:
    """多期 Brinson 效应的 Carino 链接。

    Args:
        period_effects: 每期 `brinson_single_period` 的输出（按时间升序）。
        portfolio_returns / benchmark_returns: 每期组合/基准收益（小数）。

    Returns:
        {"total": {行业: {alloc, select, inter, total}},   # Σ = R − B（恒等式）
         "portfolio_return": R, "benchmark_return": B, "excess": R − B,
         "degraded_periods": [...]}
    恒等式 Σ_total = R − B 由测试锁死——链接系数写错时它第一个红。
    """
    if not period_effects:
        return {"total": {}, "portfolio_return": 0.0, "benchmark_return": 0.0,
                "excess": 0.0, "degraded_periods": []}
    R = 1.0
    B = 1.0
    for rp, rb in zip(portfolio_returns, benchmark_returns):
        # 单期 ≤ −100% = 清零：财富因子不能为负（负数会让 R<−1，log 爆定义域）
        R *= max(1.0 + float(rp), 0.0)
        B *= max(1.0 + float(rb), 0.0)
    R = max(R - 1.0, -0.999999)                          # log1p 定义域保护
    B = max(B - 1.0, -0.999999)
    if abs(R - B) > 1e-12:
        K = (math.log1p(R) - math.log1p(B)) / (R - B)
    else:
        K = 1.0 / (1.0 + R)
    degraded = []
    total: dict[str, dict] = {}
    for eff, rp, rb in zip(period_effects, portfolio_returns, benchmark_returns):
        if rp <= -1.0 or rb <= -1.0:
            degraded.append(f"{rp:.2f}/{rb:.2f}")
        k = _carino_k(float(rp), float(rb))
        scale = k / K
        for i, e in (eff.get("industries") or {}).items():
            slot = total.setdefault(i, {"alloc": 0.0, "select": 0.0,
                                        "inter": 0.0, "total": 0.0})
            for kk in ("alloc", "select", "inter", "total"):
                slot[kk] += float(e.get(kk) or 0.0) * scale
    for slot in total.values():
        for kk in slot:
            slot[kk] = round(slot[kk], 8)
    return {"total": total, "portfolio_return": round(R, 8),
            "benchmark_return": round(B, 8), "excess": round(R - B, 8),
            "degraded_periods": degraded}


def brinson_attribution(periods: list[dict], industry: dict | None,
                        snapshot_date: str | None = None) -> dict:
    """多期 Brinson 归因入口（行业映射 + 快照标记 + Carino 链接）。

    Args:
        periods: [{"date": d, "portfolio": {code: w}, "benchmark": {code: w},
                   "returns": {code: 当期收益}}]，时间升序。
                   ⚠️ `returns`（逐票）是**必需**的：没有它，行业内的
                   rp_i/rb_i 无从谈起，行业分解只能全 0——那不是降级，
                   是没有做归因。缺 returns 的期被剔除并计入 degraded
                   （Carino 恒等式 Σ=R−B 只对**全部**期成立，部分期参与
                   会把恒等式打破——所以要么全有，要么不给行业合计）。
        industry: {code: 行业名}；None ⇒ 全部进 `__未分类__`（数据没回填时
                   归因仍可运行，但只有总超额没有行业分解——缺就是缺）。

    Returns:
        {"periods": [每期分解（含 date）], "linked": carino 输出 | None,
         "industry_snapshot": snapshot_date, "degraded": {...}}
    """
    ind = dict(industry or {})
    period_effects = []
    skipped: list[str] = []
    unclassified_all: set[str] = set()
    for p in periods or []:
        rets = p.get("returns")
        if not rets:
            skipped.append(str(p.get("date")))
            continue
        eff = brinson_single_period(p.get("portfolio") or {},
                                    p.get("benchmark") or {}, ind, rets)
        eff = dict(eff, date=p.get("date"))
        unclassified_all.update(eff.get("unclassified_codes") or [])
        period_effects.append(eff)
    # Carino 恒等式要求**全部**期参与：有缺 returns 的期就不给链接合计
    linked = None
    if period_effects and not skipped:
        linked = carino_link(
            period_effects,
            [e["portfolio_return"] for e in period_effects],
            [e["benchmark_return"] for e in period_effects])
    return {"periods": period_effects, "linked": linked,
            "industry_snapshot": snapshot_date,
            "degraded": {"snapshot_lookahead": bool(snapshot_date),
                         "unclassified_codes": sorted(unclassified_all),
                         "n_unclassified": len(unclassified_all),
                         "periods_without_returns": skipped}}
