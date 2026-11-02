# -*- coding: utf-8 -*-
"""执行层折扣：把"理想 fwd20 收益"换成"T+1 开盘真能成交的收益"。

要回答的问题
────────────
前一轮的 A/B 实验里，A 臂押注段（2025-01~2026-10）超额 +7.5%（t=0.79）。
但那个数字用的是 `fwd20`（月末收盘→20 交易日后收盘），它假设：
  ① 月末当天收盘就能买到；
  ② 任何票都能买到（涨停也能买）；
  ③ 交易成本只按换手×固定 bp 估。
现实中这三条都不成立。**本脚本量化这三条的合计代价。**

三条建模（每条都对应一个真实约束）
──────────────────────────────────
1. **T+1 开盘成交**：月末 `d` 出信号，`d` 的**下一个交易日开盘**买入（不是 `d` 收盘）；
2. **涨停不可买**：开盘价 ≥ 前收 ×(1+涨幅) 就不买，**顺位买下一名**（这才是真实操作）；
   停牌（当日无 bar）同理跳过。涨幅按**板块 + 制度分界日**取 `zt_calendar.limit_ratio`；
3. **成本走实盘撮合**：`core.matching.buy_fill/sell_fill`（含滑点、佣金、印花税），
   不是自己拍一个 bp 数。

输出
────
同一批选股、同一段窗口下的三组年化对账，差值就是折扣：
  - **理想**：`fwd20` 等权（= A/B 实验用的口径）
  - **可成交**：T+1 开盘买入 + 涨停/停牌顺位 + 20 日后开盘卖出
  - **可成交+成本**：再扣撮合费用

已知限制（**不装作没有**）
──────────────────────────
- 面板无 `is_st` ⇒ 按非 ST 判涨幅。主板 ST 2026-07-06 前是 5%，
  本脚本会**少判**（把部分涨停当成可买）⇒ 折扣被**低估**，不是高估；
- 面板只含"当前在市"股票 ⇒ 幸存者偏差，三组都一样乐观，**差值仍可比**；
- 卖出遇跌停顺延最多 `MAX_DEFER` 个交易日，之后按当日收盘估（记账口径，非真实成交）。

用法
────
    py scripts/execution_discount.py                 # top=20，严格 TopN
    py scripts/execution_discount.py --top 5 --buffer 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.factor_composite import PROMOTED_FACTORS as POOL  # noqa: E402
from core.factor_composite import composite_score           # noqa: E402
from core.zt_calendar import limit_ratio                    # noqa: E402

SCAN_CACHE = _ROOT / "temp" / "factor_scan_scan.parquet"
BARS_CACHE = _ROOT / "temp" / "exec_bars.parquet"
OUT = _ROOT / "output"

LIQ_PCT = None       # 运行时由 --liq-pct 覆盖

HORIZON = 20            # 持有 20 个交易日
MAX_DEFER = 5           # 跌停卖不掉时最多顺延几个交易日
SUBSTITUTE_DEPTH = 3    # 顺位补人时最多往下看 top×3 名
NOTIONAL = 1_000_000.0  # 每只票的名义本金（算费用用，不影响收益率）

SEGMENTS = [
    ("开发段", "2019-03-01", "2023-12-31"),
    ("验证段", "2024-01-01", "2024-12-31"),
    ("押注段", "2025-01-01", "2026-12-31"),
]


def _load_scores(guard: bool = True, no_liq_floor: bool = False,
                 liq_pct: float | None = None) -> pd.DataFrame:
    global LIQ_PCT
    LIQ_PCT = liq_pct
    if not SCAN_CACHE.exists():
        raise SystemExit(f"缺少扫描缓存 {SCAN_CACHE}，先跑 factor_paper_scan.py")
    df = pd.read_parquet(SCAN_CACHE)
    if "month_ts" not in df.columns:
        df = df.rename(columns={"date": "month_ts"})
    df["month_ts"] = pd.to_datetime(df["month_ts"])
    df["code"] = df["code"].astype(str)
    need = [c for c in POOL if c in df.columns]
    comp = composite_score(df, need, POOL)
    comp[df[need].isna().any(axis=1)] = np.nan     # 口径同 A/B 实验：缺任一因子即落选
    df["composite"] = comp
    df = df.dropna(subset=["fwd20", "month_ts"])
    if guard:
        from core.universe_guard import eligible
        before = len(df)
        df = df[eligible(df)].copy()
        print(f"[守卫] {before:,} → {len(df):,} 行（剔除 {before - len(df):,}）", flush=True)

    # ── 流动性下限（2026-10-31）：**与线上 screener 同一条线** ──
    if not no_liq_floor:
        from core import liquidity_floor as _lf
        df["amount_yuan"] = np.exp(pd.to_numeric(df["lndvol20"], errors="coerce"))
        before = len(df)
        df, liq = _lf.filter_monthly_panel(df, pct=LIQ_PCT)
        print(f"[流动性下限] {before:,} → {len(df):,} 行（{liq['note']}）", flush=True)
    return df


def _load_bars(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """长表：code/date/open/close。有缓存就直接用（只补缺失的股票）。"""
    have = pd.DataFrame()
    if BARS_CACHE.exists():
        have = pd.read_parquet(BARS_CACHE)
        have = have[have["code"].isin(codes)]
    done = set(have["code"].unique()) if len(have) else set()
    todo = [c for c in codes if c not in done]
    frames = [have] if len(have) else []
    if todo:
        from data import bars
        for i, c in enumerate(todo, 1):
            d = bars.get_bars(c, start, end)
            if d is None or len(d) == 0:
                continue
            d = d[["date", "open", "close"]].copy()
            d["date"] = pd.to_datetime(d["date"])
            d["code"] = c
            frames.append(d)
            if i % 500 == 0:
                print(f"  取日线 {i}/{len(todo)}", flush=True)
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            merged.to_parquet(BARS_CACHE, index=False)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out.drop_duplicates(subset=["code", "date"], keep="last")


def _trade_one(code: str, cal: list[str], entry_idx: int,
               bars_by_code: dict[str, pd.DataFrame],
               slip_bp: float | None = None,
               notional: float | None = None) -> dict | None:
    """按 T+1 开盘买、20 日后开盘卖，返回成交明细。不可买返回 None。

    Args:
        slip_bp: **逐笔**滑点（bp）。None = 用 `core.matching.SLIPPAGE_RATE` 全局值。
            给了就把滑点折进成交价：买 `×(1+slip)`、卖 `×(1−slip)`，等价于
            `buy_fill` 内部那一步，从而支持"每只票不同滑点"（容量测算用）。
        notional: 单笔名义金额。默认 `NOTIONAL`（100 万/只）。容量测算要按
            真实资金规模算冲击，金额必须跟着变——冲击成本对金额是平方根敏感的，
            100 万和 20 万的差别不是 5 倍而是 √5 倍。
    """
    from core.matching import buy_fill, round_lot, sell_fill

    b = bars_by_code.get(code)
    if b is None or len(b) == 0:
        return None
    idx = b.index
    # ── 入场：日历当天必须有 bar，否则=停牌 ──
    entry_day = cal[entry_idx]
    if entry_day not in idx:
        return {"blocked": "停牌"}
    row = b.loc[entry_day]
    prev_rows = b[b.index < entry_day]
    if len(prev_rows) == 0:
        return None
    prev_close = float(prev_rows["close"].iloc[-1])
    entry_px = float(row["open"])
    if entry_px <= 0 or prev_close <= 0:
        return {"blocked": "无有效价"}
    # 面板无 is_st ⇒ 按非 ST 判（会少判涨停 ⇒ 折扣被低估，见模块 docstring）
    lim = limit_ratio(code, False, entry_day)
    if entry_px >= prev_close * (1 + lim) - 1e-6:
        return {"blocked": "开盘涨停"}

    # ── 出场：entry_idx + HORIZON，跌停卖不掉就顺延 ──
    exit_idx = entry_idx + HORIZON
    deferred = 0
    exit_px = None
    while exit_idx < len(cal) and deferred <= MAX_DEFER:
        day = cal[exit_idx]
        if day in idx:
            px = float(b.loc[day, "open"])
            pc_rows = b[b.index < day]
            if len(pc_rows):
                pc = float(pc_rows["close"].iloc[-1])
                if pc > 0 and px <= pc * (1 - limit_ratio(code, False, day)) + 1e-6:
                    deferred += 1
                    exit_idx += 1
                    continue
            exit_px = px
            break
        deferred += 1                       # 停牌，顺延
        exit_idx += 1
    if exit_px is None:                     # 到期仍卖不掉 ⇒ 按最后可用收盘记账
        avail = b[b.index <= cal[min(exit_idx, len(cal) - 1)]]
        if len(avail) == 0:
            return None
        exit_px = float(avail["close"].iloc[-1])
        deferred = MAX_DEFER + 1            # 标记为"记账价"

    # 逐笔滑点折进**成交价**（数学上等价于 buy_fill 内部的 signal×(1+SLIPPAGE)）。
    # ⚠️ `gross` 必须用**未调整**的价：它的语义是"可成交价的毛收益"，
    #    若连它一起含滑点，`tradable` 就不再有独立含义（两层折扣叠在一起算不清）。
    raw_entry, raw_exit = entry_px, exit_px
    if slip_bp:
        entry_px = entry_px * (1.0 + slip_bp / 10000.0)
        exit_px = exit_px * (1.0 - slip_bp / 10000.0)
    qty = round_lot((notional or NOTIONAL) / entry_px)
    if qty <= 0:
        # ⚠️ 小资金的核心约束就是这一条：一手（100 股）都买不起 ⇒ 这只票进不来。
        #    以前返回 None 被 `_run` 静默跳过，成交率下降但看不出原因。
        return {"blocked": "不足一手"}
    buy = buy_fill(entry_px, qty)
    sell = sell_fill(exit_px, qty)
    cost_buy = -float(buy["cash_delta"])    # 买入是现金流出
    net_sell = float(sell["total"])
    if cost_buy <= 0:
        return None
    return {"gross": raw_exit / raw_entry - 1.0,
            "net": net_sell / cost_buy - 1.0,
            "deferred": deferred}


def _impact_map(df: pd.DataFrame, notional: float, y: float = 1.0) -> dict:
    """(month, code) → 该月买入这只票的**按票估算**滑点（bp）。

    平方根冲击定律：`impact_bp = Y · σ_daily · sqrt(Q / V) · 1e4`
    —— 与 `scripts/capacity_estimate.py` 同一个公式、同一份数据（`lndvol20`/`vol20`），
    两处算出来必须一致，否则"容量测算说 14bp、回测用 50bp"又会自相矛盾。
    """
    from core.impact_model import impact_bp

    v = np.exp(pd.to_numeric(df["lndvol20"], errors="coerce"))
    sig = pd.to_numeric(df["vol20"], errors="coerce")
    bp = impact_bp(v, sig, float(notional), y=y)
    keys = list(zip(df["month_ts"], df["code"].astype(str)))
    return {k: float(v2) for k, v2 in zip(keys, bp) if np.isfinite(v2)}


def _participation_map(df: pd.DataFrame, notional: float) -> dict:
    """(month, code) → 下单金额占该票日均成交额的比例。

    ⚠️ 这一列是为了回答"**买不进的票不该算收益**"（2026-10-31 实测）：
       2023 年 top5 里大量 920xxx（北交所）日均成交额只有 16~200 万，
       20 万下单就是 10%~100% 的参与率——实盘根本拿不到这个价。
       之前的回测把这类票当成正常成交，等于给策略发了**免费容量**。
    """
    v = np.exp(pd.to_numeric(df["lndvol20"], errors="coerce"))
    part = float(notional) / v.replace(0.0, np.nan)
    keys = list(zip(df["month_ts"], df["code"].astype(str)))
    return {k: float(p2) for k, p2 in zip(keys, part) if np.isfinite(p2)}


def _run(df: pd.DataFrame, top: int, buffer: float,
         impact: dict | None = None,
         notional: float | None = None,
         part: dict | None = None,
         max_part: float | None = None) -> dict:
    from core.matching import round_lot  # noqa: F401  仅确保依赖可用

    months = sorted(df["month_ts"].unique())
    # 每月按分数降序的完整候选顺序（顺位补人要用）
    order: dict[pd.Timestamp, list[str]] = {}
    for m in months:
        g = df[df["month_ts"] == m].dropna(subset=["composite"])
        if len(g):
            order[m] = g.sort_values("composite", ascending=False)["code"].tolist()

    depth = min(top * SUBSTITUTE_DEPTH, max((len(v) for v in order.values()), default=top))
    codes = sorted({c for v in order.values() for c in v[:depth]})
    print(f"候选 {len(codes):,} 只（每月前 {depth} 名并集）→ 取日线", flush=True)
    bars = _load_bars(codes, "2019-01-01", "2026-12-31")
    bars_by_code = {c: g.set_index("date").sort_index()
                    for c, g in bars.groupby("code", sort=False)}
    # ⚠️ 统一成 pd.Timestamp：numpy.datetime64 与 DatetimeIndex 的 `in` 判定不可靠，
    #    会让"停牌"被误判成"有 bar"或反之。
    cal = [pd.Timestamp(d) for d in sorted(set(bars["date"].unique()))]
    print(f"日线 {len(bars):,} 行 / {len(bars_by_code):,} 只 / 日历 {len(cal)} 天", flush=True)

    ideal, tradable, net, fills, blocks, bench = {}, {}, {}, {}, {}, {}
    prev: list[str] = []
    for m in months:
        ranked = order.get(m, [])
        if not ranked or pd.Timestamp(m) >= pd.Timestamp(cal[-1]):
            continue
        # 入场日 = 月末之后的第一根日历日
        later = [i for i, d in enumerate(cal) if d > pd.Timestamp(m)]
        if not later:
            continue
        ei = later[0]
        # 只交易日历里在 entry 之后还有 HORIZON 天的月份（否则收益未实现）
        if ei + HORIZON >= len(cal):
            continue

        # 基准：**同一个守卫宇宙**内的等权 fwd20（否则"超额"会拿污染宇宙做分母）
        bench[m] = float(df[df["month_ts"] == m]["fwd20"].mean())

        # 理想口径：该月前 top 名的 fwd20 等权（= A/B 实验用的数）
        g = df[(df["month_ts"] == m) & (df["code"].isin(ranked[:top]))]
        if len(g):
            ideal[m] = float(g["fwd20"].mean())

        # 缓冲区：旧持仓仍在 top×buffer 内就留（与 select_with_buffer 同规则）
        if buffer > 1.0 and prev:
            head = set(ranked[:max(top, int(round(top * buffer)))])
            kept = [c for c in prev if c in head]
            chosen = kept + [c for c in ranked[:depth] if c not in set(kept)][:top - len(kept)]
        else:
            chosen = ranked[:depth]

        got, gross_list, net_list, blk = [], [], [], {}
        for c in chosen:
            if len(got) >= top:
                break
            if max_part and part is not None:
                p_now = part.get((m, str(c)))
                if p_now is not None and p_now > max_part:
                    # 与"涨停买不进"同一处理：顺位补下一名，而不是硬买
                    blk["参与率超限"] = blk.get("参与率超限", 0) + 1
                    continue
            r = _trade_one(c, cal, ei, bars_by_code,
                           slip_bp=(impact or {}).get((m, str(c))),
                           notional=notional)
            if r is None or "blocked" in r:
                if r and "blocked" in r:
                    blk[r["blocked"]] = blk.get(r["blocked"], 0) + 1
                continue
            got.append(c)
            gross_list.append(r["gross"])
            net_list.append(r["net"])
        if got:
            tradable[m] = float(np.mean(gross_list))
            net[m] = float(np.mean(net_list))
            fills[m] = len(got) / top
            blocks[m] = blk
        prev = got

    return {"bench": pd.Series(bench).sort_index(),
            "ideal": pd.Series(ideal).sort_index(),
            "tradable": pd.Series(tradable).sort_index(),
            "net": pd.Series(net).sort_index(),
            "fill_rate": pd.Series(fills).sort_index(),
            "blocks": blocks}


def _metrics(r: pd.Series) -> dict:
    if len(r) < 6:
        return {"ok": False, "months": int(len(r))}
    ann = float((1 + r).prod() ** (12 / len(r)) - 1)
    vol = float(r.std(ddof=1) * math.sqrt(12))
    nav = (1 + r).cumprod()
    return {"ok": True, "months": int(len(r)), "ann": round(ann, 4),
            "sharpe": round(ann / vol, 3) if vol > 0 else None,
            "mdd": round(float((nav / nav.cummax() - 1).min()), 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--buffer", type=float, default=1.0)
    ap.add_argument("--no-guard", action="store_true",
                    help="关掉宇宙守卫（仅用于对比污染前后，结果不可用于决策）")
    ap.add_argument("--no-liq-floor", action="store_true",
                    help="关掉流动性下限（对比线上会剔掉哪些票用；默认与线上一致=开）")
    ap.add_argument("--liq-pct", type=float, default=None,
                    help="流动性下限分位（默认 0.30 = 线上口径）。小资金应测 0.10/0.20："
                         "剔太多会把收益来源一起砍掉")
    ap.add_argument("--max-participation", type=float, default=None,
                    help="单笔下单金额占日均成交额的上限（如 0.05 = 5%%）。"
                         "超限的票视为**买不进**，顺位补下一名 —— 这才是"
                         "'这个价位我们根本拿不到'的正确处理方式")
    ap.add_argument("--capacity-slippage", action="store_true",
                    help="逐笔按票算滑点（平方根冲击），替代固定 bp；配合 --nav 使用")
    ap.add_argument("--nav-grid", default=None,
                    help="逗号分隔的资金总额，如 30000,100000,500000："
                         "扫出\"资金规模 → 年化超额\"曲线（小资金能吃低流动性端）")
    ap.add_argument("--nav", type=float, default=None,
                    help="总资金（元）。每只分摊 nav/top。容量测算必须给，"
                         "否则默认按 100 万/只（那是 500 万组合的口径）")
    ap.add_argument("--slippage-bp", type=float, default=None,
                    help="覆盖滑点（单边 bp）。默认用 core.matching 的 5bp；"
                         "微盘小组合应做 20/50bp 敏感性——100 万打进日成交 1000 万的票，"
                         "5bp 显然不够")
    a = ap.parse_args(argv)

    if a.slippage_bp is not None:
        # ⚠️ 直接改模块常量：`buy_fill`/`sell_fill` 读的是模块级 SLIPPAGE_RATE。
        #    这是**有意的**敏感性开关（不是生产路径），所以就地覆盖并打印出来，
        #    免得看到数字的人以为用的还是 5bp。
        from core import matching as _m
        _m.SLIPPAGE_RATE = float(a.slippage_bp) / 10000.0
        print(f"⚠️ 滑点覆盖为 单边 {a.slippage_bp:.0f}bp（默认 {0.0005 * 10000:.0f}bp）",
              flush=True)

    df = _load_scores(guard=not a.no_guard, no_liq_floor=a.no_liq_floor,
                      liq_pct=a.liq_pct)
    print(f"截面 {len(df):,} 行 / {df['month_ts'].nunique()} 月 / top={a.top} "
          f"buffer={a.buffer}", flush=True)
    # ── 资金规模网格（2026-10-31）：回答"多小的钱能吃到低流动性那一端" ──
    if a.nav_grid:
        return _run_grid(df, a)

    impact = None
    notional = (a.nav / a.top) if a.nav else None
    if a.capacity_slippage:
        if not notional:
            notional = 1_000_000.0 / a.top          # 默认 100 万总资金
        # ⚠️ 逐笔滑点走成交价，这里必须把全局滑点归零，否则 5bp 会**叠加**上去
        from core import matching as _m2
        _m2.SLIPPAGE_RATE = 0.0
        print(f"⚠️ 按票滑点模式：全局滑点归零，逐笔用平方根冲击"
              f"（每只 {notional / 1e4:.1f} 万）", flush=True)
    part = _participation_map(df, notional) if a.max_participation else None
    if part is not None:
        print(f"⚠️ 参与率上限 {a.max_participation:.1%}：超限的票视为买不进，"
              f"顺位补下一名", flush=True)

    rep = _evaluate(df, top=a.top, buffer=a.buffer, notional=notional,
                    max_part=a.max_participation)
    report, allb, fill_mean = rep["segments"], rep["blocks"], rep["fill_rate"]
    _print_segments(report)
    print(f"\n拦截合计（按候选顺序尝试的每一次）：{allb}")
    print(f"平均成交率 {fill_mean:.2%}（目标 {a.top} 只）")


    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    p = OUT / f"执行折扣_{stamp}.json"
    p.write_text(json.dumps({"top": a.top, "buffer": a.buffer,
                             "capacity_slippage": bool(a.capacity_slippage),
                             "max_participation": a.max_participation,
                             "notional_per_name": notional,
                             "slippage_bp": a.slippage_bp,
                             "segments": report, "blocks": allb,
                             "fill_rate_mean": round(fill_mean, 4),
                             "series": rep["series"]},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"JSON: {p}")
    return 0



# ================================================================
# 资金规模网格（2026-10-31）
# ================================================================
# 为什么要做这个：用 100 万测出来"两条路都堵死"，但那只是**一个资金规模**下的结论。
# 小资金能吃下低流动性那一端（参与率低），而低流动性溢价正是这套因子的全部来源。
# 所以"能不能赚钱"不是二值问题，是**资金规模的函数**——必须扫一遍。

def _evaluate(df: pd.DataFrame, *, top: int, buffer: float,
              notional: float | None, max_part: float | None) -> dict:
    """跑一个资金规模，返回各段指标 / 拦截统计 / 平均成交率。"""
    impact = _impact_map(df, notional) if notional else None
    part = _participation_map(df, notional) if max_part else None
    res = _run(df, top, buffer, impact=impact, notional=notional,
               part=part, max_part=max_part)

    segments = {}
    for name, s0, e0 in SEGMENTS:
        lo, hi = pd.Timestamp(s0), pd.Timestamp(e0)
        pick = lambda ser: ser[(ser.index >= lo) & (ser.index <= hi)]  # noqa: E731
        mb, mi, mt, mn = (_metrics(pick(res["bench"])), _metrics(pick(res["ideal"])),
                          _metrics(pick(res["tradable"])), _metrics(pick(res["net"])))
        fr = pick(res["fill_rate"])
        disc = ((mi.get("ann", 0) - mn.get("ann", 0)) * 10000) \
            if mi.get("ok") and mn.get("ok") else None
        ex = ((mn.get("ann", 0) - mb.get("ann", 0)) * 100) \
            if mn.get("ok") and mb.get("ok") else None
        segments[name] = {
            "bench": mb, "ideal": mi, "tradable": mt, "net": mn,
            "excess_after_cost": round(ex, 2) if ex is not None else None,
            "discount_bp": round(disc, 0) if disc is not None else None,
            "fill_rate": round(float(fr.mean()), 3) if len(fr) else None}

    blocks: dict[str, int] = {}
    for blk in res["blocks"].values():
        for k, v in blk.items():
            blocks[k] = blocks.get(k, 0) + v
    return {"segments": segments, "blocks": blocks,
            "fill_rate": float(res["fill_rate"].mean()),
            "series": {
                "ideal": {str(k.date()): round(float(v), 5)
                          for k, v in res["ideal"].items()},
                "net": {str(k.date()): round(float(v), 5)
                        for k, v in res["net"].items()}}}


def _print_segments(report: dict) -> None:
    print(f"\n{'段':<8}{'基准':>9}{'理想':>9}{'可成交':>9}{'扣成本':>9}"
          f"{'超额(扣成本后)':>14}{'折扣(bp)':>10}{'成交率':>8}")
    for name, _s0, _e0 in SEGMENTS:
        r = report.get(name) or {}
        if not r:
            continue
        mb, mi, mt, mn = r["bench"], r["ideal"], r["tradable"], r["net"]
        fr, ex, disc = r["fill_rate"], r["excess_after_cost"], r["discount_bp"]
        print(f"{name:<8}{(mb.get('ann', 0) * 100):>8.2f}%{(mi.get('ann', 0) * 100):>8.2f}%"
              f"{(mt.get('ann', 0) * 100):>8.2f}%{(mn.get('ann', 0) * 100):>8.2f}%"
              f"{(ex if ex is not None else 0):>13.2f}%"
              f"{(disc if disc is not None else 0):>10.0f}"
              f"{(fr if fr is not None else 0):>8.2%}")


def _run_grid(df: pd.DataFrame, a) -> int:
    """扫资金规模，打印"资金 → 超额"曲线。"""
    navs = [float(x) for x in str(a.nav_grid).split(",") if x.strip()]
    if not navs:
        print("--nav-grid 需要逗号分隔的金额，如 30000,100000,1000000", flush=True)
        return 2

    from core import matching as _m
    if _m.SLIPPAGE_RATE != 0.0:
        _m.SLIPPAGE_RATE = 0.0      # 用按票滑点，全局必须归零（否则 5bp 叠加）
    max_part = a.max_participation if a.max_participation else None

    print(f"资金规模网格：top={a.top}｜参与率上限 "
          f"{f'{max_part:.0%}' if max_part else '不限'}｜"
          f"流动性下限{'关（小资金能吃小票）' if a.no_liq_floor else '开（与线上一致）'}",
          flush=True)
    print(f"\n{'资金':>10}{'每只':>10}{'开发段':>10}{'验证段':>10}{'押注段':>10}"
          f"{'成交率':>9}  主要拦截")

    out = []
    for nav in navs:
        notional = nav / a.top
        rep = _evaluate(df, top=a.top, buffer=a.buffer, notional=notional,
                        max_part=max_part)
        segs = rep["segments"]
        cells = []
        for name, _s, _e in SEGMENTS:
            ex = (segs.get(name) or {}).get("excess_after_cost")
            cells.append(f"{ex:+.2f}%" if ex is not None else "  n/a ")
        top_block = sorted(rep["blocks"].items(), key=lambda kv: -kv[1])[:2]
        block_txt = "、".join(f"{k}×{v}" for k, v in top_block) or "—"
        print(f"{nav / 1e4:>9.1f}万{notional / 1e4:>9.2f}万"
              + "".join(f"{c:>10}" for c in cells)
              + f"{rep['fill_rate']:>9.1%}  {block_txt}")
        out.append({"nav": nav, "notional_per_name": round(notional, 2),
                    "segments": segs, "blocks": rep["blocks"],
                    "fill_rate": round(rep["fill_rate"], 4)})

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = OUT / f"资金规模网格_{stamp}.json"
    path.write_text(json.dumps({"top": a.top, "buffer": a.buffer,
                                "max_participation": max_part,
                                "liq_floor": not a.no_liq_floor,
                                "grid": out}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\nJSON: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
