# -*- coding: utf-8 -*-
"""商品链板块：外盘商品 → A 股资源股。

承接 `validate_crossmarket_sector.py` 的发现——**能源是唯一在日频就显著的板块**
（XLE→石油石化 r=0.173, t=9.0），机制是"油价全球统一定价，A 股石化与美股能源
共享同一个外生价格，没有本土特色可稀释"。

那么同理可推：黄金、铜、铝、锂、煤这些**资源品**应该也偏高，而稀土这种
**中国定价**的品种应该没有外生锚。本脚本就是把这条推论逐个量出来。

三个问题分开答（混在一起会得出错误结论）：

  A. 美股商品 ETF → A 股资源板块，日/周/月三频率相关
     —— 回答"两个市场是不是同一个产业周期"
  B. 对手盘怎么选：金属本身（GLD）vs 矿业股（GDX）
     —— 回答"该拿哪个当信号源"（§10.2 已证明 SOXX 不如纳指，这里要复核）
  C. **国内期货夜盘 → A 股当日**
     —— 这是唯一真正可交易的窗口：期货夜盘 21:00~次日 2:30，
        A 股 9:15 集合竞价、9:30 开盘。夜盘收益在 A 股开盘前就已确定。

⚠️ 两个已知口径问题（与 §5 一致，不在这里偷偷修掉，只在结论里说明保守方向）：
  1. 期货主力连续换月有跳空，直接 `pct_change` 会把价差当收益；
     这里用 `open/prev_close` 与 `close/open` 分开看，换月影响集中在后者。
  2. 期货"开盘价"是 9:00 日盘开盘，**已包含夜盘**；故 `open/prev_close`
     衡量的是"隔夜+夜盘"的合计，不是纯夜盘。方向不变，量级略高估。

数据：
  · 美股：`ak.stock_us_daily(symbol=...)`（新浪；东财 `stock_us_hist` 已 RST）
  · 国内期货：`ak.futures_main_sina(symbol='AU0')`（新浪主力连续）
  · A 股：本地 `panel` 面板概念成分等权

复现：`python scripts/validate_crossmarket_commodity.py`
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import concept_members as cm  # noqa: E402
from data import panel_store  # noqa: E402
from scripts.validate_crossmarket_sector import board_ohlc, corr, us_close  # noqa: E402

_logger = logging.getLogger(__name__)

OUT_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output", "跨市场商品链检验_2026-10-29.json")

# (标签, A 股概念名, 美股 ETF 列表, 国内期货代码)
# ETF 列表里放多个是为了回答"金属本身 vs 矿业股"哪个更像 A 股资源股
CHAINS = [
    ("黄金", "黄金概念", ["GLD", "GDX"], "AU0"),
    ("铜", "铜", ["CPER", "COPX"], "CU0"),
    ("铝", "铝", ["XME"], "AL0"),
    ("锂", "锂矿概念", ["LIT"], "LC0"),
    ("煤炭", "煤炭", [], "JM0"),
    ("稀土", "稀土永磁", ["XME"], None),
    ("原油", "石油石化", ["USO"], "SC0"),
]
# 对照：A 股黄金有两只概念（黄金概念 74 只矿业股 / 黄金 10 只纯金矿）
GOLD_PURE = ("黄金(纯10只)", "黄金", "GLD")


def futures_ohlc(symbol: str) -> pd.DataFrame | None:
    """期货主力连续 → overnight（含夜盘）/ intraday（日盘）/ oc。

    ⚠️ `开盘价` 是 9:00 日盘开盘，**已包含 21:00~2:30 的夜盘**，
    所以 overnight 是"隔夜 + 夜盘"合计，是本脚本最关心的可交易前置信息。
    """
    import akshare as ak
    try:
        df = ak.futures_main_sina(symbol=symbol)
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[商品] 期货 %s 拉取失败: %s: %s", symbol, type(e).__name__, e)
        print(f"  [期货] {symbol} 拉取失败: {type(e).__name__}: {str(e)[:60]}")
        return None
    if df is None or df.empty:
        return None
    d = pd.DataFrame({
        "date": pd.to_datetime(df["日期"]),
        "open": pd.to_numeric(df["开盘价"], errors="coerce"),
        "close": pd.to_numeric(df["收盘价"], errors="coerce"),
    }).dropna().set_index("date").sort_index()
    out = pd.DataFrame(index=d.index)
    out["overnight"] = d["open"] / d["close"].shift(1) - 1.0
    out["intraday"] = d["close"] / d["open"] - 1.0
    out["oc"] = d["close"] / d["close"].shift(1) - 1.0
    return out * 100                            # 统一转 %


def clean_us_ret(px: pd.Series, thresh_pct: float = 30.0,
                 label: str = "") -> tuple[pd.Series, int]:
    """美股 ETF 日收益（**单位 %**）+ 剔除公司行动（拆股/并股）造成的假跳变。

    ⚠️⚠️ 实测事故：**新浪 `stock_us_daily` 不复权**。USO 在 2020-04-29 做 1:8 反向拆股，
    收盘价 2.13 → 18.00，`pct_change()` 得到 **+745%**。只要这 1 个点混在 5,170 个样本里，
    USO→石油石化的日频相关就从 0.109 被压成 **0.015**（看起来像"原油与石化毫无关系"），
    而同一板块换 XLE 是 0.192。SOXX/IGV（2024-03-07）、XBI（2015-09-11）、
    XLE（2025-12-05）、CPER（3 个点）同样中招。
    ⇒ 任何用新浪美股算收益的地方，都必须先剔掉 |r|>30% 的点（宽基/商品 ETF 单日
      真实波动不可能超 30%；2020-03-09 原油崩盘也只有 −25.3%）。

    ⚠️ 单位一致性：本函数返回**百分数**，阈值故为 `30.0`。
    写成 `0.30` 会把所有涨跌 >0.3% 的正常交易日判成拆股（已踩过一次：误剔 3,258 天）。
    """
    r = px.pct_change() * 100
    bad = r.abs() > thresh_pct
    n_bad = int(bad.sum())
    if n_bad:
        _logger.warning("[商品] %s 剔除 %d 个疑似公司行动点: %s",
                        label or "US-ETF", n_bad,
                        [str(d.date()) for d in r[bad].index][:5])
    return r.mask(bad), n_bad


def nw_t(x, lags: int = 5) -> float:
    """Newey-West 稳健 t（日收益常有自相关，普通 t 会偏）。"""
    e = np.asarray(x, float)
    e = e[np.isfinite(e)]
    n = len(e)
    if n < 30:
        return float("nan")
    mu = e.mean()
    d = e - mu
    s = (d @ d) / n
    for lag in range(1, lags + 1):
        s += 2.0 * (1.0 - lag / (lags + 1.0)) * ((d[lag:] @ d[:-lag]) / n)
    se = np.sqrt(max(s, 1e-18) / n)
    return float(mu / se)


def tf_corr(us_ret_pct: pd.Series, board_pct: pd.Series,
            freq: str) -> tuple[float, int]:
    """同一频率上的相关：日 / 周(W-FRI) / 月(ME)。

    ⚠️ 单位：入参两列都是**百分数**，函数内部统一折成小数再复利。
    曾经这里写成 `(1 + ur)` 而 `ur` 已改成百分数 ⇒ 每周被当成 +520% 复合，
    周频/月频相关直接变成负的（黄金 −0.036 / 铜 0.055）。复利前必须先 /100。
    """
    ur = (us_ret_pct / 100.0).dropna()
    br = (board_pct / 100.0).dropna()
    if freq != "D":
        rule = "W-FRI" if freq == "W" else "ME"
        ur = (1 + ur).resample(rule).prod() - 1
        br = (1 + br).resample(rule).prod() - 1
    r, n, _t = corr(ur, br)
    return r, n


def lead_slice(sig: pd.Series, board: pd.DataFrame, shift: int = 1) -> dict:
    """把 sig 对齐到 A 股 T 日，返回 gap/intraday/cc 的 r 与 t。

    `shift` 必须按"信息何时可得"分别设，不能一律用 1：
      · 美股（`shift=1`）：美股日期 T-1 的收盘发生在北京时间 T 日凌晨，
        到 A 股 T 日开盘时已知 ⇒ 需右移一天。
      · **国内期货夜盘（`shift=0`）**：`overnight = open[T]/close[T-1]`，
        其中 open 是 T 日 9:00 日盘开盘、close 是 T-1 日 15:00，
        中间的夜盘 21:00~T 日 2:30 全在 A 股 T 日 9:30 开盘之前 ⇒ **同日对齐，不能 shift**。
        这里若误用 shift=1，等于拿前一天的夜盘去测，结论会完全失真。
    """
    prev = (sig.reindex(sig.index.union(board.index)).ffill()
            .reindex(board.index).shift(shift))
    idx = prev.dropna().index.intersection(board.dropna(subset=["cc"]).index)
    s = np.asarray(prev.reindex(idx), float)
    out = {"n": int(len(idx)), "signal": s}
    for col in ("gap", "intraday", "cc"):
        y = np.asarray(board[col].reindex(idx), float)
        r, n, t = corr(pd.Series(s), pd.Series(y))
        out[f"{col}_r"] = round(r, 3) if r == r else None
        out[f"{col}_t"] = round(t, 1) if t == t else None
    out["_idx"] = idx
    return out


def main() -> int:
    concepts = cm.get_concept_members()["concepts"]
    report = {"generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
              "chains": [], "futures": [], "metal_vs_miner": []}

    # ---------------- A. 美股商品 ETF → A 股资源板块 ----------------
    print("=" * 116)
    print("A. 美股商品 ETF（T-1 收盘）→ A 股资源板块（T 日）")
    print("=" * 116)
    # ⚠️ 方向必须写清（否则就是"列名同≠口径同"）：
    #   · `同日` = 美股 t 与 A 股 t 的同一日历日 ⇒ 美股 t 要到北京 t 日 21:30 才开盘，
    #     故这一列实为 **A 股 t 先行 → 美股 t**，衡量"谁先定价"，不是传导。
    #   · `全天 r` = **美股 t-1 → A 股 t**，才是真正的跨市场传导。
    #   两者之差就是定价权方向（见 §11.4）。
    print(f"{'商品':6s}{'A股概念':10s}{'只数':>5s}{'ETF':7s}"
          f"{'同日A→美':>9s}{'周频':>7s}{'月频':>7s}"
          f"{'跳空 r':>8s}{'日内 r':>8s}{'美→A全天':>9s}{'t':>6s}")

    for label, cname, etfs, _fut in CHAINS:
        codes = (concepts.get(cname) or {}).get("codes") or []
        board = board_ohlc(codes)
        if board is None:
            print(f"{label:6s}{cname:10s} 跳过（成分不足）")
            continue
        for etf in etfs:
            px = us_close(etf)
            if px is None:
                continue
            ur, n_bad = clean_us_ret(px, label=etf)
            rd, nd = tf_corr(ur, board["cc"], "D")
            rw, _nw = tf_corr(ur, board["cc"], "W")
            rm, _nm = tf_corr(ur, board["cc"], "M")
            sl = lead_slice(ur, board)
            print(f"{label:6s}{cname:10s}{len(codes):5d}{etf:7s}"
                  f"{rd:7.3f}{rw:7.3f}{rm:7.3f}"
                  f"{sl['gap_r'] if sl['gap_r'] is not None else float('nan'):8.3f}"
                  f"{sl['intraday_r'] if sl['intraday_r'] is not None else float('nan'):8.3f}"
                  f"{sl['cc_r'] if sl['cc_r'] is not None else float('nan'):8.3f}"
                  f"{sl['cc_t'] if sl['cc_t'] is not None else float('nan'):7.1f}")
            report["chains"].append({
                "label": label, "concept": cname, "codes": len(codes), "etf": etf,
                "a_lead_us": round(rd, 3), "weekly": round(rw, 3), "monthly": round(rm, 3),
                "gap_r": sl["gap_r"], "intraday_r": sl["intraday_r"],
                "cc_r": sl["cc_r"], "cc_t": sl["cc_t"], "n": sl["n"],
                "corporate_action_points": n_bad})

    # 黄金：纯金矿（10 只）vs 黄金概念（74 只）
    label, cname, etf = GOLD_PURE
    codes = (concepts.get(cname) or {}).get("codes") or []
    board = board_ohlc(codes, min_codes=2)
    px = us_close(etf)
    if board is not None and px is not None:
        ur, _nb = clean_us_ret(px, label=etf)
        rd, _ = tf_corr(ur, board["cc"], "D")
        sl = lead_slice(ur, board)
        print(f"{label:6s}{cname:10s}{len(codes):5d}{etf:7s}"
              f"{rd:7.3f}{'':7s}{'':7s}{sl['gap_r']:8.3f}{sl['intraday_r']:8.3f}"
              f"{sl['cc_r']:8.3f}{sl['cc_t']:7.1f}")

    # ---------------- B. 金属本身 vs 矿业股 ----------------
    print()
    print("=" * 116)
    print("B. 对手盘选择：金属本身 vs 矿业股（同一 A 股概念，换信号源）")
    print("=" * 116)
    for label, cname, a_etf, b_etf in (("黄金", "黄金概念", "GLD", "GDX"),
                                       ("铜", "铜", "CPER", "COPX")):
        codes = (concepts.get(cname) or {}).get("codes") or []
        board = board_ohlc(codes)
        if board is None:
            continue
        pa, pb = us_close(a_etf), us_close(b_etf)
        if pa is None or pb is None:
            continue
        ra_ret, _ = clean_us_ret(pa, label=a_etf)
        rb_ret, _ = clean_us_ret(pb, label=b_etf)
        ra, _ = tf_corr(ra_ret, board["cc"], "D")
        rb, _ = tf_corr(rb_ret, board["cc"], "D")
        sa = lead_slice(ra_ret, board)
        sb = lead_slice(rb_ret, board)
        # 两个 ETF 之间的共线程度：解释"为什么矿业股不一定更准"
        rr, nr, _ = corr(ra_ret, rb_ret)
        print(f"  【{label}→{cname}】{a_etf} 与 {b_etf} 日频相关 {rr:.3f} (n={nr})")
        print(f"    {a_etf:6s} 日频 {ra:+.3f}  跳空 {sa['gap_r']:+.3f}  "
              f"日内 {sa['intraday_r']:+.3f}  全天 {sa['cc_r']:+.3f} (t={sa['cc_t']})")
        print(f"    {b_etf:6s} 日频 {rb:+.3f}  跳空 {sb['gap_r']:+.3f}  "
              f"日内 {sb['intraday_r']:+.3f}  全天 {sb['cc_r']:+.3f} (t={sb['cc_t']})")
        report["metal_vs_miner"].append({
            "label": label, "concept": cname,
            "etf_a": a_etf, "daily_a": round(ra, 3), "gap_a": sa["gap_r"],
            "cc_a": sa["cc_r"], "t_a": sa["cc_t"],
            "etf_b": b_etf, "daily_b": round(rb, 3), "gap_b": sb["gap_r"],
            "cc_b": sb["cc_r"], "t_b": sb["cc_t"],
            "pair_corr": round(rr, 3)})

    # ---------------- C. 国内期货夜盘 → A 股当日 ----------------
    print()
    print("=" * 116)
    print("C. 国内期货夜盘（21:00~次日2:30，A 股开盘前已知）→ A 股当日")
    print("=" * 116)
    print(f"{'商品':6s}{'期货':6s}{'A股概念':10s}{'n':>6s}"
          f"{'夜盘→跳空 r':>13s}{'夜盘→日内 r':>13s}{'夜盘→日内 t':>13s}"
          f"{'日盘 vs 日内 r':>15s}")
    for label, cname, _etfs, fut in CHAINS:
        if not fut:
            continue
        codes = (concepts.get(cname) or {}).get("codes") or []
        board = board_ohlc(codes)
        fo = futures_ohlc(fut)
        if board is None or fo is None:
            print(f"{label:6s}{fut:6s}{cname:10s} 跳过")
            continue
        # 期货夜盘（overnight）→ A 股 T 日：同日对齐（夜盘在 A 股开盘前结束）
        sl = lead_slice(fo["overnight"], board, shift=0)
        # 期货日盘（9:00~15:00）与 A 股日内是**同时段**：这是映射正确性检验
        idx = fo["intraday"].dropna().index.intersection(
            board["intraday"].dropna().index)
        rs, ns, _ts = corr(fo["intraday"].reindex(idx), board["intraday"].reindex(idx))
        # 夜盘 → A 股日内 的收益显著性（不只是相关）
        prev = (fo["overnight"].reindex(
            fo["overnight"].index.union(board.index)).ffill().reindex(board.index))
        ii = prev.dropna().index.intersection(board["intraday"].dropna().index)
        sg = np.asarray(prev.reindex(ii), float)
        it = np.asarray(board["intraday"].reindex(ii), float)
        m = np.isfinite(sg) & np.isfinite(it)
        sg, it = sg[m], it[m]
        upt = it[sg > 0]
        dnt = it[sg < 0]
        t_up = nw_t(upt) if len(upt) > 30 else float("nan")
        t_dn = nw_t(dnt) if len(dnt) > 30 else float("nan")
        print(f"{label:6s}{fut:6s}{cname:10s}{sl['n']:6d}"
              f"{sl['gap_r']:13.3f}{sl['intraday_r']:13.3f}"
              f"{sl['intraday_t']:13.1f}{rs:15.3f}")
        print(f"       夜盘涨({len(upt)}天) → 日内均值 {np.mean(upt):+.3f}% (NW t={t_up:.2f})"
              f" ｜ 夜盘跌({len(dnt)}天) → {np.mean(dnt):+.3f}% (NW t={t_dn:.2f})")
        report["futures"].append({
            "label": label, "futures": fut, "concept": cname, "n": sl["n"],
            "overnight_to_gap": sl["gap_r"], "overnight_to_intraday": sl["intraday_r"],
            "overnight_to_intraday_t": sl["intraday_t"],
            "day_session_vs_intraday": round(rs, 3),
            "up_days": int(len(upt)), "up_intraday_mean": round(float(np.mean(upt)), 3),
            "up_nw_t": round(t_up, 2) if t_up == t_up else None,
            "down_days": int(len(dnt)), "down_intraday_mean": round(float(np.mean(dnt)), 3),
            "down_nw_t": round(t_dn, 2) if t_dn == t_dn else None})

    # ---------------- D. 开盘调整是否充分（残差检验） ----------------
    # 前面所有证据都指向"信息在跳空里消化"，但没说清是**刚好消化**还是**调整不足**。
    # 只有"调整不足 ⇒ 日内补涨"才存在可交易信号：
    #   9:25 集合竞价时，期货夜盘 s 与 A 股集合竞价指示价 ĝ 都已可见；
    #   若 ĝ 相对 s 明显偏小，则 9:30 开盘买入、收盘卖出可吃到补涨。
    # 做法：全样本回归 gap = α + β·s，取残差 ε = gap − (α + β·s)；
    #   若 corr(ε, intraday) < 0，说明开盘"过头/不足"会在日内被纠正。
    print()
    print("=" * 116)
    print("D. 开盘调整是否充分：gap 对夜盘回归的残差 ε，再与日内收益相关")
    print("=" * 116)
    print(f"{'商品':6s}{'期货':6s}{'β(夜盘→跳空)':>14s}{'R²':>8s}"
          f"{'corr(ε,日内)':>14s}{'ε t':>8s}"
          f"{'ε最低组日内%':>14s}{'ε最高组日内%':>14s}")
    for label, cname, _etfs, fut in CHAINS:
        if not fut:
            continue
        codes = (concepts.get(cname) or {}).get("codes") or []
        board = board_ohlc(codes)
        fo = futures_ohlc(fut)
        if board is None or fo is None:
            continue
        sig = (fo["overnight"].reindex(
            fo["overnight"].index.union(board.index)).ffill().reindex(board.index))
        idx = sig.dropna().index.intersection(board.dropna(subset=["cc"]).index)
        s = np.asarray(sig.reindex(idx), float)
        g = np.asarray(board["gap"].reindex(idx), float)
        it = np.asarray(board["intraday"].reindex(idx), float)
        m = np.isfinite(s) & np.isfinite(g) & np.isfinite(it)
        s, g, it = s[m], g[m], it[m]
        if len(s) < 200:
            continue
        beta, alpha = np.polyfit(s, g, 1)
        pred = alpha + beta * s
        eps = g - pred
        ss_res = float(((g - pred) ** 2).sum())
        ss_tot = float(((g - g.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        re, ne, te = corr(pd.Series(eps), pd.Series(it))
        q = pd.qcut(pd.Series(eps), 5, labels=False, duplicates="drop")
        f2 = pd.DataFrame({"q": np.asarray(q), "it": it})
        agg = f2.groupby("q")["it"].mean()
        lo_q, hi_q = float(agg.iloc[0]), float(agg.iloc[-1])
        print(f"{label:6s}{fut:6s}{beta:14.3f}{r2:8.3f}"
              f"{re:14.3f}{te:8.1f}{lo_q:14.3f}{hi_q:14.3f}")

        # 交易化：ε 明显为负（开盘没跌够）⇒ 开盘买、收盘卖
        trade = eps < np.percentile(eps, 20)
        pnl = it[trade] - 2 * 0.05           # 双边各 5bp
        ann = float(np.prod(1 + pnl / 100.0) ** (244 / len(s)) - 1) if len(pnl) > 30 else float("nan")
        print(f"       ε 最低 20%（{int(trade.sum())} 天）日内涵 {np.mean(it[trade]):+.3f}%"
              f" 扣成本后 {np.mean(pnl):+.3f}%/次  胜率 {np.mean(pnl > 0) * 100:.1f}%"
              f"  NW t={nw_t(pnl):+.2f}  全样本摊年化 {ann * 100:+.2f}%")
        report.setdefault("residual", []).append({
            "label": label, "futures": fut, "beta": round(float(beta), 3),
            "r2": round(r2, 3), "resid_intraday_r": round(re, 3),
            "resid_t": round(te, 1),
            "eps_low_q_intraday": round(lo_q, 3),
            "eps_high_q_intraday": round(hi_q, 3),
            "trade_days": int(trade.sum()),
            "trade_mean_net": round(float(np.mean(pnl)), 3),
            "trade_win": round(float(np.mean(pnl > 0) * 100), 1),
            "trade_nw_t": round(nw_t(pnl), 2),
            "trade_ann": round(ann * 100, 2) if ann == ann else None})

    # ---------------- E. 副产品：A 股自身的"跳空动量"，以及它为什么不可交易 ----------------
    # 残差检验里 ε 最高组的日内收益远高于最低组，一看像是"开盘越强日内越强"。
    # 但这不是外盘信号，是 A 股自己的跳空动量。必须和宽基指数对照，否则会误判成规律：
    #   · 沪深300：中位 0.000%、corr(gap,日内) = −0.034 ⇒ 与文献一致（隔夜反转）
    #   · 中证1000：中位 +0.376% ⇒ 小盘/主题股才有
    # 概念板块是**等权组合**（30~80 只小盘为主），天然像中证1000 而不像沪深300。
    print()
    print("=" * 116)
    print("E. 副产品：A 股自身「跳空 → 日内」，概念板块 vs 宽基指数（判断是否为小盘特征）")
    print("=" * 116)
    print(f"{'标的':14s}{'n':>6s}{'corr(gap,日内)':>15s}"
          f"{'前20%均值%':>12s}{'中位%':>9s}{'5-95截尾%':>11s}{'去top10%':>10s}")

    def _gap_momentum(series_gap: pd.Series, series_intra: pd.Series, label: str):
        bb = pd.DataFrame({"gap": series_gap, "intraday": series_intra}).dropna()
        if len(bb) < 200:
            return None
        thr = np.percentile(np.asarray(bb["gap"], float), 80)
        sel = bb[bb["gap"] >= thr]["intraday"]
        med = float(sel.median())
        trim = float(sel[(sel > sel.quantile(0.05)) & (sel < sel.quantile(0.95))].mean())
        d10 = float(sel.drop(sel.nlargest(10).index).mean())
        cc = float(bb["gap"].corr(bb["intraday"]))
        print(f"{label:14s}{len(sel):6d}{cc:15.3f}{float(sel.mean()):12.3f}"
              f"{med:9.3f}{trim:11.3f}{d10:10.3f}")
        return {"label": label, "n": int(len(sel)), "corr": round(cc, 3),
                "mean": round(float(sel.mean()), 3), "median": round(med, 3),
                "trim": round(trim, 3), "drop_top10": round(d10, 3)}

    by_idx = os.path.join(panel_store.PANEL_DIR, "index")
    for cname in ("煤炭", "黄金概念", "铜"):
        board = board_ohlc((concepts.get(cname) or {}).get("codes") or [])
        if board is None:
            continue
        row = _gap_momentum(board["gap"], board["intraday"], f"{cname}(等权)")
        if row:
            report.setdefault("gap_momentum", []).append(row)
    for code, name in (("sh000300", "沪深300"), ("sh000001", "上证指数"),
                       ("sh000852", "中证1000"), ("sz399006", "创业板指")):
        p = os.path.join(by_idx, f"{code}.parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p, columns=["date", "open", "close"])
        d["date"] = pd.to_datetime(d["date"])
        d = d.set_index("date").sort_index()
        row = _gap_momentum((d["open"] / d["close"].shift(1) - 1) * 100,
                            (d["close"] / d["open"] - 1) * 100, name)
        if row:
            report.setdefault("gap_momentum", []).append(row)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print()
    print(f"[输出] {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
