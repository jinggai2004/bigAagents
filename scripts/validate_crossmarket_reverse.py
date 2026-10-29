# -*- coding: utf-8 -*-
"""「领先为负 ⇒ 能否反向预测」的检验。

背景：`scripts/validate_crossmarket_jk.py` 量到 日经/韩国 T-1 → 沪深300 T 日
r = −0.036 / −0.085（韩国 t=−2.7）。看起来"反着做"就有信号，但相关系数为负
**不等于**可交易，必须逐条排除四件事：

  1. **不是 A 股自身均值回复的伪影**：若 A 股自己就"昨跌今涨"，那么"外盘跌→A股涨"
     可能只是外盘与 A 股同期正相关（0.30~0.36）带来的镜像。⇒ 加"自身反转"对照。
  2. **分组要单调**：只看相关系数会被少数极端日带偏，要看 5 组的次日收益是否单调。
  3. **样本外/分年稳定**：4 年样本里 2022 与 2026 是两个极端市况，分年拆开看。
  4. **扣成本后还剩多少**：日频反向策略换手率极高，双边 0.1% 就能吃光 2pp 的 edge。

输出：控制台表格（年化、胜率、夏普、最大回撤、t 值）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import panel_store  # noqa: E402

A_CODE = "sh000300"
A_NAME = "沪深300"
COST_ONE_WAY = 0.0005          # 单边 5bp（ETF 佣金+冲击），双边 10bp


# ----------------------------------------------------------------------
def load_index(code: str) -> pd.Series:
    p = os.path.join(panel_store.PANEL_DIR, "index", f"{code}.parquet")
    df = pd.read_parquet(p, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].astype(float).sort_index()


def overseas(name: str) -> pd.Series:
    import akshare as ak
    df = ak.index_global_hist_sina(symbol=name)
    return pd.Series(df["close"].astype(float).values,
                     index=pd.to_datetime(df["date"])).sort_index()


def perf(daily: pd.Series, label: str) -> dict:
    """日收益序列 → 年化/胜率/夏普/最大回撤。空仓日记 0。"""
    d = daily.dropna()
    if len(d) < 60:
        return {"label": label, "n": len(d)}
    nav = (1 + d).cumprod()
    years = len(d) / 244
    ann = float(nav.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan")
    vol = float(d.std() * np.sqrt(244))
    sharpe = float(d.mean() / d.std() * np.sqrt(244)) if d.std() > 0 else float("nan")
    peak = nav.cummax()
    mdd = float((nav / peak - 1).min())
    traded = d[d != 0]
    win = float((traded > 0).mean() * 100) if len(traded) else float("nan")
    return {"label": label, "n": int(len(d)), "traded_days": int(len(traded)),
            "ann": round(ann * 100, 2), "vol": round(vol * 100, 2),
            "sharpe": round(sharpe, 2), "mdd": round(mdd * 100, 2),
            "win": round(win, 1) if win == win else None,
            "total": round(float(nav.iloc[-1] - 1) * 100, 2)}


def groups(signal: pd.Series, fwd: pd.Series, q: int = 5) -> dict:
    """按信号分位分组，看次日（或当日）收益，检验单调性。"""
    idx = signal.dropna().index.intersection(fwd.dropna().index)
    s = np.asarray(signal.reindex(idx), float)
    f = np.asarray(fwd.reindex(idx), float)
    m = np.isfinite(s) & np.isfinite(f)
    s, f = s[m], f[m]
    if len(s) < q * 30:
        return {}
    g = pd.qcut(pd.Series(s), q, labels=False, duplicates="drop")
    agg = pd.DataFrame({"g": np.asarray(g), "f": f}).groupby("g")["f"].agg(["mean", "count"])
    means = [round(float(v) * 100, 3) for v in agg["mean"]]
    # 单调性：组号与均值的秩相关
    rho = float(pd.Series(means).corr(pd.Series(range(len(means))), method="spearman"))
    return {"means_pct": means, "counts": [int(v) for v in agg["count"]],
            "monotone_rho": round(rho, 3), "n": int(len(s))}


def t_stat(x: pd.Series) -> float:
    d = x.dropna()
    if len(d) < 30 or d.std() == 0:
        return float("nan")
    return float(d.mean() / (d.std() / np.sqrt(len(d))))


# ----------------------------------------------------------------------
def main() -> int:
    a_close = load_index(A_CODE)
    a_ret = a_close.pct_change()               # 沪深300 日收益（小数）

    base = perf(a_ret, "买入持有")

    print("=" * 100)
    print(f"基准：{A_NAME} 买入持有（{a_ret.index.min().date()} ~ {a_ret.index.max().date()}）")
    print("=" * 100)
    print(f"{'策略':26s}{'年化%':>8s}{'波动%':>8s}{'夏普':>7s}{'回撤%':>8s}"
          f"{'交易胜率%':>10s}{'累计%':>9s}{'交易天数':>9s}")
    print(f"{base['label']:26s}{base['ann']:8.2f}{base['vol']:8.2f}{base['sharpe']:7.2f}"
          f"{base['mdd']:8.2f}{'—':>10s}{base['total']:9.2f}{base['n']:9d}")

    print()
    print("=" * 100)
    print("A. 反向策略：外盘 T-1 跌 → 次日持有 A 股；外盘 T-1 涨 → 空仓（扣双边 10bp）")
    print("=" * 100)
    print(f"{'策略':26s}{'年化%':>8s}{'波动%':>8s}{'夏普':>7s}{'回撤%':>8s}"
          f"{'交易胜率%':>10s}{'累计%':>9s}{'交易天数':>9s}")

    rows = []
    for name in ("日经225指数", "首尔综合指数"):
        try:
            os_close = overseas(name)
        except Exception as e:                 # noqa: BLE001
            _logger.warning("[反向] %s 拉取失败: %s: %s", name, type(e).__name__, e)
            print(f"{name}: 拉取失败 {type(e).__name__}: {e}")
            continue
        os_ret = os_close.pct_change()
        # 对齐到 A 股交易日，再取"前一交易日"
        prev = (os_ret.reindex(os_ret.index.union(a_ret.index)).ffill()
                .reindex(a_ret.index).shift(1))
        idx = prev.dropna().index.intersection(a_ret.dropna().index)
        sig = prev.reindex(idx)
        fwd = a_ret.reindex(idx)

        # ① 反向策略：sign = -昨外盘
        pos = (sig < 0).astype(float)          # 外盘跌 ⇒ 做多
        turnover = pos.diff().abs().fillna(pos)
        strat = pos * fwd - turnover * COST_ONE_WAY
        rows.append(("反向·" + name[:4], perf(strat, "反向·" + name[:4])))

        # ② 同向策略（做对照，应更差）
        pos2 = (sig > 0).astype(float)
        turnover2 = pos2.diff().abs().fillna(pos2)
        strat2 = pos2 * fwd - turnover2 * COST_ONE_WAY
        rows.append(("同向·" + name[:4], perf(strat2, "同向·" + name[:4])))

        grp = groups(sig, fwd)                 # 按外盘 T-1 分组看 A 股 T 日
        print(f"  {name} 分组（外盘 T-1 由低到高 → A 股 T 日均值 %）: {grp.get('means_pct')}")
        print(f"    单调性 rho={grp.get('monotone_rho')}  n={grp.get('n')}  "
              f"信号 t={t_stat(sig):.2f}")

    # ③ 自身反转对照：A 股自己的 T-1
    prev_a = a_ret.shift(1)
    idx = prev_a.dropna().index.intersection(a_ret.dropna().index)
    sig_a, fwd_a = prev_a.reindex(idx), a_ret.reindex(idx)
    pos_a = (sig_a < 0).astype(float)
    to_a = pos_a.diff().abs().fillna(pos_a)
    rows.append(("反向·A股自身", perf(pos_a * fwd_a - to_a * COST_ONE_WAY, "反向·A股自身")))
    grp_a = groups(sig_a, fwd_a)

    for _label, r in rows:
        if "ann" not in r:
            continue
        print(f"{r['label']:26s}{r['ann']:8.2f}{r['vol']:8.2f}{r['sharpe']:7.2f}"
              f"{r['mdd']:8.2f}{(r['win'] if r['win'] is not None else 0):10.1f}"
              f"{r['total']:9.2f}{r['traded_days']:9d}")

    print()
    print(f"  对照：A 股自身 T-1 分组 → T 日均值 %: {grp_a.get('means_pct')}")
    print(f"        单调性 rho={grp_a.get('monotone_rho')}  n={grp_a.get('n')}  "
          f"信号 t={t_stat(sig_a):.2f}")

    # ④ 分年稳定性（反向·韩国 与 反向·A股自身）
    print()
    print("=" * 100)
    print("B. 分年稳定性（年化 %）：单一时段的负相关，换个年份还在不在")
    print("=" * 100)
    series = {}
    for name in ("日经225指数", "首尔综合指数"):
        os_close = overseas(name)
        os_ret = os_close.pct_change()
        prev = (os_ret.reindex(os_ret.index.union(a_ret.index)).ffill()
                .reindex(a_ret.index).shift(1))
        idx = prev.dropna().index.intersection(a_ret.dropna().index)
        pos = (prev.reindex(idx) < 0).astype(float)
        to = pos.diff().abs().fillna(pos)
        series["反向·" + name[:4]] = (pos * a_ret.reindex(idx) - to * COST_ONE_WAY)
    pos_a2 = (a_ret.shift(1) < 0).astype(float)
    to_a2 = pos_a2.diff().abs().fillna(pos_a2)
    series["反向·A股自身"] = (pos_a2 * a_ret - to_a2 * COST_ONE_WAY)

    frame = pd.DataFrame({k: v for k, v in series.items()})
    frame["基准"] = a_ret.reindex(frame.index)
    by_year = frame.groupby(frame.index.year).apply(
        lambda d: (1 + d).prod() ** (244 / max(1, len(d))) - 1) * 100
    print(by_year.round(2).to_string())
    return 0





# ======================================================================
# C. 关键补充：不是"隔日反向持有"，而是"低开后的日内修复"
# ======================================================================
# 文献与实务都指向同一个机制（Fung et al. 2010；McGill CO-OC reversal）：
#   外盘大跌 → A 股**开盘**跳空低开 → 日内修复（隔夜收益的约 39% 会在日内反转）。
# 所以正确的用法是 open→close，而不是 close→close。上一版的"隔日持有"测错了对象。

def _ohlc(code: str) -> pd.DataFrame:
    p = os.path.join(panel_store.PANEL_DIR, "index", f"{code}.parquet")
    df = pd.read_parquet(p, columns=["date", "open", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    out = pd.DataFrame(index=df.index)
    out["gap"] = df["open"] / df["close"].shift(1) - 1.0        # 隔夜跳空
    out["intraday"] = df["close"] / df["open"] - 1.0            # 开盘→收盘
    out["cc"] = df["close"] / df["close"].shift(1) - 1.0        # 收盘→收盘
    return out


def reverse_intraday() -> int:
    ohlc = _ohlc(A_CODE)
    print()
    print("=" * 100)
    print("C. 低开后日内修复：外盘 T-1 跌 → T 日「开盘买、收盘卖」（不隔夜）")
    print("=" * 100)
    print(f"基准：全样本日内收益均值 {ohlc['intraday'].mean() * 100:.3f}% / "
          f"跳空均值 {ohlc['gap'].mean() * 100:.3f}% / "
          f"隔夜占比 {ohlc['gap'].abs().mean() / (ohlc['gap'].abs().mean() + ohlc['intraday'].abs().mean()) * 100:.0f}%")

    def _us_close() -> pd.Series:
        import akshare as ak
        df = ak.index_us_stock_sina(symbol=".IXIC")
        return pd.Series(df["close"].astype(float).values,
                         index=pd.to_datetime(df["date"])).sort_index()

    # ⚠️ 关键区别：日韩与 A 股**同一时段**交易，信息当天已消化；
    #    美股有 12 小时时差，A 股开盘前才拿到 ⇒ 只有美股才构成"隔夜未消化信息"。
    sources = []
    for nm in ("日经225指数", "首尔综合指数"):
        try:
            sources.append((nm[:4], overseas(nm)))
        except Exception as e:                 # noqa: BLE001
            _logger.warning("[反向] %s 拉取失败: %s: %s", nm, type(e).__name__, e)
    try:
        sources.append(("纳斯达克", _us_close()))
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[反向] 纳指拉取失败: %s: %s", type(e).__name__, e)

    for name, os_close in sources:
        os_ret = os_close.pct_change()
        prev = (os_ret.reindex(os_ret.index.union(ohlc.index)).ffill()
                .reindex(ohlc.index).shift(1))
        idx = prev.dropna().index.intersection(ohlc.dropna(subset=["intraday"]).index)
        sig = prev.reindex(idx)
        gap = ohlc["gap"].reindex(idx)
        intra = ohlc["intraday"].reindex(idx)

        print(f"\n  【{name}】n={len(idx)}")
        print(f"    {'信号(外盘T-1)':22s}{'天数':>6s}{'次日跳空%':>11s}{'日内收益%':>11s}{'日内胜率%':>10s}")
        cuts = [(-99, -2.0), (-2.0, -1.0), (-1.0, 0.0), (0.0, 1.0), (1.0, 99)]
        for lo, hi in cuts:
            m = (sig * 100 > lo) & (sig * 100 <= hi)
            if int(m.sum()) < 10:
                continue
            print(f"    [{lo:>5.1f}%, {hi:>5.1f}%]{'':6s}{int(m.sum()):6d}"
                  f"{gap[m].mean() * 100:11.3f}{intra[m].mean() * 100:11.3f}"
                  f"{(intra[m] > 0).mean() * 100:10.1f}")

        # 策略：外盘跌超 1% → 次日开盘买、收盘卖
        trade = (sig * 100 <= -1.0) & intra.notna()
        pnl = intra[trade] - 2 * COST_ONE_WAY     # 日内一进一出
        if len(pnl) > 20:
            ann = (1 + pnl).prod() ** (244 / len(idx)) - 1
            print(f"    → 策略(外盘≤-1% 才做): 交易 {len(pnl)} 天 / 总 {len(idx)} 天，"
                  f"日均 {pnl.mean() * 100:+.3f}%，年化(按全样本摊) {ann * 100:+.2f}%，"
                  f"胜率 {(pnl > 0).mean() * 100:.1f}%，t={t_stat(pnl):.2f}")
    return 0


if __name__ == "__main__":
    main()
    sys.exit(reverse_intraday())
