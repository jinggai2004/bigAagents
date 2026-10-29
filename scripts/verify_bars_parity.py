# -*- coding: utf-8 -*-
"""P0-1 验收②：本地面板（get_bars）与联网（新浪 qfq）的**逐值一致性**核验。

判据（必须能被反驳）：
  - 同一 (code, date) 的**日收益**相关系数 > 0.999（价格口径可能因数据源
    小数位/复权算法略有差异，收益才是因子真正消费的量）；
  - 20 日动量（close/close.shift(20)-1）在各对齐日的最大绝对差 < 5e-3
    （0.5%：实测 600028/000001 为 0.20%/0.11%，来源是**复权算法差异**
     —— tushare 的 preclose 是交易所口径，新浪 qfq 用自有因子；动量在同日的
     全市场分布标准差通常 5~10%，0.5% 的误差不会改变分位排序）；
  - 收盘价的最大相对差 < 3%（前复权基准相同，差异只应来自数据源精度）。

用法（需要联网，一次性人工核验，不进 pytest）：
    python scripts/verify_bars_parity.py
    python scripts/verify_bars_parity.py --codes 600028 000001 600519 --days 250
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import data.bars as B
from data import sina_source


def _aligned(local: pd.DataFrame, remote: pd.DataFrame) -> pd.DataFrame:
    a = local[["date", "close"]].copy()
    b = remote.copy()
    b["date"] = pd.to_datetime(b["date"])
    b = b[["date", "close"]]
    m = a.merge(b, on="date", suffixes=("_local", "_remote")).sort_values("date")
    return m


def check_code(code: str, days: int = 250) -> dict | None:
    local = B.get_bars(code)
    if local is None:
        return {"code": code, "status": "local_missing"}
    end = local["date"].max()
    start = end - pd.Timedelta(days=int(days * 1.6))
    local = local[local["date"] >= start].reset_index(drop=True)
    remote = sina_source.fetch_stock_daily(
        code, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if remote is None or remote.empty:
        return {"code": code, "status": "remote_unavailable"}

    m = _aligned(local, remote)
    if len(m) < 40:
        return {"code": code, "status": f"insufficient_overlap({len(m)})", "overlap": len(m)}

    r_loc = m["close_local"].pct_change()
    r_rem = m["close_remote"].pct_change()
    valid = r_loc.notna() & r_rem.notna()
    corr = float(np.corrcoef(r_loc[valid], r_rem[valid])[0, 1]) if valid.sum() > 5 else float("nan")

    mom_loc = m["close_local"] / m["close_local"].shift(20) - 1
    mom_rem = m["close_remote"] / m["close_remote"].shift(20) - 1
    mom_diff = (mom_loc - mom_rem).abs()
    rel_close = ((m["close_local"] - m["close_remote"]).abs()
                 / m["close_remote"].replace(0, np.nan))
    return {
        "code": code, "status": "ok", "overlap": int(len(m)),
        "ret_corr": round(corr, 6) if corr == corr else None,
        "mom_max_diff": round(float(mom_diff.max()), 6),
        "close_max_rel_diff": round(float(rel_close.max()), 6),
        "close_median_rel_diff": round(float(rel_close.median()), 8),
        "span": f"{m['date'].min().date()}~{m['date'].max().date()}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="+",
                    default=["600028", "000001", "600519", "300750", "000858"])
    ap.add_argument("--days", type=int, default=250)
    args = ap.parse_args()

    ok = 0
    bad: list[str] = []
    for c in args.codes:
        r = check_code(c, args.days)
        print(r)
        if r and r.get("status") == "ok":
            ok += 1
            if (r["ret_corr"] or 0) < 0.999:
                bad.append(f"{c}: 收益相关 {r['ret_corr']} < 0.999")
            if r["mom_max_diff"] > 5e-3:
                bad.append(f"{c}: 动量最大差 {r['mom_max_diff']} > 5e-3")
            if r["close_max_rel_diff"] > 0.03:
                bad.append(f"{c}: 收盘最大相对差 {r['close_max_rel_diff']} > 3%")
    print(f"\n[结论] 可用比对 {ok}/{len(args.codes)} 只；不达标 {len(bad)} 项")
    for b in bad:
        print("  ✗", b)
    return 0 if ok and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
