# -*- coding: utf-8 -*-
"""日韩指数 → A 股大盘（补充检验，样本受数据源限制只有约 4 年）。

为什么单独跑：日韩这条线**没有板块级数据**，只能用大盘指数。
但"指数 → 指数"也要量一下，否则报告里对日韩的判断就成了空断言。

时间关系（决定了哪个 lag 才是真领先）：
  · 日经 225 收盘 15:00 JST = 14:00 CST —— 在 A 股（15:00 CST）收盘**之前** ⇒ lag0 是同期
  · KOSPI 收盘 15:30 KST = 14:30 CST —— 同上
  · 所以"日经 T-1 收盘 → A 股 T 日"才是 A 股开盘前**已知**的信息（真领先）
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_logger = logging.getLogger(__name__)

from data import panel_store  # noqa: E402


def load_index(code: str) -> pd.Series:
    p = os.path.join(panel_store.PANEL_DIR, "index", f"{code}.parquet")
    df = pd.read_parquet(p, columns=["date", "open", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].astype(float).sort_index()


def load_gap(code: str) -> pd.Series:
    p = os.path.join(panel_store.PANEL_DIR, "index", f"{code}.parquet")
    df = pd.read_parquet(p, columns=["date", "open", "close"])
    df["date"] = pd.to_datetime(df["date"])
    g = (df["open"].astype(float) / df["close"].astype(float).shift(1) - 1) * 100
    return pd.Series(g.values, index=df["date"]).sort_index()


def overseas(name: str) -> pd.Series:
    import akshare as ak
    df = ak.index_global_hist_sina(symbol=name)
    out = pd.Series(df["close"].astype(float).values,
                    index=pd.to_datetime(df["date"]))
    return out.sort_index()


def corr(x: pd.Series, y: pd.Series) -> tuple[float, int, float]:
    idx = x.dropna().index.intersection(y.dropna().index)
    xa = np.asarray(x.reindex(idx), float)
    ya = np.asarray(y.reindex(idx), float)
    m = np.isfinite(xa) & np.isfinite(ya)
    n = int(m.sum())
    if n < 30 or np.std(xa[m]) == 0 or np.std(ya[m]) == 0:
        return float("nan"), n, float("nan")
    r = float(np.corrcoef(xa[m], ya[m])[0, 1])
    return r, n, float(r * np.sqrt((n - 2) / max(1e-12, 1 - r * r)))


def main() -> int:
    a_close = load_index("sh000300")
    a_ret = a_close.pct_change() * 100
    a_gap = load_gap("sh000300")

    print(f"{'外盘':10s}{'样本':>6s}{'样本区间':>26s}{'同期 r':>9s}{'T-1 领先 r':>12s}"
          f"{'t':>7s}{'→A股T日跳空 r':>15s}")
    for name in ("日经225指数", "首尔综合指数"):
        try:
            os_close = overseas(name)
        except Exception as e:                 # noqa: BLE001
            _logger.warning("[日韩] %s 拉取失败: %s: %s", name, type(e).__name__, e)
            print(f"{name:10s} 拉取失败: {type(e).__name__}: {e}")
            continue
        os_ret = os_close.pct_change() * 100
        idx = os_ret.dropna().index.intersection(a_ret.dropna().index)
        if len(idx) < 100:
            print(f"{name:10s} 交集不足（{len(idx)}）")
            continue
        span = f"{idx.min().date()} .. {idx.max().date()}"

        r0, n0, _t0 = corr(os_ret, a_ret)
        # 真领先：把外盘收益按日期对齐到"下一个 A 股交易日"
        prev = (os_ret.reindex(os_ret.index.union(a_ret.index)).ffill()
                .reindex(a_ret.index).shift(1))
        r1, n1, t1 = corr(prev, a_ret)
        rg, ng, _tg = corr(prev, a_gap)
        print(f"{name:10s}{n1:6d}{span:>26s}{r0:9.3f}{r1:12.3f}{t1:7.1f}{rg:15.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
