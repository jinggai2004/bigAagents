# -*- coding: utf-8 -*-
"""板块级：美股行业 ETF → A 股对应概念板块。

为什么要下沉到板块：指数级只量到 纳指 → A 股半导体 r=0.105。
但"对应板块"本该比"大盘对板块"更紧——如果连 SOXX→半导体 都只有 0.1，
那就说明跨市场传导整体很弱，不值得做；如果明显更高，说明**映射对**才是关键。

数据：
  · 美股：`ak.stock_us_daily(symbol=...)`（新浪，走 `finance.sina.com.cn`，本机可用；
    ⚠️ 东财 `stock_us_hist` 已被 RST，不要用）
  · A 股：本地 `panel` 面板（等权组合）

对齐：美股 T-1 收盘 → A 股 T 日（时差 ⇒ A 股开盘前已知）。
拆分：跳空（open/preclose-1）/ 日内（close/open-1）/ 全天（close/close-1）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import concept_members as cm  # noqa: E402
from data import panel_store  # noqa: E402

BY_CODE = os.path.join(panel_store.PANEL_DIR, "by_code")

# (标签, 美股代码, A 股概念名)  —— ETF 都是 iShares/SPDR 的行业代表
PAIRS = [
    ("半导体", "SOXX", "半导体"),
    ("软件", "IGV", "国产软件"),
    ("生物科技", "XBI", "生物制品"),
    ("能源", "XLE", "石油石化"),
    ("银行", "KBE", "银行"),
    ("必需消费", "XLP", "食品饮料"),
]
BASE = ("纳指→半导体", ".IXIC", "半导体")


def board_ohlc(codes, min_codes: int = 3) -> pd.DataFrame | None:
    """概念成分等权：gap / intraday / cc（单位 %）。"""
    frames = []
    for c in codes:
        p = os.path.join(BY_CODE, f"{c}.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p, columns=["trade_date", "open", "close", "preclose"])
        if df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date").sort_index()
        frames.append(pd.DataFrame({
            "gap": df["open"] / df["preclose"] - 1.0,
            "intraday": df["close"] / df["open"] - 1.0,
            "cc": df["close"] / df["preclose"] - 1.0,
        }))
    if len(frames) < min_codes:
        return None
    # 逐日等权（各文件 index 不完全相同，concat 后按列求均值天然对齐到并集）
    merged = pd.concat(frames, keys=range(len(frames)))
    out = merged.groupby(level=1).mean() * 100
    return out.sort_index()


def us_close(symbol: str) -> pd.Series | None:
    import akshare as ak
    try:
        if symbol.startswith("."):
            df = ak.index_us_stock_sina(symbol=symbol)
            s = pd.Series(df["close"].astype(float).values,
                          index=pd.to_datetime(df["date"]))
        else:
            df = ak.stock_us_daily(symbol=symbol)
            s = pd.Series(df["close"].astype(float).values,
                          index=pd.to_datetime(df["date"]))
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[板块] %s 拉取失败: %s: %s", symbol, type(e).__name__, e)
        print(f"  {symbol} 拉取失败: {type(e).__name__}: {str(e)[:60]}")
        return None
    return s.sort_index()


def clean_us_ret(px: pd.Series, thresh_pct: float = 30.0) -> pd.Series:
    """美股 ETF 日收益（**单位 %**），剔除非交易日涨跌造成的假跳变。

    ⚠️⚠️ 实测事故（2026-10-29 发现）：**新浪 `stock_us_daily` 不复权**。
    SOXX 与 IGV 都在 2024-03-07 做过拆股，XBI 2015-09-11、XLE 2025-12-05 各一次。
    未复权的 `pct_change()` 会把拆股当成一天 −67% 的暴跌：
      · SOXX→半导体 日频被压到 **0.092**（清洗后 **0.119**）
      · IGV →国产软件 被压到 **0.068**（清洗后 **0.105**）
    一个点就能让相关系数掉 0.03~0.04，足以把结论从"对口板块更准"翻成"更差"。
    ⇒ 宽基/行业/商品 ETF 单日真实波动不可能超 30%（2020-03-09 原油崩盘也只有 −25.3%），
      超过即判为公司行动，置 NaN。

    ⚠️ 阈值单位：本函数返回**百分数**，故阈值是 `30.0` 而非 `0.30`。
    写成 0.30 会把 3,258 个正常交易日（涨幅 >0.3%）全判成拆股 —— 已踩过一次。
    """
    r = px.pct_change() * 100
    bad = r.abs() > thresh_pct
    if int(bad.sum()):
        _logger.warning("[板块] 剔除 %d 个疑似拆股点: %s",
                        int(bad.sum()), [str(d.date()) for d in r[bad].index][:5])
    return r.mask(bad)


def corr(x: pd.Series, y: pd.Series) -> tuple[float, int, float]:
    idx = x.dropna().index.intersection(y.dropna().index)
    xa = np.asarray(x.reindex(idx), float)
    ya = np.asarray(y.reindex(idx), float)
    m = np.isfinite(xa) & np.isfinite(ya)
    n = int(m.sum())
    if n < 60 or np.std(xa[m]) == 0 or np.std(ya[m]) == 0:
        return float("nan"), n, float("nan")
    r = float(np.corrcoef(xa[m], ya[m])[0, 1])
    return r, n, float(r * np.sqrt((n - 2) / max(1e-12, 1 - r * r)))


def main() -> int:
    concepts = cm.get_concept_members()["concepts"]

    print("=" * 108)
    print("美股行业 ETF（T-1 收盘）→ A 股对应概念板块（T 日）")
    print("=" * 108)
    # ⚠️ 方向必须写清：美股与 A 股**同日期不重叠**
    #     A 股 t 日 15:00 收盘，美股 t 日 21:30(北京) 才开盘 ⇒ "同日"实为 A 股先行约 6.5h；
    #     真正的跨市场传导是"美股 t-1 → A 股 t"。
    print(f"{'板块':10s}{'ETF':7s}{'A股成分':>7s}{'n':>6s}"
          f"{'跳空 r':>9s}{'日内 r':>9s}{'全天 r':>9s}{'全天 t':>8s}{'同日A→美':>10s}")
    rows = []
    for label, code, cname in PAIRS + [BASE]:
        codes = (concepts.get(cname) or {}).get("codes") or []
        board = board_ohlc(codes)
        px = us_close(code)
        if board is None or px is None:
            print(f"{label:10s} 跳过")
            continue
        ret = clean_us_ret(px)
        # 美股 T-1 → A 股 T 日
        prev = (ret.reindex(ret.index.union(board.index)).ffill()
                .reindex(board.index).shift(1))
        rg, ng, _tg = corr(prev, board["gap"])
        ri, _ni, _ti = corr(prev, board["intraday"])
        rc, nc, tc = corr(prev, board["cc"])
        r_au, _nau, _tau = corr(ret, board["cc"])      # 同日：A 股 t → 美股 t
        rows.append({"label": label, "etf": code, "concept": cname,
                     "codes": len(codes), "n": nc,
                     "gap_r": round(rg, 3), "intraday_r": round(ri, 3),
                     "cc_r": round(rc, 3), "cc_t": round(tc, 1),
                     "a_lead_us": round(r_au, 3)})
        print(f"{label:10s}{code:7s}{len(codes):7d}{nc:6d}"
              f"{rg:9.3f}{ri:9.3f}{rc:9.3f}{tc:8.1f}{r_au:10.3f}")

    # 分组：按 ETF T-1 收益五等分，看 A 股 T 日跳空/日内
    print()
    print("=" * 108)
    print("分组（按 ETF T-1 收益由低到高 → A 股 T 日，单位 %）")
    print("=" * 108)
    for label, code, cname in PAIRS + [BASE]:
        codes = (concepts.get(cname) or {}).get("codes") or []
        board = board_ohlc(codes)
        px = us_close(code)
        if board is None or px is None:
            continue
        ret = clean_us_ret(px)
        prev = (ret.reindex(ret.index.union(board.index)).ffill()
                .reindex(board.index).shift(1))
        idx = prev.dropna().index.intersection(board.dropna(subset=["cc"]).index)
        s = np.asarray(prev.reindex(idx), float)
        # ⚠️ board_ohlc 已 *100 转成 %，这里不能再乘（否则分组均值放大 100 倍）
        g = np.asarray(board["gap"].reindex(idx), float)
        it = np.asarray(board["intraday"].reindex(idx), float)
        m = np.isfinite(s) & np.isfinite(g) & np.isfinite(it)
        s, g, it = s[m], g[m], it[m]
        if len(s) < 150:
            continue
        q = pd.qcut(pd.Series(s), 5, labels=False, duplicates="drop")
        f = pd.DataFrame({"q": np.asarray(q), "gap": g, "it": it})
        agg = f.groupby("q").agg(gap=("gap", "mean"), it=("it", "mean"), n=("it", "size"))
        print(f"  【{label}】n={len(s)}")
        print(f"    跳空: {[round(v, 3) for v in agg['gap']]}")
        print(f"    日内: {[round(v, 3) for v in agg['it']]}")
        print(f"    天数: {[int(v) for v in agg['n']]}")
    return 0


import logging  # noqa: E402

_logger = logging.getLogger(__name__)

if __name__ == "__main__":
    sys.exit(main())
