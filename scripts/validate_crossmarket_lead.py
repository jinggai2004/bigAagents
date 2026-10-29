# -*- coding: utf-8 -*-
"""跨市场领先滞后检验（一次性探索脚本，跑完再决定是否固化进 scripts/）。

要回答的问题：
  「美日韩相似板块 + 期货走势」能不能预测 A 股个股？

先不猜，直接量：
  1) 期货 T 日涨跌 → A 股对应概念板块 T+1 收益：相关系数、分组收益、样本内外分段；
  2) 同期相关（lag0）作为**上限参照**——lag0 都不显著时 lag1 更不可能；
  3) 反向（A 股 T 日 → 期货 T-1）作为安慰剂，排除"单边领先"的错觉；
  4) 外盘 T-1（美股，A 股开盘前已知）→ A 股 T 日。

输出：控制台表格 + JSON（供可行性报告引用）。
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

_logger = logging.getLogger(__name__)

BY_CODE = os.path.join(panel_store.PANEL_DIR, "by_code")
OUT_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "output", "跨市场领先滞后检验_2026-10-29.json")


# ----------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------
def board_returns(codes, min_codes: int = 3) -> pd.Series | None:
    """概念成分股等权日收益（%）。缺失跳过，不补 0（缺就是缺）。"""
    frames = []
    for c in codes:
        p = os.path.join(BY_CODE, f"{c}.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p, columns=["trade_date", "pct_chg"])
        if df.empty:
            continue
        # ⚠️ 面板的 trade_date 是**字符串**，期货/外盘 index 是 datetime64；
        #    不转类型的话 `index.intersection` 得 0 条，整张表全是 nan（本次踩过）。
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        frames.append(df.set_index("trade_date")["pct_chg"].astype(float).rename(c))
    if len(frames) < min_codes:
        return None
    wide = pd.concat(frames, axis=1).sort_index()
    return wide.mean(axis=1, skipna=True).rename("board")


def futures_series(symbol: str) -> pd.Series | None:
    """国内商品期货主力连续收盘价（新浪）。"""
    import akshare as ak
    try:
        df = ak.futures_main_sina(symbol=symbol)
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[跨市场] 期货 %s 拉取失败: %s: %s", symbol, type(e).__name__, e)
        print(f"  [期货] {symbol} 拉取失败: {type(e).__name__}: {e}")
        return None
    if df is None or df.empty:
        return None
    s = pd.Series(df["收盘价"].astype(float).values,
                  index=pd.to_datetime(df["日期"]), name=symbol)
    return s.sort_index()


def us_index_series(code: str) -> pd.Series | None:
    """美股指数收盘（新浪）。"""
    import akshare as ak
    try:
        df = ak.index_us_stock_sina(symbol=code)
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[跨市场] 美股 %s 拉取失败: %s: %s", code, type(e).__name__, e)
        print(f"  [美股] {code} 拉取失败: {type(e).__name__}: {e}")
        return None
    if df is None or df.empty:
        return None
    s = pd.Series(df["close"].astype(float).values,
                  index=pd.to_datetime(df["date"]), name=code)
    return s.sort_index()


# ----------------------------------------------------------------------
# 统计（全部显式按位置对齐，不做 pandas 隐式对齐）
# ----------------------------------------------------------------------
def _aligned(x: pd.Series, y: pd.Series,
             shift_y: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """对齐成两条等长 ndarray（含 NaN）。

    ⚠️ 不用 `pd.DataFrame({"a": x, "b": y})`：两 Series 的 index 不完全相同时
    pandas 会做并集对齐，`dropna()` 后可能整表为空，随后取列直接 KeyError
    （本次踩过）。ndarray 没有"隐式对齐"这回事。
    """
    x, y = x.dropna(), y.dropna()
    idx = x.index.intersection(y.index)
    xa = np.asarray(x.reindex(idx), dtype=float)
    ya = pd.Series(np.asarray(y.reindex(idx), dtype=float), index=idx)
    if shift_y:
        ya = ya.shift(shift_y)
    return xa, np.asarray(ya, dtype=float)


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, int, float]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 30:
        return float("nan"), n, float("nan")
    xv, yv = x[m], y[m]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return float("nan"), n, float("nan")
    r = float(np.corrcoef(xv, yv)[0, 1])
    t = float(r * np.sqrt((n - 2) / max(1e-12, 1 - r * r)))
    return r, n, t


def lag_table(fut: pd.Series, board: pd.Series) -> dict:
    r0, n0, t0 = _pearson(*_aligned(fut, board, 0))
    r1, n1, t1 = _pearson(*_aligned(fut, board, -1))   # 期货 t → 板块 t+1
    r_1, n_1, t_1 = _pearson(*_aligned(fut, board, 1))  # 期货 t → 板块 t-1（安慰剂）
    return {"lag0": {"r": r0, "n": n0, "t": t0},
            "fut_leads_1": {"r": r1, "n": n1, "t": t1},
            "fut_lags_1": {"r": r_1, "n": n_1, "t": t_1}}


def group_returns(fut: pd.Series, board: pd.Series, q: int = 5) -> dict:
    """按期货当日收益分位分组，看**次日**板块收益（%）。"""
    x, y_next = _aligned(fut, board, -1)
    m = np.isfinite(x) & np.isfinite(y_next)
    xv, yv = x[m], y_next[m]
    if len(xv) < q * 20:
        return {}
    g = pd.qcut(pd.Series(xv), q, labels=False, duplicates="drop")
    frame = pd.DataFrame({"g": np.asarray(g), "y": yv})
    agg = frame.groupby("g")["y"].agg(["mean", "count"])
    means = [round(float(v), 4) for v in agg["mean"]]
    return {"buckets": means,
            "counts": [int(v) for v in agg["count"]],
            "top_minus_bottom": round(means[-1] - means[0], 4) if len(means) >= 2 else None,
            "n": int(len(xv))}


def ic_spearman(fut: pd.Series, board: pd.Series) -> float:
    x, y_next = _aligned(fut, board, -1)
    m = np.isfinite(x) & np.isfinite(y_next)
    if int(m.sum()) < 60:
        return float("nan")
    a = pd.Series(x[m]).rank().to_numpy()
    b = pd.Series(y_next[m]).rank().to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def split_sample(fut: pd.Series, board: pd.Series, frac: float = 0.7) -> dict:
    """样本内/外分段看领先相关（防"全样本挑出来的巧合"）。"""
    x, y_next = _aligned(fut, board, -1)
    m = np.isfinite(x) & np.isfinite(y_next)
    x, y_next = x[m], y_next[m]
    k = int(len(x) * frac)
    out = {}
    for tag, sl in (("in_sample", slice(0, k)), ("out_sample", slice(k, None))):
        xv, yv = x[sl], y_next[sl]
        if len(xv) < 30 or np.std(xv) == 0 or np.std(yv) == 0:
            out[tag] = {"r": None, "n": int(len(xv))}
            continue
        out[tag] = {"r": round(float(np.corrcoef(xv, yv)[0, 1]), 4), "n": int(len(xv))}
    return out


def board_gap(codes, min_codes: int = 3) -> pd.Series | None:
    """概念成分股等权**隔夜跳空**（open/preclose-1，%）。

    为什么要单独看跳空：外盘/期货的信息若在 A 股**开盘瞬间**就被吃掉，
    那"预测收盘"是没用的——你买不到开盘前的价。可交易性取决于
    "信息落在跳空里"还是"落在日内"。
    """
    frames = []
    for c in codes:
        p = os.path.join(BY_CODE, f"{c}.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p, columns=["trade_date", "open", "preclose"])
        if df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        g = (df["open"].astype(float) / df["preclose"].astype(float) - 1.0) * 100
        frames.append(pd.Series(g.values, index=df["trade_date"], name=c))
    if len(frames) < min_codes:
        return None
    wide = pd.concat(frames, axis=1).sort_index()
    return wide.mean(axis=1, skipna=True).rename("gap")


def hit_rate(fut: pd.Series, board: pd.Series) -> dict:
    """方向命中率：期货涨 → 次日板块也涨 的比例 vs 板块无条件上涨比例。"""
    x, y_next = _aligned(fut, board, -1)
    m = np.isfinite(x) & np.isfinite(y_next) & (x != 0)
    xv, yv = x[m], y_next[m]
    if len(xv) < 60:
        return {}
    up = xv > 0
    hit = float((( yv > 0) == up).mean() * 100)
    base_up = float((yv > 0).mean() * 100)
    return {"hit": round(hit, 2), "base_up": round(base_up, 2),
            "edge": round(hit - max(base_up, 100 - base_up), 2), "n": int(len(xv))}


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
THEMES = [
    ("铜", "铜", "CU0"),
    ("铝", "铝", "AL0"),
    ("黄金", "黄金概念", "AU0"),
    ("原油", "石油石化", "SC0"),
    ("纯碱", "纯碱", "SA0"),
]
US_PAIRS = [("美股→半导体", ".IXIC", "半导体"),
            ("美股→国产软件", ".IXIC", "国产软件")]


def main() -> int:
    concepts = cm.get_concept_members()["concepts"]
    result = {"generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
              "themes": [], "us": []}

    print("=" * 96)
    print("A. 国内期货主力(T 日) → A 股概念板块(T+1)   [%]")
    print("=" * 96)
    print(f"{'主题':6s}{'样本':>6s}{'lag0 r':>9s}{'t':>7s}{'期→股 r':>9s}{'t':>7s}"
          f"{'反向 r':>9s}{'IC':>8s}{'Top-Bot':>9s}{'胜率差':>8s}")
    for label, cname, fsym in THEMES:
        codes = (concepts.get(cname) or {}).get("codes") or []
        board = board_returns(codes)
        fut = futures_series(fsym)
        if board is None or fut is None:
            print(f"{label:6s} 跳过（board={'OK' if board is not None else 'NA'}，"
                  f"fut={'OK' if fut is not None else 'NA'}）")
            continue
        fut_ret = fut.pct_change() * 100
        tab = lag_table(fut_ret, board)
        grp = group_returns(fut_ret, board)
        x, y_next = _aligned(fut_ret, board, -1)
        m = np.isfinite(x) & np.isfinite(y_next)
        xv, yv = x[m], y_next[m]
        win_top = float((yv[xv >= np.quantile(xv, 0.8)] > 0).mean() * 100) if m.sum() > 60 else float("nan")
        win_bot = float((yv[xv <= np.quantile(xv, 0.2)] > 0).mean() * 100) if m.sum() > 60 else float("nan")
        row = {"label": label, "concept": cname, "codes": len(codes),
               "span": [str(board.index.min().date()), str(board.index.max().date())],
               **tab, "groups": grp, "ic": round(ic_spearman(fut_ret, board), 4),
               "split": split_sample(fut_ret, board),
               "hit": hit_rate(fut_ret, board),
               "win_top": round(win_top, 2), "win_bottom": round(win_bot, 2)}
        result["themes"].append(row)
        print(f"{label:6s}{tab['lag0']['n']:6d}"
              f"{tab['lag0']['r']:9.3f}{tab['lag0']['t']:7.1f}"
              f"{tab['fut_leads_1']['r']:9.3f}{tab['fut_leads_1']['t']:7.1f}"
              f"{tab['fut_lags_1']['r']:9.3f}"
              f"{row['ic']:8.3f}"
              f"{(grp.get('top_minus_bottom') if grp else float('nan')):9.3f}"
              f"{win_top - win_bot:8.1f}")
        if grp:
            print(f"       次日分组均值: {grp['buckets']}  (n={grp['n']})")
        print(f"       样本内/外 lag1 r: {row['split']}  方向命中: {row['hit']}")

    print()
    print("=" * 96)
    print("C. 信息落在「跳空」还是「日内」：期货 T 日 → A 股 T+1")
    print("=" * 96)
    print(f"{'主题':6s}{'→T+1跳空 r':>13s}{'t':>7s}{'→T+1收盘 r':>13s}{'t':>7s}")
    for label, cname, fsym in THEMES:
        codes = (concepts.get(cname) or {}).get("codes") or []
        gap = board_gap(codes)
        fut = futures_series(fsym)
        if gap is None or fut is None:
            continue
        fut_ret = fut.pct_change() * 100
        rg, ng, tg = _pearson(*_aligned(fut_ret, gap, -1))
        board = board_returns(codes)
        rc, nc, tc = _pearson(*_aligned(fut_ret, board, -1))
        result.setdefault("gap", []).append(
            {"label": label, "gap_r": round(rg, 4), "gap_t": round(tg, 1), "gap_n": ng,
             "close_r": round(rc, 4), "close_t": round(tc, 1)})
        print(f"{label:6s}{rg:13.3f}{tg:7.1f}{rc:13.3f}{tc:7.1f}")

    print()
    print("=" * 96)
    print("B. 外盘（A 股开盘前已知：前一交易日收盘）→ A 股 T 日")
    print("=" * 96)
    print(f"{'组合':16s}{'样本':>6s}{'lag0 r':>9s}{'T-1 领先 r':>11s}{'t':>7s}{'IC':>8s}")
    for label, code, cname in US_PAIRS:
        codes = (concepts.get(cname) or {}).get("codes") or []
        board = board_returns(codes)
        os_ = us_index_series(code)
        if board is None or os_ is None:
            print(f"{label:16s} 跳过")
            continue
        os_ret = os_.pct_change() * 100
        a_days = board.dropna().index
        prev = (os_ret.reindex(os_ret.index.union(a_days)).ffill()
                .reindex(a_days).shift(1))
        r, n, t = _pearson(*_aligned(prev, board, 0))
        ic = ic_spearman(prev, board)
        r0, n0, _t0 = _pearson(*_aligned(os_ret, board, 0))
        row = {"label": label, "code": code, "concept": cname, "codes": len(codes),
               "lag0": {"r": r0, "n": n0},
               "prev_day": {"r": r, "n": n, "t": t}, "ic": round(ic, 4)}
        result["us"].append(row)
        print(f"{label:16s}{n:6d}{r0:9.3f}{r:11.3f}{t:7.1f}{ic:8.3f}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n已写入 {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
