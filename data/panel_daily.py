# -*- coding: utf-8 -*-
"""面板日增量 —— 把**当天全市场日线**并入"覆盖层"（2026-10-26）。

## 为什么需要它
`data/panel/by_code` 是 2026-10-18 一次性建的（5,830 只 / 1,125 万行），
建完**没有任何维护机制** ⇒ 实测 10-26 时面板 `date_max` 还停在 **10-16**。
后果不是"少几根 K 线"这么轻：
- `desk.local_closes()` 用它给持仓估值 ⇒ 交易台显示的是 **10 天前**的收盘价；
- 深度分析算 MA60 用的是 10 天前的均线；
- 因子/回测的"最新截面"整体滞后 7 个交易日。

## 为什么写"覆盖层"而不是重写 by_code
重写 5,830 个 parquet 需要读完全部 28 个分片（≈1.4 GB）再逐只覆写（实测 **167 s**，
且期间读者可能读到写了一半的文件——Windows 上 `to_parquet` 是先 truncate 再写）。
本模块改成**每天写一个小文件** `data/panel/daily_overlay/{date}.parquet`
（5,570 行 / ≈300 KB，原子 `os.replace`），由 `data/bars._load_raw()` 在读取时叠加：

    by_code/{code}.parquet（历史） + daily_overlay/*.parquet（最近若干天） → get_bars()

写侧 O(1)、读侧多一次小 parquet（进程内按文件戳缓存，只解析一次）。
消费方**无需任何改动** —— `get_bars()` 本来就是唯一入口。

## 口径
- 价量来自 tushare `daily(trade_date)`（**1 次请求拿全市场**，50 次/分、8000 次/天）。
- 单位换算只发生在 `panel_store.normalize_tushare_daily_df`（vol 手→股、amount 千元→元）。
- 估值列（`pe_ttm/pb_mrq/ps_ttm/turn`）**留空**——`daily_basic` 无权限，
  绝不拿别处 PE 冒充（口径不同，见 `core/local_screen` 的 PE 踩坑记录）。
- 同一天重复跑会**覆盖**该日文件（幂等）。
"""
from __future__ import annotations

import logging
import os
import time

import pandas as pd

from data import bars, panel_store

_logger = logging.getLogger(__name__)


def _iso(d) -> str:
    s = str(d or "").replace("-", "").replace("/", "")
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return str(d)[:10]


def overlay_dir() -> str:
    return bars.OVERLAY_DIR


def fetch_trade_day(trade_date: str) -> pd.DataFrame:
    """当日全市场日线（tushare `daily`，一次请求）。"""
    from data import tushare_source as ts
    raw = ts.fetch_daily(_iso(trade_date).replace("-", ""))
    return panel_store.normalize_tushare_daily_df(raw)


def append_trade_day(trade_date: str, *, frames: pd.DataFrame | None = None,
                     dry_run: bool = False) -> dict:
    """把 `trade_date` 写成覆盖层文件。返回摘要（失败返回 `ok:False`，不抛）。"""
    t0 = time.time()
    date = _iso(trade_date)
    try:
        df = frames if frames is not None else fetch_trade_day(date)
    except Exception as e:                                # noqa: BLE001
        _logger.warning("[面板增量] %s 取数失败: %s: %s", date, type(e).__name__,
                        str(e)[:200])
        return {"ok": False, "trade_date": date, "reason": f"{type(e).__name__}: {e}"}

    if df is None or df.empty:
        return {"ok": False, "trade_date": date,
                "reason": "当日日线为空（非交易日 或 接口无权限）"}

    df = df.copy()
    df["trade_date"] = df["trade_date"].map(_iso)
    df = df[df["trade_date"] == date]
    if df.empty:
        return {"ok": False, "trade_date": date,
                "reason": f"取回的行里没有 {date}（接口返回了别的日期）"}

    cols = list(panel_store.PANEL_COLUMNS)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    out = df[cols].copy()
    out["stock_code"] = out["stock_code"].astype(str)
    for c in ("open", "high", "low", "close", "preclose", "volume", "amount",
              "pct_chg", "turn", "pe_ttm", "pb_mrq", "ps_ttm"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    out = out.drop_duplicates(subset=["stock_code"], keep="last")

    path = os.path.join(overlay_dir(), f"{date}.parquet")
    if not dry_run:
        os.makedirs(overlay_dir(), exist_ok=True)
        tmp = path + ".tmp"
        out.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, path)                             # 原子：读者不会看到半个文件
        bars.clear_cache()                                # 让 _overlay_map / panel_date_max 重算

    elapsed = round(time.time() - t0, 1)
    _logger.info("[面板增量] %s 覆盖层写入 %d 只 → %s（%.1fs）%s",
                 date, len(out), os.path.basename(path), elapsed,
                 "（dry_run 未写盘）" if dry_run else "")
    return {"ok": True, "trade_date": date, "codes": len(out), "path": path,
            "rows": len(out), "elapsed_s": elapsed, "dry_run": dry_run,
            "date_max": bars.panel_date_max() if not dry_run else None}


def overlay_dates() -> list[str]:
    """覆盖层已有的日期（升序）。"""
    try:
        return sorted(f[:-len(".parquet")] for f in os.listdir(overlay_dir())
                      if f.endswith(".parquet"))
    except FileNotFoundError:
        _logger.debug("[面板增量] 覆盖层目录尚不存在（首次运行）: %s", overlay_dir())
        return []
