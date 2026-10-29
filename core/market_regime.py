# -*- coding: utf-8 -*-
"""市场强弱判定（**单一来源**，与 `scripts/validate_trend_gaps.py` 同口径）。

为什么单独一个模块
-----------------
P3-15 的趋势否决/超跌结论是在「**中证1000 收盘价 ≤ 其 MA20**」这个口径上验证的。
`MarketDetector` 的 `market_state` 是 14 指标综合评分，**不是同一把尺子**——
拿它去套验证过的阈值，就是"口径不一致"（本项目反复踩的坑）。
所以这里直接复用验证时的定义，读本地指数面板，**零联网、毫秒级**。

铁律：数据不足返回 None（缺就是缺），**绝不默认成强市或弱市**。
"""
from __future__ import annotations

import glob
import logging
import os
from typing import Optional

import pandas as pd

from data import panel_store

_logger = logging.getLogger(__name__)

INDEX_DIR = os.path.join(panel_store.PANEL_DIR, "index")
DEFAULT_INDEX = "sh000852"      # 中证1000（与验证脚本一致）
DEFAULT_WINDOW = 20


def _index_file(index_code: str, index_dir: Optional[str] = None) -> Optional[str]:
    base = index_dir or INDEX_DIR
    if not os.path.isdir(base):
        return None
    hits = [f for f in glob.glob(os.path.join(base, "*.parquet"))
            if index_code in os.path.basename(f)]
    return sorted(hits)[0] if hits else None


def ma20_regime(index_code: str = DEFAULT_INDEX, window: int = DEFAULT_WINDOW,
                index_dir: Optional[str] = None) -> Optional[dict]:
    """返回 ``{"weak": bool, "as_of": str, "close": float, "ma": float, "index": str}``。

    ``weak=True`` 表示最新收盘价 ≤ 最近 ``window`` 根收盘均值（含当根）。
    任何一步数据不够都返回 ``None``：文件不存在、读取失败、有效收盘不足 window 根。
    """
    if window < 2:
        return None
    path = _index_file(index_code, index_dir)
    if not path:
        return None
    try:
        df = pd.read_parquet(path, columns=["date", "close"])
    except (OSError, ValueError) as e:
        # 记日志再返回 None：可观测性基线禁止静默 except，且"读不出"必须可追查
        _logger.warning("[市场强弱] 指数面板读取失败 %s: %s", path, e)
        return None
    if df.empty:
        return None
    df = df.sort_values("date")
    close = pd.to_numeric(df["close"], errors="coerce")
    mask = close.notna() & (close > 0)
    close = close[mask]
    if len(close) < window:
        return None
    ma = float(close.iloc[-window:].mean())
    last = float(close.iloc[-1])
    return {
        "weak": bool(last <= ma),
        "as_of": str(df.loc[mask, "date"].iloc[-1]),
        "close": last,
        "ma": ma,
        "index": os.path.basename(path),
    }
