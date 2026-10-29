# -*- coding: utf-8 -*-
"""市场因子（**全市场等权收益**，单一来源）—— IVOL 的市场侧输入。

为什么要这个模块
----------------
`IVOL = 总波动率 − β²·市场波动`，所以必须有一条**市场收益序列**。项目里
现成的市场序列有三条，口径各不相同，混用会让「同一个 IVOL」在不同页面算出
不同的数：

| 序列 | 来源 | 与验证脚本一致？ |
|---|---|---|
| 全市场**等权**日收益 | 本模块（面板 close/preclose） | ✅ 是 |
| 中证1000 指数收益 | `data/panel/index/sh000852.parquet` | ❌ 不是 |
| 候选池等权收益 | 每次选股的候选集 | ❌ 更不是（池子天天变） |

`scripts/validate_literature_factors.py::rolling_stats()` 里算 IVOL 时用的是
**全市场等权**（`mkt = Σr / 有效只数`），IC −0.0826 那个结论就建立在这条序列上。
本模块把它固化成唯一实现，选股/回测/页面都从这里取。

成本（实测 2026-10-29）
-----------------------
读 28 个分片的 `close/preclose` 两列 + 分组求均值 = **6.6 秒**（2,619 个交易日，
5,600+ 只/日）。放进选股热路径不可接受，所以**落盘缓存**：

    data_root()/data/panel/market_eqw_ret.parquet

首次构建 6.6s，之后每次只是 `pd.read_parquet`（约 10ms）。缓存过期判据是
「缓存最新交易日 < 面板最新交易日」—— 与 `zt_calendar` 同一条教训：
只看文件时间会被昨天的缓存骗过去（overlay 天天新增）。

铁律
----
1. **缺就是缺**：面板读不到 / 有效日不足 ⇒ 返回 None，绝不返回一条全 0 序列
   （全 0 会让 β 不可识别、IVOL 退化成总波动率，而且**看不出**降级了）。
2. 本模块**零联网**。
3. 缓存写入用 `os.replace` 原子替换：读者不会读到半个文件。
"""
from __future__ import annotations

import glob
import logging
import os
from typing import Optional

import pandas as pd

from config.paths import data_root as _paths_data_root

_logger = logging.getLogger(__name__)

PROJECT_ROOT = _paths_data_root()
PANEL_PARTS_DIR = os.path.join(PROJECT_ROOT, "data", "panel", "parts")
PANEL_OVERLAY_DIR = os.path.join(PROJECT_ROOT, "data", "panel", "daily_overlay")
CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "panel", "market_eqw_ret.parquet")

# 至少要有这么多交易日才算"可用"（IVOL 窗口 20 日 + 余量）
MIN_DAYS = 30

# 进程内缓存（键 = 缓存文件 mtime，文件一更新就失效）
_MEM: dict = {}


def _panel_date_max() -> Optional[str]:
    """面板最新交易日（parts 与 overlay 取较晚者）。"""
    newest = None
    for path in sorted(glob.glob(os.path.join(PANEL_PARTS_DIR, "*.parquet"))):
        try:
            d = pd.read_parquet(path, columns=["trade_date"])
        except (OSError, ValueError) as e:
            _logger.warning("[市场因子] 分片 %s 读取失败（跳过）: %s",
                            os.path.basename(path), e)
            continue
        if len(d):
            newest = max(newest or "", str(d["trade_date"].max())[:10])
    for path in sorted(glob.glob(os.path.join(PANEL_OVERLAY_DIR, "*.parquet"))):
        newest = max(newest or "", os.path.basename(path)[:10])
    return newest


def _build() -> Optional[pd.Series]:
    """从面板重算全市场等权日收益（date 字符串索引，升序）。"""
    frames = []
    for path in sorted(glob.glob(os.path.join(PANEL_PARTS_DIR, "*.parquet"))):
        try:
            d = pd.read_parquet(path, columns=["stock_code", "trade_date",
                                               "close", "preclose"])
        except (OSError, ValueError) as e:
            _logger.warning("[市场因子] 分片 %s 读取失败（跳过）: %s",
                            os.path.basename(path), e)
            continue
        frames.append(d)
    # overlay 放在 parts 之后；同一天两者的样本不同（parts 某些日期只有分块子集），
    # 所以 **不能** dedup 掉整天的 parts —— 只按 (code, date) 去重，让 overlay 赢。
    for path in sorted(glob.glob(os.path.join(PANEL_OVERLAY_DIR, "*.parquet"))):
        try:
            d = pd.read_parquet(path, columns=["stock_code", "trade_date",
                                               "close", "preclose"])
        except (OSError, ValueError) as e:
            _logger.warning("[市场因子] overlay %s 读取失败（跳过）: %s",
                            os.path.basename(path), e)
            continue
        frames.append(d)
    if not frames:
        _logger.warning("[市场因子] 面板分片为空 ⇒ 无市场收益序列")
        return None
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return None
    df["trade_date"] = df["trade_date"].astype(str).str[:10]
    df = df.drop_duplicates(["stock_code", "trade_date"], keep="last")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["preclose"] = pd.to_numeric(df["preclose"], errors="coerce")
    # ⚠️ 不剔除 `close == preclose`（平盘 / 停牌填充）。验证脚本也没剔——
    #    平盘日收益本来就是 0，剔掉会让等权均值系统性偏离（震荡市里平盘股能有几百只）。
    #    这里只挡"不可计算"的行（价格缺失/为 0）。
    df = df[(df["close"] > 0) & (df["preclose"] > 0)]
    if df.empty:
        return None
    ret = (df["close"] / df["preclose"] - 1.0).groupby(df["trade_date"]).mean()
    ret = ret.sort_index()
    return ret if len(ret) >= MIN_DAYS else None


def equal_weight_returns(*, refresh: bool = False) -> Optional[pd.Series]:
    """全市场等权日收益（小数），index = `YYYY-MM-DD` 字符串。

    缓存优先；缓存缺失或**落后于面板最新交易日**时重算并落盘。
    返回 None 表示"这条序列拿不到"（调用方必须自行降级并声明口径）。
    """
    mtime = os.path.getmtime(CACHE_PATH) if os.path.exists(CACHE_PATH) else 0.0
    if not refresh and mtime:
        hit = _MEM.get("mtime")
        if hit == mtime and _MEM.get("ret") is not None:
            return _MEM["ret"]
        try:
            ret = pd.read_parquet(CACHE_PATH)["ret"]
            ret.index = ret.index.astype(str)
            newest = str(ret.index.max())[:10]
            panel_max = _panel_date_max()
            if panel_max and newest < panel_max:
                _logger.info("[市场因子] 缓存落后（%s < 面板 %s）⇒ 重算", newest, panel_max)
            elif len(ret) >= MIN_DAYS:
                _MEM.update(mtime=mtime, ret=ret)
                return ret
        except (OSError, ValueError, KeyError) as e:
            _logger.warning("[市场因子] 缓存读取失败（重算）: %s", e)

    ret = _build()
    if ret is None:
        # 重算失败但旧缓存还在 ⇒ 宁可给旧序列 + 明确日志，也不要让整条链降级
        if mtime:
            try:
                old = pd.read_parquet(CACHE_PATH)["ret"]
                old.index = old.index.astype(str)
                _logger.warning("[市场因子] 重算失败，沿用旧缓存（最新 %s）",
                                str(old.index.max())[:10])
                return old
            except (OSError, ValueError, KeyError) as e:
                _logger.warning("[市场因子] 旧缓存也不可用（返回 None 降级）: %s", e)
        return None
    try:
        out = ret.to_frame("ret")
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        tmp = CACHE_PATH + ".tmp"
        out.to_parquet(tmp)
        os.replace(tmp, CACHE_PATH)
        _MEM.update(mtime=os.path.getmtime(CACHE_PATH), ret=ret)
    except (OSError, ValueError) as e:
        _logger.warning("[市场因子] 缓存写入失败（不影响本次）: %s", e)
    return ret


def equal_weight_index(*, refresh: bool = False) -> Optional[pd.Series]:
    """等权收益累乘成的**点位序列**（起点 1000）—— 给 `indicators.residual_vol` 用。

    为什么要点位而不是收益：`residual_vol` 的契约是"两侧都传收盘价序列"，
    传收益进去会被再差分一次（结果恒接近 0，而且不会报错）。累乘可逆，
    收益序列与点位序列算出的残差完全相同。
    """
    ret = equal_weight_returns(refresh=refresh)
    if ret is None or ret.empty:
        return None
    level = (1.0 + ret.fillna(0.0)).cumprod() * 1000.0
    level[ret.isna()] = float("nan")                   # 缺失日不要伪装成平盘
    return level


def clear_cache() -> None:
    """清进程内缓存（回填/面板重建后调用）。落盘文件保留，靠新鲜度判据自动重建。"""
    _MEM.clear()
