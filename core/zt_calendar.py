# -*- coding: utf-8 -*-
"""涨停日历 + 市场宽度（B2②，2026-10-19）—— 从**本地面板**推算，零联网。

为什么需要：`core/market_detector` 的市场宽度原来抽样 12 只逐只
`ak.stock_zh_a_hist`（每只 3 源重试 ⇒ 单次 detect 约 15 次外呼），
涨停三指标依赖 `ak.stock_zt_pool_em`（1 次外呼，且离线不可用）。
本模块用一份**最近 N 个交易日**的面板切片同时喂：

  * 市场宽度（站上 MA20 占比）
  * 涨停家数 / 跌停家数（T 日，即面板最新交易日，下同）
  * 最高连板（截至 T 的当前连续涨停天数）
  * 晋级率（T 日涨停中 T-1 也涨停的占比 —— 与东财池「连板数≥2 占比」同口径）

⚠️ 口径差异（必须随结果展示）：
  * 本地版基于 **T-1/T 日收盘**（面板最新交易日），不是盘中实时池 ——
    盘前 09:25 检测时东财池本来就是空的，本地版反而有「昨日」数据可用；
  * 涨跌停幅度按**板块 + 改制日期**判定（见 `_limit_ratio`）：
    北交所 30% / 科创板 20% / 创业板 2020-08-24 前 10%、之后 20% /
    沪深主板 非 ST 10%、ST 在 2026-07-06 前 5%、之后 10%。
    **ST 只影响主板**——双创与北交所的 ST 与板块同幅度。
  * `is_st` 列由 BaoStock 分块回填（0017~0027，约 2,000 只 / 全历史 100%）；
    按日切片的老分片（0003~0016）没有该列 ⇒ 合并后最新交易日覆盖 100%、
    长区间约 36%。缺 `is_st` 时按板块幅度判，只在"2026-07-06 之前的主板 ST"
    这一种组合上造成漏判（其余两种板块与 ST 同幅度）；
  * 涨停价 = `round(preclose × (1+limit), 2)`（四舍五入到分），close 达到即判停
    （含尾盘炸板的"曾涨停"——东财池是收盘仍封板口径，家数会略高，作为近似标注）。
"""
import logging
import os
from functools import lru_cache

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# 数据根必须**运行时**解析：打包后 `__file__` 位于只读包 `_internal/` 里，
# 用它推导会指向空目录 ⇒ 读到空数据（2026-10-25 实测：双击 exe 后
# 选股/复盘/研究全 missing、交易台变空账户、系统页看不到调度器）。
# 规则见 `config/paths.data_root()`；源码运行结果与原先一致。
from config.paths import data_root as _paths_data_root  # noqa: E402
PROJECT_ROOT = _paths_data_root()
PANEL_PARTS_DIR = os.path.join(PROJECT_ROOT, "data", "panel", "parts")
# 日更增量（每天一个小文件）。**必须一起读**，否则看到的是"最后一次重建面板那天"：
# 实测 2026-10-29 时 parts 最新只到 10-19（且那天只有按股票分块的 1,371 只），
# 而 overlay 已到 10-28 / 5,570 只 ⇒ 面板上显示的是 7 个交易日前、1/4 样本的家数。
PANEL_OVERLAY_DIR = os.path.join(PROJECT_ROOT, "data", "panel", "daily_overlay")
_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "panel", "cache_recent_panel.parquet")

_COLS = ["stock_code", "trade_date", "close", "preclose", "is_st", "trade_status"]

# 涨跌停幅度的两个**改制分界日**（均由本地面板反推，不是查来的规定：
# 见 `_limit_ratio` 注释里的实测证据）。
GEM_REFORM_DATE = "2020-08-24"        # 创业板注册制改革：10% → 20%
MAIN_ST_10PCT_DATE = "2026-07-06"     # 沪深主板 ST/*ST：5% → 10%


def _as_date(v) -> str | None:
    """把 trade_date 归一成 `YYYY-MM-DD`；缺失/脏值返回 None。

    不写 `try/except`（`pd.isna` 对数组会抛）：直接看字符串形态，
    NaN→"nan"、NaT→"NaT"、None→"None" 都落不到 `YYYY-MM-DD` 这个形状上。
    """
    if v is None:
        return None
    s = str(v)[:10]
    return s if len(s) == 10 and s[4] == "-" and s[:4].isdigit() else None


def _is_st_flag(is_st) -> bool:
    """is_st 可能是 NaN / 字符串 / 数字，统一成 bool（脏值按"非 ST"）。"""
    try:
        if is_st is None or pd.isna(is_st):
            return False
    except (TypeError, ValueError) as e:                 # noqa: BLE001
        _logger.debug("[zt_calendar] is_st 脏值（%r: %s）⇒ 按非 ST 处理", is_st, e)
        return False
    try:
        return int(float(is_st)) == 1
    except (TypeError, ValueError):
        return str(is_st).strip() in ("1", "1.0", "True", "true")


def _limit_ratio(code: str, is_st, trade_date=None) -> float:
    """该股的涨跌停幅度 —— **按板块 + 日期**决定，ST 只影响主板。

    | 板块              | 非 ST | ST |
    |-------------------|-------|----|
    | 北交所            | 30%   | 30% |
    | 科创板 688/689    | 20%   | 20% |
    | 创业板 300/301    | 2020-08-24 起 20% | 2020-08-24 起 20% |
    |                   | 之前 10% | 之前 5% |
    | 沪深主板          | 10%   | 2026-07-06 起 10%，之前 5% |

    `trade_date` 缺失时按**当前**规则（主板 ST 10%、创业板 20%）——宁可对现在准，
    不能因为缺日期把今天的 ST 判回 5%。

    ⚠️ 历史坑 1（2026-10-29 修）：旧实现"先判 ST 一律 5%"，双创/北交所 ST
    被用过小幅度 ⇒ 涨 5% 就被当成涨停。**实测 2026-10-19 跌停报 14 实为 2（虚高 7 倍）**，
    12 只误报里 8 只是创业板 ST、4 只是主板 ST。
    ⚠️ 历史坑 2：也别改成"板块优先 + ST 一律 5%"——主板 ST 自 2026-07-06 起已是 10%，
    那一版在近 60 个交易日里**仍有 55 天**结果不对。

    分界日的实测依据（本地面板，BaoStock 回填分块，is_st 全覆盖）：
      · 主板 ST 逐日最大涨幅：2026-07-03 为 5.30%（≥9.8% 命中 0 天）；
        2026-07-06 起出现 9.96%，07-08 起 ≥9.8% 命中 5 天 ⇒ 分界 2026-07-06。
      · 创业板逐日最大涨幅：2020-08-21 为 10.07%（≥19.5% 命中 0 天）；
        2020-08-24 起 20.00%（≥19.5% 命中 2 天）⇒ 分界 2020-08-24。
    """
    st = _is_st_flag(is_st)
    c = str(code)
    d = _as_date(trade_date)

    if c.startswith(("83", "87", "43", "92", "88")):        # 北交所
        return 0.30
    if c.startswith(("688", "689")):                        # 科创板
        return 0.20
    if c.startswith(("300", "301")):                        # 创业板
        if d is None or d >= GEM_REFORM_DATE:
            return 0.20                                     # 改革后 ST 也是 20%
        return 0.05 if st else 0.10
    # 沪深主板
    if st:
        return 0.10 if (d is None or d >= MAIN_ST_10PCT_DATE) else 0.05
    return 0.10


def _load_recent(days: int = 40, *, use_cache: bool = True) -> pd.DataFrame | None:
    """最近 `days` 个**自然日**的面板切片（列子集）。默认 40 自然日 ≈ 27 个
    交易日 —— 市场宽度需要每股 ≥21 根（MA20），25 自然日只有 ~17 根不够
    （2026-10-19 实测 above_ma20_checked=0）。

    ⚠️ 不走 `load_panel`：它读全列 + 全量 dedup+sort（11.3M 行是分钟级）。
    这里逐 part 读**列子集**（pyarrow 列下推），读入后内存过滤
    （⚠️ 不用 pyarrow `filters`：trade_date 是 category 列，类型不匹配会把
    几乎全部行滤掉——2026-10-19 实测只剩 23 行）。落盘缓存按**数据新鲜度**
    键控：缓存里最新交易日距今 ≤3 天就复用，否则重建（周末不反复重建）。
    """
    import glob

    def _latest_overlay_date() -> str | None:
        fs = sorted(glob.glob(os.path.join(PANEL_OVERLAY_DIR, "*.parquet")))
        return os.path.basename(fs[-1])[:10] if fs else None

    if use_cache and os.path.exists(_CACHE_PATH):
        try:
            cached = pd.read_parquet(_CACHE_PATH)
            if len(cached):
                newest = pd.to_datetime(cached["trade_date"].astype(str).str[:10]).max()
                ov = _latest_overlay_date()
                # ⚠️ 只判"≤3 天"是不够的：昨天建的缓存今天仍算新鲜，会漏掉今天新落的
                #    overlay ⇒ 涨停家数永远慢一天。所以再加一条"overlay 有更新就重建"。
                if (pd.Timestamp.now().normalize() - newest).days <= 3 and not (
                        ov and str(newest)[:10] < ov):
                    return cached
                _logger.info("[zt_calendar] 缓存过期（最新 %s，overlay %s）⇒ 重建",
                             newest.date(), ov)
        except Exception as e:                              # noqa: BLE001
            _logger.warning("[zt_calendar] 缓存读取失败（重建）: %s", e)

    parts = sorted(glob.glob(os.path.join(PANEL_PARTS_DIR, "*.parquet")))
    if not parts:
        return None
    start = (pd.Timestamp.now().normalize() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    frames = []
    for p in parts:
        try:
            d = pd.read_parquet(p, columns=_COLS)
        except (ValueError, OSError):
            # 老分片可能缺 is_st 列 ⇒ 退化为可用列
            try:
                d = pd.read_parquet(p, columns=["stock_code", "trade_date",
                                                "close", "preclose"])
            except Exception as e:                          # noqa: BLE001
                _logger.warning("[zt_calendar] 分片 %s 读取失败: %s", p, e)
                continue
        d["trade_date"] = d["trade_date"].astype(str).str[:10]
        d = d[d["trade_date"] >= start]
        if not d.empty:
            keep = [c for c in _COLS if c in d.columns]
            frames.append(d[keep].astype({"close": "float32", "preclose": "float32"}))
    # 日更增量：放在 parts **之后** ⇒ 同一天的行以 overlay 为准（dedup keep="last"）。
    # overlay 只有行情列（无 is_st / trade_status），缺的列由下面统一补 NaN。
    for p in sorted(glob.glob(os.path.join(PANEL_OVERLAY_DIR, "*.parquet"))):
        try:
            d = pd.read_parquet(p)
        except (ValueError, OSError) as e:
            _logger.warning("[zt_calendar] overlay %s 读取失败: %s", p, e)
            continue
        if "trade_date" not in d.columns:
            continue
        d["trade_date"] = d["trade_date"].astype(str).str[:10]
        d = d[d["trade_date"] >= start]
        if d.empty:
            continue
        keep = [c for c in _COLS if c in d.columns]
        frames.append(d[keep].astype({"close": "float32", "preclose": "float32"}))
    if not frames:
        return None
    # 老分片的 is_st 全为 NA ⇒ concat 触发 pandas 2.2 的 all-NA dtype 过渡警告；
    # 结果行为符合预期（统一为 float NaN），显式忽略这一条而不是全局关警告。
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        df = pd.concat(frames, ignore_index=True)
    df = (df.sort_values(["stock_code", "trade_date"])
            .drop_duplicates(["stock_code", "trade_date"], keep="last"))
    df = df[df["trade_date"] >= start].reset_index(drop=True)
    if use_cache and len(df):
        try:
            tmp = _CACHE_PATH + ".tmp"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, _CACHE_PATH)
        except Exception as e:                              # noqa: BLE001
            _logger.warning("[zt_calendar] 缓存写入失败（不影响本次）: %s", e)
    return df


def clear_cache() -> None:
    """清进程内缓存 + 删除落盘缓存（回填完成后调用）。"""
    limit_up_stats.cache_clear()
    try:
        if os.path.exists(_CACHE_PATH):
            os.remove(_CACHE_PATH)
    except OSError as e:
        _logger.warning("[zt_calendar] 缓存删除失败: %s", e)


@lru_cache(maxsize=2)
def limit_up_stats(days: int = 40) -> dict:
    """涨停/跌停/连板/晋级率（基于面板最新交易日 T，零联网）。

    Returns:
        {"available": bool, "trade_date": "YYYY-MM-DD",
         "limit_up_count", "limit_down_count", "max_consecutive",
         "upgrade_rate", "above_ma20_ratio", "stocks": int,
         "caveat": str}
    """
    df = _load_recent(days)
    if df is None or df.empty:
        return {"available": False, "caveat": "本地面板无近期数据"}
    # 老分片/缓存可能缺 is_st / trade_status 列 ⇒ 补 NaN 列（zip 会炸在 None 上）
    for _c in ("is_st", "trade_status"):
        if _c not in df.columns:
            df[_c] = float("nan")
    df["trade_date"] = df["trade_date"].astype(str)
    t = str(df["trade_date"].max())[:10]
    day = df[df["trade_date"] == t].copy()
    # 正常交易行（trade_status 缺失视为正常）
    if "trade_status" in day.columns:
        day = day[(day["trade_status"].isna()) | (day["trade_status"] == 1)]
    day = day.dropna(subset=["close", "preclose"])
    day = day[day["preclose"] > 0]
    if day.empty:
        return {"available": False, "caveat": f"面板最新交易日 {t} 无有效行情"}

    ratios = [_limit_ratio(c, s, d) for c, s, d in
              zip(day["stock_code"], day.get("is_st"), day["trade_date"])]
    day = day.assign(_limit=ratios)
    limit_price_up = (day["preclose"] * (1 + day["_limit"])).round(2)
    limit_price_dn = (day["preclose"] * (1 - day["_limit"])).round(2)
    is_up = day["close"] >= limit_price_up - 1e-6
    is_dn = day["close"] <= limit_price_dn + 1e-6
    limit_up_count = int(is_up.sum())
    limit_down_count = int(is_dn.sum())

    # 连板：每股按日期升序数"截至 T 的当前连续涨停天数"
    df = df.dropna(subset=["close", "preclose"])
    df = df[df["preclose"] > 0]
    df = df.assign(_limit=[_limit_ratio(c, s, d) for c, s, d in
                           zip(df["stock_code"], df.get("is_st"), df["trade_date"])])
    df["_up"] = df["close"] >= (df["preclose"] * (1 + df["_limit"])).round(2) - 1e-6
    max_consecutive = 0
    for _code, g in df.groupby("stock_code", sort=False, observed=True):
        g = g.sort_values("trade_date")
        run = 0
        for v in g["_up"].to_numpy():
            run = run + 1 if v else 0
        if run > max_consecutive:
            max_consecutive = run

    # 晋级率（与东财池「连板数≥2 占比」同口径）：T 日涨停 且 T-1 也涨停
    prev_t = str(sorted(set(df["trade_date"]))[-2])[:10] if df["trade_date"].nunique() >= 2 else None
    if prev_t:
        prev_up = set(df.loc[(df["trade_date"] == prev_t) & df["_up"], "stock_code"])
        today_up = set(day.loc[is_up, "stock_code"])
        upgraded = len(today_up & prev_up)
        upgrade_rate = round(upgraded / len(today_up) * 100, 1) if today_up else 0.0
    else:
        upgraded, upgrade_rate = 0, 0.0

    # 市场宽度：站上 MA20 占比（每股至少 21 根）
    above, checked = 0, 0
    for _code, g in df.groupby("stock_code", sort=False, observed=True):
        g = g.sort_values("trade_date")
        closes = g["close"].to_numpy(dtype="float64")
        if len(closes) < 21:
            continue
        checked += 1
        if closes[-1] > closes[-20:].mean():
            above += 1
    above_ratio = round(above / checked * 100, 1) if checked else None

    # ⚠️ 方向要说清（2026-10-29）：幅度改为"板块 + 日期"后，`is_st` 缺失**不再造成
    #    过计数**（双创/北交所 ST 与板块同幅度）。残留误差只有一个方向：
    #    真实主板 ST 在 2026-07-06 之前是 5%，缺 is_st 时会按 10% 判 ⇒ 5%~10% 之间漏判。
    na_ratio = (float(day["is_st"].isna().mean())
                if "is_st" in day.columns else 1.0)
    caveat = ("本地推算（面板 %s 收盘口径，含尾盘炸板；与东财实时池存在口径差）"
              % t)
    if na_ratio > 0.5:
        caveat += ("；is_st 回填不足（%.0f%% 缺失）⇒ 这批股票按板块幅度判，"
                   "其中真实主板 ST 在 2026-07-06 之前（当时 5%%）"
                   "涨跌幅在 5%%~10%% 时会被漏判" % (na_ratio * 100))
    elif na_ratio > 0:
        caveat += "；is_st 仍有 %.0f%% 未回填（影响有限）" % (na_ratio * 100)

    return {"available": True, "trade_date": t,
            "limit_up_count": limit_up_count, "limit_down_count": limit_down_count,
            "max_consecutive": max_consecutive,
            "upgrade_rate": upgrade_rate, "upgraded": upgraded,
            "above_ma20_ratio": above_ratio, "above_ma20_checked": checked,
            "stocks": int(len(day)), "caveat": caveat}
