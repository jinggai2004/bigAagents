# -*- coding: utf-8 -*-
"""本地日线**单一入口**：`get_bars(code, start, end, as_of, adjust)`（P0-1，2026-10-18）。

为什么需要它（这一条解决了三个症状）：
  - 选股逐票联网 `get_stock_data(code, period="3mo")` 实测 **≈51 s/只**
    ⇒ 5 只 4.3 分钟；
  - `factor_analysis` / `condition_backtest` / `backtest` 各自联网取数
    ⇒ 因子稳健性只能在 30 只票上算；
  - 面板已建成 **1125.7 万行 / 5830 只 / 2016-01-04~2026-10-16**，却 `load_panel()`
    零调用方 —— 最大的一笔沉没成本。

两件事：
  1. **按股票索引**（`data/panel/by_code/{code}.parquet`）：按日分片（截面布局）
     取单只要读全部 17 片（分钟级）；转成按股票后单只 **5~20 ms**。
  2. **口径统一**：面板存的是**未复权**价（BaoStock adjustflag=3 / tushare daily 原始价），
     而消费方默认要前复权（除权跳空会让 MA/动量出假信号）。这里用面板自带的
     `preclose/close` 比值链重建复权因子（**不是 `pct_chg` 累乘**——后者得到的是
     收益序列而非价格缩放，除权日仍会跳空，见 `_qfq` 的推导与踩坑记录）。
     `adjust="qfq"` 锚最后一根（与联网 qfq 同基准）、`"hfq"` 锚第一根、`"none"` 原样。

⚠️ 与 `sina_source.fetch_stock_daily` 的 schema 对齐（`date open high low close
   volume amount outstanding_share turnover_pct`，`date` 为 datetime）——
   消费方改调本模块时无需改下游计算代码。
   ⚠️ **`outstanding_share` 恒为 NaN**：面板不存流通股本，该列只为 schema 兼容保留，
     任何"按流通股本归一"的计算都**不能**用它。

⚠️ **qfq 锚点随切片变化**：锚点是"传入区间的最后一根"，所以
   `get_bars(as_of=X)` 与 `get_bars()` 在同一个 `date` 上的**绝对价位不同**（比率类指标
   如动量/波动率不受影响）。**跨模块比较绝对价位前，必须先统一 `end`/`as_of`。**

⚠️ 缺就是缺：本地无该股 ⇒ 返回 `None`（调用方自己决定"联网补"还是"标注不可评"）。
   本模块**不隐式联网**。
"""
from __future__ import annotations

import glob
import json
import bisect
import logging
import os
from functools import lru_cache
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from data import panel_store

_logger = logging.getLogger(__name__)

# 数据根必须**运行时**解析：打包后 `__file__` 在只读包 `_internal/` 里，
# 用它推导会指向空目录 ⇒ 读到空数据（2026-10-25 实测）。
# 规则见 `config/paths.data_root()`；源码运行结果与原先一致。
from config.paths import data_root as _paths_data_root  # noqa: E402
_PROJECT_ROOT = _paths_data_root()
BY_CODE_DIR = os.path.join(panel_store.PANEL_DIR, "by_code")
MANIFEST_PATH = os.path.join(panel_store.PANEL_DIR, "by_code_manifest.json")

# 落 by_code 的列（省掉 trade_status/is_st 以外的冗余；float32 减半体积）
_KEEP_COLS = ["stock_code", "trade_date", "open", "high", "low", "close",
              "preclose", "volume", "amount", "pct_chg", "turn",
              "pe_ttm", "pb_mrq", "ps_ttm", "trade_status", "is_st"]

# 对外 schema（与 sina_source.fetch_stock_daily 对齐）
_OUT_COLS = ["date", "open", "high", "low", "close", "volume", "amount",
             "outstanding_share", "turnover_pct"]

# 新鲜度默认阈值（**交易日**）：超过它即认为"这不是当前行情"。
# 10 个交易日 ≈ 两周 —— A 股正常停牌（重大事项）多在 5 日内复牌，
# 超过两周仍未更新，其 MA/动量已不能代表当前状态。
# ⚠️ 这是**默认业务阈值**，消费方可传自己的 `max_lag` 覆盖。
DEFAULT_MAX_LAG_TRADING_DAYS = 10


# ================================================================
# 1. 构建按股票索引
# ================================================================

def build_by_code_index(*, force: bool = False, progress_cb=None) -> dict:
    """按日分片 → `by_code/{code}.parquet`（一次性，约 3~6 分钟）。

    Args:
        force: True 时重建全部；False 时只补缺失的股票（增量）。
        progress_cb: `(done, total)` 回调（CLI 用）。

    Returns: {"codes", "written", "skipped", "rows", "elapsed_s"}
    """
    import time as _t
    t0 = _t.time()
    os.makedirs(BY_CODE_DIR, exist_ok=True)

    existing = {os.path.splitext(f)[0] for f in os.listdir(BY_CODE_DIR)
                if f.endswith(".parquet")}
    parts = sorted(glob.glob(os.path.join(panel_store.PARTS_DIR, "*.parquet")))

    # 逐分片 groupby 累加（比全量 concat 省 ~1GB 内存）
    acc: dict[str, list[pd.DataFrame]] = {}
    _date_min, _date_max = None, None
    for p in parts:
        d = pd.read_parquet(p)
        d["stock_code"] = d["stock_code"].astype(str)
        d["trade_date"] = d["trade_date"].map(panel_store._norm_date_str)
        # 面板覆盖区间（写进 manifest）：`panel_date_max()` 的兜底来源，
        # 也是"新鲜度闸门"的基准日——缺了它拿不到参考日，只能退回今天。
        if len(d):
            _dmin, _dmax = d["trade_date"].min(), d["trade_date"].max()
            _date_min = _dmin if _date_min is None else min(_date_min, _dmin)
            _date_max = _dmax if _date_max is None else max(_date_max, _dmax)
        cols = [c for c in _KEEP_COLS if c in d.columns]
        for code, g in d.groupby("stock_code", sort=False):
            if not force and code in existing:
                continue
            acc.setdefault(code, []).append(g[cols])

    written = 0
    rows = 0
    total = len(acc)
    for code, frames in acc.items():
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("trade_date").drop_duplicates(
            subset=["trade_date"], keep="last").reset_index(drop=True)
        # float32 减半体积；代码列不落（文件名即代码）
        for c in ("open", "high", "low", "close", "preclose", "volume",
                  "amount", "pct_chg", "turn", "pe_ttm", "pb_mrq", "ps_ttm"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        out = df.drop(columns=["stock_code"], errors="ignore")
        out.to_parquet(os.path.join(BY_CODE_DIR, f"{code}.parquet"),
                       index=False, compression="zstd")
        written += 1
        rows += len(out)
        if progress_cb and written % 200 == 0:
            progress_cb(written, total)
        if written % 500 == 0:
            _logger.info("[bars] by_code 已写 %d 只（%d 行）", written, rows)

    manifest = {
        "codes": len({os.path.splitext(f)[0] for f in os.listdir(BY_CODE_DIR)
                      if f.endswith(".parquet")}),
        "written": written, "skipped": len(existing) if not force else 0,
        "rows": rows, "parts": len(parts),
        # ⚠️ `date_min/date_max` 是**面板**覆盖（含未重写的股票），不是本次写入范围
        "date_min": _date_min, "date_max": _date_max,
        "elapsed_s": round(_t.time() - t0, 1),
        "at": _t.strftime("%Y-%m-%d %H:%M:%S"),
    }
    import json
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    os.replace(tmp, MANIFEST_PATH)
    _logger.info("[bars] by_code 索引完成: %s", manifest)
    return manifest


def read_manifest() -> dict | None:
    import json
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        _logger.debug("[bars] manifest 不可读（%s）⇒ 视为无", e)
        return None


def available_codes() -> set[str]:
    """本地已有日线的代码集合（毫秒级，只列目录）。"""
    if not os.path.isdir(BY_CODE_DIR):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(BY_CODE_DIR)
            if f.endswith(".parquet")}


def has_bars(code: str) -> bool:
    return os.path.exists(os.path.join(BY_CODE_DIR, f"{_clean(code)}.parquet"))


def _clean(code: str) -> str:
    """`sh.600028` / `600028.SH` / `600028` → `600028`。"""
    c = str(code).strip().lower()
    if "." in c:
        c = c.split(".")[-1] if c.split(".")[0] in ("sh", "sz", "bj") else c.split(".")[0]
    return c.zfill(6) if c.isdigit() else c


def offline_mode() -> bool:
    """`BIGA_OFFLINE=1` ⇒ **只读本地**，缺失即"不可评"（不联网、不重试）。

    为什么需要它（2026-10-18 实测）：断网跑选股时，本地缺失的少数代码
    （B 股 900xxx、新股）会走"联网兜底 ⇒ 多数据源 × 3 次重试"，
    单个代码几十秒 —— 断网自检/离线演示直接被拖成几分钟，
    而且**掩盖了真正的问题**（本地覆盖率不够，应该被看见而不是被重试掩盖）。

    ⚠️ D4（2026-10-18）：**光有约定不够**——实测只有 4 处 K 线兜底判断了它，
    打分仍会打网络。现在它同时**确保离线闸门已安装**
    （`data.offline_gate`：任何外呼抛 `OfflineViolation`）。
    所以"检查 offline_mode()"与"真的不能联网"是同一件事，不再依赖自觉。
    """
    from data import offline_gate
    return offline_gate.ensure() or offline_gate.is_offline()


# ================================================================
# 2. 读取（带进程内 LRU）
# ================================================================

@lru_cache(maxsize=256)
def _load_base(code: str) -> pd.DataFrame | None:
    """读 `by_code/{code}.parquet`（未复权）。缓存 256 只 ≈ 90MB。

    ⚠️ 缓存的是**未复权原始行**，区间/复权在 `get_bars` 里按调用参数派生
       （避免"缓存了某个区间的切片"导致不同区间互相污染）。
    """
    path = os.path.join(BY_CODE_DIR, f"{code}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
    except (OSError, ValueError) as e:                   # noqa: PERF203
        _logger.warning("[bars] %s 读取失败: %s", code, e)
        return None
    if df.empty:
        return None
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values("trade_date").reset_index(drop=True)


# ── 日增量覆盖层（2026-10-26）────────────────────────────────────────────
# `by_code` 是 2026-10-18 一次性建的、**没有维护机制**（实测 10-26 还停在 10-16）。
# 用 `panel_daily.append_trade_day` 每天写一个 `daily_overlay/{date}.parquet`
# （5,570 行 / ≈300KB），读时在这里**叠加**到 by_code 之上：
#   · 写侧成本 O(1)（不看 5,830 个文件、不重写 1.4GB）；
#   · 读侧只多一次小 parquet 读，且整层按 `(文件名,size,mtime)` 缓存，
#     进程内只解析一次；
#   · 不用改任何消费方（`get_bars` 是唯一入口）。
OVERLAY_DIR = os.path.join(panel_store.PANEL_DIR, "daily_overlay")

_overlay_cache: dict = {"stamp": None, "map": {}}


def _overlay_map() -> dict:
    """`{code: DataFrame}`（覆盖层合并结果），按文件戳缓存。"""
    files = sorted(glob.glob(os.path.join(OVERLAY_DIR, "*.parquet")))
    if not files:
        _overlay_cache.update({"stamp": (), "map": {}})
        return {}
    try:
        stamp = tuple((os.path.basename(f), os.path.getsize(f), os.path.getmtime(f))
                      for f in files)
    except OSError as e:
        _logger.debug("[bars] 覆盖层文件戳不可读（%s）⇒ 跳过覆盖层", e)
        return _overlay_cache.get("map") or {}
    if _overlay_cache.get("stamp") == stamp:
        return _overlay_cache["map"]

    frames = []
    for f in files:
        try:
            d = pd.read_parquet(f)
        except (OSError, ValueError) as e:                # noqa: PERF203
            _logger.warning("[bars] 覆盖层 %s 读取失败: %s", os.path.basename(f), e)
            continue
        if d is None or d.empty or "stock_code" not in d.columns:
            continue
        d["stock_code"] = d["stock_code"].astype(str)
        d["trade_date"] = pd.to_datetime(d["trade_date"])
        frames.append(d)
    out: dict = {}
    if frames:
        allf = pd.concat(frames, ignore_index=True)
        allf = allf.sort_values("trade_date").drop_duplicates(
            subset=["stock_code", "trade_date"], keep="last")
        for code, g in allf.groupby("stock_code", sort=False):
            out[str(code)] = g.drop(columns=["stock_code"]).reset_index(drop=True)
    _overlay_cache.update({"stamp": stamp, "map": out})
    return out


def _load_raw(code: str) -> pd.DataFrame | None:
    """`by_code` + 日增量覆盖层（同一交易日以覆盖层为准）。"""
    base = _load_base(code)
    ov = _overlay_map().get(code)
    if ov is None or ov.empty:
        return base
    if base is None or base.empty:
        return ov.sort_values("trade_date").reset_index(drop=True)
    merged = pd.concat([base, ov], ignore_index=True)
    merged = merged.sort_values("trade_date").drop_duplicates(
        subset=["trade_date"], keep="last").reset_index(drop=True)
    return merged


# ================================================================
# 1.5 新鲜度（D5）：数据末日至参考日的**交易日**间隔
# ================================================================

_TRADE_CAL_PATH = os.path.join(_PROJECT_ROOT, "data", "trade_cal_cache.json")
_trade_days_cache: Optional[list[str]] = None
_panel_max_cache: Optional[str] = None


def _iso(d) -> Optional[str]:
    """`20261016` / `2026-10-16` / datetime / date → `2026-10-16`。"""
    if d is None or d == "":
        return None
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    s = str(d).replace("/", "-").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s[:10]


def trade_days() -> list[str]:
    """本地交易日历（ISO 升序）。来源 `data/trade_cal_cache.json`。

    ⚠️ **只读本地文件**：日历过期也不联网——离线自检若因为查日历就发请求，
    这条能力等于不存在。无缓存 ⇒ 返回 `[]`（调用方据此把滞后判为"未知"而不是 0）。
    """
    global _trade_days_cache
    if _trade_days_cache is not None:
        return _trade_days_cache
    days: list[str] = []
    try:
        with open(_TRADE_CAL_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        for d in raw.get("days") or []:
            s = _iso(d)
            if s and len(s) == 10:
                days.append(s)
        days.sort()
    except (OSError, ValueError, KeyError) as e:
        _logger.debug("[bars] 交易日历不可用（%s）⇒ 滞后判为未知", e)
    _trade_days_cache = days
    return days


def panel_date_max() -> Optional[str]:
    """面板**市场级**最新交易日（`overview.json` 优先、`by_code_manifest` 次之）。

    为什么要市场级而不是"该股最后一根"：个股停牌时它自己的最后交易日会停在
    几个月前，用"今天的日历"当基准会把**每一个周末**都算成滞后 1~2 天。
    真正有意义的基准是"这份面板覆盖到哪一天"。
    """
    global _panel_max_cache
    if _panel_max_cache is not None:
        return _panel_max_cache
    best = None
    for path, key in ((os.path.join(panel_store.PANEL_DIR, "overview.json"), "date_max"),
                      (MANIFEST_PATH, "date_max")):
        try:
            with open(path, encoding="utf-8") as f:
                v = _iso(json.load(f).get(key))
            if v and (best is None or v > best):
                best = v
        except (OSError, ValueError, KeyError) as e:
            _logger.debug("[bars] %s 不可读（%s）", os.path.basename(path), e)
    # 覆盖层可能比 by_code 更新（日增量只写覆盖层，不改 5,830 个文件）
    ov = _overlay_map()
    if ov:
        try:
            ov_max = max(_iso(df["trade_date"].iloc[-1]) for df in ov.values() if len(df))
        except (ValueError, TypeError) as e:
            _logger.debug("[bars] 覆盖层最新日不可算（%s）", e)
            ov_max = None
        if ov_max and (best is None or ov_max > best):
            best = ov_max
    if best:
        _panel_max_cache = best
        return best
    return None


def trading_day_lag(last_date, ref_date=None) -> Optional[int]:
    """`last_date` 落后 `ref_date` 多少个**交易日**（0 = 同日；未知 ⇒ None）。

    判据只在本地日历上做（`bisect`），不联网、不读 parquet。

    ⚠️ R6（2026-10-18 审查）：`ref` **超出日历覆盖范围 ⇒ 返回 None（未知）**。
       否则面板更新了、日历没更新时，`bisect` 找不到 ref ⇒ 差值恒等于
       "日历末到 last 的距离"，会把"滞后 30 天"算成 **0**（低估成最新）。
       未知必须显式，不能默认"没问题"。
    """
    days = trade_days()
    if not days:
        return None
    last = _iso(last_date)
    ref = _iso(ref_date) or panel_date_max()
    if not last or not ref:
        return None
    if ref > days[-1]:
        _logger.debug("[bars] 参考日 %s 超出交易日历（至 %s）⇒ 滞后判为未知", ref, days[-1])
        return None
    if ref <= last:
        return 0
    return bisect.bisect_right(days, ref) - bisect.bisect_right(days, last)


def _ref_date(as_of=None, end=None) -> Optional[str]:
    """新鲜度的参考日：**调用方显式要求的末日**优先，且不超过面板覆盖。

    只取 `panel_date_max()` 是错的：`get_bars(end="2026-09-01")` 是想要 9 月初的
    切片，用面板末日当基准会把"按你要的区间取到的数据"判成过期（R6）。
    """
    cand = [x for x in (_iso(as_of), _iso(end)) if x]
    ref = min(cand) if cand else None
    pmax = panel_date_max()
    if ref and pmax:
        return min(ref, pmax)
    return ref or pmax


def staleness(code: str, ref_date=None) -> dict:
    """该股**本地日线**的新鲜度（**不受 `start/end` 窗口影响**）。

    为什么单独一个函数（R1，2026-10-18 审查）：screener 取数时按
    `start=today-90d` 切片，长期停牌/退市股在窗口内 **0 行** ⇒ `get_bars` 返回
    None ⇒ 根本进不到"过期"分支，直接落联网兜底。必须先按**全历史末根**判新鲜度。

    Returns:
        {"code", "last_date", "ref_date", "lag_days", "data_lagging", "reason"}
        `data_lagging` 三态：True 过期 / False 新鲜 / **None 未知**（无数据或日历缺失）。
    """
    c = _clean(code)
    ref = _ref_date(ref_date)
    raw = _load_raw(c)
    if raw is None or raw.empty:
        return {"code": c, "last_date": None, "ref_date": ref, "lag_days": None,
                "data_lagging": None, "reason": "本地无该股日线"}
    last = _iso(raw["trade_date"].max())
    lag = trading_day_lag(last, ref)
    if lag is None:
        return {"code": c, "last_date": last, "ref_date": ref, "lag_days": None,
                "data_lagging": None, "reason": f"滞后未知（参考日 {ref}）"}
    lagging = lag > DEFAULT_MAX_LAG_TRADING_DAYS
    return {"code": c, "last_date": last, "ref_date": ref, "lag_days": lag,
            "data_lagging": lagging,
            "reason": (f"最后交易日 {last} 落后参考日 {ref} 共 {lag} 个交易日"
                       if lagging else "")}


def clear_cache() -> None:
    """清进程内缓存（测试/回填后刷新用）。"""
    global _trade_days_cache, _panel_max_cache
    _load_base.cache_clear()
    _load_turn_share.cache_clear()
    _overlay_cache.update({"stamp": None, "map": {}})
    _trade_days_cache = None
    _panel_max_cache = None


# ================================================================
# 2.5 换手率 / 流通股本回填层（2026-10-29）
# ================================================================
# 面板两代分片里 `turn` 只覆盖 1,997 只（**科创板 0、北交所 0**），
# `outstanding_share` 干脆没有（`get_bars` 那列一直是 NaN）。
# `scripts/backfill_turn_share.py` 从新浪日线补齐：
#     data/panel/turn_share/{code}.parquet  (date / turnover_pct / float_share)
# 读侧在这里叠加，消费方不感知（`get_bars` 是唯一入口）。
# ⚠️ 单位：落盘已是**百分数**（脚本里 ×100 换算过，与 BaoStock `turn` 同口径）。
TURN_SHARE_DIR = os.path.join(panel_store.PANEL_DIR, "turn_share")


@lru_cache(maxsize=256)
def _load_turn_share(code: str) -> pd.DataFrame | None:
    """读 `turn_share/{code}.parquet`（列 date(str) / turnover_pct / float_share）。

    文件不存在 ⇒ None（该股没回填过，或本来就退市）。
    缓存 256 只：与 `_load_base` 同量级，避免每只股票两次磁盘读。
    """
    path = os.path.join(TURN_SHARE_DIR, f"{code}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
    except (OSError, ValueError) as e:
        _logger.warning("[bars] %s 换手回填层读取失败: %s", code, e)
        return None
    if df.empty or "date" not in df.columns:
        return None
    return df.drop_duplicates("date", keep="last")


# ================================================================
# 3. 复权（用 preclose/close 比值链重建；面板只存未复权）
# ================================================================

def _qfq(df: pd.DataFrame, anchor: str = "last") -> pd.DataFrame:
    """未复权 → 前复权/后复权，用 `preclose/close` 比值链重建。

    Args:
        anchor: `"last"` = 前复权（qfq，锚定**最后一根**收盘，与联网 qfq 同基准）；
                `"first"` = 后复权（hfq，锚定**第一根**收盘 —— 历史价位不随新数据变化）。

    推导（**这是唯一正确的写法，别用 pct_chg 累乘**）：
      要求 qfq 序列满足 `qfq[t+1]/qfq[t] = 1 + pct_chg[t+1] = close[t+1]/preclose[t+1]`
      （`preclose` 是**除权调整后**的前收 ⇒ 除权日它也跳空，比值恰好抵消价格跳空）。
      设 `qfq[t] = f[t]·close[t]`，代入得
        `f[t+1]·close[t+1] / (f[t]·close[t]) = close[t+1]/preclose[t+1]`
        ⇒ `f[t] = f[t+1] · preclose[t+1] / close[t]`
      即 `ratio[k] = preclose[k]/close[k-1]`（非除权日 = 1，除权日 ≠ 1），
      `f[t] = ∏_{k>t} ratio[k]`，锚定 `f[T] = 1` ⇒ `qfq[T] = close[T]`。

    后复权只需要换锚点：`hfq[t] = close[t] · f[t] / f[0]` ⇒ `hfq[0] = close[0]`。
      （D3，2026-10-18 审查：原来 `adjust="hfq"` 也走锚定末日的公式、attrs 却标 hfq
       —— 元数据说谎。要么实现，要么抛错，不许静默降级。）

    ⚠️ 踩过的坑（2026-10-18）：第一版用 `pct_chg` 累乘当缩放因子，数学上给的是
       **收益序列**而不是**价格缩放** ⇒ 除权日仍然 -50%（`test_qfq_returns_reconstruct_daily_returns`
       直接抓到）。
    ⚠️ `preclose` 缺失 ⇒ 用 `close/(1+pct_chg/100)` 反推；仍不可用 ⇒ 该日不调整（ratio=1），
       绝不静默造假。
    """
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype="float64")
    n = len(close)
    if n == 0:
        return df
    pre = (pd.to_numeric(df["preclose"], errors="coerce").to_numpy(dtype="float64")
           if "preclose" in df.columns else np.full(n, np.nan))
    ret = (pd.to_numeric(df["pct_chg"], errors="coerce").to_numpy(dtype="float64") / 100.0
           if "pct_chg" in df.columns else np.full(n, np.nan))

    # preclose 缺失 ⇒ 由 pct_chg 反推（两条数据路径互备）
    need = np.isnan(pre)
    if need.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            derived = close / (1.0 + ret)
        pre = np.where(need & np.isfinite(derived) & (derived > 0), derived, pre)

    ratio = np.ones(n)
    if n > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = pre[1:] / close[:-1]
        ratio[1:] = np.where(np.isfinite(r) & (r > 0), r, 1.0)

    rev = np.cumprod(ratio[::-1])[::-1]                  # rev[t] = ∏_{k>=t} ratio[k]
    f = rev / ratio                                      # f[t]    = ∏_{k>t}  ratio[k]
    if anchor == "first":
        # 后复权：把锚点从末日移到首日（hfq[0] == close[0]）
        if not np.isfinite(f[0]) or f[0] == 0:
            _logger.warning("[bars] 首日复权因子异常，返回未复权（不造假）")
            return df
        f = f / f[0]
    if not np.isfinite(f).all() or np.any(f <= 0):
        _logger.warning("[bars] 复权因子异常，返回未复权（不造假）")
        return df

    out = df.copy()
    for col in ("open", "high", "low", "close", "preclose"):
        if col in out.columns:
            out[col] = (pd.to_numeric(out[col], errors="coerce").to_numpy(dtype="float64")
                        * f)
    return out


# ================================================================
# 4. 对外唯一入口
# ================================================================

def get_bars(code: str, start: str | None = None, end: str | None = None,
             *, as_of: str | None = None, adjust: str = "qfq",
             min_rows: int = 0,
             max_lag: int | None = None) -> pd.DataFrame | None:
    """取单只股票的本地日线（**不联网**）。

    Args:
        code: `600028` / `sh.600028` / `600028.SH` 均可。
        start / end: `YYYY-MM-DD` 或 `YYYYMMDD`（含端点）。
        as_of: 截断到此日期（含）—— **前视防护**；回测必须传。
        adjust: `qfq`（默认，前复权）/ `hfq` / `none`/`""`（原样未复权）。
        min_rows: 少于该行数视为不可用 ⇒ 返回 None（"数据太短算不出指标"）。
        max_lag: 末根落后参考交易日的**交易日数**上限；超出 ⇒ 返回 None。
            ⚠️ 默认 None = 不拦截（只把 `lag_days`/`data_lagging` 写进 attrs
            交给消费方决策）——因为"多久算陈旧"是业务判断，不该由取数层替
            调用方决定。

    Returns:
        DataFrame（列 = `date open high low close volume amount
        outstanding_share turnover_pct`，`date` 为 datetime，按时间升序），
        `df.attrs = {"source": "panel", "code", "adjust", "as_of",
                     "last_date", "ref_date", "lag_days",
                     "data_lagging", "staleness_reason"}`；
        本地无数据/不足/超期 ⇒ None（**不抛异常、不联网**）。
    """
    c = _clean(code)
    raw = _load_raw(c)
    if raw is None or raw.empty:
        return None

    df = raw
    if as_of:
        cut = pd.to_datetime(str(as_of).replace("/", "-"))
        df = df[df["trade_date"] <= cut]
    if start:
        lo = pd.to_datetime(str(start).replace("/", "-"))
        df = df[df["trade_date"] >= lo]
    if end:
        hi = pd.to_datetime(str(end).replace("/", "-"))
        df = df[df["trade_date"] <= hi]
    if df.empty:
        return None

    df = df.reset_index(drop=True)
    adj = (adjust or "none").lower()
    if adj in ("qfq", "hfq"):
        # 面板只存未复权 ⇒ 两种口径都用比值链重建，只差**锚点**：
        # qfq 锚最后一根（与联网 qfq 同基准）；hfq 锚第一根（历史价位不随新数据变）。
        df = _qfq(df, anchor="last" if adj == "qfq" else "first")
    elif adj not in ("none", ""):
        # D3：未知复权口径**不许静默当"不复权"**（否则调用方以为拿到了复权价）
        raise ValueError(f"不支持的 adjust={adjust!r}（只接受 qfq / hfq / none）")

    out = pd.DataFrame({
        "date": df["trade_date"],
        "open": df.get("open"), "high": df.get("high"),
        "low": df.get("low"), "close": df["close"],
        "volume": df.get("volume"), "amount": df.get("amount"),
        "outstanding_share": np.nan,                     # 面板不存流通股本
        "turnover_pct": df.get("turn"),                  # 已是百分数（BaoStock 口径）
    })
    out = out[[c for c in _OUT_COLS if c in out.columns]]

    # ── 换手率/流通股本：用新浪回填层补齐（2026-10-29）──
    # ⚠️ 只补**空位**（`fillna`），不覆盖面板已有的 BaoStock 值 —— 两个源口径一致
    #    （都是流通股本换手、已统一成百分数），但"谁先有就用谁"能让历史值稳定，
    #    不会因为某天重跑回填就把整段历史换一套数。
    ts = _load_turn_share(c)
    if ts is not None and len(out):
        _d = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        _map_t = dict(zip(ts["date"].astype(str).str[:10], ts["turnover_pct"]))
        _map_s = dict(zip(ts["date"].astype(str).str[:10], ts["float_share"]))
        if "turnover_pct" in out.columns:
            out["turnover_pct"] = out["turnover_pct"].fillna(_d.map(_map_t))
        if "outstanding_share" in out.columns:
            out["outstanding_share"] = out["outstanding_share"].fillna(_d.map(_map_s))
    if len(out) < int(min_rows or 0):
        return None

    # ── 新鲜度（D5）：末根 vs **参考日**（`as_of`/`end` 优先，且不超过面板覆盖）──
    last = _iso(out["date"].iloc[-1])
    ref = _ref_date(as_of, end)
    lag = trading_day_lag(last, ref)
    threshold = DEFAULT_MAX_LAG_TRADING_DAYS if max_lag is None else int(max_lag)
    # 三态（R6）：True 过期 / False 新鲜 / **None 未知**（日历缺失、参考日超范围）。
    # 未知不能落到 False —— 那等于"不知道却报平安"。
    lagging = None if lag is None else lag > threshold
    reason = (f"最后交易日 {last} 落后参考日 {ref} 共 {lag} 个交易日"
              f"（阈值 {threshold}）" if lagging else "")
    if max_lag is not None and lag is not None and lag > int(max_lag):
        _logger.warning("[bars] %s 本地数据过期（%s，上限 %s 日）⇒ 视为不可用",
                        c, reason, max_lag)
        return None
    out.attrs.update({"source": "panel", "code": c, "adjust": adj,
                      "as_of": as_of, "rows": len(out),
                      "last_date": last, "ref_date": ref, "lag_days": lag,
                      "lag_threshold": threshold,
                      "data_lagging": lagging, "staleness_reason": reason})
    return out.reset_index(drop=True)


def get_bars_many(codes: Iterable[str], start: str | None = None,
                  end: str | None = None, *, as_of: str | None = None,
                  adjust: str = "qfq", min_rows: int = 0,
                  max_lag: int | None = None) -> dict[str, pd.DataFrame]:
    """批量版（逐只走缓存；缺失的**不出现在结果里**，调用方须自行报告缺口）。"""
    out: dict[str, pd.DataFrame] = {}
    for c in codes:
        df = get_bars(c, start, end, as_of=as_of, adjust=adjust,
                      min_rows=min_rows, max_lag=max_lag)
        if df is not None:
            out[_clean(c)] = df
    return out


def coverage() -> dict:
    """本地覆盖概况（给页面/审计用）：代码数、manifest、磁盘占用。"""
    codes = available_codes()
    size = 0
    if os.path.isdir(BY_CODE_DIR):
        for f in os.listdir(BY_CODE_DIR):
            if f.endswith(".parquet"):
                try:
                    size += os.path.getsize(os.path.join(BY_CODE_DIR, f))
                except OSError as e:
                    _logger.debug("[bars] %s 大小读取失败（不计入占用）: %s", f, e)
    return {"codes": len(codes), "bytes": size,
            "mb": round(size / 1024 / 1024, 1),
            "manifest": read_manifest()}
