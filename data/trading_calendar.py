"""交易日历（Phase 1 前置）

用途：
  - 调度器：非交易日不选股、不生成/推送复盘
  - 盯盘预警：只在交易时段与非节假日运行
  - 数据窗口：取"最近交易日"而不是自然日

数据源与降级策略：
  主源 akshare `tool_trade_date_hist_sina`（实测覆盖 1990-12-19 起、含未来约一年）。
  结果本地缓存到 data/trading_calendar.json（TTL 7 天），源不可用时优先用缓存；
  连缓存都没有时降级为"周一至周五"工作日近似，并把 `degraded` 标记为 True——
  宁可明确标注不精确，也不假装知道节假日。

注意：落在这个日历覆盖范围之外的日期（例如更远的未来），也会退回工作日近似，
并通过 `out_of_range` 标记暴露出来。
"""

import json
import logging
import os
import time
from datetime import date, datetime, timedelta

_logger = logging.getLogger(__name__)

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "trading_calendar.json")
CACHE_TTL_SECONDS = 7 * 24 * 3600

# 内存缓存
_state = {
    "dates": None,          # set[str] 已知交易日
    "min": None,
    "max": None,
    "loaded_at": 0.0,
    "degraded": False,      # True = 当前用的是工作日近似而非真实日历
    "source": "unloaded",   # akshare | file_cache | weekday_fallback
}


def _today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


def _as_date_str(d) -> str:
    if d is None:
        return _today_str()
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, date):
        return d.strftime("%Y-%m-%d")
    return str(d)[:10]


def _is_weekday(ds: str) -> bool:
    return datetime.strptime(ds, "%Y-%m-%d").weekday() < 5


def _read_cache_file():
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            payload = json.load(f)
        dates = set(payload.get("dates") or [])
        return dates, float(payload.get("saved_at") or 0.0)
    except (OSError, ValueError, TypeError):
        return None, 0.0


def _write_cache_file(dates) -> bool:
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"saved_at": time.time(), "dates": sorted(dates)}, f)
        return True
    except OSError as e:
        _logger.warning("[交易日历] 缓存写入失败: %s", e)
        return False


def _fetch_from_akshare():
    """返回 set[str]；失败返回 None（由调用方决定降级）"""
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        if df is None or df.empty:
            return None
        col = df.columns[0]
        return {str(v)[:10] for v in df[col].tolist()}
    except Exception as e:
        _logger.warning("[交易日历] akshare 获取失败: %s", e)
        return None


def _install(dates, source: str, degraded: bool):
    _state.update({
        "dates": set(dates),
        "min": min(dates) if dates else None,
        "max": max(dates) if dates else None,
        "loaded_at": time.time(),
        "degraded": degraded,
        "source": source,
    })


def refresh(force: bool = False) -> dict:
    """加载日历（内存 → 文件缓存 → akshare → 工作日近似），返回状态字典"""
    now = time.time()
    if not force and _state["dates"] and now - _state["loaded_at"] < CACHE_TTL_SECONDS:
        return cache_status()

    file_dates, saved_at = _read_cache_file()
    if not force and file_dates and now - saved_at < CACHE_TTL_SECONDS:
        _install(file_dates, "file_cache", degraded=False)
        return cache_status()

    fetched = _fetch_from_akshare()
    if fetched:
        _write_cache_file(fetched)
        _install(fetched, "akshare", degraded=False)
        return cache_status()

    if file_dates:
        # 源不可用但有过期缓存：用过期缓存，仍算精确（只是可能缺未来新增日期）
        _install(file_dates, "file_cache(stale)", degraded=False)
        return cache_status()

    _install(set(), "weekday_fallback", degraded=True)
    _logger.warning("[交易日历] 无任何日历数据，降级为「周一至周五」近似（不含节假日）")
    return cache_status()


def cache_status() -> dict:
    return {
        "source": _state["source"],
        "degraded": _state["degraded"],
        "count": len(_state["dates"] or []),
        "min": _state["min"],
        "max": _state["max"],
    }


def _ensure_loaded():
    if not _state["dates"] and _state["source"] == "unloaded":
        refresh()


def is_trading_day(d=None) -> bool:
    """是否交易日。日历覆盖范围之外退回工作日近似并记日志"""
    refresh()
    ds = _as_date_str(d)
    dates = _state["dates"] or set()

    if dates and _state["min"] <= ds <= _state["max"]:
        return ds in dates

    # 覆盖范围之外（或完全没有日历）：工作日近似
    _logger.warning("[交易日历] %s 超出日历覆盖范围(%s~%s)，按工作日近似判断",
                    ds, _state["min"], _state["max"])
    return _is_weekday(ds)


def trading_days(start=None, end=None) -> list:
    """[start, end] 区间内的交易日列表（含端点）"""
    refresh()
    s = _as_date_str(start or date.today())
    e = _as_date_str(end or date.today())
    dates = _state["dates"] or set()
    if dates and _state["min"] <= s and e <= _state["max"]:
        return sorted(d for d in dates if s <= d <= e)
    # 退回工作日枚举
    out, cur = [], datetime.strptime(s, "%Y-%m-%d").date()
    last = datetime.strptime(e, "%Y-%m-%d").date()
    while cur <= last:
        if cur.weekday() < 5:
            out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def last_trading_day(ref=None, include_today: bool = True) -> str:
    """最近一个交易日（默认含当日）"""
    refresh()
    cur = datetime.strptime(_as_date_str(ref), "%Y-%m-%d").date()
    if not include_today:
        cur -= timedelta(days=1)
    for _ in range(30):
        ds = cur.strftime("%Y-%m-%d")
        if is_trading_day(ds):
            return ds
        cur -= timedelta(days=1)
    return _as_date_str(ref)


def next_trading_day(ref=None) -> str:
    """下一个交易日（不含当日）"""
    refresh()
    cur = datetime.strptime(_as_date_str(ref), "%Y-%m-%d").date() + timedelta(days=1)
    for _ in range(30):
        ds = cur.strftime("%Y-%m-%d")
        if is_trading_day(ds):
            return ds
        cur += timedelta(days=1)
    return _as_date_str(ref)


def clear_memory_cache():
    """清空内存缓存（测试用）"""
    _state.update({"dates": None, "min": None, "max": None, "loaded_at": 0.0,
                   "degraded": False, "source": "unloaded"})
