# -*- coding: utf-8 -*-
"""指数日线本地层（B2②，2026-10-19）。

为什么需要：`core/market_detector` 的 6 个指标（沪深300 MA20 / 上证 MACD /
创业板相对强度 / 两市量能 / iVIX 降级 / 大小盘风格）原来各自直接
`ak.stock_zh_index_daily(symbol=...)`，单次 detect() 要打 6~8 次外呼，
离线模式（BIGA_OFFLINE=1）下全部不可评 ⇒ 市场检测永远"数据不足（中性假设）"。

设计：
- 本地目录 `data/panel/index/{symbol}.parquet`（date/open/high/low/close/volume）；
- `backfill_index_daily()`：全量拉取后覆盖写（每次返回全历史、量小，幂等）；
  挂在 `snapshot_flow` 任务尾部（每天 5 次请求）；
- `get_index_bars()`：**不联网**；本地没有 ⇒ None（缺就是缺，与 bars.get_bars 同语义）；
  attrs 带最新日期，消费方自行判断新鲜度。
"""
import logging
import os
from functools import lru_cache

import pandas as pd

try:                                    # 项目硬依赖；测试需要能 patch 到这里
    import requests
except ImportError:                     # pragma: no cover
    requests = None

_logger = logging.getLogger(__name__)

# 数据根必须**运行时**解析：打包后 `__file__` 在只读包 `_internal/` 里，
# 用它推导会指向空目录 ⇒ 读到空数据（2026-10-25 实测）。
# 规则见 `config/paths.data_root()`；源码运行结果与原先一致。
from config.paths import data_root as _paths_data_root  # noqa: E402
PROJECT_ROOT = _paths_data_root()
INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "panel", "index")

# 市场检测用到的全部指数（与 market_detector 的 akshare symbol 一致）
DEFAULT_SYMBOLS = ("sh000001", "sz399001", "sz399006", "sh000300", "sh000852")

_COLS = ["date", "open", "high", "low", "close", "volume"]


def _path(symbol: str) -> str:
    return os.path.join(INDEX_DIR, f"{symbol}.parquet")


def _read_local(symbol: str):
    """读本地 parquet（不存在/损坏 ⇒ None）。`date` 统一成 `YYYY-MM-DD` 字符串。

    回填时要"只补缺、不覆盖已有行"，必须先知道本地已经有哪些日期 ——
    见 `backfill_index_daily` 里关于「覆盖写会截断十年历史」的注释。
    """
    p = _path(symbol)
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_parquet(p)
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[index_bars] 本地 %s 读取失败（按无历史处理）: %s", symbol, e)
        return None
    if df is None or df.empty or "date" not in df.columns:
        return None
    df = df.copy()
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    return df


_INDEX_SOURCES = ("stock_zh_index_daily_em", "stock_zh_index_daily")

# ── 腾讯指数日线直连（2026-10-26 新增）──────────────────────────────
_TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
_TENCENT_MAX_BARS = 1000        # 实测 n=1000 → 1001 根；n=3000 返回结构变化（取数失败）
_TENCENT_VOLUME_SCALE = 100.0   # 腾讯 volume 单位是「手」，新浪是「股」，差 100 倍


def _fetch_tencent_index(symbol: str, bars: int = _TENCENT_MAX_BARS,
                         timeout: float = 8.0):
    """腾讯指数日线直连，返回与 `stock_zh_index_daily` **同构**的 DataFrame。

    为什么加这一源（2026-10-26 实测，当天是周一、正常交易日）
    ------------------------------------------------------
    15:49 的回填日志报"5 成功 / 0 失败"，但 5 个 parquet 的最新日期**全是 10-23**。
    逐个源核对出来的事实：

      - `stock_zh_index_daily`（新浪）：16:12 再拉一次，最新**仍是 10-23**；
      - `stock_zh_index_daily_em`（东财）：`push2.eastmoney.com` / `80.push2` /
        `82.push2` **全集群 `RemoteDisconnected`**（而 `push2his` 返回 200 ——
        是单集群故障，不是网络全断，也不是代理问题：绕开代理同样失败）；
      - 腾讯 `web.ifzq.gtimg.cn`：**已有 10-26**，1.2s 返回。

    ⇒「指数永远落后一天」的根因是**源本身滞后**，不是回填逻辑坏了。
      连带后果不止日志难看：`market_detector` 的 6 个指数指标全部"不可评"，
      15:50~15:52 反复"回落联网"重试，复盘被拖慢两分多钟，且判市口径失真。

    ⚠️⚠️ 单位必须归一 —— 这是本模块最容易踩的坑
    ------------------------------------------
    腾讯 volume 单位是**手**，新浪是**股**。2026-10-26 对 sh000001 / sz399006 /
    sh000300 三个样本核同一交易日，比值**精确等于 100.0**。
    不换算直接写盘 ⇒「两市量能」指标骤降 100 倍 ⇒ 判成"地量" ⇒ 判熊市
    ⇒ 仓位上限被压死（典型的"列名相同 ≠ 单位相同"）。故此处一律 `* 100`。
    """
    if requests is None:                                    # pragma: no cover
        return None
    try:
        r = requests.get(
            _TENCENT_KLINE_URL,
            params={"param": f"{symbol},day,,,{int(bars)},"},   # 空复权位 = 原始日线
            headers={"User-Agent": "Mozilla/5.0",
                     "Referer": "https://gu.qq.com/"},
            timeout=timeout,
        )
        r.raise_for_status()
        payload = r.json()
    except Exception as e:                                  # noqa: BLE001
        _logger.debug("[指数日线] 腾讯源失败 %s: %s: %s",
                      symbol, type(e).__name__, e)
        return None

    node = (payload.get("data") or {}).get(symbol)
    if not isinstance(node, dict):                          # n 过大时 data 会变成 list
        _logger.debug("[指数日线] 腾讯源 %s 结构异常（data 节点非 dict）", symbol)
        return None
    rows = node.get("day") or node.get("qfqday") or []
    if not rows:
        return None

    # 腾讯行序：[date, open, close, high, low, volume] —— ⚠️ 第 3 位是**收盘**，
    # 不是 high。按位置错读会让 K 线全乱（用 10-26 上证实测校验过：
    # open=3671.00 close=3697.52 high=3699.01 low=3664.82，high>close>low 自洽）。
    parsed = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        parsed.append({
            "date": str(row[0]),
            "open": pd.to_numeric(row[1], errors="coerce"),
            "close": pd.to_numeric(row[2], errors="coerce"),
            "high": pd.to_numeric(row[3], errors="coerce"),
            "low": pd.to_numeric(row[4], errors="coerce"),
            "volume": pd.to_numeric(row[5], errors="coerce") * _TENCENT_VOLUME_SCALE,
        })
    if not parsed:
        return None
    df = pd.DataFrame(parsed)[_COLS]
    df = df.dropna(subset=["date", "close"]).reset_index(drop=True)
    return df if len(df) else None


def _fetch_best_index(symbol: str):
    """取**最新日期最大**的那份指数日线，返回 `(df, source_name)`。

    ⚠️⚠️ 为什么不能只用新浪源（2026-10-23 实测）：
      `ak.stock_zh_index_daily`（新浪）**当日收盘后仍只到 T-1** ——
      16:12 拉到的最新一根还是 10-22，而当天 15:40 的回填任务照样报
      "5 成功 / 0 失败"（文件确实重写了、只是内容是旧的）。
      后果：`market_detector` 的 6 个指数指标**永远落后一天**，
      今天创业板 +4.16% 它却判"偏弱震荡、仓位上限 20%"，选股直接被压仓。
    ⇒ 多源试取，**按数据日期择优**（而不是按"谁先成功"），
      最坏情况退回新浪，并把"数据仍是旧的"如实写进返回值。
    """
    import akshare as ak          # ⚠️ 必须在本函数内导入：原来只在 backfill 里 import，
                                  #    挪出来调用就 NameError（作用域不通）。

    best, best_src, errs = None, None, []

    # 腾讯源先试：2026-10-26 实测它能取到当日，而下面两个 akshare 源都停在 T-1。
    # 先试不等于优先采用 —— 仍然按**数据日期**择优（下面的 `last > best[0]`）。
    tdf = _fetch_tencent_index(symbol)
    if tdf is not None and len(tdf):
        try:
            best, best_src = (pd.to_datetime(tdf["date"]).max(), tdf), "tencent"
        except Exception as e:                              # noqa: BLE001
            _logger.debug("[指数日线] 腾讯源 %s 日期不可解析: %s", symbol, e)
            errs.append("tencent: 日期不可解析")
    else:
        errs.append("tencent: 空数据")

    for name in _INDEX_SOURCES:
        fn = getattr(ak, name, None)
        if fn is None:
            errs.append(f"{name}: 不存在")
            continue
        try:
            df = fn(symbol=symbol)
        except Exception as e:                              # noqa: BLE001
            errs.append(f"{name}: {type(e).__name__}")
            _logger.debug("[指数日线] %s 取数失败: %s: %s", name, type(e).__name__, e)
            continue
        if df is None or df.empty or "date" not in df.columns:
            errs.append(f"{name}: 空数据")
            continue
        try:
            last = pd.to_datetime(df["date"]).max()
        except Exception:                                   # noqa: BLE001
            errs.append(f"{name}: 日期不可解析")
            _logger.debug("[指数日线] %s 的日期列不可解析", name)
            continue
        if best is None or last > best[0]:
            best, best_src = (last, df), name
    if best is None:
        raise RuntimeError("全部指数源失败：" + "；".join(errs))
    return best[1], best_src


def backfill_index_daily(symbols: tuple = DEFAULT_SYMBOLS) -> dict:
    """拉取指数日线全历史并覆盖写本地（幂等；每次 5 个请求）。

    数据源：`ak.stock_zh_index_daily`（与 market_detector 原实现同源，保证口径
    一致——volume 单位是**成交量（股）**，见 market_detector 2026-09-28 的口径修正）。
    ⚠️ 本函数**必须联网**，只应在调度任务（snapshot_flow 尾部）或手动回填时调用；
    离线模式（BIGA_OFFLINE=1）下会抛 RuntimeError（复用 offline_gate 的语义）。
    """
    try:
        from data import offline_gate
        if offline_gate.is_offline():
            raise RuntimeError(
                "BIGA_OFFLINE=1 下禁止回填指数日线（会联网取数）。")
    except ImportError as e:                                # noqa: BLE001
        _logger.debug("[index_bars] offline_gate 不可用（按在线处理）: %s", e)

    os.makedirs(INDEX_DIR, exist_ok=True)
    import akshare as ak
    done, failed = {}, {}
    for sym in symbols:
        try:
            df, src = _fetch_best_index(sym)
            if df is None or df.empty or "date" not in df.columns:
                failed[sym] = "空数据"
                continue
            out = df.rename(columns=str.lower)
            keep = [c for c in _COLS if c in out.columns]
            out = out[keep].copy()
            out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
            out = out.sort_values("date").drop_duplicates("date", keep="last")

            # ⚠️⚠️ 不能无脑覆盖写：腾讯源单次最多 1000 根，而本地新浪历史有 8000+
            #   根（sh000001 实测 8748 根）。若把择优源的结果直接覆盖过去，十年历史
            #   会被截成四年；更糟的是**截断是永久的**——下次 `old` 也只剩四年。
            #   ⇒ 一律「本地已有行原样保留 + 只追加新日期」。
            #   重叠日期以本地为准：新浪价与腾讯价有 ~0.005 的口径微差（复权处理不同），
            #   混用会在历史曲线上留下看不见的断点。
            old = _read_local(sym)
            if old is not None and len(old):
                have = set(old["date"].astype(str))
                fresh = out[~out["date"].isin(have)]
                if len(fresh):
                    out = pd.concat([old[_COLS], fresh[_COLS]], ignore_index=True)
                    out = out.sort_values("date").drop_duplicates("date", keep="last")
                else:
                    out = old[_COLS].sort_values("date").reset_index(drop=True)
                added = len(fresh)
            else:
                added = len(out)
                if src == "tencent" and len(out) <= _TENCENT_MAX_BARS:
                    _logger.warning(
                        "[index_bars] %s 首次回填仅 %d 根（腾讯源单次上限 %d）"
                        "—— 本地无历史可合并，历史长度受限",
                        sym, len(out), _TENCENT_MAX_BARS)
            tmp = _path(sym) + ".tmp"
            out.to_parquet(tmp, index=False)
            os.replace(tmp, _path(sym))
            done[sym] = {"rows": len(out), "last_date": out["date"].iloc[-1],
                         "source": src, "added": added}
        except Exception as e:                              # noqa: BLE001
            _logger.warning("[index_bars] %s 回填失败: %s", sym, e)
            failed[sym] = str(e)
    get_index_bars.cache_clear()
    # ⚠️⚠️ "回填成功" ≠ "数据是今天的"：新浪源收盘后仍只到 T-1（2026-10-23 实测）。
    #    原来的日志只报"5 成功 / 0 失败"⇒ 数据旧了一天也没人知道。
    #    这里显式算出 stale 列表，让调度日志能一眼看出"今天这根没有"。
    try:
        expected = expected_close_date()
        stale = {sym: v["last_date"] for sym, v in done.items()
                 if isinstance(v, dict) and str(v.get("last_date")) < str(expected)}
        if stale:
            _logger.warning("[index_bars] 数据仍陈旧（应收盘 %s，实际停在）：%s",
                            expected, stale)
    except Exception as e:                                  # noqa: BLE001
        _logger.debug("[index_bars] 陈旧判断失败: %s", e)
        stale = {}
    return {"done": done, "failed": failed, "stale": stale,
            "dir": INDEX_DIR, "at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}


@lru_cache(maxsize=16)
def get_index_bars(symbol: str, start=None, end=None, min_rows: int = 0):
    """读本地指数日线（**不联网**）。

    Returns:
        DataFrame（date/open/high/low/close/volume，date 为 datetime 升序），
        `attrs = {"source": "index_panel", "symbol", "last_date", "rows"}`；
        本地没有 / 不足 `min_rows` ⇒ None（缺就是缺）。
    """
    path = _path(symbol)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[index_bars] %s 读取失败: %s", symbol, e)
        return None
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    if start is not None:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["date"] <= pd.to_datetime(end)]
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < int(min_rows or 0):
        return None
    df.attrs.update({"source": "index_panel", "symbol": symbol,
                     "rows": len(df),
                     "last_date": df["date"].iloc[-1].strftime("%Y-%m-%d")})
    return df


# 日线收盘数据通常在这个时刻前后落地（A 股 15:00 收盘）
_CLOSE_READY_HHMM = 15 * 60 + 10


def expected_close_date(now=None) -> str | None:
    """现在这个时刻，**日线收盘数据**应该已经给到哪一天（取不到 ⇒ None）。

    ⚠️⚠️ 不能拿交易日历的最后一个元素当基准（2026-10-19 审查 P1-2 的根因）：
    `trade_days()` 只覆盖到"上次回填写入的日期"，实测 2026-10-19 日历末是
    **10-16**，而数据已到 10-19 ⇒ `last >= days[-1]` 恒真，闸门等于没装；
    反过来，若数据停在 10-16 而日历末也是 10-16，同样恒真——**陈旧永远判不出来**。

    规则：
      - 非交易日 → 最近一个交易日（周六看周五的收盘，正确）；
      - 交易日但未到 15:10 → 上一个交易日（当天收盘还没出）；
      - 交易日且已过 15:10 → 当天。

    ⚠️ 只用**本地日历文件**（`data.bars.trade_days`，过期也不联网）——
    走 `data.trading_calendar` 会在缓存过期时发请求，离线自检会因此丢掉
    全部指数指标（"查日历"就把能力废掉了）。本地日历未覆盖今天时按工作日近似
    （与 trading_calendar 的降级口径一致）。
    """
    import datetime as _dt
    now = now or _dt.datetime.now()
    today = now.strftime("%Y-%m-%d")
    after_close = (now.hour * 60 + now.minute) >= _CLOSE_READY_HHMM
    try:
        from data.bars import trade_days
        days = trade_days() or []
    except Exception as e:                                  # noqa: BLE001
        _logger.debug("[index_bars] 本地交易日历不可用: %s", e)
        return None
    if not days:
        return None
    known = set(days)
    if today in known:
        is_td = True
    elif today > days[-1]:
        # 日历没覆盖今天 ⇒ 工作日近似（节假日会误判，但配合 max_lag=1 不会误杀）
        is_td = pd.Timestamp(today).weekday() < 5
    else:
        is_td = False
    if is_td and after_close:
        return today
    prev = [d for d in days if d < today]
    return prev[-1] if prev else None


def index_freshness(symbol: str, max_lag_days: int = 1) -> dict:
    """本地指数数据的**新鲜度详情**（消费方：`market_detector._index`）。

    Returns:
        {"fresh": bool, "last_date": str|None, "expected": str|None,
         "lag_days": int|None, "reason": str}

    判据：`last_date` 落后 `expected_close_date()` 不超过 `max_lag_days` 个
    **交易日**。默认容差 1（周末/收盘后到回填前的正常延迟），
    即"期望 T、数据停在 T-2"必须判**不新鲜**。
    日历不可用 ⇒ 不新鲜（未知 ≠ 没问题），由调用方决定是否回退联网。
    """
    import bisect
    df = get_index_bars(symbol)
    last = (df.attrs.get("last_date") if df is not None else None)
    if not last:
        return {"fresh": False, "last_date": None, "expected": None,
                "lag_days": None, "reason": "本地无该指数数据"}
    expected = expected_close_date()
    if not expected:
        return {"fresh": False, "last_date": last, "expected": None,
                "lag_days": None, "reason": "交易日历不可用，无法判定新鲜度"}
    if last >= expected:
        return {"fresh": True, "last_date": last, "expected": expected,
                "lag_days": 0, "reason": ""}
    try:
        from data.bars import trade_days
        days = trade_days()
        if not days:
            return {"fresh": False, "last_date": last, "expected": expected,
                    "lag_days": None, "reason": "交易日历为空，无法数交易日"}
        lag = (bisect.bisect_right(days, expected)
               - bisect.bisect_right(days, last))
        # ⚠️ 期望日可能超出日历覆盖（日历没更新）⇒ bisect 会把 lag 低估成 0，
        #    此时按自然日差兜底（宁松不误杀，但绝不假装 0）。
        if lag <= 0 and expected > days[-1]:
            lag = max(1, (pd.Timestamp(expected) - pd.Timestamp(last)).days)
    except Exception as e:                                  # noqa: BLE001
        _logger.debug("[index_bars] %s 新鲜度判定失败: %s", symbol, e)
        return {"fresh": False, "last_date": last, "expected": expected,
                "lag_days": None, "reason": f"新鲜度判定异常: {e}"}
    fresh = lag <= int(max_lag_days)
    return {"fresh": fresh, "last_date": last, "expected": expected,
            "lag_days": int(lag),
            "reason": "" if fresh else
            f"指数数据停在 {last}，应有 {expected}（落后 {lag} 个交易日）"}


def index_fresh_enough(symbol: str, max_lag_days: int = 1) -> bool:
    """本地指数数据是否"新鲜"（判据见 `index_freshness`）。"""
    return bool(index_freshness(symbol, max_lag_days=max_lag_days)["fresh"])
