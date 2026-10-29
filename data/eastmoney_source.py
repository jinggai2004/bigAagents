"""东方财富全市场快照 HTTP 直连（逐页容错 + akshare 兜底）

为什么需要
----------
2026-09-25 数据源需求体检发现「全市场快照」是唯一未稳定满足的需求，
`ak.stock_zh_a_spot_em()` 报 `ProxyError`。实测定位到真正的原因**不是东财挂了**：

    该接口**硬上限 100 条/页**（pz 传 500~6000 都只返回 100 条），
    全市场 5920 只 → 必须分 **60 页**请求；
    而 akshare 的 `fetch_paginated_data` 对这 60 次请求**没有任何容错**——
    任何一页失败就整体抛异常，前面 59 页拿到的数据全部作废。

在代理/网络抖动 1% 的环境里，一次调用的失败概率约 1-(0.99^60) ≈ 45%，
表现就是"时好时坏、看起来极不稳定"。**这是一个可用性问题，不是数据源问题。**

本模块把同一批数据按同样的分页取回来，但：
- **逐页重试**，单页失败不影响其余页；
- **连续多页失败才放弃**（判定上游确实不可用），并返回已取得的部分 + 如实标注覆盖度；
- 列名与 akshare **完全一致**（按 f-code 键名映射，不依赖列顺序），下游无需改动。

覆盖度诚实原则（**非常重要，别把部分数据当全市场**）
--------------------------------------------------
分页是按**代码升序**返回的，所以"只拿到前 N 页"得到的是**特定板块的子集**：
实测只成功前 10 页时，拿到的全是北交所 920xxx，算出的"涨跌家数比"是 **0.13**，
而全市场真实值在 1 附近量级——**偏差极大，会让市场检测得出反向结论**。

因此本模块**不返回残缺数据**：实得覆盖率低于 `MIN_COVERAGE` 时一律返回 None，
让调用方走它自己的保守默认值（例如市场检测的 4/8 中性分），
而不是拿一个有偏样本算出"看起来很确定"的错答案。少数据 ≠ 可以用。
"""

import logging
import random
import time

import pandas as pd
import requests

_logger = logging.getLogger(__name__)

# 与 akshare stock_zh_a_spot_em 同一个端点
CLIST_URL = "https://82.push2.eastmoney.com/api/qt/clist/get"

# 实得覆盖率低于此值视为不可用（宁可没有，也不要拿有偏样本算市场宽度）
MIN_COVERAGE = 0.9

# 快照 TTL：市场检测一次 detect() 要调 3 次、情绪还要再调 3 次，
# 不缓存的话同一份 60 页数据会被重复拉 6 遍（每遍 60 次请求）。
_SNAPSHOT_TTL = 60
_snapshot_cache = {"df": None, "ts": 0.0}

# 沪深京 A 股（与 akshare 的 fs 取值一致）
_MARKETS = "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048"

# 按 **f-code 键名** 取字段并映射为中文列名（与 akshare 输出同名）。
# 关键：用键名而不是位置——akshare 是按位置 rename 的，一旦上游字段顺序变化
# 就会静默错位；按 key 映射对顺序变化免疫。
_FIELD_MAP = {
    "f12": "代码", "f14": "名称", "f2": "最新价", "f3": "涨跌幅", "f4": "涨跌额",
    "f5": "成交量", "f6": "成交额", "f7": "振幅", "f8": "换手率",
    "f9": "市盈率-动态", "f10": "量比", "f11": "5分钟涨跌",
    "f15": "最高", "f16": "最低", "f17": "今开", "f18": "昨收",
    "f20": "总市值", "f21": "流通市值", "f22": "涨速", "f23": "市净率",
    "f24": "60日涨跌幅", "f25": "年初至今涨跌幅",
    "f13": "市场",
}
_MISSING = "-"

# 接口硬上限：实测 pz 传 500/2000/6000 都只返回 100 条
PAGE_SIZE = 100
_MAX_PAGES = 90                 # 全市场约 60 页，留余量
_CONSECUTIVE_FAIL_LIMIT = 5     # 连续这么多页失败 → 判定上游不可用，停止
_CONN_FAIL_STREAK_LIMIT = 3     # 已成功过时，连续这么多页"连接层"失败才熔断
                                # （单页 ProxyError 只是抖动，逐页容错是核心价值）

# ── 连接层熔断（2026-10-26）──────────────────────────────────────────
# 为什么需要：当天 15:31:07 起 `push2.eastmoney.com` / `80.push2` / `82.push2`
# **整个快照集群** `RemoteDisconnected`（而 `push2his` 同时返回 200 —— 单集群故障，
# 既不是网络全断也不是代理问题：绕开代理直连同样失败）。
# 原实现对**同一个不通的主机**仍按 `retries=2` 重试，5 页连续失败 + akshare 兜底
# 一轮白花 64s（15:31:07→15:32:11），而结论在第一次失败时就已经确定了；
# 同一晚盘后流程里跑了两轮 ⇒ 白等 128s。
#
# 判据与 `data/fund_flow._is_connection_level_error` 一致。该模块顶部
# `import akshare` 成本很高，不能反向依赖（本模块存在的意义就是绕开 akshare），
# 故此处独立实现；**两边改动必须同步**，由测试里的一致性断言守护。
_EM_CIRCUIT_COOLDOWN = 600      # 熔断冷却秒数（与 fund_flow 的东财冷却口径一致）
_em_circuit_open_until = 0.0

_CONN_ERROR_MARKERS = (
    "Max retries exceeded", "ProxyError", "RemoteDisconnected",
    "Connection aborted", "Unable to connect", "Connection reset",
    "Name or service not known", "getaddrinfo failed",
)


def _is_connection_level_error(exc) -> bool:
    """是否为"连不上"类失败（域名不可达 / 被代理拒绝 / 连接被重置）。

    区分两类失败（与 fund_flow 同口径）：
      - **连接层**（返回 True）：主机现在不通，重试同一主机没有意义 ⇒ 立即熔断降级；
      - **内容层**（返回 False）：接口通了但返回空/解析失败，可能是抖动 ⇒ 保留重试。
    混为一谈的后果要么是白等（把连接失败当抖动手动重试），
    要么是误熔断（把一次空响应当主机挂掉，整整 10 分钟不用东财）。
    """
    if isinstance(exc, (requests.exceptions.ProxyError,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout)):
        return True
    text = str(exc)
    return any(m in text for m in _CONN_ERROR_MARKERS)


def circuit_remaining() -> float:
    """熔断还剩多少秒（0 = 未熔断）。"""
    return max(0.0, _em_circuit_open_until - time.time())


def trip_circuit(reason: str) -> None:
    """打开熔断：冷却期内 `fetch_market_snapshot` 直接返回 None（由调用方兜底）。"""
    global _em_circuit_open_until
    _em_circuit_open_until = time.time() + _EM_CIRCUIT_COOLDOWN
    _logger.warning("[东财快照] 连接层故障 ⇒ 熔断 %ds，期间直接走兜底源: %s",
                    _EM_CIRCUIT_COOLDOWN, reason)


def reset_circuit() -> None:
    """清掉熔断（测试 / 手动重试用；生产不应定期调用）。"""
    global _em_circuit_open_until
    _em_circuit_open_until = 0.0



def _headers():
    return {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "*/*",
    }


def _fetch_page(page: int, timeout: float, retries: int) -> tuple:
    """取一页，返回 (diff列表, 总数)。失败抛异常。"""
    params = {
        "pn": str(page), "pz": str(PAGE_SIZE), "po": "1", "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2", "invt": "2", "fid": "f12", "fs": _MARKETS,
        "fields": ",".join(_FIELD_MAP.keys()),
        "_": str(int(time.time() * 1000)),
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(CLIST_URL, params=params, headers=_headers(),
                             timeout=timeout)
            text = (r.text or "").strip()
            if not text:
                raise ValueError(f"空响应体 (HTTP {r.status_code})")
            payload = r.json()
            data = payload.get("data") or {}
            diff = data.get("diff")
            if diff is None:
                # data 为 null：通常是超出页数，属正常终止信号而非错误
                return [], 0
            rows = list(diff.values()) if isinstance(diff, dict) else list(diff)
            return rows, int(data.get("total") or 0)
        except Exception as e:                       # noqa: BLE001
            last = e
            # 连接层失败 ⇒ 对同一个不通的主机重试没有意义，立刻上抛（由上层熔断）。
            # 这一条是 2026-10-26 白等 128s 的直接原因：RemoteDisconnected 属连接层，
            # 原来照样 sleep 0.4/0.8s 再打两次同一个死主机。
            if _is_connection_level_error(e):
                raise
            if attempt < retries - 1:
                time.sleep(0.4 * (attempt + 1) + random.uniform(0, 0.2))
    raise last


def fetch_market_snapshot(timeout: float = 12.0, retries: int = 2,
                          max_pages: int = _MAX_PAGES):
    """拉取全市场快照（逐页容错）。

    Returns:
        DataFrame（列名与 akshare 的 stock_zh_a_spot_em 一致），
        或 None（一页都没成功）。
        `df.attrs` 带 coverage / degraded / failed_pages，供调用方判断完整性。
    """
    all_rows: list = []
    total = 0
    failed_pages: list = []
    consecutive = 0
    conn_streak = 0                 # 连续"连接层"失败页数（成功一页即归零）

    # 熔断期内直接降级：调用方会自己走 akshare / 本地库，
    # 没必要再等一轮连接失败（那正是 2026-10-26 每轮白等 45s 的来源）。
    left = circuit_remaining()
    if left > 0:
        _logger.info("[东财快照] 熔断中（剩余 %ds），跳过直连直接走兜底", int(left))
        return None

    for page in range(1, max_pages + 1):
        try:
            rows, tot = _fetch_page(page, timeout, retries)
        except Exception as e:                       # noqa: BLE001
            failed_pages.append(page)
            _logger.warning("[东财快照] 第 %d 页失败(%s)", page, e)
            if _is_connection_level_error(e):
                conn_streak += 1
                # ⚠️ 判据的核心是「**有没有成功过**」，不是"是不是连接层错误"：
                #   已成功过若干页 ⇒ 主机是通的，单页连接失败只是抖动 ⇒ 继续翻页
                #     （逐页容错正是本模块存在的理由，不能因为一页失败就放弃）；
                #   从未成功 ⇒ 主机大概率不通 ⇒ 第一次失败就足以判定，
                #     不必试满 5 页再认命（2026-10-26 就是这种情况，白等 45s）。
                if not all_rows or conn_streak >= _CONN_FAIL_STREAK_LIMIT:
                    trip_circuit(f"第 {page} 页: {type(e).__name__}")
                    break
                continue
            conn_streak = 0
            consecutive += 1
            if consecutive >= _CONSECUTIVE_FAIL_LIMIT:
                _logger.error("[东财快照] 连续 %d 页失败，判定上游不可用，"
                              "返回已取得的 %d 只", consecutive, len(all_rows))
                break
            continue

        consecutive = 0
        conn_streak = 0

        consecutive = 0
        if tot:
            total = tot
        if not rows:
            break                                    # 正常翻到末尾
        all_rows.extend(rows)
        if total and len(all_rows) >= total:
            break

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    # 按 f-code 键名映射（缺的字段用 '-' 占位，与 akshare 行为一致）
    df = df.rename(columns=_FIELD_MAP)
    for col in _FIELD_MAP.values():
        if col not in df.columns:
            df[col] = _MISSING
    df = df[list(_FIELD_MAP.values())]
    for col in ("最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "振幅", "换手率",
                "市盈率-动态", "量比", "最高", "最低", "今开", "昨收",
                "总市值", "流通市值", "市净率", "涨速",
                "5分钟涨跌", "60日涨跌幅", "年初至今涨跌幅"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    coverage = (len(df) / total) if total else 0.0
    df.attrs["coverage"] = round(coverage, 4)
    df.attrs["degraded"] = bool(failed_pages) or (total and len(df) < total)
    df.attrs["failed_pages"] = failed_pages
    df.attrs["total_expected"] = total

    # 覆盖率闸门：分页按代码升序，缺页 ≠ 随机缺样（实测缺页时只剩北交所成分，
    # 涨跌家数比 0.13 vs 真实 1 量级）→ 宁可返回 None 让调用方走保守默认值。
    if coverage < MIN_COVERAGE:
        _logger.warning("[东财快照] 覆盖不足（%d/%s = %.0f%% < %.0f%%），"
                        "判定不可用并放弃——分页缺口是有偏样本，不能当全市场用",
                        len(df), total or "?", coverage * 100, MIN_COVERAGE * 100)
        return None

    if df.attrs["degraded"]:
        _logger.warning("[东财快照] 降级返回：%d/%s 只（失败页 %d 个）",
                        len(df), total or "?", len(failed_pages))
    else:
        _logger.info("[东财快照] ✅ 全市场 %d 只", len(df))
    return df


def _local_snapshot_df(want_date: str | None = None):
    """从本地 `market_snapshot` 表构造**东财口径**的中文列 DataFrame（P0-1）。

    为什么需要：`MarketDetector.detect()` 一次要打 3 次东财全市场快照
    （55 页 × 3），断网/限流时 `选股` 直接卡在 Stage 0；而 15:40 已经
    把当天的全市场截面落库了 —— 同一天再联网拉一遍是纯浪费。

    列映射（库 → 东财中文列，只映射 market_detector/情绪模块实际消费的）：
      stock_code→代码, stock_name→名称, price→收盘/最新价,
      change_pct→涨跌幅, pe→市盈率-动态, pb→市净率, amount→成交额
    ⚠️ 口径差异必须知道：库里是**某一时点的截面**（15:40 采集），不是实时。
       `df.attrs["data_date"]` 如实带出，调用方要判断"够不够新鲜"。
    """
    try:
        from db import store as _db
        date, rows = _db.load_market_snapshot(want_date)
    except Exception as e:                                # noqa: BLE001
        _logger.debug("[本地快照] 读库失败: %s", e)
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows)
    ren = {"stock_code": "代码", "stock_name": "名称", "price": "最新价",
           "change_pct": "涨跌幅", "pe": "市盈率-动态", "pb": "市净率",
           "amount": "成交额", "turnover": "换手率", "market_cap": "总市值"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    if "收盘" not in df.columns and "最新价" in df.columns:
        df["收盘"] = df["最新价"]                        # market_detector 读 df["收盘"]
    df.attrs.update({
        "coverage": 1.0, "degraded": False, "failed_pages": [],
        "total_expected": len(df), "source": "local_market_snapshot",
        "data_date": date,
    })
    return df


def get_market_snapshot(force_refresh: bool = False, *, prefer_local: bool = True):
    """对外入口：**本地库优先**（P0-1）→ 直连东财 → akshare 兜底。

    为什么改成本地优先：`MarketDetector.detect()` 一次打 3 次全市场快照
    （每次 55 页），断网时选股直接卡死在 Stage 0；而 15:40 已经把当天截面
    落进 `market_snapshot` 表了。同一天再拉一遍纯浪费，且盘中/盘后语义相同
    （都是用"最近可得截面"判断市场状态）。

    ⚠️ 判据是**数据日期**而不是"库里有就行"：
      - 库里最新日期 == 期望行情日（盘后/次日盘前）⇒ 用库（新鲜）；
      - 库里较旧（盘中）⇒ 仍尝试联网拿实时；联网失败才回退旧库，
        并让 `attrs["degraded"]=True` + `attrs["data_date"]` 如实带出滞后。
    """
    now = time.time()
    cached = _snapshot_cache["df"]
    if cached is not None and not force_refresh \
            and now - _snapshot_cache["ts"] < _SNAPSHOT_TTL:
        return cached

    # ── 1. 本地库（新鲜才用）──
    if prefer_local and not force_refresh:
        try:
            from data.realtime import _expected_quote_date
            want = _expected_quote_date()
        except Exception:                                 # noqa: BLE001
            want = None
        try:
            local = _local_snapshot_df(want)
        except Exception as e:                            # noqa: BLE001
            _logger.debug("[快照] 本地库读取失败，改走联网: %s", e)
            local = None
        if local is not None and len(local) > 0:
            _snapshot_cache["df"] = local
            _snapshot_cache["ts"] = now
            _logger.info("[快照] 用本地库（数据日期 %s，%d 只）",
                         local.attrs.get("data_date"), len(local))
            return local

    df = None
    try:
        df = fetch_market_snapshot()
    except Exception as e:                           # noqa: BLE001
        _logger.warning("[东财快照] 直连失败: %s，退回 akshare", e)
    if df is None or df.empty:
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
        except Exception as e:                       # noqa: BLE001
            _logger.error("[东财快照] akshare 兜底也失败: %s", e)
            df = None
    # ── 2. 联网也失败 ⇒ 退到本地旧库（标注滞后，绝不假装实时）──
    if (df is None or df.empty) and prefer_local:
        local = _local_snapshot_df(None)
        if local is not None and len(local) > 0:
            local.attrs["degraded"] = True
            local.attrs["stale"] = True
            _logger.warning("[快照] 实时源不可用 ⇒ 回退本地库（数据日期 %s，已滞后）",
                            local.attrs.get("data_date"))
            df = local
    if df is not None and not df.empty:
        _snapshot_cache["df"] = df
        _snapshot_cache["ts"] = now
    return df


def clear_snapshot_cache():
    """清掉快照缓存（测试/强制刷新用）"""
    _snapshot_cache["df"] = None
    _snapshot_cache["ts"] = 0.0
