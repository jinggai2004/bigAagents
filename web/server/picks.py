# -*- coding: utf-8 -*-
"""`/api/picks/latest` —— 只读 `output/daily_picks/latest.json`（零联网、零计算）。

## 三条口径（文档 §4.3 C1/C3/C4）
1. `market_signal` 一律交给 `core.market_signal_codec.coerce`（全项目唯一编解码器）；
2. `stale` 的基准是**最近应收盘的交易日**，不是"今天"——周末/节假日把"昨天"
   判成过期是假警报（项目铁律：基准不能用日历末元素，否则闸门恒真/恒假）；
3. 字段名是**中文原样透传**（`股票代码/股票名称/板块/策略`），不转英文——
   两套前端共用同一套命名，转译只会长出第二套。
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, time

from core.market_signal_codec import coerce
from web.server import cache, envelope, settings

_logger = logging.getLogger(__name__)

# 收盘后多久算"今日数据应已产出"（与 offline_gate / 复盘口径一致）
CLOSE_READY = time(15, 10)


def load_latest(now: datetime | None = None) -> dict:
    """返回 `/api/picks/latest` 的完整包络。

    `now` 仅供测试注入（同时绕过缓存，保证用例互不污染）；生产传 None 走 60s 缓存。
    """
    path = settings.picks_path()
    try:
        st = os.stat(path)
    except FileNotFoundError:
        # 正常状态（盘后流水线还没跑过），区分于"文件坏了"
        _logger.debug("[picks] latest.json 不存在: %s", path)
        return envelope.fail("missing", "output/daily_picks/latest.json 不存在")
    except OSError as e:
        _logger.warning("[picks] latest.json 不可读: %s", e)
        return envelope.fail("missing", f"{type(e).__name__}")

    if now is not None:
        return _load(path, now)
    return cache.cached("picks", (int(st.st_mtime), st.st_size),
                        settings.TTL_PICKS, lambda: _load(path, datetime.now()))


def _load(path: str, now: datetime) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        # 文件在、内容坏了 ⇒ empty（与 missing 区分：一个是没跑，一个是坏了）
        _logger.warning("[picks] latest.json 解析失败: %s: %s", type(e).__name__, e)
        return envelope.fail("empty", f"{type(e).__name__}")

    if not isinstance(raw, dict):
        _logger.warning("[picks] latest.json 顶层不是对象: %s", type(raw).__name__)
        return envelope.fail("empty", "顶层不是对象")

    date = str(raw.get("date") or "")
    signal, fmt = coerce(raw.get("market_signal"))
    expected = expected_date(now)
    lag = trading_lag(date, expected)

    data = {
        "date": date or None,
        "generated_at": raw.get("generated_at"),
        "mode": raw.get("mode"),
        "picks": raw.get("picks") or [],
        "market_signal": signal,
        # 显式回传读到的是哪种格式：旧文件仍是 repr 串，排查时不该靠猜（C1）
        "format": fmt or "unknown",
        "stage_timings": raw.get("stage_timings") or {},
        "skipped": bool(raw.get("skipped")),
        "skip_reason": raw.get("skip_reason") or "",
        "errors": raw.get("errors") or [],
        "elapsed_seconds": raw.get("elapsed_seconds"),
        "stale": (None if lag is None else lag > 0),
        "lag_trading_days": lag,
        "expected_date": expected,
    }
    return envelope.ok(data, as_of=date or None)


def expected_date(now: datetime | None = None) -> str:
    """最近一个**应该已有数据**的交易日。

    - 交易日且已过 15:10 ⇒ 今天；
    - 交易日但未收盘（含盘前/盘中）⇒ 上一交易日；
    - 非交易日（周末/节假日）⇒ 最近交易日。
    这样周六看到周五的数据不会被误判成 stale。
    """
    from data.trading_calendar import is_trading_day, last_trading_day

    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")
    try:
        if is_trading_day(today) and now.time() >= CLOSE_READY:
            return today
        return last_trading_day(today, include_today=False)
    except Exception as e:                     # noqa: BLE001 —— 日历不可用只能是"不知道"
        _logger.warning("[picks] 交易日历不可用，expected_date 退化为今天: %s: %s",
                        type(e).__name__, e)
        return today


def trading_lag(data_date: str, expected: str) -> int | None:
    """数据日期滞后几个交易日。算不出（日期非法/日历不可用）⇒ None，不猜 0。"""
    if not data_date or not expected:
        return None
    if data_date >= expected:                  # 当日或更新（盘中刚跑完）⇒ 不滞后
        return 0
    try:
        from data.trading_calendar import trading_days
        days = trading_days(data_date, expected)
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[picks] 计算交易日滞后失败: %s: %s", type(e).__name__, e)
        return None
    return max(0, len(days) - 1)
