# -*- coding: utf-8 -*-
"""多 AI 交易联赛 · 每日复盘组装（P1，2026-10-14）。

给 UI 用：对每个 AI 汇总
  · **前一日交易复盘**：AI 自述复盘（`ai_reviews`）+ **程序化事实核对**（账本说了什么）；
  · **今日交易操作的理由**：`ai_decisions.order_reason`（决策轮里模型给的原话）；
  · **账户事实**：持仓 / 当日成交 / 净值变化。

⚠️ 纪律（与本项目其它模块一致）：
  1. **数字一律来自已落库数据，不经 LLM**（"决策数字不经 LLM"铁律）；
  2. **AI 自述与程序化事实分开呈现、不合并** —— 自述可能不准，事实由账本给出，
     放在一起用户才能看出"AI 说的"与"账本记的"是否一致（不互相覆盖）；
  3. 只读、**零联网、零 LLM** ⇒ 离线模式安全；取不到的字段一律 `None`/空列表并带 warnings。

时间语义：`trade_date` = **决策日 D**（盘后决策轮出决策的那天）。
  · "今日操作理由" = `signal_date == D` 的决策；
  · "前一日复盘" = `signal_date == D 的上一交易日` 的决策 + 期间实际成交/净值变化。
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

from core import ai_personas
from db import store

_logger = logging.getLogger(__name__)


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _prev_trading_day(day: str) -> str | None:
    """D 的**前一交易日**；取不到日历返回 None（不硬编日期）。

    ⚠️ 离线模式（`BIGA_UI_OFFLINE=1`）**直接返回 None**：交易日历刷新在缓存过期时
    会请求 akshare，而取数源无超时 ⇒ 会让页面**挂起**（不是抛异常，`except` 接不住）。
    离线冒烟必须零联网，所以这里显式短路。
    """
    if os.environ.get("BIGA_UI_OFFLINE") == "1":
        return None
    try:
        from data.trading_calendar import last_trading_day
        return last_trading_day(ref=day, include_today=False)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[AI复盘] 取前一交易日失败: %s", e)
        return None


def _decisions_of(persona: str, date: str) -> list[dict]:
    """某日决策（含"今日操作理由"），按 code 排序；空 = 那天没决策。"""
    try:
        rows = store.list_ai_decisions(persona=persona, signal_date=date)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[AI复盘] 读 %s@%s 决策失败: %s", persona, date, e)
        return []
    out = []
    for r in rows:
        out.append({
            "code": r.get("code"),
            "advice_status": r.get("advice_status"),
            "direction": r.get("direction"),
            "position_size": r.get("position_size"),
            "reason": r.get("order_reason"),
            "order_id": r.get("order_id"),
            "order_status": r.get("order_status"),
        })
    return out


def _nav_at(nav_list: list, date: str) -> dict | None:
    for n in (nav_list or []):
        if str(n.get("date")) == str(date):
            return n
    return None


def _fills_on(fills: list, date: str) -> list[dict]:
    return [f for f in (fills or []) if str(f.get("date")) == str(date)]


def persona_review(name: str, day: str, prev: str | None,
                   persona: dict | None = None) -> dict:
    """组装**一个** AI 的当日复盘视图（只读）。"""
    p = persona or ai_personas.get_persona(name) or {"name": name}
    st = None
    try:
        st = store.load_paper_state(name)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[AI复盘] 读 %s 账户失败: %s", name, e)

    nav_list = (st or {}).get("nav") or []
    fills = (st or {}).get("fills") or []
    pos = (st or {}).get("positions") or {}

    today_nav = _nav_at(nav_list, day)
    prev_nav = _nav_at(nav_list, prev) if prev else None
    # ⚠️ 净值变化**只在两天口径都干净时**才给：任一天有"按成本估值"的持仓，
    #    这个变化率就是拿两个不同口径相减 ⇒ 会给用户/AI 一个假数字。
    #    （2026-10-15 实测：10-14 是真实市价、10-15 是成本价兜底 ⇒ AI 读成"净值不变"）
    draft_note = None
    _today_ac = (today_nav or {}).get("at_cost") or 0
    _prev_ac = (prev_nav or {}).get("at_cost") or 0
    if today_nav is not None and _today_ac:
        draft_note = (f"当日净值含 {_today_ac} 只**按成本估值**的持仓"
                      f"（取不到当日收盘价）⇒ 该净值不可用于判断盈亏")
    change_pct = None
    if today_nav and prev_nav and prev_nav.get("total_value") and not (_today_ac or _prev_ac):
        change_pct = round((today_nav["total_value"] / prev_nav["total_value"] - 1) * 100, 2)
    elif today_nav and prev_nav and (_today_ac or _prev_ac):
        draft_note = ((draft_note + "；" if draft_note else "") +
                      "任一日估值不完整 ⇒ **不计算净值变化率**（缺就是缺）")

    review_row = None
    try:
        review_row = store.get_ai_review(name, day)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[AI复盘] 读 %s 复盘失败: %s", name, e)

    return {
        "persona": name,
        "label": p.get("label") or name,
        "kind": p.get("kind"),
        "model": p.get("model"),
        "frequency": p.get("frequency") or "daily",
        # —— 今日操作 ——（决策日 = day）
        "today_date": day,
        "today_decisions": _decisions_of(name, day),
        "today_summary": (review_row or {}).get("summary"),
        # —— 前一日复盘 ——
        "prev_date": prev,
        "prev_decisions": _decisions_of(name, prev) if prev else [],
        "ai_review_text": (review_row or {}).get("review_text"),
        # —— 账户事实（程序化，不经 LLM）——
        "holds_days": len(nav_list),
        "cash": (st or {}).get("cash"),
        "positions": [{"code": c, "qty": d.get("quantity"),
                       "available": d.get("available"),
                       "avg_cost": d.get("avg_cost"),
                       "last_buy_date": d.get("last_buy_date")}
                      for c, d in pos.items()],
        "fills_today": _fills_on(fills, day),
        "fills_prev": _fills_on(fills, prev) if prev else [],
        "nav_today": today_nav,
        "nav_prev": prev_nav,
        "nav_change_pct": change_pct,
        "nav_note": draft_note,
    }


def daily_review(trade_date: str = None, *, personas: list = None) -> dict:
    """组装**所有** AI 的当日复盘（页面用）。

    Returns: {"as_of", "prev_trading_day", "rows": [...], "random_mean_return_pct",
              "warnings": [...]}
    """
    day = trade_date or _today()
    plist = personas if personas is not None else ai_personas.load_personas()
    prev = _prev_trading_day(day)
    warnings: list[str] = []
    if prev is None:
        warnings.append("前一交易日取不到（交易日历不可用）——复盘仅展示今日。")

    rows = [persona_review(p["name"], day, prev, persona=p) for p in plist]

    # 随机对照组当日的平均净值变动（给 LLM 组一个"今天跑赢随机了吗"的参照）
    rnd = [r["nav_change_pct"] for r in rows
           if r.get("kind") == "random" and r.get("nav_change_pct") is not None]
    rnd_mean = round(sum(rnd) / len(rnd), 2) if rnd else None

    return {"as_of": day, "prev_trading_day": prev, "rows": rows,
            "random_mean_return_pct": rnd_mean, "warnings": warnings}
