# -*- coding: utf-8 -*-
"""市场宽度（涨跌家数）—— `market_breadth()`。

## 为什么单独一个模块
首页那一格原来挂的是**写死的演示数**（3,124 / 1,689 + 炸板率 21%），
`data-demo` 标注把它列进"演示"里，用户看到的却是像真的数字 —— 这是最坏的一种
假数据：标注了但没人读。`dashboard.py` 当年不做它，理由是"从 panel 逐只统计太贵"，
这个理由成立（5830 只 × 8.6ms ≈ 50s），但**结论错了**：库里早就有两处现成的
每日全市场 `change_pct`，一次 SQL 就能聚合。

## 源链（按覆盖率择优，全程无外呼）
1. `market_snapshot`：全量域落库后 ≈5,570 只（价量全覆盖），估值只覆盖子集 ——
   但宽度只需要 `change_pct`，不碰估值列。
2. `fund_flow_snapshot`（同花顺，15:40 采集）：实测 **5,217 只/日**，历史可回溯到 10 月初。

## 铁律
- **分母必须自报**：`covered`（实际参与统计的只数）与 `universe`（A 股清单只数）
  一起给，页面显示"5,123 / 5,570"。**不允许**只给一个家数让读者以为是全市场。
- **`change_pct` 为 NULL 的行不进任何一边**（既不算涨也不算跌）—— 停牌股没有涨跌幅，
  把它算成"平盘"会把"平盘家数"抬高几十倍（旧快照里 186 只无报价退市票就是这么来的）。
- **找不到当日数据就说找不到**，绝不拿上一交易日的宽度冒充今天
  （`stale=True` + `date` 如实报，由页面决定怎么显示）。
"""
from __future__ import annotations

import logging
from datetime import datetime

from db import store

_logger = logging.getLogger(__name__)

# 涨跌家数的最小可信覆盖：低于这个比例说明快照采残了，宁可标 degraded
MIN_COVERAGE_RATIO = 0.60


def _universe_size() -> int | None:
    """A 股清单只数（本地缓存的代码清单；拿不到返回 None）。"""
    try:
        from data import sina_source
        codes = sina_source.market_codes()
        return len(codes) if codes else None
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[宽度] 取 A 股清单失败（覆盖率分母降级为 None）: %s", e)
        return None


def _counts_from_market_snapshot(date: str | None) -> dict | None:
    conn = store.get_connection()
    try:
        if not date:
            row = conn.execute(
                "SELECT MAX(snapshot_date) AS d FROM market_snapshot").fetchone()
            date = row["d"] if row else None
        if not date:
            return None
        r = conn.execute(
            """SELECT COUNT(*) AS covered,
                      SUM(CASE WHEN change_pct > 0 THEN 1 ELSE 0 END) AS up,
                      SUM(CASE WHEN change_pct < 0 THEN 1 ELSE 0 END) AS down,
                      SUM(CASE WHEN change_pct = 0 THEN 1 ELSE 0 END) AS flat,
                      SUM(CASE WHEN price IS NULL OR price <= 0 THEN 1 ELSE 0 END) AS no_quote
               FROM market_snapshot
               WHERE snapshot_date = ? AND change_pct IS NOT NULL""",
            (date,)).fetchone()
        if not r or not r["covered"]:
            return None
        return {"date": date, "source": "market_snapshot",
                "covered": int(r["covered"]), "up": int(r["up"] or 0),
                "down": int(r["down"] or 0), "flat": int(r["flat"] or 0),
                "no_quote": int(r["no_quote"] or 0)}
    finally:
        conn.close()


def _counts_from_fund_flow(date: str | None) -> dict | None:
    conn = store.get_connection()
    try:
        if not date:
            row = conn.execute(
                "SELECT MAX(snapshot_date) AS d FROM fund_flow_snapshot").fetchone()
            date = row["d"] if row else None
        if not date:
            return None
        r = conn.execute(
            """SELECT COUNT(DISTINCT stock_code) AS covered,
                      COUNT(DISTINCT CASE WHEN change_pct > 0 THEN stock_code END) AS up,
                      COUNT(DISTINCT CASE WHEN change_pct < 0 THEN stock_code END) AS down,
                      COUNT(DISTINCT CASE WHEN change_pct = 0 THEN stock_code END) AS flat
               FROM fund_flow_snapshot
               WHERE snapshot_date = ? AND change_pct IS NOT NULL""",
            (date,)).fetchone()
        if not r or not r["covered"]:
            return None
        return {"date": date, "source": "ths_fund_flow",
                "covered": int(r["covered"]), "up": int(r["up"] or 0),
                "down": int(r["down"] or 0), "flat": int(r["flat"] or 0),
                "no_quote": None}
    finally:
        conn.close()


def _expected_date() -> str | None:
    """最近应收盘的交易日（与 picks/watch 同一口径，**不用 today()**）。"""
    try:
        from web.server import picks
        return picks.expected_date(datetime.now())
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[宽度] 期望交易日取不到（stale 判定降级）: %s", e)
        return None


def market_breadth(date: str | None = None) -> dict:
    """涨跌家数（A 股口径）。

    Returns:
        {"date", "up", "down", "flat", "covered", "universe", "ratio",
         "up_ratio", "source", "stale", "expected_date", "degraded"}
        或 `{"available": False, "reason": ...}`（**不返回 0**）
    """
    cands: list[dict] = []
    for fn in (_counts_from_market_snapshot, _counts_from_fund_flow):
        try:
            c = fn(date)
        except Exception as e:                                # noqa: BLE001
            _logger.warning("[宽度] %s 聚合失败: %s", fn.__name__, e)
            c = None
        if c:
            cands.append(c)
    if not cands:
        return {"available": False,
                "reason": "无当日全市场截面（market_snapshot / fund_flow_snapshot 都没有）",
                "date": date}

    # 覆盖率优先；同日同样覆盖时按"有更多非空列"取（market_snapshot 多一个 no_quote）
    best = max(cands, key=lambda c: (c["covered"], 1 if c["no_quote"] is not None else 0))
    universe = _universe_size()
    ratio = round(best["covered"] / universe, 4) if universe else None
    expected = _expected_date()
    degraded = []
    if ratio is not None and ratio < MIN_COVERAGE_RATIO:
        degraded.append(f"覆盖 {ratio:.0%} < {MIN_COVERAGE_RATIO:.0%}")
    if best["date"] and expected and best["date"] < expected:
        degraded.append(f"截面停在 {best['date']}，最近应收盘日 {expected}")

    flat_share = best["flat"] / best["covered"] if best["covered"] else None
    if flat_share is not None and flat_share > 0.35:
        degraded.append(f"平盘占比 {flat_share:.0%} 偏高（疑似停牌/无报价混入）")

    return {
        "available": True,
        "date": best["date"],
        "expected_date": expected,
        "stale": bool(best["date"] and expected and best["date"] < expected),
        "up": best["up"], "down": best["down"], "flat": best["flat"],
        "covered": best["covered"],
        "universe": universe,
        "ratio": ratio,
        "up_ratio": round(best["up"] / best["covered"], 4) if best["covered"] else None,
        "no_quote": best["no_quote"],
        "source": best["source"],
        "degraded": degraded,
    }
