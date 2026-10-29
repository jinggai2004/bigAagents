# -*- coding: utf-8 -*-
"""`/api/dashboard` —— 分区包络（C2）。

顶层 `ok` 只表示**服务本身正常**；每个分区各自成败：
picks / health / watch / league / observations / breadth 均已实接。

⚠️ 「涨跌家数」在 2026-10-26 之前是**写死的演示数**（3,124 / 1,689），
   当年不做的理由是"要从 panel 逐只统计、太贵"。理由成立但结论错了：
   库里早就有每日全市场 `change_pct`（`fund_flow_snapshot` 5,217 只/日），
   一次 SQL 即可聚合，零外呼。现已由 `core.breadth.market_breadth()` 接管，
   并且**必须把分子分母一起给前端**（`up/down/covered/universe`）——
   只给家数会让人以为那是 5,570 只的全市场。
"""
from __future__ import annotations

import logging

from web.server import desk, envelope, health, picks, review, watch

_logger = logging.getLogger(__name__)

# 首页只要前几名，整张榜单留给交易台
LEAGUE_TOP = 4

# 指数报价走 SWR 时旧值最多容忍多少秒。
# ⚠️ 90 秒：短到不会让"盘中价格"失真到影响判断（1 分钟级别的图看的是趋势），
#    长到足以覆盖一次后台刷新（冷启外呼 ≈1.2s + 抖动）。
WATCH_SWR_STALE = 90.0


def build(now=None) -> dict:
    picks_zone = picks.load_latest(now=now)
    health_zone = health.load_health(
        now_ts=None if now is None else now.timestamp())

    data = {
        "picks": picks_zone,
        "health": health_zone,
        "watch": _watch_zone(now),
        "league": _league_zone(now),
        "observations": review.pending(limit=LEAGUE_TOP + 2),
        "breadth": _breadth_zone(),
        "sources": _sources_zone(),
    }
    return envelope.ok(data, as_of=picks_zone.get("as_of"))


# ---------------------------------------------------------------- 数据源状态
#
# ⚠️ 首页那四行"行情 / 资金流 / 快讯 / LLM"原来是**写死的文字**
#    （腾讯直连 / 同花顺快照 / 新浪+同花顺 / DeepSeek），只挂了 `data-demo`。
#    这和涨跌家数是同一类问题：标注了"演示"，但数字本身像真的，
#    而真实状态（比如"行业缺 2,943 只"这种降级）在首页完全看不到。
#    本函数只读本地库/配置，零外呼，**取不到就报"未知"，绝不返回一个好看的假状态**。

_STATE_BY_COVERAGE = {"ok": "ok", "degraded": "warn", "partial": "warn"}


def _coverage_item(key: str, label: str, dataset: str) -> dict:
    """按 `data_coverage` 最新一条生成一个数据源条目。"""
    from db import store

    try:
        latest = (store.data_coverage_summary(dataset).get("latest") or {})
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[dashboard] 数据源 %s 读取失败: %s", dataset, e)
        return {"key": key, "label": label, "state": "unknown",
                "text": "读取失败", "title": f"{type(e).__name__}"}
    if not latest:
        return {"key": key, "label": label, "state": "unknown",
                "text": "无采集记录", "title": f"data_coverage 里没有 {dataset} 的记录"}

    got, total = latest.get("got"), latest.get("total")
    text = str(latest.get("source") or "—")
    if got is not None and total:
        text += f" {got:,}/{total:,}"
    elif got is not None:
        text += f" {got:,}"
    title = " · ".join(x for x in [
        latest.get("data_date") and f"截面 {latest['data_date']}",
        latest.get("captured_at") and f"采集 {latest['captured_at']}",
        latest.get("status") and f"状态 {latest['status']}",
        latest.get("note"),
    ] if x)
    return {"key": key, "label": label,
            "state": _STATE_BY_COVERAGE.get(str(latest.get("status") or ""), "unknown"),
            "text": text, "title": title}


def _news_item() -> dict:
    """快讯源：按最近 48h 实际入库的 source 列出，不写死"新浪+同花顺"。"""
    import time as _time

    from db import store

    try:
        conn = store.get_connection()
        try:
            since = _time.strftime("%Y-%m-%d %H:%M:%S",
                                   _time.localtime(_time.time() - 48 * 3600))
            rows = conn.execute(
                "SELECT source, COUNT(*) AS n FROM news_items "
                "WHERE created_at >= ? OR published_at >= ? "
                "GROUP BY source ORDER BY n DESC", (since, since)).fetchall()
            total = sum(int(r["n"] or 0) for r in rows)
        finally:
            conn.close()
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[dashboard] 快讯源统计失败: %s", e)
        return {"key": "news", "label": "快讯", "state": "unknown",
                "text": "读取失败", "title": f"{type(e).__name__}"}

    srcs = [str(r["source"]) for r in rows if r["source"]]
    if not srcs:
        return {"key": "news", "label": "快讯", "state": "warn",
                "text": "48h 无入库", "title": "最近 48 小时新闻表没有新增"}
    return {"key": "news", "label": "快讯",
            "state": "ok" if len(srcs) >= 2 else "warn",
            "text": f"{len(srcs)} 源 · {total:,} 条/48h",
            "title": "来源：" + " / ".join(srcs)}


def _llm_item() -> dict:
    """LLM：只报**配置里实际就绪的 provider**，不写死厂商名。"""
    try:
        from web.server import system
        data = (system.models() or {}).get("data") or {}
        provs = data.get("providers") or []
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[dashboard] LLM 配置读取失败: %s", e)
        return {"key": "llm", "label": "LLM", "state": "unknown",
                "text": "配置读取失败", "title": f"{type(e).__name__}"}
    ready = [p for p in provs if p.get("ready") and p.get("enabled")]
    if not ready:
        return {"key": "llm", "label": "LLM", "state": "bad",
                "text": "无可用 provider", "title": "全部 provider 未就绪或未启用"}
    names = " / ".join(str(p.get("name") or p.get("id")) for p in ready)
    return {"key": "llm", "label": "LLM",
            "state": "ok" if len(ready) == len(provs) else "warn",
            "text": f"{len(ready)} 个就绪",
            "title": names}


def _sources_zone() -> dict:
    items = [
        _coverage_item("quotes", "行情", "market_snapshot"),
        _coverage_item("flow", "资金流", "fund_flow_stock"),
        _news_item(),
        _llm_item(),
    ]
    return envelope.ok({"items": items})


def _breadth_zone() -> dict:
    """涨跌家数（全市场宽度）。

    ⚠️ 服务本身正常但**当日无截面**时返 `ok:true, data.available:false`，
    让页面显示"暂无宽度"而不是 0 —— 0 会被读成"全市场平盘"，那是另一件事。
    """
    from core import breadth

    try:
        d = breadth.market_breadth()
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[dashboard] 涨跌家数失败: %s", e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")
    return envelope.ok(d, as_of=d.get("date"))


def _league_zone(now=None) -> dict:
    """首页联赛快照：只取日频组前 N 名 + 随机均值，不搬净值曲线。"""
    zone = desk.league(now=now)
    if not zone.get("ok"):
        return zone
    d = zone["data"]
    g = (d.get("groups") or {}).get("daily") or {}
    rows = sorted(g.get("rows") or [], key=lambda r: r.get("return_pct") or 0, reverse=True)
    return envelope.ok({
        "as_of": d.get("as_of"),
        "rows": [{"label": r.get("label") or r.get("persona"),
                  "kind": r.get("kind"), "model": r.get("model"),
                  "nav_days": r.get("nav_days"),
                  "return_pct": r.get("return_pct"),
                  "excess": r.get("excess_vs_random_mean"),
                  "conclusive": r.get("conclusive")} for r in rows[:LEAGUE_TOP]],
        "random_mean": g.get("random_mean_return_pct"),
        "random_count": g.get("random_count"),
        "total": len(rows),
        "target": d.get("target") or {},
    }, as_of=d.get("as_of"))


def _watch_zone(now=None) -> dict:
    """首页只要指数 + 沪市成交额。**不转发 spark**（首屏没人看 20 根线，白占带宽）。

    冷启动走 SWR：这格里唯一的外呼是实时报价（≈1.2s 冷启），首屏不该为它排队。
    代价是可能拿到 ≤90s 前的价 —— 所以 `data_age_seconds` 与每行的 `quote_time`
    一起给出去，前端有据可查，不是"假装很新"。
    """
    zone = watch.indices(now=now, max_stale=0 if now is not None else WATCH_SWR_STALE)
    if not zone.get("ok"):
        return zone
    d = zone["data"]
    rows = []
    for r in d.get("indices") or []:
        rows.append({"symbol": r.get("symbol"), "name": r.get("name"),
                     "price": r.get("price"), "basis": r.get("basis"),
                     "change": r.get("change"), "change_pct": r.get("change_pct"),
                     "quote_time": r.get("quote_time"),
                     "stale": r.get("stale"), "data_lagging": r.get("data_lagging")})
    return envelope.ok({
        "indices": rows,
        "live": d.get("live"),
        "amount_yi": _sh_amount(d),
        "amount_scope": "沪市",
        "data_age_seconds": watch.indices_age(),
        "note": d.get("note") or "",
    }, as_of=d.get("as_of"))


def _sh_amount(d: dict):
    """沪市成交额（亿元）。

    ⚠️ 只取上证指数那一条：`sz399001`（深证成指）的成交额是**成分股**合计，
    不是深市全部；把两者相加当成"两市成交额"会系统性低估 —— 所以宁可只报沪市。
    """
    for r in d.get("indices") or []:
        if r.get("symbol") == "sh000001":
            wan = r.get("amount_wan")
            if wan is None:
                return None
            try:
                return round(float(wan) / 1e4, 0)     # 万元 → 亿元
            except (TypeError, ValueError):
                _logger.debug("[dashboard] 成交额不可转换: %r", wan)
                return None
    return None
