# -*- coding: utf-8 -*-
"""`/api/intel/*` —— 情报面（P2）：全市场快讯流 + 窗口统计。

## 全部零外呼
只读 `data/portfolio.db` 的 `news_items` 表（采集是 `hourly_news` 任务的事）。
看情报页不该触发一次采集——那会让"打开页面"变成"打一次上游接口"。

## 三个必须说清的口径
1. **情绪标签是词表判定**，不是人工标注：`data.news_feed._SECTOR_KEYWORDS` 与
   `classify_sentiment` 都是**唯一来源**，服务层只 import，不复制一份词表
   （复制就会漂移：库内标签用旧词表、页面筛选用新词表，同一句话两个结论）；
2. **行业不是库内字段**（`news_items` 无 `industry` 列）⇒ 按标题/正文命中词表实时归类。
   所以「行业筛选」是**检索口径**，不是入库口径；
3. **统计分两层**且都标明范围：`window` = 时间窗内全量，`filtered` = 当前筛选条件下。
   不把"筛选后 46 条"说成"全市场 46 条"。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from web.server import cache, envelope

_logger = logging.getLogger(__name__)

SENTIMENTS = ("positive", "negative", "neutral")
MAX_LIMIT = 300
MAX_HOURS = 720            # 30 天；再长就该走归档，不该在页面上扫全库


def sector_keywords() -> dict:
    """行业词表**唯一来源**（避免服务层复制第二套）。"""
    from data.news_feed import _SECTOR_KEYWORDS
    return _SECTOR_KEYWORDS


def sector_names() -> list[str]:
    return list(sector_keywords().keys())


def news(hours: int = 24, limit: int = 60, industry: str | None = None,
         sentiment: str | None = None, now: datetime | None = None) -> dict:
    """快讯流 + 两层统计。"""
    hrs = _clamp_int(hours, 1, MAX_HOURS, 24)
    lim = _clamp_int(limit, 1, MAX_LIMIT, 60)

    kw = sector_keywords()
    if industry and industry not in kw:
        return envelope.fail("invalid", f"未知行业: {industry}（可选：{'/'.join(kw)}）")
    if sentiment and sentiment not in SENTIMENTS:
        return envelope.fail("invalid", f"未知情绪: {sentiment}（可选：{'/'.join(SENTIMENTS)}）")

    if now is not None:
        return _news(hrs, lim, industry, sentiment, now)
    return cache.cached("intel_news", (hrs, lim, industry, sentiment, _today_key()),
                        30, lambda: _news(hrs, lim, industry, sentiment, datetime.now()))


def _news(hours: int, limit: int, industry: str | None, sentiment: str | None,
          now: datetime) -> dict:
    kw = sector_keywords()
    since = (now - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

    window = _count(since, None, None)
    if window is None:
        return envelope.fail("empty", "news_items 不可读")
    filtered = _count(since, industry, sentiment)
    if filtered is None:
        return envelope.fail("empty", "news_items 不可读")

    items, err = _fetch(since, limit, industry, sentiment)
    if err:
        return envelope.fail("empty", err)

    rows = []
    for it in items:
        text = f"{it.get('title') or ''} {it.get('content') or ''}"
        title, preview = _title_and_preview(it)
        rows.append({
            "id": it.get("id"),
            "title": title,
            "preview": preview,
            "sentiment": it.get("sentiment"),
            "source": it.get("source"),
            "published_at": it.get("published_at"),
            "related_codes": it.get("related_codes") or [],
            "sectors": _sectors_of(text, kw),
        })

    latest = _latest()
    data = {
        "window": {"hours": hours, "since": since,
                   "total": window["total"], "by_sentiment": window["by_sentiment"]},
        "filtered": {"industry": industry, "sentiment": sentiment,
                     "total": filtered["total"], "by_sentiment": filtered["by_sentiment"]},
        "items": rows,
        "returned": len(rows),
        "industries": list(kw.keys()),
        "db": latest,
    }
    return envelope.ok(data, as_of=latest.get("latest"))


def _preview(content, n: int = 140) -> str:
    """正文预览：压空白后截断。**不做摘要**（那是 LLM 的事）。"""
    if not content:
        return ""
    s = " ".join(str(content).split())
    return s[:n] + ("…" if len(s) > n else "")


def _title_and_preview(item: dict, title_len: int = 70) -> tuple[str, str]:
    """标题 + 正文预览。

    ⚠️ **新浪源的 `title` 是空串**，正文全在 `content` 里（2026-10-25 实测：
       24h 窗口内 100+ 条 sina 快讯标题为空）。直接用 `title` 渲染就是一排空白行。
    所以标题为空时从正文起头切一段当标题，预览从切掉的位置继续 —— 不重复同一句话。
    """
    title = (item.get("title") or "").strip()
    content = item.get("content") or ""
    if title:
        return title, _preview(content)
    flat = " ".join(str(content).split())
    if not flat:
        return "（无标题）", ""
    head = flat[:title_len] + ("…" if len(flat) > title_len else "")
    return head, _preview(flat[title_len:])


def _sectors_of(text: str, kw: dict) -> list[str]:
    return [name for name, words in kw.items() if any(w in text for w in words)]


def _industry_clause(industry: str | None, kw: dict) -> tuple[str, list]:
    """行业 ⇒ SQL 片段。用词表 OR 拼接，保证分页在 SQL 侧生效（不是取完再筛）。"""
    if not industry:
        return "", []
    words = kw.get(industry) or []
    if not words:
        return "", []
    parts, params = [], []
    for w in words:
        parts.append("(title LIKE ? OR content LIKE ?)")
        params += [f"%{w}%", f"%{w}%"]
    return " AND (" + " OR ".join(parts) + ")", params


def _count(since: str, industry: str | None, sentiment: str | None) -> dict | None:
    kw = sector_keywords()
    clause, cparams = _industry_clause(industry, kw)
    sql = "SELECT sentiment, COUNT(*) FROM news_items WHERE published_at >= ?" + clause
    params: list = [since] + cparams
    if sentiment:
        sql += " AND sentiment = ?"
        params.append(sentiment)
    sql += " GROUP BY sentiment"
    try:
        from db import store
        store.init_store()
        conn = store.get_connection()
        try:
            got = {r[0]: int(r[1]) for r in conn.execute(sql, params).fetchall()}
        finally:
            conn.close()
    except Exception as e:                     # noqa: BLE001
        # 读库失败 ≠ 0 条：返回 None 让调用方报错（铁律：缺就是缺，不返 0）
        _logger.warning("[intel] 快讯计数失败: %s: %s", type(e).__name__, e)
        return None
    by = {s: got.get(s, 0) for s in SENTIMENTS}
    return {"total": sum(by.values()), "by_sentiment": by}


def _fetch(since: str, limit: int, industry: str | None,
           sentiment: str | None) -> tuple[list, str]:
    kw = sector_keywords()
    clause, cparams = _industry_clause(industry, kw)
    sql = "SELECT * FROM news_items WHERE published_at >= ?" + clause
    params: list = [since] + cparams
    if sentiment:
        sql += " AND sentiment = ?"
        params.append(sentiment)
    sql += " ORDER BY published_at DESC, id DESC LIMIT ?"
    params.append(int(limit))
    try:
        from db import store
        store.init_store()
        conn = store.get_connection()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[intel] 快讯查询失败: %s: %s", type(e).__name__, e)
        return [], f"{type(e).__name__}"
    out = []
    for r in rows:
        d = dict(r)
        try:
            import json
            d["related_codes"] = json.loads(d.get("related_codes") or "[]")
        except (TypeError, ValueError):
            d["related_codes"] = []
        out.append(d)
    return out, ""


def _latest() -> dict:
    """全库总量与最新时间（页面顶部"库内 N 条"用）。"""
    try:
        from db import store
        s = store.news_stats()
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[intel] news_stats 读取失败: %s: %s", type(e).__name__, e)
        return {"total": None, "latest": None, "today": None, "error": f"{type(e).__name__}"}
    return {"total": s.get("total"), "latest": s.get("latest"), "today": s.get("today")}


def _today_key() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _clamp_int(v, lo: int, hi: int, default: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


# ================================================================
# 行业研判 + 情报选股（2026-10-29）
# ================================================================
# 背景：情报页原先只有「快讯流 + 知乎」，链路后半段（每天 15:55 的行业判定 →
# 利好行业 × 成分股 → 个股清单）只在 Streamlit 的「热点新闻」页可见，Web 侧完全
# 没有出口 ⇒ 用户在情报页看不到"新闻 → 个股"这一步。这里把同一份产出暴露出来。
#
# ⚠️ 两条约束：
# 1) **零外呼**：全部只读库 + 只读磁盘缓存。成分映射在请求里联网 = 49 次请求≈1 分钟
#    （行业）/ 20 页≈52 秒（概念），会把只读端点变成不可预测的长尾。
# 2) **不重算口径**：行业判定与个股排序全部复用 `core.news_industry_review`，
#    服务层只做"取出来 + 包信封"，绝不复制第二套排序或词表（复制必漂移）。
MAX_TOP_N = 20
VERDICTS = ("利好", "利空", "中性")


def reviews(date: str | None = None) -> dict:
    """行业利好/利空判定（`news_industry_reviews`，每天 15:55 生成）。

    Returns: `{review_date, counts:{利好,利空,中性,total}, rows:[...], dates:[...]}`
    """
    if date is not None:
        date = str(date).strip() or None
        if date and not _looks_like_date(date):
            return envelope.fail("invalid", f"日期格式应为 YYYY-MM-DD: {date}")
    if date is None:
        return cache.cached("intel_reviews", (_today_key(),),
                            60, lambda: _reviews(None))
    return cache.cached("intel_reviews", (date, _today_key()), 60,
                        lambda: _reviews(date))


def _reviews(date: str | None) -> dict:
    from db import store
    try:
        review_date, rows = store.get_news_reviews(date)
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[intel] 行业评估读取失败: %s: %s", type(e).__name__, e)
        return envelope.fail("empty", f"news_industry_reviews 不可读: {type(e).__name__}")

    try:
        dates = store.news_review_dates(limit=30)
    except Exception as e:                     # noqa: BLE001
        # 日期列表只是辅助（页面切换用），取不到不该让整块失败
        _logger.warning("[intel] 评估日期列表读取失败: %s: %s", type(e).__name__, e)
        dates = []

    if not review_date or not rows:
        return envelope.fail(
            "missing",
            "还没有行业评估结果（每天 15:55 由 news_industry_review 任务生成）；"
            "没有结论时不会自动编造")

    counts = {v: 0 for v in VERDICTS}
    out_rows = []
    for r in rows:
        verdict = r.get("verdict") or ""
        if verdict in counts:
            counts[verdict] += 1
        out_rows.append({
            "sector": r.get("sector"),
            "verdict": verdict,
            "summary": r.get("summary") or "",
            "news_count": r.get("news_count"),
            "model": r.get("model"),
            # evidence 已在 store 里反序列化成 [{"id","title"}]；损坏时是 []
            "evidence": r.get("evidence") or [],
        })
    data = {
        "review_date": review_date,
        "counts": {**counts, "total": len(out_rows)},
        "rows": out_rows,
        "dates": dates,
        "dictionary": {"verdicts": list(VERDICTS),
                       "note": "判定由 LLM 给出、证据 id 由代码校验；"
                               "行业口径与 news_feed._SECTOR_KEYWORDS 一致"},
    }
    return envelope.ok(data, as_of=review_date)


def picks(date: str | None = None, top_n: int = 5) -> dict:
    """利好行业 → 成分股 → 按近 3 日主力净流入排序的个股清单。

    **只读**：成分映射只走磁盘缓存（`allow_network=False`）；没有 picks 时返回
    `ok=false` + `error.detail` 说明原因，不返回空表冒充正常。
    """
    n = _clamp_int(top_n, 1, MAX_TOP_N, 5)
    if date is not None:
        date = str(date).strip() or None
        if date and not _looks_like_date(date):
            return envelope.fail("invalid", f"日期格式应为 YYYY-MM-DD: {date}")
    ident = (date, n, _today_key())
    return cache.cached("intel_picks", ident, 60, lambda: _picks(date, n))


def _picks(date: str | None, top_n: int) -> dict:
    from core.news_industry_review import recommend_stocks

    try:
        r = recommend_stocks(review_date=date, top_n=top_n, allow_network=False)
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[intel] 情报选股失败: %s: %s", type(e).__name__, e)
        return envelope.fail("empty", f"推荐生成失败: {type(e).__name__}: {str(e)[:120]}")

    status = r.get("status")
    if status != "ok" or not r.get("picks"):
        # no_reviews / no_bullish / no_members 都是"没有可用结论"，
        # 把 note 透给前端（页面直接显示，不需要再猜）
        return envelope.fail("missing", r.get("note") or f"没有可推荐的个股（{status}）")

    data = {
        "review_date": r.get("review_date"),
        "flow_date": r.get("flow_date"),
        "picks": r.get("picks") or [],
        "degraded": bool(r.get("degraded")),
        "warnings": r.get("warnings") or [],
        "note": r.get("note") or "",
        "basis": {
            "rank": "近3日主力净流入（fund_flow_snapshot.net_3d）",
            "members": "行业=新浪49行业磁盘缓存；概念=东财题材磁盘缓存（开关 "
                       "BIGA_CONCEPT_PICK）",
            "read_only": True,
        },
    }
    return envelope.ok(data, as_of=r.get("review_date"))


def _looks_like_date(s: str) -> bool:
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except (TypeError, ValueError):
        return False
