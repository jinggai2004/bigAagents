# -*- coding: utf-8 -*-
"""情报面：快讯 + 知乎观点（P5 起含采集入口）。"""
from __future__ import annotations

from fastapi import APIRouter

from web.server import intel, jobs

router = APIRouter(tags=["intel"])


@router.get("/api/intel/news")
def api_intel_news(hours: int = 24, limit: int = 60,
                   industry: str | None = None,
                   sentiment: str | None = None) -> dict:
    return intel.news(hours=hours, limit=limit, industry=industry, sentiment=sentiment)


@router.get("/api/intel/reviews")
def api_intel_reviews(date: str | None = None) -> dict:
    """行业利好/利空判定（每天 15:55 生成，零外呼）。

    `date` 省略 = 最近一次；显式传 `YYYY-MM-DD` 可回看历史（格式非法 → `invalid`）。
    """
    return intel.reviews(date=date)


@router.get("/api/intel/picks")
def api_intel_picks(date: str | None = None, top_n: int = 5) -> dict:
    """利好行业 → 成分股 → 按近 3 日主力净流入排序（只读，零外呼）。

    没有结论时返回 `ok=false` + `error.detail`（不是空表）：`no_reviews`（任务没跑）、
    `no_bullish`（当天无利好行业）、`no_members`（成分映射也不可用）。
    """
    return intel.picks(date=date, top_n=top_n)


@router.get("/api/intel/zhihu/status")
def api_zhihu_status() -> dict:
    return jobs.zhihu_status()


@router.post("/api/intel/zhihu/collect")
def api_zhihu_collect(per_topic: int = 5) -> dict:
    return jobs.run_zhihu(per_topic=max(1, min(20, per_topic)))


@router.get("/api/intel/zhihu/snapshot")
def api_zhihu_snapshot() -> dict:
    return jobs.zhihu_snapshot()
