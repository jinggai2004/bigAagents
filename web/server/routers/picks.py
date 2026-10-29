# -*- coding: utf-8 -*-
"""选股 + 首页聚合（P5 起含"跑一次选股"的写入口）。"""
from __future__ import annotations

from fastapi import APIRouter

from web.server import dashboard, jobs, picks

router = APIRouter(tags=["picks"])


@router.get("/api/picks/latest")
def api_picks_latest() -> dict:
    return picks.load_latest()


@router.get("/api/dashboard")
def api_dashboard() -> dict:
    return dashboard.build()


@router.get("/api/picks/options")
def api_picks_options() -> dict:
    """初筛策略清单（前端下拉框的唯一来源，不在 JS 里写死一份）。"""
    return jobs.screen_options()


@router.post("/api/picks/run")
def api_picks_run(sector: str | None = None, strategies: str | None = None,
                  max_stocks: int = 5) -> dict:
    """触发选股。`strategies` 用逗号分隔（`?strategies=主力资金,技术突破`）。"""
    picked = [s.strip() for s in (strategies or "").split(",") if s.strip()]
    return jobs.run_screen(sector=sector, strategies=picked or None,
                           max_stocks=max_stocks)
