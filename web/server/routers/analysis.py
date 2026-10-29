# -*- coding: utf-8 -*-
"""深度分析：触发 / 轮询 / 历史 / 回看 / 胜率。

⚠️ 路由顺序要紧：`/api/analysis/history`、`/api/analysis/stats`、
   `/api/analysis/detail/{ana_id}` **必须**声明在 `/api/analysis/{code}` 之前，
   否则 FastAPI 会把字面量 `history` / `stats` / `detail` 当成股票代码去匹配
   （然后报"必须是 6 位数字"）。
"""
from __future__ import annotations

from fastapi import APIRouter

from web.server import jobs

router = APIRouter(tags=["analysis"])


@router.get("/api/analysis/history")
def api_analysis_history(code: str | None = None, limit: int = 30) -> dict:
    return jobs.analysis_history(code=code, limit=limit)


@router.get("/api/analysis/stats")
def api_analysis_stats(days: int = 180, horizon: int = 5) -> dict:
    """往期胜率 + 归因分布（只读，不触发 LLM）。"""
    return jobs.analysis_stats(days=days, horizon=horizon)


@router.get("/api/analysis/detail/{ana_id}")
def api_analysis_detail(ana_id: int) -> dict:
    """单条历史分析的完整快照（历史列表点击回看用）。"""
    return jobs.analysis_detail(ana_id)


@router.post("/api/analysis/{code}")
def api_analysis_run(code: str, force: bool = False) -> dict:
    """触发分析。已在跑时返回 `started:false, running:true`（不是错误）。"""
    return jobs.run_analysis(code, force=force)


@router.get("/api/analysis/{code}")
def api_analysis_state(code: str) -> dict:
    """轮询：进度 + 已完成节点的快照 + 最终结果（跑完后 `result` 有值）。"""
    return jobs.state(jobs.analysis_task(code.strip()), with_result=True)
