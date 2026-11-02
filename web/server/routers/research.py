# -*- coding: utf-8 -*-
"""研究：回测记录 / 因子结果 / 验证与归因（v4，2026-10-29）。"""
from __future__ import annotations

from fastapi import APIRouter

from web.server import research

router = APIRouter(tags=["research"])


@router.get("/api/research/backtest")
def api_research_backtest(file: str | None = None) -> dict:
    return research.backtest(file=file)


@router.get("/api/research/factors")
def api_research_factors() -> dict:
    return research.factors()


@router.get("/api/research/backtest/validate")
def api_research_validate() -> dict:
    """参数扫描的多重检验报告（DSR/PBO，模块 A）。只读最新扫描产物。"""
    return research.backtest_validate()


@router.get("/api/research/walkforward")
def api_research_walkforward() -> dict:
    """walk-forward 权重重估（B 模块）。只读最新产物。"""
    return research.walkforward()


@router.get("/api/research/attributions")
def api_research_attributions(name: str | None = None) -> dict:
    """交易级盈亏分解（3.1）。轻量在线计算 + cache 300s。"""
    return research.attributions(name=name)


@router.get("/api/research/brinson")
def api_research_brinson(start: str | None = None, end: str | None = None,
                         source: str | None = None, min_names: int = 0) -> dict:
    """Brinson 行业归因（配置/选股/交互 + Carino 多期链接）。

    参数：start/end 选股日期区间；source 限定 screen|pipeline；min_names 只保留
    组合规模 ≥ N 的期（用于剔除规模跃变段）。在线计算 ~2s，cache 300s。
    """
    return research.brinson(start=start, end=end, source=source, min_names=min_names)
