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
