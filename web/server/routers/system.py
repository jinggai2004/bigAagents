# -*- coding: utf-8 -*-
"""系统页：健康 / 任务表 / LLM 成本；以及通用的任务状态与取消。

## 为什么任务状态是 `/api/tasks/{task}` 而不是 `/api/picks/tasks/{id}`
`core.ui_tasks` 里 task 名就是**全局唯一标识**（进度文件名由它拼出来），
选股叫 `web_screen`、分析叫 `web_analysis_600519`、采集叫 `web_zhihu`。
按业务域分路径的话，同一个读取逻辑要写三遍，还容易漏掉新加的任务类型。
"""
from __future__ import annotations

from fastapi import APIRouter

from web.server import jobs, system

router = APIRouter(tags=["system"])


@router.get("/api/system/health")
def api_system_health() -> dict:
    return system.status()


@router.get("/api/system/tasks")
def api_system_tasks() -> dict:
    return system.tasks()


@router.get("/api/system/cost")
def api_system_cost(days: int = 7) -> dict:
    return system.cost(days=max(1, min(30, days)))


@router.get("/api/system/models")
def api_system_models() -> dict:
    return system.models()


@router.post("/api/system/models")
def api_system_models_set(payload: dict) -> dict:
    """改模型配置（功能→模型 / 默认厂商 / 厂商模型清单 / 启停）。

    body: `action` + 各自字段，见 `system.set_models`。
    ⚠️ 这是**人**的通道（下拉框点选即生效），不套两阶段确认；
       模型改配置走的是另一条路（有 preview→commit 的守护）。
    """
    p = payload or {}
    return system.set_models(p.get("action", ""), p)


@router.get("/api/system/audit")
def api_system_audit(limit: int = 50) -> dict:
    """审计与留痕：配置变更流水 / 推送流水 / 情绪标注留痕。

    ⚠️ 必须注册在 `/api/tasks/{task}` 这类参数路由**之前**的同级字面量位置，
    否则可能被前缀更短的通配路由吃掉（项目踩过的同型错误）。
    """
    return system.audit(limit=limit)


@router.get("/api/tasks/{task}")
def api_task(task: str, result: bool = False) -> dict:
    """任务进度；`?result=1` 时带上产物（选股结果 / 分析快照）。"""
    return jobs.state(task, with_result=bool(result))


@router.post("/api/tasks/{task}/cancel")
def api_task_cancel(task: str) -> dict:
    return jobs.cancel(task)
