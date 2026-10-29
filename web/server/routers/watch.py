# -*- coding: utf-8 -*-
"""看盘五区。"""
from __future__ import annotations

from fastapi import APIRouter

from web.server import watch

router = APIRouter(tags=["watch"])


@router.get("/api/watch/indices")
def api_watch_indices() -> dict:
    return watch.indices()


@router.get("/api/watch/watchlist")
def api_watch_watchlist(owner: str = "user") -> dict:
    """某轨自选。`owner=user`（我的，默认）/ `owner=ashen`（阿深的）。

    两轨**互相可见**：这个端点是只读的，任意 owner 都能读，页面用 tab 同时展示。
    写入走 `POST /api/watch/watchlist`。
    """
    return watch.watchlist(owner=owner)


@router.post("/api/watch/watchlist")
def api_watch_watchlist_edit(payload: dict) -> dict:
    """自选股增删改（人在页面上直接操作）。

    body:
        action  add | remove | update
        owner   user（默认）| ashen
        code    6 位 A 股代码
        name    可选；add 留空会用行情源补名称，update 留空/不传 = 不改
        note    可选；update 传 `""` = 清空备注

    ⚠️ 与 `tool_manage_watchlist`（模型调）的分工见 `watch.edit_watchlist` 文档串：
    模型写要 preview→commit，人写点一下即可，**不要**再套一层确认。
    """
    p = payload or {}
    return watch.edit_watchlist(
        action=p.get("action", ""),
        owner=p.get("owner", "user"),
        code=p.get("code", ""),
        name=p.get("name"),
        note=p.get("note"),
    )


@router.get("/api/watch/industry")
def api_watch_industry() -> dict:
    return watch.industry()


@router.get("/api/watch/us")
def api_watch_us() -> dict:
    return watch.us()


@router.get("/api/watch/alerts")
def api_watch_alerts(limit: int = 30, date: str | None = None) -> dict:
    # 参数名用 date（对外）但内部叫 trade_date——避免和 datetime 混
    return watch.alerts(limit=limit, trade_date=date)
