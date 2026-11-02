# -*- coding: utf-8 -*-
"""因子实验室：用户自定义因子的导入 / 回测 / 应用。

两个路径都固定用字面量 `/api/factors`（GET 读、POST 写），不用 `/api/factors/{id}`
这类参数路由——本项目踩过"字面量路由被参数路由吃掉"的坑，少一个维度少一个坑。
"""
from __future__ import annotations

from fastapi import APIRouter

from web.server import factor_lab

router = APIRouter(tags=["factors"])


@router.get("/api/factors")
def api_factors() -> dict:
    """面板概况 + 字段清单（含每个字段真实覆盖率）+ 已登记因子。"""
    return factor_lab.state()


@router.post("/api/factors")
def api_factors_submit(payload: dict) -> dict:
    """`action` = validate / preview / backtest / save / status / delete。

    - `backtest` 是**纯读**（不落库），可以随便试——但每次试都会计入 DSR 的试验次数，
      这是有意的：反复改表达式试到显著，本质就是多重比较。
    - `save` 会把回测结论一起落库；能不能 `applied` 由 core 判据决定，页面不能绕过。
    """
    p = payload or {}
    return factor_lab.submit(p.get("action", ""), p)
