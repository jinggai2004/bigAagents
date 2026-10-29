# -*- coding: utf-8 -*-
"""交易台：真持仓 / 模拟账户 / AI 联赛。"""
from __future__ import annotations

from fastapi import APIRouter

from web.server import desk

router = APIRouter(tags=["desk"])


@router.get("/api/desk/portfolio")
def api_desk_portfolio() -> dict:
    return desk.portfolio()


@router.get("/api/desk/sim")
def api_desk_sim(account: str | None = None) -> dict:
    return desk.sim(account=account)


@router.get("/api/desk/league")
def api_desk_league() -> dict:
    return desk.league()


@router.get("/api/desk/league/day")
def api_desk_league_day(date: str | None = None) -> dict:
    """某交易日的联赛决策明细 + 自述复盘（交易台「联赛每日决策」区）。

    ⚠️ 必须注册在 `/api/desk/league` 之后不影响匹配（两者都是字面量路径，无参数冲突），
    但**绝不能**写成 `/api/desk/league/{date}` —— 那样 `/api/desk/league/day` 会被
    路径参数吃掉，是项目里踩过的「字面量路由被参数路由静默吞掉」同型错误。
    """
    return desk.league_day(date=date)


@router.get("/api/desk/wake")
def api_desk_wake() -> dict:
    """AI 盘中唤醒：埋了哪些触发条件、真唤醒过几次、醒来做了什么。

    ⚠️ 字面量路径，必须与 `/api/desk/league/day` 一样避免任何同前缀的
    参数路由（本项目踩过"字面量被参数路由静默吞掉"）。
    """
    return desk.wake()


@router.post("/api/desk/sim/order")
def api_desk_sim_order(code: str, direction: str, quantity: int,
                       account: str | None = None, reason: str = "") -> dict:
    """模拟市价单。被拒业务（现金不足/无持仓）返 `ok:true, accepted:false`。"""
    return desk.place_order(account=account, code=code, direction=direction,
                            quantity=quantity, reason=reason)


@router.get("/api/desk/ashen")
def api_desk_ashen(date: str | None = None) -> dict:
    """阿深值班卡：授权参数 / 账户 / 两套自选 / 今日轮次（**零联网**）。

    `date` 只影响"看哪一天的轮次"，不影响账户与自选（那两者永远是当前的）。
    """
    return desk.ashen(day=date)


@router.post("/api/desk/ashen/wake")
def api_desk_ashen_wake(dry_run: bool = True) -> dict:
    """手动唤醒一轮。**默认 `dry_run=true`（只判定不落单）**。

    ⚠️ 这不是"重复注册的 `/api/desk/wake`"：`/api/desk/wake` 是**只读**面板
    （埋了哪些条件、命中过几次），这里才是**执行**。路径前缀相同但都是字面量，
    不存在"参数路由吃掉字面量"的问题。

    ⚠️ 真下单必须显式传 `dry_run=false`：把"按一下就可能建仓"做成默认
    等于删掉确认环节。
    """
    return desk.wake_ashen(dry_run=dry_run)


@router.post("/api/desk/ashen/chat")
def api_desk_ashen_chat(q: str) -> dict:
    """和阿深说一句话（第一人称回答，**只读**）。

    与 `/api/desk/ashen/wake` 的分工：wake 是"让它干活"（可能下单），chat 是
    "问它话"（绝不动作）。两者都是字面量路径，且**必须都注册在有任何
    `/api/desk/ashen/{x}` 参数路由之前**（本项目踩过字面量被参数路由静默吞掉）。

    用 query 参数而不是 body：项目内所有 POST 端点都是 query 风格
    （`/api/desk/sim/order` 亦然），另起一套会让前端要多写一种调用方式。
    问题长度在 `desk.ashen_chat` 里截到 600 字，URL 编码后仍远低于上限。
    """
    return desk.ashen_chat(question=q)
