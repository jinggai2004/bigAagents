# -*- coding: utf-8 -*-
"""FastAPI 应用装配（P3：路由拆到 `web/server/routers/`）。

启动：`python scripts/serve_web.py`  →  http://127.0.0.1:8765/
API 文档：/api/docs

本文件只做三件事：建 app、挂路由、挂静态资源。**不写业务**——
取数/口径/错误分类在各领域模块（picks/watch/intel/review/desk/research）。
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from web.server import settings, warmup
from web.server.routers import (analysis, chat, desk, intel, meta, picks,
                                research, review, system, watch)

_logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # 预热由环境变量开关（`scripts/serve_web.py` / exe 启动器置 1）。
    # 测试不置 ⇒ 这里什么都不做，`TestNoOutbound` 的"端点零外呼"才稳定。
    if warmup.start():
        _logger.info("[serve] 已排入首屏预热")
    yield


app = FastAPI(
    title="bigAagents API",
    version=settings.SERVER_VERSION,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=_lifespan,
)

# /api/* 必须先注册，StaticFiles 只接剩下的路径
for _r in (meta.router, picks.router, watch.router, intel.router,
           review.router, desk.router, research.router, analysis.router,
           system.router, chat.router):
    app.include_router(_r)


def _mount_ui() -> None:
    import os

    # 字体与共享脚本：必须在 `/` 之前挂，否则会被下面的 ui 挂载吞掉
    sd = settings.static_dir()
    if os.path.isdir(sd):
        app.mount("/static", StaticFiles(directory=sd), name="static")
    else:
        _logger.warning("[serve] static 目录不存在，字体与共享脚本不可用: %s", sd)

    # 阿深桌宠：独立页面 + 自带 three.js，**必须在 `/` 之前挂**（否则被兜底吞掉）。
    # 挂上之后它的对话才连得上同源的 /api/*（file:// 直接打开只会有"连不上 API"）。
    cd = settings.companion_dir()
    if os.path.isdir(cd):
        app.mount("/companion", StaticFiles(directory=cd, html=True), name="companion")
    else:
        _logger.info("[serve] companion 目录不存在，阿深桌宠不挂载: %s", cd)

    d = settings.templates_dir()
    if os.path.isdir(d):
        # 必须最后挂：先注册的 /api/* 优先，StaticFiles 只接剩下的路径
        app.mount("/", StaticFiles(directory=d, html=True), name="ui")
    else:
        _logger.warning("[serve] 模板目录不存在，UI 不挂载: %s", d)


_mount_ui()
