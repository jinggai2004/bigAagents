# -*- coding: utf-8 -*-
"""服务自身：版本 / 启动时间 / 数据目录 / 是否离线 / 模板版本。"""
from __future__ import annotations

from fastapi import APIRouter

from web.server import envelope, settings

router = APIRouter(tags=["meta"])


@router.get("/api/meta")
def api_meta() -> dict:
    from data.offline_gate import is_offline

    return envelope.ok({
        "version": settings.SERVER_VERSION,
        "started_at": settings.STARTED_AT,
        "data_root": settings.data_root(),
        "offline": bool(is_offline()),
        "template_version": settings.template_version(),
    })
