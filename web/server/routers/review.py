# -*- coding: utf-8 -*-
"""复盘：正文 / 存档 / 今日决策。"""
from __future__ import annotations

from fastapi import APIRouter

from web.server import review

router = APIRouter(tags=["review"])


@router.get("/api/review/latest")
def api_review_latest() -> dict:
    return review.latest()


@router.get("/api/review/archive")
def api_review_archive(date: str | None = None) -> dict:
    return review.archive(date=date)


@router.get("/api/review/digest")
def api_review_digest() -> dict:
    return review.digest()
