# -*- coding: utf-8 -*-
"""响应包络（文档 §4.1）。

- 顶层 `ok=false` **只表示服务自身故障**；数据不可用放在 `error.type` 的分区里。
- 分区包络（C2）：聚合端点的每个子区自带 `{ok, data, as_of, error}`，
  任一子区失败**不得**把顶层置 false —— 否则整页"不可用"。
- `error.type` 只能是下列之一；**绝不返回 0、绝不返空表冒充正常**。
"""
from __future__ import annotations

from datetime import datetime

ERROR_TYPES = (
    "timeout",        # 上游超时（akshare 无内置超时是既知坑）
    "auth",           # 鉴权失败（知乎 token 等）
    "proxy",          # 代理/网络层失败
    "empty",          # 文件存在但内容不可解析
    "stale",          # 数据过期（数据日期早于最近应收盘交易日）
    "missing",        # 数据文件不存在（P1 新增：与 empty 区分，"没跑"vs"坏了"）
    "offline",        # 主动离线（BIGA_OFFLINE=1）⇒ 不去等超时（P2 新增）
    "invalid",        # 请求参数非法（如未知行业名）——显式报错，不静默返回全量（P2 新增）
    "not_implemented",  # 该端点/子区 尚未接
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ok(data, as_of=None) -> dict:
    return {"ok": True, "data": data, "as_of": as_of, "fetched_at": _now()}


def fail(error_type: str, detail: str = "", as_of=None) -> dict:
    if error_type not in ERROR_TYPES:
        # 契约外的类型不静默放行：宁可显式错，也别让前端遇到未知枚举
        raise ValueError(f"未定义的 error.type: {error_type}")
    err = {"type": error_type}
    if detail:
        err["detail"] = detail
    return {"ok": False, "data": None, "as_of": as_of,
            "fetched_at": _now(), "error": err}


def not_implemented(what: str = "") -> dict:
    return fail("not_implemented", what)
