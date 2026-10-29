# -*- coding: utf-8 -*-
"""多 AI 交易联赛 · 研究预算（C3，2026-10-14）。

「有限调用研究资源」的关键是**限的是代码，不是提示词**：
AI 想深入分析某只票必须"申请"，批不批由本模块的额度台账决定 ——
底层唯一实现是 `db.store.consume_ai_budget`（判定与扣减同一条 UPDATE，原子）。

⚠️ 两条红线（都有测试钉住）：
- **quota 不出现在 AI 可触达的接口上** —— AI 只能问"我还剩多少"，
  不能指定/修改自己的配额；那类请求一律拒绝（S3 工具层实现拒绝，本模块不提供入口）。
- 额度必须在**发请求之前**扣减 —— 先跑完再记账会出现"并发时超额"。
"""
from __future__ import annotations

import logging
from datetime import datetime

from db import store

_logger = logging.getLogger(__name__)

# 每日默认配额（决策 4 选 C：3 persona + 固定候选池 + 按需研究）。
# 1 次 deep_analysis ≈ 11 次 LLM 调用 / ≈4.7 万 token —— 这是最贵的一类，配额最小。
DEFAULT_QUOTA: dict[str, int] = {
    "deep_analysis": 3,       # 每天可发起的深入分析次数
    "wake": 5,                # 每天可被唤醒并跑决策轮的次数
    # 定时轮班（2026-10-27）：与 `wake` **分池**。
    # 理由不是"次数不够"（把 wake 调到 6 也能跑），而是**语义不同**：
    # `wake` = 条件被击中（"跌到 15.8 了，起来看一眼"），一次事件一次；
    # `routine` = 定时上班（09:40/10:20/…/14:40），与事件无关。
    # 共用一池的后果：阿深挂了一个 `price_below` 等价位，那一轮就少一次定时，
    # 于是"第 6 轮被挤掉"取决于当天有没有触发过条件——**故障随行情出现**，
    # 测试环境永远复现不了。
    "routine": 6,             # 每天定时轮班次数（与 SLOTS 长度对齐）
    # 每天决策轮次数：2026-10-23 由 1 → 2（盘后 15:58 + 午间 13:05）。
    # 实测 1 次/天时，3 个 LLM 人格 7 个交易日只有 3 笔成交（随机组 15 笔）——
    # 一天只有一次机会，模型说一次"等待"当天就归零，频次被配额而非判断力卡死。
    "decision_round": 2,
}

# persona 可在 config/ai_personas.json 里用 "quotas": {...} 覆盖个别 kind
# （例如风控流更谨慎：deep_analysis=1）。未覆盖的用默认值。


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def quota_for(persona: dict | str, kind: str) -> int:
    """某 persona 某类资源的当日配额。persona dict 里可选 `quotas` 覆盖。"""
    if kind not in DEFAULT_QUOTA:
        return 0                                # 未登记的 kind 一律 0 配额（拒绝）
    p = persona if isinstance(persona, dict) else None
    if p is None:
        from core import ai_personas
        p = ai_personas.get_persona(str(persona)) or {}
    override = (p.get("quotas") or {}) if isinstance(p, dict) else {}
    try:
        return int(override.get(kind, DEFAULT_QUOTA[kind]))
    except (TypeError, ValueError):
        return DEFAULT_QUOTA[kind]


def consume(persona: str, kind: str, *, n: int = 1,
            trade_date: str = None) -> tuple[bool, dict]:
    """申请并扣减额度。**唯一入口**（AI 工具层只能经这里）。

    Returns:
        (是否获批, 额度视图 {"used","quota","remaining"})
        拒绝不是异常 —— 调用方（工具层）要把"额度不足"如实转述给 AI。
    """
    if kind not in DEFAULT_QUOTA:
        # ⚠️ 未登记的 kind 直接拒绝：不能让"新加一种资源"绕过配额登记这一步
        _logger.warning("[AI预算] 未登记的资源类型 %r（persona=%s），拒绝", kind, persona)
        return False, {"used": 0, "quota": 0, "remaining": 0}
    day = trade_date or _today()
    quota = quota_for(persona, kind)
    granted, view = store.consume_ai_budget(persona, kind, n=n,
                                            trade_date=day, quota=quota)
    if granted:
        _logger.info("[AI预算] %s 扣 1 次 %s（%s/%s）", persona, kind,
                     view["used"], view["quota"])
    else:
        _logger.info("[AI预算] %s 的 %s 额度不足（%s/%s），拒绝", persona, kind,
                     view["used"], view["quota"])
    return granted, view


def view(persona: str, trade_date: str = None) -> dict:
    """当日额度视图。⚠️ **没用过的 kind 也必须出现**（以配额默认值计 0 用量）——
    底层 store.ai_budget_view 只含有记录的 kind，而「没出现」≠「额度 0」；
    AI 的 get_my_budget() 工具与 UI 都用本函数，看不到额度会让 AI 无法学会省着用。"""
    day = trade_date or _today()
    out: dict = {}
    for kind, default_q in DEFAULT_QUOTA.items():
        out[kind] = {"used": 0, "quota": quota_for(persona, kind),
                     "remaining": quota_for(persona, kind)}
    recorded = store.ai_budget_view(persona, day)
    for kind, v in recorded.items():
        if kind in out:                          # 未登记 kind 的历史行不外泄
            out[kind] = v
    return out
