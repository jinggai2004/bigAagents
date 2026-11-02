# -*- coding: utf-8 -*-
"""M4#2/#3：唤醒条件 owner 归属 + 交易动作推送。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from db import store as S                     # noqa: E402
from core import ai_routine as R              # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """隔离数据库（不碰真实 data/portfolio.db）。"""
    import db.models as M
    db = tmp_path / "t.db"
    monkeypatch.setattr(M, "DB_PATH", str(db))     # get_connection 在 models 里读它
    S.init_store()
    return db


def test_owner_default_self(tmp_db):
    """不传 owner ⇒ 落库 'self'（缺省语义）。"""
    cid = S.upsert_ai_wake_condition("ai_ashen", {
        "code": "600519", "kind": "price_below", "threshold": 28.0})
    rows = S.list_ai_wake_conditions(persona="ai_ashen")
    assert len(rows) == 1 and rows[0]["owner"] == "self"
    assert rows[0]["cond_id"] == cid


def test_owner_boss_round_and_filter(tmp_db):
    """值班轮（老板留言消费轮）挂的条件 owner='boss'；list 可按 owner 过滤。"""
    S.upsert_ai_wake_condition("ai_ashen", {
        "code": "000895", "kind": "price_below", "threshold": 28.0,
        "owner": "boss", "note": "老板留言：盯伊利 28 元下方"})
    S.upsert_ai_wake_condition("ai_ashen", {
        "code": "600519", "kind": "stop_loss", "threshold": 1180.0})  # 自挂风控
    boss = S.list_ai_wake_conditions(persona="ai_ashen", owner="boss")
    self_ = S.list_ai_wake_conditions(persona="ai_ashen", owner="self")
    assert [r["code"] for r in boss] == ["000895"]
    assert [r["code"] for r in self_] == ["600519"]
    # 不传 owner 不过滤（两条都在）
    assert len(S.list_ai_wake_conditions(persona="ai_ashen")) == 2


def test_owner_upgrade_on_update(tmp_db):
    """同一条件 UPDATE 换 owner ⇒ 生效（幂等键 (persona,code,kind) 复用行）。"""
    S.upsert_ai_wake_condition("ai_ashen", {
        "code": "000895", "kind": "price_below", "threshold": 28.0})
    S.upsert_ai_wake_condition("ai_ashen", {
        "code": "000895", "kind": "price_below", "threshold": 27.0,
        "owner": "boss"})
    rows = S.list_ai_wake_conditions(persona="ai_ashen")
    assert len(rows) == 1                                  # 复用行，不新增
    assert rows[0]["owner"] == "boss" and rows[0]["threshold"] == 27.0


def test_push_trades_no_webhook_is_silent(tmp_db, monkeypatch):
    """未配置 webhook ⇒ _push_trades 安全跳过（不抛异常、不影响主流程）。"""
    from config.settings import config as cfg
    monkeypatch.setattr(cfg, "DINGTALK_WEBHOOK", "", raising=False)
    monkeypatch.setattr(cfg, "FEISHU_WEBHOOK", "", raising=False)
    out = [{"code": "600519", "action": "buy", "qty": 100, "order_id": "O1",
            "status": "submitted", "name": "贵州茅台", "reason": "x"},
           {"code": "600001", "action": "buy", "skipped": "取不到现价"}]  # 被拒不推
    R._push_trades(out, "2026-10-31")                      # 不应抛


def test_push_trades_ignores_rejected(tmp_db, monkeypatch):
    """被拒/跳过的动作（无 order_id）不进推送名单 ⇒ 不写推送留痕。"""
    called = {}
    import core.push as P
    def fake_push(text, title=None, dedupe_key=None):
        called["text"] = text
        called["key"] = dedupe_key
        return {"提示": "未配置 webhook，内容仅保存本地"}
    monkeypatch.setattr(P, "push_markdown", fake_push)
    monkeypatch.setattr("core.push.push_markdown", fake_push, raising=False)
    R._push_trades([{"code": "600519", "action": "buy", "qty": 100,
                     "order_id": "O1", "status": "submitted", "name": "茅台"}],
                   "2026-10-31")
    assert "key" in called
    assert called["key"].startswith("trade:ashen:2026-10-31:")
    assert "600519" in called["key"] and "buy" in called["key"]
    assert "茅台" in called["text"]
