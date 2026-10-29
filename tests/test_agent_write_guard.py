# -*- coding: utf-8 -*-
"""配置写守护测试（2026-10-24，P0 整改）——preview → commit → 审计 → restore。

守护的验收线（质量复核 P0-1）不是"测试全绿"，而是这几条**安全契约**：
    ① 无 change_id 不得产生任何写（preview 是纯校验）；
    ② change_id 一次性、15 分钟过期、不可枚举（复用/过期/伪造都被拒）；
    ③ 审计行含前后**完整**快照 + 可序列化 revert 描述符；
    ④ restore 能按快照撤销 delete/toggle/save。
DB 一律 use_template_db；watchlist 用 monkeypatch 指到 tmp（绝不碰真实 config/）。
"""

import json
import os

import pytest

from core import agent_write_guard as guard
from core.agent_tools import TOOL_FUNCS


@pytest.fixture()
def tpl_db(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "write_guard.db")
    from db import store
    return store


@pytest.fixture()
def wl_path(tmp_path, monkeypatch):
    import data.watchlist as wl
    p = str(tmp_path / "watchlist.json")
    monkeypatch.setattr(wl, "WATCHLIST_PATH", p)
    return p


# ================================================================
# ① 无确认不得写：preview 是纯校验，不落任何写
# ================================================================

def test_preview_writes_nothing(tpl_db):
    before = tpl_db.list_push_templates()
    r = TOOL_FUNCS["tool_push_templates"](
        action="save", key="alert", body="# {{title}}\n\n{{items}}\n\n{{footer}}")
    assert r.get("待确认") is True and r.get("change_id")
    assert tpl_db.list_push_templates() == before, "preview 不得写库"
    assert "改动前" in r and "改动后" in r, "必须带完整前后快照给模型复述"

    # 定时任务同理
    jobs_before = tpl_db.list_scheduled_jobs()
    p = TOOL_FUNCS["tool_scheduled_jobs"](
        action="create", name="守护测试", mode="watch", cron={"hour": 16, "minute": 40})
    assert p.get("待确认") is True
    assert tpl_db.list_scheduled_jobs() == jobs_before, "create preview 不得写库"


def test_commit_without_preview_rejected(tpl_db):
    r = TOOL_FUNCS["tool_push_templates"](action="commit", change_id="no-such-id")
    assert "error" in r
    r2 = TOOL_FUNCS["tool_scheduled_jobs"](action="commit", change_id="no-such-id")
    assert "error" in r2


# ================================================================
# ② change_id：一次性 / 过期 / 审计
# ================================================================

def test_change_id_single_use(tpl_db):
    p = TOOL_FUNCS["tool_push_templates"](
        action="save", key="alert", body="# {{title}}\n\n{{items}}\n\n{{footer}}")
    cid = p["change_id"]
    ok = TOOL_FUNCS["tool_push_templates"](action="commit", change_id=cid)
    assert ok.get("success") is True and ok.get("审计ID")
    again = TOOL_FUNCS["tool_push_templates"](action="commit", change_id=cid)
    assert "error" in again, "一次性：复用必须被拒"


def test_change_id_expiry(tpl_db):
    p = TOOL_FUNCS["tool_push_templates"](
        action="save", key="alert", body="# {{title}}\n\n{{items}}\n\n{{footer}}")
    cid = p["change_id"]
    # 直接让 pending 过期（ttl=-1 重新登记一个等价条目来模拟）
    guard._PENDING[cid]["expires_at"] = guard._PENDING[cid]["created_at"] - 1
    r = TOOL_FUNCS["tool_push_templates"](action="commit", change_id=cid)
    assert "error" in r and "过期" in r["error"]


def test_audit_row_has_full_snapshots(tpl_db):
    body = "# 新外壳 {{title}} / {{count}}"
    p = TOOL_FUNCS["tool_push_templates"](action="save", key="alert", body=body)
    cid = p["change_id"]
    TOOL_FUNCS["tool_push_templates"](action="commit", change_id=cid)
    row = tpl_db.get_agent_change(cid)
    assert row, "必须有审计行"
    assert row["tool"] == "tool_push_templates" and row["action"] == "save"
    old = json.loads(row["old_json"])
    new = json.loads(row["new_json"])
    assert "当前生效正文" in old, "旧快照必须完整（不是截断 200 字）"
    assert new["新正文"] == body
    assert json.loads(row["revert_json"])["kind"] == "template_save"


# ================================================================
# ③ restore：save / toggle / delete 均可回滚
# ================================================================

def test_restore_template_save(tpl_db):
    from core import push_templates as T
    original = T.get_template("alert")
    p = TOOL_FUNCS["tool_push_templates"](action="save", key="alert", body="# 改坏了 {{title}}")
    TOOL_FUNCS["tool_push_templates"](action="commit", change_id=p["change_id"])
    assert T.get_template("alert") == "# 改坏了 {{title}}"
    r = TOOL_FUNCS["tool_push_templates"](action="restore", change_id=p["change_id"])
    assert r.get("success") is True
    assert T.get_template("alert") == original, "restore 必须还原旧正文"


def test_restore_job_delete(tpl_db):
    tpl_db.upsert_scheduled_job({"job_id": "custom_del", "name": "将被删除", "mode": "watch",
                                 "cron": {"hour": 10, "minute": 10}})
    p = TOOL_FUNCS["tool_scheduled_jobs"](action="delete", job_id="custom_del")
    assert p.get("待确认") is True
    TOOL_FUNCS["tool_scheduled_jobs"](action="commit", change_id=p["change_id"])
    assert not tpl_db.get_scheduled_job("custom_del"), "commit 后应已删除"
    r = TOOL_FUNCS["tool_scheduled_jobs"](action="restore", change_id=p["change_id"])
    assert r.get("success") is True
    back = tpl_db.get_scheduled_job("custom_del")
    assert back and back.get("name") == "将被删除", "restore 必须把删除的任务插回来"


def test_restore_job_toggle(tpl_db):
    tpl_db.upsert_scheduled_job({"job_id": "custom_tg", "name": "开关", "mode": "watch",
                                 "cron": {"hour": 10, "minute": 20}})
    p = TOOL_FUNCS["tool_scheduled_jobs"](action="toggle", job_id="custom_tg", enabled=False)
    TOOL_FUNCS["tool_scheduled_jobs"](action="commit", change_id=p["change_id"])
    assert tpl_db.get_scheduled_job("custom_tg")["enabled"] is False
    r = TOOL_FUNCS["tool_scheduled_jobs"](action="restore", change_id=p["change_id"])
    assert r.get("success") is True
    assert tpl_db.get_scheduled_job("custom_tg")["enabled"] is True


# ================================================================
# ④ 任务 create/edit：校验走 validate_job 唯一入口
# ================================================================

def test_job_create_validates_mode(tpl_db):
    bad = TOOL_FUNCS["tool_scheduled_jobs"](
        action="create", name="X", mode="not_a_mode", cron={"hour": 9, "minute": 5})
    assert "error" in bad and "可选mode" in bad
    assert tpl_db.list_scheduled_jobs() == [], "校验失败不得落库"


def test_job_create_commit_and_conflict_id(tpl_db):
    p = TOOL_FUNCS["tool_scheduled_jobs"](
        action="create", name="盘后巡检", mode="watch", cron={"hour": 16, "minute": 50})
    assert p.get("待确认") is True
    job = p["改动后"]["任务"]
    assert job["job_id"].startswith("custom_")
    TOOL_FUNCS["tool_scheduled_jobs"](action="commit", change_id=p["change_id"])
    assert tpl_db.get_scheduled_job(job["job_id"]), "commit 后任务应存在"
    # 同名内置 id 必须拒绝
    clash = TOOL_FUNCS["tool_scheduled_jobs"](
        action="create", job_id="daily_review", name="顶包", mode="watch",
        cron={"hour": 17, "minute": 0})
    assert "error" in clash and "内置" in clash["error"]


def test_job_edit_changes_cron(tpl_db):
    tpl_db.upsert_scheduled_job({"job_id": "custom_ed", "name": "原任务", "mode": "watch",
                                 "cron": {"hour": 9, "minute": 0}})
    p = TOOL_FUNCS["tool_scheduled_jobs"](
        action="edit", job_id="custom_ed", cron={"hour": 15, "minute": 45})
    assert p.get("待确认") is True
    new_job = p["改动后"]["任务(修改后)"]
    assert new_job["cron"]["hour"] == 15 and new_job["cron"]["minute"] == 45
    old_job = p["改动前"]["任务(修改前)"]
    old_cron = old_job["cron"]
    if isinstance(old_cron, str):            # 快照保留 cron 原始形态（str/dict 都可能）
        old_cron = json.loads(old_cron)
    assert old_cron["hour"] == 9, "preview 必须带修改前快照"
    TOOL_FUNCS["tool_scheduled_jobs"](action="commit", change_id=p["change_id"])
    got = tpl_db.get_scheduled_job("custom_ed")
    assert got["cron"]["hour"] == 15 and got["cron"]["minute"] == 45


def test_job_edit_builtin_rejected(tpl_db):
    r = TOOL_FUNCS["tool_scheduled_jobs"](
        action="edit", job_id="daily_review", cron={"hour": 16, "minute": 0})
    assert "error" in r and "内置" in r["error"]


def test_toggle_enabled_string_rejected(tpl_db):
    tpl_db.upsert_scheduled_job({"job_id": "custom_st", "name": "S", "mode": "watch",
                                 "cron": {"hour": 11, "minute": 0}})
    r = TOOL_FUNCS["tool_scheduled_jobs"](action="toggle", job_id="custom_st", enabled="false")
    assert "error" in r, "字符串 'false' 必须拒（bool('false')==True 是已知坑）"
    assert tpl_db.get_scheduled_job("custom_st")["enabled"] is True, "不得被误写"


# ================================================================
# ⑤ watchlist：两阶段 + 损坏文件不得被写覆盖
# ================================================================

def test_watchlist_add_commit_restore(wl_path, tpl_db):
    """自选股写守卫的两阶段 + 回滚（2026-10-27 起落 SQLite，断言改为查库）。"""
    def codes():
        return [r["code"] for r in tpl_db.list_watchlist("user")]

    p = TOOL_FUNCS["tool_manage_watchlist"](action="add", code="601127", name="赛力斯")
    assert p.get("待确认") is True
    assert "601127" not in codes(), "preview 不得落库"
    ok = TOOL_FUNCS["tool_manage_watchlist"](action="commit", change_id=p["change_id"])
    assert ok.get("success") is True and ok.get("审计ID")
    assert "601127" in codes()
    r = TOOL_FUNCS["tool_manage_watchlist"](action="restore", change_id=p["change_id"])
    assert r.get("success") is True
    assert "601127" not in codes()


def test_watchlist_db_failure_blocks_write(wl_path, tpl_db, monkeypatch):
    """库读失败必须 fail-closed：**不得用默认池顶替去写**。

    JSON 时代的对应用例是"文件损坏 ⇒ 中止写入"，那个语义已随存储层消失；
    但纪律不变——读不到现状就绝不写，否则一次读失败会被放大成整份清单被覆盖。
    """
    def boom(owner="user"):
        raise RuntimeError("db down")

    monkeypatch.setattr(tpl_db, "list_watchlist", boom)
    r = TOOL_FUNCS["tool_manage_watchlist"](action="add", code="601127", name="赛力斯")
    assert "error" in r
    assert "读取失败" in r["error"] or "中止" in r["error"]


def test_watchlist_atomic_write_no_tmp_left(wl_path, tpl_db):
    p = TOOL_FUNCS["tool_manage_watchlist"](action="add", code="601127", name="赛力斯")
    TOOL_FUNCS["tool_manage_watchlist"](action="commit", change_id=p["change_id"])
    assert not os.path.exists(wl_path + ".tmp"), "原子写不得残留 .tmp"
