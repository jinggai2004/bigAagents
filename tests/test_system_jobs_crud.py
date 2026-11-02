# -*- coding: utf-8 -*-
"""定时任务 CRUD（2026-11-01）：备注 / 类型 / 使用模型 可编辑。

背景：`core/scheduled_jobs.py` 的后端（表 + 校验 + 热加载）早就写完了，但
web 层**零引用** ⇒ 页面上只有三列只读信息，用户看不到也改不了备注/类型/模型
（项目铁律"函数写好 ≠ 页面看得见"）。本次把 API 接上并补 `model` 字段。

本文件钉住的都是"会静默出错"的点：
1. 内置任务覆盖必须**局部更新**（改开关不许抹掉备注）；
2. 模型名必须在厂商清单里（写错一个字母 = 该任务每天静默失败）；
3. 内置任务不能删（只能恢复默认），自定义任务能删；
4. 停用/恢复要能反映到活着的调度器上（热同步）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from db import store as S                          # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    import db.models as M
    monkeypatch.setattr(M, "DB_PATH", str(tmp_path / "t.db"))
    S.init_store()
    return tmp_path / "t.db"


# ---------------------------------------------------------------- 内置覆盖
def test_builtin_override_partial_update(tmp_db):
    """只改 enabled 时，note/model 必须保留（整行覆盖会"改开关顺手抹备注"）。"""
    S.set_builtin_override("daily_review", note="我加的备注", model="deepseek-v4-flash")
    S.set_builtin_override("daily_review", enabled=False)
    ov = S.get_builtin_override("daily_review")
    assert ov["enabled"] is False
    assert ov["note"] == "我加的备注"
    assert ov["model"] == "deepseek-v4-flash"


def test_builtin_override_absent_means_default(tmp_db):
    """没覆盖过 ⇒ 查不到行，调用方按"启用/无备注/跟随默认"处理。"""
    assert S.get_builtin_override("daily_review") == {}
    assert S.list_builtin_overrides() == {}


def test_builtin_override_delete_restores_default(tmp_db):
    S.set_builtin_override("daily_review", enabled=False, note="x", model="deepseek-chat")
    assert S.delete_builtin_override("daily_review") is True
    assert S.get_builtin_override("daily_review") == {}


# ---------------------------------------------------------------- 校验
def test_validate_rejects_unknown_model(tmp_db):
    from core.scheduled_jobs import validate_job
    with pytest.raises(ValueError) as e:
        validate_job({"name": "x", "mode": "alpha_decay",
                      "cron": {"hour": 17, "minute": 5}, "model": "no-such-model"})
    assert "不在任何厂商的模型清单里" in str(e.value)


def test_validate_accepts_known_model(tmp_db):
    from core.scheduled_jobs import validate_job
    job = validate_job({"name": "x", "mode": "alpha_decay",
                        "cron": {"hour": 17, "minute": 5},
                        "model": "deepseek-v4-flash", "note": "备注"})
    assert job["model"] == "deepseek-v4-flash"
    assert job["note"] == "备注"


def test_validate_empty_model_means_follow_default(tmp_db):
    from core.scheduled_jobs import validate_job
    job = validate_job({"name": "x", "mode": "alpha_decay",
                        "cron": {"hour": 17, "minute": 5}})
    assert job["model"] == ""


def test_validate_rejects_unknown_mode(tmp_db):
    from core.scheduled_jobs import validate_job
    with pytest.raises(ValueError):
        validate_job({"name": "x", "mode": "不存在的类型", "cron": {"hour": 17}})


# ---------------------------------------------------------------- 自定义任务
def test_custom_job_roundtrip_with_model(tmp_db):
    from core.scheduled_jobs import validate_job
    job = validate_job({"name": "我的任务", "mode": "alpha_decay",
                        "cron": {"hour": 17, "minute": 5},
                        "note": "盯衰减", "model": "qwen-max"})
    S.upsert_scheduled_job(job)
    got = S.get_scheduled_job(job["job_id"])
    assert got["model"] == "qwen-max"
    assert got["note"] == "盯衰减"
    assert got["enabled"] is True


def test_custom_job_delete(tmp_db):
    from core.scheduled_jobs import validate_job
    job = validate_job({"name": "待删", "mode": "alpha_decay",
                        "cron": {"hour": 18, "minute": 0}})
    S.upsert_scheduled_job(job)
    assert S.delete_scheduled_job(job["job_id"]) is True
    assert S.get_scheduled_job(job["job_id"]) == {}


def test_custom_job_toggle(tmp_db):
    from core.scheduled_jobs import validate_job
    job = validate_job({"name": "开关", "mode": "alpha_decay",
                        "cron": {"hour": 18, "minute": 0}})
    S.upsert_scheduled_job(job)
    assert S.set_scheduled_job_enabled(job["job_id"], False) is True
    assert S.get_scheduled_job(job["job_id"])["enabled"] is False


# ---------------------------------------------------------------- 执行链
def test_builtin_spec_passes_model_to_run_job():
    """`_builtin_job_spec` 必须把覆盖里的 model 带出去（否则等于没配）。"""
    from core.scheduler import _builtin_job_spec
    trigger, name, slot, model = _builtin_job_spec(
        "daily_review", {"hour": 15, "minute": 50}, "review", "每日复盘",
        {"note": "备注", "model": "deepseek-v4-pro"})
    assert model == "deepseek-v4-pro"
    assert "备注" in name
    assert slot == "15:50"


def test_builtin_spec_empty_model_is_none():
    from core.scheduler import _builtin_job_spec
    _, _, _, model = _builtin_job_spec(
        "daily_review", {"hour": 15, "minute": 50}, "review", "每日复盘", {})
    assert model is None


def test_job_kwargs_only_when_model_set():
    """自定义任务：空 model 不传 kwargs（保持与旧调用完全一致）。"""
    from core.scheduled_jobs import _job_kwargs
    assert _job_kwargs({"model": ""}) == {}
    assert _job_kwargs({}) == {}
    assert _job_kwargs({"model": "qwen-max"}) == {"model": "qwen-max"}


def test_sync_builtin_disables_and_restores(tmp_db):
    """停用 ⇒ 从调度器移除；删覆盖 ⇒ 重新挂上（热生效）。"""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from core.scheduler import sync_builtin_overrides, JOBS

    sch = BlockingScheduler()
    # 先用 create_scheduler 的注册函数挂一个（未启动的调度器也能挂）
    from core.scheduler import register_builtin_job
    spec = next(j for j in JOBS if j[0] == "daily_review")
    register_builtin_job(sch, spec[0], spec[1], spec[2], spec[3], {})
    assert "daily_review" in {j.id for j in sch.get_jobs()}

    S.set_builtin_override("daily_review", enabled=False)
    out = sync_builtin_overrides(sch)
    assert "daily_review" in out["disabled"]
    assert "daily_review" not in {j.id for j in sch.get_jobs()}

    S.delete_builtin_override("daily_review")
    out2 = sync_builtin_overrides(sch)
    assert "daily_review" in out2["restored"]
    assert "daily_review" in {j.id for j in sch.get_jobs()}
