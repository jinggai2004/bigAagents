# -*- coding: utf-8 -*-
"""`scripts/run_scheduler.py --job` 手动重放（P0-6）的契约。

为什么必须有它：原来只有 `--once`（跑整个盘后全流程）⇒ 15:40 快照挂了、
15:45 复盘崩了，当天**没有第二次机会**，回填还要等下一个交易日。

锁住的边界：
  1. `--list-jobs` 列出全部内置 id（重放的前提是知道 id）；
  2. 未知 id ⇒ 退出码 2（不是"默默跑了个啥"）；
  3. `--job snapshot_flow --date D` ⇒ 调 `save_daily_market_snapshot(D)`（唯一支持显式日期）；
  4. 其他阶段忽略 `--date` 但**必须告警**（不假装支持）；
  5. `task_locks` 被占用 ⇒ 退出码 3（不与长驻调度进程叠跑）；
  6. `--force` 透传到 `run_job`（非交易日重放要人显式承担）。
"""
import importlib.util
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def _load_module():
    path = os.path.join(_ROOT, "scripts", "run_scheduler.py")
    spec = importlib.util.spec_from_file_location("run_scheduler_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sr(monkeypatch):
    """加载脚本模块并屏蔽 setup_logging（不写真日志文件）。"""
    mod = _load_module()
    monkeypatch.setattr(mod, "setup_logging", lambda: None)
    return mod


def _run(mod, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["run_scheduler.py"] + argv)
    return mod.main()


class TestListJobs:
    def test_lists_all_builtin_ids(self, sr, monkeypatch, capsys):
        assert _run(sr, monkeypatch, ["--list-jobs"]) == 0
        out = capsys.readouterr().out
        for jid in ("morning_screen", "snapshot_flow", "daily_review", "ai_daily_round"):
            assert jid in out


class TestUnknownJob:
    def test_returns_2_not_silent(self, sr, monkeypatch):
        assert _run(sr, monkeypatch, ["--job", "no_such_job"]) == 2


class TestReplay:
    def _stub_lock(self, monkeypatch, ok=True):
        calls = {"acquired": [], "released": []}
        import db.store as store

        # ⚠️⚠️ 必须同时屏蔽「常驻存活」检查（2026-10-23 实翻）：
        #    `main()` 在拿锁**之前**会读心跳，常驻调度进程活着就返回 4 并劝阻重放。
        #    原先只 stub 了锁 ⇒ 本机一旦开着调度器，5 个用例全红（assert 4 == 3）。
        #    这不是代码缺陷，是**测试与环境耦合**：断言"锁占用返回 3"的前提是
        #    "先过了常驻检查"。默认 stub 成"没有常驻进程在跑"，
        #    需要验证该分支的用例再单独覆盖（见 test_refuses_when_resident_alive）。
        import core.scheduler_health as health
        monkeypatch.setattr(health, "read_heartbeat", lambda *a, **k: None)

        def _acq(name, holder, ttl_seconds=0):
            calls["acquired"].append((name, holder))
            return ok

        def _rel(name, holder=None):
            calls["released"].append((name, holder))

        monkeypatch.setattr(store, "acquire_task_lock", _acq)
        monkeypatch.setattr(store, "release_task_lock", _rel)
        return calls

    def test_refuses_when_resident_alive(self, sr, monkeypatch):
        """常驻调度进程活着 ⇒ 不带 --force 时返回 4（不许悄悄并发同一阶段）。

        这条与 `_stub_lock` 的默认 stub 正好相反：显式证明那个分支**真的生效**，
        否则"屏蔽心跳"就成了让测试变绿的障眼法。
        """
        self._stub_lock(monkeypatch)
        import core.scheduler_health as health
        monkeypatch.setattr(health, "read_heartbeat",
                            lambda *a, **k: {"pid": os.getpid()})
        monkeypatch.setattr(health, "_process_alive", lambda pid: True)
        monkeypatch.setattr(sr, "run_job", lambda *a, **k: pytest.fail("不该执行"))
        assert _run(sr, monkeypatch, ["--job", "daily_review"]) == 4

    def test_snapshot_with_date_calls_snapshot_fn(self, sr, monkeypatch):
        self._stub_lock(monkeypatch)
        seen = {}
        import core.scheduler as sch

        def _fake_snap(d=None):
            seen["date"] = d
            return {"ok": True, "rows": 1}

        monkeypatch.setattr(sch, "save_daily_market_snapshot", _fake_snap)
        # run_job 不该被调（snapshot+date 走直通路径）
        monkeypatch.setattr(sr, "run_job",
                            lambda *a, **k: pytest.fail("snapshot+date 不该走 run_job"))
        rc = _run(sr, monkeypatch, ["--job", "snapshot_flow", "--date", "2026-10-16"])
        assert rc == 0 and seen["date"] == "2026-10-16"

    def test_other_jobs_go_through_run_job(self, sr, monkeypatch):
        self._stub_lock(monkeypatch)
        seen = {}

        def _fake_run_job(mode, desc, **kw):
            seen.update({"mode": mode, "desc": desc, "kw": kw})

        monkeypatch.setattr(sr, "run_job", _fake_run_job)
        rc = _run(sr, monkeypatch, ["--job", "daily_review"])
        assert rc == 0
        assert seen["mode"] == "review" and "daily_review" in seen["desc"]
        assert seen["kw"].get("force") is False

    def test_date_ignored_for_non_snapshot_but_warned(self, sr, monkeypatch, caplog):
        self._stub_lock(monkeypatch)
        monkeypatch.setattr(sr, "run_job", lambda *a, **k: None)
        import logging
        with caplog.at_level(logging.WARNING):
            rc = _run(sr, monkeypatch, ["--job", "daily_review", "--date", "2026-10-16"])
        assert rc == 0
        assert "--date 仅对 snapshot_flow 生效" in caplog.text, \
            "忽略 --date 必须告警（不许假装支持）"

    def test_force_is_passed_through(self, sr, monkeypatch):
        self._stub_lock(monkeypatch)
        seen = {}
        monkeypatch.setattr(sr, "run_job",
                            lambda mode, desc, **kw: seen.update(kw))
        _run(sr, monkeypatch, ["--job", "daily_review", "--force"])
        assert seen.get("force") is True

    def test_lock_busy_returns_3(self, sr, monkeypatch):
        calls = self._stub_lock(monkeypatch, ok=False)
        monkeypatch.setattr(sr, "run_job", lambda *a, **k: pytest.fail("锁占用不该执行"))
        assert _run(sr, monkeypatch, ["--job", "daily_review"]) == 3
        assert calls["released"] == [], "没拿到锁就不该释放"

    def test_lock_released_after_run(self, sr, monkeypatch):
        calls = self._stub_lock(monkeypatch)
        monkeypatch.setattr(sr, "run_job", lambda *a, **k: None)
        _run(sr, monkeypatch, ["--job", "risk_check"])
        assert len(calls["acquired"]) == 1 and len(calls["released"]) == 1
        assert calls["acquired"][0][0] == "manual_risk_check"

    def test_lock_released_even_if_job_raises(self, sr, monkeypatch):
        calls = self._stub_lock(monkeypatch)

        def _boom(*a, **k):
            raise RuntimeError("任务炸了")

        monkeypatch.setattr(sr, "run_job", _boom)
        with pytest.raises(RuntimeError):
            _run(sr, monkeypatch, ["--job", "risk_check"])
        assert len(calls["released"]) == 1, "异常路径也必须释放锁（否则阶段永久卡死）"
