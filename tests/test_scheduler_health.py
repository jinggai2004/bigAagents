"""调度器健康检查单测（临时文件/临时库隔离）

覆盖动机（批次 16 的真实教训）：调度器**从上线起就没成功启动过**
（入口脚本在 start() 之前读 `Job.next_run_time` → AttributeError → 直接崩），
而**没有任何地方能看出来**：日志照样生成、"共 N 个任务"也打出来了、
UI 照常打开，只是所有定时任务一个都没跑，`analyses` / `screen_picks` 长期空表。

本文件锁死三件事：
1. `next_run_times` 在 start 前**不能抛异常**（这是当初的崩溃点）；
2. 健康检查能区分"在跑 / 进程没了 / 心跳过期"三种状态；
3. 进程探测在中文 Windows 上不因 `tasklist` 的 GBK 输出而崩；
4. **进程探测不拉起子进程**（2026-10-29：`tasklist` 每 10 秒闪一个黑框，
   改成原生 `OpenProcess`；降级路径也必须带 `CREATE_NO_WINDOW`）。
"""
import json
import os
from datetime import datetime

import pytest

from core import scheduler_health as health


@pytest.fixture
def tmp_heartbeat(tmp_path, monkeypatch):
    """心跳文件与数据库都指向临时位置"""
    hb = tmp_path / "heartbeat.json"
    monkeypatch.setattr(health, "HEARTBEAT_PATH", str(hb))
    from tests.conftest import use_template_db
    db_path = use_template_db(tmp_path, monkeypatch, "ui.db")
    return hb


class TestNextRunTimes:
    """当初的崩溃点：APScheduler 未启动时 Job 没有 next_run_time 属性"""

    def test_does_not_raise_before_start(self):
        from core.scheduler import create_scheduler, next_run_times, JOBS
        scheduler = create_scheduler()
        rows = next_run_times(scheduler)          # 曾经在这里 AttributeError
        assert len(rows) == len(JOBS)
        assert all(t == "(未启动)" for _, t in rows)

    def test_pending_times_are_computed_before_start(self):
        """启动前也要能告诉运维"几点跑"——这段日志曾因崩溃整体丢失"""
        from core.scheduler import create_scheduler, pending_run_times
        rows = dict(pending_run_times(create_scheduler()))
        assert "盘前选股" in rows
        assert "2026-" in rows["盘前选股"] or "20" in rows["盘前选股"]
        assert "(未启动)" not in rows["盘前选股"]

    def test_entry_script_calls_are_ordered_safely(self):
        """入口脚本里 next_run_times 必须只出现在 start() 之后"""
        import inspect
        import scripts.run_scheduler as entry
        src = inspect.getsource(entry.main)
        assert "pending_run_times" in src, "启动前应使用纯计算预估"
        assert src.index("pending_run_times") < src.index("scheduler.start()")

    def test_once_runs_the_full_pipeline_not_the_watch_job(self):
        """`--once` 必须跑盘后全流程(rule)，不能因为 JOBS 末尾追加了盯盘任务而跑错。

        曾在批次 16 收尾时发现：实现写的是 `mode = JOBS[-1][2]`，而 JOBS 末尾
        现在是"盘中异动预警"(mode=watch)——--once 会静默变成跑盯盘，不报错、
        也不产出全流程结果，调试时极难察觉。
        """
        import inspect
        import scripts.run_scheduler as entry
        src = inspect.getsource(entry.main)
        # 只看代码行（注释里为了说明原因会提到 JOBS[-1]，那是允许的）
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.strip().startswith("#"))
        assert "JOBS[-1]" not in code, "--once 不能靠 JOBS 末位取任务"
        assert "afternoon_full" in code, "--once 应精确指向盘后全流程"
        from core.scheduler import JOBS
        mode = next(m for jid, _k, m, _d in JOBS if jid == "afternoon_full")
        assert mode == "rule"


class TestHeartbeat:

    def test_write_read_roundtrip(self, tmp_heartbeat):
        assert health.write_heartbeat(pid=4242, jobs=8) is True
        hb = health.read_heartbeat()
        assert hb["pid"] == 4242 and hb["jobs"] == 8
        assert "epoch" in hb and "ts" in hb

    def test_clear_removes_file(self, tmp_heartbeat):
        health.write_heartbeat(pid=1)
        health.clear_heartbeat()
        assert health.read_heartbeat() == {}

    def test_clear_missing_file_is_noop(self, tmp_heartbeat):
        health.clear_heartbeat()                  # 不该抛

    def test_read_corrupted_returns_empty(self, tmp_heartbeat):
        tmp_heartbeat.write_text("{不是合法 JSON", encoding="utf-8")
        assert health.read_heartbeat() == {}


class TestProcessAlive:
    """中文 Windows 的 tasklist 输出是 GBK——按 utf-8 解会崩在子进程线程里"""

    def test_current_process_is_alive(self):
        assert health._process_alive(os.getpid()) is True

    def test_missing_pid_is_dead(self):
        assert health._process_alive(999999) is False

    def test_invalid_input_is_dead(self):
        assert health._process_alive(None) is False
        assert health._process_alive("abc") is False
        assert health._process_alive("") is False

    # ------------------------------------------------------------------
    # 2026-10-29 事故：判活每 10 秒拉起一次 tasklist，每次都闪一个黑框
    # ------------------------------------------------------------------
    # 成因链：system.html 每 10s 轮询 /api/system/health
    #        → system.status() → health.load_health()
    #        → TTL_HEALTH 正好也是 10s（缓存几乎每次都过期）
    #        → _process_alive() → subprocess.run(["tasklist"...])
    #        → 从 --noconsole 的 GUI 进程里起控制台程序且无 CREATE_NO_WINDOW
    #        ⇒ 用户侧表现："不定期会跳窗口"。
    # 修法：Windows 走原生 OpenProcess（不拉子进程）。下面两条是结构性守护。

    @pytest.mark.skipif(os.name != "nt", reason="仅 Windows 有原生判活路径")
    def test_does_not_spawn_any_subprocess(self, monkeypatch):
        """判活**不允许**拉起任何子进程 —— 拉了就可能在 GUI 进程里闪黑框。"""
        import subprocess

        def _boom(*a, **kw):                   # noqa: ANN001
            raise AssertionError("_process_alive 不得拉起子进程（会闪黑框）")

        monkeypatch.setattr(subprocess, "run", _boom)
        monkeypatch.setattr(subprocess, "Popen", _boom)
        assert health._process_alive(os.getpid()) is True
        assert health._process_alive(999999) is False

    @pytest.mark.skipif(os.name != "nt", reason="仅 Windows 有原生判活路径")
    def test_native_path_answers_without_fallback(self):
        """原生路径必须给出确定答案（True/False），不能返回 None 去降级 tasklist。"""
        assert health._win_pid_alive(os.getpid()) is True
        assert health._win_pid_alive(999999) is False

    @pytest.mark.skipif(os.name != "nt", reason="仅 Windows 有 CREATE_NO_WINDOW")
    def test_tasklist_fallback_is_windowless(self, monkeypatch):
        """降级路径也必须隐藏窗口 —— 否则等于把事故原样留在备份通道上。"""
        import subprocess
        seen = {}

        class _R:
            stdout = b""

        def _fake_run(cmd, **kw):               # noqa: ANN001
            seen["cmd"] = cmd
            seen["kw"] = kw
            return _R()

        monkeypatch.setattr(subprocess, "run", _fake_run)
        health._tasklist_alive(1234)
        assert "tasklist" in seen["cmd"][0]
        assert seen["kw"].get("creationflags", 0) & subprocess.CREATE_NO_WINDOW, \
            "tasklist 必须传 CREATE_NO_WINDOW，否则从 GUI 进程调用时会闪黑框"


class TestCheckHealth:

    def _hb(self, pid, epoch_offset=0):
        health.write_heartbeat(pid=pid)
        hb = health.read_heartbeat()
        hb["epoch"] = datetime.now().timestamp() - epoch_offset
        with open(health.HEARTBEAT_PATH, "w", encoding="utf-8") as f:
            json.dump(hb, f)

    def test_no_heartbeat_reports_not_running(self, tmp_heartbeat):
        h = health.check_health()
        assert h["scheduler"]["running"] is False
        assert "从未启动" in h["scheduler"]["reason"]
        assert any("调度器未运行" in w for w in h["warnings"])

    def test_running_when_process_alive_and_fresh(self, tmp_heartbeat):
        self._hb(os.getpid(), epoch_offset=1)
        h = health.check_health()
        assert h["scheduler"]["running"] is True
        assert h["scheduler"]["reason"] == "心跳正常"

    def test_dead_process_detected(self, tmp_heartbeat):
        self._hb(999999, epoch_offset=1)
        h = health.check_health()
        assert h["scheduler"]["running"] is False
        assert "进程已不存在" in h["scheduler"]["reason"]
        assert any("已退出" in w for w in h["warnings"])

    def test_stale_heartbeat_detected(self, tmp_heartbeat):
        """进程还在但心跳停了 → 卡死（不能报"运行中"）"""
        self._hb(os.getpid(), epoch_offset=health.HEARTBEAT_STALE_SECONDS + 100)
        h = health.check_health()
        assert h["scheduler"]["running"] is False
        assert "心跳已停" in h["scheduler"]["reason"]

    def test_data_freshness_included(self, tmp_heartbeat):
        h = health.check_health()
        assert "analyses" in h["data"] and "screen_picks" in h["data"]
        assert h["data"]["screen_picks"]["rows"] == 0

    def test_empty_tables_warn_only_when_scheduler_runs(self, tmp_heartbeat):
        """调度没跑时不该催"数据没长"——那是两码事，提示要分主次"""
        h = health.check_health()
        assert not any("仍为空" in w for w in h["warnings"])

        self._hb(os.getpid(), epoch_offset=1)
        h2 = health.check_health()
        assert any("screen_picks 表仍为空" in w for w in h2["warnings"])

    def test_format_report_renders(self, tmp_heartbeat):
        text = health.format_report()
        assert "调度器" in text and "数据新鲜度" in text
        assert "analyses" in text


class TestCodeFreshness:
    """"进程在跑"≠"跑的是最新代码"

    批次 22 收尾时的真实事故：调度进程 14:11 启动跑 9 个任务里的 8 个
    （新增的 snapshot_flow 不在里面），心跳正常、进程活着、jobs 也对不上——
    没有任何地方能看出来。所以心跳必须自带任务**清单**，健康检查负责比对。
    """

    def _hb(self, pid, job_ids=None, epoch_offset=0):
        health.write_heartbeat(pid=pid, job_ids=job_ids)
        hb = health.read_heartbeat()
        hb["epoch"] = datetime.now().timestamp() - epoch_offset
        with open(health.HEARTBEAT_PATH, "w", encoding="utf-8") as f:
            json.dump(hb, f)

    def test_heartbeat_carries_job_ids(self, tmp_heartbeat):
        self._hb(os.getpid(), job_ids=["morning_screen", "noon_screen"])
        hb = health.read_heartbeat()
        assert hb["job_ids"] == ["morning_screen", "noon_screen"]
        assert hb["jobs"] == 2          # 个数可由清单推导，不必单独维护

    def test_matching_job_list_is_not_stale(self, tmp_heartbeat):
        from core.scheduler import JOB_IDS
        self._hb(os.getpid(), job_ids=JOB_IDS, epoch_offset=1)
        # ⚠️ check_processes=False：本用例只验**心跳清单**这条通道。
        #    "长驻进程是不是旧代码"读的是本机真实进程（另一条通道，
        #    见 tests/test_process_freshness.py），不关掉会让本用例
        #    在"跑测试的机器上恰好有陈旧进程"时变红（2026-10-09 实测）。
        h = health.check_health(check_processes=False)
        assert h["code"]["checked"] is True
        assert h["code"]["missing"] == [] and h["code"]["extra"] == []
        assert h["code"]["stale"] is False
        assert h["scheduler"]["running"] is True
        assert not any("旧代码" in w for w in h["warnings"])

    def test_missing_job_is_reported_as_stale_code(self, tmp_heartbeat):
        """旧进程少一个任务 → 必须报"跑的是旧代码"，而不是只报"在跑" """
        from core.scheduler import JOB_IDS
        old = [j for j in JOB_IDS if j != "snapshot_flow"]
        self._hb(os.getpid(), job_ids=old, epoch_offset=1)
        h = health.check_health(check_processes=False)
        assert h["scheduler"]["running"] is True      # 进程确实活着
        assert h["code"]["stale"] is True
        assert h["code"]["missing"] == ["snapshot_flow"]
        assert any("旧代码" in w and "snapshot_flow" in w for w in h["warnings"])

    def test_legacy_heartbeat_without_job_ids_warns(self, tmp_heartbeat):
        """没有 job_ids 的老心跳不能假装"代码一致"，必须说清楚判不了"""
        self._hb(os.getpid(), job_ids=None, epoch_offset=1)
        h = health.check_health(check_processes=False)
        assert h["code"]["checked"] is False
        assert any("无法判断代码新旧" in w for w in h["warnings"])

    def test_extra_job_is_reported(self, tmp_heartbeat):
        from core.scheduler import JOB_IDS
        self._hb(os.getpid(), job_ids=JOB_IDS + ["ghost_job"], epoch_offset=1)
        h = health.check_health()
        assert h["code"]["extra"] == ["ghost_job"]
        assert any("不存在的任务" in w for w in h["warnings"])

    def test_report_shows_code_line(self, tmp_heartbeat):
        from core.scheduler import JOB_IDS
        self._hb(os.getpid(), job_ids=JOB_IDS, epoch_offset=1)
        assert "✅ 与当前代码一致" in health.format_report()

    def test_job_ids_match_jobs_definition(self):
        """JOB_IDS 必须与 JOBS 同源——写成两份就会静默漂移"""
        from core.scheduler import JOBS, JOB_IDS
        assert JOB_IDS == [job_id for job_id, _c, _m, _d in JOBS]

    def test_entry_script_writes_job_ids(self):
        """入口脚本必须传 job_ids：只传个数，健康检查就退化成看不清单"""
        import pathlib
        src = pathlib.Path("scripts/run_scheduler.py").read_text(encoding="utf-8")
        assert "job_ids=JOB_IDS" in src


# ================================================================
# 长驻进程陈旧：必须经 check_health 汇报出来（2026-10-09）
# ================================================================
# 实测背景：09:12 启动的 Streamlit 在 22:38 仍按 21:49 之前的旧代码产出报告，
# 导致"修复看起来没生效"，白排查一轮。判据逻辑见 core/process_freshness.py
# 与 tests/test_process_freshness.py；这里只钉"接进健康检查并能报警"。

class TestLongRunningProcessWarning:
    def test_stale_process_surfaces_as_warning(self, tmp_heartbeat, monkeypatch):
        from core.scheduler import JOB_IDS
        from core import process_freshness
        health.write_heartbeat(pid=os.getpid(), job_ids=JOB_IDS)   # 心跳新鲜
        monkeypatch.setattr(process_freshness, "check", lambda: {
            "checked": True, "reason": None, "newest_code": {"mtime": 1.0, "file": "core/x.py"},
            "processes": [{"pid": 999, "kind": "UI (streamlit)",
                           "started_at": "2026-10-08 09:12:59", "stale": True,
                           "newer_files": ["core/agents.py"]}],
            "stale": ["UI (streamlit)"]})
        h = health.check_health()
        assert h["long_running"]["stale"] == ["UI (streamlit)"]
        assert any("旧代码" in w and "UI" in w for w in h["warnings"]), h["warnings"]

    def test_disabled_check_is_not_reported_as_healthy(self, tmp_heartbeat, monkeypatch):
        """关掉检查时不能说"没发现问题" —— 必须说"没检查" """
        from core.scheduler import JOB_IDS
        health.write_heartbeat(pid=os.getpid(), job_ids=JOB_IDS)
        h = health.check_health(check_processes=False)
        assert h["long_running"]["checked"] is False
        assert h["long_running"]["reason"]
        assert "无法判断" in health.format_report(h)
