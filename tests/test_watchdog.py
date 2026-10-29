# -*- coding: utf-8 -*-
"""调度器守护（scripts/watchdog.py）判定逻辑单测。零网络、零子进程。

设计意图：把"该不该重启"做成纯函数 `decide()`，这样这些分支才能在毫秒级被覆盖。
真去起进程测，既慢又会污染真实心跳，反而测不出问题。
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import watchdog as W  # noqa: E402

NOW = 1_800_000_000.0


def hb(age_s: float, pid: int = 4242) -> dict:
    return {"pid": pid, "epoch": NOW - age_s, "ts": "2026-10-20 10:00:00"}


# ================================================================
# 判定矩阵
# ================================================================

class TestDecide:

    def test_disabled_marker_wins_over_everything(self):
        """用户主动停止 ⇒ 绝不拉起（否则服务永远停不掉，比不装守护更糟）"""
        v = W.decide(hb(9999), NOW, disabled="2026-10-20 20:00 用户停止服务",
                     alive=lambda p: False)
        assert v["action"] == "none"
        assert "已禁用" in v["reason"]

    def test_missing_heartbeat_restarts(self):
        assert W.decide({}, NOW, alive=lambda p: False)["action"] == "restart"

    def test_fresh_heartbeat_does_nothing(self):
        v = W.decide(hb(30), NOW, alive=lambda p: True)
        assert v["action"] == "none"
        assert 25 < v["age"] < 35

    def test_stale_and_dead_restarts(self):
        v = W.decide(hb(900), NOW, alive=lambda p: False)
        assert v["action"] == "restart"
        assert v["age"] == pytest.approx(900)

    def test_stale_but_alive_only_alerts(self):
        """⭐ 进程还活着就不自动重启：可能正在跑 58 分钟全流程，杀掉会留半截产物"""
        v = W.decide(hb(900), NOW, alive=lambda p: True)
        assert v["action"] == "alert"
        assert "仍在" in v["reason"]

    def test_consecutive_limit_stops_restarting(self):
        """崩溃循环保护：连续 3 次重启仍不行 ⇒ 只告警，不再无限拉起"""
        v = W.decide(hb(900), NOW, state={"consecutive": 3}, alive=lambda p: False)
        assert v["action"] == "alert"
        assert "连续重启" in v["reason"]
        # 差一次就能继续重启
        assert W.decide(hb(900), NOW, state={"consecutive": 2},
                        alive=lambda p: False)["action"] == "restart"

    def test_missing_heartbeat_also_hits_consec_gate(self):
        """⭐ 审查抓到的真 bug（2026-10-20）：心跳不存在时**绕过**了连续失败闸门。

        第一版把 `if not epoch: return restart` 写在 `consec >= max_consec`
        之前 ⇒ 只要调度器写不出心跳（权限/磁盘/启动即崩），守护每 60 秒拉起
        一次、永不停止。这条用例就是防止顺序被改回去。
        """
        v = W.decide({}, NOW, state={"consecutive": 3}, alive=lambda p: False)
        assert v["action"] == "alert"
        assert "连续重启" in v["reason"]
        assert "心跳不存在" in v["reason"]      # 原因仍要写清楚

    def test_corrupt_epoch_also_hits_consec_gate(self):
        """同一个 bug 的第二条路径：时间戳解析失败也必须受闸门约束。"""
        v = W.decide({"pid": 1, "epoch": "abc"}, NOW,
                     state={"consecutive": 3}, alive=lambda p: False)
        assert v["action"] == "alert"

    def test_consec_gate_does_not_block_a_healthy_heartbeat(self):
        """反向保护：心跳正常时 consec 不该拦——它是"失败计数"，不是"永久禁令"。

        否则一旦计数到了 3，即使调度器已经活得好好的，也会被判 alert
        （tick 只在 action=="none" 时清零，形成死锁）。
        """
        v = W.decide(hb(30), NOW, state={"consecutive": 9}, alive=lambda p: True)
        assert v["action"] == "none"

    def test_corrupt_epoch_restarts(self):
        assert W.decide({"pid": 1, "epoch": "abc"}, NOW,
                        alive=lambda p: False)["action"] == "restart"

    def test_just_under_threshold_is_normal(self):
        """阈值边缘：299s 正常、301s 才动作（避免抖动式重启）"""
        assert W.decide(hb(299), NOW, alive=lambda p: True)["action"] == "none"
        assert W.decide(hb(301), NOW, alive=lambda p: False)["action"] == "restart"

    def test_negative_age_is_clamped(self):
        """系统时钟回拨导致 epoch > now ⇒ 不能得出"已停 负秒"这种荒谬结论"""
        v = W.decide({"pid": 1, "epoch": NOW + 500}, NOW, alive=lambda p: True)
        assert v["action"] == "none"
        assert v["age"] == 0


# ================================================================
# 进程存活判定（Windows 上不能用 os.kill 探活）
# ================================================================

class TestPidAlive:

    def test_self_is_alive(self):
        assert W.pid_alive(os.getpid()) is True

    def test_bogus_pid_is_dead(self):
        assert W.pid_alive(999_999_999) is False

    def test_garbage_input_is_dead(self):
        for bad in (None, "", "abc", 0, -1):
            assert W.pid_alive(bad) is False, bad


# ================================================================
# 禁用标记
# ================================================================

class TestDisabledMarker:

    def test_reads_reason(self, tmp_path, monkeypatch):
        p = tmp_path / ".scheduler_disabled"
        p.write_text("2026-10-20 20:00 用户停止服务\n", encoding="utf-8")
        monkeypatch.setattr(W, "DISABLED", p)
        assert "用户停止服务" in W.read_disabled()

    def test_absent_means_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(W, "DISABLED", tmp_path / "nope")
        assert W.read_disabled() is None

    def test_unreadable_marker_is_treated_as_disabled(self, tmp_path, monkeypatch):
        """标记读不了时倾向"禁用"：宁可不重启，也不要违背用户意图"""
        d = tmp_path / "isdir"
        d.mkdir()                                  # 目录当文件读 ⇒ OSError
        monkeypatch.setattr(W, "DISABLED", d)
        assert W.read_disabled() is not None


# ================================================================
# 守护自身的单实例
# ================================================================

class TestSingleton:
    """⭐ 2026-10-20 实测事故：连点两次「启动」⇒ 两个 watchdog 同时跑。

    后果不只是"多一个进程"：两边都判定 restart 时会**同时拉起调度器**、
    都往 runs 写事件、并且互相覆盖 `consecutive` 计数 —— 于是"连续 3 次
    失败才停手"永远数不到 3。
    """

    def test_blocks_when_another_live_watchdog(self, tmp_path, monkeypatch):
        pidf = tmp_path / ".watchdog.pid"
        pidf.write_text(str(os.getppid()), encoding="utf-8")   # 父进程一定活着
        monkeypatch.setattr(W, "PIDFILE", pidf)
        assert W.acquire_singleton() is False

    def test_allows_when_previous_pid_is_gone(self, tmp_path, monkeypatch):
        pidf = tmp_path / ".watchdog.pid"
        pidf.write_text("999999999", encoding="utf-8")          # 不存在的 pid
        monkeypatch.setattr(W, "PIDFILE", pidf)
        assert W.acquire_singleton() is True
        assert pidf.read_text(encoding="utf-8").strip() == str(os.getpid())

    def test_allows_on_first_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(W, "PIDFILE", tmp_path / "nope.pid")
        assert W.acquire_singleton() is True

    def test_release_only_removes_own_pidfile(self, tmp_path, monkeypatch):
        pidf = tmp_path / ".watchdog.pid"
        pidf.write_text("999999999", encoding="utf-8")
        monkeypatch.setattr(W, "PIDFILE", pidf)
        W.release_singleton()                       # 不是自己写的 ⇒ 不删
        assert pidf.exists()
        pidf.write_text(str(os.getpid()), encoding="utf-8")
        W.release_singleton()
        assert not pidf.exists()
