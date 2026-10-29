# -*- coding: utf-8 -*-
"""调度缺口检测的测试。

⚠️ 本模块**不读本机真实日志**：所有用例都用 `tmp_path` 下**构造的日志文件**，
加上注入的假 `JOBS`。理由见 MEMORY 铁律「测试不能把"本机状态"当契约」——
否则用例的成败取决于"跑测试那台机器今天跑得好不好"，这是最典型的假绿/假红。
"""
import os
from datetime import datetime

import pytest

from core import schedule_gap as SG

DAY = datetime(2026, 10, 14)          # 周三（cron 只排 mon-fri）


class _FakeStore:
    """替身：暴露 `JOBS`，形状与 `core.scheduler.JOBS` 一致。"""

    def __init__(self, jobs):
        self.JOBS = jobs


def _line(ts: str, desc: str) -> str:
    return (f'2026-10-14 {ts},005 [INFO] apscheduler.executors.default: '
            f'Running job "{desc} (trigger: cron[day_of_week=\'mon-fri\'], '
            f'next run at: 2026-10-15 {ts[:6]}00 CST)" (scheduled at 2026-10-14 {ts[:6]}00+08:00)')


def _header(ts: str = "09:00:00") -> str:
    """调度器启动行（不含 'Running job'）：用于确定 `started`，避免早于它的应跑时刻被误算。"""
    return f"2026-10-14 {ts},000 [INFO] core.scheduler: ═══ 调度器启动 ═══"


def _write_log(tmp_path, lines):
    p = tmp_path / "scheduler_20261014.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


# 每分钟任务，但只取 09:00-09:20 这 21 分钟，让用例足够小
_M_JOBS = [("watch", dict(hour=9, minute="0-20"), "watch", "盯盘")]
# 稀疏任务：15:35 一次
_S_JOBS = [("bf", dict(hour=15, minute=35), "bf", "回填")]


# ---------------------------------------------------------------- 基本行为
def test_log_missing_returns_unavailable(tmp_path):
    r = SG.analyse(date=DAY, log_path=str(tmp_path / "nope.log"), store=_FakeStore(_M_JOBS))
    assert r["available"] is False
    assert "不存在" in r["reason"]
    # ⚠️ 不可用也必须带齐键，否则调用方（format_report）会 KeyError
    assert r["jobs"] == {} and r["max_gap_minutes"] == 0 and r["warnings"] == []


def test_weekend_is_not_checked(tmp_path):
    sat = datetime(2026, 10, 17)
    p = _write_log(tmp_path, [_line("09:00:00", "盯盘")])
    r = SG.analyse(date=sat, log_path=p, store=_FakeStore(_M_JOBS))
    assert r["available"] is True and r["jobs"] == {} and r["warnings"] == []
    assert "非工作日" in r["reason"]


# ---------------------------------------------------------------- 密集任务
def test_missing_contiguous_minutes_grouped_into_one_gap(tmp_path):
    lines = [_line(f"09:{m:02d}:00", "盯盘") for m in list(range(0, 5)) + list(range(10, 21))]
    p = _write_log(tmp_path, lines)
    r = SG.analyse(date=DAY, log_path=p, store=_FakeStore(_M_JOBS),
                   now=datetime(2026, 10, 14, 10, 0))
    j = r["jobs"]["盯盘"]
    assert j["expected"] == 21 and j["missed"] == 5
    assert j["gaps"] == [{"start": "09:05", "end": "09:09", "minutes": 5}]
    assert r["max_gap_minutes"] == 5
    assert any("09:05–09:09" in w for w in r["warnings"])


def test_tiny_gap_below_threshold_is_not_reported(tmp_path):
    """偶发跳一分钟（任务耗时略超 60s）不该报警 —— 否则新规则自己就是噪声源。"""
    allm = list(range(0, 21))
    lines = [_line(f"09:{m:02d}:00", "盯盘") for m in allm if m not in (7, 8)]
    p = _write_log(tmp_path, lines)
    r = SG.analyse(date=DAY, log_path=p, store=_FakeStore(_M_JOBS),
                   now=datetime(2026, 10, 14, 10, 0))
    assert r["jobs"]["盯盘"]["missed"] == 2
    assert r["jobs"]["盯盘"]["gaps"] == []          # 2 < _MIN_GAP_REPORT(3)
    assert r["warnings"] == []


def test_runs_before_scheduler_start_are_not_counted_as_missing(tmp_path):
    """进程 09:10 才起来 → 09:00-09:09 本就不该跑，不能算缺口。"""
    lines = [_line("09:10:00", "盯盘")] + [_line(f"09:{m:02d}:00", "盯盘")
                                          for m in range(11, 21)]
    p = _write_log(tmp_path, lines)
    r = SG.analyse(date=DAY, log_path=p, store=_FakeStore(_M_JOBS),
                   now=datetime(2026, 10, 14, 10, 0))
    j = r["jobs"]["盯盘"]
    assert j["expected"] == 11 and j["missed"] == 0       # 只从 09:10 起算
    assert r["max_gap_minutes"] == 0


def test_future_expected_times_are_not_counted(tmp_path):
    """now 之后的应跑时刻还没到，不能算缺（否则每次检查都报一串未来缺口）。"""
    lines = [_line(f"09:{m:02d}:00", "盯盘") for m in range(0, 11)]
    p = _write_log(tmp_path, lines)
    r = SG.analyse(date=DAY, log_path=p, store=_FakeStore(_M_JOBS),
                   now=datetime(2026, 10, 14, 9, 10))
    assert r["jobs"]["盯盘"]["expected"] == 11 and r["jobs"]["盯盘"]["missed"] == 0


# ---------------------------------------------------------------- 稀疏任务
def test_sparse_job_missing_is_warned(tmp_path):
    p = _write_log(tmp_path, [_line("09:00:00", "盯盘")])       # 回填没跑
    r = SG.analyse(date=DAY, log_path=p, store=_FakeStore(_S_JOBS),
                   now=datetime(2026, 10, 14, 23, 0))
    assert r["jobs"]["回填"]["missed"] == 1
    assert any("回填" in w and "15:35" in w for w in r["warnings"])


def test_sparse_job_late_but_within_grace_is_covered(tmp_path):
    """14:50 的任务因机器睡眠到 15:04 才补跑 —— 算covered，但要标 late。

    ⚠️ 用例里**不能**放一条 14:50 的运行记录：那样就成了"准点跑过"，测不到 late。
    实测口径正是这样（`持仓风控巡检` 调度 14:50、当天只在该 15:04 跑过一次）。
    """
    jobs = [("r", dict(hour=14, minute=50), "risk", "巡检")]
    p = _write_log(tmp_path, [_header("09:00:00"), _line("15:04:00", "巡检")])
    r = SG.analyse(date=DAY, log_path=p, store=_FakeStore(jobs),
                   now=datetime(2026, 10, 14, 23, 0))
    j = r["jobs"]["巡检"]
    assert j["missed"] == 0 and j["late"] == ["14:50"]
    assert r["warnings"] == []


def test_sparse_job_beyond_grace_is_missed(tmp_path):
    """补跑延迟超过宽容度 → 仍是漏跑（10-14 的 14:05 新闻采集正是这种：
    15:04 才补跑，延迟 59 分钟 ≫ 宽容度 20 分钟）。"""
    jobs = [("n", dict(minute=5, hour=14), "news", "新闻")]
    p = _write_log(tmp_path, [_header("09:00:00"), _line("15:04:00", "新闻")])
    r = SG.analyse(date=DAY, log_path=p, store=_FakeStore(jobs),
                   now=datetime(2026, 10, 14, 23, 0))
    assert r["jobs"]["新闻"]["missed"] == 1


# ---------------------------------------------------------------- 结构性守护
def test_schedule_comes_from_JOBS_not_hardcoded():
    """⚠️ **必须从 `core.scheduler.JOBS` 反推**，不许硬编码时段。

    判据：两个完全不同的 cron 必须推出**不同的**期望时刻集合。
    若实现里写死了"09:00-12:00 每分钟"，第二个断言会失败 ——
    那意味着以后改 cron 时检测口径会**静默过时**，而没人会发现。
    """
    a = SG._expected_minutes(dict(hour="9-11,13-15", minute="*"), DAY)
    b = SG._expected_minutes(dict(hour=9, minute="0-20"), DAY)
    c = SG._expected_minutes(dict(hour=15, minute=35), DAY)
    assert len(a) == 360                       # 9-11 时(180) + 13-15 时(180)
    assert [t.strftime("%H:%M") for t in b] == [f"09:{m:02d}" for m in range(21)]
    assert [t.strftime("%H:%M") for t in c] == ["15:35"]


def test_expected_minutes_are_naive_local():
    """⚠️ 回归：CronTrigger 返回**带时区** datetime，日志是**本地朴素**时间。

    首版直接比较 → `TypeError: can't compare offset-naive and offset-aware`，
    整个检测静默返回 unavailable（错误被 except 兜住，最难发现）。
    这里钉住"出口必须是 naive"。
    """
    exp = SG._expected_minutes(dict(hour=9, minute=30), DAY)
    assert len(exp) == 1
    assert exp[0].tzinfo is None
    assert exp[0] == datetime(2026, 10, 14, 9, 30)


def test_never_raises_on_broken_jobs(tmp_path):
    """JOBS 形状坏掉 → 返回 unavailable，绝不抛（它不该拖垮健康检查）"""
    p = _write_log(tmp_path, [_line("09:00:00", "盯盘")])

    class _Bad:
        JOBS = [("bad", None, "x", "坏任务")]        # cron_kwargs 不是 dict

    r = SG.analyse(date=DAY, log_path=p, store=_Bad(), now=datetime(2026, 10, 14, 10, 0))
    assert r["available"] is False and "reason" in r


def test_job_name_is_truncated_before_trigger_text(tmp_path):
    """⚠️ 日志里 job 名后面紧跟着 ` (trigger: …, next run at: 2026-10-15 …)`。

    若正则把引号内容整段当名字，就会把**下一次运行的日期**一起吞进来 →
    名字永远匹配不上 JOBS 的 desc → 所有任务都"缺跑"（全线误报）。
    """
    p = _write_log(tmp_path, [_line("09:00:00", "盯盘")])
    info = SG._parse_log(p)
    assert list(info["runs"]) == ["盯盘"]


def test_describe_handles_unavailable():
    assert SG.describe({"available": False, "reason": "x"})
    assert SG.describe({"available": True, "jobs": {}, "reason": "非工作日"})


# ---------------------------------------------------------------- 健康检查接线
class TestHealthWiring:
    def test_check_health_includes_schedule_gap(self, monkeypatch):
        from core import scheduler_health as H
        sentinel = {"available": True, "max_gap_minutes": 89, "jobs": {},
                    "warnings": ["今日「盯盘」在 13:35–15:03 连续漏跑 89 分钟"]}
        monkeypatch.setattr(H, "_schedule_gap_report", lambda: sentinel)
        h = H.check_health(check_processes=False)
        assert h["schedule_gap"] is sentinel
        assert any("89 分钟" in w for w in h["warnings"])

    def test_check_health_can_skip_schedule_gap(self):
        from core import scheduler_health as H
        h = H.check_health(check_processes=False, check_schedule_gap=False)
        assert h["schedule_gap"]["available"] is False
        assert "未检查" in h["schedule_gap"]["reason"]

    def test_format_report_renders_gap_section(self):
        from core import scheduler_health as H
        h = H.check_health(check_processes=False, check_schedule_gap=False)
        txt = H.format_report(h)
        assert "调度缺口" in txt
