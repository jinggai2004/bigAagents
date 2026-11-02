# -*- coding: utf-8 -*-
"""排期冲突的**棘轮**守护（2026-10-23）。

## 为什么需要它
`core/scheduled_jobs.find_conflicts()` 早就写好了，测试也有——但都是**零散的 pair**
（15:50 / 17:30 / 15:59 / 15:43…）。结果是：**新增任务时不会撞到任何断言**。

实证（同一天里发生两次）：
- 10-18 把 `daily_review` 从 15:45 挪到 15:50，注释里写着"原来 15:45 必然与快照重叠"；
- 10-23 新增 `sentiment_ref` 又放回 **15:45**，直接踩回同一个坑。

而它的后果不是"慢一点"：复盘 15:50 要消费情绪数据，情绪采集被快照挤到 15:50 之后
⇒ 复盘会写"情绪缺失"——又是一个"看起来有、其实是缺"。

## 棘轮怎么转
现存的重叠（15:30~15:59 那一坨）是**既有设计**，不要求一次整改；
但**任何一个变化都必须显式登记**：新增冲突 ⇒ 红；旧冲突消失 ⇒ 也红（说明基线该收紧）。
`intraday_watch` 每分钟一跳，天然与所有任务重叠 ⇒ 两侧都排除（它不是"排期"问题）。
"""
import pytest

from core.scheduler import JOBS
from core import scheduled_jobs as SJ

DAY_TICK = "intraday_watch"      # 每分钟一跳，天然重叠，不参与棘轮

# 2026-11-01 重排后的基线：**只剩一处**，且是数据可用性决定的硬约束。
#
# 历史（为什么曾经有 7 条）：估算值长期低估 —— `screen` 写 12 分钟（实测 P90 53.5）、
# `rule` 写 23 分钟（实测 P90 57.4）。检测器按 23 分钟算，于是 15:50 的复盘看起来
# "排得开"，实际盘后全流程 P90 要跑到 16:28 ⇒ 15:40~16:30 的七个任务全部压在它身上。
# 2026-11-01 做三件事把它清掉：① 删掉重复的第三次全市场选股（noon_screen）；
# ② 回填挪出盘后窗口（backfill_screen→13:30、backfill_analysis→15:22）；
# ③ 下游一律排到 `afternoon_full` 的 **P90（15:30+58=16:28）之后**。
#
# ⚠️ 保留的这一条：`snapshot_flow`(15:40) 必须晚于"当日日线可用"（Tushare daily
#    收盘后约半小时才出，再早就只能拿昨天数据，10-29 实测踩过 TushareStaleData），
#    而 `afternoon_full`(15:30, P90 58 分钟) 又不能更早（它同样要当日收盘价）。
#    两者重叠约 10 分钟，都是 I/O 型、锁不同 ⇒ **接受并登记**，不再"挪时间"。
#    要消掉它只能给快照换数据源，属于另一件事。
KNOWN_CONFLICTS = {
    "afternoon_full": ["snapshot_flow"],
    "snapshot_flow": ["afternoon_full"],
}


def _cron_of(job_id: str) -> dict:
    for jid, kw, _mode, _desc in JOBS:
        if jid == job_id:
            return kw
    raise KeyError(job_id)


def _mode_of(job_id: str) -> str:
    for jid, _kw, mode, _desc in JOBS:
        if jid == job_id:
            return mode
    raise KeyError(job_id)


def _conflicts_of(job_id: str) -> list[str]:
    hits = SJ.find_conflicts(_cron_of(job_id), mode=_mode_of(job_id), exclude_id=job_id)
    return sorted(h["job_id"] for h in hits if h["job_id"] != DAY_TICK)


def _all_conflicts() -> dict[str, list[str]]:
    out = {}
    for jid, _kw, _mode, _desc in JOBS:
        if jid == DAY_TICK:
            continue
        c = _conflicts_of(jid)
        if c:
            out[jid] = c
    return out


class TestScheduleConflictRatchet:
    def test_no_new_conflicts(self):
        got = _all_conflicts()
        # 双向比较（2026-10-28 收紧）：原来只遍历 `got`，于是"某个任务从有冲突变成
        # **零冲突**"根本进不了 diff —— 基线里那条永远删不掉。`backfill_screen` 挪到
        # 15:10 后就是这样漏过一次：它的条目已失效，测试却照样绿。
        # 棘轮必须能往紧的方向转，否则只是"只增不减的登记表"。
        keys = sorted(set(got) | set(KNOWN_CONFLICTS))
        changed = {k: (KNOWN_CONFLICTS.get(k), got.get(k)) for k in keys
                   if KNOWN_CONFLICTS.get(k) != got.get(k)}
        assert not changed, (
            "排期冲突与登记基线不一致：\n"
            + "\n".join(f"  {k}: {old} → {now}" for k, (old, now) in changed.items())
            + "\n⇒ 新增任务请改时间（`find_conflicts` 可先算一遍）；"
              "确实要重叠则登记到 KNOWN_CONFLICTS 并写清理由；"
              "冲突消失则从基线里删掉对应条目。")

    def test_sentiment_ref_is_conflict_free(self):
        """⭐ 2026-10-23：它从 15:45 挪到 15:20 就是为了这条。"""
        assert _conflicts_of("sentiment_ref") == []

    def test_sentiment_ref_runs_before_daily_review(self):
        """硬约束：复盘（15:50）要消费情绪数据，采集必须更早。"""
        s, r = _cron_of("sentiment_ref"), _cron_of("daily_review")
        assert (s["hour"], s["minute"]) < (r["hour"], r["minute"]), \
            f"情绪采集 {s} 必须早于复盘 {r}，否则复盘永远读到前一天"

    def test_morning_screen_has_no_conflict(self):
        """⭐ 盘前选股必须**独占**它那一段，且真的在开盘前跑完。

        2026-10-24 的版本：选股 09:25 起、实测 696s ⇒ 跑到 09:36，而结算排在 09:32。
        2026-11-01 的版本更进一步：P90 实测 **53.5 分钟** ⇒ 09:25 起跑要到 10:18，
        所以不只是"别和结算撞"，而是**必须提前到开盘前**（现在 08:15，P90 也在 09:09 前）。
        """
        assert _conflicts_of("morning_screen") == []
        screen = _cron_of("morning_screen")
        settle = _cron_of("ai_settle_open")
        assert (settle["hour"], settle["minute"]) >= (9, 40), \
            f"开盘结算 {settle} 太早，会与选股的 P90 窗口重叠"
        # 硬约束：起跑时刻 + P90 必须落在 09:30 开盘之前
        from core.scheduled_jobs import JOB_MODES
        end_min = screen["hour"] * 60 + screen["minute"] + JOB_MODES["screen"]["minutes"]
        assert end_min <= 9 * 60 + 30, \
            f"盘前选股 P90 要到 {end_min // 60}:{end_min % 60:02d}，已经开盘了"

    def test_estimated_minutes_not_understated(self):
        """估算值必须 **≥ P90 实测**（估小 = 检测器失明）。

        数值来源：`logs/scheduler_*.log` 的 `═══ [任务] 完成: Ns` 行（2026-11-01 汇总）。
        ⚠️ 过去这里锁的是"单次实测"（696s / 1371.6s），于是随着样本变慢而失效：
        真实 P90 是 53.5 / 57.4 分钟，估算却停在 12 / 23 —— 检测器整整失明 34 分钟。
        """
        from core.scheduled_jobs import JOB_MODES
        # 盘前/午间选股：中位 24.9、P90 53.5（n=18）；盘后全流程：中位 21.6、P90 57.4（n=22）
        assert JOB_MODES["screen"]["minutes"] >= 54
        assert JOB_MODES["rule"]["minutes"] >= 58

    def test_ratchet_has_discriminance(self, monkeypatch):
        """守卫的鉴别力自检：伪造一个冲突必须被抓住。

        ⚠️ 2026-11-01：原来拿 `daily_review` 当靶子，重排后它已经零冲突
        （这正是重排的目标）⇒ 靶子换成仍然存在的那条 `afternoon_full`。
        """
        fake = dict(KNOWN_CONFLICTS)
        fake.pop("afternoon_full")
        monkeypatch.setattr("tests.test_schedule_conflicts.KNOWN_CONFLICTS", fake)
        got = _all_conflicts()
        new = {k: v for k, v in got.items() if fake.get(k) != v}
        assert "afternoon_full" in new
