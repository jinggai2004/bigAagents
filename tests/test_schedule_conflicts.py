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

# 2026-10-23 实测基线（挪走 sentiment_ref 之后）。新增任务必须为零冲突；
# 若确实要新增重叠，请在这里登记并写明理由。
# ⚠️⚠️ 2026-10-24：把 `JOB_MODES` 的估算值按**实测**改了（`screen` 6→12 分钟、
# `rule` 15→23 分钟），于是检测器第一次看见两处**本来就存在**的重叠：
#   ① `morning_screen`(09:25, 实测 695.9s) × `ai_settle_open`(09:32)  —— **已修**（挪到 09:45）；
#   ② `afternoon_full`(15:30, 实测 1371.6s) × `daily_review`(15:50) / `ai_mark_nav`(15:52)。
# ②就是 10-23 复盘"读到上一份 daily_picks"的直接原因（复盘 15:50 起跑、盘后流程 15:52 才完），
# 但**修它不能只挪时间**：15:30~16:00 这段已经排满，往下挪会连环撞
# （复盘 15:53 撞净值 15:52、挪到 15:56 撞新闻评估 15:55）。
# ⇒ 正解是 P1-3 的**提速**（选股 696s 压到 ≤240s，盘后流程整体前移），届时这里必须收紧。
# ⚠️ 登记 ≠ 认可：这条是"已知待整改"，不是"设计如此"。
# ✅ 2026-10-28：`backfill_screen` 从 15:36 挪到 **15:10**（原排期与复盘 15:50 撞车：
#    实测它跑 17.5 分钟、15:53:35 才结束，而复盘 15:50 就要读回填结果 ⇒ 永远读到
#    前一天的 T+N）。挪走后它与 `afternoon_full`(15:30)、`backfill_analysis` 都不再重叠。
KNOWN_CONFLICTS = {
    "afternoon_full": ["ai_mark_nav", "backfill_analysis",
                       "daily_review", "snapshot_flow"],
    "backfill_analysis": ["afternoon_full"],
    "snapshot_flow": ["afternoon_full"],
    "daily_review": ["afternoon_full", "ai_mark_nav"],
    "ai_mark_nav": ["afternoon_full", "daily_review", "news_review"],
    "news_review": ["ai_daily_round", "ai_mark_nav"],
    "ai_daily_round": ["news_review"],
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
        """⭐ 2026-10-24：盘前选股与开盘结算**每天重叠 5 分钟**（选股实测 696s，
        09:25 跑到 09:36:36，而结算排在 09:32）。修法是把结算挪到 09:45。

        这条用例的存在意义：当时检测器**看不见**它，因为
        `JOB_MODES["screen"]["minutes"]` 还写着 6（"实测 256 秒"是扩池前的数据）。
        """
        assert _conflicts_of("morning_screen") == []
        settle = _cron_of("ai_settle_open")
        assert (settle["hour"], settle["minute"]) >= (9, 40), \
            f"开盘结算 {settle} 太早，会与 09:25 起跑、约 09:37 结束的选股重叠"

    def test_estimated_minutes_not_understated(self):
        """估算值必须 **≥ 已知实测**（估小 = 检测器失明）。
        数值来自 `output/daily_picks/*.json` 的 `stage_timings` 与调度日志。"""
        from core.scheduled_jobs import JOB_MODES
        # screen.run = 695.87s ≈ 11.6min；afternoon_full = 1371.6s ≈ 22.9min
        assert JOB_MODES["screen"]["minutes"] >= 12
        assert JOB_MODES["rule"]["minutes"] >= 23

    def test_ratchet_has_discriminance(self, monkeypatch):
        """守卫的鉴别力自检：伪造一个冲突必须被抓住。"""
        fake = dict(KNOWN_CONFLICTS)
        fake.pop("daily_review")
        monkeypatch.setattr("tests.test_schedule_conflicts.KNOWN_CONFLICTS", fake)
        got = _all_conflicts()
        new = {k: v for k, v in got.items() if fake.get(k) != v}
        assert "daily_review" in new
