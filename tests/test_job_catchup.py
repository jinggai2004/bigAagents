# -*- coding: utf-8 -*-
"""任务执行台账 + 启动补跑。

## 这个文件守的是 2026-10-30 那次"静默丢失"

「今天 18:10 的盘后深度分析（TOP8）没跑」——查下来是**撞上实例重启窗口**：
APScheduler 的 job 是内存态，重启后没有任何"今天这个槽位跑过没有"的记忆，
而 `misfire_grace_time=3600` 只在进程还活着时起作用。日志里连"跳过"都没有，
第二天看报告才发现。

补跑要考虑的不是"能不能跑"，而是"**该不该跑**"：
- 选股类补跑会产出非常规时点的 picks ⇒ 污染 T+N 回填分母；
- 结算类绑的是盘口时点，18:00 用收盘价重算开盘单是错的；
- 会推飞书的任务补跑会让用户以为"刚生成"。

所以判据里同时有**白名单机制**（`CATCHUP_SKIP_IDS`）和**时间窗**，
并且台账写在**任务入口**——被交易日检查跳过的槽位也要留痕，
否则每次启动都会把它当成"漏跑"。
"""
from datetime import datetime, timedelta

import pytest

from core import scheduler as sch


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """用模板库的副本，绝不碰真实库（本项目守则：测试不耦合真实环境）。"""
    from tests.conftest import use_template_db

    return use_template_db(tmp_path, monkeypatch, "catchup.db")


# ================================================================
# 1. 台账本身：幂等锚点
# ================================================================

def test_record_job_run_is_idempotent(temp_db):
    """同 (job_id, date, slot) 只留第一条 —— 这是"补跑不会重复跑"的全部依据。"""
    from db import store
    d, slot = "2026-10-30", "18:10"
    assert store.record_job_run("evening_analysis", d, slot, status="started") is True
    assert store.record_job_run("evening_analysis", d, slot, status="caught_up") is False
    assert store.has_job_run("evening_analysis", d, slot) is True
    rows = [r for r in store.list_job_runs(d) if r["job_id"] == "evening_analysis"]
    assert len(rows) == 1
    assert rows[0]["status"] == "started", "先到的那条不能被后来者覆盖"


def test_has_job_run_is_per_slot(temp_db):
    """不同槽位互不影响；不同日期互不影响。"""
    from db import store
    store.record_job_run("alpha_decay", "2026-10-30", "16:45")
    assert store.has_job_run("alpha_decay", "2026-10-30", "16:45") is True
    assert store.has_job_run("alpha_decay", "2026-10-30", "16:30") is False
    assert store.has_job_run("alpha_decay", "2026-10-31", "16:45") is False


def test_finish_job_run_writes_status_and_elapsed(temp_db):
    """台账**收尾**：`status` / `finished_at` 必须真的被写。

    2026-11-01 之前这两列是**死列**：`record_job_run` 只在入口写
    `status='started'` + `finished_at=NULL`，全仓没有任何 UPDATE ⇒ 台账永远显示
    "启动了但永不结束"，既答不出成功没有、也答不出跑了多久（实测 13/13 行 NULL）。
    """
    from db import store
    d, slot = "2026-11-01", "17:15"
    store.record_job_run("turn_share_sync", d, slot, status="started",
                         detail="换手率/流通股本日更")
    assert store.finish_job_run("turn_share_sync", d, slot, status="done",
                                elapsed=12.3) == 1
    row = [r for r in store.list_job_runs(d) if r["job_id"] == "turn_share_sync"][0]
    assert row["status"] == "done"
    assert row["finished_at"], "finished_at 没被写 ⇒ 这一列又变回死列"
    assert "12.3s" in row["detail"], "耗时必须落在 detail（不加列，见函数说明）"
    assert "换手率/流通股本日更" in row["detail"], "原有 job_desc 不能被耗时冲掉"


def test_finish_job_run_is_idempotent(temp_db):
    """收尾只认 `status='started'` 的那条 ⇒ 重复收尾是 0 行，耗时不会被追加两遍。"""
    from db import store
    d, slot = "2026-11-01", "17:15"
    store.record_job_run("turn_share_sync", d, slot, detail="日更")
    assert store.finish_job_run("turn_share_sync", d, slot, elapsed=1.0) == 1
    assert store.finish_job_run("turn_share_sync", d, slot, elapsed=99.0) == 0
    row = [r for r in store.list_job_runs(d) if r["job_id"] == "turn_share_sync"][0]
    assert row["detail"].count("s") < 4, f"耗时被追加了多次：{row['detail']}"


def test_run_job_finishes_ledger_on_success(temp_db, monkeypatch):
    """`run_job` 走完（正常返回）⇒ 台账被收尾成 done。

    这条守的是**接线**：`finish` 逻辑写在 `run_job` 的 `finally` 里，
    如果哪天有人把 `_run_job_impl` 的调用挪到 try 之外（或再包一层忘了转发），
    台账就会静默回到"永不结束"，而单看 `_run_job_impl` 的测试全是绿的。

    ⚠️ patch 的是**具体任务函数**而不是 `_run_job_impl`：台账入口写入就在
    `_run_job_impl` 开头，把它整体换掉等于连"入口记账"一起跳过（实测踩过，
    表现为 list_job_runs 里查不到任何东西）。
    `force=True` 绕过交易日闸门（本用例可能在周末跑），
    `stage_readiness` 打桩绕过上游依赖检查。
    """
    from db import store
    import core.stage_health as SH

    monkeypatch.setattr(SH, "stage_readiness",
                        lambda jid: {"ready": True, "blocking": False,
                                     "missing": [], "checks": {}})
    monkeypatch.setattr(sch, "run_review_job", lambda *a, **k: None)
    d = datetime.now().strftime("%Y-%m-%d")

    sch.run_job("review", "测试收尾", job_id="daily_review", slot="16:38", force=True)

    row = [r for r in store.list_job_runs(d) if r["job_id"] == "daily_review"][0]
    assert row["status"] == "done"
    assert row["finished_at"]


def test_run_job_marks_ledger_error_when_impl_raises(temp_db, monkeypatch):
    """任务函数抛异常 ⇒ 台账记 error，异常照常向上抛（不能被收尾吞掉）。"""
    from db import store
    import core.stage_health as SH

    monkeypatch.setattr(SH, "stage_readiness",
                        lambda jid: {"ready": True, "blocking": False,
                                     "missing": [], "checks": {}})

    def _boom(*_a, **_k):
        raise RuntimeError("模拟任务崩溃")

    monkeypatch.setattr(sch, "run_review_job", _boom)
    d = datetime.now().strftime("%Y-%m-%d")

    with pytest.raises(RuntimeError):
        sch.run_job("review", "测试异常", job_id="daily_review", slot="16:38",
                    force=True)

    row = [r for r in store.list_job_runs(d) if r["job_id"] == "daily_review"][0]
    assert row["status"] == "error"
    assert "RuntimeError" in (row["detail"] or "")


# ================================================================
# 2. 补跑判据
# ================================================================

def _slot_of(job_id: str) -> str:
    for jid, cron, _mode, _desc in sch.JOBS:
        if jid == job_id:
            return f"{int(cron['hour']):02d}:{int(cron.get('minute', 0)):02d}"
    raise AssertionError(f"JOBS 里没有 {job_id}")


def test_catchup_fires_for_missed_slot(temp_db, monkeypatch):
    """错过 3 小时内的白名单任务 ⇒ 被补跑一次。"""
    ran = []
    monkeypatch.setattr(sch, "run_job",
                        lambda mode, desc, job_id=None, **kw: ran.append(job_id))
    # ⚠️ 2026-11-01：原用 alpha_decay 当前置，它已改**周频（仅周五）**，
    #    拿它当"每天都该补"的样本会与星期判据冲突 ⇒ 换成每日跑的 backfill_screen。
    slot = _slot_of("backfill_screen")                 # 13:30
    now = datetime(2026, 10, 30, 13, 30) + timedelta(minutes=40)   # 10-30 是周五

    got = sch.catchup_missed_jobs(now=now, only={"backfill_screen"})
    assert got == ["backfill_screen"]
    assert ran == ["backfill_screen"]


def test_catchup_skips_when_already_recorded(temp_db, monkeypatch):
    """台账已有记录 ⇒ 不再补跑（**这条是防止重复推飞书/重复下单的关键**）。"""
    from db import store
    ran = []
    monkeypatch.setattr(sch, "run_job",
                        lambda mode, desc, job_id=None, **kw: ran.append(job_id))
    slot = _slot_of("backfill_screen")
    store.record_job_run("backfill_screen", "2026-10-30", slot, status="started")

    now = datetime(2026, 10, 30, 13, 30) + timedelta(minutes=40)
    got = sch.catchup_missed_jobs(now=now, only={"backfill_screen"})
    assert got == [] and ran == []


def test_catchup_respects_grace_window(temp_db, monkeypatch):
    """刚过排期不到 12 分钟 ⇒ 视为"正常在跑"，不补（否则会与正在跑的那次打架）。"""
    ran = []
    monkeypatch.setattr(sch, "run_job",
                        lambda mode, desc, job_id=None, **kw: ran.append(job_id))
    now = datetime(2026, 10, 30, 13, 30) + timedelta(minutes=3)
    assert sch.catchup_missed_jobs(now=now, only={"backfill_screen"}) == []
    assert ran == []


def test_catchup_respects_max_window(temp_db, monkeypatch):
    """错过超过 3 小时 ⇒ 不补（补跑一个 8 点的任务在下午没有意义）。"""
    ran = []
    monkeypatch.setattr(sch, "run_job",
                        lambda mode, desc, job_id=None, **kw: ran.append(job_id))
    now = datetime(2026, 10, 30, 13, 30) + timedelta(minutes=200)
    assert sch.catchup_missed_jobs(now=now, only={"backfill_screen"}) == []
    assert ran == []


def test_catchup_respects_weekly_day(temp_db, monkeypatch):
    """周频任务（周五的 alpha_decay）**只在周五**能被补跑。

    ⚠️ 2026-11-01 之前补跑只看 hour/minute、不看星期：alpha_decay 改周频后，
    周一到周四每次启动都会把"今天的 17:05"当成漏跑 ⇒ 一周白跑四遍，
    还会推四条重复的飞书告警（dedupe_key 按日期，四天四条）。
    """
    ran = []
    monkeypatch.setattr(sch, "run_job",
                        lambda mode, desc, job_id=None, **kw: ran.append(job_id))
    slot = _slot_of("alpha_decay")                     # 17:05，仅周五
    h, m = int(slot[:2]), int(slot[3:])

    monday = datetime(2026, 11, 2, h, m) + timedelta(minutes=40)     # 2026-11-02 周一
    assert sch.catchup_missed_jobs(now=monday, only={"alpha_decay"}) == []
    assert ran == []

    friday = datetime(2026, 10, 30, h, m) + timedelta(minutes=40)    # 2026-10-30 周五
    assert sch.catchup_missed_jobs(now=friday, only={"alpha_decay"}) == ["alpha_decay"]
    assert ran == ["alpha_decay"]


@pytest.mark.parametrize("job_id", sorted(sch.CATCHUP_SKIP_IDS))
def test_skip_list_never_caught_up(temp_db, monkeypatch, job_id):
    """**排除名单一个都不能被补跑**。

    每条都有具体理由（选股污染分母 / 结算绑盘口时点 / 决策轮 slot 是语义 /
    推送会让用户误解）。这条参数化测试保证"以后加进来的人不会悄悄放开"。
    """
    ran = []
    monkeypatch.setattr(sch, "run_job",
                        lambda mode, desc, job_id=None, **kw: ran.append(job_id))
    # 取一个"肯定落在窗口内"的时刻：全覆盖，逐个 job 试它自己的槽位
    for _jid, cron, _m, _d in sch.JOBS:
        if _jid != job_id:
            continue
        h, m = cron.get("hour"), cron.get("minute", 0)
        if not isinstance(h, int) or not isinstance(m, int):
            return                                     # 高频任务本就不在判据内
        now = datetime(2026, 10, 31, h, m) + timedelta(minutes=30)
        assert sch.catchup_missed_jobs(now=now, only={job_id}) == []
        assert ran == []


def test_high_frequency_jobs_excluded_by_type():
    """高频任务（hour/minute 是字符串）不该进补跑判据——它们没有单一槽位。"""
    hf = {jid for jid, cron, _m, _d in sch.JOBS
          if not isinstance(cron.get("hour"), int)}
    assert hf, "应该至少有一个高频任务（intraday_watch / hourly_news）"
    ran = []
    got = sch.catchup_missed_jobs(now=datetime(2026, 10, 31, 16, 45),
                                  only=hf)
    assert got == []
    assert ran == []


# ================================================================
# 3. 台账写在入口（不是出口）
# ================================================================

def test_run_job_records_at_entry_even_if_skipped(temp_db, monkeypatch):
    """被交易日检查跳过的槽位**也要留痕**。

    没留痕会怎样：每次启动都把它当成"漏跑"去补 ⇒ 周末每天补跑一遍工作日任务，
    日志里一堆"跳过"，而真正的漏跑被淹没。
    """
    from db import store
    monkeypatch.setattr(sch, "is_trading_day", lambda *a, **k: False)

    class _Sentinel(Exception):
        pass

    # 让"跳过"分支之后的路径显形：若真的往下走会抛
    monkeypatch.setattr(sch, "run_snapshot_job",
                        lambda *a, **k: (_ for _ in ()).throw(_Sentinel()))
    try:
        sch.run_job("snapshot", "全市场快照", job_id="snapshot_flow", slot="15:40")
    except _Sentinel:
        raise AssertionError("非交易日不应执行任务体")

    assert store.has_job_run("snapshot_flow", datetime.now().strftime("%Y-%m-%d"),
                             "15:40") is True


def test_run_job_without_slot_writes_nothing(temp_db):
    """手动/自定义任务不传 slot ⇒ 不记台账（否则会污染补跑判据）。"""
    from db import store
    before = len(store.list_job_runs(limit=500))
    sch.run_job("alpha_decay", "超额衰减监控(手动)")      # slot=None
    assert len(store.list_job_runs(limit=500)) == before
