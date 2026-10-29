# -*- coding: utf-8 -*-
"""阶段依赖与数据完整性（P0-5）的契约。

锁住四件事：

1. **判据看产物不看日志**：`artifact_status` 只认"库里/磁盘上有没有"，不看进程；
2. **blocking 与 critical 分开**：`ai_mark_nav` 缺快照 ⇒ **跳过**（跑也白跑）；
   `daily_review` 缺快照 ⇒ **照跑但明写缺什么**（否则"快照一挂连复盘都没有"更糟）；
3. **完整性段由程序渲染**：缺失时正文必须出现"未评估行业资金热度"这类**具体影响**，
   不许用"数据异常"糊过去；
4. **接线真的接上了**：`run_job(..., job_id=)` 在 blocking 阶段上游缺失时
   **不调用**该阶段的实现（"验了函数" ≠ "验了接线"）。
"""
import os

import pytest

from core import stage_health as SH

DAY = "2026-10-16"


# ================================================================
# 1. 依赖表结构
# ================================================================

class TestStageDeps:
    def test_every_dep_has_an_artifact_rule(self):
        """⭐ 每个上游依赖都必须有**产物判据**，否则"就绪检查"会恒真（假安全）。"""
        known = {"snapshot_flow", "ai_mark_nav", "daily_review", "ai_daily_round",
                 "backfill_screen", "afternoon_full"}
        for jid, meta in SH.STAGE_DEPS.items():
            for dep in meta.get("deps") or []:
                assert dep in known, f"{jid} 的依赖 {dep} 没有产物判据"
                assert dep in SH.STAGE_DEPS, f"{jid} 的依赖 {dep} 未在 STAGE_DEPS 登记"

    def test_blocking_semantics(self):
        """⚠️ 关键语义：净值可阻断，复盘**不可**阻断。"""
        assert SH.STAGE_DEPS["ai_mark_nav"]["blocking"] is True
        assert SH.STAGE_DEPS["daily_review"]["blocking"] is False, \
            "复盘若也被阻断 ⇒ 快照一挂当天就没有复盘（比原来更糟）"

    def test_review_depends_on_screen_backfill(self):
        """⭐ D2（2026-10-18 审查）：复盘必须依赖**当天选股回填**。

        原来 JOBS 里复盘(15:45) 早于回填(15:50) ⇒"决策质量评估"永远用上一批样本，
        而模块 docstring 却把这条列成动机 —— 动机写进注释、问题没解决。
        """
        assert "backfill_screen" in SH.STAGE_DEPS["daily_review"]["deps"]
        assert "snapshot_flow" in SH.STAGE_DEPS["daily_review"]["deps"]

    def test_jobs_order_makes_review_see_fresh_backfill(self):
        """⭐ D2 的**时刻**判据：回填必须早于复盘（否则依赖登记是空话）。"""
        from core.scheduler import JOBS
        times = {jid: (kw.get("hour"), kw.get("minute")) for jid, kw, _m, _d in JOBS}
        bf = times["backfill_screen"]
        rv = times["daily_review"]
        assert bf < rv, f"回填 {bf} 必须早于复盘 {rv}"
        # 快照 15:40 跑约 9 分钟 ⇒ 复盘必须在 15:50 及之后，否则每天都会判"上游未就绪"
        assert rv[0] == 15 and rv[1] >= 50, f"复盘 {rv} 仍会与快照重叠（快照约 9 分钟）"


# ================================================================
# 2. 产物判据
# ================================================================

class TestArtifacts:
    def test_snapshot_present_and_absent(self, monkeypatch):
        monkeypatch.setattr(SH, "_snapshot_rows", lambda d: 3000)
        assert SH.artifact_status("snapshot_flow", DAY)["present"] is True
        monkeypatch.setattr(SH, "_snapshot_rows", lambda d: 0)
        st = SH.artifact_status("snapshot_flow", DAY)
        assert st["present"] is False and "0 行" in st["detail"]

    def test_nav_counts_rows(self, monkeypatch):
        monkeypatch.setattr(SH, "_nav_rows", lambda d: 6)
        assert SH.artifact_status("ai_mark_nav", DAY)["present"] is True
        monkeypatch.setattr(SH, "_nav_rows", lambda d: 0)
        assert SH.artifact_status("ai_mark_nav", DAY)["present"] is False

    def test_review_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(SH, "_PROJECT_ROOT", str(tmp_path))
        assert SH.artifact_status("daily_review", DAY)["present"] is False
        os.makedirs(tmp_path / "output" / "daily_reports", exist_ok=True)
        (tmp_path / "output" / "daily_reports" / "review_20261016.md").write_text(
            "x", encoding="utf-8")
        assert SH.artifact_status("daily_review", DAY)["present"] is True

    def test_unknown_stage_is_ready_by_default(self):
        """没登记的阶段不制造假依赖（否则每个新任务都要先补判据）。"""
        st = SH.artifact_status("morning_screen", DAY)
        assert st["present"] is True

    def test_afternoon_full_requires_today_date(self, tmp_path, monkeypatch):
        """⭐ R3（2026-10-18 审查）：复盘读 `daily_picks/latest.json`，
        判据必须是 **date == 今日**（昨天的文件不能当今天的流水线）。"""
        monkeypatch.setattr(SH, "_PROJECT_ROOT", str(tmp_path))
        d = tmp_path / "output" / "daily_picks"
        d.mkdir(parents=True)
        (d / "latest.json").write_text('{"date": "2026-10-15", "mode": "rule"}',
                                       encoding="utf-8")
        st = SH.artifact_status("afternoon_full", DAY)
        assert st["present"] is False and "2026-10-15" in st["detail"]
        (d / "latest.json").write_text('{"date": "2026-10-16", "mode": "rule"}',
                                       encoding="utf-8")
        st = SH.artifact_status("afternoon_full", DAY)
        assert st["present"] is True and "mode=rule" in st["detail"]

    def test_afternoon_full_rejects_morning_screen(self, tmp_path, monkeypatch):
        """⭐ F2（2026-10-18 审查）：09:25 盘前选股（mode=screen）也会把
        latest.json 的 date 刷成今天 ⇒ 只比 date 会把**上午的候选**当盘后流水线。"""
        monkeypatch.setattr(SH, "_PROJECT_ROOT", str(tmp_path))
        d = tmp_path / "output" / "daily_picks"
        d.mkdir(parents=True)
        (d / "latest.json").write_text('{"date": "2026-10-16", "mode": "screen"}',
                                       encoding="utf-8")
        st = SH.artifact_status("afternoon_full", DAY)
        assert st["present"] is False
        assert "mode='screen'" in st["detail"]

    def test_afternoon_full_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(SH, "_PROJECT_ROOT", str(tmp_path))
        st = SH.artifact_status("afternoon_full", DAY)
        assert st["present"] is False and "不存在" in st["detail"]

    def test_backfill_screen_counts_zero_is_not_missing_when_job_ran(self, monkeypatch):
        """⭐ R5（2026-10-18 审查）：**"跑了但 0 条到期" ≠ "没执行"**。

        回填的是 T+5/T+20 样本，节假日/样本未成熟时当日 0 条是正常情况；
        只看产物会让复盘天天挂一条假缺失。
        """
        monkeypatch.setattr(SH, "_screen_backfill_rows", lambda d: 0)
        monkeypatch.setattr(SH, "_job_ran_today",
                            lambda jid, d: {"ok": True, "note": "当日无到期样本"})
        st = SH.artifact_status("backfill_screen", DAY)
        assert st["present"] is True and "无到期样本" in st["detail"]

    def test_backfill_screen_zero_without_run_record_is_missing(self, monkeypatch):
        monkeypatch.setattr(SH, "_screen_backfill_rows", lambda d: 0)
        monkeypatch.setattr(SH, "_job_ran_today", lambda jid, d: None)
        assert SH.artifact_status("backfill_screen", DAY)["present"] is False


# ================================================================
# 3. 就绪判定
# ================================================================

class TestReadiness:
    def test_missing_upstream_reports_which(self, monkeypatch):
        monkeypatch.setattr(SH, "_snapshot_rows", lambda d: 0)
        rd = SH.stage_readiness("ai_mark_nav", DAY)
        assert rd["ready"] is False and rd["missing"] == ["snapshot_flow"]
        assert rd["blocking"] is True
        assert "market_snapshot" in rd["checks"]["snapshot_flow"]

    def test_ready_when_artifact_present(self, monkeypatch):
        monkeypatch.setattr(SH, "_snapshot_rows", lambda d: 2991)
        rd = SH.stage_readiness("ai_mark_nav", DAY)
        assert rd["ready"] is True and rd["missing"] == []

    def test_no_deps_stage_is_always_ready(self):
        rd = SH.stage_readiness("morning_screen", DAY)
        assert rd["ready"] is True and rd["checks"] == {}


# ================================================================
# 4. 完整性段（程序渲染）
# ================================================================

@pytest.fixture
def pipeline_ready(tmp_path, monkeypatch):
    """让 `afternoon_full` 判据看到"今日盘后流水线已跑"（mode=rule）。

    ⚠️ 不注入会让用例读**真实** `output/daily_picks/latest.json`：盘前 screen
    会把 date 刷成今天、mode=screen ⇒ 按 F2 判为缺失 ⇒ 用例失败（flaky，
    2026-10-19 实测：早上 09:25 选股跑过后本文件就开始红）。
    """
    import json as _json
    d = tmp_path / "output" / "daily_picks"
    d.mkdir(parents=True)
    # ⚠️ 必须用被测日期 DAY（不是"今天"）—— 用例断言的是 `data_integrity_markdown(DAY)`
    (d / "latest.json").write_text(_json.dumps({
        "date": DAY, "mode": "rule", "picks": []}, ensure_ascii=False),
        encoding="utf-8")
    monkeypatch.setattr(SH, "_PROJECT_ROOT", str(tmp_path))
    return tmp_path


class TestIntegrityMarkdown:
    def test_missing_snapshot_names_the_consequence(self, monkeypatch):
        """⭐⭐ 缺失时必须写**具体影响**（行业资金热度不评估），不能只写"数据异常"。"""
        monkeypatch.setattr(SH, "_snapshot_rows", lambda d: 0)
        monkeypatch.setattr(SH, "_nav_rows", lambda d: 6)
        monkeypatch.setattr(SH, "_ai_decision_rows", lambda d: 5)
        monkeypatch.setattr(SH, "_review_file", lambda d: "x.md")
        monkeypatch.setattr(SH, "_screen_backfill_rows", lambda d: 3)
        md = SH.data_integrity_markdown(DAY)
        assert "今日数据完整性" in md
        assert "未评估行业资金热度" in md
        assert "❌" in md

    def test_self_reference_excluded(self, monkeypatch, pipeline_ready):
        """⭐⭐ D1（2026-10-18 审查）：正文**不许**写「每日复盘 ❌ 缺失」。

        事故：`data_integrity_markdown()` 在写文件**之前**调用，而
        `artifact_status("daily_review")` 的判据正是"该文件是否存在" ⇒
        报告里必然出现"每日复盘缺失"，等于报告在说自己不存在。
        """
        for fn in ("_snapshot_rows", "_nav_rows", "_ai_decision_rows",
                   "_screen_backfill_rows"):
            monkeypatch.setattr(SH, fn, lambda d: 5)
        monkeypatch.setattr(SH, "_review_file", lambda d: "")   # 文件还没写（真实场景）
        md = SH.data_integrity_markdown(DAY)
        assert "| 每日复盘 |" not in md, "自指行必须排除（否则报告在说自己不存在）"
        assert "每日复盘" not in md.split("ℹ️")[-1] if "ℹ️" in md else True
        assert "❌" not in md, "本场景其余产物齐全 ⇒ 不该出现 ❌"

    def test_backfill_deadline_is_stated(self, monkeypatch):
        """D2：正文必须写明"决策质量评估用的是哪一批回填"（截止时间）。"""
        for fn in ("_snapshot_rows", "_nav_rows", "_ai_decision_rows"):
            monkeypatch.setattr(SH, fn, lambda d: 5)
        monkeypatch.setattr(SH, "_review_file", lambda d: "x.md")
        monkeypatch.setattr(SH, "_screen_backfill_rows", lambda d: 7)
        md = SH.data_integrity_markdown(DAY)
        assert "决策质量评估所用的选股回填" in md and "7 条" in md

    def test_backfill_deadline_text_follows_actual_schedule(self, monkeypatch):
        """时刻必须从 `JOBS` 现读，不能硬编码。

        回归动机（2026-10-28）：正文原来写死"回填任务排在 15:36"，回填先挪到 15:36、
        再挪到 15:10，文案一次都没跟着动 ⇒ 报告坚称"早于本报告"，实际每天都没跑完。
        这句话是"复盘为什么看不到当天回填"的唯一线索，写错等于把线索指向反方向。
        """
        from core import scheduler as SCH
        for fn in ("_snapshot_rows", "_nav_rows", "_ai_decision_rows"):
            monkeypatch.setattr(SH, fn, lambda d: 5)
        monkeypatch.setattr(SH, "_review_file", lambda d: "x.md")
        monkeypatch.setattr(SH, "_screen_backfill_rows", lambda d: 7)

        assert SH._job_time("backfill_screen") == "15:10"
        monkeypatch.setattr(SCH, "JOBS", [
            (jid, {**kw, "hour": 16, "minute": 20} if jid == "backfill_screen" else kw,
             mode, desc)
            for jid, kw, mode, desc in SCH.JOBS])
        md = SH.data_integrity_markdown(DAY)
        assert "16:20" in md, "正文没有跟着 JOBS 变（说明又退回了硬编码）"
        assert "15:10" not in md

    def test_all_present_says_ok(self, monkeypatch, pipeline_ready):
        monkeypatch.setattr(SH, "_snapshot_rows", lambda d: 2991)
        monkeypatch.setattr(SH, "_nav_rows", lambda d: 6)
        monkeypatch.setattr(SH, "_ai_decision_rows", lambda d: 5)
        monkeypatch.setattr(SH, "_review_file", lambda d: "x.md")
        monkeypatch.setattr(SH, "_screen_backfill_rows", lambda d: 9)
        md = SH.data_integrity_markdown(DAY)
        assert "关键阶段产物齐全" in md and "⚠️" not in md


# ================================================================
# 5. 接线：run_job 真的会跳过 blocking 阶段
# ================================================================

class TestRunJobWiring:
    def test_blocking_stage_skipped_when_upstream_missing(self, monkeypatch):
        """⭐⭐ 上游缺失 ⇒ **不调用**该阶段实现（这是"接线"而不是"函数"）。"""
        from core import scheduler as sch
        monkeypatch.setattr(SH, "stage_readiness",
                            lambda jid, d=None: {"ready": False,
                                                 "missing": ["snapshot_flow"],
                                                 "checks": {}, "blocking": True})
        called = []
        monkeypatch.setattr(sch, "run_ai_mark_nav_job",
                            lambda *a, **k: called.append(1))
        sch.run_job("ai_mark_nav", "测试", force=True, job_id="ai_mark_nav")
        assert called == [], "上游缺失且 blocking ⇒ 必须跳过"

    def test_non_blocking_stage_still_runs_with_warning(self, monkeypatch):
        """复盘缺快照**照样跑**（只警告）—— 否则当天连复盘都没有。"""
        from core import scheduler as sch
        monkeypatch.setattr(SH, "stage_readiness",
                            lambda jid, d=None: {"ready": False,
                                                 "missing": ["snapshot_flow"],
                                                 "checks": {}, "blocking": False})
        called = []
        monkeypatch.setattr(sch, "run_review_job", lambda *a, **k: called.append(1))
        sch.run_job("review", "测试复盘", force=True, job_id="daily_review")
        assert called == [1], "非 blocking 阶段必须继续执行"

    def test_no_job_id_skips_check_entirely(self, monkeypatch):
        """不传 job_id（自定义任务 / --once）⇒ 完全不做依赖检查，行为与旧版一致。"""
        from core import scheduler as sch

        def _boom(*a, **k):
            raise AssertionError("不该做依赖检查")

        monkeypatch.setattr(SH, "stage_readiness", _boom)
        called = []
        monkeypatch.setattr(sch, "run_risk_job", lambda *a, **k: called.append(1))
        sch.run_job("risk", "测试", force=True)
        assert called == [1]

    def test_scheduler_registers_job_id(self):
        """结构性守护：create_scheduler 必须把 job_id 传进 run_job（否则检查永不生效）。"""
        import io as _io
        src = _io.open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "core", "scheduler.py"), encoding="utf-8").read()
        assert "args=[mode, desc, None, False, job_id]" in src, \
            "create_scheduler 没把 job_id 传给 run_job ⇒ 依赖检查是死代码"


# ================================================================
# 3. AI 决策轮判据（2026-10-23 回归）
# ================================================================

class TestAiDecisionArtifact:
    """⭐ 事故：判据查询了**不存在的列** `trade_date` ⇒ 异常被吞 ⇒ 恒返 0。

    后果不是"少报"，而是**每天**在复盘"今日数据完整性"里写
    「AI 联赛决策轮 ❌ 缺失」，哪怕当天有 8 条决策 —— 与"缺就是缺"相反：
    把查询失败伪装成"没有数据"，读者会以为联赛没跑。
    """

    def test_counts_rows_by_signal_date(self, tmp_path, monkeypatch):
        from tests.conftest import use_template_db
        from db import store
        use_template_db(tmp_path, monkeypatch, "stage_ai.db")
        store.save_ai_decision({"persona": "ai_value", "signal_date": "2026-10-16",
                                "code": "-", "advice_status": "等待",
                                "position_size": 0.0})
        assert SH._ai_decision_rows("2026-10-16") == 1
        st = SH.artifact_status("ai_daily_round", "2026-10-16")
        assert st["present"] is True, st

    def test_empty_day_reports_absent(self, tmp_path, monkeypatch):
        """真的没有决策时才报缺失（判据要有区分力，不能恒 True）。"""
        from tests.conftest import use_template_db
        use_template_db(tmp_path, monkeypatch, "stage_ai2.db")
        assert SH._ai_decision_rows("2026-10-16") == 0
        assert SH.artifact_status("ai_daily_round", "2026-10-16")["present"] is False
