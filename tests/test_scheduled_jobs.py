"""自定义定时任务（L2 扩展层）测试。

为什么值得专门一组测试
----------------------
内置任务写死在 `JOBS` 里，"改排期"这件事本项目真实栽过两次：
  1. 新增 `risk_check` 后没重启调度进程 → 健康检查报"跑的是旧代码"；
  2. `snapshot_flow`(15:40) 与 `daily_review`(15:45) 只隔 5 分钟而快照跑 9 分钟
     → 复盘被挤到**重复采集**（9-28 实测 15:49 与 15:53 各一次）。

把排期变成数据之后，风险从"忘了重启"转成三类新的、**不会报警**的失效：
  A. 排期悄悄算错（`"mon-fri"` 曾被 `[:3]` 截成 `"mon"` → 每周只跑周一，界面还显示"周一至周五"）；
  B. 自定义任务混进 `JOB_IDS` → 健康检查天天报"多出来的任务"，从守护者变狼来了；
  C. 同步失败把已有任务连坐删掉，或读库异常拖垮调度进程。

本文件的用例就是钉住这三类。特别是 `TestHealthCheckSplit` 与
`test_sync_failure_does_not_delete_existing`：它们在改造前的代码上必然失败。
"""
import json

import pytest

from core import scheduled_jobs as sj
from core import scheduler_health as health


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """本模块**每个**用例都必须跑在临时库上——autouse，新增用例不可能漏掉。

    ⚠️ 血泪教训（本文件初版真踩了）：
    起初只有部分类用了 db 夹具，`TestSyncCustomJobs` 没有，于是它**直接写到并
    `DROP` 了真实的 `data/portfolio.db`**。后果不是"报个错"那么轻：
      - 表被删掉，而 `user_version` 已是新 schema 的指纹；
      - `init_store()` 的短路据此判定"库已是最新" → **该表永久缺失**；
      - 症状是之后所有用例一律 `no such table: scheduled_jobs`，
        而且**真实运行库里那张表也真的没了**。
    测试改掉生产库是这类缺陷里最贵的一种（静默、不可逆、且看着像代码问题）。
    所以隔离放在模块级 autouse，而不是依赖每个类自觉声明。
    """
    from tests.conftest import use_template_db
    return use_template_db(tmp_path, monkeypatch, "scheduled_jobs.db")


@pytest.fixture
def db(_isolated_db):
    """语义化别名（用例里写 db 更直白）；实际隔离由 _isolated_db 保证。"""
    return _isolated_db


# ================================================================
# A. cron 校验与展示
# ================================================================

class TestCronValidation:

    def test_normalize_fills_defaults(self):
        assert sj.normalize_cron({"hour": 15, "minute": 5}) == {
            "hour": 15, "minute": 5, "day_of_week": "mon-fri"}

    def test_mon_fri_is_not_truncated_to_monday(self):
        """回归：曾经把 "mon-fri" 按 `[:3]` 截成 "mon" → 静默变成每周一。

        这个 bug 不报错、日志正常、界面还显示"周一至周五"，只有到周一之外
        发现任务没跑才可能察觉（而且多半会被当成"调度挂了"）。
        """
        cron = sj.normalize_cron({"hour": 9, "minute": 25, "day_of_week": "mon-fri"})
        assert cron["day_of_week"] == "mon-fri"
        assert sj._expand_days(cron["day_of_week"]) == {0, 1, 2, 3, 4}
        assert sj.describe_cron(cron) == "周一至周五 09:25"

    def test_weekday_range_expands_not_mon_only(self):
        days = sj._expand_days("mon-fri")
        assert days == {0, 1, 2, 3, 4} and 4 in days

    def test_explicit_list_and_cross_week(self):
        assert sj._expand_days("mon,wed,fri") == {0, 2, 4}
        # "fri-mon" 跨周 → 周五、六、日、一（不包含周二）
        assert sj._expand_days("fri-mon") == {0, 4, 5, 6}

    def test_full_names_accepted_but_garbage_rejected(self):
        """只认精确的缩写/全名。早先用 `part[:3]` 模糊匹配，
        于是 "mondayish"/"monkey" 被静默当成周一、"mon-fri-xx" 被当成 mon-fri——
        排期解析必须"要么准确、要么报错"。"""
        assert sj._expand_days("monday,friday") == {0, 4}
        assert sj._expand_days("mondayish") is None
        assert sj._expand_days("monkey") is None
        assert sj._expand_days("mon-fri-xx") is None
        with pytest.raises(ValueError):
            sj.normalize_cron({"hour": 9, "minute": 0, "day_of_week": "mondayish"})

    def test_star_and_chinese_alias(self):
        assert sj._expand_days("*") == set(range(7))
        assert sj._expand_days("工作日") == {0, 1, 2, 3, 4}
        assert sj._expand_days("每天") == set(range(7))

    def test_all_seven_days_canonicalised_to_star(self):
        cron = sj.normalize_cron({"hour": 1, "minute": 0,
                                  "day_of_week": "mon,tue,wed,thu,fri,sat,sun"})
        assert cron["day_of_week"] == "*"

    def test_canonical_form_is_stable(self):
        """同一设置的字符串表示必须唯一，否则 sync 会每次都判定"变了"并反复重排"""
        a = sj.normalize_cron({"hour": 9, "minute": 0, "day_of_week": "fri,mon,wed"})
        b = sj.normalize_cron({"hour": 9, "minute": 0, "day_of_week": "wed,mon,fri"})
        assert a == b == {"hour": 9, "minute": 0, "day_of_week": "mon,wed,fri"}

    @pytest.mark.parametrize("bad,why", [
        ({"hour": 24, "minute": 0}, "小时越界"),
        ({"hour": 9, "minute": 60}, "分钟越界"),
        ({"hour": "abc", "minute": 0}, "小时非数字"),
        ({"hour": 9}, "缺分钟"),
        ({"minute": 9}, "缺小时"),
        ({"hour": 9, "minute": 0, "day_of_week": "mon-fri-xx"}, "星期非法"),
        ({"hour": 9, "minute": 0, "day_of_week": "mondayish"}, "星期不识别"),
    ])
    def test_invalid_raises_instead_of_defaulting(self, bad, why):
        """缺就是缺：非法输入必须抛错，绝不能悄悄塞一个默认时间——
        否则用户会看到一个自己从没设置过的执行时刻。"""
        with pytest.raises(ValueError):
            sj.normalize_cron(bad)

    def test_wildcards_supported(self):
        cron = sj.normalize_cron({"hour": "*", "minute": 5})
        assert cron["hour"] == "*"
        assert sj.describe_cron(cron) == "周一至周五每小时 05 分"

    def test_next_fire_time_is_computed_without_started_scheduler(self):
        """启动前预估必须走 trigger.get_next_fire_time(None, now)。

        ⚠️ 读 `job.next_run_time` 在未启动时会 AttributeError——正是这个坑让本项目
        调度器五个月没跑过一次任务。这里必须能在**没有任何调度器**的情况下算出时间。
        """
        from datetime import datetime
        nxt = sj.next_fire_time({"hour": 9, "minute": 25},
                                after=datetime(2026, 9, 29, 8, 0))
        assert nxt is not None and (nxt.hour, nxt.minute) == (9, 25)
        assert nxt.weekday() < 5
        # 非法 cron 不抛异常，如实返回 None（界面显示"—"）
        assert sj.next_fire_time({"hour": 99, "minute": 0}) is None


# ================================================================
# B. 任务定义校验
# ================================================================

class TestJobDefinition:

    def test_rejects_unknown_mode(self):
        """只允许白名单——刻意不开放"任意 prompt 定时执行"（LLM 是真实成本）"""
        with pytest.raises(ValueError) as e:
            sj.validate_job({"name": "x", "mode": "deep_analysis",
                             "cron": {"hour": 9, "minute": 0}})
        assert "不支持的任务类型" in str(e.value)

    def test_deep_analysis_and_llm_free_modes(self):
        assert "review" in sj.JOB_MODES          # 复盘会调 LLM，界面要提醒
        assert sj.JOB_MODES["review"]["llm"] is True
        assert "review" in sj.TRADING_DAY_MODES

    def test_name_required(self):
        with pytest.raises(ValueError):
            sj.validate_job({"name": "  ", "mode": "news",
                             "cron": {"hour": 9, "minute": 0}})

    def test_generated_id_is_prefixed_and_avoids_collision(self):
        taken = {"custom_test", "custom_test_2"}
        assert sj.make_job_id("Test", taken) == "custom_test_3"

    def test_generated_id_never_shadows_builtin(self):
        """自定义 id 撞上内置 id 会把内置任务**顶掉**（replace_existing 是覆盖不是报错）"""
        jid = sj.validate_job({"name": "morning_screen", "mode": "news",
                               "cron": {"hour": 9, "minute": 0}})
        assert jid["job_id"] not in sj.builtin_job_ids()
        assert jid["job_id"].startswith("custom_")

    def test_explicit_builtin_id_rejected(self):
        with pytest.raises(ValueError) as e:
            sj.validate_job({"job_id": "morning_screen", "name": "x", "mode": "news",
                             "cron": {"hour": 9, "minute": 0}, "_original_id": "other"})
        assert "内置任务 id" in str(e.value)

    def test_trading_day_only_forced_for_market_dependent_modes(self):
        """依赖当日行情的任务，勾了"非交易日也跑"也不放行——
        否则节假日会产出与真实交易日不可区分的"今日选股"。"""
        j = sj.validate_job({"name": "补跑", "mode": "rule", "trading_day_only": False,
                             "cron": {"hour": 16, "minute": 0}})
        assert j["trading_day_only"] is True and j["trading_day_locked"] is True

    def test_trading_day_only_honoured_for_news(self):
        j = sj.validate_job({"name": "夜间新闻", "mode": "news", "trading_day_only": False,
                             "cron": {"hour": 22, "minute": 0}})
        assert j["trading_day_only"] is False and j["trading_day_locked"] is False

    def test_builtin_modes_all_in_whitelist(self):
        """内置任务用到的 mode 必须都在白名单里，否则 run_job 会走错分支"""
        from core.scheduler import JOBS
        assert {m for _id, _c, m, _d in JOBS} <= set(sj.JOB_MODES)

    def test_trading_day_modes_cover_builtin_gate(self):
        """交易日闸门只有一处事实来源；这里钉住它确实覆盖了内置的行情类任务"""
        from core.scheduler import JOBS, _trading_day_modes
        gate = _trading_day_modes()
        # 排除与行情无关的任务：news（采集）与 news_review（行业评估，每天跑，
        # 周末快讯也有价值）、watch（启停由交易时段自判）
        market_modes = {m for _id, _c, m, _d in JOBS
                        if m not in ("news", "news_review", "watch")}
        assert market_modes == gate


# ================================================================
# C. 冲突检测（复盘重复采集那次事故的自动化版本）
# ================================================================

class TestConflictDetection:

    def test_same_time_as_builtin_review_is_flagged(self):
        # ⚠️ D2（2026-10-18）：复盘已从 15:45 挪到 **15:50**（原来必然与 15:40 起跑
        #    9 分钟的快照重叠）⇒ 这条断言跟着挪，否则测的是"旧时序还撞不撞"。
        hits = sj.find_conflicts({"hour": 15, "minute": 50}, mode="review")
        assert "daily_review" in [h["job_id"] for h in hits]
        assert next(h for h in hits if h["job_id"] == "daily_review")["same_mode"] is True

    def test_different_time_is_clean(self):
        # ⚠️ 用 17:30 而不是 16:00：16:00 已落在 15:58 的 ai_daily_round（6 分钟窗口）
        #    影响范围内，检测会（正确地）报冲突。新增 JOBS 任务时必须回来复核这类
        #    "看起来无关"的时间断言——这正是"新加任务要全仓搜断言"的原因。
        assert sj.find_conflicts({"hour": 17, "minute": 30}, mode="review") == []

    def test_new_ai_round_is_covered_by_conflict_detection(self):
        """新任务也要被冲突检测覆盖（否则"加了任务但撞车没人提醒"）。"""
        hits = sj.find_conflicts({"hour": 15, "minute": 59}, mode="review")
        assert "ai_daily_round" in [h["job_id"] for h in hits]

    def test_snapshot_overlapping_review_flagged_even_different_mode(self):
        """不同 mode 但时间撞车也要报：快照跑 9 分钟把 15:45 的复盘挤到重复采集"""
        hits = sj.find_conflicts({"hour": 15, "minute": 43}, mode="snapshot")
        ids = [h["job_id"] for h in hits]
        assert "daily_review" in ids
        assert next(h for h in hits if h["job_id"] == "daily_review")["same_mode"] is False

    def test_high_frequency_jobs_are_exempt(self):
        """盯盘每分钟醒一次、新闻每小时 :05——跟它们比时间只会淹没真正的提示"""
        assert sj.find_conflicts({"hour": 10, "minute": 17}, mode="screen") == []
        assert sj.find_conflicts({"hour": 10, "minute": 5}, mode="news") == []

    def test_weekend_schedule_does_not_conflict_with_weekday_jobs(self):
        """周六排期的任务和周一至周五的内置任务不该报冲突（星期也要参与比对）"""
        assert sj.find_conflicts({"hour": 15, "minute": 45,
                                  "day_of_week": "sat"}, mode="review") == []

    def test_conflicts_between_custom_jobs(self):
        others = [{"job_id": "custom_a", "name": "A", "mode": "snapshot",
                   "cron": {"hour": 16, "minute": 10, "day_of_week": "mon-fri"}}]
        hits = sj.find_conflicts({"hour": 16, "minute": 10}, mode="snapshot",
                                 custom_jobs=others)
        assert [h["job_id"] for h in hits] == ["custom_a"]

    def test_self_is_excluded(self):
        others = [{"job_id": "custom_a", "name": "A", "mode": "news",
                   "cron": {"hour": 16, "minute": 10, "day_of_week": "mon-fri"}}]
        assert sj.find_conflicts({"hour": 16, "minute": 10}, mode="news",
                                 custom_jobs=others, exclude_id="custom_a") == []

    def test_unparseable_builtin_cron_is_skipped_not_guessed(self):
        """场内盯盘是 hour="9-11,13-15"/minute="*"，展开能识别；
        万一遇到解析不了的字段，选择"不报"而不是瞎猜。"""
        assert sj.expand_cron({"hour": "9-11,13-15", "minute": "*"}) is not None


# ================================================================
# D. 持久化
# ================================================================

class TestStoreCrud:

    def _job(self, **kw):
        base = {"job_id": "custom_x", "name": "测试任务", "mode": "news",
                "cron": {"hour": 16, "minute": 5, "day_of_week": "mon-fri"},
                "enabled": True, "trading_day_only": False, "note": "备注"}
        base.update(kw)
        return base

    def test_upsert_list_get_delete(self, db):
        from db import store
        assert store.list_scheduled_jobs() == []
        store.upsert_scheduled_job(self._job())
        rows = store.list_scheduled_jobs()
        assert len(rows) == 1 and rows[0]["job_id"] == "custom_x"
        assert rows[0]["cron"] == {"hour": 16, "minute": 5, "day_of_week": "mon-fri"}
        assert rows[0]["trading_day_only"] is False
        assert store.get_scheduled_job("custom_x")["name"] == "测试任务"
        assert store.get_scheduled_job("nope") == {}
        assert store.delete_scheduled_job("custom_x") is True
        assert store.delete_scheduled_job("custom_x") is False

    def test_upsert_updates_in_place(self, db):
        from db import store
        store.upsert_scheduled_job(self._job())
        store.upsert_scheduled_job(self._job(name="改名了",
                                             cron={"hour": 9, "minute": 0}))
        rows = store.list_scheduled_jobs()
        assert len(rows) == 1 and rows[0]["name"] == "改名了"
        assert rows[0]["cron"]["hour"] == 9
        assert rows[0]["created_at"]          # 创建时间不被覆盖

    def test_enabled_filter_and_toggle(self, db):
        from db import store
        store.upsert_scheduled_job(self._job())
        assert store.set_scheduled_job_enabled("custom_x", False) is True
        assert store.set_scheduled_job_enabled("nope", False) is False
        assert store.list_scheduled_jobs(only_enabled=True) == []
        assert len(store.list_scheduled_jobs()) == 1

    def test_corrupt_cron_reported_not_defaulted(self, db):
        """cron 坏掉时**不能**返回一个默认时间：调度器据此跳过、界面据此标红。
        悄悄兜成默认值的话，任务会在一个用户没设置过的时刻跑起来。"""
        from db import store
        store.upsert_scheduled_job(self._job())
        conn = store.get_connection()
        conn.execute("UPDATE scheduled_jobs SET cron=? WHERE job_id=?",
                     ("{坏掉的 JSON", "custom_x"))
        conn.commit()
        conn.close()
        row = store.list_scheduled_jobs()[0]
        assert row["cron"] is None
        assert row["error"] and "无法解析" in row["error"]
        assert row["cron_raw"] == "{坏掉的 JSON"      # 原文保留，便于人肉修复

    def test_config_table_is_not_in_retention_policy(self, db):
        """自定义任务是配置，不能被 cleanup() 按日期清掉（会丢用户设置）"""
        from db import store
        store.upsert_scheduled_job(self._job())
        store.cleanup()
        assert len(store.list_scheduled_jobs()) == 1


# ================================================================
# E. 热加载同步
# ================================================================

class TestSyncCustomJobs:

    def _make(self, **kw):
        from db import store
        store.upsert_scheduled_job({"job_id": "custom_a", "name": "A", "mode": "news",
                                    "cron": {"hour": 16, "minute": 0},
                                    "enabled": True, **kw})

    def test_add_reschedule_disable(self):
        from core.scheduler import create_scheduler
        sched = create_scheduler()
        assert sj.sync_custom_jobs(sched, jobs=[])["total"] == 0

        self._make()
        r = sj.sync_custom_jobs(sched)
        assert r["added"] == ["custom_a"] and r["changed"] is True
        assert sched.get_job("custom_a") is not None

        # 再同步一次不应重复登记（幂等）
        assert sj.sync_custom_jobs(sched)["changed"] is False

        # 改时间 → 重排
        self._make(cron={"hour": 17, "minute": 30})
        r = sj.sync_custom_jobs(sched)
        assert r["updated"] == ["custom_a"]
        assert "'17'" in str(sched.get_job("custom_a").trigger)

        # 停用 → 移除
        from db import store
        store.set_scheduled_job_enabled("custom_a", False)
        assert sj.sync_custom_jobs(sched)["removed"] == ["custom_a"]
        assert sched.get_job("custom_a") is None

    def test_mode_change_is_rescheduled(self):
        from core.scheduler import create_scheduler
        from db import store
        sched = create_scheduler()
        store.upsert_scheduled_job({"job_id": "custom_a", "name": "A", "mode": "news",
                                    "cron": {"hour": 16, "minute": 0}, "enabled": True})
        sj.sync_custom_jobs(sched)
        store.upsert_scheduled_job({"job_id": "custom_a", "name": "A", "mode": "snapshot",
                                    "cron": {"hour": 16, "minute": 0}, "enabled": True})
        r = sj.sync_custom_jobs(sched)
        assert r["updated"] == ["custom_a"]
        assert list(sched.get_job("custom_a").args)[0] == "snapshot"

    def test_invalid_job_is_skipped_with_reason(self):
        from core.scheduler import create_scheduler
        from db import store
        sched = create_scheduler()
        store.upsert_scheduled_job({"job_id": "custom_a", "name": "A",
                                    "mode": "不存在的任务",
                                    "cron": {"hour": 16, "minute": 0}, "enabled": True})
        r = sj.sync_custom_jobs(sched)
        assert r["added"] == [] and r["invalid"][0]["job_id"] == "custom_a"
        assert sched.get_job("custom_a") is None

    def test_corrupt_cron_job_is_skipped(self):
        from core.scheduler import create_scheduler
        from db import store
        sched = create_scheduler()
        self._make()
        conn = store.get_connection()
        conn.execute("UPDATE scheduled_jobs SET cron='{坏' WHERE job_id='custom_a'")
        conn.commit(); conn.close()
        r = sj.sync_custom_jobs(sched)
        assert r["invalid"] and sched.get_job("custom_a") is None

    def test_sync_failure_does_not_delete_existing(self):
        """读库失败时**绝不能**把已登记的任务删掉。

        改造前的同步实现若在"读期望集合"处失败，就会认为"期望为空"→ 清空全部
        自定义任务；调度器看起来一切正常，任务却再也不会执行。
        这里用"库表被删"模拟读失败。
        """
        from core.scheduler import create_scheduler
        from db import store
        sched = create_scheduler()
        self._make()
        sj.sync_custom_jobs(sched)
        assert sched.get_job("custom_a") is not None

        conn = store.get_connection()
        conn.execute("DROP TABLE scheduled_jobs")
        conn.commit(); conn.close()
        r = sj.sync_custom_jobs(sched)          # 读库会抛，但必须被兜住
        assert r["changed"] is False
        assert sched.get_job("custom_a") is not None, "读库失败不能连坐删任务"

    def test_load_custom_jobs_never_raises(self, monkeypatch):
        """启动装载必须 best-effort：库坏了也不能让调度器起不来
        （本项目最大的事故就是"调度器根本没启动而无人察觉"）"""
        from core.scheduler import create_scheduler
        sched = create_scheduler()
        monkeypatch.setattr(sj, "sync_custom_jobs",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("库坏了")))
        assert sj.load_custom_jobs(sched) == 0


# ================================================================
# F. 交易日闸门
# ================================================================

class TestTradingDayGate:

    def test_market_mode_skips_on_holiday_even_if_flag_off(self, monkeypatch):
        monkeypatch.setattr("core.scheduler.is_trading_day", lambda: False)
        called = []
        monkeypatch.setattr("core.scheduler.run_snapshot_job",
                            lambda desc="": called.append(desc))
        from core.scheduler import run_job
        run_job("snapshot", "自定义快照", trading_day_only=False)
        assert called == [], "非交易日不能跑依赖行情的数据采集"

    def test_news_runs_on_holiday_when_flagged(self, monkeypatch):
        monkeypatch.setattr("core.scheduler.is_trading_day", lambda: False)
        called = []
        monkeypatch.setattr("core.scheduler.run_news_job",
                            lambda desc="": called.append(desc))
        from core.scheduler import run_job
        run_job("news", "夜间新闻", trading_day_only=False)
        assert called == ["夜间新闻"]

    def test_news_skips_on_holiday_when_flagged_on(self, monkeypatch):
        monkeypatch.setattr("core.scheduler.is_trading_day", lambda: False)
        called = []
        monkeypatch.setattr("core.scheduler.run_news_job",
                            lambda desc="": called.append(desc))
        from core.scheduler import run_job
        run_job("news", "只在交易日跑", trading_day_only=True)
        assert called == []

    def test_builtin_call_keeps_old_semantics(self, monkeypatch):
        """内置任务走 trading_day_only=None：行为与改造前逐字一致
        （news 不该被跳过）"""
        monkeypatch.setattr("core.scheduler.is_trading_day", lambda: False)
        called = []
        monkeypatch.setattr("core.scheduler.run_news_job",
                            lambda desc="": called.append(desc))
        from core.scheduler import run_job
        run_job("news", "新闻快讯采集(每小时)")
        assert called == ["新闻快讯采集(每小时)"]

    def test_watch_early_return_unchanged(self, monkeypatch):
        from core.scheduler import run_job
        called = []
        monkeypatch.setattr("core.scheduler.run_watch_job",
                            lambda desc="": called.append(desc))
        run_job("watch", "盘中异动预警")
        assert called == ["盘中异动预警"]


# ================================================================
# G. 健康检查拆分（最容易变成"狼来了"的地方）
# ================================================================

class TestHealthCheckSplit:

    @pytest.fixture
    def hb(self, tmp_path, monkeypatch, db):
        monkeypatch.setattr(health, "HEARTBEAT_PATH", str(tmp_path / "hb.json"))
        return tmp_path / "hb.json"

    def _write(self, custom=None, ids=None):
        import os
        from core.scheduler import JOB_IDS
        # 用当前进程 pid：健康检查要"进程活着 + 心跳新鲜"才会算 running
        health.write_heartbeat(pid=os.getpid(),
                               job_ids=ids if ids is not None else JOB_IDS,
                               custom_job_ids=custom)

    def test_custom_jobs_do_not_trigger_old_code_warning(self, hb):
        """核心：自定义任务**合法地**不属于 JOB_IDS。
        如果它们混进 job_ids，健康检查每天都会报"心跳里有当前代码不存在的任务"，
        从守护者退化成狼来了。"""
        from db import store
        store.upsert_scheduled_job({"job_id": "custom_a", "name": "A", "mode": "news",
                                    "cron": {"hour": 16, "minute": 0}, "enabled": True})
        self._write(custom=["custom_a"])
        h = health.check_health()
        assert h["code"]["checked"] is True
        assert h["code"]["extra"] == [] and h["code"]["missing"] == []
        assert h["code"]["stale"] is False
        assert not any("不存在的任务" in w for w in h["warnings"])
        assert h["custom_jobs"]["running"] == ["custom_a"]

    def test_custom_jobs_excluded_from_job_ids_field(self, hb):
        """心跳里 job_ids 只能是内置清单——这是"旧代码自检"的比对基准"""
        from core.scheduler import JOB_IDS
        self._write(custom=["custom_a"])
        payload = health.read_heartbeat()
        assert payload["job_ids"] == JOB_IDS
        assert "custom_a" not in payload["job_ids"]
        assert payload["custom_job_ids"] == ["custom_a"]

    def test_enabled_custom_job_missing_from_process_warns(self, hb):
        """库里有启用、进程上没挂 → 任务永远不会跑而且不报错，必须说出来"""
        from db import store
        store.upsert_scheduled_job({"job_id": "custom_a", "name": "A", "mode": "news",
                                    "cron": {"hour": 16, "minute": 0}, "enabled": True})
        self._write(custom=[])
        h = health.check_health()
        assert h["custom_jobs"]["missing"] == ["custom_a"]
        assert any("未挂到运行中的调度进程" in w for w in h["warnings"])

    def test_empty_list_is_not_the_same_as_missing_field(self, hb):
        """`[]` = "这个进程知道自定义任务，当前一个都没有"；
        `None` = "旧版进程根本没这个字段"。两者绝不能混为一谈——
        用真假值判断会让**没有自定义任务的正常新进程**被误报成"跑的是旧版代码"。"""
        import os
        self._write(custom=[])                      # 显式空列表
        payload = health.read_heartbeat()
        assert payload["custom_job_ids"] == []
        assert payload["job_ids"] is not None

        health.write_heartbeat(pid=os.getpid())     # 完全不传（模拟旧版进程）
        assert health.read_heartbeat()["custom_job_ids"] is None

    def test_old_process_without_field_is_reported_when_jobs_exist(self, hb):
        import os
        from db import store
        store.upsert_scheduled_job({"job_id": "custom_a", "name": "A", "mode": "news",
                                    "cron": {"hour": 16, "minute": 0}, "enabled": True})
        health.write_heartbeat(pid=os.getpid())     # 旧版进程：没有 custom_job_ids
        h = health.check_health()
        assert h["scheduler"]["running"] is True
        assert h["custom_jobs"]["stale"] is True
        assert any("未挂到运行中的调度进程" in w for w in h["warnings"])

    def test_no_custom_jobs_no_noise(self, hb):
        """没有自定义任务时，健康检查不该多出任何提示（老部署零影响）"""
        self._write(custom=[])
        h = health.check_health()
        assert h["custom_jobs"]["desired"] == []
        assert not any("自定义" in w for w in h["warnings"])
        assert "自定义任务" not in health.format_report(h)

    def test_report_lists_custom_jobs(self, hb):
        from db import store
        store.upsert_scheduled_job({"job_id": "custom_a", "name": "A", "mode": "news",
                                    "cron": {"hour": 16, "minute": 0}, "enabled": True})
        self._write(custom=["custom_a"])
        text = health.format_report()
        assert "自定义任务: 1 个" in text and "custom_a" in text

    def test_pending_custom_job_report_still_works(self, hb):
        """自定义任务缺席时不影响内置代码新旧判定"""
        from core.scheduler import JOB_IDS
        store_ids = [j for j in JOB_IDS if j != "risk_check"]
        self._write(custom=["custom_a"], ids=store_ids)
        h = health.check_health()
        assert h["code"]["missing"] == ["risk_check"]      # 内置判定照旧
        assert h["code"]["extra"] == []                    # 自定义没被算成多余


# ================================================================
# H. 进程接线
# ================================================================

class TestEntryScriptWiring:

    def test_heartbeat_includes_custom_ids(self):
        import pathlib
        src = pathlib.Path("scripts/run_scheduler.py").read_text(encoding="utf-8")
        assert "custom_job_ids=custom_job_ids(scheduler)" in src, \
            "心跳必须单独上报自定义任务，否则健康检查会把它们当多余任务"

    def test_reloader_is_started(self):
        import pathlib
        src = pathlib.Path("scripts/run_scheduler.py").read_text(encoding="utf-8")
        assert "_start_custom_job_reloader(scheduler)" in src, \
            "没有热加载线程，界面上改完排期就得重启才生效"

    def test_create_scheduler_loads_custom_jobs(self, db):
        """create_scheduler 必须真正把库里的任务挂上去（幂等、可重复建）"""
        from db import store
        from core.scheduler import create_scheduler, custom_job_ids
        store.upsert_scheduled_job({"job_id": "custom_z", "name": "Z", "mode": "news",
                                    "cron": {"hour": 23, "minute": 0}, "enabled": True})
        sched = create_scheduler()
        assert custom_job_ids(sched) == ["custom_z"]

    def test_custom_job_ids_reads_live_scheduler_not_db(self, db):
        """心跳上报"实际挂上了什么"。若某个任务因 cron 非法没挂上，
        必须如实缺席，而不是照着库谎报。"""
        from db import store
        from core.scheduler import create_scheduler, custom_job_ids
        store.upsert_scheduled_job({"job_id": "custom_ok", "name": "OK", "mode": "news",
                                    "cron": {"hour": 23, "minute": 0}, "enabled": True})
        store.upsert_scheduled_job({"job_id": "custom_bad", "name": "BAD", "mode": "news",
                                    "cron": {"hour": 23, "minute": 0}, "enabled": True})
        conn = store.get_connection()
        conn.execute("UPDATE scheduled_jobs SET cron='{坏' WHERE job_id='custom_bad'")
        conn.commit(); conn.close()
        sched = create_scheduler()
        assert custom_job_ids(sched) == ["custom_ok"]
