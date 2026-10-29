"""决策汇总与每日触发核对单测（临时库隔离）。

覆盖动机（2026-10-10 工作流诊断）：整条链路开环——三类决策分散在三张表里，
没有任何一处把它们组装起来，于是"选股质量"无法回答、"明日操作建议"无从算起、
复盘只能讲盘面。本文件锁死三件事：
1. 聚合口径正确（今日决策 / 质量评估 / 行动清单），且**缺数据必须显式**；
2. 触及判定**只有一份实现**（回填、每日核对、行动清单共用 `evaluate_triggers`）；
3. 样本未到期时返回 `insufficient_sample` + `next_ready_date`，而不是 0 或空。
"""
import sqlite3
from datetime import datetime

import pandas as pd
import pytest

from db import store as db_store
from core import decision_digest as dd
from core.analysis_history import check_triggers_daily, evaluate_triggers

TODAY = datetime.now().strftime("%Y-%m-%d")


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    yield str(use_template_db(tmp_path, monkeypatch, "test_digest.db"))


@pytest.fixture(autouse=True)
def _isolate_daily_picks(tmp_path, monkeypatch):
    """隔离 `output/daily_picks`（2026-10-29）。

    `today_decisions()` 现在会去读当日推荐名单（`daily_picks`）。不隔离的话
    单测读到**真实的**今天那份 ⇒ 空库用例也会冒出"5 只交通银行"，
    `test_empty_db_reports_reasons` 当场从"无记录"变成"有 5 条"。
    指向一个不存在的目录 ⇒ `_load_daily_picks` 返回 [] ⇒ 用例回到纯库内口径。
    """
    monkeypatch.setattr(dd, "DAILY_PICKS_DIR",
                        str(tmp_path / "no_such_daily_picks"))


@pytest.fixture(autouse=True)
def _no_trigger_lock(monkeypatch):
    """触发核对默认关掉跨进程锁：单测并发时会互相抢锁（同 analysis_history 的约定）"""
    import core.analysis_history as ah
    monkeypatch.setattr(ah.store, "acquire_task_lock", lambda *a, **kw: True)
    monkeypatch.setattr(ah.store, "release_task_lock", lambda *a, **kw: None)


# ---------------------------------------------------------------- helpers

def _raw_pick(db_path, **kw):
    """直插 screen_picks（含 ret/bench——insert_screen_picks 的列是固定的）"""
    p = {"pick_date": TODAY, "stock_code": "600519", "stock_name": "贵州茅台",
         "source": "screen", "strategy": "主力资金", "score": 82.0,
         "score_coverage": 0.9, "price_at_pick": 100.0, "rank_in_run": 1,
         "ret_5d": None, "bench_ret_5d": None, "backfill_status": None}
    p.update(kw)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """INSERT INTO screen_picks
               (pick_date, stock_code, stock_name, source, strategy, score,
                score_coverage, price_at_pick, rank_in_run, factors, conditions,
                ret_5d, bench_ret_5d, backfill_status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (p["pick_date"], p["stock_code"], p["stock_name"], p["source"],
             p["strategy"], p["score"], p["score_coverage"], p["price_at_pick"],
             p["rank_in_run"], "{}", "{}", p["ret_5d"], p["bench_ret_5d"],
             p["backfill_status"], f"{TODAY} 15:49:00"))
        conn.commit()
    finally:
        conn.close()


def _analysis(**kw):
    rec = {"analysis_date": TODAY, "stock_code": "600519", "stock_name": "贵州茅台",
           "decision_label": "看多", "source": "ui",
           "trade_plan": {"direction": "看多", "entry_range": [100, 102],
                          "target_price": "110元", "stop_loss": "95元",
                          "position_size": "20%"}}
    rec.update(kw)
    return db_store.insert_analysis(rec)


def _market(rows):
    return db_store.save_market_snapshot(TODAY, rows)


def _fake_fetcher(df):
    return lambda code, start, end: df


def _kline(days, base=100.0, step=0.5):
    return pd.DataFrame({
        "date": days,
        "close": [base + step * i for i in range(len(days))],
        "high": [base + step * i + 2 for i in range(len(days))],
        "low": [base + step * i - 2 for i in range(len(days))],
    })


# ================================================================
# 触及判定（唯一实现）
# ================================================================

class TestEvaluateTriggers:

    def test_long_hits_by_high_and_low(self):
        r = evaluate_triggers("看多", target=105.0, stop=90.0,
                              highs=[106.0], lows=[95.0])
        assert r["hit_target"] == 1 and r["hit_stop"] == 0

    def test_long_stop_by_low(self):
        r = evaluate_triggers("看多", 120.0, 90.0, highs=[100.0], lows=[89.0])
        assert r["hit_stop"] == 1 and r["hit_target"] == 0

    def test_bear_is_symmetric(self):
        """看空：目标在**下**（跌破算触及）、止损在**上**（涨破算触及）"""
        r = evaluate_triggers("看空", target=80.0, stop=110.0,
                              highs=[112.0], lows=[79.0])
        assert r["hit_target"] == 1 and r["hit_stop"] == 1

    def test_bear_target_not_hit_by_high(self):
        r = evaluate_triggers("看空", target=80.0, stop=None,
                              highs=[200.0], lows=[85.0])
        assert r["hit_target"] == 0

    def test_missing_price_omits_key(self):
        """解析不出价位 / 窗口为空 → 键**缺席**（"没算"≠"算了没触及"）"""
        assert evaluate_triggers("看多", None, None, [100.0], [90.0]) == {}
        assert evaluate_triggers("看多", 110.0, 90.0, [], []) == {}


# ================================================================
# 今日决策清单
# ================================================================

class TestTodayDecisions:

    def test_empty_db_reports_reasons(self, temp_store):
        out = dd.today_decisions(as_of=TODAY)
        assert out["screens"] == [] and out["analyses"] == []
        assert "今日无选股记录" in out["empty_reasons"]
        assert "今日无深度分析记录" in out["empty_reasons"]

    def test_collects_screens_analyses_trades(self, temp_store):
        _raw_pick(temp_store, stock_code="600900", stock_name="长江电力",
                  score=77.6, rank_in_run=1)
        _analysis(stock_code="600211", stock_name="西藏药业")
        db_store.save_paper_state("default", {
            "name": "default", "initial_cash": 1000000, "cash": 500000,
            "positions": {}, "orders": [],
            "fills": [{"date": TODAY, "code": "600211", "direction": "买入",
                       "quantity": 1000, "actual_price": 36.0, "total": 36050.0}],
            "nav": [{"date": TODAY, "total_value": 1000000.0}]})
        out = dd.today_decisions(as_of=TODAY)
        assert out["screens"][0]["code"] == "600900"
        # 没有 daily_picks 产物 ⇒ 名单退化为扩池前 5 名，**必须显式标出来**
        assert out["screens"][0]["is_final"] is False
        assert out["screen_pool"][0]["code"] == "600900"
        assert out["analyses"][0]["advice_status"] == "可执行"
        assert out["analyses"][0]["target"] == 110.0
        assert out["trades"][0]["code"] == "600211"
        # 唯一允许的"原因"就是刚才那条降级说明 —— 别把降级藏起来换一个空列表
        assert out["empty_reasons"] == [
            "当日无 daily_picks 产物 ⇒ 选股名单退化为扩池前 5 名"]

    def test_degraded_strategy_is_flagged(self, temp_store):
        """`strategy=内置精选池` = 候选域被收窄到 18 只硬编码（降级的直接证据）"""
        _raw_pick(temp_store, strategy="内置精选池")
        out = dd.today_decisions(as_of=TODAY)
        assert out["screens"][0]["degraded"] is True

    def test_same_day_same_code_never_duplicates(self, temp_store):
        """⭐ 同日同票只留一条（昭衍新药一天 4 条的教训）。

        2026-10-19 起 `analyses` 有 `(analysis_date, stock_code)` 唯一索引 + UPSERT，
        "重复"从**数据库层**就不可能产生，因此 `duplicate` 标记只作为
        **未迁移/索引建失败**的历史库兜底（那条路径下它仍会亮）。
        """
        _analysis(stock_code="603127")
        _analysis(stock_code="603127")       # 第二次覆盖第一次
        _analysis(stock_code="600519")
        out = dd.today_decisions(as_of=TODAY)
        by_code = {}
        for a in out["analyses"]:
            by_code.setdefault(a["code"], []).append(a["duplicate"])
        assert by_code["603127"] == [False], by_code      # 只剩一条
        assert by_code["600519"] == [False]

    def test_other_date_excluded(self, temp_store):
        _raw_pick(temp_store, pick_date="2026-09-30")
        out = dd.today_decisions(as_of=TODAY)
        assert out["screens"] == []


# ================================================================
# 质量评估
# ================================================================

class TestQualityAssessment:

    def test_insufficient_sample_has_next_ready_date(self, temp_store):
        _raw_pick(temp_store, pick_date=TODAY, ret_5d=None,
                  backfill_status=None)
        q = dd.quality_assessment()
        assert q["screen"]["status"] == "insufficient_sample"
        assert q["screen"]["n_backfilled"] == 0
        assert q["screen"]["n_pending"] == 1
        # 最新待回填日 + T+5 → "预计哪天出结果"（不省略）
        assert q["screen"]["next_ready_date"]

    def test_score_band_monotonicity(self, temp_store):
        """分档：分数高的组超额更高时，by_score_band 必须如实反映"""
        # 高分（85）两只，超额 +4 / -1；低分（65）两只，超额 0 / -3
        _raw_pick(temp_store, stock_code="600001", score=85, rank_in_run=1,
                  ret_5d=5.0, bench_ret_5d=1.0)
        _raw_pick(temp_store, stock_code="600002", score=85, rank_in_run=2,
                  ret_5d=-2.0, bench_ret_5d=-1.0)
        _raw_pick(temp_store, stock_code="600003", score=65, rank_in_run=3,
                  ret_5d=1.0, bench_ret_5d=1.0)
        _raw_pick(temp_store, stock_code="600004", score=65, rank_in_run=4,
                  ret_5d=-5.0, bench_ret_5d=-2.0)
        q = dd.quality_assessment()["screen"]
        bands = {b["band"]: b for b in q["by_score_band"]}
        assert bands["80-90"]["n"] == 2
        assert bands["80-90"]["avg_excess"] == pytest.approx(1.5, abs=1e-6)
        # 65 档两只：超额 0 与 -3 → 均值 -1.5
        assert bands["60-70"]["avg_excess"] == pytest.approx(-1.5, abs=1e-6)
        assert q["status"] == "ok"
        assert q["top5_excess"] is not None

    def test_analysis_label_hit_rate(self, temp_store):
        """看多赚 / 看空跌才算对（对称口径）"""
        a1 = _analysis(stock_code="600001", decision_label="看多")
        a2 = _analysis(stock_code="600002", decision_label="看多")
        a3 = _analysis(stock_code="600003", decision_label="看空")
        db_store.update_analysis_backfill(a1, {"ret_5d": 3.0, "backfill_status": "done"})
        db_store.update_analysis_backfill(a2, {"ret_5d": -1.0, "backfill_status": "done"})
        db_store.update_analysis_backfill(a3, {"ret_5d": -2.0, "backfill_status": "done"})
        q = dd.quality_assessment()["analysis"]
        by = {b["label"]: b for b in q["by_label"]}
        assert by["看多"]["n"] == 2 and by["看多"]["hit_rate"] == 0.5
        assert by["看空"]["n"] == 1 and by["看空"]["hit_rate"] == 1.0
        assert q["status"] == "ok"

    def test_portfolio_attribution(self, temp_store):
        _market([{"stock_code": "600211", "stock_name": "西藏药业", "price": 40.0}])
        db_store.save_paper_state("default", {
            "name": "default", "initial_cash": 1000000, "cash": 800000,
            "positions": {"600211": {"code": "600211", "quantity": 5000,
                                     "available": 5000, "total_cost": 180000.0,
                                     "last_buy_date": "2026-09-20"}},
            "orders": [], "fills": [],
            "nav": [{"date": "2026-09-28", "total_value": 1000000.0},
                    {"date": TODAY, "total_value": 1000000.0}]})
        pf = dd.quality_assessment()["portfolio"]
        # 市值 40*5000=200000，成本 180000 → 浮盈 20000
        assert pf["contributors"][0]["pnl"] == 20000.0
        assert pf["total_value"] == 1000000.0     # 800000 现金 + 200000 市值
        assert pf["total_pnl"] == 0.0
        assert pf["status"] == "ok"

    def test_portfolio_without_account(self, temp_store):
        assert dd.quality_assessment()["portfolio"]["status"] == "no_account"

    def test_benchmark_missing_is_none_not_zero(self, temp_store):
        """指数特征库取不到 → pct=None（**不是 0**，"缺就是缺"）"""
        db_store.save_paper_state("default", {
            "name": "default", "initial_cash": 1000000, "cash": 1000000,
            "positions": {}, "orders": [], "fills": [],
            "nav": [{"date": "2026-09-28", "total_value": 1000000.0}]})
        bm = dd.quality_assessment()["portfolio"]["benchmark"]
        assert bm["pct"] is None and bm["start"] is None


# ================================================================
# 行动清单
# ================================================================

class TestActionList:

    def _with_position(self, code="600211", last=40.0):
        _market([{"stock_code": code, "stock_name": "西藏药业", "price": last}])
        db_store.save_paper_state("default", {
            "name": "default", "initial_cash": 1000000, "cash": 800000,
            "positions": {code: {"code": code, "quantity": 5000, "available": 5000,
                                 "total_cost": 180000.0, "last_buy_date": "2026-09-20"}},
            "orders": [], "fills": [], "nav": []})

    def test_position_not_triggered(self, temp_store):
        self._with_position(last=100.0)
        _analysis(stock_code="600211",
                  trade_plan={"direction": "看多", "entry_range": [98, 102],
                              "target_price": "110元", "stop_loss": "95元"})
        pos = dd.action_list()["positions"][0]
        assert pos["triggered"] is None
        assert pos["to_target_pct"] == pytest.approx(10.0, abs=1e-6)
        assert pos["to_stop_pct"] == pytest.approx(-5.0, abs=1e-6)

    def test_position_triggers_stop(self, temp_store):
        self._with_position(last=90.0)
        _analysis(stock_code="600211",
                  trade_plan={"direction": "看多", "target_price": "110元",
                              "stop_loss": "95元"})
        assert dd.action_list()["positions"][0]["triggered"] == "stop"

    def test_position_triggers_target(self, temp_store):
        self._with_position(last=115.0)
        _analysis(stock_code="600211",
                  trade_plan={"direction": "看多", "target_price": "110元",
                              "stop_loss": "95元"})
        assert dd.action_list()["positions"][0]["triggered"] == "target"

    def test_position_without_plan_has_note(self, temp_store):
        self._with_position()
        pos = dd.action_list()["positions"][0]
        assert pos["stop"] is None and "没有可执行档" in pos["note"]

    def test_watch_uses_entry_conditions(self, temp_store):
        """驳回 / 等待档给出的是 `entry_conditions`（A股无融券：看空=驳回买入）"""
        _market([{"stock_code": "603127", "stock_name": "贤丰控股", "price": 5.5}])
        _analysis(stock_code="603127", decision_label="看空",
                  trade_plan={"direction": "看空", "entry_range": [5.8, 6.0],
                              "entry_conditions": ["连续两日站稳MA20", "主力净流入>3000万"]})
        w = dd.action_list()["watch"][0]
        assert w["advice_status"] == "驳回"
        assert w["entry_conditions"] == ["连续两日站稳MA20", "主力净流入>3000万"]
        assert w["to_entry_pct"] is not None

    def test_executable_plan_not_in_watch(self, temp_store):
        _market([{"stock_code": "600519", "stock_name": "贵州茅台", "price": 100.0}])
        _analysis(stock_code="600519")     # 默认看多/可执行
        assert dd.action_list()["watch"] == []

    def test_sector_outflow_risk(self, temp_store):
        for d, vals in (("2026-10-08", {"光学光电子": -20e8, "银行": 5e8}),
                        ("2026-10-09", {"光学光电子": -10e8, "银行": 3e8})):
            db_store.save_industry_flow_snapshot(
                d, [{"industry": k, "net_amount": v} for k, v in vals.items()])
        risks = dd.action_list()["risks"]
        assert len(risks) == 1
        assert "光学光电子" in risks[0]["text"]
        assert "银行" not in risks[0]["text"]        # 净流入不算风险


# ================================================================
# 每日触发核对
# ================================================================

class TestCheckTriggersDaily:

    def test_insufficient_kline_is_not_ready(self, temp_store):
        _analysis(trade_plan={"direction": "看多", "target_price": "110元",
                              "stop_loss": "95元"})
        # 只有分析日当天一根 → 窗口为空 → not_ready（不是"未触及"）
        stats = check_triggers_daily(
            use_lock=False, price_fetcher=_fake_fetcher(_kline([TODAY])))
        assert stats["not_ready"] == 1 and stats["updated"] == 0

    def test_writes_hit_stop_when_touched(self, temp_store):
        aid = _analysis(analysis_date="2026-09-20",
                        trade_plan={"direction": "看多", "target_price": "110元",
                                    "stop_loss": "95元"})
        days = ["2026-09-20", "2026-09-21", "2026-09-22", "2026-09-23"]
        df = pd.DataFrame({"date": days, "close": [100, 100, 100, 100],
                           "high": [101, 101, 101, 101],
                           "low": [99, 98, 94, 99]})     # 09-22 跌破 95 → 触及止损
        stats = check_triggers_daily(use_lock=False, price_fetcher=_fake_fetcher(df),
                                     max_age_days=3650)
        assert stats["updated"] == 1
        row = db_store.get_analysis(aid)
        assert row["hit_stop"] == 1
        assert row["hit_target"] in (None, 0)          # 未触及目标

    def test_idempotent(self, temp_store):
        _analysis(analysis_date="2026-09-20",
                  trade_plan={"direction": "看多", "target_price": "110元",
                              "stop_loss": "95元"})
        df = _kline(["2026-09-20", "2026-09-21", "2026-09-22"], base=100.0, step=5.0)
        f = _fake_fetcher(df)
        first = check_triggers_daily(use_lock=False, price_fetcher=f, max_age_days=3650)
        second = check_triggers_daily(use_lock=False, price_fetcher=f, max_age_days=3650)
        assert first["updated"] >= 0
        assert second["updated"] == 0, "重复调用不得再写（幂等）"

    def test_done_rows_are_skipped(self, temp_store):
        """已完成（backfill_status=done）的样本不再每日重复核对"""
        _analysis(trade_plan={"direction": "看多", "target_price": "110元",
                              "stop_loss": "95元"}, backfill_status="done")
        stats = check_triggers_daily(
            use_lock=False, price_fetcher=_fake_fetcher(_kline([TODAY])),
            max_age_days=3650)
        assert stats["checked"] == 0


# ================================================================
# 聚合入口
# ================================================================

class TestBuildDigest:

    def test_all_three_blocks_present(self, temp_store):
        d = dd.build_digest()
        assert set(d) == {"today", "quality", "actions"}
        assert d["today"]["date"] and d["quality"]["portfolio"]["status"] in (
            "ok", "no_account")

    def test_block_failure_isolated(self, temp_store, monkeypatch):
        """某一块炸了不影响其它块（红线 5）"""
        monkeypatch.setattr(dd, "today_decisions",
                            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("炸")))
        d = dd.build_digest()
        assert d["today"]["status"] == "error"
        assert "quality" in d and "actions" in d


# ================================================================
# 去重口径（2026-10-24）
# ================================================================
# 事故：`screen_picks` 的同一 `(pick_date, stock_code)` 会被 `pipeline` 多轮
# 回填写多行（实测同股同日最多 9 行、全表 857 行只有 401 条唯一记录）。
# 按行求均值等于**给"重复出现过的票"加权** —— 实测把"6 条 / 2 只票"放大成
# "21 条 / -5.20%"，结论强度虚高 3.5 倍。
#
# 这两条用例故意构造"按行算 ≠ 去重算"的数据，确保口径真的生效（有鉴别力）。

class TestScreenPicksDedup:
    def test_quality_uses_deduped_rows(self, temp_store):
        """A 票重复 3 行（+10%）、B 票 1 行（-10%）。

        按行算 = (+0.10×3 - 0.10)/4 = **+5%**；去重算 = **0%**。
        断言 0 —— 若退回按行统计，这条会红。
        """
        for _ in range(3):
            _raw_pick(temp_store, stock_code="600000", score=50.0, rank_in_run=9,
                      ret_5d=0.10, bench_ret_5d=0.0,
                      backfill_status="done")
        _raw_pick(temp_store, stock_code="600001", score=50.0, rank_in_run=9,
                  ret_5d=-0.10, bench_ret_5d=0.0, backfill_status="done")

        q = dd._screen_quality(5)
        assert q["n_backfilled"] == 2, q["n_backfilled"]
        assert q["all_excess"] == pytest.approx(0.0, abs=1e-9), q["all_excess"]

    def test_pending_count_is_deduped(self, temp_store):
        """待回填计数也必须去重：否则"积压 589 条"实为 303 条（虚高 1.94×）。"""
        for _ in range(3):
            _raw_pick(temp_store, stock_code="600000", backfill_status=None)
        _raw_pick(temp_store, stock_code="600001", backfill_status="partial")
        assert db_store.count_screen_picks_dedup(
            "WHERE s.backfill_status IS NULL OR s.backfill_status = 'partial'") == 2

    def test_last_written_row_wins(self, temp_store):
        """同一 (日期,代码) 保留 `id` 最大的一行 —— 后写的回填结果优先。"""
        _raw_pick(temp_store, stock_code="600000", ret_5d=None,
                  backfill_status=None)
        _raw_pick(temp_store, stock_code="600000", ret_5d=0.05, bench_ret_5d=0.0,
                  backfill_status="done")
        rows = db_store.load_screen_picks_dedup(
            "stock_code, ret_5d, backfill_status", "WHERE s.pick_date = ?", (TODAY,))
        assert len(rows) == 1
        assert rows[0]["ret_5d"] == pytest.approx(0.05)
        assert rows[0]["backfill_status"] == "done"


class TestStageHealthBackfillRows:
    def test_counts_unique_samples_not_rows(self, temp_store):
        """`_screen_backfill_rows` 同样按 (日期,代码) 去重。"""
        from core import stage_health as SH
        for _ in range(4):
            _raw_pick(temp_store, stock_code="600000",
                      backfill_status="done")
        db_path = temp_store
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("UPDATE screen_picks SET backfilled_at = ?",
                         (f"{TODAY} 15:36:00",))
            conn.commit()
        finally:
            conn.close()
        assert SH._screen_backfill_rows(TODAY) == 1


# ================================================================
# 去重口径（2026-10-24）
# ================================================================
# 事故：`screen_picks` 的同一 `(pick_date, stock_code)` 会被 `pipeline` 多轮
# 回填写多行（实测同股同日最多 9 行、全表 857 行只有 401 条唯一记录）。
# 按行求均值等于**给"重复出现过的票"加权** —— 实测把"6 条 / 2 只票"放大成
# "21 条 / -5.20%"，结论强度虚高 3.5 倍。
#
# 这两条用例故意构造"按行算 ≠ 去重算"的数据，确保口径真的生效（有鉴别力）。

class TestScreenPicksDedup:
    def test_quality_uses_deduped_rows(self, temp_store):
        """A 票重复 3 行（+10%）、B 票 1 行（-10%）。

        按行算 = (+0.10×3 - 0.10)/4 = **+5%**；去重算 = **0%**。
        断言 0 —— 若退回按行统计，这条会红。
        """
        for _ in range(3):
            _raw_pick(temp_store, stock_code="600000", score=50.0, rank_in_run=9,
                      ret_5d=0.10, bench_ret_5d=0.0,
                      backfill_status="done")
        _raw_pick(temp_store, stock_code="600001", score=50.0, rank_in_run=9,
                  ret_5d=-0.10, bench_ret_5d=0.0, backfill_status="done")

        q = dd._screen_quality(5)
        assert q["n_backfilled"] == 2, q["n_backfilled"]
        assert q["all_excess"] == pytest.approx(0.0, abs=1e-9), q["all_excess"]

    def test_pending_count_is_deduped(self, temp_store):
        """待回填计数也必须去重：否则"积压 589 条"实为 303 条（虚高 1.94×）。"""
        for _ in range(3):
            _raw_pick(temp_store, stock_code="600000", backfill_status=None)
        _raw_pick(temp_store, stock_code="600001", backfill_status="partial")
        assert db_store.count_screen_picks_dedup(
            "WHERE s.backfill_status IS NULL OR s.backfill_status = 'partial'") == 2

    def test_last_written_row_wins(self, temp_store):
        """同一 (日期,代码) 保留 `id` 最大的一行 —— 后写的回填结果优先。"""
        _raw_pick(temp_store, stock_code="600000", ret_5d=None,
                  backfill_status=None)
        _raw_pick(temp_store, stock_code="600000", ret_5d=0.05, bench_ret_5d=0.0,
                  backfill_status="done")
        rows = db_store.load_screen_picks_dedup(
            "stock_code, ret_5d, backfill_status", "WHERE s.pick_date = ?", (TODAY,))
        assert len(rows) == 1
        assert rows[0]["ret_5d"] == pytest.approx(0.05)
        assert rows[0]["backfill_status"] == "done"


class TestStageHealthBackfillRows:
    def test_counts_unique_samples_not_rows(self, temp_store):
        """`_screen_backfill_rows` 同样按 (日期,代码) 去重。"""
        from core import stage_health as SH
        for _ in range(4):
            _raw_pick(temp_store, stock_code="600000",
                      backfill_status="done")
        db_path = temp_store
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("UPDATE screen_picks SET backfilled_at = ?",
                         (f"{TODAY} 15:36:00",))
            conn.commit()
        finally:
            conn.close()
        assert SH._screen_backfill_rows(TODAY) == 1
