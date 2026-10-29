"""分析历史入库 + T+N 回填闭环单元测试（临时库隔离，假 K 线，零网络/LLM）

覆盖三层：
1. store 层：analyses 表增删查/过滤/回填白名单/保留策略
2. analysis_history 模块：save_analysis / save_failure / 快照序列化 /
   backfill 状态机（done/partial/failed/skip）/ 目标止损触及（含看空对称口径）
3. 调度接线：15:35 回填任务注册、交易日保护、分发
"""
from datetime import datetime, timedelta

import pandas as pd
import pytest

import db.models as models
from core import analysis_history as ah
from db import store as db_store


def _days_ago(n: int) -> str:
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    """数据库指向临时文件（get_connection 读 db.models.DB_PATH 模块全局）"""
    from tests.conftest import use_template_db
    db_path = use_template_db(tmp_path, monkeypatch, "test_ana.db")
    yield


def _fake_kline(days: int, base: float = 100.0, drift: float = 1.0,
                high_pad: float = 2.0, low_pad: float = 2.0) -> pd.DataFrame:
    """从 days 天前到今天（含）的连续日 K：每天收盘 base + i*drift"""
    today = datetime.now()
    dates = [(today - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(days, -1, -1)]
    closes = [base + i * drift for i in range(len(dates))]
    return pd.DataFrame({
        "date": dates, "close": closes,
        "high": [c + high_pad for c in closes],
        "low": [c - low_pad for c in closes],
    })


# ================================================================
# store 层
# ================================================================

class TestStoreAnalyses:

    def test_insert_and_list(self, temp_store):
        rid = db_store.insert_analysis({
            "analysis_date": "2026-09-01", "stock_code": "600519",
            "stock_name": "贵州茅台", "price_at_analysis": 1500.0,
            "decision_label": "看多", "model": "deepseek-chat", "source": "ui",
            "payload": {"final_decision": "ok"}, "trade_plan": {"direction": "看多"},
            "timing_regime": "range", "market_score": 55.0,
        })
        assert rid > 0
        rows = db_store.list_analyses()
        assert len(rows) == 1
        r = rows[0]
        assert r["stock_code"] == "600519" and r["decision_label"] == "看多"
        assert r["backfill_status"] is None      # 新记录待回填
        assert "payload" not in r                # 列表查询不带大 JSON

    def test_get_analysis_decodes_json(self, temp_store):
        rid = db_store.insert_analysis({
            "analysis_date": "2026-09-01", "stock_code": "600519",
            "payload": {"final_decision": "看多"}, "trade_plan": {"direction": "看多"},
        })
        full = db_store.get_analysis(rid)
        assert full["payload"] == {"final_decision": "看多"}
        assert full["trade_plan"]["direction"] == "看多"

    def test_update_backfill_whitelist(self, temp_store):
        """回填接口只允许更新回填相关字段（防止误改决策标签等）"""
        rid = db_store.insert_analysis({"analysis_date": "2026-09-01",
                                        "stock_code": "600519"})
        db_store.update_analysis_backfill(rid, {
            "ret_5d": 0.05, "backfill_status": "partial",
            "decision_label": "HACKED",            # 不在白名单，应被忽略
        })
        full = db_store.get_analysis(rid)
        assert full["ret_5d"] == 0.05
        assert full["decision_label"] is None      # 未被篡改

    def test_list_filters(self, temp_store):
        db_store.insert_analysis({"analysis_date": _days_ago(30), "stock_code": "600519"})
        db_store.insert_analysis({"analysis_date": _days_ago(10), "stock_code": "000001",
                                  "backfill_status": "done"})
        db_store.insert_analysis({"analysis_date": _days_ago(5), "stock_code": "000002",
                                  "backfill_status": "skip"})
        assert len(db_store.list_analyses()) == 3
        assert len(db_store.list_analyses(only_backfillable=True)) == 1  # 只有 NULL 那条
        assert len(db_store.list_analyses(stock_code="600519")) == 1
        assert len(db_store.list_analyses(backfill_status="done")) == 1

    def test_cleanup_retention_registered(self, temp_store):
        """analyses 表必须注册进 cleanup 保留策略（400 天）"""
        rid = db_store.insert_analysis({"analysis_date": _days_ago(401),
                                        "stock_code": "600519"})
        conn = models.get_connection()
        try:
            conn.execute(
                "UPDATE analyses SET created_at = ? WHERE id = ?",
                ((datetime.now() - timedelta(days=401)).strftime("%Y-%m-%d %H:%M:%S"), rid))
            conn.commit()
        finally:
            conn.close()
        deleted = db_store.cleanup()
        assert deleted.get("analyses", 0) == 1
        assert db_store.list_analyses() == []


# ================================================================
# save_analysis / save_failure / 快照
# ================================================================

class TestSaveAnalysis:

    def _final_state(self, **over):
        state = {
            "stock_code": "600519", "analysis_date": _days_ago(5),
            "stock_name": "贵州茅台",
            "stock_info": {"name": "贵州茅台", "current_price": 1500.0},
            "timing_signal": {"regime": "range", "total_score": 55.0},
            "trade_plan": {"direction": "看多", "target_price": "1600元",
                           "stop_loss": "1450元"},
            "final_decision": "综合评估建议看多",
            "tech_report": "t", "errors": [], "fatal_errors": [],
            "bull_history": ["b1"], "bear_history": ["s1"],
        }
        state.update(over)
        return state

    def test_full_fields_extracted(self, temp_store):
        from core.analysis_history import save_analysis
        rid = save_analysis(self._final_state(), source="ui")
        assert rid > 0
        r = db_store.list_analyses()[0]
        assert r["decision_label"] == "看多"        # 优先 trade_plan.direction
        assert r["timing_regime"] == "range"
        assert r["market_score"] == 55.0
        assert r["price_at_analysis"] == 1500.0
        assert r["stock_name"] == "贵州茅台"        # stock_info.name 优先
        assert r["analysis_date"] == _days_ago(5)
        assert r["backfill_status"] is None          # 正常样本待回填

    def test_label_fallback_to_decision_text(self, temp_store):
        """direction 缺失时回退 final_decision 文本；两者皆空返回空标签"""
        from core.analysis_history import save_analysis
        # ⚠️ 两条必须用**不同的票**：`analyses` 现在有 (date, code) 唯一索引 + UPSERT，
        #    同票同日第二次会覆盖第一次（见 test_same_day_same_code_upserts）。
        rid1 = save_analysis(self._final_state(stock_code="600519", trade_plan={},
                                              final_decision="谨慎观望为宜"))
        rid2 = save_analysis(self._final_state(stock_code="000001", stock_name="平安银行",
                                              trade_plan={}, final_decision=""))
        assert db_store.get_analysis(rid1)["decision_label"] == "观望"
        assert not db_store.get_analysis(rid2)["decision_label"]

    def test_fatal_errors_marked_skip(self, temp_store):
        """分析链被致命错误阻断：不参与胜率统计，但留痕"""
        from core.analysis_history import save_analysis
        save_analysis(self._final_state(fatal_errors=["K线: 无数据"],
                                        final_decision=""))
        r = db_store.list_analyses()[0]
        assert r["backfill_status"] == "skip"
        assert r["decision_label"] == "分析失败"

    def test_save_failure(self, temp_store):
        from core.analysis_history import save_failure
        rid = save_failure("000001", "LLM 超时", source="pipeline", stock_name="平安银行")
        assert rid > 0
        r = db_store.list_analyses()[0]
        assert r["decision_label"] == "分析失败"
        assert r["backfill_status"] == "skip"
        assert r["stock_code"] == "000001"

    def test_model_name_reflects_actual_nodes(self, temp_store):
        """回归（批次 13 端到端实跑发现）：模型名曾因 `import config` 取空包属性
        而静默落成 'unknown'，导致历史无法按模型分组——后续对照实验直接废掉。

        ⚠️ 2026-10-20 起不能只断言 `== DEFAULT_MODEL_NAME`：按功能分配不同模型后
        （深度分析写作节点 qwen、决策节点 deepseek），单一全局默认值**不代表真实
        构成**。这里断言"能反映本次分析实际用到的模型集合"：
          全同 ⇒ 该模型名；混用 ⇒ `mixed:a+b`。
        这样配置怎么改都成立，且真的验到了"按模型分组"的前提。
        """
        from config.model_routing import resolve_model
        from core.analysis_history import _ANALYSIS_FEATURES, _current_model_name, save_analysis
        got = _current_model_name()
        assert got and got != "unknown"
        expect = {resolve_model(f) for f in _ANALYSIS_FEATURES}
        if len(expect) == 1:
            assert got == expect.pop()
        else:
            assert got.startswith("mixed:"), got
            assert all(m in got for m in expect), (got, expect)
        rid = save_analysis(self._final_state(), source="ui")
        assert db_store.get_analysis(rid)["model"] != "unknown"

    def test_snapshot_json_safe(self, temp_store):
        """numpy 标量 → python 标量；DataFrame 等不可序列化对象 → None"""
        import numpy as np
        from core.analysis_history import snapshot_from_state
        snap = snapshot_from_state({
            "stock_code": "600519",
            "stock_info": {"current_price": np.float64(12.3)},
            "trade_plan": {"confidence": np.int64(85), "flag": np.bool_(True)},
            "kline_data": pd.DataFrame({"close": [1, 2]}),   # 不进快照
            "errors": ["e1"],
        })
        assert snap["current_price"] == 12.3
        assert snap["trade_plan"]["confidence"] == 85
        assert snap["trade_plan"]["flag"] is True
        assert "kline_data" not in snap
        import json
        json.dumps(snap)                                  # 全部可序列化


# ================================================================
# T+N 回填状态机
# ================================================================

class TestBackfill:

    def _insert_for_backfill(self, analysis_days_ago: int, trade_plan: dict = None) -> int:
        rid = db_store.insert_analysis({
            "analysis_date": _days_ago(analysis_days_ago),
            "stock_code": "600519", "stock_name": "贵州茅台",
            "decision_label": "看多",
            "trade_plan": trade_plan if trade_plan is not None
                          else {"direction": "看多"},
        })
        return rid

    def test_done_when_full_window(self, temp_store):
        """数据足够 → ret_5d/ret_20d 精确计算，状态 done。
        K 线从分析日起算（_fake_kline(30)）：基准=分析日收盘 100，
        T+5 收盘 105，T+20 收盘 120。"""
        from core.analysis_history import backfill_analyses
        rid = self._insert_for_backfill(30)
        stats = backfill_analyses(price_fetcher=lambda c, s, e: _fake_kline(30))
        assert stats["done"] == 1 and stats["updated"] == 1
        full = db_store.get_analysis(rid)
        assert full["ret_5d"] == pytest.approx(0.05)      # 105/100-1
        assert full["ret_20d"] == pytest.approx(0.20)     # 120/100-1
        assert full["backfill_status"] == "done"

    def test_partial_when_only_5d_available(self, temp_store):
        """只有 T+5 数据 → partial；次日任务会重试补 T+20"""
        from core.analysis_history import backfill_analyses
        rid = self._insert_for_backfill(10)
        stats = backfill_analyses(price_fetcher=lambda c, s, e: _fake_kline(10))
        assert stats["partial"] == 1
        full = db_store.get_analysis(rid)
        assert full["ret_5d"] == pytest.approx(0.05)
        assert full["ret_20d"] is None
        assert full["backfill_status"] == "partial"

    def test_too_young_stays_pending(self, temp_store):
        """分析太新（数据不足 T+5）→ 保持 NULL，明日重试"""
        from core.analysis_history import backfill_analyses
        rid = self._insert_for_backfill(3)
        stats = backfill_analyses(price_fetcher=lambda c, s, e: _fake_kline(3))
        assert stats["skipped"] == 1 and stats["updated"] == 0
        assert db_store.get_analysis(rid)["backfill_status"] is None

    def test_overage_marks_failed_without_fetch(self, temp_store):
        """超过 60 天时限 → 直接 failed，不再拉 K 线"""
        from core.analysis_history import backfill_analyses

        def _boom(c, s, e):
            raise AssertionError("超龄样本不应再拉 K 线")

        rid = self._insert_for_backfill(61)
        stats = backfill_analyses(price_fetcher=_boom)
        assert stats["failed"] == 1
        assert db_store.get_analysis(rid)["backfill_status"] == "failed"

    def test_source_error_keeps_pending(self, temp_store):
        """数据源异常（限频/熔断）→ 保持待回填，记录错误"""
        from core.analysis_history import backfill_analyses
        rid = self._insert_for_backfill(30)

        def _err(c, s, e):
            raise RuntimeError("429 Too Many Requests")

        stats = backfill_analyses(price_fetcher=_err)
        assert stats["skipped"] == 1
        assert stats["errors"] and "429" in stats["errors"][0]
        assert db_store.get_analysis(rid)["backfill_status"] is None

    def test_hit_target_and_stop_bull(self, temp_store):
        """看多：window 内 high≥target → 触及目标；low>stop → 未触止损"""
        from core.analysis_history import backfill_analyses
        rid = self._insert_for_backfill(
            30, trade_plan={"direction": "看多", "target_price": "105元",
                            "stop_loss": "95元"})
        backfill_analyses(price_fetcher=lambda c, s, e: _fake_kline(35))
        full = db_store.get_analysis(rid)
        # base=100（30天前），window=T+1..T+20：high 从 103 到 122，low 从 98 到 117
        assert full["hit_target"] == 1        # high 103 ≥ 105 出现在 T+3
        assert full["hit_stop"] == 0          # low 最低 98 > 95

    def test_hit_target_and_stop_bear(self, temp_store):
        """看空对称口径：target 在下方（low≤target 触及），stop 在上方（high≥stop 触发）"""
        from core.analysis_history import backfill_analyses
        rid = self._insert_for_backfill(
            30, trade_plan={"direction": "看空", "target_price": "120元",
                            "stop_loss": "110元"})
        backfill_analyses(price_fetcher=lambda c, s, e: _fake_kline(35))
        full = db_store.get_analysis(rid)
        assert full["hit_target"] == 1        # 后期 low（118 ≤ 120）
        assert full["hit_stop"] == 1          # T+9 起 high ≥ 110

    def test_price_parse_failure_leaves_null(self, temp_store):
        """trader 输出的价位无数字（如'视情况而定'）→ hit 字段保持 NULL"""
        from core.analysis_history import backfill_analyses
        rid = self._insert_for_backfill(
            30, trade_plan={"direction": "看多", "target_price": "视情况而定",
                            "stop_loss": ""})
        backfill_analyses(price_fetcher=lambda c, s, e: _fake_kline(35))
        full = db_store.get_analysis(rid)
        assert full["ret_5d"] is not None     # 收益率不受影响
        assert full["hit_target"] is None and full["hit_stop"] is None

    def test_include_failed_retries(self, temp_store):
        """failed 行默认不重试；include_failed=True（手动按钮）连它一起补"""
        from core.analysis_history import backfill_analyses
        rid = self._insert_for_backfill(30)
        db_store.update_analysis_backfill(rid, {"backfill_status": "failed"})
        stats = backfill_analyses(price_fetcher=lambda c, s, e: _fake_kline(35))
        assert stats["checked"] == 0                      # failed 不在默认范围
        stats2 = backfill_analyses(include_failed=True,
                                   price_fetcher=lambda c, s, e: _fake_kline(35))
        assert stats2["checked"] == 1 and stats2["done"] == 1
        assert db_store.get_analysis(rid)["backfill_status"] == "done"

    def test_kline_column_compatibility(self, temp_store):
        """K 线列名兼容：'日期'/'Date' 都能归一化"""
        from core.analysis_history import backfill_analyses
        rid = self._insert_for_backfill(30)
        df = _fake_kline(30).rename(columns={"date": "日期"})
        backfill_analyses(price_fetcher=lambda c, s, e: df)
        full = db_store.get_analysis(rid)
        assert full["ret_5d"] == pytest.approx(0.05)

    def test_skip_rows_never_backfilled(self, temp_store):
        """分析失败样本（skip）不进回填队列"""
        from core.analysis_history import backfill_analyses, save_failure
        save_failure("000001", "err")
        stats = backfill_analyses(price_fetcher=lambda c, s, e: _fake_kline(35))
        assert stats["checked"] == 0


class TestAnalysisStats:

    def test_win_rate_excludes_skip(self, temp_store):
        """胜率只统计已回填样本；skip（分析失败）不参与分母"""
        # ⚠️ 三个样本必须用**不同代码**（同日同票只留一行，见唯一索引）。
        for i, (ret, label) in enumerate(((0.05, "看多"), (-0.02, "看多"), (None, "看空"))):
            fields = {"backfill_status": "done"}
            if ret is not None:
                fields["ret_5d"] = ret
            rid = db_store.insert_analysis({
                "analysis_date": _days_ago(10), "stock_code": f"60000{i}",
                "decision_label": label})
            db_store.update_analysis_backfill(rid, fields)
        db_store.insert_analysis({"analysis_date": _days_ago(10),
                                  "stock_code": "000001",
                                  "decision_label": "分析失败",
                                  "backfill_status": "skip"})
        stats = db_store.analysis_stats(recent_days=30)
        assert stats["total"] == 4
        assert stats["bull_samples"] == 2
        assert stats["bull_win_rate"] == pytest.approx(0.5)
        assert stats["bear_samples"] == 0           # 看空样本 ret_5d 为 NULL 不计
        assert stats["pending_backfill"] == 0


# ================================================================
# 调度接线
# ================================================================

def _block_real_db(monkeypatch):
    """掐断 `run_job` 的 `runs` 落库，避免测试写**真实库**。

    ⚠️ 2026-10-28 实测：本文件两个用例直接调 `sched.run_job("backfill", "测试回填")`，
       而 `run_backfill_job` 的 `finally` 里会 `store.insert_run(...)` ⇒ 每跑一次全量，
       生产库 `runs` 表就多两行 `job:backfill_analysis`（`desc="测试回填"`）。
       后果不只是脏数据：**运维口径里的"今日跑了几次/全成功"被测试灌水**，
       查"为什么今天多跑了两轮"会查到测试头上。
    """
    from db import store as _dbstore
    monkeypatch.setattr(_dbstore, "insert_run", lambda *a, **k: None)


class TestSchedulerWiring:

    def test_backfill_job_registered(self):
        from core.scheduler import JOBS
        jobs = {j[0]: j for j in JOBS}
        assert "backfill_analysis" in jobs
        assert jobs["backfill_analysis"][2] == "backfill"

    def test_backfill_skips_non_trading_day(self, monkeypatch):
        import core.scheduler as sched
        monkeypatch.setattr(sched, "is_trading_day", lambda: False)
        _block_real_db(monkeypatch)
        called = []

        def _no_call(*a, **k):
            called.append(1)
            return {}

        import core.analysis_history as ah
        monkeypatch.setattr(ah, "backfill_analyses", _no_call)
        sched.run_job("backfill", "测试回填")
        assert called == []                          # 非交易日直接跳过

    def test_run_backfill_job_dispatches(self, monkeypatch):
        import core.scheduler as sched
        monkeypatch.setattr(sched, "is_trading_day", lambda: True)
        _block_real_db(monkeypatch)
        import core.analysis_history as ah
        result = {"called": False}

        def _fake_backfill(limit=100, include_failed=False, price_fetcher=None):
            result["called"] = True
            result["limit"] = limit
            return {"checked": 0}

        monkeypatch.setattr(ah, "backfill_analyses", _fake_backfill)
        sched.run_job("backfill", "测试回填")
        assert result["called"] and result["limit"] == 100


# ================================================================
# 决策标签：终审否决不能被显示成"观望"（2026-10-09）
# ================================================================
# 实测 id=11：trade_plan.direction = "观望"，final_decision = "**驳回。** 仓位严重超限…"。
# 旧逻辑"优先取 direction" → 历史列表显示"观望"，用户看到的是**没表态**，
# 而实际结论是**否掉**。且 direction 为"看多"而终审驳回时，该样本还会按"看多"
# 计入胜率统计（`store` 的胜率口径按 decision_label 分组）。

class TestExtractDecision:
    @staticmethod
    def _dec(*a, **kw):
        # 局部导入：与本文件其余用例一致（模块级不 import core.analysis_history）
        from core.analysis_history import _extract_decision
        return _extract_decision(*a, **kw)

    def test_rejected_overrides_watchful_plan(self):
        assert self._dec("**驳回。** 仓位严重超限…",
                                    {"direction": "观望"}) == "驳回"

    def test_rejected_overrides_bullish_plan(self):
        """❌ 旧行为：direction=看多 → label=看多 → 被算进胜率；实际计划根本没执行"""
        assert self._dec("最终决策：驳回，风控红线已触发",
                                    {"direction": "看多"}) == "驳回"

    def test_approval_keeps_direction_label(self):
        assert self._dec("**批准。** 风控意见已逐项验证",
                                    {"direction": "看多"}) == "看多"

    def test_modify_keeps_direction_label(self):
        assert self._dec("**最终决策：需修改**",
                                    {"direction": "观望"}) == "观望"

    def test_override_mentioning_rejection_is_not_a_rejection(self):
        """终审有权在写明依据后覆盖风控 —— "批准。风控曾以超限为由驳回" 不是驳回"""
        assert self._dec("批准。风控曾以超限为由驳回，但实测阈值是 25%。",
                                    {"direction": "看多"}) == "看多"

    def test_empty_decision_stays_empty(self):
        """链被阻断（没有决策文本）→ 空串，不能伪造一个标签"""
        assert self._dec("", {}) == ""


# ================================================================
# 回填：把「正常等待」与「取数坏了」分开（2026-10-09）
# ================================================================
# 起因：用户报「深度分析一次回填都没有」。日志只有 `checked=7, skipped=7, errors=[]`
# —— 两类完全不同的原因在日志与界面上**长得一模一样**，只能手工复现取数才能区分。
# 实测结论：12 行**全部**是"窗口未走完"（中秋 9/25 + 国庆 10/1~10/7 把 T+5 的
# 交易日跨度拉长，而数据源最新只到 10-08）。所以这不是故障，但**必须看得出来**。

def _build_fake_days():
    """2026-06-01 ~ 2026-12-31 的工作日，**扣掉中秋(09-25)与国庆(10-01~10-07)**。

    交易日必须足够多：T+20 需要第 21 根，日历太短会让 `_nth_trading_day` 返回 None
    → 进度被判成「无法判定」（首版夹具只放了 14 天，正是这个错）。
    """
    from datetime import date as _date
    holiday = {"2026-09-25"} | {f"2026-10-0{d}" for d in range(1, 8)}
    out, cur, end = set(), _date(2026, 6, 1), _date(2026, 12, 31)
    while cur <= end:
        ds = cur.strftime("%Y-%m-%d")
        if cur.weekday() < 5 and ds not in holiday:
            out.add(ds)
        cur += _date.resolution
    return out


class _FakeCal:
    """固定日历（用来把"是否节假日"从测试里解耦，不依赖本机缓存状态）。"""

    DAYS = _build_fake_days()

    @classmethod
    def install(cls, monkeypatch):
        """只替换日历判定；`today` 一律用显式参数传（不去 patch `datetime.now`）"""
        from data import trading_calendar as tc
        monkeypatch.setattr(tc, "is_trading_day",
                            lambda d=None: str(d)[:10] in cls.DAYS)


class TestBackfillSkipReasons:
    def test_window_not_finished_counts_as_not_ready_not_error(self, temp_store):
        """⚠️ 窗口未走完 → not_ready，**且不能写进 errors**（那不是故障）"""
        from core.analysis_history import backfill_analyses
        self._insert_for_backfill(3)
        stats = backfill_analyses(price_fetcher=lambda c, s, e: _fake_kline(3))
        assert stats["skipped"] == 1
        assert stats["not_ready"] == 1
        assert stats["unavailable"] == 0
        assert stats["errors"] == []

    def test_empty_fetch_is_unavailable_and_recorded(self, temp_store):
        """取数为空原先**完全静默**（skipped+1、无 errors）→ 现在必须留痕"""
        from core.analysis_history import backfill_analyses
        self._insert_for_backfill(30)
        stats = backfill_analyses(price_fetcher=lambda c, s, e: None)
        assert stats["skipped"] == 1
        assert stats["unavailable"] == 1
        assert stats["not_ready"] == 0
        assert stats["errors"] and "取数" in stats["errors"][0]

    def test_unnormalizable_frame_is_unavailable(self, temp_store):
        """取到 DataFrame 但缺 date/close 列（normalize_kline → None）同样要留痕"""
        import pandas as pd
        from core.analysis_history import backfill_analyses
        self._insert_for_backfill(30)
        stats = backfill_analyses(
            price_fetcher=lambda c, s, e: pd.DataFrame({"foo": [1, 2]}))
        assert stats["unavailable"] == 1 and stats["errors"]

    @staticmethod
    def _insert_for_backfill(days_ago: int):
        return db_store.insert_analysis({
            "analysis_date": _days_ago(days_ago), "stock_code": "600519",
            "stock_name": "贵州茅台", "decision_label": "看多",
            "trade_plan": {"direction": "看多"}})


class TestBackfillProgress:
    """`backfill_progress()` —— 不联网，用交易日历算「还差几个交易日」"""

    def test_nth_trading_day_skips_holidays(self, monkeypatch):
        """⚠️ 必须走**真实交易日历**，不能退回「周一至周五」枚举。

        实测踩到：`trading_calendar.trading_days(s, e)` 在 `e` 超出日历覆盖范围时
        **静默退回工作日枚举**（不报 degraded），于是 10-01~10-07 国庆假期被当成交易日，
        算出来的「可回填日」落在过去 —— 比不报更误导。
        `_nth_trading_day` 改为逐日 `is_trading_day` 推进，这里用固定日历钉住行为。
        """
        _FakeCal.install(monkeypatch)
        assert ah._nth_trading_day("2026-10-01", 0) == "2026-10-08"   # 假期顺延
        assert ah._nth_trading_day("2026-09-30", 1) == "2026-10-08"
        assert ah._nth_trading_day("2026-09-30", 2) == "2026-10-09"
        assert ah._nth_trading_day("2026-09-24", 5) == "2026-10-09"   # T+5 用第 6 根

    def test_reports_waiting_rows_with_eta(self, temp_store, monkeypatch):
        from core.analysis_history import backfill_progress
        _FakeCal.install(monkeypatch)
        db_store.insert_analysis({"analysis_date": "2026-09-24", "stock_code": "600519",
                                  "stock_name": "贵州茅台", "decision_label": "看多",
                                  "trade_plan": {}})
        p = backfill_progress(today="2026-10-09")
        assert p["pending"] == 1 and p["waiting"] == 1
        row = p["rows"][0]
        assert row["ready"][5] == "2026-10-09"
        assert row["missing"][5] == 1          # 基准 10-08（不含今天）→ 还差 1 个交易日
        assert row["stage"] == "等 T+5"
        assert "2026-10-09" in row["reason"]
        assert "还差 1 个交易日" in p["summary"]

    def test_baseline_excludes_today(self, temp_store, monkeypatch):
        """⚠️ 保守基准：不含今天 —— 当日 K 线通常收盘后/次日才有。

        把"今天"当已有数据会报出「已可回填」但实际仍写不进去（**新的假信号**）。
        """
        from core.analysis_history import backfill_progress
        _FakeCal.install(monkeypatch)
        db_store.insert_analysis({"analysis_date": "2026-09-24", "stock_code": "600519",
                                  "stock_name": "贵州茅台", "decision_label": "看多",
                                  "trade_plan": {}})
        p = backfill_progress(today="2026-10-09")
        assert p["last_trading_day"] == "2026-10-08"      # 不是 10-09
        assert p["ready_now"] == 0

    def test_old_row_is_ready(self, temp_store, monkeypatch):
        """分析日足够早（T+5 与 T+20 都走完）→ 可回填"""
        from core.analysis_history import backfill_progress
        _FakeCal.install(monkeypatch)
        db_store.insert_analysis({"analysis_date": "2026-08-20", "stock_code": "000001",
                                  "stock_name": "平安银行", "decision_label": "看多",
                                  "trade_plan": {}})
        p = backfill_progress(today="2026-10-09")
        row = p["rows"][0]
        assert row["stage"] == "可回填" and row["missing"][5] == 0
        assert row["ready"][20] <= p["last_trading_day"]
        assert p["ready_now"] == 1

    def test_does_not_touch_the_network(self, temp_store, monkeypatch):
        """⚠️ 页面每次渲染都会调它 —— 绝不能拉 K 线（12 只票 × 每个页面刷新）"""
        from core.analysis_history import backfill_progress
        _FakeCal.install(monkeypatch)
        db_store.insert_analysis({"analysis_date": "2026-09-17", "stock_code": "000001",
                                  "stock_name": "平安银行", "decision_label": "看多",
                                  "trade_plan": {}})
        monkeypatch.setattr(ah, "default_price_fetcher",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                AssertionError("回填进度不应联网")))
        assert backfill_progress(today="2026-10-09")["pending"] == 1

    def test_empty_when_nothing_pending(self, temp_store, monkeypatch):
        from core.analysis_history import backfill_progress
        _FakeCal.install(monkeypatch)
        p = backfill_progress(today="2026-10-09")
        assert p["pending"] == 0 and p["rows"] == []
        assert "0 条待回填" in p["summary"]


class TestPartialReadyAndWindow:
    """2026-10-10 用户报"600519 第一次回填该在 9 号，为什么没回填"。

    实测链路：分析日 09-24（09-25 中秋休市）→ 第 6 根 K 线是 **10-09**；
    10-09 15:35 那次任务跑时**当天 K 线还没出**（日志 `not_ready: 12`），
    而下一个回填窗口是 10-12（周一）—— 周末任务不跑。
    但页面上既看不到"已有几条可以写 T+5"，也看不到"下个窗口何时"，
    只剩"一直没回填"这一个观感。本类锁住这两条**可见性**。
    """

    def _one(self, date="2026-09-24"):
        db_store.insert_analysis({"analysis_date": date, "stock_code": "600519",
                                  "stock_name": "贵州茅台", "decision_label": "观望",
                                  "trade_plan": {}})

    def test_partial_ready_counted_separately(self, temp_store, monkeypatch):
        """T+5 已可算、T+20 未到 → 单独计 `partial_ready`，不再混进「还在等」"""
        _FakeCal.install(monkeypatch)
        from core.analysis_history import backfill_progress
        self._one()
        p = backfill_progress(today="2026-10-10")        # 周六 → 基准 10-09
        assert p["rows"][0]["stage"] == "可写 T+5"       # 不是"等 T+20"
        assert p["partial_ready"] == 1
        assert p["ready_now"] == 0                       # T+20 未到 → 不计入"完全可回填"
        assert "现在就能写入" in p["summary"]

    def test_reason_says_which_kline_and_when(self, temp_store, monkeypatch):
        """「还差 N 个交易日」必须说清是**第几根 K 线、哪天收盘后可得**。

        原话术"预计 2026-10-09 可回填"是**不准的** —— 10-09 15:35 任务跑时
        当天 K 线还没有，于是它在"预计可回填日"当天仍然 not_ready。
        """
        _FakeCal.install(monkeypatch)
        from core.analysis_history import backfill_progress
        self._one()
        p = backfill_progress(today="2026-10-09")        # 交易日当天 → 基准 10-08
        assert p["rows"][0]["missing"][5] == 1
        assert "第 6 根 K 线" in p["rows"][0]["reason"]
        assert "收盘后可得" in p["rows"][0]["reason"]

    def test_next_window_skips_weekend_and_names_today(self, monkeypatch):
        _FakeCal.install(monkeypatch)
        from datetime import datetime as _dt
        from core.analysis_history import _next_backfill_window
        assert _next_backfill_window(now=_dt(2026, 10, 10, 15, 3)) == "2026-10-12 15:35"
        assert "（今天）" in _next_backfill_window(now=_dt(2026, 10, 12, 9, 0))
        assert _next_backfill_window(now=_dt(2026, 10, 12, 16, 0)) == "2026-10-13 15:35"

    def test_summary_states_task_runs_on_trading_days_only(self, temp_store, monkeypatch):
        _FakeCal.install(monkeypatch)
        from core.analysis_history import backfill_progress
        self._one()
        p = backfill_progress(today="2026-10-10")
        assert "交易日 15:35" in p["summary"]
        assert p["next_backfill_window"]

    def test_already_written_row_waits_for_t20(self, temp_store, monkeypatch):
        """已写入 T+5（partial）的行显示「等 T+20」，不能说成「可写」。

        600519 就是这种状态：T+5 已入库（+2.1%），现在等的是 T+20；
        若还显示"可写"，用户会以为回填没生效。
        """
        _FakeCal.install(monkeypatch)
        from core.analysis_history import backfill_progress
        aid = db_store.insert_analysis({"analysis_date": "2026-09-24", "stock_code": "600519",
                                        "stock_name": "贵州茅台", "decision_label": "观望",
                                        "trade_plan": {}, "backfill_status": "partial"})
        db_store.update_analysis_backfill(aid, {"ret_5d": 0.021, "backfill_status": "partial"})
        p = backfill_progress(today="2026-10-10")
        row = p["rows"][0]
        assert row["has_ret_5d"] is True
        assert row["stage"] == "等 T+20"
        assert p["partial_ready"] == 0
