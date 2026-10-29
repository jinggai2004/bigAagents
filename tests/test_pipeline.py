"""全流程编排器与调度器单元测试（全 mock，不依赖网络/LLM）"""
import json
import os
from datetime import datetime

import pytest
from unittest.mock import patch, MagicMock

from core.pipeline import DailyPipeline


FAKE_CANDIDATES = [
    {"股票代码": "000001", "股票名称": "平安银行", "score": 82.5, "策略": "金融",
     "factors": {"资金": {"score": 20, "max": 25}}},
    {"股票代码": "600519", "股票名称": "贵州茅台", "score": 78.0, "策略": "消费",
     "factors": {}},
]


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """SQLite 指向临时文件。

    必要性：llm 模式下 `_stage_analyze_llm` 会把分析历史落库（save_analysis），
    不隔离的话单测会往真实的 data/portfolio.db 里写测试数据。
    """
    from tests.conftest import use_template_db
    db_path = use_template_db(tmp_path, monkeypatch, "pipeline.db")
    yield


@pytest.fixture
def pipeline(tmp_path):
    return DailyPipeline(output_dir=str(tmp_path / "daily_picks"))


class TestStageScreen:
    def test_stock_codes_skip_screening(self, pipeline):
        picks, signal, reason = pipeline._stage_screen(None, 5, stock_codes=["300001", "600519"])
        assert [p["股票代码"] for p in picks] == ["300001", "600519"]
        assert reason == ""

    def test_screener_success(self, pipeline):
        mock_screener = MagicMock()
        mock_screener.run.return_value = {
            "candidates": FAKE_CANDIDATES, "market_signal": {"state": "bull"}, "skipped": False,
        }
        with patch("core.screener.StockScreener", return_value=mock_screener):
            picks, signal, reason = pipeline._stage_screen(None, 5)
        assert len(picks) == 2
        assert signal == {"state": "bull"}
        assert reason == ""

    def test_stage_timings_are_recorded(self, pipeline):
        """⭐ 分阶段耗时（2026-10-21）：打分与落库必须分开计时。

        为什么单独测：`elapsed_seconds` 只有一个总数。扩池到 300 只时，若不分开
        计时就无法回答"是打分慢还是写库慢" —— 而这决定了该优化哪一边。
        """
        mock_screener = MagicMock()
        mock_screener.run.return_value = {
            "candidates": FAKE_CANDIDATES, "market_signal": {}, "skipped": False,
        }
        with patch("core.screener.StockScreener", return_value=mock_screener):
            pipeline._stage_screen(None, 5)
        assert "screen.run" in pipeline._timings
        assert pipeline._timings["screen.run"] >= 0
        assert pipeline._timings["screen.candidates"] == len(FAKE_CANDIDATES)

    def test_stage_timings_reach_saved_report(self, pipeline):
        """落盘 JSON 里必须能看到 `stage_timings` —— 对比要靠它，不是靠日志。"""
        mock_screener = MagicMock()
        mock_screener.run.return_value = {
            "candidates": FAKE_CANDIDATES, "market_signal": {}, "skipped": False,
        }
        with patch("core.screener.StockScreener", return_value=mock_screener):
            report = pipeline.run(mode="screen", max_stocks=2)
        assert "stage_timings" in report, "report 里没有 stage_timings，扩池前后无法对比"
        assert report["stage_timings"].get("screen.run") is not None

    def test_screener_bear_skip(self, pipeline):
        mock_screener = MagicMock()
        mock_screener.run.return_value = {
            "candidates": [], "market_signal": {"state": "bear"},
            "skipped": True, "reason": "熊市跳过",
        }
        with patch("core.screener.StockScreener", return_value=mock_screener):
            picks, signal, reason = pipeline._stage_screen(None, 5)
        assert picks == []
        assert reason == "熊市跳过"


class TestPipelineRun:
    def test_screen_mode_end_to_end(self, pipeline):
        mock_screener = MagicMock()
        mock_screener.run.return_value = {
            "candidates": FAKE_CANDIDATES, "market_signal": {}, "skipped": False,
        }
        with patch("core.screener.StockScreener", return_value=mock_screener):
            report = pipeline.run(mode="screen")

        assert report["skipped"] is False
        assert len(report["picks"]) == 2
        assert report["backtest"] is None
        assert report["analysis"][0]["decision"] == "规则模式：LLM分析未启用"
        assert not report["errors"]
        # 报告落盘
        files = os.listdir(pipeline.output_dir)
        assert any(f.startswith("daily_") for f in files)
        assert "latest.json" in files
        with open(os.path.join(pipeline.output_dir, "latest.json"), encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["picks"] == report["picks"]

    def test_bear_market_skip_persists_report(self, pipeline):
        mock_screener = MagicMock()
        mock_screener.run.return_value = {"candidates": [], "skipped": True, "reason": "熊市"}
        with patch("core.screener.StockScreener", return_value=mock_screener):
            report = pipeline.run(mode="rule")
        assert report["skipped"] is True
        assert report["backtest"] is None
        assert os.path.exists(os.path.join(pipeline.output_dir, "latest.json"))

    def test_no_candidates_recorded_as_error(self, pipeline):
        mock_screener = MagicMock()
        mock_screener.run.return_value = {"candidates": [], "skipped": False}
        with patch("core.screener.StockScreener", return_value=mock_screener):
            report = pipeline.run(mode="screen")
        assert any("无候选" in e for e in report["errors"])

    def test_screening_exception_recorded(self, pipeline):
        with patch.object(DailyPipeline, "_stage_screen", side_effect=RuntimeError("网络异常")):
            report = pipeline.run(mode="screen")
        assert any("选股失败" in e for e in report["errors"])
        assert "latest.json" in os.listdir(pipeline.output_dir)

    def test_market_signal_serialized_as_dict_not_repr(self, pipeline):
        """⭐ C1（2026-10-25）：MarketSignal 落盘必须是 dict，不能是 repr 串。

        根因：`json.dump(default=str)` 把 dataclass 打成
        `MarketSignal(total_score=28.4, ...)` 文本 ⇒ 新前端 /api 透传后无字段可读，
        每个消费方各自长一套正则解析（ai_daily_round 即为此而生）。
        内存里的 report 不被篡改（UI 的 render_screen_result 要属性访问）。
        """
        from core.market_detector import MarketSignal
        sig = MarketSignal(total_score=28.4, market_state="偏弱震荡", signal="卖出/减仓",
                           position_limit=0.2, strategy_bias="极端防御",
                           operation_advice="减仓观望", scores={"趋势": 12.0},
                           indicators={"hs300_ma20_deviation": 1.5},
                           risk_warnings=["北向连续流出"], timestamp="2026-10-25 15:30:00")
        picks = [dict(c) for c in FAKE_CANDIDATES]
        with patch.object(DailyPipeline, "_stage_screen", return_value=(picks, sig, "")):
            report = pipeline.run(mode="screen")
        assert isinstance(report["market_signal"], MarketSignal)
        with open(os.path.join(pipeline.output_dir, "latest.json"), encoding="utf-8") as f:
            saved = json.load(f)
        assert isinstance(saved["market_signal"], dict)
        assert saved["market_signal"]["total_score"] == 28.4
        assert saved["market_signal"]["market_state"] == "偏弱震荡"
        assert saved["market_signal"]["position_limit"] == 0.2

    def test_rule_mode_runs_backtest(self, pipeline):
        mock_screener = MagicMock()
        mock_screener.run.return_value = {"candidates": FAKE_CANDIDATES, "skipped": False}

        fake_result = MagicMock()
        fake_result.to_dict.return_value = {"total_return": 1.5, "total_trades": 3}

        mock_generator = MagicMock()
        mock_generator.generate_signals.return_value = [{"code": "000001", "direction": "买入"}]
        mock_generator._data_cache = {"000001": object()}
        mock_engine = MagicMock()
        mock_engine.run.return_value = fake_result
        mock_engine.save_report.return_value = "/tmp/fake.json"

        with patch("core.screener.StockScreener", return_value=mock_screener), \
             patch("scripts.backtest_rule_based.RuleSignalGenerator", return_value=mock_generator), \
             patch("core.backtest.BacktestEngine", return_value=mock_engine), \
             patch("data.benchmark.get_benchmark_nav", return_value=None):
            report = pipeline.run(mode="rule", backtest_days=30)

        assert report["backtest"] is not None
        assert report["backtest"]["total_return"] == 1.5
        assert report["backtest"]["window"][0] <= report["backtest"]["window"][1]
        # 缓存注入与报告保存被调用
        mock_engine.inject_kline_cache.assert_called_once()
        mock_engine.save_report.assert_called_once()

    def test_backtest_no_signals(self, pipeline):
        mock_screener = MagicMock()
        mock_screener.run.return_value = {"candidates": FAKE_CANDIDATES, "skipped": False}
        mock_generator = MagicMock()
        mock_generator.generate_signals.return_value = []

        with patch("core.screener.StockScreener", return_value=mock_screener), \
             patch("scripts.backtest_rule_based.RuleSignalGenerator", return_value=mock_generator):
            report = pipeline.run(mode="rule", backtest_days=30)
        assert report["backtest"] is not None
        assert "无交易信号" in report["backtest"]["note"]
        assert not report["errors"]

    def test_llm_mode_invokes_graph(self, pipeline):
        """llm 模式应逐只调用 analysis_graph 并汇总最终决策"""
        mock_screener = MagicMock()
        mock_screener.run.return_value = {"candidates": FAKE_CANDIDATES, "skipped": False}

        def fake_stream(state):
            yield {"data_fetcher": {"stock_info": {"name": state["stock_code"]}}}
            final = {"final_decision": "买入", "trade_plan": {"价格": 10.0},
                     "risk_assessment": "低风险", "stock_name": "测试股"}
            yield {"portfolio_manager": final}

        with patch("core.screener.StockScreener", return_value=mock_screener), \
             patch("core.graph_builder.analysis_graph") as mock_graph, \
             patch("scripts.backtest_rule_based.RuleSignalGenerator"), \
             patch("core.backtest.BacktestEngine"):
            mock_graph.stream.side_effect = lambda state: fake_stream(state)
            report = pipeline.run(mode="llm", backtest_days=7)

        assert len(report["analysis"]) == 2
        assert report["analysis"][0]["decision"] == "买入"
        assert report["analysis"][0]["trade_plan"] == {"价格": 10.0}

        # 接线验证：llm 模式的每只候选都要落一条分析历史（T+N 回看的数据源）
        from db import store as db_store
        rows = db_store.list_analyses()
        assert len(rows) == 2
        assert {r["stock_code"] for r in rows} == {"000001", "600519"}
        assert {r["source"] for r in rows} == {"pipeline"}


class TestScheduler:
    def test_job_registration(self):
        from core.scheduler import create_scheduler, JOBS
        scheduler = create_scheduler()
        jobs = {j.id: j for j in scheduler.get_jobs()}
        # 显式清单是真正的契约（防止任务被误删）；条数用 len(JOBS) 而不是写死数字，
        # 免得每加一个任务都要来这里改一次数字（但改清单要是有意的）。
        assert set(jobs) == {
            "morning_screen", "noon_screen", "afternoon_full",
            "backfill_analysis", "backfill_screen", "daily_review",
            "hourly_news", "intraday_watch",
            "snapshot_flow",           # 批次 22：盘后全市场资金流快照落库
            "risk_check",              # 2026-09-28：持仓风控巡检（14:50）
            "news_review",             # 2026-10-09：新闻行业利好/利空评估（15:55）
            "ai_daily_round",          # 2026-10-14：AI交易联赛·盘后决策轮（15:58，会花 token）
            "ai_settle_open",          # 2026-10-14：联赛 T+1 开盘结算（09:32，零 LLM）
            "ai_settle_close",         # 2026-10-14：联赛收盘结算·日内卖出（15:05，零 LLM）
            "ai_mark_nav",             # 2026-10-15：联赛当日净值（15:52，晚于快照、早于决策轮）
            "evening_analysis",        # 2026-10-19：盘后批量深度分析（18:10，DeepSeek 空闲时段）
            "ai_midday_round",         # 2026-10-23：联赛午间决策轮（13:05，与盘后轮各占 1 次配额）
            "sentiment_ref",           # 2026-10-23：市场情绪参照采集（15:45，零 LLM，只为攒历史基准）
        }
        assert len(jobs) == len(JOBS) == 18

        def field_value(trigger, name):
            for f in trigger.fields:
                if f.name == name:
                    return str(f)
            return ""

        assert field_value(jobs["morning_screen"].trigger, "hour") == "9"
        assert field_value(jobs["morning_screen"].trigger, "minute") == "25"
        assert field_value(jobs["noon_screen"].trigger, "hour") == "11"
        assert field_value(jobs["afternoon_full"].trigger, "hour") == "15"
        assert field_value(jobs["afternoon_full"].trigger, "minute") == "30"

    def test_ai_mark_nav_runs_after_snapshot_before_round(self):
        """⭐ 记净值的排期是**语义要求**，不是随便挑的时间。

        2026-10-15 实测事故：净值原本挂在 15:05，早于当日日线可用时间 ⇒ 取不到收盘价
        ⇒ 全量退化成"成本价兜底"⇒ 假净值喂给 AI 复盘（"净值不变、无盈亏"）。
        ⇒ 必须在**快照(15:40)之后**（快照含当日收盘价）、**决策轮(15:58)之前**
          （决策轮私有段要读当日净值与持仓盈亏）。
        """
        from core.scheduler import create_scheduler

        def hhmm(job_id):
            from core.scheduler import JOBS
            for jid, cron, _mode, _desc in JOBS:
                if jid == job_id:
                    return cron["hour"] * 60 + cron["minute"]
            raise AssertionError(f"JOBS 里没有 {job_id}")

        assert any(j.id == "ai_mark_nav" for j in create_scheduler().get_jobs())
        assert hhmm("snapshot_flow") < hhmm("ai_mark_nav") < hhmm("ai_daily_round"), (
            "记净值必须在快照之后、决策轮之前")

    def test_weekday_only(self):
        """除新闻采集外，所有任务都限交易日（mon-fri）。

        ⚠️ `hourly_news` 例外（2026-10-18 周日实测）：新闻与交易日无关——
        周末也有财经快讯，采集零 LLM 成本、无副作用；一律 mon-fri 会让
        页 12 周末整天空窗（最后一条停在工作日 22:31，24h 窗口恰好滑出）。
        `news_review`（评估，涉及 LLM 成本）仍 mon-fri。
        """
        from core.scheduler import create_scheduler, JOBS
        scheduler = create_scheduler()
        news_job_ids = {jid for jid, _c, m, _d in JOBS if m == "news"}
        for job in scheduler.get_jobs():
            if job.id in news_job_ids:
                assert "mon-fri" not in str(job.trigger), \
                    f"{job.id} 应每天跑（新闻与交易日无关）"
            else:
                assert "mon-fri" in str(job.trigger)

    def test_run_job_survives_pipeline_crash(self, tmp_path, caplog):
        """pipeline 崩溃不应炸掉调度线程"""
        from core.scheduler import run_job
        # 注意 patch 目标是 core.scheduler 绑定的名字（from-import 绑定）
        # 同时固定交易日判定，避免单测触发日历网络请求
        with patch("core.scheduler.is_trading_day", return_value=True), \
             patch("core.scheduler.DailyPipeline") as mock_cls:
            mock_cls.return_value.run.side_effect = RuntimeError("boom")
            run_job("screen", "测试任务")  # 不应抛出
        assert True

    # ===== Phase 1：交易日保护 =====

    def test_run_job_skips_non_trading_day(self):
        """cron 只排除周末，法定节假日必须由交易日历拦下。

        否则节假日照样选股并推送，产出的"今日选股/复盘"与交易日不可区分。
        """
        from core.scheduler import run_job
        reviewed = []
        with patch("core.scheduler.is_trading_day", return_value=False), \
             patch("core.scheduler.DailyPipeline") as mock_cls, \
             patch("core.scheduler.run_review_job", side_effect=lambda d: reviewed.append(d)):
            run_job("rule", "盘后全流程")
            run_job("review", "每日复盘")
        mock_cls.assert_not_called()
        assert reviewed == []

    def test_run_job_runs_on_trading_day(self):
        from core.scheduler import run_job
        with patch("core.scheduler.is_trading_day", return_value=True), \
             patch("core.scheduler.DailyPipeline") as mock_cls:
            run_job("rule", "盘后全流程")
        assert mock_cls.return_value.run.called

    def test_news_job_not_gated_by_trading_day(self):
        """新闻采集与交易日无关，节假日也应继续跑"""
        from core.scheduler import run_job
        fetched = []
        with patch("core.scheduler.is_trading_day", return_value=False), \
             patch("core.scheduler.run_news_job", side_effect=lambda d: fetched.append(d)):
            run_job("news", "新闻采集")
        assert fetched == ["新闻采集"]
