# -*- coding: utf-8 -*-
"""择时口径统一（2026-10-20）的守护测试。

**事故**：深度分析图的 `timing_node` 用 `TimingEngine`，而选股/复盘/`agent_tools`
用 `MarketDetector`。同一天两个结论：

    MarketDetector  37.9 分 → 偏弱震荡，仓位上限 20%   （选股据此选票）
    TimingEngine    16.9 分 → bear，     仓位上限 10%   （分析据此逐个否掉）

结果"选出来的票全被分析否掉"，一整批（8 只 / 23 分钟 / ¥0.41）白跑。

统一到 `MarketDetector` 后，两端用同一把尺子。本文件锁住三件事：
1. 映射只在一处（`_STATE_TO_REGIME`），且不偷偷把"偏弱震荡"写成 bear；
2. 默认走 market_detector，可用 `BIGA_TIMING_SOURCE=timing_engine` 一键回滚；
3. 降级链必须**留下痕迹**（两套都失败时不能静默变成"市场中性"）。
"""
from unittest.mock import MagicMock

import pytest

from core.market_detector import MarketSignal
from core.timing import TimingSignal, from_market_signal


def _ms(state="偏弱震荡", score=37.9, limit=0.2) -> MarketSignal:
    return MarketSignal(
        total_score=score, market_state=state, signal="卖出/减仓",
        position_limit=limit, strategy_bias="防御型",
        operation_advice="市场偏弱，大幅减仓",
        scores={"hs300_ma20_deviation": {"score": 3, "max_score": 10}},
        indicators={}, risk_warnings=[], timestamp="2026-10-20 20:00:00")


# ================================================================
# 1. 映射（唯一来源）
# ================================================================

class TestStateMapping:

    @pytest.mark.parametrize("state,regime", [
        ("牛市", "bull"),
        ("偏强震荡", "bull"),
        ("中性震荡", "range"),
        ("偏弱震荡", "range"),
        ("熊市", "bear"),
    ])
    def test_maps_each_state(self, state, regime):
        assert from_market_signal(_ms(state)).regime == regime

    def test_weak_range_is_not_silently_turned_into_bear(self):
        """⭐ 关键：偏弱震荡映射成 range，不能顺手写成 bear。

        把 37.9 改写成 bear 等于在适配层偷改分数，两个引擎的语义就又在
        别处分叉了——那正是本次要消灭的东西。
        """
        sig = from_market_signal(_ms("偏弱震荡"))
        assert sig.regime == "range"
        assert sig.total_score == 37.9
        assert sig.position_limit == 0.2

    def test_unknown_state_falls_back_to_range_and_says_so(self):
        sig = from_market_signal(_ms("横盘整固"))     # 不在枚举里
        assert sig.regime == "range"
        assert "未能识别" in sig.summary

    def test_breakdown_and_source_are_preserved(self):
        sig = from_market_signal(_ms())
        assert sig.source == "market_detector"
        assert "hs300_ma20_deviation" in sig.breakdown
        assert "偏弱震荡" in sig.summary


# ================================================================
# 2. timing_node 的选源与降级
# ================================================================

class TestTimingNode:

    def test_defaults_to_market_detector(self, monkeypatch):
        import core.graph_builder as G
        monkeypatch.delenv(G.ENV_TIMING_SOURCE, raising=False)
        monkeypatch.setattr("core.market_detector.MarketDetector.detect",
                            lambda self: _ms())
        out = G.timing_node({})
        assert out["timing_signal"]["source"] == "market_detector"
        assert out["position_limit"] == 0.2
        assert out["timing_signal"]["regime"] == "range"

    def test_can_roll_back_to_timing_engine(self, monkeypatch):
        """回滚开关：改环境变量即可，不需要动代码。"""
        import core.graph_builder as G
        from core.timing import get_timing_engine
        monkeypatch.setenv(G.ENV_TIMING_SOURCE, "timing_engine")
        monkeypatch.setattr(get_timing_engine(), "detect", lambda: TimingSignal(
            regime="bear", position_limit=0.1, total_score=16.9,
            breakdown={}, summary="旧口径"))
        out = G.timing_node({})
        assert out["timing_signal"]["source"] == "timing_engine"
        assert out["timing_signal"]["regime"] == "bear"

    def test_market_detector_failure_falls_back_to_timing_engine(self, monkeypatch):
        import core.graph_builder as G
        from core.timing import get_timing_engine
        monkeypatch.delenv(G.ENV_TIMING_SOURCE, raising=False)

        def boom(self):
            raise RuntimeError("网络断了")

        monkeypatch.setattr("core.market_detector.MarketDetector.detect", boom)
        monkeypatch.setattr(get_timing_engine(), "detect", lambda: TimingSignal(
            regime="range", position_limit=0.3, total_score=50,
            breakdown={}, summary="退回引擎"))
        out = G.timing_node({})
        assert out["timing_signal"]["source"] == "timing_engine"

    def test_both_engines_failing_leaves_a_visible_trace(self, monkeypatch):
        """⭐ 两套都失败时不能静默变成"市场中性"——必须写成 fallback 且写明原因。"""
        import core.graph_builder as G
        from core.timing import get_timing_engine
        monkeypatch.delenv(G.ENV_TIMING_SOURCE, raising=False)

        def boom(self):
            raise RuntimeError("网络断了")

        def boom2(*a, **k):
            raise RuntimeError("也断了")

        monkeypatch.setattr("core.market_detector.MarketDetector.detect", boom)
        monkeypatch.setattr(get_timing_engine(), "detect", boom2)
        out = G.timing_node({})
        assert out["timing_signal"]["source"] == "fallback"
        assert "不可用" in out["timing_signal"]["summary"]
        assert out["position_limit"] == 0.3


# ================================================================
# 3. 落库留痕：market_score_source 必须跟着分数一起存
# ================================================================

class TestScoreSourceIsPersisted:

    def test_save_analysis_records_source(self, tmp_path, monkeypatch):
        from tests.conftest import use_template_db
        use_template_db(tmp_path, monkeypatch, "timing_src.db")

        from core.analysis_history import save_analysis
        from db import store

        state = {
            "stock_code": "600000", "stock_info": {"name": "浦发银行"},
            "analysis_date": "2026-10-20",
            "timing_signal": {"regime": "range", "total_score": 37.9,
                              "source": "market_detector", "position_limit": 0.2},
            "trade_plan": {"direction": "观望"}, "final_decision": "观望。",
        }
        rid = save_analysis(state, source="ui")
        row = store.get_analysis(rid)
        assert row["market_score"] == 37.9
        assert row["market_score_source"] == "market_detector"

    def test_missing_source_stays_null_not_guessed(self, tmp_path, monkeypatch):
        """老 payload 没有 source ⇒ 存 NULL，不猜成 timing_engine（"不知道"≠"旧的"）。"""
        from tests.conftest import use_template_db
        use_template_db(tmp_path, monkeypatch, "timing_src2.db")

        from core.analysis_history import save_analysis
        from db import store

        rid = save_analysis({
            "stock_code": "600001", "analysis_date": "2026-10-20",
            "timing_signal": {"regime": "bear", "total_score": 16.9},
            "trade_plan": {}, "final_decision": ""}, source="ui")
        assert store.get_analysis(rid)["market_score_source"] is None
