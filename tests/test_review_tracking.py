"""预测跟踪闭环 + AI 持仓工具单元测试（临时库隔离，mock 行情/LLM）"""
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

import data.similar_days  # noqa: F401  (确保 sys.path)
from core.review_agent import _parse_verification_block
from db import store as db_store


def _days_ago(n: int) -> str:
    """相对今天的日期字符串（避免硬编码日期随时间过期）"""
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")


@pytest.fixture(autouse=True)
def _isolate_daily_picks(tmp_path, monkeypatch):
    """隔离 `output/daily_picks`（2026-10-29）。

    `today_decisions()` 的 `screens` 现在读当日推荐名单；不隔离就会读到真实的那份
    ⇒ 用例插的 600900 被真实名单顶掉，复盘正文里找不到它。
    """
    from core import decision_digest as _dd
    monkeypatch.setattr(_dd, "DAILY_PICKS_DIR",
                        str(tmp_path / "no_such_daily_picks"))


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    db_path = use_template_db(tmp_path, monkeypatch, "test_track.db")
    yield


class TestObservations:
    def test_insert_and_get_unverified(self, temp_store):
        yesterday = _days_ago(1)  # 固定基准，避免跨零点时两次调用落在不同日期
        n = db_store.insert_observations(yesterday, [
            {"content": "观察点1：守住3950", "kind": "观察点"},
            {"content": "风险：半导体流出持续", "kind": "风险"},
        ])
        assert n == 2
        assert db_store.insert_observations(yesterday, [
            {"content": "观察点1：守住3950", "kind": "观察点"}]) == 0  # 内容去重
        pending = db_store.get_unverified_observations()
        assert len(pending) == 2

    def test_today_excluded_from_pending(self, temp_store):
        """当日自己的观察点不进入待核对（避免自我核对）"""
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        db_store.insert_observations(today, [{"content": "今天的观察", "kind": "观察点"}])
        assert db_store.get_unverified_observations() == []

    def test_verify_and_stats(self, temp_store):
        today = _days_ago(0)  # 核对日（次日核对）
        db_store.insert_observations(_days_ago(1), [
            {"content": "观察A", "kind": "观察点"},
            {"content": "观察B", "kind": "观察点"},
            {"content": "观察C", "kind": "风险"},
        ])
        pending = db_store.get_unverified_observations()
        db_store.update_observation_result(pending[0]["id"], "达成", today)
        db_store.update_observation_result(pending[1]["id"], "未发生", today)
        db_store.update_observation_result(pending[2]["id"], "无法验证", today)
        stats = db_store.observation_stats()
        assert stats["verified"] == 3
        assert stats["达成"] == 1 and stats["未发生"] == 1
        assert stats["命中率"] == pytest.approx(1 / 3, abs=0.01)


class TestVerificationBlock:
    def test_parse_valid_block(self):
        md = "报告内容...\n```verification\n" + json.dumps({
            "verifications": [{"id": 1, "result": "达成", "依据": "守住了"}],
            "new_observations": [{"kind": "观察点", "content": "明日看量能"}],
        }, ensure_ascii=False) + "\n```"
        block = _parse_verification_block(md)
        assert block["verifications"][0]["id"] == 1
        assert block["new_observations"][0]["content"] == "明日看量能"

    def test_parse_missing_block(self):
        assert _parse_verification_block("无代码块的报告") is None

    def test_parse_invalid_json(self):
        assert _parse_verification_block("```verification\n{bad json}\n```") is None


class TestPortfolioTool:
    def test_portfolio_positions_with_data(self, monkeypatch):
        from core.agent_tools import TOOL_FUNCS
        pos = MagicMock(code="000001", name="平安银行", entry_date="2026-08-28",
                        entry_price=11.0, quantity=1000, stop_loss=None, take_profit=None)
        mock_pm = MagicMock()
        mock_pm.get_active_positions.return_value = [pos]
        mock_pm.load.return_value = MagicMock(total_value=11150.0, cash=98850.0)

        with patch("core.portfolio.get_portfolio_manager", return_value=mock_pm), \
             patch("data.realtime.get_realtime_quotes",
                   return_value={"000001": {"price": 11.65, "name": "平安银行"}}):
            result = TOOL_FUNCS["tool_portfolio_positions"]()

        assert result["positions"][0]["浮动盈亏"] == 650.0
        assert result["positions"][0]["盈亏%"] == pytest.approx(5.91, abs=0.05)
        assert result["总市值"] == 11150.0

    def test_portfolio_empty(self, monkeypatch):
        """空结果不能让模型以为"用户没有持仓"（2026-09-30 改）

        原文案是"当前无持仓。可在'持仓'页录入"。用户实测踩到的正是这句话：
        他问的是**模拟账户**，模型只调了这个工具、拿到"无持仓"就照说，
        而模拟账户里其实有 3 只。所以空结果必须①明说是"这本账未录入"
        ②把模型引向 tool_paper_account。
        """
        from core.agent_tools import TOOL_FUNCS
        mock_pm = MagicMock()
        # ⚠️ 2026-10-15：工具改为**先取代码（`active_codes()`，纯 DB 不取价）再批量取行情**
        #    （此前会逐只联网取价，取数受限时单只 ≈30s）⇒ 空持仓的判据从
        #    `get_active_positions()` 前移到 `active_codes()`。两个 mock 都要给，
        #    否则 MagicMock 会返回**真值**、分支走错，失败信息还只是 `KeyError: 'note'`。
        mock_pm.active_codes.return_value = []
        mock_pm.get_active_positions.return_value = []
        with patch("core.portfolio.get_portfolio_manager", return_value=mock_pm):
            result = TOOL_FUNCS["tool_portfolio_positions"]()
        assert result["positions"] == []
        assert "未录入" in result["note"]
        assert "不代表" in result["note"], "不得断言用户没有持仓"
        assert "tool_paper_account" in result["note"], "必须指向另一本账"


# ================================================================
# 决策三章（2026-10-10）：把"今天做了什么"写进复盘，且数字不经 LLM
# ================================================================

class TestDecisionSectionsRender:
    """`render_decision_sections` 是**程序填充**——数字必须来自 digest。"""

    def _digest(self):
        return {
            "today": {
                "date": "2026-10-10",
                "screens": [
                    {"code": "600900", "name": "长江电力", "score": 77.6,
                     "strategy": "主力资金", "degraded": False},
                    {"code": "000001", "name": "平安银行", "score": 70.1,
                     "strategy": "低估值成长", "degraded": False}],
                "analyses": [
                    {"code": "603127", "name": "贤丰控股", "decision_label": "看空",
                     "advice_status": "驳回", "duplicate": False},
                    {"code": "603127", "name": "贤丰控股", "decision_label": "观望",
                     "advice_status": "等待", "duplicate": True}],
                "trades": [],
                "positions": [{"code": "600211", "name": "西藏药业", "quantity": 5000}],
                "empty_reasons": [],
            },
            "quality": {
                "horizon": 5,
                "screen": {"status": "insufficient_sample", "n_backfilled": 0,
                           "n_pending": 95, "next_ready_date": "2026-10-16"},
                "analysis": {"status": "insufficient_sample", "n_backfilled": 0,
                             "next_ready_date": "2026-10-16", "by_label": []},
                "portfolio": {"status": "ok", "total_value": 1009347.02,
                              "total_pnl_pct": 0.93, "since": "2026-09-28",
                              "contributors": [{"code": "600211", "name": "西藏药业",
                                                "pnl": 1359.09, "pnl_pct": 0.74}],
                              "benchmark": {"pct": None, "start": None, "end": None}},
            },
            "actions": {
                "positions": [{"code": "600211", "name": "西藏药业", "stop": 35.71,
                               "target": 38.5, "to_stop_pct": -2.91,
                               "to_target_pct": 4.68, "triggered": None}],
                "watch": [{"code": "601398", "name": "工商银行", "advice_status": "等待",
                           "entry": None, "to_entry_pct": None,
                           "entry_conditions": ["收盘连续两日站稳 MA20"]}],
                "risks": [{"kind": "sector_outflow",
                           "text": "光学光电子 连续两日资金净流出（10-08 -42.74亿 → 10-09 -16.70亿）"}],
                "empty_reasons": [],
            },
        }

    def test_renders_all_three_sections(self):
        from core.review_agent import render_decision_sections
        md = render_decision_sections(self._digest())
        assert "## 今日决策清单" in md
        assert "## 决策质量评估" in md
        assert "## 明日待触发" in md

    def test_numbers_come_from_digest(self):
        from core.review_agent import render_decision_sections
        md = render_decision_sections(self._digest())
        assert "600900" in md and "77.6" in md          # 选股与分数
        assert "贤丰控股=看空（驳回）" in md
        assert "1 条为同日同票重复" in md                 # 重复标记可见
        assert "本期" not in md                          # 不写没有依据的措辞

    def test_insufficient_sample_writes_expected_date(self):
        """样本未到期必须写"预计 X 日"，而不是省略（"在等"≠"坏了"）"""
        from core.review_agent import render_decision_sections
        md = render_decision_sections(self._digest())
        assert "2026-10-16" in md
        assert "样本未到期" in md

    def test_benchmark_missing_is_stated(self):
        """指数取不到时明说原因，而不是留白或写 0"""
        from core.review_agent import render_decision_sections
        md = render_decision_sections(self._digest())
        assert "同期指数不可得" in md

    def test_trigger_distance_rendered(self):
        from core.review_agent import render_decision_sections
        md = render_decision_sections(self._digest())
        assert "距止损位 -2.91%" in md and "距目标位 +4.68%" in md

    def test_watch_shows_entry_conditions(self):
        from core.review_agent import render_decision_sections
        md = render_decision_sections(self._digest())
        assert "工商银行" in md and "站稳 MA20" in md

    def test_empty_day_is_explicit(self):
        from core.review_agent import render_decision_sections
        d = {"today": {"date": "2026-10-11", "screens": [], "analyses": [],
                       "trades": [], "positions": [],
                       "empty_reasons": ["今日无选股记录"]},
             "quality": {}, "actions": {}}
        md = render_decision_sections(d)
        assert "今日无选股记录" in md


class TestPromptRequiresVerifiableObservations:
    """观察点必须自带阈值与判定日——否则次日只能写"无法验证"（用户 2026-10-10 反馈）"""

    def test_prompt_asks_threshold_and_deadline(self):
        from core.review_agent import REVIEW_SYSTEM_PROMPT as P
        assert "阈值" in P and "判定日" in P and "判定口径" in P
        assert "basis" in P            # 核对要给出用到的今日数据

    def test_prompt_forbids_duplicating_decision_sections(self):
        from core.review_agent import REVIEW_SYSTEM_PROMPT as P
        assert "今日决策清单" in P and "不要重复输出这三章" in P


class TestGenerateReviewInjectsSections:
    """端到端（mock LLM）：三章必须真的出现在报告里，且插在 LLM 正文之前"""

    def _seed(self):
        today = _days_ago(0)
        db_store.insert_screen_picks([{
            "pick_date": today, "stock_code": "600900", "stock_name": "长江电力",
            "source": "screen", "strategy": "主力资金", "score": 77.6,
            "score_coverage": 0.9, "price_at_pick": 30.0, "rank_in_run": 1,
            "factors": {}, "conditions": {}}])
        db_store.save_market_snapshot(today, [
            {"stock_code": "600900", "stock_name": "长江电力", "price": 30.5}])

    def test_sections_present_and_before_llm_body(self, temp_store, monkeypatch, tmp_path):
        import core.review_agent as ra
        self._seed()
        monkeypatch.setattr(ra, "REPORT_DIR", str(tmp_path))
        fake = MagicMock()
        fake.chat.return_value = MagicMock(content="## 大盘概览\n测试正文。")
        with patch.object(ra, "_collect_market_data", return_value={}), \
             patch.object(ra, "DeepSeekClient", return_value=fake), \
             patch("data.fund_flow.ensure_snapshot_captured",
                   return_value={"skipped": True}):
            out = ra.generate_review(push=False)
        md = out["markdown"]
        assert "## 今日决策清单" in md and "## 明日待触发" in md
        assert "600900" in md
        # 程序三章在 LLM 正文之前
        assert md.index("## 今日决策清单") < md.index("## 大盘概览")

    def test_digest_failure_does_not_break_review(self, temp_store, monkeypatch, tmp_path):
        """digest 炸了只缺三章，复盘照常产出（红线 5）"""
        import core.review_agent as ra
        monkeypatch.setattr(ra, "REPORT_DIR", str(tmp_path))
        fake = MagicMock()
        fake.chat.return_value = MagicMock(content="## 大盘概览\n仅正文。")
        with patch.object(ra, "_collect_market_data", return_value={}), \
             patch.object(ra, "DeepSeekClient", return_value=fake), \
             patch("core.decision_digest.build_digest",
                   side_effect=RuntimeError("库炸了")), \
             patch("data.fund_flow.ensure_snapshot_captured",
                   return_value={"skipped": True}):
            out = ra.generate_review(push=False)
        assert "error" not in out
        assert "## 大盘概览" in out["markdown"]
        assert "## 今日决策清单" not in out["markdown"]


# ================================================================
# 百分比口径（2026-10-23 回归）
# ================================================================

class TestPctFormatting:
    """⭐ 事故：库里的 `ret_5d`/`avg_excess` 是**比值**（0.0176=1.76%），
    但渲染时用 `_fmt_pct` 直接加百分号 ⇒ 全部收益缩小 100 倍
    （「全体超额 +1.76%」印成「+0.02%」）。分档单调性方向仍对、量级全错。
    """

    def test_ratio_is_multiplied_by_100(self):
        import core.review_agent as ra
        assert ra._fmt_ret(0.0176) == "+1.76%"
        assert ra._fmt_ret(-0.0678) == "-6.78%"
        assert ra._fmt_ret(None) == "—"

    def test_points_keep_original_semantics(self):
        """`pnl_pct` / `benchmark['pct']` 已经是百分点 ⇒ 不能再乘 100。"""
        import core.review_agent as ra
        assert ra._fmt_pct(3.2) == "+3.20%"

    def test_rendered_section_uses_correct_scale(self, temp_store):
        """端到端：渲染出来的超额必须是 1.76% 量级，不是 0.02%。"""
        import core.review_agent as ra
        digest = {
            "quality": {"horizon": 5, "screen": {
                "status": "ok", "n_backfilled": 127, "n_pending": 0,
                "top5_excess": 0.0178, "all_excess": 0.0176,
                "by_score_band": [{"band": "90+", "n": 3,
                                   "avg_excess": 0.0397, "win_rate": 1.0}]},
                "analysis": {}, "portfolio": {}},
            "actions": {"as_of": "2026-10-16", "positions": []},
            "today": {}, "digest": {},
        }
        text = ra.render_decision_sections(digest)
        assert "+1.76%" in text, text
        assert "+0.02%" not in text, "仍在使用 _fmt_pct 渲染比值"


class TestIndexGrounding:
    """⭐ 事故：复盘数据块只给 3 个指数，模型自己编了沪深300/中证1000 的涨跌幅
    （10-23 实测：报告写 -3.2%/-1.6%，真实 +1.71%/+2.38%，符号全反）。
    修法是**让它有数可依**：取数扩到 5 个指数。
    """

    def test_collect_requests_five_indexes(self, monkeypatch):
        import data.realtime as rt
        calls = []

        def _fake(codes, *a, **k):
            calls.append(tuple(codes))
            return {c: {"name": c, "code": c, "price": 1.0, "change_pct": 0.0}
                    for c in codes}

        monkeypatch.setattr(rt, "get_realtime_quotes", _fake)
        import core.review_agent as ra
        data = ra._collect_market_data()
        # ⚠️ 只断言**第一次**调用（指数）；后续还有自选股/持仓等调用，不能取最后一次。
        idx_codes = calls[0]
        assert len(idx_codes) == 5, idx_codes
        assert "sh000300" in idx_codes and "sh000852" in idx_codes
        assert len(data["指数"]) == 5


class TestIndexGrounding:
    """⭐ 事故：复盘数据块只给 3 个指数，模型自己编了沪深300/中证1000 的涨跌幅
    （10-23 实测：报告写 -3.2%/-1.6%，真实 +1.71%/+2.38%，符号全反）。
    修法是**让它有数可依**：取数扩到 5 个指数。
    """

    def test_collect_requests_five_indexes(self, monkeypatch):
        import data.realtime as rt
        calls = []

        def _fake(codes, *a, **k):
            calls.append(tuple(codes))
            return {c: {"name": c, "code": c, "price": 1.0, "change_pct": 0.0}
                    for c in codes}

        monkeypatch.setattr(rt, "get_realtime_quotes", _fake)
        import core.review_agent as ra
        data = ra._collect_market_data()
        # ⚠️ 只断言**第一次**调用（指数）；后续还有自选股/持仓等调用，不能取最后一次。
        idx_codes = calls[0]
        assert len(idx_codes) == 5, idx_codes
        assert "sh000300" in idx_codes and "sh000852" in idx_codes
        assert len(data["指数"]) == 5


class TestPositionLimitUnits:
    """⭐ 事故：`market_signal.position_limit` 是**比值**（0.20），
    但数据块键名写着"仓位上限%"，直接把比值塞进去 ⇒ 模型输出"仓位上限0.2%"，
    读起来像"几乎禁止开仓"，真实值是 20%。同 `_fmt_pct` 属同一类换算错误。
    """

    def test_position_limit_is_converted_to_percent(self, monkeypatch):
        import core.review_agent as ra
        from core.market_detector import MarketSignal

        class _Sig:
            market_state = "偏弱震荡"
            total_score = 34.7
            position_limit = 0.20
            scores = {}

        monkeypatch.setattr("core.market_detector.MarketDetector.detect",
                            lambda self: _Sig())
        data = ra._collect_market_data()
        blk = data["市场检测14指标"]
        assert blk["仓位上限%"] == 20.0, blk["仓位上限%"]
