"""AI 对话 Agent 单元测试（mock LLM，不依赖网络）"""
import json
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

import core.chat_agent as ca
from core.chat_agent import ChatAgent, _brief_summary
from core.agent_tools import execute_tool, TOOL_FUNCS


def make_tool_call(tc_id, name, arguments):
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


@pytest.fixture
def agent():
    with patch.object(ca, "DeepSeekClient") as mock_client_cls:
        agent = ChatAgent()
        agent.client = mock_client_cls.return_value
        yield agent


def collect(agent, msg, history=None):
    events = []
    for ev in agent.reply(msg, history or []):
        events.append((ev[0],) + tuple(ev[1:]))
    return events


class TestAgentLoop:
    def test_direct_answer_no_tools(self, agent):
        agent.client.chat.return_value = FakeMessage(content="你好，我是A股助手。")
        events = collect(agent, "你是谁")
        assert events[-1][0] == "answer"
        assert "A股助手" in events[-1][1]

    def test_single_tool_round(self, agent):
        """首轮返回工具调用 → 执行 → 二轮返回最终回答"""
        first = FakeMessage(tool_calls=[make_tool_call("c1", "tool_stock_profile",
                                                       json.dumps({"symbol": "600519"}))])
        second = FakeMessage(content="茅台现价1297元。*仅供参考*")
        agent.client.chat.side_effect = [first, second]

        executed = {}
        with patch("core.chat_agent.execute_tool",
                   side_effect=lambda n, a: executed.update(parsed=json.loads(a)) or {"name": "贵州茅台", "price": 1297.4}):
            events = collect(agent, "茅台多少钱")

        assert executed["parsed"] == {"symbol": "600519"}
        kinds = [e[0] for e in events]
        assert kinds == ["tool_start", "tool_end", "answer"]
        assert events[1][2] != "失败: "  # 摘要非失败
        assert "1297" in events[-1][1]

    def test_tool_result_appended_as_role_tool(self, agent):
        """工具结果必须以 role=tool 消息回填（含截断）"""
        first = FakeMessage(tool_calls=[make_tool_call("c1", "tool_market_signal", "{}")])
        second = FakeMessage(content="市场偏强。")
        agent.client.chat.side_effect = [first, second]
        big_result = {"市场状态": "bull", "payload": "x" * 5000}

        captured = {}
        def fake_chat(messages, tools=None, **kw):
            captured.setdefault("messages", []).append(list(messages))
            return (first, second)[len(captured["messages"]) - 1]

        with patch.object(agent.client, "chat", side_effect=fake_chat), \
             patch("core.chat_agent.execute_tool", return_value=big_result):
            events = collect(agent, "大盘怎么样")

        tool_msgs = [m for m in captured["messages"][1] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "c1"
        assert len(tool_msgs[0]["content"]) <= 1600  # 已截断

    def test_max_rounds_force_summary(self, agent):
        """连续 MAX_ROUNDS 轮工具调用后强制汇总"""
        rounds = []
        for i in range(ca.MAX_ROUNDS):
            rounds.append(FakeMessage(tool_calls=[make_tool_call(f"c{i}", "tool_realtime_quotes", "{}")]))
        rounds.append(FakeMessage(content="汇总回答。"))
        agent.client.chat.side_effect = rounds

        with patch("core.chat_agent.execute_tool", return_value={"quotes": []}):
            events = collect(agent, "行情")

        assert events[-1][0] == "answer"
        assert "汇总回答" in events[-1][1]
        assert agent.client.chat.call_count == ca.MAX_ROUNDS + 1

    def test_llm_exception_yields_error_event(self, agent):
        agent.client.chat.side_effect = RuntimeError("网络炸了")
        events = collect(agent, "你好")
        assert events[-1][0] == "error"
        assert "网络炸了" in events[-1][1]

    def test_history_trimmed(self, agent):
        """历史超过 MAX_HISTORY 时截断，system 始终在首位"""
        agent.client.chat.return_value = FakeMessage(content="ok")
        history = [{"role": "user", "content": f"旧消息{i}"} for i in range(30)]
        captured = {}
        with patch.object(agent.client, "chat",
                          side_effect=lambda messages, **kw: captured.update(messages=list(messages)) or FakeMessage(content="ok")):
            collect(agent, "新问题", history)
        messages = captured["messages"]
        assert messages[0]["role"] == "system"
        assert len(messages) == ca.MAX_HISTORY + 2  # system + 截断历史 + 本轮
        assert messages[-1]["content"] == "新问题"


class TestToolDispatch:
    def test_unknown_tool(self):
        result = execute_tool("tool_not_exist", "{}")
        assert "error" in result

    def test_bad_arguments(self):
        result = execute_tool("tool_stock_profile", "{invalid json")
        assert "error" in result

    def test_tool_exception_wrapped(self):
        with patch.dict(TOOL_FUNCS, {"tool_x": lambda **kw: 1 / 0}):
            result = execute_tool("tool_x", "{}")
        assert "error" in result
        assert "ZeroDivisionError" in result["error"]


class TestBriefSummary:
    def test_various_results(self):
        assert "失败" in _brief_summary({"error": "超时"})
        assert "3 条行情" in _brief_summary({"quotes": [1, 2, 3]})
        assert "选出 5 只" in _brief_summary({"候选数": 5})
        assert _brief_summary({}) == "完成"


# ===== Phase3: 图表事件与人格化 =====

class TestChartEvents:
    def test_technicals_yields_kline_chart(self, agent):
        """技术面工具应触发 kline 图表事件"""
        first = FakeMessage(tool_calls=[make_tool_call("c1", "tool_stock_technicals",
                                                       json.dumps({"symbol": "600519"}))])
        second = FakeMessage(content="多头排列。")
        agent.client.chat.side_effect = [first, second]
        tech_result = {"代码": "600519", "现价": 100.0, "均线形态": "多头排列",
                       "kline_sample": {"dates": ["d1", "d2"], "closes": [99.0, 100.0]}}
        with patch("core.chat_agent.execute_tool", return_value=tech_result):
            events = collect(agent, "茅台技术面")
        kinds = [e[0] for e in events]
        assert kinds == ["tool_start", "tool_end", "chart", "answer"]
        chart = events[2][1]
        assert chart["kind"] == "kline"
        assert chart["closes"] == [99.0, 100.0]

    def test_fund_flow_yields_fund5_chart(self, agent):
        first = FakeMessage(tool_calls=[make_tool_call("c1", "tool_fund_flow",
                                                       json.dumps({"symbol": "600519"}))])
        second = FakeMessage(content="资金面良好。")
        agent.client.chat.side_effect = [first, second]
        fund_result = {"五维评分": "17/25", "维度明细": {"方向": {"score": 6, "max": 6}}}
        with patch("core.chat_agent.execute_tool", return_value=fund_result):
            events = collect(agent, "茅台资金面")
        chart = [e for e in events if e[0] == "chart"][0][1]
        assert chart["kind"] == "fund5"
        assert chart["score"] == "17/25"

    def test_no_chart_for_quotes(self, agent):
        agent.client.chat.return_value = FakeMessage(content="好的。")
        with patch("core.chat_agent.execute_tool", return_value={"quotes": []}):
            # 无工具轮次，直接回答 → 无图表
            events = collect(agent, "你好")
        assert all(e[0] != "chart" for e in events)


class TestPersona:
    def test_persona_appended_to_system(self, agent):
        """选择人格化视角时，系统提示词应包含对应补充要求"""
        captured = {}
        with patch.object(ca, "DeepSeekClient"), \
             patch.object(agent.client, "chat",
                          side_effect=lambda messages, **kw: captured.update(m=messages) or FakeMessage(content="ok")):
            agent2 = ChatAgent(persona="价值投资视角")
            agent2.client = agent.client
            list(agent2.reply("你好"))
        assert "价值投资者" in captured["m"][0]["content"]

    def test_default_persona_no_extra(self, agent):
        captured = {}
        with patch.object(agent.client, "chat",
                          side_effect=lambda messages, **kw: captured.update(m=messages) or FakeMessage(content="ok")):
            list(agent.reply("你好"))
        assert "补充要求" not in captured["m"][0]["content"]


# ===== 复盘 Agent 数据收集 =====

@pytest.fixture
def offline_market_data():
    """把 `_collect_market_data` 涉及的**所有**外部数据源都打掉。

    回归背景：原先只 patch 了 realtime / fund_flow 三处，剩下的市场检测、市场观点、
    相似日、新闻仍会真的联网——两个用例各跑 80 秒（占全量测试 2.5 分钟），
    而且网络抖动时会变成随机失败。"单元测试不联网"必须是完整的，不能只补几处。
    """
    with patch("data.realtime.get_index_quotes", side_effect=Exception("net down")), \
         patch("data.realtime.get_realtime_quotes", side_effect=Exception("net down")), \
         patch("data.fund_flow.get_industry_flow_snapshot", return_value=None), \
         patch("data.fund_flow.get_stored_industry_flow", return_value=(None, None)), \
         patch("data.news_feed.get_latest_news", return_value=[]), \
         patch("data.market_opinions.get_market_opinions",
               return_value={"error": "offline"}), \
         patch("data.similar_days.find_similar_days",
               return_value={"error": "offline"}), \
         patch("core.market_detector.MarketDetector") as detector:
        detector.return_value.detect.side_effect = Exception("net down")
        yield


class TestReviewAgent:
    def test_collect_market_data_tolerant(self, offline_market_data):
        """数据收集各项失败应独立容忍，不抛异常"""
        from core.review_agent import _collect_market_data
        data = _collect_market_data()
        assert isinstance(data, dict)
        assert "指数" not in data and "自选股" not in data

    def test_collect_reads_latest_pipeline(self, tmp_path, offline_market_data):
        import os
        from core import review_agent as ra
        latest_dir = tmp_path / "output" / "daily_picks"
        latest_dir.mkdir(parents=True)
        # ⚠️ F2（2026-10-18 审查）：mode 必须 ∈ (rule, llm) —— 盘前选股
        #    （mode="screen"）也会把 date 刷成今天，不能当"今日盘后流水线"。
        (latest_dir / "latest.json").write_text(json.dumps({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "mode": "rule", "picks": [{"股票代码": "600519", "股票名称": "贵州茅台", "score": 80}],
            "backtest": {"total_return": 1.5},
        }, ensure_ascii=False), encoding="utf-8")
        with patch.object(ra, "_PROJECT_ROOT", str(tmp_path)):
            data = ra._collect_market_data()
        assert data["流水线"]["候选"][0]["名称"] == "贵州茅台"
        assert data["流水线"]["模式"] == "rule"

    def test_collect_ignores_morning_screen_as_pipeline(self, tmp_path, offline_market_data):
        """⭐ F2：盘前选股（mode=screen）不许冒充"今日盘后流水线"——必须明写实情。"""
        import os
        from core import review_agent as ra
        latest_dir = tmp_path / "output" / "daily_picks"
        latest_dir.mkdir(parents=True)
        (latest_dir / "latest.json").write_text(json.dumps({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "mode": "screen", "picks": [{"股票代码": "600519"}],
        }, ensure_ascii=False), encoding="utf-8")
        with patch.object(ra, "_PROJECT_ROOT", str(tmp_path)):
            data = ra._collect_market_data()
        assert isinstance(data["流水线"], str)
        assert "screen" in data["流水线"] and "盘后全流程" in data["流水线"], data["流水线"]


# ================================================================
# 时间基准与"数据日期 ≠ 查询时间"（2026-09-25 实测发现）
# ================================================================

class TestTimeContext:
    """系统提示必须注入"今天几号/是否交易日/上一交易日"。

    修复前实测（2026-09-25 休市）：问"兴业银行当前是否值得投资"，
    模型答"**当前的实时行情**：17.88 元"——17.88 是 09-24 收盘价，
    而模型既不知道今天休市，也无从判断这价格属于哪天。
    """

    def test_injects_today_weekday_and_trading_day(self):
        ctx = ca.time_context()
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in ctx
        assert "时间基准" in ctx
        assert ("交易日" in ctx)

    def test_calendar_failure_does_not_break_prompt(self):
        with patch("data.trading_calendar.is_trading_day",
                   side_effect=RuntimeError("cal down")):
            ctx = ca.time_context()
        assert "时间基准" in ctx and datetime.now().strftime("%Y-%m-%d") in ctx

    def test_system_message_carries_time_context(self, agent):
        """每轮都要带上时间基准（长会话跨天后不能沿用旧日期）"""
        agent.client.chat.return_value = FakeMessage(content="ok")
        events = collect(agent, "现在几点")
        assert events[-1][0] == "answer"
        sent = agent.client.chat.call_args_list[0][0][0]
        system = [m for m in sent if m["role"] == "system"][0]["content"]
        assert datetime.now().strftime("%Y-%m-%d") in system

    def test_prompt_demands_separating_query_time_from_data_date(self):
        assert "数据日期" in ca.SYSTEM_PROMPT
        assert "as_of" in ca.SYSTEM_PROMPT, "必须点明 as_of 是查询时间而非数据时间"


class TestToolDataDates:
    """工具输出必须带**数据自身**的日期，而不是查询时间"""

    def test_realtime_quotes_reports_data_date_and_honest_status(self):
        import core.agent_tools as at
        row = {"code": "601166", "name": "兴业银行", "price": 17.88, "change_pct": 0.06,
               "change": 0.01, "turnover_rate": 0.28, "amount_wan": 106800.0,
               "quote_time": "16:14:56", "quote_date": "2026-09-24",
               "quote_datetime": "2026-09-24 16:14:56",
               "stale": False, "age_seconds": 0.0, "data_lagging": False}
        with patch("data.realtime.get_realtime_quotes", return_value={"601166": row}), \
             patch.object(at, "_market_clock",
                          return_value={"今天": "2026-09-25", "是否交易日": False,
                                        "最近交易日": "2026-09-24"}):
            r = execute_tool("tool_realtime_quotes", {"codes": ["601166"]})
        q = r["quotes"][0]
        assert q["数据日期"] == "2026-09-24"
        assert q["行情时间"] == "2026-09-24 16:14:56", "日期不能丢"
        assert "实时" not in q["数据状态"].replace("非当日实时", ""), \
            "非当日数据不得标成'实时'"
        assert "2026-09-24" in q["数据状态"]
        assert r["今天是否交易日"] is False

    def test_realtime_quotes_flags_lagging_data(self):
        row = {"code": "601166", "name": "兴业银行", "price": 17.88, "change_pct": 0.0,
               "change": 0.0, "turnover_rate": None, "amount_wan": None,
               "quote_time": "15:00:00", "quote_date": "2026-09-23",
               "quote_datetime": "2026-09-23 15:00:00",
               "stale": False, "age_seconds": 0.0, "data_lagging": True,
               "staleness_reason": "行情日期 2026-09-23 早于应有的 2026-09-24"}
        with patch("data.realtime.get_realtime_quotes", return_value={"601166": row}), \
             patch("core.agent_tools._market_clock",
                   return_value={"今天": "2026-09-25", "是否交易日": False,
                                 "最近交易日": "2026-09-24"}):
            r = execute_tool("tool_realtime_quotes", {"codes": ["601166"]})
        assert "滞后" in r["quotes"][0]["数据状态"]
        assert "警告" in r

    def test_technicals_dates_from_datetimeindex(self):
        """`get_stock_data` 返回 DatetimeIndex 而非 `Date` 列。

        修复前只在 df.columns 里找日期 → 找不到就退化成 ["0".."59"]，
        图表横轴变成数字、模型也看不到数据截止日。
        """
        import pandas as pd
        idx = pd.date_range("2026-01-05", periods=30, freq="B")
        df = pd.DataFrame({"Open": 1.0, "Close": [float(i) for i in range(30)],
                           "High": 1.0, "Low": 1.0, "Volume": 1.0}, index=idx)
        import core.agent_tools as at
        dates = at._kline_dates(df)
        assert dates[0] == "2026-01-05"
        assert dates[-1] == idx[-1].strftime("%Y-%m-%d")

    def test_technicals_returns_data_through_and_real_dates(self):
        import pandas as pd
        idx = pd.date_range("2026-01-05", periods=30, freq="B")
        df = pd.DataFrame({"Close": [10.0 + i for i in range(30)]}, index=idx)
        with patch("data.stock_data.StockDataFetcher.get_stock_data", return_value=df):
            r = execute_tool("tool_stock_technicals", {"symbol": "601166"})
        assert r["数据截止"] == idx[-1].strftime("%Y-%m-%d")
        assert r["kline_sample"]["dates"][-1] == r["数据截止"], "图表末点要与数据截止一致"

    def test_profile_fills_valuation_from_tencent_when_eastmoney_down(self):
        """东财被墙时用腾讯实时行情补市值/PE/PB，并标注来源与数据日期"""
        quote = {"601166": {"code": "601166", "name": "兴业银行", "price": 17.88,
                            "change_pct": 0.06, "pe": 5.01, "pb": 0.45,
                            "total_mcap_yi": 3783.92, "quote_date": "2026-09-24"}}
        with patch("data.stock_data.StockDataFetcher.get_stock_info",
                   return_value={"symbol": "601166", "name": "兴业银行",
                                 "current_price": 17.88, "pe_ratio": None,
                                 "pb_ratio": None, "market_cap": "N/A"}), \
             patch("data.realtime.get_realtime_quotes", return_value=quote):
            r = execute_tool("tool_stock_profile", {"symbol": "601166"})
        assert r["market_cap"] == "3783.92亿"
        assert r["pe_ratio"] == 5.01 and r["pb_ratio"] == 0.45
        assert "缺失字段" not in r
        assert r["估值数据日期"] == "2026-09-24", "兜底数据也要给数据日期"

    def test_profile_lists_missing_fields_when_all_sources_fail(self):
        with patch("data.stock_data.StockDataFetcher.get_stock_info",
                   return_value={"symbol": "601166", "name": "兴业银行",
                                 "current_price": 17.88, "pe_ratio": 5.01,
                                 "pb_ratio": 0.45, "market_cap": "N/A"}), \
             patch("data.realtime.get_realtime_quotes", return_value={}):
            r = execute_tool("tool_stock_profile", {"symbol": "601166"})
        assert r["缺失字段"] == ["market_cap"]


class TestMarketSignalTool:
    """`tool_market_signal` 曾因字段名写错而**从未成功过**。

    旧实现读 `signal.dimension_scores`，而 `MarketSignal` 的真实字段是 `scores`
    → AttributeError 被 @_tool 兜住 → 每次都返回 {"error": ...}。
    实测后果：模型拿不到大盘环境，却在回答里写"当前市场环境**可能**偏震荡"（凭空推测）。
    """

    def _signal(self, **over):
        from core.market_detector import MarketSignal
        scores = {
            "hs300_ma20_deviation": {"score": 5, "max_score": 10, "weight": 0.10,
                                     "weighted_score": 5.0, "detail": "偏离-1%"},
            "sh_macd": {"score": 5, "max_score": 10, "weight": 0.10,
                        "weighted_score": 5.0, "detail": "金叉"},
            "cyb_relative_strength": {"score": 0, "max_score": 10, "weight": 0.10,
                                      "weighted_score": 0.0, "detail": "弱于大盘"},
            "advance_decline_ratio": {"score": 4, "max_score": 8, "weight": 0.08,
                                      "weighted_score": 4.0, "detail": "0.8",
                                      "unavailable": True},
        }
        scores.update(over.pop("scores", {}))
        return MarketSignal(total_score=43.0, market_state="中性震荡",
                            signal="观望/轻仓", position_limit=0.4,
                            strategy_bias="防御型", operation_advice="降低仓位",
                            scores=scores, indicators={}, risk_warnings=["cyb: 弱于大盘"],
                            timestamp="2026-09-25 12:40:00")

    def test_returns_real_fields_not_error(self):
        with patch("core.market_detector.MarketDetector.detect",
                   return_value=self._signal()):
            r = execute_tool("tool_market_signal", {})
        assert "error" not in r, r
        assert r["市场状态"] == "中性震荡"
        assert r["总分"] == 43.0
        assert r["仓位上限"] == "40%"
        assert r["维度得分"]["趋势"] == "10.0/30分"

    def test_lists_missing_indicators(self):
        """缺失指标必须明说（它决定结论是否保守）"""
        with patch("core.market_detector.MarketDetector.detect",
                   return_value=self._signal()):
            r = execute_tool("tool_market_signal", {})
        assert "涨跌家数比" in r["数据缺失指标"]
        assert "偏保守" in r["说明"]


# ================================================================
# 财务工具（tool_financials）——2026-09-25 新增
# ================================================================

import core.agent_tools as _at


def _fin_payload(*, bank: bool) -> dict:
    """构造 8 期财务三表（含去年同报告期，用于同比）"""
    def periods(vals):
        # (报告期, 值)
        return vals
    if bank:
        bs = [
            {"报告日": "20260630", "资产总计": 1.1460940e13, "负债合计": 1.0500117e13,
             "归属于母公司股东的权益": 9.4879e11, "发放贷款及垫款": 6.164553e12,
             "减:贷款损失准备": 1.49585e11},
            {"报告日": "20250630", "资产总计": 1.10e13, "负债合计": 1.01e13,
             "归属于母公司股东的权益": 9.0e11, "发放贷款及垫款": 6.0e12,
             "减:贷款损失准备": 1.4e11},
            {"报告日": "20250331", "资产总计": 1.09e13, "负债合计": 1.00e13,
             "归属于母公司股东的权益": 8.9e11},
        ]
    else:
        bs = [
            {"报告日": "20260630", "资产总计": 3.09051e11, "负债合计": 4.6954e10,
             "归属于母公司股东的权益": 2.51254e11, "货币资金": 5.3519e10, "存货": 5.0e10,
             "发放贷款及垫款": 4.16e8},
            {"报告日": "20250630", "资产总计": 2.9e11, "负债合计": 4.4e10,
             "归属于母公司股东的权益": 2.3e11, "货币资金": 5.0e10, "存货": 4.6e10},
        ]
    inc = [
        {"报告日": "20260630", "营业收入": 1.10177e11, "净利润": 4.1315e10,
         "归属于母公司的净利润": 4.1131e10, "营业利润": 4.8315e10,
         **({"净利息收入": 7.2804e10} if bank else {})},
        {"报告日": "20250630", "营业收入": 1.1045e11, "净利润": 4.3360e10,
         "归属于母公司的净利润": 4.3140e10, "营业利润": 5.0e10,
         **({"净利息收入": 7.31e10} if bank else {})},
        {"报告日": "20250331", "营业收入": 5.5e10, "净利润": 2.0e10},
    ]
    cf = [
        {"报告日": "20260630", "经营活动产生的现金流量净额": -7.309e9},
        {"报告日": "20250630", "经营活动产生的现金流量净额": 3.0e9},
    ]
    return {"balance_sheet": bs, "income_statement": inc, "cash_flow": cf,
            "financial_ratios": {}}


class TestFinanceHelpers:
    def test_num_parsing(self):
        import math
        n = _at._num
        assert n(None) is None
        assert n(float("nan")) is None, "DataFrame 空值是 NaN，不是 None"
        assert n("--") is None and n("N/A") is None and n("") is None
        assert n(1234.5) == 1234.5
        assert n("1,234.5") == 1234.5
        assert n("1.5亿") == 1.5e8
        assert n("2万") == 2e4
        assert n("abc") is None

    def test_period_key_delegates_to_stock_data(self):
        """报告期归一必须复用 data.stock_data 的实现，不许再写第二套口径"""
        from data.stock_data import _period_key
        for v in (20260630, "20260630", "2026-06-30", "1998"):
            assert _at._fin_period_key(v) == _period_key(v)

    def test_series_skips_mostly_empty_candidate(self):
        """「资产」这类总额科目在部分期次会断档，应改用它之外的候选

        实测 601166：报表里「资产」在最新期是 NaN，但「资产总计」有值
        （114609.40 亿）。若采用断档候选，最新一期就会是空白。
        """
        rows = [("20260630", {"资产": float("nan"), "资产总计": 1.1460940e13}),
                ("20260331", {"资产": 1.0e13, "资产总计": 1.1282907e13}),
                ("20251231", {"资产总计": 1.1094256e13})]
        col, vals = _at._fin_series(rows, ("资产", "资产总计"), 3)
        assert col == "资产总计", "应跳过断档的「资产」改用「资产总计」"
        assert vals == [114609.40, 112829.07, 110942.56]      # 已是亿元

    def test_series_prefers_candidate_with_latest_period(self):
        """两个候选都"够用"时，优先取**最新报告期有值**的那个"""
        rows = [("20260630", {"A": float("nan"), "B": 2.0e11}),
                ("20260331", {"A": 1.0e11, "B": 1.0e11})]
        col, vals = _at._fin_series(rows, ("A", "B"), 2)
        assert col == "B"
        assert vals == [2000.0, 1000.0]


class TestFinancialsTool:
    def test_registered(self):
        from core.agent_tools import TOOL_FUNCS, TOOL_SCHEMAS
        assert "tool_financials" in TOOL_FUNCS
        assert any(s["function"]["name"] == "tool_financials" for s in TOOL_SCHEMAS)

    def test_bank_subjects_and_disclosure(self):
        with patch("data.stock_data.StockDataFetcher.get_financial_data",
                   return_value=_fin_payload(bank=True)):
            r = execute_tool("tool_financials", {"symbol": "601166", "periods": 2})
        assert r["数据截止"] == "2026-06-30"
        assert r["报告期"] == ["2026-06-30", "2025-06-30"]
        assert list(r["资产负债表(亿元)"] )[:3] == ["资产总计", "负债合计", "归母权益"]
        assert "贷款损失准备" in r["资产负债表(亿元)"], "银行的贷款科目必须取到"
        assert "货币资金" not in r["资产负债表(亿元)"], "银行没有货币资金科目"
        # 三表里根本没有的银行专项指标必须显式声明
        assert "不良贷款率" in r["三表外的关键指标"]
        assert "不得据此推测" in r["说明"]
        # 口径提示：防止把银行的高杠杆（存款为负债）读成"财务风险高"
        assert "正常业务模式" in r["银行口径提示"]
        assert "不可与工商企业横向比较" in r["银行口径提示"]

    def test_non_bank_not_mislabeled(self):
        """回归：600519 也有小额'发放贷款及垫款'（财务公司），不得据此判成银行"""
        with patch("data.stock_data.StockDataFetcher.get_financial_data",
                   return_value=_fin_payload(bank=False)):
            r = execute_tool("tool_financials", {"symbol": "600519"})
        assert "三表外的关键指标" not in r
        assert "银行口径提示" not in r, "工商股不该出现银行口径提示"
        assert "货币资金" in r["资产负债表(亿元)"]
        assert "存货" in r["资产负债表(亿元)"]

    def test_yoy_uses_same_report_period(self):
        """同比必须与去年**同报告期**比（财报是累计数，跨期不可比）"""
        with patch("data.stock_data.StockDataFetcher.get_financial_data",
                   return_value=_fin_payload(bank=True)):
            r = execute_tool("tool_financials", {"symbol": "601166", "periods": 2})
        # 1101.77 / 1104.5 - 1 ≈ -0.25%
        assert r["同比%"]["营业收入"] == pytest.approx(-0.25, abs=0.02)
        assert r["同比%"]["净利润"] == pytest.approx(-4.72, abs=0.05)

    def test_computed_ratios(self):
        with patch("data.stock_data.StockDataFetcher.get_financial_data",
                   return_value=_fin_payload(bank=True)):
            r = execute_tool("tool_financials", {"symbol": "601166"})
        rt_ = r["自算比率"]
        assert rt_["资产负债率%"] == pytest.approx(91.62, abs=0.05)
        assert rt_["净利率%"] == pytest.approx(37.5, abs=0.2)
        assert rt_["ROE%(报告期累计,未年化)"] == pytest.approx(4.35, abs=0.1)
        assert rt_["贷款损失准备/贷款%"] == pytest.approx(2.43, abs=0.05)

    def test_missing_core_subject_reported(self):
        payload = _fin_payload(bank=False)
        for r_ in payload["income_statement"]:
            r_.pop("营业收入", None)      # 营业总收入也没有 → 核心科目缺失
        with patch("data.stock_data.StockDataFetcher.get_financial_data",
                   return_value=payload):
            r = execute_tool("tool_financials", {"symbol": "600519"})
        assert "营业收入" in r["未取到核心科目"]

    def test_balance_sheet_none_does_not_break_tool(self):
        """回归：取不到某张表时 `_get_chinese_financial_data` 存的是 **None**（键仍在）。

        `dict.get(key, default)` 的默认值只在**键不存在**时生效 —— 写成
        `fin.get("balance_sheet", [{}])[0]` 会 `None[0]` 抛 TypeError，整个工具变 error。
        """
        payload = _fin_payload(bank=True)
        payload["balance_sheet"] = None
        with patch("data.stock_data.StockDataFetcher.get_financial_data",
                   return_value=payload):
            r = execute_tool("tool_financials", {"symbol": "601166"})
        assert "error" not in r, r
        assert r["代码"] == "601166"
        assert "资产负债表(亿元)" not in r, "无资产负债表时不该凭空造出这一段"
        assert "利润表(亿元)" in r

    def test_ths_fallback_source_is_labeled(self):
        """兜底源（同花顺主要指标，非完整报表）必须原样透出，不能冒充新浪三表"""
        payload = _fin_payload(bank=False)
        for r_ in payload["income_statement"]:
            r_["_source"] = "ths主要指标(非完整报表)"
        with patch("data.stock_data.StockDataFetcher.get_financial_data",
                   return_value=payload):
            r = execute_tool("tool_financials", {"symbol": "600519"})
        assert "ths" in r["数据来源"]

    def test_equity_candidates_shared_between_display_and_roe(self):
        """资产表展示段与 ROE 计算必须共用同一份权益候选表

        原先 ROE 那处写成 `_FIN_FIELDS["资产负债表"][2][1]`（靠下标取值），
        重排候选表就会静默用错科目。
        """
        assert _at._FIN_FIELDS["资产负债表"][2] == ("归母权益", _at._FIN_EQUITY_CANDIDATES)

    def test_bad_symbol(self):
        assert "error" in execute_tool("tool_financials", {"symbol": "abc"})
        assert "error" in execute_tool("tool_financials", {"symbol": "60051"})

    def test_no_data_returns_error(self):
        with patch("data.stock_data.StockDataFetcher.get_financial_data",
                   return_value={"balance_sheet": None, "income_statement": None,
                                 "cash_flow": None, "financial_ratios": {}}):
            r = execute_tool("tool_financials", {"symbol": "601166"})
        assert "error" in r

    def test_output_is_field_ordered_by_importance(self):
        """工具结果超长会从尾部截断，所以关键字段必须排在前"""
        with patch("data.stock_data.StockDataFetcher.get_financial_data",
                   return_value=_fin_payload(bank=True)):
            r = execute_tool("tool_financials", {"symbol": "601166"})
        keys = list(r.keys())
        assert keys.index("数据截止") < keys.index("利润表(亿元)")
        assert keys.index("利润表(亿元)") < keys.index("现金流量表(亿元)")
        assert keys.index("自算比率") < keys.index("现金流量表(亿元)")

    def test_output_fits_tool_result_budget(self):
        """输出必须能塞进 agent 的工具结果上限，否则模型只看到半截"""
        with patch("data.stock_data.StockDataFetcher.get_financial_data",
                   return_value=_fin_payload(bank=True)):
            r = execute_tool("tool_financials", {"symbol": "601166", "periods": 4})
        assert len(json.dumps(r, ensure_ascii=False, default=str)) < ca.TOOL_RESULT_MAX_CHARS


# ================================================================
# 注入上下文（context_blocks）：深度分析页「就这份报告提问」用
# ================================================================

class TestInjectedContext:
    def _agent(self, blocks):
        with patch.object(ca, "DeepSeekClient") as mock_client_cls:
            a = ChatAgent(context_blocks=blocks)
            a.client = mock_client_cls.return_value
        a.client.chat.return_value = FakeMessage(content="答")
        return a

    @staticmethod
    def _system(a):
        return a.client.chat.call_args[0][0][0]["content"]

    def test_block_reaches_system_message(self):
        a = self._agent(["CTX-SENTINEL 用户正在看的报告"])
        collect(a, "为什么舆情乐观却看空")
        system = self._system(a)
        assert "CTX-SENTINEL" in system

    def test_block_comes_after_role_prompt(self):
        """注入块要在角色提示之后 —— 先给通用规则，再给"本会话聚焦的那份报告\""""
        a = self._agent(["CTX-SENTINEL"])
        collect(a, "问题")
        system = self._system(a)
        assert system.index("你是 bigAagents") < system.index("CTX-SENTINEL")

    def test_blank_blocks_are_dropped(self):
        """空块不能污染提示（否则每轮都多一段空白）"""
        a = self._agent(["   ", "", None, "CTX-SENTINEL"])
        assert a.context_blocks == ["CTX-SENTINEL"]

    def test_no_blocks_keeps_message_shape(self):
        """不注入时消息结构必须与以前完全一致（回归保护）"""
        a = self._agent(None)
        collect(a, "你好")
        messages = a.client.chat.call_args[0][0]
        assert [m["role"] for m in messages] == ["system", "user"]

    def test_history_still_sent_after_system(self):
        a = self._agent(["CTX"])
        collect(a, "追问", [{"role": "user", "content": "上一个问题"},
                           {"role": "assistant", "content": "上一个回答"}])
        messages = a.client.chat.call_args[0][0]
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]


class TestSplitFund5Dims:
    """⚠️ 不可评的维度不能画成 0 分柱。

    2026-10-10 用户报"最近一次 ai 助手提问资金面五维评分为空"，追下去是两条：
    ① 评分口径把"取不到的结构维度"当 0 分计入 25 总分；
    ② 页面把不可评维度画成 0 分柱 → 图上空柱，看不出是"资金差"还是"没数据"。
    """

    def test_unavailable_dim_separated(self):
        from core.chat_agent import split_fund5_dims
        r = split_fund5_dims({
            "方向": {"score": 6, "max": 6, "available": True, "desc": "3/3窗口为正"},
            "结构": {"score": None, "max": 5, "available": False,
                     "desc": "同花顺快照源无单型拆分 —— 不可评，不计入分母"},
            "持续性": {"score": 0, "max": 3, "available": True, "desc": "今日流出"},
        })
        assert [s["name"] for s in r["scored"]] == ["方向", "持续性"]
        assert [u["name"] for u in r["unavailable"]] == ["结构"]
        assert "无单型拆分" in r["unavailable"][0]["desc"]

    def test_all_zero_stays_scored(self):
        """全 0 是**数据结果**，必须留在 scored（页面据此提示"不是渲染失败"）"""
        from core.chat_agent import split_fund5_dims
        r = split_fund5_dims({"a": {"score": 0, "max": 6}, "b": {"score": 0, "max": 5}})
        assert len(r["scored"]) == 2 and r["unavailable"] == []

    def test_legacy_breakdown_without_available_flag(self):
        """旧数据无 available 字段 → 视为可评（向后兼容），不能整块消失"""
        from core.chat_agent import split_fund5_dims
        r = split_fund5_dims({"方向": {"score": 6, "max": 6, "desc": "x"}})
        assert len(r["scored"]) == 1 and r["unavailable"] == []

    def test_empty(self):
        from core.chat_agent import split_fund5_dims
        assert split_fund5_dims({}) == {"scored": [], "unavailable": []}
