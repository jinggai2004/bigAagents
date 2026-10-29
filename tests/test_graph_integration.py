"""整图集成测试（v2 / M0 0.3）

为什么需要这个文件：
  tests/test_pipeline.py 用 `patch.object(..., "analysis_graph")` 把整张 LangGraph
  图替换成 mock，因此下面两类真实缺陷永远不会被测试发现：
    1. 节点写了未在 AnalysisState 中声明的字段（timing_signal / risk_report /
       candidates 等），LangGraph 只维护 schema 内的通道 → 字段被静默丢弃，
       risk_node 拿到空 candidates，系统级风控（仓位计划/止损止盈）完全空转；
    2. 任一 errors 非空即让下游节点全部短路 → 可选数据源（资金流）抖动就使整条
       决策链空跑，且 debate_count 不再递增 → 辩论路由无限回边，
       until LangGraph 递归上限报错。

本文件真跑 analysis_graph，只 mock 外部依赖（数据源 / LLM / 择时引擎 / 风控引擎）。
"""

import os

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

import data.news  # noqa: F401  (确保 patch 目标可解析)
from core.graph_builder import analysis_graph
from core.timing import TimingSignal


def _timing_signal() -> TimingSignal:
    return TimingSignal(regime="range", position_limit=0.5, total_score=55.0,
                        breakdown={}, summary="震荡市，建议半仓")


class _CapturingRiskEngine:
    """记录 assess() 收到的 candidates / timing，用于验证风控接线。"""

    def __init__(self):
        self.calls = []

    def assess(self, candidates, timing_signal, current_portfolio=None):
        self.calls.append({"candidates": list(candidates), "timing": timing_signal})
        report = MagicMock()
        report.to_dict.return_value = {
            "overall_risk": "medium",
            "position_plans": [{"code": "000001", "weight": 0.2}],
        }
        report.warnings = ["测试告警：单票集中度偏高"]
        return report


# 基本面节点的财务数据。
# ⚠️ v3 修正（2026-09-28）：**不要 patch `core.agent_tools.tool_financials` 本身**。
# 上一版正是这么写的，于是 `tool_financials` 的真身从未被执行 —— 而它当时被
# `core/agents.py` 用位置参数调用，`_tool` 包装层签名是 `wrapper(**kwargs)`，
# 抛出的 `TypeError: wrapper() takes 0 positional arguments but 1 was given`
# 被工具自己的 except 兜成 `{"error": ...}` → 基本面**永远走"未评估"分支**，
# 而测试全绿（Mock 上任何属性/调用都成立）。
# 现在只 patch 最底层的数据获取叶子：真实 `_tool` 包装 + `tool_financials` 函数体
# 全部真实执行，调用签名一旦对不上，这里立刻红。
_RAW_FINANCIALS = {
    "symbol": "000001",
    "income_statement": [
        {"报告日": "20260630", "营业收入": 1.2e9, "净利润": 2.0e8,
         "归属于母公司股东的净利润": 1.9e8, "营业利润": 2.5e8, "_source": "新浪"},
        {"报告日": "20260331", "营业收入": 6.0e8, "净利润": 1.0e8,
         "归属于母公司股东的净利润": 9.5e7, "营业利润": 1.2e8, "_source": "新浪"},
        {"报告日": "20250630", "营业收入": 1.0e9, "净利润": 1.7e8,
         "归属于母公司股东的净利润": 1.6e8, "营业利润": 2.1e8, "_source": "新浪"},
    ],
    "balance_sheet": [
        {"报告日": "20260630", "资产总计": 5.0e9, "负债合计": 1.5e9},
        {"报告日": "20260331", "资产总计": 4.8e9, "负债合计": 1.4e9},
    ],
    "cash_flow": [
        {"报告日": "20260630", "经营活动产生的现金流量净额": 3.0e8},
    ],
}


def _initial_state(**overrides) -> dict:
    state = {
        "stock_code": "000001", "stock_name": "平安银行", "analysis_date": "",
        "stock_info": {}, "kline_data": None, "indicators": {}, "fund_flow_data": {},
        "data_through": "", "data_availability": {},
        "tech_report": "", "fundamental_report": "", "news_report": "",
        "bull_argument": "", "bear_argument": "", "debate_count": 0,
        "max_debate_rounds": 2, "bull_history": [], "bear_history": [],
        "research_summary": "", "trade_plan": {}, "risk_assessment": "",
        "final_decision": "", "screener_score": 82.5, "screener_factors": {},
        "screener_sector": "主力资金",
        "candidates": [{"股票代码": "000001", "股票名称": "平安银行",
                        "score": 82.5, "板块": "主板深市"}],
        "timing_signal": {}, "position_limit": None,
        "risk_report": {}, "risk_warnings": [],
        "messages": [], "errors": [], "fatal_errors": [],
    }
    state.update(overrides)
    return state

@pytest.fixture
def env():
    """mock 外部依赖，保留真实的图结构与节点逻辑"""
    fetcher = MagicMock()
    fetcher.get_stock_info.return_value = {
        "name": "平安银行", "current_price": 12.5,
        "change_percent": 1.2, "pe_ratio": 6.5,
    }
    fetcher.get_stock_data.return_value = pd.DataFrame(
        {"Close": [10.0 + i * 0.1 for i in range(30)]})
    fetcher.calculate_technical_indicators.side_effect = lambda df: df
    fetcher.get_latest_indicators.return_value = {"ma20": 11.0, "rsi": 55.0}

    llm_instance = MagicMock()
    llm_instance.technical_analysis.return_value = "技术面：均线多头"
    llm_instance.fundamental_analysis.return_value = "基本面：估值偏低"
    llm_instance.call_api.return_value = "（mock 的分析内容）"
    llm_cls = MagicMock(return_value=llm_instance)

    flow_cls = MagicMock()
    flow_cls.return_value.get_fund_flow_data.return_value = {
        "data_success": True,
        "fund_flow_data": {"data": [{"主力净流入-净额": 1_000_000, "date": "2026-01-02"}]},
    }

    risk_engine = _CapturingRiskEngine()

    # ⚠️ 2026-10-20：`timing_node` 默认改走 `MarketDetector`（与选股同口径），
    #    只 patch `get_timing_engine` 会**静默失效** —— 真实 MarketDetector 会去
    #    联网取指数/资金流，测试既慢又受外部接口影响。这里显式锁回旧路径，
    #    新默认路径由 `tests/test_timing_source.py` 覆盖。
    os.environ["BIGA_TIMING_SOURCE"] = "timing_engine"

    with patch("core.agents.StockDataFetcher", return_value=fetcher), \
         patch("core.agents.DeepSeekClient", llm_cls), \
         patch("data.fund_flow.FundFlowAkshareDataFetcher", flow_cls), \
         patch("data.news.NewsFlowDataFetcher") as news_cls, \
         patch("data.stock_data.StockDataFetcher.get_financial_data",
               return_value=dict(_RAW_FINANCIALS)) as fin_fetch, \
         patch("core.graph_builder.get_timing_engine") as timing_engine, \
         patch("core.graph_builder.get_risk_engine", return_value=risk_engine):
        news_cls.return_value.get_news_flow_data.return_value = {
            "data_success": True, "news_list": [{"title": "公告", "time": "2026-01-02"}]}
        timing_engine.return_value.detect.return_value = _timing_signal()
        try:
            yield {"fetcher": fetcher, "llm": llm_instance, "risk_engine": risk_engine,
                   "fin_fetch": fin_fetch}
        finally:
            os.environ.pop("BIGA_TIMING_SOURCE", None)


# ===== 1. 风控接线（原实现恒为空 candidates）=====

def test_graph_wires_candidates_into_risk_engine(env):
    final = analysis_graph.invoke(_initial_state(), config={"recursion_limit": 25})

    assert env["risk_engine"].calls, "系统级风控未被调用"
    got = env["risk_engine"].calls[0]["candidates"]
    assert [c["股票代码"] for c in got] == ["000001"], \
        "risk_node 必须收到候选全集（原实现因未注入 candidates 而恒为空列表）"
    assert final["risk_report"] != {}, "风控报告未写回 state（字段未声明会被丢弃）"
    assert final["risk_warnings"] == ["测试告警：单票集中度偏高"]
    assert final["final_decision"]
    assert final["fatal_errors"] == []


# ===== 2. 择时字段贯通（原实现未声明 → 被静默丢弃）=====

def test_timing_signal_reaches_risk_engine(env):
    final = analysis_graph.invoke(_initial_state(), config={"recursion_limit": 25})

    assert final["position_limit"] == 0.5
    assert final["timing_signal"]["regime"] == "range"
    assert env["risk_engine"].calls[0]["timing"].position_limit == 0.5


# ===== 3. 可选数据源失败只降级：不空跑、不死循环 =====

def test_graph_completes_when_optional_source_fails(env):
    with patch("data.fund_flow.FundFlowAkshareDataFetcher") as flow_cls:
        flow_cls.return_value.get_fund_flow_data.side_effect = RuntimeError("资金流接口超时")
        final = analysis_graph.invoke(_initial_state(), config={"recursion_limit": 25})

    assert final["fatal_errors"] == [], "可选数据源失败不得被当作致命错误"
    assert any("资金流向" in e for e in final["errors"]), "降级必须被记录下来"
    assert final["final_decision"], "非致命错误不应阻断决策链"
    assert final["tech_report"] == "技术面：均线多头"
    # 辩论跑满 2 轮：原实现会在此处无限回边直到递归上限
    assert len(final["bull_history"]) == 2
    assert len(final["bear_history"]) == 2


# ===== 4. 致命错误快速终止（原实现在此处抛 GraphRecursionError）=====

def test_graph_aborts_without_recursion_error_on_fatal(env):
    env["fetcher"].get_stock_info.return_value = {"error": "接口不可达"}
    env["fetcher"].get_stock_data.return_value = {"error": "接口不可达"}

    final = analysis_graph.invoke(_initial_state(), config={"recursion_limit": 25})

    assert final["fatal_errors"], "数据不可用必须标记为致命错误"
    assert final["tech_report"] == ""
    assert final["final_decision"] == ""


# ===== 5. 并行写 state 不得崩（2026-09-28 实测踩到）=====
#
# 三个分析师节点并行执行，LangGraph 对未挂 reducer 的 key 每个 step 只接受一个值：
# 两个节点同时记录 errors 会直接抛
#   InvalidUpdateError: At key 'errors': Can receive only one value per step.
# 逐节点单测（tests/test_agents.py）永远发现不了这类问题——只有真跑整张图才会暴露。

def test_parallel_analysts_can_all_record_errors(env):
    """技术面与舆情在同一 step 都产生"幻觉告警"→ 必须合并而不是崩"""
    env["llm"].technical_analysis.return_value = "分析日期：2024年X月X日，据 Wind 统计……"
    env["llm"].call_api.return_value = "截至2024年6月28日，据巨潮资讯网……"

    final = analysis_graph.invoke(_initial_state(), config={"recursion_limit": 25})

    joined = " | ".join(final["errors"])
    assert "可核实性" in joined, "两份报告的幻觉线索都该被标出来"
    assert "2024" in joined
    assert final["final_decision"], "审计告警不得阻断决策链"


# ===== 6. 报告口径与数据可用性贯通 =====

def test_graph_exposes_report_scope(env):
    final = analysis_graph.invoke(_initial_state(), config={"recursion_limit": 25})

    assert final["analysis_date"], "分析日期必须写进 state（报告口径）"
    # mock 的 K 线是整数索引 → 没有可靠日期 → 必须是空而不是猜一个
    assert final["data_through"] == ""
    avail = final["data_availability"]
    assert avail.get("财务报表") is True      # tool_financials 已 mock 成有数据
    assert avail.get("新闻") is True           # news mock 返回了 1 条
    assert avail.get("基本信息") is True
