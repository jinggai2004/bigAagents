"""A股多Agent分析系统 - LangGraph 图编排 (多轮辩论增强版 + 择时+风控)

流程:
START → timing (择时) → 数据获取 → [并行: 技术/基本面/舆情] → bull → bear → [debate]
                                                          ↓
                                        research_manager → trader → risk (系统风控) → portfolio → END
"""

import logging
import os

from langgraph.graph import END, START, StateGraph
from core.state import AnalysisState
from core.agents import (
    data_fetcher_node,
    tech_analyst_node,
    fundamental_analyst_node,
    news_sentiment_node,
    bull_researcher_node,
    bear_researcher_node,
    research_manager_node,
    trader_node,
    risk_manager_node,
    portfolio_manager_node,
)
from core.timing import from_market_signal, get_timing_engine, TimingSignal
from core.risk import get_risk_engine, RiskReport

# 择时口径来源（2026-10-20）：默认 `market_detector`，可回滚到 `timing_engine`。
# ⚠️ 为什么要有开关：统一口径会改变 `analyses.market_score` 的历史可比性，
#    必须能一键退回；且回滚不需要改代码（改环境变量 + 重启即可）。
ENV_TIMING_SOURCE = "BIGA_TIMING_SOURCE"
import logging

_logger = logging.getLogger(__name__)


def _debate_router(state: AnalysisState) -> str:
    """辩论条件路由：继续 or 终止

    v2 变更（M0）：删除"论文字符数增长 <10% 即视为收敛"的判据——用 LLM 散文的
    字符长度充当收敛信号与信息量无关。改为纯计数器驱动，并增加"空头本轮未产出
    任何论点即终止"的保险，保证任何情况下都不会无限回边。
    """
    current_round = state.get("debate_count") or 0
    max_rounds = state.get("max_debate_rounds") or 2

    if current_round >= max_rounds:
        return "research_manager"

    # 保险：空头本轮未产出论点（上游失败/短路）→ 终止，不再回边
    if not state.get("bear_argument") and not state.get("bear_history"):
        return "research_manager"

    return "bull_researcher"


def timing_node(state: AnalysisState) -> dict:
    """
    择时节点：在选股前执行多周期共振判断。

    更新 state 中的:
      - timing_signal: TimingSignal
      - position_limit: float

    ⚠️ 2026-10-20：默认改用 `MarketDetector`（与选股/复盘同一套口径）。
       此前分析图用 `TimingEngine`、选股用 `MarketDetector`，同日两套分数
       （16.9/bear vs 37.9/偏弱震荡）⇒ 选出来的票逐个被分析否掉，整批白跑。
       详见 `core.timing.from_market_signal` 的说明。

    降级链（如实，不补造）：
      market_detector 失败 → timing_engine → 中性兜底（range/0.3 + 明确写出降级）
    """
    _logger.info("\n[Timing] 执行择时判断...")
    source = (os.getenv(ENV_TIMING_SOURCE, "market_detector") or "").strip().lower()
    signal = None
    if source != "timing_engine":
        try:
            from core.market_detector import MarketDetector
            signal = from_market_signal(MarketDetector().detect())
        except Exception as e:                  # noqa: BLE001
            _logger.warning("[Timing] MarketDetector 失败，退回 TimingEngine：%s: %s",
                            type(e).__name__, e)
    if signal is None:
        try:
            signal = get_timing_engine().detect()
        except Exception as e:                  # noqa: BLE001
            # 两套都失败：给中性兜底，但**必须留下痕迹**——不能让"取数失败"
            # 在下游看起来像"市场中性"
            _logger.warning("[Timing] TimingEngine 也失败，使用中性兜底：%s: %s",
                            type(e).__name__, e)
            signal = TimingSignal(
                regime="range", position_limit=0.3, total_score=50.0,
                breakdown={}, source="fallback",
                summary="⚠️ 择时数据不可用（两个引擎均失败），按中性 50 分处理，请勿据此判断市场")

    _logger.info("[Timing] (%s) %s", signal.source, signal.summary)
    state["timing_signal"] = signal.to_dict()
    state["position_limit"] = signal.position_limit
    return {"timing_signal": signal.to_dict(), "position_limit": signal.position_limit}


def risk_node(state: AnalysisState) -> dict:
    """
    风控节点：在选股结果出来后，进行三层风控评估。

    输入:
      - state.candidates: 选股结果 (从 screener 输出)
      - state.timing_signal: 择时信号

    输出:
      - state.risk_report: RiskReport (含仓位计划+止损止盈)
      - state.risk_warnings: list[str]
    """
    _logger.info("\n[Risk] 执行风控评估...")

    candidates = state.get("candidates", [])
    timing_signal_dict = state.get("timing_signal", {})

    # 构造 TimingSignal 对象
    timing = TimingSignal(
        regime=timing_signal_dict.get("regime", "range"),
        position_limit=timing_signal_dict.get("position_limit", 0.5),
        total_score=timing_signal_dict.get("total_score", 50),
        breakdown=timing_signal_dict.get("breakdown", {}),
        summary=timing_signal_dict.get("summary", ""),
    )

    engine = get_risk_engine()
    report = engine.assess(candidates, timing)

    _logger.info(f"[Risk] 风控评估完成: 风险等级={report.overall_risk}, 仓位计划={len(report.position_plans)}只")

    state["risk_report"] = report.to_dict()
    state["risk_warnings"] = report.warnings
    return {"risk_report": report.to_dict(), "risk_warnings": report.warnings}


def build_analysis_graph() -> StateGraph:
    """构建并返回编译后的分析图（多轮辩论版 + 择时风控）"""

    workflow = StateGraph(AnalysisState)

    # ==== 1. 添加所有节点 ====
    workflow.add_node("timing", timing_node)                     # 新增：择时
    workflow.add_node("data_fetcher", data_fetcher_node)
    workflow.add_node("tech_analyst", tech_analyst_node)
    workflow.add_node("fundamental_analyst", fundamental_analyst_node)
    workflow.add_node("news_analyst", news_sentiment_node)
    workflow.add_node("bull_researcher", bull_researcher_node)
    workflow.add_node("bear_researcher", bear_researcher_node)
    workflow.add_node("research_manager", research_manager_node)
    workflow.add_node("trader", trader_node)
    workflow.add_node("risk_manager", risk_manager_node)         # 原有 agent 风控
    workflow.add_node("system_risk", risk_node)                  # 新增：系统级风控
    workflow.add_node("portfolio_manager", portfolio_manager_node)

    # ==== 2. 定义边 ====

    # START → 择时（先在选股前判断市场环境）
    workflow.add_edge(START, "timing")

    # 择时 → 数据获取
    workflow.add_edge("timing", "data_fetcher")

    # 数据获取 → 3个分析师并行
    workflow.add_edge("data_fetcher", "tech_analyst")
    workflow.add_edge("data_fetcher", "fundamental_analyst")
    workflow.add_edge("data_fetcher", "news_analyst")

    # 3个分析师 → 全部汇聚到 Bull Researcher
    workflow.add_edge("tech_analyst", "bull_researcher")
    workflow.add_edge("fundamental_analyst", "bull_researcher")
    workflow.add_edge("news_analyst", "bull_researcher")

    # Bull → Bear
    workflow.add_edge("bull_researcher", "bear_researcher")

    # Bear → 辩论控制器（条件路由）
    workflow.add_conditional_edges(
        "bear_researcher",
        _debate_router,
        {
            "bull_researcher": "bull_researcher",
            "research_manager": "research_manager",
        },
    )

    # 研究经理 → 交易员 → agent风控 → 系统风控 → 组合经理 → END
    workflow.add_edge("research_manager", "trader")
    workflow.add_edge("trader", "risk_manager")
    workflow.add_edge("risk_manager", "system_risk")
    workflow.add_edge("system_risk", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    # ==== 3. 编译 ====
    return workflow.compile()


# 全局图实例
analysis_graph = build_analysis_graph()