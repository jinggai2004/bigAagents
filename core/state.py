"""A股多Agent分析系统 - 状态定义"""

import operator
from typing import TypedDict, List, Optional, Annotated, Dict, Any
from langgraph.graph.message import add_messages


def _merge_dict(a: dict, b: dict) -> dict:
    """两个 dict 浅合并（后者覆盖前者）——用于并发写入的 `data_availability`。

    三个分析师节点是**并行**执行的，LangGraph 对同一 key 每个 step 只接受一个值，
    并发写会直接抛 `InvalidUpdateError: Can receive only one value per step`
    （2026-09-28 实测踩到）。所以这类"多方各写一块"的字段必须挂 reducer。
    """
    return {**(a or {}), **(b or {})}


# 通道 reducer 的**手工等价实现**：给"用 stream() 自己累积状态"的调用方用。
# 为什么要它：图内部会按 reducer 自动归并，但 UI（ui/pages/2_深度分析.py）和
# pipeline（core/pipeline.py）为了报进度 / 汇总，都是手工 `final_state.update(delta)`，
# 而 `dict.update` 是**覆盖**语义——不按同样的规则归并，整条流程只会保住
# 最后一个节点写的内容（errors、审计告警会丢）。
_ADDITIVE_KEYS = ("errors", "fatal_errors")


def merge_delta(state: dict, delta: dict) -> dict:
    """按 reducer 语义把一次节点更新合并进 state 快照（返回新的 delta）"""
    delta = dict(delta or {})
    for key in _ADDITIVE_KEYS:
        if key in delta:
            delta[key] = list((state or {}).get(key) or []) + list(delta[key] or [])
    if "data_availability" in delta:
        delta["data_availability"] = _merge_dict(
            (state or {}).get("data_availability"), delta["data_availability"])
    return delta


class AnalysisState(TypedDict):
    """LangGraph 全局状态"""

    # === 输入 ===
    stock_code: str           # 股票代码（6位）
    stock_name: str           # 股票名称
    analysis_date: str        # 分析日期

    # === 股票数据（由数据获取阶段填充） ===
    stock_info: dict          # 基本信息 {symbol, name, current_price, change_percent, pe_ratio, pb_ratio...}
    kline_data: Any           # 历史K线 DataFrame
    indicators: dict          # 最新技术指标 {price, rsi, macd, ma5, ma10, ma20, ma60...}
    fund_flow_data: dict      # 【新增】资金流向原始数据

    # === 数据口径（v3 2026-09-28）===
    # 此前 analysis_date 在 data_fetcher_node 里算了却**没有任何分析师节点把它传下去**，
    # 三个分析师提示词里一个日期都没有 → 模型只能抄训练语料里的模板头
    # （实测输出「分析日期：2024年X月X日」）。现在两个维度一起注入：
    data_through: str         # 数据截止 = 输入中最后一根 K 线的日期（数据自带）
    data_availability: Annotated[dict, _merge_dict]  # 各输入是否真的取到（供报告头部显示"缺了什么"）

    # === 分析师输出（3个分析师并行） ===
    tech_report: str          # 技术面分析报告
    fundamental_report: str   # 基本面分析报告
    news_report: str          # 新闻舆情分析报告

    # === 辩论层 ===
    bull_argument: str        # 多头论点（最新一轮）
    bear_argument: str        # 空头论点（最新一轮）
    debate_count: int         # 辩论轮数计数
    max_debate_rounds: int    # 最大辩论轮数（默认2）
    bull_history: list[str]   # 多头各轮论点历史
    bear_history: list[str]   # 空头各轮论点历史

    # === 决策层 ===
    research_summary: str     # 研究汇总（研究经理输出）
    trade_plan: dict          # 交易计划 {direction, position, target, stop_loss...}
    risk_assessment: str      # 风控审核意见
    final_decision: str       # 最终决策（组合经理输出）

    # === 选股阶段 ===
    screener_score: float      # 选股综合评分
    screener_factors: dict     # 因子拆分明细
    screener_sector: str       # 所属板块

    # === 择时 / 风控 / 候选（v2：显式声明为 LangGraph 通道）===
    # 这些字段原本由节点写入但未在 schema 中声明；LangGraph 只维护 schema 内的通道，
    # 因而 timing_signal / risk_report 等被静默丢弃，risk_node 只能以空 candidates
    # 评估，系统级风控（仓位计划/止损止盈）实际空转。此处显式声明。
    candidates: list           # 选股候选列表（供系统级风控评估）
    timing_signal: dict        # 择时信号（TimingSignal.to_dict()）
    position_limit: float      # 仓位上限（来自择时）
    risk_report: dict          # 系统级风控报告（RiskReport.to_dict()）
    risk_warnings: list        # 风控告警

    # === 元数据 ===
    messages: Annotated[list, add_messages]  # LangGraph 消息历史
    # errors / fatal_errors 用 `operator.add` 归并：三个分析师节点**并行**执行，
    # 而 LangGraph 对未挂 reducer 的 key 每 step 只接受一个值——两个分析师同时
    # 记录错误就会抛 InvalidUpdateError，整条分析链直接报错。
    # 因此各写入点只返回**新增**的条目，由通道负责拼接。
    errors: Annotated[list, operator.add]        # 非致命错误 / 降级记录（不阻断分析链）
    fatal_errors: Annotated[list, operator.add]  # 致命错误（阻断分析链：数据缺失到无法分析）
