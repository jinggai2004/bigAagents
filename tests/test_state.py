"""测试 AnalysisState 类型定义完整性"""

from core.state import AnalysisState


REQUIRED_KEYS = [
    "stock_code", "stock_name", "analysis_date",
    "stock_info", "kline_data", "indicators", "fund_flow_data",
    # v3（2026-09-28）：报告口径。此前 analysis_date 算了却没传下去、报告里一个日期都没有，
    # 模型只能抄训练语料里的模板头（实测「分析日期：2024年X月X日」）。
    "data_through", "data_availability",
    "tech_report", "fundamental_report", "news_report",
    "bull_argument", "bear_argument", "debate_count",
    "max_debate_rounds", "bull_history", "bear_history",
    "research_summary", "trade_plan", "risk_assessment", "final_decision",
    "screener_score", "screener_factors", "screener_sector",
    # v2（M0）：这些字段原本由节点写入但未声明，LangGraph 只维护 schema 内通道，
    # 导致 timing_signal / risk_report 被丢弃、系统级风控空转。
    "candidates", "timing_signal", "position_limit",
    "risk_report", "risk_warnings", "fatal_errors",
    "messages", "errors",
]


def test_state_has_all_required_keys():
    missing = [k for k in REQUIRED_KEYS if k not in AnalysisState.__annotations__]
    assert missing == [], f"缺失字段: {missing}"


def test_state_field_count():
    actual = len(AnalysisState.__annotations__)
    expected = len(REQUIRED_KEYS)
    extra = set(AnalysisState.__annotations__) - set(REQUIRED_KEYS)
    assert actual == expected, f"字段数不匹配: 期望{expected}, 实际{actual}, 多出={extra}"


# ===== 并发写入的字段必须挂 reducer（2026-09-28 实测踩到）=====
#
# 三个分析师节点是并行执行的，而 LangGraph 对**未挂 reducer** 的 key 每个 step 只接受
# 一个值，两个节点同时写同一个 key 会直接抛：
#   InvalidUpdateError: At key 'xxx': Can receive only one value per step.
# 这个错误是在跑通整张图之后才暴露的（单元测试逐个节点调用永远发现不了），
# 所以在此把"哪些 key 必须有 reducer"固化成断言。

def test_concurrently_written_keys_have_reducers():
    for key in ("errors", "fatal_errors", "data_availability"):
        ann = AnalysisState.__annotations__[key]
        assert getattr(ann, "__metadata__", None), \
            f"{key} 会被多个并行节点写入，必须声明 Annotated[..., reducer]"


def test_merge_dict_reducer_merges_instead_of_overwriting():
    from core.state import _merge_dict
    assert _merge_dict({"a": True}, {"b": False}) == {"a": True, "b": False}
    assert _merge_dict({"a": True}, {"a": False}) == {"a": False}
    assert _merge_dict(None, {"a": 1}) == {"a": 1}
