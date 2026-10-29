"""测试 LangGraph 图编排 graph_builder.py"""

import pytest
from core.graph_builder import _debate_router, build_analysis_graph


def make_state(debate_count=1, max_rounds=2, bull_hist=None, bear_hist=None):
    return {
        "debate_count": debate_count,
        "max_debate_rounds": max_rounds,
        "bull_history": bull_hist or [],
        "bear_history": bear_hist or [],
    }


# ===== _debate_router 测试 =====

def test_debate_router_max_rounds_reached():
    s = make_state(debate_count=2, max_rounds=2)
    assert _debate_router(s) == "research_manager"


def test_debate_router_exceeds_max():
    s = make_state(debate_count=5, max_rounds=2)
    assert _debate_router(s) == "research_manager"


def test_debate_router_continue_first_round():
    s = make_state(debate_count=1, max_rounds=3,
                   bull_hist=["论点A1"], bear_hist=["论点B1"])
    assert _debate_router(s) == "bull_researcher"


def test_debate_router_terminates_when_bear_produced_nothing():
    """v2 回归：空头本轮无任何产出（上游失败/短路）→ 直接进研究经理，绝不回边。

    原实现在此情形下返回 bull_researcher，而 debate_count 又因节点短路不再递增，
    于是 bull↔bear 无限循环直到 LangGraph 递归上限报错。
    """
    s = make_state(debate_count=1, max_rounds=3)
    assert _debate_router(s) == "research_manager"


def test_debate_router_missing_counter_is_safe():
    """计数器缺失（0）但空头已有论点 → 继续，且计数上限保证最终终止"""
    s = {"max_debate_rounds": 2, "bear_argument": "空头R1", "bear_history": ["空头R1"]}
    assert _debate_router(s) == "bull_researcher"


def test_debate_router_has_no_character_length_heuristic():
    """v2：已删除"论文字符数增长<10% 即视为收敛"的判据（与信息量无关）"""
    s = make_state(debate_count=2, max_rounds=3,
                   bull_hist=["短论点" * 10, "短论点" * 10],
                   bear_hist=["空论点" * 10, "空论点" * 10])
    s["bear_argument"] = "空论点"
    assert _debate_router(s) == "bull_researcher"


def test_debate_router_always_terminates_at_max_rounds():
    """无论论点内容如何，轮数达到上限必须终止"""
    for count in (2, 3, 5):
        s = make_state(debate_count=count, max_rounds=2,
                       bull_hist=["a"], bear_hist=["b"])
        s["bear_argument"] = "b"
        assert _debate_router(s) == "research_manager"


# ===== 图结构测试 =====

def test_graph_compiles():
    graph = build_analysis_graph()
    assert graph is not None


def test_graph_has_all_nodes():
    graph = build_analysis_graph()
    node_names = list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []
    required = [
        "data_fetcher", "tech_analyst", "fundamental_analyst", "news_analyst",
        "bull_researcher", "bear_researcher", "research_manager",
        "trader", "risk_manager", "portfolio_manager",
    ]
    for node in required:
        assert node in node_names, f"缺失节点: {node}"
