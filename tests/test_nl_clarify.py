"""选项式澄清契约测试（`core/nl_clarify.py`）

守护三件事：

1. **不猜**：给不出候选映射的片段，保持 `unparsed` 报错，**不得**硬塞一个指标。
2. **选项合法**：每个选项都是一个能过白名单校验的 `Condition`
   （规则表写错必须在模块加载时就暴露，而不是等用户点下去才炸）。
3. **不误导**：量级反问的提示要**按单位给对**——给换手率提示"万元当亿元"是错的。
"""

import pytest

from core import nl_clarify as nc
from core import nl_screener as nl
from core.nl_screener import NLQuery, parse


# ================================================================
# 1. 规则表自身的合法性
# ================================================================

def test_rule_table_self_check_passes():
    assert nc.RULE_PROBLEMS == [], nc.RULE_PROBLEMS


def test_all_option_fields_are_in_whitelist():
    from core.indicator_registry import INDICATORS
    for rule in nc._RULES:
        for _, conds in rule["options"]:
            for c in conds:
                assert c["field"] in INDICATORS, c


def test_triggers_are_not_empty():
    for rule in nc._RULES:
        assert rule["triggers"], rule
        assert rule["options"], rule
        labels = [label for label, _ in rule["options"]]
        assert all(len(x) >= 4 for x in labels), f"选项描述过短：{labels}"
        assert len(set(labels)) == len(labels), f"选项重复：{labels}"


# ================================================================
# 2. 命中：模糊表达 → 候选映射
# ================================================================

@pytest.mark.parametrize("text,expect_trigger", [
    ("便宜一点的股票", "便宜"),
    ("帮我找小盘股", "小盘股"),
    ("大盘股为主", "大盘股"),
    ("走势强势的票", "强势"),
    ("交投活跃的", "活跃"),
    ("要稳健的", "稳健"),
    ("主力进场的", "主力进场"),
    ("消息面好的", "消息面好"),
    ("优质公司", "优质"),
])
def test_high_frequency_phrases_get_clarified(text, expect_trigger):
    q = parse(text)
    questions = nc.detect(q)
    assert questions, f"{text} 应当触发澄清"
    assert any(expect_trigger in qq.question for qq in questions)


def test_option_conditions_are_usable():
    """点下去的条件必须是**可直接执行**的合法 Condition"""
    q = parse("便宜一点的股票")
    questions = nc.detect(q)
    assert questions
    for opt in questions[0].options:
        for cond in opt.conditions:
            assert cond.field in nl.METRICS
            assert cond.op in nl.OPS
            assert cond.source == "clarify"
            # 必须能被核对逻辑接受
            report = nl.evaluate({cond.field: cond.value}, NLQuery(conditions=[cond]))
            assert report is not None


# ================================================================
# 3. 不猜：给不出映射就不产出澄清
# ================================================================

def test_unknown_phrase_yields_no_clarification():
    """「帮我看看今天有什么机会」没有任何可映射的指标 —— 保持原样报错，不猜"""
    q = parse("帮我看看今天有什么机会")
    assert nc.detect(q) == []
    assert q.unparsed, "仍应保留 unparsed 报错"


def test_valid_condition_yields_no_clarification():
    q = parse("市盈率低于20")
    assert nc.detect(q) == []
    assert not q.unparsed


def test_partial_match_keeps_other_fragments_unparsed():
    """一个片段被澄清、另一个仍无法映射 → 后者必须继续报错"""
    q = parse("便宜的股票，还有那种有想象空间的")
    questions = nc.detect(q)
    assert any("便宜" in c.question for c in questions)
    assert len(questions) == 1, "「有想象空间」不构成可映射表达，不应产出澄清"


def test_trigger_dedup():
    """同一触发词出现多次只问一次（否则页面会被同一个问题刷屏）"""
    q = parse("便宜的股票，要很便宜的那种")
    questions = nc.detect(q)
    assert len([c for c in questions if "便宜" in c.question]) == 1


# ================================================================
# 4. 量级反问：按单位给对提示
# ================================================================

def test_market_cap_out_of_range_asks_about_unit():
    q = parse("市值大于500000")
    questions = nc.sanity_questions(q)
    assert questions
    assert "市值" in questions[0].question
    assert "万元" in questions[0].question, "亿元类应提示「万元/元」混淆"


def test_turnover_out_of_range_asks_about_percent():
    q = parse("换手率大于200%")
    questions = nc.sanity_questions(q)
    assert questions
    assert "100" in questions[0].question, "百分数类应提示 ×100 换算，而不是「万元当亿元」"


def test_change_pct_beyond_limit_flagged():
    q = parse("今日涨幅大于30%")
    assert nc.sanity_questions(q), "超出涨跌停限制应当被反问"


@pytest.mark.parametrize("text", [
    "市盈率低于20", "市值大于500亿", "换手率低于5%", "今日涨幅大于2%",
])
def test_normal_values_not_flagged(text):
    assert nc.sanity_questions(parse(text)) == []


def test_negative_pe_not_flagged():
    """亏损股 PE 为负是正常现象"""
    assert nc.sanity_questions(parse("市盈率低于-10")) == []


# ================================================================
# 5. 合并：同指标替换 + 清掉已解答的 unparsed
# ================================================================

def test_merge_replaces_same_field():
    q = parse("市盈率低于20")
    cond = nl.Condition(field="pe", op="lt", value=15, source="clarify")
    merged = nc.merge_condition(q, cond)
    assert len(merged.conditions) == 1
    assert merged.conditions[0].value == 15


def test_merge_appends_new_field():
    q = parse("市盈率低于20")
    cond = nl.Condition(field="turnover", op="gt", value=5, source="clarify")
    merged = nc.merge_condition(q, cond)
    assert {c.field for c in merged.conditions} == {"pe", "turnover"}


def test_merge_clears_answered_unparsed_item():
    """点了选项之后，页面不该还在警告「便宜」没有对应的可执行指标"""
    q = parse("便宜的股票")
    assert q.unparsed
    question = nc.detect(q)[0]
    cond = question.options[0].conditions[0]
    merged = nc.merge_condition(q, cond, question.fragment)
    assert merged.unparsed == [], merged.unparsed
    assert merged.conditions


def test_merge_without_fragment_falls_back():
    q = parse("便宜的股票")
    cond = nl.Condition(field="pe", op="lt", value=15, source="clarify")
    merged = nc.merge_condition(q, cond)
    assert isinstance(merged, NLQuery)


def test_merge_preserves_sector_and_st_flag():
    q = parse("便宜的医药股，包括ST")
    cond = nl.Condition(field="pe", op="lt", value=15, source="clarify")
    merged = nc.merge_condition(q, cond, nc.detect(q)[0].fragment if nc.detect(q) else None)
    assert merged.sector == "医药"
    assert merged.exclude_st is False
