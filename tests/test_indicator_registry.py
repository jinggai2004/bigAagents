"""指标注册表契约测试（`core/indicator_registry.py`）

守护四件事：

1. **元数据完整性**：12 个指标齐备，别名跨指标不冲突，`source` 取值合法。
2. **量级约束能抓住单位错**：本项目已因"同列名不同单位"栽过三次
   （融资余额 元/亿、同花顺 `'20.03%'` 被金额解析器吃成 0、成交量当成交额）。
   `sanity` 区间是唯一能自动发现这类错的手段 —— 所以这里**用真实的错误值**去撞它。
3. **单一来源**：`nl_screener.METRICS` 必须与注册表逐项一致（防止两处漂移）。
4. **prompt 由注册表生成**：新增指标必须自动出现在提示词里（手拼字符串会漏）。
"""

import pytest

from core import indicator_registry as reg
from core.indicator_registry import INDICATORS, SOURCE_LABELS

SNAPSHOT_KEYS = {"pe", "market_cap", "turnover", "amount", "change_pct"}
PRECISION_KEYS = {"momentum20", "volatility", "ma_bullish", "ma20_above",
                  "fund_net", "news_positive", "score"}


# ================================================================
# 1. 元数据完整性
# ================================================================

def test_exactly_twelve_indicators():
    assert len(INDICATORS) == 12
    assert set(INDICATORS) == SNAPSHOT_KEYS | PRECISION_KEYS


def test_registry_key_matches_indicator_key():
    """字典键与 `Indicator.key` 不一致会造出"幽灵指标"（列表里有、取不到）"""
    for k, v in INDICATORS.items():
        assert k == v.key


def test_every_indicator_is_well_formed():
    for k, v in INDICATORS.items():
        assert v.label, k
        assert v.kind in ("num", "bool"), k
        assert v.aliases, k
        assert v.source in SOURCE_LABELS, k
        assert v.unit is not None, k
        if v.kind == "num":
            assert v.unit, f"{k} 是数值指标，必须有单位"


def test_bool_indicators_have_no_unit_or_sanity():
    for k, v in INDICATORS.items():
        if v.kind == "bool":
            assert v.unit == "", k
            assert v.sanity is None, f"{k} 是形态类，不该有量级区间"


def test_aliases_unique_across_indicators():
    """同一别名挂到两个指标上 → 解析结果取决于遍历顺序（破坏可复现性）"""
    seen = {}
    for k, v in INDICATORS.items():
        for a in v.aliases:
            assert a not in seen, f"别名「{a}」同时属于 {seen[a]} 与 {k}"
            seen[a] = k


def test_long_aliases_present_for_disambiguation():
    """长别名必须在（否则"20日涨幅"会被"涨幅"抢走，判成当日涨跌幅）"""
    assert "近20日涨幅" in INDICATORS["momentum20"].aliases
    assert "涨幅" in INDICATORS["change_pct"].aliases


# ================================================================
# 2. 分层：粗筛可用的只有快照类
# ================================================================

def test_snapshot_and_precision_split():
    assert set(reg.snapshot_keys()) == SNAPSHOT_KEYS
    assert set(reg.precision_keys()) == PRECISION_KEYS
    assert not (set(reg.snapshot_keys()) & set(reg.precision_keys()))


def test_only_five_are_snapshot_sourced():
    """粗筛层只有这 5 个可见 —— 它们原先全都依赖问财初筛，现由新浪行业明细提供"""
    snap = [v.key for v in INDICATORS.values() if v.is_snapshot]
    assert set(snap) == SNAPSHOT_KEYS


# ================================================================
# 3. 量级约束：用真·错误值去撞
# ================================================================

class TestSanityCatchesUnitErrors:
    """每条都对应一次真实踩过的单位错"""

    def test_market_cap_in_wan_is_rejected(self):
        """忘了把「万元」换成「亿元」→ 15776361 亿元（比 A 股总市值高两个数量级）"""
        bad = {"market_cap": 15776361.845}
        assert reg.sanity_violations(bad, "market_cap"), "万元当亿元用必须被抓到"

    def test_amount_in_yuan_is_rejected(self):
        """忘了把「元」换成「亿元」→ 3761312051"""
        assert reg.sanity_violations({"amount": 3761312051.0}, "amount")

    def test_turnover_double_scaled_is_rejected(self):
        """日线 turnover 是小数、行业是百分数 —— 多乘一次 100 会得到 239%"""
        assert reg.sanity_violations({"turnover": 239.0}, "turnover")

    def test_change_pct_beyond_limit_is_rejected(self):
        """A 股涨跌停 ±20%，出现 ±30% 说明口径错了"""
        assert reg.sanity_violations({"change_pct": -33.0}, "change_pct")

    def test_news_positive_over_100_is_rejected(self):
        assert reg.sanity_violations({"news_positive": 150.0}, "news_positive")

    @pytest.mark.parametrize("key,value", [
        ("pe", 47.985), ("market_cap", 1577.636), ("turnover", 2.39085),
        ("amount", 37.613), ("change_pct", -0.705),
        ("momentum20", 12.5), ("volatility", 34.2), ("news_positive", 66.7),
        ("fund_net", -3.2), ("score", 78.4),
    ])
    def test_real_values_pass(self, key, value):
        """实测真实值必须落在区间内（区间太紧会把正常数据判成错）"""
        assert reg.sanity_violations({key: value}, key) == []

    def test_negative_pe_is_allowed(self):
        """亏损股 PE 为负是正常现象，不能被当成口径错误"""
        assert reg.sanity_violations({"pe": -35.0}, "pe") == []

    def test_none_and_bool_are_skipped(self):
        """缺值与布尔不当成越界（缺就是缺，不参与量级判断）"""
        assert reg.sanity_violations({"pe": None}, "pe") == []
        assert reg.sanity_violations({"ma_bullish": True}, "ma_bullish") == []

    def test_check_facts_aggregates_all(self):
        facts = {"market_cap": 15776361.0, "turnover": 239.0, "pe": 20.0}
        problems = reg.check_facts(facts)
        assert len(problems) == 2
        assert any("总市值" in p for p in problems)
        assert any("换手率" in p for p in problems)


# ================================================================
# 4. 别名匹配（委托给解析器的那份规则）
# ================================================================

@pytest.mark.parametrize("segment,expect", [
    ("市盈率低于20", "pe"),
    ("20日涨幅超过10%", "momentum20"),        # 长别名优先，不能被"涨幅"抢走
    ("今日涨幅大于3%", "change_pct"),
    ("站上20日均线", "ma20_above"),
    ("站上20日线", "ma20_above"),
    ("总市值大于500亿", "market_cap"),
    ("主力资金净流入为正", "fund_net"),
    ("换手率低于5%", "turnover"),
    ("年化波动率低于30%", "volatility"),
    ("综合评分不低于75", "score"),
])
def test_find_metric(segment, expect):
    key, pos, alias = reg.find_metric(segment)
    assert key == expect, f"{segment} → {key}({alias})"


def test_find_metric_returns_none_for_unknown():
    assert reg.find_metric("rsi低于30")[0] is None


def test_find_metric_is_order_independent():
    """同一输入多次调用结果一致（可复现是硬要求）"""
    results = {reg.find_metric("20日涨幅超过10%") for _ in range(20)}
    assert len(results) == 1


# ================================================================
# 5. 单一来源：与 nl_screener 不得漂移
# ================================================================

def test_nl_screener_metrics_mirror_registry():
    """`nl_screener.METRICS` 是从注册表派生的，必须逐项一致。

    若有人手改了其中一份，解析器与执行层就会用两套元数据——
    最典型的症状是"注册表加了字段，解析器不认"（不报错，只是少解析）。
    """
    from core import nl_screener as nl
    assert set(nl.METRICS) == set(INDICATORS)
    for k, v in INDICATORS.items():
        m = nl.METRICS[k]
        assert m["label"] == v.label
        assert m["unit"] == v.unit
        assert m["kind"] == v.kind
        assert m["source"] == v.source
        assert list(m["aliases"]) == list(v.aliases)
        assert m["sanity"] == v.sanity


def test_parser_uses_registry_alias_table():
    """解析器与注册表用的是**同一份**别名表（不是各留一份）"""
    from core import nl_screener as nl
    for k, v in INDICATORS.items():
        for alias in v.aliases:
            found, _, _ = nl._find_metric(alias)
            assert found == k, f"别名「{alias}」在解析器里没落到 {k}"


# ================================================================
# 6. 提示词由注册表生成
# ================================================================

def test_prompt_contains_every_registry_field():
    """新增指标必须自动出现在提示词里（手拼字符串做不到这一点）"""
    text = reg.describe_for_prompt()
    for k in INDICATORS:
        assert k in text, f"提示词里缺 {k}"
    for v in INDICATORS.values():
        assert v.label in text


def test_nl_llm_prompt_includes_all_fields():
    from core import nl_screener as nl
    prompt = nl.llm_system_prompt()
    for k in INDICATORS:
        assert k in prompt


def test_ui_table_has_one_row_per_indicator():
    rows = reg.describe_for_ui()
    assert len(rows) == len(INDICATORS)
    assert {"指标", "键", "单位", "类型", "取数口径", "粗筛层可用"} <= set(rows[0])


def test_ui_table_marks_snapshot_layer():
    rows = {r["键"]: r for r in reg.describe_for_ui()}
    for k in SNAPSHOT_KEYS:
        assert rows[k]["粗筛层可用"] == "是"
    for k in PRECISION_KEYS:
        assert rows[k]["粗筛层可用"].startswith("否")
