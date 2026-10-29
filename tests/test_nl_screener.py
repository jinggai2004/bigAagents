"""core/nl_screener 测试：白名单 DSL、确定性解析、问财映射、本地核对

验收口径（对应开发计划 Phase 3）：
  - ≥10 条自然语言用例 → 期望 DSL；
  - 同一输入两次解析结果一致（可复现）；
  - 白名单：模型/用户给出的未知字段、非法比较符一律拒绝；
  - 解析失败必须显式报错，**不得静默降级到精选池**；
  - 本地核对：命中要给出实际值，数据缺失算未通过，本地拿不到原始值的条件标"未核对"。
"""

import pytest

from core import nl_screener as nl
from core.nl_screener import (METRICS, Condition, NLParseError, NLQuery,
                              evaluate, facts_from_candidate, filter_candidates,
                              annotate_candidates, parse, parse_with_llm,
                              require_conditions, validate_payload)


def cond(query: NLQuery, field: str):
    for c in query.conditions:
        if c.field == field:
            return c
    return None


# ================================================================
# 10+ 条自然语言用例
# ================================================================

class TestParseCases:
    def test_pe_below(self):
        q = parse("帮我找市盈率低于20的股票")
        c = cond(q, "pe")
        assert (c.op, c.value) == ("lt", 20.0)

    def test_fund_positive_and_st(self):
        q = parse("主力资金净流入为正，非ST")
        c = cond(q, "fund_net")
        assert (c.op, c.value) == ("gt", 0.0)
        assert q.exclude_st is True

    def test_momentum_with_sector(self):
        q = parse("20日涨幅超过10%的新能源股票")
        assert (cond(q, "momentum20").op, cond(q, "momentum20").value) == ("gt", 10.0)
        assert q.sector == "新能源"

    def test_ma_bullish_on_chinext(self):
        q = parse("均线多头排列的创业板股票")
        assert cond(q, "ma_bullish").op == "is_true"
        assert q.sector == "创业板"
        assert q.board == "创业板"

    def test_between_and_market_cap_unit(self):
        q = parse("市盈率10到20之间，市值大于500亿")
        c = cond(q, "pe")
        assert (c.op, c.value, c.value2) == ("between", 10.0, 20.0)
        assert cond(q, "market_cap").value == 500.0        # 亿元

    def test_volatility_and_news_rate(self):
        q = parse("波动率低于30%，新闻正面率大于50%")
        assert cond(q, "volatility").value == 30.0
        assert cond(q, "news_positive").value == 50.0

    def test_score_at_least(self):
        q = parse("综合评分不低于80分")
        c = cond(q, "score")
        assert (c.op, c.value) == ("ge", 80.0)

    def test_amount_and_turnover(self):
        q = parse("成交额超过10亿，换手率大于5%")
        assert cond(q, "amount").value == 10.0
        assert cond(q, "turnover").value == 5.0

    def test_ma20_above_and_st_phrase(self):
        q = parse("股价站上20日线，不要ST股")
        assert cond(q, "ma20_above").op == "is_true"
        assert q.exclude_st is True

    def test_wan_unit_conversion(self):
        """金额类 DSL 单位统一是亿元：20000万 → 2亿"""
        q = parse("总市值20000万以下")
        assert cond(q, "market_cap").value == pytest.approx(2.0)

    def test_change_pct_with_decline_word(self):
        """「跌幅超过5%」= 当日涨跌幅 < -5%"""
        q = parse("跌幅超过5%")
        c = cond(q, "change_pct")
        assert (c.op, c.value) == ("lt", -5.0)

    def test_momentum_less_than_with_sector(self):
        q = parse("近20日涨幅不到3%的医药股")
        assert cond(q, "momentum20").op == "lt"
        assert q.sector == "医药"

    def test_sector_only(self):
        q = parse("帮我找新能源板块的股票")
        assert q.conditions == []
        assert q.sector == "新能源"
        assert q.is_empty is False          # 只有板块也算可执行条件
        assert require_conditions(q) is q

    def test_include_st_explicit(self):
        q = parse("包含ST股的市盈率低于20")
        assert q.exclude_st is False


class TestNoSilentDegrade:
    def test_vague_request_reports_reason(self):
        q = parse("找一些最近走势不错、估值便宜的股票")
        assert q.conditions == []
        assert q.is_empty is True
        assert any("走势不错" in u for u in q.unparsed)
        assert any("阈值" in u for u in q.unparsed), q.unparsed

    def test_require_conditions_raises_with_reason(self):
        with pytest.raises(NLParseError) as e:
            require_conditions(parse("找一些最近走势不错的股票"))
        msg = str(e.value)
        assert "没能" in msg and "市盈率低于20" in msg   # 提示里给出正确写法

    def test_empty_text_rejected(self):
        with pytest.raises(NLParseError):
            parse("   ")

    def test_duplicate_condition_kept_first(self):
        q = parse("市盈率低于20，市盈率大于5")
        assert len(q.conditions) == 1
        assert cond(q, "pe").value == 20.0
        assert any("重复" in u for u in q.unparsed)


class TestConjunctionSplitting:
    """连接词必须切开，否则第二个条件会**静默消失**。

    ⚠️ 这是 2026-10-07 实跑条件回测时抓到的真缺陷：
    `_split_segments` 只按标点切 → `"均线多头排列且换手率大于1%"` 是**一个**片段
    → `_parse_segment` 取位置最靠前的指标就返回 → 换手率条件**丢了**，
    而且 `unparsed` 是空的（连"没听懂"都不报）。
    症状最恶劣的地方：回测/选股照常出结果，只是**少了一条约束**，
    用户没有任何线索。所以这里用"条件数"直接断言，而不是断言"没报错"。
    """

    CASES = [
        ("均线多头排列且换手率大于1%", ["ma_bullish", "turnover"]),
        ("20日动量大于0且换手率大于1%", ["momentum20", "turnover"]),
        ("换手率大于1%且20日动量大于0", ["turnover", "momentum20"]),
        ("均线多头排列 且 20日动量大于0 且 换手率大于1%",
         ["ma_bullish", "momentum20", "turnover"]),
        ("市盈率低于20、市值大于500亿、均线多头排列",
         ["pe", "market_cap", "ma_bullish"]),
        ("波动率低于30%和新闻正面率大于50%", ["volatility", "news_positive"]),
        ("市盈率低于20以及市值大于500亿", ["pe", "market_cap"]),
    ]

    @pytest.mark.parametrize("text,fields", CASES)
    def test_all_conditions_parsed(self, text, fields):
        q = parse(text)
        assert [c.field for c in q.conditions] == fields, (
            f"「{text}」解析出的条件与预期不符（明显是漏解析）")

    @pytest.mark.parametrize("text,fields", CASES)
    def test_no_leftover_silently_dropped(self, text, fields):
        """切分对了，就不该再有"残留指标"告警（守卫不能误报）"""
        q = parse(text)
        assert not any("一条片段只解析一个条件" in u for u in q.unparsed), q.unparsed

    def test_leftover_metric_is_reported(self):
        """切不开的写法（无分隔符）也必须**报出来**，不能静默丢"""
        q = parse("均线多头排列换手率大于1%")
        assert any("一条片段只解析一个条件" in u for u in q.unparsed), q.unparsed

    def test_single_condition_has_no_false_alarm(self):
        """单条件（含指标别名自带数字）不得被守卫误报"""
        for text in ("站上20日均线", "均线多头排列的创业板股票",
                     "20日涨幅超过10%的新能源股票", "市盈率低于20"):
            q = parse(text)
            assert not any("一条片段只解析一个条件" in u for u in q.unparsed), (
                text, q.unparsed)


class TestDeterminism:
    CASES = [
        "帮我找市盈率低于20的股票",
        "主力资金净流入为正，非ST",
        "20日涨幅超过10%的新能源股票",
        "均线多头排列的创业板股票",
        "市盈率10到20之间，市值大于500亿",
        "波动率低于30%，新闻正面率大于50%",
        "综合评分不低于80分",
        "成交额超过10亿，换手率大于5%",
        "股价站上20日线，不要ST股",
        "跌幅超过5%",
        "总市值20000万以下",
        "找一些最近走势不错、估值便宜的股票",
    ]

    @pytest.mark.parametrize("text", CASES)
    def test_two_parses_identical(self, text):
        assert parse(text).to_dict() == parse(text).to_dict()

    def test_condition_order_follows_text(self):
        q = parse("市盈率低于20，换手率大于5%，成交额超过10亿")
        assert [c.field for c in q.conditions] == ["pe", "turnover", "amount"]


# ================================================================
# 白名单校验
# ================================================================

class TestValidatePayload:
    def test_valid(self):
        q = validate_payload({"conditions": [{"field": "pe", "op": "lt", "value": 20}],
                              "sector": "科技", "exclude_st": True})
        assert cond(q, "pe").value == 20.0
        assert q.sector == "科技"
        assert q.source == "llm"

    def test_unknown_field_rejected(self):
        with pytest.raises(NLParseError) as e:
            validate_payload({"conditions": [{"field": "rsi", "op": "lt", "value": 30}]})
        assert "rsi" in str(e.value) and "白名单" in str(e.value)

    def test_unknown_op_rejected(self):
        with pytest.raises(NLParseError):
            validate_payload({"conditions": [{"field": "pe", "op": "approx", "value": 20}]})

    def test_bool_metric_needs_is_true(self):
        with pytest.raises(NLParseError):
            validate_payload({"conditions": [{"field": "ma_bullish", "op": "lt", "value": 1}]})

    def test_num_metric_rejects_is_true(self):
        with pytest.raises(NLParseError):
            validate_payload({"conditions": [{"field": "pe", "op": "is_true"}]})

    def test_non_numeric_value_rejected(self):
        with pytest.raises(NLParseError):
            validate_payload({"conditions": [{"field": "pe", "op": "lt", "value": "便宜"}]})
        with pytest.raises(NLParseError):
            validate_payload({"conditions": [{"field": "pe", "op": "lt", "value": True}]})

    def test_between_needs_value2(self):
        with pytest.raises(NLParseError):
            validate_payload({"conditions": [{"field": "pe", "op": "between", "value": 10}]})
        q = validate_payload({"conditions": [{"field": "pe", "op": "between",
                                              "value": 20, "value2": 10}]})
        assert (cond(q, "pe").value, cond(q, "pe").value2) == (10.0, 20.0)  # 自动排序

    def test_bad_sector_and_board(self):
        with pytest.raises(NLParseError):
            validate_payload({"conditions": [], "sector": "白酒"})
        with pytest.raises(NLParseError):
            validate_payload({"conditions": [], "board": "纳斯达克"})

    def test_parses_json_text_with_fences(self):
        text = '```json\n{"conditions":[{"field":"score","op":"ge","value":75}]}\n```'
        q = validate_payload(text)
        assert cond(q, "score").value == 75.0

    def test_garbage_rejected(self):
        with pytest.raises(NLParseError):
            validate_payload("我建议买入茅台")


# ================================================================
# LLM 兜底
# ================================================================

class TestLlmFallback:
    def test_rule_hit_skips_llm(self):
        calls = []

        def llm_fn(system, user):
            calls.append(user)
            return "{}"

        q = parse_with_llm("市盈率低于20", llm_fn=llm_fn)
        assert calls == [], "规则解析成功时不该调用模型（省成本、保可复现）"
        assert cond(q, "pe").value == 20.0

    def test_rule_miss_uses_llm_and_validates(self):
        def llm_fn(system, user):
            assert "白名单" in system or "禁止自造指标" in system
            return '{"conditions":[{"field":"pe","op":"lt","value":15}],"exclude_st":true}'

        q = parse_with_llm("帮我找便宜点的好公司", llm_fn=llm_fn)
        assert cond(q, "pe").value == 15.0
        assert q.source == "llm"
        assert q.raw_text == "帮我找便宜点的好公司"

    def test_llm_invented_field_rejected(self):
        def llm_fn(system, user):
            return '{"conditions":[{"field":"macd_golden","op":"is_true"}]}'

        with pytest.raises(NLParseError) as e:
            parse_with_llm("帮我找个金叉的", llm_fn=llm_fn)
        assert "macd_golden" in str(e.value)

    def test_no_llm_fn_returns_rule_result(self):
        q = parse_with_llm("帮我找便宜点的好公司", llm_fn=None)
        assert q.is_empty is True


# ================================================================
# 问财链路已移除（D1 决策，2026-10-07）—— 守护"没有被悄悄加回来"
#
# 原 `TestWencai`（4 个用例）测的是 `to_wencai()` 的输出格式。移除该函数后，
# 这些断言失去了被测对象；若直接删掉，"问财不能复活"就没人守了。
# 所以它们被替换成下面这组**结构性守护**。
# ================================================================

class TestWencaiRemoved:
    def test_no_wencai_functions(self):
        """`to_wencai` / `local_only_fields` 必须已不存在。

        留着它们，"某些条件只能靠第三方初筛保证"这种状态就会复活——
        而现在的约定是**12 个指标全部本地可核对**。
        """
        assert not hasattr(nl, "to_wencai")
        assert not hasattr(nl, "local_only_fields")

    def test_metrics_have_no_wencai_field(self):
        """`wencai` 字段应已从指标元数据中消失"""
        for key, spec in METRICS.items():
            assert "wencai" not in spec, key

    def test_five_snapshot_metrics_are_locally_available(self):
        """这 5 个原先只能靠问财初筛（local=False），现由本地快照提供"""
        from core.indicator_registry import SOURCE_SNAPSHOT
        snap = {k for k, v in METRICS.items() if v["source"] == SOURCE_SNAPSHOT}
        assert snap == {"pe", "market_cap", "turnover", "amount", "change_pct"}


# ================================================================
# 本地核对
# ================================================================

class TestEvaluate:
    def test_hit_reports_actual_value(self):
        q = parse("市盈率低于20")
        report = evaluate({"pe": 15.3}, q)
        assert report["pass"] is True
        assert "实际 15.3倍" in report["hits"][0]

    def test_miss_reports_actual_value(self):
        q = parse("市盈率低于20")
        report = evaluate({"pe": 25.0}, q)
        assert report["pass"] is False
        assert "实际 25倍" in report["misses"][0]

    def test_missing_data_is_not_a_pass(self):
        q = parse("市盈率低于20")
        report = evaluate({"pe": None}, q)
        assert report["pass"] is False
        assert "数据缺失" in report["misses"][0]

    def test_bool_condition(self):
        q = parse("均线多头排列")
        assert evaluate({"ma_bullish": True}, q)["pass"] is True
        assert evaluate({"ma_bullish": False}, q)["pass"] is False
        assert evaluate({"ma_bullish": None}, q)["pass"] is False

    def test_between(self):
        q = parse("市盈率10到20之间")
        assert evaluate({"pe": 15}, q)["pass"] is True
        assert evaluate({"pe": 25}, q)["pass"] is False

    def test_snapshot_condition_without_row_is_missing_not_unverified(self):
        """快照类条件在**没有快照行**时按"数据缺失"处理，**不得放行**。

        改造前：市值/换手率/成交额/当日涨跌幅这 4 个本地拿不到原始值 →
        标"由问财初筛保证"、不算未通过（等于放行）。改造后它们由本地全市场快照
        提供；调用方没传快照行时，就该如实判"数据缺失 → 未通过"。
        """
        q = parse("市值大于500亿，市盈率低于20")
        report = evaluate({"pe": 15}, q)
        assert report["pass"] is False, "缺数据不得放行"
        assert any("数据缺失" in m for m in report["misses"])

    def test_snapshot_condition_checked_when_row_given(self):
        """传了快照行时，市值条件才能真的核对（这是改造的核心收益）"""
        from core.nl_screener import facts_from_candidate
        cand = {"股票代码": "600519", "股票名称": "贵州茅台", "factors": {}}
        facts = facts_from_candidate(cand, {"market_cap": 21000.0, "pe": 30.0})
        assert facts["market_cap"] == 21000.0

        q = parse("市值大于500亿")
        report = evaluate(facts, q)
        assert report["pass"] is True
        assert any("总市值" in h and "21000" in h for h in report["hits"])

        q2 = parse("市值大于30000亿")
        assert evaluate(facts, q2)["pass"] is False


class TestFilterCandidates:
    def _candidate(self, code, pe, ma_bullish):
        return {"股票代码": code, "股票名称": f"股票{code}", "score": 80.0,
                "factors": {"PE估值": {"score": 10, "max": 15, "status": "ok",
                                       "value": {"pe": pe}},
                            "均线排列": {"score": 10, "max": 10, "status": "ok",
                                         "value": {"bullish": ma_bullish}}}}

    def test_filter_keeps_passing_and_attaches_report(self):
        q = parse("市盈率低于20，均线多头排列")
        kept = filter_candidates([self._candidate("600519", 15, True),
                                  self._candidate("000001", 25, True),
                                  self._candidate("300750", 15, False)], q)
        assert [c["股票代码"] for c in kept] == ["600519"]
        assert kept[0]["nl_report"]["pass"] is True
        assert "pe" not in kept[0]["nl_report"]

    def test_annotate_keeps_all(self):
        q = parse("市盈率低于20")
        rows = annotate_candidates([self._candidate("600519", 15, True),
                                    self._candidate("000001", 25, True)], q)
        assert len(rows) == 2
        assert rows[1]["nl_report"]["pass"] is False

    def test_empty_query_filter_keeps_everything(self):
        q = parse("新能源板块")
        assert len(filter_candidates([self._candidate("600519", 15, True)], q)) == 1


# ================================================================
# facts 抽取 + 与 screener 的契约
# ================================================================

class TestFactsFromCandidate:
    def test_units(self):
        cand = {"股票代码": "600519", "股票名称": "贵州茅台", "score": 82.5,
                "factors": {
                    "PE估值": {"value": {"pe": 30.0}},
                    "趋势": {"value": {"ret20": 0.123, "above_ma20": True}},
                    "均线排列": {"value": {"bullish": True}},
                    "波动率": {"value": {"volatility": 22.5}},
                    "新闻正面率": {"value": {"positive_rate": 0.75, "sample": 4}},
                    "主力资金": {"value": {"net_inflow_wan": 35000.0}},
                }}
        facts = facts_from_candidate(cand)
        assert facts["pe"] == 30.0
        assert facts["momentum20"] == pytest.approx(12.3)       # 小数 → %
        assert facts["ma20_above"] is True
        assert facts["ma_bullish"] is True
        assert facts["volatility"] == 22.5
        assert facts["news_positive"] == pytest.approx(75.0)
        assert facts["fund_net"] == pytest.approx(3.5)          # 万元 → 亿元
        assert facts["score"] == 82.5

    def test_missing_and_wrong_types_are_none(self):
        cand = {"股票代码": "000001", "factors": {
            "PE估值": {"status": "unknown"},
            "趋势": {"value": "不是字典"},
            "均线排列": {"value": {"bullish": "yes"}},
            "波动率": {"value": {"volatility": None}},
        }}
        facts = facts_from_candidate(cand)
        for key in ("pe", "momentum20", "ma20_above", "ma_bullish", "volatility",
                    "news_positive", "fund_net", "score"):
            assert facts[key] is None, key

    def test_no_factors_at_all(self):
        facts = facts_from_candidate({})
        assert facts["pe"] is None and facts["name"] == ""


class TestScreenerContract:
    """screener 必须在因子明细里给出原始值（自然语言本地核对的前提）"""

    def _df(self, n=30, start=10.0, step=0.2):
        import pandas as pd
        closes = [start + step * i for i in range(n)]
        return pd.DataFrame({"Close": closes})

    def test_pe_value(self):
        from core.screener import StockScreener
        s = StockScreener.__new__(StockScreener)          # 不走 __init__（避免建数据源）
        assert s._score_pe(12.5)["value"]["pe"] == 12.5
        assert "value" not in s._score_pe(None)

    def test_trend_value(self):
        from core.screener import StockScreener
        s = StockScreener.__new__(StockScreener)
        out = s._score_trend(self._df())
        assert out["value"]["ret20"] == pytest.approx(0.33898305, abs=1e-3)  # 存的是 4 位小数
        assert out["value"]["above_ma20"] is True
        assert "value" not in s._score_trend(None)

    def test_ma_and_volatility_value(self):
        from core.screener import StockScreener
        s = StockScreener.__new__(StockScreener)
        ma = s._score_ma_alignment(self._df())
        assert ma["value"]["bullish"] is True
        vol = s._score_volatility(self._df())
        assert vol["value"]["volatility"] > 0

    def test_facts_from_real_scorer_output(self):
        """端到端契约：scorer 输出 → facts → 条件核对"""
        from core.screener import StockScreener
        s = StockScreener.__new__(StockScreener)
        cand = {"股票代码": "600519", "股票名称": "贵州茅台", "score": 90.0,
                "factors": {"PE估值": s._score_pe(12.5),
                            "趋势": s._score_trend(self._df()),
                            "均线排列": s._score_ma_alignment(self._df()),
                            "波动率": s._score_volatility(self._df())}}
        q = parse("市盈率低于20，均线多头排列，波动率低于25%")
        facts = facts_from_candidate(cand)
        report = evaluate(facts, q)
        assert report["pass"] is True, report
        assert "均线多头排列" in "".join(report["hits"])


class TestUncoveredFields:
    def test_every_metric_has_label_alias_and_source(self):
        """每个指标的元数据必须齐备，且 `source` 只取注册表声明的 5 个值之一。

        （原先断言的是 `local` 布尔标记与 `wencai` 口径词；两者都已随
        本地化改造被 `source` 取代。）
        """
        from core.indicator_registry import SOURCE_LABELS
        for key, spec in METRICS.items():
            assert spec["label"] and spec["aliases"], key
            assert spec["kind"] in ("num", "bool"), key
            assert spec["source"] in SOURCE_LABELS, key
            assert set(spec) >= {"label", "unit", "kind", "aliases", "source"}

    def test_aliases_are_unique_across_metrics(self):
        seen = {}
        for key, spec in METRICS.items():
            for alias in spec["aliases"]:
                assert alias.lower() not in seen, f"{alias} 同时属于 {seen.get(alias)} 和 {key}"
                seen[alias.lower()] = key

    def test_condition_scaled_text(self):
        q = parse("市盈率低于20")
        assert cond(q, "pe").scaled() == "市盈率(PE) 小于 20倍"
        assert parse("均线多头排列").conditions[0].scaled() == "均线多头排列"
