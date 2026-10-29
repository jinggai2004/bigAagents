# -*- coding: utf-8 -*-
"""「数值关系」守护：价格↔指标 / 指标↔指标 / 成交量↔均量（v2，2026-10-13）

## 起源报障

600028 中国石化深度分析报告（`output/ui_runs/analysis_600028.json`）技术面段写：

    「当前收盘价5.46元高于MA5（5.408）、MA10（5.315）和MA20（5.295）……
      然而，收盘价仍低于MA60（5.1615）」

而 **5.46 > 5.1615**。用项目自身取数链复算：price=5.46、ma60=5.1615，收盘价**高于** MA60。

## 为什么既有三道审计全都拦不住

它们检查的是"**每个数字有没有出处**"，没有一个检查"**数字之间的关系是否成立**"：

| 审计 | 本次为何放行 |
|---|---|
| `audit_numbers`（数值可追溯） | 5.46 与 5.1615 **都逐字真在输入里** → 判"可追溯" |
| `audit_citations`（伪造引用） | 报告没声称"输入中列出…" → 不触发 |
| `audit_authorities`（来源名/年份/文档名） | 三者都没有 → 一条不命中 |

⇒ **数字全对、只错关系**，是这套审计的结构性盲区。
危害：该错误被多头引用 2 次、空头引用 9 次，空头还以它作为核心论据。

## 三层修复（一防一查 + 执行层铺开）

1. **输入侧（防）** `ma_position_block()`：把"收盘价相对各均线的高低"**预先算好**写进提示词。
   根因在此 —— 原提示词给的是平铺裸数字，模型要自己比 4 个均线，第 4 个比错了。
2. **审计侧（查）** `audit_relations()`：用真值重算方向，矛盾即标记。覆盖三个族。
3. **执行层** `execution_indicator_block()`：交易员/风控也拿到完整均线 + ATR14 + 关系预判，
   同时**消除字面量 `N/A`**（见 `TestExecutionIndicatorBlock`）。

## 本文件的四条纪律

- **真报与误报都要验**：条件句 / 历史句 / 否定句 / 无真值 / 不收的族**一条都不许报**。
- **必须验"接线"**：算出来但没进提示词、实现了但没被调用，是本项目反复出现的失效形态。
- **必须验"缺就是缺"**：真值缺失时不判，而不是拿 0 或 N/A 硬算。
- **上限必须验**：关系告警排在最前，不限量会把数值/年份类整批挤出配额。
"""

import pandas as pd
import pytest

from core import agents
from core.agents import _audit, _audit_downstream, risk_manager_node, trader_node
from core.report_audit import _REL_MAX_WARN, audit_relations, audit_risk_limits
from llm.deepseek_client import (
    DeepSeekClient, execution_indicator_block, ma_position_block,
)


# ===== 基准：600028 的**真实**值（2026-10-13 由项目自身取数链复算）=====
IND = {
    "price": 5.46,
    "ma5": 5.4080, "ma10": 5.3150, "ma20": 5.2950, "ma60": 5.1615,
    "bb_middle": 5.2950,
    "k_value": 88.6792, "d_value": 82.3899,
    "macd": 0.0568, "macd_signal": 0.0374, "macd_histogram": 0.0194,
    "rsi": 59.5681, "atr14": 0.1341,
    "volume": 228443300.0, "volume_ma5": 247532640.0, "volume_ma20": 223270305.0,
}
# 报障原句（技术面报告原文，逐字保留）
REAL_BAD_SENTENCE = (
    "从均线系统来看，当前收盘价5.46元高于MA5（5.408）、MA10（5.315）和MA20（5.295），"
    "表明短期和中期趋势较为强劲。然而，收盘价仍低于MA60（5.1615），"
    "这可能意味着长期趋势尚未完全转强。"
)
# 审计用的"输入可搜索文本"：让 `audit_report` 保持安静，
# 从而**隔离**验证关系检查本身（否则数值可追溯类告警会混进来）
ALLOWED = "收盘价 5.46　MA5 5.408　MA10 5.315　MA20 5.295　MA60 5.1615"


def _state(**over):
    st = {
        "stock_code": "600028", "stock_name": "中国石化",
        "stock_info": {"symbol": "600028", "name": "中国石化", "current_price": 5.46},
        "indicators": dict(IND),
        "tech_report": "t", "fundamental_report": "f", "news_report": "n",
        "bull_history": [], "bear_history": [],
        "research_summary": "（研究汇总占位）",
        "trade_plan": {"direction": "观望", "position_size": "0%"},
        "analysis_date": "2026-10-13", "data_through": "2026-10-13",
        "errors": [], "fatal_errors": [],
    }
    st.update(over)
    return st


def _make_client():
    """不会联网的 DeepSeekClient（跳过 __init__ 的 API Key 校验）"""
    client = DeepSeekClient.__new__(DeepSeekClient)
    client.model = "test"
    box = {}

    def fake_call(messages, **kw):
        box["messages"] = messages
        box["kwargs"] = kw
        return "报告正文"

    client.call_api = fake_call
    return client, box


def _kline(n=25):
    idx = pd.date_range("2026-08-01", periods=n, freq="B")
    return pd.DataFrame({"Close": [5 + i * 0.01 for i in range(n)],
                         "Volume": [1000 * (i + 1) for i in range(n)]}, index=idx)


# ============================================================
# 一、输入侧预判（根治层）
# ============================================================

class TestMaPositionBlock:
    def test_lists_relation_for_every_ma(self):
        out = ma_position_block(IND)
        assert "均线相对位置" in out
        # MA60 是报障处：真值是"高于"，必须明写
        assert "收盘价**高于**MA60（+5.47%）" in out
        assert "收盘价**高于**MA5（+0.95%）" in out

    def test_marks_bull_alignment(self):
        assert "多头排列" in ma_position_block(IND)

    def test_marks_bear_alignment(self):
        ind = {"price": 10.0, "ma5": 11.0, "ma10": 12.0, "ma20": 13.0, "ma60": 14.0}
        out = ma_position_block(ind)
        assert "空头排列" in out
        assert "收盘价**低于**MA60" in out

    def test_marks_tangled_when_no_consistent_order(self):
        ind = {"price": 10.0, "ma5": 11.0, "ma10": 10.5, "ma20": 12.0, "ma60": 9.0}
        assert "相互纠缠" in ma_position_block(ind)

    def test_empty_without_price(self):
        """缺收盘价 → "高低"无从谈起，返回空串（**不许**返回半段空壳）"""
        assert ma_position_block({"ma60": 5.16}) == ""
        assert ma_position_block({}) == ""
        assert ma_position_block(None) == ""

    def test_empty_when_no_ma_present(self):
        assert ma_position_block({"price": 5.46}) == ""

    def test_no_alignment_line_when_some_ma_missing(self):
        """缺任一均线就不给排列结论 —— 排列必须四线齐全才成立"""
        out = ma_position_block({"price": 5.46, "ma5": 5.40, "ma60": 5.16})
        assert "均线排列" not in out
        assert "MA60" in out

    def test_skips_nan_and_none(self):
        ind = {"price": 5.46, "ma5": float("nan"), "ma60": None, "ma20": 5.295}
        out = ma_position_block(ind)
        assert "MA5" not in out and "MA60" not in out
        assert "收盘价**高于**MA20" in out


# ============================================================
# 二、接线：预判段必须**真的进提示词**
# ============================================================

class TestPromptCarriesMaPosition:
    """⚠️ 防的是"算出来却没导出"——本项目已复发 3 次的失效形态"""

    def _prompt(self, indicators):
        client, box = _make_client()
        client.technical_analysis({"symbol": "600028", "name": "中国石化",
                                   "current_price": 5.46},
                                  _kline(), indicators,
                                  analysis_date="2026-10-13",
                                  data_through="2026-10-13")
        return box["messages"][1]["content"]

    def test_technical_prompt_contains_relation_block(self):
        user = self._prompt(IND)
        assert "均线相对位置" in user
        assert "收盘价**高于**MA60" in user

    def test_prompt_tells_model_not_to_recompare(self):
        user = self._prompt(IND)
        assert "不要自己再比较" in user

    def test_block_absent_when_price_missing(self):
        user = self._prompt({"ma20": 5.295, "rsi": 59.57})
        assert "均线相对位置" not in user
        assert "N/A" not in user


# ============================================================
# 三、族1：价格 ↔ 指标（含反向主语）
# ============================================================

class TestFamilyPriceVsMa:
    def test_catches_the_real_600028_case(self):
        out = audit_relations(REAL_BAD_SENTENCE, IND)
        assert len(out) == 1
        assert "方向写反" in out[0]
        assert "MA60" in out[0] and "高于" in out[0]
        assert "5.46" in out[0] and "5.1615" in out[0]

    def test_correct_statement_is_not_reported(self):
        assert audit_relations("收盘价5.46元高于MA60（5.1615）", IND) == []
        assert audit_relations("当前收盘价5.46元高于MA5（5.408）、MA10（5.315）", IND) == []

    def test_correctly_reports_when_truth_is_below(self):
        ind = {"price": 5.0, "ma60": 5.5}
        out = audit_relations("收盘价5.0元高于MA60（5.5）", ind)
        assert out and "低于" in out[0]

    # ---- 反向主语：指标在左（实测 4×，v6 漏掉的纯遗漏）----

    def test_reversed_subject_correct_is_not_reported(self):
        """「MA60（39.406）…远低于现价」= 价格高于 MA60 → 与真值一致"""
        assert audit_relations("MA60（39.406）持续上行且远低于现价", IND) == []

    def test_reversed_subject_wrong_is_reported(self):
        out = audit_relations("MA60（5.90）远高于现价", IND)
        assert out and "高于" in out[0]

    def test_reversed_subject_above_form(self):
        """「MA20在现价下方」= 价格高于 MA20 → 正确"""
        assert audit_relations("MA20在现价下方", IND) == []

    # ---- 误报防护 ----

    @pytest.mark.parametrize("text", [
        "若收盘价低于MA60则视为弱势",
        "如果股价低于MA60，应减仓",
        "一旦跌破MA60需止损",
        "历史 上股价低于MA60时往往反弹",
        "此前股价曾低于MA60，现已修复",
        "收盘价不会低于MA60",
        "收盘价未低于MA60",
    ])
    def test_legal_statements_are_not_reported(self, text):
        assert audit_relations(text, IND) == [], f"误报: {text}"

    def test_generic_ma_word_is_not_reported(self):
        assert audit_relations("股价在短期均线之上，中期趋势稳健", IND) == []

    def test_correct_above_form_is_not_reported(self):
        assert audit_relations("股价在MA10和MA20之上", IND) == []
        assert audit_relations("当前收盘价5.46元位于布林带中轨（5.295）之上", IND) == []

    def test_bollinger_midline_is_checked(self):
        out = audit_relations("股价在中轨 之下运行", IND)
        assert out and "布林带中轨" in out[0]

    def test_bb_upper_lower_deliberately_out_of_scope(self):
        """⚠️ 锁定"上轨/下轨不检查"这个**有意**的取舍（实测多为"接近上轨"这类模糊关系）"""
        assert audit_relations("股价在布林带上轨 5.588 之上", IND) == []
        assert audit_relations("股价在布林带下轨 5.002 之下", IND) == []


# ============================================================
# 四、族2：指标 ↔ 指标（MA 之间 / K-D / MACD-信号线 / MACD-零轴）
# ============================================================

class TestFamilyIndicatorVsIndicator:
    def test_macd_correct_not_reported(self):
        assert audit_relations(
            "MACD线（0.0568）位于信号线（0.0374）之上，且MACD柱（0.0194）为正值", IND) == []

    def test_macd_wrong_is_reported(self):
        ind = {**IND, "macd": 0.01, "macd_signal": 0.05}
        out = audit_relations("MACD线位于信号线之上", ind)
        assert out and "MACD信号线" in out[0]

    def test_kd_correct_not_reported(self):
        ind = {**IND, "k_value": 85.2, "d_value": 78.1}
        assert audit_relations("K值85.2高于D值78.1，显示多方力量占优", ind) == []

    def test_kd_wrong_is_reported(self):
        """真值 K(88.68) > D(82.39)，报告写"低于" → 报"""
        out = audit_relations("K值低于D值，KDJ呈死叉迹象", IND)
        assert out and "K值" in out[0]

    def test_ma_order_symbol_form_correct(self):
        assert audit_relations("MA5 > MA10 > MA20 > MA60，呈多头排列", IND) == []

    def test_ma_order_symbol_form_wrong(self):
        """真值 MA60(5.1615) < MA20(5.295)，写成"MA60 > MA20" → 报"""
        out = audit_relations("MA60 > MA20，长期强于中期", IND)
        assert out and "MA60" in out[0]

    def test_macd_vs_zero_correct(self):
        ind = {**IND, "macd": -0.02}
        assert audit_relations("MACD值位于零轴下方，中期动量尚未翻多", ind) == []

    def test_macd_vs_zero_wrong(self):
        """真值 MACD=+0.0568，写成"位于零轴下方" → 报"""
        out = audit_relations("MACD值位于零轴下方", IND)
        assert out and "零轴" in out[0]

    # ---- ⭐ 误配守卫（`_gap_clean`）：来自实测暴露的真实误配机制 ----

    def test_mispair_guard_bare_digit_across_clause(self):
        """「三者均大于0且MACD略高于信号线」中的"大于"主语是**三者**，不是 MACD柱。
        跨过裸数字去配 "MACD柱 大于 MACD" 是**误配** → 必须不报。"""
        assert audit_relations("MACD柱0.0009，三者均大于0且MACD略高于信号线", IND) == []

    def test_mispair_guard_bare_digit_second_form(self):
        assert audit_relations("MACD值0.0906，信号线0.0897，MACD略高于信号线", IND) == []

    def test_parenthesized_values_still_checked(self):
        """⚠️ 反向验证：守卫**不能**把合法的括号写法一起挡掉（否则等于关掉整个族2）"""
        ind = {**IND, "macd": 0.01, "macd_signal": 0.05}
        assert audit_relations("MACD线（0.01）位于信号线（0.05）之上", ind), \
            "括号内数值的合法写法必须仍受检查"

    def test_qualitative_terms_are_not_checked(self):
        """定性词（金叉/死叉/多头排列）不做推断 —— 有意不收，避免误报"""
        assert audit_relations("MACD弱金叉，动量温和转多", IND) == []


# ============================================================
# 五、族3：成交量 ↔ 均量
# ============================================================

class TestFamilyVolumeVsAverage:
    def test_correct_not_reported(self):
        assert audit_relations(
            "当前成交量为228,443,300股，略低于5日均量（247,532,640股），"
            "但高于20日均量（223,270,305股）", IND) == []

    def test_wrong_is_reported(self):
        out = audit_relations("成交量高于5日均量，属于放量", IND)
        assert out and "5日均量" in out[0]

    def test_bare_average_word_not_checked(self):
        """裸称"均量"无法确定窗口（5日还是20日）→ 不判"""
        assert audit_relations("成交量高于均量", IND) == []

    def test_conditional_not_reported(self):
        assert audit_relations("若成交量低于5日均量则应观望", IND) == []


# ============================================================
# 六、缺真值 / 上限 / 容差
# ============================================================

class TestAuditRelationsBoundaries:
    def test_missing_truth_is_not_judged(self):
        """⚠️ 缺就是缺：真值缺失时不判，**不许**用 0 或 N/A 硬算"""
        assert audit_relations("K值高于D值", {"price": 5.46}) == []
        assert audit_relations("收盘价低于MA60", {"price": 5.46}) == []
        assert audit_relations("成交量高于5日均量", {"price": 5.46}) == []

    def test_nan_treated_as_missing(self):
        assert audit_relations("收盘价低于MA60",
                               {"price": 5.46, "ma60": float("nan")}) == []

    def test_no_price_returns_empty(self):
        assert audit_relations(REAL_BAD_SENTENCE, {"ma60": 5.16}) == []
        assert audit_relations(REAL_BAD_SENTENCE, {}) == []
        assert audit_relations(REAL_BAD_SENTENCE, None) == []

    def test_empty_report(self):
        assert audit_relations("", IND) == []
        assert audit_relations(None, IND) == []

    def test_near_equal_is_not_reported(self):
        """与均线几乎相等（≤0.1%）时"高于/低于"本身无意义 → 不报"""
        assert audit_relations("收盘价低于MA60", {"price": 5.16, "ma60": 5.1605}) == []

    def test_dedupes_repeated_claim(self):
        """同一个错关系在辩论里被引用十几次 → 只报一条（噪声控制）"""
        rep = "收盘价仍低于MA60，长期趋势尚未转强。" * 6
        assert len(audit_relations(rep, IND)) == 1

    def test_default_cap_is_small_and_independent(self):
        """⚠️ 关系告警排在最前，上限必须独立且小 —— 否则会挤满 MAX_WARNINGS=6"""
        assert _REL_MAX_WARN <= 2
        rep = "收盘价低于MA5。收盘价低于MA10。收盘价低于MA20。收盘价低于MA60。"
        assert len(audit_relations(rep, IND)) <= _REL_MAX_WARN

    def test_max_warn_override(self):
        rep = "收盘价低于MA5。收盘价低于MA10。收盘价低于MA20。"
        assert len(audit_relations(rep, IND, max_warn=3)) == 3


# ============================================================
# 七、接线：关系告警必须真的并入 errors
# ============================================================

class TestAuditWiring:
    def test_audit_prepends_relation_warning(self):
        out = _audit("技术面", REAL_BAD_SENTENCE, ALLOWED, indicators=IND)
        assert out, "关系矛盾必须被报出来"
        assert out[0].startswith("[技术面] 关系矛盾")
        assert "MA60" in out[0]

    def test_relation_warning_survives_full_warning_quota(self):
        """⚠️ 关系矛盾是硬矛盾，必须排在数值/年份类之前，不能被配额挤掉"""
        noisy = REAL_BAD_SENTENCE + "\n另有 999.11、888.22、777.33、666.44、555.55 五个无出处数值。"
        out = _audit("技术面", noisy, ALLOWED, indicators=IND)
        assert any("关系矛盾" in w for w in out)

    def test_audit_without_indicators_reports_no_relation(self):
        """向后兼容：不传 indicators 时行为与改造前一致（**不新增告警**）"""
        out = _audit("技术面", REAL_BAD_SENTENCE, ALLOWED)
        assert not any("关系矛盾" in w for w in out)

    def test_downstream_audit_reads_indicators_from_state(self):
        """辩论/风控/终审节点走 `_audit_downstream` → 必须从 state 取到真值"""
        out = _audit_downstream("空头R2", REAL_BAD_SENTENCE, _state())
        assert any("关系矛盾" in w for w in out), "下游节点必须同样受保护"

    def test_downstream_audit_tolerates_missing_indicators(self):
        """state 里没有 indicators（旧数据/异常路径）不许抛异常拖垮分析链"""
        out = _audit_downstream("空头R2", REAL_BAD_SENTENCE, _state(indicators={}))
        assert not any("关系矛盾" in w for w in out)


# ============================================================
# 八、执行层（交易员 / 风控）：指标注入 + N/A 消除
# ============================================================
#
# 两个实测缺口（2026-10-13）：
#   ① **风控审"止损是否恰当"却拿不到 ATR** —— 风控 prompt 原先只注入
#      `股票/现价/trade_plan/风控规则块`，而 ATR 进入视野的唯一通道是 `trade_plan`，
#      但 `trade_plan = json.loads(模型输出)`、代码从不注入 atr
#      ⇒ 实测 10 份产物**只有 2 份**带 atr（模型自己多写的）。
#      "时有时无"比"完全没给"更难发现。
#   ② **交易员只有 MA20**，且两个节点都用**字面量 `N/A`** 兜底
#      （该反模式在技术面/基本面早已禁掉，这两处属漏改）。

class TestExecutionIndicatorBlock:
    def test_contains_all_key_values(self):
        out = execution_indicator_block(IND, role="risk")
        for token in ("MA5", "MA10", "MA20", "MA60", "ATR14(元)", "RSI", "现价"):
            assert token in out, f"缺 {token}"
        assert "均线相对位置" in out
        assert "收盘价**高于**MA60" in out

    def test_no_na_literal_when_missing(self):
        """⚠️ 缺失必须走"未提供的字段"声明，**不许**渲染成 `N/A`"""
        out = execution_indicator_block({"ma20": 5.295}, role="trader")
        assert "N/A" not in out
        assert "【本次未提供的字段】" in out
        assert "ATR14" in out          # 缺失字段也要被列出，让模型知道"是缺不是没有"

    def test_role_scope_note_differs(self):
        """两种角色的用途限定句必须不同（防风控越界 / 防交易员被引导）"""
        trader = execution_indicator_block(IND, role="trader")
        risk = execution_indicator_block(IND, role="risk")
        assert "不要**据此推翻上游研究汇总" in trader
        assert "不要**据此重做技术分析" in risk

    def test_empty_indicators(self):
        out = execution_indicator_block({}, role="risk")
        assert "（本次未取到任何技术指标）" in out
        assert "N/A" not in out


class TestTraderAndRiskPrompts:
    def _render(self, node, **over):
        client, box = _make_client()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(DeepSeekClient, "call_api", lambda self, messages, **kw: (
                box.update(messages=messages) or "报告正文"))
            node(_state(**over))
        return box["messages"][1]["content"]

    @pytest.mark.parametrize("node", [trader_node, risk_manager_node])
    def test_prompt_has_full_indicators_and_atr(self, node):
        user = self._render(node)
        assert "ATR14(元)：0.1341" in user, "ATR 必须由代码注入"
        assert "MA60：5.1615" in user, "均线必须给全（原先交易员只有 MA20）"
        assert "均线相对位置" in user
        assert "收盘价**高于**MA60" in user

    @pytest.mark.parametrize("node", [trader_node, risk_manager_node])
    def test_prompt_has_no_na_literal(self, node):
        assert "N/A" not in self._render(node)

    @pytest.mark.parametrize("node", [trader_node, risk_manager_node])
    def test_prompt_degrades_honestly_without_indicators(self, node):
        user = self._render(node, indicators={}, stock_info={"name": "测试"})
        assert "N/A" not in user
        assert "【本次未提供的字段】" in user and "ATR14" in user

    def test_risk_prompt_requires_atr_against_stop_loss(self):
        """⚠️ 注入还不够 —— 必须**明确要求**风控拿 ATR 去审止损，否则它可能不用"""
        user = self._render(risk_manager_node)
        assert "必须对照上面的 ATR14" in user
        assert "无波动率标尺" in user


# ============================================================
# 九、风控阈值 / 「仓位 vs 上限」（v8）
# ============================================================
#
# ⚠️ 这一族的**真值来源与前面几族不同**：不在 `indicators` 里，
# 而在 `RiskEngine`（阈值）与 `trade_plan`（计划仓位）。
# 对应项目最严重的两次风控事故：
#   · 2026-10-08 风控编出「单票上限 8%、关注类 5%」，把本在限额内的 15% 判为
#     "超限 200%、触发熔断、驳回"——真实上限是 25%；
#   · 同日终审又自造「合规上限 3.3%」把方案驳回。
# `audit_numbers` 抓不到（8 / 3.3 都能由输入数字凑出来）⇒ 必须按语义位置检。

RISK_LIMITS = {"single": 0.25, "sector": 0.30, "drawdown": 0.15, "per_trade": 0.02}


class TestAuditRiskLimits:
    """用例全部取**真实报告原句**（逐字），真值取自 `RiskEngine`。"""

    @pytest.mark.parametrize("text,pos", [
        ("- **单票最大仓位上限**：25%", "0%"),
        ("- **结论**：计划仓位为0%，未超过单票最大仓位上限25%", "0%"),
        ("→ 计划 0% vs 上限 25%", "0%"),
        ("✅ **计划 15% vs 上限 25%** → **未超限**，符合规则", "15%"),
        ("> 注：行业仓位上限为30%，但本计划仅涉及单只股票", "15%"),
        ("- 单票最大仓位上限：`25%`（规则第一条）", "15%"),
        ("**最大回撤预估值为0%**。", "15%"),          # ⚠️ 「回撤预估值」不是「熔断线」
        ("组合最大回撤熔断线：15%", "15%"),
        ("每笔交易最大风险：2%（口径：仓位 × 止损幅度）", "15%"),
        ("计划采用分批建仓，首笔不超过 15%，单票上限 25%", "5%"),  # ⚠️ 首笔比例≠计划仓位
    ])
    def test_real_phrases_are_not_reported(self, text, pos):
        assert audit_risk_limits(text, position_size=pos, limits=RISK_LIMITS) == [], \
            f"误报: {text}"

    def test_fabricated_limit_is_reported(self):
        """⚠️ 最严重的一类：编造合规阈值。项目原话：编造合规依据比不给依据更危险。"""
        out = audit_risk_limits("- 单票上限 8%，据此判定超限 200%",
                                position_size="15%", limits=RISK_LIMITS)
        assert out and "25%" in out[0], out

    def test_limit_not_in_rules_is_reported(self):
        out = audit_risk_limits("计划 15% vs 上限 8% → 超限，驳回",
                                position_size="15%", limits=RISK_LIMITS)
        assert any("不在项目风控规则内" in w for w in out), out

    def test_contradictory_conclusion_is_reported(self):
        """写"未超过"但数值上超过了 → 结论与数值矛盾"""
        out = audit_risk_limits("计划仓位为30%，未超过单票最大仓位上限25%",
                                position_size="30%", limits=RISK_LIMITS)
        assert any("结论与数值矛盾" in w for w in out), out

    def test_misquoted_plan_position_is_reported(self):
        """报告把计划仓位引错（计划是 0%，报告写 15%）"""
        out = audit_risk_limits("计划仓位为15%，未超过单票最大仓位上限25%",
                                position_size="0%", limits=RISK_LIMITS)
        assert any("计划仓位写成" in w for w in out), out

    def test_limits_accepts_percent_form(self):
        """调用方若直接传百分数（25 而非 0.25）也要能用"""
        out = audit_risk_limits("单票上限 8%", position_size="15%",
                                limits={"single": 25})
        assert out, "百分数形式应被容忍"

    def test_boundaries(self):
        assert audit_risk_limits("", position_size="15%", limits=RISK_LIMITS) == []
        assert audit_risk_limits("单票上限 8%", position_size="15%", limits=None) == []
        assert audit_risk_limits("单票上限 8%", position_size="15%", limits={}) == []


class TestRiskLimitsWiring:
    def test_thresholds_come_from_risk_engine(self):
        """⚠️ 阈值必须**只认** `RiskEngine` 一处（与 risk_rules_block 同源），
        审计里不得重抄数字 —— 否则审计与提示词会各自漂移。"""
        from core.risk import RiskEngine
        from core.agents import _risk_limits_truth
        limits, pos = _risk_limits_truth(_state(trade_plan={"position_size": "15%"}))
        assert limits["single"] == RiskEngine.MAX_SINGLE_STOCK_WEIGHT
        assert limits["sector"] == RiskEngine.MAX_SECTOR_WEIGHT
        assert limits["drawdown"] == RiskEngine.MAX_DRAWDOWN_LIMIT
        assert limits["per_trade"] == RiskEngine.DEFAULT_RISK_PER_TRADE
        assert pos == "15%"

    def test_downstream_audit_reports_fabricated_limit(self):
        """风控/终审走 `_audit_downstream` → 必须能从 state 拿到阈值与仓位并报出"""
        st = _state(trade_plan={"direction": "看多", "position_size": "15%"})
        out = _audit_downstream("风控审核",
                                "计划 15% 超过单票最大仓位上限 8%，驳回", st)
        assert any("风控阈值" in w for w in out), out

    def test_downstream_audit_quiet_on_correct_limit(self):
        st = _state(trade_plan={"direction": "看多", "position_size": "15%"})
        out = _audit_downstream(
            "风控审核", "计划 15% vs 上限 25% → 未超限，符合规则", st)
        assert not any("风控阈值" in w for w in out), out

    def test_risk_limits_failure_does_not_break_audit(self):
        """RiskEngine 读不到时不许抛异常拖垮分析链"""
        from core.agents import _risk_limits_truth
        limits, pos = _risk_limits_truth({})
        assert pos is None            # 空 state：仓位取不到，但不抛
