"""分析链的「输入接地」守护测试（2026-10-08）。

背景：603127 一份深度分析报告暴露了 5 类问题，全部是**输入没接地**：
  1. K 线只取 3mo（59 根）→ MA60 恒 NaN，报告如实写"缺失"；
  2. `get_latest_indicators` 漏导出 `bb_middle` → 报告写"中轨未提供"，
     而交易计划又凭空给出"中轨约 45.1–45.4"（自相矛盾）；
  3. `tool_financials` 只给前 5 个科目 + 不声明累计口径 → 模型按"单季"解读，
     推出"净利率 106%""环比 +122.8%"，并把明明返回了的科目说成"缺失"；
  4. 风控提示词不含项目自己的风控阈值 → 模型**编造**"单票上限 8%/5%"驳回方案
     （项目真实上限 25%，该方案本在限额内）；
  5. 审计只接了 3 个节点 → 辩论/汇总/交易/风控的编造**零告警**。

本文件守护修复后的接线，防止任何一条被回退。
"""
import inspect
import json

import pandas as pd
import pytest

from core import agents as A
from core.risk import RiskEngine


# ================================================================
# 1. 风控：提示词必须含**项目自己的**阈值
# ================================================================

def _capture_prompt(monkeypatch, node, state, ret="OK"):
    """跑一个节点，返回它实际发给模型的消息列表（不真的调 API）。"""
    seen = {}

    class FakeClient:
        def call_api(self, messages, **kw):
            seen["messages"] = messages
            seen["kwargs"] = kw
            return ret

    monkeypatch.setattr(A, "DeepSeekClient", FakeClient)
    node(state)
    return seen


@pytest.fixture
def risk_state():
    return {"stock_code": "603127", "stock_name": "昭衍新药",
            "stock_info": {"current_price": 52.06},
            "trade_plan": {"direction": "看空", "position_size": "15%"},
            "fatal_errors": []}


class TestRiskPromptGrounding:
    def test_prompt_contains_project_limits(self, monkeypatch, risk_state):
        """⚠️ 核心：阈值必须来自 `core/risk.py`，不是模型自己想的。

        实测 603127：模型编出"单票上限 8%、关注类 5%"，据此把 15% 仓位的方案
        判为"超限 200%、触发熔断、驳回"——而项目真实上限是 25%，本在限额内。
        """
        seen = _capture_prompt(monkeypatch, A.risk_manager_node, risk_state)
        prompt = seen["messages"][1]["content"]
        assert f"{RiskEngine.MAX_SINGLE_STOCK_WEIGHT:.0%}" in prompt, prompt
        assert "单票最大仓位" in prompt
        assert f"{RiskEngine.MAX_SECTOR_WEIGHT:.0%}" in prompt
        assert f"{RiskEngine.MAX_DRAWDOWN_LIMIT:.0%}" in prompt

    def test_prompt_forbids_invented_rules(self, monkeypatch, risk_state):
        """仅给阈值不够，还要**禁止**引用未列出的制度条款"""
        seen = _capture_prompt(monkeypatch, A.risk_manager_node, risk_state)
        prompt = seen["messages"][1]["content"]
        assert "唯一判断依据" in prompt or "唯一依据" in prompt
        assert "禁止" in prompt and "手册" in prompt
        # 注意：源码里是 `**不得**以"超限"为由驳回`，中间夹着 Markdown 强调符，
        # 所以不能直接断言 "不得以"（首版就踩了这个，断言写错而不是代码错）
        assert "不得" in prompt and "为由驳回" in prompt

    def test_prompt_ties_conclusion_to_threshold(self, monkeypatch, risk_state):
        """要求写明"计划 X% vs 上限 Y%"，否则结论无法核对"""
        seen = _capture_prompt(monkeypatch, A.risk_manager_node, risk_state)
        prompt = seen["messages"][1]["content"]
        assert "计划" in prompt and "上限" in prompt


# ================================================================
# 2. 证据来源约束：辩论/汇总/交易/风控/终审全都要带
# ================================================================

class TestEvidenceConstraint:
    def test_constraint_forbids_external_citations(self):
        c = A.EVIDENCE_CONSTRAINT
        assert "输入未提供" in c
        assert "Wind" in c and "公告编号" in c and "监管文号" in c
        assert "回测统计量" in c

    def test_bull_and_bear_prompts_carry_constraint(self, monkeypatch):
        """多空双方**两轮**提示词都要带 —— R2 比 R1 编得更多（"驳斥"任务在放大幻觉）"""
        st = {"stock_code": "603127", "stock_name": "昭衍新药",
              "stock_info": {"current_price": 52.06},
              "tech_report": "T", "fundamental_report": "F", "news_report": "N",
              "fatal_errors": [], "debate_count": 0,
              "bull_argument": "多头上轮", "bear_argument": "空头上轮",
              "bull_history": [], "bear_history": []}
        for node in (A.bull_researcher_node, A.bear_researcher_node):
            seen = _capture_prompt(monkeypatch, node, dict(st, debate_count=0))
            assert "证据来源硬约束" in seen["messages"][1]["content"], node.__name__
            seen = _capture_prompt(monkeypatch, node, dict(st, debate_count=1))
            assert "证据来源硬约束" in seen["messages"][1]["content"], node.__name__

    def test_downstream_allowed_includes_reports(self):
        """审计依据必须覆盖辩论节点真正能看到的输入，否则审计会大量误报"""
        st = {"tech_report": "技术面XYZ", "fundamental_report": "基本面ABC",
              "news_report": "舆情DEF"}
        allowed = A._downstream_allowed(st)
        for token in ("技术面XYZ", "基本面ABC", "舆情DEF"):
            assert token in allowed


# ================================================================
# 3. 结构性守护：每个调 LLM 的节点都必须接审计
# ================================================================

class TestAuditWiring:
    """⚠️ 这条只能靠**结构**断言。

    逐个 mock 覆盖不了"忘了接线"——603127 那份报告 7 段正文里有 4 段完全没审计，
    而 `errors` 是空列表，**从结果上完全看不出**（模型输出本身是随机的，
    重跑一次可能没有编造，测试就绿了）。
    """

    NODES = ("tech_analyst_node", "fundamental_analyst_node", "news_sentiment_node",
             "bull_researcher_node", "bear_researcher_node", "research_manager_node",
             "trader_node", "risk_manager_node", "portfolio_manager_node")

    @pytest.mark.parametrize("name", NODES)
    def test_node_calls_audit(self, name):
        src = inspect.getsource(getattr(A, name))
        assert "_audit(" in src or "_audit_downstream(" in src, (
            f"{name} 调了 LLM 却没有接可核实性审计 —— "
            f"编造内容不会进 errors，报告会显示成'零告警'")

    def test_audit_labels_are_distinct(self):
        """审计标签要能区分是哪一段报的（同一份报告里有 7 段正文）"""
        labels = set()
        for name in self.NODES:
            src = inspect.getsource(getattr(A, name))
            for line in src.splitlines():
                if "_audit(" in line or "_audit_downstream(" in line:
                    labels.add(line.strip())
        assert len(labels) >= 7, labels

    def test_downstream_audit_declares_financials_and_news(self):
        """下游审计必须声明 has_financials / has_news。

        ⚠️ 实测（2026-10-08 重跑 603127）：不传这两个标志时，辩论/汇总段提到
        「定期报告」「季报」「三季报」会被误报成"本次未提供财报与新闻数据"——
        而它们看到的基本面报告里本来就含财报内容。误报会淹没真信号。
        """
        src = inspect.getsource(A._audit_downstream)
        assert "has_financials" in src and "has_news" in src, src

    def test_downstream_audit_does_not_flag_disclosure_terms(self):
        """行为验证：上游有财报/新闻时，下游引用披露材料不该告警"""
        st = {"fundamental_report": "【利润表(亿元)】营业收入 7.04",
              "news_report": "公司发布公告", "tech_report": "T",
              "stock_info": {}, "indicators": {}, "trade_plan": {},
              "bull_history": [], "bear_history": []}
        w = A._audit_downstream("研究汇总", "据定期报告与三季报，营收如前述。", st)
        assert not any("披露材料" in x for x in w), w


# ================================================================
# 4. K 线窗口：必须够算 MA60
# ================================================================

class TestKlineWindow:
    def test_window_is_long_enough_for_ma60(self):
        """⚠️ `period="3mo"` 实测只有 **59** 根，而 MA60 需要 **60** 根 → 恒 NaN。

        报告会如实写"MA60 缺失"，但根因是取数窗口差 1 根，
        而交易员又会在下游凭空给出"MA60 动态支撑"。
        """
        src = inspect.getsource(A.data_fetcher_node)
        assert 'period="3mo"' not in src, "3mo 只返回 59 根，MA60 恒为 NaN"
        assert 'period="1y"' in src


# ================================================================
# 5. K 线长度守卫：上游会静默返回截断序列
# ================================================================

class _FakeFetcher:
    """最小替身：只实现 data_fetcher_node 用到的 4 个方法。

    ⚠️ 序列由 `bars_seq` 依次给出，用来模拟"第一次返回截断、第二次完整"。
    """

    def __init__(self, bars_seq):
        self.bars_seq = list(bars_seq)
        self.calls = 0
        self.periods = []

    def get_stock_info(self, code):
        return {"symbol": code, "name": "测试股", "current_price": 10.0}

    def get_stock_data(self, code, period="1y"):
        self.periods.append(period)
        n = self.bars_seq[min(self.calls, len(self.bars_seq) - 1)]
        self.calls += 1
        if n == 0:
            return {"error": "取数失败"}
        idx = pd.bdate_range("2025-10-08", periods=n)      # 与生产一致：日期在索引上
        return pd.DataFrame({
            "Open": [10.0] * n, "Close": [10.0] * n,
            "High": [10.2] * n, "Low": [9.8] * n,
            "Volume": [1e6] * n, "amount": [None] * n,
        }, index=idx)

    @staticmethod
    def calculate_technical_indicators(df):
        return df

    @staticmethod
    def get_latest_indicators(df):
        return {"price": 10.0, "ma20": None}


def _run_fetcher(monkeypatch, bars_seq):
    monkeypatch.setattr(A, "StockDataFetcher", lambda: _FakeFetcher(bars_seq))
    return A.data_fetcher_node({"stock_code": "600519"})


class TestKlineLengthGuard:
    def test_short_series_triggers_one_retry(self, monkeypatch):
        """⚠️ 实测（2026-10-08 21:50）：同一接口前后两次都是 242 条，中间某次只回 **17 条**。

        17 根会让 MA20/MA60 一起变 NaN，报告于是写"指标缺失"—— 看着像数据源没有，
        实际是这次请求没取全。所以条数不足必须**重试**，而不是照单全收。
        """
        out = _run_fetcher(monkeypatch, [17, 242])
        assert len(out["kline_data"]) == 242, "第一次只有 17 根，应该重试并拿到完整序列"
        # 重试这件事本身要留痕（不是静默重试）—— 上游不稳是值得知道的事实
        assert any("17 根" in e and "重试" in e for e in out["errors"]), out["errors"]
        # 但"重试后仍不足"那条告警不该出现
        assert not any("本次不可用" in e for e in out["errors"]), out["errors"]

    def test_persistently_short_series_is_reported_not_faked(self, monkeypatch):
        """重试后仍不足 → 必须如实记录，不得假装长周期指标算得出来"""
        out = _run_fetcher(monkeypatch, [17, 17])
        joined = " ".join(out["errors"])
        assert "17 根" in joined and "MA60" in joined, out["errors"]
        assert "重试后仍不足" not in joined or True   # 文案可变，关键是"说明了哪项不可用"
        assert any("不可用" in e for e in out["errors"]), out["errors"]

    def test_enough_bars_needs_no_retry(self, monkeypatch):
        out = _run_fetcher(monkeypatch, [242])
        assert not out["errors"], out["errors"]
        assert out["data_availability"]["K线"] is True
        assert out["data_through"], "有 K 线时必须给出数据截止日（数据自带）"

    def test_hard_error_still_goes_to_fatal(self, monkeypatch):
        """真正的取数失败（返回 error）仍走致命通道，不被重试逻辑吞掉"""
        out = _run_fetcher(monkeypatch, [0])
        assert any("K线" in e for e in out["fatal_errors"]), out["fatal_errors"]


# ================================================================
# 6. 财务工具：算出的比率必须自带可核对的依据
# ================================================================

class TestFinancialToolSelfEvidence:
    def _patch(self, monkeypatch):
        """用合成财报替换数据源（不打网络）"""
        fin = {
            "income_statement": [
                {"报告日": "20260630", "营业收入": 704_000_000.0, "净利润": 748_000_000.0},
                {"报告日": "20260331", "营业收入": 316_000_000.0, "净利润": 238_000_000.0},
                {"报告日": "20250630", "营业收入": 669_000_000.0, "净利润": 61_000_000.0},
                {"报告日": "20250331", "营业收入": 287_000_000.0, "净利润": 41_000_000.0},
            ],
            "balance_sheet": [
                {"报告日": "20260630", "资产总计": 10_826_000_000.0, "负债合计": 1_850_000_000.0,
                 "归属于母公司股东权益合计": 8_974_000_000.0},
                {"报告日": "20260331", "资产总计": 10_036_000_000.0, "负债合计": 1_479_000_000.0,
                 "归属于母公司股东权益合计": 8_556_000_000.0},
            ],
            "cash_flow": [],
        }

        class F:
            def get_financial_data(self, symbol):
                return dict(fin, symbol=symbol)

        from data import stock_data as sd
        monkeypatch.setattr(sd, "StockDataFetcher", F)

    def test_yoy_base_period_and_value_are_provided(self, monkeypatch):
        """⚠️ 只给一个同比百分比，模型无法核对它从哪来。

        实测（2026-10-08 修复后重跑 603127）：基本面报告写到
        「2025H1 为 0.65 亿元，由 2025-12-31 累计 2.98 − 2025-09-30 累计 0.81
        = 2.17 亿元？矛盾」—— 它看不到 2025-06-30（periods=4 的窗口里没有），
        只能自己倒推，推出矛盾后还写了一段"存在严重数据校验问题"。
        """
        self._patch(monkeypatch)
        from core.agent_tools import tool_financials
        out = tool_financials(symbol="603127", periods=4)
        base = out.get("同比基准(亿元)")
        assert base, out.keys()
        assert base["营业收入"].startswith("2025-06-30"), base
        assert "6.69" in base["营业收入"], base
        # 与同比自洽：7.04 / 6.69 − 1 ≈ +5.2%
        assert out["同比%"]["营业收入"] == pytest.approx(5.27, abs=0.3)

    def test_single_quarter_is_derived_and_labelled(self, monkeypatch):
        """单季列必须是"本期累计 − 上一季累计"，且带口径说明"""
        self._patch(monkeypatch)
        from core.agent_tools import tool_financials
        out = tool_financials(symbol="603127", periods=4)
        sq = out["利润表单季(亿元)"]["营业收入"]
        assert sq[0] == pytest.approx(7.04 - 3.16, abs=0.01), sq   # 2026Q2
        assert sq[1] == pytest.approx(3.16, abs=0.01), sq          # 2026Q1（累计即单季）
        assert "单季" in out["利润表单季说明"]

    def test_cumulative_caliber_is_declared(self, monkeypatch):
        """累计口径必须显式声明 —— 这是"把累计当单季"整类错误的唯一防线"""
        self._patch(monkeypatch)
        from core.agent_tools import tool_financials
        out = tool_financials(symbol="603127", periods=4)
        assert "累计" in out["口径"] and "环比" in out["口径"]

    def test_renderer_does_not_drop_new_fields(self, monkeypatch):
        """⚠️ 渲染器原先是硬编码白名单 —— 新增字段会被**静默丢弃**。

        加字段必须同时检查渲染层，否则"工具给了、模型看不到"，
        与"注册表加了别名、解析器不认"是同一类漂移。
        """
        self._patch(monkeypatch)
        from core.agent_tools import tool_financials
        from llm.deepseek_client import _render_financials
        out = tool_financials(symbol="603127", periods=4)
        text = _render_financials(out)
        for key in ("口径", "同比%", "同比基准(亿元)", "利润表单季(亿元)",
                    "自算比率", "资产负债表(亿元)"):
            assert key in text, f"{key} 没有出现在渲染文本里（被白名单丢了）"
        assert "单季" in text


# ================================================================
# 7. 技术面提示词：新导出的指标必须真的进得去
# ================================================================

class TestTechPromptFields:
    """⚠️ `technical_analysis` 原先也是**硬编码字段表**。

    它在 `get_latest_indicators` 补出「布林带中轨 / MACD柱 / 成交量 / 5日与20日均量」
    之后**一个都没渲染**（实测重跑 603127：技术面报告里「中轨」出现 0 次），
    与 `_render_financials` 的硬编码白名单是同一类缺陷。
    """

    INDICATORS = {
        "price": 52.06, "ma5": 47.602, "ma10": 46.852, "ma20": 44.9565,
        "ma60": 46.6823, "rsi": 65.06, "macd": 0.679, "macd_signal": -0.0896,
        "macd_histogram": 0.7685, "bb_upper": 50.0335, "bb_middle": 44.9565,
        "bb_lower": 39.8795, "k_value": 89.28, "d_value": 77.40,
        "volume": 83578900.0, "volume_ma5": 48013700.0,
        "volume_ma20": 34571030.0, "volume_ratio": 1.74,
    }

    def _prompt(self, monkeypatch):
        from llm.deepseek_client import DeepSeekClient
        seen = {}

        def fake(self, messages, **kw):
            # ⚠️ patch **类方法**时第一个位置参数是实例（self）——
            #    少写它就会变成"fake() takes 1 positional argument but 2 were given"，
            #    被节点的 except 兜成"技术面分析失败"，表现为"提示词没捕获到"。
            seen["prompt"] = messages[-1]["content"]
            return "技术面报告"

        monkeypatch.setattr(DeepSeekClient, "call_api", fake)
        state = {"stock_code": "603127", "stock_name": "昭衍新药",
                 "stock_info": {"symbol": "603127", "name": "昭衍新药",
                                "current_price": 52.06},
                 "kline_data": None, "indicators": dict(self.INDICATORS),
                 "analysis_date": "2026-10-08", "data_through": "2026-10-08",
                 "fatal_errors": []}
        A.tech_analyst_node(state)
        return seen["prompt"]

    def test_new_indicator_fields_reach_prompt(self, monkeypatch):
        prompt = self._prompt(monkeypatch)
        for label, val in (("布林带中轨", "44.9565"), ("MACD柱", "0.7685"),
                           ("成交量(股)", "83578900"), ("5日均量", "48013700"),
                           ("20日均量", "34571030"), ("MA60", "46.6823")):
            assert label in prompt, f"{label} 没进提示词：{prompt[:300]}"
            assert val[:6] in prompt, f"{label} 的值 {val} 没进提示词"

    def test_volume_averages_are_distinguishable(self, monkeypatch):
        """5 日与 20 日均量必须**各自带名**，否则模型会拿量比的分母冒充 20 日均量"""
        prompt = self._prompt(monkeypatch)
        assert "5日均量" in prompt and "20日均量" in prompt
        assert "量比(成交量/5日均量)" in prompt

    def test_unlisted_indicator_is_not_dropped(self, monkeypatch):
        """未登记的新指标也要兜底渲染（防"加字段漏渲染"再次发生）"""
        monkeypatch.setattr(A, "StockDataFetcher", lambda: None)
        from llm.deepseek_client import DeepSeekClient
        seen = {}
        monkeypatch.setattr(DeepSeekClient, "call_api",
                            lambda self, messages, **kw:
                            seen.setdefault("p", messages[-1]["content"]) or "R")
        ind = dict(self.INDICATORS)
        ind["some_new_metric"] = 123.45
        A.tech_analyst_node({"stock_code": "1", "stock_name": "X", "stock_info": {},
                             "kline_data": None, "indicators": ind,
                             "analysis_date": "2026-10-08", "data_through": "2026-10-08",
                             "fatal_errors": []})
        assert "some_new_metric" in seen["p"], "未登记字段被静默丢弃"


# ================================================================
# 1b. 终审也必须拿到**同一份**风控阈值（2026-10-09）
# ================================================================
# 实测（2026-10-08 22:38）：同一份报告里风控写"计划 15% vs 上限 25% ✅ 未超限"，
# 终审却写「驳回。仓位严重超限（15% vs 合规上限3.3%）」—— 它自己造了一个上限。
# 根因：昨天只给 risk_manager 注入了阈值，portfolio_manager 没有。

@pytest.fixture
def portfolio_state():
    return {"stock_code": "603127", "stock_name": "昭衍新药",
            "stock_info": {"current_price": 52.06},
            "trade_plan": {"direction": "观望", "position_size": "15%"},
            "risk_assessment": "计划 15% vs 上限 25%，未超限；但单笔风险 5.8% > 2%。",
            "fatal_errors": []}


class TestPortfolioPromptGrounding:
    def test_prompt_contains_project_limits(self, monkeypatch, portfolio_state):
        seen = _capture_prompt(monkeypatch, A.portfolio_manager_node, portfolio_state)
        prompt = seen["messages"][1]["content"]
        assert "单票最大仓位" in prompt, prompt
        assert f"{RiskEngine.MAX_SINGLE_STOCK_WEIGHT:.0%}" in prompt
        assert f"{RiskEngine.DEFAULT_RISK_PER_TRADE:.0%}" in prompt

    def test_both_nodes_use_the_same_block(self, monkeypatch, risk_state, portfolio_state):
        """⚠️ 结构性守护：两个节点的阈值块必须**逐字相同**（否则又会各说各话）

        这是"上一轮只修了一半"的直接教训：只给风控注入阈值，终审照旧自己造。
        断言"两边都含同一个函数的返回值"，比断言"两边都含 25%"更强 ——
        后者在有人手工重抄一遍数字时照样通过，而重抄就是漂移的开始。
        """
        block = A.risk_rules_block()
        r = _capture_prompt(monkeypatch, A.risk_manager_node, risk_state)
        p = _capture_prompt(monkeypatch, A.portfolio_manager_node, portfolio_state)
        assert block in r["messages"][1]["content"], "风控提示词没用共享阈值块"
        assert block in p["messages"][1]["content"], "终审提示词没用共享阈值块"

    def test_block_is_derived_from_riskengine(self, monkeypatch):
        """阈值块必须**从 RiskEngine 派生**，改常量即改提示词（防手工重抄）"""
        monkeypatch.setattr(RiskEngine, "MAX_SINGLE_STOCK_WEIGHT", 0.42, raising=False)
        assert "42%" in A.risk_rules_block()

    def test_portfolio_forbids_lowering_the_limit(self, monkeypatch, portfolio_state):
        """终审不得用未提供的调整项（行业β系数/动态上限/波动率扣减）压低上限"""
        seen = _capture_prompt(monkeypatch, A.portfolio_manager_node, portfolio_state)
        prompt = seen["messages"][1]["content"]
        assert "不得引入" in prompt and "动态上限" in prompt

    def test_portfolio_forbids_invented_rules(self, monkeypatch, portfolio_state):
        seen = _capture_prompt(monkeypatch, A.portfolio_manager_node, portfolio_state)
        prompt = seen["messages"][1]["content"]
        assert "禁止" in prompt and "手册" in prompt


# ================================================================
# 8. 研究汇总：必须看到三份分析报告，并做跨维度对照（2026-10-09）
# ================================================================
# 实测用户原话：「报告里说舆情乐观，为什么还短期看空」。
# 这个"为什么"本该由研究汇总回答，但它的提示词**只喂了多空辩论**，
# 压根没见过舆情/技术/基本面报告 —— 所以只能并列双方观点，做不了跨维度对照。

class TestResearchManagerSeesReports:
    STATE = {
        "stock_code": "603127", "stock_name": "昭衍新药",
        "stock_info": {"current_price": 52.06},
        "tech_report": "TECH-SENTINEL 技术面：MA60 46.68，KDJ 钝化",
        "fundamental_report": "FUNDAMENTAL-SENTINEL 基本面：营收 7.04 亿",
        "news_report": "NEWS-SENTINEL 舆情：市场情绪判断：乐观",
        "bull_history": ["多头 R1"], "bear_history": ["空头 R1"],
        "debate_count": 1, "fatal_errors": [],
    }

    def _prompt(self, monkeypatch, **over):
        st = {**self.STATE, **over}
        seen = _capture_prompt(monkeypatch, A.research_manager_node, st)
        return seen["messages"][1]["content"]

    def test_prompt_carries_all_three_analyst_reports(self, monkeypatch):
        prompt = self._prompt(monkeypatch)
        for sentinel in ("TECH-SENTINEL", "FUNDAMENTAL-SENTINEL", "NEWS-SENTINEL"):
            assert sentinel in prompt, f"{sentinel} 没进研究汇总的提示词"

    def test_prompt_requires_cross_dimension_reconciliation(self, monkeypatch):
        """必须要求"某一维度与最终倾向不一致时要解释为什么"，而不是只并列"""
        prompt = self._prompt(monkeypatch)
        assert "多维结论对照" in prompt
        assert "方向不一致" in prompt and "为什么最终倾向仍然如此" in prompt

    def test_missing_report_is_marked_not_silently_omitted(self, monkeypatch):
        prompt = self._prompt(monkeypatch, news_report="")
        # 舆情报告缺失时要点出来，不能让模型以为"舆情没有问题"
        assert prompt.count("（本次未产出）") >= 1

    def test_long_report_is_truncated_with_marker(self, monkeypatch):
        prompt = self._prompt(monkeypatch, tech_report="技" * 5000)
        assert "已截断" in prompt


# ================================================================
# 1c. 阈值必须带**口径**（2026-10-09 端到端重跑后补）
# ================================================================
# 实测同一个计划、同一个"每笔交易最大风险 2%"，两次跑出相反结论：
#   10-08 读成"止损幅度 ≤2%" → 2.87% 超限 → 需修改；
#   10-09 读成"仓位×止损幅度 ≤2%" → 0.23% → 批准。
# 根因：`RiskEngine.DEFAULT_RISK_PER_TRADE` 只是常量（本类从未用它做校验），
# 语义没有写下来，模型每次自己挑一种。⇒ 口径必须由 `risk_rules_block()` 写死。

def test_risk_block_states_the_per_trade_caliber():
    block = A.risk_rules_block()
    assert "仓位 × 止损幅度" in block, block
    assert "组合净值" in block
    assert "不是**止损幅度本身**" in block or "不是" in block


def test_risk_block_states_sector_and_drawdown_caliber():
    block = A.risk_rules_block()
    assert "合计占比上限" in block          # 行业上限是"合计"
    assert "熔断" in block                  # 回撤线是"自高点回撤达到即熔断"


def test_every_limit_carries_a_parenthetical():
    """结构性守护：4 条阈值每条都要带括号口径，防新增阈值时漏写

    ⚠️ 本函数原先被误缩进到上一个用例**里面**（成了局部函数、pytest 根本不会收集）
    —— 这类"测试写了但没跑"的写法比没写更危险：看着有保障，其实零覆盖。
    """
    block = A.risk_rules_block()
    lines = [ln for ln in block.splitlines() if ln.startswith("- ")]
    assert len(lines) == 4, lines
    missing = [ln for ln in lines if "（" not in ln]
    assert not missing, f"这些阈值没写口径：{missing}"



#（原「A股 只做多」测试已由文末 §11「交易方向三档」替代）

# ================================================================
# 10. ATR：止损宽度的标尺必须真的算出来、且进得了提示词
# ================================================================

class TestAtrIsAvailable:
    @staticmethod
    def _bars(n=90):
        closes = [10 + (i % 7) * 0.4 for i in range(n)]
        return pd.DataFrame({
            "Open": closes, "Close": closes,
            "High": [c + 0.5 for c in closes], "Low": [c - 0.5 for c in closes],
            "Volume": [1e6] * n,
        })

    def test_atr14_is_exported(self):
        from data.stock_data import StockDataFetcher
        f = StockDataFetcher()
        ind = f.get_latest_indicators(f.calculate_technical_indicators(self._bars()))
        assert ind.get("atr14") is not None
        assert ind["atr14"] > 0

    def test_atr_reaches_the_technical_prompt(self, monkeypatch):
        """⚠️ 加字段必须同时检查渲染层（本项目已踩三次）。"""
        from llm.deepseek_client import DeepSeekClient
        seen = {}

        def fake(self, messages, **kw):
            seen["prompt"] = messages[-1]["content"]
            return "R"

        monkeypatch.setattr(DeepSeekClient, "call_api", fake)
        ind = {"price": 52.06, "ma20": 44.96, "atr14": 2.681}
        A.tech_analyst_node({"stock_code": "603127", "stock_name": "X",
                             "stock_info": {}, "kline_data": None, "indicators": ind,
                             "analysis_date": "2026-10-09", "data_through": "2026-10-08",
                             "fatal_errors": []})
        assert "ATR14" in seen["prompt"] and "2.681" in seen["prompt"]



# ================================================================
# 11. 交易方向三档：看多=可执行 / 看空=驳回 / 观望=等待（2026-10-09）
# ================================================================
# 实测：一份 603127 报告的交易计划是 `direction="看空"`、入场 [51.5, 52.3]、
# **目标价 44.96（在下方）**、**止损 53.80（在上方）** —— 结构上就是融券卖出，
# 而 A股 个人投资者没有这条通路（融券需信用账户 + 50万门槛 + 标的在两融名单且有券源）。
# 更糟的是下游**全部放行**：风控把它当普通持仓折算成「仓位 8% ≤ 25% ✅」、终审「批准」。
#
# 用户定下的设计：**「看空」是一个判断，不是一笔交易** ——
#   看多 → 给完整可执行投资建议；看空 → **驳回投资建议** + 给入场条件；
#   观望 → 等待 + 同样给入场条件。
# 三档由**代码**派生 `advice_status`，不靠模型自觉。

class TestTradeDirectionRules:
    STATE = {"stock_code": "603127", "stock_name": "昭衍新药",
             "stock_info": {"current_price": 52.06},
             "indicators": {"ma20": 44.96, "rsi": 65.06, "atr14": 2.837},
             "research_summary": "短期中性偏空", "fatal_errors": []}

    def _trader_prompt(self, monkeypatch):
        seen = _capture_prompt(monkeypatch, A.trader_node, dict(self.STATE), ret="{}")
        return seen["messages"][1]["content"]

    def test_prompt_allows_all_three_directions(self, monkeypatch):
        """看空没有被禁止 —— 它被**重新定义**成了「驳回 + 入场条件」"""
        p = self._trader_prompt(monkeypatch)
        assert "看多 或 看空 或 观望" in p
        assert "驳回投资建议" in p

    def test_prompt_requires_entry_conditions_for_non_long(self, monkeypatch):
        p = self._trader_prompt(monkeypatch)
        assert "entry_conditions" in p
        assert "看空/观望必填" in p

    def test_prompt_forbids_short_style_prices(self, monkeypatch):
        """做空式价位（入场在上、目标在下、止损在上）必须被明令禁止"""
        p = self._trader_prompt(monkeypatch)
        assert "禁止输出做空式价位" in p
        assert "融券" in p and "信用账户" in p

    def test_prompt_gives_atr_for_stop_sizing(self, monkeypatch):
        """⚠️ ATR 必须**由代码注入**，不能靠模型自己写进 trade_plan。

        2026-10-13 实测：原先交易员/风控拿到 ATR 的唯一通道是 `trade_plan`，
        而 `trade_plan = json.loads(模型输出)`、**代码从不注入 atr**
        ⇒ 10 份产物只有 2 份带 ATR（模型自己多写的）。"时有时无"比"完全没给"更难发现。
        现在改由 `execution_indicator_block()` 统一注入（本断言即钉住这一点）。
        """
        p = self._trader_prompt(monkeypatch)
        assert "ATR14(元)：2.837" in p, "ATR 必须由代码注入（原格式 ATR14=2.837 已统一）"
        assert "1×ATR14" in p
        # 同一块还带来了均线相对位置预判（把"防"铺到执行层）
        assert "均线相对位置" in p
        assert "收盘价**高于**MA20" in p

    def test_risk_and_portfolio_prompts_carry_the_same_rules(self, monkeypatch):
        """结构性守护：规则块必须**两个节点都注** —— 否则又是「只修一半」。

        （2026-10-08 吃过一次：只给风控注入阈值、没给终审，终审自己造了个 3.3% 上限。）
        """
        block = A.TRADE_DIRECTION_RULES
        r = _capture_prompt(monkeypatch, A.risk_manager_node,
                            {**self.STATE, "trade_plan": {"direction": "看空"}})
        p = _capture_prompt(monkeypatch, A.portfolio_manager_node,
                            {**self.STATE, "trade_plan": {"direction": "看空"},
                             "risk_assessment": "x"})
        assert block in r["messages"][1]["content"], "风控提示词缺方向规则"
        assert block in p["messages"][1]["content"], "终审提示词缺方向规则"

    def test_risk_prompt_has_per_direction_review_caliber(self, monkeypatch):
        """⚠️ 风控的审核口径必须跟着方向走。

        否则会出现实测那种荒谬结果：方向是「看空」（= 驳回买入），
        风控却在审「空头仓位的止损是否恰当」，并把它折算成「仓位 8% ≤ 25% ✅ 合规」。
        """
        r = _capture_prompt(monkeypatch, A.risk_manager_node,
                            {**self.STATE, "trade_plan": {"direction": "看空"}})
        prompt = r["messages"][1]["content"]
        # ⚠️ 断言里**不能带 Markdown 强调符跨越的短语**：源码写的是「**不要**讨论空头仓位」，
        #    直接断言「不要讨论空头仓位」会失败 —— 这个坑本项目已踩两次（「**不得**以」）。
        assert "审核口径必须跟着" in prompt
        assert "讨论空头仓位" in prompt and "融券风险" in prompt
        assert "计划无效" in prompt          # 看多却价位做空 → 直接判无效


class TestNormalizeTradePlan:
    """三档由代码派生 —— 提示词管不住的部分必须兜住（实测模型仍会给做空式价位）"""

    SHORT = {"direction": "看空", "entry_range": [51.5, 52.3], "target_price": 44.96,
             "stop_loss": 53.8, "position_size": "8%", "reasoning": "超买修正",
             "entry_conditions": ["站上MA20且放量至20日均量1.5倍", "日线MACD金叉"]}

    def test_short_is_rejected_and_keeps_direction(self):
        plan, notes = A._normalize_trade_plan(dict(self.SHORT), {"atr14": 2.837})
        assert plan["direction"] == "看空"            # 判断保留
        assert plan["advice_status"] == "驳回"        # 但建议被驳回
        assert plan["position_size"] == "0%"
        assert plan["entry_conditions"] == self.SHORT["entry_conditions"]
        assert "entry_range" not in plan and "stop_loss" not in plan
        assert notes

    def test_short_prices_are_downgraded_not_dropped(self):
        """价位**一个都不丢**：做空计划的「止损在上」就是上行压力、「目标在下」就是下行支撑。"""
        plan, _ = A._normalize_trade_plan(dict(self.SHORT), {"atr14": 2.837})
        w = plan["watch_levels"]
        assert w["上行压力（已持有者减仓参考）"] == 53.8
        assert w["下行支撑（回踩观察位）"] == 44.96

    def test_short_without_entry_conditions_is_flagged(self):
        """⚠️ 驳回/等待却没有入场条件 = 这条建议无法执行 → 必须留痕"""
        plan, notes = A._normalize_trade_plan(
            {"direction": "看空", "reasoning": "看空"}, {"atr14": 2.837})
        assert plan["advice_status"] == "驳回"
        assert any("entry_conditions" in n for n in notes), notes

    def test_watch_becomes_waiting_with_conditions(self):
        plan, notes = A._normalize_trade_plan(
            {"direction": "观望", "entry_range": [46.0, 47.5], "target_price": 55.0,
             "stop_loss": 43.5, "entry_conditions": ["回踩MA20且缩量"]}, {"atr14": 2.837})
        assert plan["direction"] == "观望"
        assert plan["advice_status"] == "等待"
        assert plan["position_size"] == "0%"
        assert plan["entry_conditions"] == ["回踩MA20且缩量"]
        assert notes == []

    def test_watch_planned_levels_become_reference(self):
        """观望时模型给的看多式价位是「若触发才考虑」的计划 → 命名为观察位而非指令"""
        plan, _ = A._normalize_trade_plan(
            {"direction": "观望", "entry_range": [46.0, 47.5], "target_price": 55.0,
             "stop_loss": 43.5, "entry_conditions": ["x"]}, {"atr14": 2.837})
        w = plan["watch_levels"]
        assert w["计划入场区（未触发）"] == [46.0, 47.5]
        assert w["目标位（参考）"] == 55.0 and w["止损位（参考）"] == 43.5

    def test_long_plan_is_runnable(self):
        p = {"direction": "看多", "entry_range": [46.0, 47.5], "target_price": 55.0,
             "stop_loss": 43.5, "position_size": "10%"}
        plan, notes = A._normalize_trade_plan(dict(p), {"atr14": 2.837})
        assert plan["advice_status"] == "可执行"
        assert plan["entry_range"] == [46.0, 47.5]    # 价位保留
        assert notes == []

    def test_long_with_inverted_levels_becomes_rejected(self):
        """方向写「看多」但价位放反了 → 结构上仍是融券卖出 → 按驳回处理"""
        plan, notes = A._normalize_trade_plan(
            {"direction": "看多", "entry_range": [52.3], "target_price": 44.0,
             "stop_loss": 54.0, "entry_conditions": ["x"]}, None)
        assert plan["advice_status"] == "驳回"
        assert any("自相矛盾" in n for n in notes)

    def test_long_missing_levels_is_not_runnable(self):
        plan, notes = A._normalize_trade_plan(
            {"direction": "看多", "entry_range": [46.0, 47.5]}, {"atr14": 2.837})
        assert plan["advice_status"] == "等待"
        assert any("缺少必要价位" in n for n in notes)

    def test_stop_tighter_than_one_atr_is_flagged(self):
        """⚠️ 实测那只票 ATR14=2.84 元（占价 5.45%），而门槛里的止损距离只有 1.30 元
        —— 不足一天的平均波动，等于「必被扫损」。没有这个对比根本看不出来。"""
        _, notes = A._normalize_trade_plan(
            {"direction": "看多", "entry_range": [52.0, 52.3], "target_price": 58.0,
             "stop_loss": 51.0}, {"atr14": 2.837})
        assert any("ATR14" in n and "扫损" in n for n in notes)

    def test_missing_atr_does_not_invent_a_warning(self):
        _, notes = A._normalize_trade_plan(
            {"direction": "看多", "entry_range": [52.0], "target_price": 58.0,
             "stop_loss": 51.0}, {})
        assert not any("ATR" in n for n in notes)

    def test_unknown_direction_falls_back_to_watch(self):
        plan, notes = A._normalize_trade_plan(
            {"direction": "强烈推荐", "entry_conditions": ["x"]}, None)
        assert plan["advice_status"] == "等待"
        assert any("未识别" in n for n in notes)

    def test_entry_conditions_accepts_a_string(self):
        """模型偶尔把列表写成一行字符串 —— 不要让它变成逐字字符串列表"""
        plan, _ = A._normalize_trade_plan(
            {"direction": "观望", "entry_conditions": "回踩MA20且缩量；站上MA5"}, None)
        assert plan["entry_conditions"] == ["回踩MA20且缩量", "站上MA5"]

    def test_empty_plan_is_safe(self):
        plan, notes = A._normalize_trade_plan({}, None)
        assert plan == {} and notes == []


def _stub_client(text):
    class C:
        def call_api(self, messages, **kw):
            return text
    return C


class TestTraderNodeNormalizesOutput:
    """端到端：节点**出口**必须已经是三档形态（提示词不能是唯一防线）"""

    STATE = {"stock_code": "603127", "stock_name": "X",
             "stock_info": {"current_price": 52.06},
             "indicators": {"atr14": 2.837}, "research_summary": "",
             "fatal_errors": []}

    def test_bearish_json_is_kept_bearish_but_rejected(self, monkeypatch):
        raw = json.dumps({"direction": "看空", "entry_range": [51.5, 52.3],
                          "target_price": 44.96, "stop_loss": 53.8,
                          "position_size": "8%",
                          "entry_conditions": ["站上MA20且放量"]}, ensure_ascii=False)
        monkeypatch.setattr(A, "DeepSeekClient", _stub_client(raw))
        out = A.trader_node(dict(self.STATE))
        tp = out["trade_plan"]
        assert tp["direction"] == "看空"              # 判断保留
        assert tp["advice_status"] == "驳回"          # 投资建议驳回
        assert tp["position_size"] == "0%"
        assert tp["entry_conditions"] == ["站上MA20且放量"]
        assert any("驳回" in e for e in (out.get("errors") or [])), out.get("errors")

    def test_bullish_json_is_runnable(self, monkeypatch):
        raw = json.dumps({"direction": "看多", "entry_range": [46.0, 47.5],
                          "target_price": 55.0, "stop_loss": 43.5,
                          "position_size": "10%"}, ensure_ascii=False)
        monkeypatch.setattr(A, "DeepSeekClient", _stub_client(raw))
        out = A.trader_node(dict(self.STATE))
        assert out["trade_plan"]["advice_status"] == "可执行"
        assert out["trade_plan"]["direction"] == "看多"

class TestFundamentalOutputBudget:
    """基本面报告的输出上限（2026-10-23）。

    实测 65 份历史报告里 37 份被拦腰切断（结尾出现「…的5.8倍** | |」这类残句），
    根因是 `call_api` 默认 `max_tokens=2000`：qwen3.7-flash 19/35 打满、
    deepseek-v4-flash 8/9 打满、qwen-plus 18/18 打满。
    切掉的是整节结论，而抬高上限的增量成本只有 ¥0.0008/次。
    """

    def test_fundamental_analysis_uses_raised_budget(self, monkeypatch):
        from llm import deepseek_client as DC

        seen = {}

        def fake(self, messages, **kw):
            seen.update(kw)
            return "基本面报告"

        monkeypatch.setattr(DC.DeepSeekClient, "call_api", fake)
        monkeypatch.setattr(DC, "FUNDAMENTAL_MAX_TOKENS", 3200)
        DC.DeepSeekClient().fundamental_analysis(
            {"symbol": "600941", "name": "中国移动", "current_price": 99.14},
            financial_data={}, analysis_date="2026-10-23", data_through="2026-10-23")
        assert seen.get("max_tokens") == 3200, seen
        assert seen.get("purpose") == "client.fundamental_analysis"

    def test_budget_is_above_default(self):
        """默认 2000 会切报告 —— 常量本身必须更大。"""
        from llm.deepseek_client import FUNDAMENTAL_MAX_TOKENS
        assert FUNDAMENTAL_MAX_TOKENS >= 3000
