"""深度分析的"数据闭环"测试（v3，2026-09-28）

三份报告曾经在输入为空时照样输出极专业的内容。本文件把"必须真的把数据喂进去、
没数据就不许编"固化成断言：

  1. 提示词必须有【时间口径】（分析日期 + 数据截止）——原实现里一个日期都没有，
     模型只能抄训练语料里的模板头，输出「分析日期：2024年X月X日」；
  2. K 线必须真的进提示词——原 `technical_analysis` 收了 `stock_data` 却从未引用；
  3. 财务报表必须真的进提示词；取不到时**不调用 LLM**、直接说"未评估"；
  4. 舆情提示词里不能再有"如新闻不足请据此做出有限度的合理推断"这句**编造许可**。
"""

from unittest.mock import patch

import pandas as pd
import pytest

from core.agents import (
    fundamental_analyst_node,
    news_sentiment_node,
    tech_analyst_node,
    FUNDAMENTAL_UNAVAILABLE_TEMPLATE,
)
from llm.deepseek_client import DATA_GUARD_RULES, DeepSeekClient


def _make_client():
    """构造一个不会联网的 DeepSeekClient（跳过 __init__ 的 API Key 校验）"""
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
    return pd.DataFrame({"Close": [30 + i * 0.2 for i in range(n)],
                         "Volume": [1000 * (i + 1) for i in range(n)]}, index=idx)


# ===== 1. 时间口径 =====

def test_technical_prompt_carries_time_anchors_and_kline():
    client, box = _make_client()
    kline = _kline()

    client.technical_analysis({"symbol": "600211", "name": "西藏药业", "current_price": 36.48},
                              kline, {"ma20": 36.69, "rsi": 46.51},
                              analysis_date="2026-09-25", data_through="2026-09-04")
    user = box["messages"][1]["content"]

    assert "分析日期（本次分析执行日）：2026-09-25" in user
    assert "数据截止（输入中最后一根 K 线日期）：2026-09-04" in user
    # 报告首行格式被明确规定，避免模型自己发挥
    assert "数据截止：2026-09-04｜分析日期：2026-09-25" in user
    # K 线真的进去了（原实现 stock_data 只出现在签名里）
    assert "2026-09-04" in user and "收" in user
    assert DATA_GUARD_RULES.splitlines()[0] in user, "缺少反幻觉硬约束"


def test_technical_prompt_declares_missing_kline_instead_of_faking():
    """没有可靠日期（整数索引）→ 明说数据缺失，而不是编一段走势"""
    client, box = _make_client()
    client.technical_analysis({"symbol": "600211"}, pd.DataFrame({"Close": [1.0, 2.0, 3.0]}),
                              {}, analysis_date="2026-09-25", data_through="")
    user = box["messages"][1]["content"]
    assert "【历史K线数据】未取到" in user
    assert "数据缺失，无法判断" in DATA_GUARD_RULES


def test_technical_dates_default_to_unknown_not_guessed():
    client, box = _make_client()
    client.technical_analysis({}, None, {})
    user = box["messages"][1]["content"]
    assert "分析日期（本次分析执行日）：未知" in user
    assert "2024" not in user, "不允许任何未经输入确认的年份"


# ===== 2. 财务数据 =====

FIN = {"报告期": ["2025-06-30", "2025-03-31"], "单位": "亿元",
       "数据来源": "新浪财报三表",
       "利润表(亿元)": {"营业收入": [10.5, 3.2]},
       "同比%": {"营业收入": 12.3}}


def test_fundamental_prompt_renders_real_financials():
    client, box = _make_client()
    client.fundamental_analysis({"symbol": "600211", "name": "西藏药业"},
                                financial_data=FIN,
                                analysis_date="2026-09-25", data_through="2026-09-04")
    user = box["messages"][1]["content"]
    assert "2025-06-30" in user and "营业收入" in user
    assert "【财务报表（真实输入）】" in user
    assert "本次**未提供**任何财务报表数据" not in user
    # 有财报时不该再被要求分析"8期季报趋势"这类没有数据支撑的维度
    assert "最近8期" not in user


def test_fundamental_prompt_narrows_scope_without_financials():
    client, box = _make_client()
    client.fundamental_analysis({"symbol": "600211", "name": "西藏药业"},
                                financial_data=None, analysis_date="2026-09-25")
    user = box["messages"][1]["content"]
    assert "本次**未提供**任何财务报表数据" in user
    assert "只允许基于上面的基本信息与估值指标写一段简短说明" in user
    assert "报告期、定期报告、研报、机构名称" in user


# ===== 3. 缺失字段渲染：不许出现字面量 N/A 占位，缺失要集中声明 =====
#
# 用户实测反馈：基本面报告的「公司质地分析」整段是
#   「业务模式与核心竞争力、行业地位、护城河：数据缺失，无法判断。」
# 根因有两层：
#   ① 提示词把 `stock_info.get('sector', 'N/A')` 渲染成**字面量 N/A**
#      （`get_stock_info()` 根本不返回 sector/industry），模型分不清
#      「本次没有这个字段」和「字段值就是 N/A」；
#   ② 提示词又要求分析「业务模式/行业地位/护城河」这类**数据永远给不了**的定性维度，
#      模型只能逐项写"数据缺失"来满足要求。

def test_technical_prompt_lists_only_present_fields():
    client, box = _make_client()
    client.technical_analysis({"symbol": "600211", "name": "西藏药业", "current_price": 36.48},
                              _kline(), {"ma20": 36.69, "rsi": 46.51},
                              analysis_date="2026-09-25", data_through="2026-09-04")
    user = box["messages"][1]["content"]

    assert "- MA20：36.69" in user and "- RSI：46.51" in user
    assert "最新技术指标：\n（本次未取到任何技术指标）" not in user
    # 未取到的指标不许渲染成 "MA60：N/A"
    assert "MA60：N/A" not in user and "N/A" not in user
    assert "【本次未提供的字段】" in user
    assert "MA60" in user and "MACD" in user          # 缺失字段被集中列出


def test_technical_prompt_when_no_indicators_says_so_once():
    client, box = _make_client()
    client.technical_analysis({"symbol": "600211", "name": "西藏药业"},
                              None, {},
                              analysis_date="2026-09-25", data_through=None)
    user = box["messages"][1]["content"]
    assert "（本次未取到任何技术指标）" in user
    assert "N/A" not in user


def test_fundamental_prompt_does_not_render_na_placeholder():
    """`get_stock_info()` 不返回 sector/industry → 不许显示「行业：N/A」"""
    client, box = _make_client()
    client.fundamental_analysis({"symbol": "600211", "name": "西藏药业",
                                 "current_price": 36.48, "pe_ratio": 12.02},
                                financial_data={"报告期": ["2026-06-30"],
                                                "利润表(亿元)": {"营业收入": [12.0]}},
                                analysis_date="2026-09-25", data_through="2026-09-24")
    user = box["messages"][1]["content"]

    assert "N/A" not in user, "缺失字段被渲染成了字面量 N/A"
    assert "- 市盈率(PE)：12.02" in user
    assert "【本次未提供的字段】" in user
    assert "行业" in user and "细分行业" in user


def test_fundamental_prompt_forbids_enumerating_missing_dimensions():
    """要求分析"业务模式/护城河"却没数据 = 逼模型逐条写"数据缺失"，必须显式禁止"""
    client, box = _make_client()
    client.fundamental_analysis({"symbol": "600211", "name": "西藏药业"},
                                financial_data={"报告期": ["2026-06-30"],
                                                "利润表(亿元)": {"营业收入": [12.0]}},
                                analysis_date="2026-09-25", data_through="2026-09-24")
    user = box["messages"][1]["content"]

    assert "公司质地（仅限财务可推断的部分）" in user
    assert "本次完全未提供" in user          # 定性信息缺口被显式声明
    assert "不要逐项写「数据缺失」" in user
    assert "逐条" in user


def test_missing_note_absent_when_all_fields_present():
    from llm.deepseek_client import field_block, missing_note

    text, missing = field_block((("A", 1), ("B", 0), ("C", "x")))
    assert missing == []
    assert "- B：0" in text, "0 是有效值，不能当成缺失"
    assert missing_note(missing) == ""


def test_field_block_treats_source_placeholders_as_missing():
    from llm.deepseek_client import field_block

    text, missing = field_block((("行业", "N/A"), ("细分行业", "未知"), ("PE", None),
                                 ("PB", "nan"), ("PS", 3.1)))
    assert missing == ["行业", "细分行业", "PE", "PB"]
    assert text == "- PS：3.1"


# ===== 4. 节点级：取不到财报就不调用 LLM =====

def _min_state(**over):
    state = {
        "stock_code": "600211", "stock_name": "西藏药业",
        "analysis_date": "2026-09-25", "data_through": "2026-09-04",
        "stock_info": {"symbol": "600211", "name": "西藏药业",
                       "current_price": 36.48, "pe_ratio": 18.5},
        "kline_data": _kline(), "indicators": {"ma20": 36.69},
        "fund_flow_data": {}, "tech_report": "", "fundamental_report": "",
        "news_report": "", "errors": [], "fatal_errors": [],
    }
    state.update(over)
    return state


@patch("core.agents.DeepSeekClient")
@patch("core.agent_tools.tool_financials")
def test_fundamental_without_financials_does_not_call_llm(fin_tool, mock_client):
    fin_tool.return_value = {"error": "未取到财务报表（新浪/同花顺均不可用）"}
    result = fundamental_analyst_node(_min_state())

    mock_client.return_value.fundamental_analysis.assert_not_called()
    assert "基本面未评估" in result["fundamental_report"]
    assert "不做推测性补全" in result["fundamental_report"]
    assert "PE=18.5" in result["fundamental_report"], "可用字段要如实列出"
    assert result["data_availability"]["财务报表"] is False
    assert any("未评估" in e for e in result["errors"])


@patch("core.agents.DeepSeekClient")
@patch("core.agent_tools.tool_financials")
def test_fundamental_with_financials_passes_them_to_llm(fin_tool, mock_client):
    fin_tool.return_value = dict(FIN)
    mock_client.return_value.fundamental_analysis.return_value = "基本面报告"
    result = fundamental_analyst_node(_min_state())

    assert result["fundamental_report"] == "基本面报告"
    assert result["data_availability"]["财务报表"] is True
    kwargs = mock_client.return_value.fundamental_analysis.call_args.kwargs
    assert kwargs["financial_data"]["报告期"] == ["2025-06-30", "2025-03-31"]
    assert kwargs["analysis_date"] == "2026-09-25"
    assert kwargs["data_through"] == "2026-09-04"


def test_fundamental_unavailable_template_needs_name_code_fields():
    text = FUNDAMENTAL_UNAVAILABLE_TEMPLATE.format(name="西藏药业", code="600211", fields="PE=18.5")
    assert "西藏药业" in text and "600211" in text and "PE=18.5" in text


# ===== 5. 舆情节点：没有编造许可 =====

@patch("core.agents.DeepSeekClient")
@patch("data.news.NewsFlowDataFetcher")
def test_news_prompt_forbids_fabrication_when_no_news(mock_fetcher, mock_client):
    mock_fetcher.return_value.get_news_flow_data.return_value = {
        "data_success": False, "news_list": [], "count": 0,
        "note": "库内 242 条快讯中未检索到与 600211 关联的条目",
    }
    result = news_sentiment_node(_min_state())

    user = mock_client.return_value.call_api.call_args.args[0][1]["content"]
    assert "合理推断" not in user, "这句就是编造许可，必须删掉"
    assert "本次未检索到该标的的近期新闻/公告" in user
    assert "机构研报观点" not in user, "本项目从不提供研报，不该再要求模型给研报观点"
    assert DATA_GUARD_RULES.splitlines()[0] in user
    # 检索范围必须交代给模型，否则它会把"没查到"说成"没有"
    assert "库内 242 条快讯中未检索到" in user
    assert result["data_availability"]["新闻"] is False


@patch("core.agents.DeepSeekClient")
@patch("data.news.NewsFlowDataFetcher")
def test_news_availability_true_when_direct_source_hits(mock_fetcher, mock_client):
    """v4：按个股直查命中时必须置 `新闻=True`，并把条目（含链接）真的放进提示词。

    这是"消息面长期 0 条"修复后的**验收用例**：冷门个股靠全市场快讯库恒为 0 条，
    只有"按个股直查"这条路能让 availability 变 True。
    """
    mock_fetcher.return_value.get_news_flow_data.return_value = {
        "data_success": True, "count": 1, "sources": ["个股新闻 1 条"],
        "note": "个股新闻源（东财）：近7天；最近一条 2026-08-14（距今 45 天）",
        "news_list": [{"title": "西藏药业2026年中报净利润为6.08亿元", "content": "界面新闻",
                       "source": "界面新闻", "published_at": "2026-08-14 09:24:26",
                       "url": "http://finance.eastmoney.com/a/202608143841316271.html"}],
    }
    result = news_sentiment_node(_min_state())

    user = mock_client.return_value.call_api.call_args.args[0][1]["content"]
    assert result["data_availability"]["新闻"] is True
    assert "西藏药业2026年中报净利润为6.08亿元" in user
    assert "本次未检索到该标的的近期新闻/公告" not in user
    # 命中要交代"哪些源给了数据"，并把真实链接一起给模型
    assert "个股新闻 1 条" in user
    assert "http://finance.eastmoney.com/a/202608143841316271.html" in user
    # 时间窗口被放宽过也必须说清（否则一个月前的新闻会被当成近期事件）
    assert "距今" in user


@patch("core.agents.DeepSeekClient")
@patch("data.news.NewsFlowDataFetcher")
def test_scope_note_not_reported_as_degradation_when_news_found(mock_fetcher, mock_client):
    """检索范围说明只在"没查到"时才是问题。

    命中时它是提示词的一部分（模型据此判断时效），若同时塞进 errors，
    UI 会**常驻**一条"舆情数据降级"假告警 —— errors 必须只留真问题，
    否则真告警会被日常噪声淹没（本项目已有一次同类教训：`errors` 是 UI 顶部的告警区）。
    """
    mock_fetcher.return_value.get_news_flow_data.return_value = {
        "data_success": True, "count": 1, "sources": ["个股新闻 1 条"],
        "note": "个股新闻源（东财）：近7天仅2条，已放宽至近30天；最近一条 2026-09-21",
        "news_list": [{"title": "西藏药业回购", "content": "内容", "source": "财中社",
                       "published_at": "2026-09-01 17:22:00", "url": ""}],
    }
    mock_client.return_value.call_api.return_value = "舆情报告"
    result = news_sentiment_node(_min_state())

    assert result["data_availability"]["新闻"] is True
    assert not result.get("errors"), f"命中时不应有 errors，实际 {result.get('errors')}"


@patch("core.agents.DeepSeekClient")
@patch("data.news.NewsFlowDataFetcher")
def test_scope_note_is_reported_when_nothing_found(mock_fetcher, mock_client):
    """反之：没查到时必须把检索范围报进 errors，用户才知道"是没检索到，不是没有\""""
    mock_fetcher.return_value.get_news_flow_data.return_value = {
        "data_success": False, "count": 0,
        "note": "个股新闻源（东财）未收录 832000 的新闻（源以 KeyError 表示无结果）"}
    mock_client.return_value.call_api.return_value = "舆情报告"
    result = news_sentiment_node(_min_state())

    assert result["data_availability"]["新闻"] is False
    assert any("未收录" in e for e in result["errors"]), result.get("errors")


@patch("core.agents.DeepSeekClient")
@patch("data.news.NewsFlowDataFetcher")
def test_all_sources_empty_keeps_scope_and_availability_false(mock_fetcher, mock_client):
    """三源全空：必须如实置 `新闻=False` 并交代检索范围。

    原用例守护的是"主源正常返回空（不抛异常）时降级源也要走"——那个坑的载体
    （`_fetch_news_fallback`）已在 v4 摘除，但**同一个坑会换形式复发**：
    现在轮到"直查源未命中时快讯库/公告源仍要被调用"。该断言下沉到
    `tests/test_stock_news_source.py` 的调用计数用例，这里只守住节点侧的如实报告。
    """
    mock_fetcher.return_value.get_news_flow_data.return_value = {
        "data_success": False, "count": 0,
        "note": "个股直查与快讯库均未检索到与 600211/西藏药业 相关的条目"}
    result = news_sentiment_node(_min_state())

    user = mock_client.return_value.call_api.call_args.args[0][1]["content"]
    assert result["data_availability"]["新闻"] is False
    assert "本次未检索到该标的的近期新闻/公告" in user
    assert "个股直查与快讯库均未检索到" in user


@patch("core.agents.DeepSeekClient")
@patch("data.news.NewsFlowDataFetcher")
def test_news_fetcher_receives_stock_name_for_pool_fallback(mock_fetcher, mock_client):
    """必须把股票简称传下去：related_codes 只覆盖关注池，非关注池个股按代码查恒为 0 条"""
    mock_fetcher.return_value.get_news_flow_data.return_value = {
        "data_success": True, "news_list": [{"title": "公告", "source": "cls",
                                            "published_at": "2026-09-24 10:00:00"}]}
    news_sentiment_node(_min_state())

    args, kwargs = mock_fetcher.return_value.get_news_flow_data.call_args
    assert "600211" in args
    assert kwargs.get("stock_name") == "西藏药业"


# ===== 6. 审计接入节点 =====

@patch("core.agents.DeepSeekClient")
def test_tech_report_with_invented_year_gets_flagged(mock_client):
    mock_client.return_value.technical_analysis.return_value = (
        "**西藏药业（600211）专业技术分析报告**\n*数据截止：最新交易日收盘后｜分析日期：2024年X月X日*")
    result = tech_analyst_node(_min_state())

    assert any("可核实性" in e and "2024" in e for e in result["errors"]), result["errors"]


@patch("core.agents.DeepSeekClient")
def test_clean_tech_report_has_no_audit_noise(mock_client):
    mock_client.return_value.technical_analysis.return_value = (
        "数据截止：2026-09-04｜分析日期：2026-09-25\n现价 36.48，MA20 36.69。")
    result = tech_analyst_node(_min_state())
    assert not result.get("errors")
