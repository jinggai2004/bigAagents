"""「读深度分析报告」工具的契约测试（2026-10-09 新增）

背景（实测 `chat_logs` id=47/48）：用户问
「看最新的关于603127的深度分析报告，为什么舆情乐观仍短期看空」，
助手**只调了 `tool_latest_news`**，然后写"短期看空**可能**基于…**例如**，**若**股价已处于高位"。
根因不是态度问题：当时 15 个工具里**没有任何一个能读到报告正文**，
所以它只能拿行情/新闻**重新推一遍** —— 而"报告里是怎么写的"与"按现在数据我会怎么看"是两件事。

本文件钉住四件事：
1. 工具真的注册进去了（`TOOL_FUNCS` + schema），否则模型看不到它；
2. 按 code / 段落能取到正文，且段落名覆盖 `snapshot_from_state` 的真实字段（防改名）；
3. **必须**把该条报告**自身**的可核实性告警一起返回 —— 报告本身可能含幻觉，
   只给正文会让模型把幻觉当权威结论转述；
4. 读不到时返回 `error`（提示词规定"返回 error 就如实说拿不到"），
   而不是返回空结构让模型自己编一个"报告大概说了什么"。
"""

import pytest

from core import analysis_history as ah
from core import agent_tools as at


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """⚠️ 模块级 autouse 隔离：本模块所有用例都不得碰真实 `data/portfolio.db`。

    （历史事故：无夹具的用例 `DROP TABLE scheduled_jobs` 打到真实库 →
      schema 指纹短路 → 那张表永久不再重建。）
    """
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "test_report_tool.db")


def _state(**over):
    """一份最小但完整的"分析结果"state（字段与 snapshot_from_state 对齐）"""
    st = {
        "stock_code": "603127", "stock_name": "昭衍新药",
        "analysis_date": "2026-10-08", "data_through": "2026-10-08",
        "stock_info": {"name": "昭衍新药", "current_price": 52.06},
        "data_availability": {"K线": True, "新闻": True, "财务报表": True},
        "tech_report": "技术面：MA60 46.68，中轨 44.9565。",
        "fundamental_report": "基本面：营业收入 7.04 亿。",
        "news_report": "舆情：市场情绪判断：乐观。",
        "bull_history": ["多头 R1 …", "多头 R2 …"],
        "bear_history": ["空头 R1 …", "空头 R2 …"],
        "research_summary": "研究汇总：短期中性偏空。",
        "trade_plan": {"direction": "观望", "entry_range": [46.5, 48.0]},
        "risk_assessment": "风控：计划 15% vs 上限 25%，未超限。",
        "final_decision": "需修改。",
        "errors": [], "fatal_errors": [],
    }
    st.update(over)
    return st


class TestRegistration:
    def test_registered_in_funcs_and_schemas(self):
        assert "tool_analysis_report" in at.TOOL_FUNCS
        names = [s["function"]["name"] for s in at.TOOL_SCHEMAS]
        assert "tool_analysis_report" in names, names
        assert len(names) == len(set(names)), "schema 名称重复"

    def test_schema_exposes_sections_enum(self):
        """段落枚举必须与实现同一份清单 —— 否则模型会传一个取不到的段落名"""
        schema = next(s for s in at.TOOL_SCHEMAS
                      if s["function"]["name"] == "tool_analysis_report")
        enum = schema["function"]["parameters"]["properties"]["sections"]["items"]["enum"]
        assert set(enum) == set(at._SECTION_NAMES), (enum, at._SECTION_NAMES)

    def test_section_keys_exist_in_snapshot(self):
        """结构性守护：声明的段落键必须真的存在于 `snapshot_from_state` 的输出里。

        ⚠️ 这是防"改名后静默取空"——`payload.get(key)` 对不存在的键返回 None，
        工具会返回空段落，模型看到的是"这一段没有内容"，而不是报错。
        """
        keys = set(ah.snapshot_from_state(_state()).keys())
        missing = [k for _n, k, _m in at._REPORT_SECTIONS if k not in keys]
        assert not missing, f"这些段落键在快照里不存在（改名了？）：{missing}"


class TestReading:
    def test_returns_latest_report_with_sections(self):
        ah.save_analysis(_state(), source="ui")
        r = at.tool_analysis_report(code="603127", sections=["舆情", "最终决策"])
        assert r.get("找到份数") == 1, r
        rep = r["报告"][0]
        assert rep["analysis_id"] > 0
        assert rep["股票"] == "昭衍新药(603127)"
        assert rep["分析日期"] == "2026-10-08"
        assert rep["数据截止"] == "2026-10-08"
        assert set(rep["段落"]) == {"舆情", "最终决策"}, rep["段落"]
        assert "乐观" in rep["段落"]["舆情"]

    def test_list_sections_are_joined_not_lost(self):
        """多空辩论是**列表**（每轮一段），取出来必须是两段都在，不能只留最后一段"""
        ah.save_analysis(_state(), source="ui")
        rep = at.tool_analysis_report(code="603127", sections=["空头辩论"])["报告"][0]
        text = rep["段落"]["空头辩论"]
        assert "空头 R1" in text and "空头 R2" in text, text

    def test_dict_section_is_serialized(self):
        """交易计划是 dict —— 必须被序列化成可读文本，不能变成 "{'direction': ...}" 的 repr"""
        ah.save_analysis(_state(), source="ui")
        rep = at.tool_analysis_report(code="603127", sections=["交易计划"])["报告"][0]
        assert "观望" in rep["段落"]["交易计划"]

    def test_limit_returns_multiple_newest_first(self):
        # ⚠️ 必须跨日：同日同票有唯一索引（第二次会覆盖第一次），
        # 这里测的是"多份历史按时间倒序"，不是 UPSERT。
        ah.save_analysis(_state(final_decision="第一份", analysis_date="2026-10-07"),
                         source="ui")
        ah.save_analysis(_state(final_decision="第二份", analysis_date="2026-10-08"),
                         source="ui")
        r = at.tool_analysis_report(code="603127", sections=["最终决策"], limit=2)
        assert r["找到份数"] == 2
        assert "第二份" in r["报告"][0]["段落"]["最终决策"], "应按时间倒序（最新在前）"

    def test_long_section_is_truncated_with_marker(self):
        """长段落截断时**必须留标记**，否则模型会以为读到的是全文"""
        ah.save_analysis(_state(tech_report="技" * 6000), source="ui")
        rep = at.tool_analysis_report(code="603127", sections=["技术面"])["报告"][0]
        text = rep["段落"]["技术面"]
        assert "已截断" in text and "6000" in text, text[:200]

    def test_unknown_section_name_falls_back_to_all(self):
        """传了实现不认识的段落名 → 退回全部（而不是静默返回空）"""
        ah.save_analysis(_state(), source="ui")
        rep = at.tool_analysis_report(code="603127", sections=["不存在的段落"])["报告"][0]
        assert set(rep["段落"]) == set(at._SECTION_NAMES)

    def test_empty_section_is_empty_string_not_fabricated(self):
        ah.save_analysis(_state(news_report=""), source="ui")
        rep = at.tool_analysis_report(code="603127", sections=["舆情"])["报告"][0]
        assert rep["段落"]["舆情"] == ""


class TestHonesty:
    def test_error_when_no_record(self):
        """⚠️ 读不到必须是 `error`：提示词规定"返回 error 就如实说拿不到"。

        返回空结构会让模型有机会自己编一个"报告大概说了什么"——实测就是这么发生的。
        """
        r = at.tool_analysis_report(code="999999")
        assert "error" in r, r
        assert "999999" in r["error"]

    def test_carries_report_own_audit_warnings(self):
        """⚠️ 核心设计：报告自身可疑项必须随正文一起返回。

        报告可能含幻觉（2026-10-08 就出现过整段编造的监管框架）。
        只把正文交给模型，它会把这些内容当**权威结论**转述给用户 —— 比读不到更糟。
        """
        warn = "[风控审核] 可核实性: 引用了输入中不存在的文件/制度《某某风控办法》"
        ah.save_analysis(_state(errors=[warn]), source="ui")
        rep = at.tool_analysis_report(code="603127")["报告"][0]
        assert rep["本报告的可核实性告警"] == [warn], rep
        assert "必须一并说明" in rep["可核实性说明"]

    def test_empty_audit_is_not_advertised_as_verified(self):
        """「零告警」不等于「已核实」——旧版本进程产出的报告压根没跑审计（实测 id=11）"""
        ah.save_analysis(_state(errors=[]), source="ui")
        rep = at.tool_analysis_report(code="603127")["报告"][0]
        assert rep["本报告的可核实性告警"] == []
        assert "未经核实" in rep["可核实性说明"], rep["可核实性说明"]

    def test_other_errors_do_not_leak_audit_duplication(self):
        """普通 errors（如"取数失败"）与审计告警要分开列，且不重复计"""
        warn = "[空头R1] 可核实性: 提及年份「2020」，但输入数据中无此年份"
        ah.save_analysis(_state(errors=[warn, "基本面: 三表取数失败"]), source="ui")
        rep = at.tool_analysis_report(code="603127")["报告"][0]
        assert rep["本报告的可核实性告警"] == [warn]
        assert rep["本报告的其他记录"] == ["基本面: 三表取数失败"]

    def test_usage_note_requires_citation_and_dates(self):
        """使用须知必须写明：标段落+analysis_id、空段落≠没问题、告警要转述、日期要带"""
        ah.save_analysis(_state(), source="ui")
        note = at.tool_analysis_report(code="603127")["使用须知"]
        assert "analysis_id" in note and "段落" in note
        assert "不等于" in note
        assert "告警" in note
        assert "数据截止" in note

    def test_tool_never_raises_on_bad_input(self):
        """工具层契约：失败返回 error dict，绝不把异常抛进 Agent 循环

        （`limit="x"` 会在 `int()` 抛 ValueError、`sections=123` 会在迭代时抛 TypeError ——
         两者都必须被 `_tool` 兜成 error dict，而不是中断整个对话。）
        """
        for kwargs in ({"code": 123}, {"code": "", "sections": None, "limit": "x"},
                       {"code": "603127", "sections": 123, "limit": -5}):
            r = at.tool_analysis_report(**kwargs)
            assert isinstance(r, dict), (kwargs, r)
