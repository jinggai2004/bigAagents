"""深度分析报告导出（Markdown 渲染）的测试（core/report_render.py）

为什么需要：分析结果原本有**两份并行产物** —— `ui_runs/analysis_*.json`（UI 读它）
与手工脚本渲染的 `output/*.md`。手工那份必然漂移：实测 2026-10-08，
磁盘上的 md 渲染的是 22:06 那次运行，而 UI 里显示的是 22:38 那次
（后者还是**旧代码**产出的）—— 复核的人和看页面的人手里的根本不是同一份报告，
结论还分别是"批准 / 需修改 / 驳回"。所以改成：json 唯一事实来源，md 按需导出。
"""

from core import report_render as rr


def _payload(**over):
    p = {
        "stock_code": "603127", "stock_name": "昭衍新药",
        "current_price": 52.06, "analysis_date": "2026-10-08", "data_through": "2026-10-08",
        "data_availability": {"K线": True, "新闻": True, "财务报表": False},
        "tech_report": "MA60 46.68", "fundamental_report": "营收 7.04 亿",
        "news_report": "情绪：乐观", "bull_history": ["多头 R1", "多头 R2"],
        "bear_history": ["空头 R1", "空头 R2"], "research_summary": "短期中性偏空",
        "trade_plan": {"direction": "观望", "entry_range": [46.5, 48.0]},
        "risk_assessment": "计划 15% vs 上限 25%，未超限",
        "final_decision": "需修改", "errors": [], "audit_warnings": [],
        "fatal_errors": [],
    }
    p.update(over)
    return p


class TestRenderMarkdown:
    def test_header_carries_dates_and_price(self):
        md = rr.render_markdown(_payload(), generated_at="2026-10-09 09:30:00")
        assert "昭衍新药（603127）" in md
        assert "2026-10-09 09:30:00" in md
        assert "分析日：2026-10-08" in md and "数据截止：2026-10-08" in md
        assert "52.06" in md

    def test_all_sections_present_in_analysis_order(self):
        """章节顺序 = 分析链的实际执行顺序（论点→汇总→计划→风控→决策）

        摆错顺序不是"排版问题"：把风控排在交易计划之前，读者会以为风控先于计划存在。
        """
        md = rr.render_markdown(_payload())
        idx = [md.index(x) for x in ("技术面分析", "基本面分析", "舆情分析", "多空辩论",
                                    "研究汇总", "交易计划", "风控审核", "最终决策")]
        assert idx == sorted(idx), f"章节顺序被打乱：{idx}"

    def test_round_numbered_debates_are_kept(self):
        md = rr.render_markdown(_payload())
        for key in ("多头 R1", "多头 R2", "空头 R1", "空头 R2"):
            assert key in md, key

    def test_trade_plan_leads_with_direction_and_status(self):
        """⚠️ 交易计划**不能只丢一段 JSON**，且方向与建议状态必须顶在最前。

        实测（2026-10-09）：报告里 `direction="看空"`、目标价 44.96 在**下方**、
        止损 53.80 在**上方** —— 结构上就是融券卖出，而 A股 个人投资者没有这条通路。
        读者看到一段裸 JSON 根本不会意识到「这份计划不能照做」。
        现在按三档渲染：看多→🟢可执行；看空→🔴驳回 + 入场条件；观望→⚪等待 + 入场条件。
        """
        md = rr.render_markdown(_payload())
        assert "**方向：观望**" in md
        assert "建议：等待" in md
        assert "不建立仓位" in md
        # 方向/状态必须出现在价位之前
        assert md.index("**方向：观望**") < md.index("原计划价位")

    def test_bearish_plan_renders_rejection_and_entry_conditions(self):
        """看空 = **驳回买入建议** + 给入场条件；且必须写明「不等于可以做空」"""
        md = rr.render_markdown(_payload(trade_plan={
            "direction": "看空", "advice_status": "驳回", "position_size": "0%",
            "entry_conditions": ["站上MA20且放量", "MACD金叉"],
            "watch_levels": {"上行压力": 53.8, "下行支撑": 44.96}}))
        assert "建议：驳回" in md
        assert "投资建议已驳回" in md
        assert "可以做空" in md          # 必须点明"不等于可以做空"
        assert "1. 站上MA20且放量" in md and "2. MACD金叉" in md
        assert "上行压力" in md and "53.8" in md

    def test_runnable_plan_shows_executable_prices(self):
        md = rr.render_markdown(_payload(trade_plan={
            "direction": "看多", "advice_status": "可执行", "entry_range": [46.0, 47.5],
            "target_price": 55.0, "stop_loss": 43.5, "position_size": "10%"}))
        assert "建议：可执行" in md
        for key in ("入场区间", "目标价", "止损价", "仓位"):
            assert key in md, key

    def test_missing_entry_conditions_is_flagged_in_render(self):
        """驳回/等待却没人场条件 → 渲染层必须点出来（否则这条建议无法执行）"""
        md = rr.render_markdown(_payload(trade_plan={
            "direction": "看空", "advice_status": "驳回", "entry_conditions": []}))
        assert "未给出入场条件" in md

    def test_legacy_plan_without_status_infers_from_direction(self):
        """旧产物没有 `advice_status` 字段 → 由方向推断，不能渲染成空白"""
        md = rr.render_markdown(_payload(trade_plan={
            "direction": "看空", "entry_range": [51.5, 52.3], "target_price": 44.96,
            "stop_loss": 53.8}))
        assert "建议：驳回" in md
        assert "原计划价位" in md and "51.5" in md      # 价位降级显示，不丢

    def test_other_dict_sections_still_fall_back_to_json(self):
        """只改交易计划，别把其他 dict 段落的 json 兜底一起弄丢"""
        lines = []
        rr._fmt_block("试验段", "weird", {"weird": {"a": 1}}, lines)
        assert "```json" in "\n".join(lines)

    def test_audit_warnings_are_rendered_first_class(self):
        md = rr.render_markdown(_payload(
            audit_warnings=["[风控审核] 可核实性: 引用了输入中不存在的文件/制度《X办法》"],
            errors=["[风控审核] 可核实性: 引用了输入中不存在的文件/制度《X办法》",
                    "基本面: 三表取数失败"]))
        assert "《X办法》" in md
        assert "其他记录" in md and "三表取数失败" in md

    def test_no_warning_is_not_advertised_as_verified(self):
        """⚠️ 「零告警」不等于「已核实」——旧版本进程产出的报告压根没跑审计（实测 id=11）

        渲染层如果不写这句，导出件会被当成"审计通过"，那是最糟的误导。
        """
        md = rr.render_markdown(_payload(audit_warnings=[], errors=[]))
        assert "不等于" in md and "未启用审计" in md

    def test_missing_sections_are_explicit(self):
        md = rr.render_markdown(_payload(tech_report="", fundamentals=None))
        assert "（本次未产出这一段）" in md

    def test_fatal_errors_are_shown(self):
        md = rr.render_markdown(_payload(fatal_errors=["K线取数失败"]))
        assert "致命错误" in md and "K线取数失败" in md

    def test_empty_payload_does_not_crash(self):
        md = rr.render_markdown({})
        assert md.strip().startswith("#")
        assert "未知" in md or "（）" in md   # 股票名缺失也不报错


class TestFilename:
    def test_filename_has_code_and_data_date(self):
        assert rr.suggest_filename(_payload()) == "603127_深度分析_20261008.md"

    def test_filename_falls_back_when_dates_missing(self):
        name = rr.suggest_filename({"stock_code": "600519"})
        assert name.startswith("600519_深度分析_") and name.endswith(".md")

    def test_filename_never_empty_for_unknown_code(self):
        assert rr.suggest_filename({}).startswith("unknown_")


# ================================================================
# 给对话 Agent 的上下文块（「就这份报告提问」，2026-10-09）
# ================================================================
# 为什么用"注入上下文"而不是"让模型自己调工具查报告"：
# ① 模型不必猜"用户说的是哪一份"（同一只股票同一天可能有 4 条记录）；
# ② 不会读到错版本（注入的就是页面上正在看的那一份）；
# ③ 不消耗工具轮次。

class TestChatContext:
    def test_identity_and_dates_are_present(self):
        ctx = rr.build_chat_context(_payload(), analysis_id=11)
        assert "昭衍新药(603127)" in ctx
        assert "分析日期：2026-10-08" in ctx and "数据截止：2026-10-08" in ctx
        assert "id=11" in ctx

    def test_requires_citing_report_not_refetching(self):
        """必须明确"报告原文已给出、不需要再调 tool_analysis_report"。

        否则模型会去调工具取"最新一份"，而同一只股票可能有多条历史记录 ——
        很容易取到**不是用户正在看的那一份**，然后自信地讲错。
        """
        ctx = rr.build_chat_context(_payload())
        assert "tool_analysis_report" in ctx and "不需要" in ctx
        assert "段落名" in ctx
        assert "当前数据" in ctx          # 要求区分报告口径与当前数据

    def test_carries_audit_warnings(self):
        ctx = rr.build_chat_context(_payload(
            audit_warnings=["[风控审核] 可核实性: 引用了《X办法》"], errors=[]))
        assert "《X办法》" in ctx
        assert "必须一并说明" in ctx or "一并说明" in ctx

    def test_empty_audit_is_not_advertised_as_verified(self):
        ctx = rr.build_chat_context(_payload(audit_warnings=[], errors=[]))
        assert "不等于已核实" in ctx

    def test_says_report_has_no_content_rather_than_filling_in(self):
        ctx = rr.build_chat_context(_payload())
        assert "报告里没写" in ctx

    def test_missing_section_is_explicit(self):
        ctx = rr.build_chat_context(_payload(news_report=""))
        assert "（本次未产出这一段）" in ctx

    def test_long_section_is_capped_with_marker(self):
        """上下文每轮请求都会带上 —— 不限长会持续放大成本"""
        ctx = rr.build_chat_context(_payload(tech_report="技" * 5000))
        assert "已截断" in ctx
        assert len(ctx) < 20000, len(ctx)

    def test_debate_rounds_are_joined(self):
        ctx = rr.build_chat_context(_payload())
        assert "多头 R1" in ctx and "多头 R2" in ctx
        assert "空头 R1" in ctx and "空头 R2" in ctx

    def test_fatal_errors_are_shown(self):
        ctx = rr.build_chat_context(_payload(fatal_errors=["K线: 无数据"]))
        assert "K线: 无数据" in ctx

    def test_empty_payload_does_not_crash(self):
        assert rr.build_chat_context({}).strip()
