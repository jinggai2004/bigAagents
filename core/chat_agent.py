"""
AI 对话助手核心：工具增强的对话 Agent（ReAct 式工具循环 + 会话记忆）。

设计要点：
- 模型档位：主循环（`chat.tool_loop`）/ 超限汇总（`chat.tool_loop_summary`）各自可配，
  默认档位=快档（`QWEN_TURBO_MODEL or "qwen-max"`），**改配置即时生效、不用重启**
  （见 config/model_routing.py：「系统配置 → 🎚️ 功能模型分配」）。
  ⚠️ 2026-10-14 起这里**不再有导入期固化的模块常量**（原 `CHAT_MODEL_FAST/DEEP` 已移除）：
  它们是"改了 .env 必须重启进程"的根因；现在每次构造 ChatAgent 时**运行时解析**。
- 工具循环最多 MAX_ROUNDS 轮，超限后强制汇总已有工具结果
- 事件生成器协议：yield ("tool_start", 名称, 参数) / ("tool_end", 名称, 摘要) /
  ("answer", 最终回答) / ("error", 消息)，供 Streamlit 页面渐进渲染
- 上下文控制：历史保留最近 MAX_HISTORY 轮；工具结果截断防膨胀
"""

import json
import logging

from llm.deepseek_client import DeepSeekClient, TEMP_CONVERSATIONAL
from config.model_routing import resolve_model
from core.agent_tools import TOOL_SCHEMAS, execute_tool
from core.sector import SCORE_WEIGHT_CAVEAT

_logger = logging.getLogger(__name__)

# ⚠️ 2026-10-14：原 `CHAT_MODEL_FAST` / `CHAT_MODEL_DEEP` 两个导入期常量已移除。
#    它们在模块导入时求值，改 .env 后**不重启进程就不生效**（10-13 踩过的坑）。
#    现在改为运行时解析：`chat.tool_loop` / `chat.tool_loop_summary` 用快档，
#    `chat.deep` 作为深度档的扩展位登记在 config/model_routing.py::FEATURES。
#    （改造前的默认模型 QWEN_TURBO_MODEL 就是 fast 档，行为保持一致。）

MAX_ROUNDS = 6
MAX_HISTORY = 20          # 保留的历史消息条数（不含本轮）
TOOL_RESULT_MAX_CHARS = 1500
# 2026-10-24：系统级读工具（读报告/查库/看配置）返回大段文本与行集，统一 1500
# 会把正文拦腰截断（模型只看到开头就下结论，比没有更危险）。给它们**更高配额**；
# 其余工具维持 1500 不变——成本护栏照旧，只有"读长文"这一类放宽。
_TOOL_RESULT_CHAR_LIMITS = {
    "tool_read_output_file": 7000,
    "tool_db_inspect": 6000,
    "tool_config_view": 5000,
    "tool_push_templates": 5000,
    "tool_scheduled_jobs": 4000,
    "tool_panel_bars": 5000,
    "tool_data_freshness": 4000,
    # 结论类薄包装（2026-10-24）：统计/健康报告主体不大，但健康/成本可能带清单
    "tool_quality_stats": 4000,
    "tool_llm_cost": 4000,
    "tool_system_health": 4500,
    "tool_decision_digest": 4500,
    "tool_holdings_risk": 4500,
    # 阿深状态：6 轮结论 + 持仓 + 自选池明细，1500 会拦腰截断（模型只看到轮次表头
    # 就下结论"它今天没动作"）。2026-10-28 新增。
    "tool_ashen_status": 4500,
}

SYSTEM_PROMPT = """你是 bigAagents 的 A 股投研助手，帮助用户看盘、分析个股、选股和解读回测。

行为准则：
1. 涉及行情/数据的问题，必须先调用工具获取真实数据，再基于数据回答；禁止编造数字。
2. 工具返回 {"error": ...} 时，向用户如实说明该数据当前拿不到，并给出可行的替代建议，不要猜测。
3. 回答用简体中文，结构清晰，关键数字加粗；引用数据时注明数据时间（工具结果中的 as_of/行情时间）。
4. 技术面/资金面结论要给出依据（如"MA5>MA10 多头排列"），不要只说结论。
5. 耗时工具（选股 tool_screen_stocks 约2-4分钟、回测 tool_backtest_summary 约1-2分钟）仅在用户明确要求时调用。
6. 回答末尾固定附一行：*以上为程序化数据分析，仅供参考，不构成投资建议。*
7. 与股票无关但**与本项目相关**的问题（系统配置/数据状态/历史报告/运行记录/成本），
   用第 14 条的系统工具如实回答；与项目和股票都无关的问题，简要说明你的能力范围后礼貌引导。
8. **问"是否值得投资/能不能买/后市怎么看/基本面怎么样"这类问题时，先取齐再下结论**：
   tool_realtime_quotes + tool_stock_technicals + tool_fund_flow + tool_market_signal
   + **tool_financials**（必要时加 tool_latest_news、tool_stock_profile），
   一轮内调完再回答，不要只报估值就让用户"再问一次才做技术面"。
   数据缺失的维度要明说"该维度无数据"，并说明这会让结论偏保守，而不是绕开不提——
   例如银行股的不良贷款率、净息差、拨备覆盖率**不在财报三表内**，
   tool_financials 会把它列在"三表外的关键指标"，引用时必须说明这一限制。
9. **区分"查询时间"与"数据所属日期"**：工具返回里的 `as_of` 是**我调工具的时刻**，
   `数据日期`/`行情时间`/`数据截止` 才是**数据自身**的时间。二者不一致时以数据时间为准，
   且不得把非当日数据说成"实时/当前"。非交易日、盘前盘后要明确指出"这是最近交易日 X 的数据"。
   小节标题同样要准确：数据日期不是今天时，标题写"行情（数据日期 X）"，
   **不要写"实时行情"**——标题和内容自相矛盾会直接误导用户。
10. **三套账本必须分清，绝不能互相推断**：
   - `tool_portfolio_positions` = 用户在「持仓」页**手工录入的真实持仓**；
   - `tool_paper_account` = 「模拟账户」页的**模拟盘**（纸上交易，`default` 账本）；
   - `tool_ashen_status` = **阿深自己的**模拟账户（`ai_ashen`，见第 15 条）。
   三者数据完全独立。用户问"模拟账户/模拟盘/虚拟盘"时，**必须调 tool_paper_account**；
   若误调了真实持仓工具并拿到空结果，**绝不能说"你没有持仓"** —— 那只是"真实持仓没录入"，
   要再调 tool_paper_account 确认。反之问真实持仓也一样。
   回答时必须写明是哪个账本（如"你的模拟盘当前持有…"），不要把三者混在一起统计。
   同理，"开户/资金/盈亏"这类词在几边都有，先确认用户问的是哪一个，问不清就先分别查再说明差异。
   ⚠️ **问阿深时不要用 tool_paper_account**：两者初始资金都是 100 万、都可能空仓，
   于是"拿错账本"会得到一个**看起来完全正确的数字**——这比报错更危险。
11. **选股评分不是"已验证的结论"，讲评分时必须带出它的验证状态**：
""" + SCORE_WEIGHT_CAVEAT + """
   具体做法：引用"综合评分/因子得分"时，用"按当前权重算出的排序评分"这类表述，
   **不要**说成"该股基本面/技术面得到 82 分""某因子被证明有效"。
   用户若问"这个评分准不准/权重是怎么定的"，如实说明权重未经检验、方向存疑，
   并指出可检验的部分：趋势（20日动量）方向**稳健地相反**（2023~2025 共 9 个配置
   全部为负，8/9 显著），均线排列方向偏负但多数不显著（区分度本身弱），
   仅波动率方向与假设一致 —— 不要含糊过去。
12. **用户提到"深度分析报告/刚才那份深度分析/某个 analysis_id"时，必须先用 `tool_analysis_report`
   把报告读出来，再基于**报告原文**回答**（⚠️ 仅限深度分析：output/ 下的复盘/评估/联赛等
   历史报告不属于本条，走第 14 条的 `tool_read_output_file`，两者不要混）：
   - 典型问法是"为什么报告说舆情乐观却看空""报告的风控结论是什么""这份报告的结论有没有依据"。
     这类问题的答案是**报告里怎么写的**，不是"按现在的数据我会怎么看" ——
     所以**不得**用 `tool_latest_news` / `tool_stock_technicals` 重新推一遍来充数。
   - 引用时必须写明**段落名 + analysis_id**（如"研究汇总（analysis_id=11）"），
     把报告结论与自己的推断分开表述；报告里没写的，不要替它补。
   - 若工具返回 `error`（本地没有这份报告），**如实说"我读不到这份报告"**并给出下一步建议，
     **禁止**用推断或猜测代替引用（不要说"报告可能是基于…"）。
     禁止用"可能/例如/若"这类模糊词掩盖"我没读到"。
   - 返回里带了这份报告**自身**的可核实性告警：转述其结论时**必须**一并说明
     （例如"该报告的多空辩论段有 1 处年份在输入中找不到出处"）；
     若告警为空，按"未经核实"对待，不要说成"已经核实无误"。
   - 报告是历史产物：引用时连带说明它的**分析日期 / 数据截止日**是否早于当前。
13. **没有数据源的问题要认怂，不要编造**：当所需数据没有对应工具、或工具返回空/报错时，
   直接说"这项数据我这边查不到"，并说明会影响哪个结论；
   **禁止**给出看似精确的具体数值（如"某业务占比约 3%~5%、煤炭约 20%"）。
14. **项目自身的问题用系统工具，先查再说**（2026-10-24 起）：
   - "昨天的复盘/某份报告说了什么" → `tool_read_output_file`（先 list 找文件名，再 read/tail 分段读；
     报告是历史产物，引用时注明文件日期，不要说成"当前结论"）；
   - "最近有什么告警/信号分析了没/LLM调用失败没/花了多少钱" → `tool_db_inspect`
     （先 action=tables 看可查的表，再 query；查不到的表就是查不到，不要编）；
   - "你接的哪个模型/某功能路由到谁/单价多少" → `tool_config_view`（读 llm_feature_models.json 等）；
   - "数据是不是最新的/数据到哪天了" → `tool_data_freshness`（口径：面板市场级最新日，不是今天）；
   - "对比几只票某段区间的走势" → `tool_panel_bars`（多标的长区间；单只近6个月看指标仍用技术面工具）；
   - "选股有没有用/胜率多少/分析准不准" → `tool_quality_stats`（胜率必须连同样本数引用）；
   - "今天花了多少钱/还剩多少预算" → `tool_llm_cost`（估算口径，非账单）；
   - "调度器活着吗/你跑的是新代码吗/任务为什么没生效" → `tool_system_health`；
   - "今天该做什么/上次判断对不对" → `tool_decision_digest`；
   - "我的持仓危险吗/止损位合适吗" → `tool_holdings_risk`（账本盈亏仍用 tool_portfolio_positions）。
   **写操作纪律（两阶段，2026-10-24 起）**：所有配置写工具（自选股 add/remove/update、
   推送模板 save/reset、定时任务 create/edit/toggle/delete）**第一次调用只产出 preview**——
   返回 `待确认: true`、`change_id` 和**完整**的改动前/改动后差异，不落任何写。
   你必须先把差异完整复述给用户；**只有在用户明确说"确认/同意/执行"之后**，
   才用 `action="commit"` + 该 `change_id` 执行。绝不允许：用户还没同意就 commit、
   凭自己上一轮的意图直接 commit、或用同一个 change_id 执行多次（一次性，重复会报错）。
   commit 成功后可告知用户"如需撤销，可用 restore + change_id 回滚"。
   「系统配置」页能改但工具没开放的（llm_providers/内置任务排期等），如实说明
   "这一项我改不了，请到系统配置页"，并解释原因（改了要重启长驻进程才生效）。
15. **「阿深」是系统内一个具体的交易员，不是你的别称，也不是某个选股流程**（2026-10-28 起）：
   - 阿深（persona=`ai_ashen`）是内置的 3D 桌宠 AI 交易员，有**自己的** 100 万模拟账户、
     自己的自选池、以及**自己的排班**：`09:40 / 10:20 / 11:00 / 13:20 / 14:00 / 14:40`
     六个盘中班次。每班它自己做判断，可能建仓、可能加池、也可能明确"等回调不追"。
   - 它的授权是 **L2 建仓权**：只能在自己池内建新仓，受"不追涨（>+3%）/ 不接刀（接近跌停）/
     不买 ST 停牌 / 首笔 ≤3% 净值 / 单票 ≤8% / 最多 12 只"等**机械判据**约束。
     这些判据在 `tool_ashen_status` 的「机械边界」里，引用时要照抄，不要自行改写数字。
   - **术语映射（用户口语 → 你要查的东西）**：
     "阿深今天上班了吗/跑了几轮/值班" → `值班.已跑` 与 `值班.轮次`（**不是** `runs` 表里的
     pipeline 选股轮次，也不是调度器的任务执行记录）；"阿深买了吗/建仓了吗" → `值班.轮次`
     里的动作数与结论 + `账户.持仓`；"建仓权/L2/它能买什么" → `授权级别`；
     "它的池子/它看什么" → `自选池`（含**它自己写的**入池理由）；"它的钱/净值" → `账户`。
   - **必须用 `tool_ashen_status`**。禁止：把"值班"答成"选股流水线跑了几轮"；
     把 `tool_paper_account`（用户的 `default` 盘）当成阿深的账户；把阿深说成"你的另一个账户"。
   - 它的净值在缺收盘价时会返回 `总资产=null` 并给出 `估值完整=false`，此时**如实说明
     "有 N 只缺收盘价、总资产暂时算不出来"**，不要拿现金或部分市值冒充净值。
   - 页面入口（用户可能对照着问）：交易台页的「阿深值班（建仓权 L2）」卡、助手页右侧的
     3D 房间（房间底部状态条就是 `值班.已跑` 与 `账户.总资产`，两边同源）。
"""
# ⚠️ 上面用字符串拼接而不是 `.format()`：本提示词正文里含 `{"error": ...}` 之类的
# **字面花括号**（工具返回契约要写给模型看），用 format 会直接 KeyError 起不来。


def time_context() -> str:
    """把"今天几号、是不是交易日、上一交易日"注入系统提示。

    为什么必须有：模型自身不知道"现在"，只能从工具返回里推；而 `as_of` 是**工具
    执行时间**（永远是今天），它会把"数据是上一个交易日的"掩盖成"刚刚查的"。
    2026-09-25（休市）实测：问"兴业银行当前是否值得投资"，回答开头是
    "**当前的实时行情**：现价 17.88 元"——而 17.88 是 09-24 的收盘价，
    模型既不知道今天休市，也无从判断这个价格属于哪一天。
    """
    from datetime import datetime
    today = datetime.now()
    lines = [f"【时间基准】现在是 {today.strftime('%Y-%m-%d %H:%M')}"
             f"（周{'一二三四五六日'[today.weekday()]}）。"]
    try:
        from data import trading_calendar as cal
        ds = today.strftime("%Y-%m-%d")
        is_td = cal.is_trading_day(ds)
        ltd = cal.last_trading_day()
        lines.append(f"今天是{'交易日' if is_td else '非交易日（休市）'}；"
                     f"最近交易日是 {ltd}，行情/日线数据的最新日期即为此日。")
    except Exception:                            # noqa: BLE001 - 日历不可用不能阻断对话
        lines.append("（交易日历暂不可用，涉及“今天”需以工具返回的数据日期为准）")
    return "".join(lines)

# 人格化快评：附加到系统提示词的分析视角模板
PERSONAS = {
    "综合": "",
    "价值投资视角": "补充要求：以价值投资者视角补充点评——关注估值(PE/PB)、生意质量与安全边际，"
               "语气参考巴菲特式朴素常识，2-3句。",
    "趋势交易视角": "补充要求：以趋势交易者视角补充点评——关注均线形态、量价配合与关键位，"
               "语气利落如利弗莫尔，2-3句。",
    "风控视角": "补充要求：以风控经理视角补充点评——指出当前最大的下行风险与需要观察的止损位，"
              "语气保守审慎，2-3句。",
}


class ChatAgent:
    """工具增强对话 Agent"""

    def __init__(self, model: str = None, persona: str = "综合",
                 context_blocks: list = None):
        """Args:
            context_blocks: 额外注入到系统提示的上下文块（每轮请求都会带上）。

                用途见「深度分析页 → 就这份报告提问」：把**当前那一份**分析报告
                直接注入会话，模型就不必去猜"用户说的是哪一份报告"。
                这比让模型自己调工具查报告更省事、也更不容易读错版本
                （实测 2026-10-08 就有"磁盘上的 md 落后两代"的问题）。

                ⚠️ 它每轮都进上下文，所以**必须限长**（`report_render.build_chat_context`
                已按段落设上限）。空块/空白会被丢掉，不会污染提示。
        """
        self.model = model or resolve_model("chat.tool_loop")
        self.client = DeepSeekClient(model=self.model)
        self.usage_total = {"prompt": 0, "completion": 0}
        self.tools_used: list[str] = []
        self.context_blocks = [str(b) for b in (context_blocks or [])
                               if str(b or "").strip()]
        persona_extra = PERSONAS.get(persona, "")
        self.system_prompt = SYSTEM_PROMPT + (f"\n{persona_extra}" if persona_extra else "")

    def _track_usage(self):
        """累计本轮对话的 token 用量（client.chat 每次调用后更新 last_usage）"""
        usage = getattr(self.client, "last_usage", None)
        if usage:
            self.usage_total["prompt"] += usage.get("prompt") or 0
            self.usage_total["completion"] += usage.get("completion") or 0

    # ──────────────────────────────────────────────

    def _build_messages(self, history: list[dict], user_msg: str) -> list[dict]:
        trimmed = history[-MAX_HISTORY:] if history else []
        # 时间基准每轮重算（跨天/跨时段的长会话也必须给出正确的"今天"）
        system = f"{self.system_prompt}\n{time_context()}"
        if self.context_blocks:
            # 注入上下文放在**时间之后**：它比"今天是几号"更具体，
            # 也让模型先看到通用规则、再看到"本会话聚焦的那份报告"。
            system += "\n\n" + "\n\n".join(self.context_blocks)
        return [{"role": "system", "content": system}] + trimmed + [
            {"role": "user", "content": user_msg}]

    @staticmethod
    def _summarize_result(result: dict, tool_name: str = None) -> str:
        """工具结果转字符串并截断，控制上下文膨胀（系统级读工具配额更高，见 _TOOL_RESULT_CHAR_LIMITS）"""
        try:
            text = json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(result)
        limit = _TOOL_RESULT_CHAR_LIMITS.get(tool_name, TOOL_RESULT_MAX_CHARS)
        if len(text) > limit:
            text = text[:limit] + f" ...(已截断，原长{len(text)})"
        return text

    def reply(self, user_msg: str, history: list[dict] = None):
        """
        处理一轮用户消息（事件生成器）。

        yield 事件:
          ("tool_start", 工具名, 参数dict)
          ("tool_end",   工具名, 一句话摘要)
          ("chart",      图表数据dict，见 _extract_chart)
          ("answer",     最终回答文本)
          ("error",      错误消息)
        """
        messages = self._build_messages(history or [], user_msg)
        tool_trace = []  # (工具名, 参数, 结果) 供超限汇总

        try:
            for _round in range(MAX_ROUNDS):
                message = self.client.chat(messages, tools=TOOL_SCHEMAS,
                                           temperature=TEMP_CONVERSATIONAL,
                                           purpose="chat.tool_loop")
                self._track_usage()

                if not getattr(message, "tool_calls", None):
                    answer = (message.content or "").strip() or "（模型返回空响应，请重试）"
                    yield "answer", answer
                    return

                # 记录 assistant 的工具调用消息
                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name,
                                      "arguments": tc.function.arguments}}
                        for tc in message.tool_calls
                    ],
                })

                for tc in message.tool_calls:
                    name = tc.function.name
                    self.tools_used.append(name)
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    yield "tool_start", name, args
                    result = execute_tool(name, tc.function.arguments)
                    summary = _brief_summary(result)
                    yield "tool_end", name, summary
                    # 附带图表事件（页面据此渲染 K 线 / 五维评分图）
                    chart = _extract_chart(name, result)
                    if chart:
                        yield "chart", chart
                    tool_trace.append((name, args, result))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": self._summarize_result(result, name),
                    })

            # 超过最大轮数：强制汇总
            _logger.warning("[ChatAgent] 达到最大工具轮数(%d)，强制汇总", MAX_ROUNDS)
            messages.append({"role": "user", "content":
                             "（系统提示）工具调用次数已达上限，请基于以上已获取的数据直接回答用户问题，"
                             "并说明未能完成的部分。"})
            message = self.client.chat(messages, temperature=TEMP_CONVERSATIONAL,
                                       purpose="chat.tool_loop_summary")
            self._track_usage()
            yield "answer", (message.content or "").strip() or "工具调用超限且汇总失败，请重试或缩小问题范围。"

        except Exception as e:
            _logger.exception("[ChatAgent] 处理失败")
            yield "error", f"助手处理出错：{type(e).__name__}: {e}"


def _extract_chart(tool_name: str, result: dict) -> dict | None:
    """从工具结果提取可渲染的图表数据（页面据 kind 渲染）"""
    if not isinstance(result, dict) or "error" in result:
        return None
    if tool_name == "tool_stock_technicals":
        sample = result.get("kline_sample") or {}
        if sample.get("dates") and sample.get("closes"):
            return {"kind": "kline", "title": f"{result.get('代码', '')} 收盘价走势",
                    "dates": sample["dates"], "closes": sample["closes"],
                    "ma": result.get("均线", {})}
    if tool_name == "tool_fund_flow":
        breakdown = result.get("维度明细") or {}
        if breakdown:
            return {"kind": "fund5", "title": "资金面五维评分",
                    "breakdown": breakdown, "score": result.get("五维评分")}
    return None


def split_fund5_dims(breakdown: dict) -> dict:
    """把五维资金的 breakdown 分成「可评」与「不可评」两组（页面渲染用）。

    抽成**纯函数**的原因：页面里的渲染函数无法单独测（import 页面模块就会执行整段
    Streamlit 脚本），而"不可评的维度不能画成 0 分柱"这条规则值得被测试钉住。

    ⚠️ 为什么要分两组（2026-10-10 用户报"资金面五维评分为空"）：
    数据源取不到的维度（快照源的「结构」无单型拆分）若照旧画成 0 分柱，图上是一根空柱，
    看起来既像"渲染失败"也像"该维度 0 分"，两种解读都不对；它们应单独列出并说明
    "已从总分分母剔除"。反过来，**全 0 必须说清是数据结果**（资金持续流出）而非故障。

    Returns:
        {"scored": [{"name","score","max","desc"}], "unavailable": [{"name","desc"}]}
    """
    scored, unavailable = [], []
    for name, d in (breakdown or {}).items():
        d = d or {}
        if d.get("available") is False:
            unavailable.append({"name": name, "desc": d.get("desc") or "不可评"})
        else:
            scored.append({"name": name, "score": d.get("score") or 0,
                           "max": d.get("max", 1), "desc": d.get("desc") or ""})
    return {"scored": scored, "unavailable": unavailable}


def _brief_summary(result: dict) -> str:
    """工具结果的一句话摘要（UI 进度展示用）"""
    if not isinstance(result, dict):
        return "完成"
    if "error" in result:
        return f"失败: {result['error']}"
    if "quotes" in result:
        return f"获取到 {len(result['quotes'])} 条行情"
    if "候选数" in result:
        return f"选出 {result['候选数']} 只候选"
    if "自选股" in result:
        return f"自选股共 {len(result['自选股'])} 只"
    if "五维评分" in result:
        return f"五维评分 {result['五维评分']}"
    if "总资产" in result:
        # 模拟账户（tool_paper_account）。放在 tool_portfolio_positions 之后判断没有歧义：
        # 后者用的是"总市值"字段名，两者不会撞。
        return (f"模拟账户总资产 {result['总资产']}，"
                f"持仓 {len(result.get('持仓') or [])} 只")
    if "positions" in result:
        return f"真实持仓 {len(result['positions'])} 只"
    if "市场状态" in result:
        return f"市场状态: {result['市场状态']}({result.get('总分')})"
    if "均线形态" in result:
        return f"现价 {result.get('现价')}，{result['均线形态']}"
    if "信息比率" in result or "窗口" in result:
        return f"回测完成: 总收益 {result.get('总收益%')}%"
    if "success" in result:
        return result.get("message", "完成")
    return "完成"
