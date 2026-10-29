"""把深度分析结果（payload）渲染成可读 Markdown —— **产物的唯一出口**。

为什么要有这个模块（2026-10-09）：
分析结果原本有**两份并行产物** —— `output/ui_runs/analysis_{code}.json`（UI 读它）
与 `output/xxx深度分析报告.md`（用临时脚本手工渲染）。手工那份必然漂移：
实测 2026-10-08 的 Markdown 就**落后了两代**（它渲染的是 22:06 那次运行，
而 UI 里显示的是 22:38 那次），于是"用户看到的报告"和"我复核的报告"根本不是同一份，
两边结论（批准 / 需修改 / 驳回）还不一样。

所以：**json 是唯一事实来源，Markdown 由本函数按需生成**（UI 上就是"导出"按钮）。
不再有"顺手多写一个 md 文件"这种会漂移的路径。
"""

import json
from datetime import datetime

_SECTIONS = (
    ("技术面分析", "tech_report"),
    ("基本面分析", "fundamental_report"),
    ("舆情分析", "news_report"),
)
# 辩论之后的部分：按分析链的**实际执行顺序**排（研究汇总 → 交易计划 → 风控 → 终审），
# 这样读者能顺着"论点 → 汇总 → 计划 → 风控 → 决策"走一遍，而不是看到风控出现在计划之前。
_TAIL_SECTIONS = (
    ("研究汇总", "research_summary"),
    ("交易计划", "trade_plan"),
    ("风控审核", "risk_assessment"),
    ("最终决策", "final_decision"),
)


def _fmt_trade_plan(plan: dict) -> str:
    """交易计划**不能只丢一段 JSON** 给读者看。

    ⚠️ 实测（2026-10-09）：报告里 `direction="看空"`、目标价 44.96 在下方、
    止损 53.80 在上方 —— 这是**融券卖出**的结构，而 A股 个人投资者无法执行。
    读者只看到一段 JSON，根本不会意识到"这份计划不能照做"。

    现在按**三档建议形态**渲染（用户 2026-10-09 定的设计）：
      看多 → 可执行（给完整投资建议）；看空 → 驳回（不给做空价位，给入场条件）；
      观望 → 等待（同样给入场条件）。
    """
    plan = dict(plan or {})
    if not plan:
        return "（本次未产出这一段）"
    direction = str(plan.get("direction") or "未提供")
    status = str(plan.get("advice_status") or "")
    if not status:                       # 旧产物没有这个字段 → 由方向推断
        status = {"看空": "驳回"}.get(direction, "可执行" if "看多" in direction else "等待")
    icon = {"可执行": "🟢", "驳回": "🔴", "等待": "⚪"}.get(status, "⚪")

    lines = [f"- **方向：{direction}**　｜　{icon} **建议：{status}**"]
    if plan.get("advice"):
        lines.append(f"- 投资建议：{plan['advice']}")
    if status == "可执行":
        if plan.get("position_size") is not None:
            lines.append(f"- 仓位：{plan['position_size']}")
        if plan.get("entry_range"):
            lines.append(f"- 入场区间：{plan['entry_range']}"
                         "（风险敞口按**最不利**一端算，做多即最高入场价）")
        if plan.get("target_price") is not None:
            lines.append(f"- 目标价：{plan['target_price']}")
        if plan.get("stop_loss") is not None:
            lines.append(f"- 止损价：{plan['stop_loss']}")
        if plan.get("holding_period"):
            lines.append(f"- 持有周期：{plan['holding_period']}")
    else:
        lines.append(f"- **本次不建立仓位**（{status}）。")
        if status == "驳回":
            lines.append("- 🔴 **投资建议已驳回**：判断为看空，**不建议买入**。"
                         "⚠️ 不等同于「可以做空」 —— A股 个人投资者没有融券通路"
                         "（需信用账户 + 50万门槛 + 标的在两融名单且有券源），"
                         "「看空」只能落成「不参与 / 已持有则减仓」。")
        else:
            lines.append("- ⚪ **等待**：暂不具备入场条件，等下面这些信号出现再评估。")
        conds = plan.get("entry_conditions")
        if isinstance(conds, (list, tuple)) and conds:
            lines.append("- **入场条件（满足后才重新考虑买入）**：")
            lines += [f"    {i}. {c}" for i, c in enumerate(conds, 1)]
        elif "entry_conditions" in plan:
            lines.append("- ⚠️ 未给出入场条件 —— 驳回/等待必须说明「什么信号出现才重新考虑入场」，"
                         "否则这条建议无法执行。")
    watch = plan.get("watch_levels")
    if isinstance(watch, dict) and watch:
        lines.append("- **观察位（仅观察，不构成交易指令）**：")
        lines += [f"    - {k}：{v}" for k, v in watch.items()]
    elif status != "可执行":
        # 旧产物（规范化之前生成的）没有 watch_levels，却留着入场/目标/止损三个字段。
        # 直接丢掉它们等于隐藏信息（压力/支撑仍有价值），所以降级成"参考位"显示，
        # 并明确标注它们**不是**可执行指令。
        legacy = [(k, plan[k]) for k in ("entry_range", "target_price", "stop_loss")
                  if plan.get(k) is not None]
        if legacy:
            lines.append("- **原计划价位（仅供参考，不构成交易指令 —— 本次不建议建仓）**：")
            lines += [f"    - {k}：{v}" for k, v in legacy]
    if plan.get("reasoning"):
        lines += ["", "**理由**", "", str(plan["reasoning"])]
    # 未列出的字段（将来新增的）也不能丢 —— 同"渲染白名单"那类坑
    known = {"direction", "advice_status", "advice", "market_view", "position_size",
             "entry_range", "target_price", "stop_loss", "holding_period",
             "entry_conditions", "watch_levels", "reasoning"}
    extra = {k: v for k, v in plan.items() if k not in known}
    if extra:
        lines += ["", "**其他字段**", "",
                  "```json",
                  json.dumps(extra, ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines)


def trade_plan_markdown(plan: dict) -> str:
    """交易计划的可读 Markdown —— UI 与导出**共用同一套语义**。

    单独暴露成公开函数是为了避免 UI 自己拼一遍：UI 一旦自己拼，
    "方向在最前 + 观望的含义提示"这两条就会随时间漂移（同一份计划
    在页面上是一套说法、在导出的 md 里又是另一套）。
    """
    return _fmt_trade_plan(plan)


def _fmt_block(title: str, key: str, payload: dict, lines: list) -> None:
    value = payload.get(key)
    if key == "trade_plan" and isinstance(value, dict):
        body = _fmt_trade_plan(value)
    elif isinstance(value, (dict, list)):
        body = "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"
    else:
        body = str(value or "").strip() or "（本次未产出这一段）"
    lines += [f"## {title}", "", body, ""]


def render_markdown(payload: dict, *, generated_at: str = None) -> str:
    """payload → Markdown 文本（纯函数，便于测试与按需导出）。

    Args:
        payload: `core.analysis_history.snapshot_from_state` 形状的结果快照
        generated_at: 生成时间戳；None 取当前时间

    Returns:
        完整 Markdown 文本（以换行结尾）
    """
    p = dict(payload or {})
    name = p.get("stock_name") or ""
    code = p.get("stock_code") or ""
    gen = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# {name}（{code}）深度分析报告",
        "",
        f"> 导出时间：{gen}　｜　现价：{p.get('current_price')}"
        f"　｜　分析日：{p.get('analysis_date')}　｜　数据截止：{p.get('data_through')}"
        "　｜　数据来源：本地分析记录（`analyses` 表）",
        "",
        "## 0 数据可用性",
        "",
    ]
    avail = p.get("data_availability") or {}
    lines.append("｜".join(f"{k} {'✅' if v else '❌'}" for k, v in avail.items())
                 or "（无记录）")

    lines += ["", "## 1 可核实性审计", ""]
    warnings = list(p.get("audit_warnings") or [])
    others = [e for e in (p.get("errors") or []) if e not in warnings]
    if warnings:
        lines.append("**审计规则在本报告中发现的可疑项**（这些内容在输入里找不到出处）：")
        lines.append("")
        lines += [f"- ⚠️ {w}" for w in warnings]
    else:
        # 零告警**不是**"已核实"：旧版本进程产出的报告压根没跑审计（实测 2026-10-08 id=11）。
        lines.append("本次未产生审计告警。⚠️ **这不等于内容已核实** —— "
                     "审计只覆盖年份/来源/依据/数值四类线索，"
                     "且若该次分析运行时未启用审计（如由旧版本进程产出）同样会是空。")
    if others:
        lines += ["", "**其他记录**（非审计项）：", ""]
        lines += [f"- {e}" for e in others]
    fatal = p.get("fatal_errors") or []
    if fatal:
        lines += ["", f"**致命错误**：{fatal}"]
    lines.append("")

    for title, key in _SECTIONS:
        _fmt_block(title, key, p, lines)

    lines += ["## 多空辩论", ""]
    bulls = list(p.get("bull_history") or [])
    bears = list(p.get("bear_history") or [])
    if not bulls and not bears:
        lines += ["（本次未产出辩论内容）", ""]
    for i, txt in enumerate(bulls, 1):
        lines += [f"### 多头 R{i}", "", str(txt), ""]
    for i, txt in enumerate(bears, 1):
        lines += [f"### 空头 R{i}", "", str(txt), ""]

    for title, key in _TAIL_SECTIONS:
        _fmt_block(title, key, p, lines)

    return "\n".join(lines).rstrip() + "\n"


def suggest_filename(payload: dict) -> str:
    """导出用的建议文件名（带代码与数据截止日，便于留档区分）。"""
    code = str((payload or {}).get("stock_code") or "unknown")
    through = str((payload or {}).get("data_through") or "").replace("-", "") or "nodate"
    return f"{code}_深度分析_{through}.md"


# ================================================================
# 给对话 Agent 的上下文块（「深度分析页 → 就这份报告提问」，2026-10-09）
# ================================================================
# 为什么用"注入上下文"而不是"让模型自己调工具查报告"：
# ① 模型不必猜"用户说的是哪一份"（同一只股票同一天可能有 4 条记录）；
# ② 不会读到错版本 —— 注入的就是**页面上正在看的这一份**；
# ③ 不消耗工具轮次（工具轮次有限，且每次调用都要重新序列化一遍）。
# 工具仍然保留：需要"最新行情/新闻"时该调还是调，但要求它**区分**报告口径与当前数据。
_CHAT_SECTIONS = (
    ("技术面", "tech_report", 1500, False),
    ("基本面", "fundamental_report", 1500, False),
    ("舆情", "news_report", 1200, False),
    ("多头辩论", "bull_history", 1500, True),
    ("空头辩论", "bear_history", 1500, True),
    ("研究汇总", "research_summary", 2500, False),
    ("交易计划", "trade_plan", 900, False),
    ("风控审核", "risk_assessment", 1500, False),
    ("最终决策", "final_decision", 500, False),
)


def _chat_section_text(payload: dict, key: str, cap: int, is_list: bool) -> str:
    if is_list:
        text = "\n\n".join(str(x) for x in (payload.get(key) or []))
    elif key == "trade_plan" and isinstance(payload.get(key), dict):
        # 交易计划走专用渲染：读者（和模型）必须先看到「方向」，再看到价位。
        # 只丢 JSON 的话，模型会照着"入场/目标/止损"讲成一个**可执行计划**，
        # 而这份计划可能是做空结构（A股 个人无法执行）。
        text = _fmt_trade_plan(payload.get(key))
    else:
        value = payload.get(key)
        text = (json.dumps(value, ensure_ascii=False, indent=2)
                if isinstance(value, (dict, list)) else str(value or ""))
    text = text.strip()
    if not text:
        return "（本次未产出这一段）"
    return text[:cap] + f"…（本节共 {len(text)} 字，已截断）" if len(text) > cap else text


def build_chat_context(payload: dict, *, analysis_id=None) -> str:
    """把一份分析报告变成「给对话 Agent 的系统上下文块」。

    包含三部分：报告元信息 / 回答这类问题时的硬性要求 / 报告原文（按段落限长）。
    ⚠️ 每轮请求都会带上，所以**必须有上限**；此处各段落上限合计 ≈ 1.2 万字。
    """
    p = dict(payload or {})
    warn = list(p.get("audit_warnings") or [])
    avail = p.get("data_availability") or {}
    head = [
        "【本会话聚焦的一份深度分析报告】",
        f"- 股票：{p.get('stock_name') or ''}({p.get('stock_code') or ''})"
        f"　分析日期：{p.get('analysis_date') or '—'}"
        f"　数据截止：{p.get('data_through') or '—'}"
        + (f"　本地记录 id={analysis_id}" if analysis_id else ""),
        f"- 现价（分析时点）：{p.get('current_price')}",
        "- 数据可用性：" + ("｜".join(f"{k}{'✅' if v else '❌'}"
                                for k, v in avail.items()) or "（无记录）"),
        "",
        "【回答这类问题时的硬性要求】",
        "1. 用户问的是「**这份报告为什么这么说**」，必须基于下面的报告原文回答；"
        "**不要**用行情/新闻工具重新推一遍来替代引用。"
        "报告原文已在下文完整给出，**不需要**再调 `tool_analysis_report` 去取"
        "（同一只股票可能有多条历史记录，再取一次容易取到不是这一份）。"
        "需要最新行情/新闻时可以调工具，但要**明确区分**"
        "「报告里写的（截至上述数据截止日）」与「当前数据」。",
        "2. 引用必须标明段落名（如「研究汇总」「风控审核」），"
        "不要把自己的推断说成报告内容。",
        "3. 报告自身的问题必须一并说明：" + (
            "；".join(f"⚠️ {w}" for w in warn) if warn else
            "本次未产生审计告警，但**不等于已核实**"
            "（审计只覆盖年份/来源/依据/数值四类线索，"
            "且由旧版本进程产出的报告压根没跑审计）。"),
        "4. 报告里没有的内容，直接说「报告里没写」，不要替它补。",
        "5. 报告是历史产物：其数据截止日可能早于今天，回答里要带上日期口径。",
        "6. **交易计划分三档，别把「看空」讲成「做空」**："
        "`看多` = 有可执行的投资建议（入场/目标/止损/仓位）；"
        "`看空` = **投资建议已被驳回**（不建议买入），并给出「入场条件」；"
        "`观望` = 等待，同样给「入场条件」。"
        "A股 个人投资者没有融券通路（需信用账户 + 50万门槛 + 标的在两融名单且有券源），"
        "**不要**建议用户做空/融券，也不要按「入场在上/目标在下/止损在上」的方式解读。",
        "",
        "【报告原文】",
    ]
    body = []
    for title, key, cap, is_list in _CHAT_SECTIONS:
        body += [f"## {title}", "", _chat_section_text(p, key, cap, is_list), ""]
    fatal = p.get("fatal_errors") or []
    if fatal:
        body += [f"> ⚠️ 本次分析的致命错误：{fatal}", ""]
    return "\n".join(head + body).rstrip() + "\n"
