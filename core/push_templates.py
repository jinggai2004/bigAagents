# -*- coding: utf-8 -*-
"""推送消息模板（2026-10-23）—— 让「发什么」变成**可编辑配置**。

为什么需要：
    此前推送正文**全部硬编码在代码里**（`format_alerts_markdown` 的 f-string、
    `build_review_summary` 的拼装、`run_risk_job` 的 body）。想改一句措辞就得
    改代码 + 重启调度器进程 —— 而"重启才生效"正是本项目反复踩的坑。
    ⇒ 把**外壳与条目格式**抽成模板，放进数据库，在「系统配置」页直接编辑，
    **改完即时生效、不用重启**。

模板语法（刻意不用 `str.format`）：
    占位符写成 `{{变量名}}`，用**正则替换**实现。
    ⚠️ 为什么不用 `.format()`：模板里一旦出现字面花括号（JSON 示例、代码片段）
    就会炸 —— 本项目在 LLM 提示词上已经踩过同一个坑。
    ⚠️ 语义：
      · **已知但没给值**的变量 → 渲染成 `—`（不留空、不报错）
      · **不认识的**占位符 → **原样保留**（让用户一眼看出自己拼错了变量名）

口径纪律：
    · **数字不经模板改写**。模板只负责排版，变量值由代码算好后传入
      （项目铁律：决策数字不经 LLM，同理不经模板二次加工）。
    · **坏配置退回默认**：模板读取失败 / 正文为空 ⇒ 退回内置默认并记日志，
      绝不能因为配置坏了就让推送整个挂掉。
    · **默认文案必须与改造前逐字一致** —— 否则用户会发现"改了配置反而变了消息"。

存储位置选**数据库**（表 `push_templates`）而不是 `config/*.json`：
    ① 与 `scheduled_jobs`（用户可编辑的自定义定时任务）一致；
    ② 调度器长驻进程与 Web 页面**天然读到同一份**，不用管文件 mtime 缓存；
    ③ 用户改文案不该在代码库里产生 diff。
"""

from __future__ import annotations

import logging
import re

_logger = logging.getLogger(__name__)

# ── 模板键 ──────────────────────────────────────────────────────────────
TPL_ALERT = "alert"                      # 盘中异动 · 外壳
TPL_ALERT_ITEM = "alert_item"            # 盘中异动 · 每一条
TPL_REVIEW = "review_summary"            # 每日复盘摘要 · 整篇
TPL_RISK = "position_risk"               # 持仓风控 · 外壳
TPL_RISK_ITEM = "position_risk_item"     # 持仓风控 · 每一条
TPL_SIGNAL = "signal_conclusion"         # 买入信号结论 · 外壳
TPL_SIGNAL_ITEM = "signal_item"          # 买入信号结论 · 每一条

# ⭐ 每个模板的**合法变量清单**（单一事实来源）。
#   它有两个用途：① 配置页展示"可用变量"；② 渲染时区分两种情况 ——
#   **在清单里但这次没给值** → 渲染成 `—`（不留空）；
#   **不在清单里**（用户自己拼的）→ **原样保留**（让用户一眼看出拼错了）。
KNOWN_VARS: dict = {
    TPL_ALERT: {"title", "date", "time", "count", "items", "footer"},
    TPL_ALERT_ITEM: {"icon", "name", "code", "message", "price", "quote_time",
                     "relation", "note"},
    TPL_REVIEW: {"title", "date", "time", "lede", "picks", "analyses", "positions",
                 "watch", "risks", "integrity", "report_path", "footer"},
    TPL_RISK: {"title", "date", "time", "count", "items", "note", "footer"},
    TPL_RISK_ITEM: {"icon", "type", "code", "name", "detail"},
    TPL_SIGNAL: {"title", "date", "time", "items", "footer"},
    TPL_SIGNAL_ITEM: {"icon", "name", "code", "detail", "direction", "entry",
                      "target", "stop", "position_size", "data_through"},
}

# 模板说明（配置页展示用）—— **由 KNOWN_VARS 生成**，避免两处维护。
TEMPLATE_LABELS: dict = {
    TPL_ALERT: "盘中异动 · 外壳",
    TPL_ALERT_ITEM: "盘中异动 · 每一条",
    TPL_REVIEW: "每日复盘摘要 · 整篇",
    TPL_RISK: "持仓风控 · 外壳",
    TPL_RISK_ITEM: "持仓风控 · 每一条",
    TPL_SIGNAL: "买入信号结论 · 外壳",
    TPL_SIGNAL_ITEM: "买入信号结论 · 每一条",
}
TEMPLATE_META: dict = {
    key: (TEMPLATE_LABELS[key], "变量：" + " ".join(sorted(vars_)))
    for key, vars_ in KNOWN_VARS.items()
}

# ── 内置默认模板 ────────────────────────────────────────────────────────
# ⚠️ 默认文案的渲染结果必须与**改造前逐字一致**（否则用户会发现消息变了）。
DEFAULT_TEMPLATES: dict = {
    TPL_ALERT: (
        "# {{title}}（{{count}} 条）\n\n"
        "{{items}}\n\n"
        "{{footer}}"
    ),
    TPL_ALERT_ITEM: (
        "- {{icon}} **{{name}}({{code}})** {{message}} · 现价 {{price}} · {{quote_time}}"
        "{% if relation %}　【{{relation}}】{% endif %}"
        "{% if note %}\n　　└ {{note}}{% endif %}"
    ),
    TPL_REVIEW: (
        "# {{title}} {{date}}\n\n"
        "> 生成 {{time}}｜全文：`{{report_path}}`\n\n"
        "**结论**：{{lede}}\n\n"
        "**今日选股**\n{{picks}}"
        "**深度分析**：{{analyses}}\n"
        "**持仓**：{{positions}}\n\n"
        "**明日待触发**\n{{watch}}\n\n"
        "**风险**\n{{risks}}\n\n"
        "**数据完整性**：{{integrity}}\n\n"
        "{{footer}}"
    ),
    TPL_RISK: (
        "## {{title}} {{date}} {{time}}\n\n"
        "持仓 {{count}} 只\n\n"
        "{{items}}\n\n"
        "{{note}}\n\n"
        "{{footer}}"
    ),
    TPL_RISK_ITEM: (
        "- {{icon}} **{{type}}**{% if code %} {{code}} {{name}}{% endif %}：{{detail}}"
    ),
    TPL_SIGNAL: (
        "## {{title}} · {{date}}\n\n"
        "{{items}}\n\n"
        "{{footer}}"
    ),
    TPL_SIGNAL_ITEM: (
        "- {{icon}} **{{name}}({{code}})** ｜ {{detail}}"
        "{% if direction %} ｜ 方向 {{direction}}{% endif %}"
        "{% if entry %} ｜ 入场 {{entry}}{% endif %}"
        "{% if target %} ｜ 目标 {{target}}{% endif %}"
        "{% if stop %} ｜ 止损 {{stop}}{% endif %}"
        "{% if position_size %} ｜ 仓位 {{position_size}}{% endif %}"
        "{% if data_through %} ｜ 数据截至 {{data_through}}{% endif %}"
    ),
}

# 常用脚注文案（也可被模板覆盖）
FOOTERS: dict = {
    TPL_ALERT: "*程序化监控，仅供参考，不构成投资建议。*",
    TPL_REVIEW: "*程序化数据分析，仅供参考，不构成投资建议。*",
    TPL_RISK: ("> 止损/止盈由 ATR 与行业倍数算出，仅供参考，不构成投资建议。\n"
               "> 止损建议不会自动生效，请在「持仓管理」页确认后应用。"),
    TPL_SIGNAL: ("> 这些票在盘中达到买入条件（涨幅 + 相对放量），已做深度分析，"
                 "结论为**可执行**。\n> 仅供参考，不构成投资建议。"),
}

# ── 渲染 ────────────────────────────────────────────────────────────────
_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
# 条件块：{% if 变量 %}…{% endif %}（只支持"变量非空"这一种，避免引入模板引擎）
_IF_RE = re.compile(r"\{%\s*if\s+([A-Za-z_][A-Za-z0-9_]*)\s*%\}(.*?)\{%\s*endif\s*%\}",
                     re.S)


def _value_of(name: str, context: dict, missing: str = "—") -> str:
    if name not in context:
        return missing
    val = context.get(name)
    if val is None or val == "":
        return missing
    return str(val)


def render(template: str, context: dict, known: set = None, missing: str = "—") -> str:
    """渲染模板。

    · 条件块 `{% if 变量 %}…{% endif %}`：变量为空则整块去掉；
    · `{{变量}}`：
        - 在 `known` 清单里但这次没给值 → `missing`（默认 `—`）；
        - **不在** `known` 清单里 → **原样保留**（暴露用户拼错的变量名）；
    · `known=None` 时不做区分（所有变量都当已知）。
    """
    ctx = context or {}

    def _if_sub(m):
        body = m.group(2)
        raw = ctx.get(m.group(1))
        if raw is None or raw == "":
            return ""
        return body

    text = _IF_RE.sub(_if_sub, template or "")

    def _var_sub(m):
        name = m.group(1)
        if known is not None and name not in known:
            return m.group(0)          # 不认识的变量 ⇒ 原样保留
        return _value_of(name, ctx, missing)

    return _VAR_RE.sub(_var_sub, text)


# ── 存取 ────────────────────────────────────────────────────────────────
def load_templates() -> dict:
    """全部模板（用户覆盖优先）。**读取失败一律退回内置默认**（不抛异常）。"""
    out = dict(DEFAULT_TEMPLATES)
    try:
        from db import store
        for key, body in store.list_push_templates().items():
            if key in out and body and body.strip():
                out[key] = body
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[推送模板] 读取用户模板失败（退回默认）: %s", e)
    return out


def get_template(key: str) -> str:
    """单个模板正文（用户覆盖优先；空/损坏 ⇒ 默认）。"""
    default = DEFAULT_TEMPLATES.get(key, "")
    try:
        from db import store
        body = store.list_push_templates().get(key)
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[推送模板] 读取 %s 失败（用默认）: %s", key, e)
        return default
    if body and body.strip():
        return body
    return default


def save_template(key: str, body: str) -> bool:
    """保存用户模板。空正文 ⇒ 视为"恢复默认"（删记录）。"""
    if key not in DEFAULT_TEMPLATES:
        return False
    try:
        from db import store
        if not body or not body.strip():
            return store.delete_push_template(key)
        return store.set_push_template(key, body)
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[推送模板] 保存 %s 失败: %s", key, e)
        return False


def reset_template(key: str) -> bool:
    """恢复内置默认。"""
    try:
        from db import store
        return store.delete_push_template(key)
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[推送模板] 重置 %s 失败: %s", key, e)
        return False


def render_with(key: str, context: dict, missing: str = "—") -> str:
    """按 key 取模板并渲染（**推送侧的统一入口**，唯一实现）。"""
    return render(get_template(key), context, known=KNOWN_VARS.get(key), missing=missing)


def user_overrides() -> dict:
    """用户改过的模板（配置页用来标"已改"）。"""
    try:
        from db import store
        return store.list_push_templates()
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[推送模板] 读取覆盖列表失败: %s", e)
        return {}


# ── 预览用示例上下文（配置页"预览效果"按钮用）──────────────────────────
SAMPLE_CONTEXT: dict = {
    TPL_ALERT: {
        "title": "bigAagents 盘中异动 14:30", "date": "2026-10-23", "time": "14:30",
        "count": "2",
        "items": ("- 🔴 **贵州茅台(600519)** 涨停 +10.00% · 现价 1329.04 · 14:30:00"
                  "　【持仓·自选】\n"
                  "　　└ 持仓 100 股，止损 1245.00（现价差 -6.3%）\n"
                  "- 🟠 **宁德时代(300750)** 涨幅 +7.80% · 现价 236.96 · 14:30:00　【自选】"),
        "footer": FOOTERS[TPL_ALERT],
    },
    TPL_REVIEW: {
        "title": "A股每日复盘摘要", "date": "2026-10-23", "time": "15:52",
        "lede": "今日三大指数集体收涨，成长风格显著占优；资金主动流向成长方向。",
        "picks": "177 只 → TOP5：恒瑞医药(600276) 92.9｜工商银行(601398) 82.9\n",
        "analyses": "南京港(002040) 驳回",
        "positions": "无",
        "watch": ("- 中国神华(601088，等待)：股价回踩MA20（48.73元）附近并缩量企稳\n"
                  "- 迈瑞医疗(300760，等待)：股价缩量回调至MA20（162.18元）附近\n"),
        "risks": "- 公路铁路运输 连续两日资金净流出（-0.55亿 → -0.10亿）\n",
        "integrity": "各项齐全",
        "report_path": "output/daily_reports/review_20261023.md",
        "footer": FOOTERS[TPL_REVIEW],
    },
    TPL_RISK: {
        "title": "🛡️ 持仓风控巡检", "date": "2026-10-23", "time": "14:50",
        "count": "2",
        "items": ("- 🔴 **止损触发** 600276 恒瑞医药：现价 43.10 vs 设定 44.00（-2.05%）\n"
                  "- 📐 601088 中国神华 止损 48.73 → 49.10（追踪上移）"),
        "note": ("> 集中度/流动性口径：占比 = 当前市值 ÷ 总资产；"
                 "流动性 = 持仓市值 ÷ 该股当日成交额（天）。\n"
                 "行业与成交额取自 2026-10-23 的估值截面快照。"),
        "footer": FOOTERS[TPL_RISK],
    },
    TPL_SIGNAL: {
        "title": "🟢 买入信号已确认（深度分析结论可行）", "date": "2026-10-23",
        "time": "18:40", "items": "", "footer": FOOTERS[TPL_SIGNAL],
    },
}
SAMPLE_CONTEXT[TPL_SIGNAL]["items"] = (
    "- 🟢 **贵州茅台(600519)** ｜ 涨幅 +5.20%、成交额 2.1×20日均额（均额截至 2026-10-16）"
    " ｜ 方向 看多 ｜ 入场 1320.0 ｜ 目标 1400.0 ｜ 止损 1280.0 ｜ 仓位 20%"
    " ｜ 数据截至 2026-10-23"
)

# 条目级模板的示例上下文
SAMPLE_ITEM_CONTEXT: dict = {
    TPL_ALERT_ITEM: {
        "icon": "🔴", "name": "贵州茅台", "code": "600519",
        "message": "涨停 +10.00%", "price": "1329.04", "quote_time": "14:30:00",
        "relation": "持仓·自选", "note": "持仓 100 股，止损 1245.00（现价差 -6.3%）",
    },
    TPL_RISK_ITEM: {
        "icon": "🔴", "type": "止损触发", "code": "600276", "name": "恒瑞医药",
        "detail": "现价 43.10 vs 设定 44.00（-2.05%）",
    },
    TPL_SIGNAL_ITEM: {
        "icon": "🟢", "name": "贵州茅台", "code": "600519",
        "detail": "涨幅 +5.20%、成交额 2.1×20日均额（均额截至 2026-10-16）",
        "direction": "看多", "entry": "1320.0", "target": "1400.0", "stop": "1280.0",
        "position_size": "20%", "data_through": "2026-10-23",
    },
}
