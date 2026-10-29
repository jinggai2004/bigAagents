# -*- coding: utf-8 -*-
"""财务派生指标表测试（core/derived_fin.py）

背景（2026-10-23/24）：同一模型在**隔离算术题**上 8/8 全对，却在长报告里把
`296.56亿元（790.1 - 293.54）`（应得 496.56，错 40%）、
`28.44%（47.97/177.47）`（应得 27.03%）算错。修法不是换模型，而是**别让它算**。

本文件守三件事：
1. 派生逻辑本身（累计→单季、比率、同比、缺数据不猜）；
2. **渲染出的每个算式都必须能被 `audit_arithmetic` 判为成立** ——
   否则程序给了"正确"的数，模型照抄后反被自己的审计判错，比不给更糟；
3. 报告期格式（带横线 vs 8 位）这类"静默算不出"的坑。
"""

from core.derived_fin import (DERIVED_USAGE_RULES, build, render)
from core.report_audit import audit_arithmetic


def _fin(**over):
    """一份结构真实的 `tool_financials` 输出（数字取 600941 的那组事故值）。"""
    base = {
        "代码": "600941",
        "数据截止": "2026-09-30",
        "报告期": ["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"],
        "单位": "亿元",
        "口径": "累计（年初至今）",
        "利润表(亿元)": {
            "营业收入": [7901.0, 5100.0, 2500.0, 9800.0, 7200.0],
            # 790.10 - 293.54 = 496.56 —— 事故报告写的是 296.56
            "净利润": [790.10, 293.54, 120.00, 560.00, 700.00],
            "归母净利润": [780.00, 290.00, 118.00, 555.00, 690.00],
            "营业利润": [1050.00, 400.00, 160.00, 730.00, 930.00],
            "研发费用": [30.0, 19.0, 9.0, 38.0, 27.0],
        },
        "资产负债表(亿元)": {
            "资产总计": [21000.0, 20500.0, 20000.0, 19500.0, 19000.0],
            "负债合计": [7000.0, 6800.0, 6600.0, 6400.0, 6200.0],
            "归母权益": [13500.0, 13200.0, 13000.0, 12800.0, 12600.0],
        },
        "现金流量表(亿元)": {"经营活动净额": [1500.0, 900.0, 400.0, 1700.0, 1200.0]},
        "同比%": {"营业收入": 9.74, "净利润": 12.87, "归母净利润": 13.04},
        "自算比率": {"资产负债率%": 33.33, "净利率%": 10.0},
    }
    base.update(over)
    return base


def _by_name(items, prefix):
    return [x for x in items if x["name"].startswith(prefix)]


# ---- 单季：累计相减（也就是那次 40% 错误的正解）----

def test_single_quarter_subtracts_cumulative_values():
    items = build(_fin())
    q3 = [x for x in _by_name(items, "净利润(单季") if "2026-09-30" in x["name"]]
    assert q3, [x["name"] for x in items]
    assert q3[0]["value"] == 496.56
    assert q3[0]["formula"] == "790.10 - 293.54"


def test_q1_cumulative_is_itself_single_quarter_and_has_no_formula():
    fin = _fin(**{"报告期": ["2026-03-31", "2025-12-31", "2025-09-30"]})
    fin["利润表(亿元)"]["净利润"] = [120.00, 560.00, 700.00]
    items = build(fin)
    q1 = [x for x in _by_name(items, "净利润(单季") if "2026-03-31" in x["name"]]
    assert q1 and q1[0]["value"] == 120.00
    assert q1[0]["formula"] is None          # 定义式减法，不给算式（给了也无可核对）


def test_single_quarter_skips_when_previous_quarter_missing():
    """窗口里没有上一季 ⇒ **不产出该期单季**，绝不拿相邻期顶替。

    注意 Q1 是例外：Q1 累计本身就是单季（定义），不需要上一季，
    所以 2026-03-31 那条应当仍然存在。
    """
    fin = _fin(**{"报告期": ["2026-09-30", "2026-03-31"]})
    fin["利润表(亿元)"]["净利润"] = [790.10, 120.00]
    items = build(fin)
    assert not [x for x in _by_name(items, "净利润(单季") if "2026-09-30" in x["name"]]
    q1 = [x for x in _by_name(items, "净利润(单季") if "2026-03-31" in x["name"]]
    assert q1 and q1[0]["value"] == 120.00


def test_period_with_hyphen_is_recognised():
    """`报告期` 是带横线的 `2026-09-30`；首版直接取 `pk[4:8]` 得到 `-09-` ⇒ 全空表。"""
    assert build(_fin()), "带横线的报告期必须能算出派生指标"
    fin = _fin(**{"报告期": ["20260930", "20260630"]})
    assert _by_name(build(fin), "净利润(单季")


# ---- 比率：分子分母同口径、缺分母不猜 ----

def test_ratios_have_verifiable_formula():
    items = build(_fin())
    net_margin = _by_name(items, "净利率(")[0]
    assert net_margin["value"] == 10.0
    assert net_margin["formula"] == "790.10 / 7901.00 × 100"


def test_roe_uses_same_scope_numerator_and_denominator():
    """ROE 分子必须是归母净利润（与"归母权益"同口径）。

    `tool_financials` 的 `自算比率` 里那条用的是净利润（含少数股东），
    这里刻意不复刻那个口径混搭 —— 780 / 13500 = 5.78%，而不是 790.1/13500 = 5.85%。
    """
    items = build(_fin())
    roe = [x for x in items if x["name"].startswith("ROE")]
    assert len(roe) == 1
    assert roe[0]["value"] == 5.78
    assert "780.00" in roe[0]["formula"]


def test_ratio_skipped_when_denominator_missing_or_zero():
    fin = _fin()
    fin["利润表(亿元)"]["营业收入"] = [0.0, 5100.0, 2500.0, 9800.0, 7200.0]
    names = [x["name"] for x in build(fin)]
    assert not any(n.startswith("净利率(") for n in names), names


# ---- 同比：口径一致才给算式 ----

def test_yoy_formula_given_when_consistent():
    items = build(_fin())
    yoy = [x for x in items if x["name"] == "净利润同比"][0]
    assert yoy["value"] == 12.87
    assert "(790.10 - 700.00) ÷ 700.00 × 100" == yoy["formula"]


def test_yoy_formula_dropped_when_source_uses_other_scope():
    """源自带的同比与"累计直算"不一致（扣非/追溯调整）⇒ 不给算式，避免按直算复核。"""
    fin = _fin(**{"同比%": {"净利润": 30.0}})
    yoy = [x for x in build(fin) if x["name"] == "净利润同比"][0]
    assert yoy["value"] == 30.0
    assert yoy["formula"] is None


def test_yoy_kept_without_formula_when_base_period_outside_window():
    """`periods=4` 时常缺"去年同报告期" ⇒ 仍要给值，只是不给算式。

    首版在这里直接 `continue`，于是"同比"整类从表里消失 ——
    而同比是报告里被引用最多的一类数字，等于把最需要的项静默丢了。
    """
    fin = _fin(**{"报告期": ["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31"]})
    for k in ("净利润", "归母净利润", "营业利润"):
        fin["利润表(亿元)"][k] = fin["利润表(亿元)"][k][:4]
    fin["资产负债表(亿元)"] = {k: v[:4] for k, v in fin["资产负债表(亿元)"].items()}
    fin["现金流量表(亿元)"] = {k: v[:4] for k, v in fin["现金流量表(亿元)"].items()}
    yoy = [x for x in build(fin) if x["name"] == "净利润同比"]
    assert yoy and yoy[0]["value"] == 12.87
    assert yoy[0]["formula"] is None


# ---- 结构性守护：渲染出来的算式必须自己站得住 ----

def test_every_rendered_formula_passes_own_arithmetic_audit():
    """把每条 `值（算式）` 拼成报告句式，`audit_arithmetic` 必须零告警。

    这是本模块最重要的不变量：程序给的算式若被自家审计判错，
    模型照抄就会被标成"算式错误" —— 比不提供派生表更糟。
    """
    items = build(_fin())
    assert len(items) >= 10
    bad = []
    for it in items:
        if not it["formula"]:
            continue
        val = f"{it['value']:.2f}".rstrip("0").rstrip(".")
        unit = it["unit"]
        claim = f"{val}{unit}" if unit == "%" else f"{val}{unit}"
        sentence = f"该指标为 {claim}（{it['formula']}）。"
        warns = audit_arithmetic(sentence)
        if warns:
            bad.append((it["name"], sentence, warns))
    assert bad == [], bad


def test_render_is_empty_for_empty_items():
    assert render([]) == ""
    assert build({}) == []
    assert build({"error": "未取到财务报表"}) == []
    assert build(None) == []


def test_render_keeps_full_precision_for_large_values():
    """不能用 `:g`（只留 6 位有效数字）：银行资产 114609.40 会被写成 114609。"""
    fin = _fin()
    fin["资产负债表(亿元)"]["资产总计"] = [114609.40, 110000.0, 108000.0, 106000.0, 104000.0]
    text = render(build(fin))
    assert "114609.4" in text, text


def test_render_percent_has_no_space_before_sign():
    """`10%` 不能渲染成 `10 %`：`audit_arithmetic` 的除法分支要求 `claim%` 紧贴。"""
    text = render(build(_fin()))
    assert "10%" in text
    assert "10 %" not in text


def test_usage_rules_cover_pe_and_missing_indicators():
    assert "PE" in DERIVED_USAGE_RULES
    assert "数据缺失" in DERIVED_USAGE_RULES
    # 规则块会被拼进 f-string 之外的独立段，含花括号会炸 `.format()` 调用方
    assert "{" not in DERIVED_USAGE_RULES and "}" not in DERIVED_USAGE_RULES


def test_limit_caps_item_count():
    items = build(_fin(), limit=3)
    assert len(items) == 3


def test_all_three_cashflow_nets_are_derived_with_bracketed_negatives():
    """现金流三净额必须**成片**覆盖。

    首版只给"经营活动净额"，实测（2026-10-24 600941 A/B）模型立刻自己外推
    投资/筹资两项，并在正文里留下「-107.45 - (-645.9) ? 此处需注意」「*再次核对*」
    这类草稿痕迹。补齐后它才有可抄的配对；负数必须带括号，否则算式成病句。
    """
    fin = _fin()
    fin["现金流量表(亿元)"] = {
        "经营活动净额": [1500.0, 900.0, 400.0, 1700.0, 1200.0],
        "投资活动净额": [-895.05, -645.9, -300.0, -1000.0, -800.0],
        "筹资活动净额": [-943.94, -443.09, -200.0, -1100.0, -900.0],
    }
    items = build(fin)
    names = [x["name"] for x in items]
    for k in ("经营活动净额", "投资活动净额", "筹资活动净额"):
        assert any(n.startswith(f"{k}(单季") for n in names), (k, names)
    inv = [x for x in items if x["name"].startswith("投资活动净额(单季")
           and "2026-09-30" in x["name"]][0]
    assert inv["value"] == -249.15
    assert inv["formula"] == "(-895.05) - (-645.90)"


# ---- 接线：派生表必须真的进了提示词（"验了函数" ≠ "验了接线"）----

def _capture_fundamental_prompt(monkeypatch, financial_data):
    from llm import deepseek_client as DC

    seen = {}

    def fake(self, messages, **kw):
        seen["messages"] = messages
        seen.update(kw)
        return "基本面报告"

    monkeypatch.setattr(DC.DeepSeekClient, "call_api", fake)
    DC.DeepSeekClient().fundamental_analysis(
        {"symbol": "600941", "name": "中国移动", "current_price": 99.14},
        financial_data=financial_data, analysis_date="2026-10-24", data_through="2026-10-23")
    return seen["messages"], seen


def test_derived_table_is_wired_into_prompt(monkeypatch):
    messages, kw = _capture_fundamental_prompt(monkeypatch, _fin())
    prompt = messages[1]["content"]
    assert "【派生指标表" in prompt
    assert "790.10 - 293.54" in prompt          # 单季净利润的算式真的在里面
    assert "派生指标使用规则" in prompt           # 引用规则必须同时下发
    assert kw.get("max_tokens") == 3200          # 回归：不破坏上一个截断修复


def test_no_derived_section_when_no_financials(monkeypatch):
    """无财报时**绝不**出现派生表段 —— 空表会诱导模型编数字。"""
    messages, _ = _capture_fundamental_prompt(monkeypatch, {})
    prompt = messages[1]["content"]
    assert "【派生指标表" not in prompt
    assert "派生指标使用规则" not in prompt


def test_derived_build_failure_degrades_to_plain_prompt(monkeypatch):
    """派生表构造抛异常时必须降级成"没有该段"，不能让基本面整节失败。"""
    from llm import deepseek_client as DC
    import core.derived_fin as DF

    def boom(*a, **k):
        raise RuntimeError("模拟派生失败")

    monkeypatch.setattr(DF, "build", boom)
    messages, _ = _capture_fundamental_prompt(monkeypatch, _fin())
    prompt = messages[1]["content"]
    assert "【财务报表（真实输入）】" in prompt    # 原财报段还在
    assert "【派生指标表" not in prompt            # 派生段缺席，但没炸
