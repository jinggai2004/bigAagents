"""选项式澄清：把"没听懂的话"变成"一次点击"

背景
----
`nl_screener.parse()` 对无法映射的片段只会丢进 `unparsed` 报错
（"「便宜」没有对应的可执行指标"）。用户看到的是**"没懂"**，而不是**"你是想说哪个"**。

调研（`output/自然语言选股_业界做法调研与对比.md` §1.A）里，问财的做法是请几十人标注
十年、覆盖两万条语义；零售端横评的原话则是"NL 筛选的风险是模糊请求会被误读，
所以仍需检查 AI 搭出来的条件"。我们选**成本可承受的那条路**：
不追长尾表达，用一小组高频映射 + 选项式反问把 80% 的场景收掉。

设计立场
-------
1. **不猜**。给不出候选映射的片段，保持 `unparsed` 原样报错——
   硬塞一个指标进去等于替用户做了他没做的决定。
2. **选项必须来自白名单**。每个选项都是一个合法 `Condition`，模块加载时就用
   `validate_payload` 校验一遍，规则表写错会**立刻**暴露（而不是等用户点下去才炸）。
3. **量级越界先问一句**。本项目已因"同列名不同单位"栽过三次，把这条经验做进交互
   （"市值 50 万 vs 50 亿"）比写进文档有用。
"""

from dataclasses import dataclass, field

from core.indicator_registry import INDICATORS
from core.nl_screener import Condition, NLQuery, validate_payload


@dataclass
class Option:
    label: str                       # 按钮文字
    conditions: list = field(default_factory=list)   # 点击后加入的条件
    note: str = ""                   # 补充说明（可选）


@dataclass
class Clarification:
    fragment: str                    # 触发澄清的原文片段
    question: str
    options: list = field(default_factory=list)


# ================================================================
# 高频模糊表达 → 候选映射
#
# ⚠️ 只放**白名单内**的指标；选项文字里要写清口径，因为用户点的就是这个口径。
# ================================================================

def _c(field_, op, value=None, value2=None) -> dict:
    d = {"field": field_, "op": op}
    if value is not None:
        d["value"] = value
    if value2 is not None:
        d["value2"] = value2
    return d


_RULES = [
    {
        "triggers": ("便宜", "估值低", "低估", "不贵", "估值便宜"),
        "question": "「{frag}」按哪个口径筛？",
        "options": [
            ("市盈率低于 15", [_c("pe", "lt", 15)]),
            ("市盈率低于 25", [_c("pe", "lt", 25)]),
            ("市盈率 0~20（顺带排除亏损股）", [_c("pe", "between", 0, 20)]),
        ],
    },
    {
        "triggers": ("强势", "涨得好", "趋势好", "走强", "表现好"),
        "question": "「{frag}」用哪个指标表达？",
        "options": [
            ("20 日涨幅超过 10%", [_c("momentum20", "gt", 10)]),
            ("站上 20 日均线", [_c("ma20_above", "is_true")]),
            ("均线多头排列", [_c("ma_bullish", "is_true")]),
        ],
    },
    {
        "triggers": ("大盘股", "大市值", "龙头", "权重股"),
        "question": "「{frag}」的市值下限取多少？",
        "options": [
            ("总市值大于 1000 亿", [_c("market_cap", "gt", 1000)]),
            ("总市值大于 500 亿", [_c("market_cap", "gt", 500)]),
            ("总市值大于 100 亿", [_c("market_cap", "gt", 100)]),
        ],
    },
    {
        "triggers": ("小盘股", "小市值", "小票"),
        "question": "「{frag}」的市值上限取多少？",
        "options": [
            ("总市值低于 100 亿", [_c("market_cap", "lt", 100)]),
            ("总市值低于 50 亿", [_c("market_cap", "lt", 50)]),
        ],
    },
    {
        "triggers": ("活跃", "交投活跃", "热门", "人气高"),
        "question": "「{frag}」用换手率还是成交额衡量？",
        "options": [
            ("换手率大于 5%", [_c("turnover", "gt", 5)]),
            ("换手率大于 3%", [_c("turnover", "gt", 3)]),
            ("成交额大于 10 亿", [_c("amount", "gt", 10)]),
        ],
    },
    {
        "triggers": ("稳健", "波动小", "比较稳", "低波动", "风险小"),
        "question": "「{frag}」的波动率上限取多少？",
        "options": [
            ("年化波动率低于 20%", [_c("volatility", "lt", 20)]),
            ("年化波动率低于 30%", [_c("volatility", "lt", 30)]),
        ],
    },
    {
        "triggers": ("主力进场", "有资金", "资金流入", "主力关注", "资金青睐"),
        "question": "「{frag}」用哪个资金口径？",
        "options": [
            ("近 5 日主力净流入为正", [_c("fund_net", "gt", 0)]),
            ("近 5 日主力净流入大于 1 亿", [_c("fund_net", "gt", 1)]),
        ],
        "note": "⚠️ 舆情/资金类指标依赖外部数据源，当前可能不可用（页面会标注）。",
    },
    {
        "triggers": ("消息面好", "利好", "舆情好", "有题材"),
        "question": "「{frag}」用新闻正面率表达吗？",
        "options": [
            ("新闻正面率大于 50%", [_c("news_positive", "gt", 50)]),
            ("新闻正面率大于 60%", [_c("news_positive", "gt", 60)]),
        ],
        "note": "⚠️ 舆情数据源当前覆盖有限，命中数可能很少（详见页面候选域说明）。",
    },
    {
        "triggers": ("优质", "好公司", "质量高", "评分高"),
        "question": "「{frag}」按综合评分取多少分？",
        "options": [
            ("综合评分大于 75", [_c("score", "gt", 75)]),
            ("综合评分大于 80", [_c("score", "gt", 80)]),
        ],
    },
]


# ── 模块加载即校验规则表：写错就立刻暴露，而不是等用户点下去才炸 ──
def _self_check() -> list:
    problems = []
    for rule in _RULES:
        for _, conds in rule["options"]:
            for cond in conds:
                key = cond.get("field")
                if key not in INDICATORS:
                    problems.append(f"规则里的指标 {key} 不在白名单")
                    continue
                try:
                    validate_payload({"conditions": [cond]}, source="user")
                except Exception as e:                # noqa: BLE001
                    problems.append(f"{key} 的选项非法: {e}")
    return problems


RULE_PROBLEMS = _self_check()


def _match_rule(fragment: str):
    for rule in _RULES:
        for trigger in rule["triggers"]:
            if trigger in fragment:
                return rule, trigger
    return None, None


def detect(query: NLQuery) -> list:
    """从 `unparsed` 片段里挑出**能给出候选映射**的，生成澄清问题。

    没命中任何规则的片段**不在这里处理** —— 它们继续以 `unparsed` 原样报错。
    这是刻意的：澄清机制只在"我们知道用户可能想说什么"时出手，不硬猜。
    """
    out = []
    seen = set()
    for item in query.unparsed or []:
        fragment = str(item)
        rule, trigger = _match_rule(fragment)
        if rule is None or trigger in seen:
            continue
        seen.add(trigger)
        options = [
            Option(label=label,
                   conditions=[Condition(field=c["field"], op=c["op"],
                                         value=c.get("value"), value2=c.get("value2"),
                                         source="clarify")
                               for c in conds],
                   note=rule.get("note", ""))
            for label, conds in rule["options"]
        ]
        out.append(Clarification(
            fragment=fragment,
            question=rule["question"].format(frag=trigger),
            options=options))
    return out


# 单位提示要**按单位给对**：给换手率提示"万元当亿元"是错的（本项目一贯要求
# 提示词与口径一致，错提示比没有提示更误导）。
_UNIT_HINTS = {
    "亿元": "是不是单位写错了？（例如把「万元」当成「亿元」、或把「元」当成「亿元」）",
    "%": "是不是多乘/少乘了 100？（百分数与小数的换算）",
    "倍": "是不是把别的口径当成了倍数？",
    "分": "评分区间是 0~100。",
}


def _unit_hint(ind) -> str:
    return _UNIT_HINTS.get(ind.unit, "请核对单位。")


def sanity_questions(query: NLQuery) -> list:
    """数值越出注册表声明的量级 → 反问一句（**不阻断**，用户可坚持）。

    这是把"同列名不同单位"这三次教训做进交互：
    市值写成 500000（本意可能是 50 万）时，直接执行会得到一个空结果，
    而用户不会知道是自己写错了单位。
    """
    out = []
    for cond in query.conditions:
        ind = INDICATORS.get(cond.field)
        if ind is None or not ind.sanity or cond.value is None:
            continue
        lo, hi = ind.sanity
        for v in (cond.value, cond.value2):
            if v is None:
                continue
            if lo <= v <= hi:
                continue
            out.append(Clarification(
                fragment=cond.scaled(),
                question=(f"「{cond.scaled()}」的数值 {v:g} 越出了常见量级"
                          f"（{ind.label} 通常 {lo:g}~{hi:g}{ind.unit}）。"
                          + _unit_hint(ind)),
                options=[]))
            break
    return out


def merge_condition(query: NLQuery, cond: Condition, fragment: str = None) -> NLQuery:
    """把澄清选中的条件并进 query。

    两件事：
      1. **同指标替换**——避免"市盈率低于20"和"市盈率低于15"同时存在（第二条会被
         `parse` 的"同指标重复已忽略"规则丢掉，用户会以为自己点了却没生效）；
      2. **清掉已被解答的 unparsed 项**——用户点了"市盈率低于 15"之后，
         页面不该还在警告"「便宜」没有对应的可执行指标"。
         `fragment` 传澄清问题的原文片段以精确匹配；不传时退化为按条件文字匹配。
    """
    conds = [c for c in query.conditions if c.field != cond.field]
    conds.append(cond)
    if fragment:
        unparsed = [u for u in query.unparsed if str(fragment) not in str(u)]
    else:
        unparsed = [u for u in query.unparsed if cond.scaled() not in str(u)]
    return NLQuery(
        raw_text=query.raw_text, conditions=conds, sector=query.sector,
        board=query.board, exclude_st=query.exclude_st,
        unparsed=unparsed, source=query.source)
