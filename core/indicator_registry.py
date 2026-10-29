"""指标注册表：把自然语言里的"指标"统一成「元数据 + 取数口径 + 量级约束」

为什么要有它
-----------
改造前同一批指标元数据散在三处、各自维护：

1. `nl_screener.METRICS`      —— 别名/单位/白名单
2. `nl_screener.facts_from_candidate` —— 硬编码 7 个字段名（缺 4 个快照类指标）
3. `nl_screener.llm_system_prompt`    —— 手拼字符串生成提示词

加一个指标要改三处，而且**漏改不报错**（该指标永远返回 None，看起来只是"数据没有"）。
注册表把三者合一：元数据是唯一来源，提示词由它生成，取数口径由 `source` 声明。

与业界的对应关系（见 `output/自然语言选股_业界做法调研与对比.md` §3）：
shibui.finance 用一份 674 行的「SQL 模式库」注入 prompt 解决同类问题；
本注册表是它的等价物——**prompt 由注册表生成，而不是手写**。

`source` 的语义（决定指标在哪一层可见）
------------------------------------
    snapshot  全市场标量（新浪行业明细）→ **粗筛层即可用**
    kline     需个股 K 线（本地计算）
    flow      需资金流数据
    news      需新闻库 / 按个股直查
    score     选股评分链产物

⚠️ 粗筛层只能拿到 `snapshot` 类；其余必须在候选缩小后逐票精算——
这正是"信息面 Agent 属于第二层"的架构约束在代码里的落点。

`sanity` 的用途
--------------
每个指标给一个**量级合理区间**，用于运行时与测试的越界检查。
本项目已因"同列名不同单位"栽过三次（融资余额 元/亿、同花顺 `'20.03%'` 被
金额解析器吃成 0、成交量当成交额），量级断言是唯一能自动抓住这类错的手段。
"""

from dataclasses import dataclass, field

# 取数口径（决定指标在哪一层可见）
SOURCE_SNAPSHOT = "snapshot"
SOURCE_KLINE = "kline"
SOURCE_FLOW = "flow"
SOURCE_NEWS = "news"
SOURCE_SCORE = "score"

SOURCE_LABELS = {
    SOURCE_SNAPSHOT: "全市场快照（新浪行业明细）",
    SOURCE_KLINE: "个股 K 线（本地计算）",
    SOURCE_FLOW: "资金流数据",
    SOURCE_NEWS: "新闻库 / 按个股直查",
    SOURCE_SCORE: "选股评分链",
}


@dataclass(frozen=True)
class Indicator:
    key: str
    label: str
    unit: str
    kind: str                       # num（带阈值）| bool（形态类）
    aliases: tuple
    source: str
    sanity: tuple = None            # (lo, hi) 合理量级；None = 不做校验
    note: str = ""

    @property
    def needs_threshold(self) -> bool:
        return self.kind == "num"

    @property
    def is_snapshot(self) -> bool:
        return self.source == SOURCE_SNAPSHOT

    def to_dict(self) -> dict:
        """给下游（nl_screener / UI）用的普通字典形态"""
        d = {"label": self.label, "unit": self.unit, "kind": self.kind,
             "aliases": list(self.aliases), "source": self.source,
             "sanity": self.sanity, "note": self.note}
        return d


# ================================================================
# 12 个指标（白名单）
#
# ⚠️ 别名**长的写在前面**：`_find_metric` 按"位置最靠前 + 同位置取最长"匹配，
#    短别名排在前面不会出错，但长别名必须存在，否则"20日涨幅"会被当日涨幅吃掉。
# ================================================================

INDICATORS: dict = {
    # ── 快照类（粗筛层可用；原先 local=False，靠新浪行业明细救活）──
    "pe": Indicator(
        key="pe", label="市盈率(PE)", unit="倍", kind="num", source=SOURCE_SNAPSHOT,
        aliases=("市盈率", "pe", "估值"),
        sanity=(-5000, 50000),
        note="新浪行业明细 per；亏损股为负值"),
    "market_cap": Indicator(
        key="market_cap", label="总市值", unit="亿元", kind="num",
        source=SOURCE_SNAPSHOT,
        aliases=("总市值", "市值"),
        sanity=(0.1, 50000),
        note="新浪 mktcap 原始单位是**万元**，已在 data/sina_source 换算成亿元"),
    "turnover": Indicator(
        key="turnover", label="换手率", unit="%", kind="num", source=SOURCE_SNAPSHOT,
        aliases=("换手率",),
        sanity=(0, 100),
        note="新浪 turnoverratio；日线自算口径需 ×100（见 sina_source）"),
    "amount": Indicator(
        key="amount", label="成交额", unit="亿元", kind="num", source=SOURCE_SNAPSHOT,
        aliases=("成交额", "成交金额"),
        sanity=(0, 5000),
        note="新浪 amount 原始单位是**元**，已换算成亿元"),
    "change_pct": Indicator(
        key="change_pct", label="当日涨跌幅", unit="%", kind="num",
        source=SOURCE_SNAPSHOT,
        aliases=("今日涨幅", "今日涨跌幅", "涨跌幅", "涨幅", "跌幅"),
        sanity=(-25, 25),
        note="A 股涨跌停制度 ±20%（创业板/科创板），越界即口径有误"),

    # ── K 线类（需逐票精算）──
    "momentum20": Indicator(
        key="momentum20", label="20日动量", unit="%", kind="num", source=SOURCE_KLINE,
        aliases=("近20日涨幅", "近20个交易日涨幅", "20日涨跌幅", "20日涨幅",
                 "20日动量", "近一个月涨幅", "月涨幅"),
        sanity=(-100, 1000)),
    "volatility": Indicator(
        key="volatility", label="年化波动率", unit="%", kind="num", source=SOURCE_KLINE,
        aliases=("年化波动率", "波动率"),
        sanity=(0, 500)),
    "ma_bullish": Indicator(
        key="ma_bullish", label="均线多头排列", unit="", kind="bool", source=SOURCE_KLINE,
        aliases=("均线多头排列", "多头排列", "均线多头")),
    "ma20_above": Indicator(
        key="ma20_above", label="站上20日均线", unit="", kind="bool", source=SOURCE_KLINE,
        aliases=("站上20日线", "站上20日均线", "20日线上方", "突破20日线", "20日线之上")),

    # ── 精算类（候选缩小后才可见）──
    "fund_net": Indicator(
        key="fund_net", label="近5日主力净流入", unit="亿元", kind="num",
        source=SOURCE_FLOW,
        aliases=("主力资金净流入", "主力净流入", "资金净流入", "主力资金", "净流入"),
        sanity=(-500, 500)),
    "news_positive": Indicator(
        key="news_positive", label="新闻正面率", unit="%", kind="num",
        source=SOURCE_NEWS,
        aliases=("新闻正面率", "舆情正面率", "正面率"),
        sanity=(0, 100),
        note="⚠️ 2026-10-07 实测覆盖率为 0：库内关联率 0.9% + 直查被上游拦截"),
    "score": Indicator(
        key="score", label="综合评分", unit="分", kind="num", source=SOURCE_SCORE,
        aliases=("综合评分", "评分", "总分", "分数"),
        sanity=(0, 100)),
}

# 契约：键与 `Indicator.key` 必须一致（写错时立刻炸，而不是留个幽灵指标）
for _k, _v in INDICATORS.items():
    assert _k == _v.key, f"注册表键 {_k} 与 Indicator.key {_v.key} 不一致"


def get(key: str) -> Indicator | None:
    return INDICATORS.get(key)


def keys() -> list:
    return list(INDICATORS)


def by_source(source: str) -> list:
    return [v for v in INDICATORS.values() if v.source == source]


def snapshot_keys() -> list:
    """**粗筛层可用**的指标（全市场标量）"""
    return [v.key for v in by_source(SOURCE_SNAPSHOT)]


def precision_keys() -> list:
    """必须逐票精算的指标"""
    return [v.key for v in INDICATORS.values() if v.source != SOURCE_SNAPSHOT]


def all_aliases() -> list:
    """全部别名，按长度降序（长别名优先匹配，避免短别名抢占）"""
    out = []
    for v in INDICATORS.values():
        out.extend(v.aliases)
    return sorted(set(out), key=len, reverse=True)


def find_metric(segment: str):
    """从一段文本里找指标 → (key, 别名起始位置, 别名原文)。

    判定规则：**位置最靠前优先，同位置取最长别名** —— 结果与字典遍历顺序无关，
    保证同一输入永远得到同一解析（可复现是硬要求）。
    """
    lowered = segment.lower()
    best = (None, None, None)
    best_len = -1
    for v in INDICATORS.values():
        for alias in v.aliases:
            pos = lowered.find(alias.lower())
            if pos < 0:
                continue
            if (best[1] is None or pos < best[1]
                    or (pos == best[1] and len(alias) > best_len)):
                best, best_len = (v.key, pos, alias), len(alias)
    return best


def sanity_violations(facts: dict, key: str) -> list:
    """检查某指标值是否越出合理量级 → 返回问题描述列表（空 = 正常）。

    这不是"业务校验"，而是**口径守卫**：市值若出现在 1e5 以上量级，
    多半是忘了从"元"换算成"亿元"；换手率若到 200，多半是多乘了一次 100。
    """
    ind = INDICATORS.get(key)
    if ind is None or not ind.sanity:
        return []
    value = facts.get(key)
    if value is None or isinstance(value, bool):
        return []
    lo, hi = ind.sanity
    if not (lo <= value <= hi):
        return [f"{ind.label}={value:g}{ind.unit} 越出合理量级 [{lo:g}, {hi:g}]，"
                f"疑似单位口径有误"]
    return []


def check_facts(facts: dict) -> list:
    """对一只股票的完整 facts 做量级体检（UI / 测试可用）"""
    problems = []
    for key in INDICATORS:
        problems.extend(sanity_violations(facts, key))
    return problems


def describe_for_prompt() -> str:
    """生成 LLM 提示词里的字段清单（**由注册表生成，不再手写**）"""
    lines = []
    for v in INDICATORS.values():
        unit = v.unit or "无"
        lines.append(f"- {v.key}：{v.label}（单位 {unit}，{v.kind}，"
                     f"来源 {SOURCE_LABELS[v.source]}）")
    return "\n".join(lines)


def describe_for_ui() -> list:
    """给 UI 的"指标可执行性"表格：每个指标在哪一层可核对"""
    rows = []
    for v in INDICATORS.values():
        rows.append({
            "指标": v.label,
            "键": v.key,
            "单位": v.unit or "—",
            "类型": "数值" if v.kind == "num" else "形态",
            "取数口径": SOURCE_LABELS[v.source],
            "粗筛层可用": "是" if v.is_snapshot else "否（需逐票精算）",
        })
    return rows
