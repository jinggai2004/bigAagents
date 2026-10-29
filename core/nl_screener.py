"""自然语言选股（Phase 3）：把一句话变成**受白名单约束的可执行条件**

设计立场（与需求规划的 Phase 3 硬要求一一对应）：

1. **字段白名单**：指标、比较符、板块全部来自本模块的常量表。模型（或规则）
   一旦给出白名单以外的字段，直接抛 `NLParseError`——不做"忽略未知字段继续跑"。
2. **解析失败必须显式报错**：解析不出任何可执行条件时给出明确原因（哪段没懂、
   哪个指标缺阈值），**绝不静默降级到内置精选池**假装选股成功。
3. **可复现**：确定性规则解析是主路径，同样输入永远得到同样结果；LLM 只作兜底，
   且其输出必须过同一套 `validate_payload` 校验（字段/比较符/数值/板块）。
4. **诚实标注（2026-10-07 重大变更）**：原先有 4 个指标（市值/换手率/成交额/
   当日涨跌幅）本地拿不到原始值，只能交给问财初筛、本地不核对。现已接入
   **本地全市场快照**（`data/sina_source`，新浪行业明细），12 个指标**全部本地可核对**，
   问财链路（`to_wencai()`）按 D1 决策**彻底移除**。
   同时新增一条硬边界：**快照不可用时明确报错，绝不降级到内置精选池**——
   拿硬编码股票冒充"按条件选出"比报错更糟。

术语：
- **DSL**：`NLQuery`（条件列表 + 板块 + 是否剔除 ST + 未解析片段）。
- **facts**：单只股票的指标原始值字典（单位见 `METRICS[...]["unit"]`），
  由 `facts_from_candidate` 从选股结果的因子明细里取。
"""

import json
import re
from dataclasses import dataclass, field

from core.indicator_registry import (
    INDICATORS, SOURCE_SNAPSHOT, describe_for_prompt,
    find_metric as _registry_find_metric,
)

# ================================================================
# 白名单：指标表
#
# **唯一来源是 `core.indicator_registry`**（元数据 + 取数口径 + 量级约束）。
# 保留 `METRICS` 这个名字只为不打断既有调用方；内容已改为注册表派生。
#
# 2026-10-07 改造：原先这张表自己维护 `wencai`（问财口径词）与 `local`
# （能否本地核对）两个字段。按 D1 决策彻底移除问财后：
#   - `wencai` 已删除（`to_wencai()` 一并移除）
#   - `local` 被 `source` 取代：`source == "snapshot"` 即"粗筛层可用"，
#     其余需逐票精算——比布尔标记更能表达"这个值在哪一层看得见"
# ================================================================

METRICS: dict = {k: v.to_dict() for k, v in INDICATORS.items()}

OPS = ("lt", "le", "gt", "ge", "eq", "between", "is_true")

_OP_WORDS = {"lt": "小于", "le": "不大于", "gt": "大于", "ge": "不小于",
             "eq": "等于", "between": "介于", "is_true": ""}

# 板块：与 core/screener.SECTOR_HARDCODED 的键一致（避免两处漂移）
SECTORS = ("科技", "金融", "消费", "新能源", "医药", "创业板")
BOARDS = ("创业板", "科创板", "北交所", "主板", "沪市", "深市")


class NLParseError(ValueError):
    """自然语言解析失败（字段非法 / 无可用条件 / 模型输出不合规）"""


# ================================================================
# 数据结构
# ================================================================

@dataclass(frozen=True)
class Condition:
    field: str                 # METRICS 的键
    op: str                    # OPS 之一
    value: float = None
    value2: float = None       # op == "between" 时的上界
    source: str = "rule"       # rule | llm | user

    @property
    def spec(self) -> dict:
        return METRICS[self.field]

    def scaled(self) -> str:
        """人类可读的条件（含单位）"""
        label, unit = self.spec["label"], self.spec["unit"]
        if self.op == "is_true":
            return label
        if self.op == "between":
            return f"{label} 介于 {_fmt(self.value)}{unit}~{_fmt(self.value2)}{unit}"
        return f"{label} {_OP_WORDS[self.op]} {_fmt(self.value)}{unit}"

    def to_dict(self) -> dict:
        d = {"field": self.field, "op": self.op, "value": self.value, "source": self.source}
        if self.value2 is not None:
            d["value2"] = self.value2
        return d


@dataclass
class NLQuery:
    raw_text: str = ""
    conditions: list = field(default_factory=list)
    sector: str = None
    board: str = None
    exclude_st: bool = True
    unparsed: list = field(default_factory=list)   # 没听懂的片段 / 缺阈值的指标
    source: str = "rule"                           # rule | llm | mixed

    @property
    def is_empty(self) -> bool:
        """没有任何可执行条件（此时必须报错，不得静默降级）"""
        return not self.conditions and not self.sector and not self.board

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "sector": self.sector,
            "board": self.board,
            "exclude_st": self.exclude_st,
            "conditions": [c.to_dict() for c in self.conditions],
            "unparsed": list(self.unparsed),
            "source": self.source,
        }

    def describe(self) -> list:
        """条件的中文回显（UI 里给用户确认/编辑用）"""
        out = [c.scaled() for c in self.conditions]
        if self.sector:
            out.append(f"板块：{self.sector}")
        if self.exclude_st:
            out.append("剔除 ST")
        return out


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return f"{v:g}"


# ================================================================
# 文本归一化与片段切分
# ================================================================

_FULLWIDTH = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4", "５": "5", "６": "6",
    "７": "7", "８": "8", "９": "9", "．": ".", "，": ",", "、": ",", "。": ".",
    "；": ";", "：": ":", "％": "%", "（": "(", "）": ")", "～": "~",
})

# 无语义的填充词：片段里只剩这些就当成"听懂了但没信息"
_FILLER = (
    "帮我", "帮忙", "请", "我想", "我要", "想要", "找", "选", "筛选", "选出", "挑",
    "一些", "几只", "只", "的", "了", "看看", "有没有", "满足", "符合", "条件",
    "股票", "个股", "标的", "公司", "a股", "沪深", "全部", "所有", "股票池",
    "板块", "行业", "里面", "中", "和", "与", "以及", "而且", "并且",
    "非st", "剔除st", "不要st", "不含st", "排除st", "去掉st", "st股", "非st股",
) + SECTORS + BOARDS


def _normalize(text: str) -> str:
    return text.translate(_FULLWIDTH).replace(" ", "").replace("\u3000", "")


def _split_segments(text: str) -> list:
    """按标点**与连接词**切分为条件片段（保留顺序 → 结果可复现）

    ⚠️ 连接词必须在这里切分。第一版只按标点切：
    `"均线多头排列且换手率大于1%"` 是**一个**片段 → `_parse_segment` 取位置最靠前的
    指标（均线多头排列）就返回 → 后半句**静默消失**，而且 `unparsed` 还是空的。
    用户看到"解析出 1 个条件"，完全看不出丢了条件。
    现在由「连接词切分」+ `_warn_leftover` 残留守卫 + 测试三重保护。
    """
    parts = re.split(r"[,.;，。；\n]+|(?:并且|而且|同时|以及|且|与|和|、)", text)
    return [p for p in (s.strip() for s in parts) if p]


def _warn_leftover(segment: str, pos: int, alias: str, out_unparsed: list) -> None:
    """片段里除了已用别名，是否还藏着**别的**指标别名（"静默漏解析"的最后一道防线）

    一个片段只解析一个条件。若切分没切开（用户用了没预料到的连接方式），
    这里保证**漏掉的条件会出现在 `unparsed` 里**，而不是凭空少一条。
    """
    rest = segment[:pos] + segment[pos + len(alias):]
    k2, _, _ = _find_metric(rest)
    if k2:
        out_unparsed.append(
            f"「{segment}」里还识别到「{METRICS[k2]['label']}」，"
            f"但一条片段只解析一个条件；多个条件请用「、」或「且」分隔")


def _strip_filler(segment: str) -> str:
    s = segment.lower()
    for w in _FILLER:
        s = s.replace(w, "")
    return s


# ================================================================
# 条件解析
# ================================================================

_NUM = r"(-?\d+(?:\.\d+)?)"
_COMPARATORS = (
    (r"(不高于|不超过|不大于|最多|以内|以下)", "le"),
    (r"(不低于|不少于|不小于|至少|以上)", "ge"),
    (r"(小于|低于|少于|不足|不到|之内|<)", "lt"),
    (r"(大于|高于|超过|多于|>)", "gt"),
    (r"(等于|达到|=)", "ge"),          # "达到80分" 按 >= 处理（口语里就是"至少"）
)


def _parse_number(segment: str, metric_key: str) -> tuple:
    """从片段里取出 (数值, 区间上界)；按指标基准单位换算单位后缀。

    金额类指标（市值/成交额/资金）的 DSL 单位统一是**亿元**：
    "500亿" → 500，"20000万" → 2，"1.2万亿" → 12000。
    """
    base_yi = METRICS[metric_key]["unit"] == "亿元"
    m = re.search(_NUM + r"(?:到|至|~|-)" + _NUM, segment)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if base_yi and "万" in segment:
            lo, hi = lo / 1e4, hi / 1e4
        return (lo, hi)
    m = re.search(_NUM + r"(万亿|亿|万|千|倍|%|个点)?", segment)
    if not m:
        return (None, None)
    value = float(m.group(1))
    unit = m.group(2) or ""
    if base_yi:
        if unit == "万亿":
            value *= 1e4
        elif unit == "万":
            value /= 1e4
        elif unit == "千":
            value *= 1e3 / 1e8
    return (value, None)


def _find_metric(segment: str):
    """返回 (指标键, 别名起始位置, 别名原文)；没找到返回 (None, None, None)

    委托给 `indicator_registry.find_metric` —— **别名表只能有一份**，
    否则"注册表加了别名、解析器不认"这种漂移不会报错，只会静默少解析出条件。
    判定规则：位置最靠前优先 + 同位置取最长别名（结果与遍历顺序无关，可复现）。
    """
    return _registry_find_metric(segment)


def _parse_segment(segment: str, out_unparsed: list):
    """解析单个片段 → Condition 或 None（并把原因写进 out_unparsed）"""
    metric_key, pos, alias = _find_metric(segment)
    if metric_key is None:
        if _strip_filler(segment):
            out_unparsed.append(f"「{segment}」没有对应的可执行指标")
        return None

    # 本片段是否还藏着别的指标（防"静默漏解析"）——对 bool / 数值两条分支都生效
    _warn_leftover(segment, pos, alias, out_unparsed)

    spec = METRICS[metric_key]
    if spec["kind"] == "bool":
        return Condition(field=metric_key, op="is_true")

    # 去掉指标别名后再找阈值：别名自身可能带数字（"20日涨幅"）、
    # 阈值也可能写在指标前面（"低于20倍市盈率"）——两种语序都要能解析。
    remainder = segment[:pos] + segment[pos + len(alias):]

    # 方向词："为正 / 为负 / 净流出"
    if re.search(r"(为正|是正|净流入|流入)", segment) and not re.search(_NUM, remainder):
        return Condition(field=metric_key, op="gt", value=0.0)
    if re.search(r"(为负|净流出|流出)", segment) and not re.search(_NUM, remainder):
        return Condition(field=metric_key, op="lt", value=0.0)

    value, value2 = _parse_number(remainder, metric_key)
    if value is None:
        out_unparsed.append(
            f"「{segment}」识别到指标「{spec['label']}」但没给出阈值（例如「市盈率低于20」）")
        return None

    if value2 is not None:
        return Condition(field=metric_key, op="between", value=min(value, value2),
                         value2=max(value, value2))

    op = None
    for pattern, candidate in _COMPARATORS:
        if re.search(pattern, remainder):
            op = candidate
            break
    if op is None:
        # 只给指标+数字（如"市盈率20"）→ 视为"不超过"，与口语习惯一致
        op = "le"

    # "跌幅大于5%" = 当日涨跌幅 < -5%：方向相反，数值取负
    if metric_key == "change_pct" and "跌幅" in segment and "涨跌幅" not in segment:
        flip = {"gt": "lt", "ge": "lt", "lt": "gt", "le": "gt"}
        op = flip.get(op, op)
        value = -abs(value)

    return Condition(field=metric_key, op=op, value=value)


def parse(text: str) -> NLQuery:
    """规则解析（确定性，不调用任何模型）。

    同一输入永远得到同一结果；解析不出条件时**不抛异常**，而是把原因放进
    `unparsed`，由调用方（`require_conditions`/UI）决定如何提示。
    """
    if not text or not str(text).strip():
        raise NLParseError("请输入选股条件，例如：市盈率低于20、均线多头排列的新能源股票")
    normalized = _normalize(str(text))
    query = NLQuery(raw_text=str(text).strip())
    unparsed: list = []

    for key in SECTORS:
        if key in normalized:
            query.sector = key
            break
    for key in BOARDS:
        if key in normalized:
            query.board = key
            break

    # 是否剔除 ST（默认剔除；显式说"含/包括 ST"才不剔除）
    if re.search(r"(含|包括|包含|只要)st", normalized.lower()):
        query.exclude_st = False
    else:
        query.exclude_st = True

    seen = set()
    for segment in _split_segments(normalized):
        # 板块 / ST 说明本身不是数值条件，避免被误判成"没听懂"
        if not _strip_filler(segment):
            continue
        cond = _parse_segment(segment, unparsed)
        if cond is None:
            continue
        if cond.field in seen:                       # 同一指标重复出现 → 取第一条
            unparsed.append(f"「{segment}」与前面同一指标的条件重复，已忽略")
            continue
        seen.add(cond.field)
        query.conditions.append(cond)

    query.unparsed = unparsed
    return query


def require_conditions(query: NLQuery) -> NLQuery:
    """没有可执行条件 → 抛 NLParseError（**绝不静默降级到精选池**）"""
    if query.is_empty:
        reasons = "；".join(query.unparsed) if query.unparsed else "没有识别到任何指标或板块"
        raise NLParseError(
            f"没能从这句话里解析出可执行条件：{reasons}。"
            "请写成「指标 + 比较 + 数值」的形式，例如「市盈率低于20、主力资金净流入为正」。")
    return query


# ================================================================
# 白名单校验（LLM 输出 / 用户在 UI 上手工编辑的条件都走这里）
# ================================================================

def validate_payload(payload, source: str = "llm") -> NLQuery:
    """校验结构化条件（模型输出或用户编辑），任一非法字段直接报错。

    这是"字段白名单"的落点：模型自造指标（rsi、macd…）不会被执行，也不会被
    悄悄忽略——直接抛错，让用户看到"模型给的字段不在白名单里"。
    """
    if isinstance(payload, str):
        payload = _extract_json(payload)
    if not isinstance(payload, dict):
        raise NLParseError("结构化条件必须是一个 JSON 对象")

    query = NLQuery(raw_text=str(payload.get("raw_text") or ""), source=source)
    sector = payload.get("sector")
    if sector:
        if sector not in SECTORS:
            raise NLParseError(f"板块「{sector}」不在白名单里（可用：{'、'.join(SECTORS)}）")
        query.sector = sector
    board = payload.get("board")
    if board:
        if board not in BOARDS:
            raise NLParseError(f"市场板块「{board}」不在白名单里（可用：{'、'.join(BOARDS)}）")
        query.board = board
    if "exclude_st" in payload:
        query.exclude_st = bool(payload["exclude_st"])

    conditions = payload.get("conditions") or []
    if not isinstance(conditions, list):
        raise NLParseError("conditions 必须是数组")
    for raw in conditions:
        if not isinstance(raw, dict):
            raise NLParseError(f"条件必须是对象：{raw!r}")
        fkey = raw.get("field")
        if fkey not in METRICS:
            raise NLParseError(
                f"指标「{fkey}」不在白名单里（可用：{'、'.join(METRICS)}）")
        op = raw.get("op")
        if op not in OPS:
            raise NLParseError(f"比较符「{op}」不在白名单里（可用：{'、'.join(OPS)}）")
        spec = METRICS[fkey]
        if spec["kind"] == "bool" and op != "is_true":
            raise NLParseError(f"「{spec['label']}」是形态类指标，只能用 is_true")
        if spec["kind"] == "num" and op == "is_true":
            raise NLParseError(f"「{spec['label']}」需要阈值，不能用 is_true")
        value = value2 = None
        if op != "is_true":
            value = _as_number(raw.get("value"), fkey)
            if op == "between":
                value2 = _as_number(raw.get("value2"), fkey)
                if value2 is None:
                    raise NLParseError(f"「{spec['label']}」的 between 需要 value2")
                value, value2 = min(value, value2), max(value, value2)
        query.conditions.append(
            Condition(field=fkey, op=op, value=value, value2=value2, source=source))
    return query


def _as_number(value, fkey: str):
    if isinstance(value, bool) or value is None:
        raise NLParseError(f"「{METRICS[fkey]['label']}」的阈值必须是数字，收到 {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise NLParseError(f"「{METRICS[fkey]['label']}」的阈值必须是数字，收到 {value!r}")


def _extract_json(text: str):
    """从模型回复里取出 JSON（容忍 ```json 包裹与前后废话）"""
    if not isinstance(text, str):
        raise NLParseError("模型没有返回文本")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except ValueError as e:
            raise NLParseError(f"模型返回的不是合法 JSON：{e}")
    raise NLParseError("模型返回里找不到 JSON 对象")


# ================================================================
# LLM 兜底（规则解析不出条件时才用，且输出必须过白名单）
# ================================================================

LLM_SYSTEM_PROMPT = """你是选股条件解析器。把用户的中文选股要求转成 JSON 条件，不要输出任何解释。

只能用下面这些 field（禁止自造指标，禁止输出没列出的字段）：
{fields}

op 只能用：lt(<) le(<=) gt(>) ge(>=) eq(=) between(区间, 需给 value2) is_true(形态类，无阈值)
金额类指标的数值单位统一是**亿元**，百分比类用 %（例如"20日涨幅超过10%" → value=10）。

输出格式：
{{"conditions":[{{"field":"pe","op":"lt","value":20}}],
  "sector": null, "board": null, "exclude_st": true}}

sector 只能是：{sectors}；board 只能是：{boards}。
无法确定的一律用 null / 不输出该条件。"""


def llm_system_prompt() -> str:
    """字段清单**由注册表生成**，不再手写。

    手拼字符串的后果是"注册表加了指标、prompt 里没有"——模型永远不会给这个字段，
    而且不报错。调研里 shibui.finance 的「SQL 模式库」解决的是同一类问题。
    """
    return LLM_SYSTEM_PROMPT.format(fields=describe_for_prompt(),
                                    sectors="、".join(SECTORS),
                                    boards="、".join(BOARDS))


def parse_with_llm(text: str, llm_fn=None) -> NLQuery:
    """规则优先；规则解析不出任何条件时才请模型兜底。

    `llm_fn(system_prompt, user_prompt) -> str`（返回 JSON 文本）。
    未提供 llm_fn 时直接返回规则结果（离线/测试环境可用）。
    """
    rule_query = parse(text)
    if rule_query.conditions or llm_fn is None:
        return rule_query
    payload_text = llm_fn(llm_system_prompt(), text)
    query = validate_payload(payload_text, source="llm")   # 白名单校验；非法即抛错
    query.raw_text = str(text).strip()
    query.unparsed = list(rule_query.unparsed)
    return query


def make_llm_fn(temperature: float = 0.0, max_tokens: int = 800):
    """构造给 `parse_with_llm` 用的 llm_fn；未配置任何密钥时返回 None。

    temperature=0 是刻意的：解析结果必须可复现（需求规划的硬要求）。
    """
    from config import settings as config
    if not getattr(config, "DEEPSEEK_API_KEY", "") and not getattr(config, "QWEN_API_KEY", ""):
        return None
    from llm.deepseek_client import DeepSeekClient
    client = DeepSeekClient()

    def llm_fn(system: str, user: str) -> str:
        return client.call_api(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_tokens, purpose="nl_screener.parse")

    return llm_fn


# ================================================================
# （已移除）问财语句映射
#
# 2026-10-07：按 D1 决策**彻底移除问财链路**。原因（见 output/NL选股改进方案v2 §2）：
#   现有白名单只有 12 个指标，而原 `to_wencai()` **只映射白名单里那几个**——
#   也就是说问财从来没有表达过白名单以外的条件，它扮演的只是"拿候选池"的角色。
#   一旦 4 个原本 local=False 的指标搬到本地（新浪行业明细一次给全），
#   这个角色就是**纯冗余**，移除它不会丢失任何当前能表达的能力。
#
# 候选池改由 `core/local_pool.LocalCandidatePool` 产生（本地快照粗筛 → 逐票精算）。
# 被删掉的函数：`to_wencai()`、`local_only_fields()`。
# ================================================================


# ================================================================
# 本地核对：facts × conditions
# ================================================================

def _snap_value(row, key):
    """从快照行（dict / pandas Series）取数值；取不到返回 None（**不是 0**）"""
    if row is None:
        return None
    try:
        raw = row.get(key) if hasattr(row, "get") else row[key]
    except Exception:                                 # noqa: BLE001
        return None
    if raw is None or isinstance(raw, bool):
        return None
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return None
    return None if f != f else f                           # NaN → None


# 快照提供的 5 个全市场标量（改造后才可本地核对）
_SNAPSHOT_FIELDS = ("pe", "market_cap", "turnover", "amount", "change_pct")


def facts_from_candidate(candidate: dict, snapshot_row=None) -> dict:
    """从选股结果与快照行取指标原始值（单位与注册表一致）。

    **两个来源**（2026-10-07 改造）：
      - `candidate["factors"]`：K线类 / 资金流 / 舆情 / 评分（screener 精算产物）
      - `snapshot_row`：全市场标量（PE / 总市值 / 换手率 / 成交额 / 当日涨跌幅）
        —— 这 5 个原先本地拿不到（报告里只能标"由问财初筛保证"），
        接入 `data/sina_source` 后**本地可核对**。

    缺失一律 None（不是 0），下游才能区分"没数据"与"数值恰好为 0"。
    快照类字段**以快照为准**（同一时点、全市场统一口径），而不是混用 screener
    自己那一份——同一次选股里两个 PE 并存且不一致是最难查的那种错。
    """
    factors = candidate.get("factors") or {}

    def _inner(name, key):
        entry = factors.get(name)
        if not isinstance(entry, dict):
            return None
        value = entry.get("value")
        if not isinstance(value, dict):
            return None
        raw = value.get(key)
        return raw if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None

    def _inner_bool(name, key):
        entry = factors.get(name)
        if not isinstance(entry, dict):
            return None
        value = entry.get("value")
        if not isinstance(value, dict):
            return None
        raw = value.get(key)
        return raw if isinstance(raw, bool) else None

    trend_ret = _inner("趋势", "ret20")
    news_rate = _inner("新闻正面率", "positive_rate")
    fund_wan = _inner("主力资金", "net_inflow_wan")

    facts = {
        "code": candidate.get("股票代码", ""),
        "name": candidate.get("股票名称", ""),
        "pe": _inner("PE估值", "pe"),
        "momentum20": None if trend_ret is None else trend_ret * 100,
        "ma20_above": _inner_bool("趋势", "above_ma20"),
        "ma_bullish": _inner_bool("均线排列", "bullish"),
        "volatility": _inner("波动率", "volatility"),
        "news_positive": None if news_rate is None else news_rate * 100,
        "fund_net": None if fund_wan is None else fund_wan / 1e4,   # 万元 → 亿元
        "score": candidate.get("score"),
    }
    # 快照类字段以快照为准（缺则回落 factors 里的同名值）
    for key in _SNAPSHOT_FIELDS:
        snap = _snap_value(snapshot_row, key)
        if snap is not None:
            facts[key] = snap
        else:
            facts.setdefault(key, None)
    return facts


def _compare(actual, cond: Condition) -> bool:
    if cond.op == "is_true":
        return bool(actual)
    if cond.op == "between":
        return cond.value <= actual <= cond.value2
    return {
        "lt": actual < cond.value,
        "le": actual <= cond.value,
        "gt": actual > cond.value,
        "ge": actual >= cond.value,
        "eq": actual == cond.value,
    }[cond.op]


def evaluate(facts: dict, query: NLQuery) -> dict:
    """核对单只股票，返回命中/未命中/未核对三类明细。

    规则：**数据缺失一律算"未通过"**（不假装命中）。用户要求"市盈率低于20"时，
    拿不到 PE 的股票不能算满足条件——这与评分卡"缺数据不给分"的立场一致。

    2026-10-07：`unverified` 这一路基本退休了 —— 12 个指标全部本地可核对
    （原先 4 个只能靠问财、标"未核对"）。字段保留以兼容 UI 渲染，
    语义变为"整条数据源不可用"（目前仅剩快照完全缺失这一种，而那时上层已直接报错）。
    """
    hits, misses, unverified = [], [], []
    for cond in query.conditions:
        spec = METRICS[cond.field]
        actual = facts.get(cond.field)
        if actual is None:
            misses.append(f"{cond.scaled()}：数据缺失，无法确认（按未通过处理）")
            continue
        if _compare(actual, cond):
            hits.append(f"{cond.scaled()}：实际 {_fmt(round(float(actual), 4))}{spec['unit']}")
        else:
            misses.append(f"{cond.scaled()}：实际 {_fmt(round(float(actual), 4))}{spec['unit']}")
    return {"pass": not misses, "hits": hits, "misses": misses, "unverified": unverified}


def filter_candidates(candidates: list, query: NLQuery, snapshot_lookup=None) -> list:
    """对候选集做本地核对，返回通过者（每只附带 nl_report 供页面回显）

    `snapshot_lookup(code) -> row|None` 用于取全市场快照行（提供 pe/总市值/换手率/
    成交额/涨跌幅）——不传时这 5 个指标恒为 None，含相关条件的股票会被判"数据缺失"。
    """
    kept = []
    for cand in candidates or []:
        facts = facts_from_candidate(cand, _lookup_row(snapshot_lookup, cand))
        report = evaluate(facts, query)
        if report["pass"]:
            enriched = dict(cand)
            enriched["nl_report"] = report
            enriched["nl_facts"] = facts
            kept.append(enriched)
    return kept


def annotate_candidates(candidates: list, query: NLQuery, snapshot_lookup=None) -> list:
    """给每只候选附上核对明细（不筛掉），供页面显示"为什么留下/为什么被过滤" """
    out = []
    for cand in candidates or []:
        facts = facts_from_candidate(cand, _lookup_row(snapshot_lookup, cand))
        enriched = dict(cand)
        enriched["nl_report"] = evaluate(facts, query)
        enriched["nl_facts"] = facts
        out.append(enriched)
    return out


def _lookup_row(snapshot_lookup, cand):
    if not snapshot_lookup:
        return None
    try:
        return snapshot_lookup(cand.get("股票代码", ""))
    except Exception:                                 # noqa: BLE001 - 取不到按缺数据处理
        return None
