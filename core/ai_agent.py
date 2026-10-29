# -*- coding: utf-8 -*-
"""多 AI 交易联赛 · 工具层与护栏（C2 代码兜底 + C3 工具面，2026-10-14）。

职责边界：
- 本模块只做**纯函数层**（夹紧 / 工具注册 / 校验 / 随机对照组的规则决策），
  与 LLM 的多轮编排循环在 S4（`ai_daily_round`）里实现——先钉死边界，再做编排。

三条护栏（都有测试钉住，缺一不可）：
1. `clamp_plan`：**不信任模型自己写的仓位**——提示词管不住的部分由代码兜（项目铁律）。
2. `validate_tool_call`：未注册工具 / 越权（动配额）/ 唤醒场景越界（重活与加仓）一律拒绝。
3. 随机对照组（kind=random）**完全不调 LLM**——决策由本模块的规则生成，
   与回测 `--baseline N` 的随机零模型同一方法论（没有它排行不成立）。
"""
from __future__ import annotations

import logging
import random as _random
from dataclasses import dataclass, field

from core.risk import RiskEngine

_logger = logging.getLogger(__name__)

# ---------------- C2 代码兜底：仓位夹紧 ----------------
# (单票仓位系数, 最多同时持有只数, 每日最多下单数)
_BIAS: dict[str, tuple[float, int, int]] = {
    "conservative": (0.5, 3, 2),
    "balanced":     (1.0, 5, 3),
    "aggressive":   (1.0, 8, 5),
}


def _num(v):
    """宽松取数：容忍 "30%" / "30" / 0.3 三种写法（统一成分数）。"""
    if isinstance(v, str):
        v = v.strip().rstrip("%")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# 单票**最小有效仓位**（2026-10-23）：低于此值的"买入"视为噪声。
# 实测模型很爱给 2%~4% 的象征性仓位 —— 扣掉最低佣金后几乎没有盈亏意义，
# 却把"今天动手了"这件事占掉了（额度 1 次/天时尤其致命）。
# 5% 的口径：100 万账户 ⇒ 5 万元，够买 1 手高价股，也够产生可观测的盈亏。
MIN_EFFECTIVE_POSITION = 0.05


def clamp_plan(plan: dict, bias: str, single_limit: float = None,
               min_position: float = None) -> dict:
    """把模型给出的计划夹进风格允许的范围。

    - 仓位乘风格系数后，再夹到 [0, 单票上限]（上限取 persona.max_position，
      缺省用 RiskEngine.MAX_SINGLE_STOCK_WEIGHT=25%）；
    - "30%" / 30 / 0.3 三种写法都归一到分数（>1 视为百分数）；
    - **只夹不弃**：越界降级而非丢弃（与 _normalize_trade_plan 同哲学：标记不阻断）。

    Args:
        min_position: 最小有效仓位的覆盖值。缺省 `MIN_EFFECTIVE_POSITION`（5%）。
            ⚠️ 阿深（`ai_ashen`）必须传自己的 `first_buy_pct`（3%）：它的建仓
            是"单票分 3 笔、首笔 3%"，而默认的 5% 抬升会把首笔顶到 5%，
            再被 `ai_autonomy` 的"首笔 ≤3%"判据拒掉 —— 结果是**每一笔首笔建仓
            都失败**，且现象看起来像"模型乱给仓位"。
            这类"两个数字分别看都合理、放在一起矛盾"的错误只能靠改一处、
            让另一处显式传参来消除，不能靠调数值凑。
    """
    limit = float(single_limit if single_limit is not None
                  else RiskEngine.MAX_SINGLE_STOCK_WEIGHT)
    floor = (MIN_EFFECTIVE_POSITION if min_position is None
             else max(0.0, float(min_position)))
    k, _, _ = _BIAS.get(bias or "balanced", _BIAS["balanced"])
    size = _num(plan.get("position_size")) or 0.0
    if size > 1.0:                    # "30%" 或 30 → 0.30
        size /= 100.0
    size = max(0.0, min(size * k, limit))
    # 提到最小有效仓位：**只抬不砍**，且不得越过单票上限
    # （上限本身低于最小仓位时以上限为准 —— 风格约束优先于"有效性"）。
    if 0 < size < floor:
        size = min(floor, limit)
    return {**plan, "position_size": round(size, 4),
            "position_clamped": True}


def bias_limits(bias: str) -> tuple[float, int, int]:
    """风格的 (单票系数, 最多持仓只数, 每日最多下单数)——S4 编排用它做第二道闸。"""
    return _BIAS.get(bias or "balanced", _BIAS["balanced"])


# ---------------- C2' 唤醒阈值合理性（唯一实现） ----------------
# ⚠️ 为什么必须**在写入时**也校验（2026-10-15 实测得出的缺陷）：
#    `core.ai_wake.evaluate()` 早已会拦下无效阈值（不触发 + 记日志），
#    但**条件仍然留在库里**，而 `ensure_stops()` 只看"有没有 `stop_loss`"
#    ⇒ 一条**永不触发**的止损会把 −8% 的代码兜底**堵死** ⇒ 持仓实际裸奔。
#    实测：`ai_tech` 给贵州茅台（现价 1312.97）写 `stop_loss threshold=5.0`。
# ⇒ 校验必须**写时（本函数）+ 判时（ai_wake）共用同一实现**，否则两处会分叉。

# 阈值相对现价的最大允许偏离：超过即视为"写错的空壳条件"（永不触发）。
WAKE_MAX_DEVIATION = 0.8

_POSITION_KINDS = ("stop_loss", "take_profit")


def wake_threshold_ok(kind: str, threshold, price,
                      *, max_deviation: float = None) -> tuple[bool, str]:
    """校验持仓类唤醒条件的**绝对价阈值**是否"可能触发"。

    Returns: (是否可接受, 原因)。非持仓类（price_above/below/time）恒为可接受。

    判据（与 `core.ai_wake` 的盘中判定**同源**）：
      · `stop_loss` 的阈值若 **远低于**现价（偏离 > max_deviation）⇒ 永不触发 ⇒ 无效；
      · `take_profit` 的阈值若 **远高于**现价 ⇒ 永不触发 ⇒ 无效；
    ⚠️ **反向不算无效**：止损价高于现价 = 现价已跌破计划位，按口径应**立即触发**。

    三种"不判"情形（宁可放过也不误删，避免无依据地丢弃 AI 的意图）：
      · 非持仓类条件；· 阈值走 `threshold_pct`（天然有界）；· **没有现价**（判不了）。
    """
    if kind not in _POSITION_KINDS:
        return True, ""
    if threshold is None:
        return True, ""                      # 走 threshold_pct，本来就落在 (0, 成本) 区间内
    if not price:
        return True, ""                      # 无现价 ⇒ 判不了，不在写入时删（判时还有一道）
    try:
        t = float(threshold)
        px = float(price)
    except (TypeError, ValueError):
        return False, f"阈值/现价非数值（threshold={threshold!r} price={price!r}）"
    if px <= 0:
        return True, ""
    md = WAKE_MAX_DEVIATION if max_deviation is None else float(max_deviation)
    if kind == "stop_loss" and t < px * (1 - md):
        return False, (f"止损价 {t:g} 远低于现价 {px:g}"
                       f"（偏离 >{md:.0%}，永不触发）")
    if kind == "take_profit" and t > px * (1 + md):
        return False, (f"止盈价 {t:g} 远高于现价 {px:g}"
                       f"（偏离 >{md:.0%}，永不触发）")
    return True, ""


# ---------------- C5 盘中持仓内补仓（基础版，2026-10-15） ----------------
# 背景：原先唤醒轮**禁止一切买入**，理由是"盘中追涨是最容易被高估收益的动作"。
# 但 `set_wake` 的本质是 **AI 提前预埋的条件单**（触发逻辑预先约定，不是盘中临时拍脑袋），
# 只开放卖出等于**浪费了条件触发的时效性**，与"把被动等待变成主动响应"的目标相悖
# （用户 2026-10-15 的结论）。⇒ 放开，但必须做**三重边界**：
#
#   ① 标的边界：**仅限已有持仓的标的**（仓位微调，不是开新仓）；
#   ② 场景边界：**仅 `price_below`（下跌触发）**——上涨突破只许卖出止盈，从根上规避追涨；
#   ③ 额度边界：四个上限**取孰低**（见下表），且单票单日 1 次、全日 ≤2 次。
#
# ⚠️ 校验不通过一律**降级为"仅唤醒、不交易"**（盘后再决策）——绝不"部分执行"。
# ⚠️ 量级取"持仓市值"时用**触发时点的现价**（不是成本价），否则跌下来的仓位
#    会因分母变小而允许更小的加仓，与"越跌越买"的直觉相反且难解释。
WAKE_BUY_LIMITS = {
    "max_pct_of_position": 0.03,        # 单次 ≤ 该标的当前持仓市值的 3%
    # ⚠️ 2026-10-23：单次 1% → 3%、全日 2% → 6%、次数 2 → 4。
    #    实测 7 个交易日里 LLM 组对盘中唤醒**几乎全是 pass**（ai_tech 一次唤醒里连拒 4 只），
    #    原因之一就是"1% 连一手都买不到"——按 100 万权益算只有 1 万，高价股折完不足 1 手
    #    直接判拒绝。限额小到无法执行，等于把盘中通道关掉。
    "max_pct_of_equity": 0.03,          # 单次 ≤ 总账户权益的 3%
    "max_orders_per_code": 1,           # 单个标的单日最多 1 次加仓
    "max_orders_per_day": 4,            # 全账户单日加仓总次数 ≤ 4
    "max_daily_pct_of_equity": 0.06,    # 全账户单日加仓总金额 ≤ 总权益的 6%
}

# 仅这一种触发类型允许补仓（"下跌触发"）
_WAKE_BUY_KINDS = ("price_below",)


def wake_buy_limits(persona: dict = None) -> dict:
    """盘中补仓的额度上限（可按 persona 覆盖：`persona["wake_buy_limits"]`）。

    ⚠️ 允许覆盖是为了让"设置页"能调，但**不提供"关闭校验"的开关**——
       边界本身是不可协商的底线，能调的只是宽严。
    """
    lim = dict(WAKE_BUY_LIMITS)
    override = (persona or {}).get("wake_buy_limits")
    if isinstance(override, dict):
        for k, v in override.items():
            if k in lim:
                try:
                    lim[k] = float(v)
                except (TypeError, ValueError):
                    _logger.warning("[AI护栏] wake_buy_limits.%s 非法值 %r，用默认", k, v)
    return lim


def wake_buy_check(*, code: str, kind: str, holdings, position_value, equity,
                   cash, price, bought_today: int = 0, orders_today: int = 0,
                   spent_today: float = 0.0, limits: dict = None) -> tuple[bool, str, float]:
    """判定"这次盘中触发是否允许补仓"，并算出**可成交金额**。**纯函数，不执行**。

    Returns:
        `(是否允许, 原因, 允许的金额)`；不允许时金额为 0.0。

    边界（顺序即优先级，一条不过就拒绝并给出**可转述的理由**）：
      1. 标的必须是**已有持仓**；
      2. 触发类型必须是 `price_below`（下跌触发）；
      3. 现价必须有效（否则算不出股数，宁可不动）；
      4. 单票单日 ≤ N 次、全日 ≤ M 次；
      5. 金额 = min(持仓市值×3%, 权益×1%, 当日剩余额度)；
      6. 折成**整手**后若不足 1 手 ⇒ 拒绝（不成交比"买 3 股"更合理）。
    """
    from core import matching                                     # 局部导入：避免循环

    lim = dict(limits or WAKE_BUY_LIMITS)
    code = str(code or "")
    holdings = {str(c) for c in (holdings or [])}

    if code not in holdings:
        return False, (f"盘中补仓只允许**已有持仓**的标的（{code} 不在持仓内）——"
                       f"盘中不开新仓，避免缺盘后全市场校验的误判"), 0.0
    if str(kind) not in _WAKE_BUY_KINDS:
        return False, (f"盘中补仓只允许由 `price_below`（下跌触发）引发，"
                       f"本次是 `{kind}` —— 上涨突破只允许卖出止盈，不许追涨"), 0.0
    try:
        px = float(price)
    except (TypeError, ValueError):
        return False, f"现价无效（{price!r}），无法折算股数 ⇒ 不成交", 0.0
    if px <= 0:
        return False, f"现价非正（{px}）⇒ 不成交", 0.0

    if bought_today >= int(lim["max_orders_per_code"]):
        return False, (f"该标的当日已补仓 {bought_today} 次"
                       f"（上限 {int(lim['max_orders_per_code'])} 次/日）"), 0.0
    if orders_today >= int(lim["max_orders_per_day"]):
        return False, (f"当日盘中补仓次数已达上限 "
                       f"{int(lim['max_orders_per_day'])} 次"), 0.0

    try:
        pos_val = float(position_value or 0.0)
        eq = float(equity or 0.0)
        cash = float(cash or 0.0)
        spent = float(spent_today or 0.0)
    except (TypeError, ValueError):
        return False, "账户数据不可用（持仓市值/权益/现金非数值）⇒ 不成交", 0.0
    if eq <= 0 or pos_val <= 0:
        return False, "账户权益或持仓市值不可用 ⇒ 不成交", 0.0

    cap_pos = pos_val * float(lim["max_pct_of_position"])
    cap_eq = eq * float(lim["max_pct_of_equity"])
    cap_day = eq * float(lim["max_daily_pct_of_equity"]) - spent
    amount = min(cap_pos, cap_eq, cap_day)
    if amount <= 0:
        return False, (f"当日盘中补仓额度已用完"
                       f"（已用 {spent:.0f} / 上限 {eq * float(lim['max_daily_pct_of_equity']):.0f} 元）"), 0.0

    # 折整手 + 现金上限（留 0.1% 余量覆盖佣金/滑点）
    shares = matching.round_lot(amount / px)
    if shares <= 0:
        return False, (f"允许金额 {amount:.0f} 元不足 1 手（{px:g}×100≈{px * 100:.0f} 元）"
                       f"⇒ 不成交"), 0.0
    cash_cap = int((cash / (px * 1.001)) // 100) * 100
    if cash_cap < shares:
        shares = matching.round_lot(cash_cap)
    if shares <= 0:
        return False, f"现金不足（可用 {cash:.0f} 元，1 手需 ≈{px * 100:.0f} 元）⇒ 不成交", 0.0
    # ⚠️ 百分比从 lim 里取，**不要写死**（2026-10-23 调额时正文明写"权益 1%"，
    #    改了常量却忘了改文案 ⇒ 给模型的理由与实际额度不一致）。
    return True, (f"允许补仓 {shares} 股（≈{shares * px:.0f} 元；"
                  f"上限取孰低：持仓市值 {lim['max_pct_of_position']:.0%}= {cap_pos:.0f}／"
                  f"权益 {lim['max_pct_of_equity']:.0%}= {cap_eq:.0f}／"
                  f"当日剩余 {max(0.0, cap_day):.0f}）"), shares * px


# ---------------- C3 工具面（声明式注册表） ----------------
# cost 单位 = deep_analysis 类额度（1 credit ≈ 11 次 LLM 调用 / 4.7 万 token）。
# allowed_in：daily=盘后决策轮；wake=盘中唤醒轮（收窄：卖出/调条件/pass）。

@dataclass(frozen=True)
class Tool:
    name: str
    cost: int = 0                     # deep_analysis 额度的 credit 数
    desc: str = ""
    params: tuple = ()                # ((参数名, 必填), ...)
    allowed_in: frozenset = frozenset({"daily"})


TOOL_REGISTRY: dict[str, Tool] = {t.name: t for t in [
    # —— 免费只读（不新增采集，读的都是已落库数据）——
    Tool("get_quote", 0, "现价与数据日期", (("code", True),)),
    Tool("get_my_positions", 0, "自己的持仓"),
    Tool("get_my_nav", 0, "自己的净值"),
    Tool("get_my_budget", 0, "自己的剩余额度"),
    Tool("get_news", 0, "读已有新闻快讯", (("limit", False),)),
    Tool("get_sector_news", 0, "按行业读新闻评估", (("sector", True),)),
    Tool("list_reports", 0, "已有深度分析报告清单"),
    Tool("read_report", 0, "读一份已存在的深度分析报告", (("code", True),)),
    # —— 收费（重活）——
    # ⚠️ 成本实测（2026-10-15）：一次 ≈11 次 LLM 调用 / ≈3.5 万 token / ≈3.2~3.9 分钟，
    #    是整个决策轮（≈1.3 万 token）的约 2.7 倍 ⇒ 描述里必须写清代价，
    #    让 AI 自己权衡"值不值得为这只票花"，而不是随手点满额度。
    Tool("request_deep_analysis", 1,
         "现场跑一次完整深度分析（**本项目最贵的动作**：消耗 1 credit，"
         "≈3.5 万 token、约 3~4 分钟）。仅在候选池信息不足以决策时使用；"
         "同日同票已分析过会**自动复用**、不重复跑",
         (("code", True),)),
    # —— 动作 ——
    Tool("place_order", 0, "下单意图（仓位会被代码夹紧）。⚠️ 唤醒轮只允许**持仓内小额度补仓**"
                          "（须 `price_below` 触发；单次 ≤持仓市值 3%/权益 1%，"
                          "单票 1 次/日、全日 ≤2 次、合计 ≤权益 2%）——其余一律拒绝",
         (("code", True), ("direction", True), ("position_size", True),
          ("reason", True)), frozenset({"daily", "wake"})),
    Tool("set_wake_condition", 0, "设定唤醒条件（止损/止盈/价位/定时）",
         (("code", True), ("kind", True), ("reason", True))),
    Tool("pass", 0, "本轮不动（必须给理由）", (("reason", True),)),
    Tool("sell", 0, "卖出持仓（唤醒轮允许）",
         (("code", True), ("reason", True)), frozenset({"wake"})),
    # place_order 允许在 wake 出现，但"买入"方向必须**先过 wake_buy_check 语义护栏**
    # （见 validate_tool_call 的 wake_buy_ok 参数）——工具层不做金额/标的校验，
    # 那属于语义层（它需要读账户），两层分工不能混。
]}

# 唤醒轮里**禁止**的动作语义（计划 §3.4：盘中不做重活）：
#  - request_deep_analysis（重活）：allowed_in 根本不含 wake ⇒ 天然拒绝
#  - 买入加仓：**不再一刀切禁止**（2026-10-15 放开），但必须由调用方**先过
#    `wake_buy_check()` 的语义护栏**（仅持仓内 + 仅 price_below + 三重额度），
#    通过后才把 `wake_buy_ok=True` 传进来 —— 默认 False 保持"未校验即拒绝"的安全默认。
_FORBIDDEN_WAKE_ORDER_DIR = ("买入", "buy")


def validate_tool_call(name: str, args: dict | None, *,
                       context: str = "daily",
                       wake_buy_ok: bool = False) -> tuple[bool, str]:
    """校验一次工具调用。Returns: (是否放行, 原因)。**只校验，不执行**——执行在 S4 编排。

    拒绝永远不是异常：编排层要把原因如实转述给 AI（让它学会合规）。

    Args:
        wake_buy_ok: 调用方**已经**用 `wake_buy_check()` 校验过这次买入（仅持仓/仅下跌触发/
            额度三重边界）——只有这种"已过语义护栏"的买入才在唤醒场景放行。
            ⚠️ 默认 `False`：**没校验就是拒绝**（安全默认，防止新调用方忘传而悄悄放开）。
    """
    args = args or {}
    ctx = context or "daily"

    # ① 未注册工具 —— 一律拒绝（防"编造一个工具名"）
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return False, f"未注册的工具: {name}（可用: {', '.join(sorted(TOOL_REGISTRY))}）"

    # ② 越权红线：不许 AI 修改自己的配额/上限（口径 5：预算代码强制）。
    #    判据用**参数键名**（quota/limit/budget）而非值——工具的合法参数里
    #    根本没有这些键，出现即夹带；只读的 get_* 工具不在此列。
    if not name.startswith("get_") and any(
            w in k.lower() for k in args for w in ("quota", "limit", "budget")):
        _logger.warning("[AI工具] 越权请求被拒: %s%s", name, args)
        return False, "不得修改自己的配额或上限（额度由系统管理）"

    # ③ 场景白名单
    if ctx not in tool.allowed_in:
        return False, f"{name} 不允许在 {ctx} 场景使用（盘中只做轻量动作）"

    # ④ 必填参数齐全（缺 reason 的 pass 会被这里拦下）
    for pname, required in tool.params:
        if required and not str(args.get(pname) or "").strip():
            return False, f"缺少必填参数: {pname}"

    # ⑤ 唤醒场景买入加仓：只有**已过语义护栏**（仅持仓 + 仅下跌触发 + 三重额度）才放行。
    #    ⚠️ 这里只做"有没有被校验过"的门禁，不重复算金额/标的（那需要读账户，
    #       属于 `wake_buy_check` 的职责）——两层分工，避免两处各写一份后分叉。
    if name == "place_order" and ctx == "wake" and \
            str(args.get("direction")) in _FORBIDDEN_WAKE_ORDER_DIR and not wake_buy_ok:
        return False, ("唤醒轮买入须先通过盘中补仓护栏（仅持仓内标的 / 仅 price_below 触发 / "
                       "单次≤持仓市值3%且≤权益1% / 单票1次/日 / 全日≤2次且≤权益2%）；"
                       "本次未通过 ⇒ 拒绝")

    return True, ""


# ---------------- 随机对照组（kind=random，零 LLM） ----------------
# 方法论与回测 --baseline N 的随机零模型同源：从**同一候选池**里随机决策，
# 用于给 LLM 组提供经验分布——"跑赢随机"才是本事，光赚钱不算。

def random_decisions(persona: dict, *, candidates: list[str],
                     holdings: list[str] | None = None,
                     rng: _random.Random = None,
                     p_trade: float = 0.4) -> list[dict]:
    """生成随机对照组的当日决策（规则，完全不调 LLM）。

    决策分布（刻意简单且**与 LLM 组同一夹紧口径**，保证可比）：
    - 以 p_trade 概率开新仓：随机选 1 只候选，仓位在 [2%, 单票上限×风格系数] 均匀取值；
    - 已有持仓以 0.3 概率随机卖出 1 只；
    - 入场同时随机设止损（-5%~-15%）与止盈（+5%~+20%）唤醒条件——对照组也要有出场纪律，
      否则"只会买入的随机组"会让对照失去意义。

    Returns: 决策列表（形状与 LLM 组一致：action/code/position_size/...）
    """
    r = rng or _random.Random()
    _, max_hold, _ = bias_limits(persona.get("risk_bias"))
    limit = persona.get("max_position") or RiskEngine.MAX_SINGLE_STOCK_WEIGHT
    out: list[dict] = []
    holdings = list(holdings or [])

    if holdings and r.random() < 0.3:
        code = r.choice(holdings)
        out.append({"action": "sell", "code": code,
                    "reason": "随机对照：随机卖出"})
    if candidates and len(holdings) < max_hold and r.random() < p_trade:
        code = r.choice(candidates)
        # 仓位口径与 clamp_plan 一致：均匀取值后走同一夹紧（不留特殊通道）
        plan = clamp_plan({"position_size": r.uniform(0.02, limit)},
                          persona.get("risk_bias"), limit)
        stop_pct = r.uniform(0.05, 0.15)
        take_pct = r.uniform(0.05, 0.20)
        out.append({"action": "buy", "code": code,
                    "position_size": plan["position_size"],
                    "stop_pct": round(stop_pct, 4),
                    "take_pct": round(take_pct, 4),
                    "reason": "随机对照：随机开仓"})
        out.append({"action": "set_wake", "code": code, "kind": "stop_loss",
                    "stop_pct": round(stop_pct, 4),
                    "reason": "随机对照：随机止损"})
        out.append({"action": "set_wake", "code": code, "kind": "take_profit",
                    "take_pct": round(take_pct, 4),
                    "reason": "随机对照：随机止盈"})
    if not out:
        out.append({"action": "pass", "reason": "随机对照：本轮随机不动"})
    return out
