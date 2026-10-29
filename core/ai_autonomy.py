"""自营授权的参数与自洽校验（2026-10-27）

只有**参数**和**校验**，不碰任何业务逻辑。P1 的 `autonomy_check()`
（判断一个动作是否越权）从这里取阈值。

## 为什么单独一个文件

原方案把 `总仓 20% / 单票 8% / 现金地板 20%` 三个数字直接写进文档，结果是
**现金地板永远不会触发**：总仓上限 20% 意味着现金恒 ≥ 80%，"低于 20% 不许开仓"
这条判据是死代码。同理 `单票 8% × 3 只 = 24% > 20%`，实际最多只能持 2 只，
而文档写的是"池内建仓"——听起来像能选好几只。

两个数字分别看都合理，放在一起就矛盾。这类错误靠"再看一遍文档"发现不了，
只能靠机器算。所以：**参数集中一处 + `validate()` 机械校验 + 地板派生**。

## 口径

- 所有比例都是占**当前净值**（nav）的比例，不是初始资金。
  亏损后净值变小，同样的百分比对应更少的钱——这是保守方向，不修正。
- `cash_floor` **派生**自总仓上限（`1 - max_total_pct`），不单独配置。
  一旦允许分别配置，就会出现"地板高于隐含值"的死条款，而它不会报错。
"""

from __future__ import annotations

import logging
import re

_logger = logging.getLogger(__name__)

# L2 参数。改这里要同步跑 `tests/test_ai_autonomy.py`。
#
# ⚠️ 2026-10-27 第二次调整：老板明确要求「建仓权开 + 总仓给 100%」。
#    总仓 24% → 100% 这一个数字的改动，**牵动另外两个数字**，不是改一个值的事：
#
#    ① `cash_floor_pct` 派生自 `1 - max_total` ⇒ 变成 **0**，即"低于现金地板
#       不许开仓"这条判据**永久失效**。这不算 bug，是"允许满仓"的必然含义，
#       但必须**显式表达**（见 `has_cash_floor`），否则它就是第二个死条款——
#       和当初"地板 20% 永不触发"一模一样的形状。
#    ② `max_stocks × max_position` 必须真的**用得到** 100%。若沿用 3 只，
#       上限只有 24%，总仓 100% 就成了一句永远触不到顶的话。故放到 12 只：
#       12 × 8% = 96%，留 4% 给"最后一只分批没建完"的自然余量。
#       `validate()` 里有机器判据（`min_usable_total_pct`），不许再改回 3 只。
DEFAULTS: dict = {
    "max_position_pct": 0.08,   # 单票上限，占总净值（不变：8% 分散度才是保护）
    "max_total_pct": 1.0,       # 总仓上限，占总净值（老板要求满仓可用）
    "first_buy_pct": 0.03,      # 首笔建仓，占总净值（单票 8% 需分 3 笔）
    "max_stocks": 12,           # 最大持仓只数（= 100% / 8% 向上取整 12.5 → 12）
    "time_stop_days": 5,        # 时间止损：N 个交易日未按预期即减半/平
    # 止损距离下限，单位是 ATR14。**不能低于 1.5**：项目已知坑——止损设得太近
    # 会被日内正常波动打掉，实测这类"被打掉的止损"占全部止损的多数。
    "stop_loss_atr_mult": 1.5,
}

# 总仓上限的「可用率」下限：`max_stocks × max_position_pct` 至少要做到
# 总仓上限的这个比例，否则总仓上限是死条款。
# 取 0.9 而非 1.0：留一成余量给"最后一只没有正好建满"的常态，
# 但不能低到 24%/100% 那种"给了等于没给"。
MIN_USABLE_TOTAL_RATIO = 0.9

# 硬边界（不随 persona 覆盖而放宽的部分）。写成常量而不是配置，是因为
# 这些是"实现约束"而非"策略偏好"：改小了没意义，改大了会越过风控。
HARD_MIN_STOP_ATR = 1.5
_EPS = 1e-9


def resolve(overrides: dict | None = None) -> dict:
    """合并覆盖并**派生** `cash_floor_pct`。

    覆盖来源可以是 `config/ai_personas.json` 里 ashen 的 `autonomy_params`，
    也可以直接传 dict（测试用）。非法值不在这里拦——`validate()` 负责报错，
    保持"解析"与"校验"分离，这样校验能对任意来源的参数生效。
    """
    p = dict(DEFAULTS)
    for k, v in (overrides or {}).items():
        if k in DEFAULTS and isinstance(v, (int, float)):
            p[k] = v
    # 派生：现金地板 = 1 - 总仓上限。**不读 overrides**，这是刻意的——
    # 允许配置它就会重新出现"地板 > 1 - 总仓"的死条款。
    p["cash_floor_pct"] = round(1.0 - float(p["max_total_pct"]), 6)
    # 地板是否**真实存在**。总仓 100% ⇒ 地板 0 ⇒ "现金低于地板不许开仓"永久失效。
    # 写成布尔量而不是让下游去比较 `> 0`，是为了让"已关闭"这件事可被断言、
    # 可被打印、可被审计 —— 一个恒不触发的 if 是查不出来的。
    p["has_cash_floor"] = p["cash_floor_pct"] > 1e-9
    # 单票满仓所需笔数（向上取整）。首笔 3% / 单票 8% ⇒ 3 笔。
    fb = float(p["first_buy_pct"]) or 1.0
    p["lots_per_position"] = max(1, int(round(float(p["max_position_pct"]) / fb)))
    return p


def validate(params: dict) -> list[str]:
    """返回违规项列表，**空列表 = 通过**。

    ⚠️ 每条都必须能被**外部传入**的参数违反。若某项只能由本模块的派生逻辑
    决定（例如 `cash_floor == 1 - max_total`），就不能拿它做断言——那会是
    恒真守卫，看起来有保护、实际永远绿。`tests/test_ai_autonomy.py`
    里对每条都构造了违规参数，验证它真的会红。
    """
    errs: list[str] = []

    def num(key):
        v = params.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    mp = num("max_position_pct")
    mt = num("max_total_pct")
    fb = num("first_buy_pct")
    ms = num("max_stocks")
    cf = num("cash_floor_pct")
    tsd = num("time_stop_days")
    atr = num("stop_loss_atr_mult")

    for k, v in (("max_position_pct", mp), ("max_total_pct", mt), ("first_buy_pct", fb)):
        if v is None:
            errs.append(f"{k} 缺失或不是数字")
        elif not (0 < v <= 1):
            errs.append(f"{k}={v} 必须在 (0, 1] 内")

    if mp is not None and mt is not None and mp > mt + _EPS:
        errs.append(f"单票上限 {mp:.2%} 大于总仓上限 {mt:.2%}")
    if fb is not None and mp is not None and fb > mp + _EPS:
        errs.append(f"首笔 {fb:.2%} 大于单票上限 {mp:.2%}（一笔就打满，等于没有分批）")

    if ms is None:
        errs.append("max_stocks 缺失或不是数字")
    elif ms < 1:
        errs.append(f"max_stocks={ms:g} 必须 ≥ 1")
    elif mp is not None and mt is not None and ms * mp > mt + 1e-6:
        # 这才是原方案那个 bug 的机器判据：8% × 3 = 24% > 20%。
        errs.append(
            f"{ms:g} 只 × 单票 {mp:.2%} = {ms * mp:.2%} 超过总仓上限 {mt:.2%}"
            f"（实际最多只能持 {int(mt / mp + _EPS)} 只）")

    # 现金地板：允许显式传入，但**不得高于**隐含值，否则是死条款。
    if cf is not None:
        if not (0 <= cf < 1):
            errs.append(f"cash_floor_pct={cf} 必须在 [0, 1) 内")
        elif mt is not None and cf > (1.0 - mt) + 1e-6:
            errs.append(
                f"现金地板 {cf:.2%} 高于隐含值 {1 - mt:.2%}（= 1 − 总仓上限），"
                f"该判据永远不会触发")

    # 总仓上限必须**用得完**（这是"守卫不可恒真"的**反向**应用：不可恒假）。
    # 给了 100% 总仓却只许 3 只 × 8%（= 24%），那个 100% 永远触不到，
    # 等于没给 —— 纸面上权限很大，机器里用不出来，比不给更糟（人会误判风险敞口）。
    if mp is not None and mt is not None and ms is not None and ms > 0 and mp > 0:
        usable = ms * mp
        if usable < mt * MIN_USABLE_TOTAL_RATIO - 1e-6:
            errs.append(
                f"{ms:g} 只 × 单票 {mp:.2%} = {usable:.2%}，"
                f"只用到总仓上限 {mt:.2%} 的 {usable / mt:.0%}"
                f"（需 ≥ {MIN_USABLE_TOTAL_RATIO:.0%}）——"
                f"总仓上限永远触不到顶，等于没给；请提高 max_stocks 或 max_position_pct")

    if tsd is not None and tsd < 1:
        errs.append(f"time_stop_days={tsd:g} 必须 ≥ 1 个交易日")

    if atr is not None and atr < HARD_MIN_STOP_ATR - _EPS:
        errs.append(f"stop_loss_atr_mult={atr:g} 低于硬下限 {HARD_MIN_STOP_ATR}"
                    f"（止损太近会被日内噪声打掉）")

    # 分批：单票必须至少 2 笔才能建满，否则"分批建仓"是空话。
    lots = params.get("lots_per_position")
    if isinstance(lots, (int, float)) and lots < 2:
        errs.append(f"单票满仓只需 {lots:g} 笔，等于一次打满")

    return errs


def check_or_raise(params: dict) -> dict:
    """`validate()` 的便捷包装：有违规就抛 `ValueError`。

    给"加载配置"的路径用（启动时 / 每次读 persona 时）。不要在每轮决策里调——
    参数在一次进程内不会变，每轮校验是纯浪费。
    """
    errs = validate(params)
    if errs:
        raise ValueError("自营授权参数不自洽：" + "；".join(errs))
    return params


# ================================================================
# 授权级别
# ================================================================
# L0 观察 / L1 存量经营 / L2 池内建仓 / L3 研究（异步）
LEVELS = ("L0", "L1", "L2", "L3")

LEVEL_DESC = {
    "L0": "只观察、只记录，不产生任何动作",
    "L1": "只管存量：卖出、止损止盈、持仓内补仓、调唤醒条件；**不开新仓**",
    "L2": "可在阿深自选池内建新仓（首笔 ≤3% nav，单票 ≤8%，最多 12 只、合计 "
          "≤100% nav），受「不追涨/不接刀/不买停牌」等机械判据约束",
    "L3": "可提研究请求，入队后由盘后任务执行；不改变任何仓位",
}


def normalize_level(level) -> str:
    """非法级别一律回退 `L0`（最保守），**不抛异常**。

    理由：这是"权限"，读错时的安全方向只有一种——更保守。抛异常会让
    整轮决策挂掉，而回退 L0 只是"这轮不动"，代价可控。
    """
    s = str(level or "").strip().upper()
    return s if s in LEVELS else "L0"


# ================================================================
# 建仓权判据：autonomy_check()
# ================================================================
# 这一节回答一个具体问题："阿深这一轮给出的动作，哪些**真的可以执行**？"
#
# 设计三条铁律：
#  1. **降级不抛异常**。盘中一个越权动作不该让整轮决策挂掉——降级为 `pass`
#     并留痕，其余动作照跑。抛异常等于"一个动作废掉一轮"，而盘中每 40 分钟
#     才有一轮。
#  2. **缺数据 = 拒绝，不是放行**。净值算不出来时有两条路可走：放行（赌它
#     不会超）或拒绝（这轮不建仓）。选拒绝——盘中缺数据是常态，放行的
#     期望成本远高于"少做一笔"。
#  3. **仓位上限只有一个实现**。`params` 必须来自 `resolve()`，不要在调用方
#     另写一份 `0.08`。（2026-10-14 实测教训：两处阈值必然分叉。）

# 动作 → 允许执行的最低级别。未登记的动作**一律拒绝**（白名单，不是黑名单）。
ACTION_MIN_LEVEL = {
    "pass": "L0",
    "study": "L0",          # 只登记问题、不改仓位，最低级别也允许
    "set_wake": "L1",       # 挂条件不改变当前持仓
    "watch_add": "L1",      # 维护自己的自选池，不动仓位
    "watch_remove": "L1",
    "sell": "L1",
    "buy": "L2",            # ← 建仓权：这是 L2 与 L1 的唯一实质区别
}

# 每轮 `study` 上限（§4.5）。研究是盘后花钱的活，不能一轮登记 10 只。
MAX_STUDY_PER_ROUND = 1
# 阿深自选池上限（§5.2）。超了会顶爆 `ai_wake.MAX_ARMED_CODES=50` 并静默截断。
MAX_ASHEN_WATCH = 25

# ── 买入的机械否决条件（本项目实现，逐条可判，**不看模型怎么说**）──
# ⚠️ `near_limit_down` 的 −7% 是"接近跌停保护"，**不是跌停判据**：
#    主板跌停 −10%、创业板/科创板 −20%，叫"跌停"名不副实（记忆里记过这个坑）。
BUY_BLOCK_CHASE_PCT = 3.0        # 当日涨幅 > +3% ⇒ 追涨，不买
BUY_BLOCK_NEAR_DOWN_PCT = -7.0   # 当日跌幅 ≤ −7% ⇒ 接近跌停，不接刀
BUY_BLOCK_MIN_AMOUNT_WAN = 5000  # 成交额 < 5000 万元 ⇒ 流动性不足
# 限价的合理上限：比现价高 3% 以上就不是限价单了，是"写了个价糊弄校验"。
BUY_LIMIT_MAX_PREMIUM = 0.03

# 风险股简称的判据（ST / *ST / SST / S*ST 是**前缀**，退市整理股的「退」可在首尾）。
# ⚠️ 2026-10-27 由子串 `"ST" in name` 收紧为锚定匹配：子串写法把任何含 ST 的
#    串都当风险股，实测 `name="TEST"` 被当成 ST 拒掉（审查探针构造的反例）。
#    A 股简称不会出现英文 ST，误伤概率≈0，但"靠数据里恰好不出现"不是判据——
#    将来接港股/美股或特殊简称，会变成一条静默的错误否决。
_RISK_NAME_RE = re.compile(r"^(?:S?\*?ST|退)|退$")


def _num(v) -> float | None:
    """只接受真数字。`bool` 是 `int` 的子类，要单独排除（否则 True 变成 1）。"""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _is_a_share(code) -> bool | None:
    """转发 `core.scheduler.is_a_share_code`（**唯一实现**，不许在这里复制代码段规则）。

    延迟导入的理由：`core.scheduler` 顶层拉 apscheduler + pipeline，本模块
    被测试和 web 层频繁导入，不该连带这些重依赖。

    Returns: True/False/None（None = 校验器不可用，调用方必须**拒买**）
    """
    try:
        from core.scheduler import is_a_share_code
        return bool(is_a_share_code(str(code or "")))
    except Exception as e:                              # noqa: BLE001
        # 静默降级不行：它会让"所有买入被拒"看起来像"模型给了非 A 股"。
        _logger.warning("[自营授权] A 股代码校验不可用（将按「不可校验」拒买）: %s", e)
        return None


def buy_blocks(quote: dict | None) -> list[tuple[str, str]]:
    """逐条判定"这只票现在为什么不能买"，返回 `[(key, 人读理由)]`。

    ⚠️ `quote` 为 None 时返回 `no_quote` 而不是空列表 —— "没有行情"不等于
    "没有否决理由"，这两者混淆过一次就会买入一只查不到价的票。
    """
    if not isinstance(quote, dict):
        return [("no_quote", "没有行情数据，无法判断，不买")]
    out: list[tuple[str, str]] = []
    price = _num(quote.get("price"))
    prev = _num(quote.get("prev_close"))
    chg = _num(quote.get("change_pct"))
    if chg is None and price and prev:
        chg = round((price - prev) / prev * 100, 2)
    high, low = _num(quote.get("high")), _num(quote.get("low"))
    vol = _num(quote.get("volume_hand"))
    amt = _num(quote.get("amount_wan"))
    name = str(quote.get("name") or "")

    if not price or price <= 0:
        out.append(("no_price", "现价为 0 或缺失（停牌/未开盘），不买"))
    if vol is not None and vol <= 0:
        out.append(("halted", "成交量为 0（停牌），不买"))
    if chg is not None and chg > BUY_BLOCK_CHASE_PCT:
        out.append(("chasing", f"当日已涨 {chg:.2f}%，超过 +{BUY_BLOCK_CHASE_PCT:g}% "
                               f"⇒ 追涨，不买（建仓权只允许下跌接手）"))
    if chg is not None and chg <= BUY_BLOCK_NEAR_DOWN_PCT:
        out.append(("near_limit_down",
                    f"当日跌 {chg:.2f}%，已达 −{abs(BUY_BLOCK_NEAR_DOWN_PCT):g}% "
                    f"⇒ 接近跌停保护，不接刀"))
    if high and low and abs(high - low) < 1e-9:
        out.append(("one_word_board", f"最高=最低={high}（一字板），挂单也成交不了"))
    if _RISK_NAME_RE.search(name):
        out.append(("risk_name", f"名称含 ST/退（{name}），结构性剔除"))
    if amt is not None and amt < BUY_BLOCK_MIN_AMOUNT_WAN:
        out.append(("illiquid", f"成交额 {amt:,.0f} 万元 < {BUY_BLOCK_MIN_AMOUNT_WAN:,} "
                                f"万元，流动性不足"))
    return out


def _portfolio_facts(portfolio: dict | None, quotes: dict | None) -> dict | None:
    """把账户快照算成判仓位需要的四个数。**任何一项算不出来就返回 None**。

    持仓项形状：`{"market_value": 80000}` 或 `{"quantity": 8000}`（+ 现价）。
    数量字段**只认 `quantity`**——那是项目里 `Position.quantity` 的本名，
    接受 `qty`/`shares` 之类别名会让两套口径同时存在，早晚对不上。

    ⚠️ 遇到**不认识的形状必须返回 None**，不能 `continue` 跳过：
    跳过的后果是"12 只持仓被当成空仓"，于是只数上限和总仓上限全部失效，
    而这看起来一切正常（没有报错、没有 None）。这是"缺就是缺"的直接应用。

    Returns: {"nav", "cash", "total_mv", "held": {code: 市值}} 或 None
    """
    if not isinstance(portfolio, dict):
        return None
    cash = _num(portfolio.get("cash"))
    if cash is None or cash < 0:
        return None
    held: dict[str, float] = {}
    raw = portfolio.get("holdings") or {}
    items = raw.items() if isinstance(raw, dict) else [
        (str(h.get("code")), h) for h in raw if isinstance(h, dict)]
    for code, h in items:
        if not isinstance(h, dict) or not code:
            return None
        mv = _num(h.get("market_value"))
        if mv is None:
            qty = _num(h.get("quantity"))
            q = (quotes or {}).get(str(code)) if isinstance(quotes, dict) else None
            px = _num(q.get("price")) if isinstance(q, dict) else None
            if qty is None or px is None:
                return None            # 有一只算不出市值 ⇒ 总仓/单票上限都不可信
            mv = qty * px
        held[str(code)] = mv
    total_mv = sum(held.values())
    nav = _num(portfolio.get("nav"))
    if nav is None:
        nav = cash + total_mv
    if nav <= 0:
        return None
    return {"nav": nav, "cash": cash, "total_mv": total_mv, "held": held}


def portfolio_from_account(account, prices: dict | None = None) -> dict:
    """把 `PaperAccount` 转成本模块认识的快照形状（**唯一适配层**）。

    存在的理由：`_portfolio_facts` 只认 `market_value` / `quantity` 两种写法的
    持仓项，而账户对象是 `Position(code, quantity, available, total_cost, ...)`。
    让每个调用方自己去凑这个 dict，凑错一个字段就会触发"整轮买入被拒"，
    而且**看不出原因**（表现得像"没行情"）。所以转换只写一次。

    Args:
        prices: `{code: 现价}`。缺价的持仓没有 `market_value`，
                此时 `_portfolio_facts` 会返回 None ⇒ 买入被拒（保守方向）。
    """
    px = {str(k): _num(v) for k, v in (prices or {}).items()}
    holdings: dict[str, dict] = {}
    for code, pos in (getattr(account, "positions", {}) or {}).items():
        qty = _num(getattr(pos, "quantity", None))
        if not qty or qty <= 0:
            continue
        row: dict = {"quantity": qty}
        p = px.get(str(code))
        if p:
            row["market_value"] = qty * p
        holdings[str(code)] = row
    try:
        nav = _num(account.total_value(px))
    except Exception as e:                              # noqa: BLE001
        _logger.warning("[自营授权] 账户净值计算失败（本轮买入将被拒）: %s", e)
        nav = None
    return {"cash": _num(getattr(account, "cash", None)), "nav": nav,
            "holdings": holdings}


def autonomy_check(actions, *, level, params: dict, portfolio: dict | None,
                   my_watch=None, quotes: dict | None = None,
                   user_watch=None) -> dict:
    """把一轮动作过一遍授权判据，返回**可执行的动作**与**被拦的理由**。

    Args:
        actions: 已归一的动作列表（`{"action","code","position_size","reason",...}`）。
                 `buy` 的 `position_size` 是**本笔占总净值的比例**（与
                 `ai_daily_round.normalize_actions` 的 `clamp_plan` 同语义）。
        level: 字符串级别，非法值经 `normalize_level` 回退 L0。
        params: `resolve()` 的输出。**不要传 DEFAULTS**（少了派生键）。
        portfolio: `{"cash", "nav"?, "holdings"}`。缺项 ⇒ 买入一律拒。
        my_watch: 阿深自选池的代码集合（L2 建仓的**允许范围**）。
        quotes: `{code: row}`，用于六不买与限价校验。
        user_watch: 老板的自选池。**只读**：阿深对它做写动作一律拒。

    Returns:
        {"level", "checked", "kept": [动作], "blocked": [{"action","code","reason"}],
         "violations": [str], "notes": [str]}
        `kept` 里的 `pass` 保证**永远非空**（全被拦时补一条，便于落库留痕）。
    """
    lv = normalize_level(level)
    rank = {v: i for i, v in enumerate(LEVELS)}
    my_watch = {str(c) for c in (my_watch or [])}
    user_watch = {str(c) for c in (user_watch or [])}
    quotes = quotes if isinstance(quotes, dict) else {}
    facts = _portfolio_facts(portfolio, quotes)

    kept: list[dict] = []
    blocked: list[dict] = []
    violations: list[str] = []
    notes: list[str] = []

    def deny(a: dict, why: str):
        blocked.append({"action": a.get("action"), "code": a.get("code"), "reason": why})
        violations.append(f"{a.get('action')}/{a.get('code') or '-'}: {why}")

    # 先扫一遍：同批里给了止损的票（三件套的第三件允许由 set_wake 提供）
    stops_in_batch = {str(a.get("code")) for a in (actions or [])
                      if isinstance(a, dict) and a.get("action") == "set_wake"
                      and a.get("kind") == "stop_loss" and a.get("code")}
    study_used = 0
    watch_added = 0

    for a in (actions or []):
        if not isinstance(a, dict):
            continue
        act = str(a.get("action") or "").strip().lower()
        code = str(a.get("code") or "").strip()

        if act not in ACTION_MIN_LEVEL:
            deny(a, f"未登记的动作类型「{act}」，一律拒绝（白名单机制）")
            continue
        if rank.get(lv, 0) < rank[ACTION_MIN_LEVEL[act]]:
            deny(a, f"当前授权 {lv} 不足以执行 {act}（需 {ACTION_MIN_LEVEL[act]}）")
            continue

        # ⚠️ 这里**刻意不判** `code in user_watch`。
        #    早期版本写了"老板自选里的票不许动"，实测立刻误伤：同一只票**同时**
        #    出现在两套自选里是**正常状态**（老板提名给阿深后大概率如此，见
        #    `watchlist_items` 表注释），而 `watch_add` 只写 `owner='ashen'`，
        #    根本碰不到老板的清单。
        #    ⇒ "老板自选只读"由**结构性事实**保证（动作只写 ashen），
        #      不靠代码黑名单 —— 黑名单会把合法状态判成越权，且越权理由看起来
        #      还挺合理，排查时会绕很久。真要加写老板池的动作，
        #      它不在 `ACTION_MIN_LEVEL` 白名单里，会被下面"未登记动作"直接拒。
        if act in ("watch_add", "watch_remove") and code:
            ok = _is_a_share(code)
            if ok is not True:
                deny(a, "代码不是 A 股（或代码校验不可用），不进自选池")
                continue
        if act == "watch_add":
            # 池子上限（§5.2）：超了会顶爆 `ai_wake.MAX_ARMED_CODES=50` 并**静默截断**，
            # 表现为"明明在自选里却不盯盘"。按"当前池 + 本批已通过"计。
            if len(my_watch) + watch_added >= MAX_ASHEN_WATCH:
                deny(a, f"自选池已达上限 {MAX_ASHEN_WATCH} 只，先 watch_remove 再 add")
                continue
            watch_added += 1
        if act == "study":
            study_used += 1
            if study_used > MAX_STUDY_PER_ROUND:
                deny(a, f"每轮最多登记 {MAX_STUDY_PER_ROUND} 只研究")
                continue
        if act == "sell":
            if not facts or code not in facts["held"] or facts["held"].get(code, 0) <= 0:
                deny(a, "没有该票的可卖持仓")
                continue
        if act == "buy":
            why = _check_buy(code, a, facts=facts, quotes=quotes, my_watch=my_watch,
                             params=params, stops_in_batch=stops_in_batch)
            if why:
                deny(a, "；".join(why))
                continue

        kept.append(a)

    if not any(k.get("action") == "pass" for k in kept):
        kept.append({"action": "pass", "code": None,
                     "reason": "本轮无动作可执行（被授权判据拦下 "
                               f"{len(blocked)} 条）" if blocked else "本轮不动"})
    if not (params or {}).get("has_cash_floor", True):
        notes.append("现金地板已关闭（总仓上限 100%）——「现金不足不许开仓」"
                     "这条判据本参数下不会触发，靠 max_stocks 与单票上限约束敞口")
    if facts is None:
        notes.append("账户快照不完整（现金/净值/持仓市值算不出）⇒ 本轮所有买入被拒")
    return {"level": lv, "checked": len(actions or []), "kept": kept,
            "blocked": blocked, "violations": violations, "notes": notes}


def _check_buy(code: str, a: dict, *, facts, quotes, my_watch, params,
               stops_in_batch) -> list[str]:
    """单笔买入的全部否决理由。**空列表 = 可以买**。

    顺序刻意从"最容易错、最不该錯"开始：级别已在外层判过 → 池子 → 三件套 →
    六不买 → 仓位数学。先挡"这票根本不该买"，再算"买多少"。
    """
    errs: list[str] = []
    if not code:
        return ["缺 code"]
    # ① 池子：L2 的建仓权**只在阿深自己的自选池内**有效
    if code not in my_watch:
        errs.append("不在阿深自选池内（L2 建仓权只覆盖自己的池子）")

    # ② 三件套：限价、止损、理由
    if not str(a.get("reason") or "").strip():
        errs.append("缺买入理由（事后无法判断对错，这笔不算数）")
    limit = _num(a.get("limit_price") or a.get("limit"))
    px = _num((quotes.get(code) or {}).get("price"))
    if limit is None or limit <= 0:
        errs.append("缺限价（没有限价的买单＝接受任意开盘价）")
    elif px and px > 0:
        if limit > px * (1 + BUY_LIMIT_MAX_PREMIUM):
            errs.append(f"限价 {limit:g} 高于现价 {px:g} 的 "
                        f"{BUY_LIMIT_MAX_PREMIUM:.0%} 以上，不是限价单而是市价单")
    if code not in stops_in_batch and _num(a.get("stop_loss")) is None:
        errs.append("缺止损价（这笔亏损无上限）")

    # ③ 六不买（含"没有行情"）
    for _key, why in buy_blocks(quotes.get(code)):
        errs.append(why)

    # ④ 仓位数学
    size = _num(a.get("position_size"))
    if size is None or size <= 0:
        errs.append("缺 position_size（占总净值比例）")
        return errs
    if size > 1.0:                        # "30%" 写法
        size /= 100.0
    if facts is None:
        errs.append("账户快照不完整，算不出仓位上限")
        return errs
    nav, cash = facts["nav"], facts["cash"]
    amount = size * nav
    mp = float(params["max_position_pct"])
    mt = float(params["max_total_pct"])
    fb = float(params["first_buy_pct"])
    ms = int(float(params["max_stocks"]))
    held_mv = facts["held"].get(code, 0.0)

    if held_mv + amount > mp * nav + 1e-6:
        errs.append(f"单票 {(held_mv + amount) / nav:.2%} 超过上限 {mp:.2%}")
    if facts["total_mv"] + amount > mt * nav + 1e-6:
        errs.append(f"总仓 {(facts['total_mv'] + amount) / nav:.2%} 超过上限 {mt:.2%}")
    if amount > cash + 1e-6:
        errs.append(f"本笔需 {amount:,.0f}，现金只有 {cash:,.0f}")
    if params.get("has_cash_floor", True):
        cf = float(params["cash_floor_pct"])
        if (cash - amount) < cf * nav - 1e-6:
            errs.append(f"买完现金只剩 {(cash - amount) / nav:.2%}，"
                        f"低于地板 {cf:.2%}")
    if held_mv <= 0:
        if len(facts["held"]) >= ms:
            errs.append(f"已持 {len(facts['held'])} 只，达上限 {ms} 只")
        if amount > fb * nav + 1e-6:
            errs.append(f"首笔 {amount / nav:.2%} 超过 {fb:.2%}（单票要分 "
                        f"{int(params.get('lots_per_position') or 1)} 笔建满）")

    # ⑤ 一手成本必须买得起（A 股最小交易单位 = 100 股）。
    # ⚠️ 这条不是"额外保险"，是**真实会拦掉一大批标的**的硬约束：
    #    100 万账户 × 单票 8% = 8 万，而贵州茅台一手 ≈ 16 万 ⇒ 8% 上限下
    #    一手都买不起。没有这条判据时，执行层只会返回"不足一手"，
    #    看起来像"模型仓位给太小"，实际是"这票在这个账户规模下不可买"。
    #    说清原因比悄悄丢掉重要得多：否则阿深会反复提名同一只买不了的票。
    if px and px > 0:
        lot_cost = px * 100
        if lot_cost > mp * nav + 1e-6:
            errs.append(f"一手（100 股）需 {lot_cost:,.0f}，超过单票上限 "
                        f"{mp * nav:,.0f}（{mp:.0%} × 净值 {nav:,.0f}）—— "
                        f"当前账户规模下买不了这只票")
    return errs


def describe(params: dict) -> str:
    """把参数渲染成一行人读边界，供 prompt 与 UI 使用。

    ⚠️ 不要在 prompt 里手写"8%、24%、3 只"这类数字：改了参数忘了改 prompt，
    模型就会按旧数字规划，而代码按新数字拦——**两边都对，结果全被拒**。
    """
    p = params or resolve()
    parts = [
        f"单票 ≤{p['max_position_pct']:.0%}",
        f"总仓 ≤{p['max_total_pct']:.0%}",
        f"最多 {int(float(p['max_stocks']))} 只",
        f"首笔 ≤{p['first_buy_pct']:.0%}（单票分 {int(p.get('lots_per_position') or 1)} 笔建满）",
    ]
    if p.get("has_cash_floor"):
        parts.append(f"现金地板 {p['cash_floor_pct']:.0%}")
    else:
        parts.append("现金地板已关闭（可满仓）")
    parts.append(f"止损距离 ≥{p['stop_loss_atr_mult']:g}×ATR14")
    return "，".join(parts)
