# -*- coding: utf-8 -*-
"""多 AI 交易联赛 · 唤醒条件（C4，2026-10-14，S5）。

设计要点：**复用盯盘（`core.watcher.AlertWatcher`）已经取好的同一份 quotes**
——盘中请求量受 0.5 限频约束，多拉一遍会让盯盘自己超时。
所以本模块只做"用给定 quotes 判定 AI 条件"，取数由 `AlertWatcher.scan_once` 负责。

判定口径（每条都有测试钉住）：
  · stop_loss / take_profit —— **仅持仓有效**；阈值可用绝对价（threshold）
    或相对持仓成本的百分比（threshold_pct，随机对照组用）
  · price_above / price_below —— 观察价，非持仓也有效
  · time —— `now.strftime('%H:%M') >= wake_at`
  · ⚠️ `quote.stale=True` 一律不判（过期价会误触发，页 8 已有这条纪律）
  · ⚠️ `expires_on < trade_date` 不判（并标记失活）
  · ⚠️⚠️ **阈值合理性校验**（2026-10-14 首轮实跑发现）：模型会给无效价 ——
    实测 `ai_tech` 给贵州茅台（千元级）写 `stop_loss threshold=5.0`。
    止损价必须 **< 现价**、止盈价必须 **> 现价**，否则视为**无效条件**（标记、不触发），
    否则它会在盘中"永不触发"或"立即触发"，两种都是静默错误。
    ⚠️ 判据的**唯一实现是 `core.ai_agent.wake_threshold_ok`**（2026-10-15 起）：
    本模块判时调用它，**写入时（`ensure_stops`）也调用它** —— 因为只在判时拦下时，
    这条无效条件仍留在库里，会让 `ensure_stops` 误判"已有止损"而**不补代码兜底**，
    持仓实际裸奔（僵尸止损）。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from core import ai_agent, ai_budget, ai_personas
from db import store

_logger = logging.getLogger(__name__)

# 盘中并入盯盘的代码数上限（防"AI 条件写太多把限频预算吃光"）
MAX_ARMED_CODES = 50

# 持仓类条件（仅持仓有效）与观察类条件
_POSITION_KINDS = ("stop_loss", "take_profit")
_OBSERVE_KINDS = ("price_above", "price_below")

# 全部合法 kind（**唯一来源**）。写条件的模块（如 `ai_routine._execute`）必须
# 引用这一个，不要在别处再抄一份字面量：抄一份就多一处会漂的副本，
# 而写错的 kind 不会报错 —— 它只是永远不会触发。
WAKE_KINDS = _POSITION_KINDS + _OBSERVE_KINDS + ("time",)

# ⚠️ 阈值合理性（"永不触发的空壳条件"）的**唯一实现**在
#    `core.ai_agent.wake_threshold_ok`（写时 + 判时共用，避免两处分叉）。
#    本模块的 `evaluate()` 直接调用它。


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _positions_of_state(state: dict) -> list:
    """从账户状态里取 `(code, quantity)`。

    ⚠️⚠️ `positions` 是 **dict**（`{code: asdict(Position)}`），**不是 list**。
    首版按 list 遍历 ⇒ 每次都得到"空持仓" ⇒ **止损/止盈条件永不触发**，
    而 S5 的单测因为**注入了** `holdings_of`/`cost_of` 把默认实现完全遮住
    （"注入绕过真实实现"陷阱 —— 注入是为了可测，但默认实现必须单独测）。
    """
    pos = (state or {}).get("positions") or {}
    out = []
    if isinstance(pos, dict):
        for code, p in pos.items():
            qty = _num((p or {}).get("quantity")) if isinstance(p, dict) else None
            if qty is not None:
                out.append((str(code), qty))
    elif isinstance(pos, list):                     # 容错：历史快照若曾用 list
        for p in pos:
            if isinstance(p, dict) and p.get("code"):
                qty = _num(p.get("quantity"))
                if qty is not None:
                    out.append((str(p["code"]), qty))
    return out


def _holdings_of(persona: str) -> set:
    """该 persona 的持仓代码集合。读不到返回空集（"不知道持仓"⇒ 持仓类条件不判）。"""
    try:
        st = store.load_paper_state(persona)
        if not st:
            return set()
        return {c for c, qty in _positions_of_state(st) if qty > 0}
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[AI唤醒] 读取 %s 持仓失败: %s", persona, e)
        return set()


def _cost_of(persona: str, code: str) -> float | None:
    """持仓成本价。

    ⚠️ `Position.avg_cost` 是 **property**，`asdict()` **不会**带它
    ⇒ 必须用 `total_cost / quantity` 反算（首版直接读 `avg_cost` 会永远拿到 None）。
    """
    try:
        st = store.load_paper_state(persona) or {}
        pos = st.get("positions") or {}
        p = pos.get(str(code)) if isinstance(pos, dict) else None
        if not isinstance(p, dict):
            return None
        qty, total_cost = _num(p.get("quantity")), _num(p.get("total_cost"))
        if qty and total_cost is not None:
            return total_cost / qty
        return _num(p.get("avg_cost"))              # 容错：某些快照可能直接带它
    except Exception:                                        # noqa: BLE001
        return None


def random_only_codes(trade_date: str = None) -> set:
    """**只**被随机对照组关注的代码集合（供推送层过滤"实验噪音"）。

    ⚠️⚠️ 为什么**只过滤推送、不剔监控池**（2026-10-23）：
    `AlertWatcher.scan_once` 把 armed codes 并进 quotes，是为了让 `evaluate()`
    能判定这些条件。随机对照组的条件一旦不在 quotes 里就**永远不触发**
    ⇒ `ai_wake_events` 缺记录 ⇒ **联赛行为被改变**（公平性受损）。
    所以这里只回答"哪些代码是纯实验票"，由**推送层**决定要不要给他们发消息。

    判定依据是 `config/ai_personas.json` 的 **`kind` 字段**（"llm" vs "random"），
    **不是名字前缀** —— 按字段判定才是权威口径。

    一只票若**同时**被真实 persona（kind=llm）关注，就**不在**返回集合里
    （它对用户有意义，照常推送）。
    """
    day = trade_date or _today()
    try:
        personas = ai_personas.load_personas()
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[AI唤醒] 读取人格注册表失败（不过滤实验票）: %s", e)
        return set()
    random_codes: set = set()
    llm_codes: set = set()
    kind_of = {str(p.get("name")): str(p.get("kind") or "") for p in personas}
    try:
        conds = store.list_ai_wake_conditions(trade_date=day)
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[AI唤醒] 读取条件失败（不过滤实验票）: %s", e)
        return set()
    for c in conds:
        name = str(c.get("persona") or "")
        if not c.get("armed"):
            continue
        code = str(c.get("code") or "").strip()
        if not code:
            continue
        kind = kind_of.get(name, "")
        if kind == "random":
            random_codes.add(code)
        elif kind == "llm":
            llm_codes.add(code)
    return random_codes - llm_codes


def armed_codes(trade_date: str = None) -> list:
    """所有启用 persona 的**未过期、已武装**条件涉及的代码（供并入盯盘 codes）。

    ⚠️ 只报告"**判定可能需要的**代码"——不判断条件是否有效（阈值是否合理），
    那是 evaluate 的事。这里多带几个代码只是多几个行情字段，不会误触发。

    ⚠️ 这里**故意包含随机对照组**：联赛的公平性依赖它们被同样判定。
    "不给用户发实验票的噪音"是**推送层**的事（见 `random_only_codes`）。
    """
    day = trade_date or _today()
    active = {p["name"] for p in ai_personas.load_personas() if p["enabled"]}
    codes: list[str] = []
    for c in store.list_ai_wake_conditions(trade_date=day):
        if c["persona"] not in active:
            continue
        code = str(c.get("code") or "").strip()
        if code and code not in codes:
            codes.append(code)
        if len(codes) >= MAX_ARMED_CODES:
            _logger.warning("[AI唤醒] 条件代码数达到上限 %d，其余本轮不监控",
                            MAX_ARMED_CODES)
            break
    return codes


def _threshold_price(cond: dict, quote: dict, cost: float | None):
    """把条件换算成**价格阈值**。返回 (price, 来源)；无法换算返回 (None, 原因)。"""
    t = cond.get("threshold")
    if t is not None:
        try:
            return float(t), "threshold"
        except (TypeError, ValueError):
            return None, "threshold 非法"
    pct = cond.get("threshold_pct")
    if pct is not None:
        if not cost:
            return None, "缺持仓成本，无法按百分比换算"
        try:
            pct = abs(float(pct))
        except (TypeError, ValueError):
            return None, "threshold_pct 非法"
        kind = cond.get("kind")
        # 止损在成本之下，止盈在成本之上
        return (cost * (1 - pct) if kind == "stop_loss" else cost * (1 + pct)), "pct"
    return None, "既无 threshold 也无 threshold_pct"


def evaluate(quotes: dict, *, trade_date: str = None, now: datetime = None,
             holdings_of=_holdings_of, cost_of=_cost_of) -> list[dict]:
    """用**盯盘已取好的同一份** quotes 判定所有启用 AI 的唤醒条件。

    Args:
        quotes: `{code: {price, stale, name, ...}}` —— 由调用方（盯盘）提供，**不自己取数**
        holdings_of / cost_of: 可注入（测试用），默认读模拟账户

    Returns:
        命中列表，每项含 `{persona, cond_id, code, kind, price, trigger_px, threshold_used, reason}`。
        **invalid 的条件不返回**（但会记日志，便于发现"模型写了无效价"）。
    """
    now = now or datetime.now()
    day = trade_date or now.strftime("%Y-%m-%d")
    hhmm = now.strftime("%H:%M")
    active = {p["name"] for p in ai_personas.load_personas() if p["enabled"]}
    hits: list[dict] = []
    invalid: list[str] = []

    for cond in store.list_ai_wake_conditions(trade_date=day):
        persona = cond["persona"]
        if persona not in active:
            continue
        code = str(cond.get("code") or "")
        kind = cond.get("kind")
        q = quotes.get(code) or {}
        price = q.get("price")
        # ⚠️ stale 一律不判：过期价会误触发
        if q.get("stale") or not price:
            continue
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue

        if kind == "time":
            if cond.get("wake_at") and hhmm >= str(cond["wake_at"]):
                hits.append({"persona": persona, "cond_id": cond["cond_id"],
                             "code": code, "kind": kind, "price": price,
                             "trigger_px": price,
                             "reason": f"定时条件 {cond['wake_at']} 已到"})
            continue

        if kind in _POSITION_KINDS and code not in holdings_of(persona):
            continue                                    # 持仓类条件：没持仓不判

        if kind in _POSITION_KINDS:
            cost = cost_of(persona, code)
            target, src = _threshold_price(cond, q, cost)
            if target is None:
                invalid.append(f"{persona}/{code}/{kind}: {src}")
                continue
            # ⚠️ 量级合理性校验（2026-10-14 首轮实跑实测：ai_tech 给贵州茅台
            #    （千元级）写 `stop_loss threshold=5.0` ⇒ 按口径**永不触发**，
            #    等于给用户一个"看似有保护、实际不存在"的止损）。
            #    ⚠️ 判据**唯一实现在 `ai_agent.wake_threshold_ok`**（写时也走它，
            #    否则一条空壳止损会堵死 `ensure_stops` 的代码兜底）。
            ok, why = ai_agent.wake_threshold_ok(kind, cond.get("threshold"), price)
            if not ok:
                invalid.append(f"{persona}/{code}/{kind}: {why} ⇒ 视为无效")
                continue
            hit = (price <= target) if kind == "stop_loss" else (price >= target)
            if hit:
                hits.append({"persona": persona, "cond_id": cond["cond_id"],
                             "code": code, "kind": kind, "price": price,
                             "trigger_px": price, "threshold_used": target,
                             "threshold_src": src,
                             "reason": f"{'止损' if kind == 'stop_loss' else '止盈'}"
                                       f"触发：现价 {price:g} vs {target:g}"
                                       f"（{src}）"})
            continue

        if kind in _OBSERVE_KINDS:
            target, src = _threshold_price(cond, q, None)
            if target is None:
                invalid.append(f"{persona}/{code}/{kind}: {src}")
                continue
            hit = (price >= target) if kind == "price_above" else (price <= target)
            if hit:
                hits.append({"persona": persona, "cond_id": cond["cond_id"],
                             "code": code, "kind": kind, "price": price,
                             "trigger_px": price, "threshold_used": target,
                             "reason": f"价位条件触发：现价 {price:g} vs {target:g}"})

    if invalid:
        # 无效条件必须留痕：它意味着"AI 定的条件形同虚设"，是真实的质量问题
        _logger.warning("[AI唤醒] %d 条无效条件未参与判定: %s",
                        len(invalid), invalid[:5])
    return hits


# ---------------- 唤醒后的轻量决策轮 ----------------

_WAKE_PROMPT = """你是「{label}」这个模拟交易 AI，盘中条件被触发，需要**立即**做一次轻量决策。

【触发】{trigger_desc}
【触发类型】{kind}
【该标的当前价】{price}
【你的持仓】{holdings}
【你已设定的条件】{conds}

【可选动作（本轮**只允许**这些；只输出一个 JSON 对象）】
{{"final": {{"actions": [
   {{"action": "sell", "code": "{code}", "reason": "…"}},
   {{"action": "buy", "code": "{code}", "position_size": 0.005,
     "reason": "…（仅当触发类型是 price_below 时可能被接受）"}},
   {{"action": "set_wake", "code": "{code}", "kind": "stop_loss", "threshold": 5.0,
     "reason": "…"}},
   {{"action": "pass", "code": "{code}", "reason": "…"}}
], "summary": "本轮总体思路（必填）"}}}}

【盘中补仓的硬边界（**由代码强制，不通过则降级为"仅唤醒、不成交"**）】
- ① **只允许补仓已有持仓的标的**（`{code}` 是否在持仓里见上）——盘中**不开新仓**；
- ② **只有 `price_below`（下跌触发）才允许补仓**；`price_above`（上涨突破）
  只允许卖出止盈，**不许追涨**；
- ③ 额度取孰低：单次 ≤ 该标的持仓市值的 **3%**、≤ 总权益的 **3%**；
  单票单日 **1** 次、全账户单日 ≤ **4** 次且合计 ≤ 总权益的 **6%**。
- ⇒ 想补仓就给 `buy`，**别自己算金额**（代码会夹紧到下限并折算整手）；
  被拒时系统会记下"仅唤醒"，你可以在**盘后**再决策。

【纪律】
- ⚠️ 禁止申请深度分析（重活，盘中不做）。
- `summary` 必填；不动的理由必须具体（说清"等什么价位/什么信号"才动手）。
"""


def _wake_buy_usage(persona: str, trade_date: str) -> tuple[dict, int, float]:
    """当日该 persona 的**盘中补仓用量**：`({code: 次数}, 总次数, 总金额)`。

    ⚠️ 单一事实来源 = `ai_wake_events.action`（决策轮落库时的动作摘要）。
    不从 `paper_fills` 反推 —— 那样会把**盘后 T+1 成交**混进"盘中补仓"统计里。
    """
    per_code: dict = {}
    total, spent = 0, 0.0
    for e in store.list_wake_events(persona=persona, trade_date=trade_date):
        try:
            acts = json.loads(e.get("action") or "[]")
        except (TypeError, ValueError):
            continue
        for a in acts if isinstance(acts, list) else []:
            if isinstance(a, dict) and a.get("action") == "buy":
                c = str(a.get("code") or "")
                per_code[c] = per_code.get(c, 0) + 1
                total += 1
                try:
                    spent += float(a.get("amount") or 0.0)
                except (TypeError, ValueError):
                    pass
    return per_code, total, spent


def _default_buy_guard(hit: dict, persona: dict, trade_date: str,
                       action: dict = None) -> tuple[bool, str, float]:
    """**默认实现**：读真实账户 → 判定这次盘中补仓是否放行。

    ⚠️ 必须提供可用的默认实现（"注入遮住默认实现"陷阱）：测试注入 `buy_guard` 只能用于
       **构造边界场景**，生产路径必须真的走这一份，否则线上等于没有护栏。

    量级口径：
      · **权益** = 现金 + 全部持仓按**成本价**（保守，不用取不到的现价虚增）；
      · **该标的持仓市值** = 持仓股数 × **本次触发价**（不是成本价）。
    """
    name = persona["name"]
    # ⚠️ 先看**自动回退闸门**（连续 N 个负收益补仓日 ⇒ 自动回到"盘中仅卖出"）。
    #    放在最前面：被回退时连账户都不用读，理由也最直接。
    try:
        from core import ai_league
        status = ai_league.wake_buy_status(trade_date)
        if not status.get("enabled"):
            return False, status.get("reason") or "盘中补仓当前关闭", 0.0
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[AI唤醒] 读取盘中补仓开关失败（按关闭处理）: %s", e)
        return False, f"盘中补仓开关不可用（{type(e).__name__}）⇒ 保守关闭", 0.0
    try:
        st = store.load_paper_state(name)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[AI唤醒] 读取 %s 账户失败: %s", name, e)
        st = None
    if not st:
        return False, "读不到账户状态 ⇒ 不成交（宁可不动，不猜）", 0.0
    from core.paper_account import PaperAccount
    acc = PaperAccount.from_state(st)
    price = hit.get("price")
    equity = acc.cash + sum(p.quantity * p.avg_cost for p in acc.positions.values())
    pos = acc.positions.get(str(hit.get("code")))
    try:
        pos_val = (pos.quantity * float(price)) if (pos and price) else 0.0
    except (TypeError, ValueError):
        pos_val = 0.0
    per_code, orders_today, spent = _wake_buy_usage(name, trade_date)
    return ai_agent.wake_buy_check(
        code=hit.get("code"), kind=hit.get("kind"),
        holdings=list(acc.positions), position_value=pos_val, equity=equity,
        cash=acc.cash, price=price,
        bought_today=per_code.get(str(hit.get("code")), 0),
        orders_today=orders_today, spent_today=spent,
        limits=ai_agent.wake_buy_limits(persona))


def run_wake_round(persona: dict, hit: dict, *, trade_date: str, now: datetime = None,
                   client=None, buy_guard=None) -> dict:
    """触发后的轻量决策轮：**单轮** LLM 调用（不循环）。

    ⚠️ 为什么是单轮（与盘后轮不同）：唤醒发生在**盘中**，延迟本身是成本；
    且允许的动作已被收窄到"卖出/持仓内小额度补仓/调条件/pass"，不需要多轮探索。
    成本 = 1 次调用（实测 ≈4k tokens 量级）。

    额度：先扣 `wake`（默认 5 次/日）；扣不到只记录事件、不跑 LLM。

    Args:
        buy_guard: 注入点（测试用）。**默认走 `_default_buy_guard`（读真实账户）**——
            注入只能用于构造边界场景，生产路径必须验证默认实现真的生效。
    """
    name = persona["name"]
    granted, view = ai_budget.consume(name, "wake", trade_date=trade_date)
    if not granted:
        return {"persona": name, "status": "quota_exhausted", "decisions": [],
                "summary": "", "error": f"wake 额度不足（{view}）"}

    holdings = sorted(_holdings_of(name))
    conds = [c for c in store.list_ai_wake_conditions(persona=name, trade_date=trade_date)
             if c.get("code") == hit.get("code")]
    prompt = _WAKE_PROMPT.format(
        label=persona.get("label") or name,
        trigger_desc=hit.get("reason") or hit.get("kind"),
        kind=hit.get("kind"),
        price=hit.get("price"),
        holdings=json.dumps(holdings, ensure_ascii=False),
        conds=json.dumps([{k: c.get(k) for k in
                           ("kind", "threshold", "threshold_pct", "note")}
                          for c in conds], ensure_ascii=False),
        code=hit.get("code"))

    from llm.deepseek_client import DeepSeekClient, model_override, TEMP_CONSERVATIVE
    if client is None:
        client = DeepSeekClient()
    try:
        with model_override(persona.get("model")):
            raw = client.call_api([{"role": "user", "content": prompt}],
                                  temperature=TEMP_CONSERVATIVE, max_tokens=700,
                                  purpose=f"ai.{name}.wake")
    except Exception as e:                                   # noqa: BLE001
        return {"persona": name, "status": "error", "decisions": [], "summary": "",
                "error": f"LLM 调用失败: {type(e).__name__}: {str(e)[:120]}"}

    from core.ai_daily_round import parse_actions_reply, normalize_actions
    body = parse_actions_reply(raw)
    decisions = normalize_actions(body.get("actions"), persona,
                                  fallback_code=hit.get("code"))
    joined = json.dumps(decisions, ensure_ascii=False) + "\n" + str(raw) + "\n" + \
        json.dumps(hit, ensure_ascii=False)
    decisions = _apply_wake_guards(decisions, hit, persona, trade_date,
                                   buy_guard=buy_guard, joined=joined)
    if not decisions:
        decisions = [{"action": "pass", "code": hit.get("code"),
                      "reason": "唤醒轮未形成动作"}]
    return {"persona": name, "status": "ok", "decisions": decisions,
            "summary": str(body.get("summary") or ""), "error": ""}


def _apply_wake_guards(decisions: list, hit: dict, persona: dict, trade_date: str,
                       *, buy_guard=None, joined: str = "") -> list:
    """对唤醒轮产出的动作做**逗号级**过滤（唯一实现）：

    · `buy` —— 必须过盘中补仓三重护栏；**不过则降级为一条 `pass`**（仅唤醒不成交），
      而不是静默丢弃（丢掉会让用户以为"AI 什么都没做"）。
    · 其余动作（sell/set_wake/pass）原样通过。
    """
    guard = buy_guard or _default_buy_guard
    out: list = []
    for d in decisions:
        if d.get("action") != "buy":
            out.append(d)
            continue
        try:
            ok, why, amount = guard(hit, persona, trade_date, d)
        except Exception as e:                               # noqa: BLE001
            _logger.warning("[AI唤醒] 补仓护栏异常（按拒绝处理）: %s", e)
            ok, why, amount = False, f"护栏异常 {type(e).__name__}", 0.0
        if ok:
            try:
                px = float(hit.get("price") or 0)
            except (TypeError, ValueError):
                px = 0.0
            shares = int(round(float(amount) / px)) if px > 0 else 0
            out.append({**d, "intraday": True, "amount": round(float(amount), 2),
                        "shares": shares,
                        "reason": (str(d.get("reason") or "") +
                                   f"｜盘中补仓护栏通过：{why}")})
        else:
            _logger.info("[AI唤醒] %s/%s 盘中补仓被拒: %s",
                         persona.get("name"), hit.get("code"), why)
            out.append({"action": "pass", "code": hit.get("code"),
                        "reason": f"⚠️ 盘中补仓被护栏拒绝（{why}）⇒ 降级为仅唤醒，盘后再决策"})
    # ⚠️ 兜底再拦一次"买入"字样：护栏只认 `action=buy`，若模型用别的字段名夹带买入
    #    （例如 `action=place_order, direction=买入`），这里也不放行。
    if "买入" in joined and not any(d.get("intraday") for d in out):
        out = [d for d in out if d.get("action") not in ("buy", "place_order")]
    return out


def record_and_fire(hits: list, *, trade_date: str = None, now: datetime = None,
                    run_decision: bool = True, client_factory=None) -> dict:
    """写唤醒事件（幂等）+ 受额度限制地跑轻量决策轮。

    ⚠️ `run_decision=False` 时**只记录不跑 LLM** —— 手动检查（UI「立即检查」按钮）
    必须走这条，否则点一下按钮就花钱。
    ⚠️ 幂等靠 `ai_wake_events` 主键（persona, trade_date, code, cond_id）：
    进程重启后重复扫描同一天不会重复触发、不会重复花钱。
    """
    now = now or datetime.now()
    day = trade_date or now.strftime("%Y-%m-%d")
    fired, skipped, results = 0, 0, []
    for h in hits:
        is_new = store.record_wake_event(h["persona"], day, h["code"], h["cond_id"],
                                        hit_at=now.strftime("%H:%M:%S"),
                                        trigger_px=h.get("trigger_px"))
        if not is_new:
            skipped += 1                             # 今天已触发过（幂等）
            continue
        if not run_decision:
            store.mark_wake_event(h["persona"], day, h["code"], h["cond_id"],
                                  fired=False, skip_reason="仅记录（手动检查模式）")
            continue
        persona = ai_personas.get_persona(h["persona"]) or {"name": h["persona"]}
        if persona.get("kind") != "llm":
            store.mark_wake_event(h["persona"], day, h["code"], h["cond_id"],
                                  fired=False, skip_reason="随机对照组不跑盘中决策")
            continue
        client = client_factory(persona) if client_factory else None
        r = run_wake_round(persona, h, trade_date=day, now=now, client=client)
        # ⚠️ 动作摘要必须带上 `position_size`/`amount`：它是**当日盘中补仓用量**的
        #    唯一事实来源（`_wake_buy_usage` 靠它数次数与金额），漏了就等于没有额度护栏。
        store.mark_wake_event(h["persona"], day, h["code"], h["cond_id"],
                              fired=(r["status"] == "ok"),
                              action=json.dumps([{k: d.get(k) for k in
                                                  ("action", "code", "kind", "position_size",
                                                   "amount", "shares", "intraday")}
                                                 for d in r["decisions"]],
                                                ensure_ascii=False),
                              skip_reason=r.get("error") or "")
        if r["status"] == "ok":
            fired += 1
        results.append({"hit": h, "result": r})
    return {"trade_date": day, "hits": len(hits), "fired": fired,
            "skipped_idempotent": skipped, "results": results}
