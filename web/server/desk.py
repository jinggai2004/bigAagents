# -*- coding: utf-8 -*-
"""交易台（`/api/desk/*`）—— P3。

## 一条硬纪律：价格**作入参**，端点不联网
`core.portfolio._price_of` 在缺价时会回落到 `akshare` 逐只取价（实测单只 ≈30s，
持仓页一次渲染触发 ≈9 次）。只读服务层绝不允许这样：
1. 先从**本地日线**批量取收盘价（`data.bars`，8.6ms/只，零外呼）；
2. 再把 `prices` 和 `allow_network=False` 一起传给 `PortfolioManager`；
3. 本地也没有该只 ⇒ `price=null` + `available=false`，**不返 0**。

## 估值不完整就说"不完整"
`PortfolioManager.load()` 只累加**取到价**的仓位 ⇒ 缺价时 `total_value` 会系统性偏低。
所以缺价一律把 `total_value` 置 `null` + `valuation_complete=false`，
另给 `market_value_priced`（仅已定价部分），绝不拿偏低的合计冒充总资产。
"""
from __future__ import annotations

import logging
import re
import threading
from datetime import datetime

from web.server import cache, envelope, picks

_logger = logging.getLogger(__name__)

TTL_PRICES = 30
TTL_LEAGUE = 60

# 模拟账户下单是「读整份 state → 改 → 写回」：两个请求并发会丢单。
# `store.save_paper_state` 自身是单事务，但**读**在事务外，所以锁必须在这里。
_ORDER_LOCK = threading.Lock()


# ---------------------------------------------------------------- 本地价
def local_closes(codes) -> tuple[dict, str | None, list]:
    """本地日线最后收盘价。返回 `(prices, price_date, missing_codes)`。

    `price_date` = 各只最后交易日的**最大值**（不同标的停牌日不同，取最大＝面板最新日）；
    缺数据的代码进 `missing_codes`，不塞 0、不塞成本价。
    """
    from data import bars

    codes = [str(c).strip() for c in codes if str(c).strip()]
    if not codes:
        return {}, None, []

    def _load():
        prices, dates, missing = {}, [], []
        for c in codes:
            try:
                df = bars.get_bars(c)
            except Exception as e:                            # noqa: BLE001
                _logger.warning("[desk] %s 本地日线读取失败: %s", c, e)
                missing.append(c)
                continue
            if df is None or len(df) == 0:
                missing.append(c)
                continue
            try:
                prices[c] = round(float(df["close"].iloc[-1]), 4)
            except (TypeError, ValueError, KeyError) as e:
                _logger.debug("[desk] %s 收盘价不可用: %s", c, e)
                missing.append(c)
                continue
            dates.append(str(df["date"].iloc[-1])[:10])
        return prices, (max(dates) if dates else None), missing

    ident = (tuple(sorted(codes)), _panel_stamp())
    return cache.cached("desk.prices", ident, TTL_PRICES, _load)


def _panel_stamp():
    try:
        from data import bars
        return bars.panel_date_max()
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 面板日期不可用: %s", e)
        return None


# ---------------------------------------------------------------- 真持仓
def portfolio(now=None) -> dict:
    from core.portfolio import get_portfolio_manager

    pm = get_portfolio_manager()
    codes = pm.active_codes()
    prices, price_date, missing = local_closes(codes)

    try:
        port = pm.load(prices=prices, allow_network=False)
        positions = pm.get_active_positions(prices=prices, allow_network=False)
        alerts = pm.check_stop_loss(prices=prices, allow_network=False)
        exposure = pm.get_exposure(prices=prices, allow_network=False)
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 持仓读取失败: %s", e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")

    rows, priced_value = [], 0.0
    for p in positions:
        px = prices.get(str(p.code))
        mv = None if px is None else round(px * p.quantity, 2)
        if mv is not None:
            priced_value += mv
        cost = round(p.entry_price * p.quantity, 2) if p.entry_price else None
        rows.append({
            "code": p.code, "name": p.name or "",
            "quantity": p.quantity,
            "entry_date": p.entry_date, "entry_price": p.entry_price,
            "price": px, "price_basis": "close" if px is not None else None,
            "available": px is not None,
            "market_value": mv, "cost": cost,
            "pnl": None if (mv is None or cost is None) else round(mv - cost, 2),
            "pnl_pct": None if (mv is None or not cost) else round((mv - cost) / cost * 100, 2),
            "stop_loss": p.stop_loss, "take_profit": p.take_profit,
            "status": p.status,
        })

    complete = not missing
    data = {
        "cash": round(port.cash, 2),
        "market_value_priced": round(priced_value, 2),
        "total_value": round(port.total_value, 2) if complete else None,
        "total_pnl_pct": port.total_pnl_pct if complete else None,
        "valuation_complete": complete,
        "missing_codes": missing,
        "positions": rows,
        "count": len(rows),
        "alerts": alerts,
        "exposure": exposure,
        "price_date": price_date,
        "price_basis": "close",
        "expected_date": picks.expected_date(now),
    }
    return envelope.ok(data, as_of=price_date or data["expected_date"])


# ---------------------------------------------------------------- 模拟账户
def sim(account: str | None = None) -> dict:
    from db import store
    from core.paper_account import PaperAccount

    try:
        names = store.list_paper_accounts()
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 模拟账户列表失败: %s", e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")

    name = account
    if name is None:
        # 交易台默认看 AI 主账户（联赛里唯一"人设 + 净值"齐全的那个）；
        # 没有就退手动模拟盘 default，再退第一个 —— 不猜空账户
        for cand in ("ai_value", "default"):
            if cand in names:
                name = cand
                break
        name = name or (names[0] if names else None)
    if name is None:
        return envelope.fail("missing", "无模拟账户")
    if account is not None and account not in names:
        return envelope.fail("invalid", f"无此模拟账户: {account}")

    try:
        state = store.load_paper_state(name)
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 模拟账户读取失败: %s", e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")
    if state is None:
        return envelope.fail("missing", f"账户 {name} 无状态记录")

    acct = PaperAccount.from_state(state)
    prices, price_date, missing = local_closes(list(acct.positions.keys()))

    rows = []
    for code, pos in acct.positions.items():
        px = prices.get(str(code))
        avg = round(pos.avg_cost, 4)
        mv = None if px is None else round(px * pos.quantity, 2)
        rows.append({
            "code": code, "quantity": pos.quantity, "available": pos.available,
            "avg_cost": avg, "price": px, "price_basis": "close" if px is not None else None,
            "available_price": px is not None,
            "market_value": mv, "cost": round(pos.total_cost, 2),
            "pnl": None if mv is None else round(mv - pos.total_cost, 2),
            "pnl_pct": None if (mv is None or not pos.total_cost)
            else round((mv - pos.total_cost) / pos.total_cost * 100, 2),
            "last_buy_date": pos.last_buy_date,
        })

    nav = [n for n in acct.nav][-120:]
    data = {
        "account": name,
        "accounts": names,
        "initial_cash": acct.initial_cash,
        "cash": round(acct.cash, 2),
        "positions": rows,
        "pending_orders": [o.to_dict() for o in acct.orders if o.status == "pending"],
        "recent_orders": [o.to_dict() for o in acct.orders][-30:],
        "nav": nav,
        "nav_date": nav[-1]["date"] if nav else None,
        "total_value": nav[-1]["total_value"] if nav else None,
        "return_pct": nav[-1].get("return_pct") if nav else None,
        "valuation_at_cost": bool(nav and nav[-1].get("at_cost")),
        "missing_codes": missing,
        "price_date": price_date,
        "price_basis": "close",
    }
    return envelope.ok(data, as_of=data["nav_date"] or price_date)


# ---------------------------------------------------------------- 模拟下单（P5 写操作）
_DIRECTIONS = {"买入": "买入", "卖出": "卖出", "buy": "买入", "sell": "卖出"}


def place_order(account: str | None, code: str, direction: str, quantity,
                reason: str = "") -> dict:
    """模拟账户市价单（**T+1 撮合**，最早下一交易日开盘成交）。

    ## 为什么"被拒"不是 error
    数量不足一手 / 现金不足 / 无持仓 / 未知方向 —— 这些都是**业务结果**，请求本身合法。
    返 `ok:true` + `accepted:false` + `reason`，让前端把它显示成红色提示而不是
    "接口坏了"。真正的参数错误（代码不是 6 位数字、数量不是正整数）才返 `invalid`。

    ## 为什么这里不查现价
    `submit_order` 只做现金下限提示与持仓校验，成交价在下一交易日 `settle_open`
    用真实盘口撮合。**现在拿收盘价去"预演成交"会造出一笔并未发生的成交。**
    """
    from core.paper_account import PaperAccount
    from db import store

    code = (code or "").strip()
    if not (len(code) == 6 and code.isdigit()):
        return envelope.fail("invalid", "股票代码必须是 6 位数字")
    d = _DIRECTIONS.get((direction or "").strip())
    if d is None:
        return envelope.fail("invalid", f"未知方向：{direction}（只接受 买入/卖出）")
    try:
        qty = int(quantity)
    except (TypeError, ValueError):
        return envelope.fail("invalid", "数量必须是整数")
    if qty <= 0:
        return envelope.fail("invalid", "数量必须为正")

    try:
        names = store.list_paper_accounts()
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 模拟账户列表失败: %s", e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")
    name = account or ("default" if "default" in names else (names[0] if names else None))
    if name is None:
        return envelope.fail("missing", "无模拟账户")
    if name not in names:
        return envelope.fail("invalid", f"无此模拟账户: {name}")

    with _ORDER_LOCK:
        state = store.load_paper_state(name)
        if state is None:
            return envelope.fail("missing", f"账户 {name} 无状态记录")
        acct = PaperAccount.from_state(state)
        order = acct.submit_order(code=code, direction=d, quantity=qty,
                                  date=datetime.now().strftime("%Y-%m-%d"),
                                  reason=reason or "")
        try:
            store.save_paper_state(name, acct.to_state())
        except Exception as e:                                # noqa: BLE001
            _logger.warning("[desk] 模拟账户落库失败: %s", e)
            return envelope.fail("empty", f"{type(e).__name__}: {e}")

    data = {"account": name, "order": order.to_dict(),
            "accepted": order.status != "cancelled",
            "reason": order.reason or ""}
    return envelope.ok(data, as_of=datetime.now().strftime("%Y-%m-%d"))


# ---------------------------------------------------------------- AI 联赛
def league(now=None) -> dict:
    from core.ai_league import leaderboard

    try:
        board = cache.cached("desk.league", _panel_stamp(), TTL_LEAGUE, leaderboard)
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 联赛榜单失败: %s", e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")

    # 目标仓位读 latest.json（C1 后 market_signal 是对象）；不现算 detect()
    target = {"position_limit": None, "market_state": None, "signal": None,
              "date": None}
    zone = picks.load_latest(now=now)
    if zone.get("ok"):
        sig = (zone.get("data") or {}).get("market_signal") or {}
        if isinstance(sig, dict):
            target = {
                "position_limit": sig.get("position_limit"),
                "market_state": sig.get("market_state"),
                "signal": sig.get("signal"),
                "date": (zone.get("data") or {}).get("date"),
            }

    data = {
        "as_of": board.get("as_of"),
        "groups": board.get("groups") or {},
        "caveat": board.get("caveat") or "",
        "target": target,
    }
    return envelope.ok(data, as_of=data["as_of"])


def league_day(date: str | None = None) -> dict:
    """AI 联赛**某一交易日的决策明细 + 自述复盘 + 当日净值**。

    ## 为什么必须补这个端点（2026-10-26）
    交易台原来只有净值榜：看得到"谁赚了 X%"，看不到"他今天买了什么、为什么"。
    要回答后者只能去翻 `output/` 下的 markdown。而 `ai_decisions`（每人设每票一条）
    与 `ai_reviews`（每人设每日一段）**每天 15:58 都在写**，两张表一直有数据、
    web 层零引用 —— 属于典型的「函数写好了但用户在页面上看不见」。

    ⚠️ 口径说明（必须原样透给前端，不许在页面里另算）：
    · `position_size` 是**决策时的目标仓位**（0~1 或 0~100 两种历史口径都有，
      由写侧决定），页面只做"≤1 则 ×100"的显示换算，不改语义。
    · 随机对照组（`kind=random`）**不调模型**，`model` 为 None、复盘是模板文本。
    · 无决策的日期不编造：`missing` 交前端显示"该日无决策记录"。
    """
    from db import store

    try:
        dates = store.list_ai_decision_dates(limit=60)
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 联赛决策日期读取失败: %s", e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")
    if not dates:
        return envelope.fail("missing", "尚无 AI 决策记录（15:58 决策轮未跑过）")

    day = date if date in dates else dates[0]
    if date and date not in dates:
        # 显式要的一天没有记录 ⇒ 返回该日空结构（不是"最新一天"），否则用户以为看的是那天
        day = date if date else dates[0]

    try:
        decisions = store.list_ai_decisions(signal_date=day)
        reviews = {r["persona"]: r for r in store.list_ai_reviews(trade_date=day)}
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 联赛决策/复盘读取失败(%s): %s", day, e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")

    # 代码 → 名称：用**最新**快照（名称不随日期变化；当日快照可能还没跑）
    name_of: dict = {}
    codes = sorted({d.get("code") for d in decisions if d.get("code")})
    if codes:
        try:
            _, rows = store.load_market_snapshot(codes=codes)
            name_of = {r["stock_code"]: r.get("stock_name") or "" for r in rows}
        except Exception as e:                                # noqa: BLE001
            _logger.warning("[desk] 名称补全失败（不影响决策展示）: %s", e)

    # 当日净值：paper_nav 里的人设行（缺失就 None，**不用初始资金冒充**）
    nav_of: dict = {}
    try:
        conn = store.get_connection()
        try:
            for r in conn.execute(
                    "SELECT name, total_value, return_pct FROM paper_nav WHERE date = ?",
                    (day,)):
                nav_of[r["name"]] = {"total_value": r["total_value"],
                                     "return_pct": r["return_pct"]}
        finally:
            conn.close()
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 当日净值读取失败(%s): %s", day, e)

    # 当日成交：⚠️ 结算口径是 T+1 开盘 ⇒ `day` 的成交对应的是**前一交易日**的决策。
    # 名字必须照实说，不能让用户以为"今天说买就今天成交"。
    fills_of: dict = {}
    try:
        for f in store.list_paper_fills(date=day):
            fills_of.setdefault(f.get("name"), []).append({
                "code": f.get("code"),
                "name": name_of.get(f.get("code")) or "",
                "direction": f.get("direction"),
                "quantity": f.get("quantity"),
                "price": f.get("actual_price"),
            })
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 当日成交读取失败(%s): %s", day, e)

    try:
        from core import ai_personas
        metas = {p["name"]: p for p in ai_personas.load_personas()}
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 人设配置读取失败，退化为用 persona 名: %s", e)
        metas = {}

    by_persona: dict = {}
    for d in decisions:
        by_persona.setdefault(d.get("persona"), []).append({
            "code": d.get("code"),
            "name": name_of.get(d.get("code")) or "",
            "direction": d.get("direction"),
            "advice_status": d.get("advice_status"),
            "position_size": d.get("position_size"),
            "entry": d.get("entry"), "target": d.get("target"),
            "stop": d.get("stop"),
            "order_status": d.get("order_status"),
            "order_reason": d.get("order_reason"),
        })

    # 人设顺序按配置走（LLM 三人设在前、随机组在后），未在配置里的排末尾
    order = [p["name"] for p in (metas.values() or [])]
    seen = [p for p in order if p in by_persona or p in reviews]
    extra = [p for p in list(by_persona) + list(reviews)
             if p and p not in seen]
    rows = []
    for p in seen + sorted(set(extra)):
        meta = metas.get(p) or {}
        rv = reviews.get(p) or {}
        rows.append({
            "persona": p,
            "label": meta.get("label") or p,
            "kind": meta.get("kind") or ("random" if (meta.get("model") is None) else "llm"),
            "model": meta.get("model"),
            "decisions": by_persona.get(p) or [],
            "fills": fills_of.get(p) or [],
            "review": ({"summary": rv.get("summary") or "",
                        "text": rv.get("review_text") or "",
                        "model": rv.get("model"),
                        "created_at": rv.get("created_at")} if rv else None),
            "nav": nav_of.get(p),
        })

    data = {
        "date": day,
        "dates": dates,
        "personas": rows,
        "coverage": {
            "decisions": len(decisions),
            "reviews": len(reviews),
            "personas": len(rows),
            "nav": len(nav_of),
            "fills": sum(len(v) for v in fills_of.values()),
        },
    }
    return envelope.ok(data, as_of=day)


def wake() -> dict:
    """AI 盘中唤醒：**埋了哪些触发条件** + **真的唤醒过几次、做了什么**。

    ## 为什么补这个端点（2026-10-26）
    `ai_wake_conditions`（60 条）与 `ai_wake_events`（26 条）一直在写，web 层零引用。
    用户能看到的只有"AI 每天 15:58 决策一次"，但联赛里 AI 其实会在盘中被
    价格阈值唤醒（`price_below` / `price_above` / `wake_at`），醒来后可能
    `sell` / `pass`（看过但不动作），也可能**根本没跑**（随机对照组）。
    不把这层摊开，用户无法判断"AI 是没动，还是动了但没记录"。

    ## 三条不许含糊的口径
    · `fired=0` 且 `skip_reason` 非空 ⇒ 命中条件但**没执行**，原因原样透出
      （如"随机对照组不跑盘中决策"）。
    · `armed=1` 只是"条件仍生效"，**价位没到就不会有事件行** —— 页面不许显示
      "60 条待触发"当成待办，必须写成"盯盘中"。
    · `action` 是 JSON 字符串（写入侧存的是列表），这里解析成动作名列表；
      解析失败**原样返回字符串**，不吞掉证据。
    """
    import json as _json

    from db import store

    try:
        stats = store.wake_event_stats(limit=60)
        conds = store.list_wake_conditions(armed_only=True, limit=300)
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 唤醒记录读取失败: %s", e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")

    def _acts(raw):
        """`[{"action":"sell",...}]` → `["sell"]`；解析不了就原样留着。"""
        if not raw:
            return []
        try:
            obj = _json.loads(raw)
            if isinstance(obj, list):
                return [str(x.get("action")) for x in obj
                        if isinstance(x, dict) and x.get("action")]
            return []
        except (ValueError, TypeError):
            return [str(raw)[:40]]

    events = []
    for r in stats.get("events") or []:
        events.append({**r, "actions": _acts(r.get("action"))})

    data = {
        "total_events": stats.get("total") or 0,
        "last_date": stats.get("last_date"),
        "by_persona": stats.get("by_persona") or [],
        "skip_reasons": stats.get("skip_reasons") or [],
        "events": events,
        "armed": conds,
        "armed_count": len(conds),
        # 人设中文名：页面显示 `ai_tech` 这种内部 id 用户看不懂
        "labels": {},
    }
    try:
        from core.ai_personas import load_personas
        # ⚠️ `load_personas()` 返回 **list[dict]**（不是 {id: meta}）；
        #    按 dict 解包会静默得到空 labels，页面上就只剩 `ai_tech` 这种内部 id。
        for meta in (load_personas() or []):
            # 字段名是 `name`（不是 id/persona）—— 猜字段名的代价是页面只剩内部 id
            pid = (meta or {}).get("name")
            if pid:
                data["labels"][pid] = (meta or {}).get("label") or pid
    except Exception as e:                                    # noqa: BLE001
        _logger.debug("[desk] 人设表不可用，唤醒面板只显示内部 id: %s", e)

    return envelope.ok(data, as_of=stats.get("last_date"))


# ---------------------------------------------------------------- 阿深值班（P1）
# 这两段的设计约束都来自同一个问题：**"权限开了"和"真的在建仓"是两件事**。
# 只看 `autonomy=L2` 会以为它在工作；实际可能因为池子空 / 额度用尽 / 授权参数
# 不自洽 / 一手成本超上限 而一只都没买，且每种原因的表象都是"没动作"。
# 所以卡片必须把**每一个可能的卡点**单独摊开，而不是给一个"运行中"的绿点。

def _ashen_pools() -> dict:
    """两套自选池。读失败 ⇒ `available=False`，**不许退化成"0 只"**。

    ⚠️ `snapshot()` 是严格语义（读失败抛异常），而 `load_watchlist()` 吞异常。
    页面必须用前者：把"库读挂了"显示成"自选股为空"会让人以为清单被删了。
    """
    from data import watchlist as wl

    out: dict = {}
    for owner in ("ashen", "user"):
        try:
            snap = wl.snapshot(owner)
            rows = snap.get("stocks") or []
            out[owner] = {"available": True, "count": len(rows),
                          "label": snap.get("label") or owner,
                          "rows": [{"code": str(r.get("code") or ""),
                                    "name": r.get("name") or "",
                                    "note": r.get("note") or ""} for r in rows]}
        except Exception as e:                                # noqa: BLE001
            _logger.warning("[desk] %s 自选池读取失败: %s", owner, e)
            out[owner] = {"available": False, "count": None,
                          "label": owner, "rows": []}
    return out


def ashen(day: str | None = None) -> dict:
    """阿深值班卡：授权参数 / 账户 / 两套自选 / 今日轮次。**零联网**。

    ## 三处不许含糊的口径
    · `account.total_value` 缺收盘价时置 `null` + `valuation_complete=false`，
      **不拿"只算到价那几只"的合计数冒充净值**（同 `sim()` 的理由）。
    · `pool_hint` 由后端给出成句结论（如"池子空 ⇒ L2 无标的"），页面不自己拼
      —— 前端拼条件会出现"池子空但页面没提"的情况。
    · `autonomy.violations` 非空说明**参数自相矛盾**，这时建仓权名义上开着、
      实际上每笔都会被拒。必须在卡片最显眼处报红，不能只写"L2 已开启"。
    """
    from core import ai_autonomy, ai_budget, ai_personas, ai_routine
    from db import store

    today = day or datetime.now().strftime("%Y-%m-%d")
    try:
        persona = ai_personas.get_persona(ai_routine.PERSONA_NAME)
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 人设表读取失败: %s", e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")
    if not persona:
        return envelope.fail("missing", "阿深未注册（config/ai_personas.json 缺 ai_ashen）")

    level = ai_autonomy.normalize_level(persona.get("autonomy"))
    params = ai_autonomy.resolve()
    violations = ai_autonomy.validate(params)
    try:
        budget = ai_budget.view(ai_routine.PERSONA_NAME, today)
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 额度视图读取失败: %s", e)
        budget = {}

    # ── 账户（只读：不存在就说 exists=false，**不在 GET 里建账户**）──
    acct_out = {"exists": False, "cash": None, "total_value": None,
                "valuation_complete": False, "missing_codes": [],
                "positions": [], "position_count": 0, "market_value": None,
                "position_pct": None, "pending_orders": 0,
                "price_date": None, "return_pct": None, "initial_cash": None}
    try:
        state = store.load_paper_state(ai_routine.PERSONA_NAME)
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 阿深账户读取失败: %s", e)
        state = None
    if state:
        from core.paper_account import PaperAccount

        acct = PaperAccount.from_state(state)
        prices, price_date, missing = local_closes(list(acct.positions))
        rows, mv_all, priced = [], 0.0, True
        for code, pos in acct.positions.items():
            px = prices.get(str(code))
            mv = None if px is None else round(px * pos.quantity, 2)
            if mv is None:
                priced = False
            else:
                mv_all += mv
            rows.append({"code": str(code), "quantity": pos.quantity,
                         "available": getattr(pos, "available", None),
                         "avg_cost": round(pos.avg_cost, 4), "price": px,
                         "market_value": mv,
                         "pnl_pct": (None if (mv is None or not pos.total_cost)
                                     else round((mv - pos.total_cost)
                                                / pos.total_cost * 100, 2))})
        total_value = round(acct.cash + mv_all, 2) if priced else None
        acct_out.update({
            "exists": True,
            "cash": round(acct.cash, 2),
            "initial_cash": acct.initial_cash,
            "market_value": round(mv_all, 2) if priced else None,
            "total_value": total_value,
            "valuation_complete": priced,
            "missing_codes": missing,
            "positions": rows,
            "position_count": len(rows),
            "position_pct": (None if (total_value is None or not total_value)
                             else round(mv_all / total_value * 100, 2)),
            "pending_orders": sum(1 for o in acct.orders if o.status == "pending"),
            "price_date": price_date,
            "return_pct": (None if total_value is None or not acct.initial_cash
                           else round((total_value - acct.initial_cash)
                                      / acct.initial_cash * 100, 2)),
        })

    pools = _ashen_pools()
    # ⚠️ `count` 可能是 `None`（读失败），不是 0 —— 下面每个分支都必须先看 available
    ashen_n = pools["ashen"]["count"]

    try:
        hist = ai_routine.history(today)
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 值班记录读取失败: %s", e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")

    # 今日候选：解释"池子空 + 有候选却没入池"这个最常见的卡点
    try:
        candidates = ai_routine._today_candidates(today, limit=8)
    except Exception as e:                                    # noqa: BLE001
        _logger.debug("[desk] 今日候选不可用（不影响值班卡）: %s", e)
        candidates = []

    now = datetime.now()
    due = ai_routine.due_slot(now)
    nxt = next((s for s in ai_routine.SLOTS if s > now.strftime("%H:%M")), None)

    if not pools["ashen"]["available"]:
        hint = "阿深自选池读取失败（不是空池）——先查库，此时不要动授权"
    elif ashen_n == 0:
        hint = (f"阿深自选池 0 只 ⇒ L2 建仓没有标的；今日候选 {len(candidates)} 只"
                "可让它自己 watch_add 入池（同轮加池+买入一定会被拒）")
    elif acct_out["positions"] and acct_out["position_pct"] is not None \
            and acct_out["position_pct"] >= params["max_total_pct"] * 100:
        hint = f"总仓 {acct_out['position_pct']:.1f}% 已达上限，只能减不能加"
    else:
        hint = f"池内 {ashen_n} 只、可建仓"

    # 它**自己**挂着的等待条件 —— 与 prompt 里「你当前挂着的等待条件」**同源**
    # （同一个 `_armed_conditions`）。两处各写一遍 SQL 就会出现"页面说它在等
    # 28 元、prompt 里却没有"的错位，而那种错位没人看得见。
    # ⚠️ 读失败 ⇒ `available=False`，**不许退化成 count=0**：前者要查库，
    #    后者是"它说了在等、其实一条都没挂"—— 恰恰是要暴露的敷衍。
    try:
        _armed = ai_routine._armed_conditions(strict=True)
        armed_out = {"available": True, "count": len(_armed), "rows": _armed}
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 阿深等待条件读取失败: %s", e)
        armed_out = {"available": False, "count": None, "rows": []}

    data = {
        "trade_date": today,
        "persona": {"name": persona.get("name"), "label": persona.get("label"),
                    "enabled": bool(persona.get("enabled")),
                    "frequency": persona.get("frequency"),
                    "league_eligible": bool(persona.get("league_eligible"))},
        "armed": armed_out,
        "autonomy": {
            "level": level,
            "raw": persona.get("autonomy"),
            # 非法级别回落 L0 时必须能看出来（否则"权限被关"会像"它不想买"）
            "fallback": str(persona.get("autonomy") or "").strip().upper() != level,
            "desc": ai_autonomy.LEVEL_DESC.get(level, ""),
            "params": params,
            "describe": ai_autonomy.describe(params),
            "violations": violations,
            "consistent": not violations,
        },
        "budget": {k: budget.get(k) for k in ("routine", "wake", "decision_round")
                   if k in budget},
        "account": acct_out,
        "pools": pools,
        "pool_hint": hint,
        "candidates": candidates,
        "today": {**hist, "due_slot": due, "next_slot": nxt,
                  "now": now.strftime("%H:%M")},
    }
    return envelope.ok(data, as_of=acct_out.get("price_date") or today)


# ---------------------------------------------------------------- 和阿深说话（2026-10-28）
# 为什么要有这个端点：房间里的阿深原来是"点一下说 5 句硬编码台词"，用户想追问
# "为什么是 28 块""你凭什么是 28"没有地方问 —— 只能绕到左边通用助手，由助手
# 转述"它今天跑了 6 轮"。**转述不是互动**：人称变了、上下文丢了、每问一句都要
# 重新介绍它是谁。
#
# 三条硬约束：
# 1. **只读**。这个端点不出动作、不写主链表、不扣 routine 额度。让它在这里
#    接受"帮我买入"就是把下单通道从"每轮一次授权判定"降级成了"聊天框"。
# 2. **事实只来自 `ashen()`**（同一份实现）。另拼一份 prompt 口径必然漂 ——
#    页面显示"池 3 只"而它说"池 5 只"是最难查的一类错。
# 3. **不许估算**。没有的数据就说看不到。它是交易员，编一个价格比不回答危险得多。

_CHAT_RULES = """【怎么回答】
- 用第一人称，像交易员跟老板当面说话：直接、具体、带数字。≤120 字，不要小标题、不要分点列表
- 只依据上面给的事实。没给你的数据就直说"我这边看不到 XXX"，**绝对不许估算或回忆**
- 你只是在回答，**不能承诺下单、不能代替下一轮决策**。老板让你买什么，你说清"这要等下一轮按纪律判定"
- 老板如果问和你判断有关的事，先想一眼你上面那 6 轮是怎么说的，别自相矛盾；确实改主意了就说清为什么改
- **你有脸，表情会跟着你变**：在回答的最后一行单独写一行情绪标记，
  格式固定为 `【情绪】happy`、`【情绪】sad`、`【情绪】think`、`【情绪】neutral` 四选一。
  讲成交/进展用 happy；讲没做到、遗憾用 sad；在权衡、在等条件用 think；其余用 neutral。
  这一行会被程序读走、不显示给老板，但它**真的会改你房间里的表情**。
  所以别说"我变不了表情"——那是旧的自我认知，现在你可以
"""

# 情绪标记：`【情绪】happy` 或 `[情绪] happy`。四选一，非法值直接丢弃（走代码兜底）。
_MOOD_RE = re.compile(r"[【\[]\s*情绪\s*[】\]]\s*(happy|sad|think|neutral)", re.I)


def _guess_mood(text: str) -> str:
    """代码兜底表情。**只在模型没给标记时用**，来源会标成 fallback。

    为什么要兜底而不是默认 neutral：一句"今天没动"如果配一张微笑脸，
    比没有表情更违和。这里只做粗判，不追求语义准确——
    精确的那份由模型自己给（见 `_CHAT_RULES`）。
    """
    t = str(text or "")
    if any(w in t for w in ("没动", "没买", "没成交", "没做成", "遗憾", "抱歉", "可惜")):
        return "sad"
    if any(w in t for w in ("成交", "建仓", "买了", "加池", "挂上", "触发", "完成")):
        return "happy"
    if any(w in t for w in ("等", "观察", "权衡", "考虑", "看情况", "不一定")):
        return "think"
    return "neutral"


def _split_mood(raw: str) -> tuple[str, str, str]:
    """把情绪标记从正文里剥出来。返回 `(正文, mood, 来源)`。

    来源记 `model` / `fallback`：要分清"他自己说要开心"与"代码替他选的表情"——
    前者是表达，后者是渲染兜底，混着看就没法判断"给标记"这条指令有没有生效。
    """
    text = str(raw or "").strip()
    m = _MOOD_RE.search(text)
    if m:
        mood = m.group(1).lower()
        body = (text[:m.start()] + text[m.end():]).strip()
        return body, mood, "model"
    return text, _guess_mood(text), "fallback"


def ashen_chat(question: str) -> dict:
    """和阿深对话（第一人称，只读）。返回 `{question, answer, as_of, facts_used}`。"""
    from core import ai_personas

    q = str(question or "").strip()
    if not q:
        return envelope.fail("invalid", "问题为空")
    if len(q) > 600:
        q = q[:600]

    env = ashen()
    if not env.get("ok"):
        # 阿深自己的状态都读不出来，就别假装能对话（回一个编出来的答案更糟）
        return env
    d = env["data"]
    acct = d.get("account") or {}
    today = d.get("today") or {}
    pools = d.get("pools") or {}
    au = d.get("autonomy") or {}
    armed = d.get("armed") or {}

    # 6 轮判断：按时间序拼成一条线（跳过轮没有结论，不占位置）
    trail = []
    for r in sorted((today.get("runs") or []),
                    key=lambda x: str(x.get("created_at") or "")):
        s = str(r.get("summary") or "").strip()
        if s:
            trail.append(f"{r.get('slot')} {s}")
    trail_txt = " ｜ ".join(trail) if trail else "（今天还没有判断）"

    def _pool(owner: str) -> str:
        p = pools.get(owner) or {}
        if p.get("available") is False:
            return "读取失败（不是空的）"
        rows = p.get("rows") or []
        if not rows:
            return "（空）"
        return "、".join(f"{r.get('code')} {r.get('name') or ''}".strip() for r in rows[:12])

    pos = acct.get("positions") or []
    pos_txt = ("、".join(f"{p.get('code')} {p.get('quantity')}股 成本{p.get('avg_cost')}"
                         for p in pos[:10]) if pos else "空仓")
    def _cond(c: dict) -> str:
        """一条等待条件说成人话：`600887 price_below 28`。"""
        v = "—"
        if c.get("wake_at"):
            v = f"{c['wake_at']} 定时"
        elif c.get("threshold") is not None:
            v = f"{float(c['threshold']):g}"
        elif c.get("threshold_pct") is not None:
            v = f"{float(c['threshold_pct']) * 100:g}%"
        return f"{c.get('code')} {c.get('kind')} {v}"

    if armed.get("available") is False:
        armed_txt = "读取失败（不是没挂）"
    elif not (armed.get("rows") or []):
        armed_txt = "（没有）—— 说在等什么的时候其实没挂条件"
    else:
        armed_txt = "、".join(_cond(c) for c in armed["rows"])

    # 时段剧本：让"现在"不只是一个时间戳。
    # 同一句"你在想什么"，09:40 该答开盘怎么定性、14:40 该答要不要过夜。
    # 不告诉它这一点，任何时刻问都得到同一段泛泛的话——这正是"互动性差"的一半。
    # 与值班轮共用 `ai_routine.phase_of`（**同源**）：两处各写一套时段表，
    # 就会出现"值班说在等尾盘、对话说还在开盘"的错位，而这种错位没人看得见。
    from core.ai_routine import phase_of
    ph = phase_of(datetime.now())

    prompt = f"""你是阿深，一名 A 股交易员。老板正在你的房间外问你话。

【现在】{today.get('now') or ''}，{d.get('trade_date')}
【这个时段】{ph[1]} —— 此时本该关注：{ph[2]}
    答"在等什么 / 为什么不动"时请按这个时段的角度说，别泛泛而谈；
    时段边界：{ph[4]}
【你的账户】净值 {acct.get('total_value') if acct.get('total_value') is not None else '读不到'}，
现金 {acct.get('cash') if acct.get('cash') is not None else '读不到'}，
总仓 {acct.get('position_pct') if acct.get('position_pct') is not None else '—'}%
【你的持仓】{pos_txt}
【你的自选池】{_pool('ashen')}
【老板的自选】{_pool('user')}   ← 只读，你不能改
【建仓权】{au.get('level')}（{au.get('desc') or ''}）· {au.get('describe') or ''}
【今天的判断（你自己说的，按时间）】{trail_txt}
【你挂着的等待条件】{armed_txt}

【老板问】{q}

{_CHAT_RULES}"""

    try:
        from core import ai_personas as _ps
        from llm.deepseek_client import DeepSeekClient, TEMP_ANALYTICAL, model_override
        persona = _ps.get_persona("ai_ashen") or {}
        # purpose 写成 f-string 而不是字面量 "ai.ai_ashen.chat"：与
        # `ai_daily_round` / `ai_wake` / `ai_routine` 三处保持**同一种形态**，
        # 结构守护才能把它扫成 `ai.*.chat` 并对应上族键。
        # 用字面量时它会变成一个"只在 web 层出现、族键匹配不到"的孤儿键
        # （扫描范围是 core/llm/data/ui，不含 web）。
        p_name = str(persona.get("name") or "ai_ashen")
        client = DeepSeekClient()
        with model_override(persona.get("model")):
            raw = client.call_api([{"role": "user", "content": prompt}],
                                  temperature=TEMP_ANALYTICAL, max_tokens=400,
                                  purpose=f"ai.{p_name}.chat")
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 阿深对话失败: %s: %s", type(e).__name__, e)
        return envelope.fail("proxy", f"{type(e).__name__}: {str(e)[:120]}")

    answer = str(raw or "").strip()
    if not answer:
        return envelope.fail("empty", "模型返回空文本")

    # 情绪标记剥出来单独传：正文给气泡，mood 给房间里的脸。
    # 两者**不混在一个字段**——否则前端要么把标记显示给用户看，
    # 要么自己再切一次（两处各写一遍正则就会漂移）。
    answer, mood, mood_src = _split_mood(answer)

    return envelope.ok({"question": q, "answer": answer[:800],
                        # 房间里的角色靠它换表情；`mood_source` 让"给标记"这条指令
                        # 有没有生效可查（model = 他自己给的，fallback = 代码兜底）
                        "mood": mood, "mood_source": mood_src,
                        "as_of": d.get("trade_date"),
                        # 让页面/审计能看到"它答的时候手上有哪些事实"
                        "facts": {"rounds": len(trail), "positions": len(pos),
                                  "armed": len(armed.get("rows") or [])}})


def wake_ashen(dry_run: bool = True) -> dict:
    """手动唤醒一轮（**默认演练**：只判定、不落单）。

    ## 为什么默认 dry_run
    页面上的按钮和"盯盘自动轮"共用同一段判定代码，但**触发者不同**：自动轮是
    系统按时刻表跑，手动轮是有人在看。让人按一下就可能真金白银建仓，等于把
    "确认"这一步删掉了。所以要真下单必须显式 `dry_run=false`，页面另加二次确认。

    ## 三条纪律
    · **不改额度语义**：手动轮同样 `consume(kind="routine")`，额度不足照拒
      （否则"多按几次"就是免费提额）。
    · **非交易时段不假装能建仓**：`execute=true` 而此刻不在交易时段时，
      委托仍是 T+1 挂单 —— 这一点在 `note` 里说明，不静默。
    · 取不到行情 ⇒ `accepted=false` + 原因，**不返 ok:false**：这是行情源的
      业务结果，不是接口故障。
    """
    from core import ai_routine

    dry = True if dry_run is None else bool(dry_run)
    now = datetime.now()

    # ── 需要行情的代码集合：**盯盘范围的唯一定义处**（两套自选 ∪ 今日候选 ∪ 持仓）──
    # ⚠️ 不许在这里另抄一份"自选池 ∪ 持仓"：漏掉今日候选的后果是"入池的票没有
    #    名称/没有价格"，而 watch_add 的主要来源恰恰就是候选（实测踩到：三条
    #    入池记录 name 全空）。`attention_codes()` 是对应函数，`_default_codes()` 也用它。
    codes: list = []
    try:
        from data import watchlist as wl

        codes = [str(c) for c in wl.attention_codes() if c]
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 关注池读取失败，退化为自选池 ∪ 持仓: %s", e)
        pools = _ashen_pools()
        codes = [r["code"] for r in pools["ashen"]["rows"]] + \
                [r["code"] for r in pools["user"]["rows"]]
    try:
        from db import store

        state = store.load_paper_state(ai_routine.PERSONA_NAME)
        if state:
            from core.paper_account import PaperAccount
            codes += [str(c) for c in PaperAccount.from_state(state).positions]
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 手动唤醒：持仓读取失败（只按关注池取行情）: %s", e)
    codes = [c for c in dict.fromkeys(codes) if c]

    if not codes:
        return envelope.ok({"accepted": False, "dry_run": dry,
                            "reason": "两套自选池都为空、无持仓 ⇒ 没有可判的标的",
                            "run": None}, as_of=None)

    try:
        from data.realtime import get_realtime_quotes

        quotes = get_realtime_quotes(codes, force_refresh=True)
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[desk] 手动唤醒取行情失败: %s", e)
        quotes = {}
    # ⚠️ `{}` 与"部分缺失"要分开：前者是行情源不可达，后者是停牌/退市
    if not quotes:
        return envelope.ok({"accepted": False, "dry_run": dry,
                            "reason": f"取不到实时行情（{len(codes)} 只全部无返回），"
                                      "未跑判定",
                            "run": None}, as_of=None)

    with _ORDER_LOCK:                     # 与手动下单互斥：两处都会读改写同一个 state
        run = ai_routine.run_due(quotes, now=now, force=True, execute=not dry)
    if not isinstance(run, dict):
        return envelope.fail("empty", "值班轮次返回了非 dict 结果")

    accepted = bool(run.get("llm_called"))
    data = {
        "accepted": accepted,
        "dry_run": dry,
        "reason": ("" if accepted else str(run.get("skipped") or "未到点/未获批")),
        "quotes_count": len(quotes),
        "note": ("演练：只判定、不落单" if dry else
                 "真下单：委托按 T+1 下一交易日开盘撮合"
                 + ("" if _in_session(now) else "；此刻非交易时段，属按计划的挂单")),
        "run": run,
    }
    return envelope.ok(data, as_of=now.strftime("%Y-%m-%d %H:%M"))


def _in_session(now: datetime) -> bool:
    """是否在交易时段（与盯盘同一判据，避免页面另立一套）。"""
    try:
        from core.watcher import in_trading_session
        return bool(in_trading_session(now))
    except Exception as e:                                    # noqa: BLE001
        _logger.debug("[desk] 交易时段判断不可用: %s", e)
        return False
