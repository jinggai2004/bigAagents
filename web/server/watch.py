# -*- coding: utf-8 -*-
"""`/api/watch/*` —— 看盘五区（P2）。

## 零外呼在这里必然破例，守的是另一条线
P1 的"端点零外呼"对看盘不成立：指数点位、自选价、外盘**必须联网**。
所以这里守的是「**每个外呼都有超时与显式降级**」：

| 区 | 取数 | 外呼 | 失败时的样子 |
|---|---|---|---|
| 指数 | `realtime.get_index_quotes` | 是（10s timeout、腾讯/新浪双源、源级熔断） | 用本地日线最后收盘顶位，标 `basis:"close"` |
| 自选 | `realtime.get_realtime_quotes` | 是（同上） | 该只 `price:null` + `available:false`，**不返 0** |
| 行业资金 | `fund_flow.get_stored_industry_flow` | **否**（读落库快照） | `empty` + 说明"盘后 15:30 落库" |
| 外盘 | `us_market.get_overnight_us` | 是（自带 30min 缓存，失败返 unavailable） | `proxy` + 上游 note |
| 异动 | `store.get_recent_alerts` | **否**（读 alerts 表） | 空列表（"今日暂无"是正常态，不是错误） |

## 三条口径
1. `spark` / 行业 / 异动 **一律读本地**：不因为看盘要画图就去打行情源；
2. 指数的 `stale` 与 `lag_trading_days` 用 `picks.expected_date()` **同一口径**
   ——不另写一套"今天/昨天"判断（项目踩过：基准用日历末元素让闸门恒真）；
3. 离线（`BIGA_OFFLINE=1`）**不去等超时**，直接给 `offline` 或纯本地快照。
"""
from __future__ import annotations

import logging
from datetime import date, datetime

from web.server import cache, envelope, picks

_logger = logging.getLogger(__name__)

# 指数面板 = 三大 + 沪深300。
# ⚠️ 不摆科创50：本地 `data/panel/index/` 没有 sh000688（`get_index_bars` 返回 None），
#    放上去就是一张永远空着的卡 —— 缺就是缺，不拿别的指数冒充。
INDEX_PANEL = (
    ("sh000001", "上证指数"),
    ("sz399001", "深证成指"),
    ("sz399006", "创业板指"),
    ("sh000300", "沪深300"),
)

SPARK_POINTS = 20          # sparkline 取最近 20 个交易日（≈一个月）
TOP_INDUSTRY = 8           # 行业资金榜条数（流入榜/流出榜各 8）


def _offline() -> bool:
    try:
        from data.offline_gate import is_offline
        return bool(is_offline())
    except Exception as e:                     # noqa: BLE001
        _logger.debug("[watch] 离线闸门不可用，按在线处理: %s: %s", type(e).__name__, e)
        return False


def _today() -> str:
    return date.today().isoformat()


# ================================================================
# 指数
# ================================================================
def indices(now: datetime | None = None, *, max_stale: float = 0.0) -> dict:
    """四指数：实时点位（可用时）+ 本地日线 sparkline（始终有）。

    Args:
        max_stale: >0 时走 SWR（过期给旧值 + 后台刷新）。**只有首屏该用**：
            这里唯一的外呼是实时报价（冷启 ≈1.2s），而首屏没人盯着指数等 1 秒。
            看盘页传 0（默认）——用户主动打开看盘页就是要"现在"的价格。
            旧值的年龄由 `indices_age()` 报给前端，不许把 40 秒前的价当实时。
    """
    if now is not None:                        # 测试注入 ⇒ 绕过缓存
        return _indices(now)
    if max_stale > 0:
        return cache.swr("watch_indices", _today(), 15,
                         lambda: _indices(datetime.now()), max_stale=max_stale)
    return cache.cached("watch_indices", _today(), 15, lambda: _indices(datetime.now()))


def indices_age():
    """当前指数报价在进程内的年龄（秒）；None ⇒ 不是从缓存来的。"""
    return cache.age_of("watch_indices", _today())


def _indices(now: datetime) -> dict:
    live, live_err = {}, None
    if _offline():
        live_err = "offline"
    else:
        try:
            from data import realtime
            # ⚠️ 不用 `get_index_quotes()`：它只覆盖 `INDEX_CODES` 那三大，
            #    沪深300 会静默缺席（第一版就是这样，sh000300 一直显示 basis:"close"）。
            #    一次批量取四个，反而只有一次外呼。
            live = realtime.get_realtime_quotes([s for s, _ in INDEX_PANEL]) or {}
        except Exception as e:                 # noqa: BLE001
            _logger.warning("[watch] 指数实时报价失败: %s: %s", type(e).__name__, e)
            live_err = "proxy"

    expected = picks.expected_date(now)
    rows, dates = [], []
    for symbol, name in INDEX_PANEL:
        q = live.get(symbol) or {}
        local = _local_spark(symbol)
        price = q.get("price")
        basis = "live" if price is not None else ("close" if local["ok"] else None)
        if price is None:
            price = local.get("last_close")
        last_local = local.get("last_date")
        lag = picks.trading_lag(last_local, expected) if last_local else None
        rows.append({
            "symbol": symbol,
            "name": name,
            "price": price,
            "basis": basis,                    # live / close / None
            "change": q.get("change"),
            "change_pct": q.get("change_pct"),
            "quote_time": q.get("quote_time"),
            "source": q.get("source"),
            "amount_wan": q.get("amount_wan"),   # 腾讯口径：万元
            "stale": q.get("stale"),           # 缓存年龄（realtime 语义）
            "data_lagging": q.get("data_lagging"),  # 数据日期落后
            "spark": local.get("close") or [],
            "spark_dates": local.get("dates") or [],
            "local_last_date": last_local,
            "lag_trading_days": lag,
        })
        if last_local:
            dates.append(last_local)

    as_of = max(dates) if dates else None
    data = {
        "indices": rows,
        "live": live_err is None,
        "expected_date": expected,
        "as_of": as_of,                        # 本地日线最新日（sparkline 的日期）
        "note": _index_note(live_err, rows),
    }
    return envelope.ok(data, as_of=as_of)


def _index_note(live_err: str | None, rows: list[dict]) -> str:
    if live_err == "offline":
        return "离线模式：无实时报价，显示本地日线最后收盘"
    if live_err == "proxy":
        return "行情源不可达：显示本地日线最后收盘"
    stale = [r["name"] for r in rows if r.get("stale")]
    if stale:
        return "过期缓存：" + "、".join(stale)
    lagging = [r["name"] for r in rows if r.get("data_lagging")]
    if lagging:
        return "行情日期落后：" + "、".join(lagging)
    return ""


def _local_spark(symbol: str) -> dict:
    """本地日线最近 N 根。**不联网**（`get_index_bars` 只读 parquet）。"""
    try:
        from data.index_bars import get_index_bars
        df = get_index_bars(symbol)
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[watch] 本地指数读取失败 %s: %s: %s", symbol, type(e).__name__, e)
        return {"ok": False}
    if df is None or df.empty:
        return {"ok": False}
    tail = df.tail(SPARK_POINTS)
    try:
        closes = [round(float(x), 2) for x in tail["close"].tolist()]
        dates = [str(x)[:10] for x in tail["date"].tolist()]
    except (KeyError, TypeError, ValueError) as e:
        _logger.warning("[watch] 本地指数列异常 %s: %s", symbol, e)
        return {"ok": False}
    if not closes:
        return {"ok": False}
    return {"ok": True, "close": closes, "dates": dates,
            "last_close": closes[-1], "last_date": dates[-1]}


# ================================================================
# 自选股
# ================================================================
def watchlist(now: datetime | None = None, owner: str = "user") -> dict:
    """某轨自选（我的 / 阿深的）+ 两轨条数。

    ⚠️ 缓存身份**必须带 owner**：不带的话，先看"我的自选"再切"阿深自选"，
    第二次会命中第一次留下的缓存 ⇒ 两栏显示同一份数据，而且看不出任何异常
    （价格都是对的，只是根本不是那套清单）。这是"缓存键必须带作用域"的典型。
    """
    owner = _owner(owner)
    if now is not None:
        return _watchlist(now, owner)
    return cache.cached("watch_list", (owner, _today()), 15,
                        lambda: _watchlist(datetime.now(), owner))


def _owner(v) -> str:
    """owner 白名单（`user` / `ashen`），非法值回退 `user` 并记 error。"""
    try:
        from data.watchlist import normalize_owner
        return normalize_owner(v)
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[watch] owner 规范化失败，按 user 处理: %s", e)
        return "user"


def _watchlist(now: datetime, owner: str) -> dict:
    """⚠️ 这里直读 `db.store` 而不是 `watchlist.load_watchlist`：后者为了不打断
    新闻采集会吞异常返回 `[]`，而页面必须区分"真的是空清单"和"库读挂了"
    （前者正常展示，后者要显式报错，否则用户以为自选被删了）。
    """
    try:
        from data import watchlist as wl
        snap = wl.snapshot(owner)
        stocks = snap["stocks"] or []
        counts = snap["counts"]
        label = snap["label"]
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[watch] 自选股读取失败 owner=%s: %s: %s",
                        owner, type(e).__name__, e)
        return envelope.fail("empty", f"{type(e).__name__}")

    if not stocks:
        return envelope.ok({"owner": owner, "owner_label": label, "counts": counts,
                            "rows": [], "total": 0, "available": 0,
                            "live": not _offline(), "stale_codes": [],
                            "note": "自选股为空"}, as_of=None)

    quotes, live_err = {}, None
    if _offline():
        live_err = "offline"
    else:
        try:
            from data import realtime
            quotes = realtime.get_realtime_quotes(
                [s.get("code") for s in stocks if s.get("code")]) or {}
        except Exception as e:                 # noqa: BLE001
            _logger.warning("[watch] 自选行情获取失败: %s: %s", type(e).__name__, e)
            live_err = "proxy"

    rows, stale_codes = [], []
    for s in stocks:
        code = s.get("code") or ""
        q = quotes.get(code) or {}
        if not q:
            rows.append({"code": code, "name": s.get("name") or "", "note": s.get("note") or "",
                         "source": s.get("source") or "", "added_at": s.get("added_at") or "",
                         "price": None, "change": None, "change_pct": None,
                         "amount_wan": None, "turnover_rate": None, "quote_time": None,
                         "available": False, "stale": None})
            continue
        if q.get("stale"):
            stale_codes.append(code)
        rows.append({
            "code": q.get("code") or code,
            "name": s.get("name") or q.get("name") or code,
            "note": s.get("note") or "",
            "source": s.get("source") or "",
            "added_at": s.get("added_at") or "",
            "price": q.get("price"),
            "change": q.get("change"),
            "change_pct": q.get("change_pct"),
            "high": q.get("high"),
            "low": q.get("low"),
            "amount_wan": q.get("amount_wan"),
            "turnover_rate": q.get("turnover_rate"),
            "quote_time": q.get("quote_time"),
            "age_seconds": q.get("age_seconds"),
            "stale": q.get("stale"),
            "data_lagging": q.get("data_lagging"),
            "available": True,
        })

    data = {
        "owner": owner,
        "owner_label": label,
        "counts": counts,                      # 两轨条数（本地 SQL，零外呼）⇒ tab 上直接显示
        "rows": rows,
        "total": len(rows),
        "available": sum(1 for r in rows if r.get("available")),
        "live": live_err is None,
        "stale_codes": stale_codes,
        "note": {"offline": "离线模式：无实时报价",
                 "proxy": "行情源不可达"}.get(live_err or "", ""),
    }
    return envelope.ok(data, as_of=_today())


# ================================================================
# 行业资金（读落库快照，零外呼）
# ================================================================
# ================================================================
# 自选股写入（页面直接编辑）
# ================================================================
WL_ACTIONS = ("add", "remove", "update")


def _a_share_code(code: str) -> tuple[bool, str]:
    """6 位代码 + A 股代码段校验。返回 (是否通过, 不通过的原因)。

    ⚠️ 为什么要挡非 A 股：`900/200` 是 B 股、其它乱串根本不是股票。放进去之后
    行情源永远取不到价，页面就一直显示"无报价"，而人以为是行情挂了。
    判定走 `core.scheduler.is_a_share_code`（结构性判断、不依赖清单，清单拿不到也永远可用），
    但**延迟 import**：web 进程不该为了一个校验把整个调度器拉进来。
    """
    c = str(code or "").strip()
    if len(c) != 6 or not c.isdigit():
        return False, "股票代码须为 6 位数字"
    try:
        from core.scheduler import is_a_share_code
        ok = is_a_share_code(c)
    except Exception as e:                     # noqa: BLE001
        # 拿不到判定器就退到"6 位数字"，不因为校验器坏了而拒绝正常写入
        _logger.warning("[watch] A 股代码判定器不可用，退化为仅校验位数: %s", e)
        return True, ""
    if not ok:
        return False, f"{c} 不是 A 股代码（B 股/其它市场加进来永远取不到报价）"
    return True, ""


def _guess_name(code: str) -> str:
    """用行情源补一次名称。失败返回空串——**名字缺失不该阻止写入**。"""
    if _offline():
        return ""
    try:
        from data import realtime
        q = realtime.get_realtime_quotes([code]) or {}
        return str((q.get(code) or {}).get("name") or "")
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[watch] 补名称失败 %s: %s: %s", code, type(e).__name__, e)
        return ""


def edit_watchlist(action: str, owner: str = "user", code: str = "",
                   name=None, note=None) -> dict:
    """自选股增删改（**人在页面上直接点**，不走 AI 的两阶段确认）。

    ## 为什么要和 `tool_manage_watchlist` 分成两条通道
    那个是**模型调**的：模型会自作主张改清单，所以必须先 preview、把差异念给你听、
    你点头后才 commit。这里是**你自己点按钮**——确认动作就是点击本身，
    再套一层 preview 只会让人以为"点了没反应"。两条通道写同一张表，谁先谁后都一致。

    ## owner=ashen 也允许写，但标 `source="manual"`
    阿深那份清单原本只有它的研究轮能改（为了保留"它为什么选这只"的归因）。
    人工介入时写 `manual`，页面把两种来源分开显示——**不是**禁止人改，
    而是让"谁改的"可查。删掉阿深的一只，当天它的决策记录也不会被篡改。

    ## 字段语义（沿用 `data.watchlist.update_stock`）
    `name=None` 不改 · `name=""` = 不改（股票名不该被清空）· `note=None` 不改 ·
    `note=""` = **清空备注**。
    """
    from data import watchlist as wl

    act = str(action or "").strip().lower()
    if act not in WL_ACTIONS:
        return envelope.fail("invalid", f"未知 action：{action}（可用 add/remove/update）")
    own = _owner(owner)
    code_s = str(code or "").strip()

    if act in ("add", "remove", "update"):
        ok, why = _a_share_code(code_s)
        if not ok:
            return envelope.fail("invalid", why)

    try:
        if act == "add":
            name_s = str(name or "").strip()
            note_s = str(note or "").strip()
            if not name_s:
                # 名称留空时补一次（页面随即就能显示名字，而不是一串代码）
                name_s = _guess_name(code_s)
            # source 一律 manual：这条通道只由人触发（阿深自己的写入走 ai_routine，
            # 那里写 ashen）。页面据此区分"我改的"和"它自己加的"。
            ok, msg = wl.add_stock(code_s, name_s, note_s, owner=own, source="manual")
        elif act == "remove":
            ok, msg = wl.remove_stock(code_s, owner=own)
        else:
            name_arg = None if name is None or str(name).strip() == "" else str(name).strip()
            note_arg = None if note is None else str(note)
            if name_arg is None and note_arg is None:
                return envelope.fail("invalid", "update 需要 name 或 note")
            ok, msg = wl.update_stock(code_s, name_arg, note_arg, owner=own)
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[watch] 自选写入失败 %s/%s/%s: %s: %s",
                        act, own, code_s, type(e).__name__, e)
        return envelope.fail("error", f"{type(e).__name__}: {e}")

    if not ok:
        return envelope.fail("invalid", msg)

    # ⚠️ 写成功必须让缓存失效：`watch_list` 的 ttl 是 15s，不失效的话紧接着的
    #    读会命中旧清单 ⇒ 页面显示"改了但没变"，而价格全对、接口也 ok:true。
    try:
        cache.invalidate("watch_list")
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[watch] 自选缓存失效失败（15s 内可能读到旧清单）: %s", e)

    return envelope.ok({"action": act, "owner": own, "code": code_s, "message": msg})


def industry(now: datetime | None = None) -> dict:
    if now is not None:
        return _industry()
    return cache.cached("watch_industry", _today(), 600, _industry)


def _industry() -> dict:
    try:
        from data.fund_flow import get_stored_industry_flow
        data_date, df = get_stored_industry_flow()
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[watch] 行业资金快照读取失败: %s: %s", type(e).__name__, e)
        return envelope.fail("empty", f"{type(e).__name__}")

    if df is None or df.empty:
        return envelope.fail("missing", "行业资金快照无数据（盘后 15:30 落库）")

    try:
        top = df.head(TOP_INDUSTRY)
        bottom = df.tail(TOP_INDUSTRY).iloc[::-1]
        rows_in = [{"industry": str(r["行业"]), "net_yi": _num(r.get("净额")),
                    "change_pct": _num(r.get("涨跌幅")), "leader": r.get("领涨股") or ""}
                   for _, r in top.iterrows()]
        rows_out = [{"industry": str(r["行业"]), "net_yi": _num(r.get("净额")),
                     "change_pct": _num(r.get("涨跌幅")), "leader": r.get("领涨股") or ""}
                    for _, r in bottom.iterrows()]
    except (KeyError, TypeError, ValueError) as e:
        _logger.warning("[watch] 行业资金表列异常: %s: %s", type(e).__name__, e)
        return envelope.fail("empty", "快照列不符合契约")

    data = {"data_date": data_date, "total": int(len(df)),
            "top_in": rows_in, "top_out": rows_out}
    return envelope.ok(data, as_of=data_date)


def _num(v):
    """NaN/None ⇒ None（缺就是缺，不返 0）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, 2)     # NaN 自己不等于自己


# ================================================================
# 隔夜外盘
# ================================================================
def us(now: datetime | None = None) -> dict:
    if now is not None:
        return _us()
    return cache.cached("watch_us", _today(), 1800, _us)


def _us() -> dict:
    if _offline():
        return envelope.fail("offline", "离线模式不取外盘")
    try:
        from data.us_market import get_overnight_us
        d = get_overnight_us()
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[watch] 外盘获取异常: %s: %s", type(e).__name__, e)
        return envelope.fail("proxy", f"{type(e).__name__}")

    if not d or d.get("unavailable") or not d.get("indexes"):
        return envelope.fail("proxy", (d or {}).get("note") or "外盘源不可用")

    rows = [{"name": r.get("name"), "close": r.get("close"),
             "change_pct": r.get("change_pct"), "level": r.get("level"),
             "level_text": r.get("level_text")} for r in d["indexes"]]
    data = {"rows": rows, "data_date": d.get("data_date"), "days_ago": d.get("days_ago"),
            "level": d.get("level"), "level_text": d.get("level_text"),
            "note": d.get("note") or ""}
    return envelope.ok(data, as_of=d.get("data_date"))


# ================================================================
# 盘中异动（读 alerts 表，零外呼）
# ================================================================
def alerts(limit: int = 30, trade_date: str | None = None, now: datetime | None = None) -> dict:
    """当日异动。空列表是**正常态**（非交易日/无命中），不是错误。"""
    day = trade_date or _today()
    if now is not None or trade_date:
        return _alerts(limit, day)
    return cache.cached("watch_alerts", (day, limit), 30, lambda: _alerts(limit, day))


def _alerts(limit: int, day: str) -> dict:
    try:
        from db import store
        rows = store.get_recent_alerts(limit=limit, trade_date=day) or []
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[watch] 异动记录读取失败: %s: %s", type(e).__name__, e)
        return envelope.fail("empty", f"{type(e).__name__}")

    items = [{"time": (r.get("created_at") or "")[11:19],
              "level": r.get("level"), "code": r.get("code"),
              "name": r.get("name") or "", "rule": r.get("rule"),
              "message": r.get("message"), "price": r.get("price"),
              "change_pct": r.get("change_pct")} for r in rows]
    return envelope.ok({"date": day, "rows": items, "total": len(items)}, as_of=day)
