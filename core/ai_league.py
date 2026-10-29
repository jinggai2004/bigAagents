# -*- coding: utf-8 -*-
"""多 AI 交易联赛 · 结算与榜单（S6，2026-10-14）。

三件事：
1. **结账**：日频决策 T+1 开盘成交（`settle_open`）；盘中唤醒的卖出按
   **扫描时刻最新价 + 滑点**立即成交（口径 2，无分钟线数据源的唯一可得价）。
   ⚠️ 口径 2 必须**同时记录触发价与实际成交价** —— 否则无法事后检验高估程度。
   另：`settle_at_close`（**加速确认**，用手动/盘后）按**当日收盘价**把未成交买入撮合成持仓，
   供"先跑一轮立刻看到持仓/净值"；它**不是**正式口径，返回值带 `mode="close_confirm"`。
2. **净值**：`mark_to_market` + 落库（`paper_nav`）。
3. **榜单**：随机对照组分位 + 样本门槛 + 按决策频率分组 + caveat。
   ⚠️ 项目共识：**不用"胜率"当唯一判据** —— 用「相对随机组的超额 + 分位 + 样本充分性」。

⚠️ 诚实边界（写死在返回值里，页面必须展示）：
· 日内成交价用的是"扫描时刻最新价"，**隐含"触发即成交"假设** ⇒ 日内收益被系统性高估
  （跳空/跌停场景尤其）。用户已认可该口径，但**必须带 caveat 展示**。
· 无分钟线数据源 ⇒ 无法回放"触发后下一根分钟线"的真实成交价。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from statistics import mean

from core import ai_personas
from core.paper_account import PaperAccount
from core.risk import RiskEngine
from db import store

_logger = logging.getLogger(__name__)

# 样本门槛：低于这两个数就**不具备统计意义**（页面必须明说，不能静默排行）
MIN_DAYS = 20
MIN_TRADES = 30

# 单只行情取数超时（秒）。⚠️ 底层取数源没有超时 ⇒ 批量取数会无限挂起
#    （项目实测跑过 9 小时）。线程超时不能真正中断底层请求，但能让**主流程**不挂。
_BAR_TIMEOUT = 20


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ---------------- 行情（bars） ----------------

def daily_bars(codes: list, date: str, *, timeout: int = _BAR_TIMEOUT) -> dict:
    """取 `{code: {"open","close","prev_close"}}` —— `settle_open` 需要的最小形状。

    ⚠️ 逐只取日线（`market_snapshot` 只有 price，没有 open/prev_close，不能当 bars 源）；
    单只超时/失败一律**视为停牌**（不成交），不用收盘价伪造开盘价。
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout

    fetcher = None
    out: dict = {}
    for code in list(codes or []):
        try:
            if fetcher is None:
                from data.stock_data import StockDataFetcher
                fetcher = StockDataFetcher()
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fetcher.get_stock_data, code, "1mo")
                df = fut.result(timeout=timeout)
        except FTimeout:
            _logger.warning("[联赛结算] %s 取行情超时（%ds）⇒ 视为停牌", code, timeout)
            continue
        except Exception as e:                               # noqa: BLE001
            _logger.warning("[联赛结算] %s 取行情失败: %s", code, e)
            continue
        try:
            idx = [str(i)[:10] for i in df.index]
            if date not in idx:
                continue                                     # 该日无 K 线（停牌/未开盘）
            pos = idx.index(date)
            row = df.iloc[pos]
            prev_close = float(df.iloc[pos - 1]["Close"]) if pos > 0 else None
            out[code] = {"open": float(row["Open"]), "close": float(row["Close"]),
                         "prev_close": prev_close}
        except Exception as e:                               # noqa: BLE001
            _logger.warning("[联赛结算] %s 解析行情失败: %s", code, e)
    return out


def open_bars(codes: list, date: str) -> dict:
    """取 `date` 的**开盘撮合 bar**：`{code: {"open","close","prev_close"}}`。

    唯一实现（`settle_daily` 的 ① 提交价 与 ② 撮合 bar 共用，口径不会两处不一致）。

    优先级（2026-10-16 修订）：
      ① **实时行情**（`data.realtime`）—— 09:32 时**当日日线尚未生成**，这是唯一可用来源。
         ⚠️ 三条校验缺一不可（宁可"不成交"也不能用错价成交，且必须是**显式**失败）：
           · `stale` / `data_lagging` 为真 ⇒ 拒绝（过期缓存不是开盘价）；
           · `quote_date` 缺失或 ≠ `date` ⇒ 拒绝（前一日快照当今日开盘价是**静默错账**）；
           · `open` 必须为正数。
      ② **日线兜底**（`daily_bars`）—— 当日日线收盘后才生成，故只在盘后补跑 / 手动重算时生效。

    Returns: 只含**真的取到的**标的；取不到由调用方显式记录原因（不伪造、不补 0）。
    """
    wanted = [str(c).strip() for c in (codes or []) if str(c).strip()]
    if not wanted:
        return {}
    out: dict = {}

    # ① 实时行情
    try:
        from data import realtime
        quotes = realtime.get_realtime_quotes(list(wanted)) or {}
        for code in wanted:
            row = quotes.get(code) or {}
            px = row.get("open")
            if not isinstance(px, (int, float)) or isinstance(px, bool) or px <= 0:
                continue
            if row.get("stale") or row.get("data_lagging"):
                _logger.warning("[联赛结算] %s 实时行情已过期（%s）⇒ 不用于开盘价",
                                code, row.get("staleness_reason") or "stale")
                continue
            qd = str(row.get("quote_date") or "")
            if qd != date:
                _logger.warning("[联赛结算] %s 实时行情日期 %r ≠ %s ⇒ 不用于开盘价",
                                code, qd or "(缺日期)", date)
                continue
            live = row.get("price")
            out[code] = {
                "open": float(px),
                # close 只作 `exec_price` 的兜底（`open` 存在时不会用到），用当前价近似；
                # prev_close 参与涨跌停判定，必须如实传递。
                "close": float(live) if isinstance(live, (int, float)) and live > 0 else float(px),
                "prev_close": (float(row["prev_close"])
                               if isinstance(row.get("prev_close"), (int, float)) else None),
            }
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[联赛结算] 实时行情取开盘价失败（回退日线）: %s", e)

    # ② 日线兜底（只补实时缺的）
    missing = [c for c in wanted if c not in out]
    if missing:
        for code, bar in (daily_bars(missing, date) or {}).items():
            px = (bar or {}).get("open")
            if isinstance(px, (int, float)) and not isinstance(px, bool) and px > 0:
                out[code] = {"open": float(px),
                             "close": float((bar or {}).get("close") or px),
                             "prev_close": (bar or {}).get("prev_close")}
    return out


def close_prices(codes: list, trade_date: str) -> dict:
    """取某日**收盘价**（唯一实现：结算 / 记净值 / 给 AI 看的净值**共用**）。

    取价优先级（2026-10-15 修订）：
      ① `market_snapshot` —— 15:40 落库的估值截面，含当日收盘价；**零额外取数、零联网**，
         且**晚于** 15:05 的记净值时刻可用的唯一来源；
      ② `daily_bars` —— 逐只带超时的日线兜底（⚠️ 当日日线在 15:05 通常**尚未生成**，
         这正是"每日净值全按成本价兜底"事故的根因）。

    Returns: `{code: price}` —— **只含真的取到价的标的**；取不到就不出现在结果里
             （缺就是缺，由调用方决定"按成本估值并标记"还是"不记这条"）。
    """
    codes = [str(c).strip() for c in (codes or []) if str(c).strip()]
    if not codes:
        return {}
    prices: dict = {}
    # ① 估值截面（权威、离线）：日期必须严格等于 trade_date，否则宁可不取
    try:
        got_date, rows = store.load_market_snapshot(trade_date, list(codes))
        if got_date == trade_date:
            for r in rows or []:
                code = str(r.get("stock_code") or "").strip()
                px = r.get("price")
                if code and isinstance(px, (int, float)) and not isinstance(px, bool) and px > 0:
                    prices[code] = float(px)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[联赛取价] 快照读取失败（回退日线）: %s", e)
    # ② 日线兜底：只补快照缺的
    missing = [c for c in codes if c not in prices]
    if missing:
        try:
            for c, bar in (daily_bars(missing, trade_date) or {}).items():
                px = (bar or {}).get("close")
                if isinstance(px, (int, float)) and not isinstance(px, bool) and px > 0:
                    prices[c] = float(px)
        except Exception as e:                               # noqa: BLE001
            _logger.warning("[联赛取价] 日线兜底失败: %s", e)
    return prices


# ---------------- 账户读写 ----------------

def _load(name: str, initial_cash: float = None) -> PaperAccount:
    st = store.load_paper_state(name)
    if st:
        return PaperAccount.from_state(st)
    return PaperAccount(float(initial_cash or 1_000_000), name=name)


def _save(acct: PaperAccount) -> None:
    store.save_paper_state(acct.name, acct.to_state())


def _prev_trading_day(day: str) -> str | None:
    """`day` 的**前一交易日**；取不到（日历不可用）返回 None —— **不硬编日期**。"""
    try:
        from data.trading_calendar import last_trading_day
        return last_trading_day(ref=day, include_today=False)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[联赛结算] 取前一交易日失败: %s", e)
        return None


def _pending_buys(trade_date: str, persona: str) -> list:
    """该 persona 在**前一交易日**出、且尚未成交的买入决策（T+1 待执行）。

    ⚠️ 严格限定"前一交易日"（2026-10-16 修订）：原条件是 `signal_date < trade_date`，
    会把**积压多日**的决策一并捞出来，而结算用**当日**开盘价撮合
    ⇒ 本该 T+1 成交的单被按 T+N 的价成交，**口径错位且无痕**。
    错过窗口的决策改由 `_stale_buys` **显式作废**（不静默重试、也不用错价成交）。
    ⚠️ 日历不可用 ⇒ 返回空（宁可不结算，也不拿错日期的价成交）。
    """
    prev = _prev_trading_day(trade_date)
    if not prev:
        return []
    return _pending_buys_at(prev, persona)


def _pending_buys_at(signal_date: str, persona: str) -> list:
    """`signal_date` 当日出、尚未成交的买入决策。"""
    rows = [r for r in store.list_ai_decisions(persona=persona)
            if r["signal_date"] == signal_date and r["advice_status"] == "可执行"
            and float(r.get("position_size") or 0) > 0 and not r.get("order_id")]
    rows.sort(key=lambda r: r["code"])
    return rows


def _stale_buys(trade_date: str, persona: str, prev: str | None) -> list:
    """**早于前一交易日**仍未成交的买入决策 —— 已错过 T+1 窗口，须显式作废。

    ⚠️ 为什么必须显式标记而不是留着：结算按**当日开盘价**撮合，
    积压单会被按今天的价成交（错价）；而"错过了"这件事必须让用户看得见。
    """
    if not prev:
        return []
    rows = [r for r in store.list_ai_decisions(persona=persona)
            if r["signal_date"] < prev and r["advice_status"] == "可执行"
            and float(r.get("position_size") or 0) > 0 and not r.get("order_id")]
    rows.sort(key=lambda r: (r["signal_date"], r["code"]))
    return rows


def _stale_sells(trade_date: str, persona: str, prev: str | None) -> list:
    """**早于前一交易日**仍未成交的**卖出**决策 —— 同 `_stale_buys`，已错过窗口须作废。"""
    if not prev:
        return []
    rows = [r for r in store.list_ai_decisions(persona=persona)
            if r["signal_date"] < prev and r["advice_status"] == "卖出"
            and not r.get("order_id")]
    rows.sort(key=lambda r: (r["signal_date"], r["code"]))
    return rows


def _pending_sells_at(signal_date: str, persona: str) -> list:
    """`signal_date` 当日出、尚未成交的**卖出**决策（T+1 待执行）。

    ⚠️ 补齐"买卖不对称"（2026-10-16 修）：原先**全部**结算筛选都是
    `advice_status == "可执行"`（= 买入），而 `persist_round` 会把 `action="sell"`
    写成 `advice_status="卖出"` ⇒ **盘后决策里的卖出永远不会成交**
    （`settle_intraday_wakes` 处理的是**唤醒轮**的卖出，与决策表无关）。

    卖出语义 = **清仓**（`normalize_actions` 对 sell 不给数量、`position_size` 记为 0），
    与唤醒轮的卖出口径一致。
    """
    rows = [r for r in store.list_ai_decisions(persona=persona)
            if r["signal_date"] == signal_date and r["advice_status"] == "卖出"
            and not r.get("order_id")]
    rows.sort(key=lambda r: r["code"])
    return rows


def _pending_sells_on_or_before(trade_date: str, persona: str) -> list:
    """`trade_date` **当日或之前**尚未成交的**卖出**决策。

    与 `_pending_sells_at` 的唯一差别是允许 `signal_date == trade_date`
    —— 仅供 `settle_at_close`（当日收盘价"加速确认"口径）使用。
    """
    rows = [r for r in store.list_ai_decisions(persona=persona)
            if r["signal_date"] <= trade_date and r["advice_status"] == "卖出"
            and not r.get("order_id")]
    rows.sort(key=lambda r: (r["signal_date"], r["code"]))
    return rows


def _pending_buys_on_or_before(trade_date: str, persona: str) -> list:
    """该 persona 在 trade_date **当日或之前**出、且尚未确认成交的买入决策。

    与 `_pending_buys` 的唯一差别：允许 `signal_date == trade_date`
    —— 用于"盘后按**当日收盘价**确认持仓"（加速确认口径，见 `settle_at_close`）。
    正式 T+1 结算仍走 `_pending_buys`（严格 `<`），两处口径互不污染。
    """
    rows = [r for r in store.list_ai_decisions(persona=persona)
            if r["signal_date"] <= trade_date and r["advice_status"] == "可执行"
            and float(r.get("position_size") or 0) > 0 and not r.get("order_id")]
    rows.sort(key=lambda r: (r["signal_date"], r["code"]))
    return rows


def settle_daily(trade_date: str = None, *, personas: list = None,
                 dry_run: bool = False) -> dict:
    """日频结算：把 T-1 及更早的买入决策转成委托 → **用当日开盘价撮合** → 回填订单号。

    Returns: {"trade_date", "results": [{persona, submitted, settled, rejected, ...}]}
    """
    day = trade_date or _today()
    plist = personas if personas is not None else ai_personas.load_personas()
    results = []
    for p in plist:
        if not p.get("enabled"):
            continue
        name = p["name"]
        acct = _load(name, p.get("initial_cash"))
        prev = _prev_trading_day(day)
        pend = _pending_buys(day, name)
        sells = _pending_sells_at(prev, name) if prev else []
        stale = _stale_buys(day, name, prev) + _stale_sells(day, name, prev)
        submitted, items = [], []
        # ⓪ 错过 T+1 窗口的积压决策（买 + 卖）：**显式作废**
        #    —— 既不拿"今天的开盘价"替它成交（口径错位），也不静默留着让它一直挂。
        for d in stale:
            items.append({"code": d["code"], "signal_date": d["signal_date"],
                          "direction": "卖出" if d["advice_status"] == "卖出" else "买入",
                          "skipped": f"错过 T+1 窗口（应于 {prev} 成交）⇒ 作废，不再结算"})
            if not dry_run:
                store.save_ai_decision({**d, "order_id": None,
                                        "order_status": f"未结算作废（错过 T+1 窗口 {prev}）"})
        # ① 提交委托。**取价只取一次**（买入 + 卖出同一份 `open_bars`）——
        #    ⚠️ 走 `open_bars`（**实时行情优先**）：09:32 时当日日线尚未生成，
        #    原实现用 `daily_bars` 取开盘价 ⇒ 每笔都因"无当日行情"被静默跳过，
        #    日志只报"提交 0 笔"看不出原因 ⇒ 联赛自 10-15 起决策从未成交（2026-10-16 修）。
        prices = {c: pos.avg_cost for c, pos in acct.positions.items()}
        need_codes = sorted({d["code"] for d in pend} | {d["code"] for d in sells})
        bars_now = open_bars(need_codes, day) if need_codes else {}

        # ①-a 卖出决策：**先卖**（回笼现金；且与 `settle_open` 的"先卖后买"顺序一致）
        #      ⚠️ 必须先 `unlock_t1`，否则 `available` 恒为 0 ⇒ 卖单必被拒
        #         （解冻原先只发生在"有成交调用"时，读取方不会自动解冻）。
        if sells:
            acct.unlock_t1(day)
        for d in sells:
            code = d["code"]
            pos = acct.positions.get(code)
            avail = pos.available if pos else 0
            if avail <= 0:
                items.append({"code": code, "signal_date": d.get("signal_date"),
                              "direction": "卖出",
                              "skipped": "无可卖持仓（T+1 未解锁或已清仓）"})
                continue
            # ⚠️ 与买入**对称**的取价检查（2026-10-16 真机演练发现）：
            #    原先卖出不问价格就提交，缺价时会生成一张 `@0.0 pending` 的卖单；
            #    而 `order_id` 已被回填（幂等锚点用掉）⇒ 不会再重试，
            #    那张挂单只能等**下一个**交易日被 `settle_open` 撮合
            #    ⇒ 「10-19 该成交」变成「10-20 开盘成交」，**口径错位且无痕**。
            #    ⇒ 取不到价就**不提交**（宁可当日不成交，也不用错日期的价成交）。
            bar = bars_now.get(code) or {}
            if not (bar.get("open") or bar.get("close")):
                items.append({"code": code, "signal_date": d.get("signal_date"),
                              "direction": "卖出",
                              "skipped": f"取不到 {day} 开盘价（实时行情不可用且当日日线未生成）"})
                continue
            order = acct.submit_order(code, "卖出", avail, date=d["signal_date"],
                                      reason=(d.get("order_reason") or "")[:80])
            submitted.append((d, order))
            items.append({"code": code, "direction": "卖出", "qty": avail,
                          "order_id": order.order_id, "status": order.status})

        # ①-b 买入决策（按仓位比例 × 当前总资产估算股数；整手；现金不足由账户层拒绝）
        for d in pend:
            code = d["code"]
            bar = bars_now.get(code) or {}
            px = bar.get("open") or bar.get("close")
            if not px:
                items.append({"code": code, "signal_date": d.get("signal_date"),
                              "direction": "买入",
                              "skipped": f"取不到 {day} 开盘价（实时行情不可用且当日日线未生成）"})
                continue
            equity = acct.total_value(prices) or acct.initial_cash
            budget = float(d.get("position_size") or 0) * equity
            from core import matching
            qty = matching.round_lot(int(budget // px))
            if qty <= 0:
                items.append({"code": code, "direction": "买入",
                              "skipped": f"仓位过小（预算 {budget:.0f} / 价 {px:g}）"})
                continue
            order = acct.submit_order(code, "买入", qty, date=d["signal_date"],
                                      reason=(d.get("order_reason") or "")[:80])
            submitted.append((d, order))
            items.append({"code": code, "direction": "买入", "qty": qty,
                          "order_id": order.order_id, "status": order.status})
        # ② 撮合（T+1：submit_date < day 才可能成交）
        #    ⚠️ 撮合 bar 与 ① **同源**（`open_bars`），只对实时没覆盖到的标的才去取日线
        #       —— 既避免 09:32 白等一次注定失败的日线请求，也避免两处价格口径不一致。
        codes = sorted({o.code for _d, o in submitted})
        bars: dict = {}
        need_daily = [c for c in codes if c not in bars_now]
        if need_daily:
            bars = daily_bars(need_daily, day)
        bars.update({c: bars_now[c] for c in codes if c in bars_now})
        settled = acct.settle_open(day, bars) if codes else []
        # ③ 回填订单号与状态到决策
        #    ⚠️ **演练（dry_run）不得回填真实库** —— 原实现漏了这个判断，
        #       `settle_daily(dry_run=True)` 会污染 ai_decisions 的 order_id/order_status。
        #       "零写入"承诺必须逐条核对每个写点，漏一处就毁掉演练的意义。
        if not dry_run:
            for d, order in submitted:
                store.save_ai_decision({**d,
                                        "order_id": order.order_id,
                                        "order_status": f"{order.status}"
                                                        + (f"（{order.reason}）" if order.reason else "")})
            _save(acct)
        # ⚠️ 返回结构固定：汇总字段**永远存在**（本轮踩过：跳过项的明细混进
        #    results 顶层 ⇒ 调用方读 ["submitted"] 直接 KeyError）。明细一律放 items。
        results.append({"persona": name, "submitted": len(submitted),
                        "settled": len(settled), "items": items,
                        "prev_trading_day": prev,
                        "orders": [o.to_dict() for _d, o in submitted]})
    return {"trade_date": day, "results": results}


def settle_at_close(trade_date: str = None, *, personas: list = None,
                    dry_run: bool = False) -> dict:
    """用**当日收盘价**把未确认成交的买入决策撮合成持仓（"加速确认"口径）。

    用途：盘后跑完决策轮后，**立刻**看到每个 AI 的持仓与净值（否则要等次日 09:32 的
    T+1 开盘结算）。用户明确要求该口径，但它**不是**真实可行的成交时点
    （15:58 决策时已收盘），因此：

    ⚠️ 口径必须显式区分（沿用项目"两价并存 / 口径单一来源"的纪律）：
      · `settle_daily`  = T+1 **开盘价**成交 —— 正式口径（调度 09:32，真实可行）；
      · `settle_at_close` = **当日收盘价**成交、且允许当日决策当日确认 ——
        **模拟联赛的加速确认**，页面上必须标注；返回值带 `mode="close_confirm"`。

    ⚠️ 幂等（关键）：成交后回填 `ai_decisions.order_id` + 把订单置 `filled`
      ⇒ 次日 `settle_daily` 的 `_pending_buys`（按 `not order_id`）与 `settle_open`
      （只处理 `pending`）都不会再撮合同一笔，**杜绝重复买入**。

    撮合走 `PaperAccount.apply_trade`（内部 `core.matching`：税费/最低佣金/滑点唯一实现），
    不另造一套价格口径。取不到当日收盘价的标的一律**跳过并如实记录**，不用开盘价或成本价伪造。
    """
    day = trade_date or _today()
    plist = personas if personas is not None else ai_personas.load_personas()
    results = []
    for p in plist:
        if not p.get("enabled"):
            continue
        name = p["name"]
        acct = _load(name, p.get("initial_cash"))
        pend = _pending_buys_on_or_before(day, name)
        sells = _pending_sells_on_or_before(day, name)
        items: list[dict] = []
        filled = 0

        # ① 待确认标的的当日收盘价一次取齐（快照优先 → 日线兜底；取不到=停牌/缺数据）
        #    买入 + 卖出共用同一份价，避免两处口径不一致
        codes = sorted({d["code"] for d in pend} | {d["code"] for d in sells})
        prices = close_prices(codes, day) if codes else {}
        from core import matching

        # ①-a 卖出决策：**先卖**（回笼现金；与 `settle_open` 的先卖后买顺序一致）
        #      ⚠️ 加速口径也必须结算卖出 —— 否则"AI 今天决定卖出"在盘后视图里**看不到效果**
        #         （正式口径 `settle_daily` 已补，两者必须对称，否则页面数字会对不上）。
        if sells:
            acct.unlock_t1(day)
        for d in sells:
            code = d["code"]
            pos = acct.positions.get(code)
            avail = pos.available if pos else 0
            px = prices.get(code)
            if avail <= 0:
                items.append({"code": code, "direction": "卖出", "status": "skipped",
                              "reason": "无可卖持仓（T+1 未解锁或已清仓）"})
                continue
            if not px:
                items.append({"code": code, "direction": "卖出", "status": "skipped",
                              "reason": "无当日收盘价（停牌/数据缺失），不伪造成交"})
                continue
            order = acct.submit_order(code, "卖出", avail, date=d["signal_date"],
                                      reason=(d.get("order_reason") or "")[:80])
            if order.status != "pending":
                items.append({"code": code, "direction": "卖出", "order_id": order.order_id,
                              "status": order.status, "reason": order.reason})
                continue
            rec = acct.apply_trade({"date": day, "code": code, "direction": "卖出",
                                    "price": float(px), "quantity": avail})
            if not rec or rec.get("status") == "rejected":
                order.status = "cancelled"
                order.reason = (rec or {}).get("reason") or "成交失败"
                items.append({"code": code, "direction": "卖出", "order_id": order.order_id,
                              "status": "rejected", "reason": order.reason})
                continue
            order.status = "filled"
            order.filled_quantity = int(rec.get("quantity") or avail)
            order.filled_price = float(rec.get("actual_price") or px)
            order.filled_date = day
            filled += 1
            if not dry_run:
                store.save_ai_decision({**d, "order_id": order.order_id,
                                        "order_status": f"filled@{day}（收盘价确认）"})
            items.append({"code": code, "direction": "卖出", "qty": avail, "close": px,
                          "filled_price": order.filled_price,
                          "order_id": order.order_id, "status": "filled"})

        for d in pend:
            code = d["code"]
            px = prices.get(code)
            if not px:
                items.append({"code": code, "status": "skipped",
                              "reason": "无当日收盘价（停牌/数据缺失），不伪造成交"})
                continue
            valuations = {c: prices.get(c, pos.avg_cost) for c, pos in acct.positions.items()}
            equity = acct.total_value(valuations) or acct.initial_cash
            budget = float(d.get("position_size") or 0) * equity
            qty = matching.round_lot(int(budget // px))
            if qty <= 0:
                items.append({"code": code, "status": "skipped",
                              "reason": f"仓位过小（预算 {budget:.0f} / 价 {px:g}）"})
                continue
            order = acct.submit_order(code, "买入", qty, date=d["signal_date"],
                                      reason=(d.get("order_reason") or "")[:80])
            if order.status != "pending":
                items.append({"code": code, "order_id": order.order_id,
                              "status": order.status, "reason": order.reason})
                continue
            rec = acct.apply_trade({"date": day, "code": code, "direction": "买入",
                                    "price": float(px), "quantity": qty})
            if not rec or rec.get("status") == "rejected":
                order.status = "cancelled"
                order.reason = (rec or {}).get("reason") or "成交失败"
                items.append({"code": code, "order_id": order.order_id,
                              "status": "rejected", "reason": order.reason})
                continue
            # 成交成功：回填订单（**幂等锚点**）
            order.status = "filled"
            order.filled_quantity = int(rec.get("quantity") or qty)
            order.filled_price = float(rec.get("actual_price") or px)
            order.filled_date = day
            filled += 1
            # ⚠️ dry_run 必须**零写入**：回填 order_id 也算写入（首版漏了 guard，
            #    演练把 order_id 写进了库、却因 dry_run 没保存账户 ⇒ 真跑时
            #    "已确认"但无持仓，数据自相矛盾）。演练只演练，不落任何库。
            if not dry_run:
                store.save_ai_decision({**d, "order_id": order.order_id,
                                        "order_status": f"filled@{day}（收盘价确认）"})
            items.append({"code": code, "qty": qty, "close": px,
                          "filled_price": order.filled_price,
                          "order_id": order.order_id, "status": "filled"})

        # ② 净值：与 `mark_nav` 同一口径（缺价按成本估值 + `at_cost` 随数落库；
        #    全部缺价则不记）。这里**不复用 mark_nav** 是为了让"确认成交"与"记净值"
        #    在同一次 load/save 里完成（避免两次读写同一账户造成中间态）。
        acct.unlock_t1(day)
        codes2 = list(acct.positions.keys())
        nav_prices = close_prices([c for c in codes2 if c not in prices], day) if codes2 else {}
        nav_prices = {**nav_prices, **{c: prices[c] for c in codes2 if c in prices}}
        at_cost = len([c for c in codes2 if c not in nav_prices])
        nav_rec = None
        if codes2 and at_cost < len(codes2):
            nav_rec = acct.mark_to_market(
                day, {c: nav_prices.get(c, acct.positions[c].avg_cost) for c in codes2})
            nav_rec["at_cost"] = at_cost
        elif codes2:
            _logger.warning("[联赛确认] %s 当日(%s)持仓收盘价全缺 ⇒ 不记净值", name, day)
        if not dry_run:
            _save(acct)
        results.append({"persona": name, "filled": filled, "items": items,
                        "nav": nav_rec, "at_cost": at_cost,
                        "positions": len(codes2)})
    return {"trade_date": day, "results": results, "mode": "close_confirm"}


def _wake_actions(events: list) -> tuple[list, list]:
    """把唤醒事件的 `action` JSON 摊平成 `(卖出项, 买入项)`，每项 `(event, action)`。

    ⚠️ 一个事件可能带多个动作（同一条件触发后既调条件又计补仓）⇒ 必须**逐动作**处理，
    按 `"sell" in e.action` 之类的**子串匹配**会把 `set_wake`(kind=stop_loss) 也算进来。
    """
    sells, buys = [], []
    for e in events:
        try:
            acts = json.loads(e.get("action") or "[]")
        except (TypeError, ValueError):
            continue
        for a in (acts if isinstance(acts, list) else []):
            if not isinstance(a, dict):
                continue
            if a.get("action") == "sell":
                sells.append((e, a))
            elif a.get("action") == "buy":
                buys.append((e, a))
    return sells, buys


def settle_intraday_wakes(trade_date: str = None, *, personas: list = None,
                          dry_run: bool = False) -> dict:
    """把唤醒轮产出的**卖出与持仓内补仓**按「扫描价 + 滑点」立即成交（口径 2）。

    ⚠️ 三条硬约束：
      ① **同时记录触发价与实际成交价**（`trigger_px` vs fill price）——口径 2 的核心，
         否则无法事后检验"高估了多少"；
      ② 当日买入的持仓 T+1 不可卖 ⇒ `apply_trade` 会拒绝，**如实记录不成交**（不伪造）；
      ③ **补仓成交带 `tag="intraday_buy"`** 且以它作**幂等锚点**：同一标的当日已有
         盘中补仓成交就跳过 ⇒ 重复跑结算不会重复买入（`settle_*` 会被重复触发）。
         ⚠️ 不能靠"重跑护栏"来兜底幂等：护栏数的是**已落库的 buy 事件**，
            结算时再跑一遍会把自己数进去而误判"当日已达上限"。
    """
    day = trade_date or _today()
    plist = personas if personas is not None else ai_personas.load_personas()
    out = []
    for p in plist:
        if not p.get("enabled") or p.get("kind") != "llm":
            continue
        name = p["name"]
        events = store.list_wake_events(persona=name, trade_date=day)
        sells, buys = _wake_actions(events)
        if not sells and not buys:
            continue
        acct = _load(name, p.get("initial_cash"))
        # ⚠️ 必须显式解锁 T+1：`apply_trade` 只看 `available`，而当日买入的持仓
        #    available=0；不解锁会把"昨日买入、今日止损"误判成"无可卖持仓"。
        try:
            acct.unlock_t1(day)
        except Exception as e:                               # noqa: BLE001
            _logger.warning("[联赛结算] %s 解锁 T+1 失败: %s", name, e)
        done, sold, bought = [], [], []
        for e, _a in sells:
            code = e["code"]
            pos = acct.positions.get(code)
            if pos is None or pos.available <= 0:
                item = {"code": code, "kind": "sell", "price": e.get("trigger_px"),
                        "status": "cancelled",
                        "reason": "无可卖持仓（T+1 未解锁或已清仓）"}
                done.append(item); sold.append(item)
                continue
            px = e.get("trigger_px")
            if not px:
                item = {"code": code, "kind": "sell", "status": "cancelled",
                        "reason": "缺触发价，无法成交（不伪造价格）"}
                done.append(item); sold.append(item)
                continue
            r = acct.apply_trade({"date": day, "code": code, "direction": "卖出",
                                  "price": float(px), "quantity": pos.available,
                                  "tag": "intraday_sell"})
            if isinstance(r, dict):
                r = dict(r)
                r["trigger_px"] = e.get("trigger_px")      # 口径 2：两价并存
                r["kind"] = "sell"
            item = r or {"code": code, "kind": "sell", "status": "unknown"}
            done.append(item); sold.append(item)

        for e, a in buys:
            code = e["code"]
            px, qty = e.get("trigger_px"), int(a.get("shares") or 0)
            # 幂等锚点：当日/同标的已有盘中补仓成交 ⇒ 跳过（重复结算不再买）
            if any(f.get("tag") == "intraday_buy" and str(f.get("code")) == code
                   and str(f.get("date")) == day for f in acct.fills):
                item = {"code": code, "kind": "buy", "status": "skipped",
                        "reason": "当日已有盘中补仓成交（幂等跳过）"}
                done.append(item); bought.append(item)
                continue
            if not px or qty <= 0:
                item = {"code": code, "kind": "buy", "status": "cancelled",
                        "reason": "缺触发价或可成交股数（护栏未给出数量）⇒ 不伪造成交"}
                done.append(item); bought.append(item)
                continue
            r = acct.apply_trade({"date": day, "code": code, "direction": "买入",
                                  "price": float(px), "quantity": qty,
                                  "tag": "intraday_buy"})
            if isinstance(r, dict):
                r = dict(r)
                r["trigger_px"] = e.get("trigger_px")
                r["kind"] = "buy"
            item = r or {"code": code, "kind": "buy", "status": "unknown"}
            done.append(item); bought.append(item)

        if not dry_run:
            _save(acct)
        out.append({"persona": name, "sells": sold, "buys": bought, "actions": done})
    return {"trade_date": day, "results": out, "mode": "intraday_scan_price"}


def mark_nav(trade_date: str = None, *, personas: list = None, dry_run: bool = False) -> dict:
    """记录所有 AI 账户的当日净值（**只用真取到的当日收盘价**）。

    ⚠️⚠️ 2026-10-15 重写 —— 原实现有严重口径缺陷，实测已造成 AI 幻觉级错误结论：
      原版在取不到当日收盘价时用 `avg_cost` 兜底，且兜底**不留痕**（不落库、不告警、
      不进 AI 上下文）⇒ 净值被伪装成"成本"，恰好 ≈ 初始资金。
      实测 10-15：三个持仓账户净值**全部精确 = 1,000,000.00**，真实应为
      991,731.39 / 1,002,600.53 / 999,918.24；AI 读到后写出
      "今日净值保持不变、该笔交易暂未带来盈亏"（实际茅台 +2.10%）与
      "中国石化表现稳定"（实际 −4.07%）——**幻觉的源头是数据管道，不是模型**。

    现在的口径（**缺就是缺**）：
      · 价格来源 = `close_prices()`（快照优先 → 日线兜底，唯一实现）；
      · 个别标的取不到价 ⇒ 按成本估值，但 **`at_cost` 计数随 nav 一起落库**
        （口径随数字走：AI 与页面都能看到"这条净值里有 N 只是估的"）；
      · **全部持仓都取不到价 ⇒ 当场不记这条 nav**（整条都是假数没有存在价值），
        返回 `recorded=False` 并告警 —— **绝不把成本价冒充成净值**；
      · 顺带 `unlock_t1`：消除"同日买入的不同账户 T+1 解冻状态不一致"
        （原版解冻寄生在"是否恰好被某条路径 load+save 过"）。
    """
    day = trade_date or _today()
    plist = personas if personas is not None else ai_personas.load_personas()
    out = []
    for p in plist:
        if not p.get("enabled"):
            continue
        name = p["name"]
        acct = _load(name, p.get("initial_cash"))
        codes = list(acct.positions.keys())
        prices: dict = {}
        at_cost = 0
        if codes:
            prices = close_prices(codes, day)
            at_cost = len([c for c in codes if c not in prices])
            # 全部取不到价 ⇒ 不记（整条假数没有存在价值）
            if at_cost == len(codes):
                _logger.warning("[联赛净值] %s 当日(%s)一个收盘价都取不到 ⇒ **不记净值**"
                                "（不用成本价冒充）", name, day)
                out.append({"persona": name, "nav": None, "recorded": False,
                            "priced": 0, "at_cost": at_cost, "positions": len(codes),
                            "reason": "当日收盘价全部取不到，未记净值（缺就是缺）"})
                continue
        # 估值价：取到的用市价，缺的退回收盘成本（缺的只数已计入 at_cost）
        valuations = {c: prices.get(c, acct.positions[c].avg_cost) for c in codes}
        acct.unlock_t1(day)                     # 让 available 与"已过 T+1"一致（幂等）
        rec = acct.mark_to_market(day, valuations)
        # ⚠️ mark_to_market 已把同一 dict 对象 append 进 acct.nav ⇒ 这里补键即可随之落库
        rec["at_cost"] = at_cost
        if not dry_run:
            _save(acct)
        out.append({"persona": name, "nav": rec, "recorded": True,
                    "priced": len(prices), "at_cost": at_cost,
                    "positions": len(codes)})
    return {"trade_date": day, "results": out,
            "recorded": sum(1 for r in out if r.get("recorded")),
            "skipped": sum(1 for r in out if not r.get("recorded"))}


# ---------------- 榜单 ----------------

def _stats(name: str) -> dict | None:
    st = store.load_paper_state(name)
    if not st:
        return None
    nav = st.get("nav") or []
    fills = st.get("fills") or []
    if nav:
        vals = [float(n.get("total_value") or 0) for n in nav]
        init = float(st.get("initial_cash") or 1_000_000)
        ret = (vals[-1] / init - 1) if init else 0.0
        peak, mdd = vals[0] if vals else init, 0.0
        for v in vals:
            peak = max(peak, v)
            if peak:
                mdd = min(mdd, v / peak - 1)
    else:
        ret, mdd = 0.0, 0.0

    # ---- 参与度（2026-10-15 加）----
    # 为什么必须有：只比收益会让"永久空仓"免费拿分（0 收益但 0 回撤），
    # 而这恰恰是"主动性不足"的温床。把**平均仓位**与**空仓天数**摆到榜单上，
    # 让"不作为"在榜面上可见（判据仍不用胜率，见 caveat）。
    exposures: list[float] = []
    for n in nav:
        tv = float(n.get("total_value") or 0)
        if tv > 0:
            exposures.append(max(0.0, float(n.get("position_value") or 0)) / tv)
    active_days = sum(1 for e in exposures if e > 1e-4)
    avg_exposure = (mean(exposures) * 100) if exposures else None

    # ---- 盘中交易占比 / 盘中加仓胜率（2026-10-15 加，配合"放开持仓内补仓"）----
    # 为什么必须单列：放开盘中补仓后，"AI 到底靠盘后判断还是靠盘中响应赚钱"是一个
    # **口径问题**——两类成交的时点、价格假设（盘中＝扫描价+滑点，收益系统性高估）
    # 都不同，混在总收益里无法判断这条路该不该继续放宽。
    # ⚠️ 用成交记录自带的 `tag` 区分（`intraday_buy` / `intraday_sell`），
    #    绝不按"日期"或"是否持仓"猜——那会把盘后 T+1 成交算成盘中。
    intraday = [f for f in fills if str(f.get("tag") or "").startswith("intraday_")]
    intraday_buys = [f for f in fills if f.get("tag") == "intraday_buy"]
    closes = [f for f in fills if f.get("tag") == "intraday_sell"]
    wins, closed = 0, 0
    for f in intraday_buys:
        code, day = str(f.get("code")), str(f.get("date"))
        # 该笔盘中补仓**之后**是否已被平掉——**只有平掉才算得出胜负**
        later = [x for x in fills
                 if str(x.get("code")) == code and str(x.get("direction")) == "卖出"
                 and str(x.get("date")) >= day and x.get("pnl") is not None]
        if not later:
            continue                                     # 未平仓 ⇒ 不进分母（不拍 0）
        closed += 1
        if float(later[-1].get("pnl") or 0) > 0:
            wins += 1
    return {"persona": name, "nav_days": len(nav), "trades": len(fills),
            "return_pct": round(ret * 100, 2), "max_drawdown_pct": round(mdd * 100, 2),
            # 交易日里有持仓的天数 / 占比（空仓天数 = 总天数 − 持仓天数）
            "holding_days": active_days,
            "participation_pct": (round(active_days / len(exposures) * 100, 1)
                                  if exposures else None),
            "avg_exposure_pct": (round(avg_exposure, 2) if avg_exposure is not None else None),
            # 盘中相关（口径 2，收益被系统性高估 —— 页面须带 caveat）
            # ⚠️ `intraday_buy_win_rate` 是用户 2026-10-15 明确要求的**诊断项**，
            #    不是排名判据（项目共识"判据不用胜率"针对的是**排名口径**）——
            #    命名带 `intraday_` 前缀 + 只在"有平仓结果"时才有值，避免被当成成绩单。
            "intraday_trades": len(intraday),
            "intraday_buys": len(intraday_buys),
            "intraday_sells": len(closes),
            "intraday_trade_pct": (round(len(intraday) / len(fills) * 100, 1)
                                   if fills else None),
            "intraday_buy_win_rate": (round(wins / closed * 100, 1) if closed else None),
            "intraday_buy_closed": closed,
            "last_total_value": (nav[-1]["total_value"] if nav else None)}


def random_group_intraday(trade_date: str = None) -> dict:
    """随机对照组的**盘中成交**聚合（当日）—— 供公共简报做信息对称披露。

    ⚠️ 公平性：只暴露**随机组的聚合次数与金额占比**（不含代码、不含方向明细），
    对所有 AI 逐字节相同 —— 让 LLM 组知道自己"盘中参与度"的相对位置，
    但不泄露任何可用于抄作业的信息。
    """
    day = trade_date or _today()
    plist = [p for p in ai_personas.load_personas()
             if p.get("kind") == "random" and p.get("enabled")]
    n, amount, exposure = 0, 0.0, []
    for p in plist:
        st = store.load_paper_state(p["name"])
        if not st:
            continue
        fills = [f for f in (st.get("fills") or [])
                 if str(f.get("date")) == day and str(f.get("tag") or "").startswith("intraday_")]
        n += len(fills)
        amount += sum(abs(float(f.get("total") or 0)) for f in fills)
        nav = [x for x in (st.get("nav") or []) if str(x.get("date")) == day]
        if nav:
            tv = float(nav[-1].get("total_value") or 0)
            if tv > 0:
                exposure.append(max(0.0, float(nav[-1].get("position_value") or 0)) / tv)
    return {"date": day, "available": bool(plist), "n_personas": len(plist),
            "intraday_trades": n, "intraday_amount": round(amount, 2),
            "avg_exposure_pct": (round(mean(exposure) * 100, 2) if exposure else None),
            "note": "随机对照组当日盘中成交聚合（**不含标的代码**，仅用于信息对称）"}


# ---------------- 盘中补仓的自动回退（2026-10-15） ----------------
# 用户要求："若连续 5 个交易日盘中加仓收益为负，自动切回「wake 仅卖出」模式"。
#
# ⚠️ 判据必须**可实现且诚实**，所以钉死三条口径：
#   ① 只统计**有平仓结果**的补仓：一笔 `intraday_buy` 的盈亏取"同标的、买入日之后
#      第一笔带 `pnl` 的卖出"——**没平掉的补仓不计入判据**（算不出来，不许拍 0 当"没亏"）；
#   ② 分组单位是**补仓日**（不是自然日）：当天那批补仓分摊到的已实现盈亏之和；
#   ③ "连续 N 天为负"指的是**最近 N 个有平仓结果的补仓日全部为负**——
#      而不是日历上最近 N 天（否则中间夹着没成交的日子会永远凑不满）。
#
# ⚠️ 回退是**自动的、但可人工复位**：一旦禁用，就没有新的盘中成交 ⇒ 判据窗口
#    永远不动 ⇒ **死锁**。所以留 `manual_enable` 覆盖（设置页可点），
#    并把"为什么被禁用"和"依据的数字"一并写进状态文件，便于复盘而不是拍脑袋放开。
WAKE_BUY_ROLLBACK_DAYS = 5
_WAKE_BUY_STATE = "ai_wake_buy_state.json"


def _wake_buy_state_path():
    # ⚠️ 状态文件可写 ⇒ 跟数据根走。用 `__file__` 推导的话打包后是只读包内的
    #    `_internal/temp/`，写入失败或下次启动失踪（表现：唤醒买入去重失效）。
    import os

    from config.paths import data_root
    return os.path.join(data_root(), "temp", _WAKE_BUY_STATE)


def read_wake_buy_state() -> dict:
    """读盘中补仓的**人工覆盖**状态（文件缺失/损坏都按"无人干预"处理）。"""
    import json as _json
    import os
    try:
        with open(_wake_buy_state_path(), encoding="utf-8") as f:
            d = _json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:                                        # noqa: BLE001
        return {}


def write_wake_buy_state(**kw) -> bool:
    """原子写人工覆盖状态（`manual_enable=True/False`，`reason`）。失败不静默。"""
    import json as _json
    import os
    # ⚠️ 统一转 str：`_wake_buy_state_path` 在测试里被换成 `pathlib.Path`，
    #    直接 `p + ".tmp"` 会 TypeError（Path + str）⇒ 开关写不进去、静默按"开放"处理。
    p = str(_wake_buy_state_path())
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        state = read_wake_buy_state()
        state.update(kw)
        state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(state, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)
        return True
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[联赛] 写盘中补仓状态失败: %s", e)
        return False


def intraday_buy_daily_pnl() -> dict:
    """按"补仓日"汇总**已实现**盈亏：`{date: {"pnl": float, "n": int}}`。

    口径见上方注释① ②：一笔补仓的盈亏 = 同标的、补仓日之后第一笔带 `pnl` 的卖出。
    """
    per_day: dict = {}
    for p in ai_personas.load_personas():
        if not p.get("enabled"):
            continue
        st = store.load_paper_state(p["name"])
        if not st:
            continue
        fills = st.get("fills") or []
        for i, f in enumerate(fills):
            if f.get("tag") != "intraday_buy":
                continue
            code, day = str(f.get("code")), str(f.get("date"))
            match = next((x for x in fills[i + 1:]
                          if str(x.get("code")) == code
                          and str(x.get("direction")) == "卖出"
                          and str(x.get("date")) >= day
                          and x.get("pnl") is not None), None)
            if match is None:
                continue                                     # 未平仓 ⇒ 不计入（算不出）
            rec = per_day.setdefault(day, {"pnl": 0.0, "n": 0})
            rec["pnl"] += float(match.get("pnl") or 0.0)
            rec["n"] += 1
    return per_day


def wake_buy_status(trade_date: str = None) -> dict:
    """盘中补仓当前是否开放（含自动回退与人工复位）。

    Returns: `{enabled, reason, auto_disabled, manual, evidence}`
    —— `evidence` 必须原样展示给用户（"为什么被关掉"不能只给结论）。
    """
    manual = read_wake_buy_state()
    per_day = intraday_buy_daily_pnl()
    days = sorted(per_day)
    tail = days[-WAKE_BUY_ROLLBACK_DAYS:]
    streak_negative = bool(tail) and len(tail) >= WAKE_BUY_ROLLBACK_DAYS and \
        all(per_day[d]["pnl"] < 0 for d in tail)
    evidence = {"days": tail, "pnl": {d: round(per_day[d]["pnl"], 2) for d in tail},
                "threshold_days": WAKE_BUY_ROLLBACK_DAYS}

    if streak_negative and not manual.get("manual_enable"):
        return {"enabled": False, "auto_disabled": True, "manual": manual,
                "reason": (f"自动回退：最近 {WAKE_BUY_ROLLBACK_DAYS} 个有平仓结果的补仓日"
                           f"盘中加仓收益**全部为负** ⇒ 暂回到「盘中仅卖出」，"
                           f"待策略优化后在设置页人工复位"),
                "evidence": evidence}
    if manual.get("manual_enable"):
        return {"enabled": True, "auto_disabled": streak_negative, "manual": manual,
                "reason": f"人工复位开放（{manual.get('updated_at') or '—'}）："
                          f"{manual.get('reason') or '用户手动开启'}",
                "evidence": evidence}
    return {"enabled": True, "auto_disabled": False, "manual": manual,
            "reason": ("开放（未有连续 "
                       f"{WAKE_BUY_ROLLBACK_DAYS} 个负收益补仓日）"
                       if days else "开放（暂无已平仓的盘中补仓样本）"),
            "evidence": evidence}


def random_group_exposure(nav_date: str = None) -> dict:
    """随机对照组的**平均仓位**（当日）—— 放进公共简报做"社会比较"，提高主动性。

    为什么要它（2026-10-15）：AI 空仓时只会自我说服"环境不明朗"；给出一个
    **同口径的对照数字**（"随机组今天平均仓位 12%"）能让"不作为"变成需要解释的选择。
    ⚠️ 只暴露**随机组的聚合仓位**（不含任何代码、不含 LLM 组的任何信息）⇒
    对所有 AI 逐字节相同，不破坏公平性。
    """
    plist = [p for p in ai_personas.load_personas()
             if p.get("kind") == "random" and p.get("enabled")]
    vals: list[float] = []
    for p in plist:
        st = store.load_paper_state(p["name"])
        if not st:
            continue
        nav = st.get("nav") or []
        if not nav:
            continue
        rec = nav[-1] if nav_date is None else next(
            (n for n in nav if str(n.get("date")) == str(nav_date)), None)
        if not rec:
            continue
        tv = float(rec.get("total_value") or 0)
        if tv > 0:
            vals.append(max(0.0, float(rec.get("position_value") or 0)) / tv)
    if not vals:
        return {"available": False, "avg_exposure_pct": None, "n": 0,
                "note": "随机对照组暂无净值记录，无法给出对照仓位"}
    return {"available": True, "avg_exposure_pct": round(mean(vals) * 100, 2),
            "n": len(vals),
            "note": "随机对照组当日平均仓位（同口径对照；不含任何持仓代码）"}


def leaderboard(*, frequency: str = None) -> dict:
    """按决策频率分组的榜单（日频/日内**不可混榜**，口径 3）。

    判据设计（项目共识，**不用胜率**）：
      · `excess_vs_random_mean`：相对随机对照组均值的超额收益
      · `random_percentile`：收益落在随机组分布中的分位（越高说明越不像随机）
      · `conclusive`：样本是否达到 MIN_DAYS/MIN_TRADES（不足时**不得当作结论**）

    Returns: {"groups": {frequency: {...}}, "caveat": [...]}
    """
    personas = {p["name"]: p for p in ai_personas.load_personas()}
    rows = []
    for name in sorted(personas):
        p = personas[name]
        # 不参加联赛的人格（阿深：模拟自营，研究方法不同）不进任何榜。
        # ⚠️ 判据是 `league_eligible` 字段，**不是** `frequency`——按频率猜
        # "intraday 就是阿深"会在将来加第二个人格时静默错位。
        if not p.get("league_eligible", True):
            continue
        s = _stats(name)
        if s is None:
            continue
        s.update({"label": p.get("label"), "kind": p.get("kind"),
                  "model": p.get("model"), "frequency": p.get("frequency") or "daily"})
        rows.append(s)

    caveats = [
        "⚠️ 日内成交按「扫描时刻最新价 + 滑点」计价（无分钟线数据源）⇒ "
        "隐含『触发即成交』假设，日内收益被**系统性高估**，跳空/跌停场景尤甚。",
        f"⚠️ 样本门槛：{MIN_DAYS} 个交易日 / {MIN_TRADES} 笔成交；不足时**不具备统计意义**。",
        "⚠️ 判据用「相对随机组的超额 + 分位」，**不用胜率**（项目共识）。",
        # 2026-10-15 加：只比收益会让"永久空仓"免费拿分 ⇒ 单列参与度，别只看收益排名
        "⚠️ **参与度单列**（持仓天数占比 / 平均仓位）：收益 0 且从不持仓 = 没参与比赛，"
        "不能与「积极参与但小幅亏损」等价看待。",
    ]
    groups: dict = {}
    for freq in ("daily", "intraday"):
        if frequency and freq != frequency:
            continue
        grp = [r for r in rows if r["frequency"] == freq]
        if not grp:
            continue
        rnd = [r["return_pct"] for r in grp if r["kind"] == "random"]
        rnd_mean = mean(rnd) if rnd else None
        for r in grp:
            if rnd_mean is not None and r["kind"] == "llm":
                r["excess_vs_random_mean"] = round(r["return_pct"] - rnd_mean, 2)
                rank = sum(1 for x in rnd if x < r["return_pct"])
                r["random_percentile"] = round(rank / len(rnd) * 100, 1) if rnd else None
            else:
                r["excess_vs_random_mean"] = None
                r["random_percentile"] = None
            r["conclusive"] = (r["nav_days"] >= MIN_DAYS and r["trades"] >= MIN_TRADES)
        grp.sort(key=lambda x: x["return_pct"], reverse=True)
        groups[freq] = {
            "random_mean_return_pct": (round(rnd_mean, 2) if rnd_mean is not None else None),
            "random_count": len(rnd), "rows": grp,
            "conclusive": all(r["conclusive"] for r in grp) if grp else False,
        }
    return {"as_of": _today(), "groups": groups, "caveat": caveats}
