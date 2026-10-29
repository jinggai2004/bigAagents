# -*- coding: utf-8 -*-
"""交易级盈亏分解（回测验证设计 3.1，2026-10-29）。

回答"每一笔钱：择时贡献多少、持有贡献多少、执行摩擦吃掉多少"。

口径（多头；卖出对称取反进合计）
──────────────────────────────
  entry_gap  = fp_buy / dp_buy − 1        # 决策日收盘 → 买入成交价（买贵=摩擦）
  exit_gap   = 1 − fp_sell / dp_sell      # 卖出成交价 → 决策日收盘（卖低=摩擦）
  holding    = fp_sell / fp_buy − 1       # 真正的"判断对不对"
  counterfactual = dp_sell / dp_buy − 1   # 若两端都按决策价成交
  fees       = 佣金 + 印花税（来自 fills.payload，按配对数量分摊）
  恒等式（对数口径）：ln(1+cf) = ln(1+holding) + ln(fp_b/dp_b) + ln(dp_s/fp_s)

【F6 数据事实与自愈】`paper_orders.decision_price` 存量 35/35 全 NULL、
`slippage` 35/35 是 0.0（"没算"被写成"零滑点"）。本模块做两件事：
  1. `rebuild_decision_prices`：决策价用**面板收盘**（本地零网络）重建，
     滑点顺手按 `filled_price − decision_price` 重算，经 `save_paper_state`
     回写——数据库里的 NULL/假零自愈（写路径 `_pos_or_none` 语义本就正确）；
  2. 分解时只信重建后的值：`decision_price<=0` 的笔进 `degraded`，**绝不
     按 0 参与合计**（"缺就是缺"）。

⚠️ 不改 `PaperAccount.from_state` 的 NULL→0.0 强转：那条对 limit_price 是
   load-bearing 的（pending 订单重启后丢失限价意图会变成"开盘即成交"，
   见 store.py 内注释）；分析器绕开它直接读 state dict 即可。
"""

from __future__ import annotations

import logging
from datetime import datetime

from db import store

_logger = logging.getLogger(__name__)

# 面板收盘查询的结果缓存（进程级；键 (code, date_str)）
_CLOSE_CACHE: dict[tuple, float] = {}


def _panel_close(code: str, date_str: str) -> float:
    """≤date_str 的最近面板收盘价（本地 parquet，零网络）；取不到返回 0.0。

    ⚠️ 测试通过 monkeypatch 本函数注入确定值。
    """
    key = (code, str(date_str)[:10])
    if key in _CLOSE_CACHE:
        return _CLOSE_CACHE[key]
    px = 0.0
    try:
        from data.bars import get_bars
        df = get_bars(code, end=str(date_str)[:10])
        if df is not None and len(df) > 0 and "close" in df.columns:
            dcol = "date" if "date" in df.columns else df.columns[0]
            sub = df[df[dcol].astype(str).str[:10] <= str(date_str)[:10]]
            if len(sub) > 0:
                px = float(sub["close"].iloc[-1])
    except Exception as e:                              # noqa: BLE001
        _logger.info("[盈亏分解] %s 面板收盘取不到（%s），按缺失处理", code, e)
    _CLOSE_CACHE[key] = px
    return px


def _accounts(name: str | None) -> list[str]:
    return [name] if name else list(store.list_paper_accounts())


def _rebuild_in_state(state: dict) -> dict:
    """在**这一份** state dict 上原地重建决策价/滑点（供持久化与分解共用）。

    ⚠️ 必须共用同一个实现：rebuild_decision_prices（要回写）与
    decompose_round_trips（只在内存里用）各自写一遍必然漂移——
    分解用的是重建后的值，回写的也是重建后的值，两个入口一个真相。
    """
    out = {"checked": 0, "rebuilt_price": 0, "rebuilt_slippage": 0, "skipped_no_price": 0}
    for o in state.get("orders") or []:
        out["checked"] += 1
        dp = o.get("decision_price")
        dp_ok = isinstance(dp, (int, float)) and dp > 0
        if not dp_ok and o.get("submit_date"):
            px = _panel_close(o.get("code") or "", o["submit_date"])
            if px > 0:
                o["decision_price"] = px
                out["rebuilt_price"] += 1
                dp = px
            else:
                out["skipped_no_price"] += 1
        # 滑点假零自愈：已成交、有决策价、但 slippage 缺失/为 0
        if isinstance(dp, (int, float)) and dp > 0 \
                and o.get("status") == "filled" \
                and float(o.get("filled_price") or 0) > 0 \
                and not (isinstance(o.get("slippage"), (int, float)) and o["slippage"]):
            o["slippage"] = round(float(o["filled_price"]) - float(dp), 4)
            out["rebuilt_slippage"] += 1
    return out


def rebuild_decision_prices(name: str | None = None,
                            persist: bool = True) -> dict:
    """重建 paper_orders 的 decision_price / slippage（F6 自愈）。

    - 决策价 = submit_date 当日（或≤其最近一日）的面板收盘；
    - 已成交且滑点为 0/缺 的行，按 `filled_price − decision_price` 重算；
    - `persist=True` 时经 `save_paper_state` 回写（写路径语义本就正确，
      `_pos_or_none` 会把"仍是 0"的写回 NULL）。

    Returns: {accounts, checked, rebuilt_price, rebuilt_slippage, skipped_no_price, persisted}
    """
    out = {"accounts": [], "checked": 0, "rebuilt_price": 0,
           "rebuilt_slippage": 0, "skipped_no_price": 0, "persisted": False}
    for acc in _accounts(name):
        state = store.load_paper_state(acc)
        if not state:
            continue
        out["accounts"].append(acc)
        counts = _rebuild_in_state(state)
        out.update({k: out[k] + counts[k] for k in counts})
        if persist and (counts["rebuilt_price"] or counts["rebuilt_slippage"]):
            state["name"] = acc
            store.save_paper_state(acc, state)
            out["persisted"] = True
    return out


def _fill_fee_index(fills: list[dict]) -> dict[tuple, list[dict]]:
    """(code, direction, date) → [fill]；payload 里的佣金/印花税按此索引。"""
    idx: dict[tuple, list[dict]] = {}
    for f in fills or []:
        key = (str(f.get("code")), str(f.get("direction")), str(f.get("date"))[:10])
        idx.setdefault(key, []).append(f)
    return idx


def _fees_for(idx, code: str, direction: str, date: str, qty: int) -> tuple[float, bool]:
    """该腿的手续费（按配对数量占成交数量的比例分摊）；找不到 fill ⇒ (0.0, degraded)"""
    for f in idx.get((str(code), str(direction), str(date)[:10]), []):
        fq = int(f.get("quantity") or 0)
        if fq <= 0:
            continue
        ratio = min(1.0, qty / fq)
        fee = (float(f.get("commission") or 0) + float(f.get("stamp_tax") or 0)) * ratio
        return fee, False
    return 0.0, True


def decompose_round_trips(name: str | None = None) -> dict:
    """把成交订单配对成回合（FIFO），逐回合分解盈亏来源。

    Returns:
        {accounts, n_trips, trips: [...], summary: {...}, degraded: n}
        trip = {account, code, qty, entry: {submit, decision_price, fill_price,
                fill_date, entry_gap_pct}, exit: {...同构, exit_gap_pct},
                holding_pct, counterfactual_pct, fees, degraded: bool}
        - degraded=True 的回合**不进 summary 合计**（决策价缺失，无法分解）。
    """
    rebuild_decision_prices(name=name, persist=False)   # 顺便把库里存量也刷一遍
    trips: list[dict] = []
    degraded = 0
    accounts = _accounts(name)
    for acc in accounts:
        state = store.load_paper_state(acc)
        if not state:
            continue
        _rebuild_in_state(state)                        # 在**这一份** state 上原地重建
        orders = [o for o in (state.get("orders") or [])
                  if o.get("status") == "filled" and float(o.get("filled_price") or 0) > 0]
        orders.sort(key=lambda o: (str(o.get("submit_date")), int(o.get("order_id") or 0)))
        fee_idx = _fill_fee_index(state.get("fills") or [])
        open_lots: dict[str, list[dict]] = {}
        for o in orders:
            code = o.get("code") or ""
            direction = str(o.get("direction"))
            qty = int(o.get("filled_quantity") or 0)
            fp = float(o.get("filled_price") or 0)
            dp = float(o.get("decision_price") or 0)
            if direction == "买入":
                open_lots.setdefault(code, []).append(o)
                continue
            # 卖出：FIFO 消费同 code 的买入 lots
            remaining = qty
            while remaining > 0 and open_lots.get(code):
                lot = open_lots[code][0]
                lot_qty = int(lot.get("filled_quantity") or 0)
                used = min(remaining, lot_qty)
                dp_b = float(lot.get("decision_price") or 0)
                fp_b = float(lot.get("filled_price") or 0)
                dp_s, fp_s = dp, fp
                trip = {
                    "account": acc, "code": code, "qty": used,
                    "entry": {"submit_date": lot.get("submit_date"),
                              "decision_price": dp_b or None,
                              "fill_price": fp_b, "fill_date": lot.get("filled_date")},
                    "exit": {"submit_date": o.get("submit_date"),
                             "decision_price": dp_s or None,
                             "fill_price": fp_s, "fill_date": o.get("filled_date")},
                    "entry_gap_pct": None, "exit_gap_pct": None,
                    "holding_pct": None, "counterfactual_pct": None,
                    "fees": None, "degraded": False,
                }
                bad = False
                if dp_b > 0 and fp_b > 0:
                    trip["entry_gap_pct"] = round((fp_b / dp_b - 1.0) * 100, 4)
                else:
                    bad = True
                if dp_s > 0 and fp_s > 0:
                    trip["exit_gap_pct"] = round((1.0 - fp_s / dp_s) * 100, 4)
                else:
                    bad = True
                if fp_b > 0 and fp_s > 0:
                    trip["holding_pct"] = round((fp_s / fp_b - 1.0) * 100, 4)
                else:
                    bad = True
                if dp_b > 0 and dp_s > 0:
                    trip["counterfactual_pct"] = round((dp_s / dp_b - 1.0) * 100, 4)
                else:
                    bad = True
                fee_b, miss_b = _fees_for(fee_idx, code, "买入",
                                          str(lot.get("filled_date")), used)
                fee_s, miss_s = _fees_for(fee_idx, code, "卖出",
                                          str(o.get("filled_date")), used)
                trip["fees"] = round(fee_b + fee_s, 2)
                trip["degraded"] = bad or miss_b or miss_s
                trip["degrade_reasons"] = [r for r, hit in (
                    ("决策价缺失", bad and (dp_b <= 0 or dp_s <= 0)),
                    ("无对手续费记录", miss_b or miss_s)) if hit]
                if trip["degraded"]:
                    degraded += 1
                else:
                    trips.append(trip)
                # 消耗 lot
                if used >= lot_qty:
                    open_lots[code].pop(0)
                else:
                    lot["filled_quantity"] = lot_qty - used
                remaining -= used

    # ── 合计（只算非 degraded）──
    n = len(trips)
    def _avg(key):
        vals = [t[key] for t in trips if t.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    summary = {
        "n_trips": n,
        "entry_gap_avg_pct": _avg("entry_gap_pct"),
        "exit_gap_avg_pct": _avg("exit_gap_pct"),
        "holding_avg_pct": _avg("holding_pct"),
        "counterfactual_avg_pct": _avg("counterfactual_pct"),
        "fees_total": round(sum(t.get("fees") or 0 for t in trips), 2),
        "note": "entry_gap>0=买贵（摩擦）；exit_gap>0=卖低（摩擦）；"
                "counterfactual=两端按决策价成交的理想收益",
    }
    _logger.info("[盈亏分解] 账户 %s：配对 %d 回合（degraded %d）", accounts, n, degraded)
    return {"accounts": accounts, "n_trips": n, "trips": trips,
            "summary": summary, "degraded": degraded}
