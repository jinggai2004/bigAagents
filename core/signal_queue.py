# -*- coding: utf-8 -*-
"""待深度分析候选的检测（盘中，**零 LLM / 零额外联网**）

⚠️⚠️ **命名更正（2026-10-23 晚）**：本模块早先叫"买入信号"，那是**误称**。
事件研究（`scripts/eval_signal_criterion.py`，990 交易日 × 5725 只）实测：
    判据（涨幅≥3% 且 相对放量≥1.5×）的 T+5 超额 **-0.94%**、T+20 超额 **-1.89%**，
    且**分年五年一致为负**（T+5：-1.01 / -1.00 / -1.13 / -0.86 / -0.67%）。
    拆解显示伤害主要来自"**当日涨了 3%**"（追高），而不是"放量"
    （仅放量不涨：T+5 超额 -0.14%）；涨得越多越差（3~5% → -0.55%，>10% → -1.87%），
    放得越大越差。
⇒ **它没有买入预测力，不能叫"买入信号"**。它的真实职责是：
    **从"与用户有关"的票里挑出值得花 LLM 额度做深度分析的候选**，
    真正的买入判断由深度分析的「可执行」结论给出（`core.evening_analysis`）。
⇒ 定位因此是"**候选筛选**"，宁可多挑（多花的额度有限），
   **漏掉才是事故**（漏了 ⇒ 永远没人告诉用户这只票值得研究）。

为什么有这个模块（2026-10-23）：
  用户要的是"**达到买入条件/信号后深度分析，可行再发通知**"，
  而此前盘中推送只有"涨了 5%"这类**现象**，没有判据，
  更没有"判据成立 → 深度分析 → 只在结论可行时通知"这条链路。

⚠️ 设计上的三条硬约束：
  1. **盘中只落库，不推送**。判定用盘中数据（残缺），结论必须由收盘后的
     深度分析给出 —— 否则就是在用不完整数据下结论（项目已有同类教训：
     15:05 取不到收盘价却把净值伪装成"≈初始资金"）。
  2. **判据必须确定性、可复现、零 LLM**。数字不经模型改写。
  3. **只对"与用户有关"的票判定**（自选 / 今日候选 / 持仓），不做全市场扫描。
     ⚠️ 线上范围约 21 只（自选 17 + 候选 5 + 持仓）⇒ 日均命中 1~3 只，量级可接受
     （"全市场日均命中 299 只"是**回放口径**，不是线上口径 —— 别拿它当线上问题）。

判据（两条同时成立才算候选）：
  · 涨幅 ≥ `MIN_CHANGE_PCT`（盘中现价 vs 昨收，昨收取自行情源自带值 ✅）
  · 相对放量：当日成交额 ≥ `VOLUME_RATIO` × 20 日均额
    —— **相对量**而非绝对值。绝对值（"成交额 ≥ 10 亿"）对大盘蓝筹恒真、
    对小盘恒假（10-23 实测：154 条告警里 125 条是这类假异动）。

⚠️ 20 日均额来自**本地面板**（`data.bars.get_bars`，单只 ≈8.6ms、**零联网**）。
⚠️⚠️ 面板当前最新只到 2026-10-16（滞后 5 个交易日，10-18 生成的静态快照）
⇒ 均额是"截至面板最新日"的 20 日均值。**量级判断对几天滞后不敏感**
（比值稳定），但必须**在消息里标出均额口径日期** —— 否则用户会以为是当日均额。
⚠️ 已知局限：均额用**均值**，会被单日极端成交额拉高；中位数更稳，
   但会让线上口径与回放口径不一致 ⇒ 暂不改，记录在案。
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

# 涨幅门槛（%）：盘中"值得看一眼"的起点。比 alert_engine 的 5% 更早，
# 因为这里**只落库不推送**，噪音成本为零；真正推给用户的是深度分析结论。
MIN_CHANGE_PCT = 3.0
# 相对放量倍数（当日成交额 / 20 日均额）
VOLUME_RATIO = 1.5
# 每日最多记多少条信号（预算闸门：信号票要占深度分析的额度）
MAX_SIGNALS_PER_DAY = 10
# 参与判定的"今日候选"名次上限
TOP_CANDIDATES = 10


def _avg_amount(code: str) -> dict:
    """本地面板的 20 日均额（元）与口径日期。取不到返回 {}（缺就是缺，不猜）。"""
    try:
        from data.bars import get_bars
        df = get_bars(code, adjust="qfq")
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[信号] %s 面板读取失败: %s", code, e)
        return {}
    if df is None or len(df) < 20:
        return {}
    try:
        amounts = df["amount"].tail(20)
        avg = float(amounts.mean())
    except (KeyError, TypeError, ValueError) as e:
        _logger.warning("[信号] %s 成交额列不可用: %s", code, e)
        return {}
    if not avg or avg <= 0:
        return {}
    return {"avg_amount": avg, "as_of": str(df.attrs.get("last_date") or "")}


def detect_signals(quotes: dict, scope_codes: list, relations: dict = None,
                   day: str = None, min_change_pct: float = MIN_CHANGE_PCT,
                   volume_ratio: float = VOLUME_RATIO,
                   max_signals: int = MAX_SIGNALS_PER_DAY) -> list[dict]:
    """在给定行情里挑出"达到买入信号"的票（**只判定，不落库、不推送**）。

    Args:
        quotes: `{code: 行}`（盘中行情，行内含 price/change_pct/amount_wan）。
        scope_codes: 参与判定的代码范围（自选 ∪ 候选 TOP ∪ 持仓）。
        relations: `{code: "持仓·自选"}` —— 写进信号详情，让通知能说清关系。
        day: 交易日（默认今天）。
        max_signals: 最多返回几条（按涨幅降序）。

    Returns: `[{"code","name","kind","detail","price","change_pct","volume_ratio",
               "amount_ref","relation"}]`
    """
    relations = relations or {}
    out: list = []
    for code in (scope_codes or []):
        c = str(code or "").strip()
        row = (quotes or {}).get(c)
        if not row:
            continue
        pct = row.get("change_pct")
        if not isinstance(pct, (int, float)) or pct < min_change_pct:
            continue
        price = row.get("price")
        amount_wan = row.get("amount_wan")
        amount = float(amount_wan) * 10000.0 if isinstance(amount_wan, (int, float)) else 0.0
        ref = _avg_amount(c)
        ratio = (amount / ref["avg_amount"]) if (ref and amount > 0) else None
        # ⚠️ 均额取不到 ⇒ **不判定放量**（缺就是缺），但涨幅达标仍算信号，
        #    并在详情里写明"相对放量不可判定"。
        if ratio is not None and ratio < volume_ratio:
            continue
        detail = f"涨幅 {pct:+.2f}%"
        if ratio is not None:
            detail += f"、成交额 {ratio:.1f}×20日均额（均额截至 {ref['as_of']}）"
        else:
            detail += "、相对放量不可判定（面板无该票日线）"
        # ⚠️ 措辞必须是"**候选**"而不是"买入信号"：事件研究证明这个判据本身
        #    没有买入预测力（T+5 超额 -0.94%），它只是"值得做深度分析的票"。
        detail += "｜待深度分析确认"
        out.append({
            "code": c,
            "name": row.get("name") or c,
            "kind": "surge_volume",
            "detail": detail,
            "price": price if isinstance(price, (int, float)) else None,
            "change_pct": round(float(pct), 2),
            "volume_ratio": round(ratio, 2) if ratio is not None else None,
            "amount_ref": ref.get("as_of") or "",
            "relation": relations.get(c) or "",
        })
    out.sort(key=lambda s: (-(s.get("change_pct") or 0), s["code"]))
    return out[:max_signals]


def record_signals(signals: list, day: str = None) -> list:
    """把检测到的信号落库（**幂等**：同一天同票同类型只记一次）。

    Returns: 新记录的信号列表（已存在的不重复返回）。
    """
    from datetime import datetime
    from db import store

    d = day or datetime.now().strftime("%Y-%m-%d")
    fresh = []
    for s in signals or []:
        try:
            ok = store.insert_signal(
                d, s.get("code"), s.get("name"), s.get("kind"), s.get("detail"),
                price=s.get("price"), ma20=s.get("ma20"),
                change_pct=s.get("change_pct"), relation=s.get("relation"))
        except Exception as e:                              # noqa: BLE001
            _logger.warning("[信号] 落库失败（忽略）: %s", e)
            continue
        if ok:
            fresh.append(s)
    return fresh


def pending_codes(day: str = None) -> list:
    """当日**尚未被深度分析覆盖**的信号票代码（收盘后分析用）。"""
    from datetime import datetime
    from db import store

    d = day or datetime.now().strftime("%Y-%m-%d")
    try:
        rows = store.list_signals(d, analyzed_only=True)
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[信号] 读取待分析队列失败: %s", e)
        return []
    out, seen = [], set()
    for r in rows:
        code = str(r.get("code") or "")
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def mark_analyzed(day: str, codes: list) -> int:
    """标记信号票已被深度分析覆盖（幂等）。"""
    from datetime import datetime
    from db import store

    d = day or datetime.now().strftime("%Y-%m-%d")
    try:
        return store.mark_signals_analyzed(d, codes)
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[信号] 标记已分析失败: %s", e)
        return 0
