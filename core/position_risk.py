# -*- coding: utf-8 -*-
"""
模块名: core/position_risk.py
用途: **已持有组合**的结构性体检 —— 集中度（单票 / 行业）与流动性（退出天数）

与 `core/risk.py` 的分工（避免"两套说法"）：
  - `RiskEngine.assess()`        面向"一批选股候选"，回答**还没买**时怎么分配；
  - `RiskEngine.suggest_levels()` 面向"单只标的"，回答止损止盈放哪儿；
  - `core/portfolio.level_sanity()` 面向"录入的那一组价位"，回答价位自不自洽；
  - **本模块**面向"现在手里的这个组合"，回答"结构上有没有隐患"。

为什么需要它（2026-10-17）：`RiskEngine` 里 `MAX_SINGLE_STOCK_WEIGHT` /
`MAX_SECTOR_WEIGHT` 早就定义好了，但**从未对真实持仓做过校验**；页面上所谓"行业敞口"
是按**代码前缀**（60/00/30/68）粗分的，与真实行业无关 ⇒ 结构性问题（一只票占半个仓位、
钱全压在一个行业）在界面上**看不见**。

设计铁律（沿用本项目既有口径）：
  1. **阈值只认 `RiskEngine` 一处**（`_limits()` 直接读它的常量，本模块不重抄数字）。
  2. **缺就是缺**：行业取不到 ⇒ 该只**不参与**行业集中度计算，并列入 `missing_industry`；
     成交额取不到 ⇒ 流动性**不可评**（列入 `unavailable_liquidity`），**绝不当成 0**
     （当成 0 会算出"市值 ÷ 0 = 无穷大"，把"没数据"伪装成"最坏结论"）。
  3. **纯计算**：本模块函数不联网、不写库（`build_holdings()` 只读 DB，取价靠调用方透传）。
  4. 严重度用 `high`/`medium`，**刻意不复用** `core.portfolio.SEVERITY_WARN`
     —— `level_sanity_blocking()` 会把 `"warn"` 解读为"必须人工确认才能保存"，
     而本模块的结果是**只读提示**，任何地方都不该因此阻断用户操作。
"""

from __future__ import annotations

import logging
from typing import Optional

_logger = logging.getLogger(__name__)

SEV_HIGH = "high"
SEV_MEDIUM = "medium"


def _limits() -> dict:
    """阈值字典 —— **只读 `RiskEngine` 一处**，本模块不写死任何数字。

    ⚠️ 不在这里放 `getattr(..., 默认值)`：那等于把数字抄了第二份，
    改 RiskEngine 时会**静默**不生效（本项目踩过"配置改了但断言/实现没跟着变"）。
    """
    from core.risk import RiskEngine
    return {
        "single": RiskEngine.MAX_SINGLE_STOCK_WEIGHT,
        "sector": RiskEngine.MAX_SECTOR_WEIGHT,
        "liquidity_days": RiskEngine.MAX_LIQUIDITY_DAYS,
        "liquidity_caution_days": RiskEngine.LIQUIDITY_CAUTION_DAYS,
    }


# ================================================================
# 输入归一化
# ================================================================

def _mv(h: dict) -> float:
    """持仓市值（元）；非法/缺失 ⇒ 0（调用方随后会把它排除，不参与占比）。"""
    v = h.get("market_value")
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    return 0.0


def _amount(h: dict) -> Optional[float]:
    """当日成交额（元）。**缺失/非法 ⇒ None**（不是 0）—— 0 会让占比变成无穷大。"""
    v = h.get("amount")
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    return None


# ================================================================
# ① 集中度
# ================================================================

def concentration_issues(holdings: list[dict], total_equity: float,
                         *, limits: dict | None = None) -> list[dict]:
    """单票 / 行业集中度超限清单。

    Args:
        holdings: `[{"code","name","market_value","industry"(可为 None)}]`
        total_equity: 组合总资产（现金 + 持仓市值）。**≤0 ⇒ 无法评估，返回 []**
                      （不猜、不用持仓成本顶替）。

    Returns: `[{"kind":"single"|"sector","severity","code"|"sector","weight",
                "limit","members","message"}]`
    """
    lim = limits or _limits()
    if not (isinstance(total_equity, (int, float)) and total_equity > 0):
        return []

    issues: list[dict] = []

    # ── 单票 ──
    for h in sorted(holdings, key=lambda x: -_mv(x)):
        mv = _mv(h)
        if mv <= 0:
            continue
        w = mv / total_equity
        if w > lim["single"]:
            code = str(h.get("code") or "")
            name = h.get("name") or ""
            issues.append({
                "kind": "single", "severity": SEV_HIGH, "code": code, "name": name,
                "weight": round(w, 4), "limit": lim["single"],
                "members": [code],
                "message": (f"单票集中：**{code} {name}** 占净值 **{w:.1%}**"
                            f"（上限 {lim['single']:.0%}）⇒ 这一只的涨跌基本决定组合表现，"
                            f"个股黑天鹅无法通过分散化解。"),
            })

    # ── 行业（按真实行业；缺行业的持仓先列出，绝不并进"其他"里凑数）──
    by_sector: dict[str, list[str]] = {}
    for h in holdings:
        sec = str(h.get("industry") or "").strip()
        mv = _mv(h)
        if not sec or mv <= 0:
            continue
        by_sector.setdefault(sec, []).append(str(h.get("code") or ""))

    for sec, codes in sorted(by_sector.items(),
                             key=lambda kv: -_sector_mv(kv[0], holdings)):
        mv = _sector_mv(sec, holdings)
        w = mv / total_equity
        if w > lim["sector"]:
            issues.append({
                "kind": "sector", "severity": SEV_HIGH, "sector": sec,
                "weight": round(w, 4), "limit": lim["sector"], "members": codes,
                "message": (f"行业集中：**{sec}** 合计占净值 **{w:.1%}**"
                            f"（上限 {lim['sector']:.0%}，{len(codes)} 只："
                            f"{'、'.join(codes)}）⇒ 同一行业的系统性风险无法分散。"),
            })
    return issues


def _sector_mv(sector: str, holdings: list[dict]) -> float:
    return sum(_mv(h) for h in holdings
               if str(h.get("industry") or "").strip() == sector)


# ================================================================
# ② 流动性（退出所需天数）
# ================================================================

def liquidity_issues(holdings: list[dict],
                     *, limits: dict | None = None) -> tuple[list[dict], list[str]]:
    """流动性清单。口径：**持仓市值 ÷ 当日成交额**（天）。

    Returns: `(issues, unavailable_codes)`
        - `issues`: `[{"kind":"liquidity","severity","code","days","ratio","message",...}]`
        - `unavailable_codes`: 成交额取不到的持仓 —— 它们**不可评**，
          调用方必须如实展示，**不能**当成"没问题"（不列出来就等于默认宣称"流动没问题"）。
    """
    lim = limits or _limits()
    issues: list[dict] = []
    unavailable: list[str] = []

    for h in sorted(holdings, key=lambda x: -_mv(x)):
        mv = _mv(h)
        if mv <= 0:
            continue
        code = str(h.get("code") or "")
        amt = _amount(h)
        if amt is None:
            unavailable.append(code)
            continue
        days = mv / amt
        name = h.get("name") or ""
        if days >= lim["liquidity_days"]:
            issues.append({
                "kind": "liquidity", "severity": SEV_HIGH, "code": code, "name": name,
                "days": round(days, 2), "ratio": round(days, 4),
                "limit": lim["liquidity_days"],
                "message": (f"流动性风险：**{code} {name}** 持股市值相当于该股"
                            f" **{days:.2f} 个交易日**的全部成交额（警戒线 "
                            f"{lim['liquidity_days']:g} 天）⇒ 想一次性清仓，"
                            f"市场当天吃不下这么多，实际卖出价会显著低于现价。"),
            })
        elif days >= lim["liquidity_caution_days"]:
            issues.append({
                "kind": "liquidity", "severity": SEV_MEDIUM, "code": code, "name": name,
                "days": round(days, 2), "ratio": round(days, 4),
                "limit": lim["liquidity_caution_days"],
                "message": (f"流动性偏紧：**{code} {name}** 持股市值约为该股日成交额的"
                            f" **{days:.2f} 天**（≥{lim['liquidity_caution_days']:g} 天）"
                            f"⇒ 建议分笔挂单退出，避免一次性砸低价格。"),
            })
    return issues, unavailable


# ================================================================
# ③ 汇总报告
# ================================================================

def position_risk_report(holdings: list[dict], total_equity: float,
                         *, limits: dict | None = None) -> dict:
    """组合层体检报告（纯计算）。见模块 docstring 第 2、4 条铁律。"""
    lim = limits or _limits()
    equity_ok = isinstance(total_equity, (int, float)) and total_equity > 0
    equity = float(total_equity) if equity_ok else 0.0

    position_value = round(sum(_mv(h) for h in holdings), 2)
    pos_rows = [h for h in holdings if _mv(h) > 0]

    # 排序用的"单票占比表"（即使没超限也展示，用户才知道离上限多远）
    ranked = sorted(pos_rows, key=lambda x: -_mv(x))
    top = None
    if ranked and equity_ok:
        h = ranked[0]
        top = {"code": str(h.get("code") or ""), "name": h.get("name") or "",
               "market_value": round(_mv(h), 2),
               "weight": round(_mv(h) / equity, 4)}

    # 行业敞口（真实行业；无行业的记为 None，不并进"其他"）
    sector_exposure: dict[str, float] = {}
    missing_industry: list[str] = []
    for h in pos_rows:
        sec = str(h.get("industry") or "").strip()
        if not sec:
            missing_industry.append(str(h.get("code") or ""))
            continue
        if equity_ok:
            sector_exposure[sec] = round(
                sector_exposure.get(sec, 0.0) + _mv(h) / equity, 4)

    liq_issues, unavailable_liq = liquidity_issues(holdings, limits=lim)
    issues = concentration_issues(holdings, equity, limits=lim) + liq_issues

    report = {
        "evaluated": bool(equity_ok and pos_rows),
        "unavailable": None if equity_ok else (
            "总资产不可用（现价取不到或账户为空），无法计算占比 —— 未用成本价顶替"),
        "total_equity": round(equity, 2),
        "position_value": position_value,
        "position_ratio": round(position_value / equity, 4) if equity_ok else None,
        "cash_ratio": (round(max(0.0, (equity - position_value) / equity), 4)
                       if equity_ok else None),
        "positions": [{"code": str(h.get("code") or ""), "name": h.get("name") or "",
                       "market_value": round(_mv(h), 2),
                       "weight": (round(_mv(h) / equity, 4) if equity_ok else None),
                       "industry": h.get("industry") or None,
                       "amount": _amount(h),
                       "liquidity_days": (round(_mv(h) / _amount(h), 2)
                                          if _amount(h) else None)}
                      for h in ranked],
        "top_position": top,
        "sector_exposure": dict(sorted(sector_exposure.items(), key=lambda kv: -kv[1])),
        "issues": issues,
        "missing_industry": missing_industry,
        "unavailable_liquidity": unavailable_liq,
        "limits": dict(lim),
    }
    return report


# ================================================================
# ④ 只读取数：把 DB 里的持仓 + 估值截面拼成体检输入
# ================================================================

def build_holdings(prices: dict | None = None,
                   snapshot_date: str | None = None) -> dict:
    """从**数据库**读活跃持仓，并从 `market_snapshot` 取真实行业与当日成交额。

    ⚠️ 本函数**不联网取价**：`prices`（`{code: 现价}` 或页面行情字典）由调用方传入，
       缺的代码回落 `PortfolioManager._price_of`（旧行为）。

    ⚠️ `snapshot_date=None` 时用 `load_market_snapshot()` 返回的**实际日期** ——
       页面必须把它显示出来（"数据日期"），否则用户会把几天前的行业/成交额
       当今天的（本项目铁律：涉行情必给数据日期）。

    Returns: `{"holdings": [...], "snapshot_date": str|None, "missing_snapshot": [code],
               "total_equity": float, "cash": float, "unavailable": str|None}`
        - `missing_snapshot`：持仓里有、但当日截面**没有**的代码（新股/退市/未覆盖）
          ⇒ 这些的行业与成交额都是 `None`（不可评），**不猜**。
    """
    from core.portfolio import get_portfolio_manager
    from db.store import load_market_snapshot

    pm = get_portfolio_manager()
    try:
        pf = pm.load(prices=prices)
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[持仓体检] 读取持仓失败：%s: %s", type(e).__name__, e)
        return {"holdings": [], "snapshot_date": None, "missing_snapshot": [],
                "total_equity": 0.0, "cash": 0.0,
                "unavailable": f"读取持仓失败：{type(e).__name__}: {e}"}

    codes = [p.code for p in pf.positions]
    date, rows = (None, [])
    if codes:
        try:
            date, rows = load_market_snapshot(snapshot_date, codes)
        except Exception as e:                                # noqa: BLE001
            _logger.warning("[持仓体检] 读取估值截面失败：%s: %s", type(e).__name__, e)
    by_code = {str(r.get("stock_code")): r for r in (rows or [])}

    holdings: list[dict] = []
    missing: list[str] = []
    for p in pf.positions:
        px = pm._price_of(p.code, prices)
        snap = by_code.get(str(p.code))
        if snap is None:
            missing.append(str(p.code))
        holdings.append({
            "code": str(p.code),
            "name": p.name or "",
            "quantity": p.quantity,
            "current_price": px if px > 0 else None,
            "market_value": round(px * p.quantity, 2) if px > 0 else 0.0,
            "industry": (snap or {}).get("industry") or None,
            "amount": (snap or {}).get("amount") or None,
        })

    return {"holdings": holdings, "snapshot_date": date, "missing_snapshot": missing,
            "total_equity": float(pf.total_value or 0.0),
            "cash": float(pf.cash or 0.0), "unavailable": None}


def evaluate_positions(prices: dict | None = None,
                       snapshot_date: str | None = None) -> dict:
    """一步到位：`build_holdings()` + `position_risk_report()`（页面/AI 工具入口）。"""
    built = build_holdings(prices=prices, snapshot_date=snapshot_date)
    report = position_risk_report(built["holdings"], built["total_equity"])
    report.update({
        "snapshot_date": built["snapshot_date"],
        "missing_snapshot": built["missing_snapshot"],
        "cash": built["cash"],
        "data_unavailable": built["unavailable"],
    })
    return report
