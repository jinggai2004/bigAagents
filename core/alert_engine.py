"""盘中异动预警规则引擎（Phase 1）

输入：data.realtime.get_realtime_quotes 的输出（{code: row}）
输出：命中的预警列表，每条含 code / name / rule / level / message / price / change_pct / quote_time

设计要点：
  1. 阈值集中在 DEFAULT_RULES，可按需覆盖；
  2. **行情 stale=True（双源失败回退的过期缓存）时不产生任何交易类预警**，
     只发一条 data_stale。否则数据源一故障，就会按旧价刷出一堆假异动——
     这正是 0.4 加 stale 标记要防的事；
  3. 涨停幅度按板块判定（创业板/科创板 20%，其余 10%），与回测里的口径一致；
  4. MA20、止损价这类需要外部数据的规则通过参数传入，缺省即不判定：
     引擎本身不拉数据，保持纯函数、可单测。

本引擎只做"提示"，不下任何交易结论。
"""

import logging
from datetime import datetime

_logger = logging.getLogger(__name__)

DEFAULT_RULES = {
    "change_pct_surge": 5.0,       # 涨幅 ≥ 该值（%）
    "change_pct_plunge": -5.0,     # 跌幅 ≤ 该值（%）
    "amount_yi_spike": 10.0,       # 成交额 ≥ 该值（亿元）
    "turnover_rate_spike": 10.0,   # 换手率 ≥ 该值（%）
}

_LEVEL_ORDER = {"danger": 0, "warn": 1, "info": 2}


def limit_pct(code: str) -> float:
    """涨跌停幅度：创业板/科创板 20%，其余主板 10%（ST 的 5% 因数据无标记未纳入）"""
    return 0.20 if str(code).startswith(("300", "301", "688")) else 0.10


class AlertEngine:
    """把实时行情翻译成"值得看一眼"的异动事件"""

    def __init__(self, rules: dict = None):
        self.rules = {**DEFAULT_RULES, **(rules or {})}

    # ------------------------------------------------------------
    def evaluate(self, quotes: dict, ma20: dict = None, stop_loss: dict = None) -> list:
        """quotes: {code: row}；ma20/stop_loss: {code: value}（可选）"""
        ma20 = ma20 or {}
        stop_loss = stop_loss or {}
        alerts = []
        for raw, row in (quotes or {}).items():
            if not row:
                continue
            code = str(row.get("code") or raw)
            if row.get("stale"):
                alerts.append(self._mk(row, code, "data_stale", "warn",
                                       f"行情为过期缓存（{row.get('age_seconds', 0):.0f} 秒前），"
                                       f"已暂停异动判定"))
                continue
            alerts.extend(self._eval_one(row, code, ma20, stop_loss))
        alerts.sort(key=lambda a: (_LEVEL_ORDER.get(a["level"], 9), a["code"], a["rule"]))
        return alerts

    # ------------------------------------------------------------
    def _eval_one(self, row, code, ma20, stop_loss) -> list:
        out = []
        price = row.get("price")
        if not isinstance(price, (int, float)) or price <= 0:
            return out
        prev = row.get("prev_close") or 0
        pct = row.get("change_pct")
        pct_txt = f"{pct:+.2f}%" if isinstance(pct, (int, float)) else "N/A"

        # 涨跌停（与回测同口径）
        if prev > 0:
            lim = limit_pct(code)
            if price >= prev * (1 + lim) - 1e-6:
                out.append(self._mk(row, code, "limit_up", "danger", f"涨停 {pct_txt}"))
            elif price <= prev * (1 - lim) + 1e-6:
                out.append(self._mk(row, code, "limit_down", "danger", f"跌停 {pct_txt}"))

        if isinstance(pct, (int, float)):
            if pct >= self.rules["change_pct_surge"]:
                out.append(self._mk(row, code, "price_surge", "warn", f"涨幅 {pct_txt}"))
            elif pct <= self.rules["change_pct_plunge"]:
                out.append(self._mk(row, code, "price_plunge", "warn", f"跌幅 {pct_txt}"))

        amount_yi = (row.get("amount_wan") or 0) / 1e4
        if amount_yi >= self.rules["amount_yi_spike"]:
            out.append(self._mk(row, code, "volume_spike", "info", f"成交额 {amount_yi:.1f} 亿"))

        turnover = row.get("turnover_rate")
        if isinstance(turnover, (int, float)) and turnover >= self.rules["turnover_rate_spike"]:
            out.append(self._mk(row, code, "turnover_spike", "info", f"换手率 {turnover:.1f}%"))

        # 均线突破/跌破：用前收判断"是否当日发生"，避免整天重复报同一条
        m = ma20.get(code)
        if isinstance(m, (int, float)) and m > 0 and prev > 0:
            if price > m >= prev:
                out.append(self._mk(row, code, "break_ma20", "warn", f"上穿 MA20（{m:.2f}）"))
            elif price < m <= prev:
                out.append(self._mk(row, code, "breakdown_ma20", "warn", f"下穿 MA20（{m:.2f}）"))

        sl = stop_loss.get(code)
        if isinstance(sl, (int, float)) and sl > 0 and price <= sl:
            out.append(self._mk(row, code, "stop_loss", "danger", f"触及止损价 {sl:.2f}"))
        return out

    # ------------------------------------------------------------
    @staticmethod
    def _mk(row, code, rule, level, message) -> dict:
        return {
            "code": code,
            "name": row.get("name") or "",
            "rule": rule,
            "level": level,
            "message": message,
            "price": row.get("price"),
            "change_pct": row.get("change_pct"),
            "quote_time": row.get("quote_time") or "",
            "source": row.get("source") or "",
        }


_LEVEL_ICON = {"danger": "🔴", "warn": "🟠", "info": "🔵"}


def format_alerts_markdown(alerts: list, limit: int = 20, title: str = "盘中异动提醒",
                           relations: dict = None, notes: dict = None) -> str:
    """把预警列表渲染成可直接推送的 Markdown。

    Args:
        alerts: 预警列表。
        relations: `{code: "持仓·自选"}` —— **这条告警跟用户是什么关系**。
            没有它，用户看到一串陌生代码，无法判断该不该管（2026-10-23 加）。
        notes: `{code: "持仓 1000 股，止损 45.20（现价差 -3.2%）"}` —— 补充信息。

    ⚠️ 2026-10-23：外壳与**每条格式**都改由**可编辑模板**决定
    （`core/push_templates`，用户在「系统配置」页改，改完即时生效）。
    默认模板与本函数改造前的输出**内容一致**。
    """
    from core import push_templates as T
    relations = relations or {}
    notes = notes or {}
    if not alerts:
        return f"# {title}\n\n本次检查未命中任何异动。"
    # ⚠️ 模板**取一次**再逐条渲染：每条都查一次库是 N 次往返（每分钟一轮会放大）
    tpl_item = T.get_template(T.TPL_ALERT_ITEM)
    rows = []
    for a in alerts[:limit]:
        price = a.get("price")
        price_txt = f"{price:.2f}" if isinstance(price, (int, float)) else "N/A"
        code = str(a.get("code") or "")
        rows.append(T.render(tpl_item, {
            "icon": _LEVEL_ICON.get(a.get("level"), "•"),
            "name": a.get("name", ""), "code": code,
            "message": a.get("message", ""), "price": price_txt,
            "quote_time": a.get("quote_time", ""),
            "relation": relations.get(code) or "",
            "note": notes.get(code) or "",
        }))
    items = "\n".join(rows)
    if len(alerts) > limit:
        items += f"\n\n…另有 {len(alerts) - limit} 条未列出"
    return T.render(T.get_template(T.TPL_ALERT), {
        "title": title, "count": str(len(alerts)),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M"),
        "items": items,
        "footer": T.FOOTERS.get(T.TPL_ALERT, ""),
    })
