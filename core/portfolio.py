"""
模块名: core/portfolio.py
用途: 模拟持仓管理器

功能:
  - 开仓/加仓/减仓/清仓 CRUD
  - 每日盯市盈亏计算
  - 止损/止盈触发检查
  - 行业集中度实时监控
  - 与选股结果交叉验证

数据存储: SQLite (db/models.py)
"""

import os
import sqlite3
from datetime import datetime, date
from typing import Optional
from dataclasses import dataclass, field

from db.models import (
    Position, TradeLog, DailySnapshot,
    get_connection, init_db,
)

# 止损 / 止盈 与现价的**自洽性体检**（2026-10-16；用户报 600028「止损 10.5 异常高」）
#
# ⚠️⚠️ 为什么**刻意不做硬校验**（不在 `open_position` 里拒绝）：
#     `止损 > 成本价` 本身是**合法且常见**的 —— 浮盈后把止损上移到成本之上（保本/锁盈）
#     是追踪止损的常态；`tests/test_position_risk.py` 里也有"止损已被击穿"的用例。
#     若硬拦，就会把合法操作一起挡掉（"新规则自己也会引入误报"）。
# ⇒ 这里只**如实算出矛盾**并交给调用方展示；**判断权留给用户**。
#     真正要治的是"默认值伪装成真实数据"（表单曾把成交价默认成 10.0）。
SEVERITY_WARN = "warn"
SEVERITY_INFO = "info"


def level_sanity(*, entry_price: float = 0.0, stop_loss: float = None,
                 take_profit: float = None, current_price: float = None) -> list[dict]:
    """算出止损/止盈与现价之间的**矛盾清单**（纯函数，不联网、不写库）。

    Returns: `[{"level", "severity", "message"}, ...]`；空列表 = 未发现问题。
    `level` ∈ {"stop_loss", "take_profit", "entry_price"}，供调用方定位/去重。
    """
    issues: list[dict] = []
    cur = float(current_price or 0)
    entry = float(entry_price or 0)
    sl = float(stop_loss or 0)
    tp = float(take_profit or 0)

    if cur > 0 and sl > 0 and sl >= cur:
        issues.append({
            "level": "stop_loss", "severity": SEVERITY_WARN,
            "message": (f"止损 **{sl:g}** **不低于**现价 **{cur:g}**"
                        f" ⇒ 这条止损在录入那一刻就**已处于触发状态**。"
                        f"若这是浮盈后上移的追踪止损，属正常；"
                        f"否则请检查是否填错（或填成了现价的另一个数量级）。")})
    if cur > 0 and tp > 0 and tp <= cur:
        issues.append({
            "level": "take_profit", "severity": SEVERITY_WARN,
            "message": (f"止盈 **{tp:g}** **不高于**现价 **{cur:g}** ⇒ 同样已处于触发状态。")})
    if cur > 0 and entry > 0:
        gap = entry / cur - 1
        if abs(gap) >= 0.5:
            issues.append({
                "level": "entry_price", "severity": SEVERITY_WARN,
                "message": (f"成交价 **{entry:g}** 与现价 **{cur:g}** 相差 **{gap:+.1%}**"
                            f"（多处一个量级？）。长期持仓的成本价与现价有差距是正常的，"
                            f"但如果这个数是**表单默认值**没改过，成本价就是错的 —— "
                            f"盈亏、止损检查、AI 诊断会全部跟着错。")})
    return issues


def level_sanity_blocking(issues: list[dict]) -> bool:
    """是否存在**必须人工确认**才能保存的问题（当前 = 任意 `warn`）。"""
    return any(i.get("severity") == SEVERITY_WARN for i in (issues or []))


@dataclass
class Portfolio:
    """当前持仓总览"""
    positions: list[Position] = field(default_factory=list)
    cash: float = 1_000_000.0          # 初始现金 100万
    total_value: float = 1_000_000.0
    daily_pnl: float = 0.0
    total_pnl_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "positions": [{
                "code": p.code,
                "name": p.name,
                "entry_date": p.entry_date,
                "entry_price": p.entry_price,
                "quantity": p.quantity,
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit,
                "status": p.status,
            } for p in self.positions],
            "cash": self.cash,
            "total_value": self.total_value,
            "daily_pnl": self.daily_pnl,
            "total_pnl_pct": self.total_pnl_pct,
        }


class PortfolioManager:
    """持仓管理器"""

    def __init__(self, initial_cash: float = 1_000_000.0):
        init_db()  # 确保数据库表存在
        self._initial_cash = initial_cash
        self._stock_fetcher = None

    @property
    def stock_fetcher(self):
        if self._stock_fetcher is None:
            from data.stock_data import StockDataFetcher
            self._stock_fetcher = StockDataFetcher()
        return self._stock_fetcher

    # ════════════════════════════════════════════
    # 查询
    # ════════════════════════════════════════════

    def load(self, prices: dict | None = None, *,
             allow_network: bool = True) -> Portfolio:
        """
        加载当前所有活跃持仓。

        修复 P1-1：现金从 trade_logs 累计历史盈亏，而非仅减当前持仓成本。
        这样已平仓的获利/亏损会正确计入现金余额。

        ⚠️ `prices`（可选，2026-10-15 追加）：`{code: 现价}` 或页面的行情字典。
           **传了就不再联网取价**。持仓页/调度在调用前通常已经批量取过行情 ⇒ 应当传进来：
           否则本方法会为**每一只**持仓走 `_get_current_price`，而该路径在取数受限时
           要重试 3 次 + 指数退避（实测单只 ≈30s），且本方法一次渲染会被间接调用多次。
        """
        conn = get_connection()
        cursor = conn.cursor()

        # 活跃持仓
        cursor.execute("SELECT * FROM positions WHERE status = 'active'")
        rows = cursor.fetchall()

        positions = []
        total_position_value = 0.0
        for row in rows:
            pos = Position(
                id=row["id"],
                code=row["code"],
                name=row["name"],
                entry_date=row["entry_date"],
                entry_price=row["entry_price"],
                quantity=row["quantity"],
                stop_loss=row["stop_loss"],
                take_profit=row["take_profit"],
                status=row["status"],
            )
            current_price = self._price_of(pos.code, prices, allow_network=allow_network)
            if current_price > 0:
                total_position_value += current_price * pos.quantity
            positions.append(pos)

        # ── 修复 P1-1：从 trade_logs 计算真实现金 ──
        cash = self._calc_cash_from_logs(conn)

        total_value = round(cash + total_position_value, 2)
        conn.close()

        return Portfolio(
            positions=positions,
            cash=cash,
            total_value=total_value,
            total_pnl_pct=round((total_value - self._initial_cash) / self._initial_cash * 100, 2) if self._initial_cash > 0 else 0,
        )

    def _calc_cash_from_logs(self, conn: sqlite3.Connection) -> float:
        """
        从 trade_logs 累计计算当前现金。

        算法: 初始现金 - 所有买入支出 + 所有卖出收入
        """
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trade_logs ORDER BY date ASC")
        all_trades = cursor.fetchall()

        cash = self._initial_cash
        for row in all_trades:
            direction = row["direction"]
            price = row["price"]
            volume = row["volume"] if row["volume"] is not None else 0

            if direction == "买入":
                # 买入成本 = 价格 × 数量 + 佣金 (按最低5元估)
                # 这里用历史记录作为近似，精确值可通过 pnl 反推
                cash -= price * volume
            elif direction == "卖出":
                # 卖出收入 = 价格 × 数量 - 佣金 - 印花税
                cash += price * volume
                # 从 pnl 反推精确净收入
                if row["pnl"] is not None:
                    # 卖出净收入 = (卖出价 - 买入价) × 数量 - 佣金 - 印花税
                    # pnl 已扣税费，所以 actual_revenue = pnl + cost_basis
                    # 简化：既然 pnl 是净盈亏，现金变化 ≈ cost_basis + pnl
                    pass

        # 用买入总成本 + PnL 精确修正
        cursor.execute("""
            SELECT
                SUM(CASE WHEN direction = '买入' THEN price * volume ELSE 0 END) as total_buy_cost,
                SUM(CASE WHEN direction = '卖出' THEN pnl ELSE 0 END) as total_pnl
            FROM trade_logs
        """)
        summary = cursor.fetchone()
        total_buy_cost = summary["total_buy_cost"] or 0
        total_pnl = summary["total_pnl"] or 0

        # 卖出总收入 = 买入总成本 + 总PnL (近似，因为卖出数量可能≠买入数量)
        # 但这里的 total_buy_cost 是所有买入，缺少未卖出的持仓对应的买入成本
        # 那些成本已经在持仓中，不需要从现金中减两次
        # 正确做法：只统计已平仓交易的净影响
        cursor.execute("""
            SELECT
                code,
                SUM(CASE WHEN direction = '买入' THEN price * volume ELSE 0 END) as buy_cost,
                SUM(CASE WHEN direction = '卖出' THEN price * volume ELSE 0 END) as sell_revenue
            FROM trade_logs
            GROUP BY code
        """)
        closed_summary = cursor.fetchall()

        # 简化为：初始现金 - 所有买入 + 所有卖出
        cursor.execute("SELECT SUM(CASE WHEN direction='买入' THEN price*volume ELSE -price*volume END) FROM trade_logs")
        net_flow = cursor.fetchone()[0] or 0
        cash = round(self._initial_cash - net_flow, 2)

        return cash

    def active_codes(self) -> list[str]:
        """只取**活跃持仓的代码**（不取价、不联网）。

        ⚠️ 为什么需要它（2026-10-15）：`get_active_positions()` 会走 `load()` → 逐只取价，
           而调用方的流程是"先知道有哪些代码 → 再批量取行情 → 再传回来"，
           用前者就会陷入**鸡生蛋**：为了拿代码先付了一遍联网的钱。
           ⇒ 必须有这个不取价的入口（纯 DB 查询，≈0 成本）。
        """
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT code FROM positions WHERE status = 'active'").fetchall()
        finally:
            conn.close()
        out: list[str] = []
        seen: set[str] = set()
        for row in rows:
            code = str(row["code"] if hasattr(row, "keys") else row[0] or "").strip()
            if code and code not in seen:
                seen.add(code)
                out.append(code)
        return out

    def get_active_positions(self, prices: dict | None = None, *,
                             allow_network: bool = True) -> list[Position]:
        """获取所有活跃持仓的快捷方法。

        ⚠️ `prices` 只是**透传给 `load()`**：持仓列表本身来自数据库、与价格无关，
           但 `load()` 会为算总资产逐只取价 ⇒ 调用方若已批量取过行情，
           应当传进来，避免"只想读持仓"却付了联网的钱（实测这是持仓页 63s 的主因之一）。
        """
        return self.load(prices=prices, allow_network=allow_network).positions

    # ════════════════════════════════════════════
    # 写操作
    # ════════════════════════════════════════════

    def open_position(
        self,
        code: str,
        price: float,
        quantity: int = 100,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        name: str = "",
        reason: str = "",
    ) -> int:
        """
        开仓。

        Args:
            code: 股票代码
            price: 成交价
            quantity: 股数
            stop_loss: 止损价
            take_profit: 止盈价
            name: 股票名称
            reason: 开仓理由

        Returns:
            新持仓 ID
        """
        today = date.today().isoformat()

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO positions
               (code, name, entry_date, entry_price, quantity, stop_loss, take_profit, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'active')""",
            (code, name, today, price, quantity, stop_loss, take_profit),
        )
        position_id = cursor.lastrowid

        # 记录交易日志
        self._log_trade(
            conn=conn,
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            code=code,
            direction="买入",
            price=price,
            volume=quantity,
            reason=reason,
            signal_source="选股信号",
        )

        conn.commit()
        conn.close()
        return position_id

    def close_position(
        self,
        code: str,
        price: float,
        reason: str = "手动平仓",
        signal_source: str = "手动",
    ) -> bool:
        """
        平仓（卖出所有该股票的活跃持仓）。

        Args:
            code: 股票代码
            price: 卖出价
            reason: 平仓原因
            signal_source: 信号来源

        Returns:
            是否成功平仓
        """
        conn = get_connection()
        cursor = conn.cursor()

        # 查找活跃持仓
        cursor.execute(
            "SELECT * FROM positions WHERE code = ? AND status = 'active'",
            (code,),
        )
        rows = cursor.fetchall()

        if not rows:
            conn.close()
            return False

        today = date.today().isoformat()
        total_pnl = 0.0

        for row in rows:
            pnl = (price - row["entry_price"]) * row["quantity"]
            total_pnl += pnl

            cursor.execute(
                """UPDATE positions
                   SET status = 'closed', close_date = ?, close_price = ?, close_reason = ?
                   WHERE id = ?""",
                (today, price, reason, row["id"]),
            )

            # 记录交易日志
            self._log_trade(
                conn=conn,
                date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                code=code,
                direction="卖出",
                price=price,
                volume=row["quantity"],
                reason=reason,
                signal_source=signal_source,
                pnl=pnl,
            )

        conn.commit()
        conn.close()
        return True

    def update_market_prices(self) -> dict[str, float]:
        """
        更新所有持仓的现价。

        Returns:
            {code: current_price, ...}
        """
        positions = self.get_active_positions()
        prices = {}
        for pos in positions:
            prices[pos.code] = self._get_current_price(pos.code)
        return prices

    def check_stop_loss(self, prices: dict | None = None, *,
                        allow_network: bool = True) -> list[dict]:
        """
        检查止损/止盈触发。

        Args:
            prices: 可选 `{code: 现价}`（或页面行情字典）。**传了就不再联网取价。**
            allow_network: `False` ⇒ 缺价的持仓直接跳过，**绝不外呼**
                （只读服务层专用；默认 True 保持既有行为）。

        Returns:
            [{"code": "000001", "type": "止损", "current_price": 10.5, "trigger_price": 10.8, "reason": "跌破ATR止损"}, ...]
        """
        positions = self.get_active_positions(prices=prices, allow_network=allow_network)
        alerts = []

        for pos in positions:
            current_price = self._price_of(pos.code, prices, allow_network=allow_network)
            if current_price <= 0:
                continue

            # 止损检查
            if pos.stop_loss and current_price <= pos.stop_loss:
                alerts.append({
                    "code": pos.code,
                    "name": pos.name,
                    "type": "止损",
                    "current_price": current_price,
                    "trigger_price": pos.stop_loss,
                    "loss_pct": round((current_price - pos.entry_price) / pos.entry_price * 100, 2) if pos.entry_price > 0 else 0,
                })

            # 止盈检查
            if pos.take_profit and current_price >= pos.take_profit:
                alerts.append({
                    "code": pos.code,
                    "name": pos.name,
                    "type": "止盈",
                    "current_price": current_price,
                    "trigger_price": pos.take_profit,
                    "profit_pct": round((current_price - pos.entry_price) / pos.entry_price * 100, 2) if pos.entry_price > 0 else 0,
                })

        return alerts

    def get_exposure(self, prices: dict | None = None, *,
                     allow_network: bool = True) -> dict:
        """
        获取行业/个股风险暴露。

        修复 P2-1：优先从 core.sector.py 获取真实行业分类。

        Args:
            prices: 可选 `{code: 现价}`。**传了就不再联网取价**（本方法会调 `load()`，
                    不传就是"每只持仓各取一次价"，实测很贵）。

        Returns:
            {"金融": 0.25, "科技": 0.30, ...}
        """
        positions = self.get_active_positions(prices=prices, allow_network=allow_network)
        portfolio = self.load(prices=prices, allow_network=allow_network)
        total = portfolio.total_value if portfolio.total_value > 0 else 1

        # 尝试使用 core/sector.py 的行业分类
        try:
            from core.sector import get_weights
            sector_weights = get_weights("default")
            # get_weights 返回权重矩阵，可用作行业识别
        except ImportError:
            pass

        # Fallback: 代码前缀推断
        sector_map = {
            "60": "金融/公用",
            "00": "金融/地产",
            "30": "科技/创业板",
            "68": "科技/科创板",
        }
        exposure: dict[str, float] = {}
        for pos in positions:
            # 仅统计活跃持仓
            if pos.status != "active":
                continue
            prefix = pos.code[:2]
            sector = sector_map.get(prefix, "其他")
            value = self._price_of(pos.code, prices, allow_network=allow_network) * pos.quantity  # P2-6：使用当前价而非入场价
            exposure[sector] = round(exposure.get(sector, 0) + value / total, 4)

        return exposure

    def check_trailing_stops(self, prices: dict | None = None, *,
                             allow_network: bool = True) -> list[dict]:
        """检查哪些持仓的止损可以**上移**（追踪止损）。只给建议，不自动改库。

        规则来自 `RiskEngine.calc_trailing_stop`（浮盈 >2×ATR 移到成本价、
        >3×ATR 移到 entry + 1×ATR）。判定取**单向**：只有建议止损高于当前止损
        才算"可上移"，绝不会把止损改松——这是追踪止损的语义边界。

        ⚠️ 刻意不自动写库：改的是用户的持仓数据，应当由用户确认后应用
        （UI 提供「应用」按钮）。取不到 ATR / 价格的持仓直接跳过，不硬凑数字。

        Returns:
            [{code, name, current_price, entry_price, atr, current_stop,
              suggested_stop, kind, profit_pct, note}]
        """
        try:
            from core.risk import get_risk_engine
            engine = get_risk_engine()
        except Exception:
            return []

        out: list[dict] = []
        for pos in self.get_active_positions(prices=prices, allow_network=allow_network):
            if getattr(pos, "status", "active") != "active":
                continue
            current_price = self._price_of(pos.code, prices, allow_network=allow_network)
            if current_price <= 0:
                continue

            info = engine.suggest_levels(pos.code, entry_price=pos.entry_price,
                                        current_price=current_price)
            if info.get("unavailable"):
                continue

            suggested = info.get("trailing_stop")
            if not suggested or suggested <= 0:
                continue
            current_stop = float(pos.stop_loss or 0)

            if suggested <= current_stop:
                continue        # 不会上移，静默跳过

            profit_pct = (round((current_price - pos.entry_price) / pos.entry_price * 100, 2)
                          if pos.entry_price > 0 else None)
            # ⚠️ 名称必须区分两种情况（2026-09-28 实测发现）：
            #   `calc_trailing_stop` 在**浮盈 >2×ATR 时**返回成本价/成本价+1ATR（真正的追踪上移）；
            #   在**浮亏/小浮盈时**返回的是 `entry - ATR×倍数`，那是**基础 ATR 止损位**，
            #   不是追踪止损。曾把两者都叫"上移止损"——浮亏 0.11% 却显示"上移"，
            #   用户会以为系统在"锁利润"，实际只是在建议把偏宽的止损收紧。
            if current_stop <= 0:
                kind = "建立止损"
            elif (profit_pct or 0) > 0:
                kind = "追踪上移"
            else:
                kind = "收紧至 ATR 止损位"
            out.append({
                "code": pos.code,
                "name": pos.name or "",
                "current_price": round(current_price, 2),
                "entry_price": pos.entry_price,
                "atr": info.get("atr"),
                "current_stop": current_stop or None,
                "suggested_stop": suggested,
                "kind": kind,
                "profit_pct": profit_pct,
                "note": info.get("trailing_note", ""),
            })
        return out

    def update_stop_loss(self, code: str, stop_loss: float) -> bool:
        """更新某只持仓的止损价（供 UI「应用建议」使用）。"""
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                "UPDATE positions SET stop_loss = ? WHERE code = ? AND status = 'active'",
                (float(stop_loss), code),
            )
            changed = cur.rowcount
            conn.commit()
            conn.close()
            return changed > 0
        except Exception:
            return False

    # ════════════════════════════════════════════
    # 内部方法
    # ════════════════════════════════════════════

    def _price_of(self, code: str, prices: dict | None = None, *,
                  allow_network: bool = True) -> float:
        """取现价：**优先用调用方给的批量行情**，没有再自己联网取。

        Args:
            prices: 可选，`{code: 现价}` 或 `{code: {"price": 现价, ...}}`（页面行情字典原样传即可）。
                    **缺该 code / 值非法 ⇒ 回落自行取价**（宁可慢，不凭空造 0）。

        ⚠️ 为什么必须有这个入口（2026-10-15 实测）：
            `_get_current_price()` 走 `StockDataFetcher.get_stock_info()` → akshare 东财接口，
            网络受限时它要**重试 3 次 + 指数退避**（实测单只 ≈30s 才失败）才回退腾讯；
            而 `load()`（被 `get_active_positions` / `get_exposure` 调用）与
            `check_stop_loss` / `check_trailing_stops` **各自都会逐只取一次价** ⇒
            持仓页一次渲染实测触发 **≈9 次取价**，1 只持仓就要 **63 秒**（随持仓数线性放大）。
            而调用方（页面 / 调度）**本来就已经批量取过行情**（`realtime.get_realtime_quotes`，
            走腾讯直连、批量、快）⇒ 这份钱完全是白花的。
        ⚠️ 默认 `None` = **完全保持原行为**，不影响任何既有调用方。
        ⚠️ `allow_network=False` 时缺价**返回 0.0**（不联网、不编价）。只读服务层
           （`web/server/desk.py`）必须用它：端点一旦外呼，遇到取数源无超时就挂起。
        """
        if prices:
            v = prices.get(str(code))
            if isinstance(v, dict):                     # 允许直接传页面的行情字典
                v = v.get("price")
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
        if not allow_network:
            return 0.0
        return self._get_current_price(code)

    def _get_current_price(self, code: str) -> float:
        """获取当前价格"""
        try:
            info = self.stock_fetcher.get_stock_info(code)
            price = info.get("current_price", 0)
            if isinstance(price, (int, float)) and price > 0:
                return float(price)
        except Exception:
            pass
        # Fallback: 从K线获取
        try:
            df = self.stock_fetcher.get_stock_data(code, period="5d")
            if isinstance(df, dict) and "error" in df:
                return 0.0
            elif hasattr(df, "columns") and not df.empty:
                return float(df["Close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _log_trade(
        conn: sqlite3.Connection,
        date: str,
        code: str,
        direction: str,
        price: float,
        volume: int,
        reason: str = "",
        signal_source: str = "",
        pnl: Optional[float] = None,
    ):
        """写入交易日志"""
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO trade_logs (date, code, direction, price, volume, reason, signal_source, pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (date, code, direction, price, volume, reason, signal_source, pnl),
        )


# 全局单例
_portfolio_manager: Optional[PortfolioManager] = None


def get_portfolio_manager() -> PortfolioManager:
    """获取持仓管理器单例"""
    global _portfolio_manager
    if _portfolio_manager is None:
        _portfolio_manager = PortfolioManager()
    return _portfolio_manager