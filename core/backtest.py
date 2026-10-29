"""
模块名: core/backtest.py
用途: 真实事件驱动回测引擎

功能:
  1. 从 SQLite 加载历史交易记录
  2. 事件驱动模拟（分批成交、滑点、税费）
  3. 输出净值曲线 + 逐年/逐月收益
  4. 与沪深300基准对比
  5. 因子收益归因

交易成本:
  - 印花税: 卖出 0.1%
  - 佣金: 买入/卖出 0.025%
  - 滑点: 0.05% (模拟实际成交与信号价的偏差)
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from db.models import get_connection, init_db
from core import matching
# 交易成本参数：唯一实现在 core/matching.py（回测与模拟盘共用，见 Phase 2 硬约束），
# 这里重新导出以保持既有引用（如 core/backtest.SLIPPAGE_RATE）不失效。
from core.matching import (STAMP_TAX_RATE, COMMISSION_RATE,
                           SLIPPAGE_RATE, MIN_COMMISSION)

_logger = logging.getLogger(__name__)

# ── 项目根目录 ──
# 数据根必须**运行时**解析：打包后 `__file__` 位于只读包 `_internal/` 里，
# 用它推导会指向空目录 ⇒ 读到空数据（2026-10-25 实测：双击 exe 后
# 选股/复盘/研究全 missing、交易台变空账户、系统页看不到调度器）。
# 规则见 `config/paths.data_root()`；源码运行结果与原先一致。
from config.paths import data_root as _paths_data_root  # noqa: E402
_PROJECT_ROOT = Path(_paths_data_root())
RECORDS_DIR = str(_PROJECT_ROOT / "data" / "backtest_records")


def _period_covering(start_date: str) -> str:
    """把"需要覆盖到 start_date"翻译成取数接口的 period 参数。

    v3（P1-6）：`StockDataFetcher.get_stock_data` 的 period 终点固定是**今天**，
    原实现写死 `period="3mo"` —— 回测 2024 年时取到的是"今天往前 90 天"，
    历史日期一条都命不中，于是全部落回成本价（净值虚高、浮亏被抹掉）。
    这里按回测起点动态算跨度，保证区间被覆盖。起点在未来或解析失败时退到 1 年。
    """
    if not start_date:
        return "1y"
    try:
        days = (datetime.now() - pd.to_datetime(start_date)).days
    except Exception:                                   # noqa: BLE001
        return "1y"
    if days <= 90:
        return "3mo"
    if days <= 180:
        return "6mo"
    return f"{max(1, int(days / 365) + 1)}y"


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float = 0.0         # 总收益率 (%)
    annual_return: float = 0.0        # 年化收益率 (%)
    sharpe_ratio: float = 0.0         # 夏普比率
    calmar_ratio: float = 0.0         # 卡玛比率
    max_drawdown: float = 0.0         # 最大回撤 (%)
    win_rate: float = 0.0             # 胜率 (%)
    profit_loss_ratio: float = 0.0    # 盈亏比
    total_trades: int = 0             # 总交易次数
    turnover_rate: float = 0.0        # 年化换手率 (%)
    # v3（2026-10-29，回测验证设计模块 C）：累计费用拖累。
    # ⚠️ turnover_rate 已有（F4 评审修正，别重复造）；本字段只补"毛 vs 净"的差——
    #    Σ(佣金 + 印花税) / 初始资金 ×100，口径是**累计占初始资金**的百分点，
    #    不是年化（复利下不可直接加回年化收益，解读时按"净收益里有多少被吃掉"用）。
    cost_drag: float = 0.0            # 累计费用拖累（占初始资金 %）
    information_ratio: float = 0.0    # 信息比率
    monthly_win_rate: float = 0.0     # 月度胜率 (%)
    # v3（P1-4）：被拒订单数与平均仓位。
    # 原实现买不起就静默跳过——策略以为买了 5 只、实际成交 3 只，回测结果与策略意图
    # 不符且无从察觉。这两个字段让"没成交"和"没满仓"变成可观测的事实。
    rejected_orders: int = 0          # 因现金不足被拒的买单数
    avg_position_pct: float = 0.0     # 平均仓位 (%)：见 P1-5，半仓口径必须披露
    nav_curve: list[dict] = field(default_factory=list)  # 净值曲线
    benchmark_nav: list[dict] = field(default_factory=list)   # 基准净值
    yearly_returns: dict = field(default_factory=dict)  # 逐年收益
    monthly_returns: dict = field(default_factory=dict)  # 逐月收益
    trade_history: list[dict] = field(default_factory=list)   # 交易明细

    def to_dict(self) -> dict:
        # 按时间窗口采样（首尾保留，中间等距采样），而非硬截断
        nav_sampled = self._sample_list(self.nav_curve, 100)
        trade_sampled = self._sample_list(self.trade_history, 50)
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_loss_ratio": self.profit_loss_ratio,
            "total_trades": self.total_trades,
            "turnover_rate": self.turnover_rate,
            "cost_drag": self.cost_drag,
            "information_ratio": self.information_ratio,
            "monthly_win_rate": self.monthly_win_rate,
            "rejected_orders": self.rejected_orders,
            "avg_position_pct": self.avg_position_pct,
            "nav_curve": nav_sampled,
            "benchmark_nav": self._sample_list(self.benchmark_nav, 100) if self.benchmark_nav else [],
            "yearly_returns": self.yearly_returns,
            "trade_history": trade_sampled,
        }

    @staticmethod
    def _sample_list(items: list, max_count: int) -> list:
        """等距采样，保证首尾保留"""
        n = len(items)
        if n <= max_count:
            return items
        indices = [int(i * (n - 1) / (max_count - 1)) for i in range(max_count)]
        return [items[i] for i in indices]


class BacktestEngine:
    """事件驱动回测引擎"""

    def __init__(self, records_dir: str = RECORDS_DIR):
        os.makedirs(records_dir, exist_ok=True)
        self.records_dir = records_dir
        self._stock_fetcher = None
        self._kline_cache: dict[str, "pd.DataFrame"] = {}
        self._benchmark_nav = None
        # v3（P0-1）：逐日估值会反复取同一只票的价格，这里缓存归一化后的
        # (日期数组, 收盘价数组)，把每次查价从"重建 DataFrame 副本"降为二分查找。
        self._price_series_cache: dict[str, tuple] = {}
        # 取数失败的标的只尝试一次，避免逐日循环里反复联网
        self._kline_failed: set[str] = set()

    def inject_kline_cache(self, cache: dict[str, "pd.DataFrame"]):
        """注入外部预加载的K线缓存，避免重复API调用。"""
        self._kline_cache.update(cache)
        self._price_series_cache.clear()
        self._kline_failed.difference_update(cache.keys())

    @property
    def stock_fetcher(self):
        if self._stock_fetcher is None:
            from data.stock_data import StockDataFetcher
            self._stock_fetcher = StockDataFetcher()
        return self._stock_fetcher

    # ════════════════════════════════════════════
    # 公共入口
    # ════════════════════════════════════════════

    def run(self, trades: Optional[list] = None,
            start_date: str = "", end_date: str = "") -> BacktestResult:
        """
        执行回测计算。

        Args:
            trades: 交易记录列表，格式:
                    [{"date": "2026-07-20", "code": "000001", "direction": "买入",
                      "price": 12.50, "quantity": 1000}, ...]
                    如果为 None，从数据库加载
            start_date: 回测区间起点（YYYY-MM-DD）。留空则取最早成交日。
            end_date:   回测区间终点（YYYY-MM-DD）。留空则取最晚成交日。
                        **显式给区间很重要**：逐日估值需要一个明确的终点，
                        否则"买入后一直持有、样本结束时仍持仓"的区间会只算到买入那天，
                        净值曲线被截断、回撤又变成 0。

        Returns:
            BacktestResult
        """
        # 加载交易记录
        if trades is None:
            trades = self._load_from_db()

        if not trades:
            return BacktestResult()

        # 排序
        trades.sort(key=lambda t: t.get("date", ""))

        # ── 事件驱动模拟 ──
        initial_cash = 1_000_000.0
        cash = initial_cash
        holdings: dict[str, dict] = {}  # {code: {"quantity": int, "total_cost": float}}
        nav_curve = []
        trade_details = []
        rejected_orders = 0

        # v3（P0-1）：按**交易日历逐日**推进，而不是只遍历"有交易的日子"。
        # 原实现的 nav_curve 只在有成交的日子追加一个点：持有一只票 30 个交易日、
        # 期间跌 30% 再涨回来，曲线上只有首尾两点、**回撤记为 0**。
        # 后果是 max_drawdown 被系统性低估 → calmar 被高估，且越是不频繁交易的
        # 策略失真越重（HOLDING_DAYS_MAX 最长 30 个交易日）。
        # 注意：这个偏差方向是**偏乐观**，比偏保守危险——它让策略看起来更能扛。
        calendar = self._build_calendar(trades, start_date, end_date)

        # 同日先卖后买——原实现按传入顺序处理，买入可能排在卖出之前，
        # 于是当日卖出回笼的现金无法用于当日买入，并可能造成"现金不足"的静默丢弃。
        trades_by_date: dict[str, list] = {}
        for t in trades:
            trades_by_date.setdefault(t.get("date", ""), []).append(t)
        for _d, _items in trades_by_date.items():
            _items.sort(key=lambda t: 0 if t.get("direction") == "卖出" else 1)

        last_price: dict[str, float] = {}   # 最近一次有效价（顺延用）
        no_data_warned: set[str] = set()

        for trade_date in calendar:
            for trade in trades_by_date.get(trade_date, []):
                code = trade.get("code", "")
                direction = trade.get("direction", "")
                signal_price = trade.get("price", 0)
                quantity = trade.get("quantity", 100)

                if direction == "买入":
                    fill = matching.buy_fill(signal_price, quantity)

                    if matching.can_afford(cash, fill):
                        cash += fill["cash_delta"]
                        if code not in holdings:
                            holdings[code] = {"quantity": 0, "total_cost": 0}
                        holdings[code]["quantity"] += fill["quantity"]
                        holdings[code]["total_cost"] += fill["total"]
                        last_price[code] = fill["actual_price"]

                        trade_details.append({
                            "date": trade_date,
                            "code": code,
                            "direction": "买入",
                            "status": "filled",
                            "signal_price": signal_price,
                            "actual_price": fill["actual_price"],
                            "quantity": fill["quantity"],
                            "cost": fill["cost"],
                            "commission": fill["commission"],
                            "cash_after": round(cash, 2),
                        })
                    else:
                        # v3（P1-4）：买不起就**记录下来**，不再静默丢弃。
                        # 原实现没有 else：策略以为买了 5 只、实际只成交 3 只，
                        # 回测结果与策略意图不符且无从察觉。
                        rejected_orders += 1
                        trade_details.append({
                            "date": trade_date,
                            "code": code,
                            "direction": "买入",
                            "status": "rejected",
                            "reason": "现金不足",
                            "signal_price": signal_price,
                            "actual_price": fill["actual_price"],
                            "quantity": fill["quantity"],
                            "required": round(fill["total"], 2),
                            "cash_before": round(cash, 2),
                        })

                elif direction == "卖出":
                    if code not in holdings or holdings[code]["quantity"] <= 0:
                        continue

                    sell_quantity = min(quantity, holdings[code]["quantity"])
                    fill = matching.sell_fill(signal_price, sell_quantity)

                    # 计算盈亏（成交价 - 持仓均价，扣除佣金与印花税）
                    avg_entry = holdings[code]["total_cost"] / holdings[code]["quantity"]
                    pnl = matching.pnl_of(fill, avg_entry)

                    cash += fill["cash_delta"]
                    last_price[code] = fill["actual_price"]
                    # 成本基数按均价扣减，保持均价不变。原实现只减数量不减成本，
                    # 部分卖出后均价会被高估（信号生成器只做整仓卖出，故此前未暴露）；
                    # 模拟盘用同一套口径，这里修正以保证两个实现真正一致。
                    holdings[code]["total_cost"] -= avg_entry * sell_quantity
                    holdings[code]["quantity"] -= sell_quantity
                    if holdings[code]["quantity"] <= 0:
                        del holdings[code]

                    trade_details.append({
                        "date": trade_date,
                        "code": code,
                        "direction": "卖出",
                        "status": "filled",
                        "signal_price": signal_price,
                        "actual_price": fill["actual_price"],
                        "quantity": sell_quantity,
                        "revenue": fill["revenue"],
                        "commission": fill["commission"],
                        "stamp_tax": fill["stamp_tax"],
                        "pnl": round(pnl, 2),
                        "cash_after": round(cash, 2),
                    })

            # ── 逐日 mark-to-market（v3 / P0-1）──
            # 当天结算后记录净值：**所有**持仓都按当日（或最近可得）收盘价估值。
            # v2（0.1.4）：原实现用循环外残留的 signal_price 作为估值兜底价，
            # 而该变量属于"当天最后一笔交易的标的"，会用别的股票的价格给本股票估值。
            position_value = 0.0
            for code, h in holdings.items():
                if h["quantity"] <= 0:
                    continue
                price = self._price_on_or_before(code, trade_date)
                if price is None or price <= 0:
                    # 当日无行情（停牌/未上市/取数缺失）→ 顺延最近一次有效价
                    price = last_price.get(code)
                if price is None or price <= 0:
                    # 完全无数据：退到成本价兜底，但**必须告警**。
                    # 静默用成本价会虚增净值、把浮亏整个抹掉（这正是 P1-6 的形态）。
                    price = (h["total_cost"] / h["quantity"]) if h["quantity"] else 0.0
                    if code not in no_data_warned:
                        _logger.warning(
                            "[回测] %s 至 %s 仍无可用行情，估值退化为成本价 %.2f"
                            "（回撤/夏普会失真）", code, trade_date, price)
                        no_data_warned.add(code)
                else:
                    last_price[code] = price
                position_value += price * h["quantity"]

            total_value = cash + position_value
            nav_curve.append({
                "date": trade_date,
                "total_value": round(total_value, 2),
                "cash": round(cash, 2),
                "position_value": round(position_value, 2),
                "return_pct": round((total_value / initial_cash - 1) * 100, 2),
            })

        # ── 计算指标 ──
        result = self._compute_metrics(nav_curve, trade_details, initial_cash)
        result.rejected_orders = rejected_orders
        return result

    def run_from_db(self, start_date: str = "", end_date: str = "") -> BacktestResult:
        """
        从数据库加载交易记录执行回测。

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            BacktestResult
        """
        init_db()
        conn = get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM trade_logs"
        params = []
        conditions = []
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY date ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        trades = []
        for row in rows:
            trades.append({
                "date": row["date"][:10] if row["date"] else "",
                "code": row["code"],
                "direction": row["direction"],
                "price": row["price"],
                "quantity": row["volume"],
                "reason": row["reason"],
            })

        return self.run(trades, start_date=start_date, end_date=end_date)

    # ════════════════════════════════════════════
    # 指标计算
    # ════════════════════════════════════════════

    def _compute_metrics(
        self,
        nav_curve: list[dict],
        trade_details: list[dict],
        initial_cash: float,
    ) -> BacktestResult:
        """计算所有回测指标"""
        if not nav_curve:
            return BacktestResult()

        result = BacktestResult()

        # ── 费用拖累（2026-10-29，模块 C）──────────────────────────
        # Σ(佣金 + 印花税) / 初始资金 ×100。被拒订单没有费用（没成交），
        # 字段缺失按 0 取（.get 兜底）；口径是**累计占初始资金**的百分点。
        _total_cost = sum(
            float(t.get("commission") or 0) + float(t.get("stamp_tax") or 0)
            for t in trade_details)
        result.cost_drag = round(_total_cost / initial_cash * 100, 4) if initial_cash > 0 else 0.0

        # ── 净值曲线 ──
        values = [n["total_value"] for n in nav_curve]
        dates_str = [n["date"] for n in nav_curve]

        # 总收益率
        final_value = values[-1]
        result.total_return = round((final_value / initial_cash - 1) * 100, 2)

        # 年化收益率（v2 0.1.4）：样本不足半年时不再外推
        # 原实现 years = max(days/365, 0.1)：10 个交易日 +5% 会被年化成 >200%。
        if len(dates_str) >= 2:
            end_dt = pd.to_datetime(dates_str[-1])
            start_dt = pd.to_datetime(dates_str[0])
            days = (end_dt - start_dt).days
            years = days / 365.0
            if years >= 0.5:
                result.annual_return = round(((final_value / initial_cash) ** (1 / years) - 1) * 100, 2)
            else:
                result.annual_return = result.total_return
                _logger.warning("[回测] 样本区间仅 %d 天（<半年），不做年化外推，annual_return 取区间收益", days)
            result.turnover_rate = round(len(trade_details) * 100 / max(years, 0.1), 1)
        else:
            result.annual_return = result.total_return
            result.turnover_rate = 0

        # ── 夏普比率（v2 0.1.4）：按交易日展开，且不再删除零收益日 ──
        # 原实现把净值 reindex 到自然日再 ffill，随后 `daily_returns != 0` 把填充出的
        # 零收益日连同真实的横盘日一起删掉：分子被抬高、样本数被缩小，
        # 夏普比率被系统性高估。
        # v3（P2-5）：展开用的日历从 `freq="B"`（自然工作日）换成**真实 A 股交易日历**。
        # A 股节假日休市，用自然工作日展开会把这些天 ffill 成"零收益日"，
        # 稀释波动率、抬高夏普。
        # v3（P0-1）：净值现在已逐日估值，因此这里不再有"持仓期波动未标记"的问题。
        if len(values) >= 3:
            dt_index = pd.to_datetime(dates_str)
            full_dates = self._sharpe_calendar(dates_str)
            nav_series = pd.Series(values, index=dt_index).reindex(full_dates).ffill()
            daily_returns = nav_series.pct_change().dropna()
            if len(daily_returns) >= 2 and daily_returns.std() > 0:
                rf_daily = 0.02 / 252  # 无风险利率 2%
                excess = daily_returns.mean() - rf_daily
                result.sharpe_ratio = round(excess / daily_returns.std() * np.sqrt(252), 2)
            else:
                result.sharpe_ratio = 0
        else:
            result.sharpe_ratio = 0

        # 最大回撤
        peak = initial_cash
        max_dd = 0.0
        for v in values:
            peak = max(peak, v)
            dd = (v - peak) / peak
            max_dd = min(max_dd, dd)
        result.max_drawdown = round(max_dd * 100, 2)

        # 卡玛比率（v2 0.1.4：max_drawdown 是百分数口径，原阈值 0.01 几乎恒真）
        if abs(result.max_drawdown) > 0.5:
            result.calmar_ratio = round(result.annual_return / abs(result.max_drawdown), 2)

        # 胜率 & 盈亏比
        sell_trades = [t for t in trade_details if t.get("direction") == "卖出" and "pnl" in t]
        if sell_trades:
            wins = [t for t in sell_trades if t["pnl"] > 0]
            losses = [t for t in sell_trades if t["pnl"] <= 0]
            result.win_rate = round(len(wins) / len(sell_trades) * 100, 1)
            avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
            avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 1
            result.profit_loss_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0
            result.total_trades = len(sell_trades)

        # 信息比率（P2-3 修复）：基于基准净值计算
        if hasattr(self, '_benchmark_nav') and self._benchmark_nav is not None:
            result.information_ratio = self._calc_information_ratio(values, dates_str)
            result.benchmark_nav = self._benchmark_nav
        else:
            result.information_ratio = 0

        # 平均仓位（P1-5）：MAX_STOCKS(5) × POSITION_BUDGET_PCT(10%) = **50% 仓位上限**，
        # 约一半现金长期闲置。与基准（满仓的指数）对比时超额收益与信息比率会被压低——
        # 这不是 bug，但口径必须显式给出，否则会被误读成"策略跑输基准"。
        ratios = [n.get("position_value", 0) / n["total_value"]
                  for n in nav_curve if n.get("total_value")]
        if ratios:
            result.avg_position_pct = round(sum(ratios) / len(ratios) * 100, 1)

        # ── 存储 ──
        result.nav_curve = nav_curve
        result.trade_history = trade_details

        # ── 逐年/逐月收益 ──
        if dates_str:
            years_set = set(d[:4] for d in dates_str)
            for y in sorted(years_set):
                year_values = [values[i] for i, d in enumerate(dates_str) if d.startswith(y)]
                if year_values and len(year_values) > 1:
                    start_v = year_values[0]
                    end_v = year_values[-1]
                    if start_v > 0:
                        result.yearly_returns[y] = round((end_v / start_v - 1) * 100, 2)

            months_set = set(d[:7] for d in dates_str)
            for m in sorted(months_set):
                month_values = [values[i] for i, d in enumerate(dates_str) if d.startswith(m)]
                if month_values and len(month_values) > 1:
                    start_v = month_values[0]
                    end_v = month_values[-1]
                    if start_v > 0:
                        result.monthly_returns[m] = round((end_v / start_v - 1) * 100, 2)

            # 月度胜率
            if result.monthly_returns:
                positive_months = sum(1 for v in result.monthly_returns.values() if v > 0)
                result.monthly_win_rate = round(positive_months / len(result.monthly_returns) * 100, 1)

        return result

    def _calc_information_ratio(self, portfolio_values: list, dates_str: list) -> float:
        """
        计算信息比率 (Information Ratio)。

        信息比率 = 年化超额收益 / 年化跟踪误差
        跟踪误差 = 超额收益的标准差
        """
        if not hasattr(self, '_benchmark_nav') or self._benchmark_nav is None:
            return 0

        try:
            bench_values = [b.get("total_value", 0) for b in self._benchmark_nav]
            bench_dates = [b.get("date", "") for b in self._benchmark_nav]

            # 日期对齐
            benchmark_series = pd.Series(bench_values, index=pd.to_datetime(bench_dates))
            portfolio_series = pd.Series(portfolio_values, index=pd.to_datetime(dates_str))

            # 只取两者都有数据的日期
            common_dates = portfolio_series.index.intersection(benchmark_series.index)
            if len(common_dates) < 10:
                return 0

            portfolio_aligned = portfolio_series.loc[common_dates]
            benchmark_aligned = benchmark_series.loc[common_dates]

            # 计算累计收益率
            port_returns = portfolio_aligned.pct_change().dropna()
            bench_returns = benchmark_aligned.pct_change().dropna()

            # 超额收益
            excess_returns = port_returns - bench_returns
            if excess_returns.std() > 0 and len(excess_returns) >= 10:
                ann_excess = excess_returns.mean() * 252
                ann_te = excess_returns.std() * np.sqrt(252)
                return round(ann_excess / ann_te, 2) if ann_te > 0 else 0
        except Exception:
            pass
        return 0

    # ════════════════════════════════════════════
    # 数据加载
    # ════════════════════════════════════════════

    def set_benchmark(self, benchmark_nav: list[dict]):
        """设置基准净值曲线（用于信息比率计算）"""
        self._benchmark_nav = benchmark_nav

    def _load_from_db(self) -> list[dict]:
        """从 SQLite 数据库加载交易记录"""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trade_logs ORDER BY date ASC")
            rows = cursor.fetchall()
            conn.close()

            trades = []
            for row in rows:
                trades.append({
                    "date": row["date"][:10] if row["date"] else "",
                    "code": row["code"],
                    "direction": row["direction"],
                    "price": row["price"],
                    "quantity": row["volume"],
                    "reason": row["reason"],
                })
            return trades
        except Exception:
            return []

    def _get_price_on_date(self, code: str, trade_date: str, fallback: float) -> float:
        """获取指定日期的价格（兼容大写/小写/中文列名）。

        语义：缓存命中 → 返回 ≤trade_date 的最近一个收盘价；缓存缺失时按需加载一次；
        完全取不到才返回 `fallback`。**不抛异常**——它挂在回测主循环里。
        """
        price = self._price_on_or_before(code, trade_date)
        if price is not None and price > 0:
            return price
        return fallback

    # ════════════════════════════════════════════
    # v3（P0-1）：交易日历与逐日估值
    # ════════════════════════════════════════════

    def _build_calendar(self, trades: list,
                        start_date: str = "", end_date: str = "") -> list[str]:
        """回测区间的**完整交易日序列**（P0-1 的地基）。

        原实现只遍历"有交易发生的日期"，净值曲线上持仓期的浮亏完全没有被标记。
        这里改为遍历区间内每一个交易日：
          1. 区间取 `[start_date or 最早成交日, end_date or 最晚成交日]`；
          2. 优先用项目自带的 A 股交易日历（`data/trading_calendar.json`，
             1990~2026、完全离线、含法定节假日），不可用时降级为自然工作日并告警；
          3. 交易记录自身的日期一定并入——历史记录里可能存在非交易日数据，
             丢掉它们会让当天的成交凭空消失。
        """
        days = sorted({t.get("date", "") for t in trades if t.get("date")})
        if not days and not (start_date and end_date):
            return []
        start = start_date or days[0]
        end = end_date or days[-1]
        if start > end:
            start, end = end, start
        cal: list[str] = []
        try:
            from data import trading_calendar as tc
            cal = [str(d) for d in tc.trading_days(start, end)]
        except Exception as e:                          # noqa: BLE001 - 日历不可用只降级
            _logger.warning("[回测] 交易日历不可用(%s)，降级为自然工作日", e)
        if not cal:
            cal = [d.strftime("%Y-%m-%d")
                   for d in pd.date_range(start, end, freq="B")]
        return sorted(set(cal) | set(days))

    @staticmethod
    def _sharpe_calendar(dates_str: list[str]) -> "pd.DatetimeIndex":
        """夏普展开用的日频索引（P2-5）：真实 A 股交易日历 ∪ 净值自身日期。

        并集是必须的：净值序列若含非交易日（测试数据或历史脏数据），
        只按日历 reindex 会把这些点丢掉、凭空改变收益序列。
        """
        dts = pd.to_datetime(list(dates_str))
        cal = pd.DatetimeIndex([])
        try:
            from data import trading_calendar as tc
            days = tc.trading_days(str(dts[0].date()), str(dts[-1].date()))
            if days:
                cal = pd.DatetimeIndex(pd.to_datetime([str(d) for d in days]))
        except Exception as e:                          # noqa: BLE001 - 只降级
            _logger.debug("[回测] 夏普日历不可用(%s)，降级为自然工作日", e)
        if len(cal) == 0:
            cal = pd.date_range(start=dts[0], end=dts[-1], freq="B")
        return cal.union(dts)

    def _price_series(self, code: str) -> Optional[tuple]:
        """返回 (日期数组, 收盘价数组)，按日期升序；无数据返回 None。

        结果按 code 记忆化：逐日估值会反复取同一只票的价格，每次重建 DataFrame
        副本（原 `_get_price_on_date` 的做法）在 120 只票 × 数百个交易日下会成为瓶颈。
        """
        if code in self._price_series_cache:
            return self._price_series_cache[code]
        df = self._kline_cache.get(code)
        if df is None or getattr(df, "empty", True):
            series = None
        else:
            try:
                date_col = next((c for c in ("Date", "date", "日期") if c in df.columns), None)
                price_col = next((c for c in ("Close", "close", "收盘") if c in df.columns), None)
                if not date_col or not price_col:
                    series = None
                else:
                    dates = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d").values
                    closes = pd.to_numeric(df[price_col], errors="coerce").values
                    order = np.argsort(dates)
                    series = (dates[order], closes[order])
            except Exception as e:                      # noqa: BLE001
                _logger.debug("[回测] %s K线归一化失败: %s", code, e)
                series = None
        self._price_series_cache[code] = series
        return series

    def _price_on_or_before(self, code: str, date_str: str) -> Optional[float]:
        """返回 ≤date_str 的最近一个有效收盘价；无数据返回 None。

        与 `_get_price_on_date` 的区别：这里**只在必要时加载一次 K 线**，
        且不做"退到成本价"——成本价兜底由调用方决定并负责告警（P1-6）。
        """
        if code not in self._kline_cache:
            self._ensure_kline(code, date_str)
        series = self._price_series(code)
        if not series:
            return None
        dates, closes = series
        idx = int(np.searchsorted(dates, date_str, side="right")) - 1
        while idx >= 0:
            value = closes[idx]
            if value is not None and not pd.isna(value) and float(value) > 0:
                return float(value)
            idx -= 1
        return None

    def _ensure_kline(self, code: str, ref_date: str = ""):
        """按需加载一次 K 线（每个 code 只尝试一次）。

        v3（P1-6）：原实现硬编码 `period="3mo"`，而 `StockDataFetcher` 的 period
        终点固定在**今天**——回测 2024 年时取到的是"今天往前 90 天"，
        历史日期一条都命不中，于是全部落回成本价（净值虚高、浮亏被抹掉）。
        现在按回测起点动态取跨度。
        """
        if code in self._kline_cache or code in self._kline_failed:
            return
        df = None
        # ⭐ P0-1（2026-10-18）：**本地优先** —— 全历史一次读入仅 5~20 ms，
        # 且天然覆盖任意 ref_date（原来靠 `_period_covering` 猜跨度，猜短了
        # 就会"历史日期一条都命不中"）。本地没有才联网。
        try:
            from data import bars as _bars
            df = _bars.get_bars(code)                 # 全历史（升序，qfq）
        except Exception as e:                          # noqa: BLE001
            _logger.debug("[回测] %s 本地K线不可用: %s", code, e)
        if df is None:
            try:
                from data import bars as _bars2
                _offline = _bars2.offline_mode()
            except Exception as e:                        # noqa: BLE001
                _logger.debug("[回测] offline_mode 读取失败，按在线处理: %s", e)
                _offline = False
            if not _offline:
                try:
                    df = self.stock_fetcher.get_stock_data(code, period=_period_covering(ref_date))
                except Exception as e:                      # noqa: BLE001
                    _logger.warning("[回测] %s K线加载失败: %s", code, e)
        if isinstance(df, pd.DataFrame) and not df.empty:
            self._kline_cache[code] = df
        else:
            self._kline_failed.add(code)
            _logger.warning("[回测] %s 无可用K线（估值将退化为成本价，回撤会失真）", code)

    # ════════════════════════════════════════════
    # 报告输出
    # ════════════════════════════════════════════

    def save_report(self, result: BacktestResult, filepath: str = "") -> str:
        """保存回测报告为 JSON"""
        if not filepath:
            filepath = os.path.join(
                self.records_dir,
                f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        try:
            from db import store as db_store
            db_store.insert_run("backtest", {
                "file": os.path.basename(filepath),
                "total_return": result.total_return, "sharpe": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown, "trades": result.total_trades,
                "information_ratio": result.information_ratio,
            })
        except Exception:
            pass  # 元数据落库失败不影响主流程
        return filepath

    def format_report(self, result: BacktestResult) -> str:
        """生成格式化报告文本"""
        lines = [
            "=" * 50,
            "  回测报告",
            "=" * 50,
            f"  总收益率:     {result.total_return:>8.2f}%",
            f"  年化收益率:   {result.annual_return:>8.2f}%",
            f"  夏普比率:     {result.sharpe_ratio:>8.2f}",
            f"  卡玛比率:     {result.calmar_ratio:>8.2f}",
            f"  最大回撤:     {result.max_drawdown:>8.2f}%",
            f"  胜率:         {result.win_rate:>8.1f}%",
            f"  盈亏比:       {result.profit_loss_ratio:>8.2f}",
            f"  总交易次数:   {result.total_trades:>8}",
            f"  月度胜率:     {result.monthly_win_rate:>8.1f}%",
            f"  信息比率:     {result.information_ratio:>8.2f}",
            f"  平均仓位:     {result.avg_position_pct:>8.1f}%",
            f"  被拒订单:     {result.rejected_orders:>8}",
            "-" * 50,
        ]
        if result.yearly_returns:
            lines.append("  逐年收益:")
            for year, ret in result.yearly_returns.items():
                lines.append(f"    {year}: {ret:+.2f}%")
        return "\n".join(lines)


# 全局单例
_backtest_engine: Optional[BacktestEngine] = None


def get_backtest_engine() -> BacktestEngine:
    """获取回测引擎单例"""
    global _backtest_engine
    if _backtest_engine is None:
        _backtest_engine = BacktestEngine()
    return _backtest_engine