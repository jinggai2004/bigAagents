"""模拟交易账户（Phase 2）

与回测共用 `core/matching` 的撮合、费用与涨跌停口径（硬约束：不允许两套撮合）。
本模块只维护账户状态：现金、持仓（含 T+1 可卖数量）、挂单、成交、净值。

两种用法：
  1) **回放**：`apply_trades(带价格的成交指令)` —— 与 BacktestEngine 的结果可逐项对齐，
     一致性测试靠它证明"模拟盘与回测同规则"；
  2) **挂单撮合**：`submit_order(...)` 下单 → `settle_open(date, bars)` 用当日开盘价成交。
     买单在开盘涨停/停牌时取消；卖单顺延到下一个可成交日（最多 MAX_DEFER_DAYS 个交易日）。
     **限价单**（`limit_price > 0`）：买入遇开盘价高于限价、卖出遇开盘价低于限价 ⇒
     撤单且**不顺延**（盘中理由隔夜已失效，顺延等于用过期理由追价）。
     不传 `limit_price` 即市价单，行为与旧版完全一致。

账户不接券商、不产生任何真实委托（项目原则）。
"""

import logging
from dataclasses import asdict, dataclass
from datetime import datetime

from core import matching

_logger = logging.getLogger(__name__)

DEFAULT_INITIAL_CASH = 1_000_000.0
MAX_DEFER_DAYS = 10          # 卖单最多顺延的交易日数，超过即取消


@dataclass
class Position:
    code: str
    quantity: int = 0            # 持仓总量
    available: int = 0           # 可卖数量（T+1：当日买入次日才可卖）
    total_cost: float = 0.0      # 含费用的累计成本
    last_buy_date: str = ""      # 最近一次买入日；T+1 解冻按它推导（可落库、可跨进程恢复）

    @property
    def avg_cost(self) -> float:
        return (self.total_cost / self.quantity) if self.quantity else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["avg_cost"] = round(self.avg_cost, 4)
        return data


@dataclass
class Order:
    code: str
    direction: str
    quantity: int
    submit_date: str
    order_id: int = 0
    status: str = "pending"        # pending | filled | cancelled
    reason: str = ""
    filled_quantity: int = 0
    filled_price: float = 0.0
    filled_date: str = ""
    # 限价（2026-10-27）：0 表示市价单——老订单/回放路径都是 0，行为与改动前一致。
    # 只加"最高买入价/最低卖出价"这一种语义，不引入 A 股的五种市价申报类型：
    # 模拟盘需要的只是"开盘价不满足就不成交"，多余的申报类型没有对应数据可撮合。
    limit_price: float = 0.0
    decision_price: float = 0.0    # 决策时看到的价，滑点的比较基准
    slippage: float = 0.0          # filled_price - decision_price

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def has_limit(self) -> bool:
        """是否限价单。**唯一判据只在这里**，别在下游写 `> 0` 或 `is not None`。"""
        return bool(self.limit_price and self.limit_price > 0)


class PaperAccount:
    """虚拟资金账户"""

    def __init__(self, initial_cash: float = DEFAULT_INITIAL_CASH, name: str = "default"):
        self.name = name
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.fills: list[dict] = []
        self.nav: list[dict] = []
        self._order_seq = 0
        self._defer_days: dict[int, int] = {}        # order_id -> 已顺延交易日数

    # ================= 查询 =================

    def position(self, code: str) -> Position:
        return self.positions.get(code) or Position(code=code)

    def holdings_value(self, prices: dict) -> float:
        """持仓市值。缺价的标的退回收盘均价（宁可保守，不用 0 抹掉持仓）"""
        total = 0.0
        for code, pos in self.positions.items():
            price = (prices or {}).get(code)
            if not isinstance(price, (int, float)) or price <= 0:
                price = pos.avg_cost
            total += price * pos.quantity
        return total

    def total_value(self, prices: dict) -> float:
        return self.cash + self.holdings_value(prices)

    def to_dict(self, prices: dict = None) -> dict:
        return {
            "name": self.name,
            "initial_cash": self.initial_cash,
            "cash": round(self.cash, 2),
            "positions": {c: p.to_dict() for c, p in self.positions.items()},
            "total_value": round(self.total_value(prices or {}), 2),
            "pending_orders": len([o for o in self.orders if o.status == "pending"]),
            "fills": len(self.fills),
        }

    # ================= 内部 =================

    def _unlock_t1(self, date: str):
        """把"买入日早于当前日期"的持仓解冻为可卖（T+1）。

        用 `last_buy_date` 推导而不是维护一个内存队列：队列落库后会丢，
        重启后当日买入的持仓将永远解不了冻。判据是"最近一次买入日在今天之前"，
        当日新买入的数量不会因此被算作可卖。
        """
        for pos in self.positions.values():
            if pos.last_buy_date and pos.last_buy_date < date:
                pos.available = pos.quantity

    def unlock_t1(self, date: str = None) -> int:
        """**对外**的 T+1 解冻入口：把"买入日早于 `date`"的持仓置为可卖。返回被解冻的只数。

        ⚠️ 为什么必须显式对外暴露（2026-10-10 用户实际踩到）：
        `_unlock_t1` 原先**只**在 `apply_trade` / `settle_open` 里被调，
        即"**只有发生成交时才解冻**"。后果：
          - 用户周五买入 → 周六/周日（非交易日）无任何成交调用 → 仓位一直是
            `available=0`。用户点"卖出"看到的是 **"T+1 未解锁或无可用持仓"**，
            但真实原因只是"还没有任何一次调用帮你解冻"，与 T+1 规则本身无关；
          - **光刷新页面不解冻**（页面只做只读渲染），必须等下一次成交。
        所以把解冻做成可独立调用的动作，由**读取方**（页面 / 工具 / 调度）在渲染前主动调，
        而不是寄生在写入路径上。

        幂等：解冻只把 `available` 抬到 `quantity`，重复调用无副作用
        （已解冻的仓位 `last_buy_date < date` 仍成立，赋值结果相同）。
        不会影响"当日买入"的仓位——它们 `last_buy_date == date`，不满足 `<`。

        Args:
            date: 判定基准日（YYYY-MM-DD）。默认取**今天**。
                测试可显式传入以构造跨日场景。
        Returns:
            本次实际发生解冻的持仓只数（已可卖的不会重复计数）。
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        date = str(date)
        unlocked = 0
        for pos in self.positions.values():
            # 判据与 `_unlock_t1` 完全一致（同一条件，避免两处各写一份后分叉）
            if pos.last_buy_date and pos.last_buy_date < date and pos.available < pos.quantity:
                pos.available = pos.quantity
                unlocked += 1
        if unlocked:
            _logger.info("[模拟盘] T+1 解冻 %d 只持仓（基准日 %s）", unlocked, date)
        return unlocked

    def _apply_fill(self, date: str, code: str, fill: dict, avg_cost: float = None,
                    tag: str = None, slippage: float = None) -> dict:
        self.cash += fill["cash_delta"]
        record = {"date": date, "code": code, **fill}
        # ⚠️ `tag` 是**成交来源标记**（如 "intraday_buy"）：盘后 T+1 成交与盘中补仓
        #    必须可区分，否则"盘中加仓胜率"这类统计会把两类成交混在一起。
        #    它随 `payload` 一起落库，无需改表。
        if tag:
            record["tag"] = str(tag)
        # 滑点同时写进成交记录：`paper_orders` 会被下一次下单覆盖，
        # 而成交是**不可变的历史**——复盘算"挂单阈值准不准"必须从 fills 取。
        # ⚠️ 这里记的是**决策滑点**（决策价 → 实际成交价的全部偏差），
        #    与 `matching.SLIPPAGE_RATE` 的"撮合冲击成本"不是同一个量。
        #    两者靠 fills 里已有的字段就能拆开，不需要再加列：
        #      撮合冲击 = actual_price − signal_price     （signal_price 就是基准价）
        #      隔夜跳空 = signal_price − decision_price
        #      总偏差   = actual_price − decision_price = slippage
        if slippage is not None:
            record["slippage"] = float(slippage)
            # 决策价 = 实际成交价 − 滑点。落进 fills 是为了让复盘能**只读 fills**
            # 就还原当时看到的价，而不必去查会被覆盖的 paper_orders。
            if isinstance(record.get("actual_price"), (int, float)):
                record["decision_price"] = round(
                    float(record["actual_price"]) - float(slippage), 4)

        if fill["direction"] == "买入":
            pos = self.positions.setdefault(code, Position(code=code))
            pos.quantity += fill["quantity"]
            pos.total_cost += fill["total"]
            pos.last_buy_date = max(pos.last_buy_date, str(date))
        else:
            pos = self.positions[code]
            basis = pos.avg_cost if avg_cost is None else avg_cost
            record["pnl"] = matching.pnl_of(fill, basis)
            pos.total_cost -= basis * fill["quantity"]
            pos.quantity -= fill["quantity"]
            pos.available = max(0, pos.available - fill["quantity"])
            if pos.quantity <= 0:
                self.positions.pop(code, None)

        record["cash_after"] = round(self.cash, 2)
        self.fills.append(record)
        return record

    @staticmethod
    def _reject(date: str, code: str, direction: str, reason: str) -> dict:
        _logger.info("[模拟盘] 拒单 %s %s %s: %s", date, direction, code, reason)
        return {"date": date, "code": code, "direction": direction,
                "status": "rejected", "reason": reason}

    # ================= 回放（与回测口径一致） =================

    def apply_trade(self, trade: dict) -> dict | None:
        """按给定价格回放一笔成交指令（买单现金不足/T+1 未解锁会被拒）

        trade 可带可选字段 `tag`（成交来源标记，如 "intraday_buy"），会写进成交记录。
        """
        date = str(trade.get("date", ""))
        code = str(trade.get("code", ""))
        direction = trade.get("direction", "")
        price = trade.get("price", 0)
        quantity = int(trade.get("quantity", 0) or 0)
        tag = trade.get("tag") or None
        if not code or quantity <= 0 or not isinstance(price, (int, float)) or price <= 0:
            return None

        self._unlock_t1(date)

        if direction == "买入":
            fill = matching.buy_fill(price, quantity)
            if not matching.can_afford(self.cash, fill):
                return self._reject(date, code, "买入",
                                    f"现金不足（需 {fill['total']:.2f}，可用 {self.cash:.2f}）")
            return self._apply_fill(date, code, fill, tag=tag)

        if direction == "卖出":
            pos = self.positions.get(code)
            if pos is None or pos.available <= 0:
                return self._reject(date, code, "卖出", "T+1 未解锁或无可用持仓")
            sell_qty = min(quantity, pos.available)
            fill = matching.sell_fill(price, sell_qty)
            return self._apply_fill(date, code, fill, avg_cost=pos.avg_cost, tag=tag)

        return None

    def apply_trades(self, trades: list) -> list[dict]:
        """按日回放，同日先卖后买（与 BacktestEngine 的处理顺序一致）"""
        results = []
        dates = sorted({str(t.get("date", "")) for t in trades})
        for date in dates:
            day = [t for t in trades if str(t.get("date", "")) == date]
            day.sort(key=lambda t: 0 if t.get("direction") == "卖出" else 1)
            for trade in day:
                out = self.apply_trade(trade)
                if out is not None:
                    results.append(out)
        return results

    # ================= 挂单撮合 =================

    def submit_order(self, code: str, direction: str, quantity: int,
                     date: str, reason: str = "",
                     limit_price: float = None, decision_price: float = None) -> Order:
        """下一笔挂单（最早在下一个交易日开盘成交）。

        Args:
            limit_price: 限价。**None/0 ⇒ 市价单**（改动前的唯一行为，老调用方不受影响）。
                买入：开盘价 **高于** 限价 ⇒ 撤单（不追价，不顺延）；
                卖出：开盘价 **低于** 限价 ⇒ 撤单（不割在更差的价）。
                ⚠️ 撤单而不是顺延：盘中"跌到 15.8 才买"的判断基于当天信息，
                隔夜后价格与理由都可能失效，顺延等于用一个过期理由追价。
            decision_price: 决策时看到的价（滑点基准）。缺省取 `limit_price`。
        """
        quantity = matching.round_lot(quantity)
        self._order_seq += 1
        lp = float(limit_price) if isinstance(limit_price, (int, float)) and limit_price > 0 else 0.0
        dp = (float(decision_price) if isinstance(decision_price, (int, float)) and decision_price > 0
              else lp)
        order = Order(code=str(code), direction=direction, quantity=quantity,
                      submit_date=str(date), order_id=self._order_seq, reason=reason,
                      limit_price=lp, decision_price=dp)
        if quantity <= 0:
            order.status, order.reason = "cancelled", "数量不足一手"
        elif direction == "买入":
            est = matching.buy_fill(0.01, quantity)["total"]  # 仅用于极小的现金下限提示
            if self.cash < est:
                order.status, order.reason = "cancelled", "现金不足"
        elif direction == "卖出":
            pos = self.positions.get(code)
            if pos is None or pos.quantity <= 0:
                order.status, order.reason = "cancelled", "无持仓"
        else:
            order.status, order.reason = "cancelled", f"未知方向: {direction}"
        self.orders.append(order)
        return order

    def _defer_or_cancel(self, order: Order, reason: str):
        days = self._defer_days.get(order.order_id, 0) + 1
        self._defer_days[order.order_id] = days
        if days > MAX_DEFER_DAYS:
            order.status = "cancelled"
            order.reason = f"{reason}，连续顺延 {days} 个交易日，已取消"
        else:
            order.reason = f"{reason}，顺延第 {days} 次"

    def settle_open(self, date: str, bars: dict) -> list[dict]:
        """用当日盘口撮合挂单。

        Args:
            date: 当前交易日
            bars: {code: {"open","close","prev_close"}}；缺该 code 视为停牌
        Returns:
            本轮处理过的订单结果列表
        """
        self._unlock_t1(date)
        results = []
        pending = [o for o in self.orders if o.status == "pending"]
        # 先卖后买：与回测引擎同日处理顺序一致，且卖出回笼的现金当日可用于买入
        pending.sort(key=lambda o: 0 if o.direction == "卖出" else 1)

        for order in pending:
            if order.submit_date >= date:
                continue                    # T 日提交，最早 T+1 成交
            bar = (bars or {}).get(order.code)
            if bar is None:
                if order.direction == "买入":
                    order.status, order.reason = "cancelled", "停牌/无行情，买单取消"
                else:
                    self._defer_or_cancel(order, "停牌/无行情")
                results.append(order.to_dict())
                continue

            exec_price = bar.get("open") or bar.get("close")
            blocked = matching.blocked_by_price_limit(
                order.direction, order.code, exec_price, bar.get("prev_close"))
            if blocked:
                if order.direction == "买入":
                    order.status, order.reason = "cancelled", blocked
                else:
                    self._defer_or_cancel(order, blocked)
                results.append(order.to_dict())
                continue

            # ── 限价闸门（2026-10-27）──
            # 放在涨跌停判断**之后**：开盘涨停时"根本买不到"是更根本的原因，
            # 用"高于限价"去解释会掩盖真正发生了什么（复盘会误判成阈值设太紧）。
            # ⚠️ `exec_price` 在缺 open 时退回收盘价，此时闸门按收盘价判——
            #    比按开盘价更严格（收盘可能更高），宁可漏成交也不追价。
            if order.has_limit:
                if order.direction == "买入" and exec_price > order.limit_price:
                    order.status = "cancelled"
                    order.reason = (f"开盘 {exec_price:.2f} 高于限价 "
                                    f"{order.limit_price:.2f}，放弃追价")
                    results.append(order.to_dict())
                    continue
                if order.direction == "卖出" and exec_price < order.limit_price:
                    order.status = "cancelled"
                    order.reason = (f"开盘 {exec_price:.2f} 低于限价 "
                                    f"{order.limit_price:.2f}，放弃成交")
                    results.append(order.to_dict())
                    continue

            if order.direction == "买入":
                fill = matching.buy_fill(exec_price, order.quantity)
                if not matching.can_afford(self.cash, fill):
                    order.status = "cancelled"
                    order.reason = f"现金不足（需 {fill['total']:.2f}，可用 {self.cash:.2f}）"
                    results.append(order.to_dict())
                    continue
            else:
                pos = self.positions.get(order.code)
                if pos is None or pos.available <= 0:
                    order.status, order.reason = "cancelled", "T+1 未解锁或无可用持仓"
                    results.append(order.to_dict())
                    continue
                fill = matching.sell_fill(exec_price, min(order.quantity, pos.available))

            self._apply_fill(date, order.code, fill,
                             avg_cost=(self.positions[order.code].avg_cost
                                       if order.code in self.positions else None),
                             slippage=(round(fill["actual_price"] - order.decision_price, 4)
                                       if order.decision_price > 0 else None))
            order.status = "filled"
            order.filled_quantity = fill["quantity"]
            order.filled_price = fill["actual_price"]
            order.filled_date = date
            # 滑点口径**只有一种**：`成交价 − 决策价`。
            #   买入为正 ⇒ 追高吃亏；卖出为负 ⇒ 割肉吃亏。
            # 不做"正数一定代表不利"的翻转：那会让页面必须知道方向才能解读，
            # 而报表里一聚合就分不清这笔是买还是卖。
            if order.decision_price > 0:
                order.slippage = round(order.filled_price - order.decision_price, 4)
            results.append(order.to_dict())

        return results

    # ================= 净值 =================

    def mark_to_market(self, date: str, prices: dict) -> dict:
        """记录一条净值（每日盈亏按累计收益率给出）"""
        value = self.total_value(prices)
        record = {
            "date": str(date),
            "cash": round(self.cash, 2),
            "position_value": round(value - self.cash, 2),
            "total_value": round(value, 2),
            "return_pct": round((value / self.initial_cash - 1) * 100, 2) if self.initial_cash else 0.0,
        }
        self.nav.append(record)
        return record

    # ================= 状态序列化（供 db.store 落库 / 跨进程恢复） =================

    def to_state(self) -> dict:
        return {
            "name": self.name,
            "initial_cash": self.initial_cash,
            "cash": round(self.cash, 2),
            "positions": {code: asdict(pos) for code, pos in self.positions.items()},
            "orders": [o.to_dict() for o in self.orders],
            "fills": [dict(f) for f in self.fills],
            "nav": [dict(n) for n in self.nav],
        }

    @classmethod
    def from_state(cls, state: dict) -> "PaperAccount":
        state = state or {}
        acct = cls(initial_cash=float(state.get("initial_cash") or DEFAULT_INITIAL_CASH),
                   name=str(state.get("name") or "default"))
        acct.cash = float(state.get("cash") or 0.0)

        for code, p in (state.get("positions") or {}).items():
            last_buy = str(p.get("last_buy_date") or "")
            quantity = int(p.get("quantity") or 0)
            acct.positions[code] = Position(
                code=code,
                quantity=quantity,
                # 缺 last_buy_date（旧数据/手工构造）时视为已解冻，
                # 否则该持仓会永远卖不出去——宁可放松 T+1 也不能把仓位锁死
                available=(int(p.get("available") or 0) if last_buy else quantity),
                total_cost=float(p.get("total_cost") or 0.0),
                last_buy_date=last_buy,
            )

        for o in (state.get("orders") or []):
            order = Order(
                code=str(o.get("code", "")), direction=str(o.get("direction", "")),
                quantity=int(o.get("quantity") or 0), submit_date=str(o.get("submit_date", "")),
                order_id=int(o.get("order_id") or 0), status=str(o.get("status") or "pending"),
                reason=str(o.get("reason") or ""),
                filled_quantity=int(o.get("filled_quantity") or 0),
                filled_price=float(o.get("filled_price") or 0.0),
                filled_date=str(o.get("filled_date") or ""),
                # 落库是 NULL（旧数据/市价单）⇒ 0.0。读不回来会让限价意图在
                # **进程重启后静默失效**：订单还在 pending，却变成"开盘即成交"。
                limit_price=float(o.get("limit_price") or 0.0),
                decision_price=float(o.get("decision_price") or 0.0),
                slippage=float(o.get("slippage") or 0.0),
            )
            acct.orders.append(order)
            acct._order_seq = max(acct._order_seq, order.order_id)

        acct.fills = [dict(f) for f in (state.get("fills") or [])]
        acct.nav = [dict(n) for n in (state.get("nav") or [])]
        return acct
