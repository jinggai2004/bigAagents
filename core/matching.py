"""撮合与交易成本（Phase 2）

从 `core/backtest.py` 抽出，供**回测**与**模拟盘**共用。计划里定死了一条硬约束：
不允许两套撮合——两套规则必然分叉，而"模拟盘与同区间回测在相同信号下结果一致"
正是 Phase 2 的验收条件之一。这个模块就是那个唯一实现。

A 股口径约定：
  - 买入：成交价 = 信号价 × (1 + 滑点)；支出 = 成交价×数量 + 佣金
          （佣金率 0.025%，最低 5 元）
  - 卖出：成交价 = 信号价 × (1 - 滑点)；收入 = 成交价×数量 - 佣金 - 印花税（0.1%）
  - 印花税只在卖出端收（A 股规则）
  - 过户费未计入：历史回测实现如此，为保持口径可比而延续；若将来要加，
    只改这里一处即可让回测与模拟盘同时生效
  - 数量按 100 股整手（`round_lot`），撮合本身不校验整手，由调用方负责
"""

STAMP_TAX_RATE = 0.001      # 卖出印花税 0.1%
COMMISSION_RATE = 0.00025   # 佣金 0.025%
SLIPPAGE_RATE = 0.0005      # 滑点 0.05%
MIN_COMMISSION = 5.0        # 最低佣金 5 元
LOT_SIZE = 100              # 一手 100 股


def commission_of(amount: float) -> float:
    """佣金：按成交金额计提，最低 5 元"""
    return max(MIN_COMMISSION, amount * COMMISSION_RATE)


def round_lot(quantity: int, lot: int = LOT_SIZE) -> int:
    """按整手向下取整（不足一手返回 0）"""
    if quantity is None or quantity <= 0:
        return 0
    return int(quantity // lot * lot)


def buy_fill(signal_price: float, quantity: int) -> dict:
    """买入成交明细。不判断现金是否够——由调用方用 can_afford 决定是否成交。"""
    quantity = int(quantity)
    actual_price = signal_price * (1 + SLIPPAGE_RATE)
    cost = actual_price * quantity
    commission = commission_of(cost)
    total = cost + commission
    return {
        "direction": "买入",
        "signal_price": signal_price,
        "actual_price": round(actual_price, 2),
        "quantity": quantity,
        "cost": round(cost, 2),
        "commission": round(commission, 2),
        "stamp_tax": 0.0,
        "total": round(total, 2),
        "cash_delta": -round(total, 2),
    }


def sell_fill(signal_price: float, quantity: int) -> dict:
    """卖出成交明细（含印花税）"""
    quantity = int(quantity)
    actual_price = signal_price * (1 - SLIPPAGE_RATE)
    revenue = actual_price * quantity
    commission = commission_of(revenue)
    stamp_tax = revenue * STAMP_TAX_RATE
    net = revenue - commission - stamp_tax
    return {
        "direction": "卖出",
        "signal_price": signal_price,
        "actual_price": round(actual_price, 2),
        "quantity": quantity,
        "revenue": round(revenue, 2),
        "commission": round(commission, 2),
        "stamp_tax": round(stamp_tax, 2),
        "total": round(net, 2),
        "cash_delta": round(net, 2),
    }


def pnl_of(fill: dict, avg_cost: float) -> float:
    """卖出盈亏 =（成交价 - 持仓均价）× 数量 - 佣金 - 印花税"""
    return round((fill["actual_price"] - avg_cost) * fill["quantity"]
                 - fill["commission"] - fill["stamp_tax"], 2)


def can_afford(cash: float, fill: dict) -> bool:
    """现金是否足以支付这笔买入"""
    return cash >= fill["total"]


# ----------------------------------------------------------------
# 涨跌停口径（Phase 2：模拟盘与回测信号生成器共用同一套判断）
# ----------------------------------------------------------------

LIMIT_EPS = 1e-6


def limit_pct(code: str) -> float:
    """涨跌停幅度：创业板/科创板 20%，其余主板 10%。

    ST 的 5% 未纳入——数据里没有 ST 标记，宁可少判也不要错判。
    """
    return 0.20 if str(code).startswith(("300", "301", "688")) else 0.10


def blocked_by_price_limit(direction: str, code: str,
                           exec_price, prev_close) -> str:
    """开盘即涨停买不到 / 开盘即跌停卖不掉。可交易返回空串，否则返回原因。

    买 卖 双 向 都 用 同 一 个 判 定 —— 回测信号生成器与模拟盘撮合都调它，
    避免两处各写一份阈值后悄悄分叉。
    """
    prev = float(prev_close or 0.0)
    price = float(exec_price or 0.0)
    if prev <= 0 or price <= 0:
        return ""
    lim = limit_pct(code)
    if direction == "买入" and price >= prev * (1 + lim) - LIMIT_EPS:
        return f"开盘涨停({price:.2f})不可买"
    if direction == "卖出" and price <= prev * (1 - lim) + LIMIT_EPS:
        return f"开盘跌停({price:.2f})不可卖"
    return ""
