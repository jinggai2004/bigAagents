#!/usr/bin/env python3
"""
回测冒烟测试 (Backtest Smoke Test)

插入 5 笔假交易记录 → 运行回测 → 验证指标合理性。
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.models import init_db, get_connection, drop_all_tables
from core.backtest import BacktestEngine, BacktestResult

# 1. 初始化空数据库
drop_all_tables()
init_db()

# 2. 插入测试数据
conn = get_connection()
cursor = conn.cursor()

test_trades = [
    ("2025-01-10", "000001", "买入", 12.50, 1000, "测试开仓", "选股信号", None),
    ("2025-03-15", "000001", "卖出", 13.80, 1000, "测试平仓", "止盈", 1300.0),
    ("2025-04-01", "600519", "买入", 1300.00, 100, "测试开仓", "选股信号", None),
    ("2025-06-15", "600519", "卖出", 1280.00, 100, "测试平仓", "止损", -2000.0),
    ("2025-07-01", "300750", "买入", 200.00, 500, "测试开仓", "选股信号", None),
]

for trade in test_trades:
    cursor.execute(
        """INSERT INTO trade_logs (date, code, direction, price, volume, reason, signal_source, pnl)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        trade,
    )
conn.commit()
conn.close()

# 3. 运行回测
engine = BacktestEngine()
result = engine.run_from_db()

# 4. 验证
print("=" * 50)
print("  回测冒烟测试结果")
print("=" * 50)
print(f"  总收益率:     {result.total_return:.2f}%")
print(f"  年化收益率:   {result.annual_return:.2f}%")
print(f"  夏普比率:     {result.sharpe_ratio:.2f}")
print(f"  卡玛比率:     {result.calmar_ratio:.2f}")
print(f"  最大回撤:     {result.max_drawdown:.2f}%")
print(f"  胜率:         {result.win_rate:.1f}%")
print(f"  盈亏比:       {result.profit_loss_ratio:.2f}")
print(f"  总交易次数:   {result.total_trades}")
print(f"  信息比率:     {result.information_ratio:.2f}")
print(f"  月度胜率:     {result.monthly_win_rate:.1f}%")
print(f"  净值曲线点数: {len(result.nav_curve)}")
print(f"  交易明细数:   {len(result.trade_history)}")

# 验证逻辑
errors = []

# 1. 净值不应为负
if result.nav_curve:
    min_value = min(n["total_value"] for n in result.nav_curve)
    if min_value < 0:
        errors.append(f"净值出现负值: {min_value}")
    else:
        print(f"  ✅ 净值最小值: {min_value:.2f} (> 0)")

# 2. 净值曲线非空
if len(result.nav_curve) == 0:
    errors.append("净值曲线为空")
else:
    print(f"  ✅ 净值曲线有 {len(result.nav_curve)} 个数据点")

# 3. 夏普比率数量级合理 (|sharpe| < 10)
if abs(result.sharpe_ratio) > 10:
    errors.append(f"夏普比率超出合理范围: {result.sharpe_ratio}")
else:
    print(f"  ✅ 夏普比率 {result.sharpe_ratio:.2f} 在合理范围")

# 4. 最大回撤绝对值合理
if abs(result.max_drawdown) > 100:
    errors.append(f"最大回撤超出范围: {result.max_drawdown}%")
else:
    print(f"  ✅ 最大回撤 {result.max_drawdown:.2f}% 在合理范围")

# 5. 胜率在 0-100 之间
if result.total_trades > 0 and not (0 <= result.win_rate <= 100):
    errors.append(f"胜率不在0-100范围: {result.win_rate}")
elif result.total_trades > 0:
    print(f"  ✅ 胜率 {result.win_rate:.1f}% 在合理范围")

# 汇总
print()
if errors:
    print(f"❌ 冒烟测试失败: {len(errors)} 个错误")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)
else:
    print("🎉 冒烟测试通过！回测引擎逻辑正常。")

# 清理测试数据
drop_all_tables()