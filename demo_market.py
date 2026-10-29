"""市场检测模块独立验证"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.market_detector import MarketDetector


def main():
    print(f"\n{'='*60}")
    print(f"  📊 市场环境检测器 - 独立验证")
    print(f"  14个指标 / 4维度 / 5级市场状态")
    print(f"{'='*60}\n")

    detector = MarketDetector()
    print("⏳ 正在检测市场环境（14个指标并发获取）...\n")

    signal = detector.detect()

    # === 汇总 ===
    print(f"📊 市场状态: {signal.market_state}")
    print(f"📈 总分: {signal.total_score}/100")
    print(f"🎯 信号: {signal.signal}")
    print(f"💰 仓位上限: {signal.position_limit*100:.0f}%")
    print(f"🎯 策略偏向: {signal.strategy_bias}")
    print(f"💡 建议: {signal.operation_advice}")
    print(f"⏰ 检测时间: {signal.timestamp}")

    # === 各维度得分 ===
    print(f"\n{'─'*60}")
    print("维度得分明细:")
    dimensions = {"趋势": 0, "量能": 0, "情绪": 0, "结构": 0}
    dim_map = {
        "hs300_ma20_deviation": "趋势", "sh_macd": "趋势", "cyb_relative_strength": "趋势",
        "volume_vs_ma20": "量能", "advance_decline_ratio": "量能", "north_flow": "量能",
        "limit_up_down_ratio": "情绪", "max_consecutive_board": "情绪",
        "market_breadth": "情绪", "ivix_level": "情绪",
        "sector_rotation_speed": "结构", "size_style_bias": "结构",
        "margin_balance_change": "结构", "limit_up_upgrade_rate": "结构",
    }
    for key, detail in signal.scores.items():
        dim = dim_map.get(key, "其他")
        dimensions[dim] += detail.get("weighted_score", 0)
        if detail.get("detail"):
            print(f"  [{dim}] {detail['detail']}")

    print(f"\n{'─'*60}")
    for dim, score in dimensions.items():
        max_s = {"趋势": 30, "量能": 25, "情绪": 25, "结构": 20}[dim]
        print(f"  {dim}: {score:.1f}/{max_s}")

    # === 风险提示 ===
    if signal.risk_warnings:
        print(f"\n⚠️ 风险提示:")
        for w in signal.risk_warnings[:3]:
            print(f"  - {w}")

    print(f"\n{'='*60}")
    print(f"  ✅ 市场检测完成！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()