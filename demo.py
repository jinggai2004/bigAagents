"""
最小闭环验证脚本
1. 获取 A 股数据
2. 调用 DeepSeek LLM 进行技术分析
3. 打印结果
"""

import sys
import os

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.stock_data import StockDataFetcher
from llm.deepseek_client import DeepSeekClient


def main(stock_code="000001"):
    print(f"\n{'='*60}")
    print(f"  A股多Agent分析系统 - 最小闭环验证")
    print(f"{'='*60}\n")

    # 1. 获取股票数据
    print("📊 [1/3] 正在获取股票数据...")
    fetcher = StockDataFetcher()

    stock_info = fetcher.get_stock_info(stock_code)
    print(f"   股票: {stock_info.get('name', 'N/A')} ({stock_code})")
    print(f"   现价: {stock_info.get('current_price', 'N/A')}")
    print(f"   涨跌: {stock_info.get('change_percent', 'N/A')}%")

    # 2. 获取历史K线并计算指标
    print("\n📈 [2/3] 正在获取历史K线 & 计算技术指标...")
    df = fetcher.get_stock_data(stock_code, period="3mo")
    if isinstance(df, dict) and "error" in df:
        print(f"   ❌ 获取K线失败: {df['error']}")
        return
    print(f"   获取到 {len(df)} 条日K线")

    df = fetcher.calculate_technical_indicators(df)
    indicators = fetcher.get_latest_indicators(df)
    print(f"   收盘价: {indicators.get('price', 'N/A')}")
    print(f"   RSI: {indicators.get('rsi', 'N/A')}")
    print(f"   MACD: {indicators.get('macd', 'N/A')}")

    # 3. 调用 DeepSeek 技术分析
    print("\n🤖 [3/3] 正在调用 DeepSeek LLM 进行技术分析...")
    client = DeepSeekClient()
    try:
        report = client.technical_analysis(stock_info, df, indicators)
        print("\n" + "="*60)
        print("  📋 DeepSeek 技术分析报告")
        print("="*60)
        print(report)
    except Exception as e:
        print(f"   ❌ LLM 调用失败: {e}")

    print(f"\n{'='*60}")
    print(f"  ✅ 最小闭环验证完成！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 支持命令行传参: python demo.py 000001
    code = sys.argv[1] if len(sys.argv) > 1 else "000001"
    main(code)