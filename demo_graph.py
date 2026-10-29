"""多Agent分析系统 - LangGraph 端到端验证"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.graph_builder import analysis_graph
from core.state import AnalysisState


def main(stock_code="000001"):
    print(f"\n{'='*60}")
    print(f"  A股多Agent分析系统 - LangGraph 端到端验证")
    print(f"  流程: 数据获取 → 3Agent并行 → Bull/Bear → 决策")
    print(f"{'='*60}\n")

    # 初始状态
    initial_state: AnalysisState = {
        "stock_code": stock_code,
        "stock_name": "",
        "analysis_date": "",
        "stock_info": {},
        "kline_data": None,
        "indicators": {},
        "tech_report": "",
        "fundamental_report": "",
        "news_report": "",
        "bull_argument": "",
        "bear_argument": "",
        "debate_count": 0,
        "research_summary": "",
        "trade_plan": {},
        "risk_assessment": "",
        "final_decision": "",
        "messages": [],
        "errors": [],
    }

    print(f"🚀 开始分析: {stock_code}")
    print("-" * 40)

    # 流式执行图（只执行一次，边执行边收集状态）
    step_count = 0
    final_state = None
    for event in analysis_graph.stream(initial_state):
        step_count += 1
        node_name = list(event.keys())[0]
        node_data = event[node_name]
        final_state = node_data  # 累积最新状态

        # 显示进度（使用 node_name 作为步骤标识）
        print(f"  [{step_count}] {node_name}")

        # 显示特殊输出
        if node_name == "data_fetcher" and "stock_info" in node_data:
            info = node_data["stock_info"]
            print(f"       股票: {info.get('name', 'N/A')} | 现价: {info.get('current_price', 'N/A')}")

        if "errors" in node_data and node_data["errors"]:
            for err in node_data["errors"]:
                print(f"       ⚠️ {err}")

    print("-" * 40)
    print(f"✅ 分析完成！共执行 {step_count} 步")
    print(f"{'='*60}\n")

    if final_state:
        print("📊 最终结果:")
        fdc = final_state.get('final_decision', 'N/A')
        print(f"  最终决策: {str(fdc)[:200]}...")


if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "000001"
    main(code)