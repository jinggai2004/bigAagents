"""A股多Agent分析系统 — Streamlit 主页

导航到:
- pages/1_📊_选股.py      全市场选股
- pages/2_🔍_深度分析.py  深度分析(复用现有分析界面)
- pages/3_📊_看盘.py      看盘
- pages/4_📈_回测看板.py  回测看板
- pages/5_💬_AI助手.py    AI对话助手
- pages/6_💼_持仓.py      持仓管理
- pages/7_⚙️_系统配置.py  系统配置
- pages/8_💰_模拟账户.py  模拟交易账户
- pages/9_🗣️_自然语言选股.py  自然语言选股（Phase 3）
- pages/10_📰_每日复盘.py 每日复盘（正文 + 历史存档 + 预测跟踪）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(page_title="A股多Agent分析系统", page_icon="📈", layout="wide")

# 进程级**后台预热**（2026-10-15）：把 LangGraph / akshare 等重依赖提前导入，
# 消除"第一次点某页卡十几秒"（实测冷导入 9~13s，见 ui/_preload.py 注释）。
# ⚠️ 只预热、不改语义；失败只记日志。测试环境由 BIGA_UI_OFFLINE 自动关闭。
from ui._preload import ensure_preloaded                       # noqa: E402

ensure_preloaded()

st.title("📈 A股多Agent分析系统")
st.markdown("> AI驱动的A股智能分析 — 技术/基本面/舆情三Agent协作 + 多空辩论 + 决策链")
st.divider()

# 快速入口
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/1_📊_选股.py", label="📊 全市场选股", icon="📊")
    st.caption("五路本地初筛（全市场快照）→ 六因子精筛 → TOP5（后台线程执行，刷新不中断）")
with col2:
    st.page_link("pages/2_🔍_深度分析.py", label="🔍 深度分析", icon="🔍")
    st.caption("多Agent分析 → 多空辩论 → 决策链")
col3, col4 = st.columns(2)
with col3:
    st.page_link("pages/3_📊_看盘.py", label="📊 看盘", icon="📊")
    st.caption("指数面板 → 自选股行情 → 行业热度")
with col4:
    st.page_link("pages/4_📈_回测看板.py", label="📈 回测看板", icon="📈")
    st.caption("净值曲线 → 回撤 → 交易明细 → 行业分布")
col5, col6 = st.columns(2)
with col5:
    st.page_link("pages/10_📰_每日复盘.py", label="📰 每日复盘", icon="📰")
    st.caption("大盘概览 → 资金板块 → 自选股 → 相似日 → 明日观察点（附命中率闭环）")
with col6:
    st.page_link("pages/6_💼_持仓.py", label="💼 持仓管理", icon="💼")
    st.caption("持仓录入 → 实时盈亏 → 风险敞口")
col7, col8 = st.columns(2)
with col7:
    st.page_link("pages/8_💰_模拟账户.py", label="💰 模拟账户", icon="💰")
    st.caption("虚拟资金 → 下单 → 撮合(T+1) → 净值曲线")
with col8:
    st.page_link("pages/5_💬_AI助手.py", label="💬 AI助手", icon="💬")
    st.caption("对话式查询持仓/行情/选股")
col9, col10 = st.columns(2)
with col9:
    st.page_link("pages/9_🗣️_自然语言选股.py", label="🗣️ 自然语言选股", icon="🗣️")
    st.caption("一句话 → 白名单条件 → 确认后执行")
with col10:
    st.page_link("pages/7_⚙️_系统配置.py", label="⚙️ 系统配置", icon="⚙️")
    st.caption("API Key / 模型接入 / 运行健康 / 定时任务")

st.divider()

# 系统状态
st.markdown("### 📂 系统模块状态")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.success("✅ 数据层")
    st.caption("AKShare + 降级策略")
with col_b:
    st.success("✅ Agent层")
    st.caption("10个Agent节点 + 短路保护")
with col_c:
    st.success("✅ 选股引擎")
    st.caption("MarketDetector + 单一权重集（六因子）")

st.divider()
st.caption("⚠️ AI分析仅供参考，不构成投资建议 | Powered by DeepSeek + LangGraph")