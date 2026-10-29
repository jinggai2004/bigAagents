"""市场检测仪表盘组件"""

import streamlit as st
from core.market_detector import MarketSignal


def render_market_dashboard(signal: MarketSignal):
    """渲染市场检测仪表盘"""

    if signal is None:
        st.info("暂无市场检测数据，请点击「开始检测」")
        return

    # === 状态信号卡片 ===
    state_colors = {
        "牛市": ("#E74C3C", "🟢"),
        "偏强震荡": ("#F39C12", "🟡"),
        "中性震荡": ("#95A5A6", "⚪"),
        "偏弱震荡": ("#E67E22", "🟠"),
        "熊市": ("#C0392B", "🔴"),
        # P0-1（2026-10-18）：指标可评率 < 50% ⇒ 结论降级中性（灰色 + 问号），
        # 与"真实的熊市"必须视觉可区分，否则用户会把"没数据"读成"市场差"。
        "数据不足（中性假设）": ("#95A5A6", "❔"),
    }
    color, emoji = state_colors.get(signal.market_state, ("#95A5A6", "⚪"))

    st.markdown(f"""
    <div style="border:2px solid {color}; border-radius:12px; padding:20px; margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h2 style="margin:0;">{emoji} {signal.market_state}</h2>
                <p style="margin:4px 0; color:#666;">总分: {signal.total_score} / 100 &nbsp;|&nbsp;
                信号: {signal.signal} &nbsp;|&nbsp; 策略: {signal.strategy_bias}</p>
            </div>
            <div style="text-align:right;">
                <span style="font-size:32px; font-weight:bold; color:{color};">{signal.position_limit*100:.0f}%</span>
                <p style="margin:0; color:#666;">仓位上限</p>
            </div>
        </div>
        <div style="background:#eee; border-radius:4px; height:8px; margin-top:12px;">
            <div style="background:{color}; width:{signal.total_score}%; height:100%; border-radius:4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # === 四维仪表盘 ===
    dimensions = {
        "📈 趋势": 0, "📊 量能": 0, "🎭 情绪": 0, "🏗️ 结构": 0
    }
    dim_weights = {
        "hs300_ma20_deviation": "📈 趋势", "sh_macd": "📈 趋势",
        "cyb_relative_strength": "📈 趋势",
        "volume_vs_ma20": "📊 量能", "advance_decline_ratio": "📊 量能",
        "north_flow": "📊 量能",
        "limit_up_down_ratio": "🎭 情绪", "max_consecutive_board": "🎭 情绪",
        "market_breadth": "🎭 情绪", "ivix_level": "🎭 情绪",
        "sector_rotation_speed": "🏗️ 结构", "size_style_bias": "🏗️ 结构",
        "margin_balance_change": "🏗️ 结构", "limit_up_upgrade_rate": "🏗️ 结构",
    }

    for key, detail in signal.scores.items():
        dim = dim_weights.get(key)
        if dim:
            dimensions[dim] += detail.get("weighted_score", 0)

    cols = st.columns(4)
    dim_max = {"📈 趋势": 30, "📊 量能": 25, "🎭 情绪": 25, "🏗️ 结构": 20}
    for i, (dim_name, dim_score) in enumerate(dimensions.items()):
        with cols[i]:
            max_s = dim_max[dim_name]
            ratio = dim_score / max_s if max_s > 0 else 0
            bar = "█" * int(ratio * 10) + "░" * (10 - int(ratio * 10))
            st.markdown(f"**{dim_name}**")
            st.markdown(f"`{dim_score:.0f}/{max_s}`")
            st.markdown(f"`{bar}`")

    # === 关键指标详情 ===
    with st.expander("📋 关键指标详情", expanded=False):
        for key, detail in signal.scores.items():
            if detail.get("detail"):
                st.caption(f"- {detail['detail']}")

    # === 风险提示 ===
    if signal.risk_warnings:
        warning_text = "\n\n".join(f"- {w}" for w in signal.risk_warnings[:3])
        st.warning(f"⚠️ 风险提示:\n\n{warning_text}")

    # === 操作建议 ===
    if signal.position_limit > 0:
        if signal.position_limit >= 0.6:
            st.success(f"📌 {signal.operation_advice}")
        elif signal.position_limit >= 0.4:
            st.info(f"📌 {signal.operation_advice}")
        else:
            st.warning(f"📌 {signal.operation_advice}")
    else:
        st.error(f"🔴 {signal.operation_advice}")