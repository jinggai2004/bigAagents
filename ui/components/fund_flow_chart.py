"""资金流向分析可视化

compute_five_dim_score() 从 data.fund_flow 导入

⚠️ **两条独立渲染路径**（2026-09-28）：
  - 逐日模式（东财/tushare）：有单型拆分（主力/超大单/大单/中单/小单）+ 逐日序列
  - 快照模式（同花顺降级源，盘后 15:40 落库）：**只有各窗口累计净额**
    （`净额_今日/3日/5日/10日`），没有单型拆分、没有逐日序列

原实现只有逐日模式一套模板，直接取 `recent_5["主力净流入_亿"]`。深度分析走的是快照模式，
映射不出这一列 → `KeyError: '主力净流入_亿'` → 被兜底吞成一句
「资金流向数据渲染失败」，用户看不到任何数据也看不到原因。
"""

import streamlit as st
import pandas as pd

from data.fund_flow import (compute_five_dim_score, COL_MAIN_FLOW, COL_SUPER_LARGE,
                            COL_LARGE, SNAPSHOT_COLS)

COL_MEDIUM = "中单净流入-净额"
COL_SMALL = "小单净流入-净额"
COL_DATE = "日期"

# 源列 → (展示用「亿元」列名, 指标标题)
_AMOUNT_COLS = {
    COL_MAIN_FLOW: ("主力净流入_亿", "主力净流入"),
    COL_SUPER_LARGE: ("超大单净流入_亿", "超大单"),
    COL_LARGE: ("大单净流入_亿", "大单"),
}

# 快照模式的窗口列 → 展示名
SNAPSHOT_WINDOWS = (("净额_今日", "今日"), ("净额_3日", "3日"),
                    ("净额_5日", "5日"), ("净额_10日", "10日"))


def _yi(value):
    """元 → 亿元。取不到返回 None（**绝不返回 0**——0 会被读成"净额恰好为 0"）。"""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num:                      # NaN
        return None
    return num / 1e8


def _is_snapshot(flow_data: dict, df) -> bool:
    """是否同花顺快照模式。

    两重判据：`mode` 字段（新数据）+ 列名特征（历史存档 JSON 里可能没有 mode）。
    """
    if str(flow_data.get("mode") or "").lower() == "snapshot":
        return True
    try:
        return SNAPSHOT_COLS.issubset(set(df.columns)) and COL_MAIN_FLOW not in df.columns
    except Exception:                   # noqa: BLE001 - 判据失败按逐日模式处理
        return False


def render_five_dim_score(five_dim: dict):
    """渲染五维评分仪表盘。

    ⚠️ 两处修正（2026-10-10，用户报"资金面五维评分为空"）：
    1. **不可评维度**（`available=False`，如快照源的「结构」无单型拆分）**没有分数**，
       不能再做 `score / dim_max` —— 那会抛 `TypeError` 并被上层 except 吞成
       "资金流向数据渲染失败"，用户既看不到数据也看不到原因。它们照常列出，
       显示 `—` 并给出"不可评"的原因。
    2. **0 分不再直接 return**：全 0 是"资金持续净流出"这一**结论**，
       跳过渲染等于把它藏起来 —— 用户看到的正是"资金面为空"。
    """
    if not five_dim:
        return
    max_s = five_dim.get("max")
    if not max_s:
        return

    st.markdown("### 📊 资金流向五维评分")

    breakdown = five_dim.get("breakdown", {}) or {}
    cols = st.columns(max(len(breakdown), 1))
    for i, (dim_name, dim_data) in enumerate(breakdown.items()):
        with cols[i]:
            dim_data = dim_data or {}
            dim_max = dim_data.get("max", 0)
            score = dim_data.get("score")
            st.markdown(f"**{dim_name}**")
            if score is None or dim_data.get("available") is False:
                # 不可评：画成 0 分会让人误以为"该维度得 0 分"
                st.caption(f"`—/{dim_max}` `{'░' * 10}`")
                st.caption(dim_data.get("desc") or "不可评")
                continue
            ratio = score / dim_max if dim_max > 0 else 0
            filled = int(ratio * 10)
            bar = "█" * filled + "░" * (10 - filled)
            st.caption(f"`{score}/{dim_max}` `{bar}`")
            st.caption(dim_data.get("desc", ""))

    # 总分条
    score = five_dim.get("score") or 0
    ratio = (score / max_s) if max_s else 0
    color = "#27AE60" if ratio >= 0.6 else "#F39C12" if ratio >= 0.3 else "#E74C3C"
    st.markdown(f"""
    <div style="background:#eee; border-radius:4px; height:20px; margin:4px 0 12px 0;">
        <div style="background:{color}; width:{ratio*100}%; height:100%; border-radius:4px;
             display:flex; align-items:center; justify-content:center; color:white; font-size:12px;">
            总分: {score}/{max_s}
        </div>
    </div>
    """, unsafe_allow_html=True)
    if score == 0:
        st.caption("⚠️ 总分为 0 是**数据结果**（资金持续净流出），不是渲染失败；"
                   "各维度依据见上方。")


def _render_snapshot_section(flow_data: dict, df) -> None:
    """快照模式：只有各窗口累计净额。"""
    record = df.iloc[-1].to_dict() if len(df) else {}
    data_date = flow_data.get("data_date") or record.get("数据日期") or "未知"
    source = flow_data.get("source") or "ths_snapshot"

    st.markdown("#### 💰 资金流向（盘后落库快照）")
    st.caption(
        f"数据日期：{data_date}　|　来源：{source}　|　口径：各窗口**累计净额**"
        "（全单型合计，近似主力）。快照源**无单型拆分、无逐日序列**，"
        "因此不提供逐日柱状图，也不计算「近5日流入天数」。"
    )

    cols = st.columns(len(SNAPSHOT_WINDOWS))
    for (key, label), col in zip(SNAPSHOT_WINDOWS, cols):
        value = _yi(record.get(key))
        with col:
            st.metric(f"{label}净额", "—" if value is None else f"{value:+.2f}亿")

    absent = [label for key, label in SNAPSHOT_WINDOWS if record.get(key) is None]
    if absent:
        st.caption(f"⚠️ 本次快照缺少窗口：{'、'.join(absent)}")

    five_dim = compute_five_dim_score(df)
    render_five_dim_score(five_dim)
    if five_dim.get("note"):
        st.caption(f"评分口径：{five_dim['note']}")


def _render_daily_section(df) -> None:
    """逐日模式：有单型拆分与逐日序列。"""
    # 只对**存在**的列做单位换算；后面每处取值前都先判列是否存在
    for src, (dst, _label) in _AMOUNT_COLS.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce") / 1e8

    if not any(dst in df.columns for dst, _l in _AMOUNT_COLS.values()):
        st.warning("资金流向数据里没有任何单型金额列，无法渲染明细。"
                   f"实际字段：{list(df.columns)}")
        return

    five_dim = compute_five_dim_score(df)
    render_five_dim_score(five_dim)

    recent_5 = df.tail(5)
    st.markdown("#### 近5日资金流向")
    cols = st.columns(4)

    for (src, (dst, label)), col in zip(_AMOUNT_COLS.items(), cols[:3]):
        with col:
            if dst not in df.columns:
                st.metric(label, "—")
                st.caption(f"缺列 {src}")
                continue
            total = recent_5[dst].sum()
            st.metric(label, f"{total:.2f}亿",
                      delta=f"{total:+.2f}亿" if total and src == COL_MAIN_FLOW else None)

    with cols[3]:
        if COL_MAIN_FLOW in df.columns:
            series = recent_5[COL_MAIN_FLOW]
            st.metric("净流入天数", f"{(series > 0).sum()}/{len(series)}")
        else:
            st.metric("净流入天数", "—")
            st.caption(f"缺列 {COL_MAIN_FLOW}")

    chart_cols = [dst for dst, _l in _AMOUNT_COLS.values() if dst in df.columns]
    if COL_DATE in df.columns and chart_cols:
        st.bar_chart(df.set_index(COL_DATE)[chart_cols].tail(20), height=250)
    elif not chart_cols:
        st.info("缺少单型金额列，跳过逐日柱状图。")

    with st.expander("📋 近5日资金流向明细", expanded=False):
        display_cols = [c for c in (COL_DATE, *chart_cols) if c in df.columns]
        if display_cols:
            table = recent_5[display_cols]
            if COL_DATE in table.columns:
                table = table.set_index(COL_DATE)
            st.dataframe(table.style.format("{:.2f}"), width="stretch")


def render_fund_flow_section(fund_flow_data: dict):
    """渲染资金流向分析区域（自动分流快照模式 / 逐日模式）"""

    if not fund_flow_data or not fund_flow_data.get("data_success"):
        st.info("📊 暂无资金流向数据")
        return

    try:
        flow_data = fund_flow_data.get("fund_flow_data") or {}
        df_list = flow_data.get("data") or []

        if not df_list:
            st.info("📊 资金流向数据为空")
            return

        df = pd.DataFrame(df_list)

        if _is_snapshot(flow_data, df):
            _render_snapshot_section(flow_data, df)
        else:
            _render_daily_section(df)

    except Exception as e:                      # noqa: BLE001 - 渲染失败不影响报告其余部分
        # 只写 "渲染失败: 'xxx'" 的话，用户既看不到数据也看不到原因（原实现的问题）。
        # 这里把异常类型和**实际字段**一起给出，便于直接定位是缺列还是格式问题。
        st.warning(f"资金流向数据渲染失败: {type(e).__name__}: {e}")
        try:
            records = (fund_flow_data.get("fund_flow_data") or {}).get("data") or []
            if records:
                st.caption(f"（诊断用）数据实际字段：{list(records[0].keys())}")
        except Exception:                       # noqa: BLE001
            pass
