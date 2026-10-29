"""回测看板页面 — 读取 data/backtest_records 回测报告，展示净值/回撤/交易/行业分布"""

import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scripts.backtest_rule_based import INDUSTRY_MAP
from ui.components.section_nav import section_nav, visible

RECORDS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "backtest_records")

st.set_page_config(page_title="回测看板", page_icon="📈", layout="wide")
st.title("📈 回测看板")

DIRECTION_MAP = {"买入": "买入", "卖出": "卖出", "BUY": "买入", "SELL": "卖出"}


@st.cache_data(ttl=60)
def list_reports():
    """列出回测报告（文件名倒序 = 时间倒序），返回 (显示名, 绝对路径) 列表"""
    files = sorted(glob.glob(os.path.join(RECORDS_DIR, "backtest_*.json")), reverse=True)
    return [(os.path.basename(f), f) for f in files]


def load_report(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def drawdown_series(nav_df: pd.DataFrame) -> pd.Series:
    peak = nav_df["total_value"].cummax()
    return (nav_df["total_value"] / peak - 1) * 100


def render_nav_chart(report):
    nav = report.get("nav_curve") or []
    if not nav:
        st.info("该报告无净值数据")
        return
    nav_df = pd.DataFrame(nav)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav_df["date"], y=nav_df["total_value"], name="策略净值",
        line=dict(color="#1f77b4", width=2), fill="tozeroy", fillcolor="rgba(31,119,180,0.06)"))
    bench = report.get("benchmark_nav")
    if bench:
        bench_df = pd.DataFrame(bench)
        fig.add_trace(go.Scatter(
            x=bench_df["date"], y=bench_df["total_value"], name="基准(沪深300)",
            line=dict(color="#ff7f0e", width=1.6, dash="dash")))
    fig.update_layout(
        height=420, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="组合市值 (元)")
    st.plotly_chart(fig, width="stretch")


def render_drawdown_chart(report):
    nav = report.get("nav_curve") or []
    if not nav:
        return
    nav_df = pd.DataFrame(nav)
    dd = drawdown_series(nav_df)
    fig = go.Figure(go.Scatter(
        x=nav_df["date"], y=dd, name="回撤",
        line=dict(color="#d62728", width=1.5), fill="tozeroy", fillcolor="rgba(214,39,40,0.15)"))
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="回撤 (%)", yaxis=dict(ticksuffix="%"))
    st.plotly_chart(fig, width="stretch")


def render_industry_chart(report):
    """行业分布：从买入交易按 INDUSTRY_MAP 聚合买入金额"""
    trades = report.get("trade_history") or []
    # ⚠️ v3（P1-4）：trade_history 里现在可能含"被拒订单"（status=rejected，没有 cost），
    # 它们不构成持仓，计入行业金额会凭空多出 0 值行
    trades = [t for t in trades if t.get("status") != "rejected"]
    buys = [t for t in trades if DIRECTION_MAP.get(str(t.get("direction", "")).strip()) == "买入"]
    if not buys:
        st.caption("无买入交易")
        return
    rows = []
    for t in buys:
        rows.append({
            "行业": INDUSTRY_MAP.get(str(t.get("code", "")), "其他"),
            "买入金额": abs(float(t.get("cost", 0) or 0)),
        })
    df = pd.DataFrame(rows).groupby("行业", as_index=False)["买入金额"].sum()
    fig = go.Figure(go.Pie(
        labels=df["行业"], values=df["买入金额"], hole=0.45,
        textinfo="label+percent", marker=dict(line=dict(color="#fff", width=1))))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      showlegend=False)
    st.plotly_chart(fig, width="stretch")


def render_trades(report):
    trades = report.get("trade_history") or []
    if not trades:
        st.info("该报告无交易明细")
        return
    df = pd.DataFrame(trades)
    if "direction" in df.columns:
        df["方向"] = df["direction"].map(lambda v: DIRECTION_MAP.get(str(v).strip(), str(v)))
    # v3（P1-4）：被拒订单也进明细（否则"策略想买却没买成"这件事就查不到）；
    # 排最后并标出状态，避免被误读成一笔成交
    if "status" in df.columns and (df["status"] == "rejected").any():
        df["状态"] = df["status"].map(lambda v: "未成交(被拒)" if v == "rejected" else "成交")
        df = df.sort_values("status", key=lambda s: (s == "rejected"))
    show_cols = [c for c in ["date", "code", "方向", "状态", "quantity", "actual_price",
                             "revenue", "pnl"] if c in df.columns]
    rename = {"date": "日期", "code": "代码", "quantity": "数量", "actual_price": "成交价",
              "revenue": "发生金额", "pnl": "盈亏"}
    df = df[show_cols].rename(columns=rename)
    # 买入交易无盈亏/发生金额：统一为数值列并格式化为字符串显示（"—" 或千分位），
    # 避免 object 列混入 str/float 导致 pyarrow 序列化失败
    for col in ("发生金额", "盈亏"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].map(lambda v: "—" if pd.isna(v) else f"{v:,.2f}")
    st.dataframe(df, width="stretch", hide_index=True, height=320)


def main():
    reports = list_reports()
    if not reports:
        st.warning(f"未找到回测报告（目录: `{RECORDS_DIR}`）")
        st.markdown("请在项目根目录运行：`python scripts/backtest_rule_based.py --quick`")
        return

    top_col1, top_col2 = st.columns([4, 1])
    with top_col1:
        choice = st.selectbox(
            "选择回测报告",
            reports,
            format_func=lambda x: x[0],
            index=0,
        )
    with top_col2:
        st.write("")
        st.write("")
        refresh = st.button("🔄 刷新列表", width="stretch")
    if refresh:
        st.cache_data.clear()
        st.rerun()

    report = load_report(choice[1])


    # ===== 区块快速定位（2026-10-15）=====
    # 本页 5 个区块纵向很长 ⇒ 顶部下拉框切换。⚠️ 默认「全部显示」保持改造前行为；
    #   报告选择/刷新等**共用前置**仍在各区块之外，任何选择下都先算好。
    SEC_OVERVIEW = "概览指标（收益/夏普/回撤…）"
    SEC_NAV = '净值曲线'
    SEC_DD = '回撤 / 行业分布'
    SEC_YEAR = '逐年收益'
    SEC_TRADES = '交易明细'
    _SEL = section_nav((SEC_OVERVIEW, SEC_NAV, SEC_DD, SEC_YEAR, SEC_TRADES),
                       key="bt_section", group_label="📈 回测看板 · 板块")
    if visible(_SEL, SEC_OVERVIEW):
        # ===== 指标卡 =====
        metrics_row1 = st.columns(5)
        metrics_row2 = st.columns(5)
        m = {
            "总收益": f"{report.get('total_return', 0):+.2f}%",
            "年化收益": f"{report.get('annual_return', 0):+.2f}%",
            "夏普比率": f"{report.get('sharpe_ratio', 0):.2f}",
            "最大回撤": f"{report.get('max_drawdown', 0):.2f}%",
            "信息比率": f"{report.get('information_ratio', 0):.2f}",
            "胜率": f"{report.get('win_rate', 0):.1f}%",
            "盈亏比": f"{report.get('profit_loss_ratio', 0):.2f}",
            "卡玛比率": f"{report.get('calmar_ratio', 0):.2f}",
            "月度胜率": f"{report.get('monthly_win_rate', 0):.1f}%",
            "交易次数": f"{report.get('total_trades', 0)}",
        }
        for col, (label, value) in zip(metrics_row1, list(m.items())[:5]):
            col.metric(label, value)
        for col, (label, value) in zip(metrics_row2, list(m.items())[5:]):
            col.metric(label, value)

        has_bench = bool(report.get("benchmark_nav"))
        if not has_bench:
            st.caption("ℹ️ 该报告未接入基准净值（旧版报告），信息比率按 0 显示。重新运行回测即可获得基准对比。")

        # ===== v3：口径类指标（旧报告没有这两个字段 → 整块不显示，避免出现一排 0）=====
        if "avg_position_pct" in report or "rejected_orders" in report:
            extra = st.columns(3)
            extra[0].metric("平均仓位", f"{report.get('avg_position_pct', 0):.1f}%")
            extra[1].metric("被拒订单", f"{report.get('rejected_orders', 0)}")
            extra[2].metric("净值点数", f"{len(report.get('nav_curve') or [])}")
            st.caption(
                "⚠️ **半仓口径**：单笔预算 10% × 最多 5 只 = 仓位上限 **50%**。"
                "与满仓的指数基准对比时，超额收益与信息比率会被系统性压低——"
                "解读时需按平均仓位调整口径，不要直接读成「策略跑输基准」。\n\n"
                "⚠️ **股票池偏差**：回测池为当期行业分层抽样（见 `data/backtest_pool.json`），"
                "已消除「手挑 15 只白马」那一层选样偏差，但**幸存者偏差仍在**"
                "（此前退市的票不在今天的成分里）。结论应与随机对照（`--baseline`）一起读。")

    if visible(_SEL, SEC_NAV):
        # ===== 净值曲线 =====
        st.subheader(SEC_NAV)
        render_nav_chart(report)

    if visible(_SEL, SEC_DD):
        left, right = st.columns([3, 2])
        with left:
            st.subheader("回撤")
            render_drawdown_chart(report)
        with right:
            st.subheader("行业分布（买入金额）")
            render_industry_chart(report)

    if visible(_SEL, SEC_YEAR):
        # ===== 逐年收益 =====
        yearly = report.get("yearly_returns") or {}
        if yearly:
            st.subheader(SEC_YEAR)
            ydf = pd.DataFrame({"年份": list(yearly.keys()),
                                "收益率(%)": [f"{v:+.2f}" for v in yearly.values()]})
            st.dataframe(ydf, width="stretch", hide_index=True)

    if visible(_SEL, SEC_TRADES):
        # ===== 交易明细 =====
        st.subheader(SEC_TRADES)
        render_trades(report)


main()
