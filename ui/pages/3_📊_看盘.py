"""看盘页面 — 指数面板 + 自选股实时行情 + 行业资金热度榜"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from data import realtime
from data import watchlist
from data.fund_flow import get_industry_flow_snapshot
from ui.components.section_nav import section_nav, visible

st.set_page_config(page_title="看盘", page_icon="📊", layout="wide")
st.title("📊 看盘")

# ================================================================
# 取数缓存（2026-10-15 实测驱动）
# ================================================================
# ⚠️ 为什么必须缓存：Streamlit **每次交互都重跑整个脚本** ⇒ 切区块 / 点任何控件都会把
#    四个数据源重新取一遍（实测冷启动合计 ≈7.2s：外盘 2.96s、行业资金流 2.65s、
#    指数与自选行情各 0.79s）。
# ⚠️ TTL 一律**对齐各数据源自带的缓存口径**（外盘 30min / 行情 15s / 行业资金流 600s）
#    ⇒ 缓存只影响"取数次数"，**不改变任何数据新鲜度语义**。
# ⚠️ 代价是"手动刷新"必须显式清缓存（见下方 refresh_btn），否则会出现"点了刷新没反应"。
@st.cache_data(ttl=1800, show_spinner=False)
def _cached_overnight_us():
    from data.us_market import get_overnight_us
    return get_overnight_us()


@st.cache_data(ttl=15, show_spinner=False)
def _cached_index_quotes():
    return realtime.get_index_quotes()


@st.cache_data(ttl=15, show_spinner=False)
def _cached_watch_quotes(codes: tuple):
    # ⚠️ 入参用 **tuple**：list 不可哈希，`st.cache_data` 无法建键（会直接报错）。
    return realtime.get_realtime_quotes(list(codes))


@st.cache_data(ttl=600, show_spinner=False)
def _cached_industry_flow():
    # ⚠️ 返回值可能是 None（源不可达）—— 这是**有意的**：`st.cache_data` **不缓存异常**，
    #    所以只有把失败变成"正常返回值"，失败才会进缓存，否则每次 rerun 都要重等一遍
    #    （外部 TTL 只在成功路径生效，失败路径是每次重试）。
    return get_industry_flow_snapshot()


def _clear_page_caches() -> None:
    """手动"刷新行情"必须**同时**清掉页面级缓存，否则点了没反应（缓存命中旧值）。"""
    _cached_index_quotes.clear()
    _cached_watch_quotes.clear()


def color_change_pct(val):
    """A股惯例：红涨绿跌"""
    if isinstance(val, (int, float)):
        color = "#e4574c" if val > 0 else ("#3fa66a" if val < 0 else "#888")
        return f"color: {color}; font-weight: bold"


def render_overnight_us():
    """隔夜外盘 —— 作为 A 股**开盘情绪**的参考。

    放在最上面是因为外盘的作用窗口就在开盘（9:30–10:00 冲击最强，之后由内资主导）。
    看盘时第一眼要回答的问题就是"今天大概率高开还是低开"。

    为什么只给涨跌幅 + 分级、不给一个评分：外盘与 A 股的**宽基日度相关性只有 0.10 左右**，
    且 ±2% 以内对次日涨跌没有决定力（见 `data/us_market.py` docstring 的实证出处）。
    把它变成一个分数塞进总评分，只会制造精确的错觉。
    """
    try:
        data = _cached_overnight_us()
    except Exception as e:                                  # noqa: BLE001
        st.warning(f"外盘数据读取失败：{type(e).__name__}: {e}")
        return

    if not data or data.get("unavailable") or not data.get("indexes"):
        st.info(f"外盘数据暂不可用：{(data or {}).get('note', '未知原因')}")
        return

    cols = st.columns(len(data["indexes"]))
    for col, row in zip(cols, data["indexes"]):
        col.metric(row["name"], f"{row['close']:,.2f}",
                   delta=f"{row['change_pct']:+.2f}%",
                   # 红涨绿跌（A股惯例），故用 inverse —— Streamlit 默认 normal 是绿涨
                   delta_color="inverse" if row["change_pct"] >= 0 else "normal")
    st.caption(f"数据日期 **{data['data_date']}**（距今 {data['days_ago']} 天）"
               f"｜{data['level_text']}｜{data['note']}")


def render_index_panel():
    idx = _cached_index_quotes()
    cols = st.columns(3)
    for col, (full_code, name) in zip(cols, realtime.INDEX_CODES.items()):
        row = idx.get(full_code)
        with col:
            if row:
                col.metric(
                    name, f"{row['price']:,.2f}",
                    delta=f"{row['change']:+.2f} ({row['change_pct']:+.2f}%)",
                    delta_color="normal" if row["change"] >= 0 else "inverse",
                )
            else:
                col.metric(name, "数据不可用")
    stale_idx = [name for full_code, name in realtime.INDEX_CODES.items()
                 if (idx.get(full_code) or {}).get("stale")]
    if stale_idx:
        st.warning("⚠️ 指数报价为**过期缓存**（行情源暂时不可用）：" + "、".join(stale_idx)
                   + "。请勿据此做盘中判断。")


def render_watch_table():
    codes = watchlist.watchlist_codes()
    if not codes:
        st.info("自选股为空，请在下方添加")
        return None
    quotes = _cached_watch_quotes(tuple(codes))
    # 双源全失败时会回退过期缓存（带 stale 标记）——必须显式提示，
    # 否则用户会把昨天的价格当成实时价（v2/M0 0.4）。
    stale_codes = [c for c in codes if (quotes.get(c) or {}).get("stale")]
    if stale_codes:
        max_age = max(quotes[c].get("age_seconds", 0) for c in stale_codes)
        st.warning(f"⚠️ 行情源暂时不可用，以下 {len(stale_codes)} 只显示的是**过期缓存**"
                   f"（最长 {max_age:.0f} 秒前）：{'、'.join(stale_codes)}。请勿据此做盘中判断。")
    rows = []
    for code in codes:
        q = quotes.get(code)
        if not q:
            rows.append({"代码": code, "名称": "—", "现价": None, "涨跌": None,
                         "涨跌幅%": None, "最高": None, "最低": None,
                         "换手率%": None, "成交额(亿)": None, "时间": "不可用",
                         "数据状态": "不可用"})
            continue
        rows.append({
            "代码": q["code"], "名称": q["name"] or code,
            "现价": q["price"], "涨跌": q["change"], "涨跌幅%": q["change_pct"],
            "最高": q["high"], "最低": q["low"],
            "换手率%": q["turnover_rate"],
            "成交额(亿)": round(q["amount_wan"] / 1e4, 2) if q["amount_wan"] else None,
            "时间": q["quote_time"],
            "数据状态": (f"过期 {q.get('age_seconds', 0):.0f}s" if q.get("stale") else "实时"),
        })
    df = pd.DataFrame(rows)
    styler = df.style.map(color_change_pct, subset=["涨跌幅%"])
    st.dataframe(styler, width="stretch", hide_index=True, height=420)
    return quotes


def render_industry_heat():
    df = _cached_industry_flow()
    if df is None or df.empty:
        st.caption("行业资金流数据不可用（同花顺源暂时不可达）")
        return
    top_in = df.head(10).iloc[::-1]
    top_out = df.tail(10).sort_values("净额")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_in["行业"], x=top_in["净额"], orientation="h",
        marker_color="#e4574c", name="净流入"))
    fig.add_trace(go.Bar(
        y=top_out["行业"], x=top_out["净额"], orientation="h",
        marker_color="#3fa66a", name="净流出"))
    fig.update_layout(
        height=420, showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="主力净流入 (亿元)")
    st.plotly_chart(fig, width="stretch")
    with st.expander("完整行业资金流（90个行业）"):
        show = df[["行业", "净额", "流入资金", "流出资金", "涨跌幅", "领涨股"]].copy()
        show = show.style.map(color_change_pct, subset=["涨跌幅"])
        st.dataframe(show, width="stretch", hide_index=True, height=360)


def render_today_alerts():
    """盘中异动预警（Phase 1）：读当日 alerts 表并在页面上展示"""
    from db import store
    try:
        rows = store.get_recent_alerts(limit=30)
    except Exception as e:               # 读库失败不该让整页崩掉
        st.caption(f"预警记录读取失败: {e}")
        return
    if not rows:
        st.caption("今日暂无命中异动（交易时段每分钟检查一次；同标的同规则 30 分钟内不重复提醒）")
        return
    icons = {"danger": "🔴", "warn": "🟠", "info": "🔵"}
    df = pd.DataFrame([{
        "时间": (r.get("created_at") or "")[11:19],
        "级别": icons.get(r.get("level"), "•"),
        "代码": r.get("code"), "名称": r.get("name") or "—",
        "规则": r.get("rule"), "说明": r.get("message"),
        "现价": r.get("price"), "涨跌幅%": r.get("change_pct"),
    } for r in rows])
    st.dataframe(df, width="stretch", hide_index=True, height=280)


# ===== 控制栏 =====
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    refresh_btn = st.button("🔄 刷新行情", type="primary", width="stretch")
with col2:
    manage_on = st.toggle("管理自选股")
with col3:
    st.caption("最后更新：{}（行情缓存 15 秒；外盘 30 分钟、行业资金流 10 分钟按各自周期）"
               .format(datetime.now().strftime("%H:%M:%S")))

if refresh_btn:
    # ⚠️ 两处都要清：`realtime` 的内部缓存 + **页面级 `st.cache_data`**。
    #    只清前者会出现"点了刷新，页面还是旧价"（页面缓存命中，函数压根没被调用）。
    realtime.clear_quote_cache()
    _clear_page_caches()
    st.rerun()

# ===== 自选股管理 =====
if manage_on:
    with st.container(border=True):
        st.markdown("**添加自选股**")
        add_col1, add_col2, add_col3 = st.columns([2, 2, 1])
        with add_col1:
            new_code = st.text_input("股票代码", placeholder="6位代码，如 601127", max_chars=6,
                                     label_visibility="collapsed")
        with add_col2:
            new_note = st.text_input("备注（可选）", placeholder="备注", label_visibility="collapsed")
        with add_col3:
            st.write("")
            add_btn = st.button("➕ 添加", width="stretch")
        if add_btn and new_code:
            ok, msg = watchlist.add_stock(new_code, note=new_note)
            (st.success if ok else st.error)(msg)

        st.markdown("**移除自选股**")
        del_col1, del_col2 = st.columns([3, 1])
        stocks = watchlist.load_watchlist()
        options = [f"{s['code']} {s['name'] or ''}".strip() for s in stocks]
        with del_col1:
            to_remove = st.selectbox("选择要移除的股票", options, label_visibility="collapsed")
        with del_col2:
            st.write("")
            del_btn = st.button("🗑️ 移除", width="stretch")
        if del_btn and to_remove:
            ok, msg = watchlist.remove_stock(to_remove.split()[0])
            (st.success if ok else st.error)(msg)

# ===== 主体 =====
# 区块快速定位（2026-10-15）：本页 5 个区块纵向很长，靠滚动找很费劲 ⇒ 顶部下拉框切换。
# ⚠️ 默认「全部显示」⇒ 与改造前行为一致；各区块的数据取用互不依赖，可安全独立渲染。
SEC_US = "隔夜外盘"
SEC_INDEX = "大盘指数"
SEC_WATCH = "自选股行情"
SEC_HEAT = "行业资金热度（今日）"
SEC_ALERT = "盘中异动（今日）"
_SEL = section_nav((SEC_US, SEC_INDEX, SEC_WATCH, SEC_HEAT, SEC_ALERT),
                   key="pan_section", group_label="📊 看盘 · 板块")

if visible(_SEL, SEC_US):
    st.subheader(SEC_US)
    render_overnight_us()

if visible(_SEL, SEC_INDEX):
    st.subheader(SEC_INDEX)
    render_index_panel()

if visible(_SEL, SEC_WATCH):
    st.subheader(SEC_WATCH)
    render_watch_table()

if visible(_SEL, SEC_HEAT):
    st.subheader(SEC_HEAT)
    render_industry_heat()

if visible(_SEL, SEC_ALERT):
    st.subheader(SEC_ALERT)
    alert_col1, alert_col2 = st.columns([1, 4])
    with alert_col1:
        if st.button("🔔 立即检查", width="stretch"):
            # 手动检查：跳过交易时段闸门，便于非盘中验证规则是否生效；
            # 推送关闭，避免手动点一下就发通知。
            from core.watcher import AlertWatcher
            with st.spinner("检查中…"):
                res = AlertWatcher(push=False).scan_once(force=True)
            if res.get("skipped"):
                st.warning(f"未执行：{res.get('reason')}")
            else:
                st.success(f"检查 {res.get('checked')} 只，命中 {res.get('alerts')} 条，"
                           f"新增记录 {res.get('saved')} 条")
                if res.get("note"):
                    st.caption(res["note"])
    with alert_col2:
        st.caption("盘中每分钟自动检查（交易日 09:30–11:30 / 13:00–15:00）；"
                   "同一标的同一规则 30 分钟内不重复提醒，推送走钉钉/飞书。")
    render_today_alerts()

st.caption("⚠️ 行情数据来自腾讯/新浪公开接口，可能有延迟，仅供参考，不构成投资建议。")
