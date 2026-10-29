"""AI 助手页面 — 与投研对话 Agent 聊天（工具调用过程可视化 + 对话内图表 + 人格化快评）

持久化（P1 / 批次 14）
---------------------
以前对话只写 `chat_logs`、没有读取路径：session_id 存在 `st.session_state` 里，
刷新页面就换号 → 历史"写了也读不出来"。
现在：
- `chat_sessions` 会话维表提供**会话列表 / 回放 / 删除**；
- 当前会话 id 固化到 URL 查询参数 `?sid=xxx`，刷新与分享链接都能回到同一会话；
- 「清空对话」同步清库，不再出现"页面上清了、刷新又回来"。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging

import streamlit as st
import plotly.graph_objects as go

from core.chat_agent import ChatAgent, PERSONAS, split_fund5_dims
from core import chat_history as chat_store

_logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI 助手", page_icon="💬", layout="wide")
st.title("💬 AI 助手")
st.caption("查行情 · 看资金面 · 技术面 · 选股 · 回测 · 管理自选股 · "
           "**读项目自己的报告/运行记录/配置** — 助手会自动调用工具取真实数据")

# ===== 会话状态 =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # [{"role": "user"/"assistant", "content": str}]


EXAMPLES = [
    "现在自选股行情怎么样？",
    "帮我看看茅台的资金面",
    "平安银行技术面如何？",
    "当前大盘环境适合买吗？",
    "把 601127 加到自选股",
    "昨天的复盘报告说了什么？",
    "最近有什么告警、数据是不是最新的？",
]

SID_PARAM = "sid"     # URL 查询参数名：当前会话 id（刷新后靠它恢复）


# ================================================================
# 会话切换：URL ?sid= 是"刷新后"的唯一真相，session_state 是运行时的真相
# ================================================================

def _query_sid() -> str | None:
    """读 URL 里的会话 id（可能重复出现，取第一个非空值）"""
    try:
        raw = st.query_params.get(SID_PARAM)
    except Exception:
        return None
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    return raw or None


def _write_query_sid(sid: str | None):
    """把会话 id 写回 URL。⚠️ 只是给浏览器发消息、不会触发重跑，需要重跑时自己 st.rerun()"""
    try:
        if sid:
            st.query_params[SID_PARAM] = sid
        elif SID_PARAM in st.query_params:
            del st.query_params[SID_PARAM]
    except Exception:
        pass  # URL 写不进去不影响聊天，只影响刷新后能否恢复


def _load_session(sid: str):
    """从库里恢复会话：消息 + 累计 token + 当时的人格"""
    st.session_state.chat_session_id = sid
    st.session_state.chat_history = chat_store.load_messages(sid)
    meta = chat_store.load_session_meta(sid)
    st.session_state.chat_usage = {
        "prompt": meta.get("prompt_tokens") or 0,
        "completion": meta.get("completion_tokens") or 0,
    }
    if meta.get("persona"):
        st.session_state.persona = meta["persona"]


def _sync_session_from_url():
    """脚本开头同步一次：URL 指向别的会话就切换过去"""
    sid = _query_sid()
    if not sid or sid == st.session_state.get("chat_session_id"):
        return
    _load_session(sid)


def _switch_to(sid: str | None):
    """切换/新建会话：改状态 + 改 URL + 重跑"""
    if sid:
        _load_session(sid)
    else:
        st.session_state.pop("chat_session_id", None)
        st.session_state.chat_history = []
        st.session_state.chat_usage = {"prompt": 0, "completion": 0}
    _write_query_sid(sid)
    st.rerun()


def _chat_model_label() -> str:
    """当前对话主循环实际用的模型名（**运行时解析**，不写死）。

    为什么不在页面里写死：原先硬编码 "qwen-turbo"，路由早已切到
    deepseek-v4-flash 而文案一直是旧的（"一个变量不能承载两类语义"的同款坑：
    展示口径与真实路由必须同源）。resolve 失败也不阻断页面，退回"默认档"。
    """
    try:
        from config.model_routing import resolve_model
        return resolve_model("chat.tool_loop") or "默认档"
    except Exception as e:                              # noqa: BLE001 - 文案失败不能挡住聊天
        _logger.debug("[AI助手页] 模型名解析失败（退回默认档文案）: %s", e)
        return "默认档"


def render_kline_chart(chart: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart["dates"], y=chart["closes"], name="收盘价",
        line=dict(color="#1f77b4", width=2)))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                      title=chart.get("title", ""))
    st.plotly_chart(fig, width="stretch")


def render_fund5_chart(chart: dict):
    """资金面五维评分柱状图。

    ⚠️ 两件事必须做对（2026-10-10 用户报"资金面五维评分为空"，追出的就是这里）：
    1. **不可评的维度不画柱**：数据源取不到的维度（快照源的「结构」无单型拆分）
       若照旧画成 0 分，图上就是一根空柱 —— 看起来像"渲染失败"，也像"该维度 0 分"，
       两种解读都不对；它们应单独列出并说明"已从分母剔除"。
    2. **全 0 要说清是"真的差"**：各维度都 0 时柱子高度全为 0 → 图上一片空白，
       必须显式写出依据（每个维度的 desc），否则用户无法区分
       "资金持续流出"与"数据没取到"。
    """
    groups = split_fund5_dims(chart.get("breakdown", {}) or {})
    scored, unavailable = groups["scored"], groups["unavailable"]

    def _unavail_text() -> str:
        return "；".join(f"{u['name']}（{u['desc']}）" for u in unavailable)

    if not scored:
        st.caption("资金面五维评分：**全部维度都不可评**（数据源限制），无法给出评分。")
        if unavailable:
            st.caption("⚪ 不可评维度：" + _unavail_text())
        return

    names = [s["name"] for s in scored]
    scores = [s["score"] for s in scored]
    maxes = [s["max"] for s in scored]
    descs = [s["desc"] for s in scored]

    fig = go.Figure(go.Bar(
        x=names, y=scores, marker_color="#e4574c",
        text=[f"{s}/{m}" for s, m in zip(scores, maxes)], textposition="outside",
        hovertext=descs, hoverinfo="text"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                      title=f"{chart.get('title', '')}  总分 {chart.get('score', '')}",
                      yaxis_title="得分")
    st.plotly_chart(fig, width="stretch")

    # 逐维依据：让用户能判断"0 分"到底是资金差还是没数据
    st.caption("　".join(f"**{n}**：{d}" for n, d in zip(names, descs) if d))
    if all(s <= 0 for s in scores):
        st.warning("⚠️ 五维评分**各维度均为 0 分** —— 这是**数据结果**（资金持续净流出），"
                   "不是渲染失败；上方逐维依据可核对。")
    if unavailable:
        st.caption("⚪ 不可评维度（数据源取不到，**已从总分分母剔除**，"
                   "**不等于**该维度得 0 分）：" + _unavail_text())


def render_chart(chart: dict):
    try:
        if chart.get("kind") == "kline":
            render_kline_chart(chart)
        elif chart.get("kind") == "fund5":
            render_fund5_chart(chart)
    except Exception:
        pass  # 图表渲染失败不影响文字回答


def render_session_list():
    """侧边栏会话列表：切换 / 删除 / 新建"""
    st.markdown("### 🗂️ 历史会话")
    if st.button("＋ 新建对话", width="stretch"):
        _switch_to(None)
    try:
        sessions = chat_store.list_sessions(limit=15)
    except Exception:
        sessions = []
    current = st.session_state.get("chat_session_id")
    if not sessions:
        st.caption("暂无历史会话；发出第一条消息后自动生成。")
        return
    for s in sessions[:10]:
        sid = s.get("session_id") or ""
        is_cur = sid == current
        title = chat_store.session_label(s)
        updated = (s.get("updated_at") or "")[5:16]
        left, right = st.columns([0.8, 0.2])
        with left:
            if st.button(f"{'🔵' if is_cur else '⚪'} {title}",
                         key=f"cs_{sid}", width="stretch",
                         help=f"{s.get('message_count', 0)} 条 · 更新于 {updated}"):
                _switch_to(sid)
        with right:
            if st.button("🗑", key=f"cd_{sid}", width="stretch", help="删除该会话"):
                chat_store.delete_session(sid)
                _switch_to(None if is_cur else current)
        st.caption(f"<sub>{s.get('message_count', 0)} 条 · {updated}</sub>",
                   unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("### 🎭 分析视角")
        personas = list(PERSONAS.keys())
        saved = st.session_state.get("persona")
        idx = personas.index(saved) if saved in personas else 0
        persona = st.selectbox("人格化快评", personas, index=idx,
                               label_visibility="collapsed")
        st.session_state.persona = persona
        st.divider()
        render_session_list()
        st.divider()
        st.markdown("### 💬 当前会话")
        if st.button("🧹 清空当前对话", width="stretch"):
            # 以前只清内存：刷新后消息又从库里回来了。这里连库一起清。
            chat_store.clear_session(st.session_state.get("chat_session_id"))
            st.session_state.chat_history = []
            st.session_state.chat_usage = {"prompt": 0, "completion": 0}
            st.rerun()
        usage = st.session_state.get("chat_usage") or {}
        if usage:
            st.caption(f"本次会话 tokens：入 {usage.get('prompt', 0):,} / "
                       f"出 {usage.get('completion', 0):,}"
                       f"（会话 {st.session_state.get('chat_session_id', '—')}）")
        st.caption("更细的调用耗时/失败原因见「系统配置」页的 LLM 调用链面板。")
        st.markdown("### 💡 试试这些")
        for ex in EXAMPLES:
            if st.button(ex, key=f"ex_{ex}", width="stretch"):
                st.session_state.pending_input = ex
                st.rerun()
        st.caption(f"模型：{_chat_model_label()}｜涉及实时数据的回答由工具查询后生成，"
                   "仅供参考，不构成投资建议。")
    return persona


def run_agent(user_msg: str, persona: str):
    """执行 Agent 并渐进渲染工具过程、图表与最终回答；对话落库（会话维 + 消息维）"""
    with st.chat_message("assistant"):
        status = st.status("思考中...", expanded=True)
        answer = None
        charts = []
        agent = ChatAgent(persona=persona)
        try:
            for event, *payload in agent.reply(user_msg, st.session_state.chat_history):
                if event == "tool_start":
                    name, args = payload
                    status.update(label=f"🔧 调用工具 {name} {args if args else ''}",
                                  state="running")
                elif event == "tool_end":
                    name, summary = payload
                    status.write(f"✅ {name} — {summary}")
                elif event == "chart":
                    charts.append(payload[0])
                elif event == "answer":
                    answer = payload[0]
                elif event == "error":
                    answer = f"⚠️ {payload[0]}"
            if answer is None:
                answer = "（未获得回答，请重试）"
            status.update(label="完成", state="complete", expanded=False)
        except Exception as e:
            status.update(label="出错", state="error")
            answer = f"⚠️ 助手异常：{e}"
        st.markdown(answer)
        for chart in charts:
            render_chart(chart)
        # Phase 6.2：token 用量当场可见（原先只写进 chat_logs 表，页面上看不到）
        usage = getattr(agent, "usage_total", {}) or {}
        if usage.get("prompt") or usage.get("completion"):
            total = st.session_state.setdefault("chat_usage",
                                                {"prompt": 0, "completion": 0})
            total["prompt"] += usage.get("prompt") or 0
            total["completion"] += usage.get("completion") or 0
            st.caption(f"🧠 本轮 tokens：入 {usage.get('prompt', 0):,} / "
                       f"出 {usage.get('completion', 0):,}｜本次会话累计："
                       f"入 {total['prompt']:,} / 出 {total['completion']:,}")
    # 对话落库（失败不影响交互）：会话维 + 一问一答两行消息
    sid = st.session_state.get("chat_session_id")
    try:
        chat_store.save_turn(
            sid, user_msg, answer or "",
            usage=getattr(agent, "usage_total", {}) or {},
            tools_used=getattr(agent, "tools_used", []),
            persona=persona, model=getattr(agent, "model", None), source="ui")
    except Exception:
        pass
    return answer


def main():
    _sync_session_from_url()
    persona = render_sidebar()

    # 渲染历史
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 示例按钮注入的输入
    pending = st.session_state.pop("pending_input", None)
    user_input = st.chat_input("输入你的问题，如：茅台资金面怎么样？")
    user_msg = pending or user_input
    if not user_msg:
        return

    # 会话 id 在第一条消息时才创建，并立即写进 URL —— 之后刷新页面仍是同一会话
    if not st.session_state.get("chat_session_id"):
        st.session_state.chat_session_id = chat_store.start_session(
            persona=persona, source="ui")
        _write_query_sid(st.session_state.chat_session_id)

    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    answer = run_agent(user_msg, persona)
    st.session_state.chat_history.append({"role": "assistant", "content": answer or ""})


main()
