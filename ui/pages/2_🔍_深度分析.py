"""深度分析页面 — 复用 LangGraph 多Agent编排（多轮辩论增强版）"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from core import analysis_runner
from core.analysis_history import snapshot_from_state
import core.ui_runs as ui_runs
import core.ui_tasks as ui_tasks
from core import report_render
from ui.components.fund_flow_chart import render_fund_flow_section

st.set_page_config(page_title="深度分析", page_icon="🔍", layout="wide")
st.title("🔍 深度分析")


# ========== 结果快照与回填渲染（P1：刷新后不再回到空白页） ==========
# 快照逻辑已抽到 core.analysis_history.snapshot_from_state（分析历史入库复用同一份）

def _snapshot(final_state: dict) -> dict:
    """委托 core.analysis_history.snapshot_from_state（保持页面内调用点不变）"""
    return snapshot_from_state(final_state)


def _offer_markdown_export(payload: dict, key_suffix: str = ""):
    """「导出 Markdown」——把**当前这一份** payload 按需渲染。

    为什么不做成"跑完自动写一个 md 文件"（2026-10-09 决定）：
    那就又变成两份并行产物，早晚漂移。实测 10-08：磁盘上的 md 渲染的是 22:06 那次运行，
    而 UI 里显示的是 22:38 那次（后者还是旧代码产出的）——复核的人和看页面的人
    手里的根本不是同一份报告。⇒ **json 是唯一事实来源，md 按需导出**。
    """
    if not payload:
        return
    st.download_button(
        "⬇️ 导出 Markdown（含可核实性审计）",
        data=report_render.render_markdown(payload),
        file_name=report_render.suggest_filename(payload),
        mime="text/markdown",
        help="导出的是**当前页面上这一份**结果，与本地 JSON 同源，不会出现版本不一致。",
        key=f"dl_report{key_suffix}_{payload.get('stock_code','')}_"
            f"{payload.get('analysis_date','')}",
    )


def _run_report_qa(payload: dict, analysis_id, question: str, state_key: str):
    """跑一轮"就这份报告提问"：把**当前这一份** payload 注入会话上下文（方案 B）。

    为什么用注入而不是让模型自己调工具取报告：
    ① 模型不必猜"用户说的是哪一份"（同一只股票同一天可能有好几条记录）；
    ② 不会读到错版本（注入的就是页面上正在看的这一份）；
    ③ 不消耗工具轮次。工具仍可用 —— 需要"最新行情/新闻"时该调还是调，
       但上下文里要求它**区分**报告口径与当前数据。
    """
    from core.chat_agent import ChatAgent

    history = st.session_state.setdefault(state_key, [])
    metas = st.session_state.setdefault(f"{state_key}_meta", {})
    agent = ChatAgent(context_blocks=[
        report_render.build_chat_context(payload, analysis_id=analysis_id)])
    answer, err, tools = None, None, []
    with st.spinner("正在阅读这份报告并作答…"):
        try:
            for event, *data in agent.reply(question, history):
                if event == "tool_start":
                    tools.append(data[0])
                elif event == "answer":
                    answer = data[0]
                elif event == "error":
                    err = data[0]
        except Exception as e:                                # noqa: BLE001
            err = f"{type(e).__name__}: {e}"

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant",
                    "content": answer or f"⚠️ 没能得到回答（{err or '未知原因'}）"})
    if tools:
        metas[len(history) - 1] = "、".join(dict.fromkeys(tools))


def _ask_about_report(payload: dict, context_key: str, analysis_id=None):
    """「就这份报告提问」区块（方案 B：把 payload 直接注入会话）。

    与「AI 助手」页的会话完全独立（各自 st.session_state key），
    避免把"围绕某份报告的追问"混进日常对话历史。
    """
    if not payload:
        return
    state_key = f"report_qa_{context_key}"
    q_key, btn_key = f"qa_q_{context_key}", f"qa_btn_{context_key}"
    with st.expander("💬 就这份报告提问（已注入本报告上下文，不必再说明是哪一份）"):
        st.caption("这是**独立**的问答，不会写进「AI 助手」页的会话历史。"
                   "回答基于上面这份报告的原文；需要最新行情/新闻时它会另调工具，"
                   "并区分「报告口径」与「当前数据」。")
        history = st.session_state.get(state_key) or []
        metas = st.session_state.get(f"{state_key}_meta") or {}
        for i, msg in enumerate(history):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if i in metas:
                    st.caption(f"🔧 本轮另调用了：{metas[i]}")
        if history and st.button("🧹 清空这段问答", key=f"qa_clear_{context_key}"):
            st.session_state[state_key] = []
            st.session_state[f"{state_key}_meta"] = {}
            st.rerun()
        question = st.text_input("你的问题（回车或点「提问」）", key=q_key,
                                 placeholder="例如：为什么舆情乐观却短期看空？")
        if st.button("提问", key=btn_key, type="primary"):
            if question.strip():
                _run_report_qa(payload, analysis_id, question.strip(), state_key)
                st.rerun()
            else:
                st.warning("请先输入问题。")
                # ⚠️ 这里**不能**去清空 `st.session_state[q_key]`：
                #    该 key 的 widget 本轮已经创建过，再写就抛 StreamlitAPIException
                #    （本项目已在持仓页踩过这个坑）。要清空只能走"中间态 + 下次渲染前落盘"。


def _render_data_scope(d: dict):
    """报告头部：数据口径 + 缺失项 + 可核实性告警（v3，2026-09-28）。

    必要性：旧版报告**通篇没有一个日期**，用户无从判断"这是哪天的数据"；
    而"输入其实是空的"也只体现在报告里没提——两者都从报告正文里看不出来。
    现在把口径和缺失直接摆到最前面，让人一眼能看出这份报告可信到什么程度。
    """
    parts = []
    if d.get("analysis_date"):
        parts.append(f"分析日期 {d['analysis_date']}")
    parts.append(f"数据截止 {d.get('data_through') or '未知'}")
    st.caption("🗓️ " + " ｜ ".join(parts))

    avail = d.get("data_availability") or {}
    solid = [k for k, v in avail.items() if v is True]
    missing = [k for k, v in avail.items() if v is False]
    if solid:
        st.caption("✅ 已取到：" + "、".join(solid))
    if missing:
        st.warning("⚠️ 本次未取到：" + "、".join(missing)
                   + " —— 相关维度在报告里应当是「数据缺失」，若出现结论即为无依据。")

    for w in (d.get("audit_warnings") or []):
        st.warning(f"🔎 可核实性告警：{w}")


def render_analysis_snapshot(d: dict):
    """静态渲染一份已保存的分析结果"""
    if not d:
        st.info("没有可展示的分析结果")
        return
    name = d.get("stock_name") or d.get("stock_code") or ""
    price = d.get("current_price")
    st.info(f"📊 {name}（{d.get('stock_code', '')}） | 现价: {price if price is not None else 'N/A'}")
    _render_data_scope(d)
    if d.get("fatal_errors"):
        st.error("分析被致命错误中断：" + "；".join(d["fatal_errors"]))

    flow = d.get("fund_flow_data") or {}
    if flow.get("data_success"):
        with st.expander("💰 资金流向", expanded=False):
            render_fund_flow_section(flow)
    with st.expander("🔬 技术面分析报告", expanded=False):
        st.markdown(d.get("tech_report") or "—")
    with st.expander("📋 基本面分析报告", expanded=False):
        st.markdown(d.get("fundamental_report") or "—")
    with st.expander("📰 舆情分析报告", expanded=False):
        st.markdown(d.get("news_report") or "—")

    bull_hist = d.get("bull_history") or []
    bear_hist = d.get("bear_history") or []
    if bull_hist or bear_hist:
        st.subheader("⚔️ 多空辩论")
        for r in range(max(len(bull_hist), len(bear_hist))):
            st.markdown(f"**第 {r + 1} 轮**")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("🐂 多头")
                st.markdown(bull_hist[r] if r < len(bull_hist) else "—")
            with c2:
                st.caption("🐻 空头")
                st.markdown(bear_hist[r] if r < len(bear_hist) else "—")

    if d.get("research_summary"):
        with st.expander("🧭 研究汇总", expanded=False):
            st.markdown(d["research_summary"])

    st.subheader("🏆 最终投资决策")
    st.markdown(d.get("final_decision") or "—")
    if d.get("trade_plan"):
        _tp = d["trade_plan"]
        _dir = str(_tp.get("direction") or "")
        # 三档建议状态（新产物由后端派生；旧产物按方向推断，保持前后一致）
        _status = str(_tp.get("advice_status") or "") or (
            "驳回" if "看空" in _dir else ("可执行" if "看多" in _dir else "等待"))
        _icon = {"可执行": "🟢", "驳回": "🔴", "等待": "⚪"}.get(_status, "⚪")
        st.markdown(
            f"{_icon} **交易方向：{_dir or '未提供'}**　｜　**投资建议：{_status}**"
            + (f"　｜　仓位：{_tp.get('position_size')}"
               if _tp.get("position_size") else ""))
        if _status == "驳回":
            st.error("🔴 **投资建议已驳回**：判断为看空，**不建议买入**。\n\n"
                     "⚠️ 这不等于「可以做空」 —— A股 个人投资者没有融券卖出的通路"
                     "（信用账户 + 50万门槛 + 标的在两融名单且有券源），"
                     "「看空」只能落成「不参与／已持有则减仓」。")
        elif _status == "等待":
            st.info("⚪ **等待**：暂不具备入场条件。"
                    "**这不等于做空** —— 本次不建立任何仓位，"
                    "满足下面的入场条件后才重新考虑买入。")
        _conds = _tp.get("entry_conditions")
        if _status != "可执行":
            if isinstance(_conds, (list, tuple)) and _conds:
                st.markdown("**入场条件（满足后才重新考虑买入）**：")
                st.markdown("\n".join(f"{i}. {c}" for i, c in enumerate(_conds, 1)))
            else:
                st.warning("⚠️ 未给出「入场条件」—— 驳回/等待必须说明"
                           "「什么信号出现才重新考虑入场」，否则这条建议无法执行。")
        with st.expander("📋 交易计划（完整字段）", expanded=False):
            st.markdown(report_render.trade_plan_markdown(_tp))
    if d.get("risk_assessment"):
        with st.expander("🛡️ 风控意见", expanded=False):
            st.markdown(d["risk_assessment"])
    if d.get("errors"):
        with st.expander(f"⚠️ 降级/失败记录（{len(d['errors'])} 条）", expanded=False):
            for e in d["errors"]:
                st.caption(f"- {e}")

# ================================================================
# 📋 分析队列（来自选股页「将 TOP5 加入分析队列」）
# ================================================================
# ⚠️ 必须放在**控制栏之前**：这些按钮要写 `analysis_code_input`，而 Streamlit 不允许
#    在 widget 已实例化之后修改它的 key（会抛 StreamlitAPIException）。放前面即可安全写入。
_queue = st.session_state.get("analysis_queue") or []
if _queue:
    st.info("📋 **分析队列**（来自选股页）：" + "、".join(_queue))
    _qcols = st.columns(len(_queue) + 1)
    for _qi, _qcode in enumerate(_queue):
        if _qcols[_qi].button(_qcode, key=f"queue_go_{_qcode}"):
            st.session_state["analysis_code_input"] = _qcode
            st.rerun()
    if _qcols[-1].button("🗑 清空队列", key="queue_clear"):
        st.session_state["analysis_queue"] = []
        st.rerun()


# 分析控制栏
col1, col2 = st.columns([3, 1])
with col1:
    # ⚠️ `prefill_code`：热点新闻页「深度分析」按钮通过 session_state 传过来的
    #    预填代码。必须在 widget 创建**之前**写入 `session_state[key]`——
    #    写在 value= 参数上时，key 一旦有了自己的状态，后续 prefill 会被静默忽略
    #    （Streamlit 的规则：key 已存在时 value 参数不再生效）。
    _prefill = st.session_state.pop("prefill_code", None)
    if _prefill:
        st.session_state["analysis_code_input"] = _prefill
    stock_code = st.text_input("股票代码", value="000001",
                               placeholder="输入6位A股代码", max_chars=6,
                               key="analysis_code_input")
with col2:
    st.write("")
    analyze_btn = st.button("🔍 开始分析", type="primary", width="stretch")

_valid_input = bool(stock_code) and len(stock_code) == 6 and stock_code.isdigit()
TASK = f"analysis_{stock_code}" if _valid_input else ""


def _analysis_job(code: str):
    """构造在工作线程里执行的深度分析任务（**委托 `core.analysis_runner`**）。

    ⚠️ 线程里**不能调用任何 Streamlit API**（没有 ScriptRunContext）：
    要报进度就调用 `progress()`，它只写文件；页面每 2 秒读一次，
    因此刷新/关页面都不会打断分析（详见 core/ui_tasks.py）。

    ⚠️ 为什么改成委托（2026-10-15）：AI 交易联赛要在**调度线程**里跑同一套流程
    （建 state → stream → 按 reducer 语义归并 → 落库）。两处各写一份必然分叉
    ⇒ 流程下沉到 `core.analysis_runner.run_analysis`，本页只负责**报进度 + 存快照**。
    """
    def job(progress):
        def on_step(node_name, final_state, step_idx):
            # 每步既报进度、也带上"到目前为止的快照"：页面据此渐进式渲染，
            # 刷新后重新连上还能看到已经跑完的那几个节点。
            progress({
                "percent": min(step_idx / analysis_runner.NODE_STEPS, 1.0) * 100,
                "message": f"步骤 {step_idx}: {node_name}",
                "node": node_name,
                "snapshot": _snapshot(final_state),
            })

        res = analysis_runner.run_analysis(code, source="ui", on_step=on_step)
        if not res["ok"]:
            # 失败样本已由执行器留痕（save_failure）；这里抛给 ui_tasks 记 error 状态
            raise RuntimeError(res["error"])
        ui_runs.save_result(f"analysis_{code}", res["snapshot"])

    return job


# 进度轮询间隔（秒），见选股页同名常量说明
_POLL_SECONDS = float(os.environ.get("BIGA_UI_POLL_SECONDS") or 2.0)


@st.fragment(run_every=_POLL_SECONDS)
def _live_analysis(task: str):
    """轮询后台进度：画进度条 + 渲染已完成的节点。任务结束后整页重跑。"""
    prog = ui_tasks.read_progress(task) or {}
    try:
        pct = max(0.0, min(1.0, float(prog.get("percent") or 0) / 100.0))
    except (TypeError, ValueError):
        pct = 0.0
    try:
        elapsed = float(prog.get("elapsed") or 0)
    except (TypeError, ValueError):
        elapsed = 0.0
    st.progress(pct, text=f"{prog.get('message') or '准备中…'}（已用 {elapsed:.0f}s）")
    if prog.get("snapshot"):
        render_analysis_snapshot(prog["snapshot"])
    if not ui_tasks.is_running(task):
        st.rerun(scope="app")


# ================================================================
# 同日同票去重（2026-10-10）
# ================================================================
# 为什么需要：实测有股票在**同一天被分析 4 次**，结论还互相矛盾（观望/看空/观望）——
# 既浪费 LLM 成本，又污染胜率统计（同一只票一天贡献 4 个样本，且方向相反）。
# 这里在**入口**提示并提供「看已有结果」的捷径；要不要重跑由用户决定（不硬拦）。
def _todays_analyses(code: str) -> list:
    """今日已分析该股的记录（时间倒序）。读库失败返回 []（提示不该阻断分析）。"""
    try:
        from datetime import datetime as _dt
        from db import store as _store
        today = _dt.now().strftime("%Y-%m-%d")
        rows = _store.list_analyses(stock_code=code, limit=30)
        return [r for r in rows if str(r.get("analysis_date") or "")[:10] == today]
    except Exception:                                  # noqa: BLE001
        return []


_dup_today = _todays_analyses(stock_code) if _valid_input else []
_redo = False
if _dup_today:
    _latest = _dup_today[0]
    st.info(
        f"ℹ️ **今日已分析过 {stock_code}**（{len(_dup_today)} 条，最近 "
        f"{(_latest.get('created_at') or '')[:16]}，结论：**{_latest.get('decision_label')}**）。"
        "同日同票重复分析会产出**结论可能矛盾**的记录，并让同一只票在胜率统计里被重复计数。")
    _dc1, _dc2, _dc3 = st.columns([1, 1, 3])
    if _dc1.button("📄 查看今日已有结果", key=f"dup_view_{stock_code}"):
        st.session_state["_show_existing_analysis"] = _latest["id"]
    _redo = _dc2.button("♻️ 仍要重新分析", key=f"dup_redo_{stock_code}")

_show_id = st.session_state.pop("_show_existing_analysis", None)
if _show_id:
    _full = None
    try:
        from db import store as _store
        _full = _store.get_analysis(_show_id)
    except Exception as _e:                            # noqa: BLE001
        st.warning(f"读取历史分析失败：{type(_e).__name__}: {_e}")
    if _full:
        st.caption(f"以下是今日已存的第 {_show_id} 号分析结果（**未重新调用模型**）")
        render_analysis_snapshot(_full.get("payload") or {})


if _redo:
    if not ui_tasks.start(TASK, _analysis_job(stock_code), {"stock_code": stock_code}):
        st.error(f"⚠️ {stock_code} 已有分析任务在运行，本次点击已忽略——"
                 "等它跑完再试（心跳停止 3 分钟后会自动清理）。")
    else:
        st.rerun()
elif analyze_btn:
    if not _valid_input:
        st.error("请输入6位A股代码")
    elif _dup_today:
        # 不静默重跑：明确告诉用户"为什么没跑"（上方的重跑按钮才是确认路径）
        st.warning("今日已分析过该股，本次未重复执行 —— 如需重跑请点上方「♻️ 仍要重新分析」；"
                   "也可以先点「📄 查看今日已有结果」复用已存的报告。")
    elif not ui_tasks.start(TASK, _analysis_job(stock_code), {"stock_code": stock_code}):
        st.error(f"⚠️ {stock_code} 已有分析任务在运行，本次点击已忽略——"
                 "等它跑完再试（心跳停止 3 分钟后会自动清理）。")
    else:
        st.rerun()

if _valid_input and ui_tasks.is_running(TASK):
    record = ui_runs.get_running_any(TASK) or {}
    st.warning(f"⏳ {stock_code} 的分析正在**后台线程**里运行（{record.get('started_at', '')} 启动）。"
               "刷新页面、关掉标签页都不影响它；进度条每 2 秒读一次进度文件，"
               "已完成的节点会陆续出现。")
    _live_analysis(TASK)
    if st.button("✖️ 取消本次分析"):
        ui_tasks.request_cancel(TASK)
        st.info("已请求取消：会在当前这个 Agent 节点跑完后停止。")
else:
    if _valid_input:
        ui_runs.clear_dead_running()
        prog = ui_tasks.read_progress(TASK) or {}
        status = prog.get("status")
        if status == "cancelled":
            st.info(f"ℹ️ {stock_code} 上一次分析已取消，结果未更新。")
        elif status == "error":
            st.error(f"❌ {stock_code} 上一次分析失败：{prog.get('message', '')}")
        elif status in ("running", "starting"):
            st.warning(f"⚠️ {stock_code} 上一次分析没有正常结束（进程被关闭或崩溃），任务已停止。")
            ui_tasks.clear_progress(TASK)
        saved = ui_runs.load_result(TASK)
        if saved:
            st.caption(f"📌 下面是上次保存的分析结果（{ui_runs.age_text(saved)} 前生成）。"
                       "刷新、关页面、换标签页都不会丢；点「开始分析」会重新执行并覆盖它。")
            _payload = saved.get("payload") or {}
            _offer_markdown_export(_payload)
            render_analysis_snapshot(_payload)
            _ask_about_report(_payload, context_key=f"{_payload.get('stock_code', '')}_"
                                                     f"{_payload.get('analysis_date', '')}")
        elif status not in ("cancelled", "error", "running", "starting"):
            st.info("👆 输入A股代码并点击「开始分析」")
    else:
        st.info("👆 输入A股代码并点击「开始分析」")


# ================================================================
# 📚 分析历史（含 T+N 回看闭环）
# 此前结果只存 output/ui_runs/analysis_{code}.json（每只股票仅最后一份且覆盖），
# 现在每次分析都入库（analyses 表），可回放任意历史分析、核对 T+5/T+20 实际收益。
# ================================================================

_STATUS_BADGE = {None: "⏳ 待回填", "partial": "🕐 部分回填", "done": "✅ 已核对",
                 "failed": "❌ 超龄失败", "skip": "⏭️ 不参与"}


def _fmt_ret(v):
    if v is None:
        return "—"
    return f"{v:+.1%}"


with st.expander("📚 分析历史（含 T+N 回看）", expanded=False):
    try:
        from db import store as db_store
        from core.analysis_history import backfill_analyses, backfill_progress

        stats = db_store.analysis_stats(recent_days=30)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("近30天分析", stats["total"])
        c2.metric("看多5日胜率",
                  f"{stats['bull_win_rate']:.0%}" if stats["bull_win_rate"] is not None else "—",
                  f"{stats['bull_samples']} 样本")
        c3.metric("看空5日胜率",
                  f"{stats['bear_win_rate']:.0%}" if stats['bear_win_rate'] is not None else "—",
                  f"{stats['bear_samples']} 样本")
        c4.metric("待回填", stats["pending_backfill"])

        # ⚠️ 光有「待回填 N」看不出是**在等**还是**坏了**（2026-10-09 用户报"一次回填都没有"）。
        #    这里把结论摆在最显眼处：还差几个交易日、预计哪天可回填。
        #    `backfill_progress()` 完全离线（只用交易日历，不拉 K 线），每次渲染都能调。
        try:
            _prog = backfill_progress(limit=100)
        except Exception as _e:                       # noqa: BLE001
            _prog = None
            st.caption(f"（回填进度计算失败：{type(_e).__name__}: {_e}）")
        if _prog is not None:
            if _prog["pending"] == 0:
                st.caption("✅ 没有待回填记录。")
            else:
                st.caption("⏳ " + _prog["summary"])

        rows = db_store.list_analyses(limit=100)
        if not rows:
            st.info("还没有分析历史——完成第一次「开始分析」后，这里会记录每一次分析并跟踪其 T+5/T+20 实际收益。")
        else:
            import pandas as pd
            df = pd.DataFrame([{
                "ID": r["id"],
                "日期": r["analysis_date"],
                "代码": r["stock_code"],
                "名称": r["stock_name"],
                "决策": r["decision_label"],
                "分析价": r["price_at_analysis"],
                "T+5收益": _fmt_ret(r["ret_5d"]),
                "T+20收益": _fmt_ret(r["ret_20d"]),
                "回填状态": _STATUS_BADGE.get(r["backfill_status"], r["backfill_status"]),
                "来源": r["source"],
            } for r in rows])
            st.dataframe(df, width="stretch", hide_index=True)

            st.caption("💡 回填说明：每日收盘后 15:35 自动补 T+5/T+20 收益（交易日）；"
                       "「超龄失败」= 分析日 +60 天仍缺数据；「不参与」= 分析失败样本，保留只为避免胜率虚高。")

            if _prog is not None and _prog["rows"]:
                h5, h20 = _prog["horizons"]
                with st.expander("⏳ 这些记录为什么还没回填（还要等多久）", expanded=False):
                    st.caption(
                        f"T+{h5} 需要 **{h5 + 1} 根** K 线（分析日算第 1 根）、"
                        f"T+{h20} 需要 **{h20 + 1} 根**；周末与法定节假日不算交易日。"
                        f"基准取到 **{_prog['last_trading_day']}** 及以前"
                        "（当日 K 线通常要收盘后或次日才拿得到，故不计入）。")
                    st.dataframe([{
                        "ID": x["id"], "代码": x["code"], "分析日": x["analysis_date"],
                        f"T+{h5} 可回填日": x["ready"].get(h5),
                        f"还差(交易日)": x["missing"].get(h5),
                        f"T+{h20} 可回填日": x["ready"].get(h20),
                        "阶段": x["stage"],
                    } for x in _prog["rows"][:30]], width="stretch", hide_index=True)
                    st.caption("说明：**「等 T+5/T+20」是正常等待，不是故障**——"
                               "只有当上面出现「取数不可用」才是需要查数据源。"
                               "回填任务每次会打印的 `not_ready` / `unavailable` 两个计数正是区分这两者。")

            # ---- 历史回放 ----
            options = {f"{r['analysis_date']} {r['stock_code']} {r['stock_name']}（{r['decision_label']}）": r["id"]
                       for r in rows}
            if options:
                picked = st.selectbox("回放历史分析", list(options.keys()), index=None,
                                      placeholder="选择一条历史记录查看完整报告…")
                if picked:
                    full = db_store.get_analysis(options[picked])
                    if full:
                        st.caption(f"🗓️ {full['analysis_date']} | 模型: {full['model']} | "
                                   f"择时: {full.get('timing_regime') or '—'} | "
                                   f"T+5: {_fmt_ret(full['ret_5d'])} | T+20: {_fmt_ret(full['ret_20d'])}")
                        _hp = full.get("payload") or {}
                        _offer_markdown_export(_hp, key_suffix=f"_h{full['id']}")
                        render_analysis_snapshot(_hp)
                        _ask_about_report(_hp, context_key=f"h{full['id']}",
                                          analysis_id=full["id"])

            # ---- 复盘归因（2026-10-20）----
            # 回填只回答"对不对"；这里回答"为什么不准、下一批该改什么"。
            # ⚠️ 对错由代码判定（`mechanical_verdict`），模型只在**候选原因**里选
            #    —— 让模型判对错，会把"我错了"写在本来正确的样本上。
            with st.expander("🧠 复盘：为什么不准 / 该改什么", expanded=False):
                try:
                    from core import analysis_review as _review
                    # ⚠️ 2026-10-20 审查 B3：回填的复盘可能**整批失败**（新旧模块混用时
                    #    `upsert_analysis_review` 不存在），而原先只有一条 warning —— 页面
                    #    完全看不出来。当天真的发生过"回填成功、11 条复盘全废"。
                    from db import store as _st_bf
                    _last_bf = _st_bf.get_last_run("job:backfill_analysis")
                    if _last_bf:
                        _bf = _last_bf.get("summary") or {}
                        _re_n = int(_bf.get("review_errors") or 0)
                        if _re_n:
                            st.warning(
                                f"最近一次回填（{_last_bf.get('created_at')}）有 "
                                f"**{_re_n} 条复盘失败**——下面的统计只含成功的那部分。"
                                f"　样例：{'; '.join(str(x) for x in (_bf.get('review_error_sample') or [])[:2])}")
                        elif _bf.get("ok") is False:
                            st.warning(f"最近一次回填（{_last_bf.get('created_at')}）失败："
                                       f"{_bf.get('error')}")
                    # ⚠️ 两个窗口必须分开看：T+5 的"判对"和 T+20 的"判对"不是一回事。
                    #    "T+5 判对、T+20 判错"恰恰是 timing_wrong 的证据，混口径就看不见。
                    _hz = st.radio("复盘窗口", ("完整（T+20）", "早期（T+5）"),
                                   horizontal=True, key="_review_hz")
                    _hz_n = 20 if _hz.startswith("完整") else 5
                    _agg = _review.aggregate_recent(days=180, horizon=_hz_n)
                    if not _agg.get("n"):
                        st.caption(f"还没有 T+{_hz_n} 窗口的可复盘样本。"
                                   + ("回填（15:35）会自动为到期样本产出**机械复盘**，零 LLM 成本；"
                                      "今天只有 T+5 数据的，请切到「早期（T+5）」。"
                                      if _hz_n == 20 else
                                      "需要分析日后至少走完 5 个交易日。"))
                    else:
                        _verdict = _agg.get("by_verdict") or {}
                        _c1, _c2, _c3 = st.columns(3)
                        _c1.metric("已复盘", _agg["n"])
                        _c2.metric("判错", _verdict.get("wrong", 0),
                                   f"{_verdict.get('wrong', 0) / _agg['n']:.0%}"
                                   if _agg["n"] else None)
                        _top = (_agg.get("by_cause") or [{}])[0]
                        _c3.metric("首要原因", str(_top.get("label") or "—")[:14],
                                   f"{_top.get('count', 0)} 条" if _top else None)
                        st.code(_agg["text"], language=None)

                        _revs = db_store.list_analysis_reviews(
                            limit=200, since=_agg["since"], horizon=_hz_n)
                        if _revs:
                            st.dataframe([{
                                "分析日": r["analysis_date"],
                                "代码": r["stock_code"],
                                "名称": r["stock_name"],
                                "决策": r["decision_label"],
                                "结果": {"correct": "✅ 判对",
                                         "partially_correct": "➖ 方向对但幅度小",
                                         "wrong": "❌ 判错",
                                         "unclear": "❔ 没走出来",
                                         "opportunity_missed": "😶 观望后大涨",
                                         "no_position": "— 未预测"}.get(
                                             r["verdict"], r["verdict"] or "—"),
                                "主因": (_review.CAUSE_CODES.get(r["primary_cause"])
                                         or r["primary_cause"] or "—")[:26],
                                "模型归因": ((r.get("llm_review") or {}).get("verdict_reason")
                                             or "（仅机械归因）")[:60],
                            } for r in _revs[:50]], width="stretch", hide_index=True)
                            st.caption("「仅机械归因」= 只由代码按判据推导，还没让模型解释。"
                                       "机械归因不含 ATR（回填时只拉了分析日前 7 天），"
                                       "补全 ATR 请点下面的按钮。")
                except Exception as _re:                  # noqa: BLE001
                    st.caption(f"复盘模块暂不可用：{type(_re).__name__}: {_re}")

                if st.button("🧠 让模型复盘最近 10 条（补 ATR + 受限归因 + 证据校验）",
                             help="每次约 10 次 LLM 调用（≈¥0.05 空闲价）。"
                                  "模型只能从代码判定的候选原因里选，"
                                  "证据字段与改进项都会被校验，编造的一律剔除。"):
                    with st.spinner("正在归因…（逐条调用，约 30~60 秒）"):
                        _rr = _review.review_pending(limit=10, with_llm=True)
                    st.session_state["_review_msg"] = (
                        f"复盘完成：处理 {_rr['checked']} 条，成功 {_rr['reviewed']} 条"
                        f"（模型归因 {_rr['llm_ok']}，失败 {_rr['llm_error']}，跳过 {_rr['skipped']}）")
                    if _rr.get("errors"):
                        st.session_state["_review_errs"] = _rr["errors"][:10]
                    st.rerun(scope="app")
                # ⚠️ 提示存 session_state：`st.rerun()` 会把紧邻的 success 抹掉
                if st.session_state.get("_review_msg"):
                    st.success(st.session_state.pop("_review_msg"))
                if st.session_state.get("_review_errs"):
                    with st.expander("⚠️ 本次复盘的部分失败"):
                        for _e in st.session_state.pop("_review_errs"):
                            st.caption(f"- {_e}")

            # ---- 手动回填 ----
            if st.button("🔄 立即回填 T+N 收益（含重试超龄失败）",
                         help="零 LLM 成本，只拉 K 线。每日 15:35 也会自动执行。"):
                with st.spinner("正在回填…（逐只拉取 K 线，可能需要一点时间）"):
                    result = backfill_analyses(limit=100, include_failed=True)
                if result.get("locked"):
                    # 撞上另一个进程的回填（调度 15:35 或另一个页面）：
                    # 此时 checked=0 不是"没有需要回填的"，必须说清楚，否则用户以为坏了
                    st.info("⏳ 已有回填任务在运行中（每日 15:35 的调度，或另一个页面），"
                            "本次已跳过——同一批记录没必要重复拉 K 线。稍后再看结果即可。")
                else:
                    st.success(f"回填完成：核对 {result['checked']} 条，更新 {result['updated']} 条"
                               f"（done={result['done']}, partial={result['partial']}, "
                               f"failed={result['failed']}）"
                               + (f"；复盘 {result['reviewed']} 条" if result.get("reviewed") else ""))
                    # ⚠️ 把 skipped 拆开报：否则"更新 0 条"看着像故障，实际多半是正常等待。
                    if result.get("not_ready"):
                        st.info(f"⏳ 其中 **{result['not_ready']}** 条属于「**窗口未走完**」"
                                "（正常等待，不是故障）：T+5 需要分析日之后再有 5 个交易日。"
                                "展开上面的「这些记录为什么还没回填」可看到各自还差几天。")
                    if result.get("unavailable"):
                        st.warning(f"⚠️ 其中 **{result['unavailable']}** 条属于「**取数不可用**」"
                                   "（这个才需要查数据源）——详见下方错误列表。")
                if result["errors"]:
                    with st.expander(f"⚠️ 数据源异常 {len(result['errors'])} 条（明日自动重试）"):
                        for e in result["errors"][:20]:
                            st.caption(f"- {e}")
                st.rerun(scope="app")
    except Exception as e:
        st.warning(f"分析历史暂不可用：{e}")