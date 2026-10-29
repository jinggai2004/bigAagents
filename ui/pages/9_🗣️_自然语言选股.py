"""自然语言选股页面（Phase 3）

三步不变（解析 → 确认/编辑 → 执行），但**执行层已在 2026-10-07 搬到本地**：

1. **解析**：一句话 → 白名单条件。规则优先（免费、可复现、离线可用）；
   规则解析不出条件时才请模型兜底，且输出必须过同一套白名单校验。
2. **确认/编辑**：条件、没听懂的片段全部摆出来，用户可去掉不要的条件。
3. **执行**（**本次改动最大**）：
       L1 本地全市场快照粗筛（PE/总市值/换手率/成交额/涨跌幅，毫秒级）
       L2 逐票因子精算（K线/资金流/舆情/评分）
       L3 本地逐条核对（每条结论都带实际值）
   问财链路已按 D1 决策移除 —— 理由见 `output/NL选股改进方案v2_数据层与使用边界.md` §2。

诚实性约定（三条，都是"宁可难看也不能骗人"）
------------------------------------------
1. **候选域必须展示**：本地快照只覆盖全市场 **54%**，且板块偏斜
   （科创板 5% / 创业板 34%）——"在覆盖域内找到 N 只" ≠ "在全市场找到 N 只"。
2. **快照不可用时明确报错**，不降级到内置精选池：拿硬编码股票冒充"按条件选出"，
   比直接说"这次算不了"糟得多。
3. **截断必须如实告知**：粗筛命中超过逐票精算上限时，显示"另有 N 只未逐一核对"
   ——不是「不符合」，是「还没算」。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

from core.screener import StockScreener
from core import nl_screener as nl
from core import local_pool
from core.nl_screener import NLParseError
import core.ui_runs as ui_runs
import core.ui_tasks as ui_tasks

st.set_page_config(page_title="自然语言选股", page_icon="🗣️", layout="wide")
st.title("🗣️ 自然语言选股")
st.caption("用一句话描述条件 → 解析成白名单条件 → 你确认后执行。"
           "**执行在本地完成**（全市场快照粗筛 → 逐票精算 → 逐条核对），不依赖第三方选股接口。")

TASK = "nl_screen"
BT_TASK = "nl_backtest"
_POLL_SECONDS = float(os.environ.get("BIGA_UI_POLL_SECONDS") or 2.0)
EXAMPLES = [
    "市盈率低于20、主力资金净流入为正、非ST",
    "总市值大于500亿、换手率低于5%的金融股票",
    "均线多头排列的新能源股票，波动率低于30%",
]


# ================================================================
# 后台任务
# ================================================================

def _nl_job(query_dict: dict, max_stocks: int, precision_cap: int):
    """构造在工作线程里执行的选股任务（线程内不得调用任何 Streamlit API）"""
    def job(progress):
        query = nl.validate_payload(query_dict, source="user")

        def _cb(ev: dict):
            progress({
                "percent": ev.get("percent", 0),
                "message": ev.get("message", ""),
                "current": ev.get("current", 0),
                "total": ev.get("total", 0),
            })

        payload = {
            "query": query.to_dict(),
            "conditions": query.describe(),
            "requested": max_stocks,
        }

        # ── Stage 0: 市场环境（市况极差时不出信号；失败不阻塞选股）──
        try:
            from core.market_detector import MarketDetector
            progress({"percent": 3, "message": "市场环境检测"})
            signal = MarketDetector().detect()
            payload["market_state"] = getattr(signal, "market_state", "")
            payload["market_score"] = getattr(signal, "total_score", None)
            if getattr(signal, "position_limit", 1) == 0:
                payload.update({
                    "skipped": True,
                    "reason": f"市场状态 {signal.market_state}"
                              f"（{signal.total_score} 分），自动跳过选股"})
                ui_runs.save_result(TASK, payload)
                return
        except Exception as e:                        # noqa: BLE001
            payload["market_error"] = f"{type(e).__name__}: {e}"

        # ── Stage 1: 本地全市场快照粗筛 ──
        progress({"percent": 8, "message": "本地全市场快照粗筛（新浪行业明细）"})
        try:
            pool = local_pool.build_pool(query, max_precision=precision_cap)
        except local_pool.PoolUnavailable as e:
            payload.update({"unavailable": True, "reason": str(e)})
            ui_runs.save_result(TASK, payload)
            return

        payload["pool"] = pool.to_dict()
        payload["snapshot_note"] = local_pool.pool_note(pool)
        if pool.empty:
            payload.update({"candidates": [], "passed": [], "checked": 0})
            ui_runs.save_result(TASK, payload)
            return

        # ── Stage 2: 逐票因子精算 ──
        def _cb_score(ev: dict):
            pct = float(ev.get("percent") or 0)
            _cb({"percent": 10 + pct * 0.8, "message": ev.get("message", ""),
                 "current": ev.get("current", 0), "total": ev.get("total", 0)})
        try:
            candidates = StockScreener().score_codes(pool.codes, progress_cb=_cb_score)
        except Exception as e:                        # noqa: BLE001
            payload.update({"unavailable": True,
                            "reason": f"因子精算失败：{type(e).__name__}: {e}"})
            ui_runs.save_result(TASK, payload)
            return

        # ── Stage 3: 本地逐条核对（带快照行，5 个快照类指标才核对得上）──
        progress({"percent": 92, "message": "按条件逐条核对"})
        annotated = nl.annotate_candidates(
            candidates, query,
            snapshot_lookup=lambda c: local_pool.snapshot_row_for(pool, c))

        passed = [c for c in annotated if c["nl_report"]["pass"]]
        passed.sort(key=lambda x: (x.get("score") is None, -(x.get("score") or 0.0)))
        payload.update({"candidates": annotated, "passed": passed[:max_stocks],
                        "checked": len(annotated)})
        ui_runs.save_result(TASK, payload)

        # 只把"条件核对通过"的推荐落库（供条件命中率回看）
        try:
            from core.screen_history import save_screen_result
            save_screen_result(
                {"skipped": False, "candidates": payload["passed"],
                 "market_signal": None}, source="nl_screen")
        except Exception:
            pass

    return job


@st.fragment(run_every=_POLL_SECONDS)
def _live_progress(task: str = TASK):
    prog = ui_tasks.read_progress(task) or {}
    try:
        pct = max(0.0, min(1.0, float(prog.get("percent") or 0) / 100.0))
    except (TypeError, ValueError):
        pct = 0.0
    try:
        elapsed = float(prog.get("elapsed") or 0)
    except (TypeError, ValueError):
        elapsed = 0.0
    cur, total = prog.get("current") or 0, prog.get("total") or 0
    counter = f" {cur}/{total}" if total else ""
    st.progress(pct, text=f"{prog.get('message') or '准备中…'}{counter}（已用 {elapsed:.0f}s）")
    if not ui_tasks.is_running(task):
        st.rerun(scope="app")


# ================================================================
# 一键回测（后台任务）
# ================================================================

def _bt_job(query_dict: dict, start: str, end: str, hold_days: int, precision_cap: int):
    """在后台线程里做"条件重放 → 回测"。候选域与选股**同源**（都是本地快照粗筛）。"""
    def job(progress):
        from core import condition_backtest as cbt
        query = nl.validate_payload(query_dict, source="user")
        progress({"percent": 5, "message": "构建候选域（本地全市场快照）"})
        try:
            pool = local_pool.build_pool(query, max_precision=precision_cap)
        except local_pool.PoolUnavailable as e:
            ui_runs.save_result(BT_TASK, {"unavailable": True, "reason": str(e)})
            return
        progress({"percent": 15, "message":
                  f"逐日重放 {len(pool.codes)} 只（{start} ~ {end}）"})
        try:
            result, plan = cbt.run_condition_backtest(
                pool.codes, query, start, end, holding_days=hold_days)
        except Exception as e:                        # noqa: BLE001
            ui_runs.save_result(BT_TASK, {
                "unavailable": True,
                "reason": f"回测执行失败：{type(e).__name__}: {e}"})
            return

        payload = {"plan": plan.to_dict(), "pool": pool.to_dict(),
                   "snapshot_note": local_pool.pool_note(pool),
                   "start": start, "end": end, "holding_days": hold_days,
                   "codes": len(pool.codes)}
        if result is None:
            payload["no_trades"] = True
        else:
            payload["metrics"] = {
                "总收益率(%)": result.total_return,
                "年化收益率(%)": result.annual_return,
                "夏普比率": result.sharpe_ratio,
                "最大回撤(%)": result.max_drawdown,
                "卡玛比率": result.calmar_ratio,
                "胜率(%)": result.win_rate,
                "交易次数": result.total_trades,
                "平均仓位(%)": round(getattr(result, "avg_position_pct", 0.0) or 0.0, 1),
                "净值点数": len(result.nav_curve),
            }
        ui_runs.save_result(BT_TASK, payload)

    return job


# ================================================================
# 结果渲染
# ================================================================

def _render_pipeline(payload: dict):
    """选股管道：每层剩多少 + 哪些条件在哪层执行（这是"结果怎么来的"）"""
    pool = payload.get("pool") or {}
    counts = pool.get("stage_counts") or {}
    if not counts:
        return
    with st.expander("🔧 选股管道（条件在哪一层执行、每层剩多少只）", expanded=False):
        st.markdown("**逐层过滤**")
        rows = [{"阶段": k, "剩余只数": v} for k, v in counts.items()]
        st.dataframe(rows, width="stretch", hide_index=True)

        applied = pool.get("conditions_applied") or []
        if applied:
            st.markdown("**已在粗筛层执行的条件**（本地快照，毫秒级）")
            for a in applied:
                st.markdown(f"- {a}")
        deferred = pool.get("conditions_deferred") or []
        if deferred:
            st.markdown("**需逐票精算的条件**（K线 / 资金流 / 舆情 / 评分）")
            for d in deferred:
                st.markdown(f"- {d}")
        for note in pool.get("notes") or []:
            st.caption("· " + note)


def render_nl_result(payload: dict):
    if not payload:
        st.info("没有可展示的结果")
        return

    # ── 不可用：明确失败，不给"看起来像结果"的东西 ──
    if payload.get("unavailable"):
        st.error(f"❌ **本次无法选股**：{payload.get('reason')}")
        if payload.get("query"):
            st.caption("条件本身已解析成功，是数据层不可用；稍后重试即可。")
        return

    conds = payload.get("conditions") or []
    if conds:
        st.caption("执行条件：" + "；".join(conds))
    if payload.get("snapshot_note"):
        st.caption("📊 候选域：" + payload["snapshot_note"])
    if payload.get("market_state"):
        st.caption(f"市场环境：{payload['market_state']}"
                   f"（{payload.get('market_score')} 分）")
    if payload.get("market_error"):
        st.caption(f"（市场检测不可用，已跳过该环节：{payload['market_error']}）")

    if payload.get("skipped"):
        st.warning(f"🔴 {payload.get('reason') or '市场环境不佳，已跳过选股'}")
        return

    passed = payload.get("passed") or []
    checked = payload.get("checked") or len(payload.get("candidates") or [])
    if not checked:
        st.warning("候选池为空：没有任何股票通过粗筛条件。"
                   "（可能是条件过严，或该板块在本地覆盖域内样本很少）")
        _render_pipeline(payload)
        return
    if not passed:
        st.warning(f"本地核对后没有股票满足全部条件（共核对 {checked} 只）。")
    else:
        st.success(f"✅ {len(passed)} 只满足全部条件（共核对 {checked} 只）")

    _render_pipeline(payload)

    for i, cand in enumerate(passed):
        report = cand.get("nl_report") or {}
        st.markdown(f"### #{i + 1} {cand.get('股票名称', '')} ({cand.get('股票代码', '')})")
        score = cand.get("score")
        st.caption(f"综合评分：**{score:.1f}**" if isinstance(score, (int, float))
                   else "综合评分：数据不足（未评分）")
        for hit in report.get("hits") or []:
            st.markdown(f"- ✅ {hit}")
        for unverified in report.get("unverified") or []:
            st.markdown(f"- ⚠️ {unverified}")
        st.divider()

    if checked:
        with st.expander(f"📋 全部核对明细（{checked} 只，含未通过）", expanded=False):
            rows = []
            for cand in payload.get("candidates") or []:
                report = cand.get("nl_report") or {}
                rows.append({
                    "代码": cand.get("股票代码", ""),
                    "名称": cand.get("股票名称", ""),
                    "评分": cand.get("score"),
                    "结论": "通过" if report.get("pass") else "未通过",
                    "未命中": "；".join(report.get("misses") or []) or "—",
                })
            st.dataframe(rows, width="stretch", hide_index=True)


def render_bt_result(payload: dict):
    if not payload:
        return
    if payload.get("unavailable"):
        st.error(f"❌ 回测无法执行：{payload['reason']}")
        return

    st.subheader("🔬 条件回测结果")
    plan = payload.get("plan") or {}
    st.caption(f"区间 {payload.get('start')} ~ {payload.get('end')}｜"
               f"持有 {payload.get('holding_days')} 个交易日｜"
               f"候选域 {payload.get('codes')} 只｜"
               f"可回放条件 {plan.get('replayable_count')} 条")
    if payload.get("snapshot_note"):
        st.caption("📊 " + payload["snapshot_note"])
    if plan.get("skipped"):
        st.warning("本次回测**未包含**以下条件（无法历史验证，不假装参与）："
                   + "、".join(x["条件"] for x in plan["skipped"]))

    if payload.get("no_trades"):
        st.info("该条件下没有产生任何交易信号——可能是条件过严，"
                "或候选域内没有满足的标的（注意候选域只覆盖全市场一部分）。")
        return

    metrics = payload.get("metrics") or {}
    cols = st.columns(3)
    for i, (k, v) in enumerate(metrics.items()):
        with cols[i % 3]:
            st.metric(k, v)
    st.caption("⚠️ 回测结论只在候选域与区间内成立；"
               "「是否只是运气」需要用同一套流程跑随机对照（见回测看板）。")


# ================================================================
# 第一步：输入与解析
# ================================================================

if "nl_text" not in st.session_state:
    st.session_state["nl_text"] = ""

cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    with col:
        if st.button(example, width="stretch", key=f"ex_{abs(hash(example))}"):
            st.session_state["nl_text"] = example
            st.session_state.pop("nl_query", None)
            st.rerun()

text = st.text_area("用一句话描述选股条件", key="nl_text", height=80,
                    placeholder="例如：市盈率低于20、主力资金净流入为正、均线多头排列的新能源股票")

parse_col, run_col = st.columns([1, 3])
with parse_col:
    parse_btn = st.button("🔍 解析条件", type="primary", width="stretch")

if parse_btn:
    st.session_state.pop("nl_query", None)
    try:
        with st.spinner("正在解析条件…"):        # 规则路径是毫秒级；只有兜底模型时才等
            llm_fn = nl.make_llm_fn()
            query = nl.parse_with_llm(text, llm_fn=llm_fn)
            query = nl.require_conditions(query)  # 解析不出可执行条件 → 显式报错
        st.session_state["nl_query"] = query.to_dict()
        if query.source == "llm":
            st.info("规则没解析出条件，已请模型兜底（输出已过白名单校验）。")
    except NLParseError as e:
        st.error(f"❌ 解析失败：{e}")
    except Exception as e:                        # 模型/网络异常也算解析失败，不静默降级
        st.error(f"❌ 解析失败：{type(e).__name__}: {e}")

# ================================================================
# 第二步：条件回显 + 编辑
# ================================================================

query_dict = st.session_state.get("nl_query")
if query_dict:
    try:
        query = nl.validate_payload(query_dict, source="user")
    except NLParseError as e:
        st.error(f"条件已失效：{e}")
        query = None
    if query is not None:
        pairs = [(c.scaled(), c.to_dict()) for c in query.conditions]
        st.subheader("解析出的条件")
        if pairs:
            for label, _ in pairs:
                st.markdown(f"- {label}")
            options = list(range(len(pairs)))
            chosen = st.multiselect(
                "要执行的条件（可删掉不需要的）", options=options,
                default=options, format_func=lambda i: pairs[i][0])
            kept = [pairs[i][1] for i in chosen]
        else:
            kept = []
            st.caption("这次没有数值型条件（只有板块/ST 说明）。")
        st.caption(("板块：" + query.sector if query.sector else "")
                   + ("；剔除 ST" if query.exclude_st else "；包含 ST"))

        if query.unparsed:
            st.warning("以下片段没有被执行（如实告知，不做猜测）：\n\n"
                       + "\n".join(f"- {u}" for u in query.unparsed))

        # ── 选项式澄清：把"没听懂"变成"点一下" ──
        # 只在**我们能给出候选映射**时出手；给不出就保持上面的报错原样（不猜）。
        from core import nl_clarify as clarify
        questions = clarify.detect(query) + clarify.sanity_questions(query)
        if questions:
            st.subheader("🤔 需要你确认一下")
            for qi, q_ in enumerate(questions):
                st.markdown(f"**{q_.question}**")
                if not q_.options:
                    st.caption("（请在上面的条件列表里手动调整数值）")
                    continue
                cols = st.columns(len(q_.options))
                for oi, (col, opt) in enumerate(zip(cols, q_.options)):
                    with col:
                        if st.button(opt.label, key=f"clar_{qi}_{oi}", width="stretch"):
                            merged = query
                            for cond in opt.conditions:
                                merged = clarify.merge_condition(merged, cond, q_.fragment)
                            st.session_state["nl_query"] = merged.to_dict()
                            if opt.note:
                                st.session_state["_clarify_note"] = opt.note
                            st.rerun()

        effective = {"conditions": kept, "sector": query.sector,
                     "board": query.board, "exclude_st": query.exclude_st}
        effective_query = nl.validate_payload(effective, source="user")

        # 告诉用户每个条件将在哪一层落地（粗筛层 vs 逐票精算）
        from core.indicator_registry import INDICATORS, SOURCE_SNAPSHOT
        snap_conds = [c.scaled() for c in effective_query.conditions
                      if INDICATORS[c.field].source == SOURCE_SNAPSHOT]
        deep_conds = [c.scaled() for c in effective_query.conditions
                      if INDICATORS[c.field].source != SOURCE_SNAPSHOT]
        if snap_conds:
            st.caption("🟢 粗筛层即刻生效（全市场快照）：" + "、".join(snap_conds))
        if deep_conds:
            st.caption("🔵 需逐票精算（只能覆盖精算上限内的候选）：" + "、".join(deep_conds))

        # ── 一键回测：回测的是"这套条件"，不是"这次选出的 5 只票" ──
        # 前者是历史每天重新选股（数百笔交易，有统计意义）；
        # 后者是 5 只 × 1 个时点，几乎说明不了任何事。
        from core import condition_backtest as cbt
        bt_plan = cbt.plan(effective_query)
        with st.expander(
                f"🔬 回测这套条件（可回放 {len(bt_plan.replayable)} 条 · "
                f"无法验证 {len(bt_plan.skipped)} 条）", expanded=False):
            for c_ in bt_plan.replayable:
                st.markdown(f"- ✅ {c_.scaled()}")
            if bt_plan.skipped:
                st.warning("以下条件**无法历史验证**，回测中不参与——"
                           "不假装它们参与了（历史估值/资金流/舆情无稳定免费源）：")
                for label, why in bt_plan.skipped:
                    st.markdown(f"- ⚠️ {label} —— {why}")
            if not bt_plan.ok:
                st.info("至少要有一条可回放的条件才能回测。")
            else:
                b1, b2, b3 = st.columns(3)
                with b1:
                    bt_start = st.text_input("回测起始", "2024-01-01", key="bt_start")
                with b2:
                    bt_end = st.text_input("回测结束", "2025-06-30", key="bt_end")
                with b3:
                    bt_hold = st.number_input("持有天数", 5, 60, 20, key="bt_hold")
                if st.button("🔬 开始回测（后台运行）", key="bt_run", width="stretch"):
                    if not ui_tasks.start(BT_TASK,
                                          _bt_job(effective, bt_start, bt_end,
                                                  int(bt_hold), precision_cap),
                                          {"conditions": effective_query.describe()}):
                        st.error("已有回测在运行，本次点击已忽略。")
                    else:
                        st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            max_stocks = st.slider("最多展示几只", 3, 10, 5)
        with c2:
            precision_cap = st.slider(
                "逐票精算上限（只）", 20, 200, local_pool.MAX_PRECISION, step=10,
                help="粗筛命中数超过此上限时，按成交额降序取前 N 只精算，"
                     "其余会如实告知「未逐一核对」。")
        with run_col:
            st.write("")
            run_btn = st.button("✅ 按这些条件选股", type="primary", width="stretch")
        if run_btn:
            if not effective_query.conditions and not effective_query.sector:
                st.error("至少保留一个条件或一个板块，否则没有可执行的条件。")
            elif not ui_tasks.start(TASK, _nl_job(effective, max_stocks, precision_cap),
                                    {"conditions": effective_query.describe()}):
                st.error("已有自然语言选股任务在运行，本次点击已忽略。")
            else:
                st.rerun()
else:
    st.info("👆 输入一句选股要求，先点「解析条件」看看解析成了什么。")

# ================================================================
# 第三步：进度与结果
# ================================================================

if ui_tasks.is_running(TASK):
    record = ui_runs.get_running_any(TASK) or {}
    st.warning(f"⏳ 选股正在**后台线程**里运行（{record.get('started_at', '')} 启动）。"
               "刷新页面、关掉标签页都不影响它；跑完结果会自动落盘。")
    _live_progress(TASK)
    if st.button("✖️ 取消本次选股"):
        ui_tasks.request_cancel(TASK)
        st.info("已请求取消：会在当前这只股票处理完后停止。")
else:
    ui_runs.clear_dead_running()
    prog = ui_tasks.read_progress(TASK) or {}
    status = prog.get("status")
    if status == "cancelled":
        st.info("ℹ️ 上一次选股已取消，结果未更新。")
    elif status == "error":
        st.error(f"❌ 上一次选股失败：{prog.get('message', '')}")
    elif status in ("running", "starting"):
        st.warning("⚠️ 上一次选股没有正常结束（进程被关闭或崩溃）。")
        ui_tasks.clear_progress(TASK)

    saved = ui_runs.load_result(TASK)
    if saved:
        st.divider()
        st.caption(f"📌 结果生成于 {saved.get('generated_at', '')}"
                   f"（{ui_runs.age_text(saved)} 前）；刷新不会丢。")
        render_nl_result(saved.get("payload") or {})

# ================================================================
# 第四步：一键回测的进度与结果
# ================================================================

if ui_tasks.is_running(BT_TASK):
    st.divider()
    st.info("⏳ 条件回测正在后台运行（逐日重放，可能几分钟）。")
    _live_progress(BT_TASK)
else:
    _bt_saved = ui_runs.load_result(BT_TASK)
    if _bt_saved:
        st.divider()
        st.caption(f"📌 回测结果生成于 {_bt_saved.get('generated_at', '')}"
                   f"（{ui_runs.age_text(_bt_saved)} 前）")
        render_bt_result(_bt_saved.get("payload") or {})

# ===== 条件命中率（P2：本页解析出的条件到底有没有区分度）=====
try:
    from ui.components.screen_history_panel import render as render_screen_history
    render_screen_history(title="🎯 条件命中率回测", show_conditions=True)
except Exception as e:
    st.caption(f"命中率面板不可用: {e}")
