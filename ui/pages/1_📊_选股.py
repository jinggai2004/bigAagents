"""全市场选股页面 — 使用 screener 后端 + 因子拆解渲染"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
from core.screener import StockScreener, SECTOR_HARDCODED
from core.sector import SCORE_WEIGHT_CAVEAT
import core.ui_runs as ui_runs
import core.ui_tasks as ui_tasks
from ui.components.market_dashboard import render_market_dashboard

st.set_page_config(page_title="全市场选股", page_icon="📊", layout="wide")
st.title("📊 全市场选股")

ANALYSIS_PAGE = "pages/2_🔍_深度分析.py"


def _goto_analysis(code: str):
    """预填代码并跳转深度分析页（与热点新闻页同一机制）。

    ⚠️ 必须在 `switch_page` **之前**写 `prefill_code` —— 跳转后本页脚本即终止。
    """
    st.session_state["prefill_code"] = str(code)
    st.switch_page(ANALYSIS_PAGE)


# ========== 结果渲染（新跑与"刷新后回填"共用同一套） ==========

def _signal_to_dict(signal):
    """MarketSignal 是 dataclass，落盘前转 dict；失败返回 None（不阻塞主流程）"""
    if signal is None or isinstance(signal, dict):
        return signal
    try:
        import dataclasses
        return dataclasses.asdict(signal)
    except Exception:
        return None


def _signal_from_dict(data):
    """回填时把 dict 还原成 MarketSignal（render_market_dashboard 需要属性访问）"""
    if not data:
        return None
    try:
        from core.market_detector import MarketSignal
        return MarketSignal(**data)
    except Exception:
        return None


def render_screen_result(result: dict):
    if not result:
        st.info("没有可展示的结果")
        return

    signal = result.get("market_signal")
    if isinstance(signal, dict):
        signal = _signal_from_dict(signal)
    if signal is not None:
        render_market_dashboard(signal)

    if result.get("skipped"):
        st.warning(f"🔴 {result.get('reason', '市场环境不佳，已自动跳过选股')}")
        return

    candidates = result.get("candidates", [])
    if not candidates:
        st.info("未找到符合条件的股票")
        return

    st.success(f"✅ 共选出 {len(candidates)} 只候选股")
    # ⚠️ 评分说明放在**结果区的第一行**，而不是折叠或页面底部：
    #    "综合评分 82" 是这一页最醒目的数字，用户会直接拿它做判断。
    #    说明如果不跟着数字出现，等于默认把未验证的权重当成了结论。
    st.info("📌 **评分怎么读**：" + SCORE_WEIGHT_CAVEAT)
    # 选股 → 分析 的衔接（2026-10-10 用户反馈"选股选不出优质股，导致深度分析范围加大"）：
    # 此前"到底该分析哪只"全靠人工挑，于是范围要么漏要么失控（同一只票一天被分析 4 次）。
    # 队列把 TOP5 显式带过去，深度分析页会真的消费它（不是装饰）。
    _qc1, _qc2 = st.columns([2, 5])
    if _qc1.button("📋 将 TOP5 加入分析队列", key="queue_top5"):
        _top5 = [c.get("股票代码", "") for c in candidates[:5] if c.get("股票代码")]
        st.session_state["analysis_queue"] = _top5
        st.success("已加入分析队列：" + "、".join(_top5)
                   + " —— 到「🔍 深度分析」页可直接逐个点开。")
    _qc2.caption("提示：每只候选右侧也有「🔎 深度分析」按钮，可单独跳转（自动填入代码）。")
    # 初筛明细（2026-10-09 本地化后新增）：哪几路跑了、哪路不可用、本地条件
    # 与原问财条件的差异、被截断了多少只 —— "缺就是缺"必须让人看见。
    _smeta = result.get("strategy_meta") or {}
    _routes = _smeta.get("routes") or []
    if _routes:
        _bad = [r for r in _routes if r.get("status") != "ok"]
        if _bad:
            st.warning("⚠️ 本次初筛有 {} 路不可用：{}".format(
                len(_bad), "；".join(f"{r['label']}（{r['note']}）" for r in _bad)))
        with st.expander("🔎 初筛明细（每路条件 / 可用性 / 与问财原条件的差异）",
                         expanded=bool(_bad)):
            for r in _routes:
                icon = "🟢" if r.get("status") == "ok" else "⚪"
                st.markdown(f"{icon} **{r['label']}**（{r.get('count', 0)} 只）—— {r.get('note', '')}")
                if r.get("query"):
                    st.caption(f"原问财条件：{r['query']}")
            _dd = _smeta.get("data_dates") or {}
            st.caption(f"数据日期：估值快照 {_dd.get('market_snapshot') or '无'} ｜ "
                       f"资金流快照 {_dd.get('fund_flow_snapshot') or '无'}")
            if _smeta.get("truncated"):
                st.caption(f"⚠️ 另有 **{_smeta['truncated']}** 只因精算上限（{_smeta.get('total_limit')}）"
                           "未逐一评分——它们也满足条件，只是没排进本轮。")
    _pf = result.get("prefetch") or {}
    if _pf.get("ok") or _pf.get("failed"):
        st.caption(f"⚡ 并发预取：成功 **{_pf.get('ok', 0)}** 只"
                   + (f"、失败 {_pf['failed']} 只（已退回逐票取数）" if _pf.get("failed") else "")
                   + f"，取数耗时 {_pf.get('elapsed', '?')}s"
                   "（取数与算因子分离：候选域因此能放大到上百只）")
    if result.get("degraded"):
        st.warning("⚠️ 本次结果包含降级项：" + "；".join(result.get("degrade_reasons", [])))
    if result.get("insufficient_data"):
        st.info("ℹ️ 以下候选有效因子不足、未给出评分：" + "、".join(result["insufficient_data"]))
    # R2（2026-10-18 审查）：因**本地日线过期**被剔除的标的必须可见。
    # 与 `insufficient_data` 同一处展示——"降级必须随数据走"，
    # 否则用户只看到候选变少，却不知道为什么（也可能是自己以为的"全市场"其实少了 4.4%）。
    _stale = result.get("stale_skipped") or {}
    if _stale:
        _items = list(_stale.items())
        _show = "、".join(f"{c}（{r}）" for c, r in _items[:8])
        _more = f" 等 {len(_items)} 只" if len(_items) > 8 else ""
        st.warning(f"⏳ 因本地日线过期而不参与排名（停牌/退市，最后交易日距今过久）："
                   f"{_show}{_more}")
    # 流动性下限（2026-10-29）：与 stale_skipped 同一纪律 —— 门槛必须可复核，
    # 否则用户只看到候选变少，以为是自己条件太严或者行情太差。
    _liq = result.get("liquidity_floor") or {}
    if _liq.get("applied") and _liq.get("dropped"):
        _d = _liq["dropped"]
        _show = "、".join(f"{x['股票代码']}（{x['成交额']}亿）" for x in _d[:8])
        _more = f" 等 {len(_d)} 只" if len(_d) > 8 else ""
        st.caption(f"💧 流动性下限：剔除成交额低于**本板块** {_liq.get('date')} 30% 分位的候选"
                   f"（门槛：{_liq.get('note', '')}）：{_show}{_more}")
    elif _liq and not _liq.get("applied") and _liq.get("note"):
        st.caption(f"💧 流动性下限未生效：{_liq['note']}")
    # 波动率口径：换了 IVOL 之后必须让用户看得到（否则"分数怎么变的"无从追溯）
    _vb = result.get("vol_basis") or {}
    _vs = result.get("vol_pct_scope")
    if _vb.get("ivol"):
        _scope_txt = {"industry": "行业内分位", "pool": "全候选分位（行业映射不可用）",
                      "mixed": "行业内 + 未分行业池"}.get(_vs or "", "")
        st.caption(f"📐 波动率口径：残差波动率（IVOL）{_vb['ivol']} 只"
                   + (f"、总波动率兜底 {_vb['total']} 只" if _vb.get("total") else "")
                   + (f"；分位口径 {_scope_txt}" if _scope_txt else "")
                   + "。IVOL 扣掉市场因子，并按行业分别排分位 —— 避免整个名单被推向银行。")

    # ====== 排序与分散明细（2026-10-10）======
    _rk = result.get("rank_meta") or {}
    if _rk:
        _skip = _rk.get("skipped_by_sector") or []
        if _skip:
            st.caption("🔀 行业分散：同行业最多 {} 只，以下候选因行业上限未入选：{}".format(
                _rk.get("sector_cap"),
                "、".join(f"{s['股票代码']}({s['行业']})" for s in _skip)))
        if _rk.get("cap_relaxed"):
            st.caption("ℹ️ 行业分散导致候选不足，已放宽行业上限补足名单（分散是优先而非牺牲数量）")

    # ====== 相似度监控（2026-10-10）======
    # 把"每次都趋于同几只"变成页面**自己会报的警**：原来这个问题只能靠人工比对
    # screen_picks 才发现（跨日 TOP5 重合 4/5、同日 5/5）。
    try:
        from core.screen_history import recent_top_overlap
        # `skip_latest=True`：页面上看到的结果**已经落库**（最新批次＝本次自己），
        # 不跳过就会自比 100% 重合、每次都弹"高度重合"的误报。
        _ov = recent_top_overlap([c.get("股票代码", "") for c in candidates],
                                 lookback=5, top_n=5, skip_latest=True)
    except Exception:                            # noqa: BLE001
        _ov = {}                                 # 提示信息读不到就算了，绝不阻断
    if _ov:
        if _ov.get("alert"):
            st.warning(
                f"⚠️ 本次 TOP{_ov['top_n']} 与最近 {_ov['batch_count']} 次选股**高度重合**"
                f"（最高 {_ov['max_overlap']}/{_ov['top_n']}）——"
                "可能来自候选域收窄或排序偏置（低 PE、低波动类因子绝对值分档恒定占优），"
                "**不代表这次真的该选它们**。建议同时看因子的有效权重覆盖度与行业分布。")
        with st.expander(f"🔁 与最近 {_ov['batch_count']} 次选股的重合度"
                         f"（最高 {_ov['max_overlap']}/{_ov['top_n']}，均值 {_ov['avg_overlap']}）",
                         expanded=False):
            st.dataframe([{"批次时间": d["time"], "重合只数": d["overlap"],
                           "该次入选": "、".join(d["codes"])} for d in _ov["details"]],
                         width="stretch", hide_index=True)
            st.caption("阈值：重合 ≥{} 只即告警。若多为 5/5，说明排序缺少轮动——"
                       "见 `output/选股重复度过高_根因分析与改进方案.md`。"
                       .format(_ov["threshold"]))

    # ====== 因子拆解渲染 ======
    for i, stock in enumerate(candidates):
        code = stock.get("股票代码", "")
        name = stock.get("股票名称", "")
        score = stock.get("score")
        strategy = stock.get("策略", "")
        factors = stock.get("factors", {})
        coverage = stock.get("score_coverage")

        with st.container():
            _hd1, _hd2 = st.columns([5, 1])
            _hd1.markdown(f"### #{i+1} {name} ({code})")
            if code and _hd2.button("🔎 深度分析", key=f"goto_analysis_{code}_{i}"):
                _goto_analysis(code)
            if isinstance(score, (int, float)):
                cov_txt = (f" | 有效因子权重 {coverage:.0%}"
                           if isinstance(coverage, (int, float)) else "")
                st.caption(f"策略: {strategy} | 综合评分: **{score:.1f}**{cov_txt}")
            else:
                st.caption(f"策略: {strategy} | 综合评分: **数据不足（未评分）**")
                if stock.get("score_detail"):
                    st.caption(stock["score_detail"])
            if stock.get("degraded"):
                st.warning(f"⚠️ 降级结果：{stock.get('degrade_reason', '')}")

            if factors:
                # 因子条（status=unknown 的因子显示为"数据缺失"，不画进度）
                factor_names = list(factors.keys())
                cols = st.columns(len(factor_names))
                for j, fname in enumerate(factor_names):
                    fd = factors[fname]
                    f_score = fd.get("score")
                    f_max = fd.get("max", 1)
                    unknown = (fd.get("status") == "unknown"
                               or not isinstance(f_score, (int, float)))
                    ratio = (f_score / f_max) if (not unknown and f_max > 0) else 0
                    filled = int(ratio * 8)
                    bar = "█" * filled + "░" * (8 - filled)
                    with cols[j]:
                        st.markdown(f"**{fname}**")
                        st.caption("`数据缺失`" if unknown else f"`{f_score}/{f_max}`")
                        st.caption(f"`{bar}`")
                        if fd.get("detail"):
                            st.caption(fd["detail"])
            st.divider()


# ========== 控制栏 ==========
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    sector_choice = st.selectbox(
        "板块",
        ["全市场"] + list(SECTOR_HARDCODED.keys()),
        index=0,
    )
with col2:
    max_stocks = st.slider("候选数量", 3, 10, 5)
with col3:
    st.write("")
    st.write("")
    run_btn = st.button("🔍 开始选股", type="primary", width="stretch")

with st.expander("⚙️ 初筛策略（默认全部五路）", expanded=False):
    from core.screener import STRATEGIES
    strategy_choice = st.multiselect(
        "选择启用的初筛策略",
        list(STRATEGIES.keys()),
        default=list(STRATEGIES.keys()),
        format_func=lambda s: f"{s} — {STRATEGIES[s][:20]}…",
    )
    if not strategy_choice:
        st.caption("未选任何策略时将使用默认全部五路")

# ========== 主体 ==========
TASK = "screen"


def _screen_job(sector, strategy_arg, max_stocks):
    """构造在工作线程里执行的选股任务。

    ⚠️ 线程里**不能调用任何 Streamlit API**（没有 ScriptRunContext），
    要报进度就调用传入的 `progress()`，它只写文件。
    """
    def job(progress):
        def _cb(ev: dict):
            progress({
                "percent": ev.get("percent", 0),
                "message": ev.get("message", ""),
                "current": ev.get("current", 0),
                "total": ev.get("total", 0),
            })

        result = StockScreener().run(sector_filter=sector, max_stocks=max_stocks,
                                     strategies=strategy_arg, progress_cb=_cb)
        # 先落盘：刷新/关页面/换标签页后仍能看到这次结果
        ui_runs.save_result(TASK, {
            **result,
            "market_signal": _signal_to_dict(result.get("market_signal")),
        })
        # 候选进候选库：这才是"选股有没有用"的样本来源（每日只有几次深度分析，
        # 攒不够胜率统计需要的样本量；选股一次就有 5~10 只）
        try:
            from core.screen_history import save_screen_result
            save_screen_result(result, source="screen")
        except Exception:
            pass          # 落库失败绝不能让选股结果丢失

    return job


def _fmt_elapsed(prog: dict) -> float:
    try:
        return float(prog.get("elapsed") or 0)
    except (TypeError, ValueError):
        return 0.0


# 进度轮询间隔（秒）。默认 2 秒；冒烟测试把它调大，免得 AppTest 真的起一个
# 轮询线程空转（run_every 在 AppTest 里是生效的）。
_POLL_SECONDS = float(os.environ.get("BIGA_UI_POLL_SECONDS") or 2.0)


@st.fragment(run_every=_POLL_SECONDS)
def _live_progress():
    """后台任务的实时进度：每 2 秒重跑这个片段，只读进度文件。

    片段只在"确实有任务在跑"时被渲染，所以任务结束后这里的 st.rerun 必然
    进入"无任务"分支、片段不再被渲染，不会自激循环。
    """
    prog = ui_tasks.read_progress(TASK) or {}
    try:
        pct = max(0.0, min(1.0, float(prog.get("percent") or 0) / 100.0))
    except (TypeError, ValueError):
        pct = 0.0
    cur, total = prog.get("current") or 0, prog.get("total") or 0
    counter = f" {cur}/{total}" if total else ""
    st.progress(pct, text=f"{prog.get('message') or '准备中…'}{counter}"
                          f"（已用 {_fmt_elapsed(prog):.0f}s）")
    if not ui_tasks.is_running(TASK):
        st.rerun(scope="app")          # 跑完了 → 整页重跑，切到结果视图


if run_btn and not ui_tasks.is_running(TASK):
    sector = None if sector_choice == "全市场" else sector_choice
    strategy_arg = strategy_choice if 0 < len(strategy_choice) < len(STRATEGIES) else None
    if not ui_tasks.start(TASK, _screen_job(sector, strategy_arg, max_stocks),
                          {"sector": sector_choice, "max_stocks": max_stocks}):
        st.error("已有选股任务在运行，本次点击已忽略——重复启动会叠加任务并重复请求数据源。")
    else:
        st.rerun()
elif run_btn:
    record = ui_runs.get_running_any(TASK) or {}
    st.error(f"已有选股任务在运行（{record.get('started_at', '')} 启动），"
             "本次点击已忽略——重复启动会叠加任务并重复请求数据源。")

if ui_tasks.is_running(TASK):
    record = ui_runs.get_running_any(TASK) or {}
    st.warning(f"⏳ 选股正在**后台线程**里运行（{record.get('started_at', '')} 启动）。"
               "刷新页面、关掉标签页、换个浏览器都不影响它，跑完结果会自动落盘；"
               "这个进度条只是「读者」，每 2 秒读一次进度文件。")
    _live_progress()
    if st.button("✖️ 取消本次选股"):
        ui_tasks.request_cancel(TASK)
        st.info("已请求取消：会在当前这只股票处理完后停止（不会留下半截结果）。")
else:
    # 没有活着的任务：如实说明上一次的结局，然后展示已落盘的结果
    ui_runs.clear_dead_running()
    prog = ui_tasks.read_progress(TASK) or {}
    status = prog.get("status")
    if status == "cancelled":
        st.info("ℹ️ 上一次选股已取消，结果未更新。")
    elif status == "error":
        st.error(f"❌ 上一次选股失败：{prog.get('message', '')}")
    elif status in ("running", "starting"):
        # 标记没了但进度还停在"运行中" → 进程被关闭/崩溃，线程跟着没了
        st.warning("⚠️ 上一次选股没有正常结束（进程被关闭或崩溃），任务已停止且没有结果。")
        ui_tasks.clear_progress(TASK)

    saved = ui_runs.load_result(TASK)
    if saved:
        st.caption(f"📌 下面是上次保存的选股结果（{ui_runs.age_text(saved)} 前生成）。"
                   "刷新页面不会丢；点「开始选股」会重新执行并覆盖它。")
        render_screen_result(saved.get("payload") or {})
    elif status not in ("cancelled", "error", "running", "starting"):
        st.info("👆 选择板块，点击「开始选股」")
        st.markdown("""
    ### 选股流程
    1. 📊 **市场检测** — 14指标评估大盘环境，熊市自动跳过
    2. 🔍 **五路本地初筛** — 主力资金/技术突破/低估值/龙虎榜/行业龙头
       （2026-10-09 起改用本地全市场快照：问财接口已失效，此前每次都降级到
       18 只硬编码池，导致选股结果高度重复）
    3. ⚖️ **因子精筛（单一权重集）** — 主力资金/PE估值/趋势(20日动量)/均线排列/波动率/新闻正面率；
       缺失数据的因子标记为 unknown 并从权重中剔除，总分只在有效权重上归一化
    4. 🏆 **TOP5输出** — 每只股票附带因子拆分明细
    """)
# ===== 选股历史与命中率（P2：这些条件到底有没有用）=====
try:
    from ui.components.screen_history_panel import render as render_screen_history
    render_screen_history()
except Exception as e:
    st.caption(f"选股历史面板不可用: {e}")
