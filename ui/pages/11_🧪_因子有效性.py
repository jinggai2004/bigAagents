"""因子有效性页面

一句话：**回答「这些因子的权重有没有依据」**。

为什么单独一页
------------
项目有一条"六因子加权 → 打分 → 排序"的链（`core/sector.py::FACTOR_WEIGHTS`），
但那些权重是**硬编码常量、从未被数据验证过**。本页把 IC / 分层收益 / 相关性
摆出来，让"权重该不该改"有数据可依 —— 而不是靠感觉。

操作设计（刻意压到三步，默认值即可跑）
------------------------------------
1. 选股票池与区间（**默认值直接能用**；首次想快速看效果就勾「快速模式」）
2. 点「开始分析」
3. 读结论表

后台线程运行（刷新不中断、结果落盘），与选股页 / 回测页同一套惯例。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

from core import factor_analysis as fa
import core.ui_runs as ui_runs
import core.ui_tasks as ui_tasks

st.set_page_config(page_title="因子有效性", page_icon="🧪", layout="wide")
st.title("🧪 因子有效性")
st.caption("IC / 分层收益 / 因子相关性。**为什么要有这一页**：项目的六因子权重"
           "（0.2353 / 0.1765 / …）是硬编码常量，从未被数据验证——这里补上那一步。")

TASK = "factor_analysis"
_POLL_SECONDS = float(os.environ.get("BIGA_UI_POLL_SECONDS") or 2.0)


# ================================================================
# 股票池
# ================================================================

@st.cache_data(show_spinner=False)
def _load_pool():
    try:
        with open("data/backtest_pool.json", encoding="utf-8") as f:
            d = json.load(f)
        return d.get("pool") or [], d
    except Exception as e:                            # noqa: BLE001
        return [], {"caveat": f"股票池不可用: {e}"}


POOL, POOL_META = _load_pool()


# ================================================================
# 后台任务
# ================================================================

def _job(codes, factors, start, end, horizon, step, groups):
    def job(progress):
        def _cb(ev):
            progress({"percent": ev.get("percent", 0),
                      "message": ev.get("message", ""),
                      "current": ev.get("current", 0),
                      "total": ev.get("total", 0)})
        progress({"percent": 2, "message": "准备中"})
        try:
            res = fa.run(codes, factors=factors, start=start, end=end,
                         horizon=horizon, step=step, groups=groups,
                         progress_cb=_cb)
        except Exception as e:                        # noqa: BLE001
            ui_runs.save_result(TASK, {"error": f"{type(e).__name__}: {e}"})
            return
        payload = res.to_dict()
        payload["params"] = {"start": start, "end": end, "horizon": horizon,
                             "step": step, "groups": groups,
                             "codes": len(codes), "factors": list(factors)}
        ui_runs.save_result(TASK, payload)

    return job


def _job_robustness(codes, factors, horizons, step, groups):
    """样本外稳健性：多区间 × 多持有期。日线只抓一次，所以比多次单区间跑快得多。"""
    def job(progress):
        def _cb(ev):
            progress({"percent": ev.get("percent", 0),
                      "message": ev.get("message", ""),
                      "current": ev.get("current", 0),
                      "total": ev.get("total", 0)})
        progress({"percent": 2, "message": "准备中"})
        try:
            res = fa.robustness(codes, factors=factors, horizons=horizons,
                                step=step, groups=groups, progress_cb=_cb)
        except Exception as e:                        # noqa: BLE001
            ui_runs.save_result(TASK, {"mode": "robustness",
                                       "error": f"{type(e).__name__}: {e}"})
            return
        payload = res.to_dict()
        payload["mode"] = "robustness"
        payload["params"] = {"codes": len(codes), "step": step, "groups": groups,
                             "horizons": list(horizons), "factors": list(factors),
                             "periods": res.periods}
        ui_runs.save_result(TASK, payload)

    return job


@st.fragment(run_every=_POLL_SECONDS)
def _live_progress():
    prog = ui_tasks.read_progress(TASK) or {}
    try:
        pct = max(0.0, min(1.0, float(prog.get("percent") or 0) / 100.0))
    except (TypeError, ValueError):
        pct = 0.0
    cur, total = prog.get("current") or 0, prog.get("total") or 0
    counter = f" {cur}/{total}" if total else ""
    st.progress(pct, text=f"{prog.get('message') or '准备中…'}{counter}")
    if not ui_tasks.is_running(TASK):
        st.rerun(scope="app")


# ================================================================
# 结果渲染
# ================================================================

def _verdict(row: dict) -> str:
    t = row.get("t值")
    if t is None:
        return "⚪ 样本不足"
    if row.get("结论") == "显著有效":
        return "🟢 显著有效"
    if row.get("结论") == "显著但方向相反":
        return "🟠 显著但方向相反"
    return "🟡 不显著"


def render_result(payload: dict):
    if not payload:
        st.info("还没有结果。")
        return
    if payload.get("error"):
        st.error(f"❌ 分析失败：{payload['error']}")
        return

    p = payload.get("params") or {}
    st.caption(f"区间 {p.get('start')} ~ {p.get('end')}｜持有期 {p.get('horizon')} 交易日｜"
               f"采样步长 {p.get('step')}｜分组 {p.get('groups')}｜"
               f"股票池 {p.get('codes')} 只（有效 {payload.get('names')}）｜"
               f"期数 {payload.get('periods')}｜面板 {payload.get('panel_rows')} 行")
    if payload.get("note"):
        st.warning(payload["note"])

    # ── IC 汇总：本页的主结论 ──
    st.subheader("① IC 汇总")
    summary = payload.get("summary") or []
    if not summary:
        st.warning("没有算出 IC —— 可能是样本太少或区间太短。")
    else:
        rows = []
        for x in summary:
            rows.append({
                "因子": x["因子"], "方向": x["方向"],
                "IC 均值": x.get("IC均值"), "ICIR": x.get("ICIR"),
                "t 值": x.get("t值"), "IC>0 比例": x.get("IC>0比例"),
                "期数": x.get("期数"), "判定": _verdict(x),
                "说明": x.get("结论"),
            })
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption("判据：|t| > 2 粗略视为显著；ICIR = IC 均值 / IC 标准差，衡量稳定性。"
                   "**IC 均值接近 0 或不显著，说明这个因子在当期数据上没有区分度**——"
                   "这正是「权重该不该改」的依据。")

    # ── 分层收益：看单调性 ──
    st.subheader("② 分层收益（按因子值分组，看未来收益是否单调）")
    layered = payload.get("layered") or {}
    if not layered:
        st.caption("没有可用的分层结果。")
    for key, info in layered.items():
        label = fa.FACTORS.get(key, {}).get("label", key)
        st.markdown(f"**{label}** —— 多空差 {info.get('多空差%')}% ，"
                    f"{'单调 ✅' if info.get('单调') else '不单调 ⚠️'}")
        rows = info.get("rows") or []
        if rows:
            st.bar_chart({r["组"]: r["平均收益%"] for r in rows}, height=180)
        st.caption(info.get("说明", ""))

    # ── 相关性 ──
    st.subheader("③ 因子相关性（均值矩阵）")
    corr = payload.get("correlation")
    if corr:
        st.dataframe(corr, width="stretch")
        st.caption("相关性高（|ρ| > 0.7）的两个因子在加权时会**重复计分**——"
                   "这也是权重该不该调的判断依据之一。")
        cp = payload.get("corr_const_periods") or 0
        ctot = payload.get("corr_periods") or 0
        if cp and ctot and cp >= ctot:
            st.caption(f"⚠️ 本页相关矩阵里**整行整列为空**的因子，是因为它在参与计算的 "
                       f"全部 {ctot} 期里**取值恒定**（布尔因子整期全 0 或全 1），"
                       f"方差为 0 → 截面 Spearman 相关**数学上无定义**。"
                       f"这是诚实结果，不是算错：这类因子在池子里没有区分度，"
                       f"本就无法与其他因子比较相关性。")
        elif cp:
            st.caption(f"⚠️ 有 **{cp} / {ctot}** 期的相关矩阵含缺失值：布尔因子"
                       f"（均线多头排列 / 站上20日均线）在那些期里**整期取值恒定**"
                       f"（全 0 或全 1），方差为 0，Spearman 无定义 → 该期它的"
                       f"相关系数不可计算。矩阵里的空格就是这种情况，"
                       f"按**其余期**求平均得到，不是算错。")
    else:
        st.caption("相关性不足（每期样本太少）。")

    # ── 诚实披露：哪些因子这一版算不了 ──
    unavailable = payload.get("unavailable") or {}
    snap = payload.get("snapshot_status") or {}
    if unavailable:
        with st.expander(f"📌 这一版算不了的因子（{len(unavailable)} 个）与原因", expanded=True):
            for k, why in unavailable.items():
                label = fa.FACTORS.get(k, {}).get("label", k)
                st.markdown(f"- **{label}**（`{k}`）：{why}")
            st.info(fa.snapshot_gap_note())
            # 把"需要历史截面"从一句解释变成一条**进度**：
            # 没有数字用户无法判断是"再等几天"还是"根本没这条路"。
            if snap:
                need = max(1, int(snap.get("need") or 1))
                got = int(snap.get("dates") or 0)
                if snap.get("enough"):
                    st.success(f"✅ 估值截面快照已积累 **{got}** 个交易日"
                               f"（{snap.get('first')} ~ {snap.get('last')}）——"
                               f"已达到 {need} 天的门槛，可以算 PE / 市值 / 换手率的 IC 了。")
                else:
                    st.progress(min(1.0, got / need),
                                text=f"估值截面快照积累进度：**{got} / {need}** 个交易日"
                                     f"（{snap.get('first') or '—'} ~ {snap.get('last') or '—'}）"
                                     f"，调度任务 `snapshot_flow` 每交易日 15:40 自动落库")


def render_robustness(payload: dict):
    """样本外稳健性结果：IC 矩阵 + 符号一致率判定"""
    if not payload:
        st.info("还没有结果。")
        return
    if payload.get("error"):
        st.error(f"❌ 分析失败：{payload['error']}")
        return

    p = payload.get("params") or {}
    periods = payload.get("periods") or []
    horizons = payload.get("horizons") or []
    st.caption(f"股票池 {p.get('codes')} 只（有效 {payload.get('names')}）｜"
               f"区间 {len(periods)} 段 × 持有期 {horizons} ｜"
               f"步长 {p.get('step')}｜分组 {p.get('groups')}")
    if payload.get("note"):
        st.warning(payload["note"])

    st.subheader("① 各配置的 IC 均值")
    if not payload.get("summary"):
        st.warning("没有算出结果 —— 可能是样本太少或区间太短。")
        return
    cells = payload.get("cells") or []
    for row in payload["summary"]:
        key = row["key"]
        st.markdown(f"**{row['因子']}** —— 方向假设：{row['方向假设']}")
        table = []
        for pr in periods:
            label = pr[0][:4] if pr[0][:4] == pr[1][:4] else f"{pr[0][:4]}~{pr[1][:4]}"
            line = {"区间": label}
            for h in horizons:
                hit = [c for c in cells if c["key"] == key and c["start"] == pr[0]
                       and c["end"] == pr[1] and c["持有期"] == h]
                if hit and hit[0]["IC均值"] is not None:
                    v = hit[0]["IC均值"]
                    line[f"持有{h}日"] = f"{'↑' if v > 0 else '↓'} {v:+.3f}"
                else:
                    line[f"持有{h}日"] = "—"
            table.append(line)
        st.dataframe(table, width="stretch", hide_index=True)

    st.subheader("② 符号一致性汇总（判定）")
    rows = []
    for row in payload["summary"]:
        pct = row.get("符号一致率")
        rows.append({
            "因子": row["因子"], "方向假设": row["方向假设"],
            "与假设一致": f"{row.get('一致')} / {row.get('有效配置')}",
            "一致率": (f"{pct:.0%}" if pct is not None else "—"),
            "平均 IC": row.get("平均IC"),
            "判定": row.get("判定"),
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(f"判据：一致率 ≥ {fa.ROBUSTNESS_AGREE_THRESHOLD:.0%} 才算稳健"
               f"（0.5 等于抛硬币，没有信息量）。"
               f"**稳健 ≠ 正确**——这里检验的是「符号在样本外还在不在」，"
               f"不是「改了权重就能赚钱」。")


# ================================================================
# 参数区
# ================================================================

if not POOL:
    st.error("股票池不可用（`data/backtest_pool.json`）。"
             "可先运行 `python scripts/build_backtest_pool.py` 重建，或改用其它池。")
    st.stop()

st.subheader("参数（默认值可直接跑）")

# 两种分析的区别要说人话，别让用户猜：
# 单区间 = "这段样本上方向如何"；稳健性 = "换区间/换持有期后符号还在不在"。
# 后者才是能下结论的那一步（本项目就吃过"单区间显著、样本外不显著"的亏）。
MODE_SINGLE = "单区间 IC（+分层/相关性）"
MODE_ROBUST = "样本外稳健性（多区间 × 多持有期）"
mode = st.radio("分析类型", [MODE_SINGLE, MODE_ROBUST], horizontal=True,
                key="fa_mode",
                help="先跑单区间看整体，再跑稳健性确认方向是不是稳定的 —— "
                     "单区间的「显著」不能直接采信。")
robust = (mode == MODE_ROBUST)

c1, c2, c3 = st.columns(3)
with c1:
    quick = st.checkbox("⚡ 快速模式（前 20 只）", value=True, key="fa_quick",
                        help="首次体验建议勾上；正式分析再取消。")
    if quick:
        max_n = min(20, len(POOL))
        st.caption(f"快速模式：用池内前 {max_n} 只")
    else:
        max_n = st.slider("用池内前 N 只", 10, len(POOL), len(POOL), step=10)
with c2:
    if robust:
        # 区间固定为 2023/2024/2025 三段：这三段要"覆盖不同市场形态"，
        # 让用户随意改反而容易选出三段高度重叠的区间，看不出稳健性。
        st.caption("区间（固定三段）："
                   + " / ".join(f"{a[:4]} 年" for a, _b in fa.DEFAULT_ROBUSTNESS_PERIODS))
        horizons = st.multiselect("持有期（交易日）", [5, 10, 20, 40, 60],
                                  default=list(fa.DEFAULT_ROBUSTNESS_HORIZONS),
                                  key="fa_hs",
                                  help="多个持有期一起看，才能发现「换个窗口符号就翻」的情况。")
    else:
        start = st.text_input("起始", "2024-01-01" if not quick else "2024-06-01",
                              key="fa_start")
        end = st.text_input("结束", "2025-06-30", key="fa_end")
with c3:
    if not robust:
        horizon = st.number_input("持有期（交易日）", 5, 60, 20, key="fa_h")
    step = st.number_input("采样步长（每 N 个交易日）", 1, 20,
                           (5 if robust else (5 if not quick else 3)), key="fa_step")
    groups = st.number_input("分层组数", 3, 10, 5, key="fa_g")

factor_labels = {k: v["label"] for k, v in fa.FACTORS.items()}
chosen = st.multiselect("要分析的因子", options=list(fa.FACTORS),
                        default=list(fa.FACTORS),
                        format_func=lambda k: factor_labels[k],
                        help="这一版只支持 K 线类因子；PE / 市值 / 换手率需要历史截面，"
                             "见页面底部的说明。")

codes = POOL[:int(max_n)]
st.caption(f"📊 股票池：{len(codes)} 只（`backtest_pool.json`，共 {len(POOL)} 只）。"
           + str(POOL_META.get("caveat", ""))[:110])

if robust:
    n_cfg = len(fa.DEFAULT_ROBUSTNESS_PERIODS) * max(1, len(horizons))
    run_btn = st.button(f"🚀 开始稳健性检验（{n_cfg} 个配置，约 2 分钟）",
                        type="primary", width="stretch")
else:
    run_btn = st.button("🚀 开始分析", type="primary", width="stretch")

if run_btn:
    if not chosen:
        st.error("至少选一个因子。")
    elif robust and not horizons:
        st.error("至少选一个持有期。")
    elif robust:
        ok = ui_tasks.start(TASK, _job_robustness(codes, chosen, horizons,
                                                  int(step), int(groups)),
                            {"mode": "robustness", "factors": list(chosen)})
        if not ok:
            st.error("已有分析在运行，本次点击已忽略。")
        else:
            st.rerun()
    elif not ui_tasks.start(TASK, _job(codes, chosen, start, end,
                                       int(horizon), int(step), int(groups)),
                            {"factors": list(chosen)}):
        st.error("已有分析在运行，本次点击已忽略。")
    else:
        st.rerun()

# ================================================================
# 进度与结果
# ================================================================

if ui_tasks.is_running(TASK):
    st.divider()
    st.info("⏳ 分析在**后台线程**里运行（要逐只拉日线，首次较慢）。刷新不会中断。")
    _live_progress()
else:
    ui_runs.clear_dead_running()
    saved = ui_runs.load_result(TASK)
    if saved:
        st.divider()
        st.caption(f"📌 结果生成于 {saved.get('generated_at', '')}"
                   f"（{ui_runs.age_text(saved)} 前）；刷新不会丢。")
        payload = saved.get("payload") or {}
        if payload.get("mode") == "robustness":
            render_robustness(payload)
        else:
            render_result(payload)
    else:
        st.info("👆 默认参数已就绪：**先跑「单区间 IC」看整体，再跑「样本外稳健性」"
                "确认方向稳不稳** —— 单区间的显著不能直接采信。")

# ================================================================
# 口径说明（放在最后，避免打断主流程）
# ================================================================

with st.expander("📖 这些指标怎么读（IC / ICIR / t 值 / 分层）", expanded=False):
    st.markdown("""
- **IC**：某日全市场「因子值排名」与「之后 N 日收益排名」的 **Spearman 秩相关**。
  IC > 0 说明因子值越大、后续收益越高（对"越小越好"的因子则相反）。
- **ICIR** = IC 均值 / IC 标准差。衡量**稳定性**——均值高但忽正忽负，ICIR 就低。
- **t 值** ≈ ICIR × √期数，|t| > 2 粗略视为显著（不是严格检验，够用即可）。
- **分层收益**：把股票按因子值分 N 组，看各组未来收益。**单调**才说明因子有区分度；
  不单调往往是"因子只在两端有效"或"样本噪声"。
- **因子相关性**：|ρ| > 0.7 的两个因子在加权时会重复计分，等效于偷偷加大了这个方向的权重。
""")
