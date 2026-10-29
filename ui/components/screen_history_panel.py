"""选股历史与命中率面板（选股页 / 自然语言选股页共用）。

单独成组件的原因与 core 层一样：**命中率口径只有一处**。
两个页面都想知道"这些条件到底有没有用"，若各写一遍，迟早出现
"这页说有效、那页说无效"的自相矛盾。

展示的三层信息（从粗到细）：
1. 整体：样本数 / T+5 平均收益 / 胜率 / **超额**收益（vs 同期上证）
2. 条件：每个条件的样本收益 vs 全样本基线（edge 为正 = 该条件有正向区分度）
3. 因子：因子得分与 T+5 的相关系数（近似 IC）

⚠️ 小样本时这些数字只是"参考方向"不是结论，面板会显式提示样本量门槛。
"""

import streamlit as st

from db import store as db_store
from core import screen_history as sh


def _pct(v, digits=2):
    if v is None:
        return "—"
    return f"{v * 100:.{digits}f}%"


def render(title: str = "📈 选股历史与命中率", recent_days: int = 180,
           show_conditions: bool = True, expanded: bool = False):
    """渲染选股历史面板。数据/存储异常时不崩，退化为一条说明。"""
    with st.expander(title, expanded=expanded):
        try:
            stats = db_store.screen_pick_stats(recent_days=recent_days)
        except Exception as e:
            st.caption(f"选股历史不可用: {e}")
            return

        if not stats.get("samples"):
            st.caption(
                "还没有可用于回看的样本：每次选股结束会把候选写进 `screen_picks` 表，"
                "T+5 / T+20 收益由每日收盘后的回填任务补上（与分析历史同一套状态机）。"
                "样本攒起来后，这里会显示胜率与**超额**收益。")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("样本数", f"{stats['samples']:,}")
        c2.metric("T+5 平均收益", _pct(stats.get("avg_ret_5d")))
        c3.metric("T+5 胜率", _pct(stats.get("win_rate_5d"), 0))
        c4.metric("T+5 超额收益", _pct(stats.get("avg_excess_5d")),
                  help="个股收益 − 同期上证收益。有基准的样本数：%d"
                       % (stats.get("excess_samples") or 0))
        if stats.get("pending_backfill"):
            st.caption(f"⏳ 还有 {stats['pending_backfill']} 条候选未到 T+20 回看窗口")
        if stats.get("samples", 0) < 30:
            st.caption("⚠️ 样本不足 30 条，下面的数字只能看方向，不能当结论。")

        if st.button("🔄 立即回填", key="btn_backfill_screen"):
            try:
                res = sh.backfill_screen_picks(limit=200)
                if res.get("locked"):
                    st.info("⏳ 已有回填任务在运行中…")
                else:
                    st.success(f"回填完成：更新 {res.get('updated', 0)} 条"
                               f"（done {res.get('done', 0)} / partial {res.get('partial', 0)}）")
                    st.rerun()
            except Exception as e:
                st.error(f"回填失败: {e}")

        # ---- 最近候选 ----
        try:
            rows = db_store.list_screen_picks(limit=20)
        except Exception:
            rows = []
        if rows:
            st.markdown("**最近候选**")
            st.dataframe(
                [{"选股日": r.get("pick_date"), "代码": r.get("stock_code"),
                  "名称": r.get("stock_name"), "评分": r.get("score"),
                  "名次": r.get("rank_in_run"),
                  "T+5": _pct(r.get("ret_5d")),
                  "超额": (_pct(r.get("ret_5d") - r.get("bench_ret_5d"))
                           if r.get("ret_5d") is not None
                           and r.get("bench_ret_5d") is not None else "—"),
                  "来源": r.get("source"), "状态": r.get("backfill_status") or "待回填"}
                 for r in rows],
                width="stretch", hide_index=True)

        # ---- 条件区分度（P2 的核心）----
        if show_conditions:
            try:
                cs = db_store.condition_stats(recent_days=recent_days, min_samples=3)
            except Exception:
                cs = None
            conds = (cs or {}).get("conditions") or []
            if conds:
                st.markdown("**条件区分度**（命中该条件的样本 vs 全样本基线）")
                st.dataframe(
                    [{"条件": c["condition"], "样本": c["samples"],
                      "T+5 平均": _pct(c.get("avg_ret_5d")),
                      "胜率": _pct(c.get("win_rate_5d"), 0),
                      "超额": _pct(c.get("avg_excess_5d")),
                      "区分度": _pct(c.get("edge_vs_base"))}
                     for c in conds],
                    width="stretch", hide_index=True)
                st.caption("区分度 = 该条件样本的平均收益 − 全样本平均收益。"
                           "为正说明这条条件有正向区分度；样本 <3 的条件不列出。")
            elif cs:
                st.caption("暂无「命中≥3 次」的条件样本——多用几次自然语言选股就有了。")

        # ---- 因子相关性 ----
        try:
            fs = db_store.factor_stats(recent_days=recent_days, min_samples=5)
        except Exception:
            fs = None
        factors = (fs or {}).get("factors") or []
        if factors:
            st.markdown("**因子与 T+5 的相关性**（近似 IC，|值| 越大越有效）")
            st.dataframe(
                [{"因子": f["factor"], "样本": f["samples"],
                  "平均得分": f.get("avg_score"), "IC": f.get("ic_ret_5d")}
                 for f in factors],
                width="stretch", hide_index=True)
            st.caption("数据缺失的因子不参与（缺失 ≠ 0 分）；样本 <5 的因子不列出。")
