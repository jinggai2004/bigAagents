# -*- coding: utf-8 -*-
"""实盘（`screen_picks`）绩效序列：把"选股 + 后续收益"折成月度超额。

为什么不用研究实验的 JSON
──────────────────────────
`output/组合AB实验_*.json` 是**一次性实验**的产物，文件不会自己更新；
拿它做"每日监控"等于监控一份历史快照 —— 策略偷偷失效了也看不出来。
真正要盯的是**线上每天都在产出的选股**：`screen_picks` 里每一条都带
`ret_5d/ret_20d` 与 `bench_ret_5d/bench_ret_20d`（T+N 回填任务写的）。

口径（三条，写清楚免得被数字骗）
────────────────────────────────
1. **按 pick_date 的月份归组**，取组内**均值**（等权）。
   一个月内多次选股会重叠，所以月度超额不是严格的独立样本 —— 只能看**趋势**，
   不能拿来做显著性检验（这是监控用的近似，不是研究结论）。
2. 只统计 `ret_{h}d` 与 `bench_ret_{h}d` **都非空**的行；覆盖率单独返回，
   覆盖率太低时**必须**让上层看到（否则"3 条样本算出来的超额"会被当成结论）。
3. `horizon` 默认 20（与项目 T+20 口径一致）；T+5 可用于**早期预警**，
   但噪声大，判据门槛要相应放宽（见 `alpha_decay` 的 `min_months`）。
"""
from __future__ import annotations

import logging
import sqlite3

import pandas as pd

_logger = logging.getLogger("live_performance")

DEFAULT_HORIZON = 20
MIN_ROWS_PER_MONTH = 3      # 单月少于这个条数，该月不参与（避免 1 条样本定结论）


def _connect():
    from db import store
    return store.get_connection()


def monthly_pick_excess(horizon: int = DEFAULT_HORIZON, *,
                        conn=None, rank_max: int | None = None) -> dict:
    """按选股月份汇总实盘超额。

    Args:
        horizon: 5 或 20（对应 `ret_5d/ret_20d`）。
        conn: 复用连接（测试注入）。
        rank_max: 只统计 `rank_in_run <= rank_max` 的（模拟 top-N 组合）。

    Returns:
        {ok, ret: Series(月), bench: Series(月), months, rows,
         coverage: {total, filled, pct}, reason?}
        - `ok=False` 时 `reason` 说明为什么不可用（不返空 Series 冒充 0）。
    """
    col, bcol = f"ret_{horizon}d", f"bench_ret_{horizon}d"
    own = conn is None
    c = conn or _connect()
    try:
        cols = [r[1] for r in c.execute("PRAGMA table_info(screen_picks)")]
        if col not in cols or bcol not in cols:
            return {"ok": False, "reason": f"screen_picks 缺列 {col}/{bcol}"}
        where = f"{col} IS NOT NULL AND {bcol} IS NOT NULL"
        params: tuple = ()
        if rank_max is not None:
            where += " AND rank_in_run <= ?"
            params = (int(rank_max),)
        q = (f"SELECT pick_date, {col} AS ret, {bcol} AS bench "
             f"FROM screen_picks WHERE {where}")
        rows = [dict(r) for r in c.execute(q, params)]
        total = int(c.execute("SELECT COUNT(*) FROM screen_picks").fetchone()[0])
    except sqlite3.Error as e:
        # 读库失败必须留痕：上层拿到的只是"ok=False"，排查时得能分清
        # "库坏了"与"还没回填"——两者的处置完全不同。
        _logger.warning("[live_performance] 读 screen_picks 失败: %s: %s",
                        type(e).__name__, e)
        return {"ok": False, "reason": f"读 screen_picks 失败: {type(e).__name__}: {e}"}
    finally:
        if own:
            c.close()

    if not rows:
        return {"ok": False, "reason": f"还没有任何 {col} 回填（T+{horizon} 需要等够交易日）",
                "coverage": {"total": total, "filled": 0, "pct": 0.0}}

    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["pick_date"]).dt.to_period("M")
    g = df.groupby("month").agg(ret=("ret", "mean"), bench=("bench", "mean"),
                                n=("ret", "size"))
    thin = g[g["n"] < MIN_ROWS_PER_MONTH]
    g = g[g["n"] >= MIN_ROWS_PER_MONTH]

    idx = g.index.to_timestamp("M")
    return {"ok": len(g) > 0,
            "ret": pd.Series(g["ret"].to_numpy(), index=idx),
            "bench": pd.Series(g["bench"].to_numpy(), index=idx),
            "months": int(len(g)),
            "rows": int(len(df)),
            "dropped_thin_months": int(len(thin)),
            "coverage": {"total": total, "filled": int(len(df)),
                         "pct": round(len(df) / max(total, 1), 3)},
            **({} if len(g) else {"reason": "所有月份样本都太薄（<3 条）"})}


def coverage_note(perf: dict, horizon: int = DEFAULT_HORIZON) -> str:
    """给人看的一句话（日志/推送共用，避免两处措辞不一致）。"""
    cov = perf.get("coverage") or {}
    return (f"T+{horizon}：已回填 {cov.get('filled', 0)}/{cov.get('total', 0)} 条"
            f"（{cov.get('pct', 0):.0%}），可用月份 {perf.get('months', 0)} 个")
