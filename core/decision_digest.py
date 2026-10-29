"""决策汇总（Decision Digest）：今日决策 / 质量评估 / 明日行动清单。

## 为什么需要这个模块（2026-10-10 工作流诊断）
用户的三点反馈都有实证支撑：**选股看不出质量、复盘不讲决策、明日没有操作建议**。
根因不是数据缺，而是**整条链路开环**——三类决策（选股 / 分析 / 交易）分散在三张表里，
**没有任何一处把它们组装起来**，于是：
- 复盘只能讲盘面（模型看不到"今天选了什么、分析了几只、模拟盘动没动"）；
- "明日操作建议"无从算起（三档建议的价位没有落成"距触发多远"）；
- "选股质量"始终无法回答（回填结果没人汇总成分档单调性 / 结论胜率）。

本模块把这三件事变成**只读的聚合查询**，供复盘报告与 UI 直接使用。

## 红线（详见 output/工作流改进详细设计_2026-10-10.md）
1. **只读本地库、零联网**（可以在页面每次渲染时调用）；
2. **决策数字不经 LLM**——模型只写"解读"，不改数，避免编造；
3. **缺数据必须显式**：`empty_reasons` / `status="insufficient_sample"` /
   `next_ready_date`，禁止用 0 或省略代替（"今天没选股"与"选股数据丢了"是两件事）；
4. 触及判定**只保留一份实现** → 复用 `analysis_history.evaluate_triggers`；
5. 每个子块 `try/except` 包住并带 `status`，**任何一块失败不影响其它块**。
"""

import logging
from datetime import datetime, timedelta

import db.store as store
from core.analysis_history import (_nth_trading_day, _parse_price,
                                   evaluate_triggers)

_logger = logging.getLogger(__name__)

# 低于这个样本数不给"结论性"统计（几只股票的均值只是噪声，不是质量证据）
MIN_SAMPLE = 3

# 持仓/观察价位的默认跟踪窗口（交易日）——与 `check_triggers_daily` 的 20 日窗口一致
DEFAULT_WATCH_DAYS = 30

_SCORE_BANDS = (("90+", 90, 1e9), ("80-90", 80, 90), ("70-80", 70, 80),
                ("60-70", 60, 70), ("<60", -1e9, 60))

# `advice_status` 缺失（旧数据）时按方向推断的兜底口径
_STATUS_BY_DIRECTION = (("看多", "可执行"), ("看空", "驳回"), ("观望", "等待"))


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _num(value):
    """宽松取数（含 "18.5元" / "12%" 这类字符串）；取不到返回 None（不返回 0）"""
    if isinstance(value, (int, float)):
        return float(value)
    return _parse_price(value)


def _price_map(snapshot_date: str = None) -> tuple:
    """当日截面 → (实际日期, {code: {"price","name","industry"}})。

    用 `market_snapshot`（15:40 落库）而非联网行情——复盘/页面会在没有网络的
    环境下渲染，这里必须**零联网**。
    """
    date, rows = store.load_market_snapshot(snapshot_date)
    out = {}
    for r in rows or []:
        out[r["stock_code"]] = {"price": r.get("price"), "name": r.get("stock_name"),
                                "industry": r.get("industry")}
    return date, out


def plan_view(plan: dict) -> dict:
    """交易计划的**展示口径**（三档建议 → 可读字段）。

    ⚠️ A股个人投资者无融券通路（2026-10-09 用户明确）：「看空」是一个判断，
    不是一笔交易——它的动作是**驳回买入 + 给入场条件**，价位应看 `watch_levels`。
    因此这里按 `advice_status` 区分：只有「可执行」档才把 entry/target/stop 当作
    可执行价位；驳回/等待档返回 `entry_conditions` 供"什么条件下重新考虑"。
    """
    plan = plan or {}
    direction = str(plan.get("direction") or "")
    status = str(plan.get("advice_status") or "")
    if not status:
        for key, val in _STATUS_BY_DIRECTION:
            if key in direction:
                status = val
                break
    rng = plan.get("entry_range")
    entry = None
    if isinstance(rng, (list, tuple)) and rng:
        nums = [n for n in (_num(x) for x in rng) if n is not None]
        if nums:
            entry = round(sum(nums) / len(nums), 3)
    conds = plan.get("entry_conditions")
    if isinstance(conds, str):
        conds = [conds] if conds.strip() else []
    elif isinstance(conds, (list, tuple)):
        conds = [str(c).strip() for c in conds if str(c).strip()]
    else:
        conds = []
    return {
        "direction": direction,
        "advice_status": status,
        "entry": entry,
        "target": _num(plan.get("target_price")),
        "stop": _num(plan.get("stop_loss")),
        "position_size": plan.get("position_size"),
        "entry_conditions": conds,
    }


def _index_return(start_date: str, end_date: str, symbol: str = "sh000001") -> dict:
    """指数在两个日期之间的收益。**必须带实际数据日期**。

    ⚠️ 返回 `pct=None` 的两种情况（"缺就是缺"，不能用 0 冒充"指数持平"）：
      1. 指数特征库取不到该区间（`daily_features` 只在被回填时才更新，
         近期日期常常还没有）；
      2. 两个端点落到**同一条** K 线（说明区间内根本没有新数据）。
    2026-10-10 实测踩到第 2 种：`2026-09-28 → 2026-10-09` 算出 0.0%，
    实际是特征库最新只到 09-28，两点取了同一根收盘。
    """
    out = {"symbol": symbol, "start": None, "end": None, "pct": None}
    if not start_date or not end_date:
        return out
    conn = store.get_connection()
    try:
        def _row(before: str):
            return conn.execute(
                "SELECT date, close FROM daily_features WHERE symbol = ? AND date <= ? "
                "ORDER BY date DESC LIMIT 1", (symbol, str(before)[:10])).fetchone()
        a, b = _row(start_date), _row(end_date)
    finally:
        conn.close()
    if not a or not b:
        return out
    out["start"], out["end"] = a["date"], b["date"]
    if a["date"] == b["date"] or not a["close"]:
        return out
    out["pct"] = round((b["close"] / a["close"] - 1) * 100, 2)
    return out


# ================================================================
# ① 今日决策清单
# ================================================================

# 推荐名单目录（`output/daily_picks`）。**测试可注入**：
# 默认 None ⇒ 运行时用 `app_root()/output/daily_picks`。
# ⚠️ 不这么做的话 `today_decisions()` 会读**真实**产物 ⇒ 单测读到今天的真实名单，
#    在空库用例里当场变成"5 只交通银行"（2026-10-29 实测踩到）。
DAILY_PICKS_DIR: str | None = None


def _picks_dir() -> str:
    if DAILY_PICKS_DIR:
        return DAILY_PICKS_DIR
    import os as _os
    from config.paths import app_root as _app_root
    # ⚠️ 用 `app_root()` 而不是 `__file__`：frozen 下 `__file__` 在只读的
    #    `_internal/` 里，`output/` 实际在 exe 同级。
    return _os.path.join(_app_root(), "output", "daily_picks")


def _load_daily_picks(date: str) -> list:
    """当日流水线的**推荐名单**（`output/daily_picks/daily_YYYYMMDD_*.json`，取最新一份）。

    为什么必须单独读它（2026-10-29 用户反馈"名单里 4 只银行"）
    ---------------------------------------------------------
    `screen_picks` 落的是**扩池候选**（`ranked`，纯分数排序、**未做行业分散**，
    一天 100~220 行），而 `candidates`（做过 `MAX_PER_SECTOR` 分散的那 5 只）
    只写在 `daily_picks/*.json` 里、**从不进 `screen_picks`**。

    于是复盘报告的"今日选股 N 只"读 `screen_picks` 时，拿到的是扩池排名：
    实测 2026-10-29 展示成"交通/农业/工商/建设 4 只银行"，
    而真实推荐名单只有 2 只银行（交通 + 农业）。**配额一直是对的，是报告读错了源。**

    Returns: `[{code, name, score, strategy, rank, batch_time, degraded, is_final}]`
             当日没有产物 ⇒ `[]`（调用方退回 `screen_picks` 并标注来源）。
    """
    import glob as _glob
    import json as _json
    import os as _os
    try:
        base = _picks_dir()
        stem = str(date).replace("-", "")[:8]
        files = sorted(_glob.glob(_os.path.join(base, f"daily_{stem}_*.json")))
        if not files:
            return []
        with open(files[-1], encoding="utf-8") as f:
            d = _json.load(f)
        if str(d.get("date"))[:10] != str(date)[:10]:
            return []
        out = []
        for i, p in enumerate(d.get("picks") or [], 1):
            code = str(p.get("股票代码") or "").strip()
            if not code:
                continue
            out.append({
                "code": code, "name": p.get("股票名称") or "",
                "score": p.get("score"),
                "coverage": p.get("score_coverage"),
                "strategy": p.get("策略") or "—",
                "rank": p.get("rank_in_run") or i,
                "batch_time": str(d.get("generated_at") or "")[:16],
                "degraded": bool(p.get("degraded")),
                "is_final": True,
            })
        return out
    except (OSError, ValueError, KeyError) as e:
        _logger.warning("[决策汇总] daily_picks 读取失败（退回 screen_picks）: %s", e)
        return []


def today_decisions(as_of: str = None) -> dict:
    """当日三类决策的统一视图（全部只读本地库，零联网）。

    Returns:
        {"date", "screens": [...], "screen_pool": [...], "analyses": [...],
         "trades": [...], "positions": [...], "empty_reasons": [...]}

    ⚠️ **`screens` 是推荐名单，`screen_pool` 是扩池候选**（2026-10-29 拆分）。
       原来只有一个 `screens` 同时承担两种语义：复盘要"今天推荐了哪几只"
       （5 只、做过行业分散），阿深的系统工具要"今天筛出了多大一个池子"
       （100+ 行、纯分数排名）。一个变量两类语义 ⇒ 复盘把扩池排名当名单展示，
       用户看到"4 只银行"而真实名单只有 2 只。**两个目的不能共用一份数据。**
    """
    date = (as_of or _today())[:10]
    out: dict = {"date": date, "screens": [], "screen_pool": [], "analyses": [],
                 "trades": [], "positions": [], "empty_reasons": []}

    # ── 选股 ──
    # ① 推荐名单（`daily_picks`，已做行业分散）——复盘/推送该用它
    out["screens"] = _load_daily_picks(date)
    # ② 扩池候选（`screen_picks`，纯分数排名）——阿深的"候选数"该用它
    try:
        conn = store.get_connection()
        try:
            # ⚠️ 必须按 stock_code 去重（2026-10-22）：一天有早/中/尾三次选股，
            #    扩池后每次落库 100+ 行 ⇒ 当日 `screen_picks` 有 348 行，
            #    不去重会把同一只票在"决策摘要"里列 3 遍（7 只时代重复不明显，
            #    扩池后被放大成 3 倍噪音）。
            #    ⚠️ 不能改用唯一索引解决：`screen_history.recent_top_overlap`
            #    依赖 `created_at` 分批比较重合度，唯一键会把它废掉。
            rows = conn.execute(
                "SELECT stock_code, stock_name, MAX(score) AS score, "
                "       MAX(score_coverage) AS score_coverage, "
                "       MAX(strategy) AS strategy, "
                "       MIN(COALESCE(rank_in_run, 999)) AS rank_in_run, "
                "       MAX(created_at) AS created_at "
                "  FROM screen_picks WHERE pick_date = ? "
                " GROUP BY stock_code "
                " ORDER BY rank_in_run, stock_code", (date,)).fetchall()
        finally:
            conn.close()
        for r in rows:
            d = dict(r)
            out["screen_pool"].append({
                "code": d["stock_code"], "name": d["stock_name"],
                "score": d["score"], "coverage": d["score_coverage"],
                "strategy": d["strategy"], "rank": d["rank_in_run"],
                "batch_time": (d.get("created_at") or "")[:16],
                # 降级标记：run() 的 degraded 未落库，但"内置精选池"这一策略名
                # 就是**候选域被收窄到 18 只硬编码**的直接证据（2026-10-09 定位）
                "degraded": str(d.get("strategy") or "") == "内置精选池",
                "is_final": False,
            })
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[决策汇总] 选股读取失败: %s", e)
    if not out["screens"] and out["screen_pool"]:
        # 当日还没跑流水线（只有早/中各轮的初筛落库）⇒ 用扩池前几名兜底，
        # 但**必须标出来**：这是"候选排名"不是"最终名单"。
        out["screens"] = [dict(x, is_final=False) for x in out["screen_pool"][:5]]
        out["empty_reasons"].append("当日无 daily_picks 产物 ⇒ 选股名单退化为扩池前 5 名")
    if not out["screens"]:
        out["empty_reasons"].append("今日无选股记录")

    # ── 深度分析（含同日同票的重复标记）──
    try:
        seen: dict = {}
        for r in store.list_analyses_since(date, limit=300):
            if str(r.get("analysis_date") or "")[:10] != date:
                continue
            pv = plan_view(r.get("trade_plan") or {})
            key = r.get("stock_code")
            seen[key] = seen.get(key, 0) + 1
            out["analyses"].append({
                "code": r.get("stock_code"), "name": r.get("stock_name"),
                "decision_label": r.get("decision_label"),
                "advice_status": pv["advice_status"],
                "entry": pv["entry"], "target": pv["target"], "stop": pv["stop"],
                "position_size": pv["position_size"],
                "entry_conditions": pv["entry_conditions"],
                "duplicate": seen[key] > 1,      # 同日同票的第 2+ 条
                "created_at": (r.get("created_at") or "")[:16],
            })
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[决策汇总] 分析读取失败: %s", e)
    if not out["analyses"]:
        out["empty_reasons"].append("今日无深度分析记录")

    # ── 模拟交易与持仓 ──
    try:
        state = store.load_paper_state() or {}
        for f in state.get("fills") or []:
            if str(f.get("date") or "")[:10] != date:
                continue
            out["trades"].append({
                "code": f.get("code"), "direction": f.get("direction"),
                "quantity": f.get("quantity"), "price": f.get("actual_price"),
                "amount": f.get("total"), "status": f.get("status", "filled"),
            })
        snap_date, prices = _price_map()
        for code, pos in (state.get("positions") or {}).items():
            last = (prices.get(code) or {}).get("price")
            qty = pos.get("quantity") or 0
            cost = pos.get("total_cost") or 0.0
            pnl = (last * qty - cost) if isinstance(last, (int, float)) else None
            out["positions"].append({
                "code": code, "name": (prices.get(code) or {}).get("name") or code,
                "quantity": qty, "avg_cost": round(cost / qty, 3) if qty else None,
                "last_price": last, "pnl": round(pnl, 2) if pnl is not None else None,
                "pnl_pct": (round(pnl / cost * 100, 2)
                            if pnl is not None and cost else None),
                "price_date": snap_date,
            })
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[决策汇总] 交易/持仓读取失败: %s", e)
    return out


# ================================================================
# ② 决策质量评估
# ================================================================

def _screen_quality(horizon: int) -> dict:
    """选股质量：分档单调性（分数高的组收益真的更高吗）+ TOP 与全体超额。"""
    col = f"ret_{horizon}d"
    # ⚠️⚠️ 必须按 (pick_date, stock_code) 去重（2026-10-24）：
    #   同一只票同一天会被 `pipeline` 多轮回填写多行（最多 9 行），按行统计等于
    #   **给"重复出现过的票"加权**。实测 857 行只有 401 条唯一记录 ⇒ 分档样本、
    #   全体超额、TOP5 超额全部失真（一次分析里"21 条"去重后只有 6 条 / 2 只票）。
    rows = [dict(r) for r in store.load_screen_picks_dedup(
        f"score, rank_in_run, {col} AS ret, bench_ret_{horizon}d AS bench",
        f"WHERE s.{col} IS NOT NULL")]
    pending = store.count_screen_picks_dedup(
        "WHERE s.backfill_status IS NULL OR s.backfill_status = 'partial'")
    conn = store.get_connection()
    try:
        # 待回填里**最新**的选股日 → 算它什么时候能出结果（"预计 X 日"）
        newest = conn.execute(
            """SELECT MAX(pick_date) FROM screen_picks WHERE backfill_status IS NULL
               OR backfill_status = 'partial'""").fetchone()[0]
    finally:
        conn.close()

    def _excess(r):
        return (r["ret"] - r["bench"]) if r.get("bench") is not None else None

    bands = []
    for name, lo, hi in _SCORE_BANDS:
        grp = [r for r in rows if r.get("score") is not None and lo <= r["score"] < hi]
        ex = [e for e in (_excess(r) for r in grp) if e is not None]
        bands.append({"band": name, "n": len(grp),
                      "avg_excess": round(sum(ex) / len(ex), 4) if ex else None,
                      "win_rate": (round(sum(1 for e in ex if e > 0) / len(ex), 2)
                                   if ex else None)})
    all_ex = [e for e in (_excess(r) for r in rows) if e is not None]
    top = [r for r in rows if (r.get("rank_in_run") or 99) <= 5]
    top_ex = [e for e in (_excess(r) for r in top) if e is not None]
    return {
        "n_backfilled": len(rows), "n_pending": int(pending or 0),
        "next_ready_date": (_nth_trading_day(str(newest)[:10], horizon)
                            if newest else None),
        "by_score_band": bands,
        "top5_excess": round(sum(top_ex) / len(top_ex), 4) if top_ex else None,
        "all_excess": round(sum(all_ex) / len(all_ex), 4) if all_ex else None,
        "status": "ok" if len(rows) >= MIN_SAMPLE else "insufficient_sample",
    }


def _analysis_quality(horizon: int) -> dict:
    """分析质量：各结论的平均收益与命中率（看多赚/看空跌才算对）、目标止损命中率。"""
    col = f"ret_{horizon}d"
    conn = store.get_connection()
    try:
        rows = [dict(r) for r in conn.execute(
            f"""SELECT decision_label AS label, {col} AS ret, hit_target, hit_stop
                FROM analyses WHERE decision_label IN ('看多', '看空', '观望')""").fetchall()]
        newest = conn.execute(
            """SELECT MAX(analysis_date) FROM analyses WHERE backfill_status IS NULL
               OR backfill_status = 'partial'""").fetchone()[0]
    finally:
        conn.close()

    by_label = []
    for label in ("看多", "看空", "观望"):
        grp = [r for r in rows if r["label"] == label and r["ret"] is not None]
        rets = [r["ret"] for r in grp]
        if label == "看多":
            hits = [r for r in grp if r["ret"] > 0]
        elif label == "看空":
            hits = [r for r in grp if r["ret"] < 0]
        else:
            hits = []
        by_label.append({
            "label": label, "n": len(grp),
            "avg_ret": round(sum(rets) / len(rets), 4) if rets else None,
            "hit_rate": round(len(hits) / len(rets), 2) if (rets and label != "观望") else None,
        })
    tgt = [r["hit_target"] for r in rows if r.get("hit_target") is not None]
    stop = [r["hit_stop"] for r in rows if r.get("hit_stop") is not None]
    n_done = sum(b["n"] for b in by_label)
    return {
        "by_label": by_label,
        "hit_target_rate": round(sum(tgt) / len(tgt), 2) if tgt else None,
        "hit_stop_rate": round(sum(stop) / len(stop), 2) if stop else None,
        "n_backfilled": n_done,
        "next_ready_date": (_nth_trading_day(str(newest)[:10], horizon)
                            if newest else None),
        "status": "ok" if n_done >= MIN_SAMPLE else "insufficient_sample",
    }


def _portfolio_quality() -> dict:
    """持仓归因：每只贡献多少盈亏 + 同期指数（超额才能说明选股/择时有价值）。"""
    state = store.load_paper_state()
    if not state:
        return {"status": "no_account", "note": "尚未创建模拟账户"}
    initial = float(state.get("initial_cash") or 0.0)
    cash = float(state.get("cash") or 0.0)
    nav = state.get("nav") or []
    snap_date, prices = _price_map()
    contributors, pos_value = [], 0.0
    for code, pos in (state.get("positions") or {}).items():
        qty = pos.get("quantity") or 0
        cost = pos.get("total_cost") or 0.0
        last = (prices.get(code) or {}).get("price")
        value = last * qty if isinstance(last, (int, float)) else cost
        pos_value += value
        contributors.append({
            "code": code, "name": (prices.get(code) or {}).get("name") or code,
            "value": round(value, 2),
            "pnl": round(value - cost, 2),
            "pnl_pct": round((value / cost - 1) * 100, 2) if cost else None,
        })
    total_value = cash + pos_value
    for c in contributors:
        # 权重 = 该持仓市值 / 总资产（不是"占总盈亏"，后者在盈亏接近 0 时无意义）
        c["weight"] = (round(c["value"] / total_value * 100, 1)
                       if total_value else None)
    nav_sorted = sorted(nav, key=lambda n: str(n.get("date") or ""))
    start_date = str(nav_sorted[0].get("date") or "")[:10] if nav_sorted else None
    return {
        "status": "ok",
        "initial_cash": initial, "cash": round(cash, 2),
        "total_value": round(total_value, 2),
        "total_pnl": round(total_value - initial, 2),
        "total_pnl_pct": round((total_value / initial - 1) * 100, 2) if initial else None,
        "contributors": sorted(contributors, key=lambda c: -abs(c["pnl"] or 0)),
        "since": start_date, "price_date": snap_date,
        # 同期指数：带实际数据日期（pct=None 表示取不到，不是"持平"）
        "benchmark": _index_return(start_date, snap_date) if start_date else None,
    }


def quality_assessment(as_of: str = None, horizon: int = 5,
                       recent_days: int = 180) -> dict:
    """决策质量评估。样本未到期时**明确返回状态与预计日期**，不返回空。

    Returns:
        {"as_of", "horizon", "screen": {...}, "analysis": {...}, "portfolio": {...}}
    """
    date = (as_of or _today())[:10]
    out: dict = {"as_of": date, "horizon": horizon}
    for key, fn in (("screen", lambda: _screen_quality(horizon)),
                    ("analysis", lambda: _analysis_quality(horizon)),
                    ("portfolio", _portfolio_quality)):
        try:
            out[key] = fn()
        except Exception as e:                  # noqa: BLE001
            _logger.warning("[决策汇总] 质量评估 %s 失败: %s", key, e)
            out[key] = {"status": "error", "error": str(e)}
    return out


# ================================================================
# ③ 明日行动清单
# ================================================================

def _sector_outflow_risks(limit: int = 3) -> list:
    """行业资金**连续两日**净流出 → 风险提示（本地快照，零联网）。"""
    conn = store.get_connection()
    try:
        dates = [r["d"] for r in conn.execute(
            "SELECT DISTINCT snapshot_date AS d FROM industry_flow_snapshot "
            "ORDER BY snapshot_date DESC LIMIT 2").fetchall()]
        if len(dates) < 2:
            return []
        latest, prev = dates[0], dates[1]
        cur = {r["industry"]: r["net_amount"] for r in conn.execute(
            "SELECT industry, net_amount FROM industry_flow_snapshot "
            "WHERE snapshot_date = ?", (latest,)).fetchall()}
        old = {r["industry"]: r["net_amount"] for r in conn.execute(
            "SELECT industry, net_amount FROM industry_flow_snapshot "
            "WHERE snapshot_date = ?", (prev,)).fetchall()}
    finally:
        conn.close()
    out = []
    for ind, v in cur.items():
        if v is not None and v < 0 and old.get(ind) is not None and old[ind] < 0:
            out.append({"kind": "sector_outflow",
                        "text": f"{ind} 连续两日资金净流出"
                                f"（{prev} {old[ind] / 1e8:.2f}亿 → {latest} {v / 1e8:.2f}亿）"})
    out.sort(key=lambda x: x["text"])
    return out[:limit]


def action_list(as_of: str = None, watch_days: int = DEFAULT_WATCH_DAYS) -> dict:
    """明日行动清单：把三档建议的价位变成"当前距触发多远"。

    这是"明日操作建议"的**数据来源**——由真实价位与最新收盘算出，
    而不是让模型再编一段话。

    Returns:
        {"as_of", "positions": [...], "watch": [...], "risks": [...]}
    """
    date = (as_of or _today())[:10]
    snap_date, prices = _price_map()
    out: dict = {"as_of": date, "price_date": snap_date,
                 "positions": [], "watch": [], "risks": []}

    # 最近 `watch_days` 天、有价位的分析（用作持仓风控位 / 观察档的入场条件来源）
    cutoff = (datetime.now() - timedelta(days=watch_days)).strftime("%Y-%m-%d")
    try:
        analyses = store.list_analyses_since(cutoff, limit=200)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[决策汇总] 行动清单读取分析失败: %s", e)
        analyses = []
    latest_plan: dict = {}
    for r in analyses:                          # 已按 id DESC，首见即最新
        code = r.get("stock_code")
        if code and code not in latest_plan:
            pv = plan_view(r.get("trade_plan") or {})
            # 带上分析记录里的名称：快照没覆盖的票（新股/停牌）只有这里能拿到名字，
            # 否则观察清单里会出现"603127（603127）"这种代码当名字的条目。
            pv["name"] = r.get("stock_name") or ""
            latest_plan[code] = pv

    # ── 持仓：距止损 / 目标的距离，及是否已触发 ──
    try:
        state = store.load_paper_state() or {}
        for code, pos in (state.get("positions") or {}).items():
            last = (prices.get(code) or {}).get("price")
            pv = latest_plan.get(code) or {}
            stop = pv.get("stop") if pv.get("advice_status") == "可执行" else None
            target = pv.get("target") if pv.get("advice_status") == "可执行" else None
            item = {"code": code, "name": (prices.get(code) or {}).get("name") or code,
                    "quantity": pos.get("quantity"), "last": last,
                    "stop": stop, "target": target, "triggered": None}
            if isinstance(last, (int, float)) and (stop or target):
                # 复用**唯一**的触及判定（`evaluate_triggers`）：把"最新收盘价"当作
                # 单点窗口。多头口径下即 last<=stop / last>=target，
                # 与回填、每日核对完全同源（本项目多次踩"两侧各注一份"）。
                res = evaluate_triggers("看多", target, stop, [last], [last])
                if res.get("hit_stop"):
                    item["triggered"] = "stop"
                elif res.get("hit_target"):
                    item["triggered"] = "target"
                if stop:
                    item["to_stop_pct"] = round((stop / last - 1) * 100, 2)
                if target:
                    item["to_target_pct"] = round((target / last - 1) * 100, 2)
            if stop is None and target is None:
                item["note"] = "该持仓没有可执行档的分析计划（无止损/目标位）"
            out["positions"].append(item)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[决策汇总] 行动清单持仓失败: %s", e)

    # ── 观察档（驳回 / 等待）：给出"距入场条件多远" ──
    for code, pv in latest_plan.items():
        if pv.get("advice_status") not in ("驳回", "等待"):
            continue
        last = (prices.get(code) or {}).get("price")
        entry = pv.get("entry")
        nm = pv.get("name") or (prices.get(code) or {}).get("name") or code
        item = {"code": code, "name": nm,
                "direction": pv.get("direction"), "advice_status": pv.get("advice_status"),
                "entry": entry, "last": last,
                "entry_conditions": pv.get("entry_conditions")}
        if isinstance(last, (int, float)) and entry:
            item["to_entry_pct"] = round((entry / last - 1) * 100, 2)
        out["watch"].append(item)

    try:
        out["risks"] = _sector_outflow_risks()
        if not out["risks"]:
            out["empty_reasons"] = ["行业资金快照不足两天或无明显连续净流出"]
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[决策汇总] 行动清单风险块失败: %s", e)
    return out


def build_digest(as_of: str = None, horizon: int = 5) -> dict:
    """一次性聚合三块（复盘报告与页面共用入口）。

    三块各自独立容错：某块失败只影响它自己（调用方按 `status`/空值展示）。
    """
    out = {}
    for key, fn in (("today", lambda: today_decisions(as_of)),
                    ("quality", lambda: quality_assessment(as_of, horizon)),
                    ("actions", lambda: action_list(as_of))):
        try:
            out[key] = fn()
        except Exception as e:                  # noqa: BLE001
            _logger.warning("[决策汇总] %s 失败: %s", key, e)
            out[key] = {"status": "error", "error": str(e)}
    return out
