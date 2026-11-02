# -*- coding: utf-8 -*-
"""在线 Brinson 归因装配层（把 `core.attribution` 接上真实数据）。

回答的问题
──────────
"选股组合的超额收益，是**押对了行业**（配置效应），还是**在行业内选对了股**（选股效应）？"

`core.attribution` 早把三效应与 Carino 多期链接算对了（测试手算锁死），
但**没有任何调用方**（2026-10-30 核对：grep 只有定义与测试）——模块是死代码。
本模块负责把它的入参从本地面板装配出来。

数据来源（全部本地，零联网）
────────────────────────────
- 组合：`screen_picks`（当日选股名单）· **等权**，持有到**下一个有选股记录的交易日**
- 基准：同期**全市场等权**（面板内两个端点都可交易的票）
- 收益：`data/panel/` 的 close/preclose 逐日收益**复利**（含除权除息调整）
- 行业：`data/industry_swa.py` 申万 L1（5,220 只）；缺失回退快照 industry_ts

为什么不用 `market_snapshot` 当数据源（踩过）
──────────────────────────────────────────────
实测快照的**覆盖率中途会变**：2026-09-30 ~ 10-23 每天只有 **2,617~2,802** 只
（缺创业板/科创板/北交所），10-26 起才 5,572 只。拿它做基准，等于 16 个期里
前 13 期和最后 3 期的基准是**两个不同的市场**，Carino 链接会把它们串成一个
假的总超额。而面板 `daily_overlay` + `parts` 一直覆盖 5,564~5,572 只，
用 pyarrow 谓词下推只读窗口内行（实测 **0.05s / 10 天 / 5,572 只**）。

口径纪律（写进返回值，不靠文档口口相传）
────────────────────────────────────────
1. **为什么基准不是沪深300**：项目里只有沪深300 的**指数点位**（`data/index_bars`），
   没有**成分股权重**。Brinson 必须有基准的逐票权重；权重拿不到就只能换基准。
   等权全市场是唯一"权重可得且可复核"的选择 ⇒ 配置效应 = "相对全市场等权，
   超配/低配了哪些行业"，**不是**"相对沪深300"。这个差别必须写在输出里。
2. **期的边界 = 两个"有选股记录"的交易日之间**：不按自然日切，也不给没有选股
   的日子里编一个组合。持有期收益按**逐日复利**算，所以跨周末/长假天然正确
   （快照那套用 `change_pct` 会把 3 天收益少算成 1 天）。
3. **可交易过滤**：两个端点都要 `volume>0` 且有 close/preclose；停牌票不进组合、
   也不进基准（否则基准里塞一批 0 收益票，等于压低基准）。
4. **入选但取不到收益的票**不静默丢：计进 `dropped_picks`。
5. **行业缺失**进 `__未分类__` 桶参与分解（`core.attribution` 原生行为），
   `unclassified_ratio` 用**当期基准股数**做分母——分母用累计集会让比值 >1。
6. **前视偏差**：行业归属用**当前**申万快照，不是历史成分 ⇒ `snapshot_lookahead=True`。
   这是快照归因，不是精确归因。
"""
from __future__ import annotations

import glob
import logging
import os
import sqlite3

from config.paths import data_root

from core import attribution as _att

_logger = logging.getLogger(__name__)

# 一个周期至少要几只票才有归因意义（1 只票的"选股效应"恒为 0，纯噪声）
MIN_PICKS_PER_PERIOD = 3
# 行业明细返回条数上限（按 |总效应| 排序）
TOP_INDUSTRIES = 12
# 期明细返回上限（防把几十期全塞进响应）
MAX_PERIODS_OUT = 40

_PANEL_COLS = ("stock_code", "trade_date", "close", "preclose", "volume")


def _panel_dir() -> str:
    return os.path.join(data_root(), "data", "panel")


def _connect():
    """复用项目连接入口（测试可 monkeypatch `db.models.DB_PATH` 指到临时库）。"""
    from db.models import get_connection
    return get_connection()


# ================================================================
# 数据装配
# ================================================================

def load_daily(start: str, end: str) -> "object":
    """读窗口内的全市场日线（面板 parts + 日更 overlay），返回 DataFrame。

    两种分片布局**都读**：`parts/` 里既有按日期切的（0015/0016，全市场 × 一段
    日期区间），也有按股票切的（200 只 × 全历史）。只读一种会缺数据。
    用 pyarrow 谓词下推，不在窗口内的分片几乎零成本（实测每个 0.05s）。
    `daily_overlay` 是日更增量，**后合并并去重保留它**（它比 parts 新）。
    """
    import pandas as pd

    base = _panel_dir()
    candidates = (sorted(glob.glob(os.path.join(base, "parts", "*.parquet")))
                  + sorted(glob.glob(os.path.join(base, "daily_overlay", "*.parquet"))))
    frames, skipped = [], []
    for i, path in enumerate(candidates):
        try:
            df = pd.read_parquet(path, columns=list(_PANEL_COLS),
                                 filters=[("trade_date", ">=", str(start)),
                                          ("trade_date", "<=", str(end))])
        except (OSError, ValueError, KeyError) as e:
            # 单个分片坏/缺列不该让整个归因失败，但必须留痕（不静默）
            _logger.warning("[Brinson] 分片 %s 读取失败（跳过）: %s: %s",
                            os.path.basename(path), type(e).__name__, e)
            skipped.append(os.path.basename(path))
            continue
        if df is None or df.empty:
            continue
        df["__src_order"] = i                      # overlay 的 i 更大 ⇒ 去重时保留它
        frames.append(df)
    if not frames:
        return pd.DataFrame(), skipped
    all_df = pd.concat(frames, ignore_index=True)
    all_df["trade_date"] = all_df["trade_date"].astype(str)
    all_df["stock_code"] = all_df["stock_code"].astype(str).str.strip()
    all_df = (all_df.sort_values("__src_order")
              .drop_duplicates(subset=["stock_code", "trade_date"], keep="last")
              .drop(columns=["__src_order"]))
    return all_df, skipped


def _load_picks(conn, start: str, end: str, source: str | None = None) -> dict:
    """读选股名单 → {pick_date: [code, ...]}（同日同票已在库层唯一，见 2026-10-29 迁移）。

    `source` 可筛 'screen'（选股页手动跑）/ 'pipeline'（调度器盘后全流程）。
    ⚠️ 两者**不是同一件事**：实测同一天可能都有记录，混在一起等于把
    "用户手点的选股"和"调度器的候选池"当同一个策略。
    """
    out: dict[str, list[str]] = {}
    sql = ("SELECT pick_date, stock_code FROM screen_picks "
           "WHERE pick_date >= ? AND pick_date <= ?")
    params: list = [str(start), str(end)]
    if source:
        sql += " AND source = ?"
        params.append(str(source))
    sql += " ORDER BY pick_date"
    try:
        for row in conn.execute(sql, params):
            d = str(row["pick_date"])
            code = str(row["stock_code"] or "").strip()
            if code:
                out.setdefault(d, []).append(code)
    except sqlite3.Error as e:
        _logger.warning("[Brinson] 读取 screen_picks 失败（按无选股处理）: %s", e)
        return {}
    return out


def _load_industry(conn) -> tuple[dict, str]:
    """行业映射：申万 L1 优先（`attribution` 文档指定的口径），回退快照 industry_ts。"""
    try:
        from data import industry_swa
        swa = industry_swa.load_industry_swa()
    except (ImportError, OSError, ValueError) as e:
        _logger.warning("[Brinson] 申万行业读取失败，回退快照: %s: %s", type(e).__name__, e)
        swa = None
    if swa and swa.get("industry"):
        return dict(swa["industry"]), str(swa.get("snapshot_date") or "申万快照")
    # 回退：快照里按 code 取**最近一次**非空行业（industry_ts 优先）
    ind: dict[str, str] = {}
    try:
        for row in conn.execute(
                "SELECT stock_code, industry_ts, industry FROM market_snapshot "
                "WHERE (industry_ts IS NOT NULL AND industry_ts <> '') "
                "   OR (industry IS NOT NULL AND industry <> '') "
                "ORDER BY snapshot_date"):
            code = str(row["stock_code"] or "").strip()
            if not code:
                continue
            val = (row["industry_ts"] or "").strip() or (row["industry"] or "").strip()
            if val:
                ind[code] = val
    except sqlite3.Error as e:
        _logger.warning("[Brinson] 快照行业读取失败（行业全缺，全部进未分类）: %s", e)
        return {}, ""
    return ind, "market_snapshot 最近一期"


def _compound_return(wide_ret, wide_ok, d0: str, d1: str):
    """d0（不含）到 d1（含）的区间收益：逐日复利，要求期内每天都可交易。

    返回 (returns: Series, n_days)。任一天不可交易/缺价的票被剔除（不补 0）。
    """
    dates = [d for d in wide_ret.index if d0 < d <= d1]
    if not dates:
        return None, 0
    sub_ret = wide_ret.loc[dates]
    sub_ok = wide_ok.loc[dates]
    alive = sub_ok.all(axis=0) & sub_ret.notna().all(axis=0)
    if not bool(alive.any()):
        return None, len(dates)
    grown = (1.0 + sub_ret.loc[:, alive]).prod(axis=0) - 1.0
    return grown, len(dates)


# ================================================================
# 主入口
# ================================================================

def run(start: str | None = None, end: str | None = None,
        min_picks: int = MIN_PICKS_PER_PERIOD, source: str | None = None,
        min_names: int = 0) -> dict:
    """跑一遍在线 Brinson 归因。

    Args:
        start/end: 选股日期区间（YYYY-MM-DD，含端点）；None = screen_picks 全量
        min_picks: 单期最少可取收益选股数（不足则该期剔除并进 `skipped_periods`）
        source: 'screen' / 'pipeline' / None=全部（两者不是同一策略，见 `_load_picks`）
        min_names: 只保留组合规模 ≥ 该值的期（0=不过滤）。用于剔除规模跃变段

    Returns:
        {"ok": True, ...} 或 {"ok": False, "code": "missing"|"empty", "reason": ...}
    """
    conn = _connect()
    try:
        span = conn.execute(
            "SELECT MIN(pick_date) a, MAX(pick_date) b FROM screen_picks").fetchone()
        picks_start = str(start or (span["a"] if span else "") or "")
        picks_end = str(end or (span["b"] if span else "") or "")
        if not picks_start or not picks_end:
            return {"ok": False, "code": "missing",
                    "reason": "screen_picks 为空：还没有选股历史，无组合可归因"}
        picks = _load_picks(conn, picks_start, picks_end, source)
        industry, industry_source = _load_industry(conn)
    except sqlite3.Error as e:
        _logger.warning("[Brinson] 读库失败: %s", e)
        return {"ok": False, "code": "empty", "reason": f"读库失败: {e}"}
    finally:
        conn.close()

    if not picks:
        return {"ok": False, "code": "empty",
                "reason": f"{picks_start}~{picks_end} 内没有选股记录"
                          f"（source={source or '全部'}；熊市跳过/未跑都算）"}

    daily, bad_parts = load_daily(picks_start, picks_end)
    if daily is None or daily.empty:
        return {"ok": False, "code": "missing",
                "reason": "本地面板在该区间没有日线（先更新面板）",
                "bad_parts": bad_parts}

    import pandas as pd
    daily = daily[(daily["close"] > 0) & (daily["preclose"] > 0)].copy()
    daily["ret"] = daily["close"] / daily["preclose"] - 1.0
    # 用 int8 而不是 bool：pivot 会让 bool 列退化成 object（缺格填 NaN），
    # 之后每次 fillna 都发 FutureWarning 且将来语义会变。
    daily["ok"] = (daily["volume"].fillna(0) > 0).astype("int8")
    wide_ret = daily.pivot(index="trade_date", columns="stock_code", values="ret")
    wide_ok = daily.pivot(index="trade_date", columns="stock_code", values="ok")
    wide_ok = wide_ok.fillna(0).astype(bool)
    wide_ret = wide_ret.sort_index()
    wide_ok = wide_ok.sort_index()
    panel_dates = list(wide_ret.index)

    # 期边界 = 两个"有选股记录"的交易日之间（且该日在面板里）
    pick_dates = sorted(d for d in picks if d in set(panel_dates))
    if len(pick_dates) < 2:
        return {"ok": False, "code": "empty",
                "reason": f"有效的选股日期只有 {len(pick_dates)} 个（需 ≥2 才能算区间收益）"}

    periods, skipped, last_bench_n = [], [], 0
    for d0, d1 in zip(pick_dates, pick_dates[1:]):
        picked = picks.get(d0) or []
        grown, n_days = _compound_return(wide_ret, wide_ok, d0, d1)
        if grown is None:
            skipped.append({"date": d0, "reason": "区间内没有可交易标的",
                            "picks": len(picked), "priced": 0})
            continue
        rets = grown.to_dict()
        priced = [c for c in picked if c in rets]
        if len(priced) < min_picks:
            skipped.append({"date": d0, "reason": f"可取收益的选股不足 {min_picks} 只",
                            "picks": len(picked), "priced": len(priced)})
            continue
        if min_names and len(priced) < min_names:
            skipped.append({"date": d0, "reason": f"组合规模 {len(priced)} < min_names={min_names}",
                            "picks": len(picked), "priced": len(priced)})
            continue
        w_p = 1.0 / len(priced)
        w_b = 1.0 / len(rets)
        periods.append({
            "date": d0, "next_date": d1, "n_days": n_days,
            "portfolio": {c: w_p for c in priced},
            "benchmark": {c: w_b for c in rets},
            "returns": rets,
            "dropped_picks": sorted(set(picked) - set(priced)),
        })
        last_bench_n = len(rets)

    if not periods:
        return {"ok": False, "code": "empty",
                "reason": "没有一期可用（选股与面板无交集）",
                "skipped_periods": skipped}

    res = _att.brinson_attribution(
        [{"date": p["date"], "portfolio": p["portfolio"],
          "benchmark": p["benchmark"], "returns": p["returns"]} for p in periods],
        industry, snapshot_date=industry_source)

    linked = res.get("linked") or {}
    total = linked.get("total") or {}
    alloc = sum(float(v.get("alloc") or 0) for v in total.values())
    select = sum(float(v.get("select") or 0) for v in total.values())
    inter = sum(float(v.get("inter") or 0) for v in total.values())
    excess = float(linked.get("excess") or 0)

    ranked = sorted(total.items(),
                    key=lambda kv: abs(float(kv[1].get("total") or 0)), reverse=True)
    top_industries = [{"industry": k,
                       **{kk: v.get(kk) for kk in ("alloc", "select", "inter", "total")}}
                      for k, v in ranked[:TOP_INDUSTRIES]]

    period_brief = [{
        "date": p["date"], "next_date": p["next_date"], "days": p["n_days"],
        "portfolio_return": e.get("portfolio_return"),
        "benchmark_return": e.get("benchmark_return"),
        "excess": e.get("excess"),
        "n_portfolio": len(p["portfolio"]),
        "n_benchmark": len(p["benchmark"]),
        "dropped_picks": len(p["dropped_picks"]),
    } for e, p in zip(res.get("periods") or [], periods)][:MAX_PERIODS_OUT]

    unclassified = len((res.get("degraded") or {}).get("unclassified_codes") or [])

    # ── 规模画像与警示 ──
    # 实测 2026-10-30：`screen_picks` 每日只数 09-25~10-21 是 2~13 只，
    # 10-22 起跃变到 110~177 只（全市场域扩到 5,572 只后选股产出随之变大）。
    # 把两种规模串成一个 Carino 序列 = 把两个策略混成一个结论 ⇒ 必须显式警示。
    sizes = sorted(len(p["portfolio"]) for p in periods)
    size_median = sizes[len(sizes) // 2] if sizes else 0
    warnings: list[str] = []
    if sizes and sizes[-1] > 3 * max(1, sizes[0]):
        warnings.append(
            f"组合规模在期间内从 {sizes[0]} 只变到 {sizes[-1]} 只（>3×）："
            f"前后不是同一个策略口径，合计超额应看作混合结果。"
            f"可用 min_names 参数只保留某一规模段。")
    if size_median and size_median < 20:
        warnings.append(
            f"组合规模中位数仅 {size_median} 只：超额高度依赖少数个股，"
            f"是**诊断性归因**，不能当作策略有效性证据。")
    if (res.get("degraded") or {}).get("unclassified_codes"):
        warnings.append("部分标的缺申万行业，已进 __未分类__ 桶参与分解（不是被丢弃）")

    return {
        "ok": True,
        "source": source or "全部",
        "periods_used": len(periods),
        "periods_range": [periods[0]["date"], periods[-1]["next_date"]],
        "skipped_periods": skipped,
        "summary": {
            "portfolio_return": linked.get("portfolio_return"),
            "benchmark_return": linked.get("benchmark_return"),
            "excess": excess,
            "alloc": round(alloc, 8),
            "select": round(select, 8),
            "inter": round(inter, 8),
            # 恒等式 Σ(三效应) == 超额；非 0 说明归因算错，页面直接显示
            "identity_gap": round(excess - (alloc + select + inter), 10),
            "linked": linked is not None,
        },
        "size_profile": {"min": sizes[0] if sizes else 0,
                         "median": size_median,
                         "max": sizes[-1] if sizes else 0},
        "warnings": warnings,
        "top_industries": top_industries,
        "periods": period_brief,
        "industry_snapshot": res.get("industry_snapshot"),
        "industry_source": industry_source,
        "industry_coverage": len(industry),
        "unclassified_ratio": (round(unclassified / last_bench_n, 4)
                               if last_bench_n else None),
        "degraded_periods": linked.get("degraded_periods") or [],
        "bad_parts": bad_parts,
        "caliber": {
            "portfolio": "screen_picks 当日名单 · 等权，持有到下一个有选股记录的交易日",
            "benchmark": "面板内该期首尾都可交易的**全市场**等权",
            "benchmark_note": ("沪深300 只有指数点位、没有成分权重 ⇒ 基准用全市场等权；"
                               "因此配置效应=相对全市场等权的行业超低配，不是相对沪深300"),
            "return": "close/preclose 逐日收益复利（含除权除息调整），跨周末/长假正确",
            "industry": "申万 L1 快照（data/industry_swa.py），缺则 __未分类__",
            "tradable_filter": "两个端点 volume>0 且有 close/preclose",
            "snapshot_lookahead": True,
            "lookahead_note": "行业归属用当前申万快照而非历史成分 ⇒ 快照归因，非精确归因",
        },
    }
