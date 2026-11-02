# -*- coding: utf-8 -*-
"""交易拥挤度闸门（2026-10-30）

## 一句话
同一天里，**换手率与成交额都排在最前 10%、且已经涨了 20 日涨幅前 20%** 的股票，
未来 5 日跑输大盘。所以它是一条**减法闸门**（不新开仓），不是一个评分因子。

## 口径（11 年全市场验证，产物 `output/ui_runs/crowding_gate_11y.json`）
    crowd_trade = mean(pct(换手率), pct(log 成交额))     # 行业内分位，1 = 最拥挤
    mom20       = pct(20 日涨幅)                         # 行业内分位，1 = 涨最多
    闸门：crowd_trade >= 0.90 且 mom20 >= 0.80 ⇒ 不新开仓

## 为什么拆成两个而不合成一个加权分
双排序实测效应是**交互**的：高波动 + 拥挤 −0.97pp、低波动 + 拥挤 −0.33pp。
线性加权会把交互项平均掉（横截面正交化后效应归零就是同一机制的两种表现）。
知乎那篇 `R = 0.25A + 0.35P + 0.25V + 0.15M` 抓不住它，原因也在这里。

## 为什么是闸门不是因子
价量类因子在 11 年里 IC 全负（ivol20 −0.083 / max20 −0.074 / ret20 −0.059，均 11/11 年）。
"拥挤度越高 ⇒ 未来越差"意味着**乘法方向与打分方向相反**：进了 `FACTOR_WEIGHTS` 就会
变成"越拥挤分数越高"，正好选到该回避的那一端。`core/sentiment_reference.py` 已用
AST 断言把"情绪类不进评分链"锁死，本模块遵守同一条纪律（见
`tests/test_crowding.py::test_not_in_factor_weights`）。

## 文献
- Baker & Stein (2004, JFM 7:271-299)：高换手/高流动性预测后续低回报；
  机制是卖空约束下非理性投资者主导 —— D10 跑输的理论解释。
- Brown, Howard & Lundblad (2022, RFS 35:3231-3271)：拥挤度的原义是**持仓拥挤**
  （机构持仓 ÷ 日均成交额）。本模块是它的**交易维度**映射，不是同一个量。

## 数据来源与"尺子"纪律
- 换手率/成交额：`market_snapshot`（新浪口径百分数，5,572 只，单日一条 SQL，毫秒级）。
  ⚠️ 已核对与 `data/panel/turn_share/`（历史回填层）同一天 5,565 只：比值中位 **1.0**、
     差异中位 **0.0pp**、相关系数 **1.0** ⇒ 两源同尺，可跨历史与当日拼接。
- 20 日涨幅：`data/bars.get_bars(adjust="none")`（唯一取数入口，自动叠加日增量覆盖层）。
- 行业：**申万一级 31 类**（`data/panel/industry_swa_l1.parquet`）。
  ⚠️ 这里刻意**不用**选股链的 `industry_ts`（tushare 111 类）：112 类平均每组只有
     50 只，分位是噪声；且申万是唯一能同时覆盖 11 年验证与当日生产的行业层
     ——**验证与生产必须同一把尺子**，否则阈值 0.9 是在另一种刻度上标定的。
     同理不用新浪 49 类（只覆盖 2020 前老股，实测空行业组首根 K 线中位 2020-09-16）。

## 存储
`data/panel/crowd/{date}.parquet`（列 `code, crowd_trade, mom20, turn, amount, ret20,
industry`）+ `{date}.json`（meta/统计）。预计算落表而不是每次现算：单日全市场截面
实测 **32s**，放进 695s 的选股任务里每次重算是浪费，放进页面更会卡死 UI。
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Iterable, Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# ── 闸门参数（阈值来自 11 年分位实测，不是拍脑袋）────────────────────────
# 0.90 = 当日全市场行业内前 10% 拥挤；0.80 = 20 日涨幅前 20%。
# ⚠️ 改这两个数等于改一个已验证过的策略，必须重跑 `scripts/crowding_gate_11y.py`。
GATE_CROWD_PCT = 0.90
GATE_MOM_PCT = 0.80
# 行业内分位的最小分组。申万 31 类平均每组 ~168 只，20 是"还能排分位"的下限。
MIN_GROUP = 20
# 20 日涨幅需要 21 根日线（t 与 t-20）
MOM_LOOKBACK = 20
# 快照允许的最大滞后（交易日）。1 天是**正常**链路（16:30 产出 → 次日 09:25 消费）；
# 再多说明定时任务漏跑了，此时"用一周前的拥挤度砍今天的候选"比不用更糟。
MAX_SNAPSHOT_LAG_DAYS = 5

_SW_PATH = None          # 延迟解析（测试可注入）
_SW_CACHE: dict = {"stamp": None, "map": {}}


# ================================================================
# 1. 纯函数：分位
# ================================================================

def pct_rank(values: dict) -> dict:
    """{code: 数值} → {code: 0~1 分位}，**1 = 值最大**（最拥挤）。

    并列取平均名次（避免"同值看谁在前"这种不确定性）。
    与 `StockScreener._percentile_map(higher_is_better=True)` 逐值一致 ——
    这条一致性由 `tests/test_crowding.py::test_pct_rank_matches_screener` 钉住，
    否则两套分位会在同一页面上给出互相矛盾的排名而没人看得出来。

    n == 1 ⇒ 返回 0.0（"唯一一只 = 不拥挤"）：这是保守方向，不会误剔除。
    """
    clean = {str(c): float(v) for c, v in (values or {}).items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)
             and np.isfinite(float(v))}
    n = len(clean)
    if n == 0:
        return {}
    if n == 1:
        return {next(iter(clean)): 0.0}
    ordered = sorted(clean.items(), key=lambda kv: kv[1])
    out = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0                     # 0-based 平均名次，升序
        pct = avg_rank / (n - 1)                     # 最小 → 0.0，最大 → 1.0
        for k in range(i, j + 1):
            out[ordered[k][0]] = round(pct, 4)
        i = j + 1
    return out


def industry_pct_rank(values: dict, industry_of: dict, *,
                      min_group: int = MIN_GROUP) -> tuple:
    """{code: 数值} → ({code: 分位}, 统计) —— **行业内**分位，不足则合并进"其余"池。

    为什么必须行业内（2026-10-30）
    -----------------------------
    换手率与成交额都带**行业常数**：银行日换手 0.3%、题材股 8%，全市场分位下
    银行永远在最低档、永远不拥挤 —— 那测的就不是"谁更拥挤"，而是"这是哪个行业"。
    实测记录里 PE/波动率都踩过同一个坑（全池分位下银行恒满分），这里不重犯。

    规则（与 `StockScreener._percentile_by_industry` 完全一致）
    ---------------------------------------------------------
    1. 按行业分组，**组内**做分位；
    2. 组内样本 < `min_group` ⇒ 不单独分位，合并进"其余"池统一分位
       （20 只以下排分位等于按噪声排序；但也**不能**直接跳过 —— 那会让缺行业的
       股票永远拿不到分位，从而永远不被闸门看见）；
    3. 缺行业 ⇒ 同样进"其余"池。

    ⚠️ 代价必须说清：合并进"其余"池的股票是在**混合群体**里排序，
       与行业内分位不同尺。所以返回的 `stats` 必须被报告（写进 meta），
       而不是默默发生。

    Returns: `(pct_map, {"industry": n, "pool": n, "groups": n, "missing": n})`
    """
    clean = {str(c): float(v) for c, v in (values or {}).items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)
             and np.isfinite(float(v))}
    stats = {"industry": 0, "pool": 0, "groups": 0, "missing": 0}
    if not clean:
        return {}, stats
    ind = industry_of or {}
    buckets: dict = {}
    rest: dict = {}
    for code, v in clean.items():
        name = ind.get(code)
        if not name:
            stats["missing"] += 1
            rest[code] = v
            continue
        buckets.setdefault(name, {})[code] = v

    out: dict = {}
    for name, group in buckets.items():
        if len(group) < int(min_group):
            rest.update(group)
            continue
        out.update(pct_rank(group))
        stats["industry"] += len(group)
        stats["groups"] += 1
    if rest:
        out.update(pct_rank(rest))
        stats["pool"] = len(rest)
    return out, stats


def crowd_trade_of(turn: Optional[float], amount: Optional[float]) -> Optional[float]:
    """换手率与成交额两个分位取**等权均值**。

    只有一个可用时就用那一个（不是当 0 处理 —— "缺"不等于"低"）。
    两个都缺 ⇒ None（下游因此不判、不剔除）。
    """
    vals = [v for v in (turn, amount) if isinstance(v, (int, float))
            and np.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


# ================================================================
# 2. 行业层（申万一级）
# ================================================================

def sw_path() -> str:
    global _SW_PATH
    if _SW_PATH:
        return _SW_PATH
    from data import panel_store
    return os.path.join(panel_store.PANEL_DIR, "industry_swa_l1.parquet")


def sw_industry_map(path: Optional[str] = None) -> dict:
    """{code: 申万一级行业名}。文件不可用 ⇒ 空 dict（分位退回"其余"池，并会被报告）。"""
    p = path or sw_path()
    try:
        st = os.stat(p)
        stamp = (p, st.st_size, st.st_mtime)
    except OSError as e:
        _logger.warning("[拥挤度] 申万行业表不可用（%s）⇒ 行业内分位退化为全池分位", e)
        return {}
    if _SW_CACHE.get("stamp") == stamp:
        return _SW_CACHE["map"]
    try:
        d = pd.read_parquet(p, columns=["stock_code", "industry_name"])
        d["stock_code"] = d["stock_code"].astype(str).str.zfill(6)
        m = dict(zip(d["stock_code"], d["industry_name"].astype(str)))
    except Exception as e:                            # noqa: BLE001
        _logger.warning("[拥挤度] 申万行业表读取失败（%s）: %s", p, e)
        return {}
    _SW_CACHE.update({"stamp": stamp, "map": m})
    return m


# ================================================================
# 3. 数据读取
# ================================================================

def latest_snapshot_date() -> Optional[str]:
    """`market_snapshot` 里最新的一天（换手率/成交额的当日截面来源）。"""
    from db import store
    conn = store.get_connection()
    try:
        row = conn.execute("SELECT MAX(snapshot_date) AS d FROM market_snapshot").fetchone()
        return (row["d"] if row else None) or None
    finally:
        conn.close()


def read_snapshot_rows(date: str) -> pd.DataFrame:
    """某日全市场换手率/成交额/行业（一条 SQL，毫秒级）。

    ⚠️ 只保留 `turnover > 0 且 amount > 0` 的行：停牌/未上市当天没有分位可谈，
       把它们当 0 参与分位会把整条分布往下压。
    """
    from db import store
    conn = store.get_connection()
    try:
        rows = conn.execute(
            "SELECT stock_code, turnover, amount, industry_ts FROM market_snapshot "
            "WHERE snapshot_date = ? AND turnover IS NOT NULL AND turnover > 0 "
            "AND amount IS NOT NULL AND amount > 0", (date,)).fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame(columns=["code", "turn", "amount", "industry_ts"])
    return pd.DataFrame([{"code": str(r["stock_code"]), "turn": float(r["turnover"]),
                          "amount": float(r["amount"]),
                          "industry_ts": r["industry_ts"] or ""} for r in rows])


def ret20_map(codes: Iterable[str], as_of: str) -> dict:
    """{code: 20 日涨幅}（未复权 close 口径，与 11 年验证逐字一致）。

    ⚠️ 用 `get_bars(adjust="none")`：验证脚本读的是 `by_code` 里的**未复权** close。
       换成 qfq 会让 20 日涨幅在分红除权日出现假跌，信号与验证对不上。
    ⚠️ `get_bars` 是唯一取数入口 ⇒ 自动叠加 `daily_overlay`。实测 `by_code` 停在
       2026-10-16 而覆盖层到 10-29：只读 `by_code` 会让整条信号**静默滞后 9 个交易日**
       （这正是记录里 `parts/` 与 `daily_overlay/` 两种布局踩过的坑）。
       不足 21 根的（新股/长停）⇒ 不出现在结果里（缺就是缺，不填 0）。
    """
    from data import bars
    out = {}
    need = MOM_LOOKBACK + 1
    for code in codes:
        try:
            df = bars.get_bars(code, end=as_of, adjust="none", min_rows=need)
        except Exception as e:                        # noqa: BLE001
            _logger.debug("[拥挤度] %s 取数失败: %s", code, e)
            continue
        if df is None or len(df) < need:
            continue
        c = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype="float64")
        if not (c[-1] > 0 and c[-need] > 0):
            continue
        r = c[-1] / c[-need] - 1.0
        if np.isfinite(r):
            out[str(code)] = float(r)
    return out


# ================================================================
# 4. 计算与落盘
# ================================================================

def snapshot_dir() -> str:
    from data import panel_store
    return os.path.join(panel_store.PANEL_DIR, "crowd")


def compute(as_of: Optional[str] = None, *, industry_of: Optional[dict] = None,
            ret20: Optional[dict] = None) -> dict:
    """算某日全市场拥挤度截面（**不落盘**，落盘交给 `save()`）。

    Args:
        as_of: 日期（YYYY-MM-DD）。None ⇒ `market_snapshot` 最新日。
        industry_of / ret20: 测试注入用；生产路径不传。

    Returns: `{"date", "rows": [...], "meta": {...}}`；数据不足 ⇒ `date=None`。
    """
    date = as_of or latest_snapshot_date()
    if not date:
        return {"date": None, "rows": [], "meta": {"note": "market_snapshot 无数据"}}
    frame = read_snapshot_rows(date)
    if frame.empty:
        return {"date": date, "rows": [],
                "meta": {"note": f"{date} 快照无有效换手率/成交额行"}}

    ind = industry_of if industry_of is not None else sw_industry_map()
    mom_raw = ret20 if ret20 is not None else ret20_map(frame["code"].tolist(), date)

    turn_raw = dict(zip(frame["code"], frame["turn"]))
    # 原始成交额（元）用于展示；取 log 只用于分位（量级差 4 个数量级，
    # 不取 log 的话分位会被几只千亿成交的权重股吃掉全部高分区）。
    amt_val = dict(zip(frame["code"], frame["amount"]))
    amt_raw = {c: float(np.log(max(v, 1e4))) for c, v in amt_val.items()}

    turn_pct, turn_stats = industry_pct_rank(turn_raw, ind)
    amt_pct, amt_stats = industry_pct_rank(amt_raw, ind)
    mom_pct, mom_stats = industry_pct_rank(mom_raw, ind)

    rows = []
    for code in frame["code"]:
        rows.append({
            "code": code,
            "crowd_trade": crowd_trade_of(turn_pct.get(code), amt_pct.get(code)),
            "mom20": mom_pct.get(code),
            "turn": turn_raw.get(code),
            "amount": amt_val.get(code),
            "ret20": mom_raw.get(code),
            "industry": ind.get(code) or "",
        })
    n_crowd = sum(1 for r in rows if r["crowd_trade"] is not None)
    n_mom = sum(1 for r in rows if r["mom20"] is not None)
    gated = sum(1 for r in rows if is_gated(r["crowd_trade"], r["mom20"]))
    meta = {
        "date": date,
        "rows": len(rows),
        "crowd_n": n_crowd,
        "mom_n": n_mom,
        "gated": gated,
        "gated_pct": round(gated / max(1, len(rows)), 4),
        "industry_groups": turn_stats["groups"],
        "industry_covered": turn_stats["industry"],
        "industry_pool": turn_stats["pool"],
        "industry_missing": turn_stats["missing"],
        "mom_industry_groups": mom_stats["groups"],
        "mom_industry_pool": mom_stats["pool"],
        "industry_source": ("申万一级"
                            if ind else "缺失（已退化为全池分位）"),
        "gate_crowd_pct": GATE_CROWD_PCT,
        "gate_mom_pct": GATE_MOM_PCT,
        "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "basis": "close→close 未复权；换手率=新浪百分数；成交额=元（取 log）",
    }
    if not ind:
        meta["degraded"] = ["申万行业表不可用 ⇒ 行业内分位退化为全池分位"]
    return {"date": date, "rows": rows, "meta": meta}


def save(result: dict) -> Optional[str]:
    """落盘 `{date}.parquet` + `{date}.json`（原子 `os.replace`）。

    ⚠️ `to_parquet` 先 truncate 再写 ⇒ 读者可能读到半个文件。
       这与面板覆盖层踩过的是同一个坑：先写临时文件、再原子替换。
    """
    date = result.get("date")
    if not date or not result.get("rows"):
        return None
    d = snapshot_dir()
    os.makedirs(d, exist_ok=True)
    frame = pd.DataFrame(result["rows"])
    pq = os.path.join(d, f"{date}.parquet")
    js = os.path.join(d, f"{date}.json")
    tmp_pq, tmp_js = pq + ".tmp", js + ".tmp"
    frame.to_parquet(tmp_pq, index=False)
    with open(tmp_js, "w", encoding="utf-8") as f:
        json.dump(result["meta"], f, ensure_ascii=False, indent=2)
    os.replace(tmp_pq, pq)
    os.replace(tmp_js, js)
    return pq


def available_dates() -> list:
    d = snapshot_dir()
    if not os.path.isdir(d):
        return []
    return sorted(f[:-8] for f in os.listdir(d) if f.endswith(".parquet"))


def load(as_of: Optional[str] = None) -> dict:
    """读最新（或 <= as_of 的最近一份）拥挤度快照。

    Returns: `{"date", "rows": {code: {...}}, "meta", "available", "note"}`
             无快照 ⇒ `date=None`、`rows={}`、`note` 说明原因。
    """
    ds = available_dates()
    if as_of:
        ds = [x for x in ds if x <= str(as_of)]
    if not ds:
        return {"date": None, "rows": {}, "meta": {}, "available": [],
                "note": "无拥挤度快照（需先跑 python -m core.crowding --refresh）"}
    date = ds[-1]
    pq = os.path.join(snapshot_dir(), f"{date}.parquet")
    meta = {}
    js = os.path.join(snapshot_dir(), f"{date}.json")
    try:
        with open(js, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        _logger.debug("[拥挤度] %s meta 读取失败: %s", date, e)
    try:
        frame = pd.read_parquet(pq)
    except Exception as e:                            # noqa: BLE001
        _logger.warning("[拥挤度] 快照 %s 读取失败: %s", date, e)
        return {"date": None, "rows": {}, "meta": meta, "available": ds,
                "note": f"快照 {date} 读取失败: {e}"}
    rows = {}
    for r in frame.to_dict("records"):
        rows[str(r["code"])] = {
            "crowd_trade": _f(r.get("crowd_trade")),
            "mom20": _f(r.get("mom20")),
            "industry": r.get("industry") or "",
        }
    return {"date": date, "rows": rows, "meta": meta, "available": ds, "note": ""}


def _f(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError) as e:
        # 快照里 None/NaN 是**正常缺值**（当日算不出来的那几只），不是故障；
        # 但"静默 except"会掩盖 dtype 漂移（比如整列变成字符串），所以仍记一条 debug。
        _logger.debug("[拥挤度] 数值转换失败（按缺值处理）: %r (%s)", v, e)
        return None
    return f if np.isfinite(f) else None


# ================================================================
# 5. 闸门
# ================================================================

def is_gated(crowd_trade: Optional[float], mom20: Optional[float],
             *, crowd_pct: float = GATE_CROWD_PCT,
             mom_pct: float = GATE_MOM_PCT) -> bool:
    """是否命中闸门（= 不新开仓）。

    ⚠️ 两个条件**都缺一不可**（and，不是 or）：双排序实测"高波+拥挤"才是最危险那一格，
       拆开任一维度都会把 11 年验证过的 -0.97pp 削成噪声。
    ⚠️ 任一为 None ⇒ **不判闸门**（False）。"没数据"不等于"安全"，但在闸门这种
       减法机制里，缺数据时保留候选比误杀安全 —— 与流动性下限同一条纪律。
    """
    if crowd_trade is None or mom20 is None:
        return False
    return bool(crowd_trade >= crowd_pct and mom20 >= mom_pct)


def gate_enabled() -> bool:
    """开关（运行时解析）。默认**开**；`BIGA_CROWD_GATE=0/false/off/no` 关。"""
    v = str(os.environ.get("BIGA_CROWD_GATE", "1")).strip().lower()
    return v not in ("0", "false", "off", "no", "")


def _snapshot_lag(date: Optional[str]) -> Optional[int]:
    """快照日距今天几个**交易日**（0 = 就是今天；None = 判不出来）。

    ⚠️ 参考日必须是"今天"，**不能用 `panel_date_max()`**：面板本身也可能滞后，
       拿它当基准等于"两个都落后 ⇒ 互相证明对方是新的"。
    ⚠️ 判不出来（日历没覆盖到今天）返回 None，**不返回 0** —— 未知不能报平安。
    """
    if not date:
        return None
    try:
        from data import bars
        return bars.trading_day_lag(date, time.strftime("%Y-%m-%d"))
    except Exception as e:                            # noqa: BLE001
        _logger.debug("[拥挤度] 快照滞后计算失败（%s）: %s", date, e)
        return None


def apply_gate(codes: Iterable[str], snapshot: Optional[dict] = None, *,
               as_of: Optional[str] = None, min_keep: int = 10) -> tuple:
    """对一批候选应用拥挤度闸门 → `(保留, 被剔清单, meta)`。

    ⚠️ 四条边界（与 `_apply_liquidity_floor` 同一纪律）：
      1. 快照取不到 ⇒ **不过滤**，meta 写明"未生效"；
      2. 快照**过期**（滞后 > `MAX_SNAPSHOT_LAG_DAYS` 个交易日）⇒ 不过滤 ——
         一周前的拥挤度不能拿来砍今天的候选，而且必须说出来；
      3. 单只缺 crowd_trade/mom20 ⇒ **不剔除**；
      4. 过滤后少于 `min_keep` 只 ⇒ 放弃过滤（池子已很小，再砍只剩噪声）。
    """
    meta = {"enabled": gate_enabled(), "applied": False, "date": None,
            "dropped": [], "note": "", "crowd_pct": GATE_CROWD_PCT,
            "mom_pct": GATE_MOM_PCT, "rows": 0, "lag_days": None}
    codes = [str(c) for c in (codes or [])]
    if not meta["enabled"]:
        meta["note"] = "拥挤度闸门已关闭（BIGA_CROWD_GATE=0）"
        return codes, [], meta
    snap = snapshot if snapshot is not None else load(as_of=as_of)
    if not snap.get("date") or not snap.get("rows"):
        meta["note"] = snap.get("note") or "无拥挤度快照 ⇒ 闸门未生效（不剔除任何候选）"
        _logger.warning("[选股] %s", meta["note"])
        return codes, [], meta
    meta["date"] = snap["date"]
    meta["rows"] = len(snap["rows"])

    lag = _snapshot_lag(snap["date"])
    meta["lag_days"] = lag
    if lag is not None and lag > MAX_SNAPSHOT_LAG_DAYS:
        meta["note"] = (f"拥挤度快照滞后 {lag} 个交易日"
                        f"（> {MAX_SNAPSHOT_LAG_DAYS}）⇒ 闸门未生效（不剔除任何候选）")
        _logger.warning("[选股] %s；末次快照 %s", meta["note"], snap["date"])
        return codes, [], meta

    rows = snap["rows"]
    keep, dropped = [], []
    for c in codes:
        info = rows.get(c)
        if not info:
            keep.append(c)                            # 快照没这只：缺数据，保留
            continue
        if is_gated(info.get("crowd_trade"), info.get("mom20")):
            dropped.append({"股票代码": c,
                            "拥挤度": info.get("crowd_trade"),
                            "20日涨幅分位": info.get("mom20"),
                            "行业": info.get("industry") or ""})
        else:
            keep.append(c)
    if len(keep) < int(min_keep):
        meta["note"] = (f"按拥挤度闸门本应剔除 {len(dropped)} 只，但剩余仅 "
                        f"{len(keep)} 只 ⇒ 放弃过滤（样本太小）")
        _logger.warning("[选股] %s", meta["note"])
        return codes, [], meta
    meta["applied"] = True
    meta["note"] = (f"拥挤度闸门（拥挤≥{int(GATE_CROWD_PCT * 100)}% 且 "
                    f"20日涨幅分位≥{int(GATE_MOM_PCT * 100)}%，快照日 {snap['date']}）："
                    f"剔除 {len(dropped)} 只")
    if lag and lag > 1:
        # 滞后 1 天是**正常**的（16:30 产出、次日 09:25 消费）；再多就要显式提醒
        meta["note"] += f"（快照滞后 {lag} 个交易日）"
    if dropped:
        _logger.info("[选股] %s；被剔除：%s", meta["note"],
                     "、".join(d["股票代码"] for d in dropped[:10]))
    return keep, dropped, meta


# ================================================================
# 6. CLI
# ================================================================

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="交易拥挤度快照（预计算落表）")
    ap.add_argument("--refresh", action="store_true", help="算最新一天并落盘")
    ap.add_argument("--date", default=None, help="指定日期（默认 market_snapshot 最新日）")
    ap.add_argument("--show", action="store_true", help="打印最新快照概况")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.refresh:
        t0 = time.time()
        r = compute(as_of=args.date)
        if not r.get("date"):
            _logger.warning("未生成：%s", r["meta"].get("note"))
            return 1
        p = save(r)
        m = r["meta"]
        # ⚠️ 走 `_logger` 而不是 `print`：`core/` 里禁止非 CLI print
        #    （守卫 `test_no_noncli_print_in_data_and_core`），因为长驻进程的
        #    标准输出进不了日志文件 —— 一次排障时"日志里什么都没有"就是这么来的。
        _logger.info(
            "落盘 %s\n  日期 %s 行 %s 拥挤度可用 %s 涨幅可用 %s 命中闸门 %s（%.1f%%）"
            "\n  行业 %s 组 %s 覆盖 %s 合并池 %s 缺 %s\n  耗时 %.1fs",
            p, m["date"], m["rows"], m["crowd_n"], m["mom_n"], m["gated"],
            m["gated_pct"] * 100, m["industry_source"], m["industry_groups"],
            m["industry_covered"], m["industry_pool"], m["industry_missing"],
            time.time() - t0)
        return 0
    if args.show:
        s = load(as_of=args.date)
        if not s["date"]:
            _logger.warning("%s", s["note"])
            return 1
        m = s["meta"]
        _logger.info("快照日 %s 可用 %s 只，命中闸门 %s（%s）",
                     s["date"], len(s["rows"]), m.get("gated"), m.get("gated_pct"))
        return 0
    _logger.info("%s", __doc__.split("## 一句话")[1].split("## 口径")[0].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
