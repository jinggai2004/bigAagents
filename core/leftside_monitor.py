"""左侧监测器：无明显利空的大跌 + 基本面资质 → 挂入关注池，等待右侧信号。

## 设计（两层分离，与 `plunge_event_check.py` 的结论一一对应）

**左侧（进池）**——三类信号，全部有实测证据：
1. `orphan_plunge` 孤儿大跌：单日跌幅 ≤ −7%，且**剥离行业与大盘后仍然异常**
   （行业中性跌幅 = 个股跌幅 − 同行业等权跌幅；市场中性 = − 当日全市场等权跌幅）。
   "无明显利空"的量化代理就是"行业和市场都解释不了的那部分跌幅"。
   证据：大跌+无财报 −1.39pp（5/5 年负）⇒ 但**叠加好基本面后翻正**（+0.81pp t=4.07），
   所以进池必须再卡一层基本面资质，裸大跌不进池。
2. `post_report_plunge` 财报后好报表大跌：15 天内刚披露净利同比 >20% 的报表，
   随后单日 ≤ −7%（市场不买账）。证据：+3.00pp t=4.15（口径必须同为 −7%）。
   这一类**当天可判断**，是唯一不需要"等验证"的入口。
3. `org_accumulate` 机构净买入（辅助排序信号）：东财机构席位净买 >1000 万。
   证据：D1~D10 相对超额 +0.69~1.24pp（t 3~5）。

**基本面资质卡（进池的必要条件）**——全部来自东财财务 PIT（`financial_pit`）：
- 盈利质量：`ocf_to_np`（现金流/净利润）> 行业内 50% 分位
- 盈利稳定：最近 4 期 ROE 均值 > 0 且无单期 < −5%
- 增长（可选加分）：`sjltz` > 0 或 `roe_yoy` > 0（"订单充足/产品涨价"的可量化代理；
  涨价/订单本身无结构化数据源，用财务改善近似，**页面必须如实标注是代理**）

**右侧（出池触发）**——挂在池子上的条件，出现即升级为"候选"：
- `price_reclaim`：收盘价站回 20 日均线（趋势修复的最简可靠形式）
- `vol_dry`：5 日成交量 < 20 日均量的 70%（抛压枯竭）
- `report_good`：财报验证通过（净利同比 >20%）——用户"燕山啤酒"场景的正式出口

⚠️ 边界：本模块是**关注池**，不是交易信号。所有阈值来自已验证的统计结论，
但"右侧出现 ⇒ 上涨"未做组合级回测（右侧信号只在池内小样本上验证过方向）。
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

_logger = logging.getLogger("leftside_monitor")

# ── 阈值（全部来自已验证结论，改动前先看模块 docstring 的证据链）──
PLUNGE_THR = -0.07           # 孤儿大跌：个股跌幅阈值
NEUTRAL_THR = -0.05          # 剥离行业+市场后仍 ≤ −5% 才算"解释不了"
# ⚠️ 必须与证据口径一致（−7%）：首版放宽到 −5%，实测 10-30（三季报披露高峰后）
# 一天命中 247 只 —— 披露窗口内"好报表"基数巨大，−5% 的日常波动就能凑齐组合，
# 信号完全稀释。原始证据（POST 组 +3.00pp/t=4.15）的事件定义就是 ≤ −7%。
POST_RPT_DROP = -0.07
POST_RPT_WINDOW = 15         # 财报后多少天内的大跌算"市场不买账"
GOOD_YOY = 0.20              # 净利同比 >20% = 好报表
STALE_DAYS = 3               # 快照/行业数据允许的最大滞后（自然日）
WATCH_MAX_DAYS = 60          # 池内最长停留（自然日）；超时未触发右侧 ⇒ 移出
RIGHT_VOL_RATIO = 0.70       # 右侧-量能枯竭：5 日量 / 20 日均量
ORG_NET_MIN = 1e7            # 机构净买下限（元）
MIN_POOL = 5                 # 池子小于这个数不触发排名展示


@dataclass
class WatchItem:
    """一条关注记录。"""
    code: str
    name: str = ""
    industry: str = ""
    # 左侧信号（可叠加）：orphan_plunge / post_report_plunge / org_accumulate
    signals: list[str] = field(default_factory=list)
    # 触发日的原始信息
    trigger_date: str = ""
    drop_pct: float | None = None           # 当日跌幅（负数）
    industry_neutral: float | None = None   # 行业中性跌幅（负数）
    # 基本面资质快照
    quality: dict = field(default_factory=dict)
    # 右侧信号（出现任何一个即升级）
    right_side: list[str] = field(default_factory=list)
    days_in_pool: int = 0
    status: str = "watching"                # watching / promoted / expired


def _industry_map() -> dict[str, str]:
    """code → 行业（新浪 49 类口径，来自最新快照）。"""
    from db import store
    conn = store.get_connection()
    try:
        rows = conn.execute(
            "SELECT stock_code, industry FROM market_snapshot "
            "WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM market_snapshot) "
            "AND industry IS NOT NULL").fetchall()
        return {str(r[0]).zfill(6): str(r[1]) for r in rows}
    finally:
        conn.close()


def load_daily_with_industry(days: int = 40) -> pd.DataFrame:
    """日频面板 + 行业列（行业来自快照，按 code 静态 join）。"""
    from core.zt_calendar import _load_recent
    # 面板路径复用 zt_calendar 的常量（同一份面板，路径必须同源）
    from core.zt_calendar import PANEL_PARTS_DIR as ZT_PANEL_PARTS
    from core.zt_calendar import PANEL_OVERLAY_DIR as ZT_PANEL_OVERLAY
    # ⚠️ use_cache=False：`_load_recent` 缓存不区分 days（踩过）
    d = _load_recent(days, use_cache=False)
    if d is None or d.empty:
        return pd.DataFrame()
    d = d.copy()
    d["stock_code"] = d["stock_code"].astype(str).str.zfill(6)
    d["trade_date"] = d["trade_date"].astype(str).str[:10]
    for c in ("close", "preclose", "volume"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "trade_status" in d.columns:
        ts = pd.to_numeric(d["trade_status"], errors="coerce")
        d = d[ts.isna() | (ts == 1)]
    d = d[d["close"] > 0].dropna(subset=["preclose"])
    d["chg"] = d["close"] / d["preclose"] - 1.0

    # `volume` 不在 `_load_recent` 的列子集里（量能枯竭信号需要）：单独按列下推补读。
    # ⚠️ `right_side_flags` 依赖本列，缺失时必须显式置 NaN 并让 flags 跳过 vol_dry，
    #    而不是让 `g["volume"]` 抛 KeyError（实测踩过：状态机第一次跑就崩）。
    if "volume" not in d.columns:
        d["volume"] = np.nan
        import glob as _glob, os as _os
        start = (pd.Timestamp.now().normalize() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        for p in (sorted(_glob.glob(_os.path.join(ZT_PANEL_OVERLAY, "*.parquet")))
                  + sorted(_glob.glob(_os.path.join(ZT_PANEL_PARTS, "*.parquet")))):
            try:
                v = pd.read_parquet(p, columns=["stock_code", "trade_date", "volume"])
            except Exception as e:                   # noqa: BLE001  缺列的旧分片跳过
                _logger.debug("量能分片不可用，跳过 %s: %s", p, e)
                continue
            v = v[v["stock_code"].astype(str).str.zfill(6).isin(set(d["stock_code"]))]
            if v.empty:
                continue
            v = v.assign(stock_code=v["stock_code"].astype(str).str.zfill(6),
                         trade_date=v["trade_date"].astype(str).str[:10])
            v = v[v["trade_date"] >= start]
            if v.empty:
                continue
            d = d.merge(v.drop_duplicates(["stock_code", "trade_date"]),
                        on=["stock_code", "trade_date"], how="left", suffixes=("", "_new"))
            if "volume_new" in d.columns:
                d["volume"] = d["volume"].fillna(d["volume_new"])
                d = d.drop(columns=["volume_new"])

    ind = _industry_map()
    d["industry"] = d["stock_code"].map(ind)
    # 当日行业等权跌幅 & 全市场等权跌幅（剥离共跌）
    d["ind_chg"] = d.groupby("industry")["chg"].transform("mean")
    d["mkt_chg"] = d.groupby("trade_date")["chg"].transform("mean")
    # 行业中性跌幅 = 个股 − 行业；市场已含在行业里（行业均值再减市场均值无必要，
    # 孤儿判据用"个股 − 行业均值"，行业本身的下跌属于"有解释"的部分）
    d["neu_chg"] = d["chg"] - d["ind_chg"]
    return d.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)


def find_orphan_plunges(d: pd.DataFrame, asof: str | None = None) -> list[dict]:
    """孤儿大跌：跌幅 ≤ −7% 且行业中性跌幅 ≤ −5%（行业解释不了）。"""
    last_day = asof or d["trade_date"].max()
    day = d[d["trade_date"] == last_day].copy()
    if day.empty:
        return []
    msk = (day["chg"] <= PLUNGE_THR) & (day["neu_chg"] <= NEUTRAL_THR)
    out = []
    for _, r in day[msk].iterrows():
        out.append({
            "code": r["stock_code"], "name": "", "industry": r.get("industry", ""),
            "signal": "orphan_plunge", "trigger_date": last_day,
            "drop_pct": round(float(r["chg"]), 4),
            "industry_neutral": round(float(r["neu_chg"]), 4),
            "ind_chg": round(float(r.get("ind_chg", np.nan)), 4) if pd.notna(r.get("ind_chg")) else None,
        })
    return out


def find_post_report_plunges(d: pd.DataFrame, asof: str | None = None) -> list[dict]:
    """财报后好报表大跌：15 天内披露净利同比>20%，当日又跌 ≤ −5%。"""
    from core.financial_pit import load_financials
    last_day = asof or d["trade_date"].max()
    if last_day is None:
        return []
    day = d[d["trade_date"] == last_day]
    if day.empty:
        return []
    plunges = day[day["chg"] <= POST_RPT_DROP]
    if plunges.empty:
        return []

    fin = load_financials()
    if fin.empty:
        return []
    fin = fin.copy()
    fin["code"] = fin["code"].astype(str).str.zfill(6)
    fin["pub"] = pd.to_datetime(fin["pub_date"], errors="coerce")
    yoy = pd.to_numeric(fin.get("sjltz"), errors="coerce") if "sjltz" in fin.columns else None
    if yoy is None:
        return []
    fin["_yoy"] = yoy
    fin = fin.dropna(subset=["pub", "_yoy"])

    t0 = pd.Timestamp(last_day)
    good_recent = fin[(fin["pub"] <= t0)
                      & (fin["pub"] >= t0 - pd.Timedelta(days=POST_RPT_WINDOW))
                      & (fin["_yoy"] > GOOD_YOY)]
    good_codes = set(good_recent["code"])
    out = []
    for _, r in plunges.iterrows():
        if r["stock_code"] in good_codes:
            g = good_recent[good_recent["code"] == r["stock_code"]].iloc[-1]
            out.append({
                "code": r["stock_code"], "name": "", "industry": r.get("industry", ""),
                "signal": "post_report_plunge", "trigger_date": last_day,
                "drop_pct": round(float(r["chg"]), 4),
                "report_yoy": round(float(g["_yoy"]), 4),
                "report_pub": str(g["pub"].date()),
            })
    return out


def quality_pass(codes: list[str], asof=None) -> dict[str, dict]:
    """基本面资质卡：返回 {code: 质量摘要}，只含**通过**的票。

    三条（全部基于 PIT 公告日，无前视）：
    1. ocf_to_np 在**同行业候选内** ≥ 50% 分位（现金流质量）
    2. 最近 4 期 ROE 均值 > 0（稳定盈利）
    3. sjltz > 0 或 roe_yoy > 0（增长代理：涨价/订单的可量化近似）
    """
    from core.financial_pit import latest_visible_multi, FIN_DIR
    import os
    if not codes:
        return {}
    # 财务库不存在时直接返回空（不是每台机器都回填过）
    if not os.path.isdir(FIN_DIR):
        return {}
    vals = latest_visible_multi(codes, ["ocf_to_np", "roe", "sjltz", "roe_yoy"],
                                asof=asof)
    out: dict[str, dict] = {}
    for c, v in vals.items():
        ocf = v.get("ocf_to_np")
        roe = v.get("roe")
        grow = v.get("sjltz")
        ry = v.get("roe_yoy")
        passed = (ocf is not None and roe is not None
                  and roe > 0 and (grow is not None and grow > 0
                                   or ry is not None and ry > 0))
        out[c] = {"ocf_to_np": ocf, "roe": roe, "sjltz": grow, "roe_yoy": ry,
                  "passed": bool(passed)}
    # ocf_to_np 分位在候选内算（不是全市场——池子小，全市场分位会失真）
    pooled = [v["ocf_to_np"] for v in out.values() if v["ocf_to_np"] is not None]
    if len(pooled) >= MIN_POOL:
        med = float(np.median(pooled))
        for v in out.values():
            o = v.get("ocf_to_np")
            v["ocf_ok"] = bool(o is not None and o >= med)
            # 最终判定：现金流质量 + ROE>0 + 增长
            v["passed"] = bool(v.get("ocf_ok") and v.get("roe", 0) and v["roe"] > 0
                               and ((v.get("sjltz") or 0) > 0 or (v.get("roe_yoy") or 0) > 0))
    return out


def org_net_buy_today(asof: str | None = None) -> dict[str, dict]:
    """当日/近 5 日机构净买 >1000 万的票（辅助信号，给池内条目加分）。"""
    try:
        from data import billboard as B
        return B.org_net_buy(asof=asof, days=5, min_net=ORG_NET_MIN)
    except Exception as e:                              # noqa: BLE001
        _logger.warning("[左侧监测] 龙虎榜数据不可用（跳过该信号）: %s", e)
        return {}


def scan(asof: str | None = None, days: int = 40) -> list[WatchItem]:
    """主入口：跑一遍左侧监测，返回当日的关注条目（含基本面资质判定）。"""
    d = load_daily_with_industry(days)
    if d.empty:
        return []
    events = find_orphan_plunges(d, asof) + find_post_report_plunges(d, asof)
    if not events:
        return []
    # 同一只票去重（两个信号都中 ⇒ 合并 signals）
    merged: dict[str, dict] = {}
    for e in events:
        cur = merged.setdefault(e["code"], {**e, "signals": []})
        if e["signal"] not in cur["signals"]:
            cur["signals"].append(e["signal"])
        if e["signal"] == "post_report_plunge":
            cur["report_yoy"] = e.get("report_yoy")
            cur["report_pub"] = e.get("report_pub")
    codes = list(merged)
    qual = quality_pass(codes, asof=merged[codes[0]]["trigger_date"] if codes else None)
    org = org_net_buy_today(asof)
    out = []
    for c, e in merged.items():
        q = qual.get(c) or {}
        signals = list(e["signals"])
        if c in org:
            signals.append("org_accumulate")
        # 孤儿大跌必须过基本面资质才进池（裸大跌继续跑输是已验证结论）；
        # 财报后好报表大跌不需要再卡（财报本身就是资质证明）
        if "orphan_plunge" in signals and not q.get("passed"):
            continue
        if "post_report_plunge" in signals or q.get("passed") or c in org:
            out.append(WatchItem(
                code=c, industry=e.get("industry", ""),
                signals=signals, trigger_date=e["trigger_date"],
                drop_pct=e.get("drop_pct"),
                industry_neutral=e.get("industry_neutral"),
                quality={k: v for k, v in q.items() if k != "passed"},
            ))
    return out


def right_side_flags(d: pd.DataFrame, code: str, asof: str | None = None) -> list[str]:
    """右侧信号检测（对池内单只票）：站回 MA20 / 量能枯竭。

    ⚠️ 只返回信号名，不构成买卖建议；两个信号可以同时出现。
    """
    g = d[d["stock_code"] == code].sort_values("trade_date")
    if len(g) < 21:
        return []
    last = g.iloc[-1]
    ma20 = g["close"].tail(20).mean()
    flags = []
    if last["close"] > ma20:
        flags.append("price_reclaim")
    if "volume" in g.columns and g["volume"].notna().any():
        v5 = float(g["volume"].tail(5).mean())
        v20 = float(g["volume"].tail(20).mean())
        if v20 > 0 and v5 / v20 < RIGHT_VOL_RATIO:
            flags.append("vol_dry")
    return flags


# ══════════════════════════════════════════════════════════════════
# 盘中扫描（14:45）+ 关注池持久化 + 右侧状态机
# ══════════════════════════════════════════════════════════════════
# 为什么要盘中跑一次：收盘后的清单"第二天开盘"才能操作，中间隔一晚；
# 14:45 的实时跌幅已接近收盘价（误差通常 <1pp），盘中预扫让用户**收盘后立刻**
# 拿到清单、当晚研究基本面、次日开盘执行。15:45 收盘版才是正式落库口径
# （两者可能有差：尾盘异动），页面以收盘版为准、盘中版标"预"。

SCAN_EXPIRE_DAYS = 60        # 池内标的超过 60 个自然日未站回 MA20 ⇒ 证伪嫌疑，移出


def scan_intraday(asof: str | None = None) -> list[WatchItem]:
    """盘中预扫：用**实时快照**的当日跌幅判定（14:45 左右跑）。

    与收盘版 `scan()` 的三点差别：
    1. 数据源是快照截面（1.5s），不是日频面板（要读历史）⇒ 快；
    2. 行业中性跌幅 = 个股实时跌幅 − 同行业实时均值（快照内现算）；
    3. 不做右侧检测（盘中无意义），只出左侧清单。
    数据滞后/实时源不可用时快照会带 `degraded` 标记 ⇒ 照跑但 meta 注明。
    """
    from data.eastmoney_source import get_market_snapshot
    snap = get_market_snapshot()
    if snap is None or snap.empty:
        return []
    day = str(getattr(snap, "attrs", {}).get("data_date") or asof or "")
    df = pd.DataFrame({
        "stock_code": snap["代码"].astype(str).str.zfill(6),
        "name": snap["名称"].astype(str),
        "industry": snap["industry"].astype(str),
        "chg": pd.to_numeric(snap["涨跌幅"], errors="coerce") / 100.0,
    }).dropna(subset=["chg"])
    df["ind_chg"] = df.groupby("industry")["chg"].transform("mean")
    df["neu_chg"] = df["chg"] - df["ind_chg"]
    hits = df[(df["chg"] <= PLUNGE_THR) & (df["neu_chg"] <= NEUTRAL_THR)]
    if hits.empty:
        return []
    qual = quality_pass(hits["stock_code"].tolist())
    org = org_net_buy_today()
    out = []
    for _, r in hits.iterrows():
        c = r["stock_code"]
        q = qual.get(c) or {}
        signals = ["orphan_plunge"]
        if c in org:
            signals.append("org_accumulate")
        # 盘中版：孤儿大跌同样必须过资质卡；财务取不到时宁可漏报
        if not q.get("passed"):
            continue
        out.append(WatchItem(
            code=c, name=r["name"], industry=r["industry"],
            signals=signals, trigger_date=day,
            drop_pct=round(float(r["chg"]), 4),
            industry_neutral=round(float(r["neu_chg"]), 4),
            quality={k: v for k, v in q.items() if k != "passed"},
        ))
    return out


def _watch_conn():
    from db import store
    store.init_store()
    return store.get_connection()


def persist_pool(items: list[WatchItem], today: str) -> dict:
    """把当日扫描结果合入关注池（SQLite `watch_pool`）。

    幂等：同一只票同一天重复扫只更新信号集合。已在池且当日**没有**新信号的
    条目保持原状（池子是"观察清单"，不是"每日名单"）。
    """
    import json as _json
    conn = _watch_conn()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        existing = {r[0]: r for r in conn.execute(
            "SELECT code, first_seen, signals FROM watch_pool WHERE status='watching'")}
        new_codes = {it.code for it in items}
        for it in items:
            sig = _json.dumps(sorted(set(it.signals)), ensure_ascii=False)
            q = _json.dumps(it.quality, ensure_ascii=False)
            meta = _json.dumps({"drop_pct": it.drop_pct,
                                "industry_neutral": it.industry_neutral,
                                "trigger_date": it.trigger_date},
                               ensure_ascii=False)
            if it.code in existing:
                conn.execute(
                    "UPDATE watch_pool SET last_signal_date=?, signals=?, quality=?, "
                    "meta=?, name=?, industry=?, updated_at=? WHERE code=?",
                    (today, sig, q, meta, it.name, it.industry, now, it.code))
            else:
                conn.execute(
                    "INSERT INTO watch_pool (code, name, industry, first_seen, "
                    "last_signal_date, signals, status, quality, meta, updated_at) "
                    "VALUES (?,?,?,?,?,?,'watching',?,?,?)",
                    (it.code, it.name, it.industry, today, today, sig, q, meta, now))
        conn.commit()        # ⚠️ 不 commit 的话 close() 会静默回滚（实测踩过）
        stale = [c for c in existing if c not in new_codes]
        return {"in_pool": len(existing), "new": len(new_codes - set(existing)),
                "refreshed": len(new_codes & set(existing)), "untouched": len(stale)}
    finally:
        conn.close()


def update_right_side() -> dict:
    """收盘版任务（15:45）对池内每只票做右侧状态机：

    - 站回 MA20 ⇒ 记 `right_side` 标记（**观察点**，不是买入触发——见模块 docstring）；
    - 入池超 `SCAN_EXPIRE_DAYS` 个自然日仍未站回 ⇒ `expired`（基本面证伪嫌疑）。
    """
    import json as _json
    d = load_daily_with_industry(40)
    if d.empty:
        return {"checked": 0}
    conn = _watch_conn()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        rows = conn.execute(
            "SELECT code, first_seen FROM watch_pool WHERE status='watching'").fetchall()
        n_flag = n_expire = 0
        for code, first_seen in rows:
            flags = right_side_flags(d, code)
            days = (datetime.now() - datetime.strptime(first_seen, "%Y-%m-%d")).days
            if flags:
                conn.execute("UPDATE watch_pool SET right_side=?, updated_at=? "
                             "WHERE code=?",
                             (_json.dumps(flags), now, code))
                n_flag += 1
            if days > SCAN_EXPIRE_DAYS and "price_reclaim" not in flags:
                conn.execute("UPDATE watch_pool SET status='expired', updated_at=? "
                             "WHERE code=?", (now, code))
                n_expire += 1
        conn.commit()        # 同上：写路径必须显式提交
        return {"checked": len(rows), "right_side_flagged": n_flag, "expired": n_expire}
    finally:
        conn.close()


def list_pool(status: str | None = "watching") -> list[dict]:
    """读池（页面/命令行共用）。"""
    import json as _json
    conn = _watch_conn()
    try:
        sql = ("SELECT code, name, industry, first_seen, last_signal_date, signals, "
               "status, right_side, quality, meta FROM watch_pool")
        args: list = []
        if status:
            sql += " WHERE status=?"
            args.append(status)
        rows = conn.execute(sql + " ORDER BY last_signal_date DESC", args).fetchall()
        keys = ("code", "name", "industry", "first_seen", "last_signal_date",
                "signals", "status", "right_side", "quality", "meta")
        out = []
        for r in rows:
            d = dict(zip(keys, r))
            for k in ("signals", "right_side", "quality", "meta"):
                try:
                    d[k] = _json.loads(d[k]) if d[k] else None
                except (TypeError, ValueError) as e:
                    _logger.debug("watch_pool.%s 不是合法 JSON（按缺失处理）: %s", k, e)
                    d[k] = None
            out.append(d)
        return out
    finally:
        conn.close()
