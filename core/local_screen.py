"""本地五路初筛 —— 替代已失效的问财（2026-10-09 本地化）

## 为什么必须做
pywencai 实测已失效（`safe_get` 报 `'NoneType' object has no attribute 'get'`，
是**响应体解析失败**而非网络不可达）。而 `core/screener.py` 原实现把问财
当作唯一初筛入口，于是**每一次选股都降级到 `SECTOR_HARDCODED` 前 18 只硬编码池**：
- `screen_picks` 最近 8 批的 `strategy` 全部是「内置精选池」
- 选股结果跨日高度重复（TOP5 4/5 重合）、同日多次运行 5/5 完全一致
- 招商银行出现 17 次 / 5 个交易日、工商银行 17 次、景嘉微 14 次

⇒ 候选域被锁死在 18 只，这是"每次都选同几只"的**根因**（详见
`output/选股重复度过高_根因分析与改进方案.md`）。

## 本模块怎么做
用**本地已落库的全市场数据**实现五路初筛，候选域从 18 只回到全市场（2600+ 只）：

| 路 | 数据源 | 条件 |
|---|---|---|
| 主力资金 | `fund_flow_snapshot`（5210 行/日） | 近 5 日主力净流入排序取前 N |
| 低估值成长 | `market_snapshot`（**5572 行/日**，全量域） | 0 < PE < 30，按 PE 升序 |
| 技术突破 | `market_snapshot` | 当日涨幅 0.5%~7% + 换手 1%~15%，按成交额降序 |
| 行业龙头 | `market_snapshot` | 总市值 > 200 亿元，按市值降序 |
| 龙虎榜活跃 | —— | **无本地数据源 → 显式 `unavailable`**（不静默跳过） |

⚠️ **2026-10-26 起 `market_snapshot` 从"新浪 49 行业明细"（2,624 行 / 47% 覆盖）换成
   "全量域"（`data/full_market`，5,572 行 / 99.9% 覆盖，带 PE/PB/市值/换手）。
   此前创业板只有 31%、科创板只有 5% 在域内，`300750`、`688111` **根本选不到** ——
   五路条件一个字没改，候选域却从半个市场变成整个市场，这是"数据源覆盖率"
   而不是"条件松紧"决定的。**改条件前先看覆盖率**，别把域缺失误读成条件太严。

## 三条铁律
1. **单位是元**（⚠️ 本项目的经典坑）：`market_snapshot.market_cap` 与
   `fund_flow_snapshot.net_*` 单位都是**元**，不是亿元。200 亿元 = `2e10`。
2. **降配必须可见**：本地条件比原问财语句窄（如"净利润同比/ROE"无数据源），
   每条路都要给出 `note` 说明"与原条件的差异"，由报告展示。
3. **缺就是缺**：数据源无当日快照 → 该路 `unavailable` + 原因；不拿旧日期数据充数。
4. ⚠️ **PE 口径坑（2026-10-09 实测）**：本模块用的是 `market_snapshot.pe`
   （**新浪** per），而因子评分里 `_score_pe` 用的是 `get_stock_info()['pe_ratio']`
   （**东财**）——两者**不是同一个数**：长鑫科技(688825) 1700.0 vs 42.32（差 40 倍）、
   茅台 19.235 vs 19.39。因此**不能**把评分的 PE 换成快照 PE（曾试过，省 60% 耗时，
   但会让评分口径漂移且与历史 `screen_picks` 不可比，已回退）。
   要用必须先做口径对齐（同源同时点）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from db import store

_logger = logging.getLogger(__name__)

# 每路取前 N。
# ⚠️ 必须满足 `4 × STRATEGY_TOP_N > TOTAL_LIMIT`，否则上限**形同虚设**
#    （曾出现 4×25=100 < TOTAL_LIMIT=120：无论怎么跑都到不了上限，
#      `truncated` 恒为 0，测试 test_union_is_round_robin_truncated 当场抓到）。
#    并发预取落地后单只取数 1.57s，35×4=140 只 ≈ 3.7 分钟，可接受。
STRATEGY_TOP_N = 35
# 五路并集上限（按路轮流截断，保证各路均衡而不是被单一路占满）。
# ⚠️ 直接决定选股耗时：因子精筛每只约 1 次 K 线请求（实测 ≈ 9.6s/只，受网络影响）。
#    2026-10-09 P0 已实施**并发预取**（data/prefetch.py：8 并发 + 5 req/s 限速，
#    取数与算因子分离）——联网耗时从 9.6s/只 降到约 0.1s/只（摊薄），
#    因此候选域从 40 放大到 **120**，是原 18 只硬编码池的 6.7 倍。
TOTAL_LIMIT = 120

# 原问财语句（保留用于追溯"本地条件与原条件的差异"）
ORIGINAL_QUERIES = {
    "主力资金": "主力资金净流入前20，剔除ST，剔除上市不足60天",
    "技术突破": "MACD金叉，均线多头排列，5日涨幅3%-15%，剔除ST",
    "低估值成长": "市盈率<TTM>小于30，净利润同比增长大于20%，ROE大于10%，剔除ST",
    "龙虎榜活跃": "近5日龙虎榜上榜，机构净买入大于1000万，剔除ST",
    "行业龙头": "行业龙头，总市值大于200亿，近20日涨幅小于15%，剔除ST",
}

# 单位换算（写死在常量里并注释，避免到处散落 magic number）
YI = 1e8                      # 1 亿元 = 1e8 元
MIN_MKTCAP_YUAN = 200 * YI    # 龙头路：200 亿元


def _is_st(name: str) -> bool:
    """ST / *ST / 退市（原问财每路都带"剔除ST"，本地化后必须自己剔）"""
    n = str(name or "").upper()
    return "ST" in n or "退" in n


@dataclass
class Route:
    label: str
    status: str = "ok"            # ok | unavailable
    note: str = ""                # 与原问财条件的差异 / 不可用原因
    count: int = 0
    codes: list = field(default_factory=list)


def _latest_dates() -> tuple[str | None, str | None]:
    """(估值快照最新日, 资金流快照最新日) —— 各自独立取，缺失时分开报。"""
    conn = store.get_connection()
    try:
        mv = conn.execute("SELECT MAX(snapshot_date) AS d FROM market_snapshot").fetchone()
        fv = conn.execute("SELECT MAX(snapshot_date) AS d FROM fund_flow_snapshot").fetchone()
        return (mv["d"] if mv else None), (fv["d"] if fv else None)
    finally:
        conn.close()


def _route_fund_flow(top_n: int, data_date: str | None) -> Route:
    r = Route("主力资金",
              note="本地条件：近5日主力净流入排序取前 N"
                   "（原问财为「净流入前20，剔除上市不足60天」——上市天数无本地数据源）")
    if not data_date:
        r.status, r.note = "unavailable", "无资金流快照（15:40 采集任务未跑或刚清库）"
        return r
    conn = store.get_connection()
    try:
        rows = conn.execute(
            """SELECT stock_code, stock_name, net_5d FROM fund_flow_snapshot
               WHERE snapshot_date = ? AND net_5d IS NOT NULL
               ORDER BY net_5d DESC LIMIT ?""",
            (data_date, top_n * 3)).fetchall()
    finally:
        conn.close()
    r.codes = [{"股票代码": x["stock_code"], "股票名称": x["stock_name"] or "",
                "策略": r.label}
               for x in rows if not _is_st(x["stock_name"])][:top_n]
    r.count = len(r.codes)
    if not r.codes:
        r.status, r.note = "unavailable", f"{data_date} 资金流快照无有效行"
    return r


def _route_snapshot(label: str, where: str, order: str, top_n: int,
                    data_date: str | None, note: str) -> Route:
    """快照类三路的公共实现（低估值 / 技术突破 / 龙头）。

    ⚠️ 名称里可能含 ST → 多取 3 倍再在 Python 侧剔除（ST 占比通常 <5%）。
    """
    r = Route(label, note=note)
    if not data_date:
        r.status, r.note = "unavailable", "无估值快照（15:40 采集任务未跑或刚清库）"
        return r
    conn = store.get_connection()
    try:
        rows = conn.execute(
            f"""SELECT stock_code, stock_name, industry FROM market_snapshot
                WHERE snapshot_date = ? AND {where}
                ORDER BY {order} LIMIT ?""",
            (data_date, top_n * 3)).fetchall()
    finally:
        conn.close()
    r.codes = [{"股票代码": x["stock_code"], "股票名称": x["stock_name"] or "",
                "板块": x["industry"] or "", "策略": r.label}
               for x in rows if not _is_st(x["stock_name"])][:top_n]
    r.count = len(r.codes)
    if not r.codes:
        r.status, r.note = "unavailable", f"{data_date} 快照中无满足条件的股票"
    return r


def run_local_strategies(labels: list | None = None,
                         progress_cb=None) -> dict:
    """执行本地五路初筛。

    Args:
        labels: 只跑其中几路（None = 全部）。未知 label 会被忽略并记入 notes。
        progress_cb: 进度回调（与 screener 一致的 {"percent","message",...}）

    Returns:
        {"candidates": [{股票代码, 股票名称, 策略, ...}],
         "meta": {"routes": [Route.to_dict()], "truncated": int,
                  "notes": [...], "data_dates": {...}}}
    """
    mv_date, flow_date = _latest_dates()
    wanted = set(labels) if labels else None

    def _cb(pct, msg):
        if progress_cb:
            try:
                progress_cb({"percent": pct, "message": msg})
            except Exception:                       # noqa: BLE001
                pass

    routes: list[Route] = []
    _cb(6, "本地初筛：主力资金（近5日净流入）")
    if wanted is None or "主力资金" in wanted:
        routes.append(_route_fund_flow(STRATEGY_TOP_N, flow_date))
    _cb(8, "本地初筛：低估值成长（PE<30）")
    if wanted is None or "低估值成长" in wanted:
        routes.append(_route_snapshot(
            "低估值成长", "pe > 0 AND pe < 30", "pe ASC", STRATEGY_TOP_N, mv_date,
            note="本地条件：0<PE<30 按 PE 升序（原问财还含「净利润同比>20%、ROE>10%」"
                 "——**无本地数据源**，本路未包含这两个条件）"))
    _cb(10, "本地初筛：技术突破（涨幅+换手+成交额）")
    if wanted is None or "技术突破" in wanted:
        routes.append(_route_snapshot(
            "技术突破", "change_pct BETWEEN 0.5 AND 7 AND turnover BETWEEN 1 AND 15",
            "amount DESC", STRATEGY_TOP_N, mv_date,
            note="本地条件：当日涨幅 0.5%~7% + 换手 1%~15%，按成交额降序取前 N"
                 "（原问财的「MACD金叉」需逐票 K 线，由后续「均线排列/趋势」因子体现）"))
    _cb(12, "本地初筛：行业龙头（市值>200亿）")
    if wanted is None or "行业龙头" in wanted:
        routes.append(_route_snapshot(
            "行业龙头", f"market_cap > {MIN_MKTCAP_YUAN:.0f}", "market_cap DESC",
            STRATEGY_TOP_N, mv_date,
            note="本地条件：总市值 > 200 亿元（快照 market_cap 单位是**元**，已换算），"
                 "按市值降序（原问财的「近20日涨幅<15%」由趋势因子的过热扣分体现）"))
    # 龙虎榜：无本地数据源 —— 显式 unavailable，不静默跳过
    if wanted is None or "龙虎榜活跃" in wanted:
        routes.append(Route(
            "龙虎榜活跃", status="unavailable",
            note="**无本地数据源**（问财失效后龙虎榜无法获取）——该路本轮不参与，"
                 "候选域因此少一路覆盖；若要恢复需接入新的龙虎榜数据源"))

    notes = []
    if labels:
        unknown = [x for x in labels if x not in ORIGINAL_QUERIES]
        if unknown:
            notes.append(f"忽略未知策略：{unknown}")
    notes += [f"{r.label}：{r.note}" for r in routes if r.status == "unavailable"]

    # ── 并集 + 按路轮流截断（保证均衡，记录被截断数）──
    # ⚠️ 不能写 `while any(buckets)`——buckets 里的子列表**本身永远非空**，
    #    那样是死循环（首版实测挂死 180s 被 faulthandler 抓到）。
    #    循环条件必须是"还有未取下标"，即 idx 小于最长子列表长度。
    seen: set = set()
    merged: list[dict] = []
    buckets = [[c for c in r.codes] for r in routes if r.status == "ok"]
    truncated = 0
    max_len = max((len(b) for b in buckets), default=0)
    for idx in range(max_len):
        for b in buckets:
            if idx >= len(b):
                continue
            item = b[idx]
            code = item["股票代码"]
            if code in seen:
                continue
            if len(merged) >= TOTAL_LIMIT:
                truncated += 1
                continue
            seen.add(code)
            merged.append(item)

    meta = {
        "routes": [{"label": r.label, "status": r.status, "count": r.count,
                    "note": r.note, "query": ORIGINAL_QUERIES.get(r.label, "")}
                   for r in routes],
        "truncated": truncated,
        "total_limit": TOTAL_LIMIT,
        "notes": notes,
        "data_dates": {"market_snapshot": mv_date, "fund_flow_snapshot": flow_date},
        "source": "local",
    }
    _logger.info("[本地初筛] 五路合并 %d 只（截断 %d）；各路: %s",
                 len(merged), truncated,
                 {r.label: f"{r.status}/{r.count}" for r in routes})
    return {"candidates": merged, "meta": meta}
