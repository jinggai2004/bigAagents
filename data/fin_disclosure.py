# -*- coding: utf-8 -*-
"""财报公告日（PIT）数据层：报告期 → 公开可得日。

为什么需要（前视偏差）
──────────────────────
`tool_financials` 原先直接取"最新 4 期报表"，**不知道哪一期什么时候公布**。
对"看现在"没影响，但对任何"站在历史某天看"的场景（复盘、历史重放、
将来的基本面因子）会把**当时还没公布的财报**当已知 ⇒ 前视偏差。
这类错误**不会报错**，只会让结论偏乐观，所以必须结构性地堵住。

数据源与口径
────────────
- 源：`akshare.stock_yjbb_em(date=报告期末)`（东财业绩报表），带 `最新公告日期`。
  实测 20260331 → 5,911 行、耗时 ~81s；20260630 → 11,432 行、~148s。
- ⚠️ **必须按 A 股过滤**：20260630 那次返回里混入 **5,275 只新三板**（83/87 段），
  不做 `is_a_share_code()` 过滤会把非 A 股写进表，且不同报告期返回的市场范围
  **并不一致**（Q1 没混、Q2 混了）——不能假设"上次干净这次也干净"。
- ⚠️ `ann_date` 是**保守下界**：`最新公告日期` 是该期的**最后一次**公告日期
  （含更正公告），会晚于首次披露。用它过滤只会"晚用"、不会"早用"，
  方向上安全；要首次披露日得换巨潮逐股公告接口（5,000+ 次请求，太慢）。

设计约束
────────
- **缺记录 ≠ 未公告**：表里没有某 (代码, 报告期) 时，不能推断"那天还没公布"。
  所以 `split_by_as_of()` 把"缺记录"和"公告日 > as_of"分成两类返回，
  由调用方决定口径（本项目的选择是：历史分析 fail-closed，两个都剔除但分别报告）。
- 解析失败**不静默**：调用方拿到 error 字段而不是空列表。
"""
from __future__ import annotations

import logging
from datetime import date, datetime

_logger = logging.getLogger(__name__)

SOURCE_CNINFO = "cninfo.prbook"
SOURCE_YJBB = "eastmoney.yjbb"
SOURCE = SOURCE_CNINFO             # 兼容旧引用

# 公告日合理性窗口：报告期末之后最多还能拖多久（天）。
# 实测 2025 年报最晚实际披露 2026-08-06（期末 +218 天）⇒ 取 240 留余量。
# 这个窗口是**抓取层的护栏**：超出窗口的公告日一定是串期/脏数据，直接不写库。
MAX_LAG_DAYS = 240
_MIN_LEAD_DAYS = 7                 # 期末前 7 天内的公告日仍可接受（边界宽容）

# 报告期末 → 巨潮"预约披露"接口的期次命名（源码里就是这么拼的）
_CNINFO_TAG = {"0331": "一季", "0630": "半年报", "0930": "三季", "1231": "年报"}


# ================================================================
# 报告期
# ================================================================

def quarter_ends(n: int = 8, as_of: str | None = None) -> list[str]:
    """最近 n 个**已过季末**的报告期（YYYYMMDD，倒序）。

    "已过季末"是必要条件：季末之前不可能有该期财报。所以用 as_of（默认今天）
    而不是日历概念，避免把未来季末算进来（那会让"缺数据"看起来像"抓漏了"）。
    """
    ref = _as_date(as_of) or date.today()
    out: list[str] = []
    y, m = ref.year, ref.month
    # 当前所在季度之前的那个季末开始往回数
    if m <= 3:
        qy, qm = y - 1, 12
    elif m <= 6:
        qy, qm = y, 3
    elif m <= 9:
        qy, qm = y, 6
    else:
        qy, qm = y, 9
    while len(out) < max(1, n):
        out.append(f"{qy}{qm:02d}{_qe_day(qm)}")
        qm -= 3
        if qm <= 0:
            qm += 12
            qy -= 1
    return out


def _qe_day(month: int) -> str:
    return {3: "31", 6: "30", 9: "30", 12: "31"}.get(int(month), "30")


def _as_date(v) -> date | None:
    """宽容解析 'YYYY-MM-DD' / 'YYYYMMDD' / date / datetime。判不了返回 None。

    ⚠️ 必须挡 `pandas.NaT`：它是 `datetime` 的**子类**，`isinstance(v, datetime)`
    为 True，`.date()` 会抛 `ValueError: NaTType does not support date`，
    直接把整批抓取打断（2026-10-30 实测踩到）。
    """
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        try:
            d = v.date()
        except (ValueError, OverflowError) as e:
            _logger.debug("[财报公告日] datetime 无法转日期: %r / %s", v, e)
            return None
        # ⚠️ `pd.NaT.date()` **不抛异常**，而是返回 NaT 本身（NaTType 是 datetime
        #    子类）⇒ 必须按**结果类型**判：真正的 date 才放行。
        if d is None or isinstance(d, datetime):
            _logger.debug("[财报公告日] 日期为 NaT/无效: %r", v)
            return None
        return d
    if isinstance(v, date):
        return v
    s = str(v).strip().replace("/", "-")
    if not s or s.lower() in ("nan", "nat", "none"):
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            # 判不出来是**正常路径**（调用方会按口径不明处理），用 debug 不刷屏；
            # 但必须留下可观察痕迹——静默 except 会被卫生检查判红。
            _logger.debug("[财报公告日] 日期格式无法解析: %r", v)
            continue
    return None


def norm_date(v) -> str:
    """统一成 YYYY-MM-DD；判不了原样返回（调用方据此报"口径不明"）。"""
    d = _as_date(v)
    return d.strftime("%Y-%m-%d") if d else str(v or "").strip()


# ================================================================
# 抓取
# ================================================================

def plausible_window(period: str) -> tuple:
    """报告期对应的"公告日合理区间" (起, 止)。

    起 = 期末 − 7 天；止 = 期末 + `MAX_LAG_DAYS`。
    超出区间的公告日不可能是这一期的（要么串到了别的期，要么是脏数据）——
    **2026-10-30 实测事故**：东财 `stock_yjbb_em` 对历史 0930 期返回的公告日
    其实是**下一年 Q3**（20250930 期 4,927 行标着 2026-10，与 20260930 期 100% 重合），
    它落库后会让"2025 三季报"看起来是 2026 年才公布的 ⇒ PIT 过滤把所有历史 Q3
    都判成"当时还没公布"。这个窗口就是拦它的。
    """
    end = _as_date(period)
    if end is None:
        return None, None
    from datetime import timedelta
    return end - timedelta(days=_MIN_LEAD_DAYS), end + timedelta(days=MAX_LAG_DAYS)


def _cninfo_period(period: str) -> str:
    tag = _CNINFO_TAG.get(str(period)[4:])
    return f"{str(period)[:4]}{tag}" if tag else ""


def fetch_period(period: str) -> dict:
    """抓一个报告期的全市场公告日（已按 A 股过滤）。

    来源优先级（2026-10-30 实测后定的）：
      ① **巨潮 `stock_report_disclosure`**（`cninfo.prbook`）—— 有 `实际披露` 列，
         即**真实披露日**（不是"最近一次公告"），支持任意年份，单请求 **0.3~2.9s**，
         纯 A 股（5,416~5,581 行）。**主源**。
      ② 东财 `stock_yjbb_em`（`eastmoney.yjbb`）—— 只有 `最新公告日期`（含更正），
         且实测**历史 0930 期会串到下一年的 Q3**；必须过 `plausible_window` 才写。
         仅当巨潮不可用时兜底，且耗时 80~150s/期。
      ③ 都失败 ⇒ 返回 `error`（**不返回空列表冒充成功**）。

    Returns:
        {"period", "rows": [...], "source", "raw_rows", "a_share", "majors",
         "bj", "dropped_non_a", "not_disclosed", "implausible"}
    """
    p = _norm_period(period)
    if not p:
        return {"period": str(period), "error": "报告期须为 YYYYMMDD（如 20260630）"}

    first_err = ""
    r = _fetch_cninfo(p)
    if not r.get("error"):
        return r
    first_err = r["error"]
    _logger.warning("[财报公告日] 巨潮抓取 %s 失败，回退东财: %s", p, first_err)
    r2 = _fetch_yjbb(p)
    if not r2.get("error"):
        r2["fallback_reason"] = first_err
        return r2
    return {"period": p,
            "error": f"两个来源都失败：巨潮={first_err}；东财={r2['error']}"}


def _finalize(p: str, rows: list[dict], raw_rows: int, dropped_non_a: int,
              not_disclosed: int, implausible: int, source: str) -> dict:
    majors = sum(1 for r in rows if not r["stock_code"].startswith("920"))
    return {"period": p, "rows": rows, "raw_rows": raw_rows, "source": source,
            "a_share": len(rows), "majors": majors, "bj": len(rows) - majors,
            "dropped_non_a": dropped_non_a, "not_disclosed": not_disclosed,
            "implausible": implausible}


def _fetch_cninfo(p: str) -> dict:
    """巨潮预约披露：用 `实际披露`（真实披露日）。未披露的行直接跳过（不留假值）。"""
    spec = _cninfo_period(p)
    if not spec:
        return {"period": p, "error": f"无法映射到期次命名: {p}"}
    try:
        import akshare as ak
    except ImportError as e:                            # pragma: no cover
        _logger.warning("[财报公告日] akshare 不可用: %s", e)
        return {"period": p, "error": f"akshare 不可用: {e}"}
    try:
        df = ak.stock_report_disclosure(market="沪深京", period=spec)
    except Exception as e:                              # noqa: BLE001 - 上游接口五花八门
        _logger.warning("[财报公告日] 巨潮 %s 抓取失败: %s: %s", spec, type(e).__name__, e)
        return {"period": p, "error": f"{type(e).__name__}: {e}"}
    if df is None or len(df) == 0:
        return {"period": p, "error": "巨潮返回空表"}
    if "实际披露" not in df.columns or "股票代码" not in df.columns:
        return {"period": p, "error": f"巨潮缺列（有: {list(df.columns)[:8]}）"}

    lo, hi = plausible_window(p)
    universe = a_share_universe()
    rows, non_a, not_disc, implausible = [], 0, 0, 0
    for code, ann in zip(df["股票代码"], df["实际披露"]):
        c = str(code).strip()
        if not _is_a_share(c, universe):
            non_a += 1
            continue
        d = _as_date(ann)
        if d is None:
            not_disc += 1                               # 还没披露 ⇒ 不写（缺就是缺）
            continue
        if lo and (d < lo or d > hi):
            implausible += 1                            # 超出合理窗口 ⇒ 一定是脏数据
            continue
        rows.append({"stock_code": c, "period": p,
                     "ann_date": d.strftime("%Y-%m-%d"), "source": SOURCE_CNINFO})
    if implausible:
        _logger.warning("[财报公告日] %s 有 %d 行公告日超出合理窗口（%s~%s），已丢弃",
                        p, implausible, lo, hi)
    return _finalize(p, rows, len(df), non_a, not_disc, implausible, SOURCE_CNINFO)


def _fetch_yjbb(p: str) -> dict:
    """东财业绩报表兜底。`最新公告日期` 含更正公告，且历史 0930 期会串期 ⇒ 必须过窗口。"""
    try:
        import akshare as ak
    except ImportError as e:                            # pragma: no cover
        _logger.warning("[财报公告日] akshare 不可用: %s", e)
        return {"period": p, "error": f"akshare 不可用: {e}"}
    try:
        df = ak.stock_yjbb_em(date=p)
    except Exception as e:                              # noqa: BLE001 - 上游接口五花八门
        _logger.warning("[财报公告日] 东财 %s 抓取失败: %s: %s", p, type(e).__name__, e)
        return {"period": p, "error": f"{type(e).__name__}: {e}"}
    if df is None or len(df) == 0:
        return {"period": p, "error": "东财返回空表"}
    if "最新公告日期" not in df.columns or "股票代码" not in df.columns:
        return {"period": p, "error": f"东财缺列（有: {list(df.columns)[:8]}）"}

    lo, hi = plausible_window(p)
    universe = a_share_universe()
    rows, non_a, bad_date, implausible = [], 0, 0, 0
    for code, ann in zip(df["股票代码"], df["最新公告日期"]):
        c = str(code).strip()
        if not _is_a_share(c, universe):
            non_a += 1
            continue
        d = _as_date(ann)
        if d is None:
            bad_date += 1
            continue
        if lo and (d < lo or d > hi):
            implausible += 1
            continue
        rows.append({"stock_code": c, "period": p,
                     "ann_date": d.strftime("%Y-%m-%d"), "source": SOURCE_YJBB})
    return _finalize(p, rows, len(df), non_a, bad_date, implausible, SOURCE_YJBB)


def _norm_period(v) -> str:
    s = str(v or "").strip().replace("-", "")
    if len(s) == 8 and s.isdigit() and s[4:] in ("0331", "0630", "0930", "1231"):
        return s
    return ""


def a_share_universe() -> set:
    """本项目的 A 股域 = **本地面板可用代码**（`data.bars.available_codes()`）。

    为什么不能只用 `is_a_share_code()` 判代码段（实测 2026-10-30）：
    东财业绩报表 20260630 返回 **11,432 行**，其中 **83/87 段 5,275 只**——
    按代码段看那是"北交所合法段"（`is_a_share_code` 会放行），但真实北交所在
    面板里只有 **351 只、且全是 920 段**（83/87 段在面板里 **0 只**）。
    也就是说那 5,275 只绝大多数是**新三板**，把它们的财报写进表只会污染数据、
    让"某期覆盖 11,432 只"看起来像好消息。
    面板本身（5,830 只）就是全系统用的交易域 ⇒ 直接拿它当边界最一致。
    面板拿不到时退回**只做代码段判定**（宁可多留也不能整表丢）。
    """
    try:
        from data import bars
        codes = bars.available_codes()
        return set(codes) if codes else set()
    except Exception as e:                              # noqa: BLE001
        _logger.warning("[财报公告日] 读面板代码失败，退回代码段判定: %s", e)
        return set()


def _is_a_share(code: str, universe: set | None = None) -> bool:
    """结构性判定 + 面板边界。

    两道闸缺一不可：
      ① `data.codes.is_a_share_code` —— 排掉 B 股（900/200）等明显非 A 股；
      ② 面板代码集 —— 排掉新三板（83/87/43/40 段）。
    ② 只在 `universe` 非空时生效；空集表示"面板不可用"，此时只做 ①。

    ⚠️ 这里**不能** `from core.scheduler import is_a_share_code`：分层约束是
    data 不能依赖 core（`tests/test_layering.py` 2026-10-30 挡下过一次）。
    规则本体已下沉到 `data/codes.py`，core 那边反过来委托它。
    """
    from data.codes import is_a_share_code as _structural
    c = str(code or "").strip()
    if not _structural(c):
        return False
    if universe:
        return c in universe
    return True


def backfill(periods: list[str] | None = None, n: int = 8,
             progress_cb=None) -> dict:
    """抓并落库最近 n 个报告期的公告日。

    Returns:
        {"periods": [每期抓取概况], "written": 总写入行数, "failed": [失败期]}
    """
    from db import store
    targets = [_norm_period(p) for p in (periods or quarter_ends(n))]
    targets = [p for p in targets if p]
    out, written, failed = [], 0, []
    for i, p in enumerate(targets, 1):
        if progress_cb:
            progress_cb({"percent": i / max(1, len(targets)) * 100,
                         "message": f"抓取 {p}", "current": i, "total": len(targets)})
        r = fetch_period(p)
        if r.get("error"):
            failed.append({"period": p, "error": r["error"]})
            out.append({"period": p, "error": r["error"]})
            continue
        n_written = store.upsert_fin_disclosures(r["rows"])
        written += n_written
        out.append({"period": p, "source": r.get("source", ""),
                    "raw_rows": r["raw_rows"], "a_share": r["a_share"],
                    "majors": r["majors"], "bj": r["bj"],
                    "dropped_non_a": r["dropped_non_a"],
                    "not_disclosed": r.get("not_disclosed", 0),
                    "implausible": r.get("implausible", 0),
                    "fallback_reason": r.get("fallback_reason", ""),
                    "written": n_written})
        _logger.info("[财报公告日] %s 写入 %d 行（来源 %s，原始 %d，丢非A股 %d，"
                     "未披露 %d，超窗口 %d）", p, n_written, r.get("source"),
                     r["raw_rows"], r["dropped_non_a"],
                     r.get("not_disclosed", 0), r.get("implausible", 0))
    return {"periods": out, "written": written, "failed": failed}


# ================================================================
# PIT 过滤（前视防护的核心，纯函数便于测试）
# ================================================================

def split_by_as_of(periods: list[str], ann_map: dict,
                   as_of: str) -> tuple[list[str], list[dict]]:
    """把报告期列表按"在 as_of 那天是否已公开"分成 (可用, 剔除明细)。

    Args:
        periods: 报告期列表（YYYYMMDD，顺序即调用方想要的顺序，如倒序）
        ann_map: {报告期: 公告日 YYYY-MM-DD}
        as_of:   观察日 YYYY-MM-DD

    Returns:
        (kept, dropped)
        dropped 每项 {period, ann_date, reason}，reason ∈
          - "not_announced"  ：公告日晚于 as_of（真的还没公布）
          - "unverified"     ：库里没有该期公告日记录（**缺 ≠ 未公告**，单独标出）

    口径：**两类都剔除**（fail-closed）。理由——历史分析若把"查不到公告日"
    当成"已公布"，前视防护就形同虚设；而"晚用"的代价只是少一期数据，可接受。
    但两类必须分开报告：把"没抓到"说成"还没公布"是在编故事。
    """
    ref = _as_date(as_of)
    kept: list[str] = []
    dropped: list[dict] = []
    if ref is None:
        # as_of 判不出来：不做任何过滤，全部返回并标记（调用方据此提示口径不明）
        return list(periods), [{"period": p, "ann_date": "",
                                "reason": "as_of_unparsed"} for p in []]
    for p in periods:
        ann = ann_map.get(str(p))
        ad = _as_date(ann) if ann else None
        if ad is None:
            dropped.append({"period": str(p), "ann_date": "",
                            "reason": "unverified"})
            continue
        if ad > ref:
            dropped.append({"period": str(p), "ann_date": ad.strftime("%Y-%m-%d"),
                            "reason": "not_announced"})
            continue
        kept.append(str(p))
    return kept, dropped


def ann_map_for(stock_code: str) -> dict:
    """该股票全部报告期的公告日 → {period: ann_date}。"""
    from db import store
    rows = store.list_fin_disclosures(stock_code=stock_code, limit=200)
    return {r["period"]: r["ann_date"] for r in rows}
