# -*- coding: utf-8 -*-
"""PIT 财务面板回填——东财 datacenter-web 业绩报表版（RPT_LICO_FN_CPD）。

为什么换源（2026-10-31，替代 BaoStock 版）
──────────────────────────────────────────
BaoStock 版按 (code, year, quarter) 逐季查询，36 次往返/只 × 5,197 只
实测 ≈ 16 秒/只 ⇒ **全市场 20 小时以上**，不可行。东财业绩报表按
**报告期**分页拉取（500 行/页 × ~12 页/报告期 × 0.3 秒/页），
36 个报告期全市场 ≈ **3 分钟**，数量级差距。

东财相对 BaoStock 的额外收益
────────────────────────────
1. **每季度都有营收**（TOTAL_OPERATE_INCOME）——BaoStock 的 MBRevenue
   只有 Q2/Q4（半年报/年报）有值，Q1/Q3 结构性缺失（实测覆盖 0.3%）；
2. **自带同比增速**（YSTZ 营收同比 / SJLTZ 净利同比）——不用自己差分累计值；
3. **覆盖北交所**（920/8x/4x）——BaoStock 结构性无北交所；
4. 每股经营现金流（MGJYXJJE）/每股净资产（BPS）/基本每股收益（BASIC_EPS）。

⚠️ 已知限制（必须记录，不能当"取到 0"）
────────────────────────────────────────
- `NOTICE_DATE` 是**首次公告日**（PIT 锚点），但数值是东财**最新修正版**
  （财报重述后数值会被覆盖）。BaoStock 同样返回最新值 ⇒ 两源 PIT 语义相同，
  没有变差；但严格意义上"公告时点看到的原始值"任何一源都拿不到。
- `net_profit` 是**归母净利润**（PARENT_NETPROFIT），BaoStock 是净利润
  （含少数股东损益）。归母是因子研究更常用的口径，单一源内部一致即可。
- `roe`：东财是加权平均 ROE（百分数 ⇒ 本脚本 /100 转小数），
  BaoStock 是平均净资产收益率（小数）。两源有 1% 量级差异 ⇒ **不混源**。
- `eps` 是报告期**累计** EPS（年报口径），不是 TTM —— 列名从 BaoStock 的
  `eps_ttm` 改为 `eps`，避免"列名同、口径不同"的假一致（铁律：列名同≠单位同）。
- `XSMLL`（销售毛利率）银行等金融股为 null ⇒ gp_margin 缺是正常的，不补 0。

存储（可断点续跑）
──────────────────
两层：
1. 原始层 `data/panel/financials_em_raw/<REPORTDATE>.parquet`
   —— 按报告期落盘（36 个文件），已存在则跳过。重跑只补缺的报告期；
2. 规范层 `data/panel/financials/<code>.parquet`
   —— 从原始层聚合出 per-code 文件（与 BaoStock 版同布局，
   `scripts/financial_factor_ic.py::load_financials()` 零改动）。

用法
────
    py scripts/backfill_financials_em.py               # 全量（约 3 分钟）
    py scripts/backfill_financials_em.py --from 2021   # 只拉 2021 年起
    py scripts/backfill_financials_em.py --rebuild     # 只从原始层重建规范层（不联网）
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RAW_ROOT = _ROOT / "data" / "panel" / "financials_em_raw"
FIN_DIR = _ROOT / "data" / "panel" / "financials"

API_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REPORT_NAME = "RPT_LICO_FN_CPD"
# 058001001 = A 股（实测不过滤时 count 11,622，含 B 股/其他类型；过滤后 5,580 且零重复）
A_SHARE_TYPE = "058001001"
PAGE_SIZE = 500            # 实测单页 0.2~0.4s；再大收益递减
REQUEST_TIMEOUT = 20
PAGE_RETRIES = 3
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# 东财字段 → 规范列名。⚠️ roe/gp_margin/ystz/sjltz 东财是百分数，落盘前 /100
# （铁律 §3：新浪 turnover 小数 vs BaoStock turn 百分数差 100 倍的同类坑）。
RENAME = {
    "NOTICE_DATE": "pub_date",
    "REPORTDATE": "stat_date",
    "WEIGHTAVG_ROE": "roe_pct",
    "XSMLL": "gp_margin_pct",
    "PARENT_NETPROFIT": "net_profit",
    "TOTAL_OPERATE_INCOME": "revenue",
    "BASIC_EPS": "eps",
    "YSTZ": "ystz_pct",
    "SJLTZ": "sjltz_pct",
    "BPS": "bps",
    "MGJYXJJE": "ocfps",
}
DROP_COLS = {"UPDATE_DATE", "QDATE", "DATATYPE", "DATAYEAR", "DATEMMDD",
             "EITIME", "SECUCODE", "ISNEW", "ASSIGNDSCRPT", "PAYYEAR",
             "PUBLISHNAME", "ZXGXL", "YSHZ", "SJLHZ", "ORG_CODE",
             "TRADE_MARKET_ZJG", "BOARD_NAME", "ORI_BOARD_CODE", "BOARD_CODE",
             "SECURITY_NAME_ABBR", "TRADE_MARKET", "TRADE_MARKET_CODE",
             "SECURITY_TYPE", "SECURITY_TYPE_CODE", "DEDUCT_BASIC_EPS"}

_logger = logging.getLogger("backfill_fin_em")


def _quarter_ends(from_year: int, to_year: int) -> list[str]:
    """报告期列表 'YYYY-03-31' …（Q3 只到当年已披露的，接口自然为空则跳过）。"""
    out = []
    for y in range(from_year, to_year + 1):
        for md in ("-03-31", "-06-30", "-09-30", "-12-31"):
            out.append(f"{y}{md}")
    return out


def _session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def _fetch_quarter(s, report_date: str) -> pd.DataFrame | None:
    """一个报告期的全部 A 股行。全部页失败返回 None（区分于真的空）。"""
    from data import offline_gate

    if offline_gate.is_offline():
        raise RuntimeError("离线模式（BIGA_OFFLINE=1）不联网拉财务数据")

    rows: list[dict] = []
    page, total_pages = 1, 1
    n_fail = 0
    while page <= total_pages:
        got = None
        for attempt in range(PAGE_RETRIES):
            try:
                r = s.get(API_URL, params={
                    "reportName": REPORT_NAME, "columns": "ALL",
                    "filter": f"(REPORTDATE='{report_date}')"
                              f"(SECURITY_TYPE_CODE=\"{A_SHARE_TYPE}\")",
                    "pageSize": PAGE_SIZE, "pageNumber": page,
                    "sortColumns": "SECURITY_CODE", "sortTypes": "1",
                }, timeout=REQUEST_TIMEOUT)
                payload = r.json()
                result = payload.get("result") or {}
                got = (result.get("data") or [],
                       int(result.get("pages") or 1),
                       bool(payload.get("success")))
                break
            except Exception as e:                    # noqa: BLE001
                _logger.warning("[%s] 第 %d 页第 %d 次失败 %s: %s",
                                report_date, page, attempt + 1,
                                type(e).__name__, e)
                time.sleep(1.0 * (attempt + 1))
        if got is None:
            n_fail += 1
            page += 1
            continue
        data, total_pages, ok = got
        if not ok and page == 1 and not data:
            # 未来报告期（尚未披露）接口返回 success=False 且空表 —— 这是
            # 正常的"还没到披露时间"，落空标记而不是报失败。
            if pd.Timestamp(report_date) > pd.Timestamp.now():
                _logger.info("[%s] 未披露（未来报告期），落空标记", report_date)
                return pd.DataFrame()
            _logger.error("[%s] 接口返回 success=False 且空表（filter/报告名可能失效）",
                          report_date)
            return None
        rows.extend(data)
        page += 1

    if not rows:
        # 该报告期确实没有披露（未来的 Q4）或全失败——用 n_fail 区分
        if n_fail:
            _logger.error("[%s] %d 页失败且无数据 ⇒ 返回 None（可重试）",
                          report_date, n_fail)
            return None
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """原始页 → 规范列。百分数转小数；缺就是缺（NaN），不补 0。"""
    if df.empty:
        return df
    keep = [c for c in RENAME if c in df.columns]
    out = df[["SECURITY_CODE"] + keep].rename(columns={"SECURITY_CODE": "code",
                                                       **RENAME})
    for c in ("pub_date", "stat_date"):
        out[c] = pd.to_datetime(out[c], errors="coerce")
    for c in out.columns:
        if c not in ("code", "pub_date", "stat_date"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # 百分数 → 小数（与 BaoStock 小数口径一致；单一源内部自洽）
    for src, dst in (("roe_pct", "roe"), ("gp_margin_pct", "gp_margin"),
                     ("ystz_pct", "ystz"), ("sjltz_pct", "sjltz")):
        out[dst] = out[src] / 100.0
    out = out.drop(columns=["roe_pct", "gp_margin_pct", "ystz_pct", "sjltz_pct"])
    # np_margin 现算（东财不给净利率；营收为 0/NaN 时如实 NaN）
    out["np_margin"] = out["net_profit"] / out["revenue"].where(out["revenue"] > 0)
    out["source"] = "em"
    out = out.dropna(subset=["pub_date", "stat_date"])
    out = out.sort_values(["code", "pub_date"])
    return out.reset_index(drop=True)


def fetch_all(from_year: int, to_year: int) -> int:
    """拉全部报告期到原始层。返回失败报告期数。"""
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    s = _session()
    n_fail = 0
    rds = _quarter_ends(from_year, to_year)
    t0 = time.time()
    for i, rd in enumerate(rds, 1):
        raw_path = RAW_ROOT / f"{rd}.parquet"
        if raw_path.exists():
            continue
        t = time.time()
        df = _fetch_quarter(s, rd)
        if df is None:
            n_fail += 1
            print(f"  [{i}/{len(rds)}] {rd} 失败（可重跑续补）", flush=True)
            continue
        if df.empty:
            # 未披露的报告期（未来季度）也落一个空标记文件，避免每次重查
            df.to_parquet(raw_path, index=False)
            print(f"  [{i}/{len(rds)}] {rd} 无数据（空标记）", flush=True)
            continue
        df.to_parquet(raw_path, index=False)
        print(f"  [{i}/{len(rds)}] {rd} {len(df)} 行  %.2fs" % (time.time() - t),
              flush=True)
    print(f"原始层完成：{len(rds)} 个报告期，失败 {n_fail}，"
          f"用时 {(time.time() - t0) / 60:.1f} 分 → {RAW_ROOT}", flush=True)
    return n_fail


def rebuild() -> int:
    """原始层 → 规范层 per-code 文件。不联网。"""
    raws = sorted(RAW_ROOT.glob("*.parquet"))
    if not raws:
        print(f"原始层为空：{RAW_ROOT}", flush=True)
        return 1
    frames = [pd.read_parquet(p) for p in raws]
    raw = pd.concat(frames, ignore_index=True)
    # 同一 (code, stat_date) 多版本：保留 NOTICE_DATE 最新的（修正版）
    # ⚠️ 去重发生在 _normalize 之前 ⇒ 用原始列名（NOTICE_DATE/REPORTDATE）
    raw = raw.sort_values(["SECURITY_CODE", "REPORTDATE", "NOTICE_DATE"])
    raw = raw.drop_duplicates(subset=["SECURITY_CODE", "REPORTDATE"], keep="last")
    fin = _normalize(raw)
    FIN_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for code, g in fin.groupby("code"):
        g.reset_index(drop=True).to_parquet(FIN_DIR / f"{code}.parquet", index=False)
        n += 1
    print(f"规范层完成：{n} 只 → {FIN_DIR}（合计 {len(fin)} 期）", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_year", type=int, default=2018)
    ap.add_argument("--to", dest="to_year", type=int, default=2026)
    ap.add_argument("--rebuild", action="store_true",
                    help="只从原始层重建规范层（不联网）")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if a.rebuild:
        return rebuild()
    n_fail = fetch_all(a.from_year, a.to_year)
    rc = rebuild()
    return rc or (1 if n_fail else 0)


if __name__ == "__main__":
    raise SystemExit(main())
