# -*- coding: utf-8 -*-
"""PIT 财务面板回填（BaoStock `query_profit_data`，自带公告日 pubDate）。

为什么是它
──────────
- `ak.stock_financial_analysis_indicator` **41.4 秒/只**（8 次分页），5,600 只 ≈ 64 小时，不可行；
- BaoStock `query_profit_data` **0.08 秒/季**，且字段自带 `pubDate`（公告日）
  与 `statDate`（报告期）——**PIT 对齐的唯一正确锚点**就是 pubDate，
  用报告期对齐等于提前知道自己未来才看到的财报（前视偏差）。
- 项目已在用 BaoStock（turn_share 回填同源），有降级链与限频经验。

字段（`query_profit_data` 全字段，实测 2026-10-30）
─────────────────────────────────────────────────
code / pubDate / statDate / roeAvg(ROE) / npMargin(净利率) / gpMargin(毛利率) /
netProfit(净利润) / epsTTM / MBRevenue(主营收入) / totalShare / liqaShare

⚠️ 已知限制（**必须记录，不能当成"取到 0"**）
────────────────────────────────────────────
- BaoStock **无北交所**（920/8x/4x）⇒ 这些股票 `unavailable`，不写 0；
- 并发上限 **4**（超了会连不上，不是"网络慢"）；
- 停牌/未披露季度可能返回空行 ⇒ 该季度缺失是正常的，不补 0；
- `MBRevenue` 在部分季度为空串（实测 600276 2025Q3 就是空的）⇒ 转 NaN。

存储（可断点续跑）
──────────────────
`data/panel/financials/<code>.parquet`，一只一文件。已存在的股票默认跳过，
`--force` 重取。这样中断后重跑不会从头来过（turn_share 回填同款策略）。

用法
────
    py scripts/backfill_financials.py --limit 40          # 先小样本验证
    py scripts/backfill_financials.py                     # 全市场（约 1 小时，4 并发）
    py scripts/backfill_financials.py --verify 600276     # 看单只的 PIT 对齐结果
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

PANEL_ROOT = _ROOT / "data" / "panel" / "financials"
PANEL = PANEL_ROOT          # 当前表的目标目录（run() 里按 table 改写）
WORKERS = 4                     # BaoStock 硬上限（并发 >4 会连不上）
# ── 三张表：各自的 BaoStock 查询名、字段、落盘子目录、列改名 ──
# profit 沿用旧目录（`data/panel/financials/<code>.parquet`），保持已回填的 621 只可直接复用。
TABLE_SPECS: dict[str, dict] = {
    "profit": {
        "query": "query_profit_data",
        "fields": ["code", "pubDate", "statDate", "roeAvg", "npMargin", "gpMargin",
                   "netProfit", "epsTTM", "MBRevenue", "totalShare", "liqaShare"],
        "subdir": "",
        "rename": {"roeAvg": "roe", "npMargin": "np_margin", "gpMargin": "gp_margin",
                   "netProfit": "net_profit", "epsTTM": "eps_ttm",
                   "MBRevenue": "revenue", "totalShare": "total_share",
                   "liqaShare": "liqa_share"},
    },
    "cashflow": {
        # CFOToNP = 经营现金流/净利润 ⇒ **盈利质量的直接度量**（应计操纵的镜像）
        "query": "query_cash_flow_data",
        "fields": ["code", "pubDate", "statDate", "CFOToNP", "CFOToOR", "CFOToGr",
                   "CAToAsset", "NCAToAsset", "tangibleAssetToAsset", "ebitToInterest"],
        "subdir": "cashflow",
        "rename": {"CFOToNP": "cfo_to_np", "CFOToOR": "cfo_to_or", "CFOToGr": "cfo_to_gr",
                   "CAToAsset": "ca_to_asset", "NCAToAsset": "nca_to_asset",
                   "tangibleAssetToAsset": "tangible_to_asset",
                   "ebitToInterest": "ebit_to_interest"},
    },
    "balance": {
        "query": "query_balance_data",
        "fields": ["code", "pubDate", "statDate", "currentRatio", "quickRatio",
                   "cashRatio", "YOYLiability", "liabilityToAsset", "assetToEquity"],
        "subdir": "balance",
        "rename": {"currentRatio": "current_ratio", "quickRatio": "quick_ratio",
                   "cashRatio": "cash_ratio", "YOYLiability": "yoy_liability",
                   "liabilityToAsset": "liability_to_asset",
                   "assetToEquity": "asset_to_equity"},
    },
}
DEFAULT_TABLE = "profit"
FIELDS = TABLE_SPECS[DEFAULT_TABLE]["fields"]        # 兼容旧引用

_logger = logging.getLogger("backfill_fin")

_stat = {"ok": 0, "empty": 0, "bad_code": 0, "error": 0}


def _table_dir(table: str):
    """该表的落盘目录。profit 沿用旧目录（已回填的 621 只可直接复用）。"""
    sub = TABLE_SPECS[table]["subdir"]
    return PANEL_ROOT / sub if sub else PANEL_ROOT


def _bs_code(code: str) -> str | None:
    """6 位代码 → BaoStock 代码。**北交所返回 None**（BaoStock 不覆盖）。"""
    c = str(code).strip()
    if len(c) != 6 or not c.isdigit():
        return None
    if c.startswith(("920", "83", "87", "88", "43", "82")):
        return None                                  # 北交所：无数据源，如实记 unavailable
    if c.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return f"sh.{c}"
    if c.startswith(("000", "001", "002", "003", "300", "301", "200")):
        return f"sz.{c}"
    return None


RETRIES = 3              # 单季最多尝试次数（BaoStock 会返回"接收数据异常"式限频）
BACKOFF_S = 0.6          # 退避基数（0.6/1.2/2.4 秒）


def _fetch_rows(bcode: str, years: list[int], table: str) -> tuple[list[dict], int, int]:
    """单只票、单张表的全部季度行。

    Returns:
        (rows, n_error, n_empty)
        - `n_error`：**连接/限频层失败**的季度数（`error_code != "0"` 或抛异常）；
        - `n_empty`：查询成功但**真的没有数据**的季度数。
        ⚠️ 两者必须分开：2026-10-31 实测 BaoStock 限频时返回
           `error_code != "0"`（日志「接收数据异常，请稍后再试」），
           若把它当"空数据"，会得到 `ok=5 / 空数据=603` 这种**看起来正常**的假结果
           （项目铁律：HTTP 200≠成功 ⇒ 必须解析业务码）。
    """
    import time

    import baostock as bs

    fn = getattr(bs, TABLE_SPECS[table]["query"])
    rows: list[dict] = []
    n_error = 0
    n_empty = 0
    for y in years:
        for q in (1, 2, 3, 4):
            rs = None
            for attempt in range(RETRIES):
                try:
                    rs = fn(code=bcode, year=y, quarter=q)
                except Exception as e:               # noqa: BLE001  连接层抖动
                    _logger.warning("[%s] %s %dQ%d 第 %d 次异常 %s: %s",
                                    bcode, table, y, q, attempt + 1, type(e).__name__, e)
                    rs = None
                    time.sleep(BACKOFF_S * (2 ** attempt))
                    continue
                if rs.error_code == "0":
                    break
                _logger.warning("[%s] %s %dQ%d 业务码 %s: %s（第 %d 次）",
                                bcode, table, y, q, rs.error_code, rs.error_msg, attempt + 1)
                time.sleep(BACKOFF_S * (2 ** attempt))
                rs = None
            if rs is None or rs.error_code != "0":
                n_error += 1
                continue
            got = 0
            while rs.next():
                rows.append(dict(zip(rs.fields, rs.get_row_data())))
                got += 1
            if got == 0:
                n_empty += 1
    return rows, n_error, n_empty


def _normalize(rows: list[dict], code: str, table: str) -> pd.DataFrame | None:
    """原始行 → 规范化表。空返回 None（不返回空表，避免和"取到 0 行"混淆）。"""
    if not rows:
        return None
    spec = TABLE_SPECS[table]
    df = pd.DataFrame(rows)
    df = df[[c for c in spec["fields"] if c in df.columns]].copy()
    num_cols = [c for c in df.columns if c not in ("code", "pubDate", "statDate")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c].replace("", None), errors="coerce")
    df = df.rename(columns={**spec["rename"], "pubDate": "pub_date",
                            "statDate": "stat_date"})
    df["code"] = str(code)
    df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce")
    df["stat_date"] = pd.to_datetime(df["stat_date"], errors="coerce")
    df = df.dropna(subset=["pub_date", "stat_date"]).sort_values("pub_date")
    df = df.drop_duplicates(subset=["stat_date"], keep="last")
    return df.reset_index(drop=True)


SOCK_TIMEOUT = 45        # 单个 socket 读写超时（秒）
WATCHDOG_TICK = 30       # 每 30 秒检查一次是否有结果返回
STALL_LIMIT = 300        # 连续 5 分钟零结果 ⇒ 判定卡死并中止
LOGIN_RETRIES = 3


def _worker_init() -> None:
    """每个子进程**自己登录一次**。

    ⚠️⚠️ BaoStock 的 Python 客户端是模块级单 socket：4 个**线程**共享一条连接
    会让响应交错——实测 000001 出现"Q2 净利 1.92 亿"（真值 253 亿）、
    整季缺失。**并发只能用多进程，每个进程独立 login**（2026-10-30 踩到）。

    ⚠️⚠️ **必须设 socket 超时**（2026-10-31 修）：BaoStock 内部用
    `socket.socket(AF_INET, SOCK_STREAM)` 建连且**不设 settimeout** ⇒ 默认阻塞、
    `recv()` 无限等待。限频时 worker 会**永久挂住**，主进程的 `as_completed`
    永远等不到，整个回填僵死——实测 10-30 那轮挂了 **16 小时 55 分**才因为
    手工杀进程而"退出"（还报 exit 0，看起来像正常完成）。
    `socket.setdefaulttimeout()` 对 `socket.socket()` 新建的连接生效（CPython 在
    `__init__` 里读 `_GLOBAL_DEFAULT_TIMEOUT`），所以放在 login **之前**即可。
    """
    import socket
    import time

    import baostock as bs

    socket.setdefaulttimeout(SOCK_TIMEOUT)
    for attempt in range(LOGIN_RETRIES):
        try:
            lg = bs.login()
            if getattr(lg, "error_code", "0") == "0":
                return
            _logger.warning("login 失败（第 %d 次）: %s", attempt + 1,
                            getattr(lg, "error_msg", "?"))
        except Exception as e:                       # noqa: BLE001
            _logger.warning("login 异常（第 %d 次）: %s: %s",
                            attempt + 1, type(e).__name__, e)
        time.sleep(2 * (attempt + 1))
    _logger.error("login 连续 %d 次失败，该 worker 将无法取数", LOGIN_RETRIES)


def _worker(payload: tuple[str, list[int], str]) -> tuple[str, str, str | None]:
    """返回 (code, status, 错误信息)。status ∈ ok/empty/bad_code/error。"""
    code, years, table = payload
    bcode = _bs_code(code)
    if bcode is None:
        return code, "bad_code", None
    try:
        rows, n_err, n_empty = _fetch_rows(bcode, years, table)
        df = _normalize(rows, code, table)
    except Exception as e:                          # noqa: BLE001
        _logger.warning("[%s] %s 规范化失败 %s: %s", code, table, type(e).__name__, e)
        return code, "error", f"{type(e).__name__}: {e}"
    if df is None or df.empty:
        # ⚠️ 有连接错误就报 error（可重试），不能报"空数据"（会看起来像真的没报表）
        if n_err:
            return code, "error", f"{n_err} 个季度连接/限频失败（{n_empty} 个季度确实无数据）"
        return code, "empty", None
    out = _table_dir(table)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{code}.parquet", index=False)
    return code, "ok", None


def run(limit: int | None, years: list[int], force: bool,
        sample: int | None = None, table: str = DEFAULT_TABLE) -> int:
    from concurrent.futures import (FIRST_COMPLETED, ProcessPoolExecutor,
                                   wait)

    from data import bars

    target = _table_dir(table)
    target.mkdir(parents=True, exist_ok=True)
    codes = sorted(bars.available_codes())
    if sample:
        # ⚠️ 不能用 codes[:N]：那是"深市 000/001 全部"，不是样本，会让 IC 结论
        #    只代表主板大盘。等距抽样能跨过 000→300→600→688 各板。
        step = max(1, len(codes) // sample)
        codes = codes[::step][:sample]
    elif limit:
        codes = codes[:limit]
    todo = [c for c in codes if force or not (target / f"{c}.parquet").exists()]
    print(f"[{table}] 候选 {len(codes)} 只，待回填 {len(todo)} 只，"
          f"年份 {years[0]}~{years[-1]}，{WORKERS} 进程 → {target}", flush=True)

    t0 = time.time()
    done_n = 0
    stalled = False
    ex = ProcessPoolExecutor(max_workers=WORKERS, initializer=_worker_init)
    try:
        futs = {ex.submit(_worker, (c, years, table)): c for c in todo}
        pending = set(futs)
        last_progress = time.time()
        while pending:
            # ⚠️⚠️ 必须用 `wait(timeout=)` 而不是 `as_completed`（2026-10-31 修）：
            #    `as_completed` 没有超时，worker 一旦卡在 socket read 上，
            #    主进程会**永远等下去**——实测僵死 16h55m，日志停在 600/1494、
            #    落盘文件停在 00:38，而进程还活着（占用 8 条 BaoStock 连接）。
            done, pending = wait(pending, timeout=WATCHDOG_TICK,
                                 return_when=FIRST_COMPLETED)
            if not done:
                idle = time.time() - last_progress
                if idle >= STALL_LIMIT:
                    stalled = True
                    print(f"\n  [STALL] {idle/60:.1f} 分钟没有任何结果返回 ⇒ "
                          f"判定卡死，中止（已完成 {done_n}/{len(todo)}）", flush=True)
                    _logger.error("回填卡死：%.1f 分钟无结果，已完成 %d/%d",
                                  idle / 60, done_n, len(todo))
                    break
                continue
            last_progress = time.time()
            for fut in done:
                done_n += 1
                try:
                    code, status, err = fut.result()
                except Exception as e:               # noqa: BLE001
                    _stat["error"] += 1
                    _logger.warning("子进程失败 %s: %s", type(e).__name__, e)
                    print(f"  [ERROR] 子进程失败: {type(e).__name__}: {e}", flush=True)
                    continue
                _stat[status] = _stat.get(status, 0) + 1
                if err:
                    print(f"  [ERROR] {code}: {err}", flush=True)
                if done_n % 200 == 0:
                    el = time.time() - t0
                    rate = done_n / el
                    print(f"  {done_n}/{len(todo)}  {rate:.1f} 只/s  "
                          f"已用 {el/60:.1f} 分  预计剩余 "
                          f"{(len(todo)-done_n)/max(rate,1e-9)/60:.1f} 分", flush=True)
    finally:
        # 卡死时 cancel 不会杀掉**正在跑**的 worker ⇒ 必须显式结束进程，
        # 否则它们会带着 BaoStock 连接一直挂着（10-30 那轮残留了 8 个）。
        if stalled:
            for pr in list(getattr(ex, "_processes", {}).values()):
                try:
                    pr.terminate()
                except Exception as e:               # noqa: BLE001
                    _logger.warning("终止 worker 失败: %s", e)
        ex.shutdown(wait=not stalled, cancel_futures=True)

    el = time.time() - t0
    print(f"\n{'中止' if stalled else '完成'}：ok={_stat['ok']} 空数据={_stat['empty']} "
          f"无数据源={_stat['bad_code']} 异常={_stat['error']} "
          f"已处理 {done_n}/{len(todo)} 用时 {el/60:.1f} 分")
    print(f"落盘目录：{target}")
    if stalled:
        # ⚠️ 卡死必须返回**非 0**：上一轮僵死 16 小时后报 exit 0，
        #    看起来像"正常跑完"，于是没人发现只做了 40%。
        print("⚠️ 未跑完（卡死中止）。已落盘的文件可直接复用，"
              "重跑同一命令会从缺口继续（已存则跳过）。")
        return 3
    if done_n < len(todo):
        print(f"⚠️ 仍有 {len(todo) - done_n} 只未处理")
        return 4
    return 0


def verify(code: str) -> int:
    """打印某只票的 PIT 对齐结果，并演示"某月末能看到哪些财报"。"""
    p = PANEL / f"{code}.parquet"
    if not p.exists():
        print(f"❌ 未回填：{p}")
        return 1
    df = pd.read_parquet(p)
    print(f"{code}: {len(df)} 期，区间 {df['stat_date'].min():%Y-%m-%d} ~ "
          f"{df['stat_date'].max():%Y-%m-%d}")
    print(df[["stat_date", "pub_date", "roe", "gp_margin", "np_margin",
              "net_profit", "revenue"]].tail(8).to_string(index=False))
    print("\nPIT 演示（2024-06-30 收盘时**实际已知**的最新一期）：")
    known = df[df["pub_date"] <= pd.Timestamp("2024-06-30")]
    if known.empty:
        print("  （无）")
    else:
        r = known.iloc[-1]
        print(f"  报告期 {r['stat_date']:%Y-%m-%d}，公告日 {r['pub_date']:%Y-%m-%d}，"
              f"ROE {r['roe']:.2%}" if pd.notna(r["roe"]) else "  ROE 缺失")
    later = df[(df["stat_date"] <= pd.Timestamp("2024-06-30"))
               & (df["pub_date"] > pd.Timestamp("2024-06-30"))]
    if len(later):
        print(f"  ⚠️ 同期还有 {len(later)} 期是**当时未公告**的（用报告期对齐就会前视）")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None,
                    help="跨板块等距抽 N 只（先验证 IC，别一上来全量）")
    ap.add_argument("--from-year", type=int, default=2018)
    ap.add_argument("--to-year", type=int, default=2026)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", default=None)
    ap.add_argument("--table", default=DEFAULT_TABLE,
                    choices=sorted(TABLE_SPECS), help="哪张报表")
    a = ap.parse_args(argv)
    if a.verify:
        return verify(a.verify)
    years = list(range(a.from_year, a.to_year + 1))
    return run(a.limit, years, a.force, a.sample, a.table)


if __name__ == "__main__":
    raise SystemExit(main())
