# -*- coding: utf-8 -*-
"""BaoStock 全市场日线面板回填（一次性，跑一晚）—— CLI 薄封装。

真实逻辑都在 `data/panel_store.py`（可注入、可测试）；本脚本只做参数解析、
真实取数与覆盖率落库。

用法：
    # ① 先看计划（不取数、不写盘）
    python scripts/backfill_panel_baostock.py

    # ② 小规模真跑 5 只（验证链路，≈10s）
    python scripts/backfill_panel_baostock.py --apply --limit 5

    # ③ 正式回填 10 年（沪深 ~5100 只，实测外推 ≈5 小时；支持断点续跑）
    python scripts/backfill_panel_baostock.py --apply --start 2016-01-01

    # 研究用途建议带退市股（防幸存者偏差）：
    python scripts/backfill_panel_baostock.py --apply --include-delisted

⚠️ 依赖：`pip install baostock`（0.9.4）；首次 import 前必须打 pandas2 垫片
   （`panel_store.apply_baostock_pandas_shim()`，本脚本已自动处理）。
"""
from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, __file__.rsplit("\\scripts", 1)[0] if "\\scripts" in __file__
                else __file__.split("/scripts")[0])

from data import panel_store as PS                          # noqa: E402

_logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BaoStock 全市场日线面板回填")
    p.add_argument("--start", default="2016-01-01", help="起始交易日（默认 2016-01-01）")
    p.add_argument("--end", default=None, help="结束日期（默认今天）")
    p.add_argument("--include-delisted", action="store_true",
                   help="包含退市股（因子研究建议开，防幸存者偏差；默认只在市）")
    p.add_argument("--limit", type=int, default=0, help="只回填前 N 只（0=不限，测试用）")
    p.add_argument("--batch-size", type=int, default=200,
                   help="每多少只落一个 parquet 分片（默认 200 ≈15MB/片）")
    p.add_argument("--apply", action="store_true",
                   help="真正执行（缺省只打印计划，不取数不写盘）")
    p.add_argument("--refresh", action="store_true",
                   help="忽略断点进度，全部重跑（默认断点续跑）")
    p.add_argument("--source", choices=("baostock", "tushare"), default="baostock",
                   help="回填数据源：baostock=按股×全字段（估值列全）；"
                        "tushare=按日×价量（估值列留空，BaoStock 行后补覆盖）。"
                        "⚠️ BaoStock 黑名单期间用 tushare 先把价量 10 年拿到（≈53 分钟）")
    p.add_argument("--workers", type=int, default=3,
                   help="并行 worker 进程数（0/1=单进程顺序；⚠️ 默认 3——实测 7 并发会触发"
                        "BaoStock「黑名单」服务器保护，4 并发安全）")
    p.add_argument("--stall-timeout", type=int, default=600,
                   help="看门狗：多少秒内**一只都没完成**就判定取数 hang 并熔断退出"
                        "（BaoStock 请求无超时；已完成的行会先落盘，重启续跑）。默认 600")
    return p


def _login_with_retry(attempts: int = 3):
    """BaoStock 登录失败有两种，**对策不同**：

    ① 服务器瞬时抖动 ⇒ 稍等即好；
    ② ⚠️⚠️ **短间隔频繁登录会触发 IP 临时限制**（实测：几十分钟内登录 8 次 ⇒
       连续 5 次"网络接收错误"，**等 2 分钟后自行恢复**）。

    ⇒ 重试必须"**次数少、间隔长**"—— 密集重试（3/8/15s）只会把限制越撞越死。
       真跑不通就退出：任务支持断点续跑，隔一段时间重跑同一命令即可。
    """
    import time as _time
    import baostock as bs          # 垫片已在 main 里打过；此处自取，别依赖调用方局部名
    delays = (60, 180, 300)
    last = None
    for i in range(1, attempts + 1):
        lg = bs.login()
        if lg.error_code == "0":
            if i > 1:
                print(f"[登录] 第 {i} 次重试成功")
            return lg
        last = f"{lg.error_code}: {lg.error_msg}"
        if i < attempts:
            print(f"[登录] 第 {i}/{attempts} 次失败（{lg.error_msg}）⇒ "
                  f"等 {delays[i - 1]}s 再试（⚠️ 别密集重试，会撞 IP 临时限制）", flush=True)
            _time.sleep(delays[i - 1])
    raise SystemExit(f"❌ BaoStock 连续 {attempts} 次登录失败（{last}）。"
                     f"多为短期频繁登录触发的临时限制 ⇒ **等 5~10 分钟后重跑同一命令**"
                     f"（支持断点续跑，已完成的代码自动跳过）")


def _run_tushare_backfill(args, end: str) -> int:
    """tushare 按日回填 —— ⚠️ **完全不碰 BaoStock**（不 login、不取清单）。

    第一版把这条分支放在了 BaoStock `login` 之后 ⇒ tushare 模式也先撞
    黑名单 ⇒ **按数据源分派必须在任何 BaoStock 调用之前**。
    """
    from data import tushare_source as _ts
    print(f"[计划] 区间 {args.start} ~ {end} | 数据源=tushare（按交易日 × 价量，"
          f"估值列留空待 BaoStock 覆盖） | 批大小={args.batch_size} | "
          f"{'**执行**' if args.apply else '仅预览（加 --apply 才真正执行）'}")
    days = _ts.fetch_trade_cal(args.start, end)
    print(f"[tushare] 交易日 {len(days)} 天，按日回填（1 次/日全市场，50/分钟 ⇒ "
          f"≈{len(days) * 1.7 / 60:.0f} 分钟）")
    if not args.apply:
        print("[预览] 未取数、未写盘。确认无误后加 --apply 执行。")
        return 0
    stats = PS.run_backfill_tushare(
        trade_days=days, fetch_one_day=_ts.fetch_daily,
        batch_size=args.batch_size, resume=not args.refresh)
    print("[结果]", {k: v for k, v in stats.items() if k != "elapsed_s"},
          f"| {stats.get('elapsed_s')}s")
    print("[面板]", PS.panel_status())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    import datetime as _dt
    end = args.end or _dt.date.today().strftime("%Y-%m-%d")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s")

    # ⚠️ 按数据源分派必须**先于任何 BaoStock 调用**（login/清单都别碰）——
    #    tushare 模式与 BaoStock 黑名单完全无关。
    if args.source == "tushare":
        return _run_tushare_backfill(args, end)

    # ── 垫片必须在 import baostock / 任何查询之前 ──
    PS.apply_baostock_pandas_shim()
    import baostock as bs

    print(f"[计划] 区间 {args.start} ~ {end} | 在市={not args.include_delisted} | "
          f"limit={args.limit or '不限'} | 批大小={args.batch_size} | "
          f"{'**执行**' if args.apply else '仅预览（加 --apply 才真正执行）'}")

    lg = _login_with_retry()
    if lg.error_code != "0":
        print(f"❌ BaoStock 登录失败: {lg.error_msg}")
        return 2
    try:
        codes, info = PS.load_stock_universe(listed_only=not args.include_delisted)
        print(f"[清单] 原始 {info['raw_rows']} 行 → 股票 {info['stock_rows']} 只 → "
              f"可用代码 {info['codes']} 只（剔除北交所/非沪深）")
        if args.include_delisted:
            print("[提示] 已包含退市股 —— 防**幸存者偏差**，因子研究口径正确")
        else:
            print("[提示] 只含在市股；做**因子 IC** 研究时建议加 --include-delisted")
        if args.limit:
            codes = codes[:args.limit]
        # ⚠️ 估算按**实测**：1 年 1.57s/只；10 年按年分段 17~32s/只（顺序）。
        per_code_s = 1.57 if str(args.start)[:4] == str(end)[:4] else 25.0
        eff = max(1, args.workers) if args.apply else 1
        print(f"[估算] {len(codes)} 只 × ~{per_code_s:.0f}s/只 ÷ {eff} worker ≈ "
              f"{len(codes) * per_code_s / eff / 3600:.1f} 小时"
              f"{'（并行执行）' if eff > 1 else '（顺序执行）'}")

        if not args.apply:
            print("[预览] 未取数、未写盘。确认无误后加 --apply 执行。")
            return 0

        if args.workers and args.workers > 1:
            # 并行：每个 worker 进程自己 login（BaoStock 连接是模块级全局 socket
            # ⇒ 线程共享会串包，必须进程隔离）。父进程是唯一写者（分片+进度）。
            print(f"[并行] {args.workers} 个 worker 进程，各自 login；"
                  f"父进程单写者落盘")
            stats = PS.run_backfill_parallel(
                codes=codes, start=args.start, end=end,
                worker_fetch=PS.mp_fetch_one, workers=args.workers,
                batch_size=args.batch_size, resume=not args.refresh,
                stall_timeout=args.stall_timeout)
            if stats.get("stalled"):
                print("⚠️ 取数 hang 熔断（BaoStock 请求无超时）：已完成的行已落盘，"
                      "进度已保留 ⇒ 直接重跑同一命令续传即可。")
                return 4
            if stats.get("blacklisted"):
                print("❌ 触发 BaoStock 黑名单保护（并发会话超限）。"
                      "进度已保留 ⇒ 等 10~30 分钟后用 --workers 3（或更少）重跑同一命令续传。")
                return 3
        else:
            stats = PS.run_backfill(codes=codes, start=args.start, end=end,
                                    fetch_one=PS.fetch_one_baostock,
                                    batch_size=args.batch_size,
                                    resume=not args.refresh)
        print("[结果]", {k: v for k, v in stats.items() if k != "elapsed_per_code"},
              f"| {stats['elapsed_per_code']}s/只")

        ratio = PS.record_backfill_coverage(stats, len(codes),
                                            start=args.start, end=end)
        print(f"[覆盖率] 已落库 data_coverage：got={stats['fetched']} / "
              f"total={len(codes)} ratio={ratio}（北交所缺口如实记录）")
        print("[面板]", PS.panel_status())
        return 0
    finally:
        bs.logout()


if __name__ == "__main__":
    raise SystemExit(main())
