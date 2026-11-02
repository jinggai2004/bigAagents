# -*- coding: utf-8 -*-
"""回填财报公告日（PIT）—— `financial_disclosures` 表。

用法
────
    py scripts/backfill_disclosure_dates.py                 # 最近 8 个报告期（默认）
    py scripts/backfill_disclosure_dates.py --periods 4     # 只补最近 4 期
    py scripts/backfill_disclosure_dates.py --period 20260630
    py scripts/backfill_disclosure_dates.py --dry-run

为什么是 CLI 而不是调度任务
──────────────────────────
单个报告期实测 **81~148 秒**（东财分页 12~23 页 × ~6s），8 期约 10~15 分钟。
放进 15:xx 的调度窗口会和盘后选股/资金流抢资源，也会拖慢"当日必跑完"的链路。
公告日**不需要每天更新**：报告期内公告日基本固定，季末后一两周补一次即可。

⚠️ 重跑是幂等的：`(stock_code, period)` 主键 + upsert，且 `ann_date` 只向**更晚**
   推进（见 `store.upsert_fin_disclosures`），不会把保守下界改小、重新引入前视。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data import fin_disclosure as fd          # noqa: E402
from db import store                           # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="回填财报公告日（PIT 前视防护）")
    ap.add_argument("--periods", type=int, default=8, help="回填最近 N 个报告期")
    ap.add_argument("--period", default="", help="只回填指定报告期（YYYYMMDD）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将抓哪些期")
    ap.add_argument("--reset", action="store_true",
                    help="先清空整表再抓（来源口径变更/发现脏数据时用）")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")

    if args.reset and not args.dry_run:
        removed = store.delete_fin_disclosures()
        print(f"已清空 financial_disclosures：{removed} 行")

    targets = ([args.period] if args.period
               else fd.quarter_ends(args.periods))
    print(f"将抓取 {len(targets)} 个报告期：{targets}")
    if args.dry_run:
        print("--dry-run：不执行")
        return 0

    def _cb(ev: dict) -> None:
        print(f"  [{float(ev.get('percent') or 0):5.1f}%] {ev.get('message') or ''}",
              flush=True)

    res = fd.backfill(periods=targets, progress_cb=_cb)
    print(f"\n写入合计 {res['written']} 行")
    for p in res["periods"]:
        if p.get("error"):
            print(f"  ✗ {p['period']}: {p['error'][:90]}")
        else:
            fb = f"（回退: {p['fallback_reason'][:40]}）" if p.get("fallback_reason") else ""
            print(f"  ✓ {p['period']}: 来源={p.get('source')}{fb} "
                  f"A股 {p['a_share']} 行（沪深创科 {p['majors']} / 北交 {p['bj']}）"
                  f" 未披露 {p.get('not_disclosed', 0)}"
                  f" 超窗口丢弃 {p.get('implausible', 0)}"
                  f" 非A股 {p['dropped_non_a']} → 写 {p['written']}")
    cov = store.fin_disclosure_coverage()
    print(f"\n表内概况：{cov['period_count']} 个报告期 / {cov['rows']} 行")
    for r in cov["periods"][:10]:
        print(f"  {r['period']}: {r['n']:5d} 行，公告日 {r['a']} ~ {r['b']}")
    return 0 if not res["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
