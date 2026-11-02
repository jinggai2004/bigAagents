# -*- coding: utf-8 -*-
"""PIT 财务因子的月度截面 IC 验证（第一批 5 个字段）。

目的
────
回答一个问题：**基本面因子在 A 股还有没有可用信息，值不值得把回填扩到全市场？**
在此之前不做任何"加进选股"的动作——量价因子那轮的教训是"37 个显著里只剩 1 个赌注"，
这里同样先过 DSR/PBO，再谈能不能用。

PIT 对齐（唯一正确做法）
────────────────────────
每月末 `d` 时，只能用 `pub_date <= d` 的财报，取其中 `stat_date` 最大的那一期。
用 `stat_date` 对齐等于提前用上未来才公告的报表——这是 A 股回测最常见的前视偏差。

因子（第一批，全部只用 profit_data 一张表）
───────────────────────────────────────────
| 因子 | 定义 | 方向先验 |
|---|---|---|
| `roe_lvl` | 累计 ROE（同月对比 ⇒ 季节一致） | + |
| `gp_margin` | 毛利率 | + |
| `np_margin` | 净利率 | + |
| `roe_yoy` | ROE 同比变化（t − t−4 季，消季节性） | + |
| `profit_yoy` | 净利润同比增速（t / t−4 − 1） | + |

⚠️ 银行无毛利率（NaN 是**正确语义**，不是缺失）；北交所无数据源（absent，不补 0）。

用法
────
    py scripts/financial_factor_ic.py                  # 用已回填的全部股票
    py scripts/financial_factor_ic.py --min-stocks 300 # 样本太少就拒跑
"""
from __future__ import annotations

import argparse
import glob
import logging
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FIN_DIR = _ROOT / "data" / "panel" / "financials"
SCAN_CACHE = _ROOT / "temp" / "factor_scan_scan.parquet"
OUT = _ROOT / "output"

# PIT 加载/贴合逻辑唯一来源在 core/financial_pit.py（A/B 实验共用）
from core.financial_pit import (FIN_DIR, add_derived,      # noqa: E402,F401
                                load_financials, pit_snapshot)

_logger = logging.getLogger("fin_ic")

# 统计口径唯一来源：core/factor_stats.py（因子实验室共用同一个函数）
from core.factor_stats import (MIN_MONTHS, MIN_STOCKS_PER_MONTH,  # noqa: E402,F401
                               ic_table, ls_series)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-stocks", type=int, default=MIN_STOCKS_PER_MONTH)
    a = ap.parse_args(argv)

    if not SCAN_CACHE.exists():
        raise SystemExit(f"缺少量价扫描缓存 {SCAN_CACHE}（提供 fwd20 与月度截面）")
    scan = pd.read_parquet(SCAN_CACHE)
    if "month_ts" not in scan.columns:
        scan = scan.rename(columns={"date": "month_ts"})
    scan["month_ts"] = pd.to_datetime(scan["month_ts"])
    scan["code"] = scan["code"].astype(str)

    fin = add_derived(load_financials())
    months = pd.DatetimeIndex(sorted(scan["month_ts"].unique()))
    snap = pit_snapshot(fin, months)
    print(f"财报 {len(fin):,} 期 / {fin['code'].nunique():,} 只 ⇒ PIT 快照 {len(snap):,} 行")

    panel = snap.merge(scan[["month_ts", "code", "fwd20"]],
                       on=["month_ts", "code"], how="inner")
    cov = panel.groupby("month_ts")["code"].nunique()
    print(f"与量价面板交集 {len(panel):,} 行；月度覆盖 中位 {int(cov.median())} 只"
          f"（{int(cov.min())}~{int(cov.max())}）")
    if cov.median() < a.min_stocks:
        print(f"⚠️ 覆盖不足 {a.min_stocks} 只/月，结论不可信 —— 先扩回填样本")

    specs = [("roe_lvl", "roe"), ("gp_margin", "gp_margin"),
             ("np_margin", "np_margin"), ("roe_yoy", "roe_yoy"),
             ("profit_yoy", "profit_yoy"),
             # 东财源独有：接口直给的同比增速（覆盖 ~96%，比自算 profit_yoy
             # 允许负基数）与盈利质量代理
             ("ystz", "ystz"), ("sjltz", "sjltz"), ("ocf_to_np", "ocf_to_np")]
    # 现金流质量（应计的镜像）——只有回填了 cashflow 表才有列
    for name, col in (("cfo_to_np", "cfo_to_np"), ("cfo_np_yoy", "cfo_np_yoy")):
        if col in panel.columns:
            specs.append((name, col))
    results = [ic_table(panel, n, c, a.min_stocks) for n, c in specs]

    # ── 多重检验校正（DSR + PBO）：8 个因子互为试验，方向已按样本内选定 ──
    from core.factor_multiple_testing import adjust_by_multiple_testing
    series_map = {}
    for n, c in specs:
        s = ls_series(panel, c)
        if s is not None:
            series_map[n] = s
    mt = adjust_by_multiple_testing(results, series_map)
    print(f"多重检验：{mt.get('n_trials', 0)} 个试验，"
          f" hurdle(月频SR*)={mt.get('sr_star_monthly')}，"
          f"PBO={((mt.get('pbo') or {}).get('pbo'))}")

    print(f"\n{'因子':>12} {'IC':>7} {'ICIR':>7} {'t':>7} {'Q5-Q1':>9}  结论")
    for r in results:
        if "t_stat" in r:
            print(f"{r['name']:>12} {r['ic_mean']:+7.3f} {r['ic_ir']:+7.2f} "
                  f"{r['t_stat']:+7.1f} {str(r['q5_q1_spread']):>9}  {r['verdict']}")
        else:
            print(f"{r['name']:>12} {'-':>7} {'-':>7} {'-':>7} {'-':>9}  {r['verdict']}")

    stamp = datetime.now().strftime("%Y%m%d")
    OUT.mkdir(exist_ok=True)
    p = OUT / f"财务因子IC_{stamp}.json"
    p.write_text(json.dumps({"stocks": int(fin["code"].nunique()),
                             "months": int(panel["month_ts"].nunique()),
                             "coverage_median": int(cov.median()),
                             "multiple_testing": mt,
                             "results": results}, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    print(f"\nJSON: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
