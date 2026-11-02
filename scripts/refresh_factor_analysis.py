# -*- coding: utf-8 -*-
"""刷新 `output/ui_runs/factor_analysis.json`（因子有效性产物）—— 命令行入口。

为什么需要这个脚本
──────────────────
`/api/research/factors` 只读 `output/ui_runs/factor_analysis.json`，而这个文件
此前**只有 Streamlit 页面（`ui/pages/11_🧪_因子有效性.py`）跑完才会写**。
Web 端没有触发入口 ⇒ 一旦页面不打开，研究面板的因子页就永远停在旧结果
（实测 2026-10-30 时仍是 **10-17** 的产物，且只有 20 只样本 / 4 个因子）。

本脚本复用页面**同一个** `core.factor_analysis.run`，只是把结果落到同一个
`core.ui_runs.save_result("factor_analysis", payload)`，因此 Web 读到的结构与
页面写出的**逐字段一致**（`payload.params` / `summary` / `layered` / `correlation`）。

口径
────
- 取数：`fa.load_bars` → **本地面板优先**（`data/panel/` 单只 5~20ms），本地缺的
  才走新浪。所以正常情况是离线的、秒级到分钟级。
- 股票池：默认 `data/backtest_pool.json` 的 `pool`（120 只，带行业分散）。
- 因子：默认 `FACTORS` 全部（不是页面里手选的 4 个）——池子够大时全跑才有意义。
- `horizon=20 / step=3 / groups=5`：与页面默认一致，便于和旧产物对比。

用法
────
    py scripts/refresh_factor_analysis.py
    py scripts/refresh_factor_analysis.py --start 2023-01-01 --step 5 --limit 60
    py scripts/refresh_factor_analysis.py --factors momentum20 volatility --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import factor_analysis as fa          # noqa: E402
import core.ui_runs as ui_runs                  # noqa: E402

_log = logging.getLogger("refresh_factor_analysis")
TASK = "factor_analysis"


def load_pool(path: str, limit: int | None = None) -> tuple[list[str], dict]:
    """读回测股票池。文件缺失/坏掉时**不静默返回空**——直接报错，让调用方知道。"""
    try:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit(f"股票池不可读：{path} → {type(e).__name__}: {e}") from e
    pool = [str(c).strip() for c in (meta.get("pool") or []) if str(c).strip()]
    if not pool:
        raise SystemExit(f"股票池为空：{path}（先跑 scripts/build_backtest_pool.py 或等价入口）")
    if limit:
        pool = pool[:limit]
    return pool, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="刷新因子有效性产物（写 output/ui_runs/factor_analysis.json）")
    ap.add_argument("--pool", default=str(_ROOT / "data" / "backtest_pool.json"))
    ap.add_argument("--start", default="2024-06-01")
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--horizon", type=int, default=20, help="未来收益窗口（交易日）")
    ap.add_argument("--step", type=int, default=3, help="每多少个交易日采样一次")
    ap.add_argument("--groups", type=int, default=5, help="分层组数")
    ap.add_argument("--limit", type=int, default=0, help="只取股票池前 N 只（0=全部）")
    ap.add_argument("--factors", nargs="*", default=None,
                    help="只跑指定因子键（默认全部）")
    ap.add_argument("--dry-run", action="store_true", help="只打印参数与池子规模，不跑")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    codes, meta = load_pool(args.pool, args.limit or None)
    factors = args.factors or list(fa.FACTORS.keys())
    unknown = [f for f in factors if f not in fa.FACTORS]
    if unknown:
        raise SystemExit(f"未知因子键：{unknown}\n可用：{list(fa.FACTORS.keys())}")

    print(f"股票池：{len(codes)} 只（文件共 {len(meta.get('pool') or [])} 只，"
          f"built_at={meta.get('built_at')}）")
    print(f"因子：{len(factors)} 个 → {factors}")
    print(f"区间：{args.start} ~ {args.end}  horizon={args.horizon} "
          f"step={args.step} groups={args.groups}")
    if args.dry_run:
        print("--dry-run：不执行")
        return 0

    def _cb(ev: dict) -> None:
        pct = ev.get("percent")
        msg = ev.get("message") or ""
        if pct is not None:
            print(f"  [{float(pct):5.1f}%] {msg}", flush=True)

    res = fa.run(codes, factors=factors, start=args.start, end=args.end,
                 horizon=args.horizon, step=args.step, groups=args.groups,
                 progress_cb=_cb)
    payload = res.to_dict()
    payload["params"] = {"start": args.start, "end": args.end,
                         "horizon": args.horizon, "step": args.step,
                         "groups": args.groups, "codes": len(codes),
                         "factors": list(factors)}

    if not ui_runs.save_result(TASK, payload):
        print("！产物写入失败（output/ui_runs 不可写？）", file=sys.stderr)
        return 1

    print(f"\n已写入：{os.path.join(ui_runs.UI_RUNS_DIR, ui_runs.result_filename(TASK))}")
    print(f"期数 {payload.get('periods')} | 标的 {payload.get('names')} | "
          f"面板行 {payload.get('panel_rows')}")
    # 摘要前几名（`summary` 的键是**中文**：因子/IC均值/ICIR/t值/结论——
    # 别按英文猜，猜错了打印全是 None，看起来像"没跑出结果"）
    for row in (payload.get("summary") or [])[:8]:
        print(f"  {str(row.get('因子') or row.get('key') or '?'):<12} "
              f"IC均值={row.get('IC均值')} ICIR={row.get('ICIR')} "
              f"t={row.get('t值')} 期数={row.get('期数')} → {row.get('结论')}")
    if payload.get("unavailable"):
        print(f"  不可用因子：{list(payload['unavailable'].keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
