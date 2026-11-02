# -*- coding: utf-8 -*-
"""超额衰减监控（CLI）：读最近的 A/B 实验结果，输出三层判据结论。

用法
────
    py scripts/alpha_decay_monitor.py                 # 用最新一份 组合AB实验_*.json
    py scripts/alpha_decay_monitor.py --json output/组合AB实验_20261031_0848.json
    py scripts/alpha_decay_monitor.py --arm "A_缓冲3.0x" --recent 6

退出码：`alert=2` / `watch=1` / `ok=0` / `unknown=3`
——这样挂到调度器或 bat 里就能用退出码判断，不需要人去读日志。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.alpha_decay import (LEVEL_ALERT, LEVEL_UNKNOWN, LEVEL_WATCH,  # noqa: E402
                              decay_report)

OUT = _ROOT / "output"
DEFAULT_ARM = "A_缓冲3.0x"      # A/B 实验里表现最稳、换手最低的那一臂

EXIT = {LEVEL_ALERT: 2, LEVEL_WATCH: 1, LEVEL_UNKNOWN: 3}


def _latest_json() -> Path:
    files = sorted(glob.glob(str(OUT / "组合AB实验_*.json")))
    if not files:
        raise SystemExit("找不到 组合AB实验_*.json，先跑 `py scripts/portfolio_ab_test.py`")
    return Path(files[-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--arm", default=DEFAULT_ARM)
    ap.add_argument("--recent", type=int, default=6)
    a = ap.parse_args(argv)

    path = Path(a.json) if a.json else _latest_json()
    d = json.loads(path.read_text(encoding="utf-8"))
    arm = (d.get("arms") or {}).get(a.arm)
    if arm is None:
        print(f"❌ 该文件里没有臂 {a.arm!r}，可选：{list((d.get('arms') or {}))}")
        return EXIT[LEVEL_UNKNOWN]

    # 策略用**净**收益（已扣成本，`series` 落的是毛收益，净额见 segments）；
    # 基准用同文件里的**逐月序列**（同一守卫宇宙的等权），不能用段级年化代替。
    # ⚠️ 必须用**净**序列：毛序列里含着换手成本，会把成本当成 alpha
    series = arm.get("net_series") or arm.get("series") or {}
    ret = dict(series)
    bench_raw = d.get("benchmark_series") or {}

    print(f"数据：{path.name}｜臂：{a.arm}")
    print(f"策略月度序列 {len(ret)} 个月"
          f"（{min(ret) if ret else '-'} ~ {max(ret) if ret else '-'}）")
    if not ret:
        print("❌ 该臂没有逐月序列")
        return EXIT[LEVEL_UNKNOWN]

    if not bench_raw:
        print("⚠️ 该文件没有基准逐月序列（`benchmark_series`），无法算超额衰减。"
              "请重跑 `py scripts/portfolio_ab_test.py`（新版本会落这个字段）。")
        return EXIT[LEVEL_UNKNOWN]

    import pandas as pd
    r = pd.Series({pd.Timestamp(k): v for k, v in ret.items()}).sort_index()
    b = pd.Series({pd.Timestamp(k): v for k, v in bench_raw.items()}).sort_index()
    print(f"基准：逐月序列 {len(b)} 个月（同宇宙等权），覆盖 "
          f"{b.index.min():%Y-%m} ~ {b.index.max():%Y-%m}")
    rep = decay_report(r, b, recent=a.recent)

    print(f"\n判定：{rep['level'].upper()}｜{rep['reason']}")
    if rep.get("hist_median") is not None:
        print(f"  最近 {a.recent} 月超额 {rep['recent_excess']:+.2%}"
              f"｜历史中位 {rep['hist_median']:+.2%}｜比值 {rep['ratio']}")
    print(f"  历史长度 {rep['months']} 个月｜最后一月 {rep.get('last_month', '-')}")
    return EXIT.get(rep["level"], 0)


if __name__ == "__main__":
    raise SystemExit(main())
