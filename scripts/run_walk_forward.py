#!/usr/bin/env python3
"""B 模块：walk-forward 权重重估（手动/月度批任务，**不进 JOBS**）。

为什么是脚本不是调度任务：全市场矩阵装载 + 逐折 ICIR/复利要几十分钟，
且它是"改 FACTOR_WEIGHTS 之前的证据"，不是每日例行——放进调度会让
"每天固定花几十分钟"变成默认行为。

用法:
  python scripts/run_walk_forward.py                          # T+5，训 4 测 1（1500 天缓存）
  python scripts/run_walk_forward.py --horizons 5 20 --days 1500
  python scripts/run_walk_forward.py --train-years 3

产出:
  output/walk_forward_<日期>.json        逐折权重 + OOS 对比（可复查）
  stdout                                 汇总 + WEIGHT_CHANGES 填空模板

纪律：`core/sector.py::FACTOR_WEIGHTS` 的任何改动必须附本脚本的 OOS 对比
（旧权重 vs 新权重/重估的样本外表现），模板见 `docs/WEIGHT_CHANGES.md`。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core import replay_eval as rv  # noqa: E402
from utils.logger import get_console_logger  # noqa: E402

_log = get_console_logger("scripts.walk_forward")


def main() -> int:
    ap = argparse.ArgumentParser(description="walk-forward 权重重估（B 模块）")
    ap.add_argument("--horizons", type=int, nargs="+", default=[5],
                    help="评估用的前瞻窗口（默认 5）")
    ap.add_argument("--days", type=int, default=1500,
                    help="矩阵回看天数（须与已缓存的天数一致，否则要重建缓存）")
    ap.add_argument("--train-years", type=int, default=4)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    _log.info("装载矩阵 days=%s（首次构建可能较慢）", args.days)
    m = rv.load_matrices(days=args.days)
    if not m or m.get("close") is None or m["close"].empty:
        print("矩阵装载失败或为空——检查 data/panel/ 缓存")
        return 1

    for h in args.horizons:
        res = rv.walk_forward_from_market(m, horizons=(h,),
                                          train_years=args.train_years,
                                          step=args.step, top_n=args.top_n)
        s = res.get("summary") or {}
        if not s.get("available"):
            print(f"\n【T+{h}】不可用：{s.get('reason')}")
            continue
        print(f"\n═══ Walk-forward T+{h}（训 {args.train_years} 年 → 测 1 年，"
              f"{s['n_folds']} 折）═══")
        print(f"{'测试年':<8}{'重估权重':<46}{'OOS超额(重估)':>14}{'OOS超额(基线)':>14}"
              f"{'复合IC(重估)':>12}{'复合IC(基线)':>12}")
        for fold in res["folds"]:
            w = " ".join(f"{k.split('_')[0]}:{v:.2f}" for k, v in
                         (fold["weights_refit"] or {}).items())
            re_, ba = fold["oos_refit"], fold["oos_baseline"]
            fmt = lambda v: (f"{v:+.4f}" if isinstance(v, (int, float)) else "—")  # noqa: E731
            print(f"{fold['test_year']:<8}{w:<46}"
                  f"{fmt(re_.get('top_excess_mean')):>14}{fmt(ba.get('top_excess_mean')):>14}"
                  f"{fmt(re_.get('composite_ic_mean')):>12}{fmt(ba.get('composite_ic_mean')):>12}")
        print("汇总：", json.dumps({k: v for k, v in s.items() if k != "note"},
                                   ensure_ascii=False))
        print("判读：", s["note"])

    out = {"built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "days": args.days, "train_years": args.train_years, "horizons": {}}
    for h in args.horizons:
        res = rv.walk_forward_from_market(m, horizons=(h,),
                                          train_years=args.train_years,
                                          step=args.step, top_n=args.top_n)
        out["horizons"][str(h)] = res
    out_path = _PROJECT_ROOT / "output" / \
        f"walk_forward_{datetime.now().strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"\n已保存: {out_path}")
    print("改 FACTOR_WEIGHTS 前，把上面的逐折对比填进 docs/WEIGHT_CHANGES.md 模板。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
