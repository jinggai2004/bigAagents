# -*- coding: utf-8 -*-
"""因子闸门扫描：把 IC 研究的结论变成"该用哪几个闸门、剔多少"。

背景
────
因子研究只给"哪个因子有信息"，给不了"怎么用"。本项目已用 A/B 证明：
- 混进等权合成 = 稀释（AF 臂 −9.47%）；
- 当闸门（只剔最差 N%、排序不动）= 有效（AFG 臂 +2.26%）。

本脚本把那条结论从"一个因子"扩到"全部因子 × 全部剔除比例 × 闸门叠加"，
产出可直接照着做的配置。

用法
────
    py scripts/factor_gate_sweep.py                # 全部量价因子 × 10/20/30%
    py scripts/factor_gate_sweep.py --pcts 0.1 0.2 # 指定比例
    py scripts/factor_gate_sweep.py --stack        # 额外做闸门叠加

⚠️ 这是**批量试验**：扫了 N 组就等于做了 N 次尝试，真要落地上线必须过 DSR
   （试验次数 = 扫过的组数）。本脚本给的是"值得进一步验证"的排序，不是"已确认有效"。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import factor_dsl as D                    # noqa: E402
from core import factor_gate_eval as G              # noqa: E402
from core import factor_lab as FL                   # noqa: E402

OUT = _ROOT / "output"


def _best_dir(rows: list[dict], col: str) -> int:
    """从逐因子结果里取该列的最优剔除方向（`evaluate_gate` 已双向实测过）。"""
    for r in rows:
        if r["col"] == col:
            return int(r.get("direction") or 1)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcts", nargs="*", type=float, default=[0.10, 0.20, 0.30])
    ap.add_argument("--stack", action="store_true", help="额外做闸门叠加")
    ap.add_argument("--top", type=int, default=15)
    # Top-N 命中率体检的持仓数；0=跳过。默认与实盘 top5~20 同量级
    ap.add_argument("--topn", type=int, default=20,
                    help="Top-N 命中率体检的持仓数（0=跳过）")
    a = ap.parse_args(argv)

    panel = FL.load_panel(need_fin=False)
    cols = [c for c in D.PRICE_FIELDS if c in panel.columns]
    print(f"面板 {len(panel):,} 行 / {panel['month_ts'].nunique()} 个月 / "
          f"{panel['code'].nunique():,} 只；扫描 {len(cols)} 因子 × {len(a.pcts)} 比例")

    rows = G.sweep(panel, cols, pcts=tuple(a.pcts))
    print(f"\n{'因子':<12}{'剔除':>6}{'超额pp':>9}{'t':>8}{'月胜率':>8}"
          f"{'保留':>7}  方向vsIC  判定")
    for r in rows[: a.top]:
        print(f"{r['col']:<12}{r['pct']*100:5.0f}%{r['gain_pp']:+9.2f}"
              f"{r['t_stat']:+8.2f}{r['pos_rate']*100:7.0f}%{r['avg_kept']:7d}  "
              f"{r['direction_vs_ic']:<8}  {r['verdict']}")

    mm = G.direction_mismatch(rows)
    if mm:
        print(f"\n⚠️ 有 {len(mm)} 组的**最优剔除方向与 IC 符号相反**（最容易被误用）：")
        for r in mm[:5]:
            print(f"  {r['col']:<12} 剔 {r['pct']*100:.0f}%：最优 {r['gain_pp']:+.2f}pp"
                  f"（t={r['t_stat']:+.1f}）vs 按 IC 方向 {r['opposite_gain_pp']:+.2f}pp"
                  f"（t={r['opposite_t']:+.1f}）")

    stacks = []
    if a.stack:
        # 按"单因子最优收益"排序取前几名，逐个叠加
        best_by_col = {}
        for r in rows:
            if r["col"] not in best_by_col:
                best_by_col[r["col"]] = r
        picks = list(best_by_col)[:4]
        print(f"\n{'闸门叠加':<44}{'剔除':>6}{'超额pp':>9}{'t':>8}{'月胜率':>8}{'保留率':>7}")
        for n in range(1, len(picks) + 1):
            for pct in (0.10, 0.20):
                r = G.evaluate_stack(panel, picks[:n], pct=pct)
                if not r.get("ok"):
                    continue
                stacks.append({"cols": picks[:n], "pct": pct, **r})
                print(f"{'+'.join(picks[:n]):<44}{pct*100:5.0f}%{r['gain_pp']:+9.2f}"
                      f"{r['t_stat']:+8.2f}{r['pos_rate']*100:7.0f}%"
                      f"{r['keep_ratio']*100:6.0f}%")

    # ── Top-N 命中率体检（2026-11-01 加的硬性一步）──
    # 面板级超额 ≠ 组合级收益。本项目实测过：三道闸门面板 +0.76pp/月 t=5.00，
    # 组合级 A/B 里与现状**逐位相同**——1,800 个 Top20 席位只剔到 1 个。
    # 所以任何要上线的闸门，先看它剔掉的票有多少本来就在 TopN 里。
    impacts = []
    if a.topn > 0 and stacks:
        print(f"\n{'Top{0} 命中率体检'.format(a.topn):<44}{'命中/席位':>14}"
              f"{'命中率':>9}{'被剔分位':>10}  判定")
        for st in stacks[-3:]:          # 只看最后 3 组（通常是层数最多的）
            dirs = {c: (1 if _best_dir(rows, c) > 0 else -1) for c in st["cols"]}
            r = G.topn_impact(panel, st["cols"], dirs, pct=st["pct"], topn=a.topn)
            if not r.get("ok"):
                print(f"{'+'.join(st['cols']):<44}{r.get('reason')}")
                continue
            impacts.append({"cols": st["cols"], "pct": st["pct"],
                            **{k: r[k] for k in
                               ("hit", "topn_slots", "hit_rate",
                                "dropped_composite_quantile_median",
                                "verdict", "has_increment")}})
            print(f"{'+'.join(st['cols'])[:42]:<44}"
                  f"{r['hit']:>6}/{r['topn_slots']:<7}"
                  f"{r['hit_rate']*100:8.2f}%"
                  f"{(r['dropped_composite_quantile_median'] or 0)*100:9.0f}%"
                  f"  {r['verdict']}")

    OUT.mkdir(exist_ok=True)
    p = OUT / f"因子闸门扫描_{datetime.now():%Y%m%d}.json"
    p.write_text(json.dumps({
        "panel": FL.panel_info(), "pcts": list(a.pcts),
        "n_trials": len(rows), "results": rows, "stacks": stacks,
        "topn_impact": impacts,
        "note": "批量试验：落地上线前须按 n_trials 过 DSR",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nJSON: {p}")
    print("提示：① 面板超额 ≠ 组合收益，看上面 Top-N 命中率；"
          "② 上线前还得跑 `scripts/portfolio_ab_test.py` 做组合级 A/B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
