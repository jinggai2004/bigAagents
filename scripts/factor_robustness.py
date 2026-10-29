#!/usr/bin/env python
"""因子方向样本外稳健性检验（多区间 × 多持有期）

为什么单独一个脚本
----------------
单区间算出来的 IC 只能回答"在这段样本上方向如何"。2024 是 A 股反转特征明显的
一段，动量反向完全可能是"这一段的特性"而不是"规律"。要下结论必须看
**换区间 / 换持有期之后符号还在不在**。

用法
----
    python scripts/factor_robustness.py                # 默认 30 只 / 3 区间 × 3 持有期
    python scripts/factor_robustness.py --codes 60     # 扩大股票池
    python scripts/factor_robustness.py --json out.json

性能：日线**只抓一次**（覆盖全部配置），之后每格只在内存数据上重算面板。
"""
import argparse
import io
import json
import logging
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")

socket.setdefaulttimeout(25)

from core import factor_analysis as fa            # noqa: E402
from scripts.backtest_rule_based import load_universe  # noqa: E402


def _mark(ic) -> str:
    if ic is None:
        return "  —  "
    return " ↑  " if ic > 0 else " ↓  "


def main() -> int:
    ap = argparse.ArgumentParser(description="因子方向样本外稳健性检验")
    ap.add_argument("--codes", type=int, default=30, help="股票池只数（默认 30）")
    ap.add_argument("--step", type=int, default=5, help="采样步长（交易日）")
    ap.add_argument("--horizons", default="10,20,60", help="未来收益窗口，逗号分隔")
    ap.add_argument("--json", default="", help="把结果额外写到该 JSON 文件")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    try:
        codes, _meta, _src = load_universe()
    except Exception as e:                          # noqa: BLE001
        print(f"股票池不可用：{type(e).__name__}: {e}")
        return 2
    codes = codes[:args.codes]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    print(f"股票池 {len(codes)} 只 | 区间 {[p[0][:4] for p in fa.DEFAULT_ROBUSTNESS_PERIODS]}"
          f" | 持有期 {horizons} | 步长 {args.step}", flush=True)

    t0 = time.time()

    def prog(ev):
        print(f"  [{float(ev.get('percent') or 0):5.1f}%] {ev.get('message', '')}",
              flush=True)

    res = fa.robustness(codes, horizons=horizons, step=args.step, progress_cb=prog)
    print(f"\n完成，耗时 {time.time() - t0:.0f}s | 有效股票 {res.names}", flush=True)
    if res.note:
        print("注意：", res.note, flush=True)
    if not res.summary:
        return 1

    display = {f["key"]: f["因子"] for f in res.summary}
    # ── 逐因子 IC 矩阵 ──
    print("\n=== 各配置的 IC 均值（↑=正 ↓=负，括号内为期数）===")
    for row in res.summary:
        key = row["key"]
        print(f"\n【{row['因子']}】方向假设：{row['方向假设']}")
        header = f"  {'区间':<12}" + "".join(f"{'持有' + str(h) + '日':>14}" for h in res.horizons)
        print(header)
        print("  " + "-" * (12 + 14 * len(res.horizons)))
        for p in res.periods:
            label = p[0][:4] if p[0][:4] == p[1][:4] else f"{p[0][:4]}~{p[1][:4]}"
            line = f"  {label:<12}"
            for h in res.horizons:
                hit = [c for c in res.cells
                       if c["key"] == key and c["start"] == p[0]
                       and c["end"] == p[1] and c["持有期"] == h]
                if not hit or hit[0]["IC均值"] is None:
                    line += f"{'—':>14}"
                else:
                    c = hit[0]
                    line += f"{_mark(c['IC均值'])}{c['IC均值']:+.3f}({c['期数']})".rjust(14)
            print(line)

    # ── 汇总（本脚本的主结论）──
    print("\n=== 符号一致性汇总（判据：符号与权重假设一致的比例）===")
    print(f"  {'因子':<12}{'方向假设':<12}{'一致/配置':>10}{'一致率':>9}{'平均IC':>10}  判定")
    print("  " + "-" * 84)
    for row in res.summary:
        pct = row["符号一致率"]
        pct_txt = f"{pct:.0%}" if pct is not None else "—"
        avg = f"{row['平均IC']:+.4f}" if row.get("平均IC") is not None else "—"
        print(f"  {row['因子']:<12}{row['方向假设']:<12}"
              f"{str(row['一致']) + '/' + str(row['有效配置']):>10}{pct_txt:>9}{avg:>10}"
              f"  {row['判定']}")

    print("\n判据门槛：一致率 ≥ {:.0%} 才算稳健（0.5 等于抛硬币，没有信息量）。".format(
        fa.ROBUSTNESS_AGREE_THRESHOLD))
    print("⚠️ 稳健 ≠ 正确：这里检验的是「符号在样本外还在不在」，不是「改动就能赚钱」。")

    if args.json:
        Path(args.json).write_text(
            json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
