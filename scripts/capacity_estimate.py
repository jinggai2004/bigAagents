# -*- coding: utf-8 -*-
"""top=5 组合的**容量测算**：把"滑点 50bp"从假设变成按票估算。

## 为什么需要这个脚本

2026-10-31 的执行折扣实验发现：top=5 组合在 5bp 滑点下押注段超额 +17.7%，
50bp 下只剩 +3.4%，开发段还转负。但 **50bp 是从哪来的？** 那是拍的。

真正的问题是：**这套策略买的票能不能吃下我们的钱。** 因子池是"低流动性+高冲击成本"，
它天生选日成交额小的票。所以正确做法不是假设一个滑点，而是：

    给定每只票的日均成交额 V 和日波动 σ，反解"买多少钱会把价格推高 X bp"。

## 模型（平方根冲击定律）

    impact_bps = Y · σ_daily · sqrt(Q / V) · 10000

- `Q` = 我们买/卖的金额（元），`V` = 该票日均成交额（元），`σ_daily` = 日收益标准差；
- `Y` 是市场相关的经验系数（文献与业界实践多落在 0.5~1.5）。**本项目没有自己的
  实证校准**，所以 `Y` 必须做敏感性，不能只报一个数；
- 反解容量：`Q_max(target) = V · (target/10000 / (Y·σ))²`。

⚠️ 模型的边界（必须一起读）：
  1. 它估的是**冲击成本**，不含佣金/印花税/价差（那部分在 `core/matching` 里已单独计）；
  2. 假设订单在**一个交易日内**完成；拆单到多日能放大容量，但也放大暴露时间；
  3. `V` 用 20 日均额，**大行情日成交额会放大**（有利于实际执行）、**缩量日会缩小**；
  4. `σ` 是历史日波动，事件驱动的跳空不在此列。

用法::

    py scripts/capacity_estimate.py                # 默认 top=5
    py scripts/capacity_estimate.py --top 20
    py scripts/capacity_estimate.py --nav 3000000  # 换一个资金规模
"""
from __future__ import annotations

import argparse
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

import logging

from core.factor_composite import PROMOTED_FACTORS, composite_score  # noqa: E402
from core.impact_model import Y_DEFAULT, Y_RANGE, capacity_for, impact_bp  # noqa: E402

_logger = logging.getLogger("capacity")

CACHE = _ROOT / "temp" / "factor_scan_scan.parquet"
OUT_DIR = _ROOT / "output"

SEGMENTS = [
    ("开发段", "2019-03-01", "2023-12-31"),
    ("验证段", "2024-01-01", "2024-12-31"),
    ("押注段", "2025-01-01", "2026-10-31"),
]
TARGET_BPS = (5.0, 20.0, 50.0)      # 反解"要做到这个冲击成本，最多能买多少"
# Y_DEFAULT / Y_RANGE 从 core.impact_model 取 —— 公式与系数**只有一份定义**
DEFAULT_NAV = 1_000_000.0            # 与 paper_accounts.initial_cash 一致


def _load(top: int) -> pd.DataFrame:
    """读扫描缓存 → 合成分数 → 取每月 topN → 套宇宙守卫。"""
    if not CACHE.exists():
        raise SystemExit(f"缺少扫描缓存 {CACHE}，请先跑 scripts/factor_paper_scan.py")
    df = pd.read_parquet(CACHE)
    if "month_ts" not in df.columns:
        df = df.rename(columns={"date": "month_ts"})
    df["month_ts"] = pd.to_datetime(df["month_ts"])
    df["code"] = df["code"].astype(str)

    need = [c for c in PROMOTED_FACTORS if c in df.columns]
    missing = [c for c in PROMOTED_FACTORS if c not in df.columns]
    if missing:
        raise SystemExit(f"缓存缺因子列：{missing}")

    # ⚠️ 与 portfolio_ab_test 同口径：**全表排名**后再把不齐的行置 NaN。
    #    先 dropna 再排名会缩小分母改变百分位（2026-10-31 踩过）。
    sub = df.dropna(subset=need)
    df["composite"] = np.nan
    df.loc[sub.index, "composite"] = composite_score(sub, need, PROMOTED_FACTORS)
    df = df.dropna(subset=["composite", "fwd20"])

    from core.universe_guard import eligible
    before = len(df)
    df = df[eligible(df, month_col="month_ts")].copy()
    _logger.info("[守卫] %d → %d 行（剔除 %.1f%%）",
                 before, len(df), 100.0 * (1 - len(df) / max(before, 1)))

    # 容量只关心"被选中的那几只"，排序后取 top 即可
    picks = (df.sort_values(["month_ts", "composite"], ascending=[True, False])
               .groupby("month_ts", sort=True).head(top).copy())
    picks["rank"] = picks.groupby("month_ts")["composite"].rank(
        ascending=False, method="first")
    return picks


def _capacity_table(picks: pd.DataFrame, y: float = Y_DEFAULT) -> pd.DataFrame:
    """给每只被选中的票算：日均成交额、日波动、各冲击档位下的可买金额。"""
    out = picks.copy()
    # lndvol20 = log(成交额 20 日均值)，单位元
    out["amount_daily"] = np.exp(pd.to_numeric(out["lndvol20"], errors="coerce"))
    # vol20 是**日收益标准差**（`r.rolling(20).std()`，未年化）——见 factor_paper_scan
    out["sigma_daily"] = pd.to_numeric(out["vol20"], errors="coerce")

    ok = (out["amount_daily"] > 0) & (out["sigma_daily"] > 0)
    out = out[ok].copy()
    for bps in TARGET_BPS:
        out[f"qmax_{int(bps)}bp"] = capacity_for(
            out["amount_daily"], out["sigma_daily"], bps, y=y)
    return out


def _impact_at(cap: pd.DataFrame, nav: float, y: float = Y_DEFAULT) -> pd.Series:
    """给定资金规模，按"等权摊到被选中的票"算每只的实际冲击（bp）。

    每月独立分摊（某月只选出 4 只就摊 4 份），而不是固定 nav/top——
    否则"某只没成交"会让分摊金额算错。
    """
    n = cap.groupby("month_ts")["code"].transform("count")
    return impact_bp(cap["amount_daily"], cap["sigma_daily"], nav / n, y=y)


def _seg(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df["month_ts"] >= pd.Timestamp(start)) & (df["month_ts"] <= pd.Timestamp(end))]


def _fmt_yuan(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if x >= 1e8:
        return f"{x / 1e8:.2f} 亿"
    if x >= 1e4:
        return f"{x / 1e4:.0f} 万"
    return f"{x:.0f} 元"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="topN 组合容量测算（平方根冲击模型）")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--nav", type=float, default=DEFAULT_NAV, help="资金规模（元）")
    ap.add_argument("--y", type=float, default=Y_DEFAULT, help="平方根系数（默认 1.0）")
    ap.add_argument("--json", default="", help="额外把结果写到这里")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    picks = _load(a.top)
    print(f"样本：{picks['month_ts'].nunique()} 个月 × top{a.top} = {len(picks):,} 个 (月,票)"
          f"｜覆盖 {picks['month_ts'].min():%Y-%m} ~ {picks['month_ts'].max():%Y-%m}")

    cap = _capacity_table(picks, a.y)
    print(f"\n被选中票的流动性与波动（全部样本，{len(cap):,} 个观测）：")
    print(f"  日均成交额 V     中位 {_fmt_yuan(cap['amount_daily'].median())}"
          f"｜10% 分位 {_fmt_yuan(cap['amount_daily'].quantile(.1))}"
          f"｜90% 分位 {_fmt_yuan(cap['amount_daily'].quantile(.9))}")
    print(f"  日波动 σ         中位 {cap['sigma_daily'].median():.2%}"
          f"｜10% 分位 {cap['sigma_daily'].quantile(.1):.2%}"
          f"｜90% 分位 {cap['sigma_daily'].quantile(.9):.2%}")

    # ── ① 反解：做到 X bp 冲击，单票最多能买多少 ──
    print(f"\n① 反解容量（Y={a.y}）：单票在给定冲击成本下最多能买多少")
    print(f"{'段':<8}{'5bp':>14}{'20bp':>14}{'50bp':>14}")
    res: dict = {"top": a.top, "nav": a.nav, "y": a.y,
                 "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "segments": {}}
    for name, s0, e0 in SEGMENTS:
        seg = _seg(cap, s0, e0)
        if seg.empty:
            continue
        row = {}
        line = f"{name:<8}"
        for bps in TARGET_BPS:
            med = float(seg[f"qmax_{int(bps)}bp"].median())
            row[f"qmax_{int(bps)}bp_median"] = med
            row[f"qmax_{int(bps)}bp_p25"] = float(seg[f"qmax_{int(bps)}bp"].quantile(.25))
            line += f"{_fmt_yuan(med):>14}"
        res["segments"][name] = row
        print(line)

    # ── ② 给定 NAV，实际冲击是多少（这是替代"50bp 假设"的数）──
    print(f"\n② 给定 NAV={_fmt_yuan(a.nav)}（等权摊到 top{a.top}，每只 "
          f"{_fmt_yuan(a.nav / a.top)}），按票算的实际冲击")
    print(f"{'段':<8}{'中位冲击':>10}{'75%分位':>10}{'90%分位':>10}"
          f"{'占日均额中位':>14}{'超5%占比':>10}")
    cap["impact_bps"] = _impact_at(cap, a.nav, a.y)
    cap["participation"] = (a.nav / cap.groupby("month_ts")["code"].transform("count")
                            / cap["amount_daily"])
    for name, s0, e0 in SEGMENTS:
        seg = _seg(cap, s0, e0)
        if seg.empty:
            continue
        r = res["segments"][name]
        r["impact_bps_median"] = round(float(seg["impact_bps"].median()), 1)
        r["impact_bps_p75"] = round(float(seg["impact_bps"].quantile(.75)), 1)
        r["impact_bps_p90"] = round(float(seg["impact_bps"].quantile(.90)), 1)
        r["participation_median"] = round(float(seg["participation"].median()), 4)
        r["part_over5pct"] = round(float((seg["participation"] > 0.05).mean()), 3)
        print(f"{name:<8}{r['impact_bps_median']:>9.0f}bp{r['impact_bps_p75']:>9.0f}bp"
              f"{r['impact_bps_p90']:>9.0f}bp{r['participation_median']:>13.1%}"
              f"{r['part_over5pct']:>10.0%}")

    # ── ③ Y 的敏感性（系数是经验值，不给区间就是耍流氓）──
    print("\n③ 平方根系数 Y 的敏感性（押注段，NAV=%s）" % _fmt_yuan(a.nav))
    print(f"{'Y':<8}{'中位冲击':>10}{'qmax@20bp 中位':>18}")
    sens = {}
    for y in Y_RANGE:
        sub = _capacity_table(_seg(picks, "2025-01-01", "2026-10-31"), y)
        if sub.empty:
            continue
        imp = _impact_at(sub, a.nav, y)
        q20 = float(sub["qmax_20bp"].median())
        sens[str(y)] = {"impact_bps_median": round(float(imp.median()), 1),
                        "qmax_20bp_median": q20}
        print(f"{y:<8}{imp.median():>9.0f}bp{_fmt_yuan(q20):>18}")
    res["y_sensitivity"] = sens

    # ── ④ 与回测滑点假设的对照（核心结论）──
    print("\n④ 与回测滑点假设对照")
    for name in ("押注段",):
        r = res["segments"].get(name) or {}
        if not r:
            continue
        print(f"  回测用 5bp（core.matching 默认）｜按票估的中位冲击 "
              f"{r['impact_bps_median']:.0f}bp"
              f"（75% 分位 {r['impact_bps_p75']:.0f}bp）")
        print(f"  ⇒ 默认滑点{'低估' if r['impact_bps_median'] > 5 else '高估'}了 "
              f"{r['impact_bps_median'] / 5:.1f} 倍")
        # 反过来问：要给到中位冲击这么多，单票要买多大的量级
        print(f"  单票买入占日均成交额中位 {r['participation_median']:.1%}"
              f"（超过 5% 的样本占 {r['part_over5pct']:.0%}）")

    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    p = Path(a.json) if a.json else OUT_DIR / f"容量测算_top{a.top}_{stamp}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nJSON: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
