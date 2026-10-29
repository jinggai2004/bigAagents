# -*- coding: utf-8 -*-
"""买入信号判据的事件研究（2026-10-23）—— 用**本地面板**检验判据有没有预测力。

为什么需要这个脚本：
    `core/signal_queue.py` 的买入信号判据是**拍脑袋定的阈值**
    （涨幅 ≥3% 且 相对放量 ≥1.5×20 日均额）。本项目有条硬规矩：
    **任何结论必须过"多区间 × 多持有期"**，只能说"本样本无依据"，
    禁止"因子无效/已证伪"。⇒ 阈值不能凭感觉上线，要拿数据说话。

复刻的线上口径（必须与 `core/signal_queue.py` 一致）：
    信号(T) = pct_chg[T] >= 3.0  且  amount[T] >= 1.5 × mean(amount[T-20:T-1])
    ⚠️ 线上用 `df["amount"].tail(20).mean()`（**不含当天**）⇒ 等价于 shift(1) 的滚动均值。
    ⚠️ 线上是**盘中**出信号 ⇒ 最早 T+1 开盘成交 ⇒ 回放用 `entry="next_open"`。

口径纪律：
    · 成交口径 = T+1 开盘买入、T+1+h 开盘卖出（项目唯一口径，见 `_fwd_one` 注释）；
    · **step = horizon**（避免重叠样本把显著性撑虚）；
    · 截面不足的交易日**剔除**（`DEFAULT_MIN_CROSS = 3000`），不拿残截面算等权基准；
    · 基准 = 同一交易日**全市场等权**（同期全体票的同一口径收益）；
    · 超额 = 信号组均值 − 基准均值，**按期等权**（不是按票加权）。

运行：
    python scripts/eval_signal_criterion.py [--days 1500]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.replay_eval import (  # noqa: E402
    DEFAULT_MIN_CROSS, forward_returns, load_matrices,
)

HORIZONS = (1, 5, 20)
MIN_CHANGE_PCT = 3.0
VOLUME_RATIO = 1.5


# ── 工具 ────────────────────────────────────────────────────────────────
def _daily_group_mean(r: pd.DataFrame, mask: pd.DataFrame) -> tuple:
    """逐日：信号组均值 与 当日全市场等权基准。

    返回 (sig_mean, base_mean, sig_count)，均为按日期索引的 Series。
    """
    rv = r.to_numpy(dtype="float64")
    mv = mask.to_numpy(dtype=bool)
    valid = ~np.isnan(rv)
    hit = mv & valid
    den = hit.sum(axis=1)
    sig = np.where(den > 0, np.where(hit, rv, 0.0).sum(axis=1) / np.maximum(den, 1), np.nan)
    base = np.where(valid, rv, 0.0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    return (pd.Series(sig, index=r.index),
            pd.Series(base, index=r.index),
            pd.Series(den, index=r.index))


def _excess(r: pd.DataFrame, mask: pd.DataFrame, step: int,
            min_cross: int = DEFAULT_MIN_CROSS) -> pd.DataFrame:
    """逐期超额（信号组 − 当日等权基准），并按 step 抽样去重叠。

    返回 DataFrame[date, sig, base, excess, n_sig, n_base]。
    """
    sig, base, n_sig = _daily_group_mean(r, mask)
    # 基准也要"当天截面够大"才有代表性 ⇒ 用全市场有效票数
    rv = r.to_numpy(dtype="float64")
    n_base = pd.Series((~np.isnan(rv)).sum(axis=1), index=r.index)
    df = pd.DataFrame({"sig": sig, "base": base, "n_sig": n_sig, "n_base": n_base})
    df["excess"] = df["sig"] - df["base"]
    df = df[(df["n_base"] >= min_cross) & df["sig"].notna()]
    # step 抽样：每 step 个交易日取一期（避免重叠样本）
    return df.iloc[::step]


def _fmt(x, pct=True) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:+.2f}%" if pct else f"{x:+.3f}"


def _report(title: str, rows: list) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    hdr = f"{'区间/持有期':<18}{'期数':>6}{'信号组':>11}{'基准':>11}{'超额':>11}{'胜率':>8}"
    print(hdr)
    print("-" * 78)
    for label, df in rows:
        if df is None or len(df) == 0:
            print(f"{label:<18}{'0':>6}{'—':>11}{'—':>11}{'—':>11}{'—':>8}")
            continue
        win = float((df["excess"] > 0).mean())
        print(f"{label:<18}{len(df):>6}{_fmt(df['sig'].mean()):>11}"
              f"{_fmt(df['base'].mean()):>11}{_fmt(df['excess'].mean()):>11}"
              f"{win * 100:>7.0f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    args = ap.parse_args()

    print(f"载入回放矩阵（days={args.days}，缓存命中则秒级）…")
    m = load_matrices(days=args.days)
    if not m:
        print("面板不可用 ⇒ 无法评估")
        return 1
    close, open_ = m["close"], m["open"]
    pct, amount = m["pct_chg"], m["amount"]
    meta = m.get("meta") or {}
    print(f"矩阵：{close.shape[0]} 交易日 × {close.shape[1]} 只"
          f"（{close.index[0]} ~ {close.index[-1]}）")
    print(f"面板指纹：{meta.get('signature', '?')}")

    # ── 判据（与线上口径逐字一致）──
    avg20 = amount.rolling(20, min_periods=15).mean().shift(1)
    up = pct >= MIN_CHANGE_PCT
    dn = pct <= -MIN_CHANGE_PCT
    vol = amount >= VOLUME_RATIO * avg20
    signal = up & vol
    # 位置：近 20 日累计涨幅（追高程度）
    mom20 = close.pct_change(20, fill_method=None)

    fwd = forward_returns(close, horizons=HORIZONS, entry="next_open", open_=open_)

    print(f"\n信号样本密度：全样本 {int(signal.sum().sum())} 个 (日,票) "
          f"命中，日均 {signal.sum(axis=1).mean():.0f} 只")

    variants = {
        "线上判据（涨3%+放量1.5×）": signal,
        "仅涨3%（不放量）": up,
        "仅放量（不涨3%）": vol & ~up,
        "反向（跌3%+放量）": dn & vol,
        "线上判据+不追高(20日<15%)": signal & (mom20 < 0.15),
        "线上判据+跑赢大盘": signal & (pct.sub(pct.mean(axis=1), axis=0) > 0),
    }

    for h in HORIZONS:
        r = fwd[h]
        rows = []
        for name, mask in variants.items():
            df = _excess(r, mask, step=h)
            rows.append((f"{name}", df))
        _report(f"持有期 T+{h} 开盘→T+{h + 1} 开盘（step={h}，剔除重叠样本）", rows)

    # ── 涨幅分档（同一持有期，看"涨多少"的影响）──
    h = 5
    r = fwd[h]
    bands = {
        "涨 3~5% + 放量": signal & (pct < 5.0),
        "涨 5~7% + 放量": signal & (pct >= 5.0) & (pct < 7.0),
        "涨 7~10% + 放量": signal & (pct >= 7.0) & (pct < 10.0),
        "涨 >10% + 放量": signal & (pct >= 10.0),
    }
    rows = [(k, _excess(r, v, step=h)) for k, v in bands.items()]
    _report(f"涨幅分档（持有期 T+{h}，看'涨多少'是不是越涨越差）", rows)

    # ── 放量分档 ──
    ratio = amount / avg20
    vol_bands = {
        "涨3% + 放量1.5~2×": up & (ratio >= 1.5) & (ratio < 2.0),
        "涨3% + 放量2~3×": up & (ratio >= 2.0) & (ratio < 3.0),
        "涨3% + 放量>3×": up & (ratio >= 3.0),
    }
    rows = [(k, _excess(r, v, step=h)) for k, v in vol_bands.items()]
    _report(f"放量分档（持有期 T+{h}）", rows)

    # ── 分年（多区间检验）──
    for h in (5, 20):
        r = fwd[h]
        df = _excess(r, signal, step=h)
        rows = []
        years = pd.to_datetime(df.index).year
        for year, g in df.groupby(years):
            rows.append((f"{year} 年", g))
        rows.append(("全区间", df))
        _report(f"分年超额 · 线上判据 · 持有期 T+{h}", rows)

    # ── 候选筛选器视角：判据每天命中多少只（决定要不要收紧范围）──
    daily = signal.sum(axis=1)
    print(f"\n{'=' * 78}\n判据作为『候选筛选器』的产出量（是否筛得住）\n{'=' * 78}")
    print(f"全市场 {close.shape[1]} 只 ⇒ 日均命中 {daily.mean():.0f} 只"
          f"（占 {daily.mean() / close.shape[1] * 100:.1f}%），"
          f"中位 {daily.median():.0f} 只，P90 {daily.quantile(0.9):.0f} 只")
    return 0


if __name__ == "__main__":
    sys.exit(main())
