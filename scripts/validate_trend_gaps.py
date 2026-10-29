# -*- coding: utf-8 -*-
"""P3-15 补验：趋势否决/弱市超跌结论在「可成交 + 有流动性」样本上是否还成立。

对应 `output/趋势否决阈值验证_2026-10-24.md` 第八节列出的三项未覆盖：

1. **涨停不可买入偏差** —— 原口径假设「T+1 开盘价一定能买到」。本脚本按板块
   涨跌幅上限计算 T+1 涨停价，开盘价已封板（一字涨停）的样本直接剔除。
2. **ST 未剔** —— 面板 `is_st` 整列几乎全 NULL（抽样 60 只仅 1 只有值），
   **无法按历史 ST 过滤 ⇒ 本项标「不可评」**，不伪造验证结论。
   副作用：主板 ST 实际限 5%，本脚本按 10% 判涨停，会**低估**不可买比例，
   属保守方向（过滤偏少，若结论仍成立则更稳）。
3. **未在真实候选池上验证** —— 初筛是「按成交额降序截断」而非绝对下限，
   但结论可能被低流动小票主导（`replay_eval` 铁律）。故用 **20 日均成交额**
   分档加约束（3,000 万 / 1 亿）做敏感性，而不是凭一个阈值下结论。

口径与 v2 完全一致：前复权 `f[t]=f[t+1]*preclose[t+1]/close[t]`、T+1 开盘买 /
T+1+h 开盘卖、停牌按 `volume>0`、上市满 120 个交易日、减**全市场等权**基准
（四种样本口径共用同一基准，保证差异只来自样本构成，不来自基准漂移）。

用法
----
    python scripts/validate_trend_gaps.py --limit 200     # 冒烟（约 20 秒）
    python scripts/validate_trend_gaps.py                 # 全量（约 3~6 分钟，零联网）

输出：控制台表格 + `output/趋势否决三项补验_<date>.json`（全部数值，供写报告复核）。
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import panel_store  # noqa: E402

_logger = logging.getLogger(__name__)

BARS_DIR = os.path.join(panel_store.PANEL_DIR, "by_code")
INDEX_DIR = os.path.join(panel_store.PANEL_DIR, "index")
COLS = ["trade_date", "open", "close", "preclose", "volume", "amount"]
HS = (5, 10, 20)
EDGES = np.array([-np.inf, -0.10, -0.03, 0.03, 0.10, 0.15, 0.25, np.inf])
LABELS = ["<-10%", "-10~-3%", "-3~3%", "3~10%", "10~15%", "15~25%", ">25%"]
MIN_HISTORY = 120
MIN_CELL = 200
LIQ_FLOOR_MID = 3.0e7      # 20 日均成交额 3,000 万元
LIQ_FLOOR_HIGH = 1.0e8     # 20 日均成交额 1 亿元
SCENARIOS = ("S0_原始全截面", "S1_剔涨停不可买", "S2_S1加均额3000万", "S3_S1加均额1亿")
YEAR_SCENARIOS = (1, 2)    # 分年只算 S1 / S2，控制耗时


def board_limit(code: str) -> float:
    """按板块返回涨跌幅上限（不含 ST 的 5%，原因见模块 docstring）。

    创业板 30xxxx / 科创板 68xxxx → 20%；北交所 920xxx / 8xxxxx / 4xxxxx → 30%；
    其余（沪深主板 60/00/001/002/003）→ 10%。
    """
    if code.startswith(("30", "68")):
        return 0.20
    if code.startswith("920") or code.startswith(("8", "4")):
        return 0.30
    return 0.10


def _factor(close: np.ndarray, pre: np.ndarray) -> np.ndarray:
    """前复权因子（锚末根）。⚠️ 不能用 pct_chg 累乘。"""
    n = len(close)
    f = np.ones(n)
    for t in range(n - 2, -1, -1):
        c, p = close[t], pre[t + 1]
        f[t] = f[t + 1] * (p / c) if (c > 0 and p > 0) else f[t + 1]
    return f


def _load(path: str) -> dict | None:
    """读单只票，返回信号/收益/可成交/流动性数组。数据不足返回 None。"""
    code = os.path.splitext(os.path.basename(path))[0]
    try:
        df = pd.read_parquet(path, columns=COLS)
    except (OSError, ValueError) as e:
        _logger.warning("[跳过] %s: %s", code, e)
        return None
    if len(df) < MIN_HISTORY + 25:
        return None
    df = df.sort_values("trade_date").reset_index(drop=True)
    n = len(df)

    def num(col: str) -> np.ndarray:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")

    close, openp, pre = num("close"), num("open"), num("preclose")
    vol, amt = num("volume"), num("amount")
    ok = np.isfinite(close) & np.isfinite(openp) & np.isfinite(pre) & (close > 0) & (openp > 0)
    traded = ok & np.isfinite(vol) & (vol > 0)      # 停牌=无量（不能用 trade_status）
    if ok.sum() < MIN_HISTORY + 25:
        return None

    f = _factor(np.where(ok, close, 1.0), np.where(ok, pre, 1.0))
    qclose, qopen = f * close, f * openp
    ret20 = np.full(n, np.nan)
    ret20[20:] = qclose[20:] / qclose[:-20] - 1

    fwd: dict[int, np.ndarray] = {}
    for h in HS:
        v = np.full(n, np.nan)
        m = n - (1 + h)
        if m > 0:
            v[:m] = qopen[1 + h:1 + h + m] / qopen[1:1 + m] - 1
        fwd[h] = v

    seq = np.cumsum(np.where(ok, 1, 0))
    valid = traded & (seq >= MIN_HISTORY) & np.isfinite(ret20)
    for h in HS:
        m = n - (1 + h)
        nxt = np.zeros(n, dtype=bool)
        if m > 0:
            nxt[:m] = traded[1:1 + m] & traded[1 + h:1 + h + m]
        valid &= nxt

    # ── 补验①：T+1 开盘一字涨停 ⇒ 现实买不进。用**未复权**价格判定（涨停基于真实价）。
    limit = board_limit(code)
    limit_px = np.round(pre * (1.0 + limit), 2)
    blocked = np.zeros(n, dtype=bool)
    blocked[:n - 1] = openp[1:] >= limit_px[1:] - 1e-6
    buyable = ~blocked

    # ── 补验③：20 日均成交额（截至信号日 T，T+1 才买 ⇒ 无前视）。不足 20 日为 NaN ⇒ 剔除。
    liq20 = pd.Series(amt).rolling(20, min_periods=20).mean().to_numpy()

    return {
        "dates": df["trade_date"].astype(str).to_numpy(),
        "ret20": ret20,
        "fwd": fwd,
        "valid": valid,
        "masks": (
            np.ones(n, dtype=bool),
            buyable,
            buyable & (liq20 >= LIQ_FLOOR_MID),
            buyable & (liq20 >= LIQ_FLOOR_HIGH),
        ),
        "blocked": blocked,
    }


def index_panel() -> tuple[str, np.ndarray, pd.Series]:
    pick = None
    for want in ("sh000852", "000852", "sh000300", "000300"):
        hit = [f for f in glob.glob(os.path.join(INDEX_DIR, "*.parquet"))
               if want in os.path.basename(f)]
        if hit:
            pick = sorted(hit)[0]
            break
    if not pick:
        raise SystemExit(f"找不到指数文件：{INDEX_DIR}")
    df = pd.read_parquet(pick).sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce")
    return os.path.basename(pick), df["date"].astype(str).to_numpy(), close


def main() -> int:
    ap = argparse.ArgumentParser(description="P3-15 三项补验")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 只（冒烟用；0=全量）")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(BARS_DIR, "*.parquet")))
    if args.limit > 0:
        files = files[:args.limit]
    name, idates, iclose = index_panel()
    didx = {d: i for i, d in enumerate(idates)}
    D = len(idates)
    ma20 = iclose.rolling(20).mean().to_numpy()
    weak_mask = (iclose.to_numpy() <= ma20)          # 主口径：中证1000 收盘 ≤ MA20
    years = sorted({d[:4] for d in idates})
    yidx = {y: i for i, y in enumerate(years)}
    NY, NB = len(years), len(LABELS)
    t_start = time.monotonic()

    print(f"股票数：{len(files)}｜指数：{name}｜弱市占比 {np.nanmean(weak_mask) * 100:.1f}%"
          f"｜分年 {years[0]}~{years[-1]}", flush=True)

    # ── pass1：全市场等权基准（四种样本口径共用）──
    ssum = {h: defaultdict(float) for h in HS}
    scnt = {h: defaultdict(int) for h in HS}
    for i, fp in enumerate(files, 1):
        got = _load(fp)
        if got is not None:
            for h in HS:
                v = got["fwd"][h]
                m = got["valid"] & np.isfinite(v)
                if m.any():
                    for d, val in zip(got["dates"][m], v[m]):
                        ssum[h][d] += float(val)
                        scnt[h][d] += 1
        if i % 1000 == 0:
            print(f"  pass1 {i}/{len(files)}  {time.monotonic() - t_start:.0f}s", flush=True)
    bench = {h: {d: ssum[h][d] / scnt[h][d] for d in ssum[h] if scnt[h][d]} for h in HS}

    # ── pass2：四种样本口径 × 强/弱市 × 档；分年另算 S1/S2 ──
    shape = (2, D, NB)          # 轴0：0=强市 1=弱市
    sum_exc = {h: np.zeros((len(SCENARIOS),) + shape) for h in HS}
    cnt_exc = {h: np.zeros((len(SCENARIOS),) + shape, dtype=np.int64) for h in HS}
    yr_sum = {s: np.zeros(NY * NB) for s in YEAR_SCENARIOS}
    yr_cnt = {s: np.zeros(NY * NB, dtype=np.int64) for s in YEAR_SCENARIOS}
    blocked_total = 0
    valid_total = 0

    for i, fp in enumerate(files, 1):
        got = _load(fp)
        if got is None:
            continue
        dates, ret20, valid = got["dates"], got["ret20"], got["valid"]
        rows = np.array([didx.get(d, -1) for d in dates])
        if not (valid & (rows >= 0)).any():
            continue
        bidx = np.digitize(ret20, EDGES[1:-1])
        weak = weak_mask[np.clip(rows, 0, D - 1)]
        yarr = np.array([yidx.get(d[:4], -1) for d in dates])
        valid_total += int((valid & (rows >= 0)).sum())
        blocked_total += int((valid & (rows >= 0) & got["blocked"]).sum())

        for h in HS:
            bmap = np.array([bench[h].get(d, np.nan) for d in dates])
            exc = got["fwd"][h] - bmap
            for si in range(len(SCENARIOS)):
                m = valid & got["masks"][si] & (rows >= 0) & np.isfinite(exc)
                if not m.any():
                    continue
                r, b, e = rows[m], bidx[m], exc[m]
                w = weak[m].astype(np.int64)
                np.add.at(sum_exc[h][si], (w, r, b), e)
                np.add.at(cnt_exc[h][si], (w, r, b), 1)
                if h == 10 and si in YEAR_SCENARIOS:
                    mw = m & weak
                    if mw.any():
                        key = yarr[mw] * NB + bidx[mw]
                        good = key >= 0
                        yr_sum[si] += np.bincount(key[good], weights=exc[mw][good],
                                                  minlength=NY * NB)
                        yr_cnt[si] += np.bincount(key[good], minlength=NY * NB)
        if i % 1000 == 0:
            print(f"  pass2 {i}/{len(files)}  {time.monotonic() - t_start:.0f}s", flush=True)

    def cell(h: int, si: int, weak_side: int, j: int) -> tuple[float, int]:
        n = int(cnt_exc[h][si][weak_side, :, j].sum())
        if n < MIN_CELL:
            return float("nan"), n
        return float(sum_exc[h][si][weak_side, :, j].sum()) / n, n

    print(f"\n有效样本 {valid_total}，其中 T+1 开盘涨停不可买 {blocked_total}"
          f"（{blocked_total / max(valid_total, 1) * 100:.2f}%）")
    print("\n═══ 主表：h=10 平均超额（%），括号内样本数；n<200 记 — ═══")
    for side, side_name in ((1, "弱市（指数≤MA20）"), (0, "强市")):
        print(f"\n--- {side_name} ---")
        print(f"{'ret20档':>9} | " + " | ".join(f"{s:>22}" for s in SCENARIOS))
        print("-" * 108)
        for j, lb in enumerate(LABELS):
            cells = []
            for si in range(len(SCENARIOS)):
                v, n = cell(10, si, side, j)
                cells.append(f"{v * 100:+7.2f}% ({n:>7})" if np.isfinite(v)
                             else f"{'—':>7}  (n={n:>6})")
            print(f"{lb:>9} | " + " | ".join(cells))

    print("\n═══ 分年一致性（h=10，仅弱市）═══")
    yearly: dict[str, dict] = {}
    for si in YEAR_SCENARIOS:
        print(f"\n--- {SCENARIOS[si]} ---")
        yearly[SCENARIOS[si]] = {}
        for j, lb in enumerate(LABELS):
            cnt = np.array([yr_cnt[si][y * NB + j] for y in range(NY)])
            tot_n = int(cnt.sum())
            if tot_n == 0:
                continue
            means = np.full(NY, np.nan)
            for y in range(NY):
                if cnt[y] > 0:
                    means[y] = yr_sum[si][y * NB + j] / cnt[y]
            tot = float(sum_exc[10][si][1, :, j].sum()) / tot_n if tot_n else float("nan")
            neg = int(np.nansum(means < 0))
            have = int(np.isfinite(means).sum())
            print(f"  {lb:>7} 全样本 {tot * 100:+.2f}%  n={tot_n}  负年份 {neg}/{have}")
            print("      " + "  ".join(f"{years[y]}:{means[y] * 100:+5.2f}%"
                                       for y in range(NY) if np.isfinite(means[y])))
            yearly[SCENARIOS[si]][lb] = {
                "n": tot_n, "mean_excess": tot, "neg_years": neg, "years_with_data": have,
                "by_year": {years[y]: (float(means[y]) if np.isfinite(means[y]) else None)
                            for y in range(NY)},
            }

    payload = {
        "date": str(date.today()),
        "index": name,
        "stock_files": len(files),
        "valid_samples": valid_total,
        "blocked_by_limit_up_open": blocked_total,
        "weak_share": float(np.nanmean(weak_mask)),
        "scenarios": list(SCENARIOS),
        "labels": LABELS,
        "horizons": list(HS),
        "st_note": "is_st 几乎全 NULL，无法按历史 ST 过滤 ⇒ 本项不可评；"
                   "主板 ST 实际限 5%，本脚本按 10% 判涨停，低估不可买比例（保守方向）。",
        "main_h10": {
            side_name: {
                lb: {SCENARIOS[si]: {"mean_excess": cell(10, si, side, j)[0],
                                     "n": cell(10, si, side, j)[1]}
                     for si in range(len(SCENARIOS))}
                for j, lb in enumerate(LABELS)
            } for side, side_name in ((1, "weak"), (0, "strong"))
        },
        "yearly_h10_weak": yearly,
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }
    out = os.path.join(panel_store.PROJECT_ROOT if hasattr(panel_store, "PROJECT_ROOT")
                       else os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "output", f"趋势否决三项补验_{date.today()}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\n数值已写：{out}\n总耗时 {payload['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
