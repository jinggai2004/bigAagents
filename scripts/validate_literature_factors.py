# -*- coding: utf-8 -*-
"""文献因子在本地面板上的可实践性检验（2016-2026 全历史，只依赖 OHLCV）。

候选因子全部来自已发表论文，且**不需要任何新数据源**：

| 因子 | 出处 | 假设方向 |
|---|---|---|
| `ivol20`   | Ang et al.(2006)；中国证据 IREF(2018)：**IVOL 不被 MAX 吸收** | 负 |
| `sd20`     | 20 日总波动率（项目现用 `base_volatility20` 口径） | 负 |
| `max5/20`  | Bali et al.(2011)；中国 Carpenter et al.(2021) −0.85%/月 | 负（彩票） |
| `min5`     | IREF(2018) MIN 效应，弱于 MAX | 正/弱 |
| `skew20`   | 特质偏度（彩票偏好） | 负 |
| `ret5`     | 短期反转 | 负 |
| `ret20`    | 中期动量（项目已测为负） | 负 |
| `amihud20` | Amihud(2002) 非流动性 | 正 |
| `log_amt20`| 成交额（流动性/规模代理） | 负 |

⚠️ **面板字段分两代**（必须先知道）：
  0003~0016 按日切片，只有 close/preclose/pct_chg/amount/volume；
  0017~0027 按股票分块，才有 turn/pe_ttm/is_st（仅覆盖 30% 行）。
  ⇒ 换手率、E/P 这类**文献里最强的因子在本面板上算不全**，本轮只做 OHLCV 因子。

两件事必须做，否则数字会骗人：
  1. **增量检验**：对 [log 成交额, 20日总波动率] 截面正交化后再算 IC。
     否则 MAX/偏度/IVOL 只是把"波动率"重新发现一遍。
  2. **CH-3 分组**：Liu-Stambaugh-Yuan(2019) 指出 A 股最小的 30% 是借壳"壳股"、
     定价失真 ⇒ 按成交额分三层，看因子在小盘组是否失效。

复现：python scripts/validate_literature_factors.py
"""
from __future__ import annotations

import glob
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = sorted(glob.glob(os.path.join(ROOT, "data", "panel", "parts", "*.parquet")))
_LIMIT = int(os.environ.get("BIGA_FACTOR_LIMIT", "0"))
if _LIMIT:
    PARTS = PARTS[:_LIMIT]
OUT_JSON = os.path.join(ROOT, "output", "文献因子可实践性检验_2026-10-29.json")

COLS = ["stock_code", "trade_date", "close", "preclose", "pct_chg", "amount"]
WINDOW = 20
MIN_STOCKS = 300
HORIZONS = (5, 10)


def load_panel() -> pd.DataFrame:
    frames = []
    for p in PARTS:
        frames.append(pd.read_parquet(p, columns=COLS))
    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str).str[:10]
    df = (df.sort_values(["stock_code", "trade_date"])
            .drop_duplicates(["stock_code", "trade_date"], keep="last"))
    df = df.dropna(subset=["close", "preclose"])
    return df[df["preclose"] > 0].reset_index(drop=True)


def wide(df: pd.DataFrame, col: str, columns=None) -> pd.DataFrame:
    """宽表。

    ⚠️ `pivot` 会**丢掉全空列** ⇒ 不同字段的列集可能不相等
    （实测 amount 5815 列、pct_chg 5814 列），后续按位置做数组运算会直接崩。
    所以一律以 `close` 的列集为基准 reindex。
    """
    w = df.pivot(index="trade_date", columns="stock_code", values=col).astype("float32")
    if columns is not None:
        w = w.reindex(columns=columns)
    return w


def rolling_stats(r: np.ndarray, w: int) -> dict:
    """滚动 w 日：总波动率 / 与等权市场的 β / IVOL —— 累加和实现，避免逐股 rolling。"""
    valid = np.isfinite(r)
    xz = np.where(valid, r, 0.0)
    cnt_d = valid.sum(axis=1)
    mkt = np.divide(xz.sum(axis=1), cnt_d,
                    out=np.full(len(r), np.nan), where=cnt_d > 0)
    mz = np.where(valid, mkt[:, None], 0.0)

    def csum(a):
        out = np.zeros((a.shape[0] + 1, a.shape[1]))
        np.cumsum(a, axis=0, out=out[1:])
        return out

    def win(c):
        out = np.full_like(c, np.nan)
        out[w:] = c[w:] - c[:-w]
        return out

    n = win(csum(valid.astype("float64")))
    sx, sxx = win(csum(xz)), win(csum(xz * xz))
    sm, smm, sxm = win(csum(mz)), win(csum(mz * mz)), win(csum(xz * mz))
    with np.errstate(divide="ignore", invalid="ignore"):
        mx = sx / n
        var_x = sxx / n - mx ** 2
        mm = sm / n
        var_m = smm / n - mm ** 2
        cov = sxm / n - mx * mm
        beta = cov / var_m
        ivol = np.sqrt(np.maximum(var_x - beta ** 2 * var_m, 0.0))
        sd = np.sqrt(np.maximum(var_x, 0.0))
    ok = n >= w * 0.8
    for a in (beta, ivol, sd):
        a[~ok] = np.nan
    # ⚠️ 累加和数组有 rows+1 行（首行是全 0 哨兵）⇒ 上面几个数组也多一行，
    #    必须裁掉，否则和 `log_amt` 这类「原生 rows 行」的数组做 stack 会形状不符。
    k = len(r)
    return {"ivol": ivol[:k], "sd": sd[:k], "beta": beta[:k], "mkt": mkt}


def main() -> int:
    t0 = time.time()
    print("[1/6] 读面板…")
    df = load_panel()
    print(f"      {len(df):,} 行 / {df['stock_code'].nunique()} 只 / "
          f"{df['trade_date'].nunique()} 日  ({time.time() - t0:.0f}s)")

    print("[2/6] 转宽表…")
    close = wide(df, "close")
    ref_cols = close.columns
    ref_idx = close.index
    r_all = wide(df, "pct_chg", ref_cols).reindex(ref_idx).to_numpy(dtype="float64") / 100.0
    amt = wide(df, "amount", ref_cols).reindex(ref_idx).to_numpy(dtype="float64")
    idx, cols = ref_idx, ref_cols
    del df
    print(f"      矩阵 {close.shape}  ({time.time() - t0:.0f}s)")
    c = close.to_numpy(dtype="float64")
    del close

    print("[3/6] 滚动统计…")
    st = rolling_stats(r_all, WINDOW)

    def roll(a: np.ndarray, w: int, fn: str) -> np.ndarray:
        return getattr(pd.DataFrame(a).rolling(w, min_periods=max(2, int(w * 0.75))), fn)().to_numpy()

    max5, max20 = roll(r_all, 5, "max"), roll(r_all, 20, "max")
    min5 = roll(r_all, 5, "min")
    skew20 = roll(r_all, 20, "skew")
    ret5, ret20 = roll(r_all, 5, "sum"), roll(r_all, 20, "sum")
    with np.errstate(divide="ignore", invalid="ignore"):
        amih = np.abs(r_all) / np.where(amt > 0, amt, np.nan)
    amihud20 = roll(amih, 20, "mean") * 1e9
    mean_amt20 = roll(amt, 20, "mean")
    log_amt = np.log(np.where(mean_amt20 > 0, mean_amt20, np.nan))
    print(f"      完成 ({time.time() - t0:.0f}s)")

    factors = {
        "ivol20": st["ivol"], "sd20": st["sd"], "beta20": st["beta"],
        "max5": max5, "max20": max20, "min5": min5, "skew20": skew20,
        "ret5": ret5, "ret20": ret20,
        "amihud20": amihud20, "log_amt20": log_amt,
    }

    print("[4/6] 前瞻收益…")
    # ⚠️ 必须用 **pct_chg 复利**，不能用 close[t+h]/close[t]：
    #    面板的 close 是原始价，跨除权除息日会凭空产生一段"亏损"（分红越大越负），
    #    而 `pct_chg` 是按 `preclose`（已含除权调整）算的，是唯一干净的日收益。
    lr = np.log1p(r_all)
    fwd = {}
    for h in HORIZONS:
        rs = pd.DataFrame(lr).rolling(h, min_periods=h).sum().to_numpy()
        f = np.full_like(r_all, np.nan)
        f[:-h] = rs[h:]                      # t 日的 h 日前瞻 = t+1..t+h 的累计对数收益
        fwd[h] = np.expm1(f)
    print(f"      完成 ({time.time() - t0:.0f}s)")

    def ic_series(fac, y, mask=None):
        ics, days = [], []
        for i in range(len(idx)):
            a, b = fac[i], y[i]
            m = np.isfinite(a) & np.isfinite(b)
            if mask is not None:
                m &= mask[i]
            if m.sum() < MIN_STOCKS:
                continue
            ra = pd.Series(a[m]).rank().to_numpy()
            rb = pd.Series(b[m]).rank().to_numpy()
            sa, sb = ra.std(), rb.std()
            if sa == 0 or sb == 0:
                continue
            ics.append(float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb)))
            days.append(idx[i])
        return ics, days

    def summarize(ics):
        if len(ics) < 60:
            return {}
        a = np.array(ics)
        mu, sd = a.mean(), a.std(ddof=1)
        icir = mu / sd if sd > 0 else float("nan")
        t = icir * np.sqrt(len(a)) if sd > 0 else float("nan")
        return {"ic": round(mu, 4), "icir": round(float(icir), 3),
                "t": round(float(t), 2), "pos": round(float((a > 0).mean()), 3),
                "n": int(len(a))}

    def yearly(ics, days):
        s = pd.Series(ics, index=pd.to_datetime(days))
        return {str(k): round(float(v), 4) for k, v in s.groupby(s.index.year).mean().items()}

    def quintile(fac, y):
        buckets = {k: [] for k in range(5)}
        for i in range(len(idx)):
            a, b = fac[i], y[i]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < MIN_STOCKS:
                continue
            q = pd.qcut(pd.Series(a[m]), 5, labels=False, duplicates="drop")
            if q.nunique() < 5:
                continue
            qa = q.to_numpy()
            for k in range(5):
                buckets[k].append(float(b[m][qa == k].mean()))
        means = [round(float(np.mean(v)) * 100, 3) if v else None for v in buckets.values()]
        spread = (round(means[4] - means[0], 3)
                  if means[0] is not None and means[4] is not None else None)
        return {"quintiles": means, "q5_minus_q1": spread}

    print("[5/6] 算 IC / 分年 / 分位…")
    results = {}
    for name, arr in factors.items():
        for h in HORIZONS:
            ics, days = ic_series(arr, fwd[h])
            s = summarize(ics)
            if not s:
                continue
            results[f"{name}@{h}d"] = {"factor": name, "horizon": h, **s,
                                       "yearly": yearly(ics, days),
                                       "quintile": quintile(arr, fwd[h])}
        print(f"      {name:10s} ({time.time() - t0:.0f}s)")

    # ---- 增量：对 [log 成交额, sd20] 正交化 ----
    print("[6/6] 增量检验 + CH-3 分层…")
    base = np.stack([log_amt, st["sd"]], axis=-1)
    incr = {}
    for name, arr in factors.items():
        if name in ("log_amt20", "sd20"):
            continue
        resid = np.full(arr.shape, np.nan)
        for i in range(len(idx)):
            a, b = arr[i], base[i]
            m = np.isfinite(a) & np.isfinite(b).all(axis=1)
            if m.sum() < MIN_STOCKS:
                continue
            X = np.column_stack([np.ones(m.sum()), b[m]])
            try:
                coef, *_ = np.linalg.lstsq(X, a[m], rcond=None)
            except np.linalg.LinAlgError as e:       # 基准列共线（极少），跳过当日
                _logger.debug("[文献因子] %s 正交化失败(%s): %s: %s",
                              name, idx[i], type(e).__name__, e)
                continue
            resid[i, m] = a[m] - X @ coef
        ics, days = ic_series(resid, fwd[5])
        s = summarize(ics)
        if s:
            incr[f"{name}@5d"] = {"factor": name, **s,
                                  "yearly": yearly(ics, days),
                                  "quintile": quintile(resid, fwd[5])}

    # ---- CH-3：按当日成交额三分层，看因子在小盘组是否失效 ----
    tiers = np.full(log_amt.shape, np.nan)
    for i in range(len(idx)):
        a = log_amt[i]
        m = np.isfinite(a)
        if m.sum() < MIN_STOCKS:
            continue
        q = pd.qcut(pd.Series(a[m]), 3, labels=False, duplicates="drop")
        tiers[i, m] = q.to_numpy()
    small = tiers == 0
    mid = tiers == 1

    tier_res = {}
    for name in ("ivol20", "max5", "ret5"):
        arr = factors[name]
        for tag, mask in (("小盘1/3", small), ("中盘1/3", mid)):
            ics, days = ic_series(arr, fwd[5], mask=mask)
            s = summarize(ics)
            if s:
                tier_res[f"{name}@{tag}"] = {"factor": f"{name}@{tag}", **s}
        print(f"      分层 {name:8s} ({time.time() - t0:.0f}s)")

    payload = {"generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
               "sample": {"rows": int(np.isfinite(r_all).sum()),
                          "stocks": int(len(cols)), "days": int(len(idx)),
                          "start": str(idx[0]), "end": str(idx[-1])},
               "raw": results, "orthogonalized": incr, "tiers": tier_res}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    def dump(title, rows, with_mono=True):
        print()
        print("=" * 104)
        print(title)
        print("=" * 104)
        head = f"{'因子':12s}{'IC':>9s}{'ICIR':>8s}{'t':>8s}{'同向率':>8s}{'Q1':>8s}{'Q5':>8s}{'Q5-Q1':>8s}"
        print(head + ("单调" if with_mono else ""))
        for v in sorted(rows, key=lambda x: -abs(x["ic"])):
            q = v["quintile"]["quintiles"]
            mono = ""
            if with_mono and all(x is not None for x in q):
                d = np.diff(q)
                mono = "是" if (d > 0).all() or (d < 0).all() else "否"
            print(f"{v['factor']:12s}{v['ic']:9.4f}{v['icir']:8.3f}{v['t']:8.2f}"
                  f"{v['pos']:8.1%}"
                  f"{(q[0] if q[0] is not None else float('nan')):8.3f}"
                  f"{(q[4] if q[4] is not None else float('nan')):8.3f}"
                  f"{(v['quintile']['q5_minus_q1'] or float('nan')):8.3f}{mono}")

    dump("原始 IC · 前瞻 5 日", [v for k, v in results.items() if v["horizon"] == 5])
    dump("原始 IC · 前瞻 10 日", [v for k, v in results.items() if v["horizon"] == 10])
    dump("增量 IC（对 成交额 + 总波动率 正交化后，前瞻 5 日）",
         list(incr.values()), with_mono=False)

    print()
    print("=" * 104)
    print("CH-3 分层（Liu-Stambaugh-Yuan 2019：最小 30% 是壳股、定价失真）· 前瞻 5 日")
    print("=" * 104)
    print(f"{'因子@分层':22s}{'IC':>9s}{'ICIR':>8s}{'t':>8s}{'n':>7s}")
    for k, v in tier_res.items():
        print(f"{k:22s}{v['ic']:9.4f}{v['icir']:8.3f}{v['t']:8.2f}{v['n']:7d}")

    print()
    print(f"[输出] {OUT_JSON}")
    print(f"[耗时] {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
