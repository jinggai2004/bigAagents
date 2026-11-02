"""择时信号检验：市场状态检测器的那些指标，到底能不能预测下期收益？

## 为什么要做这个

`core/market_detector.py` 把 14 个指标加权成总分，再按阈值映射到 5 档仓位上限
（≥80→80% / ≥60→60% / ≥40→40% / ≥20→20% / else→0%）。这套映射**从未验证过**：
`grep -c "IC\\|回测\\|验证" core/market_detector.py` 的结果是 0，阈值纯经验标定。

组合级 A/B 已经给出一个刺眼的数字：现状策略全期超额 t=0.73、月胜率 48%（90 个月），
从来没显著过。如果选股端补不上，那"该不该停/该不该降仓"就成了更值钱的问题——
而这正是 detector 负责的事。

## 做法（成本最低、结论最硬的那种）

不去重构 detector（它的 14 个 `_score_*` 各自内部取数、没有 `asof` 参数，改它等于动
线上路径）。改为**直接检验它依赖的假设**：月末可观测的技术指标，能不能预测下月收益。

指标全部从指数日线本地计算（`data/index_bars`，沪深300 有 2002 年至今 6,020 行），
覆盖 detector 趋势维度 3 个 + 量能维度 1 个（权重合计 0.42）。

两个目标口径：
- **主：全市场等权月度收益**（与策略的 benchmark 同口径，90 个月）
- 辅：沪深300 月度收益（20 年样本，看长周期是否稳定）

## 判据

- 单信号月度 IC + t（Spearman，与因子研究同一套口径）
- 分 3 档看下月收益是否单调
- 最后按 detector 的 5 档映射模拟"择时降仓 vs 始终满仓"

⚠️ 择时信号容易过拟合，且这里的试验次数同样要计入多重检验。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import logging

_logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUT = _ROOT / "output"

# 与 detector 的 5 档映射完全一致（阈值 → 仓位上限）
DETECTOR_BANDS = [(80, 0.80), (60, 0.60), (40, 0.40), (20, 0.20), (0, 0.00)]


def _band_limit(score: float) -> float:
    for thr, lim in DETECTOR_BANDS:
        if score >= thr:
            return lim
    return 0.0


def index_daily(symbol: str = "sh000300") -> pd.DataFrame:
    """指数日线（本地，含 date/close/volume）。"""
    import data.index_bars as ib
    df = ib.get_index_bars(symbol, min_rows=60)
    if df is None or df.empty:
        raise SystemExit(f"本地缺 {symbol} 日线（跑 `py -m data.index_bars --backfill`）")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ("close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """日频指标 → 月末快照（只用**当日及以前**的数据，无前视）。"""
    d = df.copy()
    close = d["close"]
    ret = close.pct_change()

    d["ma20_dev"] = close / close.rolling(20).mean() - 1.0        # detector: hs300_ma20_deviation
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = (dif - dea) / close                          # detector: sh_macd（归一化）
    d["mom20"] = close / close.shift(20) - 1.0
    d["vol20"] = ret.rolling(20).std() * np.sqrt(252)
    vol = d["volume"]
    d["amt_ratio"] = vol / vol.rolling(20).mean() - 1.0           # detector: volume_vs_ma20

    SIG = ["ma20_dev", "macd_hist", "mom20", "vol20", "amt_ratio"]
    d["ym"] = d["date"].dt.to_period("M")
    month_end = d.groupby("ym").tail(1).set_index("ym")
    monthly = month_end[["date", *SIG]].copy()
    # 下月收益：本月末收盘 → 下月末收盘（= 用当月可观测信号预测下月）
    monthly["fwd"] = (month_end["close"].shift(-1) / month_end["close"] - 1.0).values
    return monthly.dropna(subset=["fwd"])


def ic_report(m: pd.DataFrame, target: str) -> list[dict]:
    out = []
    for c in ("ma20_dev", "macd_hist", "mom20", "vol20", "amt_ratio"):
        d = m[[c, target]].dropna()
        if len(d) < 24:
            out.append({"signal": c, "ok": False, "reason": f"样本不足（{len(d)}）"})
            continue
        ic = d[c].corr(d[target], method="spearman")
        # IC 的 t：用逐月秩相关的近似（单序列 + 大样本），与 factor_stats 同思路
        n = len(d)
        t = ic * np.sqrt(max(n - 2, 1)) / np.sqrt(max(1 - ic ** 2, 1e-12))
        # 三档
        try:
            q = pd.qcut(d[c], 3, labels=["低", "中", "高"], duplicates="drop")
        except ValueError as e:
            _logger.debug("信号 %s 分档失败: %s", c, e)
            out.append({"signal": c, "ok": False, "reason": "分档失败"})
            continue
        g = d.groupby(q, observed=True)[target].mean() * 100
        out.append({"signal": c, "ok": True, "ic": round(float(ic), 4),
                    "t": round(float(t), 2), "n": n,
                    "low": round(float(g.get("低", np.nan)), 3),
                    "mid": round(float(g.get("中", np.nan)), 3),
                    "high": round(float(g.get("高", np.nan)), 3),
                    "monotonic": bool(g.get("低", 0) < g.get("中", 0) < g.get("高", 0)
                                      or g.get("低", 0) > g.get("中", 0) > g.get("高", 0))
                    if all(k in g.index for k in ("低", "中", "高")) else False})
    return out


# detector 的方向：ma20_dev/macd/mom/amt_ratio 越高越好；vol20 越高越差
SIGNAL_WEIGHTS = {"ma20_dev": 0.10, "macd_hist": 0.10, "mom20": 0.10,
                  "amt_ratio": 0.12, "vol20": -0.07}


def signal_frame(m: pd.DataFrame) -> pd.DataFrame:
    """月度信号 → `_score`(0~100) 与 `_limit`(仓位上限)。**唯一实现**。

    ⚠️⚠️ 一律用**扩展窗口**（只看当期及以前）算标准化与分位。
    首版写成 `(d[c]-d[c].mean())/d[c].std()` + `raw.rank(pct=True)` —— 那是拿
    **全样本**均值/标准差/排名，等于用了未来信息，会把仓位分布人为摊平到 5 档均匀，
    回测很漂亮但实盘做不出来。这类"技术指标择时"最容易在这里作弊。
    """
    d = m.copy()
    z = {}
    for c in SIGNAL_WEIGHTS:
        mu = d[c].expanding(min_periods=24).mean().shift(1)
        sd = d[c].expanding(min_periods=24).std(ddof=1).shift(1)
        z[c] = (d[c] - mu) / sd.where(sd > 0)
    zdf = pd.DataFrame(z)
    raw = (sum(zdf[c] * w for c, w in SIGNAL_WEIGHTS.items())
           / sum(abs(w) for w in SIGNAL_WEIGHTS.values()))
    # 注意 raw=True 时回调拿到的是 ndarray，不是 Series ⇒ 用 s[-1] 不是 s.iloc[-1]
    score = raw.expanding(min_periods=24).apply(
        lambda s: float((s <= s[-1]).mean()) * 100, raw=True).clip(0, 100)
    d = d.assign(_score=score, _limit=score.map(_band_limit))
    # ⚠️ 必须同时要求 `_score` 非空：`map(_band_limit)` 对 NaN 会返回 0.0（所有比较都 False），
    #    只按 `_limit` dropna 的话，预热期的 NaN 分位会带着"0% 仓位"混进结果，
    #    后面 `between(0,100)` 之类的判断就会在一个假 NaN 上失败。
    #    预热期 = 24 期扩充分位 + 1 期 shift ⇒ 前 25 期一律剔除。
    return d.dropna(subset=["_score", "_limit"])


def overlay_on_strategy(m: pd.DataFrame, strat: pd.Series) -> dict:
    """把择时仓位系数叠到**策略月净收益**上，看收益/回撤/超额 t 怎么变。

    ⚠️ 纯多头策略乘仓位系数，收益必然按比例降——所以判据不是"年化更高"，
    而是"单位风险的收益是否改善、灾难年是否少亏"。现金收益按 0 计（保守）。
    """
    sig = signal_frame(m)
    sig.index = sig.index.to_period("M") if not isinstance(sig.index, pd.PeriodIndex) else sig.index
    st = strat.copy()
    if not isinstance(st.index, pd.PeriodIndex):
        st.index = pd.PeriodIndex(st.index, freq="M")
    lim = sig["_limit"].reindex(st.index)
    df = pd.DataFrame({"strat": st, "limit": lim}).dropna()
    if len(df) < 24:
        return {"ok": False, "reason": f"重叠月份不足（{len(df)}）"}

    def _m(r):
        n = len(r)
        ann = (1 + r).prod() ** (12 / n) - 1
        vol = r.std(ddof=1) * np.sqrt(12)
        nav = (1 + r).cumprod()
        return {"ann": round(float(ann), 4),
                "sharpe": round(float(ann / vol), 3) if vol > 0 else None,
                "mdd": round(float((nav / nav.cummax() - 1).min()), 4),
                "vol": round(float(vol), 4)}

    base = _m(df["strat"])
    rules = {}
    # 三条不调参的规则。⚠️ 首版把后两条写成 `x.where(x > 0, 0.0)` / `x.where(x > 0, 0.5)`
    # —— 那是"x<=0 时置 0/0.5，其余保留原值"，与"最低档降仓、其余满仓"完全不是一回事，
    # 而且 `where(x>0, 0.0)` 因为 x 本来就是 0 ⇒ **原样返回**，等于第一条规则的副本
    # （实测三行数字一模一样才发现）。正确语义：低档降仓，其余**满仓**。
    for tag, lim_fn in (
        ("全档映射(仓位=上限)", lambda x: x),
        ("最低档空仓_其余满仓", lambda x: (x > 0).astype(float)),
        ("最低档半仓_其余满仓", lambda x: (x > 0).astype(float).where(x > 0, 0.5)),
    ):
        pos = lim_fn(df["limit"])
        r = df["strat"] * pos
        mm = _m(r)
        mm["avg_position"] = round(float(pos.mean()), 3)
        mm["delta_ann"] = round(mm["ann"] - base["ann"], 4)
        mm["delta_sharpe"] = (round(mm["sharpe"] - base["sharpe"], 3)
                              if mm["sharpe"] and base["sharpe"] else None)
        mm["delta_mdd"] = round(mm["mdd"] - base["mdd"], 4)
        rules[tag] = mm
    return {"ok": True, "months": len(df), "baseline": base, "rules": rules}


def timing_sim(m: pd.DataFrame, target: str) -> dict:
    """按 detector 的 5 档映射模拟择时：市场分 → 仓位上限，乘到下月收益上。

    市场分 = 5 个信号的横截面 z-score **按 detector 的方向**加权合成后线性映射到 0~100。
    方向沿用 detector 的假设（趋势/量能偏高=看多，波动偏高=看空），
    这样检验的正是它那套方向假设——如果 IC 符号反了，这里会直接体现在择时收益上。
    """
    d = signal_frame(m)
    weights = SIGNAL_WEIGHTS
    d["_timed"] = d["_limit"] * d[target]
    d["_full"] = d[target]

    def _stats(sub: pd.DataFrame) -> dict:
        n = len(sub)
        t_ann = (1 + sub["_timed"]).prod() ** (12 / n) - 1
        f_ann = (1 + sub["_full"]).prod() ** (12 / n) - 1
        vol_t = sub["_timed"].std(ddof=1) * np.sqrt(12)
        vol_f = sub["_full"].std(ddof=1) * np.sqrt(12)
        nav_t = (1 + sub["_timed"]).cumprod()
        nav_f = (1 + sub["_full"]).cumprod()
        return {"months": n, "timed_ann": round(float(t_ann), 4),
                "full_ann": round(float(f_ann), 4),
                "delta": round(float(t_ann - f_ann), 4),
                "timed_sharpe": round(float(t_ann / vol_t), 3) if vol_t > 0 else None,
                "full_sharpe": round(float(f_ann / vol_f), 3) if vol_f > 0 else None,
                "timed_mdd": round(float((nav_t / nav_t.cummax() - 1).min()), 4),
                "full_mdd": round(float((nav_f / nav_f.cummax() - 1).min()), 4),
                "avg_position": round(float(sub["_limit"].mean()), 3)}

    seg = {}
    for name, s0, s1 in (("全期", "2000-01-01", "2030-01-01"),
                         ("2019起", "2019-01-01", "2030-01-01"),
                         ("2022灾难年", "2022-01-01", "2022-12-31"),
                         ("2026至今", "2026-01-01", "2030-01-01")):
        sub = d[(d["date"] >= pd.Timestamp(s0)) & (d["date"] <= pd.Timestamp(s1))]
        if len(sub) >= 6:
            seg[name] = _stats(sub)
    return {"segments": seg,
            "band_distribution": {f"{lim:.0%} 仓位": int(n)
                                  for lim, n in d["_limit"].value_counts().sort_index().items()},
            "weights": weights}


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    symbol = "sh000300"
    for i, a in enumerate(argv):
        if a == "--symbol" and i + 1 < len(argv):
            symbol = argv[i + 1]

    df = index_daily(symbol)
    print(f"{symbol} 日线 {len(df)} 行 / {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    m = build_signals(df)
    print(f"月度样本 {len(m)} 个 / {m['date'].iloc[0].date()} ~ {m['date'].iloc[-1].date()}")

    # 补充：全市场等权月收益作为第二目标（与策略 benchmark 同口径）
    mkt = None
    for f in sorted(OUT.glob("组合AB实验_*.json")):
        try:
            bm = json.loads(f.read_text(encoding="utf-8")).get("benchmark_series")
        except Exception as e:                                  # noqa: BLE001
            _logger.debug("跳过不可读的基准序列文件 %s: %s", f.name, e)
            continue
        if bm:
            mkt = pd.Series({pd.Timestamp(k): v for k, v in bm.items()}).sort_index()
            mkt_src = f.name
            break

    report = {"symbol": symbol, "daily_rows": len(df), "months": len(m),
              "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    for tag, tgt in (("沪深300 下月收益", "fwd"),):
        print(f"\n=== 目标：{tag} ===")
        print(f"{'信号':<12}{'IC':>8}{'t':>7}{'低档':>9}{'中档':>9}{'高档':>9}  单调")
        rows = ic_report(m, tgt)
        for r in rows:
            if not r.get("ok"):
                print(f"{r['signal']:<12}  {r.get('reason')}")
                continue
            print(f"{r['signal']:<12}{r['ic']:+8.3f}{r['t']:+7.2f}"
                  f"{r['low']:+8.2f}%{r['mid']:+8.2f}%{r['high']:+8.2f}%  "
                  f"{'是' if r['monotonic'] else '否'}")
        report["ic_hs300"] = rows

    if mkt is not None:
        mm = m.set_index(m["date"].dt.to_period("M"))
        mkt_p = mkt.copy()
        mkt_p.index = mkt_p.index.to_period("M")
        mm = mm.assign(fwd_mkt=mkt_p.reindex(mm.index).values)
        sub = mm.dropna(subset=["fwd_mkt"])
        print(f"\n=== 目标：全市场等权下月收益（{mkt_src}，{len(sub)} 个月）===")
        print(f"{'信号':<12}{'IC':>8}{'t':>7}{'低档':>9}{'中档':>9}{'高档':>9}  单调")
        rows2 = ic_report(sub, "fwd_mkt")
        for r in rows2:
            if not r.get("ok"):
                print(f"{r['signal']:<12}  {r.get('reason')}")
                continue
            print(f"{r['signal']:<12}{r['ic']:+8.3f}{r['t']:+7.2f}"
                  f"{r['low']:+8.2f}%{r['mid']:+8.2f}%{r['high']:+8.2f}%  "
                  f"{'是' if r['monotonic'] else '否'}")
        report["ic_market"] = rows2
        report["sim_on_market"] = timing_sim(sub, "fwd_mkt")

    print(f"\n=== 按 detector 5 档映射模拟择时（扩展窗口分位，无前视）===")
    sim = timing_sim(m, "fwd")
    print(f"{'区间':<12}{'月数':>5}{'择时年化':>10}{'满仓年化':>10}{'差':>9}"
          f"{'择时夏普':>9}{'满仓夏普':>9}{'择时回撤':>10}{'满仓回撤':>10}{'均仓位':>8}")
    for name, s in sim["segments"].items():
        print(f"{name:<12}{s['months']:>5}{s['timed_ann']:+10.1%}{s['full_ann']:+10.1%}"
              f"{s['delta']:+9.1%}{s['timed_sharpe']:>9}{s['full_sharpe']:>9}"
              f"{s['timed_mdd']:+10.1%}{s['full_mdd']:+10.1%}{s['avg_position']:>8.0%}")
    report["sim_hs300"] = sim
    print(f"\n仓位分布：{sim['band_distribution']}")

    # ── 叠到策略上（这才是"值不值得接"的判据）──
    strat, strat_src = None, None
    for f in sorted(OUT.glob("组合AB实验_*.json"), reverse=True):
        try:
            d0 = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:                                  # noqa: BLE001
            _logger.debug("跳过不可读的 AB 实验文件 %s: %s", f.name, e)
            continue
        a0 = (d0.get("arms") or {}).get("A_现状严格Top") or {}
        if a0.get("net_series"):
            strat = pd.Series({pd.Timestamp(k): v
                               for k, v in a0["net_series"].items()}).sort_index()
            strat_src = f.name
            break
    if strat is not None:
        ov = overlay_on_strategy(m, strat)
        print(f"\n=== 叠到策略月净收益上（{strat_src}，{ov.get('months')} 个月）===")
        if not ov.get("ok"):
            print("  ", ov.get("reason"))
        else:
            b = ov["baseline"]
            print(f"{'规则':<22}{'年化':>9}{'夏普':>8}{'回撤':>10}"
                  f"{'Δ年化':>9}{'Δ夏普':>8}{'Δ回撤':>9}{'均仓位':>8}")
            print(f"{'满仓（现状）':<22}{b['ann']:+9.1%}{b['sharpe']:>8}"
                  f"{b['mdd']:+10.1%}{'—':>9}{'—':>8}{'—':>9}{1.0:>8.0%}")
            for tag, m0 in ov["rules"].items():
                print(f"{tag:<22}{m0['ann']:+9.1%}{m0['sharpe']:>8}"
                      f"{m0['mdd']:+10.1%}{m0['delta_ann']:+9.1%}"
                      f"{m0['delta_sharpe']:>8}{m0['delta_mdd']:+9.1%}"
                      f"{m0['avg_position']:>8.0%}")
            print("⚠️ 判据不是\"年化更高\"（纯多头乘仓位必然降收益），"
                  "而是单位风险收益与灾难年少亏是否值得")
        report["overlay_on_strategy"] = ov
        report["strategy_source"] = strat_src

    OUT.mkdir(exist_ok=True)
    p = OUT / f"择时信号检验_{datetime.now():%Y%m%d}.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nJSON: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
