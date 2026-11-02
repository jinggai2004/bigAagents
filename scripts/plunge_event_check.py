"""「突然大跌、但事后财报很好」——这类股票该不该进关注列表？

## 用户的具体场景

> 燕山啤酒出财报前跌停，但财报很好看，是否可以对此类突然大跌而并无明显原因的
> 股票列到关注列表中？

这跟"月度跌幅分位"是**两件事**：前者是慢变量分位（混杂了趋势、行业、流动性），
后者是**离散事件**（某一天突然砸下来）。事件研究能把它拆干净。

## 关键的诚实之处

大跌当天你**不可能知道财报好不好**。所以必须分两层看：

1. **可操作层**：所有"突然大跌"事件之后的 T+1/T+5/T+20 表现——
   这是你当天就能拿到的全部信息，决定"要不要放进关注列表"。
2. **验证层**：把事件按"随后 45 天内披露的财报好坏"分组。
   如果好财报组明显更好、差财报组明显更差，说明**"等财报"这个动作有信息价值**，
   那关注列表的正确用法就是"先挂着，等财报再定" —— 这正是用户的思路。
   如果两层没差异，说明财报信息已经被价格吸收完了，挂着也没意义。

⚠️ 验证层**不能**当成可交易策略回测（它是事后分组）。它只回答"该不该等"。

## 方法要点

- 事件定义：单日跌幅 ≤ 阈值（默认 −7%），同一只票在 `cooldown` 个交易日内只记一次
  （否则连续跌停会被算成多个独立事件，样本严重重复计数）。
- 剔除：停牌、当日无成交（`trade_status`）、ST（跌停幅度不同，混在一起口径不齐）。
- 超额 = 事件股的前瞻收益 − **同一交易日**全市场等权的前瞻收益（同 horizon、同日基准，
  否则会把市场 beta 当成 alpha）。
- 财报关联：用 `financial_pit` 的 `pub_date`（首次公告日，PIT 锚点），
  只认事件后 45 天内**首次**披露的那一份。
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUT = _ROOT / "output"

PLUNGE_THR = -0.07        # 单日跌幅阈值（"突然大跌"）
COOLDOWN = 5              # 同一只票的冷却交易日（防连续跌停重复计数）
HORIZONS = (1, 5, 20)
FIN_WINDOW = 45           # 事件后多少自然日内披露的财报算"这次大跌的验证"
MIN_EVENTS = 100          # 分组最少事件数
GOOD_YOY = 0.20           # 净利同比 >20% 视为"财报很好"
BAD_YOY = 0.0             # 净利同比 <0 视为"财报差"


def load_daily(days: int = 1500) -> pd.DataFrame:
    """全市场日频（列子集，float32 省内存）。

    复用 `core.zt_calendar._load_recent`（它已经处理了 parts 两种布局 + overlay 增量
    的 dedup 顺序，自己再写一遍必然走样）。
    """
    from core.zt_calendar import _load_recent
    # ⚠️⚠️ 必须 `use_cache=False`：`_load_recent` 的缓存只判"最新交易日距今 ≤3 天"
    # 与 overlay 是否更新，**不区分请求的 `days`**。用缓存会拿到之前按 40 天建的
    # 那份（实测只返回 24 个交易日），于是 fwd20 全落在样本尾部、事件样本严重有偏，
    # 而输出看起来完全正常（表照打、t 值照算）。这正是"缓存键必须带作用域"那道坑。
    d = _load_recent(days, use_cache=False)
    if d is None or d.empty:
        raise SystemExit("日频面板不可用（parts/overlay 缺失）")
    d = d.copy()
    d["stock_code"] = d["stock_code"].astype(str).str.zfill(6)
    d["trade_date"] = d["trade_date"].astype(str).str[:10]
    for c in ("close", "preclose"):
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
    # 只保留正常交易的记录：停牌/无收盘价会把收益率算成 0% 或 inf
    if "trade_status" in d.columns:
        ts = pd.to_numeric(d["trade_status"], errors="coerce")
        d = d[ts.isna() | (ts == 1)]
    d = d[d["close"] > 0]
    return d.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)


def add_returns(d: pd.DataFrame) -> pd.DataFrame:
    """当日涨跌幅 + 各 horizon 的前瞻收益（按**交易日**数，不是自然日）。"""
    d = d.copy()
    d["chg"] = d["close"] / d["preclose"] - 1.0
    g = d.groupby("stock_code", sort=False)["close"]
    for h in HORIZONS:
        fut = g.shift(-h)
        d[f"fwd{h}"] = (fut / d["close"] - 1.0).astype("float32")
    return d


def find_plunges(d: pd.DataFrame, thr: float = PLUNGE_THR,
                 cooldown: int = COOLDOWN) -> pd.DataFrame:
    """找突然大跌事件（同一只票 cooldown 交易日内只留第一次）。"""
    ev = d[d["chg"] <= thr].copy()
    if ev.empty:
        return ev
    ev = ev.sort_values(["stock_code", "trade_date"])
    dates = pd.to_datetime(ev["trade_date"])
    # 同一只票：与上一个事件间隔不足 cooldown 个自然日就丢弃（交易日≈自然日×1.4，
    # 这里用宽松阈值只为去掉"连续跌停"这一个明显重复源，不追求精确交易日差）
    gap = dates.groupby(ev["stock_code"]).diff().dt.days
    ev = ev[(gap.isna()) | (gap >= int(cooldown * 1.4))]
    return ev.reset_index(drop=True)


def attach_benchmark(ev: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """同日全市场等权前瞻收益（同 horizon 同日基准，否则市场 beta 会被当成 alpha）。"""
    ev = ev.copy()
    for h in HORIZONS:
        bench = d.groupby("trade_date")[f"fwd{h}"].mean()
        ev[f"bench{h}"] = ev["trade_date"].map(bench).astype("float32")
        ev[f"ex{h}"] = ev[f"fwd{h}"] - ev[f"bench{h}"]
    return ev


def attach_financials(ev: pd.DataFrame, window: int = FIN_WINDOW) -> pd.DataFrame:
    """关联"事件后 `window` 天内首次披露"的财报，标记好坏。

    用 `pub_date`（首次公告日）作 PIT 锚点：只用事件之后才公开的报表，
    事件当时还看不到的报表**不能**用来给这次大跌定性（那是前视）。
    """
    from core.financial_pit import load_financials
    fin = load_financials()
    if fin.empty:
        ev["fin_yoy"] = np.nan
        ev["fin_lag_days"] = np.nan
        return ev

    fin = fin.copy()
    fin["code"] = fin["code"].astype(str).str.zfill(6)
    fin["pub"] = pd.to_datetime(fin["pub_date"], errors="coerce")
    fin = fin.dropna(subset=["pub"])
    # 净利同比（东财 sjltz 优先，缺失用 profit_yoy）
    yoy = fin.get("sjltz")
    if yoy is None:
        yoy = fin.get("profit_yoy")
    fin["yoy"] = pd.to_numeric(yoy, errors="coerce") if yoy is not None else np.nan

    by_code = {c: g.sort_values("pub") for c, g in fin.groupby("code")}
    out_yoy, out_lag, out_period, out_prev = [], [], [], []
    for code, day in zip(ev["stock_code"], ev["trade_date"]):
        g = by_code.get(code)
        t0 = pd.Timestamp(day)
        # 事件**之前**最近一次公告距今多少天：用来区分"财报前大跌"与"财报后大跌"。
        # 这两件事性质完全不同 —— 前者是信息不对称/恐慌（可能有错杀），
        # 后者是市场对已知财报的反应（通常继续消化利空）。混在一起测会把两者抵消。
        prev_gap = np.nan
        if g is not None:
            past = g[g["pub"] <= t0]
            if not past.empty:
                prev_gap = float((t0 - past["pub"].iloc[-1]).days)
        out_prev.append(prev_gap)
        if g is None:
            out_yoy.append(np.nan)
            out_lag.append(np.nan)
            out_period.append(None)
            continue
        cand = g[(g["pub"] > t0) & (g["pub"] <= t0 + pd.Timedelta(days=window))]
        if cand.empty:
            out_yoy.append(np.nan)
            out_lag.append(np.nan)
            out_period.append(None)
            continue
        row = cand.iloc[0]
        out_yoy.append(float(row["yoy"]) if pd.notna(row["yoy"]) else np.nan)
        out_lag.append(int((row["pub"] - t0).days))
        out_period.append(str(row.get("stat_date", ""))[:10])
    ev = ev.copy()
    ev["fin_yoy"] = out_yoy
    ev["fin_lag_days"] = out_lag
    ev["fin_period"] = out_period
    ev["prev_pub_gap"] = out_prev      # 事件前最近公告距今（天）；NaN=从未公告
    return ev


def _stats(x: pd.Series) -> dict:
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 2:
        return {"n": n, "mean_pp": None, "t": None, "win": None}
    sd = float(x.std(ddof=1))
    # ⚠️ 不能写 `sd > 0`：常数序列的 std 会有 1e-18 量级的浮点残差，
    #    除下去得到 1e16 这种"天文 t 值"，看着像极端显著其实是零方差。
    return {"n": n,
            "mean_pp": round(float(x.mean()) * 100, 3),
            "t": (round(float(x.mean() / sd * math.sqrt(n)), 2)
                  if sd > 1e-12 else None),
            "win": round(float((x > 0).mean()), 3)}


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    days = 1500
    for i, a in enumerate(argv):
        if a == "--days" and i + 1 < len(argv):
            days = int(argv[i + 1])

    print("加载日频面板…")
    d = add_returns(load_daily(days))
    tr = d["trade_date"]
    print(f"  {len(d):,} 行 / {d['stock_code'].nunique():,} 只 / "
          f"{tr.nunique()} 交易日 / {tr.min()} ~ {tr.max()}")

    ev = attach_benchmark(find_plunges(d), d)
    print(f"\n突然大跌事件（单日 ≤ {PLUNGE_THR:.0%}，冷却 {COOLDOWN} 日）：{len(ev):,} 个")

    report = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "rows": int(len(d)), "trading_days": int(tr.nunique()),
              "period": [str(tr.min()), str(tr.max())],
              "threshold": PLUNGE_THR, "cooldown": COOLDOWN,
              "events": int(len(ev))}
    if ev.empty:
        print("无事件")
        return 1

    # ── 第一层：可操作（大跌当天能拿到的全部信息）──
    print(f"\n=== 第一层：所有大跌事件之后（n={len(ev):,}）===")
    print(f"{'horizon':<9}{'事件股':>10}{'同日基准':>10}{'超额pp':>9}{'t':>7}{'胜率':>7}")
    layer1 = {}
    for h in HORIZONS:
        e, b = _stats(ev[f"fwd{h}"]), _stats(ev[f"bench{h}"])
        ex = _stats(ev[f"ex{h}"])
        layer1[h] = {"event": e, "bench": b, "excess": ex}
        print(f"T+{h:<7}{e['mean_pp']:+9.2f}%{b['mean_pp']:+9.2f}%"
              f"{ex['mean_pp']:+9.2f}{str(ex['t']):>7}{ex['win']*100:6.0f}%")
    report["layer1_all_events"] = layer1

    # ── 第二层：按"事件后 45 天内披露的财报"分组 ──
    print("\n关联事件后 45 天内首次披露的财报…")
    ev = attach_financials(ev)
    has_fin = ev["fin_yoy"].notna()
    print(f"  有后续财报的事件：{int(has_fin.sum()):,} / {len(ev):,} "
          f"({has_fin.mean():.0%})，披露滞后中位 "
          f"{ev.loc[has_fin, 'fin_lag_days'].median():.0f} 天")

    groups = {
        "财报很好(净利同比>20%)": ev[has_fin & (ev["fin_yoy"] > GOOD_YOY)],
        "财报一般(0~20%)": ev[has_fin & (ev["fin_yoy"] >= BAD_YOY)
                             & (ev["fin_yoy"] <= GOOD_YOY)],
        "财报差(净利同比<0)": ev[has_fin & (ev["fin_yoy"] < BAD_YOY)],
        "无后续财报": ev[~has_fin],
    }
    print(f"\n=== 第二层：按随后财报好坏分组（这是**事后**分组，只回答\"该不该等\"）===")
    print(f"{'分组':<24}{'事件数':>7}"
          f"{'T+1超额':>10}{'T+5超额':>10}{'T+20超额':>10}{'T+20 t':>8}")
    layer2 = {}
    for name, sub in groups.items():
        if len(sub) < MIN_EVENTS:
            print(f"{name:<24}{len(sub):>7}   样本不足")
            continue
        row = {"n": int(len(sub))}
        cells = []
        for h in HORIZONS:
            st = _stats(sub[f"ex{h}"])
            row[f"ex{h}"] = st
            cells.append(f"{st['mean_pp']:+9.2f}" if st["mean_pp"] is not None else "     —")
        layer2[name] = row
        t20 = row["ex20"]["t"]
        print(f"{name:<24}{len(sub):>7}" + "".join(f"{c:>10}" for c in cells)
              + f"{str(t20):>8}")
    report["layer2_by_financials"] = layer2

    # 关键对比：好财报组 vs 差财报组 的差
    good = groups["财报很好(净利同比>20%)"]
    bad = groups["财报差(净利同比<0)"]
    if len(good) >= MIN_EVENTS and len(bad) >= MIN_EVENTS:
        diff = (good["ex20"].reset_index(drop=True)
                - bad["ex20"].reset_index(drop=True))
        st = _stats(diff)
        # 两组独立样本，月度不对齐 ⇒ 用 Welch 近似（这里按事件配对无意义，取均值差的 t）
        se = math.sqrt((good["ex20"].var(ddof=1) / len(good))
                       + (bad["ex20"].var(ddof=1) / len(bad)))
        welch_t = (good["ex20"].mean() - bad["ex20"].mean()) / se if se > 0 else None
        print(f"\n好财报组 − 差财报组（T+20 超额）："
              f"{(good['ex20'].mean() - bad['ex20'].mean())*100:+.2f}pp  "
              f"Welch t={round(float(welch_t), 2) if welch_t else None}")
        report["good_minus_bad"] = {
            "pp": round(float((good["ex20"].mean() - bad["ex20"].mean()) * 100), 3),
            "welch_t": round(float(welch_t), 2) if welch_t else None,
            "good_n": int(len(good)), "bad_n": int(len(bad)),
        }

    # 顺带：事件当天跌幅分档（跌得越狠，反弹越强？）
    print(f"\n=== 跌幅分档（T+20 超额）===")
    tiers = [(-0.095, -0.07, "−7%~−9.5%"), (-0.15, -0.095, "−9.5%~−15%"),
             (-1.0, -0.15, "−15% 以上")]
    tier_out = {}
    for lo, hi, label in tiers:
        sub = ev[(ev["chg"] > lo) & (ev["chg"] <= hi)]
        st = _stats(sub["ex20"])
        tier_out[label] = {"n": st["n"], **st}
        print(f"{label:<16}{st['n']:>7}  T+20 超额 {st['mean_pp']:+.2f}pp  t={st['t']}")
    report["by_drop_tier"] = tier_out

    OUT.mkdir(exist_ok=True)
    p = OUT / f"大跌事件检验_{datetime.now():%Y%m%d}.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nJSON: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
