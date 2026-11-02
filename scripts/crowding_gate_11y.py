# -*- coding: utf-8 -*-
"""11 年全市场「拥挤度闸门」终验 —— 直接调用生产函数 `core.crowding`。

为什么必须重跑一遍（不能复用上一轮 11 年结果）
------------------------------------------------
上一轮（`temp/crowd_11y_v2.py`）测的是 `mean(换手率分位, log成交额分位, ret20分位)`
的**全市场分位**。生产上线的是：
  · `crowd_trade = mean(换手率分位, log成交额分位)` —— **不含** ret20（动量不是拥挤）；
  · 三个分位全部改成**行业内**（申万一级 31 类）；
  · 判据改成 `crowd_trade>=0.9 且 mom20>=0.8`。
尺子换了，阈值 0.9/0.8 就是在另一种刻度上标定的 —— 这种"口径改了但阈值照抄"
正是本项目记录过的死路径（PE 分位从 10-22 起一直没生效、两源换手率差 100 倍）。

所以这里**导入生产模块的纯函数**，保证"验证的"就是"上线的"。

口径
----
- 样本：2016-01-04 ~ 2026-10-16（`by_code` 的最新日；`turn_share` 到 10-28，
  但未复权 close 只有 by_code + 覆盖层，覆盖层 `turn` 100% NaN ⇒ 需要两源同时
  有值的日子，故截止 by_code 末日）。
- 窗口：`step=5`（不重叠），`horizon=5`，close→close，等权，未剔涨跌停、未计成本。
- 行业：申万一级（5,220 只，不含北交所）。⚠️ 是**当前**成分快照，属于行业归属上的
  轻度前视（不含价格前视）。这是唯一能同时覆盖 11 年与当日的行业层。
- 分位：`core.crowding.pct_rank` / `industry_pct_rank` / `crowd_trade_of` / `is_gated`。

产物：`output/ui_runs/crowding_gate_11y.json`
"""
import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")

# ⚠️ 项目根**按文件位置推**，不写死绝对路径：写死某个本机目录会让任何克隆到
#    别处的开发者一跑就 ModuleNotFoundError（开源后可复现性的底线要求）。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from core import crowding as C
import logging

_logger = logging.getLogger(__name__)

ROOT = _ROOT
TS_DIR = os.path.join(ROOT, "data/panel/turn_share")
BC_DIR = os.path.join(ROOT, "data/panel/by_code")
OUT = os.path.join(ROOT, "output/ui_runs/crowding_gate_11y.json")
START, END = "2016-01-01", "2026-10-16"
# ⚠️ 宇宙守卫（2026-10-31）：本脚本读的是**原始面板**（绕开 get_bars），
#    2016~2021 的 920xxx 是原新三板历史映射，与连续竞价市场不可比。
#    `--no-guard` 复现旧结论用，**结果不可用于决策**。
GUARD = "--no-guard" not in sys.argv
H, STEP = 5, 5
t0 = time.time()

# ── 1. 行业层（生产同一个函数）──────────────────────────────────────
IND = C.sw_industry_map()
print(f"申万行业 {len(IND)} 只 / {len(set(IND.values())) if IND else 0} 类")

# ── 2. 逐只读量价 + 换手率，算 ret20 / fwd5 ──────────────────────────
cal = pd.read_parquet(os.path.join(TS_DIR, "000001.parquet"), columns=["date"])
cal["date"] = pd.to_datetime(cal["date"])
cal = cal[(cal.date >= START) & (cal.date <= END)].sort_values("date").reset_index(drop=True)
sample_dates = set(cal["date"].iloc[::STEP])
print(f"交易日 {len(cal)}，抽样 {len(sample_dates)} 期（step={STEP}）")

frames = []
skipped = {"no_ts": 0, "no_bc": 0, "short": 0, "empty": 0, "uncomparable": 0}
trimmed_rows = 0
for fn in sorted(os.listdir(TS_DIR)):
    if not fn.endswith(".parquet"):
        continue
    code = fn[:-8]
    bc_path = os.path.join(BC_DIR, fn)
    if not os.path.exists(bc_path):
        skipped["no_bc"] += 1
        continue
    try:
        ts = pd.read_parquet(os.path.join(TS_DIR, fn))
        ts["date"] = pd.to_datetime(ts["date"])
        bc = pd.read_parquet(bc_path, columns=["trade_date", "close", "amount", "is_st"])
        bc["date"] = pd.to_datetime(bc["trade_date"])
    except Exception as e:                             # noqa: BLE001
        _logger.debug("该股时间序列不可用，跳过（已计入 skipped.no_ts）: %s", e)
        skipped["no_ts"] += 1
        continue
    d = bc.merge(ts[["date", "turnover_pct", "float_share"]], on="date", how="inner")
    if GUARD and len(d):
        from core import universe_guard as _ug
        d, _cut = _ug.trim_with_stats(d, code)
        trimmed_rows += _cut
        if len(d) == 0:
            # 不是"数据短"，是这段区间对该票不可比 —— 分开计数
            skipped["uncomparable"] += 1
            continue
    if len(d) < 80:
        skipped["short"] += 1
        continue
    d = d.sort_values("date")
    d["ret20"] = d["close"] / d["close"].shift(C.MOM_LOOKBACK) - 1
    d["fwd5"] = d["close"].shift(-H) / d["close"] - 1
    d["vol20"] = np.log(d["close"]).diff().rolling(C.MOM_LOOKBACK).std()
    d["float_mv"] = pd.to_numeric(d["float_share"], errors="coerce") * d["close"]
    d["code"] = code
    d["is_st"] = pd.to_numeric(d.get("is_st"), errors="coerce")
    d = d[d["date"].isin(sample_dates)]
    d = d[(d["is_st"] != 1) & (d["close"] > 0) & (d["amount"] > 0)
          & (d["turnover_pct"] > 0)]
    if d.empty:
        skipped["empty"] += 1
        continue
    frames.append(d[["date", "code", "turnover_pct", "amount", "ret20", "fwd5",
                     "vol20", "float_mv"]])

df = pd.concat(frames, ignore_index=True)
print(f"读入 {len(df):,} 行 / {df['code'].nunique():,} 只 / {df['date'].nunique()} 期"
      f"  跳过 {skipped}  耗时 {time.time() - t0:.0f}s")
if GUARD:
    print(f"宇宙守卫：剪掉 {trimmed_rows:,} 行不可比历史")

# ── 3. 逐期行业内分位（生产函数）────────────────────────────────────
rows = []
for date, g in df.groupby("date", sort=True):
    turn_raw = dict(zip(g["code"], g["turnover_pct"]))
    amt_raw = {c: float(np.log(max(a, 1e4)))
               for c, a in zip(g["code"], g["amount"])}
    mom_raw = {c: float(r) for c, r in zip(g["code"], g["ret20"])
               if np.isfinite(r)}
    tp, tstats = C.industry_pct_rank(turn_raw, IND)
    ap, _ = C.industry_pct_rank(amt_raw, IND)
    mp, mstats = C.industry_pct_rank(mom_raw, IND)
    rows.append(pd.DataFrame({
        "date": date,
        "code": g["code"].values,
        "fwd5": g["fwd5"].values,
        "amount": g["amount"].values,
        "vol20": g["vol20"].values,
        "float_mv": g["float_mv"].values,
        "crowd_trade": [C.crowd_trade_of(tp.get(c), ap.get(c)) for c in g["code"]],
        "mom20": [mp.get(c) for c in g["code"]],
        "ind_groups": tstats["groups"],
        "ind_pool": tstats["pool"],
        "mom_pool": mstats["pool"],
    }))
X = pd.concat(rows, ignore_index=True)
X = X.dropna(subset=["fwd5", "crowd_trade"])
print(f"有效 {len(X):,} 行 / {X['date'].nunique()} 期"
      f"  行业组中位 {int(X['ind_groups'].median())} 合并池中位 {int(X['ind_pool'].median())}"
      f"  耗时 {time.time() - t0:.0f}s")

# 同日等权基准（超额 = 相对当日全样本均值）
X["mkt"] = X.groupby("date")["fwd5"].transform("mean")
X["exc"] = (X["fwd5"] - X["mkt"]) * 100          # pp

# ── 4. 十分位（拥挤度水平单独看）────────────────────────────────────
X["dec"] = X.groupby("date")["crowd_trade"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
deciles = []
for d, g in X.groupby("dec"):
    deciles.append({"decile": int(d) + 1, "n": int(len(g)),
                    "crowd_mean": round(float(g["crowd_trade"].mean()), 3),
                    "exc_pp": round(float(g["exc"].mean()), 4)})
D10, D1 = deciles[-1]["exc_pp"], deciles[0]["exc_pp"]
print("\n十分位（拥挤度）→ 未来5日超额 pp")
for r in deciles:
    print(f"  D{r['decile']:<2d} crowd={r['crowd_mean']:.3f}  {r['exc_pp']:+.3f}")

# ── 5. 分年（D10−D1）────────────────────────────────────────────────
X["year"] = X["date"].dt.year
per_year = []
for y, g in X.groupby("year"):
    q = g.groupby("dec")["exc"].mean()
    per_year.append({"year": int(y), "d10_minus_d1_pp": round(float(q.iloc[-1] - q.iloc[0]), 4),
                     "d1_pp": round(float(q.iloc[0]), 4),
                     "d10_pp": round(float(q.iloc[-1]), 4)})
neg_years = sum(1 for r in per_year if r["d10_minus_d1_pp"] < 0)
print(f"\n分年 D10−D1：{neg_years}/{len(per_year)} 年为负")
for r in per_year:
    print(f"  {r['year']}  {r['d10_minus_d1_pp']:+.3f}pp")

# ── 6. 闸门本身（这才是要上线的判据）───────────────────────────────
X["gated"] = [C.is_gated(a, b) for a, b in zip(X["crowd_trade"], X["mom20"])]
g_on, g_off = X[X["gated"]], X[~X["gated"]]
gate_spread = float(g_on["exc"].mean() - g_off["exc"].mean())

# 逐期 spread 序列（Neyman 口径：每天各自的命中组 vs 未命中组）
per_date = (X[X["gated"]].groupby("date")["exc"].mean()
            - X[~X["gated"]].groupby("date")["exc"].mean()).dropna()
mu, sd, n = float(per_date.mean()), float(per_date.std(ddof=1)), len(per_date)
t_plain = mu / (sd / np.sqrt(n)) if n > 1 and sd > 0 else float("nan")


def newey_west(series: pd.Series, lag: int = 3) -> float:
    """NW t 值（重叠窗口下 t 会虚高，用 NW 校正）。"""
    x = series.dropna().to_numpy(dtype=float)
    n_ = len(x)
    if n_ < lag + 2:
        return float("nan")
    e = x - x.mean()
    g0 = float(np.dot(e, e) / n_)
    var = g0
    for L in range(1, lag + 1):
        gl = float(np.dot(e[L:], e[:-L]) / n_)
        var += 2 * (1 - L / (lag + 1)) * gl
    se = np.sqrt(max(var, 0) / n_)
    return float(x.mean() / se) if se > 0 else float("nan")


t_nw = newey_west(per_date, lag=3)
gate_by_year = []
for y, g in X.groupby("year"):
    on, off = g[g["gated"]], g[~g["gated"]]
    if len(on) < 30 or len(off) < 30:
        continue
    gate_by_year.append({"year": int(y), "n_gated": int(len(on)),
                         "spread_pp": round(float(on["exc"].mean() - off["exc"].mean()), 4)})
gate_neg = sum(1 for r in gate_by_year if r["spread_pp"] < 0)

n_days = X["date"].nunique()
print(f"\n闸门（crowd≥{C.GATE_CROWD_PCT} 且 mom20≥{C.GATE_MOM_PCT}）")
print(f"  命中 {len(g_on):,} 行 / {len(X):,}（{len(g_on)/len(X):.2%}），"
      f"日均 {len(g_on)/n_days:.1f} 只 / {X.groupby('date').size().mean():.0f} 只")
print(f"  命中组超额 {g_on['exc'].mean():+.3f}pp  未命中组 {g_off['exc'].mean():+.3f}pp"
      f"  差 {gate_spread:+.3f}pp")
print(f"  逐期 spread {mu:+.3f}pp  t={t_plain:.2f}  NW(3) t={t_nw:.2f}  n={n} 期")
print(f"  分年同向：{gate_neg}/{len(gate_by_year)} 年为负")
for r in gate_by_year:
    print(f"    {r['year']}  {r['spread_pp']:+.3f}pp (命中 {r['n_gated']})")

# ── 7. 阈值敏感性（0.9/0.8 是不是被挑出来的最好值）──────────────────
sens = []
for cp in (0.80, 0.85, 0.90, 0.95):
    for mp_ in (0.60, 0.70, 0.80, 0.90):
        m = [C.is_gated(a, b, crowd_pct=cp, mom_pct=mp_)
             for a, b in zip(X["crowd_trade"], X["mom20"])]
        on = X.loc[m, "exc"]
        if len(on) < 200:
            continue
        sens.append({"crowd_pct": cp, "mom_pct": mp_, "n": int(len(on)),
                     "exc_pp": round(float(on.mean() - X.loc[~np.array(m), "exc"].mean()), 4)})
print("\n阈值敏感性（命中组 − 未命中组, pp）")
for r in sens:
    print(f"  crowd≥{r['crowd_pct']:.2f} mom≥{r['mom_pct']:.2f}  "
          f"{r['exc_pp']:+.3f}pp  命中 {r['n']:,}")

# ── 6.5 稳健性：排除"闸门只是选到了小盘股 / 高波股"这个解释 ──────────
# 这是本项目记录的硬纪律：拥挤度含换手率与成交额，两者都与市值、波动率高度相关
# ⇒ 不证明"剔掉它们之后还在"，结论就会是 IVOL / 小盘因子的另一种写法。
def _spread(sub: pd.DataFrame) -> dict:
    """子样本内：命中组 − 未命中组（pp）+ NW t。样本不足返回 None。"""
    on, off = sub[sub["gated"]], sub[~sub["gated"]]
    if len(on) < 200 or len(off) < 200:
        return None
    ser = (sub[sub["gated"]].groupby("date")["exc"].mean()
           - sub[~sub["gated"]].groupby("date")["exc"].mean()).dropna()
    return {"n_gated": int(len(on)),
            "spread_pp": round(float(on["exc"].mean() - off["exc"].mean()), 4),
            "t_nw3": round(newey_west(ser, lag=3), 2),
            "neg_years": int(sum(
                1 for _y, g in sub.groupby("year")
                if len(g[g["gated"]]) >= 30 and len(g[~g["gated"]]) >= 30
                and g[g["gated"]]["exc"].mean() - g[~g["gated"]]["exc"].mean() < 0)),
            "years": int(sum(
                1 for _y, g in sub.groupby("year")
                if len(g[g["gated"]]) >= 30 and len(g[~g["gated"]]) >= 30))}


# (a) 生产顺序复现：先过流动性下限（成交额 ≥ 自身板块 30% 分位），再判闸门
from core.sector import classify_by_board                                # noqa: E402

X["board"] = [classify_by_board(c) for c in X["code"]]


def _liq_keep(g: pd.DataFrame) -> pd.Series:
    """当日按板块 30% 分位门槛（与 `StockScreener._liquidity_threshold` 同口径）。"""
    outs = []
    for _b, gb in g.groupby("board"):
        if len(gb) < 100:                      # 板块样本太小 ⇒ 用全市场阈值
            outs.append(gb["amount"] >= g["amount"].quantile(0.30))
        else:
            outs.append(gb["amount"] >= gb["amount"].quantile(0.30))
    return pd.concat(outs).reindex(g.index).fillna(True)


X["liq_ok"] = X.groupby("date", group_keys=False).apply(_liq_keep)
robust = {}
robust["after_liquidity_floor"] = _spread(X[X["liq_ok"]])

# (b) 市值三分层内（同日截面）
for name, col in (("size", "float_mv"), ("vol", "vol20")):
    buckets = []
    for lvl in (0, 1, 2):
        sub = X[X[col].notna()].copy()
        sub["_b"] = sub.groupby("date")[col].transform(
            lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
        buckets.append({"level": lvl, "spread": _spread(sub[sub["_b"] == lvl])})
    robust[f"by_{name}_tercile"] = buckets

print("\n稳健性：闸门效应是不是小盘/高波的马甲")
r0 = robust["after_liquidity_floor"]
print(f"  先过流动性下限（板块30%分位）后：{r0['spread_pp']:+.3f}pp "
      f"NW(3) t={r0['t_nw3']}  命中 {r0['n_gated']:,}  分年 {r0['neg_years']}/{r0['years']} 负")
for name in ("size", "vol"):
    label = "市值" if name == "size" else "波动率"
    for b in robust[f"by_{name}_tercile"]:
        s = b["spread"]
        tag = ("小" if b["level"] == 0 else "中" if b["level"] == 1 else "大") \
            if name == "size" else f"{b['level'] + 1}/3"
        print(f"  {label} {tag:>3s}：{s['spread_pp']:+.3f}pp NW(3) t={s['t_nw3']} "
              f"分年 {s['neg_years']}/{s['years']} 负" if s else f"  {label} {tag}：样本不足")

res = {
    "window": {"start": START, "end": END, "step": STEP, "horizon": H,
               "basis": "等权 close→close，未复权，未剔涨跌停，未计成本"},
    "rows": int(len(X)), "codes": int(X["code"].nunique()), "periods": int(n_days),
    "industry": {"source": "申万一级", "codes": len(IND),
                 "groups_median": int(X["ind_groups"].median()),
                 "pool_median": int(X["ind_pool"].median()),
                 "caveat": "当前成分为快照，行业归属轻度前视（无价格前视）"},
    "deciles_crowd": deciles,
    "spread_d10_d1_pp": round(D10 - D1, 4),
    "per_year": per_year, "neg_years": neg_years,
    "gate": {"crowd_pct": C.GATE_CROWD_PCT, "mom_pct": C.GATE_MOM_PCT,
             "n_gated": int(len(g_on)), "share": round(len(g_on) / len(X), 4),
             "per_day": round(len(g_on) / n_days, 1),
             "exc_gated_pp": round(float(g_on["exc"].mean()), 4),
             "exc_ungated_pp": round(float(g_off["exc"].mean()), 4),
             "spread_pp": round(gate_spread, 4),
             "per_period_spread_pp": round(mu, 4),
             "t_plain": round(t_plain, 2), "t_newey_west3": round(t_nw, 2),
             "per_year": gate_by_year, "neg_years": gate_neg,
             "years_tested": len(gate_by_year)},
    "universe_guard": {"enabled": bool(GUARD), "trimmed_rows": int(trimmed_rows)},
    "sensitivity": sens,
    "robust": robust,
    "elapsed_s": round(time.time() - t0, 1),
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"\n落盘 {OUT}  总耗时 {res['elapsed_s']}s")
