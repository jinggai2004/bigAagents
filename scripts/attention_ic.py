# -*- coding: utf-8 -*-
"""关注度因子 IC 检验（复用 factor_analysis 的 build_panel 与前视防护）

设计要点：
- **复用** `core.factor_analysis.build_panel`：它对未来收益的处理是"该票自己的第 horizon 根"，
  且 `_pos_on_or_before` 是唯一的前视出口 ⇒ 不另造一套（本项目铁律：单一实现）。
- 关注度列由 `data.attention` 按日期 merge 进 bars，再当作普通因子列交给 build_panel。
- **自校验**：同时也跑项目现有的 4 个因子，与已公布结论对照
  （波动率 IC −0.0913 / 动量 −0.1292），以证明本脚本的 IC 口径与既有报告一致。
- ⚠️ 关注度原始 `rank` 是"绝对值"（越小越热），必须做**同日横截面分位**才有可比性。
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
# 直接 `python scripts/xxx.py` 时，Python 只把 `scripts/` 放进 sys.path，
# 项目根不在里面 → `import core` 会失败。显式补上，与其它 scripts 一致。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core import factor_analysis as FA
from data import attention as AT

POOL = json.load(open("data/backtest_pool.json", encoding="utf-8"))["pool"]
# ⚠️ 支持限量：**底层取数源没有超时**，某只票卡住会让整脚本无限挂起
#    （实测曾跑 9 小时不返回）。小样本先跑通、可观测，再放大。
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if LIMIT > 0:
    POOL = POOL[:LIMIT]
START, END = "2025-02-01", "2026-06-30"
HORIZON, STEP = 5, 5

t0 = time.time()
asked = len(POOL)
print(f"池 {asked} 只；区间 {START}~{END}；horizon={HORIZON} 日，每 {STEP} 日采样", flush=True)

print("[1/4] 取关注度（东财人气榜逐日历史）…", flush=True)
_t = [time.time()]
def _attn_cb(ev):
    if ev["current"] % 10 == 0 or ev["current"] == ev["total"]:
        print(f"    关注度 {ev['current']}/{ev['total']} {ev['code']}"
              f"  累计 {time.time() - _t[0]:.0f}s", flush=True)
attn = AT.load_map(POOL, START, END, progress_cb=_attn_cb)
print(f"  → 取到 {len(attn)}/{asked} 只；{AT.coverage_note(len(attn), asked)}", flush=True)

print("[2/4] 取日线并派生因子…", flush=True)
_done = [0]
def _bar_cb(ev):
    _done[0] = ev.get("current", 0)
    if _done[0] % 10 == 0 or _done[0] == ev.get("total"):
        print(f"    日线 {ev.get('current')}/{ev.get('total')}"
              f"  累计 {time.time() - _t[0]:.0f}s", flush=True)
bars_map = FA.load_bars(POOL, START, END, progress_cb=_bar_cb)
print(f"  → {len(bars_map)} 只日线", flush=True)

print("[3/4] 把关注度按日期 merge 进 bars…", flush=True)
merged = 0
for code, b in bars_map.items():
    a = attn.get(code)
    if a is None or a.empty:
        b["attn_rank"] = np.nan
        b["attn_new_fans"] = np.nan
        continue
    ad = a.rename(columns={"date": "date"})[["date", "rank", "new_fans"]]
    ad["date"] = pd.to_datetime(ad["date"])
    b["date"] = pd.to_datetime(b["date"])
    m = b[["date"]].merge(ad, on="date", how="left")
    b["attn_rank"] = m["rank"].values
    b["attn_new_fans"] = m["new_fans"].values
    if b["attn_rank"].notna().any():
        merged += 1
print(f"  → {merged}/{len(bars_map)} 只成功对齐到关注度", flush=True)

print("[4/4] 构造面板并算 IC…", flush=True)
dates = FA._sample_dates(START, END, STEP)
factors = ["momentum20", "volatility", "ma_bullish", "ma20_above",
           "attn_hot_pct", "attn_new_fans"]
panel = FA.build_panel(bars_map, factors, dates, HORIZON)
print(f"  面板 {len(panel)} 行（{panel.date.nunique()} 个采样日 × {panel.code.nunique()} 只）", flush=True)

# ⚠️⚠️ **方向口径必须写清，否则结论会整段反掉**：
#   `attn_rank` 是东财人气**排名**，**数值越小越热**（1 = 最热）。
#   首版写成 `rank(pct=True)` —— 它把 **1.0 赋给最大值**，
#   即"分位高 = 排名数字大 = **最冷**"，而标签却写成"越大越热" ⇒ **分层的最热/最冷两档被标反**。
#   现改为升序=假：热度分位 **越大越热**（pct=1 给 rank 最小的那只）。
panel["attn_hot_pct"] = panel.groupby("date")["attn_rank"].rank(pct=True, ascending=False)


def ic_stats(col: str, need_min: int = 8) -> dict:
    """逐日 Spearman IC → 均值 / 标准差 / t 值 / IR / 同号占比"""
    sub = panel[["date", col, "fwd_ret"]].dropna()
    ics, ns = [], []
    for d, g in sub.groupby("date"):
        if len(g) < need_min:
            continue
        ic = g[col].corr(g["fwd_ret"], method="spearman")
        if ic == ic:
            ics.append(ic)
            ns.append(len(g))
    if len(ics) < 5:
        return {"n_periods": len(ics), "insufficient": True}
    arr = np.array(ics)
    mean, sd = float(arr.mean()), float(arr.std(ddof=1))
    t = mean / (sd / np.sqrt(len(arr))) if sd > 0 else float("nan")
    return {"n_periods": len(arr), "mean_ic": round(mean, 4), "std_ic": round(sd, 4),
            "t_stat": round(t, 2), "ic_ir": round(mean / sd, 3) if sd else None,
            "pos_ratio": round(float((arr > 0).mean()), 3),
            "avg_names": int(np.mean(ns)),
            "verdict": ("显著" if abs(t) >= 2 else "不显著")}


FACTOR_LABELS = {
    "momentum20": "20日动量", "volatility": "年化波动率",
    "ma_bullish": "均线多头排列", "ma20_above": "站上MA20",
    "attn_hot_pct": "关注度·热度分位(越大越热)",
    "attn_new_fans": "关注度·新晋粉丝占比",
}

print()
print("=" * 78)
print(f"IC 检验（horizon={HORIZON} 日，每 {STEP} 日采样）")
print("=" * 78)
print(f"{'因子':30s} {'期数':>4s} {'均值IC':>8s} {'t值':>7s} {'IR':>7s} {'同号':>6s}  判定")
rows = {}
for f in factors:
    if f not in panel.columns:
        continue
    st = ic_stats(f)
    rows[f] = st
    if st.get("insufficient"):
        print(f"{FACTOR_LABELS.get(f, f):30s} {st['n_periods']:>4d} —— 期数不足")
        continue
    print(f"{FACTOR_LABELS.get(f, f):30s} {st['n_periods']:>4d} {st['mean_ic']:>8.4f} "
          f"{st['t_stat']:>7.2f} {st['ic_ir']:>7.3f} {st['pos_ratio']:>6.2f}  {st['verdict']}")

# 分层：关注度分位 5 档 → 未来收益均值
print()
print("=" * 78)
print("分层检验：关注度·排名分位 5 档 → 未来收益（%）")
print("=" * 78)
sub = panel[["date", "attn_hot_pct", "fwd_ret"]].dropna()
if len(sub):
    sub = sub.copy()
    sub["q"] = sub.groupby("date")["attn_hot_pct"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop"))
    g = sub.groupby("q")["fwd_ret"].agg(["mean", "count"])
    for q, r in g.iterrows():
        tag = "← 最热" if q == 4 else ("← 最冷" if q == 0 else "")
        print(f"  Q{int(q) + 1}（{'最冷' if q == 0 else ('最热' if q == 4 else '中')}）"
              f"  平均收益 {r['mean'] * 100:+.3f}%  N={int(r['count'])}  {tag}")
    if 4 in g.index and 0 in g.index:
        print(f"  → 最热 − 最冷 = {(g.loc[4, 'mean'] - g.loc[0, 'mean']) * 100:+.3f}%")

out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
       "pool_size": asked, "attn_covered": len(attn),
       "range": [START, END], "horizon": HORIZON, "step": STEP,
       "panel_rows": int(len(panel)), "ic": rows,
       "coverage_note": AT.coverage_note(len(attn), asked)}
with open("output/ui_runs/attention_ic.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\n已保存 output/ui_runs/attention_ic.json，用时 {time.time() - t0:.0f}s")
