"""「基本面好 + 跌得多 ⇒ 值得买」是否成立？

## 用户的问题

> 找此前基本面好但下跌幅度较大的股票，找买入时机可以吗？

这是"好公司被错杀"的经典思路（quality × mean-reversion）。它在本项目里有两个**方向互补**
的已知证据，所以值得认真测：

- `ocf_to_np`（盈利质量）IC **+0.015 / t=+3.5 / 分年 7/8 正**（唯一过多重检验的基本面因子，正 IC）
- 量价 IC **全负**（11 年 ivol20 −0.083 / max20 −0.074，均 11/11 年）⇒ 反转方向有效

组合起来看起来就该是"质量高 + 跌得多"。

## 为什么必须做交互检验

只看"跌得多的股票之后收益高"是**纯反转**，跟基本面没关系——量价 IC 全负早就说明这件事了。
要证明"质量"起作用，必须看四格：

|  | 跌得多 | 跌得少 |
|---|---|---|
| **质量高** | A（想要的） | B |
| **质量低** | C | D |

判据是**交互项** `(A−B) − (C−D)` 显著为正：质量把跌幅的价值放大了。
若 A 好但 C 也一样好 ⇒ 那是纯反转，质量是装饰。

⚠️ 另外两个陷阱：
1. 反转效应在**小盘/低流动性**端更强 ⇒ 必须同时报"控制流动性后还在不在"。
2. `fwd20` 是面板口径（月末→下月末），不含交易成本与换手假设，**不等于**组合收益。
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

MIN_STOCKS = 100          # 单月有效样本下限
MIN_CELL = 8              # 单格下限（太少不出结论）
MIN_MONTHS = 24

# 质量口径（越大越好）
QUALITY = {
    # 水平类（"一直很好"）
    "ocf_to_np": "经营现金流/净利润（盈利质量，已验证正 IC）",
    "roe": "ROE（加权，小数）",
    "np_margin": "净利率",
    # 变化类（"正在变好"）——本项目已知 ROE **水平**无效而 roe_yoy 有效（IC +0.017/t=2.1），
    # 所以"错杀"的准确口径应该是"股价跌了但基本面在改善"，不是"跌了但一直很好"。
    "roe_yoy": "ROE 同比改善（差值）",
    "sjltz": "净利同比（东财直给）",
}
# 跌幅口径（脚本内部统一取负号 ⇒ 值越大表示跌得越多）
DROP = {
    "drop1m": ("mom1m", "近 1 月跌幅"),
    "drop6m": ("mom6m", "近 6 月跌幅"),
    "lowpos": ("boll_pos", "布林带位置低（跌到区间下沿）"),
}


def build(panel: pd.DataFrame) -> pd.DataFrame:
    """补出跌幅列（统一成"越大越跌"）。"""
    d = panel.copy()
    for name, (src, _desc) in DROP.items():
        if src not in d.columns:
            d[name] = np.nan
        else:
            d[name] = -pd.to_numeric(d[src], errors="coerce")
    return d


def monthly_cells(panel: pd.DataFrame, qcol: str, dcol: str,
                  q_thr: float = 0.5, d_thr: float = 0.7) -> pd.DataFrame:
    """逐月切四格，返回每格的月均 `fwd20`（另带样本数与全市场基准）。"""
    rows = []
    for m, g in panel.groupby("month_ts", sort=True):
        g = g.dropna(subset=[qcol, dcol, "fwd20"])
        if len(g) < MIN_STOCKS:
            continue
        q = g[qcol].rank(pct=True)
        d = g[dcol].rank(pct=True)        # d 已是"跌幅"，分位高 = 跌得多
        hi_q, big_d = q > q_thr, d > d_thr
        cells = {"QH_D": hi_q & big_d, "QH_d": hi_q & ~big_d,
                 "QL_D": ~hi_q & big_d, "QL_d": ~hi_q & ~big_d}
        r = {"month": m, "all": float(g["fwd20"].mean()), "n": len(g)}
        for k, msk in cells.items():
            r[k] = float(g.loc[msk, "fwd20"].mean()) if msk.sum() >= MIN_CELL else np.nan
            r[k + "_n"] = float(msk.sum())
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("month")


def _tstat(v: pd.Series) -> float | None:
    v = v.dropna()
    if len(v) < 6 or v.std(ddof=1) == 0:
        return None
    return round(float(v.mean() / v.std(ddof=1) * math.sqrt(len(v))), 2)


def analyse(cells: pd.DataFrame, qcol: str, dcol: str,
            q_thr: float, d_thr: float) -> dict:
    """四格均值 + 交互项检验。"""
    if cells.empty or len(cells) < MIN_MONTHS:
        return {"ok": False, "reason": f"可用月份不足（{len(cells)}）"}
    out = {"ok": True, "qcol": qcol, "dcol": dcol,
           "q_thr": q_thr, "d_thr": d_thr, "months": int(len(cells))}
    base = cells["all"]
    for k in ("QH_D", "QH_d", "QL_D", "QL_d"):
        v = cells[k]
        ex = v - base
        out[k] = {"mean_pp": round(float(v.mean()) * 100, 3),
                  "excess_pp": round(float(ex.mean()) * 100, 3),
                  "t_excess": _tstat(ex),
                  "t_abs": _tstat(v),
                  "win": round(float((v > base).mean()), 3),
                  "avg_n": int(cells[k + "_n"].mean())}
    # 交互项：(高质量&大跌 − 高质量&小跌) − (低质量&大跌 − 低质量&小跌)
    inter = (cells["QH_D"] - cells["QH_d"]) - (cells["QL_D"] - cells["QL_d"])
    out["interaction"] = {"pp": round(float(inter.mean()) * 100, 3),
                          "t": _tstat(inter),
                          "pos_rate": round(float((inter > 0).mean()), 3)}
    # 跌幅的价值（在不同质量下分别看）——这才是判"是不是纯反转"的关键
    out["drop_effect_in_highQ"] = round(
        float((cells["QH_D"] - cells["QH_d"]).mean()) * 100, 3)
    out["drop_effect_in_lowQ"] = round(
        float((cells["QL_D"] - cells["QL_d"]).mean()) * 100, 3)
    out["quality_effect_in_bigDrop"] = round(
        float((cells["QH_D"] - cells["QL_D"]).mean()) * 100, 3)
    out["quality_effect_in_smallDrop"] = round(
        float((cells["QH_d"] - cells["QL_d"]).mean()) * 100, 3)
    base_mean = float(base.mean()) * 100
    out["market_pp"] = round(base_mean, 3)
    # 判定
    ti, td = out["interaction"]["t"], out["QH_D"]["t_excess"]
    if ti is not None and ti >= 2.0 and td is not None and td >= 2.0:
        out["verdict"] = "质量放大了跌幅价值（交互显著）⇒ 值得做"
    elif td is not None and td >= 2.0:
        out["verdict"] = "A 格本身有效，但交互不显著 ⇒ 更像纯反转，质量是装饰"
    else:
        out["verdict"] = "不成立"
    return out


def grid(panel: pd.DataFrame, qcol: str, dcol: str, n: int = 5) -> dict:
    """质量 × 跌幅 的 n×n 矩阵（看单调性，而不只看两个极端格）。"""
    rows = []
    for m, g in panel.groupby("month_ts", sort=True):
        g = g.dropna(subset=[qcol, dcol, "fwd20"])
        if len(g) < MIN_STOCKS:
            continue
        qb = pd.qcut(g[qcol].rank(method="first"), n, labels=False)
        db = pd.qcut(g[dcol].rank(method="first"), n, labels=False)
        for qi in range(n):
            for di in range(n):
                msk = (qb == qi) & (db == di)
                if msk.sum() >= MIN_CELL:
                    rows.append({"month": m, "q": qi, "d": di,
                                 "fwd": float(g.loc[msk, "fwd20"].mean())})
    if not rows:
        return {}
    t = pd.DataFrame(rows)
    piv = t.groupby(["q", "d"])["fwd"].mean().unstack() * 100
    return {"matrix_pp": {f"Q{q+1}": {f"D{d+1}": round(float(piv.loc[q, d]), 3)
                                      for d in piv.columns}
                          for q in piv.index},
            "note": "Q5=质量最高，D5=跌得最多（数值=月均 fwd20%）"}


def main(argv=None) -> int:
    import core.factor_lab as FL

    panel = build(FL.load_panel(need_fin=True))
    print(f"面板 {len(panel):,} 行 / {panel['month_ts'].nunique()} 月 / "
          f"{panel['code'].nunique():,} 只")

    report = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "panel": FL.panel_info(), "results": []}

    for qcol in QUALITY:
        if qcol not in panel.columns:
            continue
        for dcol in DROP:
            for q_thr, d_thr in ((0.5, 0.7),):
                cells = monthly_cells(panel, qcol, dcol, q_thr, d_thr)
                r = analyse(cells, qcol, dcol, q_thr, d_thr)
                report["results"].append(r)
                if not r.get("ok"):
                    print(f"\n{qcol} × {dcol}: {r.get('reason')}")
                    continue
                print(f"\n=== {qcol} × {dcol}（质量>{q_thr:.0%}、跌幅>{d_thr:.0%}分位，"
                      f"{r['months']} 个月，基准月均 {r['market_pp']:+.2f}%）===")
                print(f"{'格':<22}{'月均%':>9}{'超额pp':>9}{'t(超额)':>9}{'胜率':>7}{'只/月':>7}")
                cn = {"QH_D": "质量高 & 跌得多", "QH_d": "质量高 & 跌得少",
                      "QL_D": "质量低 & 跌得多", "QL_d": "质量低 & 跌得少"}
                for k, label in cn.items():
                    c = r[k]
                    print(f"{label:<22}{c['mean_pp']:+8.2f}%{c['excess_pp']:+9.2f}"
                          f"{str(c['t_excess']):>9}{c['win']*100:6.0f}%{c['avg_n']:>7}")
                it = r["interaction"]
                print(f"  交互项 (A−B)−(C−D) = {it['pp']:+.3f}pp  t={it['t']}  "
                      f"正比例 {it['pos_rate']*100:.0f}%")
                print(f"  跌幅效应：高质量组 {r['drop_effect_in_highQ']:+.3f}pp | "
                      f"低质量组 {r['drop_effect_in_lowQ']:+.3f}pp")
                print(f"  质量效应：跌得多组 {r['quality_effect_in_bigDrop']:+.3f}pp | "
                      f"跌得少组 {r['quality_effect_in_smallDrop']:+.3f}pp")
                print(f"  判定：{r['verdict']}")

    # 主口径的 5×5 矩阵（看单调性）
    main_q, main_d = "ocf_to_np", "drop1m"
    if main_q in panel.columns:
        g = grid(panel, main_q, main_d, 5)
        report["grid"] = {"qcol": main_q, "dcol": main_d, **g}
        if g:
            print(f"\n=== {main_q} × {main_d} 5×5 矩阵（月均 fwd20%）===")
            hdr = "        " + "".join(f"{d:>9}" for d in g["matrix_pp"]["Q1"])
            print(hdr)
            for qk, row in g["matrix_pp"].items():
                print(f"{qk:<8}" + "".join(f"{v:>9.2f}" for v in row.values()))

    OUT.mkdir(exist_ok=True)
    p = OUT / f"质量x跌幅检验_{datetime.now():%Y%m%d}.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nJSON: {p}")
    print("⚠️ 面板口径：不含交易成本与换手假设，不等于组合收益；"
          "上线前仍需 `portfolio_ab_test.py` 做组合级 A/B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
