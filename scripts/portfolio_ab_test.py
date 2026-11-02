# -*- coding: utf-8 -*-
"""组合构建 A/B 实验：验证「打分越高越买」到底对不对，闸门是否真的救得回来。

背景（2026-10-30 的 61 因子回测结论）
──────────────────────────────────────
- 量价因子 IC **全为负**：lndvol20 −0.109、upshd20 −0.095、turn_std20 −0.090；
- 61 次试验过 DSR/PBO 后只剩 2 个（lndvol20 / amihud20），且两者多空收益
  相关系数 −0.94（翻正后 +0.94）⇒ 只是**一个**独立赌注；
- 而 `core/screener._rank_and_select` 的机制是「加权求和 → 取分数最高 N 只」。

所以本实验只回答一个问题：**在同一个因子池上，把"高分买入"换成"反向/闸门"，
月频组合收益会不会变好？** 若不会，说明"IC 全负"这个诊断虽对但不可交易
（大概率是低流动性溢价吃不到），那就不该动 screener。

四臂（同一因子池、同一候选月、同一口径）
──────────────────────────────────────────
- **A 现状**：direction 调整后加权求和，**取分数最高** N 只（当前机制）；
- **A2 反向**：同一分数，**取分数最低** N 只（把"分数越高越买"整个反过来的对照）；
- **B 闸门**：先按负向因子极端分位**剔除**（上影/换手波动/波动率/换手/成交额上沿），
  再在存活池内用同一分数排序取 N 只 + 缓冲区调仓（`select_with_buffer`）；
- **C LightGBM**：滚动训练（不可用时跳过并标 skipped，不静默当成跑过）。

口径与已知限制
──────────────
- 收益 = 每只票的前瞻 20 交易日收益（`fwd20`，面板 qfq）；组合 = 等权 `fwd20` 均值；
- **未计交易成本**：因此额外输出换手率，便于按双边 0.2% 折算；
- **无涨跌停 / 停牌过滤**：缓存里没有当日价与成交量。四臂同等乐观，
  但绝对收益不能当预期收益（见报告"可行性"）；
- 幸存者偏差：面板只含当前在市股票 ⇒ 绝对收益偏乐观，**相对排序仍可用**；
- 三段切分：开发 / 验证 / 押注，押注段只在最后看一次。

用法
────
    py scripts/portfolio_ab_test.py            # 用扫描缓存
    py scripts/portfolio_ab_test.py --top 5    # 改组合只数
    py scripts/portfolio_ab_test.py --no-guard # 关守卫（对比污染前后用，结果不可用于决策）
    py scripts/portfolio_ab_test.py --no-liq-floor  # 关流动性下限（对比"线上会剔哪些票"用）
"""
from __future__ import annotations

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

# ⚠️ 必须在 sys.path 设置**之后**导入（否则 ModuleNotFoundError: core）
# 因子池与方向的**唯一来源**（两处各写一份必然走样）
from core.factor_composite import PROMOTED_FACTORS  # noqa: E402

CACHE = _ROOT / "temp" / "factor_scan_scan.parquet"
OUT = _ROOT / "output"

# 6 个已晋身因子 + 方向（−1=低值好 / +1=高值好），定义在 core/factor_composite
POOL = PROMOTED_FACTORS

# screener 里"趋势"(0.1765) + "均线排列"(0.1176) 合计 29.4% 是**正向**权重（越高越好），
# 而本池全部是反转/低风险方向。A3 臂就是把这两个加回去，检验"动量类是否在拖累"。
POOL_MOM = {**POOL, "macd_hist": +1}

# 量价池 + 盈利质量（ocf_to_np：经营现金流/归母净利，东财 PIT 财报）。
# IC +0.015 / t=+3.5 / 过 DSR(p=0.015) / 分年 7/8 正 —— 61 个量价因子全灭后
# 第一个过多重检验的因子，且是**正 IC**（与量价池方向互补）。
# ⚠️ 方向 +1 与 IC 符号一致（高盈利质量 → 高收益）。
POOL_FIN = {**POOL, "ocf_to_np": +1}

# 闸门：只对**负向**因子做"剔除高端"（正向因子没有"越高越差"的先验，不闸）
GATE_FACTORS = ("upshd20", "turn_std20", "lndvol20", "pctvol20", "turn20")

# 三道闸门（2026-11-01 `core/factor_gate_eval.py` 的结论）。
# 逐因子双向实测取更优方向，再叠加看是否单调：3 道×各剔 20%
# = **+0.85pp/月 t=5.00 胜率 76% 保留 51%**，第 4 道（up_streak）回落到 +0.59pp
# 且砍到 41% ⇒ 3 道是甜点。
# ⚠️ 方向**不是从 IC 符号推的**（尾部与全截面可反号，实测 20 组反号）。
# 这里的方向来自开发段（2019-03~2023-12）实测，已验证与全样本一致且开发段更强
# （lndvol20 t=6.22 / amihud20 t=5.02 / upshd20 t=4.44）⇒ 无前视。
GATE3 = {"lndvol20": -1, "amihud20": +1, "upshd20": -1}
GATE3_PCT = 0.20              # 各剔 20%（单闸实测 20% 普遍优于 10%）
GATE3_PRE_TOP_MULT = 10       # 前置 TopK 模式：先按 composite 取 top×10 当"候选池"
GATE_QUANTILE = 0.90          # 剔除各因子最差的 10%
AMOUNT_LOW_Q = 0.10           # 容量保护：剔除成交额最低 10%（"进不去"的票）
AMOUNT_HIGH_Q = 0.95          # 负 alpha 端：剔除成交额最高的 5%

SEGMENTS = [
    ("开发段", "2019-03-01", "2023-12-31"),
    ("验证段", "2024-01-01", "2024-12-31"),
    ("押注段", "2025-01-01", "2026-12-31"),
]


_LIQ_PCT = None      # 运行时由 --liq-pct 覆盖（小资金应测 0.10/0.20）


def _load(guard: bool = True) -> pd.DataFrame:
    """读扫描缓存并算合成分数（不联网、不重算因子）。

    `guard=True` 时套 `core.universe_guard.eligible`（上市满一年 + 北交所开市后才纳入）。
    ⚠️ 2026-10-31 实测：不加这条守卫，2019~2023 的 top20 里北交所占 21%~55%，
       且 2019~2020 的 920xxx 是**新三板历史映射**（当期停牌、复牌跳空造出 +142% 伪收益）。
       不做守卫的数字不可用于决策。
    """
    if not CACHE.exists():
        raise SystemExit(
            f"缺少扫描缓存 {CACHE}。先跑 `py scripts/factor_paper_scan.py` 生成。")
    df = pd.read_parquet(CACHE)
    if "month_ts" not in df.columns:
        df = df.rename(columns={"date": "month_ts"})
    df["month_ts"] = pd.to_datetime(df["month_ts"])
    # 20 日成交额（元）＝ exp(lndvol20)，用于容量闸门
    df["amt20"] = np.exp(pd.to_numeric(df["lndvol20"], errors="coerce"))
    df["pctvol20"] = pd.to_numeric(df.get("vol20"), errors="coerce")
    need = [c for c in POOL if c in df.columns]
    missing = [c for c in POOL if c not in df.columns]
    if missing:
        raise SystemExit(f"缓存缺因子列：{missing}（重跑 factor_paper_scan.py）")
    df = df.dropna(subset=["fwd20", "month_ts"])

    # ⚠️ 口径必须和 2026-10-30 那版一致，否则对不上已发布的数字：
    #    ① 每个因子在**各自的有效集**内排名（不能先 dropna 成"六个都全"的子集再排，
    #       那样分母变小、百分位全变——实测押注段超额从 6.95% 漂到 6.00%）；
    #    ② 六个因子缺任一个的行 ⇒ 合成分数置 NaN（等同"落选"）。
    from core.factor_composite import composite_score

    def _assign(col: str, cols: list[str], dirs: dict) -> None:
        comp = composite_score(df, cols, dirs)      # 全表排名（各自有效集）
        comp[df[cols].isna().any(axis=1)] = np.nan  # 不全的行不参与
        df[col] = comp

    _assign("composite", need, POOL)
    mom_cols = [c for c in POOL_MOM if c in df.columns]
    if len(mom_cols) > len(need):                     # macd_hist 存在才算
        _assign("composite_mom", mom_cols, POOL_MOM)

    # ── PIT 财务快照（--fin 时）：月末只看 pub_date 已公告的报表 ──
    # merge 不改行数；财报不可见（未披露/新上市）的行 ocf_to_np 为 NaN，
    # 与其他因子同一口径 ⇒ composite_fin 置 NaN（落选）。
    if "--fin" in sys.argv:
        from core.financial_pit import add_derived, load_financials, \
            merge_into_monthly_panel

        fin = add_derived(load_financials())
        before = df["code"].nunique()
        df = merge_into_monthly_panel(df, fin, ["ocf_to_np"])
        cov = df["ocf_to_np"].notna().mean()
        print(f"[财务快照] ocf_to_np 覆盖 {cov:.1%}（merge 前后股票数 "
              f"{before:,}→{df['code'].nunique():,}，行数不变）", flush=True)
        if df["code"].nunique() != before:
            raise SystemExit("财务 merge 改变了股票集 —— PIT 快照键不对，禁止继续")
        fin_cols = [c for c in POOL_FIN if c in df.columns]
        _assign("composite_fin", fin_cols, POOL_FIN)

    if guard:
        from core.universe_guard import eligible
        before = len(df)
        df = df[eligible(df)].copy()
        print(f"[守卫] 上市满一年 + 北交所开市后：{before:,} → {len(df):,} 行"
              f"（剔除 {before - len(df):,}，{1 - len(df) / max(before, 1):.1%}）", flush=True)

    # ── 流动性下限（2026-10-31）：**与线上 screener 同一条线** ──
    # 线上剔"板块内成交额最小 30%"，回测之前没有 ⇒ 两边在选不同的票，
    # 而因子池（低流动性+高冲击成本）恰好偏好线上会剔掉的那批。
    # ⚠️ 口径差异：线上用当日快照成交额，这里用 20 日均额（`lndvol20` 还原）。
    if "--no-liq-floor" not in sys.argv:
        from core import liquidity_floor as _lf
        df["amount_yuan"] = np.exp(pd.to_numeric(df["lndvol20"], errors="coerce"))
        before = len(df)
        df, liq = _lf.filter_monthly_panel(df, pct=_LIQ_PCT)
        print(f"[流动性下限] {before:,} → {len(df):,} 行（{liq['note']}）", flush=True)
    return df


def _gate_mask(g: pd.DataFrame) -> pd.Series:
    """闸门掩码（True=保留）。全部在**当月截面内**算分位。"""
    keep = pd.Series(True, index=g.index)
    for c in GATE_FACTORS:
        if c not in g.columns:
            continue
        v = pd.to_numeric(g[c], errors="coerce")
        if v.notna().sum() < 20:
            continue
        thresh = v.quantile(GATE_QUANTILE)
        keep &= ~(v > thresh)                    # 高值剔除（这些因子都是"越高越差"）
    amt = pd.to_numeric(g["amt20"], errors="coerce")
    if amt.notna().sum() >= 20:
        keep &= amt >= amt.quantile(AMOUNT_LOW_Q)       # 容量下限
        keep &= amt <= amt.quantile(AMOUNT_HIGH_Q)      # 负 alpha 上沿
    return keep


def _gate3_mask(g: pd.DataFrame, pct: float = GATE3_PCT) -> tuple:
    """三道闸门掩码（True=保留）+ 逐因子剔除数。

    `GATE3[c] = -1` ⇒ 剔**高端**（值越大越差）；`+1` ⇒ 剔**低端**。
    缺失值**不剔**（缺数据 ≠ 差，与盈利质量闸门同一纪律）。

    分位在**传入的池内**算，不跨池。这是硬要求：用全市场分位去切一个 200 只的
    候选池会把它剔光（本项目在流动性下限和自定义因子闸门上各踩过一次）。
    """
    keep = pd.Series(True, index=g.index)
    dropped = {}
    for c, direction in GATE3.items():
        v = pd.to_numeric(g.get(c), errors="coerce")
        if v.notna().sum() < 50:
            dropped[c] = None                 # 有效样本太少，该道不生效
            continue
        if direction < 0:
            bad = v > v.quantile(1 - pct)
        else:
            bad = v < v.quantile(pct)
        bad &= v.notna()                      # NaN 不剔
        dropped[c] = int(bad.sum())
        keep &= ~bad
    return keep, dropped


def _monthly_pick(g: pd.DataFrame, top: int, arm: str, prev: list[str],
                  buffer_mult: float = 1.0) -> tuple[list[str], dict]:
    """单月选股。

    `buffer_mult=1.0` = 严格 TopN（每次全按排名重选，换手最高）；
    >1 = TopK+Drop 缓冲（跌出 top×mult 才卖），这是**压换手**的唯一开关。
    """
    from core.portfolio_construct import select_with_buffer

    pool = g
    gate_stats = {}
    if arm == "B":
        keep = _gate_mask(g)
        gate_stats = {"before": int(len(g)), "after": int(keep.sum())}
        pool = g[keep]
    elif arm in ("G3", "G3T"):
        # 三道闸门。G3T 先按 composite 取 top×10 当"候选池"，再在池内闸——
        # 更贴近线上（线上是五路初筛出 ~120 只后闸门），也能看出
        # "月度面板上的 +0.85pp 在组合层面还剩多少"。
        scope = g
        if arm == "G3T":
            _sc = pd.to_numeric(g["composite"], errors="coerce")
            _pre = _sc.rank(ascending=False, method="first") <= top * GATE3_PRE_TOP_MULT
            scope = g[_pre.fillna(False)]
        if len(scope):
            keep3, dropped3 = _gate3_mask(scope)
            gate_stats = {"before": int(len(scope)), "after": int(keep3.sum()),
                          "dropped_by_factor": dropped3,
                          "scope": "topK" if arm == "G3T" else "monthly_panel"}
            pool = scope[keep3]
        else:
            pool = scope
    elif arm == "AFG":
        # 盈利质量当**闸门**不当分数：剔除 ocf_to_np 最差 10%（应计操纵风险端），
        # 排序仍用原量价 composite —— AF 臂证明弱 IC 因子进等权合成是稀释，
        # 但它的 IC 是正的，最低分位可能是"最危险"的一端。
        v = pd.to_numeric(g.get("ocf_to_np"), errors="coerce")
        if v.notna().sum() >= 20:
            keep = v.isna() | (v >= v.quantile(0.10))   # NaN 不剔（数据缺失≠质量差）
            gate_stats = {"before": int(len(g)), "after": int(keep.sum())}
            pool = g[keep]
    if len(pool) == 0:
        return [], gate_stats

    if arm == "AF":
        score_col = "composite_fin"
    elif arm in ("AFG", "G3", "G3T"):
        score_col = "composite"          # 排序不变，只闸
    else:
        score_col = "composite_mom" if arm == "A3" else "composite"
    asc = (arm == "A2")
    ranked = pool.sort_values(score_col, ascending=asc)["code"].astype(str).tolist()
    if buffer_mult > 1.0 and prev:
        sel = select_with_buffer(ranked, prev, top, buffer_mult=buffer_mult)
        return sel["codes"], {**gate_stats, "relaxed": sel["relaxed"],
                              "turnover": sel["turnover"]}
    return ranked[:top], gate_stats


def _run_arm(df: pd.DataFrame, arm: str, top: int,
             buffer_mult: float = 1.0) -> dict:
    """跑一个臂，返回逐月**毛收益**与逐月换手（净收益在 `_metrics` 里按换手扣）。"""
    gross, turns, gate_info = {}, {}, {}
    prev: list[str] = []
    for m, g in df.groupby("month_ts", sort=True):
        codes, gs = _monthly_pick(g, top, arm, prev, buffer_mult)
        if gs:
            gate_info = gs
        if not codes:
            prev = []
            continue
        fwd = g[g["code"].astype(str).isin(set(codes))].dropna(subset=["fwd20"])
        if len(fwd):
            gross[m] = float(fwd["fwd20"].mean())
            turns[m] = len(set(codes) - set(prev)) / max(len(codes), 1)
        prev = codes
    return {"returns": pd.Series(gross).sort_index(),
            "turnover": pd.Series(turns).sort_index(),
            "avg_turnover": round(float(np.mean(list(turns.values()))), 3) if turns else None,
            "gate": gate_info}


def _metrics(r: pd.Series, bench: pd.Series, turnover: pd.Series | None = None,
             cost_bp: float = 0.0) -> dict:
    """组合指标。r/bench 是**月频**收益；换手 × 双边成本 = 当月成本。

    Args:
        cost_bp: **单边**成本（bp）。双边 = 2×cost_bp。A 股现实量级：
            佣金 2.5bp + 印花税 5bp（卖出单边）+ 滑点 5~10bp ⇒ 单边 ~10bp。
    """
    if len(r) < 6:
        return {"ok": False, "reason": f"月数不足（{len(r)}）"}
    gross = r
    net = r
    if cost_bp > 0 and turnover is not None and len(turnover):
        t = turnover.reindex(r.index).fillna(0.0)
        net = r - t * 2.0 * cost_bp / 10000.0          # 买卖两侧
    out = {"ok": True, "months": int(len(r))}
    for tag, ser in (("gross", gross), ("net", net)):
        ann = float((1 + ser).prod() ** (12 / len(ser)) - 1)
        vol = float(ser.std(ddof=1) * math.sqrt(12))
        nav = (1 + ser).cumprod()
        out[f"{tag}_ann"] = round(ann, 4)
        out[f"{tag}_sharpe"] = round(ann / vol, 3) if vol > 0 else None
        if tag == "net":
            out["max_drawdown"] = round(float((nav / nav.cummax() - 1).min()), 4)
    ex = (net - bench.reindex(net.index)).dropna()
    out["excess_ann"] = (round(float((1 + ex).prod() ** (12 / len(ex)) - 1), 4)
                         if len(ex) >= 6 else None)
    out["excess_win"] = round(float((ex > 0).mean()), 3) if len(ex) else None
    out["excess_t"] = (round(float(ex.mean() / ex.std(ddof=1) * math.sqrt(len(ex))), 2)
                       if len(ex) > 2 and ex.std(ddof=1) > 0 else None)
    return out


def _segment(r: pd.Series, bench: pd.Series, start: str, end: str,
             turnover: pd.Series | None = None, cost_bp: float = 0.0) -> dict:
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    m = (r.index >= a) & (r.index <= b)
    t = turnover[m] if turnover is not None else None
    return _metrics(r[m], bench[m], t, cost_bp)


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    top, cost_bp = 20, 10.0
    for i, a in enumerate(argv):
        if a == "--top" and i + 1 < len(argv):
            top = int(argv[i + 1])
        if a == "--cost-bp" and i + 1 < len(argv):
            cost_bp = float(argv[i + 1])

    global _LIQ_PCT
    for i, x in enumerate(argv):
        if x == "--liq-pct" and i + 1 < len(argv):
            _LIQ_PCT = float(argv[i + 1])
    df = _load(guard="--no-guard" not in argv)
    print(f"截面 {len(df):,} 行 / {df['month_ts'].nunique()} 个月 / "
          f"{df['code'].nunique():,} 只 / top={top} / 单边成本 {cost_bp}bp")

    bench = df.groupby("month_ts")["fwd20"].mean()

    # (标签, 臂, 缓冲区倍数)
    specs = [
        ("A_现状严格Top", "A", 1.0),
        ("A_缓冲2.0x", "A", 2.0),
        ("A_缓冲3.0x", "A", 3.0),
        ("A3_混入趋势因子", "A3", 2.0),
        ("A2_反向(对照)", "A2", 1.0),
        ("B_闸门+缓冲1.5x", "B", 1.5),
        # 三道闸门（11-01 factor_gate_eval 结论）：严格 Top 看纯效果 + 缓冲看可实现性
        ("G3_三道闸门", "G3", 1.0),
        ("G3_三道闸门_缓冲2.0x", "G3", 2.0),
        ("G3T_闸门在TopK池内_缓冲2.0x", "G3T", 2.0),
    ]
    if "--fin" in argv:
        # AF = 同一机制、因子池混入 ocf_to_np（先严格 Top 看纯效果，再看缓冲）
        # AFG = ocf_to_np 只当闸门（剔最差 10%），排序仍用原量价分数
        specs = [("AF_量价+盈利质量_严格Top", "AF", 1.0),
                 ("AF_量价+盈利质量_缓冲2.0x", "AF", 2.0),
                 ("AFG_剔盈利质量差10%_缓冲2.0x", "AFG", 2.0),
                 ("AFG_剔盈利质量差10%_严格Top", "AFG", 1.0)] + specs
    out = {}
    for label, key, bm in specs:
        res = _run_arm(df, key, top, bm)
        r, t = res["returns"], res["turnover"]
        out[label] = {
            "overall": _metrics(r, bench, t, cost_bp),
            "segments": {name: _segment(r, bench, s0, e0, t, cost_bp)
                         for name, s0, e0 in SEGMENTS},
            "avg_turnover": res["avg_turnover"],
            "gate": res["gate"],
            # 毛序列（不扣成本）：保留用于对账
            "series": {str(k.date()): round(float(v), 5) for k, v in r.items()},
            # **净序列**（扣双边成本）：alpha_decay_monitor 判超额衰减必须用这个，
            # 用毛序列会把换手成本当成 alpha（本策略换手 0.8~0.9/月，差得很远）。
            "net_series": {str(k.date()): round(float(v), 5)
                           for k, v in (r - t.reindex(r.index).fillna(0.0)
                                        * 2.0 * cost_bp / 10000.0).items()},
        }
        o = out[label]["overall"]
        print(f"\n=== {label} | 平均换手 {res['avg_turnover']} | "
              f"全期毛 {o.get('gross_ann', 0):+.2%} 净 {o.get('net_ann', 0):+.2%} ===")
        for name, _, _ in SEGMENTS:
            seg = out[label]["segments"][name]
            if seg.get("ok"):
                print(f"  {name} 毛 {seg['gross_ann']:+.2%} 净 {seg['net_ann']:+.2%} "
                      f"夏普 {seg['net_sharpe']} 回撤 {seg['max_drawdown']:+.2%} "
                      f"超额 {seg['excess_ann']:+.2%} 胜率 {seg['excess_win']} t={seg['excess_t']}")
            else:
                print(f"  {name} {seg.get('reason')}")

    bm_out = {}
    print("\n=== 基准（全市场等权，无成本）===")
    for name, s0, e0 in SEGMENTS:
        a, b = pd.Timestamp(s0), pd.Timestamp(e0)
        sub = bench[(bench.index >= a) & (bench.index <= b)]
        m = _metrics(sub, sub)
        bm_out[name] = m
        if m.get("ok"):
            print(f"  {name} 年化 {m['gross_ann']:+.2%} 夏普 {m['gross_sharpe']} "
                  f"回撤 {m['max_drawdown']:+.2%}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    OUT.mkdir(exist_ok=True)
    path = OUT / f"组合AB实验_{stamp}.json"
    path.write_text(json.dumps({"top": top, "cost_bp": cost_bp, "pool": POOL,
                                "pool_fin": POOL_FIN if "--fin" in argv else None,
                                "gate3": {"factors": GATE3,
                                          "pct": GATE3_PCT,
                                          "pre_top_mult": GATE3_PRE_TOP_MULT,
                                          "direction_source": "development_segment_2019_2023"},
                                "gate": {"factors": GATE_FACTORS,
                                         "quantile": GATE_QUANTILE,
                                         "amount_low_q": AMOUNT_LOW_Q,
                                         "amount_high_q": AMOUNT_HIGH_Q},
                                "benchmark": bm_out,
                                # 逐月基准序列：alpha_decay_monitor 要用它算滚动超额
                                "benchmark_series": {str(k.date()): round(float(v), 5)
                                                     for k, v in bench.items()},
                                "arms": out},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nJSON: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
