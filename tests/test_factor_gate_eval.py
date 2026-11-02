# -*- coding: utf-8 -*-
"""因子闸门评估（`core/factor_gate_eval.py`）的一致性守护。

为什么这个文件几乎全是"一致性"测试
──────────────────────────────────
这个模块的错法很特殊：**算错了不会报错，只会给出反向结论**。
实测踩到的真事故：`evaluate_stack` 用 `dropna(fwd20)` 的**超集**当基准，
把"因因子缺失而不参与筛选"的票算进了对照，于是 `lndvol20` 从 +0.19pp
翻成 −0.59pp——结论完全反号，而所有返回值都"看起来正常"。

所以这里钉的不是"数值对不对"，而是"两个入口在**同一件事**上必须给同一个答案"：
单列叠加 == 单因子评估。这条一破，说明基准池口径又分叉了。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import factor_gate_eval as G               # noqa: E402


def _panel(months: int = 30, n: int = 300, seed: int = 3) -> pd.DataFrame:
    """合成面板：`good` 高值好（IC 正），`bad` 低值好（IC 负），`noise` 无关。

    `tail_hi` 特意做成"全截面看是负 IC、但极端高值端反而更好"——用来验证
    "方向不能照 IC 符号定"这条结论。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i, m in enumerate(pd.date_range("2023-01-31", periods=months, freq="ME")):
        for k in range(n):
            good = rng.normal()
            bad = rng.normal()
            noise = rng.normal()
            tail_hi = rng.normal()
            # fwd20 由 good/bad 驱动（噪声占大头，模拟真实 IC≈0.05 的量级）
            fwd = 0.02 * good - 0.02 * bad + 0.05 * rng.normal()
            if tail_hi > 2.5:
                fwd += 0.03          # 极端高值端额外为正 ⇒ 与全截面 IC 反号
            rows.append({"month_ts": m, "code": f"{k:06d}", "good": good,
                         "bad": bad, "noise": noise, "tail_hi": tail_hi,
                         "fwd20": fwd})
    return pd.DataFrame(rows)


def test_single_col_stack_matches_single_gate():
    """⚠️ 核心一致性：单列叠加必须与单因子评估给出同一个数。"""
    p = _panel()
    for col in ("good", "bad", "noise"):
        for pct in (0.10, 0.20):
            a = G.evaluate_gate(p, col, pct, min_stocks=100)
            b = G.evaluate_stack(p, [col], pct=pct, min_stocks=100)
            assert a["ok"] and b["ok"]
            assert abs(a["gain_pp"] - b["gain_pp"]) < 1e-6, (
                f"{col}@{pct}: 单因子 {a['gain_pp']} vs 叠加 {b['gain_pp']} "
                "⇒ 两处基准池口径又分叉了")


def test_direction_positive_ic_drops_low_end():
    p = _panel()
    r = G.evaluate_gate(p, "good", 0.10, min_stocks=100)
    assert r["direction"] == 1, "IC 为正 ⇒ 剔低端"
    assert r["gain_pp"] > 0


def test_direction_negative_ic_drops_high_end():
    p = _panel()
    r = G.evaluate_gate(p, "bad", 0.10, min_stocks=100)
    assert r["direction"] == -1, "IC 为负 ⇒ 剔高端"
    assert r["gain_pp"] > 0


def test_noise_factor_has_no_edge():
    p = _panel()
    r = G.evaluate_gate(p, "noise", 0.10, min_stocks=100)
    assert r["verdict"] in ("无效", "方向可用"), "纯噪音不该被判成值得验证"


def test_both_directions_are_reported():
    """双向实测：必须同时给出反向的数字，否则看不出"剔反了会怎样"。"""
    p = _panel()
    r = G.evaluate_gate(p, "good", 0.10, min_stocks=100)
    assert r["opposite_gain_pp"] is not None
    assert r["direction_vs_ic"] in ("一致", "相反")


def test_tail_can_contradict_ic():
    """全截面 IC 为负但极端高值端更好 ⇒ 最优方向应与 IC 相反（真事故的合成版）。"""
    p = _panel()
    ic, _ = G._monthly_ic(p, "tail_hi", 100)
    r = G.evaluate_gate(p, "tail_hi", 0.10, min_stocks=100)
    if ic < 0 and r["direction"] == 1:
        assert r["direction_vs_ic"] == "相反"
    # 不强制一定相反（合成数据有随机性），但要保证两者一致地报出来
    assert (r["direction_vs_ic"] == "相反") == (r["direction"] != (1 if ic >= 0 else -1))


def test_direction_mismatch_filters_correctly():
    p = _panel()
    rows = G.sweep(p, ["good", "bad", "tail_hi"], pcts=(0.10,))
    mm = G.direction_mismatch(rows)
    assert all(x["direction_vs_ic"] == "相反" for x in mm)
    assert len(mm) == sum(1 for x in rows if x["direction_vs_ic"] == "相反")


def test_stack_reports_keep_ratio():
    """叠加必须报保留率——不然"收益变好但只剩 30 只"看不出来。"""
    p = _panel()
    r = G.evaluate_stack(p, ["good", "bad"], pct=0.10, min_stocks=100)
    assert r["ok"]
    assert 0.5 < r["keep_ratio"] < 1.0
    assert r["avg_kept"] < r["avg_pool"]


def test_insufficient_months_is_not_ok():
    p = _panel(months=6)
    r = G.evaluate_gate(p, "good", 0.10, min_stocks=100)
    assert r["ok"] is False
    assert "月份不足" in r["reason"]


def test_stack_drops_factor_nan_pool_consistently():
    """因子有大片 NaN 时，基准池必须只含该因子可用的票（这就是翻号的那个 bug）。"""
    p = _panel()
    # 让 30% 的行 `sparse` 缺失，且这些行的 fwd20 显著更高
    p["sparse"] = np.where(np.arange(len(p)) % 10 < 3, np.nan, np.random.default_rng(1).normal(size=len(p)))
    p.loc[p["sparse"].isna(), "fwd20"] += 0.05
    a = G.evaluate_gate(p, "sparse", 0.10, min_stocks=100)
    b = G.evaluate_stack(p, ["sparse"], pct=0.10, min_stocks=100)
    assert abs(a["gain_pp"] - b["gain_pp"]) < 1e-6


# ════════════════════════════════════════════��═════════════════════
# Top-N 命中率：防止"面板有效 ⇒ 以为组合有效"（2026-11-01 实测教训）
# ══════════════════════════════════════════════════════════════════
def _toy_panel(months: int = 30, n: int = 200) -> tuple:
    """合成面板：`score` 越高越好（code 000000 最高），`gate_indep` 与排名交错。

    `gate_collinear = -score` ⇒ 它的高端正好是 score 的低端 ⇒ 用它当闸门剔的是
    分数最差的那批，与 TopN 零重叠。这正是本项目三道闸门踩到的情形。
    """
    rows = []
    for m in range(months):
        ts = pd.Timestamp("2020-01-31") + pd.offsets.MonthEnd(m)
        for i in range(n):
            rows.append({
                "month_ts": ts, "code": f"{i:06d}",
                "score": float(n - i),
                "gate_collinear": -float(n - i),
                # ⚠️ 必须是**连续**的伪随机：用 i%2 只有两取值 ⇒ quantile(0.8)=1、
                #    `v>1` 只剔掉 0 只，用例会"通过得莫名其妙"。乘 37 再取模打散。
                "gate_indep": float((i * 37) % 101),
                "fwd20": 0.001 * (1 if i < n // 2 else -1),
            })
    p = pd.DataFrame(rows)
    p["_s"] = p["score"]
    return p, p["_s"]


def test_topn_impact_collinear_gate_has_no_increment():
    """与打分共线的闸门 ⇒ TopN 命中率≈0，必须判"无增量"。

    这是本项目的真实教训：evaluate_stack 报 +0.76pp/月 t=5.00，看着该上线，
    但组合级 A/B 里 G3 臂与现状臂**逐位相同**，因为 1,800 个 Top20 席位里
    闸门只剔到 1 个。没有这条判据，就会把"面板有效"误当"组合有效"。
    """
    panel, score = _toy_panel()
    r = G.topn_impact(panel, ["gate_collinear"], {"gate_collinear": -1},
                      pct=0.20, topn=20, score=score)
    assert r["ok"] is True
    assert r["hit_rate"] < G.TOPN_MIN_HIT, f"应判无增量，实际 hit_rate={r['hit_rate']}"
    assert r["has_increment"] is False
    assert "无增量" in r["verdict"]
    # 面板层面它其实"有效"（被剔票的 fwd 更差）——正是这个反差会骗人
    assert r["panel_kept_fwd_pp"] > r["panel_dropped_fwd_pp"]
    # 被剔票集中在分数差的一端
    assert r["dropped_composite_quantile_median"] > 0.5


def test_topn_impact_independent_gate_does_hit():
    """与打分无关的闸门 ⇒ TopN 命中率≈剔除比例，判"可能有效"。"""
    panel, score = _toy_panel()
    r = G.topn_impact(panel, ["gate_indep"], {"gate_indep": -1},
                      pct=0.20, topn=20, score=score)
    assert r["ok"] is True
    assert r["hit_rate"] > G.TOPN_MIN_HIT, f"应判有增量，实际 {r['hit_rate']}"
    assert r["has_increment"] is True
    # 随机剔 20% ⇒ TopN 里约 20% 被剔（容忍区间）
    assert G.TOPN_MIN_HIT < r["hit_rate"] < 0.6


def test_topn_impact_buckets_are_monotonic():
    """共线闸门的分位段剔除率必须单调递增（分数越差剔得越多）。"""
    panel, score = _toy_panel()
    r = G.topn_impact(panel, ["gate_collinear"], {"gate_collinear": -1},
                      pct=0.20, topn=20, score=score)
    b = r["drop_rate_by_bucket"]
    keys = ["前0~5%", "前5~10%", "前10~20%", "前20~40%", "前40%以后"]
    vals = [b[k] for k in keys]
    assert vals == sorted(vals), f"剔除率应单调递增: {vals}"
    # 200 只剔 40 只 ⇒ "前40%以后" 那 120 只里最多剔到 40/120=33%（再高不可能）
    assert vals[-1] > 0.3 and vals[0] < 0.05


def test_topn_impact_nan_gate_value_not_dropped():
    """缺失值不许被剔（缺数据 ≠ 差）。"""
    panel, score = _toy_panel(months=26, n=200)
    # 让最高分的那批票闸门值缺失：若被剔，hit 就会异常高
    top = panel.index[panel["code"] < "000020"]
    panel = panel.copy()
    panel.loc[top, "gate_collinear"] = np.nan
    r = G.topn_impact(panel, ["gate_collinear"], {"gate_collinear": -1},
                      pct=0.20, topn=20, score=score)
    assert r["ok"] is True
    assert r["hit"] == 0, f"NaN 不该被剔，实际 hit={r['hit']}"


def test_topn_impact_too_few_months():
    panel, score = _toy_panel(months=6)
    r = G.topn_impact(panel, ["gate_collinear"], {"gate_collinear": -1},
                      pct=0.20, topn=20, score=score)
    assert r["ok"] is False and "月份不足" in r["reason"]


def test_topn_impact_without_score_column_fails_cleanly():
    """没传 score 且面板缺 PROMOTED_FACTORS 列 ⇒ 明确报错，不许静默用别的东西打分。"""
    panel, _ = _toy_panel(months=30)
    r = G.topn_impact(panel[["month_ts", "code", "fwd20", "gate_indep"]],
                      ["gate_indep"], {"gate_indep": -1})
    assert r["ok"] is False and "PROMOTED_FACTORS" in r["reason"]
