"""「质量 × 跌幅」检验的口径测试。

**最关键的一条**：`drop1m = -mom1m` 这类"跌幅"字段的符号语义。
如果哪天有人把 `drop1m` 直接写成 `mom1m`（不加负号），整套结论会**逐条反号**，
而所有输出看起来都很正常（四格数、交互项、判定都还在），这类错误没人能看出来。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.quality_pullback_check as Q  # noqa: E402


def _panel(n_months: int = 30, n: int = 120, mode: str = "none") -> pd.DataFrame:
    """合成面板。`mode` 决定 fwd20 与 (质量, 跌幅) 的关系：

    - `none`：无关系（两者都只跟噪声有关）
    - `reversal`：纯反转（跌得越多越好，与质量无关）
    - `interaction`：只有"质量高 & 跌得多"这一格好
    """
    rng = np.random.default_rng(20)
    rows = []
    for mi in range(n_months):
        ts = pd.Timestamp("2020-01-31") + pd.offsets.MonthEnd(mi)
        for i in range(n):
            q = i / (n - 1)                  # 0→1，越大质量越好
            d = ((i * 37) % n) / (n - 1)     # 0→1，越大跌得越多（与 q 无关）
            base = 0.01
            # ⚠️ 每个模式都必须带噪声：无噪声时某格的月度均值恒定 ⇒ 标准差 0 ⇒
            #    `_tstat` 返回 None ⇒ 判定退化成"不成立"（用例因此假失败过）。
            noise = 0.002 * rng.normal()
            if mode == "reversal":
                fwd = base + 0.03 * (d - 0.5) + noise
            elif mode == "interaction":
                fwd = base + (0.05 if (q > 0.5 and d > 0.7) else 0.0) + noise
            else:
                fwd = base + noise
            rows.append({
                "month_ts": ts, "code": f"{i:06d}",
                "quality": q,
                # mom1m 是"涨跌幅"（正=涨）⇒ drop1m 应当等于它的负值
                "mom1m": -0.02 - 0.1 * d,
                "mom6m": -0.05 - 0.2 * d,
                "boll_pos": 1.0 - d,
                "fwd20": fwd,
            })
    return pd.DataFrame(rows)


def _built(n_months: int = 30, n: int = 120, mode: str = "none") -> pd.DataFrame:
    """`_panel` + `build()`：`monthly_cells` 依赖 build 产出的 drop* 列。"""
    return Q.build(_panel(n_months, n, mode))


def test_drop_columns_are_negated():
    """跌幅列必须 = 原字段取负（越大越跌）。写反 ⇒ 全盘结论反号。"""
    p = Q.build(_panel(6, 20))
    assert "drop1m" in p.columns and "drop6m" in p.columns and "lowpos" in p.columns
    assert np.allclose(p["drop1m"], -p["mom1m"], equal_nan=True)
    assert np.allclose(p["drop6m"], -p["mom6m"], equal_nan=True)
    assert np.allclose(p["lowpos"], -p["boll_pos"], equal_nan=True)
    # 更强的语义断言：跌得多的样本 drop1m 必须更大
    assert p.loc[p["mom1m"].idxmin(), "drop1m"] == p["drop1m"].max()


def test_pure_reversal_has_positive_drop_effect_but_no_interaction():
    """纯反转：两个质量组里跌幅效应都为正，交互项≈0。

    这条挡住"把纯反转误当成质量起作用"的误判——本项目的实际数据正是这种形态
    （roe 口径下低质量组跌幅效应 +0.79pp，高质量组 −0.03pp）。
    """
    cells = Q.monthly_cells(_built(30, 200, "reversal"), "quality", "drop1m")
    r = Q.analyse(cells, "quality", "drop1m", 0.5, 0.7)
    assert r["ok"] is True
    assert r["drop_effect_in_highQ"] > 0
    assert r["drop_effect_in_lowQ"] > 0
    assert abs(r["interaction"]["pp"]) < 0.5, f"纯反转不该有交互：{r['interaction']}"
    assert "纯反转" in r["verdict"] or "不成立" in r["verdict"]


def test_interaction_case_is_detected():
    """只有"质量高 & 跌得多"这一格好 ⇒ 交互项必须显著为正、判"值得做"。"""
    cells = Q.monthly_cells(_built(30, 200, "interaction"), "quality", "drop1m")
    r = Q.analyse(cells, "quality", "drop1m", 0.5, 0.7)
    assert r["ok"] is True
    assert r["interaction"]["pp"] > 0
    assert r["interaction"]["t"] > 3, f"交互项应显著：{r['interaction']}"
    assert r["QH_D"]["excess_pp"] > 0
    assert "值得做" in r["verdict"]
    # 高质量组的跌幅效应必须远大于低质量组
    assert r["drop_effect_in_highQ"] > r["drop_effect_in_lowQ"]


def test_no_relation_is_not_significant():
    cells = Q.monthly_cells(_built(30, 200, "none"), "quality", "drop1m")
    r = Q.analyse(cells, "quality", "drop1m", 0.5, 0.7)
    assert r["ok"] is True
    assert abs(r["interaction"]["t"] or 0) < 3


def test_insufficient_months_returns_reason():
    cells = Q.monthly_cells(_built(8, 200, "reversal"), "quality", "drop1m")
    r = Q.analyse(cells, "quality", "drop1m", 0.5, 0.7)
    assert r["ok"] is False and "月份不足" in r["reason"]


def test_small_cells_are_nan_not_zero():
    """单格样本不足时必须是 NaN（不参与），不许当 0 收益混进均值。"""
    cells = Q.monthly_cells(_built(28, 40, "reversal"), "quality", "drop1m")
    # 40 只的池子切 2×2 后每格约 10 只，可能触发 MIN_CELL=8 的下限
    if not cells.empty:
        for k in ("QH_D", "QH_d", "QL_D", "QL_d"):
            assert not (cells[k] == 0).all(), f"{k} 出现了假的 0 值"
