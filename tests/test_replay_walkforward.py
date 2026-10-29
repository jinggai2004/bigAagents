# -*- coding: utf-8 -*-
"""Walk-forward 权重重估测试（replay_eval B 模块）：合成矩阵，零面板依赖。

核心断言（机制层，不赌 alpha）：
1. composite_score 的 weights 参数：默认 = 旧行为（PORTFOLIO_WEIGHTS）；
   自定义权重改变打分；≤0 权重的因子不进复合。
2. 折结构：test 年的权重只用**严格早于**它的年份算（前视防护的结构性验证）。
3. 机制有效性：训练窗里 ma_align 无区分度 ⇒ 该折权重为 0；进测试期后
   有区分度的因子在后续折被赋权（机制"能捡起新信号"，不是固定权重复读）。
4. OOS 指标对称产出：refit 与 baseline 同口径可比。
"""
import numpy as np
import pandas as pd
import pytest

from core import replay_eval as rv


def _synth_world(n_years: int = 6, codes: int = 200, seed: int = 7):
    """合成因子世界：momentum_20 在前半段有 IC、后半段失效；ma_align 相反。

    fwd_5 直接由因子构造（测试不走 compute_factors/close——
    那条链路的真实数据口径由 test_replay_eval.py 覆盖）。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_years * 240)
    zA = pd.DataFrame(rng.normal(size=(len(dates), codes)), index=dates)
    zB = pd.DataFrame(rng.normal(size=(len(dates), codes)), index=dates)
    zC = pd.DataFrame(rng.normal(size=(len(dates), codes)), index=dates)
    # 制度切换：zA 驱动 2019-2022 的收益，zB 驱动 2023 起（与 train_years=4 的
    # 折切分对齐：2023 折的训练窗 2019-2022 里 ma_align 无 IC）
    years = dates.year
    early = np.asarray(years <= 2022)
    late = ~early
    noise = pd.DataFrame(rng.normal(0, 0.01, size=(len(dates), codes)), index=dates)
    fwd5 = (0.02 * zA.mul(early, axis=0)
            + 0.02 * zB.mul(late, axis=0)
            + noise)
    factors = {"momentum_20": zA, "ma_align": zB, "volatility_20": zC}
    return factors, {5: fwd5}


def test_composite_score_weights_param():
    factors, _ = _synth_world(n_years=2)
    default = rv.composite_score(factors)
    assert not default.empty
    # 自定义权重（去掉波动因子）⇒ 与默认不同
    custom = rv.composite_score(factors, weights={"momentum_20": 0.6, "ma_align": 0.4})
    assert not custom.empty
    assert not default.equals(custom)
    # 全零/空权重 ⇒ 空表（不是报错、不是全 0 假分）
    assert rv.composite_score(factors, weights={"momentum_20": 0.0}).empty


def test_walk_forward_fold_structure_no_lookahead():
    """折结构：test 年的 train_years 必须严格早于 test 年（前视防护硬验证）"""
    factors, fwd = _synth_world(n_years=6)
    res = rv.walk_forward_refit(factors, fwd, horizon=5, train_years=4)
    assert res["summary"]["available"] and res["summary"]["n_folds"] == 2
    fold_2023, fold_2024 = res["folds"]
    assert fold_2023["test_year"] == 2023
    assert fold_2023["train_years"] == [2019, 2020, 2021, 2022]
    assert fold_2024["test_year"] == 2024
    assert fold_2024["train_years"] == [2020, 2021, 2022, 2023]  # 滚动，不含 2019


def test_walk_forward_mechanism_picks_up_regime():
    """机制有效性：训练窗（2019-2022）里 ma_align 无 IC ⇒ 首折权重 0；
    滚动到含 2023 的训练窗（2023 起 ma_align 有 IC）⇒ 后折被赋权。"""
    factors, fwd = _synth_world(n_years=6)
    res = rv.walk_forward_refit(factors, fwd, horizon=5, train_years=4)
    f23, f24 = res["folds"]
    assert f23["weights_refit"].get("ma_align", 0.0) == pytest.approx(0.0, abs=1e-9)
    assert f23["weights_refit"].get("momentum_20", 0.0) > 0.5   # 早期 momentum 主导
    assert sum(f23["weights_refit"].values()) == pytest.approx(1.0, abs=1e-6)
    # 2024 折的训练窗含 2023（ma_align 开始有 IC）⇒ 权重被重新分配
    assert f24["weights_refit"].get("ma_align", 0.0) > 0.0
    # OOS 指标对称产出（refit 与 baseline 同口径）
    for fold in (f23, f24):
        assert fold["oos_refit"]["available"]
        assert fold["oos_baseline"]["available"]
        assert fold["oos_refit"]["composite_ic_mean"] is not None


def test_walk_forward_summary_aggregates():
    factors, fwd = _synth_world(n_years=6)
    s = rv.walk_forward_refit(factors, fwd, horizon=5, train_years=4)["summary"]
    assert s["n_folds"] == 2
    for k in ("top_excess_mean_refit", "top_excess_mean_baseline",
              "composite_ic_mean_refit", "composite_ic_mean_baseline"):
        assert s[k] is not None


def test_walk_forward_insufficient_years_returns_unavailable():
    """不足 train_years+1 年 ⇒ 明确 unavailable（不硬算假折）"""
    factors, fwd = _synth_world(n_years=2)
    res = rv.walk_forward_refit(factors, fwd, horizon=5, train_years=4)
    assert not res["summary"]["available"]


def test_walk_forward_negative_icir_factor_zeroed():
    """ICIR<0 的因子权重归 0 而不是反向押注（方向语义守卫）"""
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2019-01-02", periods=4 * 240)
    # 构造 momentum 与 fwd **负相关**的训练窗（语义方向 +1 但数据反着走）
    zA = pd.DataFrame(rng.normal(size=(len(dates), 200)), index=dates)
    fwd5 = -0.02 * zA + pd.DataFrame(rng.normal(0, 0.01, size=(len(dates), 200)),
                                     index=dates)
    factors = {"momentum_20": zA, "ma_align": pd.DataFrame(rng.normal(
        size=(len(dates), 200)), index=dates)}
    w = rv._icir_weights(factors, fwd5,
                         pd.Series(True, index=dates), step=5, min_names=100)
    assert w.get("momentum_20", 0.0) == 0.0
    if "ma_align" in w:
        assert w["ma_align"] >= 0.0
