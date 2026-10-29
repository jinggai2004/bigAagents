# -*- coding: utf-8 -*-
"""P1-2 回放引擎（`core/replay_eval.py`）的契约。

锁住的不是"能算"，而是几条**错了会让结论反过来的**语义：

  1. `forward_returns` 是唯一碰未来价的函数，口径必须 `close[t+h]/close[t]-1`；
  2. IC 是**截面 Spearman**，样本不足/无方差 ⇒ None（**不是 0**，
     否则"没数据"会被平均成"没效应"）；
  3. `bootstrap_ci` 样本 < 10 期 ⇒ 不给区间（不给假精度）；
  4. 组合回放的 `step` 默认 = `horizon` —— step < horizon 时持仓重叠，
     逐期复利会把同一段收益重复计入（实测曾算出"年化 -98%"，纯假）；
  5. 过热诊断：剔除"已涨一大段"的票后，TOP 超额应显著改善（本次核心发现）。
"""
import numpy as np
import pandas as pd
import pytest

import core.replay_eval as R


# ================================================================
# 1. 前后视边界
# ================================================================

class TestForwardReturns:
    def test_forward_return_is_shift_minus_h(self):
        close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]},
                             index=["d1", "d2", "d3", "d4"])
        fwd = R.forward_returns(close, horizons=(1,))[1]
        assert fwd.loc["d1", "A"] == pytest.approx(0.1)     # 11/10-1
        assert fwd.loc["d3", "A"] == pytest.approx(13 / 12 - 1)
        assert pd.isna(fwd.loc["d4", "A"]), "最后一天没有未来 ⇒ NaN，不能填 0"

    def test_factors_do_not_look_ahead(self, monkeypatch):
        """因子只能用 T 日及之前：把未来价格改掉，T 日的因子值不该变。"""
        idx = [f"d{i}" for i in range(1, 41)]
        base = pd.DataFrame({"A": np.linspace(10, 20, 40)},
                            index=idx)
        m1 = {"close": base, "pct_chg": base.pct_change() * 100,
              "amount": pd.DataFrame({"A": [1e7] * 40}, index=idx)}
        f1 = R.compute_factors(m1)
        # 篡改"未来"（最后 10 天）
        tampered = base.copy()
        tampered.iloc[-10:, 0] = tampered.iloc[-10:, 0] * 3
        m2 = {"close": tampered, "pct_chg": tampered.pct_change() * 100,
              "amount": m1["amount"]}
        f2 = R.compute_factors(m2)
        # 第 25 天（未来尚未发生）的动量/均线必须一致
        assert f1["momentum_20"].loc["d25", "A"] == pytest.approx(
            f2["momentum_20"].loc["d25", "A"])


# ================================================================
# 2. IC 与置信区间
# ================================================================

def _synth(n_dates=30, n_codes=200, seed=7, invert=False):
    rng = np.random.default_rng(seed)
    dates = [f"2025-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n_dates)]
    codes = [f"{600000 + i}" for i in range(n_codes)]
    f = pd.DataFrame(rng.normal(size=(n_dates, n_codes)), index=dates, columns=codes)
    noise = rng.normal(scale=0.5, size=(n_dates, n_codes))
    r = (-f if invert else f) + noise
    return f, pd.DataFrame(r, index=dates, columns=codes)


class TestIC:
    def test_positive_ic_when_factor_predicts(self):
        f, r = _synth()
        s = R.ic_series(f, r, step=1, min_names=100)
        assert len(s) > 10 and s.mean() > 0.6, s.mean()

    def test_negative_ic_when_reversed(self):
        f, r = _synth(invert=True)
        s = R.ic_series(f, r, step=1, min_names=100)
        assert s.mean() < -0.6, s.mean()

    def test_ic_none_when_sample_too_small(self):
        f = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]}, index=["d1", "d2"])
        r = f.copy()
        assert R._rank_ic(f.loc["d1"], r.loc["d1"], min_names=100) is None

    def test_ic_none_when_factor_constant(self):
        # 200 只但因子值全相同 ⇒ 无方差 ⇒ None（不是 0）
        codes = [f"S{i}" for i in range(200)]
        f = pd.Series([5.0] * 200, index=codes)
        r = pd.Series(np.random.default_rng(1).normal(size=200), index=codes)
        assert R._rank_ic(f, r, min_names=100) is None

    def test_bootstrap_ci_refuses_with_few_periods(self):
        out = R.bootstrap_ci([0.1, 0.2, 0.3])
        assert out["n"] == 3 and out["lo"] is None and out["hi"] is None
        assert out["significant"] is None

    def test_bootstrap_ci_detects_significance(self):
        vals = [0.05 + 0.01 * (i % 3) for i in range(60)]   # 全正、稳定
        out = R.bootstrap_ci(vals, n_boot=500)
        assert out["significant"] is True and out["lo"] > 0

    def test_bootstrap_ci_not_significant_when_centered_on_zero(self):
        rng = np.random.default_rng(3)
        out = R.bootstrap_ci(rng.normal(size=80), n_boot=1000)
        assert out["significant"] is False


# ================================================================
# 3. 组合回放（step 语义 —— 踩过的坑）
# ================================================================

class TestPortfolio:
    def _setup(self, n_dates=120, n_codes=300):
        rng = np.random.default_rng(11)
        dates = [f"2025-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n_dates)]
        codes = [f"{600000 + i}" for i in range(n_codes)]
        score = pd.DataFrame(rng.normal(size=(n_dates, n_codes)),
                             index=dates, columns=codes)
        # ⚠️ 收益必须由**前一日**分数决定（`shift(1)`）：直接用同期 score
        # 等于"收益由未来信息决定"，TOP 组合与未来收益无关 ⇒ 超额恒 ≈ 0（已踩）。
        ret = score.shift(1).fillna(0) * 0.01 + \
            rng.normal(scale=0.005, size=(n_dates, n_codes))
        close = (1 + ret).cumprod() * 10
        return score, close

    def test_step_defaults_to_horizon(self):
        """⭐ step 缺省必须 = horizon：否则重叠持仓 + 复利 = 虚假年化。"""
        score, close = self._setup()
        p = R.portfolio_backtest(score, close, horizon=20, top_n=20)
        assert p["available"] is True
        assert p["step"] == 20
        # 期数 ≈ n_dates / 20（不重叠）
        assert p["periods"] <= len(score) // 20 + 1

    def test_top_beats_bench_when_factor_predicts(self):
        score, close = self._setup()
        p = R.portfolio_backtest(score, close, horizon=5, top_n=20)
        assert p["excess_mean"] > 0, p
        assert p["win_rate"] > 0.5


# ================================================================
# 4. 过热诊断（本次核心发现的可回归锁）
# ================================================================

class TestOverheat:
    def test_mask_improves_top_excess(self):
        """⭐ 诊断执行体：把「已涨一大段」的票从候选里剔掉，TOP 超额必须改善。

        用**可控矩阵**直接测 `_top_bottom_returns`（`overheat_diagnosis` 的执行体）：
          score = 涨幅幅度（幅度越大分越高）
          fwd   = -幅度 × 0.02（幅度越大，未来跌越多）
        ⇒ 不剔时 TOP 组合必然亏；剔掉幅度最大的一批后必然改善。
        （"mask 由 close 的 20 日涨幅算出来"这一层在真实面板已验收，
          这里只锁聚合逻辑，避免合成 close 的时序把测试搞脆。）
        """
        n_dates, n_codes = 30, 300
        dates = [f"d{i:03d}" for i in range(n_dates)]
        codes = [f"S{i}" for i in range(n_codes)]
        amp = np.linspace(0.0, 1.0, n_codes)
        score = pd.DataFrame(np.tile(amp, (n_dates, 1)), index=dates, columns=codes)
        fwd = pd.DataFrame(np.tile(-amp * 0.02, (n_dates, 1)),
                           index=dates, columns=codes)
        mask_ok = pd.DataFrame(np.tile(amp < 0.7, (n_dates, 1)),
                               index=dates, columns=codes)

        raw = R._top_bottom_returns(score, fwd, step=5, top_n=20)
        ex = R._top_bottom_returns(score, fwd, step=5, top_n=20, mask=mask_ok)
        assert raw["available"] and ex["available"]
        assert raw["top_excess"] < 0, raw
        assert ex["top_excess"] > raw["top_excess"], (raw, ex)
        # 反向（最低 20）在"幅度越大跌越多"时反而占优 ⇒ 两端都有效时的对照
        assert raw["bot_excess"] > 0, raw


# ================================================================
# 6. 可复现性（P1-1 / P2-1，2026-10-19 审查）
# ================================================================
#
# 事故背景：面板在 BaoStock 回填期间**一直在变**（分片 17→26，10-19 截面 400→
# 1000 只，pe_ttm 覆盖 12→…），而报告里没有任何字段能区分"结论变了"和
# "数据变了"；同时"矩阵缓存"只有 `pickle.load`、全仓没有一处 `dump` ——
# 所谓"30 秒"只在作者机器上成立（冷跑实测 135s）。

class TestReproducibility:
    def test_pivot_many_equals_pivot(self):
        """向量化建矩阵必须与 `DataFrame.pivot` **逐值相等**（含 NaN 与排序）。"""
        df = pd.DataFrame({
            "trade_date": ["2026-01-06", "2026-01-05", "2026-01-07",
                           "2026-01-05", "2026-01-06"],
            "stock_code": ["600000", "600000", "600000", "000001", "000001"],
            "close": [10.5, 10.0, np.nan, 20.0, 20.5],
            "open": [10.4, 9.9, 10.9, 19.9, 20.4],
        })
        fast = R._pivot_many(df, ("close", "open"))
        for col in ("close", "open"):
            ref = df.pivot(index="trade_date", columns="stock_code",
                           values=col).sort_index().sort_index(axis=1)
            pd.testing.assert_frame_equal(fast[col], ref)

    def test_drop_thin_dates_removes_partial_cross_section(self):
        """⭐ 回填当天的残截面（如 400 只 vs 正常 5000+）必须被剔除。

        只有 100 只的一天算出来的"全市场等权"和截面 IC 没有代表性，
        混进样本会让最后几期随机跳动，看起来像"结论变了"。
        """
        idx = ["d1", "d2", "d3"]
        cols = [f"S{i}" for i in range(100)]
        close = pd.DataFrame(1.0, index=idx, columns=cols)
        close.loc["d3", :] = np.nan
        close.loc["d3", cols[:2]] = 1.0          # d3 只有 2 只（残截面）
        m = {"close": close, "pct_chg": close * 0}
        out, info = R.drop_thin_dates(m, min_cross=50)
        assert info["dropped"] == ["d3"], info
        assert list(out["close"].index) == ["d1", "d2"]
        assert info["min_cross"] == 50

    def test_drop_thin_dates_is_noop_when_all_full(self):
        close = pd.DataFrame(1.0, index=["d1", "d2"], columns=[f"S{i}" for i in range(50)])
        out, info = R.drop_thin_dates({"close": close}, min_cross=10)
        assert info["dropped"] == [] and len(out["close"]) == 2

    def test_parts_signature_changes_when_part_changes(self, tmp_path, monkeypatch):
        """指纹必须对"分片被追加/改写"敏感（否则缓存会把旧数据当新的）。"""
        monkeypatch.setattr(R, "PARTS_DIR", str(tmp_path))
        (tmp_path / "p1.parquet").write_bytes(b"x" * 10)
        sig1 = R._parts_signature(100)
        (tmp_path / "p2.parquet").write_bytes(b"y" * 10)
        sig2 = R._parts_signature(100)
        assert sig1["parts"] == 1 and sig2["parts"] == 2
        assert sig1["signature"] != sig2["signature"]

    def test_cache_roundtrip_and_invalidation(self, tmp_path, monkeypatch):
        """⭐ 冷跑必须**写**缓存；指纹变了必须重建（不是无条件命中）。"""
        parts = tmp_path / "parts"
        parts.mkdir()
        monkeypatch.setattr(R, "PARTS_DIR", str(parts))
        cache = tmp_path / "cache.pkl"
        monkeypatch.setattr(R, "_CACHE_TMPL", str(cache))
        R.load_matrices.cache_clear()

        def _write_part(rows):
            pd.DataFrame(rows).to_parquet(parts / "p1.parquet", index=False)

        _write_part([
            {"stock_code": "600000", "trade_date": "2026-01-05", "close": 10.0,
             "open": 9.9, "preclose": 9.8, "pct_chg": 2.0, "amount": 1e7,
             "pe_ttm": 12.0},
            {"stock_code": "000001", "trade_date": "2026-01-05", "close": 20.0,
             "open": 19.9, "preclose": 19.5, "pct_chg": 2.5, "amount": 2e7,
             "pe_ttm": 8.0},
        ])
        m1 = R.load_matrices(365)
        assert cache.exists(), "冷跑后必须写出缓存（原实现只有 load，没有 dump）"
        assert m1["meta"]["parts"] == 1 and m1["meta"]["n_codes"] == 2

        # 指纹相同 ⇒ 命中（不再重建）
        R.load_matrices.cache_clear()
        m1b = R.load_matrices(365)
        assert m1b["meta"]["signature"] == m1["meta"]["signature"]

        # 面板变了（追加一只）⇒ 指纹变化 ⇒ 必须重建
        _write_part([
            {"stock_code": "600000", "trade_date": "2026-01-05", "close": 10.0,
             "open": 9.9, "preclose": 9.8, "pct_chg": 2.0, "amount": 1e7,
             "pe_ttm": 12.0},
            {"stock_code": "000001", "trade_date": "2026-01-05", "close": 20.0,
             "open": 19.9, "preclose": 19.5, "pct_chg": 2.5, "amount": 2e7,
             "pe_ttm": 8.0},
            {"stock_code": "600519", "trade_date": "2026-01-05", "close": 1600.0,
             "open": 1590.0, "preclose": 1580.0, "pct_chg": 1.3, "amount": 3e9,
             "pe_ttm": 30.0},
        ])
        R.load_matrices.cache_clear()
        m2 = R.load_matrices(365)
        assert m2["meta"]["signature"] != m1["meta"]["signature"]
        assert m2["meta"]["n_codes"] == 3, "指纹变化后必须重新读面板"
        R.load_matrices.cache_clear()


# ================================================================
# 7. 口径一致性（P2-2 / P2-3，2026-10-19 审查）
# ================================================================

class TestEntryTimingAndOverlap:
    def test_next_open_entry_uses_t_plus_1_open(self):
        """⭐ `entry="next_open"` = **T 日收盘出信号、T+1 开盘成交**。

        项目在批次 3 修过这个坑（`scripts/backtest_rule_based` 主循环用 T+1 开盘），
        回放若按 `close[t+h]/close[t]` 算就是"同日收盘成交"，两处纪律不一致。
        """
        idx = ["d1", "d2", "d3", "d4"]
        close = pd.DataFrame({"A": [10.0, 999.0, 999.0, 999.0]}, index=idx)
        open_ = pd.DataFrame({"A": [1.0, 20.0, 22.0, 24.0]}, index=idx)
        fwd = R.forward_returns(close, horizons=(2,), entry="next_open",
                                open_=open_)[2]
        # d1 信号 → 次日(d2)开盘 20 买入 → d4 开盘 24 卖出
        assert fwd.loc["d1", "A"] == pytest.approx(24 / 20 - 1)

    def test_next_open_falls_back_to_close_without_open(self):
        close = pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=["d1", "d2", "d3"])
        a = R.forward_returns(close, (1,), entry="next_open", open_=None)[1]
        b = R.forward_returns(close, (1,), entry="close")[1]
        pd.testing.assert_frame_equal(a, b)

    def test_ic_table_flags_overlapping_labels(self):
        """h > step ⇒ 相邻期标签重叠 ⇒ bootstrap CI 偏窄，必须**显式标记**。"""
        n, codes = 120, 60
        idx = [f"d{i}" for i in range(n)]
        cols = [f"S{i}" for i in range(codes)]
        rng = np.random.default_rng(0)
        base = pd.DataFrame(np.cumsum(rng.normal(size=(n, codes)), axis=0) * 0.01 + 10,
                            index=idx, columns=cols)
        f = R.compute_factors({"close": base})
        fwd = R.forward_returns(base, (5, 60))
        tbl = R.ic_table(f, fwd, step=5)
        assert tbl["momentum_20"][5]["overlapping"] is False
        assert tbl["momentum_20"][60]["overlapping"] is True

    def test_rank_ic_matches_pandas_spearman(self):
        """向量化 IC 与 `Series.corr(method="spearman")` 数值等价（1e-12 内）。"""
        rng = np.random.default_rng(7)
        n = 300
        idx = [f"S{i}" for i in range(n)]
        f = pd.Series(rng.normal(size=n), index=idx)
        r = pd.Series(rng.normal(size=n), index=idx)
        f.iloc[:20] = np.nan                    # 有缺失也必须一致
        ref_df = pd.concat([f, r], axis=1, keys=["f", "r"]).dropna()
        ref = float(ref_df["f"].corr(ref_df["r"], method="spearman"))
        assert R._rank_ic(f, r) == pytest.approx(ref, abs=1e-12)
