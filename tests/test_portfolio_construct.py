# -*- coding: utf-8 -*-
"""`core/portfolio_construct.py` 的行为契约。

这个模块最危险的失败模式是**静默**：闸门看起来在跑，实际全放过（或者全拦死），
以及涨跌停被写死 10% 把双创/北交所判错。所以用例围绕三件事：
1. 涨跌停必须按板块+日期（用创业板 15% 涨幅做对照，防"写死 1.1"）；
2. 每个拒绝原因都要能对上账（sum(rejected) + kept == total）；
3. 缓冲区真的在压换手，但压不住时必须报 relaxed 而不是假装成功。
"""
from __future__ import annotations

import pandas as pd
import pytest

from core import portfolio_construct as pc


def _df(rows: list[dict]) -> pd.DataFrame:
    base = {"code": "600000", "price": 10.0, "prev_close": 10.0,
            "amount": 2e8, "volume": 1e6, "is_st": False}
    return pd.DataFrame([{**base, **r} for r in rows])


# ── 1. 涨跌停：必须按板块 + 日期 ─────────────────────────────

def test_limit_up_is_board_aware():
    """主板 +10% 算涨停；**创业板 +15% 不算**（那是 20% 的板）。"""
    df = _df([
        {"code": "600000", "price": 11.0},      # 主板 10% ⇒ 涨停
        {"code": "300750", "price": 11.5},      # 创业板 15% ⇒ 未到 20%
        {"code": "920001", "price": 11.5},      # 北交所 15% ⇒ 未到 30%
    ])
    kept, st = pc.apply_gates(df, pc.GateConfig())
    assert st["rejected"].get(pc.REJ_LIMIT_UP) == 1
    assert list(kept["code"]) == ["300750", "920001"]


def test_limit_up_respects_st_regime_date():
    """主板 ST：2026-06 是 5%（+6% 即涨停），2026-10-30 起是 10%（+6% 不算）。"""
    df = _df([{"code": "600000", "price": 10.6, "prev_close": 10.0, "is_st": True}])
    _, old = pc.apply_gates(df, pc.GateConfig(), trade_date="2026-06-01")
    _, new = pc.apply_gates(df, pc.GateConfig(), trade_date="2026-10-30")
    assert old["rejected"].get(pc.REJ_LIMIT_UP) == 1
    assert new["rejected"].get(pc.REJ_LIMIT_UP, 0) == 0


def test_missing_price_columns_are_reported_not_silent():
    """缺 prev_close ⇒ 涨停这道闸必须进 missing_inputs，**不能假装通过又不记录**。"""
    df = pd.DataFrame([{"code": "600000", "price": 11.0, "volume": 1e6,
                        "amount": 2e8}])
    kept, st = pc.apply_gates(df, pc.GateConfig())
    assert any("涨停" in name for name, _ in st["missing_inputs"])
    assert len(kept) == 1                     # 不因为缺列就全砍


# ── 2. 停牌与成交额 ─────────────────────────────────────────

def test_suspended_is_rejected():
    df = _df([{"code": "A", "volume": 0.0}, {"code": "B", "volume": None},
              {"code": "C", "volume": 1e6}])
    kept, st = pc.apply_gates(df, pc.GateConfig())
    assert st["rejected"].get(pc.REJ_SUSPENDED) == 2
    assert list(kept["code"]) == ["C"]


def test_amount_gate_both_sides():
    """下限保容量、上沿剔除极端高流动性端 —— 两侧都要能生效。

    ⚠️ 分位需要 ≥20 只存活才可估（模块的显式下限），所以这里造 40 只：
    1 只低于下限 + 1 只极端高 + 38 只正常。
    """
    amounts = [1e6, 1e10] + [1e8 + i * 1e6 for i in range(38)]
    df = _df([{"code": f"S{i}", "amount": a} for i, a in enumerate(amounts)])
    kept, st = pc.apply_gates(
        df, pc.GateConfig(min_amount_yuan=5e7, max_amount_pct=0.95))
    assert st["rejected"].get(pc.REJ_MIN_AMOUNT) == 1        # S0
    # ⚠️ 0.95 分位在 39 只上会切掉 1~2 只（分位是连续的），关键是**极端值必须被切**
    assert st["rejected"].get(pc.REJ_MAX_AMOUNT, 0) >= 1
    assert "S1" not in set(kept["code"]) and "S0" not in set(kept["code"])
    assert float(kept["amount"].max()) < 1e10


def test_stats_always_reconcile():
    """拒绝原因之和 + 存活 == 总数（只记第一次被拒，所以能对上账）。"""
    df = _df([{"code": f"S{i}", "price": 11.0 if i % 2 else 10.0,
               "volume": 0.0 if i % 3 == 0 else 1e6,
               "amount": 1e3 if i % 5 == 0 else 2e8} for i in range(40)])
    _, st = pc.apply_gates(df, pc.GateConfig(min_amount_yuan=5e7, min_price=5.0))
    assert sum(st["rejected"].values()) + st["kept"] == st["total"] == 40


# ── 3. 因子闸门：high/low 与 NaN 语义 ───────────────────────

def test_factor_gate_high_drops_top_decile():
    vals = list(range(100))
    df = _df([{"code": f"S{i}", "upshd20": float(v)} for i, v in enumerate(vals)])
    kept, st = pc.apply_gates(
        df, pc.GateConfig(factor_gates=(("upshd20", "high", 0.90),)))
    assert st["rejected"].get(f"{pc.REJ_FACTOR_HIGH}:upshd20") == pytest.approx(10, abs=2)
    assert float(kept["upshd20"].max()) < 99


def test_factor_gate_low_drops_bottom_decile():
    df = _df([{"code": f"S{i}", "f": float(v)} for i, v in enumerate(range(100))])
    kept, st = pc.apply_gates(df, pc.GateConfig(factor_gates=(("f", "low", 0.90),)))
    assert st["rejected"].get(f"{pc.REJ_FACTOR_LOW}:f") == pytest.approx(10, abs=2)
    assert float(kept["f"].min()) > 0


def test_factor_gate_nan_passes_but_is_counted():
    """因子缺失 ⇒ 通过（不当极端值砍），但必须计数，否则覆盖率永远看着是 100%。"""
    rows = [{"code": f"S{i}", "f": float(v)} for i, v in enumerate(range(50))]
    rows += [{"code": "N1", "f": None}, {"code": "N2", "f": None}]
    kept, st = pc.apply_gates(_df(rows),
                              pc.GateConfig(factor_gates=(("f", "high", 0.90),)))
    assert st["factor_nan"]["f"] == 2
    assert {"N1", "N2"} <= set(kept["code"])


def test_factor_gate_bad_side_raises():
    df = _df([{"code": "A", "f": 1.0}])
    with pytest.raises(ValueError):
        pc.apply_gates(df, pc.GateConfig(factor_gates=(("f", "middle", 0.9),)))


def test_empty_input_is_safe():
    out, st = pc.apply_gates(pd.DataFrame(), pc.GateConfig())
    assert st["total"] == 0 and st["kept"] == 0


# ── 4. 缓冲区调仓 ───────────────────────────────────────────

def test_buffer_keeps_old_holding_that_slipped_a_bit():
    ranked = [f"s{i}" for i in range(20)]
    # prev 含 s12（排第 13）：k=10, buffer=15 ⇒ 第 13 名仍在缓冲区内 ⇒ 留
    out = pc.select_with_buffer(ranked, ["s0", "s12"], k=10, buffer_mult=1.5)
    assert "s12" in out["codes"] and "s12" in out["kept"]
    assert out["buffer"] == 15
    assert len(out["codes"]) == 10


def test_buffer_drops_holding_outside_buffer():
    ranked = [f"s{i}" for i in range(40)]
    out = pc.select_with_buffer(ranked, ["s30"], k=10, buffer_mult=1.5)  # 第 31 名
    assert "s30" not in out["codes"]
    assert "s30" in out["sold"]


def test_holding_missing_from_ranked_is_sold():
    """不在 ranked 里 = 今天没过闸门（停牌/涨停/被剔除）⇒ 卖。"""
    out = pc.select_with_buffer(["a", "b", "c"], ["zzz"], k=2)
    assert "zzz" in out["sold"]


def test_turnover_is_single_side_fraction():
    ranked = [f"s{i}" for i in range(30)]
    out = pc.select_with_buffer(ranked, [], k=10, buffer_mult=1.5)
    assert out["turnover"] == 1.0                    # 从空仓建仓 = 100% 换手
    out2 = pc.select_with_buffer(ranked, ranked[:10], k=10, buffer_mult=1.5)
    assert out2["turnover"] == 0.0                   # 完全不动
    assert out2["kept"] == ranked[:10]


def test_small_pool_sets_relaxed_flag():
    out = pc.select_with_buffer(["a", "b"], [], k=5)
    assert out["relaxed"] is True and len(out["codes"]) == 2
    assert pc.select_with_buffer([f"s{i}" for i in range(10)], [], k=5)["relaxed"] is False


@pytest.mark.parametrize("k,bm", [(0, 1.5), (-1, 1.5), (5, 0.5)])
def test_invalid_params_raise(k, bm):
    with pytest.raises(ValueError):
        pc.select_with_buffer(["a"], [], k=k, buffer_mult=bm)


# ── 5. 权重 ─────────────────────────────────────────────────

def test_equal_weights_empty_is_empty_not_zero():
    assert pc.equal_weights([]) == {}
    assert pc.equal_weights(["a", "b"]) == {"a": 0.5, "b": 0.5}


def test_inverse_vol_tilts_to_low_vol_and_sums_to_one():
    w = pc.inverse_vol_weights(["a", "b"], {"a": 0.2, "b": 0.4})
    assert w["a"] > w["b"]
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


def test_inverse_vol_cap_binds():
    w = pc.inverse_vol_weights(["a", "b", "c"],
                               {"a": 0.01, "b": 1.0, "c": 1.0}, cap=0.5)
    assert max(w.values()) <= 0.5 + 1e-9
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


def test_inverse_vol_missing_uses_median_not_dropped():
    """缺波动率的票用中位数补，**不剔除**（剔除=隐性按数据完整度选股）。"""
    w = pc.inverse_vol_weights(["a", "b", "c"], {"a": 0.2, "b": 0.4})
    assert set(w) == {"a", "b", "c"}
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
