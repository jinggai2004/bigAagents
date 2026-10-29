# -*- coding: utf-8 -*-
"""IVOL（残差波动率）与流动性下限的结构性守护（2026-10-29）。

为什么单独一个文件
------------------
两处改动都属于"**算术上等价、行为上不等价**"的类型 —— 出错了分数照样是
0~100 的正常数字，页面看不出任何异常：

1. IVOL 取不到市场序列时会**静默退回**总波动率（分数完全正常，只是排序变了）；
2. 流动性阈值取不到时会**静默不过滤**（候选数看着"没变少"，其实门槛没生效）。

所以这里不测"分数对不对"，而是钉住**降级必须可见**与**边界必须显式**。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core import indicators
from core.screener import StockScreener


# ================================================================
# indicators.residual_vol
# ================================================================

def _make_pair(k: int = 60, beta: float = 1.2, seed: int = 7):
    """构造个股/市场两条收盘价序列（个股 = β·市场 + 独立噪声）。"""
    rng = np.random.default_rng(seed)
    rm = rng.normal(0.0005, 0.012, k)
    ri = beta * rm + rng.normal(0.0, 0.008, k)
    mkt = np.cumprod(1.0 + rm) * 1000.0
    stk = np.cumprod(1.0 + ri) * 20.0
    return stk, mkt


def test_residual_vol_matches_closed_form():
    """与"总体矩 + β²·var_m 分解"的闭式解逐值相等（口径不许漂）。"""
    stk, mkt = _make_pair()
    got = indicators.residual_vol(stk, mkt, 20)
    assert got is not None and got > 0

    rx = np.diff(stk[-21:]) / stk[-21:-1]
    rm = np.diff(mkt[-21:]) / mkt[-21:-1]
    vx, vm = rx.var(ddof=0), rm.var(ddof=0)
    cov = ((rx - rx.mean()) * (rm - rm.mean())).mean()
    want = math.sqrt(max(vx - cov ** 2 / vm, 0.0)) * math.sqrt(indicators.TRADING_DAYS)
    assert got == pytest.approx(want, rel=1e-12)


def test_residual_vol_never_exceeds_total_vol():
    """残差方差 = 总方差 − β²·var_m ≤ 总方差（数学恒等，用来挡符号写反）。"""
    rng = np.random.default_rng(11)
    for beta in (0.0, 0.5, 1.0, 2.0):
        stk, mkt = _make_pair(beta=beta, seed=int(beta * 10) + 1)
        iv = indicators.residual_vol(stk, mkt, 20)
        tv = indicators.annualized_vol(stk, 20)
        assert iv <= tv + 1e-12, f"β={beta} 时 IVOL 超过总波动率"


def test_residual_vol_zero_beta_equals_total_vol():
    """个股与市场完全无关 ⇒ IVOL ≈ 总波动率（只差 β 项的抽样误差）。

    ⚠️ 容差不能给太紧：60 个点的样本里 cov 偶然非零，IVOL 会低几个百分点 ——
    这条测的是"β≈0 时不至于差一个量级"，不是数值恒等（数值恒等在
    `test_residual_vol_matches_closed_form` 里逐位钉死）。
    """
    rng = np.random.default_rng(3)
    stk = np.cumprod(1.0 + rng.normal(0, 0.01, 200)) * 10.0
    mkt = np.cumprod(1.0 + rng.normal(0, 0.01, 200)) * 1000.0
    iv = indicators.residual_vol(stk, mkt, 60)
    tv = indicators.annualized_vol(stk, 60)
    assert iv == pytest.approx(tv, rel=0.15)


@pytest.mark.parametrize("stk,mkt,n", [
    (None, [1.0, 2.0], 20),                       # 个股缺失
    ([1.0, 2.0], None, 20),                       # 市场缺失
    ([1.0, 2.0, 3.0], [1.0, 2.0], 2),             # 长度不等
    ([1.0] * 30, [1.0] * 30, 20),                 # 市场零波动 ⇒ β 不可识别
    ([float("nan")] * 30, [1.0 + i for i in range(30)], 20),   # 全 NaN
])
def test_residual_vol_returns_none_on_bad_input(stk, mkt, n):
    """缺就是缺：返回 None，绝不返回 0（0 会被当成"最低波动"拿满分）。"""
    assert indicators.residual_vol(stk, mkt, n) is None


def test_residual_vol_needs_80pct_valid():
    """有效点 < 80% 判 None —— 与验证脚本 `n >= w*0.8` 同一口径。

    20 日窗口容 4 天缺失（16 点），第 5 天缺失就要判不可算。
    """
    stk, mkt = _make_pair(k=40)
    stk = np.array(stk, dtype=float)
    mkt = np.array(mkt, dtype=float)
    stk[-4:] = np.nan                            # 缺 4 天 ⇒ 16/20 仍可算
    assert indicators.residual_vol(stk, mkt, 20) is not None
    stk[-5:] = np.nan                            # 缺 5 天 ⇒ 15/20 不可算
    assert indicators.residual_vol(stk, mkt, 20) is None


# ================================================================
# screener._ivol：必须能在测试里**完全不碰真实面板**
# ================================================================

def _df_with_index(idx, close=None):
    if close is None:
        close = [10.0 + 0.1 * i for i in range(len(idx))]
    return pd.DataFrame({"Close": list(close)}, index=idx)


def test_ivol_skips_non_date_index_without_loading_market_factor(monkeypatch):
    """RangeIndex（合成用例）⇒ 直接 None，且**不得**触发市场因子构建。

    ⚠️ 这条守的是"测试不耦合真实环境"：若 `_ivol` 先加载市场序列再判索引，
    单测会去读 14 个真实分片、还会往 data/panel 写缓存 —— 一次 6.6 秒 + 污染仓库。
    """
    s = StockScreener()
    from core import market_factor

    def _boom(*a, **k):
        raise AssertionError("不该加载市场因子")

    monkeypatch.setattr(market_factor, "equal_weight_index", _boom)
    df = pd.DataFrame({"Close": [10.0 + 0.1 * i for i in range(40)]})   # RangeIndex
    assert s._ivol(df) is None
    assert s._mkt_index_cache is None, "预检失败时不该把缓存置位"


def test_ivol_uses_date_index_and_returns_percent(monkeypatch):
    """日期索引 + 可对齐的市场序列 ⇒ 返回**百分数**（与 `_annualized_vol` 同单位）。"""
    s = StockScreener()
    stk, mkt = _make_pair(k=80)
    idx = pd.bdate_range("2026-06-01", periods=80)
    level = pd.Series(mkt, index=idx.strftime("%Y-%m-%d"))
    s._mkt_index_cache = level
    got = s._ivol(_df_with_index(idx, stk))
    assert got is not None
    # 与直接调 indicators 的结果一致（×100 只在调用点做一次）
    want = indicators.residual_vol(stk, mkt, 20) * 100.0
    assert got == pytest.approx(want, rel=1e-9)
    assert 10 <= got <= 100, "百分数口径下不该是 0.xx"


def test_ivol_none_when_axis_too_short(monkeypatch):
    """个股区间内市场交易日不足 21 个 ⇒ None（退回总波动率，不硬算）。"""
    s = StockScreener()
    idx = pd.bdate_range("2026-06-01", periods=80)
    s._mkt_index_cache = pd.Series(np.linspace(1000, 1100, 5),   # 只有 5 天
                                   index=idx[:5].strftime("%Y-%m-%d"))
    assert s._ivol(_df_with_index(idx)) is None


def test_volatility_reports_basis(monkeypatch):
    """口径必须写进结果：用了 IVOL 就标 ivol，退回总波动率就标 total。"""
    s = StockScreener()
    df = pd.DataFrame({"Close": [10.0 + 0.01 * i for i in range(30)]})    # RangeIndex
    r = s._score_volatility(df, 10, pct=0.9, peers=40)
    assert r["value"]["vol_basis"] == "total"
    assert s.last_vol_basis["total"] >= 1

    idx = pd.bdate_range("2026-06-01", periods=30)
    stk, mkt = _make_pair(k=30)
    s._mkt_index_cache = pd.Series(mkt, index=idx.strftime("%Y-%m-%d"))
    df2 = pd.DataFrame({"Close": stk}, index=idx)
    r2 = s._score_volatility(df2, 10, pct=0.9, peers=40)
    assert r2["value"]["vol_basis"] == "ivol"
    assert "残差波动率" in r2["detail"]
    assert s.last_vol_basis["ivol"] >= 1


# ================================================================
# 流动性下限
# ================================================================

def _fake_threshold(monkeypatch, overall=5e7, by_board=None):
    monkeypatch.setattr(StockScreener, "_liquidity_threshold",
                        staticmethod(lambda: {"date": "2026-10-28",
                                              "pct": 0.30,
                                              "overall": overall,
                                              "by_board": dict(by_board or {}),
                                              "n_market": 5000}))


def test_liq_floor_drops_below_threshold_and_lists_them(monkeypatch):
    """低于阈值必须剔除，且**被剔除的票可枚举**（不能只看到候选变少）。"""
    _fake_threshold(monkeypatch)
    s = StockScreener()
    s._kline_cache = {
        "600000": pd.DataFrame({"amount": [1e8] * 3}),   # 1 亿 > 0.5 亿
        "600001": pd.DataFrame({"amount": [1e6] * 3}),   # 100 万 < 0.5 亿
    }
    cands = [{"股票代码": c, "股票名称": c} for c in
             ["600000", "600001"] + [f"60{i:04d}" for i in range(2, 22)]]
    for c in cands[2:]:
        s._kline_cache[c["股票代码"]] = pd.DataFrame({"amount": [2e8] * 3})
    keep, meta = s._apply_liquidity_floor(cands)
    assert meta["applied"] is True
    assert [d["股票代码"] for d in meta["dropped"]] == ["600001"]
    assert all(c["股票代码"] != "600001" for c in keep)
    assert "1 只" in meta["note"] or "剔除 1" in meta["note"]


def test_liq_floor_uses_board_specific_threshold(monkeypatch):
    """北交所用**自己板块**的阈值，不被全市场阈值整段剔掉。

    ⚠️ 这条是真实事故的回归：全市场 30% 分位 = 0.49 亿，而北交所日成交
    天然在 400~1600 万 ⇒ 一条全市场门槛等于"静默取消北交所的选股权"。
    2026-10-29 首次上线时 106 只候选被剔的 4 只里 2 只是北交所。
    """
    # 全市场阈值 0.5 亿；北交所自己 0.1 亿
    _fake_threshold(monkeypatch, overall=5e7, by_board={"北交所": 1e7})
    s = StockScreener()
    s._kline_cache = {
        "920187": pd.DataFrame({"amount": [2e7] * 3}),   # 北交所 2000 万
        "600861": pd.DataFrame({"amount": [3e7] * 3}),   # 主板 3000 万 < 0.5 亿 ⇒ 剔
    }
    cands = [{"股票代码": "920187", "股票名称": "通领科技"},
             {"股票代码": "600861", "股票名称": "北京人力"}]
    cands += [{"股票代码": f"60{i:04d}", "股票名称": ""} for i in range(2, 22)]
    for c in cands[2:]:
        s._kline_cache[c["股票代码"]] = pd.DataFrame({"amount": [1e9] * 3})
    keep, meta = s._apply_liquidity_floor(cands)
    codes = [c["股票代码"] for c in keep]
    assert "920187" in codes, "北交所 2000 万应通过（板块内阈值 1000 万）"
    assert "600861" not in codes, "主板 3000 万应被剔（全市场阈值 5000 万）"
    assert meta["by_board"] == {"主板沪市": 0.5}


def test_liq_floor_missing_amount_keeps_candidate(monkeypatch):
    """成交额取不到 ⇒ **保留**。"没数据"不等于"不流动"。"""
    _fake_threshold(monkeypatch)
    s = StockScreener()
    s._kline_cache = {}
    cands = [{"股票代码": f"60{i:04d}", "股票名称": ""} for i in range(12)]
    keep, meta = s._apply_liquidity_floor(cands)
    assert meta["applied"] is True
    assert len(keep) == 12 and meta["dropped"] == []


def test_liq_floor_without_snapshot_does_not_filter(monkeypatch):
    """阈值取不到（快照表空/离线）⇒ 不过滤 + note 写明未生效。"""
    monkeypatch.setattr(StockScreener, "_liquidity_threshold",
                        staticmethod(lambda: None))
    s = StockScreener()
    cands = [{"股票代码": f"60{i:04d}", "股票名称": ""} for i in range(30)]
    keep, meta = s._apply_liquidity_floor(cands)
    assert meta["applied"] is False
    assert meta["threshold"] is None
    assert "未生效" in meta["note"]
    assert len(keep) == 30


def test_liq_floor_gives_up_when_pool_too_small(monkeypatch):
    """过滤后不足 10 只 ⇒ 放弃过滤（池子小了再砍只剩噪声）。"""
    _fake_threshold(monkeypatch)
    s = StockScreener()
    s._kline_cache = {f"60{i:04d}": pd.DataFrame({"amount": [1e6] * 3}) for i in range(12)}
    cands = [{"股票代码": f"60{i:04d}", "股票名称": ""} for i in range(12)]
    keep, meta = s._apply_liquidity_floor(cands)
    assert meta["applied"] is False
    assert len(keep) == 12
    assert "放弃过滤" in meta["note"]


def test_liq_floor_can_be_disabled(monkeypatch):
    """开关关闭时连阈值都不去查（省一次 SQL，也让"关了"这件事一目了然）。"""
    monkeypatch.setenv("BIGA_LIQ_FLOOR", "0")
    monkeypatch.setattr(StockScreener, "_liquidity_threshold",
                        staticmethod(lambda: (_ for _ in ()).throw(
                            AssertionError("关闭时不该查阈值"))))
    s = StockScreener()
    cands = [{"股票代码": "600000", "股票名称": ""}]
    keep, meta = s._apply_liquidity_floor(cands)
    assert meta["enabled"] is False and meta["applied"] is False
    assert len(keep) == 1


# ================================================================
# 行业内分位（2026-10-29：治 IVOL 的行业偏向）
# ================================================================

def test_percentile_by_industry_ranks_within_group(monkeypatch):
    """同行业内部排名，而不是全候选排名。

    4 只股票两两同行业：银行组内 600000 波动更低 → 1.0，600036 → 0.0；
    如果错写成全池分位，600000 会拿到 1.0、600036 会拿到约 0.67（而不是 0）。
    """
    s = StockScreener()
    monkeypatch.setattr(s, "_industry_map",
                        lambda: {"600000": "银行", "600036": "银行",
                                 "600519": "白酒", "000858": "白酒"})
    out, st = s._percentile_by_industry(
        {"600000": 10.0, "600036": 20.0, "600519": 30.0, "000858": 40.0},
        higher_is_better=False, min_group=2)
    assert out["600000"] == 1.0 and out["600036"] == 0.0
    assert out["600519"] == 1.0 and out["000858"] == 0.0, "白酒组内 600519 更优"
    assert st["groups"] == 2 and st["industry"] == 4 and st["pool"] == 0


def test_percentile_by_industry_small_group_merges_into_pool(monkeypatch):
    """组内样本 < min_group ⇒ 不单独分位，合并进"其余"池（不能丢分位，
    否则 `_score_single` 会把这些股票判 unknown，白白拉低覆盖度）。"""
    s = StockScreener()
    monkeypatch.setattr(s, "_industry_map",
                        lambda: {"600000": "银行", "600036": "银行"})   # 只有 2 只
    vals = {"600000": 10.0, "600036": 20.0, "600519": 30.0, "000858": 40.0}
    out, st = s._percentile_by_industry(vals, higher_is_better=False, min_group=6)
    assert st["industry"] == 0 and st["pool"] == 4
    assert set(out) == set(vals), "四只都必须拿到分位（池内统一排）"
    assert out["600000"] == 1.0 and out["000858"] == 0.0


def test_percentile_by_industry_missing_map_falls_back(monkeypatch):
    """行业映射为空（快照不可用）⇒ 全部进池，行为等价旧版但不崩。"""
    s = StockScreener()
    monkeypatch.setattr(s, "_industry_map", lambda: {})
    vals = {f"60{i:04d}": float(i) for i in range(10)}
    out, st = s._percentile_by_industry(vals, higher_is_better=False, min_group=6)
    assert st["industry"] == 0 and st["pool"] == 10 and st["missing"] == 10
    assert len(out) == 10


def test_cross_section_scope_is_visible(monkeypatch):
    """作用域必须带出来：industry（行业映射可用）/ pool（不可用）。
    全部落进 pool 时分数看起来完全正常，唯一线索就是这个字段。"""
    s = StockScreener()
    codes = [f"60{i:04d}" for i in range(12)]
    s._info_cache = {c: {"pe_ratio": 20.0 + i} for i, c in enumerate(codes)}
    s._kline_cache = {c: None for c in codes}
    # 实例属性直接赋 lambda（不要包 staticmethod：描述符只对**类**属性生效，
    # 赋到实例上会变成一个不可调用的 staticmethod 对象）
    s._annualized_vol = lambda df: None

    monkeypatch.setattr(s, "_industry_map",
                        lambda: {c: ("银行" if i % 2 == 0 else "白酒")
                                 for i, c in enumerate(codes)})
    # 直接给 IVOL 一组假值，绕开真实市场序列；值要各不相同，否则分位会全并列
    _box = {"i": 0}

    def _fake_ivol(df):
        _box["i"] += 1
        return 10.0 + _box["i"] % 9

    monkeypatch.setattr(s, "_ivol", _fake_ivol)
    ctx = s._build_cross_section([{"股票代码": c} for c in codes])
    assert ctx["vol_scope"] == "industry"
    assert ctx["vol_groups"] == 2

    monkeypatch.setattr(s, "_industry_map", lambda: {})
    ctx2 = s._build_cross_section([{"股票代码": c} for c in codes])
    assert ctx2["vol_scope"] == "pool", "行业映射不可用时必须显式标 pool"
