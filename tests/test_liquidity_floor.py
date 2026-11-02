# -*- coding: utf-8 -*-
"""流动性下限（CH-3）的共享实现测试。

这个模块的存在理由：线上选股和回测**必须用同一条线**。原来线上有、回测没有，
两边在选不同的票——而"回测选出的正是线上会剔掉的票"这种错位不像 bug，
只像"策略变差了"。

守四类契约：
1. 阈值口径（30% 分位 / **按板块** / 板块样本 <100 退回全市场）；
2. 三条边界（判据缺失不剔 / 个股缺额保留 / 池子太小放弃）；
3. `applied` 的语义是"**闸门生效**"而不是"剔了票"（页面标记依赖它）；
4. 线上 `screener` 与回测走同一份常量与函数（防漂移哨兵）。
"""
import numpy as np
import pandas as pd
import pytest

from core import liquidity_floor as lf


def _amounts(n_hu=200, n_cy=200, base=1e8):
    """造两只板块的成交额：沪市 60xxxx、创业板 300xxx，逐只递增。"""
    out = {}
    for i in range(n_hu):
        out[f"60{i:04d}"] = base + i * 1e6
    for i in range(n_cy):
        out[f"30{i:04d}"] = base / 2 + i * 5e5
    return out


# ================================================================
# 1. 阈值口径
# ================================================================

def test_thresholds_are_percentile_within_board():
    """**按板块**各算 30% 分位：创业板的阈值不该被沪市的大额拉高。"""
    amt = _amounts()
    th = lf.build_thresholds(amt)
    assert th is not None
    hu = sorted(v for c, v in amt.items() if c.startswith("60"))
    cy = sorted(v for c, v in amt.items() if c.startswith("30"))
    assert th["by_board"]["主板沪市"] == pytest.approx(hu[int(len(hu) * 0.30)])
    assert th["by_board"]["创业板"] == pytest.approx(cy[int(len(cy) * 0.30)])
    # 创业板整体更小，所以它的阈值必须低于全市场阈值（否则就是"一条线通吃"）
    assert th["by_board"]["创业板"] < th["overall"]


def test_small_board_falls_back_to_overall():
    """板块样本 < 100 ⇒ 该板块不设阈值，退回全市场（保证不至于整体免检）。"""
    amt = _amounts(n_hu=200, n_cy=50)     # 创业板只有 50 只
    th = lf.build_thresholds(amt)
    assert "创业板" not in th["by_board"]
    assert "主板沪市" in th["by_board"]
    # 该板块的票走全市场阈值
    assert lf.limit_for("300000", th) == th["overall"]


def test_too_few_samples_returns_none():
    """全市场样本 < 100 ⇒ None（调用方据此"不过滤"）。"""
    assert lf.build_thresholds({f"60{i:04d}": 1e8 for i in range(50)}) is None
    assert lf.build_thresholds({}) is None


def test_zero_and_none_amounts_ignored():
    amt = _amounts()
    amt["600000"] = 0.0
    amt["600001"] = None
    th = lf.build_thresholds(amt)
    assert th["n_market"] == len(amt) - 2


# ================================================================
# 2. 三条边界
# ================================================================

def test_missing_threshold_does_not_filter():
    """边界 1：判据缺失（th=None）⇒ 原样放行，且明确标未生效。"""
    items = [("600000", 1e6), ("300001", 1e5)]
    res = lf.split_by_floor(items, None)
    assert sorted(res["kept"]) == ["300001", "600000"]
    assert res["applied"] is False
    assert "未生效" in res["note"]


def test_missing_amount_keeps_that_code():
    """边界 2：个股成交额取不到 ⇒ 保留。"没数据"不等于"不流动"。"""
    th = lf.build_thresholds(_amounts())
    # ⚠️ 其余票的成交额必须**高于板块阈值**（沪市分位 ≈1.6 亿），
    #    否则会一起被剔、剩余 <10 只 ⇒ 触发"放弃过滤"，测的就不是边界 2 了
    items = [("600900", 1e4)] + [(f"60{i:04d}", 3e8 + i * 1e7) for i in range(20)]
    items.append(("600999", None))          # 取不到
    res = lf.split_by_floor(items, th)
    assert res["applied"] is True
    assert "600999" in res["kept"]
    assert "600900" in [d["code"] for d in res["dropped"]]


def test_pool_too_small_gives_up():
    """边界 3：过滤后 < 10 只 ⇒ 放弃过滤（池子太小，再砍只剩噪声）。"""
    th = {"pct": 0.30, "overall": 1e9, "by_board": {}, "n_market": 5000}
    items = [(f"60{i:04d}", 1e7) for i in range(15)]     # 全部低于 10 亿门槛
    res = lf.split_by_floor(items, th)
    assert len(res["kept"]) == 15, "应放弃过滤、原样返回"
    assert res["applied"] is False
    assert "放弃过滤" in res["note"]


# ================================================================
# 3. applied 语义（页面标记依赖它）
# ================================================================

def test_applied_true_even_when_nothing_dropped():
    """生效但一只都没到门槛 ⇒ `applied=True`。

    混成"剔了才算生效"会让页面显示"流动性下限未生效"，
    而实际它在跑——排查时会去查数据源，方向全错。
    """
    th = {"pct": 0.30, "overall": 1.0, "by_board": {}, "n_market": 5000}
    items = [(f"60{i:04d}", 1e8) for i in range(20)]
    res = lf.split_by_floor(items, th)
    assert res["applied"] is True
    assert res["dropped"] == []


# ================================================================
# 4. 月度面板版（回测用）
# ================================================================

def test_monthly_panel_drops_30pct_per_month(monkeypatch):
    monkeypatch.delenv("BIGA_LIQ_FLOOR", raising=False)
    rows = []
    for m in ("2025-01-31", "2025-02-28"):
        for i in range(200):
            rows.append({"month_ts": pd.Timestamp(m), "code": f"60{i:04d}",
                         "amount_yuan": 1e8 + i * 1e6})
    df = pd.DataFrame(rows)
    out, meta = lf.filter_monthly_panel(df)
    assert meta["applied"] is True and meta["months"] == 2
    assert len(out) == pytest.approx(len(df) * 0.70, abs=4)
    # 每月都剩 70% 左右，而不是某一个月被砍光
    per_month = out.groupby("month_ts").size()
    assert all(135 <= n <= 145 for n in per_month), per_month.to_dict()


def test_monthly_panel_skips_month_without_threshold(monkeypatch):
    """某月样本不足（<100 只）⇒ 该月不剔，不影响其它月。"""
    monkeypatch.delenv("BIGA_LIQ_FLOOR", raising=False)
    big = [{"month_ts": pd.Timestamp("2025-01-31"), "code": f"60{i:04d}",
            "amount_yuan": 1e8 + i * 1e6} for i in range(200)]
    small = [{"month_ts": pd.Timestamp("2025-02-28"), "code": f"60{i:04d}",
              "amount_yuan": 1e7} for i in range(30)]
    out, meta = lf.filter_monthly_panel(pd.DataFrame(big + small))
    assert meta["months"] == 1
    assert (out["month_ts"] == pd.Timestamp("2025-02-28")).sum() == 30, "小样本月应原样保留"


def test_monthly_panel_disabled_by_env(monkeypatch):
    monkeypatch.setenv("BIGA_LIQ_FLOOR", "0")
    df = pd.DataFrame([{"month_ts": pd.Timestamp("2025-01-31"), "code": "600000",
                        "amount_yuan": 1e6}])
    out, meta = lf.filter_monthly_panel(df)
    assert len(out) == 1 and meta["applied"] is False


def test_monthly_panel_missing_amount_column_does_not_filter():
    """缺成交额列 ⇒ 不剔（边界 1 的同一逻辑），并写明原因。"""
    df = pd.DataFrame([{"month_ts": pd.Timestamp("2025-01-31"), "code": "600000"}])
    out, meta = lf.filter_monthly_panel(df)
    assert len(out) == 1
    assert "缺" in meta["note"]


# ================================================================
# 5. 防漂移哨兵：线上与回测必须是同一份口径
# ================================================================

def test_screener_uses_shared_constants():
    """`StockScreener.LIQ_FLOOR_PCT` 必须就是共享模块那个值。

    如果有人为了"临时调一下线上阈值"在 screener 里另写一个 0.3，
    回测与线上就会永久错位——这条会在那一刻变红。
    """
    from core.screener import StockScreener
    assert StockScreener.LIQ_FLOOR_PCT == lf.LIQ_FLOOR_PCT


def test_screener_floor_switch_delegates_to_shared(monkeypatch):
    from core.screener import StockScreener
    monkeypatch.setenv("BIGA_LIQ_FLOOR", "0")
    assert StockScreener._liq_floor_enabled() is False
    monkeypatch.setenv("BIGA_LIQ_FLOOR", "1")
    assert StockScreener._liq_floor_enabled() is True


def test_screener_threshold_equals_shared_function(monkeypatch):
    """给定同一批成交额，`_liquidity_threshold()` 的结果必须与共享函数逐字相同。"""
    from core.screener import StockScreener

    amt = _amounts(n_hu=200, n_cy=200)
    rows = [{"stock_code": c, "amount": v} for c, v in amt.items()]

    class _R(dict):
        """同时支持取行与取列表的最简假结果。"""

        def fetchone(self):
            return self

        def fetchall(self):
            return self.get("__rows__", [])

    class _Cur:
        def execute(self, sql, params=()):
            if "MAX(snapshot_date)" in sql:
                return _R(d="2026-10-30")
            return _R(__rows__=[_R(**r) for r in rows])

        def close(self):
            pass

    from db import store
    monkeypatch.setattr(store, "get_connection", lambda: _Cur())
    got = StockScreener._liquidity_threshold()
    want = lf.build_thresholds(amt)
    assert got["pct"] == want["pct"]
    assert got["overall"] == want["overall"]
    assert got["by_board"] == want["by_board"]
    assert got["date"] == "2026-10-30"


def test_monthly_panel_honours_custom_pct(monkeypatch):
    """`pct` 必须真的改分位，而不只是改文案。

    为什么这条重要：小资金实盘要测"下限设 10%/20%/30% 有什么差别"，
    如果参数被忽略（或只在 note 里变了），会得到"三档结果完全一样"这种
    看着正常、其实没生效的结论。
    """
    monkeypatch.delenv("BIGA_LIQ_FLOOR", raising=False)
    rows = [{"month_ts": pd.Timestamp("2025-01-31"), "code": f"60{i:04d}",
             "amount_yuan": 1e8 + i * 1e6} for i in range(200)]
    df = pd.DataFrame(rows)
    n = {}
    for pct in (0.10, 0.30):
        out, meta = lf.filter_monthly_panel(df, pct=pct)
        n[pct] = len(out)
        assert f"{int(pct * 100)}% 分位" in meta["note"]
    assert n[0.10] > n[0.30], "10% 分位应比 30% 分位留下更多行"
    assert n[0.30] == pytest.approx(len(df) * 0.70, abs=3)
