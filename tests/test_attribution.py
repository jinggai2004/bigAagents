# -*- coding: utf-8 -*-
"""Brinson 归因测试（attribution）：合成数据，数字全部手算锁死。

覆盖：
1. 单期三效应恒等式 Σ(alloc+select+inter) = rp − rb（行业与全组合两层）；
2. Carino 链接的定义性恒等式 Σ 链接后效应 = R − B（多期）；
3. Carino 退化边界（r_t = b_t 的期；r ≤ −1 的爆仓期标 degraded）；
4. 未分类桶：缺行业映射的票进 `__未分类__` 参与分解，不静默丢；
5. 数据层：行业映射 loader 缺文件 ⇒ None；回填（注入 fetch）→ 落盘 → 读回。
"""
import pytest

from core import attribution as att
from data import industry_swa as isw


# ================================================================
# 1. 单期 Brinson
# ================================================================

def _two_industry_period():
    """手工可算的基准情形：两行业 × 每行业两票（select/inter 才非平凡）。

    收益（两套权重用**同一份**逐票收益——Brinson 的语义）：
      甲 +10%、乙 +2%、丙 −4%、丁 +6%
    组合：甲 0.3 + 乙 0.1（X=0.4）+ 丙 0.3 + 丁 0.3（Y=0.6）⇒ rp = 0.038
    基准：甲 0.1 + 乙 0.15（X=0.25）+ 丙 0.45 + 丁 0.30（Y=0.75）⇒ rb = 0.013
    行业收益：rp_X = 0.032/0.4 = 0.08；rb_X = 0.013/0.25 = 0.052
              rp_Y = 0.006/0.6 = 0.01；rb_Y = 0/0.75 = 0
    X：alloc = 0.15·0.052 = +0.0078；select = 0.25·0.028 = +0.007；
       inter = 0.15·0.028 = +0.0042 ⇒ X = +0.019
    Y：alloc = (−0.15)·0 = 0；select = 0.75·0.01 = +0.0075；
       inter = (−0.15)·0.01 = −0.0015 ⇒ Y = +0.006
    总超额 = 0.038 − 0.013 = +0.025 = 0.019 + 0.006 ✓
    """
    return {"portfolio": {"600001": 0.3, "600002": 0.1, "000001": 0.3, "000002": 0.3},
            "benchmark": {"600001": 0.1, "600002": 0.15, "000001": 0.45, "000002": 0.30},
            "industry": {"600001": "X", "600002": "X", "000001": "Y", "000002": "Y"},
            "returns": {"600001": 0.10, "600002": 0.02, "000001": -0.04, "000002": 0.06}}


def test_brinson_single_period_identity_and_values():
    p = _two_industry_period()
    eff = att.brinson_single_period(p["portfolio"], p["benchmark"],
                                    p["industry"], p["returns"])
    assert eff["portfolio_return"] == pytest.approx(0.038, abs=1e-9)
    assert eff["benchmark_return"] == pytest.approx(0.013, abs=1e-9)
    assert eff["excess"] == pytest.approx(0.025, abs=1e-9)
    x = eff["industries"]["X"]
    assert x["alloc"] == pytest.approx(0.0078, abs=1e-9)
    assert x["select"] == pytest.approx(0.007, abs=1e-9)
    assert x["inter"] == pytest.approx(0.0042, abs=1e-9)
    assert x["total"] == pytest.approx(0.019, abs=1e-9)
    y = eff["industries"]["Y"]
    assert y["alloc"] == pytest.approx(0.0, abs=1e-9)
    assert y["select"] == pytest.approx(0.0075, abs=1e-9)
    assert y["inter"] == pytest.approx(-0.0015, abs=1e-9)
    # 恒等式：Σ 行业 total = 组合超额
    assert sum(i["total"] for i in eff["industries"].values()) == \
        pytest.approx(eff["excess"], abs=1e-9)


def test_brinson_unclassified_bucket_not_dropped():
    """缺行业映射的票进 `__未分类__` 参与分解，不静默丢；恒等式仍闭合"""
    p = _two_industry_period()
    eff = att.brinson_single_period(p["portfolio"], p["benchmark"],
                                    {"600001": "X", "600002": "X"},   # Y 行业两票无映射
                                    p["returns"])
    assert eff["industries"]["__未分类__"]["wp"] == pytest.approx(0.6, abs=1e-9)
    assert set(eff["unclassified_codes"]) == {"000001", "000002"}
    assert sum(i["total"] for i in eff["industries"].values()) == \
        pytest.approx(eff["excess"], abs=1e-9)


def test_brinson_missing_return_tracked_not_silent():
    """缺收益的票按 0 收益参与但**留痕**（codes_without_return）——比静默剔除诚实"""
    p = _two_industry_period()
    eff = att.brinson_single_period(p["portfolio"], p["benchmark"],
                                    p["industry"], {"600001": 0.10})
    assert "000002" in eff["codes_without_return"]


# ================================================================
# 2. Carino 链接
# ================================================================

def test_carino_identity_two_periods():
    """定义性恒等式：Σ 链接后行业效应 = R − B（算术总超额）。
    （2026-10-29 实测：k_t 写成倒数时这里算出 0.0432 ≠ 0.0448——第一个红的就是它）"""
    p1 = _two_industry_period()
    e1 = att.brinson_single_period(p1["portfolio"], p1["benchmark"],
                                   p1["industry"], p1["returns"])
    p2 = {"portfolio": {"600001": 0.4, "600002": 0.2, "000001": 0.2, "000002": 0.2},
          "benchmark": {"600001": 0.25, "600002": 0.25, "000001": 0.25, "000002": 0.25},
          "industry": p1["industry"],
          "returns": {"600001": -0.03, "600002": 0.01, "000001": 0.05, "000002": 0.04}}
    e2 = att.brinson_single_period(p2["portfolio"], p2["benchmark"],
                                   p2["industry"], p2["returns"])
    rp1, rp2 = e1["portfolio_return"], e2["portfolio_return"]
    rb1, rb2 = e1["benchmark_return"], e2["benchmark_return"]
    linked = att.carino_link([e1, e2], [rp1, rp2], [rb1, rb2])
    R = (1 + rp1) * (1 + rp2) - 1
    B = (1 + rb1) * (1 + rb2) - 1
    assert linked["excess"] == pytest.approx(R - B, abs=1e-9)
    total_effect = sum(i["total"] for i in linked["total"].values())
    assert total_effect == pytest.approx(linked["excess"], abs=1e-8)


def test_carino_equal_returns_period_edge():
    """r_t = b_t 的期：k_t 取极限 1/(1+r)，不得除零"""
    k = att._carino_k(0.02, 0.02)
    assert k == pytest.approx(1.0 / 1.02, abs=1e-12)


def test_carino_blowup_period_flagged_degraded():
    """r ≤ −1 的爆仓期：对数无定义 ⇒ 退化系数 + degraded 留痕（不抛异常）"""
    linked = att.carino_link(
        [{"industries": {"X": {"alloc": 0.01, "select": 0.0, "inter": 0.0,
                               "total": 0.01}}}],
        [-1.2], [0.01])
    assert linked["degraded_periods"]
    assert linked["total"]["X"]["total"] != 0.0


# ================================================================
# 3. 入口与 degraded 语义
# ================================================================

def test_brinson_attribution_requires_returns():
    """缺逐票收益的期被剔除并留痕，且**不给**链接合计（恒等式要求全期参与）"""
    p = _two_industry_period()
    res = att.brinson_attribution(
        [{"date": "2026-01-05", **p},
         {"date": "2026-01-12", "portfolio": p["portfolio"],
          "benchmark": p["benchmark"], "returns": None}],
        industry=p["industry"], snapshot_date="2026-10-29")
    assert len(res["periods"]) == 1
    assert res["linked"] is None
    assert res["degraded"]["periods_without_returns"] == ["2026-01-12"]
    assert res["industry_snapshot"] == "2026-10-29"      # 快照标记必须透传


def test_brinson_attribution_full_flow_snapshot_marked():
    p = _two_industry_period()
    res = att.brinson_attribution(
        [{"date": "2026-01-05", **p},
         {"date": "2026-01-12",
          "portfolio": {"600001": 0.4, "600002": 0.2, "000001": 0.2, "000002": 0.2},
          "benchmark": {"600001": 0.25, "600002": 0.25, "000001": 0.25, "000002": 0.25},
          "industry": p["industry"],
          "returns": {"600001": -0.03, "600002": 0.01, "000001": 0.05, "000002": 0.04}}],
        industry=p["industry"], snapshot_date="2026-10-29")
    assert res["linked"] is not None
    total_effect = sum(i["total"] for i in res["linked"]["total"].values())
    assert total_effect == pytest.approx(res["linked"]["excess"], abs=1e-8)
    assert res["degraded"]["snapshot_lookahead"] is True   # 前视标记：快照归因


# ================================================================
# 4. 行业映射数据层
# ================================================================

def test_industry_loader_missing_file_returns_none(tmp_path, monkeypatch):
    """本地没有行业文件 ⇒ None（缺就是缺），不抛异常"""
    monkeypatch.setattr(isw, "INDUSTRY_PATH", str(tmp_path / "nope.parquet"))
    assert isw.load_industry_swa() is None


def test_industry_backfill_and_loader_roundtrip(tmp_path, monkeypatch):
    """回填（fetch 注入罐头数据）→ 落盘 → 读回；snapshot_date 透传"""
    import pandas as pd
    monkeypatch.setattr(isw, "INDUSTRY_PATH", str(tmp_path / "industry_swa_l1.parquet"))

    def fake_fetch(symbol):
        # 每个行业返回**自己的**票（801010 → 600010），避免跨行业同名覆盖
        return pd.DataFrame({"证券代码": [symbol.replace("801", "600")],
                             "证券名称": ["测试票"],
                             "计入日期": ["2021-12-13"]})

    monkeypatch.setattr(isw, "_fetch_one", fake_fetch)
    out = isw.backfill_industry_swa(sleep_secs=0.0)
    assert out["industries_ok"] == len(isw.SW_L1)
    assert out["stocks"] == len(isw.SW_L1) and out["industries_failed"] == 0
    loaded = isw.load_industry_swa()
    assert loaded is not None
    assert loaded["industry"]["600010"] == "农林牧渔"      # 801010 → 600010
    assert loaded["industry"]["600980"] == "美容护理"      # 801980 → 600980
    assert loaded["snapshot_date"]


def test_industry_backfill_partial_failure_keeps_rest(tmp_path, monkeypatch):
    """部分行业拉取失败 ⇒ 跳过并留痕，成功的照常落盘（部分成功比零强）"""
    import pandas as pd
    monkeypatch.setattr(isw, "INDUSTRY_PATH", str(tmp_path / "industry_swa_l1.parquet"))
    calls = {"n": 0}

    def flaky_fetch(symbol):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("限频")
        return pd.DataFrame({"证券代码": ["600001"], "计入日期": ["2021-12-13"]})

    monkeypatch.setattr(isw, "_fetch_one", flaky_fetch)
    out = isw.backfill_industry_swa(retry=0, sleep_secs=0.0)
    assert out["industries_failed"] == 1 and out["industries_ok"] == len(isw.SW_L1) - 1
    assert out["failed"] and "801010" in out["failed"][0]
    assert isw.load_industry_swa() is not None
