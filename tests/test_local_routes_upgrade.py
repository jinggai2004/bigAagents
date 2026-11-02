# -*- coding: utf-8 -*-
"""五路初筛的补充（2026-11-01）：龙虎榜数据源 + 低估值成长的成长条件。

用户报的问题：**五路初筛实际只有三路能用**。核对结果比这更细：
- 4 路有数据，但 3 路是"降配版"（只剩一个条件）；
- 龙虎榜一路 `unavailable`（问财失效后没有数据源）。

本文件钉住的都是"回退会静默出错"的点：
1. **方向**：龙虎榜数据源只走缓存，且滞后超阈值必须判 unavailable（不拿旧数据充数）；
2. **单位**：`roe`/`sjltz` 是**小数**（0.025 = 2.5%），写成 10/20 会把全市场滤空；
3. **缺 ≠ 0**：无可见财报的票不满足成长条件，但不能被当成"增长 0%"计入统计；
4. **同源**：`latest_visible` 必须是 `latest_visible_multi` 的薄封装，不然 PIT 规则两处各写一遍。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import local_screen as L                   # noqa: E402
from data import billboard as B                      # noqa: E402


# ---------------------------------------------------------------- 日期工具
def test_weekdays_back_skips_weekend_and_descends():
    days = B._weekdays_back("2026-10-31", n_calendar=8)
    # 2026-10-31 是周六 ⇒ 最近一个工作日是 10-30
    assert days[0] == "2026-10-30"
    assert days == sorted(days, reverse=True)
    for d in days:
        assert pd.Timestamp(d).weekday() < 5, f"{d} 是周末"


def test_weekday_lag_counts_workdays():
    # 10-30(五) → 10-31(六) → 11-01(日) → 11-02(一)：1 个工作日
    assert L._weekday_lag("2026-10-30", "2026-11-02") == 1
    assert L._weekday_lag("2026-10-30", "2026-10-30") == 0
    assert L._weekday_lag(None, "2026-11-02") is None


# ---------------------------------------------------------------- 龙虎榜数据源
@pytest.fixture()
def bb_cache(tmp_path, monkeypatch):
    """临时龙虎榜缓存：3 个交易日、4 只票。"""
    monkeypatch.setattr(B, "CACHE_DIR", tmp_path / "billboard")
    B.CACHE_DIR.mkdir()
    # 必须挡掉联网补数：`ensure_days` 会把真实数据抓进临时目录，
    # 测试就会"通过"得莫名其妙（实测踩到：断言里出现了真实的 688836）。
    monkeypatch.setattr(B, "ensure_days", lambda days, force=False: {})
    rows = [
        ("000001", "平安银行", "2026-10-28", 3e7, 0.5),
        ("000001", "平安银行", "2026-10-29", 2e7, 0.2),
        ("600519", "贵州茅台", "2026-10-28", 5e6, -0.3),   # 不够 1000 万
        ("300750", "宁德时代", "2026-10-29", -2e7, 0.1),   # 净卖出
        ("900901", "云赛B股", "2026-10-29", 9e7, 0.0),     # B 股，不该进
    ]
    for code, name, d, net, _ in rows:
        f = B.CACHE_DIR / f"{d}.parquet"
        rec = pd.DataFrame([{"code": code, "SECURITY_CODE": code,
                             "SECURITY_NAME_ABBR": name, "trade_date": d,
                             "NET_BUY_AMT": net, "BUY_AMT": abs(net),
                             "SELL_AMT": 0.0, "BUY_COUNT": 2, "SELL_COUNT": 0,
                             "EXPLANATION": "x"}])
        if f.exists():
            rec = pd.concat([pd.read_parquet(f), rec], ignore_index=True)
        rec.to_parquet(f, index=False)
    return tmp_path / "billboard"


def test_org_net_buy_aggregates_and_filters(bb_cache):
    got = B.org_net_buy(asof="2026-10-30", days=5, min_net=1e7)
    # 000001 两天累计 5e7；300750 是净卖出；600519 不够门槛
    assert set(got) == {"000001"}
    assert got["000001"]["net"] == pytest.approx(5e7)
    assert got["000001"]["times"] == 2
    assert got["000001"]["name"] == "平安银行"


def test_org_net_buy_excludes_b_shares(bb_cache):
    """B 股（900xxx）必须被结构性剔除，不是靠名字。"""
    got = B.org_net_buy(asof="2026-10-30", days=5, min_net=1)
    assert "900901" not in got


def test_org_net_buy_empty_when_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "CACHE_DIR", tmp_path / "nope")
    assert B.org_net_buy(asof="2026-10-30", days=5) == {}


def test_route_billboard_ok(bb_cache):
    r = L._route_billboard(35, asof="2026-10-30")
    assert r.status == "ok"
    assert [x["股票代码"] for x in r.codes] == ["000001"]
    assert r.data_date == "2026-10-29"


def test_route_billboard_unavailable_when_stale(bb_cache):
    """缓存停在 10-29，跑到 11-10 ⇒ 滞后过大，必须判 unavailable（不拿旧数据充数）。"""
    r = L._route_billboard(35, asof="2026-11-10")
    assert r.status == "unavailable"
    assert "滞后" in r.note


def test_route_billboard_never_raises(monkeypatch):
    """数据源异常不许炸选股（与"预取异常不得阻断选股"同一条纪律）。"""
    def _boom(*a, **k):
        raise RuntimeError("接口挂了")
    monkeypatch.setattr(B, "ensure_days", _boom)
    r = L._route_billboard(35, asof="2026-10-30")
    assert r.status == "unavailable"
    assert "接口挂了" in r.note


# ---------------------------------------------------------------- 低估值成长
@pytest.fixture()
def snap_db(tmp_path, monkeypatch):
    """临时库：3 只 PE<30 的票 + 1 只 PE 超界。"""
    import db.models as M
    monkeypatch.setattr(M, "DB_PATH", str(tmp_path / "t.db"))
    from db import store as S
    S.init_store()
    conn = S.get_connection()
    try:
        conn.execute("""INSERT INTO market_snapshot
                        (snapshot_date, stock_code, stock_name, pe, market_cap,
                         amount, turnover, change_pct)
                        VALUES ('2026-10-30','000001','平安银行',6.0,2e11,1e9,1.0,0.5)""")
        conn.execute("""INSERT INTO market_snapshot
                        (snapshot_date, stock_code, stock_name, pe, market_cap,
                         amount, turnover, change_pct)
                        VALUES ('2026-10-30','600519','贵州茅台',20.0,2e12,1e9,1.0,0.5)""")
        conn.execute("""INSERT INTO market_snapshot
                        (snapshot_date, stock_code, stock_name, pe, market_cap,
                         amount, turnover, change_pct)
                        VALUES ('2026-10-30','300750','宁德时代',25.0,1e12,1e9,1.0,0.5)""")
        conn.execute("""INSERT INTO market_snapshot
                        (snapshot_date, stock_code, stock_name, pe, market_cap,
                         amount, turnover, change_pct)
                        VALUES ('2026-10-30','601398','工商银行',60.0,2e12,1e9,1.0,0.5)""")
        conn.commit()
    finally:
        conn.close()
    return S


def test_low_valuation_requires_growth_conditions(snap_db, monkeypatch):
    """三条件都要满足：PE<30 ✓ 但 ROE 或净利同比不够的必须被剔。"""
    import core.financial_pit as fp
    fake = {
        "000001": {"roe": 0.15, "sjltz": 0.30},    # 全过
        "600519": {"roe": 0.25, "sjltz": -0.01},   # 净利同比负 ⇒ 剔
        "300750": {"roe": 0.05, "sjltz": 0.50},    # ROE 不够 ⇒ 剔
    }
    monkeypatch.setattr(fp, "latest_visible_multi",
                        lambda codes, cols, asof=None: {k: v for k, v in fake.items()
                                                       if k in set(codes)})
    r = L._route_low_valuation(35, "2026-10-30")
    assert r.status == "ok"
    assert [x["股票代码"] for x in r.codes] == ["000001"]


def test_low_valuation_skips_missing_financials(snap_db, monkeypatch):
    """无可见财报 ⇒ 跳过并在 note 里报数量（缺 ≠ 0，不能当成"增长 0%"）。"""
    import core.financial_pit as fp
    monkeypatch.setattr(fp, "latest_visible_multi",
                        lambda codes, cols, asof=None: {"000001": {"roe": 0.15,
                                                                   "sjltz": 0.30}})
    r = L._route_low_valuation(35, "2026-10-30")
    assert [x["股票代码"] for x in r.codes] == ["000001"]
    assert "无可见财报" in r.note


def test_low_valuation_unavailable_without_snapshot(snap_db, monkeypatch):
    r = L._route_low_valuation(35, None)
    assert r.status == "unavailable"


def test_growth_thresholds_are_fractions():
    """阈值单位必须是小数——写成 10/20 会把全市场滤空，而且不报错。"""
    assert L.GROWTH_MIN_ROE == pytest.approx(0.10)
    assert L.GROWTH_MIN_SJLTZ == pytest.approx(0.20)
    assert L.PE_MAX_LOW == pytest.approx(30.0)


# ---------------------------------------------------------------- 同源
def test_latest_visible_is_thin_wrapper(tmp_path, monkeypatch):
    """`latest_visible` 必须与 `latest_visible_multi` 同源（PIT 规则只写一遍）。"""
    import core.financial_pit as fp
    calls = {"n": 0}

    def _fake(codes, cols, asof=None):
        calls["n"] += 1
        return {"000001": {"roe": 0.1}, "600519": {"roe": 0.2}}

    monkeypatch.setattr(fp, "latest_visible_multi", _fake)
    got = fp.latest_visible(["000001", "600519"], "roe", asof="2026-10-31")
    assert got == {"000001": 0.1, "600519": 0.2}
    assert calls["n"] == 1, "必须走 multi 版，不能自己再读一遍文件"
