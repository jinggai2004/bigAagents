# -*- coding: utf-8 -*-
"""在线 Brinson 归因装配层测试（`core/attribution_online.py`）。

为什么需要这一层测试
────────────────────
`core/attribution.py` 的三效应与 Carino 链接早已被 `test_attribution.py`
用合成数据手算锁死；**真正没被测过的是"装配"**——把 `screen_picks`（组合）
和面板日线（收益/基准）接成 `periods` 的那段代码。2026-10-30 之前这层根本
不存在（`brinson_attribution` 无任何调用方，是死代码），所以这里的用例锁的是：

1. 区间收益必须**逐日复利**（跨周末/长假不能少算成一天）；
2. 期内任一天停牌 ⇒ 该票**整体剔除**（不补 0 收益，否则基准被稀释）；
3. `daily_overlay` 必须**覆盖** `parts` 的同键记录（日更增量更新）；
4. 恒等式 Σ(配置+选股+交互) == 超额（`identity_gap == 0`）；
5. 组合规模跃变/小样本要出**警示**（不能把 1 只票的组合说成策略结论）；
6. 缺数据要报 `missing`/`empty`，不许返回空归因冒充"没超额"。

全部离线：临时 parquet 面板 + 临时 sqlite 库，不碰真实环境。
"""
from __future__ import annotations

import pandas as pd
import pytest

from core import attribution_online as ao


# ================================================================
# 夹具：临时面板 + 临时库
# ================================================================

_COLS = ["stock_code", "trade_date", "close", "preclose", "volume"]


@pytest.fixture
def temp_panel(tmp_path, monkeypatch):
    """临时面板目录（parts/ 与 daily_overlay/ 两个子目录）。"""
    base = tmp_path / "panel"
    (base / "parts").mkdir(parents=True)
    (base / "daily_overlay").mkdir()
    monkeypatch.setattr(ao, "_panel_dir", lambda: str(base))
    return base


def _write_parquet(path, rows: list[dict]) -> None:
    pd.DataFrame(rows, columns=_COLS).to_parquet(path, index=False)


def _bar(code: str, date: str, close: float, preclose: float, volume: float = 1e6) -> dict:
    return {"stock_code": code, "trade_date": date, "close": close,
            "preclose": preclose, "volume": volume}


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    return str(use_template_db(tmp_path, monkeypatch, "test_attr_online.db"))


def _insert_picks(db_path: str, rows: list[tuple]) -> None:
    """rows: [(pick_date, stock_code, source)]"""
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO screen_picks (pick_date, stock_code, source, score) "
            "VALUES (?, ?, ?, 80.0)", rows)
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def fixed_industry(monkeypatch):
    """固定行业映射（不读真实的申万 parquet，保证可复现）。"""
    mapping = {"000001": "甲", "000002": "甲", "000003": "乙",
               "000004": "乙", "000005": "丙"}
    monkeypatch.setattr(ao, "_load_industry",
                        lambda conn: (dict(mapping), "测试快照"))
    return mapping


# ================================================================
# 1. 区间收益：逐日复利 + 停牌剔除
# ================================================================

def _wide(prices: dict[str, list[float]], ok: dict[str, list[bool]]):
    """按 价格/可交易矩阵 造 wide_ret / wide_ok（两行两天）。"""
    dates = ["2026-01-05", "2026-01-06"]
    ret = {}
    for code, (p0, p1) in prices.items():
        ret[code] = [0.0, p1 / p0 - 1.0]        # 第一天无所谓（d0 不含）
    wide_ret = pd.DataFrame(ret, index=dates)
    wide_ok = pd.DataFrame(ok, index=dates)
    return wide_ret, wide_ok


def test_compound_return_multiplies_not_sums():
    """两天各 +10% 必须是 +21%，不是 +20%（复利 vs 相加）。"""
    wide_ret = pd.DataFrame({"A": [0.0, 0.10], "B": [0.0, 0.10]},
                            index=["2026-01-05", "2026-01-06"])
    wide_ok = pd.DataFrame({"A": [True, True], "B": [True, True]},
                           index=["2026-01-05", "2026-01-06"])
    grown, n = ao._compound_return(wide_ret, wide_ok, "2026-01-05", "2026-01-06")
    assert n == 1
    assert grown["A"] == pytest.approx(0.10, abs=1e-12)
    # 3 天时才是复利：0.10 → 0.21
    wide3 = pd.DataFrame({"A": [0.0, 0.10, 0.10]},
                         index=["2026-01-05", "2026-01-06", "2026-01-07"])
    ok3 = pd.DataFrame({"A": [True, True, True]},
                       index=["2026-01-05", "2026-01-06", "2026-01-07"])
    grown3, n3 = ao._compound_return(wide3, ok3, "2026-01-05", "2026-01-07")
    assert n3 == 2
    assert grown3["A"] == pytest.approx(0.21, abs=1e-12)


def test_compound_return_excludes_suspended_code_not_zero_filled():
    """期内有一天停牌 ⇒ 该票**整体剔除**；绝不能补 0 收益留在基准里。"""
    wide_ret = pd.DataFrame({"A": [0.0, 0.10], "B": [0.0, 0.05]},
                            index=["2026-01-05", "2026-01-06"])
    wide_ok = pd.DataFrame({"A": [True, True], "B": [True, False]},
                           index=["2026-01-05", "2026-01-06"])
    grown, _ = ao._compound_return(wide_ret, wide_ok, "2026-01-05", "2026-01-06")
    assert "A" in grown.index
    assert "B" not in grown.index, "停牌票必须被剔除，不能以 0 收益混进基准"


def test_compound_return_start_date_excluded():
    """d0 当天收益不算（仓位在 d0 收盘才建立），否则等于偷了当天的涨幅。"""
    wide_ret = pd.DataFrame({"A": [0.50, 0.10]}, index=["2026-01-05", "2026-01-06"])
    wide_ok = pd.DataFrame({"A": [True, True]}, index=["2026-01-05", "2026-01-06"])
    grown, _ = ao._compound_return(wide_ret, wide_ok, "2026-01-05", "2026-01-06")
    assert grown["A"] == pytest.approx(0.10, abs=1e-12)


# ================================================================
# 2. load_daily：overlay 覆盖 parts
# ================================================================

def test_overlay_overrides_parts_for_same_day(temp_panel):
    """同一天两处都有 ⇒ 用 `daily_overlay`（日更增量比 parts 新）。"""
    _write_parquet(temp_panel / "parts" / "daily_part_0000.parquet",
                   [_bar("000001", "2026-01-05", 10.0, 9.9)])
    _write_parquet(temp_panel / "daily_overlay" / "2026-01-05.parquet",
                   [_bar("000001", "2026-01-05", 12.0, 9.9)])
    df, bad = ao.load_daily("2026-01-05", "2026-01-05")
    assert bad == []
    got = df[df["stock_code"] == "000001"]["close"].tolist()
    assert got == [12.0], f"overlay 应覆盖 parts，实际 {got}"


def test_load_daily_filters_outside_window(temp_panel):
    """窗口外的行不该读进来（谓词下推要真的生效，否则会拖进全历史）。"""
    _write_parquet(temp_panel / "parts" / "daily_part_0000.parquet", [
        _bar("000001", "2025-12-01", 8.0, 7.9),
        _bar("000001", "2026-01-05", 10.0, 9.9),
        _bar("000001", "2026-03-01", 11.0, 10.9),
    ])
    df, _ = ao.load_daily("2026-01-01", "2026-01-31")
    assert df["trade_date"].tolist() == ["2026-01-05"]


def test_load_daily_broken_part_does_not_kill_everything(temp_panel):
    """单个分片损坏要跳过并留痕，不能让整个归因挂掉。"""
    (temp_panel / "parts" / "daily_part_bad.parquet").write_bytes(b"not a parquet")
    _write_parquet(temp_panel / "parts" / "daily_part_0000.parquet",
                   [_bar("000001", "2026-01-05", 10.0, 9.9)])
    df, bad = ao.load_daily("2026-01-01", "2026-01-31")
    assert len(df) == 1
    assert bad == ["daily_part_bad.parquet"], "坏分片必须出现在 bad_parts 里（不静默）"


def test_load_daily_empty_panel_returns_empty(temp_panel):
    df, bad = ao.load_daily("2026-01-01", "2026-01-31")
    assert df.empty and bad == []


# ================================================================
# 3. 端到端（临时库 + 临时面板）
# ================================================================

def _setup_e2e(temp_panel, temp_db: str, picks: list[tuple]) -> None:
    """5 只票 × 3 个交易日，价格固定好手算。

    000001: 10 → 11.0 (+10%) → 12.1 (+10%)
    000002: 10 →  9.0 (-10%) →  9.9 (+10%)
    000003: 20 → 20.0 ( 0%)  → 20.0 ( 0%)
    000004: 10 → 10.0 ( 0%)  → 10.0 ( 0%)
    000005: 10 → 10.0 ( 0%)  → 10.0 ( 0%)
    """
    rows = [
        _bar("000001", "2026-01-05", 10.0, 9.9), _bar("000001", "2026-01-06", 11.0, 10.0),
        _bar("000001", "2026-01-07", 12.1, 11.0),
        _bar("000002", "2026-01-05", 10.0, 9.9), _bar("000002", "2026-01-06", 9.0, 10.0),
        _bar("000002", "2026-01-07", 9.9, 9.0),
        _bar("000003", "2026-01-05", 20.0, 20.0), _bar("000003", "2026-01-06", 20.0, 20.0),
        _bar("000003", "2026-01-07", 20.0, 20.0),
        _bar("000004", "2026-01-05", 10.0, 10.0), _bar("000004", "2026-01-06", 10.0, 10.0),
        _bar("000004", "2026-01-07", 10.0, 10.0),
        _bar("000005", "2026-01-05", 10.0, 10.0), _bar("000005", "2026-01-06", 10.0, 10.0),
        _bar("000005", "2026-01-07", 10.0, 10.0),
    ]
    _write_parquet(temp_panel / "parts" / "daily_part_0000.parquet", rows)
    _insert_picks(temp_db, picks)


def test_run_identity_closes_and_numbers_match(temp_panel, temp_db, fixed_industry):
    """1 只票的组合 vs 5 只票等权基准：超额可手算，恒等式必须闭合。

    ⚠️ 期数 = 选股日数 − 1（期边界是"两个相邻的选股日之间"）⇒ 要 2 期就必须
    有 **3 个** 选股日。第一版只放了 2 个选股日却断言 2 期，是测试写错了。
    """
    _setup_e2e(temp_panel, temp_db, [("2026-01-05", "000001", "pipeline"),
                                     ("2026-01-06", "000001", "pipeline"),
                                     ("2026-01-07", "000001", "pipeline")])
    r = ao.run(min_picks=1)
    assert r["ok"], r
    s = r["summary"]
    assert r["periods_used"] == 2
    # 组合：两期都只持有 000001 ⇒ 每期 +10% ⇒ 累计 1.1×1.1−1 = 21%
    assert s["portfolio_return"] == pytest.approx(0.21, abs=1e-9)
    # 基准：每期 5 只等权。
    #   期1 = (0.10 − 0.10 + 0 + 0 + 0)/5 = 0
    #   期2 = (0.10 + 0.10 + 0 + 0 + 0)/5 = 0.04
    assert s["benchmark_return"] == pytest.approx(1.0 * 1.04 - 1.0, abs=1e-9)
    assert s["excess"] == pytest.approx(0.21 - 0.04, abs=1e-9)
    # 恒等式：Σ三效应 == 超额（归因算错的第一个信号）
    assert s["identity_gap"] == pytest.approx(0.0, abs=1e-8)
    assert s["linked"] is True
    assert {x["industry"] for x in r["top_industries"]} == {"甲", "乙", "丙"}


def test_run_warns_when_portfolio_size_jumps(temp_panel, temp_db, fixed_industry):
    """组合规模从 1 只跳到 4 只 ⇒ 必须出"口径变了"的警示（不能当同一策略）。"""
    _setup_e2e(temp_panel, temp_db, [
        ("2026-01-05", "000001", "pipeline"),
        *[("2026-01-06", c, "pipeline")
          for c in ("000001", "000002", "000003", "000004")],
        *[("2026-01-07", c, "pipeline")
          for c in ("000001", "000002", "000003", "000004")],
    ])
    r = ao.run(min_picks=1)
    assert r["ok"], r
    assert r["size_profile"]["min"] == 1 and r["size_profile"]["max"] == 4
    assert any("规模" in w for w in r["warnings"]), r["warnings"]


def test_run_warns_small_sample(temp_panel, temp_db, fixed_industry):
    """组合只有 1 只票 ⇒ 必须提示"诊断性归因，不是策略证据"。"""
    _setup_e2e(temp_panel, temp_db, [("2026-01-05", "000001", "pipeline"),
                                     ("2026-01-06", "000001", "pipeline"),
                                     ("2026-01-07", "000001", "pipeline")])
    r = ao.run(min_picks=1)
    assert any("中位数" in w or "诊断" in w for w in r["warnings"]), r["warnings"]


def test_run_min_names_filters_and_reports_skipped(temp_panel, temp_db, fixed_industry):
    """`min_names` 剔除的期必须进 `skipped_periods`（不许静默少算还不说）。"""
    _setup_e2e(temp_panel, temp_db, [
        ("2026-01-05", "000001", "pipeline"),
        *[("2026-01-06", c, "pipeline")
          for c in ("000001", "000002", "000003", "000004")],
        *[("2026-01-07", c, "pipeline")
          for c in ("000001", "000002", "000003", "000004")],
    ])
    r = ao.run(min_picks=1, min_names=3)
    assert r["ok"], r
    assert r["periods_used"] == 1
    assert any("min_names" in (s.get("reason") or "") for s in r["skipped_periods"])


def test_run_source_filter_separates_strategies(temp_panel, temp_db, fixed_industry):
    """`source` 必须真的过滤：screen 与 pipeline 是两个不同口径。

    断言的是**用的是哪一天的选股**（期首日），而不是"总数变了"——
    两个 source 各 2 个选股日时总数都是 1 期，只有期首日能区分。

    ⚠️ 两个 source 必须用**不同代码**：`screen_picks` 有 (pick_date, stock_code)
    唯一索引（2026-10-29 迁移），同日同票只能有一条记录 ⇒ 同代码无法同时属于
    两个 source。这条约束本身也值得记住。
    """
    _setup_e2e(temp_panel, temp_db, [
        ("2026-01-05", "000001", "screen"),
        ("2026-01-06", "000001", "screen"),
        ("2026-01-06", "000002", "pipeline"),
        ("2026-01-07", "000002", "pipeline"),
    ])
    scr = ao.run(min_picks=1, source="screen")
    pip = ao.run(min_picks=1, source="pipeline")
    assert scr["ok"] and str(scr["periods"][0]["date"]) == "2026-01-05"
    assert pip["ok"] and str(pip["periods"][0]["date"]) == "2026-01-06"


def test_run_missing_when_no_picks(temp_panel, temp_db, fixed_industry):
    """库里没有选股记录 ⇒ `missing`（"还没跑"），不许返回 0 超额。"""
    _setup_e2e(temp_panel, temp_db, [])
    r = ao.run()
    assert r["ok"] is False and r["code"] == "missing"


def test_run_empty_when_no_panel(temp_panel, temp_db, fixed_industry):
    """有选股但面板没数据 ⇒ `missing` 并说明先更新面板。"""
    _insert_picks(temp_db, [("2026-01-05", "000001", "pipeline"),
                            ("2026-01-06", "000001", "pipeline")])
    r = ao.run()
    assert r["ok"] is False and r["code"] == "missing"
    assert "面板" in r["reason"]


def test_run_empty_when_single_pick_date(temp_panel, temp_db, fixed_industry):
    """只有一个选股日 ⇒ 算不出区间收益，必须报 `empty` 并说清原因。"""
    _setup_e2e(temp_panel, temp_db, [("2026-01-05", "000001", "pipeline")])
    r = ao.run(min_picks=1)
    assert r["ok"] is False and r["code"] == "empty"
    assert "≥2" in r["reason"] or "2" in r["reason"]


def test_run_caliber_is_reported_not_implied(temp_panel, temp_db, fixed_industry):
    """口径必须写进返回值：基准不是沪深300、收益复利、行业是快照（前视偏差）。"""
    _setup_e2e(temp_panel, temp_db, [("2026-01-05", "000001", "pipeline"),
                                     ("2026-01-06", "000001", "pipeline")])
    c = ao.run(min_picks=1)["caliber"]
    assert "全市场" in c["benchmark"]
    assert "沪深300" in c["benchmark_note"]
    assert c["snapshot_lookahead"] is True
    assert "复利" in c["return"]
