# -*- coding: utf-8 -*-
"""宇宙守卫（`core/universe_guard.py`）的铺设测试。

2026-10-31 的背景：北交所伪数据同时污染了**因子 IC、组合 A/B、执行折扣**三层，
而当时只有两个脚本接了这个守卫。本文件守两件事：

1. **原语本身**：`comparable_since` / `comparable_mask` / `trim` / `trim_matrix`
   判定一致，且"未知票"永远不放行（宁可少算，不拿来源不明的历史去排名）；
2. **接线**：三个真正跑回测的入口（条件回测 `preload`、因子分析 `load_bars` /
   `run` 注入路径、规则回测 `preload`）**确实调用了**守卫——写了函数没接线
   等于零覆盖（本项目已经踩过"函数写好≠页面看得见"）。
"""
import pandas as pd
import pytest

from core import universe_guard as ug


# ── 夹具：假清单（绝不允许测试去读真实 temp/listing_table.parquet）──

@pytest.fixture
def listing(monkeypatch):
    """一只老票 / 一只 2022-06 上市的新票 / 一只 2016 就有数据的北交所票。"""
    lst = pd.DataFrame([
        ["600000", pd.Timestamp("2010-01-04"), pd.Timestamp("2026-10-16"), 4000],
        ["301999", pd.Timestamp("2022-06-01"), pd.Timestamp("2026-10-16"), 1000],
        ["920001", pd.Timestamp("2016-01-04"), pd.Timestamp("2026-10-16"), 2400],
    ], columns=["code", "first_date", "last_date", "rows"])
    # 关键：让**所有**入口内部的无参调用都拿到这份假清单
    monkeypatch.setattr(ug, "build_listing_table", lambda *a, **k: lst)
    return lst


def _daily(start: str, end: str, code: str = "600000") -> pd.DataFrame:
    """最小但**结构完整**的日线：`_enrich` 会读 high/low/open/amount/turnover_pct，
    少了任何一列都会以 KeyError 的形式失败（第一版就只给了 close）。"""
    n = len(pd.date_range(start, end, freq="D"))
    return pd.DataFrame({
        "date": pd.date_range(start, end, freq="D"),
        "code": code,
        "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
        "volume": 1_000_000.0, "amount": 1e8,
        "turnover_pct": [1.0 + 0.01 * i for i in range(n)],
    })


# ================================================================
# 1. 可比起点
# ================================================================

def test_comparable_since_adds_one_year(listing):
    assert ug.comparable_since("600000", listing=listing) == pd.Timestamp("2011-01-04")
    assert ug.comparable_since("301999", listing=listing) == pd.Timestamp("2023-06-01")


def test_comparable_since_lifts_bse_to_open_date(listing):
    """北交所票：首日 + 365 天仍在开市前 ⇒ 抬到 2021-11-15。"""
    assert ug.comparable_since("920001", listing=listing) == ug.BSE_START


def test_comparable_since_unknown_is_none(listing):
    assert ug.comparable_since("000404", listing=listing) is None


# ================================================================
# 2. 逐行掩码 / 裁剪
# ================================================================

def test_trim_keeps_old_stock_untouched(listing):
    df = _daily("2019-01-01", "2019-03-01")
    out, cut = ug.trim_with_stats(df, "600000")
    assert cut == 0 and len(out) == len(df)


def test_trim_cuts_first_year_of_new_listing(listing):
    df = _daily("2022-06-01", "2023-12-31", code="301999")
    out, cut = ug.trim_with_stats(df, "301999")
    assert cut > 0
    assert out["date"].min() == pd.Timestamp("2023-06-01")


def test_trim_bse_before_open_returns_empty_but_keeps_columns(listing):
    df = _daily("2019-01-01", "2021-06-30", code="920001")
    out, cut = ug.trim_with_stats(df, "920001")
    assert len(out) == 0
    assert list(out.columns) == list(df.columns), "空表必须保留列结构"
    assert cut == len(df)


def test_trim_unknown_code_drops_everything(listing):
    """清单里没有的票一律不可比 —— 宁可少算，不拿来源不明的历史排名。"""
    df = _daily("2020-01-01", "2020-02-01", code="000404")
    out, _ = ug.trim_with_stats(df, "000404")
    assert len(out) == 0


def test_trim_does_not_mutate_input(listing):
    df = _daily("2022-06-01", "2023-12-31", code="301999")
    before = len(df)
    ug.trim_with_stats(df, "301999")
    assert len(df) == before, "守卫必须返回副本，不能就地改调用方的数据"


def test_trim_matrix_masks_with_nan_not_zero(listing):
    """宽表版：不可比格子必须 NaN。置 0 会被当成'当天涨跌 0%'混进均值。"""
    m = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        index=pd.to_datetime(["2023-05-01", "2023-07-01"]),
        columns=["600000", "301999", "000404"])
    out = ug.trim_matrix(m, listing=listing)
    # 2023-05-01：新票（since 2023-06-01）与未知票都不可比
    assert pd.isna(out.loc["2023-05-01", "301999"])
    assert pd.isna(out.loc["2023-05-01", "000404"])
    assert out.loc["2023-05-01", "600000"] == 1.0
    # 2023-07-01：新票已满一年，保留；未知票永远 NaN
    assert out.loc["2023-07-01", "301999"] == 5.0
    assert pd.isna(out.loc["2023-07-01", "000404"])


# ================================================================
# 3. 两种用法必须是**同一判据**（防止月度/日频口径分叉）
# ================================================================

def test_monthly_eligible_agrees_with_daily_trim(listing):
    """月度 `eligible` 判"月末那天"，日频 `trim` 判"逐行" —— 结论必须一致。

    这条哨兵防的是：某天有人把日频规则的 365 自然日改成"250 个交易日"，
    月度侧却没改，于是同一份数据在两处得到不同的样本集，而**不会报错**。
    """
    months = pd.date_range("2022-01-31", "2024-06-30", freq="ME")
    monthly = pd.DataFrame({"month_ts": months, "code": "301999"})
    ok = ug.eligible(monthly, listing=listing)

    daily = _daily("2022-01-01", "2024-06-30", code="301999")
    kept = set(ug.trim(daily, code="301999", listing=listing)["date"])
    for ts, flag in zip(months, ok):
        assert bool(flag) == (ts in kept), f"{ts.date()} 两处判定不一致"


# ================================================================
# 4. 开关
# ================================================================

def test_guard_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("BIGA_NO_UNIVERSE_GUARD", raising=False)
    assert ug.guard_enabled() is True
    for v in ("1", "true", "YES"):
        monkeypatch.setenv("BIGA_NO_UNIVERSE_GUARD", v)
        assert ug.guard_enabled() is False, v
    monkeypatch.setenv("BIGA_NO_UNIVERSE_GUARD", "0")
    assert ug.guard_enabled() is True


# ================================================================
# 5. 接线：三个回测入口真的调了守卫
# ================================================================

def test_condition_backtest_preload_applies_guard(listing, monkeypatch):
    """条件回测 `preload` 必须剪掉不可比区间（它是页面「一键回测」的入口）。"""
    from core import condition_backtest as cb
    from data import bars as _bars

    def _fake_get_bars(code, start=None, end=None, **kw):
        # 故意返回整段（含上市首年），模拟"取数层不知道研究语义"
        return _daily(start or "2019-01-01", end or "2023-12-31", code=code)

    monkeypatch.setattr(_bars, "get_bars", _fake_get_bars)
    monkeypatch.setattr(_bars, "offline_mode", lambda: True)

    gen = cb.ConditionSignalGenerator.__new__(cb.ConditionSignalGenerator)
    gen.codes, gen._data, gen._no_data, gen._loaded_range = ["301999"], {}, set(), None
    gen.preload("2022-01-01", "2023-12-31")

    assert "301999" in gen._data
    assert gen._data["301999"]["date"].min() >= pd.Timestamp("2023-06-01"), \
        "守卫没生效：上市首年的行还在"


def test_condition_backtest_guard_off_keeps_raw(listing, monkeypatch):
    """对照：关掉守卫后必须回到原样（证明上面那条不是恒真）。"""
    from core import condition_backtest as cb
    from data import bars as _bars

    monkeypatch.setenv("BIGA_NO_UNIVERSE_GUARD", "1")
    monkeypatch.setattr(_bars, "get_bars",
                        lambda code, start=None, end=None, **kw:
                        _daily(start, end, code=code))
    monkeypatch.setattr(_bars, "offline_mode", lambda: True)

    gen = cb.ConditionSignalGenerator.__new__(cb.ConditionSignalGenerator)
    gen.codes, gen._data, gen._no_data, gen._loaded_range = ["301999"], {}, set(), None
    gen.preload("2022-01-01", "2023-12-31")
    assert gen._data["301999"]["date"].min() == pd.Timestamp("2022-01-01")


def test_factor_load_bars_applies_guard(listing, monkeypatch):
    """`load_bars` 直接调用方（attention_ic 等）也必须吃到守卫。"""
    import core.factor_analysis as FA
    from data import bars as _bars

    monkeypatch.setattr(_bars, "get_bars",
                        lambda code, start=None, end=None, **kw:
                        _daily(start, end, code=code))
    monkeypatch.setattr(_bars, "offline_mode", lambda: True)
    monkeypatch.setattr(_bars, "_clean", lambda c: str(c).zfill(6))

    out = FA.load_bars(["301999"], "2022-01-01", "2023-12-31")
    assert "301999" in out
    assert out["301999"]["date"].min() >= pd.Timestamp("2023-06-01")


def test_factor_run_guards_injected_bars_map(listing):
    """注入 `bars_map` 的路径（研究面板复用缓存）不能被漏掉。"""
    import core.factor_analysis as FA

    raw = {}
    for code in ("600000", "301999"):
        df = _daily("2022-01-01", "2024-06-30", code=code)
        raw[code] = FA._enrich(df)
    guarded = FA._guard_bars_map(raw)
    assert guarded["301999"]["date"].min() >= pd.Timestamp("2023-06-01")
    assert guarded["600000"]["date"].min() == pd.Timestamp("2022-01-01")


def test_rule_backtest_preload_applies_guard(listing, monkeypatch):
    """规则回测是 `pipeline.run_backtest`（线上盘后）的入口，也必须有守卫。"""
    from scripts.backtest_rule_based import RuleSignalGenerator

    gen = RuleSignalGenerator.__new__(RuleSignalGenerator)
    gen._data_cache = {}
    gen._score_cache = {}
    gen._atr_cache = {}
    gen._close_series_cache = {}
    gen._no_data_codes = set()
    gen._loaded_range = None

    class _DM:
        def get_stock_hist_data(self, code, start_date=None, end_date=None, adjust=None):
            return _daily("2022-01-01", "2023-12-31", code=code)

    gen.dm = _DM()
    monkeypatch.setattr("scripts.backtest_rule_based.WATCHLIST", ["301999"])

    gen.preload("2023-01-01", "2023-12-31")
    df = gen._data_cache.get("301999")
    assert df is not None, "该票在窗口内应可比"
    assert df["date"].min() >= pd.Timestamp("2023-06-01"), "守卫没生效"


# ================================================================
# 6. 判据缺失时 fail-open（而不是"全体不可比"）
# ================================================================

def test_empty_listing_fails_open(monkeypatch, caplog):
    """清单整体为空 ⇒ **放行 + 告警**，不能把全部历史判成不可比。

    为什么这条必须存在：exe 若被拷到没有祖先 `data/` 的目录，`by_code` 读不到、
    清单为空。若照常判"不可比"，研究面板 IC / 条件回测 / 盘后回测会**集体返回
    空**，而界面上看不出原因——"守卫把功能打死"比"没有守卫"更糟。
    注意这**不是**给"清单里缺某只票"开口子：那种情况仍然剔除（见上面两条）。
    """
    monkeypatch.setattr(ug, "build_listing_table",
                        lambda *a, **k: pd.DataFrame(columns=["code", "first_date"]))
    monkeypatch.setattr(ug, "_empty_warned", False, raising=False)
    df = _daily("2020-01-01", "2020-03-01")
    out, cut = ug.trim_with_stats(df, "600000")
    assert len(out) == len(df), "清单为空时必须整体放行"
    assert cut == 0


def test_empty_listing_masks_nothing_in_matrix(monkeypatch):
    monkeypatch.setattr(ug, "build_listing_table",
                        lambda *a, **k: pd.DataFrame(columns=["code", "first_date"]))
    monkeypatch.setattr(ug, "_empty_warned", False, raising=False)
    m = pd.DataFrame([[1.0, 2.0]], index=pd.to_datetime(["2020-01-01"]),
                     columns=["600000", "301999"])
    out = ug.trim_matrix(m)
    assert not out.isna().any().any(), "判据缺失时宽表不应被置 NaN"
