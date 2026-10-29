"""外盘数据模块 `data/us_market.py`

为什么这些用例重要：

1. 外盘的 bar 与查询日**天然不同天**（周一早上取到的是上周五）。如果不显式返回
   数据日期，下游会把 3 天前的行情当成"昨夜美股"来解读 —— 这种错误写进报告里
   读起来毫无破绽（同 `_score_volume_vs_ma20` 拿两天前的量配今天涨幅是同一类坑）。
2. 取不到时的失败形态必须是 `unavailable: True` + 原因，而**不能**是空 dict 或
   0.00% —— 0 会被下游当成"外盘平盘"，那是凭空造出来的事实。
"""

import pandas as pd
import pytest
from unittest.mock import patch

from data import us_market


@pytest.fixture(autouse=True)
def _clear_cache():
    """模块带 30 分钟缓存，测试之间必须互不影响。"""
    us_market.reset_cache()
    yield
    us_market.reset_cache()


def _df(closes, dates=None):
    dates = dates or [f"2026-09-{20 + i:02d}" for i in range(len(closes))]
    return pd.DataFrame({"date": dates, "close": closes})


# ===== 分级阈值 =====

def test_classify_thresholds():
    """分级边界：±2% 以内对 A 股次日无决定力（10 年统计），故 2.0 才算 notable。"""
    assert us_market._classify(0.0)[0] == "neutral"
    assert us_market._classify(0.99)[0] == "neutral"
    assert us_market._classify(1.0)[0] == "mild"
    assert us_market._classify(1.99)[0] == "mild"
    assert us_market._classify(2.0)[0] == "notable"
    assert us_market._classify(3.99)[0] == "notable"
    assert us_market._classify(4.0)[0] == "extreme"


# ===== 正常取数 =====

def test_change_pct_computed_from_last_two_bars():
    with patch("akshare.index_us_stock_sina", return_value=_df([100.0, 105.0])):
        d = us_market.get_overnight_us(force=True)
    assert not d.get("unavailable"), d
    row = d["indexes"][0]
    assert row["change_pct"] == 5.0
    assert row["close"] == 105.0


def test_data_date_and_query_date_are_both_present():
    """数据日期与查询日期正交，两个都要给（周一取到上周五是正常的，但必须能看出来）。"""
    with patch("akshare.index_us_stock_sina",
               return_value=_df([100.0, 101.0], dates=["2026-09-24", "2026-09-25"])):
        d = us_market.get_overnight_us(force=True)
    assert d["data_date"] == "2026-09-25"
    assert d["query_date"], "查询日期不能为空——否则无法判断数据有多旧"
    assert isinstance(d["days_ago"], int) and d["days_ago"] >= 0


def test_driver_is_largest_absolute_move():
    """整体等级由**波动最大**的指数决定，不能被"道指平稳"平均掉。"""
    seq = {".IXIC": [100.0, 95.0], ".INX": [100.0, 100.5], ".DJI": [100.0, 100.2]}
    with patch("akshare.index_us_stock_sina", side_effect=lambda symbol: _df(seq[symbol])):
        d = us_market.get_overnight_us(force=True)
    assert d["driver"] == "纳斯达克"
    assert d["level"] == "extreme"
    assert d["max_abs_change_pct"] == 5.0


def test_flat_market_is_neutral_and_note_says_no_signal():
    """三大指数都在 1% 以内时，结论要明确说"不必据此调仓"，而不是含糊其辞。"""
    with patch("akshare.index_us_stock_sina", return_value=_df([100.0, 100.3])):
        d = us_market.get_overnight_us(force=True)
    assert d["level"] == "neutral"
    assert "无决定力" in d["note"] or "不必据此" in d["note"]


def test_extreme_move_note_warns_low_open_but_not_all_day_down():
    """暴跌时既要提示低开，也要给出"低开≠全天跌"的统计事实，避免制造恐慌。"""
    with patch("akshare.index_us_stock_sina", return_value=_df([100.0, 95.0])):
        d = us_market.get_overnight_us(force=True)
    assert d["level"] == "extreme"
    assert "低开" in d["note"]


# ===== 失败形态 =====

def test_all_sources_failing_marks_unavailable_not_zero():
    """取不到必须标 unavailable + 带原因，绝不能返回 0.00% 假装"平盘"。"""
    with patch("akshare.index_us_stock_sina", side_effect=Exception("network down")):
        d = us_market.get_overnight_us(force=True)
    assert d.get("unavailable") is True
    assert "失败" in d["note"]
    assert "network down" in d["note"], "失败原因要能看出是哪一层出的问题"
    assert "indexes" not in d or not d["indexes"]


def test_partial_failure_still_returns_available_ones():
    def fake(symbol):
        if symbol == ".IXIC":
            raise Exception("boom")
        return _df([100.0, 102.0])

    with patch("akshare.index_us_stock_sina", side_effect=fake):
        d = us_market.get_overnight_us(force=True)
    assert not d.get("unavailable")
    names = [r["name"] for r in d["indexes"]]
    assert "纳斯达克" not in names
    assert "标普500" in names and "道琼斯" in names


def test_single_bar_rejected():
    """只有 1 根 bar 算不出涨跌幅 —— 拿它跟自己比会得出 0%，那是假的。"""
    with patch("akshare.index_us_stock_sina", return_value=_df([100.0])):
        d = us_market.get_overnight_us(force=True)
    assert d.get("unavailable") is True


def test_none_dataframe_rejected():
    with patch("akshare.index_us_stock_sina", return_value=None):
        d = us_market.get_overnight_us(force=True)
    assert d.get("unavailable") is True


# ===== 缓存 =====

def test_cache_avoids_repeated_source_calls():
    with patch("akshare.index_us_stock_sina", return_value=_df([100.0, 101.0])) as m:
        us_market.get_overnight_us(force=True)
        first = m.call_count
        us_market.get_overnight_us()          # 应命中缓存
        assert m.call_count == first
    assert first == len(us_market.US_INDEXES)


def test_force_bypasses_cache():
    with patch("akshare.index_us_stock_sina", return_value=_df([100.0, 101.0])) as m:
        us_market.get_overnight_us(force=True)
        first = m.call_count
        us_market.get_overnight_us(force=True)
        assert m.call_count == first * 2
