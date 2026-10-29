"""交易日历测试（Phase 1）

覆盖：真实日历判定、区间枚举、上一/下一交易日、文件缓存、源不可用时的降级，
以及"落在日历覆盖范围之外"的处理。
"""

import json
import time
from unittest.mock import patch

import pytest

import data.trading_calendar as tc


# 2026-09-21(一) 22(二) 24(四) 25(五) 28(一)；故意跳过 23(三) 模拟节假日
FAKE_DATES = {"2026-09-21", "2026-09-22", "2026-09-24", "2026-09-25", "2026-09-28"}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "CACHE_PATH", str(tmp_path / "cal.json"))
    tc.clear_memory_cache()
    yield
    tc.clear_memory_cache()


def _fake_ak(dates=None):
    return patch.object(tc, "_fetch_from_akshare", return_value=set(dates or FAKE_DATES))


class TestRealCalendar:
    def test_is_trading_day_true(self):
        with _fake_ak():
            assert tc.is_trading_day("2026-09-22") is True

    def test_holiday_inside_range_is_false(self):
        """落在覆盖范围内的非交易日（节假日）必须判 False，不能被工作日近似放过"""
        with _fake_ak():
            assert tc.is_trading_day("2026-09-23") is False

    def test_trading_days_range(self):
        with _fake_ak():
            assert tc.trading_days("2026-09-21", "2026-09-25") == [
                "2026-09-21", "2026-09-22", "2026-09-24", "2026-09-25"]

    def test_last_and_next_trading_day(self):
        with _fake_ak():
            assert tc.last_trading_day("2026-09-23") == "2026-09-22"
            assert tc.last_trading_day("2026-09-24", include_today=False) == "2026-09-22"
            assert tc.next_trading_day("2026-09-22") == "2026-09-24"

    def test_out_of_range_falls_back_to_weekday(self):
        with _fake_ak():
            assert tc.is_trading_day("2027-01-04") is True    # 周一，超出覆盖范围
            assert tc.is_trading_day("2027-01-02") is False   # 周六

    def test_cache_status_reports_source(self):
        with _fake_ak():
            tc.is_trading_day("2026-09-22")
            st = tc.cache_status()
        assert st["source"] == "akshare"
        assert st["degraded"] is False
        assert st["count"] == len(FAKE_DATES)
        assert st["min"] == "2026-09-21" and st["max"] == "2026-09-28"


class TestFileCache:
    def test_writes_then_reads_without_refetch(self):
        with _fake_ak():
            tc.refresh()
        with open(tc.CACHE_PATH, encoding="utf-8") as f:
            assert set(json.load(f)["dates"]) == FAKE_DATES

        tc.clear_memory_cache()
        with patch.object(tc, "_fetch_from_akshare") as mock_fetch:
            assert tc.is_trading_day("2026-09-22") is True
        mock_fetch.assert_not_called()
        assert tc.cache_status()["source"] == "file_cache"

    def test_stale_cache_used_when_source_down(self):
        with _fake_ak():
            tc.refresh()
        with open(tc.CACHE_PATH, encoding="utf-8") as f:
            payload = json.load(f)
        payload["saved_at"] = time.time() - tc.CACHE_TTL_SECONDS - 10
        with open(tc.CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        tc.clear_memory_cache()
        with patch.object(tc, "_fetch_from_akshare", return_value=None):
            assert tc.is_trading_day("2026-09-22") is True
        assert tc.cache_status()["source"] == "file_cache(stale)"


class TestDegradedFallback:
    """源不可用且无缓存时降级为工作日近似，但必须显式标记，不假装精确"""

    def test_marks_degraded_with_weekday_approximation(self):
        with patch.object(tc, "_fetch_from_akshare", return_value=None):
            st = tc.refresh(force=True)
        assert st["degraded"] is True
        assert st["source"] == "weekday_fallback"

    def test_weekday_approximation_semantics(self):
        with patch.object(tc, "_fetch_from_akshare", return_value=None):
            assert tc.is_trading_day("2026-09-23") is True    # 周三
            tc.clear_memory_cache()
        with patch.object(tc, "_fetch_from_akshare", return_value=None):
            assert tc.is_trading_day("2026-09-26") is False   # 周六
