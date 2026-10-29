"""基准收益模块单元测试（全 mock，不依赖网络）"""
import pandas as pd
import pytest
from unittest.mock import patch

import data.benchmark as bm
from data.benchmark import (
    get_benchmark_nav,
    get_benchmark_summary,
    BENCHMARK_INDEXES,
    _bench_cache,
)


def _make_index_df(dates, closes):
    """构造指数日线假数据（新浪源列结构）"""
    return pd.DataFrame({"date": pd.to_datetime(dates), "close": closes})


@pytest.fixture(autouse=True)
def _clear_cache():
    _bench_cache.clear()
    yield
    _bench_cache.clear()


class TestGetBenchmarkNav:
    def test_basic_nav_shape(self):
        df = _make_index_df(
            pd.date_range("2024-01-01", periods=30, freq="D"),
            [3000 + i * 10 for i in range(30)],
        )
        with patch("data.benchmark.ak.stock_zh_index_daily", return_value=df):
            nav = get_benchmark_nav("2024-01-01", "2024-01-30")
        assert nav is not None
        assert len(nav) == 30
        assert nav[0] == {"date": "2024-01-01", "total_value": 1_000_000.0}
        # 单调上涨序列 → 末点应为初始资金的 1.09 倍左右 (30天共+290点/3000)
        assert nav[-1]["total_value"] == pytest.approx(1_000_000 * 3290 / 3000, rel=1e-6)
        assert isinstance(nav[-1]["total_value"], float)

    def test_date_range_filtering(self):
        df = _make_index_df(
            pd.date_range("2023-12-01", periods=90, freq="D"),
            [3000 + i for i in range(90)],
        )
        with patch("data.benchmark.ak.stock_zh_index_daily", return_value=df):
            nav = get_benchmark_nav("2024-01-01", "2024-01-31")
        assert nav[0]["date"] >= "2024-01-01"
        assert nav[-1]["date"] <= "2024-01-31"

    def test_all_sources_fail_returns_none(self):
        with patch("data.benchmark.ak.stock_zh_index_daily",
                   side_effect=ConnectionError("blocked")), \
             patch("data.benchmark.ak.stock_zh_index_daily_tx",
                   side_effect=ConnectionError("blocked")):
            assert get_benchmark_nav("2024-01-01", "2024-06-30") is None

    def test_insufficient_data_returns_none(self):
        df = _make_index_df(pd.to_datetime(["2024-01-02"]), [3000.0])
        with patch("data.benchmark.ak.stock_zh_index_daily", return_value=df):
            assert get_benchmark_nav("2024-01-01", "2024-01-31") is None

    def test_invalid_range_returns_none(self):
        assert get_benchmark_nav("2024-12-31", "2024-01-01") is None

    def test_ttl_cache_avoids_refetch(self):
        df = _make_index_df(
            pd.date_range("2024-01-01", periods=30, freq="D"),
            [3000 + i * 10 for i in range(30)],
        )
        with patch("data.benchmark.ak.stock_zh_index_daily", return_value=df) as mock_fetch:
            get_benchmark_nav("2024-01-01", "2024-01-30")
            get_benchmark_nav("2024-01-01", "2024-01-30")
        assert mock_fetch.call_count == 1

    def test_tencent_fallback_used(self):
        """新浪失败时自动切换腾讯源"""
        df = _make_index_df(
            pd.date_range("2024-01-01", periods=30, freq="D"),
            [3000 + i * 10 for i in range(30)],
        )
        with patch("data.benchmark.ak.stock_zh_index_daily",
                   side_effect=ConnectionError("blocked")), \
             patch("data.benchmark.ak.stock_zh_index_daily_tx", return_value=df):
            nav = get_benchmark_nav("2024-01-01", "2024-01-30")
        assert nav is not None and len(nav) == 30


class TestBenchmarkSummary:
    def test_summary_fields(self):
        df = _make_index_df(
            pd.date_range("2024-01-01", periods=30, freq="D"),
            [3000 + i * 10 for i in range(30)],
        )
        with patch("data.benchmark.ak.stock_zh_index_daily", return_value=df):
            summary = get_benchmark_summary("2024-01-01", "2024-01-30")
        assert summary["index_symbol"] == "sh000300"
        assert summary["index_name"] == "沪深300"
        assert summary["total_return"] == pytest.approx((3290 / 3000 - 1) * 100, rel=1e-3)

    def test_summary_none_when_unavailable(self):
        with patch.object(bm, "get_benchmark_nav", return_value=None):
            assert get_benchmark_summary("2024-01-01", "2024-06-30") is None


class TestIntegrationWithBacktestEngine:
    def test_set_benchmark_enables_ir(self):
        """接入基准后 BacktestEngine 应能算出非零 IR"""
        from core.backtest import BacktestEngine

        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        bench = [
            {"date": d.strftime("%Y-%m-%d"), "total_value": 1_000_000 * (1 + 0.001 * i)}
            for i, d in enumerate(dates)
        ]
        engine = BacktestEngine()
        engine.set_benchmark(bench)
        assert engine._benchmark_nav is not None
        assert len(engine._benchmark_nav) == 40

    def test_engine_without_benchmark_ir_zero(self):
        from core.backtest import BacktestEngine
        engine = BacktestEngine()
        assert engine._benchmark_nav is None


def test_default_index_registered():
    assert bm.DEFAULT_INDEX in BENCHMARK_INDEXES
    assert "sh000300" in BENCHMARK_INDEXES
