"""相似历史日模块单元测试（合成K线数据，临时库隔离）"""
import numpy as np
import pandas as pd
import pytest

import data.similar_days as sd
from db import store as db_store


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    db_path = use_template_db(tmp_path, monkeypatch, "test_similar.db")
    yield


def make_kline(closes, start="2018-01-01"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"date": dates, "close": closes,
                         "volume": np.linspace(1e6, 2e6, len(closes))})


class TestBuildFeatures:
    def test_shape_and_columns(self):
        closes = 100 + np.cumsum(np.random.RandomState(0).normal(0, 1, 200))
        feats = sd.build_features(make_kline(closes))
        assert len(feats) == 200 - 59  # 前59行因 rolling(60) 首个有效在索引59
        for key in sd.FEATURE_KEYS:
            assert key in feats.columns
        assert feats[[*sd.FEATURE_KEYS]].notna().all().all()

    def test_trend_signature(self):
        """单调上涨序列：ret_1d>0、连涨、均线多头偏离为正"""
        closes = np.linspace(100, 200, 120)
        feats = sd.build_features(make_kline(closes))
        last = feats.iloc[-1]
        assert last["ret_1d"] > 0
        assert last["streak"] > 0
        assert last["ma5_dev"] > 0 and last["ma20_dev"] > 0

    def test_missing_close_raises(self):
        with pytest.raises(ValueError):
            sd.build_features(pd.DataFrame({"day": [1, 2]}))


class TestBackfillAndSearch:
    def test_backfill_and_find(self, temp_store, monkeypatch):
        """回填→检索→含统计分布与免责声明"""
        rng = np.random.RandomState(42)
        closes = 3000 + np.cumsum(rng.normal(0, 20, 300))
        monkeypatch.setattr(sd, "_fetch_daily", lambda s, years=10: make_kline(closes))
        result = sd.backfill_history("sh000001", years=10)
        assert result["days"] == 300 - 59

        search = sd.find_similar_days("sh000001", top_k=5)
        assert "error" not in search
        assert len(search["similar"]) == 5
        assert search["stats"]["后5日"]["样本数"] == 5
        assert "不构成预测" in search["disclaimer"]
        # 目标日自身不在相似列表
        assert all(s["date"] != search["target"]["date"] for s in search["similar"])

    def test_overlap_exclusion(self, temp_store, monkeypatch):
        """目标日近5日应被排除（窗口重叠）"""
        rng = np.random.RandomState(7)
        closes = 3000 + np.cumsum(rng.normal(0, 20, 300))
        monkeypatch.setattr(sd, "_fetch_daily", lambda s, years=10: make_kline(closes))
        sd.backfill_history("sh000001")
        history = db_store.load_features("sh000001")
        target_date = history[-1]["date"]

        search = sd.find_similar_days("sh000001", top_k=10)
        dates = [s["date"] for s in search["similar"]]
        recent_dates = [h["date"] for h in history[-6:]]
        assert not set(dates) & set(recent_dates)  # 最后6天（含目标）全被排除

    def test_insufficient_history(self, temp_store):
        search = sd.find_similar_days("sz399001")
        assert "error" in search
        assert "回填" in search["error"]

    def test_target_date_filter(self, temp_store, monkeypatch):
        rng = np.random.RandomState(1)
        closes = 3000 + np.cumsum(rng.normal(0, 20, 300))
        monkeypatch.setattr(sd, "_fetch_daily", lambda s, years=10: make_kline(closes))
        sd.backfill_history("sh000001")
        history = db_store.load_features("sh000001")
        mid_date = history[150]["date"]
        search = sd.find_similar_days("sh000001", target_date=mid_date, top_k=3)
        assert search["target"]["date"] == mid_date
        # 相似日必须早于目标日（未来数据泄露防护）
        assert all(s["date"] <= mid_date for s in search["similar"])
