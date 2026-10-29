# -*- coding: utf-8 -*-
"""情绪快照的持久化层：相对基准依赖它，所以"读回来的到底是什么"必须钉死。

⭐ 这里最重要的一条是 `test_history_excludes_same_day`：
   把当日自己算进"近 20 日均值"，偏离度会被稀释成 1/20 ——
   当日再异常也看不出来。这是**自比较陷阱**，且不会报错、不会 flaky，
   只会让整个相对基准层悄悄失效。
"""
from __future__ import annotations

import pytest

from db import store


@pytest.fixture(autouse=True)
def _db(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "sentiment.db")


def _snap(date: str, bull_ratio: float, source: str = "t", quality: str = "ok",
          n_total: int = 40):
    n_scored = int(n_total * 0.8)
    bull = int(n_scored * bull_ratio)
    return {
        "source": source, "as_of": f"{date} 15:00:00",
        "window_start": f"{date} 09:00:00", "window_end": f"{date} 15:00:00",
        "n_total": n_total, "n_scored": n_scored,
        "bull": bull, "bear": n_scored - bull, "neutral": 0,
        "unscored": n_total - n_scored,
        "bull_ratio": bull_ratio, "dispersion": 0.3,
        "quality": quality, "notes": [], "samples": [],
    }


class TestSaveSentimentSnapshot:
    def test_roundtrip(self):
        assert store.save_sentiment_snapshot(_snap("2026-10-20", 0.6))
        hist = store.get_sentiment_history("t", days=10)
        assert len(hist) == 1
        assert hist[0]["bull_ratio"] == pytest.approx(0.6)
        assert hist[0]["source"] == "t"

    def test_same_day_overwrites_not_accumulates(self):
        """同一天跑两次 ⇒ 后者覆盖前者（快照是状态，不是事件流）。"""
        store.save_sentiment_snapshot(_snap("2026-10-20", 0.4))
        store.save_sentiment_snapshot(_snap("2026-10-20", 0.7))
        hist = store.get_sentiment_history("t", days=10)
        assert len(hist) == 1, "同一天不应出现两行"
        assert hist[0]["bull_ratio"] == pytest.approx(0.7)

    def test_missing_source_is_rejected(self):
        bad = _snap("2026-10-20", 0.5)
        bad["source"] = ""
        assert store.save_sentiment_snapshot(bad) is False

    def test_sources_are_isolated(self):
        """不同来源不能串味：股吧的情绪比绝不能进知乎的基准。"""
        store.save_sentiment_snapshot(_snap("2026-10-20", 0.2, source="zhihu:member"))
        store.save_sentiment_snapshot(_snap("2026-10-20", 0.9, source="guba:600519"))
        zh = store.get_sentiment_history("zhihu:member")
        gb = store.get_sentiment_history("guba:600519")
        assert [h["bull_ratio"] for h in zh] == [pytest.approx(0.2)]
        assert [h["bull_ratio"] for h in gb] == [pytest.approx(0.9)]


class TestGetSentimentHistory:
    def test_history_excludes_same_day(self):
        """⭐ 自比较陷阱：当日必须被 `< before_date` 排除掉。"""
        for d, r in (("2026-10-20", 0.3), ("2026-10-21", 0.35),
                     ("2026-10-22", 0.9), ("2026-10-23", 0.95)):
            store.save_sentiment_snapshot(_snap(d, r))
        hist = store.get_sentiment_history("t", days=20, before_date="2026-10-23")
        got = sorted(h["bull_ratio"] for h in hist)
        assert got == [pytest.approx(0.3), pytest.approx(0.35), pytest.approx(0.9)]
        assert all(h["as_of"][:10] != "2026-10-23" for h in hist)

    def test_empty_quality_rows_are_skipped(self):
        """空样本日（quality=empty）不参与基准 —— 否则"可用天数"虚高。"""
        store.save_sentiment_snapshot(_snap("2026-10-20", 0.5))
        empty = _snap("2026-10-21", 0.0, quality="empty")
        empty["bull_ratio"] = None
        store.save_sentiment_snapshot(empty)
        hist = store.get_sentiment_history("t", days=20)
        assert len(hist) == 1

    def test_newest_first_and_limited(self):
        for i in range(1, 8):
            store.save_sentiment_snapshot(_snap(f"2026-10-{10 + i:02d}", 0.1 * i))
        hist = store.get_sentiment_history("t", days=3)
        assert len(hist) == 3
        assert hist[0]["as_of"].startswith("2026-10-17"), "应按日期倒序"

    def test_unknown_source_returns_empty(self):
        store.save_sentiment_snapshot(_snap("2026-10-20", 0.5))
        assert store.get_sentiment_history("nope") == []
