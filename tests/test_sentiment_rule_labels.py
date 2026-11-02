# -*- coding: utf-8 -*-
"""rule 口径也要落逐条标签（2026-11-01 修）。

事故：`_persist_labels` 只在 `labeler="llm"` 分支调用 ⇒ 每天 15:20 的定时采集
只写日聚合快照，系统页「情绪标注留痕」表只在那天有人手动点过 LLM 判定时才有
数据。实测 2026-10-23 之后 9 天空白，页面看起来像"功能坏了"。

本文件钉两条：
1. `labeler="rule"`（定时任务路径）必须写逐条标签；
2. 写入必须幂等（同一天重复采集不翻倍）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import sentiment_reference as SR          # noqa: E402
from db import store as S                          # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    import db.models as M
    monkeypatch.setattr(M, "DB_PATH", str(tmp_path / "t.db"))
    S.init_store()
    return tmp_path / "t.db"


ITEMS = [
    {"text": "满仓了，感觉要反弹，看多", "title": "加仓"},
    {"text": "减仓跑一半，不看好后市", "title": "降低仓位"},
    {"text": "今天没什么想法，观望", "title": "观察"},
]


def test_rule_labeler_writes_逐条标签(tmp_db):
    """定时任务（labeler=rule）必须写 sentiment_labels，不能只写快照。"""
    res = SR.build_for_source(list(ITEMS), source="zhihu:member",
                              as_of="2026-11-01 15:20:00")
    assert res["snapshot"]["n_total"] == len(ITEMS)
    labels = S.get_sentiment_labels("2026-11-01", source="zhihu:member")
    assert len(labels) == len(ITEMS), "rule 分支没有写逐条标签（只剩日聚合）"
    assert {r["labeler"] for r in labels} == {"rule"}
    # stance 允许 None（词表判不出），但不能整列缺失
    assert all("stance" in r for r in labels)


def test_rule_labeler_idempotent(tmp_db):
    """同一天重复采集不翻倍（ON CONFLICT(trade_date,source,post_key,labeler)）。"""
    SR.build_for_source(list(ITEMS), source="zhihu:member",
                        as_of="2026-11-01 15:20:00")
    n1 = S.sentiment_label_summary(days=30)["total"]
    SR.build_for_source(list(ITEMS), source="zhihu:member",
                        as_of="2026-11-01 15:20:00")
    n2 = S.sentiment_label_summary(days=30)["total"]
    assert n1 == n2 == len(ITEMS)


def test_summary_reports_staleness(tmp_db):
    """汇总必须给 last_date / stale_days / stale_alert（页面标红依据）。"""
    empty = S.sentiment_label_summary(days=30)
    assert empty["last_date"] is None and empty["stale_alert"] is False
    SR.build_for_source(list(ITEMS), source="zhihu:member",
                        as_of="2026-11-01 15:20:00")
    s = S.sentiment_label_summary(days=30)
    assert s["last_date"] == "2026-11-01"
    assert s["stale_days"] is not None and s["stale_days"] >= 0
    assert s["stale_alert"] is False          # 今天刚写，不该告警


def test_stale_alert_triggers_for_old_data(tmp_db):
    """数据停在 4 天前 ⇒ stale_alert 必须为 True（页面标红）。"""
    S.save_sentiment_labels([{
        "trade_date": "2026-01-05", "source": "zhihu:member",
        "post_key": "k1", "labeler": "rule", "stance": "bull",
        "quote": "", "quote_ok": None, "why": "词表规则", "text": "看多"}])
    s = S.sentiment_label_summary(days=30)
    assert s["last_date"] == "2026-01-05"
    assert s["stale_days"] > 3
    assert s["stale_alert"] is True
