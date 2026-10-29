"""盘中异动预警落库测试（Phase 1）

轮询是每分钟一次，所以"同交易日 + 同标的 + 同规则 + 同消息只记一行"的去重必须成立，
否则盘中会刷出几百条重复记录。
"""

import pytest


def _temp_store(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    db_path = use_template_db(tmp_path, monkeypatch, "alerts.db")
    from db import store
    return store


def _alert(**over):
    base = {"code": "600519", "name": "贵州茅台", "rule": "limit_up", "level": "danger",
            "message": "涨停 +10.00%", "price": 110.0, "change_pct": 10.0,
            "quote_time": "14:30:00", "source": "tencent"}
    base.update(over)
    return base


class TestAlertsStore:
    def test_insert_and_read_back(self, tmp_path, monkeypatch):
        store = _temp_store(tmp_path, monkeypatch)
        assert store.insert_alerts([_alert()], trade_date="2026-09-22") == 1
        rows = store.get_recent_alerts(trade_date="2026-09-22")
        assert len(rows) == 1
        assert rows[0]["code"] == "600519"
        assert rows[0]["level"] == "danger"
        assert rows[0]["price"] == pytest.approx(110.0)

    def test_duplicate_suppressed_same_day(self, tmp_path, monkeypatch):
        store = _temp_store(tmp_path, monkeypatch)
        assert store.insert_alerts([_alert()], trade_date="2026-09-22") == 1
        assert store.insert_alerts([_alert()], trade_date="2026-09-22") == 0
        assert len(store.get_recent_alerts(trade_date="2026-09-22")) == 1

    def test_same_message_next_day_allowed(self, tmp_path, monkeypatch):
        store = _temp_store(tmp_path, monkeypatch)
        store.insert_alerts([_alert()], trade_date="2026-09-22")
        assert store.insert_alerts([_alert()], trade_date="2026-09-23") == 1

    def test_other_rule_or_code_not_deduped(self, tmp_path, monkeypatch):
        store = _temp_store(tmp_path, monkeypatch)
        store.insert_alerts([_alert()], trade_date="2026-09-22")
        assert store.insert_alerts([_alert(rule="price_surge")], trade_date="2026-09-22") == 1
        assert store.insert_alerts([_alert(code="000001")], trade_date="2026-09-22") == 1

    def test_empty_list_is_noop(self, tmp_path, monkeypatch):
        store = _temp_store(tmp_path, monkeypatch)
        assert store.insert_alerts([], trade_date="2026-09-22") == 0

    def test_missing_optional_fields_tolerated(self, tmp_path, monkeypatch):
        """预警字段缺失不应导致写库失败（QUOTE 里某些源没有换手率等）"""
        store = _temp_store(tmp_path, monkeypatch)
        n = store.insert_alerts([{"code": "600519", "rule": "volume_spike",
                                  "level": "info", "message": "成交额 12.0 亿"}],
                                trade_date="2026-09-22")
        assert n == 1
        assert store.get_recent_alerts(trade_date="2026-09-22")[0]["name"] is None

    def test_read_returns_only_requested_day(self, tmp_path, monkeypatch):
        store = _temp_store(tmp_path, monkeypatch)
        store.insert_alerts([_alert()], trade_date="2026-09-22")
        store.insert_alerts([_alert(message="另一天")], trade_date="2026-09-23")
        assert len(store.get_recent_alerts(trade_date="2026-09-22")) == 1
        assert len(store.get_recent_alerts(trade_date="2026-09-23")) == 1
