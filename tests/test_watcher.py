"""盘中异动预警轮询器测试（Phase 1）

覆盖：交易时段边界、非交易日/非时段跳过、命中后落库与推送、冷却去重与过期、
落库失败不影响结果、关闭推送时不推送。
"""

from datetime import datetime
from unittest.mock import patch

import pytest

import core.watcher as w
from core.alert_engine import AlertEngine
from tests.conftest import use_template_db


def _quote(code="600519", price=110.0, prev=100.0, pct=10.0):
    return {"code": code, "name": "贵州茅台", "price": price, "prev_close": prev,
            "change_pct": pct, "amount_wan": 0.0, "turnover_rate": None,
            "quote_time": "14:30:00", "source": "tencent", "stale": False}


def _watcher(**kw):
    return w.AlertWatcher(engine=AlertEngine(), cooldown_seconds=1800, **kw)


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """⚠️ 本文件原来**没有任何 DB 夹具**，于是 `watcher.check()` 会一路走到真实库：
    ① 读**真实库**的唤醒条件；② 命中后在**真实库**写 `ai_wake_events`；
    ③ 走 `record_and_fire(run_decision=True)` ⇒ **真调 LLM（真花钱）**。

    2026-10-16 实证：一次全量回归多出 1 次 `ai.ai_tech.wake`（821 tokens）+ 2 条
    `ai_wake_events`，全部落进真实库。conftest 的生产库哨兵只判"丢表"这类
    **结构级**破坏，行级污染不报警 —— 所以这个洞一直没被发现。

    隔离后 `armed_codes` 读到空条件表 ⇒ `evaluate` 根本不会被调用 ⇒ 零 LLM。
    """
    use_template_db(tmp_path, monkeypatch, "watcher.db")


def test_importing_watcher_applies_request_patch():
    """盯盘的行情请求也要走统一限频/请求头补丁，避免出现没人管的请求路径"""
    import data.akshare_helper as ah
    import core.watcher  # noqa: F401  （导入即应用补丁）
    assert ah._patched is True


class TestTradingSession:
    @pytest.mark.parametrize("hour, minute, expected", [
        (9, 29, False), (9, 30, True), (11, 30, True), (11, 31, False),
        (12, 0, False), (13, 0, True), (15, 0, True), (15, 1, False),
    ])
    def test_boundaries(self, hour, minute, expected):
        now = datetime(2026, 9, 22, hour, minute)   # 周二
        with patch.object(w, "is_trading_day", return_value=True):
            assert w.in_trading_session(now) is expected

    def test_holiday_is_not_session(self):
        with patch.object(w, "is_trading_day", return_value=False):
            assert w.in_trading_session(datetime(2026, 9, 22, 10, 0)) is False


class TestScanOnce:
    def test_skips_outside_session(self):
        with patch.object(w, "is_trading_day", return_value=True):
            r = _watcher().scan_once(codes=["600519"], now=datetime(2026, 9, 22, 12, 0))
        assert r["skipped"] is True and r["reason"] == "非交易时段"

    def test_skips_on_holiday(self):
        with patch.object(w, "is_trading_day", return_value=False):
            r = _watcher().scan_once(codes=["600519"], now=datetime(2026, 9, 22, 10, 0))
        assert r["skipped"] is True and r["reason"] == "非交易日"

    def test_skips_when_codes_empty(self):
        # ⚠️ 必须 mock 掉 AI 唤醒条件来源：它默认读**真实库**，而真实库里确实可能有
        #    AI 条件（正式运行产生的）⇒ 不 mock 就会让本用例依赖跑测试那台机器的状态
        #    （项目铁律：测试不能把"本机状态"当契约）。
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("core.ai_wake.armed_codes", return_value=[]):
            r = _watcher().scan_once(codes=[], now=datetime(2026, 9, 22, 10, 0))
        assert r["skipped"] is True and r["reason"] == "监控列表为空"

    def test_fires_persists_and_pushes(self):
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("data.realtime.get_realtime_quotes", return_value={"600519": _quote()}), \
             patch("db.store.insert_alerts", return_value=1) as mock_ins, \
             patch("core.push.push_markdown", return_value={"钉钉": "已推送"}) as mock_push:
            r = _watcher().scan_once(codes=["600519"], now=datetime(2026, 9, 22, 10, 0),
                                     stop_loss={})
        assert r["skipped"] is False
        assert r["fired"] >= 1 and r["saved"] == 1
        assert mock_ins.called and mock_push.called
        assert r["push"] == {"钉钉": "已推送"}

    def test_nothing_happens_when_no_alert(self):
        quiet = _quote(price=101.0, prev=100.0, pct=1.0)
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("data.realtime.get_realtime_quotes", return_value={"600519": quiet}), \
             patch("db.store.insert_alerts") as mock_ins, \
             patch("core.push.push_markdown") as mock_push:
            r = _watcher().scan_once(codes=["600519"], now=datetime(2026, 9, 22, 10, 0),
                                     stop_loss={})
        assert r["alerts"] == 0 and r["fired"] == 0
        mock_ins.assert_not_called()
        mock_push.assert_not_called()

    def test_cooldown_suppresses_repeat(self):
        now = datetime(2026, 9, 22, 10, 0)
        wat = _watcher()
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("data.realtime.get_realtime_quotes", return_value={"600519": _quote()}), \
             patch("db.store.insert_alerts", return_value=1), \
             patch("core.push.push_markdown", return_value={}):
            first = wat.scan_once(codes=["600519"], now=now, stop_loss={})
            second = wat.scan_once(codes=["600519"], now=now, stop_loss={})
        assert first["fired"] >= 1
        assert second["fired"] == 0, "冷却期内不得重复提醒"

    def test_cooldown_expires(self):
        wat = _watcher()
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("data.realtime.get_realtime_quotes", return_value={"600519": _quote()}), \
             patch("db.store.insert_alerts", return_value=1), \
             patch("core.push.push_markdown", return_value={}):
            wat.scan_once(codes=["600519"], now=datetime(2026, 9, 22, 10, 0), stop_loss={})
            later = wat.scan_once(codes=["600519"], now=datetime(2026, 9, 22, 10, 40),
                                  stop_loss={})
        assert later["fired"] >= 1, "超过冷却期后应再次提醒"

    def test_persist_failure_does_not_break_scan(self):
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("data.realtime.get_realtime_quotes", return_value={"600519": _quote()}), \
             patch("db.store.insert_alerts", side_effect=RuntimeError("db locked")), \
             patch("core.push.push_markdown", return_value={}):
            r = _watcher().scan_once(codes=["600519"], now=datetime(2026, 9, 22, 10, 0),
                                     stop_loss={})
        assert r["saved"] == 0 and r["fired"] >= 1

    def test_push_failure_does_not_break_scan(self):
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("data.realtime.get_realtime_quotes", return_value={"600519": _quote()}), \
             patch("db.store.insert_alerts", return_value=1), \
             patch("core.push.push_markdown", side_effect=RuntimeError("webhook down")):
            r = _watcher().scan_once(codes=["600519"], now=datetime(2026, 9, 22, 10, 0),
                                     stop_loss={})
        assert r["skipped"] is False and "错误" in r["push"]

    def test_push_can_be_disabled(self):
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("data.realtime.get_realtime_quotes", return_value={"600519": _quote()}), \
             patch("db.store.insert_alerts", return_value=1), \
             patch("core.push.push_markdown") as mock_push:
            _watcher(push=False).scan_once(codes=["600519"],
                                           now=datetime(2026, 9, 22, 10, 0), stop_loss={})
        mock_push.assert_not_called()

    def test_default_codes_used_when_none(self):
        """codes=None 时走默认监控范围（自选股 ∪ 今日候选）"""
        with patch.object(w, "is_trading_day", return_value=True), \
             patch.object(w, "_default_codes", return_value=["600519"]) as mock_codes, \
             patch("core.ai_wake.armed_codes", return_value=[]), \
             patch("data.realtime.get_realtime_quotes", return_value={"600519": _quote()}), \
             patch("db.store.insert_alerts", return_value=1), \
             patch("core.push.push_markdown", return_value={}):
            r = _watcher().scan_once(now=datetime(2026, 9, 22, 10, 0), stop_loss={})
        mock_codes.assert_called_once()
        assert r["codes"] == 1

    def test_ai_armed_codes_are_merged_into_monitored_codes(self):
        """⭐ S5：AI 已武装条件的代码必须并入监控范围（在**取数之前**），
        且不得改变盯盘原有行为。"""
        captured = {}

        def fake_quotes(codes, *a, **kw):
            captured["codes"] = list(codes)
            return {"600519": _quote(), "600028": _quote()}

        with patch.object(w, "is_trading_day", return_value=True), \
             patch.object(w, "_default_codes", return_value=["600519"]), \
             patch("core.ai_wake.armed_codes", return_value=["600028"]), \
             patch("data.realtime.get_realtime_quotes", side_effect=fake_quotes), \
             patch("db.store.insert_alerts", return_value=0), \
             patch("core.push.push_markdown", return_value={}), \
             patch("core.ai_wake.evaluate", return_value=[]):
            r = _watcher().scan_once(now=datetime(2026, 9, 22, 10, 0), stop_loss={})
        assert captured["codes"] == ["600519", "600028"], "AI 代码必须并入同一次取数"
        assert r["ai_armed_codes"] == 1


class TestManualForce:
    """force=True 供 UI 手动检查：跳过时段闸门，但如实标注数据口径"""

    def test_force_bypasses_gates(self):
        with patch.object(w, "is_trading_day", return_value=False), \
             patch("data.realtime.get_realtime_quotes", return_value={"600519": _quote()}), \
             patch("db.store.insert_alerts", return_value=1), \
             patch("core.push.push_markdown", return_value={}):
            r = _watcher().scan_once(codes=["600519"], now=datetime(2026, 9, 26, 20, 0),
                                     stop_loss={}, force=True)
        assert r["skipped"] is False and r["forced"] is True
        assert r["fired"] >= 1

    def test_force_marks_non_session_data(self):
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("data.realtime.get_realtime_quotes", return_value={"600519": _quote()}), \
             patch("db.store.insert_alerts", return_value=1), \
             patch("core.push.push_markdown", return_value={}):
            r = _watcher().scan_once(codes=["600519"], now=datetime(2026, 9, 22, 20, 0),
                                     stop_loss={}, force=True)
        assert "非交易时段" in r.get("note", "")

    def test_force_inside_session_has_no_note(self):
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("data.realtime.get_realtime_quotes", return_value={"600519": _quote()}), \
             patch("db.store.insert_alerts", return_value=1), \
             patch("core.push.push_markdown", return_value={}):
            r = _watcher().scan_once(codes=["600519"], now=datetime(2026, 9, 22, 10, 0),
                                     stop_loss={}, force=True)
        assert "note" not in r

    def test_non_forced_still_skips(self):
        """默认行为不变：非时段仍然跳过（force 只是手动入口）"""
        with patch.object(w, "is_trading_day", return_value=True):
            r = _watcher().scan_once(codes=["600519"], now=datetime(2026, 9, 22, 20, 0))
        assert r["skipped"] is True and r["reason"] == "非交易时段"
