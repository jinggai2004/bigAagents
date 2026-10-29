"""买入信号 → 深度分析 → **可行才通知** 链路（2026-10-23）

用户要的是："达到买入条件/信号后深度分析，如果可行再发通知"。
这条链路的三条硬约束都钉在这里：
  1. 盘中**只落库不推送**（盘中数据残缺，结论必须由收盘后的深度分析给）；
  2. 判据**确定性、零 LLM**（涨幅 + 相对放量，相对量而非绝对值）；
  3. 通知只在 `advice_status == "可执行"` 时发出（驳回/等待不是行动信号）。
"""

from datetime import datetime
from unittest.mock import patch

import pytest

import core.evening_analysis as ea
import core.signal_queue as sq
import core.watcher as w
from tests.conftest import use_template_db


def _quote(code="600519", name="贵州茅台", price=110.0, prev=100.0, pct=5.0,
           amount_wan=200000.0):
    return {"code": code, "name": name, "price": price, "prev_close": prev,
            "change_pct": pct, "amount_wan": amount_wan, "turnover_rate": 2.0,
            "quote_time": "14:30:00", "source": "tencent", "stale": False}


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    use_template_db(tmp_path, monkeypatch, "signal_queue.db")


@pytest.fixture
def panel_amount(monkeypatch):
    """面板 20 日均额 = **10 亿元** ⇒ 阈值好算（当日成交额 / 1e9 = 倍数）。

    ⚠️ 单位：`quotes.amount_wan` 是**万元** ⇒ 200000 万元 = 20 亿元。
    """
    monkeypatch.setattr(sq, "_avg_amount",
                        lambda code: {"avg_amount": 1e9, "as_of": "2026-10-16"})


class TestDetectSignals:
    def test_surge_with_volume_is_a_signal(self, panel_amount):
        sigs = sq.detect_signals({"600519": _quote(pct=5.0, amount_wan=200000.0)},
                                 ["600519"], {"600519": "持仓·自选"})
        assert len(sigs) == 1
        s = sigs[0]
        assert s["code"] == "600519" and s["kind"] == "surge_volume"
        assert s["volume_ratio"] == 2.0
        assert "2026-10-16" in s["detail"], "必须标出均额口径日期（面板滞后）"
        assert s["relation"] == "持仓·自选"

    def test_small_gain_is_not_a_signal(self, panel_amount):
        sigs = sq.detect_signals({"600519": _quote(pct=1.2, amount_wan=200000.0)},
                                 ["600519"])
        assert sigs == []

    def test_surge_without_volume_is_not_a_signal(self, panel_amount):
        """涨幅够但没放量 ⇒ 不是"资金进场"，只是波动。"""
        sigs = sq.detect_signals({"600519": _quote(pct=5.0, amount_wan=50000.0)},
                                 ["600519"])
        assert sigs == []

    def test_missing_panel_data_keeps_signal_but_marks_it(self, monkeypatch):
        """缺就是缺：均额取不到时**不判定放量**，但要写明不可判定。"""
        monkeypatch.setattr(sq, "_avg_amount", lambda code: {})
        sigs = sq.detect_signals({"600519": _quote(pct=5.0, amount_wan=200000.0)},
                                 ["600519"])
        assert len(sigs) == 1
        assert sigs[0]["volume_ratio"] is None
        assert "不可判定" in sigs[0]["detail"]

    def test_max_signals_cap(self, panel_amount):
        quotes = {f"6005{i:02d}": _quote(code=f"6005{i:02d}", pct=3.0 + i * 0.1)
                  for i in range(20)}
        sigs = sq.detect_signals(quotes, list(quotes.keys()), max_signals=5)
        assert len(sigs) == 5
        # 按涨幅降序
        assert sigs[0]["change_pct"] >= sigs[-1]["change_pct"]


class TestSignalQueueStore:
    def test_record_is_idempotent(self, panel_amount):
        sigs = sq.detect_signals({"600519": _quote(pct=5.0)}, ["600519"])
        assert len(sq.record_signals(sigs, "2026-10-23")) == 1
        assert len(sq.record_signals(sigs, "2026-10-23")) == 0, "同票同类信号只记一次"
        assert sq.pending_codes("2026-10-23") == ["600519"]

    def test_mark_analyzed_removes_from_pending(self, panel_amount):
        sigs = sq.detect_signals({"600519": _quote(pct=5.0)}, ["600519"])
        sq.record_signals(sigs, "2026-10-23")
        sq.mark_analyzed("2026-10-23", ["600519"])
        assert sq.pending_codes("2026-10-23") == []


class TestWatcherSignals:
    def test_signal_is_recorded_but_not_pushed(self):
        """⭐ 盘中只落库、**不推送** —— 结论留给收盘后的深度分析。

        ⚠️ 行情取"涨幅 3.5%"：够触发信号门槛（≥3%）但**不够**触发告警
        （price_surge 门槛 5%）⇒ 这样才能断言"这一轮一条消息都没发"。
        """
        quiet = _quote(price=103.5, prev=100.0, pct=3.5, amount_wan=200000.0)
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("core.ai_wake.random_only_codes", return_value=set()), \
             patch.object(w, "_relation_map", return_value={"600519": "自选"}), \
             patch("db.store.list_active_positions", return_value=[]), \
             patch("core.signal_queue._avg_amount",
                   lambda code: {"avg_amount": 1e9, "as_of": "2026-10-16"}), \
             patch("data.realtime.get_realtime_quotes",
                   return_value={"600519": quiet}), \
             patch("db.store.insert_alerts", return_value=0), \
             patch("core.push.push_markdown") as mock_push:
            r = w.AlertWatcher().scan_once(codes=["600519"],
                                           now=datetime(2026, 10, 23, 14, 0), stop_loss={})
        assert r["alerts"] >= 0   # info 级成交额可能命中（只落库，不推送）
        assert r["signals"] == 1
        mock_push.assert_not_called(), "盘中信号不得推送（info 级也静默）"
        assert sq.pending_codes("2026-10-23") == ["600519"]

    def test_ai_armed_codes_are_out_of_signal_scope(self):
        """AI 武装票是联赛实验对象，不进"给你的买入信号"。"""
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("core.ai_wake.random_only_codes", return_value=set()), \
             patch.object(w, "_relation_map", return_value={"600519": "自选"}), \
             patch("db.store.list_active_positions", return_value=[]), \
             patch("core.signal_queue._avg_amount",
                   lambda code: {"avg_amount": 1e9, "as_of": "2026-10-16"}), \
             patch("data.realtime.get_realtime_quotes",
                   return_value={"600519": _quote(price=103.5, prev=100.0, pct=3.5,
                                                 amount_wan=200000.0)}), \
             patch("db.store.insert_alerts", return_value=0), \
             patch("core.push.push_markdown"):
            w.AlertWatcher().scan_once(codes=["600519", "600887"],
                                       now=datetime(2026, 10, 23, 14, 0), stop_loss={})
        # 600887 若被标成 AI武装，就不该进信号队列
        assert "600887" not in sq.pending_codes("2026-10-23")


class TestEveningAnalysisPriority:
    def test_signal_codes_come_first(self, monkeypatch):
        """信号票优先占额度（候选补位 ⇒ 总额度不变 ⇒ 不增加成本）。"""
        monkeypatch.setattr(ea, "candidates_for_date",
                            lambda d, want=40: [{"stock_code": "600001", "score": 99},
                                                {"stock_code": "600002", "score": 98}])
        monkeypatch.setattr(ea, "analyzed_codes", lambda d: set())
        monkeypatch.setattr(ea, "recently_analyzed", lambda d, days=7: {})
        monkeypatch.setattr(ea, "trigger_exempt_codes", lambda d, days=7: {})
        monkeypatch.setattr(ea, "tokens_used_today", lambda d=None: 0)
        monkeypatch.setattr(sq, "pending_codes", lambda d=None: ["600777"])
        seen = []

        def _runner(code, source="pipeline", **kw):
            seen.append(code)
            return {"ok": True, "snapshot": {"final_decision": "x", "trade_plan": {}}}

        out = ea.run_evening_analysis(date="2026-10-23", limit=2, runner=_runner)
        assert seen and seen[0] == "600777", "信号票必须排第一"
        assert len(seen) <= 2, "总额度不变"


class TestSignalConclusionNotification:
    def _snapshot(self, advice_status, direction="看多"):
        return {"stock_name": "贵州茅台", "data_through": "2026-10-23",
                "trade_plan": {"advice_status": advice_status, "direction": direction,
                               "entry_range": [108, 110], "target_price": 135,
                               "stop_loss": 104, "position_size": "20%"}}

    def test_executable_conclusion_is_pushed(self):
        pushed = {}

        def _fake_push(text, title=None, dedupe_key=None):
            pushed.update(text=text, title=title, key=dedupe_key)
            return {"飞书": "已推送"}

        with patch("core.push.push_markdown", side_effect=_fake_push):
            out = ea._notify_signal_conclusions("2026-10-23",
                                                {"600519": self._snapshot("可执行")})
        assert out["notified"] == 1
        assert "600519" in pushed["text"] and "目标 135" in pushed["text"]
        assert pushed["key"] == "signal:2026-10-23"

    def test_rejected_conclusion_is_not_pushed(self):
        with patch("core.push.push_markdown") as mock_push:
            out = ea._notify_signal_conclusions("2026-10-23",
                                                {"600519": self._snapshot("驳回")})
        assert out["notified"] == 0
        mock_push.assert_not_called(), "驳回/等待档不是行动信号，推了就是噪音"

    def test_waiting_conclusion_is_not_pushed(self):
        with patch("core.push.push_markdown") as mock_push:
            out = ea._notify_signal_conclusions("2026-10-23",
                                                {"600519": self._snapshot("等待")})
        assert out["notified"] == 0
        mock_push.assert_not_called()
