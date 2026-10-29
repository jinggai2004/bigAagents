"""盘中异动预警规则引擎测试（Phase 1）"""

import pytest

from core.alert_engine import AlertEngine, format_alerts_markdown, limit_pct


def _q(code="600519", price=100.0, prev=100.0, pct=0.0, amount_wan=0.0,
       turnover=None, stale=False, age=0.0, source="tencent", name="测试股份"):
    return {"code": code, "name": name, "price": price, "prev_close": prev,
            "change_pct": pct, "amount_wan": amount_wan, "turnover_rate": turnover,
            "quote_time": "14:30:00", "source": source, "stale": stale,
            "age_seconds": age}


def _rules(alerts, rule):
    return [a for a in alerts if a["rule"] == rule]


class TestLimitPct:
    @pytest.mark.parametrize("code,expected", [
        ("300750", 0.20), ("301536", 0.20), ("688981", 0.20),
        ("600519", 0.10), ("000001", 0.10),
    ])
    def test_by_board(self, code, expected):
        assert limit_pct(code) == expected


class TestPriceRules:
    def test_surge(self):
        alerts = AlertEngine().evaluate({"600519": _q(price=106.0, prev=100.0, pct=6.0)})
        assert _rules(alerts, "price_surge")

    def test_plunge(self):
        alerts = AlertEngine().evaluate({"600519": _q(price=94.0, prev=100.0, pct=-6.0)})
        assert _rules(alerts, "price_plunge")

    def test_normal_move_has_no_alert(self):
        assert AlertEngine().evaluate({"600519": _q(price=101.0, prev=100.0, pct=1.0)}) == []

    def test_limit_up_main_board(self):
        alerts = AlertEngine().evaluate({"600519": _q(price=110.0, prev=100.0, pct=10.0)})
        hit = _rules(alerts, "limit_up")
        assert hit and hit[0]["level"] == "danger"

    def test_growth_board_uses_20pct_limit(self):
        """创业板 20% 涨停：涨 10% 不算涨停，涨 20% 才算（与回测同口径）"""
        eng = AlertEngine()
        ten = eng.evaluate({"300750": _q(code="300750", price=110.0, prev=100.0, pct=10.0)})
        assert not _rules(ten, "limit_up")
        twenty = eng.evaluate({"300750": _q(code="300750", price=120.0, prev=100.0, pct=20.0)})
        assert _rules(twenty, "limit_up")

    def test_limit_down(self):
        alerts = AlertEngine().evaluate({"600519": _q(price=90.0, prev=100.0, pct=-10.0)})
        assert _rules(alerts, "limit_down")


class TestVolumeTurnoverAndConfig:
    def test_volume_spike(self):
        alerts = AlertEngine().evaluate({"600519": _q(amount_wan=120_000.0)})   # 12 亿
        assert _rules(alerts, "volume_spike")

    def test_turnover_spike(self):
        alerts = AlertEngine().evaluate({"600519": _q(turnover=12.5)})
        assert _rules(alerts, "turnover_spike")

    def test_thresholds_are_configurable(self):
        eng = AlertEngine(rules={"change_pct_surge": 1.0})
        assert _rules(eng.evaluate({"600519": _q(price=101.5, prev=100.0, pct=1.5)}), "price_surge")


class TestMa20AndStopLoss:
    def test_break_ma20_only_on_cross(self):
        eng = AlertEngine()
        up = eng.evaluate({"600519": _q(price=101.0, prev=99.0, pct=2.0)},
                          ma20={"600519": 100.0})
        assert _rules(up, "break_ma20")
        # 已在 MA20 上方停留 → 不重复报
        stay = eng.evaluate({"600519": _q(price=105.0, prev=104.0, pct=1.0)},
                            ma20={"600519": 100.0})
        assert not _rules(stay, "break_ma20")

    def test_breakdown_ma20(self):
        alerts = AlertEngine().evaluate({"600519": _q(price=99.0, prev=101.0, pct=-2.0)},
                                        ma20={"600519": 100.0})
        assert _rules(alerts, "breakdown_ma20")

    def test_stop_loss_is_danger(self):
        alerts = AlertEngine().evaluate({"600519": _q(price=95.0, prev=96.0, pct=-1.0)},
                                        stop_loss={"600519": 95.5})
        hit = _rules(alerts, "stop_loss")
        assert hit and hit[0]["level"] == "danger"

    def test_missing_external_data_means_no_judgement(self):
        """没有 MA20/止损数据就不判定，而不是猜一个"""
        alerts = AlertEngine().evaluate({"600519": _q(price=99.0, prev=101.0, pct=-2.0)})
        assert not _rules(alerts, "breakdown_ma20") and not _rules(alerts, "stop_loss")


class TestStaleSafety:
    def test_stale_quote_only_emits_data_stale(self):
        """安全阀：过期缓存不得参与异动判定。

        否则数据源一故障就会按旧价刷出一堆假预警——这正是 0.4 加 stale 标记要防的事。
        """
        alerts = AlertEngine().evaluate(
            {"600519": _q(price=110.0, prev=100.0, pct=10.0, stale=True, age=600)})
        assert [a["rule"] for a in alerts] == ["data_stale"]
        assert alerts[0]["level"] == "warn"

    def test_missing_price_or_empty_row_ignored(self):
        assert AlertEngine().evaluate({"600519": _q(price=None)}) == []
        assert AlertEngine().evaluate({"600519": {}}) == []
        assert AlertEngine().evaluate({}) == []
        assert AlertEngine().evaluate(None) == []


class TestOrderingAndFormat:
    def test_danger_sorted_first(self):
        alerts = AlertEngine().evaluate({
            "600519": _q(price=110.0, prev=100.0, pct=10.0, amount_wan=200_000.0)})
        order = {"danger": 0, "warn": 1, "info": 2}
        levels = [a["level"] for a in alerts]
        assert levels == sorted(levels, key=lambda x: order[x])
        assert alerts[0]["level"] == "danger"

    def test_format_contains_key_fields(self):
        alerts = AlertEngine().evaluate({"600519": _q(price=110.0, prev=100.0, pct=10.0)})
        text = format_alerts_markdown(alerts)
        assert "600519" in text and "测试股份" in text and "程序化监控" in text

    def test_format_empty(self):
        assert "未命中" in format_alerts_markdown([])

    def test_format_truncates(self):
        alerts = [{"code": str(i), "name": "x", "rule": "r", "level": "info",
                   "message": "m", "price": 1.0, "change_pct": 0.0, "quote_time": ""}
                  for i in range(30)]
        text = format_alerts_markdown(alerts, limit=5)
        assert "另有 25 条" in text
