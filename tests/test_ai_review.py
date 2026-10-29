# -*- coding: utf-8 -*-
"""每日复盘组装（`core/ai_review.py`）的测试。

⭐ 核心验收：
  ① 每行同时给出**今日操作理由**（来自 ai_decisions.order_reason）与**前一日复盘**；
  ② **AI 自述**与**账本事实**分栏（不合并）——自述缺失时如实显示"未留下自述"，不编造；
  ③ 净值变化由账本两点计算；缺数据返回 None（不拿 0 冒充）；
  ④ 离线模式（`BIGA_UI_OFFLINE=1`）不碰交易日历（否则会联网挂起）。

⚠️ 全离线：库走 use_template_db（不 mock 网络——本模块本就不联网）。
"""
import pytest

from core import ai_review as R, ai_personas
from db import store
from tests.conftest import use_template_db


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    use_template_db(tmp_path, monkeypatch, "ai_review.db")
    monkeypatch.setattr(ai_personas, "PERSONAS_PATH", tmp_path / "ai_personas.json")
    ai_personas.load_personas()
    yield
    ai_personas.invalidate()


def test_review_includes_today_reason_and_prev_day():
    store.save_ai_decision({"persona": "ai_value", "signal_date": "2026-10-14",
                            "code": "600028", "advice_status": "可执行",
                            "direction": "看多", "position_size": 0.2,
                            "order_reason": "原油涨价利好炼化"})
    store.save_ai_decision({"persona": "ai_value", "signal_date": "2026-10-13",
                            "code": "600000", "advice_status": "等待",
                            "position_size": 0.0, "order_reason": "等待回调"})
    store.save_ai_review("ai_value", "2026-10-14",
                         review_text="昨日按兵不动是对的。", summary="今天加仓石化")

    rv = R.daily_review("2026-10-14", personas=[ai_personas.get_persona("ai_value")])
    row = rv["rows"][0]
    assert row["today_decisions"][0]["reason"] == "原油涨价利好炼化"
    assert row["ai_review_text"] == "昨日按兵不动是对的。"
    assert row["today_summary"] == "今天加仓石化"
    # 非离线（本机有日历缓存）时：前一交易日应推断出来并带出昨日决策
    assert row["prev_date"], "前一交易日应可推断（除非离线）"
    assert any(d["code"] == "600000" for d in row["prev_decisions"])


def test_review_does_not_fabricate_when_missing():
    """没有任何账户/复盘数据时：如实返回空，不编造净值与自述。"""
    rv = R.daily_review("2026-10-14", personas=[ai_personas.get_persona("ai_risk")])
    row = rv["rows"][0]
    assert row["ai_review_text"] in (None, "")
    assert row["nav_today"] is None
    assert row["nav_change_pct"] is None
    assert row["positions"] == []


def test_nav_change_computed_from_two_points():
    """净值变化必须由**账本两点**算出（today vs prev），缺一为 None。"""
    from core.paper_account import PaperAccount
    acct = PaperAccount(1_000_000, name="ai_value")
    acct.mark_to_market("2026-10-13", {})
    acct.cash = 990_000
    acct.mark_to_market("2026-10-14", {})
    store.save_paper_state("ai_value", acct.to_state())

    row = R.persona_review("ai_value", "2026-10-14", "2026-10-13",
                           persona=ai_personas.get_persona("ai_value"))
    assert row["nav_change_pct"] is not None
    assert abs(row["nav_change_pct"] - (-1.0)) < 0.01


def test_offline_skips_trading_calendar(monkeypatch):
    """⭐ 离线模式必须短路交易日历（取数源无超时 ⇒ 缓存过期时会挂起页面）。"""
    monkeypatch.setenv("BIGA_UI_OFFLINE", "1")
    assert R._prev_trading_day("2026-10-14") is None


def test_nav_change_hidden_when_valuation_incomplete():
    """⭐ 任一日净值含"按成本估值"的持仓 ⇒ **不给变化率**（两个口径相减 = 假数字）。

    实测场景（2026-10-15）：10-14 是真实市价、10-15 退化成成本价兜底 ⇒
    AI 把"成本价恰好≈初始资金"读成"净值不变、没赚没亏"。
    """
    from core.paper_account import PaperAccount
    acct = PaperAccount(1_000_000, name="ai_value")
    acct.mark_to_market("2026-10-13", {})
    acct.cash = 990_000
    rec = acct.mark_to_market("2026-10-14", {})
    rec["at_cost"] = 1                      # 标成"估值不完整"
    store.save_paper_state("ai_value", acct.to_state())

    row = R.persona_review("ai_value", "2026-10-14", "2026-10-13",
                           persona=ai_personas.get_persona("ai_value"))
    assert row["nav_change_pct"] is None, "口径不干净时不得给变化率"
    assert row["nav_note"] and "按成本估值" in row["nav_note"]


def test_nav_change_given_when_valuation_clean():
    """口径干净（两天都是真实市价）时照常给变化率。"""
    from core.paper_account import PaperAccount
    acct = PaperAccount(1_000_000, name="ai_value")
    acct.mark_to_market("2026-10-13", {})
    acct.cash = 990_000
    acct.mark_to_market("2026-10-14", {})
    store.save_paper_state("ai_value", acct.to_state())

    row = R.persona_review("ai_value", "2026-10-14", "2026-10-13",
                           persona=ai_personas.get_persona("ai_value"))
    assert row["nav_change_pct"] is not None
    assert row["nav_note"] is None
