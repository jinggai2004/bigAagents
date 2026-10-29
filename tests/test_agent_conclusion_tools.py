# -*- coding: utf-8 -*-
"""结论类薄包装工具测试（2026-10-24 审查建议落地）——tool_quality_stats / tool_llm_cost /
tool_system_health / tool_decision_digest / tool_holdings_risk。

审查的验收模板（每条都测）：
    ① as_of + 口径写进返回值；
    ② "没有"与"0"分开（样本不足/缺失/就是 0 三态）；
    ③ n < 门槛不报结论——实测 bull_samples=4 报 100%，包装层必须拦；
    ④ 进 _TOOL_RESULT_CHAR_LIMITS（不截断）；
    ⑤ 正常 + 空/越界 两条用例。
这些工具是"页面同款函数"的薄包装：store/llm_cost/scheduler_health/decision_digest/
position_risk 自身都有测试，这里只 mock 这些叶子函数测包装层语义（口径/门槛/裁剪）。
"""

import pytest

from core.agent_tools import TOOL_FUNCS, TOOL_SCHEMAS


# ================================================================
# 1. tool_quality_stats：样本门槛是核心契约
# ================================================================

def _fake_analysis_stats(bull_n, bull_win_rate):
    def _fn(recent_days=30):
        return {"recent_days": recent_days, "total": bull_n, "by_label": {"看多": {"partial": bull_n}},
                "bull_samples": bull_n,
                "bull_win_rate": bull_win_rate if bull_n else None,
                "bear_samples": 0, "bear_win_rate": None, "pending_backfill": 0}
    return _fn


def test_quality_stats_gates_small_sample(monkeypatch):
    """实测坑：bull_samples=4 时 win_rate=1.0——包装层必须只报样本数、不报胜率。"""
    from db import store
    monkeypatch.setattr(store, "analysis_stats", _fake_analysis_stats(4, 1.0))
    monkeypatch.setattr(store, "screen_pick_stats",
                        lambda d: {"recent_days": d, "samples": 0, "avg_ret_5d": None,
                                   "win_rate_5d": None, "excess_samples": 0,
                                   "avg_excess_5d": None, "excess_win_rate": None,
                                   "pending_backfill": 0})
    r = TOOL_FUNCS["tool_quality_stats"](kind="analysis")
    assert r["看多"]["样本数"] == 4
    assert r["看多"]["胜率"] is None, "n<10 绝不报胜率（100% 是噪声）"
    assert "不作结论" in r["看多"]["结论门槛"]
    assert "口径" in r, "必须带口径"


def test_quality_stats_reports_when_enough(monkeypatch):
    from db import store
    monkeypatch.setattr(store, "analysis_stats", _fake_analysis_stats(12, 0.67))
    monkeypatch.setattr(store, "screen_pick_stats",
                        lambda d: {"recent_days": d, "samples": 54, "avg_ret_5d": 0.002,
                                   "win_rate_5d": 0.63, "excess_samples": 54,
                                   "avg_excess_5d": 0.018, "excess_win_rate": 0.80,
                                   "pending_backfill": 3})
    r = TOOL_FUNCS["tool_quality_stats"](kind="overview")
    assert r["深度分析"]["看多胜率"]["胜率"] == "67%"
    assert r["选股"]["T+5胜率"]["胜率"] == "63%"
    assert r["选股"]["待回填"] == 3, "待回填单列，不是失败也不是 0"
    assert "提醒" in r


def test_quality_stats_empty_and_unknown_kind(monkeypatch):
    from db import store
    empty = {"recent_days": 90, "samples": 0, "avg_ret_5d": None, "win_rate_5d": None,
             "excess_samples": 0, "avg_excess_5d": None, "excess_win_rate": None,
             "pending_backfill": 0}
    monkeypatch.setattr(store, "screen_pick_stats", lambda d: dict(empty, recent_days=d))
    monkeypatch.setattr(store, "analysis_stats", _fake_analysis_stats(0, None))
    r = TOOL_FUNCS["tool_quality_stats"](kind="picks")
    assert r["样本数"] == 0
    assert "不可评" in r["T+5胜率"]["胜率"], "空样本是'不可评'不是 0%"
    assert "error" in TOOL_FUNCS["tool_quality_stats"](kind="nope")


# ================================================================
# 2. tool_llm_cost
# ================================================================

def _fake_daily_report(days=7, until=None):
    return {"days": [{"date": "2026-10-24", "cost": 0.087, "calls": 13,
                      "tokens": 120000, "failed": 1}],
            "by_purpose": [{"key": "chat.tool_loop", "cost": 0.05, "calls": 6, "tokens": 60000}],
            "by_model": [{"key": "deepseek-v4-flash", "cost": 0.087, "calls": 13, "tokens": 120000}],
            "total_cost": 0.087, "total_calls": 13, "total_tokens": 120000,
            "avg_workday_cost": 0.268, "workday_count": 4}


def test_llm_cost_normal_and_budget(monkeypatch):
    from core import llm_cost
    monkeypatch.setattr(llm_cost, "daily_report", _fake_daily_report)
    monkeypatch.setattr(llm_cost, "budget_status", lambda **kw: {
        "cap": 5.0, "spent": 0.0875, "remaining": 4.9125, "ratio": 0.0175,
        "exceeded": False, "enabled": True, "degraded": False})
    r = TOOL_FUNCS["tool_llm_cost"](days=3)
    assert r["总花费元"] == 0.087 and r["总调用"] == 13
    assert r["预算"]["剩余元"] == 4.9125 and r["预算"]["已超限"] is False
    assert "估算" in r["口径"], "估算口径必须写明（非账单）"


def test_llm_cost_days_clamped(monkeypatch):
    from core import llm_cost
    seen = {}
    def _fake(days=7, until=None):
        seen["days"] = days
        return _fake_daily_report(days)
    monkeypatch.setattr(llm_cost, "daily_report", _fake)
    monkeypatch.setattr(llm_cost, "budget_status", lambda **kw: {"degraded": True})
    TOOL_FUNCS["tool_llm_cost"](days=9999)
    assert seen["days"] == 30, "days 必须夹紧到上限"
    r = TOOL_FUNCS["tool_llm_cost"](days=9999)
    assert r["预算"]["账目异常"] is True, "degraded 要透出，不是'没花钱'"


# ================================================================
# 3. tool_system_health
# ================================================================

def test_system_health_maps_and_caps(monkeypatch):
    from core import scheduler_health
    seen = {}
    def _fake(check_processes=True, check_schedule_gap=True):
        seen["processes"] = check_processes
        return {"scheduler": {"running": False, "reason": "心跳停止", "ts": "x", "idle_seconds": 600},
                "code": {"stale": True, "missing": ["core/a.py", "core/b.py"], "extra": []},
                "custom_jobs": {"missing": ["custom_x"]},
                "schedule_gap": {"max_gap_minutes": 120},
                "long_running": {"stale": ["ui"]},
                "data": {"big_table": {}},
                "warnings": [f"警告{i}" for i in range(20)]}
    monkeypatch.setattr(scheduler_health, "check_health", _fake)
    r = TOOL_FUNCS["tool_system_health"]()
    assert seen["processes"] is True, "生产工具必须开进程新鲜度检查"
    assert r["调度器"]["运行中"] is False
    assert r["代码新鲜度"]["比代码旧的起始文件"] == ["core/a.py", "core/b.py"]
    assert len(r["需要处理"]) == 8, "警告必须封顶（防截断）"
    assert "data" not in r, "大表数据不带进对话"


# ================================================================
# 4. tool_decision_digest
# ================================================================

def test_decision_digest_empty_is_not_failure(monkeypatch):
    from core import decision_digest
    monkeypatch.setattr(decision_digest, "today_decisions",
                        lambda as_of=None: {"date": "2026-10-24", "screens": [],
                                            "analyses": [], "positions": [], "trades": [],
                                            "empty_reasons": ["熊市：市场评分低于阈值，跳过选股"]})
    monkeypatch.setattr(decision_digest, "action_list",
                        lambda as_of=None: {"watch": [], "risks": [], "positions": []})
    r = TOOL_FUNCS["tool_decision_digest"]()
    assert r["选股"]["候选数"] == 0
    assert r["选股"]["跳过原因"] == ["熊市：市场评分低于阈值，跳过选股"], "空候选必须带原因"
    assert "说明" in r, "空≠故障，说明必须有"


def test_decision_digest_caps_lists(monkeypatch):
    from core import decision_digest
    monkeypatch.setattr(decision_digest, "today_decisions", lambda as_of=None: {
        "date": "2026-10-24", "screens": [{"c": i} for i in range(30)], "analyses": [],
        "positions": [], "trades": []})
    monkeypatch.setattr(decision_digest, "action_list", lambda as_of=None: {
        "watch": [{"w": i} for i in range(30)], "risks": [], "positions": []})
    r = TOOL_FUNCS["tool_decision_digest"]()
    assert len(r["选股"]["候选"]) <= 10 and len(r["待触发观察"]) <= 10, "明细封顶防截断"


# ================================================================
# 5. tool_holdings_risk
# ================================================================

def _fake_eval(prices=None, snapshot_date=None):
    return {"snapshot_date": "2026-10-23", "total_equity": 100000.0, "cash": 30000.0,
            "cash_ratio": 0.3, "position_ratio": 0.7,
            "top_position": {"code": "600519", "ratio": 0.25},
            "sector_exposure": {"白酒": 0.25},
            "positions": [{"code": "600519"}], "issues": ["单票集中度偏高"],
            "limits": [{"code": "600519", "stop": 1200.0}],
            "data_unavailable": [], "missing_industry": [], "missing_snapshot": [],
            "unavailable_liquidity": ["600519"]}


def test_holdings_risk_normal(monkeypatch):
    from core import position_risk
    from data import realtime
    monkeypatch.setattr(position_risk, "build_holdings",
                        lambda prices=None, snapshot_date=None: {
                            "holdings": [{"code": "600519"}], "total_equity": 100000,
                            "cash": 30000, "snapshot_date": "2026-10-23",
                            "missing_snapshot": [], "unavailable": []})
    seen = {}
    monkeypatch.setattr(realtime, "get_realtime_quotes",
                        lambda codes: seen.setdefault("codes", codes) or {"600519": {"price": 1300}})
    monkeypatch.setattr(position_risk, "evaluate_positions", _fake_eval)
    r = TOOL_FUNCS["tool_holdings_risk"]()
    assert r["仓位占比"] == 0.7 and r["现金占比"] == 0.3
    assert r["风险问题"] == ["单票集中度偏高"]
    assert r["不可评说明"]["流动性不可评"] == ["600519"], "不可评单列"


def test_holdings_risk_error_path(monkeypatch):
    from core import position_risk
    def _boom(**kw):
        raise RuntimeError("快照不可用")
    monkeypatch.setattr(position_risk, "build_holdings", _boom)
    r = TOOL_FUNCS["tool_holdings_risk"]()
    assert "error" in r and "快照不可用" in r["error"], "失败给 error，不编结论"


# ================================================================
# 6. 注册与配额
# ================================================================

def test_conclusion_tools_registered_and_limited():
    from core.chat_agent import _TOOL_RESULT_CHAR_LIMITS
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    for n in ("tool_quality_stats", "tool_llm_cost", "tool_system_health",
              "tool_decision_digest", "tool_holdings_risk"):
        assert n in names and n in TOOL_FUNCS, f"{n} 未注册"
        assert n in _TOOL_RESULT_CHAR_LIMITS, f"{n} 未进字符配额表（会被1500截断）"
    # 工具数量是**成本护栏**：每个 schema 都进每次请求的 prompt，不能无节制追加。
    # 2026-10-28 加上 `tool_ashen_status`（阿深第三套账本）后**用满 30** ⇒
    # 之后再要加工具，必须先合并或下线一个，不能直接改这个数字了事。
    assert len(names) == 30, "工具数 30（预算上限 30，已用满）"
