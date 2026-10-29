# -*- coding: utf-8 -*-
"""研究预算（core/ai_budget.py）的测试。

⚠️ 走 use_template_db 隔离真实库；persona 配置走 tmp_path 隔离真实 JSON。
"""
import pytest

from core import ai_budget as B
from core import ai_personas as AP
from db import store
from tests.conftest import use_template_db


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    use_template_db(tmp_path, monkeypatch, "ai_budget.db")
    monkeypatch.setattr(AP, "PERSONAS_PATH", tmp_path / "ai_personas.json")
    AP.load_personas()                     # 落盘默认值（供 quota_for 查 persona 覆盖）
    yield
    AP.invalidate()


class TestConsume:
    def test_grant_deny_and_view(self):
        """授予 → 用尽 → 拒绝；视图数字随之变化。

        ⚠️ 额度上限从配置读（`B.quota_for`），**不写死 3** —— deep_analysis 是
        可配置的成本护栏（实测单次 ≈3.5 万 token，2026-10-15 起 LLM persona 配为 1）。
        """
        quota = B.quota_for("ai_value", "deep_analysis")
        assert quota >= 1
        results = [B.consume("ai_value", "deep_analysis") for _ in range(quota)]
        assert all(ok for ok, _ in results), results
        ok_extra, v_extra = B.consume("ai_value", "deep_analysis")
        assert ok_extra is False
        assert results[0][1]["remaining"] == quota - 1
        assert v_extra["used"] == quota and v_extra["remaining"] == 0

    def test_unregistered_kind_is_rejected(self):
        """⚠️ 未登记的 kind 直接拒绝——不能让"新加一种资源"绕过配额登记。"""
        ok, v = B.consume("ai_value", "mystery_resource")
        assert ok is False and v["quota"] == 0

    def test_default_quotas_match_decisions(self):
        """决策 4（选 C）：deep_analysis=3 / wake=5 / decision_round=2。

        ⚠️ 2026-10-23：decision_round 1 → 2（午间 13:05 + 盘后 15:58）。
        一天只有一次机会时，模型说一次"等待"当天就归零（实测 3 个 LLM 人格
        7 个交易日仅 3 笔成交，随机组 15 笔）⇒ 频次被配额卡死，不是判断力问题。

        ⚠️ 2026-10-27：新增 `routine: 6`，与 `wake` **分池**。共用一池时，
        阿深当天有没有触发过条件会决定第 6 轮能不能跑——故障随行情出现，
        测试环境永远复现不了。
        """
        assert B.DEFAULT_QUOTA == {"deep_analysis": 3, "wake": 5, "routine": 6,
                                   "decision_round": 2}

    def test_persona_quota_override(self, tmp_path, monkeypatch):
        """persona 可用 quotas 字段覆盖个别 kind（如风控流更谨慎）。"""
        personas = AP.load_personas()
        personas[2]["quotas"] = {"deep_analysis": 1}       # ai_risk
        assert AP.save_personas(personas) is True
        ok1, _ = B.consume("ai_risk", "deep_analysis")
        ok2, _ = B.consume("ai_risk", "deep_analysis")
        assert ok1 is True and ok2 is False
        # 未覆盖的 kind 仍用默认
        assert B.quota_for("ai_risk", "wake") == 5

    def test_quota_for_unknown_persona_uses_defaults(self):
        assert B.quota_for("nobody", "deep_analysis") == 3

    def test_routine_pool_is_independent_from_wake(self):
        """`routine`（定时轮班）与 `wake`（条件触发）**互不抢占**。

        ⚠️ 这是分池的全部意义：共用一池时，"阿深当天挂了一个 price_below 条件"
        会吃掉一次定时轮班的额度，于是第 6 轮（14:40，收盘前最后一次调整）
        被静默跳过。**故障随行情出现**——有触发条件的日子才会少一轮，
        测试环境永远复现不了，线上表现为"偶尔收盘前没动作"。
        """
        # 先把 wake 用满
        wq = B.quota_for("ai_ashen", "wake")
        for _ in range(wq):
            assert B.consume("ai_ashen", "wake")[0] is True
        assert B.consume("ai_ashen", "wake")[0] is False, "wake 应已用尽"

        # routine 必须完全不受影响
        rq = B.quota_for("ai_ashen", "routine")
        assert rq == 6, "定时轮班数必须与 6 个 slot 对齐"
        for i in range(rq):
            ok, v = B.consume("ai_ashen", "routine")
            assert ok is True, f"第 {i + 1} 轮不该被拒：{v}"
        assert B.consume("ai_ashen", "routine")[0] is False

        # 视图里两个池各自记账，不会因为对方用尽而互相影响
        view = B.view("ai_ashen")
        assert view["routine"]["used"] == 6 and view["wake"]["used"] == wq


class TestView:
    def test_unused_kinds_visible(self):
        """⚠️ 「没用过」≠「额度 0」：视图必须把未用 kind 以配额默认值补全，
        否则 AI 看不到自己还有多少额度，无法学会省着用。"""
        B.consume("ai_value", "wake")                      # 只用过 wake
        v = B.view("ai_value")
        assert set(v) == {"deep_analysis", "wake", "routine", "decision_round"}
        # ⚠️ 断言口径：**满额 = 该 persona 的配置配额**（不写死 3）——
        #    deep_analysis 是可配置的成本护栏（2026-10-15 起 LLM persona 配为 1），
        #    写死数字会让"收紧成本"这种合理改动把用例弄红。
        full = B.quota_for("ai_value", "deep_analysis")
        assert v["deep_analysis"]["used"] == 0
        assert v["deep_analysis"]["remaining"] == full
        assert v["wake"]["used"] == 1

    def test_view_hides_unregistered_history_kinds(self):
        """历史遗留的未登记 kind 行不外泄（但也不炸）。"""
        store.consume_ai_budget("ai_value", "legacy_kind",
                                trade_date="2026-10-14", quota=9)
        v = B.view("ai_value", trade_date="2026-10-14")
        assert "legacy_kind" not in v

    def test_explicit_trade_date(self):
        B.consume("ai_value", "wake", trade_date="2026-10-13")
        old = B.view("ai_value", trade_date="2026-10-13")
        today = B.view("ai_value", trade_date="2026-10-14")
        assert old["wake"]["used"] == 1 and today["wake"]["used"] == 0


class TestNoSelfServiceQuota:
    def test_module_exposes_no_quota_setter(self):
        """⚠️ 结构性守护：AI 不能修改自己的配额 —— 本模块**不存在**任何
        「set/patch quota」入口（那类请求在工具层一律拒绝且记账）。"""
        public = [n for n in dir(B) if not n.startswith("_")]
        assert not any("set" in n.lower() and "quota" in n.lower() for n in public), public
        # consume 的签名里也没有 quota 参数（quota 由服务端定，AI 传不进来）
        import inspect
        sig = inspect.signature(B.consume)
        assert "quota" not in sig.parameters
