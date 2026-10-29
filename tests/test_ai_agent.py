# -*- coding: utf-8 -*-
"""工具层与护栏（core/ai_agent.py）的测试。"""
import random

import pytest

from core import ai_agent as A
from core.risk import RiskEngine


class TestClampPlan:
    def test_percent_and_fraction_both_normalized(self):
        assert A.clamp_plan({"position_size": "30%"}, "balanced")["position_size"] == 0.25
        assert A.clamp_plan({"position_size": 30}, "balanced")["position_size"] == 0.25
        assert A.clamp_plan({"position_size": 0.30}, "balanced")["position_size"] == 0.25

    def test_bias_multiplier_then_cap(self):
        # conservative: 0.2 × 0.5 = 0.10（未触上限）
        assert A.clamp_plan({"position_size": 0.2}, "conservative")["position_size"] == 0.10
        # aggressive: 0.9 → 夹到 0.25（上限先于系数生效）
        assert A.clamp_plan({"position_size": 0.9}, "aggressive")["position_size"] == 0.25

    def test_persona_single_limit_overrides(self):
        out = A.clamp_plan({"position_size": 0.2}, "aggressive", single_limit=0.10)
        assert out["position_size"] == 0.10

    def test_garbage_size_becomes_zero_not_crash(self):
        assert A.clamp_plan({"position_size": "很多"}, "balanced")["position_size"] == 0.0
        assert A.clamp_plan({}, "balanced")["position_size"] == 0.0

    def test_marks_clamped(self):
        out = A.clamp_plan({"position_size": 0.1, "direction": "看多"}, "balanced")
        assert out["direction"] == "看多" and out["position_clamped"] is True

    def test_bias_limits_table(self):
        assert A.bias_limits("conservative") == (0.5, 3, 2)
        assert A.bias_limits("unknown") == A.bias_limits("balanced")


class TestToolRegistry:
    def test_every_tool_has_desc_and_params(self):
        for t in A.TOOL_REGISTRY.values():
            assert t.desc, t.name

    def test_only_one_tool_costs_credits(self):
        """只有 request_deep_analysis 收费——工具面成本结构不许悄悄变。"""
        paid = {n for n, t in A.TOOL_REGISTRY.items() if t.cost > 0}
        assert paid == {"request_deep_analysis"}
        assert A.TOOL_REGISTRY["request_deep_analysis"].cost == 1

    def test_deep_analysis_not_allowed_in_wake(self):
        """盘中不做重活（计划 §3.4）——靠 allowed_in 白名单天然拒绝。"""
        ok, why = A.validate_tool_call("request_deep_analysis", {"code": "600028"},
                                       context="wake")
        assert ok is False and "不允许" in why


class TestValidateToolCall:
    def test_ok_call_passes(self):
        ok, why = A.validate_tool_call(
            "place_order", {"code": "600028", "direction": "买入",
                            "position_size": 0.1, "reason": "均线多头"})
        assert ok is True and why == ""

    def test_unknown_tool_rejected(self):
        ok, why = A.validate_tool_call("set_quota", {"quota": 99})
        assert ok is False and "未注册" in why

    def test_quota_smuggling_rejected(self):
        """⚠️ 越权红线：合法工具名 + 夹带配额参数 → 拒绝。"""
        ok, why = A.validate_tool_call(
            "place_order", {"code": "600028", "direction": "买入",
                            "position_size": 0.1, "reason": "x", "quota": 99})
        assert ok is False and "配额" in why

    def test_get_budget_is_read_only_allowed(self):
        ok, _ = A.validate_tool_call("get_my_budget", {})
        assert ok is True

    def test_pass_requires_reason(self):
        ok, why = A.validate_tool_call("pass", {})
        assert ok is False and "reason" in why
        ok2, _ = A.validate_tool_call("pass", {"reason": "今日无合适标的"})
        assert ok2 is True

    def test_wake_forbids_buy(self):
        """唤醒轮禁止买入加仓（盘中追涨是收益最容易被高估的动作）。"""
        ok, why = A.validate_tool_call(
            "place_order", {"code": "600028", "direction": "买入",
                            "position_size": 0.1, "reason": "x"}, context="wake")
        assert ok is False and "买入" in why
        # 卖出在唤醒轮是合法的
        ok2, _ = A.validate_tool_call(
            "sell", {"code": "600028", "reason": "止损触发"}, context="wake")
        assert ok2 is True

    def test_sell_not_available_in_daily(self):
        """日频轮的卖出走 place_order（T+1 口径）；wake 专属的 sell 不进日频。"""
        ok, _ = A.validate_tool_call("sell", {"code": "600028", "reason": "x"})
        assert ok is False

    def test_missing_required_param(self):
        ok, why = A.validate_tool_call("read_report", {})
        assert ok is False and "code" in why


class TestRandomDecisions:
    PERSONA = {"name": "ai_random_1", "kind": "random", "risk_bias": "balanced"}

    def test_deterministic_with_seed(self):
        """同一随机种子 → 完全一致的决策（可复现是对照组的生命线）。"""
        r1 = A.random_decisions(self.PERSONA, candidates=["600028", "000001"],
                                rng=random.Random(42))
        r2 = A.random_decisions(self.PERSONA, candidates=["600028", "000001"],
                                rng=random.Random(42))
        assert r1 == r2

    def test_never_empty(self):
        """没有动作也必须显式 pass（带理由）——不许静默。"""
        out = A.random_decisions(self.PERSONA, candidates=["600028"],
                                 rng=random.Random(7))
        assert out and all("reason" in d for d in out)

    def test_buy_respects_clamp(self):
        """随机组的仓位必须与 LLM 组**同一夹紧口径**（不留特殊通道，否则不可比）。"""
        for seed in range(50):
            out = A.random_decisions(self.PERSONA, candidates=["600028"],
                                     rng=random.Random(seed))
            for d in out:
                if d["action"] == "buy":
                    assert 0 < d["position_size"] <= RiskEngine.MAX_SINGLE_STOCK_WEIGHT

    def test_buy_comes_with_exit_discipline(self):
        """买入必须同时带止损+止盈条件——"只会买入的随机组"会让对照失去意义。"""
        bought = False
        for seed in range(50):
            out = A.random_decisions(self.PERSONA, candidates=["600028"],
                                     rng=random.Random(seed))
            if any(d["action"] == "buy" for d in out):
                bought = True
                kinds = {d["kind"] for d in out if d["action"] == "set_wake"}
                assert {"stop_loss", "take_profit"} <= kinds
        assert bought, "50 个种子里应至少有一次买入"

    def test_holding_may_be_sold(self):
        sold = False
        for seed in range(50):
            out = A.random_decisions(self.PERSONA, candidates=[],
                                     holdings=["600028"], rng=random.Random(seed))
            if any(d["action"] == "sell" for d in out):
                sold = True
        assert sold, "50 个种子里应至少有一次随机卖出"


class TestWakeThresholdOk:
    """唤醒阈值的**量级合理性**（唯一实现：写时与判时共用）。

    ⚠️ 这是"僵尸止损堵死代码兜底"那个真实缺陷的守护（2026-10-15 实测）。
    """

    def test_stop_loss_far_below_price_is_invalid(self):
        ok, why = A.wake_threshold_ok("stop_loss", 5.0, 1312.97)   # 茅台 + 5 元止损
        assert not ok and "永不触发" in why

    def test_take_profit_far_above_price_is_invalid(self):
        ok, _ = A.wake_threshold_ok("take_profit", 9999.0, 28.0)
        assert not ok

    def test_stop_loss_above_price_is_valid_not_invalid(self):
        """⚠️ 方向守卫：止损价**高于**现价 = 现价已跌破计划位 ⇒ 应**立即触发**，
        不是"无效"（首版曾把方向写反，测试抓到）。"""
        ok, _ = A.wake_threshold_ok("stop_loss", 10.0, 8.0)
        assert ok

    def test_near_price_is_valid(self):
        assert A.wake_threshold_ok("stop_loss", 5.2, 5.65)[0]
        assert A.wake_threshold_ok("take_profit", 6.2, 5.65)[0]

    def test_no_judgement_cases(self):
        """三种"不判"：非持仓类 / 无绝对阈值 / 无现价 —— 宁可放过也不误删。"""
        assert A.wake_threshold_ok("price_above", 1.0, 100.0)[0]
        assert A.wake_threshold_ok("stop_loss", None, 100.0)[0]      # 走 threshold_pct
        assert A.wake_threshold_ok("stop_loss", 5.0, None)[0]        # 没有现价

    def test_bad_values_rejected(self):
        ok, why = A.wake_threshold_ok("stop_loss", "abc", 10.0)
        assert not ok and "非数值" in why


class TestWakeBuyLimits:
    """盘中补仓额度（2026-10-23 由 1%/2%/2 次 调到 3%/6%/4 次）。

    ⚠️ 起因：实测 7 个交易日里 LLM 组对盘中唤醒几乎全是 pass ——
    ai_tech 一次唤醒里连拒 4 只。按 100 万权益算，1% 只有 1 万元，
    高价股折整手后经常不足 1 手直接判拒绝 ⇒ **限额小到无法执行 = 通道关闭**。
    """

    def test_default_limits_are_executable(self):
        lim = A.wake_buy_limits(None)
        assert lim["max_pct_of_equity"] == 0.03
        assert lim["max_daily_pct_of_equity"] == 0.06
        assert lim["max_orders_per_day"] == 4

    def test_single_topup_capped_by_position_value(self):
        ok, reason, amount = A.wake_buy_check(
            code="600028", kind="price_below", holdings=["600028"],
            position_value=200000, equity=1000000, cash=500000, price=5.0)
        assert ok, reason
        # min(持仓 20 万×3%=6000, 权益 100 万×3%=30000) ⇒ 6000
        assert abs(amount - 6000) <= 5.0 * 100     # 整手误差在一手之内

    def test_daily_budget_is_six_percent(self):
        """当日已用满 6 万（6%）⇒ 拒绝，不能"再给 6 万"。"""
        ok, reason, amount = A.wake_buy_check(
            code="600028", kind="price_below", holdings=["600028"],
            position_value=900000, equity=1000000, cash=500000, price=5.0,
            spent_today=60000)
        assert ok is False and amount == 0.0
        assert "额度已用完" in reason

    def test_remaining_budget_below_one_lot_is_refused(self):
        """只剩 1000 元、股价 50 元 ⇒ 折不出 1 手，宁可不动（不成交比"买 20 股"合理）。"""
        ok, reason, amount = A.wake_buy_check(
            code="600028", kind="price_below", holdings=["600028"],
            position_value=900000, equity=1000000, cash=500000, price=50.0,
            spent_today=59000)
        assert ok is False and amount == 0.0
        assert "不足 1 手" in reason

    def test_persona_override_still_works(self):
        lim = A.wake_buy_limits({"wake_buy_limits": {"max_pct_of_equity": 0.05}})
        assert lim["max_pct_of_equity"] == 0.05
        assert lim["max_daily_pct_of_equity"] == 0.06   # 未覆盖的保持默认


class TestMinEffectivePosition:
    """⭐ 2026-10-23：低于 5% 的"象征性建仓"会被抬到 5%（只抬不砍、不越单票上限）。

    实测模型爱给 2%~4%，扣掉最低佣金后没有盈亏意义，却把当天唯一一次
    `decision_round` 用掉了。资金利用率长期只有 8%~25%，这是直接原因之一。
    """

    def test_tiny_size_is_lifted_to_floor(self):
        out = A.clamp_plan({"position_size": 0.02}, "balanced")
        assert out["position_size"] == 0.05

    def test_zero_stays_zero(self):
        """pass/观望仍然必须是 0，不能被"抬"成一个仓位。"""
        assert A.clamp_plan({"position_size": 0.0}, "balanced")["position_size"] == 0.0
        assert A.clamp_plan({}, "balanced")["position_size"] == 0.0

    def test_floor_never_exceeds_style_limit(self):
        """保守风格的 0.5 系数把 3% 压到 1.5%，但上限 5% ⇒ 抬到 min(5%, limit)。"""
        out = A.clamp_plan({"position_size": 0.03}, "conservative", single_limit=0.03)
        assert out["position_size"] == 0.03, "上限比下限更严时必须服从上限"

    def test_normal_size_untouched(self):
        assert A.clamp_plan({"position_size": 0.12}, "balanced")["position_size"] == 0.12
