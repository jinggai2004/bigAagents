# -*- coding: utf-8 -*-
"""盘后决策轮（core/ai_daily_round.py）的测试。

⭐ 三条核心验收：
  ① **公平性**：公共简报对所有 AI 逐字节相同（私有段才允许不同）；
  ② **随机组零 LLM**：random persona 的轮次绝不创建 LLM 客户端；
  ③ **幂等与额度**：同日重跑被 decision_round 额度挡下；LLM 失败**不落库**。
⚠️ 全离线：LLM 用假 client（脚本化返回），库走 use_template_db。
"""
import json

import pytest

from core import ai_agent, ai_budget, ai_personas, ai_daily_round as R
from db import store
from tests.conftest import use_template_db


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    use_template_db(tmp_path, monkeypatch, "ai_round.db")
    monkeypatch.setattr(ai_personas, "PERSONAS_PATH", tmp_path / "ai_personas.json")
    # ⚠️ 决策轮现在要取**当日收盘价**（`ai_league.close_prices`：快照优先 → 日线兜底）。
    #    测试一律禁止联网 ⇒ 把日线兜底置空，只让快照生效；用例需要价就自己写
    #    `store.save_market_snapshot`（这也是"快照是权威来源"的自然表达）。
    from core import ai_league
    monkeypatch.setattr(ai_league, "daily_bars", lambda codes, d, **kw: {})
    ai_personas.load_personas()
    yield
    ai_personas.invalidate()


class FakeClient:
    """脚本化假客户端：按顺序吐出预设回复，并记录每次调用的 purpose/model。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def call_api(self, messages, **kw):
        from llm.deepseek_client import _MODEL_OVERRIDE
        self.calls.append({"purpose": kw.get("purpose"),
                           "model_override": _MODEL_OVERRIDE.get()})
        if not self.replies:
            return '{"final": {"actions": [{"action": "pass", "reason": "没话说了"}], "summary": "s"}}'
        return self.replies.pop(0)


def _persona(name="ai_value", **kw):
    p = ai_personas.get_persona(name) or {"name": name, "kind": "llm"}
    return {**p, **kw}


class TestBriefingFairness:
    def test_common_briefing_is_deterministic(self):
        """⚠️ 公平性铁律：同一交易日组装两次，JSON 必须**逐字节相同**。"""
        a = json.dumps(R.build_common_briefing("2026-10-14"), ensure_ascii=False, sort_keys=True)
        b = json.dumps(R.build_common_briefing("2026-10-14"), ensure_ascii=False, sort_keys=True)
        assert a == b

    def test_common_has_no_private_fields(self):
        """公共简报里**不许**出现任何人的持仓/净值/预算——否则泄露的就是别人的底牌。"""
        c = json.dumps(R.build_common_briefing("2026-10-14"), ensure_ascii=False)
        for k in ("holdings", "nav", "budget", "positions"):
            assert f'"{k}"' not in c

    def test_empty_db_degrades_honestly(self):
        """没有选股/新闻产物时如实给空 + warnings，不补造、不抛异常。"""
        c = R.build_common_briefing("2026-10-14")
        assert c["screen_picks"] == [] and c["reports"] == []
        assert isinstance(c["warnings"], list)

    def test_private_overlay_is_per_persona(self):
        o1 = R._private_overlay("ai_value", "2026-10-14")
        # 私有段 = 持仓/净值/预算 + **复盘所需的自身轨迹**
        # （recent_fills / prev_decisions / nav_recent / prev_trading_day）。
        # ⚠️ 新增字段必须在此显式登记：这条断言的意图是"私有段只含这些、
        #    且**不含**公共简报字段"，改集合时别顺手放宽成字典包含式断言。
        # `exposure`（2026-10-23 加）：当前仓位 = 持仓市值÷总权益，
        # 与公共段 `market_signal.target_exposure` 对照才有意义。
        assert set(o1) == {"holdings", "nav", "exposure", "nav_at_cost", "nav_note",
                           "budget", "recent_fills", "prev_decisions", "nav_recent",
                           "prev_trading_day"}
        # `routine`（2026-10-27 加）：定时轮班额度，与 `wake`（条件触发）**分池**。
        # 必须出现在简报的预算段里，否则阿深看不到自己今天还剩几轮定时，
        # 会误以为"额度已经用完"而主动放弃后面的轮次。
        assert set(o1["budget"]) == {"deep_analysis", "wake", "routine", "decision_round"}

    def test_private_overlay_values_at_close_not_cost(self):
        """⭐ 给 AI 看的净值必须按**当日收盘价**估值（不是成本价）。

        旧版用 `{c: avg_cost}` ⇒ 净值恒等于成本 ⇒ AI 读出"净值没变"，写出
        "未产生盈亏"的错误复盘（2026-10-15 实测 ai_tech / ai_value 都这么错）。
        """
        from core.paper_account import PaperAccount
        acct = PaperAccount(1_000_000, name="ai_value")
        acct.apply_trade({"date": "2026-10-13", "code": "600028",
                          "direction": "买入", "price": 5.0, "quantity": 1000})
        store.save_paper_state("ai_value", acct.to_state())
        store.save_market_snapshot("2026-10-14", [{"stock_code": "600028", "price": 6.00}])
        o = R._private_overlay("ai_value", "2026-10-14")
        assert o["nav_at_cost"] == 0
        h = o["holdings"][0]
        assert h["price"] == 6.0, h
        # 成本含手续费（avg_cost ≈ 5.0075）⇒ 涨幅略低于 20%，断言区间而非精确值
        assert 19.0 < h["pnl_pct"] < 20.0, h
        # 成本口径的特征是**恰好 = 初始资金**（买入时现金 −成本、持仓 +成本）。
        # 旧 bug 正是用它冒充净值 ⇒ 断言"净值必须≠成本口径、且因股价上涨而更高"。
        cost_basis = acct.cash + sum(p.total_cost for p in acct.positions.values())
        assert abs(cost_basis - 1_000_000) < 1, cost_basis
        assert o["nav"] != cost_basis, "净值被按成本口径计算（2026-10-15 修掉的 bug）"
        assert o["nav"] > cost_basis, o["nav"]
        assert "收盘价" in (o["nav_note"] or "")

    def test_private_overlay_marks_at_cost_when_price_missing(self):
        """取不到当日价 ⇒ 净值里**必须留下 at_cost 标记**，否则 AI 会把假净值当真的用。"""
        from core.paper_account import PaperAccount
        acct = PaperAccount(1_000_000, name="ai_value")
        acct.apply_trade({"date": "2026-10-13", "code": "600028",
                          "direction": "买入", "price": 5.0, "quantity": 1000})
        store.save_paper_state("ai_value", acct.to_state())
        o = R._private_overlay("ai_value", "2026-10-14")
        assert o["nav_at_cost"] == 1
        assert o["holdings"][0]["price"] is None
        assert "不可用于盈亏判断" in o["nav_note"]


class TestParse:
    def test_tool_and_final_forms(self):
        assert R.parse_round_reply('{"tool": "get_quote", "args": {"code": "1"}}')["tool"] == "get_quote"
        assert "final" in R.parse_round_reply('{"final": {"actions": []}}')

    def test_tolerates_surrounding_text(self):
        got = R.parse_round_reply('好的，我的决定是：\n```json\n{"final": {"actions": []}}\n```')
        assert got and "final" in got

    def test_garbage_returns_none(self):
        assert R.parse_round_reply("") is None
        assert R.parse_round_reply("完全不是 JSON") is None
        assert R.parse_round_reply("[1,2,3]") is None
        assert R.parse_round_reply('{"other": 1}') is None


class TestNormalize:
    def test_buy_is_clamped(self):
        out = R.normalize_actions([{"action": "buy", "code": "600028",
                                    "position_size": "80%", "reason": "x"}],
                                 _persona(risk_bias="conservative"))
        assert out[0]["position_size"] == 0.25           # 80%×0.5=40% > 25% ⇒ 夹到上限

    def test_duplicate_buy_deduped(self):
        out = R.normalize_actions([{"action": "buy", "code": "600028", "reason": "a"},
                                   {"action": "buy", "code": "600028", "reason": "b"}],
                                  _persona())
        assert len(out) == 1

    def test_unknown_action_and_missing_code_dropped(self):
        out = R.normalize_actions([{"action": "teleport", "code": "1"},
                                   {"action": "buy"}], _persona())
        assert out == []

    def test_pass_kept_with_reason(self):
        out = R.normalize_actions([{"action": "pass", "reason": "无合适标的"}], _persona())
        assert out[0]["action"] == "pass" and out[0]["reason"] == "无合适标的"

    def test_set_wake_normalized(self):
        out = R.normalize_actions([{"action": "set_wake_condition", "code": "600028",
                                    "kind": "stop_loss", "threshold": 5.0}], _persona())
        assert out[0]["action"] == "set_wake" and out[0]["kind"] == "stop_loss"

    def test_random_stop_pct_becomes_threshold_pct(self):
        """⚠️ 实测缺陷守卫：随机组给的是**幅度**（stop_pct），若只认 threshold
        会让条件静默变成空壳（threshold=None ⇒ 永远判不出来）。"""
        out = R.normalize_actions([{"action": "set_wake", "code": "600887",
                                    "kind": "stop_loss", "stop_pct": 0.08},
                                   {"action": "set_wake", "code": "600887",
                                    "kind": "take_profit", "take_pct": 0.15}], _persona())
        assert out[0]["threshold_pct"] == 0.08 and out[1]["threshold_pct"] == 0.15

    def test_llm_absolute_threshold_still_works(self):
        """LLM 组给绝对价时 threshold 仍然保留（两种口径并存，互不覆盖）。"""
        out = R.normalize_actions([{"action": "set_wake", "code": "600028",
                                    "kind": "price_above", "threshold": 6.0}], _persona())
        assert out[0]["threshold"] == 6.0 and out[0]["threshold_pct"] is None

    def test_buy_not_overwritten_by_set_wake(self):
        """⭐⭐ 主键覆盖缺陷守卫：同一 code 的 buy + set_wake 不得互相覆盖。
        决策行必须仍是 buy（可执行 + 非零仓位）——首版把 set_wake 也写进决策表，
        后写的 set_wake 会把 buy 覆盖成"等待/0 仓位"，买入意图**静默丢失**。"""
        res = {"persona": "ai_value", "summary": "买",
               "decisions": [{"action": "buy", "code": "600887",
                              "position_size": 0.1, "reason": "看多"},
                             {"action": "set_wake", "code": "600887",
                              "kind": "stop_loss", "threshold_pct": 0.08}]}
        R.persist_round(res, {"name": "ai_value", "model": "qwen-max"},
                        trade_date="2026-10-14")
        rows = store.list_ai_decisions(persona="ai_value", signal_date="2026-10-14")
        assert len(rows) == 1, "同一 code 只应有一行决策"
        assert rows[0]["advice_status"] == "可执行" and rows[0]["position_size"] == 0.1
        conds = store.list_ai_wake_conditions(persona="ai_value", trade_date="2026-10-14")
        assert len(conds) == 1 and conds[0]["threshold_pct"] == 0.08

    def test_buy_without_stop_gets_system_default(self):
        """⚠️ 实测缺陷守卫：提示词要求"买入必带止损"，但模型**并不总是遵守**
        （实测 ai_value 买 600028 没给任何止损）⇒ 代码兜底补 -8% 并标记来源。"""
        decisions = [{"action": "buy", "code": "600028", "position_size": 0.2,
                      "reason": "x"}]
        R.ensure_stops(decisions, _persona())
        stops = [d for d in decisions if d["action"] == "set_wake"]
        assert len(stops) == 1 and stops[0]["code"] == "600028"
        assert stops[0]["threshold_pct"] == R.DEFAULT_STOP_PCT
        assert "系统兜底" in stops[0]["reason"]      # 可与"AI 自己定的"区分

    def test_buy_with_stop_untouched(self):
        """AI 自己给了止损就不许覆盖（兜底只补空缺，不改 AI 的意图）。"""
        decisions = [{"action": "buy", "code": "600028", "position_size": 0.2},
                     {"action": "set_wake", "code": "600028", "kind": "stop_loss",
                      "threshold": 5.0}]
        R.ensure_stops(decisions, _persona())
        stops = [d for d in decisions if d["action"] == "set_wake"]
        assert len(stops) == 1 and stops[0]["threshold"] == 5.0

    def test_invalid_stop_dropped_and_replaced_by_default(self):
        """⭐⭐ **僵尸止损**回归（2026-10-15 实测缺陷）。

        `ai_tech` 给贵州茅台（现价千元级）写 `stop_loss threshold=5.0`：
        盘中 `ai_wake` 判它无效不触发，而 `ensure_stops` 又因"已存在 stop_loss"
        **不补 −8%** ⇒ 持仓实际裸奔。修法：**先剔除无效条件，再补兜底**。
        """
        decisions = [{"action": "buy", "code": "600519", "position_size": 0.2},
                     {"action": "set_wake", "code": "600519", "kind": "stop_loss",
                      "threshold": 5.0}]
        recorded: list = []
        R.ensure_stops(decisions, _persona(), prices={"600519": 1312.97},
                       record=recorded)
        stops = [d for d in decisions if d["action"] == "set_wake"]
        assert len(stops) == 1, "无效止损应被剔除、且补上一条兜底"
        assert stops[0]["threshold"] is None and stops[0]["threshold_pct"] == R.DEFAULT_STOP_PCT
        assert "系统兜底" in stops[0]["reason"]
        assert recorded and "永不触发" in recorded[0]

    def test_valid_stop_kept_when_price_given(self):
        """现价在合理范围内 ⇒ 保留 AI 自己的止损（校验不能误伤正常条件）。"""
        decisions = [{"action": "buy", "code": "600028", "position_size": 0.2},
                     {"action": "set_wake", "code": "600028", "kind": "stop_loss",
                      "threshold": 5.2}]
        R.ensure_stops(decisions, _persona(), prices={"600028": 5.65})
        stops = [d for d in decisions if d["action"] == "set_wake"]
        assert len(stops) == 1 and stops[0]["threshold"] == 5.2

    def test_persist_keeps_pct(self):
        """端到端：落库后 threshold_pct 必须可查（否则就是"条件空壳"回归）。"""
        res = {"persona": "ai_random_1",
               "decisions": [{"action": "set_wake", "code": "600887",
                              "kind": "stop_loss", "threshold_pct": 0.08}]}
        R.persist_round(res, {"name": "ai_random_1"}, trade_date="2026-10-14")
        conds = store.list_ai_wake_conditions(persona="ai_random_1",
                                              trade_date="2026-10-14")
        assert conds and conds[0]["threshold_pct"] == 0.08


class TestExecuteTool:
    def _ctx(self):
        return {"persona": _persona(), "common": R.build_common_briefing("2026-10-14"),
                "private": {"holdings": [], "nav": 100, "budget": {}},
                "trade_date": "2026-10-14"}

    def test_unregistered_rejected(self):
        r = R.execute_tool("set_quota", {"quota": 9}, **self._ctx())
        assert r["ok"] is False

    def test_quota_smuggling_rejected(self):
        r = R.execute_tool("place_order", {"code": "600028", "direction": "买入",
                                          "position_size": 0.1, "reason": "x",
                                          "limit": 1}, **self._ctx())
        assert r["ok"] is False and "配额" in r["error"]

    def test_read_report_miss_tells_you_what_to_do(self):
        r = R.execute_tool("read_report", {"code": "600028"}, **self._ctx())
        assert r["ok"] is False and "request_deep_analysis" in r["error"]

    def test_deep_analysis_consumes_credit(self):
        """⭐ 收费工具：**有执行器时**先扣 credit；额度耗尽后拒绝。
        （无执行器时不扣——见 test_no_runner_is_honest_not_fake）

        ⚠️ 额度上限**从配置读**（`ai_budget.quota_for`），不写死数字：deep_analysis
        是可配置的成本护栏（2026-10-15 起 LLM persona 配为 1，因实测单次 ≈3.5 万 token）。
        """
        from core import ai_budget
        ctx = self._ctx()
        quota = ai_budget.quota_for(ctx["persona"], "deep_analysis")
        assert quota >= 1
        runner = lambda code: {"advice": "看多"}            # noqa: E731
        for _ in range(quota):
            r = R.execute_tool("request_deep_analysis", {"code": "600028"},
                               analysis_runner=runner, **ctx)
            assert r["ok"] is True
        r = R.execute_tool("request_deep_analysis", {"code": "600028"},
                           analysis_runner=runner, **ctx)
        assert r["ok"] is False and "额度不足" in r["error"]

    def test_deep_analysis_failure_is_reported_honestly(self):
        """⭐ 执行器报 ok=False 时，工具**不得**转述成成功（否则模型会把"没拿到"
        当成"拿到了空报告"继续推理）。额度已在跑前扣掉，要如实告知已消耗。"""
        ctx = self._ctx()
        r = R.execute_tool("request_deep_analysis", {"code": "600028"},
                           analysis_runner=lambda c: {"ok": False, "error": "LLM 挂了"},
                           **ctx)
        assert r["ok"] is False and "LLM 挂了" in r["error"]
        assert r["credit_consumed"] is True

    def test_analysis_result_gets_a_bigger_shrink_budget(self):
        """⭐ 深度分析结果**不能**按普通工具的 400 字符截断 ——
        它的价值恰在报告正文与 errors/audit_warnings，压到 400 等于白跑一趟。"""
        assert R._ANALYSIS_RESULT_CHARS > 400
        long_result = {"ok": True, "result": {"errors": ["可核实性告警"],
                                              "final_decision": "看多" + "详" * 800}}
        kept = R._shrink(long_result, R._ANALYSIS_RESULT_CHARS)
        assert isinstance(kept, dict), "在预算内不该被截断成字符串"
        assert kept["result"]["errors"] == ["可核实性告警"]

    def test_deep_analysis_uses_injected_runner(self):
        ctx = self._ctx()
        r = R.execute_tool("request_deep_analysis", {"code": "600028"},
                           analysis_runner=lambda c: {"advice": "看多"}, **ctx)
        assert r["ok"] is True and r["result"] == {"advice": "看多"}

    def test_no_runner_is_honest_not_fake(self):
        """⚠️ 没接执行器时必须如实说"不可用"，**且不扣额度**（绝不假装跑过、
        也不让用户为一个没发生的动作付钱）。"""
        r = R.execute_tool("request_deep_analysis", {"code": "600028"}, **self._ctx())
        assert r["ok"] is False and "未接入" in r["error"] and "未消耗额度" in r["error"]


class TestPersonaRound:
    def test_tool_loop_then_final(self):
        client = FakeClient([
            '{"tool": "list_reports", "args": {}}',
            '{"final": {"actions": [{"action": "buy", "code": "600028", '
            '"position_size": 0.1, "reason": "均线多头"}], "summary": "看多"}}',
        ])
        r = R.run_persona_round(_persona(), R.build_common_briefing("2026-10-14"),
                                trade_date="2026-10-14", client=client)
        assert r["status"] == "ok" and r["decisions"][0]["code"] == "600028"
        assert len(r["transcript"]) == 1                 # 一轮工具 + 一轮 final
        # ⚠️ 模型由 ContextVar 钉住 + purpose 带 ai.<name>. 前缀（口径 5 归因）
        # 断言跟配置走（而不是写死模型名）：换供应商时这条测试不该改，
        # 但必须挡住"回退到 qwen"（额度已耗尽）。
        assert client.calls[0]["model_override"] == _persona()["model"]
        assert client.calls[0]["model_override"].startswith("deepseek")
        assert client.calls[0]["purpose"] == "ai.ai_value.decision"

    def test_quota_blocks_after_two_rounds_same_day(self):
        """⭐ 幂等/额度：配额 2 次/日后，**第 3 次**轮次被挡（不产生 LLM 调用）。

        ⚠️ 2026-10-23 配额 1→2（午间 13:05 + 盘后 15:58）。断言跟着
        `ai_budget.DEFAULT_QUOTA` 走，别再写死"第二次就被挡"。
        """
        from core.ai_budget import DEFAULT_QUOTA
        quota = int(DEFAULT_QUOTA["decision_round"])
        for _ in range(quota):
            R.run_persona_round(_persona(), R.build_common_briefing("2026-10-14"),
                                trade_date="2026-10-14", client=FakeClient([]))
        client_n = FakeClient([])
        r = R.run_persona_round(_persona(), R.build_common_briefing("2026-10-14"),
                                trade_date="2026-10-14", client=client_n)
        assert r["status"] == "skipped" and client_n.calls == []

    def test_llm_error_reported_not_silent(self):
        class Boom:
            def call_api(self, *a, **k):
                raise RuntimeError("网络断了")
        r = R.run_persona_round(_persona(), R.build_common_briefing("2026-10-14"),
                                trade_date="2026-10-14", client=Boom())
        assert r["status"] == "error" and "网络断了" in r["error"]

    def test_no_final_within_rounds(self):
        client = FakeClient(['{"tool": "get_my_budget", "args": {}}'] * 2)
        r = R.run_persona_round(_persona(), R.build_common_briefing("2026-10-14"),
                                trade_date="2026-10-14", client=client, max_rounds=2)
        assert r["status"] == "no_final" and len(r["transcript"]) == 2


class TestDailyRound:
    def test_random_group_never_calls_llm(self):
        """⭐ 随机对照组零 LLM：client_factory 只为 llm persona 被调用。"""
        made = []

        def factory(p):
            made.append(p["name"])
            return FakeClient([])

        out = R.run_daily_round("2026-10-14", client_factory=factory, dry_run=True)
        assert not any(n.startswith("ai_random") for n in made)
        assert set(made) == {"ai_value", "ai_tech", "ai_risk"}
        assert out["summary"]["total"] == 6

    def test_random_group_is_deterministic_per_day(self):
        """随机组种子由「名字|日期」派生 ⇒ 同一天重算完全一致（可复现）。"""
        import random as _r
        p = ai_personas.get_persona("ai_random_1")
        seed = f"{p['name']}|2026-10-14"
        a = ai_agent.random_decisions(p, candidates=["600028"], holdings=[],
                                      rng=_r.Random(seed))
        b = ai_agent.random_decisions(p, candidates=["600028"], holdings=[],
                                      rng=_r.Random(seed))
        assert a == b

    def test_second_run_same_day_is_blocked_by_quota(self):
        """⭐ 幂等（比"重复跑得到同样结果"更强）：同日第二次编排**整体被额度挡下**，
        既不产生第二次 LLM 调用、也不重复落库。"""
        from core.ai_budget import DEFAULT_QUOTA
        quota = int(DEFAULT_QUOTA["decision_round"])      # 2026-10-23 起是 2
        outs = [R.run_daily_round("2026-10-14", dry_run=True,
                                  client_factory=lambda p: FakeClient([]))
                for _ in range(quota)]
        assert all(o["summary"]["ok"] == 6 for o in outs), "配额内的每一轮都该正常跑"
        blocked = R.run_daily_round("2026-10-14", dry_run=True,
                                    client_factory=lambda p: FakeClient([]))
        assert blocked["summary"]["ok"] == 0 and blocked["summary"]["skipped"] == 6

    def test_manual_rerun_does_not_consume_daily_quota(self):
        """⭐ 人工补跑（`consume_budget=False`）：**不扣当日额度** ⇒
        补跑之后，当天 15:58 的正式盘后轮仍能正常跑（而不是被挤成 skipped）。

        ⚠️ 这正是"补跑会把正式轮挤掉"的回归守护：`decision_round` 只有 1 次/日，
        若哪天有人把这个参数删了，人工补跑会**静默地吃掉当天唯一的决策轮额度**，
        表现为"15:58 什么都没有跑"——不报错、只是没结果。
        """
        out1 = R.run_daily_round("2026-10-14", dry_run=True, consume_budget=False,
                                 client_factory=lambda p: FakeClient([]))
        assert out1["summary"]["ok"] == 6, "补跑本身必须真的跑起来"
        for name in ("ai_value", "ai_tech", "ai_risk"):
            assert ai_budget.view(name, "2026-10-14")["decision_round"]["used"] == 0, \
                f"{name} 的补跑不得占用额度"
        out2 = R.run_daily_round("2026-10-14", dry_run=True,
                                 client_factory=lambda p: FakeClient([]))
        assert out2["summary"]["ok"] == 6 and out2["summary"]["skipped"] == 0, \
            "补跑不得把随后的正式轮挤成 skipped"

    def test_persona_round_supports_manual_rerun(self):
        """`run_persona_round` 直调同样支持补跑语义（额度纹丝不动）。"""
        common = R.build_common_briefing("2026-10-14")
        p = _persona()
        r = R.run_persona_round(p, common, trade_date="2026-10-14",
                                client=FakeClient([]), consume_budget=False)
        assert r["status"] == "ok"
        assert ai_budget.view(p["name"], "2026-10-14")["decision_round"]["used"] == 0

    def test_dry_run_writes_nothing(self):
        R.run_daily_round("2026-10-14", dry_run=True,
                          client_factory=lambda p: FakeClient([]))
        assert store.list_ai_decisions(signal_date="2026-10-14") == []

    def test_persist_writes_decisions_and_wake_conditions(self):
        # ⚠️ 必须**先调一次工具**（2026-10-17 起：首轮直接 final 会被编排层退回，
        #    目的是治"工具循环在生产中从未被使用"）—— 这是新契约下的推荐形态。
        out = R.run_daily_round("2026-10-14",
                                client_factory=lambda p: FakeClient([
                                    '{"tool": "get_my_budget", "args": {}}',
                                    '{"final": {"actions": [{"action": "buy", "code": "600028", '
                                    '"position_size": 0.1, "reason": "x"}, {"action": "set_wake", '
                                    '"code": "600028", "kind": "stop_loss", "threshold": 5.0, '
                                    '"reason": "止损"}], "summary": "s"}}']))
        assert out["summary"]["ok"] == 6
        rows = store.list_ai_decisions(signal_date="2026-10-14")
        assert rows, "决策必须落库"
        conds = store.list_ai_wake_conditions(trade_date="2026-10-14")
        assert any(c["kind"] == "stop_loss" for c in conds), "唤醒条件必须落库"

    def test_failed_persona_not_persisted(self):
        """LLM 失败的 persona 不得产生任何决策行（失败不装成功）。"""
        def factory(p):
            if p["name"] == "ai_tech":
                class Boom:
                    def call_api(self, *a, **k):
                        raise RuntimeError("quota exceeded")
                return Boom()
            return FakeClient([])
        R.run_daily_round("2026-10-14", client_factory=factory)
        rows = store.list_ai_decisions(persona="ai_tech", signal_date="2026-10-14")
        assert rows == []

    def test_pass_only_round_still_leaves_trace(self):
        """只 pass 的一轮也要留痕（"今天为什么不动"必须可复盘）。"""
        R.run_daily_round("2026-10-14",
                          client_factory=lambda p: FakeClient([
                              '{"final": {"actions": [{"action": "pass", "reason": "空仓等待"}], '
                              '"summary": "无合适标的"}}']),
                          personas=[_persona()])
        rows = store.list_ai_decisions(persona="ai_value", signal_date="2026-10-14")
        assert len(rows) == 1 and rows[0]["advice_status"] == "等待"


# ================================================================
# 自述复盘：提示词要"问"、落库要"存"（P1，2026-10-14）
# ================================================================
#
# ⚠️ "验了函数 ≠ 验了接线"：只断言 `save_ai_review` 能写库是不够的，
#    必须验**决策轮的提示词确实索要了 review**、且**真的把它落了库**。

class TestSelfReviewWiring:
    def test_prompt_asks_for_review(self):
        p = R._round_prompt({"label": "测试", "style_prompt": "x"},
                            {"as_of": "2026-10-14"}, {"holdings": []}, [], 1, 3)
        assert '"review"' in p, "提示词没有索要 review（AI 不会给复盘）"
        # 必须告诉模型复盘要基于哪些私有字段，否则它会凭空编
        for k in ("recent_fills", "prev_decisions", "nav_recent"):
            assert k in p

    def test_review_is_persisted_by_round(self):
        # ⚠️ 同样要先取一次事实（见 TestToolFirstForcing 的说明）
        out = R.run_daily_round("2026-10-14",
                                client_factory=lambda p: FakeClient([
                                    '{"tool": "get_my_budget", "args": {}}',
                                    '{"final": {"review": "昨日按兵不动，正确。",'
                                    ' "actions": [{"action": "pass", "reason": "继续等"}],'
                                    ' "summary": "等待机会"}}']),
                                personas=[_persona()])
        assert out["summary"]["ok"] == 1
        row = store.get_ai_review("ai_value", "2026-10-14")
        assert row and row["review_text"] == "昨日按兵不动，正确。"
        assert row["summary"] == "等待机会"


# ================================================================
# 自述复盘：提示词要"问"、落库要"存"（P1，2026-10-14）
# ================================================================
#
# ⚠️ "验了函数 ≠ 验了接线"：只断言 `save_ai_review` 能写库是不够的，
#    必须验**决策轮的提示词确实索要了 review**、且**真的把它落了库**。

class TestSelfReviewWiring:
    def test_prompt_asks_for_review(self):
        p = R._round_prompt({"label": "测试", "style_prompt": "x"},
                            {"as_of": "2026-10-14"}, {"holdings": []}, [], 1, 3)
        assert '"review"' in p, "提示词没有索要 review（AI 不会给复盘）"
        # 必须告诉模型复盘要基于哪些私有字段，否则它会凭空编
        for k in ("recent_fills", "prev_decisions", "nav_recent"):
            assert k in p

    def test_review_is_persisted_by_round(self):
        # ⚠️ 同样要先取一次事实（见 TestToolFirstForcing 的说明）
        out = R.run_daily_round("2026-10-14",
                                client_factory=lambda p: FakeClient([
                                    '{"tool": "get_my_budget", "args": {}}',
                                    '{"final": {"review": "昨日按兵不动，正确。",'
                                    ' "actions": [{"action": "pass", "reason": "继续等"}],'
                                    ' "summary": "等待机会"}}']),
                                personas=[_persona()])
        assert out["summary"]["ok"] == 1
        row = store.get_ai_review("ai_value", "2026-10-14")
        assert row and row["review_text"] == "昨日按兵不动，正确。"
        assert row["summary"] == "等待机会"


# ================================================================
# 候选池去重与扩容 + 对照仓位（2026-10-15，主动性改进）
# ================================================================
#
# 背景：实测 `list_screen_picks(limit=20)` 里同一只票有 3 行
# ⇒ 候选池实际只覆盖 ~7 只**不同**标的 ⇒ AI "只能反复选那几只"。
# 修法：取足量原始行 → 按代码去重（保留最高分）→ 候选池上限提到 30。

class TestCandidatePool:
    def _seed(self, code, score, source="pipeline"):
        store.insert_screen_picks([{
            "pick_date": "2026-10-15", "stock_code": code, "stock_name": f"股{code}",
            "source": source, "strategy": "主力资金", "score": score}])

    def test_dedup_keeps_highest_score(self):
        rows = [{"stock_code": "600028", "score": 81.4, "id": 1},
                {"stock_code": "600028", "score": 88.2, "id": 2},
                {"stock_code": "600028", "score": 77.0, "id": 3},
                {"stock_code": "600519", "score": 60.0, "id": 4}]
        out = R._dedup_by_code(rows)
        assert [r["stock_code"] for r in out] == ["600028", "600519"]
        assert out[0]["id"] == 2, "同票多行应保留分数最高的那条"

    def test_briefing_candidates_have_no_duplicates(self):
        """⭐ 同一只票落 3 行时，简报里的候选池**不得**出现重复代码。"""
        for s in (81.4, 88.2, 77.0):
            self._seed("600028", s)
        for s in (72.0, 70.0):
            self._seed("601398", s)
        c = R.build_common_briefing("2026-10-15")
        assert c["candidates"] == sorted(set(c["candidates"]), key=c["candidates"].index)
        assert len(c["candidates"]) == len(set(c["candidates"]))
        assert set(c["candidates"]) == {"600028", "601398"}
        assert c["candidate_note"].startswith("候选池已按代码去重")
        assert [p["code"] for p in c["screen_picks"]] == ["600028", "601398"]

    def test_narrow_pool_is_declared_not_hidden(self):
        """池子窄时必须**如实说**（别让 AI 以为"市场就只有这些票"）。"""
        self._seed("600028", 80.0)
        c = R.build_common_briefing("2026-10-15")
        assert any("候选池偏窄" in w for w in c["warnings"]), c["warnings"]


class TestMarketContext:
    def test_briefing_carries_reference_exposure(self):
        """⭐ 公共简报必须带**对照仓位**（随机组当日平均仓位）—— 空仓要能对照解释。"""
        from core.paper_account import PaperAccount
        acct = PaperAccount(1_000_000, name="ai_random_1")
        rec = acct.mark_to_market("2026-10-15", {})
        rec["position_value"] = 100_000          # 10% 仓位
        rec["total_value"] = 1_000_000
        store.save_paper_state("ai_random_1", acct.to_state())
        c = R.build_common_briefing("2026-10-15")
        mc = c["market_context"]
        assert mc["available"] is True and mc["avg_exposure_pct"] == 10.0

    def test_prompt_points_at_reference_and_observation_condition(self):
        """接线守卫：提示词必须引用 market_context，并要求 pass 带观察条件。"""
        p = R._round_prompt({"label": "x", "style_prompt": "y"},
                            {"as_of": "2026-10-15"}, {"holdings": []}, [], 1, 3)
        assert "market_context" in p
        assert "观察条件" in p and "price_above" in p


class TestToolFirstForcing:
    """首轮直接出结论 ⇒ **退回一次**要求先取事实（2026-10-17 修）。

    实测背景：每个 persona 每轮**只调 1 次** LLM、第 1 轮就返回 `final`
    ⇒ 工具（含"接通了的深度分析"）**全程未被使用**。
    提示词里写"你必须调用工具"**管不住**模型 ⇒ 由编排层拦一次。

    ⚠️ 三条纪律：① **只退回一次**（成本上限 +1 次调用）；② **不许拦正常的"先工具后 final"**
    （否则每天白花一次调用）；③ 第 2 轮仍不调工具就**接受**它的答案（硬拦会变死循环）。
    """

    def test_final_without_tool_is_rejected_once(self):
        client = FakeClient([
            '{"final": {"actions": [{"action":"pass","reason":"不动"}], "summary":"首轮结论"}}',
            '{"tool": "get_my_budget", "args": {}}',
            '{"final": {"actions": [{"action":"pass","reason":"取完再说"}], "summary":"二轮结论"}}',
        ])
        r = R.run_persona_round(_persona(), R.build_common_briefing("2026-10-14"),
                                trade_date="2026-10-14", client=client)
        assert r["status"] == "ok"
        assert r["tool_calls"] == 1
        assert r["tool_retry"] is False               # 第 2 轮就调了工具
        assert len(client.calls) == 3
        assert r["transcript"][0].get("note") and "退回" in r["transcript"][0]["note"]
        assert r["summary"] == "二轮结论"              # 用的是取过事实后的结论

    def test_forced_round_prompt_forbids_final(self):
        """被强制的那一轮，提示词必须**明写**禁止直接 final（否则模型不知道要改）。"""
        seen = []

        class Cap(FakeClient):
            def call_api(self, messages, **kw):
                seen.append(messages[0]["content"])
                return super().call_api(messages, **kw)

        c = Cap(['{"final": {"actions": [], "summary": "s1"}}',
                 '{"final": {"actions": [], "summary": "s2"}}'])
        R.run_persona_round(_persona(), R.build_common_briefing("2026-10-14"),
                            trade_date="2026-10-14", client=c)
        assert "禁止直接给 final" in seen[1], seen[1][:200]

    def test_tool_first_flow_is_untouched(self):
        """⭐ 鉴别力：正常的"先工具后 final"**绝不能**被退回（否则每天白花一次调用）。"""
        client = FakeClient([
            '{"tool": "get_my_budget", "args": {}}',
            '{"final": {"actions": [], "summary": "ok"}}',
        ])
        r = R.run_persona_round(_persona(), R.build_common_briefing("2026-10-14"),
                                trade_date="2026-10-14", client=client)
        assert len(client.calls) == 2
        assert r["tool_calls"] == 1 and r["tool_retry"] is False

    def test_forcing_happens_at_most_once(self):
        """⭐ 成本上限：模型连着不调工具时只退回一次（总调用 = 2，不是 max_rounds 次）。"""
        client = FakeClient(['{"final": {"actions": [], "summary": "a"}}'] * 4)
        r = R.run_persona_round(_persona(), R.build_common_briefing("2026-10-14"),
                                trade_date="2026-10-14", client=client)
        assert len(client.calls) == 2, "只许退回一次（否则 token 会失控）"
        assert r["tool_calls"] == 0 and r["tool_retry"] is True


# ================================================================
# 补跑不得消耗正式额度（2026-10-23 事故）
# ================================================================

class TestManualRerunBudget:
    """⭐ 事故：中午用 `--force` 补跑一次决策轮 ⇒ 吃掉每个 AI 当日**唯一**一次
    `decision_round` 配额 ⇒ 15:58 正式轮 6 个全部"额度跳过"，当天无正式决策。
    修法：补跑走 `--no-budget`（`consume_budget=False`），产物 UPSERT 覆盖，
    正式轮跑完仍以正式轮为准。
    """

    def test_run_job_forwards_no_budget(self, monkeypatch):
        from core import scheduler as sch
        seen = {}

        def _fake(desc, consume_budget=True):
            seen["consume"] = consume_budget

        monkeypatch.setattr(sch, "run_ai_round_job", _fake)
        sch.run_job("ai_round", "测试补跑", trading_day_only=False,
                    force=True, no_budget=True)
        assert seen["consume"] is False

    def test_default_still_consumes(self, monkeypatch):
        """定时路径行为一字不变：不传 no_budget ⇒ 照常扣额度。"""
        from core import scheduler as sch
        seen = {}

        def _fake(desc, consume_budget=True):
            seen["consume"] = consume_budget

        monkeypatch.setattr(sch, "run_ai_round_job", _fake)
        sch.run_job("ai_round", "定时轮", trading_day_only=False, force=True)
        assert seen["consume"] is True


class TestMarketSignalNoRecompute:
    """⭐ 2026-10-23 自伤回归：把 `MarketDetector().detect()` 塞进简报构建，
    每个决策轮白等 **30.4 秒**，测试里被放大到整套跑不完（14 分钟未结束）。
    目标仓位改为读当日流水线产物（`daily_picks/latest.json`）。
    """

    def test_briefing_does_not_call_detector(self, monkeypatch):
        import core.ai_daily_round as R

        def _boom(*a, **k):
            raise AssertionError("build_common_briefing 不得现算市场检测（30s）")

        monkeypatch.setattr("core.market_detector.MarketDetector.detect", _boom)
        R.build_common_briefing("2026-10-14")          # 不抛异常即通过

    def test_signal_missing_is_reported_not_invented(self, monkeypatch, tmp_path):
        import core.ai_daily_round as R
        monkeypatch.setattr(R, "__file__", str(tmp_path / "core" / "x.py"))
        out = R._load_market_signal("2026-10-14")
        assert out["available"] is False and out["target_exposure"] is None
        assert out["note"]


class TestMarketSignalNoRecompute:
    """⭐ 2026-10-23 自伤回归：把 `MarketDetector().detect()` 塞进简报构建，
    每个决策轮白等 **30.4 秒**，测试里被放大到整套跑不完（14 分钟未结束）。
    目标仓位改为读当日流水线产物（`daily_picks/latest.json`）。
    """

    def test_briefing_does_not_call_detector(self, monkeypatch):
        import core.ai_daily_round as R

        def _boom(*a, **k):
            raise AssertionError("build_common_briefing 不得现算市场检测（30s）")

        monkeypatch.setattr("core.market_detector.MarketDetector.detect", _boom)
        R.build_common_briefing("2026-10-14")          # 不抛异常即通过

    def test_signal_missing_is_reported_not_invented(self, monkeypatch, tmp_path):
        import core.ai_daily_round as R
        monkeypatch.setattr(R, "__file__", str(tmp_path / "core" / "x.py"))
        out = R._load_market_signal("2026-10-14")
        assert out["available"] is False and out["target_exposure"] is None
        assert out["note"]


# ================================================================
# 情绪面票进简报（2026-10-24）
# ================================================================
# 动机：`news_stock_hits` 每天产出 100+ 只"被新闻点名"的票，但**只有盘后分析用了 2 席**，
# 6 个 AI 完全看不到。放进公共段后要守住三件事：① 缺数据必须说"没有"而不是空表；
# ② 只给 Top N（否则输入被撑爆）；③ 绝不能进私有段（那是公平性红线）。

class TestNewsHitsInBriefing:
    DAY = "2026-10-20"

    def _seed(self, n: int = 12, hits: int = None) -> None:
        rows = []
        for i in range(n):
            rows.append({
                "trade_date": self.DAY, "stock_code": f"60{i:04d}",
                "stock_name": f"票{i}", "hits": hits if hits is not None else 20 - i,
                "positive": 1, "negative": 0,
                "first_at": f"{self.DAY} 09:30", "last_at": f"{self.DAY} 14:30",
                "sources": "sina",
                "samples": json.dumps(["很长的标题" * 30], ensure_ascii=False),
            })
        store.upsert_news_stock_hits(rows)

    def test_absent_is_disclosed_not_invented(self):
        """没有命中 ⇒ `available=False` + note + warnings，**不回退昨天**。"""
        nh = R.build_common_briefing(self.DAY)["news_hits"]
        assert nh["available"] is False
        assert nh["rows"] == [] and nh["total"] == 0
        assert nh["note"], "缺数据必须给出原因，不能只留空"
        assert any("情绪面" in w for w in R.build_common_briefing(self.DAY)["warnings"])

    def test_rows_capped_and_headline_truncated(self):
        self._seed(n=12)
        nh = R.build_common_briefing(self.DAY)["news_hits"]
        assert nh["available"] is True
        assert len(nh["rows"]) == R._NEWS_HITS_SHOWN      # 只给前 8，不是全给
        assert nh["total"] == 12                          # 但如实披露总量
        for r in nh["rows"]:
            assert len(r["headline"]) <= R._NEWS_SAMPLE_CHARS + 1, r["headline"]

    def test_single_mention_is_filtered(self):
        """只被提 1 次的多半是行业词误命中 ⇒ 不进简报（阈值 `_NEWS_HITS_MIN`）。"""
        self._seed(n=3, hits=1)
        assert R.build_common_briefing(self.DAY)["news_hits"]["available"] is False

    def test_sorted_by_hits_desc(self):
        self._seed(n=5)
        hits = [r["hits"] for r in R.build_common_briefing(self.DAY)["news_hits"]["rows"]]
        assert hits == sorted(hits, reverse=True)

    def test_never_leaks_into_private_overlay(self):
        """⭐ 公平性：它是公共段事实，私有段出现即意味着"某些 AI 知道、另一些不知道"。"""
        self._seed(n=3)
        assert "news_hits" not in R._private_overlay("ai_value", self.DAY)

    def test_briefing_stays_deterministic_with_hits(self):
        """有数据时也要逐字节可重复（否则同一轮 6 个 AI 看到的不是同一份）。"""
        self._seed(n=6)
        a = json.dumps(R.build_common_briefing(self.DAY), ensure_ascii=False, sort_keys=True)
        b = json.dumps(R.build_common_briefing(self.DAY), ensure_ascii=False, sort_keys=True)
        assert a == b


# ================================================================
# 情绪面票进简报（2026-10-24）
# ================================================================
# 动机：`news_stock_hits` 每天产出 100+ 只"被新闻点名"的票，但**只有盘后分析用了 2 席**，
# 6 个 AI 完全看不到。放进公共段后要守住三件事：① 缺数据必须说"没有"而不是空表；
# ② 只给 Top N（否则输入被撑爆）；③ 绝不能进私有段（那是公平性红线）。

class TestNewsHitsInBriefing:
    DAY = "2026-10-20"

    def _seed(self, n: int = 12, hits: int = None) -> None:
        rows = []
        for i in range(n):
            rows.append({
                "trade_date": self.DAY, "stock_code": f"60{i:04d}",
                "stock_name": f"票{i}", "hits": hits if hits is not None else 20 - i,
                "positive": 1, "negative": 0,
                "first_at": f"{self.DAY} 09:30", "last_at": f"{self.DAY} 14:30",
                "sources": "sina",
                "samples": json.dumps(["很长的标题" * 30], ensure_ascii=False),
            })
        store.upsert_news_stock_hits(rows)

    def test_absent_is_disclosed_not_invented(self):
        """没有命中 ⇒ `available=False` + note + warnings，**不回退昨天**。"""
        nh = R.build_common_briefing(self.DAY)["news_hits"]
        assert nh["available"] is False
        assert nh["rows"] == [] and nh["total"] == 0
        assert nh["note"], "缺数据必须给出原因，不能只留空"
        assert any("情绪面" in w for w in R.build_common_briefing(self.DAY)["warnings"])

    def test_rows_capped_and_headline_truncated(self):
        self._seed(n=12)
        nh = R.build_common_briefing(self.DAY)["news_hits"]
        assert nh["available"] is True
        assert len(nh["rows"]) == R._NEWS_HITS_SHOWN      # 只给前 8，不是全给
        assert nh["total"] == 12                          # 但如实披露总量
        for r in nh["rows"]:
            assert len(r["headline"]) <= R._NEWS_SAMPLE_CHARS + 1, r["headline"]

    def test_single_mention_is_filtered(self):
        """只被提 1 次的多半是行业词误命中 ⇒ 不进简报（阈值 `_NEWS_HITS_MIN`）。"""
        self._seed(n=3, hits=1)
        assert R.build_common_briefing(self.DAY)["news_hits"]["available"] is False

    def test_sorted_by_hits_desc(self):
        self._seed(n=5)
        hits = [r["hits"] for r in R.build_common_briefing(self.DAY)["news_hits"]["rows"]]
        assert hits == sorted(hits, reverse=True)

    def test_never_leaks_into_private_overlay(self):
        """⭐ 公平性：它是公共段事实，私有段出现即意味着"某些 AI 知道、另一些不知道"。"""
        self._seed(n=3)
        assert "news_hits" not in R._private_overlay("ai_value", self.DAY)

    def test_briefing_stays_deterministic_with_hits(self):
        """有数据时也要逐字节可重复（否则同一轮 6 个 AI 看到的不是同一份）。"""
        self._seed(n=6)
        a = json.dumps(R.build_common_briefing(self.DAY), ensure_ascii=False, sort_keys=True)
        b = json.dumps(R.build_common_briefing(self.DAY), ensure_ascii=False, sort_keys=True)
        assert a == b


# ================================================================
# market_signal 双格式兼容（C1，2026-10-25）
# ================================================================
class TestMarketSignalDualFormat:
    """latest.json 的 market_signal 两种落盘格式都必须能吃。

    新格式 = dict（2026-10-25 起，`core/pipeline._json_default` 把 MarketSignal
    dataclass 落盘为 dict）；旧格式 = repr 字符串（此前的文件仍在磁盘上，
    联赛读历史 latest.json 不能炸）。
    """

    DAY = "2026-10-14"

    @pytest.fixture
    def _fake_latest(self, tmp_path, monkeypatch):
        import core.ai_daily_round as R
        monkeypatch.setattr(R, "__file__", str(tmp_path / "core" / "x.py"))
        d = tmp_path / "output" / "daily_picks"
        d.mkdir(parents=True)

        def _write(obj):
            (d / "latest.json").write_text(
                json.dumps(obj, ensure_ascii=False), encoding="utf-8")

        return _write

    def test_dict_format_direct_read(self, _fake_latest):
        import core.ai_daily_round as R
        _fake_latest({"date": self.DAY,
                      "market_signal": {"total_score": 28.4, "market_state": "偏弱震荡",
                                        "position_limit": 0.2}})
        out = R._load_market_signal(self.DAY)
        assert out["available"] is True
        assert out["score"] == 28.4
        assert out["state"] == "偏弱震荡"
        assert out["target_exposure"] == 0.2

    def test_legacy_repr_still_parses(self, _fake_latest):
        """旧文件（repr 串）不能因为改了落盘格式而失效。"""
        import core.ai_daily_round as R
        _fake_latest({"date": self.DAY,
                      "market_signal": "MarketSignal(total_score=28.4, "
                                       "market_state='偏弱震荡', signal='卖出/减仓', "
                                       "position_limit=0.2, strategy_bias='极端防御', "
                                       "operation_advice='x', scores={}, indicators={}, "
                                       "risk_warnings=[], timestamp='2026-10-25 15:30:00')"})
        out = R._load_market_signal(self.DAY)
        assert out["available"] is True
        assert out["score"] == 28.4
        assert out["state"] == "偏弱震荡"
        assert out["target_exposure"] == 0.2

    def test_dict_missing_fields_not_invented(self, _fake_latest):
        """缺就是缺：字段不全时标不可用，绝不拿默认值兜底。"""
        import core.ai_daily_round as R
        _fake_latest({"date": self.DAY, "market_signal": {"total_score": 28.4}})
        out = R._load_market_signal(self.DAY)
        assert out["available"] is False
        assert out["target_exposure"] is None
        assert out["note"]
