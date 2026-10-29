# -*- coding: utf-8 -*-
"""唤醒条件（core/ai_wake.py + watcher 接线）的测试。

⭐ 核心验收：
  ① 5 类条件判定正确；`stale` 与**过期条件**一律不判；
  ② **幂等**：同 (persona, 日期, code, cond_id) 只触发一次（重启重扫不重复花钱）；
  ③ **额度受限**：wake 配额用尽时只记录事件、不跑 LLM；
  ④ **阈值合理性**：模型给的无效价（如茅台写止损 5.0）必须被识别为无效**且不触发**；
  ⑤ **不增加行情请求**：判定只吃调用方给的 quotes（结构性守护）。
⚠️ 全离线：LLM 用假 client；库走 use_template_db。
"""
import json
from datetime import datetime

import pytest

from core import ai_personas, ai_wake
from db import store
from tests.conftest import use_template_db


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    use_template_db(tmp_path, monkeypatch, "ai_wake.db")
    monkeypatch.setattr(ai_personas, "PERSONAS_PATH", tmp_path / "ai_personas.json")
    ai_personas.load_personas()
    # ⚠️⚠️ 盘中补仓的开关/自动回退状态是**真实运行状态**（temp/ai_wake_buy_state.json）。
    #    不重定向的话，"自动回退"用例会把真实文件写成"已禁用" ⇒ **真实系统被测试关掉**，
    #    而且后续用例的判据也跟着变（本文件踩过一次同类的"测试污染真实状态"）。
    from core import ai_league
    monkeypatch.setattr(ai_league, "_wake_buy_state_path",
                        lambda: tmp_path / "wake_buy_state.json")
    yield
    ai_personas.invalidate()


DAY = "2026-10-14"
NOW = datetime(2026, 10, 14, 10, 30)

# 注入用的持仓/成本（避免读真实模拟账户）
_HOLD = lambda persona: {"600028"} if persona == "ai_value" else set()   # noqa: E731
_COST = lambda persona, code: 5.0                                        # noqa: E731


def _cond(persona="ai_value", code="600028", kind="stop_loss", **kw):
    cid = store.upsert_ai_wake_condition(persona, {"code": code, "kind": kind, **kw})
    return cid


def _q(price, stale=False, code="600028"):
    return {code: {"price": price, "stale": stale, "name": "中国石化"}}


class TestArmedCodes:
    def test_collects_only_enabled_personas(self):
        _cond(persona="ai_value")
        _cond(persona="ai_tech", code="600519")
        store.upsert_ai_persona({"name": "ai_tech", "label": "技术派", "enabled": False})
        # ai_personas 的 JSON 是权威（DB 只是快照）⇒ 用一份禁用配置覆盖
        personas = ai_personas.load_personas()
        for p in personas:
            if p["name"] == "ai_tech":
                p["enabled"] = False
        assert ai_personas.save_personas(personas) is True
        codes = ai_wake.armed_codes(DAY)
        assert "600028" in codes and "600519" not in codes

    def test_dedupes_codes(self):
        _cond(cond_id=None, code="600028", kind="stop_loss", threshold=4.0)
        _cond(cond_id=None, code="600028", kind="take_profit", threshold=6.0)
        assert ai_wake.armed_codes(DAY).count("600028") == 1


class TestEvaluate:
    def test_stop_loss_triggers_when_price_at_or_below_threshold(self):
        """止损触发口径：`现价 <= 止损价`。"""
        _cond(kind="stop_loss", threshold=4.9)
        hits = ai_wake.evaluate(_q(4.85), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST)
        assert len(hits) == 1 and hits[0]["kind"] == "stop_loss"
        assert hits[0]["trigger_px"] == 4.85

    def test_take_profit_triggers_above_threshold(self):
        _cond(kind="take_profit", threshold=5.5)
        hits = ai_wake.evaluate(_q(5.6), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST)
        assert len(hits) == 1 and hits[0]["kind"] == "take_profit"

    def test_price_above_works_without_position(self):
        """观察价条件非持仓也有效（止损/止盈才要求持仓）。"""
        _cond(persona="ai_tech", code="600519", kind="price_above", threshold=1500)
        hits = ai_wake.evaluate({"600519": {"price": 1550, "stale": False}},
                                trade_date=DAY, now=NOW,
                                holdings_of=lambda p: set(), cost_of=_COST)
        assert len(hits) == 1

    def test_stop_loss_ignored_without_position(self):
        """持仓类条件：没持仓不判（否则"空仓票跌到止损价"会误触发）。"""
        _cond(kind="stop_loss", threshold=4.9)
        hits = ai_wake.evaluate(_q(4.0), trade_date=DAY, now=NOW,
                                holdings_of=lambda p: set(), cost_of=_COST)
        assert hits == []

    def test_time_condition(self):
        _cond(kind="time", wake_at="10:00", code="600028")
        hits = ai_wake.evaluate(_q(5.0), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST)
        assert len(hits) == 1 and hits[0]["kind"] == "time"
        # 未到点不触发
        hits2 = ai_wake.evaluate(_q(5.0), trade_date=DAY,
                                 now=datetime(2026, 10, 14, 9, 0),
                                 holdings_of=_HOLD, cost_of=_COST)
        assert hits2 == []

    def test_pct_threshold_uses_cost(self):
        """随机对照组用百分比（threshold_pct）：按持仓成本换算。"""
        _cond(kind="stop_loss", threshold_pct=0.10)          # 成本 5.0 ⇒ 止损 4.5
        assert ai_wake.evaluate(_q(4.55), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST) == []
        hits = ai_wake.evaluate(_q(4.45), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST)
        assert len(hits) == 1 and hits[0]["threshold_used"] == pytest.approx(4.5)

    def test_stale_quote_never_judged(self):
        """⚠️ 页 8 已有纪律：过期价会误触发 ⇒ stale 一律不判。"""
        _cond(kind="stop_loss", threshold=4.9)
        assert ai_wake.evaluate(_q(4.0, stale=True), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST) == []

    def test_expired_condition_not_judged(self):
        _cond(kind="stop_loss", threshold=4.9, expires_on="2026-10-13")
        assert ai_wake.evaluate(_q(4.0), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST) == []

    # ---------- ⚠️ 首轮实跑实测的无效价（茅台写止损 5.0） ----------
    def test_absurd_stop_price_is_invalid_not_triggered(self):
        """止损价**远低于**现价 ⇒ 永不触发 ⇒ 视为无效，不得报"已触发"。
        实测来源：ai_tech 给贵州茅台（现价千元级）写 stop_loss threshold=5.0。"""
        _cond(persona="ai_tech", code="600519", kind="stop_loss", threshold=5.0)
        hits = ai_wake.evaluate({"600519": {"price": 1500, "stale": False}},
                                trade_date=DAY, now=NOW,
                                holdings_of=lambda p: {"600519"}, cost_of=_COST)
        assert hits == [], "永不触发的空壳条件不能算命中"

    def test_absurd_take_profit_is_invalid(self):
        """止盈价**远高于**现价 ⇒ 永不触发 ⇒ 无效。"""
        _cond(kind="take_profit", threshold=500.0)          # 现价 5.0
        hits = ai_wake.evaluate(_q(5.0), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST)
        assert hits == []

    def test_absurd_stop_price_actually_fires_when_already_breached(self):
        """⚠️ **反向不是无效**：止损位高于现价 = "现价已跌破计划止损位" ⇒ **立即触发**。
        （首版把这个方向判成"无效"，正好写反——测试抓到。）"""
        _cond(kind="stop_loss", threshold=4.9)
        hits = ai_wake.evaluate(_q(4.85), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST)
        assert len(hits) == 1 and hits[0]["kind"] == "stop_loss"

    def test_normal_stop_not_triggered_yet(self):
        """常规情形：现价在止损位之上 ⇒ 不触发。"""
        _cond(kind="stop_loss", threshold=4.5)
        assert ai_wake.evaluate(_q(4.85), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST) == []

    def test_missing_threshold_is_skipped(self):
        _cond(kind="stop_loss")                              # 既无 threshold 也无 pct
        assert ai_wake.evaluate(_q(4.0), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST) == []


class TestNoExtraQuoteRequest:
    def test_evaluate_takes_quotes_from_caller(self):
        """⭐ 结构性守护：判定函数**只接受 quotes 参数**，不得自行取数。
        （盘中请求量受 0.5 限频约束，多拉一遍会让盯盘自己超时。）"""
        import inspect
        sig = inspect.signature(ai_wake.evaluate)
        assert "quotes" in sig.parameters
        src = inspect.getsource(ai_wake.evaluate)
        assert "get_realtime_quotes" not in src, "判定不得自己取行情"

    def test_watcher_merges_codes_before_fetch(self):
        """⭐ 并入必须在**取数之前**（复用同一次请求），否则等于多拉一遍。"""
        import inspect
        from core import watcher
        src = inspect.getsource(watcher.AlertWatcher.scan_once)
        i_merge = src.index("ai_wake.armed_codes")
        i_fetch = src.index("get_realtime_quotes")
        assert i_merge < i_fetch, "codes 必须在取数前并入"


class TestRecordAndFire:
    class FakeClient:
        def __init__(self, reply=None):
            # ⚠️ 与提示词契约一致：外层必须是 {"final": {...}}
            #（首版测试用了裸 actions，而 parse_round_reply 只认 tool/final
            #  ⇒ 静默落到"未形成动作"的兜底 pass，测试才暴露出来）
            self.reply = reply or json.dumps({
                "final": {"actions": [{"action": "sell", "code": "600028",
                                       "reason": "止损"}],
                          "summary": "止损离场"}})
            self.calls = []

        def call_api(self, messages, **kw):
            self.calls.append(kw.get("purpose"))
            return self.reply

    def test_idempotent_same_day(self):
        """⭐ 重启后重复扫描同一天 ⇒ 只触发一次（不重复花钱）。"""
        cid = _cond(kind="stop_loss", threshold=4.9)
        hits = ai_wake.evaluate(_q(4.0), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST)
        c = self.FakeClient()
        r1 = ai_wake.record_and_fire(hits, trade_date=DAY, now=NOW,
                                     client_factory=lambda p: c)
        r2 = ai_wake.record_and_fire(hits, trade_date=DAY, now=NOW,
                                     client_factory=lambda p: c)
        assert r1["fired"] == 1 and r2["fired"] == 0
        assert r2["skipped_idempotent"] == 1
        assert len(c.calls) == 1, "第二次不得再调 LLM"

    def test_manual_check_does_not_spend_llm(self):
        """⚠️ 手动检查（UI 按钮）只记录不跑 LLM —— 否则点一下按钮就花钱。"""
        _cond(kind="stop_loss", threshold=4.9)
        hits = ai_wake.evaluate(_q(4.0), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST)
        c = self.FakeClient()
        r = ai_wake.record_and_fire(hits, trade_date=DAY, now=NOW,
                                    run_decision=False, client_factory=lambda p: c)
        assert c.calls == [] and r["fired"] == 0
        ev = store.list_wake_events(persona="ai_value", trade_date=DAY)[0]
        assert ev["fired"] == 0 and "手动检查" in (ev["skip_reason"] or "")

    def test_quota_exhausted_records_but_does_not_call(self):
        """⭐ 额度受限：达到 wake 配额上限后只记录事件、不再调 LLM。"""
        c = self.FakeClient()
        # 先把配额用满（默认 5 次）
        for _ in range(5):
            store.consume_ai_budget("ai_value", "wake", trade_date=DAY, quota=5)
        cid = _cond(kind="stop_loss", threshold=4.9)
        hits = ai_wake.evaluate(_q(4.0), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST)
        r = ai_wake.record_and_fire(hits, trade_date=DAY, now=NOW,
                                    client_factory=lambda p: c)
        assert c.calls == [] and r["fired"] == 0
        ev = store.list_wake_events(persona="ai_value", trade_date=DAY)[0]
        assert "额度不足" in (ev["skip_reason"] or "")

    def test_random_persona_does_not_call_llm(self):
        _cond(persona="ai_random_1", code="600887", kind="price_above", threshold=1.0)
        hits = ai_wake.evaluate({"600887": {"price": 2.0, "stale": False}},
                                trade_date=DAY, now=NOW,
                                holdings_of=lambda p: set(), cost_of=_COST)
        c = self.FakeClient()
        r = ai_wake.record_and_fire(hits, trade_date=DAY, now=NOW,
                                    client_factory=lambda p: c)
        assert c.calls == [] and r["fired"] == 0

    def test_wake_round_cannot_buy_on_non_dip_trigger(self):
        """⚠️ 双保险：**非下跌触发**时即使模型输出买入也必须被拦下。

        （2026-10-15 语义变更：买入不再一律禁止，但 `price_below` 之外的触发类型
          —— 如本用例的 `stop_loss` —— 仍然一律拒绝。放开后的完整边界见
          `TestWakeBuyGuards`。）
        """
        _cond(kind="stop_loss", threshold=4.9)
        hits = ai_wake.evaluate(_q(4.0), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST)
        client = self.FakeClient(json.dumps({
            "final": {"actions": [{"action": "buy", "code": "600028",
                                   "position_size": 0.3, "reason": "跌了想抄底"}],
                      "summary": "抄底"}}))
        r = ai_wake.record_and_fire(hits, trade_date=DAY, now=NOW,
                                    client_factory=lambda p: client)
        dec = r["results"][0]["result"]["decisions"]
        assert all(d["action"] != "buy" for d in dec), "非 price_below 触发不得买入"
        assert any("降级为仅唤醒" in (d.get("reason") or "") for d in dec), \
            "被拒必须留痕（降级为 pass），不能静默丢弃"

    def test_bare_actions_still_parsed(self):
        """容错：模型直接给裸 `{"actions": [...]}`（不带 final）也要能解析——
        宽松接受、严格解释，避免"格式微差 ⇒ 决策丢失"。"""
        _cond(kind="stop_loss", threshold=4.9)
        hits = ai_wake.evaluate(_q(4.0), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST)
        client = self.FakeClient(json.dumps({
            "actions": [{"action": "sell", "code": "600028", "reason": "止损"}],
            "summary": "裸格式"}))
        r = ai_wake.record_and_fire(hits, trade_date=DAY, now=NOW,
                                    client_factory=lambda p: client)
        dec = r["results"][0]["result"]["decisions"]
        assert dec and dec[0]["action"] == "sell"

    def test_sell_decision_is_recorded(self):
        _cond(kind="stop_loss", threshold=4.9)
        hits = ai_wake.evaluate(_q(4.0), trade_date=DAY, now=NOW,
                                holdings_of=_HOLD, cost_of=_COST)
        client = self.FakeClient()
        r = ai_wake.record_and_fire(hits, trade_date=DAY, now=NOW,
                                    client_factory=lambda p: client)
        assert r["fired"] == 1
        ev = store.list_wake_events(persona="ai_value", trade_date=DAY)[0]
        assert ev["fired"] == 1 and "sell" in (ev["action"] or "")


class TestDefaultHoldingsImpl:
    """⭐ 默认实现（**不注入** holdings_of/cost_of）必须能被真实账户状态驱动。

    背景（2026-10-14 实战抓到）：`positions` 是 **dict**（`{code: asdict(Position)}`），
    首版按 **list** 遍历 ⇒ 每次都得到空持仓 ⇒ **止损/止盈条件永不触发**，
    而上面那些注入式用例把它**完全遮住**了。
    ⇒ 注入是为了可测，但**默认实现必须单独测**（"注入绕过真实实现"陷阱）。
    """

    def _acct(self, name="ai_value"):
        from core.paper_account import PaperAccount
        acct = PaperAccount(1_000_000, name=name)
        acct.apply_trade({"date": "2026-10-13", "code": "600028",
                          "direction": "买入", "price": 5.0, "quantity": 1000})
        store.save_paper_state(name, acct.to_state())
        return acct

    def test_positions_state_is_dict(self):
        self._acct()
        st = store.load_paper_state("ai_value")
        assert isinstance(st["positions"], dict), \
            "结构若变了，ai_wake 的解析必须同步改（这里就是防它静默失效的锚点）"

    def test_holdings_of_reads_real_state(self):
        self._acct()
        assert ai_wake._holdings_of("ai_value") == {"600028"}

    def test_cost_of_derives_from_total_cost(self):
        """⚠️ `avg_cost` 是 property，`asdict()` 不带它 ⇒ 必须用 total_cost/quantity 反算。"""
        self._acct()
        assert ai_wake._cost_of("ai_value", "600028") == pytest.approx(5.0, rel=0.02)

    def test_stop_loss_fires_end_to_end_without_injection(self):
        """端到端（真实状态 + 真实默认实现）：成本 5.0、止损 -8%（4.6）⇒ 现价 4.5 应触发。"""
        self._acct()
        store.upsert_ai_wake_condition("ai_value", {
            "code": "600028", "kind": "stop_loss", "threshold_pct": 0.08})
        hits = ai_wake.evaluate({"600028": {"price": 4.5, "stale": False}},
                                trade_date=DAY, now=NOW)          # 不注入任何东西
        assert len(hits) == 1 and hits[0]["kind"] == "stop_loss"

    def test_unknown_account_yields_empty(self):
        assert ai_wake._holdings_of("nobody") == set()
        assert ai_wake._cost_of("nobody", "600028") is None


# ================================================================
# 盘中持仓内小额度补仓（2026-10-15 用户拍板"基础版"）
# ================================================================
# 三重边界：① 仅持仓内标的；② 仅 `price_below`（下跌触发）；③ 四档额度取孰低。
# ⚠️ 三条纪律必须钉死：
#   · **不通过就降级为"仅唤醒"**（不是静默丢弃，用户要看得见"AI 想做但被拦"）；
#   · **默认实现必须真的读账户**（注入只能构造场景，否则线上等于没护栏）；
#   · 自动回退时**连账户都不用读**（开关最先判）。

class TestWakeBuyGuards:
    class FakeBuyClient:
        def __init__(self, reply):
            self.reply = reply

        def call_api(self, messages, **kw):
            return self.reply

    def _buy_reply(self, code="600028", size=0.005):
        return json.dumps({"final": {
            "actions": [{"action": "buy", "code": code, "position_size": size,
                         "reason": "跌到计划位，按纪律小额度补仓"}],
            "summary": "补仓"}})

    def _acct(self, cash=1_000_000, qty=1000, price=5.0, name="ai_value"):
        from core.paper_account import PaperAccount
        acct = PaperAccount(cash, name=name)
        acct.apply_trade({"date": "2026-10-13", "code": "600028",
                          "direction": "买入", "price": price, "quantity": qty})
        store.save_paper_state(name, acct.to_state())
        return acct

    def _hit(self, kind="price_below", price=4.5, code="600028"):
        return {"persona": "ai_value", "code": code, "kind": kind, "price": price,
                "trigger_px": price, "reason": f"{kind} 触发"}

    def test_prompt_allows_buy_with_boundaries(self):
        """提示词必须**放开**买入并写清硬边界（否则 AI 永远只会 pass）。"""
        p = ai_wake._WAKE_PROMPT.format(label="价值派", trigger_desc="x", kind="price_below",
                                        price=4.5, holdings="[]", conds="[]", code="600028")
        assert '"action": "buy"' in p
        assert "price_below" in p
        # ⚠️ 额度数字**跟常量走**，不写死：2026-10-23 由 1%/2% 调到 3%/6%，
        #    写死的断言会把"改了限额"误报成"提示词坏了"。
        from core.ai_agent import WAKE_BUY_LIMITS as LIM
        assert f"{LIM['max_pct_of_position']:.0%}" in p
        assert f"{LIM['max_pct_of_equity']:.0%}" in p

    def test_default_guard_rejects_non_holding(self):
        """⭐ 真实默认实现：不在持仓 ⇒ 拒绝（盘中不开新仓）。"""
        self._acct()                                  # 只有 600028
        ok, why, amt = ai_wake._default_buy_guard(
            self._hit(code="600519", price=1200.0),
            {"name": "ai_value", "risk_bias": "balanced"}, DAY)
        assert ok is False and "已有持仓" in why and amt == 0.0

    def test_default_guard_rejects_above_trigger(self):
        """⭐ 上涨突破触发 ⇒ 只许卖出止盈，不许追涨。"""
        self._acct()
        ok, why, _ = ai_wake._default_buy_guard(
            self._hit(kind="price_above", price=5.5),
            {"name": "ai_value", "risk_bias": "balanced"}, DAY)
        assert ok is False and "price_below" in why

    def test_default_guard_allows_small_add_on_dip(self):
        """⭐ 真实默认实现放行：持仓内 + 下跌触发 ⇒ 额度取孰低并折整手。"""
        self._acct(qty=10_000)                        # 10000 股 @5.0 = 50,000 元市值
        ok, why, amt = ai_wake._default_buy_guard(
            self._hit(price=5.0), {"name": "ai_value", "risk_bias": "balanced"}, DAY)
        assert ok is True, why
        # 持仓市值 50,000×3% = 1,500；权益 1e6×1% = 10,000 ⇒ 取 1,500 ⇒ 300 股（整手）
        assert amt == pytest.approx(1500.0, rel=0.02), why

    def test_guard_rejects_when_amount_below_one_lot(self):
        """⭐ 额度折成整手不足 1 手 ⇒ **拒绝**（不成交比"买 3 股"更合理）。"""
        self._acct(qty=100)                           # 100 股 @5.0 = 500 元 ⇒ 3% = 15 元
        ok, why, amt = ai_wake._default_buy_guard(
            self._hit(price=5.0), {"name": "ai_value", "risk_bias": "balanced"}, DAY)
        assert ok is False and "1 手" in why and amt == 0.0

    def test_guard_rejection_degrades_to_pass_not_silent_drop(self):
        """⭐ 被拒 ⇒ 产出**一条 pass 并写明原因**（"仅唤醒、不成交，盘后再决策"）。"""
        persona = {"name": "ai_value", "label": "价值派", "kind": "llm",
                   "risk_bias": "balanced", "model": "qwen-max"}
        r = ai_wake.run_wake_round(
            persona, self._hit(code="600519", price=1200.0), trade_date=DAY, now=NOW,
            client=self.FakeBuyClient(self._buy_reply(code="600519")),
            buy_guard=lambda *a, **k: (False, "非持仓", 0.0))
        acts = [d["action"] for d in r["decisions"]]
        assert "buy" not in acts and "pass" in acts
        assert "降级为仅唤醒" in r["decisions"][0]["reason"]

    def test_guard_pass_marks_intraday_and_keeps_shares(self):
        """⭐ 通过 ⇒ 决策带 `intraday/amount/shares`（结算靠 shares 执行，不再重算护栏）。"""
        persona = {"name": "ai_value", "label": "价值派", "kind": "llm",
                   "risk_bias": "balanced", "model": "qwen-max"}
        r = ai_wake.run_wake_round(
            persona, self._hit(price=5.0), trade_date=DAY, now=NOW,
            client=self.FakeBuyClient(self._buy_reply()),
            buy_guard=lambda *a, **k: (True, "允许补仓 30 股", 150.0))
        d = r["decisions"][0]
        assert d["action"] == "buy" and d["intraday"] is True
        assert d["shares"] == 30 and d["amount"] == pytest.approx(150.0)

    def test_usage_counted_from_events(self):
        """用量必须从 `ai_wake_events.action` 数 —— 单一事实来源。"""
        store.record_wake_event("ai_value", DAY, "600028", 1, hit_at="10:30:00")
        store.mark_wake_event("ai_value", DAY, "600028", 1, fired=True,
                              action=json.dumps([{"action": "buy", "code": "600028",
                                                  "amount": 150.0}]))
        per_code, total, spent = ai_wake._wake_buy_usage("ai_value", DAY)
        assert per_code == {"600028": 1} and total == 1 and spent == pytest.approx(150.0)
        # 卖出不算补仓用量
        store.record_wake_event("ai_value", DAY, "600028", 2, hit_at="10:31:00")
        store.mark_wake_event("ai_value", DAY, "600028", 2, fired=True,
                              action=json.dumps([{"action": "sell", "code": "600028"}]))
        per_code, total, _ = ai_wake._wake_buy_usage("ai_value", DAY)
        assert total == 1, "卖出事件不得计入盘中补仓次数"

    def test_disabled_switch_blocks_before_reading_account(self):
        """⭐ 自动回退生效时**连账户都不用读**（开关最先判，理由最直接）。"""
        from core import ai_league
        monkeypatch_ok = ai_league.write_wake_buy_state(manual_enable=False)
        assert monkeypatch_ok
        # 造"最近 5 个补仓日全亏"的账本
        from core.paper_account import PaperAccount
        acct = PaperAccount(1_000_000, name="ai_value")
        for i, d in enumerate(["2026-10-06", "2026-10-07", "2026-10-08",
                               "2026-10-09", "2026-10-10"], start=1):
            acct.fills.append({"date": d, "code": "600028", "direction": "买入",
                               "quantity": 100, "actual_price": 5.0, "total": 500.0,
                               "cash_after": 0, "tag": "intraday_buy"})
            acct.fills.append({"date": d, "code": "600028", "direction": "卖出",
                               "quantity": 100, "actual_price": 4.8, "total": 480.0,
                               "cash_after": 0, "pnl": -20.0})
        store.save_paper_state("ai_value", acct.to_state())
        st = ai_league.wake_buy_status(DAY)
        assert st["enabled"] is False and st["auto_disabled"] is True
        assert "自动回退" in st["reason"] and len(st["evidence"]["days"]) == 5
        ok, why, amt = ai_wake._default_buy_guard(
            self._hit(), {"name": "ai_value", "risk_bias": "balanced"}, DAY)
        assert ok is False and "自动回退" in why
        ai_league.write_wake_buy_state(manual_enable=True, reason="测试人工复位")
        assert ai_league.wake_buy_status(DAY)["enabled"] is True


class TestValidateWakeBuy:
    @staticmethod
    def _validate(**kw):
        from core.ai_agent import validate_tool_call
        return validate_tool_call(**kw)

    def test_wake_buy_rejected_without_semantic_check(self):
        ok, why = self._validate(
            name="place_order",
            args={"code": "600028", "direction": "买入",
                  "position_size": 0.005, "reason": "x"}, context="wake")
        assert ok is False and "补仓护栏" in why

    def test_wake_buy_allowed_after_semantic_check(self):
        ok, why = self._validate(
            name="place_order",
            args={"code": "600028", "direction": "买入",
                  "position_size": 0.005, "reason": "x"},
            context="wake", wake_buy_ok=True)
        assert ok is True, why
