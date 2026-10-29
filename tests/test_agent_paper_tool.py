"""AI 助手的两套账本：真实持仓 vs 模拟账户（2026-09-30）

## 用户报的问题

"问 AI 助手模拟账户建议时显示未有持仓（实际有）"。

## 根因

`core/agent_tools.py` 里**只有** `tool_portfolio_positions`（读 `positions` 表 =
用户在「持仓」页手工录入的**真实持仓**），**根本没有读模拟账户的工具**。
实测现场：`positions` 表 0 行，而 `paper_positions` 表有 3 只（002040/002129/600211）。
于是模型唯一的持仓工具返回"当前无持仓"，它只能如实照说——**不是模型编造，
是我们没给它看模拟账户的眼睛**（本项目反复出现的"功能已实现但从未接线"）。

系统提示里当时也**一个字都没提**模拟账户，模型连"还有另一本账"都不知道。

## 这个文件的守护点

1. 模拟账户有持仓时，工具必须真的看得见（而不是返回空）；
2. 真实持仓为空 ≠ 用户没有持仓：工具必须显式提醒"另一本账要用另一个工具"，
   否则模型会理直气壮地说"你没有持仓"；
3. 系统提示必须写明两套账本要分清（防止后续编辑把这条准则删掉）；
4. 取不到现价时退回收盘均价估值，**不用 0 抹掉持仓**（与 `PaperAccount.holdings_value` 同口径）。
"""

import pytest

from core.agent_tools import tool_paper_account, tool_portfolio_positions


@pytest.fixture
def store(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "paper_tool.db")
    from db import store as _store
    return _store


def _seed_paper(store, *, cash=453_186.05, positions=None, name="default"):
    """往模拟账户里放一份持仓（走真实 save_paper_state，不 mock）"""
    from core.paper_account import PaperAccount
    acct = PaperAccount(initial_cash=1_000_000.0, name=name)
    for p in (positions or []):
        acct.apply_trade({"date": "2026-09-25", "code": p["code"],
                          "direction": "买入", "price": p["price"],
                          "quantity": p["quantity"]})
    acct.cash = float(cash)
    store.save_paper_state(name, acct.to_state())
    return acct


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """行情一律用替身（只 mock 叶子：realtime 的取价函数）"""
    from data import realtime
    monkeypatch.setattr(realtime, "get_realtime_quotes",
                        lambda codes: {c: {"name": f"名称{c}", "price": 10.0,
                                           "quote_time": "2026-09-30 15:00:00"}
                                       for c in (codes or [])})
    yield


# ================================================================
# 1. 核心回归：模拟账户有持仓，工具必须看得见
# ================================================================

class TestPaperToolSeesPositions:
    def test_positions_are_visible(self, store):
        _seed_paper(store, positions=[
            {"code": "002040", "price": 9.08, "quantity": 20000},
            {"code": "600211", "price": 36.5, "quantity": 5000}])
        out = tool_paper_account()

        assert "error" not in out, out
        assert out["持仓只数"] == 2
        codes = {r["代码"] for r in out["持仓"]}
        assert codes == {"002040", "600211"}
        for r in out["持仓"]:
            assert r["数量"] > 0 and r["成本价"] > 0
            assert r["市值"] > 0

    def test_account_totals_are_consistent(self, store):
        _seed_paper(store, cash=1000.0,
                    positions=[{"code": "600211", "price": 10.0, "quantity": 1000}])
        out = tool_paper_account()
        # 现价 10.0 × 1000 = 10000（替身行情给的就是 10.0）
        assert out["持仓市值"] == pytest.approx(10000.0, rel=1e-6)
        assert out["总资产"] == pytest.approx(out["可用现金"] + out["持仓市值"], rel=1e-6)
        assert out["累计收益率%"] == pytest.approx(
            (out["总资产"] / out["初始资金"] - 1) * 100, abs=0.01)

    def test_t1_available_is_exposed(self, store):
        """可卖数量必须给出：T+1 下"有持仓但当天卖不掉"是模拟盘最容易误判的点"""
        _seed_paper(store, positions=[{"code": "002129", "price": 9.12, "quantity": 20000}])
        out = tool_paper_account()
        assert "可卖" in out["持仓"][0]

    def test_ledger_label_is_present_so_model_cannot_conflate(self, store):
        _seed_paper(store, positions=[{"code": "600211", "price": 10.0, "quantity": 100}])
        assert "模拟账户" in tool_paper_account()["账本"]


# ================================================================
# 2. 真实持仓为空 ≠ 用户没有持仓
# ================================================================

class TestEmptyRealPortfolioDoesNotClaimNoHoldings:
    def test_empty_real_positions_points_to_the_other_ledger(self, store):
        """真实持仓表空、模拟账户有 3 只时，真实持仓工具的 note 必须把模型引向另一个工具。

        这正是用户遇到的情形：模型只调了真实持仓工具，拿到空结果就回答"未有持仓"。
        """
        _seed_paper(store, positions=[{"code": "002040", "price": 9.08, "quantity": 20000}])
        out = tool_portfolio_positions()

        assert out["positions"] == []
        assert "tool_paper_account" in out["note"], \
            "空结果必须显式指向模拟账户工具，否则模型会直接回答'你没有持仓'"
        assert "真实持仓" in out["账本"]

    def test_empty_note_does_not_say_no_holdings(self, store):
        note = tool_portfolio_positions()["note"]
        # 旧文案是"当前无持仓。可在'持仓'页录入"——直接断言用户没有持仓
        assert "当前无持仓" not in note
        assert "不代表" in note, "必须写明：空结果只是这本账没录入，不是用户没有持仓"
        assert "未录入" in note


# ================================================================
# 3. 空账户 / 缺价
# ================================================================

class TestDegradation:
    def test_no_paper_state_is_explicit(self, store):
        out = tool_paper_account()
        assert out["持仓"] == []
        assert "note" in out and "尚未初始化" in out["note"]

    def test_empty_but_initialised_account_is_stated(self, store):
        _seed_paper(store, positions=[])
        out = tool_paper_account()
        assert out["持仓"] == []
        assert "无股票持仓" in out.get("note", "")

    def test_missing_price_falls_back_to_cost_not_zero(self, store, monkeypatch):
        """取不到现价时按成本价估值——用 0 会把持仓从总资产里抹掉"""
        from data import realtime
        monkeypatch.setattr(realtime, "get_realtime_quotes", lambda codes: {})
        _seed_paper(store, positions=[{"code": "600211", "price": 20.0, "quantity": 1000}])
        out = tool_paper_account()

        assert out["持仓"][0]["现价"] is None
        # 成本价含手续费，所以是 20.015 而不是 20.00 —— 断言与成本价一致即可
        row = out["持仓"][0]
        assert row["市值"] == pytest.approx(row["成本价"] * row["数量"], rel=1e-3)
        assert row["市值"] > 0, "不得因缺价把持仓市值算成 0"
        assert out["持仓市值"] > 0, "不得因缺价把持仓市值算成 0"
        assert "成本价" in out["note"]


# ================================================================
# 4. 系统提示与工具描述必须区分两套账本
# ================================================================

class TestPromptDistinguishesLedgers:
    def test_system_prompt_mentions_both_tools(self):
        from core.chat_agent import SYSTEM_PROMPT
        assert "tool_portfolio_positions" in SYSTEM_PROMPT
        assert "tool_paper_account" in SYSTEM_PROMPT, \
            "系统提示必须告诉模型存在模拟账户账本，否则它只会查真实持仓"

    def test_system_prompt_forbids_inferring_empty_from_wrong_ledger(self):
        from core.chat_agent import SYSTEM_PROMPT
        assert "没有持仓" in SYSTEM_PROMPT and "绝不能" in SYSTEM_PROMPT, \
            "必须写成硬约束：查错账本拿到空结果时不得回答'没有持仓'"

    def test_tool_schemas_describe_the_distinction(self):
        from core.agent_tools import TOOL_SCHEMAS
        desc = {s["function"]["name"]: s["function"]["description"]
                for s in TOOL_SCHEMAS}
        assert "模拟" in desc["tool_portfolio_positions"]
        assert "tool_portfolio_positions" in desc["tool_paper_account"]


# ================================================================
# 5. UI 进度摘要不能把两本账说成一本
# ================================================================

def test_brief_summary_labels_the_paper_ledger():
    from core.chat_agent import _brief_summary
    s = _brief_summary({"总资产": 993336.05, "持仓": [{"代码": "600211"}]})
    assert "模拟账户" in s

    s2 = _brief_summary({"positions": [{"代码": "600211"}, {"代码": "002040"}]})
    assert "真实持仓" in s2
