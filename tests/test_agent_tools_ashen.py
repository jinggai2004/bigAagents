# -*- coding: utf-8 -*-
"""助手 → 阿深：**第三套账本**必须真的接上（2026-10-28）

## 用户报的现象

"ai助手好像没接阿深"。

## 机械核查结论（不是印象）

`core/chat_agent.py` 与 `core/agent_tools.py` 里"阿深"出现 **0 次**；29 个助手工具
没有一个碰 `ai_ashen`。实测两问两错，而且**都不报错**：

· 问"阿深今天跑了几轮值班" → 助手去查 `runs` 表，答成"流水线跑了 3 轮选股"
  （把"阿深值班"误解成"选股 pipeline"）；
· 问"阿深模拟账户净值" → 助手调 `tool_paper_account`，读的是用户自己的 `default` 盘。
  两者初始资金都是 100 万、当天都空仓 ⇒ **错误的数据源给出了正确的数字**，
  用户完全看不出问题。这比报错危险得多。

这是本项目第三次同类问题（前两次：模拟账户没接、3D 入口没接）。本文件守的点：

1. 工具读的是 `ai_ashen` 而不是 `default` —— **两者金额必须不同**才验得出；
2. 兜底的 `{"action": "pass"}` 不算"动作"（否则每轮都显示"动作数 1"，
   模型会说"它今天做了 6 个动作"，真实情况是"6 轮都明确选择不动"）；
3. 轮次按时间正序（`history()` 是 `slot DESC`，直接透传会让"今天"倒着读）；
4. 没有账户时 `存在=False` 且不抛异常（"查不到" ≠ "0"）；
5. 工具必须**同时**在 `TOOL_SCHEMAS` 与 `TOOL_FUNCS` 里（只加一个 ⇒ 模型调不到）；
6. 系统提示必须点名阿深与该工具（防止后续编辑把这条准则删掉）。
"""
from __future__ import annotations

import pytest

from core.agent_tools import (
    TOOL_FUNCS,
    TOOL_SCHEMAS,
    tool_ashen_status,
)

DAY = "2026-10-28"


@pytest.fixture()
def tpl_db(tmp_path, monkeypatch):
    """全部用例走模板库副本，不碰生产库。"""
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "agent_ashen.db")
    from db import store
    return store


@pytest.fixture(autouse=True)
def _persona(monkeypatch):
    """不耦合真实 `config/ai_personas.json`：给一份最小人设。

    ⚠️ 真实配置里阿深可能是 L2、也可能被临时禁用，断言"级别=L2"会随配置漂移。
    """
    from core import ai_personas
    p = {"name": "阿深", "label": "阿深", "enabled": True, "autonomy": "L2",
         "frequency": "intraday", "league_eligible": False, "model": "qwen-test"}
    monkeypatch.setattr(ai_personas, "get_persona",
                        lambda name: p if name == "ai_ashen" else None)
    return p


def _seed_account(store, name: str, cash: float):
    from core.paper_account import PaperAccount
    acct = PaperAccount(initial_cash=1_000_000.0, name=name)
    acct.cash = float(cash)
    store.save_paper_state(name, acct.to_state())
    return acct


def _set_created(store, slot: str, ts: str):
    conn = store.get_connection()
    with conn:
        conn.execute("UPDATE ai_routine_runs SET created_at = ? WHERE persona = 'ai_ashen'"
                     " AND trade_date = ? AND slot = ?", (ts, DAY, slot))
    conn.close()


class TestAccountIsTheRightLedger:
    def test_reads_ashen_account_not_users_paper_account(self, tpl_db):
        """工具必须读 ai_ashen。金额故意不同 —— 一样的话这条断言验不出错账本。"""
        _seed_account(tpl_db, "default", 111_111.0)      # 用户的模拟盘
        _seed_account(tpl_db, "ai_ashen", 888_888.0)     # 阿深的
        r = tool_ashen_status(DAY)
        assert r["账户"]["现金"] == 888_888.0
        assert r["账户"]["存在"] is True

    def test_missing_account_is_not_zero(self, tpl_db):
        """没开户 ⇒ `存在=False` + 一句人话，不是 `现金=0`（那会被读成"钱亏光了"）。"""
        r = tool_ashen_status(DAY)
        assert r["账户"]["存在"] is False
        assert r["账户"]["现金"] is None
        assert "自动创建" in r["账户"]["note"]

    def test_ledger_note_warns_against_three_way_confusion(self, tpl_db):
        """返回里必须写明"这是第三套账本"，否则模型仍会拿它和用户盘混着说。"""
        r = tool_ashen_status(DAY)
        assert "ai_ashen" in r["账本说明"]
        assert "tool_paper_account" in r["账本说明"]


class TestRoutineRuns:
    def test_pass_action_is_not_counted_as_an_action(self, tpl_db):
        """`actions` 里永远有一条兜底 pass —— 数它会让"没动手"显示成"动了 1 次"。"""
        tpl_db.save_routine_run(
            "ai_ashen", DAY, "09:40", status="ok", llm_called=True,
            summary="等回调", actions=[{"action": "pass", "code": None,
                                        "reason": "三只都没到买点"}])
        r = tool_ashen_status(DAY)
        row = r["值班"]["轮次"][0]
        assert row["动作数"] == 0, row
        assert row["动作"] == []
        assert "明确选择不动" in r["值班"]["口径"]

    def test_real_action_is_counted(self, tpl_db):
        tpl_db.save_routine_run(
            "ai_ashen", DAY, "09:40", status="ok", llm_called=True,
            actions=[{"action": "buy", "code": "600887", "reason": "回踩到位"}])
        row = tool_ashen_status(DAY)["值班"]["轮次"][0]
        assert row["动作数"] == 1
        assert row["动作"][0]["代码"] == "600887"

    def test_runs_are_in_chronological_order(self, tpl_db):
        """`history()` 按 slot DESC 返回，直接透传会让"今天"从 14:40 读到 09:40。"""
        for slot, ts in (("14:40", "2026-10-28 14:40:05"),
                         ("09:40", "2026-10-28 09:40:05")):
            tpl_db.save_routine_run("ai_ashen", DAY, slot, status="ok", llm_called=True)
            _set_created(tpl_db, slot, ts)
        slots = [x["班次"] for x in tool_ashen_status(DAY)["值班"]["轮次"]]
        assert slots == ["09:40", "14:40"], slots

    def test_skipped_round_is_not_counted_as_ran(self, tpl_db):
        """「已跑」只数真调了模型的轮；跳过轮算进去会让"6/6 到岗"变成假象。"""
        tpl_db.save_routine_run("ai_ashen", DAY, "09:40", status="ok", llm_called=True)
        tpl_db.save_routine_run("ai_ashen", DAY, "10:20", status="skipped",
                               llm_called=False, skip_reason="无变化")
        v = tool_ashen_status(DAY)["值班"]
        assert v["已跑"] == 1
        assert v["跳过"] == 1


class TestWiring:
    def test_tool_is_registered_in_both_tables(self):
        """只加 schema 或只加 func 都会出问题：前者模型看得见调不到，后者永远调不到。"""
        names = [t["function"]["name"] for t in TOOL_SCHEMAS]
        assert "tool_ashen_status" in names
        assert "tool_ashen_status" in TOOL_FUNCS
        assert len(names) == len(TOOL_FUNCS), "schema 与实现数量必须一一对应"

    def test_schema_description_mentions_third_ledger(self):
        spec = next(t for t in TOOL_SCHEMAS
                    if t["function"]["name"] == "tool_ashen_status")["function"]
        assert "阿深" in spec["description"]
        # 必须点明"不可与另外两套账本混用"，否则模型仍会拿 tool_paper_account 顶替
        assert "tool_paper_account" in spec["description"]

    def test_system_prompt_teaches_ashen(self):
        """写了工具但没告诉模型 = 零覆盖（本项目铁律）。"""
        from core.chat_agent import SYSTEM_PROMPT
        assert "阿深" in SYSTEM_PROMPT
        assert "tool_ashen_status" in SYSTEM_PROMPT
        # "值班"必须被映射到轮次，而不是选股流水线
        assert "值班" in SYSTEM_PROMPT

    def test_result_limit_allows_full_runs(self):
        """1500 字符会拦腰截断 6 轮明细，模型只看到表头就说"它今天没动作"。"""
        from core.chat_agent import _TOOL_RESULT_CHAR_LIMITS
        assert _TOOL_RESULT_CHAR_LIMITS.get("tool_ashen_status", 0) >= 4000
