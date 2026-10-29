# -*- coding: utf-8 -*-
"""AI 交易联赛 S1 数据底座的测试（db/store.py 的 ai_* 表与读写函数）。

⚠️ 全部走 `use_template_db`（conftest 提供）——**绝不碰真实库**（项目最高危铁律）。
⚠️ 有一个用例专门在"**已有数据的库**"上验证迁移：这正是 init_store 哈希版本号
   假安全问题的场景（新表 IF NOT EXISTS 能建出来；旧表数据不能被动）。
"""
import sqlite3

import pytest

from tests.conftest import use_template_db
from db import store


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    use_template_db(tmp_path, monkeypatch, name="ai_store.db")


class TestTablesCreated:
    def test_five_ai_tables_exist(self):
        store.init_store()
        conn = sqlite3.connect(store.DB_PATH)
        try:
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            conn.close()
        for t in ("ai_personas", "ai_budget", "ai_wake_conditions",
                  "ai_wake_events", "ai_decisions"):
            assert t in names, f"缺表 {t}"

    def test_existing_tables_keep_data_after_reinit(self):
        """⚠️ 迁移场景：旧表已有数据，重放 schema 后数据必须原样（假安全问题的直接后果）。"""
        store.insert_news([{"hash": "h1", "title": "旧新闻", "source": "t"}])
        before = store.store_stats()["news_items"]["rows"]
        store.init_store(force=True)            # 无条件重放全部建表语句
        after = store.store_stats()["news_items"]["rows"]
        assert before == after == 1


class TestEnsureColumn:
    def test_init_store_backfills_new_column_on_existing_db(self):
        """⭐ 迁移守卫的真实用途：库已存在（缺 threshold_pct）时，init_store 必须补上。
        这正是 CREATE TABLE IF NOT EXISTS + 哈希版本号的**假安全**场景。"""
        conn = sqlite3.connect(store.DB_PATH)
        try:
            # 模拟"旧库"：把这一列真的删掉（SQLite 3.35+ 支持 DROP COLUMN）
            cols = {r[1] for r in conn.execute("PRAGMA table_info(ai_wake_conditions)")}
            if "threshold_pct" in cols:
                conn.execute("ALTER TABLE ai_wake_conditions DROP COLUMN threshold_pct")
                conn.commit()
            conn.execute("PRAGMA user_version = 0")      # 让 init_store 不短路
            conn.commit()
        finally:
            conn.close()
        store.init_store()
        conn = sqlite3.connect(store.DB_PATH)
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(ai_wake_conditions)")}
        finally:
            conn.close()
        assert "threshold_pct" in cols, "迁移守卫必须给已存在的表补列"

    def test_adds_missing_column(self):
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE t (a INTEGER)")
            store._ensure_column(conn, "t", "b", "TEXT")
            cols = {r[1] for r in conn.execute("PRAGMA table_info(t)")}
            assert {"a", "b"} <= cols
        finally:
            conn.close()

    def test_idempotent(self):
        """已存在时不重复加（重复 ALTER 会直接抛 duplicate column）。"""
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
            store._ensure_column(conn, "t", "b", "TEXT")   # 不应抛
            assert {r[1] for r in conn.execute("PRAGMA table_info(t)")} == {"a", "b"}
        finally:
            conn.close()


class TestPersonasCrud:
    def test_upsert_and_get(self):
        store.upsert_ai_persona({"name": "ai_value", "label": "价值派",
                                 "model": "qwen-max", "style_prompt": "sp",
                                 "risk_bias": "balanced"})
        p = store.get_ai_persona("ai_value")
        assert p and p["label"] == "价值派" and p["kind"] == "llm"
        assert p["frequency"] == "daily" and p["initial_cash"] == 1_000_000

    def test_upsert_keeps_initial_cash(self):
        """⚠️ initial_cash 只在首次建账生效：改初始资金不能追溯重置账本。"""
        store.upsert_ai_persona({"name": "x", "label": "X", "initial_cash": 1_000_000})
        store.upsert_ai_persona({"name": "x", "label": "X改名", "initial_cash": 2_000_000})
        p = store.get_ai_persona("x")
        assert p["label"] == "X改名"                    # 其他字段更新
        assert p["initial_cash"] == 1_000_000           # 初始资金不动

    def test_kind_random_has_no_model_even_if_given(self):
        """随机对照组给了 model 也必须被忽略（防止把 LLM 成本算到对照组头上）。"""
        # 该防线在 core/ai_personas._sanitize；DB 层原样存，这里验证 list 过滤
        store.upsert_ai_persona({"name": "r1", "label": "R", "kind": "random"})
        assert store.get_ai_persona("r1")["model"] is None

    def test_list_filters(self):
        store.upsert_ai_persona({"name": "a", "label": "A"})
        store.upsert_ai_persona({"name": "b", "label": "B", "kind": "random"})
        store.upsert_ai_persona({"name": "c", "label": "C", "enabled": False})
        names = {p["name"] for p in store.list_ai_personas()}
        assert names == {"a", "b"}                       # c 被停用
        assert {p["name"] for p in store.list_ai_personas(kind="random")} == {"b"}
        assert {p["name"] for p in store.list_ai_personas(enabled_only=False)} == {"a", "b", "c"}

    def test_delete_keeps_paper_ledger(self):
        """删除人格只删定义——账本（paper_*）是账户页的资产，不许连带带走。"""
        store.upsert_ai_persona({"name": "gone", "label": "G"})
        assert store.delete_ai_persona("gone") is True
        assert store.delete_ai_persona("gone") is False  # 幂等
        assert store.get_ai_persona("gone") is None


class TestBudget:
    def test_grant_then_deny(self):
        ok1, v1 = store.consume_ai_budget("p", "deep_analysis", trade_date="2026-10-14",
                                          quota=3)
        ok2, v2 = store.consume_ai_budget("p", "deep_analysis", trade_date="2026-10-14",
                                          quota=3)
        ok3, _ = store.consume_ai_budget("p", "deep_analysis", trade_date="2026-10-14",
                                         quota=3)
        ok4, v4 = store.consume_ai_budget("p", "deep_analysis", trade_date="2026-10-14",
                                          quota=3)
        assert (ok1, ok2, ok3, ok4) == (True, True, True, False)
        assert v4["used"] == 3 and v4["remaining"] == 0

    def test_kinds_are_independent(self):
        ok_a, _ = store.consume_ai_budget("p", "wake", trade_date="2026-10-14", quota=1)
        ok_b, _ = store.consume_ai_budget("p", "deep_analysis", trade_date="2026-10-14",
                                          quota=1)
        assert ok_a and ok_b

    def test_quota_change_applies_same_day_only(self):
        """配额快照语义：当天改配额当天生效；已过日期的行永不被碰。"""
        store.consume_ai_budget("p", "wake", trade_date="2026-10-13", quota=1)
        ok_old, _ = store.consume_ai_budget("p", "wake", trade_date="2026-10-13", quota=5)
        assert ok_old is False                          # 旧日期 used=1/quota=1，不受新配额影响
        ok_new, v = store.consume_ai_budget("p", "wake", trade_date="2026-10-14", quota=5)
        assert ok_new and v["quota"] == 5

    def test_view_shows_unused_kinds_zero_is_lie(self):
        """⚠️ 视图只包含**用过的** kind —— 「没出现」≠「额度 0」。
        调用方（ai_budget 模块）负责把未用的 kind 以配额默认值补全，这里验证原始行为。"""
        store.consume_ai_budget("p", "wake", trade_date="2026-10-14", quota=5)
        v = store.ai_budget_view("p", "2026-10-14")
        assert set(v) == {"wake"} and v["wake"] == {"used": 1, "quota": 5, "remaining": 4}

    def test_n_zero_never_grants(self):
        ok, _ = store.consume_ai_budget("p", "wake", n=0, trade_date="2026-10-14", quota=5)
        assert ok is False


class TestWakeEvents:
    def _cond(self, persona="p", cond_id=None, **kw):
        """建一条唤醒条件，返回**真实** cond_id。

        ⚠️ 幂等键是 `(persona, code, kind)`（2026-10-16 修订）⇒ 同键第二次调用是
        **UPDATE 而非新增**。需要多行时必须换 `code` 或 `kind`，不能靠 cond_id 区分。
        """
        payload = {"code": "600028", "kind": "stop_loss", "threshold": 5.0, **kw}
        if cond_id is not None:
            payload["cond_id"] = cond_id
        return store.upsert_ai_wake_condition(persona, payload)

    def test_upsert_is_idempotent_per_code_kind(self):
        """⭐ 同 `(persona, code, kind)` 重复 upsert 必须是 **UPDATE**，不得新增行。

        原实现按 `(persona, cond_id)` 冲突、而 cond_id 每次自增 ⇒ 永远 INSERT
        ⇒ 同一只票累积多个止损位（实测 600519 有 1184.01 与 1250.0 两条，
        600028 有 5.2 与 5.0 两条）⇒ **风控语义被破坏**：到底按哪个价位止损？
        """
        c1 = self._cond(threshold=5.0)
        c2 = self._cond(threshold=4.5)
        assert c1 == c2, "同键必须复用同一 cond_id（走 UPDATE）"
        rows = store.list_ai_wake_conditions(persona="p", armed_only=False)
        assert len(rows) == 1, f"不得累积重复条件：{rows}"
        assert rows[0]["threshold"] == 4.5, "阈值应更新为最新值"

    def test_different_kind_are_separate_rows(self):
        """不同 kind 是**不同**条件（同一只票可以同时有止损与止盈）。"""
        a = self._cond(kind="stop_loss")
        b = self._cond(kind="take_profit")
        assert a != b
        assert len(store.list_ai_wake_conditions(persona="p", armed_only=False)) == 2

    def test_record_is_idempotent(self):
        """口径 4 的核心：同 (persona, trade_date, code, cond_id) 只触发一次。
        进程重启后重复扫描同一天，绝不能重复触发。"""
        cid = self._cond()
        first = store.record_wake_event("p", "2026-10-14", "600028", cid,
                                        hit_at="10:00:00", trigger_px=4.95)
        again = store.record_wake_event("p", "2026-10-14", "600028", cid,
                                        hit_at="10:05:00", trigger_px=4.90)
        assert first is True and again is False

    def test_same_day_different_cond_is_new_event(self):
        """同一只票的**不同 kind** 是两个条件 ⇒ 各自触发一次事件。

        ⚠️ 这里靠 `kind` 区分而不是 cond_id：幂等键是 `(persona, code, kind)`，
        同键再写是 UPDATE（唯一索引兜底），拿 cond_id 造两条同行会直接撞约束。
        """
        c1 = self._cond(kind="stop_loss")
        c2 = self._cond(kind="take_profit")
        assert c1 != c2
        assert store.record_wake_event("p", "2026-10-14", "600028", c1, hit_at="10:00:00")
        assert store.record_wake_event("p", "2026-10-14", "600028", c2, hit_at="10:00:00")

    def test_expiry_deactivates_but_keeps_row(self):
        """过期条件失活但**不删行**——台账要留着复盘"AI 定的条件后来怎么样了"。"""
        self._cond(code="600028", expires_on="2026-10-13")
        self._cond(code="600519", expires_on="2026-10-20")
        self._cond(code="600000")                        # 无过期 = 永不过期
        n = store.deactivate_expired_wake_conditions("2026-10-14")
        assert n == 1
        armed = store.list_ai_wake_conditions(trade_date="2026-10-14")
        codes = {c["code"] for c in armed}
        assert "600028" not in codes and {"600519", "600000"} <= codes   # 过滤生效
        all_rows = store.list_ai_wake_conditions(armed_only=False)
        assert len([r for r in all_rows if r["code"] == "600028"]) == 1   # 行还在

    def test_mark_event_result(self):
        cid = self._cond()
        store.record_wake_event("p", "2026-10-14", "600028", cid, hit_at="10:00:00")
        assert store.mark_wake_event("p", "2026-10-14", "600028", cid,
                                     fired=False, skip_reason="超出唤醒配额")
        ev = store.list_wake_events(persona="p", trade_date="2026-10-14")[0]
        assert ev["fired"] == 0 and ev["skip_reason"] == "超出唤醒配额"


class TestDecisions:
    def _d(self, **kw):
        base = {"persona": "p", "signal_date": "2026-10-14", "code": "600028",
                "advice_status": "可执行", "direction": "看多",
                "position_size": 0.15, "model": "qwen-max"}
        return {**base, **kw}

    def test_save_and_list(self):
        store.save_ai_decision(self._d())
        rows = store.list_ai_decisions(persona="p", signal_date="2026-10-14")
        assert len(rows) == 1 and rows[0]["direction"] == "看多"

    def test_rerun_preserves_filled_order(self):
        """⚠️ 决策重跑不能抹掉已成交的订单信息（COALESCE 保留）。
        这是"重跑当天补跑分析"场景下的资金安全属性。"""
        store.save_ai_decision(self._d(order_id=42, order_status="已成交"))
        store.save_ai_decision(self._d(position_size=0.20))      # 重跑，未带订单信息
        row = store.list_ai_decisions(persona="p")[0]
        assert row["position_size"] == 0.20                      # 决策字段更新
        assert row["order_id"] == 42 and row["order_status"] == "已成交"   # 订单保留

    def test_rerun_can_fill_empty_order(self):
        store.save_ai_decision(self._d())
        store.save_ai_decision(self._d(order_id=7, order_status="已提交"))
        row = store.list_ai_decisions(persona="p")[0]
        assert row["order_id"] == 7
