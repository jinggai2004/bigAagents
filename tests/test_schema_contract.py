# -*- coding: utf-8 -*-
"""建表契约：`_SCHEMA` 里**一张表都不能丢**（2026-10-23）。

## 为什么必须有这条测试
`db/store.py` 的建表段是**两条开发线共用的契约文件**。同一天里：
- 我在 `_SCHEMA` 里加了 `news_stock_hits`；
- 另一个会话加了 `push_log` / `signal_events` / `sentiment_snapshots` / `sentiment_labels`。

两边靠 `CREATE TABLE IF NOT EXISTS` 侥幸共存了，但这是**巧合而不是保障**：
只要有人整段覆盖（或改 `_SCHEMA` 时漏抄），那张表就**永远不会再被建出来** ——
因为 `init_store` 的版本号是 `_SCHEMA` 的哈希，哈希只会随内容变，
**没有"表缺失就补建"的逻辑**（项目铁律：schema 版本号是假安全）。

所以这里在**全新空库**上跑一次 `init_store()`，再逐张核对必需表。
任何一次覆盖式编辑导致丢表，这条测试立刻红 —— 比"上线后某功能莫名报 no such table"早得多。

⚠️ 新增表时请把它加进 `REQUIRED_TABLES`（这是**有意的**摩擦：
逼你确认"我这张表是不是该长期存在"）。
"""
import sqlite3

import pytest

from db import store
from tests.conftest import use_template_db

# 业务不可丢的表（按主题分组，新增表请追加到对应组）
REQUIRED_TABLES = {
    # ── 选股 / 行情 ──
    "screen_picks", "market_snapshot", "daily_snapshots", "daily_features",
    "fund_flow_snapshot", "industry_flow_snapshot", "data_coverage",
    # ── 分析 / 复盘 ──
    "analyses", "analysis_reviews", "news_items", "news_industry_reviews",
    # ── 模拟盘 ──
    "paper_accounts", "paper_positions", "paper_orders", "paper_fills", "paper_nav",
    "positions", "trade_logs", "alerts",
    # ── AI 联赛（2026-10-14 起）──
    "ai_personas", "ai_budget", "ai_decisions", "ai_reviews",
    "ai_wake_conditions", "ai_wake_events",
    # ── 情绪面选股（P1-5，2026-10-23）──
    "news_stock_hits",
    # ── 推送 / 信号（2026-10-23 另一条线）──
    "push_log", "push_templates", "signal_events",
    # ── 市场情绪参照（2026-10-23）──
    "sentiment_snapshots", "sentiment_labels",
    # ── 预测跟踪闭环 / 观点缓存（2026-10-23 审查指出漏登记）──
    "observations", "opinion_cache",
    # ── 基础设施 ──
    "llm_calls", "runs", "scheduled_jobs", "task_locks", "chat_logs", "chat_sessions",
    # ── AI 助手配置写审计（2026-10-24，P0 写守护的追责/回滚依据，只增不删）──
    "agent_config_changes",
    # ── 自选股双轨（2026-10-27）──
    # watchlist_items：用户与阿深各一套自选，**唯一存放处**（JSON 已退役）。
    # app_flags：一次性迁移标记（如 watchlist_migrated_v1）。⚠️ 它不能丢 ——
    #   丢了会导致"已迁移"状态消失 ⇒ 下次启动重新导入 → 用户删空的自选被灌回默认池。
    "watchlist_items", "app_flags",
    # ── 阿深定时值班（P1，2026-10-27）──
    # ai_routine_runs：每轮值班的**幂等锚点**（persona, trade_date, slot）。
    #   ⚠️ 它不能丢 —— 丢了等于"今天这些轮次都没跑过" ⇒ 同一个 slot 会被重跑，
    #   而盘中每分钟扫一次，重跑意味着同一只票可能被买 N 遍。
    #   它同时是成本对账的唯一来源（llm_called 计数）。
    "ai_routine_runs",
    # ── 来自 db/models.py 这第二套 DDL（不在 _SCHEMA 里，但同样必须存在）──
    "positions", "trade_logs", "daily_snapshots",
}

# 显式声明"不需要保护"的表：留空是**有意的**，新增表请优先登记进 REQUIRED_TABLES。
# 只有当某张表确实"丢了也无所谓"（如纯缓存）时才登记到这里，并写明理由。
EXEMPT_FROM_PROTECTION: set[str] = set()


class TestSchemaContract:
    @pytest.fixture(autouse=True)
    def _db(self, tmp_path, monkeypatch):
        use_template_db(tmp_path, monkeypatch, "schema_contract.db")

    def test_all_required_tables_exist(self):
        store.init_store()
        conn = store.get_connection()
        try:
            got = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'")}
        finally:
            conn.close()
        missing = sorted(REQUIRED_TABLES - got)
        assert not missing, (
            f"下列表在新建库里不存在：{missing}\n"
            f"⇒ 说明 `db/store.py::_SCHEMA` 被覆盖式编辑过（或新表忘了登记）。\n"
            f"⚠️ schema 版本号是哈希，**不会自动补建丢掉的表**。")

    def test_every_ddl_table_is_registered(self):
        """⭐ 反向断言（2026-10-23 审查建议）：**DDL 里出现的每张表都必须被登记**。

        为什么需要：`_SCHEMA`（`db/store.py`）与 `db/models.py` 是**两套 DDL**，
        而 `REQUIRED_TABLES` 是手工并集 ⇒ 必然漏（这次就漏了 `observations`、
        `opinion_cache` 两张真表）。有了这条，"新表忘登记"会红，而不是静默不保护。
        """
        import re
        from db import models, store

        def _tables(src: str) -> set:
            # 必须带左括号才算 DDL —— 否则注释里的
            # "全部是新表（CREATE TABLE IF NOT EXISTS 够用）"会被当成表名。
            return set(re.findall(
                r"CREATE TABLE IF NOT EXISTS\s+(\w+)\s*\(", src, re.I))

        with open(models.__file__, encoding="utf-8") as f:
            from_models = _tables(f.read())
        declared = _tables(store._SCHEMA) | from_models
        unregistered = sorted(declared - REQUIRED_TABLES - EXEMPT_FROM_PROTECTION)
        assert not unregistered, (
            f"这些表写进了 DDL 但没登记到 REQUIRED_TABLES：{unregistered}\n"
            f"⇒ 丢了也不会有测试发现。请加进 REQUIRED_TABLES（或显式 EXEMPT 并写理由）。")

    def test_news_stock_hits_survives_reinit(self):
        """重复 init 不得让表消失（惰性短路走的是"版本号相同就跳过"）。"""
        store.init_store()
        store.init_store()
        conn = store.get_connection()
        try:
            assert conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='news_stock_hits'").fetchone()
        finally:
            conn.close()
