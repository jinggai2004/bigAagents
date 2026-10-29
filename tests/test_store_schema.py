"""init_store 的幂等短路守护

背景：全仓 50+ 处调用 `init_store()`（每个写函数开头都调一次）。在新建库上
executescript 建 20+ 张表实测 1.6s 量级（Windows + 实时防护），因此加了
"schema 哈希指纹 → 存在 PRAGMA user_version → 命中就跳过"的短路。

这个优化的正确性完全依赖一件事：**schema 变了指纹必须跟着变**。
否则会出现"改了建表语句但谁也不生效"的静默故障——本文件就是守这条线。
"""

import sqlite3

import pytest

from db import store as db_store
import db.models as models


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    path = tmp_path / "store.db"
    monkeypatch.setattr(models, "DB_PATH", str(path))
    monkeypatch.setattr(db_store, "DB_PATH", str(path))
    return path


def _tables(path) -> set:
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def test_first_call_creates_all_tables(tmp_db):
    db_store.init_store()
    tables = _tables(tmp_db)
    assert {"news_items", "analyses", "screen_picks"} <= tables
    assert len(tables) >= 15


def test_schema_fingerprint_is_written(tmp_db):
    db_store.init_store()
    conn = sqlite3.connect(str(tmp_db))
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    assert version == db_store._SCHEMA_VERSION


def test_second_call_is_a_noop_but_still_idempotent(tmp_db):
    db_store.init_store()
    before = _tables(tmp_db)
    db_store.init_store()          # 走短路
    assert _tables(tmp_db) == before
    # 短路后表依然可用（不是"以为建过了其实没建"）
    assert db_store.insert_news([{
        "hash": "h1", "title": "t", "content": "c", "source": "s",
        "published_at": "2026-09-25 10:00:00", "related_codes": [],
        "sentiment": "neutral"}]) == 1


def test_changed_schema_invalidates_the_fingerprint(tmp_db, monkeypatch):
    """改了建表语句 → 指纹必须变 → 已存在的库也要把新表补建出来"""
    db_store.init_store()
    assert "guard_extra_table" not in _tables(tmp_db)

    # 模拟"开发者往 _SCHEMA 里加了一张表"
    monkeypatch.setattr(
        db_store, "_SCHEMA",
        db_store._SCHEMA + "\nCREATE TABLE IF NOT EXISTS guard_extra_table (id INTEGER);")
    monkeypatch.setattr(
        db_store, "_SCHEMA_VERSION",
        int(__import__("hashlib").sha1(
            db_store._SCHEMA.encode("utf-8")).hexdigest(), 16) % (2**31))

    db_store.init_store()
    assert "guard_extra_table" in _tables(tmp_db), \
        "schema 变了却没重建：说明指纹没跟着变，短路会把新表永久吞掉"


def test_force_rebuilds_after_manual_drop(tmp_db):
    """人工修复通道：表被误删时 force=True 必须能重建（正常短路不会）"""
    db_store.init_store()
    conn = sqlite3.connect(str(tmp_db))
    conn.execute("DROP TABLE news_items")
    conn.commit()
    conn.close()
    assert "news_items" not in _tables(tmp_db)

    db_store.init_store()                 # 短路，不会重建（这是设计的取舍）
    assert "news_items" not in _tables(tmp_db)

    db_store.init_store(force=True)       # 显式修复
    assert "news_items" in _tables(tmp_db)


# ⚠️ 已知平台现象（与本次短路改动无关，故不写成断言测试，避免 flaky）：
# Windows 上多线程/多进程**同时首次**建同一个新库时，部分连接会收到
# "attempt to write a readonly database"（4 线程实测 3 失败 1 成功）。
# 生产上不构成问题：库一旦存在就不会再走"首次建表"路径，而正常写入由
# db.models.get_connection 的 busy_timeout 排队。若将来要支持"多进程冷启动
# 抢着建库"，需要先解决这一层，而不是靠 init_store 重试掩盖。


# ================================================================
# F1（2026-10-18 审查）：**旧库升级** —— 缺列库必须能补列
# ================================================================
#
# 事故推演：`pe_source` 当天只加在 `_ensure_column`、没改 `_SCHEMA` ⇒ 哈希不变 ⇒
# 长期运行的旧库在"版本号相等"处早退，迁移永远不执行 ⇒ `insert_screen_picks`
# （INSERT 已带新列）在开盘首日静默丢掉**全部**选股样本（异常被自己的
# except 吞掉，只剩一条 warning）。本用例就是这条事故路径的回归锁。

def test_old_db_missing_column_gets_upgraded(tmp_db, monkeypatch):
    """⭐ 哈希相等但缺列的旧库 ⇒ init_store 后列出现、写入返回 1（不许静默 0 条）。"""
    # ① 构造"旧库"：建表 → 删列 → 把指纹改回当前值（模拟"哈希没变的长期运行库"）
    db_store.init_store()
    conn = sqlite3.connect(str(tmp_db))
    conn.execute("ALTER TABLE screen_picks DROP COLUMN pe_source")
    conn.execute(f"PRAGMA user_version = {db_store._SCHEMA_VERSION}")
    conn.commit()
    conn.close()

    def _cols():
        c = sqlite3.connect(str(tmp_db))
        try:
            return {r[1] for r in c.execute("PRAGMA table_info(screen_picks)")}
        finally:
            c.close()

    assert "pe_source" not in _cols()

    # ② 升级：列迁移在**版本短路之前**执行 ⇒ 缺列也能补上
    monkeypatch.setattr(db_store, "_COLUMNS_CHECKED", False)
    db_store.init_store()
    assert "pe_source" in _cols()

    # ③ 原事故路径：INSERT 带新列 ⇒ 必须成功落库（不许返回 0）
    n = db_store.insert_screen_picks([{
        "pick_date": "2026-10-19", "stock_code": "600028",
        "stock_name": "中国石化", "source": "ui", "strategy": "主力资金",
        "score": 65.7, "score_coverage": 0.8235, "price_at_pick": 5.53,
        "rank_in_run": 1, "pe_source": "snapshot_sina",
    }])
    assert n == 1, "缺列升级后写入必须成功（F1 事故推演：静默 0 条 ⇒ 丢样本）"
    rows = db_store.list_screen_picks(limit=1, stock_code="600028")
    assert rows[0]["pe_source"] == "snapshot_sina"


def test_new_column_is_in_schema_text(tmp_db):
    """⭐ 双保险之二：新列必须写进 `_SCHEMA` ⇒ 新建库天生带列 + 哈希随之变化。

    只加 `_ensure_column` 不改 `_SCHEMA` 的写法，正是 F1 事故的根因
    （哈希不变 ⇒ 旧库永远早退）。这条断言强迫"加列必须两处同步"。
    """
    assert "pe_source" in db_store._SCHEMA
    db_store.init_store()
    conn = sqlite3.connect(str(tmp_db))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(screen_picks)")}
    finally:
        conn.close()
    assert "pe_source" in cols
