"""模板库夹具的等价性守护

提速的前提是"复制模板"与"新建文件 + init_store"**语义严格等价**。
只要模板里多一张表、少一张表、或者残留一行数据，测试就会出现
"只在全量跑时才偶发红"的诡异现象——所以必须有守护。

守护三件事：
1. 表结构一致（不多不少）
2. 没有残留数据（sqlite_sequence 也要空，否则 AUTOINCREMENT 的 id 会从奇怪的值开始）
3. 用例拿到的是**自己的副本**：写数据不会污染模板，也不会污染别的用例
"""

import sqlite3

import pytest

from tests.conftest import template_db_path, use_template_db
import db.models as models
from db import store as db_store


def _tables(path) -> set:
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def _row_counts(path) -> dict:
    conn = sqlite3.connect(str(path))
    out = {}
    try:
        for t in sorted(_tables(path)):
            out[t] = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    finally:
        conn.close()
    return out


def _build_from_scratch(tmp_path, monkeypatch):
    """对照实现：老办法（新建文件 + init_db + init_store）"""
    path = tmp_path / "scratch.db"
    monkeypatch.setattr(models, "DB_PATH", str(path))
    monkeypatch.setattr(db_store, "DB_PATH", str(path))
    models.init_db()
    db_store.init_store()
    return path


def test_template_matches_scratch_built_database(tmp_path, monkeypatch):
    """模板的表集合必须与"老办法建出来的"完全一致"""
    from_scratch = _tables(_build_from_scratch(tmp_path, monkeypatch))
    from_template = _tables(template_db_path())
    assert from_template == from_scratch, (
        f"模板与新建库不一致：只在新库有 {from_scratch - from_template}，"
        f"只在模板有 {from_template - from_scratch}")


def test_template_has_no_residual_data():
    """模板必须是空库——含 sqlite_sequence（否则自增 id 起点会串味）"""
    counts = _row_counts(template_db_path())
    dirty = {t: n for t, n in counts.items() if n}
    assert dirty == {}, f"模板库里有残留数据：{dirty}"


def test_use_template_db_patches_both_paths(tmp_path, monkeypatch):
    """store 从 models 导入时绑定了同名 DB_PATH，只 patch 一侧会静默失效"""
    dst = use_template_db(tmp_path, monkeypatch, "x.db")
    assert models.DB_PATH == dst
    assert db_store.DB_PATH == dst
    assert "x.db" in dst


def test_writes_do_not_leak_into_template(tmp_path, monkeypatch):
    """用例往自己的库写数据，模板必须保持空白（否则会污染后续所有用例）"""
    use_template_db(tmp_path, monkeypatch, "y.db")
    assert db_store.insert_news([{
        "hash": "leak-check", "title": "t", "content": "c", "source": "s",
        "published_at": "2026-09-25 10:00:00", "related_codes": [],
        "sentiment": "neutral"}]) == 1

    tpl = _row_counts(template_db_path())
    assert tpl.get("news_items", 0) == 0, "写进模板了——用例之间会互相污染"


def test_each_call_gets_an_independent_copy(tmp_path, monkeypatch):
    """同一个用例里两次调用（不同文件名）应得到互不干扰的两个库"""
    a = use_template_db(tmp_path, monkeypatch, "a.db")
    db_store.insert_news([{
        "hash": "only-in-a", "title": "t", "content": "c", "source": "s",
        "published_at": "2026-09-25 10:00:00", "related_codes": [],
        "sentiment": "neutral"}])
    use_template_db(tmp_path, monkeypatch, "b.db")     # 切换库
    rows = db_store.get_recent_news(limit=10)
    assert all(r["hash"] != "only-in-a" for r in rows), "换库后仍能读到上一个库的数据"
