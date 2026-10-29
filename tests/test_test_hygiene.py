# -*- coding: utf-8 -*-
"""测试自身的卫生守护：不许有测试文件绕过 DB 隔离。

背景（2026-10-16 实证）：`tests/test_watcher.py` **没有任何 DB 夹具**，于是
`watcher.check()` 一路走到真实库 —— 读真实库的唤醒条件 → 命中 →
`record_and_fire(run_decision=True)` ⇒ **真调 LLM（真花钱）** + 把 `ai_wake_events`
写进真实库。一次全量回归实测多出 1 次 `ai.ai_tech.wake`（821 tokens）+ 2 条事件。

⚠️ 为什么必须用"结构检查"兜底：`conftest` 的生产库哨兵只判**结构级**破坏
（丢表 / `user_version`），对**行级**污染不报警 —— 否则交易时段调度进程的正常写入
会天天误报。行级污染只能靠"每个写库文件都得有夹具"这条**结构**约束来兜。

⚠️ 这条断言是**结构性事实**（文件里有没有夹具），换机器、换时段结论都一样，
不会像耗时断言那样 flaky。
"""
from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# store 的"写"家族（只读函数不在此列：读真实库至多慢，不会污染）
_WRITE_CALL = re.compile(
    r"\bstore\.(?:insert|save|upsert|record|mark|consume|delete|update)\w*\s*\(")
# 两种被认可的隔离方式
_ISOLATION_HINTS = ("use_template_db", "DB_PATH")


def _test_files() -> list[Path]:
    return sorted(p for p in TESTS_DIR.glob("test_*.py"))


def test_every_store_writer_has_db_isolation():
    """⭐ 调了 `store.*` 写函数、却没有 DB 隔离的测试文件一律失败。"""
    offenders = []
    for path in _test_files():
        src = path.read_text(encoding="utf-8")
        if not _WRITE_CALL.search(src):
            continue
        if any(h in src for h in _ISOLATION_HINTS):
            continue
        offenders.append(path.name)
    assert not offenders, (
        f"这些测试文件调用了 store 的写函数却没有 DB 隔离（会污染真实库）：{offenders}\n"
        "修法：加 `use_template_db(tmp_path, monkeypatch, '<name>.db')` 的 autouse 夹具，"
        "或显式 monkeypatch `DB_PATH`（并在文件头写清为什么）。")


def test_guard_itself_can_detect_a_violation(tmp_path):
    """⚠️ 鉴别力自检：这条守护**必须真的能抓到违规**。

    否则它就是"恒真断言"——看着在守，实际什么也没守（项目里踩过的坑）。
    用一个临时造出来的违规文件验证检测逻辑本身有效。
    """
    bad = tmp_path / "test_violation.py"
    bad.write_text("from db import store\n\n"
                   "def test_x():\n    store.save_ai_decision({'a': 1})\n",
                   encoding="utf-8")
    src = bad.read_text(encoding="utf-8")
    assert _WRITE_CALL.search(src), "正则必须能识别 store 写调用"
    assert not any(h in src for h in _ISOLATION_HINTS), "违规样本不应含隔离线索"
