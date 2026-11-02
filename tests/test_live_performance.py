# -*- coding: utf-8 -*-
"""`core/live_performance.py`（实盘绩效月度序列）的行为契约。

这个模块的失败模式是"把 3 条样本算出的超额当成结论"，
所以用例围绕：**样本不足要明说、薄月份要剔除、覆盖率要能算出来**。
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from core import live_performance as lp

SCHEMA = """
CREATE TABLE screen_picks (
  id INTEGER PRIMARY KEY, pick_date TEXT, stock_code TEXT, stock_name TEXT,
  source TEXT, strategy TEXT, score REAL, score_coverage REAL,
  price_at_pick REAL, rank_in_run INTEGER,
  ret_5d REAL, ret_20d REAL, bench_ret_5d REAL, bench_ret_20d REAL,
  backfill_status TEXT
)
"""


def _db(tmp_path, rows: list[tuple]) -> sqlite3.Connection:
    """rows: (pick_date, code, rank, ret_20d, bench_20d, ret_5d, bench_5d)"""
    c = sqlite3.connect(tmp_path / "t.db")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    c.executemany(
        "INSERT INTO screen_picks (pick_date, stock_code, rank_in_run, ret_20d,"
        " bench_ret_20d, ret_5d, bench_ret_5d) VALUES (?,?,?,?,?,?,?)", rows)
    c.commit()
    return c


def test_missing_columns_reports_reason(tmp_path):
    c = sqlite3.connect(tmp_path / "x.db")
    c.executescript("CREATE TABLE screen_picks (id INTEGER, pick_date TEXT)")
    out = lp.monthly_pick_excess(20, conn=c)
    assert out["ok"] is False and "缺列" in out["reason"]
    c.close()


def test_no_backfilled_rows_is_not_ok(tmp_path):
    """一条都没回填 ⇒ ok=False + 覆盖率，**不能返回空序列冒充 0 超额**。"""
    c = _db(tmp_path, [("2026-10-09", "600000", 1, None, None, None, None)])
    out = lp.monthly_pick_excess(20, conn=c)
    assert out["ok"] is False
    assert "还没有任何" in out["reason"]
    assert out["coverage"]["filled"] == 0 and out["coverage"]["total"] == 1
    c.close()


def test_monthly_mean_and_coverage(tmp_path):
    rows = [("2026-09-30", f"60000{i}", i, 0.05, 0.01, None, None) for i in range(4)]
    rows += [("2026-10-30", f"60001{i}", i, 0.02, 0.01, None, None) for i in range(4)]
    c = _db(tmp_path, rows)
    out = lp.monthly_pick_excess(20, conn=c)
    assert out["ok"] is True and out["months"] == 2
    assert out["ret"].iloc[0] == pytest.approx(0.05)
    assert out["coverage"]["pct"] == 1.0
    c.close()


def test_thin_month_is_dropped(tmp_path):
    """单月只有 2 条 ⇒ 该月剔除（否则 2 条样本就能决定一个月的好坏）。"""
    rows = [("2026-09-30", "600000", 1, 0.05, 0.01, None, None),
            ("2026-09-30", "600001", 2, 0.05, 0.01, None, None)]
    rows += [("2026-10-30", f"60001{i}", i, 0.02, 0.01, None, None) for i in range(4)]
    c = _db(tmp_path, rows)
    out = lp.monthly_pick_excess(20, conn=c)
    assert out["months"] == 1 and out["dropped_thin_months"] == 1
    assert len(out["ret"]) == 1
    c.close()


def test_rank_max_filters(tmp_path):
    rows = [("2026-09-30", "600000", 1, 0.10, 0.01, None, None),
            ("2026-09-30", "600001", 2, 0.10, 0.01, None, None),
            ("2026-09-30", "600002", 3, 0.10, 0.01, None, None),
            ("2026-09-30", "600099", 99, -0.50, 0.01, None, None)]
    c = _db(tmp_path, rows)
    allrows = lp.monthly_pick_excess(20, conn=c)
    top = lp.monthly_pick_excess(20, conn=c, rank_max=5)
    assert allrows["ret"].iloc[0] == pytest.approx((0.1 * 3 - 0.5) / 4)
    assert top["ret"].iloc[0] == pytest.approx(0.10)      # 第 99 名被剔除
    c.close()


def test_horizon_switch_uses_right_columns(tmp_path):
    rows = [("2026-09-30", f"60000{i}", i, None, None, 0.03, 0.01) for i in range(4)]
    c = _db(tmp_path, rows)
    assert lp.monthly_pick_excess(20, conn=c)["ok"] is False     # 20 日没回填
    out5 = lp.monthly_pick_excess(5, conn=c)
    assert out5["ok"] is True and out5["ret"].iloc[0] == pytest.approx(0.03)
    c.close()


def test_coverage_note_mentions_both_numbers(tmp_path):
    c = _db(tmp_path, [("2026-09-30", "600000", 1, None, None, None, None)])
    note = lp.coverage_note(lp.monthly_pick_excess(20, conn=c), 20)
    assert "0/1" in note and "T+20" in note
    c.close()
