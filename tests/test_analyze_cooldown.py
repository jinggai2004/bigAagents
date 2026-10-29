# -*- coding: utf-8 -*-
"""软冷却选股（2026-10-21）。

**问题**：`candidates_for_date` 只 `ORDER BY score DESC LIMIT N`，而高分股的因子
是稳定的 ⇒ 同一批票天天上榜。实测重合度 10-20 = 40%、10-21 = **56%（5/9 重复）**。

**为什么是"软"冷却**：审查用真实数据算过 —— 7 个交易日的**硬**冷却会把候选打到
**1 只**（10-21 当日池 7 只、其中 6 只 7 日内已分析）。那离"每日分析 10-15 只"
更远。软冷却只改变**排序**，永不把候选清零。

本文件锁住三件事：
1. 软冷却**不减少**候选数量（与硬冷却的根本区别）；
2. 很久没看的票排到前面、新面孔拿满新鲜度；
3. 触发豁免是**加分项**，能让低分但已触发的票翻上来。
"""
import os
import sys
from datetime import datetime, timedelta

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import core.evening_analysis as E  # noqa: E402


def _cands(n=5, base_score=90):
    return [{"stock_code": f"60000{i}", "score": base_score - i * 5,
             "rank_in_run": i + 1, "stock_name": f"票{i}"} for i in range(n)]


# ================================================================
# 纯函数：排序语义
# ================================================================

class TestRankCandidates:

    def test_soft_cooldown_never_empties_the_pool(self):
        """⭐⭐ 核心断言：硬冷却会把 7 只打到 1 只，软冷却必须一只不少。

        这是"软"与"硬"的唯一区别，也是最容易被后来人改回去的地方。
        """
        cands = _cands(7)
        history = {f"60000{i}": 1 for i in range(6)}   # 6 只昨天刚分析过
        ranked = E.rank_candidates(cands, history, {})
        assert len(ranked) == 7, "软冷却不允许减少候选数量"

    def test_older_faces_rank_above_recent_ones(self):
        cands = [{"stock_code": "A", "score": 90}, {"stock_code": "B", "score": 90}]
        ranked = E.rank_candidates(cands, {"A": 1, "B": 6}, {})
        assert ranked[0]["stock_code"] == "B"

    def test_new_face_gets_full_freshness(self):
        ranked = E.rank_candidates([{"stock_code": "A", "score": 50}], {}, {})
        assert ranked[0]["_fresh"] is None          # 从未分析过
        assert ranked[0]["_priority"] == pytest.approx(0.6 * 0.5 + 0.4)

    def test_trigger_bonus_can_outrank_a_higher_score(self):
        """触发豁免必须是**加分**，否则"连续涨停但 3 天前看过"会被压下去。

        A：满分、今天刚看过 ⇒ 0.6×1.0 + 0.4×0 = 0.60
        B：70 分、今天刚看过、但已触及止损 ⇒ 0.6×0.7 + 0 + 0.5 = 0.92
        """
        cands = [{"stock_code": "A", "score": 100}, {"stock_code": "B", "score": 70}]
        ranked = E.rank_candidates(cands, {"A": 0, "B": 0}, {"B": "触及止损"})
        assert ranked[0]["stock_code"] == "B"
        assert ranked[0]["_trigger"] == "触及止损"

    def test_missing_score_is_treated_as_zero_not_crash(self):
        """`score` 可能是 None（用户直接指定代码时 `score_status=not_screened`）。"""
        ranked = E.rank_candidates([{"stock_code": "A", "score": None}], {}, {})
        assert ranked[0]["_priority"] == pytest.approx(0.4)

    def test_zero_cooldown_days_does_not_divide_by_zero(self):
        ranked = E.rank_candidates([{"stock_code": "A", "score": 80}],
                                   {"A": 3}, {}, cooldown_days=0)
        assert len(ranked) == 1

    def test_input_list_is_not_mutated(self):
        """返回新列表：调用方可能还要用原始的 `cands` 做别的事。"""
        cands = _cands(3)
        before = [dict(c) for c in cands]
        E.rank_candidates(cands, {"600000": 2}, {"600001": "触及目标"})
        assert cands == before
        assert "_priority" not in cands[0]

    def test_empty_code_is_dropped(self):
        ranked = E.rank_candidates([{"stock_code": "", "score": 90},
                                    {"stock_code": "A", "score": 80}], {}, {})
        assert [r["stock_code"] for r in ranked] == ["A"]


# ================================================================
# 查库：最近分析过哪些、哪些已触发
# ================================================================

class TestQueries:

    @pytest.fixture(autouse=True)
    def _db(self, tmp_path, monkeypatch):
        from tests.conftest import use_template_db
        use_template_db(tmp_path, monkeypatch, "cooldown.db")
        yield

    def _insert(self, code: str, days_ago: int):
        from db import store
        d = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        store.insert_analysis({"analysis_date": d, "stock_code": code,
                               "stock_name": code, "decision_label": "观望"})

    def test_recently_analyzed_returns_days_ago(self):
        self._insert("600001", 0)
        self._insert("600002", 3)
        out = E.recently_analyzed(days=7)
        assert out.get("600001") == 0
        assert out.get("600002") == 3

    def test_recently_analyzed_excludes_rows_outside_window(self):
        self._insert("600003", 30)
        out = E.recently_analyzed(days=7)
        assert "600003" not in out, "窗口外的旧记录不该参与新鲜度计算"

    def test_trigger_exempt_only_returns_hit_rows(self):
        from db import store
        self._insert("600010", 2)
        self._insert("600011", 2)
        # 手工把 600010 标成"触及目标"
        conn = store.get_connection()
        try:
            conn.execute("UPDATE analyses SET hit_target = 1 WHERE stock_code = ?", ("600010",))
            conn.commit()
        finally:
            conn.close()
        out = E.trigger_exempt_codes(days=7)
        assert out.get("600010") == "触及目标"
        assert "600011" not in out
