# -*- coding: utf-8 -*-
"""盘后批量深度分析（core/evening_analysis）与 analyses 唯一性。

背景：自动深度分析写了但从未跑过（`mode="llm"` 没有调度任务），
`analyses.source` 只有 ui/rerun。这批测试锁住"它现在真的会跑，且不会重复算样本"。
"""
import pytest

from tests.conftest import use_template_db
from core import evening_analysis as EA
from db import store


@pytest.fixture
def env(tmp_path, monkeypatch):
    use_template_db(tmp_path, monkeypatch)
    monkeypatch.delenv(EA.ENV_LIMIT, raising=False)
    monkeypatch.delenv(EA.ENV_TOKEN_CAP, raising=False)
    return tmp_path


def _seed_picks(date="2026-10-19", rows=None):
    rows = rows or [
        ("600028", 88.0, 1), ("000001", 82.0, 2), ("600519", 75.0, 3),
        ("002415", 70.0, 4), ("601318", 65.0, 5),
    ]
    store.insert_screen_picks([
        {"pick_date": date, "stock_code": c, "stock_name": c, "source": "ui",
         "strategy": "主力资金", "score": s, "score_coverage": 0.8,
         "price_at_pick": 10.0, "rank_in_run": r, "pe_source": "eastmoney"}
        for c, s, r in rows])


class TestCandidates:
    def test_dedupes_multiple_screen_runs(self, env):
        """⭐ 一天三次选股会重复落库（同日 20 行 / 8 只票），候选必须按代码去重。"""
        _seed_picks()
        _seed_picks()                     # 模拟第二次选股：同票再落一遍
        cands = EA.candidates_for_date("2026-10-19", want=20)
        codes = [c["stock_code"] for c in cands]
        assert len(codes) == len(set(codes)) == 5
        assert codes[0] == "600028"       # 按分数降序

    def test_empty_when_no_picks(self, env):
        assert EA.candidates_for_date("2026-10-19") == []


class TestRun:
    def test_analyzes_top_n_and_records(self, env):
        """⭐ 成功路径：按分数取前 N，逐只调用 runner，并写 runs 记录。"""
        _seed_picks()
        calls = []

        def fake(code, source="pipeline"):
            calls.append((code, source))
            return {"ok": True}

        stats = EA.run_evening_analysis(date="2026-10-19", limit=3,
                                        runner=fake)
        assert [c[0] for c in calls] == ["600028", "000001", "600519"]
        assert stats["analyzed"] == ["600028", "000001", "600519"]
        assert stats["skipped_reason"] is None
        conn = store.get_connection()
        try:
            n = conn.execute("SELECT COUNT(*) FROM runs WHERE run_type=?",
                             ("job:evening_analysis",)).fetchone()[0]
        finally:
            conn.close()
        assert n >= 1, "必须写 runs 记录（否则无法判断今天分析了几只）"

    def test_skips_already_analyzed_same_day(self, env):
        """⭐ 幂等：同日已分析过的票跳过（唯一索引 + 这里双保险）。"""
        _seed_picks()
        store.insert_analysis({
            "analysis_date": "2026-10-19", "stock_code": "600028",
            "stock_name": "中国石化", "decision_label": "看多",
            "payload": {}, "trade_plan": {}})

        def fake(code, source="pipeline"):
            return {"ok": True}

        stats = EA.run_evening_analysis(date="2026-10-19", limit=5, runner=fake)
        assert "600028" in stats["skipped_existing"]
        assert "600028" not in stats["analyzed"]
        assert len(stats["analyzed"]) == 4

    def test_token_cap_stops(self, env, monkeypatch):
        """⭐ 预算闸门：当日 token 已达上限 ⇒ 一只都不跑（不能靠"跑完再看账单"）。"""
        _seed_picks()
        monkeypatch.setenv(EA.ENV_TOKEN_CAP, "1000")
        conn = store.get_connection()
        try:
            conn.execute("INSERT INTO llm_calls (created_at, model, purpose, ok, "
                         "prompt_tokens, completion_tokens) VALUES (?,?,?,?,?,?)",
                         ("2026-10-19 12:00:00", "deepseek-v4-flash", "config.test", 1,
                          800, 400))
            conn.commit()
        finally:
            conn.close()
        called = []
        stats = EA.run_evening_analysis(date="2026-10-19", limit=5,
                                        runner=lambda *a, **k: called.append(1))
        assert called == []
        assert "token" in (stats["skipped_reason"] or "")

    def test_empty_verdict_is_counted_as_failure(self, env):
        """⭐ 空壳报告必须记失败（2026-10-22 实测）。

        LLM 中途连接失败时，链路后段（研究汇总/交易员/风控）会空转，产出
        `final_decision=""` + `trade_plan={}`，但 `run_analysis` 的 `ok` 仍是 True
        —— 它只判"有没有抛异常"。真实事故：600900 长江电力 `errors` 里有 4 次
        `Connection error.`，整批却报"成功 8 / 失败 0"。

        判据：有 `snapshot` 但 `final_decision` 与 `trade_plan.direction` 都空 ⇒ 失败。
        """
        _seed_picks()

        def fake(code, source="pipeline"):
            return {"ok": True, "snapshot": {"stock_code": code,
                                             "final_decision": "",
                                             "trade_plan": {}}}

        stats = EA.run_evening_analysis(date="2026-10-19", limit=2, runner=fake)
        assert stats["analyzed"] == [], "空壳报告不该算分析成功"
        assert len(stats["failed"]) == 2
        assert "未产出结论" in stats["failed"][0]["error"]

    def test_runner_without_snapshot_is_not_penalized(self, env):
        """非标准 runner（不返回 snapshot）不能被误判成失败 —— 结构不同 ≠ 分析失败。"""
        _seed_picks()
        stats = EA.run_evening_analysis(date="2026-10-19", limit=1,
                                        runner=lambda *a, **k: {"ok": True})
        assert len(stats["analyzed"]) == 1
        assert stats["failed"] == []

    def test_failure_does_not_abort_batch(self, env):
        """单只失败不能拖垮整批（这是"批量"相对"手点"的关键差异）。"""
        _seed_picks()

        def fake(code, source="pipeline"):
            if code == "000001":
                raise RuntimeError("模型超时")
            return {"ok": True}

        stats = EA.run_evening_analysis(date="2026-10-19", limit=5, runner=fake)
        assert stats["analyzed"] == ["600028", "600519", "002415", "601318"]
        assert stats["failed"][0]["code"] == "000001"

    def test_dry_run_calls_nothing(self, env):
        _seed_picks()
        called = []
        stats = EA.run_evening_analysis(date="2026-10-19", limit=2, dry_run=True,
                                        runner=lambda *a, **k: called.append(1))
        assert called == []
        assert stats["analyzed"] == ["600028", "000001"]     # 只是"将会分析"

    def test_no_candidates_reports_reason(self, env):
        stats = EA.run_evening_analysis(date="2026-10-19", runner=lambda *a, **k: None)
        assert "无选股候选" in stats["skipped_reason"]


class TestAnalysisUniqueness:
    def test_same_day_same_code_upserts_not_duplicates(self, env):
        """⭐ 同日同票只留一行（否则一只票被算成多个样本，胜率被放大）。"""
        for label in ("看多", "观望"):
            store.insert_analysis({
                "analysis_date": "2026-10-19", "stock_code": "603127",
                "stock_name": "昭衍新药", "decision_label": label,
                "payload": {"v": label}, "trade_plan": {}})
        conn = store.get_connection()
        try:
            n = conn.execute("SELECT COUNT(*) FROM analyses WHERE analysis_date='2026-10-19' "
                             "AND stock_code='603127'").fetchone()[0]
            label = conn.execute("SELECT decision_label FROM analyses WHERE "
                                 "analysis_date='2026-10-19' AND stock_code='603127'").fetchone()[0]
        finally:
            conn.close()
        assert n == 1, "唯一索引未生效"
        assert label == "观望", "UPSERT 必须保留最新一次分析"
