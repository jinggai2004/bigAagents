# -*- coding: utf-8 -*-
"""深度分析无头执行器（`core/analysis_runner.py`）的测试。

⭐ 核心验收：
  ① `run_analysis` 保留 **merge reducer 语义**（errors 只追加 / data_availability 合并）；
  ② 失败时**落"分析失败"记录**并如实返回 `ok=False`；
  ③ `summarize_for_ai` **必须带 errors/audit_warnings/data_missing，且排在最前**
     （结果回灌会被**按字符截断**，放最后等于没有）；
  ④ `ai_deep_analysis_runner`：**同日同票复用**（不再烧 ≈3.5 万 token）；未命中才真跑；
  ⑤ **结构性守卫**：UI 页面不得再自写一遍 stream 循环（否则必然与执行器分叉）。

⚠️ 全离线：图被替换为假图；库走 use_template_db。
"""
import os

import pytest

from core import analysis_runner as AR
from db import store
from tests.conftest import use_template_db


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    use_template_db(tmp_path, monkeypatch, "analysis_runner.db")
    yield


class FakeGraph:
    """脚本化假图：按顺序吐事件，可选在末尾抛异常。"""

    def __init__(self, events=None, boom=None):
        self.events = events or []
        self.boom = boom

    def stream(self, initial_state):
        for e in self.events:
            yield e
        if self.boom is not None:
            raise self.boom


def _patch_graph(monkeypatch, graph):
    import core.graph_builder as gb
    monkeypatch.setattr(gb, "analysis_graph", graph)


class TestRunAnalysis:
    def test_merge_keeps_additive_and_dict_semantics(self, monkeypatch):
        """⭐ errors 只追加、data_availability 合并 —— 裸 update 会丢掉前面的内容。"""
        _patch_graph(monkeypatch, FakeGraph([
            {"data_fetcher_node": {"errors": ["E1"], "stock_info": {"name": "测试股"},
                                   "data_availability": {"kline": True}}},
            {"tech_analyst_node": {"errors": ["E2"],
                                   "data_availability": {"fund_flow": False}}},
        ]))
        res = AR.run_analysis("600028", source="test")
        assert res["ok"] and res["steps"] == 2
        errs = res["snapshot"]["errors"]
        assert errs == ["E1", "E2"], errs          # 两次都保留（不被覆盖）
        assert res["snapshot"]["data_availability"] == {"kline": True, "fund_flow": False}

    def test_success_persists_analysis_and_returns_id(self, monkeypatch):
        _patch_graph(monkeypatch, FakeGraph([
            {"portfolio_manager_node": {"final_decision": "看多", "trade_plan": {"direction": "买入"}}},
        ]))
        res = AR.run_analysis("600028", source="test")
        assert res["ok"] and res["analysis_id"] > 0
        rows = store.list_analyses(stock_code="600028", limit=5)
        assert rows and rows[0]["source"] == "test"

    def test_failure_is_recorded_and_reported_honestly(self, monkeypatch):
        """⭐ 异常 ⇒ 落"分析失败"样本 + 返回 ok=False（不许假装成功）。"""
        _patch_graph(monkeypatch, FakeGraph([{"data_fetcher_node": {"errors": ["x"]}}],
                                            boom=RuntimeError("网络断了")))
        res = AR.run_analysis("600028", source="test")
        assert res["ok"] is False and "网络断了" in res["error"]
        rows = store.list_analyses(stock_code="600028", limit=5)
        assert rows and rows[0]["decision_label"] == "分析失败"


class TestSummarizeForAi:
    def test_critical_fields_come_first(self):
        """⭐ errors / audit_warnings / data_missing 必须排在最前（会被截断）。"""
        snap = {"stock_code": "600028", "errors": ["可核实性：年份无出处"],
                "audit_warnings": ["可核实性：年份无出处"],
                "data_availability": {"fund_flow": False, "kline": True},
                "tech_report": "技术面" * 500}
        out = AR.summarize_for_ai(snap)
        keys = list(out)
        assert keys[:3] == ["errors", "audit_warnings", "data_missing"], keys
        assert out["data_missing"] == ["fund_flow"]

    def test_long_reports_are_truncated(self):
        out = AR.summarize_for_ai({"tech_report": "多" * 5000}, max_text=100)
        assert len(out["tech_report"]) < 200 and "已截断" in out["tech_report"]


class TestAiDeepAnalysisRunner:
    def test_reuses_todays_analysis_without_rerunning(self, monkeypatch):
        """⭐ 同日同票复用：命中当日记录就直接返回，**不再花 3.5 万 token**。"""
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        store.insert_analysis({"analysis_date": today, "stock_code": "600028",
                               "stock_name": "中国石化", "decision_label": "看多",
                               "source": "ui", "payload": {"stock_code": "600028",
                                                           "final_decision": "看多"}})

        def _boom(code, **kw):
            raise AssertionError("命中复用却仍然真跑了分析")
        monkeypatch.setattr(AR, "run_analysis", _boom)
        res = AR.ai_deep_analysis_runner(trade_date=today)("600028")
        assert res["ok"] and res["reused"] is True
        assert res["summary"]["final_decision"] == "看多"

    def test_runs_when_no_todays_record(self, monkeypatch):
        called = {}

        def _fake(code, **kw):
            called["code"] = code
            return {"ok": True, "analysis_id": 7, "steps": 3, "error": "",
                    "snapshot": {"stock_code": code, "final_decision": "观望"}}
        monkeypatch.setattr(AR, "run_analysis", _fake)
        res = AR.ai_deep_analysis_runner(trade_date="2026-10-15")("600519")
        assert called["code"] == "600519" and res["reused"] is False
        assert res["analysis_id"] == 7 and res["summary"]["final_decision"] == "观望"

    def test_failed_run_not_reused_and_reported(self, monkeypatch):
        """失败记录**不得**被当成可复用结果（否则 AI 会拿到空报告当结论）。"""
        store.insert_analysis({"analysis_date": "2026-10-15", "stock_code": "600519",
                               "decision_label": "分析失败", "source": "ui",
                               "payload": {"fatal_errors": ["boom"]}})
        monkeypatch.setattr(AR, "run_analysis",
                            lambda code, **kw: {"ok": False, "error": "LLM 挂了",
                                                "snapshot": {}, "steps": 1})
        res = AR.ai_deep_analysis_runner(trade_date="2026-10-15")("600519")
        assert res["ok"] is False and "分析失败" in res["error"]


class TestSingleImplementationGuard:
    """结构性守卫：流程只有一份（UI 页面委托执行器）。"""

    def test_ui_page_delegates_instead_of_reimplementing(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(root, "ui", "pages", "2_🔍_深度分析.py"),
                   encoding="utf-8").read()
        assert "analysis_runner.run_analysis" in src, "页面必须委托执行器"
        assert ".stream(" not in src, (
            "页面里又出现了 stream 循环 ⇒ 与 core.analysis_runner 分叉了（单一实现被破坏）")

    def test_scheduler_wires_the_runner(self):
        """⭐ 接线守卫：调度任务必须真的把执行器传进去（=None 就是死工具）。"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(root, "core", "scheduler.py"), encoding="utf-8").read()
        assert "ai_deep_analysis_runner" in src
        assert "analysis_runner=ai_deep_analysis_runner" in src
