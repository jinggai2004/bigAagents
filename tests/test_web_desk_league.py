# -*- coding: utf-8 -*-
"""交易台「联赛每日决策与复盘」+ 仪表盘「涨跌家数」的接线测试。

存在理由（一句话）：这两块数据**早就写在库里**（`ai_decisions` 62 行 / `ai_reviews`
48 行 / `market_snapshot` 每日全市场），但 web 层零引用 —— 用户在页面上看不到。
本项目已踩过同型问题（"报告无法回看、胜率无法回看"：函数有实现有单测，前端零引用）。

## 守什么
1. `/api/desk/league/day` 真的把决策/复盘搬出来（断言**该端点独有**的
   `coverage` 字段 —— 兜底路由下 `ok:true` 是恒真的，只断言 ok 等于没断言）；
2. 没有决策的日期返 `missing`（不是空表冒充正常）；
3. 首页「涨跌家数」不再是 `data-demo` 写死值（静态文本守卫）。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from web.server.app import app
    import web.server.cache as C
    C.clear()
    yield TestClient(app)
    C.clear()


@pytest.fixture
def db(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    return use_template_db(tmp_path, monkeypatch, "test_desk_league.db")


@pytest.fixture
def root(tmp_path, monkeypatch):
    import core.ui_runs as ur
    (tmp_path / "output" / "ui_runs").mkdir(parents=True)
    (tmp_path / "output" / "daily_picks").mkdir(parents=True)
    (tmp_path / "temp").mkdir()
    monkeypatch.setenv("BIGA_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(ur, "UI_RUNS_DIR", str(tmp_path / "output" / "ui_runs"))
    yield tmp_path


def _seed():
    from db import store
    store.save_ai_decision({"persona": "ai_value", "signal_date": "2026-10-26",
                            "code": "601088", "advice_status": "可执行",
                            "direction": "看多", "position_size": 0.08,
                            "order_reason": "煤炭龙头，估值合理",
                            "model": "deepseek-v4-flash"})
    store.save_ai_decision({"persona": "ai_risk", "signal_date": "2026-10-26",
                            "code": "-", "advice_status": "等待",
                            "direction": None, "position_size": 0.0,
                            "order_reason": "空仓等待"})
    store.save_ai_review("ai_value", "2026-10-26",
                         review_text="## 今日复盘\n分散建仓 3 只。",
                         summary="分散建仓", model="deepseek-v4-flash")
    store.save_market_snapshot("2026-10-26", [
        {"stock_code": "601088", "stock_name": "中国神华", "price": 45.0,
         "change_pct": 1.9},
        {"stock_code": "600519", "stock_name": "贵州茅台", "price": 1350.0,
         "change_pct": -0.3},
    ])


class TestLeagueDay:
    def test_returns_decisions_and_reviews(self, root, db, client):
        _seed()
        z = client.get("/api/desk/league/day?date=2026-10-26").json()
        assert z["ok"] is True, z
        d = z["data"]
        # ⚠️ 断言端点**独有**字段（兜底路由下 ok:true 恒真）
        assert d["coverage"]["decisions"] == 2
        assert d["coverage"]["reviews"] == 1
        assert d["date"] == "2026-10-26"
        by = {p["persona"]: p for p in d["personas"]}
        assert by["ai_value"]["decisions"][0]["code"] == "601088"
        assert by["ai_value"]["decisions"][0]["name"] == "中国神华", "名称没补上"
        assert "分散建仓" in by["ai_value"]["review"]["text"]
        assert by["ai_risk"]["decisions"][0]["direction"] is None   # 观望如实为 None

    def test_persona_order_and_label(self, root, db, client):
        _seed()
        d = client.get("/api/desk/league/day?date=2026-10-26").json()["data"]
        labels = [p["label"] for p in d["personas"]]
        assert "价值派" in labels and "风控流" in labels
        # LLM 人设排在随机对照前（配置顺序）
        if "随机对照 1" in labels:
            assert labels.index("价值派") < labels.index("随机对照 1")

    def test_no_decisions_is_missing(self, root, db, client):
        z = client.get("/api/desk/league/day").json()
        assert z["ok"] is False
        assert (z["error"] or {}).get("type") == "missing", z

    def test_dates_list_for_picker(self, root, db, client):
        _seed()
        d = client.get("/api/desk/league/day").json()["data"]
        assert d["dates"] == ["2026-10-26"]
        assert d["date"] == "2026-10-26"


class TestBreadthZone:
    def test_dashboard_has_breadth(self, root, db, client, monkeypatch):
        _seed()
        from core import breadth
        monkeypatch.setattr(breadth, "_universe_size", lambda: 2)
        monkeypatch.setattr(breadth, "_expected_date", lambda: "2026-10-26")
        z = client.get("/api/dashboard").json()
        assert z["ok"] is True
        bz = z["data"]["breadth"]
        assert bz["ok"] is True
        b = bz["data"]
        assert b["available"] is True
        assert (b["up"], b["down"]) == (1, 1)
        assert b["covered"] == 2 and b["universe"] == 2

    def test_home_no_longer_hardcodes_breadth(self):
        """首页「涨跌家数」不能再挂写死的演示数（3,124 / 1,689 / 炸板率 21%）"""
        import os
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "web", "templates", "index.html")
        html = open(p, encoding="utf-8").read()
        assert 'data-demo="breadth"' not in html, "涨跌家数仍是演示数据"
        assert "3,124" not in html and "炸板率" not in html
        assert 'id="kpi-breadth"' in html, "缺少真实宽度的渲染锚点"

    def test_desk_has_league_day_panel(self):
        """交易台模板必须真的引用这个端点（"函数写好了"≠"页面看得见"）"""
        import os
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "web", "templates", "desk.html")
        html = open(p, encoding="utf-8").read()
        assert "/api/desk/league/day" in html
        assert 'id="ld-body"' in html
