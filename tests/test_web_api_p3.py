# -*- coding: utf-8 -*-
"""P3 契约测试：复盘 / 交易台 / 研究（8 端点）+ 模板结构守卫。

## 这组测试守的四条线
1. **缺就是缺**：没有产物 ⇒ `missing`；产物坏了 ⇒ `empty`；两者不可合并。
2. **不返 0**：读不到收盘价 ⇒ `price=null` 且 `valuation_complete=false`，
   `total_value` 置 `null`（偏低的总资产比没有更坏——它会让人算错仓位）。
3. **端点不联网**：`PortfolioManager` 必须收到 `allow_network=False` 与批量价，
   否则 akshare 无超时会挂住整个服务。
4. **前端零外网依赖**：模板里不许出现 http(s) 外链（字体已本地化），
   且每个 `data-demo` 都必须带可读名（页脚汇总靠它）。
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


# ================================================================
# 夹具
# ================================================================
@pytest.fixture
def client():
    from web.server.app import app
    import web.server.cache as C
    C.clear()
    yield TestClient(app)
    C.clear()


@pytest.fixture
def root(tmp_path, monkeypatch):
    """临时数据根 + 独立复盘目录（路径全部运行时解析，见 settings 的纪律）。"""
    import web.server.cache as C
    C.clear()
    (tmp_path / "output" / "daily_picks").mkdir(parents=True)
    (tmp_path / "output" / "daily_reports").mkdir(parents=True)
    (tmp_path / "output" / "ui_runs").mkdir(parents=True)
    (tmp_path / "data" / "backtest_records").mkdir(parents=True)
    (tmp_path / "temp").mkdir()
    monkeypatch.setenv("BIGA_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("BIGA_REVIEW_DIR", raising=False)
    yield tmp_path
    C.clear()


@pytest.fixture
def db(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    return use_template_db(tmp_path, monkeypatch)


def _md(d: str, text: str = "# 复盘\n\n## 大盘\n\n上证 3,690") -> str:
    return f"# A股每日复盘 {d}\n\n> 生成时间 15:52\n\n## 大盘概览\n\n{text}\n"


def _report(root, d: str, text: str = None):
    """d 形如 20261023。"""
    p = root / "output" / "daily_reports" / f"review_{d}.md"
    p.write_text(text if text is not None else _md(d), encoding="utf-8")
    return p


# ================================================================
# 复盘
# ================================================================
class TestReview:
    def test_latest_missing_when_no_file(self, root, client):
        body = client.get("/api/review/latest").json()
        assert body["ok"] is False
        assert body["error"]["type"] == "missing"

    def test_latest_picks_newest_by_date_not_mtime(self, root, client):
        """按**文件名日期**取最新。曾经按 mtime ⇒ 手工回填的旧报告会抢到"最新"。"""
        import os
        old = _report(root, "20261020")
        new = _report(root, "20261023")
        os.utime(old, (2_000_000_000, 2_000_000_000))   # 旧文件 mtime 更新
        os.utime(new, (1_000_000_000, 1_000_000_000))
        body = client.get("/api/review/latest").json()
        assert body["data"]["report"]["data"]["date"] == "2026-10-23"
        assert body["data"]["report"]["data"]["file"] == "review_20261023.md"

    def test_latest_marks_stale_against_expected_date(self, root, client, monkeypatch):
        from web.server import picks
        monkeypatch.setattr(picks, "expected_date", lambda now=None: "2026-10-24")
        _report(root, "20261023")
        body = client.get("/api/review/latest").json()
        assert body["data"]["stale"] is True
        assert body["data"]["expected_date"] == "2026-10-24"

    def test_latest_truncates_instead_of_blowing_up(self, root, client, monkeypatch):
        from web.server import review
        monkeypatch.setattr(review, "_MAX_CHARS", 20)
        _report(root, "20261023", text="x" * 500)
        rep = client.get("/api/review/latest").json()["data"]["report"]["data"]
        assert rep["truncated"] is True
        assert rep["chars"] == 20

    def test_stats_failure_does_not_kill_report(self, root, client, monkeypatch):
        """统计坏掉只影响统计区：正文照常返回（分区包络的意义）。"""
        import db.store as store
        monkeypatch.setattr(store, "observation_stats",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        _report(root, "20261023")
        body = client.get("/api/review/latest").json()
        assert body["ok"] is True
        assert body["data"]["report"]["ok"] is True
        assert body["data"]["stats"]["ok"] is False
        assert body["data"]["stats"]["error"]["type"] == "empty"

    def test_archive_invalid_date(self, root, client):
        _report(root, "20261023")
        body = client.get("/api/review/archive", params={"date": "2026/10/23"}).json()
        assert body["ok"] is False
        assert body["error"]["type"] == "invalid"

    def test_archive_missing_specific_date(self, root, client):
        _report(root, "20261023")
        body = client.get("/api/review/archive", params={"date": "2026-10-22"}).json()
        assert body["error"]["type"] == "missing"

    def test_archive_returns_markdown(self, root, client):
        _report(root, "20261023")
        body = client.get("/api/review/archive", params={"date": "2026-10-23"}).json()
        assert body["ok"] is True
        assert body["data"]["markdown"].startswith("# A股每日复盘")

    def test_archive_list_desc(self, root, client):
        _report(root, "20261020")
        _report(root, "20261023")
        items = client.get("/api/review/archive").json()["data"]["items"]
        assert [r["date"] for r in items] == ["2026-10-23", "2026-10-20"]

    def test_pending_empty_is_ok_not_error(self, root, db, client):
        """模板库 0 条观察点 ⇒ 空列表是**正常态**（不是 error）。"""
        _report(root, "20261023")
        rv = client.get("/api/review/latest").json()
        assert rv["data"]["pending"]["ok"] is True
        assert rv["data"]["pending"]["data"]["count"] == 0


# ================================================================
# 交易台 · 真持仓
# ================================================================
def _fake_pm(monkeypatch, *, positions, cash=1000.0, captured=None):
    """替换 PortfolioManager 的四个读方法，只观察**服务层怎么调用**。"""
    from core.portfolio import Portfolio, PortfolioManager

    captured = captured if captured is not None else {}

    def _load(self, prices=None, *, allow_network=True):
        captured["load"] = {"prices": dict(prices or {}), "allow_network": allow_network}
        mv = 0.0
        for p in positions:
            px = (prices or {}).get(p.code)
            if isinstance(px, (int, float)) and px > 0:
                mv += px * p.quantity
        return Portfolio(positions=list(positions), cash=cash, total_value=cash + mv,
                         total_pnl_pct=0.0)

    def _active(self, prices=None, *, allow_network=True):
        captured["active"] = allow_network
        return list(positions)

    def _stops(self, prices=None, *, allow_network=True):
        captured["stops"] = allow_network
        return []

    def _expo(self, prices=None, *, allow_network=True):
        captured["expo"] = allow_network
        return {"科技": 0.5}

    monkeypatch.setattr(PortfolioManager, "load", _load)
    monkeypatch.setattr(PortfolioManager, "get_active_positions", _active)
    monkeypatch.setattr(PortfolioManager, "check_stop_loss", _stops)
    monkeypatch.setattr(PortfolioManager, "get_exposure", _expo)
    monkeypatch.setattr(PortfolioManager, "active_codes", lambda self: [p.code for p in positions])
    monkeypatch.setattr(PortfolioManager, "_get_current_price",
                        lambda self, code: pytest.fail("端点不得联网取价"))
    return captured


def _pos(code="600000", name="浦发银行", entry=10.0, qty=100):
    from db.models import Position
    return Position(id=1, code=code, name=name, entry_date="2026-10-20",
                    entry_price=entry, quantity=qty, stop_loss=9.0,
                    take_profit=12.0, status="active")


def _bars_df(close: float, date: str = "2026-10-23"):
    import pandas as pd
    return pd.DataFrame({"date": pd.to_datetime([date]), "close": [close]})


class TestDeskPortfolio:
    def test_passes_prices_and_disables_network(self, root, client, monkeypatch):
        """核心纪律：`allow_network=False` + 批量价一起传，端点零外呼。"""
        import data.bars as B
        monkeypatch.setattr(B, "get_bars", lambda c, *a, **k: _bars_df(10.5))
        monkeypatch.setattr(B, "panel_date_max", lambda *a, **k: "2026-10-23")
        cap = _fake_pm(monkeypatch, positions=[_pos()])
        body = client.get("/api/desk/portfolio").json()
        assert body["ok"] is True
        assert cap["load"]["allow_network"] is False
        assert cap["load"]["prices"]["600000"] == 10.5
        assert cap["active"] is False and cap["stops"] is False and cap["expo"] is False
        assert body["data"]["positions"][0]["price"] == 10.5
        assert body["data"]["valuation_complete"] is True

    def test_missing_price_is_null_not_zero(self, root, client, monkeypatch):
        """本地无该只 ⇒ null + 估值不完整 + total_value=null（偏低的合计更危险）。"""
        import data.bars as B
        monkeypatch.setattr(B, "get_bars", lambda c, *a, **k: None)
        monkeypatch.setattr(B, "panel_date_max", lambda *a, **k: None)
        _fake_pm(monkeypatch, positions=[_pos()])
        d = client.get("/api/desk/portfolio").json()["data"]
        assert d["positions"][0]["price"] is None
        assert d["positions"][0]["market_value"] is None
        assert d["positions"][0]["available"] is False
        assert d["valuation_complete"] is False
        assert d["total_value"] is None
        assert d["missing_codes"] == ["600000"]

    def test_empty_positions_is_ok(self, root, client, monkeypatch):
        import data.bars as B
        monkeypatch.setattr(B, "get_bars", lambda c, *a, **k: None)
        monkeypatch.setattr(B, "panel_date_max", lambda *a, **k: None)
        _fake_pm(monkeypatch, positions=[])
        body = client.get("/api/desk/portfolio").json()
        assert body["ok"] is True
        assert body["data"]["count"] == 0
        assert body["data"]["valuation_complete"] is True   # 无仓位就不存在缺价

    def test_failure_is_empty_not_zero(self, root, client, monkeypatch):
        from core.portfolio import PortfolioManager

        def _boom(self, *a, **k):
            raise RuntimeError("db locked")

        monkeypatch.setattr(PortfolioManager, "load", _boom)
        monkeypatch.setattr(PortfolioManager, "active_codes", lambda self: [])
        body = client.get("/api/desk/portfolio").json()
        assert body["ok"] is False
        assert body["error"]["type"] == "empty"
        assert body["data"] is None


# ================================================================
# 交易台 · 模拟账户 / 联赛
# ================================================================
class TestDeskSim:
    def test_invalid_account(self, root, db, client):
        body = client.get("/api/desk/sim", params={"account": "nope"}).json()
        assert body["ok"] is False
        assert body["error"]["type"] == "invalid"

    def test_no_account_at_all(self, root, db, client):
        body = client.get("/api/desk/sim").json()
        assert body["ok"] is False
        assert body["error"]["type"] in ("missing",)

    def test_accounts_list_present(self, root, db, client, monkeypatch):
        import db.store as store
        monkeypatch.setattr(store, "list_paper_accounts", lambda: ["ai_value", "default"])
        monkeypatch.setattr(store, "load_paper_state", lambda name="default": {
            "name": name, "initial_cash": 1_000_000.0, "cash": 500_000.0,
            "positions": {}, "orders": [], "fills": [],
            "nav": [{"date": "2026-10-23", "cash": 500_000.0,
                     "position_value": 0.0, "total_value": 500_000.0,
                     "return_pct": -50.0, "at_cost": 0}],
        })
        d = client.get("/api/desk/sim").json()["data"]
        assert d["account"] == "ai_value"        # 优先 AI 主账户，不选空账户
        assert d["accounts"] == ["ai_value", "default"]
        assert d["total_value"] == 500_000.0
        assert d["return_pct"] == -50.0


class TestDeskLeague:
    def _board(self):
        return {"as_of": "2026-10-23", "caveat": "样本不足",
                "groups": {"daily": {"random_mean_return_pct": -0.39, "random_count": 3,
                                     "rows": [{"persona": "ai_tech", "label": "技术派",
                                               "kind": "llm", "return_pct": 1.2,
                                               "nav_days": 8, "conclusive": False}]}}}

    def test_rows_and_target_from_latest(self, root, client, monkeypatch):
        import core.ai_league as L
        monkeypatch.setattr(L, "leaderboard", self._board)
        (root / "output" / "daily_picks" / "latest.json").write_text(json.dumps({
            "date": "2026-10-23", "picks": [],
            "market_signal": {"position_limit": 0.2, "market_state": "偏弱震荡",
                              "signal": "谨慎"},
        }, ensure_ascii=False), encoding="utf-8")
        d = client.get("/api/desk/league").json()["data"]
        assert d["groups"]["daily"]["rows"][0]["label"] == "技术派"
        assert d["target"]["position_limit"] == 0.2      # 读文件，不现算 detect()
        assert d["target"]["date"] == "2026-10-23"

    def test_target_absent_is_null_not_guessed(self, root, client, monkeypatch):
        import core.ai_league as L
        monkeypatch.setattr(L, "leaderboard", self._board)
        d = client.get("/api/desk/league").json()["data"]
        assert d["target"]["position_limit"] is None

    def test_failure(self, root, client, monkeypatch):
        import core.ai_league as L

        def _boom(**k):
            raise RuntimeError("sqlite locked")
        monkeypatch.setattr(L, "leaderboard", _boom)
        body = client.get("/api/desk/league").json()
        assert body["ok"] is False
        assert body["error"]["type"] == "empty"


# ================================================================
# 研究
# ================================================================
def _bt_record(total_return=-5.16, trades=3):
    return {
        "total_return": total_return, "annual_return": -5.16, "sharpe_ratio": -3.15,
        "calmar_ratio": -0.71, "max_drawdown": -7.29, "win_rate": 21.4,
        "profit_loss_ratio": 0.63, "total_trades": trades, "turnover_rate": 17033.3,
        "information_ratio": 3.83, "monthly_win_rate": 33.3, "rejected_orders": 0,
        "avg_position_pct": 45.3,
        "nav_curve": [{"date": "2026-10-22", "total_value": 1_000_000.0},
                      {"date": "2026-10-23", "total_value": 948_439.75}],
        "benchmark_nav": [{"date": "2026-10-22", "total_value": 1_000_000.0}],
        "yearly_returns": {"2026": -5.16},
        "trade_history": [{"date": f"2026-10-0{i}", "code": "000045"} for i in range(1, trades + 1)],
    }


class TestResearchBacktest:
    def test_missing_dir(self, root, client):
        body = client.get("/api/research/backtest").json()
        assert body["ok"] is False
        assert body["error"]["type"] == "missing"

    def test_picks_newest_file(self, root, client):
        rec = root / "data" / "backtest_records"
        (rec / "backtest_20261020_100000.json").write_text(
            json.dumps(_bt_record(total_return=1.0)), encoding="utf-8")
        (rec / "backtest_20261023_155251.json").write_text(
            json.dumps(_bt_record()), encoding="utf-8")
        d = client.get("/api/research/backtest").json()["data"]
        assert d["file"] == "backtest_20261023_155251.json"
        assert d["metrics"]["total_return"] == -5.16
        assert d["end"] == "2026-10-23"

    def test_bad_json_is_empty(self, root, client):
        (root / "data" / "backtest_records" / "backtest_20261023_1.json").write_text(
            "{not json", encoding="utf-8")
        body = client.get("/api/research/backtest").json()
        assert body["ok"] is False
        assert body["error"]["type"] == "empty"

    def test_path_traversal_rejected(self, root, client):
        (root / "data" / "backtest_records" / "backtest_20261023_1.json").write_text(
            json.dumps(_bt_record()), encoding="utf-8")
        body = client.get("/api/research/backtest",
                          params={"file": "../secret.json"}).json()
        assert body["ok"] is False
        assert body["error"]["type"] == "invalid"


class TestResearchFactors:
    def test_missing(self, root, client):
        body = client.get("/api/research/factors").json()
        assert body["error"]["type"] == "missing"

    def test_passes_summary_through(self, root, client):
        path = root / "output" / "ui_runs" / "factor_analysis.json"
        path.write_text(json.dumps({
            "generated_at": "2026-10-17 22:51:45",
            "payload": {"summary": [{"key": "momentum20", "因子": "20日动量",
                                     "IC均值": -0.1456}],
                        "layered": {}, "correlation": {}, "periods": 74, "names": 20,
                        "horizon": 20, "step": 3, "groups": 5,
                        "unavailable": {"pe": "需要历史截面"},
                        "snapshot_status": {"dates": 8, "need": 60}},
        }, ensure_ascii=False), encoding="utf-8")
        d = client.get("/api/research/factors").json()["data"]
        assert d["summary"][0]["因子"] == "20日动量"
        assert d["unavailable"]["pe"] == "需要历史截面"
        assert d["snapshot_status"]["need"] == 60

    def test_corrupt_file(self, root, client):
        (root / "output" / "ui_runs" / "factor_analysis.json").write_text(
            "]]", encoding="utf-8")
        body = client.get("/api/research/factors").json()
        assert body["error"]["type"] == "empty"


# ================================================================
# 核心：allow_network 开关（默认不变，只读层能关）
# ================================================================
class TestOfflineValuation:
    def test_disabled_returns_zero_and_never_fetches(self, db, monkeypatch):
        from core.portfolio import PortfolioManager
        pm = PortfolioManager()
        monkeypatch.setattr(pm, "_get_current_price",
                            lambda code: pytest.fail("allow_network=False 仍去取价"))
        assert pm._price_of("600000", {}, allow_network=False) == 0.0

    def test_default_still_falls_back_to_fetch(self, db, monkeypatch):
        """默认行为不变：既有调用方（Streamlit 页/调度）不传参就得照旧能取到价。"""
        from core.portfolio import PortfolioManager
        pm = PortfolioManager()
        monkeypatch.setattr(pm, "_get_current_price", lambda code: 12.5)
        assert pm._price_of("600000", {}) == 12.5

    def test_prices_win_over_fetch(self, db, monkeypatch):
        from core.portfolio import PortfolioManager
        pm = PortfolioManager()
        monkeypatch.setattr(pm, "_get_current_price",
                            lambda code: pytest.fail("给了价还去取"))
        assert pm._price_of("600000", {"600000": 10.0}, allow_network=False) == 10.0


# ================================================================
# 模板结构守卫（A1/A2）
# ================================================================
class TestTemplates:
    def _pages(self):
        import glob
        import os
        return sorted(glob.glob(os.path.join("web", "templates", "*.html")))

    def test_no_external_assets(self):
        """P6 硬门槛的前置：模板里不能有外网 link/script（字体已本地化）。"""
        import re
        bad = []
        for p in self._pages():
            s = open(p, encoding="utf-8").read()
            for m in re.finditer(r'<(?:link|script|img)[^>]*(?:href|src)="(https?://[^"]+)"', s):
                bad.append((p, m.group(1)))
        assert bad == []

    def test_local_fonts_referenced(self):
        for p in self._pages():
            s = open(p, encoding="utf-8").read()
            assert "../static/fonts/fonts.css" in s, p

    def test_font_files_exist(self):
        import os
        for f in ("web/static/fonts/fonts.css", "web/static/fonts/space-grotesk.woff2",
                  "web/static/fonts/jetbrains-mono.woff2", "web/static/js/demo.js"):
            assert os.path.getsize(f) > 0, f

    def test_every_demo_marker_has_name(self):
        """页脚"演示 …"由 `data-demo-name` 汇总 ⇒ 漏写名字就漏报（假绿）。"""
        import re
        for p in self._pages():
            s = open(p, encoding="utf-8").read()
            for tag in re.findall(r"<[^>]*data-demo=[^>]*>", s):
                assert "data-demo-name=" in tag, (p, tag[:80])

    def test_demo_script_included(self):
        for p in self._pages():
            s = open(p, encoding="utf-8").read()
            assert "../static/js/demo.js" in s, p


# ================================================================
# 研究 v4（2026-10-29）：验证与归因三端点
# ================================================================

class TestResearchValidate:
    def test_missing_when_no_sweep(self, root, client):
        body = client.get("/api/research/backtest/validate").json()
        assert body["ok"] is False
        assert body["error"]["type"] == "missing"

    def test_reads_latest_sweep(self, root, client):
        rec = root / "data" / "backtest_records"
        rec.mkdir(parents=True, exist_ok=True)
        payload = {"built_at": "2026-10-29 15:00:00", "start": "2024-01-01",
                   "end": "2025-12-31",
                   "grid": {"thresholds": [55, 60], "atr_mults": [2.0], "holding_days": [20]},
                   "results": [{}, {}],
                   "multiple_testing": {"n_trials": 2,
                                         "best_config": {"threshold": 60, "atr_mult": 2.0,
                                                         "holding_days": 20},
                                         "dsr": {"ok": True, "p_value": 0.03,
                                                 "significant": True, "sr_star": 0.2,
                                                 "hurdle_source": "trials",
                                                 "degraded": False},
                                         "pbo": {"ok": True, "pbo": 0.1, "warn": False,
                                                 "n_combos": 12870, "s_blocks": 16}}}
        (rec / "sweep_insample_2024-01-01_2025-12-31.json").write_text(
            json.dumps(payload), encoding="utf-8")
        d = client.get("/api/research/backtest/validate").json()["data"]
        assert d["n_results"] == 2
        assert d["multiple_testing"]["n_trials"] == 2
        assert d["multiple_testing"]["dsr"]["significant"] is True

    def test_bad_json_is_empty(self, root, client):
        rec = root / "data" / "backtest_records"
        rec.mkdir(parents=True, exist_ok=True)
        (rec / "sweep_insample_x.json").write_text("{not json", encoding="utf-8")
        body = client.get("/api/research/backtest/validate").json()
        assert body["ok"] is False and body["error"]["type"] == "empty"


class TestResearchWalkforward:
    def test_missing_when_no_file(self, root, client):
        body = client.get("/api/research/walkforward").json()
        assert body["ok"] is False and body["error"]["type"] == "missing"

    def test_reads_latest(self, root, client):
        path = root / "output" / "walk_forward_20261029.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"built_at": "2026-10-29 17:00:00", "train_years": 4,
                   "horizons": {"5": {"summary": {"available": True, "n_folds": 2},
                                       "folds": [{"test_year": 2026,
                                                  "train_years": [2022, 2023, 2024, 2025],
                                                  "weights_refit": {"momentum_20": 1.0},
                                                  "oos_refit": {"available": True,
                                                                "top_excess_mean": 0.01},
                                                  "oos_baseline": {"available": True,
                                                                   "top_excess_mean": -0.0067}}]}}}
        path.write_text(json.dumps(payload), encoding="utf-8")
        d = client.get("/api/research/walkforward").json()["data"]
        assert d["horizons"]["5"]["summary"]["n_folds"] == 2
        assert d["horizons"]["5"]["folds"][0]["test_year"] == 2026


class TestResearchAttributions:
    def test_empty_account_ok_with_zero_trips(self, root, db, client):
        """模板库（0 订单）⇒ ok + 0 回合（新装机的合法空态，不是 missing）。
        ⚠️ 必须挂 db 夹具：不重定向 DB 会读到真库的 7 回合"""
        body = client.get("/api/research/attributions").json()
        assert body["ok"] is True
        assert body["data"]["n_trips"] == 0
