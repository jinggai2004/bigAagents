# -*- coding: utf-8 -*-
"""P1 端点契约测试（文档 §6.1 五条验收判据的可执行版本）。

放在 `tests/` 而不是 `web/tests/` 是有意的：项目全量入口是
`pytest tests`（`运行测试.bat`），放 `web/tests/` 就没人跑——**测试写了没跑 = 零覆盖**。

覆盖：
1. `/api/meta` 结构与离线标志；
2. `/api/picks/latest`：新格式 dict / 旧 repr（format="legacy"）/ 文件缺失 / 内容损坏；
3. `stale` 与 `lag_trading_days`：周末不算过期、10-20 → 滞后 3 个交易日（C3）；
4. `/api/dashboard` 分区包络：picks 挂了顶层仍 ok，watch/league 显式 not_implemented（C2）；
5. 零外呼：`gate.blocked() == 0 and socket_attempts() == 0`。
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

NEW_MS = {
    "total_score": 28.4, "market_state": "偏弱震荡", "signal": "卖出/减仓",
    "position_limit": 0.2, "strategy_bias": "极端防御",
    "operation_advice": "建议减仓, 保持防御(等右侧)", "scores": {"trend": 5.0},
    "indicators": {}, "risk_warnings": [], "timestamp": "2026-10-23 15:52:00",
}

LEGACY_MS = (
    "MarketSignal(total_score=28.4, market_state='偏弱震荡', signal='卖出/减仓', "
    "position_limit=0.2, strategy_bias='极端防御', operation_advice='建议减仓, 保持防御', "
    "scores={'trend': 5.0}, indicators={}, risk_warnings=[], timestamp='2026-10-23 15:52:00')"
)

PICKS = [
    {"股票代码": "600276", "股票名称": "恒瑞医药", "板块": "医药",
     "策略": "趋势", "score": 92.9, "score_coverage": 1.0},
    {"股票代码": "000001", "股票名称": "平安银行", "板块": "金融",
     "策略": "低估值", "score": 88.1, "score_coverage": 0.6},
]


@pytest.fixture
def root(tmp_path, monkeypatch):
    """临时数据根 + 独立心跳路径；不碰真实 output/ 与 temp/。"""
    import core.scheduler_health as SH
    import web.server.cache as C

    C.clear()
    (tmp_path / "output" / "daily_picks").mkdir(parents=True)
    (tmp_path / "temp").mkdir()
    monkeypatch.setenv("BIGA_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(SH, "HEARTBEAT_PATH", str(tmp_path / "temp" / "scheduler_heartbeat.json"))
    return tmp_path


@pytest.fixture
def client(root):
    from web.server.app import app
    return TestClient(app)


def write_latest(root, *, date="2026-10-23", market_signal=None, picks=None):
    path = root / "output" / "daily_picks" / "latest.json"
    payload = {
        "date": date, "generated_at": date + " 15:52:00", "mode": "rule",
        "picks": PICKS if picks is None else picks,
        "market_signal": NEW_MS if market_signal is None else market_signal,
        "stage_timings": {"screening": 12.3, "factor": 3.1, "rank": 0.4},
        "skipped": False, "skip_reason": "", "errors": [], "elapsed_seconds": 15.8,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- meta
class TestMeta:
    def test_shape(self, client, root):
        r = client.get("/api/meta")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        d = body["data"]
        for k in ("version", "started_at", "data_root", "offline", "template_version"):
            assert k in d, f"meta 缺字段 {k}"
        assert d["data_root"] == str(root)          # BIGA_DATA_ROOT 生效
        assert d["offline"] is False
        assert d["template_version"] != "unknown"


# ---------------------------------------------------------------- picks
class TestPicksLatest:
    def test_new_format_dict(self, client, root):
        """C1：market_signal 必须是对象，不是 repr 串。"""
        write_latest(root)
        body = client.get("/api/picks/latest").json()
        assert body["ok"] is True
        d = body["data"]
        assert d["format"] == "dict"
        assert isinstance(d["market_signal"], dict)
        assert d["market_signal"]["total_score"] == 28.4
        assert d["market_signal"]["position_limit"] == 0.2
        assert body["as_of"] == "2026-10-23"

    def test_chinese_keys_passthrough(self, client, root):
        """C4：中文键原样透传，不转英文。"""
        write_latest(root)
        d = client.get("/api/picks/latest").json()["data"]
        first = d["picks"][0]
        assert first["股票代码"] == "600276"
        assert first["股票名称"] == "恒瑞医药"
        assert "code" not in first and "name" not in first

    def test_legacy_repr_flagged(self, client, root):
        """旧文件（repr 串）⇒ format=legacy，且仍能给出对象（不静默、可排查）。"""
        write_latest(root, market_signal=LEGACY_MS)
        body = client.get("/api/picks/latest").json()
        assert body["ok"] is True
        d = body["data"]
        assert d["format"] == "legacy"
        assert d["market_signal"]["total_score"] == 28.4
        assert d["market_signal"]["market_state"] == "偏弱震荡"

    def test_missing_file(self, client):
        """判据 1：文件不存在 ⇒ ok=false + error.type=missing（不返空表）。"""
        body = client.get("/api/picks/latest").json()
        assert body["ok"] is False
        assert body["data"] is None
        assert body["error"]["type"] == "missing"

    def test_corrupt_file(self, client, root):
        """文件在但坏了 ⇒ empty，与 missing 区分（没跑 vs 坏了）。"""
        (root / "output" / "daily_picks" / "latest.json").write_text("{oops", encoding="utf-8")
        body = client.get("/api/picks/latest").json()
        assert body["ok"] is False
        assert body["error"]["type"] == "empty"

    def test_unknown_signal_format_not_invented(self, client, root):
        """认不出的 market_signal ⇒ null + format=unknown，绝不造默认值。"""
        write_latest(root, market_signal="garbage")
        d = client.get("/api/picks/latest").json()["data"]
        assert d["market_signal"] is None
        assert d["format"] == "unknown"


# ---------------------------------------------------------------- stale / lag
class TestStale:
    """判据 2：stale 的基准是"最近应收盘的交易日"，不是日历末元素。"""

    def test_weekend_not_stale(self, root):
        """周六（10-24）看周五（10-23）的数据 ⇒ 不滞后、不告警。"""
        from web.server import picks
        write_latest(root, date="2026-10-23")
        d = picks.load_latest(now=datetime(2026, 10, 24, 10, 0))["data"]
        assert d["lag_trading_days"] == 0
        assert d["stale"] is False

    def test_intraday_expects_previous_day(self, root):
        """交易日盘中（10-23 10:00）⇒ 昨日（10-22）才是应有数据，10-23 的更"新"不算滞后。"""
        from web.server import picks
        write_latest(root, date="2026-10-23")
        d = picks.load_latest(now=datetime(2026, 10, 23, 10, 0))["data"]
        assert d["expected_date"] == "2026-10-22"
        assert d["stale"] is False

    def test_acceptance_lag_three_days(self, root):
        """判据 2 原文：date 改成 10-20 → 滞后 3 个交易日。"""
        from web.server import picks
        write_latest(root, date="2026-10-20")
        d = picks.load_latest(now=datetime(2026, 10, 25, 14, 0))["data"]
        assert d["expected_date"] == "2026-10-23"
        assert d["lag_trading_days"] == 3
        assert d["stale"] is True


# ---------------------------------------------------------------- dashboard
class TestDashboard:
    def test_zones_all_present(self, client, root, monkeypatch):
        # P2 起 watch 已接：这里也 patch 行情源——dashboard 测试不该联网，
        # 断言的是"分区都在、各自成败"，不是"今天指数多少"。
        # P3 起 league / observations 也实接：同样 patch 掉数据源，
        # 否则会读真实库（测试不该耦合真实环境）。
        import data.index_bars as IB
        import data.realtime as RT
        import core.ai_league as L
        monkeypatch.setattr(IB, "get_index_bars", lambda s: None)
        monkeypatch.setattr(RT, "get_realtime_quotes",
                            lambda codes, force_refresh=False: {})
        monkeypatch.setattr(L, "leaderboard", lambda **k: {
            "as_of": "2026-10-23", "caveat": "",
            "groups": {"daily": {"rows": [], "random_mean_return_pct": None,
                                 "random_count": 0}}})
        monkeypatch.setattr("web.server.review.pending",
                            lambda limit=50: {"ok": True, "data": {"count": 0, "items": []},
                                              "as_of": None, "fetched_at": "x"})
        write_latest(root)
        body = client.get("/api/dashboard").json()
        assert body["ok"] is True                     # 顶层只表服务正常
        zones = body["data"]
        assert zones["picks"]["ok"] is True
        assert zones["health"]["ok"] is True
        assert zones["watch"]["ok"] is True           # P2 已接（不再是 not_implemented）
        assert all(r["price"] is None for r in zones["watch"]["data"]["indices"])
        assert zones["league"]["ok"] is True          # P3 已接
        assert zones["league"]["data"]["rows"] == []
        assert zones["observations"]["ok"] is True

    def test_picks_zone_failure_keeps_top_ok(self, client):
        """C2 核心：picks 挂了，dashboard 顶层仍是 ok，health 照常。"""
        body = client.get("/api/dashboard").json()
        assert body["ok"] is True
        assert body["data"]["picks"]["ok"] is False
        assert body["data"]["picks"]["error"]["type"] == "missing"
        assert body["data"]["health"]["ok"] is True

    def test_health_without_heartbeat(self, client):
        """没有心跳 = 调度器没跑：是数据，不是错误。"""
        d = client.get("/api/dashboard").json()["data"]["health"]
        assert d["ok"] is True
        assert d["data"]["running"] is False
        assert d["data"]["pid"] is None

    def test_health_with_heartbeat(self, client, root):
        import time
        hb = {"pid": 999999, "jobs": 18, "ts": "2026-10-25 14:50:00",
              "epoch": time.time() - 30, "started_at": time.time() - 600}
        (root / "temp" / "scheduler_heartbeat.json").write_text(
            json.dumps(hb), encoding="utf-8")
        d = client.get("/api/dashboard").json()["data"]["health"]["data"]
        assert d["age_seconds"] is not None and d["age_seconds"] < 60
        assert d["stale"] is False
        assert d["jobs"] == 18
        assert d["running"] is False      # 假 PID：进程检查必须如实 False


# ---------------------------------------------------------------- 零外呼
class TestNoOutbound:
    def test_endpoints_do_not_touch_network(self, client, root):
        """判据 5：P1 只读文件与 DB，端点不得发起任何外呼。

        ⚠️ 为什么不断言 `socket_attempts() == 0`（2026-10-25 实测）：
        Windows 的 asyncio proactor 会用**回环 socketpair** 做自管道，
        TestClient 每跑一次就记 3 次 `socket.connect(('127.0.0.1', ...))`——
        这是进程内 IPC，不是外呼。断言"没有回环以外的地址"才既真又严。
        """
        from data import offline_gate as gate
        write_latest(root)
        gate.install()
        gate.reset()
        try:
            for url in ("/api/meta", "/api/picks/latest", "/api/dashboard"):
                assert client.get(url).status_code == 200
            stats = gate.stats()
            assert stats["blocked"] == 0                  # 没有任何被拦截的 HTTP 外呼
            external = [s for s in stats["socket_samples"]
                        if "127.0.0.1" not in s["url"] and "::1" not in s["url"]]
            assert not external, f"端点发起了非回环 socket 连接：{external[:3]}"
        finally:
            gate.uninstall()
            gate.reset()


# ---------------------------------------------------------------- 静态页
class TestStaticUI:
    def test_index_served(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "bigAagents" in r.text.lower() or "<html" in r.text.lower()
