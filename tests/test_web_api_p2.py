# -*- coding: utf-8 -*-
"""P2 端点契约测试：看盘五区 + 情报面。

## 为什么这组测试必须 mock 行情源
`/api/watch/*` 是全项目**唯一允许外呼**的端点。测试若真打腾讯/新浪：
① 断网即失败（项目已有 `data/offline_gate.py`，测试不该依赖网络）；
② 结果随时间变（今天涨明天跌），断言无从写起。
所以这里 patch 掉 `data.realtime` / `data.index_bars` / `data.us_market` 的入口函数，
**验的是"服务层怎么处理这些返回值"**，而不是"行情源今天给了什么"。

## 覆盖的判据（都是踩过的坑）
1. 实时价优先、缺失时用本地日线最后收盘并标 `basis:"close"`（**不返 0**）；
2. 本地也没有 ⇒ `price=None`（缺就是缺）；
3. 离线 ⇒ 不去等超时，直接走本地；
4. 自选某只取不到 ⇒ `available:false`，不拿 0 顶位；
5. 行业快照空 ⇒ `missing`（不是空表冒充正常）；NaN ⇒ `None`；
6. 异动空列表是**正常态**（非交易日），不是 error；
7. 情报的统计分 `window`（时间窗全量）与 `filtered`（当前筛选）两层，不混；
8. 行业筛选在 **SQL 侧**生效（不是取一页再筛，否则分页数会骗人）；
9. 未知行业名 ⇒ `invalid`，不静默返回全量；
10. 读库失败 ⇒ `empty`，**绝不返 0 条**。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ================================================================
# 夹具
# ================================================================
def _bars(n: int = 4):
    """假日线：日期递增、收盘递增，便于断言 spark 的顺序与最后值。"""
    dates = pd.date_range("2026-10-20", periods=n, freq="D")
    return pd.DataFrame({"date": dates, "close": [3680.0 + i for i in range(n)]})


@pytest.fixture
def stubs(monkeypatch):
    """把三个外呼源换成可编程的假函数。"""
    import data.index_bars as IB
    import data.realtime as RT
    import data.us_market as UM

    state = {
        "quotes": {},
        "bars": lambda symbol: _bars(),
        "us": {"unavailable": True, "note": "假源不可用"},
        "offline": False,
    }
    monkeypatch.setattr(IB, "get_index_bars", lambda s: state["bars"](s))
    monkeypatch.setattr(RT, "get_index_quotes", lambda force_refresh=False: state["quotes"])
    monkeypatch.setattr(RT, "get_realtime_quotes",
                        lambda codes, force_refresh=False: {
                            c: state["quotes"][c] for c in codes if c in state["quotes"]})
    monkeypatch.setattr(UM, "get_overnight_us", lambda force=False: state["us"])
    import data.offline_gate as G
    monkeypatch.setattr(G, "is_offline", lambda: state["offline"])
    return state


@pytest.fixture
def client():
    from web.server.app import app
    import web.server.cache as C
    C.clear()
    return TestClient(app)


@pytest.fixture
def db(tmp_path, monkeypatch):
    """独立空库（从模板复制）。

    ⚠️ `use_template_db` 是 conftest 里的**普通函数**而不是 fixture
    （它要显式接收 tmp_path/monkeypatch），所以这里包一层。
    不包就会直接写真实 `data/portfolio.db` —— 模板库有 0 条快讯，
    插进去的就是真库，会被生产库污染哨兵抓。
    """
    from tests.conftest import use_template_db
    return use_template_db(tmp_path, monkeypatch)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """临时数据根（picks / 心跳都指向它）。"""
    import core.scheduler_health as SH
    import web.server.cache as C
    C.clear()
    (tmp_path / "output" / "daily_picks").mkdir(parents=True)
    (tmp_path / "temp").mkdir()
    monkeypatch.setenv("BIGA_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(SH, "HEARTBEAT_PATH",
                        str(tmp_path / "temp" / "scheduler_heartbeat.json"))
    return tmp_path


def _write_latest(root, date="2026-10-23"):
    import json
    payload = {"date": date, "generated_at": date + " 15:52:00", "mode": "rule",
               "picks": [], "market_signal": {"total_score": 28.4,
                                              "market_state": "偏弱震荡",
                                              "position_limit": 0.2},
               "stage_timings": {}, "errors": []}
    (root / "output" / "daily_picks" / "latest.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _quote(price=3700.0, change=10.0, change_pct=0.27, **kw):
    """假的行情行。

    ⚠️ **不要在这里给 `code` / `name` 默认值**：`watch._watchlist` 优先用
    行情返回的 code，默认值会把自选里的两只都改写成同一个代码
    （第一版就是这么错的：rows 里 600519 变成了 000001）。
    """
    row = {"price": price, "change": change, "change_pct": change_pct,
           "quote_time": "15:00:00", "source": "tencent", "stale": False,
           "data_lagging": False, "amount_wan": 52_000_000.0}
    row.update(kw)
    return row


# ================================================================
# 看盘 · 指数
# ================================================================
class TestWatchIndices:
    def test_live_quote_preferred(self, stubs, client):
        stubs["quotes"] = {"sh000001": _quote(3700.0)}
        body = client.get("/api/watch/indices").json()
        assert body["ok"] is True
        row = next(r for r in body["data"]["indices"] if r["symbol"] == "sh000001")
        assert row["price"] == 3700.0
        assert row["basis"] == "live"
        assert row["spark"] == [3680.0, 3681.0, 3682.0, 3683.0]
        assert row["local_last_date"] == "2026-10-23"

    def test_falls_back_to_local_close(self, stubs, client):
        """报价拿不到 ⇒ 用本地日线最后收盘，并**标明来源**（不假装实时）。"""
        stubs["quotes"] = {}
        body = client.get("/api/watch/indices").json()
        row = next(r for r in body["data"]["indices"] if r["symbol"] == "sh000001")
        assert row["price"] == 3683.0
        assert row["basis"] == "close"
        assert row["change_pct"] is None      # 没有就没有，不猜 0

    def test_no_local_no_price(self, stubs, client):
        """本地也没有 ⇒ None。绝不返 0（0 会被前端读成"跌到零"）。"""
        stubs["quotes"] = {}
        stubs["bars"] = lambda s: None
        body = client.get("/api/watch/indices").json()
        for row in body["data"]["indices"]:
            assert row["price"] is None
            assert row["basis"] is None
            assert row["spark"] == []

    def test_offline_skips_network(self, stubs, client):
        stubs["offline"] = True
        stubs["quotes"] = {"sh000001": _quote(9999.0)}   # 有值也不该被用
        body = client.get("/api/watch/indices").json()
        assert body["data"]["live"] is False
        assert "离线" in body["data"]["note"]
        row = next(r for r in body["data"]["indices"] if r["symbol"] == "sh000001")
        assert row["basis"] == "close"

    def test_spark_truncated_to_window(self, stubs, client):
        """20 根上限：给 40 根也只回 20，且是**最后** 20 根（不是最早）。"""
        stubs["bars"] = lambda s: pd.DataFrame({
            "date": pd.date_range("2026-08-01", periods=40, freq="D"),
            "close": [float(i) for i in range(40)]})
        body = client.get("/api/watch/indices").json()
        row = body["data"]["indices"][0]
        assert len(row["spark"]) == 20
        assert row["spark"][-1] == 39.0

    def test_stale_flag_passthrough(self, stubs, client):
        stubs["quotes"] = {"sh000001": _quote(stale=True)}
        body = client.get("/api/watch/indices").json()
        row = next(r for r in body["data"]["indices"] if r["symbol"] == "sh000001")
        assert row["stale"] is True
        assert "过期" in body["data"]["note"]


# ================================================================
# 看盘 · 自选
# ================================================================
class TestWatchList:
    @pytest.fixture(autouse=True)
    def _watchlist(self, tmp_path, monkeypatch):
        """替换**严格快照**（页面端点的唯一取数入口）。

        ⚠️ 2026-10-27：自选股落 SQLite 后，页面不再走 `load_watchlist`（那个会
        吞异常），改为 `snapshot()`。所以这里 patch 的是 snapshot —— 继续
        patch 旧名字会让这些用例静默读到模板库，测试照样"通过"但测的不是页面。
        """
        import data.watchlist as W
        stocks = [{"code": "600519", "name": "贵州茅台", "note": ""},
                  {"code": "000001", "name": "平安银行", "note": "银行"}]
        monkeypatch.setattr(W, "snapshot", lambda owner="user": {
            "owner": owner, "label": W.OWNER_LABELS.get(owner, owner),
            "stocks": list(stocks), "counts": {"user": 2, "ashen": 0}})
        return W

    def test_missing_quote_not_zeroed(self, stubs, client):
        stubs["quotes"] = {"600519": _quote(1486.2, 18.1, 1.24)}
        body = client.get("/api/watch/watchlist").json()
        assert body["ok"] is True
        rows = {r["code"]: r for r in body["data"]["rows"]}
        assert rows["600519"]["available"] is True
        assert rows["600519"]["name"] == "贵州茅台"      # 自选里的名字优先
        assert rows["000001"]["available"] is False
        assert rows["000001"]["price"] is None
        assert body["data"]["available"] == 1

    def test_stale_collected(self, stubs, client):
        stubs["quotes"] = {"600519": _quote(stale=True)}
        body = client.get("/api/watch/watchlist").json()
        assert body["data"]["stale_codes"] == ["600519"]

    def test_empty_watchlist_is_ok(self, stubs, client, monkeypatch):
        import data.watchlist as W
        monkeypatch.setattr(W, "snapshot", lambda owner="user": {
            "owner": owner, "label": W.OWNER_LABELS.get(owner, owner),
            "stocks": [], "counts": {"user": 0, "ashen": 0}})
        body = client.get("/api/watch/watchlist").json()
        assert body["ok"] is True
        assert body["data"]["total"] == 0

    def test_owner_scope_does_not_share_cache(self, stubs, client, monkeypatch):
        """两轨**必须各自缓存**。

        缓存键不带 owner 的话，先看"我的"再切"阿深"会命中同一份缓存 ——
        页面显示的价格全是对的，只是根本不是那套清单，没有任何异常可循。
        """
        from web.server import cache
        cache.clear()
        import data.watchlist as W

        def snap(owner="user"):
            rows = ([{"code": "600519", "name": "贵州茅台", "note": "我的备注"}]
                    if owner == "user"
                    else [{"code": "300750", "name": "宁德时代", "note": "阿深的理由"}])
            return {"owner": owner, "label": W.OWNER_LABELS.get(owner, owner),
                    "stocks": rows, "counts": {"user": 1, "ashen": 1}}

        monkeypatch.setattr(W, "snapshot", snap)
        a = client.get("/api/watch/watchlist?owner=user").json()["data"]
        b = client.get("/api/watch/watchlist?owner=ashen").json()["data"]
        assert [r["code"] for r in a["rows"]] == ["600519"]
        assert [r["code"] for r in b["rows"]] == ["300750"], "阿深那栏拿到了我的缓存"
        assert b["owner"] == "ashen" and b["owner_label"] == "阿深自选"
        assert b["rows"][0]["note"] == "阿深的理由"
        assert a["counts"] == {"user": 1, "ashen": 1}

    def test_invalid_owner_falls_back_to_user(self, stubs, client):
        """非法 owner 回退 user（而不是 500）：owner 只该来自白名单，
        但真出现时页面要还能用，错误靠日志暴露。"""
        body = client.get("/api/watch/watchlist?owner=root").json()
        assert body["ok"] is True
        assert body["data"]["owner"] == "user"


# ================================================================
# 看盘 · 行业资金
# ================================================================
class TestWatchIndustry:
    def test_empty_snapshot_is_missing(self, client, monkeypatch):
        import data.fund_flow as F
        monkeypatch.setattr(F, "get_stored_industry_flow", lambda: (None, None))
        body = client.get("/api/watch/industry").json()
        assert body["ok"] is False
        assert body["error"]["type"] == "missing"

    def test_top_in_out_and_nan(self, client, monkeypatch):
        import data.fund_flow as F
        df = pd.DataFrame({
            "行业": [f"行业{i}" for i in range(20)],
            "净额": [float(20 - i) for i in range(20)],
            "涨跌幅": [None] + [1.0] * 19,          # 含 NaN，必须变 None 不是 0
            "领涨股": ["甲"] * 20,
        })
        monkeypatch.setattr(F, "get_stored_industry_flow", lambda: ("2026-10-23", df))
        body = client.get("/api/watch/industry").json()
        assert body["ok"] is True
        d = body["data"]
        assert d["data_date"] == "2026-10-23"
        assert len(d["top_in"]) == 8
        assert len(d["top_out"]) == 8
        assert d["top_in"][0]["industry"] == "行业0"
        assert d["top_out"][0]["industry"] == "行业19"   # 流出榜从最负开始
        nan_row = next(r for r in d["top_in"] if r["industry"] == "行业0")
        assert nan_row["change_pct"] is None


# ================================================================
# 看盘 · 外盘
# ================================================================
class TestWatchUS:
    def test_unavailable_maps_to_proxy(self, stubs, client):
        stubs["us"] = {"unavailable": True, "note": "上游超时"}
        body = client.get("/api/watch/us").json()
        assert body["ok"] is False
        assert body["error"]["type"] == "proxy"
        assert "上游超时" in body["error"]["detail"]

    def test_rows_passthrough(self, stubs, client):
        stubs["us"] = {"data_date": "2026-10-24", "days_ago": 1, "level": "neutral",
                       "level_text": "无异动", "note": "",
                       "indexes": [{"name": "道琼斯", "close": 42100.0,
                                    "change_pct": 0.31, "level": "neutral",
                                    "level_text": "无异动"}]}
        body = client.get("/api/watch/us").json()
        assert body["ok"] is True
        assert body["data"]["rows"][0]["name"] == "道琼斯"
        assert body["as_of"] == "2026-10-24"


# ================================================================
# 看盘 · 异动
# ================================================================
class TestWatchAlerts:
    def test_empty_is_normal(self, client, db):
        """非交易日没有异动是常态，不该报 error（否则页面天天红一片）。"""
        body = client.get("/api/watch/alerts").json()
        assert body["ok"] is True
        assert body["data"]["rows"] == []
        assert body["data"]["total"] == 0


# ================================================================
# 情报 · 快讯流
# ================================================================
def _seed(db_path: str, rows: list[tuple]) -> None:
    """插测试快讯。行格式：(title, sentiment, content)。"""
    now = datetime.now()
    con = sqlite3.connect(db_path)
    try:
        for i, (title, sentiment, content) in enumerate(rows):
            pub = (now - timedelta(minutes=10 + i)).strftime("%Y-%m-%d %H:%M:%S")
            con.execute(
                "INSERT INTO news_items (hash, title, content, source, published_at,"
                " related_codes, sentiment, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (f"seed{i}", title, content, "ths", pub, "[]", sentiment, pub))
        con.commit()
    finally:
        con.close()


class TestIntelNews:
    def test_two_layer_counts(self, client, db):
        """window = 时间窗全量；filtered = 当前筛选。两者**都要给**。"""
        _seed(db, [
            ("半导体设备订单爆发", "positive", "国产替代加速"),
            ("某公司净利同比下降", "negative", "由盈转亏"),
            ("行业会议召开", "neutral", "例行会议"),
        ])
        body = client.get("/api/intel/news?hours=24").json()
        assert body["ok"] is True
        d = body["data"]
        assert d["window"]["total"] == 3
        assert d["window"]["by_sentiment"] == {"positive": 1, "negative": 1, "neutral": 1}
        assert d["filtered"]["total"] == 3
        assert d["returned"] == 3

    def test_industry_filter_in_sql(self, client, db):
        """行业筛选在 SQL 侧生效：filtered.total 只算命中的，window 仍是全量。"""
        _seed(db, [
            ("半导体设备订单爆发", "positive", "国产替代加速"),
            ("银行板块获增持", "positive", "央行降准"),
            ("行业会议召开", "neutral", "例行会议"),
        ])
        body = client.get("/api/intel/news?hours=24&industry=科技").json()
        d = body["data"]
        assert d["window"]["total"] == 3          # 全量不变
        assert d["filtered"]["total"] == 1        # 只有半导体那条
        assert d["items"][0]["sectors"] == ["科技"]

    def test_sentiment_filter(self, client, db):
        _seed(db, [
            ("A 利好", "positive", "增长"),
            ("B 利空", "negative", "亏损"),
        ])
        d = client.get("/api/intel/news?hours=24&sentiment=positive").json()["data"]
        assert d["filtered"]["total"] == 1
        assert d["window"]["total"] == 2
        assert d["filtered"]["by_sentiment"]["negative"] == 0

    def test_unknown_industry_rejected(self, client, db):
        """拼错行业名要报错——静默返回全量会被当成"这个行业没新闻"。"""
        body = client.get("/api/intel/news?industry=芯片").json()
        assert body["ok"] is False
        assert body["error"]["type"] == "invalid"

    def test_unknown_sentiment_rejected(self, client, db):
        body = client.get("/api/intel/news?sentiment=利好").json()
        assert body["ok"] is False
        assert body["error"]["type"] == "invalid"

    def test_db_failure_is_empty_not_zero(self, client, db, monkeypatch):
        from db import store
        def boom(*a, **kw):
            raise sqlite3.OperationalError("库被锁")
        monkeypatch.setattr(store, "get_connection", boom)
        body = client.get("/api/intel/news?hours=24").json()
        assert body["ok"] is False
        assert body["error"]["type"] == "empty"

    def test_limit_clamped(self, client, db):
        _seed(db, [(f"标题{i}", "neutral", "正文") for i in range(5)])
        d = client.get("/api/intel/news?hours=24&limit=99999").json()["data"]
        assert d["returned"] == 5          # 不超过库里条数，也不报错
        assert d["window"]["hours"] == 24

    def test_preview_truncated(self, client, db):
        _seed(db, [("长文", "neutral", "字" * 400)])
        d = client.get("/api/intel/news?hours=24").json()["data"]
        assert len(d["items"][0]["preview"]) <= 141
        assert d["items"][0]["preview"].endswith("…")

    def test_empty_title_uses_content_head(self, client, db):
        """新浪源标题是空串（实测 100+ 条）——不能渲染成空白行。

        用正文起头当标题，预览从切掉的位置继续，**同一句话不重复两遍**。
        """
        _seed(db, [("", "positive", "央行今日开展逆回购操作" + "细节" * 60)])
        it = client.get("/api/intel/news?hours=24").json()["data"]["items"][0]
        assert it["title"].startswith("央行今日开展逆回购操作")
        assert it["title"].endswith("…")
        assert not it["preview"].startswith("央行今日")   # 不重复
        assert it["sectors"] == ["金融"]

    def test_no_title_no_content(self, client, db):
        """标题正文都空 ⇒ 明确占位，不是空字符串（空字符串在页面上是隐形 bug）。"""
        _seed(db, [("", "neutral", "")])
        it = client.get("/api/intel/news?hours=24").json()["data"]["items"][0]
        assert it["title"] == "（无标题）"


# ================================================================
# 首页看盘分区
# ================================================================
class TestDashboardWatchZone:
    def test_zone_has_indices_and_amount(self, stubs, client, root):
        _write_latest(root)
        stubs["quotes"] = {"sh000001": _quote(3700.0, amount_wan=52_000_000.0)}
        body = client.get("/api/dashboard").json()
        w = body["data"]["watch"]
        assert w["ok"] is True
        assert w["data"]["amount_scope"] == "沪市"
        assert w["data"]["amount_yi"] == 5200.0     # 5200 万万元 = 5200 亿
        assert "spark" not in w["data"]["indices"][0]   # 首屏不转发 spark

    def test_zone_failure_keeps_top_ok(self, stubs, client, root):
        """watch 挂了，picks 仍要能显示（C2 分区包络）。"""
        _write_latest(root)
        stubs["bars"] = lambda s: None
        stubs["quotes"] = {}
        body = client.get("/api/dashboard").json()
        assert body["ok"] is True
        assert body["data"]["picks"]["ok"] is True
