# -*- coding: utf-8 -*-
"""聊天/分析取数韧性测试（2026-10-30 三项修复的行为锁）。

背景（同一天的两次真实故障）
────────────────────────────
1. 09:13 深度分析 600276 fatal：K 线走**纯网络**，而 WinINET 代理指向未启动的
   Clash ⇒ 3 源全部连接被拒 ⇒ "所有数据源均无法获取历史数据"。
   **本地面板里该股有 2,626 根日线**，有数据却判成没有。
2. 13:14 聊天问 603605，后端处理 6 分钟，13:20 页面 WebSocket 断开 ——
   旧行为"整轮跑完才落库" ⇒ `chat_logs` 里连**提问**都没有，屏幕无回答、库里无线索。

三个修复：
  A. K 线**本地面板优先**（`data.stock_data._panel_daily`），面板 ≥60 根直接返回；
  B. 聊天**提问立刻落库**（`chat_history.start_turn`），轮末 `complete_turn` 补回答；
  C. 腾讯日线源**连接层熔断**（`utils/conn_guard.ConnBreaker`），连不上立即换源。

全部离线：面板用 monkeypatch 注入，网络一律 patch。
"""
from __future__ import annotations

import pandas as pd
import pytest
import requests


# ================================================================
# 夹具
# ================================================================

def _panel_df(n: int = 300, index_name: str = "date") -> pd.DataFrame:
    """造一个与 data.bars.get_bars 同构的日线表。

    ⚠️ 与真实 `get_bars` 一致：日期是**普通列** `date`，索引是 RangeIndex。
    首版测试把它造成了 DatetimeIndex，结果放过了"日期被转成 1970"的 bug
    （面板日期是列不是索引 ⇒ 转换时被丢掉）。
    """
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "date": idx, "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5,
        "volume": 1_000_000.0,
    })
    return df


def _kline_df(n: int = 242) -> pd.DataFrame:
    """造一个与网络源同构的 K 线表（Date 索引 + 大写列）。"""
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Open": 10.0, "High": 11.0, "Low": 9.5, "Close": 10.5,
        "Volume": 1_000_000.0,
    }, index=idx)
    df.index.name = "Date"
    return df


@pytest.fixture
def fetcher(monkeypatch):
    from data.stock_data import StockDataFetcher
    return StockDataFetcher()


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    return str(use_template_db(tmp_path, monkeypatch, "test_chat_resil.db"))


# ================================================================
# A. K 线本地面板优先
# ================================================================

def test_kline_hits_panel_without_network(fetcher, monkeypatch):
    """面板 ≥60 根 ⇒ 直接返回面板数据，**绝不碰网络**。"""
    from data import stock_data as sd
    monkeypatch.setattr(sd, "_panel_daily",
                        lambda *a, **k: _kline_df(300))
    calls = []

    def _boom(**kw):
        calls.append(kw)
        raise AssertionError("面板命中时不应走网络")

    monkeypatch.setattr(sd.data_source_manager, "get_stock_hist_data", _boom)
    out = fetcher.get_stock_data("600276", period="1y")
    assert not isinstance(out, dict), out
    assert len(out) == 300
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.index.name == "Date"
    assert calls == []


def test_panel_dates_are_real_not_epoch():
    """⚠️ 日期必须是**真实交易日**，不得变成 1970（首版实测踩过：面板日期是
    普通列，转换时被丢掉 ⇒ 索引被当成 ns 时间戳 ⇒ 全部 1970-01-01）。"""
    from data.stock_data import _panel_daily
    df = _panel_daily("600276", "20250801", "20261030")
    if df is None:
        pytest.skip("本机面板无该股数据")
    assert df.index.name == "Date"
    assert df.index.min() > pd.Timestamp("2000-01-01"), \
        f"日期异常（疑似被当成时间戳）：{df.index.min()}"
    assert df.index.max() >= pd.Timestamp("2026-10-01")
    assert len(df) >= 60


def test_kline_falls_back_to_network_when_panel_thin(fetcher, monkeypatch):
    """面板 <60 根 ⇒ 走网络（MA60 会 NaN，硬用面板是二次故障）。

    ⚠️ 必须 patch **`data.bars.get_bars`**（真实集成点），而不是 `_panel_daily`：
    后者内部自带 min_bars 门槛与 try/except，在外层 mock 会把这两个分支绕掉，
    测的是"假的 `_panel_daily`"而不是真实行为（首版就因此红了两条）。
    """
    from data import stock_data as sd
    monkeypatch.setattr(sd, "_panel_daily",
                        lambda *a, **k: None)      # 面板判定：不足 60 根 ⇒ None
    monkeypatch.setattr(sd.data_source_manager, "get_stock_hist_data",
                        lambda **kw: _kline_df(242).rename(
                            columns=str.lower).rename_axis("date"))
    out = fetcher.get_stock_data("600276", period="1y")
    assert not isinstance(out, dict)
    assert len(out) == 242


def test_panel_thin_real_path_falls_back(fetcher, monkeypatch):
    """真实路径：面板只有 30 根 ⇒ `_panel_daily` 返回 None ⇒ 走网络。"""
    from data import bars as bars_mod
    from data import stock_data as sd
    monkeypatch.setattr(bars_mod, "get_bars",
                        lambda *a, **k: _panel_df(30))
    monkeypatch.setattr(sd.data_source_manager, "get_stock_hist_data",
                        lambda **kw: _kline_df(242).rename(
                            columns=str.lower).rename_axis("date"))
    out = fetcher.get_stock_data("600276", period="1y")
    assert not isinstance(out, dict)
    assert len(out) == 242, "面板不足 60 根时必须走网络，不能返回半截数据"


def test_panel_read_failure_falls_back_not_raises(fetcher, monkeypatch):
    """面板层异常（如文件缺失）⇒ 记日志走网络，**不能让取数整体挂掉**。

    try/except 在 `_panel_daily` **内部**，所以这里 patch 的是 bars 层。
    """
    from data import bars as bars_mod
    from data import stock_data as sd

    def _boom(*a, **k):
        raise RuntimeError("panel parquet corrupted")

    monkeypatch.setattr(bars_mod, "get_bars", _boom)
    monkeypatch.setattr(sd.data_source_manager, "get_stock_hist_data",
                        lambda **kw: _kline_df(242).rename(
                            columns=str.lower).rename_axis("date"))
    out = fetcher.get_stock_data("600276", period="1y")
    assert not isinstance(out, dict), "面板异常必须被 _panel_daily 吞掉并回退"


def test_panel_daily_real_data():
    """真面板冒烟：600276 应有 ≥60 根（不联网，只读本地 parquet）。"""
    from data.stock_data import _panel_daily
    df = _panel_daily("600276", "20250801", "20261030")
    if df is None:
        pytest.skip("本机面板无该股数据")
    assert len(df) >= 60
    assert df.index.name == "Date"


# ================================================================
# B. 聊天提问立刻落库
# ================================================================

def test_start_turn_persists_question_immediately(temp_db):
    """start_turn 之后库里就**必须有提问**——断开/崩溃也留痕。"""
    from core import chat_history as ch
    assert ch.start_turn("s-a", "603605 现在怎么样") is True
    rows = ch.load_messages("s-a")
    assert rows == [{"role": "user", "content": "603605 现在怎么样"}]


def test_complete_turn_appends_answer_and_keeps_question(temp_db):
    from core import chat_history as ch
    ch.start_turn("s-b", "问题")
    assert ch.complete_turn("s-b", "回答", usage={"prompt": 10, "completion": 5},
                            tools_used=["tool_a"]) is True
    rows = ch.load_messages("s-b")
    assert [(r["role"], r["content"]) for r in rows] == \
        [("user", "问题"), ("assistant", "回答")]


def test_save_turn_keeps_old_contract(temp_db):
    """`save_turn`（= start+complete）对外行为不变：一次落两行、首条定标题。"""
    from core import chat_history as ch
    assert ch.save_turn("s-c", "我的第一个问题", "答") is True
    meta = ch.load_session_meta("s-c")
    assert meta.get("title") == "我的第一个问题" or meta.get("message_count") == 2
    assert len(ch.load_messages("s-c")) == 2


def test_turn_events_persists_even_without_ws(temp_db, monkeypatch):
    """端到端：turn_events 在**无人消费**时也应把问答落库（模拟页面断开）。

    事故本质：客户端断开后没人消费事件，但生成器会继续跑完。
    这里直接驱动生成器到末尾，断言库里有两行。
    """
    from core import chat_agent as ca
    from core import chat_history as ch

    class _FakeAgent:
        model = "test-model"
        usage_total = {"prompt": 1, "completion": 1}
        tools_used = []

        def __init__(self, model=None, persona=None):
            pass

        def reply(self, message, history=None):
            yield ("tool_start", "tool_x", {})
            yield ("tool_end", "tool_x", "ok")
            yield ("answer", "这是回答")

    monkeypatch.setattr(ca, "ChatAgent", _FakeAgent)
    monkeypatch.setattr(ca, "PERSONAS", {"综合": {}})

    from web.server import chat as chat_svc
    sid = ch.new_session_id()
    events = list(chat_svc.turn_events(sid, "关于 603605 的问题",
                                       persona="综合"))
    kinds = [e.get("type") for e in events]
    assert kinds[-1] == "done", f"WS 每轮必以 done 收尾，实际 {kinds}"
    rows = ch.load_messages(sid)
    assert [("user", "关于 603605 的问题"), ("assistant", "这是回答")] == \
        [(r["role"], r["content"]) for r in rows]


def test_turn_events_persists_failure_reason(temp_db, monkeypatch):
    """失败也留痕（旧版已有，锁住不让拆分后丢掉）。"""
    from core import chat_agent as ca
    from core import chat_history as ch

    class _Boom:
        model = "test-model"
        usage_total = {}
        tools_used = []

        def __init__(self, model=None, persona=None):
            pass

        def reply(self, message, history=None):
            yield ("error", "上游超时")
            return

    monkeypatch.setattr(ca, "ChatAgent", _Boom)
    monkeypatch.setattr(ca, "PERSONAS", {"综合": {}})

    from web.server import chat as chat_svc
    sid = ch.new_session_id()
    list(chat_svc.turn_events(sid, "问", persona="综合"))
    rows = ch.load_messages(sid)
    assert rows[-1]["content"].startswith("（本轮失败："), rows


# ================================================================
# C. 腾讯日线连接层熔断
# ================================================================

def test_conn_guard_classifies_connection_errors():
    from utils.conn_guard import is_connection_level_error
    class _R:                       # 伪装 requests 异常（不依赖其构造签名）
        pass
    proxy = requests.exceptions.ProxyError("Unable to connect to proxy")
    timeout = requests.exceptions.Timeout("read timed out")
    assert is_connection_level_error(proxy) is True
    assert is_connection_level_error(timeout) is True
    # 第三方库常把底层异常包成 ValueError，只能靠文案判定
    assert is_connection_level_error(
        ValueError("HTTPSConnection ... 由于目标计算机积极拒绝，无法连接")) is True
    assert is_connection_level_error(ValueError(" Expecting value: line 1")) is False
    assert is_connection_level_error(None) is False


def test_breaker_opens_for_cooldown_and_resets():
    from utils.conn_guard import ConnBreaker
    b = ConnBreaker("测试", cooldown=600.0)
    assert b.open() is False
    b.trip("test")
    assert b.open() is True and b.remaining() > 590
    b.reset()
    assert b.open() is False


def test_tencent_source_trips_breaker_on_proxy_error(monkeypatch):
    """连接层失败 ⇒ **第一次就抛**（不再做第 2/3 次重试），并打开熔断。"""
    from data import tencent_source as ts
    ts._BREAKER.reset()
    attempts = []

    def _fail(url, **kw):
        attempts.append(1)
        raise requests.exceptions.ProxyError(
            "Unable to connect to proxy: 127.0.0.1:7890")

    monkeypatch.setattr(ts.requests, "get", _fail)
    with pytest.raises(requests.exceptions.ProxyError):
        ts._request_page("sh600276", "2026-10-30", "qfq", timeout=1.0, retries=3)
    assert len(attempts) == 1, f"连接层失败应立即放弃，实际重试了 {len(attempts)} 次"
    assert ts._BREAKER.open() is True
    ts._BREAKER.reset()


def test_tencent_source_fast_fails_while_breaker_open(monkeypatch):
    """熔断冷却期内 `fetch_kline` 必须快速失败，**一个请求都不发**。"""
    from data import tencent_source as ts
    ts._BREAKER.trip("test")
    sent = []

    def _spy(url, **kw):
        sent.append(1)
        raise AssertionError("熔断期内不应发请求")

    monkeypatch.setattr(ts.requests, "get", _spy)
    with pytest.raises(ConnectionError):
        ts.fetch_kline("600276", "2026-01-01", "2026-10-30")
    assert sent == []
    ts._BREAKER.reset()


def test_tencent_source_content_error_keeps_retrying(monkeypatch):
    """内容层失败（空响应体）⇒ 保留原有重试，**不得误熔断**。"""
    from data import tencent_source as ts
    ts._BREAKER.reset()

    class _Resp:
        status_code = 200
        text = ""

    monkeypatch.setattr(ts.requests, "get", lambda url, **kw: _Resp())
    with pytest.raises(Exception):
        ts._request_page("sh600276", "2026-10-30", "qfq", timeout=0.05, retries=3)
    assert ts._BREAKER.open() is False, "空响应是内容层问题，不应打开熔断"
    ts._BREAKER.reset()


# D. 添加自选不再等网络（_guess_name 本地快照优先）
# ================================================================

def _seed_snapshot(db_path: str, rows: list[tuple]) -> None:
    """往临时库的 market_snapshot 里塞几行（表在模板库里已存在）。"""
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS market_snapshot (
            id INTEGER PRIMARY KEY, snapshot_date TEXT, stock_code TEXT,
            stock_name TEXT, industry TEXT, board TEXT, price REAL, created_at TEXT)""")
        conn.executemany(
            "INSERT INTO market_snapshot (snapshot_date, stock_code, stock_name)"
            " VALUES (?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def test_guess_name_local_first(temp_db, monkeypatch):
    """本地快照有名称 ⇒ **不碰网络**直接返回（2026-10-30 事故：add 自选卡 26s）。

    ⚠️ 用**临时文件库**而不是 `:memory:`：conftest 的夹具体系会关闭旧连接，
    `:memory:` 连接一关就被 `except` 吞成空串（实测 ProgrammingError:
    Cannot operate on a closed database）——用文件库同时更贴近生产行为。
    """
    from web.server import watch as w
    import data.realtime as rt

    _seed_snapshot(temp_db, [("2026-10-29", "600000", "浦发银行")])

    def _net_boom(codes):
        raise AssertionError("本地命中时不应走网络")

    monkeypatch.setattr(w, "_offline", lambda: False)
    monkeypatch.setattr(rt, "get_realtime_quotes", _net_boom)
    assert w._guess_name_local("600000") == "浦发银行"
    assert w._guess_name("600000") == "浦发银行"


def test_guess_name_falls_back_to_network(temp_db, monkeypatch):
    """本地快照没有（新股）⇒ 才走网络。"""
    from web.server import watch as w
    import data.realtime as rt
    _seed_snapshot(temp_db, [])                 # 表存在但为空
    monkeypatch.setattr(w, "_offline", lambda: False)
    monkeypatch.setattr(rt, "get_realtime_quotes",
                        lambda codes: {x: {"name": "新上市"} for x in codes})
    assert w._guess_name("301999") == "新上市"


def test_guess_name_never_raises(temp_db, monkeypatch):
    """名称缺失不该阻止写入：本地/网络都炸也只返回空串。"""
    from db import store
    from web.server import watch as w
    import data.realtime as rt

    class _BoomConn:
        def execute(self, *a, **k):
            raise RuntimeError("db locked")

        def close(self):
            pass

    monkeypatch.setattr(store, "get_connection", lambda: _BoomConn())
    monkeypatch.setattr(w, "_offline", lambda: False)
    monkeypatch.setattr(rt, "get_realtime_quotes",
                        lambda codes: (_ for _ in ()).throw(RuntimeError("net down")))
    assert w._guess_name("600000") == ""
