# -*- coding: utf-8 -*-
"""B2② 本地化模块：`data/index_bars.py`（指数日线）与 `core/zt_calendar.py`
（涨停日历/市场宽度）。

锁住的契约：
  1. **取数层不联网**：`get_index_bars` 本地没有 ⇒ None（缺就是缺）；
  2. **回填拒绝离线**：`backfill_index_daily` 在 `BIGA_OFFLINE=1` 下抛 RuntimeError；
  3. **涨停判定分档**：ST 5% / 创业板·科创板 20% / 北交所 30% / 其余 10%，
     涨停价 = `round(preclose × (1+limit), 2)`（四舍五入到分）；
  4. **晋级率口径**：T 日涨停且 T-1 也涨停的占比（与东财池「连板数≥2」同义）。
"""
import json

import pandas as pd
import pytest

import data.index_bars as IB
import core.zt_calendar as ZC


# ================================================================
# 1. index_bars
# ================================================================

@pytest.fixture
def index_env(tmp_path, monkeypatch):
    monkeypatch.setattr(IB, "INDEX_DIR", str(tmp_path / "index"))
    IB.get_index_bars.cache_clear()
    yield tmp_path
    IB.get_index_bars.cache_clear()


def _write_index(tmp_path, symbol: str, days: int = 40) -> None:
    import os
    os.makedirs(os.path.join(str(tmp_path), "index"), exist_ok=True)
    df = pd.DataFrame({
        "date": pd.date_range("2026-08-01", periods=days, freq="B").strftime("%Y-%m-%d"),
        "open": 4000.0, "high": 4010.0, "low": 3990.0,
        "close": 4000.0 + pd.Series(range(days)) * 0.5,
        "volume": 1e9,
    })
    df.to_parquet(os.path.join(str(tmp_path), "index", f"{symbol}.parquet"),
                  index=False)


class TestIndexBars:
    def test_get_returns_none_when_missing(self, index_env):
        assert IB.get_index_bars("sh000001") is None

    def test_get_reads_local_and_carries_attrs(self, index_env):
        _write_index(index_env, "sh000300")
        df = IB.get_index_bars("sh000300")
        assert df is not None and len(df) == 40
        assert df.attrs["source"] == "index_panel"
        assert df.attrs["last_date"].startswith("2026-09")

    def test_min_rows_gate(self, index_env):
        _write_index(index_env, "sh000300", days=10)
        assert IB.get_index_bars("sh000300", min_rows=21) is None

    def test_backfill_refuses_offline(self, index_env, monkeypatch):
        monkeypatch.setenv("BIGA_OFFLINE", "1")
        with pytest.raises(RuntimeError, match="BIGA_OFFLINE"):
            IB.backfill_index_daily(symbols=("sh000001",))


# ================================================================
# 1b. 指数新鲜度（P1-2，2026-10-19 审查）
# ================================================================
#
# 事故：原实现拿交易日历的**最后一个元素**当基准
#   `days = trade_days(); if last >= days[-1]: return True`
# 而 `trade_days()` 只覆盖到"上次回填写入的日期"（实测日历末 10-16、
# 数据已到 10-19）⇒ 恒真，闸门等于没装；反过来"数据停在 10-16、日历末也是
# 10-16"同样恒真 ⇒ **陈旧永远判不出来**。这个闸门已经接进选股判定
# （市场检测），所以必须用"最近应收盘的交易日"当基准。

class TestIndexFreshness:
    def _env(self, monkeypatch, last_date, expected):
        """构造：本地指数末根 = last_date，期望收盘日 = expected。"""
        df = pd.DataFrame({"date": pd.to_datetime([last_date]),
                           "open": 1.0, "high": 1.0, "low": 1.0,
                           "close": 1.0, "volume": 1.0})
        df.attrs = {"source": "index_panel", "symbol": "sh000300",
                    "rows": 1, "last_date": last_date}
        # ⚠️⚠️ 必须 patch **读取入口**，否则 `index_freshness` 会去读真实的
        #   `data/panel/index/sh000300.parquet` —— 用例结果随"今天回填到哪一天"
        #   漂移。铁证：2026-10-26 把当日这根数据补进 parquet 后，本类三个用例
        #   立刻由绿转红（真实文件末根从 10-23 变成 10-26，而用例期望的是 10-19
        #   口径）—— 这不是代码回归，是测试耦合了真实数据。
        monkeypatch.setattr(IB, "get_index_bars",
                            lambda symbol, *a, **k: df)
        monkeypatch.setattr(IB, "expected_close_date", lambda now=None: expected)
        monkeypatch.setattr("data.bars.trade_days",
                            lambda: ["2026-10-13", "2026-10-14", "2026-10-15",
                                     "2026-10-16", "2026-10-19"])

    def test_up_to_date_is_fresh(self, monkeypatch):
        self._env(monkeypatch, "2026-10-19", "2026-10-19")
        fr = IB.index_freshness("sh000300")
        assert fr["fresh"] is True and fr["lag_days"] == 0

    def test_one_day_lag_tolerated(self, monkeypatch):
        """收盘后到回填前（tolerance=1）不应误杀。"""
        self._env(monkeypatch, "2026-10-16", "2026-10-19")
        assert IB.index_freshness("sh000300")["lag_days"] == 1
        assert IB.index_fresh_enough("sh000300") is True

    def test_two_day_lag_must_be_stale(self, monkeypatch):
        """⭐ 指数停在 T-2、期望 T ⇒ **必须**判不新鲜（审查原话）。"""
        self._env(monkeypatch, "2026-10-15", "2026-10-19")
        fr = IB.index_freshness("sh000300")
        assert fr["lag_days"] == 2
        assert fr["fresh"] is False
        assert "2026-10-15" in fr["reason"] and "2026-10-19" in fr["reason"]

    def test_missing_data_is_stale(self, monkeypatch):
        # 同上：不 patch 就会读到真实文件，"本地没有数据"这个前提根本不成立
        monkeypatch.setattr(IB, "get_index_bars", lambda symbol, *a, **k: None)
        fr = IB.index_freshness("sh000300")
        assert fr["fresh"] is False and fr["last_date"] is None

    def test_unknown_expected_is_stale_not_silently_fresh(self, monkeypatch):
        """日历不可用 ⇒ 不能假装新鲜（未知 ≠ 没问题）。"""
        self._env(monkeypatch, "2026-10-19", None)
        assert IB.index_freshness("sh000300")["fresh"] is False

    def test_expected_close_date_pre_market_uses_previous_day(self, monkeypatch):
        """盘前（未到 15:10）当天收盘还没出 ⇒ 期望日应为**上一交易日**。"""
        import datetime as dt
        # ⚠️ 必须只用**本地日历**（`data.bars.trade_days`）：若走
        # `data.trading_calendar`，缓存过期时会联网，离线自检会因"查日历"
        # 丢掉全部指数指标。
        monkeypatch.setattr("data.bars.trade_days",
                            lambda: ["2026-10-15", "2026-10-16", "2026-10-19"])
        assert IB.expected_close_date(dt.datetime(2026, 10, 19, 9, 25)) == "2026-10-16"
        assert IB.expected_close_date(dt.datetime(2026, 10, 19, 16, 0)) == "2026-10-19"

    def test_expected_close_date_weekend_uses_last_trading_day(self, monkeypatch):
        import datetime as dt
        monkeypatch.setattr("data.bars.trade_days",
                            lambda: ["2026-10-15", "2026-10-16"])
        assert IB.expected_close_date(dt.datetime(2026, 10, 18, 10, 0)) == "2026-10-16"

    def test_expected_close_date_does_not_touch_trading_calendar(self, monkeypatch):
        """离线自检的硬约束：查期望日**不能**触发联网日历刷新。"""
        import datetime as dt

        def _boom(*a, **k):
            raise AssertionError("不许走 data.trading_calendar（可能联网）")

        monkeypatch.setattr("data.trading_calendar.refresh", _boom, raising=False)
        monkeypatch.setattr("data.bars.trade_days",
                            lambda: ["2026-10-16", "2026-10-19"])
        assert IB.expected_close_date(dt.datetime(2026, 10, 19, 16, 0)) == "2026-10-19"


# ================================================================
# 2. zt_calendar
# ================================================================

@pytest.fixture
def zt_env(tmp_path, monkeypatch):
    """隔离的 parts / overlay 目录 + 缓存路径。

    ⚠️ `PANEL_OVERLAY_DIR` 也必须隔离：它默认指向真实 `data/panel/daily_overlay`，
    不 patch 的话测试会读到当天真实行情（既不确定，又可能让"无数据"用例变成有数据）。
    """
    parts = tmp_path / "parts"
    parts.mkdir()
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    monkeypatch.setattr(ZC, "PANEL_PARTS_DIR", str(parts))
    monkeypatch.setattr(ZC, "PANEL_OVERLAY_DIR", str(overlay))
    monkeypatch.setattr(ZC, "_CACHE_PATH", str(tmp_path / "cache.parquet"))
    ZC.limit_up_stats.cache_clear()
    yield parts
    ZC.limit_up_stats.cache_clear()


def _mk_panel(rows: list[tuple[str, str, float, float, int]]) -> pd.DataFrame:
    """rows: (code, trade_date, close, preclose, is_st)"""
    return pd.DataFrame(rows, columns=["stock_code", "trade_date", "close",
                                       "preclose", "is_st"])


class TestLimitCalendar:
    def test_limit_up_tiers(self, zt_env, monkeypatch):
        """⭐ 涨停判定分档：主板 10% / 创业板 20% / 主板 ST 10%（2026-07-06 起）；
        差一分钱都不算。

        ⚠️ 2026-10-29 修：旧版此处把「ST 5%」当通用规则（600002 收 10.50 就算涨停），
        而该用例日期 2026-10-16 已在新规之后 ⇒ **测试本身编码了 bug**。
        现在补一只 10.50 的主板 ST（600005）专门守住"不能再按 5% 判"。
        """
        monkeypatch.setattr(ZC, "_load_recent", lambda days=40, use_cache=True: _mk_panel([
            # 主板 10%：preclose 10.00 ⇒ 涨停价 11.00
            ("600001", "2026-10-15", 11.00, 10.00, 0),
            ("600001", "2026-10-16", 12.10, 11.00, 0),      # 二连板
            # 创业板 20%：preclose 10.00 ⇒ 12.00
            ("300001", "2026-10-16", 12.00, 10.00, 0),
            # 主板 ST：2026-10-16 已改 10% ⇒ 必须 11.00 才算涨停
            ("600002", "2026-10-16", 11.00, 10.00, 1),
            ("600005", "2026-10-16", 10.50, 10.00, 1),      # +5%：旧口径会误判成涨停
            # 差一分钱不算涨停
            ("600003", "2026-10-16", 10.99, 10.00, 0),
            # 跌停：主板 10% ⇒ 9.00
            ("600004", "2026-10-16", 9.00, 10.00, 0),
        ]))
        r = ZC.limit_up_stats()
        assert r["available"] is True
        assert r["trade_date"] == "2026-10-16"
        assert r["limit_up_count"] == 3, r     # 600001 二连板 + 300001 + 600002
        assert r["limit_down_count"] == 1
        assert r["max_consecutive"] == 2       # 600001 的二连板
        # 晋级率：T 日涨停 3 只中 600001 昨日也涨停 ⇒ 1/3 ≈ 33.3%
        assert r["upgrade_rate"] == pytest.approx(33.3, abs=0.1)

    @pytest.mark.parametrize("code, is_st, trade_date, want", [
        # 主板：非 ST 恒 10%
        ("600001", 0, "2026-10-16", 0.10),
        ("000001", 0, "2020-08-21", 0.10),
        # 主板 ST：2026-07-06 分界
        ("600001", 1, "2026-07-03", 0.05),
        ("600001", 1, "2026-07-06", 0.10),
        ("600001", 1, "2026-10-16", 0.10),
        # 创业板：2020-08-24 分界，改革后 ST 与板块同幅度
        ("300001", 0, "2020-08-21", 0.10),
        ("300001", 1, "2020-08-21", 0.05),
        ("300001", 0, "2020-08-24", 0.20),
        ("300001", 1, "2020-08-24", 0.20),
        ("301001", 1, "2026-10-16", 0.20),
        # 科创板 ST 也是 20%
        ("688001", 0, "2026-10-16", 0.20),
        ("688001", 1, "2026-10-16", 0.20),
        # 北交所 ST 也是 30%
        ("830001", 1, "2026-10-16", 0.30),
        ("920001", 0, "2026-10-16", 0.30),
        # 缺日期 ⇒ 按当前规则（宁可对现在准）
        ("600001", 1, None, 0.10),
        ("300001", 1, None, 0.20),
        # 脏 is_st 按非 ST
        ("600001", None, "2026-10-16", 0.10),
        ("600001", "abc", "2026-10-16", 0.10),
    ])
    def test_limit_ratio_matrix(self, code, is_st, trade_date, want):
        """结构性守护：幅度表本身逐格锁死（不经面板，最快定位回归）。"""
        assert ZC._limit_ratio(code, is_st, trade_date) == pytest.approx(want), \
            (code, is_st, trade_date)

    def test_gem_st_5pct_drop_is_not_limit_down(self, zt_env, monkeypatch):
        """⭐ 回归：创业板 ST 跌 5.4% 曾被判成跌停（实际限 20%）。

        2026-10-19 实测误报 12 只，8 只是这类；主板 ST 仍是 10%（该判就判）。
        """
        monkeypatch.setattr(ZC, "_load_recent", lambda days=40, use_cache=True: _mk_panel([
            ("300167", "2026-10-19", 9.46, 10.00, 1),   # -5.4%，限 20% ⇒ 非跌停
            ("002789", "2026-10-19", 9.00, 10.00, 1),   # -10%，限 10% ⇒ 是跌停
        ]))
        r = ZC.limit_up_stats()
        assert r["available"] is True
        assert r["limit_down_count"] == 1, r

    def test_overlay_is_merged_and_wins(self, zt_env, monkeypatch):
        """⭐ 回归：只读 parts 会停在"最后一次重建面板那天"。

        实测事故（2026-10-29）：parts 最新 10-19 且只有 1,371 只（按股票分块），
        overlay 已到 10-28 / 5,570 只 ⇒ 面板显示的是 7 天前、1/4 样本的家数。
        本用例锁死两件事：① overlay 的日期要进结果；② 同一天以 overlay 为准。
        """
        overlay = zt_env.parent / "overlay"
        pd.DataFrame([
            ("600001", "2026-10-16", 11.00, 10.00, 0),
        ], columns=["stock_code", "trade_date", "close", "preclose", "is_st"]
        ).to_parquet(zt_env / "daily_part_0001.parquet", index=False)
        # overlay 多一天，且把 10-16 那行改成"没涨停"（10.50）——检验 keep="last"
        pd.DataFrame([
            ("600001", "2026-10-16", 10.50, 10.00),
            ("600001", "2026-10-20", 11.00, 10.00),
        ], columns=["stock_code", "trade_date", "close", "preclose"]
        ).to_parquet(overlay / "2026-10-20.parquet", index=False)

        r = ZC.limit_up_stats()
        assert r["available"] is True
        assert r["trade_date"] == "2026-10-20", "overlay 的日期必须进结果"
        assert r["limit_up_count"] == 1, r

    def test_st_missing_is_flagged(self, zt_env, monkeypatch):
        """is_st 全缺失 ⇒ caveat 必须写明 ST 可能漏判（不能装作没问题）。"""
        monkeypatch.setattr(ZC, "_load_recent", lambda days=40, use_cache=True: _mk_panel([
            ("600001", "2026-10-16", 11.00, 10.00, None),
        ]))
        r = ZC.limit_up_stats()
        assert r["available"] is True
        assert "is_st" in r["caveat"]

    def test_no_panel_is_unavailable(self, zt_env):
        r = ZC.limit_up_stats()
        assert r["available"] is False

    def test_cache_reused_within_3_days(self, zt_env):
        """首次从 parts 构建并落盘缓存；清空 parts 后新鲜缓存仍可复用。"""
        import os
        pd.DataFrame([
            ("600001", "2026-10-16", 11.00, 10.00, 0),
        ], columns=["stock_code", "trade_date", "close", "preclose", "is_st"]
        ).to_parquet(zt_env / "daily_part_0001.parquet", index=False)
        r1 = ZC.limit_up_stats()
        assert r1["available"] is True
        assert os.path.exists(ZC._CACHE_PATH), "首次构建必须落盘缓存"
        # 清空 parts ⇒ 只能靠缓存
        for f in zt_env.iterdir():
            f.unlink()
        r2 = ZC.limit_up_stats()
        assert r2["available"] is True, "缓存新鲜期内必须可复用"


# ================================================================
# 1c. 指数多源择优（2026-10-23）
# ================================================================
#
# 事故：新浪源 `stock_zh_index_daily` 收盘后仍只到 T-1（16:12 实测最新为 10-22），
# 而 15:40 的回填照样报「5 成功 / 0 失败」——文件重写了、内容是旧的。
# ⇒ 市场检测 6 个指数指标**永远落后一天**（10-23 创业板 +4.16%，仍判"偏弱震荡"）。

class TestIndexSourceSelection:
    @pytest.fixture(autouse=True)
    def _no_tencent(self, monkeypatch):
        """默认关掉腾讯源：这组用例只验 akshare 两源之间的择优关系。

        ⚠️ 不关会**真实联网**（`_fetch_best_index` 现在会先试腾讯）。
        测试耦合真实环境是铁律禁止的：断网/限流时它会红，而代码其实没问题。
        """
        import data.index_bars as IB
        monkeypatch.setattr(IB, "_fetch_tencent_index", lambda *a, **k: None)

    def test_picks_fresher_source_not_first_success(self, monkeypatch):
        import akshare as ak
        import data.index_bars as IB
        old = pd.DataFrame({"date": ["2026-10-21", "2026-10-22"], "close": [1, 2]})
        new = pd.DataFrame({"date": ["2026-10-22", "2026-10-23"], "close": [2, 3]})
        monkeypatch.setattr(ak, "stock_zh_index_daily", lambda symbol: old)
        monkeypatch.setattr(ak, "stock_zh_index_daily_em", lambda symbol: new)
        df, src = IB._fetch_best_index("sh000001")
        assert src == "stock_zh_index_daily_em", "应选日期更新的源，不是先成功的那个"
        assert str(df["date"].iloc[-1]) == "2026-10-23"

    def test_falls_back_when_fresh_source_errors(self, monkeypatch):
        import akshare as ak
        import data.index_bars as IB
        old = pd.DataFrame({"date": ["2026-10-22"], "close": [2]})

        def _boom(symbol):
            raise RuntimeError("push2his 不可达")

        monkeypatch.setattr(ak, "stock_zh_index_daily_em", _boom)
        monkeypatch.setattr(ak, "stock_zh_index_daily", lambda symbol: old)
        df, src = IB._fetch_best_index("sh000001")
        assert src == "stock_zh_index_daily"
        assert len(df) == 1

    def test_all_sources_failing_raises(self, monkeypatch):
        """全失败要抛错（不能返回空表冒充"今天没有行情"）。"""
        import akshare as ak
        import data.index_bars as IB

        def _boom(symbol):
            raise RuntimeError("断网")

        monkeypatch.setattr(ak, "stock_zh_index_daily_em", _boom)
        monkeypatch.setattr(ak, "stock_zh_index_daily", _boom)
        with pytest.raises(RuntimeError):
            IB._fetch_best_index("sh000001")


# ================================================================
# 1d. 腾讯指数源 + 合并写（2026-10-26）
# ================================================================
#
# 事故：5 个 index parquet 全部停在 2026-10-23（当天是周一、正常交易日），
# 而 15:49 的回填日志报「5 成功 / 0 失败」——文件确实重写了、内容是旧的。
# 逐源核对（16:12）：
#   新浪 stock_zh_index_daily      → 最新仍是 10-23
#   东财 stock_zh_index_daily_em   → push2/80.push2/82.push2 全集群 RemoteDisconnected
#   腾讯 web.ifzq.gtimg.cn         → **已有 10-26**，1.2s 返回
# ⇒ 根因是「源本身滞后」，不是回填逻辑坏了。

class TestTencentIndexSource:
    def test_column_order_and_volume_scale(self, monkeypatch):
        """⚠️ 两个必须同时守住的口径，任一错都会让市场检测给出自信的错答案：

        1. 腾讯行序是 `[date, open, close, high, low, volume]` —— 第 3 位是**收盘**，
           不是 high。按位置错读会让整根 K 线错位。
           实测反证（2026-10-26 上证）：open=3671.00 close=3697.52 high=3699.01
           low=3664.82，`high > close > low` 自洽 —— 位置读对才有这个关系。
        2. `volume` 单位是**手**，新浪是**股**。三个指数同一天比值精确 100.0。
           忘乘 100 ⇒「两市量能」骤降百倍 ⇒ 判"地量" ⇒ 判熊市 ⇒ 仓位被压死。
        """
        import data.index_bars as IB

        fake = {"data": {"sh000001": {"day": [
            ["2026-10-26", "3671.00", "3697.52", "3699.01", "3664.82", "432794318"],
        ]}}}

        class _R:
            def raise_for_status(self):
                pass

            def json(self):
                return fake

        monkeypatch.setattr(IB.requests, "get", lambda *a, **k: _R())
        df = IB._fetch_tencent_index("sh000001")

        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
        row = df.iloc[0]
        assert float(row["close"]) == 3697.52, "第 3 位必须是收盘（不是 high）"
        assert float(row["high"]) == 3699.01
        assert float(row["low"]) == 3664.82
        assert float(row["volume"]) == 432794318 * 100, \
            "volume 必须 ×100 归一到「股」，否则量能指标缩百倍"
        # 自洽性：high ≥ close ≥ low 且 high ≥ open ≥ low
        assert float(row["high"]) >= max(float(row["close"]), float(row["open"]))
        assert float(row["low"]) <= min(float(row["close"]), float(row["open"]))

    def test_tencent_wins_when_it_has_newer_date(self, monkeypatch):
        """腾讯能给当日时必须被选中 —— 这正是本次事故的修复点。"""
        import akshare as ak
        import data.index_bars as IB

        stale = pd.DataFrame({"date": ["2026-10-22", "2026-10-23"],
                              "open": [1, 2], "high": [1, 2], "low": [1, 2],
                              "close": [1, 2], "volume": [10, 20]})
        fresh = pd.DataFrame({"date": ["2026-10-23", "2026-10-26"],
                              "open": [2, 3], "high": [2, 3], "low": [2, 3],
                              "close": [2, 3], "volume": [20, 30]})
        monkeypatch.setattr(IB, "_fetch_tencent_index", lambda *a, **k: fresh)
        monkeypatch.setattr(ak, "stock_zh_index_daily", lambda symbol: stale)
        monkeypatch.setattr(ak, "stock_zh_index_daily_em", lambda symbol: stale)

        df, src = IB._fetch_best_index("sh000001")
        assert src == "tencent"
        assert str(df["date"].iloc[-1]) == "2026-10-26"

    def test_tencent_failure_does_not_break_fallback(self, monkeypatch):
        """腾讯挂了不能连累整条链（它只是多一个源，不是必经之路）。"""
        import akshare as ak
        import data.index_bars as IB

        ok = pd.DataFrame({"date": ["2026-10-23"], "open": [1], "high": [1],
                           "low": [1], "close": [1], "volume": [10]})
        monkeypatch.setattr(IB, "_fetch_tencent_index", lambda *a, **k: None)
        monkeypatch.setattr(ak, "stock_zh_index_daily", lambda symbol: ok)
        monkeypatch.setattr(ak, "stock_zh_index_daily_em",
                            lambda symbol: (_ for _ in ()).throw(RuntimeError("挂了")))
        df, src = IB._fetch_best_index("sh000001")
        assert src == "stock_zh_index_daily"
        assert len(df) == 1


class TestMergeWriteKeepsHistory:
    """回填必须是「只补缺」，不能覆盖写。

    ⚠️⚠️ 为什么这条是硬要求而不是优化项：
      腾讯源单次最多 1000 根，本地新浪历史 8748 根（sh000001 实测）。
      若把择优源结果直接覆盖过去，十年历史会被截成四年；
      **更糟的是截断是永久的** —— 下次读到的 `old` 也只剩四年，越跑越短。
    """

    def test_append_only_does_not_truncate(self, tmp_path, monkeypatch):
        import data.index_bars as IB

        old = pd.DataFrame({
            "date": ["2026-10-22", "2026-10-23"],
            "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0],
            "close": [1.0, 2.0], "volume": [1e10, 2e10],
        })
        old.to_parquet(tmp_path / "sh000001.parquet", index=False)

        new = pd.DataFrame({
            "date": ["2026-10-23", "2026-10-26"],
            "open": [2.0, 3.0], "high": [2.0, 3.0], "low": [2.0, 3.0],
            "close": [2.0, 3.0], "volume": [2e10, 3e10],
        })
        monkeypatch.setattr(IB, "INDEX_DIR", str(tmp_path))
        monkeypatch.setattr(IB, "_fetch_best_index",
                            lambda symbol: (new, "tencent"))
        monkeypatch.setattr(IB, "expected_close_date", lambda: "2026-10-26")

        r = IB.backfill_index_daily(symbols=("sh000001",))
        got = pd.read_parquet(tmp_path / "sh000001.parquet")
        assert len(got) == 3, "重叠日 10-23 不能重复插入"
        assert list(got["date"]) == ["2026-10-22", "2026-10-23", "2026-10-26"]
        assert r["done"]["sh000001"]["added"] == 1
        # 重叠日以本地为准：10-23 的 volume 必须还是旧值
        assert float(got.loc[got["date"] == "2026-10-23", "volume"].iloc[0]) == 2e10
        assert r["stale"] == {}, "已补到当日就不该再报陈旧"


# ================================================================
# 1d. 腾讯指数源 + 合并写（2026-10-26）
# ================================================================
#
# 事故：5 个 index parquet 全部停在 2026-10-23（当天是周一、正常交易日），
# 而 15:49 的回填日志报「5 成功 / 0 失败」——文件确实重写了、内容是旧的。
# 逐源核对（16:12）：
#   新浪 stock_zh_index_daily      → 最新仍是 10-23
#   东财 stock_zh_index_daily_em   → push2/80.push2/82.push2 全集群 RemoteDisconnected
#   腾讯 web.ifzq.gtimg.cn         → **已有 10-26**，1.2s 返回
# ⇒ 根因是「源本身滞后」，不是回填逻辑坏了。
