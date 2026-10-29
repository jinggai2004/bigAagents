# -*- coding: utf-8 -*-
"""tushare 当日增量（A 档）与按日回填（备选路径）的语义边界。

锁住的不是"能跑"，而是几条**一旦写错就静默造错数据/静默降级**的边界：
  1. **HTTP 200 ≠ 成功**：业务码 40203=限频（⇒ 降级）、其它 ⇒ 显式报错；
  2. **陈旧判据 = 数据自带日期 vs 期望日期**：当日截面未生成绝不能拿昨日顶替（前视）；
  3. **单位换算只做一次**：vol 手→股、amount 千元→元、total_mv/circ_mv 万元→元
     —— 写错任何一处，因子值整体失真（新浪"亿元→元"踩过同款坑）；
  4. **缺就是缺**：单边缺失（只有 daily 没有 daily_basic）→ 该字段 None，绝不补 0；
  5. **进度语义分离**：按股 done/failed 与按日 done_dates/failed_dates 分键，
     旧写出口不得把按日进度覆盖丢掉；
  6. **tushare 不可用 ⇒ 回退新浪**（15:40 快照），降级必须可查。
"""
import os

import pandas as pd
import pytest

from tests.conftest import use_template_db

import data.tushare_source as TS
import data.panel_store as PS


@pytest.fixture(autouse=True)
def _reset_throttle(tmp_path, monkeypatch):
    """`_last_call_at` / 能力表都是模块级状态 ⇒ 测试间必须归零。

    ⚠️ 只清 `_last_call_at` 不够：`call_api` 现在会 `_record()` 并**落盘**，
    不隔离就会把 mock 出来的 "40203" 写进真实的 `temp/tushare_capability.json`
    （第一次跑就污染成 `daily.reject_count=22`）。桶一律指到 tmp_path。
    """
    TS._last_call_at.clear()
    monkeypatch.setattr(TS, "_CAPABILITY_PATH", str(tmp_path / "cap_autouse.json"))
    monkeypatch.setattr(TS, "_CAP_STATE", {})
    monkeypatch.setattr(TS, "_CAP_LOADED", False)
    yield
    TS._last_call_at.clear()


# ================================================================
# 1. 业务码与限频
# ================================================================


def _js_ok(items, fields):
    return {"code": 0, "msg": None, "data": {"fields": fields, "items": items}}


class TestCallApi:
    def test_rate_limited_by_local_throttle(self, monkeypatch):
        """⚠️ trade_cal 1 次/小时 ⇒ 本地节流第二次**立即**抛 RateLimited
        （调度任务宁可降级也不能挂死等 1 小时）。

        ⚠️ 2026-10-26 起 `daily_basic` **不再参与本地节流** —— 它对 120 积分档是
        无权限（logs 547 次失败 / 0 次成功），"限频多少秒"没有意义，由能力表直接跳过。
        这里换 `trade_cal` 承担"调度接口必须抛而不是等"的角色。
        """
        monkeypatch.setattr(TS, "_load_token", lambda: "tok")
        monkeypatch.setattr(TS.requests, "post",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("不该真发请求")))
        TS._last_call_at["trade_cal"] = pd.Timestamp.now().timestamp()  # 刚调过
        with pytest.raises(TS.TushareRateLimited, match="本地节流"):
            TS.call_api("trade_cal", {"trade_date": "20261016"})

    def test_server_rate_limit_maps_to_rate_limited(self, monkeypatch):
        monkeypatch.setattr(TS, "_load_token", lambda: "tok")

        class _R:
            status_code = 200

            def json(self):
                return {"code": 40203, "msg": "抱歉，您每小时最多访问该接口1次"}

        monkeypatch.setattr(TS.requests, "post", lambda *a, **k: _R())
        with pytest.raises(TS.TushareRateLimited, match="40203"):
            TS.call_api("daily", {"trade_date": "20261016"})

    def test_business_error_propagates(self, monkeypatch):
        monkeypatch.setattr(TS, "_load_token", lambda: "tok")

        class _R:
            status_code = 200

            def json(self):
                return {"code": 10001011, "msg": "黑名单用户，请与管理员联系"}

        monkeypatch.setattr(TS.requests, "post", lambda *a, **k: _R())
        with pytest.raises(TS.TushareApiError, match="黑名单"):
            TS.call_api("daily", {"trade_date": "20261016"})

    def test_stale_date_rejected(self, monkeypatch):
        """⭐ 当日截面未生成（返回了昨日）⇒ 显式报错，**绝不拿昨日顶替**（前视）。"""
        monkeypatch.setattr(TS, "_load_token", lambda: "tok")

        class _R:
            status_code = 200

            def json(self):
                return _js_ok([["20261015", "5.5"]], ["trade_date", "close"])

        monkeypatch.setattr(TS.requests, "post", lambda *a, **k: _R())
        with pytest.raises(TS.TushareStaleData, match="20261016"):
            TS.fetch_daily("2026-10-16")

    def test_daily_throttle_sleeps_not_raises(self, monkeypatch):
        """⭐⭐ 回填接口（daily）节流必须 **sleep 等待**——第一版全用"抛异常"，
        逐日回填 2565 天全被自己的节流器打回（2026-10-17 15:18 事故）。"""
        import time as _t
        monkeypatch.setattr(TS, "_load_token", lambda: "tok")
        # ⚠️⚠️ `_throttle` 内部用 **time.monotonic()** —— 这里若用墙钟 time.time()
        #    设 last，两者差值巨大 ⇒ remain 变天文数字 ⇒ **sleep 永不返回**
        #    （本测试第一版就卡死了 pytest，2026-10-17）。
        TS._last_call_at["daily"] = _t.monotonic()                   # 刚调过
        t0 = _t.monotonic()
        TS._throttle("daily")                                        # 不抛 ⇒ sleep 补足间隔
        elapsed = _t.monotonic() - t0
        assert 1.0 <= elapsed <= 1.5, f"应 sleep 补足 1.2s 间隔，实际 {elapsed:.2f}s"
        # 调度接口语义不变：仍抛（用 trade_cal 代表仍受节流约束的调度接口）
        TS._last_call_at["trade_cal"] = _t.monotonic()
        with pytest.raises(TS.TushareRateLimited):
            TS._throttle("trade_cal")

    def test_no_token_is_explicit(self, monkeypatch):
        monkeypatch.setattr(TS, "_load_token", lambda: "")
        with pytest.raises(TS.TushareApiError, match="TUSHARE_TOKEN"):
            TS.call_api("daily", {"trade_date": "20261016"})


# ================================================================
# 2. 单位换算（normalize_tushare_daily_df）
# ================================================================


class TestNormalizeTushareDaily:
    def test_units_hand_to_share_and_kyuan_to_yuan(self):
        """⭐⭐ vol 手→股（×100）、amount 千元→元（×1000）——写错一处因子全错。"""
        raw = pd.DataFrame([{
            "ts_code": "600028.SH", "trade_date": "20261016",
            "open": "5.50", "high": "5.60", "low": "5.48", "close": "5.53",
            "pre_close": "5.50", "vol": "1234",           # 手
            "amount": "870497.3",                          # 千元
            "pct_chg": "0.55",
        }])
        out = PS.normalize_tushare_daily_df(raw)
        assert list(out.columns) == PS.PANEL_COLUMNS
        row = out.iloc[0]
        assert row["stock_code"] == "600028"
        assert row["preclose"] == 5.50
        assert row["volume"] == pytest.approx(1234 * 100.0)
        assert row["amount"] == pytest.approx(870497.3 * 1000.0)
        # 估值列保持 NaN（等 BaoStock 行覆盖补齐），**绝不造 0**
        for col in ("turn", "pe_ttm", "pb_mrq", "ps_ttm", "trade_status", "is_st"):
            assert pd.isna(row[col]), col

    def test_trade_date_normalized_to_iso(self):
        """⭐⭐ tushare 的 8 位日期必须归一成 `YYYY-MM-DD` 再落盘。

        混入 `20240517` 会让 `load_panel` 字符串区间过滤双向出错
        （`'20240101' > '2024-01-02'` 字符串比较为 True），同键去重失效
        （同股同日两种格式各一行 ⇒ 双行）。实测 2026-10-17 按日回填
        1101 万行全是不带连字符的格式，靠此断言焊死。
        """
        raw = pd.DataFrame([
            {"ts_code": "600028.SH", "trade_date": "20261016",
             "open": "1", "high": "1", "low": "1", "close": "1",
             "pre_close": "1", "vol": "1", "amount": "1", "pct_chg": "0"},
            {"ts_code": "000001.SZ", "trade_date": "2026-10-15",   # 已是 ISO ⇒ 原样
             "open": "1", "high": "1", "low": "1", "close": "1",
             "pre_close": "1", "vol": "1", "amount": "1", "pct_chg": "0"},
        ])
        out = PS.normalize_tushare_daily_df(raw)
        # 函数内部按 trade_date 排序 ⇒ 归一后的 ISO 顺序
        assert list(out["trade_date"]) == ["2026-10-15", "2026-10-16"]

    def test_empty_returns_empty_frame(self):
        out = PS.normalize_tushare_daily_df(pd.DataFrame())
        assert out.empty and list(out.columns) == PS.PANEL_COLUMNS


# ================================================================
# 3. 估值截面组装（build_market_snapshot_rows）
# ================================================================


class TestBuildMarketSnapshotRows:
    def _patch(self, monkeypatch):
        basic = pd.DataFrame([{
            "ts_code": "600028.SH", "trade_date": "20261016", "close": "5.53",
            "turnover_rate": "10.2542", "pe": "133.0", "pe_ttm": "131.5",
            "pb": "4.583", "total_mv": "851171.1",          # 万元
            "circ_mv": "851144.0",
        }])
        daily = pd.DataFrame([{
            "ts_code": "600028.SH", "trade_date": "20261016", "close": "5.53",
            "pre_close": "5.50", "vol": "123400",           # 手
            "amount": "870497.3",                            # 千元
            "pct_chg": "0.55",
        }])
        info = pd.DataFrame([{
            "ts_code": "600028.SH", "symbol": "600028", "name": "中国石化",
            "industry": "石油行业", "market": "主板上海",
        }])
        monkeypatch.setattr(TS, "fetch_daily_basic", lambda d: basic)
        monkeypatch.setattr(TS, "fetch_daily", lambda d: daily)
        monkeypatch.setattr(TS, "fetch_stock_basic", lambda: info)

    def test_units_and_join(self, monkeypatch):
        self._patch(monkeypatch)
        rows, attrs = TS.build_market_snapshot_rows("2026-10-16")
        r = rows[0]
        assert attrs["source"] == "tushare"
        assert r["stock_code"] == "600028" and r["stock_name"] == "中国石化"
        assert r["industry"] == "石油行业" and r["board"] == "主板上海"
        assert r["price"] == 5.53 and r["change_pct"] == pytest.approx(0.55)
        assert r["pe"] == 131.5, "pe 列 = TTM 口径（与 BaoStock peTTM 一致）"
        assert r["pb"] == pytest.approx(4.583)
        assert r["market_cap"] == pytest.approx(851171.1 * 1e4), "万元 → 元"
        assert r["float_cap"] == pytest.approx(851144.0 * 1e4)
        assert r["turnover"] == pytest.approx(10.2542), "百分数不换算"
        assert r["amount"] == pytest.approx(870497.3 * 1e3), "千元 → 元"

    def test_single_side_missing_stays_none(self, monkeypatch):
        """⭐ 只有一边有数据 ⇒ 该边字段 None，**绝不补 0**（缺就是缺）。"""
        basic = pd.DataFrame([{"ts_code": "600028.SH", "trade_date": "20261016",
                               "close": "5.53", "pe_ttm": "131.5", "pb": "4.5",
                               "total_mv": "851171.1", "circ_mv": "851144.0",
                               "turnover_rate": "10.0"}])
        daily = pd.DataFrame()                            # daily 整体缺
        monkeypatch.setattr(TS, "fetch_daily_basic", lambda d: basic)
        monkeypatch.setattr(TS, "fetch_daily", lambda d: daily)
        monkeypatch.setattr(TS, "fetch_stock_basic", lambda: pd.DataFrame())
        rows, attrs = TS.build_market_snapshot_rows("2026-10-16")
        r = rows[0]
        assert r["amount"] is None and r["change_pct"] is None
        assert r["pe"] == 131.5 and r["market_cap"] == pytest.approx(851171.1 * 1e4)
        assert any("daily" in x for x in attrs["degraded"]), "单边缺失必须进 degraded"


# ================================================================
# 4. 按日回填编排（run_backfill_tushare）
# ================================================================


@pytest.fixture
def panel_env(tmp_path, monkeypatch):
    monkeypatch.setattr(PS, "PANEL_DIR", str(tmp_path / "panel"))
    monkeypatch.setattr(PS, "PARTS_DIR", str(tmp_path / "panel" / "parts"))
    monkeypatch.setattr(PS, "PROGRESS_PATH", str(tmp_path / "panel" / "progress.json"))
    return tmp_path


def _day_df(d: str, n: int = 2) -> pd.DataFrame:
    return pd.DataFrame([{
        "ts_code": f"{600000 + i}.SH", "trade_date": d.replace("-", ""),
        "open": "1", "high": "1", "low": "1", "close": "1", "pre_close": "1",
        "vol": "100", "amount": "1000", "pct_chg": "0",
    } for i in range(n)])


class TestRunBackfillTushare:
    def test_by_date_and_progress_key_separated(self, panel_env):
        days = ["20261015", "20261016"]
        stats = PS.run_backfill_tushare(
            trade_days=days, fetch_one_day=lambda d: _day_df(d),
            batch_size=10, progress_every=0)
        assert stats["fetched"] == 2 and stats["rows"] == 4
        # ⚠️ 按日进度必须写在 done_dates（与按股 done 分键），按股进度不受污染
        prog = PS._read_progress()
        assert set(prog.get("done_dates") or {}) == set(days)
        assert prog.get("done") in (None, {}, ) or "600028" not in (prog.get("done") or {})

    def test_resume_skips_done_dates(self, panel_env):
        days = ["20261015", "20261016"]
        PS.run_backfill_tushare(trade_days=days, fetch_one_day=lambda d: _day_df(d),
                                batch_size=10, progress_every=0)
        calls = []

        def fetch(d):
            calls.append(d)
            return _day_df(d)

        stats = PS.run_backfill_tushare(trade_days=days, fetch_one_day=fetch,
                                        batch_size=10, progress_every=0)
        assert calls == [], "已完成的日期必须全部跳过"
        assert stats["skipped_done"] == 2 and stats["fetched"] == 0

    def test_single_day_failure_recorded_not_aborting(self, panel_env):
        def fetch(d):
            if d == "20261016":
                raise TS.TushareStaleData("DATE_MISMATCH", "当日截面未生成")
            return _day_df(d)

        stats = PS.run_backfill_tushare(
            trade_days=["20261015", "20261016", "20261017"],
            fetch_one_day=fetch, batch_size=10, progress_every=0)
        assert stats["failed"] == 1 and stats["fetched"] == 2
        prog = PS._read_progress()
        assert "当日截面未生成" in (prog.get("failed_dates") or {})["20261016"]
        assert len(PS.load_panel()) == 4

    def test_overlap_deduped(self, panel_env):
        for _ in range(2):
            PS.run_backfill_tushare(trade_days=["20261016"],
                                    fetch_one_day=lambda d: _day_df(d, 3),
                                    batch_size=10, resume=False, progress_every=0)
        df = PS.load_panel()
        assert len(df) == 3 and not df.duplicated(["stock_code", "trade_date"]).any()


# ================================================================
# 5. 调度接线：tushare 优先、失败回退新浪
# ================================================================


class TestSchedulerSnapshotFallback:
    @pytest.fixture(autouse=True)
    def db(self, tmp_path, monkeypatch):
        use_template_db(tmp_path, monkeypatch, name="tushare_snap.db")

    def _run(self, monkeypatch, tushare_ok: bool):
        from core import scheduler
        if tushare_ok:
            rows, attrs = TS.build_market_snapshot_rows.__wrapped__(  # noqa: SLF001
                None) if False else (None, None)
            # 直接打桩三个 fetch（走真实组装逻辑）
            basic = pd.DataFrame([{"ts_code": "600028.SH", "trade_date": "20261016",
                                   "close": "5.53", "pe_ttm": "131.5", "pb": "4.5",
                                   "total_mv": "851171.1", "circ_mv": "851144.0",
                                   "turnover_rate": "10.0"}])
            daily = pd.DataFrame([{"ts_code": "600028.SH", "trade_date": "20261016",
                                   "close": "5.53", "pre_close": "5.5", "vol": "100",
                                   "amount": "1000", "pct_chg": "0.55"}])
            info = pd.DataFrame([{"ts_code": "600028.SH", "symbol": "600028",
                                  "name": "中国石化", "industry": "石油行业",
                                  "market": "主板上海"}])
            monkeypatch.setattr(TS, "fetch_daily_basic", lambda d: basic)
            monkeypatch.setattr(TS, "fetch_daily", lambda d: daily)
            monkeypatch.setattr(TS, "fetch_stock_basic", lambda: info)
            monkeypatch.setattr(scheduler, "_snapshot_quote_date",
                                lambda: "2026-10-16")
        else:
            def _boom(d):
                raise TS.TushareRateLimited(40203, "每小时 1 次")
            monkeypatch.setattr(TS, "build_market_snapshot_rows", _boom)
            monkeypatch.setattr(scheduler, "_snapshot_quote_date",
                                lambda: "2026-10-16")
            # 新浪回退路径打桩：给最小可用快照
            import pandas as _pd
            df = _pd.DataFrame([{"code": "600028", "name": "中国石化",
                                 "industry": "石油行业", "board": "主板上海",
                                 "price": 5.53, "change_pct": 0.55, "pe": 131.5,
                                 "pb": 4.5, "market_cap": 85.1, "float_cap": 85.1,
                                 "turnover": 10.0, "amount": 8.7}])
            df.attrs.update({"coverage": 0.9, "total_market": 1, "degraded": [],
                             "caveat": ""})
            monkeypatch.setattr("data.sina_source.get_sector_snapshot",
                                lambda: df)
        return scheduler.save_daily_market_snapshot("2026-10-16")

    def test_tushare_success_writes_with_source(self, monkeypatch):
        out = self._run(monkeypatch, tushare_ok=True)
        assert out["ok"] and out["rows"] == 1
        from db import store
        d, rows = store.load_market_snapshot("2026-10-16")
        assert rows[0]["source"] == "tushare"
        assert rows[0]["pe"] == 131.5
        assert rows[0]["amount"] == pytest.approx(1000 * 1e3)
        cov = store.load_data_coverage(dataset="market_snapshot")
        assert cov and cov[0]["source"] == "tushare"

    def test_tushare_failure_falls_back_to_sina(self, monkeypatch, caplog):
        out = self._run(monkeypatch, tushare_ok=False)
        assert out["ok"], "回退新浪必须成功"
        from db import store
        d, rows = store.load_market_snapshot("2026-10-16")
        assert rows[0]["source"] == "sina_sector"


# ================================================================
# 8. 「无权限」vs「限频」——一个业务码不能承载两类语义（2026-10-26）
# ================================================================
#
# 事故还原：`daily_basic` 对 120 积分档**根本没有权限**，但服务端不说"无权限"，
# 而是轮换着用三种互相矛盾的"频率超限"话术软拒绝。logs/ 全量统计 547 次失败：
#     339 × 「频率超限(1次/小时)」  195 × 「频率超限(1次/分钟)」  13 × 「频率超限(5次/天)」
# **同一个接口不可能同时有三个上限** —— 三个数字互相矛盾本身就说明它不是额度。
# 而旧代码 `if code == 40203 or "每分钟" in msg or "每小时" in msg: raise RateLimited`
# 把它判成"限频"⇒ 调用方以为等一会儿就好 ⇒ 每个交易日 15:40/15:47 各白试一次，
# 一天两次、一年 500 次，**成功率 0**。
#
# 更隐蔽的是：`moneyflow` 的「没有接口访问权限」**业务码同样是 40203**。
# 只看码会把"永远别来了"当成"等会儿再来"。


@pytest.fixture
def cap(tmp_path, monkeypatch):
    """能力表隔离到 tmp_path —— 绝不碰真实 `temp/tushare_capability.json`。

    ⚠️ `_CAPABILITY_PATH` 是模块级缓存（首次使用时按 data_root() 解析），
    只 `setenv BIGA_DATA_ROOT` **挡不住**它，必须同时 patch 缓存本身。
    """
    path = tmp_path / "tushare_capability.json"
    monkeypatch.setattr(TS, "_CAPABILITY_PATH", str(path))
    monkeypatch.setattr(TS, "_CAP_STATE", {})
    monkeypatch.setattr(TS, "_CAP_LOADED", False)
    return path


def _reject(msg, code=40203):
    class _R:
        status_code = 200

        def json(self):
            return {"code": code, "msg": msg, "data": None}

    return _R()


class TestPermissionVsRateLimit:
    def test_explicit_no_permission_wording(self, monkeypatch, cap):
        """「您没有接口(moneyflow)访问权限」⇒ NoPermission，**不是** RateLimited。"""
        monkeypatch.setattr(TS, "_load_token", lambda: "tok")
        monkeypatch.setattr(TS.requests, "post", lambda *a, **k: _reject(
            "抱歉，您没有接口(moneyflow)访问权限，权限的具体详情访问：https://tushare.pro/document/1?doc_id=108。"))
        with pytest.raises(TS.TushareNoPermission):
            TS.call_api("moneyflow", {"trade_date": "20261016"})

    def test_same_code_different_semantics(self, cap):
        """⚠️ 核心：**同一个业务码 40203**，文案不同 ⇒ 异常类型必须不同。"""
        e1 = TS._classify_api_error("moneyflow", 40203, "抱歉，您没有接口(moneyflow)访问权限")
        e2 = TS._classify_api_error("some_api", 40203, "抱歉，您访问接口(x)频率超限(1次/分钟)")
        assert isinstance(e1, TS.TushareNoPermission), "无权限不能被当成限频"
        assert isinstance(e2, TS.TushareRateLimited), "首两次数不清是无权限，应先当限频"
        assert not isinstance(e2, TS.TushareNoPermission)

    def test_repeated_soft_reject_becomes_no_permission(self, cap):
        """「从未成功过 + 累计被拒 ≥3」⇒ 那三个数字就不是额度，是无权限的软拒绝。"""
        kinds = []
        for _ in range(TS._SOFT_REJECT_LIMIT):
            kinds.append(type(TS._classify_api_error(
                "daily_basic", 40203, "抱歉，您访问接口(daily_basic)频率超限(1次/小时)")).__name__)
        assert kinds[:2] == ["TushareRateLimited", "TushareRateLimited"], \
            "前两次要给调用方留余地（可能真的是瞬时限频）"
        assert kinds[-1] == "TushareNoPermission", \
            "从未成功过还一直报限频 ⇒ 必须升级判定，否则每个交易日都白试"

    def test_success_resets_the_soft_reject_count(self, cap):
        """成功过就不能再判无权限 —— 那是**真限频**，等一等是能好的。"""
        for _ in range(5):
            TS._classify_api_error("daily_basic", 40203, "频率超限(1次/小时)")
        TS._record("daily_basic", ok=True)
        # 成功过之后再被拒，不该立刻又判无权限
        e = TS._classify_api_error("daily_basic", 40203, "频率超限(1次/小时)")
        assert isinstance(e, TS.TushareRateLimited)

    def test_no_permission_is_persisted_across_processes(self, monkeypatch, cap):
        """⭐⭐ 关键：能力表必须**落盘**。

        旧的 `_MONEYFLOW_DENIED` 是进程内的，注释里写"重启后重试"——
        而调度器每个交易日都会重启一次 ⇒ 每个交易日又白试一轮。
        """
        monkeypatch.setattr(TS, "_load_token", lambda: "tok")
        monkeypatch.setattr(TS.requests, "post", lambda *a, **k: _reject(
            "抱歉，您没有接口(moneyflow)访问权限"))
        with pytest.raises(TS.TushareNoPermission):
            TS.call_api("moneyflow", {})
        assert cap.exists(), "能力表必须落盘，否则重启就忘"

        # 模拟"进程重启"：清空所有内存状态，只留磁盘文件
        TS._CAP_STATE.clear()
        TS._CAP_LOADED = False
        called = []
        monkeypatch.setattr(TS.requests, "post", lambda *a, **k: called.append(1) or _reject("x"))
        with pytest.raises(TS.TushareNoPermission, match="无权限"):
            TS.call_api("moneyflow", {})
        assert called == [], "确认无权限后必须直接拒，**一个请求都不该再发**"

    def test_probe_after_interval_allows_retry(self, monkeypatch, cap):
        """7 天探针：用户升级积分后不需要改代码就能自愈。"""
        monkeypatch.setattr(TS, "_load_token", lambda: "tok")
        monkeypatch.setattr(TS.requests, "post", lambda *a, **k: _reject(
            "抱歉，您没有接口(moneyflow)访问权限"))
        with pytest.raises(TS.TushareNoPermission):
            TS.call_api("moneyflow", {})
        assert TS.unavailable_reason("moneyflow"), "刚确认过，应判不可用"

        # 把探针时间拨到过去 ⇒ 应放行一次
        TS._CAP_STATE["moneyflow"]["probe_after"] = 0
        assert TS.unavailable_reason("moneyflow") == "", "过了探针期必须放行试一次"

    def test_ok_api_is_never_blocked(self, monkeypatch, cap):
        """`daily` 是通的 ⇒ 不能被能力表误伤（120 积分档唯一能用的接口）。"""
        monkeypatch.setattr(TS, "_load_token", lambda: "tok")
        monkeypatch.setattr(TS.requests, "post", lambda *a, **k: type("_R", (), {
            "status_code": 200,
            "json": lambda self: _js_ok([["000001.SZ", 1.0]], ["ts_code", "close"])})())
        df = TS.call_api("daily", {"trade_date": "20261016"})
        assert len(df) == 1
        assert TS.unavailable_reason("daily") == ""
        assert TS.capability_snapshot()["daily"]["status"] == "ok"
