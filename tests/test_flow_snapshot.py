"""全市场资金流/行业资金流快照：落库、读库、以及"选股不再现场拉全市场"的回归。

背景（2026-09-25 实测）：同花顺快照一个窗口就要拉**全市场 5210 行**，
  即时 16.7s / 3日 20.8s / 5日 15.6s / 10日 28.6s
  → 为拿到**一只**股票的资金流要 **75.9 秒**（4 次全市场）。
而快照内存缓存只有 300 秒，选股一轮 4~5 分钟**中途必然过期再拉一轮**
→ 一次选股光在这上面就白花约 150 秒。
改为：盘后 15:40 采集一次落库（实测 103 秒，全市场 5210 只 + 90 行业），
选股只读库（首查读全表 269ms，其后 0ms/只）。
"""
import time
from unittest.mock import patch

import pandas as pd
import pytest

import db.models as models
from db import store as db_store
import data.fund_flow as ff


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    """数据库指向临时文件，隔离真实 portfolio.db"""
    from tests.conftest import use_template_db
    db_path = use_template_db(tmp_path, monkeypatch, "test_flow.db")
    yield


@pytest.fixture(autouse=True)
def _clear_mem_cache():
    """进程内整表缓存必须逐测清理，否则会串味"""
    ff._snapshot_mem_cache.update({"at": 0.0, "date": None, "by_code": {}})
    yield
    ff._snapshot_mem_cache.update({"at": 0.0, "date": None, "by_code": {}})


# ================================================================
# 落库 / 读库
# ================================================================

class TestStoreSnapshot:
    def test_save_and_load(self, temp_store):
        n = db_store.save_fund_flow_snapshot("2026-09-24", [
            {"stock_code": "601166", "stock_name": "兴业银行", "net_today": 2.61e7,
             "net_3d": 2.88e8, "net_5d": 1.49e8, "net_10d": -7.4e8,
             "amount": 1.068e9, "change_pct": 0.06, "price": 17.88},
            {"stock_code": "600519", "stock_name": "贵州茅台", "net_today": -4.23e8},
        ])
        assert n == 2
        date, data = db_store.load_fund_flow_snapshot()
        assert date == "2026-09-24"
        assert data["601166"]["net_5d"] == pytest.approx(1.49e8)
        assert data["600519"]["net_today"] == pytest.approx(-4.23e8)
        assert data["600519"]["net_5d"] is None, "未提供的窗口必须留空，不得补造"

    def test_upsert_same_date_and_code(self, temp_store):
        db_store.save_fund_flow_snapshot("2026-09-24", [
            {"stock_code": "601166", "net_today": 1.0}])
        db_store.save_fund_flow_snapshot("2026-09-24", [
            {"stock_code": "601166", "net_today": 2.0, "net_5d": 3.0}])
        _, data = db_store.load_fund_flow_snapshot()
        assert len(data) == 1, "同一 date+code 必须覆盖而不是新增"
        assert data["601166"]["net_today"] == 2.0
        assert data["601166"]["net_5d"] == 3.0

    def test_latest_date_wins(self, temp_store):
        db_store.save_fund_flow_snapshot("2026-09-23", [{"stock_code": "601166", "net_today": 1.0}])
        db_store.save_fund_flow_snapshot("2026-09-24", [{"stock_code": "601166", "net_today": 2.0}])
        assert db_store.latest_fund_flow_date() == "2026-09-24"
        date, data = db_store.load_fund_flow_snapshot()
        assert date == "2026-09-24" and data["601166"]["net_today"] == 2.0

    def test_codes_filter(self, temp_store):
        db_store.save_fund_flow_snapshot("2026-09-24", [
            {"stock_code": "601166", "net_today": 1.0},
            {"stock_code": "600519", "net_today": 2.0}])
        _, data = db_store.load_fund_flow_snapshot(codes=["601166"])
        assert list(data) == ["601166"]

    def test_empty_store_returns_none_date(self, temp_store):
        assert db_store.latest_fund_flow_date() is None
        assert db_store.load_fund_flow_snapshot() == (None, {})
        assert db_store.load_industry_flow_snapshot() == (None, [])

    def test_industry_snapshot(self, temp_store):
        n = db_store.save_industry_flow_snapshot("2026-09-24", [
            {"industry": "银行", "net_amount": 1.605e9, "change_pct": 0.5, "leader": "兴业银行"},
            {"industry": "半导体", "net_amount": -1.32e10, "change_pct": -2.0, "leader": "中芯国际"},
        ])
        assert n == 2
        date, rows = db_store.load_industry_flow_snapshot()
        assert date == "2026-09-24"
        assert rows[0]["industry"] == "银行", "应按净额降序"

    def test_empty_rows_noop(self, temp_store):
        assert db_store.save_fund_flow_snapshot("2026-09-24", []) == 0
        assert db_store.save_fund_flow_snapshot("", [{"stock_code": "x"}]) == 0
        assert db_store.latest_fund_flow_date() is None

    def test_cleanup_covers_snapshots(self, temp_store):
        """新表必须登记保留策略，否则会无限增长"""
        db_store.save_fund_flow_snapshot("2000-01-01", [{"stock_code": "601166"}])
        db_store.save_industry_flow_snapshot("2000-01-01", [{"industry": "银行"}])
        deleted = db_store.cleanup()
        assert "fund_flow_snapshot" in deleted and "industry_flow_snapshot" in deleted
        assert db_store.latest_fund_flow_date() is None


# ================================================================
# 采集
# ================================================================

def _window_df(window: str, code: str, net, name: str = "测试股") -> pd.DataFrame:
    """构造同花顺单窗口返回（列名与真实接口一致）"""
    return pd.DataFrame([{
        "股票代码": code, "股票简称": name, "净额": net,
        "成交额": 1.0e9, "涨跌幅": "0.50", "最新价": "10.00",
    }])


class TestCapture:
    def test_capture_merges_windows_and_saves(self, temp_store):
        windows = {"即时": 2.61e7, "3日排行": 2.88e8, "5日排行": 1.49e8, "10日排行": -7.4e8}

        def fake_snapshot(window):
            return _window_df(window, "601166", windows[window])

        with patch("data.fund_flow.get_ths_snapshot", side_effect=fake_snapshot), \
             patch("data.fund_flow.get_industry_flow_snapshot", return_value=None), \
             patch("data.fund_flow._snapshot_data_date", return_value="2026-09-24"):
            res = ff.capture_market_flow_snapshot()

        assert res["snapshot_date"] == "2026-09-24"
        assert res["stocks"] == 1
        date, data = db_store.load_fund_flow_snapshot()
        rec = data["601166"]
        assert rec["net_today"] == pytest.approx(2.61e7)
        assert rec["net_3d"] == pytest.approx(2.88e8)
        assert rec["net_5d"] == pytest.approx(1.49e8)
        assert rec["net_10d"] == pytest.approx(-7.4e8)

    def test_missing_window_is_left_empty_not_fabricated(self, temp_store):
        """回归：窗口取不到时必须留空。

        原实现 `record.setdefault("净额_3日", record.get("净额_今日", 0.0))`
        会把缺失窗口填成"今日"的值 —— 等于凭空造出"三个窗口都为正"的方向分。
        """
        def fake_snapshot(window):
            if window == "即时":
                return _window_df(window, "601166", 2.61e7)
            return None                                  # 3/5/10 日窗全部失败

        with patch("data.fund_flow.get_ths_snapshot", side_effect=fake_snapshot), \
             patch("data.fund_flow.get_industry_flow_snapshot", return_value=None), \
             patch("data.fund_flow._snapshot_data_date", return_value="2026-09-24"):
            ff.capture_market_flow_snapshot()

        _, data = db_store.load_fund_flow_snapshot()
        rec = data["601166"]
        assert rec["net_today"] == pytest.approx(2.61e7)
        assert rec["net_3d"] is None and rec["net_5d"] is None and rec["net_10d"] is None

    def test_capture_bypasses_memory_cache(self, temp_store):
        """采集必须拿最新，不能被 300 秒内存缓存挡住"""
        calls = []

        def fake_snapshot(window):
            calls.append(window)
            return _window_df(window, "601166", 1.0e8)

        ff._ths_snapshot_cache["即时"] = (pd.DataFrame([{"股票代码": "x"}]), time.time())
        with patch("data.fund_flow.get_ths_snapshot", side_effect=fake_snapshot), \
             patch("data.fund_flow.get_industry_flow_snapshot", return_value=None), \
             patch("data.fund_flow._snapshot_data_date", return_value="2026-09-24"):
            ff.capture_market_flow_snapshot()
        assert len(calls) == 4, "4 个窗口都要重新拉"

    def test_capture_saves_industry(self, temp_store):
        ind = pd.DataFrame([{"行业": "银行", "净额": 16.05, "涨跌幅": 0.5, "领涨股": "兴业银行"}])
        with patch("data.fund_flow.get_ths_snapshot", return_value=None), \
             patch("data.fund_flow.get_industry_flow_snapshot", return_value=ind), \
             patch("data.fund_flow._snapshot_data_date", return_value="2026-09-24"):
            res = ff.capture_market_flow_snapshot()
        assert res["industries"] == 1
        _, rows = db_store.load_industry_flow_snapshot()
        # 同花顺行业净额单位是"亿"，落库统一转成"元"
        assert rows[0]["net_amount"] == pytest.approx(16.05e8)


# ================================================================
# 读库路径（选股走的）
# ================================================================

class TestStoredRecord:
    def _seed(self):
        db_store.save_fund_flow_snapshot("2026-09-24", [{
            "stock_code": "601166", "stock_name": "兴业银行", "net_today": 2.61e7,
            "net_3d": 2.88e8, "net_5d": 1.49e8, "net_10d": -7.4e8,
            "amount": 1.068e9, "change_pct": 0.06, "price": 17.88}])

    def test_record_shape(self, temp_store):
        self._seed()
        rec = ff.get_stored_snapshot_record("601166")
        assert rec["股票简称"] == "兴业银行"
        assert rec["净额_今日"] == pytest.approx(2.61e7)
        assert rec["净额_10日"] == pytest.approx(-7.4e8)
        assert rec["数据日期"] == "2026-09-24", "必须能说出这份资金流是哪天的"

    def test_miss_returns_none(self, temp_store):
        self._seed()
        assert ff.get_stored_snapshot_record("000000") is None

    def test_memory_cache_avoids_repeated_db_reads(self, temp_store):
        self._seed()
        ff.get_stored_snapshot_record("601166")          # 首次：读全表
        with patch("db.store.load_fund_flow_snapshot",
                   side_effect=AssertionError("TTL 内不应再读库")) as m:
            for _ in range(50):
                rec = ff.get_stored_snapshot_record("601166")
            assert rec is not None
        assert m.call_count == 0

    def test_batch_lookup_is_fast(self, temp_store):
        """一轮选股 18 只的读库总耗时必须是毫秒级（修复前是 75.9s/只）"""
        db_store.save_fund_flow_snapshot("2026-09-24", [
            {"stock_code": f"60000{i}", "net_today": float(i)} for i in range(10)])
        ff.get_stored_snapshot_record("600000")          # 预热（读全表）
        t0 = time.time()
        for i in range(10):
            ff.get_stored_snapshot_record(f"60000{i}")
        assert time.time() - t0 < 0.5


# ================================================================
# 消费者
# ================================================================

class TestSectorRotationUsesStoredSnapshot:
    def test_prefers_stored_snapshot(self):
        """市场检测的板块轮动每轮选股都跑 —— 必须读落库快照，不再每轮打东财"""
        from core.market_detector import MarketDetector
        ind = pd.DataFrame({"行业": ["银行", "半导体", "钢铁", "传媒"],
                            "净额": [16.05, -132.0, 0.5, -0.2]})
        with patch("data.fund_flow.get_stored_industry_flow",
                   return_value=("2026-09-24", ind)), \
             patch("core.market_detector.ak.stock_sector_fund_flow_rank",
                   side_effect=AssertionError("有快照时不该再打东财")) as em:
            r = MarketDetector()._score_sector_rotation()
        assert em.call_count == 0
        assert r["max_score"] == 5
        assert "2026-09-24" in r["detail"], "detail 要能看出数据日期"

    def test_falls_back_to_eastmoney_when_no_snapshot(self):
        from core.market_detector import MarketDetector
        em = pd.DataFrame({"主力净流入-净额": [1.0, -2.0, 3.0, -4.0, 5.0]})
        with patch("data.fund_flow.get_stored_industry_flow", return_value=(None, None)), \
             patch("core.market_detector.ak.stock_sector_fund_flow_rank", return_value=em):
            r = MarketDetector()._score_sector_rotation()
        assert r["max_score"] == 5 and "东财" in r["detail"]


class TestSchedulerJob:
    def test_snapshot_job_registered(self):
        from core.scheduler import JOBS
        entry = [j for j in JOBS if j[0] == "snapshot_flow"]
        assert entry, "必须注册盘后采集任务，否则选股读不到快照"
        _, cron, mode, _ = entry[0]
        assert mode == "snapshot"
        assert cron.get("hour") == 15 and cron.get("minute") == 40

    def test_snapshot_runs_on_trading_days_only(self):
        """非交易日不该采集（数据不会更新）"""
        import core.scheduler as sch
        called = []
        with patch.object(sch, "is_trading_day", return_value=False), \
             patch.object(sch, "run_snapshot_job", side_effect=lambda *a: called.append(a)):
            sch.run_job("snapshot", "测试")
        assert called == []

    def test_snapshot_dispatched(self):
        import core.scheduler as sch
        called = []
        with patch.object(sch, "is_trading_day", return_value=True), \
             patch.object(sch, "run_snapshot_job", side_effect=lambda *a: called.append(a)):
            sch.run_job("snapshot", "测试")
        assert len(called) == 1


class TestReviewUsesRealFields:
    def test_no_dimension_scores_anywhere(self):
        """回归：`MarketSignal` 没有 `dimension_scores` 字段。

        `core/review_agent.py` 原先读它 → AttributeError 被 except 吞掉 →
        复盘报告里"市场检测14指标"**从来没出现过**，而且不报错。
        这类"字段名写错但被兜底吞掉"的静默缺失，只能靠源码断言守住。
        """
        import inspect
        import core.review_agent as ra
        src = inspect.getsource(ra)
        # 只看**代码行**，注释里解释这个坑是允许的
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith("#"))
        assert ".dimension_scores" not in code, "不得再读不存在的字段"
        assert "signal.scores" in code


# ================================================================
# 复盘兜底：确保当天快照存在（幂等）
# ================================================================

class TestEnsureCaptured:
    def test_captures_when_missing(self, temp_store):
        """复盘时若当天快照不存在则采集一次（用户要求"复盘时取好全行业数据"）"""
        with patch("data.fund_flow.get_ths_snapshot", return_value=None), \
             patch("data.fund_flow.get_industry_flow_snapshot", return_value=None), \
             patch("data.fund_flow._snapshot_data_date", return_value="2026-09-24"):
            r = ff.ensure_snapshot_captured()
        assert r["skipped"] is False
        assert r["snapshot_date"] == "2026-09-24"

    def test_skips_when_already_present(self, temp_store):
        """已存在时只做 SELECT，**绝不重复拉全市场**（这是"不一次次跑"的关键）"""
        db_store.save_fund_flow_snapshot("2026-09-24", [{"stock_code": "601166"}])
        db_store.save_industry_flow_snapshot("2026-09-24", [{"industry": "银行"}])
        with patch("data.fund_flow.capture_market_flow_snapshot",
                   side_effect=AssertionError("已存在就不该再采集")) as m, \
             patch("data.fund_flow._snapshot_data_date", return_value="2026-09-24"):
            r = ff.ensure_snapshot_captured()
        assert r["skipped"] is True
        assert m.call_count == 0

    def test_recaptures_when_industry_missing(self, temp_store):
        """个股有、行业缺 → 仍要补采（行业是板块轮动的输入）"""
        db_store.save_fund_flow_snapshot("2026-09-24", [{"stock_code": "601166"}])
        with patch("data.fund_flow.get_ths_snapshot", return_value=None), \
             patch("data.fund_flow.get_industry_flow_snapshot", return_value=None), \
             patch("data.fund_flow._snapshot_data_date", return_value="2026-09-24"):
            r = ff.ensure_snapshot_captured()
        assert r["skipped"] is False


class TestReviewHooksSnapshot:
    def test_generate_review_ensures_snapshot(self):
        """复盘入口必须保证快照存在——这是用户明确要求的分工：
        盘后复盘取全行业数据，选股只读。"""
        import inspect
        import core.review_agent as ra
        src = inspect.getsource(ra.generate_review)
        assert "ensure_snapshot_captured" in src


# ================================================================
# 数值解析：百分比字符串
# ================================================================

class TestParseNumbers:
    def test_number_parser_handles_percent(self):
        p = ff._parse_ths_number
        assert p("1.25%") == pytest.approx(1.25)
        assert p("-1.27%") == pytest.approx(-1.27)
        assert p("17.88") == pytest.approx(17.88)
        assert p("1,234.5") == pytest.approx(1234.5)
        assert p(2.5) == pytest.approx(2.5)
        # 取不到必须是 None，不能是 0 —— 0 会被读成"今天没涨没跌"
        for bad in (None, "", "--", "-", "N/A", "abc", float("nan")):
            assert p(bad) is None, f"{bad!r} 应返回 None"

    def test_amount_parser_would_mangle_percent(self):
        """对照：金额解析器遇到百分比会返回 0.0 —— 这正是原来那个静默 bug 的成因"""
        assert ff._parse_ths_amount("1.25%") == 0.0


class TestCaptureParsesPercent:
    def test_change_pct_is_not_zeroed(self, temp_store):
        """回归（2026-09-25 实测事故）：落库 5210 行 change_pct 全是 0.0。

        真实数据里 5081/5210 行涨跌幅非零，原始值是 `'20.03%'` 这种**百分比字符串**；
        采集时误用了金额解析器 `_parse_ths_amount`（它不认 `%`），
        于是全市场看起来"一点都没涨没跌"，既不报错也难察觉。
        """
        def fake_snapshot(window):
            if window != "即时":
                return None
            return pd.DataFrame([
                {"股票代码": "601166", "股票简称": "兴业银行", "净额": 2.61e7,
                 "成交额": 1.0e9, "涨跌幅": "1.25%", "最新价": 17.88},
                {"股票代码": "600519", "股票简称": "贵州茅台", "净额": -4.2e8,
                 "成交额": 2.0e9, "涨跌幅": "-2.30%", "最新价": 1237.0},
                {"股票代码": "000001", "股票简称": "平安银行", "净额": 0.0,
                 "成交额": 0.0, "涨跌幅": "--", "最新价": "--"},     # 停牌
            ])

        with patch("data.fund_flow.get_ths_snapshot", side_effect=fake_snapshot), \
             patch("data.fund_flow.get_industry_flow_snapshot", return_value=None), \
             patch("data.fund_flow._snapshot_data_date", return_value="2026-09-24"):
            ff.capture_market_flow_snapshot()

        _, data = db_store.load_fund_flow_snapshot()
        assert data["601166"]["change_pct"] == pytest.approx(1.25)
        assert data["600519"]["change_pct"] == pytest.approx(-2.30)
        assert data["601166"]["price"] == pytest.approx(17.88)
        assert data["000001"]["change_pct"] is None, "停牌股缺失必须是 None，不是 0.0"
        assert data["000001"]["price"] is None
