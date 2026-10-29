# -*- coding: utf-8 -*-
"""BaoStock 全市场面板回填（`data/panel_store.py`）的语义边界。

锁住的不是"能跑"，而是几条**一旦写错就会静默造错数据**的边界：
  1. **缺失 ≠ 0**：BaoStock 用空字符串表示缺失 ⇒ `to_numeric(coerce)` 后必须是 NaN；
  2. **北交所缺口必须显式**：`bs_code()` 对 4/8/9 开头返回 None（实测 `sh.920196` 0 行、
     `bj.*` 直接报错）——猜前缀硬查会得到"0 行的成功"，比失败更危险；
  3. **断点续跑 + 去重**：5 小时的任务必须可中断；重复分片靠
     `(stock_code, trade_date)` keep=last 去重；
  4. **单只失败不中断整体**（失败原因进 progress.json，可重试）；
  5. **进度文件损坏 ⇒ 视为无进度重跑**，绝不静默当"全部完成"；
  6. 覆盖率走 P2-7 的 `data_coverage`，缺口（北交所）如实记 partial。
"""
import os

import pandas as pd
import pytest

from tests.conftest import use_template_db

import data.panel_store as PS

# ================================================================
# 1. 代码映射
# ================================================================


class TestBsCode:
    @pytest.mark.parametrize("code6,expected", [
        ("600028", "sh.600028"),
        ("601398", "sh.601398"),
        ("688981", "sh.688981"),          # 科创板也是沪市
        ("000001", "sz.000001"),
        ("300750", "sz.300750"),
        ("002230", "sz.002230"),
        ("920196", None),                  # 北交所
        ("830799", None),                  # 北交所
        ("430047", None),                  # 北交所
        ("", None), ("abc123", None), ("60002", None),   # 非法输入
    ])
    def test_mapping(self, code6, expected):
        assert PS.bs_code(code6) == expected


# ================================================================
# 2. 归一化：缺失 ≠ 0
# ================================================================


def _raw(rows: list[dict]) -> pd.DataFrame:
    cols = ["date", "code", "open", "high", "low", "close", "preclose", "volume",
            "amount", "turn", "tradestatus", "pctChg", "peTTM", "pbMRQ", "psTTM", "isST"]
    return pd.DataFrame(rows, columns=cols)


class TestNormalize:
    def test_missing_strings_become_nan_not_zero(self):
        """⭐⭐ BaoStock 用 "" 表示缺失 ⇒ 必须是 NaN（写成 0 会污染因子值）。"""
        raw = _raw([{"date": "2026-10-15", "code": "sh.600028", "open": "5.50",
                     "high": "5.60", "low": "5.48", "close": "5.53", "preclose": "5.50",
                     "volume": "1000", "amount": "553000", "turn": "0.5",
                     "tradestatus": "1", "pctChg": "0.55", "peTTM": "8.1",
                     "pbMRQ": "0.76", "psTTM": "", "isST": "0"},
                    {"date": "2026-10-16", "code": "sh.600028", "open": "", "high": "",
                     "low": "", "close": "5.53", "preclose": "5.53", "volume": "",
                     "amount": "", "turn": "", "tradestatus": "0", "pctChg": "0",
                     "peTTM": "", "pbMRQ": "", "psTTM": "", "isST": "0"}])
        out = PS.normalize_baostock_df(raw, "600028")
        assert list(out["trade_date"]) == ["2026-10-15", "2026-10-16"]
        assert out["open"].isna().tolist() == [False, True], "缺失必须是 NaN"
        assert out["pe_ttm"].isna().tolist() == [False, True]
        assert out.loc[0, "ps_ttm"] is pd.NA or pd.isna(out.loc[0, "ps_ttm"])
        # 数值列一个 0 都不许凭空出现
        assert (out[["open", "high", "low", "volume", "pe_ttm"]].fillna(-1) == 0).sum().sum() == 0

    def test_schema_and_code(self):
        raw = _raw([{"date": "2026-10-15", "code": "sz.000001", "open": "1",
                     "high": "1", "low": "1", "close": "1", "preclose": "1",
                     "volume": "1", "amount": "1", "turn": "1", "tradestatus": "1",
                     "pctChg": "0", "peTTM": "1", "pbMRQ": "1", "psTTM": "1",
                     "isST": "0"}])
        out = PS.normalize_baostock_df(raw, "000001")
        assert list(out.columns) == PS.PANEL_COLUMNS
        assert out["stock_code"].tolist() == ["000001"], "落库用 6 位代码（与快照/持仓可join）"

    def test_empty_raw_returns_empty_frame(self):
        out = PS.normalize_baostock_df(pd.DataFrame(), "600028")
        assert out.empty and list(out.columns) == PS.PANEL_COLUMNS


# ================================================================
# 3. 回填主流程（fetch_one 注入，不联网）
# ================================================================


@pytest.fixture
def panel_env(tmp_path, monkeypatch):
    monkeypatch.setattr(PS, "PANEL_DIR", str(tmp_path / "panel"))
    monkeypatch.setattr(PS, "PARTS_DIR", str(tmp_path / "panel" / "parts"))
    monkeypatch.setattr(PS, "PROGRESS_PATH", str(tmp_path / "panel" / "progress.json"))
    return tmp_path


def _fake_fetch(rows_by_code: dict[str, int | Exception]):
    def fetch(code6, start, end):
        n = rows_by_code.get(code6)
        if isinstance(n, Exception):
            raise n
        raw = _raw([{"date": f"2026-10-{10 + i:02d}", "code": "sh.600028",
                     "open": "1", "high": "1", "low": "1", "close": "1",
                     "preclose": "1", "volume": "1", "amount": "1", "turn": "1",
                     "tradestatus": "1", "pctChg": "0", "peTTM": "1", "pbMRQ": "1",
                     "psTTM": "1", "isST": "0"} for i in range(n or 0)])
        return raw
    return fetch


class TestRunBackfill:
    def test_writes_parts_and_progress(self, panel_env):
        stats = PS.run_backfill(
            codes=["600028", "000001"], start="2026-01-01", end="2026-10-17",
            fetch_one=_fake_fetch({"600028": 3, "000001": 2}),
            batch_size=10, progress_every=0)
        assert stats["fetched"] == 2 and stats["rows"] == 5 and stats["failed"] == 0
        assert stats["parts"] == 1                       # 5 行 < batch_size ⇒ 收尾落 1 片
        df = PS.load_panel()
        assert len(df) == 5 and set(df["stock_code"]) == {"600028", "000001"}
        assert os.path.exists(PS.PROGRESS_PATH)

    def test_overlapping_rerun_is_deduped_keep_last(self, panel_env):
        """续跑/重跑产生重复 ⇒ `(code, date)` keep=last，不许出现双行。"""
        for _ in range(2):
            PS.run_backfill(codes=["600028"], start="2026-01-01", end="2026-10-17",
                            fetch_one=_fake_fetch({"600028": 3}),
                            batch_size=10, progress_every=0)
        df = PS.load_panel()
        assert len(df) == 3, f"重复分片必须去重，实际 {len(df)} 行"
        assert not df.duplicated(["stock_code", "trade_date"]).any()

    def test_resume_skips_done_codes(self, panel_env):
        PS.run_backfill(codes=["600028"], start="2026-01-01", end="2026-10-17",
                        fetch_one=_fake_fetch({"600028": 3}),
                        batch_size=10, progress_every=0)
        calls = []

        def fetch(code6, start, end):
            calls.append(code6)
            return _raw([])

        stats = PS.run_backfill(codes=["600028", "000001"], start="2026-01-01",
                                end="2026-10-17", fetch_one=fetch,
                                batch_size=10, progress_every=0)
        assert calls == ["000001"], "已完成的代码必须跳过（断点续跑）"
        assert stats["skipped_done"] == 1 and stats["fetched"] == 1

    def test_single_failure_does_not_abort(self, panel_env):
        """⭐ 一只失败只记录、不中断（失败原因可查）。"""
        stats = PS.run_backfill(
            codes=["600028", "000001", "600519"],
            start="2026-01-01", end="2026-10-17",
            fetch_one=_fake_fetch({"600028": RuntimeError("连接被拒"),
                                   "000001": 2, "600519": 2}),
            batch_size=10, progress_every=0)
        assert stats["failed"] == 1 and stats["fetched"] == 2
        assert "RuntimeError" in PS._read_progress()["failed"]["600028"]
        assert len(PS.load_panel()) == 4

    def test_bj_codes_recorded_as_failed(self, panel_env):
        """北交所不在 BaoStock 覆盖内 ⇒ 计入 failed 并写明原因（不许静默消失）。"""
        stats = PS.run_backfill(codes=["920196", "600028"],
                                start="2026-01-01", end="2026-10-17",
                                fetch_one=_fake_fetch({"600028": 2}),
                                batch_size=10, progress_every=0)
        assert stats["failed"] == 1 and stats["fetched"] == 1
        assert "北交所" in PS._read_progress()["failed"]["920196"]

    def test_empty_result_is_counted_not_hidden(self, panel_env):
        """新股/长期停牌 ⇒ 0 行也是事实，必须计数（不许假装没这只）。"""
        stats = PS.run_backfill(codes=["600028", "000001"],
                                start="2026-01-01", end="2026-10-17",
                                fetch_one=_fake_fetch({"600028": 0, "000001": 2}),
                                batch_size=10, progress_every=0)
        assert stats["empty"] == 1 and stats["rows"] == 2

    def test_crash_before_flush_does_not_mark_done(self, panel_env):
        """⭐⭐ 未落盘的行**不许**记 done（2026-10-18 事故：50 只"假完成"）。

        事故链：done 在 append buffer 时写 ⇒ `_flush` 只在满 batch(200) 时落盘
        ⇒ 进程在 50 只时被杀 ⇒ progress 说"50 只完成"但分片 0 行 ⇒ 续跑永久跳过
        （"假完成"不会被重试，比"失败"更危险）。
        这里用 KeyboardInterrupt 模拟"进程被杀、收尾未执行"（`except Exception` 不捕获）。
        """
        calls = {"n": 0}

        def fetch(code6, start, end):
            calls["n"] += 1
            if calls["n"] == 3:
                raise KeyboardInterrupt("模拟进程被杀（收尾未执行）")
            return _raw([{"date": "2026-10-15", "code": "sh.600028", "open": "1",
                          "high": "1", "low": "1", "close": "1", "preclose": "1",
                          "volume": "1", "amount": "1", "turn": "1", "tradestatus": "1",
                          "pctChg": "0", "peTTM": "1", "pbMRQ": "1", "psTTM": "1", "isST": "0"}])

        with pytest.raises(KeyboardInterrupt):
            PS.run_backfill(codes=["600028", "000001", "600519"],
                            start="2026-01-01", end="2026-10-17",
                            fetch_one=fetch, batch_size=10, progress_every=1)
        prog = PS._read_progress()
        assert prog["done"] == {}, "buffer 未落盘 ⇒ done 必须为空（否则续跑跳过丢失数据）"
        assert len(PS.load_panel()) == 0


class TestOverview:
    """面板概况（落盘 JSON）—— 页面读它而非现算，必须与分片事实一致。"""

    def test_overview_matches_parts(self, panel_env, monkeypatch):
        monkeypatch.setattr(PS, "OVERVIEW_PATH", str(panel_env / "panel" / "overview.json"))
        PS.run_backfill(codes=["600028", "000001"], start="2026-01-01",
                        end="2026-10-17",
                        fetch_one=_fake_fetch({"600028": 3, "000001": 2}),
                        batch_size=10, progress_every=0)
        ov = PS.write_overview()
        assert ov["parts"] == 1 and ov["rows_disk"] == 5 and ov["codes"] == 2
        assert ov["date_min"] == "2026-10-10" and ov["date_max"] == "2026-10-12"
        assert ov["valued_codes"] == 2, "夹具 peTTM=1（非空）⇒ 两只都算有估值"
        got = PS.read_overview()
        assert got["codes"] == 2 and got["rows_disk"] == 5

    def test_read_overview_missing_returns_none(self, panel_env, monkeypatch):
        monkeypatch.setattr(PS, "OVERVIEW_PATH", str(panel_env / "panel" / "nope.json"))
        assert PS.read_overview() is None


class TestStallWatchdog:
    """BaoStock 请求**没有超时** ⇒ 单只 hang 会把整个回填永久挂住。

    2026-10-18 实测事故：14:18 之后 26 分钟零进展，父进程 1.6GB 内存一直挂着。
    看门狗判据：`stall_timeout` 秒内**一只都没完成** ⇒ 熔断（正常 25 只要 2~5 分钟）。
    """

    class _HangingExecutor:
        """前 `n_ok` 只立即完成，其余**永不完成**（模拟 hang）。"""

        def __init__(self, n_ok=2):
            self._n_ok = n_ok
            self._count = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, *args, **kwargs):
            import concurrent.futures as _f
            fut = _f.Future()
            self._count += 1
            if self._count <= self._n_ok:
                try:
                    fut.set_result(fn(*args, **kwargs))
                except Exception as e:                      # noqa: BLE001
                    fut.set_exception(e)
            return fut

    def test_hang_aborts_and_keeps_finished_rows(self, panel_env):
        """⭐⭐ 卡死熔断时：已完成的行必须**先落盘**，不做"假完成"也不白跑。"""
        stats = PS.run_backfill_parallel(
            codes=["600028", "000001", "600519", "000002"],
            start="2026-01-01", end="2026-10-17",
            worker_fetch=lambda code6: (code6, "ok", 2,
                                        [{"trade_date": "2026-10-15", "stock_code": code6,
                                          "close": 1.0}]),
            workers=1, batch_size=2, progress_every=0, stall_timeout=1,
            executor_factory=lambda w: self._HangingExecutor(n_ok=2))
        assert stats.get("stalled") is True, "1 秒零完成必须判定卡死"
        assert stats["fetched"] == 2
        df = PS.load_panel()
        assert len(df) == 2, "已完成的两只必须已落盘（熔断前 flush）"
        assert set(PS._read_progress()["done"]) == {"600028", "000001"}, \
            "未完成的不许记 done（否则续跑跳过）"


class TestProgressKeys:
    """一份进度文件承载两类语义 ⇒ 任一写路径都**不许**抹掉对方的键。"""

    def test_write_progress_preserves_dates_keys(self, panel_env):
        """⭐⭐ 按股写入（_write_progress）必须保留 done_dates/failed_dates。

        2026-10-18 事故：BaoStock 按股续跑一次，把 tushare 按日回填的 2618 天
        进度整个抹掉 ⇒ 下次按日回填从头再来（74 分钟 + 2618 次限频调用）。
        """
        PS._write_progress_dates_full({"20261016": {"rows": 5567, "at": "x"}},
                                      {"20261015": "RuntimeError: 网络"})
        PS._write_progress({"600028": {"rows": 2618, "at": "y"}}, {})
        prog = PS._read_progress()
        assert "20261016" in (prog.get("done_dates") or {}), "按股写入抹掉了按日进度"
        assert "20261015" in (prog.get("failed_dates") or {}), "按股写入抹掉了按日失败"
        assert "600028" in prog["done"]

    def test_write_dates_full_preserves_stock_keys(self, panel_env):
        """反向同理：按日写入不许抹掉按股 done/failed。"""
        PS._write_progress({"600028": {"rows": 2, "at": "y"}},
                           {"920196": "北交所"})
        PS._write_progress_dates_full({"20261016": {"rows": 1, "at": "x"}}, {})
        prog = PS._read_progress()
        assert "600028" in prog["done"] and "920196" in prog["failed"]
        assert "20261016" in prog["done_dates"]


class TestLoadPanel:
    def test_range_and_code_filter(self, panel_env):
        PS.run_backfill(codes=["600028", "000001"], start="2026-01-01",
                        end="2026-10-17",
                        fetch_one=_fake_fetch({"600028": 5, "000001": 3}),
                        batch_size=10, progress_every=0)
        df = PS.load_panel(start="2026-10-12", end="2026-10-14", codes=["600028"])
        assert set(df["stock_code"]) == {"600028"}
        assert df["trade_date"].min() >= "2026-10-12"
        assert df["trade_date"].max() <= "2026-10-14"
        assert len(df) == 3

    def test_mixed_date_formats_normalized_on_read(self, panel_env):
        """⭐⭐ 分片里混入 tushare 8 位日期 ⇒ 读取侧归一后再过滤/去重。

        字符串区间过滤只在唯一格式下才可靠：`'20240101' > '2024-01-02'`
        字符串比较为 True，语义上却是更早的一天；同股同日两种格式各一行
        也必须归一后去重成 1 行（实测 2026-10-17 按日回填踩坑）。
        """
        import pandas as pd
        os.makedirs(PS.PARTS_DIR, exist_ok=True)
        bad = pd.DataFrame([{"stock_code": "600028", "trade_date": "20240101",
                             "close": 1.0},
                            {"stock_code": "600028", "trade_date": "2024-01-01",
                             "close": 2.0}])          # 同股同日两种格式
        bad.to_parquet(os.path.join(PS.PARTS_DIR, "daily_part_0000.parquet"))
        df = PS.load_panel(start="2024-01-02", end="2024-12-31")
        # 8 位归一成 2024-01-01 ⇒ 在 [01-02, 12-31] 区间外 ⇒ 过滤后为空
        assert df.empty, "归一后 2024-01-01 不在区间内，必须被过滤掉"
        df2 = PS.load_panel()
        assert len(df2) == 1, "同股同日两种格式必须归一后去重成 1 行"
        assert df2.iloc[0]["trade_date"] == "2024-01-01"
        assert df2.iloc[0]["close"] == 2.0            # keep=last（后写的覆盖）

    def test_status_on_empty_panel(self, panel_env):
        st = PS.panel_status()
        assert st["rows"] == 0 and st["codes"] == 0


# ================================================================
# 4. 覆盖率落库（P2-7 集成）
# ================================================================


class TestRecordCoverage:
    @pytest.fixture(autouse=True)
    def db(self, tmp_path, monkeypatch):
        use_template_db(tmp_path, monkeypatch, name="panel_cov.db")
        return None

    def test_partial_because_bj_gap(self):
        """5577 只清单里只成 5100 ⇒ 必须 partial（北交所缺口不许记 ok）。"""
        stats = {"fetched": 5100, "failed": 0}
        ratio = PS.record_backfill_coverage(stats, 5577, start="2016-01-01",
                                            end="2026-10-17")
        from db import store
        row = store.load_data_coverage(dataset="baostock_backfill")[0]
        assert row["status"] == "partial" and row["got"] == 5100
        assert row["total"] == 5577
        assert ratio == pytest.approx(5100 / 5577, abs=1e-3)
        assert "北交所" in (row["note"] or "")

    def test_failed_when_nothing_fetched(self):
        PS.record_backfill_coverage({"fetched": 0, "failed": 3}, 100,
                                    start="2026-01-01", end="2026-10-17")
        from db import store
        assert store.load_data_coverage()[0]["status"] == "failed"

    def test_ok_when_full(self):
        PS.record_backfill_coverage({"fetched": 100, "failed": 0}, 100,
                                    start="2026-01-01", end="2026-10-17")
        from db import store
        assert store.load_data_coverage()[0]["status"] == "ok"


# ================================================================
# 5. 并行编排（父进程单写者；内联执行器 ⇒ 完全离线可测）
# ================================================================


class TestRunBackfillParallel:
    def _inline(self):
        return PS._InlineExecutor()

    def _fetch_ok(self, code6):
        df = PS.normalize_baostock_df(_raw([
            {"date": f"2026-10-{10 + i:02d}", "code": "sh.600028", "open": "1",
             "high": "1", "low": "1", "close": "1", "preclose": "1", "volume": "1",
             "amount": "1", "turn": "1", "tradestatus": "1", "pctChg": "0",
             "peTTM": "1", "pbMRQ": "1", "psTTM": "1", "isST": "0"} for i in range(2)]), code6)
        return code6, "ok", int(len(df)), df.to_dict("records")

    def test_parallel_writes_parts_progress_and_dedup(self, panel_env):
        codes = ["600028", "000001", "600519"]
        stats = PS.run_backfill_parallel(
            codes=codes, start="2026-01-01", end="2026-10-17",
            worker_fetch=self._fetch_ok, workers=3, batch_size=2,
            progress_every=0, executor_factory=lambda w: self._inline())
        assert stats["fetched"] == 3 and stats["failed"] == 0
        assert stats["rows"] == 6 and stats["parts"] == 2   # 6 行 / batch 2 ⇒ 2 片
        assert stats["workers"] == 3
        df = PS.load_panel()
        assert len(df) == 6 and not df.duplicated(["stock_code", "trade_date"]).any()
        assert set(df["stock_code"]) == set(codes)

    def test_worker_error_recorded_not_aborting(self, panel_env):
        """worker 返回 error（断线重登仍失败）⇒ 单只计入 failed，整体继续。"""
        def fetch(code6):
            if code6 == "000001":
                return code6, "error", 0, "重登失败: 网络接收错误"
            return self._fetch_ok(code6)

        stats = PS.run_backfill_parallel(
            codes=["600028", "000001", "600519"], start="2026-01-01",
            end="2026-10-17", worker_fetch=fetch, workers=2, batch_size=10,
            progress_every=0, executor_factory=lambda w: self._inline())
        assert stats["failed"] == 1 and stats["fetched"] == 2
        assert "重登失败" in PS._read_progress()["failed"]["000001"]

    def test_worker_exception_isolated(self, panel_env):
        """worker 直接抛异常（如登录初始化失败）⇒ 同样单只失败、不炸整体。"""
        def fetch(code6):
            if code6 == "600519":
                raise ConnectionError("登录会话已断")
            return self._fetch_ok(code6)

        stats = PS.run_backfill_parallel(
            codes=["600028", "600519"], start="2026-01-01", end="2026-10-17",
            worker_fetch=fetch, workers=2, batch_size=10,
            progress_every=0, executor_factory=lambda w: self._inline())
        assert stats["failed"] == 1 and stats["fetched"] == 1
        assert "ConnectionError" in PS._read_progress()["failed"]["600519"]

    def test_resume_and_bj_same_as_sequential(self, panel_env):
        """与顺序版同语义：已完成跳过；北交所就地 failed（不进任务队列）。"""
        PS._write_progress({"600028": {"rows": 2, "at": "x"}}, {})
        stats = PS.run_backfill_parallel(
            codes=["600028", "920196", "000001"], start="2026-01-01",
            end="2026-10-17", worker_fetch=self._fetch_ok, workers=2,
            batch_size=10, progress_every=0, executor_factory=lambda w: self._inline())
        assert stats["skipped_done"] == 1 and stats["failed"] == 1
        assert stats["fetched"] == 1
        assert "北交所" in PS._read_progress()["failed"]["920196"]

    def test_refresh_re_fetches_and_dedup(self, panel_env):
        """resume=False（--refresh）⇒ 已完成也重取；重叠分片靠 load 去重。"""
        for _ in range(2):
            PS.run_backfill_parallel(
                codes=["600028"], start="2026-01-01", end="2026-10-17",
                worker_fetch=self._fetch_ok, workers=1, batch_size=10,
                resume=False, progress_every=0,
                executor_factory=lambda w: self._inline())
        df = PS.load_panel()
        assert len(df) == 2, "重跑两次必须去重成 2 行"


class TestFetchOneBaostock:
    """`fetch_one_baostock` 的**按年分段拼装**边界（monkeypatch 假 baostock，不联网）。"""

    def test_yearly_segments_and_concat(self, monkeypatch):
        calls = []

        class _RS:
            error_code, error_msg = "0", ""

            def __init__(self, y):
                self._y = y

            def get_data(self):
                import pandas as pd
                return pd.DataFrame({"date": [f"{self._y}-06-01"], "code": ["sh.600028"]})

        class _FakeBS:
            @staticmethod
            def query_history_k_data_plus(code, fields, start_date, end_date,
                                          frequency, adjustflag):
                calls.append((code, start_date, end_date))
                return _RS(int(start_date[:4]))

        import types
        monkeypatch.setattr(sys := __import__("sys"), "modules",
                            {**sys.modules, "baostock": types.SimpleNamespace(
                                query_history_k_data_plus=_FakeBS.query_history_k_data_plus)})
        # bs_code 600028 → sh.600028；2016~2017 两年 ⇒ 2 段，边界各年归位
        df = PS.fetch_one_baostock("600028", "2016-07-01", "2017-06-30")
        assert calls == [("sh.600028", "2016-07-01", "2016-12-31"),
                         ("sh.600028", "2017-01-01", "2017-06-30")]
        assert list(df["date"]) == ["2016-06-01", "2017-06-01"]

    def test_error_propagates_with_year(self, monkeypatch):
        class _RS:
            error_code, error_msg = "10001", "网络接收错误"

            def get_data(self):
                return None

        import types
        fake = types.SimpleNamespace(
            query_history_k_data_plus=lambda *a, **k: _RS())
        monkeypatch.setitem(__import__("sys").modules, "baostock", fake)
        with pytest.raises(RuntimeError, match="2026"):
            PS.fetch_one_baostock("600028", "2026-01-01", "2026-10-17")


# ================================================================
# 离线拒绝（R4，2026-10-18 审查）
# ================================================================

class TestOfflineGuard:
    """`BIGA_OFFLINE=1` 下回填必须**明确拒绝**，不能"静默照跑"。

    BaoStock/tushare 走裸 TCP，`offline_gate` 只对它们**计数**（不阻断，
    否则 Windows 本地 IPC 会被误伤）⇒ 三个回填入口必须自己守门。
    """

    def test_run_backfill_refuses_when_offline(self, panel_env, monkeypatch):
        monkeypatch.setenv("BIGA_OFFLINE", "1")
        with pytest.raises(RuntimeError, match="BIGA_OFFLINE=1"):
            PS.run_backfill(codes=["600028"], start="2026-01-01", end="2026-02-01",
                            fetch_one=lambda *a: None)

    def test_run_backfill_tushare_refuses_when_offline(self, panel_env, monkeypatch):
        monkeypatch.setenv("BIGA_OFFLINE", "1")
        with pytest.raises(RuntimeError, match="BIGA_OFFLINE=1"):
            PS.run_backfill_tushare(trade_days=["20260105"],
                                    fetch_one_day=lambda d: None)

    def test_parallel_refuses_when_offline(self, panel_env, monkeypatch):
        monkeypatch.setenv("BIGA_OFFLINE", "1")
        with pytest.raises(RuntimeError, match="BIGA_OFFLINE=1"):
            PS.run_backfill_parallel(codes=["600028"], start="2026-01-01",
                                     end="2026-02-01", worker_fetch=lambda *a: None)
