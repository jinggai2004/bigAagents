"""并发预取模块测试（零网络：fetcher 全为假对象）

背景（2026-10-09 P0）：因子精筛逐票串行取数 9.6s/只，把候选域锁死在 40 只。
本模块把取数批量并发化（实测单只 1.6s，候选域 18 → 81 只）。

这些用例锁住三件容易出错的事：
1. **真的并发**（不是换个写法的串行——否则收益为 0）；
2. **限速生效**（对上游礼貌；限速失效可能被封，属于静默风险）；
3. **并发只在安全路径上**：预取必须 `skip_em=True`，绝不把 akshare
   （py_mini_racer 不支持多线程）拖进线程池。
"""
import threading
import time

import pandas as pd
import pytest

from data import prefetch as pf


class _FakeFetcher:
    """记录调用并模拟单次耗时；线程安全计数。"""

    def __init__(self, kline_delay=0.05, info_delay=0.05, fail_kline=(), fail_info=()):
        self._kline_delay, self._info_delay = kline_delay, info_delay
        self._fail_kline, self._fail_info = set(fail_kline), set(fail_info)
        self.kline_calls, self.info_calls = [], []
        self.skip_em_flags = []
        self.skip_valuation_flags = []
        self._lock = threading.Lock()
        self.max_concurrent = 0
        self._live = 0

    def _enter(self):
        with self._lock:
            self._live += 1
            self.max_concurrent = max(self.max_concurrent, self._live)

    def _exit(self):
        with self._lock:
            self._live -= 1

    def get_stock_data(self, code, period=None):
        self._enter()
        try:
            with self._lock:
                self.kline_calls.append(code)
            time.sleep(self._kline_delay)
            if code in self._fail_kline:
                return {"error": "模拟K线失败"}
            # ⚠️ 形状必须是实现认可的 DataFrame（有 columns 且非空）——
            #    首版返回字符串，被实现正确判为"取数失败"，于是 6 个用例一起红。
            #    假数据的形状要和真实取数一致，否则测的是幻影。
            return pd.DataFrame({"date": ["2026-10-09"], "close": [10.0]})
        finally:
            self._exit()

    def get_stock_info(self, code, skip_em=False, skip_valuation=False):
        self._enter()
        try:
            with self._lock:
                self.info_calls.append(code)
                self.skip_em_flags.append(skip_em)
                self.skip_valuation_flags.append(skip_valuation)
            time.sleep(self._info_delay)
            if code in self._fail_info:
                raise RuntimeError("模拟信息失败")
            return {"name": f"N-{code}", "pe_ratio": None if skip_valuation else 10.0}
        finally:
            self._exit()


class TestConcurrency:

    def test_runs_concurrently_not_serially(self):
        """⚠️ 核心：必须真的并发。30 只 × 0.05s 串行需 1.5s，并发应远小于它。"""
        f = _FakeFetcher(kline_delay=0.05, info_delay=0.05)
        codes = [f"6000{i:02d}" for i in range(30)]
        t0 = time.time()
        res = pf.prefetch_market_data(codes, fetcher=f, max_workers=8, rate_per_sec=1000)
        elapsed = time.time() - t0
        assert len(res.klines) == 30
        assert elapsed < 0.8, f"耗时 {elapsed:.2f}s，疑似没并发"
        assert f.max_concurrent > 1, "从未出现并发执行"

    def test_respects_max_workers(self):
        f = _FakeFetcher(kline_delay=0.02, info_delay=0.02)
        pf.prefetch_market_data([f"6000{i:02d}" for i in range(40)],
                                fetcher=f, max_workers=4, rate_per_sec=1000)
        assert f.max_concurrent <= 4, f"并发度 {f.max_concurrent} 超过 max_workers"

    def test_rate_limiter_throttles(self):
        """⚠️ 限速必须真的生效（对上游礼貌；失效属于静默风险）"""
        f = _FakeFetcher(kline_delay=0, info_delay=0)
        t0 = time.time()
        # 10 只 × 2 请求 = 20 次请求，限速 20/s → 至少 ≈0.95s
        pf.prefetch_market_data([f"6000{i:02d}" for i in range(10)],
                                fetcher=f, max_workers=8, rate_per_sec=20)
        elapsed = time.time() - t0
        assert elapsed >= 0.85, f"限速未生效（{elapsed:.2f}s 跑完 20 次请求）"

    def test_rate_limiter_interval_is_global(self):
        rl = pf._RateLimiter(rate_per_sec=50)      # 间隔 20ms
        t0 = time.time()
        for _ in range(5):
            rl.wait()
        assert time.time() - t0 >= 0.07, "限速器应在全局范围内串行化发起时刻"


class TestSafety:

    def test_prefetch_requires_skip_em(self):
        """⚠️ 并发安全前提：预取必须传 skip_em=True（避开 akshare 的线程不安全性）"""
        f = _FakeFetcher()
        pf.prefetch_market_data(["600519", "000001"], fetcher=f, rate_per_sec=1000)
        assert f.skip_em_flags == [True, True], "预取把 akshare 路径带进了并发"

    def test_prefetch_also_skips_valuation(self):
        """⭐ 预取必须传 skip_valuation=True（2026-10-22 实测事故）。

        百度 `stock_zh_valuation_baidu` 是**逐票**接口。选股一次要取 100+ 只，
        并发打过去会被限流并返回 HTML（`Expecting value: line 1 column 1`），
        每只失败重试把选股从 6 分钟拖到 **43 分钟**（10-22 09:25 实测 2592s）。

        PE/PB 改由评分侧读 `market_snapshot`（PE 覆盖 65.5% / PB 91%，零网络）。
        这条用例防止有人把 skip_valuation 删掉——删了不会报错，只会慢。
        """
        f = _FakeFetcher()
        pf.prefetch_market_data(["600519", "000001"], fetcher=f, rate_per_sec=1000)
        assert f.skip_valuation_flags == [True, True], \
            "预取又去打百度估值接口了 —— 会被限流，选股会慢几十倍"

    def test_info_can_be_disabled(self):
        """只要行情时不应白跑一遍基本信息（每个信息请求都是成本）"""
        f = _FakeFetcher()
        res = pf.prefetch_market_data(["600519"], fetcher=f, with_info=False,
                                      rate_per_sec=1000)
        assert res.klines and not f.info_calls


class TestFailureHonesty:

    def test_kline_failure_is_recorded_with_reason(self):
        f = _FakeFetcher(fail_kline={"600002"})
        res = pf.prefetch_market_data(["600001", "600002"], fetcher=f, rate_per_sec=1000)
        assert "600001" in res.klines
        assert "600002" in res.failed and "K线失败" in res.failed["600002"]

    def test_info_failure_keeps_kline(self):
        """信息失败不该丢掉已取到的行情（PE 因子会标 unknown，K 线仍可用）"""
        f = _FakeFetcher(fail_info={"600001"})
        res = pf.prefetch_market_data(["600001"], fetcher=f, rate_per_sec=1000)
        assert "600001" in res.klines
        assert "600001" not in res.infos
        assert any("基本信息失败" in w for w in res.warnings)

    def test_empty_input_returns_empty(self):
        f = _FakeFetcher()
        res = pf.prefetch_market_data([], fetcher=f)
        assert res.klines == {} and res.failed == {} and f.kline_calls == []

    def test_duplicate_codes_are_fetched_once(self):
        f = _FakeFetcher()
        pf.prefetch_market_data(["600519", "600519", " 600519 "], fetcher=f,
                                rate_per_sec=1000)
        assert f.kline_calls.count("600519") == 1


class TestProgress:

    def test_progress_counts_reach_total(self):
        f = _FakeFetcher()
        seen = []
        pf.prefetch_market_data([f"6000{i:02d}" for i in range(5)], fetcher=f,
                                rate_per_sec=1000,
                                progress_cb=lambda d, t: seen.append((d, t)))
        assert seen[-1] == (5, 5)

    def test_progress_callback_exception_does_not_break(self):
        f = _FakeFetcher()

        def boom(_d, _t):
            raise RuntimeError("UI 炸了")
        res = pf.prefetch_market_data(["600519"], fetcher=f, rate_per_sec=1000,
                                      progress_cb=boom)
        assert res.klines, "进度回调异常不得影响预取结果"

    def test_summary_mentions_failures(self):
        f = _FakeFetcher(fail_kline={"600002"})
        res = pf.prefetch_market_data(["600001", "600002"], fetcher=f, rate_per_sec=1000)
        assert "成功" in res.summary() and "失败" in res.summary()
