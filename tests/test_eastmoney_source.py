"""东财全市场快照直连单测（全 mock，不联网）

覆盖动机（2026-09-25 数据源需求体检）：「全市场快照」是唯一未稳定满足的需求。
定位到真因**不是东财挂了**，而是：

    该接口硬上限 100 条/页，全市场 5920 只 → 必须分 60 页；
    akshare 的 fetch_paginated_data 对 60 次请求零容错，任何一页失败即整体作废。

在 1% 抖动环境下 60 页全成的概率只有约 55%，表现就是"时好时坏"。

本文件锁死新模块的关键行为：
1. 列名与 akshare 一致（按 **f-code 键名**映射，不依赖列顺序）；
2. **逐页容错**：少数页失败仍能返回，并如实标注覆盖率；
3. **覆盖率闸门**：缺页是有偏样本（实测只拿前 10 页时全是北交所，
   涨跌家数比 0.13 vs 真实 1 量级）→ 低于阈值一律返回 None，
   宁可让调用方走保守默认值，也不能喂它一个"看起来很确定"的错答案；
4. 连续多页失败要提前止损，不把 60 页全试一遍；
5. TTL 缓存：市场检测 + 情绪一共要调 6 次，不缓存就是 360 次请求。
"""
import json

import pytest
import requests

from data import eastmoney_source as em


class _Resp:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status

    def json(self):
        return json.loads(self.text)


def _fake_get(total_pages=60, total=None, fail_pages=(), empty_pages=()):
    """模拟分页接口：每页 100 条，total 由第 1 页声明"""
    total = total if total is not None else total_pages * 100

    def fake(url, params=None, headers=None, timeout=None):
        pn = int((params or {}).get("pn", 1))
        if pn in empty_pages:
            return _Resp("")                     # 空响应体
        if pn in fail_pages:
            raise requests.exceptions.ProxyError("proxy boom")
        start = (pn - 1) * 100
        rows = [{"f12": f"{600000 + i:06d}", "f14": f"股票{i}", "f2": 10.0 + i,
                 "f3": 1.5, "f8": 2.0} for i in range(start, min(start + 100, total))]
        if not rows:
            return _Resp(json.dumps({"data": None}))     # 翻到末尾
        return _Resp(json.dumps({"data": {"total": total, "diff": rows}}))
    return fake


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    em.clear_snapshot_cache()
    # 熔断是**模块级全局态**：某个用例把它打开（连接层失败）后不清，
    # 后续用例会全部拿到 None ⇒ 一片"莫名失败"。与 cache_data 同理。
    em.reset_circuit()
    monkeypatch.setattr(em.time, "sleep", lambda *_: None)
    yield
    em.clear_snapshot_cache()
    em.reset_circuit()


# ----------------------------------------------------------------
# 列名口径：必须与 akshare 一致，下游才不用改
# ----------------------------------------------------------------

class TestSchema:
    def test_columns_match_akshare_names(self, monkeypatch):
        monkeypatch.setattr(em.requests, "get", _fake_get(total_pages=1, total=100))
        df = em.fetch_market_snapshot()
        assert df is not None
        for col in ("代码", "名称", "最新价", "涨跌幅", "换手率", "总市值", "市净率"):
            assert col in df.columns, f"缺列 {col}（下游按这些中文列名读取）"
        assert df.loc[0, "代码"] == "600000"
        assert df.loc[0, "名称"] == "股票0"

    def test_numeric_columns_converted(self, monkeypatch):
        monkeypatch.setattr(em.requests, "get", _fake_get(total_pages=1, total=100))
        df = em.fetch_market_snapshot()
        assert str(df["涨跌幅"].dtype).startswith("float")
        assert df["最新价"].iloc[0] == pytest.approx(10.0)

    def test_mapping_is_by_fcode_not_position(self, monkeypatch):
        """上游若调整字段顺序，按位置 rename 会静默错位；按 key 映射不受影响"""
        def fake(url, params=None, headers=None, timeout=None):
            return _Resp(json.dumps({"data": {"total": 1, "diff": [
                {"f14": "顺序被打乱的名称", "f3": 9.9, "f12": "300001", "f2": 5.5}]}}))
        monkeypatch.setattr(em.requests, "get", fake)
        df = em.fetch_market_snapshot()
        assert df.loc[0, "代码"] == "300001"
        assert df.loc[0, "名称"] == "顺序被打乱的名称"
        assert df.loc[0, "涨跌幅"] == pytest.approx(9.9)


# ----------------------------------------------------------------
# 逐页容错 vs 覆盖率闸门
# ----------------------------------------------------------------

class TestToleranceAndCoverage:
    def test_few_failed_pages_still_returns_data(self, monkeypatch):
        """60 页里坏 1 页 → 覆盖 99.7%，应当返回（并标 degraded）"""
        monkeypatch.setattr(em.requests, "get",
                            _fake_get(total_pages=60, fail_pages={30}))
        df = em.fetch_market_snapshot()
        assert df is not None, "缺 1 页不该让 59 页的数据全废"
        assert df.attrs["degraded"] is True
        assert 30 in df.attrs["failed_pages"]
        assert df.attrs["coverage"] > 0.9

    def test_partial_data_below_threshold_is_rejected(self, monkeypatch):
        """只拿到前 10 页（全是北交所）→ 有偏样本，必须返回 None 而不是喂给调用方

        实测：这种残缺样本算出的"涨跌家数比"是 0.13，而真实值在 1 量级
        —— 会让市场检测得出反向结论。宁可没有数据。
        """
        monkeypatch.setattr(em.requests, "get",
                            _fake_get(total_pages=60, fail_pages=set(range(11, 61))))
        assert em.fetch_market_snapshot() is None
        assert em.fetch_market_snapshot.__doc__  # 保留文档

    def test_consecutive_failures_abort_early(self, monkeypatch):
        """连续 5 页失败即判定上游不可用，不再把剩余页试完"""
        calls = []

        def fake(url, params=None, headers=None, timeout=None):
            calls.append((params or {}).get("pn"))
            raise requests.exceptions.ProxyError("down")

        monkeypatch.setattr(em.requests, "get", fake)
        assert em.fetch_market_snapshot() is None
        # 5 次连续失败 → 止损；每次失败内部还会重试 2 次
        assert len(calls) <= em._CONSECUTIVE_FAIL_LIMIT * 2

    def test_empty_body_is_diagnosable(self, monkeypatch):
        monkeypatch.setattr(em.requests, "get",
                            _fake_get(total_pages=60, empty_pages=set(range(1, 61))))
        assert em.fetch_market_snapshot() is None


# ----------------------------------------------------------------
# 缓存与兜底
# ----------------------------------------------------------------

class TestCacheAndFallback:
    def test_ttl_cache_avoids_refetch(self, monkeypatch):
        """6 个调用点共用一份快照：第二次不应再打网络"""
        calls = []
        base = _fake_get(total_pages=1, total=100)

        def counting(url, params=None, headers=None, timeout=None):
            calls.append((params or {}).get("pn"))
            return base(url, params=params, headers=headers, timeout=timeout)

        monkeypatch.setattr(em.requests, "get", counting)
        a = em.get_market_snapshot(prefer_local=False)
        n1 = len(calls)
        b = em.get_market_snapshot(prefer_local=False)
        assert n1 >= 1 and len(calls) == n1, "第二次应命中缓存"
        assert b is a

    def test_force_refresh_bypasses_cache(self, monkeypatch):
        calls = []
        base = _fake_get(total_pages=1, total=100)

        def counting(url, params=None, headers=None, timeout=None):
            calls.append(1)
            return base(url, params=params, headers=headers, timeout=timeout)

        monkeypatch.setattr(em.requests, "get", counting)
        em.get_market_snapshot(prefer_local=False)
        n1 = len(calls)
        em.get_market_snapshot(force_refresh=True)
        assert len(calls) > n1

    def test_falls_back_to_akshare(self, monkeypatch):
        """直连全失败时退回 akshare（保持原有行为，不作为唯一路径）"""
        import akshare as ak
        import pandas as pd

        def boom(url, params=None, headers=None, timeout=None):
            raise requests.exceptions.ProxyError("down")

        monkeypatch.setattr(em.requests, "get", boom)
        sentinel = pd.DataFrame([{"代码": "600000", "涨跌幅": 1.0}])
        monkeypatch.setattr(ak, "stock_zh_a_spot_em", lambda: sentinel)
        got = em.get_market_snapshot(prefer_local=False)
        assert got is sentinel

    def test_returns_none_when_both_fail(self, monkeypatch):
        import akshare as ak

        def boom(url, params=None, headers=None, timeout=None):
            raise requests.exceptions.ProxyError("down")

        monkeypatch.setattr(em.requests, "get", boom)
        monkeypatch.setattr(ak, "stock_zh_a_spot_em",
                            lambda: (_ for _ in ()).throw(RuntimeError("ak boom")))
        assert em.get_market_snapshot(prefer_local=False) is None


# ----------------------------------------------------------------
# 本地库优先（P0-1，2026-10-18）
# ----------------------------------------------------------------

class TestLocalFirst:
    """同一天的截面不该再联网拉 55 页 —— `market_snapshot` 表 15:40 已落库。

    这是"断网可跑完一次选股"的前置条件：`MarketDetector.detect()` 一次打
    3 次全市场快照，联网时 55 页/次，断网时直接把选股卡死在 Stage 0。
    """

    def _df(self, date="2026-10-16", degraded=False):
        import pandas as pd
        df = pd.DataFrame({"代码": ["600028", "000001"], "涨跌幅": [1.0, -0.8],
                           "收盘": [5.53, 11.41]})
        df.attrs.update({"data_date": date, "degraded": degraded,
                         "source": "local_market_snapshot", "coverage": 1.0})
        return df

    def test_uses_local_without_network(self, monkeypatch):
        """⭐⭐ 库里已有期望日期的截面 ⇒ **一次网络都不打**。"""
        local = self._df()
        monkeypatch.setattr(em, "_local_snapshot_df", lambda want=None: local)
        monkeypatch.setattr("data.realtime._expected_quote_date",
                            lambda: "2026-10-16")

        def _boom(*a, **k):
            raise AssertionError("本地命中后不许联网（P0-1 契约）")

        monkeypatch.setattr(em, "fetch_market_snapshot", _boom)
        em.clear_snapshot_cache()
        got = em.get_market_snapshot()
        assert got is local
        assert got.attrs["source"] == "local_market_snapshot"

    def test_stale_local_still_tries_remote(self, monkeypatch):
        """库里没有期望日期（盘中：库里是昨天）⇒ 仍联网拿实时。"""
        fresh = self._df(date="2026-10-17")

        def fake_local(want=None):
            return None if want else fresh      # 显式要"今天"时库里没有

        monkeypatch.setattr(em, "_local_snapshot_df", fake_local)
        monkeypatch.setattr("data.realtime._expected_quote_date",
                            lambda: "2026-10-17")
        called = []

        def fake_fetch(*a, **k):
            called.append(1)
            return self._df(date="2026-10-17")

        monkeypatch.setattr(em, "fetch_market_snapshot", fake_fetch)
        em.clear_snapshot_cache()
        got = em.get_market_snapshot()
        assert called and got is not None

    def test_remote_failure_falls_back_to_stale_local_and_flags_it(self, monkeypatch):
        """⭐⭐ 实时源全挂 ⇒ 回退旧库，但**必须标 degraded/stale**（降级随数据走）。"""
        import akshare as ak
        stale = self._df(date="2026-10-15")

        def fake_local(want=None):
            return None if want else stale

        monkeypatch.setattr(em, "_local_snapshot_df", fake_local)
        monkeypatch.setattr("data.realtime._expected_quote_date",
                            lambda: "2026-10-17")
        monkeypatch.setattr(em, "fetch_market_snapshot",
                            lambda *a, **k: (_ for _ in ()).throw(
                                requests.exceptions.ProxyError("down")))
        monkeypatch.setattr(ak, "stock_zh_a_spot_em",
                            lambda: (_ for _ in ()).throw(RuntimeError("ak boom")))
        em.clear_snapshot_cache()
        got = em.get_market_snapshot()
        assert got is stale
        assert got.attrs["degraded"] is True and got.attrs.get("stale") is True
        assert got.attrs["data_date"] == "2026-10-15", "滞后日期必须如实带出"

    def test_all_sources_fail_returns_none(self, monkeypatch):
        """本地也没有、联网也全挂 ⇒ None（不许编一个空壳 DataFrame）。"""
        import akshare as ak
        monkeypatch.setattr(em, "_local_snapshot_df", lambda want=None: None)
        monkeypatch.setattr(em, "fetch_market_snapshot",
                            lambda *a, **k: (_ for _ in ()).throw(
                                requests.exceptions.ProxyError("down")))
        monkeypatch.setattr(ak, "stock_zh_a_spot_em",
                            lambda: (_ for _ in ()).throw(RuntimeError("ak boom")))
        em.clear_snapshot_cache()
        assert em.get_market_snapshot() is None


# ----------------------------------------------------------------
# 与消费者的接线
# ----------------------------------------------------------------

class TestWiring:
    def test_market_detector_uses_cached_entrypoint(self):
        """市场检测原先 3 次独立调 akshare 快照 → 现在走带缓存的统一入口"""
        import inspect
        from core import market_detector
        src = inspect.getsource(market_detector)
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith("#"))
        assert "get_market_snapshot()" in code
        assert "ak.stock_zh_a_spot_em()" not in code, \
            "不应再直接调 akshare 快照（绕过缓存与容错）"
    # 原 test_sentiment_uses_cached_entrypoint 随 data/sentiment.py 一并删除：
    # 该模块生产零引用，而上面这条对 market_detector 的守护已覆盖真实生产路径。


# ----------------------------------------------------------------
# 连接层熔断（2026-10-26）
# ----------------------------------------------------------------

class TestConnectionCircuit:
    """「同一个死主机不要反复试」。

    2026-10-26 盘后实测事实：
      - `push2.eastmoney.com` / `80.push2` / `82.push2` **全集群** RemoteDisconnected，
        同时 `push2his` 返回 200 ⇒ 单集群故障（不是网络全断，也不是代理：
        绕开代理直连同样失败，两种方式报的错都是 RemoteDisconnected）；
      - 原实现对**同一个不通的主机**仍按 retries=2 重试，再叠加"连续 5 页才停"，
        15:31:07→15:32:11 白等 64s；同一晚盘后流程跑了两轮 ⇒ 90s+；
      - 结论在第一次失败时就已经确定了。

    改造后实测：第 1 次 1.3s（原 45s），第 2 次 0.0s（原 45s）。

    这两条守的是一条纪律：**连接层失败 ≠ 内容层抖动**。
    混为一谈的两种后果都很难看 —— 要么白等（把连接失败当抖动反复重试），
    要么误熔断（把一次空响应当主机挂掉，整整 10 分钟不用东财）。
    """

    def test_connection_error_not_retried_and_trips_circuit(self, monkeypatch):
        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise requests.exceptions.ConnectionError(
                "('Connection aborted.', RemoteDisconnected('Remote end closed "
                "connection without response'))")

        monkeypatch.setattr(em.requests, "get", boom)
        assert em.fetch_market_snapshot(retries=2) is None
        assert len(calls) == 1, "连接层失败必须立刻上抛，不能对死主机重试"
        assert em.circuit_remaining() > 0, "连接层失败必须打开熔断"

    def test_circuit_skips_network_entirely(self, monkeypatch):
        calls = []
        monkeypatch.setattr(em.requests, "get", lambda *a, **k: calls.append(1))
        em.trip_circuit("测试")
        assert em.fetch_market_snapshot() is None
        assert calls == [], "熔断期内不应发出任何请求（否则又白等一轮）"

    def test_content_error_still_retries_and_does_not_trip(self, monkeypatch):
        calls = []

        def bad_json(*a, **k):
            calls.append(1)
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

        monkeypatch.setattr(em.requests, "get", bad_json)
        assert em.fetch_market_snapshot(max_pages=1, retries=3) is None
        assert len(calls) == 3, "内容层失败可能是抖动，必须保留重试"
        assert em.circuit_remaining() == 0, \
            "内容层失败不能熔断，否则一次空响应就废掉东财 10 分钟"

    def test_error_classification(self):
        assert em._is_connection_level_error(requests.exceptions.Timeout())
        assert em._is_connection_level_error(requests.exceptions.ProxyError())
        assert em._is_connection_level_error(
            RuntimeError("Max retries exceeded with url"))
        assert not em._is_connection_level_error(ValueError("空响应体 (HTTP 200)"))

    def test_marker_table_in_sync_with_fund_flow(self):
        """⚠️ 两边是独立实现（fund_flow 顶部 import akshare，不能反向依赖），
        判据字符串表**必须同步**，否则同一台机器上两个模块给出相反结论。"""
        import inspect

        from data import fund_flow

        ff_src = inspect.getsource(fund_flow._is_connection_level_error)
        for m in em._CONN_ERROR_MARKERS:
            assert m in ff_src, f"fund_flow 判据缺少 {m!r}，两边已漂移"
