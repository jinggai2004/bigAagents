"""限频器 / 429 退避 / 失败负缓存测试（v2 / M0 0.4）

背景：原实现对同一数据源没有任何节流——选股一次要逐股拉资金流与 K 线，
即将上线的"盘中每分钟盯盘"会把频率再提高一个数量级。限流与退避必须在数据层
统一实现，而不是指望每个调用点自觉 sleep。
"""

import time
from unittest.mock import patch

import pytest

import data.akshare_helper as ah
from data.akshare_helper import (NegativeCache, RATE_LIMIT_BASE_DELAY, is_rate_limited,
                                retry_on_failure, source_key, throttle)


@pytest.fixture(autouse=True)
def _clear_state():
    ah._last_call.clear()
    yield
    ah._last_call.clear()


class TestSourceKey:
    @pytest.mark.parametrize("url, expected", [
        ("https://push2.eastmoney.com/api/qt/clist/get", "eastmoney"),
        ("https://datacenter-web.eastmoney.com/api/x", "eastmoney"),
        ("https://d.10jqka.com.cn/v6/line/hs_000001/01/last.js", "ths"),
        ("https://hq.sinajs.cn/list=sh600519", "sina"),
        ("https://qt.gtimg.cn/q=sh600519", "tencent"),
        ("https://example.com/x", "default"),
    ])
    def test_classification(self, url, expected):
        assert source_key(url) == expected


class TestThrottle:
    def test_enforces_min_interval(self, monkeypatch):
        monkeypatch.setattr(ah, "MIN_INTERVAL_SCALE", 1.0)   # 不受离线环境变量影响
        monkeypatch.setitem(ah._MIN_INTERVAL, "default", 0.15)
        t0 = time.time()
        throttle("https://example.com/a")
        throttle("https://example.com/b")
        assert time.time() - t0 >= 0.15

    def test_no_wait_after_interval_elapsed(self, monkeypatch):
        monkeypatch.setattr(ah, "MIN_INTERVAL_SCALE", 1.0)
        monkeypatch.setitem(ah._MIN_INTERVAL, "default", 0.05)
        throttle("https://example.com/a")
        time.sleep(0.06)
        t0 = time.time()
        throttle("https://example.com/b")
        assert time.time() - t0 < 0.05

    def test_scale_zero_disables_throttle(self, monkeypatch):
        """AKSHARE_MIN_INTERVAL_SCALE=0 时完全关闭节流（离线/测试加速）"""
        monkeypatch.setattr(ah, "MIN_INTERVAL_SCALE", 0.0)
        monkeypatch.setitem(ah._MIN_INTERVAL, "default", 5.0)
        t0 = time.time()
        throttle("https://example.com/a")
        throttle("https://example.com/b")
        assert time.time() - t0 < 0.1


class TestRateLimitDetection:
    def test_status_code_429(self):
        class Resp:
            status_code = 429
        assert is_rate_limited(Resp()) is True

    @pytest.mark.parametrize("text", [
        "HTTP 429 Too Many Requests",
        "请求过于频繁，请稍后再试",
        "访问频率受限",
        "rate limit exceeded",
    ])
    def test_text_markers(self, text):
        assert is_rate_limited(RuntimeError(text)) is True

    def test_ordinary_error_is_not_rate_limited(self):
        assert is_rate_limited(RuntimeError("No value to decode")) is False


class TestRetryBackoff:
    def test_rate_limit_uses_longer_backoff(self):
        """命中限流必须退避更久：按普通失败的 1s 间隔重试只会加重封禁"""
        sleeps = []
        calls = {"n": 0}

        @retry_on_failure(max_retries=2, base_delay=0.01, backoff=2.0)
        def flaky():
            calls["n"] += 1
            raise RuntimeError("429 Too Many Requests")

        with patch("data.akshare_helper.time.sleep", side_effect=sleeps.append):
            with pytest.raises(RuntimeError):
                flaky()

        assert calls["n"] == 2
        assert sleeps and sleeps[0] >= RATE_LIMIT_BASE_DELAY, \
            f"限流退避应 ≥{RATE_LIMIT_BASE_DELAY}s，实际 {sleeps}"

    def test_ordinary_error_uses_base_delay(self):
        sleeps = []

        @retry_on_failure(max_retries=2, base_delay=0.01, backoff=2.0)
        def flaky():
            raise RuntimeError("boom")

        with patch("data.akshare_helper.time.sleep", side_effect=sleeps.append):
            with pytest.raises(RuntimeError):
                flaky()
        assert sleeps and sleeps[0] < RATE_LIMIT_BASE_DELAY


class TestNegativeCache:
    def test_marks_and_reports_dead(self):
        nc = NegativeCache(ttl=60)
        assert nc.is_dead("k") is False
        nc.mark_failure("k")
        assert nc.is_dead("k") is True
        assert nc.size() == 1

    def test_expires_after_ttl(self):
        nc = NegativeCache(ttl=0.05)
        nc.mark_failure("k")
        assert nc.is_dead("k") is True
        time.sleep(0.06)
        assert nc.is_dead("k") is False
        assert nc.size() == 0

    def test_clear_single_and_all(self):
        nc = NegativeCache(ttl=60)
        nc.mark_failure("a")
        nc.mark_failure("b")
        nc.clear("a")
        assert nc.is_dead("a") is False and nc.is_dead("b") is True
        nc.clear()
        assert nc.size() == 0
