"""实时行情模块单元测试（全 mock，不依赖网络）"""
import time
import pytest
from unittest.mock import patch

import data.realtime as rt
from data.realtime import (
    get_realtime_quotes,
    get_index_quotes,
    _to_full_code,
    _parse_tencent_line,
    _parse_sina_line,
    _quote_cache,
    _circuit,
    QUOTES_TTL,
)


def _make_tencent_line(fields: dict, full_code: str = "sh600519", n: int = 50) -> str:
    """按索引精确构造腾讯行情行，避免手数分隔符错位"""
    parts = [""] * n
    for idx, val in fields.items():
        parts[idx] = val
    return f'v_{full_code}="{"~".join(parts)}";'


TENCENT_LINE = _make_tencent_line({
    1: "贵州茅台", 2: "600519", 3: "1297.40", 4: "1292.00", 5: "1294.00",
    30: "20260830150003", 31: "+5.40", 32: "0.39", 33: "1301.00", 34: "1290.00",
    # 单位：成交量=手、成交额=万元。389220 万元 ÷ (30000 手 × 100 股) = 1297.4 = 现价 ✓
    36: "30000", 37: "389220.00", 38: "0.13", 39: "25.3", 43: "2.1",
    44: "18000.00", 45: "21000.00", 46: "4.5",
})

# 新浪单位：成交量=股、成交额=元。3,000,000 股 = 30000 手；3,892,200,000 元 = 389220 万元
SINA_LINE = (
    'var hq_str_sh600519="贵州茅台,1294.00,1292.00,1297.40,1301.00,1290.00,'
    '1295.00,1298.00,3000000,3892200000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2026-08-30,15:00:03,00";'
)


@pytest.fixture(autouse=True)
def _clear_state():
    _quote_cache.clear()
    _circuit.update({"tencent": 0.0, "sina": 0.0})
    yield
    _quote_cache.clear()
    _circuit.update({"tencent": 0.0, "sina": 0.0})


class TestToFullCode:
    def test_prefixes(self):
        assert _to_full_code("600519") == "sh600519"
        assert _to_full_code("688981") == "sh688981"
        assert _to_full_code("000001") == "sz000001"
        assert _to_full_code("300750") == "sz300750"
        assert _to_full_code("830799") == "bj830799"

    def test_passthrough_and_strip(self):
        assert _to_full_code("SH600519") == "sh600519"
        assert _to_full_code(" sh000001 ") == "sh000001"


class TestParsers:
    def test_parse_tencent(self):
        row = _parse_tencent_line(TENCENT_LINE)
        assert row is not None
        assert row["code"] == "600519"
        assert row["name"] == "贵州茅台"
        assert row["price"] == 1297.40
        assert row["change_pct"] == 0.39
        assert row["turnover_rate"] == 0.13
        assert row["quote_time"] == "15:00:03"
        assert row["source"] == "tencent"

    def test_parse_tencent_garbage(self):
        assert _parse_tencent_line("pv_none=1;") is None
        assert _parse_tencent_line("") is None

    def test_parse_sina(self):
        row = _parse_sina_line(SINA_LINE, "sh600519")
        assert row is not None
        assert row["name"] == "贵州茅台"
        assert row["price"] == 1297.40
        assert row["change_pct"] == pytest.approx(0.42, abs=0.05)
        assert row["source"] == "sina"


class TestGetRealtimeQuotes:
    def _mock_tencent_ok(self):
        return patch.object(rt, "_fetch_batch",
                            side_effect=lambda codes, source: (
                                {c: {"full_code": c, "code": c[2:], "name": f"股{c[-4:]}",
                                     "price": 10.0, "change_pct": 1.0, "source": source}
                                 for c in codes}
                                if source == "tencent" else {}))

    def test_success_tencent(self):
        with self._mock_tencent_ok():
            result = get_realtime_quotes(["600519", "000001"])
        assert set(result.keys()) == {"600519", "000001"}
        assert result["600519"]["source"] == "tencent"

    def test_ttl_cache_no_refetch(self):
        with patch.object(rt, "_fetch_batch",
                          side_effect=lambda codes, source: {
                              c: {"full_code": c, "code": c[2:], "name": "x", "price": 1.0,
                                  "change_pct": 0.0, "source": source} for c in codes}) as mock_fetch:
            get_realtime_quotes(["600519"])
            get_realtime_quotes(["600519"])
        assert mock_fetch.call_count == 1  # 第二次命中缓存

    def test_force_refresh_bypasses_cache(self):
        with patch.object(rt, "_fetch_batch",
                          side_effect=lambda codes, source: {
                              c: {"full_code": c, "code": c[2:], "name": "x", "price": 1.0,
                                  "change_pct": 0.0, "source": source} for c in codes}) as mock_fetch:
            get_realtime_quotes(["600519"])
            get_realtime_quotes(["600519"], force_refresh=True)
        assert mock_fetch.call_count == 2

    def test_fallback_to_sina(self):
        """腾讯失败自动切换新浪"""
        def fake_fetch(codes, source):
            if source == "tencent":
                raise ConnectionError("blocked")
            return {c: {"full_code": c, "code": c[2:], "name": "新浪数据", "price": 9.9,
                        "change_pct": -1.0, "source": "sina"} for c in codes}
        with patch.object(rt, "_fetch_batch", side_effect=fake_fetch):
            result = get_realtime_quotes(["600519"])
        assert result["600519"]["source"] == "sina"

    def test_circuit_breaker_skips_failed_source(self):
        calls = []

        def fake_fetch(codes, source):
            calls.append(source)
            if source == "tencent":
                raise ConnectionError("blocked")
            return {c: {"full_code": c, "code": c[2:], "name": "s", "price": 1,
                        "change_pct": 0, "source": source} for c in codes}

        with patch.object(rt, "_fetch_batch", side_effect=fake_fetch):
            get_realtime_quotes(["600519"])   # 腾讯失败→新浪成功，腾讯进入冷却
            get_realtime_quotes(["000001"])   # 直接走新浪
        assert calls == ["tencent", "sina", "sina"]

    def test_all_sources_fail_returns_stale_cache(self):
        with patch.object(rt, "_fetch_batch",
                          side_effect=ConnectionError("down")):
            first = get_realtime_quotes(["600519"])
            assert first == {}
            # 预置过期缓存 → 双源全败时回退旧数据
            _quote_cache["sh600519"] = ({"full_code": "sh600519", "code": "600519",
                                         "name": "旧数据", "price": 1.0}, time.time() - 999)
            second = get_realtime_quotes(["600519"])
        assert second["600519"]["name"] == "旧数据"
        assert _circuit["tencent"] > time.time()


class TestIndexQuotes:
    def test_index_codes(self):
        with patch.object(rt, "_fetch_batch",
                          side_effect=lambda codes, source: {
                              c: {"full_code": c, "code": c[2:], "name": rt.INDEX_CODES.get(c, c),
                                  "price": 3000.0, "change_pct": 0.5, "source": source}
                              for c in codes}):
            idx = get_index_quotes()
        assert set(idx.keys()) == {"sh000001", "sz399001", "sz399006"}
        assert idx["sh000001"]["name"] == "上证指数"


# ===== v2（M0 0.4）单位口径 =====

class TestUnits:
    """两个数据源的成交量/成交额必须归一化到同一单位（手 / 万元）。

    实测基准（2026-09-22 sh600519，真实行情）：
      腾讯 p[36]=24573(手)、p[37]=308853(万元)
          → 308853×1e4 ÷ (24573×100) = 1256.88 ≈ 现价 1253.80 ✓
      新浪 f[8]=2457294(股)、f[9]=3088526148(元)
          → f[9] ÷ f[8] = 1256.88 ≈ 现价 ✓
    原实现把新浪的（股, 元）直接命名为 volume_hand/amount_wan，使降级到新浪时
    看盘页"成交额(亿)"虚高 1e4 倍、成交量虚高 100 倍。
    """

    def test_both_sources_agree_on_units(self):
        tx = _parse_tencent_line(TENCENT_LINE)
        sn = _parse_sina_line(SINA_LINE, "sh600519")
        assert sn["volume_hand"] == pytest.approx(tx["volume_hand"], rel=1e-6)
        assert sn["amount_wan"] == pytest.approx(tx["amount_wan"], rel=1e-6)

    def test_implied_vwap_matches_price(self):
        """隐含均价 = 成交额 ÷ 成交量(股) 必须≈现价 —— 两个源都要成立"""
        for row in (_parse_tencent_line(TENCENT_LINE),
                    _parse_sina_line(SINA_LINE, "sh600519")):
            vwap = row["amount_wan"] * 1e4 / (row["volume_hand"] * 100)
            assert vwap == pytest.approx(row["price"], rel=0.02), f"源={row['source']}"

    def test_sina_shares_converted_to_lots(self):
        row = _parse_sina_line(SINA_LINE, "sh600519")
        assert row["volume_hand"] == pytest.approx(3_000_000 / 100)      # 股 → 手
        assert row["amount_wan"] == pytest.approx(3_892_200_000 / 1e4)   # 元 → 万元

    def test_real_measured_raw_fields(self):
        """直接使用实测到的原始字段值，防止将来换源/改字段时单位再次错位"""
        tx = _parse_tencent_line(_make_tencent_line({
            1: "贵州茅台", 2: "600519", 3: "1253.80", 4: "1252.57", 5: "1252.15",
            30: "20260922161442", 31: "-0.60", 32: "-0.05",
            33: "1265.88", 34: "1248.10", 36: "24573", 37: "308853.00",
        }))
        assert tx["volume_hand"] == pytest.approx(24573)
        assert tx["amount_wan"] == pytest.approx(308853.0)
        assert tx["amount_wan"] * 1e4 / (tx["volume_hand"] * 100) == pytest.approx(1256.88, abs=0.5)

        sina_raw = ('var hq_str_sh600519="贵州茅台,1252.150,1252.570,1253.800,1265.880,'
                    '1248.100,1253.800,1254.000,2457294,3088526148.000,300,'
                    + ",".join(["0"] * 19) + ',2026-09-22,16:14:42,00";')
        sn = _parse_sina_line(sina_raw, "sh600519")
        assert sn["volume_hand"] == pytest.approx(24573, abs=1)
        assert sn["amount_wan"] == pytest.approx(308852.6, abs=1.0)


# ===== v2（M0 0.4）新鲜度标记 =====

class TestFreshness:
    def _ok_fetch(self, tag="x"):
        return patch.object(rt, "_fetch_batch",
                            side_effect=lambda codes, source: {
                                c: {"full_code": c, "code": c[2:], "name": tag, "price": 1.0,
                                    "change_pct": 0.0, "source": source} for c in codes})

    def test_fresh_fetch_not_stale(self):
        with self._ok_fetch():
            result = get_realtime_quotes(["600519"])
        assert result["600519"]["stale"] is False
        assert result["600519"]["age_seconds"] == 0.0

    def test_cache_hit_within_ttl_not_stale(self):
        with self._ok_fetch():
            get_realtime_quotes(["600519"])
            result = get_realtime_quotes(["600519"])
        assert result["600519"]["stale"] is False
        assert result["600519"]["age_seconds"] >= 0.0

    def test_expired_cache_fallback_is_flagged(self):
        with patch.object(rt, "_fetch_batch", side_effect=ConnectionError("down")):
            _quote_cache["sh600519"] = ({"full_code": "sh600519", "code": "600519",
                                         "name": "旧数据", "price": 1.0}, time.time() - 600)
            result = get_realtime_quotes(["600519"])
        q = result["600519"]
        assert q["stale"] is True, "过期缓存回退必须标记 stale，否则会被当成实时价"
        assert q["age_seconds"] == pytest.approx(600, abs=10)

    def test_stale_marking_does_not_poison_cache(self):
        """返回的是副本：调用方修改返回值不得影响缓存中的原始行"""
        with patch.object(rt, "_fetch_batch", side_effect=ConnectionError("down")):
            _quote_cache["sh600519"] = ({"full_code": "sh600519", "code": "600519",
                                         "name": "旧数据", "price": 1.0}, time.time() - 600)
            first = get_realtime_quotes(["600519"])
            first["600519"]["stale"] = False
            first["600519"]["name"] = "被改了"
            second = get_realtime_quotes(["600519"])
        assert second["600519"]["stale"] is True
        assert second["600519"]["name"] == "旧数据"
        assert "stale" not in _quote_cache["sh600519"][0]


# ================================================================
# 数据日期与新陈旧判据（2026-09-25 实测发现）
# ================================================================

class TestQuoteDate:
    """行情自带的日期必须保留。

    此前腾讯分支用 `strftime("%H:%M:%S")` —— **日期被丢掉**，对外只剩 `16:14:56`。
    实测后果（2026-09-25 休市）：17.88 是 09-24 收盘价，但工具写着"数据状态: 实时"，
    AI 助手据此回答"**当前的实时行情**：现价 17.88 元"。
    """

    def test_tencent_keeps_date(self):
        row = _parse_tencent_line(TENCENT_LINE)
        assert row["quote_date"] == "2026-08-30"
        assert row["quote_time"] == "15:00:03"
        assert row["quote_datetime"] == "2026-08-30 15:00:03"

    def test_sina_keeps_date_and_matches_tencent_shape(self):
        """两个源必须同形，否则降级切换时下游看到两种格式"""
        row = _parse_sina_line(SINA_LINE, "sh600519")
        assert row["quote_date"] == "2026-08-30"
        assert row["quote_time"] == "15:00:03"
        assert row["quote_datetime"] == "2026-08-30 15:00:03"

    def test_bad_timestamp_does_not_crash(self):
        line = _make_tencent_line({1: "X", 2: "600519", 3: "10.0", 30: "not-a-ts"})
        row = _parse_tencent_line(line)
        assert row is not None
        assert row["quote_date"] == ""

    def test_date_only_timestamp(self):
        line = _make_tencent_line({1: "X", 2: "600519", 3: "10.0", 30: "20260830"})
        assert _parse_tencent_line(line)["quote_date"] == "2026-08-30"


class TestExpectedQuoteDate:
    """`_expected_quote_date`：现在这个时刻行情**应该**给到哪一天"""

    @pytest.fixture(autouse=True)
    def _clear_ltd(self):
        rt._EXPECTED_DATE_CACHE.update({"date": None, "at": 0.0})
        yield
        rt._EXPECTED_DATE_CACHE.update({"date": None, "at": 0.0})

    def test_holiday_expects_last_trading_day(self):
        """休市日（2026-09-25 实测）应为最近交易日 09-24 —— 这样 09-24 的数据不算陈旧"""
        from datetime import datetime
        with patch("data.trading_calendar.is_trading_day", return_value=False), \
             patch("data.trading_calendar.last_trading_day",
                   return_value="2026-09-24") as m:
            got = rt._expected_quote_date(datetime(2026, 9, 25, 12, 36))
        assert got == "2026-09-24"
        m.assert_called_with(include_today=False)

    def test_trading_day_after_open_expects_today(self):
        from datetime import datetime
        with patch("data.trading_calendar.is_trading_day", return_value=True):
            got = rt._expected_quote_date(datetime(2026, 9, 24, 10, 30))
        assert got == "2026-09-24", "盘中应有当日数据，不该退回昨日"

    def test_trading_day_before_open_expects_previous(self):
        from datetime import datetime
        with patch("data.trading_calendar.is_trading_day", return_value=True), \
             patch("data.trading_calendar.last_trading_day",
                   return_value="2026-09-23"):
            got = rt._expected_quote_date(datetime(2026, 9, 24, 9, 10))
        assert got == "2026-09-23", "盘前只有昨日收盘数据可用，不该要求当日"

    def test_calendar_unavailable_returns_none(self):
        from datetime import datetime
        with patch("data.trading_calendar.is_trading_day",
                   side_effect=RuntimeError("cal down")):
            assert rt._expected_quote_date(datetime(2026, 9, 25, 12, 0)) is None


class TestDataLagging:
    """`data_lagging` 与 `stale` 是两条独立判据"""

    @pytest.fixture(autouse=True)
    def _clear_ltd(self):
        rt._EXPECTED_DATE_CACHE.update({"date": None, "at": 0.0})
        yield
        rt._EXPECTED_DATE_CACHE.update({"date": None, "at": 0.0})

    def _row(self, qd):
        return {"full_code": "sh601166", "code": "601166", "name": "兴业银行",
                "price": 17.88, "quote_date": qd, "quote_time": "15:00:00"}

    def test_latest_trading_day_is_not_lagging(self):
        with patch.object(rt, "_expected_quote_date", return_value="2026-09-24"):
            out = rt._with_freshness(self._row("2026-09-24"), 0.0)
        assert out["data_lagging"] is False
        assert out["stale"] is False

    def test_older_than_expected_is_lagging(self):
        """有更新的数据却没取到 → 必须报出来（这是日期维度的陈旧）"""
        with patch.object(rt, "_expected_quote_date", return_value="2026-09-24"):
            out = rt._with_freshness(self._row("2026-09-23"), 0.0)
        assert out["data_lagging"] is True
        assert "2026-09-23" in out["staleness_reason"]

    def test_row_without_date_is_not_accused(self):
        with patch.object(rt, "_expected_quote_date",
                          side_effect=AssertionError("无日期时不该查日历")):
            out = rt._with_freshness({"code": "601166", "price": 1.0}, 0.0)
        assert out["data_lagging"] is False

    def test_old_cache_still_stale(self):
        with patch.object(rt, "_fetch_batch", side_effect=ConnectionError("down")):
            _quote_cache["sh601166"] = (self._row("2026-09-24"), time.time() - 600)
            with patch.object(rt, "_expected_quote_date", return_value="2026-09-24"):
                res = get_realtime_quotes(["601166"])
        assert res["601166"]["stale"] is True
