"""腾讯 HTTP 直连数据源单测（全部用 mock，不联网）

覆盖动机：2026-09-25 排查"akshare 极其不稳定"，日志显示
**586/587 条取数失败全部来自 akshare 的 stock_zh_a_hist_tx**，而它是历史 K 线的主路径。
读源码确认三个脆弱点：demjson 解码空响应（报 "No value to decode"）、
每次多打一次上市年份请求、按年循环任一失败即整体失败。

本文件锁死新直连模块的关键行为：
1. 代码/日期归一化口径；
2. **空响应体**要报出可诊断的原因（而不是"解码失败"）、且重试后再放弃；
3. 复权口径**不得互相回退**（请求 qfq 却只有 day → 视为无数据，不许静默变成不复权）；
4. 翻页语义：该端点忽略起始日期，只返回"截至 end 的最近 count 根"，
   因此必须**向后翻页**才能覆盖长区间；
5. 列口径与 source_manager 一致：volume = 第 6 字段 × 100（是手，不是成交额）、amount 为 NaN。
"""
import json

import pandas as pd
import pytest

from data import tencent_source as tx


# ----------------------------------------------------------------
# 归一化
# ----------------------------------------------------------------

class TestNormalize:
    def test_to_tx_code_markets(self):
        assert tx.to_tx_code("600519") == "sh600519"
        assert tx.to_tx_code("000001") == "sz000001"
        assert tx.to_tx_code("300750") == "sz300750"
        assert tx.to_tx_code("430047") == "bj430047"
        assert tx.to_tx_code("830799") == "bj830799"

    def test_to_tx_code_passthrough_and_invalid(self):
        assert tx.to_tx_code("sh600519") == "sh600519"      # 已带前缀原样返回
        assert tx.to_tx_code("SZ000001") == "sz000001"      # 大写也认
        assert tx.to_tx_code("60051") == "60051"            # 位数不对 → 原样
        assert tx.to_tx_code("") == ""

    def test_norm_date_variants(self):
        assert tx._norm_date("20240101") == "2024-01-01"
        assert tx._norm_date("2024-01-01") == "2024-01-01"
        assert tx._norm_date("2024/01/01") == "2024-01-01"
        import datetime
        assert tx._norm_date(datetime.date(2024, 1, 1)) == "2024-01-01"


# ----------------------------------------------------------------
# mock 工具
# ----------------------------------------------------------------

def _rows(dates, base=10.0, vol=1234):
    """构造该端点的行：[date, open, close, high, low, vol_hand, {}, ...]"""
    out = []
    for i, d in enumerate(dates):
        p = base + i
        out.append([d, f"{p}", f"{p + 1}", f"{p + 2}", f"{p - 1}", f"{vol}", {}, "0.1"])
    return out


def _days(start, n):
    return [d.strftime("%Y-%m-%d")
            for d in pd.date_range(start, periods=n, freq="D")]


class _FakeResp:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status


def _make_get(pages, log=None, key="qfqday"):
    """pages: {结束日期: 行列表}；按 param 里的 end 返回对应页。

    端点会忽略起始日期、只按 end 返回最近 count 根 —— 这里如实模拟该语义。
    """
    def fake_get(url, params=None, headers=None, timeout=None):
        param = (params or {}).get("param", "")
        parts = param.split(",")
        end = parts[3] if len(parts) > 3 else ""
        if log is not None:
            log.append(end)
        rows = pages.get(end)
        if rows is None:
            # 未定义该结束日 → 返回空体（模拟上游被限流/无响应）
            return _FakeResp("")
        return _FakeResp(json.dumps({"code": 0, "data": {
            "sh600519": {key: rows}}}))
    return fake_get


# ----------------------------------------------------------------
# 空响应体：必须报出可诊断的原因，且重试后放弃
# ----------------------------------------------------------------

class TestEmptyBody:
    def test_empty_body_is_retried_then_reported(self, monkeypatch):
        """akshare 在这里报 'No value to decode'（看不出原因）；我们要报"空响应体\""""
        calls = []

        def fake_get(url, params=None, headers=None, timeout=None):
            calls.append(1)
            return _FakeResp("")            # 一直是空体

        monkeypatch.setattr(tx.requests, "get", fake_get)
        monkeypatch.setattr(tx.time, "sleep", lambda *_: None)

        assert tx.fetch_kline("600519", "2024-01-01", "2024-12-31") is None
        # 空响应体必须重试若干次后才放弃，不能一次就判定失败
        assert len(calls) >= 3, f"空响应体必须重试，实际只请求了 {len(calls)} 次"

    def test_empty_body_error_message_is_actionable(self, monkeypatch):
        monkeypatch.setattr(tx.requests, "get",
                            lambda *a, **k: _FakeResp("", status=502))
        monkeypatch.setattr(tx.time, "sleep", lambda *_: None)
        with pytest.raises(ValueError) as ei:
            tx._request_page("sh600519", "2024-12-31", "qfq", 1, 1)
        msg = str(ei.value)
        assert "空响应体" in msg and "502" in msg, msg

    def test_recovers_when_retry_succeeds(self, monkeypatch):
        """前两次空体、第三次成功 → 应当恢复，而不是整体失败"""
        state = {"n": 0}

        def fake_get(url, params=None, headers=None, timeout=None):
            state["n"] += 1
            if state["n"] < 3:
                return _FakeResp("")
            return _FakeResp(json.dumps({"code": 0, "data": {
                "sh600519": {"qfqday": _rows(["2024-01-02", "2024-01-03"])}}}))

        monkeypatch.setattr(tx.requests, "get", fake_get)
        monkeypatch.setattr(tx.time, "sleep", lambda *_: None)
        df = tx.fetch_kline("600519", "2024-01-01", "2024-01-05")
        assert df is not None and len(df) == 2


# ----------------------------------------------------------------
# 复权口径不得互相回退
# ----------------------------------------------------------------

class TestAdjustStrictness:
    def test_qfq_request_refuses_other_adjusted_series(self, monkeypatch):
        """请求前复权、响应里只有 `hfqday`（另一种已复权口径）→ 视为无数据。

        静默回退会把"前复权"悄悄变成"后复权"，是最难发现的一类错误。
        """
        def fake_get(url, params=None, headers=None, timeout=None):
            return _FakeResp(json.dumps({"code": 0, "data": {"sh600519": {
                "day": _rows(["2024-01-02"]),
                "hfqday": _rows(["2024-01-02"]),
            }}}))
        monkeypatch.setattr(tx.requests, "get", fake_get)
        monkeypatch.setattr(tx.time, "sleep", lambda *_: None)
        assert tx.fetch_kline("600519", "2024-01-01", "2024-12-31",
                              adjust="qfq") is None

    def test_no_adjustment_events_uses_day(self, monkeypatch):
        """响应只有 `day`（该股无分红/送转）→ day 本来就是前复权序列，应当采用。

        实测依据：sh688981（中芯国际）请求 qfq 只返回 `day`，而同为科创板的
        sh688111 正常返回 `qfqday`。若一律判"无数据"，会白丢这一整类个股。
        """
        monkeypatch.setattr(tx.requests, "get", _make_get(
            {"2024-12-31": _rows(["2024-01-02", "2024-01-03"])}, key="day"))
        monkeypatch.setattr(tx.time, "sleep", lambda *_: None)
        df = tx.fetch_kline("600519", "2024-01-01", "2024-12-31", adjust="qfq")
        assert df is not None and len(df) == 2

    def test_matching_adjust_key_is_used(self, monkeypatch):
        monkeypatch.setattr(tx.requests, "get", _make_get(
            {"2024-12-31": _rows(["2024-01-02"])}, key="qfqday"))
        df = tx.fetch_kline("600519", "2024-01-01", "2024-12-31", adjust="qfq")
        assert df is not None and len(df) == 1

    def test_unadjusted_uses_day_key(self, monkeypatch):
        monkeypatch.setattr(tx.requests, "get", _make_get(
            {"2024-12-31": _rows(["2024-01-02"])}, key="day"))
        df = tx.fetch_kline("600519", "2024-01-01", "2024-12-31", adjust="")
        assert df is not None and len(df) == 1


# ----------------------------------------------------------------
# 翻页：接口忽略起始日期，只给"截至 end 的最近 count 根"
# ----------------------------------------------------------------

class TestPaging:
    def test_pages_backwards_until_start_is_covered(self, monkeypatch):
        """第一页只覆盖到 6 月，必须再向前翻一页才能拿到 1~5 月

        端点语义：返回"截至 end 的最近 count 根"，所以第 1 页的**最后一天就是请求的 end**，
        而它可能远没有覆盖到目标起始日。
        """
        # 第 1 页（end=2024-12-31）：最早只到 2024-06-03 → 覆盖不到 2024-01-01
        page1 = _rows(["2024-06-03", "2024-07-01", "2024-08-01", "2024-09-02",
                       "2024-10-08", "2024-11-01", "2024-12-02", "2024-12-31"])
        # 第 2 页：cursor = 2024-06-03 − 1 天 = 2024-06-02；最早 2023-12-29 ≤ start → 停止翻页
        page2 = _rows(["2023-12-29", "2024-01-02", "2024-02-01", "2024-03-01",
                       "2024-04-01", "2024-05-06"])
        pages = {"2024-12-31": page1, "2024-06-02": page2}
        seen = []
        monkeypatch.setattr(tx.requests, "get", _make_get(pages, log=seen))

        df = tx.fetch_kline("600519", "2024-01-01", "2024-12-31", adjust="qfq")
        assert seen == ["2024-12-31", "2024-06-02"], f"翻页顺序不对: {seen}"
        assert df is not None
        # 裁剪到 [start, end]：2023-12-29 被排除
        assert df["date"].min().strftime("%Y-%m-%d") == "2024-01-02"
        assert df["date"].max().strftime("%Y-%m-%d") == "2024-12-31"
        assert len(df) == len(page1) + len(page2) - 1

    def test_stops_at_page_limit(self, monkeypatch):
        """接口行为若变成"永远返回同一批更早的数据"，翻页必须有上限而死不掉"""
        rows = _rows(["2000-01-01", "2000-01-02"])
        monkeypatch.setattr(tx.requests, "get",
                            lambda *a, **k: _FakeResp(json.dumps(
                                {"code": 0, "data": {"sh600519": {"qfqday": rows}}})))
        df = tx.fetch_kline("600519", "1990-01-01", "2024-12-31", adjust="qfq")
        # 关键是没有死循环；能返回就已经过关
        assert df is None or len(df) >= 1

    def test_result_is_filtered_and_deduped_to_range(self, monkeypatch):
        rows = _rows(_days("2023-12-01", 40))          # 跨过区间两端
        monkeypatch.setattr(tx.requests, "get",
                            lambda *a, **k: _FakeResp(json.dumps(
                                {"code": 0, "data": {"sh600519": {"qfqday": rows}}})))
        df = tx.fetch_kline("600519", "2024-01-01", "2024-01-10", adjust="qfq")
        assert df is not None
        assert df["date"].min() >= pd.Timestamp("2024-01-01")
        assert df["date"].max() <= pd.Timestamp("2024-01-10")
        assert df["date"].is_unique, "同一日期不得重复"


# ----------------------------------------------------------------
# 列口径（必须与 source_manager 现有实现一致）
# ----------------------------------------------------------------

class TestSchema:
    def test_columns_units_and_na(self, monkeypatch):
        monkeypatch.setattr(tx.requests, "get", _make_get(
            {"2024-12-31": _rows(["2024-01-02"], base=10.0, vol=1234)}))
        df = tx.fetch_kline("600519", "2024-01-01", "2024-12-31", adjust="qfq")
        assert list(df.columns) == ["date", "open", "close", "high", "low",
                                    "volume", "amount"]
        assert str(df["date"].dtype).startswith("datetime64")
        # 第 6 字段是"手"，换算成"股"
        assert df.iloc[0]["volume"] == 1234 * 100
        # 该源不提供成交额：显式 NaN，不许编造
        assert pd.isna(df.iloc[0]["amount"])

    def test_no_data_returns_none(self, monkeypatch):
        monkeypatch.setattr(tx.requests, "get", _make_get({
            "2024-12-31": _rows(["2025-06-01", "2025-06-02"])}))   # 全在区间之外
        assert tx.fetch_kline("600519", "2024-01-01", "2024-12-31") is None


# ----------------------------------------------------------------
# 与上层降级链的接线
# ----------------------------------------------------------------

class TestSourceManagerIntegration:
    def test_direct_source_is_tier_zero(self):
        """直连必须在 akshare 之前被尝试（源码顺序断言，防被改回去）"""
        import inspect
        from data import source_manager
        src = inspect.getsource(source_manager.DataSourceManager.get_stock_hist_data)
        assert "tencent_source" in src
        assert src.index("tencent_source") < src.index("import akshare"), \
            "腾讯直连应作为第一层，akshare 退居兜底"
