"""资金流模块单元测试（快照降级链 + 五维评分路由）—— 全部 mock，不依赖网络"""
import time
import pandas as pd
import pytest
from unittest.mock import patch

from data.fund_flow import (
    FundFlowAkshareDataFetcher,
    _parse_ths_amount,
    _compute_snapshot_score,
    compute_five_dim_score,
    get_ths_snapshot_record,
    _ths_snapshot_cache,
    _em_circuit_open_until,
)


# ================================================================
# _parse_ths_amount
# ================================================================

class TestParseThsAmount:
    def test_yi_positive(self):
        assert _parse_ths_amount("1.09亿") == pytest.approx(1.09e8)

    def test_yi_negative(self):
        assert _parse_ths_amount("-1.27亿") == pytest.approx(-1.27e8)

    def test_wan(self):
        assert _parse_ths_amount("5034万") == pytest.approx(5.034e7)

    def test_plain_number(self):
        assert _parse_ths_amount("123.45") == pytest.approx(123.45)

    def test_numeric_passthrough(self):
        assert _parse_ths_amount(2.5) == pytest.approx(2.5)

    def test_invalid_inputs(self):
        for v in (None, "", "-", "--", "N/A", "abc"):
            assert _parse_ths_amount(v) == 0.0


# ================================================================
# 快照模式五维评分
# ================================================================

class TestSnapshotScore:
    def test_strong_inflow(self):
        result = _compute_snapshot_score({
            "净额_今日": 2e8, "净额_3日": 3e8, "净额_5日": 4e8, "净额_10日": 5e8,
        })
        # 方向6(3窗口正) + 金额6(5日>1亿) + 趋势5(加速流入) + 持续性3 = 20
        # ⚠️ 「结构」在快照源**不可评**（无单型拆分）→ **不计入分母** ⇒ 满分 20 而非 25
        assert result["score"] == 20.0
        assert result["max"] == 20
        assert result["max_total"] == 25
        assert result["unavailable_dims"] == ["结构"]
        assert result["mode"] == "snapshot"
        assert result["breakdown"]["趋势"]["desc"] == "加速流入"

    def test_persistent_outflow(self):
        result = _compute_snapshot_score({
            "净额_今日": -1e8, "净额_3日": -2e8, "净额_5日": -3e8, "净额_10日": -5e8,
        })
        # 全窗口为负: 方向0 + 金额0 + 趋势0(持续流出) + 持续性0 ；结构不计入分母
        assert result["score"] == 0.0
        assert result["breakdown"]["趋势"]["desc"] == "持续流出"

    def test_structure_dim_not_scored_and_not_deducted(self):
        """⚠️ 「不可评」≠「0 分」：结构维度必须 available=False 且**不计入分母**。

        原实现写 `dim3 = 0` 并计入 25 分总分 —— 等于把"**取不到数据**"当成
        "**该维度最差**"：每只票被系统性扣 5 分，且与逐日模式（有单型拆分）不可比。
        用户报的"贤丰控股 五维评分 0/25"里就有 5 分是这么来的；
        贵州茅台 20/25 若剔除无效维度其实是 **20/20**（满分）。
        （2026-10-10 修正，口径与市场检测 14 指标一致：缺失既不计分也不计入分母。）
        """
        strong = _compute_snapshot_score({"净额_今日": 5e8, "净额_3日": 5e8,
                                          "净额_5日": 5e8, "净额_10日": 5e8})
        weak = _compute_snapshot_score({})
        for r in (strong, weak):
            s = r["breakdown"]["结构"]
            assert s["available"] is False and s["score"] is None
            assert "不计入分母" in s["desc"]
        assert strong["score"] == strong["max"] == 20      # 满分票就是满分
        assert weak["score"] == 0 and weak["max"] == 20

    def test_score_within_bounds(self):
        for t, d3, d5, d10 in [(1e9, 0, 0, 0), (0, 0, 0, 1e9), (-1, 1, -1, 1)]:
            result = _compute_snapshot_score({
                "净额_今日": t, "净额_3日": d3, "净额_5日": d5, "净额_10日": d10,
            })
            assert 0 <= result["score"] <= result["max"] <= result["max_total"]

    def test_missing_fields_default_zero(self):
        result = _compute_snapshot_score({})
        assert result["score"] == 0.0


# ================================================================
# compute_five_dim_score 模式路由
# ================================================================

class TestFiveDimRouting:
    def test_daily_mode_passthrough(self):
        df = pd.DataFrame({"主力净流入-净额": [1e7] * 10, "超大单净流入-净额": [5e6] * 10})
        result = compute_five_dim_score(df)
        assert result["max"] == 25
        assert result["score"] > 0
        assert "error" not in result

    def test_snapshot_mode_detected(self):
        df = pd.DataFrame([{
            "股票代码": "600519", "净额_今日": 2e8, "净额_3日": 3e8,
            "净额_5日": 4e8, "净额_10日": 5e8,
        }])
        result = compute_five_dim_score(df)
        assert result.get("mode") == "snapshot"
        assert result["score"] == 20.0

    def test_missing_column_returns_error(self):
        df = pd.DataFrame({"收盘价": [10.0, 11.0]})
        result = compute_five_dim_score(df)
        assert result["score"] == 0
        assert "error" in result

    def test_empty_dataframe(self):
        result = compute_five_dim_score(pd.DataFrame())
        assert result["score"] == 0
        assert result.get("error") == "数据为空"

    def test_daily_mode_missing_super_large_col_excludes_dim(self):
        """逐日模式缺「超大单净流入-净额」列 → 结构**不可评**（分母 20）。

        ⚠️ 原实现 desc 自己写着"未评分"，score 却给 0 并计入 25 分 —— 说未评分却按
        最差分算，凭空扣 5 分（2026-10-10 与快照模式一并修正）。
        """
        df = pd.DataFrame({"主力净流入-净额": [1e7] * 10})
        r = compute_five_dim_score(df)
        struct = r["breakdown"]["结构"]
        assert struct["available"] is False and struct["score"] is None
        assert "不计入分母" in struct["desc"]
        assert r["unavailable_dims"] == ["结构"]
        assert r["max"] == 20

    def test_daily_mode_all_dims_scored_when_columns_present(self):
        df = pd.DataFrame({"主力净流入-净额": [1e7] * 10, "超大单净流入-净额": [5e6] * 10})
        r = compute_five_dim_score(df)
        assert r["breakdown"]["结构"]["available"] is True
        assert r["unavailable_dims"] == []
        assert r["max"] == 25


# ================================================================
# 同花顺快照获取（mock 接口）
# ================================================================

def _make_ths_df(window, rows):
    """构造同花顺接口的假返回"""
    if window == "即时":
        return pd.DataFrame(rows, columns=[
            "序号", "股票代码", "股票简称", "最新价", "涨跌幅",
            "换手率", "流入资金", "流出资金", "净额", "成交额"])
    return pd.DataFrame(rows, columns=[
        "序号", "股票代码", "股票简称", "最新价", "阶段涨跌幅",
        "连续换手率", "资金流入净额"])


def _ths_row(window, code, amount):
    if window == "即时":
        return [1, code, "贵州茅台", "1296.31", "0.31%", "0.12%", "9.49亿", "8.40亿", amount, "19.20亿"]
    return [1, code, "贵州茅台", "1296.31", "-0.59%", "0.49%", amount]


@pytest.fixture(autouse=True)
def _clear_ths_cache():
    _ths_snapshot_cache.clear()
    import data.fund_flow as fm
    fm._em_circuit_open_until = 0.0
    yield
    _ths_snapshot_cache.clear()
    fm._em_circuit_open_until = 0.0


class TestThsSnapshotRecord:
    def test_record_built_from_windows(self):
        def fake_fetch(symbol):
            return _make_ths_df(symbol, [_ths_row(symbol, "600519", "1.09亿" if symbol == "即时" else "8.28亿")])

        with patch("data.fund_flow.ak.stock_fund_flow_individual", side_effect=fake_fetch):
            record = get_ths_snapshot_record("600519")
        assert record is not None
        assert record["净额_今日"] == pytest.approx(1.09e8)
        assert record["净额_5日"] == pytest.approx(8.28e8)
        assert record["净额_10日"] == pytest.approx(8.28e8)
        assert record["股票简称"] == "贵州茅台"

    def test_unknown_code_returns_none(self):
        def fake_fetch(symbol):
            return _make_ths_df(symbol, [_ths_row(symbol, "600519", "1.09亿")])

        with patch("data.fund_flow.ak.stock_fund_flow_individual", side_effect=fake_fetch):
            assert get_ths_snapshot_record("000000") is None

    def test_snapshot_ttl_cache(self):
        calls = []

        def fake_fetch(symbol):
            calls.append(symbol)
            return _make_ths_df(symbol, [_ths_row(symbol, "600519", "1.09亿")])

        with patch("data.fund_flow.ak.stock_fund_flow_individual", side_effect=fake_fetch):
            get_ths_snapshot_record("600519")
            get_ths_snapshot_record("600519")
        # 4 个窗口各只拉取一次（第二次命中缓存）
        assert len(calls) == 4


# ================================================================
# 降级链整合
# ================================================================

class TestFallbackChain:
    def test_ths_fallback_df(self):
        fetcher = FundFlowAkshareDataFetcher()
        with patch("data.fund_flow.get_stored_snapshot_record",
                   return_value={"股票代码": "600519", "净额_今日": 1e8, "净额_5日": 2e8,
                                 "数据日期": "2026-09-24"}):
            df = fetcher._get_ths_fallback_df("600519")
        assert df is not None and len(df) == 1
        assert df.iloc[0]["净额_今日"] == 1e8

    def test_ths_fallback_df_none(self):
        fetcher = FundFlowAkshareDataFetcher()
        with patch("data.fund_flow.get_stored_snapshot_record", return_value=None):
            assert fetcher._get_ths_fallback_df("600519") is None

    def test_fallback_must_not_fetch_whole_market(self):
        """回归（2026-09-25）：选股路径**绝不允许**现场拉全市场快照。

        原实现走 `get_ths_snapshot_record`，而每个窗口都要拉全市场 5210 行
        （4 窗口合计实测 75.9 秒），且内存缓存只有 300 秒，选股一轮 4~5 分钟
        必然中途过期重拉 → 一次选股白花约 150 秒。
        现在必须只读落库快照（盘后 15:40 采集）。
        """
        fetcher = FundFlowAkshareDataFetcher()
        with patch("data.fund_flow.get_ths_snapshot_record",
                   side_effect=AssertionError("选股路径不得现场拉全市场快照")) as live, \
             patch("data.fund_flow.get_stored_snapshot_record",
                   return_value={"股票代码": "601166", "净额_今日": 1e8,
                                 "数据日期": "2026-09-24"}):
            df = fetcher._get_ths_fallback_df("601166")
        assert df is not None
        assert live.call_count == 0

    def test_get_fund_flow_data_falls_back_to_ths(self):
        """东财与tushare都失败时，应走同花顺快照并标记 mode=snapshot"""
        fetcher = FundFlowAkshareDataFetcher()
        with patch("data.fund_flow.ak.stock_individual_fund_flow",
                   side_effect=ConnectionError("blocked")), \
             patch.object(fetcher, "_get_ths_fallback_df") as mock_ths:
            mock_ths.return_value = pd.DataFrame([{
                "股票代码": "600519", "净额_今日": 2e8, "净额_5日": 4e8,
            }])
            # 跳过tushare重试等待：直接视为不可用
            with patch("data.fund_flow.data_source_manager") as mock_dsm:
                mock_dsm.tushare_available = False
                result = fetcher.get_fund_flow_data("600519")
        assert result["data_success"] is True
        assert result["source"] == "ths_snapshot"
        assert result["fund_flow_data"]["mode"] == "snapshot"

    def test_all_sources_fail(self):
        fetcher = FundFlowAkshareDataFetcher()
        with patch("data.fund_flow.ak.stock_individual_fund_flow",
                   side_effect=ConnectionError("blocked")), \
             patch.object(fetcher, "_get_ths_fallback_df", return_value=None), \
             patch("data.fund_flow.data_source_manager") as mock_dsm:
            mock_dsm.tushare_available = False
            result = fetcher.get_fund_flow_data("600519")
        assert result["data_success"] is False

    def test_non_chinese_symbol_rejected(self):
        fetcher = FundFlowAkshareDataFetcher()
        result = fetcher.get_fund_flow_data("AAPL")
        assert result["data_success"] is False

    def test_circuit_breaker_skips_retries(self):
        """熔断打开后：不再重试东财（0次调用），直接走降级源"""
        import data.fund_flow as fm
        fetcher = FundFlowAkshareDataFetcher()
        with patch("data.fund_flow.ak.stock_individual_fund_flow",
                   side_effect=ConnectionError("blocked")) as mock_em, \
             patch.object(fetcher, "_get_ths_fallback_df",
                          return_value=pd.DataFrame([{"股票代码": "600519", "净额_今日": 1e8}])), \
             patch("data.fund_flow.data_source_manager") as mock_dsm:
            mock_dsm.tushare_available = False
            result1 = fetcher.get_fund_flow_data("600519")   # 第一只：真实触发熔断
            assert result1["data_success"] is True
            assert mock_em.call_count == 3
            result2 = fetcher.get_fund_flow_data("000001")   # 第二只：熔断中，0次调用
            assert result2["data_success"] is True
            assert mock_em.call_count == 3  # 未增加

    def test_circuit_breaker_expires(self):
        """冷却过期后恢复对东财的探测"""
        import data.fund_flow as fm
        assert fm._em_circuit_open_until == 0.0
        # 模拟历史熔断已过期
        fm._em_circuit_open_until = time.time() - 1
        fetcher = FundFlowAkshareDataFetcher()
        with patch("data.fund_flow.ak.stock_individual_fund_flow",
                   side_effect=ConnectionError("blocked")) as mock_em, \
             patch.object(fetcher, "_get_ths_fallback_df", return_value=None), \
             patch("data.fund_flow.data_source_manager") as mock_dsm:
            mock_dsm.tushare_available = False
            result = fetcher.get_fund_flow_data("600519")
        assert result["data_success"] is False
        assert mock_em.call_count == 3
        # 失败后重新打开熔断
        assert fm._em_circuit_open_until > time.time()
