"""stock_data 数据层单元测试（东财详情接口熔断器回归）—— 全 mock 不依赖网络"""
import pytest
from unittest.mock import MagicMock, patch

import data.stock_data as sd
from data.stock_data import StockDataFetcher


@pytest.fixture
def fetcher():
    return StockDataFetcher()


@pytest.fixture(autouse=True)
def _reset_breaker(monkeypatch):
    monkeypatch.setattr(sd, "_em_info_circuit_open_until", 0.0)
    monkeypatch.setattr(sd.time, "sleep", lambda s: None)
    yield
    monkeypatch.setattr(sd, "_em_info_circuit_open_until", 0.0)


def _mock_sina_hq(monkeypatch):
    """屏蔽新浪 hq 兜底的真实网络调用"""
    fake_resp = MagicMock(status_code=404, text="")
    monkeypatch.setattr("requests.get", lambda *a, **kw: fake_resp)


class TestEmInfoCircuitBreaker:
    def test_no_unboundlocal_and_name_from_basic_info(self, fetcher, monkeypatch):
        """回归: 熔断变量必须以 global 声明读写——否则 UnboundLocalError 使
        全部股票名称退化为'股票+代码'(2026-08-30 浏览器实测发现的缺陷)"""
        monkeypatch.setattr(
            fetcher.data_source_manager, "get_stock_basic_info",
            lambda s: {"symbol": s, "name": "贵州茅台", "industry": "消费"})

        def _fail(**kw):
            raise ConnectionError("blocked")

        monkeypatch.setattr(sd.ak, "stock_individual_info_em", _fail)
        _mock_sina_hq(monkeypatch)

        info = fetcher._get_chinese_stock_info("600519")

        assert info["name"] == "贵州茅台"
        assert sd._em_info_circuit_open_until > 0  # 全败后熔断打开

    def test_circuit_open_skips_em_and_uses_fallback(self, fetcher, monkeypatch):
        """熔断期内**零东财调用**，名称由兜底源给出，且不续期。

        ⚠️ 2026-10-09 调整：原用例把名称来源锁在 `get_stock_basic_info` 的 mock 上，
        而该调用现在在熔断期内会被**跳过**（它内部同样先打东财，熔断期等于再白等 0.6s；
        名称在下面有新浪兜底，重复取一次纯浪费）。断言改为"行为契约"——
        零东财调用 + 名称非退化 + 熔断不续期，不锁死"名称由谁提供"这个实现细节。
        """
        basic_calls = []
        monkeypatch.setattr(
            fetcher.data_source_manager, "get_stock_basic_info",
            lambda s: basic_calls.append(s) or {"name": "不应被用到"})
        calls = []

        def _fail(**kw):
            calls.append(kw)
            raise ConnectionError("blocked")

        monkeypatch.setattr(sd.ak, "stock_individual_info_em", _fail)
        monkeypatch.setattr(sd, "_em_info_circuit_open_until",
                            sd.time.time() + 600)
        # 兜底源给出有效响应（原实现 mock 成 404，会把兜底路径本身也屏蔽掉）
        fake_resp = MagicMock(status_code=200,
                              text='var hq_str_sz000001="平安银行,10.00,9.90,10.10";')
        monkeypatch.setattr("requests.get", lambda *a, **kw: fake_resp)

        info = fetcher._get_chinese_stock_info("000001")

        assert calls == [], "熔断期内不得调用东财"
        assert basic_calls == [], "熔断期内应跳过 basic_info（它内部同样先打东财）"
        assert info["name"] == "平安银行", "名称必须由兜底源补齐"
        # 冷却期内调用不续期
        assert sd._em_info_circuit_open_until < sd.time.time() + 600.5

    def test_em_success_sets_name_and_no_breaker(self, fetcher, monkeypatch):
        """东财成功时解析名称且不打开熔断"""
        monkeypatch.setattr(
            fetcher.data_source_manager, "get_stock_basic_info",
            lambda s: None)

        def _ok(**kw):
            import pandas as pd
            return pd.DataFrame([
                {"item": "股票简称", "value": "五粮液"},
                {"item": "市盈率-动态", "value": "18.5"},
                {"item": "总市值", "value": "5000"},
            ])

        monkeypatch.setattr(sd.ak, "stock_individual_info_em", _ok)
        info = fetcher._get_chinese_stock_info("000858")
        assert info["name"] == "五粮液"
        assert info["pe_ratio"] == 18.5
        assert sd._em_info_circuit_open_until == 0.0


# ===== 财务数据口径回归（2026-09-25 实测发现）=====

class TestChineseFinancialDataPeriods:
    """深度分析的"基本面"曾拿到 **1998~2005 年**的数据，且三张报表内容相同。

    三处错误叠加：
      1. `stock_financial_abstract_ths` 的 `indicator` 传了**非法值**
         （该接口只接受 按报告期/按年度/按单季度），于是三张表返回**同一份"主要指标"**
         —— "资产负债表"里连一个资产科目都没有；
      2. 该接口是**最老在前**（1998→2025），代码却用 `head(8)`
         → 取到的是上市以来**最早**的 8 期（实测 600519 = 1998~2005）；
      3. 结果被当作"最新财报"喂给 LLM。
    """

    def test_period_key_normalization(self):
        assert sd._period_key("20260630") == "20260630"
        assert sd._period_key("2026-06-30") == "20260630"
        assert sd._period_key(1998) == "19981231"      # 同花顺年度写法
        assert sd._period_key("1998") == "19981231"

    def test_latest_records_ignore_source_row_order(self):
        """无论源是"最老在前"还是"最新在前"，都必须取到最近的 N 期"""
        import pandas as pd
        asc = pd.DataFrame({"报告期": [1998, 1999, 2000, 2001, 2002], "v": range(5)})
        desc = asc.iloc[::-1].reset_index(drop=True)
        expect = [2002, 2001, 2000]
        assert [r["报告期"] for r in sd._latest_period_records(asc, 3)] == expect
        assert [r["报告期"] for r in sd._latest_period_records(desc, 3)] == expect

    def test_excludes_earliest_when_more_than_n(self):
        """回归：有 10 期时只能给最近 8 期，最早两期必须被排除

        旧实现（最老在前的源 + head(8)）恰好会给出**最早**的 8 期。
        """
        import pandas as pd
        df = pd.DataFrame({"报告期": [str(1998 + i) for i in range(10)],
                           "货币资金": range(10)})
        got = [r["报告期"] for r in sd._latest_period_records(df, 8)]
        assert got[0] == "2007" and got[-1] == "2000"
        assert "1998" not in got and "1999" not in got

    def test_three_statements_have_distinct_subjects(self, fetcher):
        """三张报表必须是三份不同科目，不能被填成同一份'主要指标'"""
        import pandas as pd
        fake = {
            "资产负债表": pd.DataFrame({"报告日": ["20260630", "20260331"],
                                        "流动资产": [1, 2], "货币资金": [3, 4]}),
            "利润表": pd.DataFrame({"报告日": ["20260630", "20260331"],
                                    "营业总收入": [5, 6]}),
            "现金流量表": pd.DataFrame({"报告日": ["20260630", "20260331"],
                                        "经营活动产生的现金流量": [7, 8]}),
        }
        with patch("data.stock_data.ak.stock_financial_report_sina",
                   side_effect=lambda stock, symbol: fake[symbol]), \
             patch("data.stock_data.ak.stock_financial_abstract",
                   return_value=pd.DataFrame({"选项": [], "指标": []})):
            d = fetcher.get_financial_data("600519")
        assert "流动资产" in d["balance_sheet"][0]
        assert "营业总收入" in d["income_statement"][0]
        assert "经营活动产生的现金流量" in d["cash_flow"][0]

    def test_falls_back_to_ths_with_legal_indicator(self, fetcher):
        """新浪失败时退回同花顺，且**必须用它的合法 indicator**，并标注来源"""
        import pandas as pd
        ths = pd.DataFrame({"报告期": ["1998-12-31", "2026-06-30"], "净利润": [1, 2]})
        with patch("data.stock_data.ak.stock_financial_report_sina",
                   side_effect=RuntimeError("sina down")), \
             patch("data.stock_data.ak.stock_financial_abstract_ths",
                   return_value=ths) as m, \
             patch("data.stock_data.ak.stock_financial_abstract",
                   return_value=pd.DataFrame({"选项": [], "指标": []})):
            d = fetcher.get_financial_data("600519")
        # 只允许合法取值
        for c in m.call_args_list:
            assert c.kwargs.get("indicator") in ("按报告期", "按年度", "按单季度"), c
        assert d["balance_sheet"][0]["报告期"] == "2026-06-30", "兜底也要取最新期"
        assert "ths主要指标" in d["balance_sheet"][0]["_source"]
