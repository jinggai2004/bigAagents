# -*- coding: utf-8 -*-
"""持仓组合层体检：集中度（单票/行业）+ 流动性（P2-5，2026-10-17）。

背景：`RiskEngine.MAX_SINGLE_STOCK_WEIGHT` / `MAX_SECTOR_WEIGHT` 早就定义好了，
但**从未对真实持仓做过任何校验**；页面上那块"行业敞口"是按**代码前缀**
（60→"金融/公用"、00→"金融/地产"…）粗分的 —— 与真实行业无关，
把中石化和招商银行算成同一个"行业"，集中度结论本身就是错的。

本文件锁住的是**判定力**，不是"页面不崩"：
  1. 阈值**只认 `RiskEngine` 一处**（改常量即改报告，防手工重抄）；
  2. **正常组合一条都不许报**（误报会把面板变成噪声，用户就不看了）；
  3. 行业必须按**真实行业**合并 —— 同前缀不同行业**不得**并在一起；
  4. 缺数据**不可评**：行业取不到就不进行业统计；成交额取不到就单列，
     **绝不当成 0**（0 会算出"市值 ÷ 0 = 无穷大"，把"没数据"伪装成"最坏结论"）；
  5. 严重度刻意用 `high`/`medium`，**不是** `core.portfolio.SEVERITY_WARN`
     —— 后者会被 `level_sanity_blocking()` 解读为"必须人工确认才能保存"。
"""

import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import use_template_db

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(_ROOT, "ui", "pages", "6_💼_持仓.py")
sys.path.insert(0, _ROOT)

DAY = "2026-10-16"


# ================================================================
# 1. 阈值单源
# ================================================================

class TestThresholdsComeFromRiskEngine:
    def test_limits_are_read_from_risk_engine(self, monkeypatch):
        """⭐ 结构性守护：改 `RiskEngine` 常量必须立刻反映到报告里。"""
        from core.risk import RiskEngine
        from core import position_risk as P
        monkeypatch.setattr(RiskEngine, "MAX_SINGLE_STOCK_WEIGHT", 0.42)
        monkeypatch.setattr(RiskEngine, "MAX_LIQUIDITY_DAYS", 7.0)
        lim = P._limits()
        assert lim["single"] == 0.42, "阈值被抄了第二份 ⇒ 改配置会静默不生效"
        assert lim["liquidity_days"] == 7.0

    def test_module_does_not_hardcode_the_numbers(self):
        """⚠️ 源码级守护：本模块**不许**出现 0.25 / 0.30 这类字面量阈值。"""
        import io
        src = io.open(os.path.join(_ROOT, "core", "position_risk.py"),
                      encoding="utf-8").read()
        body = src.split("def _limits()")[1].split("return {")[1].split("}")[0]
        assert "0.25" not in body and "0.30" not in body and "1.0" not in body, body
        assert "RiskEngine." in body

    def test_severity_is_not_the_blocking_one(self):
        """严重度不得复用 `SEVERITY_WARN`，否则会被当成"保存前必须确认"。"""
        from core import position_risk as P
        from core.portfolio import SEVERITY_WARN
        assert P.SEV_HIGH != SEVERITY_WARN and P.SEV_MEDIUM != SEVERITY_WARN


# ================================================================
# 2. 集中度
# ================================================================

def _h(code, mv, industry=None, amount=None, name=""):
    return {"code": code, "name": name, "market_value": mv,
            "industry": industry, "amount": amount}


class TestConcentration:
    def test_healthy_portfolio_reports_nothing(self):
        """⭐⭐ 鉴别力关键：分散在 4 个行业、每只 ~12% ⇒ **一条都不许报**。"""
        from core import position_risk as P
        h = [_h("600036", 120_000, "银行"), _h("600519", 120_000, "白酒"),
             _h("002230", 120_000, "软件服务"), _h("300750", 120_000, "电池")]
        issues = P.concentration_issues(h, 1_000_000.0)
        assert issues == [], issues

    def test_single_over_limit_is_flagged(self):
        from core import position_risk as P
        h = [_h("002230", 400_000, "软件服务", name="科大讯飞"),
             _h("600519", 100_000, "白酒")]
        out = [i for i in P.concentration_issues(h, 1_000_000.0)
               if i["kind"] == "single"]
        assert len(out) == 1 and out[0]["code"] == "002230"
        assert out[0]["weight"] == pytest.approx(0.4)
        assert out[0]["limit"] == P._limits()["single"]
        assert "单票集中" in out[0]["message"]

    def test_sector_is_merged_by_real_industry_not_code_prefix(self):
        """⭐⭐ 鉴别力关键：三只**同为 60 开头**但行业不同 ⇒ 不得并成一个行业。

        这正是原实现（按代码前缀归类）会犯的错：600036/600519/600887 会被
        全部算成"金融/公用"。按真实行业它们分属银行/白酒/乳品，各自都没超限。
        """
        from core import position_risk as P
        h = [_h("600036", 300_000, "银行"), _h("600519", 300_000, "白酒"),
             _h("600887", 300_000, "乳品")]
        out = [i for i in P.concentration_issues(h, 1_000_000.0)
               if i["kind"] == "sector"]
        assert out == [], out

    def test_sector_over_limit_is_flagged_with_members(self):
        from core import position_risk as P
        h = [_h("600036", 200_000, "银行"), _h("601398", 150_000, "银行"),
             _h("600519", 100_000, "白酒")]
        out = [i for i in P.concentration_issues(h, 1_000_000.0)
               if i["kind"] == "sector"]
        assert len(out) == 1 and out[0]["sector"] == "银行"
        assert out[0]["weight"] == pytest.approx(0.35)
        # 必须能回答"到底是哪几只"——否则用户没法动手改
        assert set(out[0]["members"]) == {"600036", "601398"}

    def test_missing_industry_is_not_merged_into_other(self):
        """⚠️ 行业缺的持仓**不进**行业统计（也不并进"其他"凑数）。"""
        from core import position_risk as P
        h = [_h("600036", 200_000, "银行"), _h("601398", 200_000, "银行"),
             _h("002230", 300_000, None)]        # 缺行业，且金额最大
        out = P.concentration_issues(h, 1_000_000.0)
        sec = [i for i in out if i["kind"] == "sector"]
        assert [i["sector"] for i in sec] == ["银行"]
        assert sec[0]["weight"] == pytest.approx(0.4)      # 只算银行那两只
        assert not any(i.get("sector") in ("其他", "未知", "None") for i in out)

    def test_no_equity_means_no_guess(self):
        """总资产 ≤0 ⇒ 无法评估 ⇒ 返回空，**不**用持仓成本顶替。"""
        from core import position_risk as P
        h = [_h("002230", 999_999.0, "软件服务")]
        assert P.concentration_issues(h, 0.0) == []
        assert P.concentration_issues(h, None) == []


# ================================================================
# 3. 流动性
# ================================================================

class TestLiquidity:
    def test_liquid_position_reports_nothing(self):
        """⭐ 鉴别力：市值只占日成交额 5% ⇒ 不报（否则面板天天报警）。"""
        from core import position_risk as P
        h = [_h("600028", 50_000, "石油行业", amount=1_000_000)]
        issues, unavail = P.liquidity_issues(h)
        assert issues == [] and unavail == []

    def test_illiquid_position_is_flagged(self):
        from core import position_risk as P
        h = [_h("002230", 300_000, "软件服务", amount=100_000, name="科大讯飞")]
        issues, unavail = P.liquidity_issues(h)
        assert len(issues) == 1
        assert issues[0]["severity"] == P.SEV_HIGH
        assert issues[0]["days"] == pytest.approx(3.0)
        assert "3.00 个交易日" in issues[0]["message"]
        assert unavail == []

    def test_moderately_tight_is_medium(self):
        from core import position_risk as P
        h = [_h("002230", 50_000, "软件服务", amount=200_000)]     # 0.25 天
        issues, _ = P.liquidity_issues(h)
        assert len(issues) == 1 and issues[0]["severity"] == P.SEV_MEDIUM

    def test_missing_amount_is_unavailable_not_zero(self):
        """⭐⭐ **最关键的一条**：成交额缺失 ⇒ 不可评，**不是**"流动性极差"。

        若把它当 0，`市值 ÷ 0` 会变成无穷大 ⇒ 报告出一个**凭空的**最高危结论。
        这正是本项目反复踩的"把缺失伪装成数值"。
        """
        from core import position_risk as P
        h = [_h("002230", 300_000, "软件服务", amount=None)]
        issues, unavail = P.liquidity_issues(h)
        assert issues == [], "缺成交额不得推出任何流动性结论"
        assert unavail == ["002230"], "必须**显式**列出来，否则等于默示'没问题'"

    def test_zero_and_negative_amount_treated_as_missing(self):
        from core import position_risk as P
        h = [_h("A", 100_000, "x", amount=0), _h("B", 100_000, "x", amount=-5.0)]
        issues, unavail = P.liquidity_issues(h)
        assert issues == [] and set(unavail) == {"A", "B"}


# ================================================================
# 4. 汇总报告
# ================================================================

class TestReport:
    def test_report_shape_and_caliber(self):
        from core import position_risk as P
        h = [_h("600036", 400_000, "银行"), _h("600519", 100_000, "白酒", amount=10**9)]
        rep = P.position_risk_report(h, 1_000_000.0)
        assert rep["evaluated"] is True
        assert rep["position_value"] == pytest.approx(500_000.0)
        assert rep["position_ratio"] == pytest.approx(0.5)
        assert rep["cash_ratio"] == pytest.approx(0.5)
        assert rep["top_position"]["code"] == "600036"
        assert rep["limits"]["single"] == P._limits()["single"]
        # 银行 40% > 30% ⇒ 既有单票也有行业告警
        kinds = sorted(i["kind"] for i in rep["issues"])
        assert kinds == ["sector", "single"]
        assert rep["unavailable"] is None

    def test_unavailable_equity_is_declared(self):
        from core import position_risk as P
        rep = P.position_risk_report([_h("600036", 400_000, "银行")], 0.0)
        assert rep["evaluated"] is False
        assert rep["unavailable"], "无法评估必须写明原因，不能静默给 0"
        assert rep["position_ratio"] is None
        assert rep["issues"] == []


# ================================================================
# 5. 只读取数（真实库夹具 + 注入行情，不联网）
# ================================================================

class TestBuildHoldings:
    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        use_template_db(tmp_path, monkeypatch, name="posrisk.db")
        return tmp_path

    def _seed(self):
        from core.portfolio import PortfolioManager
        pm = PortfolioManager(initial_cash=1_000_000.0)
        pm.open_position(code="600028", price=5.53, quantity=10_000, name="中国石化")
        pm.open_position(code="600519", price=1500.0, quantity=100, name="贵州茅台")
        return pm

    def _snapshot(self):
        from db.store import save_market_snapshot
        save_market_snapshot(DAY, [
            {"code": "600028", "name": "中国石化", "industry": "石油行业",
             "price": 5.53, "amount": 1_463_995_556.0},
            {"code": "600519", "name": "贵州茅台", "industry": "白酒",
             "price": 1500.0, "amount": 2_000_000.0},
        ])

    def test_reads_real_industry_and_amount_from_snapshot(self, db):
        from core import position_risk as P
        self._seed()
        self._snapshot()
        built = P.build_holdings(prices={"600028": 5.53, "600519": 1500.0})
        assert built["snapshot_date"] == DAY, "必须返回**实际数据日期**（页面要展示）"
        by = {h["code"]: h for h in built["holdings"]}
        assert by["600028"]["industry"] == "石油行业"
        assert by["600028"]["amount"] == pytest.approx(1_463_995_556.0)
        assert by["600519"]["industry"] == "白酒"
        assert built["missing_snapshot"] == []

    def test_position_missing_from_snapshot_is_listed_not_guessed(self, db):
        """持仓不在当日截面里（新股/未覆盖）⇒ 行业与成交额都是 None，并**显式列出**。"""
        from core import position_risk as P
        pm = self._seed()
        pm.open_position(code="601127", price=100.0, quantity=100, name="赛力斯")
        self._snapshot()
        built = P.build_holdings(prices={"600028": 5.53, "600519": 1500.0,
                                         "601127": 100.0})
        assert built["missing_snapshot"] == ["601127"]
        rep = P.position_risk_report(built["holdings"], built["total_equity"])
        assert rep["missing_industry"] == ["601127"], "缺行业必须列出来，不许静默略过"
        assert rep["unavailable_liquidity"] == ["601127"]

    def test_injected_prices_are_used_no_network(self, db, monkeypatch):
        """⭐ 结构性守护：传了 prices 就**不许**联网取价。"""
        from core.portfolio import PortfolioManager
        def _boom(self, code):
            raise AssertionError("不该联网取价（已注入 prices）")
        monkeypatch.setattr(PortfolioManager, "_get_current_price", _boom)
        self._seed()
        self._snapshot()
        from core import position_risk as P
        built = P.build_holdings(prices={"600028": 5.53, "600519": 1500.0})
        assert {h["code"] for h in built["holdings"]} == {"600028", "600519"}

    def test_no_positions_returns_empty_report(self, db):
        from core import position_risk as P
        rep = P.evaluate_positions(prices={})
        assert rep["positions"] == []
        assert rep["evaluated"] is False

    def test_end_to_end_flags_concentration_and_liquidity(self, db):
        """端到端：一只占 30% 且市值 = 3 天成交额 ⇒ 单票 + 流动性双告警。

        ⚠️ 仓位以**库里的持仓 + 组合初始资金 100 万**为准（`build_holdings()` 走
           `get_portfolio_manager()` 单例，不是本用例 new 出来的那个实例）——
           所以 300,000 元持仓 ⇒ 占净值 30%。这一点很反直觉，写在这里免得下次又算错。
        """
        from core.portfolio import PortfolioManager
        from db.store import save_market_snapshot
        from core import position_risk as P
        PortfolioManager().open_position(code="002230", price=5.0, quantity=60_000,
                                         name="科大讯飞")
        save_market_snapshot(DAY, [{"code": "002230", "name": "科大讯飞",
                                    "industry": "软件服务", "price": 5.0,
                                    "amount": 100_000.0}])
        rep = P.evaluate_positions(prices={"002230": 5.0})
        assert rep["position_ratio"] == pytest.approx(0.30), rep["position_ratio"]
        kinds = {i["kind"] for i in rep["issues"]}
        assert "single" in kinds and "liquidity" in kinds, rep["issues"]
        assert rep["snapshot_date"] == DAY


# ================================================================
# 6. 页面接线
# ================================================================

class TestPositionPageStructure:
    PAGE = PAGE

    @pytest.fixture(autouse=True)
    def page_env(self, tmp_path, monkeypatch):
        """⚠️ UI 夹具三件套（沿用 `test_ui_pages_smoke.py` 的口径）：
        换库 + 关轮询/联网 + **清 `st.cache_data`**（进程级全局缓存，
        不清会出现"单跑通过、串行失败"，而报错长得像"页面没渲染"）。"""
        import streamlit as st
        use_template_db(tmp_path, monkeypatch, name="posrisk_ui.db")
        monkeypatch.setenv("BIGA_UI_POLL_SECONDS", "3600")
        monkeypatch.setenv("BIGA_UI_OFFLINE", "1")
        st.cache_data.clear()
        yield
        st.cache_data.clear()

    def _texts(self, at) -> str:
        blob = []
        for attr in ("markdown", "caption", "info", "warning", "error", "success"):
            blob += [str(getattr(x, "value", "")) for x in getattr(at, attr) or []]
        return "\n".join(blob)

    def _seed(self):
        """一只占净值 30%、市值 = 3 天成交额的持仓（组合初始资金 100 万）。"""
        from core.portfolio import PortfolioManager
        from db.store import save_market_snapshot
        PortfolioManager().open_position(code="002230", price=5.0, quantity=60_000,
                                         name="科大讯飞")
        save_market_snapshot(DAY, [{"code": "002230", "name": "科大讯飞",
                                    "industry": "软件服务", "price": 5.0,
                                    "amount": 100_000.0}])

    def _run(self, monkeypatch, prices):
        monkeypatch.setattr("data.realtime.get_realtime_quotes",
                            lambda codes: {c: {"price": prices.get(c), "name": ""}
                                           for c in codes})
        from core.portfolio import PortfolioManager
        monkeypatch.setattr(PortfolioManager, "_get_current_price",
                            lambda self, code: prices.get(code, 0.0))
        at = AppTest.from_file(self.PAGE, default_timeout=120)
        at.run()
        return at

    def test_structure_panel_shows_concentration_and_liquidity(self, monkeypatch):
        self._seed()
        at = self._run(monkeypatch, {"002230": 5.0})
        assert not at.exception, [str(e.value) for e in at.exception]
        text = self._texts(at)
        assert "持仓结构体检" in text
        assert "单票集中" in text, "超 25% 的单票必须显示"
        assert "流动性风险" in text, "市值≫日成交额必须显示"
        # ⚠️ 行业名出现在 **metric 标签**与表格里，`_texts()` 两个都不收
        #    ⇒ 必须单独从 at.metric / at.dataframe 取（别顺手把断言写成恒真）。
        labels = " ".join(str(m.label) for m in (at.metric or []))
        assert "软件服务" in labels, labels
        df_cols = [c for t in (at.dataframe or [])
                   for c in (list(t.value.columns) if hasattr(t.value, "columns") else [])]
        assert "行业" in df_cols and "退出需(天)" in df_cols, df_cols

    def test_industry_is_not_from_code_prefix(self, monkeypatch):
        """⭐⭐ **鉴别力关键（页面级）**：行业必须来自估值截面，不是代码前缀。

        原实现按前缀把 60 开头都叫"金融/公用" —— 若哪天真退化成那样，
        002230（30 开头）会被叫"科技/创业板"。这里直接断言页面上出现的
        是截面的真实行业名。
        """
        self._seed()
        at = self._run(monkeypatch, {"002230": 5.0})
        labels = " ".join(str(m.label) for m in (at.metric or []))
        assert "科技/创业板" not in labels, "又退回了按代码前缀粗分"
        assert "软件服务" in labels, labels

    def test_missing_data_is_reported_honestly(self, monkeypatch):
        """截面里没有该持仓 ⇒ 必须写"不可评"，**不能**静默当作没问题。"""
        from core.portfolio import PortfolioManager
        PortfolioManager(initial_cash=100_000.0).open_position(
            code="002230", price=5.0, quantity=20_000, name="科大讯飞")
        at = self._run(monkeypatch, {"002230": 5.0})
        assert not at.exception, [str(e.value) for e in at.exception]
        text = self._texts(at)
        assert "不可评" in text
        assert "不等于" in text or "未并进" in text

    def test_normal_portfolio_shows_pass_line(self, monkeypatch):
        """⭐ 鉴别力：分散、充裕流动性的组合应显示"结构体检通过"，不刷告警。"""
        from core.portfolio import PortfolioManager
        from db.store import save_market_snapshot
        pm = PortfolioManager(initial_cash=1_000_000.0)
        pm.open_position(code="600028", price=5.53, quantity=10_000, name="中国石化")
        pm.open_position(code="600519", price=1500.0, quantity=100, name="贵州茅台")
        save_market_snapshot(DAY, [
            {"code": "600028", "name": "中国石化", "industry": "石油行业",
             "price": 5.53, "amount": 1_463_995_556.0},
            {"code": "600519", "name": "贵州茅台", "industry": "白酒",
             "price": 1500.0, "amount": 2_000_000_000.0},
        ])
        at = self._run(monkeypatch, {"600028": 5.53, "600519": 1500.0})
        assert not at.exception, [str(e.value) for e in at.exception]
        text = self._texts(at)
        assert "结构体检通过" in text, text
        assert "单票集中" not in text and "流动性风险" not in text
