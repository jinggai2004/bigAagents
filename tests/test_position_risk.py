"""持仓风控接线：`RiskEngine.suggest_levels` / `PortfolioManager.check_trailing_stops`
/ `check_stop_loss` / `run_risk_job`

背景（2026-09-28 盘点）：ATR 止损止盈、追踪止损、持仓止损检查这三件事**代码早就
写好了，但从来没有被调用过** —— `core/portfolio.py::check_stop_loss`、
`core/risk.py::calc_trailing_stop`、`get_exposure` 在库里躺着，持仓录了止损价，
系统却从不检查它有没有被打穿。本文件锁住"接线之后"的语义边界：

  1. 取不到数据 → `unavailable` + 原因，**不用固定百分比硬凑**一个看起来合理的价；
  2. 追踪止损**只上移不下移**（建议低于当前止损时静默跳过）；
  3. `check_trailing_stops()` **不自动改库** —— 改用户持仓数据必须由用户确认；
  4. 无持仓时巡检任务安静跳过，不推送、不刷日志。
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock

from core.risk import RiskEngine


# ===== 构造：K线与价格 =====

def _kline_df(n: int = 30) -> pd.DataFrame:
    """可预测的 30 根 K 线：High/Low/Close 交替，产生稳定的 ATR。"""
    hi = [25.5, 25.8, 26.0, 25.3, 25.7] * (n // 5)
    lo = [24.5, 24.8, 25.0, 24.3, 24.7] * (n // 5)
    cl = [25.0, 25.5, 25.3, 25.0, 25.5] * (n // 5)
    return pd.DataFrame({"High": hi, "Low": lo, "Close": cl})


def _engine(price: float = 25.0, bars: int = 30) -> RiskEngine:
    e = RiskEngine()
    f = MagicMock()
    f.get_stock_info.return_value = {"current_price": price}
    f.get_stock_data.return_value = _kline_df(bars)
    e._stock_fetcher = f
    return e


# ===== suggest_levels：单只标的的止损/止盈 =====

def test_suggest_levels_basic_geometry():
    e = _engine(price=25.0)
    info = e.suggest_levels("000001", entry_price=25.0)
    assert not info.get("unavailable"), info
    assert info["stop_loss"] < 25.0 < info["take_profit"]
    # 盈亏比固定 1:2 —— 这是设计约定，不该随行情漂移
    risk = 25.0 - info["stop_loss"]
    reward = info["take_profit"] - 25.0
    assert reward == pytest.approx(risk * 2, abs=0.02)
    assert info["risk_reward_ratio"] == 2.0
    assert info["risk_pct"] > 0


def test_sector_multiplier_widens_stop():
    """波动大的板块用更宽的止损带，否则正常回撤就会被打掉。"""
    e = _engine(price=25.0)
    fin = e.suggest_levels("000001", entry_price=25.0, sector="金融")
    tech = e.suggest_levels("000001", entry_price=25.0, sector="科技")
    assert fin["multiplier"] == 2.0 and tech["multiplier"] == 3.0
    assert tech["stop_loss"] < fin["stop_loss"], "科技股止损应更宽（更低）"


def test_unknown_sector_falls_back_to_default():
    e = _engine(price=25.0)
    info = e.suggest_levels("000001", entry_price=25.0, sector="查无此行业")
    assert info["multiplier"] == e.ATR_MULTIPLIER["default"]


def test_insufficient_bars_is_unavailable_not_fabricated():
    """K线不足时**不给建议**，而不是拿 7 根算出个数字糊弄过去。"""
    e = _engine(price=25.0, bars=10)
    info = e.suggest_levels("000001", entry_price=25.0)
    assert info.get("unavailable") is True
    assert "K线" in info["note"]
    assert "stop_loss" not in info


def test_no_price_is_unavailable():
    e = _engine(price=0.0)
    e._stock_fetcher.get_stock_data.return_value = pd.DataFrame()
    info = e.suggest_levels("000001", entry_price=0.0)
    assert info.get("unavailable") is True
    assert "价格" in info["note"]


def test_current_price_below_stop_is_warned():
    """现价已经跌破建议止损 → 必须明说"照此执行即为立即止损"，否则用户以为建议无效。"""
    e = _engine(price=18.0)          # 远低于 25 的入场价
    info = e.suggest_levels("000001", entry_price=25.0, current_price=18.0)
    assert not info.get("unavailable")
    assert "warning" in info and "立即止损" in info["warning"]


def test_trailing_note_reflects_profit_level():
    e = _engine(price=25.0)
    low = e.suggest_levels("000001", entry_price=25.0, current_price=25.2)
    assert "暂不上移" in low["trailing_note"]
    high = e.suggest_levels("000001", entry_price=25.0, current_price=40.0)
    assert "3×ATR" in high["trailing_note"] or "成本价" in high["trailing_note"]


# ===== check_trailing_stops：只给建议，不改库 =====

@pytest.fixture
def pm(tmp_path, monkeypatch):
    """隔离库 + mock 价格的 PortfolioManager。"""
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "risk.db")
    from core.portfolio import PortfolioManager
    m = PortfolioManager()
    f = MagicMock()
    f.get_stock_info.return_value = {"current_price": 30.0}
    f.get_stock_data.return_value = _kline_df()
    m._stock_fetcher = f
    return m


@pytest.fixture
def patched_engine(monkeypatch):
    """把 core.risk.get_risk_engine 换成价格可控的实例（check_trailing_stops 内部会调用它）。"""
    def _install(price: float = 30.0):
        e = _engine(price=price)
        monkeypatch.setattr("core.risk.get_risk_engine", lambda: e)
        return e
    return _install


def test_check_trailing_stops_does_not_modify_db(pm, patched_engine):
    """**核心语义**：巡检只给建议，绝不自动改用户的止损价。"""
    patched_engine(30.0)
    pm.open_position(code="000001", price=25.0, quantity=100, stop_loss=20.0)
    out = pm.check_trailing_stops()
    assert out, "浮盈 20% 应给出上移建议"
    assert pm.get_active_positions()[0].stop_loss == 20.0, "巡检不能自动写库"


def test_trailing_suggestion_reported_and_tightens(pm, patched_engine):
    patched_engine(30.0)
    pm.open_position(code="000001", price=25.0, quantity=100, stop_loss=20.0)
    t = pm.check_trailing_stops()[0]
    assert t["suggested_stop"] > 20.0
    assert t["kind"] == "追踪上移", "浮盈 20%（>2×ATR）才是真正的追踪止损上移"
    assert t["profit_pct"] == pytest.approx(20.0, abs=0.01)
    assert t["code"] == "000001"


def test_losing_position_kind_is_not_called_trailing(pm, patched_engine):
    """浮亏时 `calc_trailing_stop` 返回的是**基础 ATR 止损位**，不是追踪上移。

    实测踩过：600211 浮动 −0.11%（亏损）却显示"上移止损"，用户会以为系统在锁利润，
    实际只是在建议把偏宽的止损收紧。名称必须区分，否则提示本身在误导。
    """
    # ⚠️ `check_trailing_stops` 的现价取自 **PortfolioManager**（不是 RiskEngine），
    #    两边的 mock 都得改，否则一边设 24 一边还是 30，断言会指向错误的分支。
    pm._stock_fetcher.get_stock_info.return_value = {"current_price": 24.0}
    patched_engine(24.0)                      # 低于 25 的入场价 → 浮亏
    pm.open_position(code="000001", price=25.0, quantity=100, stop_loss=22.0)   # 比 ATR 位更宽
    out = pm.check_trailing_stops()
    assert len(out) == 1
    assert out[0]["kind"] == "收紧至 ATR 止损位"
    assert out[0]["kind"] != "追踪上移"
    assert out[0]["profit_pct"] < 0


def test_trailing_never_loosens_stop(pm, patched_engine):
    """建议低于当前止损时必须静默跳过 —— 追踪止损只朝一个方向走。"""
    patched_engine(30.0)
    pm.open_position(code="000001", price=25.0, quantity=100, stop_loss=29.9)
    assert pm.check_trailing_stops() == []


def test_missing_stop_loss_reported_as_setup(pm, patched_engine):
    """没设止损的持仓也应被覆盖到，类型标记为「建立止损」。"""
    patched_engine(26.0)
    pm.open_position(code="000001", price=25.0, quantity=100)   # 未设止损
    out = pm.check_trailing_stops()
    assert len(out) == 1
    assert out[0]["kind"] == "建立止损"
    assert out[0]["current_stop"] is None
    assert out[0]["suggested_stop"] > 0


def test_update_stop_loss_persists(pm, patched_engine):
    """用户点了「应用」之后才真正落库。"""
    patched_engine(30.0)
    pm.open_position(code="000001", price=25.0, quantity=100, stop_loss=20.0)
    assert pm.update_stop_loss("000001", 23.5) is True
    assert pm.get_active_positions()[0].stop_loss == 23.5


def test_update_stop_loss_unknown_code_returns_false(pm):
    assert pm.update_stop_loss("999999", 10.0) is False


def test_check_stop_loss_detects_breach(pm):
    """check_stop_loss 之前从未被调用过；这里确认它确实能发现击穿。"""
    pm.open_position(code="000001", price=30.0, quantity=100, stop_loss=31.0)
    alerts = pm.check_stop_loss()
    assert len(alerts) == 1
    assert alerts[0]["type"] == "止损"
    assert alerts[0]["code"] == "000001"


# ===== run_risk_job：调度任务的静默/推送行为 =====

def _fake_pm(stop_alerts=None, trails=None, exposure=None, n_positions=1):
    m = MagicMock()
    m.get_active_positions.return_value = [object()] * n_positions
    m.check_stop_loss.return_value = stop_alerts or []
    m.check_trailing_stops.return_value = trails or []
    m.get_exposure.return_value = exposure or {}
    return m


def _patch_structure(monkeypatch, holdings, total_equity=100_000.0, snapshot="2026-10-16"):
    """把 `run_risk_job` 里的结构体检打桩到**指定持仓**（不碰真实库/截面）。

    ⚠️ 2026-10-17 起 `run_risk_job` 不再调用 `get_exposure()`（按代码前缀粗分的
       "行业敞口"会把中石化和招行算成同一行业），改用 `core.position_risk` 的
       **真实行业 + 流动性**体检 ⇒ 相关用例必须跟着改驱动方式，而不是继续喂 exposure。
    """
    from core import position_risk as PR
    monkeypatch.setattr(PR, "build_holdings",
                        lambda prices=None, snapshot_date=None: {
                            "holdings": holdings, "snapshot_date": snapshot,
                            "missing_snapshot": [], "total_equity": total_equity,
                            "cash": 0.0, "unavailable": None})


def test_risk_job_silent_when_no_positions(monkeypatch):
    from core import scheduler
    monkeypatch.setattr("core.portfolio.get_portfolio_manager",
                        lambda: _fake_pm(n_positions=0))
    sent = {}
    monkeypatch.setattr("core.push.push_markdown",
                        lambda body, title="", **kw: sent.update(body=body) or {"ok": True})
    scheduler.run_risk_job("测试巡检")
    assert sent == {}, "无持仓不该推送"


def test_risk_job_silent_when_nothing_triggered(monkeypatch):
    from core import scheduler
    monkeypatch.setattr("core.portfolio.get_portfolio_manager", lambda: _fake_pm())
    sent = {}
    monkeypatch.setattr("core.push.push_markdown",
                        lambda body, title="", **kw: sent.update(body=body) or {"ok": True})
    scheduler.run_risk_job("测试巡检")
    assert sent == {}, "无触发不该推送"


def test_risk_job_pushes_stop_loss_breach(monkeypatch):
    from core import scheduler
    pm = _fake_pm(stop_alerts=[{"code": "000001", "name": "平安银行", "type": "止损",
                                "current_price": 9.0, "trigger_price": 10.0,
                                "loss_pct": -10.0}])
    monkeypatch.setattr("core.portfolio.get_portfolio_manager", lambda: pm)
    sent = {}
    monkeypatch.setattr("core.push.push_markdown",
                        lambda body, title="", **kw: sent.update(body=body, title=title) or {"ok": True})
    scheduler.run_risk_job("测试巡检")
    assert "止损触发" in sent.get("body", "")
    assert "000001" in sent["body"]
    assert "不构成投资建议" in sent["body"]


def test_risk_job_pushes_trailing_suggestion_without_applying(monkeypatch):
    """推送里必须写明"不会自动生效"，否则用户以为止损已经被系统改了。"""
    from core import scheduler
    pm = _fake_pm(trails=[{"code": "600519", "name": "贵州茅台", "current_stop": 1500.0,
                           "suggested_stop": 1600.0, "kind": "追踪上移"}])
    monkeypatch.setattr("core.portfolio.get_portfolio_manager", lambda: pm)
    sent = {}
    monkeypatch.setattr("core.push.push_markdown",
                        lambda body, title="", **kw: sent.update(body=body) or {"ok": True})
    scheduler.run_risk_job("测试巡检")
    body = sent.get("body", "")
    assert "600519" in body and "追踪上移" in body
    assert "不会自动生效" in body


def test_risk_job_flags_sector_overexposure(monkeypatch):
    """行业集中度超限必须出现在推送里（**用真实行业**，不是代码前缀）。"""
    from core import scheduler
    pm = _fake_pm()
    monkeypatch.setattr("core.portfolio.get_portfolio_manager", lambda: pm)
    # 同一行业（银行）合计 45%，另一行业 10%
    holdings = [
        {"code": "600036", "name": "招商银行", "market_value": 30_000.0, "industry": "银行"},
        {"code": "601398", "name": "工商银行", "market_value": 15_000.0, "industry": "银行"},
        {"code": "600519", "name": "贵州茅台", "market_value": 10_000.0, "industry": "白酒"},
    ]
    _patch_structure(monkeypatch, holdings, total_equity=100_000.0)
    sent = {}
    monkeypatch.setattr("core.push.push_markdown",
                        lambda body, title="", **kw: sent.update(body=body) or {"ok": True})
    scheduler.run_risk_job("测试巡检")
    body = sent.get("body", "")
    assert "45.0%" in body and "银行" in body, body
    assert "行业敞口" in body, body
    # 10% 的那只远未超 30% 上限 ⇒ 不得被一起报出来
    assert "贵州茅台" not in body, body


def test_risk_job_flags_single_position_and_liquidity(monkeypatch):
    """单票集中 + 流动性风险也要进推送（P2-5 新增的两类结构性问题）。"""
    from core import scheduler
    monkeypatch.setattr("core.portfolio.get_portfolio_manager", lambda: _fake_pm())
    holdings = [
        # 单票 40%（超 25%），且市值 = 1.5 天成交额（超 1.0 天警戒线）
        {"code": "002230", "name": "科大讯飞", "market_value": 40_000.0,
         "industry": "软件服务", "amount": 26_666.0},
    ]
    _patch_structure(monkeypatch, holdings, total_equity=100_000.0)
    sent = {}
    monkeypatch.setattr("core.push.push_markdown",
                        lambda body, title="", **kw: sent.update(body=body) or {"ok": True})
    scheduler.run_risk_job("测试巡检")
    body = sent.get("body", "")
    assert "单票集中" in body and "40.0%" in body, body
    assert "流动性" in body, body
    assert "1.50 个交易日" in body, body


def test_risk_job_survives_structure_failure(monkeypatch):
    """⭐ 结构体检查失败**不能**连累止损巡检（那是更硬的告警）。

    结构性守护的反面：新加的能力不许把已有的能力拉下水。
    """
    from core import scheduler
    from core import position_risk as PR
    monkeypatch.setattr("core.portfolio.get_portfolio_manager",
                        lambda: _fake_pm(stop_alerts=[{"code": "000001", "name": "平安银行",
                                                       "type": "止损", "current_price": 9.0,
                                                       "trigger_price": 10.0, "loss_pct": -10.0}]))
    def _boom(prices=None, snapshot_date=None):
        raise RuntimeError("截面库挂了")
    monkeypatch.setattr(PR, "build_holdings", _boom)
    sent = {}
    monkeypatch.setattr("core.push.push_markdown",
                        lambda body, title="", **kw: sent.update(body=body) or {"ok": True})
    scheduler.run_risk_job("测试巡检")
    assert "止损触发" in sent.get("body", ""), "结构体检查失败不该吞掉止损告警"


def test_risk_check_is_registered_and_skips_holidays():
    """任务已注册；且它必须走"非交易日跳过"的闸门（与选股/复盘同类）。"""
    import inspect
    from core import scheduler
    ids = [j[0] for j in scheduler.JOBS]
    assert "risk_check" in ids
    entry = next(j for j in scheduler.JOBS if j[0] == "risk_check")
    assert entry[2] == "risk"
    src = inspect.getsource(scheduler.run_job)
    assert '"risk"' in src, "risk 必须加入非交易日跳过清单，否则周末也会跑"
