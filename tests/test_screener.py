"""测试选股引擎 screener.py（mock 数据源）"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from core.screener import StockScreener


# ===== 夹具 =====

@pytest.fixture
def screener():
    s = StockScreener()
    mock_info = MagicMock()
    mock_info.get_stock_info.return_value = {
        "name": "测试股份", "current_price": 25.0,
        "change_percent": 3.5, "pe_ratio": 15.0,
    }
    s._stock_fetcher = mock_info
    mock_flow = MagicMock()
    # 提供结构合理的模拟数据，而非空列表 → 避免 compute_five_dim_score 空 DataFrame 失败
    mock_flow.get_fund_flow_data.return_value = {
        "data_success": True,
        "fund_flow_data": {
            "data": [
                {"主力净流入": 5000000, "超大单净流入": 2000000, "date": "2026-01-02"},
                {"主力净流入": 3000000, "超大单净流入": 1500000, "date": "2026-01-03"},
                {"主力净流入": 7000000, "超大单净流入": 3000000, "date": "2026-01-06"},
                {"主力净流入": -2000000, "超大单净流入": -1000000, "date": "2026-01-07"},
                {"主力净流入": 4000000, "超大单净流入": 1800000, "date": "2026-01-08"},
                {"主力净流入": 1000000, "超大单净流入": 500000, "date": "2026-01-09"},
                {"主力净流入": 6000000, "超大单净流入": 2500000, "date": "2026-01-10"},
                {"主力净流入": 8000000, "超大单净流入": 3500000, "date": "2026-01-13"},
                {"主力净流入": 3000000, "超大单净流入": 1200000, "date": "2026-01-14"},
                {"主力净流入": 2000000, "超大单净流入": 800000, "date": "2026-01-15"},
            ],
        },
    }
    s._flow_fetcher = mock_flow
    return s


def make_kline_df(close_values):
    return pd.DataFrame({"Close": close_values})


# ===== _score_pe =====

@pytest.mark.parametrize("pe, expected_score", [
    (8.0, 15), (15.0, 11), (30.0, 7), (60.0, 3), (100.0, 0),
])
def test_score_pe_positive(screener, pe, expected_score):
    result = screener._score_pe(pe)
    assert result["score"] == expected_score
    assert result["max"] == 15


def test_score_pe_missing_is_unknown(screener):
    """v2：无 PE 数据标记为 unknown（不再给中位分 7，避免"缺失"被读成"估值适中"）"""
    result = screener._score_pe(None)
    assert result["score"] is None
    assert result["status"] == "unknown"
    assert "无PE" in result["detail"]


def test_score_pe_negative_is_unknown(screener):
    result = screener._score_pe(-5.0)
    assert result["score"] is None
    assert result["status"] == "unknown"


# ===== _score_trend（v2：20 日动量，不再是当日涨跌幅）=====

def _kline_with_ret(ret20: float, bars: int = 30) -> pd.DataFrame:
    """构造 20 日累计涨幅为 ret20 的收盘价序列"""
    start, end = 10.0, 10.0 * (1 + ret20)
    path = [start + (end - start) * i / 20 for i in range(21)]
    return pd.DataFrame({"Close": [10.0] * (bars - 21) + path})


@pytest.mark.parametrize("ret20, expected_score", [
    (0.05, 15),    # 稳健上行
    (0.20, 11),    # 偏强
    (0.50, 3),     # 过热 → 降级（原实现对"一天涨>5%"给满分，等价于追涨停）
    (0.00, 7),     # 横盘
    (-0.05, 3),    # 偏弱
    (-0.20, 0),    # 下行
])
def test_score_trend_buckets(screener, ret20, expected_score):
    result = screener._score_trend(_kline_with_ret(ret20))
    assert result["score"] == expected_score
    assert result["status"] == "ok"
    assert result["max"] == 15


def test_score_trend_uses_kline_not_single_day_change(screener):
    """回归：趋势因子只依赖 K 线（20 日动量），不再接受"当日涨跌幅"。

    原签名为 _score_trend(change_pct)：一天涨 >5% 直接满分，即"追当天涨得最猛的票"，
    而 A 股单日涨超 5% 往往已接近涨停、不可买入。
    """
    assert screener._score_trend(_kline_with_ret(0.0))["score"] == 7  # 横盘
    assert screener._score_trend(None)["status"] == "unknown"
    short = pd.DataFrame({"Close": [10.0] * 5})
    assert screener._score_trend(short)["status"] == "unknown"


# ===== _score_ma_alignment =====

def test_ma_bullish_alignment(screener):
    closes = list(range(1, 31))
    df = pd.DataFrame({"Close": closes})
    result = screener._score_ma_alignment(df)
    assert result["score"] == 10
    assert "多头排列" in result["detail"]


def test_ma_bearish_alignment(screener):
    closes = list(range(30, 0, -1))
    df = pd.DataFrame({"Close": closes})
    result = screener._score_ma_alignment(df)
    assert result["score"] == 0
    assert "空头排列" in result["detail"]


def test_ma_flat(screener):
    closes = [10.0] * 30
    df = pd.DataFrame({"Close": closes})
    result = screener._score_ma_alignment(df)
    assert result["score"] == 5
    assert "粘合" in result["detail"]


def test_ma_short_on_long(screener):
    """短期偏多: MA5 > MA10 但 MA10 < MA20"""
    # MA5≈13.0, MA10≈12.5, MA20≈14.0 → 短期偏多
    closes = [8.0] * 5 + [22.0] * 10 + [9.0] * 5 + [12.0] * 5 + [13.0] * 5
    df = pd.DataFrame({"Close": closes})
    result = screener._score_ma_alignment(df)
    assert result["score"] == 6  # 2/3 of 10
    assert "短期偏多" in result["detail"]


# ===== _score_volatility =====

def test_volatility_low(screener):
    closes = [10.0 + i * 0.01 for i in range(30)]
    df = pd.DataFrame({"Close": closes})
    result = screener._score_volatility(df)
    assert result["score"] == 10
    assert "低波" in result["detail"]


def test_volatility_missing_data_is_unknown(screener):
    """v2：K线缺失/列名不符 → unknown（原实现给中间分，把"没数据"当成"中低波"）"""
    assert screener._score_volatility(None)["status"] == "unknown"
    assert screener._score_volatility(None)["score"] is None
    bad = screener._score_volatility(pd.DataFrame({"WrongColumn": [10.0] * 30}))
    assert bad["score"] is None and bad["status"] == "unknown"


def test_ma_missing_data_is_unknown(screener):
    """v2：均线排列同样不再用中位分兜底"""
    assert screener._score_ma_alignment(None)["status"] == "unknown"
    short = pd.DataFrame({"Close": [10.0] * 5})
    assert screener._score_ma_alignment(short)["status"] == "unknown"


# ===== _score_single 集成（v2：unknown 剔除 + 只在有效权重上归一化）=====

def _v2_weights():
    from core.sector import FACTOR_WEIGHTS, FACTOR_MAX_SCORES
    return dict(FACTOR_WEIGHTS), dict(FACTOR_MAX_SCORES)


def test_score_single_breakdown_keys_match_weights(screener):
    """契约：breakdown 键必须与权重表逐字一致，否则权重取不到、因子被静默丢弃"""
    weights, max_scores = _v2_weights()
    saved = screener._get_kline_data
    screener._get_kline_data = lambda code: make_kline_df(list(range(1, 31)))
    try:
        result = screener._score_single("000001", weights, max_scores)
    finally:
        screener._get_kline_data = saved

    assert result["status"] == "ok"
    assert set(result["breakdown"]) == set(weights)
    assert 0 <= result["total"] <= 100


def test_score_single_excludes_unknown_from_denominator(screener):
    """v2：unknown 因子从分子与分母同时剔除，而不是按中位分计入"""
    weights, max_scores = _v2_weights()
    saved_kline = screener._get_kline_data
    saved_news = screener._score_news_factor
    screener._get_kline_data = lambda code: make_kline_df(list(range(1, 31)))
    # ⚠️ 桩必须接受真实签名的全部关键字（D4 之后多传了 allow_fetch）
    screener._score_news_factor = lambda code, max_score=15, name=None, **_kw: {
        "score": None, "max": max_score, "status": "unknown", "detail": "近3日无关联新闻"}
    try:
        result = screener._score_single("000001", weights, max_scores)
    finally:
        screener._get_kline_data = saved_kline
        screener._score_news_factor = saved_news

    bd = result["breakdown"]
    assert bd["新闻正面率"]["status"] == "unknown"
    expected_cov = sum(w for k, w in weights.items() if k != "新闻正面率")
    assert result["coverage"] == pytest.approx(expected_cov, abs=1e-4)
    weighted = sum(bd[k]["score"] / bd[k]["max"] * w * 100
                   for k, w in weights.items()
                   if k != "新闻正面率" and bd[k]["status"] == "ok")
    assert result["total"] == pytest.approx(weighted / expected_cov, abs=0.1)


def test_score_single_insufficient_data_returns_none(screener):
    """v2：有效权重不足 50% 时不给分数。

    否则"只拿到一个因子且它满分"会被归一化成 100 分，系统性高估
    数据稀缺标的（新股、复牌、数据源不可达），比中位分占位更危险。
    """
    weights, max_scores = _v2_weights()
    saved = (screener._get_kline_data, screener._flow_fetcher, screener._stock_fetcher)
    screener._get_kline_data = lambda code: None          # 趋势/均线/波动率 → unknown
    flow = MagicMock()
    flow.get_fund_flow_data.return_value = {"data_success": False}   # 资金 → unknown
    screener._flow_fetcher = flow
    info = MagicMock()
    info.get_stock_info.return_value = {"name": "测试股份"}           # 无 PE → unknown
    screener._stock_fetcher = info
    try:
        result = screener._score_single("000001", weights, max_scores)
    finally:
        screener._get_kline_data, screener._flow_fetcher, screener._stock_fetcher = saved

    assert result["total"] is None
    assert result["status"] == "insufficient_data"
    assert "低于阈值" in result["detail"]


def test_score_single_zero_weights_is_insufficient(screener):
    """权重全 0 → 无任何有效因子 → 数据不足（原实现返回 0.0，语义完全不同）"""
    weights = {k: 0.0 for k in _v2_weights()[0]}
    max_scores = {k: 15 for k in weights}
    result = screener._score_single("000001", weights, max_scores)
    assert result["total"] is None
    assert result["status"] == "insufficient_data"


# ===== 舆情因子关联池（v2 0.2.3）=====

def test_news_factor_accepts_stock_name(screener):
    """v2（0.2.3）：_score_news_factor 必须把股票简称透传给新闻查询"""
    captured = {}

    def fake(code, days=3, name=None):
        captured["code"], captured["name"] = code, name
        return {"total": 3, "positive": 3, "negative": 0, "positive_rate": 1.0}

    with patch("data.news_feed.news_sentiment_for_stock", side_effect=fake):
        result = screener._score_news_factor("002415", max_score=15, name="海康威视")

    assert captured == {"code": "002415", "name": "海康威视"}
    assert result["status"] == "ok"
    assert result["score"] == 15


def test_score_single_passes_stock_name_to_news(screener):
    """v2（0.2.3）：_score_single 应把 get_stock_info 的简称传给舆情因子"""
    captured = {}
    saved_news = screener._score_news_factor
    saved_kline = screener._get_kline_data
    screener._get_kline_data = lambda code: make_kline_df(list(range(1, 31)))

    def fake_news(code, max_score=15, name=None, **_kw):
        captured["code"], captured["name"] = code, name
        captured["allow_fetch"] = _kw.get("allow_fetch")
        return {"score": None, "max": max_score, "status": "unknown", "detail": "占位"}

    screener._score_news_factor = fake_news
    try:
        weights, max_scores = _v2_weights()
        screener._score_single("000001", weights, max_scores)
    finally:
        screener._score_news_factor = saved_news
        screener._get_kline_data = saved_kline

    assert captured["code"] == "000001"
    assert captured["name"] == "测试股份"  # 来自夹具的 mock get_stock_info


# ===== run() 熊市短路 =====

def test_run_bear_market_short_circuit(screener):
    from core.market_detector import MarketSignal
    signal = MarketSignal(
        total_score=20, market_state="熊市", signal="强烈卖出",
        position_limit=0.0, strategy_bias="清仓",
        operation_advice="建议空仓", scores={}, indicators={},
        risk_warnings=["大盘暴跌"], timestamp="2026-01-01",
    )
    with patch("core.market_detector.MarketDetector") as mock_det:
        mock_det.return_value.detect.return_value = signal
        result = screener.run(sector_filter=None, max_stocks=5)
    assert result["skipped"] is True
    assert result["candidates"] == []


# ===== _build_sector_top5 =====

def test_build_sector_top5_returns_list(screener):
    result = screener._build_sector_top5("科技", max_stocks=3)
    assert len(result) == 3
    for s in result:
        assert "股票代码" in s


def test_build_sector_unknown_fallback(screener):
    result = screener._build_sector_top5("不存在的板块")
    assert result == []


def test_build_sector_top5_has_no_fake_score(screener):
    """v2（0.2.7）：降级候选不再伪造 50.0 分与假的"板块推荐"因子。

    原实现返回硬编码代码 + score=50.0 + factors={"板块推荐": {"score":50,"max":100}}，
    UI 会把它画成一根半满的因子条，用户无法分辨这是精选池还是真实评分结果；
    字段名还写成英文 strategy，与其它路径的"策略"不一致。
    """
    result = screener._build_sector_top5("科技", max_stocks=3)
    assert len(result) == 3
    for item in result:
        assert item["score"] is None, "降级项不得带任何分数（含占位分）"
        assert item["factors"] == {}
        assert item["degraded"] is True
        assert "degrade_reason" in item
        assert "策略" in item and "strategy" not in item


# ===== _get_kline_data =====

def test_get_kline_data_returns_df(screener, monkeypatch):
    # ⚠️ P0-1 后 `_get_kline_data` **本地优先**：不隔离本地分支，这条用例
    #    实际走的是面板（mock 的 get_stock_data 根本没被调用）⇒ 变成假通过。
    monkeypatch.setattr("data.bars.get_bars", lambda *a, **k: None)
    df = make_kline_df([10, 11, 12])
    screener._stock_fetcher.get_stock_data.return_value = df
    result = screener._get_kline_data("000001")
    assert isinstance(result, pd.DataFrame)


def test_get_kline_data_error_returns_none(screener, monkeypatch):
    monkeypatch.setattr("data.bars.get_bars", lambda *a, **k: None)   # 本地也没有
    screener._stock_fetcher.get_stock_data.return_value = {"error": "网络错误"}
    result = screener._get_kline_data("000001")
    assert result is None


def test_get_kline_data_prefers_local_panel(screener, monkeypatch):
    """⭐⭐ P0-1：本地命中 ⇒ **绝不再联网**（这条是"选股 4.3 分钟"的性能契约）。

    面板覆盖 5830 只，联网 51 s/只 ⇒ 一旦本地命中还去联网，提速就白做了。
    """
    import pandas as pd
    fake = pd.DataFrame({
        "date": pd.date_range("2026-08-01", periods=25, freq="B"),
        "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
        "volume": 1e6, "amount": 1e7,
    })
    monkeypatch.setattr("data.bars.get_bars", lambda *a, **k: fake)
    # ⚠️ 新鲜度判定也要隔离：`bars.staleness` 读的是**真实/上一条用例的**面板，
    #    不 patch 就会因为"夹具数据比真实面板旧"被判过期（R1 之后判定提前了）。
    monkeypatch.setattr("data.bars.staleness",
                        lambda *a, **k: {"data_lagging": False, "lag_days": 0,
                                         "last_date": None, "reason": ""})

    def _boom(*a, **k):
        raise AssertionError("本地命中后不许联网（P0-1 性能契约）")

    screener._stock_fetcher.get_stock_data.side_effect = _boom
    result = screener._get_kline_data("600028")
    assert isinstance(result, pd.DataFrame)
    # 下游读大写列 + Date 索引 ⇒ 转换必须做
    assert "Close" in result.columns and result.index.name == "Date"


def test_stale_local_never_falls_back_to_network(screener, monkeypatch):
    """⭐ R1（2026-10-18 审查）：**过期就是过期，在线也不许联网补齐**。

    审查实测：002731 末根 2026-08-31（滞后 28 个交易日），原实现先记
    `stale_skipped`、随后"本地没有就联网"又把它补回来 ⇒ 日志说"记不可评"、
    实际照样参与排名。而且联网拿回的仍是同一段停牌前历史，补了也没意义。
    """
    monkeypatch.setattr(
        "data.bars.staleness",
        lambda *a, **k: {"data_lagging": True, "lag_days": 28,
                         "last_date": "2026-08-31",
                         "reason": "最后交易日 2026-08-31 落后参考日 2026-10-16 共 28 个交易日"})
    monkeypatch.setattr("data.bars.get_bars", lambda *a, **k: None)

    def _boom(*a, **k):
        raise AssertionError("过期标的**不许**联网兜底（R1：补了也是同一段停牌前历史）")

    screener._stock_fetcher.get_stock_data.side_effect = _boom
    assert screener._get_kline_data("002731") is None
    assert "002731" in screener._stale_codes
    assert "28" in screener._stale_codes["002731"]


def test_stale_check_precedes_window_slice(screener, monkeypatch):
    """⭐ R1：长期停牌股在 90 天窗口里 0 行，也必须先被判过期（不许直接落联网兜底）。"""
    calls = []

    def _stale(code, *a, **k):
        calls.append(code)
        return {"data_lagging": True, "lag_days": 2300, "last_date": "2016-05-11",
                "reason": "最后交易日 2016-05-11 落后参考日 2026-10-16 共 2300 个交易日"}

    monkeypatch.setattr("data.bars.staleness", _stale)
    monkeypatch.setattr("data.bars.get_bars",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("不该走到取数")))
    screener._stock_fetcher.get_stock_data.side_effect = \
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("不许联网"))
    assert screener._get_kline_data("600656") is None
    assert calls == ["600656"], "新鲜度必须先于窗口取数被调用"
    assert "600656" in screener._stale_codes


# ===== 问财降级链（内置精选池）=====

def test_preset_pool_all_sectors(screener):
    candidates = screener._preset_pool_candidates(None)
    codes = [c["股票代码"] for c in candidates]
    assert len(codes) == 18  # PRESET_POOL_LIMIT 截断
    assert len(codes) == len(set(codes))
    assert all(c["策略"] == "内置精选池" for c in candidates)


def test_preset_pool_sector_filter(screener):
    candidates = screener._preset_pool_candidates("创业板")
    assert len(candidates) == 10
    assert all(c["股票代码"].startswith("300") for c in candidates)


def test_preset_pool_marks_degraded(screener):
    """v2（0.2.7）：内置精选池必须显式标记降级，不再静默替代"全市场选股" """
    candidates = screener._preset_pool_candidates(None)
    assert all(c["degraded"] is True for c in candidates)
    assert all(c["degrade_reason"] for c in candidates)


def test_run_marks_degraded_when_preset_pool_used(screener):
    """问财全失败 → run() 返回值必须带 degraded 标记与原因，供报告/UI 提示"""
    mock_signal = MagicMock()
    mock_signal.position_limit = 1
    with patch("core.market_detector.MarketDetector") as mock_md, \
         patch.object(screener, "_run_strategies", return_value=[]), \
         patch.object(screener, "_score_candidates",
                      side_effect=lambda cands, progress_cb=None: cands):
        mock_md.return_value.detect.return_value = mock_signal
        result = screener.run()
    assert result["degraded"] is True
    assert result["degrade_reasons"], "必须给出降级原因"
    assert all(c.get("degraded") for c in result["candidates"])


def test_run_not_degraded_on_normal_screen(screener):
    """正常初筛路径不得误报降级"""
    mock_signal = MagicMock()
    mock_signal.position_limit = 1
    cands = [{"股票代码": "600519", "股票名称": "贵州茅台", "策略": "主力资金"}]
    scored = [{"股票代码": "600519", "股票名称": "贵州茅台", "score": 80.0, "factors": {}}]
    with patch("core.market_detector.MarketDetector") as mock_md, \
         patch.object(screener, "_run_strategies", return_value=cands), \
         patch.object(screener, "_score_candidates", return_value=scored):
        mock_md.return_value.detect.return_value = mock_signal
        result = screener.run()
    assert result["degraded"] is False
    assert result["degrade_reasons"] == []
    assert result["insufficient_data"] == []


def test_run_puts_insufficient_data_last(screener):
    """数据不足（score=None）的候选必须排在末尾，且不参与名次竞争"""
    mock_signal = MagicMock()
    mock_signal.position_limit = 1
    cands = [{"股票代码": "600519", "股票名称": "贵州茅台", "策略": "主力资金"}]
    scored = [
        {"股票代码": "000001", "score": None, "factors": {}},
        {"股票代码": "600519", "score": 80.0, "factors": {}},
        {"股票代码": "300750", "score": 55.0, "factors": {}},
    ]
    with patch("core.market_detector.MarketDetector") as mock_md, \
         patch.object(screener, "_run_strategies", return_value=cands), \
         patch.object(screener, "_score_candidates", return_value=scored):
        mock_md.return_value.detect.return_value = mock_signal
        result = screener.run(max_stocks=3)
    assert [c["股票代码"] for c in result["candidates"]] == ["600519", "300750", "000001"]
    assert result["insufficient_data"] == ["000001"]


def test_run_falls_back_to_preset_pool_when_wencai_fails(screener):
    """问财全失败时，run() 应降级到内置精选池并产出候选"""
    mock_signal = MagicMock()
    mock_signal.position_limit = 1
    mock_signal.market_state = "bull"
    mock_signal.total_score = 80
    fake_scored = [{"股票代码": "600519", "股票名称": "贵州茅台", "score": 75.0,
                    "策略": "内置精选池", "factors": {}}]
    with patch("core.market_detector.MarketDetector") as mock_md, \
         patch.object(screener, "_run_strategies", return_value=[]), \
         patch.object(screener, "_score_candidates", return_value=fake_scored):
        mock_md.return_value.detect.return_value = mock_signal
        result = screener.run()
    assert result["skipped"] is False
    assert result["candidates"]
    assert result["candidates"][0]["策略"] == "内置精选池"


def test_sector_filter_matches_by_board_prefix(screener):
    """创业板筛选应按代码前缀匹配（不限于硬编码池）"""
    mock_signal = MagicMock()
    mock_signal.position_limit = 1
    candidates_in = [{"股票代码": "301536", "股票名称": "某创业板股", "策略": "技术突破"}]
    fake_scored = [{"股票代码": "301536", "score": 70.0, "factors": {}}]
    with patch("core.market_detector.MarketDetector") as mock_md, \
         patch.object(screener, "_run_strategies", return_value=candidates_in), \
         patch.object(screener, "_score_candidates", return_value=fake_scored):
        mock_md.return_value.detect.return_value = mock_signal
        result = screener.run(sector_filter="创业板")
    assert result["candidates"][0]["股票代码"] == "301536"


# ===== 单一权重集（v2：板块权重矩阵已取消）=====

def test_score_candidates_uses_single_weight_set(screener):
    """所有股票使用同一套权重，且"板块"字段为交易所板块。

    原实现有一个"全市场模式下按代码前缀自动套用板块权重"的分支，
    但生产路径传入的 sector 永为真值，该分支不可达（死代码）；
    且原板块矩阵含 screener 未实现的因子，一旦启用会静默丢失权重。
    """
    captured = {}

    def fake_score_single(code, weights, max_scores, context=None):
        captured[code] = weights
        return {"total": 60.0, "breakdown": {}, "status": "ok", "coverage": 1.0}

    with patch.object(screener, "_score_single", side_effect=fake_score_single):
        result = screener._score_candidates(
            [{"股票代码": "300750", "股票名称": "宁德时代"},
             {"股票代码": "600519", "股票名称": "贵州茅台"}])

    assert captured["300750"] == captured["600519"], "所有股票必须使用同一套权重"
    assert "营收增速" not in captured["300750"], "未实现的因子不得出现在权重里"
    assert result[0]["板块"] == "创业板"
    assert result[1]["板块"] == "主板沪市"


# ===== 策略子集选择 =====

def test_run_strategies_subset_uses_local_screen(screener):
    """指定策略子集 → 只把那几路传给本地初筛"""
    fake = {"candidates": [{"股票代码": "600519", "策略": "技术突破"}],
            "meta": {"routes": [], "notes": [], "truncated": 0, "data_dates": {}}}
    with patch("core.local_screen.run_local_strategies", return_value=fake) as m:
        out = screener._run_strategies(["技术突破"])
    assert m.call_args.kwargs["labels"] == ["技术突破"]
    assert out == [{"股票代码": "600519", "策略": "技术突破"}]
    assert screener.last_strategy_meta["truncated"] == 0


def test_run_strategies_default_all_uses_all_routes(screener):
    """None / 空 / 全非法名 → 跑**全部五路**。

    断言"跑全部"这个语义（labels 允为 None 或五路全集），不锁死实现细节——
    首版断言 `labels is None` 就红了：实现传的是五路名单而非 None，两者等价。
    """
    from core.screener import STRATEGIES
    fake = {"candidates": [], "meta": {"routes": [], "notes": []}}
    with patch("core.local_screen.run_local_strategies", return_value=fake) as m:
        for arg in (None, [], ["不存在的策略"]):
            screener._run_strategies(arg)
            labels = m.call_args.kwargs["labels"]
            assert labels is None or set(labels) == set(STRATEGIES), labels


def test_run_strategies_no_longer_calls_wencai(screener):
    """⚠️ 回归守护：初筛**不得再依赖问财**（2026-10-09 本地化）。

    背景：pywencai 实测失效（响应体解析失败）→ 旧实现每次都降级到
    `SECTOR_HARDCODED` 前 18 只 → 选股结果跨日高度重复（TOP5 4/5 重合、
    同日 5/5 完全一致）。若有人把问财调用加回初筛，这条用例必须变红。
    """
    fake = {"candidates": [{"股票代码": "600519", "策略": "主力资金"}],
            "meta": {"routes": [], "notes": []}}
    with patch("core.screener.safe_get") as mock_get, \
         patch("core.local_screen.run_local_strategies", return_value=fake):
        out = screener._run_strategies(["主力资金"])
    assert mock_get.call_count == 0, "初筛不应再调用问财"
    assert out, "本地初筛应有候选"


def test_custom_only_skips_builtin_routes(screener):
    """custom_only：只跑自定义条件，不混入内置五路"""
    from core.local_pool import PoolResult
    with patch("core.local_screen.run_local_strategies") as m_local, \
         patch("core.nl_screener.parse", return_value="Q"), \
         patch("core.local_pool.build_pool", return_value=PoolResult(codes=["600519"])):
        out = screener._run_strategies(custom_queries=["市盈率小于20倍"], custom_only=True)
    assert m_local.call_count == 0, "custom_only 不应跑内置五路"
    assert [c["策略"] for c in out] == ["自然语言条件"]


def test_run_passes_strategies_to_screen(screener):
    """run() 的 strategies 参数应传递到 _run_strategies"""
    mock_signal = MagicMock()
    mock_signal.position_limit = 1  # 非熊市，走到初筛
    with patch("core.market_detector.MarketDetector") as mock_md, \
         patch.object(screener, "_run_strategies", return_value=[]) as mock_run:
        mock_md.return_value.detect.return_value = mock_signal
        screener.run(strategies=["低估值成长"])
    mock_run.assert_called_once_with(["低估值成长"], progress_cb=None, custom_queries=None,
                                     custom_only=False)


def test_run_passes_custom_queries(screener):
    """自然语言选股生成的问财语句要真的传进初筛（不然只是装饰）"""
    mock_signal = MagicMock()
    mock_signal.position_limit = 1
    with patch("core.market_detector.MarketDetector") as mock_md, \
         patch.object(screener, "_run_strategies", return_value=[]) as mock_run:
        mock_md.return_value.detect.return_value = mock_signal
        screener.run(custom_queries=["市盈率小于20倍，非ST"])
    assert mock_run.call_args.kwargs["custom_queries"] == ["市盈率小于20倍，非ST"]


def test_custom_queries_go_through_local_pool(screener):
    """自定义条件走**本地解析**（nl_screener + local_pool），不再调问财"""
    from core.local_pool import PoolResult
    empty_local = {"candidates": [], "meta": {"routes": [], "notes": []}}
    with patch("core.local_screen.run_local_strategies", return_value=empty_local), \
         patch("core.nl_screener.parse", return_value="QUERY") as mp, \
         patch("core.local_pool.build_pool",
               return_value=PoolResult(codes=["600519"])) as bp:
        out = screener._run_strategies(["主力资金"],
                                       custom_queries=["市盈率小于20倍", "   "])
    assert [c["策略"] for c in out] == ["自然语言条件"]
    assert mp.call_count == 1 and bp.call_count == 1     # 空白语句被丢掉


def test_custom_query_parse_failure_is_reported(screener):
    """⚠️ 解析失败不能静默——必须写进 strategy_meta.notes 让用户看到"""
    empty_local = {"candidates": [], "meta": {"routes": [], "notes": []}}
    with patch("core.local_screen.run_local_strategies", return_value=empty_local), \
         patch("core.nl_screener.parse", side_effect=ValueError("解析不了")):
        out = screener._run_strategies(["主力资金"], custom_queries=["随便写点什么"])
    assert out == []
    assert any("无法本地解析" in n for n in screener.last_strategy_meta["notes"])


def test_custom_query_pool_unavailable_is_reported(screener):
    from core.local_pool import PoolUnavailable
    empty_local = {"candidates": [], "meta": {"routes": [], "notes": []}}
    with patch("core.local_screen.run_local_strategies", return_value=empty_local), \
         patch("core.nl_screener.parse", return_value="Q"), \
         patch("core.local_pool.build_pool",
               side_effect=PoolUnavailable("快照不可用")):
        out = screener._run_strategies(["主力资金"], custom_queries=["市盈率小于20倍"])
    assert out == []
    assert any("执行失败" in n for n in screener.last_strategy_meta["notes"])

def test_run_strategies_custom_only_requires_query(screener):
    """什么都不给就要求"只用自定义" → 显式报错（不能静默跑空后降级到精选池）"""
    with pytest.raises(ValueError):
        screener._run_strategies(custom_queries=[], custom_only=True)
    with pytest.raises(ValueError):
        screener._run_strategies(custom_queries=["   "], custom_only=True)


def test_run_passes_custom_only(screener):
    mock_signal = MagicMock()
    mock_signal.position_limit = 1
    with patch("core.market_detector.MarketDetector") as mock_md, \
         patch.object(screener, "_run_strategies", return_value=[]) as mock_run:
        mock_md.return_value.detect.return_value = mock_signal
        screener.run(custom_queries=["市盈率小于20倍"], custom_only=True)
    assert mock_run.call_args.kwargs["custom_only"] is True


# ===== Phase 1：选股进度回调 =====

def _ok_signal():
    s = MagicMock()
    s.position_limit = 1
    s.market_state = "偏强震荡"
    s.total_score = 72
    return s


def test_progress_callback_emits_stages_and_monotonic_percent(screener):
    events = []
    cands = [{"股票代码": "600519", "股票名称": "贵州茅台", "策略": "主力资金"},
             {"股票代码": "000001", "股票名称": "平安银行", "策略": "技术突破"}]
    scored = [{"股票代码": "600519", "score": 80.0, "factors": {}},
              {"股票代码": "000001", "score": 70.0, "factors": {}}]
    with patch("core.market_detector.MarketDetector") as mock_md, \
         patch.object(screener, "_run_strategies", return_value=cands), \
         patch.object(screener, "_score_candidates", return_value=scored):
        mock_md.return_value.detect.return_value = _ok_signal()
        screener.run(max_stocks=2, progress_cb=events.append)

    stages = [e["stage"] for e in events]
    assert "market" in stages and "dedupe" in stages and "done" in stages
    percent = [e["percent"] for e in events]
    assert percent == sorted(percent), f"进度必须单调不减: {percent}"
    assert percent[-1] == 100.0
    assert all(0 <= p <= 100 for p in percent)


def test_progress_reports_per_stock_during_scoring(screener):
    """逐股评分要上报细粒度进度——UI 上"12/30"就是这里来的"""
    events = []
    cands = [{"股票代码": f"60000{i}", "股票名称": "x"} for i in range(3)]
    with patch.object(screener, "_score_single",
                      return_value={"total": 50.0, "breakdown": {}, "status": "ok",
                                    "coverage": 1.0, "name": "x"}):
        screener._score_candidates(cands, progress_cb=events.append)

    score_events = [e for e in events if e["stage"] == "score"]
    assert [e["current"] for e in score_events] == [0, 1, 2, 3]
    assert all(e["total"] == 3 for e in score_events)
    assert score_events[-1]["percent"] == 98.0


def test_progress_reports_local_screen_stage(screener):
    """初筛阶段必须上报进度（本地化后不再是"每路一次问财调用"的序号）"""
    events = []
    fake = {"candidates": [],
            "meta": {"routes": [{"label": "主力资金", "count": 0, "status": "ok",
                                 "note": "", "query": ""}], "notes": []}}
    with patch("core.local_screen.run_local_strategies", return_value=fake):
        screener._run_strategies(["主力资金", "技术突破"], progress_cb=events.append)
    screen_events = [e for e in events if e["stage"] == "screen"]
    assert screen_events, "初筛阶段必须上报进度"
    assert any("本地初筛" in e["message"] for e in screen_events)


def test_progress_callback_exception_does_not_break_screening(screener):
    """回调抛异常绝不能影响选股主流程——否则一个 UI 小 bug 就能让选股失败"""
    def boom(_event):
        raise RuntimeError("UI 炸了")

    with patch("core.market_detector.MarketDetector") as mock_md, \
         patch.object(screener, "_run_strategies", return_value=[]), \
         patch.object(screener, "_score_candidates", return_value=[]):
        mock_md.return_value.detect.return_value = _ok_signal()
        result = screener.run(progress_cb=boom)   # 不应抛出
    assert result["skipped"] is False


def test_emit_progress_defaults_and_zero_total():
    from core.screener import _emit_progress
    out = []
    _emit_progress(out.append, "score", 0, 0, "空")
    assert out[0]["percent"] == 98.0        # 总数为 0 视为该阶段已完成
    _emit_progress(None, "score", 1, 1)     # 无回调不得报错
    _emit_progress(out.append, "unknown_stage", 1, 2, "x")
    assert 0 <= out[-1]["percent"] <= 100


# ================================================================
# 并发预取接入（2026-10-09 P0：扩展候选域）
# ================================================================

@pytest.fixture(autouse=True)
def _no_network_prefetch(monkeypatch):
    """单元测试不联网：`_prefetch` 通过该环境变量短路（专门的预取测试会自行开启）。

    为什么需要：`run()` 现在会在精筛前真并发取数（8 线程 × 上百只），
    测试若不禁用会真的打上游接口 —— 既慢又对外不礼貌。
    """
    monkeypatch.setenv("BIGA_DISABLE_PREFETCH", "1")


def test_run_calls_prefetch_before_scoring(screener):
    """预取必须在**精筛之前**发生（顺序错了等于没优化）"""
    order = []
    mock_signal = MagicMock()
    mock_signal.position_limit = 1
    cands = [{"股票代码": "600519", "股票名称": "贵州茅台", "策略": "主力资金"}]

    with patch("core.market_detector.MarketDetector") as mock_md, \
         patch.object(screener, "_run_strategies", return_value=cands), \
         patch.object(screener, "_prefetch",
                      side_effect=lambda codes, progress_cb=None: order.append("prefetch") or {}), \
         patch.object(screener, "_score_candidates",
                      side_effect=lambda c, progress_cb=None: order.append("score") or c):
        mock_md.return_value.detect.return_value = mock_signal
        screener.run()
    assert order == ["prefetch", "score"], order


def test_run_tolerates_prefetch_failure(screener):
    """预取整体失败不能阻断选股（逐票串行路径仍在）"""
    mock_signal = MagicMock()
    mock_signal.position_limit = 1
    cands = [{"股票代码": "600519", "股票名称": "贵州茅台", "策略": "主力资金"}]
    scored = [{"股票代码": "600519", "score": 80.0, "factors": {}}]
    with patch("core.market_detector.MarketDetector") as mock_md, \
         patch.object(screener, "_run_strategies", return_value=cands), \
         patch.object(screener, "_prefetch",
                      side_effect=RuntimeError("预取炸了")), \
         patch.object(screener, "_score_candidates", return_value=scored):
        mock_md.return_value.detect.return_value = mock_signal
        result = screener.run()
    assert result.get("candidates"), "预取失败也必须产出结果"


def test_disabled_prefetch_is_a_noop(screener, monkeypatch):
    """开关生效：禁用时不发起任何请求，且返回 disabled 标记"""
    monkeypatch.setenv("BIGA_DISABLE_PREFETCH", "1")
    stats = screener._prefetch(["600519", "000001"])
    assert stats.get("disabled") is True and stats["ok"] == 0


def test_scored_info_comes_from_prefetch_cache(screener, monkeypatch):
    """⚠️ 关键守护：预取填过缓存的股票，评分时**不得再联网取基本信息**"""
    monkeypatch.delenv("BIGA_DISABLE_PREFETCH", raising=False)
    from data import prefetch as pref_mod

    class _FakeRes:
        klines = {"600519": "KLINE"}
        infos = {"600519": {"name": "贵州茅台", "pe_ratio": 20.0}}
        failed = {}
        ok_count = 1
        elapsed = 0.1
        warnings = []

    monkeypatch.setattr(pref_mod, "prefetch_market_data",
                        lambda codes, **kw: _FakeRes())
    screener._stock_fetcher = MagicMock()          # 任何真实取数都应被发现
    screener._prefetch(["600519"])
    assert screener._info_cache["600519"]["pe_ratio"] == 20.0
    assert screener._kline_cache["600519"] == "KLINE"

    with patch.object(screener, "_score_pe", return_value={"score": 10, "max": 15,
                                                           "status": "ok", "detail": "x"}), \
         patch.object(screener, "_get_fund_factor", return_value={"score": None, "max": 25,
                                                                 "status": "unknown"}), \
         patch.object(screener, "_score_news_factor", return_value={"score": None, "max": 15,
                                                                    "status": "unknown"}):
        screener._score_single("600519", {"PE估值": 1.0}, {"PE估值": 15})
    assert screener._stock_fetcher.get_stock_info.call_count == 0, \
        "预取缓存命中时不得再联网取基本信息"


# ================================================================
# 排序 tie-break 与行业分散（2026-10-10）
# ================================================================

def _cand(code, score, cov=1.0, inflow=None):
    f = {}
    if inflow is not None:
        f["主力资金"] = {"score": 10, "max": 25, "status": "ok",
                        "value": {"net_inflow_wan": inflow}}
    return {"股票代码": code, "股票名称": f"股{code}", "score": score,
            "score_coverage": cov, "factors": f}


class TestRankAndSelect:
    """把"每次都同几只"的第二层成因钉死：同分无 tie-break + 行业不分散。"""

    def test_return_all_gives_full_order_not_sector_diversified(self, screener):
        """⭐ 候选扩池（2026-10-22）：`return_all=True` 必须返回**完整纯排序**。

        为什么需要它：落库只拿 `candidates`（做过行业分散的 5~10 只），
        深度分析每天就只能从个位数里挑 —— 10-21 实测连续两日重合 56%。
        而行业分散是"给用户看的名单"要的，扩池要的是"按分数全排"，两个目的不同。
        """
        scored = [_cand("600001", 90.0), _cand("600002", 80.0), _cand("600003", 70.0),
                  _cand("600004", 60.0)]
        top, meta, ordered = screener._rank_and_select(
            scored, max_stocks=2, return_all=True)
        assert len(top) == 2, "top 仍是最终名单（受 max_stocks 约束）"
        assert [c["股票代码"] for c in ordered] == ["600001", "600002", "600003", "600004"], \
            "ordered 必须是完整排序，不能按 max_stocks 截断"
        assert len(ordered) > len(top)
        assert isinstance(meta, dict)

    def test_return_all_false_keeps_old_two_tuple_contract(self, screener):
        """默认行为不变 —— 现有调用方（run/score_codes/测试）都按 2 元组解包。"""
        out = screener._rank_and_select([_cand("600001", 90.0)], max_stocks=1)
        assert len(out) == 2

    def test_tiebreak_by_coverage_then_flow_then_code(self, screener):
        """同分时依次比：覆盖度 → 近5日主力净流入 → 代码（完全确定性）"""
        screener._industry_cache = {}
        scored = [
            _cand("600001", 70.0, cov=0.82, inflow=1e4),
            _cand("600002", 70.0, cov=1.0, inflow=1e4),
            _cand("600003", 70.0, cov=1.0, inflow=9e4),
        ]
        top, _ = screener._rank_and_select(scored, max_stocks=3)
        assert [c["股票代码"] for c in top] == ["600003", "600002", "600001"]

    def test_tiebreak_ignores_float_noise(self, screener):
        """浮点噪声造成的"假差异"不应决定名次：68.2333 与 68.2334 视为同分。

        ⚠️ 首版用例写成 68.24 vs 68.16 —— 那是 **0.08 分的真实差异**，不该算同分
        （实现按 `round(x, 2)` 归并，正好正确地把它们区分开）。改为真正的噪声量级，
        才是在测"消除浮点噪声"这件事。
        """
        screener._industry_cache = {}
        scored = [_cand("600001", 68.2333, cov=0.82), _cand("600002", 68.2334, cov=1.0)]
        top, _ = screener._rank_and_select(scored, max_stocks=2)
        # 两者都归并为 68.23 → 由覆盖度决定：数据更全的 600002 在前
        assert [c["股票代码"] for c in top] == ["600002", "600001"]

    def test_real_score_difference_still_wins(self, screener):
        """真实差异（0.08 分）必须压过覆盖度——否则 tie-break 就喧宾夺主了"""
        screener._industry_cache = {}
        scored = [_cand("600001", 68.24, cov=0.82), _cand("600002", 68.16, cov=1.0)]
        top, _ = screener._rank_and_select(scored, max_stocks=2)
        assert [c["股票代码"] for c in top] == ["600001", "600002"]

    def test_insufficient_data_stays_last(self, screener):
        screener._industry_cache = {}
        top, _ = screener._rank_and_select(
            [_cand("600001", None), _cand("600002", 50.0)], max_stocks=2)
        assert [c["股票代码"] for c in top] == ["600002", "600001"]

    def test_sector_cap_limits_same_industry(self, screener):
        """同行业上限：3 只金融里只能进 2 只（第 3 只进跳过明细）"""
        screener._industry_cache = {"600001": "金融行业", "600002": "金融行业",
                                    "600003": "金融行业", "600004": "医药"}
        scored = [_cand(f"60000{i}", 80.0 - i) for i in range(1, 5)]
        top, meta = screener._rank_and_select(scored, max_stocks=3)
        codes = [c["股票代码"] for c in top]
        assert codes.count("600003") == 0, codes          # 第三只金融被挡下
        assert meta["skipped_by_sector_count"] == 1
        assert meta["skipped_by_sector"][0]["股票代码"] == "600003"

    def test_sector_cap_relaxes_to_fill_list(self, screener):
        """⚠️ 分散是**优先**而非牺牲数量：只有 3 只同行业且要 3 只时，必须补足"""
        screener._industry_cache = {f"60000{i}": "金融行业" for i in range(1, 4)}
        scored = [_cand(f"60000{i}", 80.0 - i) for i in range(1, 4)]
        top, meta = screener._rank_and_select(scored, max_stocks=3)
        assert len(top) == 3, "不能因为行业分散只给 2 只"
        assert meta["cap_relaxed"] is True

    def test_missing_industry_map_disables_cap(self, screener):
        """行业数据整体不可用 → 不做分散（而不是把所有股票当同一行业砍到 2 只）"""
        screener._industry_cache = {}                     # 快照缺失
        scored = [_cand(f"60000{i}", 80.0 - i) for i in range(1, 5)]
        top, meta = screener._rank_and_select(scored, max_stocks=4)
        assert len(top) == 4
        assert meta["sector_cap"] is None and not meta["skipped_by_sector"]

    def test_stock_without_industry_is_exempt(self, screener):
        """单只缺行业（新股/快照未覆盖）应豁免，不与他人挤占同一个"未知"名额"""
        screener._industry_cache = {"600001": "金融行业"}   # 2/3/4 不在映射里
        scored = [_cand(f"60000{i}", 80.0 - i) for i in range(1, 5)]
        top, _ = screener._rank_and_select(scored, max_stocks=4)
        assert len(top) == 4


class TestCoverageThreshold:

    def test_threshold_raised_to_07(self):
        """守护：阈值必须是 0.7（0.5 太松 —— 缺两个因子也能拿分参与排名）"""
        assert StockScreener.MIN_WEIGHT_COVERAGE == 0.7

    def test_missing_kline_and_news_is_insufficient(self, screener):
        """缺 K 线三因子 + 舆情（覆盖 ≈0.41）→ 数据不足，不给分（阈值 0.7 生效）

        ⚠️ 注意 `get_weights()` 返回的是**嵌套结构** `{"weights": {...}, "max_scores": {...}}`，
        不是扁平的权重字典。首版直接把它当 weights 传，结果 `weights.get("主力资金")`
        恒为 None → 一个因子都没算 → coverage=0 也让用例"通过"（**假阳性**，
        测的不是阈值而是传参错误）。这里显式用 FACTOR_WEIGHTS / FACTOR_MAX_SCORES。
        """
        from core.sector import FACTOR_WEIGHTS, FACTOR_MAX_SCORES
        with patch.object(screener, "_get_kline_data", return_value=None), \
             patch.object(screener, "_score_news_factor",
                          return_value={"score": None, "max": 15, "status": "unknown",
                                        "detail": "样本不足"}):
            res = screener._score_single("600519", FACTOR_WEIGHTS, FACTOR_MAX_SCORES)
        assert res["status"] == "insufficient_data"
        assert res["total"] is None
        assert 0 < res["coverage"] < 0.7, res["coverage"]

    def test_single_missing_factor_still_scored(self, screener):
        """只缺 1 个因子（覆盖 0.8235）→ 仍给分；阈值不该把正常数据也判死"""
        from core.sector import FACTOR_WEIGHTS, FACTOR_MAX_SCORES
        with patch.object(screener, "_score_news_factor",
                          return_value={"score": None, "max": 15, "status": "unknown",
                                        "detail": "样本不足"}), \
             patch.object(screener, "_score_trend",
                          return_value={"score": 8, "max": 15, "status": "ok",
                                        "detail": "x"}), \
             patch.object(screener, "_score_ma_alignment",
                          return_value={"score": 6, "max": 10, "status": "ok",
                                        "detail": "x"}), \
             patch.object(screener, "_score_volatility",
                          return_value={"score": 7, "max": 10, "status": "ok",
                                        "detail": "x"}):
            res = screener._score_single("600519", FACTOR_WEIGHTS, FACTOR_MAX_SCORES)
        assert 0.7 <= res["coverage"] < 1.0, res["coverage"]
        assert res["status"] == "ok"


# ================================================================
# 横截面分位化（2026-10-10，P2）
#
# 背景：候选域 18→112、排序确定化之后，TOP5 成员**仍然完全不变** —— 因为成员由分数
# 决定，而分数被绝对值分档支配（PE 低→满分、波动率低→满分，银行天然占优；
# 实测满分率 PE 49% / 波动率 59%）。分位化把口径换成"在当日候选里排第几"。
# 这些用例锁住：分位语义、并列处理、绝对 sanity、样本过小回退、以及"计算只有一份实现"。
# ================================================================

class TestPercentileMap:

    def test_lower_is_better(self):
        pct = StockScreener._percentile_map(
            {"a": 5.0, "b": 10.0, "c": 20.0}, higher_is_better=False)
        assert pct["a"] == 1.0 and pct["c"] == 0.0
        assert 0 < pct["b"] < 1

    def test_higher_is_better(self):
        pct = StockScreener._percentile_map(
            {"a": 5.0, "b": 10.0, "c": 20.0}, higher_is_better=True)
        assert pct["c"] == 1.0 and pct["a"] == 0.0

    def test_ties_share_average_rank(self):
        """并列取平均名次——否则"同值看谁在列表前"又变成隐藏的排序依据"""
        pct = StockScreener._percentile_map(
            {"a": 10.0, "b": 10.0, "c": 1.0}, higher_is_better=False)
        assert pct["a"] == pct["b"], "同值必须同分位"
        assert pct["c"] == 1.0

    def test_empty_and_single(self):
        assert StockScreener._percentile_map({}, higher_is_better=False) == {}
        assert StockScreener._percentile_map({"a": 1.0},
                                             higher_is_better=False) == {"a": 1.0}

    def test_ignores_non_numeric(self):
        pct = StockScreener._percentile_map(
            {"a": 5.0, "b": None, "c": 15.0}, higher_is_better=False)
        assert set(pct) == {"a", "c"}


class TestCrossSection:

    def test_disabled_when_too_few_candidates(self, screener):
        """候选太少（<8）→ 退回绝对分档（分位在 3 只里排名没有统计意义）"""
        screener._info_cache = {"600001": {"pe_ratio": 10.0}}
        ctx = screener._build_cross_section([{"股票代码": "600001"}])
        assert ctx["disabled"] is True and ctx["by_code"] == {}
        assert "未启用分位化" in ctx["note"]

    def test_enabled_and_uses_prefetch_cache_only(self, screener):
        """启用时只用预取缓存（零联网）：PE 来自 _info_cache"""
        screener._info_cache = {f"60000{i}": {"pe_ratio": 5.0 * i} for i in range(1, 10)}
        cands = [{"股票代码": f"60000{i}"} for i in range(1, 10)]
        ctx = screener._build_cross_section(cands)
        assert ctx["disabled"] is False
        assert ctx["pe_n"] == 9
        # PE 最低的 600001 分位最高
        assert ctx["by_code"]["600001"]["pe_pct"] == 1.0
        assert ctx["by_code"]["600009"]["pe_pct"] == 0.0

    def test_extreme_pe_excluded_from_percentile(self, screener):
        """亏损(PE<=0)与超上限(>300)不参与分位——相对口径不等于放弃常识"""
        # ⚠️ 隔离当日快照（2026-10-29）：`_build_cross_section` 现在会为"预取没给 PE"
        #    的候选去 `market_snapshot` 兜底，而 600099 林海股份是**真实代码** ⇒
        #    真实快照会补上一个 PE，把"亏损股不得进样本"的断言打成 9≠8。
        screener._snapshot_fields = {}
        screener._info_cache = {f"60000{i}": {"pe_ratio": 5.0 * i} for i in range(1, 9)}
        screener._info_cache["600099"] = {"pe_ratio": -3.0}      # 亏损
        screener._info_cache["600098"] = {"pe_ratio": 9999.0}    # 超上限
        cands = [{"股票代码": f"60000{i}"} for i in range(1, 9)] + \
                [{"股票代码": "600099"}, {"股票代码": "600098"}]
        ctx = screener._build_cross_section(cands)
        assert ctx["pe_n"] == 8, "异常值不得进入分位样本"
        assert "600099" not in ctx["by_code"] and "600098" not in ctx["by_code"]


class TestFactorPercentileScoring:

    def test_pe_uses_percentile_when_given(self, screener):
        r = screener._score_pe(50.0, 15, pct=0.9, peers=50)
        assert r["score"] == 15, "分位最高档应得满分（哪怕绝对 PE 是 50）"
        assert "候选集内" in r["detail"]
        assert r["value"]["basis"] == "cross_section"

    def test_pe_low_percentile_scores_zero(self, screener):
        assert screener._score_pe(3.0, 15, pct=0.05, peers=50)["score"] == 0

    def test_pe_falls_back_to_absolute_without_pct(self, screener):
        r = screener._score_pe(8.0, 15)
        assert r["score"] == 15
        assert "绝对分档" in r["detail"]
        assert r["value"]["basis"] == "absolute"

    def test_pe_sanity_beats_percentile(self, screener):
        """PE 超上限：即便分位给满分也必须 0 分（绝对 sanity 优先）"""
        r = screener._score_pe(9999.0, 15, pct=1.0, peers=50)
        assert r["score"] == 0
        assert "超出合理上限" in r["detail"]

    def test_pe_loss_still_unknown(self, screener):
        """亏损股判 unknown（数据性质），而不是"相对最便宜"拿满分"""
        r = screener._score_pe(-5.0, 15, pct=1.0, peers=50)
        assert r["status"] == "unknown" and r["score"] is None

    def test_volatility_uses_percentile(self, screener):
        """⚠️ 假数据必须落在 sanity 区间内：首版用 ±(1~5)% 交替，算出来年化 >100%
        被 `VOL_SANE_MAX` 正确拦成 0 分（用例失败的原因是用例数据不合理，不是实现错）。
        这里用 ±0.2% 的温和波动（年化约 6%）。"""
        import pandas as pd
        closes = [10.0 + 0.02 * (-1) ** i for i in range(30)]
        df = pd.DataFrame({"Close": closes})
        assert StockScreener._annualized_vol(df) < 20, "用例数据本身应在低波区间"
        r = screener._score_volatility(df, 10, pct=0.95, peers=40)
        assert r["score"] == 10 and "候选集内" in r["detail"]

    def test_volatility_sanity_beats_percentile(self, screener):
        """极端波动 → 年化 >100% → 即便分位满分也 0 分。

        ⚠️ 首版用"每天固定 ×1.5"构造"高波动"：日收益恒为 50% → **标准差为 0**
        → 算出来的波动率≈0，用例反而期望错了。交替 ×1.3 / ×0.7 才是真的高波动。
        """
        import pandas as pd
        closes, cur = [], 10.0
        for i in range(30):
            cur *= 1.3 if i % 2 == 0 else 0.7
            closes.append(cur)
        df = pd.DataFrame({"Close": closes})
        assert StockScreener._annualized_vol(df) > 100, "用例数据本身应是极端波动"
        r = screener._score_volatility(df, 10, pct=1.0, peers=40)
        assert r["score"] == 0
        assert "超出合理上限" in r["detail"]


class TestSingleImplementation:

    def test_annualized_vol_and_momentum_are_single_source(self, screener):
        """⚠️ 因子评分与分位构建必须用**同一份**波动率/动量计算。

        两处各写一份必然在窗口/复权口径上分叉（本项目已多次踩"两侧各注一份"）。
        这里用同一 DataFrame 断言两个调用点结果一致。
        """
        import pandas as pd
        closes = [10.0 + i * 0.1 for i in range(30)]
        df = pd.DataFrame({"Close": closes})
        vol = StockScreener._annualized_vol(df)
        mom = StockScreener._momentum20(df)
        assert vol is not None and mom is not None
        # 趋势因子内部用的动量必须与 _momentum20 一致（差值 < 1e-9）
        r = screener._score_trend(df, 15)
        assert abs(r["value"]["ret20"] - round(mom, 4)) < 1e-9

    def test_short_kline_returns_none(self):
        import pandas as pd
        assert StockScreener._annualized_vol(pd.DataFrame({"Close": [1.0, 2.0]})) is None
        assert StockScreener._momentum20(pd.DataFrame({"Close": [1.0, 2.0]})) is None


# ===== B2-①：资金流本地快照（2026-10-18 审查批次 C 后落地）=====
#
# 口径纪律：逐日模式（东财，含单型拆分，25 分制）与快照模式（全单型合计，20 分制）
# **不可比** ⇒ 本地快照只在 ① 离线模式、② 在线取数失败的降级 时使用，
# 且 detail 必须写明来源与口径 —— 替换在线正常路径会让评分与历史 screen_picks 不可比。

def test_offline_fund_factor_uses_local_snapshot(screener, tmp_path, monkeypatch):
    """⭐ 离线模式：本地快照命中 ⇒ 快照口径可评（20 分制），且**零联网**。"""
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch)
    from db import store
    store.save_fund_flow_snapshot("2026-10-16", [{
        "stock_code": "600028", "stock_name": "中国石化",
        "net_today": 1.78e7, "net_3d": 3.77e8, "net_5d": 6.99e8,
        "net_10d": 1.919e9, "amount": 1.464e9, "change_pct": 2.03, "price": 5.53,
    }])
    screener._offline = True
    screener._flow_fetcher = MagicMock()
    screener._flow_fetcher.get_fund_flow_data.side_effect = \
        AssertionError("离线模式不许调实时资金流")
    res = screener._get_fund_factor("600028")
    assert res["status"] == "ok"
    assert res["max"] == 20, "快照口径「结构」不可评 ⇒ 分母 20（不是 25）"
    assert "本地快照 2026-10-16" in res["detail"]
    assert res["score"] is not None and res["score"] > 0


def test_offline_fund_factor_unknown_when_snapshot_stale(screener, tmp_path, monkeypatch):
    """该股落后于**全市场**最新快照日 ⇒ 它的"最新"不是市场的最新 ⇒ 不可评（缺就是缺）。"""
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch)
    from db import store
    store.save_fund_flow_snapshot("2026-10-16", [
        {"stock_code": "000001", "net_today": 1e7, "net_3d": 2e7,
         "net_5d": 3e7, "net_10d": 4e7}])
    store.save_fund_flow_snapshot("2026-10-10", [
        {"stock_code": "600028", "net_today": 1e7, "net_3d": 2e7,
         "net_5d": 3e7, "net_10d": 4e7}])
    screener._offline = True
    res = screener._get_fund_factor("600028")
    assert res["status"] == "unknown"


def test_online_fund_factor_degrades_to_local_snapshot_on_failure(
        screener, tmp_path, monkeypatch):
    """在线取数**失败** ⇒ 降级用本地快照，detail 必须写明口径与来源。"""
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch)
    from db import store
    store.save_fund_flow_snapshot("2026-10-16", [{
        "stock_code": "600028", "net_today": 1.78e7, "net_3d": 3.77e8,
        "net_5d": 6.99e8, "net_10d": 1.919e9,
        "amount": 1.464e9, "change_pct": 2.03, "price": 5.53,
    }])
    screener._offline = False
    screener._flow_fetcher = MagicMock()
    screener._flow_fetcher.get_fund_flow_data.side_effect = RuntimeError("网络挂了")
    res = screener._get_fund_factor("600028")
    assert res["status"] == "ok" and res.get("degraded") is True
    assert "降级" in res["detail"] and "本地快照" in res["detail"]


def test_offline_fund_factor_unknown_without_snapshot(screener, tmp_path, monkeypatch):
    """离线 + 本地无快照 ⇒ unknown（缺就是缺，不编分数）。"""
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch)
    screener._offline = True
    res = screener._get_fund_factor("600999")
    assert res["status"] == "unknown" and res["score"] is None


# ===== B2③：PE 双口径 + 来源标注（2026-10-18 用户拍板 ③）=====
#
# 背景：本地 `market_snapshot` 的 PE 是**新浪口径**，与东财实时口径存在系统性差异
# （项目实测 688825 快照 1700 vs 东财 42.32）。用户拍板"双写并标来源"⇒
# 离线/降级时用本地快照可评，但明细必须写来源、`screen_picks.pe_source` 必须落库。

def test_offline_pe_uses_local_snapshot_and_labels_source(
        screener, tmp_path, monkeypatch):
    """⭐ 离线：本地快照 PE 可评 + 明细标「新浪口径」+ pe_source=snapshot_sina。"""
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch)
    from db import store
    store.save_market_snapshot("2026-10-16", [
        {"stock_code": "600028", "stock_name": "中国石化", "industry": "石油行业",
         "board": "主板沪市", "price": 5.53, "change_pct": 0.5,
         "pe": 8.9, "pb": 0.8, "market_cap": 1.0e11, "float_cap": 9e10,
         "turnover": 0.5, "amount": 3.0e8}])
    screener._offline = True
    screener._kline_cache.clear()
    weights, max_scores = _v2_weights()
    res = screener._score_single("600028", weights, max_scores)
    assert res["pe_source"] == "snapshot_sina"
    pe = res["breakdown"]["PE估值"]
    assert pe["status"] == "ok" and pe.get("pe_source") == "snapshot_sina"
    assert "新浪口径" in pe["detail"] and "8.9" in pe["detail"]


def test_pe_source_eastmoney_when_info_cached(screener, monkeypatch):
    """在线且 `_info_cache` 有东财 PE ⇒ pe_source=eastmoney，明细**不带**本地快照标注。"""
    screener._info_cache["600028"] = {"pe_ratio": 15.0, "name": "中国石化"}
    saved = screener._get_kline_data
    saved_news = screener._score_news_factor
    screener._get_kline_data = lambda code: make_kline_df(list(range(1, 31)))
    screener._score_news_factor = lambda code, max_score=15, name=None, **_kw: {
        "score": None, "max": max_score, "status": "unknown", "detail": "占位"}
    screener._flow_fetcher = MagicMock()
    screener._flow_fetcher.get_fund_flow_data.return_value = {"data_success": False}
    try:
        weights, max_scores = _v2_weights()
        res = screener._score_single("600028", weights, max_scores)
    finally:
        screener._get_kline_data = saved
        screener._score_news_factor = saved_news
    assert res["pe_source"] == "eastmoney"
    assert "新浪口径" not in res["breakdown"]["PE估值"]["detail"]


def test_screen_picks_persists_pe_source(tmp_path, monkeypatch):
    """⭐ `screen_picks.pe_source` 落库（B2③ 的验收硬判据）。"""
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch)
    from db import store
    import core.screen_history as sh
    result = {"market_signal": None, "candidates": [{
        "股票代码": "600028", "股票名称": "中国石化", "score": 66.0,
        "score_coverage": 0.88, "factors": {}, "pe_source": "snapshot_sina",
        "策略": "主力资金",
    }]}
    n = sh.save_screen_result(result, source="ui", pick_date="2026-10-16")
    assert n == 1
    rows = store.list_screen_picks(limit=5, stock_code="600028")
    assert rows and rows[0]["pe_source"] == "snapshot_sina"


def test_offline_run_marks_degraded_with_reason(screener, tmp_path, monkeypatch):
    """⭐ F3（2026-10-18 审查）：离线名单必须**结果级**标 degraded + 原因。

    否则"5 只 100 分的离线名单"和"5 只 89 分的在线名单"在 UI 上完全无法区分
    （页 1 只渲染 degraded/degrade_reasons）——用户只会看到"今天选出了满分票"。
    """
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch)
    from db import store
    store.save_market_snapshot("2026-10-16", [
        {"stock_code": "600028", "stock_name": "中国石化", "industry": "石油行业",
         "board": "主板沪市", "price": 5.53, "change_pct": 0.5,
         "pe": 8.9, "pb": 0.8, "market_cap": 1.0e11, "float_cap": 9e10,
         "turnover": 0.5, "amount": 3.0e8}])
    store.save_fund_flow_snapshot("2026-10-16", [{
        "stock_code": "600028", "net_today": 1.78e7, "net_3d": 3.77e8,
        "net_5d": 6.99e8, "net_10d": 1.919e9,
        "amount": 1.464e9, "change_pct": 2.03, "price": 5.53}])
    screener._offline = True
    monkeypatch.setattr(screener, "_run_strategies",
                        lambda *a, **k: [{"股票代码": "600028", "股票名称": "中国石化",
                                          "策略": "主力资金"}])
    monkeypatch.setattr(screener, "_prefetch", lambda *a, **k: {"ok": 0, "failed": 0})
    monkeypatch.setattr(screener, "_industry_map", lambda: {})
    import types
    screener.last_strategy_meta = []
    res = screener.run(max_stocks=5)
    assert res["degraded"] is True
    assert any("离线模式" in r for r in res["degrade_reasons"])
    assert res["candidates"] and res["candidates"][0].get("pe_source") == "snapshot_sina"


def test_online_run_with_snapshot_pe_also_marks_degraded(screener, tmp_path, monkeypatch):
    """在线但部分候选 PE 来自快照 ⇒ 同样要标（口径混排必须可见）。"""
    screener._offline = False
    monkeypatch.setattr(screener, "_run_strategies",
                        lambda *a, **k: [{"股票代码": "600028", "股票名称": "中国石化",
                                          "策略": "主力资金"}])
    monkeypatch.setattr(screener, "_prefetch", lambda *a, **k: {"ok": 0, "failed": 0})
    monkeypatch.setattr(screener, "_industry_map", lambda: {})
    # 直接在候选上放 pe_source（模拟评分结果），走 run 的收尾逻辑
    orig_rank = screener._rank_and_select

    def _rank(scored, max_stocks, return_all=False):
        for s in scored:
            s["pe_source"] = "snapshot_sina"
        # ⚠️ 必须透传 return_all（2026-10-22）：`run` 现在用
        #    `_rank_and_select(..., return_all=True)` 取完整排序做候选扩池，
        #    假函数不透传会 TypeError。
        return orig_rank(scored, max_stocks, return_all=return_all)

    monkeypatch.setattr(screener, "_rank_and_select", _rank)
    import data.bars as _bars
    monkeypatch.setattr(_bars, "staleness",
                        lambda *a, **k: {"data_lagging": False, "lag_days": 0,
                                         "last_date": None, "reason": ""})
    monkeypatch.setattr("data.bars.get_bars", lambda *a, **k: None)
    screener._kline_cache.clear()
    screener.last_strategy_meta = []
    res = screener.run(max_stocks=5)
    assert res["degraded"] is True
    assert any("新浪口径" in r for r in res["degrade_reasons"])


# ================================================================
# 资金流预取：**所有打分入口都必须调**（2026-10-24）
# ================================================================
# 事故：10-23 把"资金流并发预取"只挂在 `score_codes()`（页面/工具走那条），
# 而调度器走 `run()` → 直接调 `_score_candidates` ⇒ **预取一次都没执行**。
# 证据：10-23 15:30 那轮日志里 `资金流并发预取` 出现 0 次，
# 打分阶段 293s / 118 只 = 2.5s/只（本地实测 0.12s/只）。
#
# 这正是铁律"能力只在一半路径生效"的又一例 ⇒ 用**结构性**断言守住：
# 凡调用 `_score_candidates` 的方法，必须同时调用 `_ensure_fund_prefetched`。

class TestFundPrefetchOnEveryEntry:
    def test_every_score_entry_prefetches(self):
        import ast
        import io
        import core.screener as S

        tree = ast.parse(io.open(S.__file__, encoding="utf-8").read())

        def _calls(fn, name):
            return any(isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute)
                       and n.func.attr == name
                       for n in ast.walk(fn))

        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if _calls(node, "_score_candidates") and not _calls(node, "_ensure_fund_prefetched"):
                offenders.append(node.name)
        assert not offenders, (
            f"这些入口调了 `_score_candidates` 却没调 `_ensure_fund_prefetched`：{offenders}\n"
            "⇒ 该路径下资金流会退化成逐只串行（实测 2.5s/只，118 只=293s）。")

    def test_prefetch_is_skipped_for_empty_rows(self, screener):
        assert screener._ensure_fund_prefetched([])["skipped"] is True

    def test_prefetch_forwards_codes_once(self, screener, monkeypatch):
        seen = {}

        def _fake(codes):
            seen["codes"] = list(codes)
            return {"ok": len(seen["codes"]), "total": len(seen["codes"])}

        monkeypatch.setattr(screener, "_prefetch_fund", _fake)
        screener._ensure_fund_prefetched([{"股票代码": "600000"}, {"股票代码": "600001"}])
        assert seen["codes"] == ["600000", "600001"]

    def test_prefetch_failure_does_not_block(self, screener, monkeypatch):
        """预取是加速手段：抛异常也必须返回（不阻断打分）。"""
        def _boom(codes):
            raise RuntimeError("源挂了")

        monkeypatch.setattr(screener, "_prefetch_fund", _boom)
        out = screener._ensure_fund_prefetched([{"股票代码": "600000"}])
        assert out["ok"] == 0 and "源挂了" in out["error"]
