"""测试市场检测器 market_detector.py"""

import re

import pandas as pd
import pytest
from unittest.mock import patch

from core.market_detector import MarketDetector, MarketSignal


def test_market_signal_defaults():
    s = MarketSignal(
        total_score=50.0, market_state="中性震荡", signal="观望",
        position_limit=0.3, strategy_bias="均衡型",
        operation_advice="谨慎操作", scores={}, indicators={},
        risk_warnings=[], timestamp="2026-01-01",
    )
    assert isinstance(s.total_score, float)
    assert isinstance(s.position_limit, float)


def test_detector_init():
    d = MarketDetector()
    assert len(d.weights) >= 10
    assert "hs300_ma20_deviation" in d.weights


# ===== 量能指标口径（2026-09-28 修正）=====
#
# 事故：`ak.stock_zh_index_daily` 的 `volume` 列是**成交量（股）**，
# 但原实现 `/1e8` 后渲染成"两市成交965亿" → 复盘报告写出
# "两市成交965亿元""成交额回升至1000亿元以上"，真实两市成交额约 1 万亿元，
# **小了一个数量级**，模型还据此继续演绎（缩量恐慌 → 结论被放大）。

def _index_df(last_date: str, last_vol: float = 43_853_041_200,
              base_vol: float = 40_000_000_000, n: int = 25) -> pd.DataFrame:
    dates = pd.date_range(end=last_date, periods=n, freq="D").strftime("%Y-%m-%d")
    vols = [base_vol] * n
    vols[-1] = last_vol
    return pd.DataFrame({"date": dates, "open": 1.0, "high": 1.0, "low": 1.0,
                         "close": 1.0, "volume": vols})


def _score_volume(d, bar_date: str, expected: str) -> dict:
    # B2②：数据走本地 index_panel ⇒ patch `MarketDetector._index`（本地优先层），
    # mock akshare 不再生效。
    with patch.object(MarketDetector, "_index",
                      return_value=_index_df(bar_date)), \
         patch("core.market_detector.last_trading_day", return_value=expected):
        return d._score_volume_vs_ma20()


def test_volume_detail_uses_shares_not_turnover():
    """必须写成"成交量 X 亿股"，绝不能渲染成金额口径。"""
    res = _score_volume(MarketDetector(), "2026-09-28", "2026-09-28")
    detail = res["detail"]
    assert "成交量" in detail, detail
    assert "亿股" in detail, detail
    assert "成交额" not in detail, detail
    # 关键：不得再出现 "成交NNN亿(" 这种会被读成金额的写法
    assert not re.search(r"成交\d+(?:\.\d+)?亿\s*[\(（]", detail), detail
    assert res["score"] <= res["max_score"]


def test_volume_detail_flags_stale_bar_date():
    """bar 不是最近交易日 → 必须写明数据日期，否则会把上一交易日的量当今日量。"""
    res = _score_volume(MarketDetector(), "2026-09-24", "2026-09-28")
    assert "2026-09-24" in res["detail"], res["detail"]
    assert "上一交易日" in res["detail"], res["detail"]


def test_volume_detail_no_stale_note_when_date_matches():
    res = _score_volume(MarketDetector(), "2026-09-28", "2026-09-28")
    assert "上一交易日" not in res["detail"], res["detail"]


# ===== 各子方法异常处理 =====

def test_score_hs300_ma20_error_handling():
    d = MarketDetector()
    with patch("core.market_detector.ak.stock_zh_index_daily",
               side_effect=Exception("network")):
        result = d._score_hs300_ma20()
    assert "score" in result
    assert result["score"] <= result["max_score"]


def test_score_sh_macd_error_handling():
    d = MarketDetector()
    with patch("core.market_detector.ak.stock_zh_index_daily",
               side_effect=Exception("fail")):
        result = d._score_sh_macd()
    assert "score" in result


def test_score_cyb_relative_error_handling():
    d = MarketDetector()
    with patch("core.market_detector.ak.stock_zh_index_daily",
               side_effect=Exception("fail")):
        result = d._score_cyb_relative_strength()
    assert "score" in result


# ===== detect() 集成 =====

def test_detect_fully_mocked():
    """全 mock → 返回合法 MarketSignal"""
    d = MarketDetector()

    def mock_method(name, ret):
        return patch.object(d, name, return_value=ret)

    sub_methods = [
        "_score_hs300_ma20", "_score_sh_macd", "_score_cyb_relative_strength",
        "_score_volume_vs_ma20", "_score_advance_decline", "_score_north_flow",
        "_score_limit_up_down", "_score_max_consecutive_board", "_score_market_breadth",
        "_score_ivix", "_score_sector_rotation", "_score_size_style",
        "_score_margin_balance", "_score_upgrade_rate",
    ]

    patches = [mock_method(m, {"score": 5, "max_score": 10})
               for m in sub_methods]
    for p in patches:
        p.start()

    try:
        result = d.detect()
    finally:
        for p in patches:
            p.stop()

    assert isinstance(result, MarketSignal)
    assert 0 <= result.total_score <= 100
    assert result.position_limit >= 0.0
    assert result.market_state in [
        "牛市", "偏强震荡", "中性震荡", "偏弱震荡", "熊市",
    ]


def test_detect_score_bounds():
    d = MarketDetector()

    scores_map = {
        "_score_hs300_ma20": {"score": 5, "max_score": 10},
        "_score_sh_macd": {"score": 5, "max_score": 10},
        "_score_cyb_relative_strength": {"score": 5, "max_score": 10},
        "_score_volume_vs_ma20": {"score": 6, "max_score": 12},
        "_score_advance_decline": {"score": 4, "max_score": 8},
        "_score_north_flow": {"score": 3, "max_score": 5},
        "_score_limit_up_down": {"score": 4, "max_score": 8},
        "_score_max_consecutive_board": {"score": 2, "max_score": 5},
        "_score_market_breadth": {"score": 3, "max_score": 7},
        "_score_ivix": {"score": 2, "max_score": 5},
        "_score_sector_rotation": {"score": 2, "max_score": 5},
        "_score_size_style": {"score": 2, "max_score": 5},
        "_score_margin_balance": {"score": 2, "max_score": 5},
        "_score_upgrade_rate": {"score": 2, "max_score": 5},
    }

    patches = [patch.object(d, name, return_value=ret)
               for name, ret in scores_map.items()]
    for p in patches:
        p.start()
    try:
        result = d.detect()
    finally:
        for p in patches:
            p.stop()

    assert 0 <= result.total_score <= 100
    assert len(result.scores) > 0


# ===== v2（M0 0.4）：不可用指标不计分也不计入分母 =====

_SUB_METHODS = [
    "_score_hs300_ma20", "_score_sh_macd", "_score_cyb_relative_strength",
    "_score_volume_vs_ma20", "_score_advance_decline", "_score_north_flow",
    "_score_limit_up_down", "_score_max_consecutive_board", "_score_market_breadth",
    "_score_ivix", "_score_sector_rotation", "_score_size_style",
    "_score_margin_balance", "_score_upgrade_rate",
]


def _detect_with(d, overrides):
    """在完全 mock 的环境下跑 detect()（含涨停池，避免联网）"""
    patches = [patch.object(d, "_get_zt_pool_info", return_value={})]
    defaults = {"score": 10, "max_score": 10}
    for m in _SUB_METHODS:
        patches.append(patch.object(d, m, return_value=overrides.get(m, defaults)))
    for p in patches:
        p.start()
    try:
        return d.detect()
    finally:
        for p in patches:
            p.stop()


def test_north_flow_is_marked_unavailable():
    """北向日频数据自 2024-08-19 停止披露 → 指标停用，不再给虚假的 2/5 分"""
    d = MarketDetector()
    r = d._score_north_flow()
    assert r["unavailable"] is True
    assert r["max_score"] == 0


def test_unavailable_factor_does_not_deflate_total_score():
    """停用/取数失败的指标不得拉低总分，其余指标按有效权重归一化。

    原实现把取数失败当 0 分计入：既能造成"接口挂了 = 市场看空"，
    也让总分随数据源可用性漂移。
    """
    d = MarketDetector()
    baseline = _detect_with(d, {}).total_score
    assert baseline == pytest.approx(100.0, abs=0.5), "全部满分时总分应为 100"

    mixed = {"_score_north_flow": {"score": 0, "max_score": 0, "unavailable": True}}
    score = _detect_with(d, mixed).total_score
    assert score == pytest.approx(baseline, abs=0.5), "停用一个指标不应改变总分标度"


def test_unavailable_factor_is_reported_in_details():
    d = MarketDetector()
    result = _detect_with(d, {"_score_north_flow": {"score": 0, "max_score": 0,
                                                    "unavailable": True,
                                                    "detail": "已停用"}})
    assert result.scores["north_flow"]["unavailable"] is True
    assert result.scores["north_flow"]["weighted_score"] == 0.0


# ===== 覆盖率闸门（2026-10-18，P0-1 断网验收暴露）=====
#
# 事故：断网跑选股，14 个指标多数"不可评"，只剩少数低分指标参与归一化
#   ⇒ 总分 18.2 ⇒ 判"熊市" ⇒ position_limit=0 ⇒ **自动跳过选股**。
# 用户看到"市场极度弱势"，实际是"数据没取到" —— 「不可评 ≠ 0 分」的翻版：
# 分母也会失真（只剩 2/14 指标 ⇒ 结论没代表性）。

def test_low_coverage_degrades_to_neutral_not_bear():
    """⭐⭐ 可评指标不足一半 ⇒ 必须降级"数据不足（中性假设）"，且**不阻断选股**。"""
    d = MarketDetector()
    # 只留 1 个指标可评（给最低分），其余 13 个全标不可评
    overrides = {m: {"score": 0, "max_score": 10, "unavailable": True,
                     "detail": "离线"}
                 for m in _SUB_METHODS}
    first = _SUB_METHODS[0]
    overrides[first] = {"score": 0, "max_score": 10}      # 唯一可评且 0 分
    sig = _detect_with(d, overrides)
    assert sig.market_state == "数据不足（中性假设）", \
        f"低覆盖不许判成{sig.market_state}（会被读成市场真的差）"
    assert sig.position_limit > 0, "数据不足不应阻断选股（只提示不阻断）"
    assert any("覆盖率" in w and "50%" in w for w in sig.risk_warnings), \
        "必须显式说明有多少指标不可评"


def test_full_coverage_keeps_original_thresholds():
    """覆盖率充足时判档行为不变（不许因新闸门改动原有语义）。"""
    d = MarketDetector()
    sig = _detect_with(d, {})                              # 全部满分
    assert sig.market_state == "牛市" and sig.total_score >= 80


# ===== 融资余额指标：口径修正回归（2026-09-25 实测发现）=====

class TestMarginBalanceIndicator:
    """`_score_margin_balance` 曾是一个"恒满分"的假指标。

    旧实现三处错误互相掩盖：
      1. 查询日用 `datetime.now()`，而上交所融资数据**收盘后才发布** → 盘中必然取不到；
      2. 用 `stock_margin_detail_sse`（**逐标的明细** 2000 行）却取 `.iloc[-1]`
         —— 那是**某一只标的**的余额（实测 168,474,612），不是全市场；
      3. 该字段单位是**元**，判分阈值却按"亿"设计（>50 得 5 分）
         → 以元计恒 > 50 → **无条件满分**，5% 权重白送，总分虚高。
    正确口径：`stock_margin_sse`（交易所汇总按日）取最近两个交易日 `融资余额` 之差 ÷1e8。
    """

    @staticmethod
    def _df(prev, latest):
        import pandas as pd
        return pd.DataFrame({"信用交易日期": ["20260924", "20260923"],
                             "融资余额": [latest, prev]})   # 接口按日期倒序返回

    def test_reports_two_day_change_in_yi(self):
        d = MarketDetector()
        with patch("core.market_detector.ak.stock_margin_sse",
                   return_value=self._df(7.28e11, 7.28e11 - 3e9)):   # 降 30 亿
            r = d._score_margin_balance()
        assert r["raw"] == pytest.approx(-30.0, abs=0.1), r
        assert r["score"] == 0, "降 30 亿应得 0 分（<-20 亿）"
        assert "20260923→20260924" in r["detail"]

    def test_regression_detail_table_must_not_be_used(self):
        """回归：绝不能再走"逐标的明细"并把它当作全市场融资余额。

        旧实现下该路径恒得满分——某只标的余额 1.68e8（元）与阈值 50 比较恒为真。
        """
        d = MarketDetector()
        with patch("core.market_detector.ak.stock_margin_detail_sse",
                   side_effect=AssertionError("不应调用逐标的明细接口")), \
             patch("core.market_detector.ak.stock_margin_sse",
                   return_value=self._df(7.28e11, 7.29e11)):          # 增 10 亿
            r = d._score_margin_balance()
        assert r["score"] == 3 and r["raw"] == pytest.approx(10.0, abs=0.1), r

    def test_query_date_is_last_trading_day_not_today(self):
        """查询日必须是最近交易日：用"今天"在盘中必然取不到（收盘后才发布）"""
        from data import trading_calendar as cal
        d = MarketDetector()
        seen = {}

        def fake(start_date=None, end_date=None):
            seen["end"] = end_date
            return self._df(7.28e11, 7.29e11)

        with patch("core.market_detector.ak.stock_margin_sse", side_effect=fake):
            d._score_margin_balance()
        expected = cal.last_trading_day().replace("-", "")
        assert seen["end"] == expected, f"应查最近交易日 {expected}，实际 {seen['end']}"

    def test_insufficient_rows_falls_back_and_says_why(self):
        import pandas as pd
        d = MarketDetector()
        one_row = pd.DataFrame({"信用交易日期": ["20260924"], "融资余额": [7.28e11]})
        with patch("core.market_detector.ak.stock_margin_sse", return_value=one_row):
            r = d._score_margin_balance()
        assert r["score"] == 2, "取数不足时给保守默认分"
        assert "失败" in r["detail"], "原因要写进 detail（体检脚本据此识别）"


# ===== 口径约定：取不到数据必须标 unavailable（2026-09-28）=====
#
# 事故：除 north_flow 外，其余 `_score_*` 在取数失败/数据不足时都只返回一个
# **中间默认分**（5/10、4/8、3/7…）而**不打 `unavailable` 标志**，于是 `detect()`
# 把它当作真实得分计入总分与分母 —— "数据缺失"被伪装成"中等偏弱"。
# 当日 `涨跌家数(4/8)`、`市场宽度(3/7)` 因东财快照断连被顶上默认分，白送 15% 权重。

def test_market_breadth_empty_snapshot_is_unavailable():
    # B2②：本地统计不可用（zt_stats unavailable）才走快照回退路径
    d = MarketDetector()
    with patch.object(MarketDetector, "_zt_stats",
                      return_value={"available": False, "caveat": "test"}), \
         patch("core.market_detector.get_market_snapshot", return_value=None):
        r = d._score_market_breadth()
    assert r.get("unavailable") is True, r
    assert "失败" in r["detail"]


def test_market_breadth_exception_is_unavailable():
    d = MarketDetector()
    with patch.object(MarketDetector, "_zt_stats",
                      return_value={"available": False, "caveat": "test"}), \
         patch("core.market_detector.get_market_snapshot",
               side_effect=Exception("boom")):
        r = d._score_market_breadth()
    assert r.get("unavailable") is True, r


def test_market_breadth_from_local_panel():
    """⭐ B2②：本地面板统计命中 ⇒ 直接可用（不再抽样联网）。"""
    d = MarketDetector()
    with patch.object(MarketDetector, "_zt_stats", return_value={
            "available": True, "trade_date": "2026-10-16",
            "above_ma20_ratio": 16.4, "above_ma20_checked": 5531}):
        r = d._score_market_breadth()
    assert r.get("unavailable") is not True, r
    assert "本地面板" in r["detail"] and "5531" in r["detail"]


def test_zt_pool_from_local_panel():
    """⭐ B2②：涨停池本地版带 source=panel 与 caveat。"""
    d = MarketDetector()
    with patch.object(MarketDetector, "_zt_stats", return_value={
            "available": True, "trade_date": "2026-10-16",
            "limit_up_count": 66, "limit_down_count": 10,
            "max_consecutive": 11, "upgrade_rate": 9.1,
            "caveat": "本地推算"}):
        info = d._get_zt_pool_info()
    assert info["success"] is True and info["source"] == "panel"
    assert info["limit_up_count"] == 66 and info["max_consecutive"] == 11
    r = d._score_upgrade_rate(info)
    assert r["status"] if False else r.get("unavailable") is not True
    assert "本地面板" in r["detail"]


def test_advance_decline_failure_is_unavailable():
    d = MarketDetector()
    with patch("core.market_detector.get_market_snapshot",
               side_effect=Exception("boom")):
        r = d._score_advance_decline()
    assert r.get("unavailable") is True, r
    assert "失败" in r["detail"]


@pytest.mark.parametrize("method", [
    "_score_hs300_ma20", "_score_sh_macd",
    "_score_cyb_relative_strength", "_score_volume_vs_ma20",
])
def test_index_daily_failure_is_unavailable(method):
    # B2②：本地 index_panel 有数据 ⇒ 可评；要测"全失败"必须本地+联网都拿不到
    # （patch `MarketDetector._index` 返回 None，同时让它离线以防真实回退）。
    d = MarketDetector()
    d._offline = True
    with patch.object(MarketDetector, "_index", return_value=None), \
         patch("data.bars.offline_mode", return_value=True):
        r = getattr(d, method)()
    assert r.get("unavailable") is True, (method, r)


def test_index_panel_hit_is_available():
    """⭐ B2②：本地 index_panel 命中 ⇒ 可评（离线模式零联网）。"""
    import pandas as pd
    df = pd.DataFrame({
        "date": pd.date_range("2026-08-01", periods=40, freq="B"),
        "open": 4000.0, "high": 4010.0, "low": 3990.0, "close": 4005.0,
        "volume": 1e9,
    })
    df.attrs = {"source": "index_panel", "symbol": "sh000300", "rows": 40,
                "last_date": "2026-09-26"}
    d = MarketDetector()
    with patch.object(MarketDetector, "_index", return_value=df), \
         patch("data.index_bars.index_fresh_enough", return_value=True):
        r = d._score_hs300_ma20()
    assert r.get("unavailable") is not True, r
    assert "沪深300" in r["detail"]


def test_all_fallback_branches_carry_unavailable_flag():
    """结构性守护：源码里所有硬编码默认分的失败分支都必须带 unavailable。

    用 AST/正则扫源码，新增 `_score_*` 时若忘记打标即失败——
    这类"忘了标"的缺口靠逐个 mock 是覆盖不全的。
    """
    import pathlib
    import re as _re
    src = pathlib.Path(__file__).resolve().parents[1] / "core" / "market_detector.py"
    text = src.read_text(encoding="utf-8")
    pat = _re.compile(
        r'return \{"score": \d+, "max_score": \d+, "(?:raw|detail)":')
    bad = [ln.strip() for ln in text.splitlines()
           if pat.search(ln) and "unavailable" not in ln]
    assert not bad, f"以下失败分支缺少 unavailable 标志：{bad}"


def test_unavailable_flag_makes_default_score_inert():
    """标了 unavailable → 保守默认分不再影响总分（既不计分也不进分母）。"""
    d = MarketDetector()

    def run(mb, ad):
        return _detect_with(d, {
            "_score_ivix": {"score": 2, "max_score": 5},
            "_score_market_breadth": mb,
            "_score_advance_decline": ad,
        }).total_score

    low = run({"unavailable": True, "score": 3, "max_score": 7},
              {"unavailable": True, "score": 4, "max_score": 8})
    high = run({"unavailable": True, "score": 7, "max_score": 7},
               {"unavailable": True, "score": 8, "max_score": 8})
    assert low == high, "标了 unavailable，默认分就该完全惰性"


def test_default_score_counts_when_not_flagged():
    """反证：不标 unavailable 时默认分确实拉动总分 —— 即当日事故的形态。"""
    d = MarketDetector()

    def run(mb, ad):
        return _detect_with(d, {
            "_score_ivix": {"score": 2, "max_score": 5},
            "_score_market_breadth": mb,
            "_score_advance_decline": ad,
        }).total_score

    low = run({"score": 3, "max_score": 7}, {"score": 4, "max_score": 8})
    high = run({"score": 7, "max_score": 7}, {"score": 8, "max_score": 8})
    assert low < high, "未标 unavailable 时默认分进入了总分（这正是要被修掉的形态）"
