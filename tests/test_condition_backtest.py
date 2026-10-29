"""条件回测契约测试（`core/condition_backtest.py`）

重点守护 **前视防护**：任何一天 T 的判定只能用 `date <= T` 的数据。
这类 bug 的形态极其隐蔽——回测"稳赚"，指标漂亮，而**看不出错**。
所以这里用"把未来价格改成极端值"的方式正面验证：改未来不该影响过去。

其余守护：T+1 开盘成交、涨跌停放弃、持有到期/止损卖出、
以及"不可回放指标必须显式列出而不是静默丢弃"。
"""

import pandas as pd
import pytest

from core import condition_backtest as cb
from core import nl_screener as nl


# ================================================================
# 夹具
# ================================================================

def _bars(closes, start="2025-01-02"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [1_000_000.0] * len(closes),
        "amount": [100_000_000.0] * len(closes),
        "turnover_pct": [2.0] * len(closes),
    })


def _gen(text, data: dict, start="2025-01-02", end="2025-06-30", **kw):
    """构造生成器并直接注入数据（绕过 preload / 网络）"""
    g = cb.ConditionSignalGenerator(list(data), nl.parse(text), **kw)
    for code, closes in data.items():
        g._data[code] = cb.ConditionSignalGenerator._prepare(_bars(closes, start))
    g._loaded_range = (start, end)
    return g


# ================================================================
# 1. 可回放性方案
# ================================================================

class TestPlan:
    def test_replayable_and_skipped_split(self):
        p = cb.plan(nl.parse("均线多头排列、市盈率低于20、换手率低于5%"))
        assert {c.field for c in p.replayable} == {"ma_bullish", "turnover"}
        assert [label for label, _ in p.skipped][0].startswith("市盈率")

    def test_skipped_carries_reason(self):
        """不可回放的条件必须带**原因**，不能只是消失"""
        p = cb.plan(nl.parse("新闻正面率大于50%"))
        assert p.skipped
        label, why = p.skipped[0]
        assert "新闻正面率" in label
        assert "无稳定免费源" in why, "必须说明为什么回放不了"

    def test_every_not_replayable_field_has_a_reason(self):
        for field_name in ("pe", "market_cap", "fund_net", "news_positive", "score"):
            assert field_name in cb.NOT_REPLAYABLE, field_name
            assert cb.NOT_REPLAYABLE[field_name]

    def test_replayable_and_not_replayable_are_disjoint_and_complete(self):
        from core.indicator_registry import INDICATORS
        assert not (cb.REPLAYABLE_FIELDS & set(cb.NOT_REPLAYABLE))
        assert cb.REPLAYABLE_FIELDS | set(cb.NOT_REPLAYABLE) == set(INDICATORS)

    def test_plan_ok_requires_at_least_one_replayable(self):
        assert cb.plan(nl.parse("均线多头排列")).ok is True
        assert cb.plan(nl.parse("市盈率低于20")).ok is False

    def test_plan_to_dict_reports_counts(self):
        d = cb.plan(nl.parse("均线多头排列、市盈率低于20")).to_dict()
        assert d["replayable_count"] == 1
        assert d["skipped_count"] == 1
        assert d["total_conditions"] == 2


# ================================================================
# 2. 前视防护（本文件最重要的部分）
# ================================================================

def test_slice_never_returns_future_rows():
    g = _gen("均线多头排列", {"600519": [10.0] * 30})
    cut = "2025-02-10"
    bars = g._slice("600519", cut)
    assert (bars["date"] <= pd.Timestamp(cut)).all()
    assert len(bars) < len(g._data["600519"]), "必须真的裁掉了后续行"


def test_future_prices_do_not_change_past_judgement():
    """把 T 之后的价格改成极端值，T 日的判定必须**一字不变**。

    如果哪天有人图省事把全量数据喂给判定，这条会立刻变红——
    而那种错在回测报告里只会表现为"策略很赚钱"，肉眼看不出来。
    """
    flat = [10.0] * 25
    later = [10.0] * 25 + [999.0] * 5
    d = "2025-02-03"                       # 落在前 25 天内

    g_flat = _gen("均线多头排列", {"600519": flat})
    g_spike = _gen("均线多头排列", {"600519": later})

    assert g_flat.passes("600519", d) == g_spike.passes("600519", d)
    f1, f2 = g_flat.facts_on("600519", d), g_spike.facts_on("600519", d)
    for key in f1:
        assert f1[key] == f2[key], f"{key} 受到了未来数据影响"


def test_facts_on_returns_empty_when_history_too_short():
    """K 线不足 21 根 → 返回空 facts（→ 判定为不通过），不得凭少量数据硬算"""
    g = _gen("均线多头排列", {"600519": [10.0] * 10})
    assert g.facts_on("600519", "2025-01-15") == {}
    assert g.passes("600519", "2025-01-15") is False


def test_amount_unit_is_yi_not_yuan():
    """日线 amount 是**元**，facts 里必须是**亿元**（否则量级差 1e8）"""
    g = _gen("成交额大于0.5亿", {"600519": [10.0] * 25})
    facts = g.facts_on("600519", "2025-02-03")
    assert facts["amount"] == pytest.approx(1.0), "1e8 元 = 1 亿元"
    assert 0 < facts["amount"] < 1000


# ================================================================
# 3. 判定：复用 nl_screener.evaluate
# ================================================================

def test_passes_uses_shared_evaluation():
    """回测判定与选股判定必须是同一套实现（否则两边会慢慢分叉）"""
    g = _gen("均线多头排列", {"600519": [10.0 + i * 0.2 for i in range(30)]})
    d = "2025-02-10"
    facts = g.facts_on("600519", d)
    expected = nl.evaluate(facts, g.subquery)["pass"]
    assert g.passes("600519", d) == expected


def test_uptrend_satisfies_ma_bullish():
    g = _gen("均线多头排列", {"600519": [10.0 + i * 0.3 for i in range(30)]})
    assert g.passes("600519", "2025-02-10") is True


def test_downtrend_fails_ma_bullish():
    g = _gen("均线多头排列", {"600519": [20.0 - i * 0.3 for i in range(30)]})
    assert g.passes("600519", "2025-02-10") is False


def test_non_replayable_condition_is_not_in_subquery():
    """不可回放的条件不得混进回测判定（否则等于用当期值当历史值）"""
    g = _gen("均线多头排列、市盈率低于20", {"600519": [10.0 + i * 0.3 for i in range(30)]})
    assert {c.field for c in g.subquery.conditions} == {"ma_bullish"}


# ================================================================
# 4. 成交：T+1 开盘 / 涨跌停 / 卖出
# ================================================================

def _trades(g, start="2025-01-02", end="2025-06-30"):
    return g.generate_signals(start, end)


def test_signal_fills_at_next_day_open():
    """T 日收盘产生的信号，在 **T+1 开盘**成交——这是防前视的关键一环"""
    closes = [10.0 + i * 0.3 for i in range(30)]
    g = _gen("均线多头排列", {"600519": closes}, holding_days=5)
    trades = _trades(g)
    buys = [t for t in trades if t["direction"] == "买入"]
    assert buys, "上涨趋势 + 均线多头 应当产生买入信号"
    first = buys[0]
    # 成交价必须等于该日开盘价（本夹具 open == close），而不是前一日收盘
    bars = g._data["600519"]
    same_day = bars[bars["date"] == pd.Timestamp(first["date"])]
    assert len(same_day) == 1
    assert first["price"] == pytest.approx(float(same_day.iloc[0]["open"]))


def test_no_trade_on_final_day():
    """最后一天不再产生新信号（否则信号永远无法成交）"""
    g = _gen("均线多头排列", {"600519": [10.0 + i * 0.3 for i in range(30)]})
    trades = _trades(g)
    if trades:
        assert all(t["date"] < "2025-06-30" for t in trades)


def test_holding_period_triggers_sell():
    closes = [10.0 + i * 0.3 for i in range(60)]
    g = _gen("均线多头排列", {"600519": closes}, holding_days=5)
    trades = _trades(g)
    sells = [t for t in trades if t["direction"] == "卖出"]
    assert sells, "持有到期应产生卖出"
    assert any("到期" in t["reason"] for t in sells)


def test_stop_loss_triggers_sell():
    """买入后暴跌 → 止损卖出（默认 8%）

    ⚠️ 数据必须**自洽**：第一版写成"先建上涨趋势、再单独改 close 列"，
    结果 open 仍是原值、前收被砍半 → 被判成"开盘涨停 103%" → 买单被正确拦掉，
    用例假红。真实数据里 open 与前收不会这样脱节。
    """
    closes = [10.0 + i * 0.3 for i in range(30)] + [6.0] * 6
    g = _gen("均线多头排列", {"600519": closes}, holding_days=20, stop_loss_pct=8.0)
    trades = _trades(g)
    sells = [t for t in trades if t["direction"] == "卖出"]
    assert sells, "暴涨后暴跌应当触发止损卖出"
    assert any("止损" in t["reason"] for t in sells), [t["reason"] for t in sells]


def test_limit_up_open_blocks_buy():
    """T+1 开盘即涨停 → 放弃该单（不追涨）"""
    closes = [10.0 + i * 0.3 for i in range(30)]
    g = _gen("均线多头排列", {"600519": closes}, holding_days=5)
    bars = g._data["600519"]
    # 构造 T+1 开盘涨停：把某天 open 设为前收 × 1.10
    idx = 25
    prev_close = float(bars.iloc[idx - 1]["close"])
    g._data["600519"].loc[idx, "open"] = round(prev_close * 1.10, 2)
    block_date = bars.iloc[idx]["date"].strftime("%Y-%m-%d")
    trades = _trades(g)
    assert not any(t["direction"] == "买入" and t["date"] == block_date for t in trades), \
        "开盘涨停当日不得成交买入"


def test_no_signals_when_no_replayable_condition():
    """全是不可回放条件 → 不出信号（而不是拿当期值硬跑一遍）"""
    g = _gen("市盈率低于20", {"600519": [10.0] * 30})
    assert g.generate_signals("2025-01-02", "2025-06-30") == []


def test_missing_code_is_not_retried():
    """取不到数据的标的只尝试一次（本项目踩过"缺失标的每天联网"的坑）"""
    g = cb.ConditionSignalGenerator(["600519", "999999"], nl.parse("均线多头排列"))
    g._data["600519"] = cb.ConditionSignalGenerator._prepare(_bars([10.0] * 25))
    g._no_data.add("999999")
    assert g.passes("999999", "2025-02-10") is False
    assert g.facts_on("999999", "2025-02-10") == {}


def test_max_positions_is_respected():
    """持仓上限生效：构造 6 只都满足条件，最多只买 max_positions 只"""
    data = {f"60000{i}": [10.0 + j * 0.3 for j in range(30)] for i in range(6)}
    g = _gen("均线多头排列", data, max_positions=2, holding_days=20)
    trades = _trades(g)
    # 同一天最多 2 笔买入
    by_date = {}
    for t in trades:
        if t["direction"] == "买入":
            by_date.setdefault(t["date"], []).append(t)
    for d, items in by_date.items():
        assert len(items) <= 2, f"{d} 买了 {len(items)} 只，超过上限"
