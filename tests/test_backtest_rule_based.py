"""回测信号生成器：T+1 成交与 A 股交易约束（v2 / M0 0.1.1、0.1.3）

背景：原实现是"用 T 日收盘价算出的信号，按 T 日收盘价成交"——评分本身包含 T 日
收盘（以及由它算出的 MA/动量/ATR），属于系统性未来函数，使全部回测指标失效。

本文件锁死四件事：
  1. 决策只用 ≤T 的数据，成交必须发生在 T 之后的 K 线上，且用该 K 线的开盘价；
  2. 涨跌停：开盘涨停买不到（当日放弃）；开盘跌停卖不掉（挂单顺延）；
  3. 停牌日（无 K 线）不得成交；
  4. 持仓天数按交易日计、买入量受当日成交量参与率约束。
"""

import json

import pandas as pd
import pytest
from unittest.mock import patch

import scripts.backtest_rule_based as mod
from scripts.backtest_rule_based import RuleSignalGenerator


def _bars(rows) -> pd.DataFrame:
    """rows: [(date_str, open, close, volume)] → 预加载格式的 DataFrame"""
    return pd.DataFrame([
        {"date": pd.to_datetime(d), "open": float(o), "close": float(c),
         "high": max(o, c), "low": min(o, c), "volume": float(v)}
        for d, o, c, v in rows
    ])


@pytest.fixture
def gen(monkeypatch):
    """单标的、评分恒达标的信号生成器（K 线与日期由各用例注入）"""
    monkeypatch.setattr(mod, "WATCHLIST", ["000001"])
    with patch.object(mod, "DataSourceManager"):
        g = RuleSignalGenerator()
    g.preload = lambda *a, **k: None          # 数据由用例直接放进 _data_cache
    g.score_stock = lambda code, ds: 80.0     # 恒满足 SCORE_THRESHOLD
    g.calc_atr = lambda code, ds: 1.0
    return g


# ===== 1. T+1 成交（核心：消除未来函数）=====

def test_buy_fills_at_next_day_open_not_signal_close(gen):
    gen._data_cache["000001"] = _bars([
        ("2026-01-05", 10.0, 11.0, 1_000_000),   # D1 决策日，收盘 11.0
        ("2026-01-06", 12.0, 12.5, 1_000_000),   # D2 开盘 12.0 —— 应在此成交
        ("2026-01-07", 13.0, 13.5, 1_000_000),
    ])
    trades = gen.generate_signals("2026-01-05", "2026-01-07")

    buys = [t for t in trades if t["direction"] == "买入"]
    assert len(buys) == 1
    assert buys[0]["date"] == "2026-01-06", "成交必须发生在决策日之后的交易日"
    assert buys[0]["price"] == 12.0, "应以次日开盘价成交"
    assert buys[0]["price"] != 11.0, "不得再按决策日收盘价成交（原未来函数）"


def test_sell_fills_after_decision_day(gen):
    """卖出同样错开一日：止损在 T 日收盘判定，T+1 开盘成交"""
    gen._data_cache["000001"] = _bars([
        ("2026-01-05", 10.0, 11.0, 1_000_000),
        ("2026-01-06", 12.0, 9.0, 1_000_000),    # 开盘买入 12.0；收盘 9.0 触发止损(9.5)
        ("2026-01-07", 9.2, 9.3, 1_000_000),     # 次日开盘 9.2 成交
    ])
    trades = gen.generate_signals("2026-01-05", "2026-01-07")

    sells = [t for t in trades if t["direction"] == "卖出"]
    assert sells and sells[0]["date"] == "2026-01-07"
    assert sells[0]["price"] == 9.2
    assert "ATR止损" in sells[0]["reason"]


# ===== 2. 涨跌停约束 =====

def test_buy_blocked_when_next_day_opens_limit_up(gen):
    """开盘即涨停（+10%）买不到 → 当日放弃，改日再评估"""
    gen._data_cache["000001"] = _bars([
        ("2026-01-05", 10.0, 11.0, 1_000_000),   # 收盘 11.0
        ("2026-01-06", 12.1, 12.1, 1_000_000),   # 开盘 = 11.0 × 1.1 → 涨停
        ("2026-01-07", 12.2, 12.3, 1_000_000),
    ])
    trades = gen.generate_signals("2026-01-05", "2026-01-07")

    buys = [t for t in trades if t["direction"] == "买入"]
    assert all(b["date"] != "2026-01-06" for b in buys), "涨停日不得成交"
    assert buys and buys[0]["date"] == "2026-01-07"


def test_sell_deferred_when_next_day_opens_limit_down(gen):
    """开盘即跌停（-10%）卖不掉 → 挂单顺延到下一交易日"""
    gen._data_cache["000001"] = _bars([
        ("2026-01-05", 10.0, 11.0, 1_000_000),
        ("2026-01-06", 12.0, 9.0, 1_000_000),    # 买入 12.0；收盘 9.0 触发止损
        ("2026-01-07", 8.1, 8.1, 1_000_000),     # 开盘 = 9.0 × 0.9 → 跌停，卖不掉
        ("2026-01-08", 8.2, 8.3, 1_000_000),     # 恢复成交
    ])
    trades = gen.generate_signals("2026-01-05", "2026-01-08")

    sells = [t for t in trades if t["direction"] == "卖出"]
    assert sells, "止损最终必须成交"
    assert sells[0]["date"] == "2026-01-08"
    assert "顺延" in sells[0]["reason"]
    assert "ATR止损" in sells[0]["reason"]


def test_limit_pct_by_board():
    assert RuleSignalGenerator._limit_pct("300750") == 0.20   # 创业板
    assert RuleSignalGenerator._limit_pct("301536") == 0.20
    assert RuleSignalGenerator._limit_pct("688981") == 0.20   # 科创板
    assert RuleSignalGenerator._limit_pct("600519") == 0.10   # 主板
    assert RuleSignalGenerator._limit_pct("000001") == 0.10


def test_blocked_by_price_limit_directions():
    up = {"open": 11.0, "close": 11.0, "prev_close": 10.0}
    assert RuleSignalGenerator._blocked_by_price_limit("买入", "000001", up) != ""
    assert RuleSignalGenerator._blocked_by_price_limit("卖出", "000001", up) == ""

    down = {"open": 9.0, "close": 9.0, "prev_close": 10.0}
    assert RuleSignalGenerator._blocked_by_price_limit("卖出", "000001", down) != ""
    assert RuleSignalGenerator._blocked_by_price_limit("买入", "000001", down) == ""

    normal = {"open": 10.3, "close": 10.3, "prev_close": 10.0}
    assert RuleSignalGenerator._blocked_by_price_limit("买入", "000001", normal) == ""
    assert RuleSignalGenerator._blocked_by_price_limit("卖出", "000001", normal) == ""


# ===== 3. 停牌 =====

def test_no_fill_on_suspended_day(gen):
    """停牌日（无 K 线）不得成交，恢复交易后才成交"""
    gen._data_cache["000001"] = _bars([
        ("2026-01-05", 10.0, 11.0, 1_000_000),   # D1 决策
        # 2026-01-06 停牌（无 K 线）
        ("2026-01-07", 12.0, 12.5, 1_000_000),   # D3 恢复
    ])
    trades = gen.generate_signals("2026-01-05", "2026-01-07")

    buys = [t for t in trades if t["direction"] == "买入"]
    assert buys and buys[0]["date"] == "2026-01-07", "停牌日不得成交"
    assert buys[0]["price"] == 12.0


# ===== 4. 持仓天数按交易日 =====

def test_trading_days_held_ignores_weekend(gen):
    """持仓天数按交易日计（原实现用自然日：跨周末会被提前平仓）"""
    gen._data_cache["000001"] = _bars([
        ("2026-01-08", 10.0, 10.0, 1_000_000),   # 周四
        ("2026-01-09", 10.0, 10.0, 1_000_000),   # 周五
        ("2026-01-12", 10.0, 10.0, 1_000_000),   # 下周一
    ])
    gen._build_index()

    assert gen._trading_days_held("000001", "2026-01-08", "2026-01-12") == 2
    # 自然日差为 4 天，原实现会据此提前平仓
    assert (pd.to_datetime("2026-01-12") - pd.to_datetime("2026-01-08")).days == 4


def test_holding_limit_counts_trading_days(gen, monkeypatch):
    """持仓上限按交易日触发：周五入场、下周一只是第 2 个交易日，不应平仓"""
    monkeypatch.setattr(mod, "HOLDING_DAYS_MAX", 3)
    gen._data_cache["000001"] = _bars([
        ("2026-01-07", 10.0, 11.0, 1_000_000),   # 周三 决策
        ("2026-01-08", 12.0, 12.0, 1_000_000),   # 周四 成交（entry）
        ("2026-01-09", 12.0, 12.0, 1_000_000),   # 周五 第 1 个交易日
        ("2026-01-12", 12.0, 12.0, 1_000_000),   # 周一 第 2 个交易日（自然日=4）
        ("2026-01-13", 12.0, 12.0, 1_000_000),   # 周二 第 3 个交易日 → 触发
        ("2026-01-14", 12.0, 12.0, 1_000_000),   # 周三 成交
    ])
    trades = gen.generate_signals("2026-01-07", "2026-01-14")

    sells = [t for t in trades if t["direction"] == "卖出"]
    assert sells, "必须触发持仓超限卖出"
    assert all(s["date"] != "2026-01-12" for s in sells), "周一仅为第 2 个交易日，不得平仓"
    assert "个交易日超限" in sells[0]["reason"]


# ===== 5. 流动性约束 =====

def test_buy_quantity_capped_by_volume_participation(gen):
    """单笔买入不超过当日成交量的 VOLUME_PARTICIPATION（此处 5 万 × 1% = 500 股）"""
    gen._data_cache["000001"] = _bars([
        ("2026-01-05", 10.0, 11.0, 1_000_000),
        ("2026-01-06", 12.0, 12.0, 50_000),
        ("2026-01-07", 12.0, 12.0, 1_000_000),
    ])
    trades = gen.generate_signals("2026-01-05", "2026-01-07")

    buys = [t for t in trades if t["direction"] == "买入"]
    assert buys and buys[0]["quantity"] == 500


def test_buy_quantity_uses_budget_when_volume_ample(gen):
    """成交量充足时按预算下单（10 万 / 12 元 → 8300 股）"""
    gen._data_cache["000001"] = _bars([
        ("2026-01-05", 10.0, 11.0, 100_000_000),
        ("2026-01-06", 12.0, 12.0, 100_000_000),
        ("2026-01-07", 12.0, 12.0, 100_000_000),
    ])
    trades = gen.generate_signals("2026-01-05", "2026-01-07")

    buys = [t for t in trades if t["direction"] == "买入"]
    assert buys and buys[0]["quantity"] == 8300


# ===== 6. 滚动样本外验证（v2 0.1.2）=====

def test_split_periods_is_contiguous_and_covers_range():
    segs = mod._split_periods("2024-01-01", "2024-12-31", 3)
    assert len(segs) == 3
    assert segs[0][0] == "2024-01-01"
    assert segs[-1][1] == "2024-12-31"
    for a, b in segs:
        assert a <= b
    # 相邻区间不重叠：下一段的起点晚于上一段的终点
    for (_, prev_end), (next_start, _) in zip(segs, segs[1:]):
        assert next_start > prev_end


def test_split_periods_rejects_too_short_range():
    with pytest.raises(ValueError):
        mod._split_periods("2024-01-01", "2024-01-20", 3)


def test_walk_forward_selects_on_train_and_evaluates_out_of_sample(monkeypatch):
    """v2 回归：参数必须在训练段选出、在检验段评估。

    原 _sweep 在同一段全样本上挑最大 Sharpe，报出来的"最优配置"没有外推依据。
    """
    calls = []

    def fake_run_single(generator, engine, start, end, thr, atr, hd):
        calls.append((start, end, thr, atr, hd))
        # 训练段总是从区间起点开始；检验段不是
        on_train = start == "2024-01-01"
        sharpe = (thr - 50) / 10.0 if on_train else -1.0
        # v3（P2-1）：门槛已从 3 提到 20，样本数必须给足才可能被选中
        return {"threshold": thr, "atr_mult": atr, "holding_days": hd, "sharpe": sharpe,
                "total_trades": 25, "total_return": -3.0, "max_drawdown": -8.0,
                "win_rate": 40.0, "calmar": -0.4, "annual_return": -5.0}

    monkeypatch.setattr(mod, "_run_single", fake_run_single)
    folds = mod._walk_forward(object(), object(), "2024-01-01", "2025-06-30", n_folds=3)

    assert len(folds) == 2, "3 折应产生 2 个（训练→检验）折"
    for f in folds:
        # 训练段选出的最优 threshold 必是网格中最大的 70（sharpe 随 thr 单调递增）
        assert f["config"]["threshold"] == max(mod.SWEEP_THRESHOLDS)
        # 检验段必须使用训练段选出的同一组参数
        oos_calls = [c for c in calls if c[0] == f["test"][0]]
        assert oos_calls, "检验段必须被评估"
        assert oos_calls[0][2] == f["config"]["threshold"]
        assert oos_calls[0][3] == f["config"]["atr_mult"]
        assert oos_calls[0][4] == f["config"]["holding_days"]
        # 样本内为正、样本外转负 —— 这正是过拟合的可观测信号
        assert f["in_sample"]["sharpe"] > 0 > f["out_sample"]["sharpe"]


def test_sweep_min_trades_threshold_is_effective(monkeypatch):
    """P2-1：交易笔数不足的配置不得被选为"最优参数"。

    原来是 3 笔就算数——3 笔交易的 Sharpe 噪声比信号大，
    选出来的"最优配置"基本等于随机挑一组。
    """
    assert mod.SWEEP_MIN_TRADES >= 20, "门槛必须显著高于 3"

    def fake_run_single(generator, engine, start, end, thr, atr, hd):
        return {"threshold": thr, "atr_mult": atr, "holding_days": hd, "sharpe": 9.9,
                "total_trades": mod.SWEEP_MIN_TRADES - 1, "total_return": 5.0,
                "max_drawdown": -1.0, "win_rate": 50.0, "calmar": 5.0, "annual_return": 5.0}

    monkeypatch.setattr(mod, "_run_single", fake_run_single)
    assert mod._best_config_on(object(), object(), "2024-01-01", "2024-06-30") is None, \
        "样本数不达标时宁可不选，也不得报出一个假的最优配置"


def test_walk_forward_skips_fold_without_valid_config(monkeypatch):
    """训练段没有任何配置满足最少交易数 → 该折跳过，不得伪造结果"""
    def fake_run_single(generator, engine, start, end, thr, atr, hd):
        return {"threshold": thr, "atr_mult": atr, "holding_days": hd, "sharpe": 1.0,
                "total_trades": 0, "total_return": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "calmar": 0.0, "annual_return": 0.0}

    monkeypatch.setattr(mod, "_run_single", fake_run_single)
    assert mod._walk_forward(object(), object(), "2024-01-01", "2025-06-30", n_folds=3) == []


# ===== 7. 涨跌停按板块取（v3 / P1-3）=====

def _rising_bars(flat: int = 20, moves: int = 5, pct: float = 0.10) -> pd.DataFrame:
    """先横盘 flat 天，再连续 moves 天涨 pct —— 用来制造"涨停日"样本"""
    closes = [10.0] * flat
    for _ in range(moves):
        closes.append(round(closes[-1] * (1 + pct), 4))
    start = pd.Timestamp("2025-01-02")
    return _bars([((start + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                   closes[i - 1] if i else closes[0], c, 1_000_000)
                  for i, c in enumerate(closes)])


def test_sentiment_limit_pct_by_board(gen):
    """P1-3：涨跌停判定按板块取阈值。

    原实现硬编码 0.098——创业板（涨停 20%）上一个 +10% 的正常交易日会被算成"涨停"，
    情绪分被系统性抬高（该子因子占评分 10% 权重：40 分 vs 20 分 → 差 2.0 分）。
    """
    df = _rising_bars(flat=20, moves=5, pct=0.10)
    ref_date = df["date"].iloc[-1].strftime("%Y-%m-%d")
    gen._data_cache["000001"] = df                  # 主板：+10% 就是涨停
    gen._data_cache["300001"] = df.copy()           # 创业板：+10% 不是涨停

    main_board = gen._score_sentiment("000001", ref_date)
    gem = gen._score_sentiment("300001", ref_date)

    assert main_board > gem, "主板连续 +10% 应判为涨停，创业板不应"
    # `_score_sentiment` 返回的是"ATR 子分 + 涨跌停子分"（0~100），×0.10 的权重
    # 发生在 `score_stock` 里，所以这里差的是 40 vs 20 = 20 分
    assert main_board - gem == pytest.approx(20.0, abs=0.5), \
        "差异应恰好来自涨跌停子因子（40 vs 20）"


def test_limit_pct_helper_matches_matching_module(gen):
    """板块阈值必须来自 core.matching.limit_pct（唯一实现），不能再各写一份"""
    from core import matching
    for code in ("600519", "000001", "300750", "301536", "688981"):
        assert RuleSignalGenerator._limit_pct(code) == matching.limit_pct(code)


# ===== 8. 高价股必须能买 / 资金不足必须留痕（v3 / P1-2）=====

def test_high_price_stock_is_not_silently_dropped(gen):
    """P1-2：名义预算（10 万）买不起一手时，只要现金够就买一手。

    原实现 `qty < 100 → return 0`：一手 15 万的标的在**整个回测里从未被买入**，
    而且零日志——票池静默缩水。
    """
    gen._cash = mod.INITIAL_CASH
    row = {"open": 1500.0, "close": 1500.0, "volume": 10_000_000, "prev_close": 1500.0}
    qty = gen._size_buy({"code": "600519", "decision_date": "2025-01-02"}, 1500.0, row)
    assert qty == 100, "15 万一手、10 万名义预算 → 应允许买一手，而不是 0"

    # 资金确实不够时返回 0，但**必须记录原因**
    gen._cash = 50_000
    assert gen._size_buy({"code": "600519", "decision_date": "2025-01-02"}, 1500.0, row) == 0
    assert gen._skip_stats.get("资金不足") == 1, "买不起必须留痕，不能静默"


def test_buy_budget_is_capped_so_one_stock_cannot_take_the_whole_account(gen):
    """单笔预算有硬上限，且"上限内买不起一手"要留痕，不能吃掉整个账户"""
    gen._cash = mod.INITIAL_CASH
    cap = mod.INITIAL_CASH * mod.POSITION_BUDGET_CAP_PCT

    # 一手 20 万，落在上限（25 万）内 → 买一手
    row_ok = {"open": 2000.0, "close": 2000.0, "volume": 10_000_000, "prev_close": 2000.0}
    qty = gen._size_buy({"code": "600519", "decision_date": "2025-01-02"}, 2000.0, row_ok)
    assert qty == 100
    assert qty * 2000.0 <= cap + 1e-6

    # 一手 90 万，超出上限 → 不买（否则单只就能吃掉九成本金），但必须留痕
    row_too_expensive = {"open": 9000.0, "close": 9000.0,
                         "volume": 10_000_000, "prev_close": 9000.0}
    assert gen._size_buy({"code": "600519", "decision_date": "2025-01-02"},
                         9000.0, row_too_expensive) == 0
    assert gen._skip_stats.get("资金不足") == 1


def test_skip_stats_distinguish_cash_from_price_limit(gen):
    """P1-2：跳过买分两类——"资金不足"与"涨停买不到"不能混为一谈"""
    gen._cash = 1_000_000
    gen._skip("600519", "2025-01-02", "开盘涨停放弃", "开盘涨停(1650.00)不可买")
    gen._cash = 100
    gen._size_buy({"code": "600519", "decision_date": "2025-01-02"}, 1500.0,
                  {"open": 1500.0, "close": 1500.0, "volume": 10_000_000, "prev_close": 1500.0})
    assert gen._skip_stats == {"开盘涨停放弃": 1, "资金不足": 1}


# ===== 9. 中性化只改排序、不改达标判定（v3 / P1-1）=====

def test_neutralize_preserves_raw_score_and_never_lifts_sub_threshold(gen, monkeypatch):
    """P1-1 守护：**顺序**是先按原始分过滤、后中性化。

    报告曾判断"中性化会把 80 分压到 65、把 30 分抬到 65，于是普跌行情仍满仓"。
    实测的顺序恰好相反（`:score >= SCORE_THRESHOLD` 发生在中性化之前），
    所以"抬进候选"不会发生。这条用例把顺序钉死——以后谁把两步调换，它立刻变红。
    """
    # (a) 单元层：中性化前必须留下 score_raw，且它不等于中性化后的值
    cands = [{"code": "600519", "score": 80.0},
             {"code": "000858", "score": 70.0},
             {"code": "002714", "score": 60.0}]
    gen._neutralize_scores(cands)
    assert [c["score_raw"] for c in cands] == [80.0, 70.0, 60.0], "原始分必须原样保留"

    # (b) 行为层：全体低于阈值（普跌行情）→ 一个候选都不该有
    monkeypatch.setattr(mod, "WATCHLIST", ["000001", "000002", "000003"])
    gen.score_stock = lambda code, ds: 30.0        # 远低于 SCORE_THRESHOLD
    gen.calc_atr = lambda code, ds: 1.0
    rows = [("2026-01-05", 10.0, 10.0, 1_000_000), ("2026-01-06", 10.0, 10.0, 1_000_000)]
    for c in ("000001", "000002", "000003"):
        gen._data_cache[c] = _bars(rows)

    trades = gen.generate_signals("2026-01-05", "2026-01-06")
    assert [t for t in trades if t["direction"] == "买入"] == [], \
        "原始分不达标就进不了候选，中性化抬不动它"


def test_sub_threshold_stock_never_selected_even_with_single_industry(gen, monkeypatch):
    """单行业（中性化退化为大组）也不得把低于阈值的票选进来"""
    monkeypatch.setattr(mod, "WATCHLIST", ["600519", "000858", "002714", "000568"])
    gen.score_stock = lambda code, ds: mod.SCORE_THRESHOLD - 0.1
    gen.calc_atr = lambda code, ds: 1.0
    rows = [("2026-01-05", 10.0, 10.0, 1_000_000), ("2026-01-06", 10.0, 10.0, 1_000_000)]
    for c in ("600519", "000858", "002714", "000568"):
        gen._data_cache[c] = _bars(rows)

    trades = gen.generate_signals("2026-01-05", "2026-01-06")
    assert [t for t in trades if t["direction"] == "买入"] == []


# ===== 10. 股票池与零模型（v3 / P0-2、P2-3）=====

def test_pool_file_is_broad_and_industry_labelled():
    """池文件的数据契约：≥100 只、每只都有行业、行业数足够多"""
    import json
    from pathlib import Path
    path = Path(mod.POOL_FILE)
    assert path.exists(), "请先跑 scripts/build_backtest_pool.py 生成股票池"
    data = json.loads(path.read_text(encoding="utf-8"))
    pool, imap = data["pool"], data["industry_map"]

    assert len(pool) >= 100, f"池只有 {len(pool)} 只，不满足 100+ 的要求"
    assert len(set(pool)) == len(pool), "池内不得重复"
    assert all(c.isdigit() and len(c) == 6 for c in pool)
    assert all(c in imap for c in pool), "每只都必须有行业标签（中性化依赖它）"
    assert len(set(imap.values())) >= 20, "行业数太少会让中性化退化成一个大组"
    assert "幸存者偏差" in data.get("caveat", ""), "残留偏差必须写进池文件并公开展示"


def test_load_universe_falls_back_when_pool_missing(tmp_path):
    """池文件缺失时回退内置池，且**必须标明**它带前视选样偏差"""
    codes, imap, desc = mod.load_universe(tmp_path / "nope.json")
    assert codes == list(mod.LEGACY_WATCHLIST)
    assert "偏差" in desc or "不可外推" in desc, "回退池必须自带警告，不能悄悄用"


def test_load_universe_rejects_too_small_pool(tmp_path):
    """池文件存在但太小 → 同样回退（避免半成品文件静默生效）"""
    p = tmp_path / "small.json"
    p.write_text('{"pool": ["600519"], "industry_map": {"600519": "消费"}}', encoding="utf-8")
    codes, _, desc = mod.load_universe(p)
    assert len(codes) == len(mod.LEGACY_WATCHLIST)


def test_random_seed_switches_to_random_selection(gen, monkeypatch):
    """零模型：random_seed 非 None 时**不经过评分门槛**，随机抽票。

    这是"策略 vs 随机"对照的基础——随机分支若还带评分过滤，对照就没有意义。
    """
    monkeypatch.setattr(mod, "WATCHLIST", ["000001", "000002", "000003", "000004"])
    gen.score_stock = lambda code, ds: 0.0          # 评分全为 0，远低于阈值
    gen.calc_atr = lambda code, ds: 1.0
    rows = [("2026-01-05", 10.0, 10.0, 1_000_000), ("2026-01-06", 10.0, 10.0, 1_000_000)]
    for c in ("000001", "000002", "000003", "000004"):
        gen._data_cache[c] = _bars(rows)

    # 策略分支：分数为 0 → 不买
    gen.random_seed = None
    assert [t for t in gen.generate_signals("2026-01-05", "2026-01-06")
            if t["direction"] == "买入"] == []

    # 随机分支：不看分数 → 买满可买的名额（此处池内仅 4 只）
    gen.random_seed = 42
    buys = [t for t in gen.generate_signals("2026-01-05", "2026-01-06")
            if t["direction"] == "买入"]
    assert len(buys) == min(mod.MAX_STOCKS, 4), \
        f"随机分支应买满名额，实际 {len(buys)}"


def test_random_seed_is_deterministic(gen, monkeypatch):
    """同一 seed 必须复现同一组交易——否则对照实验不可复核"""
    monkeypatch.setattr(mod, "WATCHLIST", ["000001", "000002", "000003", "000004", "000005"])
    gen.score_stock = lambda code, ds: 0.0
    gen.calc_atr = lambda code, ds: 1.0
    rows = [("2026-01-05", 10.0, 10.0, 1_000_000), ("2026-01-06", 10.0, 10.0, 1_000_000)]
    for c in ("000001", "000002", "000003", "000004", "000005"):
        gen._data_cache[c] = _bars(rows)

    gen.random_seed = 7
    first = [(t["code"], t["direction"]) for t in gen.generate_signals("2026-01-05", "2026-01-06")]
    second = [(t["code"], t["direction"]) for t in gen.generate_signals("2026-01-05", "2026-01-06")]
    assert first == second


def test_random_baseline_reports_percentile(monkeypatch):
    """P2-3：对照必须给出"策略落在随机分布的哪个分位"，而不只是两个孤立的数字"""
    counter = {"i": 0}

    def fake_run_single(generator, engine, start, end, thr, atr, hd, random_seed=None):
        if random_seed is None:
            return {"threshold": thr, "atr_mult": atr, "holding_days": hd, "sharpe": 5.0,
                    "total_trades": 30, "total_return": 12.0, "max_drawdown": -5.0,
                    "win_rate": 55.0, "calmar": 2.4, "annual_return": 12.0,
                    "rejected_orders": 0, "avg_position_pct": 50.0}
        counter["i"] += 1
        return {"threshold": thr, "atr_mult": atr, "holding_days": hd,
                "sharpe": float(counter["i"]), "total_trades": 30, "total_return": float(counter["i"]),
                "max_drawdown": -5.0, "win_rate": 50.0, "calmar": 1.0, "annual_return": 10.0,
                "rejected_orders": 0, "avg_position_pct": 50.0}

    monkeypatch.setattr(mod, "_run_single", fake_run_single)
    summary = mod._random_baseline(object(), object(), "2024-01-01", "2025-06-30",
                                   60.0, 2.5, 20, trials=10)

    assert summary["trials"] == 10
    assert summary["random_sharpe_best"] == 10.0
    assert summary["strategy"]["sharpe"] == 5.0
    # 随机 Sharpe 为 1..10，策略 5.0 → 5/10 = 50 分位
    assert summary["strategy_sharpe_percentile"] == pytest.approx(50.0, abs=0.01)
    assert 0 <= summary["strategy_return_percentile"] <= 100


# ===== 11. 预加载缓存判据（v3 性能回归）=====

def _fresh_gen(monkeypatch, codes, hist_fn):
    """不打桩 preload 的生成器（`gen` 夹具把 preload 打桩了，测不了缓存判据）"""
    monkeypatch.setattr(mod, "WATCHLIST", list(codes))
    with patch.object(mod, "DataSourceManager"):
        g = RuleSignalGenerator()
    g.dm.get_stock_hist_data = hist_fn
    return g


def _hist_rows(first_bar: str = "2023-10-09"):
    return pd.DataFrame({"date": pd.to_datetime([first_bar, "2024-06-28"]),
                         "open": [10.0, 11.0], "high": [10.5, 11.5],
                         "low": [9.5, 10.5], "close": [10.0, 11.0],
                         "volume": [1_000_000.0, 1_000_000.0]})


def test_preload_does_not_refetch_when_same_range_requested_again(monkeypatch):
    """回归：长假后不得再误判"缓存不覆盖"而重拉全部标的。

    原判据是 `(数据首根K线 - 5天) <= 区间起点 - LOOKBACK_DAYS`。
    2024-01-01 减 90 天 = 2023-10-03，而 2023 年国庆休市到 10-09，
    数据里最早只有 10-09 → **每次都判"缓存不覆盖"**。
    后果不是报错，而是 `--sweep`（100 组参数）与 `--baseline`（N 次对照）
    退化成 100/N 次全量网络抓取——实测一次运行 3 小时以上仍未完成。
    """
    calls = {"n": 0}

    def fake_hist(code, start_date=None, end_date=None, adjust="qfq"):
        calls["n"] += 1
        # 首根 K 线 2023-10-09 —— 晚于 2023-10-03，正是触发旧 bug 的形态
        return _hist_rows("2023-10-09")

    gen = _fresh_gen(monkeypatch, ["000001"], fake_hist)

    gen.preload("2024-01-01", "2025-06-30")
    assert calls["n"] == 1
    gen.preload("2024-01-01", "2025-06-30")
    assert calls["n"] == 1, "同一区间第二次调用不得重拉网络"

    # 评分缓存也必须保留（这是让 100 组参数能复用同一份评分的前提）
    gen._score_cache[("000001", "2024-01-02")] = 80.0
    gen.preload("2024-01-01", "2025-06-30")
    assert ("000001", "2024-01-02") in gen._score_cache


def test_preload_refetches_when_window_widens_and_invalidates_scores(monkeypatch):
    """区间变大时必须重拉，且基于旧数据的评分缓存要作废"""
    calls = {"n": 0}

    def fake_hist(code, start_date=None, end_date=None, adjust="qfq"):
        calls["n"] += 1
        return _hist_rows("2022-10-09")

    gen = _fresh_gen(monkeypatch, ["000001"], fake_hist)
    gen.preload("2024-01-01", "2025-06-30")
    gen._score_cache[("000001", "2024-01-02")] = 80.0

    gen.preload("2023-01-01", "2025-06-30")          # 起点前移 → 重新拉
    assert calls["n"] == 2
    assert gen._score_cache == {}, "换了数据源就必须丢掉旧评分，否则是拿旧数据的结论"


def test_loaded_range_covers_tail_even_when_some_codes_failed(monkeypatch):
    """部分标的取数失败也要算"覆盖过"，否则每次调用都重拉失败标的（网络惩罚）"""
    calls = {"n": 0}

    def fake_hist(code, start_date=None, end_date=None, adjust="qfq"):
        calls["n"] += 1
        if code == "999999":
            return None                               # 永久取不到
        return _hist_rows("2023-10-09")

    gen = _fresh_gen(monkeypatch, ["000001", "999999"], fake_hist)
    gen.preload("2024-01-01", "2025-06-30")
    assert calls["n"] == 2 and "999999" not in gen._data_cache
    gen.preload("2024-01-01", "2025-06-30")
    assert calls["n"] == 2, "失败标的不得让整个区间每次重拉"


# ===== 12. 收盘价取数（v3 性能回归）=====

def test_close_on_date_uses_cached_series_and_carries_forward(gen):
    """收盘价取数：命中当日、早于首根 K 线返回 0、晚于末根顺延最近收盘价"""
    gen._data_cache["000001"] = _bars([
        ("2026-01-05", 10.0, 10.0, 1_000_000),
        ("2026-01-06", 10.0, 12.0, 1_000_000),
    ])
    assert gen._get_close_on_date("000001", "2026-01-06") == 12.0
    assert gen._get_close_on_date("000001", "2026-01-04") == 0.0    # 早于首根
    assert gen._get_close_on_date("000001", "2026-01-09") == 12.0   # 顺延最近收盘价


def test_missing_code_is_not_refetched_every_day(gen, monkeypatch):
    """缺失标的**不得每天联网重试**。

    v3：随机对照分支会对"池内每一只 × 每个交易日"取收盘价，
    而原实现在缓存未命中时每次调用都走一次网络请求
    （5 只缺失 × 391 个交易日 ≈ 2000 次/轮），实测一轮跑了 11 分钟仍未结束。
    现在只允许尝试一次，失败后记住并跳过。
    """
    monkeypatch.setattr(mod, "WATCHLIST", ["000001", "999999"])
    gen._data_cache["000001"] = _bars([("2026-01-05", 10.0, 10.0, 1_000_000)])
    calls = {"n": 0}

    def fake_hist(code, start_date=None, end_date=None, adjust="qfq"):
        calls["n"] += 1
        return None                                    # 永远取不到

    gen.dm.get_stock_hist_data = fake_hist
    for d in ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"):
        assert gen._get_close_on_date("999999", d) == 0.0
    assert calls["n"] == 1, "只允许尝试一次网络，失败后必须记住"
    assert "999999" in gen._no_data_codes


# ===== v4（2026-10-29，模块 A 接线）：扫描结果喂给 DSR/PBO =====

def _sweep_result(idx: int, dates: list[str], drift: float = 0.0, seed: int = 0) -> dict:
    """构造一条扫描结果（日收益 + 与之一致的 sharpe），供对齐与报告测试用"""
    import numpy as np
    rng = np.random.default_rng(seed + idx)
    rets = rng.normal(0.0, 0.01, size=len(dates)) + drift
    sharpe = float(np.mean(rets) / np.std(rets, ddof=1))
    return {"threshold": 55.0 + idx * 5, "atr_mult": 1.5, "holding_days": 20,
            "sharpe": sharpe, "total_trades": 30, "total_return": 0.0,
            "daily_returns": list(map(float, rets)), "nav_dates": list(dates)}


def test_multiple_testing_report_full_flow():
    """6 组扫描（1 真信号 + 5 噪声）→ 报告带 DSR/PBO，试验数=组合数，最优=真信号组"""
    import numpy as np
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=400)]
    results = [_sweep_result(i, dates, drift=(0.0015 if i == 3 else 0.0), seed=100)
               for i in range(6)]
    report = mod._multiple_testing_report(results)
    assert report is not None
    assert report["n_trials"] == 6
    assert report["dsr"]["ok"] and report["dsr"]["hurdle_source"] == "trials"
    assert report["dsr"]["n_trials"] == 6
    assert report["pbo"]["ok"] and report["pbo"]["n_configs"] == 6
    assert report["best_config"]["threshold"] == pytest.approx(55.0 + 3 * 5)


def test_align_matrix_uses_common_dates_only():
    """对齐口径 = 各配置日期的**交集**；交集 <20 行 ⇒ None（不零填充冒充）"""
    import numpy as np
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=300)]
    r1 = _sweep_result(0, dates, seed=1)
    r2 = _sweep_result(1, dates[15:-5], seed=2)              # 缺头去尾
    m = mod._align_returns_matrix([r1, r2])
    assert m is not None and m.shape == (280, 2)             # 交集 280 天
    r3 = _sweep_result(2, dates[:10], seed=3)                # 只剩 10 天交集
    assert mod._align_returns_matrix([r1, r2, r3]) is None


def test_multiple_testing_report_needs_two_series():
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=100)]
    one = [_sweep_result(0, dates)]
    assert mod._multiple_testing_report(one) is None
    stale = [{"threshold": 60, "atr_mult": 2.0, "holding_days": 20, "sharpe": 1.0}]  # 旧形状（无日收益）
    assert mod._multiple_testing_report(stale) is None


def test_run_single_empty_result_carries_return_keys(gen, monkeypatch):
    """零交易路径也要带新键（daily_returns/cost_drag）——消费方 .get() 的契约"""
    monkeypatch.setattr(mod.RuleSignalGenerator, "generate_signals",
                        lambda self, s, e: [])
    r = mod._run_single(gen, mod.BacktestEngine(), "2024-01-02", "2024-02-10", 60, 2.5, 20)
    assert r["daily_returns"] == [] and r["nav_dates"] == []
    assert r["cost_drag"] == 0.0


def test_sweep_end_to_end_wires_dsr_pbo(gen, tmp_path, monkeypatch):
    """端到端冒烟：注入 K 线跑真实 _sweep（2 组参数）⇒ DSR/PBO 有值且 JSON 落盘。

    这条守住"接线"本身——_run_single 的日收益提取、对齐矩阵、报告与落盘，
    任何一环断了它就红。参数网格缩到 2 组（阈值 60/61 都低于恒定评分 80），
    保证两组都有成交、PBO 才有 C=2 可算。
    """
    engine = mod.BacktestEngine(records_dir=str(tmp_path))
    dates = pd.bdate_range("2023-10-09", periods=260)
    gen._data_cache["000001"] = pd.DataFrame({
        "date": dates,
        "open": [10.0 + i * 0.01 for i in range(260)],
        "close": [10.0 + i * 0.01 for i in range(260)],
        "high": [10.05 + i * 0.01 for i in range(260)],
        "low": [9.95 + i * 0.01 for i in range(260)],
        "volume": [1_000_000.0] * 260,
    })
    monkeypatch.setattr(mod, "SWEEP_THRESHOLDS", [60.0, 61.0])
    monkeypatch.setattr(mod, "SWEEP_ATR_MULTS", [2.0])
    monkeypatch.setattr(mod, "SWEEP_HOLDING_DAYS", [5])
    monkeypatch.setattr(mod, "SWEEP_MIN_TRADES", 1)

    results = mod._sweep(gen, engine, "2024-04-01", "2024-09-30")

    assert len(results) == 2
    for r in results:
        assert r["daily_returns"], "有成交的配置必须产出日收益（接线断点检测）"
        assert r["nav_dates"]
    saved = list(tmp_path.glob("sweep_insample_*.json"))
    assert saved, "扫描结果必须落盘"
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["multiple_testing"] is not None
    mt = payload["multiple_testing"]
    assert mt["n_trials"] == 2
    assert mt["dsr"]["ok"] and mt["dsr"]["hurdle_source"] == "trials"
    assert mt["pbo"]["ok"] and mt["pbo"]["n_configs"] == 2
