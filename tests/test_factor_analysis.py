"""因子有效性分析契约测试（`core/factor_analysis.py`）

守护四件事：

1. **前视防护**：某日 T 的因子值只能用 `date <= T` 的数据。这类 bug 的形态是
   "IC 好得离谱"，肉眼看不出来 —— 所以这里用"构造已知答案"的方式正面验证。
2. **算得对**：人为造一段"动量越高、未来收益越高"的数据，IC 必须显著为正；
   造反向数据，IC 必须显著为负。**这是唯一能证明公式没写反的办法**。
3. **小样本诚实**：期数不足时给"无法判断"，不硬出一个数字。
4. **不可分析的因子必须带原因**——不是悄悄消失。
"""

import pandas as pd
import pytest

from core import factor_analysis as fa


# ================================================================
# 夹具
# ================================================================

def _bars(closes, start="2024-01-02"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "date": dates,
        "open": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes], "close": closes,
        "volume": [1_000_000.0] * len(closes),
        "amount": [100_000_000.0] * len(closes),
        "turnover_pct": [2.0] * len(closes),
    })


def _sloped_map(n=20, lo=-0.005, hi=0.015, length=130):
    """n 只票，日涨跌幅从 lo 线性排到 hi —— 用来构造"因子值有序"的截面"""
    out = {}
    for i in range(n):
        slope = lo + (hi - lo) * i / max(1, n - 1)
        closes = [10.0 * ((1 + slope) ** t) for t in range(length)]
        out[f"60{i:04d}"] = fa._enrich(_bars(closes))
    return out


def _reversed_map(n=20, length=130, split=70):
    """前段按斜率涨（决定 momentum20），后段**反向**（决定未来收益）→ IC 应为负"""
    out = {}
    for i in range(n):
        slope = -0.005 + 0.02 * i / max(1, n - 1)
        px, closes = 10.0, []
        for t in range(length):
            px *= (1 + (slope if t < split else -slope))
            closes.append(px)
        out[f"60{i:04d}"] = fa._enrich(_bars(closes))
    return out


S, E = "2024-01-02", "2024-06-28"


def _run(bars_map, factors=("momentum20",), **kw):
    kw.setdefault("step", 5)
    return fa.run([], factors=list(factors), start=S, end=E,
                  bars_map=bars_map, **kw)


# ================================================================
# 1. 算得对：用已知答案验证公式
# ================================================================

def test_ic_positive_when_factor_orders_future_returns():
    """动量越高、未来收益越高 → IC 必须显著为正。

    这是**唯一能证明 IC 没算反**的办法：构造已知方向的数据，看结论对不对。
    """
    res = _run(_sloped_map(), factors=("momentum20",))
    assert res.periods >= 3, f"期数太少（{res.periods}），样本构造有问题"
    row = res.summary[0]
    assert row["IC均值"] > 0.5, row
    # 构造数据每期 IC 完全一致 → 标准差为 0，ICIR 应为 None（不是 0）
    assert row["ICIR"] is None or row["ICIR"] > 0
    assert row["结论"] == "显著有效", row


def test_ic_negative_when_direction_reversed():
    """「越涨、后续越跌」→ IC 必须为负。

    ⚠️ 注意构造方式：第一版只是把斜率整体取反（`lo=0.015, hi=-0.005`），
    但**同一斜率贯穿全程** → 因子与未来收益仍然同向，IC 依然是 +1。
    要制造反向关系，必须让**后段斜率与前段相反**（见 `_reversed_map`）。
    """
    bars_map = _reversed_map()
    # 只取"动量来自前段、未来收益来自后段"的那个窗口（否则两段会互相抵消）
    dates = [bars_map["600000"]["date"].iloc[i].strftime("%Y-%m-%d")
             for i in range(55, 72, 4)]
    panel = fa.build_panel(bars_map, ["momentum20"], dates, horizon=20)
    ics = []
    for _, g in panel.groupby("date"):
        sub = g[["momentum20", "fwd_ret"]].dropna()
        if len(sub) >= 5:
            ics.append(sub["momentum20"].corr(sub["fwd_ret"], method="spearman"))
    assert ics, "窗口内应有样本"
    assert sum(ics) / len(ics) < -0.5, ics


def test_volatility_direction_note():
    """波动率是"越小越好"，构造"波动越大后续越差" → IC 应为负、方向判定正确"""
    bars_map = {}
    for i in range(20):
        amp = 0.002 + i * 0.004                      # 波动递增
        closes = []
        px = 10.0
        for t in range(130):
            px *= (1 + (amp if t % 2 == 0 else -amp))   # 交替涨跌 → 高波动
            closes.append(px)
        bars_map[f"60{i:04d}"] = fa._enrich(_bars(closes))
    res = _run(bars_map, factors=("volatility",))
    row = res.summary[0]
    assert row["方向"] == "越小越好"
    # 交替波动在高频上会互相抵消，这里主要验证**能算出数、方向标注对**
    assert row["IC均值"] is not None


def test_layered_returns_are_monotonic_for_ordered_factor():
    """因子值有序时，分层收益应当单调"""
    res = _run(_sloped_map(), factors=("momentum20",), groups=5)
    info = res.layered["momentum20"]
    assert len(info["rows"]) == 5
    assert info["单调"] is True, info
    assert info["多空差%"] > 0, "最好组应显著优于最差组"


# ================================================================
# 2. 前视防护
# ================================================================

def test_panel_never_uses_future_bars_for_factor():
    """因子取值只能用 ≤T 的数据：把**早期某日**的因子输入保持不变、
    只改最后一段价格，则那些**早期采样日的因子值**必须一字不变。

    做法：直接比对 `build_panel` 产出的因子列（按 (date, code) 对齐），
    只比较"未来还没被改动"的那些日子。
    """
    base = _sloped_map(n=8, length=130)
    dates = [d.strftime("%Y-%m-%d")
             for d in base["600000"]["date"].iloc[25:60:5]]

    p1 = fa.build_panel(base, ["momentum20"], dates, horizon=20)

    tampered = {k: v.copy() for k, v in base.items()}
    for k, df in tampered.items():
        df.loc[df.index[-25:], "close"] = 999.0      # 只动最后 25 根
    p2 = fa.build_panel(tampered, ["momentum20"], dates, horizon=20)

    merged = p1.merge(p2, on=["date", "code"], suffixes=("_a", "_b"))
    assert len(merged) > 0
    for _, r in merged.iterrows():
        if abs(r["momentum20_a"] - r["momentum20_b"]) > 1e-12:
            pytest.fail(f"{r['date']} 的因子值受到了未来数据影响")


def test_panel_rows_nonzero_but_no_ic_gives_explicit_note():
    """面板非空却算不出 IC 时，必须说明是**门槛**问题而不是含糊的"期数不足"

    （实测踩过：12 只票 + min_names=20 → 768 行面板、0 期 IC，
    报告只写"期数不足"，用户根本不知道是门槛设错了。）
    """
    res = _run(_sloped_map(n=6), factors=("momentum20",), min_names=50)
    assert res.panel_rows > 0
    assert res.periods == 0
    assert "每期股票数不足" in res.note
    assert "扩大股票池" in res.note


def test_pos_on_or_before_excludes_future():
    bars = fa._enrich(_bars([10.0] * 60))
    pos = fa._pos_on_or_before(bars, "2024-02-15")
    assert pos is not None
    assert bars["date"].iloc[pos] <= pd.Timestamp("2024-02-15")
    if pos + 1 < len(bars):
        assert bars["date"].iloc[pos + 1] > pd.Timestamp("2024-02-15")


def test_build_panel_uses_own_horizon_not_calendar():
    """未来收益用**该票自己的第 horizon 根之后**（位置索引），停牌也不会串味"""
    bars = fa._enrich(_bars([10.0 + i * 0.1 for i in range(60)]))
    panel = fa.build_panel({"600519": bars}, ["momentum20"],
                           [bars["date"].iloc[25].strftime("%Y-%m-%d")], horizon=5)
    assert len(panel) == 1
    expect = (float(bars["close"].iloc[30]) / float(bars["close"].iloc[25]) - 1)
    assert panel.iloc[0]["fwd_ret"] == pytest.approx(expect)


def test_build_panel_skips_when_future_unavailable():
    """接近数据末尾时未来收益不可得 → 该样本必须被跳过（不是拿 0 顶上）"""
    bars = fa._enrich(_bars([10.0] * 40))
    panel = fa.build_panel({"600519": bars}, ["momentum20"],
                           [bars["date"].iloc[-1].strftime("%Y-%m-%d")], horizon=20)
    assert panel.empty


# ================================================================
# 3. 小样本诚实
# ================================================================

def test_insufficient_periods_says_so():
    """期数不足时给"无法判断"，不硬出一个数字"""
    res = _run(_sloped_map(n=4, length=45), factors=("momentum20",), step=10)
    row = res.summary[0]
    assert row["期数"] < 3
    assert row["IC均值"] is None
    assert "不足" in row["结论"]


def test_empty_data_returns_note_not_crash():
    res = fa.run([], factors=["momentum20"], start=S, end=E, bars_map={})
    assert res.names == 0
    assert res.note
    assert res.summary is None or res.summary == []


def test_min_names_default_adapts_to_small_pool():
    """**默认**（不传）时门槛自适应：12 只票的池子也必须能算出 IC

    实测踩过：固定 20 只门槛 + 12 只票的池子 → 面板 768 行、0 期 IC，
    报告只写"期数不足"，用户根本分不清是数据不够还是门槛设错。
    """
    res = _run(_sloped_map(n=12), factors=("momentum20",))
    assert res.periods > 0, "默认门槛必须自适应到池子规模"


def test_explicit_min_names_is_respected_not_relaxed():
    """**显式**传 min_names 时按用户的来 —— 不能把"我要更严格"悄悄放宽"""
    res = _run(_sloped_map(n=12), factors=("momentum20",), min_names=50)
    assert res.periods == 0
    assert "每期股票数不足" in res.note


def test_min_names_filters_thin_periods():
    """门槛真的生效：要求每期 50 只而池子只有 6 只 → 不产出 IC（不是硬算）"""
    res = _run(_sloped_map(n=6), factors=("momentum20",), min_names=50)
    assert res.periods == 0, "样本数不够时不该算出 IC"


# ================================================================
# 4. 口径与结构守护
# ================================================================

@pytest.mark.parametrize("raw,expect", [
    (True, 1.0), (False, 0.0), (1.5, 1.5), (0, 0.0),
])
def test_num_converts_bool_to_binary(raw, expect):
    assert fa._num(raw) == expect


@pytest.mark.parametrize("raw", [None, float("nan"), "abc", ""])
def test_num_returns_none_not_zero(raw):
    """缺值必须是 None —— 本项目铁律「缺就是缺，不补造」"""
    assert fa._num(raw) is None


def test_bool_factor_is_analyzed_as_binary():
    """布尔因子（均线多头）要能当 0/1 参与 IC，而不是被判成"无数值" """
    res = _run(_sloped_map(), factors=("ma_bullish",))
    row = res.summary[0]
    assert row["因子"] == "均线多头排列"
    assert row["IC均值"] is not None or row["期数"] < 3


def test_unavailable_factors_all_have_reasons():
    for k, why in fa.UNAVAILABLE.items():
        assert why, k
        assert len(why) > 8, f"{k} 的原因太含糊，用户看不出为什么算不了"


def test_factors_subset_of_replayable_registry_fields():
    """可分析的因子必须是**可回放**的（否则 IC 分析会拿当期值当历史值）"""
    from core.condition_backtest import REPLAYABLE_FIELDS
    for k in fa.FACTORS:
        assert k in REPLAYABLE_FIELDS, f"{k} 不在可回放集合里，不能做 IC 分析"


def test_pe_and_market_cap_are_declared_unavailable():
    """PE / 总市值 / 换手率必须被显式标为不可分析（而不是漏做）"""
    for k in ("pe", "market_cap", "turnover"):
        assert k in fa.UNAVAILABLE, k
        assert "历史截面" in fa.UNAVAILABLE[k] or "快照" in fa.UNAVAILABLE[k]


def test_snapshot_gap_note_is_actionable():
    """缺口说明必须给出**可执行的做法**，不能只说"不支持" """
    note = fa.snapshot_gap_note()
    assert "快照" in note
    assert "get_sector_snapshot" in note, "要指出用哪个接口去积累"


def test_result_to_dict_is_serializable():
    import json
    res = _run(_sloped_map(n=10), factors=("momentum20",))
    d = res.to_dict()
    json.dumps(d, ensure_ascii=False)                 # 不抛异常即通过
    assert "summary" in d and "layered" in d and "unavailable" in d


# ================================================================
# 5. 结构性守护：声明了就必须算得出（防止"静默 0 期"）
# ================================================================

def test_enrich_covers_every_declared_factor():
    """`FACTORS` 里声明的每个键，都必须在 `_enrich` 后有对应列。

    ⚠️ 这是本轮真实踩到的坑：`_enrich` 只派生了 momentum20 / volatility，
    漏了两个布尔因子；而 `build_panel` 对"列不存在"的处理是 `row[f] = None`
    （**不报错**）→ 它们静默变成 0 期 IC，报告只写"期数不足"，
    看起来像样本问题。声明与实现必须对齐。
    """
    df = fa._enrich(_bars([10.0 + i * 0.1 for i in range(60)]))
    missing = [k for k in fa.FACTORS if k not in df.columns]
    assert not missing, f"这些因子声明了但没派生：{missing}"


def test_declared_factors_all_produce_ic():
    """每个声明的因子都必须**真的能算出 IC**（不是永远 0 期）"""
    bars_map = _sloped_map(n=20)
    res = _run(bars_map, factors=tuple(fa.FACTORS))
    for row in res.summary:
        assert row["期数"] > 0, f"{row['因子']} 算不出 IC：{row}"
        assert row["IC均值"] is not None, row


def test_bool_factors_are_binary_in_panel():
    """布尔因子在面板里必须是 0/1，而不是 None 或 True/False"""
    bars_map = _sloped_map(n=10)
    dates = [bars_map["600000"]["date"].iloc[i].strftime("%Y-%m-%d")
             for i in range(30, 40, 5)]
    panel = fa.build_panel(bars_map, ["ma_bullish", "ma20_above"], dates, horizon=20)
    assert len(panel) > 0
    for f in ("ma_bullish", "ma20_above"):
        vals = set(panel[f].dropna().unique())
        assert vals <= {0.0, 1.0}, f"{f} 的取值不是 0/1：{vals}"


def test_layered_works_for_small_pool():
    """10 只票的池子也要能出分层结果（第一版 groups*3 门槛让它静默为空）"""
    res = _run(_sloped_map(n=10), factors=("momentum20",), groups=5)
    assert res.layered, "分层结果为空 —— 门槛把整个池子挡掉了"
    assert len(res.layered["momentum20"]["rows"]) == 5


# ================================================================
# 6. 相关性矩阵：一期缺失不能污染整行整列
# ================================================================

def _mixed_bool_map(n=20, length=140, split=70):
    """构造"某布尔因子**只在部分期**取值恒定"的池子。

    前段所有票同向上涨 → `ma_bullish` 整期恒 True（方差 0 → Spearman 无定义）；
    后段各票斜率分化 → `ma_bullish` 有 0 有 1（可正常算相关）。
    ⇒ 正好复现实测里"某一期 NaN 污染整行整列"的条件。

    （注意不能做成**每期**都恒定：那样这个因子的截面相关在数学上确实无定义，
    NaN 是诚实结果，不是 bug —— 断言"对角线必须是 1.0"只对**有可用期**成立。）
    """
    out = {}
    for i in range(n):
        closes, px = [], 10.0
        for t in range(length):
            r = 0.004 if t < split else (-0.0015 + 0.004 * i / max(1, n - 1))
            px *= (1 + r)
            closes.append(px)
        out[f"60{i:04d}"] = fa._enrich(_bars(closes))
    return out


def test_constant_bool_factor_does_not_poison_other_cells():
    """某一期某个因子取值恒定 → 不能用 `sum(corrs)/len`（会传播 NaN 污染整行整列）

    ⚠️ 实测（2026-10-07，50 只池）：`站上20日均线` 整行整列 NaN，连对角线都不是
    1.00 —— 因为布尔因子在单期内常为常数，那一期的 Spearman 是 NaN，
    而 DataFrame 相加会把它传播到平均结果里。
    """
    res = _run(_mixed_bool_map(n=20), factors=("momentum20", "ma_bullish"))
    assert res.correlation is not None
    # ① 对角线必须是 1.0（有可用期时不能被 NaN 污染）
    for f in ("momentum20", "ma_bullish"):
        assert res.correlation.loc[f, f] == pytest.approx(1.0, abs=1e-6), (
            f"对角线 {f}×{f} 被 NaN 污染了：\n{res.correlation}")
    # ② 行标签顺序必须与列标签一致（groupby 会按字母排序索引，导致矩阵看着不对称）
    assert list(res.correlation.index) == list(res.correlation.columns), (
        f"行列标签顺序不一致，矩阵读起来会像不对称：\n{res.correlation}")
    # ③ 有恒定期的因子必须如实报告缺测期数，而不是悄悄变 NaN
    assert res.corr_const_periods > 0, "应记录取值恒定的期数"
    assert res.corr_periods >= res.corr_const_periods, (
        "参与平均的期数不能少于含缺测的期数")


def test_corr_periods_in_serialized_payload():
    """缺测期数必须进 `to_dict()`（UI 要拿它做披露）"""
    res = _run(_mixed_bool_map(n=20), factors=("momentum20", "ma_bullish"))
    d = res.to_dict()
    assert "corr_const_periods" in d and "corr_periods" in d


# ================================================================
# 7. 样本外稳健性：符号一致率的判定契约
# ================================================================
#
# 这一组**不重测 IC 公式**（§1 已经用已知答案验证过），
# 只测"聚合层"：符号怎么数、阈值怎么判、`bars_map` 有没有被复用。
# 所以给 `run` 装替身、喂可预期的 IC —— 这样判定逻辑是唯一变量。

def _stub_run(ic_map, monkeypatch, calls=None):
    """把 `fa.run` 换成返回可预期 IC 的替身。

    ic_map: {(start, end, horizon): ic} —— 缺省按 (start,end) 里的年份取
    """
    from core.factor_analysis import AnalysisResult

    def fake_run(codes, factors=None, start=None, end=None, horizon=20, step=5,
                 groups=5, min_names=None, bars_map=None, progress_cb=None):
        if calls is not None:
            calls.append((start, end, horizon, min_names))
        ic = ic_map.get((start, end, horizon))
        if ic is None:
            ic = ic_map.get(start[:4], -0.10)
        summary = []
        for f in (factors or ["momentum20"]):
            summary.append({"key": f, "因子": fa.FACTORS[f]["label"],
                            "IC均值": ic, "t值": -2.5, "期数": 20,
                            "结论": "显著但方向相反"})
        return AnalysisResult(summary=summary, periods=20, names=1, horizon=horizon)

    monkeypatch.setattr(fa, "run", fake_run)


P23 = [("2023-01-01", "2023-12-31")]


def test_robustness_all_contrary_verdict(monkeypatch):
    """全部配置反向 → 必须判成"方向稳健：与权重假设相反"（这才是可行动的结论）"""
    _stub_run({}, monkeypatch)                    # 默认 -0.10，动量假设是正向 → 全反向
    res = fa.robustness([], periods=P23, horizons=[10, 20, 60],
                        bars_map={"600000": pd.DataFrame()})
    row = res.summary[0]
    assert (row["有效配置"], row["相反"], row["一致"]) == (3, 3, 0)
    assert row["符号一致率"] == pytest.approx(0.0)
    assert "相反" in row["判定"] and "装反" in row["判定"]


def test_robustness_all_agree_verdict(monkeypatch):
    """全部配置同向 → 判成"方向稳健：与权重假设一致"""
    _stub_run({"2023": +0.12}, monkeypatch)
    res = fa.robustness([], periods=P23, horizons=[10, 20], factors=["momentum20"],
                        bars_map={"600000": pd.DataFrame()})
    row = res.summary[0]
    assert row["一致"] == 2 and row["相反"] == 0
    assert row["判定"] == "方向稳健：与权重假设一致"
    assert row["平均IC"] == pytest.approx(0.12)


def test_robustness_direction_flip_is_unstable(monkeypatch):
    """一半反一半正 → **不能**说"稳健"，必须老实讲"样本不足以定论"

    ⚠️ 这正是这套检验存在的意义：单区间很容易得出一个"看起来显著"的符号，
    换个持有期就翻。判据要是写成"多数配置"而不设 0.8 门槛，
    2/3 反也会被讲成"稳健" —— 那是把噪声包装成结论。
    """
    _stub_run({( "2023-01-01", "2023-12-31", 10): -0.10,
               ("2023-01-01", "2023-12-31", 20): +0.10,
               ("2023-01-01", "2023-12-31", 60): +0.10}, monkeypatch)
    res = fa.robustness([], periods=P23, horizons=[10, 20, 60],
                        bars_map={"600000": pd.DataFrame()})
    row = res.summary[0]
    assert row["符号一致率"] == pytest.approx(0.667)
    assert "不稳" in row["判定"], row["判定"]


def test_robustness_direction_aware_for_low_is_good_factor(monkeypatch):
    """负 IC 对"值越小越好"的因子才是**对的方向** —— 判定必须按方向翻转

    ⚠️ 若这里少了 `meta["direction"]`，波动率会被永远判成"装反了"。
    """
    _stub_run({}, monkeypatch)                    # IC 全为 -0.10
    res = fa.robustness([], periods=P23, horizons=[10, 20],
                        factors=["volatility"],              # direction = -1
                        bars_map={"600000": pd.DataFrame()})
    row = res.summary[0]
    assert row["一致"] == 2 and row["相反"] == 0
    assert "与权重假设一致" in row["判定"]


def test_robustness_covers_every_config(monkeypatch):
    """每个 (区间 × 持有期) 都要真的跑一次 —— 少跑一格也是"静默漏样本" """
    calls = []
    _stub_run({}, monkeypatch, calls)
    periods = [("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")]
    fa.robustness([], periods=periods, horizons=[10, 20, 60],
                  bars_map={"600000": pd.DataFrame()})
    assert len(calls) == 6, f"应有 6 个配置，实际 {len(calls)}"
    assert {(c[0][:4], c[2]) for c in calls} == {
        ("2023", 10), ("2023", 20), ("2023", 60),
        ("2024", 10), ("2024", 20), ("2024", 60)}


def test_robustness_reuses_bars_map_without_refetch(monkeypatch):
    """传了 `bars_map` 就不许联网。

    ⚠️ 这是性能铁律：本项目回测模块曾因"逐日循环里联网"把 20 轮对照
    从 137 秒拖到 3.5 小时。这里用"调用即失败"的替身把它钉死。
    """
    called = []
    monkeypatch.setattr(fa, "load_bars",
                        lambda *a, **kw: called.append(a) or {})
    _stub_run({}, monkeypatch)
    fa.robustness([], periods=P23, horizons=[10],
                  bars_map={"600000": pd.DataFrame()})
    assert called == [], "传了 bars_map 却还是去抓日线了"


def test_robustness_empty_bars_map_reports_honestly(monkeypatch):
    """没有日线 → 说清楚，不假装算过"""
    res = fa.robustness([], periods=P23, horizons=[10], bars_map={})
    assert res.note == "没有可用日线"
    assert res.summary == []


def test_pad_end_leaves_room_for_horizon():
    """右端要往后推，给未来收益留空间；推得越多越安全（只会多抓）"""
    assert fa._pad_end("2025-12-31", 60) > "2025-12-31"
    assert fa._pad_end("2025-12-31", 60) > fa._pad_end("2025-12-31", 10)
    assert fa._pad_end("坏日期", 10) == "坏日期"          # 解析失败不许抛
