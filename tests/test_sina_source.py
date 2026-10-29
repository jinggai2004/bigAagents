"""新浪行业快照 / 个股日线 数据通路契约测试

守护三件事（每一条都有历史教训）：

1. **单位换算**。本项目已因"同列名不同单位"栽过三次（融资余额 元/亿、同花顺
   `'20.03%'` 被金额解析器吃成 0、成交量当成交额）。所以每个字段都断言**量级**，
   而不是只断言"有值"。
2. **覆盖域诚实**。新浪行业只覆盖全市场 54%，且板块严重偏斜（科创板 5%）。
   覆盖率与 caveat 必须随数据返回，**不得宣称"全市场"**。
3. **失败语义**。取不到 → 返回 None、不抛异常；行业成功率不足 → 判上游不稳；
   拿不到全市场清单 → 不做过滤但**如实标注**（不是悄悄放过退市代码）。

所有用例都 mock 叶子（`akshare` 的三个函数），不打真网络。
"""

import os

import pandas as pd
import pytest

from data import sina_source as ss
from core.sector import classify_by_board


# ================================================================
# 夹具
# ================================================================

@pytest.fixture(autouse=True)
def _clean_cache(tmp_path, monkeypatch):
    """清缓存 + 把落盘清单隔离到 tmp（否则用例会通过真实文件互相串味）"""
    monkeypatch.setattr(ss, "_CODE_LIST_FILE", str(tmp_path / "stock_code_list.json"))
    ss.clear_cache()
    yield
    ss.clear_cache()


# 实测样本（2026-10-07 真实返回，用于锁定单位）
ROCK = {
    "symbol": "sh600176", "code": "600176", "name": "中国巨石",
    "trade": 39.41, "pricechange": -0.28, "changepercent": -0.705,
    "buy": 39.41, "sell": 39.42, "settlement": 39.69, "open": 39.91,
    "high": 40.19, "low": 39.21, "volume": 94954728, "amount": 3761312051,
    "ticktime": "15:34:59", "per": 47.985, "pb": 4.758,
    "mktcap": 15776361.845048, "nmc": 15652024.477348, "turnoverratio": 2.39085,
}


def _sectors(labels=("new_blhy", "new_yh"), names=("玻璃行业", "银行")):
    return pd.DataFrame({"label": list(labels), "板块": list(names)})


def _patch(monkeypatch, detail_rows=None, sector_df=None, code_list=None,
           detail_raises_for=()):
    """mock 叶子接口。detail_raises_for 里的 label 会抛异常（模拟单行业失败）。

    ⚠️ 同时**恒定**把网络清单源掐掉（`_fetch_code_list_network` → None），
    只有显式给 `code_list` 时才返回该清单。
    理由：清单源现在会走真实 HTTP（新浪 8 并发分页），不掐掉就等于
    "单测在跑网络"——本项目已因"测试耦合真实环境"红过多次。
    """
    import akshare as ak

    rows = detail_rows if detail_rows is not None else [ROCK]
    sdf = sector_df if sector_df is not None else _sectors()

    monkeypatch.setattr(ak, "stock_sector_spot", lambda **kw: sdf)

    served = {"done": False}

    def _detail(sector=None, **kw):
        if sector in detail_raises_for:
            raise RuntimeError(f"boom {sector}")
        # 只有**第一个成功的行业**返回成分，其余返回空——
        # 这样"单行业失败"的用例不会被"后面某个行业又给了数据"掩盖
        if served["done"]:
            return pd.DataFrame(columns=list(ROCK))
        served["done"] = True
        return pd.DataFrame(rows)

    monkeypatch.setattr(ak, "stock_sector_detail", _detail)
    if code_list is not None:
        monkeypatch.setattr(ss, "_fetch_code_list_network",
                            lambda timeout=0: ({str(c) for c in code_list}, "test"))
    else:
        monkeypatch.setattr(ss, "_fetch_code_list_network",
                            lambda timeout=0: None)


# ================================================================
# 1. 单位换算（量级断言）—— 本模块最要紧的守护
# ================================================================

def test_market_cap_converted_from_wan_to_yi(monkeypatch):
    """`mktcap` 单位是**万元**，输出必须是**亿元**。

    中国巨石：15,776,361.845 万元 = 1,577.6 亿元。
    若忘记除以 1e4，会得到 1577 万亿的市值——本项目"融资余额元/亿"错法同源。
    """
    _patch(monkeypatch)
    df = ss.fetch_sector_snapshot()
    row = df[df["code"] == "600176"].iloc[0]
    assert row["market_cap"] == pytest.approx(1577.636, rel=1e-4)
    assert 100 < row["market_cap"] < 100_000, "量级必须落在「亿元」区间"
    assert row["float_cap"] == pytest.approx(1565.202, rel=1e-4)


def test_amount_converted_from_yuan_to_yi(monkeypatch):
    """`amount` 单位是**元**，输出必须是**亿元**（3,761,312,051 元 = 37.6 亿）。"""
    _patch(monkeypatch)
    df = ss.fetch_sector_snapshot()
    row = df[df["code"] == "600176"].iloc[0]
    assert row["amount"] == pytest.approx(37.613, rel=1e-3)
    assert 1 < row["amount"] < 1000, "量级必须落在「亿元」区间"


def test_turnover_and_change_pct_are_percent_not_fraction(monkeypatch):
    """`turnoverratio` 与 `changepercent` 已是**百分数**，原样输出（不做 ×100）。"""
    _patch(monkeypatch)
    row = ss.fetch_sector_snapshot().iloc[0]
    assert row["turnover"] == pytest.approx(2.39085)
    assert row["change_pct"] == pytest.approx(-0.705)
    assert 0 <= row["turnover"] <= 100, "换手率若变成 239% 说明多乘了 100"


def test_pe_pb_are_multiples(monkeypatch):
    _patch(monkeypatch)
    row = ss.fetch_sector_snapshot().iloc[0]
    assert row["pe"] == pytest.approx(47.985)
    assert row["pb"] == pytest.approx(4.758)


def test_daily_turnover_is_a_fraction_but_output_is_percent(monkeypatch):
    """⚠️ 新浪**日线**的 `turnover` 与行业快照的 `turnoverratio` **单位不同**：
    日线是小数（0.004187）、行业是百分数（2.39）。本模块统一输出百分数 `turnover_pct`。
    """
    import akshare as ak
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]), "open": [9.29], "high": [9.37],
        "low": [9.22], "close": [9.22], "volume": [16762392.0],
        "amount": [165215109.0], "outstanding_share": [4003136728.0],
        "turnover": [0.004187314383432172],
    })
    monkeypatch.setattr(ak, "stock_zh_a_daily", lambda **kw: raw.copy())

    df = ss.fetch_stock_daily("600176", "20240102", "20240115")
    assert df.iloc[0]["turnover_pct"] == pytest.approx(0.4187, rel=1e-3)
    assert 0 < df.iloc[0]["turnover_pct"] < 5, "小数 ×100 后必须落在合理换手率区间"


def test_daily_turnover_self_consistent_with_shares(monkeypatch):
    """换手率可自算：volume / outstanding_share × 100 ≈ turnover_pct（口径交叉验证）"""
    import akshare as ak
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]), "open": [9.29], "high": [9.37],
        "low": [9.22], "close": [9.22], "volume": [16762392.0],
        "amount": [165215109.0], "outstanding_share": [4003136728.0],
        "turnover": [0.004187314383432172],
    })
    monkeypatch.setattr(ak, "stock_zh_a_daily", lambda **kw: raw.copy())
    df = ss.fetch_stock_daily("600176")
    self_calc = df.iloc[0]["volume"] / df.iloc[0]["outstanding_share"] * 100
    assert self_calc == pytest.approx(df.iloc[0]["turnover_pct"], rel=0.05)


# ================================================================
# 2. 代码归一与板块分类
# ================================================================

@pytest.mark.parametrize("raw,expect", [
    ("600176", "600176"), ("sh600176", "600176"), ("SZ000001", "000001"),
    ("bj830799", "830799"), (1, "000001"), ("000001", "000001"),
    (600519, "600519"),
])
def test_clean_code_normalizes(raw, expect):
    assert ss._clean_code(raw) == expect


@pytest.mark.parametrize("raw", [None, "", "  ", float("nan"), "abcdef", "12345678"])
def test_clean_code_rejects_garbage(raw):
    assert ss._clean_code(raw) == ""


def test_board_of_matches_core_sector():
    """板块分类**只能有一份实现**。本模块为了不反向依赖 core 写了本地版，
    必须与 `core.sector.classify_by_board` 逐例一致——否则会出现
    "UI 按创业板筛、覆盖率按科创板统计"这种静默错位。
    """
    codes = ["600176", "600519", "000001", "002594", "300750", "301536",
             "688981", "689009", "830799", "430047", "920002", "900001", "123456"]
    for code in codes:
        assert ss._board_of(code) == classify_by_board(code), code


def test_prefix_rules():
    assert ss._prefixed("600176") == "sh600176"
    assert ss._prefixed("688981") == "sh688981"
    assert ss._prefixed("000001") == "sz000001"
    assert ss._prefixed("300750") == "sz300750"
    assert ss._prefixed("830799") == "bj830799"


# ================================================================
# 3. 覆盖域诚实——不得宣称"全市场"
# ================================================================

def test_coverage_and_by_board_are_reported(monkeypatch):
    """覆盖率与分板块覆盖必须随数据返回（测试用 1 只 / 3 只的构造样本）"""
    _patch(monkeypatch, code_list=["600176", "000001", "300750"])
    df = ss.fetch_sector_snapshot()
    assert df.attrs["got"] == 1
    assert df.attrs["total_market"] == 3
    assert df.attrs["coverage"] == pytest.approx(1 / 3, abs=1e-3)
    bb = df.attrs["by_board"]
    assert bb["主板沪市"]["got"] == 1 and bb["主板沪市"]["total"] == 1
    assert bb["创业板"]["got"] == 0 and bb["创业板"]["total"] == 1


def test_caveat_explicitly_denies_full_market_claim(monkeypatch):
    """caveat 必须明说"不可外推至全市场"——这是给用户看的那句话"""
    _patch(monkeypatch, code_list=["600176", "000001", "300750"])
    df = ss.fetch_sector_snapshot()
    caveat = df.attrs["caveat"]
    assert "候选域" in caveat
    assert "不可外推" in caveat, "必须显式否认「全市场」的说法"


def test_caveat_still_honest_when_market_list_unavailable(monkeypatch):
    """全市场清单拿不到 → 不能编造覆盖率，要明说"无法给出覆盖率" """
    _patch(monkeypatch)          # detail/sector 正常；清单源返回 None
    df = ss.fetch_sector_snapshot()
    assert df.attrs["coverage"] is None
    assert df.attrs["filtered"] is False
    assert "无法给出覆盖率" in df.attrs["caveat"]
    assert "不可外推" in df.attrs["caveat"]


def test_snapshot_note_readable():
    """`snapshot_note` 是 UI 直接用的文案，不能返回 None/空"""
    class _Fake:
        attrs = {"caveat": "候选域 100 只（覆盖 50%）"}
    assert "候选域" in ss.snapshot_note(_Fake())
    assert ss.snapshot_note(None) == "候选域数据不可用"


# ================================================================
# 4. 失败语义：返回 None、不抛异常、闸门只针对自身健康度
# ================================================================

def test_returns_none_when_sector_list_fails(monkeypatch):
    import akshare as ak

    def _boom(**kw):
        raise RuntimeError("行业列表服务不可用")

    monkeypatch.setattr(ak, "stock_sector_spot", _boom)
    assert ss.fetch_sector_snapshot() is None


def test_returns_none_when_sector_list_empty(monkeypatch):
    _patch(monkeypatch, sector_df=pd.DataFrame({"label": [], "板块": []}))
    assert ss.fetch_sector_snapshot() is None


def test_partial_sector_failure_is_tolerated(monkeypatch):
    """单个行业失败不影响其余（49 个行业里偶尔挂一个不该毁掉整次选股）

    ⚠️ 比例要按真实量级构造：真实 49 个行业。若只用 2 个行业做样本，
    "挂 1 个"就成了 50%——那是**测试构造失真**，不是闸门有问题。
    """
    labels = tuple(f"L{i}" for i in range(10))
    _patch(monkeypatch, sector_df=_sectors(labels=labels, names=labels),
           detail_raises_for=("L0",))            # 10 个挂 1 个 → 90% ≥ 闸门
    df = ss.fetch_sector_snapshot()
    assert df is not None and len(df) == 1
    assert any("L0" in f for f in df.attrs["failed_sectors"])
    assert df.attrs["sector_success_rate"] == pytest.approx(0.9)


def test_empty_rows_returns_none(monkeypatch):
    """所有行业都返回空 → 判无数据（不是"零只可选"这么轻的说法）"""
    _patch(monkeypatch, detail_rows=[])
    assert ss.fetch_sector_snapshot() is None


def test_returns_none_when_upstream_too_unhealthy(monkeypatch):
    """行业成功率低于下限 → 判上游不稳并放弃（宁可没有，也不要残缺样本）。

    ⚠️ 这个闸门针对的是**本条通路自身的健康度**，**不是**"对全市场的覆盖率"——
    后者只有 54% 是已知特性，若拿它做闸门会让本通路永远返回 None。
    """
    labels = tuple(f"L{i}" for i in range(10))
    _patch(monkeypatch, sector_df=_sectors(labels=labels, names=labels),
           detail_raises_for=labels[3:])          # 10 个行业挂 7 个 → 30%
    assert ss.fetch_sector_snapshot() is None


def test_delisted_codes_are_filtered_out(monkeypatch):
    """新浪分类里混着 364 只已不在 A 股清单的代码，必须按清单过滤"""
    ghosts = dict(ROCK, code="999999", name="幽灵股")
    _patch(monkeypatch, detail_rows=[ROCK, ghosts], code_list=["600176"])
    df = ss.fetch_sector_snapshot()
    assert list(df["code"]) == ["600176"]
    assert df.attrs["filtered"] is True


def test_no_filter_when_market_list_unavailable(monkeypatch):
    """清单拿不到时不做过滤，但**必须标注 filtered=False**（不是悄悄放过）"""
    _patch(monkeypatch, detail_rows=[ROCK])   # 清单源为 None
    df = ss.fetch_sector_snapshot()
    assert df.attrs["filtered"] is False
    assert len(df) == 1


# ================================================================
# 5. 日线：失败语义与日期格式
# ================================================================

def test_daily_returns_none_on_failure(monkeypatch):
    import akshare as ak
    monkeypatch.setattr(ak, "stock_zh_a_daily",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("500")))
    assert ss.fetch_stock_daily("600176") is None


def test_daily_returns_none_on_empty(monkeypatch):
    import akshare as ak
    monkeypatch.setattr(ak, "stock_zh_a_daily", lambda **kw: pd.DataFrame())
    assert ss.fetch_stock_daily("600176") is None


def test_daily_rejects_bad_code(monkeypatch):
    assert ss.fetch_stock_daily("abc") is None
    assert ss.fetch_stock_daily("") is None


def test_daily_accepts_both_date_formats(monkeypatch):
    """`20240102` 与 `2024-01-02` 都要能用（下游调用方写法不一）"""
    import akshare as ak
    seen = {}

    def _fake(symbol=None, start_date=None, end_date=None, adjust=None):
        seen["start"], seen["end"] = start_date, end_date
        return pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02"]), "open": [9.0], "high": [9.1],
            "low": [8.9], "close": [9.05], "volume": [100.0], "amount": [900.0],
            "outstanding_share": [10000.0], "turnover": [0.01]})

    monkeypatch.setattr(ak, "stock_zh_a_daily", _fake)
    ss.fetch_stock_daily("600176", "2024-01-02", "2024-01-15")
    assert seen["start"] == "20240102" and seen["end"] == "20240115"


def test_daily_caches_same_window(monkeypatch):
    """同一区间第二次取不得再打网络（本项目踩过"缓存判据失效→重拉全量"的坑）"""
    import akshare as ak
    calls = {"n": 0}

    def _fake(**kw):
        calls["n"] += 1
        return pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02"]), "open": [9.0], "high": [9.1],
            "low": [8.9], "close": [9.05], "volume": [100.0], "amount": [900.0],
            "outstanding_share": [10000.0], "turnover": [0.01]})

    monkeypatch.setattr(ak, "stock_zh_a_daily", _fake)
    ss.fetch_stock_daily("600176", "20240102", "20240115")
    ss.fetch_stock_daily("600176", "20240102", "20240115")
    assert calls["n"] == 1


# ================================================================
# 6. 结构性守护：列名与 NaN 语义
# ================================================================

def test_snapshot_columns_stable(monkeypatch):
    """输出列固定（下游按名取用，列漂移要显式失败而不是静默给 None）"""
    _patch(monkeypatch)
    df = ss.fetch_sector_snapshot()
    assert list(df.columns) == ss.SNAPSHOT_COLUMNS


def test_missing_numeric_stays_none_not_zero(monkeypatch):
    """缺值必须是 None（**不能是 0**）——本项目铁律「缺就是缺，不补造」"""
    _patch(monkeypatch, detail_rows=[dict(ROCK, per=None, mktcap=None, amount=None)])
    row = ss.fetch_sector_snapshot().iloc[0]
    assert row["pe"] is None
    assert row["market_cap"] is None
    assert row["amount"] is None


def test_dedup_keeps_first_across_sectors(monkeypatch):
    """同一只票出现在多个行业时按首次保留（实测有 5 条跨行业重复）"""
    _patch(monkeypatch, detail_rows=[ROCK, dict(ROCK)])
    df = ss.fetch_sector_snapshot()
    assert len(df) == 1


# ================================================================
# 7. 全市场清单：限时 + 落盘 + 多源回退（实测 akshare 那个接口 257.7s 才失败）
# ================================================================

def test_code_list_timeout_does_not_block():
    """`_run_with_timeout` 超时必须放弃而不是让选股卡住"""
    import time as _t

    def _slow():
        _t.sleep(5)
        return "x"

    t0 = _t.time()
    ok, out = ss._run_with_timeout(_slow, 0.3)
    elapsed = _t.time() - t0
    assert ok is False and out is None
    assert elapsed < 2, f"必须在限时内返回，实际 {elapsed:.1f}s"


def test_code_list_saved_to_disk_and_reused(monkeypatch):
    """成功后落盘；随后即使网络挂掉也能从盘上读到（这才叫"可用的缓存"）"""
    calls = {"n": 0}

    def _ok(timeout=0):
        calls["n"] += 1
        return {"600176", "000001"}, "test"

    monkeypatch.setattr(ss, "_fetch_code_list_network", _ok)
    assert ss.market_codes() == {"600176", "000001"}
    assert calls["n"] == 1
    assert os.path.exists(ss._CODE_LIST_FILE), "必须落盘"
    assert ss.last_code_list_meta()["source"] == "test", "来源必须可追溯"

    ss.clear_cache(disk=False)           # 只清内存，保留磁盘
    monkeypatch.setattr(ss, "_fetch_code_list_network", lambda timeout=0: None)
    assert ss.market_codes() == {"600176", "000001"}, "网络挂了也应从盘上读到"


def test_code_list_failure_returns_none(monkeypatch):
    monkeypatch.setattr(ss, "_fetch_code_list_network", lambda timeout=0: None)
    assert ss.market_codes() is None


def test_stale_disk_list_used_when_all_sources_fail(monkeypatch):
    """⚠️ 2026-10-26 真实事故：清单接口连挂 10 天 ⇒ 原实现"过期即返回 None" ⇒
    完全不过滤 ⇒ 186 只退市票（价格 0）每天被写进快照。

    退市是**单向不可逆**事件 ⇒ 过期清单仍然正确，用它过滤远好于不过滤；
    但必须标出来（degraded），否则没人知道自己在用过期清单。
    """
    import time as _t
    import json as _json

    old = _t.time() - 19 * 86400
    with open(ss._CODE_LIST_FILE, "w", encoding="utf-8") as f:
        _json.dump({"fetched_at": old, "count": 2,
                    "codes": ["600176", "000001"]}, f)

    monkeypatch.setattr(ss, "_fetch_code_list_network", lambda timeout=0: None)
    got = ss.market_codes()
    assert got == {"600176", "000001"}, "过期清单也要用（不能退化成不过滤）"
    meta = ss.last_code_list_meta()
    assert meta["source"] == "disk_stale"
    assert meta["age_days"] == pytest.approx(19.0, abs=0.2)


def test_stale_code_list_shows_up_as_degraded(monkeypatch):
    """用过期清单过滤时，快照 attrs 必须显式降级（不能看起来像正常）"""
    import time as _t
    import json as _json

    with open(ss._CODE_LIST_FILE, "w", encoding="utf-8") as f:
        _json.dump({"fetched_at": _t.time() - 19 * 86400, "count": 1,
                    "codes": ["600176"]}, f)
    monkeypatch.setattr(ss, "_fetch_code_list_network", lambda timeout=0: None)
    _patch(monkeypatch, detail_rows=[ROCK], code_list=None)
    # _patch 会把网络源设成 None，但磁盘上的旧清单仍生效
    df = ss.fetch_sector_snapshot()
    assert any(str(x).startswith("code_list_stale") for x in df.attrs["degraded"]), \
        f"降级原因必须列出，实际 {df.attrs['degraded']}"
    assert df.attrs["code_list_source"] == "disk_stale"


def test_sina_list_is_tried_before_akshare(monkeypatch):
    """新浪清单（1.5s）必须排在 akshare（18 页 ≈ 24s 且当前稳定报错）之前"""
    calls = []

    def _sina(timeout=0):
        calls.append("sina")
        return {"600176", "000001"}

    def _ak(timeout=0):
        calls.append("akshare")
        return {"999999"}

    monkeypatch.setattr(ss, "_fetch_code_list_sina", _sina)
    monkeypatch.setattr(ss, "_fetch_code_list_akshare", _ak)
    codes, src = ss._fetch_code_list_network(10)
    assert src == "sina" and "akshare" not in calls, "新浪成功就不该再碰 akshare"
    assert codes == {"600176", "000001"}


def test_network_chain_falls_through_to_akshare(monkeypatch):
    """新浪挂掉时才轮到 akshare（回退链不能反向吞掉权威源）"""
    monkeypatch.setattr(ss, "_fetch_code_list_sina", lambda timeout=0: None)
    monkeypatch.setattr(ss, "_fetch_code_list_akshare", lambda timeout=0: {"600176"})
    codes, src = ss._fetch_code_list_network(10)
    assert (src, codes) == ("akshare", {"600176"})


def test_short_sina_list_rejected(monkeypatch):
    """⚠️ 拉到半份清单比拉不到更危险：会把 90% 的股票当退市剔掉。

    条数下限就是这条守卫 —— 必须能在"只返回一页"时拒绝采用。
    """
    class _Resp:
        def __init__(self, text):
            self.text = text

    # count 接口给的是正常量级，但分页全返回同一页 ⇒ 只拿 100 只
    def _get(url, params=None, headers=None, timeout=None, **kw):
        if "StockCount" in url:
            return _Resp('"5576"')
        return _Resp('[{"code":"600176","name":"中国巨石"}]')

    import requests
    monkeypatch.setattr(requests, "get", _get)
    assert ss._fetch_code_list_sina(timeout=5) is None


def test_filtering_is_optional_not_blocking(monkeypatch):
    """清单拿不到时**仍然**要能出快照（过滤只是省时优化，不是前置条件）"""
    _patch(monkeypatch)          # 清单源为 None
    df = ss.fetch_sector_snapshot()
    assert df is not None and len(df) == 1
    assert df.attrs["filtered"] is False


# ================================================================
# 8. 无报价标的必须丢弃（2026-10-26：快照里常驻 186 只 price=0 的退市票）
# ================================================================

def test_no_quote_rows_are_dropped_and_counted(monkeypatch):
    """`trade=0` 不是"0 元"，是**没有报价**（退市/长期停牌）⇒ 丢掉且报数。

    实测样本：`000022 深赤湾A`、`000003 PT金田A` 每天返回 `trade=0/changepercent=0`，
    原实现原样入库，下游按低价排序时会把这些"0 元票"排在最前。
    """
    dead = dict(ROCK, code="000022", name="深赤湾A", trade=0,
                changepercent=0, amount=0, turnoverratio=0)
    _patch(monkeypatch, detail_rows=[ROCK, dead], code_list=["600176", "000022"])
    df = ss.fetch_sector_snapshot()
    assert list(df["code"]) == ["600176"], "无报价标的不能进快照"
    assert df.attrs["dropped_no_quote"] == 1, "丢了几只必须可查（不能静默缩水）"
    assert df.attrs["got"] == len(df)


def test_all_no_quote_returns_none(monkeypatch):
    """全部无报价 ⇒ 判上游异常返回 None，不能返回空快照冒充成功"""
    dead = dict(ROCK, trade=0, changepercent=0)
    _patch(monkeypatch, detail_rows=[dead], code_list=["600176"])
    assert ss.fetch_sector_snapshot() is None
