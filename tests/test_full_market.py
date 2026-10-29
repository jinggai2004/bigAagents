# -*- coding: utf-8 -*-
"""全量域（`data.full_market`）：口径、单位、结构性剔除、采残拒写。

存在理由（一句话）：选股域被"新浪 49 行业只覆盖 3,038 只"锁在 47% 覆盖率，
创业板 31% / 科创板 5%，`300750`、`688111` 长期选不到 —— 这不是条件太严，
是**域缺了一半市场**。本模块把域扩到 99.9%，这些测试守住三件容易复发的事：

1. **口径不漂移**：`_is_a_share` / `_board_of` 必须与 `core` 逐字一致
   （历史上"沪主板 vs 主板沪市"就是这么漂的）；
2. **单位只换一次**：新浪 `mktcap` 是**万元**，落库必须是**元**；
3. **采残不落库**：半份截面（< 4,000 只）比采不到更危险 —— 它会把
   一半市场当"退市"。
"""
from unittest.mock import patch

import pytest

from data import full_market as fm


# ================================================================
# 1. 口径一致性
# ================================================================

def test_is_a_share_matches_core():
    """与 `core.scheduler.is_a_share_code` 逐字一致（两处口径漂移会让"剔除"失真）"""
    from core.scheduler import is_a_share_code

    samples = ["600519", "601398", "688111", "000001", "002594", "300750", "301111",
               "830799", "430047", "920000", "900901", "200011", "159915", "12345",
               "", None, "abcdef"]
    bad = [(c, fm._is_a_share(c), is_a_share_code(c))
           for c in samples if fm._is_a_share(c) != is_a_share_code(c)]
    assert not bad, f"判定不一致（code, full_market, core）: {bad}"


def test_board_of_matches_core_including_bse_920():
    """板块口径一致，且 **920xxx 必须判北交所**（此前落进"其他"垃圾桶）"""
    from core.sector import classify_by_board
    from data.sina_source import _board_of

    samples = ["600519", "688111", "000001", "300750", "301111", "830799",
               "430047", "920000", "920285", "900901"]
    bad = [(c, _board_of(c), classify_by_board(c))
           for c in samples if _board_of(c) != classify_by_board(c)]
    assert not bad, f"板块判定不一致: {bad}"
    assert _board_of("920000") == "北交所"


# ================================================================
# 2. 采集与单位
# ================================================================

def _raw(code, name, trade="10.00", pct="1.5", mktcap="100000.0", nmc="50000.0"):
    """新浪 `Market_Center.getHQNodeData` 的原始条目形状（全是字符串）"""
    return {"symbol": ("sh" if code.startswith("6") else "sz") + code,
            "code": code, "name": name, "trade": trade, "changepercent": pct,
            "volume": "1000000", "amount": "12345678",
            "per": "18.4", "pb": "1.78", "mktcap": mktcap, "nmc": nmc,
            "turnoverratio": "0.62"}


def test_units_and_fields(monkeypatch):
    """mktcap/nmc **万元 → 元**；amount 本来就是元（不再乘）"""
    monkeypatch.setattr(fm, "_fetch_page",
                        lambda page, timeout, retries=2: [_raw("600519", "贵州茅台")])
    q = fm.fetch_universe_quotes(timeout=1)
    assert len(q) == 1
    r = q[0]
    assert r["price"] == 10.0
    assert r["pe"] == 18.4 and r["pb"] == 1.78
    assert r["market_cap"] == pytest.approx(100000.0 * 1e4)
    assert r["float_cap"] == pytest.approx(50000.0 * 1e4)
    assert r["amount"] == 12345678.0          # 元，未换算
    assert r["turnover"] == 0.62              # 百分数


def test_non_a_share_and_no_quote_dropped(monkeypatch):
    """B 股/无报价（停牌 `trade=0`）都不能进截面"""
    rows = [_raw("600519", "贵州茅台"), _raw("900901", "云赛B股"),
            _raw("200011", "深物业B"), _raw("000001", "平安银行", trade="0.000")]
    monkeypatch.setattr(fm, "_fetch_page",
                        lambda page, timeout, retries=2: rows if page == 1 else [])
    q = fm.fetch_universe_quotes(timeout=1)
    assert [r["code"] for r in q] == ["600519"], "B 股与无报价必须被剔除"


def test_first_page_empty_raises(monkeypatch):
    monkeypatch.setattr(fm, "_fetch_page", lambda page, timeout, retries=2: [])
    with pytest.raises(RuntimeError):
        fm.fetch_universe_quotes(timeout=1)


# ================================================================
# 3. 采残拒写
# ================================================================

def test_half_universe_rejected(monkeypatch):
    """条数 < UNIVERSE_MIN ⇒ 抛错**不落库**（半份截面会把一半市场当退市）"""
    half = [{"code": f"{600000 + i}", "name": "x", "price": 10.0,
             "change_pct": 1.0, "volume": 1.0, "amount": 1.0, "pe": 10.0,
             "pb": 1.0, "market_cap": 1e9, "float_cap": 1e9, "turnover": 1.0,
             "board": "主板沪市"} for i in range(100)]
    with pytest.raises(RuntimeError, match="采残"):
        fm.build_rows(quotes=half)


def test_build_rows_merges_stock_basic(tmp_path, monkeypatch):
    """行业**分列**：industry_ts=stock_basic(tushare 口径)、industry=新浪历史映射（2026-10-29）"""
    monkeypatch.setattr(fm, "_basic_cache_paths",
                        lambda: (str(tmp_path / "a.json"), str(tmp_path / "b.json")))
    monkeypatch.setattr(fm, "stock_basic_map",
                        lambda **_kw: ({"600519": {"name": "贵州茅台",
                                                   "industry": "白酒",
                                                   "market": "主板"}}, "cache"))
    # 新浪口径兜底映射（行业口径隔离：两列不可混）
    from db import store
    monkeypatch.setattr(store, "latest_industry_map",
                        lambda **_kw: {"600519": "酿酒行业"})
    quotes = [{"code": "600519", "name": "旧名", "price": 10.0, "change_pct": 1.0,
               "volume": 1.0, "amount": 1.0, "pe": 10.0, "pb": 1.0,
               "market_cap": 1e9, "float_cap": 1e9, "turnover": 1.0,
               "board": "主板沪市"}]
    monkeypatch.setattr(fm, "UNIVERSE_MIN", 0)
    rows, attrs = fm.build_rows(quotes=quotes)
    assert rows[0]["industry"] == "酿酒行业"        # 新浪口径（新闻荐股匹配用）
    assert rows[0]["industry_ts"] == "白酒"          # tushare 口径（选股配额用）
    assert rows[0]["stock_name"] == "贵州茅台"        # stock_basic 优先
    assert attrs["industry_missing"] == 0
    assert attrs["degraded"] == []


def test_stock_basic_cache_roundtrip(tmp_path, monkeypatch):
    """缓存能写能读（stock_basic 是 1 次/分钟 ⇒ 不能靠实时拉）"""
    p = str(tmp_path / "basic.json")
    monkeypatch.setattr(fm, "_basic_cache_paths", lambda: (p, p + ".bak"))
    fm._save_basic_cache({"600519": {"name": "贵州茅台", "industry": "白酒",
                                     "market": "主板"}})
    got = fm._load_basic_cache()
    assert got and got["600519"]["industry"] == "白酒"
    # 过期 ⇒ 不返回（由调用方决定是否用过期缓存兜底）
    assert fm._load_basic_cache(now=9e9) is None


# ================================================================
# 5. 运维入口（`python -m data.full_market`）
# ================================================================
#
# 存在的理由：`stock_basic` 是 1 次/小时硬限频，撞上 40203 时全量域只能带着
# `degraded` 落库。补行业必须有一个**不改代码**的一行入口，否则运维只能等，
# 而"等"在历史上就等于"永远没补"。这两个用例锁住 exit code 语义。

def test_cli_refresh_basic_exit_codes(monkeypatch, caplog):
    """刷新成功 exit=0；限频/失败 exit=1（脚本化补数必须能判成败）

    ⚠️ 输出走 `_logger` 不走 stdout（`data/` 层禁止 `__main__` 块外 print），
       所以用 `caplog` 断言，不是 `capsys`。
    """
    import logging

    caplog.set_level(logging.INFO, logger=fm.__name__)

    monkeypatch.setattr(fm, "stock_basic_map",
                        lambda **_kw: ({"600519": {"name": "贵州茅台", "industry": "白酒",
                                                   "market": "主板"}}, "tushare"))
    assert fm.main(["--refresh-basic"]) == 0
    assert "source=tushare rows=1" in caplog.text

    caplog.clear()
    monkeypatch.setattr(fm, "stock_basic_map",
                        lambda **_kw: ({}, "empty"))
    assert fm.main(["--refresh-basic"]) == 1
    assert "失败/限频" in caplog.text


def test_cli_snapshot_normalizes_date(monkeypatch):
    """`--snapshot 20261026` 必须归一成 `2026-10-26` 再落库。

    ⚠️ 跨源日期口径唯一：`YYYYMMDD` 混进 `YYYY-MM-DD` 会让按日过滤**双向**
       失效（查不到、也去重不掉），这是本项目踩过最贵的一类坑。
    """
    seen = {}

    def _fake_save(d, **_kw):
        seen["date"] = d
        return {"ok": True, "rows": 5572, "covered": 5572, "universe": 5576,
                "industry_missing": 2943, "degraded": []}

    monkeypatch.setattr(fm, "save_full_snapshot", _fake_save)
    assert fm.main(["--snapshot", "20261026"]) == 0
    assert seen["date"] == "2026-10-26"

    assert fm.main(["--snapshot", "2026-10-26"]) == 0
    assert seen["date"] == "2026-10-26"


def test_cli_no_args_prints_help():
    """空参数不静默成功（否则 cron 里写错参数会被当成"跑过了"）"""
    assert fm.main([]) == 2


# ================================================================
# 6. 行业兜底口径（`store.latest_industry_map`）
# ================================================================

def test_industry_map_is_history_union(tmp_path, monkeypatch):
    """行业兜底必须是**历史并集**，不是"最新那一天"。

    真实事故：10-26 全量域采集时 `stock_basic` 撞限频，行业只能靠历史截面补；
    旧实现只读最新一天，于是把 10-25 及更早有行业、10-26 没采到的那些股票
    一起丢掉（当天 2,948 只缺行业）。行业名一年变不了几次，旧值 ≫ "未知"。
    """
    from db import store

    from tests.conftest import use_template_db

    use_template_db(tmp_path, monkeypatch, "industry_union.db")
    store.save_market_snapshot("2026-10-23", [
        {"stock_code": "600519", "stock_name": "贵州茅台", "industry": "白酒"},
        {"stock_code": "300750", "stock_name": "宁德时代", "industry": "电池"},
    ], source="t")
    store.save_market_snapshot("2026-10-26", [
        {"stock_code": "600519", "stock_name": "贵州茅台", "industry": "白酒"},
        {"stock_code": "300750", "stock_name": "宁德时代", "industry": ""},
    ], source="t")

    m = store.latest_industry_map()
    assert m.get("600519") == "白酒"
    assert m.get("300750") == "电池", "10-26 没采到行业 ⇒ 必须回退到 10-23 的已知值"

    # 显式传日期时保持单日语义（研究场景要精确复现某天的截面）
    one_day = store.latest_industry_map("2026-10-26")
    assert "300750" not in one_day, "显式传日期必须是单日口径，不能悄悄改成并集"
