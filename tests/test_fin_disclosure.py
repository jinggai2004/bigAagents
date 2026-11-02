# -*- coding: utf-8 -*-
"""财报公告日（PIT）与前视防护测试。

锁的是什么
──────────
`tool_financials` 原先直接返回"最新 N 期报表"，**不知道哪一期什么时候公布**。
对"看现在"没影响，但对"站在历史某天看"（复盘/历史重放/将来的基本面因子）
会把**当时还没公布的财报**当已知 ⇒ 前视偏差。这类错误**不会报错**，
只会让结论系统性偏乐观，所以必须有结构性守护：

1. `split_by_as_of` 三态：可用 / 公告日晚于 as_of（真没公布）/ 库里没记录（缺≠未公布）；
2. 缺记录走 **fail-closed**（剔除而不是放行），且两类在返回值里**分开报告**；
3. `tool_financials(as_of=...)` 真实裁期，`as_of` 不传时**行为与改造前一致**；
4. A 股边界必须是**面板代码集**：东财业绩报表会混入 5,000+ 只新三板
   （83/87 段，`is_a_share_code` 会当北交所放行），只按代码段判会污染数据；
5. 渲染层 `_render_financials` 必须真的把 `公告日` 送到模型面前——
   它的兜底循环只认 dict 与标量，**list 会被静默丢掉**（已踩）。

全部离线：临时库 + 合成财报，不打网络。
"""
from __future__ import annotations

import pytest

from data import fin_disclosure as fd
from db import store as db_store


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    return str(use_template_db(tmp_path, monkeypatch, "test_fin_disc.db"))


# ================================================================
# 1. 报告期推导
# ================================================================

def test_quarter_ends_no_future_quarter():
    """不能把"还没过的季末"算进来——否则缺数据会被误判成"抓漏了"。

    边界：**当天正好季末时，该季末本身不算"已完成"**（当天报表还没出），
    取上一个季末。这条规则在 6-30 / 9-30 这类日子决定"要不要去抓这一期"。
    """
    assert fd.quarter_ends(4, as_of="2026-10-30") == [
        "20260930", "20260630", "20260331", "20251231"]
    # 年初：当前季度之前是去年 Q4
    assert fd.quarter_ends(2, as_of="2026-01-15") == ["20251231", "20250930"]
    # 季末当天：该季末还没走完 ⇒ 从上一期算
    assert fd.quarter_ends(1, as_of="2026-06-30") == ["20260331"]
    # 季末次日：该季末已完成 ⇒ 可用
    assert fd.quarter_ends(1, as_of="2026-07-01") == ["20260630"]


def test_norm_period_and_date():
    assert [fd._norm_period(x) for x in ("20260630", "2026-06-30", "202606", "abc")] == \
        ["20260630", "20260630", "", ""]
    assert [fd.norm_date(x) for x in ("2026-04-08", "20260408", "", None)] == \
        ["2026-04-08", "2026-04-08", "", ""]


# ================================================================
# 2. PIT 过滤三态
# ================================================================

def test_split_by_as_of_three_states():
    kept, dropped = fd.split_by_as_of(
        ["20260630", "20260331", "20251231"],
        {"20260630": "2026-08-20", "20260331": "2026-10-30"},
        "2026-09-01")
    assert kept == ["20260630"], "只有公告日 ≤ as_of 的期可用"
    reasons = {d["period"]: d["reason"] for d in dropped}
    assert reasons == {"20260331": "not_announced", "20251231": "unverified"}, \
        f"缺记录与真未公布必须分开报，实际 {reasons}"


def test_split_by_as_of_same_day_counts_as_known():
    """公告日 == as_of 当天必须**可用**（当天公布当天就知道）。"""
    kept, dropped = fd.split_by_as_of(["20260630"], {"20260630": "2026-08-20"},
                                      "2026-08-20")
    assert kept == ["20260630"] and dropped == []


def test_split_by_as_of_unparsed_does_not_filter():
    """as_of 判不出来时**不过滤**（宁可如实返回，也不要静默返回空）。"""
    kept, _ = fd.split_by_as_of(["20260630"], {"20260630": "2026-08-20"}, "不是日期")
    assert kept == ["20260630"]


# ================================================================
# 3. A 股边界（面板代码集）
# ================================================================

def test_new_third_board_is_excluded_by_universe():
    """83/87 段在新三板 ⇒ 必须被面板边界排除（只按代码段判会放行）。"""
    universe = {"600519", "300750", "920001"}
    assert fd._is_a_share("600519", universe) is True
    assert fd._is_a_share("920001", universe) is True, "北交所 920 段是合法 A 股"
    assert fd._is_a_share("830001", universe) is False, "83 段新三板必须剔除"
    assert fd._is_a_share("874746", universe) is False, "87 段新三板必须剔除"


def test_b_share_excluded_structurally():
    """沪B(900)/深B(200) 不是 A 股——即使面板误收也要挡在结构判定这一层。"""
    assert fd._is_a_share("900901", {"900901"}) is False
    assert fd._is_a_share("200001", {"200001"}) is False


def test_empty_universe_falls_back_to_structural():
    """面板不可用时只做结构判定（宁可多留，也不能整表丢掉）。"""
    assert fd._is_a_share("600519", set()) is True
    assert fd._is_a_share("900901", set()) is False


# ================================================================
# 4. 落库与查询
# ================================================================

def test_upsert_is_idempotent_and_last_write_wins(temp_db):
    """重抓幂等，且**后写覆盖**。

    ⚠️ 第一版是"`ann_date` 只向更晚推进"，2026-10-30 被实测推翻：东财把历史 0930 期
    的公告日串成了下一年的 Q3（20250930 有 4,927 行标着 2026-10，与 20260930 期 100%
    重合）。"只更晚"规则让这些**更晚的错误值永远无法被正确值覆盖**——防前视的规则
    反过来把脏数据钉死了。⇒ 落库改成覆盖，把关挪到抓取层（来源优先级 + 合理性窗口）。
    """
    rows = [{"stock_code": "600519", "period": "20260630",
             "ann_date": "2026-08-20", "source": "test"}]
    assert db_store.upsert_fin_disclosures(rows) == 1
    assert db_store.upsert_fin_disclosures(rows) == 1          # 再写一次不报错
    assert db_store.get_fin_disclosure("600519", "20260630")["ann_date"] == "2026-08-20"

    # 更晚 ⇒ 更新（更正公告）
    db_store.upsert_fin_disclosures([{"stock_code": "600519", "period": "20260630",
                                      "ann_date": "2026-09-01"}])
    assert db_store.get_fin_disclosure("600519", "20260630")["ann_date"] == "2026-09-01"

    # 更早 ⇒ **也覆盖**（这正是修正串期脏数据需要的能力）
    db_store.upsert_fin_disclosures([{"stock_code": "600519", "period": "20260630",
                                      "ann_date": "2026-07-01"}])
    assert db_store.get_fin_disclosure("600519", "20260630")["ann_date"] == "2026-07-01"


def test_delete_allows_reset(temp_db):
    """`delete_fin_disclosures()` 必须能清空——来源口径变更时要能重来，
    否则抓不到的脏行会一直留着（幂等 upsert 只覆盖本次抓到的那几期）。"""
    db_store.upsert_fin_disclosures([
        {"stock_code": "600519", "period": "20260630", "ann_date": "2026-08-20"},
        {"stock_code": "000001", "period": "20251231", "ann_date": "2026-01-20"},
    ])
    assert db_store.delete_fin_disclosures("20260630") == 1
    assert [r["period"] for r in db_store.list_fin_disclosures()] == ["20251231"]
    assert db_store.delete_fin_disclosures() == 1
    assert db_store.fin_disclosure_coverage()["rows"] == 0


def test_upsert_skips_rows_missing_fields(temp_db):
    assert db_store.upsert_fin_disclosures(
        [{"stock_code": "", "period": "20260630", "ann_date": "2026-08-20"}]) == 0
    assert db_store.upsert_fin_disclosures([]) == 0


def test_list_filters_by_as_of(temp_db):
    db_store.upsert_fin_disclosures([
        {"stock_code": "600519", "period": "20260630", "ann_date": "2026-08-20"},
        {"stock_code": "600519", "period": "20260331", "ann_date": "2026-04-25"},
    ])
    all_rows = db_store.list_fin_disclosures(stock_code="600519")
    assert len(all_rows) == 2
    early = db_store.list_fin_disclosures(stock_code="600519", as_of="2026-06-30")
    assert [r["period"] for r in early] == ["20260331"]


def test_coverage_reports_periods(temp_db):
    db_store.upsert_fin_disclosures([
        {"stock_code": "600519", "period": "20260630", "ann_date": "2026-08-20"},
        {"stock_code": "000001", "period": "20260630", "ann_date": "2026-08-21"},
    ])
    cov = db_store.fin_disclosure_coverage()
    assert cov["rows"] == 2 and cov["period_count"] == 1
    assert cov["periods"][0]["n"] == 2


# ================================================================
# 5. tool_financials 的 as_of（真实裁期）
# ================================================================

def _patch_fin(monkeypatch):
    """三期财报：H1（8/20 公布）、Q1（4/25 公布）、去年同期（无公告日记录）。"""
    fin = {
        "income_statement": [
            {"报告日": "20260630", "营业收入": 704_000_000.0, "净利润": 74_800_000.0},
            {"报告日": "20260331", "营业收入": 316_000_000.0, "净利润": 23_800_000.0},
            {"报告日": "20250630", "营业收入": 669_000_000.0, "净利润": 6_100_000.0},
        ],
        "balance_sheet": [
            {"报告日": "20260630", "资产总计": 1_082_600_000.0, "负债合计": 185_000_000.0},
        ],
        "cash_flow": [],
    }

    class F:
        def get_financial_data(self, symbol):
            import copy
            return copy.deepcopy(dict(fin, symbol=symbol))

    from data import stock_data as sd
    monkeypatch.setattr(sd, "StockDataFetcher", F)


def test_tool_financials_without_as_of_is_unchanged(temp_db, monkeypatch):
    """不传 as_of ⇒ 不过滤（与加入本参数前行为一致），且给出公告日。"""
    db_store.upsert_fin_disclosures([
        {"stock_code": "603127", "period": "20260630", "ann_date": "2026-08-20"},
        {"stock_code": "603127", "period": "20260331", "ann_date": "2026-04-25"},
    ])
    _patch_fin(monkeypatch)
    from core.agent_tools_fin import tool_financials
    out = tool_financials(symbol="603127", periods=3)
    assert out["报告期"][:3] == ["2026-06-30", "2026-03-31", "2025-06-30"]
    assert out["公告日"][:2] == ["2026-08-20", "2026-04-25"]
    assert out["公告日"][2] == "", "缺记录的期给空串，不当成'未公布'"
    assert out["公告日缺失期数"] == 1
    assert "前视防护" not in out, "没传 as_of 不该出现前视防护块"


def test_tool_financials_as_of_drops_unannounced(temp_db, monkeypatch):
    """as_of=2026-06-30 ⇒ H1(8/20) 当时还没公布、去年同期无记录 ⇒ 只剩 Q1。"""
    db_store.upsert_fin_disclosures([
        {"stock_code": "603127", "period": "20260630", "ann_date": "2026-08-20"},
        {"stock_code": "603127", "period": "20260331", "ann_date": "2026-04-25"},
    ])
    _patch_fin(monkeypatch)
    from core.agent_tools_fin import tool_financials
    out = tool_financials(symbol="603127", periods=3, as_of="2026-06-30")
    assert out["报告期"] == ["2026-03-31"], out.get("报告期")
    assert out["数据截止"] == "2026-03-31"
    vp = out["前视防护"]
    assert vp["as_of"] == "2026-06-30" and vp["可用期数"] == 1 and vp["总期数"] == 3
    reasons = {d["period"]: d["reason"] for d in vp["已剔除"]}
    assert reasons.get("20260630") == "not_announced"
    assert reasons.get("20250630") == "unverified", "缺记录必须标 unverified 而不是 not_announced"


def test_tool_financials_as_of_fail_closed_when_nothing_known(temp_db, monkeypatch):
    """库里有数据但全都晚于 as_of ⇒ 必须报错并说清，不能返回全部（那就是前视）。"""
    db_store.upsert_fin_disclosures([
        {"stock_code": "603127", "period": "20260630", "ann_date": "2026-08-20"},
        {"stock_code": "603127", "period": "20260331", "ann_date": "2026-04-25"},
    ])
    _patch_fin(monkeypatch)
    from core.agent_tools_fin import tool_financials
    out = tool_financials(symbol="603127", periods=3, as_of="2026-01-01")
    assert "error" in out and "前视防护" in out
    dropped = out["前视防护"]["已剔除"]
    assert len(dropped) == 3, f"三期都该被剔除，实际 {dropped}"
    assert {d["reason"] for d in dropped} == {"not_announced", "unverified"}


def test_tool_financials_empty_disclosure_table_is_fail_closed(temp_db, monkeypatch):
    """公告日表空（没回填）⇒ 传了 as_of 就一期都不给（fail-closed），而不是放行。"""
    _patch_fin(monkeypatch)
    from core.agent_tools_fin import tool_financials
    out = tool_financials(symbol="603127", periods=3, as_of="2026-06-30")
    assert "error" in out
    assert all(d["reason"] == "unverified" for d in out["前视防护"]["已剔除"])


# 6. 公告日覆盖：并入 tool_data_freshness（工具预算上限 30，不再单独开工具）
# ================================================================

def test_freshness_reports_disclosure_coverage(temp_db):
    """公告日覆盖必须出现在 `tool_data_freshness` 里。

    为什么并入而不单开工具：助手工具数有**硬预算 30**（`test_agent_conclusion_tools`
    挡着），而"公告日数据覆盖到哪了"本质就是数据新鲜度问题。
    """
    db_store.upsert_fin_disclosures([
        {"stock_code": "600519", "period": "20260630", "ann_date": "2026-08-15"},
        {"stock_code": "000001", "period": "20260630", "ann_date": "2026-08-16"},
    ])
    from core.agent_tools_sys import tool_data_freshness
    out = tool_data_freshness()
    pit = out.get("财报公告日(PIT)") or {}
    assert pit.get("已回填报告期") == 1
    assert pit.get("总行数") == 2
    assert pit["最近三期"][0]["报告期"] == "20260630"
    assert pit["最近三期"][0]["股票数"] == 2
    assert "fail-closed" in pit["说明"]


# 7. 渲染层：公告日必须真的送到模型面前
# ================================================================

def test_renderer_emits_disclosure_date_list(temp_db, monkeypatch):
    """兜底循环只认 dict/标量，**list 会被静默丢掉** ⇒ 公告日必须显式渲染。"""
    db_store.upsert_fin_disclosures([
        {"stock_code": "603127", "period": "20260630", "ann_date": "2026-08-20"},
    ])
    _patch_fin(monkeypatch)
    from core.agent_tools_fin import tool_financials
    from llm.deepseek_client import _render_financials
    out = tool_financials(symbol="603127", periods=3, as_of="2026-09-30")
    text = _render_financials(out)
    assert "公告日" in text, text
    assert "2026-08-20" in text, "公告日列表没渲染出来（被兜底循环丢了）"
    assert "前视防护" in text and "2026-09-30" in text


# ================================================================
# 8. 抓取层：巨潮主源 / 合理性窗口 / 东财兜底
# ================================================================

def _fake_disclosure(df, monkeypatch, universe):
    import akshare as ak
    monkeypatch.setattr(ak, "stock_report_disclosure", lambda **kw: df)
    monkeypatch.setattr(fd, "a_share_universe", lambda: set(universe))


def test_cninfo_uses_actual_disclosure_and_skips_undisclosed(monkeypatch):
    """主源用 `实际披露`（真实披露日）；未披露的行不写（缺就是缺，不留假值）。"""
    import pandas as pd
    df = pd.DataFrame({
        "股票代码": ["600519", "000001", "830001"],
        "实际披露": [pd.Timestamp("2026-08-20").date(), pd.NaT,
                     pd.Timestamp("2026-08-21").date()],
    })
    _fake_disclosure(df, monkeypatch, {"600519", "000001"})
    r = fd.fetch_period("20260630")
    assert "error" not in r, r
    assert r["source"] == fd.SOURCE_CNINFO
    assert [x["stock_code"] for x in r["rows"]] == ["600519"]
    assert r["rows"][0]["ann_date"] == "2026-08-20"
    assert r["not_disclosed"] == 1, "未披露必须计数，不能当成已披露"
    assert r["dropped_non_a"] == 1, "新三板(83 段)必须被面板边界挡掉"


def test_contaminated_dates_are_rejected(monkeypatch):
    """⚠️ 实测事故的回归：历史 0930 期的公告日被串成**下一年 Q3**，必须拒收。

    20250930 期的合理窗口 = 2025-09-23 ~ 2026-05-28；2026-10 的日期超出 ⇒ 丢弃。
    若不拦，PIT 会把"2025 三季报"判成 2026 年才公布，历史 Q3 全部变成"当时看不到"。
    """
    import pandas as pd
    df = pd.DataFrame({
        "股票代码": ["600519", "000001"],
        "实际披露": [pd.Timestamp("2025-10-25").date(),     # 真实
                     pd.Timestamp("2026-10-30").date()],    # 串期
    })
    _fake_disclosure(df, monkeypatch, {"600519", "000001"})
    r = fd.fetch_period("20250930")
    assert "error" not in r, r
    assert [x["stock_code"] for x in r["rows"]] == ["600519"]
    assert r["implausible"] == 1, "超窗口的行必须被丢弃并计数"


def test_plausible_window_boundaries():
    lo, hi = fd.plausible_window("20250930")
    assert (lo.year, lo.month, lo.day) == (2025, 9, 23)
    assert (hi.year, hi.month, hi.day) == (2026, 5, 28)      # 期末 +240 天
    assert fd.plausible_window("bad") == (None, None)


def test_falls_back_to_yjbb_when_cninfo_fails(monkeypatch):
    """巨潮挂掉时回退东财，并在返回值里说明原因（不是静默换源）。"""
    import pandas as pd
    import akshare as ak

    def _boom(**kw):
        raise RuntimeError("cninfo down")

    monkeypatch.setattr(ak, "stock_report_disclosure", _boom)
    monkeypatch.setattr(ak, "stock_yjbb_em", lambda date: pd.DataFrame({
        "股票代码": ["600519"],
        "最新公告日期": [pd.Timestamp("2026-08-20").date()],
    }))
    monkeypatch.setattr(fd, "a_share_universe", lambda: {"600519"})
    r = fd.fetch_period("20260630")
    assert r["source"] == fd.SOURCE_YJBB
    assert "cninfo down" in r["fallback_reason"]
    assert r["rows"][0]["ann_date"] == "2026-08-20"


def test_both_sources_failing_reports_error(monkeypatch):
    """两个源都挂 ⇒ 必须报错，**不能返回空列表冒充成功**。"""
    import akshare as ak

    def _boom(**kw):
        raise RuntimeError("down")

    monkeypatch.setattr(ak, "stock_report_disclosure", _boom)
    monkeypatch.setattr(ak, "stock_yjbb_em", _boom)
    r = fd.fetch_period("20260630")
    assert "error" in r and "两个来源都失败" in r["error"]


def test_bad_period_is_rejected():
    assert "error" in fd.fetch_period("2026-06")


# ================================================================
# 9. 独立工具 tool_fin_disclosures（预算 30→31，2026-10-30 用户同意加）
# ================================================================

def test_disclosures_tool_by_code(temp_db):
    db_store.upsert_fin_disclosures([
        {"stock_code": "600519", "period": "20260630", "ann_date": "2026-08-15"},
        {"stock_code": "600519", "period": "20260331", "ann_date": "2026-04-25"},
    ])
    from core.agent_tools_fin import tool_fin_disclosures
    out = tool_fin_disclosures(code="600519")
    assert out["记录数"] == 2
    assert out["公告日"][0]["报告期"] == "2026-06-30"       # 期倒序
    assert "保守下界" in out["口径"]


def test_disclosures_tool_respects_as_of(temp_db):
    db_store.upsert_fin_disclosures([
        {"stock_code": "600519", "period": "20260630", "ann_date": "2026-08-15"},
        {"stock_code": "600519", "period": "20260331", "ann_date": "2026-04-25"},
    ])
    from core.agent_tools_fin import tool_fin_disclosures
    out = tool_fin_disclosures(code="600519", as_of="2026-06-30")
    assert out["记录数"] == 1
    assert out["公告日"][0]["报告期"] == "2026-03-31"


def test_disclosures_tool_unknown_code_says_so(temp_db):
    from core.agent_tools_fin import tool_fin_disclosures
    out = tool_fin_disclosures(code="600519")
    assert "error" in out and "提示" in out


def test_disclosures_tool_global_overview(temp_db):
    db_store.upsert_fin_disclosures([
        {"stock_code": "600519", "period": "20260630", "ann_date": "2026-08-15"},
    ])
    from core.agent_tools_fin import tool_fin_disclosures
    out = tool_fin_disclosures()
    assert out["已回填报告期"] == 1 and out["总行数"] == 1
    assert out["详情"][0]["报告期"] == "2026-06-30"


def test_disclosures_tool_bad_code(temp_db):
    from core.agent_tools_fin import tool_fin_disclosures
    assert "error" in tool_fin_disclosures(code="6005")


def test_disclosures_tool_in_char_limits():
    """新工具必须进字符配额表，否则默认 1500 会截断（历史踩过的坑）。"""
    from core.chat_agent import _TOOL_RESULT_CHAR_LIMITS
    assert _TOOL_RESULT_CHAR_LIMITS.get("tool_fin_disclosures", 0) >= 3000
