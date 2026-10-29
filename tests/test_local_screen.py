"""本地五路初筛单元测试（临时库隔离，零网络）

背景（2026-10-09）：pywencai 失效 → 选股每次都降级到 `SECTOR_HARDCODED` 前 18 只
→ 结果跨日高度重复。本模块把初筛改到本地全市场数据（`market_snapshot` +
`fund_flow_snapshot`），这些用例锁住新契约：

1. 五路各自的取数逻辑与**条数上限**
2. **龙虎榜路是 unavailable**（无数据源必须显式，不静默跳过）
3. **ST 剔除**（原问财每路都带"剔除ST"，本地化后必须自己剔）
4. ⚠️ **单位坑**：快照 `market_cap` 是**元**，200 亿元阈值 = 2e10
5. 快照缺失 → 该路 unavailable（不拿旧日期数据充数）
6. 并集**按路轮流截断**（保证各路均衡）并记录被截断数
"""
import pytest

from db import store as db_store


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "test_local_screen.db")
    yield


DAY = "2026-10-09"
YI = 1e8


def _seed_market(rows):
    """rows: [{code, name, pe, cap_yi(亿元), turnover, amount, change_pct, industry}]"""
    db_store.save_market_snapshot(DAY, [
        {"stock_code": r["code"], "stock_name": r["name"], "industry": r.get("industry", "测试行业"),
         "board": "主板沪市", "price": 10.0, "change_pct": r.get("change_pct", 0.0),
         "pe": r.get("pe"), "pb": 1.0,
         "market_cap": (r.get("cap_yi") or 0) * YI,      # ⚠️ 库里单位是**元**
         "float_cap": (r.get("cap_yi") or 0) * YI, "turnover": r.get("turnover", 0.0),
         "amount": (r.get("amount_yi") or 0) * YI}
        for r in rows], source="test")


def _seed_flow(rows):
    """rows: [{code, name, net5_yi(亿元)}]"""
    db_store.save_fund_flow_snapshot(DAY, [
        {"stock_code": r["code"], "stock_name": r["name"],
         "net_today": (r.get("net5_yi") or 0) * YI / 5, "net_3d": (r.get("net5_yi") or 0) * YI * 0.6,
         "net_5d": (r.get("net5_yi") or 0) * YI, "net_10d": (r.get("net5_yi") or 0) * YI,
         "amount": 1 * YI, "change_pct": 1.0, "price": 10.0}
        for r in rows], source="test")


def _routes(meta):
    return {r["label"]: r for r in meta["routes"]}


class TestRoutes:

    def test_all_routes_present_with_dragon_tiger_unavailable(self, temp_store):
        """五路都要出现；龙虎榜**显式 unavailable**（无数据源）"""
        from core.local_screen import run_local_strategies
        _seed_market([{"code": "600001", "name": "甲", "pe": 10, "cap_yi": 300,
                       "turnover": 2, "amount_yi": 5, "change_pct": 1.0}])
        _seed_flow([{"code": "600001", "name": "甲", "net5_yi": 3}])
        res = run_local_strategies()
        r = _routes(res["meta"])
        assert set(r) == {"主力资金", "低估值成长", "技术突破", "行业龙头", "龙虎榜活跃"}
        assert r["龙虎榜活跃"]["status"] == "unavailable"
        assert "无本地数据源" in r["龙虎榜活跃"]["note"]
        # 一路不可用必须出现在 notes 里（不静默）
        assert any("龙虎榜" in n for n in res["meta"]["notes"])

    def test_every_ok_route_reports_condition_difference(self, temp_store):
        """本地条件比原问财窄 → 每路都要说明差异（如缺"净利润同比/ROE"）"""
        from core.local_screen import run_local_strategies
        _seed_market([{"code": "600001", "name": "甲", "pe": 10, "cap_yi": 300,
                       "turnover": 2, "amount_yi": 5, "change_pct": 1.0}])
        _seed_flow([{"code": "600001", "name": "甲", "net5_yi": 3}])
        r = _routes(run_local_strategies()["meta"])
        for label in ("主力资金", "低估值成长", "技术突破", "行业龙头"):
            assert r[label]["note"], f"{label} 必须说明与原问财条件的差异"
        assert "无本地数据源" in r["低估值成长"]["note"]      # 净利同比/ROE 确实没有
        assert "MACD" in r["技术突破"]["note"]

    def test_original_query_kept_for_traceability(self, temp_store):
        """原问财语句要随路一起给出（报告可追溯"本地条件 vs 原条件"）"""
        from core.local_screen import run_local_strategies
        _seed_market([{"code": "600001", "name": "甲", "pe": 10, "cap_yi": 300,
                       "turnover": 2, "amount_yi": 5, "change_pct": 1.0}])
        _seed_flow([{"code": "600001", "name": "甲", "net5_yi": 3}])
        r = _routes(run_local_strategies()["meta"])
        assert "主力资金净流入前20" in r["主力资金"]["query"]


class TestUnits:
    """⚠️ 单位坑守护：这是本项目反复踩的坑（列名相同 ≠ 单位相同）"""

    def test_market_cap_threshold_is_yuan(self, temp_store):
        """199 亿必须被排除、201 亿必须入选（若误当"亿元"比较，两者都会入选）"""
        from core.local_screen import run_local_strategies
        _seed_market([
            {"code": "600001", "name": "小盘", "pe": 10, "cap_yi": 199,
             "turnover": 2, "amount_yi": 1, "change_pct": 1.0},
            {"code": "600002", "name": "大盘", "pe": 10, "cap_yi": 201,
             "turnover": 2, "amount_yi": 1, "change_pct": 1.0}])
        res = run_local_strategies(labels=["行业龙头"])
        codes = [c["股票代码"] for c in res["candidates"]]
        assert "600002" in codes and "600001" not in codes, codes

    def test_fund_flow_ranking_is_yuan_scale(self, temp_store):
        """资金流按元排序：3 亿 > 1 亿（若单位混用会被压平/颠倒）"""
        from core.local_screen import run_local_strategies
        _seed_market([{"code": "600001", "name": "甲"}, {"code": "600002", "name": "乙"}])
        _seed_flow([{"code": "600001", "name": "甲", "net5_yi": 1},
                    {"code": "600002", "name": "乙", "net5_yi": 3}])
        codes = [c["股票代码"] for c in run_local_strategies(labels=["主力资金"])["candidates"]]
        assert codes == ["600002", "600001"]


class TestFilters:

    def test_st_is_excluded_from_all_routes(self, temp_store):
        """ST / *ST / 退市必须剔除（原问财每路都带，本地化后自己剔）"""
        from core.local_screen import run_local_strategies
        _seed_market([
            {"code": "600001", "name": "正常股", "pe": 10, "cap_yi": 300,
             "turnover": 2, "amount_yi": 5, "change_pct": 1.0},
            {"code": "600002", "name": "*ST垃圾", "pe": 10, "cap_yi": 300,
             "turnover": 2, "amount_yi": 5, "change_pct": 1.0},
            {"code": "600003", "name": "退市某某", "pe": 10, "cap_yi": 300,
             "turnover": 2, "amount_yi": 5, "change_pct": 1.0}])
        _seed_flow([{"code": "600001", "name": "正常股", "net5_yi": 1},
                    {"code": "600002", "name": "*ST垃圾", "net5_yi": 9},
                    {"code": "600003", "name": "退市某某", "net5_yi": 9}])
        codes = [c["股票代码"] for c in run_local_strategies()["candidates"]]
        assert "600001" in codes
        assert "600002" not in codes and "600003" not in codes

    def test_low_valuation_requires_positive_pe(self, temp_store):
        """亏损股 PE 为负 → 不得进"低估值"路（负 PE 不是"低估"）"""
        from core.local_screen import run_local_strategies
        _seed_market([
            {"code": "600001", "name": "盈利", "pe": 8},
            {"code": "600002", "name": "亏损", "pe": -12},
            {"code": "600003", "name": "高估", "pe": 88}])
        codes = [c["股票代码"] for c in run_local_strategies(labels=["低估值成长"])["candidates"]]
        assert codes == ["600001"]

    def test_technical_breakout_bounds(self, temp_store):
        """涨停(>7%)与跌停/滞涨(<0.5%)都不属于"突破"；换手过高的过热股也排除"""
        from core.local_screen import run_local_strategies
        _seed_market([
            {"code": "600001", "name": "合规", "change_pct": 3.0, "turnover": 5, "amount_yi": 9},
            {"code": "600002", "name": "涨停", "change_pct": 9.9, "turnover": 5, "amount_yi": 9},
            {"code": "600003", "name": "滞涨", "change_pct": 0.1, "turnover": 5, "amount_yi": 9},
            {"code": "600004", "name": "过热", "change_pct": 3.0, "turnover": 30, "amount_yi": 9}])
        codes = [c["股票代码"] for c in run_local_strategies(labels=["技术突破"])["candidates"]]
        assert codes == ["600001"]


class TestUnavailableIsExplicit:

    def test_missing_market_snapshot_marks_routes_unavailable(self, temp_store):
        """没有估值快照 → 三条快照路 unavailable + 原因（不静默、不用旧日期）"""
        from core.local_screen import run_local_strategies
        _seed_flow([{"code": "600001", "name": "甲", "net5_yi": 1}])
        res = run_local_strategies()
        r = _routes(res["meta"])
        for label in ("低估值成长", "技术突破", "行业龙头"):
            assert r[label]["status"] == "unavailable"
            assert "无估值快照" in r[label]["note"]
        assert r["主力资金"]["status"] == "ok"          # 资金流有数据 → 照常
        assert res["meta"]["data_dates"]["market_snapshot"] is None

    def test_missing_flow_snapshot_marks_fund_route_unavailable(self, temp_store):
        from core.local_screen import run_local_strategies
        _seed_market([{"code": "600001", "name": "甲", "pe": 10}])
        r = _routes(run_local_strategies()["meta"])
        assert r["主力资金"]["status"] == "unavailable"
        assert "无资金流快照" in r["主力资金"]["note"]


class TestLimits:

    def test_per_route_top_n(self, temp_store):
        """每路不超过 STRATEGY_TOP_N（防止候选域与评分耗时失控）"""
        from core.local_screen import STRATEGY_TOP_N, run_local_strategies
        _seed_market([{"code": f"6000{i:02d}", "name": f"甲{i}", "pe": 10 + i * 0.01}
                      for i in range(40)])
        res = run_local_strategies(labels=["低估值成长"])
        assert len(res["candidates"]) == STRATEGY_TOP_N

    def test_union_is_round_robin_truncated(self, temp_store):
        """并集超上限 → 按路轮流截断（各路均衡），并记录截断数。

        ⚠️ 造数据必须让四路**互不重叠**，否则去重后凑不到 TOTAL_LIMIT
        （首版让同一批股票同时满足低估值/龙头/技术 → 并集只有 50 只，`truncated=0`）。
        """
        from core.local_screen import STRATEGY_TOP_N, TOTAL_LIMIT, run_local_strategies
        n = STRATEGY_TOP_N
        rows = []
        # 低估值路专属：低 PE + 小市值 + 无涨跌（不进龙头/技术路）
        rows += [{"code": f"6000{i:02d}", "name": f"低估{i}", "pe": 5 + i * 0.01,
                  "cap_yi": 50, "turnover": 0.5, "amount_yi": 1, "change_pct": 0.0}
                 for i in range(n)]
        # 龙头路专属：大市值 + 高 PE + 无涨跌
        rows += [{"code": f"6010{i:02d}", "name": f"龙头{i}", "pe": 60 + i * 0.01,
                  "cap_yi": 500 + i, "turnover": 0.5, "amount_yi": 1, "change_pct": 0.0}
                 for i in range(n)]
        # 技术路专属：有涨幅有换手 + 高 PE + 小市值
        rows += [{"code": f"6030{i:02d}", "name": f"突破{i}", "pe": 60 + i * 0.01,
                  "cap_yi": 50, "turnover": 3.0, "amount_yi": 50 - i * 0.01,
                  "change_pct": 2.0}
                 for i in range(n)]
        _seed_market(rows)
        # 资金流路专属
        _seed_flow([{"code": f"0000{i:02d}", "name": f"资金{i}", "net5_yi": 100 - i}
                    for i in range(n)])
        res = run_local_strategies()
        assert len(res["candidates"]) == TOTAL_LIMIT
        assert res["meta"]["truncated"] > 0, "被截断的数量必须记录（否则用户以为只有这么多）"
        # 均衡性：四路各自的股票都应出现（不能有一路被完全挤掉）
        for prefix in ("6000", "6010", "6030", "0000"):
            assert any(c["股票代码"].startswith(prefix) for c in res["candidates"]), prefix

    def test_labels_subset_only_runs_those_routes(self, temp_store):
        from core.local_screen import run_local_strategies
        _seed_market([{"code": "600001", "name": "甲", "pe": 10, "cap_yi": 300,
                       "turnover": 2, "amount_yi": 5, "change_pct": 1.0}])
        _seed_flow([{"code": "600002", "name": "乙", "net5_yi": 3}])
        res = run_local_strategies(labels=["行业龙头"])
        r = _routes(res["meta"])
        assert set(r) == {"行业龙头"}
        assert [c["股票代码"] for c in res["candidates"]] == ["600001"]

    def test_unknown_label_is_reported(self, temp_store):
        from core.local_screen import run_local_strategies
        res = run_local_strategies(labels=["不存在的路"])
        assert any("未知策略" in n for n in res["meta"]["notes"])

    def test_progress_callback_is_used(self, temp_store):
        """必须上报进度（选股页的进度条依赖它），且回调异常不影响主流程"""
        from core.local_screen import run_local_strategies
        events = []
        _seed_market([{"code": "600001", "name": "甲", "pe": 10}])
        run_local_strategies(progress_cb=events.append)
        assert events and any("本地初筛" in e["message"] for e in events)

        def boom(_e):
            raise RuntimeError("UI 炸了")
        run_local_strategies(progress_cb=boom)          # 不应抛出
