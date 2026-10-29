"""估值截面快照（`market_snapshot`）：落库 / 读库 / 保留策略 / 单位换算。

存在理由（一句话）：**IC 需要"某日截面 vs 之后收益"**，而 PE / 市值 / 换手率的
绝对水平只存在于当时的快照里 —— 拿今天的 PE 算去年的 IC 就是前视。
这是"六因子权重里 PE 那一项无法验证"的前置条件。

本文件守护四件事：
1. 落库/读库/覆盖语义正确（同 date+code 覆盖，不产生重复行）；
2. **单位换算只做一次且方向正确**（新浪给的是亿元，落库统一为元）；
3. **保留策略按 `snapshot_date` 而不是 `created_at` 判过期**
   —— 补采历史数据时 created_at 是"今天"，用它会永远删不掉旧数据；
4. 上游不可用时**不写入残缺数据**、进度如实报告。
"""

from unittest.mock import patch

import pandas as pd
import pytest

from db import store as db_store


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    """数据库指向临时文件，隔离真实 portfolio.db"""
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "test_mkt.db")
    yield


def _snap_df():
    """模拟 `data.sina_source` 的输出：金额列**亿元**、百分比列**百分数**"""
    df = pd.DataFrame([
        {"code": "600519", "name": "贵州茅台", "industry": "酿酒行业",
         "board": "主板沪市", "price": 1680.0, "change_pct": 1.23,
         "pe": 19.169, "pb": 8.5, "turnover": 0.35,
         "amount": 25.6,          # 亿元
         "market_cap": 15733.8,   # 亿元
         "float_cap": 15700.0},   # 亿元
        {"code": "000001", "name": "平安银行", "industry": "金融行业",
         "board": "主板深市", "price": 11.2, "change_pct": -0.5,
         "pe": 5.1, "pb": 0.6, "turnover": 0.8,
         "amount": 12.3, "market_cap": 2173.0, "float_cap": 2170.0},
    ])
    df.attrs.update({"coverage": 0.54, "caveat": "本次候选域 2987/5500 只"})
    return df


# ================================================================
# 1. 落库 / 读库
# ================================================================

class TestMarketSnapshotStore:
    def test_save_and_load(self, temp_store):
        n = db_store.save_market_snapshot("2026-09-30", [
            {"stock_code": "600519", "stock_name": "贵州茅台", "pe": 19.169,
             "market_cap": 1.57e12, "turnover": 0.35, "industry": "酿酒行业",
             "board": "主板沪市", "price": 1680.0, "change_pct": 1.23},
        ])
        assert n == 1
        date, rows = db_store.load_market_snapshot("2026-09-30")
        assert date == "2026-09-30"
        assert len(rows) == 1
        assert rows[0]["pe"] == pytest.approx(19.169)
        assert rows[0]["market_cap"] == pytest.approx(1.57e12)

    def test_same_date_code_overwrites(self, temp_store):
        """同 date+code 必须覆盖（ON CONFLICT），不能累积重复行"""
        db_store.save_market_snapshot("2026-09-30", [
            {"stock_code": "600519", "pe": 19.0}])
        db_store.save_market_snapshot("2026-09-30", [
            {"stock_code": "600519", "pe": 20.5}])
        _date, rows = db_store.load_market_snapshot("2026-09-30")
        assert len(rows) == 1
        assert rows[0]["pe"] == pytest.approx(20.5)

    def test_dates_are_sorted_and_deduped(self, temp_store):
        for d in ("2026-09-29", "2026-09-30", "2026-09-28"):
            db_store.save_market_snapshot(d, [{"stock_code": "600519", "pe": 19.0}])
        assert db_store.market_snapshot_dates() == ["2026-09-28", "2026-09-29", "2026-09-30"]

    def test_empty_input_writes_nothing(self, temp_store):
        assert db_store.save_market_snapshot("2026-09-30", []) == 0
        assert db_store.save_market_snapshot("", [{"stock_code": "600519"}]) == 0
        assert db_store.market_snapshot_dates() == []

    def test_load_defaults_to_latest_date(self, temp_store):
        """不传日期 → 取最近一天；但**必须把实际日期返回**给调用方"""
        db_store.save_market_snapshot("2026-09-28", [{"stock_code": "600519", "pe": 18.0}])
        db_store.save_market_snapshot("2026-09-30", [{"stock_code": "600519", "pe": 20.0}])
        date, rows = db_store.load_market_snapshot()
        assert date == "2026-09-30"
        assert rows[0]["pe"] == pytest.approx(20.0)

    def test_load_missing_date_returns_empty(self, temp_store):
        assert db_store.load_market_snapshot("2000-01-01") == ("2000-01-01", [])


# ================================================================
# 2. 保留策略：按 snapshot_date 判过期
# ================================================================

class TestRetentionPolicy:
    def test_market_snapshot_is_registered_for_cleanup(self, temp_store):
        """新表必须登记保留策略 —— 否则它会无界增长（本项目明写的规矩）"""
        deleted = db_store.cleanup()
        assert "market_snapshot" in deleted, "market_snapshot 没登记保留策略"

    def test_cleanup_uses_snapshot_date_not_created_at(self, temp_store):
        """过期判据必须看**数据日期**，而不是"什么时候写进来的"。

        ⚠️ 反向：若用 created_at，补采回来的历史快照（created_at=今天）
        永远删不掉；正向：若把 snapshot_date 当成可忽略的列，老数据会越积越多。
        这里造一条"数据日期很老、created_at 是今天"的行 —— 它必须被清掉。
        """
        db_store.save_market_snapshot("2000-01-04", [{"stock_code": "600519", "pe": 5.0}])
        db_store.save_market_snapshot("2099-01-01", [{"stock_code": "600519", "pe": 5.0}])
        db_store.cleanup()
        dates = db_store.market_snapshot_dates()
        assert "2000-01-04" not in dates, "按 snapshot_date 过期：老数据必须被清掉"
        assert "2099-01-01" in dates, "未来日期不该被清掉（判据方向反了？）"


# ================================================================
# 3. 采集：单位换算只做一次，且方向正确
# ================================================================

class TestDailyCapture:
    def test_yi_to_yuan_conversion(self, temp_store):
        """新浪金额列是**亿元**，落库统一为**元**。

        ⚠️ 这是本项目反复踩过的坑（"列名相同 ≠ 单位相同"）：若漏了换算，
        贵州茅台的市值会从 1.57 万亿变成 1.57 万 —— 数字还是"有"，看不出错。
        """
        import core.scheduler as sched
        with patch("data.sina_source.get_sector_snapshot", return_value=_snap_df()):
            out = sched.save_daily_market_snapshot("2026-09-30")
        assert out["ok"] is True and out["rows"] == 2
        _date, rows = db_store.load_market_snapshot("2026-09-30")
        mt = {r["stock_code"]: r for r in rows}
        assert mt["600519"]["market_cap"] == pytest.approx(15733.8 * 1e8)
        assert mt["600519"]["amount"] == pytest.approx(25.6 * 1e8)
        assert mt["600519"]["float_cap"] == pytest.approx(15700.0 * 1e8)
        # pe / turnover 是原始口径，**不得**被换算
        assert mt["600519"]["pe"] == pytest.approx(19.169)
        assert mt["600519"]["turnover"] == pytest.approx(0.35)

    def test_upstream_unavailable_writes_nothing(self, temp_store):
        """上游不可用 → 不写入残缺数据（"缺就是缺，不补造"）"""
        import core.scheduler as sched
        with patch("data.sina_source.get_sector_snapshot", return_value=None):
            out = sched.save_daily_market_snapshot("2026-09-30")
        assert out["ok"] is False
        assert "不可用" in out["reason"]
        assert db_store.market_snapshot_dates() == [], "取不到数据却写了行"

    def test_snapshot_date_is_data_date_not_capture_time(self, temp_store):
        """落库的 snapshot_date 用**数据所属交易日**，且 captured_at 另存"""
        import core.scheduler as sched
        with patch("data.sina_source.get_sector_snapshot", return_value=_snap_df()):
            sched.save_daily_market_snapshot()          # 不传日期 → 自己推算交易日
        _date, rows = db_store.load_market_snapshot()
        assert rows, "没落库"
        row = rows[0]
        assert row["snapshot_date"], "数据日期为空"
        assert row["captured_at"], "采集时刻必须单独记录（否则补采时会误导）"

    def test_b_share_codes_are_dropped(self, temp_store):
        """新浪回退路径必须剔掉 B 股（沪B 900xxx / 深B 200xxx）。

        ⚠️ 2026-10-23 实翻：`market_snapshot` 2026-10-22 落库 2991 只里
        混了 52 只沪B + 37 只深B。tushare 主路径没有这个问题（它取的是
        stock_basic 的 A 股清单），**只有回退新浪时**才会带进来。
        后果不只是"多几行"：这些票会经 `local_screen` 进候选池，白跑取数
        （每只 30~120s 且恒失败），还挤占 STRATEGY_TOP_N 名额。
        """
        import core.scheduler as sched
        df = _snap_df()
        df = pd.concat([df, pd.DataFrame([
            {"code": "900923", "name": "江苏B股", "industry": "金融行业",
             "board": "主板沪市", "price": 1.2, "change_pct": 0.0,
             "pe": 5.0, "pb": 0.4, "turnover": 0.1,
             "amount": 0.02, "market_cap": 10.0, "float_cap": 10.0},
            {"code": "200011", "name": "深物业B", "industry": "房地产",
             "board": "主板深市", "price": 3.1, "change_pct": 0.0,
             "pe": 8.0, "pb": 0.5, "turnover": 0.2,
             "amount": 0.03, "market_cap": 20.0, "float_cap": 20.0},
        ])], ignore_index=True)
        df.attrs.update({"coverage": 0.54, "caveat": "x"})

        with patch("data.sina_source.get_sector_snapshot", return_value=df):
            out = sched.save_daily_market_snapshot("2026-09-30")

        assert out["ok"] is True
        assert out["rows"] == 2, "B 股应被剔除，只留 2 只 A 股"
        _d, rows = db_store.load_market_snapshot("2026-09-30")
        codes = {r["stock_code"] for r in rows}
        assert codes == {"600519", "000001"}
        assert "900923" not in codes and "200011" not in codes


class TestIsAShareCode:
    """`is_a_share_code` 是**纯结构判定**（不依赖网络/缓存清单）。

    为什么不复用 `sina_source.market_codes()`：它拿不到时返回 None，
    若据此"不在清单就丢"会把整张表丢光 —— 守卫本身不能成为新的故障点。
    """

    @pytest.mark.parametrize("code", [
        "600000", "601288", "603288", "605499",   # 沪主板
        "688981", "689009",                        # 科创板
        "000001", "001979", "002594", "003816",   # 深主板/中小板
        "300750", "301236",                        # 创业板
        "830799", "430047", "920000",              # 北交所
    ])
    def test_a_share_accepted(self, code):
        import core.scheduler as sched
        assert sched.is_a_share_code(code) is True

    @pytest.mark.parametrize("code", [
        "900923", "900905", "900928",              # 沪 B
        "200011", "200002",                        # 深 B
        "", "abc", "12345", "6000000",             # 非法
        "60000A",
    ])
    def test_non_a_share_rejected(self, code):
        import core.scheduler as sched
        assert sched.is_a_share_code(code) is False


# ================================================================
# 4. 进度披露：把"不可算"变成一条可行动的进度
# ================================================================

class TestSnapshotHistoryStatus:
    def test_not_enough_reports_progress_with_numbers(self, temp_store):
        from core import factor_analysis as fa
        for i in range(3):
            db_store.save_market_snapshot(f"2026-09-2{i}", [{"stock_code": "600519"}])
        st = fa.snapshot_history_status()
        assert st["dates"] == 3 and st["enough"] is False
        assert "3" in st["reason"] and str(fa.MIN_SNAPSHOT_DATES) in st["reason"], (
            "进度说明里必须同时给出'已有'与'需要'两个数字，否则用户无法判断要等多久")

    def test_enough_flag_flips_at_threshold(self, temp_store):
        from core import factor_analysis as fa
        for i in range(fa.MIN_SNAPSHOT_DATES):
            db_store.save_market_snapshot(f"2026-01-{i + 1:02d}", [{"stock_code": "600519"}])
        st = fa.snapshot_history_status()
        assert st["dates"] == fa.MIN_SNAPSHOT_DATES and st["enough"] is True
        assert st["reason"] == "", "够了就不该再有'不可算'的理由"

    def test_status_reaches_unavailable_reasons(self, temp_store):
        """进度必须接进 `run()` 的 unavailable 文案（漏接 = 用户还是看不到）"""
        from core import factor_analysis as fa
        db_store.save_market_snapshot("2026-09-30", [{"stock_code": "600519"}])
        res = fa.run([], factors=["momentum20"], bars_map={})
        for k in fa.SNAPSHOT_FACTORS:
            assert "已积累 1" in res.unavailable[k], (
                f"{k} 的理由没接上实时进度：{res.unavailable[k]}")
        assert res.snapshot_status["dates"] == 1


# ================================================================
# 5. 无报价标的不得入库（2026-10-26：快照常驻 186 只 price=0 的退市票）
# ================================================================

class TestNoQuoteRowsRejected:
    """⚠️ `price=0` 不是"0 元"，是**没有报价**（退市/长期停牌）。

    实测事故：新浪行业明细对退市票长期返回 `trade=0`，`market_snapshot` 里
    **每天 186 只**这类行（`000022 深赤湾A`、`000003 PT金田A`…），
    389 只零涨幅里 186 只属此类；它们还各白跑一次估值预取。
    """

    def test_price_zero_is_not_written(self, temp_store):
        n = db_store.save_market_snapshot("2026-10-26", [
            {"stock_code": "600519", "price": 1680.0, "change_pct": 1.2},
            {"stock_code": "000022", "stock_name": "深赤湾A", "price": 0.0,
             "change_pct": 0.0},
            {"stock_code": "000003", "stock_name": "PT金田A", "price": -1.0},
        ])
        assert n == 1, "无报价的两只不能被写进去"
        _, rows = db_store.load_market_snapshot("2026-10-26")
        assert [r["stock_code"] for r in rows] == ["600519"]

    def test_price_none_still_written(self, temp_store):
        """`price=None` 是"没提供"（PE/PB 补采只写估值列），**不是坏值** ——
        拦它会把正常的估值补采全丢掉。这条是上面那个守卫的边界。"""
        n = db_store.save_market_snapshot("2026-10-26", [
            {"stock_code": "600519", "pe": 19.0},
        ])
        assert n == 1

    def test_all_rejected_returns_zero(self, temp_store):
        n = db_store.save_market_snapshot("2026-10-26", [
            {"stock_code": "000022", "price": 0.0},
        ])
        assert n == 0


# ================================================================
# 6. 全量域主路径（2026-10-26）：不再依赖 daily_basic 权限
# ================================================================

class TestFullUniversePath:
    """`save_daily_market_snapshot` 的**第一档**必须是全量域（`data.full_market`）。

    为什么：原来第一档是 tushare `daily_basic`，而 120 积分档**无此权限**（能力表
    「被拒 3 次、成功 0 次」）⇒ 每天抛异常回退新浪 49 行业明细 ⇒ 覆盖率锁在 47%，
    创业板 31% / 科创板 5%，`300750`、`688111` 长期不在选股域内。
    """

    def test_full_market_is_primary(self, temp_store, monkeypatch):
        import core.scheduler as sched
        import data.full_market as fm

        def _fake(date, **_kw):
            db_store.save_market_snapshot(date, [
                {"stock_code": "300750", "stock_name": "宁德时代", "price": 238.0,
                 "change_pct": 0.17, "pe": 14.7, "industry": "电池"}],
                source="sina_full")
            return {"ok": True, "snapshot_date": date, "rows": 1, "covered": 1,
                    "universe": 2, "coverage": 0.5, "industry_covered": 1,
                    "stock_basic_source": "cache", "unit": "u", "degraded": []}

        monkeypatch.setattr(fm, "save_full_snapshot", _fake)

        def _boom(*_a, **_k):
            raise AssertionError("全量域成功时不该再走 tushare 路径")

        monkeypatch.setattr("data.tushare_source.build_market_snapshot_rows", _boom)

        out = sched.save_daily_market_snapshot("2026-10-26")
        assert out["ok"] is True
        assert out["coverage"] == 0.5
        _d, rows = db_store.load_market_snapshot("2026-10-26")
        assert rows[0]["stock_code"] == "300750"
        assert rows[0]["source"] == "sina_full"

    def test_full_market_failure_falls_back(self, temp_store, monkeypatch):
        """全量域失败 ⇒ 仍要落库（回退链不能因为第一档挂了就整体失败）"""
        import core.scheduler as sched
        import data.full_market as fm

        monkeypatch.setattr(fm, "save_full_snapshot",
                            lambda *_a, **_k: {"ok": False, "reason": "测试：采残"})
        with patch("data.sina_source.get_sector_snapshot", return_value=_snap_df()):
            out = sched.save_daily_market_snapshot("2026-09-30")
        assert out["ok"] is True and out["rows"] == 2
