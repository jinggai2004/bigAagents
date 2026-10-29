"""选股候选入库与条件命中率回测单测（临时库隔离）

覆盖动机：P2「条件命中率回测」原本打算用 daily_features 做，但实测它只有**上证指数**
一个标的（没有个股面板），做不了；于是把 analyses 那套闭环复制到选股层——
一次选股 5~10 只候选，是唯一能快速攒够样本、又自带条件命中明细的地方。
本文件锁死：落库口径、回填状态机、超额收益（必须减指数）、条件/因子统计。
"""
import pytest

from db import store as db_store
from core import screen_history as sh
from core.forward_returns import compute_forward_returns, normalize_kline


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    db_path = use_template_db(tmp_path, monkeypatch, "test_screen.db")
    yield str(db_path)


def _kline(n=30, start_price=100.0, step=1.0, days_ago=0):
    """假 K 线：首行 = 分析日当天（可前移 `days_ago` 天），之后每天涨 step"""
    import pandas as pd
    from datetime import datetime, timedelta
    base = (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            - timedelta(days=days_ago))
    return pd.DataFrame({
        "date": [(base + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)],
        "close": [start_price + step * i for i in range(n)],
    })


# 回填只处理"够老"的候选（`BACKFILL_MIN_AGE_DAYS` 天以前）：选股日若比这更近，
# `list_screen_picks(max_pick_date=cutoff)` 一条都取不到，`checked=0` ⇒
# 测试会以"没有数据"的方式假失败，而不是真的暴露回填逻辑错误。
_AGED_DAYS = sh.BACKFILL_MIN_AGE_DAYS + 3


def _aged_kline(n=30, start_price=100.0, step=1.0):
    """选股日落在回填窗口内的 K 线（首行 = `_AGED_DAYS` 天前）"""
    return _kline(n, start_price, step, days_ago=_AGED_DAYS)


def _aged_date(days_ago=_AGED_DAYS):
    """回填窗口内的选股日（默认 = `_aged_kline()` 的首行日期）"""
    from datetime import datetime, timedelta
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _pick(**kw):
    p = {"pick_date": "2026-09-01", "stock_code": "600519", "stock_name": "贵州茅台",
         "source": "screen", "strategy": "主力资金", "score": 82.0,
         "score_coverage": 0.9, "price_at_pick": 100.0, "rank_in_run": 1,
         "factors": {}, "conditions": {}}
    p.update(kw)
    return p


class TestForwardReturns:
    """公共 T+N 口径（与分析历史共用同一实现）"""

    def test_base_is_first_close_on_or_after(self):
        df = normalize_kline(_kline(30, 100.0, 1.0))
        out = compute_forward_returns(df, df.iloc[0]["date"])
        # 基准 100，第 5 根 105 → +5%；第 20 根 120 → +20%
        assert out["base_price"] == 100.0
        assert out["ret_5d"] == pytest.approx(0.05)
        assert out["ret_20d"] == pytest.approx(0.20)

    def test_short_window_omits_unavailable_horizon(self):
        """不足 21 根时不能拿短窗口冒充 T+20——必须是"缺席"而不是 0"""
        df = normalize_kline(_kline(10, 100.0, 1.0))
        out = compute_forward_returns(df, df.iloc[0]["date"])
        assert "ret_5d" in out and "ret_20d" not in out

    def test_no_data_after_from_date(self):
        df = normalize_kline(_kline(5))
        out = compute_forward_returns(df, "2099-01-01")
        assert out == {}

    def test_column_compat(self):
        import pandas as pd
        df = pd.DataFrame({"日期": ["2026-01-01", "2026-01-02"], "close": [10.0, 11.0]})
        out = compute_forward_returns(normalize_kline(df), "2026-01-01")
        assert out["base_price"] == 10.0


class TestStorePicks:

    def test_insert_and_list(self, temp_store):
        assert db_store.insert_screen_picks([_pick(), _pick(stock_code="000001")]) == 2
        rows = db_store.list_screen_picks()
        assert len(rows) == 2
        assert rows[0]["stock_code"] == "000001"      # id 倒序

    def test_insert_empty(self, temp_store):
        """熊市跳过选股时没有候选：不写库、不报错"""
        assert db_store.insert_screen_picks([]) == 0
        assert db_store.list_screen_picks() == []

    def test_filters(self, temp_store):
        db_store.insert_screen_picks([_pick(source="screen"),
                                      _pick(stock_code="000001", source="nl_screen")])
        assert len(db_store.list_screen_picks(source="nl_screen")) == 1
        assert len(db_store.list_screen_picks(stock_code="600519")) == 1
        assert len(db_store.list_screen_picks(only_backfillable=True)) == 2

    def test_backfill_whitelist(self, temp_store):
        db_store.insert_screen_picks([_pick()])
        rid = db_store.list_screen_picks()[0]["id"]
        db_store.update_screen_pick_backfill(rid, {"ret_5d": 0.03, "evil": 1})
        row = db_store.list_screen_picks()[0]
        assert row["ret_5d"] == 0.03
        assert "evil" not in row

    def test_benchmark_returns_converts_percent(self, temp_store):
        """daily_features 存的是百分比，必须换算成小数再与 ret_5d 相减"""
        db_store.upsert_daily_feature("sh000001", "2026-09-01", {}, 3900.0,
                                      future_ret_5d=1.5, future_ret_20d=3.0)
        assert db_store.benchmark_returns("2026-09-01") == {
            "bench_ret_5d": 0.015, "bench_ret_20d": 0.03}

    def test_benchmark_missing_is_empty(self, temp_store):
        """缺基准就是缺，绝不能补 0 冒充"""
        assert db_store.benchmark_returns("2026-09-01") == {}
        assert db_store.benchmark_returns("") == {}

    def test_cleanup_keeps_picks_within_retention(self, temp_store):
        db_store.insert_screen_picks([_pick()])
        assert db_store.cleanup()["screen_picks"] == 0
        assert db_store.store_stats()["screen_picks"]["rows"] == 1


class TestStats:
    """超额收益与条件/因子统计（P2 的核心）"""

    def _seed(self, ret=0.05, bench=0.01, conditions=None, factors=None):
        db_store.insert_screen_picks([_pick(conditions=conditions or {},
                                            factors=factors or {})])
        rid = db_store.list_screen_picks()[0]["id"]
        db_store.update_screen_pick_backfill(rid, {
            "ret_5d": ret, "bench_ret_5d": bench, "backfill_status": "partial"})

    def test_stats_excess_and_win_rate(self, temp_store):
        self._seed(ret=0.05, bench=0.01)
        st_ = db_store.screen_pick_stats(recent_days=365)
        assert st_["samples"] == 1
        assert st_["avg_ret_5d"] == 0.05
        assert st_["avg_excess_5d"] == pytest.approx(0.04)   # 0.05 - 0.01
        assert st_["win_rate_5d"] == 1.0

    def test_missing_benchmark_excluded_from_excess(self, temp_store):
        """缺基准的样本不计入超额分母（宁可少样本，也不拿 0 顶替）"""
        db_store.insert_screen_picks([_pick()])
        rid = db_store.list_screen_picks()[0]["id"]
        db_store.update_screen_pick_backfill(rid, {"ret_5d": 0.05})
        st_ = db_store.screen_pick_stats(recent_days=365)
        assert st_["samples"] == 1
        assert st_["excess_samples"] == 0
        assert st_["avg_excess_5d"] is None

    def test_condition_stats_edge_vs_baseline(self, temp_store):
        # 命中"低估值"的三只都涨，另一只没命中且跌 → 该条件应有正向 edge
        for i, (ret, cond) in enumerate([
            (0.08, ["市盈率(PE) 小于 20倍：实际 15.3倍"]),
            (0.06, ["市盈率(PE) 小于 20倍：实际 18.0倍"]),
            (0.04, ["市盈率(PE) 小于 20倍：实际 19.5倍"]),
            (-0.02, []),
        ]):
            db_store.insert_screen_picks([
                _pick(stock_code=f"60052{i}",
                      conditions={"hits": cond} if cond else {})])
            rid = db_store.list_screen_picks(stock_code=f"60052{i}")[0]["id"]
            db_store.update_screen_pick_backfill(
                rid, {"ret_5d": ret, "bench_ret_5d": 0.01, "backfill_status": "partial"})

        cs = db_store.condition_stats(recent_days=365, min_samples=3)
        assert cs["samples"] == 4
        cond_row = next(c for c in cs["conditions"]
                        if c["condition"] == "市盈率(PE) 小于 20倍")
        assert cond_row["samples"] == 3
        assert cond_row["avg_ret_5d"] == pytest.approx(0.06)
        assert cond_row["edge_vs_base"] > 0        # 优于全样本基线

    def test_condition_below_min_samples_hidden(self, temp_store):
        for i in range(2):
            db_store.insert_screen_picks([
                _pick(stock_code=f"00000{i}",
                      conditions={"hits": ["主力资金净流入为正：实际 +1.2亿"]})])
            rid = db_store.list_screen_picks(stock_code=f"00000{i}")[0]["id"]
            db_store.update_screen_pick_backfill(
                rid, {"ret_5d": 0.02, "backfill_status": "partial"})
        cs = db_store.condition_stats(recent_days=365, min_samples=3)
        assert cs["conditions"] == []             # 小样本噪声 > 信号

    def test_factor_ic_and_missing_excluded(self, temp_store):
        """因子与收益完全正相关 → IC≈1；status 非 ok 的因子不参与"""
        for i, (score, ret) in enumerate([(10, 0.01), (20, 0.02), (30, 0.03),
                                          (40, 0.04), (50, 0.05)]):
            db_store.insert_screen_picks([
                _pick(stock_code=f"30000{i}",
                      factors={"主力资金": {"score": score, "max": 25, "status": "ok"},
                               "市盈率": {"score": score, "max": 20, "status": "unknown"}})])
            rid = db_store.list_screen_picks(stock_code=f"30000{i}")[0]["id"]
            db_store.update_screen_pick_backfill(
                rid, {"ret_5d": ret, "backfill_status": "partial"})

        fs = db_store.factor_stats(recent_days=365, min_samples=5)
        names = {f["factor"]: f for f in fs["factors"]}
        assert "主力资金" in names and "市盈率" not in names
        assert names["主力资金"]["ic_ret_5d"] == pytest.approx(1.0, abs=0.01)


class TestSaveScreenResult:

    def test_saves_candidates_with_rank_and_market(self, temp_store):
        class Sig:
            market_state = "range"
            total_score = 59.8
        result = {"skipped": False, "market_signal": Sig(),
                  "candidates": [
                      {"股票代码": "600519", "股票名称": "贵州茅台", "score": 88.0,
                       "策略": "主力资金", "score_coverage": 0.83,
                       "factors": {"主力资金": {"score": 22, "max": 25, "status": "ok"}},
                       "nl_report": {"pass": True,
                                     "hits": ["市盈率(PE) 小于 20倍：实际 15.3倍"],
                                     "misses": ["换手率 大于 3%：实际 1.2%"],
                                     "unverified": []}},
                      {"股票代码": "", "股票名称": "无代码"},      # 应被丢弃
                  ]}
        assert sh.save_screen_result(result, source="nl_screen",
                                     pick_date="2026-09-01") == 1
        row = db_store.list_screen_picks()[0]
        assert row["stock_code"] == "600519"
        assert row["rank_in_run"] == 1
        assert row["market_state"] == "range" and row["market_score"] == 59.8
        assert row["source"] == "nl_screen"

    def test_conditions_and_price_extracted(self, temp_store):
        result = {"skipped": False, "market_signal": None,
                  "candidates": [{"股票代码": "000001", "股票名称": "平安银行",
                                  "现价": "12.35元"}]}
        sh.save_screen_result(result, source="screen")
        # conditions/factors 不在列表查询里，直接读库校验
        conn = db_store.get_connection()
        row = conn.execute("SELECT conditions, price_at_pick FROM screen_picks").fetchone()
        conn.close()
        assert row["price_at_pick"] == 12.35       # 从"12.35元"解析
        assert row["conditions"] == "{}"           # 非 NL 选股没有条件明细

    def test_skipped_or_empty_is_noop(self, temp_store):
        assert sh.save_screen_result({"skipped": True, "candidates": []}) == 0
        assert sh.save_screen_result({"skipped": False, "candidates": []}) == 0
        assert sh.save_screen_result(None) == 0
        assert db_store.list_screen_picks() == []


class TestBackfill:

    def test_done_when_t20_available(self, temp_store):
        db_store.insert_screen_picks([_pick(pick_date=_aged_date())])
        stats = sh.backfill_screen_picks(price_fetcher=lambda c, s, e: _aged_kline(30))
        assert stats["done"] == 1 and stats["updated"] == 1
        row = db_store.list_screen_picks()[0]
        assert row["ret_5d"] == pytest.approx(0.05)
        assert row["backfill_status"] == "done"

    def test_partial_when_only_t5(self, temp_store):
        db_store.insert_screen_picks([_pick(pick_date=_aged_date())])
        stats = sh.backfill_screen_picks(price_fetcher=lambda c, s, e: _aged_kline(10))
        assert stats["partial"] == 1
        assert db_store.list_screen_picks()[0]["backfill_status"] == "partial"

    def test_benchmark_filled_from_daily_features(self, temp_store):
        date = _aged_date()
        db_store.upsert_daily_feature("sh000001", date, {}, 3900.0,
                                      future_ret_5d=2.0, future_ret_20d=4.0)
        db_store.insert_screen_picks([_pick(pick_date=date)])
        sh.backfill_screen_picks(price_fetcher=lambda c, s, e: _aged_kline(30))
        row = db_store.list_screen_picks()[0]
        assert row["bench_ret_5d"] == 0.02 and row["bench_ret_20d"] == 0.04

    def test_benchmark_falls_back_to_live_index(self, temp_store):
        """指数特征库没覆盖到这一天时，实时拉指数 K 线补基准。

        真实场景（实测）：daily_features 只回填到 8-28，8-25 之后不足 5 根 →
        没有 future_ret_5d。若不管，近期候选的超额收益会长期全缺，
        而超额是"选股有没有用"唯一可信的指标。
        """
        date = _aged_date()
        db_store.insert_screen_picks([_pick(pick_date=date)])

        def fake(c, s, e):
            # 指数每天涨 0.5：基准 100 → 第 5 根 102.5 → +2.5%
            return (_aged_kline(30, 100.0, 0.5) if c == "sh000001"
                    else _aged_kline(30))

        sh.backfill_screen_picks(price_fetcher=fake)
        row = db_store.list_screen_picks()[0]
        assert row["bench_ret_5d"] == pytest.approx(0.025)
        assert row["ret_5d"] == pytest.approx(0.05)      # 个股 +5%
        st_ = db_store.screen_pick_stats(recent_days=365)
        assert st_["avg_excess_5d"] == pytest.approx(0.025)   # 5% - 2.5%

    def test_overage_marked_failed(self, temp_store):
        db_store.insert_screen_picks([_pick(pick_date="2020-01-02")])
        stats = sh.backfill_screen_picks(price_fetcher=lambda c, s, e: _aged_kline(30))
        assert stats["failed"] == 1               # 超龄不再每日重试
        assert db_store.list_screen_picks(backfill_status="failed")

    def test_source_error_keeps_pending(self, temp_store):
        db_store.insert_screen_picks([_pick(pick_date=_aged_date())])
        def boom(c, s, e):
            raise RuntimeError("限频")
        stats = sh.backfill_screen_picks(price_fetcher=boom)
        assert stats["skipped"] == 1
        assert db_store.list_screen_picks()[0]["backfill_status"] is None

    def test_lock_skips_without_fetching(self, temp_store):
        """与分析回填不同 task 名：两者可并行，但各自不能被第二个进程重复跑"""
        calls = []
        db_store.acquire_task_lock("backfill_screen", "other-process")
        stats = sh.backfill_screen_picks(price_fetcher=lambda c, s, e: calls.append(c))
        assert stats["locked"] is True and calls == []
        db_store.release_task_lock("backfill_screen", "other-process")
        assert sh.backfill_screen_picks(
            price_fetcher=lambda c, s, e: _kline(3))["locked"] is False

    def test_benchmark_fetched_once_per_distinct_date(self, temp_store):
        """同一天的多只候选只能触发**一次**指数行情请求。

        回归动机（批次 16 预演实测）：20 条候选分散在 5 个选股日，逐条兜底
        把同一天的 sh000001 各拉了 5 遍——重复请求会白白消耗限频额度，
        盘中若撞上熔断会让本该成功的回填整批失败。
        """
        dates = [_aged_date(_AGED_DAYS + k) for k in (2, 1, 0)]   # 3 个选股日
        picks = [_pick(pick_date=d, stock_code=f"60000{i}")
                 for d in dates for i in range(4)]                # 每天 4 只，共 12 条
        db_store.insert_screen_picks(picks)

        index_calls = []

        def fake(c, s, e):
            if c == "sh000001":
                index_calls.append(s)
            return _aged_kline(30)

        # daily_features 里不预先写入任何基准 → 每条候选都必须走实时兜底
        sh.backfill_screen_picks(price_fetcher=fake)
        assert len(index_calls) == len(dates), (
            f"每个选股日只应请求一次指数，实际 {len(index_calls)} 次: {index_calls}")
        # 每条候选仍拿到了基准（缓存复用了同一份数据）
        rows = db_store.list_screen_picks()
        assert len(rows) == 12
        assert all(r["bench_ret_5d"] is not None for r in rows)

    def test_benchmark_batch_uses_daily_features_first(self, temp_store):
        """指数特征库里有基准时，一条候选都不该去拉实时行情"""
        d1 = _aged_date(_AGED_DAYS + 1)
        d2 = _aged_date(_AGED_DAYS)
        db_store.upsert_daily_feature("sh000001", d1, {}, 3900.0,
                                      future_ret_5d=2.0, future_ret_20d=4.0)
        db_store.upsert_daily_feature("sh000001", d2, {}, 3900.0,
                                      future_ret_5d=1.0, future_ret_20d=3.0)
        db_store.insert_screen_picks([
            _pick(pick_date=d1, stock_code="600001"),
            _pick(pick_date=d2, stock_code="600002")])

        fetched = []

        def fetcher(c, s, e):
            fetched.append(c)
            return _aged_kline(30)

        sh.backfill_screen_picks(price_fetcher=fetcher)
        assert "sh000001" not in fetched, "特征库已覆盖，不应实时拉指数"
        by_code = {r["stock_code"]: r for r in db_store.list_screen_picks()}
        assert by_code["600001"]["bench_ret_5d"] == pytest.approx(0.02)
        assert by_code["600002"]["bench_ret_5d"] == pytest.approx(0.01)

    def test_window_covers_aged_pending_not_today(self, temp_store):
        """回归动机（2026-10-28 实测）：`limit=200` 按 `id DESC` 取"最新 200 条"，
        而扩池后每天落库 200~440 行 ⇒ 待回填的**老**候选永远排在窗口外：
        最老待回填是 `id=43`（09-25），窗口下界已经到 `id=1781`，全表 `ret_20d`
        覆盖 **0/2074**，复盘天天报"候选 T+N 未回填"。

        本用例锁死两件事：
        ① 不够老的候选（T+5 还没走完）不参与回填——否则白拉行情、100% 空收益；
        ② 窗口优先覆盖老候选，而不是被当天新候选挤满。
        """
        old = _aged_date(_AGED_DAYS + 5)
        db_store.insert_screen_picks([_pick(pick_date=old, stock_code="600001")])
        # 模拟"当天刚扩池落库的一大批"：它们全部未到期，一条都不该被回填
        db_store.insert_screen_picks([
            _pick(pick_date=_aged_date(0), stock_code=f"6001{i:02d}")
            for i in range(20)])

        fetched = []
        stats = sh.backfill_screen_picks(
            limit=5,
            price_fetcher=lambda c, s, e: fetched.append(c) or _aged_kline(30))
        assert "600001" in fetched, "老候选没进回填窗口（取数方向又退回去了）"
        today_codes = [c for c in fetched if c.startswith("6001") and c != "600001"]
        assert today_codes == [], f"当天的 {len(today_codes)} 条新候选不该被回填"
        assert stats["updated"] == 1

    def test_end_to_end_stats_after_backfill(self, temp_store):
        """落库 → 回填 → 统计：闭环能跑出真实数字"""
        date = _aged_date()
        db_store.upsert_daily_feature("sh000001", date, {}, 3900.0, future_ret_5d=1.0)
        sh.save_screen_result(
            {"skipped": False, "market_signal": None,
             "candidates": [{"股票代码": "600519", "股票名称": "贵州茅台", "score": 80}]},
            source="screen", pick_date=date)
        sh.backfill_screen_picks(price_fetcher=lambda c, s, e: _aged_kline(30))
        st_ = db_store.screen_pick_stats(recent_days=365)
        assert st_["samples"] == 1
        assert st_["avg_excess_5d"] == pytest.approx(0.04)   # 5% - 1%


# ================================================================
# 相似度监控（2026-10-10）：把"每次都同几只"变成页面自己会报的警
# ================================================================

def _insert_batch(batch_time: str, codes: list):
    """写入一个"批次"：同 created_at 的前 5 名。"""
    import sqlite3
    import db.models as models
    conn = sqlite3.connect(models.DB_PATH)
    try:
        for i, c in enumerate(codes, 1):
            conn.execute(
                """INSERT INTO screen_picks
                   (pick_date, stock_code, stock_name, source, strategy, score,
                    score_coverage, rank_in_run, factors, conditions, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                ("2026-10-09", c, f"股{c}", "screen", "主力资金", 80.0, 1.0, i,
                 "{}", "{}", batch_time))
        conn.commit()
    finally:
        conn.close()


class TestRecentTopOverlap:

    def test_empty_input_returns_empty(self, temp_store):
        assert sh.recent_top_overlap([]) == {}

    def test_empty_db_returns_empty(self, temp_store):
        """库为空 → 返回 {}（提示信息读不到就算了，绝不抛异常）"""
        assert sh.recent_top_overlap(["600519", "000001"]) == {}

    def test_computes_overlap_and_alerts(self, temp_store):
        """最近一批 TOP5 与本次重合 4/5 → alert（这正是用户最初报的现象）"""
        _insert_batch("2026-10-09 15:49", ["600519", "000001", "600036", "601398",
                                           "600048"])
        r = sh.recent_top_overlap(["600519", "000001", "600036", "601398", "300750"],
                                  lookback=5, top_n=5)
        assert r["batch_count"] == 1
        assert r["max_overlap"] == 4
        assert r["alert"] is True
        assert r["threshold"] == 4
        assert r["details"][0]["time"] == "2026-10-09 15:49"

    def test_low_overlap_does_not_alert(self, temp_store):
        _insert_batch("2026-10-09 15:49", ["600519", "000001", "600036"])
        r = sh.recent_top_overlap(["300750", "002594", "601012"], top_n=5)
        assert r["max_overlap"] == 0 and r["alert"] is False

    def test_lookback_limits_batches(self, temp_store):
        for t in ("2026-10-07 15:49", "2026-10-08 15:49", "2026-10-09 15:49"):
            _insert_batch(t, ["600519"])
        r = sh.recent_top_overlap(["600519"], lookback=2, top_n=5)
        assert r["batch_count"] == 2, "lookback 必须限制批次数"

    def test_db_error_is_swallowed(self, monkeypatch):
        """查询失败 → 返回 {} 而不是抛异常（这只是提示信息，不能阻断选股）"""
        def boom(*a, **kw):
            raise RuntimeError("库炸了")
        monkeypatch.setattr(sh.store, "get_connection", boom)
        assert sh.recent_top_overlap(["600519"]) == {}


class TestSkipLatestBatch:
    """⚠️ 自比误报（2026-10-10 冒烟测试抓到）

    页面渲染的结果**已经落库**，"最新批次"就是本次自己 —— 不跳过就会得到
    "与最近 1 次选股高度重合 1/1"的**误报**（用户重开页面就被警告一次）。
    """

    def test_skip_latest_excludes_self(self, temp_store):
        _insert_batch("2026-10-09 15:49", ["600519", "000001"])
        codes = ["600519", "000001"]
        assert sh.recent_top_overlap(codes, top_n=5)["max_overlap"] == 2   # 不跳过 → 自比
        assert sh.recent_top_overlap(codes, top_n=5, skip_latest=True) == {}  # 跳过 → 无批次

    def test_skip_latest_keeps_older_batches(self, temp_store):
        _insert_batch("2026-10-08 15:49", ["600000", "600519"])
        _insert_batch("2026-10-09 15:49", ["600519", "000001"])   # 最新＝本次自己
        r = sh.recent_top_overlap(["600519", "000001"], top_n=5, skip_latest=True)
        assert r["batch_count"] == 1
        assert r["details"][0]["time"] == "2026-10-08 15:49"
        assert r["max_overlap"] == 1        # 与更早批次重合 1（600519）


# ================================================================
# 候选扩池（2026-10-22）
# ================================================================

def test_save_prefers_ranked_pool_over_candidates(temp_store):
    """⭐ 有 `ranked` 时必须落**完整排序**，不能只落最终名单。

    背景：`candidates` 是做过行业分散的 5~10 只（给用户看），`ranked` 是纯排序
    （给深度分析挑）。只用前者 ⇒ `screen_picks` 每天仍是 5~10 行 ⇒
    深度分析只能从个位数里挑 ⇒ 10-21 实测连续两日重合 56%。
    """
    result = {
        "market_signal": None,
        "candidates": [{"股票代码": "600001", "股票名称": "A", "score": 90}],
        "ranked": [{"股票代码": f"6000{i:02d}", "股票名称": f"票{i}", "score": 100 - i}
                   for i in range(1, 6)],
    }
    n = sh.save_screen_result(result, source="pytest")
    assert n == 5, f"应落 ranked 的 5 只，实际 {n}"

    conn = db_store.get_connection()
    try:
        codes = {r[0] for r in conn.execute(
            "SELECT stock_code FROM screen_picks WHERE source = 'pytest'")}
        ranks = [r[0] for r in conn.execute(
            "SELECT rank_in_run FROM screen_picks WHERE source = 'pytest' ORDER BY rank_in_run")]
    finally:
        conn.close()
    assert codes == {f"6000{i:02d}" for i in range(1, 6)}
    assert ranks == [1, 2, 3, 4, 5], "rank_in_run 要按完整排序编号"


def test_save_falls_back_to_candidates_without_ranked(temp_store):
    """旧 result（无 `ranked`）行为必须与改动前一致 —— 页面/自然语言选股走这条。"""
    result = {"market_signal": None,
              "candidates": [{"股票代码": "600001", "股票名称": "A", "score": 90}]}
    assert sh.save_screen_result(result, source="pytest") == 1
