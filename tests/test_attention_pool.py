"""关注池（data/watchlist.attention_pool）单元测试 —— **双轨版**（2026-10-27）

背景（Phase 4.1）：新闻入库时的个股关联原先只看自选股，全市场候选的
`related_codes` 恒为空；盯盘范围与新闻关联范围又各自写了一遍"自选股 ∪ 今日候选"。
现在两者共用一个 `attention_pool()`，本文件锁死它的语义：
来源顺序、去重保序、名称回填、代码校验、当日候选才算数、上限、以及**单来源失败隔离**。

2026-10-27 变更：自选股从 `config/watchlist.json` 迁到 SQLite `watchlist_items`，
并分成两轨（`user` 你的 / `ashen` 阿深的）。所以本文件相应改为：
- 造数据从"写 JSON"改成"写库"（`store.replace_watchlist`）；
- 来源名 `watchlist` → `watchlist_user` / `watchlist_ashen`；
- 新增 `TestDualTrack` 锁死**两轨并集**（这是最容易静默漏接线的一处）。
"""

import json
from datetime import datetime, timedelta

import pytest

import data.watchlist as w


@pytest.fixture
def pool_env(tmp_path, monkeypatch):
    """隔离候选文件、SQLite（含 positions 表）；自选股直接写 `watchlist_items`。"""
    picks_path = tmp_path / "latest.json"
    monkeypatch.setattr(w, "PICKS_PATH", str(picks_path))
    # 迁移源也要断开真实 config/watchlist.json，否则"首次迁移"用例会读到生产文件
    monkeypatch.setattr(w, "WATCHLIST_PATH", str(tmp_path / "absent.json"))

    from tests.conftest import use_template_db
    db_path = use_template_db(tmp_path, monkeypatch, "pool.db")
    import db.models as models          # 下面 add_position 要用它的 get_connection
    from db import store

    # ⚠️ 先打迁移标记：否则写入"空清单"的用例会触发首次迁移，被默认池灌回来。
    store.init_store()
    store.set_flag(w._MIGRATED_FLAG, "test")

    def write_watch(stocks, owner="user"):
        store.replace_watchlist(owner, stocks)

    def write_picks(picks, date=None):
        picks_path.write_text(json.dumps({
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "picks": picks}, ensure_ascii=False), encoding="utf-8")

    def add_position(code, name="", status="active"):
        conn = models.get_connection()
        try:
            conn.execute(
                "INSERT INTO positions (code, name, entry_date, entry_price, quantity, status)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (code, name, "2026-01-05", 10.0, 100, status))
            conn.commit()
        finally:
            conn.close()

    return {"watch": write_watch, "picks": write_picks, "position": add_position,
            "db": db_path, "store": store}


class TestComposition:
    def test_order_dedupe_and_sources(self, pool_env):
        pool_env["watch"]([{"code": "600519", "name": "贵州茅台"},
                           {"code": "000001", "name": "平安银行"}])
        pool_env["picks"]([{"股票代码": "000001", "股票名称": "平安银行"},
                           {"股票代码": "300750", "股票名称": "宁德时代"}])
        pool_env["position"]("600036", "招商银行")
        pool = w.attention_pool()
        assert [s["code"] for s in pool] == ["600519", "000001", "300750", "600036"]
        assert [s["source"] for s in pool] == \
            ["watchlist_user", "watchlist_user", "picks", "positions"]

    def test_sources_subset(self, pool_env):
        pool_env["watch"]([{"code": "600519", "name": "贵州茅台"}])
        pool_env["picks"]([{"股票代码": "300750", "股票名称": "宁德时代"}])
        pool_env["position"]("600036", "招商银行")
        assert [s["code"] for s in w.attention_pool(("watchlist_user", "picks"))] == \
            ["600519", "300750"]
        assert [s["code"] for s in w.attention_pool(("positions",))] == ["600036"]

    def test_name_filled_from_later_source(self, pool_env):
        """自选股常没维护名称：后出现的来源里有名称就补上，但来源仍算最先出现的那个"""
        pool_env["watch"]([{"code": "600519", "name": ""}])
        pool_env["picks"]([{"股票代码": "600519", "股票名称": "贵州茅台"}])
        entry = w.attention_pool()[0]
        assert entry["name"] == "贵州茅台"
        assert entry["source"] == "watchlist_user"

    def test_invalid_codes_skipped(self, pool_env):
        """候选文件是外部产物，脏码必须在这里被拦住（库里的脏码在写入层就滤掉了）"""
        pool_env["picks"]([{"股票代码": "600519", "股票名称": "贵州茅台"},
                           {"股票代码": "abc", "股票名称": ""},
                           {"股票代码": "12345", "股票名称": ""},
                           {"股票代码": "", "股票名称": ""},
                           {"股票代码": None, "股票名称": ""}])
        assert [s["code"] for s in w.attention_pool(("picks",))] == ["600519"]

    def test_unknown_source_logged_not_fatal(self, pool_env):
        pool_env["watch"]([{"code": "600519", "name": "贵州茅台"}])
        assert [s["code"] for s in w.attention_pool(("watchlist_user", "nope"))] == ["600519"]

    def test_limit(self, pool_env):
        pool_env["watch"]([{"code": f"60000{i}", "name": ""} for i in range(1, 6)])
        assert len(w.attention_pool(limit=3)) == 3

    def test_attention_codes_matches_pool(self, pool_env):
        pool_env["watch"]([{"code": "600519", "name": "贵州茅台"}])
        pool_env["position"]("600036", "招商银行")
        assert w.attention_codes() == [s["code"] for s in w.attention_pool()]


class TestDualTrack:
    """两轨自选（2026-10-27）。这一组是**防静默漏接线**的守卫。

    漏掉 `watchlist_ashen` 的后果不是报错，而是阿深自选的票永远进不了新闻关联
    ⇒ 舆情因子恒走 0.2 兜底，而看盘页看起来完全正常（它直读自选表，不走关注池）。
    """

    def test_default_sources_contain_both_tracks(self):
        """`POOL_SOURCES` 必须同时有两轨 —— 静态契约，比行为断言更早发现问题"""
        assert "watchlist_user" in w.POOL_SOURCES
        assert "watchlist_ashen" in w.POOL_SOURCES
        assert w.POOL_SOURCES.index("watchlist_user") < w.POOL_SOURCES.index("watchlist_ashen")

    def test_ashen_codes_enter_default_pool(self, pool_env):
        pool_env["watch"]([{"code": "300750", "name": "宁德时代"}], owner="ashen")
        pool = w.attention_pool()
        assert [s["code"] for s in pool] == ["300750"]
        assert pool[0]["source"] == "watchlist_ashen"

    def test_two_tracks_union_and_order(self, pool_env):
        pool_env["watch"]([{"code": "600519", "name": "贵州茅台"}], owner="user")
        pool_env["watch"]([{"code": "300750", "name": "宁德时代"}], owner="ashen")
        pool = w.attention_pool()
        assert [s["code"] for s in pool] == ["600519", "300750"]
        assert [s["source"] for s in pool] == ["watchlist_user", "watchlist_ashen"]

    def test_same_code_in_both_tracks_appears_once(self, pool_env):
        """同一只票同时被两套持有是**正常状态**（用户提名后大概率如此），
        在池子里只出现一次，来源取先出现的那轨"""
        pool_env["watch"]([{"code": "600519", "name": "贵州茅台"}], owner="user")
        pool_env["watch"]([{"code": "600519", "name": "贵州茅台"}], owner="ashen")
        pool = w.attention_pool()
        assert [s["code"] for s in pool] == ["600519"]
        assert pool[0]["source"] == "watchlist_user"

    def test_ashen_only_subset_does_not_leak_user(self, pool_env):
        pool_env["watch"]([{"code": "600519", "name": "贵州茅台"}], owner="user")
        pool_env["watch"]([{"code": "300750", "name": "宁德时代"}], owner="ashen")
        assert [s["code"] for s in w.attention_pool(("watchlist_ashen",))] == ["300750"]
        assert [s["code"] for s in w.attention_pool(("watchlist_user",))] == ["600519"]


class TestPicksFreshness:
    def test_today_picks_included(self, pool_env):
        pool_env["picks"]([{"股票代码": "300750", "股票名称": "宁德时代"}])
        assert [s["code"] for s in w.attention_pool(("picks",))] == ["300750"]

    def test_yesterday_picks_ignored(self, pool_env):
        """隔日候选不该继续算"今日关注"（否则池子越滚越大、早就过期还当热点）"""
        old = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        pool_env["picks"]([{"股票代码": "300750", "股票名称": "宁德时代"}], date=old)
        assert w.attention_pool(("picks",)) == []

    def test_missing_picks_file(self, pool_env):
        assert w.attention_pool(("picks",)) == []


class TestIsolation:
    def test_positions_failure_does_not_break_others(self, pool_env, monkeypatch):
        """持仓表读不到（例如库里没这张表）时，其它来源照常返回"""
        pool_env["watch"]([{"code": "600519", "name": "贵州茅台"}])
        pool_env["picks"]([{"股票代码": "300750", "股票名称": "宁德时代"}])

        def boom():
            raise RuntimeError("no such table: positions")

        monkeypatch.setitem(w._POOL_BUILDERS, "positions", boom)
        pool = w.attention_pool()
        assert [s["code"] for s in pool] == ["600519", "300750"]

    def test_watchlist_failure_does_not_break_others(self, pool_env, monkeypatch):
        pool_env["position"]("600036", "招商银行")

        def boom():
            raise RuntimeError("库读失败")

        monkeypatch.setitem(w._POOL_BUILDERS, "watchlist_user", boom)
        assert [s["code"] for s in w.attention_pool()] == ["600036"]

    def test_positions_read_without_loading_portfolio(self, pool_env, monkeypatch):
        """持仓只读 SQL：绝不能走 core.portfolio（那条路会逐只取现价）"""
        pool_env["position"]("600036", "招商银行")

        def boom(*a, **k):
            raise AssertionError("不该调用 portfolio（会拉行情）")

        monkeypatch.setattr("core.portfolio.get_portfolio_manager", boom, raising=False)
        assert [s["code"] for s in w.attention_pool(("positions",))] == ["600036"]

    def test_closed_position_excluded(self, pool_env):
        pool_env["position"]("600036", "招商银行", status="closed")
        assert w.attention_pool(("positions",)) == []

    def test_all_sources_empty(self, pool_env):
        pool_env["watch"]([])
        assert w.attention_pool() == []


class TestFirstRunMigration:
    """首次启动的迁移语义（JSON → 库）。这一段替代旧的
    `test_missing_watchlist_file_falls_back_to_defaults`，语义已随存储层改变。"""

    def test_first_run_seeds_default_pool(self, tmp_path, monkeypatch):
        """全新库 + 无旧 JSON ⇒ 用默认池初始化，并写迁移标记"""
        monkeypatch.setattr(w, "WATCHLIST_PATH", str(tmp_path / "absent.json"))
        from tests.conftest import use_template_db
        use_template_db(tmp_path, monkeypatch, "fresh.db")
        rows = w.load_watchlist("user")
        assert len(rows) == len(w.DEFAULT_WATCHLIST)
        assert rows[0]["code"] == w.DEFAULT_WATCHLIST[0]

    def test_first_run_imports_legacy_json(self, tmp_path, monkeypatch):
        """有旧 JSON ⇒ 原样导入（保留名称与备注），且**不再被当作数据源**。

        ⚠️ 迁移刻意**不改名/不删除**源文件：只要有一个调用方忘了 patch
        `WATCHLIST_PATH`，归档动作就会落到真实的 `config/watchlist.json` 上
        （实测发生过两次 —— 数据迁走了、源文件也没了，连回滚余地都没有）。
        所以这里锁的是"文件留着但已失效"，而不是"文件被删掉"。
        """
        legacy = tmp_path / "watchlist.json"
        legacy.write_text(json.dumps({"stocks": [
            {"code": "600519", "name": "贵州茅台", "note": "旧备注"}]},
            ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(w, "WATCHLIST_PATH", str(legacy))
        from tests.conftest import use_template_db
        use_template_db(tmp_path, monkeypatch, "legacy.db")
        rows = w.load_watchlist("user")
        assert [(r["code"], r["name"], r["note"]) for r in rows] == \
            [("600519", "贵州茅台", "旧备注")]
        assert all(r["source"] == "migrated" for r in rows)
        assert legacy.exists(), "迁移不得改动源文件"

        # 文件已失效：改它的内容不影响读出来的结果（标记挡住二次导入）
        legacy.write_text(json.dumps({"stocks": [{"code": "000002", "name": "万科A"}]},
                                     ensure_ascii=False), encoding="utf-8")
        assert [r["code"] for r in w.load_watchlist("user")] == ["600519"]

    def test_emptied_list_is_not_reseeded(self, pool_env):
        """用户把自选删光后重启，绝不能被默认池灌回来（迁移标记的唯一理由）"""
        pool_env["watch"]([])
        assert w.load_watchlist("user") == []
        assert w.load_watchlist("user") == []
