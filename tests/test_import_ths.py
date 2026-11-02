# -*- coding: utf-8 -*-
"""同花顺导入器守护：解析（纯函数）+ 写库语义（增量、fail-closed、不覆盖）。

写库用例全部走 `use_template_db` 隔离库；并把 watchlist 迁移 flag 预先置上，
否则 `_ensure_migrated` 会拿**真实** config/watchlist.json 的 15 只默认股
灌进测试库（2026-10-30 实测踩到）。
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from scripts.import_from_ths import (commit_positions, commit_watchlist,
                                     detect_mode, parse_ledger,
                                     parse_snapshot, parse_watchlist,
                                     read_table)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    path = use_template_db(tmp_path, monkeypatch, "ths_import.db")
    from db import store
    store.set_flag("watchlist_migrated_v1", "test")  # 见模块 docstring
    return path


# ----------------------------------------------------------------
# 自选股解析
# ----------------------------------------------------------------
_WL_CSV = "股票代码,股票名称\n000001,平安银行\n600519,贵州茅台\n688981,中芯国际\n920819,万达轴承\n900901,境内B股\n110022,易方达消费\n000001,平安银行\n"


def _df(text: str, sep: str = ",") -> pd.DataFrame:
    from io import StringIO
    return pd.read_csv(StringIO(text), sep=sep, dtype=str)


class TestWatchlistParse:
    def test_csv_basic_filter_and_dedupe(self):
        rows, dropped = parse_watchlist(_df(_WL_CSV))
        codes = [r["code"] for r in rows]
        assert codes == ["000001", "600519", "688981", "920819"]  # B股/基金过滤+去重
        assert set(dropped) == {"900901", "110022"}
        assert rows[0]["name"] == "平安银行"

    def test_headerless_txt(self, tmp_path):
        p = tmp_path / "zxg.txt"
        p.write_text("000001\n600519 贵州茅台\n300750\n", encoding="utf-8")
        rows, dropped = parse_watchlist(read_table(p))
        assert [r["code"] for r in rows] == ["000001", "600519", "300750"]
        assert rows[1]["name"] == "贵州茅台"
        assert dropped == []

    def test_gbk_encoding(self, tmp_path):
        p = tmp_path / "导出.csv"
        p.write_bytes(_WL_CSV.encode("gb18030"))
        rows, _ = parse_watchlist(read_table(p))
        assert len(rows) == 4

    def test_excel_numeric_code(self):
        df = pd.DataFrame({"股票代码": ["1", "600519"], "股票名称": ["浦发银行", "贵州茅台"]})
        df["股票代码"] = df["股票代码"].astype(float)  # Excel 把 600000 读成数字
        rows, _ = parse_watchlist(df.astype(str))
        assert [r["code"] for r in rows] == ["600519"]  # "1.0" 不是 6 位 ⇒ 丢弃


class TestWatchlistCommit:
    def test_dry_run_writes_nothing(self, db):
        res = commit_watchlist([{"code": "000001", "name": "平安银行"}],
                               owner="user")
        assert res["added"] == ["000001"]
        from db import store
        assert store.count_watchlist("user") == 1
        item = store.list_watchlist("user")[0]
        assert item["source"] == "ths_import"

    def test_incremental_never_replaces(self, db):
        commit_watchlist([{"code": "000001", "name": "平安银行"}], owner="user")
        res = commit_watchlist([{"code": "600519", "name": "贵州茅台"}], owner="user")
        assert res["added"] == ["600519"] and res["skipped"] == []
        from db import store
        codes = [s["code"] for s in store.list_watchlist("user")]
        assert codes == ["000001", "600519"]          # 第一只没被清掉

    def test_skip_existing(self, db):
        commit_watchlist([{"code": "000001", "name": "平安银行"}], owner="user")
        res = commit_watchlist([{"code": "000001", "name": "平安银行"}], owner="user")
        assert res["added"] == [] and res["skipped"] == ["000001"]

    def test_owner_ashen(self, db):
        res = commit_watchlist([{"code": "000001", "name": ""}], owner="ashen")
        assert res["owner"] == "ashen"
        from db import store
        assert store.count_watchlist("ashen") == 1


# ----------------------------------------------------------------
# 持仓：detect
# ----------------------------------------------------------------
_SNAP_CSV = ("证券代码,证券名称,股票余额,可用余额,参考成本价,市价,市值,浮动盈亏,盈亏比例(%)\n"
             "600028,中国石化,2000,2000,6.10,6.35,12700,500,4.10\n"
             "000858,五粮液,300,300,128.50,130.00,39000,450,1.17\n"
             "510300,沪深300ETF,10000,10000,3.95,4.10,41000,1500,3.80\n"
             "601318,中国平安,0,0,0.00,0.00,0,0,0.00\n")
_LEDGER_1 = ("成交日期,证券代码,证券名称,买卖方向,成交价格,成交数量,成交金额,佣金,印花税,过户费,发生金额\n"
             "2026-07-03,600028,中国石化,买入,6.00,1000,6000,3,0,0.1,-6003.1\n"
             "2026-08-10,600028,中国石化,买入,6.20,1000,6200,3,0,0.1,-6203.1\n"
             "2026-09-01,000858,五粮液,买入,130.00,300,39000,12,0,0.3,-39012.3\n")
_LEDGER_2 = ("成交日期,证券代码,证券名称,买卖方向,成交价格,成交数量,成交金额,佣金,印花税,过户费,发生金额\n"
             "2026-09-15,600028,中国石化,卖出,6.40,500,3200,0,3.2,0.05,3196.75\n")


class TestDetect:
    def test_snapshot(self):
        assert detect_mode(_df(_SNAP_CSV)) == "snapshot"

    def test_ledger(self):
        assert detect_mode(_df(_LEDGER_1)) == "ledger"

    def test_statement_rejected(self):
        df = _df("发生日期,证券代码,证券名称,业务名称,发生金额\n"
                 "2026-09-01,600028,中国石化,买入,-6000\n")
        assert detect_mode(df) is None  # 对账单只有金额没有数量 ⇒ 不可重建


# ----------------------------------------------------------------
# 持仓：snapshot 解析
# ----------------------------------------------------------------
class TestSnapshotParse:
    def test_basic(self):
        rows, dropped = parse_snapshot(_df(_SNAP_CSV))
        by = {r["code"]: r for r in rows}
        assert set(by) == {"600028", "000858"}            # ETF 过滤、零余额跳过
        assert by["600028"]["quantity"] == 2000
        assert by["600028"]["entry_price"] == 6.10
        assert by["000858"]["entry_date"] is None         # 快照无日期，由写库补
        assert dropped == ["510300"]

    def test_thousands_separator(self):
        text = ("证券代码,证券名称,股票余额,参考成本价\n"
                "600028,中国石化,\"2,000\",6.10\n")
        rows, _ = parse_snapshot(_df(text))
        assert rows[0]["quantity"] == 2000


# ----------------------------------------------------------------
# 持仓：交割单重建
# ----------------------------------------------------------------
class TestLedgerRebuild:
    def test_moving_average_cost_across_files(self):
        rows, _ = parse_ledger([_df(_LEDGER_1), _df(_LEDGER_2)])
        by = {r["code"]: r for r in rows}
        p = by["600028"]
        assert p["quantity"] == 1500                      # 2000 买 − 500 卖
        assert p["entry_price"] == pytest.approx(6.10, abs=1e-4)  # 移动平均
        assert p["entry_date"] == "2026-07-03"            # 首次买入日
        assert p["warnings"] == []
        assert by["000858"]["entry_date"] == "2026-09-01"

    def test_sell_without_holding_warns_but_imports(self):
        text = ("成交日期,证券代码,证券名称,买卖方向,成交价格,成交数量,成交金额\n"
                "2026-09-15,600028,中国石化,卖出,6.40,500,3200\n"
                "2026-10-10,600028,中国石化,买入,6.00,1000,6000\n")
        rows, _ = parse_ledger([_df(text)])
        assert len(rows) == 1
        assert "区间内无持仓" in rows[0]["warnings"][0]   # 区间不完整的信号必须可见

    def test_full_close_excluded(self):
        text = ("成交日期,证券代码,证券名称,买卖方向,成交价格,成交数量,成交金额\n"
                "2026-09-01,600028,中国石化,买入,6.00,1000,6000\n"
                "2026-10-01,600028,中国石化,卖出,6.40,1000,6400\n")
        rows, _ = parse_ledger([_df(text)])
        assert rows == []                                 # 已清仓 ⇒ 无持仓

    def test_unknown_direction_skipped_with_warning(self):
        text = ("成交日期,证券代码,证券名称,买卖方向,成交价格,成交数量,成交金额\n"
                "2026-09-01,600028,中国石化,买入,6.00,1000,6000\n"
                "2026-09-20,600028,中国石化,红股入账,0,100,0\n")
        rows, _ = parse_ledger([_df(text)])
        assert rows[0]["quantity"] == 1000
        assert any("红股入账" in w for w in rows[0]["warnings"])


# ----------------------------------------------------------------
# 持仓：写库
# ----------------------------------------------------------------
def _active_codes(db_path: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT code FROM positions WHERE status='active'").fetchall()
    conn.close()
    return {r[0] for r in rows}


class TestPositionsCommit:
    def test_snapshot_commit_and_skip_existing(self, db):
        rows, _ = parse_snapshot(_df(_SNAP_CSV))
        res = commit_positions(rows)
        assert set(res["added"]) == {"600028", "000858"}
        assert _active_codes(db) == {"600028", "000858"}
        # 重复导入 ⇒ 已有活跃持仓跳过，绝不出第二条
        res2 = commit_positions(rows)
        assert res2["added"] == [] and set(res2["skipped_existing"]) == {"600028", "000858"}
        conn = sqlite3.connect(db)
        n = conn.execute("SELECT COUNT(*) FROM positions WHERE status='active'").fetchone()[0]
        conn.close()
        assert n == 2

    def test_entry_date_defaults_to_today_for_snapshot(self, db, monkeypatch):
        from datetime import date
        rows, _ = parse_snapshot(_df(_SNAP_CSV))
        commit_positions(rows[:1])
        conn = sqlite3.connect(db)
        r = conn.execute("SELECT entry_date, stop_loss FROM positions WHERE code='600028'").fetchone()
        conn.close()
        assert r[0] == date.today().isoformat()
        assert r[1] is None                                # 止损留空，监控安全跳过

    def test_ledger_commit_writes_import_trade_log(self, db):
        rows, _ = parse_ledger([_df(_LEDGER_1), _df(_LEDGER_2)])
        commit_positions(rows)
        conn = sqlite3.connect(db)
        r = conn.execute("SELECT entry_date, entry_price, quantity FROM positions "
                         "WHERE code='600028' AND status='active'").fetchone()
        log = conn.execute("SELECT direction, signal_source FROM trade_logs "
                           "WHERE code='600028'").fetchone()
        conn.close()
        assert r == ("2026-07-03", pytest.approx(6.10, abs=1e-4), 1500)
        assert log == ("买入", "外部导入")

    def test_no_network_anywhere(self, db, monkeypatch):
        """导入全程零联网（名称缺失就留空，不外呼）。"""
        import socket
        monkeypatch.setattr(socket.socket, "connect",
                            lambda *a, **k: pytest.fail("导入不应联网"))
        rows, _ = parse_snapshot(_df(_SNAP_CSV))
        commit_positions(rows)
        commit_watchlist([{"code": "000001", "name": ""}])
