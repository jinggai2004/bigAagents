# -*- coding: utf-8 -*-
"""采集覆盖率落库（P2-7，2026-10-17）：`data_coverage` 表的语义边界。

背景：`market_snapshot` 只能回答"落了哪些股票"，**回答不了"本该有多少只"**。
实测 10-16 落了 2991 只、全市场 ~5459 只 ⇒ 覆盖率 **54.8%**，而页面/日志只显示
"2991 只" —— 54.8% 被看成了 100%。本文件锁住：

  1. **"未知"不得写成 0**：`total` 缺失 ⇒ `ratio` 是 NULL（`0` 会把"不知道"
     伪装成"覆盖率为零"，正是本项目反复踩的坑）；
  2. **0 行不得记 ok**（"0 只记完成"是本项目踩过的真实事故）；
  3. 幂等（同 data_date+dataset 覆盖，不重复堆积）；
  4. 采集侧接线：快照成功 ⇒ 覆盖率落库；**覆盖率写失败不得连累快照**；
  5. 健康页能看到覆盖率（包括"不可评"这一列**如实显示**）。
"""
import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import use_template_db

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
PAGE = os.path.join(_ROOT, "ui", "pages", "7_⚙️_系统配置.py")

DAY = "2026-10-16"


def _raise_offline(name: str):
    """生成一个"离线即抛"的替身 —— 用来证明某条联网分支**确实被绕过**。

    比 `lambda: None` 更严格：若被测代码没有正确降级（比如把 None 当数据用），
    抛异常会立刻暴露，而不是静默走出一条"看似通过"的路径。
    """
    def _boom(*args, **kwargs):
        raise RuntimeError(f"离线夹具：本用例禁用 {name}（真实联网 = 测试污染）")
    return _boom


# ================================================================
# 1. store 层语义
# ================================================================

class TestSaveDataCoverage:
    @pytest.fixture(autouse=True)
    def db(self, tmp_path, monkeypatch):
        use_template_db(tmp_path, monkeypatch, name="cov.db")
        return None

    def _load(self):
        from db import store
        return store.load_data_coverage(limit=50)

    def test_roundtrip_and_ratio(self):
        from db import store
        r = store.save_data_coverage(DAY, "market_snapshot", got=2991,
                                     total=5459, status="ok", source="sina_sector")
        assert r == pytest.approx(2991 / 5459, abs=1e-4)
        rows = self._load()
        assert len(rows) == 1
        row = rows[0]
        assert row["data_date"] == DAY and row["dataset"] == "market_snapshot"
        assert row["got"] == 2991 and row["total"] == 5459
        assert row["ratio"] == pytest.approx(2991 / 5459, abs=1e-4)
        assert row["status"] == "ok"

    def test_unknown_total_is_null_not_zero(self):
        """⭐⭐ 最关键的一条：来源不给全市场数 ⇒ ratio=None，**不是 0**。"""
        from db import store
        r = store.save_data_coverage(DAY, "fund_flow_stock", got=5210, total=None,
                                     status="ok", source="ths")
        assert r is None
        row = self._load()[0]
        assert row["total"] is None and row["ratio"] is None, \
            "未知 ≠ 0：0 会把「不知道」伪装成「覆盖率为零」"

    def test_zero_or_negative_total_treated_as_unknown(self):
        from db import store
        store.save_data_coverage(DAY, "x", got=10, total=0, status="ok")
        store.save_data_coverage(DAY, "y", got=10, total=-3, status="ok")
        by = {r["dataset"]: r for r in self._load()}
        assert by["x"]["ratio"] is None and by["y"]["ratio"] is None

    def test_zero_rows_with_ok_status_is_coerced_to_failed(self):
        """⭐ "0 只记完成"是本项目踩过的真实事故 ⇒ 这里必须自相矛盾检测。"""
        from db import store
        store.save_data_coverage(DAY, "fund_flow_stock", got=0, total=None,
                                  status="ok", source="ths")
        row = self._load()[0]
        assert row["status"] == "failed"
        assert "0 行不得记 ok" in (row["note"] or "")

    def test_unknown_status_is_coerced_not_silently_accepted(self):
        from db import store
        store.save_data_coverage(DAY, "x", got=10, total=20, status="weird")
        row = self._load()[0]
        assert row["status"] == "degraded"
        assert "weird" in (row["note"] or "")

    def test_upsert_same_key_overwrites_no_duplication(self):
        from db import store
        store.save_data_coverage(DAY, "market_snapshot", got=100, total=200, status="ok")
        store.save_data_coverage(DAY, "market_snapshot", got=2991, total=5459,
                                 status="partial", note="半路挂了")
        rows = [r for r in self._load() if r["dataset"] == "market_snapshot"]
        assert len(rows) == 1, "同 (data_date, dataset) 必须覆盖，不许堆积"
        assert rows[0]["got"] == 2991 and rows[0]["status"] == "partial"

    def test_missing_key_raises(self):
        from db import store
        with pytest.raises(ValueError):
            store.save_data_coverage("", "x", got=1)
        with pytest.raises(ValueError):
            store.save_data_coverage(DAY, "", got=1)
        with pytest.raises(ValueError):
            store.save_data_coverage(DAY, "x", got=-1)

    def test_dataset_filter_and_limit(self):
        from db import store
        for i in range(5):
            store.save_data_coverage(f"2026-10-{10+i}", "a", got=i, total=10)
            store.save_data_coverage(f"2026-10-{10+i}", "b", got=i, total=10)
        only_a = store.load_data_coverage(dataset="a", limit=3)
        assert len(only_a) == 3 and {r["dataset"] for r in only_a} == {"a"}
        assert only_a[0]["data_date"] >= only_a[-1]["data_date"]   # 倒序

    def test_summary_splits_low_from_missing(self):
        """⭐ `low_ratio_days` 与 `missing_days` 必须分开 ——
        前者是"确实低"，后者是"不知道"；混在一起会把"没数据"读成"质量差"。"""
        from db import store
        store.save_data_coverage("2026-10-14", "market_snapshot", got=100, total=100)
        store.save_data_coverage("2026-10-15", "market_snapshot", got=3000, total=5459)
        store.save_data_coverage("2026-10-16", "market_snapshot", got=2991, total=None)
        s = store.data_coverage_summary("market_snapshot")
        assert s["days"] == 3
        assert s["low_ratio_days"] == 1            # 只有 10-15 的 55%
        assert s["missing_days"] == 1              # 10-16 是"不知道"，不算低
        assert s["latest"]["data_date"] == "2026-10-16"


# ================================================================
# 2. 采集侧接线
# ================================================================

class TestSnapshotWiring:
    @pytest.fixture(autouse=True)
    def db(self, tmp_path, monkeypatch):
        use_template_db(tmp_path, monkeypatch, name="covsnap.db")
        return None

    def _fake_snapshot(self, monkeypatch, coverage, total, degraded, tushare=None):
        import pandas as pd
        df = pd.DataFrame([{"code": "600028", "name": "中国石化", "industry": "石油行业",
                            "board": "主板沪市", "price": 5.53, "change_pct": 0.5,
                            "pe": 10.0, "pb": 1.0, "market_cap": 6000.0,
                            "float_cap": 5000.0, "turnover": 0.5, "amount": 1.0}])
        df.attrs.update({"coverage": coverage, "total_market": total,
                         "degraded": degraded, "caveat": ""})
        monkeypatch.setattr("data.sina_source.get_sector_snapshot", lambda: df)
        monkeypatch.setattr("db.store.save_market_snapshot",
                            lambda d, rows, source="": len(rows))
        # ⚠️⚠️ tushare A 档（2026-10-17）**优先**于新浪：不挡它就会真实联网，
        #   实测拿到 5567 只真数据 ⇒ `got == 1` 永假（2026-10-17 审计的红用例根因，
        #   与"假离线"同类事故第 4 次）。默认让它抛异常 ⇒ 用例稳定走新浪分支。
        if tushare is None:
            monkeypatch.setattr(
                "data.tushare_source.build_market_snapshot_rows",
                _raise_offline("build_market_snapshot_rows"))
        else:
            monkeypatch.setattr("data.tushare_source.build_market_snapshot_rows", tushare)

    def test_coverage_is_persisted_with_degraded_status(self, monkeypatch):
        self._fake_snapshot(monkeypatch, coverage=0.5, total=5459,
                            degraded=["code_list_unavailable"])
        from core import scheduler
        out = scheduler.save_daily_market_snapshot("2026-10-16")
        assert out["ok"] is True
        from db import store
        rows = store.load_data_coverage(dataset="market_snapshot")
        assert len(rows) == 1
        r = rows[0]
        assert r["got"] == 1 and r["total"] == 5459
        assert r["status"] == "degraded"                    # 有降级 ⇒ degraded，不是 ok
        assert "code_list_unavailable" in (r["note"] or "")
        assert r["source"] == "sina_sector", "tushare 被禁用 ⇒ 降级路径必须可查"
        assert out["total_market"] == 5459

    def test_tushare_branch_takes_priority(self, monkeypatch):
        """⭐ A 档（10-17）：tushare 成功时**优先**落库，且不许再打新浪。

        这条补的是上面那条用例的反面 —— 之前只测"新浪路径"，于是"tushare 分支
        接线"从未被验证（验了函数 ≠ 验了接线）。
        """
        ts_rows = [{"stock_code": "600028", "stock_name": "中国石化"}]
        ts_attrs = {"coverage": 0.99, "total_market": 5567,
                    "degraded": [], "caveat": ""}

        def _fake_ts(d):
            assert d == "2026-10-16"
            return list(ts_rows), dict(ts_attrs)

        self._fake_snapshot(monkeypatch, coverage=0.5, total=5459,
                            degraded=[], tushare=_fake_ts)
        sina_calls = []
        monkeypatch.setattr("data.sina_source.get_sector_snapshot",
                            lambda: sina_calls.append(1) or None)
        from core import scheduler
        out = scheduler.save_daily_market_snapshot("2026-10-16")
        assert sina_calls == [], "tushare 成功时绝不允许再打新浪（浪费一次全市场请求）"
        assert out["ok"] is True and out["total_market"] == 5567
        assert out["degraded"] == []
        from db import store
        r = store.load_data_coverage(dataset="market_snapshot")[0]
        assert r["source"] == "tushare" and r["status"] == "ok"
        assert r["got"] == 1 and r["total"] == 5567

    def test_partial_coverage_below_90pct(self, monkeypatch):
        self._fake_snapshot(monkeypatch, coverage=0.548, total=5459, degraded=[])
        from core import scheduler
        scheduler.save_daily_market_snapshot("2026-10-16")
        from db import store
        assert store.load_data_coverage()[0]["status"] == "partial"

    def test_coverage_write_failure_does_not_break_snapshot(self, monkeypatch):
        """⭐ 结构性守护：新加的落库**不许**把已经写成功的快照拖死。"""
        self._fake_snapshot(monkeypatch, coverage=0.5, total=5459, degraded=[])
        def _boom(*a, **k):
            raise RuntimeError("coverage 表炸了")
        monkeypatch.setattr("db.store.save_data_coverage", _boom)
        from core import scheduler
        out = scheduler.save_daily_market_snapshot("2026-10-16")
        assert out["ok"] is True and out["rows"] == 1, "快照必须照常成功"


# ================================================================
# 3. 健康页展示
# ================================================================

class TestHealthPageShowsCoverage:
    @pytest.fixture(autouse=True)
    def page_env(self, tmp_path, monkeypatch):
        import streamlit as st
        use_template_db(tmp_path, monkeypatch, name="cov_ui.db")
        monkeypatch.setenv("BIGA_UI_POLL_SECONDS", "3600")
        monkeypatch.setenv("BIGA_UI_OFFLINE", "1")
        st.cache_data.clear()
        yield
        st.cache_data.clear()

    @staticmethod
    def _stub_health(monkeypatch):
        from core import scheduler_health as H
        monkeypatch.setattr(
            H, "check_health",
            lambda: {"checked_at": "2026-10-17 10:00:00",
                     "scheduler": {"running": True, "reason": ""},
                     "warnings": [], "data": {}})

    def _open(self, monkeypatch, section=None):
        self._stub_health(monkeypatch)
        at = AppTest.from_file(PAGE, default_timeout=120)
        at.run()
        if section is not None:
            radio = [r for r in at.sidebar.radio]
            assert radio, "页 7 应有板块导航（P1-3）"
            radio[0].set_value(section).run()
        return at

    def _texts(self, at):
        blob = []
        for attr in ("markdown", "caption", "info", "warning", "error"):
            blob += [str(x.value) for x in getattr(at, attr) or []]
        return "\n".join(blob)

    def test_shows_table_with_real_coverage(self, monkeypatch):
        from db import store
        store.save_data_coverage("2026-10-16", "market_snapshot", got=2991,
                                 total=5459, status="partial", source="sina_sector",
                                 note="快照覆盖不完整")
        at = self._open(monkeypatch, section="🩺 运行健康")
        assert not at.exception, [str(e.value) for e in at.exception]
        text = self._texts(at)
        assert "采集覆盖率" in text
        frames = [str(f.value) for t in (at.dataframe or [])
                  for f in [t] if hasattr(t.value, "to_csv")]
        csv = "\n".join(str(t.value.to_csv(index=False))
                        for t in (at.dataframe or []) if hasattr(t.value, "to_csv"))
        # ⚠️ 页面用千分位格式化（"2,991"）⇒ 断言要按显示口径写，不是裸数字
        assert "2026-10-16" in csv and "2,991" in csv and "5,459" in csv, csv[:400]
        assert "54.8%" in csv, csv[:400]

    def test_unknown_ratio_is_shown_as_na_not_zero(self, monkeypatch):
        """⭐ 页面必须写「不可评」而不是 0% —— 与铁律"未知 ≠ 0"同一口径。"""
        from db import store
        store.save_data_coverage("2026-10-16", "fund_flow_stock", got=5210,
                                 total=None, status="ok", source="ths")
        at = self._open(monkeypatch, section="🩺 运行健康")
        assert not at.exception, [str(e.value) for e in at.exception]
        csv = "\n".join(str(t.value.to_csv(index=False))
                        for t in (at.dataframe or []) if hasattr(t.value, "to_csv"))
        assert "不可评" in csv and "未提供" in csv, csv[:400]
        assert "0.0%" not in csv and ",0%," not in csv, csv[:400]

    def test_empty_state_is_explicit(self, monkeypatch):
        at = self._open(monkeypatch, section="🩺 运行健康")
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "尚无覆盖率记录" in self._texts(at)
