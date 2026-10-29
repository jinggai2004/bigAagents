"""LLM 成本核算（core/llm_cost）与金额落库、预算闸门、跨进程观点缓存。

为什么这些用例值得写
────────────────────
成本这个模块最容易"看起来对"：改一个单价、错一个档位边界、把失败调用算成钱，
**都不会报错**，只是报表上的数字静默偏了。所以这里守的是「算式」而不是「能跑」：

- 峰谷边界（15 时 vs 20 时，同一 token 差 2 倍）
- qwen 阶梯边界（32768 / 32769，一跳 3 倍）
- 未知模型**必须保守计**（按最贵的在用模型），绝不能算成 0
- 失败调用不计费（拿不到 usage，硬算会得出假金额）
- 金额 NULL ≠ 0（NULL 要回退估算，0 会被读成"免费"）

另一组守的是"钱花在哪"：`opinion_cache` 跨进程共享。
2026-10-23 实测同一份市场观点被买了 28 次（¥0.62，占当日 41%），
根因是缓存只在模块级 dict 里 ⇒ 三个常驻进程各买一份。
"""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from tests.conftest import use_template_db


def _at(hour: int, day: int = 23) -> datetime:
    return datetime(2026, 10, day, hour, 0, 0)


# ================================================================
# 单价表
# ================================================================

class TestPriceOf:
    def test_deepseek_flash_peak_vs_idle(self):
        """同一 token，高峰是空闲的 2 倍——这是"挪到 18:10 省一半"的账面依据。"""
        from core import llm_cost as C
        table = C.load_prices(force=True)
        assert C.price_of("deepseek-v4-flash", 1000, _at(15), table)[:2] == (3.0, 9.0)
        assert C.price_of("deepseek-v4-flash", 1000, _at(20), table)[:2] == (1.5, 4.5)

    def test_peak_window_boundaries(self):
        """高峰 = 9-11 与 14-17 时；12/13 与 18 之后算空闲。"""
        from core import llm_cost as C
        table = C.load_prices(force=True)
        assert C.price_of("deepseek-v4-flash", 1, _at(9), table)[2] == "高峰"
        assert C.price_of("deepseek-v4-flash", 1, _at(11), table)[2] == "高峰"
        assert C.price_of("deepseek-v4-flash", 1, _at(12), table)[2] == "空闲"
        assert C.price_of("deepseek-v4-flash", 1, _at(13), table)[2] == "空闲"
        assert C.price_of("deepseek-v4-flash", 1, _at(14), table)[2] == "高峰"
        assert C.price_of("deepseek-v4-flash", 1, _at(18), table)[2] == "空闲"

    def test_pro_is_three_times_flash(self):
        from core import llm_cost as C
        table = C.load_prices(force=True)
        assert C.price_of("deepseek-v4-pro", 1, _at(15), table)[:2] == (9.0, 27.0)
        assert C.price_of("deepseek-v4-pro", 1, _at(21), table)[:2] == (4.5, 13.5)

    def test_qwen_max_flat_pricing(self):
        """qwen-max 没有峰谷也没有阶梯：2.4 / 9.6（阿里云官方页 2026-10-23 核对）。"""
        from core import llm_cost as C
        table = C.load_prices(force=True)
        for hour in (9, 15, 22):
            assert C.price_of("qwen-max", 8000, _at(hour), table)[:2] == (2.4, 9.6)

    def test_qwen37_tier_boundary(self):
        """阶梯按**单次请求**输入 token 判档：32768 内 0.2/0.8，越界跳 0.6/2.4。"""
        from core import llm_cost as C
        table = C.load_prices(force=True)
        assert C.price_of("qwen3.7-flash", 32768, _at(16), table)[:2] == (0.2, 0.8)
        assert C.price_of("qwen3.7-flash", 32769, _at(16), table)[:2] == (0.6, 2.4)

    def test_unknown_model_is_conservative_not_zero(self):
        """未知模型按最贵的在用模型计。算成 0 = 报表永远"免费"，比算错更危险。"""
        from core import llm_cost as C
        table = C.load_prices(force=True)
        pin, pout, label = C.price_of("some-new-model-x", 1000, _at(15), table)
        assert (pin, pout) == (2.4, 9.6)
        assert "未知" in label

    def test_prefix_match(self):
        """带日期后缀的模型名要能命中主名，否则会掉进"未知模型"高价档。"""
        from core import llm_cost as C
        table = C.load_prices(force=True)
        assert C.price_of("deepseek-v4-flash-0915", 1000, _at(20), table)[:2] == (1.5, 4.5)

    def test_fallback_table_when_file_missing(self, monkeypatch, tmp_path):
        """单价表读不到 ⇒ 用内置兜底表（不是空表、不是抛异常）。"""
        from core import llm_cost as C
        monkeypatch.setenv(C.ENV_PRICES_PATH, str(tmp_path / "nope.json"))
        C._cache.update({"path": None, "mtime": None, "data": None})
        pin, pout, _ = C.price_of("deepseek-v4-flash", 1000, _at(20))
        assert (pin, pout) == (3.0, 9.0)      # 兜底表不带峰谷 ⇒ 保守取原价
        assert C.price_of("qwen-max", 1, _at(20))[:2] == (2.4, 9.6)


class TestCallCost:
    def test_zero_tokens_zero_cost(self):
        from core import llm_cost as C
        assert C.call_cost("qwen-max", 0, 0, _at(15)) == 0.0

    def test_small_call_keeps_precision(self):
        """单次小调用常在 ¥0.000x —— 四舍五入到分就全成 0，必须保留 6 位。"""
        from core import llm_cost as C
        cost = C.call_cost("qwen3.7-flash", 800, 200, _at(16))
        assert 0 < cost < 0.001
        assert cost == round(cost, 6)

    def test_qwen_max_single_call_matches_hand_calc(self):
        """8640 输入 / 190 输出 = 0.020736 + 0.001824 = ¥0.02256（实测均值）。"""
        from core import llm_cost as C
        assert C.call_cost("qwen-max", 8640, 190, _at(16)) == pytest.approx(0.02256, abs=1e-6)


# ================================================================
# 读库汇总
# ================================================================

class TestDailyReport:
    def test_failed_call_is_not_charged(self, tmp_path, monkeypatch):
        """失败调用拿不到 usage，硬算就是凭空造钱。"""
        use_template_db(tmp_path, monkeypatch)
        from db import store
        from core import llm_cost as C
        store.insert_llm_call({"model": "qwen-max", "purpose": "x", "ok": False,
                               "prompt_tokens": 0, "completion_tokens": 0,
                               "total_tokens": 0})
        rep = C.daily_report(days=1)
        assert rep["total_cost"] == 0.0

    def test_stored_cost_wins_over_recompute(self, tmp_path, monkeypatch):
        """落库金额是"发生时价格"，不能被后来的改价覆盖。"""
        use_template_db(tmp_path, monkeypatch)
        from db import store
        from core import llm_cost as C
        store.insert_llm_call({"model": "qwen-max", "purpose": "x", "ok": True,
                               "prompt_tokens": 1000, "completion_tokens": 1000,
                               "total_tokens": 2000, "cost_cny": 9.99})
        assert C.daily_report(days=1)["total_cost"] == pytest.approx(9.99, abs=1e-6)

    def test_null_cost_falls_back_to_estimate(self, tmp_path, monkeypatch):
        """老行 cost_cny 为 NULL ⇒ 按当前单价补算，而不是当 0。"""
        use_template_db(tmp_path, monkeypatch)
        from db import store
        from core import llm_cost as C
        store.insert_llm_call({"model": "qwen-max", "purpose": "x", "ok": True,
                               "prompt_tokens": 1_000_000, "completion_tokens": 0,
                               "total_tokens": 1_000_000})
        assert C.daily_report(days=1)["total_cost"] == pytest.approx(2.4, abs=1e-6)

    def test_workday_average_ignores_idle_days(self, tmp_path, monkeypatch):
        """周末零星调用不能把日均压成"看起来几乎免费"。"""
        use_template_db(tmp_path, monkeypatch)
        from db import store
        from core import llm_cost as C
        # 一个"有工作量"的日子（>10 万 token）
        store.insert_llm_call({"model": "qwen-max", "purpose": "a", "ok": True,
                               "prompt_tokens": 200_000, "completion_tokens": 0,
                               "total_tokens": 200_000, "cost_cny": 0.48,
                               "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        # 一个只有零星调用的日子（同一窗口内，1 千 token）
        store.insert_llm_call({"model": "qwen-max", "purpose": "b", "ok": True,
                               "prompt_tokens": 1000, "completion_tokens": 0,
                               "total_tokens": 1000, "cost_cny": 0.0024,
                               "created_at": (datetime.now().replace(day=1)
                                              .strftime("%Y-%m-%d %H:%M:%S"))})
        rep = C.daily_report(days=40)
        assert rep["workday_count"] >= 1
        assert rep["avg_workday_cost"] >= 0.48     # 只按有工作量的日子平均

    def test_db_failure_returns_partial_not_raise(self, tmp_path, monkeypatch):
        use_template_db(tmp_path, monkeypatch)
        from db import store
        from core import llm_cost as C
        monkeypatch.setattr(store, "llm_cost_rows",
                            lambda **kw: (_ for _ in ()).throw(RuntimeError("db down")))
        rep = C.daily_report(days=7)
        assert rep["days"] == [] and rep["total_cost"] == 0.0
        assert "db down" in rep["error"]


class TestBudgetStatus:
    def test_disabled_when_cap_not_positive(self, tmp_path, monkeypatch):
        """cap ≤ 0 = 关闭闸门。绝不能理解成"预算为零 ⇒ 全拦"。"""
        use_template_db(tmp_path, monkeypatch)
        from core import llm_cost as C
        st = C.budget_status(daily_cap=0)
        assert st["enabled"] is False and st["exceeded"] is False

    def test_exceeded(self, tmp_path, monkeypatch):
        use_template_db(tmp_path, monkeypatch)
        from db import store
        from core import llm_cost as C
        store.insert_llm_call({"model": "qwen-max", "purpose": "x", "ok": True,
                               "prompt_tokens": 1_000_000, "completion_tokens": 0,
                               "total_tokens": 1_000_000, "cost_cny": 3.0})
        st = C.budget_status(daily_cap=2.0)
        assert st["exceeded"] is True
        assert st["spent"] >= 3.0
        assert st["remaining"] <= -1.0

    def test_degraded_does_not_block(self, tmp_path, monkeypatch):
        """账目查不清 ⇒ 标记 degraded，但**不阻止**今天所有任务。"""
        use_template_db(tmp_path, monkeypatch)
        from core import llm_cost as C
        monkeypatch.setattr(C, "today_cost",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        st = C.budget_status(daily_cap=1.0)
        assert st["degraded"] is True
        assert st["exceeded"] is False


# ================================================================
# 迁移与写入端
# ================================================================

class TestSchemaMigration:
    def test_llm_calls_has_cost_column_and_opinion_cache(self, tmp_path, monkeypatch):
        use_template_db(tmp_path, monkeypatch)
        from db import store
        store.init_store()
        conn = store.get_connection()
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(llm_calls)")}
            tabs = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            conn.close()
        assert "cost_cny" in cols
        assert "opinion_cache" in tabs

    def test_log_call_writes_cost(self, tmp_path, monkeypatch):
        """接线验证：真实写入端必须带金额，否则面板永远只有 token。"""
        use_template_db(tmp_path, monkeypatch)
        from db import store
        from llm.deepseek_client import _log_llm_call
        resp = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000))
        _log_llm_call("qwen-max", "unit.test", 0.0, response=resp, ok=True)
        rows = store.llm_cost_rows()
        assert len(rows) == 1
        assert rows[0]["cost_cny"] == pytest.approx(2.4, abs=1e-6)


# ================================================================
# 跨进程观点缓存
# ================================================================

class TestOpinionCache:
    def test_roundtrip(self, tmp_path, monkeypatch):
        use_template_db(tmp_path, monkeypatch)
        from db import store
        payload = {"date": "10月23日", "opinions": [{"source": "s", "viewpoint": "v"}]}
        store.save_opinion_cache("market_opinions:10月23日", payload)
        assert store.get_opinion_cache("market_opinions:10月23日") == payload

    def test_ttl_expired_returns_none(self, tmp_path, monkeypatch):
        use_template_db(tmp_path, monkeypatch)
        from db import store
        store.save_opinion_cache("k", {"a": 1})
        conn = store.get_connection()
        try:
            conn.execute("UPDATE opinion_cache SET created_at = '2020-01-01 00:00:00'")
            conn.commit()
        finally:
            conn.close()
        assert store.get_opinion_cache("k", max_age_sec=3600) is None
        assert store.get_opinion_cache("k", max_age_sec=0) == {"a": 1}

    def test_corrupt_payload_returns_none(self, tmp_path, monkeypatch):
        use_template_db(tmp_path, monkeypatch)
        from db import store
        store.save_opinion_cache("k", {"a": 1})
        conn = store.get_connection()
        try:
            conn.execute("UPDATE opinion_cache SET payload = '{not json' WHERE cache_key='k'")
            conn.commit()
        finally:
            conn.close()
        assert store.get_opinion_cache("k") is None      # 坏缓存只能多花一次钱，不能崩

    def test_second_process_hits_db_not_model(self, tmp_path, monkeypatch):
        """核心回归：清空内存缓存后（等价于另一个进程），不得再调模型。"""
        use_template_db(tmp_path, monkeypatch)
        import data.market_opinions as mo
        calls = {"n": 0}

        def fake_fetch(date_str):
            calls["n"] += 1
            return {"opinions": [{"source": "s", "viewpoint": "v", "stance": "中性"}], "raw": ""}

        monkeypatch.setattr(mo, "fetch_llm_opinions", fake_fetch)
        monkeypatch.setattr(mo, "_persist", lambda r: None)

        mo._opinions_cache.clear()
        mo.get_market_opinions("10月23日")
        assert calls["n"] == 1

        mo._opinions_cache.clear()          # 模拟另一个进程（内存缓存空）
        mo.get_market_opinions("10月23日")
        assert calls["n"] == 1              # 命中库，没再买一次


# ================================================================
# 盘后批量的成本闸门
# ================================================================

class TestEveningCostGate:
    def test_blocks_when_budget_exhausted(self, tmp_path, monkeypatch):
        use_template_db(tmp_path, monkeypatch)
        from db import store
        import core.evening_analysis as EA

        store.insert_llm_call({"model": "qwen-max", "purpose": "x", "ok": True,
                               "prompt_tokens": 1_000_000, "completion_tokens": 0,
                               "total_tokens": 1_000_000, "cost_cny": 9.0})
        monkeypatch.setattr(EA, "_cost_cap", lambda: 1.0)
        monkeypatch.setattr(EA, "candidates_for_date",
                            lambda *a, **k: [{"stock_code": "600000", "score": 90, "name": "浦发"}])
        monkeypatch.setattr(EA, "analyzed_codes", lambda d: set())
        monkeypatch.setattr(EA, "recently_analyzed", lambda d, days=7: {})
        monkeypatch.setattr(EA, "trigger_exempt_codes", lambda d, days=7: set())
        monkeypatch.setattr(EA, "_record", lambda s: None)

        ran = {"n": 0}
        stats = EA.run_evening_analysis(runner=lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))

        assert ran["n"] == 0, "超预算时不该调用分析"
        assert "预算" in (stats.get("skipped_reason") or "")

    def test_no_gate_when_cap_zero(self, tmp_path, monkeypatch):
        """cap=0 ⇒ 不拦（部署忘了配预算时不能全线停摆）。"""
        use_template_db(tmp_path, monkeypatch)
        import core.evening_analysis as EA
        monkeypatch.setattr(EA, "_cost_cap", lambda: 0.0)
        monkeypatch.setattr(EA, "candidates_for_date", lambda *a, **k: [])
        monkeypatch.setattr(EA, "_record", lambda s: None)
        stats = EA.run_evening_analysis(runner=lambda *a, **k: {})
        assert "预算" not in (stats.get("skipped_reason") or "")
