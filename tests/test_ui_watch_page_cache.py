# -*- coding: utf-8 -*-
"""看盘页（`ui/pages/3_📊_看盘.py`）取数缓存 + 手动刷新一致性的测试。

背景（2026-10-15 实测）：页 3 冷启动 ≈8.2s，其中
`get_overnight_us()` 2.96s + `get_industry_flow_snapshot()` 2.65s +
`get_index_quotes()` / `get_realtime_quotes()` 各 0.79s + pandas Styler 首次模板 0.63s。
而 Streamlit **每次交互都重跑整个脚本** ⇒ 不缓存就是"每个控件都付一遍联网成本"。

⭐ 三条必须钉住的语义：
  ① **TTL 内重跑不得重新取数**（这是"切换/点控件变快"的全部依据）；
  ② **"刷新行情"必须真的重新取数** —— 只清 `realtime` 内部缓存而漏清页面级
     `st.cache_data`，会表现为"点了刷新，价格纹丝不动"（最难发现的静默 bug）；
  ③ **失败（返回 None）也要进缓存** —— `st.cache_data` 不缓存异常，
     源挂掉时若每次 rerun 都重试，就是"每次切到热度块都卡 2.6 秒"。
"""
import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import use_template_db

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCH_PAGE = os.path.join(_ROOT, "ui", "pages", "3_📊_看盘.py")

sys.path.insert(0, _ROOT)


@pytest.fixture
def calls(tmp_path, monkeypatch):
    """把四个数据源全部换成**计数器 + 固定返回**：不联网、可断言调用次数。"""
    use_template_db(tmp_path, monkeypatch, "ui_watch_cache.db")
    monkeypatch.setenv("BIGA_UI_POLL_SECONDS", "3600")
    monkeypatch.setenv("BIGA_UI_OFFLINE", "1")

    import streamlit as st
    st.cache_data.clear()

    from data import fund_flow, realtime, us_market, watchlist

    counters = {"us": 0, "index": 0, "watch": 0, "flow": 0}

    def _us():
        counters["us"] += 1
        return {"unavailable": True, "note": "测试桩", "indexes": []}

    def _index():
        counters["index"] += 1
        return {}

    def _watch(codes, force_refresh=False):
        counters["watch"] += 1
        return {}

    def _flow():
        counters["flow"] += 1
        return None                     # 故意返回失败态，验证"失败也进缓存"

    # ⚠️ 页面脚本每次 run 都会重新执行 `from ... import ...`，因此补丁打在
    #    **源模块属性**上即可生效（别去补页面命名空间里那个早已绑定的名字）。
    monkeypatch.setattr(us_market, "get_overnight_us", _us)
    monkeypatch.setattr(realtime, "get_index_quotes", _index)
    monkeypatch.setattr(realtime, "get_realtime_quotes", _watch)
    monkeypatch.setattr(fund_flow, "get_industry_flow_snapshot", _flow)
    monkeypatch.setattr(watchlist, "watchlist_codes", lambda: ["600028", "600519"])
    yield counters
    st.cache_data.clear()


def _render():
    at = AppTest.from_file(WATCH_PAGE, default_timeout=90)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


class TestBlockLoadersAreCached:
    def test_first_render_calls_each_source_once(self, calls):
        _render()
        assert calls == {"us": 1, "index": 1, "watch": 1, "flow": 1}, calls

    def test_rerun_within_ttl_does_not_refetch(self, calls):
        """⭐ TTL 内重跑（= 切换区块 / 点任意控件）必须命中缓存，不再取数。"""
        at = _render()
        before = dict(calls)
        at.run()                                    # 模拟一次 rerun
        assert calls == before, f"TTL 内重跑又取数了：{before} → {calls}"

    def test_failed_source_is_negatively_cached(self, calls):
        """⭐ 源不可达（返回 None）也必须被缓存 —— 否则每次 rerun 都要重等一遍超时。"""
        at = _render()
        at.run()
        assert calls["flow"] == 1, "行业资金流失败后仍在重复重试"


class TestRefreshInvalidatesPageCache:
    def test_refresh_button_refetches_quotes(self, calls):
        """⭐ "刷新行情"必须真的重新取数。

        ⚠️ 只清 `realtime.clear_quote_cache()` 是不够的：页面级 `st.cache_data`
           会把整次调用都短路掉，函数**根本不会被调用** ⇒ 表现成"点了刷新没反应"。
        """
        at = _render()
        assert calls["index"] == 1 and calls["watch"] == 1
        at.button[0].click().run()                  # 🔄 刷新行情
        assert not at.exception, [str(e.value) for e in at.exception]
        assert calls["index"] > 1, "点刷新后指数行情没有重新取数（页面缓存未清）"
        assert calls["watch"] > 1, "点刷新后自选股行情没有重新取数（页面缓存未清）"

    def test_refresh_does_not_refetch_slow_sources(self, calls):
        """刷新按钮只刷"行情"，不该顺带把外盘（30min）与资金流（600s）也打一遍。"""
        at = _render()
        us_before, flow_before = calls["us"], calls["flow"]
        at.button[0].click().run()
        assert calls["us"] == us_before and calls["flow"] == flow_before


class TestPreloadWarmsWatchPageSources:
    def test_warm_steps_cover_the_four_sources(self):
        """预热清单必须覆盖看盘页的四个源（否则首访仍要冷等 ~7.2s）。"""
        from ui import _preload as P
        labels = " ".join(label for label, _ in P._WARM_STEPS)
        for need in ("外盘", "指数行情", "自选股行情", "行业资金流", "Styler"):
            assert need in labels, f"预热清单缺少：{need}"

    def test_warm_failure_is_swallowed(self, monkeypatch, caplog):
        """⭐ 预热失败必须只记日志：无网时页面照旧自己取数，不能被预热拖崩。"""
        from ui import _preload as P

        def _boom():
            raise RuntimeError("模拟无网")

        monkeypatch.setattr(P, "PRELOAD_TARGETS", ())
        monkeypatch.setattr(P, "_WARM_STEPS", (("坏源", _boom), ("Styler 模板", lambda: None)))
        P._run()                                    # 不抛异常即通过

    def test_warm_data_is_readonly(self, monkeypatch):
        """预热只允许调用**只读**取数（不许出现写操作）。

        ⚠️ 每新增一个 `_warm_*` 都要加进这里的检查清单 —— 漏掉就等于**新代码没被守护**，
           而这个断言仍会"全绿"（结构性守护最容易这样退化成假安全）。
        """
        import inspect
        from ui import _preload as P
        src = inspect.getsource(P._warm_data) + inspect.getsource(P._warm_us) + \
            inspect.getsource(P._warm_index) + inspect.getsource(P._warm_watchlist) + \
            inspect.getsource(P._warm_paper_quotes) + \
            inspect.getsource(P._warm_industry_flow)
        for forbidden in ("save", "submit_order", "insert", "delete", "apply_trade"):
            assert forbidden not in src, f"预热里出现了写操作关键字：{forbidden}"
