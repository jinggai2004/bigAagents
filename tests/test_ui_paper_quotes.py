# -*- coding: utf-8 -*-
"""模拟账户页（页8）行情取数：**按账户缓存 + 缺价自动补取**（2026-10-16 修复）。

用户报障原文：「模拟账户的资金需加载才能正确显示，为何不预加载」。

根因是**两处缺陷叠加**（缺任何一处都不会出问题）：

  ① 行情缓存键是**全局**的 `"paper_quotes"`，不区分账户
     ⇒ 切到 `ai_value` 等账本时，复用的是 `default`（无持仓）留下的空 dict
     ⇒ 新账户的持仓**一只价都没有**；
  ② 守卫写成 `if "paper_quotes" not in st.session_state`
     ⇒ 只要**首次进页面时账户是空仓**，键就被写成 `{}`，
       此后**任何账户都不会再自动取价**（必须手点「刷新现价」）。

⇒ 持仓只能按**成本价**估值 ⇒「总资产 = 现金 + 成本 ≈ 初始资金」，
  用户看到 `1,000,000（+0.00%）` 会以为账户被重置 —— 这正是 10-09 加"进入页面自动取价"
  要避免的误读，却因为键写错而**在切账户后必然复发**。

⭐ 本文件钉住四条语义：
  ① 切到有持仓的账户 ⇒ **自动**取价（无需点按钮）；
  ② 缓存**按账户隔离**（不再共用全局键）；
  ③ **默认账户空仓**这一触发条件不得再"堵死"后续账户的自动取价；
  ④ TTL 内重跑（点其它控件 / 切区块）**不得重复取数**。
"""
import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import use_template_db

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_PAGE = os.path.join(_ROOT, "ui", "pages", "8_💰_模拟账户.py")

sys.path.insert(0, _ROOT)

ACCT = "testacct"
CODE = "600028"
PRICE = 12.0                       # 桩行情价（与买入成本 10.0 不同 ⇒ 能区分"现价/成本价"）
QTY = 100


@pytest.fixture
def calls(tmp_path, monkeypatch):
    """模板库 + 桩行情：不联网、可断言调用次数。"""
    use_template_db(tmp_path, monkeypatch, "ui_paper_quotes.db")
    monkeypatch.setenv("BIGA_UI_POLL_SECONDS", "3600")
    monkeypatch.setenv("BIGA_UI_OFFLINE", "1")          # 关掉进程预热（会真联网）

    import streamlit as st
    st.cache_data.clear()

    # 播种一个有持仓的账户（买入日设在过去 ⇒ 已解冻，页面不会因 unlock 写库）
    from core.paper_account import PaperAccount
    from db import store
    acct = PaperAccount(1_000_000, name=ACCT)
    acct.apply_trade({"date": "2026-10-13", "code": CODE, "direction": "买入",
                      "price": 10.0, "quantity": QTY})
    store.save_paper_state(ACCT, acct.to_state())

    counter = {"n": 0}

    # ⚠️ 补丁打在**源模块属性**上：页面脚本每次 run 都会重新执行 import。
    from data import realtime
    monkeypatch.setattr(realtime, "get_realtime_quotes", _stub(counter))

    yield counter
    st.cache_data.clear()


def _stub(counter):
    def _fn(codes, *a, **k):
        counter["n"] += 1
        return {c: {"code": c, "name": "测试标的", "price": PRICE,
                    "open": PRICE, "prev_close": PRICE,
                    "quote_date": "2026-10-16", "stale": False,
                    "data_lagging": False}
                for c in codes}
    return _fn


def _render():
    at = AppTest.from_file(PAPER_PAGE, default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def _switch_to(at, name):
    sb = next(s for s in at.selectbox if "账户" in s.label)
    sb.set_value(name).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


class TestPaperQuotesAutoFetch:
    def test_empty_default_account_does_not_block_others(self, calls):
        """⭐⭐ 报障的**精确复现路径**：先停在空仓的 `default`，再切到有持仓账户。

        原实现里首次渲染把 `paper_quotes` 写成 `{}` ⇒ 切账户后**永不再取价**。
        """
        at = _render()                                  # 默认账户 default（空仓）
        assert calls["n"] == 0, "空仓账户不该发起取数"

        at = _switch_to(at, ACCT)
        assert calls["n"] >= 1, "切到有持仓账户后**必须自动取价**（不能再要手点刷新）"

        m = _metrics(at)
        total_label = next(lb for lb in m if lb.startswith("总资产"))
        assert "现价" in total_label, f"总资产仍按成本价估值：{total_label}"
        assert m[next(lb for lb in m if lb.startswith("持仓市值"))] == f"{QTY * PRICE:,.2f}"

    def test_quotes_are_cached_per_account(self, calls):
        """⭐ 结构性：缓存键必须**带账户名**（全局键正是 bug 的根）。"""
        at = _switch_to(_render(), ACCT)
        assert calls["n"] >= 1
        try:
            at.session_state[f"paper_quotes::{ACCT}"]
        except KeyError:                                # pragma: no cover
            pytest.fail("行情缓存未按账户隔离（找不到 paper_quotes::<账户> 键）")

    def test_rerun_within_ttl_does_not_refetch(self, calls):
        """⭐ TTL 内重跑不得重复取数（否则点每个控件都付一次联网成本）。"""
        at = _switch_to(_render(), ACCT)
        first = calls["n"]
        assert first >= 1
        at.run()                                        # 模拟一次普通 rerun
        at.run()
        assert calls["n"] == first, f"TTL 内重复取数：{first} → {calls['n']}"

    def test_refresh_button_refetches_prices(self, calls):
        """⭐ 「刷新现价」必须**真的**重新取数。

        ⚠️ 页面级 `st.cache_data` 加了之后，只清 `realtime` 内部缓存会被整个短路
           ⇒ "点了刷新价格纹丝不动"（看盘页踩过同一个坑）。
        """
        at = _switch_to(_render(), ACCT)
        before = calls["n"]
        btn = next(b for b in at.button if "刷新现价" in b.label)
        btn.click().run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert calls["n"] > before, "点了刷新却没有重新取价（被页面缓存短路）"

    def test_failed_fetch_is_cached_and_does_not_storm(self, calls):
        """⚠️ 取不到价（返回空）也必须是**可缓存的正常返回**。

        `st.cache_data` **不缓存异常** —— 若让异常穿透，每次 rerun 都会把同一个超时
        重等一遍（切账户/点控件都卡）。这里断言"失败后不反复重试"。
        """
        from data import realtime
        import ui.pages  # noqa: F401  （仅为保持导入顺序与生产一致）

        counter = {"n": 0}

        def _boom(codes, *a, **k):
            counter["n"] += 1
            raise RuntimeError("模拟源不可用")

        import pytest as _pytest
        mp = _pytest.MonkeyPatch()
        mp.setattr(realtime, "get_realtime_quotes", _boom)
        try:
            at = _switch_to(_render(), ACCT)
            first = counter["n"]
            at.run()
            at.run()
            assert counter["n"] == first, (
                f"取价失败后被反复重试（{first} → {counter['n']}）："
                "说明异常穿透了缓存 —— 应改为返回 {} 的正常返回")
            # 页面必须如实说明"按成本价估值"，而不是静默显示假现价
            caps = "\n".join(str(c.value) for c in at.caption)
            assert "成本价" in caps
        finally:
            mp.undo()
