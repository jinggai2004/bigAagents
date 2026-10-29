# -*- coding: utf-8 -*-
"""持仓录入：**成交价不得有假默认值** + 保存前的价位体检（2026-10-16 真实事故）。

事故原样：持仓页「成交价」的默认值是 **10.0**（`value=10.0, min_value=0.01`），
用户不改就提交 ⇒ 中石化(600028) 的成本价被**静默**写成 10.0（真实价 **5.53**），
止损又被填成 **10.5**。三重后果：

  1. 盈亏 / 止损检查 / AI 诊断全部按**错的成本价**计算（实测 `loss_pct = -44.7%`）；
  2. 止损 10.5 ≥ 现价 5.53 ⇒ **一录入就"已跌破止损"**，告警永久为真（纯噪声）；
  3. `min_value=0.01` 让 `price <= 0` 那句校验**永远不可能触发** —— 是**假校验**。

⚠️ 刻意**不做硬校验**：`止损 > 成本价` 在"浮盈后上移止损"时是**合法**的
（`tests/test_position_risk.py` 里就有"止损已被击穿"的用例）。
⇒ 只**如实算出矛盾**（`core.portfolio.level_sanity`）+ 保存前要求**人工确认**。
"""
import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import use_template_db

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(_ROOT, "ui", "pages", "6_💼_持仓.py")

sys.path.insert(0, _ROOT)

CODE = "600028"
LIVE = 5.53                      # 桩现价（中石化真实量级，与事故里的 10.0 差近一倍）


# ================================================================
# 第一部分：纯函数 `level_sanity`（不联网、不写库）
# ================================================================

class TestLevelSanity:
    def test_normal_levels_are_clean(self):
        """正常的 止损 < 成本 < 现价 < 止盈 ⇒ 一条问题都不该报。"""
        from core.portfolio import level_sanity
        assert level_sanity(entry_price=25.0, stop_loss=20.0,
                            take_profit=35.0, current_price=27.0) == []

    def test_trailing_stop_above_cost_is_legal(self):
        """⭐⭐ **鉴别力关键**：浮盈后把止损上移到成本之上是**合法**的，不得误报。

        若这里报错，就说明体检写成了"止损必须低于成本"——那会把正常操作一起拦下。
        """
        from core.portfolio import level_sanity
        out = level_sanity(entry_price=25.0, stop_loss=26.0,   # 止损 > 成本
                           take_profit=35.0, current_price=30.0)  # 但仍 < 现价
        assert out == [], out

    def test_stop_at_or_above_live_price_is_flagged(self):
        """⭐ 事故特征：止损**不低于现价** ⇒ 一录入就已触发。"""
        from core.portfolio import level_sanity
        out = level_sanity(entry_price=10.0, stop_loss=10.5, current_price=LIVE)
        lv = {i["level"]: i for i in out}
        assert "stop_loss" in lv and lv["stop_loss"]["severity"] == "warn"
        assert "已处于触发状态" in lv["stop_loss"]["message"]

    def test_entry_far_from_live_price_is_flagged(self):
        """⭐ 事故的**另一半**：成交价 10.0 vs 现价 5.53（差近一倍）必须被点出来。"""
        from core.portfolio import level_sanity
        out = level_sanity(entry_price=10.0, current_price=LIVE)
        lv = {i["level"]: i for i in out}
        assert "entry_price" in lv
        assert "默认值" in lv["entry_price"]["message"], "必须直指'默认值没改'这个成因"

    def test_no_live_price_means_no_speculation(self):
        """取不到现价 ⇒ **不报**（不猜、不按百分比硬凑）。"""
        from core.portfolio import level_sanity
        assert level_sanity(entry_price=10.0, stop_loss=10.5, current_price=None) == []

    def test_take_profit_below_live_price_is_flagged(self):
        from core.portfolio import level_sanity
        out = level_sanity(entry_price=25.0, take_profit=20.0, current_price=27.0)
        assert any(i["level"] == "take_profit" for i in out)

    def test_blocking_helper_reflects_warn_only(self):
        from core.portfolio import level_sanity_blocking
        assert level_sanity_blocking([]) is False
        assert level_sanity_blocking([{"severity": "info"}]) is False
        assert level_sanity_blocking([{"severity": "warn"}]) is True


# ================================================================
# 第二部分：页面行为（AppTest + 模板库 + 桩行情）
# ================================================================

@pytest.fixture
def page(tmp_path, monkeypatch):
    use_template_db(tmp_path, monkeypatch, "position_entry.db")
    monkeypatch.setenv("BIGA_UI_POLL_SECONDS", "3600")
    monkeypatch.setenv("BIGA_UI_OFFLINE", "1")

    import streamlit as st
    st.cache_data.clear()

    from data import realtime
    monkeypatch.setattr(realtime, "get_realtime_quotes", lambda codes, *a, **k: {
        c: {"code": c, "name": "中国石化", "price": LIVE, "prev_close": LIVE - 0.1,
            "quote_date": "2026-10-16", "stale": False, "data_lagging": False}
        for c in codes if c == CODE})

    # 风控巡检会走 ATR 取 K 线（联网）——换成"取不到"的替身，保持用例离线且快
    monkeypatch.setattr("core.risk.get_risk_engine",
                        lambda: type("E", (), {"suggest_levels": lambda *a, **k: {
                            "unavailable": True, "note": "测试桩"}})())
    yield
    st.cache_data.clear()


def _open_page():
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _set(at, label_contains, value, kind="number"):
    box = [n for n in (at.number_input if kind == "number" else at.text_input)
           if label_contains in str(n.label or "")]
    assert box, f"找不到控件：{label_contains}"
    box[0].set_value(value)
    return at


def _pos_count():
    from core.portfolio import PortfolioManager
    return len(PortfolioManager().active_codes())


class TestPriceFieldHasNoFakeDefault:
    def test_price_starts_empty_not_ten(self, page):
        """⭐⭐ 事故根因：成交价**不得**有"看起来像真实价格"的默认值。"""
        at = _open_page()
        box = [n for n in at.number_input if "成交价" in str(n.label or "")][0]
        assert box.value is None, (
            f"成交价有默认值 {box.value!r} —— 用户不改就会被当成真实成本价写进持仓"
            "（事故：中石化成本被写成 10.0，真实价 5.53）")

    def test_submit_without_price_is_blocked_and_writes_nothing(self, page):
        """⭐ 不填成交价就提交 ⇒ 明确报错 + **一行都不写库**（空值不许变成 0/None 落库）。"""
        at = _open_page()
        before = _pos_count()
        _set(at, "股票代码", CODE, kind="text").run()
        [b for b in at.button if "确认开仓" in b.label][0].click().run()
        errs = "\n".join(str(e.value) for e in at.error)
        assert "实际成交价" in errs, errs or "(没有报错)"
        assert _pos_count() == before, "不填成交价却写入了持仓"


class TestContradictoryLevelsNeedAck:
    def _fill(self, at, price, stop):
        _set(at, "股票代码", CODE, kind="text")
        _set(at, "成交价", price)
        # ⚠️ 数量**必须显式填**：原实现默认 100（"看起来像真的"），2026-10-17 已去掉
        #    —— 忘记改就会把仓位少记。本夹具因此不能再依赖那个默认值。
        _set(at, "数量", 100)
        at.run()
        sl = [n for n in at.number_input if n.key == "open_sl"][0]
        sl.set_value(stop)
        at.run()
        return at

    def test_quantity_has_no_fake_default(self, page):
        """⭐ 与成交价同理：数量也不得有"看起来像真实股数"的默认值。"""
        at = _open_page()
        box = [n for n in at.number_input if "数量" in str(n.label or "")][0]
        assert box.value is None, f"数量有默认值 {box.value!r} ⇒ 忘记改就会少记仓位"

    def test_submit_without_quantity_is_blocked(self, page):
        """⭐ 不填数量 ⇒ 不许落库（空值不得变成 0 或某个默认股数）。"""
        at = _open_page()
        before = _pos_count()
        _set(at, "股票代码", CODE, kind="text")
        _set(at, "成交价", LIVE)
        at.run()
        [b for b in at.button if "确认开仓" in b.label][0].click().run()
        errs = "\n".join(str(e.value) for e in at.error)
        assert "实际股数" in errs, errs or "(没有报错)"
        assert _pos_count() == before, "不填数量却写入了持仓"

    def test_issue_is_shown_and_submit_gated(self, page):
        """⭐ 精确复现事故输入（成交价 10.0 / 止损 10.5 / 现价 5.53）⇒ 必须显示矛盾并要求确认。"""
        at = self._fill(_open_page(), 10.0, 10.5)
        warns = "\n".join(str(w.value) for w in at.warning)
        assert "已处于触发状态" in warns, warns or "(没有提示止损已触发)"
        assert "默认值" in warns, "必须点出'成交价与现价相差过大，疑似默认值'"
        boxes = [c for c in at.checkbox if c.key == "open_ack"]
        assert boxes, "存在矛盾时必须有确认框"

        before = _pos_count()
        [b for b in at.button if "确认开仓" in b.label][0].click().run()
        errs = "\n".join(str(e.value) for e in at.error)
        assert "确认" in errs, errs or "(未勾选却放行了)"
        assert _pos_count() == before, "未确认却写入了持仓"

    def test_clean_input_needs_no_ack_and_saves(self, page):
        """对照组：价位与现价自洽 ⇒ **不该要求确认**（否则每次开仓都要多点一步），且能直接保存。"""
        at = self._fill(_open_page(), LIVE, 5.0)      # 成本=现价、止损在现价下方
        assert not [c for c in at.checkbox if c.key == "open_ack"], \
            "自洽的价位不该要求确认（否则会变成每次都要多点一步）"
        before = _pos_count()
        [b for b in at.button if "确认开仓" in b.label][0].click().run()
        assert "开仓成功" in "\n".join(str(s.value) for s in at.success), \
            [str(e.value) for e in at.error]
        assert _pos_count() == before + 1, "自洽价位应当能正常保存"

    def test_acknowledged_abnormal_input_can_be_saved(self, page):
        """⚠️ 体检只是"要人看一眼"，**不是硬拦** —— 勾选后异常价位也能存。

        理由：`止损 > 成本` 在浮盈上移止损时是合法的，硬拦会误伤正常操作。
        ⚠️ 存进去之后风控面板会**照实**把这个矛盾报成"止损触发"（那是它的职责，不是 bug），
           所以这里断言"保存成功"而不是"没有任何 error"。
        """
        at = self._fill(_open_page(), 10.0, 10.5)
        [c for c in at.checkbox if c.key == "open_ack"][0].set_value(True)
        at.run()
        before = _pos_count()
        [b for b in at.button if "确认开仓" in b.label][0].click().run()
        assert "开仓成功" in "\n".join(str(s.value) for s in at.success), \
            "勾选确认后必须能保存（缺就是缺，但判断权归用户）"
        assert _pos_count() == before + 1
        # 并且风控面板必须**如实**把这个矛盾报出来（这正是事故里用户看到的东西）
        assert any("止损触发" in str(e.value) for e in at.error), \
            "矛盾价位存进去后，风控面板必须把它报出来（不许静默）"
