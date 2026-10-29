"""飞书通知"针对性"改造的测试（2026-10-23）

钉住的判定（每条都对应一次真实事故/实测）：
1. 盘中异动**只推 danger/warn**，info 级只落库
   —— 10-23 实测 154 条告警里 135 条是 info（125 条是"成交额 ≥ 10 亿"），
      占了 88% 的消息量却零信息量。
2. **只属于随机对照组的票不推送**
   —— 但它们仍进监控池、仍参与联赛判定（剔监控池会改变联赛行为）。
3. 告警消息**带关系标签**与**持仓补充信息**（"这条跟我什么关系"）。
4. 推送**幂等**（同日同语义键只推一次）+ **留痕**（push_log）
   —— 10-23 实测：手动重放 + 定时任务把同一份复盘推了两遍。
5. 飞书改 **post 富文本**（此前 text 类型 ⇒ Markdown 全是字面量源码）。
6. 复盘推送**摘要**而非全文（4000+ 字手机上要滚十几屏）。
7. **持仓进监控范围** —— 否则 `AlertEngine` 的"触及止损"规则是**死代码**
   （持仓不在 quotes 里，永远判不到）。
"""

from datetime import datetime
from unittest.mock import patch

import pytest

import core.watcher as w
from core.push import markdown_to_post_lines, push_markdown
from tests.conftest import use_template_db


# ── 工具 ────────────────────────────────────────────────────────────────
def _quote(code="600519", name="贵州茅台", price=110.0, prev=100.0, pct=10.0,
           amount_wan=0.0, turnover=None, stale=False):
    return {"code": code, "name": name, "price": price, "prev_close": prev,
            "change_pct": pct, "amount_wan": amount_wan, "turnover_rate": turnover,
            "quote_time": "14:30:00", "source": "tencent", "stale": stale}


def _info_only_quote(code="600519"):
    """只触发 info 级（成交额 20 亿）的行情：不涨不跌、换手低。"""
    return _quote(code=code, name="贵州茅台", price=110.0, prev=108.0, pct=1.85,
                  amount_wan=200000.0, turnover=1.0)


def _danger_quote(code="600519"):
    """触发 danger 级（涨停）的行情。"""
    return _quote(code=code, price=110.0, prev=100.0, pct=10.0)


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    use_template_db(tmp_path, monkeypatch, "push_targeting.db")


@pytest.fixture
def feishu_only(monkeypatch):
    """只保留飞书通道（钉钉留空），并拦掉真实 HTTP。"""
    monkeypatch.setattr("config.settings.config.FEISHU_WEBHOOK", "https://example.invalid/hook")
    monkeypatch.setattr("config.settings.config.DINGTALK_WEBHOOK", "")
    sent = []
    monkeypatch.setattr("core.push._push_feishu",
                        lambda webhook, title, text: sent.append((title, text)))
    return sent


# ── 1. 飞书 post 富文本 ─────────────────────────────────────────────────
class TestFeishuPost:
    def test_heading_becomes_bold_line(self):
        lines = markdown_to_post_lines("## 盘中异动\n\n正文")
        assert lines[0] == [{"tag": "text", "text": "盘中异动", "style": ["bold"]}]

    def test_inline_bold_is_split_into_styled_element(self):
        lines = markdown_to_post_lines("- **贵州茅台(600519)** 涨停")
        bold = [e for e in lines[0] if e.get("style") == ["bold"]]
        assert bold and bold[0]["text"] == "贵州茅台(600519)"
        assert any(e["text"].startswith("• ") for e in lines[0])

    def test_markdown_marks_do_not_leak_as_literal(self):
        """正文里不能残留 `**`/`##` —— 此前用 text 类型时它们原样显示。"""
        lines = markdown_to_post_lines("## 标题\n\n**加粗** 和 `代码`")
        all_text = "".join(e.get("text", "") for ln in lines for e in ln)
        assert "**" not in all_text and "##" not in all_text

    def test_table_row_is_kept_as_text(self):
        """post 不支持表格 ⇒ 宁可留纯文本，也不要假装排好了版。"""
        lines = markdown_to_post_lines("| 阶段 | 状态 |\n|---|---|")
        assert any("|" in e.get("text", "") for ln in lines for e in ln)

    def test_overlong_content_is_marked_as_truncated(self):
        lines = markdown_to_post_lines("\n".join(f"第{i}行" for i in range(400)),
                                       max_lines=10)
        tail = "".join(e.get("text", "") for e in lines[-1])
        assert "截断" in tail


# ── 2. 幂等 + 留痕 ─────────────────────────────────────────────────────
class TestPushIdempotency:
    def test_same_dedupe_key_pushes_only_once(self, feishu_only):
        r1 = push_markdown("今日复盘正文", title="T", dedupe_key="review:2026-10-23")
        r2 = push_markdown("今日复盘正文", title="T", dedupe_key="review:2026-10-23")
        assert "已推送" in r1["飞书"]
        assert "跳过" in r2["飞书"]
        assert len(feishu_only) == 1, "同一天同一语义键只能推一次"

    def test_different_dedupe_key_both_pushed(self, feishu_only):
        push_markdown("A", title="T", dedupe_key="review:2026-10-23")
        push_markdown("B", title="T", dedupe_key="risk:2026-10-23")
        assert len(feishu_only) == 2

    def test_failed_push_does_not_block_retry(self, feishu_only, monkeypatch):
        """失败记录不能当"已推"（唯一索引也只约束 ok 行）。"""
        calls = {"n": 0}

        def _boom(webhook, title, text):
            calls["n"] += 1
            raise RuntimeError("webhook down")

        monkeypatch.setattr("core.push._push_feishu", _boom)
        r1 = push_markdown("X", title="T", dedupe_key="review:2026-10-23")
        assert "失败" in r1["飞书"]
        # 换成能推的再试一次 ⇒ 必须真的发出（而不是被失败记录挡住）
        monkeypatch.setattr("core.push._push_feishu", lambda w, t, m: None)
        r2 = push_markdown("X", title="T", dedupe_key="review:2026-10-23")
        assert "已推送" in r2["飞书"]

    def test_push_is_logged(self, feishu_only):
        from db import store
        push_markdown("正文内容", title="标题", dedupe_key="review:2026-10-23")
        rows = store._get_rows("SELECT * FROM push_log") if hasattr(store, "_get_rows") else None
        if rows is None:
            conn = store.get_connection()
            try:
                rows = conn.execute("SELECT * FROM push_log").fetchall()
            finally:
                conn.close()
        assert len(rows) == 1
        assert rows[0]["status"] == "ok" and rows[0]["channel"] == "飞书"


# ── 3. 盯盘分级：info 静默 ─────────────────────────────────────────────
class TestAlertLevelRouting:
    def test_info_level_is_not_pushed(self):
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("data.realtime.get_realtime_quotes",
                   return_value={"600519": _info_only_quote()}), \
             patch("db.store.insert_alerts", return_value=1), \
             patch("core.push.push_markdown") as mock_push:
            r = w.AlertWatcher().scan_once(codes=["600519"],
                                           now=datetime(2026, 10, 23, 10, 0), stop_loss={})
        assert r["alerts"] >= 1 and r["fired"] >= 1, "info 级仍要落库"
        mock_push.assert_not_called()
        assert "只落库" in r["push"].get("提示", "")

    def test_danger_level_is_pushed(self):
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("data.realtime.get_realtime_quotes",
                   return_value={"600519": _danger_quote()}), \
             patch("db.store.insert_alerts", return_value=1), \
             patch("core.push.push_markdown", return_value={"飞书": "已推送"}) as mock_push:
            r = w.AlertWatcher().scan_once(codes=["600519"],
                                           now=datetime(2026, 10, 23, 10, 0), stop_loss={})
        assert mock_push.called
        assert r["push"] == {"飞书": "已推送"}


# ── 4. 随机对照组：只过滤推送，不动监控池 ───────────────────────────────
class TestRandomControlGroup:
    def test_random_only_code_is_not_pushed(self):
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("core.ai_wake.random_only_codes", return_value={"600887"}), \
             patch("data.realtime.get_realtime_quotes",
                   return_value={"600887": _danger_quote(code="600887")}), \
             patch("db.store.insert_alerts", return_value=1), \
             patch("core.push.push_markdown") as mock_push:
            r = w.AlertWatcher().scan_once(codes=["600887"],
                                           now=datetime(2026, 10, 23, 10, 0), stop_loss={})
        mock_push.assert_not_called()
        assert r["fired"] >= 1, "实验票仍要落库（页面可查）"

    def test_code_watched_by_real_persona_is_still_pushed(self):
        """同一只票若真实 persona 也关注 ⇒ 照常推送。"""
        with patch.object(w, "is_trading_day", return_value=True), \
             patch("core.ai_wake.random_only_codes", return_value=set()), \
             patch("data.realtime.get_realtime_quotes",
                   return_value={"600887": _danger_quote(code="600887")}), \
             patch("db.store.insert_alerts", return_value=1), \
             patch("core.push.push_markdown", return_value={"飞书": "已推送"}) as mock_push:
            w.AlertWatcher().scan_once(codes=["600887"],
                                       now=datetime(2026, 10, 23, 10, 0), stop_loss={})
        assert mock_push.called

    def test_random_only_codes_uses_persona_kind_not_name_prefix(self):
        """按 `ai_personas.json` 的 kind 字段判定，不按名字前缀。"""
        from core import ai_wake
        from db import store
        store.upsert_ai_wake_condition("ai_random_1",
                                       {"code": "600887", "kind": "stop_loss", "armed": True})
        store.upsert_ai_wake_condition("ai_risk",
                                       {"code": "601398", "kind": "price_below", "armed": True})
        store.upsert_ai_wake_condition("ai_random_2",
                                       {"code": "601398", "kind": "stop_loss", "armed": True})
        out = ai_wake.random_only_codes(datetime(2026, 10, 23).strftime("%Y-%m-%d"))
        assert "600887" in out, "只有随机组关注的票才排除"
        assert "601398" not in out, "真实 persona 也关注的票必须保留"


# ── 5. 消息内容：关系标签 + 持仓补充 ───────────────────────────────────
class TestMessageContext:
    def test_relation_label_and_position_note_are_included(self):
        captured = {}

        def _capture(text, title=None):
            captured["text"] = text
            return {"飞书": "已推送"}

        with patch.object(w, "is_trading_day", return_value=True), \
             patch("core.ai_wake.random_only_codes", return_value=set()), \
             patch.object(w, "_relation_map", return_value={"600519": "持仓·自选"}), \
             patch("db.store.list_active_positions", return_value=[
                 {"code": "600519", "name": "贵州茅台", "quantity": 100,
                  "stop_loss": 100.0, "take_profit": None, "entry_price": 105.0}]), \
             patch("data.realtime.get_realtime_quotes",
                   return_value={"600519": _danger_quote()}), \
             patch("db.store.insert_alerts", return_value=1), \
             patch("core.push.push_markdown", side_effect=_capture):
            w.AlertWatcher().scan_once(codes=["600519"],
                                       now=datetime(2026, 10, 23, 10, 0), stop_loss={})
        text = captured.get("text", "")
        assert "【持仓·自选】" in text, "必须标明这条告警跟用户什么关系"
        assert "持仓 100 股" in text
        assert "止损" in text


# ── 6. 持仓进监控范围（否则止损规则是死代码）───────────────────────────
class TestPositionInWatchScope:
    def test_positions_are_included_in_default_codes(self):
        """持仓必须在盯盘范围内（否则 stop_loss 规则是死代码）。

        2026-10-27 起 `_default_codes` **不再自己抄一份来源元组**（那样会与
        `POOL_SOURCES` 各自演化），改为直接吃默认定义。所以断言两件事：
        ① 它没传参；② 默认定义里确实有 positions 与两套自选。
        """
        captured = {}

        def _fake_codes(sources=None):
            captured["sources"] = sources
            return ["600519"]

        with patch("data.watchlist.attention_codes", side_effect=_fake_codes):
            w._default_codes()
        assert captured["sources"] is None, "应使用默认 POOL_SOURCES，不要在这里再写一份"

        import data.watchlist as W
        assert "positions" in W.POOL_SOURCES, "持仓必须在盯盘范围内"
        assert {"watchlist_user", "watchlist_ashen"} <= set(W.POOL_SOURCES)

    def test_stop_loss_map_does_not_go_online(self):
        """止损价是库里的数 ⇒ 读表即可，绝不能走逐只联网取价那条路。"""
        with patch("db.store.list_active_positions", return_value=[
                {"code": "600519", "name": "贵州茅台", "quantity": 100,
                 "stop_loss": 99.5, "take_profit": None, "entry_price": 105.0}]) as m, \
             patch("core.portfolio.get_portfolio_manager") as pm:
            out = w._stop_loss_map()
        assert out == {"600519": 99.5}
        pm.assert_not_called(), "持仓读取不得触发 PortfolioManager 的联网取价路径"


# ── 7. 复盘摘要 ─────────────────────────────────────────────────────────
class TestReviewSummary:
    def _digest(self):
        return {
            "today": {
                "screens": [{"code": "600276", "name": "恒瑞医药", "score": 92.9},
                            {"code": "601398", "name": "工商银行", "score": 82.9}],
                "analyses": [{"code": "002040", "name": "南京港", "advice_status": "驳回"}],
                "positions": [],
            },
            "actions": {
                "positions": [],
                "watch": [{"code": "601088", "name": "中国神华", "advice_status": "等待",
                           "entry": None, "to_entry_pct": None,
                           "entry_conditions": ["股价回踩MA20（48.73元）附近并缩量企稳"]}],
                "risks": [{"kind": "sector_outflow",
                           "text": "公路铁路运输 连续两日资金净流出"}],
            },
        }

    def test_summary_contains_key_sections(self):
        from core.review_agent import build_review_summary
        s = build_review_summary(digest=self._digest(), integrity_md="",
                                 llm_markdown="## 大盘概览\n\n今日三大指数集体收涨，成长风格占优。",
                                 report_path="output/daily_reports/review_x.md", today="2026-10-23")
        assert "复盘摘要" in s
        assert "恒瑞医药" in s and "92.9" in s
        assert "中国神华" in s and "MA20" in s, "待触发要给出等什么条件"
        assert "公路铁路运输" in s
        assert "output/daily_reports/review_x.md" in s, "必须给出全文路径"

    def test_summary_is_much_shorter_than_full_report(self):
        from core.review_agent import build_review_summary
        long_text = "## 大盘概览\n\n" + ("市场表现与结构分析。" * 300)
        s = build_review_summary(digest=self._digest(), integrity_md="",
                                 llm_markdown=long_text,
                                 report_path="p.md", today="2026-10-23")
        assert len(s) < len(long_text) / 3

    def test_lede_skips_header_quote_line(self):
        from core.review_agent import _extract_lede
        text = "> 生成时间 15:52 ｜ 数据源：腾讯\n\n## 大盘概览\n\n真实结论在这里。"
        lede = _extract_lede(text)
        assert "真实结论在这里" in lede
        assert "生成时间" not in lede
