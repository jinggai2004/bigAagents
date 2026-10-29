# -*- coding: utf-8 -*-
"""热点新闻页 · 知乎「按博主采集」区块（2026-10-23）

背景：用户报"知乎不能返回财经博主的帖子"（`https://www.zhihu.com/people/shou-wo-rou-ti`）。
实测结论（已写进 `data/zhihu_opinions.py`）：
  · 开放平台 v1 **没有任何按用户查的端点**（10 个候选路径全部 404）；
  · 只能走网页端 `/api/v4/members/<url_token>/pins`，且**仅「想法」免登录**
    （`/answers`、`/articles` 实测 403 need_login）。

⚠️ 本文件**故意不设 `BIGA_UI_OFFLINE`**：
   整个知乎区块在离线模式下会被整段跳过，不 unset 就等于这条路径**零覆盖**
   —— 那是"测试写了没跑"的翻版。

⚠️ 全程打桩 `auth_ok` / `member_pins`：本文件只验**页面接线**，不联网、不耗额度。
"""
import os

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import use_template_db

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_PAGE = os.path.join(_ROOT, "ui", "pages", "12_📰_热点新闻.py")

SEC_ZHIHU = "💬 知乎话题参考"
MODE_MEMBER = "👤 按博主采集（网页端接口·非官方）"

_ITEM = {
    "topic": "@手握柔荑", "title": "这次还是我赢了。", "text": "这次还是我赢了。",
    "url": "https://www.zhihu.com/pins/2096981397479204169",
    "votes": 55, "comments": 51, "author": "手握柔荑",
    "published": 1792739340, "content_type": "pin",
}


@pytest.fixture
def online(tmp_path, monkeypatch):
    """不设 BIGA_UI_OFFLINE（知乎块的联网分支才会被走到），其余隔离照旧。"""
    use_template_db(tmp_path, monkeypatch, "ui_zhihu_member.db")
    monkeypatch.delenv("BIGA_UI_OFFLINE", raising=False)
    monkeypatch.setenv("BIGA_UI_POLL_SECONDS", "3600")

    import streamlit as st

    from data import zhihu_opinions as Z

    # ⚠️ 三份配置全部重定向到 tmp：页面会调 load_members()，
    #    不重定向就会在真实 `config/` 里创建 zhihu_members.json（污染仓库）。
    monkeypatch.setattr(Z, "MEMBERS_PATH", str(tmp_path / "members.json"))
    monkeypatch.setattr(Z, "TOPICS_PATH", str(tmp_path / "topics.json"))
    monkeypatch.setattr(Z, "_QUOTA_PATH", str(tmp_path / "quota.json"))
    Z._member_cache.update({"mtime": None, "members": None})
    Z._topic_cache.update({"mtime": None, "topics": None})

    # 鉴权探测打桩：否则 `_cached_zhihu_auth()` 会真打一次开放平台
    monkeypatch.setattr(Z, "auth_ok", lambda: (True, "鉴权正常"))

    # ⚠️ `st.cache_data` 是**进程级全局**，不随夹具失效 ⇒ setup 与 teardown 必须成对清
    st.cache_data.clear()
    yield tmp_path
    st.cache_data.clear()


def _texts(at) -> str:
    out = []
    for name in ("markdown", "caption", "text", "info", "warning", "error", "success"):
        for el in getattr(at, name, []) or []:
            out.append(str(getattr(el, "value", "")))
    return "\n".join(out)


def _nav(at):
    from ui.components.section_nav import ALL
    return [r for r in at.radio if ALL in [str(o) for o in r.options]][0]


def _mode_radio(at):
    return [r for r in at.radio
            if MODE_MEMBER in [str(o) for o in r.options]][0]


def _open_zhihu(at):
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    _nav(at).set_value(SEC_ZHIHU).run()
    assert not at.exception, [str(e.value) for e in at.exception]


# ================================================================
# 一、两种采集方式必须**显式选择**
# ================================================================
# ⚠️ 不能揉成"一键采集"：两条通道一条扣开放平台额度、一条不扣，
#    混起来以后"这次到底扣没扣"就说不清了。

class TestModeSwitch:
    def test_both_modes_are_offered(self, online):
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        opts = [str(o) for o in _mode_radio(at).options]
        assert any("按话题" in o for o in opts), opts
        assert any("按博主" in o for o in opts), opts

    def test_default_is_official_topic_channel(self, online):
        """默认必须是**官方**话题通道 —— 非公开接口要用户主动切过去。"""
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        assert str(_mode_radio(at).value).startswith("🔍")

    def test_topic_mode_shows_platform_quota(self, online):
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        assert "知乎搜索额度" in _texts(at)


# ================================================================
# 二、博主模式：边界声明必须在，且不得复用开放平台的额度口径
# ================================================================

class TestMemberMode:
    def test_declares_it_is_not_the_official_api(self, online):
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _mode_radio(at).set_value(MODE_MEMBER).run()
        warned = "\n".join(str(w.value) for w in at.warning)
        assert "非公开" in warned or "网页端内部接口" in warned, warned
        assert "想法" in warned, "必须说明只覆盖「想法」（answers/articles 实为 403）"

    def test_does_not_show_open_platform_quota(self, online):
        """⚠️ 本通道不消耗开放平台额度 ⇒ 页面**不能**再显示 `quota_note()`，
        否则同一个数字会被当成两条通道的共同口径。"""
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _mode_radio(at).set_value(MODE_MEMBER).run()
        text = _texts(at)
        assert "知乎搜索额度" not in text
        assert "不消耗" in text

    def test_collect_renders_items(self, online, monkeypatch):
        from data import zhihu_opinions as Z
        monkeypatch.setattr(Z, "member_pins",
                            lambda token, limit=20, offset=0: [dict(_ITEM)])
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _mode_radio(at).set_value(MODE_MEMBER).run()

        btn = [b for b in at.button if "采集想法" in str(b.label)][0]
        btn.click().run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "这次还是我赢了" in text, text
        assert "手握柔荑" in text
        assert "https://www.zhihu.com/pins/2096981397479204169" in text

    def test_isolation_promise_survives_mode_switch(self, online):
        """⭐⭐ 边界声明必须**与采集方式无关**地可见。

        这是本项目反复踩过的坑：声明被放进某个分支里，一旦走了另一条分支
        （或离线跳过）就消失，于是"不参与评分"的承诺就没人看得到了。
        """
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _mode_radio(at).set_value(MODE_MEMBER).run()
        text = _texts(at)
        assert "仅供人工参考" in text
        assert "不参与任何模型分析与评分" in text

    def test_failure_reason_is_shown_not_swallowed(self, online, monkeypatch):
        """取不到时必须说清是哪一步失败，不能只显示"未取到"。"""
        from data import zhihu_opinions as Z

        def _fail(token, limit=20, offset=0):
            Z._set_error("博主不存在", f"HTTP 404：url_token「{token}」在知乎查不到")
            return []

        monkeypatch.setattr(Z, "member_pins", _fail)
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _mode_radio(at).set_value(MODE_MEMBER).run()
        [b for b in at.button if "采集想法" in str(b.label)][0].click().run()
        text = _texts(at)
        assert "失败原因" in text, text
        assert "博主不存在" in text


# ================================================================
# 三、市场情绪参照（2026-10-23）
# ================================================================
# 这一层把"采集到的想法"提炼成**结构化指标**再交给模型。
# 三条必须守住的性质：
#   ① 采集即落库（相对基准靠历史，靠"用户想起来才点"永远凑不齐）；
#   ② 样本不足必须**写在数字旁边**，不能只给一个漂亮比例；
#   ③ LLM 判定做成按钮，不随 rerun 自动跑（那是按次计费的）。

class TestSentimentCard:
    def _collect(self, at, items):
        from data import zhihu_opinions as Z
        btn = [b for b in at.button if "采集想法" in str(b.label)][0]
        btn.click().run()
        assert not at.exception, [str(e.value) for e in at.exception]

    def test_card_appears_after_collect(self, online, monkeypatch):
        from data import zhihu_opinions as Z
        monkeypatch.setattr(Z, "member_pins",
                            lambda token, limit=20, offset=0: [dict(_ITEM)])
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _mode_radio(at).set_value(MODE_MEMBER).run()
        self._collect(at, None)
        text = _texts(at)
        assert "市场情绪参照" in text, text
        labels = [str(m.label) for m in at.metric]
        assert "判出方向" in labels, labels
        assert "多空比" in labels, labels

    def test_snapshot_is_persisted_on_collect(self, online, monkeypatch):
        """⭐ 采集即落库 —— 否则相对基准永远没有历史。"""
        from data import zhihu_opinions as Z
        from db import store

        # 造 20 条明确表态，保证 n_scored 过门槛（否则 quality=thin 也照样落库，
        # 但这里想同时验证"达门槛时 quality=ok"这条路径）
        items = ([{"topic": "@a", "title": f"满仓{i}", "text": "满仓了，看好",
                   "url": f"https://x/{i}", "votes": 10, "published": 1792739340}
                  for i in range(10)]
                 + [{"topic": "@a", "title": f"清仓{i}", "text": "清仓了，不看好",
                     "url": f"https://y/{i}", "votes": 5, "published": 1792739340}
                    for i in range(10)])
        monkeypatch.setattr(Z, "member_pins",
                            lambda token, limit=20, offset=0: [dict(x) for x in items])
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _mode_radio(at).set_value(MODE_MEMBER).run()
        self._collect(at, None)

        snaps = store.get_sentiment_day(__import__("datetime").datetime.now().strftime("%Y-%m-%d"))
        src = [s for s in snaps if s.get("source") == "zhihu:member"]
        assert src, f"采集后应落库，实得 {[s.get('source') for s in snaps]}"
        assert src[0]["n_scored"] == 20
        assert src[0]["bull_ratio"] == 0.5

    def test_thin_sample_warning_is_next_to_the_numbers(self, online, monkeypatch):
        """⚠️ 只有 1 条样本时，页面必须明说"不可作依据"，不能只显示 100%。"""
        from data import zhihu_opinions as Z
        monkeypatch.setattr(Z, "member_pins",
                            lambda token, limit=20, offset=0: [dict(_ITEM)])
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _mode_radio(at).set_value(MODE_MEMBER).run()
        self._collect(at, None)
        text = _texts(at)
        assert "样本偏薄" in text, text

    def test_judge_is_button_driven_not_automatic(self, online, monkeypatch):
        """⭐ 判定要花钱 ⇒ 必须由按钮触发；页面渲染本身不得调模型。"""
        from data import zhihu_opinions as Z
        from core import sentiment_reference as SR

        monkeypatch.setattr(Z, "member_pins",
                            lambda token, limit=20, offset=0: [dict(_ITEM)])
        called = {"n": 0}

        def _spy(*a, **k):
            called["n"] += 1
            return {"status": "ok", "stance": "数据不足", "summary": "x",
                    "divergence": "", "confidence": "低", "evidence": [],
                    "cautions": [], "warnings": []}

        monkeypatch.setattr(SR, "judge", _spy)
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _mode_radio(at).set_value(MODE_MEMBER).run()
        self._collect(at, None)
        assert called["n"] == 0, "渲染阶段不得调用模型"

        btn = [b for b in at.button if "情绪判定" in str(b.label)]
        assert btn, "必须有生成判定的按钮"
        btn[0].click().run()
        assert called["n"] == 1

    def test_shows_it_does_not_affect_scoring(self, online, monkeypatch):
        """边界声明必须与指标同屏 —— 否则读数会被当成交易信号。"""
        from data import zhihu_opinions as Z
        monkeypatch.setattr(Z, "member_pins",
                            lambda token, limit=20, offset=0: [dict(_ITEM)])
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _mode_radio(at).set_value(MODE_MEMBER).run()
        self._collect(at, None)
        text = _texts(at)
        assert "不参与评分" in text or "不参与任何模型分析与评分" in text


# ================================================================
# 四、模型逐条标注（2026-10-23）：报告 + 代码算数
# ================================================================
# 这一段的重点是**接线的方向**，不是模型水平：
#   报告与标签来自模型 → 数字由代码从标签算 → 报告只展示、不进提示词。
#
# ⚠️ 这里**不 stub `label_posts`**，而是 stub 到 `DeepSeekClient` 那一层，
#    让 prompt 构造、解析、校验、快照、落库全部真跑。只桩 `label_posts`
#    等于把被测的那段短路掉，剩下的断言只是在验自己的桩。

LABEL_LLM = "🧠 模型逐条标注（出报告 + 逐条标签）"


class _FakeLLM:
    """按脚本返回；记录调用次数与 purpose，供断言"点没点才花钱"。"""

    model = "fake-labeler"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def call_api(self, messages, temperature=None, purpose=None, **kw):
        self.calls.append({"purpose": purpose, "temperature": temperature})
        if not self.replies:
            raise RuntimeError("脚本已用完")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _label_reply(pairs, report=None):
    import json as _json
    return _json.dumps({
        "labels": [{"i": i, "stance": s, "quote": q, "why": "测试"}
                   for i, s, q in pairs],
        "report": report or {},
    }, ensure_ascii=False)


def _patch_llm(monkeypatch, replies):
    import llm.deepseek_client as DC
    holder = _FakeLLM(replies)
    monkeypatch.setattr(DC, "DeepSeekClient", lambda *a, **k: holder)
    return holder


def _label_mode_radio(at):
    return [r for r in at.radio if LABEL_LLM in [str(o) for o in r.options]][0]


def _metric(at, label):
    hit = [m for m in at.metric if str(m.label) == label]
    assert hit, f"缺少指标卡「{label}」，现有：{[str(m.label) for m in at.metric]}"
    return str(hit[0].value)


def _collect(at):
    [b for b in at.button if "采集想法" in str(b.label)][0].click().run()
    assert not at.exception, [str(e.value) for e in at.exception]


def _enter_llm_mode(at):
    """切「按博主」→ **先采集**（指标卡只在有采集结果后才出现）→ 再切模型标注。"""
    _mode_radio(at).set_value(MODE_MEMBER).run()
    _collect(at)
    _label_mode_radio(at).set_value(LABEL_LLM).run()
    assert not at.exception, [str(e.value) for e in at.exception]


class TestLlmLabelMode:
    def _three_items(self, monkeypatch):
        from data import zhihu_opinions as Z
        items = [
            {"topic": "@a", "title": "1", "text": "满仓了，看好",
             "url": "https://x/1", "votes": 10, "published": 1792739340},
            {"topic": "@a", "title": "2", "text": "别问，已梭哈完",
             "url": "https://x/2", "votes": 5, "published": 1792739340},
            {"topic": "@a", "title": "3", "text": "我降了30%的仓位",
             "url": "https://x/3", "votes": 2, "published": 1792739340},
        ]
        monkeypatch.setattr(Z, "member_pins",
                            lambda token, limit=20, offset=0: [dict(i) for i in items])
        return items

    def test_label_mode_is_explicit_and_defaults_to_rule(self, online, monkeypatch):
        """⚠️ 默认必须是零成本的词表：切到模型等于"按条计费"，不能是默认值。"""
        self._three_items(monkeypatch)
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _mode_radio(at).set_value(MODE_MEMBER).run()
        _collect(at)
        r = _label_mode_radio(at)
        assert str(r.value) != LABEL_LLM
        assert any("词表" in str(o) for o in r.options)

    def test_rendering_never_calls_the_model(self, online, monkeypatch):
        """⭐ 渲染阶段调用数必须为 0 —— 否则每次交互都花一次钱。"""
        self._three_items(monkeypatch)
        holder = _patch_llm(monkeypatch, [])
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _enter_llm_mode(at)
        _collect(at)              # 再采一次也不许触发模型
        assert not holder.calls, f"渲染阶段不该调模型，实际调用 {len(holder.calls)} 次"
        assert "不会调用任何模型" in _texts(at)

    def test_button_runs_labeling_and_shows_report(self, online, monkeypatch):
        self._three_items(monkeypatch)
        holder = _patch_llm(monkeypatch, [_label_reply(
            [(1, "bull", "满仓了，看好"), (2, "bull", "已梭哈完"),
             (3, "bear", "降了30%的仓位")],
            report={"主线": "仓位分歧加大", "多头理由": ["还有人满仓"],
                    "空头理由": ["开始减仓"], "争议焦点": ["回调深度"],
                    "异常信号": []})])
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _enter_llm_mode(at)

        run = [b for b in at.button if "开始标注" in str(b.label)]
        assert run, "必须有一个按钮显式触发标注"
        run[0].click().run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert len(holder.calls) == 1
        assert holder.calls[0]["purpose"] == "sentiment.label"

        text = _texts(at)
        assert "仓位分歧加大" in text
        assert "模型总结" in text

    def test_numbers_are_computed_from_labels(self, online, monkeypatch):
        """⭐ 3 条里 2 多 1 空 ⇒ 多空比 67%，与模型写什么无关。"""
        self._three_items(monkeypatch)
        _patch_llm(monkeypatch, [_label_reply(
            [(1, "bull", "满仓了，看好"), (2, "bull", "已梭哈完"),
             (3, "bear", "降了30%的仓位")],
            # 报告里故意写一个**假的**统计数字：它必须被丢弃，不能显示出来
            report={"主线": "八成看多", "多头理由": [], "空头理由": [],
                    "争议焦点": [], "异常信号": []})])
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _enter_llm_mode(at)
        [b for b in at.button if "开始标注" in str(b.label)][0].click().run()

        assert _metric(at, "多空比") == "67%"
        assert _metric(at, "判出方向") == "3 条"
        text = _texts(at)
        assert "八成看多" not in text, "模型自报的统计必须被丢弃"
        # 丢弃动作本身要留痕（否则"报告为什么是空的"查不出来），
        # 但留痕里不得复述被删的那句话 —— 否则等于换个位置继续展示。
        assert "丢弃" in text

    def test_report_is_declared_display_only(self, online, monkeypatch):
        """边界必须与报告同屏：报告是二手叙述，不能进复盘提示词。"""
        self._three_items(monkeypatch)
        _patch_llm(monkeypatch, [_label_reply(
            [(1, "bull", "满仓了，看好"), (2, "bear", "已梭哈完"),
             (3, "bear", "降了30%的仓位")], report={"主线": "偏空"})])
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _enter_llm_mode(at)
        [b for b in at.button if "开始标注" in str(b.label)][0].click().run()
        text = _texts(at)
        assert "不进复盘提示词" in text

    def test_dual_track_agreement_is_shown(self, online, monkeypatch):
        """词表与模型在同一条上判反了 ⇒ 必须能被看见（那是值得人工核的样本）。"""
        self._three_items(monkeypatch)
        # 词表会把"满仓了，看好"判 bull，这里让模型判 bear ⇒ 制造冲突
        _patch_llm(monkeypatch, [_label_reply(
            [(1, "bear", "满仓了，看好"), (2, "bear", "已梭哈完"),
             (3, "bear", "降了30%的仓位")], report={"主线": "偏空"})])
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _enter_llm_mode(at)
        [b for b in at.button if "开始标注" in str(b.label)][0].click().run()
        text = _texts(at)
        assert "词表/模型同向" in [str(m.label) for m in at.metric]
        assert "引用逐字命中" in [str(m.label) for m in at.metric]

    def test_bad_quote_kills_the_number_not_just_a_note(self, online, monkeypatch):
        """⭐ 引用全部对不上 ⇒ 整块作废 ⇒ 判出方向必须是 0，而不是"打了折的比例"。"""
        self._three_items(monkeypatch)
        _patch_llm(monkeypatch, [_label_reply(
            [(1, "bull", "原文里没有这句"), (2, "bull", "也没有"),
             (3, "bear", "还是没有")], report={"主线": "x"})])
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        _open_zhihu(at)
        _enter_llm_mode(at)
        [b for b in at.button if "开始标注" in str(b.label)][0].click().run()
        assert _metric(at, "判出方向") == "0 条"
        assert _metric(at, "多空比") == "—"
