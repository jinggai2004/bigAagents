"""UI 页面冒烟测试（Streamlit AppTest）

为什么需要：`ui/` 此前没有任何测试，页面级错误只能靠人肉点开才发现。两个真实例子：
  - 给调度新增 `intraday_watch`（cron 的 hour/minute 是字符串 "9-11,13-15" / "*"）后，
    系统配置页把它们按整数格式化 → 页面直接 ValueError 崩溃；
  - 一次手滑编辑把代码行并进了注释行，页面同样会崩。
用 Streamlit 官方的 AppTest 真跑每个页面的初始渲染，成本很低，但能拦住整类回归。

范围约定：只断言"能无异常渲染"与关键提示文案，不断言样式细节；
需要点击 + 真实网络/大模型调用的交互路径不在冒烟范围内。
"""

import glob
import os
import time

import pytest
from streamlit.testing.v1 import AppTest

import core.ui_runs as ui_runs

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PAGES_DIR = os.path.join(_ROOT, "ui", "pages")
PAGES = sorted(p for p in glob.glob(os.path.join(_PAGES_DIR, "*.py"))
               if not os.path.basename(p).startswith("__"))
SCREEN_PAGE = os.path.join(_PAGES_DIR, "1_📊_选股.py")
ANALYSIS_PAGE = os.path.join(_PAGES_DIR, "2_🔍_深度分析.py")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """UI 任务目录与 SQLite 都指向临时位置。

    多个页面会读写数据库（持仓、系统配置、模拟账户），测试必须隔离，
    否则会在真实的 data/portfolio.db 里留下测试数据。

    ⚠️ 这里**不能只改路径**，必须给一个**有表结构的**库（走 use_template_db
    从空库模板复制）。页面用的是进程级单例 `get_portfolio_manager()`，
    而单例只在**首次创建**时调 `init_db()`：同一进程里第二个用例换了库路径后
    单例不会再建表 → 新库是空的 → 页面读 positions 报 `no such table`。
    表现为**单跑通过、串行必挂、-n 4 偶发**（取决于 xdist 把用例分给哪个
    worker，所以它曾长期以"全绿"的假象存在）。
    """
    monkeypatch.setattr(ui_runs, "UI_RUNS_DIR", str(tmp_path / "ui_runs"))
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "ui.db")
    # AppTest 会真的执行 st.fragment(run_every=...)：不关掉的话一个"任务在跑"的
    # 页面会让测试空转轮询十几秒。调大间隔即可（片段仍会渲染一次）。
    monkeypatch.setenv("BIGA_UI_POLL_SECONDS", "3600")
    # 热榜等纯联网区块跳过（AppTest 会真执行页面脚本，联网纯属浪费/挂起）
    monkeypatch.setenv("BIGA_UI_OFFLINE", "1")
    # ⚠️⚠️ 必须清掉 `st.cache_data`：它是**进程级全局**缓存，
    #    不随夹具换库而失效 ⇒ 前一个用例读到的"空结果"会被后一个用例复用，
    #    表现为**"单跑通过、串行失败"**（2026-10-15 实测：给页面加缓存后两个用例立刻红，
    #    错误信息却是"页面没渲染出内容"，极易被误判成 DOM/断言问题）。
    import streamlit as st
    st.cache_data.clear()
    # ⚠️ 盘中补仓开关是**真实运行状态**（temp/ai_wake_buy_state.json）；页面「设置」tab 上
    #    的按钮会写它。不重定向的话，将来任何"点一下按钮"的用例都会**真的把线上功能关掉**。
    from core import ai_league as _al
    monkeypatch.setattr(_al, "_wake_buy_state_path",
                        lambda: tmp_path / "wake_buy_state.json")
    import core.ui_tasks as ui_tasks
    ui_tasks.reset_for_tests()
    yield
    st.cache_data.clear()
    ui_tasks.reset_for_tests()


def test_isolated_fixture_provides_schema():
    """夹具必须给出**有表结构**的库，而不只是一个路径。

    这是上一条注释所描述故障的守护：把 use_template_db 换回
    `monkeypatch.setattr(models, "DB_PATH", ...)`（只改路径不建表），
    本用例立刻变红。
    """
    import sqlite3

    import db.models as models

    conn = sqlite3.connect(models.DB_PATH)
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    # positions/trade_logs 来自 db.models.init_db()，scheduled_jobs 来自 db.store
    assert {"positions", "trade_logs", "scheduled_jobs"} <= names, \
        f"UI 夹具给出的库缺表，页面会报 no such table：{sorted(names)}"


def _texts(at) -> str:
    parts = []
    for attr in ("markdown", "caption", "info", "warning", "error", "success"):
        for el in getattr(at, attr, []) or []:
            parts.append(str(getattr(el, "value", "")))
    return "\n".join(parts)


def _headings(at) -> str:
    """所有 st.subheader 的文本。

    ⚠️ `_texts()` **不收集 subheader**（它只看 markdown/caption/info/warning/error/
    success）。本文件已因此踩过 4 次：断言写的是页面上真实存在的字，却因为收集器
    看不到而变红，失败信息看着像"页面没渲染"。新增断言前先问：
    **这句话是用哪个 st.xxx 渲染的？** 再选对应收集器。

    补充：这里**只加不合并**，不去改 `_texts()` 的口径——往 `_texts()` 里加内容会
    让别处所有 `in` 断言都更容易通过，等于悄悄削弱它们。
    """
    return "\n".join(str(getattr(el, "value", ""))
                     for el in (getattr(at, "subheader", None) or []))


def _frames(at) -> str:
    """所有 st.dataframe 的**完整**文本。

    ⚠️ 不要用 `str(el.value)`：那会走 pandas 的 repr，宽表中间列被 `...` 省略
    （实测 7 列时只留下首尾两列，断言 Base URL 之类的中间列必然失败，
    而失败原因看起来像"没渲染出来"，很容易误判成页面 bug）。
    """
    parts = []
    for el in getattr(at, "dataframe", []) or []:
        value = getattr(el, "value", None)
        if value is None:
            continue
        try:
            parts.append(value.to_string())
        except AttributeError:                  # 非 DataFrame（极少见）
            parts.append(str(value))
    return "\n".join(parts)


@pytest.mark.parametrize("path", PAGES, ids=[os.path.basename(p) for p in PAGES])
def test_page_renders_without_exception(path):
    at = AppTest.from_file(path, default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]


class TestScreenPageRehydration:
    """P1 修复：刷新后应能看到上次结果，而不是回到空白页"""

    def test_shows_saved_result_after_reload(self):
        ui_runs.save_result("screen", {
            "skipped": False, "market_signal": None, "degraded": False,
            "candidates": [{"股票代码": "600519", "股票名称": "贵州茅台", "score": 82.5,
                            "策略": "主力资金", "score_coverage": 0.82,
                            "factors": {"主力资金": {"score": 22, "max": 25, "status": "ok",
                                                   "detail": "五维评分 22/25"}}}],
        })
        at = AppTest.from_file(SCREEN_PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "上次保存的选股结果" in text
        assert "600519" in text and "贵州茅台" in text
        assert "82.5" in text          # 评分按一位小数渲染

    def test_shows_live_banner_when_task_in_flight(self):
        """后台任务在跑：页面必须显示「在后台线程里运行」，而不是「刷新会中断」"""
        ui_runs.mark_running("screen", {"sector": "全市场"})
        at = AppTest.from_file(SCREEN_PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "后台线程" in text
        assert "已中断" not in text

    def test_dead_marker_cleaned_and_no_false_running_banner(self):
        """心跳停止的标记必须被清理，且不能再谎报「正在运行」"""
        ui_runs.mark_running("screen")
        data = ui_runs._load_running_raw()
        data["screen"]["heartbeat_ts"] = time.time() - 600
        ui_runs._atomic_write_json(ui_runs.RUNNING_FILE, data)

        at = AppTest.from_file(SCREEN_PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert not at.warning, "心跳已停的标记不应再出现'正在运行'警告"
        assert ui_runs.get_running_any("screen") is None, "死标记应被清理"

    def test_dead_marker_with_stale_progress_reports_crash(self):
        """进程崩溃会留下「进度还在运行中、但心跳已停」的组合，要如实说明"""
        import core.ui_tasks as ui_tasks
        ui_runs.mark_running("screen")
        data = ui_runs._load_running_raw()
        data["screen"]["heartbeat_ts"] = time.time() - 600
        ui_runs._atomic_write_json(ui_runs.RUNNING_FILE, data)
        ui_tasks.write_progress("screen", {"status": "running", "percent": 30.0,
                                           "message": "因子精筛……"})

        at = AppTest.from_file(SCREEN_PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "没有正常结束" in _texts(at)
        assert ui_tasks.read_progress("screen") is None, "崩溃残留的进度应被清掉"

    def test_reading_background_progress_does_not_crash(self):
        """真后台任务在跑时渲染页面：轮询片段必须能正常渲染（无法干预任务）"""
        import threading
        import core.ui_tasks as ui_tasks
        release = threading.Event()

        def job(progress):
            progress({"percent": 45.0, "message": "因子精筛：600519 完成",
                      "current": 3, "total": 10})
            release.wait(10)

        assert ui_tasks.start("screen", job) is True
        try:
            at = AppTest.from_file(SCREEN_PAGE, default_timeout=60)
            at.run()
            assert not at.exception, [str(e.value) for e in at.exception]
            assert "后台线程" in _texts(at)
        finally:
            release.set()
            import time as _t
            for _ in range(200):
                if not ui_tasks.is_running("screen"):
                    break
                _t.sleep(0.02)

    def test_empty_state_when_nothing_saved(self):
        at = AppTest.from_file(SCREEN_PAGE, default_timeout=60)
        at.run()
        assert "开始选股" in _texts(at)


class TestScreenBackgroundExecution:
    """端到端点一次「开始选股」：任务必须真的跑在后台线程里，刷新也不丢结果。

    这里用假的 StockScreener 替换真选股（不联网、不调大模型），
    验证的是 UI 与 core/ui_tasks 的接线，而不是选股逻辑本身。
    """

    def test_click_runs_in_background_thread_and_result_survives_reload(self, monkeypatch):
        import threading
        import core.screener as screener_mod
        import core.ui_tasks as ui_tasks

        entered = threading.Event()
        release = threading.Event()

        class FakeScreener:
            def run(self, sector_filter=None, max_stocks=5, strategies=None, progress_cb=None):
                if progress_cb:
                    progress_cb({"percent": 40.0, "message": "假选股：初筛完",
                                 "current": 1, "total": 2})
                entered.set()
                release.wait(10.0)
                if progress_cb:
                    progress_cb({"percent": 100.0, "message": "假选股：完成",
                                 "current": 2, "total": 2})
                return {"skipped": False, "market_signal": None, "degraded": False,
                        "candidates": [{"股票代码": "600519", "股票名称": "贵州茅台",
                                        "score": 88.0, "策略": "冒烟测试", "factors": {}}]}

        monkeypatch.setattr(screener_mod, "StockScreener", FakeScreener)

        at = AppTest.from_file(SCREEN_PAGE, default_timeout=60)
        at.run()
        start_btn = [b for b in at.button if "开始选股" in str(b.label)]
        assert start_btn, [str(b.label) for b in at.button]
        start_btn[0].click().run()
        assert not at.exception, [str(e.value) for e in at.exception]

        try:
            assert entered.wait(5.0), "选股应被放到后台线程里执行"
            assert ui_tasks.is_running("screen") is True
            assert "后台线程" in _texts(at)
        finally:
            release.set()

        # 等任务跑完（结果落盘），再"刷新页面"重新渲染一遍
        for _ in range(500):
            if not ui_tasks.is_running("screen"):
                break
            time.sleep(0.02)
        assert ui_tasks.is_running("screen") is False, "任务应在后台自行结束"

        at2 = AppTest.from_file(SCREEN_PAGE, default_timeout=60)
        at2.run()
        assert not at2.exception, [str(e.value) for e in at2.exception]
        text = _texts(at2)
        assert "上次保存的选股结果" in text
        assert "贵州茅台" in text
        assert not at2.warning, ("任务已结束，不该再显示'正在运行'："
                               + "｜".join(str(w.value) for w in at2.warning))


class TestAnalysisPageRehydration:
    def test_shows_saved_snapshot_after_reload(self):
        ui_runs.save_result("analysis_000001", {
            "stock_code": "000001", "stock_name": "平安银行", "current_price": 12.5,
            "tech_report": "技术面：均线多头", "fundamental_report": "基本面：估值偏低",
            "news_report": "舆情：中性", "bull_history": ["多头第一轮"],
            "bear_history": ["空头第一轮"], "final_decision": "批准买入",
            "risk_assessment": "仓位建议 10%", "trade_plan": {"direction": "看多"},
            "errors": [], "fatal_errors": [],
        })
        at = AppTest.from_file(ANALYSIS_PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "上次保存的分析结果" in text
        assert "批准买入" in text and "平安银行" in text

    def test_shows_running_banner(self):
        ui_runs.mark_running("analysis_000001", {"stock_code": "000001"})
        at = AppTest.from_file(ANALYSIS_PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "后台线程" in text
        assert "已中断" not in text

    def test_partial_snapshot_rendered_while_running(self):
        """分析进行中也要能看到已完成的节点（分步快照）"""
        import core.ui_tasks as ui_tasks
        ui_runs.mark_running("analysis_000001", {"stock_code": "000001"})
        ui_tasks.write_progress("analysis_000001", {
            "status": "running", "percent": 40.0, "message": "步骤 4: tech_analyst",
            "snapshot": {"stock_code": "000001", "stock_name": "平安银行",
                         "tech_report": "技术面：均线多头排列"},
        })
        at = AppTest.from_file(ANALYSIS_PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "后台线程" in text
        assert "技术面：均线多头排列" in text

    def test_fatal_errors_surface_in_snapshot(self):
        ui_runs.save_result("analysis_000001", {
            "stock_code": "000001", "stock_name": "平安银行",
            "fatal_errors": ["K线: 无数据"], "errors": [], "final_decision": "",
        })
        at = AppTest.from_file(ANALYSIS_PAGE, default_timeout=60)
        at.run()
        assert not at.exception
        assert "致命错误" in _texts(at)


class TestNlScreenPage:
    """自然语言选股页：解析 → 确认 → 执行，且解析失败要显式报错"""

    PATH = os.path.join(_PAGES_DIR, "9_🗣️_自然语言选股.py")

    def _click(self, at, label):
        buttons = [b for b in at.button if label in str(b.label)]
        assert buttons, [str(b.label) for b in at.button]
        buttons[0].click().run()

    def test_empty_state(self):
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "解析条件" in _texts(at)
        assert "解析成了什么" in _texts(at)

    def test_parse_shows_conditions_and_their_layer(self, monkeypatch):
        """解析后回显条件，并标明每条在**哪一层**执行（粗筛 vs 逐票精算）

        改造前这里断言的是"问财语句回显（只读输入框）"。问财链路已按 D1 决策移除
        （见 output/NL选股改进方案v2），所以改为守护新的分层提示
        —— 它比原来的问财语句更重要：用户要知道哪些条件是毫秒级生效的。
        """
        import core.nl_screener as nl
        # 别让测试真的去调模型（本机可能配了密钥）
        monkeypatch.setattr(nl, "make_llm_fn", lambda *a, **k: None)
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        at.text_area(key="nl_text").set_value("市盈率低于20、主力资金净流入为正的新能源股票")
        at.run()
        self._click(at, "解析条件")
        text = _texts(at)
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "市盈率(PE) 小于 20倍" in text
        assert "剔除 ST" in text
        # 分层提示：PE 属快照类（粗筛层），资金流属精算类
        assert "粗筛层" in text, "快照类条件必须标明在粗筛层即刻生效"
        assert "逐票精算" in text, "精算类条件必须标明需逐票精算（覆盖有限）"

    def test_vague_text_reports_failure(self, monkeypatch):
        import core.nl_screener as nl
        monkeypatch.setattr(nl, "make_llm_fn", lambda *a, **k: None)
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        at.text_area(key="nl_text").set_value("找一些最近走势不错的股票")
        at.run()
        self._click(at, "解析条件")
        assert not at.exception, [str(e.value) for e in at.exception]
        errors = " ".join(str(e.value) for e in at.error)
        assert "解析失败" in errors
        assert "走势不错" in errors          # 告诉用户哪段没听懂

    def test_llm_invented_field_surfaces_as_error(self, monkeypatch):
        import core.nl_screener as nl
        monkeypatch.setattr(nl, "make_llm_fn",
                            lambda *a, **k: (lambda s, u: '{"conditions":[{"field":"kdj","op":"lt","value":20}]}'))
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        at.text_area(key="nl_text").set_value("找个kdj低位的")
        at.run()
        self._click(at, "解析条件")
        errors = " ".join(str(e.value) for e in at.error)
        assert "kdj" in errors and "白名单" in errors

    def test_renders_saved_result(self):
        ui_runs.save_result("nl_screen", {
            "query": {}, "conditions": ["市盈率(PE) 小于 20倍"], "wencai": "市盈率小于20倍，非ST",
            "local_only": [], "requested": 5, "skipped": False, "degraded": False,
            "degrade_reasons": [], "insufficient_data": [], "wencai_unavailable": False,
            "candidates": [{"股票代码": "600519", "股票名称": "贵州茅台", "score": 82.5,
                            "nl_report": {"pass": True, "hits": ["市盈率(PE) 小于 20倍：实际 15.3倍"],
                                          "misses": [], "unverified": []}}],
            "passed": [{"股票代码": "600519", "股票名称": "贵州茅台", "score": 82.5,
                        "nl_report": {"pass": True, "hits": ["市盈率(PE) 小于 20倍：实际 15.3倍"],
                                      "misses": [], "unverified": []}}],
        })
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "贵州茅台" in text and "实际 15.3倍" in text

    def test_pool_unavailable_is_reported_not_faked(self):
        """候选池不可用（快照取不到）时必须**明确报错**

        改造前这里断言的是"问财初筛降级时不展示结果"。现在候选池来自本地快照，
        不可用时直接抛 `PoolUnavailable` —— 页面要把它作为**失败**讲清楚，
        而不是展示空结果（那会被读成"没有符合条件的股票"），
        更不能回退到硬编码精选池（拿硬编码股票冒充"按条件选出"）。
        """
        ui_runs.save_result("nl_screen", {
            "conditions": ["总市值 大于 500亿元"],
            "unavailable": True,
            "reason": ("全市场快照不可用（新浪行业明细取数失败），无法进行本地选股。"
                       "本次**不降级到内置精选池**——那样得到的是硬编码股票"
                       "而非按条件选出的结果。请稍后重试。"),
        })
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "无法选股" in text
        assert "不降级" in text
        assert "本地核对后没有股票满足" not in text, \
            "不可用不能说成「没有股票满足」——那是两件事"

    def test_live_task_shows_running_not_crash(self):
        """回归（用户报告的真实故障）：任务**还在后台跑**时，不能报"没有正常结束"。

        自然语言选股在「问财初筛」整段没有进度回调，原先心跳只在进度回调里刷新，
        超过 180 秒就被页面判成"进程被关闭或崩溃"——而任务其实正常在跑，
        页面还会顺手删掉它要用的进度文件。
        """
        import threading
        import core.ui_tasks as ui_tasks

        release = threading.Event()

        def job(progress):
            progress({"percent": 30.0, "message": "问财初筛：自然语言条件"})
            release.wait(10)                 # 之后长时间静默（模拟问财请求耗时）

        assert ui_tasks.start("nl_screen", job) is True
        try:
            at = AppTest.from_file(self.PATH, default_timeout=60)
            at.run()
            assert not at.exception, [str(e.value) for e in at.exception]
            text = _texts(at)
            assert "后台线程" in text, "任务在跑就该显示运行中"
            assert "没有正常结束" not in text, "任务在跑时不得谎报崩溃"
        finally:
            release.set()
            import time as _t
            for _ in range(200):
                if not ui_tasks.is_running("nl_screen"):
                    break
                _t.sleep(0.02)


class TestConfigPage:
    PATH = os.path.join(_PAGES_DIR, "7_⚙️_系统配置.py")

    def test_renders_with_string_cron_fields(self):
        """回归：盘中盯盘任务的 cron 是字符串（"9-11,13-15" / "*"），
        配置页原先按整数格式化会直接崩溃"""
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]

    def test_llm_observability_panel_with_data(self):
        """Phase 6.2：llm_calls 有数据时，配置页要显示调用次数/失败/延迟/token 与用途明细"""
        from db import store
        store.insert_llm_call({"model": "deepseek-chat", "purpose": "agents.tech",
                               "ok": True, "latency_ms": 1200, "prompt_tokens": 100,
                               "completion_tokens": 20, "total_tokens": 120})
        store.insert_llm_call({"model": "qwen-plus", "purpose": "nl_screener.parse",
                               "ok": False, "latency_ms": 400, "error": "429 Too Many Requests"})
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        labels = [str(m.label) for m in at.metric]
        assert "调用次数" in labels and "失败" in labels and "tokens" in labels
        frames = _frames(at)
        assert "agents.tech" in frames and "nl_screener.parse" in frames
        assert "429" in _texts(at) or "429" in frames

    def test_llm_panel_empty_state(self):
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "没有调用记录" in _texts(at)

    def test_model_provider_panel_reads_registry(self, tmp_path, monkeypatch):
        """「模型接入」面板必须来自 config/llm_providers.json，而不是硬编码的两家。

        只断言"页面不崩"是抓不到这个的：把整段面板删掉，页面照样不崩。
        所以这里断言**只可能来自注册表**的字符串（第三家厂商的地址 / 名称 / 密钥变量名）。
        """
        import json
        reg = tmp_path / "llm_providers.json"
        reg.write_text(json.dumps({
            "version": 1,
            "default_provider": "kimi",
            "providers": [{
                "id": "kimi", "name": "月之暗面 Kimi",
                "base_url": "https://api.moonshot.cn/v1",
                "api_key_env": "MOONSHOT_API_KEY",
                "models": ["moonshot-v1-8k"], "match": ["moonshot-*"],
            }],
        }, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setenv("BIGA_LLM_PROVIDERS", str(reg))
        from config import providers as P
        P.invalidate()
        try:
            at = AppTest.from_file(self.PATH, default_timeout=60)
            at.run()
            assert not at.exception, [str(e.value) for e in at.exception]
            frames = _frames(at)
            assert "api.moonshot.cn" in frames, "注册表里的厂商没有渲染出来"
            assert "月之暗面 Kimi" in frames
            assert "MOONSHOT_API_KEY" in frames
            # 下拉里能选到该厂商的模型（默认模型 / 连通性测试都靠它）
            options = " ".join(str(o) for sb in at.selectbox for o in (sb.options or []))
            assert "moonshot-v1-8k" in options
            # 清单文件已存在，就不该再提示"尚未创建"
            assert "尚未创建清单文件" not in _texts(at)
        finally:
            P.invalidate()

    # ── 自定义定时任务（L2 扩展层）──
    #
    # 判别力说明：只断言"页面不崩"抓不到这一段——把整块面板删掉页面也不会崩。
    # 所以断言的是**只可能来自 scheduled_jobs 表**的字符串（任务名、按模式映射的
    # 中文任务类型、排期与下次执行时间）。

    def test_custom_schedule_panel_empty_state(self):
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "还没有自定义任务" in _texts(at)          # st.info
        assert "共 0 个自定义任务" in _texts(at)          # st.caption
        # 区块标题是 st.subheader —— 必须用 _headings 取（_texts 看不到）
        assert "自定义定时任务（改完不用重启）" in _headings(at)

    def test_custom_schedule_panel_lists_job_from_db(self):
        from db import store
        store.upsert_scheduled_job({
            "job_id": "custom_review_late", "name": "复盘补跑",
            "mode": "review", "cron": {"hour": 16, "minute": 5,
                                       "day_of_week": "mon-fri"},
            "enabled": True, "note": "15:45 那次失败时补一次",
        })
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        frames = _frames(at)
        assert "复盘补跑" in frames, "库里的自定义任务没有渲染出来"
        assert "custom_review_late" in frames
        # 任务类型必须是白名单里的**中文标签**（说明 mode → label 映射生效）
        assert "每日复盘（生成报告并推送）" in frames
        # 排期与下次执行时间（16:05）都来自 cron
        assert "周一至周五 16:05" in frames
        assert "还没有自定义任务" not in _texts(at)

    def test_custom_schedule_offers_only_whitelisted_modes(self):
        """只开放白名单内的任务类型——刻意不提供"任意 prompt 定时执行"
        （LLM 是真实成本）。这里锁住选项集合与 JOB_MODES 一致。"""
        from core import scheduled_jobs as sj
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        opts = [str(o) for sb in at.selectbox for o in (sb.options or [])]
        mode_opts = [o for o in opts if "｜" in o]
        # ⚠️ 比对的是**可调度子集**：`schedulable=False` 的只给内置任务用
        #    （2026-10-19 新增的盘后深度分析就是这类），不能出现在下拉里。
        schedulable = {k: v for k, v in sj.JOB_MODES.items()
                       if v.get("schedulable", True)}
        assert len(mode_opts) == len(schedulable), \
            f"可选任务类型与白名单不一致：{mode_opts}"
        assert any("全市场选股" in o for o in mode_opts)
        assert not any("深度分析" in o for o in mode_opts), "LLM 重链路不得开放自建"

    def test_custom_schedule_corrupt_cron_is_surfaced_not_defaulted(self):
        """cron 坏掉时必须显示"定义有误"并给出原因。

        绝不能兜成默认时间——那会让任务在一个用户没设置过的时刻跑起来，
        而且界面上看着一切正常。
        """
        from db import store
        store.upsert_scheduled_job({
            "job_id": "custom_broken", "name": "坏掉的任务",
            "mode": "news", "cron": {"hour": 22, "minute": 0}, "enabled": True,
        })
        conn = store.get_connection()
        conn.execute("UPDATE scheduled_jobs SET cron=? WHERE job_id=?",
                     ("{坏掉的 JSON", "custom_broken"))
        conn.commit()
        conn.close()
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        frames = _frames(at)
        assert "定义有误" in frames
        assert "无法解析" in frames


class TestChatPage:
    """P1 对话历史：会话列表 / 切换回放 / 清空（AppTest 真跑页面脚本）

    锁的是"刷新不丢"这条链路：消息写进 chat_logs、会话登记进 chat_sessions，
    页面靠 URL 的 ?sid= 恢复当前会话。
    """

    PATH = os.path.join(_PAGES_DIR, "5_💬_AI助手.py")

    def _session_buttons(self, at):
        return [b for b in at.button if (b.key or "").startswith("cs_")]

    def test_empty_state(self):
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "暂无历史会话" in _texts(at)

    def test_session_listed_and_replayed_after_switch(self):
        from core import chat_history as ch
        sid = ch.save_turn("sess_a", "帮我看茅台的资金面", "主力净流入 1.2 亿")
        assert sid is True
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        labels = [str(b.label) for b in self._session_buttons(at)]
        assert any("帮我看茅台的资金面" in l for l in labels), labels

        self._session_buttons(at)[0].click().run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "帮我看茅台的资金面" in text and "主力净流入" in text

    def test_clear_current_session_wipes_db_messages(self):
        """「清空对话」必须连库一起清，否则刷新后消息又回来了"""
        from db import store
        from core import chat_history as ch
        ch.save_turn("sess_a", "提问一", "回答一")
        assert store.get_chat_messages("sess_a")

        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        self._session_buttons(at)[0].click().run()      # 先进入该会话
        clear = [b for b in at.button if "清空当前对话" in str(b.label)]
        assert clear, [str(b.label) for b in at.button]
        clear[0].click().run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert store.get_chat_messages("sess_a") == []

    def test_delete_session_removes_it_from_list(self):
        from core import chat_history as ch
        ch.save_turn("gone", "待删除会话", "答")
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        del_btns = [b for b in at.button if (b.key or "").startswith("cd_")]
        assert del_btns, [str(b.label) for b in at.button]
        del_btns[0].click().run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert ch.list_sessions() == []


class TestScreenHistoryPanel:
    """P2：选股页底部的「选股历史与命中率」面板"""

    PAGE = os.path.join(_PAGES_DIR, "1_📊_选股.py")

    def test_empty_state(self):
        at = AppTest.from_file(self.PAGE, default_timeout=90)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "还没有可用于回看的样本" in _texts(at)

    def test_shows_excess_metric_with_samples(self):
        """有已回填样本时要给出胜率与**超额**收益（必须跟大盘比）"""
        from db import store
        store.insert_screen_picks([{
            "pick_date": "2026-09-01", "stock_code": "600519", "stock_name": "贵州茅台",
            "source": "screen", "strategy": "主力资金", "score": 82.0,
            "score_coverage": 0.9, "price_at_pick": 100.0, "rank_in_run": 1,
            "factors": {}, "conditions": {},
        }])
        rid = store.list_screen_picks()[0]["id"]
        store.update_screen_pick_backfill(rid, {
            "ret_5d": 0.05, "bench_ret_5d": 0.01, "backfill_status": "partial"})

        at = AppTest.from_file(self.PAGE, default_timeout=90)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        labels = [str(m.label) for m in at.metric]
        assert "样本数" in labels and "T+5 胜率" in labels and "T+5 超额收益" in labels
        text = _texts(at)
        assert "样本不足 30 条" in text        # 小样本必须显式提示，不能让人当结论


class TestPaperAccountPage:
    """Phase 2 模拟账户页：渲染 + 首次打开自动建账户"""

    PAGE = os.path.join(_PAGES_DIR, "8_💰_模拟账户.py")

    def test_renders_and_creates_account(self):
        from db import store

        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert at.title[0].value == "💰 模拟账户"
        labels = [m.label for m in at.metric]
        # ⚠️ 指标名带估值口径（2026-10-09）：空仓时持仓价覆盖为空 → "成本价口径"。
        #    此前裸"总资产"让用户把成本价估值误读成"账户被重置回 100 万"。
        assert "总资产（成本价口径）" in labels and "可用现金" in labels
        assert store.list_paper_accounts() == ["default"], "首次打开应自动建账户并落库"

    def test_empty_state_hints(self):
        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        text = _texts(at)
        assert "当前空仓" in text
        assert "还没有净值记录" in text

    def test_order_panel_defaults_to_fraction_mode(self):
        """下单默认「按仓位比例」——贴合手动操作习惯（用户 2026-10-09 要求）"""
        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        # ⚠️ radio 的 label/options 不在 _texts 收集范围里 → 用元素断言
        radios = list(at.radio)
        assert radios, "找不到下单方式 radio"
        assert "按仓位比例" in (radios[0].options or []), radios[0].options
        # 比例档位必须齐全（1/4、1/2、全仓等）
        opts = [str(o) for sb in at.selectbox for o in (sb.options or [])]
        assert any("1/4 仓" in o for o in opts), opts
        assert any("1/2 仓" in o for o in opts), opts
        assert any("全仓" in o for o in opts), opts

    def test_submit_disabled_without_complete_input(self):
        """代码/数量/现价不全时提交按钮必须禁用——越界与空单在入口就挡住"""
        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        btns = [b for b in at.button if "提交下单" in (b.label or "")]
        assert btns, "找不到提交按钮"
        assert btns[0].disabled is True

    def test_sell_without_position_is_blocked_at_entry(self):
        """卖出未持有股票：入口提示 + 提交禁用（而不是提交后才拒绝）"""
        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        # 填代码 + 选卖出（session_state 直接注入输入值）
        at.text_input(key="order_code").set_value("600519").run()
        at.selectbox(key="order_direction").set_value("卖出").run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "无该股持仓" in text or "无可卖持仓" in text, text
        btns = [b for b in at.button if "提交下单" in (b.label or "")]
        assert btns and btns[0].disabled is True


class TestReviewPage:
    """每日复盘页：正文可见性回归（2026-09-28）

    报障背景：复盘早已正常生成（`output/daily_reports/latest.md`，15:57 落盘），
    但它此前只出现在「系统配置」页最底部的一个 expander 里 —— 那一页是管 API Key
    和调度的低频配置页，把每个交易日都要看的产物藏在那里，用户在界面上找不到。

    本类锁三件事：
      1. 已生成报告 → 正文真的渲染到页面上（这是报障的核心，不是"页面不崩"）；
      2. 无报告 → 说清"为什么没有、什么时候会有"，不留白；
      3. 指标与报告日期一致、历史存档可选。
    """

    PAGE = os.path.join(_PAGES_DIR, "10_📰_每日复盘.py")

    @pytest.fixture(autouse=True)
    def _review_dir(self, tmp_path, monkeypatch):
        """把报告目录指向临时目录，避免读写真实 output/daily_reports。"""
        d = tmp_path / "daily_reports"
        d.mkdir()
        monkeypatch.setenv("BIGA_REVIEW_DIR", str(d))
        self.dir = d
        yield d

    def _write(self, date: str = None, body: str = "## 大盘概览\n沪指收涨 1.2%，两市成交 1.1 万亿。") -> str:
        """写一份复盘报告（默认用"今天"，避免测试随运行日期变脆）。"""
        if date is None:
            from datetime import datetime
            date = datetime.now().strftime("%Y%m%d")
        p = self.dir / f"review_{date}.md"
        p.write_text(f"# A股每日复盘 {date[:4]}-{date[4:6]}-{date[6:]}\n\n{body}\n", encoding="utf-8")
        return date

    def _latest(self, text: str = "# A股每日复盘\n\n## 大盘概览\n最新一份。") -> None:
        (self.dir / "latest.md").write_text(text, encoding="utf-8")

    def test_empty_state_explains_why_and_when(self):
        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "尚无复盘报告" in text
        assert "15:45" in text, "空状态必须说清什么时候会自动生成，否则用户只能干等"

    def test_generated_report_body_is_visible(self):
        """核心回归：报告生成了，用户在页面上就能看到正文。

        ⚠️ 断言必须用「只可能来自正文」的串。本页自己的说明文字里就含"大盘概览"
        （"…15:45）：大盘概览 → 资金与板块 → …"），拿它做断言等于没断言：
        实测把 `st.markdown(正文)` 整行去掉后，该断言照样通过。
        """
        marker = "BODY-MARKER-沪指收涨1.2%-两市成交1.1万亿"
        self._write(body=f"## 大盘概览\n{marker}。")
        self._latest(f"# A股每日复盘\n\n## 大盘概览\n{marker}。")
        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert marker in text, "报告已生成，但正文没有渲染到页面上"
        assert "A股每日复盘" in text

    def test_metrics_reflect_latest_report(self):
        date = self._write()
        self._latest()
        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        metrics = {str(m.label): str(m.value) for m in at.metric}
        assert metrics.get("最新复盘日期") == f"{date[:4]}-{date[4:6]}-{date[6:]}"
        assert metrics.get("历史报告数") == "1"
        # 有今天的报告 → 不该出现"今日尚未生成"的警告
        assert "尚未生成复盘" not in _texts(at)

    def test_missing_today_warns_and_history_still_readable(self):
        """今天的还没生成，但要能回看历史，且明确告知原因。"""
        old = self._write(date="20260830", body="## 大盘概览\n8月30日的旧报告。")
        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "尚未生成复盘" in text
        assert "8月30日的旧报告" in text, "今日缺失不应挡住历史回看"
        assert f"{old[:4]}-{old[4:6]}-{old[6:]}" in text

    def test_selecting_historical_report_shows_its_body(self):
        """选中历史报告时，看的是那一份、不是 latest。"""
        from datetime import datetime
        today = datetime.now().strftime("%Y%m%d")
        self._write(date="20260830", body="## 大盘概览\n八月三十日：缩量。")
        self._latest("## 大盘概览\n今天是另一份内容。")
        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        # 第二个选项 = 历史那份（第一个是 latest）
        assert len(at.selectbox) >= 1
        opts = list(at.selectbox[0].options)
        assert len(opts) == 2 and "2026-08-30" in opts[1], opts
        at.selectbox[0].select(opts[1]).run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "八月三十日：缩量" in text
        assert "今天是另一份内容" not in text

    def test_observation_tracking_panel_renders(self):
        """预测跟踪面板：无核对记录时说明"攒够数据才有命中率"，不显示假数字。"""
        self._write()
        self._latest()
        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        metrics = {str(m.label): str(m.value) for m in at.metric}
        assert metrics.get("观察点命中率(近30天)") == "—", metrics
        assert "尚无已核对的观察点" in _texts(at)


class TestPositionPage:
    """持仓页：风控巡检接线（2026-09-28）

    背景：`check_stop_loss` / `check_trailing_stops` / `get_exposure` 在 core 层
    早就实现，但**从未被任何地方调用** —— 持仓录了止损价，系统从不检查它有没有
    被打穿。这里锁"页面上真的能看到巡检结果"，而不是只保证页面不崩。

    价格全部 mock（叶子依赖），不联网。
    """

    PAGE = os.path.join(_PAGES_DIR, "6_💼_持仓.py")

    def _mock_prices(self, monkeypatch, price: float):
        """mock 价格来源：PortfolioManager 取价、RiskEngine 的 K线源、实时行情接口。"""
        import pandas as pd
        from unittest.mock import MagicMock
        from core.risk import RiskEngine

        monkeypatch.setattr("core.portfolio.PortfolioManager._get_current_price",
                            lambda self, code: price)
        monkeypatch.setattr("data.realtime.get_realtime_quotes",
                            lambda codes: {c: {"price": price, "name": "平安银行"}
                                           for c in codes})
        engine = RiskEngine()
        f = MagicMock()
        f.get_stock_info.return_value = {"current_price": price}
        n = 30
        f.get_stock_data.return_value = pd.DataFrame({
            "High": [25.5, 25.8, 26.0, 25.3, 25.7] * (n // 5),
            "Low": [24.5, 24.8, 25.0, 24.3, 24.7] * (n // 5),
            "Close": [25.0, 25.5, 25.3, 25.0, 25.5] * (n // 5),
        })
        engine._stock_fetcher = f
        monkeypatch.setattr("core.risk.get_risk_engine", lambda: engine)

    def _seed_breached_position(self):
        """建一笔已跌破止损的持仓（10 元买入、止损 10 元、现价 9 元）。"""
        from core.portfolio import PortfolioManager
        PortfolioManager().open_position(code="000001", price=10.0, quantity=100,
                                         stop_loss=10.0, name="平安银行")

    # ⚠️ `_texts()` 只收集 markdown/caption/info/warning/error/success，
    #    **不含 st.subheader**。用 `assert "风险巡检" in _texts(at)` 会两头都错：
    #    面板明明渲染了却断言失败，而"空仓不应出现面板"那条又会恒真（无效断言）。
    #    所以区块标题必须单独从 at.subheader 取。
    def _headers(self, at) -> str:
        return " ".join(str(s.value) for s in (at.subheader or []))

    def test_no_positions_hides_risk_panel(self):
        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "风险巡检" not in self._headers(at), "空仓时不该出现巡检面板"

    def test_breached_stop_loss_surfaces_on_page(self, monkeypatch):
        """现价 9 元、止损 10 元 → 页面上必须能看到"止损触发"。"""
        self._mock_prices(monkeypatch, 9.0)
        self._seed_breached_position()
        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "风险巡检" in self._headers(at)
        text = _texts(at)
        assert "止损触发" in text, "跌破止损价必须显示在页面上，否则这个功能等于没接"
        assert "000001" in text

    def test_healthy_position_reports_nothing_hit(self, monkeypatch):
        """没触及止损时要说清"无触发"，而不是让用户猜面板为什么是空的。"""
        self._mock_prices(monkeypatch, 11.0)
        self._seed_breached_position()
        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "风险巡检" in self._headers(at)
        text = _texts(at)
        assert "无持仓触及" in text, "未触发时必须有明确说明"
        assert "止损触发" not in text

    def test_atr_button_fills_levels_instead_of_raising(self, monkeypatch):
        """「按 ATR 建议止损/止盈」必须把建议填进输入框，而不是抛 StreamlitAPIException。

        用户报的"持仓界面进行计算止盈价报错"（2026-09-30）就是这条：
        `_apply_atr_suggestion` 直接写 `st.session_state["open_sl"]`/["open_tp"]，
        而这两个 key 是**同一轮里已经创建**的 number_input 的 widget key ——
        Streamlit 禁止在 widget 实例化之后改它的 key：

            StreamlitAPIException: `st.session_state.open_sl` cannot be modified
            after the widget with key `open_sl` is instantiated.

        修法：先存中间态 `_pending_levels`，下一轮在创建 widget **之前**落盘
        （`_drain_pending_levels`）。价格与风控引擎全走 `_mock_prices` 的替身，不联网。
        """
        self._mock_prices(monkeypatch, 25.0)
        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]

        [t for t in at.text_input if "股票代码" in str(t.label)][0].set_value("000001")
        [n for n in at.number_input if "成交价" in str(n.label)][0].set_value(10.0)
        at.run()

        btn = [b for b in at.button if b.key == "open_atr_btn"]
        assert btn, [b.key for b in at.button]
        btn[0].click()
        at.run()

        assert not at.exception, [str(e.value) for e in at.exception]

        sl = [n for n in at.number_input if n.key == "open_sl"][0].value
        tp = [n for n in at.number_input if n.key == "open_tp"][0].value
        assert isinstance(sl, float) and isinstance(tp, float), (sl, tp)
        # 建议必须是"止损在成本下方、止盈在上方"且为正——不是被清成 0 或 None
        assert 0 < sl < 10.0 < tp, (sl, tp)
        assert "_pending_levels" not in at.session_state, \
            "中间态必须已被消费，否则下一轮会再覆盖用户手改的值"
        # 说明行必须真的渲染出来（它只在 _atr_info 与当前代码一致时出现）
        info = _texts(at)
        assert "ATR" in info and "止盈" in info, info[:200]


class TestFactorPage:
    """第 11 页（因子有效性）：两种分析模式的渲染与结论直白性"""

    PATH = os.path.join(_PAGES_DIR, "11_🧪_因子有效性.py")
    TASK = "factor_analysis"

    @staticmethod
    def _robust_payload():
        """一份"动量方向稳健地相反"的结果 —— 与本轮真实实测同形"""
        return {
            "mode": "robustness",
            "summary": [
                {"key": "momentum20", "因子": "20日动量", "方向假设": "值越大越好",
                 "有效配置": 9, "一致": 0, "相反": 9, "符号一致率": 0.0,
                 "平均IC": -0.0788,
                 "判定": "方向稳健：与权重假设**相反**（权重符号疑似装反）"},
                {"key": "ma_bullish", "因子": "均线多头排列", "方向假设": "值越大越好",
                 "有效配置": 9, "一致": 1, "相反": 8, "符号一致率": 0.111,
                 "平均IC": -0.0325,
                 "判定": "方向不稳：随区间/持有期漂移，当前样本不足以定论"},
            ],
            "cells": [
                {"key": "momentum20", "因子": "20日动量", "区间": "2023",
                 "start": "2023-01-01", "end": "2023-12-31", "持有期": 10,
                 "IC均值": -0.0397, "t值": -0.99, "期数": 45, "结论": "不显著"},
                {"key": "momentum20", "因子": "20日动量", "区间": "2024",
                 "start": "2024-01-01", "end": "2024-12-31", "持有期": 20,
                 "IC均值": -0.0918, "t值": -2.65, "期数": 49, "结论": "显著但方向相反"},
            ],
            "periods": [["2023-01-01", "2023-12-31"], ["2024-01-01", "2024-12-31"]],
            "horizons": [10, 20],
            "names": 30,
            "params": {"codes": 30, "step": 5, "groups": 5, "horizons": [10, 20],
                       "factors": ["momentum20"], "periods": [["2023-01-01", "2023-12-31"]]},
        }

    def test_offers_both_modes(self):
        """页面必须同时提供「单区间」与「样本外稳健性」两种模式（否则稳健性只能命令行跑）"""
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        radios = [r for r in (at.radio or [])]
        assert radios, "没有分析类型选择器"
        opts = " ".join(str(o) for o in radios[0].options)
        assert "单区间" in opts and "稳健性" in opts, opts

    def test_robustness_payload_renders_verdict(self):
        """稳健性结果必须把**判定**摆出来（这是用户唯一需要读的一行）"""
        ui_runs.save_result(self.TASK, self._robust_payload())
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _frames(at)
        assert "与权重假设" in text and "装反" in text, text
        assert "0 / 9" in text, "一致率那一列没渲染出来"
        # 两种判定要能区分开：稳健相反 vs 不稳 —— 混成一句话用户就没法行动
        assert "不足以定论" in text, text
        assert "2023" in text and "2024" in text, "IC 矩阵的区间列没出来"

    def test_single_mode_payload_still_renders(self):
        """单区间模式不能被改坏（本轮为加稳健性动过参数区与结果分发）"""
        ui_runs.save_result(self.TASK, {
            "summary": [{"key": "momentum20", "因子": "20日动量", "方向": "越大越好",
                         "IC均值": -0.1292, "ICIR": -0.649, "t值": -5.19,
                         "IC>0比例": 0.219, "期数": 64, "结论": "显著但方向相反"}],
            "layered": {}, "correlation": None, "periods": 64, "names": 50,
            "panel_rows": 3200, "horizon": 20, "step": 5, "groups": 5,
            "unavailable": {}, "snapshot_status": {},
            "params": {"start": "2024-01-01", "end": "2025-06-30", "horizon": 20,
                       "step": 5, "groups": 5, "codes": 50, "factors": ["momentum20"]},
        })
        at = AppTest.from_file(self.PATH, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _frames(at)
        assert "20日动量" in text and "-0.1292" in text, text


class TestAnalysisReportQA:
    """「就这份报告提问」区块（方案 B：把当前 payload 直接注入会话上下文）"""

    @staticmethod
    def _seed():
        ui_runs.save_result("analysis_000001", {
            "stock_code": "000001", "stock_name": "平安银行", "current_price": 12.5,
            "analysis_date": "2026-10-08", "data_through": "2026-10-08",
            "tech_report": "技术面：均线多头", "fundamental_report": "基本面：估值偏低",
            "news_report": "舆情：乐观，评分 9 分",
            "bull_history": ["多头第一轮"], "bear_history": ["空头第一轮"],
            "research_summary": "研究汇总：短期中性偏空",
            "risk_assessment": "仓位 10% vs 上限 25%", "final_decision": "需修改",
            "trade_plan": {"direction": "观望"}, "errors": [], "fatal_errors": [],
        })

    def test_qa_block_is_offered(self):
        """⚠️ 不能断言展开器**标题**：`_texts()` 只看 markdown/caption/info/warning/
        error/success，**不收集 `st.expander` 的 label**（本文件已因此踩过多次）。
        这里断言区块**正文**（那段 caption 是它独有的），标题由下面的 key 断言间接覆盖。
        """
        self._seed()
        at = AppTest.from_file(ANALYSIS_PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "不会写进「AI 助手」页的会话历史" in _texts(at)

    def test_qa_keys_are_bound_to_this_report(self):
        """输入框 key 必须带股票+日期：否则换一只股票/换一份报告会**串上下文**

        （同一只股票同一天可能有多条分析记录，所以 key 不能只用代码。）
        """
        self._seed()
        at = AppTest.from_file(ANALYSIS_PAGE, default_timeout=60)
        at.run()
        keys = [str(t.key) for t in (at.text_input or [])]
        assert any(k.startswith("qa_q_000001_2026-10-08") for k in keys), keys

    def test_separate_from_chat_page_history(self):
        """本页问答用独立的 session key，不能写进「AI 助手」页的会话历史"""
        self._seed()
        at = AppTest.from_file(ANALYSIS_PAGE, default_timeout=60)
        at.run()
        assert "chat_history" not in at.session_state, list(at.session_state.keys())


class TestAnalysisTradePlanDisplay:
    """交易计划三档显示（看多=可执行 / 看空=驳回 / 观望=等待）（2026-10-09）

    实测背景：报告给出「看空 + 目标在下 + 止损在上」（融券卖出结构），
    而下游全部放行。用户定的设计是：**看空 = 驳回投资建议 + 给入场条件**，
    所以页面上必须一眼看到「建议：驳回」与「入场条件」，而不是一段裸 JSON。
    """

    # ⚠️ 键名必须是 `analysis_{页面输入框的默认代码}` —— 页面用
    #    `TASK = f"analysis_{stock_code}"` 取结果，写错键名页面就什么都不显示
    #    （表现为"还没开始分析"，而不是报错）。本文件其他用例统一用 000001。
    SEED_CODE = "analysis_000001"

    @staticmethod
    def _seed(plan):
        ui_runs.save_result(TestAnalysisTradePlanDisplay.SEED_CODE, {
            "stock_code": "000001", "stock_name": "测试股", "current_price": 52.06,
            "analysis_date": "2026-10-09", "data_through": "2026-10-08",
            "tech_report": "T", "fundamental_report": "F", "news_report": "N",
            "bull_history": ["多头"], "bear_history": ["空头"],
            "research_summary": "R", "risk_assessment": "风控",
            "final_decision": "驳回", "trade_plan": plan,
            "errors": [], "fatal_errors": [],
        })

    def _render(self):
        at = AppTest.from_file(ANALYSIS_PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        return _texts(at)

    def test_bearish_shows_rejection_and_conditions(self):
        self._seed({"direction": "看空", "advice_status": "驳回",
                    "position_size": "0%",
                    "entry_conditions": ["站上MA20且放量", "MACD金叉"]})
        text = self._render()
        assert "投资建议：驳回" in text, text
        assert "站上MA20且放量" in text, text
        assert "不等于「可以做空」" in text or "可以做空" in text, text

    def test_waiting_shows_conditions(self):
        self._seed({"direction": "观望", "advice_status": "等待",
                    "entry_conditions": ["回踩MA20 44.96 且缩量"]})
        text = self._render()
        assert "投资建议：等待" in text, text
        assert "回踩MA20 44.96 且缩量" in text, text

    def test_runnable_shows_direction(self):
        self._seed({"direction": "看多", "advice_status": "可执行",
                    "entry_range": [46.0, 47.5], "target_price": 55.0,
                    "stop_loss": 43.5, "position_size": "10%"})
        text = self._render()
        assert "投资建议：可执行" in text, text

    def test_missing_conditions_warns(self):
        """驳回/等待却没给入场条件 → 必须提示（否则这条建议无法执行）"""
        self._seed({"direction": "看空", "advice_status": "驳回",
                    "entry_conditions": []})
        text = self._render()
        assert "未给出「入场条件」" in text, text

    def test_legacy_plan_without_status_still_renders(self):
        """旧产物（没有 advice_status）也要能渲染 —— 方向推断兜底"""
        self._seed({"direction": "看空", "entry_range": [51.5, 52.3],
                    "target_price": 44.96, "stop_loss": 53.8})
        text = self._render()
        assert "投资建议：驳回" in text, text


# ================================================================
# 热点新闻页（2026-10-09 热点新闻工作流）
# ================================================================

NEWS_PAGE = os.path.join(_PAGES_DIR, "12_📰_热点新闻.py")


class TestNewsPageSmoke:
    def test_empty_state_renders_without_review(self):
        """模板库无评估结果 → 显示引导信息而非报错/空白"""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(NEWS_PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "还没有评估结果" in text, text
        assert "每天 **15:55**" in text or "15:55" in text

    def test_review_verdicts_and_evidence_render(self, monkeypatch):
        from streamlit.testing.v1 import AppTest
        from db import store as db_store
        db_store.save_news_reviews("2026-10-09", [
            {"sector": "科技", "verdict": "利好", "summary": "算力需求旺盛",
             "evidence": [{"id": 1, "title": "芯片订单增长"}], "news_count": 12},
            {"sector": "金融", "verdict": "利空", "summary": "息差承压",
             "evidence": [], "news_count": 3}])
        from core import news_industry_review as nir
        # ⚠️ 推荐区会真联网拉行业成分——冒烟测试必须 stub 掉（确定性 + 不耗时）
        monkeypatch.setattr(nir, "recommend_stocks", lambda top_n=5: {
            "status": "no_bullish", "review_date": "2026-10-09", "picks": [],
            "degraded": False, "note": "测试中禁用"})
        at = AppTest.from_file(NEWS_PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        # ⚠️ 结论在 **expander 标签**里（🟢 科技 —— 利好｜summary），
        #    `_texts` 只收 markdown/caption 等 body 元素 → 标签要单独收
        labels = "\n".join(str(e.label) for e in at.expander)
        assert "科技" in labels and "利好" in labels
        assert "算力需求旺盛" in labels
        assert "芯片订单增长" in text           # 证据必须可见（可追溯）
        assert "评估日期：**2026-10-09**" in text

    def test_picks_render_with_reason(self, monkeypatch):
        """推荐行必须带行业理由与免责——不是裸代码列表"""
        from streamlit.testing.v1 import AppTest
        from core import news_industry_review as nir
        monkeypatch.setattr(nir, "recommend_stocks", lambda top_n=5: {
            "status": "ok", "review_date": "2026-10-09",
            "picks": [{"stock_code": "300001", "stock_name": "测试股",
                       "industry": "科技", "net_3d": 5e7, "change_pct": 2.0,
                       "sector_summary": "利好行业测试", "evidence": []}],
            "degraded": False, "warnings": [], "flow_date": "2026-10-09",
            "note": "排序参考"})
        at = AppTest.from_file(NEWS_PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "300001" in text and "测试股" in text and "利好行业测试" in text

    def test_deep_analysis_accepts_prefill_from_session_state(self):
        """热点页的「深度分析」按钮靠 `prefill_code` 预填——必须真的生效。

        ⚠️ Streamlit 的坑：写在 value= 参数上时，key 一旦有了自己的状态，
        后续 prefill 会被静默忽略；正确做法是在 widget 创建前写 session_state。
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(ANALYSIS_PAGE, default_timeout=60)
        at.session_state["prefill_code"] = "600519"
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        ti = [t for t in at.text_input if t.key == "analysis_code_input"]
        assert ti, "找不到股票代码输入框"
        assert ti[0].value == "600519", ti[0].value


# ================================================================
# 今日决策三块 + 同日同票去重（2026-10-10 工作流改进）
# ================================================================

class TestReviewPageDecisionBlocks:
    """复盘页必须**实时**渲染今日决策三块 —— 不依赖 LLM，也不依赖报告已生成。"""

    PAGE = os.path.join(_PAGES_DIR, "10_📰_每日复盘.py")

    @pytest.fixture(autouse=True)
    def _review_dir(self, tmp_path, monkeypatch):
        d = tmp_path / "daily_reports"
        d.mkdir()
        monkeypatch.setenv("BIGA_REVIEW_DIR", str(d))
        yield d

    def test_decision_blocks_render_even_without_report(self):
        at = AppTest.from_file(self.PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        blob = _texts(at) + _headings(at)
        assert "今日决策（实时" in blob, blob
        for sec in ("今日决策清单", "决策质量评估", "明日待触发"):
            assert sec in blob, blob


class TestAnalysisDuplicateHint:
    """同日同票重复分析必须在**入口**提示。

    实测背景：昭衍新药在 10-08 一天内被分析 4 次，结论互相矛盾（观望/看空/观望），
    既浪费 LLM 成本又污染胜率统计（同一只票一天贡献 4 个样本）。
    """

    def test_hint_appears_when_analyzed_today(self):
        from datetime import datetime
        from db import store as db_store
        db_store.insert_analysis({
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "stock_code": "600519", "stock_name": "贵州茅台",
            "decision_label": "观望", "source": "ui", "trade_plan": {}})
        at = AppTest.from_file(ANALYSIS_PAGE, default_timeout=60)
        at.run()
        at.text_input(key="analysis_code_input").set_value("600519").run()
        assert not at.exception, [str(e.value) for e in at.exception]
        blob = _texts(at)
        assert "今日已分析过 600519" in blob, blob
        assert any("仍要重新分析" in str(b.label) for b in at.button), \
            [str(b.label) for b in at.button]

    def test_no_hint_without_todays_analysis(self):
        at = AppTest.from_file(ANALYSIS_PAGE, default_timeout=60)
        at.run()
        at.text_input(key="analysis_code_input").set_value("600519").run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert "今日已分析过" not in _texts(at)


# ================================================================
# 知乎话题参考区（2026-10-14）
# ================================================================
#
# 三条必须被固化下来的约束（否则以后有人"顺手"把它接进分析链就没人拦得住）：
#   ① **隔离**：不注入提示词、不参与 agents.news / news_industry_review、不参与评分
#   ② **离线可跳过**：AppTest 会真执行页面脚本，联网区块必须能被 BIGA_UI_OFFLINE 跳过
#   ③ **能力边界要写在页面上**：读不了指定大V（授权模型限制），改为按话题关键词检索

class TestZhihuSectionSmoke:
    def test_section_renders_and_declares_isolation(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(NEWS_PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "知乎话题参考" in text, text
        # 隔离声明必须可见 —— 这是用户明确要求的边界
        assert "不参与任何模型分析与评分" in text
        assert "仅供人工参考" in text

    def test_skipped_in_offline_mode(self):
        """离线模式下必须**不联网**并显式说明（与热榜同口径）"""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(NEWS_PAGE, default_timeout=60)
        at.run()
        text = _texts(at)
        assert "（离线模式：知乎话题参考已跳过——这只影响这一节）" in text, text


# ================================================================
# AI 交易联赛区块（2026-10-14，S6）
# ================================================================
#
# 三条必须固化的约束：
#   ① **口径 2 的 caveat 必须可见**（日内按扫描价成交 ⇒ 收益被高估）——
#      与热点页知乎区块"边界声明不能因区块被跳过就消失"是同一条纪律；
#   ② 样本门槛与"不用胜率"的口径说明必须出现；
#   ③ 本区块**只读**（不触发 LLM、不联网）⇒ 离线模式下也应正常渲染。

class TestAiLeagueSectionSmoke:
    def test_league_section_renders_with_caveats(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(os.path.join(_PAGES_DIR, "8_💰_模拟账户.py"),
                               default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "AI 交易联赛" in text, text[:600]
        # ⚠️ 口径 2 的高估 caveat 必须展示（不能只在文档里写）
        assert "高估" in text
        assert "统计意义" in text
        assert "不用胜率" in text

    def test_account_selector_lists_ai_personas(self):
        """账户选择器必须包含 AI 人格账户，且**默认账户仍在**（不剥夺原有功能）。"""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(os.path.join(_PAGES_DIR, "8_💰_模拟账户.py"),
                               default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert at.selectbox, "应有账户选择器"
        opts = list(at.selectbox[0].options)
        assert "default" in opts, "手工账户不能被剥夺"
        assert any(o.startswith("ai_") for o in opts), opts


# ================================================================
# AI 交易联赛页：每日复盘 + 设置（2026-10-14，P1）
# ================================================================
#
# 三条纪律：
#   ① 复盘区必须**同时**出现"AI 自述"与"账本事实"（两者分开呈现，不合并）；
#   ② 今日操作理由要能看到（order_reason 原话）；
#   ③ 设置区必须列出全部人格，且**离线时手动确认入口被跳过**（不联网）。

AI_LEAGUE_PAGE = os.path.join(_PAGES_DIR, "13_🤖_AI交易联赛.py")


class TestAiLeagueReviewSettingsPage:
    def test_review_shows_reason_and_split_view(self):
        from datetime import datetime

        from db import store
        # ⚠️ 必须用**当天**日期落库：页面复盘 tab 默认选"今天"，
        #    写死 2026-10-14 会让用例只在当天通过（"测试不能把本机状态当契约"——本项目踩过）。
        today = datetime.now().strftime("%Y-%m-%d")
        store.save_ai_decision({"persona": "ai_value", "signal_date": today,
                                "code": "600028", "advice_status": "可执行",
                                "direction": "看多", "position_size": 0.2,
                                "order_reason": "原油涨价利好炼化（测试理由）"})
        at = AppTest.from_file(AI_LEAGUE_PAGE, default_timeout=90)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "每日复盘" in text
        assert "原油涨价利好炼化（测试理由）" in text       # 今日操作理由
        assert "AI 自述复盘" in text and "账本事实" in text  # 两栏分开呈现

    def test_settings_lists_all_personas_and_offline_note(self):
        at = AppTest.from_file(AI_LEAGUE_PAGE, default_timeout=90)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "AI 设置" in text
        markdown_all = "\n".join(str(m.value) for m in at.markdown)
        for pname in ("价值派", "技术派", "风控流", "随机对照 1"):
            assert pname in markdown_all, f"设置区缺人格: {pname}"
        # 离线模式：手动确认入口必须被跳过（不联网）
        assert "离线模式：已跳过手动确认入口" in text


# ================================================================
# 区块快速定位下拉框（2026-10-15，用户提议）
# ================================================================
# 长页面（热点新闻 5 个区块 / 系统配置更长）靠滚动找区块很费劲 ⇒ 顶部下拉框切换。
#
# 三条必须钉住的约束：
#   ① **默认「全部显示」**：改造前后行为一致 —— 否则会静默砍掉一部分内容；
#   ② 选中某区块 ⇒ **只渲染那一块**（其余不出现在 DOM 里）；
#   ③ **全局口径常驻**：判断依据必须放在下拉框之上，任何选择下都可见
#      （旧事故是"边界声明因区块被跳过而消失"）。

class TestSectionNavOnNewsPage:
    def _render(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(NEWS_PAGE, default_timeout=60)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        return at

    @staticmethod
    def _nav(at):
        """取**左侧栏**导航控件。

        ⚠️ 2026-10-15 用户要求改为"左侧主板块下挂副模块" ⇒ 由页面顶部 selectbox
           改为侧栏 expander(主板块) + radio(副模块)。断言必须跟着结构走，
           不能再去 `at.selectbox[0]` 抓（那个位置现在是"聚合窗口"等业务下拉框，
           抓错了会**看似通过、实际验的是别的东西**）。
        """
        from ui.components.section_nav import ALL
        boxes = [r for r in at.sidebar.radio if ALL in list(r.options)]
        assert boxes, "左侧栏应有「主板块 → 副模块」导航"
        return boxes[0]

    def test_sidebar_nav_exists_and_defaults_to_show_all(self):
        from ui.components.section_nav import ALL
        at = self._render()
        box = self._nav(at)
        assert list(box.options)[0] == ALL
        assert box.value == ALL, "默认必须是「全部显示」（保持改造前行为）"
        text = _texts(at)
        # 默认下：评估区与知乎区**同时**可见（各自的关键文案都在）
        assert "还没有评估结果" in text
        assert "知乎话题参考已跳过" in text

    def test_selecting_a_section_renders_only_that_block(self):
        at = self._render()
        self._nav(at).set_value("🗞️ 全市场快讯").run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "该筛选条件下没有快讯" in text or "最多显示 300 条" in text
        # 其余区块不得出现
        assert "还没有评估结果" not in text
        assert "知乎话题参考已跳过" not in text
        assert "热榜已跳过" not in text

    def test_global_caveat_survives_any_selection(self):
        """⭐ 全局口径（推荐 ≠ 交易指令）必须常驻 —— 它在导航**之外**的主区域。"""
        at = self._render()
        self._nav(at).set_value("💬 知乎话题参考").run()
        assert not at.exception, [str(e.value) for e in at.exception]
        text = _texts(at)
        assert "不构成交易指令" in text, "全局口径不能因切换区块而消失"
        assert "知乎话题参考已跳过" in text
        assert "还没有评估结果" not in text


class TestSectionNavComponent:
    def test_visible_predicate(self):
        from ui.components.section_nav import ALL, visible
        assert visible(ALL, "任意区块") is True
        assert visible("A", "A") is True
        assert visible("A", "B") is False


class TestSectionNavOnOtherPages:
    """铺开到看盘/回测看板/模拟账户后的**默认行为守卫**。

    ⚠️ 只钉"侧栏导航存在 + 默认全部显示"：这三页的既有断言语义由各自用例覆盖，
       这里再加一遍是重复；但"默认值"必须独立守住 —— 一旦有人把默认改成某个区块，
       页面上会**静默少掉一部分内容**（而且现有断言未必覆盖得到那一块）。
    ⚠️ 2026-10-15：导航从页面顶部 selectbox 迁到**左侧栏 radio**，断言随之改变。
    """

    @pytest.mark.parametrize("page", ["3_📊_看盘.py", "4_📈_回测看板.py", "8_💰_模拟账户.py"])
    def test_defaults_to_show_all(self, page):
        from streamlit.testing.v1 import AppTest

        from ui.components.section_nav import ALL
        at = AppTest.from_file(os.path.join(_PAGES_DIR, page), default_timeout=90)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        boxes = [r for r in at.sidebar.radio if ALL in list(r.options)]
        assert boxes, f"{page} 左侧栏应有「主板块 → 副模块」导航"
        assert boxes[0].value == ALL, f"{page} 的区块定位默认值必须是「全部显示」"


# ================================================================
# 左侧栏「主板块 → 副模块」导航（2026-10-15 用户要求改版）
# ================================================================
# 用户原话："下拉应放左侧主板块下，点开某一主板块即可看到它下面的副模块"。
# ⚠️ 三条必须钉住：
#   ① 导航必须在**侧栏**、且是**折叠分组(主板块)+单选(副模块)**的形态；
#   ② 默认仍是「全部显示」（否则页面会**静默少掉一部分内容**）；
#   ③ **不能**再退回"页面顶部下拉框"（否则又变成"滚到顶才能切"）。

class TestSidebarSectionNav:
    @pytest.mark.parametrize("page,name", [
        ("3_📊_看盘.py", "📊 看盘"),
        ("8_💰_模拟账户.py", "💰 模拟账户"),
        ("12_📰_热点新闻.py", "📰 热点新闻"),
    ])
    def test_nav_lives_in_sidebar_expander(self, page, name):
        from streamlit.testing.v1 import AppTest

        from ui.components.section_nav import ALL
        at = AppTest.from_file(os.path.join(_PAGES_DIR, page), default_timeout=90)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        radios = [r for r in at.sidebar.radio if ALL in list(r.options)]
        assert radios, f"{page} 的副模块单选必须在**侧栏**"
        assert radios[0].value == ALL, f"{page} 的副模块默认必须是「全部显示」"
        # 主板块 = 侧栏折叠分组的标题（点开才看到副模块）
        labels = [str(e.label) for e in at.sidebar.expander]
        assert any(name in lb for lb in labels), f"{page} 缺少主板块分组：{labels}"
        # 不得再退回顶部下拉框
        assert not [s for s in at.selectbox if ALL in list(s.options)], \
            f"{page} 不应再有顶部「全部显示」下拉框"

    def test_cheaper_switch_does_not_refetch_unselected_blocks(self):
        """⭐ 只渲染被选中的副模块 ⇒ 未选中的区块**一行都不执行**。

        用"离线模式下知乎区块的跳过提示"当探针：选中快讯时它必须**不出现**
        （若出现，说明其他区块也在跑，切换自然慢）。
        """
        from streamlit.testing.v1 import AppTest

        from ui.components.section_nav import ALL
        at = AppTest.from_file(NEWS_PAGE, default_timeout=60)
        at.run()
        nav = [r for r in at.sidebar.radio if ALL in list(r.options)][0]
        nav.set_value("🗞️ 全市场快讯").run()
        text = _texts(at)
        assert "知乎话题参考已跳过" not in text
        assert "该筛选条件下没有快讯" in text or "最多显示 300 条" in text


class TestWakeBuySettingsPanel:
    """页13「设置」必须能看到**盘中补仓开关状态与回退判据**（用户要求可复位）。

    ⚠️ 这里的按钮会写真实状态文件 —— 夹具已把 `_wake_buy_state_path` 重定向到 tmp，
       所以点按钮不会影响线上（本用例只读不点，双重保险）。
    """

    def test_status_and_reset_controls_visible(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(AI_LEAGUE_PAGE, default_timeout=90)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        md = "\n".join(str(m.value) for m in at.markdown) + "\n" + _texts(at)
        assert "盘中补仓" in md
        assert "硬边界" in md and "price_below" in md
        assert "回退判据" in md
        labels = [b.label for b in at.button]
        assert any("人工复位" in lb for lb in labels), labels
        assert any("人工关闭" in lb for lb in labels), labels
