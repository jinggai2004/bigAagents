# -*- coding: utf-8 -*-
"""P4/P5 契约测试：助手（WS 流式）/ 系统页 / 写操作 / P6 打包脚本守卫。

## 守的五条线

1. **WS 一定以 `done` 收尾**。前端靠 `done` 解锁输入框；没有 `done` 的失败
   （生成器抛异常、线程崩了）会让输入框**永久灰住**，比报错更难受。
2. **`ui_runs.load_result` 返回的是信封**（`{task, saved_ts, generated_at, payload}`），
   不是 payload 本体。前端读 `data.result.picks` ⇒ 后端必须解包。这一条是真实
   踩到的 bug：不解包时选股页永远显示"无候选"、知乎快照永远"尚无采集结果"。
3. **写操作的"被拒"不是 error**：现金不足/无持仓/已在运行是业务结果，要能显示成
   红色提示，而不是"接口坏了"。
4. **系统页不外呼**：`/api/system/models` 只报"密钥配没配"，不探测连通性
   （每 10s 刷一次的页面不能打限频）。
5. **任务名带 `web_` 前缀**（C5）：与 Streamlit 页同名会互相覆盖进度/触发取消。

## 为什么不 import `scripts/launch_exe.py`
它在**模块级**写了 `os.environ.setdefault("BIGA_WARMUP", "1")`。测试进程 import 一次，
后面所有用例的 `TestClient` 就可能真的去预热（联网取指数）⇒ 随机变红。
所以 P6 部分只做**静态文本守卫**，不 import。
"""
from __future__ import annotations

import json
import os
import re

import pytest
from fastapi.testclient import TestClient

# ⚠️⚠️ **预导入**：下面这些模块的数据根是"**导入时求值**"的模块常量
#    （`_PROJECT_ROOT = config.paths.data_root()` 在 import 那一刻算一次）。
#    如果等到某个 patch 了 `paths.app_root` 或 `monkeypatch.setenv("BIGA_DATA_ROOT")`
#    的用例里才首次导入，常量会被**永久**绑到那个临时目录——monkeypatch 只还原
#    环境变量，还原不了已经算出来的常量。后果是后续用例读到空数据：
#    实测 2026-10-25 全量出现 `test_market_snapshot::test_yi_to_yuan_conversion`
#    失败（assert 5561 == 2）——因为它读不到本地面板缓存，回退联网抓到 5561 行真数据。
#    在模块级先导入 ⇒ 它们用**干净环境**求值一次，之后 patch 与本文件隔离。
#    （`TestDataRootResolution` 只测 `config.paths` 的纯函数，不受影响。）
import core.backtest            # noqa: F401,E402
import core.pipeline            # noqa: F401,E402
import core.process_freshness   # noqa: F401,E402
import core.review_agent        # noqa: F401,E402
import core.scheduler_health    # noqa: F401,E402
import core.stage_health        # noqa: F401,E402
import core.ui_runs             # noqa: F401,E402
import data.bars                # noqa: F401,E402
import data.index_bars          # noqa: F401,E402
import data.panel_store         # noqa: F401,E402
import data.watchlist           # noqa: F401,E402

# 与 `core.scheduler.JOBS` 对齐的期望条数（新增定时任务时**必须**同步改这里，
# 否则系统页会悄悄少一行而没人发现）
EXPECTED_JOBS = 18


# ================================================================
# 夹具
# ================================================================
@pytest.fixture
def client():
    from web.server.app import app
    import web.server.cache as C
    C.clear()
    yield TestClient(app)
    C.clear()


@pytest.fixture
def root(tmp_path, monkeypatch):
    """临时数据根。

    ⚠️ `core.ui_runs.UI_RUNS_DIR` 是**模块导入时**算出来的常量（不是每次运行时
       读环境变量），所以只设 `BIGA_DATA_ROOT` 并不能把进度/结果文件挡在项目外——
       必须同时 patch 它。否则用例会往真实 `output/ui_runs/` 里写 `web_screen.json`
       之类的文件，把用户刚跑出来的结果覆盖掉。
    """
    import core.ui_runs as ur
    import web.server.cache as C

    C.clear()
    (tmp_path / "output" / "ui_runs").mkdir(parents=True)
    (tmp_path / "output" / "daily_picks").mkdir(parents=True)
    (tmp_path / "temp").mkdir()
    monkeypatch.setenv("BIGA_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(ur, "UI_RUNS_DIR", str(tmp_path / "output" / "ui_runs"))
    yield tmp_path
    C.clear()


@pytest.fixture
def db(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    return use_template_db(tmp_path, monkeypatch)


def _ok(z: dict) -> dict:
    assert z.get("ok") is True, z
    return z["data"]


def _fail(z: dict, kind: str) -> dict:
    assert z.get("ok") is False, z
    assert (z.get("error") or {}).get("type") == kind, z
    return z["error"]


# ================================================================
# 助手：会话（P4）
# ================================================================
class TestChatSessions:
    def test_personas_available(self, root, client):
        from core.chat_agent import PERSONAS
        data = _ok(client.get("/api/chat/personas").json())
        assert set(data["personas"]) == set(PERSONAS.keys())
        assert "综合" in data["personas"]

    def test_sessions_empty_is_ok_not_error(self, root, db, client):
        """一次都没聊过 ⇒ 空列表 + ok:true（不是 missing）。"""
        data = _ok(client.get("/api/chat/sessions").json())
        assert data["sessions"] == []

    def test_create_then_read_back(self, root, db, client):
        created = _ok(client.post("/api/chat/sessions", params={"persona": "综合"}).json())
        sid = created["session_id"]
        assert sid
        data = _ok(client.get(f"/api/chat/sessions/{sid}").json())
        assert data["session_id"] == sid
        assert data["messages"] == []

    def test_unknown_session_is_missing(self, root, db, client):
        _fail(client.get("/api/chat/sessions/nope-does-not-exist").json(), "missing")

    def test_delete(self, root, db, client):
        sid = _ok(client.post("/api/chat/sessions").json())["session_id"]
        data = _ok(client.delete(f"/api/chat/sessions/{sid}").json())
        assert data["session_id"] == sid
        _fail(client.get(f"/api/chat/sessions/{sid}").json(), "missing")

    def test_list_reports_label_and_count(self, root, db, client):
        """列表必须带可读 label（前端左栏直接显示它），不能只有一串 uuid。"""
        sid = _ok(client.post("/api/chat/sessions").json())["session_id"]
        from core import chat_history
        chat_history.save_turn(sid, "招商银行怎么看", "先看息差。", usage={},
                              persona="综合", model="test", source="web")
        rows = _ok(client.get("/api/chat/sessions").json())["sessions"]
        row = next(r for r in rows if r["session_id"] == sid)
        assert row["label"]
        assert row["messages"] == 2


# ================================================================
# 助手：WebSocket 流式（P4）
# ================================================================
class TestChatStream:
    """⚠️ 这里只测**不调模型**的路径（空消息 / 未知视角 / 非法首帧）。

    真跑一轮要 5~40s 且花钱，属于本地手工验证，不进 CI。
    """

    def _collect(self, ws) -> list[dict]:
        out = []
        while True:
            ev = ws.receive_json()
            out.append(ev)
            if ev["type"] == "done":
                return out

    def test_empty_message_errors_then_done(self, root, db, client):
        with client.websocket_connect("/api/chat/stream") as ws:
            ws.send_json({"message": "   "})
            evs = self._collect(ws)
        assert evs[0]["type"] == "error"
        assert evs[0]["error"]["type"] == "invalid"
        assert evs[-1]["type"] == "done"

    def test_unknown_persona_is_invalid_not_fallback(self, root, db, client):
        """未知视角必须报错：猜一个"最接近的"比默认视角更坏。"""
        with client.websocket_connect("/api/chat/stream") as ws:
            ws.send_json({"message": "hi", "persona": "玄学派"})
            evs = self._collect(ws)
        assert evs[0]["type"] == "error"
        assert evs[0]["error"]["type"] == "invalid"
        assert "玄学派" in evs[0]["error"]["message"]
        assert evs[-1]["type"] == "done"

    def test_bad_first_frame_gets_error(self, root, db, client):
        with client.websocket_connect("/api/chat/stream") as ws:
            ws.send_text("{不是 JSON")
            ev = ws.receive_json()
        assert ev["type"] == "error"
        assert ev["error"]["type"] == "invalid"

    def test_turn_events_session_first(self, root, db):
        """事件顺序契约：`session` 必须最先出现（前端要先拿到 session_id 才能存 URL）。"""
        from web.server import chat
        evs = list(chat.turn_events("", "", persona=None, model=None))
        # 空消息直接 error+done，不会有 session —— 这里确认它没有偷偷建会话
        assert [e["type"] for e in evs] == ["error", "done"]

    def test_history_read_before_write(self, root, db):
        """历史必须在写入本轮之前读，否则当前提问会出现两次（一次在历史末尾）。"""
        from core import chat_history
        from web.server import chat
        sid = chat_history.start_session(persona="综合", source="web")
        chat_history.save_turn(sid, "第一问", "第一答", usage={}, persona="综合",
                              model="t", source="web")
        hist = chat_history.load_messages(sid, limit=chat.HISTORY_LIMIT)
        assert [m["content"] for m in hist] == ["第一问", "第一答"]
        # 这一轮还没写：load 后再 save，历史里不会重复"第二问"
        chat_history.save_turn(sid, "第二问", "第二答", usage={}, persona="综合",
                              model="t", source="web")
        hist2 = chat_history.load_messages(sid, limit=chat.HISTORY_LIMIT)
        assert [m["content"] for m in hist2] == ["第一问", "第一答", "第二问", "第二答"]


class TestChatWithoutLLMConfig:
    """exe 里没有 `.env` 时 `core.chat_agent` 的 import 链会抛（`config.settings` 校验）。

    实测（2026-10-25 exe 冒烟）：`/api/chat/personas` 直接 500，且 WS 线程只发
    `error` 不发 `done` ⇒ 前端输入框永久灰住。**配置缺失要显示成"缺配置"**。
    """

    def _break(self, monkeypatch):
        # sys.modules 里放 None ⇒ 后续 `from core.chat_agent import X` 抛 ImportError
        monkeypatch.setitem(__import__("sys").modules, "core.chat_agent", None)

    def test_personas_returns_auth_error_not_500(self, root, client, monkeypatch):
        self._break(monkeypatch)
        z = client.get("/api/chat/personas").json()
        assert z["ok"] is False
        assert z["error"]["type"] == "auth"

    def test_turn_still_ends_with_done(self, root, monkeypatch):
        """没 `done` 前端就解不了锁——比报错更难受。"""
        from web.server import chat
        self._break(monkeypatch)
        evs = list(chat.turn_events("", "招商银行怎么看"))
        assert [e["type"] for e in evs] == ["error", "done"]
        assert evs[0]["error"]["type"] == "auth"


# ================================================================
# 系统页（P5）
# ================================================================
class TestSystem:
    def test_health_reports_pid_and_root(self, root, client):
        data = _ok(client.get("/api/system/health").json())
        assert isinstance(data["pid_self"], int)
        assert data["data_root"] == str(root)
        assert "running" in data          # 调度器未跑也必须是 ok:true + running:false
        assert data["offline"] in (True, False)

    def test_tasks_lists_all_jobs(self, root, client):
        data = _ok(client.get("/api/system/tasks").json())
        assert data["total"] == len(data["jobs"]) == EXPECTED_JOBS
        job = data["jobs"][0]
        for k in ("job_id", "time", "mode", "desc", "running"):
            assert k in job, (k, job)
        ids = [j["job_id"] for j in data["jobs"]]
        assert len(ids) == len(set(ids))

    def test_tasks_reports_progress_when_running(self, root, client):
        """进度文件存在时，系统页要能显示百分比（而不是只有"运行中"）。"""
        import core.ui_tasks as ui_tasks
        from core.scheduler import JOBS
        jid = JOBS[0][0]                       # 任一真实定时任务，只验证"进度会流到行上"
        ui_tasks.write_progress(jid, {"status": "running", "percent": 42.0,
                                      "message": "初筛 3/8"})
        data = _ok(client.get("/api/system/tasks").json())
        row = next(j for j in data["jobs"] if j["job_id"] == jid)
        assert row["progress"]["percent"] == 42.0
        assert row["progress"]["message"] == "初筛 3/8"
        ui_tasks.clear_progress(jid)

    def test_running_web_task_listed_separately(self, root, client):
        """`web_` 前缀的一次性任务不在 JOBS 里 ⇒ 必须单独列，否则页面上看不见。"""
        import core.ui_runs as ui_runs
        assert ui_runs.mark_running("web_screen", {}) is True
        try:
            data = _ok(client.get("/api/system/tasks").json())
            assert "web_screen" in [t["task"] for t in data["web_tasks"]]
        finally:
            ui_runs.clear_running("web_screen")

    def test_cron_labels(self):
        """`_cron_label` 不猜：缺 hour 就是"每小时"，minute 为 '*' 就是每分钟。"""
        from web.server.system import _cron_label
        assert _cron_label({"minute": 5}) == "每小时 :05"
        assert _cron_label({"hour": 15, "minute": 20}) == "15:20"
        assert _cron_label({"hour": "*", "minute": "*"}) == "* 每分钟"
        assert _cron_label({"minute": "*"}) == "每分钟"

    def test_models_never_leaks_keys(self, root, client):
        z = client.get("/api/system/models").json()
        data = _ok(z)
        blob = json.dumps(data, ensure_ascii=False)
        assert "api_key" not in blob.lower()
        assert "sk-" not in blob
        for p in data["providers"]:
            assert "key" not in " ".join(p.keys()).lower()
            # endpoint 只给主机名（完整 base_url 可能带内网路径）
            assert "/" not in p["endpoint"]

    def test_models_marks_ready_by_config_not_by_probe(self, root, client):
        data = _ok(client.get("/api/system/models").json())
        assert data["providers"], "至少应有一家厂商"
        assert all("ready" in p for p in data["providers"])
        assert data["default_provider"]

    def test_cost_returns_budget(self, root, db, client):
        data = _ok(client.get("/api/system/cost").json())
        assert set(data["budget"]) >= {"cap", "spent", "ratio"}
        assert data["window_days"] == 7


# ================================================================
# 任务状态 / 写操作（P5）
# ================================================================
class TestTaskState:
    def test_never_ran_is_missing(self, root, client):
        _fail(client.get("/api/tasks/web_never_ran", params={"result": 1}).json(), "missing")

    def test_ran_but_no_result_is_empty_not_missing(self, root, client):
        """跑过（有进度文件）但没产物 ⇒ `result=null` + ok:true，与"从没跑过"分开。"""
        import core.ui_tasks as ui_tasks
        ui_tasks.write_progress("web_screen", {"status": "done", "percent": 100})
        data = _ok(client.get("/api/tasks/web_screen", params={"result": 1}).json())
        assert data["result"] is None
        assert data["status"] == "done"
        ui_tasks.clear_progress("web_screen")

    def test_result_is_unwrapped_payload(self, root, client):
        """⚠️ 回归：`load_result` 的信封必须在这里解掉，否则前端读不到字段。"""
        import core.ui_runs as ui_runs
        import core.ui_tasks as ui_tasks
        ui_tasks.write_progress("web_screen", {"status": "done", "percent": 100})
        ui_runs.save_result("web_screen", {"date": "20261024",
                                           "picks": [{"股票代码": "600519"}],
                                           "market_signal": {"regime": "震荡"}})
        data = _ok(client.get("/api/tasks/web_screen", params={"result": 1}).json())
        assert data["result"]["picks"][0]["股票代码"] == "600519"
        assert data["result"]["market_signal"]["regime"] == "震荡"
        assert "payload" not in data["result"]        # 信封不许漏出来
        assert data["result_at"]
        ui_tasks.clear_progress("web_screen")
        ui_runs.clear_result("web_screen")

    def test_cancel_not_running_is_missing(self, root, client):
        _fail(client.post("/api/tasks/web_screen/cancel").json(), "missing")

    def test_cancel_running_writes_marker(self, root):
        """取消是**请求**：写标记，真正停下要等任务跑到下一个进度点。"""
        import core.ui_tasks as ui_tasks
        import core.ui_runs as ui_runs
        from web.server import jobs
        assert ui_runs.mark_running("web_screen", {}) is True
        data = _ok(jobs.cancel("web_screen"))
        assert data["cancel_requested"] is True
        assert ui_tasks.is_cancelled("web_screen") is True
        ui_runs.clear_running("web_screen")
        ui_tasks.clear_cancel("web_screen")


class TestWriteGuards:
    """只测**参数校验**分支：合法参数会真的起线程跑全市场选股（联网 + 十几分钟）。"""

    def test_unknown_strategy_is_invalid(self, root):
        from web.server import jobs
        err = _fail(jobs.run_screen(strategies=["不存在的策略"]), "invalid")
        assert "不存在的策略" in err["detail"]

    def test_max_stocks_out_of_range(self, root):
        from web.server import jobs
        _fail(jobs.run_screen(max_stocks=0), "invalid")
        _fail(jobs.run_screen(max_stocks=99), "invalid")

    def test_analysis_code_must_be_6_digits(self, root):
        from web.server import jobs
        for bad in ("", "abc", "60051", "6005199", "60051a"):
            _fail(jobs.run_analysis(bad), "invalid")

    def test_analysis_history_code_validated(self, root, db):
        from web.server import jobs
        _fail(jobs.analysis_history("abc"), "invalid")

    def test_zhihu_snapshot_missing_when_never_collected(self, root):
        from web.server import jobs
        _fail(jobs.zhihu_snapshot(), "missing")

    def test_zhihu_snapshot_unwraps_payload(self, root):
        """⚠️ 回归：不解包时 `items` 恒为空 ⇒ 采集成败都报"尚无采集结果"。"""
        import core.ui_runs as ui_runs
        from web.server import jobs
        ui_runs.save_result(jobs.TASK_ZHIHU, {"items": [], "meta": {"collected_at": "2026-10-24 15:20"}})
        # 空 items ⇒ 仍是 missing（**不编造快照**），但错误信息必须来自"没有条目"
        err = _fail(jobs.zhihu_snapshot(), "missing")
        assert "尚无采集结果" in err["detail"]
        ui_runs.clear_result(jobs.TASK_ZHIHU)


class TestDeskOrder:
    def test_bad_code_is_invalid(self, root, db):
        from web.server import desk
        _fail(desk.place_order(None, "6005", "买入", 100), "invalid")

    def test_unknown_direction_is_invalid(self, root, db):
        from web.server import desk
        err = _fail(desk.place_order(None, "600519", "做多", 100), "invalid")
        assert "做多" in err["detail"]

    def test_zero_quantity_is_invalid(self, root, db):
        from web.server import desk
        _fail(desk.place_order(None, "600519", "买入", 0), "invalid")

    def test_no_account_is_missing(self, root, db):
        """空库里没有模拟账户 ⇒ missing（不是"下单成功"）。"""
        from web.server import desk
        _fail(desk.place_order(None, "600519", "买入", 100), "missing")


# ================================================================
# SWR 缓存（P4 之前的 dashboard 修复）
# ================================================================
class TestSWR:
    def test_fresh_hit_does_not_call_loader(self):
        import web.server.cache as C
        C.clear()
        calls = []

        def loader():
            calls.append(1)
            return {"v": len(calls)}

        assert C.swr("t", "k", 60, loader) == {"v": 1}
        assert C.swr("t", "k", 60, loader) == {"v": 1}
        assert len(calls) == 1
        assert C.age_of("t", "k") < 5
        C.clear()

    def test_stale_served_immediately_and_refreshed(self):
        """过期但未超 max_stale ⇒ 立刻给旧值（首屏不等），后台刷新换新值。"""
        import time

        import web.server.cache as C
        C.clear()
        calls = []

        def loader():
            time.sleep(0.2)                # 模拟真实慢 IO（本地日线 8.6ms、实时报价 1.2s）
            calls.append(1)                # 只在**取数完成**时记一笔
            return {"v": len(calls)}

        # ttl 必须固定：后台刷新沿用它，混用不同 ttl 会让"刚刷完又过期"
        TTL = 0.3
        assert C.swr("t", "k", TTL, loader) == {"v": 1}
        time.sleep(0.35)                   # 等它过期
        t0 = time.monotonic()
        assert C.swr("t", "k", TTL, loader, max_stale=60) == {"v": 1}
        spent = time.monotonic() - t0
        assert spent < 0.15, f"应按旧值立即返回，实际阻塞 {spent:.2f}s"
        assert len(calls) == 1             # 刷新还在后台睡觉，没同步重取
        for _ in range(200):               # 等后台线程把新值放进去
            if len(calls) >= 2:
                break
            time.sleep(0.01)
        assert len(calls) == 2             # 只刷一次（_REFRESHING 防并发刷）
        assert C.swr("t", "k", TTL, loader, max_stale=60) == {"v": 2}
        assert len(calls) == 2             # 新值是新鲜的，不该再刷
        C.clear()

    def test_beyond_max_stale_blocks_and_reloads(self):
        import time

        import web.server.cache as C
        C.clear()
        calls = []

        def loader():
            calls.append(1)
            return {"v": len(calls)}

        C.swr("t", "k", 0.01, loader)
        time.sleep(0.03)
        assert C.swr("t", "k", 60, loader, max_stale=0.01) == {"v": 2}
        C.clear()

    def test_age_of_unknown_is_none(self):
        import web.server.cache as C
        C.clear()
        assert C.age_of("nope", "nope") is None


# ================================================================
# 模板 / 静态资源守卫（P4/P5 新增页）
# ================================================================
class TestNewTemplates:
    PAGES = ("chat.html", "system.html", "screening.html", "analysis.html", "intel.html")

    def test_pages_exist_and_nonempty(self):
        for name in self.PAGES:
            p = os.path.join("web", "templates", name)
            assert os.path.getsize(p) > 2000, p

    def test_shared_markdown_renderer(self):
        """md.js 是 review/chat/analysis 共用的唯一一份（先转义再套标签）。"""
        p = "web/static/js/md.js"
        assert os.path.getsize(p) > 500
        s = open(p, encoding="utf-8").read()
        assert "window.bigaMd" in s
        assert "window.bigaEsc" in s
        for name in ("review.html", "chat.html", "analysis.html"):
            page = open(os.path.join("web", "templates", name), encoding="utf-8").read()
            assert "../static/js/md.js" in page, name

    def test_review_no_longer_duplicates_renderer(self):
        """抽公共件之后，review.html 里不该再有一份 mdToHtml（两份必然漂移）。"""
        s = open("web/templates/review.html", encoding="utf-8").read()
        assert "mdToHtml" not in s

    def test_chat_has_no_explanatory_boilerplate(self):
        """零解释性文案：不许出现"该页面/本页说明/使用说明"这类功能解释。"""
        for name in self.PAGES:
            s = open(os.path.join("web", "templates", name), encoding="utf-8").read()
            for bad in ("该页面", "本页说明", "使用说明", "此功能用于", "功能说明"):
                assert bad not in s, (name, bad)


# ================================================================
# P6 打包脚本守卫（静态文本，不 import —— 见模块 docstring）
# ================================================================
class TestPackaging:
    def test_scripts_exist(self):
        for p in ("scripts/build_exe.py", "scripts/launch_exe.py",
                  "requirements-build.txt"):
            assert os.path.getsize(p) > 0, p

    def test_folder_mode_not_onefile(self):
        s = open("scripts/build_exe.py", encoding="utf-8").read()
        assert '"--onedir"' in s
        assert '"--onefile"' not in s          # onefile 每次启动都要解压 300MB+

    def test_panel_data_excluded(self):
        """`data/panel/by_code` ≈387MB，打包进去会让体积和启动都崩。"""
        s = open("scripts/build_exe.py", encoding="utf-8").read()
        assert "by_code" in s                  # 必须**明确**提到它被排除
        assert "387" in s or "MB" in s

    def test_third_party_data_bundled(self):
        """第三方包的数据/二进制必须进包，否则运行时**静默降级**。"""
        s = open("scripts/build_exe.py", encoding="utf-8").read()
        assert "--collect-data" in s
        assert "akshare" in s                  # file_fold/calendar.json
        assert "py_mini_racer" in s            # mini_racer.dll + icudtl.dat（交易日历 JS 引擎）
        assert "mini_racer.dll" in s or "PACKAGE_DATA_VERBATIM" in s

    def test_templates_and_config_bundled(self):
        s = open("scripts/build_exe.py", encoding="utf-8").read()
        assert "web/templates" in s.replace("\\", "/")
        assert "web/static" in s.replace("\\", "/")
        assert "config" in s

    def test_workpath_overridable(self):
        """中间目录必须可覆盖：`build/work/<name>.exe` 被外部进程锁住时，
        PyInstaller 会在 `os.remove` 抛 `PermissionError: [WinError 5]`，
        只有一行路径、极易误判成"权限不足"。换目录即可绕开，与最终产物无关。

        ⚠️ 只做静态断言，**不调用 `build_args()`**：它会执行 `_data_args()`，
        而那个函数会 rmtree 并重建 `build/config_pack` —— 测试不该有这种副作用。
        """
        s = open("scripts/build_exe.py", encoding="utf-8").read()
        assert '"--workpath"' in s
        assert "a.workpath" in s              # CLI 参数被真正接到 build_args
        assert 'os.path.join(BUILD, "work")' in s   # 默认值不变

    def test_intermediate_artifact_guard(self):
        """`_exe_locked` 只查 `dist/` 成品是不够的：中间产物被锁是**另一种**
        失败（此时 dist 还完好，只差换目录），必须有独立的提前判据。
        """
        s = open("scripts/build_exe.py", encoding="utf-8").read()
        assert "中间产物" in s
        assert "workpath" in s
        assert "_exe_locked(inter)" in s

    def test_launcher_writes_port_file(self):
        s = open("scripts/launch_exe.py", encoding="utf-8").read()
        assert "web_port.txt" in s             # 第二次双击靠它定位已有实例
        assert "acquire_os_mutex" in s         # 单实例只认 OS 互斥
        assert "_port_file" in s
        # 端口路径必须跟 data_root 走，不能在模块级写死到项目根
        assert "settings.data_root()" in s

    def test_launcher_has_main_guard(self):
        s = open("scripts/launch_exe.py", encoding="utf-8").read()
        assert 'if __name__ == "__main__":' in s

    def test_settings_shims(self):
        from web.server import settings
        assert settings.is_frozen() is False
        assert settings.bundle_root() == settings.PROJECT_ROOT
        assert settings.app_root() == settings.PROJECT_ROOT
        # 数据根必须是一个真实存在的可写目录（打包后 = exe 同级）
        assert os.path.isdir(settings.data_root())

    @staticmethod
    def _build_mod():
        """按路径加载 `scripts/build_exe.py`（只在 `__main__` 下跑构建，import 安全）。"""
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scripts", "build_exe.py")
        spec = importlib.util.spec_from_file_location("_build_exe", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_verify_artifact_rejects_gutted_dir(self, tmp_path):
        """**行为**判据：复现 2026-10-25 的"掏空 dist"，校验必须红。

        事故：旧 exe 还在跑 ⇒ COLLECT 的 `rmtree` 删不掉它、抛 WinError 5，
        但**其余文件已被删光** ⇒ 留下"只有 exe 没有 `_internal/`"的残缺目录。
        它骗过了 `os.path.exists(exe)`，双击只弹标题为 "Error" 的系统框、
        一行 Python 都不执行。所以这里的判据必须是**按内容**，不是按存在性。
        """
        m = self._build_mod()
        d = tmp_path / "bigAagents"
        d.mkdir()
        exe = d / "bigAagents.exe"
        exe.write_bytes(b"MZ" + b"\x00" * 62)          # 只有 exe，没有 _internal

        ok, detail = m._verify_artifact(str(exe))
        assert ok is False, "残缺 onedir 目录（缺 _internal）必须判失败"
        assert "_internal" in detail

        # 补上 _internal 但文件数仍不够（COLLECT 半途而废的另一种形态）
        (d / "_internal").mkdir()
        ok2, detail2 = m._verify_artifact(str(exe))
        assert ok2 is False, "文件数不足必须判失败"
        assert "文件" in detail2

    def test_verify_artifact_accepts_complete_dir(self, tmp_path):
        """反向断言：补全后必须判成功 —— 否则前一阶段就是恒假的假闸门。"""
        m = self._build_mod()
        d = tmp_path / "bigAagents"
        internal = d / "_internal"
        internal.mkdir(parents=True)
        exe = d / "bigAagents.exe"
        exe.write_bytes(b"MZ" + b"\x00" * 62)
        for i in range(600):
            (internal / f"f{i}.dat").write_bytes(b"x")
        ok, detail = m._verify_artifact(str(exe))
        assert ok is True, detail

    def test_exe_locked_on_missing_file(self, tmp_path):
        """占用检测对不存在的文件必须返回 False（不能因为探测本身抛异常阻断构建）。"""
        m = self._build_mod()
        assert m._exe_locked(str(tmp_path / "nope.exe")) is False

    def test_build_checks_lock_before_build(self):
        """构建前必须先判占用：`rmtree` 半途而废会把 dist 掏成残缺目录。"""
        s = open("scripts/build_exe.py", encoding="utf-8").read()
        assert "_exe_locked" in s
        assert "r+b" in s, "占用检测必须真的尝试以写方式打开（只 stat 查不出来）"
        assert "return 3" in s, "被占用时应以独立退出码中止，而不是继续构建"


class TestDesktopShortcut:
    """快捷方式必须带 `LinkTargetIDList`（2026-10-25 实测：缺了 Windows 直接拒收）。

    ⚠️⚠️ 这个坑**结构上自洽、行为上不可用**：HeaderSize 0x4C、CLSID 对、LinkInfo
       各偏移都对、LocalBasePath 能解出中文路径、TerminalBlock 也在 —— 静态看
       一切正常，但双击报"没有应用程序与此操作的指定文件有关联"（不是"找不到
       目标"，而是"这不是快捷方式"）。隔离实验：拿正品 .lnk 只清掉
       `HasLinkTargetIDList` 标志位、其余字节不动 ⇒ 报同一个错。
       所以这里必须**钉死 IDList 及其终止符**，光断言"文件写出来了"等于没测。
    """

    @staticmethod
    def _mod():
        """按路径加载 `assets/make_shortcut.py`（它不是包，不能 import 语句）。"""
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", "make_shortcut.py")
        spec = importlib.util.spec_from_file_location("_mk_shortcut", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_lnk_has_idlist(self, tmp_path):
        import struct
        m = self._mod()
        exe = tmp_path / "bigaagents.exe"
        exe.write_bytes(b"MZ" + b"\x00" * 62)
        blob = m.build_lnk(str(exe), desc="qa")

        assert struct.unpack("<I", blob[0:4])[0] == 0x4C
        flags, = struct.unpack("<I", blob[20:24])
        assert flags & m._HAS_IDLIST, "缺 HasLinkTargetIDList ⇒ Windows 会拒收"

        n, = struct.unpack("<H", blob[0x4C:0x4E])
        raw = blob[0x4E:0x4E + n]
        # 逐条走完，必须正好停在 2 字节终止符上（越界/错位都说明布局算错了）
        pos, types = 0, []
        while pos < len(raw) - 1:
            sz, = struct.unpack("<H", raw[pos:pos + 2])
            if sz == 0:
                break
            assert pos + sz <= len(raw), "ItemID 越界"
            types.append(raw[pos + 2])
            pos += sz
        assert raw[pos:] == b"\x00\x00", f"IDList 终止符缺失（结尾 {raw[pos:].hex()}）"
        assert types[0] == 0x1F, "第 0 项必须是 我的电脑"
        assert types[1] == 0x2F, "第 1 项必须是卷"
        assert types[-1] in (0x31, 0x32), "最后一项必须是目标文件"

    def test_lnk_idlist_covers_real_path(self, tmp_path):
        """IDList 必须**逐级**覆盖真实目录层级：跳级会让 Windows 去解析不存在的父项。"""
        m = self._mod()
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        exe = deep / "x.exe"
        exe.write_bytes(b"MZ")
        idlist = m.build_idlist(str(exe))
        anc = idlist.decode("mbcs", "ignore")
        for part in ("a", "b", "x.exe"):
            assert part in anc, part

    def test_selfcheck_rejects_missing_idlist(self, tmp_path):
        """守卫必须**有鉴别力**：把 IDList 摘掉后自检要变红。

        恒真的检查等于没装闸门 —— 这条就是给上面那条装的"反向断言"。
        """
        import struct
        m = self._mod()
        exe = tmp_path / "bigaagents.exe"
        exe.write_bytes(b"MZ" + b"\x00" * 62)
        blob = bytearray(m.build_lnk(str(exe), desc="qa"))
        assert m._selfcheck(bytes(blob)) == [], "正常产物不应报错"

        # 清掉 HasLinkTargetIDList 标志位 + 删掉 IDList 那一段（模拟第一版的做法）
        flags, = struct.unpack("<I", blob[20:24])
        n, = struct.unpack("<H", blob[0x4C:0x4E])
        stripped = bytes(blob[:20]) + struct.pack("<I", flags & ~m._HAS_IDLIST) + bytes(blob[24:])
        stripped = stripped[:0x4C] + stripped[0x4E + n:]
        errs = m._selfcheck(stripped)
        assert any("HasLinkTargetIDList" in e for e in errs), errs


class TestDocEndpointCount:
    """文档里的端点表必须与真实 router 一致（**路径数**与**操作数**分开算）。

    ⚠️ 这不是"文档洁癖"：端点表是**人查端点时的唯一入口**，数字不对会让人以为
       某个端点没实现（或漏实现了一个却不自知）。而 `app.openapi()` 在本版
       FastAPI 里数不出来（`app.routes` 装的是 `_IncludedRouter` 惰性对象），
       所以人肉数也容易错——只能靠这条断言钉住。

    ⚠️⚠️ **两个口径不能混**：一张表一行 = 一个 path；`GET+DELETE` 挂同一个 path 时
       操作数 > 行数。把 35（path）当成 35（端点）会误判"少了 3 个端点"。
    """

    @staticmethod
    def _router_counts() -> tuple[int, int, int]:
        """返回 (唯一 path 数, HTTP 操作数, WS 操作数)。"""
        import importlib
        import pkgutil

        import web.server.routers as R
        paths: dict[str, set] = {}
        for m in pkgutil.iter_modules(R.__path__):
            rt = getattr(importlib.import_module(f"web.server.routers.{m.name}"),
                         "router", None)
            if rt is None:
                continue
            for r in rt.routes:
                p = getattr(r, "path", "")
                ms = getattr(r, "methods", None)
                paths.setdefault(p, set()).update(ms if ms else {"WS"})
        http = sum(1 for v in paths.values() for m in v if m != "WS")
        ws = sum(1 for v in paths.values() for m in v if m == "WS")
        return len(paths), http, ws

    @staticmethod
    def _doc_path() -> str:
        """取最新的前端开发文档（文件名带日期，不能写死）。"""
        import glob

        from web.server import settings
        cands = sorted(glob.glob(os.path.join(settings.PROJECT_ROOT,
                                              "output", "前端开发文档_*.md")))
        assert cands, "找不到前端开发文档"
        return cands[-1]

    def test_doc_table_matches_routers(self):
        n_paths, http, ws = self._router_counts()
        doc = open(self._doc_path(), encoding="utf-8").read()

        # 表格行：| `GET /api/xxx` ✅ | ...   —— 一行一个 path，去重
        import re
        rows = set(re.findall(r"^\| `(?:GET|POST|DELETE|PUT|PATCH|WS) (/api/[^`]+)`",
                              doc, re.M))
        # URL 里的 `?` 是"怎么调"，不是路径的一部分
        rows = {r.split("?")[0] for r in rows}
        assert len(rows) == n_paths, (
            f"文档端点表 {len(rows)} 行 ≠ 实际 {n_paths} 个路径")

        # 摘要必须把"路径 / 操作"两个数字都说清，否则下一个读的人必然混淆
        ops = http + ws
        assert f"{n_paths} 个路径" in doc, f"摘要未写「{n_paths} 个路径」"
        assert f"{ops} 个操作" in doc, f"摘要未写「{ops} 个操作」"
        assert f"{http} 条 HTTP" in doc, f"摘要未写「{http} 条 HTTP」"

    def test_doc_path_params_match_router(self):
        """路径参数名写错（`{id}` vs `{session_id}`）会让人照着文档写错自测脚本。"""
        import re

        _n, _h, _w = self._router_counts()
        import importlib
        import pkgutil

        import web.server.routers as R
        real = set()
        for m in pkgutil.iter_modules(R.__path__):
            rt = getattr(importlib.import_module(f"web.server.routers.{m.name}"),
                         "router", None)
            if rt is None:
                continue
            real.update(getattr(r, "path", "") for r in rt.routes)

        doc = open(self._doc_path(), encoding="utf-8").read()
        rows = {r.split("?")[0] for r in
                re.findall(r"^\| `(?:GET|POST|DELETE|PUT|PATCH|WS) (/api/[^`]+)`",
                           doc, re.M)}
        assert rows == real, f"文档路径与实际不符：多 {rows - real}，少 {real - rows}"

    def test_doc_does_not_use_openapi_to_count(self):
        """文档必须写明 openapi() 数不出端点——否则下一个人会重犯。"""
        doc = open(self._doc_path(), encoding="utf-8").read()
        assert "_IncludedRouter" in doc
        assert "app.openapi()" in doc


class TestDataRootResolution:
    """数据根解析规则（`config/paths.py` 是唯一来源）。

    ⚠️⚠️ 这一组直接对应 2026-10-25 的真实事故：双击 `dist/bigAagents/bigAagents.exe`
       后**只有仪表盘能用**——因为 `data_root()` 退回 `app_root()`（空目录），
       选股/复盘/研究全 `missing`、交易台变**新开的空账户**。
       而看盘/新闻/指数是联网取的 ⇒ 唯独仪表盘"看起来正常"，极难一眼看出是路径问题。

    ⚠️ 第二个坑更隐蔽：`db.models.DB_PATH` 当时用 `Path(__file__).parent.parent` 算死，
       打包后指向 `_internal/data/portfolio.db`（空库）。**即使数据根修好了**，
       交易台照样是空账户 —— 两个坑必须分开守。
    """

    def test_source_mode_points_to_project_root(self, monkeypatch):
        monkeypatch.delenv("BIGA_DATA_ROOT", raising=False)
        from config import paths
        assert paths.data_root() == paths.project_root()
        assert paths.db_path() == os.path.join(paths.project_root(), "data", "portfolio.db")

    def test_env_wins_over_everything(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BIGA_DATA_ROOT", str(tmp_path))
        from config import paths
        assert paths.data_root() == str(tmp_path)
        assert paths.db_path() == os.path.join(str(tmp_path), "data", "portfolio.db")

    def test_dev_layout_walks_up_to_project_root(self, monkeypatch):
        """exe 在 `<根>/dist/<name>/` ⇒ 必须向上找到项目根的数据。"""
        monkeypatch.delenv("BIGA_DATA_ROOT", raising=False)
        monkeypatch.delenv("BIGA_DB_PATH", raising=False)
        from config import paths
        fake_app = os.path.join(paths.project_root(), "dist", "bigAagents")
        monkeypatch.setattr(paths, "app_root", lambda: fake_app)
        assert paths.data_root() == paths.project_root()
        assert paths.db_path() == os.path.join(paths.project_root(), "data", "portfolio.db")

    def test_distribution_layout_prefers_beside_exe(self, monkeypatch, tmp_path):
        """分发布局：exe 旁边自带 data/ ⇒ 就用它，不许再往上爬（否则会读到隔壁项目）。"""
        monkeypatch.delenv("BIGA_DATA_ROOT", raising=False)
        monkeypatch.delenv("BIGA_DB_PATH", raising=False)
        fake_app = tmp_path / "app"
        (fake_app / "data").mkdir(parents=True)
        from config import paths
        monkeypatch.setattr(paths, "app_root", lambda: str(fake_app))
        assert paths.data_root() == str(fake_app)

    def test_unrelated_ancestor_not_mistaken(self, monkeypatch, tmp_path):
        """祖先目录只有 `output/`（没有 `output/daily_picks`）不算项目根。

        判据取严是为了防"读到别人桌面上的另一个项目"——读不到比读错好。
        """
        monkeypatch.delenv("BIGA_DATA_ROOT", raising=False)
        (tmp_path / "output").mkdir()
        fake_app = tmp_path / "a" / "b"
        fake_app.mkdir(parents=True)
        from config import paths
        monkeypatch.setattr(paths, "app_root", lambda: str(fake_app))
        assert paths.data_root() == str(fake_app)          # 兜底：自己的目录

    def test_models_db_path_no_longer_uses_file(self):
        """结构性守护：`db/models.py` 不许再用 `__file__` 推导数据路径。"""
        s = open("db/models.py", encoding="utf-8").read()
        assert "DB_PATH = _db_path()" in s
        assert 'Path(__file__).resolve().parent.parent / "data"' not in s


class TestDataRootResolution:
    """数据根解析规则（`config/paths.py` 是唯一来源）。

    ⚠️⚠️ 这一组直接对应 2026-10-25 的真实事故：双击 `dist/bigAagents/bigAagents.exe`
       后**只有仪表盘能用**——因为 `data_root()` 退回 `app_root()`（空目录），
       选股/复盘/研究全 `missing`、交易台变**新开的空账户**。
       而看盘/新闻/指数是联网取的 ⇒ 唯独仪表盘"看起来正常"，极难一眼看出是路径问题。

    ⚠️ 第二个坑更隐蔽：`db.models.DB_PATH` 当时用 `Path(__file__).parent.parent` 算死，
       打包后指向 `_internal/data/portfolio.db`（空库）。**即使数据根修好了**，
       交易台照样是空账户 —— 两个坑必须分开守。
    """

    def test_source_mode_points_to_project_root(self, monkeypatch):
        monkeypatch.delenv("BIGA_DATA_ROOT", raising=False)
        from config import paths
        assert paths.data_root() == paths.project_root()
        assert paths.db_path() == os.path.join(paths.project_root(), "data", "portfolio.db")

    def test_env_wins_over_everything(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BIGA_DATA_ROOT", str(tmp_path))
        from config import paths
        assert paths.data_root() == str(tmp_path)
        assert paths.db_path() == os.path.join(str(tmp_path), "data", "portfolio.db")

    def test_dev_layout_walks_up_to_project_root(self, monkeypatch):
        """exe 在 `<根>/dist/<name>/` ⇒ 必须向上找到项目根的数据。"""
        monkeypatch.delenv("BIGA_DATA_ROOT", raising=False)
        monkeypatch.delenv("BIGA_DB_PATH", raising=False)
        from config import paths
        fake_app = os.path.join(paths.project_root(), "dist", "bigAagents")
        monkeypatch.setattr(paths, "app_root", lambda: fake_app)
        assert paths.data_root() == paths.project_root()
        assert paths.db_path() == os.path.join(paths.project_root(), "data", "portfolio.db")

    def test_distribution_layout_prefers_beside_exe(self, monkeypatch, tmp_path):
        """分发布局：exe 旁边自带 data/ ⇒ 就用它，不许再往上爬（否则会读到隔壁项目）。"""
        monkeypatch.delenv("BIGA_DATA_ROOT", raising=False)
        monkeypatch.delenv("BIGA_DB_PATH", raising=False)
        fake_app = tmp_path / "app"
        (fake_app / "data").mkdir(parents=True)
        from config import paths
        monkeypatch.setattr(paths, "app_root", lambda: str(fake_app))
        assert paths.data_root() == str(fake_app)

    def test_unrelated_ancestor_not_mistaken(self, monkeypatch, tmp_path):
        """祖先目录只有 `output/`（没有 `output/daily_picks`）不算项目根。

        判据取严是为了防"读到别人桌面上的另一个项目"——读不到比读错好。
        """
        monkeypatch.delenv("BIGA_DATA_ROOT", raising=False)
        (tmp_path / "output").mkdir()
        fake_app = tmp_path / "a" / "b"
        fake_app.mkdir(parents=True)
        from config import paths
        monkeypatch.setattr(paths, "app_root", lambda: str(fake_app))
        assert paths.data_root() == str(fake_app)          # 兜底：自己的目录

    def test_models_db_path_no_longer_uses_file(self):
        """结构性守护：`db/models.py` 不许再用 `__file__` 推导数据路径。"""
        s = open("db/models.py", encoding="utf-8").read()
        assert "DB_PATH = _db_path()" in s
        assert 'Path(__file__).resolve().parent.parent / "data"' not in s


class TestNoFileBasedDataRoot:
    """**结构性守护**：`core/`、`db/`、`data/` 里不许用 `__file__` 推导"数据根"。

    ⚠️⚠️ 这不是风格洁癖，是一类真实事故（2026-10-25）：12 个 core 模块都写
       `_PROJECT_ROOT = dirname(dirname(abspath(__file__)))`，源码运行毫无问题，
       打包成 exe 后 `__file__` 落在只读包 `_internal/` 里 ⇒ 全部指向空目录：

         · `core/scheduler_health` 读不到心跳 ⇒ 系统页显示"调度器未运行"
         · `core/ui_runs` 读不到结果 ⇒ 选股/知乎结果显示"从未运行"
         · `core/pipeline`/`core/review_agent` 写错地方 ⇒ 复盘/选股产物落进包内
         · `db/models.DB_PATH` 指向空库 ⇒ 交易台变"100 万现金、0 持仓"

       逐处修完很容易复发（新模块照抄旧写法），所以用扫描兜住：
       **凡是数据目录，一律走 `config.paths.data_root()`**。

    只读资源（模板/静态/`config/*.json`）例外，它们该跟 `bundle_root()` 走。
    """

    SCAN_DIRS = ("core", "db", "data", "web")
    # 匹配与放行规则在模块级 `_scan_file_based_data_roots()` 里（同一份实现，
    # 既被本用例用于扫描全仓，也被鉴别力自检用于注入样本 —— 两处不能各写一套）。

    def test_no_file_based_data_root(self):
        from config import paths

        offenders = _scan_file_based_data_roots(paths.project_root(), self.SCAN_DIRS)
        assert not offenders, (
            "这些地方用 __file__ 推导了**数据/可写**路径（打包后指向只读包 "
            "`_internal/` ⇒ 读到空数据或写不进去）：\n  "
            + "\n  ".join(f"{p}:{ln} {v} → {u}" for p, ln, v, u in offenders))

    def test_guard_has_discriminance(self, tmp_path):
        """**守卫的鉴别力自检**：违规样本必须被抓，合规样本不许误报。

        ⚠️ 没有这一条，"检查通过"可能只是因为正则写错了（恒真）——
           本项目已有教训（守卫恒真 = 没装闸门）。
        """
        d = tmp_path / "core"
        d.mkdir()

        # ① 违规：把 __file__ 推导的根拼到 temp/
        (d / "bad.py").write_text(
            "import os\n"
            "_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n"
            'HEART = os.path.join(_ROOT, "temp", "x.json")\n', encoding="utf-8")
        hits = _scan_file_based_data_roots(str(tmp_path), ("core",))
        assert hits and hits[0][2] == "_ROOT", "违规样本没被抓到 ⇒ 守卫无效"
        (d / "bad.py").unlink()

        # ② 合规：同样的推导，但只用于 config/（随包分发，打包后在 _internal/config）
        (d / "ok.py").write_text(
            "import os\n"
            "_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n"
            'CFG = os.path.join(_ROOT, "config", "a.json")\n', encoding="utf-8")
        assert _scan_file_based_data_roots(str(tmp_path), ("core",)) == [], \
            "合规样本被误报 ⇒ 守卫过严"

    def test_paths_module_is_the_only_source(self):
        """`config/paths.py` 必须能被 core/db 依赖（不能反向依赖 web 层）。"""
        src = open("config/paths.py", encoding="utf-8").read()
        assert "import web" not in src and "from web" not in src
        assert "import core" not in src and "from core" not in src


def _scan_file_based_data_roots(root: str, dirs=("core", "db", "data", "web")):
    """扫描"用 `__file__` 推导根、又把它拼到 temp/output/data 等可写目录"的地方。

    返回 `[(相对路径, 行号, 变量名, 首个违规用法)]`；合规返回 `[]`。

    抽成模块级函数是为了让**鉴别力自检**能注入临时样本（`tmp_path`）——
    直接对着真实仓库断言"没问题"，无法证明守卫本身有效。
    """
    import glob

    pats = (
        # ⚠️ 用 `[ \t]*` 而不是 `\s*`：`\s` 含换行，`^\s*` 会从**上一行**末尾开始匹配，
        #    按字符位置算出的行号会整体偏移一行（实测踩到，把定义行认成别处）。
        re.compile(r"^[ \t]*(\w+)[ \t]*=[ \t]*os\.path\.dirname\(os\.path\.dirname\("
                   r"[^)\n]*__file__", re.M),
        re.compile(r"^[ \t]*(\w+)[ \t]*=[ \t]*Path\(__file__\)\.resolve\(\)\.parent\.parent",
                   re.M),
    )
    allowed = ('"config"', "'config'")     # 唯一放行：配置随包分发，在 _internal/config
    out = []
    for d in dirs:
        for f in glob.glob(os.path.join(root, d, "**", "*.py"), recursive=True):
            if "__pycache__" in f:
                continue
            src = open(f, encoding="utf-8").read()
            lines = src.splitlines()
            for pat in pats:
                for m in pat.finditer(src):
                    var = m.group(1)
                    idx = src[:m.start()].count("\n")
                    if any(tok in lines[idx] for tok in allowed):
                        continue               # 变量本身就是 config/ 下的文件
                    uses = [
                        ln.strip() for i, ln in enumerate(lines)
                        if i != idx
                        and not ln.strip().startswith("#")   # 注释里提到变量名不是用法
                        and re.search(rf"\b{re.escape(var)}\b", ln)
                    ]
                    bad = [u for u in uses if not any(tok in u for tok in allowed)]
                    if bad:
                        out.append((os.path.relpath(f, root), idx + 1, var, bad[0][:80]))
    return out


# ================================================================
# 深度分析回看 / 往期胜率（2026-10-25）
# ================================================================
class TestAnalysisCallback:
    """「报告点不开、胜率看不到」两个真实反馈对应的契约。

    用户原话：**"深度分析报告无法回看，往期胜率也没办法回看"**。
    事实是 `store.get_analysis` / `analysis_review.aggregate_recent` **都已存在**，
    只是从没接成 HTTP 端点、前端也没做点击 —— 所以这两条守的是"接线"，
    不是"算法"（算法早有测试）。
    """

    @staticmethod
    def _seed(monkeypatch, tmp_path):
        """造一条可回看的分析与若干复盘。

        ⚠️ 必须用 `use_template_db` 拿独立空库：写真实库会连带毁掉 `init_store`
           的惰性短路（见 conftest 的说明）。
        """
        from tests.conftest import use_template_db
        use_template_db(tmp_path, monkeypatch)
        from db import store

        # ⚠️ 用 `insert_analysis(record)` 而不是 `save_analysis(final_state)`：
        #    后者会用 `snapshot_from_state` **重算** payload，无法精确构造
        #    "payload 里有什么"这一被测前提（差一个字就会测成"字段确实不存在"）。
        ana_id = store.insert_analysis({
            "analysis_date": "2026-10-20",
            "stock_code": "600519",
            "stock_name": "贵州茅台",
            "price_at_analysis": 1500.0,
            "decision_label": "看多",
            "model": "test-model",
            "source": "test",
            "timing_regime": "range",
            "market_score": 42.5,
            "trade_plan": {"direction": "看多", "position_size": "10%"},
            "payload": {
                "stock_code": "600519", "stock_name": "贵州茅台",
                "analysis_date": "2026-10-20", "data_through": "2026-10-20",
                "tech_report": "技术面正文",
                "fundamental_report": "基本面正文",
                "news_report": "舆情正文",
                "research_summary": "研究汇总正文",
                "final_decision": "最终决策正文",
                "trade_plan": {"direction": "看多", "position_size": "10%"},
                "bull_history": ["多方论点"], "bear_history": ["空方论点"],
                "data_availability": {"kline": True, "fund": False},
                "audit_warnings": [], "errors": [],
            },
        })
        assert ana_id, "种子分析没落库"

        # 5 条复盘：3 判对 / 1 判错 / 1 观望 ⇒ 分母只算 4
        for i, verdict in enumerate(("correct", "correct", "partially_correct",
                                     "wrong", "no_position")):
            store.upsert_analysis_review({
                "analysis_id": ana_id + 100 + i,
                "stock_code": "600519", "analysis_date": "2026-10-20",
                "decision_label": "看多", "verdict": verdict,
                "primary_cause": "nothing_wrong" if verdict != "wrong" else "stop_too_tight",
                "horizon": 5, "status": "mechanical",
            })
        return ana_id

    def test_detail_returns_renderable_snapshot(self, client, tmp_path, monkeypatch):
        """回看页用**同一个渲染器**（`applySnapshot`），字段缺一个就整块显示"—"。"""
        ana_id = self._seed(monkeypatch, tmp_path)
        d = _ok(client.get(f"/api/analysis/detail/{ana_id}").json())
        snap = d["snapshot"]
        for key in ("stock_code", "stock_name", "analysis_date", "data_through",
                    "trade_plan", "tech_report", "fundamental_report", "news_report",
                    "research_summary", "final_decision",
                    "bull_history", "bear_history",
                    "data_availability", "audit_warnings", "errors"):
            assert key in snap, f"快照缺 {key}：回看页该块会是空白"
        assert snap["tech_report"] == "技术面正文"
        assert d["outcome"]["decision"] == "看多"
        assert d["meta"]["model"] == "test-model"

    def test_detail_missing_id_is_missing_not_empty(self, client, tmp_path, monkeypatch):
        """不存在的 id 必须是 `missing`（"没有这条"），不是 `empty`（"读失败"）。"""
        self._seed(monkeypatch, tmp_path)
        _fail(client.get("/api/analysis/detail/999999").json(), "missing")

    def test_detail_fills_from_table_when_payload_lacks_it(self, client, tmp_path, monkeypatch):
        """老记录 payload 可能没有 stock_code —— 必须用**表列真值**兜底。"""
        from tests.conftest import use_template_db
        use_template_db(tmp_path, monkeypatch)
        from db import store

        # ⚠️ 用 `insert_analysis(record)` 而不是 `save_analysis(final_state)`：
        #    后者会用 `snapshot_from_state` **重算** payload，无法精确构造
        #    "payload 里有什么"这一被测前提（差一个字就会测成"字段确实不存在"）。
        ana_id = store.insert_analysis({
            "analysis_date": "2026-10-20", "stock_code": "000001",
            "stock_name": "平安银行", "decision_label": "观望",
            "payload": {"tech_report": "只有正文，没有代码"},
        })
        snap = _ok(client.get(f"/api/analysis/detail/{ana_id}").json())["snapshot"]
        assert snap["stock_code"] == "000001"
        assert snap["stock_name"] == "平安银行"
        assert snap["tech_report"] == "只有正文，没有代码"

    def test_stats_denominator_excludes_no_position(self, client, tmp_path, monkeypatch):
        """⚠️⚠️ 分母 = **判出方向的条数**，观望单列。

        这是本仓库反复踩的口径坑：把 `no_position` 放进分母，会让"全观望"
        显示成 **0% 胜率**，而它其实是"没预测"，不是"猜错了"。
        """
        self._seed(monkeypatch, tmp_path)
        d = _ok(client.get("/api/analysis/stats?horizon=5").json())
        assert d["no_position"] == 1, "观望必须单列"
        assert d["hit"] == 3 and d["miss"] == 1 and d["unclear"] == 0
        assert d["directional"] == 4, "分母不能含观望"
        assert d["directional"] == d["hit"] + d["miss"] + d["unclear"]
        assert d["win_rate_pct"] == 75.0            # 3/4，不是 3/5=60

    def test_stats_null_rate_is_not_zero(self, client, tmp_path, monkeypatch):
        """没有样本时必须返回 `null`，**不是 0**。0% 会被读成"全错"。"""
        from tests.conftest import use_template_db
        use_template_db(tmp_path, monkeypatch)
        d = _ok(client.get("/api/analysis/stats?horizon=20").json())
        assert d["total_reviewed"] == 0
        assert d["win_rate_pct"] is None, "无样本必须是 null（'缺就是缺'）"

    def test_stats_rejects_unknown_horizon(self, client, tmp_path, monkeypatch):
        """T+5 与 T+20 是两件事；传 7 不能静默退默认。"""
        from tests.conftest import use_template_db
        use_template_db(tmp_path, monkeypatch)
        _fail(client.get("/api/analysis/stats?horizon=7").json(), "invalid")

    def test_literal_routes_not_swallowed_by_code_path(self, client, tmp_path, monkeypatch):
        """⚠️ 路由顺序回归守卫：`/api/analysis/stats` 与 `/detail/{id}` 必须先于
        `/api/analysis/{code}` 注册。

        写反了的表现是"请求 stats 却报『必须是 6 位数字』"——一个非常容易被
        误判成"前端传错参"的故障，所以用**行为**钉住，而不是读源码断言顺序。
        """
        from tests.conftest import use_template_db
        use_template_db(tmp_path, monkeypatch)
        z = client.get("/api/analysis/stats").json()
        assert z.get("ok") is True, f"stats 被 {{code}} 吃掉了：{z}"
        # ⚠️ 只看 `ok` 不够：`/api/analysis/{code}` 命中后走的是任务状态，
        #    任务不存在时也返回 `ok:true` —— 恒真断言等于没装闸门。
        #    必须用 **stats 独有字段** 判定（实测：顺序写反时这里是空的）。
        assert "win_rate_pct" in (z.get("data") or {}), \
            f"返回的不是胜率统计，是被 {{code}} 路由吃掉了：{z}"

        # detail 路径同理：id 不是 6 位数字也不能被判成"代码非法"
        z2 = client.get("/api/analysis/detail/1").json()
        assert (z2.get("error") or {}).get("type") == "missing", z2


# ================================================================
# 双击面板顺带拉起调度器（2026-10-26）
# ================================================================
class TestSchedulerAutostart:
    """用户反馈：**"打开 bigaagents 面板不能同时自动打开调度器吗，以及心跳"**。

    事实链（实测，不是推断）：
    - `scripts/launch_exe.py` 原样**完全不碰调度器**，只起 web；
    - 调度器要另外双击 `启动bigAagents.bat`，而那个 bat 起的是**旧版 Streamlit(8501)**，
      与新面板是两条互不相干的路径 ⇒ 真实结果就是"面板能开、调度器没起"；
    - 调度器一停，`temp/scheduler_heartbeat.json` 立刻停在最后一个 epoch，
      health 页 `running=False`（判活看 PID 不看 mtime，这一点本来就是对的）。

    所以这里守的是**接线**：面板起来了，调度器要被一起拉起来。
    """

    def test_scheduler_arg_detected(self):
        """`--scheduler` 是面板自调用的暗号，必须能被识别。"""
        from scripts import launch_exe as L
        assert L._is_scheduler_mode(["--scheduler"])
        assert L._is_scheduler_mode(["--scheduler", "--list-jobs"])
        assert not L._is_scheduler_mode([])
        # 只有别的参数时不能误判成调度器模式（否则面板双击会变成一个空转进程）
        assert not L._is_scheduler_mode(["--list-jobs"])
        assert not L._is_scheduler_mode(["--noscheduler"])

    def test_frozen_selfcall_not_py_script(self, monkeypatch):
        """⚠️⚠️ 冻结包里 `sys.executable` 是 **exe 自己**，不是 python.exe。

        写成 `[exe, "scripts/run_scheduler.py"]` 等于"用 exe 打开一个 .py"，
        noconsole 下**静默失败**（无窗口、无日志、无任何提示）。
        所以 frozen 必须走自调用。
        """
        from scripts import launch_exe as L
        monkeypatch.setattr(L.sys, "frozen", True, raising=False)
        cmd, cwd = L._scheduler_cmd()
        assert cmd[1] == L.SCHED_ARG, cmd
        assert not any(str(a).endswith(".py") for a in cmd), \
            f"冻结包不能靠 .py 脚本路径启动：{cmd}"
        assert cwd == L._data_root(), "工作目录必须是数据根（run_scheduler 用相对 logs/）"

    def test_source_mode_runs_script_with_python(self, monkeypatch):
        """源码模式相反：用 python 跑脚本，不能给 python 传 `--scheduler`。"""
        from scripts import launch_exe as L
        monkeypatch.delattr(L.sys, "frozen", raising=False)
        cmd, cwd = L._scheduler_cmd()
        assert cmd[0] == L.sys.executable, cmd
        assert cmd[1].endswith(os.path.join("scripts", "run_scheduler.py")), cmd
        assert os.path.isdir(cwd), "数据根必须存在"

    def test_skips_when_already_running(self, monkeypatch):
        """已在跑就**不能再起一个**：会重复选股/回测、叠加限频、重复落库。"""
        from scripts import launch_exe as L
        monkeypatch.delenv("BIGA_NO_SCHEDULER", raising=False)
        monkeypatch.setattr(L, "_scheduler_alive",
                            lambda: (True, {"pid": 4321, "ts": "2026-10-26 08:04:54"}))
        calls = []
        monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: calls.append(a))
        assert L._spawn_scheduler(probe_delay=0) is False
        assert not calls, "已有实例时不许 Popen"

    def test_env_switch_disables(self, monkeypatch):
        """`BIGA_NO_SCHEDULER=1` 必须能关掉（自动化验证时不能真跑选股）。"""
        from scripts import launch_exe as L
        monkeypatch.setenv("BIGA_NO_SCHEDULER", "1")
        monkeypatch.setattr(L, "_scheduler_alive", lambda: (False, {}))
        calls = []
        monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: calls.append(a))
        assert L._spawn_scheduler(probe_delay=0) is False
        assert not calls

    def test_spawns_detached_when_idle(self, monkeypatch, tmp_path):
        """空闲时应真的起进程，且**脱离**面板（关窗口不停调度）。"""
        from scripts import launch_exe as L
        monkeypatch.delenv("BIGA_NO_SCHEDULER", raising=False)
        monkeypatch.setattr(L, "_scheduler_alive", lambda: (False, {}))
        monkeypatch.setattr(L, "_log_dir", lambda: str(tmp_path))
        monkeypatch.setattr(L, "_scheduler_cmd",
                            lambda: (["bigaagents.exe", L.SCHED_ARG], str(tmp_path)))

        seen = {}

        class _FakeProc:
            pid = 9999
            def poll(self):
                return None

        def _fake_popen(cmd, **kw):
            seen["cmd"] = cmd
            seen.update(kw)
            return _FakeProc()

        monkeypatch.setattr(L.subprocess, "Popen", _fake_popen)
        assert L._spawn_scheduler(probe_delay=0) is True
        assert seen["cmd"] == ["bigaagents.exe", L.SCHED_ARG]
        assert seen["cwd"] == str(tmp_path)
        assert seen["stdin"] is L.subprocess.DEVNULL
        if os.name == "nt":
            flags = seen.get("creationflags", 0)
            assert flags & 0x00000008, "必须 DETACHED_PROCESS：关面板不能停调度"
            assert flags & 0x00000200, "必须 CREATE_NEW_PROCESS_GROUP"

    def test_spawn_failure_does_not_raise(self, monkeypatch, tmp_path):
        """调度器起不来时**面板必须照样能用**（看盘/复盘/交易台都不依赖它）。"""
        from scripts import launch_exe as L
        monkeypatch.delenv("BIGA_NO_SCHEDULER", raising=False)
        monkeypatch.setattr(L, "_scheduler_alive", lambda: (False, {}))
        monkeypatch.setattr(L, "_log_dir", lambda: str(tmp_path))

        def _boom(*a, **k):
            raise OSError("模拟缺依赖")

        monkeypatch.setattr(L.subprocess, "Popen", _boom)
        assert L._spawn_scheduler(probe_delay=0) is False   # 只返回 False，绝不抛

    def test_ensure_streams_fills_none(self, monkeypatch, tmp_path):
        """⚠️ `--noconsole` 下 `sys.stdout is None` ⇒ `StreamHandler(None)` 一 emit 就崩。

        这是 web 分支踩过的同一个坑，但调度器走的是**另一套 logging**
        （`run_scheduler.setup_logging()`），绕不过去，只能先把流补好。
        """
        from scripts import launch_exe as L
        monkeypatch.setattr(L.sys, "stdout", None)
        monkeypatch.setattr(L.sys, "stderr", None)
        monkeypatch.setattr(L, "_log_dir", lambda: str(tmp_path))
        L._ensure_streams()
        assert L.sys.stdout is not None, "stdout 必须被补上，否则调度器起不来"
        assert L.sys.stderr is not None
        L.sys.stdout.write("probe")
        L.sys.stdout.flush()

    def test_stop_script_kills_panel_spawned_scheduler(self):
        """⚠️ 必须能被「停止」脚本杀掉 —— 否则比不装自动拉起更糟。

        面板起的调度器是 `bigaagents.exe --scheduler`：**进程名不是 python**、
        命令行里也没有 `run_scheduler.py` ⇒ 原脚本那两条按 python 匹配的规则
        一条都抓不到，用户点"停止"后选股/复盘照旧在跑。
        """
        raw = open("停止bigAagents.bat", "rb").read()
        text = raw.decode("utf-8")
        # 只认**可执行行**：注释里提到 `--scheduler` 不算装了这个闸门
        hits = [ln for ln in text.splitlines()
                if "--scheduler" in ln and not ln.lstrip().lower().startswith("rem")]
        assert hits, "停止脚本缺少对面板调度器（bigaagents.exe --scheduler）的清理"
        assert any("Stop-Process" in ln for ln in hits), f"必须真的杀进程：{hits}"
        assert any("bigaagents*" in ln for ln in hits), \
            "必须限定进程名，否则会误杀别的带 --scheduler 参数的程序"
        assert raw.count(b"\n") - raw.count(b"\r\n") == 0, "bat 必须全 CRLF（LF 会让跳转出错）"


class TestProgressLiveness:
    """「深度分析好像卡住了」——2026-10-26 的真实反馈。

    实测事实：任务**没坏**。600519 在 194 秒内跑完、结果落盘 53KB、入库 id=73。
    但页面在那之前是一根不动的进度条：

        [12:31:51] live=0.0  msg=准备中…      ← 启动后 37 秒仍显示"已用 0s"
        [12:32:23] live=0.0  msg=准备中…
        [12:32:30] pct=6.7   msg=步骤 1: timing

    根因是 `progress.elapsed` 只在**节点回调**时被写一次，而择时节点要取市场快照
    （东财 5 页连续失败重试 + akshare 兜底），整段 40~80 秒没有任何回调。
    于是"已用 0s"冻在那里，用户只能判断成卡死。

    这两条守的是「界面必须能自证还活着」：
      1. `elapsed_live` 由 started_ts **现算**，与节点回调解耦；
      2. 同时给出 `idle_seconds`（距上次进度更新多久），让前端能明说
         "最后更新 Ns 前 / 该步骤较慢"，而不是让用户猜。
    """

    @staticmethod
    def _state(monkeypatch, running, started_ago=None, updated_ago=None):
        from datetime import datetime, timedelta
        import time

        from core import ui_runs, ui_tasks
        from web.server import jobs

        prog = {"status": "starting", "percent": 0.0, "elapsed": 0.0,
                "message": "准备中…"}
        if updated_ago is not None:
            prog["updated_at"] = (datetime.now() - timedelta(seconds=updated_ago)
                                  ).strftime("%Y-%m-%d %H:%M:%S")
        rec = None
        if started_ago is not None:
            rec = {
                "started_ts": time.time() - started_ago,
                "started_at": (datetime.now() - timedelta(seconds=started_ago)
                               ).strftime("%Y-%m-%d %H:%M:%S"),
            }
        # jobs.state 内部是 `import core.ui_runs as ui_runs`，拿到的是同一个模块
        # 对象 ⇒ 打在模块属性上的 patch 生效。
        monkeypatch.setattr(ui_tasks, "read_progress", lambda t: dict(prog))
        monkeypatch.setattr(ui_tasks, "is_running", lambda t: running)
        monkeypatch.setattr(ui_runs, "get_running_any", lambda t: rec)
        return jobs.state("web_analysis_600519")["data"]

    def test_live_elapsed_keeps_ticking_while_node_elapsed_is_frozen(self, monkeypatch):
        """★ 核心：节点回调冻住时，elapsed_live 必须还在走。"""
        d = self._state(monkeypatch, running=True, started_ago=44, updated_ago=44)
        assert d["running"] is True
        assert d["elapsed_live"] is not None and d["elapsed_live"] >= 43, \
            "运行中必须给出实时已用秒数，否则页面会显示「已用 0s」"
        assert d["idle_seconds"] is not None and d["idle_seconds"] >= 43, \
            "必须能报出「多久没更新」，前端才有依据说「该步骤较慢」"
        # 老字段仍然冻着——正因如此**不能**拿它当已用时长（这就是这次的 bug）
        assert d["progress"]["elapsed"] == 0.0
        assert d["elapsed_live"] > d["progress"]["elapsed"]

    def test_elapsed_live_absent_when_not_running(self, monkeypatch):
        """跑完了就别再报"已用时长"——那时应该看 progress.elapsed 的真实终值。"""
        d = self._state(monkeypatch, running=False, started_ago=None, updated_ago=600)
        assert d["running"] is False
        assert d["elapsed_live"] is None
        assert d["idle_seconds"] is not None and d["idle_seconds"] >= 599

    def test_poll_stops_before_render(self):
        """⚠️ 结构守卫：`stopPolling()` 必须在 `renderTask()` **之前**。

        原写法把 stopPolling 放在 renderTask 之后，而整条链被
        `.catch(() => null)` 吞掉 ⇒ 渲染只要抛一次异常，2 秒一次的轮询就永远
        停不下来，页面定格在最后一个快照、且没有任何提示。这与"任务真的卡住"
        在界面上完全无法区分 —— 用户报的就是这句「好像卡住了」。
        """
        html = open("web/templates/analysis.html", encoding="utf-8").read()
        start = html.index("function poll()")
        body = html[start:html.index("function startPolling()", start)]
        assert "stopPolling()" in body and "renderTask(z)" in body
        assert body.index("stopPolling()") < body.index("renderTask(z)"), \
            "必须先判结束再渲染，否则渲染异常会让轮询永久空转"
        assert "try {" in body and "catch" in body, \
            "renderTask 必须包在 try/catch 里：出错要显示原因并停轮询，不能静默吞掉"

    def test_page_renders_liveness_hints(self):
        """前端必须真的把这两个字段用起来，并有个位置显示它们。"""
        html = open("web/templates/analysis.html", encoding="utf-8").read()
        assert 'id="prog-live"' in html, "页面缺少「最后更新」的显示位置"
        assert "elapsed_live" in html, "renderTask 未使用后端现算的 elapsed_live"
        assert "idle_seconds" in html, "renderTask 未使用 idle_seconds"
        assert "该步骤较慢" in html, "超过阈值时应明确告诉用户仍在运行"


# ================================================================
# 断线 / 假数据守卫（2026-10-26 体检）
# ================================================================
#
# 这一类 bug 的共同形态：**数据在、接口在、页面不调**，或者**页面上的数字是写死的**。
# 它们不会报错、不会让测试变红，只会让用户看到"对的界面 + 错的事实"。
# 用户这半年报的三条（选股看不出质量 / 复盘不讲决策 / 涨跌家数是假数）
# 全是同一个形态，所以这里把判据固化成断言。

class TestWiringGuards:
    def test_review_page_calls_digest(self):
        """复盘页必须真的调 `/api/review/digest`（今日决策三块）。

        ⚠️ 断言的是 JS 调用字面量 `get('/api/review/digest')`，不是裸路径 ——
           模板注释里也写着这个路径，只断言 "review/digest" 会被自己的注释命中，
           那种断言恒真、等于没写。
        """
        s = open("web/templates/review.html", encoding="utf-8").read()
        assert "get('/api/review/digest')" in s
        for anchor in ("dg-screens", "dg-analyses", "dg-trades",
                       "dg-quality", "dg-analysis", "dg-actions"):
            assert anchor in s, anchor

    def test_review_digest_never_fakes_zero(self):
        """无样本时不许显示 0%/0 条 —— "没有样本"和"胜率 0%"是两件事。"""
        s = open("web/templates/review.html", encoding="utf-8").read()
        assert "样本不足" in s
        assert "观望不计入" in s, "观望档没有方向，不能进胜率分母"

    def test_index_has_no_demo_blocks(self):
        """首页必须全真：不许再出现 `data-demo` 标注块。

        历史：涨跌家数（写死 3,124/1,689）与四个数据源 chip（腾讯直连/同花顺快照/
        新浪+同花顺/DeepSeek）都挂过 `data-demo` —— 数值像真的，真实降级
        （如"行业缺 2,943 只"）在首页一个字都看不到。
        """
        s = open("web/templates/index.html", encoding="utf-8").read()
        assert "data-demo=" not in s
        assert "src-list" in s, "数据源状态区不见了"
        assert "601088" not in s.split("<script>")[0], "选股表里还有写死的占位股票"

    def test_watch_first_round_ignores_hidden(self):
        """看盘首轮不能被 `document.hidden` 挡住。

        后台标签页打开（中键）时 hidden 恒为 true；若首轮也走"暂停"，
        页面会永远是空的，直到用户切回来 —— 界面上看不出是"故意的"。
        """
        s = open("web/templates/watch.html", encoding="utf-8").read()
        assert "booted" in s
        assert "document.hidden && booted" in s

    def test_coverage_item_never_fake_ok(self, monkeypatch):
        """`data_coverage` 没记录时必须是 unknown，不能默认成 ok。"""
        import db.store as store
        from web.server import dashboard

        monkeypatch.setattr(store, "data_coverage_summary",
                            lambda *_a, **_k: {"latest": None, "days": 0})
        it = dashboard._coverage_item("quotes", "行情", "market_snapshot")
        assert it["state"] == "unknown", it
        assert it["text"] == "无采集记录", it

    def test_coverage_item_maps_degraded_to_warn(self, monkeypatch):
        """`status=degraded` 必须落到 warn，并把 note（缺行业 2,943）带进 title。"""
        import db.store as store
        from web.server import dashboard

        monkeypatch.setattr(store, "data_coverage_summary", lambda *_a, **_k: {"latest": {
            "data_date": "2026-10-26", "dataset": "market_snapshot", "source": "sina_full",
            "got": 5572, "total": 5576, "ratio": 0.9993, "status": "degraded",
            "note": "行业缺 2943 只", "captured_at": "2026-10-26 20:52:51"}})
        it = dashboard._coverage_item("quotes", "行情", "market_snapshot")
        assert it["state"] == "warn", it
        assert it["text"] == "sina_full 5,572/5,576", it
        assert "行业缺 2943" in it["title"], "降级原因必须透出，否则页面只有个黄灯没有原因"

    def test_sources_zone_has_four_real_items(self):
        """sources 分区四个条目齐全，且每条都有非空 text（空文字=沉默失败）。"""
        from web.server import dashboard
        zone = dashboard._sources_zone()
        assert zone["ok"] is True
        items = zone["data"]["items"]
        assert [i["key"] for i in items] == ["quotes", "flow", "news", "llm"]
        for i in items:
            assert i["label"] and i["text"], i
            assert i["state"] in ("ok", "warn", "bad", "unknown"), i

    # ---------------- 2026-10-26 第二批：AI 唤醒 / 审计留痕 ----------------
    def test_desk_page_calls_wake(self):
        """交易台必须真的调 `/api/desk/wake`（AI 盘中唤醒）。

        历史：`ai_wake_conditions`(60) / `ai_wake_events`(26) 一直在写、web 层零引用。
        用户看得到"每天 15:58 决策一次"，看不到"AI 在盘中被价格唤醒过 26 次、
        其中 3 次因为随机对照组压根没跑"。不摊开这一层，就无法区分"没动"与"动了没记"。
        """
        s = open("web/templates/desk.html", encoding="utf-8").read()
        assert "get('/api/desk/wake')" in s
        for anchor in ("wk-note", "wk-sum", "wk-body", "wk-cond-sum",
                       "wk-cond", "wk-foot"):
            assert anchor in s, anchor

    def test_wake_page_never_calls_armed_a_todo(self):
        """`armed=1` 不是待办：价位没到就没有事件行。

        页面文案必须是"盯盘中 / 仍生效，不代表即将触发"，不许写成"待触发 N 条"
        —— 后者会让人以为有 N 个信号在排队等着执行。
        """
        s = open("web/templates/desk.html", encoding="utf-8").read()
        assert "仍生效，不代表即将触发" in s
        assert "待触发" not in s, "armed 不是待办清单，不能写成待触发"

    def test_wake_skip_reason_is_surfaced(self):
        """命中但未执行必须显示原因（如"随机对照组不跑盘中决策"）。"""
        s = open("web/templates/desk.html", encoding="utf-8").read()
        assert "skip_reason" in s
        assert "命中但未执行" in s

    def test_system_page_calls_audit(self):
        """系统页必须真的调 `/api/system/audit`（配置变更 / 推送 / 情绪标注留痕）。"""
        s = open("web/templates/system.html", encoding="utf-8").read()
        assert "get('/api/system/audit" in s
        for anchor in ("chg-note", "chg-body", "push-note", "push-body",
                       "sent-note", "sent-body"):
            assert anchor in s, anchor

    def test_sentiment_unscored_not_neutral(self):
        """`stance='unscored'` 是"判不出方向"，页面必须单独成列并说明它不算中性。

        实测 50 条里 unscored 占 24 条 —— 把它并进"中性"会让情绪面看起来比真实
        更"平衡"，是一类不会报错的错数据。
        """
        s = open("web/templates/system.html", encoding="utf-8").read()
        assert "判不出" in s
        assert "不是中性" in s


class TestWakeAndAuditEndpoints:
    """两个新端点的**行为**守卫（不只是静态文本）。"""

    def test_wake_endpoint_shape(self, monkeypatch):
        """无唤醒记录时也要有稳定结构（`armed`/`events` 为空列表，不是 None）。"""
        import db.store as store
        from web.server import desk

        monkeypatch.setattr(store, "wake_event_stats", lambda **_kw: {
            "total": 0, "last_date": None, "by_persona": [],
            "skip_reasons": [], "events": []})
        monkeypatch.setattr(store, "list_wake_conditions",
                            lambda **_kw: [{"persona": "ai_tech", "cond_id": 1,
                                            "code": "600519", "kind": "stop_loss",
                                            "threshold": 1300.0,
                                            "threshold_pct": None, "note": "止盈保护"}])
        z = desk.wake()
        assert z["ok"] is True
        d = z["data"]
        assert d["armed_count"] == 1
        assert d["events"] == []
        assert d["labels"] is not None

    def test_wake_parses_action_json(self, monkeypatch):
        """`action` 是 JSON 字符串 ⇒ 必须解析成动作名；解析失败**原样保留**，不吞证据。"""
        import db.store as store
        from web.server import desk

        monkeypatch.setattr(store, "wake_event_stats", lambda **_kw: {
            "total": 2, "last_date": "2026-10-26", "by_persona": [],
            "skip_reasons": [],
            "events": [
                {"persona": "ai_tech", "trade_date": "2026-10-26", "code": "601939",
                 "cond_id": 8, "hit_at": "09:30:00", "trigger_px": 11.27,
                 "action": '[{"action": "sell", "code": "601939"}]',
                 "fired": 1, "skip_reason": ""},
                {"persona": "ai_tech", "trade_date": "2026-10-26", "code": "600000",
                 "cond_id": 9, "hit_at": "10:00:00", "trigger_px": 8.0,
                 "action": "not-json", "fired": 1, "skip_reason": ""},
            ]})
        monkeypatch.setattr(store, "list_wake_conditions", lambda **_kw: [])
        ev = desk.wake()["data"]["events"]
        assert ev[0]["actions"] == ["sell"]
        assert ev[1]["actions"] == ["not-json"], "解析失败要留着原文，不能变成空数组"

    def test_wake_labels_use_name_field(self, monkeypatch):
        """人设中文名取自 `load_personas()` 的 **`name`** 字段（不是 id/persona）。

        真实踩过：按 dict 解包 + 猜字段名 ⇒ `labels={}`，页面上只剩 `ai_tech`
        这种内部 id，用户看不懂是哪个流派。
        """
        import db.store as store
        from web.server import desk

        monkeypatch.setattr(store, "wake_event_stats", lambda **_kw: {
            "total": 0, "last_date": None, "by_persona": [],
            "skip_reasons": [], "events": []})
        monkeypatch.setattr(store, "list_wake_conditions", lambda **_kw: [])
        monkeypatch.setattr("core.ai_personas.load_personas", lambda: [
            {"name": "ai_tech", "label": "技术派"}])
        assert desk.wake()["data"]["labels"] == {"ai_tech": "技术派"}

    def test_audit_blocks_degrade_independently(self, monkeypatch):
        """一块读失败不许拖垮另外两块（页面上就是那一块显示原因）。"""
        import db.store as store
        from web.server import system

        def _boom(**_kw):
            raise RuntimeError("库锁了")

        monkeypatch.setattr(store, "list_agent_changes", _boom)
        monkeypatch.setattr(store, "list_push_log",
                            lambda **_kw: [{"day": "2026-10-26", "channel": "飞书",
                                            "title": "t", "status": "ok",
                                            "created_at": "2026-10-26 15:53:21"}])
        monkeypatch.setattr(store, "sentiment_label_summary", lambda **_kw: {
            "total": 5, "per_day": [], "by_labeler": []})
        z = system.audit()
        assert z["ok"] is True
        d = z["data"]
        assert "changes" in d["errors"] and "库锁了" in d["errors"]["changes"]
        assert len(d["pushes"]) == 1
        assert d["counts"]["pushes_ok"] == 1
        assert d["counts"]["sentiment"] == 5

    def test_audit_counts_consistent(self, monkeypatch):
        """`counts` 与实际返回条数一致（页头写"4 条"就必须真有 4 行）。"""
        import db.store as store
        from web.server import system

        monkeypatch.setattr(store, "list_agent_changes", lambda **_kw: [{"id": 1}] * 4)
        monkeypatch.setattr(store, "list_push_log", lambda **_kw: [
            {"status": "ok"}, {"status": "failed"}])
        monkeypatch.setattr(store, "sentiment_label_summary",
                            lambda **_kw: {"total": 50, "per_day": [], "by_labeler": []})
        c = system.audit()["data"]["counts"]
        assert c == {"changes": 4, "pushes": 2, "pushes_ok": 1,
                     "pushes_failed": 1, "sentiment": 50}

    # ---------------- 2026-10-26 第二批：AI 唤醒 / 审计留痕 ----------------
    def test_desk_page_calls_wake(self):
        """交易台必须真的调 `/api/desk/wake`（AI 盘中唤醒）。

        历史：`ai_wake_conditions`(60) / `ai_wake_events`(26) 一直在写、web 层零引用。
        用户看得到"每天 15:58 决策一次"，看不到"AI 在盘中被价格唤醒过 26 次、
        其中 3 次因为随机对照组压根没跑"。不摊开这一层，就无法区分"没动"与"动了没记"。
        """
        s = open("web/templates/desk.html", encoding="utf-8").read()
        assert "get('/api/desk/wake')" in s
        for anchor in ("wk-note", "wk-sum", "wk-body", "wk-cond-sum",
                       "wk-cond", "wk-foot"):
            assert anchor in s, anchor

    def test_wake_page_never_calls_armed_a_todo(self):
        """`armed=1` 不是待办：价位没到就没有事件行。

        页面文案必须是"盯盘中 / 仍生效，不代表即将触发"，不许写成"待触发 N 条"
        —— 后者会让人以为有 N 个信号在排队等着执行。
        """
        s = open("web/templates/desk.html", encoding="utf-8").read()
        assert "仍生效，不代表即将触发" in s
        assert "待触发" not in s, "armed 不是待办清单，不能写成待触发"

    def test_wake_skip_reason_is_surfaced(self):
        """命中但未执行必须显示原因（如"随机对照组不跑盘中决策"）。"""
        s = open("web/templates/desk.html", encoding="utf-8").read()
        assert "skip_reason" in s
        assert "命中但未执行" in s

    def test_system_page_calls_audit(self):
        """系统页必须真的调 `/api/system/audit`（配置变更 / 推送 / 情绪标注留痕）。"""
        s = open("web/templates/system.html", encoding="utf-8").read()
        assert "get('/api/system/audit" in s
        for anchor in ("chg-note", "chg-body", "push-note", "push-body",
                       "sent-note", "sent-body"):
            assert anchor in s, anchor

    def test_sentiment_unscored_not_neutral(self):
        """`stance='unscored'` 是"判不出方向"，页面必须单独成列并说明它不算中性。

        实测 50 条里 unscored 占 24 条 —— 把它并进"中性"会让情绪面看起来比真实
        更"平衡"，是一类不会报错的错数据。
        """
        s = open("web/templates/system.html", encoding="utf-8").read()
        assert "判不出" in s
        assert "不是中性" in s


class TestWakeAndAuditEndpoints:
    """两个新端点的**行为**守卫（不只是静态文本）。"""

    def test_wake_endpoint_shape(self, monkeypatch):
        """无唤醒记录时也要有稳定结构（`armed`/`events` 为空列表，不是 None）。"""
        import db.store as store
        from web.server import desk

        monkeypatch.setattr(store, "wake_event_stats", lambda **_kw: {
            "total": 0, "last_date": None, "by_persona": [],
            "skip_reasons": [], "events": []})
        monkeypatch.setattr(store, "list_wake_conditions",
                            lambda **_kw: [{"persona": "ai_tech", "cond_id": 1,
                                            "code": "600519", "kind": "stop_loss",
                                            "threshold": 1300.0,
                                            "threshold_pct": None, "note": "止盈保护"}])
        z = desk.wake()
        assert z["ok"] is True
        d = z["data"]
        assert d["armed_count"] == 1
        assert d["events"] == []
        assert d["labels"] is not None

    def test_wake_parses_action_json(self, monkeypatch):
        """`action` 是 JSON 字符串 ⇒ 必须解析成动作名；解析失败**原样保留**，不吞证据。"""
        import db.store as store
        from web.server import desk

        monkeypatch.setattr(store, "wake_event_stats", lambda **_kw: {
            "total": 2, "last_date": "2026-10-26", "by_persona": [],
            "skip_reasons": [],
            "events": [
                {"persona": "ai_tech", "trade_date": "2026-10-26", "code": "601939",
                 "cond_id": 8, "hit_at": "09:30:00", "trigger_px": 11.27,
                 "action": '[{"action": "sell", "code": "601939"}]',
                 "fired": 1, "skip_reason": ""},
                {"persona": "ai_tech", "trade_date": "2026-10-26", "code": "600000",
                 "cond_id": 9, "hit_at": "10:00:00", "trigger_px": 8.0,
                 "action": "not-json", "fired": 1, "skip_reason": ""},
            ]})
        monkeypatch.setattr(store, "list_wake_conditions", lambda **_kw: [])
        ev = desk.wake()["data"]["events"]
        assert ev[0]["actions"] == ["sell"]
        assert ev[1]["actions"] == ["not-json"], "解析失败要留着原文，不能变成空数组"

    def test_wake_labels_use_name_field(self, monkeypatch):
        """人设中文名取自 `load_personas()` 的 **`name`** 字段（不是 id/persona）。

        真实踩过：按 dict 解包 + 猜字段名 ⇒ labels 为空，页面上只剩 `ai_tech`
        这种内部 id，用户看不懂是哪个流派。
        """
        import db.store as store
        from web.server import desk

        monkeypatch.setattr(store, "wake_event_stats", lambda **_kw: {
            "total": 0, "last_date": None, "by_persona": [],
            "skip_reasons": [], "events": []})
        monkeypatch.setattr(store, "list_wake_conditions", lambda **_kw: [])
        monkeypatch.setattr("core.ai_personas.load_personas", lambda: [
            {"name": "ai_tech", "label": "技术派"}])
        assert desk.wake()["data"]["labels"] == {"ai_tech": "技术派"}

    def test_audit_blocks_degrade_independently(self, monkeypatch):
        """一块读失败不许拖垮另外两块（页面上就是那一块显示原因）。"""
        import db.store as store
        from web.server import system

        def _boom(**_kw):
            raise RuntimeError("库锁了")

        monkeypatch.setattr(store, "list_agent_changes", _boom)
        monkeypatch.setattr(store, "list_push_log",
                            lambda **_kw: [{"day": "2026-10-26", "channel": "飞书",
                                            "title": "t", "status": "ok",
                                            "created_at": "2026-10-26 15:53:21"}])
        monkeypatch.setattr(store, "sentiment_label_summary", lambda **_kw: {
            "total": 5, "per_day": [], "by_labeler": []})
        z = system.audit()
        assert z["ok"] is True
        d = z["data"]
        assert "changes" in d["errors"] and "库锁了" in d["errors"]["changes"]
        assert len(d["pushes"]) == 1
        assert d["counts"]["pushes_ok"] == 1
        assert d["counts"]["sentiment"] == 5

    def test_audit_counts_consistent(self, monkeypatch):
        """`counts` 与实际返回条数一致（页头写 4 条就必须真有 4 行）。"""
        import db.store as store
        from web.server import system

        monkeypatch.setattr(store, "list_agent_changes", lambda **_kw: [{"id": 1}] * 4)
        monkeypatch.setattr(store, "list_push_log", lambda **_kw: [
            {"status": "ok"}, {"status": "failed"}])
        monkeypatch.setattr(store, "sentiment_label_summary",
                            lambda **_kw: {"total": 50, "per_day": [], "by_labeler": []})
        c = system.audit()["data"]["counts"]
        assert c == {"changes": 4, "pushes": 2, "pushes_ok": 1,
                     "pushes_failed": 1, "sentiment": 50}

    # ---------------- 2026-10-26 第二批：AI 唤醒 / 审计留痕 ----------------
    def test_desk_page_calls_wake(self):
        """交易台必须真的调 `/api/desk/wake`（AI 盘中唤醒）。

        历史：`ai_wake_conditions`(60) / `ai_wake_events`(26) 一直在写、web 层零引用。
        用户看得到"每天 15:58 决策一次"，看不到"AI 在盘中被价格唤醒过 26 次、
        其中 3 次因为随机对照组压根没跑"。不摊开这一层，就无法区分"没动"与"动了没记"。
        """
        s = open("web/templates/desk.html", encoding="utf-8").read()
        assert "get('/api/desk/wake')" in s
        for anchor in ("wk-note", "wk-sum", "wk-body", "wk-cond-sum",
                       "wk-cond", "wk-foot"):
            assert anchor in s, anchor

    def test_wake_page_never_calls_armed_a_todo(self):
        """`armed=1` 不是待办：价位没到就没有事件行。

        页面文案必须是"盯盘中 / 仍生效，不代表即将触发"，不许写成"待触发 N 条"
        —— 后者会让人以为有 N 个信号在排队等着执行。
        """
        s = open("web/templates/desk.html", encoding="utf-8").read()
        assert "仍生效，不代表即将触发" in s
        assert "待触发" not in s, "armed 不是待办清单，不能写成待触发"

    def test_wake_skip_reason_is_surfaced(self):
        """命中但未执行必须显示原因（如"随机对照组不跑盘中决策"）。"""
        s = open("web/templates/desk.html", encoding="utf-8").read()
        assert "skip_reason" in s
        assert "命中但未执行" in s

    def test_system_page_calls_audit(self):
        """系统页必须真的调 `/api/system/audit`（配置变更 / 推送 / 情绪标注留痕）。"""
        s = open("web/templates/system.html", encoding="utf-8").read()
        assert "get('/api/system/audit" in s
        for anchor in ("chg-note", "chg-body", "push-note", "push-body",
                       "sent-note", "sent-body"):
            assert anchor in s, anchor

    def test_sentiment_unscored_not_neutral(self):
        """`stance='unscored'` 是"判不出方向"，页面必须单独成列并说明它不算中性。

        实测 50 条里 unscored 占 24 条 —— 把它并进"中性"会让情绪面看起来比真实
        更"平衡"，是一类不会报错的错数据。
        """
        s = open("web/templates/system.html", encoding="utf-8").read()
        assert "判不出" in s
        assert "不是中性" in s


class TestWakeAndAuditEndpoints:
    """两个新端点的**行为**守卫（不只是静态文本）。"""

    def test_wake_endpoint_shape(self, monkeypatch):
        """无唤醒记录时也要有稳定结构（`armed`/`events` 为空列表，不是 None）。"""
        import db.store as store
        from web.server import desk

        monkeypatch.setattr(store, "wake_event_stats", lambda **_kw: {
            "total": 0, "last_date": None, "by_persona": [],
            "skip_reasons": [], "events": []})
        monkeypatch.setattr(store, "list_wake_conditions",
                            lambda **_kw: [{"persona": "ai_tech", "cond_id": 1,
                                            "code": "600519", "kind": "stop_loss",
                                            "threshold": 1300.0,
                                            "threshold_pct": None, "note": "止盈保护"}])
        z = desk.wake()
        assert z["ok"] is True
        d = z["data"]
        assert d["armed_count"] == 1
        assert d["events"] == []
        assert d["labels"] is not None

    def test_wake_parses_action_json(self, monkeypatch):
        """`action` 是 JSON 字符串 ⇒ 必须解析成动作名；解析失败**原样保留**，不吞证据。"""
        import db.store as store
        from web.server import desk

        monkeypatch.setattr(store, "wake_event_stats", lambda **_kw: {
            "total": 2, "last_date": "2026-10-26", "by_persona": [],
            "skip_reasons": [],
            "events": [
                {"persona": "ai_tech", "trade_date": "2026-10-26", "code": "601939",
                 "cond_id": 8, "hit_at": "09:30:00", "trigger_px": 11.27,
                 "action": '[{"action": "sell", "code": "601939"}]',
                 "fired": 1, "skip_reason": ""},
                {"persona": "ai_tech", "trade_date": "2026-10-26", "code": "600000",
                 "cond_id": 9, "hit_at": "10:00:00", "trigger_px": 8.0,
                 "action": "not-json", "fired": 1, "skip_reason": ""},
            ]})
        monkeypatch.setattr(store, "list_wake_conditions", lambda **_kw: [])
        ev = desk.wake()["data"]["events"]
        assert ev[0]["actions"] == ["sell"]
        assert ev[1]["actions"] == ["not-json"], "解析失败要留着原文，不能变成空数组"

    def test_wake_labels_use_name_field(self, monkeypatch):
        """人设中文名取自 `load_personas()` 的 **`name`** 字段（不是 id/persona）。

        真实踩过：按 dict 解包 + 猜字段名 ⇒ labels 为空，页面上只剩 `ai_tech`
        这种内部 id，用户看不懂是哪个流派。
        """
        import db.store as store
        from web.server import desk

        monkeypatch.setattr(store, "wake_event_stats", lambda **_kw: {
            "total": 0, "last_date": None, "by_persona": [],
            "skip_reasons": [], "events": []})
        monkeypatch.setattr(store, "list_wake_conditions", lambda **_kw: [])
        monkeypatch.setattr("core.ai_personas.load_personas", lambda: [
            {"name": "ai_tech", "label": "技术派"}])
        assert desk.wake()["data"]["labels"] == {"ai_tech": "技术派"}

    def test_audit_blocks_degrade_independently(self, monkeypatch):
        """一块读失败不许拖垮另外两块（页面上就是那一块显示原因）。"""
        import db.store as store
        from web.server import system

        def _boom(**_kw):
            raise RuntimeError("库锁了")

        monkeypatch.setattr(store, "list_agent_changes", _boom)
        monkeypatch.setattr(store, "list_push_log",
                            lambda **_kw: [{"day": "2026-10-26", "channel": "飞书",
                                            "title": "t", "status": "ok",
                                            "created_at": "2026-10-26 15:53:21"}])
        monkeypatch.setattr(store, "sentiment_label_summary", lambda **_kw: {
            "total": 5, "per_day": [], "by_labeler": []})
        z = system.audit()
        assert z["ok"] is True
        d = z["data"]
        assert "changes" in d["errors"] and "库锁了" in d["errors"]["changes"]
        assert len(d["pushes"]) == 1
        assert d["counts"]["pushes_ok"] == 1
        assert d["counts"]["sentiment"] == 5

    def test_audit_counts_consistent(self, monkeypatch):
        """`counts` 与实际返回条数一致（页头写 4 条就必须真有 4 行）。"""
        import db.store as store
        from web.server import system

        monkeypatch.setattr(store, "list_agent_changes", lambda **_kw: [{"id": 1}] * 4)
        monkeypatch.setattr(store, "list_push_log", lambda **_kw: [
            {"status": "ok"}, {"status": "failed"}])
        monkeypatch.setattr(store, "sentiment_label_summary",
                            lambda **_kw: {"total": 50, "per_day": [], "by_labeler": []})
        c = system.audit()["data"]["counts"]
        assert c == {"changes": 4, "pushes": 2, "pushes_ok": 1,
                     "pushes_failed": 1, "sentiment": 50}
