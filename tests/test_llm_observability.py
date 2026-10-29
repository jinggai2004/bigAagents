"""LLM 调用链可观测性测试（Phase 6.2）

覆盖三件事：
  1. `llm_calls` 表的写入与统计口径（含失败率、按用途/模型分组）；
  2. `call_api` / `chat` 的埋点：成功与失败都要留下记录（失败耗时同样重要），
     且**记录失败绝不能让一次成功的调用变成失败**；
  3. 用途标签的来源优先级：显式参数 > llm_purpose() 上下文 > unknown。

注意：本文件的每个用例都隔离 SQLite（db.models.DB_PATH），否则会往真实的
data/portfolio.db 里写假调用记录。
"""

import threading
import time

import pytest
from unittest.mock import MagicMock, patch

from llm.deepseek_client import (DeepSeekClient, LLMError, current_llm_purpose,
                                 llm_purpose, reset_client_cache)


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    db_path = use_template_db(tmp_path, monkeypatch, "llm.db")
    yield


@pytest.fixture
def client():
    with patch("llm.deepseek_client._get_client_for_model") as mock_get:
        c = DeepSeekClient()
        c._openai = mock_get.return_value
        yield c


def _fake_response(content="OK", prompt_tokens=120, completion_tokens=30):
    msg = MagicMock()
    msg.content = content
    msg.reasoning_content = None
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    if prompt_tokens is None:
        resp.usage = None
    else:
        resp.usage = MagicMock(prompt_tokens=prompt_tokens,
                               completion_tokens=completion_tokens,
                               total_tokens=prompt_tokens + completion_tokens)
    return resp


# ================================================================
# 表与统计
# ================================================================

class TestStore:
    def test_insert_and_read(self):
        from db import store
        call_id = store.insert_llm_call({
            "model": "deepseek-chat", "purpose": "agents.tech", "ok": True,
            "latency_ms": 1500, "prompt_tokens": 100, "completion_tokens": 50,
            "total_tokens": 150, "prompt_chars": 400, "response_chars": 200,
        })
        assert call_id > 0
        rows = store.get_recent_llm_calls(limit=5)
        assert len(rows) == 1
        assert rows[0]["purpose"] == "agents.tech"
        assert rows[0]["ok"] == 1 and rows[0]["latency_ms"] == 1500
        assert rows[0]["total_tokens"] == 150

    def test_stats_aggregate(self):
        from db import store
        store.insert_llm_call({"model": "deepseek-chat", "purpose": "agents.tech",
                               "ok": True, "latency_ms": 1000, "prompt_tokens": 100,
                               "completion_tokens": 20, "total_tokens": 120})
        store.insert_llm_call({"model": "deepseek-chat", "purpose": "agents.tech",
                               "ok": True, "latency_ms": 3000, "prompt_tokens": 200,
                               "completion_tokens": 40, "total_tokens": 240})
        store.insert_llm_call({"model": "qwen-plus", "purpose": "nl_screener.parse",
                               "ok": False, "latency_ms": 500, "error": "429"})
        s = store.llm_call_stats(recent_days=7)
        assert s["total"] == 3 and s["failed"] == 1
        assert s["failure_rate"] == pytest.approx(1 / 3, abs=1e-3)
        assert s["avg_latency_ms"] == 2000          # 只统计成功的调用
        assert s["prompt_tokens"] == 300 and s["completion_tokens"] == 60
        by_purpose = {g["key"]: g for g in s["by_purpose"]}
        assert by_purpose["agents.tech"]["count"] == 2
        assert by_purpose["agents.tech"]["failed"] == 0
        assert by_purpose["agents.tech"]["total_tokens"] == 360
        assert by_purpose["nl_screener.parse"]["failed"] == 1
        by_model = {g["key"]: g for g in s["by_model"]}
        assert by_model["qwen-plus"]["failed"] == 1

    def test_empty_stats(self):
        from db import store
        s = store.llm_call_stats(recent_days=7)
        assert s["total"] == 0 and s["failure_rate"] == 0.0 and s["avg_latency_ms"] == 0

    def test_stats_respects_window(self):
        from db import store
        import db.store as store_mod
        store.insert_llm_call({"model": "m", "purpose": "old", "ok": True,
                               "created_at": "2020-01-01 00:00:00"})
        store.insert_llm_call({"model": "m", "purpose": "new", "ok": True})
        s = store.llm_call_stats(recent_days=7)
        assert s["total"] == 1
        assert s["by_purpose"][0]["key"] == "new"

    def test_only_failures_filter(self):
        from db import store
        store.insert_llm_call({"model": "m", "purpose": "a", "ok": True})
        store.insert_llm_call({"model": "m", "purpose": "b", "ok": False, "error": "x"})
        assert len(store.get_recent_llm_calls(only_failures=True)) == 1
        assert store.get_recent_llm_calls(only_failures=True)[0]["purpose"] == "b"
        assert len(store.get_recent_llm_calls(purpose="a")) == 1

    def test_cleanup_covers_llm_calls(self):
        from db import store
        store.insert_llm_call({"model": "m", "purpose": "old", "ok": True,
                               "created_at": "2000-01-01 00:00:00"})
        deleted = store.cleanup()
        assert deleted.get("llm_calls") == 1
        assert store.get_recent_llm_calls() == []

    def test_store_stats_includes_llm_calls(self):
        from db import store
        store.insert_llm_call({"model": "m", "purpose": "p", "ok": True})
        assert "llm_calls" in store.store_stats()
        assert store.store_stats()["llm_calls"]["rows"] == 1

    def test_write_failure_is_swallowed(self, monkeypatch):
        """可观测性写入失败不能抛给调用方"""
        from db import store
        monkeypatch.setattr(store, "get_connection",
                            lambda: (_ for _ in ()).throw(sqlite_error()))
        assert store.insert_llm_call({"model": "m", "purpose": "p", "ok": True}) == 0


def sqlite_error():
    import sqlite3
    return sqlite3.OperationalError("database is locked")


# ================================================================
# call_api 埋点
# ================================================================

class TestCallApiInstrumentation:
    def test_success_recorded_with_tokens_and_latency(self, client):
        from db import store
        client._openai.chat.completions.create.return_value = _fake_response("正文", 100, 20)
        out = client.call_api([{"role": "user", "content": "hi"}], purpose="agents.tech")
        assert out == "正文"
        row = store.get_recent_llm_calls()[0]
        assert row["ok"] == 1
        assert row["purpose"] == "agents.tech"
        assert row["prompt_tokens"] == 100 and row["completion_tokens"] == 20
        assert row["total_tokens"] == 120
        assert row["prompt_chars"] == 2 and row["response_chars"] == 2
        assert row["model"]                       # 记录了模型名
        assert row["latency_ms"] >= 0

    def test_usage_absent_is_still_recorded(self, client):
        """有些兼容端点不返回 usage：token 记空，但调用本身要留痕"""
        from db import store
        client._openai.chat.completions.create.return_value = _fake_response("x", None)
        client.call_api([{"role": "user", "content": "hi"}])
        row = store.get_recent_llm_calls()[0]
        assert row["ok"] == 1 and row["total_tokens"] is None

    def test_failure_recorded_and_reraised(self, client):
        from db import store
        client._openai.chat.completions.create.side_effect = RuntimeError("429 Too Many Requests")
        with pytest.raises(LLMError):
            client.call_api([{"role": "user", "content": "hi"}], purpose="agents.bull")
        row = store.get_recent_llm_calls()[0]
        assert row["ok"] == 0
        assert row["purpose"] == "agents.bull"
        assert "429" in row["error"]
        assert row["total_tokens"] is None
        assert row["latency_ms"] >= 0            # 失败耗时同样记录

    def test_empty_content_counts_as_failure(self, client):
        from db import store
        client._openai.chat.completions.create.return_value = _fake_response("")
        with pytest.raises(LLMError):
            client.call_api([{"role": "user", "content": "hi"}])
        row = store.get_recent_llm_calls()[0]
        assert row["ok"] == 0 and "空内容" in row["error"]

    def test_logging_failure_does_not_break_success(self, client, monkeypatch):
        """最关键的一条：写记录失败时，成功的调用必须照常返回"""
        from db import store
        monkeypatch.setattr(store, "insert_llm_call",
                            lambda record: (_ for _ in ()).throw(RuntimeError("db down")))
        client._openai.chat.completions.create.return_value = _fake_response("正文")
        assert client.call_api([{"role": "user", "content": "hi"}]) == "正文"

    def test_logging_failure_does_not_mask_llm_error(self, client, monkeypatch):
        from db import store
        monkeypatch.setattr(store, "insert_llm_call",
                            lambda record: (_ for _ in ()).throw(RuntimeError("db down")))
        client._openai.chat.completions.create.side_effect = RuntimeError("boom")
        with pytest.raises(LLMError) as e:
            client.call_api([{"role": "user", "content": "hi"}])
        assert "boom" in str(e.value)


# ================================================================
# 用途标签
# ================================================================

class TestPurpose:
    def test_explicit_purpose_wins(self, client):
        from db import store
        client._openai.chat.completions.create.return_value = _fake_response("x", None)
        with llm_purpose("外层标签"):
            client.call_api([{"role": "user", "content": "hi"}], purpose="显式标签")
        assert store.get_recent_llm_calls()[0]["purpose"] == "显式标签"

    def test_context_purpose_used(self, client):
        from db import store
        client._openai.chat.completions.create.return_value = _fake_response("x", None)
        with llm_purpose("agents.risk_manager"):
            client.call_api([{"role": "user", "content": "hi"}])
        assert store.get_recent_llm_calls()[0]["purpose"] == "agents.risk_manager"

    def test_default_is_unknown(self, client):
        from db import store
        client._openai.chat.completions.create.return_value = _fake_response("x", None)
        client.call_api([{"role": "user", "content": "hi"}])
        assert store.get_recent_llm_calls()[0]["purpose"] == "unknown"

    def test_context_restored_after_exit(self, client):
        from db import store
        client._openai.chat.completions.create.return_value = _fake_response("x", None)
        with llm_purpose("内层"):
            pass
        client.call_api([{"role": "user", "content": "hi"}])
        assert store.get_recent_llm_calls()[0]["purpose"] == "unknown"
        assert current_llm_purpose() == "unknown"

    def test_nested_context_inner_wins(self, client):
        from db import store
        client._openai.chat.completions.create.return_value = _fake_response("x", None)
        with llm_purpose("外层"):
            with llm_purpose("内层"):
                assert current_llm_purpose() == "内层"
            assert current_llm_purpose() == "外层"
        client.call_api([{"role": "user", "content": "hi"}])
        assert store.get_recent_llm_calls()[0]["purpose"] == "unknown"

    def test_label_is_per_thread(self, client):
        """用 ContextVar 而非全局变量：两个线程的标签不能互相串"""
        from db import store
        client._openai.chat.completions.create.return_value = _fake_response("x", None)
        seen = {}

        def worker(label):
            with llm_purpose(label):
                time.sleep(0.05)
                seen[label] = current_llm_purpose()
                client.call_api([{"role": "user", "content": label}])

        threads = [threading.Thread(target=worker, args=(name,))
                   for name in ("agents.bull", "agents.bear")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        assert seen == {"agents.bull": "agents.bull", "agents.bear": "agents.bear"}
        purposes = {r["purpose"] for r in store.get_recent_llm_calls()}
        assert purposes == {"agents.bull", "agents.bear"}


# ================================================================
# chat 埋点
# ================================================================

class TestChatInstrumentation:
    def _fake_chat(self, client, content="回答", tool_calls=None):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls
        resp = MagicMock()
        resp.choices = [MagicMock(message=msg)]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        client._openai.chat.completions.create.return_value = resp
        return resp

    def test_chat_recorded(self, client):
        from db import store
        self._fake_chat(client)
        msg = client.chat([{"role": "user", "content": "hi"}], purpose="chat.tool_loop")
        assert msg.content == "回答"
        row = store.get_recent_llm_calls()[0]
        assert row["purpose"] == "chat.tool_loop" and row["ok"] == 1
        assert row["total_tokens"] == 15

    def test_chat_failure_recorded_and_reraised(self, client):
        from db import store
        client._openai.chat.completions.create.side_effect = RuntimeError("connection reset")
        with pytest.raises(RuntimeError):
            client.chat([{"role": "user", "content": "hi"}], purpose="chat.tool_loop")
        row = store.get_recent_llm_calls()[0]
        assert row["ok"] == 0 and "connection reset" in row["error"]

    def test_chat_stream_recorded_after_completion(self, client):
        from db import store
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content="片段"))]
        client._openai.chat.completions.create.return_value = [chunk, chunk]
        out = "".join(client.chat_stream([{"role": "user", "content": "hi"}],
                                         purpose="chat.answer"))
        assert out == "片段片段"
        row = store.get_recent_llm_calls()[0]
        assert row["ok"] == 1 and row["response_chars"] == 2


# ================================================================
# 与真实 Agent 节点的接线
# ================================================================

class TestAgentWiring:
    def _fake_openai(self, content="报告正文"):
        fake = MagicMock()
        msg = MagicMock()
        msg.content = content
        msg.reasoning_content = None
        resp = MagicMock()
        resp.choices = [MagicMock(message=msg)]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        fake.chat.completions.create.return_value = resp
        return fake

    def test_agent_node_call_is_recorded(self, monkeypatch):
        """端到端：真跑一个 Agent 节点 → 真 call_api → llm_calls 里出现该节点的标签"""
        from db import store
        import core.agents as agents_mod

        fake = self._fake_openai()
        monkeypatch.setattr("llm.deepseek_client._get_client_for_model", lambda model: fake)

        state = {"stock_code": "600519", "stock_info": {"name": "贵州茅台"},
                 "kline_data": None, "indicators": {}, "errors": [], "fatal_errors": []}
        out = agents_mod.tech_analyst_node(state)
        assert out["tech_report"] == "报告正文"

        row = store.get_recent_llm_calls()[0]
        assert row["ok"] == 1
        assert row["purpose"] == "client.technical_analysis"
        assert row["total_tokens"] == 15

    def test_agent_node_failure_is_recorded(self, monkeypatch):
        from db import store
        import core.agents as agents_mod

        fake = MagicMock()
        fake.chat.completions.create.side_effect = RuntimeError("rate limited")
        monkeypatch.setattr("llm.deepseek_client._get_client_for_model", lambda model: fake)

        state = {"stock_code": "600519", "stock_info": {}, "kline_data": None,
                 "indicators": {}, "errors": [], "fatal_errors": []}
        out = agents_mod.tech_analyst_node(state)
        assert out["tech_report"] == "" and out["errors"]
        row = store.get_recent_llm_calls()[0]
        assert row["ok"] == 0 and "rate limited" in row["error"]

    def test_every_agent_call_site_is_labeled(self):
        """契约：每个 Agent 的 call_api 都要带 purpose 标签。

        否则统计里会悄悄多出 "unknown" 用途，而"哪一步最慢/最容易失败"就答不出来。
        新增节点时这条会红，提醒顺手打标。
        """
        import inspect
        import re
        import core.agents as agents_mod
        src = inspect.getsource(agents_mod)
        calls = len(re.findall(r"client\.call_api\(", src))
        labeled = len(re.findall(r'purpose="agents\.', src))
        assert calls == labeled, f"call_api 调用 {calls} 处，只有 {labeled} 处打了标签"
        assert calls >= 7
