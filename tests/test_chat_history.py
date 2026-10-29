"""对话历史持久化单测：chat_sessions 会话维 + chat_logs 消息维（临时库隔离）

覆盖动机：此前对话"只写不读"——session_id 只活在 st.session_state，刷新即丢。
本文件锁死"写消息即登记会话 / 会话可列表 / 可回放 / 可删除 / 老数据能补登记"这几条。
"""
import sqlite3
import pytest

from db import store as db_store
from core import chat_history as ch


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    """临时数据库（隔离真实 portfolio.db）"""
    from tests.conftest import use_template_db
    db_path = use_template_db(tmp_path, monkeypatch, "test_chat.db")
    yield str(db_path)


def _raw_insert_log(db_path, session_id, role, content, created_at="2026-01-01 00:00:00"):
    """绕过 store 直接写消息，模拟升级前"只有 chat_logs"的老数据"""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO chat_logs (session_id, role, content, tools_used, created_at)"
        " VALUES (?, ?, ?, '[]', ?)", (session_id, role, content, created_at))
    conn.commit()
    conn.close()


class TestStoreSessionTable:
    """store 层：会话维表"""

    def test_insert_log_registers_session(self, temp_store):
        """写消息必须顺带登记会话，否则列表无从下手（P1 的根因）"""
        db_store.insert_chat_log("s1", "user", "你好", prompt_tokens=10, completion_tokens=5)
        sess = db_store.get_chat_session("s1")
        assert sess is not None
        assert sess["message_count"] == 1
        assert sess["prompt_tokens"] == 10

    def test_counters_accumulate(self, temp_store):
        db_store.insert_chat_log("s1", "user", "a", prompt_tokens=10, completion_tokens=5)
        db_store.insert_chat_log("s1", "assistant", "b", tools_used=["t1"])
        db_store.insert_chat_log("s1", "user", "c", prompt_tokens=7, completion_tokens=3)
        sess = db_store.get_chat_session("s1")
        assert sess["message_count"] == 3
        assert sess["prompt_tokens"] == 17
        assert sess["completion_tokens"] == 8

    def test_insert_log_without_session_id(self, temp_store):
        """session_id 为空时不建会话行（不产生 'None' 垃圾会话）"""
        db_store.insert_chat_log(None, "user", "hi")
        assert db_store.list_chat_sessions() == []

    def test_create_session_idempotent(self, temp_store):
        assert db_store.create_chat_session("s1", title="t", persona="综合") == 1
        assert db_store.create_chat_session("s1", title="别的") == 0
        assert db_store.get_chat_session("s1")["title"] == "t"

    def test_update_session_whitelist(self, temp_store):
        db_store.create_chat_session("s1")
        db_store.update_chat_session("s1", {"title": "茅台资金面", "evil": "x"})
        sess = db_store.get_chat_session("s1")
        assert sess["title"] == "茅台资金面"
        assert "evil" not in sess

    def test_list_sessions_newest_first(self, temp_store):
        db_store.create_chat_session("old")
        db_store.create_chat_session("new")
        conn = sqlite3.connect(temp_store)
        conn.execute("UPDATE chat_sessions SET updated_at = '2026-01-01 00:00:00'"
                     " WHERE session_id = 'old'")
        conn.commit()
        conn.close()
        ids = [s["session_id"] for s in db_store.list_chat_sessions()]
        assert ids == ["new", "old"]

    def test_legacy_sessions_are_recovered(self, temp_store):
        """升级前的老会话（只有 chat_logs）必须能补登记并出现在列表里"""
        _raw_insert_log(temp_store, "legacy1", "user", "帮我看茅台")
        _raw_insert_log(temp_store, "legacy1", "assistant", "好的")
        sessions = db_store.list_chat_sessions()
        assert [s["session_id"] for s in sessions] == ["legacy1"]
        assert sessions[0]["source"] == "legacy"
        assert sessions[0]["message_count"] == 2

    def test_get_chat_messages_order_and_tools(self, temp_store):
        db_store.insert_chat_log("s1", "user", "q1")
        db_store.insert_chat_log("s1", "assistant", "a1", tools_used=["get_quote"])
        db_store.insert_chat_log("s1", "user", "q2")
        msgs = db_store.get_chat_messages("s1")
        assert [m["content"] for m in msgs] == ["q1", "a1", "q2"]
        assert msgs[1]["tools_used"] == ["get_quote"]

    def test_get_chat_messages_limit_is_last_n(self, temp_store):
        for i in range(5):
            db_store.insert_chat_log("s1", "user", f"m{i}")
        msgs = db_store.get_chat_messages("s1", limit=2)
        assert [m["content"] for m in msgs] == ["m3", "m4"]

    def test_clear_messages_keeps_session(self, temp_store):
        db_store.insert_chat_log("s1", "user", "q", prompt_tokens=3)
        assert db_store.clear_chat_messages("s1") == 1
        assert db_store.get_chat_messages("s1") == []
        sess = db_store.get_chat_session("s1")
        assert sess is not None and sess["message_count"] == 0
        assert sess["prompt_tokens"] == 0

    def test_delete_session(self, temp_store):
        db_store.insert_chat_log("s1", "user", "q")
        db_store.insert_chat_log("s1", "assistant", "a")
        assert db_store.delete_chat_session("s1") == 2
        assert db_store.get_chat_session("s1") is None
        assert db_store.get_chat_messages("s1") == []

    def test_chat_stats(self, temp_store):
        db_store.insert_chat_log("s1", "user", "q", prompt_tokens=10, completion_tokens=5)
        db_store.insert_chat_log("s1", "assistant", "a")
        st = db_store.chat_stats()
        assert st["sessions"] == 1 and st["messages"] == 2
        assert st["prompt_tokens"] == 10 and st["completion_tokens"] == 5
        assert st["recent_sessions"] == 1

    def test_cleanup_drops_empty_old_sessions(self, temp_store):
        db_store.create_chat_session("empty")
        db_store.create_chat_session("active")
        db_store.insert_chat_log("active", "user", "还在聊")
        conn = sqlite3.connect(temp_store)
        conn.execute("UPDATE chat_sessions SET updated_at = '2020-01-01 00:00:00'")
        conn.commit()
        conn.close()
        deleted = db_store.cleanup()
        assert deleted["chat_sessions"] == 1          # 只回收空壳
        assert db_store.get_chat_session("empty") is None
        assert db_store.get_chat_session("active") is not None


class TestChatHistoryModule:
    """core 层：会话生命周期与读写封装"""

    def test_new_session_id(self):
        sid = ch.new_session_id()
        assert len(sid) == ch.SESSION_ID_LENGTH
        assert sid != ch.new_session_id()
        int(sid, 16)      # hex，能塞进 URL

    def test_auto_title(self):
        long_q = "帮我看看贵州茅台最近二十个交易日的资金面变化情况"
        assert ch.auto_title(long_q) == long_q[:ch.TITLE_MAX_LEN] + "…"
        assert ch.auto_title("帮我看看茅台的资金面") == "帮我看看茅台的资金面"   # 未超长不截断
        assert ch.auto_title("  换 行\n压平  ") == "换 行 压平"
        assert ch.auto_title("") == ch.DEFAULT_TITLE

    def test_save_turn_creates_title_and_rows(self, temp_store):
        ch.save_turn("s1", "帮我看看茅台", "茅台资金流入", usage={"prompt": 8, "completion": 4},
                     tools_used=["get_fund_flow"], persona="综合", model="qwen-turbo")
        msgs = db_store.get_chat_messages("s1")
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[0]["prompt_tokens"] == 8       # token 记在提问行
        assert msgs[1]["tools_used"] == ["get_fund_flow"]
        sess = db_store.get_chat_session("s1")
        assert sess["title"] == "帮我看看茅台"
        assert sess["persona"] == "综合" and sess["model"] == "qwen-turbo"

    def test_title_not_overwritten_later(self, temp_store):
        ch.save_turn("s1", "第一句提问", "答1")
        ch.save_turn("s1", "第二句提问", "答2")
        assert db_store.get_chat_session("s1")["title"] == "第一句提问"

    def test_load_messages_shape(self, temp_store):
        ch.save_turn("s1", "q", "a")
        assert ch.load_messages("s1") == [{"role": "user", "content": "q"},
                                          {"role": "assistant", "content": "a"}]

    def test_load_messages_missing_session(self, temp_store):
        assert ch.load_messages("nope") == []
        assert ch.load_messages("") == []
        assert ch.load_session_meta("nope") == {}

    def test_list_sessions_and_label(self, temp_store):
        ch.save_turn("s1", "看看茅台", "答")
        ch.save_turn("s2", "", "答")                     # 空提问 → 标题兜底
        sessions = ch.list_sessions()
        assert len(sessions) == 2
        by_id = {s["session_id"]: s for s in sessions}
        assert ch.session_label(by_id["s1"]) == "看看茅台"
        assert ch.session_label(by_id["s2"]) == ch.DEFAULT_TITLE
        assert ch.session_label({"session_id": "abcdef123456"}) == "会话 abcdef12"

    def test_start_and_ensure_session(self, temp_store):
        sid = ch.start_session(persona="价值", model="qwen-plus")
        assert db_store.get_chat_session(sid)["persona"] == "价值"
        # ensure 不覆盖已有 persona/model
        ch.ensure_session(sid, persona="技术", model="qwen-turbo")
        assert db_store.get_chat_session(sid)["persona"] == "价值"

    def test_clear_and_delete(self, temp_store):
        ch.save_turn("s1", "q", "a")
        assert ch.clear_session("s1") == 2
        assert ch.load_messages("s1") == []
        ch.save_turn("s1", "q2", "a2")
        assert ch.delete_session("s1") == 2
        assert ch.list_sessions() == []

    def test_save_turn_swallows_storage_error(self, temp_store, monkeypatch):
        """存储挂了不能打断聊天"""
        def boom(*a, **k):
            raise RuntimeError("db down")
        monkeypatch.setattr(db_store, "insert_chat_log", boom)
        assert ch.save_turn("s1", "q", "a") is False

    def test_empty_session_id_is_noop(self, temp_store):
        assert ch.save_turn("", "q", "a") is False
        assert ch.delete_session("") == 0
        assert ch.clear_session("") == 0
