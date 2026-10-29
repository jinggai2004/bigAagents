"""对话历史持久化层：会话维（chat_sessions）+ 消息维（chat_logs）。

背景（P1 / 批次 14）
------------------
此前 `chat_logs` 是**只写不读**的：UI 把 session_id 存在 `st.session_state` 里，
浏览器一刷新 session_state 就重建 → 换号 → 历史读不出来（实测表里 12 行、页面上 0 条）。

本模块补两件事：
1. 会话维表读写（登记 / 列表 / 回放 / 删除），落库细节仍由 `db.store` 负责；
2. 把 session_id 从内存搬到 URL 查询参数 `?sid=xxx`（见 UI 层），刷新、分享链接都能回到同一会话。

设计取舍
--------
- **写入失败一律吞掉**：聊天是最不该被存储问题打断的交互，与 `analysis_history` 同策略。
- **标题自动取首条提问**：没人愿意给会话起名；首条 18 字足够区分。
- **消息数/token 在 store 层累加**（`insert_chat_log` 里），但列表展示用实时聚合，
  因为历史行可能由旧版本写入、计数器为 0。
"""

import logging
import uuid

from db import store as store

_logger = logging.getLogger(__name__)

SESSION_ID_LENGTH = 12          # 与 UI 旧实现保持一致
TITLE_MAX_LEN = 18

# 会话列表里没有标题时的兜底展示
DEFAULT_TITLE = "新对话"


# ================================================================
# 会话生命周期
# ================================================================

def new_session_id() -> str:
    """生成短会话 id（12 位 hex，够短能塞进 URL，碰撞概率可忽略）"""
    return uuid.uuid4().hex[:SESSION_ID_LENGTH]


def auto_title(text: str, max_len: int = TITLE_MAX_LEN) -> str:
    """用首条提问当会话标题：压平换行、截断、加省略号"""
    flat = " ".join((text or "").split())
    if not flat:
        return DEFAULT_TITLE
    return flat[:max_len] + ("…" if len(flat) > max_len else "")


def start_session(persona: str = None, model: str = None,
                  source: str = "ui", session_id: str = None) -> str:
    """开一个新会话（登记会话维表），返回 session_id。失败也返回 id，不影响聊天"""
    sid = session_id or new_session_id()
    try:
        store.create_chat_session(sid, title=DEFAULT_TITLE, persona=persona,
                                  model=model, source=source)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[对话历史] 创建会话失败: %s", e)
    return sid


def ensure_session(session_id: str, persona: str = None, model: str = None,
                   source: str = "ui") -> str:
    """确保会话已登记（已存在则不覆盖 persona/model），返回 session_id"""
    if not session_id:
        return start_session(persona=persona, model=model, source=source)
    try:
        store.create_chat_session(session_id, persona=persona, model=model,
                                  source=source)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[对话历史] 登记会话失败: %s", e)
    return session_id


# ================================================================
# 读写一轮对话
# ================================================================

def save_turn(session_id: str, user_msg: str, answer: str,
              usage: dict = None, tools_used: list = None,
              persona: str = None, model: str = None,
              source: str = "ui") -> bool:
    """落库一问一答，并在首条提问时定标题。

    ⚠️ token 只记在 user 行：一次调用产生的 prompt/completion 属于这一轮，
    记在 assistant 行会与"本轮用量"错位（assistant 行的 tokens 留空）。
    """
    if not session_id:
        return False
    usage = usage or {}
    try:
        ensure_session(session_id, persona=persona, model=model, source=source)
        existing = store.get_chat_messages(session_id, limit=1)
        if not existing:
            store.update_chat_session(session_id, {"title": auto_title(user_msg)})
        store.insert_chat_log(
            session_id, "user", user_msg,
            prompt_tokens=usage.get("prompt"), completion_tokens=usage.get("completion"))
        store.insert_chat_log(
            session_id, "assistant", answer or "", tools_used=tools_used or [])
        return True
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[对话历史] 保存对话失败: %s", e)
        return False


def load_messages(session_id: str, limit: int = 200) -> list[dict]:
    """回放会话，返回 ChatAgent 需要的 [{"role":..., "content":...}] 结构"""
    if not session_id:
        return []
    try:
        rows = store.get_chat_messages(session_id, limit=limit)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[对话历史] 读取会话失败: %s", e)
        return []
    return [{"role": r.get("role", "user"), "content": r.get("content", "")}
            for r in rows if r.get("role") in ("user", "assistant")]


def load_session_meta(session_id: str) -> dict:
    """会话元信息（含实时消息数与累计 token），读不到返回空 dict"""
    if not session_id:
        return {}
    try:
        return store.get_chat_session(session_id) or {}
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[对话历史] 读取会话元信息失败: %s", e)
        return {}


# ================================================================
# 列表 / 删除
# ================================================================

def list_sessions(limit: int = 30) -> list[dict]:
    """最近会话列表（UI 侧边栏）"""
    try:
        return store.list_chat_sessions(limit=limit)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[对话历史] 会话列表失败: %s", e)
        return []


def session_label(sess: dict) -> str:
    """会话在列表里的展示名：优先 title，退回首条提问，再退回 id"""
    title = (sess.get("title") or "").strip()
    first = (sess.get("first_msg") or "").strip()
    if title and title != DEFAULT_TITLE:
        return title
    if first:
        return first
    return title or f"会话 {sess.get('session_id', '—')[:8]}"


def delete_session(session_id: str) -> int:
    """删除会话及其消息，返回被删消息数"""
    if not session_id:
        return 0
    try:
        return store.delete_chat_session(session_id)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[对话历史] 删除会话失败: %s", e)
        return 0


def clear_session(session_id: str) -> int:
    """清空当前会话消息但保留会话（对应 UI「清空对话」）"""
    if not session_id:
        return 0
    try:
        return store.clear_chat_messages(session_id)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[对话历史] 清空会话失败: %s", e)
        return 0
