# -*- coding: utf-8 -*-
"""`/api/chat/*` —— 助手（P4）。

## 与 Streamlit 助手页的关系
同一套 `core.chat_agent.ChatAgent`、同一套 `core.chat_history` 落库。差别只有一处：
Streamlit 用 `st.write_stream` 消费生成器，这里把同一个生成器**搬到线程里**、
通过队列转成 WebSocket 事件。

## 为什么必须线程化
`ChatAgent.reply()` 是**同步生成器**，里面 `client.chat()` 是阻塞 HTTP（单轮 5~40s、
最多 6 轮工具调用）。直接在 async 路由里 for 循环会**焊死事件循环**——整个服务
（包括其它页面的轮询）一起停摆。所以：线程跑生成器 → `call_soon_threadsafe` 送进
`asyncio.Queue` → 协程侧 `await` 取。

## 事件契约（前端只认这几种 `type`）
| type | 字段 | 含义 |
|---|---|---|
| `session` | `session_id` | 会话已就绪（新建会话时前端要更新 URL） |
| `tool_start` | `tool` / `args` | 工具开始调用 → 渲染为"进行中"chip |
| `tool_end` | `tool` / `summary` | 工具返回一句话摘要 → chip 打勾 |
| `chart` | `chart{kind,...}` | `kline` 或 `fund5`，前端自己画 |
| `answer` | `text` | 最终回答（markdown 原文，前端渲染） |
| `error` | `error{type,message}` | 复用统一 `error.type` 枚举 |
| `done` | `usage` / `tools` | 本轮结束（**一定有**，前端据此解锁输入框） |

⚠️ **LLM 输出不回灌主链**（项目 AST 断言守着）：这里只把回答写进 `chat_logs`，
   不写 `daily_picks` / 分析快照 / 决策表。
"""
from __future__ import annotations

import logging

from web.server import envelope

_logger = logging.getLogger(__name__)

# 送进 ChatAgent 的历史条数。`core.chat_agent.MAX_HISTORY` 是它的天花板（20），
# 这里不超过它——多给的会被 agent 自己再截一次，纯浪费 IO。
HISTORY_LIMIT = 20
SESSION_LIST_LIMIT = 30


# ---------------------------------------------------------------- 会话
def sessions(limit: int = SESSION_LIST_LIMIT) -> dict:
    from core import chat_history
    rows = chat_history.list_sessions(limit=limit)
    out = [{"session_id": r.get("session_id") or r.get("id"),
            "label": chat_history.session_label(r),
            "title": r.get("title") or "",
            "persona": r.get("persona") or "",
            "model": r.get("model") or "",
            "updated_at": r.get("updated_at") or r.get("created_at") or "",
            "messages": r.get("message_count") or r.get("messages") or 0}
           for r in rows]
    return envelope.ok({"sessions": out, "total": len(out)},
                       as_of=_newest(out))


def _newest(rows: list[dict]):
    times = [r["updated_at"] for r in rows if r.get("updated_at")]
    return max(times) if times else None


def session(session_id: str, limit: int = 200) -> dict:
    """一个会话的完整回放（前端重开会话时用）。"""
    if not session_id:
        return envelope.fail("invalid", "session_id 为空")
    from core import chat_history
    meta = chat_history.load_session_meta(session_id)
    msgs = chat_history.load_messages(session_id, limit=limit)
    if not meta and not msgs:
        return envelope.fail("missing", "会话不存在")
    return envelope.ok({
        "session_id": session_id,
        "label": chat_history.session_label(meta or {"session_id": session_id}),
        "persona": (meta or {}).get("persona") or "",
        "model": (meta or {}).get("model") or "",
        "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
    }, as_of=(meta or {}).get("updated_at"))


def create(persona: str | None = None, model: str | None = None) -> dict:
    from core import chat_history
    sid = chat_history.start_session(persona=persona, model=model, source="web")
    return envelope.ok({"session_id": sid, "persona": persona or "", "model": model or ""})


def remove(session_id: str) -> dict:
    if not session_id:
        return envelope.fail("invalid", "session_id 为空")
    from core import chat_history
    n = chat_history.delete_session(session_id)
    return envelope.ok({"session_id": session_id, "deleted": n})


def personas() -> dict:
    try:
        from core.chat_agent import PERSONAS
    except Exception as e:                     # noqa: BLE001
        # `core.chat_agent` → `llm.deepseek_client` → `config.settings`，后者在
        # "一个厂商 Key 都没配"时**导入即抛** ValueError。这不是服务坏了，是配置缺失
        # ⇒ 返 `auth`，让页面显示"缺密钥"而不是一个 500 栈。
        _logger.warning("[chat] 视角清单不可用（配置缺失）: %s: %s", type(e).__name__, e)
        return envelope.fail("auth", f"助手不可用：{type(e).__name__}: {e}")
    return envelope.ok({"personas": list(PERSONAS.keys())})


# ---------------------------------------------------------------- 一轮对话
def turn_events(session_id: str, message: str, persona: str | None = None,
                model: str | None = None):
    """同步生成器：yield 事字典。**由 WS 路由放到线程里跑**。"""
    from core import chat_history

    # ⚠️ `core.chat_agent` 的 import 链会走到 `config.settings`，后者在"没有任何厂商
    #    Key"时**导入即抛**。异常发生在生成器内部 ⇒ WS 线程只发 error、不发 done
    #   ⇒ 前端输入框永久灰住。所以这里必须自己兜住，保证 error 后面跟 done。
    try:
        from core.chat_agent import PERSONAS, ChatAgent
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[chat] 助手不可用（配置缺失）: %s: %s", type(e).__name__, e)
        yield {"type": "error", "error": {"type": "auth",
                                          "message": f"助手不可用：{type(e).__name__}: {e}"}}
        yield {"type": "done", "usage": {}, "tools": []}
        return

    message = (message or "").strip()
    if not message:
        yield {"type": "error", "error": {"type": "invalid", "message": "消息为空"}}
        yield {"type": "done", "usage": {}, "tools": []}
        return

    if persona and persona not in PERSONAS:
        # 不做"猜一个最接近的视角"——错的视角比默认视角更坏
        yield {"type": "error", "error": {"type": "invalid",
                                          "message": f"未知视角：{persona}"}}
        yield {"type": "done", "usage": {}, "tools": []}
        return

    sid = chat_history.ensure_session(session_id, persona=persona, model=model,
                                      source="web") if session_id else \
        chat_history.start_session(persona=persona, model=model, source="web")
    # ⚠️ 历史必须在**写入本轮之前**读：先 save_turn 再 load_messages 会让
    #    当前提问出现两次（一次在 history 末尾、一次作为 user_msg）。
    history = chat_history.load_messages(sid, limit=HISTORY_LIMIT)
    yield {"type": "session", "session_id": sid}

    agent = ChatAgent(model=model, persona=persona or "综合")
    answer, errored = None, None
    try:
        for ev in agent.reply(message, history=history):
            kind = ev[0]
            if kind == "tool_start":
                yield {"type": "tool_start", "tool": ev[1], "args": ev[2]}
            elif kind == "tool_end":
                yield {"type": "tool_end", "tool": ev[1], "summary": ev[2]}
            elif kind == "chart":
                yield {"type": "chart", "chart": ev[1]}
            elif kind == "answer":
                answer = ev[1]
                yield {"type": "answer", "text": answer}
            elif kind == "error":
                errored = str(ev[1])
                yield {"type": "error",
                       "error": {"type": "upstream", "message": errored}}
    except Exception as e:                     # noqa: BLE001
        # agent.reply 自己会兜异常；这里兜的是"导入失败/构造失败"这类更早的错
        _logger.exception("[chat] 处理失败")
        errored = f"{type(e).__name__}: {e}"
        yield {"type": "error", "error": {"type": "upstream", "message": errored}}

    if answer:
        chat_history.save_turn(sid, message, answer, usage=agent.usage_total,
                               tools_used=agent.tools_used, persona=persona,
                               model=agent.model, source="web")
    elif errored:
        # 失败也留痕：只写提问、回答写失败原因 —— 比"这条对话消失了"可排查
        chat_history.save_turn(sid, message, f"（本轮失败：{errored}）",
                               usage=agent.usage_total, persona=persona,
                               model=agent.model, source="web")

    yield {"type": "done", "session_id": sid, "usage": dict(agent.usage_total),
           "tools": list(agent.tools_used)}
