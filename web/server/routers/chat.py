# -*- coding: utf-8 -*-
"""助手：会话 CRUD + WebSocket 流式（P4）。

⚠️ 线程化不是优化，是正确性：`ChatAgent.reply()` 是同步生成器、内部是阻塞 HTTP。
   在协程里直接迭代会让**整个服务**停摆（其它页面的轮询一起卡住），
   所以生成器跑在线程里，事件经 `asyncio.Queue` 回到协程。

⚠️ 断连处理：前端关页/切页时 `send_json` 会抛 `WebSocketDisconnect` 或
   `RuntimeError`。**必须接住**——否则线程还在跑、日志刷栈。
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from web.server import chat, envelope

_logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.get("/api/chat/sessions")
def api_sessions(limit: int = chat.SESSION_LIST_LIMIT) -> dict:
    return chat.sessions(limit=max(1, min(200, limit)))


@router.get("/api/chat/personas")
def api_personas() -> dict:
    return chat.personas()


@router.get("/api/chat/sessions/{session_id}")
def api_session(session_id: str, limit: int = 200) -> dict:
    return chat.session(session_id, limit=max(1, min(500, limit)))


@router.post("/api/chat/sessions")
def api_new_session(persona: str | None = None, model: str | None = None) -> dict:
    return chat.create(persona=persona, model=model)


@router.delete("/api/chat/sessions/{session_id}")
def api_delete_session(session_id: str) -> dict:
    return chat.remove(session_id)


@router.websocket("/api/chat/stream")
async def ws_stream(ws: WebSocket) -> None:
    """一轮对话的流式通道。

    客户端：连上来 → 发**一条** JSON `{message, session_id?, persona?, model?}`
    → 服务端逐条推事件 → 最后 `{"type":"done"}`。

    为什么用 WS 而不是 SSE：事件里有**双向**需求（前端中途要能发 cancel），
    以及工具 chip 需要"开始/结束"两段状态，WS 的 send 语义更直接。
    """
    await ws.accept()
    try:
        raw = await ws.receive_text()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("payload 必须是对象")
    except (WebSocketDisconnect, json.JSONDecodeError, ValueError) as e:
        _logger.info("[chat] 首帧非法，关闭连接: %s", type(e).__name__)
        try:
            await ws.send_json({"type": "error",
                                "error": {"type": "invalid", "message": str(e)}})
            await ws.close(code=1003)
        except Exception:                      # noqa: BLE001 - 对端已走
            pass
        return

    qp = ws.query_params
    session_id = payload.get("session_id") or qp.get("session") or ""
    message = payload.get("message") or ""
    persona = payload.get("persona") or qp.get("persona") or None
    model = payload.get("model") or qp.get("model") or None

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _END = object()

    def _worker() -> None:
        try:
            for ev in chat.turn_events(session_id, message, persona=persona,
                                       model=model):
                loop.call_soon_threadsafe(queue.put_nowait, ev)
        except Exception as e:                 # noqa: BLE001
            _logger.exception("[chat] 生成器异常")
            loop.call_soon_threadsafe(queue.put_nowait, {
                "type": "error",
                "error": {"type": "upstream", "message": f"{type(e).__name__}: {e}"}})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _END)

    threading.Thread(target=_worker, daemon=True, name="chat-turn").start()

    try:
        while True:
            ev = await queue.get()
            if ev is _END:
                break
            await ws.send_json(ev)
    except (WebSocketDisconnect, RuntimeError) as e:
        # 前端关了页面：线程会自己跑完（结果仍然落库），这里只记一行
        _logger.info("[chat] 客户端断开: %s", type(e).__name__)
        return

    try:
        await ws.close()
    except Exception:                          # noqa: BLE001
        pass
