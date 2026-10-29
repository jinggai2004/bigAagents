# -*- coding: utf-8 -*-
"""AI 助手配置写守护（2026-10-24，P0 整改）——preview → commit → 审计 → restore。

## 为什么必须有这个模块（质量复核 P0-1）

升级初版把"先向用户确认再写"只写进了 SYSTEM_PROMPT/schema 文案，**代码并不校验**：
模型一次 tool_calls 里多个写操作会被循环立即执行，`delete` 是物理 DELETE，
审计只存工具名不存前后值——误删/误改无法追责也无法恢复。

本模块把**所有配置写操作**统一改成两阶段：

    ① 工具 action（如 save/create/toggle）= **preview**：只校验 + 计算前后快照，
      生成一次性 change_id 返回给模型，**不落任何写**；
    ② 用户明确同意后，模型用 action="commit" + change_id 执行真正的写；
    ③ 每次 commit 把 tool/action/前后完整快照/回滚描述写入 `agent_config_changes`；
    ④ action="restore" + 历史 change_id 可按快照回滚。

## 设计要点

- **change_id 即授权凭证**：16 位随机 hex，15 分钟过期、一次性使用、不可枚举——
  知道 change_id = 已被展示过完整 diff 且用户已确认。不额外绑 session：
  本助手是单用户本地工具，凭证熵 + TTL + 单次已足够。
- **op/revert 用可序列化描述符**（不是闭包）：审计行里的 revert_json 能在任何
  进程里重新执行；`_execute` 是唯一执行入口（白名单 kind）。
- **in-memory pending**：preview 与 commit 在同一对话进程内完成即可；
  进程重启 → pending 丢失 → commit 返回"已过期，请重新 preview"（fail-closed）。
- 审计失败**不阻断**业务写（写已发生），但记 WARNING——审计是观测层不是事务层。
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime

_logger = logging.getLogger(__name__)

TTL_SEC = 15 * 60          # preview → commit 的有效期
_CLEANUP_EVERY = 64        # 每 N 次 begin 清一次过期 pending（惰性，不起后台线程）

_PENDING: dict[str, dict] = {}
_begin_count = 0


class WriteGuardError(Exception):
    """preview/commit 的明确业务错误（会转成工具的 error dict，不抛栈）。"""


# ================================================================
# op / revert 描述符执行（唯一入口，kind 白名单）
# ================================================================

def _execute(op: dict) -> dict:
    """执行一个写描述符，返回 {"message": ...}。"""
    kind = op.get("kind")
    if kind == "template_save":
        from core import push_templates as T
        ok = T.save_template(str(op["key"]), str(op["body"]))
        if not ok:
            raise WriteGuardError(f"模板保存失败（key={op['key']}）")
        return {"message": f"模板 {op['key']} 已保存（即时生效）"}
    if kind == "template_reset":
        from core import push_templates as T
        T.reset_template(str(op["key"]))
        return {"message": f"模板 {op['key']} 已恢复默认（即时生效）"}
    if kind == "job_upsert":
        from db import store
        store.upsert_scheduled_job(dict(op["job"]))
        return {"message": f"任务 {op['job'].get('job_id')} 已保存（调度器约60秒同步）"}
    if kind == "job_toggle":
        from db import store
        ok = store.set_scheduled_job_enabled(str(op["job_id"]), bool(op["enabled"]))
        if not ok:
            raise WriteGuardError(f"任务不存在: {op['job_id']}")
        return {"message": f"任务 {op['job_id']} 已{'启用' if op['enabled'] else '停用'}"}
    if kind == "job_delete":
        from db import store
        ok = store.delete_scheduled_job(str(op["job_id"]))
        if not ok:
            raise WriteGuardError(f"任务不存在: {op['job_id']}")
        return {"message": f"任务 {op['job_id']} 已删除"}
    if kind == "watchlist_save":
        from data import watchlist as wl
        wl.save_watchlist(list(op["stocks"]))
        return {"message": f"自选股已更新（共 {len(op['stocks'])} 只）"}
    raise WriteGuardError(f"未知写操作类型: {kind!r}")


# ================================================================
# pending 管理
# ================================================================

def _sweep():
    now = time.time()
    dead = [k for k, v in _PENDING.items() if v["expires_at"] < now]
    for k in dead:
        _PENDING.pop(k, None)


def begin(tool: str, action: str, summary: str, old, new, op: dict,
          revert_op: dict | None = None, ttl_sec: int = TTL_SEC) -> dict:
    """登记一次待确认的写。返回给工具的 preview dict（含 change_id）。"""
    global _begin_count
    _begin_count += 1
    if _begin_count % _CLEANUP_EVERY == 0:
        _sweep()

    cid = uuid.uuid4().hex[:16]
    _PENDING[cid] = {
        "tool": tool, "action": action, "summary": summary,
        "old": old, "new": new,
        "op": op, "revert_op": revert_op,
        "created_at": time.time(), "expires_at": time.time() + ttl_sec,
    }
    return {
        "待确认": True,
        "change_id": cid,
        "有效期分钟": round(ttl_sec / 60),
        "摘要": summary,
        "改动前": old,
        "改动后": new,
        "下一步": ("向用户完整复述改动前后差异；**仅在用户明确同意后**，"
                  "用 action='commit' 且 change_id=本 change_id 执行；"
                  "用户未同意前绝不要 commit"),
    }


def _take(change_id: str) -> dict:
    p = _PENDING.pop(str(change_id or ""), None)      # pop = 一次性
    if p is None:
        raise WriteGuardError("change_id 不存在或已被使用——写操作必须重新 preview 获取新凭证")
    if p["expires_at"] < time.time():
        raise WriteGuardError("change_id 已过期（超过有效期），请重新 preview 后再让用户确认")
    return p


def _audit(p: dict, change_id: str, ok: bool, message: str) -> int | None:
    try:
        from db import store
        return store.save_agent_change({
            "change_id": change_id, "tool": p["tool"], "action": p["action"],
            "summary": p["summary"], "ok": 1 if ok else 0, "message": message,
            "old_json": json.dumps(p["old"], ensure_ascii=False, default=str),
            "new_json": json.dumps(p["new"], ensure_ascii=False, default=str),
            "revert_json": json.dumps(p.get("revert_op"), ensure_ascii=False, default=str),
        })
    except Exception as e:                              # noqa: BLE001 - 审计失败不阻断已发生的写
        _logger.warning("[写守护] 审计落库失败（写已执行，审计缺失）: %s", e)
        return None


def commit(change_id: str) -> dict:
    """执行一次已确认的写并审计。"""
    try:
        p = _take(change_id)
    except WriteGuardError as e:
        _logger.debug("[写守护] commit 凭证校验失败: %s", e)
        return {"error": str(e)}

    ok, message = True, ""
    try:
        res = _execute(p["op"])
        message = res.get("message", "已执行")
    except WriteGuardError as e:
        _logger.warning("[写守护] %s.%s 被业务校验拒绝: %s", p["tool"], p["action"], e)
        ok, message = False, str(e)
    except Exception as e:                              # noqa: BLE001 - 底层异常统一进 error dict
        ok = False
        message = f"{type(e).__name__}: {e}"
        _logger.exception("[写守护] %s.%s 执行失败", p["tool"], p["action"])

    audit_id = _audit(p, str(change_id), ok, message)
    out = {"success": ok, "message": message, "change_id": change_id}
    if audit_id:
        out["审计ID"] = audit_id
    if ok and p.get("revert_op"):
        out["可回滚"] = (f"如需撤销，用 action='restore' 且 change_id={change_id}")
    return out


def restore(change_id: str) -> dict:
    """按历史审计行的 revert 快照回滚一次写。"""
    from db import store
    row = store.get_agent_change(str(change_id or ""))
    if not row:
        return {"error": f"找不到 change_id={change_id} 的审计记录"}
    raw = row.get("revert_json")
    revert_op = json.loads(raw) if raw and raw != "null" else None
    if not revert_op:
        return {"error": "该改动没有可回滚快照（restore 仅支持带 revert 的写）"}

    try:
        res = _execute(revert_op)
        audit_id = _audit({"tool": f"{row['tool']}", "action": f"restore:{row['action']}",
                           "summary": f"回滚 change_id={change_id}", "old": None, "new": None,
                           "revert_op": None},
                          f"{change_id}-restore", True, res.get("message", "已回滚"))
        out = {"success": True, "message": f"已回滚：{res.get('message', '')}", "原始change_id": change_id}
        if audit_id:
            out["审计ID"] = audit_id
        return out
    except Exception as e:                              # noqa: BLE001
        _logger.exception("[写守护] restore 失败 change_id=%s", change_id)
        return {"error": f"回滚失败: {type(e).__name__}: {e}"}


def pending_count() -> int:
    """测试用：当前未消费的 preview 数。"""
    return len(_PENDING)
