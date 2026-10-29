# -*- coding: utf-8 -*-
"""`/api/dashboard` 的 health 子区 —— 调度器心跳。

判活纪律（项目铁律）：**看 `started_at` + PID**，不是文件 mtime；
心跳陈旧但进程还在 ⇒ 只告警不杀（这里只展示，不做任何动作）。
"""
from __future__ import annotations

import logging
import time

from core.scheduler_health import (HEARTBEAT_STALE_SECONDS, _process_alive,
                                   read_heartbeat)
from web.server import cache, envelope, settings

_logger = logging.getLogger(__name__)


def load_health(now_ts: float | None = None) -> dict:
    hb = read_heartbeat()
    if not hb:
        # 没有心跳 = 调度器没在跑。这是**数据**不是服务故障 ⇒ ok=True + running=False
        _logger.debug("[health] 无心跳文件: %s", settings.heartbeat_path())
        return envelope.ok({"running": False, "pid": None, "started_at": None,
                            "heartbeat_at": None, "age_seconds": None,
                            "jobs": 0, "stale": None})

    pid = hb.get("pid")
    epoch = hb.get("epoch")
    now = time.time() if now_ts is None else now_ts
    age = None
    if isinstance(epoch, (int, float)):
        age = max(0.0, now - float(epoch))

    identity = (pid, int(epoch) if isinstance(epoch, (int, float)) else 0)
    if now_ts is not None:                     # 测试注入 ⇒ 绕过缓存
        return _compose(hb, pid, age, now)
    return cache.cached("health", identity, settings.TTL_HEALTH,
                        lambda: _compose(hb, pid, age, now))


def _compose(hb: dict, pid, age, now: float) -> dict:
    # 心跳每 60s 刷新（scripts/run_scheduler.py._start_heartbeat）⇒ PID 检查才是权威：
    # 不用 age 当门（笔记本睡眠 >1h 后唤醒时，进程活着但心跳很旧，按 age 判会误报"未运行"）。
    alive = bool(_process_alive(pid)) if pid else False
    stale = None if age is None else age > HEARTBEAT_STALE_SECONDS
    data = {
        "running": alive,
        "pid": pid,
        "started_at": hb.get("started_at"),
        "heartbeat_at": hb.get("ts"),
        "age_seconds": None if age is None else round(age, 1),
        "jobs": hb.get("jobs") or len(hb.get("job_ids") or []),
        "stale": stale,
    }
    if stale is True and alive:
        # 铁律：心跳陈旧但进程在 ⇒ 只告警不杀；这里如实两条都给出
        _logger.warning("[health] 心跳 %ss 未更新，但 PID %s 仍在（只告警）",
                        int(age or 0), pid)
    return envelope.ok(data)
