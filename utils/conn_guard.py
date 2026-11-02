# -*- coding: utf-8 -*-
"""连接层失败判定与通用熔断器（多个抓取源共用）。

为什么要有这一层
────────────────
抓取源对"失败"的处理必须区分**两类**（与 `data/fund_flow._is_connection_level_error`
同口径，2026-10-30 起抽到这里共用）：

- **连接层**（主机不通 / 被代理拒绝 / 连接被重置）⇒ 重试同一主机没有意义，
  应该**立即熔断**并把机会让给下一个源。2026-10-30 实测事故：
  WinINET 代理指向未启动的 Clash（127.0.0.1:7890），K 线路径对死代理
  3 源 × 2 重试白烧 70 秒；而资金流路径 0.5 秒就熔断降级成功 ——
  同一台死代理、同一份日志，两种处理。
- **内容层**（接口通了但返回空 / 解析失败）⇒ 可能是抖动 ⇒ 保留原有重试。

混为一谈的后果：要么白等（把连接失败当抖动重试），要么误熔断
（把一次空响应当主机挂掉，冷却期整段放弃该源）。

已有两份独立实现（`data/fund_flow` / `data/eastmoney_source`），这里是第三处
使用方（`data/tencent_source`，K 线主源）抽出的**共用实现**；旧的两份保持不动，
由各自的测试守护。判定语义必须一致，改这里时同步检查那两份。
"""
from __future__ import annotations

import logging
import threading
import time

import requests

_logger = logging.getLogger(__name__)

# "连不上"的典型文案。既匹配异常类型也匹配 str(exc)：
# akshare 等第三方库会把底层异常包成 ValueError/自定义异常，类型判定会漏。
CONN_ERROR_MARKERS = (
    "Max retries exceeded", "ProxyError", "RemoteDisconnected",
    "Connection aborted", "Unable to connect", "Connection reset",
    "Name or service not known", "getaddrinfo failed",
    "由于目标计算机积极拒绝",          # Windows WinError 10061 的中文文案
)


def is_connection_level_error(exc: BaseException | None) -> bool:
    """是否为"连不上"类失败（域名不可达 / 被代理拒绝 / 连接被重置）。

    返回 True ⇒ 调用方应**立即熔断/换源**，不要对同一主机再重试。
    """
    if exc is None:
        return False
    if isinstance(exc, (requests.exceptions.ProxyError,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout)):
        return True
    return any(m in str(exc) for m in CONN_ERROR_MARKERS)


class ConnBreaker:
    """线程安全的连接层熔断器。

    语义：`trip()` 后冷却期内 `open()` 为 True，调用方应跳过该源直接换下一个；
    冷却期结束自动恢复。**只对连接层失败打开**——内容层失败由调用方按原逻辑重试。
    """

    def __init__(self, name: str, cooldown: float = 600.0):
        self._name = str(name)
        self._cooldown = float(cooldown)
        self._open_until = 0.0
        self._lock = threading.Lock()

    def remaining(self) -> float:
        """熔断还剩多少秒（0 = 未熔断）。"""
        with self._lock:
            return max(0.0, self._open_until - time.time())

    def open(self) -> bool:
        return self.remaining() > 0

    def trip(self, reason: str) -> None:
        """打开熔断（重复 trip 会刷新冷却期，与 eastmoney_source 口径一致）。"""
        with self._lock:
            self._open_until = time.time() + self._cooldown
        _logger.warning("[%s] 连接层故障 ⇒ 熔断 %.0fs: %s",
                        self._name, self._cooldown, reason)

    def reset(self) -> None:
        """清掉熔断（测试 / 手动重试用；生产不应定期调用）。"""
        with self._lock:
            self._open_until = 0.0
