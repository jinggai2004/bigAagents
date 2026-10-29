"""
模块名: utils/logger.py
用途: 统一日志系统，替代 print()

放在 utils 层的原因：
    data 层（如 data/source_manager.py）也要记日志。日志器若挂在 core 下，
    就会形成 data → core 的反向依赖，破坏 `db ← data ← core ← ui` 的分层。
    日志只依赖标准库，下沉到 utils 后各层都能干净引用。
    core/logger.py 保留为兼容门面，新代码请直接 import 本模块。

功能:
  - 按模块分日志器
  - INFO / DEBUG / WARNING 三级
  - 输出到文件 + 控制台
  - 自动创建日志目录
"""

import logging
import os
from pathlib import Path
from datetime import datetime

# ── 日志目录 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = str(_PROJECT_ROOT / "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ── 全局格式 ──
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-7s] %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    获取模块级日志器。

    Args:
        name: 模块名（如 "core.timing"）
        level: 日志级别 ("DEBUG" / "INFO" / "WARNING")

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ⚠️ 必须关掉向 root 传播，否则日志会**打两遍**。
    # 原因：本函数已给该 logger 挂了自己的 console + file handler；而
    # `scripts/run_scheduler.py` 启动时会调 `logging.basicConfig(...)` 给 root 也挂一套。
    # 只要 propagate 保持默认 True，每条日志就会「自己的 handler 打一次 + root 的 handler 再打一次」
    # （实测复现：同一条 INFO 在控制台出现两次，日志文件里也重复）。
    # 注意：下面 `if not logger.handlers` 那道判重**只管自己**，挡不住向 root 传播——
    # 这正是原实现看起来"已经判过重"却仍然重复的原因（2026-10-14 实测）。
    logger.propagate = False

    # 避免重复添加 handler
    if not logger.handlers:
        # ── 控制台输出 ──
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(_FORMATTER)
        logger.addHandler(console_handler)

        # ── 文件输出（按模块分文件） ──
        safe_name = name.replace(".", "_")
        log_file = os.path.join(LOG_DIR, f"{safe_name}_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_FORMATTER)
        logger.addHandler(file_handler)

    return logger


def get_console_logger(name: str) -> logging.Logger:
    """仅控制台输出的日志器（用于脚本/CLI，不创建日志文件）"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 添加 console handler (不添加 file handler)，不判重以免多次调用重复添加
    # L2 修复：此前调 get_logger 会附带文件 handler，现改为仅 StreamHandler
    # ⚠️ 同样必须关掉传播（理由见 get_logger）：本函数只挂 console handler，
    # 若 root 已被 basicConfig 配上 handler，不关传播就会每条打两遍。
    logger.propagate = False
    # 判重：避免重复添加
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(_FORMATTER)
        logger.addHandler(console_handler)
    return logger
