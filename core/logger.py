"""兼容门面：日志实现已下沉到 utils/logger.py

data 层依赖 core 会破坏分层（data/source_manager.py 原 from core.logger import），
故实现移到零依赖的 utils 层。本模块只做 re-export，历史 import 路径继续可用，
**新代码请直接 `from utils.logger import get_logger`**。
"""

from utils.logger import LOG_DIR, get_console_logger, get_logger  # noqa: F401

__all__ = ["LOG_DIR", "get_logger", "get_console_logger"]
