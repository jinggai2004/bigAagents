"""问财（pywencai）查询安全封装。

补回缺失模块说明：
    本模块原被 `data/news_announcement.py:12` 以
    `from utils.pywencai_helper import safe_get` 引用，但 `utils/` 从未随仓库提交，
    导致该模块 ImportError、公告降级链完全不可用；而调用处的
    `except Exception: pass` 又把异常静默吞掉，所以这个缺陷此前不可见。

    这里重新提供 safe_get：pywencai 本身没有超时参数，且依赖 Node.js 环境，
    一旦卡住会把整条选股/分析流程挂死。实现用 daemon 线程 + join(timeout)
    做硬超时，并支持有限重试。
"""

import logging
import threading
import time

_logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 20.0   # 单次查询硬超时（秒）
DEFAULT_RETRIES = 2      # loop=True 时的最大尝试次数
RETRY_INTERVAL = 1.0     # 重试间隔（秒）


def _timed_get(query: str, timeout: float, **kwargs):
    """在 daemon 线程中执行 pywencai.get，超时返回 None。"""
    box = {}

    def _run():
        try:
            import pywencai
            box["df"] = pywencai.get(query=query, **kwargs)
        except Exception as e:  # noqa: BLE001 - 需区分"失败"与"超时"两种返回
            box["error"] = f"{type(e).__name__}: {e}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout)

    if "error" in box:
        _logger.warning("[问财] 查询失败: %s", box["error"])
        return None
    if "df" not in box:
        _logger.warning("[问财] 查询超时(>%ss): %s", timeout, str(query)[:40])
        return None
    return box["df"]


def safe_get(query: str, loop: bool = False, retries: int = DEFAULT_RETRIES,
             timeout: float = DEFAULT_TIMEOUT, **kwargs):
    """带硬超时与重试的问财查询。

    Args:
        query: 自然语言问句
        loop: True 时按 retries 重试（兼容旧调用 `safe_get(query=..., loop=True)`）；
              False 时只尝试一次
        retries: 最大尝试次数（loop=True 时生效）
        timeout: 单次调用的硬超时秒数
        **kwargs: 透传给 pywencai.get（如 retry=2）

    Returns:
        DataFrame / dict / None。全部尝试失败或超时返回 None，由调用方处理空结果。
    """
    attempts = max(1, int(retries)) if loop else 1
    for i in range(attempts):
        result = _timed_get(query, timeout, **kwargs)
        if result is not None:
            return result
        if i < attempts - 1:
            time.sleep(RETRY_INTERVAL)
    return None
