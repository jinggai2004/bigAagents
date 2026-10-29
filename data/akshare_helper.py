"""
Akshare 请求补丁模块

解决 Akshare 因缺少请求头（User-Agent）导致东方财富等服务器
关闭连接（RemoteDisconnected）的问题。

工作原理：
1. 在 import akshare 之前调用 patch_requests()，全局注入默认请求头
2. 同时修补 akshare 自带的请求重试函数，确保 timeout 参数被传递
3. 提供 retry_on_failure 装饰器，方便包装有问题的 Akshare 接口
"""

import functools
import os
import threading
import time
import random
import logging

_logger = logging.getLogger(__name__)

# 默认请求头 — 模拟现代 Chrome 浏览器
DEFAULT_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': (
        'text/html,application/xhtml+xml,application/xml;q=0.9,'
        'image/avif,image/webp,image/apng,*/*;q=0.8'
    ),
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://quote.eastmoney.com/',
    'Cache-Control': 'no-cache',
}

# 多个 User-Agent 轮换，降低被屏蔽概率
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
]

_patched = False

# ============================================================
# v2 / M0 0.4：限频器 + 429 识别 + 失败负缓存
#
# 为什么需要：免费财经接口对请求频率敏感。原实现对同一源没有任何节流，
# 逐股循环（选股一次要对几十只股票分别拉资金流/K线）会持续冲击同一域名；
# 而 P0 的"盘中每分钟盯盘"会把频率再提高一个数量级。限流与退避必须在数据层
# 统一实现，而不是指望每个调用点自觉 sleep。
# ============================================================

# 各源最小请求间隔（秒）；未识别到的域名用 default
_MIN_INTERVAL = {
    "eastmoney": 0.5,
    "ths": 0.5,
    "sina": 0.25,
    "tencent": 0.25,
    "default": 0.25,
}

# 命中 429 后的退避基数（秒）：服务端明确要求降速时，短间隔重试只会加重封禁
RATE_LIMIT_BASE_DELAY = 5.0

# 最小间隔的整体缩放系数。生产默认 1.0；离线/测试场景可设
# AKSHARE_MIN_INTERVAL_SCALE=0 关闭节流（否则联网用例会在 sleep 上多花数分钟）
MIN_INTERVAL_SCALE = float(os.getenv("AKSHARE_MIN_INTERVAL_SCALE", "1.0"))

_last_call: dict = {}
_throttle_lock = threading.Lock()


def source_key(url: str) -> str:
    """把 URL 归类到数据源（用于限频与退避）"""
    u = (url or "").lower()
    if "eastmoney" in u or "push2" in u:
        return "eastmoney"
    if "10jqka" in u or "ths" in u:
        return "ths"
    if "sinajs" in u or "sina.com" in u:
        return "sina"
    if "gtimg" in u or "qq.com" in u:
        return "tencent"
    return "default"


def throttle(url: str):
    """按源限频：与同源上次请求间隔不足时 sleep。线程安全"""
    key = source_key(url)
    interval = _MIN_INTERVAL.get(key, _MIN_INTERVAL["default"]) * MIN_INTERVAL_SCALE
    if interval <= 0:
        return
    with _throttle_lock:
        now = time.time()
        wait = interval - (now - _last_call.get(key, 0.0))
        if wait > 0:
            time.sleep(wait)
            now = time.time()
        _last_call[key] = now


def is_rate_limited(exc_or_resp) -> bool:
    """识别限流：HTTP 429 或响应体/异常里的限流文案"""
    if getattr(exc_or_resp, "status_code", None) == 429:
        return True
    text = str(exc_or_resp)
    return any(marker in text for marker in
               ("429", "Too Many Requests", "too many requests",
                "访问频率", "请求过于频繁", "限流", "rate limit"))


class NegativeCache:
    """失败负缓存：某个键失败后，TTL 内直接跳过，避免反复打已失效的源/标的。

    与"熔断器"的区别：熔断是按数据源整体冷却，负缓存是按具体请求键（例如
    "000001 的资金流"）——某只票没有数据或持续报错时，不该让整条降级链反复重试它。
    """

    def __init__(self, ttl: float = 60.0):
        self.ttl = float(ttl)
        self._dead: dict = {}
        self._lock = threading.Lock()

    def is_dead(self, key: str) -> bool:
        with self._lock:
            until = self._dead.get(key)
            if until is None:
                return False
            if time.time() >= until:
                del self._dead[key]
                return False
            return True

    def mark_failure(self, key: str):
        with self._lock:
            self._dead[key] = time.time() + self.ttl

    def clear(self, key: str = None):
        with self._lock:
            if key is None:
                self._dead.clear()
            else:
                self._dead.pop(key, None)

    def size(self) -> int:
        with self._lock:
            return len(self._dead)


def patch_requests():
    """
    全局修补 requests.get / requests.post / requests.request，
    注入默认请求头和超时，避免 Akshare 请求被东方财富服务器拒绝。
    
    在 import akshare 之前调用一次即可。
    """
    global _patched
    if _patched:
        return
    _patched = True

    import requests as req_lib

    # --- 保存原始函数 ---
    _original_get = req_lib.get
    _original_post = req_lib.post
    _original_request = req_lib.Session.request

    # --- 1. 修补 Session.request（所有 Session 方法底层都走这） ---
    @functools.wraps(_original_request)
    def _patched_session_request(self, method, url, **kwargs):
        # 合并默认请求头（用户传入的 headers 优先）
        headers = dict(DEFAULT_HEADERS)
        # 随机轮换 User-Agent
        headers['User-Agent'] = random.choice(USER_AGENTS)
        user_headers = kwargs.pop('headers', None)
        if user_headers:
            headers.update(user_headers)
        kwargs['headers'] = headers

        # 默认超时 30 秒
        if 'timeout' not in kwargs or kwargs['timeout'] is None:
            kwargs['timeout'] = 30

        throttle(url)   # v2/M0 0.4：同源最小间隔，避免逐股循环压垮免费接口
        return _original_request(self, method, url, **kwargs)

    req_lib.Session.request = _patched_session_request

    # --- 2. 修补 requests.get （兼容直接调用 requests.get 的代码） ---
    @functools.wraps(_original_get)
    def _patched_get(url, **kwargs):
        headers = dict(DEFAULT_HEADERS)
        headers['User-Agent'] = random.choice(USER_AGENTS)
        user_headers = kwargs.pop('headers', None)
        if user_headers:
            headers.update(user_headers)
        kwargs['headers'] = headers

        if 'timeout' not in kwargs or kwargs['timeout'] is None:
            kwargs['timeout'] = 30

        throttle(url)
        return _original_get(url, **kwargs)

    req_lib.get = _patched_get

    # --- 3. 修补 requests.post ---
    @functools.wraps(_original_post)
    def _patched_post(url, **kwargs):
        headers = dict(DEFAULT_HEADERS)
        headers['User-Agent'] = random.choice(USER_AGENTS)
        user_headers = kwargs.pop('headers', None)
        if user_headers:
            headers.update(user_headers)
        kwargs['headers'] = headers

        if 'timeout' not in kwargs or kwargs['timeout'] is None:
            kwargs['timeout'] = 30

        throttle(url)
        return _original_post(url, **kwargs)

    req_lib.post = _patched_post

    # --- 4. 修补 akshare 内部的 request 模块（如果有的话） ---
    try:
        import akshare as _ak
        if hasattr(_ak, 'request') and hasattr(_ak.request, 'make_request_with_retry_json'):
            _orig_retry = _ak.request.make_request_with_retry_json

            @functools.wraps(_orig_retry)
            def _patched_retry_json(url, params=None, headers=None, proxies=None, max_retries=3, retry_delay=1):
                if headers is None:
                    headers = {}
                final_headers = dict(DEFAULT_HEADERS)
                final_headers['User-Agent'] = random.choice(USER_AGENTS)
                final_headers.update(headers)
                return _orig_retry(url, params=params, headers=final_headers,
                                   proxies=proxies, max_retries=max_retries, retry_delay=retry_delay)

            _ak.request.make_request_with_retry_json = _patched_retry_json
    except (ImportError, AttributeError):
        pass

    _logger.info("[AkshareHelper] 请求补丁已应用 — 默认请求头 + 超时 30s")


def retry_on_failure(max_retries=3, base_delay=1.0, backoff=2.0, exceptions=(Exception,)):
    """
    重试装饰器 — 用于包装不稳定的数据获取函数。

    v2/M0 0.4：识别限流（HTTP 429 / "访问频率" 等文案）并采用更长的指数退避——
    服务端明确要求降速时，按普通失败的 1s/2s 间隔重试只会加重封禁。

    用法:
        @retry_on_failure(max_retries=3)
        def fetch_something():
            return ak.stock_xxx(...)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (backoff ** attempt) + random.uniform(0, 0.5)
                        if is_rate_limited(e):
                            delay = max(delay, RATE_LIMIT_BASE_DELAY * (backoff ** attempt))
                            _logger.info(f"[Retry] {func.__name__} 命中限流(429)，"
                                  f"{delay:.1f}s 后重试（第{attempt+2}次）...")
                        else:
                            _logger.warning(f"[Retry] {func.__name__} 失败 (第{attempt+1}次): {e}, "
                                  f"{delay:.1f}s 后重试...")
                        time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator

