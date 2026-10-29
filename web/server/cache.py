# -*- coding: utf-8 -*-
"""进程内 TTL 缓存。

键必须带"数据身份"——文件类用 `(mtime, size)`，比"页面+账户"更硬：
文件一改，键就变，不会把上一版当这一版（项目踩过"跨日把昨天当今天"）。

两种读法：
- `cached(...)`：过期就**同步重取**。要求"必须是新的"的端点用它（看盘页的实时价）。
- `swr(...)`：过期但未超 `max_stale` ⇒ **立即返回旧值 + 后台刷新**。首屏用它，
  代价是可能拿到最多 `max_stale` 秒前的数据 —— 所以调用方必须能表达"这份数据多旧"
  （watch 的 `quote_time`、`cache.age_of()`）。
"""
from __future__ import annotations

import logging
import threading
import time

_logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_STORE: dict = {}          # key -> (expires_monotonic, created_monotonic, value)
_REFRESHING: set = set()   # 正在后台刷新的键（防同一个键并发刷）
_MAX_ENTRIES = 128


def cached(namespace: str, identity, ttl: float, loader):
    """命中即返回；未命中则**锁外**加载（慢 IO 不阻塞其它请求）。"""
    key = (namespace, identity)
    now = time.monotonic()
    with _LOCK:
        hit = _STORE.get(key)
    if hit and hit[0] > now:
        return hit[2]
    return _put(key, ttl, loader())


def swr(namespace: str, identity, ttl: float, loader, max_stale: float = 0.0):
    """过期即**先给旧值**，后台刷新。

    Args:
        max_stale: 旧值最多可容忍多少秒。`0`（默认）等同 `cached` —— 不装成 SWR。
            超过它就必须同步重取：一份 5 分钟前的"实时价"不是降级，是错数据。
        ttl: 同一个 `(namespace, identity)` 的 ttl 要保持一致——后台刷新沿用它，
            若同一键一会儿传 1s 一会儿传 60s，会出现"刚刷完又过期"的反复抖动。
    """
    key = (namespace, identity)
    now = time.monotonic()
    with _LOCK:
        hit = _STORE.get(key)
        if hit and hit[0] > now:
            return hit[2]
        if hit and max_stale > 0 and (now - hit[1]) <= max_stale:
            if key not in _REFRESHING:
                _REFRESHING.add(key)
                threading.Thread(target=_refresh, args=(key, ttl, loader),
                                 name=f"swr-{namespace}", daemon=True).start()
            return hit[2]
    # 无缓存 / 旧到不能再用：同步取，别让首屏拿到过期数据
    return _put(key, ttl, loader())


def _refresh(key, ttl: float, loader) -> None:
    try:
        _put(key, ttl, loader())
    except Exception as e:                             # noqa: BLE001
        # 后台刷新失败**不能吞成"数据没变"**：旧值继续留着，但必须留痕
        _logger.warning("[cache] 后台刷新失败 %s: %s: %s", key[0], type(e).__name__, e)
    finally:
        with _LOCK:
            _REFRESHING.discard(key)


def _put(key, ttl: float, value):
    expire_at = time.monotonic() + ttl
    with _LOCK:
        if len(_STORE) >= _MAX_ENTRIES:
            # 简单淘汰：清掉已过期的；仍然过多就整体清空（进程内缓存，代价可接受）
            alive = {k: v for k, v in _STORE.items() if v[0] > time.monotonic()}
            _STORE.clear()
            _STORE.update(alive)
        _STORE[key] = (expire_at, time.monotonic(), value)
    return value


def age_of(namespace: str, identity):
    """该键当前值的年龄（秒）；没有条目返回 `None`。

    用途：SWR 拿到的旧值必须能自报年龄，否则前端会把 40 秒前的价格当实时价。
    """
    key = (namespace, identity)
    with _LOCK:
        hit = _STORE.get(key)
    return None if not hit else round(time.monotonic() - hit[1], 1)


def invalidate(namespace: str, identity=None) -> int:
    """按作用域失效，返回清掉的键数。

    ⚠️ **写操作之后必须调它**：`watch_list` 的 ttl 是 15s，改完自选再读会命中
    上一次的缓存 ⇒ 页面显示的还是旧清单，而价格全对、接口也 `ok:true`，
    看起来像"点了保存没反应"。这类"写成功但读不回来"最难自查。
    identity 为 None 时清掉该 namespace 下**全部**键（自选只有两个 owner、缓存极轻）。
    """
    with _LOCK:
        keys = [k for k in _STORE
                if k[0] == namespace and (identity is None or k[1] == identity)]
        for k in keys:
            _STORE.pop(k, None)
        _REFRESHING.difference_update(keys)
    return len(keys)


def clear() -> None:
    """测试夹具用：`cache_data` 是进程级全局，必须能在用例间清干净。"""
    with _LOCK:
        _STORE.clear()
        _REFRESHING.clear()
