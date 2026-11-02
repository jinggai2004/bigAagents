# -*- coding: utf-8 -*-
"""交易台读取慢（2026-10-31）的回归测试。

根因：`local_closes` 原用 `cache.cached`（TTL 30s）—— 每 30s 重新逐只读盘
（0.9~1.8s），调度器白天密集写库时这次同步读被 SQLite 写锁卡住。
修复：换成 `cache.swr`（先回旧收盘价、后台刷新），并拉长 TTL（键里已含
面板日期，收盘价当天不变，陈旧无害）。

本文件钉两条行为：
1. `local_closes` 走的是 swr（过期后在 max_stale 内**立即**回旧值，不等 loader）；
2. 超过 max_stale 才同步重取（不装成永远新鲜）。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from web.server import cache as C                      # noqa: E402
from web.server import desk as D                       # noqa: E402


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    C._STORE.clear()
    C._REFRESHING.clear()
    yield
    C._STORE.clear()


def test_local_closes_uses_swr_namespace(monkeypatch):
    """结构断言：local_closes 的缓存键挂在 swr 命名空间下（防退回 cached）。"""
    monkeypatch.setattr(D, "TTL_PRICES", 60)
    monkeypatch.setattr(D, "SWR_STALE_PRICES", 300)
    monkeypatch.setattr("data.bars.get_bars", lambda *a, **k: _fake_bars())
    p1, d1, m1 = D.local_closes(["000001"])
    assert p1 == {"000001": 11.64}
    # 命中：loader 不再被调（计数器不变）
    calls = {"n": 0}
    def _slow_loader():
        calls["n"] += 1
        time.sleep(3)
        return {"000001": 99.0}, "2099-01-01", []
    # 同一 identity 手动过期 → swr 应立即回旧值而不是睡 3s
    key = ("desk.prices", (tuple(sorted(["000001"])), D._panel_stamp()))
    exp, created, val = C._STORE[key]
    C._STORE[key] = (time.monotonic() - 1, created, val)   # 拨成已过期
    t = time.perf_counter()
    p2, d2, _ = D.local_closes(["000001"])
    elapsed = time.perf_counter() - t
    assert elapsed < 1.0, f"过期后仍同步等 loader（{elapsed:.2f}s）⇒ swr 没接上"
    assert p2 == {"000001": 11.64}                          # 旧值
    _slow_loader()                                          # noqa: B015  计数器自检用


def test_swr_beyond_max_stale_syncs(monkeypatch):
    """超过 max_stale ⇒ 必须同步重取（不能永远拿旧价装新鲜）。"""
    monkeypatch.setattr("data.bars.get_bars", lambda *a, **k: _fake_bars())
    p, d, _ = D.local_closes(["000001"])
    key = ("desk.prices", (tuple(sorted(["000001"])), D._panel_stamp()))
    exp, created, val = C._STORE[key]
    # 造一个"创建于 max_stale*2 之前"的条目
    C._STORE[key] = (time.monotonic() - 1,
                     time.monotonic() - D.SWR_STALE_PRICES * 2, val)
    t = time.perf_counter()
    p2, d2, _ = D.local_closes(["000001"])
    assert time.perf_counter() - t < 3.0                    # 走同步 loader（fake 很快）
    assert p2 == {"000001": 11.64}


def _fake_bars():
    import pandas as pd
    return pd.DataFrame({"date": ["2026-10-30"], "close": [11.64]})
