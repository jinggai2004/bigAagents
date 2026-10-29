# -*- coding: utf-8 -*-
"""首屏预热：服务起来后**在后台**把最贵的那几个数先取一遍。

为什么需要：`/api/dashboard` 冷缓存要 2~3s，其中 ≈1.2s 是实时报价的首次外呼、
≈0.8s 是 pandas 首次 import（`data.index_bars`）。这些是**进程级一次性成本**，
但用户第一次打开页面正好撞上 —— 于是"服务启动后第一眼"最慢，之后反而快。

⚠️ **只由服务进程自己调**（`scripts/serve_web.py`、exe 启动器设 `BIGA_WARMUP=1`）。
   测试绝不预热：`tests/test_web_api.py::TestNoOutbound` 断言端点零外呼，
   一个"导入 app 就联网"的预热线程会让那条断言随机变红（真实外呼 + 时序不确定）。
"""
from __future__ import annotations

import logging
import os
import threading

_logger = logging.getLogger(__name__)

WARMUP_ENV = "BIGA_WARMUP"


def enabled() -> bool:
    return str(os.environ.get(WARMUP_ENV, "0")).strip().lower() in ("1", "true", "yes", "on")


def start(delay: float = 0.8) -> bool:
    """起一个 daemon 线程做预热。返回是否真的起了。"""
    if not enabled():
        _logger.debug("[warmup] 未启用（%s 未置 1）", WARMUP_ENV)
        return False
    t = threading.Thread(target=_run, args=(delay,), name="warmup", daemon=True)
    t.start()
    return True


def _run(delay: float) -> None:
    import time
    time.sleep(delay)                       # 让 uvicorn 先把端口 bind 完，别抢启动
    for name, fn in (
        # 顺序有讲究：watch 里有外部网络（最慢），先发出去；picks 读文件顺手
        ("watch.indices", _warm_watch),
        ("picks.latest", _warm_picks),
        ("review.pending", _warm_pending),
        # 2026-10-26：联赛榜单冷启 9s、涨跌家数首次要读清单 —— 都是首屏必需项
        ("desk.league", _warm_league),
        ("breadth", _warm_breadth),
    ):
        t0 = time.monotonic()
        try:
            fn()
            _logger.info("[warmup] %s %.2fs", name, time.monotonic() - t0)
        except Exception as e:                             # noqa: BLE001
            # 预热失败**不影响服务**：该端点自己会走正常降级路径
            _logger.warning("[warmup] %s 失败: %s: %s", name, type(e).__name__, e)


def _warm_watch() -> None:
    from web.server import watch
    watch.indices()


def _warm_picks() -> None:
    from web.server import picks
    picks.load_latest()


def _warm_pending() -> None:
    from web.server import review
    review.pending(limit=6)


def _warm_league() -> None:
    """联赛榜单：实测**冷启 9 秒**（6 个账户逐个算净值 + 读面板）。

    ⚠️ 首页与交易台都要它，而 `/api/dashboard` 是"所有分区都到齐才返回"结构
       ⇒ 这一项冷启能把首屏拖到 12 秒以上（首屏看起来像"坏了"）。
       放到服务启动的后台线程里先跑一遍，用户第一眼就不会撞上。
    """
    from web.server import desk
    desk.league()


def _warm_breadth() -> None:
    """涨跌家数：冷启含 A 股清单（磁盘命中则几十毫秒，缺失时约 1.5s 外呼）。"""
    from core import breadth
    breadth.market_breadth()
