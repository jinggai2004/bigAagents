# -*- coding: utf-8 -*-
"""进程启动后的**后台预热**：把重依赖提前导入，消除"首次点某页卡十几秒"。

## 为什么需要（2026-10-15 实测驱动）

Streamlit 是**长驻进程**，但每个页面脚本各自持有自己的 import：
**第一次**渲染某个页面时才会导入它的依赖链。实测冷导入（各自新进程）：

| 模块 | 耗时 |
| --- | --- |
| `core.graph_builder` / `core.agents`（LangGraph + akshare 链） | **12.5 ~ 12.9s** |
| `core.news_industry_review` | **9.28s** |
| `data.news_feed` | 4.39s |
| `import akshare` | 4.02s |
| `import streamlit` | 2.07s |
| `from db import store` | 0.10s |

⇒ 用户体感是"点进去卡十几秒"；之后所有交互都很快（模块已缓存）。
本模块把这条链在**进程启动时**丢到后台线程导入 —— 用户停留在首页的那几秒就把它热好了。

## 三条纪律（每一条都对应一个真实的坑）

1. **只预热、不改语义**：`__import__` 是幂等且无副作用的（模块只初始化一次）。
   ⚠️ 因此 `core/` 下**不允许**有 `import streamlit` —— 否则这段会在**没有 ScriptRunContext
   的线程里**执行 Streamlit API。已核对：`core/` 目前零 streamlit 依赖。
2. **失败必须是"忽略"而非"崩"**：预热失败只记日志，绝不阻断 UI（它是加速手段，不是依赖）。
3. ⚠️ `app.py` 每次交互都会**重跑脚本**，但**模块只导入一次** ⇒ 用**模块级 flag + 锁**
   保证只起一个线程（写在页面脚本里的全局变量每次 rerun 都会重置，起不到这个作用）。
"""
from __future__ import annotations

import logging
import os
import threading

_logger = logging.getLogger(__name__)

# 预热清单：按"贵 → 便宜"排序，让最贵的先开始。
# 数字为 2026-10-15 实测（各自新进程首导入），用于以后复核是否还有更贵的漏网之鱼：
#   core.graph_builder 12.5s / core.agents 12.9s / core.news_industry_review 9.3s /
#   data.fund_flow 6.2s / core.chat_agent 4.7s / data.news_feed 4.4s /
#   import akshare 4.0s / core.analysis_history 1.4s / plotly ~0.2s（已随 pandas 摊薄）
PRELOAD_TARGETS: tuple[str, ...] = (
    "core.graph_builder",          # LangGraph + agents 链 ≈12.5s
    "core.agents",
    "core.news_industry_review",   # akshare + pandas + deepseek 链 ≈9.3s
    "data.fund_flow",              # ≈6.2s（pandas/akshare 依赖）
    "core.chat_agent",             # ≈4.7s
    "data.news_feed",
    "core.analysis_history",
    "core.analysis_runner",
    "core.screener",
    "core.ai_daily_round",
    "core.ai_league",
    "llm.deepseek_client",
    "data.realtime",
    "data.watchlist",
    "plotly.graph_objects",
    "scripts.backtest_rule_based",
)

_started = False
_lock = threading.Lock()


def preload_enabled() -> bool:
    """是否预热。

    ⚠️ 测试环境下默认**关闭**：预热会真的去 import LangGraph / akshare，
       既拖慢用例、又引入非确定性 ⇒ 由 `BIGA_UI_OFFLINE` 自动关掉。
       显式设 `BIGA_UI_PRELOAD=1/0` 可强制开关（覆盖上面的默认）。
    """
    forced = (os.getenv("BIGA_UI_PRELOAD") or "").strip()
    if forced in ("0", "false", "False"):
        return False
    if forced in ("1", "true", "True"):
        return True
    return not bool(os.environ.get("BIGA_UI_OFFLINE"))


def ensure_preloaded() -> bool:
    """幂等地启动后台预热线程。Returns: 本次是否**真的**启动了线程。"""
    global _started
    if not preload_enabled():
        return False
    with _lock:
        if _started:
            return False
        _started = True
    threading.Thread(target=_run, name="biga-preload", daemon=True).start()
    return True


def _run() -> None:
    import time
    for mod in PRELOAD_TARGETS:
        start = time.perf_counter()
        try:
            __import__(mod)
            _logger.info("[预热] %s 就绪（%.1fs）", mod, time.perf_counter() - start)
        except Exception as e:                               # noqa: BLE001
            # 预热是加速手段，不是依赖：失败只记日志，绝不阻断 UI
            _logger.warning("[预热] %s 失败（忽略，功能不受影响）: %s", mod, e)
    _warm_data()


# ---------------- 数据 + 模板预热（2026-10-15 追加） ----------------
# 为什么还要预热"数据"：单纯 import 完只解决了一半。实测「看盘」页冷启动 ≈8.2s，构成是
#   `get_overnight_us()` 2.96s + `get_industry_flow_snapshot()` 2.65s +
#   `get_index_quotes()` / `get_realtime_quotes()` 各 0.79s + pandas Styler 首次模板 0.63s。
# ⇒ **首访慢的绝大部分是联网与模板初始化**，不是代码逻辑。
#
# ⚠️ 为什么这**不改变任何口径**：下面调用的都是**自带宽 TTL 缓存的只读取数**
#    （外盘 30min / 指数与自选行情 15s / 行业资金流 600s）⇒ 预热只是把成本从
#    "用户点击那一刻" 挪到 "用户还在看首页那一刻"；数据仍是同一接口、同一来源，
#    盘后取到的是当日最后成交价（页面本来就按这个口径展示）。
#
# ⚠️ 三条纪律：
#   ① **失败一律忽略**：无网/接口挂掉时页面照旧自己取数，预热不参与正确性；
#   ② **顺序执行、只跑一次**（由 `ensure_preloaded` 的模块级 flag 保证），避免并发打满限频；
#   ③ **不预热"写操作"**（下单、落库），只预热只读取数。
def _warm_data() -> None:
    import time
    for label, fn in _WARM_STEPS:
        start = time.perf_counter()
        try:
            fn()
            _logger.info("[预热] 数据·%s 就绪（%.1fs）", label, time.perf_counter() - start)
        except Exception as e:                               # noqa: BLE001
            _logger.warning("[预热] 数据·%s 失败（忽略，页面会自行取数）: %s", label, e)


def _warm_us() -> None:
    from data.us_market import get_overnight_us
    get_overnight_us()


def _warm_index() -> None:
    from data import realtime
    realtime.get_index_quotes()


def _warm_watchlist() -> None:
    from data import realtime, watchlist
    codes = watchlist.watchlist_codes()
    if codes:
        realtime.get_realtime_quotes(codes)


def _warm_industry_flow() -> None:
    from data.fund_flow import get_industry_flow_snapshot
    get_industry_flow_snapshot()


def _warm_paper_quotes() -> None:
    """预热**模拟账户（含各 AI 独立账本）持仓**的行情。

    为什么需要（2026-10-16 用户报"资金要加载才正确"）：
      页8 的「总资产 / 持仓市值」要按**现价**估；一只价都取不到时退化为**成本价**，
      于是「总资产 = 现金 + 成本 ≈ 初始资金」——用户看到 "1,000,000（+0.00%）"
      会以为账户被重置（该误读项目已踩过两次）。

    ⚠️ 这里**只预热只读取数**：`list_paper_accounts` / `load_paper_state` 是纯 DB 读，
       `get_realtime_quotes` 是行情读；不建账户、不写库、不下单。
    ⚠️ 自选股在别的步骤已预热过一遍，这里补的是**账户持仓**里自选池之外的部分。
    """
    from db import store
    from data import realtime
    codes: list = []
    for name in store.list_paper_accounts():
        state = store.load_paper_state(name) or {}
        codes.extend((state.get("positions") or {}).keys())
    codes = sorted({c for c in codes if c})
    if codes:
        realtime.get_realtime_quotes(codes)


def _warm_styler() -> None:
    """预热 pandas Styler 的模板初始化。

    实测：`df.style.map(...)` **首次 0.63s、第 2 次 0.000s** —— 一次性成本
    （jar 模板/环境初始化）。看盘页有 3 张带颜色样式的表 ⇒ 首次渲染白付一次。
    这里先"摸"一次把模板建好；结果**丢弃**（不保留任何数据引用）。
    """
    import pandas as pd
    pd.DataFrame({"_warm": [1.0]}).style.map(lambda v: "")


_WARM_STEPS: tuple[tuple[str, object], ...] = (
    ("外盘", _warm_us),
    ("指数行情", _warm_index),
    ("自选股行情", _warm_watchlist),
    ("模拟账户持仓行情", _warm_paper_quotes),
    ("行业资金流", _warm_industry_flow),
    ("Styler 模板", _warm_styler),
)
