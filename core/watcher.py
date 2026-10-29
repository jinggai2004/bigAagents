"""盘中异动预警轮询器（Phase 1）

职责：交易时段内定时拉取"自选股 + 今日候选"行情 → AlertEngine 判定 →
冷却去重 → 落库 + 推送（钉钉/飞书，复用 core/push.py）。

边界与取舍：
  - 只读不写交易状态：本模块**不会**自动平仓或下单，触及止损只发提醒
    （项目原则：不连接券商）。
  - 冷却期：同标的同规则默认 30 分钟只提醒一次，避免每分钟轮询刷屏。
  - 落库或推送失败不影响返回结果；但失败会记日志，不做静默吞掉。
"""

import json
import logging
import os
from datetime import datetime, time as dtime

from data.trading_calendar import is_trading_day
from core.alert_engine import AlertEngine, format_alerts_markdown
# Phase 1：复用统一的 requests 补丁（请求头 + 超时 + 按源最小间隔限频）。
# 盯盘每分钟请求一次腾讯/新浪，频率虽低，但必须和其他模块走同一套节流，
# 否则会出现"某条请求路径没人管"的盲区。
from data.akshare_helper import patch_requests

patch_requests()

_logger = logging.getLogger(__name__)

# 数据根运行时解析（打包后 `__file__` 在只读包内，推导会指向空目录）。
# 本模块目前没用到这个常量，但保持一致，避免以后有人误用同一种写法。
from config.paths import data_root as _paths_data_root  # noqa: E402

_PROJECT_ROOT = _paths_data_root()

# A 股连续竞价时段
SESSIONS = ((dtime(9, 30), dtime(11, 30)), (dtime(13, 0), dtime(15, 0)))
ALERT_COOLDOWN_SECONDS = 1800

# ⚠️ 2026-10-23：**只有 danger / warn 进消息**，info 级只落库。
#    理由（实测 10-23）：154 条告警里 **135 条是 info**、其中 **125 条是
#    "成交额 ≥ 10 亿"**——大盘蓝筹（工行 51 亿 / 茅台 67.8 亿 / 宁德 234 亿）
#    日常就在这个量级，对用户零信息量，却占掉 88% 的消息量。
#    落库全保留（页面可查），只是不再刷屏。
PUSH_LEVELS = ("danger", "warn")

# 关系标签来源（**按重要性排序**：持仓排最前）
_RELATION_SOURCES = (("positions", "持仓"), ("watchlist_user", "自选"),
                     ("watchlist_ashen", "阿深自选"), ("picks", "今日候选"))


def in_trading_session(now: datetime = None) -> bool:
    """当前是否处于交易日的连续竞价时段"""
    now = now or datetime.now()
    if not is_trading_day(now):
        return False
    t = now.time()
    return any(start <= t <= end for start, end in SESSIONS)


def _default_codes() -> list:
    """监控范围 = 自选股 ∪ 今日候选 ∪ **当前持仓**

    ⚠️ 2026-10-23 改动：**持仓必须进监控范围**。
    此前刻意不含持仓（注释写"持仓有独立的止损监控路径"），但那条路径是
    `run_risk_job`（14:50，**一天一次**）⇒ 盘中触及止损永远不会被提醒：
    `AlertEngine` 的 `stop_loss` 规则要求该票出现在 quotes 里，
    而 quotes 只取监控范围 ⇒ 持仓不在池子里 ⇒ **这条规则是死代码**
    （实测 10-23 全天 danger 级告警 0 条，与此一致）。
    持仓只有 1~2 只，并入几乎不增加盘中请求量（腾讯接口是批量的）。
    """
    try:
        from data import watchlist
        # 默认来源 = 两套自选（我的 + 阿深）∪ 今日候选 ∪ 持仓。
        # ⚠️ 不写死元组：`POOL_SOURCES` 是"盯盘范围"的唯一定义处，
        #    在这里再抄一份就会各自演化（过去踩过：新闻关联与盯盘范围不一致）。
        return watchlist.attention_codes()
    except Exception as e:
        _logger.warning("[盯盘] 关注池读取失败: %s", e)
        return []


def _relation_map() -> dict:
    """code → 与用户的关系标签（"持仓·自选"这种）。

    为什么要有它：告警只有"代码/名称/规则/现价"，用户看不出
    "这只票跟我什么关系"（是我的持仓？还是流水线候选？还是 AI 实验票？）。
    ⚠️ 各来源**分别**查 `attention_pool`（它自身会去重，分开查才拿得到多来源）。
    """
    out: dict = {}
    try:
        from data import watchlist
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[盯盘] 关系标签不可用: %s", e)
        return out
    for src, label in _RELATION_SOURCES:
        try:
            for item in watchlist.attention_pool((src,)):
                code = str(item.get("code") or "").strip()
                if not code:
                    continue
                prev = out.get(code)
                out[code] = f"{prev}·{label}" if prev else label
        except Exception as e:                              # noqa: BLE001
            _logger.debug("[盯盘] 关系来源 %s 读取失败（忽略）: %s", src, e)
    return out


def _stop_loss_map() -> dict:
    """从持仓记录取止损价。取不到返回空 dict——盯盘不应因持仓读取失败而中断。

    ⚠️ 2026-10-23 改为**纯 SQL 读表**（`store.list_active_positions()`）。
    原实现走 `PortfolioManager.get_active_positions()` → `load()` →
    `_price_of(code, None)` ⇒ **逐只联网取价**（实测单只 ≈30s）。
    盯盘是**每分钟**调一次的 ⇒ 每轮白等几十秒（实测盯盘单轮 21 秒）。
    止损价是**库里存的数**，与现价无关 ⇒ 读表即可。
    """
    try:
        from db import store
        return {p["code"]: float(p["stop_loss"]) for p in store.list_active_positions()
                if p.get("stop_loss")}
    except Exception as e:
        _logger.debug("[盯盘] 读取持仓止损价失败（忽略）: %s", e)
        return {}


def _position_notes(quotes: dict, stop_loss: dict) -> dict:
    """持仓类告警的补充信息：持股数 + 距止损多少。

    ⚠️ **不算"占净值 %"**：那需要总资产（现金 + 逐只市值），而
    `PortfolioManager.load()` 在缺价时会**回落联网取价**（≈30s/只）。
    持股数与"距止损几个点"不需要总资产，且信息上更直接（"你的票快到止损了"）。
    """
    notes: dict = {}
    try:
        from db import store
        positions = store.list_active_positions()
    except Exception as e:                                  # noqa: BLE001
        _logger.debug("[盯盘] 持仓补充信息读取失败（忽略）: %s", e)
        return notes
    for p in positions:
        code = p.get("code")
        q = (quotes or {}).get(code) or {}
        price = q.get("price")
        sl = stop_loss.get(code) or p.get("stop_loss")
        bits = []
        if p.get("quantity"):
            bits.append(f"持仓 {int(p['quantity'])} 股")
        if isinstance(price, (int, float)) and price > 0 and isinstance(sl, (int, float)) and sl > 0:
            bits.append(f"止损 {sl:.2f}（现价差 {(sl / price - 1) * 100:+.1f}%）")
        if bits:
            notes[code] = "，".join(bits)
    return notes


class AlertWatcher:
    """一次 scan_once() = 一轮完整检查（拉行情 → 判定 → 去重 → 落库 → 推送）"""

    def __init__(self, engine: AlertEngine = None,
                 cooldown_seconds: int = ALERT_COOLDOWN_SECONDS, push: bool = True):
        self.engine = engine or AlertEngine()
        self.cooldown_seconds = cooldown_seconds
        self.push = push
        self._last_fired: dict = {}

    # ------------------------------------------------------------
    def _filter_cooldown(self, alerts: list, now_ts: float) -> list:
        fresh = []
        for a in alerts:
            key = f"{a['code']}|{a['rule']}"
            last = self._last_fired.get(key)
            if last is not None and now_ts - last < self.cooldown_seconds:
                continue
            self._last_fired[key] = now_ts
            fresh.append(a)
        return fresh

    def scan_once(self, codes: list = None, now: datetime = None,
                  ma20: dict = None, stop_loss: dict = None,
                  force: bool = False) -> dict:
        """跑一轮检查。

        force=True 用于**手动检查**（UI 按钮 / 调试）：跳过交易日与交易时段闸门，
        但仍如实标注数据是"最近一个交易时段的收盘快照"，避免被误读为盘中实时判断。
        """
        now_dt = now or datetime.now()
        if not force:
            if not is_trading_day(now_dt):
                return {"skipped": True, "reason": "非交易日", "alerts": 0, "fired": 0}
            if not in_trading_session(now_dt):
                return {"skipped": True, "reason": "非交易时段", "alerts": 0, "fired": 0}

        codes = codes if codes is not None else _default_codes()
        # AI 交易联赛（S5）：把"AI 已武装条件涉及的代码"并入监控范围。
        # ⚠️ 必须在**取数之前**并入 —— 目的就是复用同一次行情请求（盘中 0.5 限频预算内，
        #    多拉一遍会让盯盘自己超时）。数量上限见 ai_wake.MAX_ARMED_CODES。
        ai_armed = []
        try:
            from core import ai_wake
            ai_armed = ai_wake.armed_codes(now_dt.strftime("%Y-%m-%d"))
        except Exception as e:                              # noqa: BLE001
            _logger.warning("[盯盘] AI 唤醒条件读取失败（不影响盯盘）: %s", e)
        if ai_armed:
            codes = list(dict.fromkeys(list(codes) + ai_armed))
        if not codes:
            return {"skipped": True, "reason": "监控列表为空", "alerts": 0, "fired": 0}

        from data.realtime import get_realtime_quotes
        quotes = get_realtime_quotes(codes)
        stop_map = stop_loss if stop_loss is not None else _stop_loss_map()
        alerts = self.engine.evaluate(quotes, ma20=ma20, stop_loss=stop_map)
        fired = self._filter_cooldown(alerts, now_dt.timestamp())

        saved = self._persist(fired, now_dt)
        pushed = (self._push(fired, now_dt, quotes=quotes, stop_loss=stop_map,
                             ai_armed=ai_armed)
                  if (self.push and fired) else {})
        # ── 买入信号检测（2026-10-23）：**只落库，不推送** ──
        # 盘中数据是残缺的，结论必须由收盘后的深度分析给出；
        # 这里只把"达到买入条件"的票排进队列，由 `core.evening_analysis` 接手。
        # ⚠️ **force（手动检查）不跑信号检测**：非交易时段拿到的是收盘快照，
        #    "涨幅+放量"在快照上恒真 ⇒ 会把一批票误记成"盘中信号"。
        #    （与"手动检查不跑 LLM、不花钱"是同一条纪律。）
        signals_new = [] if force else self._detect_signals(quotes, now_dt, ai_armed)
        result = {"skipped": False, "checked": len(quotes), "alerts": len(alerts),
                  "fired": len(fired), "saved": saved, "push": pushed,
                  "codes": len(codes), "forced": bool(force),
                  "ai_armed_codes": len(ai_armed),
                  "signals": len(signals_new)}
        # AI 唤醒条件判定（**用同一份 quotes，零额外行情请求**）
        if ai_armed:
            try:
                from core import ai_wake
                hits = ai_wake.evaluate(quotes, trade_date=now_dt.strftime("%Y-%m-%d"),
                                        now=now_dt)
                if hits:
                    # ⚠️ 手动检查（force）**只记录不跑 LLM** —— 否则点一下按钮就花钱
                    wr = ai_wake.record_and_fire(
                        hits, trade_date=now_dt.strftime("%Y-%m-%d"), now=now_dt,
                        run_decision=not force)
                    result["ai_wake"] = {"hits": wr["hits"], "fired": wr["fired"],
                                         "skipped_idempotent": wr["skipped_idempotent"]}
            except Exception as e:                          # noqa: BLE001
                _logger.warning("[盯盘] AI 唤醒判定失败（不影响盯盘）: %s", e)
        if force and not in_trading_session(now_dt):
            result["note"] = "手动检查：非交易时段，行情为最近交易时段的收盘快照"
        # ── 阿深定时值班（2026-10-27）：**同一份 quotes，零额外行情请求** ──
        # 为什么挂在这里而不是新增 cron 任务：`intraday_watch` 每分钟一轮、
        # 自带"交易日 + 09:30-11:30/13:00-15:00"闸门，正好是"交易时段"的定义。
        # 新增任务会把 JOBS 从 19 条变 20 条，牵动 JOB_IDS 断言、冲突检测、
        # 文档端点计数，还会要求"改完必须重启调度器"。
        # ⚠️ 手动检查（force）**不跑 LLM** —— 与 ai_wake 的"只记录不跑"同一条纪律：
        #    点一下按钮就花钱是不可接受的。
        if not force:
            try:
                from core import ai_routine
                rr = ai_routine.run_due(quotes, now=now_dt)
                # 只把"真跑了"或"真判了动作"的结果写进返回值：
                # 每分钟一次"无到点的轮次"会把 result 撑成日志噪声。
                if rr.get("llm_called") or rr.get("kept"):
                    result["ai_routine"] = rr
            except Exception as e:                          # noqa: BLE001
                _logger.warning("[盯盘] 阿深值班失败（不影响盯盘）: %s", e)
        return result

    # ------------------------------------------------------------
    @staticmethod
    def _persist(alerts: list, now_dt: datetime) -> int:
        if not alerts:
            return 0
        try:
            from db import store
            return store.insert_alerts(alerts, trade_date=now_dt.strftime("%Y-%m-%d"))
        except Exception as e:
            _logger.warning("[盯盘] 预警落库失败（不影响推送）: %s", e)
            return 0

    # ------------------------------------------------------------
    @staticmethod
    def _random_only_codes(now_dt: datetime) -> set:
        """**只**被 AI 随机对照组关注的代码（推送时排除）。

        ⚠️ 为什么不在监控池阶段剔除：并入 armed codes 是为了让
        `ai_wake.evaluate()` 能判定这些条件；剔了它们就永远不触发
        ⇒ 联赛行为被改变（公平性受损）。所以只在这里（推送层）排除。
        """
        try:
            from core import ai_wake
            return ai_wake.random_only_codes(now_dt.strftime("%Y-%m-%d"))
        except Exception as e:                              # noqa: BLE001
            _logger.warning("[盯盘] 实验票名单读取失败（不过滤）: %s", e)
            return set()

    @staticmethod
    def _relations(ai_armed: list = None) -> dict:
        """code → 关系标签（持仓/自选/今日候选/AI武装）。"""
        rel = _relation_map()
        for code in (ai_armed or []):
            c = str(code)
            prev = rel.get(c)
            rel[c] = f"{prev}·AI武装" if prev else "AI武装"
        return rel

    @staticmethod
    def _detect_signals(quotes: dict, now_dt: datetime, ai_armed: list = None) -> list:
        """检测"达到买入条件"的票并落库（**不推送**）。

        范围 = 自选 ∪ 今日候选 ∪ 持仓（**排除 AI 武装票** —— 那是联赛实验对象，
        不该进入"给你的买入信号"）。AI 武装票在关系标签里带「AI武装」标记，
        据此排除 ⇒ 不用在两处各写一份名单。
        """
        try:
            from core import signal_queue
            relations = AlertWatcher._relations(ai_armed)
            scope = [c for c, rel in relations.items() if "AI武装" not in rel]
            if not scope:
                return []
            sigs = signal_queue.detect_signals(quotes, scope, relations)
            return signal_queue.record_signals(sigs, now_dt.strftime("%Y-%m-%d"))
        except Exception as e:                              # noqa: BLE001
            _logger.warning("[盯盘] 买入信号检测失败（不影响盯盘）: %s", e)
            return []

    def _push(self, alerts: list, now_dt: datetime, quotes: dict = None,
              stop_loss: dict = None, ai_armed: list = None) -> dict:
        """分级推送：**只推 danger/warn**，info 级只落库（2026-10-23）。

        实测依据：10-23 一天 154 条告警里 135 条是 info（其中 125 条是
        "成交额 ≥ 10 亿"的大盘蓝筹常态），占了 88% 的消息量却零信息量。
        """
        try:
            from core.push import push_markdown
            leveled = [a for a in alerts if a.get("level") in PUSH_LEVELS]
            random_codes = self._random_only_codes(now_dt)
            if random_codes:
                leveled = [a for a in leveled if str(a.get("code")) not in random_codes]
            if not leveled:
                return {"提示": "无 danger/warn 级告警（info 级只落库不推送）"}
            relations = self._relations(ai_armed)
            notes = _position_notes(quotes or {}, stop_loss or {})
            text = format_alerts_markdown(
                leveled, title=f"bigAagents 盘中异动 {now_dt:%H:%M}",
                relations=relations, notes=notes)
            return push_markdown(text, title=f"bigAagents 盘中异动 {now_dt:%H:%M}")
        except Exception as e:
            _logger.warning("[盯盘] 推送失败: %s", e)
            return {"错误": str(e)}


_watcher = None


def get_watcher() -> AlertWatcher:
    """进程内单例（冷却状态需要跨轮询保留）"""
    global _watcher
    if _watcher is None:
        _watcher = AlertWatcher()
    return _watcher
