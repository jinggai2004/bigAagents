# -*- coding: utf-8 -*-
"""写操作（P5）—— 选股 / 知乎采集 / 深度分析，全部走 `core.ui_tasks` 任务队列。

## 为什么任务名必须带 `web_` 前缀（C5）
`core/ui_tasks` 的进度/取消文件名只由 task 名决定（`progress_{task}.json`、
`cancel_{task}.json`）。Streamlit 页用的名字是 `screen` / `analysis_600519`；
网页端若同名，两边同时点会**互相覆盖进度、互相触发取消**。
⇒ 网页端一律 `web_` 前缀，两条入口并存也不会打架（新前端已是唯一入口，
   但旧页面还在本次改造周期内，不能假设没人点）。

## 为什么写操作不做缓存、不做"重试"
它们是**动作**不是查询：重试一次 = 真的再花一次钱/额度。
幂等由 `ui_tasks.start()` 的互斥（同 task 已在跑 ⇒ 返回 False）保证。

## 结果落盘位置
`output/ui_runs/result_{task}.json`（`core.ui_runs.save_result`）。
与 Streamlit 同机制 ⇒「刷新页面/换标签页后还能看到这次结果」是同一套保障，
不是另写一份。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

from web.server import envelope

_logger = logging.getLogger(__name__)

TASK_SCREEN = "web_screen"
TASK_ZHIHU = "web_zhihu"

# 深度分析一次可能跑 2~5 分钟、成本 ¥0.17；取消/重跑都必须走同一套互斥
def analysis_task(code: str) -> str:
    return f"web_analysis_{code}"


# ================================================================
# 任务状态（通用）
# ================================================================
def state(task: str, with_result: bool = False) -> dict:
    """任务的进度 + 是否在跑 +（可选）结果。任务不存在时不是 error。"""
    if not task:
        return envelope.fail("invalid", "task 为空")
    import core.ui_runs as ui_runs
    import core.ui_tasks as ui_tasks

    prog = ui_tasks.read_progress(task)
    running = ui_tasks.is_running(task)
    rec = ui_runs.get_running_any(task)

    # ⚠️ 不要拿 `progress.elapsed` 当"已用时长"：它只在**节点回调**时才被写一次。
    #    2026-10-26 实测：分析启动后的前 44 秒（择时节点取市场快照，东财 5 页
    #    连续失败重试 + akshare 兜底）里 elapsed 恒为 0.0，页面显示
    #    「准备中…（已用 0s）」四十多秒不动 —— 用户据此判断"深度分析卡住了"，
    #    而任务其实在正常推进、194 秒后成功落盘。
    #    ⇒ 用 started_ts 现算，只要进程还在，秒数就一直走；节点回调是否及时
    #      不能再决定"看起来是否还活着"。
    elapsed_live = None
    if running:
        started_ts = (rec or {}).get("started_ts")
        if isinstance(started_ts, (int, float)):
            elapsed_live = max(0.0, time.time() - float(started_ts))

    # 距上次进度更新的秒数：前端据此显示"最后更新 N 秒前"，
    # 超过阈值才提示"该步骤较慢"——比让进度条干冻着诚实。
    idle_seconds = None
    updated_at = (prog or {}).get("updated_at")
    if isinstance(updated_at, str) and updated_at:
        try:
            idle_seconds = max(0.0, (datetime.now() - datetime.strptime(
                updated_at, "%Y-%m-%d %H:%M:%S")).total_seconds())
        except ValueError:
            idle_seconds = None

    data = {
        "task": task,
        "running": running,
        "started_at": (rec or {}).get("started_at") if rec else None,
        "elapsed_live": round(elapsed_live, 1) if elapsed_live is not None else None,
        "idle_seconds": round(idle_seconds, 1) if idle_seconds is not None else None,
        "progress": prog,
        # 从没跑过 ⇒ missing（不是"完成但没结果"）；跑过但无结果 ⇒ empty
        "status": (prog or {}).get("status") or ("running" if running else None),
    }
    if with_result:
        # ⚠️ `ui_runs.load_result` 返回的是**信封** `{task, saved_ts, generated_at, payload}`，
        #    不是 payload 本体。直接塞给前端会让它把 `.picks` / `.market_signal` 读成
        #    undefined（页面显示"无候选"而结果文件其实好的）。这里必须解包。
        rec = ui_runs.load_result(task)
        if rec is not None:
            data["result"] = rec.get("payload")
            data["result_at"] = rec.get("generated_at")
        elif prog is None:
            return envelope.fail("missing", f"任务 {task} 从未运行")
        else:
            data["result"] = None
            data["result_at"] = None
    return envelope.ok(data, as_of=(prog or {}).get("updated_at"))


def cancel(task: str) -> dict:
    """请求取消。**只是请求**：真正停下要等任务跑到下一个进度点。"""
    import core.ui_tasks as ui_tasks
    if not task:
        return envelope.fail("invalid", "task 为空")
    if not ui_tasks.is_running(task):
        return envelope.fail("missing", f"任务 {task} 未在运行")
    ok = ui_tasks.request_cancel(task)
    if not ok:
        return envelope.fail("empty", "取消标记写入失败")
    return envelope.ok({"task": task, "cancel_requested": True})


# ================================================================
# 选股
# ================================================================
def run_screen(sector: str | None = None, strategies: list | None = None,
               max_stocks: int = 5) -> dict:
    """触发一次全市场选股（后台线程，2~20 分钟）。"""
    import core.ui_runs as ui_runs
    import core.ui_tasks as ui_tasks

    from core.screener import STRATEGIES

    if strategies:
        unknown = [s for s in strategies if s not in STRATEGIES]
        if unknown:
            # 未知策略名**不能静默当成"全部"**：用户以为按某个策略筛，实际跑的是别的
            return envelope.fail("invalid", "未知初筛策略：" + "、".join(unknown))
    try:
        max_stocks = int(max_stocks)
    except (TypeError, ValueError):
        return envelope.fail("invalid", "max_stocks 必须是整数")
    if not 1 <= max_stocks <= 30:
        return envelope.fail("invalid", "max_stocks 超出 1~30")

    def _job(progress):
        def _cb(ev: dict):
            progress({
                "percent": ev.get("percent", 0),
                "message": ev.get("message", ""),
                "current": ev.get("current", 0),
                "total": ev.get("total", 0),
            })

        from core.screener import StockScreener
        result = StockScreener().run(sector_filter=sector, max_stocks=max_stocks,
                                     strategies=strategies, progress_cb=_cb)
        ui_runs.save_result(TASK_SCREEN, {**result,
                                          "market_signal": _signal_dict(result.get("market_signal"))})
        try:
            from core.screen_history import save_screen_result
            save_screen_result(result, source="web")
        except Exception as e:                 # noqa: BLE001
            # 落库失败不能让这次选股结果丢失（save_result 已经存了一份）
            _logger.warning("[jobs] 候选入库失败: %s: %s", type(e).__name__, e)

    if not ui_tasks.start(TASK_SCREEN, _job, {"sector": sector,
                                              "strategies": strategies or [],
                                              "max_stocks": max_stocks}):
        return envelope.fail("stale", f"选股已在运行（task={TASK_SCREEN}）")
    return envelope.ok({"task": TASK_SCREEN, "started": True})


def _signal_dict(signal):
    """MarketSignal(dataclass) → dict；`None`/异常一律 None（不编造）。"""
    if signal is None or isinstance(signal, dict):
        return signal
    try:
        import dataclasses
        return dataclasses.asdict(signal)
    except (TypeError, ValueError) as e:
        _logger.warning("[jobs] 市场信号转 dict 失败: %s: %s", type(e).__name__, e)
        return None


def screen_options() -> dict:
    """初筛策略 + 板块清单（前端下拉/勾选的唯一来源，不在 JS 里写死一份）。"""
    from core.screener import SECTOR_HARDCODED, STRATEGIES
    sectors = sorted(SECTOR_HARDCODED.keys()) if isinstance(SECTOR_HARDCODED, dict) else []
    return envelope.ok({
        "strategies": [{"key": k, "desc": v} for k, v in STRATEGIES.items()],
        "sectors": sectors,
    })


# ================================================================
# 知乎（情报页）
# ================================================================
def zhihu_status() -> dict:
    """鉴权态 + 额度 + 当前话题表。`auth_ok()` 用不耗额度的 `/user/contents`。

    ⚠️ 2026-10-26：额外报 `web_channel`（v4 网页端通道的情况）。
    开放平台 token 与网页端 v4 是**两条独立通道**：前者要 40 位 secret 且耗额度，
    后者免 token、不扣额度，情绪采集走的就是后者。只报 `auth_ok()` 会让人误以为
    "没配 token ⇒ 知乎用不了"，然后去查一个根本用不到的凭证。
    """
    import data.zhihu_opinions as Z

    try:
        ok, msg = Z.auth_ok()
        quota = Z.quota_status()
        topics = Z.load_topics()
        members = Z.load_members()
        cookie_on = bool(Z.load_cookie())
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[jobs] 知乎状态读取失败: %s: %s", type(e).__name__, e)
        return envelope.fail("proxy", f"{type(e).__name__}: {e}")

    return envelope.ok({
        "auth": {"ok": bool(ok), "message": msg},
        "quota": quota,
        "topics": topics,
        "last_error": Z.last_error(),
        # v4 网页端：情绪采集真正走的通道，与 token 无关
        "web_channel": {
            "available": True,                # 端点本身不需要凭证；是否成功看采集结果
            "cookie_configured": cookie_on,
            "members": members,
            "members_count": len(members),
            "note": ("已配置登录 Cookie，风控概率更低" if cookie_on else
                     "未配置 Cookie（匿名访问多数时候可用；若反复出现"
                     "「N 次重试均未拿到 JSON」，把浏览器 Cookie 粘到 "
                     "config/zhihu_cookie.txt 即可）"),
        },
    })


def run_zhihu(per_topic: int = 5, topics: list | None = None) -> dict:
    """按话题采集（**消耗额度**）。结果为人工阅读用，不参与评分。"""
    import core.ui_runs as ui_runs
    import core.ui_tasks as ui_tasks
    import data.zhihu_opinions as Z

    st = Z.quota_status()
    left = max(0, st["limit"] - st["used"])
    if left <= 0:
        return envelope.fail("auth", f"知乎搜索额度已用尽（{st['used']}/{st['limit']}）")

    def _job(progress):
        progress({"percent": 0, "message": "按话题检索…"})
        items, meta = Z.collect(Z.load_topics() if not topics else topics,
                                per_topic=per_topic,
                                progress_cb=lambda ev: progress({
                                    "percent": (ev.get("current", 0) /
                                                max(1, ev.get("total", 1))) * 100,
                                    "message": f"检索「{ev.get('topic', '')}」",
                                    "current": ev.get("current", 0),
                                    "total": ev.get("total", 1)}))
        ui_runs.save_result(TASK_ZHIHU, {"items": items, "meta": meta})

    if not ui_tasks.start(TASK_ZHIHU, _job, {"per_topic": per_topic}):
        return envelope.fail("stale", f"采集已在运行（task={TASK_ZHIHU}）")
    return envelope.ok({"task": TASK_ZHIHU, "started": True,
                        "quota_left": left})


def zhihu_snapshot() -> dict:
    """最近一次采集的指标卡（Wilson 区间 / 分歧度 / 质量档）。

    ⚠️ 标注器固定 `rule`（零成本、可复现）。**不在这里调模型**：
       `build_for_source(labeler="llm")` 会真花钱，而这是每次开页面都会打的端点。
    """
    import core.ui_runs as ui_runs
    from core.sentiment_reference import build_snapshot

    # 同上：解包 `payload`。不解包会让 `items` 恒为空 ⇒ 采集成败都报"尚无采集结果"。
    pack = (ui_runs.load_result(TASK_ZHIHU) or {}).get("payload") or {}
    items = pack.get("items") or []
    if not items:
        return envelope.fail("missing", "尚无采集结果（先采集一次）")

    try:
        snap = build_snapshot(items, source="zhihu:topic",
                              as_of=(pack.get("meta") or {}).get("collected_at"))
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[jobs] 知乎情绪快照失败: %s: %s", type(e).__name__, e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")

    return envelope.ok({
        "snapshot": snap,
        "meta": pack.get("meta") or {},
        "headlines": [{"title": i.get("title") or "", "author": i.get("author") or "",
                       "votes": i.get("votes"), "comments": i.get("comments"),
                       "url": i.get("url") or ""} for i in items[:30]],
    }, as_of=snap.get("as_of"))


# ================================================================
# 深度分析
# ================================================================
def run_analysis(code: str, force: bool = False) -> dict:
    """触发一只票的深度分析（2~5 分钟、有 LLM 成本）。"""
    import core.ui_runs as ui_runs
    import core.ui_tasks as ui_tasks
    from core import analysis_runner

    code = (code or "").strip()
    if not (len(code) == 6 and code.isdigit()):
        return envelope.fail("invalid", "股票代码必须是 6 位数字")
    task = analysis_task(code)
    if ui_tasks.is_running(task) and not force:
        return envelope.ok({"task": task, "started": False, "running": True})

    def _job(progress):
        # ⚠️ 第一个进度回调要等 **timing 节点整体跑完**才发生，而它要串行算 14 个
        #    市场指标（`MarketDetector.detect()` 实为 for 循环，不是 docstring 说的
        #    线程池）。2026-10-26 实测：600519 用了 44 秒，000001 用了 **260 秒**，
        #    这期间页面的文案只有「准备中…」、进度 0% —— 被用户判断成"卡住了"。
        #    ⇒ 开局就给出具体在做什么 + 大致时长预期，别让人盯着 0% 猜。
        progress({
            "percent": 0.0,
            "message": "判市场状态 / 取行情（这一步通常 1~4 分钟，无中间进度）",
        })

        def on_step(node_name, final_state, step_idx):
            from core.analysis_history import snapshot_from_state
            progress({
                "percent": min(step_idx / analysis_runner.NODE_STEPS, 1.0) * 100,
                "message": f"步骤 {step_idx}: {node_name}",
                "node": node_name,
                "snapshot": snapshot_from_state(final_state),
            })

        res = analysis_runner.run_analysis(code, source="web", on_step=on_step)
        if not res.get("ok"):
            raise RuntimeError(res.get("error") or "分析失败")
        ui_runs.save_result(task, res.get("snapshot") or {})

    if not ui_tasks.start(task, _job, {"stock_code": code}):
        return envelope.ok({"task": task, "started": False, "running": True})
    return envelope.ok({"task": task, "started": True})


def analysis_history(code: str | None = None, limit: int = 30) -> dict:
    """历史分析列表（不解析大 payload，`list_analyses` 已如此约定）。"""
    from db import store
    code = (code or "").strip() or None
    if code is not None and not (len(code) == 6 and code.isdigit()):
        return envelope.fail("invalid", "股票代码必须是 6 位数字")
    try:
        rows = store.list_analyses(limit=max(1, min(200, limit)), stock_code=code)
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[jobs] 分析历史读取失败: %s: %s", type(e).__name__, e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")

    items = [{
        "id": r.get("id"), "date": r.get("analysis_date"),
        "code": r.get("stock_code"), "name": r.get("stock_name"),
        "price": r.get("price_at_analysis"), "decision": r.get("decision_label"),
        "model": r.get("model"), "source": r.get("source"),
        "regime": r.get("timing_regime"), "market_score": r.get("market_score"),
        "ret_5d": r.get("ret_5d"), "ret_20d": r.get("ret_20d"),
        "hit_target": r.get("hit_target"), "hit_stop": r.get("hit_stop"),
        "backfill_status": r.get("backfill_status"),
    } for r in rows]
    as_of = max((i["date"] for i in items if i["date"]), default=None)
    return envelope.ok({"rows": items, "total": len(items)}, as_of=as_of)


# ================================================================
# 历史回看（只读；不触发任何 LLM、不花钱）
# ================================================================
def analysis_detail(ana_id: int) -> dict:
    """单条分析的完整快照 —— 供历史列表点击回看。

    ⚠️ 与 `analysis_history` 的分工：列表**故意不解析 payload**（大 JSON 会拖慢
       UI，`list_analyses` 已如此约定），所以这里是唯一该调 `store.get_analysis`
       的地方。不要在列表里顺手解 payload。

    ⚠️ 返回的 `snapshot` 必须与"刚跑完的分析结果"**同名字段**：前端用同一个
       渲染器（`applySnapshot`）。字段名差一个字，回看页就整块显示"—"，
       而且看起来像"当时没算出来"，会被误当成数据缺陷。

    ⚠️ 补齐用**表列的真值**兜底，不用空串：`payload` 是分析当时的快照，
       老版本记录可能没有 `stock_code`/`trade_plan`；表列是落库时写的事实。
       已有的值**不覆盖**（payload 才是当时看到的）。
    """
    from db import store

    try:
        row = store.get_analysis(int(ana_id))
    except (TypeError, ValueError):
        return envelope.fail("invalid", "分析 id 必须是整数")
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[jobs] 分析详情读取失败 id=%s: %s: %s", ana_id, type(e).__name__, e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")
    if not row:
        return envelope.fail("missing", f"没有 id={ana_id} 的分析记录")

    payload = row.get("payload")
    snap = dict(payload) if isinstance(payload, dict) else {}

    def _fill(key, value):
        """只在缺失/空值时补，绝不覆盖 payload 里的原值。"""
        if not snap.get(key) and value not in (None, "", {}, []):
            snap[key] = value

    _fill("stock_code", row.get("stock_code"))
    _fill("stock_name", row.get("stock_name"))
    _fill("analysis_date", row.get("analysis_date"))
    _fill("current_price", row.get("price_at_analysis"))
    if not snap.get("trade_plan") and isinstance(row.get("trade_plan"), dict):
        snap["trade_plan"] = row["trade_plan"]
    # 这四项缺失会让审计区整块空白，必须给"真的没有"而不是 undefined
    _fill("data_availability", {})
    _fill("audit_warnings", [])
    _fill("errors", [])
    _fill("bull_history", [])
    _fill("bear_history", [])

    return envelope.ok({
        "id": row.get("id"),
        "snapshot": snap,
        "outcome": {
            "decision": row.get("decision_label"),
            "ret_5d": row.get("ret_5d"),
            "ret_20d": row.get("ret_20d"),
            "hit_target": row.get("hit_target"),
            "hit_stop": row.get("hit_stop"),
            "backfill_status": row.get("backfill_status"),
            "backfilled_at": row.get("backfilled_at"),
        },
        "meta": {
            "model": row.get("model"),
            "source": row.get("source"),
            "regime": row.get("timing_regime"),
            "market_score": row.get("market_score"),
            "market_score_source": row.get("market_score_source"),
            "created_at": row.get("created_at"),
        },
    }, as_of=row.get("analysis_date"))


# 复盘表只写这两种窗口；传别的值要变成"查不到数据"前的显式拒绝，不能静默退默认
REVIEW_HORIZONS = (5, 20)


def analysis_stats(days: int = 180, horizon: int = 5) -> dict:
    """往期胜率 + 归因分布（只读）。

    ## 口径（三条都踩过错，别省）
    1. **分母 = 判出方向的条数**。观望/驳回（`no_position`）单列、**绝不进分母**——
       否则"全观望"会被算成 0% 胜率，而它其实是"没预测"，不是"猜错了"。
    2. **`unclear`（±3% 内没走出来）进分母，但既不算对也不算错**，必须单列。
       把它并进"判错"会让胜率虚低，并进"判对"更糟——那是在改事实。
    3. **`horizon` 必须显式并回显**。T+5 与 T+20 的胜负是两件事，
       混算出来的比率没有任何意义（`list_analysis_reviews` 的 docstring 已强调）。

    ⚠️ 胜率字段名用 `win_rate_pct`（0~100 的百分数），**不用 `win_rate`**：
       研究页的回测 `win_rate` 也是百分数但来源完全不同（回测成交，不是分析复盘），
       同名会让人把两套口径当成一套。
    """
    from core.analysis_review import aggregate_recent

    try:
        days = max(1, min(3650, int(days)))
        horizon = int(horizon)
    except (TypeError, ValueError):
        return envelope.fail("invalid", "days / horizon 必须是整数")
    if horizon not in REVIEW_HORIZONS:
        return envelope.fail("invalid",
                             f"horizon 只支持 {REVIEW_HORIZONS}（T+5 早期结论 / T+20 完整窗口）")

    try:
        agg = aggregate_recent(days=days, horizon=horizon)
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[jobs] 复盘聚合失败: %s: %s", type(e).__name__, e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")

    bv = agg.get("by_verdict") or {}
    hit = int(bv.get("correct") or 0) + int(bv.get("partially_correct") or 0)
    miss = int(bv.get("wrong") or 0) + int(bv.get("opportunity_missed") or 0)
    unclear = int(bv.get("unclear") or 0)
    no_pos = int(bv.get("no_position") or 0)
    directional = hit + miss + unclear
    return envelope.ok({
        "days": days,
        "since": agg.get("since"),
        "horizon": int(agg.get("horizon") or horizon),
        "total_reviewed": int(agg.get("n") or 0),
        "directional": directional,
        "hit": hit,
        "miss": miss,
        "unclear": unclear,
        "no_position": no_pos,
        # None 表示"还没有可判定的样本"，与"0% 胜率"必须区分开
        "win_rate_pct": (round(hit / directional * 100, 1) if directional else None),
        "by_verdict": bv,
        "by_cause": agg.get("by_cause") or [],
        "advice": agg.get("advice") or [],
        "pending": agg.get("pending") or [],
        "min_support": agg.get("min_support"),
        "text": agg.get("text") or "",
    })
