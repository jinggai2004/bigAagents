"""
分析历史入库与 T+N 回看闭环。

背景：深度分析结果此前只存 output/ui_runs/analysis_{code}.json——每只股票仅
保留最后一份且会覆盖，无时间维度、无历史回放、无收益核对，"模型实际胜率"
这个问题永远无法回答。

本模块复制 observations 表已验证的"预测 → 次日核对 → 命中率"范式到分析决策层：
- save_analysis / save_failure：分析结束即落库（best-effort，绝不影响主流程）
- backfill_analyses：每日收盘后回填 T+5 / T+20 收益率与目标/止损触及
- 状态机：NULL → partial → done；超龄 → failed（仅手动重试）；分析失败 → skip

回填口径（与 daily_features.future_ret_5d/20d 对齐，可交叉验证）：
- 基准价 = 分析日（含）之后第一根日线收盘价（前复权）
- ret_5d  = 第 5 根后续收盘 / 基准收盘 - 1
- hit_target/hit_stop = 基准日之后的最高/最低价 vs trade_plan 目标价/止损价
  （trader 输出的价位是字符串如"18.5元"，容错解析首个数字；解析不出置 NULL）
"""

import logging
import os
import re
import time
from datetime import datetime, timedelta

import pandas as pd

import db.store as store
from core.forward_returns import (DEFAULT_HORIZONS, compute_forward_returns,
                                  default_price_fetcher, normalize_kline)

_logger = logging.getLogger(__name__)

# 回填时限（自然日）：超龄仍未满足 T+20 数据 → failed，不再每日重试。
# 60 天 > 20 个交易日 + 节假日缓冲，正常停牌复牌也能覆盖。
BACKFILL_MAX_AGE_DAYS = 60

# 每日触发核对只回看这么多自然日的分析（≈20 个交易日）。
# 更早的样本已由 T+N 回填写定 hit_target/hit_stop，无需每天重复核对。
TRIGGER_CHECK_MAX_AGE_DAYS = 30


# ================================================================
# 快照（从 ui/pages/2 移植，UI 页面改为委托此处）
# ================================================================

def _json_safe(obj):
    """转成可落盘的形态：numpy 标量 → python 标量，DataFrame 等不可序列化对象丢弃"""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return None


def snapshot_from_state(final_state: dict) -> dict:
    """从 LangGraph final_state 提取展示/落库需要的字段（DataFrame 不进快照）

    v3（2026-09-28）新增 4 个字段：
      - analysis_date / data_through：报告口径（执行日 / 数据截止），此前**不落盘**，
        用户在前端看到的报告没有任何时间锚点，也无从判断"这是哪天数据"；
      - data_availability：各输入是否真的取到（缺失项要能一眼看到，而不是靠"没写"来暗示）；
      - audit_warnings：可核实性审计挑出的幻觉线索（输入里找不到出处的年份/机构/数据商）。
    """
    info = final_state.get("stock_info") or {}
    errors = list(final_state.get("errors") or [])
    return {
        "stock_code": final_state.get("stock_code", ""),
        "stock_name": info.get("name") or final_state.get("stock_name", ""),
        "current_price": info.get("current_price"),
        "analysis_date": final_state.get("analysis_date") or "",
        "data_through": final_state.get("data_through") or "",
        "data_availability": dict(final_state.get("data_availability") or {}),
        "tech_report": final_state.get("tech_report") or "",
        "fundamental_report": final_state.get("fundamental_report") or "",
        "news_report": final_state.get("news_report") or "",
        "bull_history": list(final_state.get("bull_history") or []),
        "bear_history": list(final_state.get("bear_history") or []),
        "research_summary": final_state.get("research_summary") or "",
        "trade_plan": _json_safe(final_state.get("trade_plan") or {}),
        "risk_assessment": final_state.get("risk_assessment") or "",
        "final_decision": final_state.get("final_decision") or "",
        "errors": errors,
        "audit_warnings": [e for e in errors if "可核实性" in str(e)],
        "fatal_errors": list(final_state.get("fatal_errors") or []),
        "fund_flow_data": _json_safe(final_state.get("fund_flow_data") or {}) or {},
    }


# ================================================================
# 落库
# ================================================================

# 一次深度分析会跑到的 9 个节点（与 `core/agents.py` 的图一一对应）。
# 用于记录"这次结论实际由哪些模型产出"——按功能分配不同模型时，单一
# `DEFAULT_MODEL_NAME` 无法代表真实构成。
_ANALYSIS_FEATURES = (
    "client.technical_analysis", "client.fundamental_analysis",
    "agents.news", "agents.bull", "agents.bear",
    "agents.research_manager", "agents.trader",
    "agents.risk_manager", "agents.portfolio_manager",
)


def _current_model_name() -> str:
    """本次分析**实际用到的模型**（混用时写 `mixed:a+b`，历史归因用）。

    ⚠️ 原实现直接返回 `config.DEFAULT_MODEL_NAME` —— 在"按功能分配不同模型"的
    部署里这个值是**错的**：所有分析都记成同一个模型，于是"按模型分组比胜率"
    这条纪律变成空话（2026-10-20 换 qwen3.7-flash 时发现）。
    ⚠️ 兜底仍走 `from config.settings import config`（历史上 `import config`
    取属性会 AttributeError，被静默吞成 "unknown"，模型维度整列作废）。
    """
    try:
        from config.model_routing import resolve_model
        names = {resolve_model(f) for f in _ANALYSIS_FEATURES}
        names = {n for n in names if n}
        if len(names) == 1:
            return names.pop()
        if names:
            return "mixed:" + "+".join(sorted(names))
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[分析历史] 解析实际模型失败，退回全局默认: %s", e)
    try:
        from config.settings import config
        return config.DEFAULT_MODEL_NAME or "unknown"
    except Exception:                           # noqa: BLE001
        return "unknown"


# 终审的"否决类"措辞：出现即说明**计划没有被执行**。
# 为什么必须优先于 `trade_plan.direction`（2026-10-09 实测）：
#   id=11 的 `trade_plan.direction` 是"观望"、`final_decision` 是"**驳回。** 仓位严重超限…"，
#   而旧逻辑"优先取 direction"→ 历史列表显示"观望"，用户看到的是"没表态"，
#   实际结论是**否掉**。更糟的是当 direction 是"看多"而终审驳回时，
#   该样本会被算进胜率统计（`store` 的胜率口径按 decision_label 分组）。
_REJECT_MARKERS = ("驳回", "否决", "不予批准", "不通过", "拒绝", "撤销")


def _extract_decision(final_decision: str, trade_plan: dict) -> str:
    """决策标签：终审否决 > trade_plan.direction（结构化）> final_decision 文本

    ⚠️ 否决判据只看**结论句**（正文第一句），不看全文 —— 否则
    "批准。风控曾以超限为由驳回，但…"（终审有权在写明依据后覆盖风控）
    会被误判成"驳回"。结论句就是模型自己声明的那一句（"**驳回。**" / "最终决策：需修改"）。
    """
    text = str(final_decision or "")
    head = text[:40].split("。")[0].split("\n")[0]
    if any(mk in head for mk in _REJECT_MARKERS):
        # 被否掉的计划不该显示成"观望"，也不该按方向计入胜率
        return "驳回"
    direction = str((trade_plan or {}).get("direction") or "")
    for label in ("看多", "看空", "观望"):
        if label in direction:
            return label
    for label in ("看多", "看空", "观望"):
        if label in text:
            return label
    # 有决策文本但无法归类 → 观望；完全没跑出决策（链被阻断）→ 空串
    return "观望" if text else ""


def save_analysis(final_state: dict, source: str = "ui") -> int:
    """分析成功后落库一条历史（best-effort：任何异常吞掉记日志，返回 0）。

    source: "ui"（深度分析页）/ "pipeline"（每日流水线 llm 模式）
    """
    try:
        snap = snapshot_from_state(final_state)
        timing = final_state.get("timing_signal") or {}
        plan = snap.get("trade_plan") or {}
        record = {
            "analysis_date": final_state.get("analysis_date")
                             or datetime.now().strftime("%Y-%m-%d"),
            "stock_code": final_state.get("stock_code", ""),
            "stock_name": snap.get("stock_name", ""),
            "price_at_analysis": snap.get("current_price"),
            "decision_label": _extract_decision(snap.get("final_decision"), plan),
            "model": _current_model_name(),
            "source": source,
            "payload": snap,
            "trade_plan": plan,
            "timing_regime": timing.get("regime"),
            "market_score": timing.get("total_score"),
            # 2026-10-20：分析与选股统一到 MarketDetector 后，market_score 的
            # "尺子"变了（旧=TimingEngine 的 16.9 量级，新=MarketDetector 的 37.9 量级）。
            # 不记来源的话，按分数筛历史样本会把两把尺子混着用。
            "market_score_source": timing.get("source"),
        }
        # 分析链被致命错误阻断的样本：不参与胜率统计，但要留痕（避免幸存者偏差）。
        # 决策链已阻断，trade_plan 里即使有残留 direction 也无效——无条件标分析失败。
        if final_state.get("fatal_errors"):
            record["backfill_status"] = "skip"
            record["decision_label"] = "分析失败"
        return store.insert_analysis(record)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[分析历史] save_analysis 失败: %s", e)
        return 0


def save_failure(stock_code: str, error: str, source: str = "ui",
                 stock_name: str = "") -> int:
    """分析异常/中断的样本入库（decision_label=分析失败，backfill_status=skip）"""
    try:
        return store.insert_analysis({
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "stock_code": stock_code,
            "stock_name": stock_name,
            "decision_label": "分析失败",
            "model": _current_model_name(),
            "source": source,
            "payload": {"fatal_errors": [str(error)]},
            "backfill_status": "skip",
        })
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[分析历史] save_failure 失败: %s", e)
        return 0


# ================================================================
# T+N 回填
# ================================================================

def _parse_price(value) -> float | None:
    """从 trader 输出的价位字符串（如 '18.5元' / '18.5-19.0'）解析首个数字"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(m.group()) if m else None


def evaluate_triggers(direction: str, target, stop,
                      highs: list, lows: list) -> dict:
    """目标价 / 止损价是否被触及 —— **唯一实现**。

    回填（`_compute_backfill`）、每日核对（`check_triggers_daily`）与行动清单
    （`core.decision_digest.action_list`）都调这里。本项目多次踩过"两侧各注一份"：
    触及口径一旦分叉，"到底有没有触发"就会有两个互相矛盾的答案。

    看空用**对称口径**：目标是**下方**（跌破才算触及）、止损在**上方**（涨破才算触及）。
    这与 `TRADE_DIRECTION_RULES` 的"看空=驳回买入、给入场条件"配套 ——
    看空的价位是给"跌破后确认"用的，不是做空仓位。

    目标/止损解析不出或窗口无 K 线时，对应键**缺席**（不是 0）："没算"与"算了没触及"
    是两件事，回填的 `stats` 与界面都按"缺就是缺"处理。
    """
    out: dict = {}
    bearish = "看空" in str(direction or "")
    if target is not None and highs and lows:
        out["hit_target"] = int(any(l <= target for l in lows) if bearish
                                else any(h >= target for h in highs))
    if stop is not None and highs and lows:
        out["hit_stop"] = int(any(h >= stop for h in highs) if bearish
                              else any(l <= stop for l in lows))
    return out


def _compute_backfill(df: pd.DataFrame, analysis_date: str, trade_plan: dict) -> dict:
    """基于分析日（含）之后的第一根收盘为基准，计算可得的 T+N 字段。

    T+N 收益口径统一走 `core.forward_returns`（选股候选回看共用同一套），
    本函数只补上分析特有的"目标/止损是否触及"。
    """
    fr = compute_forward_returns(df, analysis_date)
    rets = {k: v for k, v in fr.items() if k.startswith("ret_")}
    if not rets:
        return {}          # 一根 K 线都没走完：保持原状态，明天再试
    out: dict = dict(rets)

    # 目标/止损触及：基准日之后的窗口（最多 T+20），不包含基准日当天。
    # 判定逻辑抽到 `evaluate_triggers`（唯一实现）——每日核对与行动清单共用它。
    rows = df[df["date"] >= analysis_date]
    window = rows.iloc[1:21]
    if not window.empty and trade_plan:
        direction = str(trade_plan.get("direction") or "")
        target = _parse_price(trade_plan.get("target_price"))
        stop = _parse_price(trade_plan.get("stop_loss"))
        highs = [float(h) for h in window["high"]] if "high" in window else []
        lows = [float(l) for l in window["low"]] if "low" in window else []
        out.update(evaluate_triggers(direction, target, stop, highs, lows))
    return out


def _nth_trading_day(start_date: str, n: int, max_scan: int = 420) -> str | None:
    """从 `start_date`（含）起的第 n 个交易日（n=0 即它本身，若它是交易日）。

    ⚠️ 刻意**逐日用 `is_trading_day` 推进**，而不是调 `trading_calendar.trading_days(s, e)`：
    后者在 `e` 超出日历覆盖范围时**静默退回「周一至周五」枚举**（不报 degraded），
    于是国庆假期会被算成交易日，算出来的「可回填日」落在过去 —— 比不报更误导。
    （2026-10-09 实测踩到：探针里 end 取 start+120 天越过了日历上限 2026-12-31。）
    """
    from data import trading_calendar as tc
    try:
        cur = datetime.strptime(str(start_date)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    hit = -1
    for _ in range(max_scan):
        ds = cur.strftime("%Y-%m-%d")
        if tc.is_trading_day(ds):
            hit += 1
            if hit == n:
                return ds
        cur += timedelta(days=1)
    return None


def _trading_days_between(after: str, upto: str) -> int:
    """(after, upto] 之间的交易日数量（不含 after、含 upto）"""
    from data import trading_calendar as tc
    try:
        cur = datetime.strptime(str(after)[:10], "%Y-%m-%d").date() + timedelta(days=1)
        end = datetime.strptime(str(upto)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return 0
    n = 0
    while cur <= end:
        if tc.is_trading_day(cur.strftime("%Y-%m-%d")):
            n += 1
        cur += timedelta(days=1)
    return n


def backfill_progress(limit: int = 100, today: str = None) -> dict:
    """待回填记录的可读进度：**还差几个交易日、预计哪天可回填**。

    为什么需要它（2026-10-09）：用户报「深度分析一次回填都没有」，而日志只有
    `checked=7, skipped=7, errors=[]` —— 「正常的等天数」与「取数坏了」长得一模一样，
    界面也只显示一个「待回填 12」，看不出是**在等**还是**坏了**。
    本函数把两件事分开说清楚，且**完全离线**（只用交易日历，不拉 K 线），
    所以页面每次渲染都能调。

    Returns:
        {"pending": int, "ready_now": int, "waiting": int, "degraded": bool,
         "last_trading_day": str, "horizons": (5, 20), "rows": [...],
         "summary": str, "soonest": {...} | None}
    """
    from data import trading_calendar as tc
    today = (today or datetime.now().strftime("%Y-%m-%d"))[:10]
    # ⚠️ **保守基准：不含今天**。当日 K 线通常要收盘后甚至次日才拿得到
    #    （2026-10-09 15:06 实测：数据源最新只到 10-08），把"今天"当已有数据会
    #    报出「已可回填」但实际仍然写不进去 —— 那是**新的假信号**，
    #    而这次的起因正是"看不出在等还是坏了"。
    #    宁可少报一天，也不许诺一个当天拿不到的结果。
    last_td = tc.last_trading_day(ref=today, include_today=False)
    horizons = tuple(DEFAULT_HORIZONS)
    rows = store.list_analyses(limit=limit, only_backfillable=True)

    out_rows: list = []
    for r in rows:
        ana = str(r.get("analysis_date") or "")[:10]
        item = {"id": r["id"], "code": r.get("stock_code"), "name": r.get("stock_name"),
                "analysis_date": ana, "ready": {}, "missing": {}, "stage": "", "reason": "",
                # 是否已写入过 T+5（`partial` 状态的行仍会被列出来继续等 T+20，
                # 此时它已经不是"可写"而是"已写、等 T+20"——两者不能混为一谈）
                "has_ret_5d": r.get("ret_5d") is not None}
        for h in horizons:
            ready = _nth_trading_day(ana, h) if ana else None
            item["ready"][h] = ready
            # 还差几个交易日（相对「最近一个交易日」；数据源当天收盘后才有当日 K 线）
            item["missing"][h] = (_trading_days_between(last_td, ready)
                                  if ready and ready > last_td else 0)
        # （基准为"昨天及以前"）
        m5 = item["missing"].get(horizons[0], 0) if item["ready"].get(horizons[0]) else None
        m20 = item["missing"].get(horizons[1], 0) if item["ready"].get(horizons[1]) else None
        if m5 is None or m20 is None:
            item["stage"] = "无法判定"
            item["reason"] = "分析日缺失或交易日历覆盖不足"
        elif m5 > 0:
            item["stage"] = f"等 T+{horizons[0]}"
            # 说清"第几根 K 线、哪天收盘后可得"——原先只写"预计 X 可回填"，
            # 而那天正是任务运行日 15:35 时刻**当天 K 线还没出**（2026-10-09 实测），
            # 于是 600519 在"预计可回填日"当天仍然 not_ready，看起来像坏了。
            item["reason"] = (f"T+{horizons[0]} 还差 {m5} 个交易日"
                              f"（需第 {horizons[0] + 1} 根 K 线："
                              f"{item['ready'][horizons[0]]}，该日收盘后可得）")
        elif m20 > 0:
            if item["has_ret_5d"]:
                # 已经写过 partial（T+5 已入库），现在等的是 T+20
                item["stage"] = f"等 T+{horizons[1]}"
                item["reason"] = (f"T+{horizons[0]} 已入库（部分回填），"
                                  f"T+{horizons[1]} 还差 {m20} 个交易日"
                                  f"（预计 {item['ready'][horizons[1]]}）")
            else:
                # ⚠️ 这一档**现在就能写入 T+5**（只是 T+20 未到）—— 原先写"等 T+20"，
                #    与"还要等"混为一谈，于是"已可回填 0 条"让用户以为全都没到窗口。
                item["stage"] = f"可写 T+{horizons[0]}"
                item["reason"] = (f"T+{horizons[0]} 已可算（K 线已到 "
                                  f"{item['ready'][horizons[0]]}），"
                                  f"T+{horizons[1]} 还差 {m20} 个交易日"
                                  f"（预计 {item['ready'][horizons[1]]}）")
        else:
            item["stage"] = "可回填"
            item["reason"] = "窗口已走完，下次回填任务即可写入"
        out_rows.append(item)

    waiting = [x for x in out_rows if x["stage"] != "可回填"]
    ready_now = [x for x in out_rows if x["stage"] == "可回填"]
    # 「现在就能写 T+5（partial）」的行数 = K 线已够 **且尚未写入过**。
    # `ready_now` 要求 T+20 也走完，于是这一档被算进了"还在等"—— 2026-10-10 用户
    # 就是被这条口径糊住的：600519 的分析日 09-24、第 6 根 K 线是 10-09，
    # 10-09 之后已完全可写，却显示"已可回填 0 条"。
    partial_ready = [x for x in out_rows
                     if x["ready"].get(horizons[0]) and x["missing"].get(horizons[0]) == 0
                     and not x["has_ret_5d"]]
    soonest = None
    if waiting:
        # 按"还差的交易日"排序取最快的那条
        soonest = min(waiting, key=lambda x: (x["missing"].get(horizons[0]) or 999,
                                              x["missing"].get(horizons[1]) or 999))
    status = tc.cache_status()
    summary = (f"{len(out_rows)} 条待回填；其中 **{len(partial_ready)} 条现在就能写入 "
               f"T+{horizons[0]}**（K 线已够），**{len(ready_now)} 条**连 T+{horizons[1]} "
               f"也已走完（基准：{last_td} 及以前的 K 线）。")
    if soonest:
        summary += (f" 最快的一条（{soonest['code']} {soonest['analysis_date']}）"
                    f"{soonest['reason']}。")
    if partial_ready:
        # 回填任务只在**交易日 15:35** 跑（周末/节假日不跑）—— 不说清这一点，
        # "明明已经可以回填却一直空着"就无从解释。
        window = _next_backfill_window(now=datetime.now())
        if window:
            summary += f" ⚠️ 回填任务只在**交易日 15:35** 运行，下一个窗口：{window}。"
    if status.get("degraded"):
        summary += " ⚠️ 交易日历当前为「周一至周五」近似（不含节假日），日期估算可能偏早。"
    return {"pending": len(out_rows), "ready_now": len(ready_now), "waiting": len(waiting),
            "partial_ready": len(partial_ready),
            "next_backfill_window": _next_backfill_window(now=datetime.now()),
            "degraded": bool(status.get("degraded")), "last_trading_day": last_td,
            "horizons": tuple(horizons), "rows": out_rows,
            "summary": summary, "soonest": soonest, "calendar": status}


def _next_backfill_window(now: datetime = None) -> str | None:
    """回填任务的下一个运行窗口（交易日 15:35）。

    为什么需要：任务 cron 是 `周一至周五 15:35`，**周末与节假日不跑** ——
    于是"周五收盘后才算得出的 T+N"要一直等到下周一，页面上不说清就只剩
    "怎么一直没回填"这一个观感。这里把窗口直接算出来给用户看。
    """
    from data import trading_calendar as tc
    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")
    if tc.is_trading_day(today) and (now.hour, now.minute) < (15, 35):
        return f"{today} 15:35（今天）"
    cur = now.date()
    for _ in range(15):
        cur += timedelta(days=1)
        ds = cur.strftime("%Y-%m-%d")
        if tc.is_trading_day(ds):
            return f"{ds} 15:35"
    return None


def backfill_analyses(limit: int = 100, include_failed: bool = False,
                      price_fetcher=None, use_lock: bool = True,
                      review_client=None) -> dict:
    """回填补 T+N 收益（调度每日收盘后调用 / UI 手动触发）。

    Args:
        limit: 单次最多处理行数（防止积压时一次拉太多 K 线触发限频）
        include_failed: True 时连 failed 行一起重试（手动按钮专用；
            failed 正常只能由超龄判定产生，每日任务不重试）
        price_fetcher: 可注入的 K 线获取函数 (code, start, end) -> DataFrame（测试用）
        use_lock: 跨进程互斥（默认开）。单测并发执行时需关掉，否则用例会互相抢锁。
        review_client: 里程碑 LLM 归因的客户端（测试注入桩用；None ⇒ 真客户端，
            且仅在 `analysis_review.llm_autorun_enabled()` 为真时才会构造/调用）

    Returns: {checked, updated, done, partial, failed, skipped, errors: [...], locked: bool,
              reviewed, llm_ok, llm_error, review_errors: [...]}
    """
    # 跨进程互斥：调度 15:35 与 UI 手动按钮是不同进程，撞车会重复拉同一批 K 线
    holder = f"pid{os.getpid()}-{int(time.time())}"
    if use_lock and not store.acquire_task_lock("backfill_analysis", holder):
        _logger.info("[分析历史] 回填任务已在其他进程运行中，本次跳过（避免重复拉 K 线）")
        return {"checked": 0, "updated": 0, "done": 0, "partial": 0,
                "failed": 0, "skipped": 0, "errors": [], "locked": True}

    fetcher = price_fetcher or default_price_fetcher
    rows = store.list_analyses(limit=limit, only_backfillable=True)
    if include_failed:
        rows += store.list_analyses(limit=limit, backfill_status="failed")

    # ⚠️ `skipped` 拆成两桶（2026-10-09）：原先两类完全不同的原因（"窗口还没走完"与
    #    "取数失败"）在日志里**长得一模一样**（都只是 skipped +1），于是"一次回填都没有"
    #    根本无法判断是**正常的等待**还是**真的坏了**。本次排查就是被这一点卡住的：
    #    必须手工复现取数才能区分。`skipped` 保留为两桶之和（向后兼容日志与既有测试）。
    stats = {"checked": len(rows), "updated": 0, "done": 0, "partial": 0,
             "failed": 0, "skipped": 0, "not_ready": 0, "unavailable": 0,
             "reviewed": 0, "llm_ok": 0, "llm_error": 0, "review_errors": [],
             "errors": [], "locked": False}
    try:
        _backfill_rows(rows, fetcher, stats, review_client=review_client)
    finally:
        if use_lock:
            store.release_task_lock("backfill_analysis", holder)
    return stats


def _backfill_rows(rows: list, fetcher, stats: dict,
                   review_client=None) -> None:
    """逐行回填（从 backfill_analyses 拆出，便于加锁/解锁包住整批）"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today = datetime.now().strftime("%Y-%m-%d")

    for row in rows:
        ana_id = row["id"]
        ana_date = row.get("analysis_date") or ""
        code = row.get("stock_code") or ""
        if not ana_date or not code:
            store.update_analysis_backfill(ana_id, {"backfill_status": "failed",
                                                    "backfilled_at": now})
            stats["failed"] += 1
            continue
        try:
            age_days = (datetime.now() - datetime.strptime(ana_date, "%Y-%m-%d")).days
        except ValueError:
            store.update_analysis_backfill(ana_id, {"backfill_status": "failed",
                                                    "backfilled_at": now})
            stats["failed"] += 1
            continue

        # 超龄：T+20 窗口已不可能补齐，标 failed 终止每日重试
        if age_days > BACKFILL_MAX_AGE_DAYS:
            store.update_analysis_backfill(ana_id, {"backfill_status": "failed",
                                                    "backfilled_at": now})
            stats["failed"] += 1
            continue

        # 拉分析日前 35 天至今的 K 线。
        # 前缓冲从 7 天放宽到 35 天（2026-10-29）：里程碑复盘要算 **ATR14**
        # （`_atr14_before` 需要分析日之前 ≥15 根），7 个自然日只有 ~5 根，
        # ATR 会按 5 个 TR 硬算出来 —— 宽度系统性偏窄，"止损几个 ATR"就失真。
        # 35 个自然日 ≈ 23 个交易日，扣掉春节/国庆长假也够 15 根。
        start = (datetime.strptime(ana_date, "%Y-%m-%d")
                 - timedelta(days=35)).strftime("%Y-%m-%d")
        try:
            df = normalize_kline(fetcher(code, start, today))
        except Exception as e:                  # noqa: BLE001
            # 数据源失败（限频/熔断/网络）：保持原状态，明日任务自然重试
            stats["errors"].append(f"{code}: {e}")
            stats["skipped"] += 1
            continue
        if df is None:
            # ⚠️ 这里原先**完全静默**（不写 errors、不打日志）——取数为空与"窗口未走完"
            #    都只表现为 skipped +1，排查时无法区分。现在明确记为 unavailable 并留痕。
            stats["skipped"] += 1
            stats["unavailable"] += 1
            stats["errors"].append(
                f"{code}: 取数为空或无法归一（{start}~{today}）—— 该行保持原状态，下次任务重试")
            continue

        # 取完整 trade_plan（列表查询不解析 JSON）
        full = store.get_analysis(ana_id) or {}
        fields = _compute_backfill(df, ana_date, full.get("trade_plan") or {})
        if not fields:
            # 正常情况：分析日之后还没走完 T+5 所需的最小根数（基准 + 5 根）。
            # 这不是错误，但要**单独计数**，否则它和"取数失败"混在一起看不出区别。
            stats["skipped"] += 1
            stats["not_ready"] += 1
            continue

        # 状态推进：ret_20d 到位 → done；否则有部分数据 → partial
        if "ret_20d" in fields:
            fields.update({"backfill_status": "done", "backfilled_at": now})
            stats["done"] += 1
        else:
            fields.update({"backfill_status": "partial", "backfilled_at": now})
            stats["partial"] += 1
        store.update_analysis_backfill(ana_id, fields)
        stats["updated"] += 1

        # ── 复盘：只在**里程碑**做（2026-10-29 重构）──
        # 里程碑 = ①第一次写进 T+5（此前该行没有任何收益）；②T+20 到位（partial→done）。
        # 里程碑上会做**完整复盘**：机械归因（含 ATR14 / 基准超额）+ 可选 LLM 总结
        # （`analysis_review.llm_autorun_enabled()` 默认开，即"T+5、T+20 自动调 LLM"）。
        #
        # ⚠️⚠️ 非里程碑的日常重跑**绝不重写复盘**。旧实现每天对 partial 行做一次
        #    机械 upsert，而 `upsert_analysis_review` 会把 `llm_review` 覆盖成 NULL
        #    ⇒ 里程碑时写好的 LLM 归因**第二天就被抹掉**（T+5 写、次日没、T+20 才再写），
        #    自动 LLM 等于没做。代价是 partial 行的 facts 不再逐日变长——
        #    换来的是 LLM 归因能活到 T+20 覆盖那天，这笔账划算。
        milestone = None
        if "ret_20d" in fields:
            milestone = 20
        elif "ret_5d" in fields and row.get("ret_5d") is None:
            milestone = 5

        if milestone and fields.get("backfill_status") in ("done", "partial"):
            try:
                from core import analysis_review as _review
                # ATR14 直接用本次回填的 df（前缓冲 35 天已够 14 根 TR）；
                # 基准收益只依赖本地指数 parquet（`index_bars` 不联网）。
                atr = _review._atr14_before(df, ana_date)
                span = _review.WINDOW_DAYS if milestone >= 20 else milestone
                bench = None
                try:
                    ends = df[df["date"] >= ana_date].iloc[1:1 + span]
                    if not ends.empty:
                        bench = _review._bench_return(
                            str(ends.iloc[0]["date"])[:10],
                            str(ends.iloc[-1]["date"])[:10])
                except Exception as e:              # noqa: BLE001
                    _logger.debug("[分析历史] 基准收益取不到（%s，不影响复盘）：%s",
                                  code, e)
                mech = _review.build_mechanical_review(
                    full, df, atr14=atr, bench_ret=bench, horizon=milestone)
                if mech:
                    if _review.llm_autorun_enabled():
                        llm_status = _review.enrich_with_llm(
                            mech, full, client=review_client)
                        if llm_status == "llm_error":
                            stats["llm_error"] += 1
                        else:
                            stats["llm_ok"] += 1
                    if store.upsert_analysis_review(mech):
                        stats["reviewed"] += 1
            except Exception as e:                  # noqa: BLE001
                # 复盘失败**不能**影响回填结果：回填已经 commit 了，这里只留痕
                stats["review_errors"].append(f"{code}: {e}")
                _logger.warning("[分析历史] 复盘失败（不影响回填，%s）：%s", code, e)

    if stats["checked"]:
        # 日志里把两桶拆开写清楚：not_ready 是"正常等待"，unavailable 才是"要看数据源"。
        _logger.info("[分析历史] T+N 回填: %s", stats)
        if stats.get("not_ready"):
            _logger.info("[分析历史] 其中 %s 条属于**窗口未走完**（不是故障）："
                         "T+5 需 6 根 K 线（基准+5），分析日之后还不够。"
                         "想看具体还差几天可用 `analysis_history.backfill_progress()`。",
                         stats["not_ready"])
        if stats.get("unavailable"):
            _logger.warning("[分析历史] 其中 %s 条属于**取数不可用**（需查数据源）：%s",
                            stats["unavailable"], stats["errors"][:3])


# ================================================================
# 每日触发核对（2026-10-10）
# ================================================================

def check_triggers_daily(limit: int = 200, price_fetcher=None,
                         use_lock: bool = True, today: str = None,
                         max_age_days: int = TRIGGER_CHECK_MAX_AGE_DAYS) -> dict:
    """每日盘后核对：目标价 / 止损价是否**已经被触及**（不等 T+5 窗口）。

    为什么需要它（2026-10-10 诊断）：原先 `hit_target` / `hit_stop` **只在 T+N 回填时**
    才判定一次，最快也要 T+5 —— 意味着"今天刚跌破止损"要等一周后报告才说，
    而用户要的是"明日的操作建议"，那必须在**当天收盘后**就知道。
    本次把窗口从"T+20 固定段"提前到"从分析日**至今**"，因此结果**单调**：
    一旦触及即为 1，之后保持；到 T+20 时与回填的口径完全一致（同一个 `evaluate_triggers`）。

    幂等：只写"尚未触及 → 已触及"的字段，重复调用结果不变。
    范围：只处理最近 `max_age_days` 天、且 `backfill_status` 非 done/skip 的行
    （已完成的样本由回填写定，无需每日重复拉 K 线）。

    Returns: {checked, updated, already_hit, no_plan, not_ready, unavailable,
              errors: [...], locked}
    """
    holder = f"pid{os.getpid()}-{int(time.time())}"
    if use_lock and not store.acquire_task_lock("trigger_check", holder):
        _logger.info("[触发核对] 已在其他进程运行，本次跳过")
        return {"checked": 0, "updated": 0, "already_hit": 0, "no_plan": 0,
                "not_ready": 0, "unavailable": 0, "errors": [], "locked": True}

    fetcher = price_fetcher or default_price_fetcher
    today = (today or datetime.now().strftime("%Y-%m-%d"))[:10]
    cutoff = (datetime.now() - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
    stats = {"checked": 0, "updated": 0, "already_hit": 0, "no_plan": 0,
             "not_ready": 0, "unavailable": 0, "errors": [], "locked": False}
    try:
        rows = store.list_analyses_since(cutoff, limit=limit)
        for row in rows:
            if (row.get("backfill_status") or "") in ("done", "skip"):
                continue
            plan = row.get("trade_plan") or {}
            target = _parse_price(plan.get("target_price"))
            stop = _parse_price(plan.get("stop_loss"))
            if target is None and stop is None:
                stats["no_plan"] += 1
                continue
            ana_date = str(row.get("analysis_date") or "")[:10]
            code = row.get("stock_code") or ""
            if not ana_date or not code:
                continue
            stats["checked"] += 1
            start = (datetime.strptime(ana_date, "%Y-%m-%d")
                     - timedelta(days=7)).strftime("%Y-%m-%d")
            try:
                df = normalize_kline(fetcher(code, start, today))
            except Exception as e:              # noqa: BLE001
                stats["errors"].append(f"{code}: {e}")
                stats["unavailable"] += 1
                continue
            if df is None:
                stats["unavailable"] += 1
                continue
            window = df[df["date"] >= ana_date].iloc[1:21]
            if window.empty:
                stats["not_ready"] += 1
                continue
            highs = [float(h) for h in window["high"]] if "high" in window else []
            lows = [float(l) for l in window["low"]] if "low" in window else []
            res = evaluate_triggers(plan.get("direction"), target, stop, highs, lows)
            fields = {k: 1 for k in ("hit_target", "hit_stop")
                      if res.get(k) == 1 and not row.get(k)}
            if fields:
                store.update_analysis_backfill(row["id"], fields)
                stats["updated"] += 1
            else:
                stats["already_hit"] += 1
    finally:
        if use_lock:
            store.release_task_lock("trigger_check", holder)
    if stats["checked"]:
        _logger.info("[触发核对] 每日目标/止损核对: %s", stats)
    return stats
