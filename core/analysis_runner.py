# -*- coding: utf-8 -*-
"""深度分析 · 无头执行器（单一实现，2026-10-15）。

为什么需要：
  · 「跑一次多 Agent 深度分析」的流程原本写在 UI 页面里（`_analysis_job`）；
  · AI 交易联赛要在**调度线程**里同步调用同一套流程 —— 复制一份必然分叉
    （本项目有明训："唯一实现"，两处各写一份迟早漂移）。
  ⇒ 把「建 initial_state → stream → 按 reducer 语义归并 → 落库 → 返回快照」抽到这里，
    页面与 AI 联赛**共用**。

⚠️ 本模块**不得依赖 streamlit**（决策轮跑在调度线程里，没有 ScriptRunContext）。
⚠️ `merge_delta` 的 reducer 语义必须保留：`errors`/`fatal_errors` 是**只追加**、
   `data_availability` 是 **dict 合并** —— 裸 `dict.update()` 会覆盖掉前面节点写的内容
   （快照里就只剩最后一个节点的错误）。
"""
from __future__ import annotations

import logging
from datetime import datetime

from db import store

_logger = logging.getLogger(__name__)

# 图里固定的节点步数（多轮辩论会增加步数，这里只用于进度百分比估算）
NODE_STEPS = 15


def build_initial_state(code: str) -> dict:
    """多 Agent 图的初始 state（与 UI 页面**同一份**形状）。"""
    return {
        "stock_code": code, "stock_name": "", "analysis_date": "",
        "stock_info": {}, "kline_data": None, "indicators": {}, "fund_flow_data": {},
        "data_through": "", "data_availability": {},
        "tech_report": "", "fundamental_report": "", "news_report": "",
        "bull_argument": "", "bear_argument": "", "debate_count": 0,
        "max_debate_rounds": 2, "bull_history": [], "bear_history": [],
        "research_summary": "", "trade_plan": {}, "risk_assessment": "",
        "final_decision": "",
        "screener_score": 0.0, "screener_factors": {}, "screener_sector": "",
        "messages": [], "errors": [],
    }


def run_analysis(code: str, *, source: str = "ai", on_step=None) -> dict:
    """同步跑一次完整深度分析并落库。

    Args:
        source: 落库来源标记（"ui" / "ai" / "pipeline"）。
        on_step: 可选回调 `(node_name, final_state, step_idx)`，供调用方报进度
                 （UI 用它写进度文件；AI 联赛不传）。
    Returns:
        `{"ok", "snapshot", "analysis_id", "error", "steps"}`
        —— 失败时**已落一条"分析失败"记录**（`save_failure`），`ok=False` 且带 error。
    """
    from core.analysis_history import save_analysis, save_failure, snapshot_from_state
    from core.graph_builder import analysis_graph
    from core.state import merge_delta

    initial = build_initial_state(code)
    final_state = dict(initial)
    steps = 0
    try:
        for event in analysis_graph.stream(initial):
            steps += 1
            node_name = list(event.keys())[0]
            node_data = event[node_name] or {}
            final_state.update(merge_delta(final_state, node_data))
            if on_step is not None:
                try:
                    on_step(node_name, final_state, steps)
                except Exception as e:                       # noqa: BLE001
                    # 报进度失败不该中断分析（best-effort）
                    _logger.warning("[分析执行器] 进度回调失败: %s", e)
    except Exception as e:                                   # noqa: BLE001
        save_failure(code, str(e), source=source,
                     stock_name=final_state.get("stock_name", ""))
        _logger.exception("[分析执行器] %s 分析异常: %s", code, e)
        return {"ok": False, "snapshot": {},
                "error": f"{type(e).__name__}: {e}", "steps": steps}

    snap = snapshot_from_state(final_state)
    analysis_id = 0
    try:
        analysis_id = int(save_analysis(final_state, source=source) or 0)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[分析执行器] %s 落库失败（结果仍返回）: %s", code, e)
    return {"ok": True, "snapshot": snap, "analysis_id": analysis_id,
            "error": "", "steps": steps}


# ---------------- 给 AI 看的压缩摘要 ----------------

def summarize_for_ai(snapshot: dict, *, max_text: int = 700) -> dict:
    """把完整快照压成 AI 可读的**结构化摘要**。

    ⚠️ 纪律（与"报告问答"同源）：
      · **必须同时带上该报告自身的 `errors` / `audit_warnings`** —— 否则模型会把
        幻觉当真结论转述给下游；
      · **必须带 `data_availability`**（哪些输入没取到）—— "缺就是缺"，不许让模型
        把缺失当"没有"或"没变化"；
      · 各段报告**截断**（单段 max_text 字符），避免把整份快照塞进上下文。
    """
    def _cut(s) -> str:
        s = str(s or "")
        return s if len(s) <= max_text else s[:max_text] + "…（已截断）"

    plan = snapshot.get("trade_plan") or {}
    missing = sorted(k for k, v in (snapshot.get("data_availability") or {}).items()
                     if not v)
    # ⚠️ 顺序即优先级：回灌提示词时结果会被**按字符截断**（`_shrink`），
    #    所以"必须让模型看到"的字段（口径/缺失/错误/审计告警）放**最前**——
    #    否则截断掉的恰好是最关键的几条（这正是"报告问答必须带 errors"的反面教材）。
    return {
        "errors": list(snapshot.get("errors") or []),              # ⚠️ 必须带
        "audit_warnings": list(snapshot.get("audit_warnings") or []),
        "data_missing": missing,                    # 取不到的输入（如实列出）
        "analysis_date": snapshot.get("analysis_date"),
        "data_through": snapshot.get("data_through"),
        "stock_code": snapshot.get("stock_code"),
        "stock_name": snapshot.get("stock_name"),
        "current_price": snapshot.get("current_price"),
        "trade_plan": plan,
        "final_decision": _cut(snapshot.get("final_decision")),
        "research_summary": _cut(snapshot.get("research_summary")),
        "tech_report": _cut(snapshot.get("tech_report")),
        "fundamental_report": _cut(snapshot.get("fundamental_report")),
        "news_report": _cut(snapshot.get("news_report")),
        "risk_assessment": _cut(snapshot.get("risk_assessment")),
    }


def ai_deep_analysis_runner(*, source: str = "ai", trade_date: str = None):
    """给 `run_daily_round(analysis_runner=...)` 用的执行器工厂。

    返回 `runner(code) -> dict`，行为：
      1. **同日同票复用**：先查当日已有分析记录 —— 命中就直接返回（**不再烧 ~4.7 万 token**）。
         ⚠️ 这不是"优化"，是**成本纪律**：LLM 可能在同一轮里对同一只票重复请求，
         而额度是按次扣的，重复跑=白花钱（UI 侧也有同样的同日去重理由）。
      2. 未命中才真跑，落库 `analyses(source="ai")`。
      3. 一律返回 `summarize_for_ai()` 的**压缩摘要**（不是整份快照）。

    ⚠️ 额度不在这里扣：`ai_daily_round.execute_tool` 已**先扣后跑**（并发不超额）。
       这里只负责"跑或复用"，不重复记账。
    """
    day = trade_date or datetime.now().strftime("%Y-%m-%d")

    def runner(code: str) -> dict:
        code = str(code or "").strip()
        if not code:
            return {"ok": False, "error": "缺少股票代码"}
        # ① 同日同票复用
        try:
            rows = store.list_analyses(stock_code=code, limit=20)
            for r in rows:
                if str(r.get("analysis_date") or "")[:10] != day:
                    continue
                if str(r.get("decision_label") or "") == "分析失败":
                    continue                                  # 失败记录不复用
                full = store.get_analysis(r["id"]) or {}
                payload = full.get("payload") or {}
                if payload:
                    return {"ok": True, "reused": True, "analysis_id": r["id"],
                            "note": "复用当日已有分析（未重复调用模型）",
                            "summary": summarize_for_ai(payload)}
        except Exception as e:                                # noqa: BLE001
            _logger.warning("[AI深度分析] 复用查询失败（改为真跑）: %s", e)

        # ② 真跑
        res = run_analysis(code, source=source)
        if not res["ok"]:
            # ⚠️ 失败要**如实说**（含已落"分析失败"记录的事实），让 AI 知道"这次没拿到"，
            #    而不是让它以为拿到了空报告。
            return {"ok": False, "analysis_id": res.get("analysis_id") or 0,
                    "error": f"分析失败（已记录）：{res['error']}"}
        return {"ok": True, "reused": False,
                "analysis_id": res.get("analysis_id") or 0,
                "summary": summarize_for_ai(res["snapshot"])}

    return runner
