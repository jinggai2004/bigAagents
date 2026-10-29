"""深度分析复盘：把"回填了数值"变成"知道为什么不准、该改什么"。

为什么需要它
────────────
`analyses` 回填只算出 ret_5d / ret_20d / hit_target / hit_stop。这些数字能回答
"这次对不对"，但**回答不了"为什么错"**——下一批分析于是只能靠同一个人凭同样的
直觉再猜一次，错法可以完全重复而无人察觉。

三层设计（可核实性由强到弱）
──────────────────────────
L1 机械归因（本模块主力，**零 LLM**）
    从 K 线把**客观路径**算出来：先触止损还是先触目标、扫损后有没有反弹、
    有没有跑赢基准、止损宽度相当于几个 ATR；再按**代码判据**推导候选原因码。
L2 LLM 归因（可选，`review_one(with_llm=True)`）
    只给「分析当时的报告 + L1 的事实 + 候选原因码」，让它在**候选集内**选主因。
L3 聚合（`aggregate`）
    单条归因是噪声；按原因码计数、按改进项去重计数，才是"该改什么"。

⚠️⚠️ 一条铁律：**"对不对"由代码判定，"为什么"由模型在候选集内选择**。
   反过来（让模型判定对错）会让"我错了"出现在本来正确的样本上。

⚠️⚠️ 绝不喂 analysis_date 之后的新闻/公告。后见之明偏差（hindsight bias）会让
   模型编出"当时已有征兆"的叙事，而这类叙事**无法证伪**——比不复盘更危险。
   所以本模块的 LLM 输入里，事实段只有**价格路径**，没有任何事后文本。
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta

from db import store

_logger = logging.getLogger(__name__)

# 观察窗口（交易日）：与 `analysis_history._compute_backfill` 的 `iloc[1:21]` 一致。
# ⚠️ 两处必须同口径，否则"回填说触及了、复盘说没触及"会互相矛盾。
WINDOW_DAYS = 20

VERDICTS = ("correct", "partially_correct", "wrong",
            "unclear", "opportunity_missed", "no_position")

# 方向判定阈值：±3% 以内视为"没走出来"，不判对错（噪声不该被当成结论）。
MOVE_THRESHOLD = 0.03
# "方向对了但幅度很小"的分界：达到 +2% 才算真的赚到
SMALL_GAIN = 0.02

# ------------------------------------------------------------------
# 原因码：**唯一来源**。LLM 只能从这里选，不能新增。
# ------------------------------------------------------------------
CAUSE_CODES: dict[str, str] = {
    "direction_wrong":      "方向判错：看多却明显下跌（或看空却明显上涨）",
    "market_beta":          "大盘拖累：绝对收益为负，但跑赢了同期指数（不是选股问题）",
    "sector_rotation":      "行业轮动：同行业普跌，个股并未独立走弱",
    "timing_wrong":         "时点错：方向最终对了，但 T+5 与 T+20 符号相反（太早/太晚）",
    "stop_too_tight":       "止损过近：先被止损扫出，之后价格又回到成本之上（被噪声打掉）",
    "stop_too_loose":       "止损过远：没触发止损却亏超 8%（风控没有真正生效）",
    "target_unrealistic":   "目标不现实：窗口内最大涨幅不到目标幅度的一半",
    "no_stop":              "未设止损：交易计划里没有可执行的止损位",
    "overconfidence":       "过度自信：给了高仓位（≥20%）却判错方向",
    "data_missing_decision": "带缺数据下强结论：输入有明确缺失，仍给出看多且非零仓位",
    "hallucinated_evidence": "证据不实：可核实性审计报了幻觉线索，且结果确实错",
    "opportunity_missed":   "错失机会：观望/驳回之后大涨（不进场也是有成本的）",
    "nothing_wrong":        "没有发现可归因的失误（判断成立或本来就不该下结论）",
    "unknown":              "无法归因（数据不足或超出上述所有判据）",
}

# 改进项只能指向这些可改的东西 —— 防止"下次更谨慎一些"这种不可执行的建议。
IMPROVE_TARGETS: dict[str, str] = {
    "stop_loss_rule":   "止损价规则（宽度 / ATR 倍数 / 是否设在结构位）",
    "target_price_rule": "目标价规则（是否超过近期压力位 / 涨幅是否现实）",
    "position_rule":    "仓位规则（单票上限 / 与置信度的关系）",
    "entry_timing":     "入场时点（追高 / 等回调 / T+1 开盘 vs 收盘）",
    "data_source":      "数据源（缺失项补采 / 口径核对）",
    "prompt":           "提示词（输入接地 / 缺失处理 / 约束）",
    "screening":        "选股条件（把这类票排除在候选池外）",
    "none":             "无需改动",
}


# ================================================================
# L1 机械归因：客观路径 + 候选原因
# ================================================================

def _parse_price(value) -> float | None:
    """从 trader 输出的价位字符串（'18.5元' / '18.5-19.0'）解析首个数字。

    与 `analysis_history._parse_price` 同实现——**故意重复 6 行而不跨模块引用**：
    本模块要能独立跑（复盘不依赖回填），而回填那边一旦改解析口径，这里会立刻
    在测试里暴露不一致，而不是静默跟随。
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(m.group()) if m else None


def _col(df, name):
    """取列并转 float 列表；缺列返回 None（"没有"与"全 0"必须分开）。"""
    if df is None or name not in getattr(df, "columns", []):
        return None
    try:
        return [float(v) for v in df[name].tolist()]
    except (TypeError, ValueError) as e:
        _logger.debug("[复盘] 列 %s 不是纯数值：%s", name, e)
        return None


def path_facts(df, analysis_date: str, trade_plan: dict, label: str,
               atr14: float | None = None,
               bench_ret: float | None = None) -> dict:
    """算出这次分析的**客观价格路径**（全部由代码得出，不含任何判断）。

    Args:
        df: 归一化 K 线（`date` 为 'YYYY-MM-DD' 字符串、升序），需覆盖分析日之后
            至少 1 根，最好 21 根
        trade_plan: 当时的交易计划（取 target_price / stop_loss / position_size）
        label: decision_label（看多/看空/观望/驳回/分析失败）
        atr14: 分析日**之前**的 ATR14（由调用方从本地面板取；拿不到就 None）
        bench_ret: 同期基准收益（如沪深300 的 T+20），拿不到就 None

    Returns:
        dict；`window_days == 0` 表示分析日之后还没有 K 线（此时只有 base_* 字段）。
        ⚠️ 所有"没算出来"的键**缺席或为 None**，绝不填 0——"没触及止损"与
        "没算"在归因里是两个完全不同的结论。
    """
    out: dict = {"analysis_date": str(analysis_date or "")[:10],
                 "decision_label": label or "", "window_days": 0}
    if df is None or getattr(df, "empty", True):
        return out
    try:
        rows = df[df["date"] >= out["analysis_date"]]
    except (KeyError, TypeError) as e:
        _logger.info("[复盘] K 线缺 date 列或类型异常（%s）：%s", out["analysis_date"], e)
        return out
    if rows is None or rows.empty:
        return out

    base = rows.iloc[0]
    try:
        base_price = float(base["close"])
    except (KeyError, TypeError, ValueError) as e:
        _logger.info("[复盘] 基准日收盘价不可用：%s", e)
        return out
    out.update({"base_date": str(base["date"])[:10], "base_price": base_price})

    window = rows.iloc[1:1 + WINDOW_DAYS]
    out["window_days"] = int(len(window))
    if window.empty or base_price <= 0:
        return out

    closes = _col(window, "close") or []
    highs = _col(window, "high")
    lows = _col(window, "low")
    # high/low 缺列时退化为 close（方向仍可算，但"日内触及"会漏判 ⇒ 标记 degraded）
    used_close_only = False
    if highs is None or lows is None:
        highs, lows, used_close_only = closes, closes, True
    out["high_low_degraded"] = used_close_only
    if not (closes and highs and lows):
        return out

    bearish = "看空" in str(label or "")
    # mfe/mae 统一为**正数幅度**：mfe=对持仓最有利走了多少，mae=最不利走了多少。
    # 看空的"有利"是下跌，符号处理集中在这一处，后面所有判据都不再关心方向。
    if bearish:
        mfe = 1.0 - min(lows) / base_price
        mae = max(highs) / base_price - 1.0
    else:
        mfe = max(highs) / base_price - 1.0
        mae = 1.0 - min(lows) / base_price
    out.update({"mfe_pct": round(mfe, 4), "mae_pct": round(mae, 4),
                "max_high": round(max(highs), 3), "min_low": round(min(lows), 3),
                "last_close": round(closes[-1], 3)})

    # ── T+5 / T+20（与 forward_returns 同口径：基准之后第 N 根收盘）──
    for n in (5, 20):
        if len(closes) >= n:
            out[f"ret_{n}d"] = round(closes[n - 1] / base_price - 1.0, 4)

    # ── 目标/止损：先触哪个、之后有没有回来（归因的关键证据）──
    plan = trade_plan or {}
    target = _parse_price(plan.get("target_price"))
    stop = _parse_price(plan.get("stop_loss"))
    out.update({"target_price": target, "stop_loss": stop})
    if base_price:
        if target is not None:
            out["target_dist_pct"] = round(
                (target / base_price - 1.0) * (1 if not bearish else -1), 4)
        if stop is not None:
            out["stop_dist_pct"] = round(
                (1.0 - stop / base_price) * (1 if not bearish else -1), 4)

    first_target = first_stop = None
    for i in range(len(closes)):
        h, l = highs[i], lows[i]
        # 同一天两者都触及 ⇒ 保守记"先止损"（无法从日线判断盘中先后）
        if stop is not None and first_stop is None:
            if (h >= stop) if bearish else (l <= stop):
                first_stop = i + 1
        if target is not None and first_target is None:
            if (l <= target) if bearish else (h >= target):
                first_target = i + 1
        if first_stop is not None and first_target is not None:
            break
    out.update({"days_to_target": first_target, "days_to_stop": first_stop,
                "hit_target": (1 if first_target else 0) if target is not None else None,
                "hit_stop": (1 if first_stop else 0) if stop is not None else None})

    # 扫损之后有没有回来：这是"止损过近"唯一的硬证据，必须逐根看后续走势
    if first_stop:
        after_h = highs[first_stop - 1:]
        after_l = lows[first_stop - 1:]
        back = (min(after_l) <= base_price) if bearish else (max(after_h) >= base_price)
        out["stop_then_recover"] = bool(back)
        if target is not None:
            reach = (min(after_l) <= target) if bearish else (max(after_h) >= target)
            out["stop_then_target"] = bool(reach)
    if first_target:
        out["target_then_stop"] = bool(first_stop and first_stop > first_target)

    # ── 止损宽度（ATR 倍数）：最有用的可改参数 ──
    if atr14 and atr14 > 0 and stop is not None:
        out["atr14"] = round(float(atr14), 3)
        out["stop_dist_atr"] = round(abs(base_price - stop) / float(atr14), 2)

    # ── 基准（超额才能区分"选股错"与"大盘拖累"）──
    if bench_ret is not None:
        out["bench_ret_20d"] = round(float(bench_ret), 4)
        if "ret_20d" in out:
            out["excess_20d"] = round(out["ret_20d"] - float(bench_ret), 4)
    return out


def mechanical_verdict(facts: dict, label: str, horizon: int = 20) -> str:
    """"对不对"**只由代码判定**（不看模型的话）。

    `horizon`：20=用 T+20 收益（完整窗口）；5=用 T+5 收益（早期结论）。
    ⚠️ 两个窗口的结论**不可混在一起统计**——「T+5 判对、T+20 判错」恰恰是
    `timing_wrong` 的证据，混口径就看不见它了。

    观望/驳回/分析失败单独归为 no_position —— 它们不预测方向，无从谈对错；
    但"观望之后暴涨"会单独标 opportunity_missed（错失机会也是成本，只是不该
    和方向错误混在一个胜率里）。
    """
    label = str(label or "")
    ret = facts.get("ret_20d") if horizon >= 20 else facts.get("ret_5d")
    # ⚠️ 白名单，不用排除法（审查 2026-10-20）：原写法是
    #    `if label in ("驳回","分析失败","") or "观望" in label` + `bull = "看空" not in label`。
    #    那意味着**任何新标签**（"暂缓"/"看不清"）都会漏过这道 guard，并被当成看空/看多，
    #    一有波动就判"方向错误"。这里改成"只认明确的方向词"。
    is_bull = label.startswith("看多")
    is_bear = label.startswith("看空")
    if not (is_bull or is_bear):
        if "观望" in label and ret is not None and ret > 0.10:
            return "opportunity_missed"
        return "no_position"
    if ret is None:
        return "unclear"
    signed = ret if is_bull else -ret      # 转成"对这次判断有利"的方向
    if signed <= -MOVE_THRESHOLD:
        return "wrong"
    if signed >= MOVE_THRESHOLD or (facts.get("hit_target") == 1):
        return "correct"
    if signed >= SMALL_GAIN:
        return "partially_correct"
    return "unclear"


def candidate_causes(facts: dict, label: str, trade_plan: dict,
                     payload: dict | None = None, horizon: int = 20) -> list[dict]:
    """按**代码判据**推导候选原因码；每个都带 `evidence`（实测值）。

    设计要点：模型/人只能从这里选。判据不成立的原因**不进候选**——这样
    "止损过近"不会出现在"根本没设止损"的样本上，归因分布才可信。

    `horizon`=5 时只用 5 日收益，"时点错"（5 日与 20 日符号相反）这条判据自动缺席
    —— 它需要两个窗口才能算。
    """
    plan = trade_plan or {}
    payload = payload or {}
    label = str(label or "")
    # 白名单（审查 2026-10-20）：只认明确的方向词，其余一律按"不预测方向"处理。
    # 这样将来新增标签（"暂缓"/"看不清"）会安全落到 no_position，
    # 而不是被排除法当成看多、一跌就判错。
    is_bull = label.startswith("看多")
    is_bear = label.startswith("看空")

    def signed(v):
        """把收益转成"对本次判断有利"的方向（看空则取反）"""
        return None if v is None else (v if is_bull else -v)

    full = horizon >= 20
    r20 = facts.get("ret_20d") if full else facts.get("ret_5d")
    r5 = facts.get("ret_5d") if full else None
    ex = facts.get("excess_20d")
    out: list[dict] = []
    s20 = signed(r20)

    # ── 「不预测方向」的样本走单独分支（2026-10-20 真实数据暴露）──
    # ⚠️ 第一版对观望样本也跑 direction_wrong：实测 `002141 观望 5 日 -6.12%`
    #    被判成"方向判错"，读起来极其荒谬（观望根本没有方向）。它的失误只可能是
    #    "该进没进"（错失机会），不可能是"方向看反"。
    #    路径：`bull = "看空" not in label` ⇒ 观望被当成看多 ⇒ 一跌就判错。
    # ⚠️ 判据只能看 **label**，不能看 `plan.get("direction")` 是否为空：
    #    plan 缺失 ≠ 没有方向（调用方可能只传了 label）。第一版写成
    #    `not plan.get("direction")` 时，所有"空 plan"的样本全被当成观望，
    #    六条判据同时失效——测试立刻抓到（这是"用缺失当信号"的老坑）。
    directionless = not (is_bull or is_bear)
    if directionless:
        if r20 is not None and r20 > 0.10:
            out.append({"code": "opportunity_missed",
                        "evidence": {"ret": r20, "threshold": 0.10}})
        else:
            out.append({"code": "nothing_wrong",
                        "evidence": {"ret": r20, "note": "未预测方向"}})
        return out

    # 1 方向判错
    if s20 is not None and s20 <= -MOVE_THRESHOLD:
        out.append({"code": "direction_wrong",
                    "evidence": {"ret_20d": r20, "threshold": -MOVE_THRESHOLD}})

    # 2 大盘拖累：绝对亏、但跑赢基准 >1%
    if r20 is not None and r20 < 0 and ex is not None and ex > 0.01:
        out.append({"code": "market_beta",
                    "evidence": {"ret_20d": r20, "bench_ret_20d": facts.get("bench_ret_20d"),
                                 "excess_20d": ex}})

    # 3 时点错：5 日与 20 日符号相反，且 20 日幅度显著
    if r5 is not None and r20 is not None and r5 * r20 < 0 and abs(r20) >= SMALL_GAIN:
        out.append({"code": "timing_wrong",
                    "evidence": {"ret_5d": r5, "ret_20d": r20}})

    # 4 止损过近：**先**触发止损，之后又回到成本价之上（被噪声打掉）
    if facts.get("days_to_stop") and facts.get("stop_then_recover"):
        out.append({"code": "stop_too_tight",
                    "evidence": {"days_to_stop": facts.get("days_to_stop"),
                                 "stop_dist_atr": facts.get("stop_dist_atr"),
                                 "stop_then_target": facts.get("stop_then_target")}})

    # 5 止损过远/失效：没触发止损却亏超 8%
    if (facts.get("hit_stop") == 0 and s20 is not None and s20 <= -0.08):
        out.append({"code": "stop_too_loose",
                    "evidence": {"hit_stop": 0, "ret_20d": r20,
                                 "stop_dist_pct": facts.get("stop_dist_pct")}})

    # 6 目标不现实：没触目标，且最大有利幅度不到目标幅度的一半
    # ⚠️ 只在**完整窗口**才判：目标价是给 T+20 定的，用 5~9 天的数据说"目标不可达"
    #    是错的（实测 002129 只走了 9 天、T+5 已 +2.41%，却被判"目标不现实"）。
    # ⚠️ `window_days` 缺失时按 `horizon` 走（不能因为调用方没传就静默失效）。
    tgt_dist = facts.get("target_dist_pct")
    wd = facts.get("window_days")
    enough_window = full and (wd is None or wd >= 15)
    if (enough_window and facts.get("hit_target") == 0
            and tgt_dist and tgt_dist > 0
            and (facts.get("mfe_pct") or 0) < tgt_dist * 0.5):
        out.append({"code": "target_unrealistic",
                    "evidence": {"target_dist_pct": tgt_dist,
                                 "mfe_pct": facts.get("mfe_pct")}})

    # 7 未设止损且亏损
    if not plan.get("stop_loss") and s20 is not None and s20 <= -0.05:
        out.append({"code": "no_stop",
                    "evidence": {"stop_loss": plan.get("stop_loss"), "ret_20d": r20}})

    # 8 过度自信：仓位 ≥20% 且判错
    pos = plan.get("position_size") or plan.get("position_pct")
    try:
        pos_num = float(str(pos).replace("%", "")) if pos is not None else None
    except (TypeError, ValueError) as e:
        _logger.debug("[复盘] position_size 无法解析为数字（%r）：%s", pos, e)
        pos_num = None
    if pos_num is not None and pos_num >= 20 and s20 is not None and s20 <= -MOVE_THRESHOLD:
        out.append({"code": "overconfidence",
                    "evidence": {"position_size": pos_num, "ret_20d": r20}})

    # 9 带缺数据下强结论
    avail = payload.get("data_availability") or {}
    missing = [k for k, v in avail.items() if v is False or v is None]
    if missing and is_bull and (pos_num or 0) > 0:
        out.append({"code": "data_missing_decision",
                    "evidence": {"missing_fields": missing[:6],
                                 "position_size": pos_num}})

    # 10 幻觉证据 + 结果确实错
    warnings = payload.get("audit_warnings") or []
    if warnings and s20 is not None and s20 < 0:
        out.append({"code": "hallucinated_evidence",
                    "evidence": {"audit_warning_count": len(warnings),
                                 "ret_20d": r20}})

    if not out:
        out.append({"code": "nothing_wrong" if (s20 or 0) >= 0 else "unknown",
                    "evidence": {"ret_20d": r20}})
    return out


def build_mechanical_review(row: dict, df, atr14: float | None = None,
                            bench_ret: float | None = None,
                            horizon: int | None = None) -> dict | None:
    """拼一条完整的**机械复盘**（零 LLM）。数据不足（窗口 0 天）返回 None。

    Args:
        row: `store.get_analysis()` 的返回（含 payload / trade_plan / decision_label）
        horizon: None 时自动判定 —— `ret_20d` 有值算 20，否则算 5（早期结论）
    """
    if not row:
        return None
    label = str(row.get("decision_label") or "")
    if label == "分析失败":
        return None
    if horizon is None:
        horizon = 20 if row.get("ret_20d") is not None else 5
    payload = row.get("payload") or {}
    plan = row.get("trade_plan") or {}
    facts = path_facts(df, row.get("analysis_date") or "", plan, label,
                       atr14=atr14, bench_ret=bench_ret)
    if not facts.get("window_days"):
        return None
    facts["horizon"] = horizon
    verdict = mechanical_verdict(facts, label, horizon=horizon)
    causes = candidate_causes(facts, label, plan, payload, horizon=horizon)
    return {
        "analysis_id": row.get("id"),
        "stock_code": row.get("stock_code") or "",
        "stock_name": row.get("stock_name") or "",
        "analysis_date": str(row.get("analysis_date") or "")[:10],
        "review_date": datetime.now().strftime("%Y-%m-%d"),
        "decision_label": label,
        "model": row.get("model") or "",
        "horizon": horizon,
        "verdict": verdict,
        "primary_cause": causes[0]["code"] if causes else "unknown",
        "secondary_causes": [c["code"] for c in causes[1:3]],
        "facts": facts,
        "candidates": causes,
        "status": "mechanical",
        "llm_review": None,
    }


# ================================================================
# L2 LLM 归因（受限选择 + 证据白名单）
# ================================================================

_SYSTEM = ("你是投研复盘员。任务是**归因**已发生的一次分析，不是重新分析。"
           "只使用给出的【事实】和【当时的报告】；候选原因由代码判定，"
           "你只能选不能新增；禁止用任何事后信息解释。")

_JSON_HINT = """输出 JSON（不要 markdown 代码块）：
{
  "primary_cause": "候选里的一个 code",
  "secondary_causes": ["最多两个 code"],
  "verdict_reason": "一句话说清为什么对/错，必须引用事实里的数字",
  "evidence": [{"field": "事实里的键名", "value": 数字或字符串, "means": "它意味着什么"}],
  "what_was_right": "这次判断里做对的部分（没有就写 none）",
  "improvements": [{"target": "改进项白名单里的 key", "change": "具体改成什么", "why": "依据"}]
}"""


def _truncate(text, n: int = 500) -> str:
    s = str(text or "").strip()
    return s if len(s) <= n else s[:n] + "…（截断）"


def _review_prompt(mech: dict, payload: dict) -> str:
    facts = {k: v for k, v in (mech.get("facts") or {}).items() if v is not None}
    cand = mech.get("candidates") or []
    cand_lines = "\n".join(
        f"{i}. `{c['code']}` —— {CAUSE_CODES.get(c['code'], '')}"
        f"｜实测：{json.dumps(c.get('evidence') or {}, ensure_ascii=False)}"
        for i, c in enumerate(cand, 1))
    plan = payload.get("trade_plan") or {}
    plan_brief = json.dumps({k: plan.get(k) for k in
                             ("direction", "target_price", "stop_loss",
                              "position_size", "entry_condition") if k in plan},
                            ensure_ascii=False)
    return f"""【事实】（代码从 K 线算出，可信；未列出的字段表示"没算出来"）
{json.dumps(facts, ensure_ascii=False, indent=1)}

【当时的报告】（分析日当天产出，不含之后的信息）
- 决策标签：{mech.get('decision_label')}
- 交易计划：{plan_brief or '（无）'}
- 技术面：{_truncate(payload.get('tech_report'))}
- 基本面：{_truncate(payload.get('fundamental_report'))}
- 舆情：{_truncate(payload.get('news_report'))}
- 研究汇总：{_truncate(payload.get('research_summary'))}
- 风控：{_truncate(payload.get('risk_assessment'))}
- 终审：{_truncate(payload.get('final_decision'))}

【候选原因】（代码已按判据筛过，只能从这里选）
{cand_lines}

【改进项白名单】（improvements.target 只能取这些 key）
{json.dumps(IMPROVE_TARGETS, ensure_ascii=False, indent=1)}

{_JSON_HINT}"""


def _extract_json(raw: str) -> dict:
    """从模型回复里取出 JSON 对象（容忍 ```json 包裹与前后废话）。"""
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError as e:
        # 正常路径：模型常带前后说明文字，下面再用正则兜一次
        _logger.debug("[复盘] 整段不是 JSON（%s），尝试正则提取对象", e)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group())
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError as e:
            _logger.warning("[复盘] LLM 回复里提取的对象仍不是合法 JSON：%s", e)
            return {}
    return {}


def validate_llm_review(obj: dict, mech: dict) -> tuple[dict, list[str]]:
    """代码层校验：原因必须在候选集内、证据字段必须真实存在、改进项在白名单内。

    返回 (清洗后的归因, warnings)。**校验失败不抛异常**——降级为机械归因并留痕，
    因为"复盘写歪了"不该让整批回填失败。
    """
    warnings: list[str] = []
    facts = mech.get("facts") or {}
    valid_codes = {c["code"] for c in (mech.get("candidates") or [])}

    primary = str(obj.get("primary_cause") or "").strip()
    if primary not in valid_codes:
        warnings.append(f"primary_cause「{primary[:30]}」不在候选集内 —— "
                        f"降级为机械归因「{mech.get('primary_cause')}」")
        primary = mech.get("primary_cause") or "unknown"

    secondary = []
    for c in (obj.get("secondary_causes") or [])[:3]:
        c = str(c).strip()
        if c in valid_codes and c != primary:
            secondary.append(c)
        elif c:
            warnings.append(f"次因「{c[:30]}」不在候选集内 —— 已剔除")

    evidence, dropped = [], []
    for e in (obj.get("evidence") or [])[:6]:
        if not isinstance(e, dict):
            continue
        field = str(e.get("field") or "").strip()
        # 只认 facts 里**真实存在且非空**的键 —— 这是防"证据编造"的核心闸门
        if field in facts and facts[field] is not None:
            evidence.append({"field": field, "value": e.get("value"),
                             "means": str(e.get("means") or "")[:120]})
        else:
            dropped.append(field[:30])
    if dropped:
        warnings.append(f"剔除了 {len(dropped)} 条不存在的证据字段：{dropped}")
    if not evidence:
        warnings.append("没有任何有效证据字段 —— 该条归因的可核实性降级")

    improvements, bad_targets = [], []
    for it in (obj.get("improvements") or [])[:4]:
        if not isinstance(it, dict):
            continue
        tgt = str(it.get("target") or "").strip()
        if tgt not in IMPROVE_TARGETS:
            bad_targets.append(tgt[:30])
            continue
        improvements.append({"target": tgt,
                             "change": str(it.get("change") or "").strip()[:200],
                             "why": str(it.get("why") or "").strip()[:160]})
    if bad_targets:
        warnings.append(f"改进项 target 不在白名单：{bad_targets}")

    return {
        "primary_cause": primary,
        "secondary_causes": secondary,
        "verdict_reason": str(obj.get("verdict_reason") or "").strip()[:300],
        "evidence": evidence,
        "what_was_right": str(obj.get("what_was_right") or "").strip()[:200],
        "improvements": improvements,
    }, warnings


def review_one(mech: dict, payload: dict, client=None,
               retry_on_invalid: bool = True) -> dict:
    """给一条机械复盘补上 LLM 归因。失败/校验不过 ⇒ 保留机械结论并标 `llm_error`。

    ⚠️ 只调用一次（校验不过时最多再调一次）—— 复盘是**批量**任务，
       单条重试失控会让整批跑不完。
    """
    from llm.deepseek_client import DeepSeekClient, TEMP_ANALYTICAL
    if client is None:
        client = DeepSeekClient()
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _review_prompt(mech, payload)}]
    last_err = ""
    for attempt in (0, 1):
        if attempt and not retry_on_invalid:
            break
        try:
            raw = client.call_api(messages, temperature=TEMP_ANALYTICAL,
                                  purpose="analysis.review")
        except Exception as e:                      # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            _logger.warning("[复盘] LLM 调用失败（第 %d 次）：%s", attempt + 1, e)
            break
        obj = _extract_json(raw)
        if not obj:
            last_err = f"回复无法解析为 JSON；原文前 200 字：{str(raw)[:200]}"
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": str(raw)[:400]},
                    {"role": "user", "content": "上面不是合法 JSON。请只输出 JSON 对象，不要任何解释。"}]
                continue
            break
        cleaned, warnings = validate_llm_review(obj, mech)
        cleaned["warnings"] = warnings
        cleaned["raw_chars"] = len(str(raw or ""))
        return {"status": "ok" if not warnings else "ok_with_warnings", **cleaned}
    return {"status": "llm_error", "error": last_err}


def enrich_with_llm(mech: dict, full: dict, client=None) -> str:
    """在机械复盘上补 LLM 归因（**唯一**的合并实现，两处调用方共用：
    批任务 `review_pending` 与回填里程碑 `analysis_history._backfill_rows`）。

    就地修改 `mech` 并返回最终 status（ok / ok_with_warnings / llm_error）。
    - 成功：`llm_review` / `status` 写入，主因次因以**校验后**的模型选择为准；
    - 失败：保留机械结论，`status="llm_error"` —— 复盘写歪了不能拖垮回填。

    ⚠️ 为什么抽成函数（2026-10-29）：合并逻辑此前只活在 review_pending 里，
       回填路径要用时只能复制一份 ⇒ 两处一旦漂移，"页面说主因是 A、库里是 B"
       这类错位没人看得见（"页面与 prompt 说同一件事必须同源"的老坑）。
    """
    llm = review_one(mech, full.get("payload") or {}, client=client)
    if llm.get("status") in ("ok", "ok_with_warnings"):
        mech["llm_review"] = llm
        mech["status"] = llm["status"]
        # LLM 选了候选之外的「主因」时，以模型的为准（它已在校验里被纠正）
        mech["primary_cause"] = llm.get("primary_cause") or mech["primary_cause"]
        if llm.get("secondary_causes"):
            mech["secondary_causes"] = llm["secondary_causes"]
        return llm["status"]
    mech["status"] = "llm_error"
    mech["llm_review"] = {"status": "llm_error",
                          "error": llm.get("error") or "未知错误"}
    return "llm_error"


# ================================================================
# L3 聚合：从单条归因到"该改什么"
# ================================================================

def aggregate(reviews: list[dict], min_support: int = 3) -> dict:
    """按原因码与改进项聚合。

    ⚠️ `min_support` 的意义：**只出现 1~2 次的改进项不升级为建议**。
       单样本归因极容易"过拟合出一个故事"，按次数过滤是最便宜的防过拟合。
    """
    n = len(reviews)
    by_cause: dict[str, int] = {}
    by_verdict: dict[str, int] = {}
    imp: dict[str, dict] = {}
    for r in reviews:
        by_verdict[r.get("verdict") or "unclear"] = \
            by_verdict.get(r.get("verdict") or "unclear", 0) + 1
        pc = r.get("primary_cause") or "unknown"
        by_cause[pc] = by_cause.get(pc, 0) + 1
        llm = r.get("llm_review") or {}
        for it in (llm.get("improvements") or []):
            tgt = it.get("target")
            if not tgt or tgt == "none":
                continue
            slot = imp.setdefault(tgt, {"target": tgt,
                                        "label": IMPROVE_TARGETS.get(tgt, tgt),
                                        "count": 0, "examples": []})
            slot["count"] += 1
            if len(slot["examples"]) < 3:
                slot["examples"].append(it.get("change") or "")
    causes = sorted(({"code": k, "label": CAUSE_CODES.get(k, ""), "count": v,
                      "share": round(v / n, 3) if n else None}
                     for k, v in by_cause.items()), key=lambda x: -x["count"])
    improvements = sorted(imp.values(), key=lambda x: -x["count"])
    return {
        "n": n,
        "by_verdict": by_verdict,
        "by_cause": causes,
        "improvements": improvements,
        "min_support": min_support,
        # ⚠️ key 固定叫 advice：曾写成 `"建议（出现 ≥%d 次才升级）" % min_support`，
        #    于是 min_support 一变 key 就变，消费方（summarize/页面/脚本）静默取不到。
        "advice": [i for i in improvements if i["count"] >= min_support],
        "pending": [i for i in improvements if i["count"] < min_support],
    }


def summarize(agg: dict, horizon: int = 20) -> str:
    """把聚合结果写成人能读的一段（页面/推送共用，避免两处各写一份措辞）。"""
    if not agg or not agg.get("n"):
        return "还没有可复盘的样本（需要分析后走完 T+20 的窗口）。"
    lines = [f"近 {agg['n']} 条已回填分析（T+{horizon}）的归因分布："]
    for v, k in sorted((agg.get("by_verdict") or {}).items(), key=lambda x: -x[1]):
        lines.append(f"  · {_VERDICT_CN.get(v, v)}：{k} 条")
    lines.append("主要原因：")
    for c in (agg.get("by_cause") or [])[:5]:
        share = f"{c['share'] * 100:.0f}%" if c.get("share") is not None else "—"
        lines.append(f"  · {c['label']}：{c['count']} 条（{share}）")
    top = agg.get("advice") or []
    if top:
        lines.append("值得改的（按出现次数）：")
        for i in top:
            lines.append(f"  · {i['label']} —— 出现 {i['count']} 次；例：{i['examples'][0][:60]}")
    else:
        lines.append("暂时没有出现 ≥%d 次的改进项 —— 单样本归因还不足以支持改规则。"
                     % int(agg.get("min_support") or 3))
    return "\n".join(lines)


_VERDICT_CN = {
    "correct": "判对了",
    "partially_correct": "方向对但幅度很小",
    "wrong": "判错了",
    "unclear": "没走出来（±3% 内）",
    "opportunity_missed": "观望后大涨（错失机会）",
    "no_position": "未预测方向（观望/驳回）",
}


# ================================================================
# 批量入口（供调度/页面调用）
# ================================================================

BENCH_SYMBOL = "sh000300"        # 基准：沪深300（比上证更贴近"选股能力"）
# 回填里程碑（T+5 首写 / T+20 到期）自动 LLM 归因的开关。
# 默认**开**（2026-10-29 用户需求："t+5、t+20 时自动调用 llm 填写总结"）；
# 设 BIGA_ANALYSIS_REVIEW_LLM=0|false|off|no 可关（conftest 的 autouse 夹具就是靠它
# 保证整个测试套件零 LLM/零网络）。
ENV_ENABLE_LLM = "BIGA_ANALYSIS_REVIEW_LLM"


def llm_autorun_enabled() -> bool:
    """回填里程碑是否自动调 LLM 写归因总结（运行时读 env，**不做导入期常量**——
    10-13 踩过的坑：导入期快照会让"改 .env 不重启不生效"）。

    ⚠️ 默认 True：与 `review_pending(with_llm=...)` 的显式参数互不影响——
       这里只管"回填里程碑自动触发"这一条路径。
    """
    v = (os.environ.get(ENV_ENABLE_LLM) or "").strip().lower()
    return v not in ("0", "false", "off", "no")


def _atr14_before(df, analysis_date: str, n: int = 14) -> float | None:
    """分析日**当天及之前**的 n+1 根算 ATR14。

    ⚠️ 只能用分析日之前的数据 —— 用之后的数据算 ATR 就是前视，
       而 ATR 会直接进"止损是否过近"的判据，前视会让归因整个失真。
    """
    if df is None or getattr(df, "empty", True):
        return None
    need = ("date", "high", "low", "close")
    if any(c not in df.columns for c in need):
        return None
    try:
        sub = df[df["date"] <= str(analysis_date)[:10]].tail(n + 1)
        if len(sub) < 3:
            return None
        highs = [float(v) for v in sub["high"].tolist()]
        lows = [float(v) for v in sub["low"].tolist()]
        closes = [float(v) for v in sub["close"].tolist()]
    except (TypeError, ValueError) as e:
        _logger.info("[复盘] ATR 计算失败（%s）：%s", analysis_date, e)
        return None
    trs = []
    for i in range(1, len(sub)):
        trs.append(max(highs[i] - lows[i],
                       abs(highs[i] - closes[i - 1]),
                       abs(lows[i] - closes[i - 1])))
    trs = trs[-(n):]
    return round(sum(trs) / len(trs), 4) if trs else None


def _bench_return(base_date: str, end_date: str) -> float | None:
    """基准指数从 base_date 到 end_date 的收益（两点取同一收盘，绝不混用不同日期）。"""
    try:
        from data import index_bars
        df = index_bars.get_index_bars(BENCH_SYMBOL, start=base_date, end=end_date)
        if df is None or getattr(df, "empty", True) or "close" not in df.columns:
            return None
        sub = df[df["date"] >= str(base_date)[:10]]
        if sub.empty or len(sub) < 2:
            return None
        base = float(sub.iloc[0]["close"])
        last = float(sub.iloc[-1]["close"])
        return round(last / base - 1.0, 4) if base > 0 else None
    except Exception as e:                      # noqa: BLE001
        _logger.info("[复盘] 基准收益取不到（%s~%s）：%s", base_date, end_date, e)
        return None


def _load_bars(code: str, analysis_date: str):
    """本地优先取 K 线（`data.bars`，单只 8.6ms）；拿不到再退回联网取数。

    ⚠️ 复盘是**离线可跑**的：本地面板覆盖 2016 年至今，绝大多数复盘不需要联网。
    """
    start = (datetime.strptime(analysis_date[:10], "%Y-%m-%d")
             - timedelta(days=90)).strftime("%Y-%m-%d")
    try:
        from data.bars import get_bars
        df = get_bars(code, start=start)
        if df is not None and not getattr(df, "empty", True):
            if "date" not in df.columns:
                # `get_bars` 可能把日期放在索引里（index 名不一定是 date）
                df = df.reset_index()
                for alt in ("index", "trade_date", "Date"):
                    if "date" not in df.columns and alt in df.columns:
                        df = df.rename(columns={alt: "date"})
            if "date" in df.columns:
                df = df.copy()
                df["date"] = df["date"].astype(str).str[:10]
                return df.sort_values("date")
    except Exception as e:                      # noqa: BLE001
        _logger.info("[复盘] 本地 K 线不可用（%s）：%s", code, e)
    try:
        from core.analysis_history import default_price_fetcher
        from core.forward_returns import normalize_kline
        df = normalize_kline(default_price_fetcher(code, start,
                                                   datetime.now().strftime("%Y-%m-%d")))
        return df
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[复盘] 取 K 线失败（%s）：%s", code, e)
        return None


def review_pending(limit: int = 20, with_llm: bool = False,
                   client=None) -> dict:
    """给"已 T+20 回填但没有复盘"的分析补复盘。

    Args:
        with_llm: True 时逐条调模型做归因（受限选择 + 证据白名单校验）
        limit: 单次上限 —— 复盘是后台任务，必须有界

    Returns: {"checked", "reviewed", "skipped", "llm_ok", "llm_error", "errors": [...]}
    """
    rows = store.pending_review_analyses(limit=limit, include_mechanical=True)
    stats = {"checked": len(rows), "reviewed": 0, "skipped": 0,
             "llm_ok": 0, "llm_error": 0, "errors": []}
    for r in rows:
        ana_id = r.get("id")
        try:
            full = store.get_analysis(ana_id) or {}
            code = full.get("stock_code") or r.get("stock_code") or ""
            ana_date = str(full.get("analysis_date") or r.get("analysis_date") or "")[:10]
            if not code or not ana_date:
                stats["skipped"] += 1
                continue
            df = _load_bars(code, ana_date)
            if df is None or getattr(df, "empty", True):
                stats["skipped"] += 1
                stats["errors"].append(f"{code}: 取不到 K 线，本次跳过（下次重试）")
                continue
            atr = _atr14_before(df, ana_date)
            horizon = 20 if r.get("ret_20d") is not None else 5
            span = WINDOW_DAYS if horizon >= 20 else horizon
            # 基准区间：基准日 → 窗口末日（与个股同口径，两点取同一收盘）
            ends = df[df["date"] >= ana_date].iloc[1:1 + span]
            bench = (_bench_return(str(ends.iloc[0]["date"])[:10],
                                   str(ends.iloc[-1]["date"])[:10])
                     if not ends.empty else None)
            mech = build_mechanical_review(full, df, atr14=atr, bench_ret=bench,
                                           horizon=horizon)
            if mech is None:
                stats["skipped"] += 1
                continue
            if with_llm:
                llm_status = enrich_with_llm(mech, full, client=client)
                if llm_status == "llm_error":
                    stats["llm_error"] += 1
                    stats["errors"].append(
                        f"{code}: LLM 复盘失败 {(mech.get('llm_review') or {}).get('error')}")
                else:
                    stats["llm_ok"] += 1
            store.upsert_analysis_review(mech)
            stats["reviewed"] += 1
        except Exception as e:                  # noqa: BLE001
            stats["errors"].append(f"{r.get('stock_code') or r.get('id')}: {e}")
            _logger.warning("[复盘] 单条失败（不影响其余）：%s", e)
    if stats["checked"]:
        _logger.info("[复盘] 批量完成：%s", stats)
    return stats


def aggregate_recent(days: int = 180, min_support: int = 3,
                     horizon: int | None = None) -> dict:
    """近 N 天（按分析日）的复盘聚合 + 一段可读摘要。

    `horizon`：None = 只统计**完整窗口（20）**，避免"T+5 早期结论"污染归因分布。
    想看早期信号用 `horizon=5`，页面会明确标注窗口。
    """
    since = (datetime.now() - timedelta(days=int(days))).strftime("%Y-%m-%d")
    reviews = store.list_analysis_reviews(limit=1000, since=since,
                                          horizon=horizon if horizon else 20)
    agg = aggregate(reviews, min_support=min_support)
    agg["since"] = since
    agg["horizon"] = horizon or 20
    # ⚠️ 标题里的 T+N 必须跟着实际窗口走：第一版固定写"T+20"，
    #    于是 T+5 的摘要标题写着 T+20 —— 这正是"标错口径"的老坑。
    agg["text"] = summarize(agg, horizon=agg["horizon"])
    return agg

