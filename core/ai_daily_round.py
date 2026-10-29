# -*- coding: utf-8 -*-
"""多 AI 交易联赛 · 盘后决策轮（C5，2026-10-14，S4）。

时序（挂在 15:58，晚于 15:55 的新闻行业评估）：
    15:30 盘后全流程（选股）→ 15:40 快照 → 15:45 复盘 → 15:55 新闻评估 → **15:58 ai_daily_round**

⚠️ 公平性铁律：**公共简报对所有 AI 逐字节相同**
（`build_common_briefing()` 一次组装、N 个 AI 共用；只有"自己的持仓/净值/预算"是私有段）。
否则比较的不是策略，而是"谁拿到了更多信息"——测试用 JSON 全等断言钉住。

⚠️ 成本纪律（口径 5）：每次调用带 `purpose=f"ai.{persona}.decision"`，
   便于在 `llm_calls` 里按 AI 归因；模型由 `model_override` 钉住（并发不串号）。
⚠️ 额度纪律：每个 persona 每日 1 次 decision_round 额度，**先扣后跑**
   （先跑后记会出现"并发时超额"）。同日重跑由额度 + ai_decisions 主键双重幂等。
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime

from core import ai_agent, ai_budget, ai_personas
from core.paper_account import PaperAccount
from db import store

_logger = logging.getLogger(__name__)


# ---------------- 简报组装 ----------------

# 取多少行**原始**选股候选用于组装候选池。
# ⚠️ 必须远大于候选池上限：同一只票会因多来源/多策略/多批次出现多行
#    （实测同一票 3 行）⇒ limit=20 实际只覆盖约 7 只**不同**标的。
_PICK_ROWS = 200
# 去重后进入候选池的最大标的数
_MAX_CANDIDATES = 30


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _dedup_by_code(rows: list) -> list:
    """按 `stock_code` 去重，保留**分数最高**的一条（同分保留最新 id）。

    ⚠️ 为什么要去重（2026-10-15 实测）：`store.list_screen_picks` 按 id 倒序取行，
    同一只票会有多行（pipeline / screen 多来源）⇒ 直接 `[:20]` 会让候选池被
    重复行挤占，AI 看到的"池子"其实只有 6~7 只票，于是只能反复选那几只
    （这也是"选股跨日高度重复"在联赛里的表现形式）。
    """
    best: dict = {}
    for r in (rows or []):
        code = str(r.get("stock_code") or "").strip()
        if not code:
            continue
        try:
            score = float(r.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        cur = best.get(code)
        if cur is None or score > cur[0] or (score == cur[0] and (r.get("id") or 0) > (cur[1].get("id") or 0)):
            best[code] = (score, r)
    ordered = sorted(best.values(), key=lambda t: -t[0])
    return [r for _s, r in ordered]


def _load_market_signal(day: str) -> dict:
    """读**当日流水线已算好**的市场信号（零联网、零计算）。

    ⚠️⚠️ 为什么不再现算 `MarketDetector().detect()`（2026-10-23 自伤一次）：
      它实测 **30.4 秒**（14 个指标逐个取数），而 `build_common_briefing`
      每轮都要调用 ⇒ 每个决策轮白等半分钟；更糟的是**测试里会被放大几十倍**
      （一个用例集跑 14 分钟还没完）。目标仓位只是"今天该拿多少仓"的参照，
      15:30 的流水线已经算过一遍，读它的产物即可 —— 同一份数据，不要算两遍。

    数据源：`output/daily_picks/latest.json` 的 `market_signal`。
    ⚠️ 2026-10-25 起该字段是 **dict**（`core/pipeline._json_default` 把 MarketSignal
       dataclass 落盘为 dict）；磁盘上此前的文件仍是 repr 字符串
      （`MarketSignal(total_score=28.4, ...)`）⇒ 两种格式都必须能吃。
      解析统一走 `core.market_signal_codec.coerce`（**全项目唯一编解码器**，
      `web/server` 也用它）——这里不再自带正则，避免两套实现漂移。
    """
    out = {"available": False, "score": None, "state": None,
           "target_exposure": None, "note": ""}
    try:
        import json as _json
        import os as _os
        path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                             "output", "daily_picks", "latest.json")
        with open(path, encoding="utf-8") as f:
            d = _json.load(f)
        if str(d.get("date")) != str(day):
            out["note"] = f"市场信号不是今天的（latest.json={d.get('date')}）⇒ 不参与目标仓位"
            return out
        from core.market_signal_codec import coerce, has_required
        sig, _fmt = coerce(d.get("market_signal"))
        if not has_required(sig):
            _logger.warning("[联赛简报] market_signal 字段缺失或无法解析 ⇒ 目标仓位不可用")
            out["note"] = "market_signal 字段缺失或无法解析 ⇒ 目标仓位不可用"
            return out
        try:
            out.update({"available": True,
                        "score": round(float(sig["total_score"]), 1),
                        "state": str(sig["market_state"]),
                        "target_exposure": round(float(sig["position_limit"]), 4),
                        "note": "目标仓位 = 当日流水线市场检测的仓位上限（全项目唯一口径）"})
        except (TypeError, ValueError):
            _logger.warning("[联赛简报] market_signal 字段类型异常 ⇒ 目标仓位不可用")
            out["note"] = "market_signal 字段类型异常 ⇒ 目标仓位不可用"
    except FileNotFoundError:
        # ⚠️ 正常状态（当日流水线还没跑）⇒ debug 即可，不必刷 WARNING
        _logger.debug("[联赛简报] 未找到 daily_picks/latest.json ⇒ 目标仓位不可用")
        out["note"] = "未找到 daily_picks/latest.json ⇒ 目标仓位不可用"
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[联赛简报] 市场信号读取失败: %s: %s", type(e).__name__, e)
        out["note"] = f"市场信号读取失败: {type(e).__name__}"
    return out


# 情绪面票（新闻→个股严匹配）在简报里的容量：**只给聚合与标题，不给原文**
_NEWS_HITS_SHOWN = 8          # 展示条数（Top N）
_NEWS_HITS_MIN = 2            # 低于 2 次提及视为噪声（单次提及常是行业词误命中）
_NEWS_SAMPLE_CHARS = 48       # 标题截断长度（省 token，也别把全文塞给模型）


def _load_news_hits(day: str) -> dict:
    """当日"新闻→个股"严匹配命中（零联网，读 `news_stock_hits`）。

    为什么放进**公共简报**：它是"今天谁被点名"的客观事实，对所有 AI 必须一致
    （公平性铁律）；私有段只放持仓/净值/预算这类**自身状态**。
    为什么只给 Top N + 标题截断：一天可能命中上百只，全塞进去会把输入撑爆而
    利用率接近零；AI 需要的是"注意力指向"，细节可以用工具现查。

    ⚠️ 缺就是缺：当天没跑采集/匹配、或命中不足阈值，一律 `available=False` 并写明
    原因，**不回退到昨天的数据**（昨天的新闻对"明天怎么交易"是误导）。
    """
    out: dict = {"available": False, "date": day, "total": 0,
                 "min_hits": _NEWS_HITS_MIN, "rows": [], "note": ""}
    try:
        from core import news_stock_hits
        rows = news_stock_hits.top_hits(day, limit=500, min_hits=_NEWS_HITS_MIN)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[简报] 情绪面命中读取失败: %s", e)
        out["note"] = f"读取失败: {type(e).__name__}"
        return out
    if not rows:
        out["note"] = (f"当日无命中（提及≥{_NEWS_HITS_MIN} 次的个股为 0）："
                       "可能新闻采集/情绪匹配未跑，或当天确实没有个股被反复点名")
        return out
    shown = []
    for r in rows[:_NEWS_HITS_SHOWN]:
        samples = r.get("samples") or []
        head = str(samples[0]) if samples else ""
        if len(head) > _NEWS_SAMPLE_CHARS:
            head = head[:_NEWS_SAMPLE_CHARS] + "…"
        shown.append({
            "code": r.get("stock_code"),
            "name": r.get("stock_name"),
            "hits": int(r.get("hits") or 0),
            "positive": int(r.get("positive") or 0),
            "negative": int(r.get("negative") or 0),
            "headline": head,
        })
    out.update({"available": True, "total": len(rows), "rows": shown,
                "note": (f"当日共 {len(rows)} 只被提及 ≥{_NEWS_HITS_MIN} 次，"
                         f"这里只给前 {len(shown)} 只（按提及次数降序）")})
    return out


def build_common_briefing(trade_date: str = None) -> dict:
    """组装**所有人共用**的当日简报（一次组装 N 个 AI 共用）。

    全部来自**已落库**的产物（不新增采集、不联网）：选股候选、新闻行业评估与推荐、
    可用报告清单、数据日期。取不到的字段一律 `None`/空列表并带 `warnings`
    —— **缺就是缺，不补造**（项目铁律）。
    """
    day = trade_date or _today()
    warnings: list[str] = []

    # ① 选股候选（盘后 15:30 的产物）—— **先取足量原始行，再去重**
    picks_raw: list = []
    try:
        picks_raw = store.list_screen_picks(limit=_PICK_ROWS)
    except Exception as e:                                   # noqa: BLE001
        warnings.append(f"选股候选读取失败: {type(e).__name__}")
    picks = _dedup_by_code(picks_raw)
    if picks_raw and len(picks) < 8:
        # 池子窄是**已知问题**（选股跨日重复）⇒ 如实告知 AI，别让它以为"市场就这些票"
        warnings.append(f"候选池偏窄：{len(picks_raw)} 行原始候选只对应 {len(picks)} 只不同标的")

    # ①.5 市场信号 → **目标仓位**（2026-10-23 新增）
    #   为什么放公共段：目标仓位对所有 AI 必须**逐字节相同**（公平性铁律），
    #   不同的只有"自己当前仓位多少"（在私有段）。
    #   为什么用 `position_limit` 而**不自造一套映射**：它已经是全项目唯一的
    #   "市场温度 → 仓位上限"口径（20/40/60/80）。再写第二套映射就是重复实现，
    #   迟早分叉（本项目已有"两套市场评分口径"的前车之鉴）。
    market_signal = _load_market_signal(day)
    if not market_signal.get("available"):
        warnings.append(market_signal.get("note") or "市场信号不可用")

    # ①.6 情绪面票：当日"新闻→个股"命中（2026-10-24 新增，**公共段**）
    #    它回答的是"今天市场在反复讨论哪几只票"——这是 AI 之前**完全看不到**的事实
    #    （只有盘后分析用过 2 席）。放在公共段 ⇒ 6 个 AI 逐字节相同。
    news_hits = _load_news_hits(day)
    if not news_hits.get("available"):
        warnings.append(f"情绪面命中不可用：{news_hits.get('note')}")

    # ② 新闻行业评估 + 利好行业推荐（15:55 的产物）
    news_review = {"review_date": None, "rows": [], "recommend": None}
    try:
        rdate, rows = store.get_news_reviews(day)
        if rdate is None:
            rdate, rows = store.get_news_reviews(None)
        news_review["review_date"] = rdate
        news_review["rows"] = [
            {"sector": r.get("sector"), "verdict": r.get("verdict"),
             "summary": r.get("summary")}
            for r in (rows or [])]
        if rdate:
            from core.news_industry_review import recommend_stocks
            news_review["recommend"] = recommend_stocks(rdate, top_n=5)
    except Exception as e:                                   # noqa: BLE001
        warnings.append(f"新闻行业评估读取失败: {type(e).__name__}")

    # ③ 可用报告清单（哪些票已有深度分析，可直接读；哪些要花 credit 现场跑）
    reports = []
    try:
        for a in store.list_analyses(limit=30):
            reports.append({"code": a.get("stock_code"),
                            "date": a.get("analysis_date"),
                            "analysis_id": a.get("id")})
    except Exception as e:                                   # noqa: BLE001
        warnings.append(f"报告清单读取失败: {type(e).__name__}")

    # ④ 候选池（AI 决定"研究哪只"的起点）：选股候选 ∪ 新闻推荐的利好行业成分
    cand: list[str] = []
    for p in picks:
        c = str(p.get("stock_code") or "").strip()
        if c and c not in cand:
            cand.append(c)
    rec = news_review.get("recommend") or {}
    for p in (rec.get("picks") or []):
        c = str(p.get("stock_code") or "").strip()
        if c and c not in cand:
            cand.append(c)

    # ⑤ 对照仓位（随机对照组当日平均仓位）—— 公共段，对所有 AI 逐字节相同。
    #    作用：让"今天不动"变成一个**需要解释**的选择（同口径对照摆在那里）。
    #    ⚠️ 同时披露**随机组当日盘中成交聚合**（次数/金额）：放开盘中补仓后，
    #       盘中参与度也需要信息对称，否则"别人盘中有响应、我只有盘后"这件事
    #       AI 根本不知道（用户 2026-10-15 的披露要求）。
    #       仍然**只给聚合值、不含任何标的代码** ⇒ 不破坏公平性。
    try:
        from core import ai_league
        market_context = ai_league.random_group_exposure(day)
        try:
            market_context["intraday"] = ai_league.random_group_intraday(day)
        except Exception as e:                               # noqa: BLE001
            market_context["intraday"] = {"available": False,
                                          "note": f"盘中聚合读取失败: {type(e).__name__}"}
        try:
            market_context["wake_buy"] = ai_league.wake_buy_status(day)
        except Exception as e:                               # noqa: BLE001
            market_context["wake_buy"] = {"enabled": False,
                                          "reason": f"开关读取失败: {type(e).__name__}"}
    except Exception as e:                                   # noqa: BLE001
        market_context = {"available": False, "avg_exposure_pct": None, "n": 0,
                          "note": f"对照仓位读取失败: {type(e).__name__}"}
        warnings.append(f"对照仓位读取失败: {type(e).__name__}")

    return {
        "as_of": day,
        "screen_picks": [{"code": p.get("stock_code"), "name": p.get("stock_name"),
                          "source": p.get("source"), "strategy": p.get("strategy"),
                          "score": p.get("score"), "pick_date": p.get("pick_date")}
                         for p in picks[:10]],
        "candidate_pool_size": len(cand),
        "candidate_note": (f"候选池已按代码去重（原始 {len(picks_raw)} 行 → "
                           f"{len(picks)} 只不同标的）"),
        "market_signal": market_signal,
        "news_hits": news_hits,
        "market_context": market_context,
        "news_review": news_review,
        "reports": reports,
        "candidates": cand[:_MAX_CANDIDATES],
        "warnings": warnings,
    }


def _private_overlay(persona_name: str, trade_date: str) -> dict:
    """私有段（只有这一项允许各 AI 不同）：自己的持仓/净值/剩余额度。

    ⚠️ 另含 `recent_fills` / `prev_decisions` / `nav_recent` —— AI 要写"对前一交易日的复盘"
    就必须看得到自己的成交与净值轨迹（**它的私有信息**，不是公共简报，不破坏公平性）。
    取不到一律空列表 + `None`，**缺就是缺，不补造**。
    """
    out = {"holdings": [], "nav": None, "exposure": None,
           "nav_at_cost": None, "nav_note": None,
           "budget": {}, "recent_fills": [], "prev_decisions": [], "nav_recent": [],
           "prev_trading_day": None}
    st = None
    try:
        st = store.load_paper_state(persona_name)
        if st:
            acc = PaperAccount.from_state(st)
            # ⚠️ 必须用**当日收盘价**估值，而不是 `avg_cost`（2026-10-15 修）。
            #    原版用 `{c: p.avg_cost}` ⇒ 净值恒等于成本 ⇒ AI 读到"净值没变"
            #    就写出"未产生盈亏"的错误复盘（实测 ai_tech / ai_value 都这么错）。
            #    取价唯一实现 = `ai_league.close_prices`（快照优先 → 日线兜底）。
            from core import ai_league                       # 延迟导入避免环形
            codes = [c for c, p in acc.positions.items() if p.quantity > 0]
            px = ai_league.close_prices(codes, trade_date) if codes else {}
            at_cost = len([c for c in codes if c not in px])
            valuations = {c: px.get(c, acc.positions[c].avg_cost) for c in codes}
            out["nav"] = round(acc.total_value(valuations), 2)
            out["nav_at_cost"] = at_cost
            # ⚠️ 口径必须随数字走：AI 看到 at_cost>0 就知道"这条净值里有估的"，
            #    不能拿来判断盈亏（缺就是缺，别把它读成"没涨没跌"）。
            out["nav_note"] = ("全部按当日收盘价估值" if at_cost == 0 else
                               f"其中 {at_cost} 只因取不到当日价而按**成本**估值 ⇒ "
                               f"该净值仅供连续性，**不可用于盈亏判断**")
            # 当前仓位（= 持仓市值 / 总权益）—— **私有段**，与公共段的
            # `market_signal.target_exposure` 对照，才谈得上"我该不该建仓"。
            # 取不到价时按成本估（与 nav 同口径），不制造"仓位 0"的假象。
            pos_value = sum(acc.positions[c].quantity * valuations[c] for c in codes)
            out["exposure"] = (round(pos_value / out["nav"], 4)
                               if out["nav"] else None)
            out["holdings"] = [
                {"code": c, "qty": p.quantity, "avg_cost": round(p.avg_cost, 4),
                 "price": (round(px[c], 4) if c in px else None),
                 "pnl_pct": (round((px[c] / p.avg_cost - 1) * 100, 2)
                             if c in px and p.avg_cost else None),
                 "available": p.available}
                for c, p in acc.positions.items() if p.quantity > 0]
            out["recent_fills"] = [{"date": f.get("date"), "code": f.get("code"),
                                    "direction": f.get("direction"),
                                    "qty": f.get("quantity"),
                                    "price": f.get("actual_price"),
                                    "pnl": f.get("pnl")}
                                   for f in (st.get("fills") or [])[-8:]]
            out["nav_recent"] = [{"date": n.get("date"),
                                  "total_value": n.get("total_value"),
                                  "return_pct": n.get("return_pct"),
                                  "at_cost": n.get("at_cost")}
                                 for n in (st.get("nav") or [])[-5:]]
    except Exception as e:                                   # noqa: BLE001
        out["nav"] = None
        _logger.warning("[AI决策轮] 读取 %s 账户失败: %s", persona_name, e)

    # 前一交易日的决策（供"复盘"用）。缺交易日历/取不到时留 None，不硬编。
    try:
        from data.trading_calendar import last_trading_day
        prev = last_trading_day(ref=trade_date, include_today=False)
        out["prev_trading_day"] = prev
        out["prev_decisions"] = [
            {"code": r.get("code"), "action": r.get("advice_status"),
             "position_size": r.get("position_size"),
             "reason": r.get("order_reason")}
            for r in store.list_ai_decisions(persona=persona_name, signal_date=prev)]
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[AI决策轮] 读取 %s 前一日决策失败: %s", persona_name, e)

    out["budget"] = ai_budget.view(persona_name, trade_date)
    return out


def _tools_doc() -> str:
    lines = ["可用工具（只能调用下列名称；未列出的会被拒绝）："]
    for t in sorted(ai_agent.TOOL_REGISTRY.values(), key=lambda x: x.name):
        cost = f"，消耗 {t.cost} credit" if t.cost else ""
        scenes = "/".join(sorted(t.allowed_in))
        lines.append(f"  - {t.name}（{scenes}）: {t.desc}{cost}")
    return "\n".join(lines)


def _round_prompt(persona: dict, common: dict, private: dict,
                  transcript: list, round_no: int, max_rounds: int,
                  *, is_last: bool = False, must_call_tool: bool = False) -> str:
    last_note = (
        "\n⚠️ **这是最后一轮：你必须现在给出 final，不允许再调用任何工具。**\n"
        if is_last else "")
    # ⚠️ 「首轮直接出结论」要**由代码拦一次**（2026-10-17）：实测每个 persona 每轮只调 1 次 LLM、
    #    第 1 轮就返回 final ⇒ **工具全程未被使用**，"通了深度分析"在生产中等于零。
    #    提示词里写"你必须调用一次工具"**管不住**模型，所以这里由编排层强制（见 run_persona_round）。
    must_note = (
        "\n⚠️⚠️ **本轮禁止直接给 final**：你还没有调用过任何工具，"
        "手上只有简报里的现成数字。请**先调用至少一个工具**（如 `get_stock_data` / "
        "`get_my_budget` / `get_portfolio`）取得你需要的事实，下一轮再给结论。\n"
        if must_call_tool else "")
    return f"""你是「{persona.get('label')}」这个模拟交易 AI，正在做 {common['as_of']} 盘后的**明日**交易决策。

【你的交易风格】
{persona.get('style_prompt') or '（无特定风格）'}

【今日简报（所有 AI 相同）】
{json.dumps(common, ensure_ascii=False, indent=1)}

【你的私有信息】
{json.dumps(private, ensure_ascii=False, indent=1)}

【你已经做过的调用（第 {round_no}/{max_rounds} 轮）】
{json.dumps(transcript, ensure_ascii=False, indent=1) if transcript else '（还没有）'}
{last_note}{must_note}
{_tools_doc()}

【输出格式（**只输出一个 JSON 对象，不要任何其他文字**）】
- 需要调用工具：{{"tool": "工具名", "args": {{"参数": "值"}}}}
- 想结束并给出决策：{{"final": {{"review": "对前一交易日交易的复盘（1~3 句，必填）",
   "actions": [{{"action": "buy|sell|set_wake|pass",
   "code": "600028", "position_size": 0.1, "direction": "买入", "reason": "一句话理由",
   "kind": "stop_loss|take_profit|price_above|price_below|time",
   "threshold": 5.0, "wake_at": "14:55"}}], "summary": "本轮总体思路（必填）"}}}}

【纪律（必须遵守）】
- **先写复盘再决策**：`review` 必须基于你私有信息里的 `recent_fills`/`prev_decisions`/`nav_recent`
  写实（"昨天买了什么、今天赚亏多少、判断对不对"）；**没有成交就直说没有**，禁止编造盈亏数字。
- ⚠️ **看净值先看 `at_cost`**：`nav_recent` 里某天 `at_cost > 0`（或 `nav_note` 提示"按成本估值"）
  表示那天有持仓取不到当日价、净值里有估的部分 ⇒ **那天的净值不可用于判断盈亏**，
  必须直说"当日估值不完整"，**不得据此得出"净值没变、没赚没亏"**。
- 持仓里 `price=null` 表示取不到当日价 ⇒ 同样不得凭空判断该标的的涨跌。
- ⚠️⚠️ **目标仓位（硬纪律，2026-10-23 新增）**：简报 `market_signal.target_exposure` 是
  市场检测给出的**目标仓位**（全项目唯一口径，所有 AI 相同）；你私有段里的 `exposure`
  是**你当前的仓位**（持仓市值÷总权益）。
  · 若 `exposure` 比 `target_exposure` 低 **15 个百分点以上**，你**必须**二选一：
    （a）给出 `buy` 建仓计划（可多只，见下条）；或
    （b）在 `summary`/`reason` 里写明**为什么这个目标对你不适用**（例如：你的持仓已集中、
        目标标的全在你的禁止清单、或你对市场判断与该分数相反并给出依据）。
    只写"环境不明朗/继续观察"而**不回应差距**的答案视为不完整。
  · 反过来的情况（显著高于目标）同理：要么减仓，要么说明为什么值得超配。
- ⚠️ **建仓轮一次给 2~3 只**：实测每个 AI 长期只持 1~2 只、现金闲置 70~90%，
  根因是"一轮只挑 1 只"⇒ 资金利用率被单票上限锁死。仓位分散靠**多只小仓**实现，
  不是靠单只重仓（单票上限仍由代码夹紧）。
- ⚠️ **`news_hits` 是"注意力指向"，不是买入理由**：简报里给了当日被新闻反复点名的个股
  （按提及次数降序，含正/负提及计数与标题片段）。它只回答"市场在讨论谁"，
  **不证明**基本面或价位会怎样 —— 要用它，先核实（`get_stock_data` 看量价、看是否已在高位）。
  反过来也一样：**没被点名 ≠ 不该买**，别把"不在榜上"当负面信号。
- ⚠️ **空仓要对照 `market_context`**：简报里给了**随机对照组当日平均仓位**（同口径、所有 AI 相同）。
  若你选择不动，必须在 reason 里回应"在对照仓位 X% 的情况下我为何仍不参与"——
  **不作为也是决策，要能解释**，不能只用"环境不明朗"泛泛带过。
- ⚠️ **pass 必须带"观察条件"**：不动的日子请至少给 1 条 `set_wake`（`price_above` / `price_below`，
  可对**非持仓**标的），写清"到什么价位/什么情况我会动手"—— 这样系统能替你盯盘并在触发时唤醒你，
  而不是第二天再从零判断。**没有观察条件的 pass 视为不完整的答案。**
- ⚠️ **盘中响应能力的真实边界**（别高估也别低估）：`market_context.wake_buy` 告诉你
  "盘中补仓当前是否开放"。开放时，你设的 `price_below` 条件盘中触发后，
  **可以对已有持仓的标的做补仓**（单次 ≤持仓市值 3% 且 ≤权益 3%；
  单票 1 次/日、全日 ≤4 次且 ≤权益 6%）——这是"把被动等待变成主动响应"的唯一渠道。
  ⚠️ 但你**不能**盘中开新仓（`price_above` 只用于卖出止盈），也**不能**盘中申请深度分析。
- **先想清楚再调用工具**：工具次数有限，不确定就先给 final。**任何情况下都不许超过 3 次工具调用**。
- 禁止编造简报里没有的数据；不确定就说不确定。
- 仓位只是**意图**，系统会用代码夹紧到风格上限；不要试图超过上限。
- 不动的日子必须显式 pass 并给理由（禁止沉默），pass 也是**合格答案**。
- 每个买入必须同时给该标的的止损条件（set_wake + stop_loss）。
"""


# ---------------- 解析与工具执行 ----------------

_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def parse_round_reply(text: str) -> dict | None:
    """解析模型的一轮输出。宽容取第一个 JSON 对象；解析不出返回 None（调用方按重试/失败处理）。"""
    if not text:
        return None
    m = _JSON_OBJ.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    if "tool" in obj or "final" in obj:
        return obj
    return None


def parse_actions_reply(text: str) -> dict:
    """**宽松**解析"只含 actions 的回复"（唤醒轮用）。解析不出返回 `{}`。

    与 `parse_round_reply` 的分工：后者是**工具循环**用的严格版（必须带 tool/final，
    否则会被当工具调用、白多跑一轮）；这里是**单轮动作**用的宽松版，
    同时接受 `{"final": {...}}` 与裸 `{"actions": [...]}`。
    ⚠️ 宽松接受、严格解释 —— 格式微差不该让一次盘中决策凭空丢失（实测踩过）。
    """
    if not text:
        return {}
    m = _JSON_OBJ.search(str(text))
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return {}
    if not isinstance(obj, dict):
        return {}
    body = obj.get("final") if isinstance(obj.get("final"), dict) else obj
    return body if isinstance(body, dict) else {}


def execute_tool(name: str, args: dict, *, persona: dict, common: dict,
                 private: dict, trade_date: str, context: str = "daily",
                 analysis_runner=None) -> dict:
    """执行一次已校验的工具调用。Returns: 给模型看的结果 dict。

    ⚠️ `request_deep_analysis` **先扣 credit 再跑**（先跑后记会并发超额）；
       真实分析图通过 `analysis_runner` 注入（默认 None ⇒ 如实返回"不可用"，
       绝不假装跑过 —— 见模块文档）。
    """
    ok, why = ai_agent.validate_tool_call(name, args, context=context)
    if not ok:
        return {"ok": False, "error": why}

    try:
        if name == "get_quote":
            from data.realtime import get_realtime_quotes
            q = get_realtime_quotes([args["code"]]) or {}
            info = q.get(args["code"]) or {}
            return {"ok": True, "quote": {k: info.get(k) for k in
                                          ("name", "price", "change_pct", "date", "stale")}}
        if name == "get_my_positions":
            return {"ok": True, "positions": private.get("holdings")}
        if name == "get_my_nav":
            return {"ok": True, "nav": private.get("nav")}
        if name == "get_my_budget":
            return {"ok": True, "budget": private.get("budget")}
        if name == "get_news":
            items = store.list_news(limit=int(args.get("limit") or 20)) \
                if hasattr(store, "list_news") else []
            return {"ok": True, "news": [{"title": n.get("title"),
                                          "source": n.get("source")}
                                         for n in (items or [])]}
        if name == "get_sector_news":
            return {"ok": True, "rows": [r for r in common["news_review"]["rows"]
                                         if r.get("sector") == args.get("sector")]}
        if name == "list_reports":
            return {"ok": True, "reports": common["reports"]}
        if name == "read_report":
            for r in common["reports"]:
                if r.get("code") == args.get("code"):
                    full = store.get_analysis(r["analysis_id"]) or {}
                    return {"ok": True, "code": r["code"],
                            "payload_keys": sorted((full.get("payload") or {}).keys()),
                            "errors": full.get("errors") or [],
                            "audit_warnings": full.get("audit_warnings") or []}
            return {"ok": False, "error": "简报的报告清单里没有这只票；"
                                          "如需现场分析请用 request_deep_analysis"}
        if name == "request_deep_analysis":
            # ⚠️ 执行器不可用时**直接拒绝且不扣额度**（2026-10-14 实测教训）：
            #    首版是"先扣再检查 runner" ⇒ 什么都没跑却扣了 credit，
            #    而模型看到"未接入"提示会**反复重试**（实测一轮里扣掉 2 次）⇒ 纯白花钱。
            if analysis_runner is None:
                return {"ok": False,
                        "error": "现场深度分析执行器未接入，本环境不可用（未消耗额度）；"
                                 "请读已有报告或直接基于简报决策"}
            granted, view = ai_budget.consume(persona["name"], "deep_analysis",
                                              trade_date=trade_date)
            if not granted:
                return {"ok": False, "error": f"deep_analysis 额度不足（{view}）"}
            res = analysis_runner(args["code"])
            # ⚠️ 失败必须**如实转述**（否则模型会把"没拿到"当成"拿到了空报告"继续推理）。
            #    额度已在跑之前扣掉（并发防超额），失败不退还，但要明确告知已消耗。
            if isinstance(res, dict) and res.get("ok") is False:
                return {"ok": False, "credit_left": view["remaining"],
                        "credit_consumed": True,
                        "error": f"深度分析未成功：{res.get('error') or '未知原因'}"}
            return {"ok": True, "credit_left": view["remaining"],
                    "result": res if isinstance(res, (dict, list, str)) else str(res)}
        # 动作类工具：只记录意图（S5/S6 负责执行与结算）
        if name in ("place_order", "sell", "set_wake_condition", "pass"):
            return {"ok": True, "note": "意图已记录，将在 final 中确认"}
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[AI决策轮] 工具 %s 执行失败: %s", name, e)
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}
    return {"ok": False, "error": f"工具 {name} 无执行器"}


# ---------------- 单个 persona 的一轮 ----------------

def _shrink(obj, limit: int = 400) -> object:
    """把工具结果压成可安全回灌提示词的小对象。

    ⚠️ 必要性（2026-10-14 实测）：每次调用都要重发**完整 transcript**，
    实测单次 ≈4.1~4.6k tokens ⇒ 轮次越多越贵。压缩后只保留结构与关键字段。

    ⚠️ `limit` **必须按工具区分**：深度分析的正文/errors 是它的全部价值，
    用 400 字符截断等于白跑（见 `_ANALYSIS_RESULT_CHARS`）。
    """
    s = json.dumps(obj, ensure_ascii=False) if not isinstance(obj, str) else obj
    if len(s) <= limit:
        return obj
    return s[:limit] + f"…（已截断，原长 {len(s)} 字符）"


# 深度分析结果的回灌预算（比普通工具结果大得多：报告正文 + 审计/错误都在里面）
_ANALYSIS_RESULT_CHARS = 2400


def run_persona_round(persona: dict, common: dict, *, trade_date: str,
                      client=None, analysis_runner=None,
                      max_rounds: int = 6, consume_budget: bool = True) -> dict:
    """跑一个 LLM persona 的决策轮（工具循环，最多 max_rounds 轮）。

    额度：**先扣** decision_round（1 次/日）；扣不到直接返回 skipped（幂等）。
    ⚠️ `consume_budget=False` = **人工补跑**（见 `run_daily_round` 的同名参数）：
       预算护栏约束的是**AI 自主花钱**，人工触发的补跑不该占用它的当日额度
       —— 否则补跑一次就把 15:58 的正式盘后轮挤成 skipped。
    Returns: {"persona","status","decisions","summary","transcript","error"}
    """
    name = persona["name"]
    if consume_budget:
        granted, view = ai_budget.consume(name, "decision_round", trade_date=trade_date)
        if not granted:
            return {"persona": name, "status": "skipped",
                    "decisions": [], "summary": "", "review": "", "transcript": [],
                    "error": f"decision_round 额度不足（{view}）"}

    private = _private_overlay(name, trade_date)
    transcript: list[dict] = []
    final_obj: dict | None = None

    from llm.deepseek_client import DeepSeekClient, model_override, TEMP_ANALYTICAL

    if client is None:
        client = DeepSeekClient()
    rounds = max(1, max_rounds)
    _tool_calls = 0
    # ⚠️ 只强制一次（成本上限 +1 次 LLM 调用）：若模型在"必须调工具"的那一轮仍直接给 final，
    #    就**接受**它的答案 —— 硬拦会变成死循环，而且"某轮无工具也能决策"本身不违法。
    _tool_forced = False
    for rnd in range(1, rounds + 1):
        prompt = _round_prompt(persona, common, private, transcript, rnd, rounds,
                               is_last=(rnd == rounds),
                               must_call_tool=(_tool_calls == 0 and _tool_forced))
        try:
            # ⚠️ 模型由 ContextVar 钉住（并发跑多个 AI 不串号）；purpose 打 ai.<name>.*
            with model_override(persona.get("model")):
                raw = client.call_api(
                    [{"role": "user", "content": prompt}],
                    temperature=TEMP_ANALYTICAL, max_tokens=1600,
                    purpose=f"ai.{name}.decision")
        except Exception as e:                               # noqa: BLE001
            return {"persona": name, "status": "error", "decisions": [],
                    "summary": "", "review": "", "transcript": transcript,
                    "error": f"LLM 调用失败: {type(e).__name__}: {str(e)[:120]}"}

        obj = parse_round_reply(raw)
        if obj is None:
            transcript.append({"round": rnd, "error": "输出不是合法 JSON，已忽略"})
            continue
        if "final" in obj:
            # ⚠️ **首轮就出结论 ⇒ 退回一次**（2026-10-17）：
            #    实测每个 persona 每轮只调 1 次 LLM、第 1 轮就 final ⇒ 工具全程未使用。
            #    这里不改提示词（提示词管不住），而是**编排层不接这一版答案**，
            #    让它在下一轮先取一次事实；只退回一次以控制成本。
            if _tool_calls == 0 and not _tool_forced and rounds > 1:
                _tool_forced = True
                transcript.append({
                    "round": rnd,
                    "note": "首轮未调用任何工具就想给结论 ⇒ 退回，要求先取一次事实",
                    "rejected_final_summary": str((obj.get("final") or {}).get("summary")
                                                  or "")[:120]})
                continue
            final_obj = obj["final"] if isinstance(obj["final"], dict) else {}
            break
        tool, args = obj.get("tool"), obj.get("args") or {}
        _tool_calls += 1
        result = execute_tool(tool, args, persona=persona, common=common,
                              private=private, trade_date=trade_date,
                              analysis_runner=analysis_runner)
        # ⚠️ 深度分析结果要**大得多的回灌预算**：它的价值恰恰在报告正文与
        #    `errors`/`audit_warnings`，压到 400 字符会把它们全截掉（等于白跑一趟）。
        #    其余工具结果维持 400（省 token，实测每轮 ~4.3k）。
        _limit = _ANALYSIS_RESULT_CHARS if tool == "request_deep_analysis" else 400
        transcript.append({"round": rnd, "tool": tool, "args": args,
                           "result": _shrink(result, _limit)})

    if final_obj is None:
        # ⚠️ 轮次耗尽**也要有产出**（2026-10-14 实测：模型可能连着 6 轮只调工具，
        #    首版直接判 no_final 什么都不落库 ⇒ 花掉 2.6 万 tokens 却零记录）。
        #    降级为一条 pass（带做了什么的事实描述），让复盘知道"它研究了但没定论"。
        tried = [t.get("tool") for t in transcript if t.get("tool")]
        return {"persona": name, "status": "no_final",
                "decisions": [{"action": "pass", "code": None,
                               "reason": f"{rounds} 轮内未给出结论（调用过: "
                                         f"{', '.join(tried) or '无'}）"}],
                "summary": "轮次耗尽，未形成决策",
                "review": "",
                "transcript": transcript,
                "error": f"{rounds} 轮内未给出 final"}
    decisions = normalize_actions(final_obj.get("actions"), persona, fallback_code=None)
    # ⚠️ 阈值合理性校验需要**现价**：用当日收盘价（快照优先，零联网）——
    #    15:58 决策时 15:40 的快照已落库，故正常不需要联网。
    from core import ai_league
    _codes = sorted({d["code"] for d in decisions
                     if d.get("code") and d.get("code") != "-"})
    _px = ai_league.close_prices(_codes, trade_date) if _codes else {}
    _invalid: list[str] = []
    ensure_stops(decisions, persona, prices=_px, record=_invalid)
    if _invalid:
        _logger.warning("[AI决策轮] %s 剔除无效唤醒条件 %d 条: %s",
                        name, len(_invalid), _invalid[:3])
    # ⚠️ `tool_calls` 要**出现在结果里**（2026-10-17）：这是"工具循环到底有没有被用"的
    #    唯一可观测量 —— 没有它就只能靠翻 llm_calls 反推（本轮就是这么发现的）。
    _logger.info("[AI决策轮] %s 完成：工具调用 %d 次，transcript %d 条",
                 name, _tool_calls, len(transcript))
    return {"persona": name, "status": "ok", "decisions": decisions,
            "summary": str(final_obj.get("summary") or ""),
            "review": str(final_obj.get("review") or ""),
            "transcript": transcript, "error": "",
            "tool_calls": _tool_calls,
            "tool_retry": _tool_forced and _tool_calls == 0}


# ---------------- 决策归一化与落库 ----------------

def normalize_actions(actions, persona: dict, *, fallback_code: str = None,
                      min_position: float = None) -> list[dict]:
    """把模型给的 actions 归一成统一形状，并**用代码夹紧仓位**（提示词管不住的部分代码兜）。

    - 未知名/缺 code 的条目丢弃（不让脏数据进库）；
    - buy/place_order 的仓位过 `clamp_plan`；
    - 一份 actions 里同一 code 的 buy 只保留一次（防模型重复列）。

    `min_position` 透传给 `clamp_plan`：阿深（首笔 3%）必须传自己的首笔比例，
    否则默认的 5% 最小有效仓位会把首笔顶到 5%、再被授权判据拒掉。
    """
    out: list[dict] = []
    seen_buy: set = set()
    limit = persona.get("max_position")
    for a in (actions or []):
        if not isinstance(a, dict):
            continue
        act = str(a.get("action") or "").strip().lower()
        code = str(a.get("code") or fallback_code or "").strip()
        if act in ("place_order", "buy"):
            if not code or code in seen_buy:
                continue
            seen_buy.add(code)
            plan = ai_agent.clamp_plan(a, persona.get("risk_bias"), limit,
                                       min_position=min_position)
            out.append({"action": "buy", "code": code,
                        "position_size": plan["position_size"],
                        "reason": str(a.get("reason") or "")})
        elif act == "sell":
            if code:
                out.append({"action": "sell", "code": code,
                            "reason": str(a.get("reason") or "")})
        elif act in ("set_wake", "set_wake_condition"):
            if code and a.get("kind"):
                # ⚠️ 阈值有两种口径，**两种都要留**（实测踩过：只留 threshold 会让
                #    随机对照组的 stop_pct/take_pct 静默丢失 ⇒ 条件成了空壳）：
                #    · threshold     绝对价（LLM 组常直接给价）
                #    · threshold_pct 相对持仓成本的百分比（随机组给的是幅度）
                pct = a.get("threshold_pct")
                if pct is None:
                    for k in ("stop_pct", "take_pct"):
                        if a.get(k) is not None:
                            pct = a[k]
                            break
                    if pct is not None:
                        # stop_loss 用负号语义统一到"相对成本的百分比"（正数表示幅度）
                        pct = abs(float(pct))
                out.append({"action": "set_wake", "code": code,
                            "kind": str(a["kind"]),
                            "threshold": a.get("threshold"),
                            "threshold_pct": pct,
                            "wake_at": a.get("wake_at"),
                            "reason": str(a.get("reason") or "")})
        elif act == "pass":
            out.append({"action": "pass", "code": code or None,
                        "reason": str(a.get("reason") or "")})
    return out


# 默认止损幅度（买入未给止损时的代码兜底）。
# ⚠️ 提示词要求"每个买入必须同时给止损"，但**实测模型并不总是遵守**
#    （2026-10-14：ai_value 买了 600028 却没给任何止损条件）。
#    项目铁律：提示词管不住的部分由代码兜 ⇒ 这里补一个保守的默认止损并标记来源。
DEFAULT_STOP_PCT = 0.08


def ensure_stops(decisions: list[dict], persona: dict, *, prices: dict = None,
                 record: list = None) -> list[dict]:
    """保证"每个买入都有**有效**止损"（**原地修改并返回**）。

    两步，**顺序不可颠倒**：
      ① **剔除无效**的持仓类唤醒条件（阈值不可能触发，判据见
         `ai_agent.wake_threshold_ok`）。不先剔的话，一条"永不触发的止损"会让
         第 ② 步误判"已有止损"而跳过兜底 ⇒ **持仓裸奔**
         （2026-10-15 实测：`ai_tech` 给贵州茅台现价 1312.97 写 `stop_loss threshold=5.0`，
         `ai_wake` 判它无效不触发，而 `ensure_stops` 又因"已存在 stop_loss"不补 −8%）。
      ② 给"有买入但没有**有效**止损"的标的补 −8% 兜底（带来源标记）。

    Args:
        prices: `{code: 现价}`（当日收盘价）。缺失/为空时不删任何条件（判不了就不删）。
        record: 传入 list 时会把被剔除的原因 append 进去（供调用方记日志/留痕）。
    """
    prices = prices or {}
    # ① 剔除无效的持仓类条件
    kept: list[dict] = []
    for d in (decisions or []):
        if d.get("action") == "set_wake" and d.get("kind") in ("stop_loss", "take_profit"):
            code = str(d.get("code") or "")
            ok, why = ai_agent.wake_threshold_ok(d.get("kind"), d.get("threshold"),
                                                 prices.get(code))
            if not ok:
                if record is not None:
                    record.append(f"{code}/{d.get('kind')} 阈值 {d.get('threshold')}: {why}")
                continue
        kept.append(d)
    decisions[:] = kept

    # ② 补兜底
    bought = {d["code"] for d in decisions
              if d.get("action") == "buy" and d.get("code")}
    has_stop = {d["code"] for d in decisions
                if d.get("action") == "set_wake"
                and d.get("kind") == "stop_loss" and d.get("code")}
    for code in sorted(bought - has_stop):
        decisions.append({
            "action": "set_wake", "code": code, "kind": "stop_loss",
            "threshold": None, "threshold_pct": DEFAULT_STOP_PCT,
            "wake_at": None,
            "reason": f"⚠️ 系统兜底：该买入未附带止损，按 -{DEFAULT_STOP_PCT:.0%} 设定"})
    return decisions


def persist_round(result: dict, persona: dict, *, trade_date: str) -> dict:
    """把一轮的结果落库：`ai_decisions`（每 code 一行）+ `ai_wake_conditions`。

    ⚠️⚠️ 两类产出必须分开写（2026-10-14 实战抓到的缺陷）：
    `ai_decisions` 的主键是 (persona, signal_date, code)，而一轮里同一只票常同时有
    `buy` 与 `set_wake`（买入必带止损）⇒ 若两者都写决策表，**后写的 set_wake 会覆盖
    先写的 buy**，买入意图直接丢失（决策行变成"等待/0 仓位"）。
    ⇒ 决策表只由**交易动作**（buy/sell）产生；set_wake 只写条件表。
    """
    name = persona["name"]
    n_dec = n_wake = 0

    # ⓪ 自述复盘（独立表，不与决策争主键）。空文本也写一条，让"今天没写复盘"可见。
    try:
        store.save_ai_review(name, trade_date,
                             review_text=str(result.get("review") or ""),
                             summary=str(result.get("summary") or ""),
                             model=persona.get("model"))
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[AI决策轮] %s 复盘落库失败: %s", name, e)

    # ① 唤醒条件（独立于决策，不争主键）
    for d in (result.get("decisions") or []):
        if d.get("action") != "set_wake" or not d.get("code"):
            continue
        store.upsert_ai_wake_condition(name, {
            "code": d["code"], "kind": d["kind"], "threshold": d.get("threshold"),
            "threshold_pct": d.get("threshold_pct"),
            "wake_at": d.get("wake_at"), "note": d.get("reason")})
        n_wake += 1

    # ② 决策（只记交易动作）
    for d in (result.get("decisions") or []):
        if d.get("action") not in ("buy", "sell") or not d.get("code"):
            continue
        row = {"persona": name, "signal_date": trade_date, "code": d["code"],
               "advice_status": {"buy": "可执行", "sell": "卖出"}[d["action"]],
               "direction": {"buy": "看多", "sell": "看空"}[d["action"]],
               "position_size": d.get("position_size") if d["action"] == "buy" else 0.0,
               "model": persona.get("model"),
               "order_reason": d.get("reason") or ""}
        if store.save_ai_decision(row):
            n_dec += 1

    # ③ 没有实质动作时留一行总账（让"今天为什么不动"可复盘）。
    #    判据是"有没有 buy/sell"，不是"有没有过任何决策条目"。
    substantive = [d for d in (result.get("decisions") or [])
                   if d.get("action") in ("buy", "sell") and d.get("code")]
    if not substantive:
        reason = next((d.get("reason") for d in (result.get("decisions") or [])
                       if d.get("action") == "pass" and d.get("reason")), None) \
            or result.get("summary") or "本轮无实质动作"
        store.save_ai_decision({"persona": name, "signal_date": trade_date,
                                "code": "-", "advice_status": "等待",
                                "position_size": 0.0, "model": persona.get("model"),
                                "order_reason": reason})
        n_dec += 1
    return {"decisions": n_dec, "wake_conditions": n_wake}


def run_daily_round(trade_date: str = None, *, max_rounds: int = 6,
                    dry_run: bool = False, client_factory=None,
                    analysis_runner=None, personas: list = None,
                    consume_budget: bool = True) -> dict:
    """盘后决策轮编排：随机组（零 LLM）+ LLM 组 → 落库。

    `consume_budget=False` = **人工补跑**（页面/脚本手动触发）：
      · 不扣 decision_round 额度 ⇒ **不会把当天 15:58 的正式盘后轮挤掉**；
      · 产物仍是 UPSERT（同 (persona, signal_date, code) 覆盖），
        所以正式轮跑完会**覆盖**补跑结果，以正式轮为准。

    Returns: {"trade_date", "briefing_warnings", "results": [...], "summary": {...}}
    """
    day = trade_date or _today()
    common = build_common_briefing(day)
    all_personas = personas if personas is not None else ai_personas.load_personas()
    results: list[dict] = []

    # ① 随机对照组：规则决策，**零 LLM**
    for p in [x for x in all_personas if x["kind"] == "random" and x["enabled"]]:
        if consume_budget:
            granted, view = ai_budget.consume(p["name"], "decision_round", trade_date=day)
            if not granted:
                results.append({"persona": p["name"], "status": "skipped",
                                "decisions": [], "summary": "", "review": "", "transcript": [],
                                "error": f"decision_round 额度不足（{view}）"})
                continue
        priv = _private_overlay(p["name"], day)
        import random as _r
        acts = ai_agent.random_decisions(
            p, candidates=common["candidates"],
            holdings=[h["code"] for h in priv["holdings"]],
            rng=_r.Random(f"{p['name']}|{day}"))          # 按 名字+日期 定种子 ⇒ 可复现
        results.append({"persona": p["name"], "status": "ok",
                        "decisions": normalize_actions(acts, p),
                        "summary": "随机对照组（规则决策）",
                        "review": "随机对照组：规则生成，无自述复盘。",
                        "transcript": [], "error": ""})

    # ② LLM 组
    #    ⚠️ 必须排除 `league_eligible=False`（阿深）：它走 `core/ai_routine.py`
    #    的盘中定时通道。不过滤的后果是同一个账户一天被两套逻辑各喂一次决策
    #    （花两份钱、产两套互相矛盾的持仓计划），而且两边都"看起来正常"。
    for p in [x for x in all_personas
              if x["kind"] == "llm" and x["enabled"]
              and x.get("league_eligible", True)]:
        client = client_factory(p) if client_factory else None
        results.append(run_persona_round(p, common, trade_date=day, client=client,
                                         analysis_runner=analysis_runner,
                                         max_rounds=max_rounds,
                                         consume_budget=consume_budget))

    # ③ 落库（dry_run 只演练不写）。
    #    ⚠️ `no_final`（轮次耗尽）**也要落库**——它有降级产出的 pass 行，
    #    让复盘知道"这个 AI 研究了但没定论"；只有 error/skipped 才什么都不写。
    persisted = {}
    if not dry_run:
        for r in results:
            p = ai_personas.get_persona(r["persona"]) or {"name": r["persona"]}
            if r["status"] in ("ok", "no_final"):
                persisted[r["persona"]] = persist_round(r, p, trade_date=day)

    ok = sum(1 for r in results if r["status"] == "ok")
    return {"trade_date": day,
            "briefing_warnings": common["warnings"],
            "results": results,
            "summary": {"total": len(results), "ok": ok,
                        "no_final": sum(1 for r in results
                                        if r["status"] == "no_final"),
                        "skipped": sum(1 for r in results if r["status"] == "skipped"),
                        "failed": sum(1 for r in results
                                      if r["status"] not in ("ok", "no_final", "skipped")),
                        "persisted": persisted,
                        "dry_run": dry_run}}
