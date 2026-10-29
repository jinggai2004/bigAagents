# -*- coding: utf-8 -*-
"""盘后批量深度分析（P0，2026-10-19）—— 把「自动深度分析」真正接进每日流程。

为什么需要它
────────────
`core/pipeline.py` 里 `_stage_analyze_llm()` 是完整的（9 个 Agent 节点 + 落 `analyses`），
但它被 `if mode == "llm"` 挡住，而 `JOBS` 里**没有任何 mode="llm" 的任务**
⇒ 实际跑过的分析全部来自 UI 手点（`analyses.source` 只有 `ui`/`rerun`，`pipeline` **0 条**）。
于是"深度分析胜率"永远只有个位数样本，任何结论都不成立。

为什么不复用 `DailyPipeline(mode="llm")`
──────────────────────────────────────
那会**再跑一遍选股**：09:25 / 11:30 / 15:30 已经各跑一次，18:10 再来一次
就是第 4 次重复落 `screen_picks`（实测 10-19 当天 20 行只有 8 只票）。
本模块只做"对**已有候选**跑分析"，复用 UI 用的同一个入口 `run_analysis()`，
行为与手点完全一致，且天然幂等。

三条闸门（缺一不可）
────────────────────
1. **去重**：`(analysis_date, stock_code)` 已有分析 ⇒ 跳过（表上有唯一索引，双保险）；
2. **预算**：当日 `llm_calls` 累计 token 超过 `BIGA_LLM_DAILY_TOKEN_CAP`（默认 80 万）⇒ 停；
3. **只数**：最多 `BIGA_LLM_ANALYZE_MAX`（默认 8）只，按分数取前 N。

只数和预算都可以用环境变量调；**默认值按实测成本定**（单只 ≈ 3.4 万 token / ¥0.23 高峰）。
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime

from db import store

from core import signal_queue

_logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 8
# 2026-10-19 实测：8 只票一次跑完 **442,702 token**（14.5 分钟，¥1.50 空闲价）。
# 上限按"8 只 + 聊天/联赛余量"给到 120 万；它不是节流阀，是防止跑飞的安全网。
DEFAULT_TOKEN_CAP = 1_200_000
ENV_LIMIT = "BIGA_LLM_ANALYZE_MAX"
ENV_TOKEN_CAP = "BIGA_LLM_DAILY_TOKEN_CAP"

# ── 买入信号票（2026-10-23）──
# 盘中检测到"达到买入条件"的票要优先做深度分析（用户要的链路：
# 信号 → 深度分析 → **可行才通知**）。额度上限独立可调。
# ⚠️ 它占用的是**同一份** `want` 额度（信号票排前面、候选补位）⇒ 不增加成本。
ENV_SIGNAL_LIMIT = "BIGA_ANALYZE_SIGNAL_MAX"
DEFAULT_SIGNAL_LIMIT = 5

# ── 软冷却（2026-10-21）──
# ⚠️ 为什么是"加权"而不是"排除"：实测硬冷却会把池子打到 1 只
#    （10-21 当日池 7 只、7 个交易日内已分析 6 只 ⇒ 只剩 1 只可选），
#    离"每日分析 10-15 只"的目标更远。软冷却只改变**排序**，不会把候选清零。
ENV_COOLDOWN_DAYS = "BIGA_ANALYZE_COOLDOWN_DAYS"
DEFAULT_COOLDOWN_DAYS = 7
WEIGHT_SCORE = 0.6      # 技术分权重
WEIGHT_FRESH = 0.4      # 新鲜度权重
# ⚠️ 触发豁免：**加分项**，不是开关（2026-10-22 调低 0.5 → 0.2）。
#    0.5 相当于 0.5/0.6×100 ≈ **83 分技术分差**：50 分触发票（0.3+0.4+0.5=1.2）
#    能压制 95 分老票（0.57+0.4=0.97）—— 那是"开关"不是"加分"。
#    0.2 相当于 33 分，够让触发票跨过一档，但压不倒明显更高分的票。
TRIGGER_BONUS = 0.2


def _today(date: str | None = None) -> str:
    return (date or datetime.now().strftime("%Y-%m-%d"))[:10]


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        _logger.warning("[盘后分析] %s=%r 不是整数，按默认 %s 处理", name, raw, default)
        return int(default)


def _cost_cap() -> float:
    """当日金额上限（元）。0 = 关闭闸门。

    ⚠️ 必须**每次调用都读** `config.LLM_DAILY_BUDGET_CNY`，不能写成
    `def f(cap=config.LLM_DAILY_BUDGET_CNY)` —— 默认参数在定义时求值，
    改 .env 后用 `reload_config()` 热重载也**不会**生效（2026-10-23 那次
    `label_posts(max_posts=MAX_POSTS)` 踩的就是这个坑，一红一绿两个假用例）。
    """
    try:
        import config as _cfg
        return float(getattr(_cfg, "LLM_DAILY_BUDGET_CNY", 0.0) or 0.0)
    except Exception as e:                              # noqa: BLE001
        _logger.warning("[盘后分析] 读取日预算失败，本期不启用成本闸门: %s", e)
        return 0.0


def candidates_for_date(date: str | None = None, want: int = 24) -> list[dict]:
    """当日候选（按分数降序、**按代码去重**）。

    ⚠️ 必须去重：一天三次选股会各落一次 `screen_picks`（同日 20 行 / 8 只票是常态），
    直接拿来当候选会同一只票分析多遍。这里按代码取 MAX(score)，保证一只票只出现一次。
    """
    d = _today(date)
    store.init_store()
    conn = store.get_connection()
    try:
        rows = conn.execute(
            """SELECT stock_code, MAX(score) AS score,
                      MAX(COALESCE(rank_in_run, 999)) AS rank_in_run,
                      MAX(stock_name) AS stock_name
                 FROM screen_picks
                WHERE pick_date = ?
                GROUP BY stock_code
                ORDER BY score DESC, rank_in_run ASC
                LIMIT ?""", (d, int(want))).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def analyzed_codes(date: str | None = None) -> set[str]:
    """当日**已分析过**的代码（唯一索引保证一票一行）。"""
    d = _today(date)
    store.init_store()
    conn = store.get_connection()
    try:
        return {str(r[0]) for r in conn.execute(
            "SELECT stock_code FROM analyses WHERE analysis_date = ?", (d,))}
    finally:
        conn.close()


def recently_analyzed(date: str | None = None,
                      days: int = DEFAULT_COOLDOWN_DAYS) -> dict[str, int]:
    """最近 `days` 个**自然日**内分析过的代码 → 距今天数（0 = 今天）。

    ⚠️ 用自然日而不是交易日：它只用于"新鲜度"加权，差一两天不影响排序；
       换交易日要读交易日历，而日历缓存过期会联网（历史上因此挂起过）。
    """
    d = _today(date)
    store.init_store()
    conn = store.get_connection()
    try:
        rows = conn.execute(
            "SELECT stock_code, MAX(analysis_date) AS last_date FROM analyses "
            "WHERE analysis_date >= date(?, ?) AND analysis_date <= ? "
            "GROUP BY stock_code",
            (d, f"-{int(days)} days", d)).fetchall()
    finally:
        conn.close()
    out: dict[str, int] = {}
    try:
        cur = datetime.strptime(d, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        _logger.warning("[盘后分析] 基准日期无法解析（%r），本次不做新鲜度加权", d)
        return out
    for r in rows:
        try:
            last = datetime.strptime(str(r["last_date"])[:10], "%Y-%m-%d").date()
        except (TypeError, ValueError) as e:      # 日期脏数据：跳过这一行，不中断整体
            _logger.warning("[盘后分析] 跳过无法解析的分析日期 %r：%s", r["last_date"], e)
            continue
        out[str(r["stock_code"])] = max(0, (cur - last).days)
    return out


def trigger_exempt_codes(date: str | None = None,
                         days: int = DEFAULT_COOLDOWN_DAYS) -> dict[str, str]:
    """最近 `days` 天内**已触及目标/止损**的代码 → 触发原因（冷却豁免用）。

    ⚠️ 只做 `hit_target` / `hit_stop` 两类：这两个字段由
       `check_triggers_daily`（每日 15:35）与 T+N 回填写入，零新增成本。
       "当日涨幅/量比异动""行业 verdict 反转"需要行情与评审数据，见设计文档 §十二。

    ⚠️ 注意：这里**不判"是不是今天才触发"**——`analyses` 只存 0/1 状态，没有
       触发的具体日期。用"最近 N 天内分析过且已触及"做近似，对排序够用
       （已触及的票本来就该优先复核）。
    """
    d = _today(date)
    store.init_store()
    conn = store.get_connection()
    try:
        rows = conn.execute(
            "SELECT stock_code, MAX(COALESCE(hit_target,0)) AS t, "
            "       MAX(COALESCE(hit_stop,0)) AS s "
            "  FROM analyses "
            " WHERE analysis_date >= date(?, ?) AND analysis_date <= ? "
            " GROUP BY stock_code HAVING t = 1 OR s = 1",
            (d, f"-{int(days)} days", d)).fetchall()
        return {str(r["stock_code"]): ("触及目标" if r["t"] else "触及止损") for r in rows}
    finally:
        conn.close()


def rank_candidates(cands: list[dict], history: dict[str, int],
                    triggers: dict[str, str] | None = None, *,
                    weight_score: float = WEIGHT_SCORE,
                    weight_fresh: float = WEIGHT_FRESH,
                    trigger_bonus: float = TRIGGER_BONUS,
                    cooldown_days: int = DEFAULT_COOLDOWN_DAYS) -> list[dict]:
    """按「技术分 + 新鲜度 + 触发豁免」给候选排序（**软冷却**，2026-10-21）。

    背景：原实现只 `ORDER BY score DESC LIMIT N`，而高分股的因子是稳定的
    ⇒ 同一批票天天上榜。实测重合度 10-20 为 40%、10-21 为 56%（5/9 重复）。

    设计要点（三条都是踩过的坑）：

    ⚠️ 1. **只排序、不排除**：硬冷却在池子小的时候会把候选打到 1 只
          （10-21 池子 7 只、7 日内已分析 6 只）。排序则永远不会清零。
    ⚠️ 2. **触发是加分项，不是开关**：连续涨停的票会在冷却期内，
          但恰恰最该跟踪 —— 加 `trigger_bonus` 让它能翻上来。
    ⚠️ 3. **新鲜度用"越久没看分越高"**：`min(1, 距今天数/冷却窗口)`，
          新面孔（从未分析）直接给 1.0。

    Args:
        cands: `candidates_for_date` 的输出（需含 `stock_code` / `score`）
        history: `{代码: 距今天数}`，来自 `recently_analyzed`
        triggers: `{代码: 原因}`，来自 `trigger_exempt_codes`
        cooldown_days: 计算新鲜度的窗口（>=1，防止除零）

    返回**新列表**（不改入参），按 `_priority` 降序；每项额外带
    `_priority` / `_fresh` / `_trigger` 三个下划线字段，便于 dry-run 核对。
    """
    triggers = triggers or {}
    window = max(1, int(cooldown_days))
    out: list[dict] = []
    for c in cands:
        code = str(c.get("stock_code") or "")
        if not code:
            continue
        try:
            score = float(c.get("score") or 0)
        except (TypeError, ValueError):
            # `score` 可能是 "N/A" / None（`score_status=not_screened` 的候选）
            _logger.debug("[软冷却] %s 的 score 无法解析（%r），按 0 处理",
                          code, c.get("score"))
            score = 0.0
        ago = history.get(code)
        fresh = 1.0 if ago is None else min(1.0, max(0.0, ago / float(window)))
        pri = weight_score * (score / 100.0) + weight_fresh * fresh
        hit = triggers.get(code) or ""
        if hit:
            pri += trigger_bonus
        item = dict(c)
        item["_priority"] = round(pri, 4)
        item["_fresh"] = None if ago is None else ago
        item["_trigger"] = hit
        out.append(item)
    out.sort(key=lambda x: (-x["_priority"], str(x.get("stock_code"))))
    return out


def tokens_used_today(date: str | None = None) -> int:
    """当日 `llm_calls` 累计 token（用于预算闸门）。"""
    d = _today(date)
    store.init_store()
    conn = store.get_connection()
    try:
        row = conn.execute(
            """SELECT COALESCE(SUM(COALESCE(prompt_tokens,0)
                                  + COALESCE(completion_tokens,0)), 0)
                 FROM llm_calls WHERE substr(created_at, 1, 10) = ?""", (d,)).fetchone()
        return int(row[0] or 0)
    finally:
        conn.close()


def cost_used_today(date: str | None = None) -> float:
    """当日累计金额（元）。与 token 闸门并列，理由见 `run_evening_analysis` 里的说明。

    ⚠️ 金额**不能由 token 换算**：同一 token 数，qwen3.7-flash(0.2/0.8) 与
    deepseek-v4-pro 高峰(9/27) 相差 45 倍 ⇒ "token 没超"不代表"钱没超"。
    """
    from core.llm_cost import today_cost
    d = _today(date)
    try:
        return float(today_cost(datetime.strptime(d, "%Y-%m-%d")))
    except Exception as e:                              # noqa: BLE001
        _logger.warning("[盘后分析] 读取当日成本失败（本次不做成本闸门）: %s", e)
        return 0.0


def run_evening_analysis(date: str | None = None, limit: int | None = None,
                         dry_run: bool = False, source: str = "pipeline",
                         runner=None) -> dict:
    """对当日已有候选跑批量深度分析。

    Args:
        date: 分析日（默认今天；跨日重放用）。
        limit: 最多分析几只（默认 `BIGA_LLM_ANALYZE_MAX` 或 8）。
        dry_run: 只列"会分析哪些"，不真调用（零成本，用于上线前核对）。
        source: 落库来源（默认 `pipeline`）。
        runner: 注入测试用（默认 `core.analysis_runner.run_analysis`）。

    Returns:
        {"date", "limit", "candidates", "analyzed", "skipped_existing",
         "skipped_reason", "failed", "tokens_before", "token_cap", "elapsed_s"}
    """
    d = _today(date)
    cap = _int_env(ENV_TOKEN_CAP, DEFAULT_TOKEN_CAP)
    want = int(limit or _int_env(ENV_LIMIT, DEFAULT_LIMIT))
    t0 = time.perf_counter()

    # 多取一些：软冷却会重排，取 10 倍保证够挑
    # ⚠️⚠️ 2026-10-22 实测：原来是 `want * 3`（want=8 ⇒ 只取 24 只），而
    #    扩池后 `screen_picks` 每天有 100+ 只 ⇒ **软冷却只能在这 24 只老面孔里重排**，
    #    池子扩大等于白做（日志实测："候选 24 只，12 只在冷却窗口内"）。
    #    改成 10 倍后 8 只分析 ⇒ 从 80 只里挑，新面孔才进得来。
    #    `rank_candidates` 是纯内存排序，80 只vs24 只的代价可忽略。
    raw = candidates_for_date(d, want=max(want * 10, 40))
    done = analyzed_codes(d)
    # 软冷却（2026-10-21）：只重排、不排除 —— 硬冷却在池子小的时候会把候选打到 1 只
    cooldown = _int_env(ENV_COOLDOWN_DAYS, DEFAULT_COOLDOWN_DAYS)
    hist = recently_analyzed(d, days=cooldown)
    trig = trigger_exempt_codes(d, days=cooldown)
    cands = rank_candidates(raw, hist, trig, cooldown_days=cooldown)
    todo = [c for c in cands if str(c["stock_code"]) not in done][:want]
    # ── 买入信号票优先（2026-10-23）──
    # 用户要的链路是"**达到买入条件/信号后深度分析，可行再通知**"。
    # 信号票来自盘中检测（`core.signal_queue`），它们**优先**占额度，
    # 但**总数仍受 want 限制** ⇒ 不增加 LLM 成本（只是换掉了排位靠后的候选）。
    sig_codes = signal_queue.pending_codes(d)
    sig_limit = _int_env(ENV_SIGNAL_LIMIT, DEFAULT_SIGNAL_LIMIT)
    sig_todo = [{"stock_code": c, "_signal": True} for c in sig_codes
                if c not in done][:sig_limit]
    if sig_todo:
        todo = (sig_todo + todo)[:want]
        _logger.info("═══ [盘后分析] 买入信号票 %d 只优先纳入：%s ═══",
                     len(sig_todo), "、".join(str(c["stock_code"]) for c in sig_todo))
    # ── 情绪面票（P1-5，2026-10-23）──
    # 用户要的是"**从当日利好新闻涉及的股票里选**"：新闻命中（严匹配）里
    # 正向占多、且当天还没分析过的票，固定占 2 席。
    # ⚠️ 为什么不是"合并进评分排序"：这些票的**因子分可能是 None**（不在候选池），
    #    并进排序会被永远排到末尾 —— 那这个通道就等于不存在。
    # ⚠️ 为什么固定席位数而不是全给：它是**补充视角**，不是要挤掉技术面选股；
    #    数量由 `BIGA_NEWS_PICK_SLOTS` 控制（默认 2，设为 0 即关闭）。
    news_slots = _int_env("BIGA_NEWS_PICK_SLOTS", 2)
    if news_slots > 0:
        try:
            from core import news_stock_hits as _nsh
            picked = {str(c["stock_code"]) for c in todo}
            hits = [h for h in _nsh.top_hits(d, limit=40)
                    if int(h.get("positive") or 0) > int(h.get("negative") or 0)
                    and str(h["stock_code"]) not in done
                    and str(h["stock_code"]) not in picked]
            news_todo = [{"stock_code": str(h["stock_code"]),
                          "_news": True,
                          "_news_hits": int(h.get("hits") or 0)}
                         for h in hits[:news_slots]]
            if news_todo:
                todo = (news_todo + todo)[:want]
                _logger.info("═══ [盘后分析] 情绪面票 %d 只纳入：%s ═══", len(news_todo),
                             "、".join(f"{c['stock_code']}(新闻{c['_news_hits']}次)"
                                       for c in news_todo))
        except Exception as e:                              # noqa: BLE001
            _logger.warning("═══ [盘后分析] 情绪面票读取失败（不影响主流程）: %s ═══", e)

    _logger.info("═══ [盘后分析] 候选 %d 只（软冷却 %d 天：%d 只在冷却窗口内、%d 只触发豁免）═══",
                 len(cands), cooldown,
                 sum(1 for c in cands if c.get("_fresh") is not None),
                 sum(1 for c in cands if c.get("_trigger")))

    # 不注入 runner 时才 import（测试不会碰真模型）
    if runner is None:
        from core.analysis_runner import run_analysis as runner  # noqa: PLC0415

    stats = {
        "date": d, "limit": want, "dry_run": bool(dry_run),
        "candidates": len(cands),
        "skipped_existing": sorted(str(c["stock_code"]) for c in cands
                                   if str(c["stock_code"]) in done),
        "analyzed": [], "failed": [], "skipped_reason": None,
        "tokens_before": tokens_used_today(d), "token_cap": cap,
        "cost_before": cost_used_today(d), "cost_cap": _cost_cap(),
        "elapsed_s": 0.0,
    }

    if not cands:
        stats["skipped_reason"] = "当日无选股候选"
    elif not todo:
        stats["skipped_reason"] = "候选已全部分析过"
    elif stats["tokens_before"] >= cap:
        stats["skipped_reason"] = f"当日 token 已达上限（{stats['tokens_before']} ≥ {cap}）"
    elif stats["cost_cap"] > 0 and stats["cost_before"] >= stats["cost_cap"]:
        # 金额闸门（2026-10-23）：与 token 闸门**并列而非替代**。
        # token 达标但便宜模型没花多少钱时不该拦；反过来 token 没超但撞上 pro 高峰
        # 已经把预算烧光时必须拦。两者都只对"批量跑"生效，盘中风控不受影响。
        stats["skipped_reason"] = (
            f"当日成本已达预算（¥{stats['cost_before']:.2f} ≥ ¥{stats['cost_cap']:.2f}）")
    if stats["skipped_reason"]:
        stats["elapsed_s"] = round(time.perf_counter() - t0, 1)
        _logger.info("═══ [盘后分析] 跳过：%s（候选 %d，已分析 %d）═══",
                     stats["skipped_reason"], stats["candidates"], len(done))
        _record(stats)
        return stats

    if dry_run:
        stats["analyzed"] = [str(c["stock_code"]) for c in todo]
        stats["elapsed_s"] = round(time.perf_counter() - t0, 1)
        _logger.info("═══ [盘后分析][dry-run] 将分析 %s ═══", stats["analyzed"])
        return stats

    sig_snapshots: dict = {}
    for c in todo:
        code = str(c["stock_code"])
        # 预算闸门**每只前**再查一次：前面几只的实际消耗可能超预期
        used = tokens_used_today(d)
        if used >= cap:
            stats["skipped_reason"] = f"token 达上限（已用 {used} ≥ {cap}），提前停止"
            _logger.warning("═══ [盘后分析] %s ═══", stats["skipped_reason"])
            break
        if stats["cost_cap"] > 0:
            spent = cost_used_today(d)
            if spent >= stats["cost_cap"]:
                stats["skipped_reason"] = (
                    f"成本达预算（已花 ¥{spent:.2f} ≥ ¥{stats['cost_cap']:.2f}），提前停止")
                _logger.warning("═══ [盘后分析] %s ═══", stats["skipped_reason"])
                break
        try:
            out = runner(code, source=source)
            if isinstance(out, dict) and out.get("ok") is False:
                stats["failed"].append({"code": code,
                                        "error": str(out.get("error"))[:200]})
            else:
                # ⚠️⚠️ 2026-10-22 实测：`run_analysis` 的 `ok` 只判"有没有抛异常"。
                #    LLM 中途连接失败时（`Connection error.`），链路后段的
                #    研究汇总 / 交易员 / 风控会**空转**，产出
                #    `final_decision=""` + `trade_plan={}`，但 `ok` 仍是 True
                #    ⇒ 统计成"成功 8 / 失败 0"，其中 600900 长江电力实际是空壳。
                #    这里补一道"有没有结论"的判据：两处都空就记失败
                #    （记录本身保留，`errors` 里有失败原因可追溯）。
                _snap = (out or {}).get("snapshot")
                if c.get("_signal") and isinstance(_snap, dict):
                    sig_snapshots[code] = _snap
                if _snap is None:
                    # 非标准 runner（测试注入/别的实现）不返回 snapshot ⇒ 不判失败，
                    # 否则会把"结构不同"误伤成"分析失败"。
                    stats["analyzed"].append(code)
                elif _snap.get("final_decision") or (_snap.get("trade_plan") or {}).get("direction"):
                    stats["analyzed"].append(code)
                else:
                    stats["failed"].append(
                        {"code": code, "error": "分析链未产出结论（final_decision 为空，"
                                                "多为 LLM 调用失败）"})
                    _logger.warning("[盘后分析] %s 未产出结论（判失败，记录已保留）", code)
        except Exception as e:                              # noqa: BLE001
            # 单只失败不能拖垮整批（这正是"批量"与"手点"的区别）
            _logger.warning("[盘后分析] %s 分析异常: %s", code, e)
            stats["failed"].append({"code": code, "error": f"{type(e).__name__}: {e}"[:200]})

    stats["tokens_after"] = tokens_used_today(d)
    stats["elapsed_s"] = round(time.perf_counter() - t0, 1)
    # ── 买入信号票：分析完 ⇒ **只有结论"可执行"才通知**（2026-10-23）──
    if sig_snapshots:
        try:
            signal_queue.mark_analyzed(d, list(sig_snapshots.keys()))
            _notify_signal_conclusions(d, sig_snapshots)
        except Exception as e:                              # noqa: BLE001
            _logger.warning("[盘后分析] 信号结论通知失败（不影响分析结果）: %s", e)
    _logger.info("═══ [盘后分析] 完成：成功 %d / 失败 %d，token %d → %d（上限 %d），%.1fs ═══",
                 len(stats["analyzed"]), len(stats["failed"]),
                 stats["tokens_before"], stats["tokens_after"], cap, stats["elapsed_s"])
    _record(stats)
    return stats


def _notify_signal_conclusions(day: str, snaps: dict) -> dict:
    """买入信号票的深度分析结论 → 飞书通知。

    ⚠️ **只推"可执行"档**（`advice_status == "可执行"`）：驳回/等待档不是行动
    信号，推了就是噪音（用户原话："如果可行再发通知"）。
    ⚠️ 结论判定用 `plan_view`（**唯一实现**，与复盘/决策汇总同口径）。
    """
    from core.decision_digest import plan_view
    from core import push_templates as T
    tpl_item = T.get_template(T.TPL_SIGNAL_ITEM)
    rows = []
    for code, snap in (snaps or {}).items():
        plan = plan_view((snap or {}).get("trade_plan") or {})
        if plan.get("advice_status") != "可执行":
            continue
        name = (snap or {}).get("stock_name") or code
        rows.append(T.render(tpl_item, {
            "icon": "🟢", "name": name, "code": code,
            "detail": "深度分析结论：可执行",
            "direction": plan.get("direction") or "",
            "entry": plan.get("entry"), "target": plan.get("target"),
            "stop": plan.get("stop"), "position_size": plan.get("position_size"),
            "data_through": (snap or {}).get("data_through") or "",
        }))
    if not rows:
        _logger.info("[盘后分析] 信号票结论均不可执行 ⇒ 不推送")
        return {"notified": 0}
    body = T.render(T.get_template(T.TPL_SIGNAL), {
        "title": "🟢 买入信号已确认（深度分析结论可行）",
        "date": day, "time": datetime.now().strftime("%H:%M"),
        "items": "\n".join(rows),
        "footer": T.FOOTERS.get(T.TPL_SIGNAL, ""),
    })
    try:
        from core.push import push_markdown
        pushed = push_markdown(body, title=f"bigAagents 买入信号 {day}",
                               dedupe_key=f"signal:{day}")
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[盘后分析] 信号结论推送失败: %s", e)
        return {"notified": 0, "error": str(e)}
    _logger.info("[盘后分析] 信号结论已推送 %d 条：%s", len(rows), pushed)
    return {"notified": len(rows), "push": pushed}


def _record(stats: dict) -> None:
    """写 `runs`：让"今天到底分析了几只"可查询，而不是只能翻日志。"""
    try:
        store.insert_run("job:evening_analysis", {
            "desc": "盘后批量深度分析",
            "ok": not stats.get("failed"),
            "date": stats.get("date"),
            "analyzed": len(stats.get("analyzed") or []),
            "failed": len(stats.get("failed") or []),
            "skipped_reason": stats.get("skipped_reason"),
            "tokens": stats.get("tokens_after") or stats.get("tokens_before"),
            "dry_run": stats.get("dry_run"),
        })
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[盘后分析] 运行记录写入失败（不影响主流程）: %s", e)
