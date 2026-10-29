# -*- coding: utf-8 -*-
"""阶段依赖与数据完整性（P0-5，2026-10-18）。

## 为什么需要它

`JOBS` 是一张**时刻表**（15:30 全流程 / 15:35 回填 / 15:40 快照 / 15:45 复盘 …），
`run_job` 只按 mode 分发，**不看上游是否成功**。后果：
  - 15:45 复盘跑在 15:50 选股回填**之前** ⇒ 复盘的"决策质量评估"永远看不到当天回填；
  - 15:40 快照失败时，15:45 复盘与 15:52 净值仍照跑 ⇒ 读到的是**昨天的截面**
    或"取不到价"的兜底值，而正文里不会写"今天这块数据其实是缺的"。

## 为什么**不改**成"单 worker 串行"

待办原文建议 `JOBS` 改阶段表 + 单 worker 串行。实测评估后**不可行**：
`intraday_watch` 的 cron 是 9-11/13-15 点**每分钟**触发；盘后长任务
（全流程 4.5min + 快照 9min + 复盘）在单 worker 上排队会把盘中盯盘压住 ——
用"改串行"换"复盘不读半截数据"，代价是丢掉盘中监控，得不偿失。

**本模块的形态**：保留多 worker，把依赖关系显式化：
  1. `STAGE_DEPS` —— 阶段 → 上游依赖（只登记真正有依赖的，其余默认无依赖）；
  2. `stage_readiness()` —— **判据是"上游产物是否在库里"**，不看日志/进程
     （日志说"完成"但库里没有 = 没完成，这是本项目反复踩的坑）；
  3. `data_integrity_markdown()` —— 复盘正文的"今日数据完整性"段，
     **由程序渲染**（不经模型），缺什么写什么，并明写"因 X 缺失，本次不评估 Y"。

⚠️ 判据只看"当日是否有产物行"，不判质量（质量由 `data_coverage` 的 ratio 负责）。
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime

_logger = logging.getLogger(__name__)

# 数据根必须**运行时**解析：打包后 `__file__` 位于只读包 `_internal/` 里，
# 用它推导会指向空目录 ⇒ 读到空数据（2026-10-25 实测：双击 exe 后
# 选股/复盘/研究全 missing、交易台变空账户、系统页看不到调度器）。
# 规则见 `config/paths.data_root()`；源码运行结果与原先一致。
from config.paths import data_root as _paths_data_root  # noqa: E402
_PROJECT_ROOT = _paths_data_root()

# 阶段依赖表：**只登记真正有依赖的阶段**，未登记的视为无依赖。
#   deps     —— 上游阶段 id（判据：上游**当日产物是否落库**）
#   blocking —— 上游缺失时是否**阻断本阶段**
#   critical —— 该阶段缺失会影响复盘结论（markdown 里点名并写明影响）
#
# ⚠️⚠️ `blocking` 与 `critical` 必须分开看：
#   - `ai_mark_nav`：没有当日快照就取不到收盘价 ⇒ **阻断**（跑也白跑）；
#   - `daily_review`：快照缺失只丢"行业资金热度"一章，其余（大盘/持仓/决策质量）
#     仍有效 ⇒ **不阻断**，改由正文的"今日数据完整性"段明写缺什么。
#     若这里也设阻断，等于"快照一挂，当天连复盘都没有"——比原来更糟。
STAGE_DEPS: dict[str, dict] = {
    # report_impact=True ⇒ 缺失时在复盘正文**点名并写明影响**。
    #   daily_review 自身不标（它就是"正在写的那份报告"，写"未使用每日复盘"是废话）；
    #   ai_daily_round 不标（非关键，只轻提示）。
    "snapshot_flow": {"deps": [], "blocking": False, "critical": True,
                      "report_impact": True, "label": "全市场资金流/估值截面"},
    # ⚠️ D2（2026-10-18 审查）：复盘**必须**能看到当天回填 —— 否则"决策质量评估"
    #    永远用上一批样本（原 JOBS 里复盘 15:45 早于回填 15:50）。已把回填提前到
    #    15:36、复盘后移到 15:50，并把 backfill_screen 登记为依赖。
    # ⚠️ 2026-10-28：15:36 **仍然不够早**——实测回填要跑 17.5 分钟（15:53:35 才结束），
    #    而复盘 15:50 就检查产物 ⇒ 每天仍然读到前一天的 T+N。回填已挪到 **15:10**，
    #    正文里的时刻不再硬编码，改由 `_job_time()` 从 JOBS 现读（见该函数注释）。
    "backfill_screen": {"deps": [], "blocking": False, "critical": False,
                        "report_impact": False, "label": "选股候选 T+N 回填"},
    # ⚠️ R3（2026-10-18 审查）：复盘还读 `output/daily_picks/latest.json`
    #    （= 15:30 盘后全流程 `afternoon_full` 的产物）。实测该流程典型 10~18 分钟、
    #    最坏 49 分钟 ⇒ 15:50 的复盘在多数日子**早于它完成**。
    #    `review_agent` 已校验 `latest["date"] == 今天`（不会拿昨日当今日），
    #    所以后果只是"当天复盘缺选股章节"——但**必须登记依赖**，否则
    #    "缺了"只能靠巧合被发现。
    "afternoon_full": {"deps": [], "blocking": False, "critical": True,
                       "report_impact": True, "label": "盘后全流程选股（daily_picks）"},
    "daily_review": {"deps": ["snapshot_flow", "backfill_screen", "afternoon_full"],
                     "blocking": False, "critical": True,
                     "report_impact": False, "label": "每日复盘"},
    "ai_mark_nav": {"deps": ["snapshot_flow"], "blocking": True, "critical": True,
                    "report_impact": True, "label": "AI 联赛当日净值"},
    "ai_daily_round": {"deps": ["ai_mark_nav"], "blocking": False, "critical": False,
                       "report_impact": False, "label": "AI 联赛决策轮"},
    # ⚠️ 午间轮（13:05）**不能依赖 ai_mark_nav**：当日净值 15:52 才写入，
    #    13:05 时必然"上游缺失"⇒ 每天被跳过，等于白加一个任务。
    "ai_midday_round": {"deps": [], "blocking": False, "critical": False,
                        "report_impact": False, "label": "AI 联赛午间决策轮"},
}

# 渲染"今日数据完整性"时**跳过自身**的阶段（D1，2026-10-18 审查）
# ⚠️⚠️ 事故：`data_integrity_markdown()` 在复盘**写文件之前**调用，而
#   `artifact_status("daily_review")` 的判据正是"该文件是否存在" ⇒ 正文里必然出现
#   「每日复盘 ❌ 缺失」，等于报告在说自己不存在。不是判据错，是**判据不适合评价自身**。
SELF_REFERENTIAL = {"daily_review"}

# 阶段产物缺失时，对复盘结论的**具体影响**（写进正文，避免笼统的"数据异常"）
_MISSING_IMPACT = {
    "snapshot_flow": "本次复盘**未评估行业资金热度**（行业资金流/估值截面缺失）",
    "ai_mark_nav": "本次复盘**未引用 AI 联赛当日净值**（净值未记录，避免用成本价冒充）",
    "afternoon_full": "本次复盘**未包含今日选股流水线**（`daily_picks/latest.json` 不是今天"
                      "或尚未生成 ⇒ 决策质量评估缺当日样本）",
}


def _today_str(trade_date: str | None = None) -> str:
    return str(trade_date or datetime.now().strftime("%Y-%m-%d"))


# ================================================================
# 产物判据（只读本地，零联网）
# ================================================================

def _snapshot_rows(trade_date: str) -> int:
    """当日 `market_snapshot` 行数（0 = 快照缺失）。"""
    try:
        from db import store
        _, rows = store.load_market_snapshot(trade_date)
        return len(rows)
    except Exception as e:                                # noqa: BLE001
        _logger.debug("[stage] 快照判据读取失败: %s", e)
        return 0


def _nav_rows(trade_date: str) -> int:
    """当日 `paper_nav` 行数（0 = 净值阶段没跑/全部取不到价）。"""
    try:
        from db import store
        store.init_store()
        conn = store.get_connection()
        try:
            r = conn.execute("SELECT COUNT(*) FROM paper_nav WHERE date = ?",
                             (trade_date,)).fetchone()
            return int(r[0] if r else 0)
        finally:
            conn.close()
    except Exception as e:                                # noqa: BLE001
        _logger.debug("[stage] 净值判据读取失败: %s", e)
        return 0


def _review_file(trade_date: str) -> str:
    """当日复盘文件路径（不存在返回空串）。"""
    d = trade_date.replace("-", "")
    path = os.path.join(_PROJECT_ROOT, "output", "daily_reports", f"review_{d}.md")
    return path if os.path.exists(path) else ""


def _ai_decision_rows(trade_date: str) -> int:
    try:
        from db import store
        conn = store.get_connection()
        try:
            # ⚠️⚠️ 列名是 `signal_date`，不是 `trade_date`（2026-10-23 修）：
            #   原写 `WHERE trade_date = ?` 时该表根本没有这一列 ⇒ SQL 报错被
            #   下面的 except 吞掉 ⇒ **恒返 0** ⇒ 复盘"今日数据完整性"里
            #   「AI 联赛决策轮」永远显示 ❌ 缺失，即使当天有 8 条决策。
            #   这正是"缺就是缺，不补造"的反面：把**查询失败**伪装成"没有数据"。
            r = conn.execute("SELECT COUNT(*) FROM ai_decisions WHERE signal_date = ?",
                             (trade_date,)).fetchone()
            return int(r[0] if r else 0)
        finally:
            conn.close()
    except Exception as e:                                # noqa: BLE001
        _logger.debug("[stage] AI 决策判据读取失败: %s", e)
        return 0


def _screen_backfill_rows(trade_date: str) -> int:
    """当日**回填过**的选股样本数（判据：`backfilled_at` 以当日开头）。

    ⚠️ 不能用 `pick_date == 当日`：回填的是**历史**样本（T+5/T+20 到期后才回填），
       当天回填的行 `pick_date` 可能是 10 天前。判据必须落在"什么时候回填的"。
    """
    try:
        from db import store
        # ⚠️ 按 (pick_date, stock_code) 去重：同一只票同一天会被回填多次（多轮 pipeline），
        #    按行数判"今日回填 N 条"会虚高 ~2.1 倍 ⇒ 复盘里的完整性指标失真。
        return store.count_screen_picks_dedup(
            "WHERE s.backfilled_at LIKE ?", (f"{trade_date}%",))
    except Exception as e:                                # noqa: BLE001
        _logger.debug("[stage] 选股回填判据读取失败: %s", e)
        return 0


def _job_ran_today(job_id: str, trade_date: str) -> dict | None:
    """该阶段今天**是否真的执行过**（`runs` 表 `job:<id>` 记录；R5，2026-10-18 审查）。

    为什么需要它：`backfill_screen` 的产物判据是"今日回填 N 条"，但"今天没有
    T+5/T+20 到期样本"是**正常情况**（周末/节后）——把"任务跑了但 0 条到期"
    判成缺失，会让复盘天天挂一条假告警。
    """
    try:
        from db import store
        conn = store.get_connection()
        try:
            r = conn.execute(
                "SELECT summary FROM runs WHERE run_type = ? AND created_at LIKE ? "
                "ORDER BY id DESC LIMIT 1",
                (f"job:{job_id}", f"{trade_date}%")).fetchone()
            if not r:
                return None
            try:
                return json.loads(r["summary"] or "{}")
            except (ValueError, TypeError) as e:          # noqa: BLE001
                _logger.debug("[stage] %s 执行记录内容不可解析: %s", job_id, e)
                return {}
        finally:
            conn.close()
    except Exception as e:                                # noqa: BLE001
        _logger.debug("[stage] %s 执行记录读取失败: %s", job_id, e)
        return None


def _job_time(job_id: str) -> str:
    """该任务在 `core.scheduler.JOBS` 里的**实际**排期（`HH:MM`），读不到返回 ""。

    ⚠️ 为什么不再硬编码（2026-10-28）：复盘正文里原来写死"回填任务排在 15:36"，
       而回填先挪到 15:36、后挪到 15:10，文案没跟着动 ⇒ 报告口口声声"早于本报告"，
       实际早已跑不完。这句话正是"复盘为什么看不到当天回填"的唯一线索，
       时刻只能有一个来源。
    """
    try:
        from core.scheduler import JOBS
        for jid, kw, _mode, _desc in JOBS:
            if jid != job_id:
                continue
            h, m = kw.get("hour", 0), kw.get("minute", 0)
            h = h[0] if isinstance(h, (list, tuple)) and h else h
            m = m[0] if isinstance(m, (list, tuple)) and m else m
            return f"{int(h):02d}:{int(m):02d}"
    except Exception as e:                                # noqa: BLE001
        _logger.debug("[stage] %s 排期读取失败: %s", job_id, e)
    return ""


def artifact_status(job_id: str, trade_date: str | None = None) -> dict:
    """某阶段**当日产物**是否存在。返回 `{"present": bool, "detail": str}`。"""
    d = _today_str(trade_date)
    if job_id == "snapshot_flow":
        n = _snapshot_rows(d)
        return {"present": n > 0, "detail": f"market_snapshot {d}：{n} 行"}
    if job_id == "ai_mark_nav":
        n = _nav_rows(d)
        return {"present": n > 0, "detail": f"paper_nav {d}：{n} 条"}
    if job_id == "daily_review":
        p = _review_file(d)
        return {"present": bool(p), "detail": p or f"review_{d.replace('-', '')}.md 不存在"}
    if job_id == "afternoon_full":
        p = os.path.join(_PROJECT_ROOT, "output", "daily_picks", "latest.json")
        try:
            with open(p, encoding="utf-8") as f:
                latest = json.load(f)
        except (OSError, ValueError) as e:                # noqa: BLE001
            _logger.debug("[stage] daily_picks/latest.json 不可读: %s", e)
            return {"present": False, "detail": "output/daily_picks/latest.json 不存在"}
        got = str(latest.get("date") or "")[:10]
        # ⚠️ F2（2026-10-18 审查）：**还必须看 mode** —— `pipeline._save()` 在
        #    4 个分支都写 latest.json（含 `mode="screen"`），09:25 盘前选股也会
        #    把 date 刷成今天 ⇒ 只比 date 会把**上午的候选**当"盘后流水线"。
        #    `afternoon_full` 的产物 mode 是 "rule"（llm 也算，同为盘后全流程）。
        mode = latest.get("mode")
        if mode not in ("rule", "llm"):
            return {"present": False,
                    "detail": f"latest.json 是 mode={mode!r}（非盘后全流程产物）"}
        return {"present": got == d,
                "detail": (f"latest.json 日期 {got}（mode={mode}）" if got != d
                           else f"daily_picks {got}（mode={mode}，今日盘后流水线）")}
    if job_id == "backfill_screen":
        n = _screen_backfill_rows(d)
        if n > 0:
            return {"present": True, "detail": f"当日回填 {n} 条选股样本"}
        # 双判据（R5）：产物为空时再看"任务是否真的跑过"——0 条到期 ≠ 没执行
        ran = _job_ran_today("backfill_screen", d)
        if ran is not None:
            note = ran.get("note") or ran.get("detail") or "无到期样本"
            return {"present": True, "detail": f"任务已执行，当日无回填产出（{note}）"}
        return {"present": False, "detail": "当日回填 0 条，且无执行记录"}     
    if job_id == "ai_daily_round":
        n = _ai_decision_rows(d)
        return {"present": n > 0, "detail": f"ai_decisions {d}：{n} 条"}
    return {"present": True, "detail": "无产物判据（视为就绪）"}


def stage_readiness(job_id: str, trade_date: str | None = None) -> dict:
    """某阶段**上游是否就绪**。

    Returns:
        {"ready": bool, "missing": [上游 job_id...], "checks": {job_id: detail},
         "blocking": bool}

    ⚠️ `blocking` 是**该阶段自身的**策略（见 STAGE_DEPS 注释）：
       - blocking=True 且上游缺失 ⇒ 调用方应**跳过**本阶段；
       - blocking=False ⇒ 照跑，但必须把缺失写进产物/正文（降级要如实）。
    """
    meta = STAGE_DEPS.get(job_id) or {}
    deps = meta.get("deps") or []
    checks: dict[str, str] = {}
    missing: list[str] = []
    for dep in deps:
        st = artifact_status(dep, trade_date)
        checks[dep] = st["detail"]
        if not st["present"]:
            missing.append(dep)
    return {"ready": not missing, "missing": missing, "checks": checks,
            "blocking": bool(meta.get("blocking"))}


# ================================================================
# 复盘正文：今日数据完整性（程序渲染，不经模型）
# ================================================================

def data_integrity_markdown(trade_date: str | None = None, *,
                            exclude_self: bool = True) -> str:
    """渲染"今日数据完整性"段。

    Args:
        exclude_self: 默认 True —— **排除"正在生成的这份报告"自身**（D1）。
            复盘正文里永远不该出现「每日复盘 ❌ 缺失」（写文件在渲染之后，
            判据必然为假 ⇒ 报告在说自己不存在）。
            审计/历史回看时传 False，可看到含自身在内的全貌。

    为什么必须由程序写：模型看到的是"喂给它的数据"，它**看不到**"没喂进去的那部分"——
    缺什么只有代码知道。这与"决策数字不经 LLM"是同一条原则。
    """
    d = _today_str(trade_date)
    lines = ["## 今日数据完整性", "",
             "> 由程序核对（不经模型）。只看**产物是否落库**，缺失项必须显式列出。", "",
             "| 阶段 | 状态 | 依据 |", "|---|---|---|"]
    missing_critical: list[str] = []
    missing_minor: list[str] = []
    for jid, meta in STAGE_DEPS.items():
        if exclude_self and jid in SELF_REFERENTIAL:
            continue
        st = artifact_status(jid, d)
        icon = "✅" if st["present"] else ("❌" if meta.get("critical") else "⚠️")
        lines.append(f"| {meta.get('label', jid)} | {icon} "
                     f"{'有' if st['present'] else '缺失'} | {st['detail']} |")
        if not st["present"]:
            if meta.get("report_impact"):
                missing_critical.append(jid)
            else:
                missing_minor.append(jid)
    lines.append("")
    # D2（2026-10-18 审查）：明确"决策质量评估用的是哪一批回填"
    _bf = artifact_status("backfill_screen", d)
    if _bf["present"]:
        _at = _job_time("backfill_screen")
        lines.append(f"ℹ️ 决策质量评估所用的选股回填：{_bf['detail']}"
                     f"（截止本报告生成时；回填任务排期 "
                     f"{_at or '未知'}，早于本报告）。")
    if missing_critical:
        for jid in missing_critical:
            impact = (_MISSING_IMPACT.get(jid)
                      or f"本次复盘未使用「{STAGE_DEPS[jid].get('label', jid)}」的数据")
            lines.append(f"⚠️ {impact}。")
    if missing_minor:
        labels = "、".join(STAGE_DEPS[j].get("label", j) for j in missing_minor)
        lines.append(f"ℹ️ 另有非关键产物缺失：{labels}（不改变本报告主体结论）。")
    if not missing_critical and not missing_minor:
        lines.append("✅ 关键阶段产物齐全。")
    lines.append("")
    return "\n".join(lines)


__all__ = ["STAGE_DEPS", "artifact_status", "stage_readiness",
           "data_integrity_markdown"]
