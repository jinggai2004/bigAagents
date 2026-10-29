# -*- coding: utf-8 -*-
"""阿深定时值班（P1，2026-10-27）。

## 它解决的问题

阿深原先只在"被唤醒条件击中"时醒一次（`core/ai_wake.py`）——那是**事件驱动**，
一天可能 0 次、也可能 5 次。老板要的是"交易时段**定期**唤醒它，让它看着盘、
该动手就动手"。所以这里加一条**时间驱动**的通道，与唤醒通道并存。

## 三条硬约束（每一条都对应一次实测踩坑）

1. **不新增 cron 任务**。挂 `intraday_watch` 里面（每分钟一轮、自带交易日与
   盘中时段闸门），否则 `JOBS` 从 19 条变 20 条，牵动 `JOB_IDS` 断言、任务冲突
   检测、文档端点计数，而且"新增任务必须重启调度器"这条运维要求会静默失效。
2. **零额外行情请求**。用 `scan_once` 已经拉到的 `quotes`。盘中 0.5 限频预算，
   多拉一遍会让盯盘自己超时。
3. **`due_slot` 判"间隔"而不是"过点"**。14:55 也满足 `hhmm >= '14:40'`，
   但那时补跑用的是过期信息。只在到点后 30 分钟内补一次，超时不补。

## 成本闸门

6 个 slot 里通常只有 2~3 轮真需要叫模型。做法是**先本地预筛、再扣额度**：
`_worth_llm()` 说不值得就不扣 `routine` 额度、只落一行 `status='skipped'`。
⚠️ 顺序不能反：先扣后筛的话，上午几轮空跑会把额度用光，
而 14:40 那轮（最需要动手的一轮）就没得跑了。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, time as dtime

_logger = logging.getLogger(__name__)

PERSONA_NAME = "ai_ashen"

# 6 轮/日。全部避开已知重活（见报告 §4.3）：
# 09:25 选股跑到 09:36、09:45 结算、11:30 午间选股、13:05 午间决策轮、
# 14:50 风控巡检。14:40 必须早于 14:50 —— 否则阿深的调整和风控巡检
# 抢同一份行情，且结论可能互相矛盾（阿深刚加仓、风控立刻报超限）。
SLOTS = ("09:40", "10:20", "11:00", "13:20", "14:00", "14:40")

# 时段剧本：同一个阿深，钟点不同，**该干的事就不同**。
# ⚠️ 为什么必须有（2026-10-28 实测）：6 个班次原本只拿得到"第几轮"这个信息，
#    09:40 和 14:40 用同一套判据（"跌幅够不够 1.5%、止损空间够不够"）看同一批票，
#    结论自然一模一样——当天 6 轮 summary 两两平均相似度 46%、第 5→6 轮 95%。
#    光是"带上今日轨迹"只能治复读（少说一遍），治不了"六轮在回答同一个问题"。
#    时段剧本给每一轮一个**不同的必答问题**，让 6 轮变成一条递进的线：
#    开盘定性 → 上午验证 → 半天结算 → 午后修正 → 尾盘定型 → 过夜与否。
#
# 结构：(阶段起始 HH:MM, 名称, 这一轮该看什么, 这一轮 summary 必须回答的问题, 这一时段的硬边界)
# 起始时刻与 `SLOTS` 对齐（09:30→09:40 轮、10:00→10:20 轮、13:00→13:20 轮…），
# 保证 6 个班次**落在 6 个不同阶段**，不会有两轮共用一套剧本。
PHASES = (
    ("09:30", "开盘确认期",
     "今天低开/高开之后走势是延续还是反向，量能配不配合",
     "今天开盘和你昨天收盘时的判断相比，哪一点已经不一样了",
     "刚开盘别定性：不追高、不接第一根大阴线"),
    ("10:00", "上午主段",
     "上午的方向站没站住，自选池里有没有出现计划内的价位",
     "相比开盘那轮，是哪个具体数字变了；没变就直说没变",
     "全天波动最大的一段，警惕假突破；买必须是计划内的价位"),
    ("11:00", "上午收官",
     "半天收在什么位置，离你今天的计划还差多少",
     "半天过去了，你今天的计划还差哪一步没做",
     "上午最后一个观察点，别在这个点临时改主意"),
    ("11:30", "午间休市",
     "不开盘，只能做准备：下午盯哪只、什么价位、什么条件",
     "下午开盘前，你准备盯哪一只、在什么价位动手",
     "休市不能下单，任何买入都要等 13:00 之后"),
    ("13:00", "午后开盘",
     "午休期间的消息面/外盘有没有改变上午的结论",
     "午休这段时间，有没有新信息让你想改判断（没有就说没有）",
     "午后常见情绪修复或补跌，不追第一波"),
    ("14:00", "下午中段",
     "留给尾盘的时间还有多少，仓位需不需要在尾盘前定型",
     "如果今天什么都不做，明天你会后悔哪一点",
     "新开仓要预留过夜空间：止损别设在明天的跳空缺口里"),
    ("14:30", "尾盘决策期",
     "今天要不要留仓过夜、留多少",
     "今天最后一轮：过夜还是空仓，为什么",
     "尾盘流动性变差，不追急拉；宁可不做，不留被动仓位"),
)
# 收盘后（手动补跑或 15:00 之后触发）单独一套：此时不能下单，只出计划。
PHASE_AFTER = ("15:00", "收盘后",
               "今天已经收市，判断对错要等 T+N 才知道，现在只做记录与明日计划",
               "今天没做的事，明天开盘第一件事是什么",
               "已收盘，不能下单；只回答计划，不要承诺「明天一定买」")


def phase_of(now: datetime) -> tuple:
    """当前时刻落在哪个时段（返回 `PHASES` / `PHASE_AFTER` 里的那一项）。

    只看钟点，不看交易日：手动补跑可能发生在任何时间，而"现在几点"永远有意义。
    早于首个阶段的（比如 08:00 手动跑）用第一阶段——那时最该做的就是开盘预案。
    """
    hhmm = now.strftime("%H:%M")
    picked = PHASES[0]
    for ph in PHASES:
        if hhmm >= ph[0]:
            picked = ph
        else:
            break
    if hhmm >= PHASE_AFTER[0]:
        return PHASE_AFTER
    return picked

# 手动补跑的落库槽位名（不是排班的一格）。
# ⚠️ 它必须不是一个看起来像时刻的字符串：写成 "15:10" 会被误认成第 7 个班次，
#    也会让"每天 ≤6 条"这条验收判据失去意义。
MANUAL_SLOT = "手动"

# 补跑窗口：到点后 30 分钟内还允许补一次；超过就**不补**。
# 盘中判断 60 分钟就过期了，用一个过期理由去建仓是有害的。
SLOT_WINDOW_MIN = 30

# 强制复盘间隔：即使四个信号都不成立，距上一轮超过这个分钟数也要叫一次模型。
# 理由：市场"没什么变化"本身是一种需要被记录的状态，不能因为安静就一整天不记录。
#
# ⚠️ 这个阈值**不能单独承担"多久跑一次"的语义**（2026-10-27 实测踩到）：
#    班距是 40 分钟，80 正好是 2×班距 —— 但 `created_at` 是秒级的，一轮落在
#    09:44:50、下一轮落在 10:20:05 时 gap 只有 75 分钟，判据就不成立。
#    结果是"隔一班复盘"退化成"隔两班"，而且**随机**：取决于扫描抖动。
#    真正的兜底是下面的 `FORCE_SKIP_STREAK`（按跳过轮数，与秒无关），
#    这里只保留"很久没复盘"的兜底作用。
FORCE_INTERVAL_MIN = 80

# 连续跳过几轮后强制复盘一次（与时间无关的兜底）。
# 取 1 ⇒ 平静日固定节奏"跑、跳、跑"：09:40 跑、10:20 跳、11:00 跑、13:20 跑…
# ⚠️ 为什么不用时间判据替代：见上面 `FORCE_INTERVAL_MIN` 的抖动说明。
FORCE_SKIP_STREAK = 1

# 预筛阈值（纯本地计算，不花钱）
HOLDING_MOVE_PCT = 1.5     # 持仓票涨跌超过 1.5% ⇒ 值得看一眼
WATCH_AMPLITUDE_PCT = 3.0  # 自选池里有票振幅超过 3% ⇒ 值得看一眼


def due_slot(now: datetime | None = None) -> str | None:
    """当前应跑的 slot；没有则 `None`。

    判据 = "取 <= 现在 的最近一格，且到点不超过 `SLOT_WINDOW_MIN` 分钟"。

    ⚠️ 不能只判 `hhmm >= slot`：14:55 也满足 >= 14:40，但那时已经过点了。
    也不能在窗口外"顺延补跑"——盘中决策的时效性是分钟级的。
    """
    now = now or datetime.now()
    hhmm = now.strftime("%H:%M")
    passed = [s for s in SLOTS if hhmm >= s]
    if not passed:
        return None
    last = passed[-1]
    slot_dt = datetime.combine(now.date(), dtime.fromisoformat(last))
    delta_min = (now - slot_dt).total_seconds() / 60.0
    if delta_min < 0 or delta_min > SLOT_WINDOW_MIN:
        return None
    return last


def _quote_pct(q: dict) -> float | None:
    v = q.get("change_pct")
    if isinstance(v, (int, float)):
        return float(v)
    px, prev = q.get("price"), q.get("prev_close")
    if isinstance(px, (int, float)) and isinstance(prev, (int, float)) and prev:
        return (px - prev) / prev * 100.0
    return None


def _quote_amplitude(q: dict) -> float | None:
    """振幅 = (最高 − 最低) / 昨收。自己算而不读 `amplitude` 字段：
    腾讯有、新浪没有，读字段会在降级切换时变成 None（两源口径不一致）。"""
    hi, lo, prev = q.get("high"), q.get("low"), q.get("prev_close")
    if all(isinstance(x, (int, float)) for x in (hi, lo, prev)) and prev:
        return (hi - lo) / prev * 100.0
    return None


def worth_llm(quotes: dict, *, holdings=None, my_watch=None, user_watch=None,
              last_run_at: str = "", now: datetime | None = None,
              skips_since_llm: int = 0) -> tuple[bool, str]:
    """本地预筛：这一轮值不值得叫模型。返回 `(是否值得, 原因)`。

    五个信号任一成立就调：
      ① 持仓票 |涨跌| > 1.5%
      ② 自选池（两套并集）里有票振幅 > 3%
      ③ 有已武装的唤醒条件（说明有个"等价位"承诺没兑现）
      ④ 上一次真实研判之后已连续跳过 `FORCE_SKIP_STREAK` 轮（节奏兜底）
      ⑤ 距上一轮 ≥ 80 分钟（强制复盘一次）

    ⚠️ 返回的原因必须**带具体数字**。只写"无变化"会让"为什么今天只跑了 2 轮"
    变成无法回答的问题，而它决定了日成本是 ¥0.03 还是 ¥0.08。

    ⚠️ `last_run_at` 必须是**上一次真正调了模型**的那一轮，不能是"上一次被跳过
    的那格"。跳过轮同样会落库并带 `created_at`，拿它当基准会让 gap 永远只有
    一个班距（40 分钟）< 阈值 80 ⇒ 跳过一格之后**每一格都继续跳过**（级联跳过）。
    `run_due` 负责只挑 `llm_called` 的行传进来。
    """
    now = now or datetime.now()
    holdings = set(str(c) for c in (holdings or []))
    watch = set(str(c) for c in (my_watch or [])) | set(str(c) for c in (user_watch or []))
    quotes = quotes if isinstance(quotes, dict) else {}

    moves = []
    for code in holdings:
        pct = _quote_pct(quotes.get(code) or {})
        if pct is not None:
            moves.append((abs(pct), code, pct))
    if moves:
        top, code, pct = max(moves)
        if top > HOLDING_MOVE_PCT:
            return True, (f"持仓 {code} 变动 {pct:+.2f}%"
                          f"（> {HOLDING_MOVE_PCT:g}%）")

    amps = []
    for code in watch:
        amp = _quote_amplitude(quotes.get(code) or {})
        if amp is not None:
            amps.append((amp, code))
    if amps:
        top, code = max(amps)
        if top > WATCH_AMPLITUDE_PCT:
            return True, f"自选 {code} 振幅 {top:.2f}%（> {WATCH_AMPLITUDE_PCT:g}%）"

    try:
        from core import ai_wake
        armed = ai_wake.armed_codes(now.strftime("%Y-%m-%d"))
        if armed:
            return True, f"有 {len(armed)} 个已武装的唤醒条件在等"
    except Exception as e:                                  # noqa: BLE001
        _logger.debug("[阿深值班] 唤醒条件读取失败（不影响预筛）: %s", e)

    # ④ 节奏兜底：连续跳过太多次就强制跑一轮，别让"安静"变成"整天不记录"。
    #    放在⑤之前，是因为它比"距上一轮多久"更硬：⑤依赖秒级时间戳，遇到
    #    扫描抖动会把"隔一班"拖成"隔两班"（见 FORCE_INTERVAL_MIN 的说明）。
    if skips_since_llm >= FORCE_SKIP_STREAK:
        return True, f"已连续跳过 {skips_since_llm} 轮，强制复盘一次"

    if last_run_at:
        try:
            prev = datetime.strptime(last_run_at, "%Y-%m-%d %H:%M:%S")
            gap = (now - prev).total_seconds() / 60.0
            if gap >= FORCE_INTERVAL_MIN:
                return True, f"距上一轮 {gap:.0f} 分钟（≥ {FORCE_INTERVAL_MIN}），强制复盘"
            max_move = f"{max(moves)[0]:.2f}%" if moves else "—"
            max_amp = f"{max(amps)[0]:.2f}%" if amps else "—"
            return False, (f"无变化：持仓最大变动 {max_move}（阈值 {HOLDING_MOVE_PCT:g}%）、"
                           f"自选最大振幅 {max_amp}（阈值 {WATCH_AMPLITUDE_PCT:g}%）、"
                           f"距上一轮 {gap:.0f} 分钟（阈值 {FORCE_INTERVAL_MIN}）")
        except (TypeError, ValueError) as e:
            # 上一轮的 created_at 不是预期格式 ⇒ 当作"没有上一轮"（保守：会多跑一轮）。
            _logger.warning("[阿深值班] 上一轮时间戳无法解析（%r）: %s", last_run_at, e)
    return True, "今日首轮，必跑"


def _today_candidates(trade_date: str, limit: int = 8) -> list[dict]:
    """今日选股候选（**只读**），给阿深一个"从哪挑票"的入口。

    ⚠️ 为什么必须有它：阿深的自选池初始是**空的**（历史迁移只把旧清单给了老板），
    而 L2 建仓权只允许买**自己池内**的票 ⇒ 池子空 = 建仓权开着也用不出来，
    现象是"权限开了但一只都不买"，看起来像它不想动手。

    ⚠️ 同轮 `watch_add` + `buy` 同一只票**一定会被拒**（池子是判定时的快照）。
    这是刻意的：先观察、下一轮才考虑买，不给"看到就追"留口子。prompt 里要说清，
    否则模型会每轮都试一遍并以为是自己写错了。

    读不到就返回 `[]`（**不编**），并留痕。
    """
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "output", "daily_picks", "latest.json")
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if str(d.get("date")) != str(trade_date):
            _logger.debug("[阿深值班] 候选不是今天的（latest.json=%s）", d.get("date"))
            return []
        out = []
        for p in (d.get("picks") or [])[:limit]:
            code = str(p.get("股票代码") or "").strip()
            if not code:
                continue
            out.append({"code": code, "name": str(p.get("股票名称") or ""),
                        "score": p.get("score"), "board": p.get("板块")})
        return out
    except FileNotFoundError:
        _logger.debug("[阿深值班] 未找到 daily_picks/latest.json（当日流水线可能还没跑）")
        return []
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[阿深值班] 读取今日候选失败: %s: %s", type(e).__name__, e)
        return []


def _armed_conditions(*, strict: bool = False) -> list[dict]:
    """阿深**自己**挂着的等待条件（`armed=1`，未过期）。读失败返回 `[]` 并留痕。

    为什么要把它们塞进 prompt（2026-10-28）：模型只知道自己"嘴上在等什么"
    （靠 `last_summary`），不知道系统里**真的**挂着什么。于是每轮都重新说一遍
    "等伊利 28 元下方"，而条件表里一条都没有。摊开给它看之后，它才能
    "沿用 / 覆盖 / 撤掉"，用户也才能在页面上核对它说的话与它挂的条件是否一致。

    Args:
        strict: `False`（prompt 用）⇒ 读失败按"无条件下"继续跑，一次库抖动不该
            让整轮值班挂掉；重复挂条件由 `(persona, code, kind)` 幂等键兜住。
            `True`（页面用）⇒ 异常上抛，让调用方渲染"读取失败"。
            **不能把"读失败"和"确实没挂条件"显示成同一件事**：前者要人去查库，
            后者是它自己没挂（正是要暴露的敷衍）。
    """
    from db import store
    try:
        rows = store.list_wake_conditions(persona=PERSONA_NAME, armed_only=True, limit=50)
    except Exception as e:                                  # noqa: BLE001
        if strict:
            raise
        _logger.warning("[阿深值班] 等待条件读取失败（本轮按「无条件」处理）: %s", e)
        return []
    return [r for r in (rows or []) if r.get("code")]


def build_prompt(*, now: datetime, slot: str, level: str, auton_desc: str, budget_desc: str,
                 cash: float, nav: float, holdings: list, my_watch: list,
                 user_watch: list, quotes: dict, last_summary: str = "",
                 candidates: list | None = None,
                 trail: list | None = None, armed: list | None = None) -> str:
    """值班轮 prompt（单轮、无工具、只要一个 JSON）。

    ⚠️ 边界数字一律来自 `ai_autonomy.describe()`（通过 `budget_desc` 传入），
    **不要在模板里手写"8%、24%、3 只"**：改了参数忘了改 prompt，模型会按旧
    数字规划、代码按新数字拦，两边都对、结果全被拒。

    ## 为什么要 `trail`（今天的判断轨迹，2026-10-28 加）
    原来只带 `last_summary`（**上一轮**一句话）⇒ 它看不到自己今天已经说过 4 遍
    "跌幅小、止损空间不足、等伊利 28 元下方"。实测 6 轮结论两两平均相似度
    **46%**、第 5→6 轮 **95%** —— 这不是"盯盘盯得稳"，是**每轮都在第一次发现**。
    带上全天轨迹后，"判断没变"可以只报变化的那个数字，用户读到的才是一条线。

    ## 为什么要 `armed`（它自己挂着的等待条件）
    它天天说"在等伊利 28 元下方"，而 `ai_wake_conditions` 里 ai_ashen 的记录
    曾经是 **0 条** —— 那个等待在系统里根本不存在（价位到了不会叫它）。
    把已有条件摊在 prompt 里，既避免重复挂，也让"没挂"这件事对模型可见。
    """
    def _brief(rows, keys=("code", "name", "price", "change_pct", "stop_loss")):
        out = []
        for r in rows:
            item = {k: r.get(k) for k in keys if r.get(k) is not None}
            out.append(item)
        return json.dumps(out, ensure_ascii=False, separators=(",", ":"))

    def _trail(rows) -> str:
        parts = []
        for r in (rows or []):
            s = str(r.get("summary") or "").strip()
            if not s:
                continue
            acts = [a for a in (r.get("actions") or []) if isinstance(a, dict)]
            n = len([a for a in acts
                     if str(a.get("action") or "").lower() not in ("", "pass")])
            parts.append(f"{r.get('slot')} {s[:60]}" + (f"（动作 {n} 笔）" if n else "（无动作）"))
        return " ｜ ".join(parts) if parts else "（今天还没跑过）"

    def _armed(rows) -> str:
        parts = []
        for c in (rows or []):
            thr = c.get("threshold")
            pct = c.get("threshold_pct")
            if c.get("wake_at"):
                v = f"{c['wake_at']} 定时"
            elif thr is not None:
                v = f"{float(thr):g}"
            elif pct is not None:
                v = f"{float(pct) * 100:g}%"
            else:
                v = "（无阈值）"
            parts.append(f"{c.get('code')} {c.get('kind')} {v}")
        return "、".join(parts) if parts else "（没有）"

    held = _brief(holdings)
    mine = _brief(my_watch, ("code", "name", "note", "price", "change_pct"))
    boss = _brief(user_watch, ("code", "name", "note", "price", "change_pct"))
    cand = _brief(candidates or [], ("code", "name", "score", "board"))
    # 时段剧本（见 PHASES）：拆成局部量再进模板，避免 f-string 里嵌套引号
    _ph = phase_of(now)
    ph_name, ph_focus, ph_ask, ph_rule = _ph[1], _ph[2], _ph[3], _ph[4]
    # "第 14:40 轮"读不成句，而且看不出"一天几轮、现在第几轮"。
    # 排班轮给 N/6，手动轮如实说补跑（不要让模型以为自己又在走正常节奏）。
    # ⚠️ 用 `in` 判而不是 try/except ValueError：这里没有任何"可能失败"的操作，
    #    用异常控制流程还会被"静默 except"审计记一笔（没有任何东西需要记日志）。
    if slot in SLOTS:
        slot_desc = f"第 {SLOTS.index(slot) + 1}/{len(SLOTS)} 轮（{slot}）"
    else:
        slot_desc = f"手动补跑（{slot}）"
    return f"""你是阿深，一名 A 股交易员，现在在盯盘。老板给了你一个模拟账户和两套自选股。

【现在】{now:%Y-%m-%d %H:%M}，{slot_desc}。授权级别：{level}（{auton_desc}）
【这个时段】{ph_name} —— 这一轮该看的是：{ph_focus}
    时段边界：{ph_rule}
【你的账户】现金 {cash:,.0f}，净值 {nav:,.0f}
【仓位边界】{budget_desc}
【你的持仓】{held}
【你的自选】{mine}
【老板的自选】{boss}   ← 只读。这是他让你留意的票，不代表他要买
【今日选股候选】{cand}   ← 只读。你**只能买自己自选里的票**；
    候选里看中谁，先 `watch_add` 加进自己的池子，**下一轮**才可能买（本轮加的不算数）
【今天的判断轨迹（你自己说过的话，按时间）】{_trail(trail)}
【你当前挂着的等待条件】{_armed(armed)}   ← 这是**真的会被触发的条件**，价位到了会叫你
【上一轮你的结论】{last_summary or "（今天还没有上一轮）"}

【可选动作（只输出一个 JSON 对象，不要代码块、不要解释文字）】
{{"actions":[
  {{"action":"buy","code":"600519","position_size":0.03,"limit_price":1600.0,"stop_loss":1500.0,"reason":"为什么现在买"}},
  {{"action":"sell","code":"000001","reason":"为什么卖"}},
  {{"action":"set_wake","code":"600519","kind":"price_below","threshold":1580,"reason":"在等什么"}},
  {{"action":"watch_add","code":"000858","reason":"为什么值得盯"}},
  {{"action":"watch_remove","code":"002230","reason":"为什么不再盯"}},
  {{"action":"study","code":"600519","reason":"要研究的具体问题"}},
  {{"action":"pass","reason":"在等什么信号"}}
],"summary":"本轮判断（≤80字，**必须回答这个问题：{ph_ask}**）"}}

【授权边界（代码强制，越界动作会被降级并记录）】
- 当前级别 {level}。L1 只能管存量（卖、调唤醒条件），**不可开新仓**；
  L2 可在**你自己的自选池内**建新仓，仓位上限见上面「仓位边界」
- 每笔买入必须同时给三个数：**limit_price（限价）、stop_loss（止损价）、reason（理由）**。
  缺任何一个，这笔**不作数**，会被代码直接丢掉——不是建议，是硬校验
- 只允许在下跌时接手，不追涨；不买接近跌停的票、不买停牌/一字板、不买 ST
- **先观察再加池**：`watch_add` 只影响下一轮；本轮想买不在你池里的票一定会被拒
- `study` 每轮最多 1 只：研究是**盘后**做的，现在只登记问题
- 老板的自选股：**你能看、能提意见，但不能改**

【说话的纪律（这几条决定老板会不会觉得你"在敷衍"）】
- **先答本时段那一题**：现在是「{ph_name}」，本轮必须回答「{ph_ask}」。
  六个班次问的是六件不同的事（开盘定性 → 上午验证 → 半天结算 → 午后修正 →
  尾盘定型 → 过夜与否），别每轮都回到"跌幅够不够、止损空间够不够"同一个判断上。
  同一个判断确实没变时，把它压成半句，把字数留给这一轮该答的题
- **禁止复述**：判断和上一轮相同时也别重写一遍。只写"与上一轮（几点那轮）相同"加**唯一变化的那个数字**
  （例：`仍无买点；伊利 28.92→28.61，距 28 还差 2.2%`）。
  连续几轮用同一个句式说同一件事，等于老板白花这几轮的钱
- **说在等，就必须挂上**：只要 summary 里出现"等某某价位/等回到/破位再看"，
  **必须同时输出 `set_wake`**（观察类用 `price_below` / `price_above` 并给 `threshold`）。
  没挂条件的等待属于空话——系统不会在价位到了叫你，老板也查不到你在等什么。
  这条是**代码检查的**：说了在等却没挂条件，系统会打回来让你重答一次
- **别只设一个几乎不会触发的条件**：自选票日波动常在 1% 以内，如果你把唯一
  入场条件设成"跌超 2%"，可能一整周都不触发，而你每天都只能说"还在等"。
  对看得上的票，给**三层价位**（价位全部取自上面给的真实价格，不许凭记忆写）：
  ① 现价这一档值不值得先给一小笔（`buy` + 首笔仓位 + limit_price + stop_loss）；
  ② 再跌到哪里加第二笔：`set_wake` `price_below` + 该价位；
  ③ 跌破哪里就放弃这只票：`set_wake` `price_below` + 放弃位，并在 reason 里写明"放弃位"
- **每轮至少要有一件实质动作**：`buy` / `sell` / `watch_add` / `watch_remove` /
  `set_wake` 任意一件。**只输出 `pass` 等于这一轮你没来过**——代码会打回来一次，
  再没有就由系统替你从候选里加一只票入池（那件事不是你的判断，会记在你账上）。
- 上面「你当前挂着的等待条件」是你**自己**挂的。条件已经不成立（票走远了、你不想要了）
  就用新的 `set_wake` 覆盖，或用 `watch_remove` 把它移出池子，别让它烂在那儿
- 数字必须来自上面提供的数据，不许凭记忆写价格；没给的票就是没有数据
- 不动就说不动，但要说清在等什么信号；空话（"继续观察"）视为无效回答
- 不预测涨停、不满仓、不因为上一笔亏了就想扳回来
"""


def _normalize(raw_actions, persona: dict, min_position: float = None) -> list[dict]:
    """归一动作，并**把三件套补回来**。

    ⚠️ `ai_daily_round.normalize_actions()` 只保留 action/code/position_size/reason，
    **会丢掉 `limit_price` / `stop_loss`**。如果直接把它的输出喂给授权判据，
    每一笔买入都会被判"缺限价/缺止损"⇒ 建仓权永远用不出来，而且原因看起来
    像"模型没给限价"。所以这里按 code 回填原始字段。

    `watch_add / watch_remove / study` 不在 `normalize_actions` 的白名单里
    （它只认 buy/sell/set_wake/pass），单独挑出来自己归一。

    ⚠️ `min_position` 必须传首笔比例（3%）：`clamp_plan` 默认的"最小有效仓位
    5%"会把首笔顶到 5%，然后被"首笔 ≤3%"拒掉 ⇒ 每笔首笔建仓都失败。
    """
    from core import ai_daily_round

    extras: list[dict] = []
    core: list[dict] = []
    orig_buy: dict[str, dict] = {}
    for a in (raw_actions or []):
        if not isinstance(a, dict):
            continue
        act = str(a.get("action") or "").strip().lower()
        code = str(a.get("code") or "").strip()
        if act in ("watch_add", "watch_remove", "study"):
            extras.append({"action": act, "code": code or None,
                           "reason": str(a.get("reason") or "")})
        else:
            core.append(a)
            if act in ("buy", "place_order") and code:
                orig_buy[code] = a

    out = ai_daily_round.normalize_actions(core, persona, min_position=min_position)
    for d in out:
        if d.get("action") == "buy":
            src = orig_buy.get(d.get("code")) or {}
            d["limit_price"] = src.get("limit_price", src.get("limit"))
            d["stop_loss"] = src.get("stop_loss")
    return out + extras


def _wake_kinds() -> tuple[str, ...]:
    """合法唤醒 kind —— 取自 `core.ai_wake.WAKE_KINDS`（唯一来源，不抄字面量）。

    延迟导入：`ai_wake` 会连带 import `ai_personas` / `ai_budget` / `ai_agent`，
    在模块顶层拉进来会让值班链的导入变重，而这条链挂在盯盘热路径上。
    """
    from core import ai_wake
    return ai_wake.WAKE_KINDS


# ============================================================
# 动作闸门：把"必须动"从 prompt 的软约束改成代码的硬约束
# ============================================================
#
# ## 为什么必须有这一层（2026-10-28 用户要求，实证驱动）
#
# prompt 第「说话的纪律」里早就写了「说在等，就必须挂上 set_wake」。实际结果：
#
# ```
# 2026-10-28 六轮全部：[('pass', None, None)]
# ai_wake_conditions 里 persona='ai_ashen'：0 条
# ai_decisions 里 persona='ai_ashen'：0 条；账户 fills：0 笔
# ```
#
# 它连续六轮说"等伊利 28 元下方或跌超 2%"，而这六句话在系统里**一个字都没落下**：
# 价位真跌破 28 也不会有人叫它，因为它从没挂过条件。
#
# ⇒ 结论很硬：**写成 prompt 的强制等于没有强制**。模型会礼貌地忽略它，
#   而且忽略得毫无痕迹（结论读起来完全正常）。所以判定与补正要放进代码。
#
# ## 为什么"必须动"不等于"必须买"
#
# 用户要的是"每天都有行动"，但拿"动作数不为 0"去逼模型下单，是拿账户纪律
# 换页面好看 —— 那个方向不能走。所以这里强制的是**不改钱的动作**：
# 挂条件、维护池子。买卖仍然由它自己判断，代码不催、也不替它决定。

# summary 里出现这些词 = 它在说"要等某个东西"
_WAIT_WORDS = ("等", "等待", "回到", "跌破", "站上", "下方", "上方", "再看",
               "破位", "回踩", "回调")

# **实质动作**：会改变系统状态的。`pass` 不在其中 —— 它恰恰是"什么都不做"。
# 加一个动作时记得同步这里，否则它会变成"做了也不算数"的隐形动作。
REAL_ACTIONS = ("buy", "sell", "watch_add", "watch_remove", "set_wake")


def action_gap(summary: str, actions: list) -> str:
    """这一轮缺什么。返回 `""` 表示不缺。

    两种缺口分开报，因为补正话术不同、严重程度也不同：

    | 返回值 | 含义 | 后果 |
    |---|---|---|
    | `no_action` | 只有 pass，没有任何实质动作 | 这一天对它等于没发生 |
    | `missing_wake` | 说了在等什么，但没挂条件 | 它说的等待在系统里不存在 |
    """
    acts = [a for a in (actions or []) if isinstance(a, dict)]
    real = [a for a in acts if str(a.get("action") or "").lower() in REAL_ACTIONS]
    if not real:
        return "no_action"
    if any(str(a.get("action") or "").lower() == "set_wake" for a in real):
        return ""
    if any(w in str(summary or "") for w in _WAIT_WORDS):
        return "missing_wake"
    return ""


def _retry_note(gap: str, summary: str) -> str:
    """补正提示：把"你上一轮其实什么都没做"摊开说，要求重出一次 JSON。

    ⚠️ 措辞刻意**不骂人也不哄人**，只陈述事实 + 给可选项：
    实测（记忆里记着的那条）把"你其实没挂条件"直接摊进 prompt，模型会主动
    认"这是我的疏漏"并补上；而含糊的"请再想想"只会换来另一段 pass。
    """
    if gap == "missing_wake":
        return (
            "【系统补正】你上一轮的结论里说了在等某个价位/信号，但**没有输出 set_wake**。\n"
            "那样的等待在系统里是不存在的：价位到了不会叫你，老板也查不到你在等什么。\n"
            f"  你上一轮的原话：{summary}\n"
            "  现在重新输出 JSON：把你说的那个等待**落成 set_wake**（必须给 kind 与 "
            "threshold 或 threshold_pct，价位取自上面给的真实价格）。\n"
            "  如果你判断压根不需要等，就给出别的实质动作并说明为什么。")
    return (
        "【系统补正】你上一轮只输出了 pass，没有任何实质动作。\n"
        "一天六轮全部\"不动\"，等于这一天你没有存在过 —— 老板要看到你在场、在做事。\n"
        f"  你上一轮的原话：{summary}\n"
        "  现在重新输出 JSON，至少做一件（下面三类都不动钱，可以放心做）：\n"
        "  · set_wake：把你盯的价位落成真会触发的条件（kind + threshold）\n"
        "  · watch_add / watch_remove：维护你自己的池子\n"
        "  · buy / sell：确实有把握才做，没把握别硬上\n"
        "  注意：为了凑动作去下单是错的；但\"什么都盯着、什么都没登记\"同样是错的。")


def _auto_watch(candidates: list, my_watch: list) -> dict | None:
    """兜底：从今日候选里挑一只**还不在池子里**的，替它加进来。

    为什么兜底是这个动作而不是买入：
      · `watch_add` 不动钱，最坏后果是池子里多一只票（下一轮可以 `watch_remove`）；
      · 买入花的是真金白银。为了"动作数不为 0"去自动下单，等于用账户换页面好看。

    返回 `{"action":"watch_add","code":...,"reason":...}`；没有可加的候选返回 None
    （宁可如实记"这轮确实没动作"，也不编一个动作出来）。
    """
    have = {str(w.get("code") or "") for w in (my_watch or [])}
    for c in (candidates or []):
        code = str(c.get("code") or "").strip()
        if code and code not in have:
            return {"action": "watch_add", "code": code,
                    "reason": "系统兜底：本轮没有任何实质动作，"
                              "从今日候选里取一只入池供下轮判断"}
    return None



def _execute(kept: list[dict], account, today: str, quotes: dict,
             nav: float) -> list[dict]:
    """把通过授权判据的动作**落地**：买卖落成委托单、自选变更直接写库。

    ⚠️ 自选变更必须在这里落地，不能只判不写：`autonomy_check` 判定 `watch_add`
    合法之后如果什么都不做，阿深的池子永远空着 ⇒ 建仓权永远没有标的。
    "判了但没做"是这类接线最常见的失败方式（函数写好了、页面看不出来）。

    ⚠️ 买卖只提交委托，不撮合。撮合在 `PaperAccount.settle_open()`（次日开盘），
    与联赛、`settle_daily` 是**同一条路径** —— 盘中成交价在 A 股本来就不存在，
    用现价当成交价会系统性高估收益（项目已有结论）。

    返回逐笔结果（含跳过原因），供落库审计。
    """
    from core import matching
    from db import store

    out: list[dict] = []
    if not kept:
        return out

    # 名称兜底：`watch_add` 的主要来源就是"今日候选"，而手动唤醒/盯盘取的行情
    # 集合未必包含它（实测：手动轮里三条入池记录 `name` 全是空串，页面上就是
    # 无名的代码）。候选名单在本地 json 里，查名字零成本。
    _names: dict = {}

    def _name(code: str) -> str:
        n = (quotes.get(code) or {}).get("name")
        if n:
            return str(n)
        if not _names:
            _names.update({c["code"]: c["name"] for c in
                           _today_candidates(today, limit=50) if c.get("code")})
        return _names.get(code, "")

    # ① 自选池维护（不依赖账户，先做）
    for a in kept:
        act, code = a.get("action"), str(a.get("code") or "")
        if act == "watch_add" and code:
            store.upsert_watchlist_item("ashen", code, name=_name(code),
                                        note=str(a.get("reason") or "")[:120],
                                        source="research")
            out.append({"code": code, "action": "watch_add", "status": "ok"})
        elif act == "watch_remove" and code:
            ok = store.delete_watchlist_item("ashen", code)
            out.append({"code": code, "action": "watch_remove",
                        "status": "ok" if ok else "not_found"})
        elif act == "study" and code:
            # P2 才建 ai_study_queue。现在**如实记录"登记了但没执行"**，
            # 不能假装研究已排期（那会让复盘以为有报告可看）。
            out.append({"code": code, "action": "study",
                        "skipped": "研究队列尚未上线（P2），本轮只登记问题"})
        elif act == "set_wake" and code:
            # ⚠️⚠️ 这条分支 2026-10-28 才补上，此前 `set_wake` 是**死路**：
            #    它能过授权判据（L1 就允许"调唤醒条件"）、能进 `actions_json`、
            #    页面"动作数"还会 +1，看起来像做了事 —— 但 `_execute` 里
            #    **没有任何落地分支**，条件表从不写入。
            #    实证：阿深连续 6 轮说"等伊利 28 元下方"，而
            #    `ai_wake_conditions` 里 persona='ai_ashen' 的记录是 **0 条**
            #    （全表 60 条全是 ai_tech/ai_value/ai_random）。
            #    ⇒ 它的"等待"从未变成可触发的条件，跌破 28 也没人会叫醒它。
            #    这是"函数写好 ≠ 验过接线"的又一例。
            kind = str(a.get("kind") or "").strip().lower()
            if kind not in _wake_kinds():
                out.append({"code": code, "action": "set_wake",
                            "skipped": f"kind 非法（{kind or '空'}）⇒ 条件未落库"})
            elif (a.get("threshold") is None and a.get("threshold_pct") is None
                  and not a.get("wake_at")):
                # 没价位也没时刻的条件永远不会触发。收下它等于给了用户一个
                # 假的"我在等" —— 明确拒收并留痕。
                out.append({"code": code, "action": "set_wake",
                            "skipped": "缺 threshold / threshold_pct / wake_at"})
            else:
                store.upsert_ai_wake_condition(PERSONA_NAME, {
                    "code": code, "kind": kind,
                    "threshold": a.get("threshold"),
                    "threshold_pct": a.get("threshold_pct"),
                    "wake_at": a.get("wake_at"),
                    "note": str(a.get("reason") or "")[:200]})
                out.append({"code": code, "action": "set_wake", "status": "ok",
                            "kind": kind, "threshold": a.get("threshold"),
                            "threshold_pct": a.get("threshold_pct")})

    if account is None:
        return out

    # ② 卖出（先卖回笼现金，与 `settle_open` 的"先卖后买"顺序一致）
    sells = [a for a in kept if a.get("action") == "sell" and a.get("code")]
    buys = [a for a in kept if a.get("action") == "buy" and a.get("code")]
    if sells:
        # ⚠️ 必须先解冻，否则 `available` 恒为 0 ⇒ 卖单全部被"无可卖持仓"拒
        account.unlock_t1(today)
    for a in sells:
        code = str(a["code"])
        pos = account.positions.get(code)
        avail = getattr(pos, "available", 0) if pos else 0
        if avail <= 0:
            out.append({"code": code, "action": "sell",
                        "skipped": "无可卖持仓（T+1 未解锁或已清仓）"})
            continue
        order = account.submit_order(code, "卖出", avail, date=today,
                                     reason=str(a.get("reason") or "")[:80])
        out.append({"code": code, "action": "sell", "qty": avail,
                    "order_id": order.order_id, "status": order.status,
                    "reason": order.reason})
    for a in buys:
        code = str(a["code"])
        px = (quotes.get(code) or {}).get("price")
        if not isinstance(px, (int, float)) or px <= 0:
            out.append({"code": code, "action": "buy", "skipped": "取不到现价"})
            continue
        size = a.get("position_size")
        if not isinstance(size, (int, float)) or size <= 0:
            out.append({"code": code, "action": "buy", "skipped": "仓位比例无效"})
            continue
        budget = float(size) * float(nav)
        qty = matching.round_lot(int(budget // px))
        if qty <= 0:
            out.append({"code": code, "action": "buy",
                        "skipped": f"不足一手：一手需 {px * 100:,.0f}，"
                                   f"预算只有 {budget:,.0f}（单票上限下买不起这只票）"})
            continue
        order = account.submit_order(
            code, "买入", qty, date=today, reason=str(a.get("reason") or "")[:80],
            limit_price=a.get("limit_price"), decision_price=px)
        out.append({"code": code, "action": "buy", "qty": qty,
                    "order_id": order.order_id, "status": order.status,
                    "limit_price": a.get("limit_price"), "decision_price": px,
                    "reason": order.reason})
    return out


def _load_account(name: str):
    from core.paper_account import PaperAccount
    from db import store
    state = store.load_paper_state(name)
    if state:
        return PaperAccount.from_state(state)
    return None


def _ensure_account(persona: dict):
    """取阿深的模拟账户；没有就**按 persona.initial_cash 建一个**。

    ⚠️ 必须在这里建，而不是等"第一次买入"时懒建：没有账户 ⇒ `nav=0` ⇒
    授权判据里每一笔买入都因"账户快照不完整"被拒 ⇒ 建仓权**永远不会生效**，
    而页面看起来只显示"今天没动作"（归因全错）。

    走 `store.save_paper_state()`（与联赛同一条路径，内部 `_ensure_column` 兼容），
    **不手写 INSERT** —— 项目铁律：加列必须走 `_ensure_column()`。
    """
    acct = _load_account(PERSONA_NAME)
    if acct is not None:
        return acct
    from core.paper_account import PaperAccount
    from db import store
    acct = PaperAccount(initial_cash=float(persona.get("initial_cash") or 1_000_000),
                        name=PERSONA_NAME)
    store.save_paper_state(acct.name, acct.to_state())
    _logger.info("[阿深值班] 已为 %s 建立模拟账户（初始资金 %.0f）",
                 PERSONA_NAME, acct.initial_cash)
    return acct


def run_due(quotes: dict, *, now: datetime | None = None, force: bool = False,
            execute: bool = True, client_factory=None) -> dict:
    """执行"该跑的那一轮"。由 `AlertWatcher.scan_once()` 调用。

    Args:
        quotes: **同一份**行情（`scan_once` 已取的），零额外请求。
        force: 手动补跑。跳过预筛与幂等，但仍受授权与额度约束。
        execute: 是否真的提交委托（False = 只判不落单，用于演练/测试）。
        client_factory: 测试注入 LLM 客户端。

    Returns: 永远返回 dict，`skipped` 字段说明为什么没跑（**不抛异常**）。
             盘中一轮出错不该让盯盘挂掉。
    """
    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")
    from core import ai_personas, ai_budget, ai_autonomy
    from db import store

    persona = ai_personas.get_persona(PERSONA_NAME)
    if not persona:
        return {"skipped": "阿深未注册（config/ai_personas.json 缺 ai_ashen）"}
    if not persona.get("enabled"):
        return {"skipped": "阿深未启用"}

    slot = due_slot(now)
    if not slot:
        if not force:
            return {"skipped": "无到点的轮次"}
        # 手动补跑：不在任何 slot 窗口内也必须能跑。
        # ⚠️ 否则页面上的「演练」按钮只在 6 个 5 分钟窗口内可用 —— 真实体验是
        #    "点一下提示无到点的轮次"，而这几乎永远成立（一天只有 30 分钟能用）。
        # 用固定名落库：手动轮不是排班的一部分，同一交易日多次手动**覆盖同一格**
        # （`save_routine_run` 是 UPSERT），留多条会让人误以为多了几个班次。
        slot = MANUAL_SLOT
    if not force and store.routine_exists(PERSONA_NAME, today, slot):
        return {"skipped": f"{slot} 本轮已跑过", "slot": slot}

    level = ai_autonomy.normalize_level(persona.get("autonomy"))
    if level != str(persona.get("autonomy") or "").strip().upper():
        _logger.warning("[阿深值班] 授权级别非法（%r），已回退 %s",
                        persona.get("autonomy"), level)
    params = ai_autonomy.resolve()          # ⚠️ 参数只有这一处来源，别在别处写死
    account = _ensure_account(persona)

    # ── 组装事实（零联网：全部来自 quotes 与本地库）──
    from data import watchlist as wl
    try:
        # ⚠️ `snapshot()` 返回的是 `{owner,label,stocks,counts}`，不是 list。
        #    它**读失败会抛异常**（严格语义）——这里必须接住，否则一次库抖动
        #    会让整轮值班挂掉。
        my_watch_rows = wl.snapshot("ashen")["stocks"]
        user_watch_rows = wl.snapshot("user")["stocks"]
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[阿深值班] 自选池读取失败: %s", e)
        my_watch_rows, user_watch_rows = [], []

    def _rows(rows):
        out = []
        for r in rows:
            code = str(r.get("code") or "")
            q = quotes.get(code) or {}
            out.append({"code": code, "name": r.get("name") or q.get("name") or "",
                        "note": r.get("note") or "",
                        "price": q.get("price"), "change_pct": _quote_pct(q)})
        return out

    my_watch = _rows(my_watch_rows)
    user_watch = _rows(user_watch_rows)
    holdings = []
    nav, cash = 0.0, 0.0
    prices = {c: (q or {}).get("price") for c, q in (quotes or {}).items()}
    prices = {c: p for c, p in prices.items() if isinstance(p, (int, float))}
    if account is not None:
        cash = float(account.cash)
        try:
            nav = float(account.total_value(prices))
        except Exception as e:                              # noqa: BLE001
            # 退化成"净值=现金"会让仓位上限全部变小（保守方向），但必须留痕：
            # 否则"为什么今天一笔都没买"看起来像模型不动手。
            _logger.warning("[阿深值班] 净值计算失败，退化为现金口径 %.0f: %s", cash, e)
            nav = cash
        for code, pos in account.positions.items():
            q = quotes.get(str(code)) or {}
            px = q.get("price")
            holdings.append({"code": str(code), "name": q.get("name") or "",
                             "price": px, "change_pct": _quote_pct(q),
                             "qty": getattr(pos, "quantity", 0),
                             "stop_loss": None})
    else:
        _logger.info("[阿深值班] 阿深还没有模拟账户（首次建仓时会自动创建）")

    # ── 预筛（不花钱）：不值得就不扣额度 ──
    # ⚠️ 必须把当天全部轮次读回来自己按时间排，不能用 `limit=1`。两个坑：
    #    ① `list_routine_runs` 的默认排序是 `slot DESC`，而手动轮的 slot 名是
    #       "手动"（UTF-8 字节大于任何数字）⇒ 它永远排第一，"最近一轮"直接失真。
    #    ② **被跳过的格也是一行**，也带 `created_at`。拿它当"上一轮时间"时，
    #       gap 恒等于一个班距（40 分钟）< `FORCE_INTERVAL_MIN=80` ⇒ 跳过一格
    #       之后每一格都"距上一轮 40 分钟"继续跳过 —— 级联跳过，一天只剩 2 轮。
    #       实测（把已武装条件清空模拟平静日）：09:40 跑 → 10:20 跳 → 11:00 跳
    #       → 13:20 跑 → 14:00 跳 → 14:40 跳。理由是"距上一轮 40 分钟"，
    #       而它其实只是"距上一次**跳过**40 分钟"。
    today_runs = store.list_routine_runs(trade_date=today, persona=PERSONA_NAME, limit=50)
    by_time = sorted(today_runs, key=lambda r: str(r.get("created_at") or ""))
    llm_runs = [r for r in by_time if int(r.get("llm_called") or 0)]
    last_llm = llm_runs[-1] if llm_runs else {}
    last_at = str(last_llm.get("created_at") or "")
    # 上一次真实研判之后连续跳过了几轮（只数 skipped：error 轮不是"市场平静"，
    # 把它算进来会让"模型连续失败"伪装成"行情没动"）。
    skips_since_llm = 0
    for _r in reversed(by_time):
        if int(_r.get("llm_called") or 0):
            break
        if _r.get("status") == "skipped":
            skips_since_llm += 1
    if not force:
        worth, why = worth_llm(quotes, holdings=[h["code"] for h in holdings],
                               my_watch=[w["code"] for w in my_watch],
                               user_watch=[w["code"] for w in user_watch],
                               last_run_at=last_at, now=now,
                               skips_since_llm=skips_since_llm)
        if not worth:
            store.save_routine_run(PERSONA_NAME, today, slot, status="skipped",
                                   llm_called=False, skip_reason=why,
                                   autonomy=level)
            return {"skipped": why, "slot": slot, "llm_called": False}

    # ── 扣额度（在预筛**之后**）──
    granted, view = ai_budget.consume(PERSONA_NAME, "routine", trade_date=today)
    if not granted:
        store.save_routine_run(PERSONA_NAME, today, slot, status="skipped",
                               skip_reason=f"routine 额度不足（{view}）", autonomy=level)
        return {"skipped": f"routine 额度不足（{view}）", "slot": slot, "llm_called": False}

    # ── 一次 LLM（快速研判，不跑 12 节点深度分析）──
    prompt = build_prompt(
        now=now, slot=slot, level=level,
        auton_desc=ai_autonomy.LEVEL_DESC.get(level, ""),
        budget_desc=ai_autonomy.describe(params),
        cash=cash, nav=nav, holdings=holdings, my_watch=my_watch,
        user_watch=user_watch, quotes=quotes,
        candidates=_today_candidates(today),
        # 上一轮的**结论**只能来自真的调过模型的那一轮：跳过轮的 summary 是空的，
        # 拿它当"上一轮结论"会让 prompt 里的这一段变成"（今天还没有上一轮）"，
        # 模型于是每轮都像首轮一样重新规划。
        last_summary=str(last_llm.get("summary") or ""),
        # 今天已跑轮次的结论（= 判断轨迹）。不传它，模型看不到自己今天已经把
        # 同一句话说了 4 遍 —— 实测两两平均相似度 46%、第 5→6 轮 95%。
        trail=llm_runs,
        # 它自己**真的**挂着的等待条件。它嘴上说"在等 28 元"，条件表里却一条都
        # 没有过；摊开给它看，才能"沿用/覆盖/撤掉"，用户也才能核对。
        armed=_armed_conditions())
    try:
        from llm.deepseek_client import DeepSeekClient, model_override, TEMP_ANALYTICAL
        client = client_factory(persona) if client_factory else DeepSeekClient()
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[阿深值班] LLM 客户端创建失败（%s）: %s", slot, e)
        store.save_routine_run(PERSONA_NAME, today, slot, status="error",
                               skip_reason=f"LLM 客户端创建失败: {type(e).__name__}: {str(e)[:120]}",
                               autonomy=level)
        return {"skipped": f"LLM 客户端创建失败: {type(e).__name__}", "slot": slot,
                "llm_called": False}

    from core import ai_daily_round

    def _ask(text: str) -> dict:
        """调一次模型并解析成 `{actions, summary}`。重试走同一个客户端（省连接）。"""
        with model_override(persona.get("model")):
            raw = client.call_api([{"role": "user", "content": text}],
                                  temperature=TEMP_ANALYTICAL, max_tokens=800,
                                  purpose=f"ai.{PERSONA_NAME}.routine")
        return ai_daily_round.parse_actions_reply(raw)

    try:
        body = _ask(prompt)
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[阿深值班] LLM 调用失败（%s）: %s", slot, e)
        store.save_routine_run(PERSONA_NAME, today, slot, status="error",
                               skip_reason=f"LLM 调用失败: {type(e).__name__}: {str(e)[:120]}",
                               autonomy=level)
        return {"skipped": f"LLM 调用失败: {type(e).__name__}", "slot": slot,
                "llm_called": True}

    actions = _normalize(body.get("actions"), persona,
                         min_position=params["first_buy_pct"])
    summary = str(body.get("summary") or "").strip()[:200]

    # ── 动作闸门（见 action_gap 的注释：为什么这条必须是代码而不是 prompt）──
    enforced = ""
    gap = action_gap(summary, actions)
    if gap:
        try:
            body2 = _ask(prompt + "\n\n" + _retry_note(gap, summary))
            a2 = _normalize(body2.get("actions"), persona,
                            min_position=params["first_buy_pct"])
            s2 = str(body2.get("summary") or "").strip()[:200]
            # 只在补正**确实更好**时采用：不能因为一次重试把一条好结论换掉
            before = len([a for a in actions
                          if str(a.get("action") or "") in REAL_ACTIONS])
            after = len([a for a in a2
                         if str(a.get("action") or "") in REAL_ACTIONS])
            if after > before:
                _logger.info("[阿深值班] %s 补正生效：实质动作 %d → %d", slot, before, after)
                actions, summary = a2, (s2 or summary)
                enforced = f"{gap}→retry_ok"
            else:
                enforced = f"{gap}→retry_useless"
        except Exception as e:                              # noqa: BLE001
            # 补正失败不能连累这一轮：原结论仍然有效，只是缺口没补上。
            _logger.warning("[阿深值班] %s 补正调用失败: %s", slot, e)
            enforced = f"{gap}→retry_error"

        gap = action_gap(summary, actions)
        if gap == "no_action":
            # 重试也没动作 ⇒ 机械兜底（只做不动钱的那一类）。
            # 如实留痕：reason 里写明"这是系统兜底，不是它的判断" ——
            # 否则复盘会把一只它从没提过的票读成"它的选择"。
            extra = _auto_watch(_today_candidates(today), my_watch)
            if extra:
                actions = list(actions) + [extra]
                enforced = (enforced + "+auto_watch" if enforced else "auto_watch")
                _logger.info("[阿深值班] %s 无实质动作，已兜底入池 %s", slot, extra["code"])


    # ── 授权判据（建仓权在这里生效）──
    portfolio = ai_autonomy.portfolio_from_account(account, prices) if account else None
    verdict = ai_autonomy.autonomy_check(
        actions, level=level, params=params, portfolio=portfolio,
        my_watch=[w["code"] for w in my_watch],
        user_watch=[w["code"] for w in user_watch], quotes=quotes)

    results = []
    if execute and verdict["kept"]:
        try:
            results = _execute(verdict["kept"], account, today, quotes, nav)
            if account is not None:
                from core import ai_league
                ai_league._save(account)                    # 与联赛同一条保存路径
        except Exception as e:                              # noqa: BLE001
            _logger.warning("[阿深值班] 落地动作失败: %s", e)
            results = [{"error": f"{type(e).__name__}: {str(e)[:120]}"}]

    store.save_routine_run(
        PERSONA_NAME, today, slot, status="ok", llm_called=True,
        summary=summary, actions=verdict["kept"], blocked=verdict["blocked"],
        autonomy=level, data_through=_data_through(now),
        model=str(persona.get("model") or ""),
        # 空串 = 它自己就给了实质动作；非空 = 补正/兜底过（见建表注释）
        enforced=enforced)
    return {"slot": slot, "llm_called": True, "level": level,
            # `kept` 口径 = **真实可执行的动作数**（不含兜底 pass）。
            # 混淆这两者会让"被拦 1 笔"看起来像"执行了 1 笔"。
            "kept": len([k for k in verdict["kept"] if k.get("action") != "pass"]),
            "blocked": len(verdict["blocked"]),
            "summary": summary, "orders": results,
            # 动作闸门的处置。调用方/复盘要分清"它自己想动的"与"系统推它动的"，
            # 不能只看 actions_json 里那条 watch_add —— 那可能是兜底出来的。
            "enforced": enforced,
            "violations": verdict["violations"]}


def _data_through(now: datetime) -> str:
    """这一轮可用的日线数据截止日。

    ⚠️ 盘中当天的日线还不存在（收盘价没生成、量能只有半天），所以截止日是
    **上一交易日**。这是**推导值**，不是读 parquet 得到的实际值 —— 实时读
    盘口那几轮会为了一个字符串多开一次文件，不值。
    落这一列的目的正是让事后复盘能分清"这条判断用的是哪天的数据"。
    """
    try:
        from data.trading_calendar import last_trading_day
        return str(last_trading_day(now.date(), include_today=False))
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[阿深值班] 交易日历不可用，data_through 记为未知: %s", e)
        return ""


def history(trade_date: str | None = None, limit: int = 50) -> dict:
    """当日值班记录 + 汇总（供 desk 页面用）。零联网。"""
    from db import store
    day = trade_date or datetime.now().strftime("%Y-%m-%d")
    rows = store.list_routine_runs(trade_date=day, persona=PERSONA_NAME, limit=limit)
    return {"trade_date": day, "persona": PERSONA_NAME, "slots": list(SLOTS),
            "runs": rows,
            "counts": {"total": len(rows),
                       "llm_called": sum(1 for r in rows if r.get("llm_called")),
                       "skipped": sum(1 for r in rows if r.get("status") == "skipped"),
                       "blocked": sum(len(r.get("blocked") or []) for r in rows),
                       # 手动补跑单列一格：把"排班跑了 6 轮"和"人按了 3 次按钮"
                       # 混成一个 total，事后无法判断系统自己有没有按点到岗。
                       "manual": sum(1 for r in rows if r.get("slot") == MANUAL_SLOT),
                       # ⚠️ 页面「今日轮次 N/6」的分子必须是**这一个**，不是 `total`。
                       #    `total` 含 skipped 与手动轮：17:00 手动演练一次就让
                       #    "N/6" 从 0 变成 1，看上去像排班已经到岗；而排班被跳过的
                       #    格子同样计入 total，又会让它虚高。两个方向的失真都有。
                       "scheduled": sum(1 for r in rows if r.get("slot") in SLOTS),
                       "scheduled_llm": sum(1 for r in rows
                                            if r.get("slot") in SLOTS and r.get("llm_called"))}}
