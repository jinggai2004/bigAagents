"""自定义定时任务（L2 扩展层）：把「什么时候跑」从代码变成数据。

为什么需要
----------
内置任务写死在 `core/scheduler.py::JOBS`，任何改动都要「改代码 + 重启进程」。
而"重启"这件事本项目真实缺过两次：

1. 新增 `risk_check` 后没重启 → 健康检查的 `code` 段报"跑的是旧代码"；
2. `snapshot_flow`(15:40) 与 `daily_review`(15:45) 只隔 5 分钟，而快照慢时跑 9 分钟
   → 复盘**重复采集**一次（9-28 实测 15:49 与 15:53 各一次）。

对照 vibe-trading 的调度（REST 建任务 + playbook 模板 + 持久化）后，本项目要抄的
**不是"更多任务"，而是"任务清单是数据不是代码"**。所以这一层只做三件事：

- `scheduled_jobs` 表持久化（db/store.py）
- 调度进程每 60s `sync_custom_jobs()` 热加载 → **改完不用重启**
- 与内置任务共用同一批执行函数（`core.scheduler.run_job`），不新增执行器

刻意 **不支持"任意 prompt 定时执行"**
--------------------------------
LLM 调用是真实成本，自由文本的定时任务等于给账单开一个无人看守的阀门。
所以 `mode` 必须落在 `JOB_MODES` 白名单内，深度分析（`pipelines` 的 LLM 链路）
不在其中，只能由人在界面上手动触发。

与健康检查的边界（关键）
--------------------
健康检查靠"心跳里的任务清单 vs `JOB_IDS`"来判断"进程跑的是不是最新代码"。
自定义任务是**合法地多出来**的任务，如果混进 `job_ids` 就会让健康检查天天告警，
从守护者变成"狼来了"。因此：
- `job_ids` 只写**内置**任务（语义不变）
- 自定义任务单独走 `custom_job_ids` 字段，只报告、不告警
"""

import logging
import re
from datetime import datetime

_logger = logging.getLogger(__name__)

# ── 可被调度的执行逻辑白名单 ──
#
# label          界面展示名
# needs_trading_day  依赖当日行情，非交易日必须跳过（与 run_job 的判定保持同源）
# llm            会调用大模型（界面要提醒成本）
# minutes        典型耗时的**上界估计**（分钟），只用于冲突检测的窗口重叠判断。
#                宁可估大：估小的后果是"以为排得开、实际撞车"，正是本项目
#                踩过的坑（snapshot_flow 15:40 排下去、慢时跑 9 分钟，
#                把 15:45 的复盘挤到重复采集）。界面明确标注这是估算值。
JOB_MODES: dict = {
    "screen": {
        # ⚠️ 2026-10-24：6 → 12 分钟。原值"约 256 秒"是扩池前的旧数据；扩池后
        #    实测 `stage_timings.screen.run` = **695.87s（11.6 分钟）**。
        #    估小的直接后果：检测器以为 09:25+6min=09:31 早于 09:32 的开盘结算，
        #    于是**永远不报这对冲突**，而实际每天都会重叠 5 分钟。
        "label": "全市场选股", "needs_trading_day": True, "llm": False, "minutes": 12,
        "desc": "五路本地初筛（全市场快照）→ 六因子精筛 → TOP5；实测一轮约 696 秒",
    },
    "rule": {
        "label": "盘后全流程（选股 + 规则回测）", "needs_trading_day": True,
        # ⚠️ 2026-10-24：15 → 23 分钟。实测日志 `afternoon_full` 一轮
        #    **1371.6s（22.9 分钟）**，原值 15 分钟让 15:50 的复盘看起来"排得开"，
        #    而实际 10-23 就是"复盘 15:50 起跑时 afternoon_full 还没结束（15:52 才完）"
        #    ⇒ 复盘读到的是上一份 `daily_picks`。
        "llm": False, "minutes": 23,
        "desc": "零 LLM 成本；实测约 22.9 分钟（1371.6s），安排时按 23 分钟留余量",
    },
    "review": {
        "label": "每日复盘（生成报告并推送）", "needs_trading_day": True,
        "llm": True, "minutes": 3,
        "desc": "⚠️ 会调用大模型；重复排期会重复消耗额度",
    },
    "snapshot": {
        "label": "全市场资金流快照采集", "needs_trading_day": True,
        "llm": False, "minutes": 10,
        "desc": "单次采集约 77 秒，但实测慢时可达 9 分钟（全市场 5210 行 × 4 窗口）",
    },
    "sentiment": {
        "label": "市场情绪参照采集（外部言论）", "needs_trading_day": True,
        "llm": False, "minutes": 1,
        # ⚠️ 零 LLM：只做"采集 + 规则层聚合 + 落库"，定性判定由用户在页面上按需触发。
        #    排它的唯一目的是**积累历史** —— 相对基准要求 ≥5 个交易日，
        #    靠用户手动开页面永远凑不齐（这是该层能否成立的前提，不是锦上添花）。
        "desc": "按配置的博主/标的采集公开言论，算多空比/分歧度/覆盖率并落库；"
                "不调模型、不推送；取不到时只记 WARNING，不漏到用户面前",
    },
    "risk": {
        "label": "持仓风控巡检", "needs_trading_day": True, "llm": False, "minutes": 2,
        "desc": "只检查已录入持仓；无持仓时安静跳过",
    },
    "backfill": {
        "label": "分析历史 T+N 回填", "needs_trading_day": True,
        "llm": False, "minutes": 3,
        "desc": "只拉 K 线补 ret_5d/ret_20d，零 LLM 成本",
    },
    "backfill_screen": {
        "label": "选股候选 T+N 回填", "needs_trading_day": True,
        "llm": False, "minutes": 3,
        "desc": "与上一条锁名不同，可并行",
    },
    "news": {
        "label": "新闻快讯采集", "needs_trading_day": False, "llm": False, "minutes": 2,
        "desc": "与交易日无关，节假日也采",
    },
    "news_review": {
        "label": "新闻行业利好/利空评估", "needs_trading_day": False,
        "llm": True, "minutes": 4,
        "desc": "⚠️ 会调用大模型；聚合近24h快讯按行业评估，利好行业给出成分股建议。"
                "每天跑（周末快讯也有价值），15:55 内置排期",
    },
    "watch": {
        "label": "盘中异动预警", "needs_trading_day": True, "llm": False, "minutes": 1,
        "desc": "启停由交易时段判定（09:30-11:30 / 13:00-15:00），排期只需覆盖这两段",
    },
    "ai_round": {
        "label": "AI交易联赛·盘后决策轮", "needs_trading_day": True,
        "llm": True, "minutes": 6,
        "desc": "⚠️⚠️ **本项目唯一「每跑一次必花钱」的任务**：3 个 LLM persona × 最多 6 轮"
                "工具循环（实测单次调用 4.1~4.6k tokens）。3 个随机对照组零 LLM 成本。"
                "读 15:30 选股 / 15:55 新闻评估的产物组装简报，产出明日决策与唤醒条件。"
                "内置排期 15:58（晚于新闻评估）。",
    },
    "ai_settle_open": {
        "label": "AI交易联赛·开盘结算(T+1)", "needs_trading_day": True,
        "llm": False, "minutes": 4,
        "desc": "零 LLM 成本。按**当日开盘价**撮合 T-1 及更早的买入决策（T+1 口径，"
                "与回测同源）。无行情（停牌）时**不成交且不伪造价格**。",
    },
    "ai_settle_close": {
        "label": "AI交易联赛·收盘结算(日内动作)", "needs_trading_day": True,
        "llm": False, "minutes": 4,
        "desc": "零 LLM 成本。处理盘中唤醒的**卖出与持仓内小额度补仓**"
                "（**扫描价+滑点**，口径 2，同时留存触发价）。补仓受三重边界约束"
                "（仅持仓标的 / 仅 price_below / 单次≤持仓市值3%且≤权益1%、"
                "单票1次/日、全日≤2次且≤权益2%），成交带 `intraday_buy` 标记并作幂等锚点。"
                "⚠️ 日内成交隐含『触发即成交』假设 ⇒ 日内收益被系统性高估"
                "（页面必须展示该 caveat）。⚠️ 记净值不在此任务（15:05 早于当日日线可用时间"
                "⇒ 会退化成成本价兜底、产出假净值），已拆到 15:52 的 `ai_mark_nav`。",
    },
    "llm_analysis": {
        "label": "盘后深度分析", "needs_trading_day": True,
        "llm": True, "minutes": 15,
        # ⚠️ 只要内置任务用（18:10 一次，带只数+token 上限）。
        #    **不开放给自定义定时任务**：用户一天建 10 个"深度分析"就会把账单打爆
        #    —— 这是本项目"LLM 重链路不进白名单"的既有原则，mode 登记不等于可调度。
        "schedulable": False,
        "desc": "对**当日已有候选**跑批量深度分析（不重新选股），只数上限 8、"
                "当日 token 上限 120 万；单只失败不中断整批。排期 18:10 —— "
                "DeepSeek 峰谷定价的空闲时段（半价），且避开 15:30~16:00 的 LLM 重任务。",
    },
    "ai_mark_nav": {
        "label": "AI交易联赛·当日净值", "needs_trading_day": True,
        "llm": False, "minutes": 4,
        "desc": "零 LLM 成本。用**当日收盘价**（`market_snapshot` 15:40 快照优先、日线兜底）"
                "记各 AI 账户净值。取不到价的账户**当天不记**并告警（缺就是缺，"
                "绝不用成本价冒充——2026-10-15 该缺陷曾让 AI 复盘得出『净值不变、无盈亏』）。"
                "排期 15:52：晚于快照(15:40)、早于决策轮(15:58)。",
    },
}

DEFAULT_MINUTES = 3

# 与 core.scheduler.run_job 的判定保持同源：这些 mode 依赖当日行情，
# 非交易日一律跳过（"缺就是缺"，节假日跑出来的"今日选股"与真实交易日不可区分）。
TRADING_DAY_MODES = tuple(
    m for m, cfg in JOB_MODES.items() if cfg["needs_trading_day"] and m != "watch")

DEFAULT_DAY_OF_WEEK = "mon-fri"

_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
_DOW_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_DOW_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
# 全名 → 缩写。**只接受这两个精确形式**：早先想用 `part[:3]` 兼容 "monday"，
# 结果 "monkey" / "mondayish" / "mon-fri-xx" 全被静默当成周一或 mon-fri。
# 排期解析必须"要么准确、要么报错"，模糊匹配在这里等于埋雷。
_DOW_ALIAS = {"monday": "mon", "tuesday": "tue", "wednesday": "wed",
              "thursday": "thu", "friday": "fri", "saturday": "sat",
              "sunday": "sun"}
_WEEKDAY_ALIAS = {"weekday": "mon-fri", "工作日": "mon-fri", "每天": "*", "daily": "*"}


def _day_key(part: str):
    """单个星期词 → 索引；不认识返回 None（不猜）"""
    p = str(part or "").strip().lower()
    p = _DOW_ALIAS.get(p, p)
    return _WEEKDAYS.get(p)


# ================================================================
# cron 校验与展示
# ================================================================

def _expand_days(value, default: str = DEFAULT_DAY_OF_WEEK):
    """展开「星期」字段为 {0=周一 … 6=周日}。无法解析返回 None。

    支持 `mon-fri` / `mon,wed,fri` / `*` / 大小写与中文别名。
    ⚠️ 必须处理 `a-b` 区间：最初写成 `part[:3]`，于是 `"mon-fri"` 被截成
    `"mon"` → 排期**静默变成每周一**（不报错、界面还显示"周一至周五"）。
    这类"时间悄悄错掉"的 bug 没有任何地方会报警，所以专门写了测试钉住。
    """
    text = str(value).strip().lower() if value is not None else ""
    if not text:
        text = str(default or "").strip().lower()     # 缺省 → 用默认（内置任务即 mon-fri）
    text = _WEEKDAY_ALIAS.get(text, text)
    if text in ("*", "?"):
        return set(range(7))
    days: set = set()
    for part in re.split(r"[,，]", text):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            ia, ib = _day_key(a), _day_key(b)
            if ia is None or ib is None:
                return None
            if ia <= ib:
                days.update(range(ia, ib + 1))
            else:                               # 跨周写法（如 fri-mon）
                days.update(range(ia, 7))
                days.update(range(0, ib + 1))
        else:
            idx = _day_key(part)
            if idx is None:
                return None
            days.add(idx)
    return days or None


def _format_days(days: set) -> str:
    """星期集合 → 规范字符串（同一天集合的表示唯一，便于比对"设置是否变过"）"""
    if set(days) == set(range(7)):
        return "*"
    if set(days) == {0, 1, 2, 3, 4}:
        return "mon-fri"
    return ",".join(_DOW_NAMES[d] for d in sorted(days))


def normalize_cron(cron: dict) -> dict:
    """校验并规范化 cron 字典。**不合法就抛 ValueError**（带可直接展示的中文原因），
    绝不悄悄塞一个默认时间——否则用户会看到一个自己没设置过的执行时刻。"""
    if not isinstance(cron, dict):
        raise ValueError("cron 必须是字典，例如 {\"hour\":15,\"minute\":5}")

    def _num(key, lo, hi):
        raw = cron.get(key)
        if raw in (None, ""):
            raise ValueError(f"缺少 {key}")
        if isinstance(raw, str) and raw.strip() in ("*", "?"):
            return "*"
        try:
            v = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key} 必须是 {lo}-{hi} 的整数或 *，收到 {raw!r}")
        if not (lo <= v <= hi):
            raise ValueError(f"{key} 超出范围（{lo}-{hi}）：{v}")
        return v

    hour = _num("hour", 0, 23)
    minute = _num("minute", 0, 59)

    days = _expand_days(cron.get("day_of_week", DEFAULT_DAY_OF_WEEK))
    if days is None:
        raise ValueError(f"day_of_week 无法解析：{cron.get('day_of_week')!r}；"
                         f"可用 mon/tue/wed/thu/fri/sat/sun，"
                         f"或区间 mon-fri，或 * 表示每天")

    return {"hour": hour, "minute": minute, "day_of_week": _format_days(days)}


def describe_cron(cron: dict) -> str:
    """人类可读的排期描述（界面/日志共用）"""
    if not isinstance(cron, dict):
        return "（无法解析）"
    hours, minutes = cron.get("hour"), cron.get("minute")
    days = _expand_days(cron.get("day_of_week", DEFAULT_DAY_OF_WEEK))
    if not days:
        return "（无法解析）"
    if len(days) == 7:
        day_text = "每天"
    elif days == {0, 1, 2, 3, 4}:
        day_text = "周一至周五"
    else:
        day_text = "、".join(_DOW_CN[d] for d in sorted(days))
    if hours == "*":
        if isinstance(minutes, int):
            return f"{day_text}每小时 {minutes:02d} 分"
        return f"{day_text}每小时"
    return f"{day_text} {hours:02d}:{minutes:02d}"


def expand_cron(cron: dict, default_dow: str = DEFAULT_DAY_OF_WEEK):
    """→ (星期集合, 小时集合, 分钟集合)。无法解析时返回 None。

    用于冲突检测：把 "9-11,13-15" 这类内置任务的字符串字段一并展开，
    才能发现"自定义任务排到了盯盘区间"这种情况。
    """
    if not isinstance(cron, dict):
        return None
    hours = _expand_field(cron.get("hour"), 0, 23)
    minutes = _expand_field(cron.get("minute"), 0, 59)
    days = _expand_days(cron.get("day_of_week", default_dow), default_dow)
    if hours is None or minutes is None or not days:
        return None
    return days, hours, minutes


def _expand_field(value, lo: int, hi: int):
    """展开 cron 字段为整数集合；支持 int / "5" / "*" / "9-11,13-15"。失败返回 None"""
    if isinstance(value, int):
        return {value} if lo <= value <= hi else set()
    text = str(value if value is not None else "").strip().lower()
    if text in ("*", "?"):
        return set(range(lo, hi + 1))
    if not text:
        return None
    out: set = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                a, b = int(a), int(b)
            except ValueError:
                return None
            if a > b:
                a, b = b, a
            out.update(range(a, b + 1))
        else:
            try:
                out.add(int(part))
            except ValueError:
                return None
    return {v for v in out if lo <= v <= hi}


def next_fire_time(cron: dict, after: datetime = None):
    """按 cron 预估下次执行时间（纯计算，不依赖调度器状态）。

    ⚠️ 只在"启动前预估"与界面预告里用它。**不要去读 `job.next_run_time`**：
    APScheduler 未启动时 Job 根本没有那个属性（不是 None，是 AttributeError），
    正是这个坑让本项目调度器五个月没跑过一次任务。
    """
    from apscheduler.triggers.cron import CronTrigger
    try:
        trigger = CronTrigger(**normalize_cron(cron))
    except ValueError:
        return None
    try:
        return trigger.get_next_fire_time(None, after or datetime.now())
    except Exception:                           # noqa: BLE001
        return None


# ================================================================
# 任务定义校验 / 生成
# ================================================================

def make_job_id(name: str, taken=None) -> str:
    """由名称生成 `custom_xxx` 形式的 job_id，并避开已占用的 id。

    为什么要避开：`job_id` 就是调度器里的唯一键。/once 之类按 id 精确取任务的
    逻辑、以及健康检查的清单比对，都假设内置 id 稳定——自定义任务一旦撞名
    就会把内置任务顶掉（`replace_existing=True` 是覆盖，不是报错）。
    """
    taken = set(taken or ())
    base = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    base = f"custom_{base}" if base else "custom_job"
    base = base[:48]
    if base not in taken:
        return base
    i = 2
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


def builtin_job_ids() -> list:
    """内置任务 id（延迟 import，避免把 pipeline 依赖拖进轻量调用方）"""
    try:
        from core.scheduler import JOB_IDS
        return list(JOB_IDS)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[自定义任务] 读取内置任务清单失败: %s", e)
        return []


def validate_job(job: dict, taken_ids=None, require_id: bool = False) -> dict:
    """校验并补全一条任务定义。不合法抛 ValueError。返回可直接入库的 dict。"""
    name = str(job.get("name") or "").strip()
    if not name:
        raise ValueError("任务名称不能为空")
    if len(name) > 40:
        raise ValueError("任务名称过长（建议 ≤ 40 字）")

    mode = str(job.get("mode") or "").strip()
    if mode not in JOB_MODES:
        raise ValueError(f"不支持的任务类型「{mode}」；可选："
                         + "、".join(f"{k}（{v['label']}）" for k, v in JOB_MODES.items()))

    cron = normalize_cron(job.get("cron") or {})

    job_id = str(job.get("job_id") or "").strip()
    taken = set(taken_ids if taken_ids is not None else builtin_job_ids())
    if job_id:
        if job_id in builtin_job_ids():
            raise ValueError(f"'{job_id}' 是内置任务 id，会把它顶掉，请换一个名字")
        if job_id in taken and job_id != str(job.get("_original_id") or ""):
            raise ValueError(f"任务 id「{job_id}」已存在")
    elif require_id:
        raise ValueError("缺少 job_id")
    else:
        job_id = make_job_id(name, taken)

    # 依赖当日行情的 mode，非交易日必须跳过——即使界面上勾了"非交易日也跑"也不放行。
    # 否则节假日会产出与真实交易日不可区分的"今日选股"（本项目专门修过这个问题）。
    forced = mode in TRADING_DAY_MODES
    tdo = True if forced else bool(job.get("trading_day_only", True))

    return {
        "job_id": job_id,
        "name": name,
        "mode": mode,
        "cron": cron,
        "enabled": bool(job.get("enabled", True)),
        "trading_day_only": tdo,
        "trading_day_locked": forced,
        "note": str(job.get("note") or "").strip()[:200],
    }


# ================================================================
# 冲突检测（复盘重复采集那次事故的自动化版本）
# ================================================================

# 这两个任务**刻意高频**（盯盘每分钟、新闻每小时 :05），与它们时间重叠是设计使然，
# 拿它们报警只会淹没真正有用的提示。所以排除在冲突检测之外。
_CONFLICT_EXEMPT = {"intraday_watch", "hourly_news"}


def conflict_candidates(custom_jobs=None) -> list:
    """冲突检测的比对基准：内置（排除高频项）+ 其它自定义任务"""
    out = []
    try:
        from core.scheduler import JOBS
    except Exception:                           # noqa: BLE001
        JOBS = []
    for job_id, cron_kwargs, mode, desc in JOBS:
        if job_id in _CONFLICT_EXEMPT:
            continue
        out.append({"job_id": job_id, "name": desc, "mode": mode,
                    "cron": dict(cron_kwargs), "builtin": True,
                    "minutes": _mode_minutes(mode)})
    for j in (custom_jobs or []):
        if j.get("job_id") in _CONFLICT_EXEMPT:
            continue
        out.append({"job_id": j.get("job_id"), "name": j.get("name"),
                    "mode": j.get("mode"), "cron": j.get("cron") or {},
                    "builtin": False, "minutes": _mode_minutes(j.get("mode"))})
    return out


def _mode_minutes(mode) -> int:
    return (JOB_MODES.get(mode) or {}).get("minutes") or DEFAULT_MINUTES


def occupied_minutes(cron: dict, minutes: int = None):
    """把一条排期展开成 {星期: 被占用的当日分钟集合}。

    用**窗口**而不是"同一分钟"，是因为真实事故恰恰是"起始时间不同、执行窗口重叠"：
    `snapshot_flow` 15:40 起跑、慢时跑 9 分钟 → 压到 15:45 的 `daily_review`，
    结果复盘重复采集了一次（9-28 实测 15:49 与 15:53 各一次）。
    只比"起始分钟是否相同"的话，这个坑一个都抓不到。
    """
    exp = expand_cron(cron)
    if exp is None:
        return None
    days, hours, mins = exp
    dur = max(1, int(minutes if minutes is not None else DEFAULT_MINUTES))
    out = {}
    for d in days:
        slots: set = set()
        for h in hours:
            for m in mins:
                start = h * 60 + m
                slots.update(range(start, min(start + dur, 1440)))
        out[d] = slots
    return out


def find_conflicts(cron: dict, mode: str = "", exclude_id: str = "",
                   custom_jobs=None, minutes: int = None) -> list:
    """找出与现有任务**执行窗口重叠**的排期。

    Returns: [{"job_id","name","mode","builtin","same_mode","overlap","minutes"}]

    `same_mode=True` → 同一份重活被排两次（重复拉数 / 抢同一把锁），必须改
    `same_mode=False` → 资源争夺（"快照跑 9 分钟把复盘挤到重复采集"就是这一类）

    耗时为**估算上界**（见 JOB_MODES 的 minutes），界面必须如实说明这是估算，
    不能包装成精确判断。
    """
    target = occupied_minutes(cron, minutes if minutes is not None else _mode_minutes(mode))
    if not target:
        return []
    hits = []
    for ref in conflict_candidates(custom_jobs):
        if ref["job_id"] == exclude_id:
            continue
        other = occupied_minutes(ref["cron"], ref["minutes"])
        if not other:
            continue            # 内置字符串字段解析不了 → 不猜、不报
        common = set(target) & set(other)
        first = None
        for d in sorted(common):
            both = target[d] & other[d]
            if both:
                first = (d, min(both))
                break
        if first is None:
            continue
        hh, mm = divmod(first[1], 60)
        hits.append({
            "job_id": ref["job_id"],
            "name": ref["name"],
            "mode": ref["mode"],
            "builtin": ref["builtin"],
            "minutes": ref["minutes"],
            "same_mode": bool(mode) and ref["mode"] == mode,
            "overlap": f"{_DOW_CN[first[0]]} {hh:02d}:{mm:02d}",
        })
    # 同一任务的重复执行排前面（更要紧），其余按名字稳定排序
    hits.sort(key=lambda h: (not h["same_mode"], h["job_id"]))
    return hits


# ================================================================
# 与调度器的同步（热加载）
# ================================================================

def build_trigger(cron: dict):
    from apscheduler.triggers.cron import CronTrigger
    return CronTrigger(**normalize_cron(cron))


def _job_args(job: dict) -> list:
    return [job["mode"], job["name"], bool(job.get("trading_day_only", True))]


def register_job(scheduler, job: dict):
    """把一条自定义任务挂到（可能已在运行的）调度器上。

    ⚠️ 必须先 `remove_job` 再 `add_job`，**不能只靠 `replace_existing=True`**：
    APScheduler 在**未启动**的调度器上 `add_job` 只是往 `_pending_jobs` 里追加一条，
    此时 `replace_existing` 不生效——同一个 id 会被追加两次，启动后**跑两遍**。
    `create_scheduler()` 装载自定义任务时调度器正是未启动状态（启动前预挂任务），
    所以这个坑一定会被踩到。显式先删既覆盖"未启动"（删 pending 列表里的项）
    也覆盖"运行中"（删 jobstore 里的项）。
    """
    from core.scheduler import run_job
    try:
        scheduler.remove_job(job["job_id"])
    except Exception:                           # noqa: BLE001
        pass                                    # 不存在即无需删除
    scheduler.add_job(
        run_job, build_trigger(job["cron"]), args=_job_args(job),
        id=job["job_id"], name=job["name"],
        misfire_grace_time=3600, coalesce=True, replace_existing=True,
    )


def sync_custom_jobs(scheduler, jobs=None, builtin_ids=None) -> dict:
    """把库里的自定义任务同步到调度器：新增 / 重排 / 移除。

    - `jobs=None` 时从库里读（enabled=1 的才算"应存在"）
    - 调度进程每 60s 调一次；**任何异常都不能冒泡**（同步失败不该拖垮调度主循环），
      由调用方兜住。这里只保证"读库失败时不误删已有任务"。
    """
    if jobs is None:
        from db import store as db_store
        try:
            jobs = db_store.list_scheduled_jobs(only_enabled=True)
        except Exception as e:                  # noqa: BLE001
            # 读不到"期望集合"时**必须原样返回**，绝不能把空集合当成意图：
            # 那会执行"库里没有 → 从调度器移除全部"的删除路径，任务再也不会跑，
            # 而调度器看起来一切正常（心跳新鲜、进程活着）。
            _logger.warning("[自定义任务] 读取任务表失败，本轮不同步（保留现有任务）: %s", e)
            return {"added": [], "updated": [], "removed": [], "invalid": [],
                    "changed": False, "total": None, "error": str(e)}
    builtin = set(builtin_ids if builtin_ids is not None else builtin_job_ids())

    desired, invalid = {}, []
    for job in jobs:
        jid = job.get("job_id")
        if not jid or jid in builtin:
            continue
        if not job.get("cron") or job.get("error"):
            invalid.append({"job_id": jid, "error": job.get("error") or "缺少 cron"})
            continue
        try:
            validate_job({**job, "job_id": jid, "_original_id": jid},
                         taken_ids=builtin, require_id=True)
        except ValueError as e:
            invalid.append({"job_id": jid, "error": str(e)})
            continue
        desired[jid] = job

    try:
        current = {j.id: j for j in scheduler.get_jobs() if j.id not in builtin}
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[自定义任务] 读取调度器现有任务失败，本轮不同步: %s", e)
        return {"added": [], "updated": [], "removed": [], "invalid": invalid,
                "changed": False, "total": len(desired)}

    added, updated, removed = [], [], []
    for jid, job in desired.items():
        try:
            if jid not in current:
                register_job(scheduler, job)
                added.append(jid)
                continue
            running = current[jid]
            same_args = list(running.args or []) == _job_args(job)
            try:
                same_trigger = str(running.trigger) == str(build_trigger(job["cron"]))
            except ValueError:
                same_trigger = False
            if not (same_args and same_trigger):
                register_job(scheduler, job)     # replace_existing → 原地替换
                updated.append(jid)
        except Exception as e:                  # noqa: BLE001
            _logger.warning("[自定义任务] 注册 %s 失败: %s", jid, e)

    for jid in current:
        if jid in desired:
            continue
        # 只删"我们管得着的"任务：库里已停用/已删除的。其它来源的任务不动，
        # 否则同步逻辑就成了会误伤别人的清理器。
        try:
            scheduler.remove_job(jid)
            removed.append(jid)
        except Exception as e:                  # noqa: BLE001
            _logger.warning("[自定义任务] 移除 %s 失败: %s", jid, e)

    result = {"added": added, "updated": updated, "removed": removed,
              "invalid": invalid, "total": len(desired),
              "changed": bool(added or updated or removed)}
    if result["changed"]:
        _logger.info("[自定义任务] 热加载完成：新增 %s / 重排 %s / 移除 %s（共 %d 个）",
                     added, updated, removed, len(desired))
    return result


def load_custom_jobs(scheduler, builtin_ids=None) -> int:
    """启动时装载自定义任务（best-effort）。

    ⚠️ 必须吞掉所有异常：库损坏/表缺失/字段异常都不该让调度器起不来。
    本项目最大的事故就是"调度器根本没启动而无人察觉"——绝不能让新增的一层
    配置成为新的启动失败点。
    """
    try:
        return sync_custom_jobs(scheduler, builtin_ids=builtin_ids)["total"]
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[自定义任务] 启动装载失败（不影响内置任务）: %s", e)
        return 0
