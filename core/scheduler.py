"""
定时调度器：基于 APScheduler 的每日自动化。

调度时刻（交易日）——**完整表见下方 `JOBS`，这里只画主链**：
  - 08:05  财报公告日刷新（PIT 前视防护的数据底座）
  - 08:15  盘前选股（screen，P90 54 分钟 ⇒ 开盘前出结果）
  - 09:45  AI 联赛开盘结算（T+1，按当日开盘价）
  - 13:30  选股候选 T+N 回填（不依赖当日收盘，挪出盘后窗口）
  - 14:40/14:45  持仓风控巡检 / 左侧监测盘中预扫
  - 15:20  市场情绪参照采集（必须早于复盘）
  - 15:22  分析历史 T+N 回填（含 T+5/T+20 里程碑 LLM 归因）
  - 15:30  盘后全流程（rule：选股 + 规则回测，零 LLM；**全天最长，P90 57 分钟**）
  - 15:40  全市场资金流/估值快照（当日日线落定后）
  - 16:30  起 账本与报告链：AI 联赛收盘结算+净值 → 左侧监测收盘 → 复盘 →
           新闻行业评估 → AI 联赛盘后决策轮 → 交易拥挤度（16:55）
  - 17:05  超额衰减监控（**仅周五**）
  - 18:10  盘后批量深度分析（DeepSeek 半价时段）

⚠️ 排期纪律（2026-11-01 重排后确立）：**下游排在上游 P90 之后，不是中位、更不是
估算**。近两周实测 P90：盘前选股 53.5 分钟、盘后全流程 57.4 分钟、午间选股 49 分钟
——旧表按 12/23/12 分钟排，于是 15:40~16:30 的七个任务全压在盘后全流程身上。

说明：
  - 交易日保护：cron 限定周一至周五，**法定节假日由 data/trading_calendar 判定**
    （Phase 1 新增）。原先只靠"数据层自然降级"，节假日照样选股并推送，
    产出的"今日选股/复盘"与交易日不可区分。
  - LLM 深度分析不进定时任务（成本高），由 UI「深度分析」页手动触发；
    但分析历史落库发生在分析链内部（UI/pipeline 共用），回填任务是零 LLM 成本。
"""

import logging
import json
import os
from datetime import date, datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from core.pipeline import DailyPipeline
from data.trading_calendar import is_trading_day

_logger = logging.getLogger(__name__)

# 任务定义: (job_id, cron 表达式, 任务类型, 描述)
# mode/任务类型: screen / rule / review / news / snapshot / backfill(_screen) / watch
# 任务定义: (job_id, cron 表达式, 任务类型, 描述)
# mode/任务类型: screen / rule / review / news / snapshot / backfill(_screen) / watch …
#
# ⚠️⚠️ 2026-11-01 全表重排：依据是**近两周调度日志的真实耗时**，不是估算。
#   | 任务         | 旧估算 | 中位  | P90   | 最大  | n  |
#   | 盘前选股      | 12 分  | 24.9 | 53.5 | 54.0 | 18 |
#   | 盘后全流程    | 23 分  | 21.6 | 57.4 | 73.4 | 22 |
#   | 午间选股复核   | 12 分  | 11.4 | 49.0 | 75.3 | 21 |
#   旧表把 50~57 分钟的任务当 12~23 分钟排 ⇒ 15:40~16:30 的 7 个任务**全部**
#   落在 `afternoon_full` 的窗口内，"错峰"只存在于纸面上（实测每天挂一条
#   "上游未就绪 afternoon_full"）。本次三件事：
#     ① 删掉第三次全市场选股（`noon_screen`，与 morning_screen 同 mode 同产物）；
#     ② 把不依赖当日收盘的回填挪出盘后窗口；
#     ③ 下游一律排到上游 **P90** 之后（不是中位、更不是旧估算）。
JOBS = [
    # ── 早间（盘前）──────────────────────────────────────────
    # 财报公告日刷新（PIT 前视防护，2026-10-30）：**工作日** 08:05。
    # 必须早于 08:15 的盘前选股（后者读基本面要按 as_of 判定"当时可见"）。
    # 为什么不进交易日闸门：公告日不依赖当日行情，节假日也会披露（周末公告在
    #   下一个工作日一并补上——只会"晚用"不会"早用"，方向上安全）。
    ("fin_disclosure", dict(hour=8, minute=5), "fin_disclosure", "财报公告日刷新(PIT)"),
    # 盘前选股：**08:15**（2026-11-01，原 09:25）。
    # 原时刻与实测严重不符：P90 53.5 分钟 ⇒ 09:25 起跑要跑到 **10:18**，即
    #   "盘前选股"在开盘 48 分钟后才出结果，还跨过 09:45 的开盘结算。
    # 08:15 起跑 ⇒ P90 也在 09:09 前完成，真正在开盘前给结论。
    # 它用的是**上一交易日收盘**的全市场快照（15:40 落库），与几点执行无关。
    ("morning_screen", dict(hour=8, minute=15), "screen", "盘前选股"),
    # AI 联赛开盘结算（T+1）：09:45。撮合用"当日开盘价"，而开盘价 09:30 就已确定，
    # 所以只需要"开盘之后"；原 09:32 曾与盘前选股每天重叠。
    ("ai_settle_open", dict(hour=9, minute=45), "ai_settle_open", "AI交易联赛·开盘结算(T+1)"),

    # ── 午间 ───────────────────────────────────────────────
    # AI 联赛午间决策轮：13:05。既避开上午算力高峰，又能用上午真实量价。
    ("ai_midday_round", dict(hour=13, minute=5), "ai_round", "AI交易联赛·午间决策轮"),
    # 选股候选 T+N 回填：**13:30**（2026-11-01，原 15:10）。
    # 它回填的是**历史**样本的 5/20 日收益（要已到期的），不依赖当日收盘价，
    #   没有理由挤在盘后窗口里。实测 P90 18 分钟，放午休段（午间决策轮之后、
    #   14:40 风控之前）完全不与盘后重活争资源；复盘读它时早已落库。
    ("backfill_screen", dict(hour=13, minute=30), "backfill_screen", "选股候选T+N回填"),

    # ── 尾盘 ───────────────────────────────────────────────
    # ⚠️ 尾盘的硬约束来自**阿深的轮班**（`ai_routine.SLOTS` 最后一班 = 14:40，
    #    由盯盘任务 9-11/13-15 每分钟唤醒后按 slot 触发，不在 JOBS 里——所以
    #    冲突检测器**看不见它**，只能靠这里的人工约束）：
    #      14:40 阿深调整 → 14:45 风控巡检（检查调整后的持仓）→ 14:50 左侧预扫
    # 持仓风控巡检：14:45（原 14:50）。必须晚于阿深 14:40 的调整（否则刚加的仓
    #   还没来得及被风控看到），又要留出收盘前 10 分钟处置。
    ("risk_check", dict(hour=14, minute=45), "risk", "持仓风控巡检"),
    # 左侧监测盘中预扫（2026-11-01）：14:50，收盘前 10 分钟出清单方便次日操作。
    #   用实时快照，不调模型、不推送；实时源不可用时自动回退最近截面并标 degraded。
    ("leftside_intraday", dict(hour=14, minute=50), "leftside_intraday", "左侧监测盘中预扫"),

    # ── 盘后：先做"轻"的 ────────────────────────────────────
    # 市场情绪参照：15:20。唯一硬约束是**必须早于复盘**（复盘按采集日取快照，
    #   晚了就永远读前一天）。1~2 个 HTTP、几秒完成；排它的目的是积累历史基准。
    ("sentiment_ref", dict(hour=15, minute=20), "sentiment", "市场情绪参照采集"),
    # 分析历史 T+N 回填：15:22（原 15:35）。3 分钟的小任务，原位置正好落在
    #   afternoon_full(15:30 起、P90 57 分钟) 的窗口里；提前 8 分钟彻底让开。
    #   它同时承担 T+5/T+20 的**里程碑 LLM 归因**，必须晚于收盘（要当日的 5/20 日收益）。
    ("backfill_analysis", dict(hour=15, minute=22), "backfill", "分析历史T+N回填"),
    # 盘后全流程（选股 + 规则回测）：15:30 **不动** —— 它要当日收盘价，15:30 是
    #   数据可用性决定的边界。**它是全天最长的一环（P90 57.4 分钟 ⇒ 16:28）**，
    #   下游一律排在 16:30 之后（旧表把它们排在 15:50~16:30，等于每天压在它身上）。
    ("afternoon_full", dict(hour=15, minute=30), "rule", "盘后全流程(选股+回测)"),
    # 全市场资金流 + 估值截面快照：15:40 **不动**。约束是**当日日线可用**
    #   （Tushare daily 收盘后约半小时才出），再早就只拿到昨天的数据
    #   （10-29 实测踩过 TushareStaleData）。它与 afternoon_full 有 ~10 分钟重叠，
    #   这是数据可用性决定的**唯一保留重叠**，已登记在 test_schedule_conflicts 基线。
    #   估值截面（PE/市值/换手率）逐日积累，是"六因子权重里 PE 那一项"的前置数据。
    ("snapshot_flow", dict(hour=15, minute=40), "snapshot", "全市场资金流/估值截面快照采集"),

    # ── 盘后账本与报告（全部在 afternoon_full 的 P90 之后）─────
    # AI 联赛收盘账本（2026-11-01 合并）：原 `ai_settle_close`(15:05) + `ai_mark_nav`(15:52)。
    #   两者都是**纯账本、零 LLM**：结算用的是触发时已记录的扫描价（与几点执行无关），
    #   净值必须等到快照（含当日收盘价）落库之后。合并成一个槽位，少一次启动开销。
    # ⚠️ job_id 仍叫 `ai_mark_nav`：STAGE_DEPS / 补跑跳过表 / 产物检查都按 job_id 键控，
    #   改名会让"净值依赖快照"这条依赖静默失效；变的只是 mode（`ai_close`）。
    ("ai_mark_nav", dict(hour=16, minute=30), "ai_close", "AI交易联赛·收盘结算+净值"),
    # 左侧监测收盘正式版：16:33。读日频面板 + 对全池跑右侧状态机
    #   （站回 MA20 记标记 / 60 自然日未修复 ⇒ expired 移出）。5 分钟跑完。
    ("leftside_close", dict(hour=16, minute=33), "leftside_close", "左侧监测收盘扫描"),
    # 每日复盘：16:38（原 15:50）。**这次挪动是纠错**：afternoon_full P90 57.4 分钟
    #   ⇒ 16:28 才结束，而旧排期 15:50 意味着复盘**多数日子读不到当日选股**
    #   （实测每天挂一条"上游未就绪 afternoon_full"）。挪到 P90 之后才有当日样本。
    # 上游：snapshot_flow(15:50) / backfill_screen(13:48) / afternoon_full(16:28)。
    ("daily_review", dict(hour=16, minute=38), "review", "每日复盘生成与推送"),
    # 新闻行业评估：16:42。晚于复盘，且在上游快照之后（推荐排序用当日资金流）。
    # 每天跑（周末快讯也有价值）⇒ mode 不进交易日闸门。
    ("news_review", dict(hour=16, minute=42), "news_review", "新闻行业利好/利空评估"),
    # AI 联赛盘后决策轮：16:47。简报要读新闻评估(16:42)与当日净值(16:30)的产物。
    # ⚠️ **这是 JOBS 里唯一"每跑一次必花钱"的任务**（3 个 LLM persona × 最多 6 轮）。
    #    改排期前先想清楚：随机对照组（3 个）零 LLM 成本，不受影响。
    ("ai_daily_round", dict(hour=16, minute=47), "ai_round", "AI交易联赛·盘后决策轮"),
    # 交易拥挤度截面：16:55。输入是两个**当日**产物（快照的换手率/成交额 + 面板收盘价），
    #   必须是当日最后一环；产物供**次日 08:15 盘前选股**做减法闸门
    #   （拥挤度前 10% 且 20 日涨幅前 20% ⇒ 不新开仓，11 年 11/11 年为负）。
    ("crowd_snapshot", dict(hour=16, minute=55), "crowd", "交易拥挤度截面"),
    # 超额衰减监控：**周五 17:05**（2026-10-31 新增，2026-11-01 改周频）。
    #   它读 screen_picks 的 ret_20d/bench_ret_20d，而 T+20 回填至今 0/964 可判
    #   ⇒ 日频只是每天记一条"样本不足"。等样本齐了再改回日频（删掉 day_of_week）。
    #   ⚠️ 周频靠 `day_of_week` 显式覆盖：`_builtin_job_spec` 与补跑判据都认它。
    ("alpha_decay", dict(hour=17, minute=5, day_of_week="fri"),
     "alpha_decay", "超额衰减监控(周频)"),
    # 换手率/流通股本日更：**17:15**（2026-11-01 新增）。
    # 补的是一个"会自己变坏"的缺口：`data/panel/turn_share/` 是 10-29 一次性回填的，
    #   当时只给了手工脚本、**没有日更任务** ⇒ 数据停在 2026-10-28。
    #   下游 `factor_analysis` 取 `turnover_pct` 用 `ffill(limit=3)` 兜尾部，所以
    #   缺 2 天还能撑，**第 4 个交易日（11-03）起 amihud20 等量价因子开始出 NaN**。
    # 为什么 17:15：数据源是 `market_snapshot`（15:40 落库），只需读库 + 写 parquet，
    #   实测 1~3 分钟。排在 crowd_snapshot(16:55，4 分钟) 之后、alpha_decay(17:05)
    #   之前都行，取 17:15 是为了避开两者，且离 18:10 的 evening_analysis 仍有余量。
    #   ⚠️ 第一版是打新浪接口，实测全量约 4 小时且会让进程 V8 abort ⇒ 已改快照派生。
    ("turn_share_sync", dict(hour=17, minute=15), "turn_share_sync", "换手率/流通股本日更"),

    # ── 晚间 / 高频 ────────────────────────────────────────
    # 盘后批量深度分析：18:10。两个理由：① DeepSeek 峰谷定价（9-12、14-18 高峰，
    #   其余半价）⇒ 18:10 省 50%；② 避开 15:30~17:00 的盘后重活，不让单进程排队。
    # 它**不重新选股**，只对当日已有候选跑分析（见 core/evening_analysis）。
    ("evening_analysis", dict(hour=18, minute=10), "llm_analysis", "盘后深度分析(TOP8)"),
    ("hourly_news", dict(minute=5), "news", "新闻快讯采集(每小时)"),
    # Phase 1：盘中异动预警。cron 覆盖 9-15 点每分钟，真正的启停由 core.watcher 判定
    # （交易日 + 09:30-11:30 / 13:00-15:00），此处只负责"叫醒"。
    ("intraday_watch", dict(hour="9-11,13-15", minute="*"), "watch", "盘中异动预警"),
]

# 期望的任务 id 清单。健康检查用它跟心跳里的 job_ids 比对：
# 新增任务后如果忘了重启调度进程，进程跑的仍是旧代码（少一个或多个任务），
# 光看"进程活着"完全看不出来——这正是批次 22 结束时踩到的事故形态。
JOB_IDS = [job_id for job_id, _cron, _mode, _desc in JOBS]


def _trading_day_modes() -> set:
    """依赖当日行情、非交易日必须跳过的 mode 集合。

    单一事实来源在 core/scheduled_jobs.py::TRADING_DAY_MODES（它同时驱动自定义
    任务的开关约束）。测试 `test_trading_day_modes_cover_all_builtin_modes`
    会守住"内置任务用到的 mode 都在这个集合里"，避免两边漂移。
    """
    try:
        from core.scheduled_jobs import TRADING_DAY_MODES
        return set(TRADING_DAY_MODES)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("读取交易日任务清单失败，退化为内置名单: %s", e)
        return {"screen", "rule", "review", "backfill", "backfill_screen",
                "snapshot", "risk"}


def run_leftside_job(job_desc: str = "左侧监测扫描", intraday: bool = False):
    """左侧监测：无明显利空的大跌 + 基本面资质 → 关注池。

    两个时点（用户要求"收盘前出清单、方便第二天操作"）：
    - 14:45 **盘中预扫**（intraday=True）：用实时快照的当日跌幅（已接近收盘价），
      收盘后立刻能看到清单、当晚研究基本面、次日开盘执行；
    - 15:45 **收盘正式版**：用日频面板收盘价，落库口径以此为准，
      并对池内全部条目做右侧状态机（站回 MA20 标记 / 60 天未修复判证伪移出）。

    失败只记日志不抛出：监测是附加功能，不能把调度器带崩。
    """
    try:
        from core import leftside_monitor as LM
        from data.trading_calendar import last_trading_day
        today = last_trading_day()
        if intraday:
            items = LM.scan_intraday()
            r = LM.persist_pool(items, today)
            _logger.info("[左侧监测/盘中] %s | 预清单 %d 条 | %s", job_desc, len(items), r)
        else:
            items = LM.scan()
            r = LM.persist_pool(items, today)
            u = LM.update_right_side()
            _logger.info("[左侧监测/收盘] %s | 新增/刷新 %d 条 | 状态机 %s", job_desc, len(items), u)
    except Exception as e:                              # noqa: BLE001
        _logger.warning("[左侧监测] 扫描失败（不影响其他任务）: %s", e)


def run_sentiment_job(job_desc: str = "市场情绪参照采集"):
    """盘后采集外部公开言论 → 规则层聚合 → 落库（**不调模型、不推送**）。

    ## 为什么必须有这个定时任务，而不是"用户开页面时顺带跑"

    `compare_baseline` 要求历史 ≥5 个交易日，否则**不给分位**（不给假数字）。
    靠手动开页面采集，两天打鱼三天晒网 ⇒ 永远凑不到 5 天 ⇒ 相对基准这一层
    形同不存在。所以定时任务的唯一目的就是**积累历史**，不是给谁看。

    ## 为什么不让它调模型

    定性判定要花钱，同一天判一次就够。让定时任务调模型会变成"每天固定烧钱
    去生成一段没人看的文字"。判定留在页面上按需触发（用户想看点一下）。

    ## 失败处理

    知乎 `pins` 是非公开接口，随时可能失效。失败只记 WARNING、**不推送** ——
    否则一个第三方接口的抖动会变成每天一条告警，与飞书告警那次"88% 噪声"
    是同一个错误。但**必须进日志**，否则"历史一直不涨"会被误判成"本来就没数据"。
    """
    start = datetime.now()
    _logger.info("═══ [%s] %s 开始 ═══", job_desc, start.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        from data import zhihu_opinions as Z
        from core import sentiment_reference as SR

        members = Z.load_members()
        if not members:
            _logger.info("═══ [%s] 跳过：未配置博主（config/zhihu_members.json 为空）═══",
                         job_desc)
            return
        items, meta = Z.collect_members(members, per_member=20, sleep=0.6)
        if not items:
            # ⚠️ 不能当成"今天情绪平静"：要写清是哪一步失败。
            le = meta.get("last_error") or {}
            _logger.warning("═══ [%s] 未取到任何言论：%s —— %s（errors=%s）═══",
                            job_desc, le.get("step") or "未知步骤",
                            le.get("reason") or "无结果",
                            (meta.get("errors") or [])[:2])
            return
        res = SR.build_for_source(items, source="zhihu:member")
        snap = res["snapshot"]
        _logger.info("═══ [%s] 完成：%d 条 → 判出方向 %d 条（多 %d / 空 %d），"
                     "多空比 %s，质量 %s ═══",
                     job_desc, snap["n_total"], snap["n_scored"], snap["bull"],
                     snap["bear"],
                     "—" if snap["bull_ratio"] is None else f"{snap['bull_ratio']:.0%}",
                     snap["quality"])
    except Exception as e:                      # noqa: BLE001
        _logger.exception("═══ [%s] 任务异常: %s ═══", job_desc, e)


def run_news_job(job_desc: str = "新闻采集"):
    """执行新闻快讯采集（含保留策略清理）"""
    try:
        from data.news_feed import fetch_and_store
        from db import store as db_store
        result = fetch_and_store()
        db_store.cleanup()
        _logger.info("═══ [%s] 完成: %s ═══", job_desc, result)
        # 情绪面选股（P1-5，2026-10-23）：把新入库的快讯做"新闻→个股"严匹配。
        # 增量处理（按 news_items.id），所以每小时只算新增的几十条，开销可忽略。
        # ⚠️ 失败不得影响新闻采集本身 —— 它是**附加产物**，不是主链路。
        try:
            from core import news_stock_hits
            r2 = news_stock_hits.refresh(limit=500)
            if r2.get("skipped_reason"):
                _logger.warning("═══ [%s] 情绪面匹配跳过: %s ═══",
                                job_desc, r2["skipped_reason"])
            elif r2.get("processed"):
                _logger.info("═══ [%s] 情绪面匹配: %d 条 → 命中 %d 只 ═══",
                             job_desc, r2["processed"], r2.get("stocks"))
        except Exception as e:                              # noqa: BLE001
            _logger.warning("═══ [%s] 情绪面匹配失败（不影响采集）: %s ═══", job_desc, e)
    except Exception as e:
        _logger.exception("═══ [%s] 任务异常: %s ═══", job_desc, e)


def run_news_review_job(job_desc: str = "新闻行业评估"):
    """行业利好/利空评估（热点新闻工作流，2026-10-09）。

    放在 15:55：晚于快照(15:40)与复盘(15:45)，评估"全天快讯 + 当日资金流"
    （推荐排序要用 fund_flow_snapshot 的当日数据）。先增量补采一次新闻——
    上一次 hourly_news 是 14:05（每小时:05），15:55 时库内已有约 2 小时缺口。
    LLM 失败**不落库**（core.news_industry_review 里已处理），这里只把结果
    原样打日志——status 必须可见，不能让"评估失败"伪装成"没有波动"。
    """
    try:
        from data.news_feed import fetch_and_store
        from core import news_industry_review as nir
        fetch_and_store()                       # 补上 14:05 以来的缺口
        result = nir.run_industry_review(hours=24)
        _logger.info("═══ [%s] 完成: status=%s, saved=%s, warnings=%s %s ═══",
                     job_desc, result.get("status"), result.get("saved"),
                     result.get("warnings"),
                     result.get("error") or "")
    except Exception as e:
        _logger.exception("═══ [%s] 任务异常: %s ═══", job_desc, e)


def _snapshot_quote_date() -> str:
    """本次采集的**数据所属交易日**（不是采集时刻）。

    盘后采集时两者同一天；但补采/延迟启动时会不同，而落库字段 `snapshot_date`
    是"数据属于哪一天"、`captured_at` 是"什么时候采的"——混用会让 IC 计算
    把不同日期的截面当成同一天（前视）。所以这里显式取"应有的最新交易日"，
    取不到再退化为今天。
    """
    try:
        from data.realtime import _expected_quote_date
        d = _expected_quote_date()
        if d:
            return str(d)
    except Exception as e:                        # noqa: BLE001
        _logger.warning("[估值快照] 交易日推算失败(%s)，退化为当天", e)
    return date.today().strftime("%Y-%m-%d")


def is_a_share_code(code: str) -> bool:
    """6 位代码是否属于 A 股（沪深主板/创业板/科创板/北交所）。

    为什么不用"是否在 A 股清单里"判断：清单来自 akshare 且**可能拿不到**
    （`market_codes()` 有 None 分支）——拿不到时若按"不在清单就丢"会把整表丢掉。
    这里只按代码段做**结构性**判定，不依赖任何网络/缓存，永远可用。

    ⚠️ 实现已**下沉**到 `data/codes.py`（2026-10-30）：`data/` 层也要用这条规则
    （财报公告日按 A 股过滤），而分层约束禁止 data → core。这里只做委托，
    保证全项目仍**只有一处**实现（历史上这段规则在 data 里被复制过两次）。
    """
    from data.codes import is_a_share_code as _impl
    return bool(_impl(code))


def save_daily_market_snapshot(snapshot_date: str = None) -> dict:
    """采集并落库当日全市场估值截面（PE / PB / 市值 / 换手率）。

    为什么必须**逐日**落库，而不是"用当期快照顶替"：
      IC 的定义是「某日 T 的截面因子排名 vs T 之后的收益排名」，它要求能重建
      "T 那一天的全市场截面"。K 线类因子天然满足（任何历史交易日的日线都能重算），
      而 PE / 市值 / 换手率的**绝对水平**只存在于当时的快照里 —— 拿今天的 PE 去算
      去年的 IC 就是**前视**，数字毫无意义。
      ⇒ 这正是"六因子里 PE 那一项无法验证"的根因（见 `core/factor_analysis.py`）。
      本函数就是那个前置条件：每天攒一份，攒够几十个交易日即可算 IC。

    ⚠️ 单位转换**只在这里做一次**：`data.sina_source` 的 DataFrame 里
      `amount / market_cap / float_cap` 是**亿元**，落库统一为**元**
      （与 `fund_flow_snapshot` 对齐，避免跨表比大小时踩"同名列不同量级"）。
      `turnover / change_pct` 已是**百分数**，不做换算。

    Returns:
        {"ok", "snapshot_date", "rows", "coverage", "caveat"} 或 {"ok": False, "reason"}
    """
    from data import sina_source
    from db import store as db_store

    d = snapshot_date or _snapshot_quote_date()

    # ── A+ 档（2026-10-26）：全量域 —— 新浪 `hs_a` 全市场 + tushare `stock_basic` 行业 ──
    # 为什么换掉下面那条 tushare 路径：它依赖 `daily_basic`，而 120 积分档**无此权限**
    # （能力表实测「被拒 3 次、成功 0 次」）⇒ 每天抛异常回退新浪 49 行业明细 ⇒
    # 覆盖率锁在 **47%**（创业板 31% / 科创板 5%），300750、688111 长期不在域内。
    # 全量域实测 **99.9%（5,572/5,576）**，PE/PB/市值/换手率全有，耗时 1.5s + 1.7s。
    try:
        from data import full_market as _fm
        r = _fm.save_full_snapshot(d)
        if r.get("ok"):
            if r.get("degraded"):
                _logger.warning("═══ [估值快照] 全量域完成但**降级**: 数据日期 %s，"
                                "%s 只（覆盖 %s）｜ %s ═══",
                                d, r.get("rows"), r.get("coverage"),
                                "；".join(r.get("degraded") or []))
            else:
                _logger.info("═══ [估值快照] 全量域完成: 数据日期 %s，%s 只"
                             "（覆盖 %s，行业 %s 只，stock_basic=%s）═══",
                             d, r.get("rows"), r.get("coverage"),
                             r.get("industry_covered"), r.get("stock_basic_source"))
            return {"ok": True, "snapshot_date": d, "rows": r.get("rows"),
                    "coverage": r.get("coverage"),
                    "total_market": r.get("universe"),
                    "degraded": r.get("degraded") or [],
                    "caveat": f"来源 sina_full；{r.get('unit')}"}
        _logger.warning("═══ [估值快照] 全量域未落库: %s ═══", r.get("reason"))
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("═══ [估值快照] 全量域不可用（%s: %s）⇒ 回退 tushare/新浪 ═══",
                        type(e).__name__, str(e)[:160])

    # ── A 档（2026-10-17）：tushare 全市场截面（**需要 daily_basic 权限**）──
    # 覆盖率 54.8% → ≥99%（含真实行业 stock_basic.industry、成交额 daily.amount）。
    # ⚠️ 任何失败（限频/当日截面未生成/网络）都**回退新浪**并留 WARNING —— 降级必须可查，
    #    且**当日截面未生成**（数据日期 ≠ 期望日）绝不能拿昨日数据顶替（前视）。
    try:
        from data import tushare_source as _ts
        ts_rows, ts_attrs = _ts.build_market_snapshot_rows(d)
        ts_written = db_store.save_market_snapshot(d, ts_rows, source="tushare")
        try:
            db_store.save_data_coverage(
                d, "market_snapshot", got=int(ts_written or 0),
                total=int(ts_attrs.get("total_market") or 0),
                status=("degraded" if ts_attrs.get("degraded") else "ok"),
                source="tushare",
                note=str(ts_attrs.get("caveat") or ""))
        except Exception as _e:                              # noqa: BLE001
            _logger.warning("[估值快照] 覆盖率落库失败（不影响快照）: %s: %s",
                            type(_e).__name__, _e)
        _logger.info("═══ [估值快照] tushare 完成: 数据日期 %s，%s 只（来源 tushare）═══",
                     d, ts_written)
        return {"ok": bool(ts_written), "snapshot_date": d, "rows": ts_written,
                "coverage": ts_attrs.get("coverage"),
                "total_market": ts_attrs.get("total_market"),
                "degraded": ts_attrs.get("degraded") or [],
                "caveat": ts_attrs.get("caveat")}
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[估值快照] tushare 不可用（%s: %s）⇒ 回退新浪路径",
                        type(e).__name__, str(e)[:120])

    df = sina_source.get_sector_snapshot()
    if df is None or df.empty:
        return {"ok": False, "reason": "新浪全市场快照不可用（不写入残缺数据）",
                "snapshot_date": d, "rows": 0}

    yi = 1e8                                       # 亿元 → 元
    rows = []
    skipped_non_a = []
    for r in df.to_dict("records"):
        code = str(r.get("code") or "").strip()
        if not code:
            continue
        if not is_a_share_code(code):
            # ⚠️ 2026-10-23 修：新浪行业明细**含 B 股**（沪B 900xxx / 深B 200xxx），
            #    tushare 主路径不会——所以只有回退新浪时才会混进来。实测
            #    2026-10-22 落库 2991 只里有 52 只沪B + 37 只深B（共 89 只）。
            #    后果不是"多几行"：这些票会经 `local_screen` 一路进候选池，
            #    白跑取数（每只 30~120s，且恒失败），并挤占 STRATEGY_TOP_N 名额。
            #    这里**在落库口**拦掉，而不是在选股口拦——落库口是唯一的公共入口，
            #    拦住一次，所有下游消费者（选股 / 体检 / 分析）都干净。
            skipped_non_a.append(code)
            continue
        rows.append({
            "stock_code": code, "stock_name": r.get("name") or "",
            "industry": r.get("industry") or "", "board": r.get("board") or "",
            "price": r.get("price"), "change_pct": r.get("change_pct"),
            "pe": r.get("pe"), "pb": r.get("pb"),
            "market_cap": (None if r.get("market_cap") is None
                           else r["market_cap"] * yi),
            "float_cap": (None if r.get("float_cap") is None
                          else r["float_cap"] * yi),
            "turnover": r.get("turnover"),
            "amount": (None if r.get("amount") is None else r["amount"] * yi),
        })
    written = db_store.save_market_snapshot(d, rows, source="sina_sector")
    _cov = df.attrs.get("coverage")
    _total = df.attrs.get("total_market")
    _degraded = list(df.attrs.get("degraded") or [])
    if skipped_non_a:
        # 如实记账：这是"数据比应有口径多出来的部分"，必须看得见。
        # 静默丢弃会让覆盖率分母对不上却查不出原因。
        _logger.info("[估值快照] 剔除非 A 股 %d 只（B股等）: %s%s",
                     len(skipped_non_a), ",".join(skipped_non_a[:5]),
                     " …" if len(skipped_non_a) > 5 else "")
    # ── 覆盖率落库（P2-7，2026-10-17）──
    # ⚠️ 光落 `market_snapshot` 只能算出"落了 2991 只"，**看不出它只覆盖了 54.8%**。
    #    这里把「实取 / 全市场 / 覆盖率 / 降级原因」逐日留一条，健康页就能看出趋势与断崖。
    # ⚠️ 落库失败**不得**连累快照本身（快照已经写成功了）。
    try:
        db_store.save_data_coverage(
            d, "market_snapshot", got=int(written or 0), total=_total,
            status=("degraded" if _degraded else
                    ("partial" if (_cov is not None and _cov < 0.9) else "ok")),
            source="sina_sector",
            note=("；".join(_degraded) if _degraded else str(df.attrs.get("caveat") or "")))
    except Exception as _e:                                  # noqa: BLE001
        _logger.warning("[估值快照] 覆盖率落库失败（不影响快照）: %s: %s",
                        type(_e).__name__, _e)
    # 「降级随数据走」：把来源层的 degraded 一并带出（页面/日志/健康检查都能看见）
    return {"ok": bool(written), "snapshot_date": d, "rows": written,
            "coverage": _cov, "total_market": _total,
            "degraded": _degraded,
            "caveat": df.attrs.get("caveat")}


def run_snapshot_job(job_desc: str = "全市场资金流/估值快照采集"):
    """盘后一次性采集全市场资金流 + 行业资金流 + 估值截面并落库（选股只读库）。

    为什么单独一个任务、而不放进选股里（2026-09-25 实测）：
      同花顺快照一个窗口就要拉**全市场 5210 行**，4 个窗口合计 **75.9 秒**，
      而选股一轮 4~5 分钟、内存缓存只有 300 秒 → 中途必然过期重拉，
      一次选股光在这上面就白花约 150 秒。放到盘后（15:40）采集一次即可全天复用。
    放在 15:40 是有意的：晚于全流程（15:30）、早于复盘（15:45），
    这样复盘与次日盘前选股都能读到当天的快照。
    """
    start = datetime.now()
    _logger.info("═══ [%s] %s 开始 ═══", job_desc, start.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        from data.fund_flow import capture_market_flow_snapshot
        result = capture_market_flow_snapshot()
        # ⚠️ 按**三态**记日志（2026-10-17 修）：原来无论采到几行都记 INFO"完成"，
        #    实测"个股 0 只 / 行业 0 个"被读成成功（失败原因还在被吞的 print 里）。
        _st = result.get("status") or "ok"
        # ── 覆盖率落库（P2-7）：资金流**没有"全市场应有条数"** ⇒ total=None，
        #    ratio 记 NULL（不是 0）。这条记录的价值在 "status + 失败原因"。
        try:
            from db import store as _db_store
            _f = result.get("failures") or {}
            _dt = str(result.get("snapshot_date") or "").strip()
            if _dt:                                          # 没有数据日期就不落（不编日期）
                _db_store.save_data_coverage(
                    _dt, "fund_flow_stock",
                    got=int(result.get("stocks") or 0), total=None, status=_st,
                    source="ths",
                    note=("；".join(f"{k}: {v}" for k, v in list(_f.items())[:3])
                          if _f else ""))
        except Exception as _e:                              # noqa: BLE001
            _logger.warning("[资金流] 覆盖率落库失败（不影响采集）: %s: %s",
                            type(_e).__name__, _e)
        if _st == "ok":
            _logger.info("═══ [%s] 资金流完成: 数据日期 %s，个股 %s 只 / 行业 %s 个 ═══",
                         job_desc, result.get("snapshot_date"),
                         result.get("stocks"), result.get("industries"))
        else:
            _logger.warning("═══ [%s] 资金流**%s**: 数据日期 %s，个股 %s 只 / 行业 %s 个；"
                            "窗口明细 %s；失败原因 %s ═══",
                            job_desc, "部分成功" if _st == "partial" else "失败",
                            result.get("snapshot_date"), result.get("stocks"),
                            result.get("industries"), result.get("windows"),
                            result.get("failures") or "—")
    except Exception as e:
        _logger.exception("═══ [%s] 资金流任务异常: %s ═══", job_desc, e)

    # 估值截面单独 try：它与资金流是两个上游（新浪行业明细 vs 同花顺资金流），
    # 一方不稳不该把另一方也带走 —— 否则"资金流采到了但估值没采"会表现成
    # 整个任务失败，而实际数据已经落库一半。
    try:
        m = save_daily_market_snapshot()
        if m.get("ok"):
            # ⚠️ 有降级就按 WARNING 记，并带上 caveat（口径随数据走）：
            #    "覆盖 ?" 不等于成功 —— 那通常意味着代码清单不可用、本次没过滤退市标的。
            if m.get("degraded"):
                _logger.warning("═══ [%s] 估值截面完成但**降级**: 数据日期 %s，%s 只"
                                "（覆盖 %s，降级 %s）｜ %s ═══",
                                job_desc, m.get("snapshot_date"), m.get("rows"),
                                m.get("coverage"), m.get("degraded"), m.get("caveat"))
            else:
                _logger.info("═══ [%s] 估值截面完成: 数据日期 %s，%s 只（覆盖 %s）═══",
                             job_desc, m.get("snapshot_date"), m.get("rows"),
                             m.get("coverage"))
        else:
            _logger.warning("═══ [%s] 估值截面未落库: %s ═══", job_desc, m.get("reason"))
    except Exception as e:
        _logger.exception("═══ [%s] 估值截面任务异常: %s ═══", job_desc, e)

    # 指数日线增量回填（B2②，2026-10-19）：5 个指数 = 5 次请求，市场检测的
    # 6 个指数类指标全靠它本地化（离线可用）。失败不影响主采集。
    try:
        from data.index_bars import backfill_index_daily
        r = backfill_index_daily()
        _logger.info("═══ [%s] 指数日线回填: %d 成功 / %d 失败 ═══",
                     job_desc, len(r.get("done") or {}), len(r.get("failed") or {}))
        for sym, err in (r.get("failed") or {}).items():
            _logger.warning("═══ [%s] 指数 %s 回填失败: %s ═══", job_desc, sym, err)
        # ⚠️ 写盘成功但数据停在昨日 ⇒ 市场检测会整日落后一天，必须显式告警
        stale = r.get("stale") or {}
        if stale:
            _logger.warning("═══ [%s] 指数日线**数据陈旧**（未含当日）：%s ═══",
                            job_desc, stale)
    except Exception as e:
        _logger.exception("═══ [%s] 指数日线回填异常（不影响主采集）: %s ═══", job_desc, e)

    # 面板日增量（2026-10-26）：把当日全市场日线写进覆盖层。
    # 为什么必须每日跑：`data/panel/by_code` 是 10-18 一次性建的、之后**无人维护**，
    # 实测 10-26 时 `date_max` 还停在 10-16 ⇒ 交易台持仓估值、MA60、因子截面
    # 全部滞后 7 个交易日。写覆盖层是 O(1)（一个 ≈300KB 的小文件），不是重写 5,830 个文件。
    try:
        _refresh_panel_daily(job_desc)
    except Exception as e:
        _logger.exception("═══ [%s] 面板日增量异常（不影响主采集）: %s ═══", job_desc, e)


def _refresh_panel_daily(job_desc: str) -> dict:
    """面板日增量：把缺失的交易日写进 `daily_overlay`（当日 + 停机期间漏掉的）。

    两件事：
    1. **日历自愈** —— 本地 `trade_cal_cache` 只覆盖到上次拉取日（实测 2026-10-16，
       缺 7 个交易日），而 `bars.trading_day_lag` 全靠它 ⇒ 滞后判定一律返回 None
       （"未知"不等于"新鲜"，新鲜度闸门等于没装）。发现缓存末日晚于今天时扩一次；
       `trade_cal` 是 **1 次/小时** 限频，一年也就跑几次。
    2. **补缺口** —— 从"面板最大日"补到"今天"，上限 20 个交易日；
       再久就该走全量回填，不能把日常任务变成几小时的作业。
    """
    from data import bars as _bars, panel_daily

    today = _snapshot_quote_date()
    tds = _bars.trade_days()
    if not tds or tds[-1] < today:
        try:
            from data import tushare_source as _ts
            start = tds[0] if tds else f"{today[:4]}-01-01"
            end = f"{int(today[:4]) + 1}-12-31"
            got = _ts.fetch_trade_cal(start, end)
            _bars.clear_cache()
            tds = _bars.trade_days()
            _logger.info("═══ [%s] 交易日历扩到 %s（共 %s 天）═══",
                         job_desc, tds[-1] if tds else "?", len(got))
        except Exception as e:                            # noqa: BLE001
            _logger.warning("═══ [%s] 交易日历扩展失败（滞后判定会退化为未知）: %s ═══",
                            job_desc, e)

    mx = _bars.panel_date_max()
    missing = [x for x in tds if mx and mx < x <= today][:20]
    if not missing:
        missing = [today]

    done, failed, last = [], {}, None
    for d in missing:
        r = panel_daily.append_trade_day(d)
        if r.get("ok"):
            done.append(f"{d}({r.get('codes')})")
            last = r
        else:
            failed[d] = r.get("reason")
    if failed:
        _logger.warning("═══ [%s] 面板覆盖层部分失败: %s（成功 %s）═══", job_desc, failed, done)
    else:
        _logger.info("═══ [%s] 面板覆盖层完成: %s，面板最新日 %s ═══",
                     job_desc, done, (last or {}).get("date_max") or mx)
    return {"ok": not failed, "done": done, "failed": failed}


def run_review_job(job_desc: str = "每日复盘"):
    """执行复盘 Agent：生成报告并按 .env 配置推送 webhook"""
    start = datetime.now()
    _logger.info("═══ [%s] %s 开始 ═══", job_desc, start.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        from core.review_agent import generate_review
        result = generate_review(push=True)
        if "error" in result:
            _logger.error("═══ [%s] 失败: %s ═══", job_desc, result["error"])
            return
        _logger.info("═══ [%s] 完成: %s | 推送: %s ═══", job_desc,
                     result.get("report_path"), result.get("push"))
    except Exception as e:
        _logger.exception("═══ [%s] 任务异常: %s ═══", job_desc, e)


def run_watch_job(job_desc: str = "盘中异动预警"):
    """执行一轮盘中异动检查（交易时段/交易日判定在 watcher 内部）"""
    try:
        from core.watcher import get_watcher
        result = get_watcher().scan_once()
        if result.get("skipped"):
            _logger.debug("═══ [%s] 跳过：%s ═══", job_desc, result.get("reason"))
        else:
            # ⚠️ 日志级别按**是否真有事件**分流（2026-10-17）：
            #    冷却期 30 分钟 + 每分钟扫描 ⇒ 绝大多数轮次是"命中 N 条、提醒 0 条"，
            #    实测**每天 183 行**这种噪音，把真正的"提醒 15 条"埋掉了。
            #    ⚠️ 注意：`提醒 0 条`**本身是正常结果**（不是 bug），只是不值得记 INFO。
            # ⚠️ 2026-10-23：**必须带上推送结果**。此前 `result["push"]` 被直接丢掉
            #    ⇒ 日志里查不到盘中到底推了几条、成功还是失败（观测盲区）。
            _fired = int(result.get("fired") or 0)
            _saved = int(result.get("saved") or 0)
            _push = result.get("push") or {}
            _lvl = _logger.info if (_fired or _saved) else _logger.debug
            if _push:
                _lvl("═══ [%s] 检查 %s 只，命中 %s 条，提醒 %s 条，落库 %s 条，推送=%s ═══",
                     job_desc, result.get("checked"), result.get("alerts"), _fired, _saved,
                     _push)
            else:
                _lvl("═══ [%s] 检查 %s 只，命中 %s 条，提醒 %s 条，落库 %s 条 ═══",
                     job_desc, result.get("checked"), result.get("alerts"), _fired, _saved)
    except Exception as e:
        _logger.exception("═══ [%s] 任务异常: %s ═══", job_desc, e)


def run_backfill_job(job_desc: str = "分析历史T+N回填"):
    """回填分析历史的 T+5/T+20 收益（零 LLM 成本，只拉 K 线）

    顺带做**每日目标/止损核对**（`check_triggers_daily`）：原先 hit_target/hit_stop
    只在 T+N 回填时判一次（最快 T+5），"今天刚跌破止损"要等一周才说 —— 而那正是
    "明日操作建议"最需要的信息。核对与回填共用 `evaluate_triggers` 唯一实现，
    且**不新建调度任务**（避免 JOBS 条数断言又要全仓扫一遍）。

    ⚠️ 2026-10-20 审查 B3：本函数原先**不写 runs**，而复盘失败只 append 到
    `stats["review_errors"]` + 一条 warning ⇒ 页面与健康检查都看不到。
    当天就出过"回填成功、11 条复盘全废"（新旧模块混用 `upsert_analysis_review`
    不存在），因为被埋掉而两小时无人察觉。现在：
      ① 每次都写 `runs`（与 `backfill_screen` 同一原则：跑了但 0 条 ≠ 没跑）；
      ② `review_errors` 非空 ⇒ `ok=False`，页 7「运行健康」直接可见。
    """
    stats, trig, err = None, None, None
    try:
        from core.analysis_history import backfill_analyses, check_triggers_daily
        stats = backfill_analyses(limit=100)
        _logger.info("═══ [%s] 完成: %s ═══", job_desc, stats)
        trig = check_triggers_daily(limit=200)
        _logger.info("═══ [%s] 每日触发核对: %s ═══", job_desc, trig)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        _logger.exception("═══ [%s] 任务异常: %s ═══", job_desc, e)
    finally:
        review_errs = list((stats or {}).get("review_errors") or [])
        if review_errs:
            # 升级成 ERROR：一次 warning 埋不住"N 条复盘全废"
            _logger.error("═══ [%s] ⚠️ %d 条复盘失败（回填本身可能成功）：%s ═══",
                          job_desc, len(review_errs), review_errs[:3])
        try:
            from db import store as _store
            _store.insert_run("job:backfill_analysis", {
                "desc": job_desc,
                "ok": err is None and not review_errs,
                "error": err,
                # 只留统计与计数，别把整份 stats（可能含明细）塞进 runs
                "stats": {k: v for k, v in (stats or {}).items()
                          if k != "review_errors"} or None,
                "triggers": trig,
                "review_errors": len(review_errs),
                "review_error_sample": review_errs[:3],
            })
        except Exception as e2:                 # noqa: BLE001
            _logger.warning("═══ [%s] 运行记录写入失败（不影响回填结果）：%s ═══",
                            job_desc, e2)


def run_evening_analysis_job(job_desc: str = "盘后深度分析"):
    """对当日候选跑批量深度分析（`mode="llm_analysis"`）。

    ⚠️ 加锁：AI 联赛/AI 助手也在花 token，同时跑会让"当日 token 上限"判断失真；
       且同一只票并发分析会撞 `(analysis_date, stock_code)` 唯一索引（UPSERT 会互相覆盖）。
    """
    from db import store as _store
    lock_name = "evening_analysis"
    # ⚠️⚠️ 2026-10-20 事故：这里原本写成
    #       holder = f"scheduler:{os.getpid()}"      # 而模块顶部没有 import os
    #     且**在 try 之外** ⇒ 每次调度必抛 NameError，连 except 都进不去，
    #     只在 APScheduler 的 executor 里留一条异常。而它自 10-19 接线起
    #     从未被执行过（18:10 时进程都没在跑），所以没有任何日志暴露它。
    #     两个修正：① 顶部补 import os；② 把可能抛错的东西全部圈进 try，
    #     让"哪怕还有别的未定义名字"也至少变成一条可查的 exception。
    try:
        holder = f"scheduler:{os.getpid()}"
        if not _store.acquire_task_lock(lock_name, holder, ttl_seconds=3600):
            _logger.warning("═══ [%s] 已在运行（锁被占用），本次跳过 ═══", job_desc)
            return
        try:
            from core.evening_analysis import run_evening_analysis
            stats = run_evening_analysis()
            _logger.info("═══ [%s] 结果: 成功 %s / 失败 %s / 跳过原因 %s ═══",
                         job_desc, stats.get("analyzed"), stats.get("failed"),
                         stats.get("skipped_reason"))
        finally:
            _store.release_task_lock(lock_name, holder)
    except Exception as e:
        _logger.exception("═══ [%s] 任务异常: %s ═══", job_desc, e)


def run_backfill_screen_job(job_desc: str = "选股候选T+N回填"):
    """回填选股候选的 T+5/T+20 收益与同期指数基准（零 LLM 成本，只拉 K 线）

    ⚠️ R5（2026-10-18 审查）：**必须留执行记录**。产物判据是"今日回填 N 条"，
    但"今天没有到期样本"是正常情况（周末/节后/样本还没走完 T+5）——
    没有执行记录时，"跑了但 0 条"和"根本没跑"无法区分，复盘会天天报假缺失。
    """
    stats = None
    err = None
    try:
        from core.screen_history import backfill_screen_picks
        stats = backfill_screen_picks(limit=200)
        _logger.info("═══ [%s] 完成: %s ═══", job_desc, stats)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        _logger.exception("═══ [%s] 任务异常: %s ═══", job_desc, e)
    finally:
        try:
            from db import store as _store
            s = stats if isinstance(stats, dict) else {}
            checked = int(s.get("checked") or 0)
            updated = int(s.get("updated") or 0)
            _store.insert_run("job:backfill_screen", {
                "desc": job_desc, "ok": err is None, "error": err,
                "checked": checked, "updated": updated,
                "note": (err or (f"检查 {checked} 条，更新 {updated} 条" if checked
                                 else "当日无到期样本")),
            })
        except Exception as e:                          # noqa: BLE001
            _logger.debug("[%s] 执行记录写入失败（不影响任务）: %s", job_desc, e)


def run_risk_job(job_desc: str = "持仓风控巡检"):
    """持仓风控巡检：止损/止盈触发 + 追踪止损建议 + 组合结构（集中度/流动性）。

    与 `intraday_watch` 的分工：watch 盯的是**全市场**异动（涨跌幅、放量、涨停等），
    本任务只盯**用户自己的持仓**有没有触及已设定的风控位。

    刻意不做的事：**不自动改止损价**。`check_trailing_stops()` 只给建议，
    改库要用户在持仓页点「应用」。自动改动用户的持仓数据风险太高。
    集中度/流动性同理：**只报告，不阻断、不自动减仓**。
    """
    start = datetime.now()
    _logger.info("═══ [%s] %s 开始 ═══", job_desc, start.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        from core.portfolio import get_portfolio_manager
        pm = get_portfolio_manager()
        # ⚠️ 用 `active_codes()`（纯 DB、不取价）而不是 `get_active_positions()`：
        #    后者会走 `load()` → 逐只联网取价，而我们正需要"先拿代码、再批量取价"。
        codes = pm.active_codes()
        if not codes:
            _logger.info("═══ [%s] 无活跃持仓，跳过（非异常）═══", job_desc)
            return

        # ⚠️ 行情**只批量取一次**再传进去（2026-10-15）：
        #    `check_stop_loss` / `check_trailing_stops` / `get_exposure` 内部各自会逐只取价，
        #    而 `_get_current_price()` 走 akshare 东财接口，取数受限时要重试 3 次 + 指数退避
        #    （实测单只 ≈30s）⇒ 持仓多时本任务会**线性变慢**（持仓页同款问题，实测 1 只 63s）。
        #    `realtime.get_realtime_quotes` 是批量 + 腾讯直连，快得多。
        prices: dict = {}
        try:
            from data import realtime
            prices = realtime.get_realtime_quotes(list(codes)) or {}
        except Exception as e:                               # noqa: BLE001
            _logger.warning("[%s] 批量取价失败（回落为逐个取价）: %s", job_desc, e)
        positions = pm.get_active_positions(prices=prices)

        alerts = pm.check_stop_loss(prices=prices)
        trails = pm.check_trailing_stops(prices=prices)

        # ── 结构体检（集中度 / 流动性）2026-10-17 追加，唯一实现 `core.position_risk.py` ──
        # ⚠️ 取代原来按**代码前缀**粗分的"行业敞口"（60→"金融/公用"、00→"金融/地产"…）：
        #    那种分法把中石化和招商银行算成同一个"行业"，集中度结论是**错的**。
        #    现按 `market_snapshot` 里的**真实行业**计算，并补上流动性（退出所需天数）。
        # ⚠️ 独立 try：体检失败**不能**连累止损/止盈巡检（那是更硬的告警）。
        structure = None
        try:
            from core import position_risk as PR
            built = PR.build_holdings(prices=prices)
            structure = PR.position_risk_report(built["holdings"], built["total_equity"])
            structure["snapshot_date"] = built["snapshot_date"]
        except Exception as e:                               # noqa: BLE001
            _logger.warning("[%s] 结构体检失败（不影响止损巡检）: %s: %s",
                            job_desc, type(e).__name__, e)

        lines = []
        for a in alerts:
            tag = "🔴" if a.get("type") == "止损" else "🟢"
            pct = a.get("loss_pct") if a.get("type") == "止损" else a.get("profit_pct")
            suffix = f"（{pct:+.2f}%）" if isinstance(pct, (int, float)) else ""
            lines.append({"icon": tag, "type": f"{a.get('type')}触发",
                          "code": a.get("code"), "name": a.get("name") or "",
                          "detail": f"现价 {a.get('current_price')} vs 设定 "
                                    f"{a.get('trigger_price')}{suffix}"})
        for t in trails:
            # 文案直接用 kind，不再写死"可上移" —— 浮亏时的建议是"收紧至 ATR 止损位"，
            # 不是追踪上移，写死会让两句话自相矛盾。
            lines.append({"icon": "📐", "type": "追踪止损",
                          "code": t["code"], "name": t["name"] or "",
                          "detail": f"止损 {t['current_stop'] or '未设'} → "
                                    f"{t['suggested_stop']}（{t['kind']}）"})
        for i in (structure or {}).get("issues", []):
            if i.get("kind") == "sector":
                lines.append({"icon": "⚠️", "type": "行业敞口", "code": "", "name": "",
                              "detail": f"{i['sector']} 达 {i['weight']:.1%}，已超 "
                                        f"{i['limit']:.0%} 上限（{len(i.get('members') or [])} 只）"})
            elif i.get("kind") == "single":
                lines.append({"icon": "⚠️", "type": "单票集中",
                              "code": i["code"], "name": i.get("name") or "",
                              "detail": f"占净值 {i['weight']:.1%}，已超 {i['limit']:.0%} 上限"})
            else:
                lines.append({"icon": "⚠️", "type": "流动性",
                              "code": i["code"], "name": i.get("name") or "",
                              "detail": f"持股市值 ≈ {i['days']:.2f} 个交易日成交额"})

        if not lines:
            _logger.info("═══ [%s] 完成: 持仓 %d 只，无触发 ═══", job_desc, len(positions))
            return

        # ⚠️ 2026-10-23：整篇排版改由**可编辑模板**决定（`core/push_templates`），
        #    用户可在「系统配置」页改文案，改完即时生效、不用重启调度器。
        from core import push_templates as T
        tpl_item = T.get_template(T.TPL_RISK_ITEM)
        item_lines = [T.render(tpl_item, row) for row in lines]
        note = ("> 集中度/流动性口径：占比 = 当前市值 ÷ 总资产；"
                "流动性 = 持仓市值 ÷ 该股当日成交额（天）。"
                + (f"行业与成交额取自 {structure.get('snapshot_date')} 的估值截面快照。\n"
                   if structure and structure.get("snapshot_date") else ""))
        body = T.render(T.get_template(T.TPL_RISK), {
            "title": "🛡️ 持仓风控巡检",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M"),
            "count": str(len(positions)),
            "items": "\n".join(item_lines),
            "note": note,
            "footer": T.FOOTERS.get(T.TPL_RISK, ""),
        })
        from core.push import push_markdown
        pushed = push_markdown(body, title=f"持仓风控 {datetime.now():%Y-%m-%d}",
                               dedupe_key=f"risk:{datetime.now():%Y-%m-%d}")
        _logger.info("═══ [%s] 完成: 持仓 %d 只，%d 条触发，推送=%s ═══",
                     job_desc, len(positions), len(lines), pushed)
    except Exception as e:
        _logger.exception("═══ [%s] 任务异常: %s ═══", job_desc, e)


def run_ai_round_job(job_desc: str = "AI交易联赛·盘后决策轮",
                     consume_budget: bool = True):
    """盘后 AI 决策轮（15:58）：3 个随机对照组（零 LLM）+ 3 个 LLM persona。

    ⚠️ **这是 JOBS 里唯一"每跑一次必花钱"的任务**（3 persona × 最多 6 轮 LLM 调用）。
    幂等由 decision_round 额度保证：同日重跑会被挡下（不会重复花钱、不会重复落库）。

    ⚠️⚠️ `consume_budget=False`（人工补跑，`--no-budget`）：
    2026-10-23 事故 —— 中午补跑一次决策轮，把每个 AI 当日**唯一**一次
    `decision_round` 配额吃掉 ⇒ 15:58 正式轮 6 个全部"额度跳过"，
    当天等于没有正式决策。补跑不该消耗正式额度：它只是"把漏掉的那轮补上"，
    产物 UPSERT 覆盖，正式轮跑完仍以正式轮为准。
    """
    start = datetime.now()
    try:
        from core.ai_daily_round import run_daily_round
        from core.analysis_runner import ai_deep_analysis_runner
        # ⚠️ 接通「现场深度分析」执行器（2026-10-15）：此前 `analysis_runner=None`
        #    ⇒ `request_deep_analysis` 是**死工具**（3 个 AI 都在复盘里明确说想用它研究标的，
        #    这是主动性被卡住的首因 —— 能力断线，不是模型不愿动）。
        #    额度仍由 `execute_tool` **先扣后跑**（并发不超额）；
        #    执行器内部对"同日同票"做复用，同一轮里重复请求不会重复烧 token。
        out = run_daily_round(analysis_runner=ai_deep_analysis_runner(source="ai"),
                              consume_budget=consume_budget)
        s = out["summary"]
        _logger.info("═══ [%s] 完成: %.1fs, 成功 %d / 轮次耗尽 %d / 额度跳过 %d / 失败 %d ═══",
                     job_desc, (datetime.now() - start).total_seconds(),
                     s["ok"], s["no_final"], s["skipped"], s["failed"])
        if out.get("briefing_warnings"):
            _logger.warning("═══ [%s] 简报警告（如实展示，不补造）: %s ═══",
                            job_desc, out["briefing_warnings"])
        for r in out["results"]:
            if r["status"] not in ("ok", "skipped"):
                _logger.warning("═══ [%s] %s 未正常完成: %s ═══",
                                job_desc, r["persona"], r.get("error"))
    except Exception as e:                              # noqa: BLE001
        # 兜底：任何异常都不能让调度线程死掉（本项目最大的事故是"调度器根本没在跑"）
        _logger.exception("═══ [%s] 任务异常: %s ═══", job_desc, e)


def run_ai_settle_job(job_desc: str = "AI交易联赛·结算", phase: str = "open"):
    """AI 联赛结算（零 LLM 成本）。

    Args:
        phase: "open" = 开盘后（T+1 撮合 T-1 的买入决策）；"close" = 收盘后
               （日内唤醒卖出按扫描价成交）。
    ⚠️ 记净值**不在这里**（2026-10-15 拆出）：15:05 早于当日日线可用时间，
       原版在此 mark_nav 会全量退化成"成本价兜底"⇒ 假净值。现由 15:52 的
       `ai_mark_nav` 负责（晚于 15:40 快照、早于 15:58 决策轮）。
    """
    start = datetime.now()
    try:
        from core import ai_league
        if phase == "close":
            wakes = ai_league.settle_intraday_wakes()
            _sold = sum(len(r.get("sells") or []) for r in wakes["results"])
            _bought = sum(len(r.get("buys") or []) for r in wakes["results"])
            _logger.info("═══ [%s] 完成: %.1fs, 日内唤醒动作 %d 卖 / %d 补仓 ═══",
                         job_desc, (datetime.now() - start).total_seconds(),
                         _sold, _bought)
        else:
            out = ai_league.settle_daily()
            _logger.info("═══ [%s] 完成: %.1fs, 提交 %d 笔 ═══", job_desc,
                         (datetime.now() - start).total_seconds(),
                         sum(r.get("submitted", 0) for r in out["results"]))
    except Exception as e:                              # noqa: BLE001
        _logger.exception("═══ [%s] 任务异常: %s ═══", job_desc, e)


def run_ai_mark_nav_job(job_desc: str = "AI交易联赛·当日净值"):
    """记录各 AI 账户的当日净值（零 LLM 成本，走 `market_snapshot` 的当日收盘价）。

    ⚠️ 排期必须在**快照（15:40）之后**：否则当日收盘价取不到，净值会退化成成本价兜底
    （2026-10-15 实测事故）。取不到价的账户**当天不记净值**并告警，绝不用成本价冒充。
    """
    start = datetime.now()
    try:
        from core import ai_league
        out = ai_league.mark_nav()
        _logger.info("═══ [%s] 完成: %.1fs, 记净值 %d 个 / 跳过 %d 个（无当日价）═══",
                     job_desc, (datetime.now() - start).total_seconds(),
                     out.get("recorded", 0), out.get("skipped", 0))
        for r in out["results"]:
            if not r.get("recorded"):
                _logger.warning("═══ [%s] %s 未记净值: %s ═══",
                                job_desc, r["persona"], r.get("reason"))
    except Exception as e:                              # noqa: BLE001
        _logger.exception("═══ [%s] 任务异常: %s ═══", job_desc, e)


def run_crowd_job(job_desc: str = "交易拥挤度快照"):
    """盘后算一次全市场拥挤度截面并落表（选股只读表）。

    ## 为什么单独一个任务、不塞进 15:40 的快照任务

    实测单日全市场截面 **146s**（5,572 只 × `get_bars`，其中 20 日涨幅要读 21 根）。
    `snapshot_flow` 本身已跑 9~10 分钟，再挂 2.5 分钟会把它推到 15:52 之后，
    而 15:50 的复盘把 snapshot 当上游依赖 ⇒ 每天挂一条"上游未就绪"。
    这与 `sentiment_ref` 被单独挪到 15:20 是同一个理由：**重活不叠加**。

    ## 为什么是 16:30

    必须在 `snapshot_flow`(15:40，写当日 `market_snapshot` 换手率/成交额)
    与 `_refresh_panel_daily`（写当日收盘到 `daily_overlay`）**之后** —— 两者都是输入。
    15:30~16:00 是 LLM 重任务区（复盘 15:50 / 决策轮 15:58 / 新闻 15:55），
    16:30 是干净的窗口，且下游（次日 09:25 盘前选股）离它有 17 小时余量。

    ## 失败处理

    失败只记 WARNING、不推送（与 `run_sentiment_job` / `run_fin_disclosure_job`
    同一口径）。但**必须进日志**：闸门读不到快照时会显式降级为"未生效"，
    若连日志都没有，"闸门从来不生效"会被误读成"今天没有拥挤的票"。
    """
    start = datetime.now()
    _logger.info("═══ [%s] %s 开始 ═══", job_desc, start.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        from core import crowding
        res = crowding.compute()
        if not res.get("date") or not res.get("rows"):
            _logger.warning("═══ [%s] 未生成：%s ═══", job_desc,
                            (res.get("meta") or {}).get("note"))
            return
        path = crowding.save(res)
        m = res["meta"]
        _logger.info(
            "═══ [%s] 完成 %.1fs：日期 %s 行 %d 拥挤度可用 %d 涨幅可用 %d "
            "命中闸门 %d(%.1f%%) 行业 %s 组 %d 覆盖 %d 合并池 %d ⇒ %s ═══",
            job_desc, (datetime.now() - start).total_seconds(), m["date"], m["rows"],
            m["crowd_n"], m["mom_n"], m["gated"], m["gated_pct"] * 100,
            m["industry_source"], m["industry_groups"], m["industry_covered"],
            m["industry_pool"], path)
        for w in m.get("degraded") or []:
            _logger.warning("═══ [%s] 降级：%s ═══", job_desc, w)
    except Exception as e:                              # noqa: BLE001
        _logger.exception("═══ [%s] 任务异常（闸门会降级为未生效）: %s ═══", job_desc, e)


def run_turn_share_job(job_desc: str = "换手率/流通股本日更"):
    """换手率 / 流通股本日更（`data/panel/turn_share/{code}.parquet`）。

    ## 为什么要有这个任务（2026-11-01）

    `turn_share` 层是 2026-10-29 **一次性回填**的产物：当时只提供了手工脚本
    `scripts/backfill_turn_share.py`，**没有接调度** ⇒ 数据停在 2026-10-28。
    下游 `core/factor_analysis.py` 取 `turnover_pct` 时用 `ffill(limit=3)` 兜
    尾部，所以缺 2 个交易日还能撑；**第 4 个交易日（11-03）起** amihud20 等
    量价因子开始出 NaN —— 这是"会自己变坏"的缺口，不是静态技术债。

    ## 为什么数据源是 `market_snapshot` 而不是新浪

    第一版实现直接调回填脚本打新浪接口，实测**被否决**：
    - 4 并发补 120 只花 250 秒 ⇒ 全量 5,830 只约 **4 小时**，日更不可行；
    - 更糟：akshare 的 `stock_zh_a_daily` 每次调用都新建 V8 isolate（`py_mini_racer`），
      多线程构造会让进程 `[FATAL] Check failed: !IsConfigurablePoolInitialized()`
      **abort** —— 不是 Python 异常，`except` 拦不住，会把调度器连同当天剩余任务一起带走。

    而 `snapshot_flow`（每天 15:40）已经把当日全市场 `turnover` / `float_cap` 落库，
    口径与历史层一致（000001 在 2026-10-29：快照 1.02728 vs 换手率层 1.027278）
    ⇒ 日更只需读库 + 追加 parquet，**秒级、零网络**。历史仍由脚本维护（按需手工跑）。

    ## 失败处理

    不抛、不推送（与 `run_crowd_job` / `run_sentiment_job` 同口径），但**必须落日志**：
    缺数据不会让选股报错，只会让因子悄悄变 NaN，这条 WARNING 是唯一的现场。

    Returns:
        dict: `sync_from_snapshot()` 的结果（codes/rows/missing/errors）；
        任务本身异常时返回 `{"error": ...}`。
    """
    start = datetime.now()
    _logger.info("═══ [%s] %s 开始 ═══", job_desc, start.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        from data import turn_share as TS
        res = TS.sync_from_snapshot(days=10)
        elapsed = (datetime.now() - start).total_seconds()
        if res.get("errors"):
            _logger.warning("═══ [%s] 完成 %.1fs，但 %d 只写入失败 ═══",
                            job_desc, elapsed, len(res["errors"]))
        else:
            _logger.info("═══ [%s] 完成 %.1fs：快照日 %s，更新 %d 只 / %d 行，"
                         "无历史文件的新股 %d 只（跳过）═══",
                         job_desc, elapsed, res.get("dates") or "-",
                         res.get("codes", 0), res.get("rows", 0),
                         res.get("missing", 0))
        return res
    except Exception as e:                              # noqa: BLE001
        _logger.exception("═══ [%s] 任务异常（量价因子尾部会缺当日数据）: %s ═══",
                          job_desc, e)
        return {"error": f"{type(e).__name__}: {e}"}


def run_alpha_decay_job(job_desc: str = "超额衰减监控"):
    """盘后检查"实盘选股的超额是否在衰减"，异常时推飞书。

    ## 为什么盯 `screen_picks` 而不是研究实验的 JSON

    `output/组合AB实验_*.json` 是一次性产物，文件不会自己更新；拿它做每日监控
    等于监控一份历史快照 —— 策略偷偷失效也看不出来。真正要盯的是线上每天都在
    产出的选股（`screen_picks` 带 `ret_Nd`/`bench_ret_Nd`）。

    ## 两个口径同时跑

    - **主判据 T+20**（`min_months=12`）：与项目 T+20 口径一致，噪声小、见效慢；
    - **早期预警 T+5**（`min_months=6`）：能早 3 周看到苗头，但噪声大，
      **只记日志、不推送**，避免"月度噪声"变成日常告警。

    ## 推送与幂等

    只有主判据 `alert`（最近 6 个月累计超额 ≤ 0）才推，`dedupe_key` 带日期，
    同一天不重复。判据 `unknown`（样本不足）**不推送也不当 ok** —— 见
    `core/alpha_decay.py` 的设计：没数据不等于没问题。
    """
    start = datetime.now()
    _logger.info("═══ [%s] %s 开始 ═══", job_desc, start.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        from core import alpha_decay as AD
        from core import live_performance as LP

        out: dict = {"generated_at": start.strftime("%Y-%m-%d %H:%M:%S")}

        # ── 主判据：T+20 ──
        perf20 = LP.monthly_pick_excess(20)
        note20 = LP.coverage_note(perf20, 20)
        if perf20.get("ok"):
            rep20 = AD.decay_report(perf20["ret"], perf20["bench"], recent=6,
                                    min_months=12)
        else:
            rep20 = {"level": AD.LEVEL_UNKNOWN,
                     "reason": perf20.get("reason") or "无可用样本"}
        out["t20"] = {**{k: v for k, v in rep20.items() if k != "rolling"},
                      "coverage": note20}
        _logger.info("═══ [%s] T+20 判定 %s ｜ %s ｜ %s ═══",
                     job_desc, rep20["level"].upper(), rep20.get("reason"), note20)

        # ── 早期预警：T+5（只记日志）──
        perf5 = LP.monthly_pick_excess(5)
        note5 = LP.coverage_note(perf5, 5)
        if perf5.get("ok"):
            rep5 = AD.decay_report(perf5["ret"], perf5["bench"], recent=6,
                                   min_months=6)
        else:
            rep5 = {"level": AD.LEVEL_UNKNOWN,
                    "reason": perf5.get("reason") or "无可用样本"}
        out["t5"] = {**{k: v for k, v in rep5.items() if k != "rolling"},
                     "coverage": note5}
        _logger.info("═══ [%s] T+5 早期预警 %s ｜ %s ｜ %s ═══",
                     job_desc, rep5["level"].upper(), rep5.get("reason"), note5)

        # ── 落盘（供页面/排查，不依赖日志）──
        try:
            from config.paths import data_root
            p = os.path.join(data_root(), "output", "alpha_decay_latest.json")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)
        except Exception as e:                            # noqa: BLE001
            _logger.warning("[%s] 落盘失败（不影响判定）: %s", job_desc, e)

        # ── 只在主判据 alert 时推送 ──
        if rep20["level"] == AD.LEVEL_ALERT:
            from core.push import push_markdown
            res = push_markdown(
                "**策略超额转负告警**\n\n"
                f"- 判定：{rep20['reason']}\n"
                f"- 样本：{note20}\n"
                f"- 时间：{out['generated_at']}\n\n"
                "处置建议：停止加仓 / 复盘最近 6 个月的持仓与因子暴露 / "
                "检查是否有制度或数据源变化。",
                title="策略超额衰减告警",
                dedupe_key=f"alpha_decay_{start.strftime('%Y%m%d')}")
            _logger.info("═══ [%s] 告警推送：%s ═══", job_desc, res)
    except Exception as e:                                # noqa: BLE001
        _logger.exception("═══ [%s] 任务异常（监控本身失败，不要误读为策略正常）: %s ═══",
                          job_desc, e)


def run_fin_disclosure_job(job_desc: str = "财报公告日刷新(PIT)"):
    """刷新财报公告日（PIT 前视防护的数据底座）。

    ## 为什么要有这个定时任务

    `financial_disclosures` 是**前视防护**的前提：`tool_financials(as_of=...)`
    要靠它判断"那一期财报当时公布了没有"。表为空时防护是 fail-closed 的——
    所有历史基本面分析都会返回"无可用财报"。而 exe 用户没有 Python 命令行，
    跑不了 `scripts/backfill_disclosure_dates.py` ⇒ 必须由调度自己维护，
    否则这个能力在分发布局下等于不存在。

    ## 为什么只刷"最近两期"而不是全量

    全量 8 期约 25 秒、每天跑纯属浪费；公告日只在**新一期开始披露**和
    **上一期补更正**时变化。刷最近两期（~6 秒）覆盖两种情形；历史期若要重建，
    用 `scripts/backfill_disclosure_dates.py --periods 8 --reset` 手工跑。

    ## 失败处理

    巨潮/东财都是第三方接口，失败只记 WARNING、**不推送**——一个接口抖动不该
    变成每天一条告警（与 `run_sentiment_job` 同一口径）。但**必须进日志**，
    否则"表里一直没数据"会被误判成"本来就没有财报"。
    """
    start = datetime.now()
    try:
        from data import fin_disclosure as fd
        from db import store
        periods = fd.quarter_ends(2)
        res = fd.backfill(periods=periods)
        cov = store.fin_disclosure_coverage()
        _logger.info("═══ [%s] 完成: %.1fs, 刷新 %s, 写入 %d 行, "
                     "表内累计 %d 期/%d 行 ═══",
                     job_desc, (datetime.now() - start).total_seconds(),
                     periods, res.get("written", 0),
                     cov.get("period_count", 0), cov.get("rows", 0))
        for p in res.get("periods") or []:
            if p.get("error"):
                _logger.warning("═══ [%s] %s 抓取失败（保留旧数据）: %s ═══",
                                job_desc, p.get("period"), str(p["error"])[:160])
    except Exception as e:                              # noqa: BLE001
        _logger.exception("═══ [%s] 任务异常: %s ═══", job_desc, e)


def run_job(mode: str, job_desc: str, trading_day_only: bool = None,
            force: bool = False, job_id: str = None, no_budget: bool = False,
            slot: str = None, model: str = None):
    """执行一次任务（带异常兜底，避免调度线程死亡）。

    这是 `_run_job_impl` 的**薄包装**：唯一职责是"任务级模型覆盖"
    （2026-11-01）。拆一层而不是把 `with` 塞进主体，是因为主体有一串
    early return 的分支（`if mode == "watch": ...; return`），在这些分支之间
    维护一个上下文管理器等于给每个 return 都埋一个漏 reset 的坑。

    Args:
        model: 该任务**使用哪个模型**（空/None = 跟随功能默认）。
               `model_override` 是 ContextVar，同一线程内的所有 LLM 调用都吃到；
               APScheduler 每个 job 在自己的线程池线程里同步执行 ⇒ 不会串到别的任务。
    """
    _start = datetime.now()
    _err = None
    try:
        if not str(model or "").strip():
            return _run_job_impl(mode, job_desc, trading_day_only, force, job_id,
                                 no_budget, slot)
        from llm.deepseek_client import model_override
        with model_override(str(model).strip()):
            _logger.info("[%s] 任务级模型覆盖：%s", job_desc, str(model).strip())
            return _run_job_impl(mode, job_desc, trading_day_only, force, job_id,
                                 no_budget, slot)
    except Exception as e:                      # noqa: BLE001
        _err = e
        raise
    finally:
        # 台账收尾**必须在 finally**：`_run_job_impl` 主体有十几个 early return，
        # 逐个补"写完成"等于给每个 return 埋一个漏写的坑（同一个理由见上面的 docstring）。
        # 写在 `run_job` 这一层还有一个好处：它是 APScheduler 注册的唯一入口，
        # 手动重放（scripts/run_scheduler.py）与启动补跑也都走它。
        _finish_job_ledger(job_id, slot, job_desc, _err,
                           (datetime.now() - _start).total_seconds())


def _finish_job_ledger(job_id: str, slot: str, job_desc: str, err,
                       elapsed: float = None) -> None:
    """收尾执行台账（`run_job` 的 finally 专用）。**绝不抛**。

    台账是排查辅助：写不进去绝不能让一个已经跑完的任务算失败。
    `run_date` 取当前日期（与 `_run_job_impl` 入口同一取法）；跨午夜结束的槽位
    会 UPDATE 0 行、`finished_at` 留空 —— 一年最多几次，不值得为它加一套日期传递。
    """
    if not (job_id and slot):
        return
    try:
        from db import store as _store
        detail = str(job_desc)
        if err is not None:
            detail += f" · {type(err).__name__}: {str(err)[:120]}"
        _store.finish_job_run(
            job_id, datetime.now().strftime("%Y-%m-%d"), slot,
            status="error" if err is not None else "done",
            elapsed=elapsed, detail=detail)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[台账] %s@%s 收尾失败: %s", job_id, slot, e)


def _run_job_impl(mode: str, job_desc: str, trading_day_only: bool = None,
                  force: bool = False, job_id: str = None, no_budget: bool = False,
                  slot: str = None):
    """执行一次任务（带异常兜底，避免调度线程死亡）

    Args:
        trading_day_only: 非交易日是否跳过。None=按 mode 的固有语义判定（内置任务走这条）；
            自定义任务传布尔值。**依赖当日行情的 mode 会被强制为 True**，见下。
        force: **P0-6（2026-10-18）** 显式绕过交易日检查 —— 仅供"手动重放"用。
            默认 False ⇒ 定时路径行为一字不变；非交易日重放选股会产出与交易日
            不可区分的"今日选股"，所以必须由人显式承担（`--force`）。
        job_id: **P0-5（2026-10-18）** 阶段 id（JOBS 里的键）。给了才会做
            **上游依赖检查**（`core.stage_health`）：blocking 阶段的上游缺失 ⇒ 跳过；
            非 blocking 阶段的上游缺失 ⇒ 照跑但打 WARNING（降级要如实）。
            不传 ⇒ 完全跳过检查（自定义任务、手动 --once 走这条），行为与旧版一致。
        slot: **P0（2026-10-31）** 排期时刻（"HH:MM"）。给了就在**入口处**记一条执行
            台账 `job_runs(job_id, run_date, slot)` —— 这是"这个槽位跑过没有"的唯一
            依据，启动补跑靠它判断（APScheduler 的 job 是内存态，重启即忘）。
            记在入口而不是出口：被交易日/上游检查跳过的槽位**也要留痕**，
            否则每次启动都会把它当成"漏跑"去补。
    """
    if job_id and slot:
        try:
            from db import store as _store
            _store.record_job_run(job_id, datetime.now().strftime("%Y-%m-%d"), slot,
                                  status="started", detail=job_desc)
        except Exception as e:                       # noqa: BLE001
            # 台账写不进去不能让任务不跑——但必须留痕（否则补跑判断会失效）
            _logger.warning("[台账] %s@%s 写入失败: %s", job_id, slot, e)
    if mode == "watch":
        run_watch_job(job_desc)
        return
    # Phase 1：cron 只能排除周末，法定节假日必须靠交易日历判断。
    # 新闻采集与交易日无关，不跳过；选股/回测/复盘/回填/风控在非交易日直接跳过。
    #
    # 判定基准只有一处：core/scheduled_jobs.py::TRADING_DAY_MODES（由 mode 推导）。
    # 自定义任务即使把"非交易日也跑"勾上，也只对 `news` 这类 mode 生效——
    # 否则节假日会产出与真实交易日不可区分的"今日选股"，这正是交易日历要解决的问题。
    if mode in _trading_day_modes():
        skip_if_closed = True
    else:
        skip_if_closed = bool(trading_day_only) if trading_day_only is not None else False
    if skip_if_closed and not is_trading_day():
        if force:
            _logger.warning("═══ [%s] ⚠️ 非交易日（%s）但 force=True ⇒ 强制执行 ═══",
                            job_desc, date.today())
        else:
            _logger.info("═══ [%s] 跳过：%s 非交易日 ═══", job_desc, date.today())
            return
    # ── P0-5：阶段依赖前置检查（只对传了 job_id 的内置/自定义阶段生效）──
    # 判据是"上游当日产物是否落库"，不看日志也不看进程 —— 日志说完成但库里没有
    # 等于没完成（本项目反复踩的坑）。检查本身失败不阻断（旧的调度语义优先）。
    if job_id:
        try:
            from core import stage_health as _stage
            _rd = _stage.stage_readiness(job_id)
            if not _rd["ready"]:
                if _rd["blocking"]:
                    _logger.warning("═══ [%s] 跳过：上游未就绪 %s（%s）═══",
                                    job_desc, _rd["missing"], _rd["checks"])
                    return
                _logger.warning("═══ [%s] 上游未就绪 %s ⇒ 继续执行，但产物/正文必须"
                                "标明该项缺失 ═══", job_desc, _rd["missing"])
        except Exception as _e:                           # noqa: BLE001
            _logger.warning("[%s] 阶段依赖检查失败（继续执行，不阻断）: %s", job_desc, _e)
    if mode == "risk":
        run_risk_job(job_desc)
        return
    if mode == "snapshot":
        run_snapshot_job(job_desc)
        return
    if mode == "crowd":
        run_crowd_job(job_desc)
        return
    if mode == "turn_share_sync":
        # 换手率/流通股本日更（17:15）。返回值只是失败只数，不影响调度。
        run_turn_share_job(job_desc)
        return
    if mode == "leftside_intraday":
        run_leftside_job(job_desc, intraday=True)
        return
    if mode == "leftside_close":
        run_leftside_job(job_desc, intraday=False)
        return
    if mode == "sentiment":
        run_sentiment_job(job_desc)
        return
    if mode == "fin_disclosure":
        run_fin_disclosure_job(job_desc)
        return
    if mode == "alpha_decay":
        run_alpha_decay_job(job_desc)
        return
    if mode == "review":
        run_review_job(job_desc)
        return
    if mode == "news":
        run_news_job(job_desc)
        return
    if mode == "news_review":
        run_news_review_job(job_desc)
        return
    if mode == "backfill":
        run_backfill_job(job_desc)
        return
    if mode == "backfill_screen":
        run_backfill_screen_job(job_desc)
        return
    if mode == "ai_round":
        run_ai_round_job(job_desc, consume_budget=not no_budget)
        return
    if mode == "ai_settle_open":
        run_ai_settle_job(job_desc, phase="open")
        return
    if mode == "ai_settle_close":
        run_ai_settle_job(job_desc, phase="close")
        return
    if mode == "ai_close":
        # 合并任务（2026-11-01）：**先结算日内动作，再记当日净值**。
        # 为什么能合：两步都是纯账本、零 LLM —— 结算用的是触发时已记录的扫描价
        #   （与几点执行无关），净值只要求"快照落库之后"。原先拆在 15:05 与 15:52
        #   两个槽位，白白多占一个排期点和一次进程唤醒。
        # 为什么是这个顺序：净值必须反映"当天所有成交之后的持仓" ⇒ 结算在前。
        run_ai_settle_job(job_desc, phase="close")
        run_ai_mark_nav_job(job_desc)
        return
    if mode == "ai_mark_nav":
        run_ai_mark_nav_job(job_desc)
        return
    if mode == "llm_analysis":
        run_evening_analysis_job(job_desc)
        return
    start = datetime.now()
    _logger.info("═══ [%s] %s 开始 (mode=%s) ═══", job_desc, start.strftime("%Y-%m-%d %H:%M:%S"), mode)
    try:
        pipeline = DailyPipeline()
        report = pipeline.run(mode=mode)
        _logger.info("═══ [%s] 完成: %.1fs, picks=%d, skipped=%s, errors=%d ═══",
                     job_desc, report.get("elapsed_seconds", 0),
                     len(report.get("picks", [])), report.get("skipped"),
                     len(report.get("errors", [])))
    except Exception as e:
        _logger.exception("═══ [%s] 任务异常: %s ═══", job_desc, e)


# ══════════════════════════════════════════════════════════════════
# 任务执行台账 + 启动补跑（2026-10-31）
# ══════════════════════════════════════════════════════════════════
# 为什么要有：APScheduler 的 job 是**内存态**。进程一重启，"今天这个槽位跑过没有"
# 就全忘了；而 `misfire_grace_time=3600` 只在进程还活着时才起作用。
# 后果实测：2026-10-30 的 18:10 `evening_analysis`（盘后 TOP8 深度分析）正好撞上
#   重启窗口 ⇒ 静默消失，第二天看报告才发现没跑，而日志里连"跳过"都没有。

CATCHUP_WINDOW_MIN = 180      # 只补"错过 ≤3 小时"的槽位（再久补跑没有意义）
CATCHUP_GRACE_MIN = 12        # 排期后 12 分钟内算"正常在跑"，不补
CATCHUP_DELAY_S = 25          # 启动后等多久再补（避开暖机抢 DB/网络）

# ⚠️ **不给补跑的任务**（每条都有具体理由，不是"懒得补"）：
CATCHUP_SKIP_IDS = frozenset({
    # 选股类：补跑会在非常规时点产出"今日选股"。pick_date 仍是当天，但与 08:15
    # 的口径（盘前、用上一交易日快照）不同 ⇒ 污染 screen_picks 的 T+N 回填分母。
    # ⚠️ 2026-11-01 删除 `noon_screen`（任务已合并进 morning_screen）。
    "morning_screen", "afternoon_full",
    # 结算类：开盘结算绑的是**开盘时点**，18:00 补跑等于用收盘价重算开盘单；
    # 收盘账本（mode=ai_close）同理。净值补跑会在"当天"写出一个假的收盘净值。
    "ai_settle_open", "ai_mark_nav",
    # 联赛决策轮：slot 本身就是语义（09:40 的轮次在 18:00 跑没有意义）。
    "ai_daily_round", "ai_midday_round",
    # 高频：错过这一轮，下一轮很快就到；补跑只会打乱节奏。
    "intraday_watch", "hourly_news",
    # 会推飞书：18:00 补推"今日复盘"会让用户以为刚生成，且与幂等键冲突。
    "daily_review",
})


def _weekday_matches(dow: str, when: datetime) -> bool:
    """`when` 的星期是否落在 cron 的 `day_of_week` 里。

    补跑判据原来**只看 hour/minute、不看星期** ⇒ 周频任务（2026-11-01 起
    `alpha_decay` 改周五）会在周一到周四各被"补跑"一次，等于白跑四遍。
    解析复用 `scheduled_jobs._expand_days`：星期字段只该有一个解析器，
    两处各写一遍必然走样（"mon-fri" 被截成 "mon" 的老坑）。
    """
    from core.scheduled_jobs import _expand_days
    days = _expand_days(dow)
    return bool(days) and when.weekday() in days


def catchup_missed_jobs(now=None, *, window_min: int = CATCHUP_WINDOW_MIN,
                        grace_min: int = CATCHUP_GRACE_MIN,
                        only=None) -> list:
    """补跑**今天错过**的时间槽，返回实际补跑的 job_id 列表。

    判据（全部满足才补）：
      1. 该 job 在 `JOBS` 里、且不在 `CATCHUP_SKIP_IDS`；
      2. 排期时刻是**具体整点**（高频/通配任务如 `intraday_watch` 不补）；
      3. 现在距排期 `grace_min ~ window_min` 分钟之间（太早是在跑，太晚没意义）；
      4. 台账里**没有**该 (job_id, 今天, slot) 记录。

    幂等：补跑前先写台账（status=caught_up），所以同一天重复调用不会补两次；
    补跑的任务内部若判断"非交易日/上游缺失"而跳过，台账仍有记录——语义是
    **"调度器处理过这个槽位"**，这正是补跑要保证的。
    """
    from db import store

    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")
    done = []
    for job_id, cron_kwargs, mode, desc in JOBS:
        if job_id in CATCHUP_SKIP_IDS:
            continue
        if only is not None and job_id not in only:
            continue
        h, m = cron_kwargs.get("hour"), cron_kwargs.get("minute", 0)
        if not isinstance(h, int) or not isinstance(m, int):
            continue                      # "9-11,13-15" / "*" 这类不补
        dow = cron_kwargs.get("day_of_week")
        if dow and not _weekday_matches(dow, now):
            continue                      # 周频任务（周五的 alpha_decay）不在今天
        try:
            slot_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        except ValueError as e:           # 非法时刻（如 hour=24）不该让启动失败
            _logger.warning("[补跑] %s 的排期 %s:%s 非法，跳过: %s",
                            job_id, h, m, e)
            continue
        lag = (now - slot_dt).total_seconds() / 60.0
        if lag < grace_min or lag > window_min:
            continue
        slot = f"{h:02d}:{m:02d}"
        try:
            if store.has_job_run(job_id, today, slot):
                continue
            store.record_job_run(job_id, today, slot, status="caught_up",
                                 detail=f"错过 {lag:.0f} 分钟，启动时补跑")
        except Exception as e:                       # noqa: BLE001
            _logger.warning("[补跑] %s 台账读写失败，跳过（不冒险重复执行）: %s",
                            job_id, e)
            continue
        _logger.info("═══ [补跑] %s（%s）排期 %s，已错过 %.0f 分钟 ⇒ 现在补跑 ═══",
                     job_id, desc, slot, lag)
        try:
            run_job(mode, desc, job_id=job_id)
            done.append(job_id)
        except Exception as e:                       # noqa: BLE001
            _logger.exception("═══ [补跑] %s 执行失败: %s ═══", job_id, e)
    if done:
        _logger.info("═══ [补跑] 完成 %d 个：%s ═══", len(done), "、".join(done))
    else:
        _logger.info("[补跑] 今天没有需要补的槽位")
    return done


def start_catchup_async(delay_s: float = CATCHUP_DELAY_S) -> None:
    """在后台线程里做补跑，**不阻塞调度器启动**。

    为什么要延迟：启动瞬间各模块在暖机（板块宽度、面板、快照），补跑的任务也要
    读 DB/面板，立刻抢会把启动拖长、还可能撞上"上游还没就绪"的假失败。
    为什么要后台线程：`scheduler.start()` 是阻塞调用，补跑若同步做，这段时间
    到点的任务会全部错过——正好制造我们想避免的问题。
    """
    import threading

    def _worker() -> None:
        import time as _time

        _time.sleep(max(0.0, delay_s))
        try:
            catchup_missed_jobs()
        except Exception as e:                       # noqa: BLE001
            _logger.exception("[补跑] 后台补跑异常（不影响调度器）: %s", e)

    th = threading.Thread(target=_worker, daemon=True, name="job-catchup")
    th.start()


def _builtin_job_spec(job_id: str, cron_kwargs: dict, mode: str,
                      desc: str, ov: dict) -> tuple:
    """内置任务的 (trigger, 展示名, 槽位, model)。

    抽出来是为了让"启动注册"与"热同步重建"用**同一份**口径：
    两处各写一遍，迟早在 news 每天跑 / 槽位算法 / 备注拼接上走样。
    """
    ov = ov or {}
    _note = str(ov.get("note") or "")
    if _note and _note not in desc:
        desc = f"{desc}｜{_note}"
    # 星期：cron 里显式给了 `day_of_week` 就用它（周频任务，如 alpha_decay 周五），
    # 否则默认周一至周五；news 类默认每天（含周末，快讯不停）。
    # ⚠️ 必须先把 day_of_week 从 kwargs 里摘掉：`CronTrigger(day_of_week=..., **kw)`
    #    遇到 kw 里同名字段会 TypeError（duplicate keyword argument）。
    _dow = cron_kwargs.get("day_of_week")
    _kw = {k: v for k, v in cron_kwargs.items() if k != "day_of_week"}
    if mode == "news" and not _dow:
        trigger = CronTrigger(**_kw)
    else:
        trigger = CronTrigger(day_of_week=_dow or "mon-fri", **_kw)
    _h, _m = _kw.get("hour"), _kw.get("minute", 0)
    slot = (f"{int(_h):02d}:{int(_m):02d}"
            if isinstance(_h, int) and isinstance(_m, int) else None)
    return trigger, desc, slot, (str(ov.get("model") or "") or None)


def register_builtin_job(scheduler, job_id: str, cron_kwargs: dict, mode: str,
                         desc: str, ov: dict) -> None:
    """注册/重建一个内置任务（先 remove 再 add，与自定义任务同一个理由）。"""
    try:
        scheduler.remove_job(job_id)
    except Exception as e:                       # noqa: BLE001
        _logger.debug("remove_job(%s) 失败（本来就不存在）: %s", job_id, e)
    # ⚠️ 变量名就叫 `desc`：`run_job` 的第二个位置参数是**任务描述**，契约见
    #    tests/test_stage_health.py::test_scheduler_registers_job_id。
    #    spec 可能按覆盖表改写它（用户自定义备注），所以取 spec 的返回值。
    trigger, desc, slot, model = _builtin_job_spec(job_id, cron_kwargs, mode, desc, ov)
    scheduler.add_job(
        run_job, trigger, args=[mode, desc, None, False, job_id, False, slot],
        # 任务级模型：走 kwargs（args 里塞不进去——model 在签名末尾且有默认值）
        kwargs={"model": model},
        id=job_id, name=desc, misfire_grace_time=3600, coalesce=True,
    )


def sync_builtin_overrides(scheduler) -> dict:
    """把内置任务的用户覆盖（启停/备注/模型）同步到**活着**的调度器上。

    调度进程每 60s 调一次（与 `sync_custom_jobs` 同一个循环）。
    边界：读库失败 ⇒ 本轮不动（绝不能把"读不到"当成"全部停用"）。
    """
    from db import store as _store

    try:
        ovs = _store.list_builtin_overrides()
    except Exception as e:                       # noqa: BLE001
        _logger.warning("[内置任务] 读取覆盖失败，本轮不同步: %s", e)
        return {"disabled": [], "restored": [], "updated": [], "changed": False}
    try:
        current = {j.id: j for j in scheduler.get_jobs()}
    except Exception as e:                       # noqa: BLE001
        _logger.warning("[内置任务] 读取调度器任务失败，本轮不同步: %s", e)
        return {"disabled": [], "restored": [], "updated": [], "changed": False}

    disabled, restored, updated = [], [], []
    for job_id, cron_kwargs, mode, desc in JOBS:
        ov = ovs.get(job_id) or {}
        want = ov.get("enabled", True) is not False
        exists = job_id in current
        try:
            if not want:
                if exists:
                    scheduler.remove_job(job_id)
                    disabled.append(job_id)
                continue
            if not exists:
                register_builtin_job(scheduler, job_id, cron_kwargs, mode, desc, ov)
                restored.append(job_id)
                continue
            # 已存在：只有 model / note 变了才重建（改 cron 不在覆盖层范围内）
            cur_model = str((current[job_id].kwargs or {}).get("model") or "")
            want_model = str(ov.get("model") or "")
            cur_note_ok = str(ov.get("note") or "") in str(current[job_id].name or "")
            if cur_model != want_model or not cur_note_ok:
                register_builtin_job(scheduler, job_id, cron_kwargs, mode, desc, ov)
                updated.append(job_id)
        except Exception as e:                   # noqa: BLE001
            _logger.warning("[内置任务] 同步 %s 失败: %s", job_id, e)
    if disabled or restored or updated:
        _logger.info("[内置任务] 热同步：停用 %s / 恢复 %s / 更新 %s",
                     disabled, restored, updated)
    return {"disabled": disabled, "restored": restored, "updated": updated,
            "changed": bool(disabled or restored or updated)}


def create_scheduler() -> BlockingScheduler:
    """创建已注册任务的调度器（不启动）

    除内置任务外，还会把库里的**自定义定时任务**（`scheduled_jobs` 表）一并挂上。
    装载是 best-effort：读库失败也不能让调度器起不来——本项目最大的事故恰恰是
    "调度器根本没启动而无人察觉"，新增一层配置绝不能成为新的启动失败点。
    """
    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    # 内置任务的用户覆盖层（2026-11-01）：启停 / 备注 / 使用模型。
    # 读库失败不能让调度器起不来（同上：启动失败是最大事故）。
    overrides = {}
    try:
        from db import store as _store
        overrides = _store.list_builtin_overrides()
    except Exception as e:                       # noqa: BLE001
        _logger.warning("读取内置任务覆盖失败（按代码默认值启动）: %s", e)
    for job_id, cron_kwargs, mode, desc in JOBS:
        ov = overrides.get(job_id) or {}
        if ov.get("enabled") is False:
            _logger.info("内置任务 %s (%s) 已被用户停用 ⇒ 不注册", job_id, desc)
            continue
        # 同 register_builtin_job：第二个位置参数是 desc（契约测试锁死）
        trigger, desc, slot, model = _builtin_job_spec(job_id, cron_kwargs, mode, desc, ov)
        scheduler.add_job(
            run_job, trigger, args=[mode, desc, None, False, job_id, False, slot],
            kwargs={"model": model},
            id=job_id, name=desc, misfire_grace_time=3600, coalesce=True,
        )
        # ⚠️ 不要用 %02d：hour/minute 可能是字符串（盘中盯盘任务是 "9-11,13-15" / "*"）。
        # 曾因此抛 `TypeError: %d format: a real number is required, not str`——
        # 异常发生在 logging 层（默认打印到 stderr 而不中断），启动流程看似继续，
        # 但日志乱码/中断，排障时极易误判。统一 %s。
        # （UI 配置页有过同样的坑，见 7_⚙️_系统配置.py 的 _cron_field）
        _logger.info("已注册任务: %s (%s) → mode=%s, cron=%s %s:%s",
                     job_id, desc, mode,
                     cron_kwargs.get("day_of_week")
                     or ("每天" if mode == "news" else "周一至周五"),
                     cron_kwargs.get("hour", "*"), cron_kwargs.get("minute", 0))

    # 自定义任务：与内置共用 run_job，只是排期来自库
    try:
        from core.scheduled_jobs import load_custom_jobs
        n = load_custom_jobs(scheduler)
        if n:
            _logger.info("已装载 %d 个自定义定时任务（可用「系统配置」页增删改，改完无需重启）", n)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("装载自定义定时任务失败（内置任务不受影响）: %s", e)
    return scheduler


def custom_job_ids(scheduler) -> list:
    """调度器上**非内置**的任务 id（心跳用它单独上报，不混进 JOB_IDS）。

    从活着的调度器读，而不是读库：这样上报的是"实际挂上了什么"。
    自定义任务注册失败（比如 cron 非法）时会如实缺席，而不是谎报。
    """
    builtin = set(JOB_IDS)
    try:
        return sorted(j.id for j in scheduler.get_jobs() if j.id not in builtin)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("读取自定义任务清单失败: %s", e)
        return []


def reload_custom_jobs(scheduler) -> dict:
    """热加载自定义任务 + 内置任务的用户覆盖（调度进程每 60s 调用一次）。

    异常兜底，不冒泡：同步失败不该拖垮调度主循环。
    """
    out = {"changed": False}
    try:
        from core.scheduled_jobs import sync_custom_jobs
        out = sync_custom_jobs(scheduler, builtin_ids=JOB_IDS)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("热加载自定义定时任务失败: %s", e)
        out = {"changed": False, "error": str(e)}
    try:
        builtin = sync_builtin_overrides(scheduler)
        if builtin.get("changed"):
            out = {**out, "builtin": builtin, "changed": True}
    except Exception as e:                      # noqa: BLE001
        # 内置覆盖同步失败不影响自定义任务那一轮的结果
        _logger.warning("热同步内置任务覆盖失败: %s", e)
    return out



def next_run_times(scheduler) -> list[tuple[str, str]]:
    """查询各任务下次执行时间（供入口脚本展示）。

    ⚠️ **不要在 `scheduler.start()` 之前调用**：APScheduler 的 Job 在未启动时
    **没有 `next_run_time` 属性**（不是 None，是 AttributeError）。
    曾经此处直接取 `job.next_run_time`，入口脚本又恰好放在 start() 之前 →
    启动即 AttributeError → 调度器**从未成功启动过**，所有定时任务一个都没跑
    （日志里"开始/完成"记录数为 0，`analyses`/`screen_picks` 长期空表）。

    所以这里改为 `getattr(..., None)` 容错，并如实显示"(未启动)"。
    """
    rows = []
    for job in scheduler.get_jobs():
        nxt = getattr(job, "next_run_time", None)
        rows.append((job.name, str(nxt) if nxt else "(未启动)"))
    return rows


def pending_run_times(scheduler) -> list[tuple[str, str]]:
    """启动**之前**预估各任务下次执行时间（纯计算，不依赖调度器状态）。

    用途：把"下次执行"打进入口日志。此前这段日志因为踩了上面的坑而整体丢失，
    运维时只看得到"共 N 个任务"却不知道几点跑，出问题无从对照。
    """
    now = datetime.now()
    rows = []
    for job in scheduler.get_jobs():
        trigger = getattr(job, "trigger", None)
        nxt = getattr(trigger, "get_next_fire_time", None)
        if nxt is None:
            rows.append((job.name, "(未知)"))
            continue
        try:
            rows.append((job.name, str(nxt(None, now))))
        except Exception as e:                  # noqa: BLE001
            # 只是"下次执行时间"取不到（触发器类型不支持预计算），不影响调度本身；
            # 但**必须留痕**：否则入口日志里出现"(未知)"时无从判断是哪种触发器。
            _logger.debug("任务 %s 的下次执行时间不可预计算: %s", job.name, e)
            rows.append((job.name, "(未知)"))
    return rows
