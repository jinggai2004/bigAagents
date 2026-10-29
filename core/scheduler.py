"""
定时调度器：基于 APScheduler 的每日自动化。

调度时刻（交易日，周一至周五）：
  - 09:25  盘前选股（screen 模式，快速出候选）
  - 11:30  午间选股复核（screen 模式）
  - 15:30  盘后全流程（rule 模式：选股 + 规则信号回测，零 LLM 成本）
  - 15:35  分析历史 T+N 回填（收盘价落定后补 ret_5d/ret_20d）

说明：
  - 交易日保护：cron 限定周一至周五，**法定节假日由 data/trading_calendar 判定**
    （Phase 1 新增）。原先只靠"数据层自然降级"，节假日照样选股并推送，
    产出的"今日选股/复盘"与交易日不可区分。
  - LLM 深度分析不进定时任务（成本高），由 UI「深度分析」页手动触发；
    但分析历史落库发生在分析链内部（UI/pipeline 共用），回填任务是零 LLM 成本。
"""

import logging
import os
from datetime import date, datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from core.pipeline import DailyPipeline
from data.trading_calendar import is_trading_day

_logger = logging.getLogger(__name__)

# 任务定义: (job_id, cron 表达式, 任务类型, 描述)
# mode/任务类型: screen / rule / review / news / snapshot / backfill(_screen) / watch
JOBS = [
    ("morning_screen", dict(hour=9, minute=25), "screen", "盘前选股"),
    ("noon_screen", dict(hour=11, minute=30), "screen", "午间选股复核"),
    # 持仓风控巡检：放在 14:50，留出收盘前 10 分钟的决策时间 —— 而不是收盘后
    # 才发现"今天被打穿了"。只检查**用户已录入持仓**的止损/止盈与追踪止损，
    # 不生成交易建议；持仓为空时直接返回，不推送、不产生日志噪声。
    # ⚠️ 新增任务后必须重启调度进程，否则健康检查的 code 段会报"跑的是旧代码"。
    ("risk_check", dict(hour=14, minute=50), "risk", "持仓风控巡检"),
    ("afternoon_full", dict(hour=15, minute=30), "rule", "盘后全流程(选股+回测)"),
    ("backfill_analysis", dict(hour=15, minute=35), "backfill", "分析历史T+N回填"),
    # 全市场资金流 + 估值截面快照落库。放在 15:40：晚于全流程、早于复盘，
    # 使复盘与**次日盘前选股**都能读到当天快照（选股不再现场拉全市场，见 run_snapshot_job）。
    # 估值截面（PE/市值/换手率）是**逐日积累**的：它是"六因子权重里 PE 那一项
    # 无法验证"的前置数据，攒够几十个交易日才能算 IC（见 save_daily_market_snapshot）。
    ("snapshot_flow", dict(hour=15, minute=40), "snapshot", "全市场资金流/估值截面快照采集"),
    # 市场情绪参照（2026-10-23）：15:20 —— 必须**早于 15:50 的复盘**，
    # 否则复盘读到的永远是前一天的情绪快照（`load_reference_block` 按采集日取）。
    # 只有 1~2 个 HTTP 请求 + 零 LLM，几秒完成。
    # ⚠️⚠️ 为什么是 15:20 而不是 15:45/15:37（2026-10-23 审查实测）：
    #   15:45 撞 snapshot_flow（15:40 起跑、估 10 分钟）；
    #   15:37 更糟——盘后全流程 15:30 起要跑 **22 分钟**、backfill_screen 要跑 **10 分钟**，
    #   15:30~15:49 整段是重活区 ⇒ 15:20 是"收盘后、重活前"唯一干净的窗口。
    #   它必须在 15:50 复盘之前完成，这才是硬约束；越靠近 15:50 越容易被挤。
    # ⚠️ 排它的目的**不是**给人看，而是**积累历史**：相对基准要求 ≥5 个交易日，
    #    靠用户手动开页面永远凑不齐 ⇒ 这一层会永远停在"不给分位"。
    #    失败只记 WARNING、不推送（非公开接口随时可能失效，不该天天打扰人）。
    ("sentiment_ref", dict(hour=15, minute=20), "sentiment", "市场情绪参照采集"),
    # 选股候选回填单独一个任务（锁名与 analyses 不同，两者可并行不抢锁）
    # ⚠️ D2（2026-10-18 审查）：15:50 → **15:36**。两个理由：
    #    ① 它回填的是**历史**选股样本的 5/20 日收益（要的是"已到期的样本"），
    #       不依赖当日收盘价 ⇒ 没必要排在收盘后；
    #    ② 15:40 的快照要跑 9 分钟（15:40~15:49），原来 15:50 挤在它和复盘之间，
    #       且复盘（15:45）根本看不到当天回填结果（"决策质量评估"用的永远是上一批）。
    #    提前到 15:36 后：15:39 前完成 → 15:50 的复盘能看到当天回填。
    # ⚠️⚠️ 2026-10-28 再提前到 **15:10**：上面那句"15:39 前完成"是**估算错了**。
    #    实测当天 15:36:00 起跑、**15:53:35 才结束（17.5 分钟）** ⇒ 15:50 的复盘检查时
    #    它还没写完 `runs`，于是每天挂一条「上游未就绪 ['backfill_screen']」。
    #    慢的原因是逐条走网络拉 K 线（约 5s/条 × 200 条）。
    #    15:10 起跑 + 17 分钟 = 15:27，距复盘 15:50 有 23 分钟余量；
    #    且它与 `ai_settle_close`(15:05)、`sentiment_ref`(15:20) 都不重叠。
    ("backfill_screen", dict(hour=15, minute=10), "backfill_screen", "选股候选T+N回填"),
    # ⚠️ D2（2026-10-18 审查）：15:45 → **15:50**。快照 15:40 起跑 9 分钟（~15:49），
    #    原来 15:45 必然与快照重叠（`find_conflicts` 早就把这个当例子写进测试）。
    #    挪到 15:50 后 `STAGE_DEPS["daily_review"]` 的两个上游（快照 15:49、回填 15:27）
    #    都已就绪 ⇒ 复盘的"决策质量评估"第一次能用上**当天**的样本。
    ("daily_review", dict(hour=15, minute=50), "review", "每日复盘生成与推送"),
    # 新闻行业利好/利空评估：15:55 —— 晚于快照(15:40)，推荐排序用得上当日资金流；
    # 晚于复盘(15:45)避免与 LLM 重任务叠加。每天跑（周末快讯也有价值），
    # 所以 mode 不进 _trading_day_modes。
    ("news_review", dict(hour=15, minute=55), "news_review", "新闻行业利好/利空评估"),
    # AI 交易联赛·盘后决策轮（2026-10-14）：晚于 15:55 新闻评估 —— 简报要读它的产物。
    # ⚠️ **这是 JOBS 里唯一"每跑一次必花钱"的任务**（3 个 LLM persona × 最多 6 轮）。
    #    成本实测见 output/多AI交易联赛_开发计划_2026-10-14.md；改排期前先想清楚。
    #    随机对照组（3 个）零 LLM 成本，不受影响。
    ("ai_daily_round", dict(hour=15, minute=58), "ai_round", "AI交易联赛·盘后决策轮"),
    # AI 交易联赛·午间决策轮（2026-10-23 新增）：13:05。
    # 为什么需要第二轮：配额原本 1 次/天，模型说一次"等待"当天就归零（实测 7 个交易日
    #   3 个 LLM 人格合计只有 3 笔成交，随机组 15 笔）。两轮都在**次日开盘**结算，
    #   等于每天最多可新增 2 批建仓，频次翻倍。
    # 为什么放 13:05 而不是 9:20/11:35：9:20 看不到当天 9:25 的选股产物；
    #   11:30 选股复核正在跑（算力高峰）⇒ 13:05 既避开高峰，又能用上午真实量价。
    ("ai_midday_round", dict(hour=13, minute=5), "ai_round", "AI交易联赛·午间决策轮"),
    # AI 联赛结算（2026-10-14，S6）：零 LLM 成本（纯账本 + 行情取数）
    # 09:32 开盘后：把前一日（及更早）的买入决策按**当日开盘价**撮合（T+1 口径）
    # ⚠️ 2026-10-24：09:32 → **09:45**。原时刻与 09:25 的盘前选股**每天重叠**
    #    （选股实测 695.9s ⇒ 09:25 跑到 09:36:36），两者抢网络与 SQLite。
    #    之前没人发现，是因为 `JOB_MODES["screen"]` 的估算值还写着 6 分钟
    #    （扩池前的旧数据）⇒ 冲突检测器算出"09:31 就结束了"，永远不报。
    #    挪到 09:45 **不损失任何数据**：撮合用"当日开盘价"，开盘价 09:30 就已确定。
    ("ai_settle_open", dict(hour=9, minute=45), "ai_settle_open", "AI交易联赛·开盘结算(T+1)"),
    # 15:05 收盘后：处理日内唤醒的卖出（扫描价+滑点）。
    # ⚠️ 2026-10-15 拆出净值：原版在这里顺手 mark_nav，但 **15:05 早于当日日线可用时间**
    #    ⇒ 取不到收盘价 ⇒ 用成本价兜底且不留痕 ⇒ 每日净值被伪装成"≈初始资金"，
    #    喂给 AI 复盘就得出"净值不变、无盈亏"的错误结论（实测 3 个账户全 = 1,000,000 整）。
    ("ai_settle_close", dict(hour=15, minute=5), "ai_settle_close", "AI交易联赛·收盘结算(日内卖出)"),
    # 15:52 记当日净值：**必须在快照(15:40)之后**（快照含当日收盘价，是唯一权威来源），
    # 且**在决策轮(15:58)之前**——决策轮的私有段要读当日净值与持仓盈亏。
    ("ai_mark_nav", dict(hour=15, minute=52), "ai_mark_nav", "AI交易联赛·当日净值"),
    # 盘后批量深度分析（2026-10-19，P0）：把"自动深度分析"真正接进每日流程。
    # ⚠️ 选 18:10 而不是 15:5x 有两个理由：
    #    ① DeepSeek 峰谷定价（9-12、14-18 高峰，其余半价）⇒ 18:10 省 50%；
    #    ② 避开 15:30~16:00 的 LLM 重任务叠加（复盘/AI 联赛），不让单进程排队。
    # 它**不重新选股**，只对当日已有候选跑分析（见 core/evening_analysis）。
    ("evening_analysis", dict(hour=18, minute=10), "llm_analysis", "盘后深度分析(TOP8)"),
    ("hourly_news", dict(minute=5), "news", "新闻快讯采集(每小时)"),
    # Phase 1：盘中异动预警。cron 覆盖 9-15 点每分钟，真正的启停由
    # core.watcher 判定（交易日 + 09:30-11:30 / 13:00-15:00），此处只负责"叫醒"。
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

    各段依据（交易所代码分配规则）：
      600/601/603/605 沪主板 · 688/689 科创板 · 900 沪B ❌
      000/001/002/003 深主板/中小板 · 300/301 创业板 · 200 深B ❌
      8xx / 4xx / 920 北交所（920 为 2024 起的新代码段）
    """
    c = str(code or "").strip()
    if len(c) != 6 or not c.isdigit():
        return False
    if c.startswith(("900", "200")):          # B 股
        return False
    return c.startswith(("60", "68", "000", "001", "002", "003",
                         "300", "301", "8", "4", "920"))


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


def run_job(mode: str, job_desc: str, trading_day_only: bool = None,
            force: bool = False, job_id: str = None, no_budget: bool = False):
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
    """
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
    if mode == "sentiment":
        run_sentiment_job(job_desc)
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


def create_scheduler() -> BlockingScheduler:
    """创建已注册任务的调度器（不启动）

    除内置任务外，还会把库里的**自定义定时任务**（`scheduled_jobs` 表）一并挂上。
    装载是 best-effort：读库失败也不能让调度器起不来——本项目最大的事故恰恰是
    "调度器根本没启动而无人察觉"，新增一层配置绝不能成为新的启动失败点。
    """
    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    for job_id, cron_kwargs, mode, desc in JOBS:
        # ⚠️ 新闻与交易日无关（周末也有财经快讯）：mode=news **每天**跑。
        #    原实现把所有任务一律 mon-fri ⇒ 周末整天空采（2026-10-18 周日实测：
        #    库内最后一条停在工作日 22:31，页 12 的 24h 窗口恰好滑出 ⇒ "没新闻"）。
        #    news_review（评估，涉及 LLM 成本）仍限交易日——它评的是"行情相关"。
        if mode == "news":
            trigger = CronTrigger(**cron_kwargs)
            _dow_desc = "每天"
        else:
            trigger = CronTrigger(day_of_week="mon-fri", **cron_kwargs)
            _dow_desc = "周一至周五"
        scheduler.add_job(
            run_job, trigger, args=[mode, desc, None, False, job_id],
            id=job_id, name=desc, misfire_grace_time=3600, coalesce=True,
        )
        # ⚠️ 不要用 %02d：hour/minute 可能是字符串（盘中盯盘任务是 "9-11,13-15" / "*"）。
        # 曾因此抛 `TypeError: %d format: a real number is required, not str`——
        # 异常发生在 logging 层（默认打印到 stderr 而不中断），启动流程看似继续，
        # 但日志乱码/中断，排障时极易误判。统一 %s。
        # （UI 配置页有过同样的坑，见 7_⚙️_系统配置.py 的 _cron_field）
        _logger.info("已注册任务: %s (%s) → mode=%s, cron=%s %s:%s",
                     job_id, desc, mode, _dow_desc,
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
    """热加载自定义任务（调度进程每 60s 调用一次）。异常兜底，不冒泡。"""
    try:
        from core.scheduled_jobs import sync_custom_jobs
        return sync_custom_jobs(scheduler, builtin_ids=JOB_IDS)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("热加载自定义定时任务失败: %s", e)
        return {"changed": False, "error": str(e)}



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
