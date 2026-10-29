"""A股选股引擎

五路**本地初筛**（全市场快照）→ 去重 → 市场检测 → 板块过滤 → 加权多因子精筛 → TOP5

v2 变更（M0）：改用 core/sector.py 的单一权重集 FACTOR_WEIGHTS，
删除"按代码前缀自动套用板块权重矩阵"的死代码分支（该分支在生产路径不可达）。
详见 output/bigAagents_开发计划_2026Q4.md 0.2.1 / 0.2.2。

依赖：
- core/sector.py（板块分类 + 单一权重集）
- data/fund_flow.py（资金流向 + 五维评分）
- data/stock_data.py（K线/基本面数据）
- pywencai（问财接口）
- core/market_detector.py（市场检测）
"""

import logging
import os
import time
from typing import Optional
from core.sector import get_weights, classify_by_board
from db import store
from utils.task_control import TaskCancelled
from utils.pywencai_helper import safe_get

_logger = logging.getLogger(__name__)

# 候选扩池落库上限（2026-10-22）。
# 完整确定性排序的前 N 名随 `result["ranked"]` 返回，由 `save_screen_result`
# 落进 `screen_picks`，供深度分析挑候选（原实现只落 5-10 只，导致`每天只能从
# 7-13 只里挑`，重合度 56% 的根因之一）。
# ⚠️ 为什么是 300：深度分析每天只挑 8-15 只，300 已远超需求；再大只是让
#    `screen_picks` 膨胀、`candidates_for_date` 的 GROUP BY 变慢。
# ⚠️ 这一步**不增加任何计算**：`scored` 本来就是全部候选，只是截断点后移。
PERSIST_TOP = 300

# 单次问财查询的硬超时（秒）。问财依赖 Node 环境且无超时参数，
# 超时后由内置精选池降级接管，避免整条选股流程挂死。
WENCAI_TIMEOUT = 30.0

# ============================================================
# 选股策略定义
# ============================================================

STRATEGIES = {
    # ⚠️ 2026-10-09 本地化：value 不再是"问财语句"，而是**本地条件的可读描述**
    #（真实执行条件见 core/local_screen.py；问财实测已失效，
    #  `output/选股重复度过高_根因分析与改进方案.md`）。
    "主力资金": "近5日主力净流入排序（本地资金流快照）",
    "技术突破": "当日涨幅0.5%~7% + 换手1%~15% 按成交额降序（MACD/均线由因子精筛体现）",
    "低估值成长": "0<PE<30 按 PE 升序（净利同比/ROE 暂无本地数据源）",
    "龙虎榜活跃": "暂无本地数据源（该路当前不可用，会显式标注）",
    "行业龙头": "总市值>200亿 按市值降序（近20日涨幅<15% 由趋势因子体现）",
}

# 硬编码板块精选股票池（**最后兜底**：本地快照也不可用时才会用到，覆盖六大板块）
#
# ⚠️ 2026-10-17 起**配置外置**：真实生效值读 `config/sector_fallback.json`，
#    下面的常量只是"配置不可用时的内置默认"。理由：兜底池是**会随行情过时的运维参数**
#    （换股票、调权重不该改代码、不该跑测试）。
_SECTOR_FALLBACK_BUILTIN = {
    "科技": ["002230", "000063", "688981", "002049", "603019",
             "000977", "300474", "688256", "002415", "300782"],
    "金融": ["000001", "600036", "601318", "600030", "000002",
             "601398", "601688", "600048", "000858", "002142"],
    "消费": ["600519", "000858", "002714", "600887", "000568",
             "603288", "300750", "002594", "601888", "600276"],
    "新能源": ["300750", "002594", "601012", "688599", "300274",
               "002129", "600438", "300763", "688005", "603799"],
    "医药": ["600276", "300760", "000963", "603259", "300122",
             "300015", "002007", "300347", "688180", "300595"],
    "创业板": ["300750", "300760", "300059", "300124", "300274",
               "300122", "300014", "300015", "300999", "300308"],
}


def _sector_fallback_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "config", "sector_fallback.json")


def _load_sector_fallback(path: str = None) -> dict:
    """读 `config/sector_fallback.json`；缺失/损坏时**退回内置默认**。

    ⚠️⚠️ **绝不允许"配置坏了就变成空池"**：那会让降级路径返回 0 只候选，
       比"用内置池"更糟 —— 缺（空）不等于"没有候选"，而且页面会显示成
       "选股无结果"，把运维事故伪装成行情结论。
    ⚠️ `path` 可注入（按调用时刻读取，不缓存）—— 否则测试只能去改真实配置文件。
    """
    import json
    path = path or _sector_fallback_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        pools = data.get("sectors")
        if not isinstance(pools, dict):
            raise ValueError("缺少 sectors 字段")
        out = {str(k): [str(c).strip() for c in (v or []) if str(c).strip()]
               for k, v in pools.items()}
        out = {k: v for k, v in out.items() if v}
        if not out:
            raise ValueError("配置里没有任何板块")
        return out
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[选股] 兜底池配置不可用（%s）⇒ 退回**内置默认**（不是空池）", e)
        return {k: list(v) for k, v in _SECTOR_FALLBACK_BUILTIN.items()}


SECTOR_HARDCODED = _load_sector_fallback()

# 全市场降级池上限（每只候选都要做真实因子评分，控制总耗时）
PRESET_POOL_LIMIT = 18


def _degrade_meta(candidates: list[dict]) -> dict:
    """汇总降级信息（v2 0.2.7）：让"用了内置精选池""数据不足没给分"在报告里可见。

    原实现把这些情况伪装成正常结果：精简池静默替代全市场选股、降级候选被赋予
    假的 50.0 分，调用方无法区分。
    """
    return {
        "degraded": any(c.get("degraded") for c in candidates),
        "degrade_reasons": sorted({c["degrade_reason"] for c in candidates
                                   if c.get("degrade_reason")}),
        "insufficient_data": [c.get("股票代码") for c in candidates
                              if c.get("score") is None],
    }


# 进度阶段 → 整体百分比区间（Phase 1 进度条）
# 全市场选股耗时 2~4 分钟，原先只有一个 spinner，用户容易以为卡死。
_PROGRESS_SPANS = {
    "market": (0, 5),      # 市场环境检测
    "screen": (5, 40),     # 五路本地初筛（全市场快照，毫秒级但留足余量）
    "dedupe": (40, 45),    # 去重 / 降级判定
    # 并发预取（2026-10-09）：这是**当前耗时大头**（上百只 × 2 请求 / 限速），
    # 给足区间让进度条真的在动；⚠️ 漏登记这个键会让 percent 回退到 (0,100) 兜底，
    # 直接把"单调不减"破坏掉（`test_progress_callback_emits_stages_and_monotonic_percent` 抓到过）。
    "prefetch": (45, 70),
    # 流动性下限（2026-10-29）：零联网、毫秒级，但要给一段独立区间 ——
    # 不登记就会走 (0,100) 兜底把 percent 拉回 0（这个坑注释里已经写过一次）。
    "liquidity": (70, 72),
    "score": (72, 98),     # 逐股因子精筛（本地计算，取数已在 prefetch 完成）
    "done": (98, 100),
}


def _emit_progress(cb, stage: str, current: int = 0, total: int = 1, message: str = ""):
    """向进度回调上报一帧。

    回调异常一律吞掉：进度显示只是体验，**绝不能影响选股主流程**。
    唯一的例外是 TaskCancelled（用户点了「取消」）：协作式取消就是靠它在
    下一个进度点打断主流程，所以必须原样穿透。
    """
    if cb is None:
        return
    lo, hi = _PROGRESS_SPANS.get(stage, (0, 100))
    frac = (current / total) if total else 1.0
    frac = max(0.0, min(1.0, frac))
    try:
        cb({
            "stage": stage,
            "current": current,
            "total": total,
            "percent": round(lo + (hi - lo) * frac, 1),
            "message": message,
        })
    except TaskCancelled:
        raise
    except Exception:
        pass


class StockScreener:
    def __init__(self):
        self.strategies = STRATEGIES
        # 单例 Fetcher（避免逐只创建）
        self._stock_fetcher = None
        self._flow_fetcher = None
        self._kline_cache = {}  # K线数据缓存（以code为键，避免重复请求）
        # 过期标的（D5）：code → 原因。**必须可枚举**——"不参与排名"不等于"没这回事"
        self._stale_codes: dict[str, str] = {}
        # 离线模式（D4）：构造时固定一次。运行中改环境变量不应让行为中途变化
        # （否则"前半程联网、后半程不联网"会产出不可复现的结果）。
        try:
            from data import bars as _bars0
            self._offline = _bars0.offline_mode()
        except Exception as e:                      # noqa: BLE001
            _logger.debug("[screener] offline_mode 读取失败，按在线处理: %s", e)
            self._offline = False
        # 基本信息缓存（PE/名称）——由并发预取填充，评分阶段零联网
        self._info_cache = {}
        # 在线资金流熔断（P1-7，2026-10-23）：连续失败 _FLOW_FAIL_LIMIT 次后，
        # 本次运行剩余股票**直接读本地快照**，不再逐只付一次注定失败的请求。
        # ⚠️ 只作用于本实例（一次选股），不做跨运行的长期黑名单 ——
        #    源恢复了下次仍要先试在线（口径可比性优先，降级必须能被看到）。
        self._fund_cache: dict = {}
        self._flow_fail_streak = 0
        self._flow_online_dead = False
        self._flow_dead_reason = ""
        # 当日行业映射缓存（行业分散约束用）；None = 尚未加载，{} = 加载过但不可用
        self._industry_cache = None
        # 当日快照 `{code: {pe, industry}}`（一次 SQL，PE 分位与行业映射共用）
        self._snapshot_fields = None
        # 市场因子点位序列（IVOL 的市场侧输入）；None = 尚未加载，False = 取不到
        self._mkt_index_cache = None
        # 波动率口径计数：本次运行里多少只用了 IVOL、多少只退回总波动率。
        # ⚠️ 必须可枚举——"算术上等价于没换"正是这类改动的典型幻觉
        #    （指数面板缺失时它会**静默**全部退回，而分数看上去完全正常）。
        self.last_vol_basis: dict = {"ivol": 0, "total": 0, "none": 0}
        # 波动率分位的作用域（industry / pool / mixed）；None = 还没算过
        self.last_vol_scope = None
        # 最近一次初筛的执行明细（每路状态/条数/条件降配说明），由 run() 带出
        self.last_strategy_meta: dict = {}
        # 最近一次并发预取的统计（条数/失败数/耗时）
        self.last_prefetch: dict = {}
        # v2：_north_flow_today 随北向因子一并删除（该数据 2024-08 起已停止披露）

    @property
    def stock_fetcher(self):
        if self._stock_fetcher is None:
            from data.stock_data import StockDataFetcher
            self._stock_fetcher = StockDataFetcher()
        return self._stock_fetcher

    @property
    def flow_fetcher(self):
        if self._flow_fetcher is None:
            from data.fund_flow import FundFlowAkshareDataFetcher
            self._flow_fetcher = FundFlowAkshareDataFetcher()
        return self._flow_fetcher

    def run(self, sector_filter: Optional[str] = None, max_stocks: int = 5,
            strategies: Optional[list] = None, progress_cb=None,
            custom_queries: Optional[list] = None, custom_only: bool = False) -> dict:
        """执行选股流程，返回 {"market_signal": ..., "candidates": [...]} 或 {"skipped": True}

        strategies: 指定启用的初筛策略名列表（STRATEGIES 键的子集）；None/空 = 默认全部五路
        custom_queries: 额外执行的**自然语言条件**（本地解析执行，见 core/nl_screener.py）
        custom_only: 只用 custom_queries 初筛、不跑内置五路。

            自然语言选股必须用它：本地核对只覆盖六个因子，市值/换手率/成交额这类
            条件只能靠问财初筛保证。若同时跑内置五路，会混进"并非按用户条件筛出、
            本地又无法核对"的候选，被当成"满足条件"展示——那是错的。
        progress_cb: 可选进度回调，收到
            {"stage", "current", "total", "percent", "message"}；异常会被吞掉，不影响选股。
        """
        # === Stage 0: 市场检测 ===
        from core.market_detector import MarketDetector
        _emit_progress(progress_cb, "market", 0, 1, "市场环境检测（14 指标）")
        signal = MarketDetector().detect()
        _emit_progress(progress_cb, "market", 1, 1,
                       f"市场状态：{signal.market_state}（{signal.total_score} 分）")
        if signal.position_limit == 0:
            _emit_progress(progress_cb, "done", 1, 1, "市况不佳，跳过选股")
            return {
                "market_signal": signal, "skipped": True,
                "reason": f"市场状态: {signal.market_state}({signal.total_score}分)，自动跳过选股",
                "candidates": [],
            }

        # === Stage 1: 初筛（默认五路，可由 strategies 参数选定子集） ===
        all_candidates = self._run_strategies(strategies, progress_cb=progress_cb,
                                              custom_queries=custom_queries,
                                              custom_only=custom_only)

        # === Stage 2: 去重 ===
        _emit_progress(progress_cb, "dedupe", 0, 1,
                       f"初筛返回 {len(all_candidates)} 条，去重中")
        seen = set()
        unique = []
        for stock in all_candidates:
            code = stock.get("股票代码", "")
            if code and code not in seen:
                seen.add(code)
                unique.append(stock)
        _emit_progress(progress_cb, "dedupe", 1, 1, f"去重后 {len(unique)} 只待评分")

        # === Stage 2.5: 初筛无结果时降级到内置精选池（仍走真实因子评分） ===
        if not unique:
            _logger.warning("  [选股] 本地初筛无结果（数据源可能不可用），降级到内置精选股票池")
            unique = self._preset_pool_candidates(sector_filter)
            if strategies:
                for s in unique:
                    s["策略"] = f"内置精选池({'+'.join(strategies)})"
            _emit_progress(progress_cb, "dedupe", 1, 1, "初筛无结果，已降级到内置精选池")

        # === Stage 3: 板块过滤 ===
        if sector_filter:
            valid_codes = set(SECTOR_HARDCODED.get(sector_filter, []))
            matched = [s for s in unique if s.get("股票代码") in valid_codes]
            if not matched:
                # 市场板块类筛选（如"创业板"）按代码前缀匹配
                matched = [s for s in unique
                           if classify_by_board(s.get("股票代码", "")) == sector_filter]
            if not matched:
                top = self._build_sector_top5(sector_filter, max_stocks)
                return {
                    "market_signal": signal, "skipped": False,
                    "candidates": top,
                    **_degrade_meta(top),
                }
            unique = matched

        # === Stage 3.5: 并发预取行情与基本信息（2026-10-09 P0）===
        # 把"逐票串行取数"换成"批量并发取数 + 本地算因子"，候选域才有资格放大
        # （取数产能原来锁死候选域：40 只 ≈6.4 分钟，其中联网占 90%）
        _emit_progress(progress_cb, "prefetch", 0, len(unique), "并发预取候选股行情")
        try:
            self._prefetch([s.get("股票代码", "") for s in unique],
                           progress_cb=progress_cb)
        except Exception as e:                      # noqa: BLE001
            # 契约：**预取的任何异常都不得阻断选股**（逐票串行取数仍是兜底路径）。
            # 测试 test_run_tolerates_prefetch_failure 钉住这条。
            _logger.warning(f"  [选股] 并发预取异常，改为逐票取数: {e}")

        # === Stage 3.6: 流动性下限（CH-3 域约束，2026-10-29）===
        # 放在预取**之后**：成交额优先取 K 线末根（预取已填缓存，零额外联网）。
        # ⚠️ 只作用在五路初筛这条链路上；`score_codes()`（用户给定了代码）不过滤 ——
        #    用户明确点名要看的票不该被域约束悄悄删掉。
        _emit_progress(progress_cb, "liquidity", 0, 1, "流动性下限（成交额分位）")
        unique, liq_meta = self._apply_liquidity_floor(unique, progress_cb=progress_cb)
        if liq_meta.get("note"):
            _emit_progress(progress_cb, "liquidity", 1, 1, liq_meta["note"])

        # === Stage 4: 加权因子精筛（单一权重集） ===
        # ⚠️⚠️ 资金流并发预取**必须在这里也调一次**（2026-10-24 修）：
        #   我 10-23 只把它挂在 `score_codes()`（页面/工具走那条），而调度器走的是
        #   `run()`，直接调 `_score_candidates` ⇒ **预取一次都没执行**。
        #   证据：10-23 15:30 那轮日志里 `资金流并发预取` 出现 **0 次**，
        #   打分阶段吃掉 **293s / 118 只 = 2.5s/只**（本地实测 0.12s/只）。
        #   这正是铁律"能力只在一半路径生效"的又一例：改了要问"另一条入口呢"。
        self._ensure_fund_prefetched(unique)
        scored = self._score_candidates(unique, progress_cb=progress_cb)

        # === Stage 5: 排序取 TOP（确定性排序 + 行业分散）===
        top, rank_meta, ranked_all = self._rank_and_select(
            scored, max_stocks, return_all=True)
        _emit_progress(progress_cb, "done", 1, 1, f"完成，选中 {len(top)} 只")
        result = {
            "market_signal": signal, "skipped": False,
            "candidates": top,
            # 候选扩池落库（2026-10-22）：完整确定性排序的前 PERSIST_TOP 名。
            # ⚠️ 与 `candidates` 的区别：`candidates` 是**做过行业分散**、给用户看的
            #    最终名单；`ranked` 是**纯排序**，用于落库给深度分析挑候选。
            #    两个目的不同，不能混用（给用户看的名单要分散，给机器挑的要按分数全排）。
            "ranked": ranked_all[:PERSIST_TOP],
            # 初筛明细（本地五路：每路状态/条数/条件降配说明；哪一路不可用）
            "strategy_meta": self.last_strategy_meta or {},
            "prefetch": self.last_prefetch or {},
            "rank_meta": rank_meta,
            **_degrade_meta(top),
            # D5：因本地日线过期而不参与排名的标的（**必须可枚举**，
            # "没参与"不等于"没这回事"——否则用户只会看到候选变少而不知道为什么）
            "stale_skipped": dict(self._stale_codes),
            # 流动性下限执行明细（阈值/日期/被剔除清单）：**门槛必须可复核**，
            # 否则用户只看到候选变少而不知道为什么（与 stale_skipped 同一纪律）。
            "liquidity_floor": liq_meta,
            # 本次波动率口径计数（ivol / total / none）—— 全 total 说明市场因子没接上
            "vol_basis": dict(self.last_vol_basis),
            # 波动率分位的**作用域**（industry/pool/mixed）—— pool 意味着行业映射没拿到，
            # 分位悄悄退回旧行为；这个字段是唯一能看出来的地方。
            "vol_pct_scope": getattr(self, "last_vol_scope", None),
        }
        # ── F3（2026-10-18 审查）：离线/口径降级必须**结果级**可见 ──
        # 页 1 只渲染 degraded/degrade_reasons；不标的话，"5 只 100 分的离线名单"
        # 和"5 只 89 分的在线名单"在 UI 上完全无法区分 —— 用户只会看到
        # "今天选出了满分票"，而这两份名单的绝对分数不可直接比（口径不同）。
        if self._offline:
            result["degraded"] = True
            result.setdefault("degrade_reasons", []).append(
                "离线模式（BIGA_OFFLINE=1）：仅本地数据评分——无新闻因子、"
                "PE 为本地快照·新浪口径，分数不可与在线名单直接比较")
        else:
            _snap_pe = [c for c in top if c.get("pe_source") == "snapshot_sina"]
            if _snap_pe:
                result["degraded"] = True
                result.setdefault("degrade_reasons", []).append(
                    f"{len(_snap_pe)} 只候选的 PE 为本地快照·新浪口径"
                    "（实时东财 PE 不可得），与全东财口径名单不可直接比较")
        return result

    def _snapshot_row(self) -> dict:
        """当日快照 `{code: {"pe", "industry"(新浪), "industry_ts"(tushare)}}`（一次 SQL）。

        为什么要合并成一次全表读（2026-10-29）：`_industry_map` 与 PE 分位原本各查
        一次，而 PE 那边走的是 `_info_cache` —— 但**预取用 `skip_valuation=True`，
        根本不取 PE**（2026-10-22 起改由快照兜底）。于是 `pe_raw` 恒为空、
        `_score_pe` 永远走"绝对分档"，**PE 的分位化从 10-22 起就是死路径**。
        页面上看得到"（绝对分档）"四个字，但没人会因此怀疑分位没生效。

        **行业两套口径**（2026-10-29）：`industry` 是新浪 49 类（只覆盖老股，供
        新闻荐股匹配），`industry_ts` 是 tushare 111 类（全市场、粒度细，供选股配额）。
        选股配额走 tushare —— 银行/证券/保险各自独立，`MAX_PER_SECTOR` 才拦得住
        （新浪"金融行业"把银行保险券商混成一类，配额定 2 等于只让 2 只金融股进）。

        快照取不到 ⇒ 返回 `{}`（调用方各自按"缺数据"降级，不阻断选股）。
        """
        if self._snapshot_fields is not None:
            return self._snapshot_fields
        out: dict = {}
        try:
            conn = store.get_connection()
            try:
                row = conn.execute(
                    "SELECT MAX(snapshot_date) AS d FROM market_snapshot").fetchone()
                d = row["d"] if row else None
                if d:
                    for r in conn.execute(
                            """SELECT stock_code, pe, industry, industry_ts
                               FROM market_snapshot WHERE snapshot_date = ?""", (d,)):
                        pe = r["pe"]
                        ok = isinstance(pe, (int, float)) and pe == pe      # NaN 防护
                        out[str(r["stock_code"])] = {
                            "pe": float(pe) if ok else None,
                            "industry": r["industry"] or "",
                            "industry_ts": r["industry_ts"] or "",
                        }
            finally:
                conn.close()
        except Exception as e:                      # noqa: BLE001
            _logger.debug("[选股] 当日快照不可用（无 PE 分位/行业分散）: %s", e)
        self._snapshot_fields = out
        return out

    def _industry_map(self) -> dict:
        """当日 {code: 行业名} —— **tushare 优先、新浪兜底**（本地快照，零联网）。

        用于行业分散约束与**行业内分位**。取不到时返回空 dict —— 约束自动退化为
        "不限制"，而不是阻断选股（缺数据不该让主流程失败）。

        ⚠️ 选股配额/行业内分位用 **tushare 111 类**（`industry_ts`）：银行/证券/保险
        各自独立，配额才能真正拦住"4 只银行"。新浪 49 类（`industry`）把银行保险券商
        混成"金融行业"，配额定 2 等于"金融只能 2 只"而非"银行只能 2 只"。
        tushare 缺失时退回新浪（老股才有新浪值，行为与 10-29 之前一致）。
        """
        if self._industry_cache is not None:
            return self._industry_cache
        out = {}
        for c, v in self._snapshot_row().items():
            ind = v.get("industry_ts") or v.get("industry")
            if ind:
                out[c] = ind
        self._industry_cache = out
        return out

    @staticmethod
    def _net_inflow_wan(stock: dict) -> float:
        """从因子明细里取"近5日主力净流入(万元)"；取不到按 0（不参与并列排序）。"""
        try:
            f = (stock.get("factors") or {}).get("主力资金") or {}
            v = (f.get("value") or {}).get("net_inflow_wan")
            return float(v) if isinstance(v, (int, float)) else 0.0
        except (TypeError, ValueError, AttributeError):
            return 0.0

    def _rank_and_select(self, scored: list, max_stocks: int,
                         return_all: bool = False) -> tuple:
        """排序 + 取 TOP（2026-10-10）。

        两件事在原来的 `sort(...)` 一行里做不到：

        1. **确定性 tie-break**。原实现键只有 `(-score,)`：评分相同时名次完全取决于
           "谁在列表里靠前"，而列表顺序来自固定的候选池 → 同分股票名次恒定。
           实测招行与工行同为 68.2 分，长期以固定顺序霸榜。
           现按序比较：评分（**四舍五入到 0.01**，消除浮点噪声造成的假同分）
           → 有效权重覆盖度（数据更全的更可信）→ 近5日主力净流入 → 代码（最终确定性）。
        2. **行业分散软约束**：TOP 内同一新浪行业最多 `MAX_PER_SECTOR` 只。
           原实现曾让 3 只银行同时入选，组合层面毫无分散。
           若因此凑不满 `max_stocks`，用被跳过的按序补足 —— 分散是**优先**而非**牺牲数量**，
           且放宽事实写进 `rank_meta` 由 UI 展示（不静默）。

        `return_all=True` 时额外返回**完整排序列表**（`ordered`，未做行业分散）。
        用途：候选扩池落库（2026-10-22）—— 深度分析要的是"评分靠前的 N 只"，
        不需要行业分散；而给用户看的 `top` 需要分散。两个目的不同，不能混用一份名单。
        默认 False ⇒ 现有调用方（`run`、`score_codes`）行为一字不变。

        Returns: (top, meta) 或 (top, meta, ordered)
        """
        industry_of = self._industry_map()

        def _key(x):
            score = x.get("score")
            code = str(x.get("股票代码", ""))
            if score is None:
                return (1, 0.0, 0.0, 0.0, code)
            return (0, -round(float(score), 2),
                    -float(x.get("score_coverage") or 0.0),
                    -self._net_inflow_wan(x), code)

        ordered = sorted(scored, key=_key)

        # ⚠️ 行业数据整体不可用 → **不做分散**。否则所有股票都会落进"未知"这个
        #    名义行业、被同一条上限挤掉（实测这种写法会把 5 只名单压成 2 只 + 一堆
        #    误导性的"因行业上限跳过"）。
        if not industry_of:
            _top = ordered[:max_stocks]
            _meta = {
                "sector_cap": None, "skipped_by_sector": [],
                "skipped_by_sector_count": 0, "cap_relaxed": False,
                "note": "行业数据不可用（快照缺失），未做行业分散",
            }
            return (_top, _meta, ordered) if return_all else (_top, _meta)

        top, overflow, per_sector = [], [], {}
        for stock in ordered:
            if len(top) >= max_stocks:
                break
            if stock.get("score") is None:
                # 数据不足：排最后展示，且不占行业名额（它不是"入选"而是"兜底"）
                top.append(stock)
                continue
            code = str(stock.get("股票代码", ""))
            ind = industry_of.get(code)
            if not ind:
                # 单只缺行业（新股/快照没覆盖）→ 豁免，不与他人挤占同一个"未知"名额
                top.append(stock)
                continue
            if per_sector.get(ind, 0) >= self.MAX_PER_SECTOR:
                overflow.append({"股票代码": code, "行业": ind})
                continue
            per_sector[ind] = per_sector.get(ind, 0) + 1
            top.append(stock)

        relaxed = False
        if len(top) < max_stocks and overflow:
            chosen = {str(x.get("股票代码", "")) for x in top}
            for stock in ordered:
                if len(top) >= max_stocks:
                    break
                code = str(stock.get("股票代码", ""))
                if code in chosen:
                    continue
                top.append(stock)
                chosen.add(code)
                relaxed = True

        meta = {
            "sector_cap": self.MAX_PER_SECTOR,
            "skipped_by_sector": overflow[:10],
            "skipped_by_sector_count": len(overflow),
            "cap_relaxed": relaxed,
        }
        return (top, meta, ordered) if return_all else (top, meta)

    def score_codes(self, codes, progress_cb=None, label: str = "本地快照粗筛") -> list:
        """对**给定的代码列表**做因子精算（跳过初筛）。

        供自然语言选股使用（2026-10-07）：候选池已由 `core/local_pool`
        用本地全市场快照产生（D1 决策：移除问财），不需要再走初筛策略。
        `run()` 那条链路的初筛→精算逻辑保持原样，两条互不影响。

        Args:
            codes: 6 位代码列表（顺序即展示顺序，最终按评分排序由调用方决定）
            progress_cb: 进度回调，同 run()
        """
        rows = [{"股票代码": str(c), "股票名称": "", "策略": label}
                for c in (codes or []) if str(c).strip()]
        if not rows:
            return []
        # 并发预取（2026-10-09）：NL 选股的候选池可达上百只，逐票串行取数会拖到十几分钟
        try:
            self._prefetch([r["股票代码"] for r in rows], progress_cb=progress_cb)
        except Exception as e:                      # noqa: BLE001
            _logger.warning(f"  [选股] 并发预取异常，改为逐票取数: {e}")
        # ⭐ P1-7：资金流并发预取（单只 1.05s 是评分阶段的全部成本）
        self._ensure_fund_prefetched(rows)
        _emit_progress(progress_cb, "score", 0, len(rows), f"因子精算：共 {len(rows)} 只待评分")
        return self._score_candidates(rows, progress_cb=progress_cb)

    def _fetch_query(self, query: str, label: str) -> list[dict]:
        """【已停用】原问财单条查询（2026-10-09 起内置五路改走 `core.local_screen`）。

        保留它以兼容可能的旧调用方，但 `safe_get` 目前**恒返回 None**
        （问财响应体解析失败），调用它等于走空结果——新代码不要再使用。

        失败或无结果都返回空列表（由 run() 的降级逻辑接管），
        但会把原因打印出来——静默的 except 曾让"整条降级链从未成功过"两年没被发现。
        """
        try:
            _logger.info(f"  [选股] {label}...")
            # Phase 1：改用带硬超时的 safe_get。问财依赖 Node 环境且无超时参数，
            # 一旦卡住会让 2~4 分钟的选股变成"看起来卡死"；safe_get 超时返回 None，
            # 由降级逻辑接管（并顺带修掉原先模块级 import pywencai 在
            # 未装 Node 环境时直接 ImportError 的问题）。
            df = safe_get(query=query, retry=2, timeout=WENCAI_TIMEOUT)
            if df is None or df.empty:
                _logger.info("    → 无结果")
                return []
            stocks = []
            for _, row in df.iterrows():
                code = str(row.get("股票代码", row.get("代码", "")))
                name = str(row.get("股票简称", row.get("股票名称", "")))
                if code and len(code) == 6:
                    stocks.append({"股票代码": code, "股票名称": name, "策略": label})
            _logger.info(f"    → {len(stocks)} 只")
            return stocks
        except Exception as e:
            _logger.warning(f"    → 失败: {e}")
            return []

    def _run_strategies(self, strategies: Optional[list] = None, progress_cb=None,
                        custom_queries: Optional[list] = None,
                        custom_only: bool = False) -> list[dict]:
        """执行初筛（**本地五路**，2026-10-09 起；问财已失效）。

        ⚠️ 为什么改成本地：pywencai 实测已死（`safe_get` 报
        `'NoneType' object has no attribute 'get'`，响应体解析失败），旧实现
        每次都降级到 `SECTOR_HARDCODED` 前 18 只 → 选股结果跨日高度重复
        （TOP5 4/5 重合、同日 5/5 完全一致）。改为本地全市场快照初筛后
        候选域 18 → 80 只（详见 `output/选股重复度过高_根因分析与改进方案.md`）。

        custom_queries（自然语言条件）改走**本地解析**：`nl_screener.parse`
        + `local_pool.build_pool`（与「自然语言选股」页同一条路）；解析失败或
        快照不可用时逐条记录原因，不再静默失效。

        每路的执行状态与"条件降配说明"写入 `self.last_strategy_meta`，
        由 `run()` 带出到结果，供报告与 UI 展示。
        """
        from core.local_screen import run_local_strategies

        extra = [q for q in (custom_queries or []) if q and str(q).strip()]
        if custom_only:
            if not extra:
                # 宁可显式失败：静默跑空会让 run() 降级到内置精选池，
                # 用户就看到"18 只硬编码股票"冒充"按条件选出的结果"。
                raise ValueError("custom_only=True 时必须提供至少一条非空 custom_queries")
            # ⚠️ custom_only = **不跑内置五路**。首版写成 `selected = []` 后仍调用
            #    `run_local_strategies(labels=selected or None)` —— `[] or None` 是 None
            #    = "全部五路"，于是自定义条件之外又混进了内置候选（测试当场抓到）。
            res = {"candidates": [],
                   "meta": {"routes": [], "notes": [], "truncated": 0,
                            "data_dates": {}, "source": "local"}}
        else:
            selected = [s for s in (strategies or []) if s in self.strategies]
            if not selected:
                selected = list(self.strategies.keys())
            _emit_progress(progress_cb, "screen", 0, 1,
                           f"本地初筛（{len(selected)} 路，全市场快照）")
            res = run_local_strategies(labels=selected or None, progress_cb=progress_cb)
            _emit_progress(progress_cb, "screen", 1, 1,
                           "本地初筛完成：" + "、".join(
                               f"{r['label']}{r['count']}只" for r in res["meta"]["routes"]))

        results = list(res["candidates"])
        self.last_strategy_meta = res["meta"]

        for q in extra:
            cands, note = self._local_custom_query(q)
            if note:
                self.last_strategy_meta.setdefault("notes", []).append(note)
            results.extend(cands)
        return results

    def _local_custom_query(self, text: str) -> tuple[list[dict], str]:
        """一条自然语言条件交给本地池执行。

        Returns: (候选列表, 说明)。说明非空 = 这条条件的执行情况需要让用户看到
        （解析失败 / 快照不可用 / 结果被截断），不静默。
        """
        from core import nl_screener
        from core.local_pool import PoolUnavailable, build_pool
        try:
            query = nl_screener.parse(text)
        except Exception as e:                      # noqa: BLE001
            return [], f"自定义条件「{text[:24]}」无法本地解析（{type(e).__name__}），已跳过"
        try:
            pool = build_pool(query)
        except PoolUnavailable as e:
            return [], f"自定义条件「{text[:24]}」执行失败：{e}"
        cands = [{"股票代码": c, "股票名称": "", "策略": "自然语言条件"}
                 for c in pool.codes]
        note = ""
        if pool.truncated:
            note = (f"自定义条件「{text[:24]}」命中 {len(pool.codes) + pool.truncated} 只，"
                    f"按成交额截断到 {len(pool.codes)} 只逐一精算")
        return cands, note

    def _preset_pool_candidates(self, sector_filter: Optional[str] = None) -> list[dict]:
        """**最后兜底**：内置精选池（2026-10-09 起仅当本地快照也不可用时才会走到）。

        ⚠️ 历史教训（这曾是常态而非兜底）：问财失效期间每次选股都降级到这里，
        候选域被锁死 18 只 → 选股结果跨日高度重复（TOP5 4/5 重合）。本地初筛
        （`core/local_screen`）上线后，正常路径不再经过本函数；走到这里意味着
        **全市场快照也不可用**，报告必须把该降级说清楚。

        全市场降级时限制数量（PRESET_POOL_LIMIT），避免逐股因子评分耗时过长。

        v2（0.2.7）：显式标记 degraded / degrade_reason。原实现静默用 18 只硬编码
        股票替代"全市场选股"，产出报告看不出与真实全市场选股的任何区别。
        """
        if sector_filter:
            codes = list(SECTOR_HARDCODED.get(sector_filter, []))
        else:
            seen, codes = set(), []
            for pool in SECTOR_HARDCODED.values():
                for c in pool:
                    if c not in seen:
                        seen.add(c)
                        codes.append(c)
            codes = codes[:PRESET_POOL_LIMIT]
        reason = ("本地初筛无结果（快照/资金流数据源不可用），改用内置精选池"
                  "（仍走真实因子评分，但候选域收窄到 18 只，代表性极低）")
        return [{"股票代码": c, "股票名称": "", "策略": "内置精选池",
                 "degraded": True, "degrade_reason": reason} for c in codes]

    def _build_sector_top5(self, sector: str, max_stocks: int = 5) -> list[dict]:
        """板块 Fallback（v2：未经因子评分，score 为 None，不再伪造 50.0 分）

        原实现返回硬编码代码 + 假分数 50.0 + 假的"板块推荐"因子，且字段名写成英文
        strategy（与其它路径的"策略"不一致），UI 会把它渲染成一根半满的因子条，
        用户无法分辨这是精选池还是真实评分结果。
        """
        codes = SECTOR_HARDCODED.get(sector, [])[:max_stocks]
        reason = f"板块「{sector}」无初筛命中，回退到内置精选池且未经因子评分"
        results = []
        for code in codes:
            name = code
            try:
                info = self.stock_fetcher.get_stock_info(code)
                if isinstance(info, dict):
                    name = info.get("name", code) or code
            except Exception:
                pass
            results.append({
                "股票代码": code, "股票名称": name,
                "score": None, "factors": {},
                "策略": f"板块精选-{sector}",
                "degraded": True, "degrade_reason": reason,
            })
        return results[:max_stocks]

    def _prefetch(self, codes: list, progress_cb=None) -> dict:
        """并发预取候选股行情与基本信息（2026-10-09 P0）。

        为什么放在**评分之前**而不是评分循环里：评分是"逐票串行 + 每只 1 次联网"，
        取数产能因此锁死了候选域宽度（40 只 ≈6.4 分钟）。把取数批量并发化后，
        评分循环只剩本地计算 —— 候选域可以放大到 120+ 只而总耗时基本不变。

        ⚠️ 预取走 `skip_em=True`（只 新浪+百度，纯 requests）：
        akshare 的 py_mini_racer 不支持多线程，绝不能进并发。
        失败项由评分循环的串行兜底路径接住，不会静默丢股票。
        """
        import os
        # 先 strip 再去重（否则 "600519" 与 " 600519 " 会被当成两只重复取数）
        codes = [c for c in dict.fromkeys(str(x).strip() for x in (codes or [])) if c]
        if not codes:
            return {"ok": 0, "failed": 0, "elapsed": 0.0}
        # 开关：测试/离线环境用（UI 冒烟会真执行页面脚本，单元测试不该联网）。
        # ⚠️ 只在显式设置时跳过；生产路径默认开启——它把单只取数从 9.6s 降到 1.6s。
        if os.environ.get("BIGA_DISABLE_PREFETCH"):
            return {"ok": 0, "failed": 0, "elapsed": 0.0, "disabled": True}

        # ── D4（2026-10-18）：离线模式**不发起任何外呼** ──
        # 不是"让它失败再兜底"——那仍是一次外呼尝试（且要等超时/重试）。
        # 这里只做本地部分：K 线走面板；PE/名称等本地面板里没有的维度**留空**
        # （评分侧会如实记为不可评，而不是编一个中位分）。
        # ⚠️ R7：模式在 `__init__` 固定一次，这里**不再回读环境变量**——
        #    否则"前半程离线、后半程在线"会产出不可复现的结果（与构造注释一致）。
        if self._offline:
            got = sum(1 for c in codes if self._get_kline_data(c) is not None)
            stats = {"ok": got, "failed": len(codes) - got, "elapsed": 0.0,
                     "offline": True, "mode": "local_only",
                     "warnings": ["离线模式：行情只读本地面板；"
                                  "估值/资金流/舆情等本地无数据的维度记为不可评"]}
            self.last_prefetch = stats
            _logger.info("[选股] 离线模式：跳过并发预取，本地命中 %d/%d 只",
                         got, len(codes))
            return stats

        from data.prefetch import prefetch_market_data

        def _cb(done, total):
            _emit_progress(progress_cb, "prefetch", done, total,
                           f"并发预取行情：{done}/{total}")

        try:
            res = prefetch_market_data(codes, fetcher=self.stock_fetcher,
                                       progress_cb=_cb)
        except Exception as e:                      # noqa: BLE001
            # 预取整体失败不能阻断选股——逐票串行路径仍在
            _logger.warning(f"  [选股] 并发预取失败（改为逐票取数）: {e}")
            return {"ok": 0, "failed": len(codes), "elapsed": 0.0, "error": str(e)}
        self._kline_cache.update(res.klines)
        self._info_cache.update(res.infos)
        if res.failed:
            _logger.warning(f"  [选股] 预取失败 {len(res.failed)} 只（交串行兜底）: "
                  f"{list(res.failed.items())[:2]}")
        stats = {"ok": res.ok_count, "failed": len(res.failed),
                 "elapsed": round(res.elapsed, 1), "warnings": res.warnings[:5]}
        self.last_prefetch = stats
        return stats

    # ── 流动性下限（CH-3 域约束，2026-10-29）─────────────────────────
    # 依据 Liu-Stambaugh-Yuan (2019 JFE)：中国样本必须剔除最小 30%（壳股定价失真）。
    # 本项目实测同向：因子在小盘 1/3 组一致变弱（ivol20 −0.0410 vs 中盘 −0.0582；
    # ret5 −0.0438 vs −0.0479），见 `output/文献选股因子可实践性_2026-10-29.md`。
    # ⚠️ 代价必须说清：池子缩 ≈30%，与"扩大候选域"的方向相反 —— 这是有意识的交换，
    #    不是顺手加的过滤器。
    LIQ_FLOOR_PCT = 0.30

    @staticmethod
    def _liq_floor_enabled() -> bool:
        """开关（运行时解析）。默认**开**；`BIGA_LIQ_FLOOR=0/false/off/no` 关。"""
        v = str(os.environ.get("BIGA_LIQ_FLOOR", "1")).strip().lower()
        return v not in ("0", "false", "off", "no", "")

    @staticmethod
    def _liquidity_threshold() -> Optional[dict]:
        """当日成交额分位阈值 —— **按板块各自算**（零联网，读快照表）。

        为什么按**全市场**分位而不是候选内分位：CH-3 是构建投资域时剔最小 30%，
        口径在全市场。若按候选内分位，则"今天这 120 只里最小的 36 只"必被砍掉 ——
        即便整批都是大市值股（会被迫扔掉本来合格的大盘股，这不是文献的意思）。

        **为什么还要再按板块拆（2026-10-29）**：一条全市场阈值会把北交所整段剔掉。
        北交所日成交天然在 400~1600 万（涨跌停 30%、流通盘小），全市场 30% 分位
        （0.49 亿）对它是"永远不合格"。实测 106 只候选被剔的 4 只里有 2 只北交所 ——
        等于用一条流动性门槛**静默取消了一个板块的选股权**，而页面只会显示
        "剔除了几只成交额太低的"。

        拆开后每个板块按**自己的** 30% 分位：
        · 北交所在自己板块里仍有 30% 被剔（这符合 CH-3 的原意：剔掉板块内最小的一批）；
        · 板块样本 < 100 只时退回全市场阈值（样本太小分位没有意义，且要保证
          这个板块不至于因为"样本少"而整体免检）。

        ⚠️ 用当日快照成交额（不是 20 日均额）：算全市场 20 日均额要读 14 个分片
           （实测 6.6s），放选股热路径不划算。当日单日噪声由"30% 分位"吸收。

        Returns: `{"date", "pct", "overall", "by_board": {板块: 阈值}, "n_market"}`
                 快照不可用 ⇒ None。
        """
        from core.sector import classify_by_board
        from db import store
        pct = StockScreener.LIQ_FLOOR_PCT
        conn = store.get_connection()
        try:
            row = conn.execute(
                "SELECT MAX(snapshot_date) AS d FROM market_snapshot").fetchone()
            d = row["d"] if row else None
            if not d:
                return None
            rows = conn.execute(
                "SELECT stock_code, amount FROM market_snapshot WHERE snapshot_date = ? "
                "AND amount IS NOT NULL AND amount > 0", (d,)).fetchall()
        finally:
            conn.close()
        if len(rows) < 100:
            return None
        groups: dict[str, list] = {}
        for r in rows:
            groups.setdefault(classify_by_board(str(r["stock_code"])), []).append(
                float(r["amount"]))
        allv = sorted(v for vs in groups.values() for v in vs)
        overall = allv[max(0, min(int(len(allv) * pct), len(allv) - 1))]
        by_board = {}
        for b, vs in groups.items():
            if len(vs) < 100:                       # 样本太少：该板块用全市场阈值
                continue
            vs.sort()
            by_board[b] = vs[max(0, min(int(len(vs) * pct), len(vs) - 1))]
        return {"date": d, "pct": pct, "overall": overall,
                "by_board": by_board, "n_market": len(allv)}

    @staticmethod
    def _candidate_amount(cand: dict, kline) -> Optional[float]:
        """候选的最近成交额（元）：优先 K 线末根 `amount`，退回快照字段。"""
        try:
            if kline is not None and hasattr(kline, "columns") and "amount" in kline.columns:
                import pandas as _pd
                s = _pd.to_numeric(kline["amount"], errors="coerce").dropna()
                if len(s):
                    return float(s.iloc[-1])
        except Exception as e:                      # noqa: BLE001
            _logger.debug("[选股] K线成交额不可用（%s）: %s", cand.get("股票代码"), e)
        for key in ("成交额", "amount"):
            v = cand.get(key)
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
        return None

    def _apply_liquidity_floor(self, candidates: list, progress_cb=None) -> tuple:
        """剔除成交额低于**自身板块** 30% 分位的候选。

        ⚠️ 三条边界：
          1. 阈值取不到（快照表空/离线）⇒ **不过滤**，并在 meta 里写明"未生效"——
             "缺数据"不等于"都是垃圾股"，静默砍一半候选是最坏的处理；
          2. 候选自身成交额取不到 ⇒ **不剔除**（同上）；
          3. 过滤后少于 10 只 ⇒ 放弃过滤（池子已经很小，再砍就只剩噪声）。
        """
        from core.sector import classify_by_board
        meta = {"enabled": self._liq_floor_enabled(), "applied": False,
                "dropped": [], "threshold": None, "date": None,
                "by_board": {}, "note": ""}
        if not meta["enabled"]:
            meta["note"] = "流动性下限已关闭（BIGA_LIQ_FLOOR=0）"
            return candidates, meta
        th = self._liquidity_threshold()
        if not th:
            meta["note"] = "快照成交额不可用 ⇒ 流动性下限未生效（不剔除任何候选）"
            _logger.warning("[选股] %s", meta["note"])
            return candidates, meta
        keep, dropped = [], []
        for c in candidates or []:
            code = str(c.get("股票代码") or "")
            board = classify_by_board(code)
            # 板块有自己的阈值就用它；否则退回全市场（板块样本 <100 只或市场外代码）
            limit = th["by_board"].get(board, th["overall"])
            amt = self._candidate_amount(c, self._kline_cache.get(code))
            if amt is None:
                keep.append(c)                       # 缺数据：保留
            elif amt < limit:
                dropped.append({"股票代码": code, "股票名称": c.get("股票名称") or "",
                                "板块": board, "成交额": round(amt / 1e8, 3),
                                "阈值": round(limit / 1e8, 3)})
            else:
                keep.append(c)
        if len(keep) < 10:
            meta["note"] = (f"按成交额下限本应剔除 {len(dropped)} 只，但剩余仅 "
                            f"{len(keep)} 只 ⇒ 放弃过滤（样本太小）")
            _logger.warning("[选股] %s", meta["note"])
            return candidates, meta
        # 只把**实际用到的**板块阈值写进 meta（别把 5 个板块全塞进去，读的人只关心生效的那几个）
        used = sorted({d["板块"] for d in dropped}) if dropped else []
        meta.update(applied=True, dropped=dropped, date=th["date"],
                    threshold=round(th["overall"] / 1e8, 3),
                    by_board={b: round(th["by_board"].get(b, th["overall"]) / 1e8, 3)
                              for b in used},
                    note=(f"成交额下限（{int(th['pct'] * 100)}% 分位·按板块）："
                          f"剔除 {len(dropped)} 只"
                          + ("；" + "、".join(
                              f"{b} {round(th['by_board'].get(b, th['overall']) / 1e8, 2)} 亿"
                              for b in used) if used else "")))
        if dropped:
            _logger.info("[选股] %s；被剔除：%s", meta["note"],
                         "、".join(f"{d['股票代码']}({d['成交额']}亿)" for d in dropped[:8]))
        return keep, meta

    def _score_candidates(self, candidates: list[dict], progress_cb=None) -> list[dict]:
        """
        加权因子精筛 — 单一权重集（v2：已取消板块权重矩阵）

        返回结构:
        {
            "股票代码": "000001", "股票名称": "平安银行", "策略": "主力资金",
            "score": 82.5,  # 加权总分 (0-100)
            "板块": "主板深市",  # 交易所板块（展示用）
            "factors": {
                "主力资金": {"score": 22, "max": 25, "detail": "净流入4/5天"},
                ...
            }
        }
        """
        weight_spec = get_weights()
        weights = weight_spec["weights"]
        max_scores = weight_spec["max_scores"]

        scored = []
        total = len(candidates)
        _emit_progress(progress_cb, "score", 0, total, f"因子精筛：共 {total} 只待评分")
        # 横截面分位上下文（2026-10-10 P2）：先用**预取缓存**（零联网）算好本批候选的
        # PE / 波动率分位，再逐票评分 —— 口径从"绝对值落在哪档"换成"在当日候选里排第几"，
        # 这样"低 PE + 低波"不再天然满分，成员才会随市场轮动。
        context = self._build_cross_section(candidates)
        if context.get("disabled") and context.get("note"):
            _logger.info("[选股] %s", context["note"])
        # 波动率分位作用域（industry/pool/mixed）：全落进 pool = 行业映射没拿到，
        # 那时分位等价于旧行为而分数看起来完全正常 ⇒ 必须带到 result 里可核。
        self.last_vol_scope = context.get("vol_scope")
        _logger.info("[选股] 波动率分位作用域=%s（行业内 %s 只 / 池内 %s 只 / 无行业 %s 只，%s 个行业）",
                     context.get("vol_scope"), context.get("vol_industry_n"),
                     context.get("vol_pool_n"), context.get("vol_missing_industry"),
                     context.get("vol_groups"))
        for idx, stock in enumerate(candidates, 1):
            code = stock.get("股票代码", "")
            result = self._score_single(code, weights, max_scores, context=context)
            stock["score"] = result["total"]
            stock["score_status"] = result["status"]
            stock["score_coverage"] = result["coverage"]
            stock["factors"] = result["breakdown"]
            # B2③：本次评分的 PE 口径（eastmoney / snapshot_sina / None）——
            # 双口径并存时 IC/质量评估必须能按口径过滤（screen_picks.pe_source 落库）
            stock["pe_source"] = result.get("pe_source")
            if result["status"] != "ok":
                stock["score_detail"] = result.get("detail", "")
            stock["板块"] = classify_by_board(code)
            if not stock.get("股票名称"):
                stock["股票名称"] = result.get("name", "")
            scored.append(stock)
            _emit_progress(progress_cb, "score", idx, total,
                           f"因子精筛：{code} 完成（{idx}/{total}）")
        return scored

    # 同行业最多入选几只（软约束：优先分散，但选不满时用被跳过的按序补足，
    # 不能让用户因为"分散"只拿到半份名单）
    MAX_PER_SECTOR = 2

    # 有效因子权重占比下限：低于此值视为"数据不足"，不给分数。
    # 理由：若仅靠一两个因子就重新归一化到 0-100，会系统性高估数据稀缺的标的
    # （新股、停牌复牌、数据源不可达），比原来的"缺失给中位分"更危险。
    # ⚠️ 2026-10-10 由 0.5 提到 **0.7**：实测 0.5 太松——覆盖度 0.647（缺两个因子）
    # 也能拿到分数并参与排名，等于让"数据不全的股票"用更少的证据争名次。
    # 现在：缺 1 个权重因子（如舆情 0.1765，覆盖 0.8235）仍可评；
    # 缺 2 个（0.647）则判为数据不足、排到末尾，并在结果里列出。
    #
    # ⚠️⚠️ D6（2026-10-18 审查）：**数值只能有一处定义** —— 这里引用
    #    `core.scoring.MIN_WEIGHT_COVERAGE`，不再写第二份 0.7。
    #    原来的"两份 0.7"正是 P0-2 要消灭的形态（改一处忘另一处 = 静默漂移）。
    from core.scoring import MIN_WEIGHT_COVERAGE as MIN_WEIGHT_COVERAGE

    def _score_single(self, code: str, weights: dict, max_scores: dict,
                      context: dict = None) -> dict:
        """
        对单只股票执行加权因子评分（v2：缺失数据不评分，只在有效权重上归一化）

        返回:
            {
              "total": 82.5 | None,          # None = 有效权重不足，数据不足
              "status": "ok" | "insufficient_data",
              "coverage": 0.88,              # 有效因子权重占比
              "breakdown": {"主力资金": {"score": 22, "max": 25, "weight": .2353,
                                          "status": "ok"|"unknown", "detail": "..."}},
              "name": "平安银行"
            }

        v2 变更（M0 0.2.5）：原实现对缺数据的因子统一给 max//2（中位分），
        资金因子失败则给 0，两者都不可比且不可见——"数据拿不到"被伪装成
        "该股票表现中等"或"该股票资金面差"。现在缺失的因子标记为 unknown 并从
        分子分母中同时剔除，总分 = Σ(score/max·w) / Σ(w_有效) × 100。
        """
        breakdown = {}
        # 本股在本批候选中的横截面分位（None = 未启用分位化 / 该股无有效原始值）
        _ctx = ((context or {}).get("by_code") or {}).get(code)
        info = {}
        # ⚠️ PE 与名称**必须联网取（东财口径）**，不能用当日快照（新浪口径）替代：
        #    2026-10-09 实测两个口径不同 —— 长鑫科技(688825) 快照 PE 1700.0 vs
        #    东财 42.32（差 40 倍），茅台 19.235 vs 19.39。PE 因子是**绝对分档**，
        #    换口径会让评分整体漂移、且与历史 screen_picks 不可比。
        #    （曾试图用快照省掉这次请求：8 只评分从 77s 降到 30s，但代价是口径混用，
        #      已回退。「跨实现取同一指标，先对齐定义再比数」。）
        #    ⇒ 若将来要把 PE 也本地化，必须先做**口径对齐**（同一源、同一时点），
        #      而不是简单地"库里有什么就用什么"。
        # ① 优先用**并发预取的缓存**（零联网）；② 缓存没有才逐票取（串行兜底，
        #    这条路径允许走东财 akshare——它不在并发环境里，慢一点但口径完整）。
        info = self._info_cache.get(code)
        if not isinstance(info, dict) and not self._offline:
            # ⚠️ D4（2026-10-18）：离线模式**不发起外呼** —— 预取没填就留空，
            #    PE 因子随之记不可评（本地面板估值列只覆盖 12/5830，本来就没有）。
            try:
                # ⚠️ skip_valuation=True（2026-10-22）：这里是预取失败后的**串行兜底**。
                #    若再逐票打百度，会重演"116 只并发被限流、选股 43 分钟"的事故
                #    （百度 `stock_zh_valuation_baidu` 是逐票接口，返回 HTML 时抛
                #    `Expecting value: line 1 column 1`）。
                #    PE/PB 由下面的 `_snapshot_pe()` 从 market_snapshot 读，零网络。
                info = self.stock_fetcher.get_stock_info(code, skip_valuation=True)
            except Exception:
                pass
        if not isinstance(info, dict):
            info = {}

        # ---- 预加载K线数据（趋势 + 均线排列 + 波动率共用，避免重复请求） ----
        w_ma = weights.get("均线排列", 0.0)
        w_vol = weights.get("波动率", 0.0)
        w_trend = weights.get("趋势", 0.0)
        kline = None
        if w_ma > 0 or w_vol > 0 or w_trend > 0:
            kline = self._get_kline_data(code)

        # ---- 逐因子评分（键与 core.sector.FACTOR_WEIGHTS 逐字一致） ----
        if weights.get("主力资金", 0.0) > 0:
            breakdown["主力资金"] = self._get_fund_factor(code)
        pe_ratio = info.get("pe_ratio")
        pe_source = "eastmoney" if pe_ratio is not None else None
        if pe_ratio is None:
            # B2③（2026-10-18 用户拍板：双口径 + 标来源）：联网拿不到 PE
            # （离线模式 / 预取失败）⇒ 读本地 market_snapshot（新浪口径）。
            # ⚠️ 与东财口径有系统性差异 ⇒ 明细必须标来源、screen_picks 落 pe_source。
            pe_ratio = self._snapshot_pe(code)
            if pe_ratio is not None:
                pe_source = "snapshot_sina"
        if weights.get("PE估值", 0.0) > 0:
            # 分位化（2026-10-10）：横截面分位，2026-10-29 起**行业内**；无分位数据时退回绝对分档
            _pe_res = self._score_pe(
                pe_ratio, max_scores.get("PE估值", 15),
                pct=_ctx.get("pe_pct") if _ctx else None,
                peers=(context or {}).get("pe_n", 0) if _ctx else 0,
                scope=(context or {}).get("pe_scope") if _ctx else None)
            if pe_source == "snapshot_sina" and _pe_res.get("status") == "ok":
                _pe_res = dict(_pe_res)
                _pe_res["detail"] = (
                    f"{_pe_res.get('detail')}（PE={pe_ratio:.1f}，本地快照·新浪口径——"
                    "与东财实时口径有系统性差异，来源已记录到 screen_picks.pe_source）")
                _pe_res["pe_source"] = "snapshot_sina"
            breakdown["PE估值"] = _pe_res
        if w_trend > 0:
            breakdown["趋势"] = self._score_trend(kline, max_scores.get("趋势", 15))
        if w_ma > 0:
            breakdown["均线排列"] = self._score_ma_alignment(
                kline, max_scores.get("均线排列", 10))
        if w_vol > 0:
            breakdown["波动率"] = self._score_volatility(
                kline, max_scores.get("波动率", 10),
                pct=_ctx.get("vol_pct") if _ctx else None,
                peers=(context or {}).get("vol_n", 0) if _ctx else 0,
                scope=(context or {}).get("vol_scope") if _ctx else None)
        if weights.get("新闻正面率", 0.0) > 0:
            # v2（0.2.3）：把股票简称一并传入，按"代码或简称"匹配新闻
            # D4：离线模式只查**库内**（`allow_fetch=False`），不按个股直查
            breakdown["新闻正面率"] = self._score_news_factor(
                code, max_scores.get("新闻正面率", 15), name=info.get("name") or None,
                allow_fetch=not self._offline)

        # ---- 只在有效权重上归一化 ----
        # ⭐ P0-2（2026-10-18）：归一化与"数据不足"判定搬到 `core/scoring.score_weighted`
        #    —— 原来选股 6 因子加权 / 回测 4 因子固定档各写一套，回测结论无法迁移。
        from core import scoring as _sc
        _agg = _sc.score_weighted(breakdown, weights, self.MIN_WEIGHT_COVERAGE)
        if _agg["status"] != "ok":
            return {
                "total": None, "breakdown": breakdown,
                "status": "insufficient_data", "coverage": _agg["coverage"],
                "detail": _agg["detail"],
                "name": info.get("name", ""),
                "pe_source": pe_source,
            }

        return {"total": _agg["total"], "breakdown": breakdown,
                "status": "ok", "coverage": _agg["coverage"],
                "name": info.get("name", ""),
                "pe_source": pe_source}

    # ================================================================
    # 因子评分子方法
    #
    # 统一契约：每个方法返回 {"score", "max", "status", "detail"}，
    # 其中 status ∈ {"ok", "unknown"}；unknown 表示该因子数据不可得，
    # 调用方会把它从分子与分母中同时剔除（不得用中位分之类占位值代替）。
    # ================================================================

    def _local_fund_factor(self, code: str) -> Optional[dict]:
        """本地 `fund_flow_snapshot` → **快照口径**五维评分（零联网，B2-①）。

        ⚠️ 口径纪律（为什么**不**替换在线正常路径）：资金流有两套口径 ——
        逐日模式（东财，含单型拆分，**25 分制**）与快照模式（全单型合计，
        **20 分制**，「结构」不可评）。两者不可比 ⇒ 本地快照只在
        ① 离线模式、② 在线取数**失败后的降级** 时使用，且 detail 必须写明来源。
        替换在线正常路径会让评分与历史 `screen_picks` 不可比。

        ⚠️ 新鲜度：该股的 `snapshot_date` 必须 == 全市场最新快照日 ——
        某股缺最新一天时，它的"最新"不是市场的最新，宁可不可评（缺就是缺）。
        """
        try:
            from db import store
            import pandas as pd
            from data.fund_flow import compute_five_dim_score
            conn = store.get_connection()
            try:
                row = conn.execute(
                    "SELECT snapshot_date, net_today, net_3d, net_5d, net_10d "
                    "FROM fund_flow_snapshot WHERE stock_code = ? "
                    "ORDER BY snapshot_date DESC LIMIT 1", (code,)).fetchone()
            finally:
                conn.close()
            if row is None or not row["snapshot_date"]:
                return None
            fresh = store.latest_fund_flow_date()
            if fresh and str(row["snapshot_date"]) < str(fresh):
                return None                              # 该股落后于全市场 ⇒ 不可评
            df = pd.DataFrame([{
                "净额_今日": row["net_today"], "净额_3日": row["net_3d"],
                "净额_5日": row["net_5d"], "净额_10日": row["net_10d"],
            }])
            result = compute_five_dim_score(df)
            if result.get("mode") != "snapshot":
                return None
            _unavail = result.get("unavailable_dims") or []
            return {"score": result["score"], "max": result["max"], "status": "ok",
                    "detail": (f"五维评分 {result['score']}/{result['max']}"
                               f"（本地快照 {row['snapshot_date']}，快照口径）"
                               + (f"（{'、'.join(_unavail)}不可评）" if _unavail else "")),
                    "breakdown": result.get("breakdown", {}),
                    "value": {"net_inflow_wan": result.get("total_flow_wan")}}
        except Exception as e:                          # noqa: BLE001
            _logger.debug("[screener] %s 本地资金快照不可用: %s", code, e)
            return None

    def _snapshot_pe(self, code: str) -> Optional[float]:
        """本地 `market_snapshot` 的 PE（**新浪口径**，B2③ 双口径之"本地侧"）。

        ⚠️ 只取**全市场最新快照日**的行 —— 该股不在最新一天 ⇒ 视为没有
        （与资金流本地快照同一纪律：它的"最新"不是市场的最新，宁缺毋旧）。
        ⚠️ 新浪口径与东财实时口径存在系统性差异（项目实测 688825 快照 1700 vs
        东财 42.32）⇒ 使用它的因子明细**必须**标来源，`screen_picks.pe_source`
        必须落库 —— 否则 IC/质量评估会把两套口径混在一起算。
        """
        try:
            v = self._snapshot_row().get(str(code), {}).get("pe")
            return v if v else None
        except Exception as e:                            # noqa: BLE001
            _logger.debug("[screener] %s 本地估值快照不可用: %s", code, e)
            return None

    _FLOW_FAIL_LIMIT = 3

    def _note_flow_failure(self, code: str, reason: str) -> None:
        """记账在线资金流的连续失败；达到阈值就熔断（本实例内直接走本地）。"""
        self._flow_fail_streak += 1
        if not self._flow_online_dead and self._flow_fail_streak >= self._FLOW_FAIL_LIMIT:
            self._flow_online_dead = True
            self._flow_dead_reason = reason
            _logger.warning(
                "[screener] 在线资金流连续失败 %d 次（最近一次 %s，%s）⇒ "
                "本轮改用本地快照口径，不再逐只重试（每只可省 ≈1s）",
                self._flow_fail_streak, reason, code)

    def _get_fund_factor(self, code: str) -> dict:
        """资金因子（**带缓存**）：并发预取先填好，评分阶段零外呼。

        ⭐ P1-7（2026-10-23）实测：单只资金流 **1.05s**，是整个评分阶段
        （K 线 0.021s + PE 0.004s）的**全部成本** —— 116 只 ≈2 分钟、
        300 只 ≈5 分钟，全部耗在同一件事上。并发 6 路实测 6.9s→3.0s。
        """
        if code in self._fund_cache:
            return self._fund_cache[code]
        out = self._compute_fund_factor(code)
        self._fund_cache[code] = out
        return out

    def _ensure_fund_prefetched(self, rows: list) -> dict:
        """资金流并发预取 —— **所有打分入口共用这一处**。

        ⚠️ 为什么单独抽出来：它被两个入口调用（`run()` 给调度器/回测、
        `score_codes()` 给页面/工具）。只挂一个入口的后果实测过一次
        （见 `run()` 里 Stage 4 的注释）：调度器那条路**完全没预取**，
        118 只白白多花 ~293s。**新加入口时必须调这里。**
        """
        if not rows:
            return {"ok": 0, "total": 0, "skipped": True}
        try:
            pf = self._prefetch_fund([r.get("股票代码") for r in rows])
            if pf.get("ok"):
                _logger.info("[选股] 资金流并发预取 %d/%d 只（评分阶段零外呼）",
                             pf["ok"], pf["total"])
            return pf
        except Exception as e:                      # noqa: BLE001
            # 预取是**加速**手段，失败必须退回串行兜底，不能影响选股
            _logger.warning("[选股] 资金流预取异常，退回串行: %s", e)
            return {"ok": 0, "total": len(rows), "error": str(e)}

    def _prefetch_fund(self, codes: list) -> dict:
        """并发预取资金流（口径**完全不变**，只是把串行网络等待并行化）。

        ⚠️ 为什么不去掉联网直接用本地快照：资金流有两套口径（在线 25 分制 /
        快照 20 分制），换口径会让评分与历史 `screen_picks` **不可比**。
        这里只并行、不换源 —— 降级仍由 `_compute_fund_factor` 原有的链路决定。
        ⚠️ akshare 的 py_mini_racer 不支持多线程（见 `_prefetch` 注释），
        所以并发度压到 4，且**任何异常都退回串行**（宁可慢，不可错）。
        """
        import os
        todo = [c for c in dict.fromkeys(str(x).strip() for x in (codes or []))
                if c and c not in self._fund_cache]
        if not todo or self._offline or os.environ.get("BIGA_DISABLE_PREFETCH"):
            return {"ok": 0, "total": len(todo), "skipped": True}
        from concurrent.futures import ThreadPoolExecutor
        ok = 0
        try:
            with ThreadPoolExecutor(max_workers=4) as ex:
                for c, r in zip(todo, ex.map(self._compute_fund_factor, todo)):
                    self._fund_cache[c] = r
                    ok += 1
        except Exception as e:                                # noqa: BLE001
            _logger.warning("[screener] 资金流并发预取失败，退回串行: %s", e)
        return {"ok": ok, "total": len(todo), "skipped": False}

    def _compute_fund_factor(self, code: str) -> dict:
        """资金因子 — 调用五维评分模型（**不做缓存**，供预取与串行兜底共用）

        ⚠️ D4（2026-10-18）：离线模式下**不发起外呼** —— 改读本地
        `fund_flow_snapshot`（快照口径，B2-①）；本地没有才记不可评。
        不写成"让它失败再兜底"——那仍是一次外呼尝试。
        """
        if self._offline:
            local = self._local_fund_factor(code)
            if local is not None:
                return local
            return {"score": None, "max": 25, "status": "unknown",
                    "detail": "离线模式：本地无该股资金流快照（fund_flow_snapshot）"}

        # ⭐ P1-7（2026-10-23）：**在线源连续失败后本进程内直接走本地**。
        # 为什么必须熔断：实测本账号 tushare `moneyflow` **没有权限**（每次返回
        # "没有接口(moneyflow)访问权限"），akshare 东财也在熔断中 ⇒ 每只股票都要
        # 先花 ≈1 秒发一次注定失败的请求，再降级到本地快照。
        # 116 只 = 白白多花 ~2 分钟，300 只 = ~5 分钟，而且日志被刷屏。
        # 熔断后直接读 `fund_flow_snapshot`（零联网），**口径照样在 detail 里标明**。
        if self._flow_online_dead:
            local = self._local_fund_factor(code)
            if local is not None:
                local = dict(local)
                local["detail"] += "（降级：在线资金流源本轮已判定不可用，直接读本地快照 20 分制）"
                local["degraded"] = True
                return local
            return {"score": None, "max": 25, "status": "unknown",
                    "detail": "在线资金流源本轮不可用，且本地无该股快照"}
        try:
            flow_result = self.flow_fetcher.get_fund_flow_data(code)
            if flow_result.get("data_success"):
                flow_data = flow_result["fund_flow_data"]
                df_list = flow_data.get("data", [])
                from data.fund_flow import compute_five_dim_score
                import pandas as pd
                df = pd.DataFrame(df_list)
                result = compute_five_dim_score(df)
                # ⚠️ 分母用 `result["max"]`（**可评维度**的满分和），不能硬编码 25：
                # 快照源「结构」维度无单型拆分 → 不可评，分母是 20。
                # 硬编码 25 会把"取不到"当成"得 0 分"（2026-10-10 修正）。
                _unavail = result.get("unavailable_dims") or []
                return {"score": result["score"], "max": result["max"], "status": "ok",
                        "detail": f"五维评分 {result['score']}/{result['max']}"
                                  + (f"（{'、'.join(_unavail)}不可评，已剔除分母）"
                                     if _unavail else ""),
                        "breakdown": result.get("breakdown", {}),
                        # 原始值（万元）：自然语言选股的本地核对要用，避免去解析 detail 文本
                        "value": {"net_inflow_wan": result.get("total_flow_wan")}}
            _logger.info("[screener] %s 实时资金流返回不可用（data_success=False）", code)
            self._note_flow_failure(code, "data_success=False")
        except Exception as e:
            _logger.warning(f"[WARN] 资金数据获取失败 {code}: {e}")
            self._note_flow_failure(code, type(e).__name__)
        # 网络不可得（异常 / data_success=False）⇒ 降级用本地快照
        # （快照口径 20 分制，detail 必须写明 —— 降级必须随数据走）
        local = self._local_fund_factor(code)
        if local is not None:
            local = dict(local)
            local["detail"] += "（降级：实时资金流不可得，改用本地快照口径 20 分制）"
            local["degraded"] = True
            return local
        # v2：原实现返回 score=0，等价于把"拿不到数据"当成"资金面最差"
        return {"score": None, "max": 25, "status": "unknown",
                "detail": "资金数据不可得"}

    # ================================================================
    # 横截面分位化（2026-10-10，P2）
    #
    # 为什么必须做：实测 TOP5 成员在"候选域 18→112、排序确定化"之后**依然完全不变**，
    # 因为成员由分数决定，而分数被绝对值分档结构性支配 ——
    # PE 用绝对档位（银行 5~8 恒定满分，满分率 49%）、波动率越低越好（银行天然低波，
    # 满分率 59%）。绝对口径下"同一批股票天生占优"，无论候选域多大都改不了。
    #
    # 分位化把口径换成"**在当日候选里排第几**"：最便宜的 20% 得满分、最贵的 20% 得 0 分。
    # 这样分数是相对的，成员才会随市场轮动。
    #
    # 三条纪律：
    # 1. **保留绝对 sanity**：分位是相对的，但不能荒谬 —— 亏损股（PE<=0）仍判 unknown、
    #    PE>300 仍判 0 分、波动率>100% 仍判 0 分。相对口径不等于放弃常识。
    # 2. **样本太小不分位**：候选集 < MIN_FOR_PERCENTILE 时退回绝对分档（分位在 3 只里
    #    排名毫无统计意义），并在 detail 里说明 —— 口径切换必须可见。
    # 3. **计算只有一份实现**：`_annualized_vol` / `_momentum20` 同时被因子评分与
    #    分位构建调用（两处各写一份必然分叉）。
    # ================================================================

    # 分位化最少需要的候选数（低于此值退回绝对分档）
    MIN_FOR_PERCENTILE = 8
    # 绝对值 sanity 上界（超出即为"明确差"，不参与分位）
    PE_SANE_MAX = 300.0
    VOL_SANE_MAX = 100.0

    @staticmethod
    def _annualized_vol(df) -> Optional[float]:
        """20 日年化波动率（**百分数**）—— 委托 `core.indicators`（唯一实现）。

        ⚠️ 单位：indicators 内部一律小数，这里 ×100 显式转百分数（调用点转换）。
        """
        if df is None or not hasattr(df, "columns") or len(df) < 21:
            return None
        try:
            from core import indicators as _ind
            v = _ind.annualized_vol(df["Close"], 20)
            return None if v is None else float(v * 100.0)
        except Exception:                           # noqa: BLE001
            return None

    def _market_index(self):
        """市场因子点位序列（全市场等权，`core.market_factor` 唯一来源）。

        首次调用会构建（实测 6.6s，之后读落盘缓存约 10ms）—— 所以**只在这里
        惰性建一次**，绝不在逐票循环里调。取不到返回 None（调用方负责降级）。
        """
        if self._mkt_index_cache is None:
            try:
                from core import market_factor as _mf
                _idx = _mf.equal_weight_index()
                # ⚠️ 不能写 `_idx or False`：pandas Series 的真值判断会抛
                #    "truth value of a Series is ambiguous"（实测踩过，且被
                #    外面的 except 吞成"降级"，看起来像"市场因子不可用"）。
                self._mkt_index_cache = _idx if (_idx is not None and len(_idx)) else False
                if self._mkt_index_cache is False:
                    _logger.warning("[选股] 市场因子序列不可用 ⇒ 波动率退回总波动率口径")
            except Exception as e:                  # noqa: BLE001
                _logger.warning("[选股] 市场因子加载失败（退总波动率）: %s", e)
                self._mkt_index_cache = False
        if self._mkt_index_cache is False:
            return None
        return self._mkt_index_cache      # ⚠️ 别写 `x or None`：Series 真值判断会抛

    def _ivol(self, df) -> Optional[float]:
        """20 日**残差波动率**（百分数）；拿不到可对齐的市场序列 ⇒ None。

        为什么不是 `_annualized_vol`：两者都是"越低越好"，方向一致，但扣掉市场
        因子后排序会变 —— 实测 IVOL 的 IC 比总波动率强 21%（−0.0826 vs −0.0684，
        11/11 年为负，正交化后仍 −0.0382）。见 `output/文献选股因子可实践性_2026-10-29.md`。

        ⚠️ 三条边界，缺一条就会"静默退回总波动率而看不出"：
          1. `df` 索引必须是**日期**（RangeIndex 的合成用例直接返回 None ——
             这样单测不会去读真实面板，也不会写真实缓存）；
          2. 市场序列按日期 reindex（停牌日个股侧为 NaN，成对剔除）；
          3. 不足 21 个交易日 ⇒ None。
        """
        if df is None or not hasattr(df, "columns") or "Close" not in df.columns:
            return None
        if len(df) < 21:
            return None
        try:
            import pandas as _pd
            s = _pd.to_numeric(df["Close"], errors="coerce")
            if isinstance(s.index, _pd.DatetimeIndex):
                dates = s.index.strftime("%Y-%m-%d")
            elif s.index.dtype == object:
                dates = _pd.Index(s.index.astype(str)).str.slice(0, 10)
            else:
                return None                          # 无日期轴：无从对齐
            s = s.copy()
            s.index = _pd.Index(dates)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            mkt = self._market_index()
            if mkt is None or s.empty:
                return None
            axis = mkt.index[(mkt.index >= s.index[0]) & (mkt.index <= s.index[-1])]
            if len(axis) < 21:
                return None
            from core import indicators as _ind
            v = _ind.residual_vol(s.reindex(axis).to_numpy(dtype="float64"),
                                  mkt.reindex(axis).to_numpy(dtype="float64"), 20)
            return None if v is None else float(v * 100.0)
        except Exception as e:                      # noqa: BLE001
            _logger.debug("[选股] IVOL 计算失败（退回总波动率）: %s", e)
            return None

    @staticmethod
    def _momentum20(df) -> Optional[float]:
        """20 日动量（小数，0.05 = +5%）—— 同样委托 `indicators`。"""
        if df is None or not hasattr(df, "columns") or len(df) < 21:
            return None
        try:
            from core import indicators as _ind
            return _ind.momentum(df["Close"], 20)
        except Exception:                           # noqa: BLE001
            return None

    @staticmethod
    def _percentile_map(values: dict, *, higher_is_better: bool) -> dict:
        """{code: 数值} → {code: 0~1 分位}（1 = 最好）。

        `higher_is_better=False` 表示"值越小越好"（PE、波动率都是这类）。
        并列取平均排名（避免"同值看谁在前"这种不确定性）。
        """
        clean = {c: float(v) for c, v in (values or {}).items()
                 if isinstance(v, (int, float))}
        n = len(clean)
        if n == 0:
            return {}
        ordered = sorted(clean.items(), key=lambda kv: kv[1])   # 升序：最小值 rank 0
        out = {}
        i = 0
        while i < n:
            j = i
            while j + 1 < n and ordered[j + 1][1] == ordered[i][1]:
                j += 1
            avg_rank = (i + j) / 2.0                 # 0-based 平均名次
            # 升序排名下：avg_rank=0（最小）→ base=1.0
            base = (1.0 - avg_rank / (n - 1)) if n > 1 else 1.0
            # ⚠️ 方向只在**这里**决定一次：值越小越好（PE/波动率）→ 用 base；
            #    值越大越好 → 取反。首版把 `1-pct` 套在 base 上算了两遍方向，
            #    结果"越低越好"的因子把最便宜的股票判成了最差分位
            #    （TestPercentileMap 的 5 个用例当场全红）。
            pct = base if not higher_is_better else 1.0 - base
            for k in range(i, j + 1):
                out[ordered[k][0]] = round(pct, 4)
            i = j + 1
        return out

    @staticmethod
    def _pct_to_score(pct: float, max_score: int) -> tuple:
        """分位 → 分档（保留 5 档，与绝对值口径同粒度，便于对照与解释）。

        不直接 `max*pct` 是为了可解释性：用户看到"14.55 分"无法判断含义，
        看到"候选集内前 20% → 满分"就一目了然。
        """
        if pct >= 0.8:
            return max_score, "候选集内最优 20%"
        if pct >= 0.6:
            return max_score * 3 // 4, "候选集内前 40%"
        if pct >= 0.4:
            return max_score // 2, "候选集内居中"
        if pct >= 0.2:
            return max_score // 4, "候选集内后 40%"
        return 0, "候选集内最差 20%"

    def _percentile_by_industry(self, values: dict, *, higher_is_better: bool,
                                min_group: int = 6) -> tuple:
        """{code: 数值} → ({code: 分位}, 统计) —— **行业内**分位，不足则退回全池。

        为什么要行业内（2026-10-29）
        ---------------------------
        IVOL 与行业高度相关：银行 β≈1、市场解释力强 ⇒ 残差小；成长股特异性噪声大 ⇒
        残差大。用"全候选分位"时，低波因子实际在选"哪个行业"，而不是"同行业里谁更稳"。
        离线实测（IVOL 上线当天）：前 5 名 4 只银行系 —— 这正是记录里"银行满分率 59%、
        成员长期不变"的机制，IVOL 只是把它放大了。

        规则
        ----
        1. 按 `_industry_map()`（当日快照）分行业，**组内**做分位；
        2. 组内样本 < `min_group` ⇒ 该组不单独分位，合并进"其余"池统一分位
           （6 只以下排分位等于按噪声排序；但也**不能**直接不过滤 —— 那会让缺行业
           的股票完全拿不到分位、被 `_score_single` 判 unknown 而拉低覆盖度）；
        3. 行业缺失（快照没有这只）⇒ 同样进"其余"池。

        Returns: `(pct_map, {"industry": n, "pool": n, "groups": n, "missing": n})`
        """
        clean = {c: float(v) for c, v in (values or {}).items()
                 if isinstance(v, (int, float))}
        stats = {"industry": 0, "pool": 0, "groups": 0, "missing": 0}
        if not clean:
            return {}, stats
        ind = self._industry_map() or {}
        buckets: dict[str, dict] = {}
        rest: dict[str, float] = {}
        for code, v in clean.items():
            name = ind.get(code)
            if not name:
                stats["missing"] += 1
                rest[code] = v
                continue
            buckets.setdefault(name, {})[code] = v

        out = {}
        for name, group in buckets.items():
            if len(group) < min_group:
                rest.update(group)
                continue
            out.update(self._percentile_map(group, higher_is_better=higher_is_better))
            stats["industry"] += len(group)
            stats["groups"] += 1
        if rest:
            out.update(self._percentile_map(rest, higher_is_better=higher_is_better))
            stats["pool"] = len(rest)
        return out, stats

    def _build_cross_section(self, candidates: list) -> dict:
        """为一批候选构建横截面分位上下文 {code: {pe_pct, vol_pct, ...}}。

        数据全部来自**预取缓存**（PE 来自 `_info_cache`、波动率来自 `_kline_cache`），
        因此这一步零联网、毫秒级 —— 这正是先做 P0 并发预取的意义。

        **波动率与 PE 都用行业内分位**（2026-10-29）：
        · 波动率的行业偏置已实测（IVOL 上线后前 5 名 4 只银行），必须消掉；
        · PE 的偏置更强且方向相同（银行 PE 5、科技 80，全池分位下银行恒满分）——
          只改波动率而留着 PE，等于"一只脚踩刹车"：实测改完后前 5 名仍是 3 只银行系，
          因为银行在 PE / 波动率 / 均线三个因子同时占优。
        ⚠️ 行业内分位只解决"同一把尺子量不同行业"的问题，**不解决**"低波+低 PE 的
           行业整体占优"——后者要靠行业配额（`_rank_and_select` 的 MAX_PER_SECTOR），
           属另一层机制。
        """
        pe_raw, vol_raw = {}, {}
        for c in candidates or []:
            code = str(c.get("股票代码") or "")
            if not code:
                continue
            info = self._info_cache.get(code) or {}
            pe = info.get("pe_ratio")
            if not (isinstance(pe, (int, float)) and pe > 0):
                # 预取 `skip_valuation=True` ⇒ `_info_cache` 里没有 PE（2026-10-22 起）。
                # 分位必须用**评分时真正会用的那个 PE**，否则"按 A 口径排分位、
                # 用 B 口径打分"，两者对不上而页面看不出。
                pe = self._snapshot_pe(code)
            if isinstance(pe, (int, float)) and 0 < pe <= self.PE_SANE_MAX:
                pe_raw[code] = float(pe)
            # 波动率同样交给 `_score_volatility` 的口径（IVOL 优先、总波动率兜底）——
            # 分位与打分**必须同一把尺子**，否则"分位按总波动排、打分按残差波动"，
            # 排序与分数会互相矛盾而这种矛盾看不出。
            vol = self._ivol(self._kline_cache.get(code))
            if vol is None:
                vol = self._annualized_vol(self._kline_cache.get(code))
            if vol is not None and 0 < vol <= self.VOL_SANE_MAX:
                vol_raw[code] = vol

        small = len(candidates or []) < self.MIN_FOR_PERCENTILE
        if small:
            # 样本太小：分位在几只股票里没有统计意义 → 退回绝对分档（并让 detail 说明）
            return {"by_code": {}, "pe_n": len(pe_raw), "vol_n": len(vol_raw),
                    "disabled": True, "vol_scope": "none", "pe_scope": "none",
                    "note": f"候选仅 {len(candidates or [])} 只，未启用分位化（退回绝对分档）"}

        pe_pct, pe_stats = self._percentile_by_industry(pe_raw, higher_is_better=False)
        vol_pct, vol_stats = self._percentile_by_industry(vol_raw, higher_is_better=False)
        # 结构固定为 {"by_code": {code: {...}}, "<因子>_n": 样本数, ...}：
        # `_score_single` 读 by_code，顶层放统计量（**不要把 per-code 分位摊在顶层**，
        # 否则 code 键会与 pe_n/vol_n 这类统计键混在一起，读起来靠猜）。
        by_code = {}
        for _c in set(pe_pct) | set(vol_pct):
            by_code[_c] = {"pe_pct": pe_pct.get(_c), "vol_pct": vol_pct.get(_c)}
        # 分位的作用域必须能看见：全部落进"其余池"= 行业映射没拿到，
        # 那时分位等价于旧行为 —— 而分数看起来完全正常。
        def _scope(stats: dict) -> str:
            return ("industry" if stats["industry"] and not stats["pool"]
                    else "mixed" if stats["industry"] else "pool")

        vol_scope, pe_scope = _scope(vol_stats), _scope(pe_stats)
        return {"by_code": by_code, "pe_n": len(pe_raw), "vol_n": len(vol_raw),
                "disabled": False, "vol_scope": vol_scope, "pe_scope": pe_scope,
                "vol_groups": vol_stats["groups"],
                "vol_industry_n": vol_stats["industry"], "vol_pool_n": vol_stats["pool"],
                "vol_missing_industry": vol_stats["missing"],
                "note": ""}

    def _score_pe(self, pe, max_score: int = 15, pct: float = None,
                  peers: int = 0, scope: str = None) -> dict:
        """PE 估值因子。

        口径（2026-10-10 起分位化；2026-10-29 起**行业内**分位）：
        - `pct` 给定时用横截面分位（越便宜越好），作用域由 `scope` 说明
          （industry = 同行业内、pool = 全候选池）；
        - `pct` 为 None（候选太少/无分位数据）时退回**绝对分档**，并在 detail 里说明；
        - 绝对 sanity 始终生效：PE<=0（亏损）判 unknown（这是"数据性质"不是"估值差"），
          PE>PE_SANE_MAX 判 0 分（明确极贵，不因"相对最便宜"而蒙混过关）。
        """
        if not isinstance(pe, (int, float)) or pe <= 0:
            return {"score": None, "max": max_score, "status": "unknown",
                    "detail": "无PE数据"}
        pe = float(pe)
        if pe > self.PE_SANE_MAX:
            return {"score": 0, "max": max_score, "status": "ok",
                    "detail": f"PE {pe:.1f} 超出合理上限 {self.PE_SANE_MAX:.0f}（极高）",
                    "value": {"pe": pe}}
        if pct is not None:
            score, band = self._pct_to_score(pct, max_score)
            where = {"industry": "行业内", "mixed": "行业/池内",
                     "pool": "候选集内"}.get(scope or "", "候选集内")
            if scope in ("industry", "mixed"):
                band = band.replace("候选集内", where)
            rank_txt = (f"，第 {max(1, round((1 - pct) * max(1, peers - 1)) + 1)} 低"
                        if peers > 1 else "")
            return {"score": score, "max": max_score, "status": "ok",
                    "detail": f"PE {pe:.1f}{rank_txt}（{band}）",
                    "value": {"pe": pe, "percentile": pct, "pct_scope": scope,
                              "basis": "cross_section"}}
        if pe < 10:       score, detail = max_score, f"PE {pe:.1f} (极低估值)"
        elif pe < 20:     score, detail = max_score * 3 // 4, f"PE {pe:.1f} (合理偏低)"
        elif pe < 40:     score, detail = max_score // 2, f"PE {pe:.1f} (适中)"
        elif pe < 80:     score, detail = max_score // 4, f"PE {pe:.1f} (偏高)"
        else:             score, detail = 0, f"PE {pe:.1f} (极高)"
        return {"score": score, "max": max_score, "status": "ok",
                "detail": detail + "（绝对分档）",
                "value": {"pe": pe, "basis": "absolute"}}

    # ── P3-15 A+B 灰度（2026-10-24）────────────────────────────────────
    # 依据：output/趋势否决三项补验_2026-10-24.md（1044.8 万样本、分年、流动性敏感性）
    #   A：弱市 ret20 ≥ 15% 超额 -0.96%、11/11 年负 ⇒ 不该再给"偏强"3/4 分；
    #   B：弱市 ret20 ≤ -10% 超额 +0.36%、9/11 年正 ⇒ 不该直接 0 分一票否决。
    # ⚠️ 强市**完全不动**（强市 3~10% 的证据自相冲突，按"不能单一样本调参"处理）。
    # ⚠️ 阈值唯一来源就在这里，消费方不得重抄。
    WEAK_VETO_RET20 = 0.15
    WEAK_OVERSOLD_RET20 = -0.10

    @staticmethod
    def _trend_regime_enabled() -> bool:
        """灰度开关（**运行时解析**，不写进默认参数——否则 monkeypatch 会失效）。

        `BIGA_TREND_REGIME=1/true/on/yes` 开启；缺省与任何无法识别的值都回退为**关**。
        """
        import os
        raw = os.getenv("BIGA_TREND_REGIME", "0")
        return str(raw).strip().lower() in ("1", "true", "on", "yes")

    @staticmethod
    def _weak_regime_info() -> Optional[dict]:
        """读本地指数面板判强弱；数据不足返回 None ⇒ 调用方按"不改分"处理。"""
        from core.market_regime import ma20_regime
        return ma20_regime()

    def _score_trend(self, df, max_score: int = 15, market_state: Optional[str] = None) -> dict:
        """趋势因子：20 日动量分档（v2：不再是当日涨跌幅）

        原实现用的是**当日涨跌幅**（权重键却叫 RSI、展示名又叫"趋势"），
        等价于"给当天涨得最猛的票打满分"——而 A 股单日涨超 5% 往往已接近涨停，
        不可买入，与系统其它部分的均值回归/风控口径相互矛盾。
        现改为中周期趋势，并显式惩罚过热。是否站上 MA20 只写进 detail 参与解释，
        不计分（该信息已由"均线排列"因子承担，避免重复计价）。
        """
        if df is None or not hasattr(df, "columns") or len(df) < 21:
            return {"score": None, "max": max_score, "status": "unknown",
                    "detail": "K线不足 21 根，无法计算 20 日趋势"}
        try:
            import numpy as np
            # ⚠️ 动量计算**只此一份实现**（`_momentum20`）：横截面分位构建也要用它，
            #    两处各写一份必然在"要不要复权/窗口差一"上分叉。
            ret20 = self._momentum20(df)
            if ret20 is None:
                return {"score": None, "max": max_score, "status": "unknown",
                        "detail": "20 日动量无法计算"}
            closes = df["Close"].values
            ma20 = float(np.mean(closes[-20:]))
            if ret20 > 0.30:      score, tag = max_score // 4, "过热"
            elif ret20 > 0.15:    score, tag = max_score * 3 // 4, "偏强"
            elif ret20 > 0.03:    score, tag = max_score, "稳健上行"
            elif ret20 > -0.03:   score, tag = max_score // 2, "横盘"
            elif ret20 > -0.10:   score, tag = max_score // 4, "偏弱"
            else:                 score, tag = 0, "下行"
            regime_note = ""
            if self._trend_regime_enabled():
                regime = market_state
                info = None
                if regime is None:
                    info = self._weak_regime_info()
                    # 缺数据 ⇒ regime 仍为 None ⇒ 下面不改分（缺就是缺，不默认强/弱）
                    regime = ("weak" if info["weak"] else "strong") if info else None
                if regime == "weak":
                    if ret20 >= self.WEAK_VETO_RET20:
                        score, tag = max_score // 4, "弱市追高降档"
                    elif ret20 <= self.WEAK_OVERSOLD_RET20:
                        score, tag = max_score // 2, "弱市超跌"
                    as_of = f"，指数数据日 {info['as_of']}" if info else ""
                    regime_note = f"｜弱市A+B（中证1000≤MA20{as_of}）"
            position = "站上MA20" if closes[-1] > ma20 else "位于MA20下方"
            return {"score": score, "max": max_score, "status": "ok",
                    "detail": f"20日{ret20:+.1%} {tag}，{position}{regime_note}",
                    "value": {"ret20": round(float(ret20), 4),
                              "above_ma20": bool(closes[-1] > ma20)}}
        except Exception as e:
            return {"score": None, "max": max_score, "status": "unknown",
                    "detail": f"趋势计算失败: {e}"}

    def _get_kline_data(self, code: str):
        """获取K线数据（首次拉取后缓存，后续因子复用缓存值）。

        ⭐ P0-1（2026-10-18）：**本地优先** —— 先读 `data/panel/`（实测单只 5~20ms），
        本地没有才联网（实测 ≈51 s/只）。这是"选股 4.3 分钟/5 只"的主因：
        原来每只都打一次网络。
        ⚠️ 返回列名保持**大写 + Date 索引**（下游 `_score_ma_alignment` 等读 `df["Close"]`），
        不能直接把 `get_bars` 的小写列丢过去。

        ⭐ R1（2026-10-18 审查）：**新鲜度判定必须先于窗口切片，且命中就不再联网**。
        原来的写法有两个洞：
          ① 判过期用的是 `start=today-90d` 窗口内的末根 ⇒ 长期停牌/退市股（251 只）
             窗口内 0 行 ⇒ `get_bars` 返回 None ⇒ **根本进不到过期分支**，直接联网兜底；
          ② 过期只把 `result` 留空，随后的"本地没有就联网"照样把它补齐
             ⇒ 日志说"记不可评"、实际却参与排名（`stale_skipped` 变成假记录）。
        现在：先看**全历史末根**（`bars.staleness`，不受窗口影响）；过期就到此为止，
        联网与否都一样（联网拿回的仍是同一段停牌前历史，补了也没意义）。
        """
        if code in self._kline_cache:
            return self._kline_cache[code]
        from data import bars as _bars

        # ── ① 新鲜度（全历史末根 vs 参考日；与是否离线无关）──
        try:
            st = _bars.staleness(code)
            if st.get("data_lagging"):
                reason = st.get("reason") or f"滞后 {st.get('lag_days')} 个交易日"
                self._stale_codes[code] = reason
                _logger.warning("[screener] %s 本地日线过期（%s）⇒ 不参与排名", code, reason)
                self._kline_cache[code] = None
                return None
        except Exception as e:                        # noqa: BLE001
            _logger.debug("[screener] %s 新鲜度判定失败（按不可知处理）: %s", code, e)

        # ── ② 本地面板（唯一历史行情层）──
        result = None
        try:
            from datetime import date, timedelta
            end = date.today()
            start = end - timedelta(days=90)          # 与原来 period="3mo" 同区间
            df = _bars.get_bars(code, start.isoformat(), end.isoformat(),
                                min_rows=20)
            if df is not None and not df.empty:
                result = df.rename(columns={
                    "date": "Date", "open": "Open", "close": "Close",
                    "high": "High", "low": "Low", "volume": "Volume",
                }).set_index("Date")
                result.attrs["source"] = "panel"
                result.attrs["lag_days"] = df.attrs.get("lag_days")
        except Exception as e:                        # noqa: BLE001
            _logger.debug("[screener] %s 本地K线不可用: %s", code, e)

        # ── ③ 兜底联网（本地**完全没有这只**时才走；离线模式跳过）──
        if result is None:
            if self._offline:
                _logger.info("[screener] %s 本地无日线且处于离线模式 ⇒ 记不可评", code)
            else:
                try:
                    df = self.stock_fetcher.get_stock_data(code, period="3mo")
                    if isinstance(df, dict) and "error" in df:
                        result = None
                    elif hasattr(df, 'columns') and not df.empty:
                        result = df
                    else:
                        result = None
                except Exception:                      # noqa: BLE001
                    result = None
        self._kline_cache[code] = result
        return result

    def _score_ma_alignment(self, df, max_score: int = 10) -> dict:
        """均线排列因子：MA5 > MA10 > MA20 为多头排列"""
        if df is None or not hasattr(df, "columns") or len(df) < 20:
            return {"score": None, "max": max_score, "status": "unknown",
                    "detail": "K线不足 20 根，无法计算均线排列"}
        try:
            import numpy as np
            closes = df['Close'].values
            ma5 = np.mean(closes[-5:])
            ma10 = np.mean(closes[-10:])
            ma20 = np.mean(closes[-20:])

            if ma5 > ma10 > ma20:
                score, detail = max_score, f"多头排列 (MA5={ma5:.2f}>MA10={ma10:.2f}>MA20={ma20:.2f})"
            elif ma5 > ma10 and ma10 < ma20:
                score, detail = max_score * 2 // 3, f"短期偏多 (MA5={ma5:.2f}>MA10={ma10:.2f}<MA20={ma20:.2f})"
            elif ma5 < ma10 and ma10 < ma20:
                score, detail = 0, f"空头排列 (MA5={ma5:.2f}<MA10={ma10:.2f}<MA20={ma20:.2f})"
            elif ma5 < ma10 and ma10 > ma20:
                score, detail = max_score // 3, f"短期偏空 (MA5={ma5:.2f}<MA10={ma10:.2f}>MA20={ma20:.2f})"
            else:
                score, detail = max_score // 2, f"粘合震荡 (5={ma5:.2f},10={ma10:.2f},20={ma20:.2f})"
        except Exception as e:
            return {"score": None, "max": max_score, "status": "unknown",
                    "detail": f"计算失败: {e}"}
        return {"score": score, "max": max_score, "status": "ok", "detail": detail,
                "value": {"bullish": bool(ma5 > ma10 > ma20),
                          "ma5": round(float(ma5), 3), "ma10": round(float(ma10), 3),
                          "ma20": round(float(ma20), 3)}}

    # 舆情取数窗口（天）
    NEWS_WINDOW_DAYS = 7

    def _score_news_factor(self, code: str, max_score: int = 15, name: str = None,
                           allow_fetch: bool = True) -> dict:
        """舆情因子：近 `NEWS_WINDOW_DAYS` 日个股新闻的正面率

        **两个来源**（2026-10-07 修掉"长期 0 覆盖"）：

        ① **库内快讯**（`news_sentiment_for_stock`）——快，但库内是 7x24 **宏观快讯**
           为主。实测（2026-10-07）：近 7 天 126 条快讯里，比亚迪 / 宁德时代 /
           茅台 / 兴业银行 **全部 0 条命中**——库内根本不是个股新闻源。
        ② **按个股直查**（`news_feed.fetch_stock_news`）——实测 **0.2s / 10 条**
           （贵州茅台），是唯一能"指名道姓"覆盖某只票的通路。

        改造前**只走 ①**，于是这个**权重 17.65%** 的因子对全市场候选恒为 `unknown`
        ——表现为"六因子里有一个永远缺席"，而用户完全看不出原因（不是伪装成中性，
        是连"为什么没有"都没说）。

        `allow_fetch=False` 时不打网络（只想知道"库内有没有"时用）。
        """
        stat, why = self._news_stat(code, name, allow_fetch)
        if stat is None:
            return {"score": None, "max": max_score, "status": "unknown",
                    "detail": why or f"近 {self.NEWS_WINDOW_DAYS} 日无关联新闻"}
        if stat["total"] >= 2:
            rate = stat["positive_rate"]
            return {"score": int(max_score * rate), "max": max_score, "status": "ok",
                    "detail": (f"近{self.NEWS_WINDOW_DAYS}日 {stat['total']} 条新闻"
                               f"（{stat['source']}），正面率 {rate:.0%} "
                               f"(正 {stat['positive']} / 负 {stat['negative']})"),
                    "value": {"positive_rate": round(float(rate), 4),
                              "sample": stat["total"],
                              "news_source": stat["source"]}}
        return {"score": None, "max": max_score, "status": "unknown",
                "detail": (f"近{self.NEWS_WINDOW_DAYS}日仅 1 条关联新闻"
                           f"（{stat['source']}），样本不足")}

    def _news_stat(self, code, name, allow_fetch: bool = True):
        """取个股新闻情绪统计 → (stat | None, 原因说明)

        stat = {"total","positive","negative","positive_rate","source"}

        ⚠️ 直查失败**不需要**在这里做熔断——`fetch_stock_news` 自带源级熔断
        （`SOURCE_COOLDOWN=1800s`），第一只失败后其余候选会立刻拿到"熔断中"，
        不会出现"60 只 × 12s 超时 = 十分钟"的拖死。
        """
        # ① 库内快讯（快）
        try:
            from data.news_feed import news_sentiment_for_stock
            if code or name:
                lib = news_sentiment_for_stock(code, days=3, name=name)
                if lib and lib.get("total"):
                    lib["source"] = "库内快讯"
                    return lib, None
        except Exception:                             # noqa: BLE001 - 库内失败继续走直查
            pass

        if not allow_fetch:
            return None, None

        # ② 按个股直查（慢一档但覆盖得到具体个股）
        try:
            from data.news_feed import fetch_stock_news, classify_sentiment
            hit = fetch_stock_news(code, days=self.NEWS_WINDOW_DAYS, name=name) or {}
            items = hit.get("items") or []
            if not items:
                if hit.get("error"):
                    return None, f"舆情源不可用（{hit['error']}）"
                return None, None
            pos = neg = 0
            for it in items:
                tag = classify_sentiment((it.get("title") or "")
                                         + (it.get("content") or ""))
                if tag == "positive":
                    pos += 1
                elif tag == "negative":
                    neg += 1
            total = len(items)
            return ({"total": total, "positive": pos, "negative": neg,
                     "positive_rate": round(pos / total, 2), "source": "个股直查"}, None)
        except Exception as e:                        # noqa: BLE001
            return None, f"舆情直查异常: {type(e).__name__}: {e}"

    def _bump_vol_basis(self, basis: str) -> None:
        """累计本次运行的波动率口径计数。

        ⚠️ 用 `getattr` 兜底而不是直接 `self.last_vol_basis[...]`：项目里有多处
        `StockScreener.__new__(StockScreener)`（跳过 `__init__`，避免建数据源），
        那些实例没有这个属性 —— 直接取会 AttributeError，而 `_score_volatility`
        在契约上"不该抛异常"。
        """
        d = getattr(self, "last_vol_basis", None)
        if d is None:
            d = self.last_vol_basis = {"ivol": 0, "total": 0, "none": 0}
        d[basis] = d.get(basis, 0) + 1

    def _score_volatility(self, df, max_score: int = 10, pct: float = None,
                          peers: int = 0, scope: str = None) -> dict:
        """波动率因子：越低越好。

        `scope`：分位的作用域（`industry` = 行业内、`pool` = 全候选池、`mixed` = 两者都有）。
        只影响**文案**，不影响分数 —— 但文案必须说对：用户看到"候选集内最优 20%"
        会以为是在全体候选里排的，而实际是"同行业里最稳的那个"。

        口径（2026-10-10 起分位化）：`pct` 给定时用候选集内横截面分位
        （原来绝对阈值下"低波"恒定满分 —— 银行天然低波，实测满分率 59%，
        这正是成员长期不变的成因之一）；`pct=None` 退回绝对分档并注明。
        波动率 > VOL_SANE_MAX 一律 0 分（异常标的，不参与相对排名）。

        **2026-10-29 起优先用 IVOL（残差波动率）**：优先 `_ivol`，取不到可对齐的
        市场序列时退回 `_annualized_vol`（总波动率）。两者方向相同（都越低越好），
        但 IVOL 的 IC 强 21%（−0.0826 vs −0.0684，11/11 年为负）。用了哪个口径
        写进 `value.vol_basis` 与 detail —— 不许出现"看着一样其实换了口径"。
        """
        vol = self._annualized_vol(df)
        basis = "total"
        ivol = self._ivol(df)
        if ivol is not None:
            vol, basis = ivol, "ivol"
        label = "残差波动率" if basis == "ivol" else "波动率"
        if vol is None:
            self._bump_vol_basis("none")
            return {"score": None, "max": max_score, "status": "unknown",
                    "detail": "K线不足 21 根或有效样本不足，无法计算波动率"}
        self._bump_vol_basis(basis)
        if vol > self.VOL_SANE_MAX:
            return {"score": 0, "max": max_score, "status": "ok",
                    "detail": f"{label} {vol:.1f}% 超出合理上限（异常）",
                    "value": {"volatility": round(vol, 2), "vol_basis": basis}}
        if pct is not None:
            score, band = self._pct_to_score(pct, max_score)
            # 文案随作用域走：`_pct_to_score` 说"候选集内"，行业内分位时这句是错的
            if scope == "industry":
                band = band.replace("候选集内", "行业内")
            elif scope == "mixed":
                band = band.replace("候选集内", "行业/池内")
            return {"score": score, "max": max_score, "status": "ok",
                    "detail": f"{label} {vol:.1f}%（{band}）",
                    "value": {"volatility": round(vol, 2), "percentile": pct,
                              "vol_basis": basis, "pct_scope": scope,
                              "basis": "cross_section"}}
        if vol < 20:
            score, detail = max_score, f"低波 ({vol:.1f}%)"
        elif vol < 30:
            score, detail = max_score * 3 // 4, f"中低波 ({vol:.1f}%)"
        elif vol < 40:
            score, detail = max_score // 2, f"中波 ({vol:.1f}%)"
        elif vol < 55:
            score, detail = max_score // 4, f"中高波 ({vol:.1f}%)"
        else:
            score, detail = 0, f"高波 ({vol:.1f}%)"
        return {"score": score, "max": max_score, "status": "ok",
                "detail": detail + f"（{label}·绝对分档）",
                "value": {"volatility": round(vol, 2), "vol_basis": basis,
                          "basis": "absolute"}}