#!/usr/bin/env python3
"""
B1: 规则信号生成器（零 LLM 费用回测验证）+ C2/C3 策略增强

优化版：内存预加载完整日线 → 所有查询走内存切片 ~2分钟完成全量回测
（原始方案：逐日 API 查询 ~4-5小时）

C2: 行业中性化 — 评分按行业标准分 → 避免集中低波动行业
C3: 情绪增强 — ATR/MA 比值 + 涨跌停比 → 恐慌期控仓

用法:
  py scripts/backtest_rule_based.py                        # 全量回测 2024-2025
  py scripts/backtest_rule_based.py --quick                 # 快速测试（3只股票，30天）
  py scripts/backtest_rule_based.py --start 2024-01-01 --end 2024-06-30
  py scripts/backtest_rule_based.py --sweep                 # 滚动样本外参数验证（推荐）
  py scripts/backtest_rule_based.py --sweep-insample        # 全样本扫描（易过拟合，仅对照）
"""

import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from utils.logger import get_console_logger
from core import matching
from core.backtest import BacktestEngine
from data.source_manager import DataSourceManager

_log = get_console_logger("scripts.backtest")

# ================================================================
# 配置
# ================================================================

POOL_FILE = _PROJECT_ROOT / "data" / "backtest_pool.json"

# ⚠️ v3（P0-2）：下面这 15 只是**手工挑的"今天的白马"**。
# 茅台、宁德、迈瑞、平安恰恰**因为** 2024–2025 表现好才会在 2026 年进入自选名单，
# 用它们回测那段时间是**幸存者偏差 + 前视选样**的教科书案例，收益/夏普被系统性高估，
# 且偏差量级无法从回测内部察觉。**只在 data/backtest_pool.json 缺失时兜底**，
# 一旦启用会在输出里打上明显标记。
LEGACY_WATCHLIST = [
    "000001", "600036", "601318",  # 金融
    "600519", "000858", "002714",  # 消费
    "002230", "300474", "000063",  # 科技
    "300750", "002594", "601012",  # 新能源
    "600276", "300760", "000963",  # 医药
]

LEGACY_INDUSTRY_MAP = {
    "000001": "金融", "600036": "金融", "601318": "金融",
    "600519": "消费", "000858": "消费", "002714": "消费",
    "002230": "科技", "300474": "科技", "000063": "科技",
    "300750": "新能源", "002594": "新能源", "601012": "新能源",
    "600276": "医药", "300760": "医药", "000963": "医药",
}


def load_universe(path: Path = POOL_FILE) -> tuple[list, dict, str]:
    """加载回测股票池与行业映射（P0-2）。

    Returns:
        (codes, industry_map, source_desc)
        池文件不可用/过小时回退内置 15 只，并在 `source_desc` 里标明偏差。
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pool = [str(c) for c in (data.get("pool") or [])]
        imap = {str(k): str(v) for k, v in (data.get("industry_map") or {}).items()}
        if len(pool) >= 20:
            desc = f"{path.name}：{len(pool)} 只（构建于 {data.get('built_at', '未知')}）"
            return pool, imap, desc
        _log.warning("[池] %s 只有 %d 只，回退内置池", path.name, len(pool))
    except FileNotFoundError:
        _log.warning("[池] %s 不存在 → 回退内置 15 只（⚠️ 含前视选样偏差，"
                     "请先跑 scripts/build_backtest_pool.py）", path.name)
    except Exception as e:                              # noqa: BLE001
        _log.warning("[池] 读取 %s 失败(%s) → 回退内置 15 只", path.name, e)
    return list(LEGACY_WATCHLIST), dict(LEGACY_INDUSTRY_MAP), "内置 15 只（⚠️ 前视选样，不可外推）"


WATCHLIST, INDUSTRY_MAP, UNIVERSE_SOURCE = load_universe()

INDUSTRY_NEUTRALIZE = True

INITIAL_CASH = 1_000_000.0
MAX_STOCKS = 5
SCORE_THRESHOLD = 60.0
HOLDING_DAYS_MAX = 20
ATR_MULTIPLIER = 2.5
LOOKBACK_DAYS = 90
# v2（0.1.3）：单笔买入不超过当日成交量的比例（流动性约束；无成交量数据时不施加）
VOLUME_PARTICIPATION = 0.01
# v2（0.1.1）：每笔买入的基准预算比例（与 BacktestEngine 的仓位口径一致）
POSITION_BUDGET_PCT = 0.10
# v3（P1-2）：单笔预算的硬上限（占初始资金），防止"高价股至少买一手"把仓位撑爆
POSITION_BUDGET_CAP_PCT = 0.25


def _pick_exec_price(row) -> float:
    """撮合价：优先开盘价，缺失时退回收盘价（模块级，便于静态约束函数复用）"""
    for key in ("open", "close"):
        v = row.get(key)
        if v is not None and not pd.isna(v) and float(v) > 0:
            return float(v)
    return 0.0


class RuleSignalGenerator:
    """基于技术因子的规则信号生成器（内存预加载优化版 + C2/C3 增强）

    v2（M0 0.1.1）：信号与执行错开一个交易日。原实现是"用 T 日收盘价算出的信号，
    按 T 日收盘价成交"——评分本身包含了 T 日收盘（以及与之相关的 MA/动量/ATR），
    属于系统性未来函数，导致全部回测指标无效。现在改为：T 日收盘后决策 →
    T+1 开盘价成交，订单在同一循环里以挂单队列推进。

    v2（M0 0.1.3）：加入 A 股交易约束——开盘涨停不可买、开盘跌停不可卖（挂单顺延）、
    停牌日不成交、买卖量不超过当日成交量的 VOLUME_PARTICIPATION、
    持仓天数按交易日而非自然日计算。
    """

    def __init__(self):
        self.dm = DataSourceManager()
        self._data_cache: dict[str, pd.DataFrame] = {}
        self._preloaded = False
        self._bars: dict[str, list] = {}   # code → [(date_str, row)]
        self._pos: dict[str, dict] = {}    # code → {date_str: 行号}
        # 已加载的数据区间（请求侧，不是数据侧）——见 preload 的缓存判据
        self._loaded_range: tuple[str, str] | None = None
        # ── v3：参数扫描的可行性靠这三个缓存 ──
        # `score_stock` / `calc_atr` 只依赖 (code, date)，**与被扫描的参数无关**，
        # 因此 100 组参数可以复用同一份评分。没有它，换到 100+ 只股票池后
        # `--sweep` 会退化成"100 × 全量重打分"（小时级）。
        self._score_cache: dict[tuple, float] = {}
        self._atr_cache: dict[tuple, float] = {}
        # 收盘价序列（归一化后按 code 记忆化）与"确认取不到数据的标的"
        self._close_series_cache: dict[str, tuple | None] = {}
        self._no_data_codes: set[str] = set()
        # 生成过程中的剩余现金（P1-2：买入预算要按剩余现金算，不能只看名义预算）
        self._cash = INITIAL_CASH
        self._skip_stats: dict[str, int] = {}
        # 零模型（P2-3）：非 None 时走"随机选股"，用于对照
        self.random_seed: int | None = None

    # ──────── 数据预加载 ────────

    def preload(self, start_date: str, end_date: str):
        required_start = pd.to_datetime(start_date) - timedelta(days=LOOKBACK_DAYS)
        need_start = required_start.strftime("%Y-%m-%d")

        # ── 缓存判据（v3 修复）：记录的是**请求过的区间**，不是"数据里的首根 K 线" ──
        # 原判据是 `(cached_min - 5天) <= required_start`，即要求缓存里存在早于
        # 区间起点 90 天的 K 线。这个条件**在长假后必然失败**：
        # 2024-01-01 减 90 天 = 2023-10-03，而 2023 年国庆休市到 10-09，
        # 数据里最早只有 10-09 → 判"缓存不覆盖"→ **每一个参数组合都重拉全部标的**。
        # 后果：`--sweep`（100 组参数）与 `--baseline`（N 次对照）会重拉 100/N 遍数据，
        # 实测把一次运行拖到 3 小时以上仍未完成。
        # 现在按"请求区间是否已被覆盖"判断，与数据实际有没有那几天无关。
        if self._loaded_range is not None:
            lo, hi = self._loaded_range
            if lo <= need_start and hi >= end_date:
                _log.debug("缓存已覆盖 %s~%s，跳过预加载", need_start, end_date)
                return
            _log.info("缓存区间 %s~%s 未覆盖 %s~%s，重新加载", lo, hi, need_start, end_date)

        # 数据要重拉 → 基于旧数据算出来的评分/ATR/价格序列一并作废
        self._score_cache.clear()
        self._atr_cache.clear()
        self._close_series_cache.clear()
        self._no_data_codes.clear()

        _log.info(f"预加载 {len(WATCHLIST)} 只股票日线数据...")
        preload_start_str = required_start.strftime("%Y%m%d")
        end_clean = end_date.replace("-", "")
        success = 0
        for code in WATCHLIST:
            try:
                df = self.dm.get_stock_hist_data(code, start_date=preload_start_str, end_date=end_clean, adjust="qfq")
                if df is not None and not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date").reset_index(drop=True)
                    self._data_cache[code] = df
                    success += 1
                else:
                    _log.warning(f"  {code}: 无数据，跳过")
            except Exception as e:
                _log.warning(f"  {code}: 拉取失败 ({e})，跳过")
        _log.info(f"预加载完成: {success}/{len(WATCHLIST)} 只成功")
        # 记录**请求区间**（即使部分标的取数失败也算覆盖过——失败标的已在 _data_cache 里缺席，
        # 重试一次网络也未必成功，反而会把每次调用都变成一次全量重拉）
        self._loaded_range = (need_start, end_date)

    # ──────── 内存数据辅助 ────────

    def _get_slice(self, code: str, ref_date_str: str, lookback_days: int) -> pd.DataFrame:
        df = self._data_cache.get(code)
        if df is None or df.empty:
            return None
        ref_dt = pd.to_datetime(ref_date_str)
        start_dt = ref_dt - timedelta(days=lookback_days)
        mask = (df["date"] >= start_dt) & (df["date"] <= ref_dt)
        sliced = df.loc[mask]
        return sliced if not sliced.empty else None

    def _close_series(self, code: str):
        """(日期数组, 收盘价数组)，按 code 记忆化；无数据返回 None。

        v3：`_get_close_on_date` 在随机对照分支里会被调用
        **池内每一只 × 每一个交易日**（120 × 391 ≈ 4.7 万次），
        原来的实现每次都要 `pd.to_datetime` + 整表布尔过滤，纯 pandas 开销就有几十秒；
        更要命的是下面那个"缺失就联网"的兜底（见 `_get_close_on_date`）。
        这里把 K 线归一化一次，之后走二分查找。
        """
        if code in self._close_series_cache:
            return self._close_series_cache[code]
        df = self._data_cache.get(code)
        if df is None or getattr(df, "empty", True):
            series = None
        else:
            try:
                dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
                closes = df["close"].to_numpy(dtype=float)
                series = (dates, closes)
            except Exception as e:                      # noqa: BLE001
                _log.debug("[回测] %s 收盘价序列归一化失败: %s", code, e)
                series = None
        self._close_series_cache[code] = series
        return series

    def _get_close_on_date(self, code: str, date_str: str) -> float:
        """≤date_str 的最近一个收盘价；无数据返回 0。

        ⚠️ v3 修复：**缺失时不再逐次联网**。原实现在缓存未命中时调 `_dm_get_close`
        （一次网络请求），而逐日循环里这等于"每只缺失标的每天请求一次"——
        随机对照分支实测因此一轮跑了 11 分钟仍未完成（5 只缺失 × 391 天 ≈ 2000 次/轮）。
        现在：缺失只尝试一次，失败后记入 `_no_data_codes` 并跳过该标的。
        """
        series = self._close_series(code)
        if series is None:
            if code in self._no_data_codes:
                return 0.0
            price = self._dm_get_close(code, date_str)
            if price <= 0:
                self._no_data_codes.add(code)
                _log.warning("%s 取不到日线，回测中跳过该标的（不再重试网络）", code)
            return price
        dates, closes = series
        idx = int(np.searchsorted(dates, date_str, side="right")) - 1
        return float(closes[idx]) if idx >= 0 else 0.0

    def _dm_get_close(self, code: str, date_str: str) -> float:
        try:
            clean_date = date_str.replace("-", "")
            df = self.dm.get_stock_hist_data(code, start_date=clean_date, end_date=clean_date)
            if df is not None and not df.empty:
                return float(df["close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    # ──────── v2（M0 0.1.1 / 0.1.3）：T+1 撮合 + A股交易约束 ────────

    def _ensure_index(self, code: str):
        """构建按日期索引的 K 线（含前收），供 T+1 撮合与约束判断使用"""
        if code in self._pos:
            return
        df = self._data_cache.get(code)
        bars, pos = [], {}
        if df is not None and not df.empty:
            records = df.to_dict("records")
            for i, r in enumerate(records):
                ds = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
                r["prev_close"] = float(records[i - 1]["close"]) if i > 0 else 0.0
                pos[ds] = i
                bars.append((ds, r))
        self._bars[code] = bars
        self._pos[code] = pos

    def _build_index(self):
        """（重）建全部标的的索引；每次 generate_signals 开头调用，避免缓存失效"""
        self._bars, self._pos = {}, {}
        for code in WATCHLIST:
            self._ensure_index(code)

    def _bar_on(self, code: str, date_str: str):
        """取某日精确 K 线；停牌/非交易日返回 None（不得顺延到更早的价成交）"""
        self._ensure_index(code)
        i = self._pos[code].get(date_str)
        return None if i is None else self._bars[code][i][1]

    def _trading_days_held(self, code: str, entry_date: str, current_date: str) -> int:
        """持仓的交易日数。

        原实现用自然日（trade_date - entry_date).days，HOLDING_DAYS_MAX=20
        实际只有约 14 个交易日，会系统性提前平仓。
        """
        self._ensure_index(code)
        pos = self._pos[code]
        i0, i1 = pos.get(entry_date), pos.get(current_date)
        if i0 is None or i1 is None:
            return 0
        return max(0, i1 - i0)

    @staticmethod
    def _limit_pct(code: str) -> float:
        """涨跌停幅度（唯一实现见 core.matching.limit_pct）"""
        return matching.limit_pct(code)

    @staticmethod
    def _blocked_by_price_limit(direction: str, code: str, row) -> str:
        """涨跌停约束（唯一实现见 core.matching.blocked_by_price_limit）"""
        return matching.blocked_by_price_limit(
            direction, code, _pick_exec_price(row), row.get("prev_close"))

    def _max_qty_by_volume(self, row):
        """单笔上限 = 当日成交量 × VOLUME_PARTICIPATION；无成交量数据则返回 None（不约束）"""
        vol = row.get("volume")
        if vol is None or pd.isna(vol) or float(vol) <= 0:
            return None
        return int(float(vol) * VOLUME_PARTICIPATION // 100 * 100)

    def _size_buy(self, order: dict, price: float, row) -> int:
        """买入股数（v3 / P1-2）：预算按**剩余现金**动态计算。

        原实现固定 `INITIAL_CASH * POSITION_BUDGET_PCT`（10 万），且 `qty < 100` 直接返回 0：
        一手就要 13 万的标的（如 600519 一手 ≈ 15 万）在**整个回测里从未被买入**，
        而且没有任何日志——票池静默缩水。

        现在：
          1. 名义预算不足一手时，只要现金够就允许买一手（上限 `POSITION_BUDGET_CAP_PCT`）；
          2. 同时受剩余现金与当日成交量参与率约束；
          3. 买不起时记一条 skipped（**区分"资金不足"与"涨停买不到"**），
             让票池缩水变成可观测的事实。
        """
        nominal = INITIAL_CASH * POSITION_BUDGET_PCT
        budget = max(nominal, price * 100)                     # 高价股至少允许买一手
        budget = min(budget, INITIAL_CASH * POSITION_BUDGET_CAP_PCT, self._cash)
        qty = int(budget / price // 100 * 100)
        cap = self._max_qty_by_volume(row)
        if cap is not None:
            qty = min(qty, cap)
        if qty < 100:
            self._skip(order.get("code", ""), order.get("decision_date", ""),
                       "资金不足", f"需 {price * 100:.0f} 元一手，可用现金 {self._cash:.0f}")
            return 0
        return qty

    def _skip(self, code: str, date_str: str, reason: str, detail: str = ""):
        """记录一次未成交（P1-2 的可观测性要求：区分"评分不足"与"资金不足"）"""
        self._skip_stats[reason] = self._skip_stats.get(reason, 0) + 1
        _log.debug("[跳过买入] %s %s：%s %s", date_str, code, reason, detail)

    # ──────── C3: 情绪因子 ────────

    def _score_sentiment(self, code: str, ref_date: str) -> float:
        """
        情绪评分 (0-100)，因子构成:
          - ATR/MA 比值 (0-60)：高比值=高波动恐慌，低分
          - 涨跌停比 (0-40)：涨停多→高分，跌停多→低分
        """
        df = self._get_slice(code, ref_date, LOOKBACK_DAYS)
        if df is None or df.empty or len(df) < 20:
            return 50

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        # 子因子1: ATR/MA 比值 (0-60)
        tr_values = []
        for i in range(-14, 0):
            tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
            tr_values.append(tr)
        atr_14 = float(np.mean(tr_values))
        ma20_val = float(np.mean(closes[-20:]))
        atr_ratio = atr_14 / ma20_val if ma20_val > 0 else 0
        if atr_ratio < 0.02:
            atr_score = 60
        elif atr_ratio < 0.03:
            atr_score = 50
        elif atr_ratio < 0.04:
            atr_score = 35
        elif atr_ratio < 0.05:
            atr_score = 20
        elif atr_ratio < 0.06:
            atr_score = 10
        else:
            atr_score = 0

        # 子因子2: 涨跌停比 (0-40)
        # v3（P1-3）：阈值按**板块**取。原实现硬编码 0.098——创业板/科创板涨停是 20%，
        # 那里 10% 的正常波动会被当成涨停，情绪分被系统性抬高。
        limit = self._limit_pct(code)
        threshold = limit - 0.002          # 保留原实现对 10% 板块的 9.8% 容差
        ups = downs = 0
        for i in range(-10, 0):
            prev = closes[i - 1]
            if prev <= 0:
                continue
            change = (closes[i] - prev) / prev
            if change >= threshold:
                ups += 1
            elif change <= -threshold:
                downs += 1
        limit_up, limit_down = ups, downs
        total_limits = limit_up + limit_down
        if total_limits >= 4:
            ratio = limit_up / total_limits
            limit_score = round(ratio * 40)
        elif total_limits >= 2:
            ratio = limit_up / total_limits if total_limits > 0 else 0.5
            limit_score = round(ratio * 30 + 5)
        else:
            limit_score = 20

        total = atr_score + min(limit_score, 40)
        return round(min(100, total), 1)

    # ──────── 因子评分 ────────

    def score_stock(self, code: str, ref_date: str) -> float:
        """技术因子评分（带记忆化）。

        评分只依赖 (code, date)、**与被扫描的参数无关**，所以 `--sweep` 的
        100 组参数可以复用同一份结果——否则换到 100+ 只股票池后无法在可接受
        时间内完成滚动验证。缓存由 `preload()` 在重新拉数据时清空。
        """
        key = (code, ref_date)
        if key in self._score_cache:
            return self._score_cache[key]
        value = self._score_stock_impl(code, ref_date)
        self._score_cache[key] = value
        return value

    def _score_stock_impl(self, code: str, ref_date: str) -> float:
        """对单只股票进行技术因子评分（0-100）。

        ⭐ P0-2（2026-10-18）：**因子计算与打分阈值都搬到 `core/scoring.py`** ——
        本方法只负责"取数 + 情绪分"，不再内联 35/25/30/10 与十几条分档 if。
        为什么：同一件事原来有 2~3 套实现（选股 6 因子横截面分位 vs 回测 4 因子
        固定档），靠注释"刻意保持一致"维护 ⇒ **回测结论不能迁移到选股**。
        ⚠️ 行为**逐值不变**（阈值表 `MA_TIERS/VOL_TIERS/MOMENTUM_TIERS` 照抄原分档），
        由 `tests/test_scoring.py` 的 parity 测试守护。
        """
        df = self._get_slice(code, ref_date, LOOKBACK_DAYS)
        if df is None or df.empty or len(df) < 20:
            return 0.0
        from core import scoring as _sc
        factors = _sc.compute_factors(df["close"])
        sentiment_score = self._score_sentiment(code, ref_date)
        return _sc.score_fixed(factors, sentiment_score)

    # ──────── ATR 计算 ────────

    def calc_atr(self, code: str, ref_date: str, period: int = 14) -> float:
        """ATR（带记忆化，理由同 `score_stock`）"""
        key = (code, ref_date, period)
        if key in self._atr_cache:
            return self._atr_cache[key]
        value = self._calc_atr_impl(code, ref_date, period)
        self._atr_cache[key] = value
        return value

    def _calc_atr_impl(self, code: str, ref_date: str, period: int = 14) -> float:
        df = self._get_slice(code, ref_date, 60)
        if df is None or len(df) < period + 1:
            return 0.0
        high, low, close = df["high"].values, df["low"].values, df["close"].values
        tr_list = []
        for i in range(-period, 0):
            tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
            tr_list.append(tr)
        return float(np.mean(tr_list))

    # ──────── C2: 行业中性化 ────────

    def _neutralize_scores(self, candidates: list[dict]):
        """对候选评分做行业截面标准化（**只影响排序，不影响达标判定**）。

        ⚠️ 这条边界是本模块最容易被误读的地方（见 `早期文档/回测模块修改建议报告.md` P1-1）：
        达标判定 `score >= SCORE_THRESHOLD` 发生在**本函数调用之前**，用的是
        `score_stock` 的**原始分**；这里把 `score` 换成 z-score 映射值后，
        它只被用于 `candidates.sort()`。所以"普跌行情下仍满仓"**不会**发生——
        原始分不达标就进不了候选。
        为把这条边界钉死，无论是否真的做了中性化，都给每只候选留下 `score_raw`。
        """
        for c in candidates:
            c.setdefault("score_raw", c["score"])
        if not INDUSTRY_NEUTRALIZE or len(candidates) < 3:
            return

        industry_scores: dict[str, list] = {}
        for c in candidates:
            ind = INDUSTRY_MAP.get(c["code"], "其他")
            industry_scores.setdefault(ind, []).append(c)

        for ind, group in industry_scores.items():
            if len(group) < 2:
                continue
            scores = [g["score"] for g in group]
            mean_s = sum(scores) / len(scores)
            std_s = (sum((s-mean_s)**2 for s in scores) / len(scores)) ** 0.5
            if std_s < 0.01:
                continue
            for g in group:
                z = (g["score"] - mean_s) / std_s
                g["score"] = round(max(0, min(100, 50 + z * 15)), 1)

    # ──────── 主循环 ────────

    def generate_signals(self, start_date: str = "2024-01-01", end_date: str = "2025-06-30") -> list[dict]:
        self.preload(start_date, end_date)
        mode = "随机零模型" if self.random_seed is not None else "策略"
        _log.info(f"生成信号({mode}): {start_date} → {end_date}, 股票池 {len(WATCHLIST)} 只"
                  f"（{UNIVERSE_SOURCE}）")
        # 每轮重置：扫描/对照会反复调用本函数，状态必须干净
        self._cash = INITIAL_CASH
        self._skip_stats = {}

        start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
        date_range = pd.date_range(start=start_dt, end=end_dt, freq="B")
        total_days = len(date_range)

        self._build_index()
        trades, holdings = [], {}
        pending: list[dict] = []   # T 日收盘后产生、待下一交易日开盘执行的订单
        last_progress = time.time()

        for idx, trade_date in enumerate(date_range):
            date_str = trade_date.strftime("%Y-%m-%d")
            if time.time() - last_progress > 15:
                pct = (idx + 1) / total_days * 100
                _log.info(f"  进度: {date_str} ({idx+1}/{total_days}, {pct:.0f}%) | 持仓: {len(holdings)}, 交易: {len(trades)}")
                last_progress = time.time()

            # ===== (a) 开盘撮合：执行上一交易日收盘后决定的订单（T+1 成交）=====
            still_pending: list[dict] = []
            for order in pending:
                code = order["code"]
                row = self._bar_on(code, date_str)
                if row is None:
                    # 停牌/无行情：买单放弃（不追），卖单继续挂到下一交易日
                    if order["direction"] == "卖出":
                        order["deferred"] = order.get("deferred", 0) + 1
                        still_pending.append(order)
                    else:
                        self._skip(code, date_str, "停牌无行情放弃")
                    continue
                blocked = self._blocked_by_price_limit(order["direction"], code, row)
                if blocked:
                    if order["direction"] == "卖出":
                        order["deferred"] = order.get("deferred", 0) + 1
                        still_pending.append(order)
                    else:
                        # 买入遇开盘涨停：放弃，不追高（P1-2：要记下来，别让票池静默缩水）
                        self._skip(code, date_str, "开盘涨停放弃", blocked)
                    continue
                price = _pick_exec_price(row)
                if price <= 0:
                    self._skip(code, date_str, "无有效成交价")
                    continue

                if order["direction"] == "买入":
                    qty = self._size_buy(order, price, row)
                    if qty <= 0:
                        continue
                    atr = order.get("atr", 0.0)
                    self._cash -= qty * price      # P1-2：按剩余现金约束后续买入
                    trades.append({
                        "date": date_str, "code": code, "direction": "买入",
                        "price": price, "quantity": qty,
                        # v3（P1-1）：报**原始因子分**（0-100），不是中性化后的 z-score 映射值。
                        # 原实现把 `c["score"]`（中性化后）写进订单再渲染成"评分 X/100"，
                        # 既不可复现也与其他地方的口径不一致。
                        "reason": (f"评分 {order.get('score_raw', order['score']):.0f}/100"
                                   f"（{order['decision_date']} 收盘信号，T+1 开盘成交）"),
                    })
                    holdings[code] = {
                        "entry_date": date_str, "entry_price": price, "quantity": qty,
                        "stop_loss": round(price - atr * ATR_MULTIPLIER, 2) if atr > 0 else 0,
                        "pending_exit": "",
                    }
                else:
                    h = holdings.pop(code, None)
                    if h is None:
                        continue
                    self._cash += h["quantity"] * price     # 卖出回笼现金
                    reason = order.get("reason", "卖出")
                    if order.get("deferred"):
                        reason += f"（因跌停/停牌顺延 {order['deferred']} 个交易日）"
                    trades.append({
                        "date": date_str, "code": code, "direction": "卖出",
                        "price": price, "quantity": h["quantity"], "reason": reason,
                    })
            pending = still_pending

            # ===== (b) 收盘后决策（只使用 ≤ 当日的数据），次日开盘执行 =====
            # b1. 新触发的卖出信号（已在挂单队列里的不重复触发）
            for code, h in holdings.items():
                if h.get("pending_exit"):
                    continue
                held = self._trading_days_held(code, h["entry_date"], date_str)
                cp = self._get_close_on_date(code, date_str)
                if held >= HOLDING_DAYS_MAX:
                    h["pending_exit"] = f"持仓 {held} 个交易日超限"
                elif cp > 0 and cp <= h.get("stop_loss", 0):
                    h["pending_exit"] = f"ATR止损 (收盘 {cp:.2f})"
            pending_sell_codes = {o["code"] for o in pending if o["direction"] == "卖出"}
            for code, h in holdings.items():
                if h.get("pending_exit") and code not in pending_sell_codes:
                    pending.append({"code": code, "direction": "卖出",
                                    "reason": h["pending_exit"], "decision_date": date_str})

            # b2. 买入候选
            pending_buy_codes = {o["code"] for o in pending if o["direction"] == "买入"}
            slots = MAX_STOCKS - len(holdings) - len(pending_buy_codes)
            if slots <= 0:
                continue

            if self.random_seed is None:
                # ── 策略：评分过滤 + 行业中性化排序 ──
                # ⚠️ 顺序不能变：`score >= SCORE_THRESHOLD` 用**原始分**判定，
                #    中性化只决定排序（P1-1，这条边界由测试锁死）
                candidates = []
                for code in WATCHLIST:
                    if code in holdings or code in pending_buy_codes:
                        continue
                    score = self.score_stock(code, date_str)
                    if score >= SCORE_THRESHOLD:
                        ref_close = self._get_close_on_date(code, date_str)
                        if ref_close <= 0:
                            continue
                        candidates.append({"code": code, "score": score,
                                           "atr": self.calc_atr(code, date_str),
                                           "ref_close": ref_close})
                self._neutralize_scores(candidates)
                candidates.sort(key=lambda x: x["score"], reverse=True)
            else:
                # ── 零模型（P2-3）：同一套撮合/持仓/止损规则，但候选**随机选**。
                #    没有这组对照，"策略是否优于随机"就无法回答；
                #    随机选出的票不加评分门槛，这正是"零信息"的定义。
                eligible = []
                for code in WATCHLIST:
                    if code in holdings or code in pending_buy_codes:
                        continue
                    rc = self._get_close_on_date(code, date_str)
                    if rc > 0:
                        eligible.append(code)
                rng = random.Random(self.random_seed * 1_000_003 + idx)
                picks = rng.sample(eligible, min(slots, len(eligible)))
                candidates = [{"code": c, "score": 0.0, "score_raw": 0.0,
                               "atr": self.calc_atr(c, date_str)} for c in picks]

            for c in candidates[:slots]:
                pending.append({"code": c["code"], "direction": "买入", "score": c["score"],
                                "score_raw": c.get("score_raw", c["score"]),
                                "atr": c["atr"], "decision_date": date_str})

        # 期末清仓：样本末无次日行情，按最后一日收盘价平仓（仅为让回测有终值）
        final_date = date_range[-1].strftime("%Y-%m-%d")
        for code, h in list(holdings.items()):
            sp = self._get_close_on_date(code, final_date)
            if sp > 0:
                trades.append({"date": final_date, "code": code, "direction": "卖出",
                               "price": sp, "quantity": h["quantity"],
                               "reason": "期末清仓（按末日收盘价，非 T+1 成交）"})

        _log.info(f"信号生成完成: {len(trades)} 笔 (买入 {sum(1 for t in trades if t['direction']=='买入')}, 卖出 {sum(1 for t in trades if t['direction']=='卖出')})")
        # P1-2：把"没买成"的原因如实报出来，票池缩水不能是静默行为
        if self._skip_stats:
            detail = "，".join(f"{k} {v}" for k, v in sorted(self._skip_stats.items()))
            _log.info(f"未成交的买单: {sum(self._skip_stats.values())} 笔（{detail}）")
        _log.info(f"期末剩余现金: {self._cash:,.0f} 元")
        return trades


# ================================================================
# 主入口
# ================================================================

def _run_single(generator, engine, start, end, threshold, atr_mult, holding_days,
                random_seed: int | None = None):
    """跑一组参数。

    `random_seed` 非 None 时走**零模型**（随机选股），用于 P2-3 的对照实验；
    它只在随机分支被显式传入，因此既有的 7 参数调用方式保持不变。

    v4（2026-10-29，模块 A 接线）：返回值新增 `daily_returns` / `nav_dates` /
    `cost_drag` —— 参数扫描的 100 组日收益矩阵是 PBO 的输入、Sharpes 是 DSR 的
    试验列表（此前扫描只打印表格，多重检验校正无从谈起）。
    ⚠️ 旧消费者（walk_forward/baseline 的假 _run_single 测试桩）不认识新键，
    所以所有新键的读取方一律用 `.get()` 兼容。
    """
    global SCORE_THRESHOLD, ATR_MULTIPLIER, HOLDING_DAYS_MAX
    SCORE_THRESHOLD, ATR_MULTIPLIER, HOLDING_DAYS_MAX = threshold, atr_mult, holding_days
    generator.random_seed = random_seed
    trades = generator.generate_signals(start, end)
    empty = {"threshold": threshold, "atr_mult": atr_mult, "holding_days": holding_days,
             "sharpe": 0, "calmar": 0, "total_return": 0, "annual_return": 0,
             "max_drawdown": 0, "win_rate": 0, "total_trades": 0,
             "rejected_orders": 0, "avg_position_pct": 0.0,
             "daily_returns": [], "nav_dates": [], "cost_drag": 0.0}
    if not trades:
        return empty
    # v3：显式给出区间端点——逐日估值需要知道"样本到哪天为止"
    result = engine.run(trades, start_date=start, end_date=end)
    # v4：净值曲线 → 日收益（PBO 需要 T×C 对齐矩阵、DSR 需要最优组的日收益序列）。
    # 前值非正（理论不该出现）的日剔掉，"缺就是缺"不冒充 0。
    nav = result.nav_curve or []
    nav_dates: list[str] = []
    daily_returns: list[float] = []
    for i in range(1, len(nav)):
        prev, cur = nav[i - 1]["total_value"], nav[i]["total_value"]
        if prev > 0 and cur > 0:
            nav_dates.append(nav[i]["date"])
            daily_returns.append(cur / prev - 1.0)
    return {"threshold": threshold, "atr_mult": atr_mult, "holding_days": holding_days,
            "sharpe": result.sharpe_ratio, "calmar": result.calmar_ratio, "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate, "total_return": result.total_return, "annual_return": result.annual_return,
            "total_trades": result.total_trades,
            "rejected_orders": result.rejected_orders,
            "avg_position_pct": result.avg_position_pct,
            "cost_drag": result.cost_drag,
            "daily_returns": daily_returns, "nav_dates": nav_dates}


# ================================================================
# 参数验证（v2 0.1.2：默认滚动样本外验证）
# ================================================================

def _align_returns_matrix(results: list[dict]) -> np.ndarray | None:
    """把各配置的日收益按日期对齐成 T×C 矩阵（PBO 的输入）。

    对齐口径：取**所有配置都有的日期**（交集）。某配置缺某天，整行丢弃——
    零填充会让 Sharpe/PBO 口径失真，宁可缩样本。交集 <20 行时返回 None
    （样本不足以切 16 块，PBO 会拒绝，调用方标 degraded）。
    """
    series = [(r["nav_dates"], r["daily_returns"])
              for r in results if r.get("daily_returns") and r.get("nav_dates")]
    if len(series) < 2:
        return None
    common = set(series[0][0])
    for dates, _ in series[1:]:
        common &= set(dates)
    if len(common) < 20:
        return None
    common = sorted(common)
    lookup = [dict(zip(dates, rets)) for dates, rets in series]
    return np.array([[lk[d] for lk in lookup] for d in common], dtype=float)


def _multiple_testing_report(results: list[dict], s_blocks: int = 16) -> dict | None:
    """把参数扫描的全部组合喂给多重检验校正（模块 A，`core/backtest_validate`）。

    - DSR：最优组的日收益 + **全部组合的 Sharpe 列表**（试验数 N=组合数）。
    - PBO：全部组合的 T×C 日收益对齐矩阵（CSCV）。
    - 返回 None 表示样本不足（<2 组有日收益），调用方跳过而不是硬算。

    为什么放在 `_sweep`（全样本扫描）里：这一步的"最优配置"本就是样本内挑的，
    DSR/PBO 量化的正是"这个最优有多少是运气"。
    """
    from core import backtest_validate as bv

    usable = [r for r in results if r.get("daily_returns")]
    if len(usable) < 2:
        return None
    sharpes = [float(r["sharpe"]) for r in usable]
    best = max(usable, key=lambda r: r["sharpe"])
    dsr = bv.deflated_sharpe(best["daily_returns"], sr_trials=sharpes)
    report: dict = {"n_trials": len(usable), "best_config": {
        "threshold": best["threshold"], "atr_mult": best["atr_mult"],
        "holding_days": best["holding_days"]}, "dsr": dsr, "pbo": None}

    print("\n═══ 多重检验校正（模块 A）═══")
    print(f"试验数 N = {report['n_trials']}（{SWEEP_THRESHOLDS}×{SWEEP_ATR_MULTS}"
          f"×{SWEEP_HOLDING_DAYS} 的全部组合）")
    if dsr.get("ok"):
        print(f"DSR: 最优组 SR_hat(日频)={dsr['sr_hat']:.4f}, hurdle SR*={dsr['sr_star']:.4f}"
              f"（来源 {dsr['hurdle_source']}）, z={dsr['z']:.2f}, **p={dsr['p_value']:.4f}"
              f"{' ⇒ 显著' if dsr['significant'] else ' ⇒ 不显著（大概率是挑出来的运气）'}")
        if dsr.get("degraded"):
            print("  ⚠️ degraded：SR*=0，本次没有做多重检验校正")
    else:
        print(f"DSR: 无法计算（{dsr.get('reason')}）")
    matrix = _align_returns_matrix(usable)
    if matrix is not None:
        pbo_res = bv.pbo(matrix, s_blocks=s_blocks)
        report["pbo"] = pbo_res
        if pbo_res.get("ok"):
            verdict = ("⚠️ 过拟合警报" if pbo_res["warn"] else "未触发保守警戒线")
            print(f"PBO(CSCV): {pbo_res['pbo']:.3f}（{pbo_res['n_combos']} 组对分，"
                  f"S={pbo_res['s_blocks']}，T={pbo_res['t_days']}）—— {verdict}"
                  f"（警戒线 {bv.PBO_WARN:.1f}，有意比文献 0.5 严）")
        else:
            print(f"PBO: 无法计算（{pbo_res.get('reason')}）")
    else:
        print("PBO: 各配置日收益日期交集不足（<20 行），跳过")
    print("─────────────────────────────")
    return report


SWEEP_THRESHOLDS = [55, 60, 65, 70]
SWEEP_ATR_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]
SWEEP_HOLDING_DAYS = [10, 15, 20, 25, 30]
# v3（P2-1）：原为 3 —— 3 笔交易就能当选"最优参数"，统计上等于没有门槛。
# 20 笔是滚动验证里"每组至少要能算出个像样的胜率"的底线；不达标宁可不选。
SWEEP_MIN_TRADES = 20
# v3（P2-2）：原为 3 折 → 实际只产生 **2** 次样本外检验。提到 5 折（=4 次检验）。
WALK_FORWARD_FOLDS = 5


def _split_periods(start: str, end: str, n_folds: int) -> list[tuple[str, str]]:
    """把 [start, end] 按时间等分为 n_folds 个连续、不重叠的区间"""
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    span = (e - s).days
    if span < n_folds * 20:
        raise ValueError(f"区间仅 {span} 天，不足以切成 {n_folds} 折")
    bounds = []
    for i in range(n_folds):
        a = s + timedelta(days=span * i / n_folds)
        b = e if i == n_folds - 1 else s + timedelta(days=span * (i + 1) / n_folds - 1)
        bounds.append((a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")))
    return bounds


def _best_config_on(generator, engine, start, end) -> dict | None:
    """在给定区间（训练段）上挑 Sharpe 最大的参数组合"""
    best, best_sharpe = None, -999.0
    for thr in SWEEP_THRESHOLDS:
        for atr in SWEEP_ATR_MULTS:
            for hd in SWEEP_HOLDING_DAYS:
                r = _run_single(generator, engine, start, end, thr, atr, hd)
                if r["total_trades"] >= SWEEP_MIN_TRADES and r["sharpe"] > best_sharpe:
                    best_sharpe, best = r["sharpe"], r
    return best


def _walk_forward(generator, engine, start, end,
                  n_folds: int = WALK_FORWARD_FOLDS) -> list[dict]:
    """滚动样本外验证：前段选参 → 后段检验。

    原 _sweep 在**同一段全样本**上挑 Sharpe 最大的参数（只要求 total_trades>=3），
    是典型的样本内过拟合：报出来的"最优配置"没有任何外推依据。
    这里改为：第 i 折只用 [第0段..第i-1段] 选参，只在第 i 段上评估。
    """
    try:
        segs = _split_periods(start, end, n_folds)
    except ValueError as e:
        # P2-2：样本不足时**明说**，而不是悄悄退化成"单段全样本"再拿它当结论
        print(f"\n⚠️  样本量不足以做滚动验证：{e}")
        print(f"    {n_folds} 折需要区间足够长（每折至少 20 天）；请拉长 --start/--end。")
        return []
    _log.info("滚动样本外验证：%d 折，区间 %s ~ %s", n_folds, start, end)
    folds: list[dict] = []
    for i in range(1, n_folds):
        train = (segs[0][0], segs[i - 1][1])
        test = segs[i]
        _log.info("  [第%d折] 训练 %s~%s → 检验 %s~%s", i, train[0], train[1], test[0], test[1])
        best = _best_config_on(generator, engine, train[0], train[1])
        if best is None:
            _log.warning("  [第%d折] 训练段无满足最少交易数的配置，跳过", i)
            continue
        oos = _run_single(generator, engine, test[0], test[1],
                          best["threshold"], best["atr_mult"], best["holding_days"])
        folds.append({
            "fold": i, "train": list(train), "test": list(test),
            "config": {"threshold": best["threshold"], "atr_mult": best["atr_mult"],
                       "holding_days": best["holding_days"]},
            "in_sample": best, "out_sample": oos,
        })
        _log.info("  [第%d折] 最优 thr=%.0f atr=%.1f hold=%d | 样本内 sharpe=%.2f → 样本外 sharpe=%.2f",
                  i, best["threshold"], best["atr_mult"], best["holding_days"],
                  best["sharpe"], oos["sharpe"])

    if folds:
        print("\n滚动样本外验证（样本内选参 → 样本外评估）")
        print(f"{'fold':>4} | {'train':>23} | {'test':>23} | {'thr':>4} | {'atr':>4} | {'hold':>4}"
              f" | {'IS_sharpe':>9} | {'OOS_sharpe':>10} | {'OOS_ret':>8}")
        print("-" * 120)
        for f in folds:
            c, isr, oos = f["config"], f["in_sample"], f["out_sample"]
            print(f"{f['fold']:>4} | {f['train'][0]}~{f['train'][1]} | {f['test'][0]}~{f['test'][1]}"
                  f" | {c['threshold']:>4.0f} | {c['atr_mult']:>4.1f} | {c['holding_days']:>4.0f}"
                  f" | {isr['sharpe']:>9.2f} | {oos['sharpe']:>10.2f} | {oos['total_return']:>7.1f}%")
        is_sharpes = [f["in_sample"]["sharpe"] for f in folds]
        oos_sharpes = [f["out_sample"]["sharpe"] for f in folds]
        degraded = sum(1 for a, b in zip(is_sharpes, oos_sharpes) if a > 0 >= b)
        print(f"\n样本内平均 Sharpe = {sum(is_sharpes) / len(folds):.2f}；"
              f"样本外平均 Sharpe = {sum(oos_sharpes) / len(folds):.2f}")
        print(f"样本内为正、样本外转负的折数 = {degraded}/{len(folds)}"
              f"（越大说明全样本调参的过拟合越严重）")
    return folds


def _sweep(generator, engine, start, end):
    """全样本参数扫描（**样本内**，易过拟合，仅作对照）。

    请以 `--sweep`（滚动样本外验证）的结果为准。
    """
    print("\n⚠️  全样本参数扫描：最优配置是在与评估同一段数据上选出的，"
          "存在过拟合，不能作为策略有效性的证据。")
    thresholds = SWEEP_THRESHOLDS
    atr_mults = SWEEP_ATR_MULTS
    holding_days = SWEEP_HOLDING_DAYS
    total = len(thresholds) * len(atr_mults) * len(holding_days)
    _log.info(f"参数扫描: {total} 组")
    results, best_sharpe, best_config, completed = [], -999, None, 0
    t0 = time.time()
    for thr in thresholds:
        for atr in atr_mults:
            for hd in holding_days:
                r = _run_single(generator, engine, start, end, thr, atr, hd)
                results.append(r)
                completed += 1
                if r["total_trades"] >= SWEEP_MIN_TRADES and r["sharpe"] > best_sharpe:
                    best_sharpe, best_config = r["sharpe"], r
                _log.info(f"  [{completed}/{total}] thr={thr} atr={atr:.1f} hold={hd} → sharpe={r['sharpe']:.2f}")
    elapsed = time.time() - t0
    print(f"\n参数扫描完成 ({elapsed:.0f}s)")
    print(f"\n{'threshold':>10} | {'atr_mult':>8} | {'holding':>7} | {'sharpe':>6} | {'calmar':>6} | {'max_dd':>7} | {'win_rate':>8} | {'return':>8}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True)[:20]:
        print(f"{r['threshold']:>10.0f} | {r['atr_mult']:>8.1f} | {r['holding_days']:>7} | {r['sharpe']:>6.2f} | {r['calmar']:>6.2f} | {r['max_drawdown']:>6.1f}% | {r['win_rate']:>7.1f}% | {r['total_return']:>7.1f}%")
    if best_config:
        print(f"\n⚠️ 样本内最优配置（不可外推）: threshold={best_config['threshold']:.0f}, "
              f"atr_mult={best_config['atr_mult']:.1f}, holding_days={best_config['holding_days']}")

    # v4（2026-10-29，模块 A 接线）：100 组 Sharpe 不再只是表格——
    # 全部组合作为 DSR 的试验列表、日收益矩阵作为 PBO 的输入。
    report = _multiple_testing_report(results)

    # 落盘：结果可复查（学 analysis_review 的 facts 落库口径）。
    # daily_returns 矩阵一并列出（100 组 × 数百日 ≈ 几 MB，可接受）。
    try:
        out = {"built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "start": start, "end": end,
               "grid": {"thresholds": SWEEP_THRESHOLDS, "atr_mults": SWEEP_ATR_MULTS,
                        "holding_days": SWEEP_HOLDING_DAYS, "min_trades": SWEEP_MIN_TRADES},
               "multiple_testing": report,
               "results": [{k: v for k, v in r.items()} for r in results]}
        out_path = Path(engine.records_dir) / f"sweep_insample_{start}_{end}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(f"  扫描结果已保存: {out_path}")
    except Exception as e:                                  # noqa: BLE001
        _log.warning("扫描结果落盘失败（不影响回测本身）：%s", e)
    return results


# ================================================================
# 零模型对照（v3 / P2-3、P2-4）
# ================================================================

BASELINE_TRIALS_DEFAULT = 30


def _percentile(values: list, value: float) -> float:
    """value 在 values 中的百分位（0~100）"""
    if not values:
        return float("nan")
    return round(sum(1 for x in values if x <= value) / len(values) * 100, 1)


def _random_baseline(generator, engine, start, end, threshold, atr_mult, holding_days,
                     trials: int = BASELINE_TRIALS_DEFAULT,
                     strategy: dict | None = None, seed0: int = 20261006) -> dict:
    """随机选股零模型：同一套撮合/持仓/止损规则，但候选**随机选**（P2-3）。

    为什么必须有它——两件事孤立地看都会被误读：

      1. 一个 Sharpe = 0.8 说明不了任何问题，除非知道**随机选股的分布**长什么样；
      2. 100 组参数里挑 Sharpe 最大的那组，本身就带选择偏差（P2-4）。
         所以对照要用**同一套流程去跑随机策略**，把策略值放进随机分布里定位。
         随机策略的"最优值"才是判断策略最优值是否有意义的基线。

    对照是**同规则、只换选股来源**：相同的 MAX_STOCKS、T+1 撮合、ATR 止损、
    持仓天数上限、成交量参与率与涨跌停约束。唯一区别是候选不经过评分门槛。
    """
    import statistics

    if strategy is None:
        strategy = _run_single(generator, engine, start, end, threshold, atr_mult, holding_days)

    sharpes, returns, drawdowns, trade_counts = [], [], [], []
    t0 = time.time()
    for i in range(trials):
        r = _run_single(generator, engine, start, end, threshold, atr_mult, holding_days,
                        random_seed=seed0 + i)
        sharpes.append(r["sharpe"])
        returns.append(r["total_return"])
        drawdowns.append(r["max_drawdown"])
        trade_counts.append(r["total_trades"])
        if (i + 1) % 10 == 0:
            _log.info("  随机对照进度 %d/%d", i + 1, trials)

    def _mean(xs):
        return round(statistics.mean(xs), 2) if xs else 0.0

    def _std(xs):
        return round(statistics.pstdev(xs), 2) if len(xs) > 1 else 0.0

    summary = {
        "trials": trials,
        "strategy": {"sharpe": strategy["sharpe"], "total_return": strategy["total_return"],
                     "max_drawdown": strategy["max_drawdown"], "total_trades": strategy["total_trades"]},
        "random_sharpe_mean": _mean(sharpes),
        "random_sharpe_std": _std(sharpes),
        "random_sharpe_best": max(sharpes) if sharpes else 0.0,
        "random_return_mean": _mean(returns),
        "random_drawdown_mean": _mean(drawdowns),
        "random_trades_mean": _mean(trade_counts),
        "strategy_sharpe_percentile": _percentile(sharpes, strategy["sharpe"]),
        "strategy_return_percentile": _percentile(returns, strategy["total_return"]),
    }

    print(f"\n零模型对照（{trials} 次随机选股，耗时 {time.time() - t0:.0f}s）")
    print("-" * 74)
    print(f"{'':<16}{'Sharpe':>10}{'区间收益':>12}{'最大回撤':>12}{'交易笔数':>10}")
    print(f"{'随机均值':<16}{summary['random_sharpe_mean']:>10.2f}"
          f"{summary['random_return_mean']:>11.1f}%{summary['random_drawdown_mean']:>11.1f}%"
          f"{summary['random_trades_mean']:>10.1f}")
    print(f"{'随机最好的一只':<16}{summary['random_sharpe_best']:>10.2f}")
    print(f"{'策略':<16}{strategy['sharpe']:>10.2f}{strategy['total_return']:>11.1f}%"
          f"{strategy['max_drawdown']:>11.1f}%{strategy['total_trades']:>10.0f}")
    print("-" * 74)
    print(f"策略 Sharpe 位于随机分布的 **{summary['strategy_sharpe_percentile']:.0f} 分位**"
          f"（{trials} 次随机中最好的一只为 {summary['random_sharpe_best']:.2f}）")
    print("若分位不高、或随机策略的最好一只就超过策略，"
          "则**没有证据表明策略优于随机选股**——这才是可以写进结论的表述。")
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="规则信号回测生成器")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-06-30")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--sweep", action="store_true",
                        help="滚动样本外参数验证（推荐）")
    parser.add_argument("--sweep-insample", action="store_true",
                        help="全样本参数扫描（样本内，易过拟合，仅供对照）")
    parser.add_argument("--baseline", type=int, default=0, metavar="N",
                        help="零模型对照：额外跑 N 次随机选股，报告策略所处分位（P2-3）")
    args = parser.parse_args()
    if args.quick:
        global WATCHLIST, HOLDING_DAYS_MAX
        WATCHLIST = WATCHLIST[:3]
        HOLDING_DAYS_MAX = 10
        args.start, args.end = "2024-01-02", "2024-02-10"
        _log.info(f"快速模式: {len(WATCHLIST)} 只股票, {args.start} → {args.end}")
    t0 = time.time()
    generator = RuleSignalGenerator()
    engine = BacktestEngine()
    print(f"\n股票池: {len(WATCHLIST)} 只 —— {UNIVERSE_SOURCE}")
    print(f"时间区间: {args.start} → {args.end}")
    if args.sweep_insample:
        _sweep(generator, engine, args.start, args.end)
        return
    if args.sweep:
        # v2（0.1.2）：默认做滚动样本外验证，而不是在全样本上挑最大 Sharpe
        _walk_forward(generator, engine, args.start, args.end)
        return
    trades = generator.generate_signals(args.start, args.end)
    if not trades:
        _log.error("未生成任何交易信号")
        return
    engine.inject_kline_cache(generator._data_cache)
    # 接入基准收益（沪深300），用于信息比率计算与看板对比；获取失败则保持 IR=0 的旧行为
    try:
        from data.benchmark import get_benchmark_nav
        bench_nav = get_benchmark_nav(args.start, args.end)
        if bench_nav:
            engine.set_benchmark(bench_nav)
            _log.info("基准已接入: 沪深300 %s~%s (%d个交易日)", args.start, args.end, len(bench_nav))
        else:
            _log.warning("基准数据不可用，information_ratio 将为 0")
    except Exception as bench_err:
        _log.warning("基准接入失败(%s)，information_ratio 将为 0", bench_err)
    # v3：显式给出区间端点，保证逐日估值覆盖整个样本（而不是停在最后一笔成交日）
    result = engine.run(trades, start_date=args.start, end_date=args.end)
    elapsed = time.time() - t0
    print()
    print(engine.format_report(result))
    print(f"\n  总耗时: {elapsed:.0f} 秒")
    print(f"  交易笔数: {len(trades)}")
    print(f"  股票池规模: {len(WATCHLIST)} 只（{UNIVERSE_SOURCE}）")
    sample_codes = set(t["code"] for t in trades if t.get("direction") == "买入")
    if sample_codes:
        industries = set(INDUSTRY_MAP.get(c, "其他") for c in sample_codes)
        print(f"  买入行业分布: {len(industries)} 个 / 池内 {len(set(INDUSTRY_MAP.values()))} 个"
              f" ({', '.join(sorted(industries))})")
    # P1-5：半仓口径必须显式声明，否则"跑输基准"会被误读成策略问题
    print(f"  ⚠️ 口径说明: 单笔预算 {POSITION_BUDGET_PCT:.0%} × 最多 {MAX_STOCKS} 只 "
          f"= 仓位上限 {POSITION_BUDGET_PCT * MAX_STOCKS:.0%}（半仓），"
          f"实际平均仓位 {result.avg_position_pct:.1f}%；与满仓指数对比时超额收益会被压低")
    if result.rejected_orders:
        print(f"  ⚠️ 未成交买单: {result.rejected_orders} 笔（现金不足，已记入交易明细）")

    baseline = None
    if args.baseline > 0:
        baseline = _random_baseline(
            generator, engine, args.start, args.end,
            args.threshold, ATR_MULTIPLIER, HOLDING_DAYS_MAX,
            trials=args.baseline,
            strategy={"sharpe": result.sharpe_ratio, "total_return": result.total_return,
                      "max_drawdown": result.max_drawdown, "total_trades": result.total_trades})

    report_path = engine.save_report(result)
    print(f"  报告已保存: {report_path}")
    if baseline:
        out_json = report_path.replace(".json", "_baseline.json")
        import json as _json
        Path(out_json).write_text(_json.dumps(baseline, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"  零模型对照已保存: {out_json}")


if __name__ == "__main__":
    main()