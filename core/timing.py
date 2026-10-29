"""
模块名: core/timing.py
用途: 多周期共振择时引擎（C1 增强：日线+周线双周期确认）

指标来源:
  - MA 趋势: 《期货市场技术分析》John Murphy 第4章
  - MACD: 标准技术指标
  - 布林带: John Bollinger
数据源: 新浪财经 → ak.stock_zh_index_daily() (已验证可用)
更新频率: 日频（每个交易日 15:00 后）

择时流程:
  1. 获取三大指数日线+周线（上证/深成/创业板）
  2. 分别对日线和周线计算4维度指标
  3. 周线空头时降权日线得分 → 周线共振确认
  4. 三大指数得分共振判断 → 仓位系数
  5. 输出 TimingSignal
"""

import akshare as ak
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import logging

_logger = logging.getLogger(__name__)

# 指数代码映射
INDEX_MAP = {
    "上证指数": "sh000001",
    "深证成指": "sz399001",
    "创业板指": "sz399006",
}

# ── C1: 多周期权重配置 ──
DAILY_WEIGHT = 0.6          # 日线权重
WEEKLY_WEIGHT = 0.3         # 周线权重
WEEKLY_BEARISH_PENALTY = 0.7  # 周线空头时，日线得分打 7 折


@dataclass
class TimingSignal:
    """择时信号"""
    regime: str               # "bull" | "range" | "bear"
    position_limit: float     # 0.0 ~ 1.0 建议仓位比例
    total_score: float        # 0-100 综合评分
    breakdown: dict           # 各维度得分明细
    summary: str              # 中文总结
    weekly_confirmation: Optional[dict] = None  # C1: 周线确认
    source: str = "timing_engine"               # 分数来自哪个引擎（口径留痕）

    def to_dict(self) -> dict:
        result = {
            "regime": self.regime,
            "position_limit": self.position_limit,
            "total_score": self.total_score,
            "breakdown": self.breakdown,
            "summary": self.summary,
            "source": self.source,
        }
        if self.weekly_confirmation:
            result["weekly_confirmation"] = self.weekly_confirmation
        return result


# 市场状态 → regime 的映射（**唯一来源**）。
# ⚠️ 不要在这里"顺手"把偏弱震荡映射成 bear：那等于把 MarketDetector 的
#    37.9 偷偷改写成 bear，两个引擎的分数就又在语义上分叉了。
_STATE_TO_REGIME = {
    "牛市": "bull",
    "偏强震荡": "bull",
    "中性震荡": "range",
    "偏弱震荡": "range",
    "熊市": "bear",
}


def from_market_signal(ms) -> TimingSignal:
    """把 `MarketDetector` 的 `MarketSignal` 适配成 `TimingSignal`。

    为什么需要（2026-10-20 事故）
    ────────────────────────────
    深度分析图的 `timing_node` 用 `TimingEngine`，而选股/复盘/`agent_tools`
    用 `MarketDetector` —— **同一天两个结论**：

        MarketDetector  37.9 分 → 偏弱震荡，仓位上限 20%   （选股据此选票）
        TimingEngine    16.9 分 → bear，     仓位上限 10%   （分析据此全否）

    结果是"选出来的票逐个被分析否掉"，一整批白跑。统一到 `MarketDetector`
    后，同一条链路两端用同一个市场判断。

    ⚠️ 这里**只做适配，不动 MarketDetector 的权重**。它的 37.9 分里趋势只
       贡献 3 分（3/30：MA20 3/10、MACD 0/10、创业板 0/10），确实偏乐观——
       但那是"趋势否决"要解决的问题，属于评分算法调优，必须先有历史回测，
       不能顺手改（否则所有历史 `market_score` 都失去可比性）。
    """
    state = str(getattr(ms, "market_state", "") or "")
    regime = _STATE_TO_REGIME.get(state)
    if regime is None:
        # 未知状态不猜：按中性处理，并在 summary 里如实写明
        regime = "range"
    score = float(getattr(ms, "total_score", 0) or 0)
    limit = float(getattr(ms, "position_limit", 0) or 0)
    advice = str(getattr(ms, "operation_advice", "") or "")
    parts = [f"市场状态：{state or '未知'}（{score:.0f}/100），仓位上限 {limit:.0%}"]
    if advice:
        parts.append(advice)
    if regime == "range" and state not in _STATE_TO_REGIME:
        parts.append("⚠️ 未能识别 market_state，按中性处理")
    return TimingSignal(
        regime=regime, position_limit=limit, total_score=score,
        # breakdown 直接沿用 MarketDetector 的 14 项明细（含每项 max_score）
        breakdown=dict(getattr(ms, "scores", {}) or {}),
        summary="；".join(parts),
        source="market_detector",
    )


class TimingEngine:
    """多周期共振择时引擎（日线+周线双周期）"""

    def __init__(self):
        self._index_cache: dict[str, pd.DataFrame] = {}
        self._weekly_cache: dict[str, pd.DataFrame] = {}

    # ──────── 公共入口 ────────

    def detect(self, reference_date: Optional[str] = None) -> TimingSignal:
        """
        执行多周期共振判断。

        Args:
            reference_date: 参考日期 (YYYY-MM-DD)，None 表示最新交易日

        Returns:
            TimingSignal
        """
        scores = {}
        weekly_scores = {}
        details = {}
        weekly_details = {}

        for name, symbol in INDEX_MAP.items():
            df_daily = self._fetch_index(symbol)
            if df_daily is None or df_daily.empty:
                scores[name] = 50
                weekly_scores[name] = 50
                continue

            # ── 日线评分 ──
            trend_d = self._score_trend(df_daily, "daily")
            volume_d = self._score_volume(df_daily)
            momentum_d = self._score_momentum(df_daily)
            breadth_d = self._score_breadth_fallback(df_daily)
            daily_total = round(trend_d * 0.35 + volume_d * 0.25 + breadth_d * 0.15 + momentum_d * 0.25, 1)

            # ── C1: 周线评分 ──
            df_weekly = self._fetch_index_weekly(symbol)
            if df_weekly is not None and not df_weekly.empty and len(df_weekly) >= 30:
                trend_w = self._score_trend(df_weekly, "weekly")
                volume_w = self._score_volume(df_weekly)
                momentum_w = self._score_momentum(df_weekly)
                breadth_w = self._score_breadth_fallback(df_weekly)
                weekly_total = round(trend_w * 0.35 + volume_w * 0.25 + breadth_w * 0.15 + momentum_w * 0.25, 1)
            else:
                weekly_total = daily_total  # 降级：无周线数据时回退到日线

            # ── 周线确认：周线空头时降权日线得分 ──
            if weekly_total < 40:
                daily_total *= WEEKLY_BEARISH_PENALTY

            # ── 综合得分（日线+周线加权） ──
            total = round(daily_total * DAILY_WEIGHT + weekly_total * WEEKLY_WEIGHT, 1)
            scores[name] = total
            weekly_scores[name] = weekly_total
            details[name] = {
                "趋势": trend_d,
                "量能": volume_d,
                "宽度": breadth_d,
                "动量": momentum_d,
                "日线总分": daily_total,
                "周线总分": weekly_total,
                "综合": total,
            }

        # ── 三大指数共振判断 ──
        score_values = list(scores.values())
        bullish_count = sum(1 for s in score_values if s >= 60)
        bearish_count = sum(1 for s in score_values if s < 40)
        avg_score = round(sum(score_values) / len(score_values), 1)

        if bullish_count >= 2:
            regime = "bull"
            position_limit = min(0.9, 0.5 + avg_score / 200)
        elif bearish_count >= 2:
            regime = "bear"
            position_limit = max(0.1, (avg_score - 20) / 200)
        else:
            regime = "range"
            position_limit = 0.3 + (avg_score - 30) / 200

        position_limit = round(max(0.0, min(1.0, position_limit)), 2)

        # ── 周线确认信号文本 ──
        weekly_avg = round(sum(weekly_scores.values()) / len(weekly_scores), 1)
        weekly_signal = "偏多" if weekly_avg >= 60 else ("偏空" if weekly_avg < 40 else "中性")

        # ── 生成中文总结 ──
        regime_text = {"bull": "偏多", "range": "震荡", "bear": "偏空"}[regime]
        summary = (
            f"市场处于{regime_text}状态，综合评分 {avg_score:.0f}/100，"
            f"周线评分 {weekly_avg:.0f}/100({weekly_signal})，"
            f"建议仓位 {position_limit*100:.0f}%。"
            f"强势指数 {bullish_count}/3，弱势指数 {bearish_count}/3。"
        )

        return TimingSignal(
            regime=regime,
            position_limit=position_limit,
            total_score=avg_score,
            breakdown=details,
            weekly_confirmation={"avg_score": weekly_avg, "signal": weekly_signal, "scores": weekly_scores},
            summary=summary,
        )

    # ──────── 数据获取 ────────

    def _fetch_index(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取指数日线数据（新浪财经 + 缓存）"""
        if symbol in self._index_cache:
            return self._index_cache[symbol]

        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "date": "date", "open": "open", "close": "close",
                    "high": "high", "low": "low", "volume": "volume",
                })
                for col in ["open", "close", "high", "low", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.sort_values("date").dropna(subset=["close"])
                self._index_cache[symbol] = df
                return df
        except Exception as e:
            _logger.warning(f"[Timing] 获取 {symbol} 失败: {e}")

        return None

    def _fetch_index_weekly(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        C1: 生成周线数据（从日线降采样到周线）。

        保留最近 200 根周线用于评分。
        """
        if symbol in self._weekly_cache:
            return self._weekly_cache[symbol]

        df_daily = self._fetch_index(symbol)
        if df_daily is None or df_daily.empty:
            return None

        try:
            df_daily = df_daily.set_index("date")
            weekly = df_daily.resample("W").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna(subset=["close"])
            weekly = weekly.tail(200).reset_index()
            self._weekly_cache[symbol] = weekly
            return weekly
        except Exception:
            return None

    # ──────── 评分方法 ────────

    @staticmethod
    def _ema_series(values, period: int) -> np.ndarray:
        """标准 EMA 序列（递推，首值锚定），供 MACD 等需要完整序列的场景使用"""
        values = np.asarray(values, dtype=float)
        alpha = 2 / (period + 1)
        result = np.zeros_like(values)
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
        return result

    @classmethod
    def _ema(cls, values, period: int) -> float:
        """返回序列最后一个 EMA 值（标量）"""
        return float(cls._ema_series(values, period)[-1])

    def _score_trend(self, df: pd.DataFrame, mode: str = "daily") -> float:
        """
        趋势评分 (0-100)。

        mode="weekly" → 用周线参数（MA10/MA30 替代 MA20/MA60）
        """
        closes = df["close"].values
        min_len = 60 if mode == "daily" else 30
        if len(closes) < min_len:
            return 50

        current = closes[-1]
        score = 0

        # 短周期均线
        ma_short = np.mean(closes[-10:]) if mode == "daily" else np.mean(closes[-5:])   # 周线用 MA5
        ma_short_prev = np.mean(closes[-20:-10]) if len(closes) >= 20 else ma_short

        if mode == "daily":
            ma_short_period = 20
            ma_long_period = 60
        else:
            ma_short_period = 10   # 周线 MA10
            ma_long_period = 30    # 周线 MA30

        ma20 = np.mean(closes[-ma_short_period:])
        ma20_prev = np.mean(closes[-ma_short_period*2:-ma_short_period]) if len(closes) >= ma_short_period*2 else ma20

        if current > ma20 and ma20 > ma20_prev:
            score += 40
        elif current > ma20:
            score += 25
        elif current > ma20 * 0.95:
            score += 10

        # 长周期均线
        ma60 = np.mean(closes[-ma_long_period:])
        ma60_prev = np.mean(closes[-ma_long_period*2//3:-ma_long_period//3]) if len(closes) >= ma_long_period*2//3 else ma60

        if current > ma60 and ma60 > ma60_prev:
            score += 30
        elif current > ma60:
            score += 20
        elif current > ma60 * 0.95:
            score += 10

        # MACD
        ema12_seq = self._ema_series(closes, 12)
        ema26_seq = self._ema_series(closes, 26)
        dif_seq = ema12_seq - ema26_seq
        dea_seq = self._ema_series(dif_seq, 9)

        dif_val = dif_seq[-1]
        dea_val = dea_seq[-1]
        macd_bar = 2 * (dif_val - dea_val)

        if dif_val > dea_val and macd_bar > 0:
            score += 30
        elif dif_val > dea_val:
            score += 20
        elif dif_val > -0.01 * current:
            score += 10

        return min(100, score)

    def _score_volume(self, df: pd.DataFrame) -> float:
        """量能评分 (0-100)"""
        if "volume" not in df.columns or len(df) < 20:
            return 50

        volumes = df["volume"].values
        closes = df["close"].values

        vol_5 = np.mean(volumes[-5:])
        vol_20 = np.mean(volumes[-20:])

        score = 0

        ratio = vol_5 / max(vol_20, 1)
        if ratio > 1.3:
            score += 40
        elif ratio > 1.1:
            score += 30
        elif ratio > 0.9:
            score += 25
        elif ratio > 0.7:
            score += 15
        else:
            score += 5

        price_change_5d = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        if price_change_5d > 0 and ratio > 1.0:
            score += 30
        elif price_change_5d > 0:
            score += 20
        elif ratio > 1.0:
            score += 10
        else:
            score += 5

        vol_trend = sum(1 for i in range(-5, -1) if volumes[i + 1] > volumes[i])
        if vol_trend >= 3:
            score += 30
        elif vol_trend >= 2:
            score += 20
        elif vol_trend >= 1:
            score += 10

        return min(100, score)

    def _score_momentum(self, df: pd.DataFrame) -> float:
        """动量评分 (0-100)"""
        closes = df["close"].values
        if len(closes) < 20:
            return 50

        score = 0

        ret_10d = (closes[-1] - closes[-10]) / closes[-10] * 100
        if ret_10d > 5:
            score += 40
        elif ret_10d > 2:
            score += 30
        elif ret_10d > 0:
            score += 20
        elif ret_10d > -3:
            score += 10

        ma20 = np.mean(closes[-20:])
        std20 = np.std(closes[-20:])
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20

        if std20 > 0:
            position = (closes[-1] - lower) / (upper - lower)
            if 0.4 < position < 0.6:
                score += 30
            elif 0.2 < position < 0.8:
                score += 20
            elif position > 0.9:
                score += 10
            elif position < 0.1:
                score += 15
            else:
                score += 10
        else:
            score += 15

        up_days = sum(1 for i in range(-20, -1) if closes[i + 1] > closes[i])
        stability_score = min(30, (up_days * 2)) if up_days >= 10 else (5 if up_days > 0 else 0)
        score += stability_score

        return min(100, score)

    def _score_breadth_fallback(self, df: pd.DataFrame) -> float:
        """市场宽度评分 -- 降级方案（指数自身涨跌替代全市场涨跌比）"""
        closes = df["close"].values
        highs = df["high"].values if "high" in df.columns else closes
        if len(closes) < 5:
            return 50

        score = 0

        ret_5d = (closes[-1] - closes[-5]) / closes[-5] * 100
        if ret_5d > 3:
            score += 50
        elif ret_5d > 1:
            score += 35
        elif ret_5d > 0:
            score += 25
        elif ret_5d > -2:
            score += 15
        else:
            score += 5

        up_streak = sum(1 for i in range(-3, 0) if closes[i] > closes[i - 1])
        score += up_streak * 10

        high_5d = max(highs[-5:])
        new_high_count = sum(1 for h in highs[-5:-1] if h >= high_5d * 0.99)
        score += min(20, new_high_count * 5 + 5)

        return min(100, score)


# 全局单例
_timing_engine: Optional[TimingEngine] = None


def get_timing_engine() -> TimingEngine:
    """获取择时引擎单例"""
    global _timing_engine
    if _timing_engine is None:
        _timing_engine = TimingEngine()
    return _timing_engine