"""
模块名: core/risk.py
用途: 三层风控引擎

指标来源:
  - 凯利公式: J.L. Kelly Jr., "A New Interpretation of Information Rate", 1956
  - ATR 止损: 《海龟交易法则》Curtis Faith 第5章
  - 风险平价: 桥水基金 All Weather 策略衍生

风控层次:
  1. 组合层: 总仓位上限、行业集中度、最大回撤熔断
  2. 标底层: 凯利公式仓位分配、ATR 动态止损/追踪止损
  3. 执行层: 单票上限、流动性检查、盈亏比验证
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PositionPlan:
    """单只股票的仓位计划"""
    code: str
    weight: float           # 建议仓位权重 (0.0~1.0)
    stop_loss: float        # 止损价
    take_profit: float      # 止盈价
    entry_price: float      # 入场参考价
    reason: str = ""
    risk_score: float = 0   # 风险评分 (0-100，越低越好)


@dataclass
class RiskReport:
    """风控报告"""
    overall_risk: str = "medium"     # "low" | "medium" | "high"
    position_plans: list[PositionPlan] = field(default_factory=list)
    sector_exposure: dict = field(default_factory=dict)
    max_portfolio_drawdown: float = 0.0
    max_position_weight: float = 0.25
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_risk": self.overall_risk,
            "position_plans": [{
                "code": p.code,
                "weight": p.weight,
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit,
                "entry_price": p.entry_price,
                "reason": p.reason,
                "risk_score": p.risk_score,
            } for p in self.position_plans],
            "sector_exposure": self.sector_exposure,
            "max_portfolio_drawdown": self.max_portfolio_drawdown,
            "max_position_weight": self.max_position_weight,
            "warnings": self.warnings,
        }


class RiskEngine:
    """三层风控引擎"""

    # ── 默认参数 ──
    MAX_SINGLE_STOCK_WEIGHT = 0.25   # 单只股票最大仓位
    MAX_SECTOR_WEIGHT = 0.30         # 单行业最大仓位
    MAX_DRAWDOWN_LIMIT = 0.15        # 组合最大回撤熔断线
    DEFAULT_RISK_PER_TRADE = 0.02    # 每笔交易最大风险 2%

    # ── 流动性（2026-10-17 追加，P2-5）──
    # 口径：**持仓市值 ÷ 该股当日成交额**（单位 = 天）。
    #   含义是"你手上这些货，相当于该股几天的全部成交额"。
    #   ≥1.0 天 ⇒ 想清仓，市场要花掉一整个交易日的全部成交额才吃得下，
    #   实际冲击成本会非常显著（价格会被自己推下去）；≥0.2 天（即占日成交额 20%）
    #   已经需要分笔、分批挂单，否则会明显打低成交价。
    # ⚠️ 这两个数**只被 `core/position_risk.py`（持仓体检）消费**，
    #    刻意**不**写进 `core/agents.risk_rules_block()`：那块是给 LLM 的风控规则，
    #    而深度分析的 state 里**没有成交额字段**，把规则给它只会诱导模型编造
    #    "流动性不足"的依据（`RISK_NO_INVENTED_RULES` 明确禁止引用未列出的规则）。
    MAX_LIQUIDITY_DAYS = 1.0         # 持仓市值 ≥ 1 个交易日的成交额 ⇒ 高
    LIQUIDITY_CAUTION_DAYS = 0.2     # ≥ 20% 日成交额 ⇒ 需分笔（中）
    # ⚠️ 这个常量目前**只作为"给 LLM 的风控规则"输入**（`core/agents.risk_rules_block()`），
    #    本类内部没有任何地方用它做校验或仓位计算（`_kelly_position` 用的是凯利公式）。
    #    所以它的**口径**必须写在 `risk_rules_block()` 里 ——
    #    实测（2026-10-08/09）同一个计划、同一个 2%，被读成"止损幅度 ≤2%"和
    #    "仓位×止损幅度 ≤2%"两种口径，两次结论正好相反（需修改 / 批准）。
    ATR_MULTIPLIER = {
        "default": 2.5,
        "科技": 3.0,
        "金融": 2.0,
        "消费": 2.0,
        "新能源": 3.0,
        "医药": 2.5,
    }

    def __init__(self):
        self._kline_cache: dict[str, pd.DataFrame] = {}
        self._stock_fetcher = None

    @property
    def stock_fetcher(self):
        if self._stock_fetcher is None:
            from data.stock_data import StockDataFetcher
            self._stock_fetcher = StockDataFetcher()
        return self._stock_fetcher

    # ════════════════════════════════════════════
    # 公共入口
    # ════════════════════════════════════════════

    def assess(
        self,
        candidates: list[dict],
        timing_signal,
        current_portfolio: Optional[dict] = None,
    ) -> RiskReport:
        """
        执行三层风控。

        Args:
            candidates: 选股结果 [{"股票代码": "000001", "score": 82.5, ...}, ...]
            timing_signal: 择时信号 (有 .position_limit 属性)
            current_portfolio: 当前持仓信息 (可选)

        Returns:
            RiskReport
        """
        report = RiskReport()
        warnings = []

        # ── 层1: 组合层 ──
        total_position_limit = getattr(timing_signal, "position_limit", 0.5)

        if total_position_limit <= 0.1:
            report.overall_risk = "high"
            report.warnings = ["择时信号为空仓，风控拒绝所有买入"]
            return report

        # 熊市熔断
        if getattr(timing_signal, "regime", "") == "bear":
            total_position_limit = min(total_position_limit, 0.3)
            warnings.append("熊市预警：仓位上限压缩至 30%")

        # ── 层2: 标底层 ──
        plans = []
        sector_weights: dict[str, float] = {}

        for rank, stock in enumerate(candidates[:10]):  # 最多取前10只
            code = stock.get("股票代码", "")
            score = stock.get("score", 50)
            sector = stock.get("sector", "default")
            name = stock.get("股票名称", code)

            # 获取K线计算 ATR
            kline = self._get_kline(code)
            entry_price = self._get_current_price(code)
            if entry_price <= 0:
                continue

            # ATR 止损
            atr = self._calc_atr(kline) if kline is not None else entry_price * 0.05
            multiplier = self.ATR_MULTIPLIER.get(sector, self.ATR_MULTIPLIER["default"])
            stop_loss = round(entry_price - atr * multiplier, 2)
            take_profit = round(entry_price + atr * multiplier * 2, 2)  # 盈亏比 1:2

            # ── 修复 P1-3 + P2-5：等权分配（凯利仅用于总仓位上限），
            #     个股风险评分仅用于排序，权重由层3统一分配 ──
            risk_score = round(100 - score + rank * 5, 1)

            plan = PositionPlan(
                code=code,
                weight=0.0,  # 层3统一分配，此处占位
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_price=entry_price,
                reason=f"综合评分 {score}，ATR 止损 {stop_loss}",
                risk_score=risk_score,
            )
            plans.append(plan)

        # ── 层3: 执行层约束 ──
        # 按风险评分排序（越低越好），取前5只
        plans.sort(key=lambda p: p.risk_score)
        plans = plans[:5]

        if not plans:
            report.overall_risk = "high"
            report.warnings = warnings + ["无有效候选标的"]
            return report

        # ── 修复 P1-3：等权分配 ──
        # 每只股票等权：总仓位 / 股票数量
        equal_weight = round(total_position_limit / len(plans), 4)
        for p in plans:
            p.weight = equal_weight

        # 单票上限硬约束
        for p in plans:
            if p.weight > self.MAX_SINGLE_STOCK_WEIGHT:
                warnings.append(f"{p.code} 单票仓位 {p.weight*100:.1f}% 超过上限 {self.MAX_SINGLE_STOCK_WEIGHT*100:.0f}%，已截断")
                p.weight = self.MAX_SINGLE_STOCK_WEIGHT

        # ── 汇总风控报告 ──
        if total_position_limit >= 0.7:
            report.overall_risk = "low"
        elif total_position_limit >= 0.3:
            report.overall_risk = "medium"
        else:
            report.overall_risk = "high"

        report.position_plans = plans
        report.sector_exposure = sector_weights
        report.max_position_weight = self.MAX_SINGLE_STOCK_WEIGHT
        report.warnings = warnings

        return report

    # ════════════════════════════════════════════
    # 单只标的：开仓时的止损/止盈参考价
    # ════════════════════════════════════════════

    def suggest_levels(
        self,
        code: str,
        entry_price: float = 0.0,
        sector: str = "",
        current_price: float = 0.0,
    ) -> dict:
        """为**单只标的**给出 ATR 止损/止盈参考价（持仓录入时用）。

        与 `assess()` 的分工：`assess()` 面向"一批选股候选"，还要处理组合层仓位、
        行业集中度等约束；本方法只回答一个问题——**这一只、这个价位，止损止盈放哪儿**。

        两者共用同一套 ATR 口径与 `ATR_MULTIPLIER` 行业倍数。这一点很重要：
        如果持仓页另写一套（比如固定 -8%），就会出现"分析链说止损 10.2、持仓页说 9.5"
        两套说法，用户不知道该信哪个。

        Args:
            code: 股票代码
            entry_price: 计划/实际成交价；≤0 时退回最新价
            sector: 行业，决定 ATR 倍数（科技/新能源 3.0、金融/消费 2.0、医药 2.5、默认 2.5）
            current_price: 现价（用于算浮盈与追踪止损）；≤0 时自动取

        Returns:
            成功: {code, entry_price, atr, atr_pct, multiplier, stop_loss,
                   take_profit, risk_per_share, risk_pct, risk_reward_ratio,
                   trailing_stop, trailing_note}
            **取不到数据时返回 {"code":..., "unavailable": True, "note": "原因"}**，
            绝不用固定百分比硬凑一个看起来合理的数字。
        """
        entry = float(entry_price or 0)
        cur = float(current_price or 0)
        if cur <= 0:
            cur = self._get_current_price(code)
        if entry <= 0:
            entry = cur
        if entry <= 0:
            return {"code": code, "unavailable": True,
                    "note": "取不到价格（未填成交价且最新价不可用），无法计算止损止盈"}

        kline = self._get_kline(code)
        if kline is None or len(kline) < 15:
            n = 0 if kline is None else len(kline)
            return {"code": code, "unavailable": True,
                    "note": f"K线仅 {n} 根（<15），ATR 不可靠，未给建议；请手动设定或稍后重试"}

        atr = self._calc_atr(kline)
        try:
            atr = float(atr)
        except (TypeError, ValueError):
            atr = 0.0
        if not np.isfinite(atr) or atr <= 0:
            return {"code": code, "unavailable": True,
                    "note": "ATR 计算失败（数据异常），未给建议"}

        mult = self.ATR_MULTIPLIER.get(sector or "", self.ATR_MULTIPLIER["default"])
        stop_loss = round(max(0.01, entry - atr * mult), 2)
        take_profit = round(entry + atr * mult * 2, 2)      # 盈亏比 1:2
        risk_per_share = round(entry - stop_loss, 2)

        out = {
            "code": code,
            "entry_price": round(entry, 2),
            "current_price": round(cur, 2) if cur > 0 else None,
            "atr": round(atr, 4),
            "atr_pct": round(atr / entry * 100, 2),
            "multiplier": mult,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_per_share": risk_per_share,
            "risk_pct": round(risk_per_share / entry * 100, 2),
            "risk_reward_ratio": 2.0,
        }

        # 追踪止损：只在已有浮盈时才有意义
        if cur > 0:
            trail = self.calc_trailing_stop(entry, cur, atr, mult)
            out["trailing_stop"] = round(max(0.01, trail), 2)
            profit = cur - entry
            if profit > atr * 3:
                out["trailing_note"] = "浮盈超 3×ATR，止损已上移至成本价 +1×ATR（锁住部分利润）"
            elif profit > atr * 2:
                out["trailing_note"] = "浮盈超 2×ATR，止损可上移至成本价（保本）"
            else:
                out["trailing_note"] = "浮盈未达 2×ATR，止损暂不上移"
            # 明确提示"现价已经越过建议位"的两种情形，否则用户会以为建议无效
            if cur <= stop_loss:
                out["warning"] = f"现价 {cur:.2f} 已低于建议止损 {stop_loss}，若照此执行即为立即止损"
            elif cur >= take_profit:
                out["warning"] = f"现价 {cur:.2f} 已高于建议止盈 {take_profit}，照此执行即为立即止盈"

        return out

    # ════════════════════════════════════════════
    # 子方法
    # ════════════════════════════════════════════

    @staticmethod
    def _kelly_position(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        凯利公式: f = (bp - q) / b

        Args:
            win_rate: 胜率 (0.0~1.0)
            avg_win: 平均盈利幅度
            avg_loss: 平均亏损幅度

        Returns:
            建议仓位比例 (0.0~0.25)
        """
        if avg_loss == 0:
            return 0.15
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - p
        f = (b * p - q) / b
        return max(0.0, min(f, 0.25))

    def _get_kline(self, code: str) -> Optional[pd.DataFrame]:
        """获取K线数据（缓存）"""
        if code in self._kline_cache:
            return self._kline_cache[code]
        try:
            df = self.stock_fetcher.get_stock_data(code, period="3mo")
            if isinstance(df, pd.DataFrame) and not df.empty:
                self._kline_cache[code] = df
                return df
        except Exception:
            pass
        return None

    def _get_current_price(self, code: str) -> float:
        """获取当前价格"""
        try:
            info = self.stock_fetcher.get_stock_info(code)
            price = info.get("current_price", 0)
            if isinstance(price, (int, float)) and price > 0:
                return float(price)
        except Exception:
            pass
        # 从K线获取最新收盘价
        kline = self._get_kline(code)
        if kline is not None and "Close" in kline.columns:
            return float(kline["Close"].iloc[-1])
        return 0.0

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
        """
        计算 ATR (Average True Range)。

        Args:
            df: K线 DataFrame，需含 High/Low/Close 列
            period: ATR 周期，默认 14

        Returns:
            ATR 值
        """
        try:
            high = df["High"].values
            low = df["Low"].values
            close = df["Close"].values

            tr = np.zeros(len(df))
            for i in range(1, len(df)):
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]),
                )

            # 简单移动平均
            atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
            return float(atr)
        except Exception:
            return 0.0

    @staticmethod
    def calc_trailing_stop(
        entry_price: float,
        current_price: float,
        atr: float,
        multiplier: float = 2.5,
    ) -> float:
        """
        计算追踪止损。

        规则:
          - 浮盈 > 2x ATR → 止损上移至成本价
          - 浮盈 > 3x ATR → 止损上移至 entry + 1x ATR

        Args:
            entry_price: 入场价
            current_price: 当前价
            atr: ATR 值
            multiplier: ATR 倍数

        Returns:
            当前止损价
        """
        profit = current_price - entry_price
        base_stop = entry_price - atr * multiplier

        if profit > atr * 3:
            return entry_price + atr * 1.0
        elif profit > atr * 2:
            return entry_price
        else:
            return base_stop


# 全局单例
_risk_engine: Optional[RiskEngine] = None


def get_risk_engine() -> RiskEngine:
    """获取风控引擎单例"""
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = RiskEngine()
    return _risk_engine