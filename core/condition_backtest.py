"""条件回测：把"这套条件"变成可历史重放的信号

⚠️ 回测什么？——这是最容易做错的地方
------------------------------------
（见 `output/自然语言选股与回测工作流_改造分析.md` §3.1）

    回测"这次选出的 5 只票" → 5 只 × 1 个时点，**几乎没有统计意义**
    回测"这套条件"          → 历史每天重新选股 → 数百笔交易，能算夏普/回撤/随机对照

本模块只做后者。否则就是拿 5 只票的收益冒充"回测"，而看报告的人无从分辨。

指标可回放性（实测 2026-10-07）
------------------------------
✅ **可回放（7 个）**
    ma_bullish / ma20_above / momentum20 / volatility  ← 日线即可
    change_pct / amount / turnover                     ← 新浪日线自带
❌ **回放不了（5 个）**
    pe / market_cap      ← 历史估值无稳定免费源（百度估值可补，但需逐票额外请求）
    fund_net             ← 历史资金流无稳定免费源
    news_positive        ← 历史舆情无稳定免费源
    score                ← 评分链自身的产物；回放它会形成循环依赖
⇒ 按 D3 决策**显式列出但不参与**：报告写明"本次回测条件集为 N 条，
   另有 M 条无法历史验证"——绝不假装那 M 条参与了。

前视防护（最容易出错、也最难自查的地方）
--------------------------------------
任何一天 T 的条件判定，**只能**用 `date <= T` 的数据；成交在 **T+1 开盘**。
统一由 `_slice()` 裁剪，并由 `tests/test_condition_backtest.py` 用"构造未来数据"
的方式验证——如果哪天有人图省事把全量传进去，用例会立刻变红。
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from core import nl_screener as nl

_logger = logging.getLogger(__name__)

# 可历史回放的指标（判据：只需"当日及之前"的日线数据就能算出来）
REPLAYABLE_FIELDS = frozenset({
    "ma_bullish", "ma20_above", "momentum20", "volatility",   # K 线
    "change_pct", "amount", "turnover",                        # 新浪日线自带
})

# 明确不可回放的指标 + 原因（写进报告，让读者知道条件集的完整构成）
NOT_REPLAYABLE = {
    "pe": "历史估值无稳定免费源（当期 PE 不能冒充历史 PE）",
    "market_cap": "同上（需历史股本与收盘价，当期市值不能倒推）",
    "fund_net": "历史资金流无稳定免费源",
    "news_positive": "历史舆情无稳定免费源",
    "score": "评分链自身的产物，回放会形成循环依赖",
}

DEFAULT_INITIAL_CASH = 1_000_000.0


@dataclass
class ReplayPlan:
    """条件集的可回放性方案"""
    replayable: list = field(default_factory=list)   # list[Condition]
    skipped: list = field(default_factory=list)      # list[(label, reason)]
    sector: str = None
    exclude_st: bool = True

    @property
    def ok(self) -> bool:
        """至少要有一条可回放条件——否则"回测"没有任何条件可言"""
        return bool(self.replayable)

    def to_dict(self) -> dict:
        return {
            "replayable": [c.scaled() for c in self.replayable],
            "replayable_count": len(self.replayable),
            "skipped": [{"条件": label, "原因": why} for label, why in self.skipped],
            "skipped_count": len(self.skipped),
            "total_conditions": len(self.replayable) + len(self.skipped),
            "sector": self.sector,
            "exclude_st": self.exclude_st,
        }


def plan(query) -> ReplayPlan:
    """把条件集拆成"可回放"与"不可回放"两部分（不改变条件本身）"""
    p = ReplayPlan(sector=query.sector, exclude_st=query.exclude_st)
    for cond in query.conditions:
        if cond.field in REPLAYABLE_FIELDS:
            p.replayable.append(cond)
        else:
            p.skipped.append((cond.scaled(),
                              NOT_REPLAYABLE.get(cond.field, "该指标不支持历史回放")))
    return p


class ConditionSignalGenerator:
    """把"条件集"逐日重放成可回测的交易信号。

    结构上刻意与 `scripts/backtest_rule_based.RuleSignalGenerator` 保持一致
    （预加载 → 逐日扫描 → T+1 开盘成交），因为那套流程已经踩平了几个坑：
    区间记忆化、缺失标的不重复联网、按板块取涨跌停。
    """

    def __init__(self, codes, query, holding_days: int = 20,
                 max_positions: int = 5, stop_loss_pct: float = 8.0,
                 initial_cash: float = DEFAULT_INITIAL_CASH):
        self.plan = plan(query)
        self.query = query
        self.codes = [str(c) for c in (codes or [])]
        self.holding_days = int(holding_days)
        self.max_positions = int(max_positions)
        self.stop_loss_pct = float(stop_loss_pct)
        self.initial_cash = float(initial_cash)

        # 只含可回放条件的子 query —— 核对逻辑**复用** `nl_screener.evaluate`，
        # 保证"回测的判定"与"选股的判定"是同一套实现（否则两边会慢慢分叉）
        self.subquery = nl.NLQuery(
            raw_text=query.raw_text, conditions=list(self.plan.replayable),
            sector=query.sector, board=query.board,
            exclude_st=query.exclude_st, source="backtest")

        self._data: dict = {}                 # code -> DataFrame（含派生列）
        self._loaded_range = None
        self._no_data: set = set()

    # ── 数据 ──────────────────────────────────────────────

    def preload(self, start_date: str, end_date: str):
        """预加载日线。判据按**请求区间**记录，不看数据里有没有那几天
        （本项目踩过：用"数据首根 K 线"反推判据，长假后每次都判"不覆盖"→ 重复全量抓取）。"""
        if self._loaded_range == (start_date, end_date):
            return
        from data import bars as _bars
        from data import sina_source
        ok = 0
        missing: list[str] = []
        for code in self.codes:
            if code in self._no_data:
                continue                              # 缺失标的**只尝试一次**
            # ⭐ P0-1：本地优先（单只 5~20ms），本地缺的攒起来最后统一联网兜底
            df = _bars.get_bars(code, start_date, end_date)
            if df is None:
                missing.append(code)
                continue
            self._data[code] = self._prepare(df)
            ok += 1
        if missing and not _bars.offline_mode():
            for code in missing:
                df = sina_source.fetch_stock_daily(code, start_date, end_date)
                if df is None or df.empty:
                    self._no_data.add(code)
                    continue
                self._data[code] = self._prepare(df)
                ok += 1
        elif missing:
            _logger.info("[条件回测] 离线模式：跳过 %d 只本地缺失代码", len(missing))
        if missing:
            _logger.info("[条件回测] 本地缺 %d 只", len(missing))
        self._loaded_range = (start_date, end_date)
        _logger.info("[条件回测] 预加载 %d/%d 只（%s ~ %s）",
                     ok, len(self.codes), start_date, end_date)

    @staticmethod
    def _prepare(df):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        close = pd.to_numeric(df["close"], errors="coerce")
        df["ma20"] = close.rolling(20).mean()
        df["ma10"] = close.rolling(10).mean()
        df["ma5"] = close.rolling(5).mean()
        df["chg"] = close.pct_change() * 100.0
        return df

    def _slice(self, code: str, date_str: str):
        """**前视防护的唯一出口**：只返回 `date <= date_str` 的行。

        任何判定都必须经过它。若有人绕过它直接用 `self._data[code]`，
        就等于把未来数据喂给了当天的判断——回测会"稳赚"，而看不出来。
        """
        df = self._data.get(code)
        if df is None:
            return None
        cut = pd.to_datetime(date_str)
        return df[df["date"] <= cut]

    # ── 判定 ──────────────────────────────────────────────

    def facts_on(self, code: str, date_str: str) -> dict:
        """用 ≤T 的数据算出可回放指标（单位与 `indicator_registry` 一致）"""
        bars = self._slice(code, date_str)
        if bars is None or len(bars) < 21:
            return {}
        close = bars["close"]
        last = bars.iloc[-1]
        ret = pd.to_numeric(close, errors="coerce").pct_change().dropna().tail(20)
        vol = float(ret.std() * (252 ** 0.5) * 100) if len(ret) >= 2 else None
        amount_yuan = last.get("amount")
        return {
            "change_pct": _f(last.get("chg")),
            "amount": None if amount_yuan is None else _f(amount_yuan) / 1e8,  # 元 → 亿元
            "turnover": _f(last.get("turnover_pct")),
            "momentum20": (float(close.iloc[-1]) / float(close.iloc[-21]) - 1) * 100
                          if len(close) >= 21 and float(close.iloc[-21]) else None,
            "volatility": vol,
            "ma_bullish": bool(_f(last.get("ma5")) and _f(last.get("ma10"))
                               and _f(last.get("ma20"))
                               and last["ma5"] > last["ma10"] > last["ma20"]),
            "ma20_above": bool(_f(last.get("ma20"))
                               and float(close.iloc[-1]) > float(last["ma20"])),
        }

    def passes(self, code: str, date_str: str) -> bool:
        """复用 `nl_screener.evaluate` —— 与选股同一套核对实现"""
        facts = self.facts_on(code, date_str)
        if not facts:
            return False
        return nl.evaluate(facts, self.subquery)["pass"]

    # ── 主流程 ────────────────────────────────────────────

    def generate_signals(self, start_date: str, end_date: str) -> list:
        """逐日重放 → 交易信号（买入 T+1 开盘成交；卖出同）"""
        if not self.plan.ok:
            return []
        self.preload(start_date, end_date)
        cal = self._trading_days(start_date, end_date)
        if len(cal) < 2:
            return []

        trades: list = []
        holdings: dict = {}            # code -> {qty, entry_price, entry_idx}
        cash = self.initial_cash
        pending: list = []             # 前一日收盘产生的挂单，今日开盘执行

        for i, today in enumerate(cal):
            # ① 执行**上一轮挂在今天**的单（T+1 开盘）
            #    ⚠️ 必须用"今天"的行情去成交昨天生成的挂单。第一版这里写成了
            #    `self._bar(today, None)`（code=None → 返回 None）→ 每一单都因
            #    "无行情"被跳过 → **一笔都不成交**，回测静默出空结果。
            cash = self._fill_pending(pending, holdings, trades, cash, today, i)
            pending = []

            if i == len(cal) - 1:
                break                                  # 最后一天不再产生新信号

            # ② 卖出判定（止盈/止损/到期）
            for code in list(holdings):
                h = holdings[code]
                bar = self._bar(today, code)
                if bar is None:
                    continue                           # 停牌：继续持有
                close = _f(bar.get("close"))
                if close is None:
                    continue
                held = i - h["entry_idx"]
                stopped = close <= h["entry_price"] * (1 - self.stop_loss_pct / 100)
                if held >= self.holding_days or stopped:
                    pending.append({"code": code, "direction": "卖出",
                                    "reason": ("止损" if stopped else
                                               f"持有 {held} 个交易日到期")})

            # ③ 买入判定（按成交额降序取前 N，流动性优先）
            slots = self.max_positions - len(holdings) - len(
                [p for p in pending if p["direction"] == "买入"])
            if slots > 0:
                picked = []
                for code in self.codes:
                    if code in holdings or code in self._no_data:
                        continue
                    if self.passes(code, today):
                        bar = self._bar(today, code)
                        amt = _f(bar.get("amount")) if bar is not None else None
                        picked.append((amt or 0.0, code))
                picked.sort(reverse=True)
                for _, code in picked[:slots]:
                    pending.append({"code": code, "direction": "买入",
                                    "reason": f"满足 {len(self.plan.replayable)} 条可回放条件"})

        return trades

    # ── 内部 ──────────────────────────────────────────────

    @staticmethod
    def _trading_days(start_date: str, end_date: str) -> list:
        try:
            from data import trading_calendar as tc
            days = [str(d) for d in tc.trading_days(start_date, end_date)]
            if days:
                return days
        except Exception as e:                        # noqa: BLE001
            _logger.warning("[条件回测] 交易日历不可用(%s)，降级为自然工作日", e)
        return [d.strftime("%Y-%m-%d")
                for d in pd.date_range(start_date, end_date, freq="B")]

    def _bar(self, date_str: str, code):
        """当日行情行（code=None 时仅用于资金账本，返回 None）"""
        if code is None:
            return None
        bars = self._data.get(code)
        if bars is None:
            return None
        hit = bars[bars["date"] == pd.to_datetime(date_str)]
        return None if hit.empty else hit.iloc[-1]

    def _fill_pending(self, pending, holdings, trades, cash, exec_date, idx):
        """按 `exec_date` 的**开盘价**成交；触及涨跌停一律放弃（不追涨、跌停卖不掉）"""
        for order in pending:
            code = order["code"]
            bar = self._bar(exec_date, code)
            if bar is None:
                continue                               # 停牌 / 无行情
            price = _f(bar.get("open"))
            if not price or price <= 0:
                continue
            if self._limit_blocked(code, exec_date, bar):
                continue
            if order["direction"] == "买入":
                budget = min(cash * 0.3, cash / max(1, self.max_positions))
                qty = int(budget / price // 100 * 100)
                if qty < 100:
                    continue
                cost, fee = qty * price, qty * price * 0.0003
                if cost + fee > cash:
                    continue
                cash -= cost + fee
                holdings[code] = {"qty": qty, "entry_price": price, "entry_idx": idx}
                trades.append({"date": exec_date, "code": code, "direction": "买入",
                               "price": price, "quantity": qty,
                               "reason": order.get("reason", "")})
            else:
                h = holdings.pop(code, None)
                if h is None:
                    continue
                qty = h["qty"]
                cash += qty * price * (1 - 0.0003 - 0.001)   # 佣金 + 印花税
                trades.append({"date": exec_date, "code": code, "direction": "卖出",
                               "price": price, "quantity": qty,
                               "reason": order.get("reason", "")})
        return cash

    def _prev_close(self, code, date_str):
        bars = self._data.get(code)
        if bars is None:
            return None
        before = bars[bars["date"] < pd.to_datetime(date_str)]
        return None if before.empty else _f(before.iloc[-1]["close"])

    def _limit_blocked(self, code, exec_date, bar) -> bool:
        """开盘即涨跌停 → 放弃该单（阈值按**板块**取，与 matching 同一实现）

        ⚠️ 单位：`matching.limit_pct` 返回的是**小数**（主板 0.1 = 10%），
        而这里算出的涨跌幅是**百分数**（1.74 表示 1.74%）。第一版忘了 ×100，
        于是 `1.74 >= 0.1 - 0.2 = -0.1` 恒为真 → **每一笔买单都被当成"开盘涨停"
        拦掉**，回测静默出空结果。又是"同列名不同单位"这一类错。
        """
        from core import matching
        prev, open_p = self._prev_close(code, exec_date), _f(bar.get("open"))
        if not prev or not open_p:
            return False
        limit_pct = matching.limit_pct(code) * 100        # 小数 → 百分数
        return abs(open_p / prev - 1) * 100 >= limit_pct - 0.2


def _f(v):
    """转 float；失败/NaN 一律 None（**不返回 0**）"""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def run_condition_backtest(codes, query, start_date: str, end_date: str,
                           holding_days: int = 20, max_positions: int = 5):
    """一键回测入口：生成信号 → 交给既有回测引擎（撮合/约束/指标全部复用）。

    Returns:
        (BacktestResult, ReplayPlan)；无信号时 result 为 None。
    """
    from core.backtest import BacktestEngine
    gen = ConditionSignalGenerator(codes, query, holding_days=holding_days,
                                   max_positions=max_positions)
    trades = gen.generate_signals(start_date, end_date)
    if not trades:
        return None, gen.plan
    engine = BacktestEngine()
    engine.inject_kline_cache(gen._data)
    result = engine.run(trades, start_date=start_date, end_date=end_date)
    return result, gen.plan
