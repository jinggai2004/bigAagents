"""因子有效性分析：IC / ICIR / 分层收益 / 因子相关性

为什么需要
----------
项目有一条"六因子加权 → 打分 → 排序"的链（`core/sector.py::FACTOR_WEIGHTS`），
但那些权重（主力资金 0.2353 / PE 0.1765 / …）**是硬编码的常量，从未被数据验证过**。
在 `core/ scripts/ data/` 下搜 `ic` / `icir` / `quantile` / 分层收益 **一条都没有**。

也就是说：那条链**形态上是量化**，但**权重没有依据**。
本模块补上这一步——**让权重有依据，或者诚实地证明它没有**。
答案无论正负都有用：负结果说明"当前权重是拍的"，这本身就是重要结论。

这一版的边界（可算性决定，别指望它一次算全）
-----------------------------------------
✅ **能算（K 线类，零新增数据依赖）**
    momentum20 / volatility / ma_bullish / ma20_above
❌ **这一版算不了**
    pe / market_cap / turnover / amount  ← 需要**历史截面**（见下）
    fund_net / news_positive              ← 历史无稳定免费源

⚠️ 为什么 PE/市值/换手率也算不了（这条必须讲清楚，否则会被误读成"忘了做"）
    IC 的定义是「**某日 T 的截面因子值排名** vs T 之后的收益排名」，
    所以它要求能重建"T 那一天的全市场截面"。
    K 线类因子满足这一点（任一历史交易日的日线都能重算）；
    而 PE/市值/换手率的**绝对水平**只能来自当时的快照——项目目前只存**当期**快照。
    ⇒ 想算它们的 IC，必须先**逐日积累快照**；`snapshot_gap_note()` 里写了做法。

口径
----
- **IC**：Spearman **秩**相关（对异常值稳健，业界标准）
- **ICIR** = mean(IC) / std(IC)，衡量稳定性
- **t 值** ≈ ICIR × √期数，|t| > 2 粗略视为显著
- **分层收益**：按因子值分 n 组，看各组未来收益是否**单调**
- 每期至少 `min_names` 只票才计算（样本太少的相关没意义）
"""

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.condition_backtest import ConditionSignalGenerator
from data import sina_source

_logger = logging.getLogger(__name__)

# 可分析的因子：name → (展示名, 方向, 说明)
# 方向 +1 表示"值越大越好"（IC 为正才算有效），-1 表示"值越小越好"
# （波动率越低越好，所以若它的 IC 显著为负，反而是对的方向）
FACTORS = {
    "momentum20": {"label": "20日动量", "direction": 1,
                   "note": "close / close[-20] − 1，衡量趋势"},
    "volatility": {"label": "年化波动率", "direction": -1,
                   "note": "近 20 日日收益标准差 × √252，越低越稳"},
    "ma_bullish": {"label": "均线多头排列", "direction": 1,
                   "note": "MA5 > MA10 > MA20，布尔因子"},
    "ma20_above": {"label": "站上20日均线", "direction": 1,
                   "note": "收盘价 > MA20，布尔因子"},
    # ── 论文因子扫描晋身（2026-10-30，`output/因子论文扫描_30因子_回测报告_2026-10-30.md`）──
    # 全市场月度截面回测（92 个月 / 5,614 只）｜t|≥4.5 且分年同号的 6 个。
    # ⚠️ 负 IC 在本项目语义 = "高分位跑输" ⇒ 当作**剔除闸门**（低分位好）用，
    #    与"分组不单调 ⇒ 只能分位筛不能多空"的既有结论一致。
    "upshd20": {"label": "20日上影占比", "direction": -1,
                "note": "((high−max(open,close))/close) 20 日均值；长上影=抛压（IC=−0.095, t=−5.2, 分年 8/8 负）"},
    "turn_std20": {"label": "换手波动20", "direction": -1,
                   "note": "换手率 20 日标准差（Chordia et al. 2001；IC=−0.090, t=−5.2, 分年 8/8 负）"},
    "amihud20": {"label": "Amihud非流动性", "direction": 1,
                 "note": "(|日收益|/成交额) 20 日均值×1e9（Amihud 2002；IC=+0.074, t=+4.9，Q5−Q1=+4.07%/月——容量受限）"},
    "lndvol20": {"label": "log成交额20", "direction": -1,
                 "note": "成交额 20 日均值取 log（流动性代理；IC=−0.109, t=−7.0，全场最强单因子）"},
    "mom1m": {"label": "1月反转", "direction": -1,
              "note": "close/close[−21]−1（Jegadeesh 1990；IC=−0.051, t=−3.4——A 股反转强动量弱）"},
    "pv_corr20": {"label": "量价相关20", "direction": -1,
                  "note": "日收益与 Δ换手的 20 日相关（量价背离；IC=−0.049, t=−5.2, 分年 8/8 负）"},
}

# 这一版算不了的因子 + 原因（写进报告，避免被读成"漏做"）
UNAVAILABLE = {
    "pe": "需要历史截面（当期 PE 不能当历史 PE）；需逐日积累快照",
    "market_cap": "同上（需历史截面：历史股本 × 当时收盘价）",
    # ⚠️ 2026-10-29 turn_share 全量回填后，换手率/成交额/流通股本**已可得**
    #    （turn20/amihud20/lndvol20 等已晋身 FACTORS）——留此条目仅作历史记录。
    "fund_net": "历史资金流无稳定免费源",
    "news_positive": "历史舆情无稳定免费源",
}


def snapshot_gap_note() -> str:
    """给用户看的"怎么才能算 PE/市值的 IC"（具体到可执行）。

    ⚠️ 2026-10-30 起**换手率不再在此列**：turn_share 全量回填后历史换手可得，
       amihud20/turn_std20 等已晋身 FACTORS——SNAPSHOT_FACTORS 只剩 pe/market_cap。
    """
    return ("要算 PE / 总市值的 IC，需要**逐日积累截面快照**：每个交易日收盘后由"
            " `snapshot_flow`(15:40) 调 `data.sina_source.get_sector_snapshot()` 并落库到"
            " `market_snapshot` 表，积累 ≥60 个交易日后即可用本模块分析。"
            "换手率/成交额类因子已由 turn_share 回填解决（不依赖快照）。")


# 需要"历史截面"才能算的因子。它们与 K 线因子的区别在数据形态：
# K 线因子任何历史交易日都能重算，而这几项的**绝对水平**只存在于当时的快照里。
# ⚠️ 2026-10-30 起换手率移出：turn_share 回填后不依赖快照（amihud20 等已进 FACTORS）
SNAPSHOT_FACTORS = ("pe", "market_cap")

# 攒够多少个交易日的截面，才认为 IC 有统计意义。
# 取 60（≈ 3 个月）：IC 本身是"每期一个数"，60 期才有足够的样本算 ICIR / t 值
# （t ≈ ICIR × √期数，期数太小时 t 几乎没有分辨力）。
MIN_SNAPSHOT_DATES = 60


def snapshot_history_status() -> dict:
    """已积累的估值截面天数 —— 决定 PE / 市值 / 换手率的 IC「能不能算」。

    ⚠️ 这个函数存在的意义是**把"不可算"从一句解释变成一条进度**。
    "需要历史截面"如果没有数字，用户无法判断是"再多等几天"还是"根本没有这条路"。
    """
    try:
        from db import store as db_store
        dates = db_store.market_snapshot_dates()
    except Exception as e:                            # noqa: BLE001
        return {"dates": 0, "first": None, "last": None, "enough": False,
                "need": MIN_SNAPSHOT_DATES,
                "reason": f"读取快照表失败：{type(e).__name__}: {e}"}
    n = len(dates)
    return {"dates": n, "first": dates[0] if dates else None,
            "last": dates[-1] if dates else None,
            "enough": n >= MIN_SNAPSHOT_DATES,
            "need": MIN_SNAPSHOT_DATES,
            "reason": ("" if n >= MIN_SNAPSHOT_DATES else
                       f"已积累 {n} / {MIN_SNAPSHOT_DATES} 个交易日的截面快照"
                       f"（`market_snapshot` 表，调度任务 snapshot_flow 每天 15:40 落库）"
                       f"，攒够后才能算这三项的 IC")}


@dataclass
class AnalysisResult:
    summary: list = None            # list[dict]：每个因子的 IC 汇总
    layered: dict = None            # {factor: [{"组":..,"平均收益":..}]}
    correlation: object = None      # DataFrame | None：因子相关矩阵（均值）
    periods: int = 0
    names: int = 0
    panel_rows: int = 0
    horizon: int = 20
    step: int = 5
    groups: int = 5
    unavailable: dict = None
    note: str = ""
    # 相关矩阵里有缺失值的期数（布尔因子整期取值恒定 → 该期 Spearman 无定义），
    # 以及参与平均的期数。两个值一起给，才能读出"缺了 3 / 共 64 期"这类信息。
    corr_periods: int = 0
    corr_const_periods: int = 0
    # 估值截面快照的积累进度（PE / 市值 / 换手率 IC 的前置条件）
    snapshot_status: dict = None

    def to_dict(self) -> dict:
        return {
            "summary": self.summary or [],
            "layered": self.layered or {},
            "correlation": (self.correlation.round(3).to_dict()
                            if self.correlation is not None else None),
            "periods": self.periods, "names": self.names,
            "panel_rows": self.panel_rows,
            "horizon": self.horizon, "step": self.step, "groups": self.groups,
            "unavailable": self.unavailable or {},
            "corr_periods": self.corr_periods,
            "corr_const_periods": self.corr_const_periods,
            "snapshot_status": self.snapshot_status or {},
            "note": self.note,
        }


def _num(v):
    """转 float；布尔转 0/1；失败/NaN 一律 None（**不返回 0**）"""
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _guard_one(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """单只日线的守卫裁剪（`load_bars` 用）。返回可能为空表。"""
    import os as _os

    if _os.environ.get("BIGA_NO_UNIVERSE_GUARD", "").strip().lower() in ("1", "true", "yes"):
        return df
    from core import universe_guard as _ug
    from data import bars as _bars_mod

    if df is None or len(df) == 0 or "date" not in df.columns:
        return df
    kept, _cut = _ug.trim_with_stats(df, _bars_mod._clean(code))
    return kept


def load_bars(codes, start, end, progress_cb=None, guard: bool = True) -> dict:
    """加载并派生因子列。**复用 `ConditionSignalGenerator._prepare`** ——
    与条件回测用同一套派生逻辑，避免"两处算的 MA 不一样"这类漂移。

    ⭐ P0-1（2026-10-18）：**本地优先**（`data/panel/` 单只 5~20ms），
    本地缺的才走新浪。原来全部联网 ⇒ 30 只样本要几十秒、全市场不可能。
    """
    from data import bars as _bars
    bars_map = {}
    total = max(1, len(codes))
    missing: list[str] = []
    trimmed_out: list[str] = []          # 整段不可比（不是取数失败，分开记）
    for i, code in enumerate(codes, 1):
        if progress_cb:
            progress_cb({"percent": 5 + i / total * 70,
                         "message": f"加载日线 {i}/{total}", "current": i, "total": total})
        df = _bars.get_bars(code, start, end)
        if df is None:
            missing.append(code)
            continue
        if guard:
            # ⚠️ 先裁再派生：`_enrich` 里的 MA60/波动率会吃到窗口前段价格，
            #    拿不可比区间（上市初期、停牌跨段）算出来的指标本身就是脏的。
            df = _guard_one(df, code)
            if len(df) == 0:
                trimmed_out.append(str(code))
                continue
        df = _enrich(df)
        if len(df) >= 40:
            bars_map[_bars._clean(code)] = df
    # ── 兜底：本地没有的再联网（`BIGA_OFFLINE=1` 时跳过，缺口进日志）──
    if missing and _bars.offline_mode():
        _logger.info("[factor] 离线模式：跳过 %d 只本地缺失代码（%s...）",
                     len(missing), ",".join(missing[:5]))
        if trimmed_out:
            _logger.info("[factor] 宇宙守卫：%d 只整段不可比已剔除", len(trimmed_out))
        return bars_map
    for i, code in enumerate(missing, 1):
        if progress_cb:
            progress_cb({"percent": 5 + (len(codes) - len(missing) + i) / total * 70,
                         "message": f"联网补日线 {code}", "current": i, "total": len(missing)})
        df = sina_source.fetch_stock_daily(code, start, end)
        if df is None or df.empty:
            continue
        if guard:
            df = _guard_one(df, code)
            if len(df) == 0:
                trimmed_out.append(str(code))
                continue
        df = _enrich(df)
        if len(df) >= 40:
            bars_map[code] = df
    if missing:
        _logger.info("[factor] 本地缺 %d/%d 只，已联网兜底", len(missing), len(codes))
    if trimmed_out:
        _logger.info("[factor] 宇宙守卫：%d 只整段不可比已剔除（%s）",
                     len(trimmed_out), ",".join(trimmed_out[:5]))
    return bars_map


def _enrich(df):
    """在 `_prepare` 的基础列上补**全部** `FACTORS` 列。

    ⚠️ 这里必须覆盖 `FACTORS` 里的每一个键。第一版只派生了 momentum20 与
    volatility，忘了两个布尔因子 —— 而 `build_panel` 对"列不存在"的处理是
    `row[f] = None`（不报错），于是它们**静默变成 0 期 IC**，报告上只写
    "期数不足"，看上去像是样本问题。由
    `tests/test_factor_analysis.py::test_enrich_covers_every_declared_factor` 守护。
    """
    df = ConditionSignalGenerator._prepare(df)
    close = pd.to_numeric(df["close"], errors="coerce")
    # ⚠️ P1-1：这里是**面板级向量化**计算（全市场 × N 年），不能逐点调
    #    `core.indicators` 的标量函数（那是给"单只一次"的场景用的）。
    #    两者数学等价，由 tests/test_indicators.py::TestParityWithScreener 守护：
    #    `indicators.momentum(close,20) == close/close.shift(20)-1` 在同一时点逐值相等。
    df["momentum20"] = close / close.shift(20) - 1.0
    df["volatility"] = (close.pct_change().rolling(20).std() * (252 ** 0.5))
    # 布尔因子（取值 0/1）——与条件回测的 `conditions` 口径一致
    df["ma_bullish"] = (df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"])
    df["ma20_above"] = close > df["ma20"]
    # ── 论文因子扫描晋身的 6 个（2026-10-30）──
    # ⚠️ turn/outstanding 来自 turn_share 回填，尾部 1~2 天未回填 ⇒ ffill(3)，
    #    与 `scripts/factor_paper_scan.py`（月度截面扫描）同口径。
    # ⚠️ 列可能缺失（合成测试/旧数据）⇒ 用 NaN 序列占位，**列必须存在**（覆盖守护）
    def _col(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    r = close.pct_change()                      # 日收益（amihud/量价相关用）
    turn = _col("turnover_pct").ffill(limit=3)
    amt = _col("amount")
    high, opn = pd.to_numeric(df["high"], errors="coerce"), pd.to_numeric(df["open"], errors="coerce")
    df["upshd20"] = ((high - np.maximum(opn, close)) / close).rolling(20).mean()
    df["turn_std20"] = turn.rolling(20).std()
    df["amihud20"] = (r.abs() / amt.replace(0.0, np.nan)).rolling(20).mean() * 1e9
    df["lndvol20"] = np.log(amt.rolling(20).mean().replace(0.0, np.nan))
    df["mom1m"] = close / close.shift(21) - 1.0
    df["pv_corr20"] = r.rolling(20).corr(turn.diff())
    return df


def _pos_on_or_before(bars, date) -> int | None:
    """`date <= T` 的最后一个位置 —— **前视防护的唯一出口**。

    与 `condition_backtest._slice` 同一立场：任何取值都必须经过它。
    """
    idx = int(bars["date"].searchsorted(pd.to_datetime(date), side="right")) - 1
    return None if idx < 0 else idx


def _sample_dates(start, end, step) -> list:
    """采样交易日（每 step 个交易日取一个，控制计算量）"""
    try:
        from data import trading_calendar as tc
        days = [str(d) for d in tc.trading_days(start, end)]
    except Exception as e:                            # noqa: BLE001
        _logger.warning("[因子分析] 交易日历不可用(%s)，降级为自然工作日", e)
        days = [d.strftime("%Y-%m-%d")
                for d in pd.date_range(start, end, freq="B")]
    return days[::max(1, int(step))]


def build_panel(bars_map, factors, dates, horizon) -> pd.DataFrame:
    """构造面板：每行是「某只票在某采样日的因子值 + 之后 horizon 日的收益」

    ⚠️ 未来收益用**该票自己的第 horizon 根之后**（位置索引），
    而不是日历日 —— 停牌/缺数据时不会把别人的收益算进来。
    """
    rows = []
    for code, bars in bars_map.items():
        closes = bars["close"].values
        for d in dates:
            pos = _pos_on_or_before(bars, d)
            if pos is None or pos < 20:
                continue                              # 因子需要 20 根历史
            fwd = pos + horizon
            if fwd >= len(bars):
                continue                              # 未来收益还不可得
            base = closes[pos]
            if not base or base != base:
                continue
            row = {"date": d, "code": code,
                   "fwd_ret": float(closes[fwd]) / float(base) - 1.0}
            for f in factors:
                row[f] = _num(bars[f].iloc[pos]) if f in bars.columns else None
            rows.append(row)
    return pd.DataFrame(rows)


def _guard_bars_map(bars_map: dict) -> dict:
    """对已加载的日线逐只**裁剪不可比区间**（上市满一年 / 北交所开市前）。

    为什么必须在因子分析里做：IC 是"截面排序"的统计量，只要有一只票在某个采样日
    还处于"上市初期/停牌跨段"，它的未来收益就会被复牌跳空污染，而 RankIC 对
    极端值敏感——一只 +142% 的伪样本足以把整期 IC 拉偏。

    与被裁空（整票不可比）的票**直接丢弃并计数**：它在本区间没有任何可比历史，
    留着只会贡献 NaN 行。计数要报日志，否则"票数变少"会被误读成取数失败。
    """
    import os as _os

    if _os.environ.get("BIGA_NO_UNIVERSE_GUARD", "").strip().lower() in ("1", "true", "yes"):
        return bars_map
    from core import universe_guard as _ug

    out = {}
    cut_rows = 0
    emptied = []
    unknown = []
    for code, df in bars_map.items():
        if df is None or len(df) == 0:
            continue
        if "date" not in df.columns:
            # 没有日期列就没法判可比性；如实保留并提示，不静默丢数据
            _logger.warning("[factor] %s 缺 date 列，跳过了宇宙守卫", code)
            out[code] = df
            continue
        # ⚠️ 区分两种"被剔"的原因——它们对用户的含义完全不同：
        #    unknown：清单里查不到这只票（退市 / 面板还没收进来）⇒ 无法判可比性；
        #    emptied：清单里有，但**这一整段**都不可比（上市不满一年 / 北交所开市前）。
        #    混成一条日志时，用户无法判断"是我传错了票"还是"这段历史确实不能用"。
        if _ug.comparable_since(str(code)) is None:
            unknown.append(str(code))
            continue
        kept, cut = _ug.trim_with_stats(df, code)
        cut_rows += cut
        if len(kept) == 0:
            emptied.append(str(code))
            continue
        out[code] = kept
    if cut_rows:
        _logger.info("[factor] 宇宙守卫：剪掉 %d 行不可比历史", cut_rows)
    if emptied:
        _logger.info("[factor] 宇宙守卫：%d 只整个区间不可比（%s）",
                     len(emptied), ",".join(emptied[:5]))
    if unknown:
        _logger.warning("[factor] 宇宙守卫：%d 只在面板清单里查不到、无法判可比性已剔除"
                        "（%s）—— 若是退市票属正常，若是新票请先更新面板",
                        len(unknown), ",".join(unknown[:5]))
    return out


def run(codes, factors=None, start="2024-01-01", end="2025-06-30",
        horizon: int = 20, step: int = 5, groups: int = 5,
        min_names: int = None, bars_map=None, progress_cb=None,
        guard: bool = True) -> AnalysisResult:
    """跑一遍因子有效性分析。

    Args:
        codes: 股票池
        guard: 是否套**宇宙守卫**（上市满一年 + 北交所开市后才纳入）。
            默认 True；研究结论要能复核才关（`BIGA_NO_UNIVERSE_GUARD=1` 也能关）。
            ⚠️ 关掉得到的 IC 会被"停牌跨段的复牌跳空"污染，**不可用于决策**。
        factors: 要分析的因子键（None = `FACTORS` 全部）
        start/end: 区间
        horizon: 未来收益窗口（交易日）
        step: 每多少个交易日采样一次（控制计算量）
        groups: 分层组数
        min_names: 每期最少多少只票才算 IC。
            **None = 自适应**（池子的 60%，夹在 8~30 之间）；
            显式传值则**按你给的来**（那是"我要更严格的样本门槛"的意思）。
            ⚠️ 为什么默认要自适应：12 只票的池子配固定 20 只门槛，会让 IC
            **永远是 0 期**——面板 768 行，结论却只有"期数不足"，用户无从判断
            是自己的数据不够还是门槛设错了。但反过来，**不能连用户显式给的值
            也一起改掉**，那是把"更严格"静默变成"更宽松"。
    """
    factors = [f for f in (factors or list(FACTORS)) if f in FACTORS]
    res = AnalysisResult(horizon=horizon, step=step, groups=groups,
                         unavailable=dict(UNAVAILABLE))
    # 把"暂不可算"的理由接上**实时进度**（攒了多少天截面），
    # 否则用户看到的是"需要历史截面"这种无法行动的解释。
    _snap = snapshot_history_status()
    for k in SNAPSHOT_FACTORS:
        if k in res.unavailable and _snap.get("reason"):
            res.unavailable[k] = f"{res.unavailable[k]}；{_snap['reason']}"
    res.snapshot_status = _snap

    if bars_map is None:
        bars_map = load_bars(codes, start, end, progress_cb)
    # ⚠️ 守卫放在**这里**而不是 `load_bars` 里：注入 `bars_map` 的调用方
    #    （研究面板缓存复用、脚本批量复跑）走的是另一条路，只改 load_bars 会漏。
    if guard:
        bars_map = _guard_bars_map(bars_map)
    res.names = len(bars_map)
    if not bars_map or not factors:
        res.note = "没有可用数据或没有选因子"
        return res

    # 自适应只作用于**默认值**；用户显式指定时按他的来（见 docstring）
    if min_names is None:
        effective_min = max(8, min(30, int(len(bars_map) * 0.6))) if bars_map else 8
    else:
        effective_min = max(3, int(min_names))

    dates = _sample_dates(start, end, step)
    if progress_cb:
        progress_cb({"percent": 78, "message": "构造因子面板"})
    panel = build_panel(bars_map, factors, dates, horizon)
    res.panel_rows = len(panel)
    if panel.empty:
        res.note = "面板为空：区间内没有同时满足「因子可算」与「未来收益可得」的样本"
        return res

    if progress_cb:
        progress_cb({"percent": 85, "message": "计算 IC 与分层收益"})

    # ── IC ──
    ic_rows = []
    for d, g in panel.groupby("date"):
        for f in factors:
            sub = g[[f, "fwd_ret"]].dropna()
            if len(sub) < effective_min:
                continue
            with warnings.catch_warnings():
                # 因子在该期取值恒定 → Spearman 无定义（NaN，下面会被跳过）。
                # 已知且已处理，就地静音，避免刷屏。
                warnings.simplefilter("ignore")
                ic = sub[f].corr(sub["fwd_ret"], method="spearman")
            if ic == ic:                              # 非 NaN
                ic_rows.append({"date": d, "factor": f, "ic": float(ic),
                                "n": len(sub)})
    ic_df = pd.DataFrame(ic_rows)
    res.periods = int(ic_df["date"].nunique()) if not ic_df.empty else 0

    summary = []
    for f in factors:
        meta = FACTORS[f]
        s = ic_df[ic_df["factor"] == f]["ic"] if not ic_df.empty else pd.Series(dtype=float)
        row = {"key": f, "因子": meta["label"], "方向": "越大越好" if meta["direction"] > 0 else "越小越好"}
        if len(s) < 3:
            row.update({"IC均值": None, "ICIR": None, "t值": None, "IC>0比例": None,
                        "期数": len(s), "结论": "期数不足（<3），无法判断"})
            summary.append(row)
            continue
        mean, std = float(s.mean()), float(s.std(ddof=1))
        # ⚠️ std 为 0 是**可能的**：构造型/极端一致的样本里每期 IC 完全相同，
        #    此时 mean/std 无定义。第一版写成 `icir = mean/std if std else 0.0`
        #    → 把"完全稳定"报告成 ICIR=0（即"毫无价值"），语义彻底反了。
        if std > 1e-9:
            icir = mean / std
            t = icir * (len(s) ** 0.5)
        else:
            icir = None
            t = float("inf") if abs(mean) > 0.05 else 0.0
        # 方向修正：波动率这类"越小越好"的因子，负 IC 才是对的方向。
        # ⚠️ 这里必须用 **mean**（一定存在），不能用 icir ——
        #    std=0 时 icir 是 None，`None * int` 会直接抛 TypeError。
        #    （改 icir 语义时漏改这里，8 个用例当场全红。）
        dir_ok = (mean * meta["direction"]) > 0
        row.update({
            "IC均值": round(mean, 4), "IC标准差": round(std, 4),
            "ICIR": (round(icir, 3) if icir is not None else None),
            "t值": (round(t, 2) if t != float("inf") else 999.0),
            "IC>0比例": round(float((s > 0).mean()), 3),
            "期数": int(len(s)),
            "结论": ("显著有效" if abs(t) > 2 and dir_ok else
                     "显著但方向相反" if abs(t) > 2 else
                     "不显著（|t|<2）"),
        })
        summary.append(row)
    res.summary = summary

    # ── 分层收益 ──
    layered = {}
    for f in factors:
        rows = []
        for d, g in panel.groupby("date"):
            sub = g[[f, "fwd_ret"]].dropna()
            # 分层要求每组至少有票可分：groups*2 是下限（10 只票分 5 组 = 每组 2 只）。
            # 第一版写 groups*3 → 10 只票的池子永远算不出分层（静默为空）。
            if len(sub) < max(effective_min, groups * 2):
                continue
            try:
                sub = sub.assign(grp=pd.qcut(sub[f].rank(method="first"),
                                             groups, labels=False))
            except Exception:                         # noqa: BLE001 - 取值过于集中
                continue
            for gi, grp in sub.groupby("grp"):
                rows.append({"grp": int(gi), "ret": float(grp["fwd_ret"].mean())})
        if not rows:
            continue
        agg = pd.DataFrame(rows).groupby("grp")["ret"].mean()
        series = [{"组": f"第 {int(g) + 1} 组（{'低' if int(g) == 0 else '高' if int(g) == len(agg) - 1 else '中'}）",
                   "平均收益%": round(float(v) * 100, 2)} for g, v in agg.items()]
        spread = float(agg.iloc[-1] - agg.iloc[0]) if len(agg) >= 2 else 0.0
        # 单调性：相邻组收益差是否同号
        diffs = agg.diff().dropna()
        mono = bool(len(diffs) and (diffs > 0).all()) or bool(len(diffs) and (diffs < 0).all())
        layered[f] = {"rows": series,
                      "多空差%": round(spread * 100, 2),
                      "单调": mono,
                      "说明": ("分组收益单调 → 因子有区分度" if mono
                               else "分组收益不单调 → 因子区分度弱")}
    res.layered = layered

    # ── 因子相关性 ──
    corrs = []
    const_periods = 0
    for d, g in panel.groupby("date"):
        sub = g[factors].dropna()
        if len(sub) < effective_min:
            continue
        c = None
        with warnings.catch_warnings():
            # 布尔因子整期恒定 → Spearman 无定义，pandas 会抛 ConstantInputWarning。
            # 这是**已知且已处理**的情况（见下方 nan-aware 平均 + 缺测期数披露），
            # 就地静音，免得刷屏淹没真正的告警（实测 50 只池一次刷 38 条）。
            warnings.simplefilter("ignore")
            c = sub.corr(method="spearman")
        corrs.append(c)
        if bool(c.isna().to_numpy().any()):
            const_periods += 1
    if corrs:
        # ⚠️ 这里**不能**写 `sum(corrs) / len(corrs)`：DataFrame 相加会**传播 NaN**。
        #    布尔因子在某一期可能取值恒定（整期全 1 或全 0）→ 该期它的相关系数
        #    是 NaN（方差为 0，Spearman 无定义）→ 一期的 NaN 会把平均后的
        #    **整行整列**都变成 NaN。实测（2026-10-07，50 只池）：`站上20日均线`
        #    整行整列 NaN，连对角线都不是 1.00，看上去像"这个因子算错了"。
        #    `groupby(level=0).mean()` 是 nan-aware 的：只对**非 NaN**的期取平均。
        #    ⚠️ 再补一步 `reindex`：`groupby` 会把**索引按字母排序**，而列仍保持
        #    `factors` 原序 → 同一张矩阵行标签与列标签顺序不同，读起来像"不对称"。
        c = pd.concat(corrs).groupby(level=0).mean()
        res.correlation = c.reindex(index=c.columns, columns=c.columns)
        res.corr_periods = len(corrs)
        res.corr_const_periods = const_periods

    if res.periods == 0 and res.panel_rows > 0:
        res.note = (f"面板有 {res.panel_rows} 行样本，但**每期股票数不足**"
                    f"（需要 ≥ {effective_min} 只，池内有效 {len(bars_map)} 只）"
                    f"→ 未算出 IC。扩大股票池即可解决。")

    if progress_cb:
        progress_cb({"percent": 100, "message": "完成"})
    return res


# ================================================================
# 样本外稳健性：一个区间算出来的方向，换个区间还在不在？
# ================================================================
#
# 为什么必须有这一步
# ----------------
# 单区间的 IC 只能回答"在这段样本上方向如何"，**不能**回答"这个因子是不是无效"。
# 2024 是 A 股反转特征明显的一段，动量反向完全可能是"这一段的特性"而不是"规律"。
# 所以结论要下得稳，必须看**换区间 / 换持有期之后符号还在不在**——
# 符号在多数配置下一致，才叫"稳健"；一换就翻，只能写"当前样本不足以定论"。
#
# ⚠️ 一条性能铁律：**日线只抓一次**。
#    `run()` 支持传入 `bars_map`，所以本函数抓一段**覆盖全部配置**的日线，
#    之后每个配置只在同一份内存数据上重算面板。
#    反面教材就在本项目回测模块：曾因"逐日循环里做联网"把 20 轮对照从 137 秒拖到 3.5 小时。

DEFAULT_ROBUSTNESS_PERIODS = (
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
    ("2025-01-01", "2025-12-31"),
)
DEFAULT_ROBUSTNESS_HORIZONS = (10, 20, 60)

# 判定"稳健"的门槛：多数配置下符号一致才算。
# 取 0.8 而不是 0.5：0.5 等于"抛硬币"，那样的"稳健"没有信息量。
ROBUSTNESS_AGREE_THRESHOLD = 0.8


@dataclass
class RobustnessResult:
    # ⚠️ 用 `default_factory=list` 而不是 `None`：早期返回（没有日线）时
    #    如果 `summary` 是 None，调用方 `for row in res.summary` 会直接 TypeError。
    #    "空结果"与"没有结果"必须都是**可遍历的空列表**，否则每个调用点都得写
    #    `or []`，漏一处就是运行时崩。
    cells: list = field(default_factory=list)      # 逐 (因子, 区间, 持有期) 明细
    summary: list = field(default_factory=list)    # 逐因子的符号一致率与判定
    periods: list = field(default_factory=list)
    horizons: list = field(default_factory=list)
    names: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {"cells": self.cells, "summary": self.summary,
                "periods": self.periods, "horizons": self.horizons,
                "names": self.names, "note": self.note}


def robustness(codes, periods=None, horizons=None, step: int = 5, groups: int = 5,
               min_names: int = None, factors=None, bars_map=None,
               progress_cb=None) -> RobustnessResult:
    """多区间 × 多持有期的符号一致性检验（样本外稳健性）。

    Args:
        codes: 股票池（`bars_map` 为 None 时用于抓日线）
        periods: [(start, end), ...]，默认 `DEFAULT_ROBUSTNESS_PERIODS`
        horizons: 未来收益窗口列表，默认 `(10, 20, 60)`
        bars_map: 预抓好的日线（**覆盖全部区间**）。传了就不联网。

    Returns:
        RobustnessResult；`cells` 每格一个 dict，`summary` 每因子一行。

    ⚠️ `bars_map` 的日线区间必须**覆盖所有 periods 的右端 + 最大 horizon**，
    否则靠后的样本会因为"未来收益还不可得"被静默跳过，表现为"期数很少"
    而不是报错。本函数不替你补抓——传错区间是调用方的责任，
    但 `note` 里会说明实际用到的日线范围。
    """
    periods = list(periods or DEFAULT_ROBUSTNESS_PERIODS)
    horizons = [int(h) for h in (horizons or DEFAULT_ROBUSTNESS_HORIZONS)]
    factors = [f for f in (factors or list(FACTORS)) if f in FACTORS]
    res = RobustnessResult(periods=[list(p) for p in periods],
                           horizons=horizons)

    if bars_map is None:
        lo = min(p[0] for p in periods)
        hi = max(p[1] for p in periods)
        # 右端必须留够最大持有期，否则最后一段的样本全被"未来收益不可得"吃掉
        if progress_cb:
            progress_cb({"percent": 2, "message": "抓取日线（覆盖全部配置，只抓一次）"})
        bars_map = load_bars(codes, lo, _pad_end(hi, max(horizons)),
                             progress_cb=_scaled_progress(progress_cb, 5, 75))
    res.names = len(bars_map)
    if not bars_map:
        res.note = "没有可用日线"
        return res

    total = max(1, len(periods) * len(horizons))
    cells: list = []
    for i, (start, end) in enumerate(periods):
        for j, h in enumerate(horizons):
            k = i * len(horizons) + j
            if progress_cb:
                progress_cb({"percent": 78 + k / total * 20,
                             "message": f"{start[:4]} 年 / 持有 {h} 日"})
            r = run(codes, factors=factors, start=start, end=end, horizon=h,
                    step=step, groups=groups, min_names=min_names,
                    bars_map=bars_map)
            for row in (r.summary or []):
                cells.append({
                    "key": row["key"], "因子": row["因子"],
                    "区间": f"{start[:4]}" if start[:4] == end[:4] else f"{start}~{end}",
                    "start": start, "end": end, "持有期": h,
                    "IC均值": row.get("IC均值"), "t值": row.get("t值"),
                    "期数": row.get("期数"), "结论": row.get("结论"),
                })
    res.cells = cells

    summary = []
    for f in factors:
        meta = FACTORS[f]
        vals = [c["IC均值"] for c in cells
                if c["key"] == f and c["IC均值"] is not None]
        n = len(vals)
        if n == 0:
            summary.append({"key": f, "因子": meta["label"],
                            "方向假设": _dir_label(meta["direction"]),
                            "有效配置": 0, "一致": 0, "相反": 0,
                            "符号一致率": None,
                            "判定": "所有配置都算不出 IC（样本不足）"})
            continue
        # 符号是否与权重假设一致：假设 值越大越好 → 需要 IC > 0；越小越好 → 需要 IC < 0
        agree = sum(1 for v in vals if (v > 0) == (meta["direction"] > 0))
        pct = round(agree / n, 3)
        if pct >= ROBUSTNESS_AGREE_THRESHOLD:
            verdict = "方向稳健：与权重假设一致"
        elif (1 - pct) >= ROBUSTNESS_AGREE_THRESHOLD:
            verdict = "方向稳健：与权重假设**相反**（权重符号疑似装反）"
        else:
            verdict = "方向不稳：随区间/持有期漂移，当前样本不足以定论"
        summary.append({
            "key": f, "因子": meta["label"],
            "方向假设": _dir_label(meta["direction"]),
            "有效配置": n, "一致": agree, "相反": n - agree,
            "符号一致率": pct,
            # 平均 IC 也一并给出：符号一致率只说方向，不说强度
            "平均IC": round(sum(vals) / n, 4),
            "判定": verdict,
        })
    res.summary = summary

    if progress_cb:
        progress_cb({"percent": 100, "message": "完成"})
    return res


def _dir_label(direction: int) -> str:
    return "值越大越好" if direction > 0 else "值越小越好"


def _pad_end(end: str, horizon: int) -> str:
    """把区间右端往后推 horizon 个**自然日**（留够未来收益）。

    用自然日而不是交易日，是因为这里只是"多抓一点"的粗估：
    多抓只会浪费一点流量，抓少了会让最后一段样本被静默跳过。
    """
    try:
        return str((pd.Timestamp(end) + pd.Timedelta(days=int(horizon * 1.6) + 5)).date())
    except Exception:                                 # noqa: BLE001
        return end


def _scaled_progress(cb, lo: float, hi: float):
    """把子任务的 0~100% 映射到总进度的 [lo, hi]%"""
    if cb is None:
        return None

    def inner(ev: dict):
        p = float(ev.get("percent") or 0)
        cb({**ev, "percent": lo + (hi - lo) * p / 100.0})
    return inner
