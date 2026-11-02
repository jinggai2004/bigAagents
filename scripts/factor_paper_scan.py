# -*- coding: utf-8 -*-
"""论文因子 + 开源因子库 批量扫描（61 因子 × 月度调仓截面 × 全市场等权）。

两套来源
────────
① 30 个经典论文因子（手选，见 FACTOR_DEFS，全部只依赖 OHLCV+换手+流通股本）；
② WorldQuant "101 Formulaic Alphas"（Kakushadze 2016）可算子集 **31 个**
   （`core/wq101_alphas.py`，排除 IndNeutralize/cap 类）——开源因子库口径。

为什么是"月度截面"而不是逐日 IC
──────────────────────────────
61 因子 × 5,800 只 × 8 年逐日算 IC，计算量不可控；文献标准做法本来就是
**月度调仓**。每月末取截面、看未来 20 个交易日等权收益，92 个月 × ~4,700 只。

口径（与项目既有结论可比）
──────────────────────────
- 全市场等权（`bars.available_codes()`，5,830 只含北交所）——项目铁律；
- 收益用面板 qfq 收盘价；turn/outstanding 来自 turn_share 回填（全历史）；
- 前视控制：因子只用 t 及之前的数据（rolling 右对齐），收益用 t+20；
- **幸存者偏差**：面板是"当前在市"股票 ⇒ 绝对收益偏乐观，**相对排序仍可用**；
- 未计交易成本与涨跌停不可成交 —— 见报告"可行性"节。
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
_logger = logging.getLogger("factor_scan")

START = "2018-01-01"
SAMPLE_FROM = "2019-03-01"
END = "2026-10-30"
HORIZON = 20
MIN_ROWS = 320
MIN_FACTOR_ROWS = 36

FACTOR_DEFS = [
    ("mom12_1",   "12-1月动量",   "Jegadeesh & Titman (1993); Asness et al. (2013)", +1),
    ("mom1m",     "1月反转",      "Jegadeesh (1990); Lehmann (1990)",                -1),
    ("mom1w",     "1周动量",      "短期动量（周频文献）",                             -1),
    ("mom6m",     "6月动量",      "Jegadeesh & Titman (1993)",                       +1),
    ("high52",    "52周新高距离", "George & Hwang (2004)",                           +1),
    ("ret_max5",  "5日最大涨幅",  "Kumar (2009) 彩票偏好（周窗口变体）",                -1),
    ("up_streak", "连涨天数",     "连续上涨（行为/经验类）",                           -1),
    ("ivol20",    "特质波动20",   "Ang et al. (2006)",                               -1),
    ("vol20",     "总波动20",     "总波动（对照基线）",                                -1),
    ("max20",     "MAX绝对最大",  "Bali, Cakici & Whitelaw (2011)",                  -1),
    ("dvol20",    "下行波动20",   "Ang et al. (2006) 下行风险",                      -1),
    ("skew20",    "收益偏度20",   "Amaya et al. (2015)",                             -1),
    ("kurt20",    "收益峰度20",   "高阶矩风险",                                      -1),
    ("vol_ratio", "短长波动比",   "波动率期限结构（5/60 日）",                         -1),
    ("turn20",    "平均换手20",   "Datar, Naik & Radcliffe (1998)",                  -1),
    ("turn_std20","换手波动20",   "Chordia, Subrahmanyam & Anshuman (2001)",         -1),
    ("amihud20",  "Amihud非流动性","Amihud (2002)",                                  +1),
    ("lndvol20",  "log成交额20",  "流动性代理（成交额，负向）",                        -1),
    ("turn_trend","换手趋势5/120","异常放量（换手期限比）",                            -1),
    ("turn_shock","单日换手冲击", "换手冲击（当日/20日均）",                           -1),
    ("lnfloat",   "log流通市值",  "Banz (1981) 规模效应（流通股本×价格代理）",          -1),
    ("pv_corr20", "量价相关20",   "量价背离（Karpoff 1987 综述）",                    -1),
    ("obv_slope", "量能净流向",   "OBV 思路（20 日净量/成交额）",                     +1),
    ("vr5_20",    "量比5/20",     "量比（短期放量）",                                 -1),
    ("neg_rsi14", "-RSI14",      "RSI 超买反转（内置反转方向）",                      -1),
    ("boll_pos",  "布林带位置",   "Bollinger 位置（均值回归）",                       -1),
    ("upshd20",   "20日上影占比", "长上影=抛压（K 线形态）",                          -1),
    ("macd_hist", "MACD柱/价格",  "趋势（MACD histogram）",                          +1),
    ("atr_pct",   "ATR14/价格",   "真实波幅占比（波动代理）",                          -1),
    ("gapfreq20", "跳空频率20",   "跳空行为（|开盘缺口|>1% 频率）",                   -1),
]


def _sample_dates(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """每月最后一个交易日（只取 SAMPLE_FROM 之后）。"""
    s = pd.Series(dates, index=dates)
    m = s.groupby([dates.year, dates.month]).max()
    return sorted(pd.Timestamp(d) for d in m if pd.Timestamp(d) >= pd.Timestamp(SAMPLE_FROM))


def _factor_frame(df: pd.DataFrame, mkt_ret: pd.Series) -> pd.DataFrame:
    """单只股票 → 30 论文因子列（index=交易日）。全部只用 t 及之前的数据。"""
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    v, amt = df["volume"], df["amount"]
    turn = df["turnover_pct"].ffill(limit=3)
    out_sh = df["outstanding_share"].ffill(limit=5)
    r = c.pct_change()

    f = pd.DataFrame(index=df.index)
    f["mom12_1"] = c.shift(21) / c.shift(252) - 1.0
    f["mom1m"] = c / c.shift(21) - 1.0
    f["mom1w"] = c / c.shift(5) - 1.0
    f["mom6m"] = c / c.shift(126) - 1.0
    f["high52"] = c / c.rolling(252).max() - 1.0
    f["ret_max5"] = r.rolling(5).max()
    up = (r > 0).astype(float)
    f["up_streak"] = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)

    mkt = mkt_ret.reindex(df.index)
    beta = r.rolling(20).cov(mkt) / mkt.rolling(20).var().replace(0.0, np.nan)
    f["ivol20"] = (r - beta * mkt).rolling(20).std()
    f["vol20"] = r.rolling(20).std()
    f["max20"] = r.abs().rolling(20).max()
    f["dvol20"] = r.where(r < 0).rolling(20).std()
    f["skew20"] = r.rolling(20).skew()
    f["kurt20"] = r.rolling(20).kurt()
    v5, v60 = r.rolling(5).std(), r.rolling(60).std()
    f["vol_ratio"] = v5 / v60.replace(0.0, np.nan)

    f["turn20"] = turn.rolling(20).mean()
    f["turn_std20"] = turn.rolling(20).std()
    f["amihud20"] = (r.abs() / amt.replace(0.0, np.nan)).rolling(20).mean() * 1e9
    f["lndvol20"] = np.log(amt.rolling(20).mean().replace(0.0, np.nan))
    f["turn_trend"] = turn.rolling(5).mean() / turn.rolling(120).mean().replace(0.0, np.nan)
    f["turn_shock"] = turn / turn.rolling(20).mean().replace(0.0, np.nan)
    f["lnfloat"] = np.log((out_sh * c).replace(0.0, np.nan)).rolling(20).mean()

    f["pv_corr20"] = r.rolling(20).corr(turn.diff())
    net = np.sign(r) * v
    f["obv_slope"] = net.rolling(20).sum() / amt.replace(0.0, np.nan) * 1e8
    f["vr5_20"] = v.rolling(5).mean() / v.rolling(20).mean().replace(0.0, np.nan)

    delta14 = c.diff()
    gain = delta14.clip(lower=0).rolling(14).mean()
    loss = (-delta14.clip(upper=0)).rolling(14).mean().replace(0.0, np.nan)
    f["neg_rsi14"] = -(100 - 100 / (1 + gain / loss))
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["boll_pos"] = (c - (ma - 2 * sd)) / (4 * sd).replace(0.0, np.nan)
    f["upshd20"] = ((h - np.maximum(o, c)) / c).rolling(20).mean()
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()) / c
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    f["atr_pct"] = tr.rolling(14).mean() / c
    f["gapfreq20"] = ((o / c.shift() - 1.0).abs() > 0.01).rolling(20).mean()
    return f


def _wq_matrices(bars_map: dict[str, pd.DataFrame],
                 months: list[pd.Timestamp]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """把逐股序列装进 **矩阵**（行=交易日，列=股票），再算 WQ101 子集。

    为什么矩阵化：WQ 公式里 `rank()` 是截面算子（当日全市场排序）、`ts_*` 是
    时序算子——矩阵（行=日、列=股）让两类算子都变成一次向量化运算
    （这正是 qlib / 开源 WQ 实现的通用做法），比逐股循环快一个量级。
    """
    from core import wq101_alphas as wq

    fields = {}
    for f in ("open", "high", "low", "close", "volume", "amount"):
        fields[f] = pd.DataFrame({c: d[f].astype("float32")
                                  for c, d in bars_map.items()})
    F = dict(fields)
    F["returns"] = F["close"].pct_change()
    F["vwap"] = F["amount"] / F["volume"].replace(0.0, np.nan)
    F["adv20"] = F["amount"].rolling(20, min_periods=10).mean()

    wq_cols: dict[str, pd.DataFrame] = {}
    for name, fn in wq.ALPHAS.items():
        try:
            mat = fn(F)
            wq_cols[name] = mat.reindex(months, method="ffill")   # 先抽月度截面再 stack
        except Exception as e:                                    # noqa: BLE001
            _logger.warning("[wq] %s 计算失败 %s: %s", name, type(e).__name__, e)
    stacked = {}
    for name, mat in wq_cols.items():
        st = mat.stack()
        st.index = st.index.set_names(["date", "code"])
        stacked[name] = st
    wq = pd.DataFrame(stacked).reset_index()
    wq["code"] = wq["code"].astype(str)
    return wq, F


def _agg_factor(ok: pd.DataFrame, name: str, label: str, paper: str,
                hyp: int, horizon: int, min_rows: int) -> tuple[dict, "pd.Series | None"]:
    """单因子统计：月度 RankIC / ICIR / 五分位多空（**全部列运算，无索引对齐**）。

    Returns:
        (统计行, 月度多空收益序列)。序列是 Q5−Q1 的**月频**收益，专供 DSR/PBO 用
        （见 `_aggregate` 的多重检验段）——统计行本身不含序列，避免 JSON 膨胀。
    """
    ok = ok[["month", "fwd20", name]].rename(columns={name: "f"}).dropna()
    n_months = ok["month"].nunique()
    if n_months < min_rows:
        return ({"name": name, "label": label, "paper": paper, "hypothesis": hyp,
                 "verdict": "样本不足", "n_months": int(n_months)}, None)
    # RankIC：组内百分位排名后做皮尔逊 = 斯皮尔曼（列运算，避免重复索引 reindex）
    ok["rf"] = ok.groupby("month")["f"].rank(pct=True)
    ok["ff"] = ok.groupby("month")["fwd20"].rank(pct=True)

    def _corr(x: pd.DataFrame) -> float:
        return float(x["rf"].corr(x["ff"])) if len(x) >= 10 else np.nan

    ic = ok.groupby("month")[["rf", "ff"]].apply(_corr).dropna()
    mean, std = float(ic.mean()), float(ic.std())
    icir = mean / std if std else 0.0
    tstat = icir * math.sqrt(len(ic))
    ok["q"] = ok.groupby("month")["f"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
    qmean = ok.groupby(["month", "q"])["fwd20"].mean().unstack()
    spread = float((qmean[4] - qmean[0]).mean()) if 4 in qmean and 0 in qmean else np.nan
    by_year = {int(y): round(float(v), 4) for y, v in
               ic.groupby(ic.index.year).mean().items()}
    # 月度多空序列：DSR/PBO 的输入。缺季（q=0 或 4 不存在）时为 NaN，不当 0 处理。
    spread_series = (qmean[4] - qmean[0]) if (4 in qmean and 0 in qmean) else None
    return ({"name": name, "label": label, "paper": paper, "hypothesis": hyp,
             "ic_mean": round(mean, 4), "ic_ir": round(icir, 3),
             "t_stat": round(tstat, 2), "pos_pct": round(float((ic > 0).mean()), 3),
             "q5_q1_spread": round(spread, 4), "n_months": int(len(ic)),
             "by_year": by_year,
             # verdict 先按传统判据给，过 DSR 后由 _aggregate 覆写（见下）
             "verdict": ("有效" if abs(tstat) >= 2.0 and abs(mean) >= 0.02
                         else ("方向可用" if abs(tstat) >= 1.5 else "无效"))},
            spread_series)


def _load_bars_map() -> dict:
    """全市场逐股日线（本地优先，约 10 分钟）。"""
    from data import bars
    codes = sorted(bars.available_codes())
    bars_map: dict[str, pd.DataFrame] = {}
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        df = bars.get_bars(code, START, END)
        if df is None or len(df) < MIN_ROWS:
            continue
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        if len(df) >= MIN_ROWS:
            bars_map[code] = df
        if i % 1000 == 0:
            _logger.info("  加载 %d/%d（%.0fs）", i, len(codes), time.time() - t0)
    _logger.info("实际可用 %d 只（%.0fs）", len(bars_map), time.time() - t0)
    return bars_map


def main(argv=None) -> int:
    from data import bars
    codes = sorted(bars.available_codes())
    cal = bars.get_bars(codes[0], START, END)
    months = _sample_dates(pd.DatetimeIndex(pd.to_datetime(cal["date"])))
    _logger.info("月度截面 %d 个（%s ~ %s）", len(months), months[0].date(), months[-1].date())

    no_cache = "--no-cache" in (argv or sys.argv[1:])
    scan_cache = _ROOT / "temp" / "factor_scan_scan.parquet"
    wq_cache = _ROOT / "temp" / "factor_scan_wq.parquet"

    # ① 截面表（论文因子 + fwd20）
    if scan_cache.exists() and not no_cache:
        scan = pd.read_parquet(scan_cache)
        # ⚠️ 旧版缓存的月度时间戳列叫 "date"（reset_index 未改名）—— 规范化成 month_ts
        if "month_ts" not in scan.columns:
            scan = scan.rename(columns={"date": "month_ts"})
            scan["month_ts"] = pd.to_datetime(scan["month_ts"])
        _logger.info("[缓存] 截面表 %s（%d 行 / %d 只）", scan_cache, len(scan), scan["code"].nunique())
    else:
        bars_map = _load_bars_map()
        mkt = pd.concat({c: d["close"].pct_change() for c, d in bars_map.items()}
                        ).groupby(level=0).mean().sort_index()
        frames = []
        for i, (code, df) in enumerate(bars_map.items(), 1):
            try:
                f = _factor_frame(df, mkt)
                f["fwd20"] = df["close"].shift(-HORIZON) / df["close"] - 1.0
                f["code"] = code
                frames.append(f.reindex(months, method="ffill"))
            except Exception as e:                            # noqa: BLE001
                _logger.warning("[paper] %s 计算失败 %s: %s", code, type(e).__name__, e)
            if i % 2000 == 0:
                _logger.info("  论文因子 %d/%d", i, len(bars_map))
        scan = pd.concat(frames)
        scan["month"] = scan.index.to_period("M")
        scan = scan.reset_index()
        scan = scan.rename(columns={"date": "month_ts", "index": "month_ts"})
        scan["month_ts"] = pd.to_datetime(scan["month_ts"])
        scan.to_parquet(scan_cache, index=False)
        _logger.info("截面表 %d 行已缓存 %s", len(scan), scan_cache)

    # ② WQ101 子集（矩阵化；重算一次要 25 分钟 ⇒ 单独缓存）
    if wq_cache.exists() and not no_cache:
        wq = pd.read_parquet(wq_cache)
        if "month_ts" not in wq.columns:                 # 旧缓存列名规范化（同截面表）
            wq = wq.rename(columns={"date": "month_ts"})
        wq["month_ts"] = pd.to_datetime(wq["month_ts"])
        _logger.info("[缓存] WQ 因子表 %s（%d 行）", wq_cache, len(wq))
    else:
        bars_map = _load_bars_map()
        wq, _F = _wq_matrices(bars_map, months)
        wq.to_parquet(wq_cache, index=False)
        wq["month_ts"] = pd.to_datetime(wq["date"])
        _logger.info("WQ 因子表已缓存 %s（%d 行）", wq_cache, len(wq))

    # ③ 合并 + 聚合
    wq["month"] = wq["month_ts"].dt.to_period("M")
    keys = ("month_ts", "code", "month")
    dup = [c for c in wq.columns if c in scan.columns and c not in keys]
    scan = scan.drop(columns=dup, errors="ignore")
    scan = scan.merge(wq.drop(columns=["date"], errors="ignore"),
                      on=["month_ts", "code"], how="left")
    scan["month"] = pd.to_datetime(scan["month_ts"]).dt.to_period("M")
    return _aggregate(scan, months, guard="--no-guard" not in (argv or sys.argv[1:]))


def _aggregate(scan: pd.DataFrame, months: list, guard: bool = True) -> int:
    """逐因子统计（列运算，无索引对齐）；输出 JSON + 控制台排序表。

    `guard=True` 时套 `core.universe_guard.eligible`（上市满一年 + 北交所开市后）。
    ⚠️⚠️ 2026-10-31 实测：不加这条守卫，2019~2023 的流动性类因子 top 分组里
       北交所占 21%~55%，而 2019~2020 的 920xxx 是**新三板历史映射**——
       当期停牌、复牌跳空造出 +142%/月的伪收益。**不加守卫的 IC/DSR 不可用。**
    """
    if guard:
        from core.universe_guard import eligible
        before = len(scan)
        scan = scan[eligible(scan, month_col="month_ts")].copy()
        _logger.info("[守卫] 上市满一年 + 北交所开市后：%d → %d 行（剔除 %.1f%%）",
                     before, len(scan), 100.0 * (1 - len(scan) / max(before, 1)))

    # ── ③ 逐因子统计 ──
    results = []
    series_map: dict[str, "pd.Series"] = {}
    defs = {d[0]: d for d in FACTOR_DEFS}
    wq_names = [c for c in scan.columns if c.startswith("wq")]
    for name in [d[0] for d in FACTOR_DEFS] + wq_names:
        label, paper, hyp = (defs[name][1], defs[name][2], defs[name][3]) \
            if name in defs else (name, "WorldQuant 101 (Kakushadze 2016) 开源子集", -1)
        row, ser = _agg_factor(scan, name, label, paper, hyp,
                               HORIZON, MIN_FACTOR_ROWS)
        results.append(row)
        if ser is not None:
            series_map[name] = ser

    # ── ④ 多重检验校正（DSR + PBO）——逻辑在 core/factor_multiple_testing.py ──
    from core.factor_multiple_testing import adjust_by_multiple_testing
    mt = adjust_by_multiple_testing(results, series_map)
    _logger.info("多重检验校正：%d 个试验，PBO %s", mt.get("n_trials", 0),
                 (mt.get("pbo") or {}).get("pbo"))

    out_dir = _ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    (out_dir / f"因子论文扫描_30因子_{stamp}.json").write_text(
        json.dumps({"window": [SAMPLE_FROM, END], "horizon": HORIZON,
                    "stocks": int(scan["code"].nunique()), "months": len(months),
                    "multiple_testing": mt,
                    "results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    _logger.info("JSON 已写出：%s", out_dir / f"因子论文扫描_30因子_{stamp}.json")
    rows = sorted([r for r in results if "t_stat" in r],
                  key=lambda r: abs(r["t_stat"]), reverse=True)
    for r in rows:
        dsr = r.get("dsr") or {}
        print(f"{r['name']:>12} {r['label']:<10} IC={r['ic_mean']:+.3f} "
              f"ICIR={r['ic_ir']:+.2f} t={r['t_stat']:+.1f} "
              f"Q5-Q1={r['q5_q1_spread']:+.4f} "
              f"DSRp={dsr.get('p_value') if dsr.get('ok') else 'n/a'} {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
