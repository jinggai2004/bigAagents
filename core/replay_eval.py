# -*- coding: utf-8 -*-
"""P1-2 历史回放评测 —— 在本地面板上重算因子并做「分年 × 分持有期」IC 与组合回放。

为什么要它（P1-2）：
    前向闭环（选出候选 → 等 5/20 天看结果）要**自然等待**才能积累样本，
    `ret_20d` 长期全缺、`done`=0 ⇒ "这套选股有没有用"这个问题拖了几周答不出来。
    面板已有 2016~2026 全市场日线，可以直接**回放**历史，一天出结论。

能力边界（必须随结果展示）：
    只有**价量 + 估值**能回放；「主力资金」「新闻正面率」没有历史序列
    （快照表只有 11 天）⇒ 回放覆盖的是**实盘六因子里可回放的那部分**。
    第一版默认只跑价量三因子（趋势/均线/波动率，实盘权重合计 41.17%）；
    估值因子要等 BaoStock 回填完成（pe_ttm 覆盖到位）再纳入。

前视防护（唯一硬约束）：
    因子只用 **T 日及之前**的数据；收益标签用 T+h 的收盘（`shift(-h)`）。
    `forward_returns` 是**唯一**允许碰"未来"的函数，且只用于标签。
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# 数据根必须**运行时**解析：打包后 `__file__` 位于只读包 `_internal/` 里，
# 用它推导会指向空目录 ⇒ 读到空数据（2026-10-25 实测：双击 exe 后
# 选股/复盘/研究全 missing、交易台变空账户、系统页看不到调度器）。
# 规则见 `config/paths.data_root()`；源码运行结果与原先一致。
from config.paths import data_root as _paths_data_root  # noqa: E402
PROJECT_ROOT = _paths_data_root()
PARTS_DIR = os.path.join(PROJECT_ROOT, "data", "panel", "parts")
# ⚠️ 缓存路径必须带 `days`：`_read_parts` 存的是**按 days 过滤后**的长表，
#    先跑 900 再跑 1400 会拿到缺更早数据的切片（静默错）。
_CACHE_TMPL = os.path.join(PROJECT_ROOT, "data", "panel",
                           "cache_replay_matrices_{days}.pkl")

# 只读这些列（pivot 后每个矩阵 = date × code）
_NEED = ["stock_code", "trade_date", "close", "preclose", "pct_chg",
         "amount", "open", "pe_ttm"]
_NEED_NO_PE = [c for c in _NEED if c != "pe_ttm"]   # 老分片没有估值列时的回退

# 截面覆盖下限（2026-10-19 审查 P1-1）：回填期间 10-19 截面只有 400 只
# （正常日 5563~5569），`min_names=100` 太松 ⇒ 残截面也能进 IC/组合样本。
# 只有 400 只的一天算出的"全市场等权"没有代表性。
DEFAULT_MIN_CROSS = 3000


# 可回放的因子 → 方向（+1 越大越好 / -1 越小越好）
REPLAY_FACTORS = {
    "momentum_20": +1,      # 20 日动量（实盘的「趋势」）
    "ma_align": +1,         # MA5/MA20-1（实盘的「均线排列」）
    "volatility_20": -1,    # 20 日年化波动率（实盘的「波动率」，越低越好）
    "amount_ratio": +1,     # 5 日均额 / 60 日均额（放量，独立观察，不进复合）
}

# 实盘权重（core/sector.FACTOR_WEIGHTS）中可回放三因子的占比 —— 用于归一复合分
PORTFOLIO_FACTORS = ("momentum_20", "ma_align", "volatility_20")
PORTFOLIO_WEIGHTS = {"momentum_20": 0.1765, "ma_align": 0.1176,
                     "volatility_20": 0.1176}


# ================================================================
# 1. 面板 → 宽表矩阵
# ================================================================

def _parts_signature(days: int) -> dict:
    """面板的**廉价指纹**：分片名/大小/mtime 的哈希（只 glob + stat，不读 parquet）。

    为什么需要（P1-1，2026-10-19 审查）：面板在 BaoStock 回填期间**一直在变**
    （parts 17→23→24，10-19 截面 400 只，pe_ttm 覆盖 12→1030→…）。
    没有指纹时，"同一命令跑出不同结论"无法区分**结论变了**还是**数据变了**。
    """
    import hashlib
    parts = sorted(glob.glob(os.path.join(PARTS_DIR, "*.parquet")))
    h = hashlib.sha1()
    total = 0
    newest = 0.0
    for p in parts:
        try:
            st = os.stat(p)
        except OSError as e:                                # noqa: BLE001
            _logger.debug("[replay] 指纹跳过不可读分片 %s: %s", p, e)
            continue
        h.update(f"{os.path.basename(p)}|{st.st_size}|{int(st.st_mtime)}".encode())
        total += st.st_size
        newest = max(newest, st.st_mtime)
    return {"days": int(days), "parts": len(parts), "bytes": total,
            "newest_mtime": newest,
            "signature": h.hexdigest()[:16]}


def _read_parts(days: int, use_cache: bool = True) -> pd.DataFrame | None:
    """读最近 `days` 个自然日的面板长表（列子集，逐 part 读后拼接）。

    ⚠️⚠️ 缓存**必须写**（P2-1，2026-10-19 审查）：原实现只有 `pickle.load`、
    全仓没有一处 `dump` ⇒ "30 秒"只在作者机器上成立（冷跑实测 135 秒）。
    现在：冷跑后落盘，命中时校验**面板指纹**（分片名/大小/mtime 哈希）——
    指纹不一致就重建，避免"缓存里是回填前的数据、结论却是回填后的标签"。
    """
    cache_path = _CACHE_TMPL.format(days=int(days))
    sig = _parts_signature(days)
    if use_cache and os.path.exists(cache_path):
        try:
            import pickle
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            # 兼容旧格式（裸 DataFrame）
            if isinstance(payload, pd.DataFrame):
                payload = {"df": payload, "signature": None}
            cached_sig = (payload.get("signature") or {}).get("signature")
            cached_df = payload.get("df")
            # ⚠️ 不能写 `payload.get("df") or []`：DataFrame 的 `__bool__` 会抛
            #    "truth value of a DataFrame is ambiguous" ⇒ 每次都被当成缓存损坏、
            #    白跑一次全量重建（实测 141s → 188s，日志里还只有一句 warning）。
            if cached_sig and cached_sig != sig["signature"]:
                _logger.info("[replay] 矩阵缓存已过期（面板变了 %s→%s），重建",
                             cached_sig, sig["signature"])
            elif isinstance(cached_df, pd.DataFrame) and len(cached_df) > 0:
                _logger.info("[replay] 矩阵缓存命中（%d 行，指纹 %s）",
                             len(cached_df), cached_sig)
                if not cached_df.attrs.get("panel_meta"):
                    cached_df.attrs["panel_meta"] = payload.get("meta") or {}
                return cached_df
        except Exception as e:                              # noqa: BLE001
            _logger.warning("[replay] 矩阵缓存不可用（重建）: %s", e)
        sig = _parts_signature(days)          # 读缓存期间面板可能又变了

    parts = sorted(glob.glob(os.path.join(PARTS_DIR, "*.parquet")))
    if not parts:
        return None
    start = (pd.Timestamp.now().normalize()
             - pd.Timedelta(days=int(days))).strftime("%Y-%m-%d")
    frames = []
    for p in parts:
        try:
            try:
                d = pd.read_parquet(p, columns=_NEED)
            except Exception as e:                          # noqa: BLE001
                # 早期分片没有估值列 ⇒ 退化为不带 pe_ttm 的列子集
                _logger.debug("[replay] 分片 %s 无 pe_ttm 列（回退列子集）: %s", p, e)
                d = pd.read_parquet(p, columns=_NEED_NO_PE)
        except Exception as e:                              # noqa: BLE001
            _logger.warning("[replay] 分片 %s 读取失败: %s", p, e)
            continue
        d["trade_date"] = d["trade_date"].astype(str).str[:10]
        d = d[d["trade_date"] >= start]
        if not d.empty:
            frames.append(d)
    if not frames:
        return None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        df = pd.concat(frames, ignore_index=True)
    df = (df.sort_values(["stock_code", "trade_date"])
            .drop_duplicates(["stock_code", "trade_date"], keep="last"))
    df = df[df["trade_date"] >= start]
    # 指纹（写进缓存 + 结果）：rows / 日期范围 / 股票数 / 估值覆盖
    date_max = str(df["trade_date"].max()) if len(df) else None
    pe_cov = None
    if "pe_ttm" in df.columns and date_max:
        last_slice = df[df["trade_date"] == date_max]
        pe_cov = int(last_slice["pe_ttm"].notna().sum())
    df.attrs["panel_meta"] = {
        **sig,
        "rows": int(len(df)),
        "date_min": str(df["trade_date"].min()) if len(df) else None,
        "date_max": date_max,
        "n_codes": int(df["stock_code"].nunique()) if len(df) else 0,
        "cross_section_at_max": (int((df["trade_date"] == date_max).sum())
                                 if date_max else 0),
        "pe_ttm_codes_at_max": pe_cov,
        "built_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if use_cache:
        try:
            import pickle
            tmp = cache_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"signature": sig, "df": df,
                             "meta": df.attrs["panel_meta"]}, f)
            os.replace(tmp, cache_path)
            _logger.info("[replay] 矩阵缓存已写入 %s（%d 行，指纹 %s）",
                         os.path.basename(cache_path), len(df), sig["signature"])
        except Exception as e:                              # noqa: BLE001
            _logger.warning("[replay] 矩阵缓存写入失败（不影响本次）: %s", e)
    return df


def _pivot_many(df: pd.DataFrame, cols) -> dict:
    """长表 → 宽表矩阵（`date × code`），一次 factorize + 逐列散射。

    等价于 `df.pivot(index="trade_date", columns="stock_code", values=col)`，
    前提是 (stock_code, trade_date) **唯一**（调用方已去重 keep="last"）。
    重复键时后者语义也是"最后一个" ⇒ 散射覆盖同键，行为一致。
    """
    t_codes, t_uniq = pd.factorize(df["trade_date"].to_numpy(), sort=True)
    c_codes, c_uniq = pd.factorize(df["stock_code"].to_numpy(), sort=True)
    shape = (len(t_uniq), len(c_uniq))
    idx = pd.Index(t_uniq, name="trade_date")
    col_idx = pd.Index(c_uniq, name="stock_code")
    out = {}
    for col in cols:
        if col not in df.columns:
            continue
        arr = np.full(shape, np.nan, dtype="float64")
        try:
            vals = df[col].to_numpy(dtype="float64", na_value=np.nan)
        except (TypeError, ValueError) as e:                # noqa: BLE001
            _logger.debug("[replay] 列 %s 非数值（逐值转换）: %s", col, e)
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(
                dtype="float64", na_value=np.nan)
        arr[t_codes, c_codes] = vals
        out[col] = pd.DataFrame(arr, index=idx, columns=col_idx)
    return out


@lru_cache(maxsize=2)
def load_matrices(days: int = 900) -> dict:
    """→ {"close"/"pct_chg"/"turn"/"amount": DataFrame(date × code)}。

    `days` 是**自然日**：默认 900 ≈ 600 个交易日（够算 20 日因子 + 2 年多回放）。
    """
    df = _read_parts(days)
    if df is None or df.empty:
        return {}
    meta = dict(getattr(df, "attrs", {}).get("panel_meta") or {})
    # ⚠️ 不用 `pivot_table`（groupby 聚合，5.27M 行 60s）也不用逐列 `pivot`
    #    （每列 7.7s × 5 = 38s）：`factorize` 一次 + 一次散射即可建好全部矩阵，
    #    实测 38s → ~4s。已按 (code, date) 去重 ⇒ 键唯一，散射无需聚合。
    out = _pivot_many(df, ("close", "open", "preclose", "pct_chg", "amount"))

    # ⚠️⚠️ **必须前复权**（2026-10-19 修）：面板存的是**未复权**价，
    # 除权日 close 腰斩（10 送 10）⇒ 用原始 close 算"20 日动量"和"5 日收益"
    # 会得到 -50% 的假跳空，把组合回放拖成"年化 -66%、回撤 -99%"（假象）。
    # 复权因子：`f[t] = ∏_{k>t} preclose[k]/close[k-1]`，锚定最后一天 f=1
    # （与 `data/bars._qfq` 同一推导，这里矩阵化）。
    if "close" in out and "preclose" in out:
        close, pre = out["close"], out["preclose"]
        ratio = (pre / close.shift(1)).replace([np.inf, -np.inf], np.nan)
        ratio = ratio.where(ratio > 0).fillna(1.0)
        f = ratio.iloc[::-1].shift(1).fillna(1.0).cumprod().iloc[::-1]
        out["close"] = close * f          # 前复权收盘（最后一天不变）
        out["adj_factor"] = f
    out.pop("preclose", None)
    out["meta"] = meta                    # 面板指纹（结果里必须带，见 P1-1）
    _logger.info("[replay] 矩阵就绪: %d 交易日 × %d 只（%s ~ %s，指纹 %s）",
                 len(out.get("close", [])),
                 out["close"].shape[1] if "close" in out else 0,
                 out["close"].index[0] if "close" in out else "?",
                 out["close"].index[-1] if "close" in out else "?",
                 meta.get("signature", "?"))
    return out


# ================================================================
# 2. 因子（只用 T 日及之前）
# ================================================================

def compute_factors(m: dict) -> dict:
    """价量因子矩阵（全部只用 T 日及之前的数据）。"""
    close = m.get("close")
    if close is None or close.empty:
        return {}
    pct = m.get("pct_chg")
    amount = m.get("amount")
    f = {}
    # 20 日动量
    f["momentum_20"] = close / close.shift(20) - 1
    # 均线排列强度（MA5/MA20-1；>0 即多头排列方向）
    ma5 = close.rolling(5, min_periods=5).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    f["ma_align"] = ma5 / ma20 - 1
    # 20 日年化波动率（pct_chg 是百分数 ⇒ /100）
    if pct is not None and not pct.empty:
        f["volatility_20"] = (pct / 100.0).rolling(20, min_periods=15).std() * np.sqrt(252)
    # 放量（5 日均额 / 60 日均额）
    if amount is not None and not amount.empty:
        a = amount.replace(0, np.nan)
        f["amount_ratio"] = a.rolling(5, min_periods=3).mean() / \
            a.rolling(60, min_periods=30).mean()
    return {k: v for k, v in f.items() if v is not None and not v.empty}


def _fwd_one(close: pd.DataFrame, horizon: int, entry: str = "close",
             open_: pd.DataFrame | None = None) -> pd.DataFrame:
    """单档前瞻收益。`entry="next_open"` ⇒ **T 日收盘出信号、T+1 开盘成交**。

    ⚠️ 项目自己的纪律（批次 3 已修过）："T 日收盘出信号 → **T+1 开盘价**成交"。
    回放如果按 `close.shift(-h)/close-1`（同日收盘成交）算，与实盘/回测的
    成交时点不一致 ⇒ "改善"里会混进"同日收盘 vs 次日开盘"的时点差。
    """
    h = int(horizon)
    if entry == "next_open" and open_ is not None and not open_.empty:
        entry_px = open_.shift(-1)                  # T+1 开盘买入
        exit_px = open_.shift(-(1 + h))             # T+1+h 开盘卖出
        return exit_px / entry_px - 1
    return close.shift(-h) / close - 1


def forward_returns(close: pd.DataFrame, horizons=(5, 10, 20),
                    entry: str = "close",
                    open_: pd.DataFrame | None = None) -> dict:
    """T 日的 h 日前瞻收益 —— **唯一允许读未来价的函数**（只作标签）。"""
    return {int(h): _fwd_one(close, h, entry=entry, open_=open_)
            for h in horizons}


# ================================================================
# 3. IC（截面 Spearman）+ bootstrap
# ================================================================

def _rank_ic(f: pd.Series, r: pd.Series, min_names: int = 100):
    """单期截面 Spearman IC；样本不足/方差为 0 ⇒ None（**不是 0**）。

    ⚠️ 用 numpy 版（排名 → Pearson）而不是 `Series.corr(method="spearman")`：
    后者每次调用都要 concat + 内部排序，2400 次调用实测 38s；这里 12s。
    数值上完全等价（Spearman = 秩的 Pearson，平均秩处理一致），
    差异在 1e-12 量级（有对照用例钉住）。
    """
    if not f.index.equals(r.index):
        r = r.reindex(f.index)
    fv = f.to_numpy(dtype="float64", na_value=np.nan)
    rv = r.to_numpy(dtype="float64", na_value=np.nan)
    mask = ~(np.isnan(fv) | np.isnan(rv))
    if int(mask.sum()) < min_names:
        return None
    a, b = fv[mask], rv[mask]
    if len(np.unique(a)) < 5 or len(np.unique(b)) < 5:
        return None
    ra = pd.Series(a).rank().to_numpy(dtype="float64")
    rb = pd.Series(b).rank().to_numpy(dtype="float64")
    da = ra - ra.mean()
    db = rb - rb.mean()
    den = float(np.sqrt((da * da).sum() * (db * db).sum()))
    if den <= 0 or not np.isfinite(den):
        return None
    return float((da * db).sum() / den)


def ic_series(factor: pd.DataFrame, fwd: pd.DataFrame, step: int = 5,
              min_names: int = 100) -> pd.Series:
    """逐期 IC 序列（每 `step` 个交易日取一期）。"""
    vals = {}
    for i in range(0, len(factor), step):
        d = factor.index[i]
        if d not in fwd.index:
            continue
        v = _rank_ic(factor.loc[d], fwd.loc[d], min_names=min_names)
        if v is not None:
            vals[d] = v
    return pd.Series(vals, dtype="float64")


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 42) -> dict:
    """均值的 bootstrap 置信区间（**样本 < 10 期 ⇒ 不给 CI**，只报 None）。"""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)],
                     dtype="float64")
    if len(arr) == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "significant": None}
    mean = float(arr.mean())
    if len(arr) < 10:
        return {"n": int(len(arr)), "mean": mean, "lo": None, "hi": None,
                "significant": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(int(n_boot), len(arr)))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return {"n": int(len(arr)), "mean": mean, "lo": lo, "hi": hi,
            "significant": bool(lo > 0 or hi < 0)}


def ic_table(factors: dict, fwd: dict, step: int = 5,
             min_names: int = 100) -> dict:
    """→ {factor: {h: {"all": {...}, "by_year": {2024: {...}}}}}"""
    out = {}
    for name, f in factors.items():
        out[name] = {}
        for h, r in fwd.items():
            s = ic_series(f, r, step=step, min_names=min_names)
            by_year = {}
            for year, grp in s.groupby(pd.to_datetime(s.index).year):
                by_year[int(year)] = bootstrap_ci(grp.values)
            out[name][h] = {
                "all": bootstrap_ci(s.values),
                "by_year": by_year, "n_periods": int(len(s)),
                # ⚠️ h > step ⇒ 相邻期的标签重叠（共享 h-step 天收益）⇒ 期之间自相关，
                #    iid bootstrap 的 CI **偏窄**（"显著"被高估）。方向可信，强度打折。
                "overlapping": int(h) > int(step),
            }
    return out


# ================================================================
# 4. 组合回放（TOP N 等权，逐期复利）
# ================================================================

def composite_score(factors: dict, weights: dict | None = None) -> pd.DataFrame:
    """复合因子：可回放因子按权重归一后加权（方向已对齐）。

    每个因子先做截面 z-score（按日），再加权求和 —— 与实盘"横截面分位化"
    同源（都是"在当日全市场里排第几"），但这里是 z 而不是百分位，
    只影响尺度、不影响排序。

    Args:
        factors: 因子矩阵（date × code）。
        weights: {因子名: 权重}。None ⇒ `PORTFOLIO_WEIGHTS`（实盘三因子，
            与旧行为完全一致——walk-forward 之前唯一的调用方式）。
            ⚠️ 权重 ≤0 的因子不进复合（walk-forward 的 ICIR 加权用 0 表达
            "该窗口无区分度"，而不是硬塞一个负权重进去）。
    """
    w = {k: float(v) for k, v in (weights or PORTFOLIO_WEIGHTS).items()
         if float(v) > 0} if weights is not None else dict(PORTFOLIO_WEIGHTS)
    parts, wts = [], {}
    for name in PORTFOLIO_FACTORS:
        if name not in w:
            continue
        f = factors.get(name)
        if f is None or f.empty:
            continue
        z = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1).replace(0, np.nan), axis=0)
        parts.append(z * REPLAY_FACTORS.get(name, 1))    # 方向对齐
        wts[name] = w[name]
    if not parts:
        return pd.DataFrame()
    items = list(wts.items())                             # 与 parts 同序
    total = sum(w for _, w in items)
    score = parts[0] * (items[0][1] / total)
    for arr, (_, wgt) in zip(parts[1:], items[1:]):
        score = score + arr * (wgt / total)
    return score


def portfolio_backtest(score: pd.DataFrame, close: pd.DataFrame,
                       step: int | None = None, horizon: int = 5,
                       top_n: int = 20, index_close: pd.Series | None = None,
                       entry: str = "close",
                       open_: pd.DataFrame | None = None) -> dict:
    """逐期取 TOP N 等权持有 `horizon` 日；与"全市场等权"（可选沪深300）对比。

    ⚠️⚠️ `step` 默认 = `horizon`（**必须如此**）：step < horizon 时各期持仓重叠，
    逐期复利会把同一段收益重复计入 —— 2026-10-19 实测 step=5 + horizon=60 时
    "年化 -98%" 就是这么来的（重叠 12 倍），**假的**。
    要观察重叠期的平均超额，请直接看 `excess_mean`，不要看复利年化。
    """
    step = int(step or horizon)
    fwd = _fwd_one(close, horizon, entry=entry, open_=open_)
    rows = []
    for i in range(0, len(score), step):
        d = score.index[i]
        if d not in fwd.index:
            continue
        s = score.loc[d].dropna()
        r = fwd.loc[d]
        if len(s) < top_n:
            continue
        top = s.nlargest(top_n).index
        r_top = r.reindex(top).dropna()
        r_all = r.dropna()
        if len(r_top) < top_n // 2 or len(r_all) < 100:
            continue
        rows.append({"date": d, "top_ret": float(r_top.mean()),
                     "all_ret": float(r_all.mean()),
                     "excess": float(r_top.mean() - r_all.mean()),
                     "n_all": int(len(r_all))})
    if not rows:
        return {"available": False}
    df = pd.DataFrame(rows).set_index("date")
    equity = (1 + df["top_ret"]).cumprod()
    bench = (1 + df["all_ret"]).cumprod()
    years = max((pd.to_datetime(df.index[-1]) - pd.to_datetime(df.index[0])).days / 365.25, 1e-9)
    ann = float(equity.iloc[-1] ** (1 / years) - 1)
    bench_ann = float(bench.iloc[-1] ** (1 / years) - 1)
    index_ann = None
    if index_close is not None and len(index_close):
        ic_close = index_close.reindex(pd.Index(df.index.map(str)))
        ic_close = ic_close.astype("float64")
        if ic_close.notna().sum() >= 2:
            ic_close = ic_close.dropna()
            yrs_idx = max((pd.to_datetime(ic_close.index[-1])
                           - pd.to_datetime(ic_close.index[0])).days / 365.25, 1e-9)
            index_ann = float((ic_close.iloc[-1] / ic_close.iloc[0])
                              ** (1 / yrs_idx) - 1)
    roll_max = equity.cummax()
    mdd = float((equity / roll_max - 1).min())
    by_year = {}
    for year, grp in df.groupby(pd.to_datetime(df.index).year):
        by_year[int(year)] = {"excess_mean": float(grp["excess"].mean()),
                             "win_rate": float((grp["excess"] > 0).mean()),
                             "periods": int(len(grp))}
    return {
        "available": True, "horizon": horizon, "step": step, "top_n": top_n,
        "entry": entry,
        "periods": int(len(df)), "years": round(years, 2),
        "ann_return": ann, "bench_ann": bench_ann,
        "index_ann": index_ann,          # 沪深300 同期年化（可选）
        "ann_excess": ann - bench_ann, "max_drawdown": mdd,
        "win_rate": float((df["excess"] > 0).mean()),
        "excess_mean": float(df["excess"].mean()),
        "excess_ci": bootstrap_ci(df["excess"].values),
        "by_year": by_year,
        "equity_tail": equity.tail(3).round(4).to_dict(),
    }


# ================================================================
# 4b. 过热诊断（2026-10-19 首次回放的决定性发现）
# ================================================================

def _top_bottom_returns(score: pd.DataFrame, fwd: pd.DataFrame, step: int,
                        top_n: int, mask: pd.DataFrame | None = None) -> dict:
    """按 score 取 TOP/BOTTOM N 的逐期平均收益（可选 mask 过滤候选）。"""
    rows = []
    for i in range(0, len(score), step):
        d = score.index[i]
        if d not in fwd.index:
            continue
        s = score.loc[d].dropna()
        if mask is not None and d in mask.index:
            m = mask.loc[d].reindex(s.index).fillna(False).astype(bool)
            s = s[m]
        r = fwd.loc[d]
        if len(s) < top_n:
            continue
        rt = r.reindex(s.nlargest(top_n).index).dropna()
        rb = r.reindex(s.nsmallest(top_n).index).dropna()
        ra = r.dropna()
        if len(rt) < top_n // 2 or len(ra) < 100:
            continue
        rows.append({"top": float(rt.mean()), "bot": float(rb.mean()),
                     "all": float(ra.mean())})
    if not rows:
        return {"available": False}
    df = pd.DataFrame(rows)
    return {"available": True, "periods": int(len(df)),
            "top_mean": float(df["top"].mean()),
            "bot_mean": float(df["bot"].mean()),
            "all_mean": float(df["all"].mean()),
            "top_excess": float((df["top"] - df["all"]).mean()),
            "bot_excess": float((df["bot"] - df["all"]).mean()),
            "top_win": float((df["top"] > df["all"]).mean())}


def overheat_diagnosis(m: dict, score: pd.DataFrame, horizon: int = 5,
                       step: int = 5, top_n: int = 20,
                       entry: str = "close") -> dict:
    """同一条复合分，在**不同"过热过滤"下**的 TOP 表现对比。

    为什么需要它（2026-10-19 实测）：趋势/动量类因子在 TOP 端选出的是
    "刚大涨的票"，A 股 5~20 日存在强反转 ⇒ 原始 TOP20 每期 -1.95%，
    而**剔除近 20 日涨幅 >15%** 后变成 +0.02%（超额从 -2.19% 收到 -0.22%）。
    两端都跑输（BOTTOM 也负超额）⇒ 因子在极端分位无效，不是单纯方向反了。
    """
    close = m["close"]
    fwd = _fwd_one(close, horizon, entry=entry, open_=m.get("open"))
    pct = m.get("pct_chg")
    rets20 = close / close.shift(20) - 1
    cases = {"raw": None}
    cases["ex_mom30"] = (rets20 < 0.30)
    cases["ex_mom15"] = (rets20 < 0.15)
    if pct is not None:
        cases["ex_limit_up"] = ((pct < 9.0) | pct.isna())
        cases["ex_mom15_and_limit"] = ((rets20 < 0.15) & ((pct < 9.0) | pct.isna()))
    out = {k: _top_bottom_returns(score, fwd, step=step, top_n=top_n, mask=v)
           for k, v in cases.items()}
    base = out.get("raw") or {}
    for k, v in out.items():
        if v.get("available") and base.get("available"):
            v["improvement_vs_raw"] = v["top_excess"] - base["top_excess"]
    return {"horizon": horizon, "step": step, "top_n": top_n,
            "entry": entry, "cases": out}


# ================================================================
# 4c. 截面覆盖过滤（P1-1，2026-10-19 审查）
# ================================================================

def drop_thin_dates(m: dict, min_cross: int | None = None) -> tuple[dict, dict]:
    """剔除**截面覆盖不足**的交易日（回填期间的残截面）。

    为什么必须剔：回填当天的 10-19 只有 **400 只**（正常日 5563~5569），
    在 400 只上算出的"全市场等权"和截面 IC 没有代表性；它还会让最后几期
    的 IC/超额随机跳动，看起来像"结论变了"。

    Returns:
        (过滤后的 m, {"min_cross": N, "dropped": [...], "kept": n})
    """
    close = m.get("close")
    if close is None or close.empty:
        return m, {"min_cross": 0, "dropped": [], "kept": 0}
    n_codes = int(close.shape[1])
    # 自适应：小样本（测试/小子集）下不要把自己剔光
    adaptive = max(20, int(0.6 * n_codes))
    thr = min(int(min_cross or DEFAULT_MIN_CROSS), adaptive)
    counts = close.notna().sum(axis=1)
    keep = counts >= thr
    dropped = [str(d) for d in counts.index[~keep]]
    if not dropped:
        return m, {"min_cross": thr, "dropped": [], "kept": int(len(close))}
    _logger.warning("[replay] 剔除 %d 个截面覆盖不足的交易日（<%d 只）: %s%s",
                    len(dropped), thr, ", ".join(dropped[:5]),
                    " ..." if len(dropped) > 5 else "")
    out = {}
    for k, v in m.items():
        out[k] = v.loc[keep] if isinstance(v, pd.DataFrame) else v
    return out, {"min_cross": thr, "dropped": dropped, "kept": int(keep.sum())}


# ================================================================
# 5. 一键评测
# ================================================================

@dataclass
class ReplayResult:
    days: int
    date_start: str
    date_end: str
    n_codes: int
    n_days: int
    factors: list
    ic: dict
    portfolio: dict
    caveats: list = field(default_factory=list)
    overheat: dict = field(default_factory=dict)
    # 数据指纹（P1-1，2026-10-19 审查）：没有它，"结论变了"和"数据变了"分不开
    fingerprint: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"days": self.days, "date_start": self.date_start,
                "date_end": self.date_end, "n_codes": self.n_codes,
                "n_days": self.n_days, "factors": self.factors,
                "ic": self.ic, "portfolio": self.portfolio,
                "caveats": self.caveats, "overheat": self.overheat,
                "fingerprint": self.fingerprint}


def evaluate(days: int = 900, step: int = 5, horizons=(5, 10, 20),
             top_n: int = 20, entry: str = "close",
             min_cross: int | None = None) -> ReplayResult:
    """跑一次完整回放（矩阵 → 因子 → IC → 组合）。

    Args:
        entry: `"close"`（T 日收盘成交，旧口径）/ `"next_open"`
            （**T 日收盘出信号、T+1 开盘成交**，与项目回测纪律一致）。
        min_cross: 截面覆盖下限（少于该只数的交易日直接剔除）。
    """
    m = load_matrices(days)
    if not m or "close" not in m:
        raise RuntimeError("面板矩阵不可用（检查 data/panel/parts）")
    meta = dict(m.get("meta") or {})
    m, cover = drop_thin_dates(m, min_cross=min_cross)
    close = m["close"]
    factors = compute_factors(m)
    if not factors:
        raise RuntimeError("因子计算失败（面板价量列缺失）")
    fwd = forward_returns(close, horizons, entry=entry, open_=m.get("open"))

    # ⚠️ IC 只用"因子已成熟"的区间：20 日因子前 20 天全 NaN
    ic = ic_table(factors, fwd, step=step)

    score = composite_score(factors)
    # 沪深300 基准（本地指数面板，零联网；缺失就不报）
    idx_close = None
    try:
        import data.index_bars as _ib
        _idx = _ib.get_index_bars("sh000300")
        if _idx is not None:
            idx_close = _idx.set_index(_idx["date"].dt.strftime("%Y-%m-%d"))["close"]
    except Exception as e:                                  # noqa: BLE001
        _logger.debug("[replay] 沪深300 基准不可用: %s", e)
    port = {}
    for h in horizons:
        # step 跟随 horizon（不重叠），IC 那边才按 step 采样
        port[int(h)] = portfolio_backtest(score, close, step=int(h),
                                          horizon=int(h), top_n=top_n,
                                          index_close=idx_close,
                                          entry=entry, open_=m.get("open"))

    caveats = [
        "只覆盖**价量因子**（趋势/均线/波动率，实盘权重合计 41.17%）；"
        "「主力资金」「新闻正面率」无历史序列，不在回放范围内",
        "「估值(PE)」需 pe_ttm 历史，BaoStock 回填进行中 ⇒ 第一版未纳入",
        "IC 为**截面 Spearman**，逐 %d 个交易日一期；样本 < min_names 的期已跳过" % step,
        # ⚠️ P2-3：h > step 时标签重叠 ⇒ 相邻期共享收益 ⇒ iid bootstrap CI 偏窄。
        #    "方向"（五年一致为负）可信，"显著程度"要打折。
        "IC 的 bootstrap CI 按**独立样本**计算：h=20/60 与采样步长 %d 不等时"
        "标签重叠（相邻期共享收益），CI **偏窄**、「显著」被高估 —— 方向可信，强度打折" % step,
        "组合回放为逐期独立（step=horizon），收益可复利；入场时点 = %s" % entry,
        "面板含退市股（幸存者偏差已消除），但停牌期间收益按 0 处理",
        "价格已前复权（preclose/close 比值链，锚定样本末日）——"
        "否则除权日会出 -50% 假跳空",
    ]
    if cover["dropped"]:
        caveats.append(
            f"已剔除 {len(cover['dropped'])} 个**截面覆盖不足**的交易日"
            f"（少于 {cover['min_cross']} 只）：{', '.join(cover['dropped'][:5])}"
            + (" ..." if len(cover["dropped"]) > 5 else ""))
    overheat = overheat_diagnosis(m, score, horizon=min(horizons),
                                  step=step, top_n=top_n, entry=entry)
    fingerprint = {
        "panel_signature": meta.get("signature"),
        "parts": meta.get("parts"),
        "panel_rows": meta.get("rows"),
        "panel_date_max": meta.get("date_max"),
        "panel_cross_section_at_max": meta.get("cross_section_at_max"),
        "panel_pe_ttm_codes_at_max": meta.get("pe_ttm_codes_at_max"),
        "panel_built_at": meta.get("built_at"),
        "sample_date_start": str(close.index[0]),
        "sample_date_end": str(close.index[-1]),
        "sample_days": int(len(close)),
        "sample_codes": int(close.shape[1]),
        "entry": entry,
        "min_cross": cover["min_cross"],
        "dropped_dates": len(cover["dropped"]),
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return ReplayResult(
        days=days, date_start=str(close.index[0]), date_end=str(close.index[-1]),
        n_codes=int(close.shape[1]), n_days=int(close.shape[0]),
        factors=list(factors), ic=ic, portfolio=port, caveats=caveats,
        overheat=overheat, fingerprint=fingerprint)


# ================================================================
# 5. Walk-forward 权重重估（B 模块，2026-10-29）
# ================================================================

def _icir_weights(factors: dict, fwd_row: pd.DataFrame, dates_mask: pd.Series,
                  step: int, min_names: int, min_periods: int = 8) -> dict:
    """训练窗内逐因子 IC → ICIR 加权（负 ICIR ⇒ 0："该窗口无区分度"）。

    ⚠️ 方向沿用 REPLAY_FACTORS（语义方向），ICIR 只定大小不定方向——
    一个"语义为低波好"的因子若 ICIR<0，说明这个窗口里它没区分度，
    权重归 0，而不是反向押注（反向是另一套需要单独证据的决策）。
    """
    w: dict[str, float] = {}
    for name in PORTFOLIO_FACTORS:
        f = factors.get(name)
        if f is None or f.empty:
            continue
        s = ic_series(f[dates_mask], fwd_row[dates_mask], step=step,
                      min_names=min_names)
        if len(s) < min_periods or float(s.std(ddof=1)) <= 0:
            w[name] = 0.0
            continue
        w[name] = max(float(s.mean() / s.std(ddof=1)), 0.0)
    total = sum(w.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in w.items()}


def walk_forward_refit(factors: dict, fwd: dict, horizon: int = 5,
                       train_years: int = 4, step: int = 5, top_n: int = 20,
                       min_names: int = 100) -> dict:
    """滚动"训 N 年 → 测 1 年"重估权重，样本外对比固定实盘权重（B 模块）。

    每折：
      1) 训练窗（test 年之前的 `train_years` 个自然年）内逐因子 IC → ICIR 加权；
      2) 测试年用重估权重算 composite_score → TOP-N 超额 + 复合分 OOS IC；
      3) 同一测试年用**固定 PORTFOLIO_WEIGHTS** 跑同口径基线。
    汇总：重估 vs 基线的逐折 OOS 超额/IC 对比——FACTOR_WEIGHTS 的任何改动
    都应附这份对比（`docs/WEIGHT_CHANGES.md` 模板），"不能用 1 年样本改
    权重"从纪律变成机制。

    ⚠️ 前视防护：折权重只用 `test_year` 之前的数据（年份切片硬隔离）；
       fwd 只作标签（与 replay_eval 全模块同一纪律）。

    Args:
        factors: 因子矩阵（date × code；来自 compute_factors 或直接注入——
                 后者供合成数据测试用，不依赖本地面板）。
        fwd: {horizon: 前瞻收益矩阵}（forward_returns 的输出）。
        horizon: 取 fwd 的哪一档做评估标签。

    Returns:
        {folds: [...], summary: {...}}；JSON 安全（无 numpy 类型）。
    """
    names = [f for f in PORTFOLIO_FACTORS if f in factors and not factors[f].empty]
    if not names or horizon not in fwd:
        return {"folds": [], "summary": {"available": False,
                "reason": "缺少可回放因子或该 horizon 的前瞻收益"}}
    f0 = factors[names[0]]
    years = sorted({int(y) for y in pd.to_datetime(f0.index).year})
    fwd_h = fwd[horizon]
    folds: list[dict] = []
    for i in range(train_years, len(years)):
        test_year = years[i]
        train_list = years[max(0, i - train_years):i]
        idx = pd.to_datetime(f0.index)
        train_mask = idx.year.isin(train_list)
        test_mask = idx.year == test_year
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        w_refit = _icir_weights(factors, fwd_h, train_mask, step, min_names)
        w_base = {f: PORTFOLIO_WEIGHTS[f] for f in names if f in PORTFOLIO_WEIGHTS}
        fold = {"test_year": int(test_year), "train_years": [int(y) for y in train_list],
                "weights_refit": {k: round(v, 4) for k, v in w_refit.items()},
                "weights_baseline": w_base}
        for tag, w in (("refit", w_refit), ("baseline", w_base)):
            if not w:
                fold[f"oos_{tag}"] = {"available": False, "reason": "训练窗无有效权重"}
                continue
            score_t = composite_score({f: factors[f][test_mask] for f in factors},
                                      weights=w)
            oos = _top_bottom_returns(score_t, fwd_h[test_mask], step=step,
                                      top_n=top_n)
            ic = ic_series(score_t, fwd_h[test_mask], step=step, min_names=min_names)
            entry = {"available": bool(oos.get("available")),
                     # top_excess = TOP-N 平均收益 − 全市场等权平均（逐期均差）
                     "top_excess_mean": oos.get("top_excess") if oos.get("available") else None,
                     "top_mean": oos.get("top_mean") if oos.get("available") else None,
                     "periods": oos.get("periods") if oos.get("available") else 0,
                     "composite_ic_mean": (round(float(ic.mean()), 5) if len(ic) else None)}
            fold[f"oos_{tag}"] = entry
        folds.append(fold)

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 5) if vals else None

    refit_excess = [f["oos_refit"]["top_excess_mean"] for f in folds
                    if f.get("oos_refit", {}).get("available")]
    base_excess = [f["oos_baseline"]["top_excess_mean"] for f in folds
                   if f.get("oos_baseline", {}).get("available")]
    refit_ic = [f["oos_refit"]["composite_ic_mean"] for f in folds
                if f.get("oos_refit", {}).get("composite_ic_mean") is not None]
    base_ic = [f["oos_baseline"]["composite_ic_mean"] for f in folds
               if f.get("oos_baseline", {}).get("composite_ic_mean") is not None]
    summary = {
        "available": bool(folds),
        "horizon": int(horizon), "train_years": int(train_years),
        "n_folds": len(folds),
        "top_excess_mean_refit": _mean(refit_excess),
        "top_excess_mean_baseline": _mean(base_excess),
        "composite_ic_mean_refit": _mean(refit_ic),
        "composite_ic_mean_baseline": _mean(base_ic),
        "note": "重估优于基线 ⇒ 权重重估有 OOS 证据；反之固定权重已够，"
                "不要因为单年 IC 好看就改 FACTOR_WEIGHTS",
    }
    return {"folds": folds, "summary": summary}


def walk_forward_from_market(m: dict, horizons=(5,), train_years: int = 4,
                             step: int = 5, top_n: int = 20,
                             min_names: int = 100) -> dict:
    """从行情矩阵出发的 walk-forward（`evaluate` 同源的因子/标签入口）。"""
    factors = compute_factors(m)
    close = m.get("close")
    if close is None or close.empty or not factors:
        return {"folds": [], "summary": {"available": False, "reason": "行情矩阵不足"}}
    out = {}
    for h in horizons:
        fwd = forward_returns(close, horizons=(h,))
        out[int(h)] = walk_forward_refit(factors, fwd, horizon=int(h),
                                         train_years=train_years, step=step,
                                         top_n=top_n, min_names=min_names)
    return out if len(horizons) > 1 else out[int(horizons[0])]
