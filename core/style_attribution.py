# -*- coding: utf-8 -*-
"""风格（因子）归因：r_p,t = α + Σβ_k·F_k,t + ε 的 OLS 分解（设计 3.3）。

因子集（2026-10-29，**能力边界照实写**，学 replay_eval 的做法）：
  - market      全市场等权收益（`core.market_factor.equal_weight_returns`，
                项目唯一市场因子来源，IC 建立在等权上）
  - volatility  低波−高波多空组合（面板 `volatility_20` 矩阵，方向 −1；
                与实盘用的 IVOL 同族——总波动 IC −0.068 vs IVOL −0.083 同号，
                IVOL 矩阵化待做后可直接替换）
  - **unavailable**（声明而非静默缺失）：
                规模（面板无市值历史，F2）、换手（turn 覆盖 29.6%）、
                估值（pe_ttm 待回填）
方法：因子模仿组合（每期 TOP/BOTTOM 分位等权多空）→ OLS 回归，
     t 值用 **Newey-West HAC**（默认滞后 5）——多空收益重叠期自相关，
     普通 OLS 的 t 值会被高估。
⚠️ 风险模型因子是**多空组合**：A 股无融券 ⇒ 只有解释意义（"组合涨跌有多少
   能被风格解释"），多头腿才是实盘可实现部分。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# 照实声明：这些风格**当前没有数据**（缺失原因见模块 docstring）
UNAVAILABLE_FACTORS = {
    "size": "面板无市值历史（16 列无 mktcap/industry，market_snapshot 仅 16 天）",
    "turnover": "面板 turn 覆盖率仅 29.6%",
    "value": "pe_ttm 依赖 BaoStock 回填（未完成）",
}
DEFAULT_NW_LAGS = 5


# ================================================================
# 因子模仿组合
# ================================================================

def factor_mimicking_returns(factor: pd.DataFrame, close: pd.DataFrame,
                             direction: int = 1, quantile: float = 0.2,
                             min_codes: int = 50,
                             mode: str = "forward") -> pd.Series | None:
    """因子模仿组合日收益 = TOP 分位收益 − BOTTOM 分位收益（等权）。

    Args:
        factor: 因子矩阵（date × code），只用收益窗口开始前已知的数据。
        direction: +1 因子越大越好；−1 越小越好（如波动率）。
        mode: **收益日口径**（⚠️ 必须与被归因的组合收益序列一致，
              错位一天会让 β 崩到 0——日收益近似白噪声，实测 2026-10-29）：
            - "forward"：T 日因子排序 → **T→T+1** 收益，标在 T
              （解释"这个信号之后赚多少"，walk-forward 语义）；
            - "same_day"：T−1 日因子排序 → **T 日收益**（close/close.shift(1)−1，
              标在 T）——已实现收益的风险归因用这个（style_attribution 默认）。
    Returns:
        日收益 Series（按日期升序）；截面太小/全空 ⇒ None（缺就是缺）。
    """
    if factor is None or factor.empty or close is None or close.empty:
        return None
    if mode == "forward":
        ret = close.shift(-1) / close - 1
        sort_factor = factor
    elif mode == "same_day":
        ret = close / close.shift(1) - 1
        sort_factor = factor.shift(1)                 # T 日收益只能用 T−1 日的因子排
    else:
        raise ValueError(f"mode 必须是 forward/same_day：{mode}")
    rows: dict = {}
    for d in sort_factor.index:
        s = sort_factor.loc[d].dropna()
        if len(s) < min_codes:
            continue
        k = max(int(len(s) * quantile), 5)
        if direction < 0:
            top_codes, bot_codes = s.nsmallest(k).index, s.nlargest(k).index
        else:
            top_codes, bot_codes = s.nlargest(k).index, s.nsmallest(k).index
        if d not in ret.index:
            continue
        rt = ret.loc[d].reindex(top_codes).dropna()
        rb = ret.loc[d].reindex(bot_codes).dropna()
        if len(rt) < k // 2 or len(rb) < k // 2:
            continue
        rows[d] = float(rt.mean() - rb.mean())
    if not rows:
        return None
    return pd.Series(rows, dtype="float64").sort_index()


# ================================================================
# OLS + Newey-West HAC
# ================================================================

def newey_west_ols(y: pd.Series, factors: dict[str, pd.Series],
                   lags: int = DEFAULT_NW_LAGS) -> dict:
    """带截距的 OLS，t 值用 Newey-West HAC 协方差（Bartlett 核）。

    Returns:
        {ok, n, alpha, factors: {name: {beta, t_stat}}, r_squared,
        degraded?: reason}
    """
    base = {"ok": False, "n": 0, "alpha": None,
            "factors": {k: {"beta": None, "t_stat": None} for k in factors},
            "r_squared": None}
    if not factors or y is None or len(y) < 30:
        return {**base, "degraded": f"样本不足（{0 if y is None else len(y)}<30）"}
    # 对齐：inner join 全部序列
    df = pd.DataFrame({"y": y.astype("float64")})
    for k, s in factors.items():
        df[k] = s.astype("float64")
    df = df.dropna()
    n = len(df)
    if n < 30:
        return {**base, "degraded": f"对齐后样本不足（{n}<30）"}
    names = list(factors.keys())
    X = np.column_stack([np.ones(n)] + [df[k].to_numpy() for k in names])
    yv = df["y"].to_numpy()
    # 完全共线检测（float64 下 np.linalg.inv 对奇异阵经常不抛异常而是返回
    # 天文数字的解——必须用秩显式检查）
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return {**base, "degraded": "因子共线（设计矩阵不满秩）"}
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ (X.T @ yv)
    e = yv - X @ b
    ss_res = float(e @ e)
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    # Newey-West：Γ_0 + Σ w_j(Γ_j + Γ_j')，Bartlett 权重 w_j = 1 − j/(lags+1)
    xe = X * e[:, None]
    V = (xe.T @ xe).astype("float64")
    for j in range(1, min(int(lags), n - 1) + 1):
        G = xe[j:].T @ xe[:-j]
        wj = 1.0 - j / (float(lags) + 1.0)
        V = V + wj * (G + G.T)
    se = np.sqrt(np.diag(xtx_inv @ V @ xtx_inv))
    out_factors = {}
    for i, k in enumerate(names):
        se_i = float(se[i + 1])
        out_factors[k] = {"beta": round(float(b[i + 1]), 6),
                          "t_stat": (round(float(b[i + 1]) / se_i, 3)
                                     if se_i > 0 else None)}
    return {"ok": True, "n": int(n), "alpha": round(float(b[0]), 8),
            "factors": out_factors,
            "r_squared": (round(r2, 4) if r2 is not None else None)}


# ================================================================
# 入口
# ================================================================

def style_attribution(portfolio_returns: pd.Series,
                      close: pd.DataFrame | None = None,
                      factor_matrices: dict[str, tuple[pd.DataFrame, int]] | None = None,
                      market_returns: pd.Series | None = None,
                      quantile: float = 0.2, nw_lags: int = DEFAULT_NW_LAGS,
                      min_codes: int = 50) -> dict:
    """风格暴露分解入口。

    ⚠️⚠️ **收益日口径纪律**：portfolio_returns 的"收益属于哪一天"必须与
    因子序列一致。 league/回测 nav 的日收益是**当天**收益（close/preclose
    口径）⇒ 因子用 same_day 模式（本函数默认）；walk-forward 的 forward
    语义才用 forward 模式。实测错位一天的后果：随机组合的 market β 从
    ≈+1 崩到 ≈0（日收益近似白噪声，错位后相关性消失）。
    ⚠️ 被归因组合**不能是因子模仿组合的机械子集**（如"高波动前 20%"与
    vol 因子的空头腿是同一个集合）——自引用会把 β 机械地定成 ±1
    （实测 β=−1.15、R²=0.73 全是假象）。真实组合（10~20 只）没问题。

    Args:
        portfolio_returns: 组合**日收益**序列（date → 小数）。
        factor_matrices: {因子名: (矩阵, 方向)}。None ⇒ 默认只构建 volatility
            （需要 close）；market 用 `market_returns`（缺省自动取
            `market_factor.equal_weight_returns`，当天收益口径）。
    Returns:
        {ok, factors: {name: {beta, t_stat, contrib_mean_pct}}, alpha,
        r_squared, n, unavailable: UNAVAILABLE_FACTORS, degraded?}
        contrib_mean_pct = β_k × F_k 均值 ×100 —— 该风格对组合平均日收益的贡献
        （百分点/日；年化 ×252 自行换算）。
    """
    # 1) 市场因子
    if market_returns is None:
        try:
            from core.market_factor import equal_weight_returns
            market_returns = equal_weight_returns()
        except Exception as e:                          # noqa: BLE001
            _logger.info("[风格归因] 市场因子取不到：%s", e)
    reg: dict[str, pd.Series] = {}
    if market_returns is not None and len(market_returns):
        reg["market"] = market_returns
    # 2) 风格因子（模仿组合，same_day 口径与 nav 类组合收益对齐）
    if factor_matrices:
        for name, (mat, direction) in factor_matrices.items():
            s = factor_mimicking_returns(mat, close, direction=direction,
                                         quantile=quantile, min_codes=min_codes,
                                         mode="same_day")
            if s is not None:
                reg[name] = s
            else:
                _logger.info("[风格归因] 因子 %s 模仿组合构建失败（截面不足），跳过", name)
    if not reg:
        return {"ok": False, "factors": {}, "alpha": None, "r_squared": None,
                "n": 0, "unavailable": UNAVAILABLE_FACTORS,
                "degraded": "没有任何可用的因子序列"}
    ols = newey_west_ols(portfolio_returns, reg, lags=nw_lags)
    out = {**ols, "unavailable": UNAVAILABLE_FACTORS}
    if ols.get("ok"):
        contrib = {}
        for k, st in ols["factors"].items():
            s = reg.get(k)
            mean_f = float(s.mean()) if s is not None and len(s) else 0.0
            contrib[k] = (round(st["beta"] * mean_f * 100, 6)
                          if st["beta"] is not None else None)
        out["contrib_mean_pct"] = contrib
    return out
