# -*- coding: utf-8 -*-
"""因子集的多重检验校正：DSR（单因子）+ PBO/CSCV（因子集整体）。

为什么需要这个模块
──────────────────
"扫了 61 个因子、37 个 |t|≥2"这句话本身不是结论。61 次试验下，即使全部是噪音，
按 5% 显著性水平也应有约 3 个"显著"。不做校正就把 t 值当结论，正是本研究
自身要防的那类错误。

与 `core/backtest_validate.py` 的分工
─────────────────────────────────────
- `backtest_validate` 提供**公式**（`deflated_sharpe` / `pbo` / `expected_max_sharpe`），
  面向"同一策略的多组参数"；
- 本模块负责**因子集**这一场景的接线：把每个因子的月频多空收益序列变成
  一次试验，用全部试验的 SR 分布算 hurdle（SR*），再逐因子判显著。

口径（三条，缺一条结果就会失真）
────────────────────────────────
1. **月频**：收益序列是月频多空收益（Q5−Q1），SR 也是月频。PSR/DSR 对采样
   频率不敏感——SR 与 T 同频即可（此处 T=月数）。
2. **方向对齐**：IC 为负的因子，Q5−Q1 本身就是负收益（"反过来做才对"）。
   方向是**样本内用均值符号选定**的 1 bit，所以先按 `sign(mean)` 翻正再检验，
   试验分布也用 |SR|。
   ⚠️ 不翻正会把全场最强的 lndvol20（SR=−0.69）判成 p=1.0 —— 这是接线错误，
   不是"发现该因子无效"（2026-10-30 首版实测踩到）。
3. **PBO 用未翻正的原始序列**：翻正需要全样本均值信息，用它做 CSCV 会泄露未来。

已知局限
────────
`sr_trials` 是 59 个**互不相同的因子**的 SR，其离散度里混有真实差异（不只噪音），
所以 SR* 偏保守（宁可错杀）。这与 `backtest_validate` 模块 docstring 里
"DSR 宁可错杀"的取向一致，是**有意**的。
"""
from __future__ import annotations

import math

import pandas as pd

from core.backtest_validate import deflated_sharpe, expected_max_sharpe, pbo

MIN_SERIES_LEN = 20      # 少于 20 个月无法估计 SR 的稳定性
PBO_MIN_FACTORS = 8      # CSCV 的配置数下限（与 backtest_validate.pbo 一致）
PBO_MIN_MONTHS = 24      # CSSV 时间块下限（s_blocks=16 时每块 ≥1.5 月）
PBO_COVERAGE = 0.95      # 因子覆盖率门槛：低于此值的因子不进 PBO 矩阵


def _monthly_sharpe(series: pd.Series) -> float | None:
    """月频夏普（ddof=1）。样本不足或零方差返回 None。"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < MIN_SERIES_LEN:
        return None
    sd = float(s.std(ddof=1))
    if sd <= 0:
        return None
    return float(s.mean()) / sd


def adjust_by_multiple_testing(results: list[dict],
                               series_map: dict[str, pd.Series]) -> dict:
    """就地给每个因子行补 `dsr` / `ls_direction` / `sr_monthly` / `sr_annual`，
    并按 DSR 覆写 `verdict`。

    Args:
        results: 每个元素是一个因子统计行，**会被就地修改**（要求有 `name`；
            有 `t_stat` 的行才做检验，"样本不足"行跳过）。
        series_map: {因子名: 月频多空收益 Series}（index=月度，可含 NaN）。

    Returns:
        {ok, n_trials, sign_aligned, sr_star_monthly, pbo}
        - `sr_star_monthly` 是 hurdle 本身（月频），×√12 即年化可比。
        - `pbo` 为 `backtest_validate.pbo()` 的原样字典，外加
          factors_used / months_used 两个口径字段。
    """
    usable = [n for n, s in (series_map or {}).items()
              if s is not None and len(s) >= MIN_SERIES_LEN]
    if not usable:
        return {"ok": False, "reason": "无足够长度的多空序列，未做校正"}

    mat = pd.DataFrame({n: series_map[n] for n in usable}).sort_index()
    # 试验分布 = |月频 SR|（方向已在样本内选定，见模块 docstring 第 2 条）
    sr_by_factor: dict[str, float] = {}
    for n in usable:
        sr = _monthly_sharpe(mat[n])
        if sr is not None:
            sr_by_factor[n] = abs(sr)
    sr_trials = list(sr_by_factor.values())

    for row in results:
        name = row.get("name")
        if "t_stat" not in row:
            continue
        ser = series_map.get(name)
        if ser is None:
            row["dsr"] = {"ok": False, "reason": "无多空序列"}
            continue
        s = pd.to_numeric(ser, errors="coerce").dropna()
        if len(s) < MIN_SERIES_LEN:
            row["dsr"] = {"ok": False, "reason": f"月数不足（{len(s)}）"}
            continue
        direction = 1 if float(s.mean()) >= 0 else -1
        d = deflated_sharpe((s * direction).to_numpy(), sr_trials=sr_trials)
        row["dsr"] = d
        row["ls_direction"] = direction
        row["sr_monthly"] = (round(sr_by_factor[name], 4)
                             if name in sr_by_factor else None)
        # 年化只为可读（月频 × √12）；判定仍用 DSR 的 p 值
        row["sr_annual"] = (round(row["sr_monthly"] * math.sqrt(12), 2)
                            if row["sr_monthly"] is not None else None)
        if row.get("verdict") == "有效":
            row["verdict"] = ("有效（过 DSR）" if d.get("significant")
                              else "未过多重检验校正")

    # ── PBO：59 个因子互为配置，IS 最优在 OOS 是否仍在前半 ──
    cov = mat.notna().mean()
    keep = [c for c in mat.columns if float(cov.get(c, 0.0)) >= PBO_COVERAGE]
    pbo_res: dict = {"ok": False, "reason": "可用因子不足"}
    if len(keep) >= PBO_MIN_FACTORS:
        sub = mat[keep].dropna(how="any")
        if len(sub) >= PBO_MIN_MONTHS:
            pbo_res = pbo(sub.to_numpy(dtype=float), s_blocks=16)
            pbo_res["factors_used"] = len(keep)
            pbo_res["months_used"] = int(len(sub))
        else:
            pbo_res = {"ok": False, "degraded": True,
                       "reason": f"公共月份仅 {len(sub)} 个（需 ≥{PBO_MIN_MONTHS}）"}

    return {"ok": True, "n_trials": len(sr_trials), "sign_aligned": True,
            # SR* 与 deflated_sharpe 内部同一个函数，避免两套口径
            "sr_star_monthly": (round(expected_max_sharpe(sr_trials), 4)
                                if sr_trials else None),
            "pbo": pbo_res}
