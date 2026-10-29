# -*- coding: utf-8 -*-
"""回测多重检验校正：Deflated Sharpe Ratio（DSR）+ PBO/CSCV + 逆正态工具。

为什么需要（2026-10-29，设计见 output/回测验证与绩效归因改进设计_2026-10-29.md）
──────────────────────────────
FACTOR_WEIGHTS / 阈值 / TOP_N 的组合空间随手就是几十次试验；在多重比较下，
"挑了个巧区间、扫出条好看曲线"毫无显著性可言。本模块把"运气"从"实力"里拆
出来——对错由代码判定，与复盘闭环（analysis_review）同一哲学。

⚠️⚠️ 公式口径（F1 评审教训，2026-10-29）
──────────────────────────────
期望最大"影子"夏普 SR* 里的 γ 是 **Euler–Mascheroni 常数 ≈ 0.5772156649**，
**不是收益偏度**。偏度 γ₃ 与峰度 γ₄ 只出现在 PSR 分母
``sqrt(1 − γ₃·SR + (γ₄−1)/4·SR²)`` 里。
把偏度代进 SR* 会系统性地把"运气"判成"显著"（A 股权益曲线负偏——崩盘尾部；
评审实测 N=24、std(SR)=0.10 时 SR* 偏差 −34%，同一序列判定可从 p=0.46 翻成
p=0.0009 的假阳性，恰好摧毁 DSR 的唯一用途）。
引用：Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014)；
     PSR 见 "The Sharpe Ratio Efficient Frontier" (2012, Journal of Risk)。

⚠️ 频率口径（同款坑的另一半）
──────────────────────────────
SR_hat 与 T **必须同频**：本模块一律**日频**——输入日收益序列，SR 内部现算、
T 用交易日数。外部把年化 SR 配 T=2500 会让 hurdle 差 √252 倍。

⚠️ 校准语义（2026-10-29 实测，测试 test_rate_calibration 有数值锚点）
──────────────────────────────
PSR 假设 SR_hat 服从 N(SR_true, denom/(T−1))——**单条** SR 估计的抽样方差。
但 DSR 检验的 SR_hat 是 N 条试验里**被选中的最大值**：极值统计量的方差
（约 0.6·σ_SR）远小于单条估计的方差，所以把最大值放进单条估计的 PSR 里
**天然偏保守**——白噪声下"选最优后 p<0.05"的实测率 ≈0.3%，不是 5%。
这不是 bug，方向恰好是我们要的（F1 的关切就是假阳性；DSR 宁可错杀）。
因此：
- "5% 校准"的断言位置在**单策略、SR*=0 的普通 PSR**（此时 SR_hat~N(0,denom/T)
  成立，p 值近似均匀）——见测试 test_psr_single_strategy_calibration；
- "选最优后的 DSR"只断言**无系统性假阳性**（率 ≤10%）+ 有真信号时必须显著。

⚠️ 依赖口径
──────────────────────────────
scipy 不在 requirements.txt（仅 numpy 是硬依赖），故正态 CDF/PDF 逆函数用
``math.erf`` + Acklam 近似自实现（相对误差 ~1e-9，对本用途绰绰有余），
不给 PyInstaller 打包添变数。

PBO 阈值说明：文献常以 0.5 报警；本模块常量 ``PBO_WARN=0.3`` 是**有意的保守
选择**（宁可错杀），不是学术标准——改动它要在评审里写明。
"""

from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np

# Euler–Mascheroni 常数。⚠️ 这里的 γ 不是偏度（F1）。
EULER_GAMMA = 0.5772156649015329
# 自然常数（SR* 公式的 1/(N·e) 项），单独命名防止被误改。
E_CONST = math.e

# PBO 警戒线：**有意的保守选择**（文献常用 0.5）。>该值 ⇒ IS 最优在 OOS 落入
# 后一半的比例过高，参数选择的"实力"大概率是运气。
PBO_WARN = 0.3

# DSR 判定阈值（单侧）
DSR_ALPHA = 0.05
# 最少交易日数：低于该值偏度/峰度估计噪声过大，直接拒绝而不是硬算
MIN_DAYS = 20


# ----------------------------------------------------------------
# 正态分布工具（零 scipy 依赖）
# ----------------------------------------------------------------

def norm_cdf(x: float) -> float:
    """标准正态 CDF（math.erf 精确实现）。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """标准正态分位数（Acklam 逆近似，相对误差 ~1.15e-9）。

    只服务 expected_max_sharpe 的 Φ⁻¹(1−1/N) / Φ⁻¹(1−1/(N·e))，N 为试验数。
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p 必须在 (0,1) 内：{p}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:                                   # 左尾
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p <= p_high:                                 # 中央
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = math.sqrt(-2 * math.log(1 - p))             # 右尾（对称）
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


# ----------------------------------------------------------------
# 矩工具（替代 scipy.stats，口径：总体矩，normal → skew=0, kurt=3）
# ----------------------------------------------------------------

def _skew_kurtosis(r: np.ndarray) -> tuple[float, float]:
    """(偏度, Pearson 峰度)。总体矩口径（与 scipy.stats 默认 bias=True 一致，
    normal → (0, 3)）。"""
    n = len(r)
    m2 = float(np.mean((r - r.mean()) ** 2))
    if m2 <= 0:
        return 0.0, 3.0
    m3 = float(np.mean((r - r.mean()) ** 3))
    m4 = float(np.mean((r - r.mean()) ** 4))
    skew = m3 / m2 ** 1.5
    kurt = m4 / (m2 * m2)
    return skew, kurt


def _clean_returns(daily_returns) -> np.ndarray | None:
    """清洗输入：去 None/NaN/inf；空/过短返回 None（"缺就是缺"，不冒充）。"""
    if daily_returns is None:
        return None
    arr = np.asarray([float(x) for x in daily_returns
                      if x is not None and np.isfinite(float(x))], dtype=float)
    if arr.size < MIN_DAYS:
        return None
    return arr


# ----------------------------------------------------------------
# DSR
# ----------------------------------------------------------------

def expected_max_sharpe(sr_trials: Sequence[float]) -> float:
    """N 个独立试验的期望最大夏普 SR*（**日频**）。

    SR* = sqrt(V[{SR_n}]) · [ (1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)) ]
    γ = Euler–Mascheroni 常数（⚠️ 不是偏度，见模块 docstring F1）。

    Args:
        sr_trials: 各次试验的日频夏普估计列表（如参数扫描每组的 SR）。
            方差 V[{SR_n}] 取这些估计的样本方差（ddof=1）。

    Returns:
        SR*（日频）；试验数 <2 或方差 ≤0 时返回 0.0（此时 DSR 退化为普通 PSR，
        调用方应检查 n_trials 并标 degraded）。
    """
    srs = [float(s) for s in (sr_trials or []) if s is not None and math.isfinite(float(s))]
    n = len(srs)
    if n < 2:
        return 0.0
    v = float(np.var(np.asarray(srs), ddof=1))
    if v < 1e-12:                                # 浮点噪声级别的"零方差"（如全同值）
        return 0.0
    z1 = norm_ppf(1.0 - 1.0 / n)
    z2 = norm_ppf(1.0 - 1.0 / (n * E_CONST))
    return math.sqrt(v) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(daily_returns, sr_trials: Sequence[float] | None = None,
                    sr_star: float | None = None) -> dict:
    """Deflated Sharpe Ratio：把"试了很多次之后的最好成绩"折算成显著性。

    DSR = PSR(SR*)，PSR = Φ( (SR_hat − SR*)·sqrt(T−1) / sqrt(1 − γ₃·SR + (γ₄−1)/4·SR²) )
    其中 γ₃=收益偏度、γ₄=Pearson 峰度（normal=3）——**只出现在分母**（F1）。

    Args:
        daily_returns: 策略**日收益**序列（缺口/None 会被剔除）。
        sr_trials: 各次试验的日频夏普列表（推荐：参数扫描的全部组）。
        sr_star: 直接给 hurdle（给了就忽略 sr_trials）。

    Returns:
        dict：{ok, sr_hat, t_days, n_trials, sr_star, hurdle_source, skew,
              kurtosis, z, p_value, significant, degraded, reason?}
        - p_value 是单侧 P(SR_true ≤ SR*)；< DSR_ALPHA 才算"不是运气"。
        - hurdle_source：trials（真校正）/ explicit（外部给定）/ none（SR*=0，
          此时等价普通 PSR，**没有做多重检验校正**，degraded=True）。
        - **结果不可 JSON 的 numpy 类型一律转 Python 原生**（落库/展示口径）。
    """
    base = {"ok": False, "sr_hat": None, "t_days": 0, "n_trials": 0,
            "sr_star": None, "hurdle_source": None, "skew": None, "kurtosis": None,
            "z": None, "p_value": None, "significant": False, "degraded": False}
    r = _clean_returns(daily_returns)
    if r is None:
        return {**base, "reason": f"日收益样本不足（需 ≥{MIN_DAYS} 个交易日）"}
    t = int(r.size)
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return {**base, "reason": "日收益零方差，无法计算夏普"}
    sr_hat = float(r.mean()) / sd
    skew, kurt = _skew_kurtosis(r)
    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat * sr_hat
    if denom <= 0:
        return {**base, "reason": "PSR 分母非正（偏度/峰度极端），拒绝计算"}

    trials = list(sr_trials or [])
    explicit_star_given = sr_star is not None
    if sr_star is None:
        sr_star = expected_max_sharpe(trials)
    # hurdle 来源三态：trials（真校正）/ explicit（外部给定）/ none（SR*=0 ⇒ 未校正）
    if trials:
        hurdle_source = "trials"
    elif sr_star is not None and explicit_star_given:
        hurdle_source = "explicit"
    else:
        hurdle_source = "none"
    degraded = sr_star == 0.0          # hurdle 为 0 = 没有做任何多重检验校正
    n_trials = len(trials) if trials else 1

    z = (sr_hat - float(sr_star)) * math.sqrt(t - 1) / math.sqrt(denom)
    p_value = 1.0 - norm_cdf(z)
    return {**base, "ok": True,
            "sr_hat": round(sr_hat, 6), "t_days": t,
            "n_trials": int(n_trials),
            "sr_star": round(float(sr_star), 6),
            "hurdle_source": hurdle_source,
            "skew": round(skew, 4), "kurtosis": round(kurt, 4),
            "z": round(z, 4), "p_value": round(p_value, 6),
            "significant": bool(p_value < DSR_ALPHA),
            "degraded": bool(degraded)}


# ----------------------------------------------------------------
# PBO / CSCV
# ----------------------------------------------------------------

def _sharpe_by_config(m: np.ndarray) -> np.ndarray:
    """T×C 日收益矩阵 → 每列（配置）夏普。零方差列给 −inf（永不选为最优）。"""
    mean = m.mean(axis=0)
    std = m.std(axis=0, ddof=1) if m.shape[0] > 1 else np.full(m.shape[1], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(std > 0, mean / std, -np.inf)
    return out


def pbo(returns_matrix, s_blocks: int = 16) -> dict:
    """CSCV 概率回测过拟合（Bailey et al. 2017）。

    把 T 天 × C 配置的**日收益矩阵**按时间均匀切 S 块（S 为偶数），遍历全部
    C(S, S/2) 种"训练/测试"对分：取 IS（训练）最优配置，看它在 OOS（测试）
    的相对排名 λ（1=最好）。PBO = P(λ ≤ 0.5)——IS 最优在样本外落到后一半的概率。

    Args:
        returns_matrix: T×C 序列（list[list] / ndarray）。**列 = 各次试验配置**，
            行 = 时间（升序、连续、不可打乱——时间块必须是连续段）。
        s_blocks: 时间块数（偶数；奇数会向上取偶并留痕）。

    Returns:
        dict：{ok, pbo, n_combos, s_blocks, n_configs, t_days, warn, degraded?, reason?}
        - pbo > PBO_WARN(0.3，有意保守) ⇒ warn=True。
        - 配置数 <8 时样本太薄，标 degraded（结论仅供参考）。
    """
    base = {"ok": False, "pbo": None, "n_combos": 0, "s_blocks": 0,
            "n_configs": 0, "t_days": 0, "warn": False, "degraded": False}
    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        return {**base, "reason": "需要 T×C 矩阵且配置数 ≥2"}
    m = m[np.isfinite(m).all(axis=1)]
    t, c = m.shape
    if t < 2 * 4:
        return {**base, "reason": f"时间样本太少（{t} 天）"}
    s = int(s_blocks) if s_blocks and s_blocks >= 4 else 16
    if s % 2:
        s += 1                                   # 强制偶数并留痕（结果带 s_blocks）
    if t < 2 * s:
        return {**base, "reason": f"时间样本不足切 {s} 块（需 ≥{2*s} 天）"}

    blocks = np.array_split(np.arange(t), s)     # 连续时间块；不整除时块大小近似
    combos = list(itertools.combinations(range(s), s // 2))
    lambdas: list[float] = []
    for combo in combos:
        train_idx = np.concatenate([blocks[i] for i in combo])
        test_mask = np.ones(t, dtype=bool)
        test_mask[train_idx] = False
        test_idx = np.where(test_mask)[0]
        is_fit = _sharpe_by_config(m[train_idx])
        oos_fit = _sharpe_by_config(m[test_idx])
        best = int(np.argmax(is_fit))
        oos_best = oos_fit[best]
        if not math.isfinite(oos_best):
            lambdas.append(0.0)                  # IS 最优在 OOS 无效 ⇒ 记最差
            continue
        n_better = int(np.sum(oos_fit > oos_best))
        rel = 1.0 - n_better / max(c - 1, 1)     # λ∈[0,1]，1=OOS 最好
        lambdas.append(rel)
    pbo_val = float(np.mean([1.0 if lam <= 0.5 else 0.0 for lam in lambdas]))
    return {**base, "ok": True,
            "pbo": round(pbo_val, 4),
            "n_combos": len(combos), "s_blocks": s, "n_configs": int(c),
            "t_days": int(t),
            "warn": bool(pbo_val > PBO_WARN),
            "degraded": bool(c < 8)}
