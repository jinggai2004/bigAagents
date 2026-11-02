# -*- coding: utf-8 -*-
"""流动性下限（CH-3）：**按板块** 30% 分位的成交额门槛。

## 为什么要有这个模块（2026-10-31）

线上选股（`core/screener.py`）本来就有这道闸门，而**回测没有** —— 两边在选不同的票，
于是任何回测结论都无法映射到实盘。这是"回测好看、实盘打折"的一大半成因：

- 线上剔掉"板块内成交额最小 30%"；
- 回测把最小的那批全留下（因子池是"低流动性 + 高冲击成本"，天然偏好它们）；
- 结果：回测选出来的正是线上会剔除的票。

## 判据（与文献、与线上完全一致）

依据 Liu-Stambaugh-Yuan (2019 JFE)：中国样本必须剔除市值/流动性最小 30%（壳股定价失真）。
本项目实测同向（`output/文献选股因子可实践性_2026-10-29.md`）。

三条**必须一起照抄**的边界（少一条就会得到不同的池子）：

1. 阈值取不到（数据缺失）⇒ **不过滤**，并写明"未生效"。"缺数据"不等于"都是垃圾股"；
2. 个股成交额取不到 ⇒ **该个股保留**（同上）；
3. 过滤后少于 `MIN_KEEP` 只 ⇒ **放弃过滤**（池子已经很小，再砍只剩噪声）。

## 为什么按板块拆，而不是一条全市场阈值

北交所日成交天然在 400~1600 万，全市场 30% 分位（≈0.49 亿）对它等于"永远不合格"。
实测 106 只候选被剔的 4 只里有 2 只是北交所 —— 一条门槛**静默取消了一个板块的选股权**，
而页面只会显示"剔除了几只成交额太低的"。拆开后每个板块按自己的分位，
板块样本 < `MIN_GROUP_SIZE` 时退回全市场阈值（样本太小分位没有意义）。

⚠️ **数据源口径差异（回测 vs 线上）**：线上用**当日快照成交额**（读 `market_snapshot`，
零联网）；回测只能用**20 日均额**（`lndvol20` 还原）。日频噪声由 30% 分位吸收，
但极端波动日的单日截面与 20 日均值排序不会完全相同——所以"回测加了闸门"不等于
"与线上逐只一致"，只能说**用的是同一条规则**。
"""
from __future__ import annotations

import logging
import os

_logger = logging.getLogger("liquidity_floor")

LIQ_FLOOR_PCT = 0.30      # 剔除成交额最小的 30%（CH-3）
MIN_GROUP_SIZE = 100      # 板块样本少于此 ⇒ 用全市场阈值
MIN_KEEP = 10             # 过滤后少于这个数 ⇒ 放弃过滤


def floor_enabled() -> bool:
    """开关（运行时解析）。默认**开**；`BIGA_LIQ_FLOOR=0/false/off/no` 关。

    为什么运行时读：选股与回测都要能一键关掉做对照，而模块级常量在定义时求值，
    之后 `setenv` 不起作用（项目铁律）。
    """
    v = str(os.environ.get("BIGA_LIQ_FLOOR", "1")).strip().lower()
    return v not in ("0", "false", "off", "no", "")


def build_thresholds(amounts_by_code: dict, *, pct: float = LIQ_FLOOR_PCT,
                     min_group: int = MIN_GROUP_SIZE) -> dict | None:
    """由 `{code: 成交额(元)}` 算阈值。样本不足 ⇒ `None`（调用方据此"不过滤"）。

    Returns:
        `{"pct", "overall", "by_board": {板块: 阈值}, "n_market"}`
        —— 与 `screener._liquidity_threshold()` 的返回结构逐字一致（页面契约）。
    """
    from core.sector import classify_by_board

    vals = {str(k): float(v) for k, v in (amounts_by_code or {}).items()
            if v is not None and float(v) > 0}
    if len(vals) < min_group:
        return None

    groups: dict[str, list] = {}
    for code, amt in vals.items():
        groups.setdefault(classify_by_board(code), []).append(amt)

    allv = sorted(v for vs in groups.values() for v in vs)
    overall = allv[max(0, min(int(len(allv) * pct), len(allv) - 1))]
    by_board = {}
    for b, vs in groups.items():
        if len(vs) < min_group:            # 样本太少：该板块用全市场阈值
            continue
        vs.sort()
        by_board[b] = vs[max(0, min(int(len(vs) * pct), len(vs) - 1))]
    return {"pct": pct, "overall": overall, "by_board": by_board,
            "n_market": len(allv)}


def limit_for(code: str, th: dict) -> float:
    """该票适用的门槛：板块有阈值就用板块的，否则退回全市场。"""
    from core.sector import classify_by_board

    return th["by_board"].get(classify_by_board(str(code)), th["overall"])


def split_by_floor(items, th: dict | None, *, min_keep: int = MIN_KEEP) -> dict:
    """按门槛切分。**两边（线上/回测）唯一的判定实现。**

    Args:
        items: `[(code, amount_or_None), ...]`。`amount=None` 表示"取不到"⇒ 保留。
        th: `build_thresholds()` 的结果；`None` ⇒ 原样放行（边界 1）。

    Returns:
        `{"kept": [code...], "dropped": [{"code","board","amount","limit"}...],
          "note": str, "applied": bool}`

        ⚠️ `applied` 的语义是"**闸门生效了**"，不是"剔了票"——生效但一只都没到门槛
           （当天全是流动性好的票）仍然是 `applied=True`。页面上的"流动性下限"标记
           依赖这个区分，混了会出现"闸门明明在跑却显示未生效"。
    """
    items = list(items or [])
    all_codes = [c for c, _ in items]
    if th is None:
        return {"kept": all_codes, "dropped": [],
                "note": "成交额分位不可用 ⇒ 流动性下限未生效（不剔除）",
                "applied": False}

    keep, dropped = [], []
    for code, amt in items:
        code = str(code)
        if amt is None:                      # 边界 2：缺数据保留
            keep.append(code)
            continue
        lim = limit_for(code, th)
        if float(amt) < lim:
            from core.sector import classify_by_board
            dropped.append({"code": code, "board": classify_by_board(code),
                            "amount": float(amt), "limit": float(lim)})
        else:
            keep.append(code)

    if len(keep) < min_keep:                 # 边界 3：池子太小放弃过滤
        note = (f"按成交额下限本应剔除 {len(dropped)} 只，但剩余仅 {len(keep)} 只 "
                f"⇒ 放弃过滤（样本太小）")
        _logger.warning("[流动性下限] %s", note)
        return {"kept": all_codes, "dropped": [], "note": note, "applied": False}

    used = sorted({d["board"] for d in dropped})
    note = (f"成交额下限（{int(th['pct'] * 100)}% 分位·按板块）：剔除 {len(dropped)} 只"
            + ("；" + "、".join(
                f"{b} {th['by_board'].get(b, th['overall']) / 1e8:.2f} 亿" for b in used)
               if used else ""))
    return {"kept": keep, "dropped": dropped, "note": note, "applied": True}


def filter_monthly_panel(df, *, month_col: str = "month_ts", code_col: str = "code",
                         amount_col: str = "amount_yuan",
                         min_group: int = MIN_GROUP_SIZE,
                         pct: float | None = None) -> tuple:
    """月度截面版的闸门：**每月各自算一次分位**，剔掉板块内最小的 `LIQ_FLOOR_PCT`。

    这是回测侧对线上 `_apply_liquidity_floor` 的对应物（2026-10-31 补）：
    线上在"当日候选"上剔，回测在"每月截面"上剔，**规则完全相同**（同一条线：
    按板块、30%、样本不足退化、缺额保留、池子太小放弃）。

    Args:
        df: 长表，需含 `month_col` / `code_col` / `amount_col`（成交额，元）。
        amount_col: 调用方负责先把成交额算成这一列（线上是快照当日额，
            回测是 20 日均额 `exp(lndvol20)`——口径差异见模块 docstring）。

    Returns:
        `(过滤后 df, meta)`；meta 含 `enabled/applied/months/dropped_rows/note`。
        ⚠️ 任一月份分位不可用 ⇒ **该月不剔**（与线上边界 1 一致），不整体失败。
    """
    meta = {"enabled": floor_enabled(), "applied": False, "months": 0,
            "dropped_rows": 0, "note": ""}
    if df is None or len(df) == 0:
        meta["note"] = "空表"
        return df, meta
    if not meta["enabled"]:
        meta["note"] = "流动性下限已关闭（BIGA_LIQ_FLOOR=0）"
        return df, meta
    if amount_col not in df.columns:
        # 缺列 ⇒ 不剔（边界 1 的同一逻辑：判据缺失不等于"都不合格"）
        meta["note"] = f"缺 {amount_col} 列 ⇒ 流动性下限未生效"
        _logger.warning("[流动性下限] %s", meta["note"])
        return df, meta

    kept_parts = []
    n_months = 0
    skipped_months = 0
    for _, g in df.groupby(month_col, sort=False):
        amounts = dict(zip(g[code_col].astype(str),
                           g[amount_col].astype(float)))
        th = build_thresholds(amounts, min_group=min_group,
                              pct=LIQ_FLOOR_PCT if pct is None else float(pct))
        res = split_by_floor(list(amounts.items()), th)
        if not res["applied"]:
            skipped_months += 1
            kept_parts.append(g)
            continue
        n_months += 1
        kept_parts.append(g[g[code_col].astype(str).isin(set(res["kept"]))])

    import pandas as pd
    out = pd.concat(kept_parts).sort_index()
    meta["applied"] = n_months > 0
    meta["months"] = n_months
    meta["dropped_rows"] = int(len(df) - len(out))
    meta["note"] = (f"流动性下限（{int((LIQ_FLOOR_PCT if pct is None else pct) * 100)}% 分位·按板块）："
                    f"{n_months} 个月生效，剔 {meta['dropped_rows']:,} 行"
                    + (f"；{skipped_months} 个月未生效（样本不足）"
                       if skipped_months else ""))
    _logger.info("[流动性下限] %s", meta["note"])
    return out, meta
