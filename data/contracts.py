"""数据源输出契约断言（v2 / M0 0.4）

为什么需要：免费数据源最隐蔽的失效方式不是抛异常，而是**格式漂移**——列名改了、
返回行数变少、某列变成全 NaN，而解析函数仍然"成功"返回一个 DataFrame。下游按列
取值得到 NaN 或 KeyError，再被评分逻辑当成真实数值（0 分 / 中性 / 平盘）。

本项目已有降级链可以处理"请求失败"，但对"请求成功但结构不对"无能为力；
这里提供三个轻量断言，在数据层出口显式失败，让漂移立刻可见并触发降级。

用法（在解析函数出口调用）：
    from data.contracts import require_columns, require_not_all_nan
    require_columns(df, ["股票代码", "净额"], "同花顺快照")
    require_not_all_nan(df, ["净额"], "同花顺快照")
"""

import pandas as pd


class DataContractError(RuntimeError):
    """数据源返回结构不符合契约"""


def require_columns(df, columns, source: str):
    """断言 df 是含有所需列的**非空** DataFrame，否则抛 DataContractError"""
    if df is None:
        raise DataContractError(f"[{source}] 返回 None，期望含列 {list(columns)} 的数据")
    if not isinstance(df, pd.DataFrame):
        raise DataContractError(f"[{source}] 返回类型 {type(df).__name__}，期望 DataFrame")
    if df.empty:
        raise DataContractError(f"[{source}] 返回空表，期望含列 {list(columns)} 的数据")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataContractError(
            f"[{source}] 缺少必要列 {missing}；实际列为 {list(df.columns)[:20]}")
    return df


def require_min_rows(df, min_rows: int, source: str):
    """断言 df 至少有 min_rows 行（用于"历史不足以计算指标"的场景）"""
    if df is None:
        raise DataContractError(f"[{source}] 返回 None，期望至少 {min_rows} 行")
    if len(df) < min_rows:
        raise DataContractError(f"[{source}] 仅 {len(df)} 行，少于要求的 {min_rows} 行")
    return df


def require_not_all_nan(df, columns, source: str):
    """断言这些列不是整列全 NaN / 全非数值。

    列名对但内容空，是比"缺列"更隐蔽的漂移：`require_columns` 会放行，
    但后续所有取值都变成 NaN 并被当成 0。
    """
    bad = []
    for col in columns:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().all():
            bad.append(col)
    if bad:
        raise DataContractError(
            f"[{source}] 列 {bad} 全为 NaN/非数值，数据源可能已改格式")
    return df
