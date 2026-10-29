"""T+N 前瞻收益计算（公共实现，供分析历史 / 选股候选共用）。

为什么单独成模块
----------------
批次 12 只在 `analysis_history` 里写过一遍"拉 K 线 → 归一化 → 算 T+5/T+20"。
批次 15 加选股候选回看时，同样的事要再做一遍——两套实现必然漂移
（基准价取哪根、不足 N 根怎么办、列名怎么兼容），而收益口径一旦不一致，
"条件命中率""胜率"这些结论就全废了。

因此这里的三条规则是**全局唯一口径**：
1. 基准 = 起始日（含）之后的**第一根收盘**（不是 K 线首行、也不是起始日前一根）；
2. `ret_Nd = close[N] / base - 1`，N 根未走完就**不给值**（不拿短窗口冒充 T+20）；
3. 列名兼容与排序在一处做完，调用方拿到的都是 `date`(str) + `close`(float) 升序。

⚠️ 前瞻收益只用于**事后回看**，禁止任何实时决策读取本模块。
"""

import logging
from datetime import datetime, timedelta

import pandas as pd

_logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = (5, 20)


def default_price_fetcher(code: str, start_date: str, end_date: str):
    """默认 K 线获取：走数据源降级链（腾讯/新浪/akshare，含熔断限频）"""
    from data.source_manager import data_source_manager
    return data_source_manager.get_stock_hist_data(code, start_date, end_date)


def normalize_kline(df) -> pd.DataFrame | None:
    """归一化 K 线：列名兼容（Date/日期/trade_date）+ date 转 'YYYY-MM-DD' 字符串、按日升序。

    返回 None 表示"这份数据没法用"（缺列/全空），调用方据此跳过而不是硬算。
    """
    if df is None or getattr(df, "empty", True):
        return None
    df = df.copy()
    for alt in ("Date", "日期", "trade_date"):
        if alt in df.columns and "date" not in df.columns:
            df = df.rename(columns={alt: "date"})
    if "date" not in df.columns or "close" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["close"]).sort_values("date")
    return df if not df.empty else None


def compute_forward_returns(df: pd.DataFrame, from_date: str,
                            horizons=DEFAULT_HORIZONS) -> dict:
    """从已归一化的 K 线算出可得的 T+N 收益。

    Returns:
        {"base_date", "base_price", "ret_5d"?, "ret_20d"?}
        窗口未走完的 horizon 直接缺席（不是 0、不是 None）——调用方用 `"ret_20d" in out`
        判断能否推进到 done 状态。起始日之后没有数据 → 返回 {}。
    """
    if df is None or getattr(df, "empty", True):
        return {}
    rows = df[df["date"] >= from_date]
    if rows.empty:
        return {}
    closes = [float(c) for c in rows["close"]]
    base = closes[0]
    out = {"base_date": str(rows.iloc[0]["date"]), "base_price": round(base, 4)}
    for n in horizons:
        if len(closes) >= n + 1:
            out[f"ret_{n}d"] = round(closes[n] / base - 1, 4)
    return out


def fetch_forward_returns(code: str, from_date: str, price_fetcher=None,
                          horizons=DEFAULT_HORIZONS,
                          buffer_days: int = 7) -> dict:
    """拉 K 线并算 T+N（一次调用搞定）。数据源异常向上抛，由调用方决定重试策略。

    buffer_days: 起始日前多拉几天，防止起始日当天数据缺失导致基准错位。
    """
    fetcher = price_fetcher or default_price_fetcher
    try:
        start = (datetime.strptime(from_date, "%Y-%m-%d")
                 - timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    except ValueError:
        return {}
    end = datetime.now().strftime("%Y-%m-%d")
    df = normalize_kline(fetcher(code, start, end))
    if df is None:
        return {}
    return compute_forward_returns(df, from_date, horizons=horizons)
