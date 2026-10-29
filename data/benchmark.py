"""
基准收益模块：提供沪深300等指数净值曲线，用于信息比率(IR)计算与回测看板对比。

数据源降级链：新浪 stock_zh_index_daily → 腾讯 stock_zh_index_daily_tx。
输出净值归一到 initial_capital（默认100万，与回测初始资金一致），便于看板直接叠加。
"""

import time
import logging
from datetime import datetime

import pandas as pd
import akshare as ak

_logger = logging.getLogger(__name__)

# 支持的基准指数
BENCHMARK_INDEXES = {
    "sh000300": "沪深300",
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
    "sh000852": "中证1000",
}

DEFAULT_INDEX = "sh000300"

_BENCH_TTL = 3600  # 指数日线缓存秒数
_bench_cache = {}  # (symbol, start, end) -> (list[dict], timestamp)


def _fetch_index_daily(index_symbol: str) -> pd.DataFrame:
    """获取指数日线（新浪优先，腾讯备用），列: date/close"""
    errors = []
    for fetcher, name in ((ak.stock_zh_index_daily, "新浪"),
                          (ak.stock_zh_index_daily_tx, "腾讯")):
        try:
            df = fetcher(symbol=index_symbol)
            if df is not None and not df.empty and "close" in df.columns:
                _logger.info("[基准] %s (%s) 数据获取成功 (%d行, %s源)",
                             index_symbol, BENCHMARK_INDEXES.get(index_symbol, ""), len(df), name)
                return df
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:80]}")
    raise ConnectionError(f"指数数据全部源失败: {'; '.join(errors)}")


def get_benchmark_nav(start_date: str, end_date: str,
                      index_symbol: str = DEFAULT_INDEX,
                      initial_capital: float = 1_000_000) -> list[dict] | None:
    """
    获取基准指数在指定区间的净值曲线。

    Args:
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
        index_symbol: 指数代码（见 BENCHMARK_INDEXES）
        initial_capital: 归一基准值（与回测初始资金一致便于叠加展示）

    Returns:
        [{"date": "YYYY-MM-DD", "total_value": float}, ...] 按日期升序；
        数据不可用时返回 None（调用方应跳过 set_benchmark，IR 恒为 0 的旧行为）。
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if start_ts > end_ts:
        return None

    cache_key = (index_symbol, start_ts, end_ts, initial_capital)
    cached = _bench_cache.get(cache_key)
    if cached and time.time() - cached[1] < _BENCH_TTL:
        return cached[0]

    try:
        df = _fetch_index_daily(index_symbol)
    except ConnectionError as e:
        _logger.warning("[基准] %s", e)
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].sort_values("date")
    if len(df) < 2:
        _logger.warning("[基准] %s 在 %s~%s 区间内数据不足", index_symbol, start_date, end_date)
        return None

    closes = df["close"].astype(float).values
    base = float(closes[0])
    nav = [
        {"date": d.strftime("%Y-%m-%d"), "total_value": round(float(c) / base * initial_capital, 2)}
        for d, c in zip(df["date"], closes)
    ]
    _bench_cache[cache_key] = (nav, time.time())
    return nav


def get_benchmark_summary(start_date: str, end_date: str,
                          index_symbol: str = DEFAULT_INDEX) -> dict | None:
    """基准区间摘要（供看板/报告展示），不可用时返回 None"""
    nav = get_benchmark_nav(start_date, end_date, index_symbol)
    if not nav or len(nav) < 2:
        return None
    first, last = nav[0]["total_value"], nav[-1]["total_value"]
    return {
        "index_symbol": index_symbol,
        "index_name": BENCHMARK_INDEXES.get(index_symbol, index_symbol),
        "start_date": nav[0]["date"],
        "end_date": nav[-1]["date"],
        "total_return": round((last / first - 1) * 100, 2),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("基准数据自检...")
    summary = get_benchmark_summary("2024-01-01", "2024-12-31")
    print(summary)
    nav = get_benchmark_nav("2024-01-01", "2024-12-31")
    if nav:
        print(f"净值点数: {len(nav)}, 首点: {nav[0]}, 末点: {nav[-1]}")
