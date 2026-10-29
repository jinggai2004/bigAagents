"""
相似历史日检索：把每个交易日构造为技术特征向量，余弦相似度检索历史相似日，
并给出其后 5/20 日的表现分布（统计参考，非预测）。

设计要点：
- v1 特征全部来自可回填的日K（涨跌幅/量比/均线偏离/波动率/区间分位等 12 维）；
  资金流/情绪维度无历史数据，待运行积累
- 历史库存 SQLite daily_features；检索为 numpy 暴力扫描（~2400行毫秒级），无需专业向量库
- 防伪相似：排除目标日近 5 日（窗口重叠）；输出强调"统计参考，不构成预测"
"""

import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from db import store

_logger = logging.getLogger(__name__)

FEATURE_KEYS = ["ret_1d", "ret_5d", "ret_20d", "vol_ratio", "volat_20",
                "ma5_dev", "ma10_dev", "ma20_dev", "ma60_dev",
                "dist_20d_high", "streak", "pos_60d"]


# ================================================================
# 特征构建
# ================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从日K构建逐日特征。

    df 需含 date/close 列（amount/volume 可选，缺则 vol_ratio=1）。
    返回 DataFrame[date, close, *FEATURE_KEYS]（前 min_period=60 行为 NaN，已丢弃）。
    """
    close_col = next((c for c in ("close", "Close", "收盘") if c in df.columns), None)
    date_col = next((c for c in ("date", "Date", "日期") if c in df.columns), None)
    if close_col is None or date_col is None:
        raise ValueError(f"K线缺少 date/close 列: {list(df.columns)[:8]}")
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    df["close"] = df[close_col].astype(float)

    out = pd.DataFrame({"date": df["date"], "close": df["close"]})
    out["ret_1d"] = df["close"].pct_change() * 100
    out["ret_5d"] = df["close"].pct_change(5) * 100
    out["ret_20d"] = df["close"].pct_change(20) * 100
    if "volume" in df.columns:
        vol = df["volume"].astype(float)
        out["vol_ratio"] = vol / vol.rolling(20).mean()
    else:
        out["vol_ratio"] = 1.0
    daily_ret = df["close"].pct_change()
    out["volat_20"] = daily_ret.rolling(20).std() * np.sqrt(252) * 100
    for n in (5, 10, 20, 60):
        ma = df["close"].rolling(n).mean()
        out[f"ma{n}_dev"] = (df["close"] / ma - 1) * 100
    out["dist_20d_high"] = (df["close"] / df["close"].rolling(20).max() - 1) * 100

    streak = []
    cur = 0.0
    for ret in out["ret_1d"].values:
        if pd.isna(ret) or ret == 0:
            cur = 0.0
        elif ret > 0:
            cur = cur + 1 if cur > 0 else 1.0
        else:
            cur = cur - 1 if cur < 0 else -1.0
        streak.append(cur)
    out["streak"] = streak

    lo60, hi60 = df["close"].rolling(60).min(), df["close"].rolling(60).max()
    out["pos_60d"] = (df["close"] - lo60) / (hi60 - lo60) * 100

    out = out.dropna(subset=FEATURE_KEYS).reset_index(drop=True)
    return out


# ================================================================
# 历史回填
# ================================================================

def _fetch_daily(symbol: str, years: int = 10) -> pd.DataFrame:
    """拉取历史日K。指数用新浪全历史，个股用腾讯源"""
    import akshare as ak
    if symbol.startswith(("sh000", "sz399")):
        df = ak.stock_zh_index_daily(symbol=symbol)
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
        df = df[pd.to_datetime(df["date"]) >= cutoff]
        return df
    from data.source_manager import data_source_manager
    start = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime("%Y%m%d")
    end = pd.Timestamp.now().strftime("%Y%m%d")
    return data_source_manager.get_stock_hist_data(symbol, start_date=start, end_date=end)


def backfill_history(symbol: str, years: int = 10) -> dict:
    """
    回填标的的历史特征库（覆盖式 upsert）。

    Returns: {"symbol", "days", "from", "to"}；失败抛异常。
    """
    raw = _fetch_daily(symbol, years)
    feats = build_features(raw)
    if feats.empty:
        raise ValueError("特征构建后为空（历史数据不足60日？）")

    closes = feats["close"].values
    dates = feats["date"].values
    n = len(feats)
    for i in range(n):
        future5 = future20 = None
        if i + 5 < n:
            future5 = round((closes[i + 5] / closes[i] - 1) * 100, 2)
        if i + 20 < n:
            future20 = round((closes[i + 20] / closes[i] - 1) * 100, 2)
        vector = {k: round(float(feats[k].iloc[i]), 4) for k in FEATURE_KEYS}
        store.upsert_daily_feature(symbol, str(dates[i]), vector,
                                   round(float(closes[i]), 3), future5, future20)
    result = {"symbol": symbol, "days": n, "from": str(dates[0]), "to": str(dates[-1])}
    _logger.info("[相似日] 回填完成: %s", result)
    return result


# ================================================================
# 相似检索
# ================================================================

def find_similar_days(symbol: str, target_date: str = None, top_k: int = 5) -> dict:
    """
    检索与目标日（默认最新一天）最相似的历史交易日及其后续表现分布。

    Returns:
        {"target": {...}, "similar": [{date, similarity, ret_1d, future_ret_5d, future_ret_20d}],
         "stats": {...}, "disclaimer"}；特征不足或未回填时 {"error"}。
    """
    history = store.load_features(symbol)
    if len(history) < 30:
        return {"error": f"{symbol} 特征库不足（{len(history)}天），请先运行回填 backfill_history()"}

    if target_date:
        pool = [h for h in history if h["date"] <= target_date]
        target = pool[-1] if pool else None
    else:
        target = history[-1]
    if target is None:
        return {"error": f"目标日期无特征数据: {target_date}"}

    # 排除目标日近5日（窗口重叠导致伪相似）
    t_idx = next(i for i, h in enumerate(history) if h["date"] == target["date"])
    # 候选严格限制在目标日及以前（防未来数据泄露），并排除近5日
    candidate_idx = [i for i in range(t_idx + 1) if i <= t_idx - 5]

    # 列级 z-score 标准化（全体历史）
    matrix = np.array([[h["vector"].get(k, 0.0) for k in FEATURE_KEYS] for h in history])
    mean, std = matrix.mean(axis=0), matrix.std(axis=0)
    std[std == 0] = 1.0
    norm = (matrix - mean) / std
    target_vec = norm[t_idx]

    sims = []
    for i in candidate_idx:
        cos = float(np.dot(norm[i], target_vec) /
                    (np.linalg.norm(norm[i]) * np.linalg.norm(target_vec) + 1e-9))
        sims.append((i, cos))
    sims.sort(key=lambda x: x[1], reverse=True)
    top = sims[:max(1, int(top_k))]

    similar = []
    f5, f20, f5_up, f20_up = [], [], 0, 0
    for i, cos in top:
        h = history[i]
        similar.append({
            "date": h["date"], "similarity": round(cos, 4),
            "当日涨跌幅%": h["vector"].get("ret_1d"),
            "后5日%": h["future_ret_5d"], "后20日%": h["future_ret_20d"],
        })
        if h["future_ret_5d"] is not None:
            f5.append(h["future_ret_5d"])
            f5_up += 1 if h["future_ret_5d"] > 0 else 0
        if h["future_ret_20d"] is not None:
            f20.append(h["future_ret_20d"])
            f20_up += 1 if h["future_ret_20d"] > 0 else 0

    def _dist(values, up_count):
        if not values:
            return None
        return {"上涨概率": round(up_count / len(values), 2),
                "平均%": round(float(np.mean(values)), 2),
                "中位%": round(float(np.median(values)), 2),
                "样本数": len(values)}

    return {
        "symbol": symbol,
        "target": {"date": target["date"],
                   "当日涨跌幅%": target["vector"].get("ret_1d"),
                   "现价": target["close"]},
        "similar": similar,
        "stats": {"后5日": _dist(f5, f5_up), "后20日": _dist(f20, f20_up)},
        "history_days": len(history),
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disclaimer": "相似度为技术特征统计匹配，历史表现分布仅供参考，不构成预测或投资建议。",
    }


if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print("回填上证指数历史特征（10年）...")
    print(backfill_history("sh000001", years=10))
    result = find_similar_days("sh000001", top_k=5)
    print(json.dumps(result, ensure_ascii=False, indent=2))
