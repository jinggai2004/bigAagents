"""K 线 → 提示词文本（唯一实现）

为什么单独一个模块（2026-09-28）：
    深度分析链里有两处需要"从 K 线里取点什么"：
      - `core/agents.py`：取**数据截止日**（最后一根 K 线的日期）注入报告口径；
      - `llm/deepseek_client.py`：把最近 N 根压成摘要喂进技术面提示词。
    两处必须对"哪一天才算数据截止""拿不到日期怎么办"给出**同一个答案**，
    否则报告头部的日期与实际喂给模型的序列就会互相矛盾——所以放在 utils 层共用。

放在 utils 而不是 core 的原因同 `utils/logger.py`：llm 层也要用，写成 core→llm 的
反向依赖会破坏 `db ← data ← core ← ui` 的分层。

⚠️ 取不到**可靠日期**时一律返回空串/None，绝不猜：
    `get_stock_data` 返回的是 DatetimeIndex，但也有实现返回带 Date 列的 DataFrame，
    单元测试里还用整数 RangeIndex——整数索引绝不能被当成日期。
"""

from typing import Any, Optional

_DATE_COLUMNS = ("Date", "date", "日期", "trade_date")
_CLOSE_COLUMNS = ("Close", "close", "收盘")
_VOLUME_COLUMNS = ("Volume", "volume", "成交量")


def _as_frame(stock_data: Any):
    """把输入统一成 DataFrame；不是 K 线就返回 None（含 {"error": ...} 的情况）"""
    if stock_data is None or isinstance(stock_data, dict):
        return None
    try:
        import pandas as pd
        if not isinstance(stock_data, pd.DataFrame) or stock_data.empty:
            return None
        return stock_data
    except Exception:                                   # noqa: BLE001
        return None


def _bar_dates(df, pd) -> list[str]:
    """最后一列日期字符串列表；拿不到日期返回空列表"""
    idx = getattr(df, "index", None)
    if idx is not None and not isinstance(idx, pd.RangeIndex):
        try:
            return [pd.Timestamp(v).strftime("%Y-%m-%d") for v in idx]
        except Exception:                               # noqa: BLE001
            return []
    for col in _DATE_COLUMNS:
        if col in df.columns:
            try:
                return [pd.Timestamp(v).strftime("%Y-%m-%d") for v in df[col]]
            except Exception:                           # noqa: BLE001
                return []
    return []


def last_bar_date(stock_data: Any) -> Optional[str]:
    """K 线最后一根的日期（即"数据截止"）。取不到返回 None。"""
    df = _as_frame(stock_data)
    if df is None:
        return None
    import pandas as pd
    dates = _bar_dates(df, pd)
    return dates[-1] if dates else None


def fmt_num(v) -> str:
    """数值显示；None/NaN 显示为「—」（绝不写成 0）"""
    if v is None:
        return "—"
    if isinstance(v, float) and v != v:                 # NaN
        return "—"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def digest(stock_data: Any, n: int = 20) -> str:
    """把 K 线压成「最近 n 根 日期/收盘/成交量」的可读摘要；拿不到就返回空串。

    为什么必须做：原 `technical_analysis(stock_info, stock_data, indicators)` 只用了
    stock_info 与 indicators，**stock_data 在函数体里一次都没被引用**——模型看不到任何
    价格历史，却要输出"放量突破""3 日内站稳 36.70"这类需要价格序列的结论。
    """
    df = _as_frame(stock_data)
    if df is None:
        return ""
    import pandas as pd
    dates = _bar_dates(df, pd)
    if not dates:
        return ""
    close_col = next((c for c in _CLOSE_COLUMNS if c in df.columns), None)
    if close_col is None:
        return ""
    vol_col = next((c for c in _VOLUME_COLUMNS if c in df.columns), None)

    tail = df.tail(max(1, int(n)))
    tail_dates = dates[-len(tail):]
    rows, closes = [], []
    for d, (_, r) in zip(tail_dates, tail.iterrows()):
        cv = r.get(close_col)
        line = f"{d} 收{fmt_num(cv)}"
        if vol_col is not None:
            line += f" 量{fmt_num(r.get(vol_col))}"
        rows.append(line)
        if isinstance(cv, (int, float)) and cv == cv:
            closes.append(float(cv))

    head = f"最近 {len(rows)} 根K线（{tail_dates[0]} ~ {tail_dates[-1]}）"
    if closes:
        head += f"，收盘区间 {min(closes):.2f} ~ {max(closes):.2f}"
    return head + "：\n" + "\n".join(rows)
