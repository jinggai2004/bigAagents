# -*- coding: utf-8 -*-
"""`market_signal` 落盘/读取的唯一编解码器（2026-10-25）。

## 为什么单独成一个模块
`latest.json` 的 `market_signal` 有两种历史形态：

- **新（10-25 起）**：`dict` —— `core/pipeline._json_default` 用 `dataclasses.asdict()`
  落盘，字段与 `core.market_detector.MarketSignal` 一一对应；
- **旧（10-23 及更早，磁盘上仍有）**：`MarketSignal(total_score=28.4, ...)` 的
  **repr 字符串** —— 由旧代码 `json.dump(default=str)` 造成。

消费方（`core/ai_daily_round`、`web/server`）**必须都走这里**。各自写一套正则
就是漂移的源头（项目铁律：服务层禁止第二套实现）。

## 口径
- 认不出来 ⇒ `(None, None)`。**缺就是缺**，绝不返回默认值兜底。
- 旧 repr 用 `ast` 解析关键字参数、逐个 `literal_eval`：比正则稳（引号里的逗号、
  括号不会切错——`operation_advice` 本来就含标点）。个别字段求值失败（历史数据里
  出现过 numpy 标量）就**不带这个字段**，由调用方按"缺"处理。
- `format` 原样交回调用方，让"读到旧文件"这件事**可见**（`/api/picks/latest`
  的 `format` 字段），而不是被静默归一化。

⚠️ 本模块只依赖标准库：服务层 import 它不得牵出 akshare/pandas
（`core.market_detector` 顶层 import akshare，不能在轻量路径里引）。
"""
from __future__ import annotations

import ast
import logging
from dataclasses import asdict, is_dataclass

_logger = logging.getLogger(__name__)

# 缺任一字段，联赛目标仓位就不可用（与 `MarketSignal` 的必填语义一致）
REQUIRED = ("total_score", "market_state", "position_limit")


def coerce(raw):
    """把任意落盘形态归一为 dict。

    Args:
        raw: `dict` / repr `str` / `MarketSignal` 实例 / `None`

    Returns:
        `(data, fmt)`：`fmt` ∈ `{"dict", "legacy", None}`；认不出来时 `(None, None)`
    """
    if raw is None:
        return None, None
    if is_dataclass(raw) and not isinstance(raw, type):
        return asdict(raw), "dict"
    if isinstance(raw, dict):
        return dict(raw), "dict"
    if isinstance(raw, str):
        parsed = _parse_repr(raw)
        if parsed:
            return parsed, "legacy"
    return None, None


def has_required(data) -> bool:
    """必填字段齐不齐。缺就是缺，不拿默认值凑。"""
    return bool(data) and all(k in data for k in REQUIRED)


def _parse_repr(text: str):
    """`MarketSignal(total_score=28.4, market_state='偏弱震荡', ...)` → dict。"""
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as e:
        # 不是 Python 表达式 ⇒ 不是旧 repr，按"认不出"处理（调用方标缺）
        _logger.debug("[市场信号] repr 解析失败，视为无法识别: %s", e)
        return None
    if not isinstance(node, ast.Call) or not node.keywords:
        return None
    out = {}
    for kw in node.keywords:
        if kw.arg is None:          # **kwargs 展开：没有字段名，跳过
            continue
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as e:
            # 该字段不可字面求值（历史数据里有 numpy 标量）⇒ 不带它，调用方按缺处理
            _logger.debug("[市场信号] 字段 %s 不可字面求值，已跳过: %s", kw.arg, e)
            continue
    return out or None
