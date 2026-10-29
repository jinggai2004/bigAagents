# -*- coding: utf-8 -*-
"""Agent 工具的公共底座（P2-2 拆分，2026-10-17）。

为什么单独一个模块：`core/agent_tools.py`（1292 行）要拆出财务工具块，
而财务块要用 `_tool` / `_now` ⇒ 若留在 agent_tools 会**循环导入**
（agent_tools → agent_tools_fin → agent_tools）。公共底座放这里，两边单向依赖。
"""
import functools
import logging
from datetime import datetime
from typing import Callable

_logger = logging.getLogger(__name__)

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _tool(fn: Callable) -> Callable:
    """工具执行包装：捕获一切异常，返回 error dict。

    ⚠️ **必须接受位置参数**。早期签名是 `def wrapper(**kwargs)`，只收关键字参数；
    而 Python 侧的内部调用方（如 `core/agents.py` 的基本面节点）会写
    `tool_financials(stock_code, periods=4)` → 位置参数直接抛
    `TypeError: wrapper() takes 0 positional arguments but 1 was given`。

    **注意这个 TypeError 连本函数的 except 都接不住**：参数绑定失败发生在
    wrapper 函数体执行之前，异常直接抛给调用方，所以包装层的"异常兜底"完全失效
    （实测：`_old_tool(fn)('600211', periods=4)` 直接抛出，未返回 error dict）。
    调用方各自的 try/except 把它兜成"取不到数据" → **静默降级**。

    2026-09-28 实测后果：基本面节点因此永远走"未评估"分支，从"编造 13 份定期报告"
    变成"永远没有基本面"——两种都是错的，而且第二种不报错、测试还全绿
    （测试把整个 `tool_financials` patch 掉了，真实函数体从未执行，掩盖了签名错误）。
    用法要点：`functools.wraps` 必须保留，`inspect.signature` 要靠 `__wrapped__`
    还原真实签名（`tests/test_tool_contract.py` 有断言）。
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            if isinstance(result, dict):
                result.setdefault("as_of", _now())
            return result
        except Exception as e:
            _logger.exception("[Agent工具] %s 执行失败", fn.__name__)
            return {"error": f"{type(e).__name__}: {e}", "as_of": _now()}
    return wrapper


