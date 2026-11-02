# -*- coding: utf-8 -*-
"""A 股代码段的**结构性判定**（data 层的唯一实现）。

为什么在这里
────────────
这条规则原先只写在 `core/scheduler.py::is_a_share_code`，但 `data/` 层也需要它
（财报公告日的抓取要按 A 股过滤），而分层约束是 **data 不能 import core**
（`tests/test_layering.py` 用 AST 扫描真实 import，2026-10-30 就是被它挡下的）。
把它下沉到 data，`core.scheduler` 反过来**委托**这里——仍然只有一处实现。

为什么不查"在不在 A 股清单里"
──────────────────────────────
清单来自 akshare 且**可能拿不到**（`market_codes()` 有 None 分支）；拿不到时若按
"不在清单就丢"会把整表丢掉。这里只按代码段判定，不依赖任何网络/缓存，永远可用。

⚠️ 光靠这个函数**不足以**排除新三板：83/87/4x 段按交易所规则确实是北交所合法段，
但东财业绩报表会把 **5,000+ 只新三板**也放进 83/87 段（实测 20260630 共 11,432 行，
其中 5,275 只不在面板里）。要排除它们必须再叠一道**面板代码集**边界，
见 `data/fin_disclosure.a_share_universe()`。
"""
from __future__ import annotations

# 沪深主板 / 创业板 / 科创板 / 北交所（8xx、4xx、920 为 2024 起的新代码段）
_A_SHARE_PREFIXES = ("60", "68", "000", "001", "002", "003",
                     "300", "301", "8", "4", "920")
# B 股（沪 900 / 深 200）不是 A 股
_B_SHARE_PREFIXES = ("900", "200")


def is_a_share_code(code: str) -> bool:
    """6 位代码是否属于 A 股（沪深主板/创业板/科创板/北交所）。

    各段依据（交易所代码分配规则）：
      600/601/603/605 沪主板 · 688/689 科创板 · 900 沪B ❌
      000/001/002/003 深主板/中小板 · 300/301 创业板 · 200 深B ❌
      8xx / 4xx / 920 北交所（920 为 2024 起的新代码段）
    """
    c = str(code or "").strip()
    if len(c) != 6 or not c.isdigit():
        return False
    if c.startswith(_B_SHARE_PREFIXES):
        return False
    return c.startswith(_A_SHARE_PREFIXES)
