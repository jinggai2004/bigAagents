# -*- coding: utf-8 -*-
"""PE 因子"永远无数据"的根因回归（2026-11-01）。

事故：`skip_valuation=True` 时 `get_stock_info()` 返回字符串 **`'N/A'`** 而不是
`None`，而打分侧写的是 `if pe_ratio is None:` ⇒ `'N/A'` 直接穿透，
`_snapshot_pe()` 兜底**永远不执行**，`_score_pe()` 拿到字符串走
`not isinstance(pe, (int, float))` 分支报"无PE数据"。

后果：2026-10-22 ~ 11-01 所有选股轮的 PE 因子都是 unknown（实测 10-29/10-30
每轮 5/5 只），覆盖率固定 0.8235 = 1 − 15/85，而**分位化那条路径用的是
isinstance 判断所以有值** ⇒ 分位与打分口径错位，页面只显示"无PE数据"。

本文件钉住"归一化"这一层，防止任何人把 `if pe is None` 写回来。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.screener import StockScreener       # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    ("N/A", None),          # ← 事故本体：取数器的字符串占位
    ("", None),
    ("-", None),
    (None, None),
    (float("nan"), None),
    (0, None),              # 亏损/零：口径上"无有效 PE"
    (-5.0, None),
    ("7.926", 7.926),        # 数字字符串也接受（取数器偶尔给 str）
    (7.926, 7.926),
    (19, 19.0),
])
def test_pe_value_normalization(raw, expected):
    got = StockScreener._pe_value(raw)
    if expected is None:
        assert got is None, f"{raw!r} 应归一化为 None，实际 {got!r}"
    else:
        assert got == pytest.approx(expected)


def test_score_pe_rejects_string_na():
    """`_score_pe` 收到 'N/A' 必须判 unknown，不能拿它算分（当前行为，防回退）。"""
    s = StockScreener()
    r = s._score_pe("N/A", 15)
    assert r["status"] == "unknown" and r["score"] is None


def test_score_pe_accepts_normalized_value():
    """归一化后的正常 PE 必须能打分（这是修复后应有的行为）。"""
    s = StockScreener()
    r = s._score_pe(7.9, 15)
    assert r["status"] == "ok"
    assert isinstance(r["score"], int) and r["score"] > 0


def test_na_reaches_snapshot_fallback(monkeypatch):
    """核心接线：info 里是 'N/A' 时，必须走到 `_snapshot_pe` 兜底并打到分。

    这条用例直接钉"`if pe_ratio is None` 会被 'N/A' 绕过"这个 bug：
    如果实现退回去用 `is None` 判断，`_snapshot_pe` 不会被调用 ⇒ 断言失败。
    """
    s = StockScreener()
    called = {"n": 0}

    def _fake_snapshot_pe(code):
        called["n"] += 1
        return 7.926

    monkeypatch.setattr(s, "_snapshot_pe", _fake_snapshot_pe)
    # 让取数器返回 'N/A'（真实行为），并且不联网
    monkeypatch.setattr(s, "_offline", True, raising=False)
    monkeypatch.setitem(s._info_cache, "601328", {"pe_ratio": "N/A", "name": "交通银行"})
    # 关掉其它因子（只验 PE 接线，避免联网取 K 线）
    from core.sector import get_weights
    spec = get_weights()
    w = {k: 0.0 for k in spec["weights"]}
    w["PE估值"] = spec["weights"]["PE估值"]
    ms = spec["max_scores"]

    r = s._score_single("601328", w, ms, context=None)
    pe = r["breakdown"]["PE估值"]
    assert called["n"] >= 1, "'N/A' 没有触发 _snapshot_pe 兜底（退回 is None 判断了）"
    assert pe["status"] == "ok", f"PE 仍未进入：{pe}"
    assert pe.get("pe_source") == "snapshot_sina"
