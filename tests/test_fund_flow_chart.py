"""资金流向渲染的回归测试（`ui/components/fund_flow_chart.py`）

## 背景（用户实测报障）

深度分析页的资金流向显示：
    资金流向数据渲染失败: '主力净流入_亿'

## 根因

渲染函数只有**逐日模式**一套模板，直接取 `recent_5["主力净流入_亿"]`。
而列映射是"源列存在才建新列"：

```python
for src, dst in col_mapping.items():
    if src in df.columns:
        df[dst] = df[src] / 1e8          # 源列不存在 → 不建列
...
total = recent_5["主力净流入_亿"].sum()   # 但这里无条件取用 → KeyError
```

深度分析走的偏偏是**同花顺快照模式**（盘后 15:40 落库的降级源），
数据里只有各窗口累计净额（`净额_今日/3日/5日/10日`），**没有单型拆分列**。
异常被兜底吞成一句 warning —— 用户既看不到数据，也看不到原因。

本文件锁住三条：
  1. 快照模式必须能渲染完（不抛 KeyError）；
  2. 逐日模式部分缺列时降级展示（显示「—」并注明缺哪列），而不是崩；
  3. 逐日模式完全没有单型列时给出**含实际字段名**的可诊断提示。
"""

from unittest.mock import MagicMock, patch

import pytest

from ui.components import fund_flow_chart as chart

# 快照模式的真实形状（data/fund_flow.py 的 `_get_ths_fallback_df` 落库记录）
SNAPSHOT_RECORD = {
    "股票代码": "600211", "股票简称": "西藏药业", "数据日期": "2026-09-24",
    "最新价": 36.48, "涨跌幅": -0.27,
    "净额_今日": -2.1e7, "净额_3日": 1.5e7, "净额_5日": 8.0e7, "净额_10日": 1.2e8,
}


def _snapshot_payload(*, with_mode=True):
    flow = {"data": [dict(SNAPSHOT_RECORD)], "days": 1,
            "source": "ths_snapshot", "data_date": "2026-09-24"}
    if with_mode:
        flow["mode"] = "snapshot"
    return {"data_success": True, "fund_flow_data": flow}


def _daily_payload(cols=None):
    """逐日模式（东财/tushare）：带单型拆分与日期列"""
    cols = cols if cols is not None else (
        "日期", "主力净流入-净额", "超大单净流入-净额", "大单净流入-净额")
    rows = []
    for i, day in enumerate(("2026-09-18", "2026-09-19", "2026-09-22", "2026-09-23", "2026-09-24")):
        row = {}
        for c in cols:
            row[c] = day if c == "日期" else float((i + 1) * 1e7)
        rows.append(row)
    return {"data_success": True,
            "fund_flow_data": {"data": rows, "days": len(rows),
                               "source": "eastmoney", "mode": "daily"}}


def _fake_st():
    st = MagicMock()
    st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
    return st


def _render(payload, st):
    with patch.object(chart, "st", st):
        chart.render_fund_flow_section(payload)


# ===== 1. 快照模式 =====

def test_snapshot_mode_renders_without_keyerror():
    """这条用例就是 `KeyError: '主力净流入_亿'` 的回归测试"""
    st = _fake_st()
    _render(_snapshot_payload(), st)

    assert not st.warning.called, f"快照模式不该报错: {st.warning.call_args_list}"
    labels = [c.args[0] for c in st.metric.call_args_list]
    assert labels[:4] == ["今日净额", "3日净额", "5日净额", "10日净额"], labels


def test_snapshot_mode_does_not_draw_daily_bar_chart():
    """快照只有一行窗口数据，画"逐日柱状图"是误导"""
    st = _fake_st()
    _render(_snapshot_payload(), st)
    assert not st.bar_chart.called


def test_snapshot_mode_detected_by_columns_when_mode_missing():
    """历史存档 JSON 可能没有 mode 字段 → 靠列名特征兜底识别"""
    st = _fake_st()
    _render(_snapshot_payload(with_mode=False), st)
    assert not st.warning.called
    assert not st.bar_chart.called


def test_snapshot_mode_reports_missing_windows():
    payload = _snapshot_payload()
    payload["fund_flow_data"]["data"][0].pop("净额_10日")
    st = _fake_st()
    _render(payload, st)
    captions = " ".join(c.args[0] for c in st.caption.call_args_list)
    assert "缺少窗口" in captions and "10日" in captions


# ===== 2. 逐日模式 =====

def test_daily_mode_renders_metrics_and_chart():
    st = _fake_st()
    _render(_daily_payload(), st)
    assert not st.warning.called
    assert st.bar_chart.called
    labels = [c.args[0] for c in st.metric.call_args_list]
    assert labels[:4] == ["主力净流入", "超大单", "大单", "净流入天数"], labels


def test_daily_mode_missing_one_column_degrades_instead_of_crashing():
    """缺「超大单」时该指标显示「—」并注明缺哪列，其余照常"""
    st = _fake_st()
    _render(_daily_payload(cols=("日期", "主力净流入-净额", "大单净流入-净额")), st)

    assert not st.warning.called
    metrics = {c.args[0]: c.args[1] for c in st.metric.call_args_list}
    assert metrics["超大单"] == "—"
    assert metrics["主力净流入"].endswith("亿")
    captions = " ".join(c.args[0] for c in st.caption.call_args_list)
    assert "超大单净流入-净额" in captions


def test_daily_mode_only_plots_available_columns():
    st = _fake_st()
    _render(_daily_payload(cols=("日期", "主力净流入-净额")), st)
    chart_df = st.bar_chart.call_args.args[0]
    assert list(chart_df.columns) == ["主力净流入_亿"]


# ===== 3. 无单型列：可诊断的降级 =====

def test_daily_mode_without_amount_columns_shows_diagnostic():
    """没有任何单型金额列时，必须把**实际字段**打出来，否则无从定位"""
    st = _fake_st()
    _render(_daily_payload(cols=("日期", "中单净流入-净额")), st)

    assert st.warning.called
    msg = st.warning.call_args.args[0]
    assert "没有任何单型金额列" in msg
    assert "中单净流入-净额" in msg and "日期" in msg


def test_no_data_is_not_an_error_path():
    st = _fake_st()
    _render({"data_success": True, "fund_flow_data": {"data": []}}, st)
    assert not st.warning.called
    assert st.info.called


def test_data_success_false_shows_placeholder():
    st = _fake_st()
    _render({"data_success": False}, st)
    assert not st.warning.called
    assert st.info.called


# ===== 4. 单位换算工具 =====

def test_yi_never_turns_missing_into_zero():
    """缺失必须返回 None——返回 0 会被读成"净额恰好为 0"（本项目的老教训）"""
    assert chart._yi(None) is None
    assert chart._yi("N/A") is None
    assert chart._yi("") is None
    assert chart._yi(float("nan")) is None
    assert chart._yi(0) == 0.0
    assert chart._yi(1e8) == 1.0


# ===== 4. 五维评分：不可评维度与全 0（2026-10-10）=====

def test_five_dim_unavailable_dim_does_not_crash():
    """⚠️ 不可评维度（`score=None`）不得参与比例计算。

    回归背景：把"数据源取不到的维度"从"N 分制里扣分"改成"不计入分母"（`score=None`）后，
    本组件仍做 `score / dim_max` → `TypeError: unsupported operand type(s) for /:
    'NoneType' and 'int'` → 被上层 `except` 吞成一句
    "资金流向数据渲染失败" —— 用户既看不到数据也看不到原因（正是本次要修的那类问题）。
    """
    st = _fake_st()
    five_dim = {
        "score": 6.0, "max": 20,
        "breakdown": {
            "方向": {"score": 6, "max": 6, "available": True, "desc": "3/3窗口为正"},
            "结构": {"score": None, "max": 5, "available": False,
                     "desc": "同花顺快照源无单型拆分 —— 不可评，不计入分母"},
        },
    }
    with patch.object(chart, "st", st):
        chart.render_five_dim_score(five_dim)
    assert not st.warning.called
    caps = " ".join(str(c.args[0]) for c in st.caption.call_args_list)
    assert "无单型拆分" in caps


def test_five_dim_zero_score_still_rendered():
    """⚠️ 全 0 分必须照常渲染 —— 它是"资金持续净流出"的**结论**。

    原实现 `if five_dim.get("score") == 0: return` 直接把 0 分藏起来，
    用户看到的正是"资金面五维评分为空"。
    """
    st = _fake_st()
    with patch.object(chart, "st", st):
        chart.render_five_dim_score({
            "score": 0, "max": 20,
            "breakdown": {"方向": {"score": 0, "max": 6, "available": True, "desc": "0/3窗口为正"}},
        })
    assert st.markdown.called
    caps = " ".join(str(c.args[0]) for c in st.caption.call_args_list)
    assert "0/6" in caps and "数据结果" in caps
