"""本地候选池契约测试（`core/local_pool.py`）

守护四件事：

1. **不降级**：快照取不到时抛 `PoolUnavailable`，**绝不**回退到硬编码精选池
   —— 拿硬编码股票冒充"按条件选出"比报错糟得多。
2. **分层正确**：快照类条件在 L1 过滤，精算类条件 deferred（不能在粗筛层假装筛过）。
3. **截断如实记录**：超过精算上限时给出 `truncated`，而不是静默丢弃。
4. **缺失即不通过**：NaN 参与数值比较一律 False（与评分卡/核对环节同一立场）。

全部用例注入构造快照，不打网络。
"""

import pandas as pd
import pytest

from core import local_pool as lp
from core.nl_screener import NLQuery, parse


# ================================================================
# 夹具
# ================================================================

def _row(code, name, industry, board, pe, cap, turnover, amount, chg):
    return {"code": code, "name": name, "industry": industry, "board": board,
            "price": 10.0, "change_pct": chg, "turnover": turnover, "amount": amount,
            "volume": 1_000_000.0, "pe": pe, "pb": 1.0,
            "market_cap": cap, "float_cap": cap}


ROWS = [
    # 低估值 + 高成交额
    _row("600001", "低估值股", "生物制药", "主板沪市", 12.0, 800, 2.0, 30.0, 1.0),
    # 高估值 + 低成交额
    _row("600002", "高估值股", "生物制药", "主板沪市", 80.0, 300, 8.0, 5.0, -2.0),
    # 创业板 + 高换手
    _row("300001", "创业板股", "电子信息", "创业板", 30.0, 200, 15.0, 3.0, 3.0),
    # ST
    _row("600003", "ST垃圾股", "食品行业", "主板沪市", 5.0, 100, 1.0, 1.0, 0.0),
    # 科创板 + PE 缺失（→ 任何 pe 条件都必须不通过）
    _row("688001", "科创板股", "电子器件", "科创板", None, 500, 4.0, 8.0, 1.5),
]

CAVEAT = "本次候选域 5/10 只（覆盖 50%）——**未覆盖全部创业板/科创板标的，结论不可外推至全市场。**"


def _snap(rows=None, caveat=CAVEAT):
    df = pd.DataFrame(rows if rows is not None else ROWS)
    for col in lp.sina_source.SNAPSHOT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[lp.sina_source.SNAPSHOT_COLUMNS]
    df.attrs.update({"caveat": caveat, "got": len(df), "total_market": 10,
                     "coverage": len(df) / 10, "by_board": {}, "source": "sina_sector"})
    return df


# ================================================================
# 1. 不可用 → 明确失败（不降级）
# ================================================================

def test_snapshot_unavailable_raises(monkeypatch):
    """快照取不到 → 抛 PoolUnavailable（**明确失败，不降级**）

    这里 mock 的是 `get_sector_snapshot`（真取数入口）而不是直接传空表，
    走的是"取不到"的真实代码路径。
    """
    monkeypatch.setattr(lp.sina_source, "get_sector_snapshot", lambda **kw: None)
    with pytest.raises(lp.PoolUnavailable) as ei:
        lp.build_pool(parse("市盈率低于20"))
    msg = str(ei.value)
    assert "不降级" in msg, "必须显式声明不降级到内置精选池"
    assert "快照" in msg
    assert "系统配置" in msg or "重试" in msg, "失败信息要能指路（知道去哪查）"


def test_empty_snapshot_raises():
    with pytest.raises(lp.PoolUnavailable):
        lp.build_pool(parse("市盈率低于20"), snapshot=pd.DataFrame())


# ================================================================
# 2. 分层：快照条件在 L1，精算条件 deferred
# ================================================================

def test_snapshot_condition_filters_at_first_layer():
    """`parse` 默认剔除 ST，所以 600003（ST垃圾股）不在结果里——这是**两个条件
    叠加**的结果：PE 条件 + 默认的非 ST 约定。"""
    result = lp.build_pool(parse("市盈率低于20"), snapshot=_snap())
    assert set(result.codes) == {"600001"}
    assert result.conditions_applied, "必须记录哪些条件在粗筛层执行了"


def test_missing_value_never_passes_numeric_condition():
    """PE 缺失的科创板股**不得**通过 `市盈率低于20`（NaN 比较恒 False）"""
    result = lp.build_pool(parse("市盈率低于20"), snapshot=_snap())
    assert "688001" not in result.codes


def test_deep_condition_is_deferred_not_silently_skipped():
    """精算类条件（波动率）**不能**在粗筛层假装筛过：候选照出，但标记为待精算"""
    result = lp.build_pool(parse("波动率低于30%"), snapshot=_snap())
    assert len(result.codes) == len(ROWS) - 1, "粗筛层不过滤精算类条件（-1 是默认剔除的 ST）"
    assert any("年化波动率" in d for d in result.conditions_deferred)
    assert not result.conditions_applied


def test_mixed_conditions_split_across_layers():
    result = lp.build_pool(parse("市盈率低于20，波动率低于30%"), snapshot=_snap())
    assert set(result.codes) == {"600001"}                # 只有 PE 在 L1 生效
    assert len(result.conditions_applied) == 1
    assert len(result.conditions_deferred) == 1


@pytest.mark.parametrize("text,expect", [
    ("总市值大于500亿", {"600001"}),
    ("换手率低于5%", {"600001", "688001"}),
    ("成交额大于10亿", {"600001"}),
    ("今日涨跌幅大于2%", {"300001"}),
])
def test_each_snapshot_metric_filters_correctly(text, expect):
    """5 个快照指标逐个验证（它们原先全都依赖于问财初筛）"""
    result = lp.build_pool(parse(text), snapshot=_snap())
    assert set(result.codes) == expect, text


# ================================================================
# 3. 板块：概念板块走映射，板块名走 board，映射不到要报错
# ================================================================

def test_concept_sector_maps_to_industries():
    result = lp.build_pool(parse("医药"), snapshot=_snap())
    assert set(result.codes) == {"600001", "600002"}
    assert any("生物制药" in n for n in result.notes)


def test_board_like_sector_filters_by_board():
    """`创业板` 同时出现在 SECTORS 与 BOARDS 里（历史原因）——
    必须按**交易所板块**处理，否则会当成行业名去匹配、恒为 0 只。"""
    result = lp.build_pool(parse("创业板"), snapshot=_snap())
    assert set(result.codes) == {"300001"}


def test_unknown_sector_raises_instead_of_passing_everything():
    """映射不到的板块必须**明确报错**，而不是静默放行全部"""
    with pytest.raises(lp.PoolUnavailable) as ei:
        lp.build_pool(NLQuery(sector="白酒"), snapshot=_snap())
    assert "映射" in str(ei.value)


def test_board_filter_via_board_field():
    result = lp.build_pool(parse("市值大于100亿的科创板股票"), snapshot=_snap())
    assert set(result.codes) == {"688001"}


# ================================================================
# 4. ST 过滤
# ================================================================

def test_exclude_st_removes_st_stocks():
    result = lp.build_pool(parse("市值大于50亿、非ST"), snapshot=_snap())
    assert "600003" not in result.codes
    assert any("ST" in n for n in result.notes)


def test_include_st_keeps_them():
    result = lp.build_pool(parse("市值大于50亿、包括ST"), snapshot=_snap())
    assert "600003" in result.codes


# ================================================================
# 5. 截断：如实记录，不静默丢弃
# ================================================================

def _many_rows(n):
    return [_row(f"60{i:04d}", f"股{i}", "生物制药", "主板沪市",
                 10.0, 100 + i, 1.0, float(i), 0.5) for i in range(n)]


def test_truncation_is_recorded_with_reason():
    """粗筛命中 30 只、上限 10 只 → truncated=20，且说明里写明"未逐一核对" """
    snap = _snap(rows=_many_rows(30))
    result = lp.build_pool(parse("市值大于50亿"), snapshot=snap, max_precision=10)
    assert len(result.codes) == 10
    assert result.truncated == 20
    joined = " ".join(result.notes)
    assert "20" in joined and "未逐一核对" in joined
    assert result.sort_note, "必须说明截断依据（按什么排序取前 N）"


def test_truncation_keeps_highest_amount_first():
    """截断依据是成交额降序——必须真的按它排，而不是「随便取前 10 个」"""
    snap = _snap(rows=_many_rows(30))
    result = lp.build_pool(parse("市值大于50亿"), snapshot=snap, max_precision=5)
    # _many_rows 里 amount = i，最大的是 29
    assert result.codes[0] == "600029"
    assert "成交额" in result.sort_note


def test_no_truncation_when_under_limit():
    result = lp.build_pool(parse("市值大于50亿"), snapshot=_snap(), max_precision=100)
    assert result.truncated == 0
    assert result.sort_note == ""


# ================================================================
# 6. 计数、候选域、取值
# ================================================================

def test_stage_counts_track_every_layer():
    result = lp.build_pool(parse("市盈率低于20、非ST"), snapshot=_snap())
    counts = result.stage_counts
    assert counts["覆盖域"] == len(ROWS)
    assert counts["粗筛后候选"] == len(result.codes)
    assert counts["送入精算"] == len(result.codes)


def test_caveat_is_carried_through():
    """覆盖域说明必须一路带到结果（UI 直接展示这句话）"""
    result = lp.build_pool(parse("市盈率低于20"), snapshot=_snap())
    assert "不可外推" in result.caveat
    assert "不可外推" in lp.pool_note(result)


def test_snapshot_row_for_returns_row_or_none():
    result = lp.build_pool(parse("市盈率低于20"), snapshot=_snap())
    row = lp.snapshot_row_for(result, result.codes[0])
    assert row is not None and row["code"] == result.codes[0]
    assert lp.snapshot_row_for(result, "999999") is None


def test_result_to_dict_is_serializable():
    """结果要能直接进 UI payload（必须可 JSON 化，不能带 DataFrame）"""
    import json
    result = lp.build_pool(parse("市盈率低于20"), snapshot=_snap())
    d = result.to_dict()
    json.dumps(d, ensure_ascii=False)                    # 不抛异常即通过
    assert d["codes"] == result.codes
    assert "caveat" in d and "stage_counts" in d
    assert not any(isinstance(v, pd.DataFrame) for v in d.values()), "不得带 DataFrame"


def test_all_snapshot_metrics_are_registry_declared():
    """粗筛只能碰 `source == snapshot` 的列——防止有人往快照里塞精算类字段"""
    from core.indicator_registry import SOURCE_SNAPSHOT, INDICATORS
    for key in ("pe", "market_cap", "turnover", "amount", "change_pct"):
        assert INDICATORS[key].source == SOURCE_SNAPSHOT
    for col in ("momentum20", "volatility", "news_positive", "fund_net", "score"):
        assert col not in lp.sina_source.SNAPSHOT_COLUMNS
