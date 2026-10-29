"""测试板块权重模块 sector.py（纯函数，零外部依赖）

v2（M0）：板块权重矩阵 SECTOR_FACTOR_WEIGHTS 已删除，改为单一权重集 FACTOR_WEIGHTS。
"""

import pytest
from core.sector import (
    get_weights, FACTOR_WEIGHTS, FACTOR_MAX_SCORES, SCORED_FACTORS, classify_by_board,
)

SECTORS = ["科技", "金融", "消费", "新能源", "创业板", "default", "不存在"]


@pytest.mark.parametrize("sector", SECTORS)
def test_get_weights_returns_valid_structure(sector):
    result = get_weights(sector)
    assert isinstance(result, dict)
    assert "weights" in result
    assert "max_scores" in result
    assert isinstance(result["weights"], dict)
    assert isinstance(result["max_scores"], dict)


@pytest.mark.parametrize("sector", SECTORS)
def test_sector_argument_no_longer_changes_weights(sector):
    """v2：sector 参数仅为兼容旧调用保留，不再影响权重（板块矩阵已取消）"""
    assert get_weights(sector)["weights"] == FACTOR_WEIGHTS
    assert get_weights(sector)["weights"] == get_weights()["weights"]


def test_weights_sum_to_one():
    """权重和必须为 1.0，否则总分不再落在 0-100 区间"""
    assert sum(FACTOR_WEIGHTS.values()) == pytest.approx(1.0)


def test_weights_match_max_scores_and_scored_factors():
    """契约：权重键 = 满分键 = _score_single 实际实现的因子集"""
    assert set(FACTOR_WEIGHTS) == set(FACTOR_MAX_SCORES)
    assert set(FACTOR_WEIGHTS) == set(SCORED_FACTORS)


@pytest.mark.parametrize("factor", ["研发投入", "营收增速", "利润增速", "股息率", "ROE", "北向资金"])
def test_no_unimplemented_factor_weights(factor):
    """v2 回归：不得再出现未实现或已失效的因子。

    - 研发投入/营收增速/利润增速/股息率/ROE：screener 从未实现，启用会静默丢失权重；
    - 北向资金：沪深港通自 2024-08-19 起停止披露日频净买入/成交，因子只能取到
      陈旧值或直接失败，对个股排序零贡献（详见 core/sector.py 注释）。
    """
    assert factor not in FACTOR_WEIGHTS
    assert factor not in FACTOR_MAX_SCORES


def test_weights_are_independent_copies():
    r1 = get_weights()
    r2 = get_weights()
    r1["weights"]["主力资金"] = 999.0
    assert r2["weights"]["主力资金"] != 999.0


@pytest.mark.parametrize("factor", FACTOR_MAX_SCORES)
def test_max_scores_positive(factor):
    assert FACTOR_MAX_SCORES[factor] > 0


@pytest.mark.parametrize("code, expected", [
    ("300750", "创业板"), ("301536", "创业板"), ("688981", "科创板"),
    ("600519", "主板沪市"), ("000001", "主板深市"), ("430047", "北交所"),
])
def test_classify_by_board(code, expected):
    assert classify_by_board(code) == expected
