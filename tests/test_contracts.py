"""数据契约断言测试（v2 / M0 0.4）

这些断言的意义：把"数据源格式漂移"从"静默产出坏数据"变成"显式失败并降级"。
"""

import pandas as pd
import pytest

from data.contracts import (DataContractError, require_columns, require_min_rows,
                            require_not_all_nan)


def test_require_columns_ok():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert require_columns(df, ["a", "b"], "src") is df


@pytest.mark.parametrize("bad", [None, pd.DataFrame(), "not-a-df", 123])
def test_require_columns_rejects_bad_input(bad):
    with pytest.raises(DataContractError):
        require_columns(bad, ["a"], "src")


def test_require_columns_message_contains_source_missing_and_actual():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(DataContractError) as ei:
        require_columns(df, ["a", "净额"], "同花顺快照")
    msg = str(ei.value)
    assert "净额" in msg and "同花顺快照" in msg and "'a'" in msg


def test_require_min_rows():
    df = pd.DataFrame({"a": range(3)})
    assert require_min_rows(df, 3, "src") is df
    with pytest.raises(DataContractError):
        require_min_rows(df, 4, "src")


def test_require_not_all_nan_catches_all_nan_column():
    """列名对但整列全 NaN —— 比缺列更隐蔽的漂移"""
    df = pd.DataFrame({"close": [float("nan")] * 3, "open": [1.0, 2.0, 3.0]})
    with pytest.raises(DataContractError) as ei:
        require_not_all_nan(df, ["close"], "src")
    assert "close" in str(ei.value)


def test_require_not_all_nan_catches_non_numeric_column():
    """字符串占位（如 'N/A'）同样不能当作有效数值列"""
    with pytest.raises(DataContractError):
        require_not_all_nan(pd.DataFrame({"x": ["N/A", "N/A"]}), ["x"], "src")


def test_require_not_all_nan_passes_partial_data():
    df = pd.DataFrame({"close": [1.0, float("nan")]})
    assert require_not_all_nan(df, ["close"], "src") is df
