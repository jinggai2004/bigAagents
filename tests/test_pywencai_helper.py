"""问财安全封装与公告源可导入性回归测试（v2 / M0）

背景：`data/news_announcement.py` 曾以 `from utils.pywencai_helper import safe_get`
引用一个**从未随仓库提交**的模块，导致该模块 ImportError、公告降级链永久不可用；
而调用处的 `except Exception: pass` 把异常静默吞掉，缺陷此前完全不可见。
本文件锁死两件事：模块可导入 + safe_get 不会把异常抛给调用方。
"""

from unittest.mock import patch

import pandas as pd
import pytest

from utils.pywencai_helper import safe_get


def test_news_announcement_contract_matches_caller():
    """回归：公告降级源必须能导入，且类名/方法名与调用方一致。

    历史上调用方写的是 `NewsAnnouncementFetcher().get_announcements(...)`，
    而模块实际提供 `NewsAnnouncementDataFetcher().get_news_and_announcements(...)`，
    该降级链因此从未成功执行过一次。
    """
    from data.news_announcement import NewsAnnouncementDataFetcher

    assert hasattr(NewsAnnouncementDataFetcher, "get_news_and_announcements")
    assert hasattr(NewsAnnouncementDataFetcher, "format_news_announcements_for_ai")


def test_safe_get_returns_dataframe():
    df = pd.DataFrame({"股票代码": ["000001"]})
    with patch("pywencai.get", return_value=df):
        assert safe_get("测试问句") is df


def test_safe_get_swallows_exception_returns_none():
    """safe_get 的契约是"出错返回 None"，绝不把异常抛给数据层调用方"""
    with patch("pywencai.get", side_effect=RuntimeError("Node 环境缺失")):
        assert safe_get("测试问句") is None


def test_safe_get_single_attempt_without_loop():
    with patch("pywencai.get", side_effect=RuntimeError("boom")) as mock_get:
        assert safe_get("测试问句") is None
    assert mock_get.call_count == 1


def test_safe_get_loop_retries_until_success():
    df = pd.DataFrame({"股票代码": ["000001"]})
    with patch("pywencai.get", side_effect=[RuntimeError("boom"), df]) as mock_get, \
         patch("utils.pywencai_helper.RETRY_INTERVAL", 0):
        assert safe_get("测试问句", loop=True, retries=3) is df
    assert mock_get.call_count == 2


def test_safe_get_timeout_returns_none():
    """硬超时：pywencai 卡住时必须返回 None，不能把整条流程挂死"""
    import time

    def _hang(query, **kwargs):
        time.sleep(2)
        return pd.DataFrame({"late": [1]})

    with patch("pywencai.get", side_effect=_hang):
        assert safe_get("测试问句", timeout=0.1) is None
