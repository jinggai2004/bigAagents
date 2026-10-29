"""个股新闻源契约与适配测试（v3，2026-09-28）

这个 bug 值得单独一个文件，因为它是"测试全绿但功能全废"的典型：
  `core/agents.py` 的舆情节点一直调用 `NewsFlowDataFetcher.get_news_flow_data()`，
  而 `data/news.py` 里**从来没有这个方法**（只有 get_platform_news /
  get_multi_platform_news / extract_stock_related_news）。后果不是报错，而是
  主源每次 AttributeError → 降级源同时"无数据" → **舆情节点实际拿到 0 条新闻**，
  模型随即编出「交叉核查沪深交易所/巨潮资讯网」「Wind 统计 3 家券商点评」。

为什么原来的测试发现不了：`tests/test_agents.py` 把整个 `NewsFlowDataFetcher` patch 成
Mock 之后写 `mock.return_value.get_news_flow_data.return_value = {...}`——
**Mock 上任何属性都成立**，方法名写错照样"测试通过"。
所以这里的断言必须对着**真实类**做。
"""

import inspect
from unittest.mock import patch

import pytest

from core.agents import news_sentiment_node
from data.news import NewsFlowDataFetcher


@pytest.fixture(autouse=True)
def _stub_direct_sources(monkeypatch):
    """默认把两个"按个股直查"的源打成空 —— 单测不该打真网络。

    v4（2026-09-28）`get_news_flow_data` 改成三源合并后，只 patch 快讯库的用例
    会**真的去请求东财**：既慢又不可重复。需要真实行为的用例在用例内再 patch 覆盖。
    """
    from data import news_feed
    monkeypatch.setattr(news_feed, "fetch_stock_news",
                        lambda *a, **k: {"items": [], "note": None, "error": None})
    monkeypatch.setattr(news_feed, "fetch_stock_notices",
                        lambda *a, **k: {"items": [], "note": None, "error": None})
    news_feed.clear_stock_cache()


# ===== 1. 契约：调用方用的方法名必须真实存在 =====

def test_real_class_has_the_method_the_caller_uses():
    assert hasattr(NewsFlowDataFetcher, "get_news_flow_data"), \
        "core/agents.py 调用的方法在真实类上不存在（Mock 掩盖过这个错误）"


def test_caller_really_calls_that_method():
    """反向断言：调用方代码里确实出现该方法名，避免"改了类名忘改调用"或反之"""
    src = inspect.getsource(news_sentiment_node)
    assert "get_news_flow_data" in src


@pytest.mark.parametrize("name", ["get_platform_news", "get_multi_platform_news",
                                  "extract_stock_related_news", "get_news_flow_data"])
def test_expected_public_methods_exist(name):
    assert callable(getattr(NewsFlowDataFetcher, name, None))


# ===== 2. 适配层：复用已验证的 news_feed 路径 =====

def _item():
    return {"title": "西藏药业获批文", "content": "内容", "source": "cls",
            "published_at": "2026-09-24 10:00:00", "url": ""}


def test_adapter_maps_items_into_format_news_block_shape():
    with patch("data.news_feed.smart_latest_news",
               return_value={"news": [_item()], "total": 1, "note": None}):
        out = NewsFlowDataFetcher().get_news_flow_data("600211", days=7, stock_name="西藏药业")

    assert out["data_success"] is True and out["count"] == 1
    it = out["news_list"][0]
    # format_news_block 读的就是这几个键（title/content/source/published_at）
    assert it["title"] == "西藏药业获批文"
    assert it["source"] == "cls"
    assert it["published_at"] == "2026-09-24 10:00:00"


def test_adapter_passes_code_and_name_downstream():
    """related_codes 只覆盖关注池，非关注池个股必须靠简称再匹配一次"""
    with patch("data.news_feed.smart_latest_news",
               return_value={"news": [], "note": "x"}) as smart:
        NewsFlowDataFetcher().get_news_flow_data("600211", days=3, stock_name="西藏药业")

    assert smart.call_args.kwargs["code"] == "600211"
    assert smart.call_args.kwargs["name"] == "西藏药业"
    assert smart.call_args.kwargs["hours"] == 72


def test_empty_result_keeps_scope_note_and_is_not_success():
    with patch("data.news_feed.smart_latest_news",
               return_value={"news": [], "note": "库内 242 条快讯中未检索到与 600211 关联的条目"}):
        out = NewsFlowDataFetcher().get_news_flow_data("600211")

    assert out["data_success"] is False
    assert out["news_list"] == []
    assert "未检索到" in out["note"]


def test_upstream_exception_becomes_degradation_not_crash():
    with patch("data.news_feed.smart_latest_news", side_effect=RuntimeError("库不可用")):
        out = NewsFlowDataFetcher().get_news_flow_data("600211")

    assert out["data_success"] is False
    assert "库不可用" in out["error"]


def test_bad_code_rejected_without_touching_network():
    with patch("data.news_feed.smart_latest_news") as smart:
        out = NewsFlowDataFetcher().get_news_flow_data("abc")
    smart.assert_not_called()
    assert out["error"] and out["data_success"] is False
