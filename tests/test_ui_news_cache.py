# -*- coding: utf-8 -*-
"""热点新闻页：**渲染期联网**的缓存语义（2026-10-15 实测驱动的两个改动）。

背景（实测）：
  · 「实时热榜」块每次切换过去都要 **5.0s**（该接口在本机稳定 ProxyError，等超时才失败）；
  · 首屏被**模块级重依赖导入**拖到 ≈13.7s（`news_industry_review` 9.3s + `news_feed` 4.4s）。

两条对应纪律：
  ① **失败也要进缓存**（负缓存）—— `st.cache_data` 不缓存异常，不处理就会"每次切回都重等一遍"；
  ② **重依赖延迟导入** —— 只在实际渲染那一块时才 import，切到不需要它的块就不付这个钱。
"""
import os

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import use_template_db

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_PAGE = os.path.join(_ROOT, "ui", "pages", "12_📰_热点新闻.py")


@pytest.fixture
def online(tmp_path, monkeypatch):
    """本文件**故意不设 BIGA_UI_OFFLINE**：热榜块只有非离线分支才会被走到。

    其余隔离照旧（临时库 + 清缓存 + 关轮询）。
    """
    use_template_db(tmp_path, monkeypatch, "ui_news_cache.db")
    monkeypatch.delenv("BIGA_UI_OFFLINE", raising=False)
    monkeypatch.setenv("BIGA_UI_POLL_SECONDS", "3600")
    import streamlit as st
    st.cache_data.clear()
    yield tmp_path
    st.cache_data.clear()


def _nav(at):
    from ui.components.section_nav import ALL
    return [r for r in at.radio if ALL in list(r.options)][0]


class TestHotRankNegativeCache:
    def test_failure_is_cached_so_repeat_switch_is_free(self, online, monkeypatch):
        """⭐ 失败也要缓存：连切两次热榜，接口只允许被调用 **1** 次。

        ⚠️ 这是"每次切回都重等 5 秒"那个体感问题的根因 —— `st.cache_data` 不缓存异常，
           所以必须由我们把失败也变成一个**正常的返回值**（`{"ok": False, ...}`）。
        """
        import akshare as ak
        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise RuntimeError("模拟接口不可用")

        monkeypatch.setattr(ak, "stock_hot_rank_em", boom, raising=False)
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]

        # 第一次切到热榜：触发一次失败
        _nav(at).set_value("🔥 实时热榜").run()
        assert len(calls) == 1, f"应恰好请求一次，实际 {len(calls)}"
        warned = "\n".join(str(w.value) for w in at.warning)
        assert "热榜不可用" in warned and "已缓存" in warned, warned

        # 切走再切回：缓存命中 ⇒ 不得再请求
        _nav(at).set_value("🗞️ 全市场快讯").run()
        _nav(at).set_value("🔥 实时热榜").run()
        assert len(calls) == 1, f"失败未被缓存，重复请求了 {len(calls)} 次"

    def test_success_path_shape_is_stable(self, online, monkeypatch):
        """成功路径返回形状必须稳定（页面按 `ok/df/error` 三个键读取）。"""
        import pandas as pd
        import akshare as ak
        monkeypatch.setattr(ak, "stock_hot_rank_em",
                            lambda: pd.DataFrame({"当前排名": [1], "股票代码": ["600028"],
                                                  "股票名称": ["中国石化"], "最新价": [5.4],
                                                  "涨跌幅": [1.0]}), raising=False)
        at = AppTest.from_file(NEWS_PAGE, default_timeout=120)
        at.run()
        _nav(at).set_value("🔥 实时热榜").run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert not [w for w in at.warning if "热榜不可用" in str(w.value)]


class TestHeavyImportsAreLazy:
    def test_page_does_not_import_heavy_modules_at_top_level(self):
        """⭐ 结构性守护：重依赖不得回到**模块级**导入。

        ⚠️ 为什么必须用 AST 钉住而不是靠"测一次耗时"：耗时断言在别人机器上就是 flaky；
           而"哪条 import 在模块级"是**结构事实**，改回去必然失败，且失败信息直指问题。
        """
        import ast
        tree = ast.parse(open(NEWS_PAGE, encoding="utf-8").read())
        top = set()
        for node in tree.body:                       # 只看模块级（col_offset==0）
            if isinstance(node, ast.Import):
                top |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                top.add(node.module)
                if node.module.split(".")[0] in ("core", "data"):
                    top.add(node.module.split(".")[-1])
        for heavy in ("news_industry_review", "news_feed", "akshare"):
            assert heavy not in top, f"{heavy} 又回到模块级导入了（会让首屏多等数秒）"

    def test_lazy_accessors_exist_and_are_callable(self):
        """两个延迟导入入口必须存在（页面各处通过它们取模块）。"""
        src = open(NEWS_PAGE, encoding="utf-8").read()
        assert "def _nir(" in src and "def _news_feed(" in src
