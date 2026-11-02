# -*- coding: utf-8 -*-
"""情绪面选股：新闻 → 个股的严匹配（P1-5，2026-10-23）。

锁住四件事：
1. **代码是硬证据**：正文里出现六位代码且在全市场清单内 ⇒ 直接命中；
2. **名称要有语境**：只有名称、附近没有股价/公告类动词 ⇒ 不认（否则"人形机器人板块"
   会把 `300024 机器人` 顶到榜首，实测一天 15 次）；
3. **板块后缀否决**：名称后面紧跟"板块/行业/概念"⇒ 讲的是板块不是这只票；
4. **计数是累加的**：每小时一轮快讯，后一轮不得覆盖前一轮（否则热度失真）。
"""
import pytest

from db import store
from tests.conftest import use_template_db

import core.news_stock_hits as H

NAMES = {"宁德时代": "300750", "中国石化": "600028", "机器人": "300024",
         "金融街": "000402"}
CODES = set(NAMES.values())


class TestMatchText:
    def test_code_is_structural_evidence(self):
        hits = H.match_text("(300750)公布三季报", NAMES, CODES)
        assert [h[0] for h in hits] == ["300750"]

    def test_name_with_cue_matches(self):
        hits = H.match_text("宁德时代盘中涨超5%", NAMES, CODES)
        assert [h[0] for h in hits] == ["300750"]

    def test_name_without_cue_is_rejected(self):
        """只有名字、没有任何股价/公告语境 ⇒ 不认。"""
        assert H.match_text("与会专家讨论了宁德时代的发展模式", NAMES, CODES) == []

    def test_sector_suffix_is_rejected(self):
        """`机器人板块涨超15%` 讲的是板块，不是 300024 这只票。"""
        assert H.match_text("机器人板块涨超15%", NAMES, CODES) == []

    def test_ambiguous_name_needs_code(self):
        """同名普通词（机器人/金融街）**必须**有六位代码佐证。"""
        assert H.match_text("机器人概念涨超3%", NAMES, CODES) == []
        hits = H.match_text("机器人(300024)涨超3%", NAMES, CODES)
        assert [h[0] for h in hits] == ["300024"]

    def test_longest_name_wins_and_no_double_count(self):
        names = {"中国石化": "600028", "中国石油": "601857"}
        hits = H.match_text("中国石化公告：中国石油增持", names, {"600028", "601857"})
        got = sorted(h[0] for h in hits)
        assert got == ["600028", "601857"]
        assert len(hits) == 2

    def test_spaces_in_name_are_normalized(self):
        """面板里存在 "南 京 港" 这种带空格的名字。"""
        names = {"南京港": "002040"}
        assert [h[0] for h in H.match_text("南 京 港 涨超5%", names, {"002040"})] == ["002040"]


class TestStoreAccumulates:
    @pytest.fixture(autouse=True)
    def _db(self, tmp_path, monkeypatch):
        use_template_db(tmp_path, monkeypatch, "news_hits.db")

    def test_upsert_adds_up_not_overwrites(self):
        base = {"stock_code": "300750", "stock_name": "宁德时代",
                "first_at": "2026-10-23 09:00", "last_at": "2026-10-23 09:00",
                "sources": "sina", "samples": "[]"}
        store.upsert_news_stock_hits([dict(base, trade_date="2026-10-23", hits=2, positive=1)])
        store.upsert_news_stock_hits([dict(base, trade_date="2026-10-23", hits=3, positive=2)])
        rows = store.load_news_stock_hits("2026-10-23")
        assert len(rows) == 1
        assert rows[0]["hits"] == 5 and rows[0]["positive"] == 3

    def test_min_hits_filter(self):
        base = {"trade_date": "2026-10-23", "stock_name": "x", "first_at": "", "last_at": "",
                "sources": "", "samples": "[]"}
        store.upsert_news_stock_hits([dict(base, stock_code="300750", hits=1)])
        store.upsert_news_stock_hits([dict(base, stock_code="600028", hits=5)])
        assert [r["stock_code"] for r in store.load_news_stock_hits("2026-10-23", min_hits=3)] \
            == ["600028"]


class TestNameCacheFallback:
    """名称表主源失败时必须降级到分交易所接口（2026-10-29）。

    实测事故：`stock_info_a_code_name()` 因上游 Excel 无法识别引擎整表失败
    （`Excel file format cannot be determined`），而它是 7 天 TTL 的唯一刷新源
    ⇒ 到期那天名称表直接退回覆盖不全的快照，情绪面选股会静默漏掉创业板/北交所。
    """

    def test_split_sources_used_when_primary_fails(self, tmp_path, monkeypatch):
        import pandas as pd
        import akshare as ak

        monkeypatch.setattr(H, "_NAME_CACHE", str(tmp_path / "names.json"))

        def _boom():
            raise ValueError("Excel file format cannot be determined")

        monkeypatch.setattr(ak, "stock_info_a_code_name", _boom)
        monkeypatch.setattr(ak, "stock_info_sh_name_code",
                            lambda symbol="主板A股": pd.DataFrame(
                                {"证券代码": ["600028"], "证券简称": ["中国石化"]}))
        monkeypatch.setattr(ak, "stock_info_sz_name_code",
                            lambda symbol="A股列表": pd.DataFrame(
                                {"A股代码": ["300750"], "A股简称": ["宁德时代"]}))
        monkeypatch.setattr(ak, "stock_info_bj_name_code",
                            lambda: pd.DataFrame(
                                {"证券代码": ["920001"], "证券简称": ["北交测试"]}))

        cache = H.ensure_name_cache(force=True)
        assert cache["source"].startswith("akshare 分交易所")
        assert cache["names"]["宁德时代"] == "300750"
        assert cache["names"]["中国石化"] == "600028"
        assert {"600028", "300750", "920001"} <= set(cache["codes"])

    def test_falls_back_to_cache_when_all_sources_fail(self, tmp_path, monkeypatch):
        import akshare as ak

        cache_path = tmp_path / "names.json"
        cache_path.write_text('{"fetched_at": 0, "names": {"宁德时代": "300750"}, '
                              '"codes": ["300750"], "count": 1}', encoding="utf-8")
        monkeypatch.setattr(H, "_NAME_CACHE", str(cache_path))

        def _boom(*a, **kw):
            raise RuntimeError("network down")

        for fn in ("stock_info_a_code_name", "stock_info_sh_name_code",
                   "stock_info_sz_name_code", "stock_info_bj_name_code"):
            monkeypatch.setattr(ak, fn, _boom)

        cache = H.ensure_name_cache(force=True)
        assert cache["names"]["宁德时代"] == "300750", "失败时必须退回上次缓存"
