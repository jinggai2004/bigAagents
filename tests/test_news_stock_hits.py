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
