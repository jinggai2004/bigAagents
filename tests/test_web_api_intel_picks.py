# -*- coding: utf-8 -*-
"""情报面新增两个只读端点：`/api/intel/reviews` 与 `/api/intel/picks`。

为什么单独一个文件：这两个端点把「新闻 → 行业判定 → 个股」的后半段落到了 Web 侧，
之前只有 Streamlit 页面能看到。三类断言必须同时存在：

1. **能出数**：有评估记录时，counts/picks 与库内一致；
2. **不造假**：没有记录/没有利好/成分不可用 ⇒ `ok=false` + 明确的 `error.type`
   （绝不返回空表冒充"今天没有机会"）；
3. **不联网**：只读端点在请求里**一次外呼都不许发**——成分映射全量要 49 次请求
   （行业）或 20 页 ≈52 秒（概念），放进 `/api/*` 就是不可预测的长尾。这条是本文件的
   核心守门：把"会联网的那个函数"替换成会抛 `AssertionError` 的桩，若端点仍然正常
   返回，就证明它走的是磁盘缓存路径。
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(temp_store):
    from web.server import cache
    from web.server.app import app
    cache.clear()                      # 进程级缓存：不清会把上个用例的结果带过来
    yield TestClient(app)
    cache.clear()


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "test_intel_picks.db")
    yield


def _seed(review_date="2026-10-29", rows=None):
    from db import store as db_store
    db_store.save_news_reviews(review_date, rows or [
        {"sector": "科技", "verdict": "利好", "summary": "半导体景气回升",
         "evidence": [{"id": 1, "title": "晶圆厂扩产"}], "news_count": 12},
        {"sector": "消费", "verdict": "利好", "summary": "以旧换新落地",
         "evidence": [{"id": 2, "title": "家电补贴"}], "news_count": 9},
        {"sector": "金融", "verdict": "中性", "summary": "无明确催化",
         "evidence": [], "news_count": 7},
    ])


def _patch_members(monkeypatch, mapping, concept_pick=False):
    """让成分映射走 **只读缓存桩**，并标记"联网路径"一旦被调用就炸。"""
    from data import concept_members as cm
    from data import industry_members as im

    monkeypatch.setattr(im, "load_cached_only", lambda: {
        "members": mapping, "failures": [], "fetched_at": "2026-10-29 08:00:00",
        "age_hours": 1.0, "degraded": False, "read_only": True})

    def _boom(*a, **kw):
        raise AssertionError("只读端点不得调用 get_industry_members（会联网）")

    monkeypatch.setattr(im, "get_industry_members", _boom)

    if concept_pick:
        monkeypatch.setenv("BIGA_CONCEPT_PICK", "1")
        monkeypatch.setattr(cm, "load_cached_only", lambda: {
            "concepts": {}, "failures": [], "fetched_at": "2026-10-29 08:00:00",
            "age_hours": 1.0, "degraded": False, "read_only": True})
        monkeypatch.setattr(cm, "get_concept_members",
                            lambda **kw: (_ for _ in ()).throw(
                                AssertionError("只读端点不得调用 get_concept_members")))


# ================================================================
# 行业研判
# ================================================================

class TestReviews:

    def test_ok_with_counts(self, client):
        _seed()
        body = client.get("/api/intel/reviews").json()
        assert body["ok"] is True
        d = body["data"]
        assert d["review_date"] == "2026-10-29"
        assert d["counts"] == {"利好": 2, "利空": 0, "中性": 1, "total": 3}
        assert [r["sector"] for r in d["rows"]] == ["科技", "消费", "金融"]
        assert d["rows"][0]["evidence"] == [{"id": 1, "title": "晶圆厂扩产"}]

    def test_missing_when_no_review(self, client):
        body = client.get("/api/intel/reviews").json()
        assert body["ok"] is False
        assert body["error"]["type"] == "missing"
        assert "15:55" in body["error"]["detail"]

    def test_invalid_date_is_rejected(self, client):
        body = client.get("/api/intel/reviews?date=2026/10/29").json()
        assert body["ok"] is False and body["error"]["type"] == "invalid"

    def test_history_date(self, client):
        _seed("2026-10-27", [{"sector": "医药", "verdict": "利好",
                              "summary": "创新药", "evidence": [], "news_count": 5}])
        _seed("2026-10-29", [{"sector": "科技", "verdict": "利空",
                              "summary": "回调", "evidence": [], "news_count": 6}])
        body = client.get("/api/intel/reviews?date=2026-10-27").json()
        assert body["ok"] is True
        assert body["data"]["review_date"] == "2026-10-27"
        assert body["data"]["rows"][0]["sector"] == "医药"

    def test_dates_list_present(self, client):
        _seed("2026-10-27")
        _seed("2026-10-29")
        d = client.get("/api/intel/reviews").json()["data"]
        assert "2026-10-29" in d["dates"] and "2026-10-27" in d["dates"]


# ================================================================
# 情报选股
# ================================================================

class TestPicks:

    def test_ok_and_never_network(self, client, monkeypatch):
        _seed()
        _patch_members(monkeypatch, {"电子信息": ["300001", "300002"],
                                     "食品行业": ["600887"]})
        monkeypatch.setattr("core.news_industry_review._latest_flow",
                            lambda: (None, None))
        body = client.get("/api/intel/picks?top_n=2").json()
        assert body["ok"] is True
        d = body["data"]
        assert d["review_date"] == "2026-10-29"
        assert d["basis"]["read_only"] is True
        codes = {p["stock_code"] for p in d["picks"]}
        assert codes and codes <= {"300001", "300002", "600887"}
        assert all("pick_basis" in p and "concept" in p for p in d["picks"])

    def test_concept_flag_also_stays_offline(self, client, monkeypatch):
        """开关打开时，概念成分同样必须走只读缓存（否则一次 52 秒）。"""
        _seed()
        _patch_members(monkeypatch, {"电子信息": ["300001"]}, concept_pick=True)
        monkeypatch.setattr("core.news_industry_review._latest_flow",
                            lambda: (None, None))
        body = client.get("/api/intel/picks").json()
        assert body["ok"] is True
        assert all(p["pick_basis"] == "industry" for p in body["data"]["picks"])

    def test_no_bullish_is_missing_not_empty_list(self, client, monkeypatch):
        _seed(rows=[{"sector": "金融", "verdict": "中性", "summary": "无",
                     "evidence": [], "news_count": 5}])
        _patch_members(monkeypatch, {"金融行业": ["600000"]})
        body = client.get("/api/intel/picks").json()
        assert body["ok"] is False
        assert body["error"]["type"] == "missing"
        assert "没有「利好」行业" in body["error"]["detail"]

    def test_no_reviews_is_missing(self, client):
        body = client.get("/api/intel/picks").json()
        assert body["ok"] is False and body["error"]["type"] == "missing"

    def test_members_unavailable_degrades_without_network(self, client, monkeypatch):
        """只读缓存缺失 ⇒ 抛 IndustryMembersUnavailable ⇒ 降级到样本池，仍不联网。"""
        from data import industry_members as im
        _seed()
        monkeypatch.setattr(im, "load_cached_only",
                            lambda: (_ for _ in ()).throw(
                                im.IndustryMembersUnavailable("无磁盘缓存")))
        monkeypatch.setattr(im, "get_industry_members",
                            lambda **kw: (_ for _ in ()).throw(
                                AssertionError("不得联网")))
        monkeypatch.setattr("core.news_industry_review._latest_flow",
                            lambda: (None, None))
        body = client.get("/api/intel/picks").json()
        # 要么给出（样本池兜底），要么明确 missing —— 都不许假装"今天没有机会"
        if body["ok"]:
            assert body["data"]["warnings"]
        else:
            assert body["error"]["type"] == "missing"

    def test_top_n_clamped(self, client, monkeypatch):
        _seed()
        _patch_members(monkeypatch, {"电子信息": ["300001", "300002", "300003"]})
        monkeypatch.setattr("core.news_industry_review._latest_flow",
                            lambda: (None, None))
        body = client.get("/api/intel/picks?top_n=999").json()
        assert body["ok"] is True
        # 单个利好行业最多 3 只 ⇒ 钳到 20 也不会失控
        assert len(body["data"]["picks"]) <= 20

    def test_invalid_date(self, client):
        body = client.get("/api/intel/picks?date=bad").json()
        assert body["ok"] is False and body["error"]["type"] == "invalid"
