"""热点新闻工作流单元测试（临时库隔离，零网络 / 零真实 LLM）

覆盖：
1. store 层：news_industry_reviews 建表 / upsert / 最新一期 / 保留策略按 review_date
2. 聚合：行业归类（多关键词取最强）、条数下限、空标题回退正文
3. LLM 评估：**证据 id 白名单校验（防幻觉核心）** / verdict 枚举 /
   未知行业丢弃 / 无法解析与调用失败不落库
4. 推荐：利好行业 → 成分股 → 资金流排序；行业成分不可用时降级并明示
5. 调度接线：JOBS 注册、非交易日也跑（评估与行情无关）
"""
import json
from datetime import datetime, timedelta

import pytest

from db import store as db_store


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "test_news_review.db")
    yield


def _news_item(title, content="", hours_ago=1, nid=None):
    import hashlib
    now = datetime.now() - timedelta(hours=hours_ago)
    return {"hash": hashlib.md5(f"{title}|{hours_ago}|{nid}".encode()).hexdigest(),
            "title": title, "content": content,
            "source": "sina", "published_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "related_codes": "[]", "sentiment": "neutral"}


class _StubClient:
    """替身 LLM：返回预置文本；记录收到的 prompt 供断言。"""

    def __init__(self, text):
        self.text = text
        self.model = "stub-model"
        self.calls = []

    def call_api(self, messages, **kw):
        self.calls.append(messages)
        return self.text


# ================================================================
# store 层
# ================================================================

class TestStoreNewsReviews:

    def test_save_and_get_roundtrip(self, temp_store):
        n = db_store.save_news_reviews("2026-10-09", [
            {"sector": "科技", "verdict": "利好", "summary": "算力需求旺盛",
             "evidence": [{"id": 1, "title": "x"}], "news_count": 12},
            {"sector": "金融", "verdict": "中性", "summary": "",
             "evidence": [], "news_count": 3},
        ], model="deepseek")
        assert n == 2
        date, rows = db_store.get_news_reviews()
        assert date == "2026-10-09"
        assert {r["sector"]: r["verdict"] for r in rows} == {"科技": "利好", "金融": "中性"}
        assert rows[0]["evidence"] == [{"id": 1, "title": "x"}]
        assert rows[0]["model"] == "deepseek"

    def test_upsert_same_date_sector_overwrites(self, temp_store):
        db_store.save_news_reviews("2026-10-09",
                                   [{"sector": "科技", "verdict": "利好"}])
        db_store.save_news_reviews("2026-10-09",
                                   [{"sector": "科技", "verdict": "利空"}])
        _, rows = db_store.get_news_reviews("2026-10-09")
        assert len(rows) == 1 and rows[0]["verdict"] == "利空"

    def test_get_latest_returns_newest_date(self, temp_store):
        db_store.save_news_reviews("2026-10-08", [{"sector": "科技", "verdict": "利好"}])
        db_store.save_news_reviews("2026-10-09", [{"sector": "金融", "verdict": "中性"}])
        date, rows = db_store.get_news_reviews()
        assert date == "2026-10-09"
        assert rows[0]["sector"] == "金融"

    def test_corrupted_evidence_degrades_not_raises(self, temp_store):
        """evidence 列损坏时降级为空列表 + 保留原文，**不能让页面渲染崩掉**"""
        db_store.save_news_reviews("2026-10-09", [{"sector": "科技", "verdict": "利好"}])
        import sqlite3
        conn = sqlite3.connect(db_store.DB_PATH)
        conn.execute("UPDATE news_industry_reviews SET evidence='{bad json' ")
        conn.commit()
        conn.close()
        _, rows = db_store.get_news_reviews("2026-10-09")
        assert rows[0]["evidence"] == []
        assert rows[0]["evidence_raw"] == "{bad json"

    def test_cleanup_uses_review_date_not_created_at(self, temp_store):
        """⚠️ 快照类判据铁律：补跑历史日期的评估 created_at 是"今天"，
        按创建时间清理会把该删的留着。必须按 review_date。"""
        from datetime import datetime as dt
        old = (dt.now() - timedelta(days=200)).strftime("%Y-%m-%d")
        db_store.save_news_reviews(old, [{"sector": "科技", "verdict": "利好"}])
        db_store.cleanup()                      # 默认保留 90 天
        _, rows = db_store.get_news_reviews(old)
        assert rows == []
        # 新数据不受影响
        db_store.save_news_reviews("2026-10-09", [{"sector": "科技", "verdict": "利好"}])
        db_store.cleanup()
        _, rows = db_store.get_news_reviews("2026-10-09")
        assert len(rows) == 1


# ================================================================
# 聚合
# ================================================================

class TestClassifySector:

    def test_picks_sector_with_most_keyword_hits(self):
        # "AI"×1 + "算力"×1 = 科技 2 命中；"银行"×1 = 金融 1 命中 → 科技
        from core.news_industry_review import classify_sector
        assert classify_sector("AI算力需求带动银行IT订单增长") == "科技"

    def test_no_match_returns_none(self):
        from core.news_industry_review import classify_sector
        assert classify_sector("今天天气不错") is None


class TestAggregate:

    def test_groups_by_sector_and_filters_min_news(self, temp_store):
        from core.news_industry_review import aggregate_recent_news
        items = [_news_item(f"芯片利好消息{i}") for i in range(4)]     # 科技 4 条
        items += [_news_item("某银行发布年报")]                        # 金融 1 条（< 3 不达标）
        db_store.insert_news(items)
        agg = aggregate_recent_news(hours=24)
        sectors = {b["sector"]: b for b in agg}
        assert "科技" in sectors and sectors["科技"]["news_count"] == 4
        assert "金融" not in sectors            # 缺就是缺：条数不足不评估

    def test_empty_title_falls_back_to_content(self, temp_store):
        """⚠️ 实测部分源 title 为空——证据清单里必须有可读标题"""
        from core.news_industry_review import aggregate_recent_news
        # hash 含 hours_ago：错开发布时间，避免被"hash 重复"去重成 1 条
        db_store.insert_news(
            [_news_item("", content=f"光伏产业链涨价通知全文（第{i}批）", hours_ago=i + 1)
             for i in range(4)])
        agg = aggregate_recent_news(hours=24)
        assert agg and agg[0]["items"][0]["title"].startswith("光伏产业链")


# ================================================================
# LLM 评估（证据校验 = 防幻觉核心）
# ================================================================

def _seed_24h_news(temp_store):
    """4 条科技 + 3 条金融（都达标），返回 {sector: [id,...]}。

    ⚠️ 模板库里可能已有其他新闻行——按行业归类过滤时必须容忍
    `classify_sector` 返回 None（未归类的行直接跳过）。
    """
    from core import news_industry_review as nir
    ids = {}
    for titles, key in (([f"芯片突破{i}" for i in range(4)], "科技"),
                        # ⚠️ 标题必须用 _SECTOR_KEYWORDS 的**实际词**：
                        #    金融词表是「银行/证券/保险/降准…」，"券商"不在其中
                        #    （首版写"券商利好"导致金融只凑到 2 条、被条数下限滤掉）
                        (["银行降准", "证券市场利好", "保险资金入市"], "金融")):
        db_store.insert_news([_news_item(t) for t in titles])
        rows = db_store.get_recent_news(limit=200, hours=24)
        ids[key] = [r["id"] for r in rows
                    if nir.classify_sector(r["title"] or "") == key]
    return ids


class TestRunIndustryReview:

    def test_valid_review_is_saved(self, temp_store):
        from core.news_industry_review import run_industry_review
        ids = _seed_24h_news(temp_store)
        client = _StubClient(json.dumps([
            {"sector": "科技", "verdict": "利好", "summary": "芯片突破",
             "evidence_ids": ids["科技"][:2]},
            {"sector": "金融", "verdict": "中性", "summary": "消息混合",
             "evidence_ids": []}], ensure_ascii=False))
        result = run_industry_review(client=client)
        assert result["status"] == "ok" and result["saved"] == 2
        _, rows = db_store.get_news_reviews()
        assert {r["sector"]: r["verdict"] for r in rows} == {"科技": "利好", "金融": "中性"}

    def test_fabricated_evidence_ids_are_dropped_and_flagged(self, temp_store):
        """⚠️ 防幻觉核心：LLM 编造的 id 必须被剔除 + 留痕，不能带着假证据入库"""
        from core.news_industry_review import run_industry_review
        _seed_24h_news(temp_store)
        client = _StubClient(json.dumps([
            {"sector": "科技", "verdict": "利好", "summary": "x",
             "evidence_ids": [99998, 99999]}], ensure_ascii=False))
        result = run_industry_review(client=client)
        assert result["status"] == "ok"
        assert any("无效证据" in w for w in result["warnings"])
        _, rows = db_store.get_news_reviews()
        assert rows[0]["evidence"] == []        # 假 id 全部剔除

    def test_cross_sector_evidence_is_rejected(self, temp_store):
        """拿金融的新闻当科技的证据 = 串行业引用，同样必须拦下"""
        from core.news_industry_review import run_industry_review
        ids = _seed_24h_news(temp_store)
        client = _StubClient(json.dumps([
            {"sector": "科技", "verdict": "利好", "summary": "x",
             "evidence_ids": ids["金融"][:1]}], ensure_ascii=False))
        result = run_industry_review(client=client)
        _, rows = db_store.get_news_reviews()
        assert rows[0]["evidence"] == []
        assert any("无效证据" in w for w in result["warnings"])

    def test_invalid_verdict_falls_to_neutral(self, temp_store):
        from core.news_industry_review import run_industry_review
        _seed_24h_news(temp_store)
        client = _StubClient(json.dumps([
            {"sector": "科技", "verdict": "大涨", "summary": "x", "evidence_ids": []}],
            ensure_ascii=False))
        run_industry_review(client=client)
        _, rows = db_store.get_news_reviews()
        assert rows[0]["verdict"] == "中性"

    def test_unknown_sector_is_dropped(self, temp_store):
        from core.news_industry_review import run_industry_review
        _seed_24h_news(temp_store)
        client = _StubClient(json.dumps([
            {"sector": "航空航天", "verdict": "利好", "summary": "x", "evidence_ids": []}],
            ensure_ascii=False))
        result = run_industry_review(client=client)
        assert result["status"] == "llm_error" and result["saved"] == 0
        assert any("未给出的行业" in w for w in result["warnings"])

    def test_llm_failure_does_not_write(self, temp_store):
        """⚠️ 缺就是缺：LLM 挂了不能落旧结论/空结论充数"""
        from core.news_industry_review import run_industry_review

        class Boom:
            model = "stub"
            def call_api(self, messages, **kw):
                raise TimeoutError("llm down")
        _seed_24h_news(temp_store)
        result = run_industry_review(client=Boom())
        assert result["status"] == "llm_error"
        date, rows = db_store.get_news_reviews()
        assert date is None and rows == []

    def test_unparseable_reply_reports_preview(self, temp_store):
        from core.news_industry_review import run_industry_review
        _seed_24h_news(temp_store)
        result = run_industry_review(client=_StubClient("抱歉我无法回答"))
        assert result["status"] == "llm_error"
        assert "无法解析" in result["error"]

    def test_no_data_is_honest(self, temp_store):
        from core.news_industry_review import run_industry_review
        result = run_industry_review(client=_StubClient("[]"))
        assert result["status"] == "no_data"

    def test_json_in_code_fence_is_parsed(self, temp_store):
        """LLM 爱包 ```json 围栏——必须能抠出来"""
        from core.news_industry_review import run_industry_review
        _seed_24h_news(temp_store)
        client = _StubClient("结论如下：\n```json\n"
                             '[{"sector":"科技","verdict":"利好","summary":"s","evidence_ids":[]}]'
                             "\n```\n以上。")
        assert run_industry_review(client=client)["status"] == "ok"


# ================================================================
# 推荐
# ================================================================

class TestRecommend:

    def _seed_review(self, verdict="利好", sector="科技"):
        db_store.save_news_reviews("2026-10-09", [
            {"sector": sector, "verdict": verdict, "summary": "测试理由",
             "evidence": [{"id": 1, "title": "t"}], "news_count": 5}])

    def _patch_members(self, monkeypatch, mapping):
        from data import industry_members as im
        monkeypatch.setattr(im, "get_industry_members",
                            lambda **kw: {"members": mapping, "failures": [],
                                          "fetched_at": "2026-10-09 15:00:00",
                                          "age_hours": 0.0, "degraded": False})
        monkeypatch.setattr("core.news_industry_review.im", im, raising=False)

    def test_no_reviews_is_reported_not_empty_picks(self, temp_store):
        from core.news_industry_review import recommend_stocks
        r = recommend_stocks()
        assert r["status"] == "no_reviews" and r["picks"] == []

    def test_no_bullish_is_reported(self, temp_store):
        from core.news_industry_review import recommend_stocks
        self._seed_review(verdict="利空")
        r = recommend_stocks()
        assert r["status"] == "no_bullish"

    def test_ranking_by_flow_and_reason_attached(self, temp_store, monkeypatch):
        from core.news_industry_review import recommend_stocks
        self._seed_review()
        # ⚠️ 用新浪口径的行业名（"电子信息"）——别名桥接路径必须真的被走到
        self._patch_members(monkeypatch, {"电子信息": ["300001", "300002", "300003"]})
        monkeypatch.setattr("core.news_industry_review._latest_flow", lambda: (
            {"300001": {"name": "甲", "net_3d": 1e7, "change_pct": 2.0},
             "300002": {"name": "乙", "net_3d": 5e7, "change_pct": 1.0},
             "300003": {"name": "丙", "net_3d": None, "change_pct": None}},
            "2026-10-09"))
        r = recommend_stocks(top_n=2)
        assert r["status"] == "ok" and len(r["picks"]) == 2
        assert [p["stock_code"] for p in r["picks"]] == ["300002", "300001"]
        assert r["picks"][0]["sector_summary"] == "测试理由"
        assert r["picks"][0]["stock_name"] == "乙"

    def test_missing_flow_still_lists_codes_with_warning(self, temp_store, monkeypatch):
        """没有资金流 ≠ 不能推荐——降级为代码序，但必须说清楚"""
        from core.news_industry_review import recommend_stocks
        self._seed_review()
        self._patch_members(monkeypatch, {"电子信息": ["300001", "300002"]})
        monkeypatch.setattr("core.news_industry_review._latest_flow",
                            lambda: (None, None))
        r = recommend_stocks()
        assert r["status"] == "ok" and r["picks"]
        assert any("资金流" in w for w in r["warnings"])

    def test_members_unavailable_falls_back_to_pool(self, temp_store, monkeypatch):
        """全市场成分不可用 → 降级 120 只样本池，degraded 必须为 True"""
        from core.news_industry_review import recommend_stocks
        from data import industry_members as im
        self._seed_review(sector="金融行业")            # 样本池里的行业名
        monkeypatch.setattr(im, "get_industry_members",
                            lambda **kw: (_ for _ in ()).throw(
                                im.IndustryMembersUnavailable("全部行业拉取失败")))
        r = recommend_stocks()
        assert r["status"] == "ok" and r["degraded"] is True
        assert any("样本池" in w for w in r["warnings"])


# ================================================================
# 调度接线
# ================================================================

class TestSectorAliasBridge:
    """⚠️ 口径桥接守护：评估用大类名，成分用新浪细分行名——端到端实测
    直接等值匹配得到 0 只成分股（"金融" vs "金融行业"）。别名表必须
    覆盖每个关键词大类，且子串兜底确实能接住未登记的新大类。"""

    def test_every_keyword_sector_has_alias(self):
        from core.news_industry_review import SECTOR_TO_SINA_INDUSTRY
        from data.news_feed import _SECTOR_KEYWORDS
        missing = set(_SECTOR_KEYWORDS) - set(SECTOR_TO_SINA_INDUSTRY)
        assert not missing, f"大类缺少新浪行业别名: {missing}"

    def test_substring_fallback_catches_unregistered(self):
        from core.news_industry_review import codes_for_sector
        # code_ind 的真实形状是 {code: 行业名}（与 code_to_industry 产出一致）
        code_ind = {"600001": "金融行业", "600002": "其它行业"}
        # 未登记别名的大类，靠子串匹配兜住（'金融' in '金融行业'）
        assert codes_for_sector("金融", code_ind) == {"600001"}
        # '其它行业' 不得误吸（任何大类名都不是它的子串）
        assert codes_for_sector("消费", code_ind) == set()

    def test_alias_and_fallback_union(self):
        from core.news_industry_review import codes_for_sector
        code_ind = {"600001": "金融行业", "000002": "房地产"}
        assert codes_for_sector("金融", code_ind) == {"600001", "000002"}


class TestSchedulerWiring:

    def test_news_review_registered_and_not_trading_day_gated(self):
        """评估与行情无关 → 每天都跑（周末快讯也有价值）"""
        from core import scheduler as sch
        from core.scheduled_jobs import TRADING_DAY_MODES
        assert "news_review" in sch.JOB_IDS
        assert "news_review" not in TRADING_DAY_MODES

    def test_run_job_dispatches_news_review(self, monkeypatch):
        from core import scheduler as sch
        called = []
        monkeypatch.setattr(sch, "run_news_review_job",
                            lambda desc: called.append(desc))
        sch.run_job("news_review", "测试行业评估")
        assert called == ["测试行业评估"]


# ================================================================
# 阶段 A：商品 / 周期关键词（2026-10-28）
# ================================================================

class TestSectorKeywords:
    """交接报告 output/交接报告_股票池定性信息与情报面选股_2026-10-28.md 阶段 A。

    底线：扩词是为了让"铜价涨"能进分类，**不是为了把新闻都塞进某个筐**。
    所以两类断言必须同时存在：命中（不漏）+ 防误伤（不滥）。
    """

    @pytest.mark.parametrize("text, expected", [
        ("铜价飞涨带动相关企业", "有色"),
        ("电解铝价格创年内新高", "有色"),
        ("稀土出口管制升级", "有色"),
        ("国际金价站上 3000 美元，黄金股走强", "有色"),
        ("焦煤期货主力合约涨停", "能源"),
        ("原油价格大幅回落", "能源"),
        ("铁矿石港口库存连续下降", "钢铁"),
        ("螺纹钢价格企稳反弹", "钢铁"),
        ("生猪价格环比上涨，能繁母猪存栏下降", "农业"),
        ("白羽鸡苗价格两周翻倍", "农业"),
        ("纯碱期货放量上行", "化工"),
        ("水泥价格进入旺季提价周期", "建材"),
    ])
    def test_new_sectors_hit(self, text, expected):
        from core.news_industry_review import classify_sector
        assert classify_sector(text) == expected, text

    @pytest.mark.parametrize("text, wrong", [
        # 钢琴是乐器（珠江钢琴是上市公司），不能因为带"钢"就归钢铁
        ("珠江钢琴发布新品", "钢铁"),
        # 银行是金融，不能因为"银"归有色
        ("招商银行发布三季报", "有色"),
        # 基金/资金/金融都含"金"，不能归有色
        ("多只基金加仓半导体", "有色"),
        # 鸡尾酒不是养殖
        ("某公司推出鸡尾酒新品", "农业"),
        # 酱油属于食品（消费），不是能源
        ("调味品企业酱油收入增长", "能源"),
    ])
    def test_no_false_positive(self, text, wrong):
        from core.news_industry_review import classify_sector
        hit = classify_sector(text)
        assert hit != wrong, f"{text!r} 被误归为 {wrong}（实际归 {hit}）"

    def test_no_bare_ambiguous_single_chars(self):
        """结构性守护：半个字的歧义会污染整张表，禁止裸单字入表。

        `金/银` 会命中 金融/银行/基金，`钢` 会命中 钢琴，`鸡` 会命中 鸡尾酒，
        `油` 会命中 酱油。要覆盖这些主题必须用**长词组**。
        """
        from data.news_feed import _SECTOR_KEYWORDS
        banned = {"金", "银", "钢", "鸡", "油", "铜价", "黑色"}
        flat = {w for words in _SECTOR_KEYWORDS.values() for w in words}
        leaked = banned & flat
        # 铜/铝/煤 是允许的裸单字（财经语境几乎无歧义），故不列入 banned
        assert not leaked, f"这些词过于宽泛，会误伤: {leaked}"

    def test_every_sector_has_enough_keywords(self):
        from data.news_feed import _SECTOR_KEYWORDS
        thin = {k: len(v) for k, v in _SECTOR_KEYWORDS.items() if len(v) < 3}
        assert not thin, f"以下大类的词太少（<3），几乎不可能命中: {thin}"

    def test_sector_count_within_max_sectors(self):
        """上限守护：MAX_SECTORS 小于大类数时，排在后面的行业会被静默截断。"""
        from core.news_industry_review import MAX_SECTORS
        from data.news_feed import _SECTOR_KEYWORDS
        assert len(_SECTOR_KEYWORDS) <= MAX_SECTORS, (
            f"关键词大类 {len(_SECTOR_KEYWORDS)} 个 > MAX_SECTORS {MAX_SECTORS}，"
            "超出的行业会被 aggregate_recent_news 截断")

    def test_new_sector_aliases_are_registered(self):
        """结构性守护：新增大类必须在别名表登记，否则反查得到 0 只成分股。"""
        from core.news_industry_review import SECTOR_TO_SINA_INDUSTRY
        from data.news_feed import _SECTOR_KEYWORDS
        missing = set(_SECTOR_KEYWORDS) - set(SECTOR_TO_SINA_INDUSTRY)
        assert not missing, f"缺少新浪行业别名: {missing}"
        empty = {k: v for k, v in SECTOR_TO_SINA_INDUSTRY.items() if not v}
        assert not empty, f"别名表存在空列表: {empty}"

    def test_codes_for_sector_resolves_new_alias(self):
        from core.news_industry_review import codes_for_sector
        code_ind = {"601899": "有色金属", "600019": "钢铁行业", "000001": "金融行业"}
        assert codes_for_sector("有色", code_ind) == {"601899"}
        assert codes_for_sector("钢铁", code_ind) == {"600019"}
