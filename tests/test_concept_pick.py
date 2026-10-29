# -*- coding: utf-8 -*-
"""主题（概念）选股层：别名表、主题判定、以及 `recommend_stocks` 的接线。

分工（这是本文件的核心断言）：
  事实层 —— 谁属于"铜"概念 → `data/concept_members.py`（东财题材接口）
  语义层 —— 这条新闻在说"铜"还是"黄金" → 本文件测的 `THEME_TO_CONCEPTS` /
           `_THEME_TRIGGERS`（人工显式表，**绝不模糊匹配**）
  接线   —— 默认关（`BIGA_CONCEPT_PICK=1` 开启），未命中一律回退行业
"""
import pytest

from core import news_industry_review as nr


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "test_concept_pick.db")
    yield


def _seed_review(sector="有色", evidence_titles=("铜价飞涨带动相关企业",),
                 summary="铜价上行利好上游", review_date="2026-10-09"):
    from db import store as db_store
    db_store.save_news_reviews(review_date, [
        {"sector": sector, "verdict": "利好", "summary": summary,
         "evidence": [{"id": i + 1, "title": t}
                      for i, t in enumerate(evidence_titles)],
         "news_count": 5}])


def _patch_industry(monkeypatch, mapping):
    from data import industry_members as im
    monkeypatch.setattr(im, "get_industry_members",
                        lambda **kw: {"members": mapping, "failures": [],
                                      "fetched_at": "2026-10-09 15:00:00",
                                      "age_hours": 0.0, "degraded": False})
    monkeypatch.setattr(nr, "im", im, raising=False)


def _patch_concepts(monkeypatch, concepts, degraded=False, failures=None,
                    fetch_error=None):
    from data import concept_members as cm
    payload = {"concepts": concepts, "failures": failures or [],
               "fetched_at": "2026-10-09 15:00:00", "age_hours": 1.0,
               "degraded": degraded}
    if fetch_error:
        payload["fetch_error"] = fetch_error
    monkeypatch.setattr(cm, "get_concept_members", lambda **kw: payload)


# ================================================================
# 别名表与触发词的结构守护
# ================================================================

class TestThemeTables:

    def test_every_theme_has_triggers_and_concepts(self):
        """双向一致：有映射没触发词 ⇒ 永远匹配不到；有触发词没映射 ⇒ 匹配到也没用。"""
        only_map = set(nr.THEME_TO_CONCEPTS) - set(nr._THEME_TRIGGERS)
        only_trig = set(nr._THEME_TRIGGERS) - set(nr.THEME_TO_CONCEPTS)
        assert not only_map, f"这些主题没有触发词，永远不会命中: {only_map}"
        assert not only_trig, f"这些触发词没有概念映射，命中也没用: {only_trig}"

    def test_no_empty_concept_lists(self):
        empty = [k for k, v in nr.THEME_TO_CONCEPTS.items() if not v]
        assert not empty, f"空概念列表: {empty}"

    def test_triggers_are_long_phrases(self):
        """禁裸单字：`金` 会命中金融/基金，`钢` 会命中钢琴，`油` 会命中酱油。"""
        short = [(theme, w) for theme, words in nr._THEME_TRIGGERS.items()
                 for w in words if len(w) < 2]
        assert not short, f"这些触发词太短，会误伤: {short}"

    def test_themes_do_not_overlap_with_sector_names(self):
        """主题词不能直接当大类名用（否则 `classify_sector` 与 `classify_theme` 语义打架）。

        `钢铁` 是**唯一**的合法例外：它本身就是 11 个大类里最细的那一档，
        新浪「钢铁行业」与东财「钢铁」概念几乎重合，再取别名反而失真。
        其余重叠一律视为设计错误——新增的必须走显式例外，不能默认混用。
        """
        allowed = {"钢铁"}
        overlap = (set(nr.THEME_TO_CONCEPTS) & set(nr.SECTOR_TO_SINA_INDUSTRY)) - allowed
        assert not overlap, f"主题与大类同名，语义会混: {overlap}"


# ================================================================
# 主题判定
# ================================================================

class TestClassifyTheme:

    @pytest.mark.parametrize("text, expected", [
        ("铜价飞涨带动相关企业", "铜"),
        ("电解铝价格创年内新高", "铝"),
        ("稀土出口管制升级", "稀土"),
        ("国际金价站上 3000 美元", "黄金"),
        ("焦煤期货主力合约涨停", "焦煤"),
        ("螺纹钢价格企稳反弹", "钢铁"),
        ("生猪价格环比上涨，能繁母猪存栏下降", "生猪"),
        ("白羽鸡苗价格两周翻倍", "肉鸡"),
        ("水泥价格进入旺季提价周期", "水泥"),
        ("碳酸锂价格跌破成本线", "锂"),
    ])
    def test_hit(self, text, expected):
        assert nr.classify_theme(text) == expected, text

    @pytest.mark.parametrize("text", [
        "珠江钢琴发布新品",           # 含"钢"但是乐器
        "招商银行发布三季报",         # 含"行"不是主题
        "多只基金加仓半导体",         # 含"金"
        "某公司推出鸡尾酒新品",       # 含"鸡"
        "调味品企业酱油收入增长",     # 含"油"
        "公司公告称已完成回购",       # 无主题
    ])
    def test_no_false_positive(self, text):
        assert nr.classify_theme(text) is None, f"{text!r} 被误判为 {nr.classify_theme(text)}"

    def test_multiple_themes_ordered_by_position(self):
        assert nr.themes_in_text("铜价与铝价齐涨") == ["铜", "铝"]
        assert nr.themes_in_text("铝价与铜价齐涨") == ["铝", "铜"]

    def test_empty_text(self):
        assert nr.themes_in_text("") == []
        assert nr.classify_theme(None) is None

    def test_dominant_theme_counts_evidence(self):
        evidence = [{"title": "铜价上涨"}, {"title": "铜矿减产"},
                    {"title": "金价回落"}]
        assert nr._dominant_theme(evidence, "") == "铜"

    def test_dominant_theme_falls_back_to_summary(self):
        assert nr._dominant_theme([], "铜价上行利好上游") == "铜"

    def test_dominant_theme_none_when_no_signal(self):
        assert nr._dominant_theme([{"title": "公司公告"}], "中性") is None


# ================================================================
# 概念代码反查
# ================================================================

class TestCodesForTheme:

    def test_exact_lookup(self):
        concepts = {"铜": {"codes": ["601899", "000630"]},
                    "黄金": {"codes": ["600547"]}}
        assert nr.codes_for_theme("铜", concepts) == {"601899", "000630"}

    def test_fuzzy_lookalike_is_not_included(self):
        """「铜缆高速连接」是通信概念，不能因为名字带"铜"就被算进铜价链。"""
        concepts = {"铜": {"codes": ["601899"]},
                    "铜缆高速连接": {"codes": ["300308"]}}
        assert nr.codes_for_theme("铜", concepts) == {"601899"}

    def test_unknown_theme_is_empty_not_guess(self):
        assert nr.codes_for_theme("铀", {"铜": {"codes": ["601899"]}}) == set()

    def test_missing_alias_is_reported(self):
        concepts = {"黄金": {"codes": ["600547"]}}
        drift = nr.missing_theme_concepts(concepts)
        assert "铜" in drift and "黄金" not in drift

    def test_missing_alias_empty_when_complete(self):
        concepts = {name: {"codes": []}
                    for names in nr.THEME_TO_CONCEPTS.values() for name in names}
        assert nr.missing_theme_concepts(concepts) == []


# ================================================================
# 接线：默认关 / 概念优先 / 行业兜底
# ================================================================

class TestConceptPickWiring:

    def _industry(self, monkeypatch):
        # 新浪口径的行业名（"有色金属"），一只"黄金股"冒充行业成分
        _patch_industry(monkeypatch, {"有色金属": ["600547"]})

    def _flow_none(self, monkeypatch):
        monkeypatch.setattr(nr, "_latest_flow", lambda: (None, None))

    def test_flag_off_keeps_industry_behaviour(self, temp_store, monkeypatch):
        """默认关：即使证据满是"铜价"，也绝不能悄悄换成分股来源。"""
        monkeypatch.delenv("BIGA_CONCEPT_PICK", raising=False)
        self._industry(monkeypatch)
        self._flow_none(monkeypatch)
        _patch_concepts(monkeypatch, {"铜": {"codes": ["601899"]}})
        _seed_review()

        r = nr.recommend_stocks()
        assert r["status"] == "ok"
        assert [p["stock_code"] for p in r["picks"]] == ["600547"]
        assert r["picks"][0]["concept"] is None
        assert r["picks"][0]["pick_basis"] == "industry"

    def test_flag_on_prefers_concept(self, temp_store, monkeypatch):
        monkeypatch.setenv("BIGA_CONCEPT_PICK", "1")
        self._industry(monkeypatch)
        self._flow_none(monkeypatch)
        _patch_concepts(monkeypatch, {"铜": {"codes": ["601899", "000630"]}})
        _seed_review()

        r = nr.recommend_stocks()
        codes = {p["stock_code"] for p in r["picks"]}
        assert codes == {"601899", "000630"}
        assert "600547" not in codes                       # 黄金股必须被排除
        assert all(p["concept"] == "铜" for p in r["picks"])
        assert all(p["pick_basis"] == "concept" for p in r["picks"])
        assert all(p["industry"] == "有色" for p in r["picks"])

    def test_flag_on_falls_back_when_theme_absent(self, temp_store, monkeypatch):
        monkeypatch.setenv("BIGA_CONCEPT_PICK", "1")
        self._industry(monkeypatch)
        self._flow_none(monkeypatch)
        _patch_concepts(monkeypatch, {"铜": {"codes": ["601899"]}})
        _seed_review(evidence_titles=("公司发布回购公告",), summary="无明确主题")

        r = nr.recommend_stocks()
        assert [p["stock_code"] for p in r["picks"]] == ["600547"]
        assert r["picks"][0]["concept"] is None
        assert r["picks"][0]["pick_basis"] == "industry"

    def test_flag_on_falls_back_when_concept_unavailable(self, temp_store, monkeypatch):
        monkeypatch.setenv("BIGA_CONCEPT_PICK", "1")
        self._industry(monkeypatch)
        self._flow_none(monkeypatch)
        from data import concept_members as cm

        def boom(**kw):
            raise cm.ConceptMembersUnavailable("接口返回空表")

        monkeypatch.setattr(cm, "get_concept_members", boom)
        _seed_review()

        r = nr.recommend_stocks()
        assert r["status"] == "ok"
        assert [p["stock_code"] for p in r["picks"]] == ["600547"]
        assert any("概念成分不可用" in w for w in r["warnings"])

    def test_degraded_concept_data_is_visible(self, temp_store, monkeypatch):
        monkeypatch.setenv("BIGA_CONCEPT_PICK", "1")
        self._industry(monkeypatch)
        self._flow_none(monkeypatch)
        _patch_concepts(monkeypatch, {"铜": {"codes": ["601899"]}},
                        degraded=True, failures=["page3(OSError)"])
        _seed_review()

        r = nr.recommend_stocks()
        assert any("概念成分数据不完整" in w for w in r["warnings"])

    def test_alias_drift_falls_back_and_warns(self, temp_store, monkeypatch):
        """东财改名后：主题查不到成分 ⇒ 回退行业，但必须报出漂移，不能静默。"""
        monkeypatch.setenv("BIGA_CONCEPT_PICK", "1")
        self._industry(monkeypatch)
        self._flow_none(monkeypatch)
        _patch_concepts(monkeypatch, {"黄金": {"codes": ["600547"]}})
        _seed_review()

        r = nr.recommend_stocks()
        assert [p["stock_code"] for p in r["picks"]] == ["600547"]
        assert r["picks"][0]["concept"] is None
        assert any("THEME_TO_CONCEPTS" in w for w in r["warnings"])


def test_fallback_pool_survives_foreign_cwd(temp_store, monkeypatch, tmp_path):
    """exe 下 cwd 不是项目根：兜底池 `backtest_pool.json` 仍必须读得到。

    旧实现是 `open("data/backtest_pool.json")`（相对 cwd）——源码下能跑，
    frozen 下必然失败，最终报"行业成分两个来源都不可用"。
    根因不在数据，在路径；所以这里故意把 cwd 换到 tmp 再跑。
    """
    from data import industry_members as im

    monkeypatch.chdir(tmp_path)                    # 模拟 frozen 下的陌生 cwd
    monkeypatch.setattr(im, "load_cached_only",
                        lambda: (_ for _ in ()).throw(
                            im.IndustryMembersUnavailable("无磁盘缓存")))
    monkeypatch.setattr(nr, "_latest_flow", lambda: (None, None))
    _seed_review(sector="科技")

    r = nr.recommend_stocks(allow_network=False)
    # 有池子 ⇒ 即便行业对不上也只会是 ok + warning；no_members 说明连池子都没读到
    assert r["status"] != "no_members", (
        "兜底池应按 data_root() 绝对路径读取，不能依赖 cwd")
