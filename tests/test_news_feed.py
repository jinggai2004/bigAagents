"""新闻采集与统一存储层单元测试（mock akshare，临时库隔离）"""
import hashlib
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd

import data.news_feed as nf
from db import store as db_store


def _ago(**kwargs) -> datetime:
    """相对当前时间的 datetime（避免硬编码日期随时间过期）"""
    return datetime.now().replace(microsecond=0) - timedelta(**kwargs)


def _ago_str(**kwargs) -> str:
    """相对当前时间的时间戳字符串 '%Y-%m-%d %H:%M:%S'"""
    return _ago(**kwargs).strftime("%Y-%m-%d %H:%M:%S")


# 模块级固定时间点：DataFrame 与断言共用同一基准，避免跨秒/跨日漂移
# （均在一小时内，保证库内新闻"新鲜"，同时落在 get_recent_news 的 hours 窗口内）
_SINA_T1 = _ago(minutes=10)
_SINA_T2 = _ago(minutes=20)
_SINA_T3 = _ago(minutes=30)  # 无年份列，由 _norm_time 补当前年


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    """临时数据库（隔离真实 portfolio.db）"""
    # 从 session 级空库模板复制（等价于新建+init_store，省掉 1.6s 建表）
    from tests.conftest import use_template_db
    db_path = use_template_db(tmp_path, monkeypatch, "test_store.db")
    yield db_path


SINA_DF = pd.DataFrame([
    {"时间": _SINA_T1.strftime("%Y-%m-%d %H:%M:%S"), "内容": "贵州茅台(600519) 发布利好公告，业绩预增"},
    {"时间": _SINA_T2.strftime("%Y-%m-%d %H:%M:%S"), "内容": "某公司处罚立案，股价大跌"},
    {"时间": _SINA_T3.strftime("%m-%d %H:%M"), "内容": "普通快讯一条"},
])
THS_DF = pd.DataFrame([
    {"标题": "贵州茅台获增持", "内容": "北向资金增持贵州茅台(600519)", "发布时间": _SINA_T1.strftime("%Y-%m-%d %H:%M:%S"), "链接": ""},
])


class TestStore:
    def test_insert_news_dedup(self, temp_store):
        item = {"hash": "h1", "title": "t", "content": "c", "source": "sina",
                "published_at": _ago_str(minutes=10), "related_codes": ["600519"],
                "sentiment": "positive"}
        assert db_store.insert_news([item]) == 1
        assert db_store.insert_news([item]) == 0  # hash 重复忽略
        news = db_store.get_recent_news()
        assert len(news) == 1
        assert news[0]["related_codes"] == ["600519"]

    def test_get_recent_news_filters(self, temp_store):
        db_store.insert_news([
            {"hash": "a", "title": "降息利好银行", "content": "", "source": "sina",
             "published_at": _ago_str(minutes=10), "related_codes": ["000001"], "sentiment": "positive"},
            {"hash": "b", "title": " unrelated ", "content": "光伏出口", "source": "ths",
             "published_at": _ago_str(minutes=20), "related_codes": [], "sentiment": "neutral"},
        ])
        assert len(db_store.get_recent_news(keyword="降息")) == 1
        assert len(db_store.get_recent_news(code="000001")) == 1
        assert len(db_store.get_recent_news(limit=1)) == 1

    def test_chat_logs_and_stats(self, temp_store):
        db_store.insert_chat_log("s1", "user", "你好", prompt_tokens=10, completion_tokens=5)
        db_store.insert_chat_log("s1", "assistant", "回答", tools_used=["tool_x"])
        stats = db_store.store_stats()
        assert stats["chat_logs"]["rows"] == 2
        assert stats["news_items"]["rows"] == 0

    def test_cleanup_retention(self, temp_store):
        db_store.insert_chat_log("s1", "user", "旧消息")
        # 手动把 created_at 改旧
        import sqlite3
        conn = sqlite3.connect(str(temp_store))
        conn.execute("UPDATE chat_logs SET created_at = '2020-01-01 00:00:00'")
        conn.commit()
        conn.close()
        deleted = db_store.cleanup()
        assert deleted["chat_logs"] == 1

    def test_daily_features_upsert(self, temp_store):
        db_store.upsert_daily_feature("sh000001", "2026-08-29", {"ret": 1.2}, 3900.0)
        db_store.upsert_daily_feature("sh000001", "2026-08-29", {"ret": 1.5}, 3901.0,
                                      future_ret_5d=0.3)
        feats = db_store.load_features("sh000001")
        assert len(feats) == 1  # upsert 而非新增
        assert feats[0]["vector"]["ret"] == 1.5
        assert feats[0]["future_ret_5d"] == 0.3


class TestParsers:
    def test_parse_sina(self):
        items = nf._parse_sina(SINA_DF)
        assert len(items) == 3
        assert items[0]["published_at"] == _SINA_T1.strftime("%Y-%m-%d %H:%M:%S")
        assert items[2]["published_at"].startswith(f"{datetime.now().year}-{_SINA_T3.strftime('%m-%d')}")  # 无年份补年

    def test_parse_ths(self):
        items = nf._parse_ths_generic(THS_DF)
        assert items[0]["title"] == "贵州茅台获增持"

    def test_parse_sina_rejects_column_drift(self):
        """v2/M0 0.4：列名变化必须显式失败，而不是产出坏数据"""
        from data.contracts import DataContractError
        bad = pd.DataFrame({"time": ["2026-01-01 10:00:00"], "text": ["内容"]})
        with pytest.raises(DataContractError):
            nf._parse_sina(bad)

    def test_parse_ths_rejects_column_drift(self):
        """原实现全部用 r.get()：列名漂移会静默产出空标题+空内容的记录"""
        from data.contracts import DataContractError
        bad = pd.DataFrame({"foo": ["a"], "bar": ["b"]})
        with pytest.raises(DataContractError):
            nf._parse_ths_generic(bad)

    def test_norm_time_garbage(self):
        text = "not a date"
        assert nf._norm_time(text)  # 回退当前时间


class TestEnrich:
    @pytest.fixture(autouse=True)
    def _fixed_pool(self, monkeypatch):
        """固定关注池：_enrich 现在读的是关注池（自选股∪持仓∪今日候选），
        不隔离就会依赖真实 config/watchlist.json 与本地库，测试会随用户数据漂移。"""
        import data.watchlist as wl
        monkeypatch.setattr(wl, "attention_pool", lambda *a, **k: [
            {"code": "600519", "name": "贵州茅台", "source": "watchlist"},
            {"code": "300750", "name": "宁德时代", "source": "picks"},
        ])
        yield

    def test_hash_sentiment_and_association(self):
        items = [{"title": "贵州茅台获增持", "content": "利好 600519 业绩预增", "source": "ths",
                  "published_at": _ago_str(minutes=10)}]
        enriched = nf._enrich(items)
        item = enriched[0]
        assert item["hash"] == hashlib.md5("贵州茅台获增持|利好 600519 业绩预增|ths".encode()).hexdigest()
        assert "600519" in item["related_codes"]  # 代码与名称双匹配
        assert item["sentiment"] == "positive"

    def test_candidate_in_pool_is_associated(self):
        """Phase 4.1：池子里的**非自选股候选**也要建立关联（原先只关联自选股）"""
        items = [{"title": "宁德时代新产能投产", "content": "预计明年放量", "source": "sina",
                  "published_at": _ago_str(minutes=5)}]
        item = nf._enrich(items)[0]
        assert item["related_codes"] == ["300750"]

    def test_pool_failure_falls_back_to_watchlist(self, monkeypatch):
        """关注池整体失败 → 退回自选股，不能让新闻采集跟着失败"""
        import data.watchlist as wl
        monkeypatch.setattr(wl, "attention_pool",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("池子坏了")))
        monkeypatch.setattr(wl, "load_watchlist", lambda: [
            {"code": "600519", "name": "贵州茅台"}])
        items = [{"title": "贵州茅台公告", "content": "分红", "source": "sina",
                  "published_at": _ago_str(minutes=5)}]
        assert nf._enrich(items)[0]["related_codes"] == ["600519"]

    def test_sector_keywords(self):
        items = [{"title": "半导体芯片大涨价", "content": "AI算力需求爆发", "source": "sina",
                  "published_at": _ago_str(minutes=10)}]
        item = nf._enrich(items)[0]
        assert "科技" in item["sectors"]


class TestSentimentWordlist:
    """情绪词表（2026-10-21 补齐利空词）。

    **起因**：原词表只有「下滑 / 预亏」，于是下面这两类被判成 **neutral**：

        「第三季度净利润亏损 5231 万元，同比由盈转亏」
        「净利润 1539.98 万元，同比下降 38.06%」

    把**利空当中性**。对"利好选股"影响不大（漏掉的是利空），但对**卖出提醒**
    是致命的——该报的利空报不出来。补齐 `下降/亏损/由盈转亏/净流出` 四个词。

    ⚠️ 注意：`news_items.sentiment` 是**入库当时**打的标，改词表不会回填历史行；
       要拿新标签必须实时调 `classify_sentiment`（`news_sentiment_for_stock` 已经是）。
    """

    @pytest.mark.parametrize("text", [
        "瑞泰科技：2026年第三季度归母净利润亏损5231.23万元，同比由盈转亏",
        "东华测试：2026年第三季度净利润1539.98万元，同比下降38.06%",
        "北向资金今日净流出52亿元",
    ])
    def test_bearish_news_is_negative(self, text):
        assert nf.classify_sentiment(text) == "negative"

    @pytest.mark.parametrize("text", [
        "西部矿业：2026年第三季度净利润26.2亿元，同比增长143.48%",
        "双塔食品：Q3净利润723.64万元，同比增长136.33%",
    ])
    def test_bullish_news_is_positive(self, text):
        assert nf.classify_sentiment(text) == "positive"

    def test_rate_cut_is_not_misjudged_as_bearish(self):
        """⭐ 反向保护：刻意**没有**把「下调」加进利空词。

        「下调存款准备金率 / 下调利率」是宽松**利好**，而「下调评级 / 下调目标价」
        才是利空。单个「下调」两义，加进去会稳定误判前者。要覆盖后者应加
        「下调评级」「下调目标价」这样的长词组。
        """
        assert nf.classify_sentiment("央行：下调存款准备金率0.5个百分点") != "negative"

    @pytest.mark.parametrize("text", [
        "毛利率提升，期间费用同比下降",
        "资产负债率下降至40%，财务更稳健",
        "存货周转加快，库存同比下降",
    ])
    def test_benign_drop_is_not_bearish(self, text):
        """⭐ 良性下降抑制（2026-10-22）。

        补了"下降"之后，`费用同比下降` 会被判成利空 —— 实测今日 599 条真实新闻
        里 24 条翻负、**0 条误伤**（都是真利空），但机制上这个隐患成立。
        """
        assert nf.classify_sentiment(text) != "negative"

    def test_benign_suppression_does_not_hide_a_real_loss(self):
        """抑制不能过头：同一句里"费用上升 + 净利润下降"仍必须是利空。

        这是抑制逻辑最危险的失效方式 —— 一旦按"句子里出现过费用"就整体豁免，
        真正的坏消息会被吞掉。
        """
        assert nf.classify_sentiment("费用上升，但净利润同比下降") == "negative"


class TestFetchAndStore:
    def test_sources_fallback_and_circuit(self, temp_store, monkeypatch):
        """单源失败进熔断；下一轮直接跳过（2026-10-26 换源后由 futu 承担该角色）"""
        calls = []

        def fake_fetch(name):
            calls.append(name)
            if name == "futu":
                raise TimeoutError(">30s")
            if name == "sina":
                return nf._parse_sina(SINA_DF)
            return nf._parse_ths_generic(THS_DF)

        monkeypatch.setattr(nf, "_fetch_source", fake_fetch)
        out1 = nf.fetch_and_store()
        assert out1["sources"]["sina"] == 3  # SINA_DF 3行
        assert "失败" in str(out1["sources"]["futu"])
        assert nf._circuit["futu"] > 0
        out2 = nf.fetch_and_store()
        assert "熔断中" in str(out2["sources"]["futu"])
        assert calls.count("futu") == 1  # 冷却期内不再尝试

    def test_dedup_across_runs(self, temp_store, monkeypatch):
        monkeypatch.setattr(nf, "_fetch_source",
                            lambda name: [] if name == "futu" else nf._parse_sina(SINA_DF))
        first = nf.fetch_and_store()
        second = nf.fetch_and_store()
        assert first["new"] == 3
        assert second["new"] == 0  # 全部重复

    def test_news_sentiment_for_stock(self, temp_store, monkeypatch):
        monkeypatch.setattr(nf, "_fetch_source",
                            lambda name: [] if name == "futu" else nf._parse_sina(SINA_DF))
        nf.fetch_and_store()
        stat = nf.news_sentiment_for_stock("600519", days=3)
        assert stat is not None
        assert stat["total"] >= 1 and stat["positive"] >= 1
        assert nf.news_sentiment_for_stock("999999") is None

    def test_news_sentiment_matches_by_stock_name(self, temp_store, monkeypatch):
        """v2（0.2.3）：非自选股可按股票简称匹配新闻。

        入库时 related_codes 只关联自选股（_enrich 只遍历 watchlist），
        若查询只按代码过滤，全市场候选的"新闻正面率"因子将永远无数据。
        这里用 002415 海康威视（不在 config/watchlist.json 中）验证。
        """
        df = pd.DataFrame([
            {"时间": _SINA_T1.strftime("%Y-%m-%d %H:%M:%S"),
             "内容": "海康威视(002415) 发布业绩预增公告，机构上调评级"},
        ])
        monkeypatch.setattr(nf, "_fetch_source",
                            lambda name: [] if name == "futu" else nf._parse_sina(df))
        nf.fetch_and_store()

        assert nf.news_sentiment_for_stock("002415") is None, "未关联自选股代码时按代码查不到"
        stat = nf.news_sentiment_for_stock("002415", name="海康威视")
        assert stat is not None, "按股票简称必须能命中"
        assert stat["total"] >= 1

    def test_sentiment_is_recomputed_not_read_from_db(self, temp_store):
        """库内 `sentiment` 列可能是**旧词表**打的标 —— 统计时必须重算。

        ⚠️ 原实现直接 `i.get("sentiment")`：词表一改，库内标签与当前词表就永久分叉
        （实测 2026-10-07：库内 2172 条快讯里 **71 条**重算结果与库内不一致）。
        后果是同一个舆情因子有两条口径——"库内快讯"读旧标、"个股直查"用
        `classify_sentiment` 新算，两条路给出的正面率可能不同且无从分辨。

        这里故意插入 `sentiment="neutral"` 但正文明显正面的快讯：
        只有**重算**才会得出 positive == 1（旧实现会得 0，本用例即变红）。
        """
        db_store.insert_news([
            {"hash": "recompute-1", "title": "某公司业绩预增股价大涨",
             "content": "利好落地，机构上调评级", "source": "sina",
             "published_at": _ago_str(minutes=10),
             "related_codes": ["600519"], "sentiment": "neutral"},
        ])
        stat = nf.news_sentiment_for_stock("600519", days=3)
        assert stat is not None
        assert stat["total"] == 1
        assert stat["positive"] == 1, "必须按当前词表重算，不能读库内旧标签"
        assert stat["positive_rate"] == 1.0


class TestScreenerNewsFactor:
    def test_news_factor_with_data(self, temp_store, monkeypatch):
        from core.screener import StockScreener
        s = StockScreener()
        maotai_df = pd.DataFrame([
            {"时间": _SINA_T1.strftime("%Y-%m-%d %H:%M:%S"), "内容": "贵州茅台(600519) 发布利好公告，业绩预增"},
            {"时间": _SINA_T2.strftime("%Y-%m-%d %H:%M:%S"), "内容": "贵州茅台获北向资金增持，股价上涨"},
        ])
        monkeypatch.setattr(nf, "_fetch_source",
                            lambda name: [] if name == "futu" else nf._parse_sina(maotai_df))
        nf.fetch_and_store()
        result = s._score_news_factor("600519", max_score=15)
        assert result["max"] == 15
        assert result["status"] == "ok"
        assert "正面率" in result["detail"]
        assert result["score"] >= 0

    def test_news_factor_no_data_is_unknown(self, temp_store):
        """v2：无关联新闻标记 unknown，不再给 7 分"中性占位"。

        原占位让这个 15% 权重的因子对几乎所有非自选股候选贡献同一个常数：
        看起来七因子都算了，实际该维度零信息量，且拉平了所有候选的分数差异。
        """
        from core.screener import StockScreener
        s = StockScreener()
        result = s._score_news_factor("999999", max_score=15)
        assert result["score"] is None
        assert result["status"] == "unknown"
        assert "无关联新闻" in result["detail"]


# ===== smart_latest_news（自动补采 + 关键词回退）=====

class TestSmartLatestNews:
    def test_keyword_fallback_to_recent(self, temp_store, monkeypatch):
        """关键词无命中 → 回退返回最近新闻并说明"""
        monkeypatch.setattr(nf, "_fetch_source",
                            lambda name: [] if name == "futu" else nf._parse_sina(SINA_DF))
        nf.fetch_and_store()
        result = nf.smart_latest_news(limit=5, keyword="今天")
        assert result["total"] == 3          # 回退返回全部3条
        assert "今天" in result["note"]

    def test_fresh_data_no_refetch(self, temp_store, monkeypatch):
        """库内数据新鲜时不触发补采"""
        monkeypatch.setattr(nf, "_fetch_source",
                            lambda name: [] if name == "futu" else nf._parse_sina(SINA_DF))
        nf.fetch_and_store()
        # 新鲜度判定依赖库内最新时间，这里 mock 统计返回"刚刚更新"以保证确定性
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        monkeypatch.setattr(nf.store, "news_stats",
                            lambda: {"total": 3, "today": 3, "latest": now_str})
        calls = []
        real_fetch = nf.fetch_and_store
        def spy():
            calls.append(1)
            return real_fetch()
        monkeypatch.setattr(nf, "fetch_and_store", spy)
        result = nf.smart_latest_news(limit=5, keyword="贵州茅台")
        assert calls == []                    # 新鲜数据未补采
        assert result["auto_refreshed"] is False

    def test_stale_data_triggers_refresh(self, temp_store, monkeypatch):
        """库内新闻陈旧(>60分钟) → 查询前自动补采"""
        monkeypatch.setattr(nf, "_fetch_source",
                            lambda name: [] if name == "futu" else nf._parse_sina(SINA_DF))
        nf.fetch_and_store()
        # 把库内时间改旧 2 小时
        import sqlite3
        conn = sqlite3.connect(str(temp_store))
        conn.execute("UPDATE news_items SET published_at = datetime('now', '-2 hours')")
        conn.commit()
        conn.close()
        refreshed = []
        real_fetch = nf.fetch_and_store
        def spy():
            refreshed.append(1)
            return real_fetch()
        monkeypatch.setattr(nf, "fetch_and_store", spy)
        result = nf.smart_latest_news(limit=5, keyword="茅台")
        assert len(refreshed) == 1            # 陈旧数据触发了补采
        assert result["auto_refreshed"] is True


# ================================================================
# 9. 换源 + 请求超时（2026-10-26）
# ================================================================
#
# 事故：cls 每小时必然白等满 30s，日志 7 条 `>30s`，40 条新闻照收 ——
# 看起来"只是慢"，实际是三个问题叠在一起：
#   ① 上游 `www.cls.cn/nodeapi/telegraphList` **已 404**（接口迁移，返回 HTML 错误页）；
#   ② akshare 的 `make_request_with_retry_json(url, max_retries=10)` 对着必 404 的地址重试 10 次；
#   ③ 最要命：它内部 `requests.get(url, params=..., headers=..., proxies=proxies)`
#      **完全没有 timeout** ⇒ 源不响应时那个线程**永久阻塞**。
#      外层 `join(FETCH_TIMEOUT)` 只是"不再等"，线程还挂在 socket 上 ——
#      7 次采集 = 7 个泄漏线程。

class TestSourceRoster:
    def test_cls_retired_and_replacements_present(self):
        """cls 必须摘掉；东财(em)/富途(futu) 必须接上。

        ⚠️ 这条守的是"退役的源不会悄悄回来"：cls 留着的话，每小时那趟采集
        会继续白等并泄漏线程，而日志只写一行 WARNING，很容易被忽略。
        """
        assert "cls" not in nf._SOURCES, "cls 上游已 404，留着只会泄漏线程"
        assert "em" in nf._SOURCES and "futu" in nf._SOURCES
        assert "sina" in nf._SOURCES and "ths" in nf._SOURCES
        # 熔断表必须与源表同构，否则失败的源取不到冷却位 ⇒ KeyError
        assert set(nf._circuit) == set(nf._SOURCES)

    def test_fetch_timeout_is_short(self):
        """单源实测全部 1.3s 级 ⇒ 12s 已是 9 倍余量，不该再是 30s。"""
        assert nf.FETCH_TIMEOUT <= 15, \
            "四源串行时 30s×4=120s，足以拖住整条每小时任务"

    def test_em_parser_reads_summary_column(self):
        """⚠️ 东财把正文放在 **`摘要`**，不是 `内容`（同花顺/富途是 `内容`）。

        认错列名的后果不是报错，而是**静默入库一堆只有标题、正文空白的记录** ——
        它们会在情绪打分里全判"neutral"，等于白收。
        """
        df = pd.DataFrame({"标题": ["某公司中标"], "摘要": ["正文在这里"],
                           "发布时间": ["2026-10-26 17:09:00"], "链接": ["http://x"]})
        rows = nf._parse_em(df)
        assert rows[0]["content"] == "正文在这里"
        assert rows[0]["title"] == "某公司中标"

    def test_em_parser_accepts_content_column_too(self):
        """列名漂移时退一步，而不是把整条源判死。"""
        df = pd.DataFrame({"标题": ["t"], "内容": ["c"], "发布时间": ["2026-10-26 17:09:00"]})
        assert nf._parse_em(df)[0]["content"] == "c"

    def test_em_parser_rejects_when_body_column_missing(self):
        """既没 `摘要` 也没 `内容` ⇒ 必须抛，不能产出"只有标题"的空壳。"""
        from data.contracts import DataContractError

        df = pd.DataFrame({"标题": ["t"], "发布时间": ["2026-10-26 17:09:00"]})
        with pytest.raises(DataContractError):
            nf._parse_em(df)

    def test_futu_parser(self):
        df = pd.DataFrame({"标题": ["t"], "内容": ["c"], "发布时间": ["2026-10-26 17:08:58"]})
        rows = nf._parse_futu(df)
        assert rows[0]["content"] == "c" and rows[0]["published_at"].startswith("2026-10-26")


class TestRequestTimeoutInjection:
    """⭐ 老实话：本项目**早就有**一套超时补丁（`data/akshare_helper.patch_requests()`，
    注入 30s 默认超时），所以"akshare 完全没超时"是错的。
    cls 的真实死因是「30s × 内部 max_retries=10」= 最坏 300s，而外层 `join(30)` 只等 30s
    ⇒ 日志报 `>30s`，线程继续跑 270s 才退（这才是泄漏）。

    因此本类的钩子采用 **min 语义**（只收紧、不放宽），并装在 akshare_helper 外层。
    下面的用例直接调钩子本身，不绕 `requests.get` 全链路 ——
    否则会连 akshare_helper 的 `_patched_get`（它也注入 30s）一起测进来，
    断言值取决于"谁先装的包装"，-n 4 时单文件绿、组合红就是这么来的。
    """

    def _install(self, monkeypatch):
        import requests

        seen = {}

        def fake_bottom(self, method, url, **kwargs):
            seen["timeout"] = kwargs.get("timeout")
            return "ok"

        monkeypatch.setattr(nf, "_request_patch_installed", False)
        hook = nf._install_request_timeout(orig=fake_bottom)
        return hook, seen, requests

    def test_budget_is_injected_when_timeout_absent(self, monkeypatch):
        hook, seen, _ = self._install(monkeypatch)
        nf._req_local.timeout = nf.FETCH_TIMEOUT
        try:
            hook(None, "GET", "http://x")          # ⚠️ 故意不传 timeout
        finally:
            del nf._req_local.timeout
        assert seen["timeout"] == nf.FETCH_TIMEOUT

    def test_looser_timeout_is_tightened(self, monkeypatch):
        """⭐⭐ akshare_helper 给的 30s 必须被压到预算 —— 这正是 cls 卡 30s 的解。"""
        hook, seen, _ = self._install(monkeypatch)
        nf._req_local.timeout = nf.FETCH_TIMEOUT
        try:
            hook(None, "GET", "http://x", timeout=30)     # 老补丁的默认值
        finally:
            del nf._req_local.timeout
        assert seen["timeout"] == nf.FETCH_TIMEOUT, "30s 必须收紧到 12s"

    def test_tighter_timeout_is_kept(self, monkeypatch):
        """⚠️ min 语义的另一半：调用方显式设得更小，**不许放宽**。

        放宽会把"我自己只要 2 秒"的调用变成"最多等 12 秒"，
        那是悄悄改变别人的超时契约。
        """
        hook, seen, _ = self._install(monkeypatch)
        nf._req_local.timeout = nf.FETCH_TIMEOUT
        try:
            hook(None, "GET", "http://x", timeout=2)
        finally:
            del nf._req_local.timeout
        assert seen["timeout"] == 2, "更小的显式超时不能被放宽"

    def test_tuple_timeout_is_tightened_per_element(self, monkeypatch):
        """requests 允许 `(connect, read)` 元组。"""
        hook, seen, _ = self._install(monkeypatch)
        nf._req_local.timeout = nf.FETCH_TIMEOUT
        try:
            hook(None, "GET", "http://x", timeout=(2, 60))
        finally:
            del nf._req_local.timeout
        assert seen["timeout"] == (2, nf.FETCH_TIMEOUT)

    def test_no_budget_leaves_timeout_untouched(self, monkeypatch):
        """⚠️ 没设预算的线程（本项目绝大多数请求）行为必须与改造前**逐字节一致**。"""
        hook, seen, _ = self._install(monkeypatch)
        if hasattr(nf._req_local, "timeout"):
            del nf._req_local.timeout
        hook(None, "GET", "http://x")
        assert seen["timeout"] is None
        hook(None, "GET", "http://x", timeout=30)
        assert seen["timeout"] == 30, "无预算时不许改动既有超时"

    def test_install_is_idempotent(self, monkeypatch):
        """重复装不能叠加包装（否则每采一次多套一层）。"""
        hook, _, requests = self._install(monkeypatch)
        installed = requests.Session.request
        again = nf._install_request_timeout()
        assert again is installed, "第二次必须原样返回已装的钩子"

    def test_fetch_thread_sets_the_budget(self, monkeypatch):
        """`_fetch_source` 必须在**子线程**里把预算落到 thread-local。"""
        seen = {}

        def source_probe():
            seen["budget"] = getattr(nf._req_local, "timeout", None)
            return None                    # `_fetch_source` 对 None 直接返回空列表

        monkeypatch.setitem(nf._SOURCES, "probe", (source_probe, lambda df: []))
        nf._fetch_source("probe")
        assert seen["budget"] == nf.FETCH_TIMEOUT, \
            "采集线程没设预算 ⇒ 上面的收紧逻辑等于没装"
