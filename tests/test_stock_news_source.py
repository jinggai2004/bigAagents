"""按个股直查新闻/公告源契约测试（v4，2026-09-28）

背景：消息面长期 0 条的**根因不在节点逻辑，而在源的分辨率**。
`sina`/`ths`/`cls` 三源都是"全市场快讯"，只播热门事件——实测库内 583 条快讯
按 `600211`/「西藏药业」匹配命中 **0 条**。v3 已把节点修成"如实报告"，
但"如实报告 0 条"对分析毫无信息量，所以 v4 接两个**按个股直查**的源：
东财个股新闻（0.13~0.38s / 10 条）与东财个股公告（1.2~1.4s）。

本文件守护的**不是**"函数能跑"，而是下面这些实测踩出来的坑：

  1. 北交所裸代码 `832000` 调 `ak.stock_news_em` 直接抛 `KeyError: 'code'`，
     必须补 `.BJ` 后缀；
  2. 源返回的是**近 1~1.5 个月**的 10 条而不是近 7 天，硬过滤会退化成"近乎 0 条"，
     所以有三级时间放宽，且必须把"放宽了"写进 note；
  3. `ak.stock_individual_notice_report` 的**窗口内无公告时抛 `KeyError: '代码'`**
     ——那是空结果不是源故障，若判成故障会进 1800s 熔断，把公告源永久关掉；
  4. 参数名是 `security` 不是 `symbol`（传错直接 TypeError）。

方法论沿用项目铁律：**只 mock 叶子依赖（akshare 函数 / 上游源函数），
不 mock 被测对象本身**——`tests/test_news_stock_source.py` 的反面教材就是
`@patch` 整个类之后"方法名写错也全绿"。
"""

import inspect
import pathlib
import re
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
import pytest

from data import news_feed
from data.news import NewsFlowDataFetcher


# ================================================================
# 工具
# ================================================================

def _news_df(rows, code="600211"):
    """构造东财个股新闻的返回结构（列名与实测一致）。

    `关键词` 列是源自己声明的"这条新闻属于谁"，**必须与请求代码一致**——
    它正是 `_em_rows_match_code` 闸门的依据。
    """
    return pd.DataFrame([
        {"关键词": r.get("关键词", code),
         "新闻标题": r.get("title", ""),
         "新闻内容": r.get("content", ""),
         "发布时间": r.get("published_at", ""),
         "文章来源": r.get("source", "上海证券报"),
         "新闻链接": r.get("url", "")}
        for r in rows])


def _notice_df(rows):
    return pd.DataFrame([
        {"代码": r.get("code", "600211"), "名称": r.get("name", "西藏药业"),
         "公告标题": r.get("title", ""), "公告类型": r.get("kind", "调研活动"),
         "公告日期": r.get("date", ""), "网址": r.get("url", "")}
        for r in rows])


def _ago(**kw):
    return (datetime.now() - timedelta(**kw)).strftime("%Y-%m-%d %H:%M:%S")


def _item(title, source="界面新闻", url="http://x/1"):
    return {"title": title, "content": "内容", "source": source,
            "published_at": "2026-08-14 09:24:26", "url": url}


def _empty():
    return {"items": [], "note": None, "error": None}


def _patch_direct(monkeypatch, news=None, notices=None):
    """把两个"按个股直查"的源换成叶子级替身"""
    monkeypatch.setattr(news_feed, "fetch_stock_news",
                        lambda *a, **k: news if news is not None else _empty())
    monkeypatch.setattr(news_feed, "fetch_stock_notices",
                        lambda *a, **k: notices if notices is not None else _empty())


@pytest.fixture(autouse=True)
def _clean_source_state():
    """每个用例前清缓存与熔断状态（模块级状态会跨用例串味）"""
    news_feed.clear_stock_cache()
    yield
    news_feed.clear_stock_cache()


# ================================================================
# 1. 接口契约：叶子函数真的存在，且参数名对得上
# ================================================================

def test_akshare_news_leaf_exists():
    assert callable(getattr(ak, "stock_news_em", None)), \
        "data/news_feed 依赖的 ak.stock_news_em 不存在（akshare 版本漂移）"


def test_akshare_notice_leaf_exists():
    assert callable(getattr(ak, "stock_individual_notice_report", None)), \
        "data/news_feed 依赖的 ak.stock_individual_notice_report 不存在"


def test_notice_leaf_first_param_is_security_not_symbol():
    """参数名是 `security`。写成 `symbol=` 会直接 TypeError，
    而 TypeError 会被 except 兜成"公告源失败"——又是静默降级。"""
    params = list(inspect.signature(ak.stock_individual_notice_report).parameters)
    assert params[0] == "security", params


def test_fetch_helpers_call_the_leaves():
    """反向断言：真去调叶子，而不是"返回空也算成功" """
    assert "ak.stock_news_em" in inspect.getsource(news_feed.fetch_stock_news)
    assert ("ak.stock_individual_notice_report"
            in inspect.getsource(news_feed.fetch_stock_notices))


# ================================================================
# 2. 相关性闸门：带后缀会静默返回"无关的通用新闻流"
# ================================================================

def test_only_bare_code_is_ever_passed_to_leaf(monkeypatch):
    """带交易所后缀的调用**不报错**，而是静默返回与该股无关的通用流
    （实测：'600211.SH' 的正文是赛伦生物/力鼎光电）。

    所以要断言叶子收到的永远是裸 6 位代码——这个错误值钱在"不报错"。
    """
    seen = []
    monkeypatch.setattr(
        news_feed.ak, "stock_news_em",
        lambda symbol: seen.append(symbol) or _news_df(
            [{"title": "x", "published_at": _ago(days=1)}], code="600211"))
    news_feed.fetch_stock_news("600211")
    assert seen == ["600211"], f"不得传交易所后缀，实际 {seen}"


def test_unrelated_feed_is_rejected_not_fed_to_model(monkeypatch):
    """错数据比没数据危险得多：模型会把「赛伦生物获菲律宾GMP证书」当作
    西藏药业的新闻来分析，而且全程不报错。所以整批拒收。"""
    monkeypatch.setattr(news_feed.ak, "stock_news_em", lambda symbol: _news_df(
        [{"关键词": "600211.SH", "title": "赛伦生物688163.SH：获菲律宾GMP符合性证书",
          "published_at": _ago(days=1)}], code="600211.SH"))
    out = news_feed.fetch_stock_news("600211")

    assert out["items"] == [], "无关新闻不得进入舆情输入"
    assert out["error"] and "不一致" in out["error"]
    assert "裸 6 位代码" in out["error"], "错误信息要能直接指向修法"


def test_matching_feed_is_accepted(monkeypatch):
    monkeypatch.setattr(news_feed.ak, "stock_news_em", lambda symbol: _news_df(
        [{"title": "西藏药业中报", "published_at": _ago(days=1)}]))
    out = news_feed.fetch_stock_news("600211")
    assert out["items"] and out["error"] is None


def test_partially_matching_feed_is_also_rejected(monkeypatch):
    """只要有一行声明不是本代码，说明返回的是混流 → 整批拒收（宁可 0 条）"""
    monkeypatch.setattr(news_feed.ak, "stock_news_em", lambda symbol: _news_df([
        {"关键词": "600211", "title": "本股新闻", "published_at": _ago(days=1)},
        {"关键词": "600519", "title": "茅台的新闻", "published_at": _ago(days=1)}]))
    out = news_feed.fetch_stock_news("600211")
    assert out["items"] == [] and out["error"]


def test_missing_keyword_column_fails_loudly(monkeypatch):
    """`关键词` 列消失意味着闸门失效——必须显式失败而不是放行"""
    monkeypatch.setattr(news_feed.ak, "stock_news_em",
                        lambda symbol: pd.DataFrame([{"新闻标题": "t",
                                                      "发布时间": _ago(days=1)}]))
    out = news_feed.fetch_stock_news("600211")
    assert out["items"] == [] and out["error"]


# ================================================================
# 2b. 北交所：裸码 KeyError 是"没收录"，不是"源坏了"
# ================================================================

def test_beijing_stock_without_news_is_empty_not_circuit(monkeypatch):
    """实测 871981/830799/832000 裸码直接抛 `KeyError: 'code'`。

    那是"该源没有这只票的新闻"的信号。若判成源故障就会进 1800s 熔断——
    一次北交所分析就能把新闻源关掉 30 分钟（经典的"降级链把自己关掉"）。
    """
    def _boom(symbol):
        raise KeyError("code")

    monkeypatch.setattr(news_feed.ak, "stock_news_em", _boom)
    out = news_feed.fetch_stock_news("832000")

    assert out["items"] == [] and out["error"] is None
    assert "未收录" in out["note"]
    assert news_feed._stock_circuit[news_feed.EM_NEWS_SOURCE] == 0.0, "不得触发熔断"


def test_beijing_stock_must_not_fall_back_to_bj_suffix(monkeypatch):
    """反面：绝不能靠 `.BJ` 后缀"救"北交所——那只会拿回无关的通用流"""
    seen = []
    monkeypatch.setattr(news_feed.ak, "stock_news_em",
                        lambda symbol: seen.append(symbol) or _news_df([]))
    news_feed.fetch_stock_news("832000")
    assert seen == ["832000"], f"不得尝试后缀兜底，实际 {seen}"


def test_real_failure_still_returns_error_and_does_not_raise(monkeypatch):
    """取数失败必须返回 error 字典，**绝不抛异常**——它挂在整图的一个节点上"""
    def _boom(symbol):
        raise RuntimeError("东财 500")

    monkeypatch.setattr(news_feed.ak, "stock_news_em", _boom)
    out = news_feed.fetch_stock_news("600211")

    assert out["items"] == []
    assert out["error"] and "东财 500" in out["error"]


def test_bad_code_rejected_without_network(monkeypatch):
    calls = []
    monkeypatch.setattr(news_feed.ak, "stock_news_em",
                        lambda symbol: calls.append(symbol) or _news_df([]))
    out = news_feed.fetch_stock_news("abc")
    assert calls == [] and out["error"]


# ================================================================
# 3. 空结果 / 列名漂移
# ================================================================

def test_empty_dataframe_is_empty_result_not_error(monkeypatch):
    monkeypatch.setattr(news_feed.ak, "stock_news_em", lambda symbol: pd.DataFrame())
    out = news_feed.fetch_stock_news("600211")
    assert out["items"] == [] and out["error"] is None
    assert "0 条" in out["note"]


def test_none_dataframe_is_empty_result_not_error(monkeypatch):
    monkeypatch.setattr(news_feed.ak, "stock_news_em", lambda symbol: None)
    out = news_feed.fetch_stock_news("600211")
    assert out["items"] == [] and out["error"] is None


def test_column_drift_fails_loudly_instead_of_silent_empty_records(monkeypatch):
    """列名漂移必须显式失败——否则会静默产出"空标题+空内容"的记录，
    再去重成同一个 hash 的空条目，日志里毫无痕迹（项目历史高频坑）"""
    monkeypatch.setattr(
        news_feed.ak, "stock_news_em",
        lambda symbol: pd.DataFrame([{"标题": "改了列名", "时间": "2026-09-01"}]))
    out = news_feed.fetch_stock_news("600211")
    assert out["items"] == []
    assert out["error"] and "东财个股新闻" in out["error"]


# ================================================================
# 4. 字段映射
# ================================================================

def test_item_fields_map_to_format_news_block_shape(monkeypatch):
    monkeypatch.setattr(news_feed.ak, "stock_news_em", lambda symbol: _news_df([{
        "title": "西藏药业2026年中报净利润为6.08亿元",
        "content": "界面新闻 2026年8月14日",
        "published_at": "2026-08-14 09:24:26",
        "source": "界面新闻",
        "url": "http://finance.eastmoney.com/a/1.html"}]))
    out = news_feed.fetch_stock_news("600211", days=90)

    it = out["items"][0]
    for key in ("title", "content", "source", "published_at", "url"):
        assert it.get(key), f"缺少 {key}"
    assert it["source"] == "界面新闻"
    assert it["published_at"] == "2026-08-14 09:24:26"
    assert it["url"].startswith("http")
    assert "_dt" not in it, "内部排序用的 _dt 不得泄漏给调用方"


def test_items_sorted_newest_first(monkeypatch):
    monkeypatch.setattr(news_feed.ak, "stock_news_em", lambda symbol: _news_df([
        {"title": "旧", "published_at": _ago(days=40)},
        {"title": "新", "published_at": _ago(days=2)},
        {"title": "中", "published_at": _ago(days=10)}]))
    out = news_feed.fetch_stock_news("600211", days=7)
    assert [i["title"] for i in out["items"]][:3] == ["新", "中", "旧"]


# ================================================================
# 5. 三级时间放宽（本方案最容易做错的地方）
# ================================================================

def test_within_window_when_enough_items(monkeypatch):
    rows = [{"title": f"t{i}", "published_at": _ago(days=1)} for i in range(5)]
    monkeypatch.setattr(news_feed.ak, "stock_news_em", lambda symbol: _news_df(rows))
    out = news_feed.fetch_stock_news("600211", days=7)
    assert len(out["items"]) == 5
    assert out["window"] == "近7天"
    assert "放宽" not in out["note"]


def test_widens_to_30_days_when_window_too_thin(monkeypatch):
    rows = ([{"title": f"w{i}", "published_at": _ago(days=2)} for i in range(2)]
            + [{"title": f"o{i}", "published_at": _ago(days=20)} for i in range(5)])
    monkeypatch.setattr(news_feed.ak, "stock_news_em", lambda symbol: _news_df(rows))
    out = news_feed.fetch_stock_news("600211", days=7)
    assert len(out["items"]) == 7, "应放宽到近 30 天并取回 7 条"
    assert "放宽至近30天" in out["window"]


def test_takes_all_items_when_even_30_days_is_thin(monkeypatch):
    rows = ([{"title": "近的", "published_at": _ago(days=2)}]
            + [{"title": f"远{i}", "published_at": _ago(days=60)} for i in range(9)])
    monkeypatch.setattr(news_feed.ak, "stock_news_em", lambda symbol: _news_df(rows))
    out = news_feed.fetch_stock_news("600211", days=7)
    assert len(out["items"]) == 10
    assert "全部10条" in out["window"]


def test_note_always_states_freshness_so_model_cannot_mistake_old_for_recent(monkeypatch):
    """放宽必须写进 note：否则模型会把一个月前的新闻当"近期事件"来解读"""
    rows = [{"title": f"t{i}", "published_at": _ago(days=45)} for i in range(4)]
    monkeypatch.setattr(news_feed.ak, "stock_news_em", lambda symbol: _news_df(rows))
    out = news_feed.fetch_stock_news("600211", days=7)
    assert out["note"] and "最近一条" in out["note"] and "距今" in out["note"]


def test_unparseable_published_is_none_not_now():
    """解析失败返回 None（而不是 `_norm_time` 那样回落到 now()）——
    回落到 now() 会让三条陈年新闻集体化身"刚刚发布"。"""
    assert news_feed._parse_published("不是时间") is None
    assert news_feed._parse_published("") is None
    assert news_feed._parse_published("2026-08-14") is not None


def test_slash_date_format_still_parsed(monkeypatch):
    monkeypatch.setattr(news_feed.ak, "stock_news_em",
                        lambda symbol: _news_df([{"title": "怪时间",
                                                  "published_at": "2026/08/14"}]))
    out = news_feed.fetch_stock_news("600211", days=90)
    assert out["items"]


# ================================================================
# 6. 缓存与熔断
# ================================================================

def test_cache_prevents_duplicate_fetch_within_ttl(monkeypatch):
    calls = []

    def _fake(symbol):
        calls.append(symbol)
        return _news_df([{"title": "x", "published_at": _ago(days=1)}])

    monkeypatch.setattr(news_feed.ak, "stock_news_em", _fake)
    first = news_feed.fetch_stock_news("600211")
    second = news_feed.fetch_stock_news("600211")

    assert len(calls) == 1, "TTL 内二次调用不应重复请求"
    assert first["items"] == second["items"]
    assert second.get("cached") is True


def test_cached_payload_cannot_be_mutated_by_caller(monkeypatch):
    monkeypatch.setattr(news_feed.ak, "stock_news_em",
                        lambda symbol: _news_df([{"title": "x",
                                                  "published_at": _ago(days=1)}]))
    first = news_feed.fetch_stock_news("600211")
    first["items"][0]["title"] = "被改坏了"
    second = news_feed.fetch_stock_news("600211")
    assert second["items"][0]["title"] == "x"


def test_failure_trips_circuit_and_stops_hammering(monkeypatch):
    """失败进冷却：否则一个持续失败的源会让每只票都白等一个超时"""
    calls = []

    def _boom(symbol):
        calls.append(symbol)
        raise RuntimeError("东财 500")

    monkeypatch.setattr(news_feed.ak, "stock_news_em", _boom)
    first = news_feed.fetch_stock_news("600211")
    n_after_first = len(calls)
    second = news_feed.fetch_stock_news("600211")

    assert first["error"] and "东财 500" in first["error"]
    assert second["error"] and "熔断" in second["error"]
    assert len(calls) == n_after_first == 1, "冷却期内不得再请求叶子"


# ================================================================
# 7. 个股公告源
# ================================================================

def test_notice_empty_window_keyerror_means_no_notice_not_failure(monkeypatch):
    """实测：窗口内无公告时 akshare 抛 `KeyError: '代码'`。

    若把它当源故障，会进 1800s 熔断 → 公告源被自己永久关掉。
    """
    def _boom(**kw):
        raise KeyError("代码")

    monkeypatch.setattr(news_feed.ak, "stock_individual_notice_report", _boom)
    out = news_feed.fetch_stock_notices("600211")

    assert out["items"] == []
    assert out["error"] is None, "空窗口不是源故障"
    assert "无公告" in out["note"]
    assert news_feed._stock_circuit[news_feed.EM_NOTICE_SOURCE] == 0.0, "不得触发熔断"


def test_notice_real_failure_does_trip_circuit(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("502")

    monkeypatch.setattr(news_feed.ak, "stock_individual_notice_report", _boom)
    out = news_feed.fetch_stock_notices("600211")
    assert out["error"] and "502" in out["error"]
    assert news_feed._stock_circuit[news_feed.EM_NOTICE_SOURCE] > 0


def test_notice_passes_date_window_and_security_kwarg(monkeypatch):
    seen = {}

    def _fake(**kw):
        seen.update(kw)
        return _notice_df([{"title": "投资者关系活动记录表(8月)", "kind": "调研活动",
                            "date": _ago(days=19)[:10],
                            "url": "https://data.eastmoney.com/notices/detail/600211/x.html"}])

    monkeypatch.setattr(news_feed.ak, "stock_individual_notice_report", _fake)
    out = news_feed.fetch_stock_notices("600211", days=90)

    assert seen["security"] == "600211"
    assert set(seen) == {"security", "begin_date", "end_date"}, seen
    assert re.fullmatch(r"\d{8}", seen["begin_date"])
    assert re.fullmatch(r"\d{8}", seen["end_date"])
    assert out["items"][0]["title"].startswith("投资者关系")
    assert out["items"][0]["url"].startswith("https://")


def test_notice_item_carries_type_so_reader_knows_what_it_is(monkeypatch):
    monkeypatch.setattr(news_feed.ak, "stock_individual_notice_report",
                        lambda **kw: _notice_df([{"title": "担保进展公告",
                                                  "kind": "对外担保",
                                                  "date": _ago(days=3)[:10]}]))
    out = news_feed.fetch_stock_notices("600211")
    it = out["items"][0]
    assert "对外担保" in it["content"]
    assert "对外担保" in it["source"]


def test_notice_rows_for_other_codes_are_rejected(monkeypatch):
    """公告源同款相关性闸门：`代码` 列不是本股 → 整批拒收。
    公告标题看起来都很"像"，混进来几乎不可能被肉眼发现。"""
    monkeypatch.setattr(news_feed.ak, "stock_individual_notice_report",
                        lambda **kw: _notice_df([{"code": "600519",
                                                  "name": "贵州茅台",
                                                  "title": "贵州茅台:回购进展",
                                                  "date": _ago(days=3)[:10]}]))
    out = news_feed.fetch_stock_notices("600211")
    assert out["items"] == [] and out["error"] and "不一致" in out["error"]


# ================================================================
# 8. 三源合并（data/news.py::get_news_flow_data）
# ================================================================

def test_return_shape_is_backward_compatible(monkeypatch):
    """改造时刻意不改签名与返回结构，老调用方与老断言应全部继续成立"""
    _patch_direct(monkeypatch, news={"items": [_item("A")],
                                     "note": "个股新闻源（东财）：近7天", "error": None})
    monkeypatch.setattr(news_feed, "smart_latest_news",
                        lambda **k: {"news": [], "total": 0, "note": None})
    out = NewsFlowDataFetcher().get_news_flow_data("600211", days=7, stock_name="西藏药业")

    for key in ("data_success", "news_list", "count", "note", "auto_refreshed", "fetch_time"):
        assert key in out, f"返回结构缺 {key}"
    assert out["data_success"] is True and out["count"] == 1


def test_quick_news_library_is_still_consulted_when_direct_source_is_empty(monkeypatch):
    """降级链可达性：直查源没命中时，快讯库**必须**仍被调用。

    这条守卫的是项目反复复发的"降级链实际不可达"坑——旧的问财源就是因为被写在
    `except` 分支里（而主源不再抛异常）而永远走不到。
    """
    _patch_direct(monkeypatch)                       # A / C 都空
    called = []
    monkeypatch.setattr(news_feed, "smart_latest_news",
                        lambda **k: called.append(k) or {"news": [], "note": "没有"})
    NewsFlowDataFetcher().get_news_flow_data("600211")
    assert called, "直查源为空时快讯库必须被调用"


def test_direct_source_is_consulted_even_when_library_would_hit(monkeypatch):
    """反向也一样：库里有数据不能成为"跳过按个股直查"的理由（冷门个股才是主战场）"""
    seen = []
    monkeypatch.setattr(news_feed, "fetch_stock_news",
                        lambda *a, **k: seen.append(k) or _empty())
    monkeypatch.setattr(news_feed, "fetch_stock_notices", lambda *a, **k: _empty())
    monkeypatch.setattr(news_feed, "smart_latest_news",
                        lambda **k: {"news": [_item("库里的")], "note": None})
    out = NewsFlowDataFetcher().get_news_flow_data("600211")
    assert seen, "必须调用按个股直查源"
    assert out["count"] == 1


def test_cross_source_dedup_by_title(monkeypatch):
    same = "西藏药业获批文"
    _patch_direct(monkeypatch, news={"items": [_item(same, "上海证券报")],
                                     "note": None, "error": None})
    monkeypatch.setattr(news_feed, "smart_latest_news",
                        lambda **k: {"news": [{"title": same, "content": "同一条",
                                               "source": "sina",
                                               "published_at": "2026-08-14 09:24:26",
                                               "url": ""}], "note": None})
    out = NewsFlowDataFetcher().get_news_flow_data("600211")
    assert out["count"] == 1, f"同一标题只应保留一条，实际 {out['news_list']}"


def test_sources_list_reports_where_data_came_from(monkeypatch):
    """命中时要能说清"是哪些源给的数据"——这是可核实性的依据"""
    _patch_direct(monkeypatch,
                  news={"items": [_item("个股新闻")], "note": "近7天", "error": None},
                  notices={"items": [_item("公告", "东财公告")], "note": "近90天共1条",
                           "error": None})
    monkeypatch.setattr(news_feed, "smart_latest_news", lambda **k: {"news": [], "note": None})
    out = NewsFlowDataFetcher().get_news_flow_data("600211")
    assert any("个股新闻" in s for s in out["sources"])
    assert any("个股公告" in s for s in out["sources"])


def test_all_sources_empty_reports_error_and_scope_note(monkeypatch):
    _patch_direct(monkeypatch,
                  news={"items": [], "note": None, "error": "个股新闻源失败: 502"},
                  notices={"items": [], "note": None, "error": "个股公告源失败: 502"})
    monkeypatch.setattr(news_feed, "smart_latest_news",
                        lambda **k: {"news": [], "note": "库内 583 条快讯中未检索到"})
    out = NewsFlowDataFetcher().get_news_flow_data("600211")

    assert out["data_success"] is False and out["news_list"] == []
    assert "个股新闻源失败" in out["error"] and "个股公告源失败" in out["error"]
    assert "未检索到" in out["note"]


def test_having_data_suppresses_error_noise(monkeypatch):
    """有数据时不该再报源故障——否则 UI 的 errors 会天天飘红，真问题被淹没"""
    _patch_direct(monkeypatch,
                  news={"items": [_item("A")], "note": None, "error": None},
                  notices={"items": [], "note": None, "error": "个股公告源失败: 502"})
    monkeypatch.setattr(news_feed, "smart_latest_news", lambda **k: {"news": [], "note": None})
    out = NewsFlowDataFetcher().get_news_flow_data("600211")
    assert out["data_success"] is True and "error" not in out


def test_direct_source_exception_degrades_instead_of_crashing(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("源炸了")

    monkeypatch.setattr(news_feed, "fetch_stock_news", _boom)
    monkeypatch.setattr(news_feed, "fetch_stock_notices", lambda *a, **k: _empty())
    monkeypatch.setattr(news_feed, "smart_latest_news", lambda **k: {"news": [], "note": None})
    out = NewsFlowDataFetcher().get_news_flow_data("600211")
    assert out["data_success"] is False and "源炸了" in out["error"]


def test_caller_passes_days_and_name_down_to_direct_source(monkeypatch):
    seen = {}

    def _news(code, days=7, name="", **kw):
        seen.update(code=code, days=days, name=name)
        return _empty()

    monkeypatch.setattr(news_feed, "fetch_stock_news", _news)
    monkeypatch.setattr(news_feed, "fetch_stock_notices", lambda *a, **k: _empty())
    monkeypatch.setattr(news_feed, "smart_latest_news", lambda **k: {"news": [], "note": None})
    NewsFlowDataFetcher().get_news_flow_data("600211", days=7, stock_name="西藏药业")

    assert seen == {"code": "600211", "days": 7, "name": "西藏药业"}


# ================================================================
# 9. 死源摘除 + 用例鉴别力
# ================================================================

def test_dead_wencai_source_removed_from_production_chain():
    src = pathlib.Path("core/agents.py").read_text(encoding="utf-8")
    assert "_fetch_news_fallback" not in src, "问财降级源（23.2s 恒失败）已摘除，不应复活"
    assert "news_announcement" not in src


def test_format_news_block_no_longer_accepts_dead_fallback_param():
    from core.agents import format_news_block
    params = list(inspect.signature(format_news_block).parameters)
    assert params == ["items"], f"fallback_text 参数随死源一起下线，实际 {params}"


def test_leaf_error_marker_reaches_the_caller(monkeypatch):
    """反向验证用例鉴别力：把叶子换成抛固定标记的错误，断言标记真的抵达。

    若只断言"结果为空"，那么**叶子从未被调用**也会通过——那正是
    `test_news_stock_source.py` 里"patch 整个类导致方法名写错也全绿"的同款陷阱。
    """
    def _boom(symbol):
        raise RuntimeError("CONTRACT-MARKER")

    monkeypatch.setattr(news_feed.ak, "stock_news_em", _boom)
    monkeypatch.setattr(news_feed, "fetch_stock_notices", lambda *a, **k: _empty())
    monkeypatch.setattr(news_feed, "smart_latest_news", lambda **k: {"news": [], "note": None})
    out = NewsFlowDataFetcher().get_news_flow_data("600211")

    assert "CONTRACT-MARKER" in (out.get("error") or ""), \
        "叶子错误没抵达调用方 → 用例没真的走到取数路径"
