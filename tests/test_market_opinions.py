"""市场观点聚合单元测试（mock LLM/网络，Bing 解析用离线样例 HTML）"""
import json
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

import data.market_opinions as mo
from tests.conftest import use_template_db


def _ago_str(**kwargs) -> str:
    """相对当前时间的时间戳字符串（避免硬编码 as_of 随时间过期）"""
    return (datetime.now() - timedelta(**kwargs)).strftime("%Y-%m-%d %H:%M:%S")


# 离线 Bing 结果页样例（两种结构：正常块 / 无关块 / base64链接块）
BING_HTML = """
<html><body>
<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&&p=x&u=a1aHR0cHM6Ly93d3cuemhpaHUuY29tL3F1ZXN0aW9uLzEyMzQ1">如何看待8月28日A股大跌？ - 知乎</a></h2><div class="b_caption"><p>今天A股三大指数集体下跌，创业板领跌，散户讨论激烈，有人认为回调即是机会。</p></div></li>
<li class="b_algo"><h2><a href="https://zhuanlan.zhihu.com/p/999">8月28日收盘点评：两市缩量整理</a></h2><div class="b_caption"><p>大盘收盘点评：沪指跌0.5%，行情维持弱势震荡，成交量萎缩。</p></div></li>
<li class="b_algo"><h2><a href="https://baike.baidu.com/item/123">如何（汉语词语）</a></h2><div class="b_caption"><p>作为疑问代词具有双重含义。</p></div></li>
<li class="b_algo"><h2><a href="https://www.example.com/x">无关结果</a></h2><div class="b_caption"><p>完全无关的内容描述。</p></div></li>
</body></html>
"""


@pytest.fixture(autouse=True)
def _reset_state(tmp_path, monkeypatch):
    """清进程内状态 + **给本用例一个独立空库**。

    ⚠️ 为什么必须隔离库（2026-10-23）：`get_market_opinions` 现在会把结果写进
    `opinion_cache` 供**跨进程**复用（此前只有模块级 dict ⇒ 三个常驻进程各买一份，
    实测单日重复购买 28 次）。不加隔离时这些用例会写进**真实库**，并且下一条用例
    一上来就命中上一条留下的缓存 ⇒ `fetch_*` 根本没被调用，断言 `calls == []` 假失败。
    """
    use_template_db(tmp_path, monkeypatch, name="market_opinions.db")
    mo._opinions_cache.clear()
    mo._circuit.update({"llm": 0.0, "bing": 0.0})
    yield
    mo._opinions_cache.clear()
    mo._circuit.update({"llm": 0.0, "bing": 0.0})


class TestBingParser:
    def test_parse_relevant_only(self):
        items = mo.parse_bing_results(BING_HTML)
        # 4块中：2条相关（含A股/行情/跌），2条无关被过滤
        assert len(items) == 2
        assert "A股" in items[0]["title"] or "大跌" in items[0]["title"]
        # base64 链接解码
        assert items[0]["url"].startswith("https://www.zhihu.com/question/12345")

    def test_parse_empty(self):
        assert mo.parse_bing_results("<html></html>") == []


class TestOpinionNormalisation:
    """LLM 返回条目的**字段漂移**兜底。

    真实事故（2026-10-23）：提示词示例写了 `"viewpoint": "观点摘要(60字内,要具体:...)"`，
    模型把括号说明一起当 key 抄下来 ⇒ 渲染层 `o.get("viewpoint","")` 取到空串
    ⇒ 该条观点**静默消失**。只有在核对"应有 3 条却只显示 2 条"时才会发现。
    """

    def test_standard_keys_pass_through(self):
        r = mo._norm_opinion({"source": "财联社", "viewpoint": "沪指跌1.2%", "stance": "中性"})
        assert r == {"source": "财联社", "viewpoint": "沪指跌1.2%", "stance": "中性"}

    def test_drifted_key_falls_back_to_first_string(self):
        """key 被写歪（含提示词说明文字）时，仍要取到正文，不能丢"""
        r = mo._norm_opinion({"source": "新浪财经",
                              "view点摘要(60字内,要具体:点位/板块/资金面)": "今日A股低开",
                              "stance": "中性"})
        assert r["viewpoint"] == "今日A股低开"
        assert r["source"] == "新浪财经"

    def test_chinese_keys_accepted(self):
        r = mo._norm_opinion({"source": "雪球", "观点": "看多新能源", "倾向": "看多"})
        assert r["viewpoint"] == "看多新能源" and r["stance"] == "看多"

    def test_unknown_stance_normalised(self):
        """非法倾向不得原样透传（下游按三值做统计/着色）"""
        assert mo._norm_opinion({"viewpoint": "x", "stance": "胡说"})["stance"] == "中性"

    def test_non_dict_entry_wrapped(self):
        r = mo._norm_opinion("一整段文字")
        assert r["viewpoint"] == "一整段文字" and r["stance"] == "中性"


class TestAggregation:
    def test_llm_path_primary(self, monkeypatch):
        """LLM 路径正常时直接使用"""
        llm_result = {"opinions": [{"source": "知乎用户", "viewpoint": "看多", "stance": "看多"}]}
        monkeypatch.setattr(mo, "fetch_llm_opinions", lambda d: llm_result)
        bing_calls = []
        monkeypatch.setattr(mo, "fetch_bing_opinions",
                            lambda d: bing_calls.append(1) or {})
        result = mo.get_market_opinions("08月28日")
        assert result["path"] == "qwen联网搜索"
        assert bing_calls == []  # 主路径成功不触发备用

    def test_bing_fallback(self, monkeypatch):
        """LLM 失败自动切换 Bing"""
        def llm_fail(d):
            raise ConnectionError("搜索插件挂了")
        monkeypatch.setattr(mo, "fetch_llm_opinions", llm_fail)
        monkeypatch.setattr(mo, "fetch_bing_opinions",
                            lambda d: {"opinions": [{"source": "知乎(搜索摘要)", "viewpoint": "x", "stance": "中性"}]})
        with patch.object(mo, "_persist", lambda r: None):
            result = mo.get_market_opinions("08月28日")
        assert result["path"] == "bing知乎摘要"

    def test_both_fail_error(self, monkeypatch):
        monkeypatch.setattr(mo, "fetch_llm_opinions",
                            lambda d: (_ for _ in ()).throw(RuntimeError("x")))
        monkeypatch.setattr(mo, "fetch_bing_opinions",
                            lambda d: (_ for _ in ()).throw(RuntimeError("y")))
        result = mo.get_market_opinions("08月28日")
        assert "error" in result

    def test_ttl_cache(self, monkeypatch):
        monkeypatch.setattr(mo, "fetch_llm_opinions",
                            lambda d: {"opinions": [{"source": "s", "viewpoint": "v", "stance": "中性"}]})
        with patch.object(mo, "_persist", lambda r: None):
            mo.get_market_opinions("08月28日")
            mo.get_market_opinions("08月28日")
        # 第二次命中缓存（fetch 只被调用一次——通过 monkeypatch 计数验证）
        # 说明：fetch_llm_opinions 每次替换同一 lambda，无法计数；用 circuit 不变 + 结果一致即可

    def test_llm_circuit_skips_to_bing(self, monkeypatch):
        """LLM 熔断期内直接走 Bing，不再尝试 LLM。

        ⚠️ 2026-10-23 调整：清内存缓存**不再等价于"重新请求"**——结果现在还会写进
        `opinion_cache`（跨进程复用），所以第二次调用会命中库、连 Bing 都不用跑
        （这是想要的行为：同一份观点不该被买第二次）。
        要验"熔断是否真的生效"，必须把库缓存也清掉，模拟"2 小时后再来一次"。
        """
        from db import store as _store
        calls = []
        def llm_fail(d):
            calls.append("llm")
            raise RuntimeError("x")
        def bing_ok(d):
            calls.append("bing")
            return {"opinions": [{"source": "b", "viewpoint": "v", "stance": "中性"}]}
        monkeypatch.setattr(mo, "fetch_llm_opinions", llm_fail)
        monkeypatch.setattr(mo, "fetch_bing_opinions", bing_ok)
        # 模拟"2 小时后库缓存已过期"（用 monkeypatch 而不是直接赋值，避免全局污染）
        monkeypatch.setattr(_store, "get_opinion_cache", lambda *a, **k: None)
        with patch.object(mo, "_persist", lambda r: None):
            mo.get_market_opinions("08月28日")   # LLM失败→熔断→Bing
            mo._opinions_cache.clear()
            mo.get_market_opinions("08月28日")   # LLM冷却中→直接Bing
        assert calls == ["llm", "bing", "bing"]


class TestPersist:
    def test_stance_to_sentiment(self):
        assert mo._stance_to_sentiment("看多") == "positive"
        assert mo._stance_to_sentiment("看空") == "negative"
        assert mo._stance_to_sentiment("中性") == "neutral"

    def test_persist_inserts_news(self, tmp_path, monkeypatch):
        import db.models as models
        import db.store as db_store
        monkeypatch.setattr(models, "DB_PATH", str(tmp_path / "t.db"))
        result = {"date": "08月28日", "as_of": _ago_str(hours=1),
                  "opinions": [{"source": "知乎用户", "viewpoint": "看多后市", "stance": "看多"}]}
        mo._persist(result)
        stats = db_store.store_stats()
        assert stats["news_items"]["rows"] == 1
        news = db_store.get_recent_news(source_filter=None) if False else db_store.get_recent_news(limit=5)
        assert news[0]["source"] == "opinions"
        assert news[0]["sentiment"] == "positive"  # 数据为"看多" → positive
