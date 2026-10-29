# -*- coding: utf-8 -*-
"""概念成分映射（`data/concept_members.py`）的守护测试。

为什么单独一个文件：这份映射是**事实层**（谁属于"铜"概念），一旦错了，
下游"铜价涨 → 推荐黄金股"这种错误没人看得出来。所以异常路径要和正常路径
一样被钉死：分页、去重、非法代码、离线、缓存命中、缓存陈旧、部分页失败。

纪律：**不打真实网络**。全部用假 `requests.get` 注入 + `tmp_path` 缓存，
测试不耦合真实环境（换台机器/断网也必须全绿）。
"""
import json
from datetime import datetime, timedelta

import pytest

from data import concept_members as cm


# ================================================================
# 假响应
# ================================================================

class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def _page(rows, pages, count=None):
    return _FakeResp({"result": {"data": rows, "pages": pages,
                                 "count": count if count is not None else len(rows)}})


def _row(code, board, board_code="BK0001"):
    return {"SECURITY_CODE": code, "BOARD_NAME": board,
            "NEW_BOARD_CODE": board_code}


@pytest.fixture(autouse=True)
def _no_offline(monkeypatch):
    """默认按在线跑；要测离线的用例自己再 patch 回来。"""
    from data import offline_gate
    monkeypatch.setattr(offline_gate, "is_offline", lambda: False)


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    p = tmp_path / "concept_members.json"
    monkeypatch.setattr(cm, "CACHE_PATH", str(p))
    return p


def _write_cache(path, concepts, hours_ago=0.0):
    ts = datetime.now() - timedelta(hours=hours_ago)
    path.write_text(json.dumps({"concepts": concepts, "failures": [],
                                "fetched_at": ts.strftime("%Y-%m-%d %H:%M:%S")},
                               ensure_ascii=False), encoding="utf-8")


# ================================================================
# 代码口径
# ================================================================

def test_code_filter_matches_scheduler():
    """`_is_a_share` 与 `core.scheduler.is_a_share_code` 必须同口径。

    两处实现是**故意**的重复（`data` 层不能反向 import `core.scheduler`，
    否则成环）。重复的代价就是漂移风险，所以用测试钉住，而不是靠注释。
    """
    from core.scheduler import is_a_share_code

    samples = ["600000", "601899", "603899", "605100", "688981", "689009",
               "000001", "001979", "002415", "003816", "300750", "301236",
               "830799", "430047", "920001", "900901", "200011",
               "12345", "60000A", "", "6000000"]
    diff = [(c, cm._is_a_share(c), is_a_share_code(c)) for c in samples
            if cm._is_a_share(c) != is_a_share_code(c)]
    assert not diff, f"口径漂移: {diff}"


# ================================================================
# 抓取
# ================================================================

def test_fetch_paginates_dedupes_and_filters(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(params["pageNumber"])
        if params["pageNumber"] == 1:
            return _page([_row("601899", "铜"), _row("600362", "铜"),
                          _row("900901", "铜"),      # B 股 → 剔
                          _row("12345", "铜"),       # 非 6 位 → 剔
                          _row("601899", "铜")],     # 重复 → 去重
                         2, count=6)
        return _page([_row("000630", "铜")], 2, count=6)

    monkeypatch.setattr("requests.get", fake_get)
    out = cm._fetch_all()

    assert calls == [1, 2]
    assert out["concepts"]["铜"]["codes"] == ["000630", "600362", "601899"]
    assert out["concepts"]["铜"]["board_code"] == "BK0001"
    assert out["failures"] == []


def test_fetch_keeps_board_code_and_groups_by_name(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **kw: _page(
        [_row("601899", "铜", "BK1015"), _row("000630", "黄金", "BK0547")], 1))
    out = cm._fetch_all()
    assert set(out["concepts"]) == {"铜", "黄金"}
    assert out["concepts"]["黄金"]["board_code"] == "BK0547"


def test_fetch_raises_on_empty_first_page(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **kw: _page([], 1, count=0))
    with pytest.raises(cm.ConceptMembersUnavailable):
        cm._fetch_all()


def test_fetch_offline_raises_without_network(monkeypatch):
    """离线闸门：`BIGA_OFFLINE=1` 时**一次外呼都不许发**。"""
    from data import offline_gate
    monkeypatch.setattr(offline_gate, "is_offline", lambda: True)

    def boom(*a, **kw):
        raise AssertionError("离线模式不得发起网络请求")

    monkeypatch.setattr("requests.get", boom)
    with pytest.raises(cm.ConceptMembersUnavailable):
        cm._fetch_all()


def test_partial_page_failure_is_recorded_not_silent(monkeypatch):
    """第 2 页失败 → 结果里必须有 failures，且上层能置 degraded。"""
    def fake_get(url, params=None, headers=None, timeout=None):
        if params["pageNumber"] == 1:
            return _page([_row("601899", "铜")], 2)
        raise OSError("connection reset")

    monkeypatch.setattr("requests.get", fake_get)
    out = cm._fetch_all()
    assert out["concepts"]["铜"]["codes"] == ["601899"]
    assert out["failures"] and "page2" in out["failures"][0]


# ================================================================
# 缓存与降级
# ================================================================

def test_cache_hit_skips_network(cache_path, monkeypatch):
    _write_cache(cache_path, {"铜": {"board_code": "BK1015", "codes": ["601899"]}})

    def boom(*a, **kw):
        raise AssertionError("缓存新鲜时不得联网")

    monkeypatch.setattr("requests.get", boom)
    out = cm.get_concept_members()
    assert out["concepts"]["铜"]["codes"] == ["601899"]
    assert out["degraded"] is False
    assert out["age_hours"] < 1


def test_stale_cache_degrades_when_fetch_fails(cache_path, monkeypatch):
    """宁可旧，不可无 —— 但必须自报是旧数据。"""
    _write_cache(cache_path, {"铜": {"board_code": "BK1015", "codes": ["601899"]}},
                 hours_ago=cm.CACHE_TTL_HOURS + 10)

    def boom(*a, **kw):
        raise OSError("network down")

    monkeypatch.setattr("requests.get", boom)
    out = cm.get_concept_members()
    assert out["degraded"] is True
    assert "OSError" in out["fetch_error"]
    assert out["concepts"]["铜"]["codes"] == ["601899"]


def test_no_cache_and_failure_raises(cache_path, monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **kw: (_ for _ in ()).throw(
        OSError("no route")))
    with pytest.raises(cm.ConceptMembersUnavailable):
        cm.get_concept_members()


def test_force_ignores_fresh_cache(cache_path, monkeypatch):
    _write_cache(cache_path, {"旧": {"board_code": "", "codes": ["600000"]}})
    monkeypatch.setattr("requests.get",
                        lambda *a, **kw: _page([_row("601899", "铜")], 1))
    out = cm.get_concept_members(force=True)
    assert "旧" not in out["concepts"]
    assert out["concepts"]["铜"]["codes"] == ["601899"]


def test_corrupt_cache_is_ignored_and_rebuilt(cache_path, monkeypatch):
    cache_path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr("requests.get",
                        lambda *a, **kw: _page([_row("601899", "铜")], 1))
    out = cm.get_concept_members()
    assert out["concepts"]["铜"]["codes"] == ["601899"]
    assert json.loads(cache_path.read_text(encoding="utf-8"))["concepts"]


# ================================================================
# 检索（只给核对用，不参与选股）
# ================================================================

def test_search_concepts_empty_keyword_returns_nothing():
    assert cm.search_concepts("", {"铜": {}}) == []
    assert cm.search_concepts(None, {"铜": {}}) == []


def test_search_concepts_substring_and_sorted():
    concepts = {"铜": {}, "铜缆高速连接": {}, "黄金": {}}
    assert cm.search_concepts("铜", concepts) == ["铜", "铜缆高速连接"]


# ================================================================
# 路径（frozen / exe 兼容）
# ================================================================

def test_cache_paths_derive_from_data_root():
    """缓存路径必须由 `data_root()` 派生，**不能**是 `dirname(__file__)`。

    exe（frozen）下 `__file__` 落在只读的 `_internal/`，而缓存实际写在
    `data_root()/data/`。用 `__file__` ⇒ exe 里读不到缓存 ⇒ 推荐链路退化成
    "行业成分两个来源都不可用"（2026-10-29 exe 冒烟实测：`/api/intel/picks`
    返回 `missing`，而源码环境一切正常）。

    这条是**结构性**守护：只看路径从哪来，不依赖任何运行时数据。
    """
    import os

    from config.paths import data_root
    from data import concept_members, industry_members

    root = data_root()
    for mod, name in ((concept_members, "concept_members.json"),
                      (industry_members, "industry_members.json")):
        expected = os.path.normcase(os.path.join(root, "data", name))
        actual = os.path.normcase(os.path.abspath(mod.CACHE_PATH))
        assert actual == expected, (
            f"{mod.__name__}.CACHE_PATH 未走 data_root：{actual} != {expected}")
