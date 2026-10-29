# -*- coding: utf-8 -*-
"""知乎话题雷达（`data/zhihu_opinions.py`）单元测试

⚠️ 全部**离线**：一律替换 `_session()`，绝不真打知乎（额度宝贵，且测试须确定性）。
⚠️ 配置文件与额度文件一律重定向到 `tmp_path`，**不碰真实 `config/` 与 `temp/`**。

本文件守护的重点是三条**实测得来**的约束：
1. 鉴权三要素（Bearer + **秒级**时间戳 + JSON）—— 毫秒级会被统一拒为 20001；
2. **额度护栏**：超限必须**不发起请求**（而不是发出去等报错）；
3. **隔离**：`/user/contents` 读不了他人 ⇒ 本模块只做"话题检索"，不做"按人采集"。
"""

import json
import time

import pytest

from data import zhihu_opinions as Z


# ---------------------------------------------------------------- 夹具
@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """把配置文件（话题 / 博主 / 额度）全部重定向到 tmp，并给一个可用的 token。"""
    monkeypatch.setattr(Z, "TOPICS_PATH", str(tmp_path / "topics.json"))
    monkeypatch.setattr(Z, "MEMBERS_PATH", str(tmp_path / "members.json"))
    monkeypatch.setattr(Z, "_QUOTA_PATH", str(tmp_path / "quota.json"))
    Z._topic_cache.update({"mtime": None, "topics": None})
    Z._member_cache.update({"mtime": None, "members": None})
    monkeypatch.setenv("ZHIHU_API_TOKEN", "f" * 40)
    return tmp_path


class _Resp:
    def __init__(self, payload, status=200, raw=None):
        self._p = payload
        self.status_code = status
        self.text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        # ⚠️ 生产代码（`_web_get_json`）靠 `content` 的前几个字节判断
        # "是 JSON / gzip 没解压 / 被加密替换"，夹具必须提供同名字段，
        # 否则 getattr 拿到 b"" ⇒ 被判成"空响应体" ⇒ 每次都要重试 + sleep。
        self._raw = raw
        self.content = raw if raw is not None else self.text.encode("utf-8")

    def json(self):
        return self._p


class _FakeSession:
    """记录所有请求，便于断言"该发的发了、不该发的没发"。"""

    def __init__(self, payload=None):
        self.calls = []
        self.payload = payload or {"Code": 0, "Message": "success",
                                   "Data": {"Items": []}}

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": dict(headers or {}),
                           "params": dict(params or {})})
        return _Resp(self.payload)


class _ScriptedSession:
    """按**调用次序**返回不同响应 —— 用于验证"乱码 → 重试 → 成功"。

    `responses` 每项可以是 `_Resp` 或 `Exception`。
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": dict(headers or {}),
                           "params": dict(params or {})})
        item = self.responses.pop(0) if self.responses else _Resp({"data": []})
        if isinstance(item, Exception):
            raise item
        return item


def _patch_session(monkeypatch, payload=None):
    s = _FakeSession(payload)
    monkeypatch.setattr(Z, "_session", lambda: s)
    return s


# ================================================================
# 一、鉴权三要素（实测踩坑：毫秒级会被统一拒）
# ================================================================

class TestHeaders:
    def test_has_bearer_and_json(self, isolated):
        h = Z._headers()
        assert h["Authorization"].startswith("Bearer ")
        assert h["Content-Type"] == "application/json"
        assert h["User-Agent"]

    def test_timestamp_is_seconds_not_milliseconds(self, isolated):
        """⚠️ 实测：毫秒级时间戳会被服务端**统一拒为 20001**。

        断言"取值接近当前秒级时间戳、且明显不是毫秒级"——
        毫秒级会是 13 位（×1000），秒级是 10 位。
        """
        ts = Z._headers()["X-Request-Timestamp"]
        assert ts.isdigit()
        assert len(ts) == 10, f"应为秒级（10 位），得到 {len(ts)} 位：{ts}"
        assert abs(int(ts) - int(time.time())) <= 5

    def test_token_stripped(self, isolated, monkeypatch):
        monkeypatch.setenv("ZHIHU_API_TOKEN", "  " + "a" * 40 + "  ")
        assert Z.get_token() == "a" * 40          # 前后空白必须被去掉

    def test_auth_ok_without_token(self, monkeypatch):
        monkeypatch.delenv("ZHIHU_API_TOKEN", raising=False)
        ok, msg = Z.auth_ok()
        assert ok is False and "ZHIHU_API_TOKEN" in msg

    def test_auth_ok_hint_mentions_length(self, isolated, monkeypatch):
        """⚠️ 鉴权失败提示必须带上**长度**线索 —— 实测最坑的就是"粘贴多出字符""。
        """
        _patch_session(monkeypatch, {"Code": 20001, "Message": "Authorization failed",
                                     "Data": None})
        ok, msg = Z.auth_ok()
        assert ok is False
        assert "40" in msg and "20001" in msg

    def test_auth_ok_success(self, isolated, monkeypatch):
        _patch_session(monkeypatch, {"Code": 0, "Message": "success", "Data": {}})
        ok, _ = Z.auth_ok()
        assert ok is True


# ================================================================
# 二、话题配置：可增删 + mtime 缓存
# ================================================================

class TestTopics:
    def test_creates_file_with_defaults_when_missing(self, isolated):
        assert not (isolated / "topics.json").exists()
        assert Z.load_topics() == list(Z.DEFAULT_TOPICS)
        assert (isolated / "topics.json").exists(), "首次读取应把默认值落盘（让「可增删」对用户可见）"

    def test_save_and_reload_roundtrip(self, isolated):
        assert Z.save_topics(["A股", " 大盘 ", "", "A股"])
        got = Z.load_topics()
        # 去空白、去空行；**不去重**（去重是采集层的事，配置层保留用户原样）
        assert got == ["A股", "大盘", "A股"]

    def test_mtime_cache_invalidated_after_save(self, isolated):
        """改完即生效，不用重启 —— 这是"方便增删"的实现基础。"""
        Z.load_topics()
        Z.save_topics(["新话题"])
        assert Z.load_topics() == ["新话题"], "保存后必须重新加载，不能吃旧缓存"

    def test_external_edit_is_picked_up(self, isolated):
        """用户直接改文件（不经 save_topics）也要生效"""
        Z.load_topics()
        p = isolated / "topics.json"
        p.write_text(json.dumps({"topics": ["外部改的"]}, ensure_ascii=False),
                     encoding="utf-8")
        import os
        os.utime(p, (time.time() + 2, time.time() + 2))     # 保证 mtime 变化
        Z._topic_cache["mtime"] = None
        assert Z.load_topics() == ["外部改的"]

    def test_corrupt_file_falls_back_to_defaults(self, isolated):
        p = isolated / "topics.json"
        p.write_text("{ 这不是 JSON", encoding="utf-8")
        assert Z.load_topics() == list(Z.DEFAULT_TOPICS)

    def test_empty_topics_falls_back_to_defaults(self, isolated):
        Z.save_topics([])
        assert Z.load_topics() == list(Z.DEFAULT_TOPICS)


# ================================================================
# 三、额度护栏：**超限必须不发起请求**
# ================================================================

class TestQuotaGuard:
    def test_starts_at_zero(self, isolated):
        st = Z.quota_status()
        assert st["used"] == 0 and st["limit"] == Z.DAILY_SEARCH_LIMIT

    def test_bump_increments_and_persists(self, isolated):
        Z._bump_quota()
        Z._bump_quota(2)
        assert Z.quota_status()["used"] == 3
        assert "3/60" in Z.quota_note()

    def test_day_rollover_resets(self, isolated):
        Z._bump_quota(5)
        p = isolated / "quota.json"
        p.write_text(json.dumps({"date": "2000-01-01", "used": 999}), encoding="utf-8")
        assert Z.quota_status()["used"] == 0

    def test_corrupt_quota_file_is_treated_as_zero(self, isolated):
        (isolated / "quota.json").write_text("坏文件", encoding="utf-8")
        assert Z.quota_status()["used"] == 0

    def test_exhausted_quota_does_not_call_api(self, isolated, monkeypatch):
        """⚠️ 核心守护：额度用尽时**一次请求都不能发**（省额度，且行为确定）。"""
        (isolated / "quota.json").write_text(
            json.dumps({"date": time.strftime("%Y-%m-%d"),
                        "used": Z.DAILY_SEARCH_LIMIT}), encoding="utf-8")
        s = _patch_session(monkeypatch)
        assert Z.search("A股") == []
        assert s.calls == [], "额度用尽却仍发起了请求"

    def test_successful_search_counts_once(self, isolated, monkeypatch):
        s = _patch_session(monkeypatch)
        Z.search("A股")
        assert len(s.calls) == 1
        assert Z.quota_status()["used"] == 1

    def test_failed_request_does_not_consume_quota(self, isolated, monkeypatch):
        """网络异常时不该计费（否则额度会被"失败"吃掉）"""
        def boom():
            raise RuntimeError("network down")
        monkeypatch.setattr(Z, "_session", boom)
        assert Z.search("A股") == []
        assert Z.quota_status()["used"] == 0


# ================================================================
# 四、检索与解析
# ================================================================

_ITEM = {
    "Title": "2026.10.13 A股市场的关键翻红 - 知乎",
    "ContentType": "Article",
    "ContentText": "今天A股市场单看收盘结果的话，四大市场指数都是微涨微跌状态。",
    "Url": "https://www.zhihu.com/question/123?utm_medium=openapi_platform",
    "CommentCount": 352, "VoteUpCount": 5117,
    "AuthorName": "刘鹏程Sai.L", "EditTime": 1791899645,
}


class TestSearch:
    def test_parses_all_needed_fields(self, isolated, monkeypatch):
        _patch_session(monkeypatch, {"Code": 0, "Data": {"Items": [_ITEM]}})
        out = Z.search("A股", limit=3)
        assert len(out) == 1
        it = out[0]
        assert it["title"].startswith("2026.10.13")
        assert it["author"] == "刘鹏程Sai.L"
        assert it["votes"] == 5117 and it["comments"] == 352
        assert it["text"].startswith("今天A股市场")
        assert it["topic"] == "A股"

    def test_url_strips_tracking_query(self, isolated, monkeypatch):
        """URL 必须去掉 utm 追踪参数（保持可读、可复制）"""
        _patch_session(monkeypatch, {"Code": 0, "Data": {"Items": [_ITEM]}})
        assert Z.search("A股")[0]["url"] == "https://www.zhihu.com/question/123"

    def test_sends_query_and_limit(self, isolated, monkeypatch):
        s = _patch_session(monkeypatch)
        Z.search("股市行情", limit=7)
        assert s.calls[0]["params"]["Query"] == "股市行情"
        assert s.calls[0]["params"]["Limit"] == 7

    def test_limit_is_clamped(self, isolated, monkeypatch):
        s = _patch_session(monkeypatch)
        Z.search("A股", limit=999)
        assert s.calls[0]["params"]["Limit"] == 50, "过大的 limit 必须被夹紧"

    def test_empty_keyword_short_circuits(self, isolated, monkeypatch):
        s = _patch_session(monkeypatch)
        assert Z.search("   ") == []
        assert s.calls == []

    def test_no_token_short_circuits(self, isolated, monkeypatch):
        monkeypatch.delenv("ZHIHU_API_TOKEN", raising=False)
        s = _patch_session(monkeypatch)
        assert Z.search("A股") == []
        assert s.calls == []

    def test_business_error_returns_empty(self, isolated, monkeypatch):
        """业务码非 0（如 20001）→ 返回空，不抛异常、不编造"""
        _patch_session(monkeypatch, {"Code": 20001, "Message": "Authorization failed"})
        assert Z.search("A股") == []

    def test_malformed_items_are_skipped(self, isolated, monkeypatch):
        _patch_session(monkeypatch, {"Code": 0, "Data": {"Items": [
            _ITEM, "不是字典", None, {"Title": "只有标题"}]}})
        out = Z.search("A股")
        assert len(out) == 2, "非字典条目必须被跳过而不是崩掉"


class TestCollect:
    def test_dedupes_across_topics(self, isolated, monkeypatch):
        """跨话题去重：同一 URL 出现在多个话题里只保留一条"""
        _patch_session(monkeypatch, {"Code": 0, "Data": {"Items": [_ITEM]}})
        items, meta = Z.collect(["A股", "股市行情"], per_topic=5, sleep=0)
        assert len(items) == 1
        assert meta["asked"] == 2 and meta["got"] == 1

    def test_meta_reports_quota_and_errors(self, isolated, monkeypatch):
        _patch_session(monkeypatch)
        items, meta = Z.collect(["A股"], per_topic=5, sleep=0)
        assert items == []
        assert meta["quota"]["used"] == 1
        assert meta["errors"], "取不到时必须给出原因，不能静默为空"

    def test_reports_error_when_quota_runs_out_midway(self, isolated, monkeypatch):
        """中途额度用尽 → 剩余话题记入 errors，并如实声明"""
        (isolated / "quota.json").write_text(
            json.dumps({"date": time.strftime("%Y-%m-%d"),
                        "used": Z.DAILY_SEARCH_LIMIT - 1}), encoding="utf-8")
        s = _patch_session(monkeypatch, {"Code": 0, "Data": {"Items": [_ITEM]}})
        items, meta = Z.collect(["A股", "股市行情", "大盘走势"], per_topic=5, sleep=0)
        assert any("额度不足" in e for e in meta["errors"]), meta["errors"]
        assert len(s.calls) <= 1, "额度只够 1 次时不该发出更多请求"


# ================================================================
# 失败原因必须可查（2026-10-15 用户报"知乎仍未返回结果"）
# ================================================================
# 背景：原实现把 HTTP 异常/业务码统一 `return []` 并只写日志 ⇒ 页面只能显示
# "可能额度用尽或关键词过窄"这句**没有鉴别力的猜测**，用户无法判断到底是没配 token、
# 鉴权失败、网络不通还是真没结果。⇒ 每次失败都要留下"哪一步 / 什么原因"。

class TestFailureSurfacing:
    def test_missing_token_names_the_step(self, isolated, monkeypatch):
        monkeypatch.delenv("ZHIHU_API_TOKEN", raising=False)
        assert Z.search("A股") == []
        e = Z.last_error()
        assert e["step"] == "凭证" and "ZHIHU_API_TOKEN" in e["reason"]

    def test_business_code_is_reported_with_hint(self, isolated, monkeypatch):
        _patch_session(monkeypatch, {"Code": 20001, "Message": "Authorization failed"})
        assert Z.search("A股") == []
        e = Z.last_error()
        assert e["step"] == "业务码" and "20001" in e["reason"]
        assert "40" in e["detail"], "必须提示'secret 应为 40 位'这条实测坑"

    def test_network_error_is_reported_with_proxy_hint(self, isolated, monkeypatch):
        class _Boom:
            def get(self, *a, **k):
                raise RuntimeError("proxy blocked")
        monkeypatch.setattr(Z, "_session", lambda: _Boom())
        assert Z.search("A股") == []
        e = Z.last_error()
        assert e["step"] == "网络" and "RuntimeError" in e["reason"]
        assert "BIGA_ZHIHU_USE_PROXY" in e["detail"], "必须给出逃生门"

    def test_empty_items_is_distinguished_from_failure(self, isolated, monkeypatch):
        """`Code=0` 但无内容 ≠ 失败 —— 两者原因不同，不能混成一句话。"""
        _patch_session(monkeypatch, {"Code": 0, "Data": {"Items": []}})
        assert Z.search("冷门词") == []
        assert Z.last_error()["step"] == "空结果"

    def test_collect_reports_per_topic_and_propagates_reason(self, isolated, monkeypatch):
        """逐话题结果必须可查（哪个话题 0 条、为什么）。"""
        s = _patch_session(monkeypatch, {"Code": 0, "Data": {"Items": [_ITEM]}})
        items, meta = Z.collect(["A股", "股市行情"], per_topic=5, sleep=0)
        assert len(items) == 1
        assert len(meta["per_topic"]) == 2
        assert all("topic" in r and "got" in r for r in meta["per_topic"])
        assert meta["per_topic"][0]["got"] == 1

    def test_collect_all_failures_carry_reason_in_message(self, isolated, monkeypatch):
        _patch_session(monkeypatch, {"Code": 20001, "Message": "Authorization failed"})
        items, meta = Z.collect(["A股"], per_topic=5, sleep=0)
        assert items == []
        assert any("业务码" in e for e in meta["errors"]), meta["errors"]
        assert meta["last_error"]["reason"]


# ================================================================
# 五、按博主取「想法」（网页端 v4）—— 2026-10-23 新增
# ================================================================
# 背景：用户报"知乎不能返回财经博主的帖子"（people/shou-wo-rou-ti）。
# 实测结论写死在模块里：官方 v1 没有任何按用户查的端点（10 个候选全 404），
# 只能走网页端 `/api/v4/members/<url_token>/pins`；且**仅 pins 免登录**。
# 本组测试守护三件事：
#   1. token 归一化（用户一定会粘整条 URL）；
#   2. v4 通道**绝不能**带上开放平台 token（带了更易被风控，也说明通道被混用）；
#   3. 失败原因必须可查（404/403/非 JSON/空结果 四者不可混成一句话）。

_PIN = {
    "id": 2096981397479204169,
    "created": 1792739340,
    "content": [{"content": "这次还是我赢了。", "own_text": "这次还是我赢了。",
                 "type": "text", "fold_type": "raw"}],
    "like_count": 55, "comment_count": 51,
    "url": "/pins/2096981397479204169",
    "excerpt_title": "",
    "author": {"name": "手握柔荑", "url_token": "shou-wo-rou-ti"},
}


class _StatusSession:
    """可控状态码的假会话（用于 404 / 403 分支）。"""

    def __init__(self, payload, status=200, text=None):
        self.payload, self.status, self.text = payload, status, text
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": dict(headers or {}),
                           "params": dict(params or {})})
        return _Resp(self.payload, status=self.status)


class TestMemberToken:
    def test_accepts_full_profile_url(self, isolated):
        """用户粘的一定是整条网页链接 —— 必须能直接用。"""
        assert Z._norm_member_token("https://www.zhihu.com/people/shou-wo-rou-ti") == "shou-wo-rou-ti"

    def test_accepts_relative_and_bare(self, isolated):
        assert Z._norm_member_token("/people/shou-wo-rou-ti/") == "shou-wo-rou-ti"
        assert Z._norm_member_token("shou-wo-rou-ti") == "shou-wo-rou-ti"

    def test_strips_subpath_so_it_does_not_404(self, isolated):
        """⚠️ `people/xxx/answers` 若整段当 token，请求必 404。"""
        assert Z._norm_member_token("https://www.zhihu.com/people/abc/answers") == "abc"

    def test_strips_query(self, isolated):
        assert Z._norm_member_token("https://www.zhihu.com/people/abc?utm=1") == "abc"

    def test_empty(self, isolated):
        assert Z._norm_member_token("  ") == ""


class TestMemberConfig:
    def test_creates_file_with_defaults(self, isolated):
        assert not (isolated / "members.json").exists()
        assert Z.load_members() == list(Z.DEFAULT_MEMBERS)
        assert (isolated / "members.json").exists()

    def test_save_normalizes_and_dedupes(self, isolated):
        assert Z.save_members(["https://www.zhihu.com/people/a", "a", " b ", ""])
        assert Z.load_members() == ["a", "b"], "必须归一化并去重，否则同一博主会被采两次"

    def test_corrupt_file_falls_back(self, isolated):
        (isolated / "members.json").write_text("{坏", encoding="utf-8")
        assert Z.load_members() == list(Z.DEFAULT_MEMBERS)


class TestMemberPins:
    def test_parses_and_completes_relative_url(self, isolated, monkeypatch):
        _patch_session(monkeypatch, {"data": [_PIN], "paging": {"is_end": True}})
        out = Z.member_pins("shou-wo-rou-ti", limit=5)
        assert len(out) == 1
        it = out[0]
        assert it["author"] == "手握柔荑"
        assert it["text"] == "这次还是我赢了。"
        assert it["votes"] == 55 and it["comments"] == 51
        assert it["published"] == 1792739340
        assert it["url"] == "https://www.zhihu.com/pins/2096981397479204169", \
            "v4 返回的是 /pins/<id> 相对路径，必须补全，否则页面链接点不开"
        assert it["content_type"] == "pin"

    def test_must_not_send_open_platform_token(self, isolated, monkeypatch):
        """⚠️ 两条通道必须**互不带对方的凭证**：v4 带 Authorization 只会更易被风控。"""
        s = _patch_session(monkeypatch, {"data": [_PIN]})
        Z.member_pins("shou-wo-rou-ti")
        assert "Authorization" not in s.calls[0]["headers"], "v4 通道混用了开放平台 token"
        assert s.calls[0]["headers"]["Referer"].startswith("https://www.zhihu.com")

    def test_limit_clamped_to_49(self, isolated, monkeypatch):
        """实测 limit=50 只回 49 ⇒ 夹到 49，别让调用方以为要到了 50。"""
        s = _patch_session(monkeypatch, {"data": [_PIN]})
        Z.member_pins("x", limit=999)
        assert s.calls[0]["params"]["limit"] == Z.MAX_PINS_PER_MEMBER == 49

    def test_flattens_text_and_image_blocks(self, isolated, monkeypatch):
        pin = dict(_PIN, content=[
            {"type": "text", "content": "原始", "own_text": "去折叠后"},
            {"type": "image", "content": {"url": "https://pic.zhihu.com/a.jpg"}},
        ])
        _patch_session(monkeypatch, {"data": [pin]})
        assert Z.member_pins("x")[0]["text"] == "去折叠后\nhttps://pic.zhihu.com/a.jpg"

    def test_title_falls_back_to_text_head(self, isolated, monkeypatch):
        """想法多数没有标题 ⇒ 用正文首行，**不能显示空标题**。"""
        _patch_session(monkeypatch, {"data": [_PIN]})
        assert Z.member_pins("x")[0]["title"] == "这次还是我赢了。"

    def test_404_is_named_not_silent(self, isolated, monkeypatch):
        monkeypatch.setattr(Z, "_session",
                            lambda: _StatusSession({"error": {}}, status=404, text="not found"))
        assert Z.member_pins("no-such-user") == []
        e = Z.last_error()
        assert e["step"] == "博主不存在" and "no-such-user" in e["reason"]

    def test_403_is_named_as_blocked(self, isolated, monkeypatch):
        """answers/articles 就是 403；必须说清"要登录态、本模块不做"，不能含糊。"""
        monkeypatch.setattr(Z, "_session",
                            lambda: _StatusSession({"error": {"need_login": True}}, status=403))
        assert Z.member_pins("x") == []
        e = Z.last_error()
        assert e["step"] == "被拦截"
        assert "x-zse-96" in e["detail"] or "登录" in e["detail"]

    def test_non_json_is_reported(self, isolated, monkeypatch):
        """被风控时返回的是 HTML 而不是 JSON —— 必须与"没内容"区分开。"""
        class _Html:
            status_code = 200
            text = "<html><meta id='zh-zse-ck'>challenge</html>"

            def json(self):
                raise ValueError("Expecting value: line 1 column 1")

        monkeypatch.setattr(Z, "_session",
                            lambda: type("S", (), {"get": lambda *a, **k: _Html()})())
        assert Z.member_pins("x") == []
        assert Z.last_error()["step"] == "解析"

    def test_business_error_is_reported(self, isolated, monkeypatch):
        _patch_session(monkeypatch, {"error": {"code": 40353, "message": "need login"}})
        assert Z.member_pins("x") == []
        assert "40353" in Z.last_error()["reason"]

    def test_empty_data_is_distinguished(self, isolated, monkeypatch):
        _patch_session(monkeypatch, {"data": [], "paging": {"is_end": True}})
        assert Z.member_pins("x") == []
        assert Z.last_error()["step"] == "空结果"

    def test_malformed_items_skipped(self, isolated, monkeypatch):
        _patch_session(monkeypatch, {"data": [_PIN, "不是字典", None]})
        assert len(Z.member_pins("x")) == 1


class TestCollectMembers:
    def test_dedupes_across_members(self, isolated, monkeypatch):
        _patch_session(monkeypatch, {"data": [_PIN]})
        items, meta = Z.collect_members(["a", "b"], per_member=5, sleep=0)
        assert len(items) == 1, "同一条想法出现在两个博主下，只应保留一条"
        assert meta["asked"] == 2 and meta["got"] == 1

    def test_quota_is_none_not_zero(self, isolated, monkeypatch):
        """⚠️ 本通道不消耗开放平台额度 ⇒ `quota` 必须是 `None`，
        不能沿用 `quota_status()` 的 0/60 —— 那会让页面显示"没扣额度"的同时
        又给出一个"已用 N 次"的假数字，两个通道混成一个口径。
        """
        _patch_session(monkeypatch, {"data": [_PIN]})
        _, meta = Z.collect_members(["a"], sleep=0)
        assert meta["quota"] is None
        assert meta["source"] == "web_v4_pins"

    def test_reports_per_member_and_reason(self, isolated, monkeypatch):
        _patch_session(monkeypatch, {"data": [], "paging": {"is_end": True}})
        items, meta = Z.collect_members(["a", "b"], sleep=0)
        assert items == []
        assert len(meta["by_member"]) == 2
        assert all(r["error"] for r in meta["by_member"]), meta["by_member"]
        assert meta["errors"], "全为 0 条时必须给出原因"

    def test_member_cap(self, isolated, monkeypatch):
        """单次最多采 MAX_MEMBERS_PER_RUN 个 —— 防一次点下去打出几十个请求。"""
        s = _patch_session(monkeypatch, {"data": [_PIN]})
        Z.collect_members([f"m{i}" for i in range(50)], sleep=0)
        assert len(s.calls) == Z.MAX_MEMBERS_PER_RUN


# ================================================================
# 十二、间歇性反爬：响应体自检 + 重试 + Cookie（2026-10-26）
# ================================================================
#
# 事故：15:20 那趟采集返回 HTTP **200**，但响应体是二进制乱码：
#   `[知乎·博主] 「shou-wo-rou-ti」返回非 JSON（HTTP 200）: ��Ȉ��^��&...`
# 而同一天 16:12 / 17:0x 多次直连同一端点**全部正常**
# （3/3 返回 gzip JSON，magic `7b227061` 就是 `{"pa`）。
# 所以这不是"知乎 API 用不了"，是**间歇性反爬**。
# 旧代码 `json()` 一抛错就 `return []` ⇒ 整趟采集判"未取到任何言论"，
# 当天的情绪快照就没了 —— 一次抖动被当成永久失败。

@pytest.fixture
def no_sleep(monkeypatch):
    """重试的退避 sleep 必须短路，否则每条用例都要真等 3~4 秒。"""
    monkeypatch.setattr(Z.time, "sleep", lambda *_: None)


class TestBodyKind:
    def test_classifies_json(self):
        assert Z._body_kind(b'{"paging":{}}') == "json"
        assert Z._body_kind(b'[{"a":1}]') == "json"

    def test_classifies_compression_residue(self):
        """⚠️ 这两类要和"被加密"分开：处置方式完全不同。

        `1f8b` 是 gzip 魔数 —— 说明解压没生效（requests 的 Accept-Encoding 协商问题），
        自己 gunzip 一下就好；而"其它二进制"是被风控替换，只能重试或换登录态。
        """
        assert "gzip" in Z._body_kind(b"\x1f\x8b\x08\x00rest")
        assert "zstd" in Z._body_kind(b"\x28\xb5\x2f\xfd xxx")

    def test_classifies_encrypted_payload(self):
        """还原事故现场：那次乱码既不是 JSON 也不是已知压缩头。"""
        kind = Z._body_kind(b"\xbf\xb4\xc8\x88\xbf\xb4\x5e\xbf")
        assert "非 JSON" in kind and "gg" not in kind          # 不是 gzip
        assert "bfb4c888" in kind, "必须记下 magic bytes，否则下次没法判断"

    def test_empty_body(self):
        assert Z._body_kind(b"") == "空响应体"


class TestRetryOnGarbage:
    def test_garbage_then_success(self, isolated, monkeypatch, no_sleep):
        """⭐⭐ 核心：一次乱码**不能**毁掉整趟采集。

        实测形状：15:20 那一次就是"服务端 200 但响应体是乱码"，
        而同一分钟之外重试是好的 ⇒ 必须重试。
        """
        bad = _Resp({"whatever": 1}, raw=b"\xbf\xb4\xc8\x88\xbf\xb4")   # 乱码
        good = _Resp({"data": [_PIN]})
        s = _ScriptedSession([bad, good])
        monkeypatch.setattr(Z, "_session", lambda: s)

        items = Z.member_pins("someone", limit=1)
        assert len(items) == 1, "第二次拿到正常 JSON 就必须返回数据，不能空手而归"
        assert len(s.calls) == 2, "第一次乱码后必须重试一次"

    def test_all_retries_fail_reports_reason(self, isolated, monkeypatch, no_sleep):
        """全失败要**说清是什么形状的失败**，而不是含糊的"未取到"。"""
        s = _ScriptedSession([
            _Resp({"x": 1}, raw=b"\xbf\xb4\xc8\x88"),
            _Resp({"x": 1}, raw=b"\xbf\xb4\xc8\x88"),
            _Resp({"x": 1}, raw=b"\xbf\xb4\xc8\x88"),
        ])
        monkeypatch.setattr(Z, "_session", lambda: s)
        assert Z.member_pins("someone", limit=1) == []
        assert len(s.calls) == Z._WEB_RETRIES
        le = Z.last_error()
        assert "重试" in le.get("reason", "") or "重试" in le.get("detail", ""), \
            f"错误详情必须说明重试过，实际 {le}"
        assert "cookie" in (le.get("detail") or "").lower(), \
            "应给出可操作建议（配 Cookie），否则用户只知道坏了"

    def test_404_is_not_retried(self, isolated, monkeypatch, no_sleep):
        """404 = 博主不存在，重试 3 次只是浪费 —— 必须立即返回。"""
        s = _ScriptedSession([_Resp({"error": "not found"}, status=404)])
        monkeypatch.setattr(Z, "_session", lambda: s)
        assert Z.member_pins("nobody", limit=1) == []
        assert len(s.calls) == 1, "404 不该重试"

    def test_403_is_not_retried(self, isolated, monkeypatch, no_sleep):
        s = _ScriptedSession([_Resp({"error": "need_login"}, status=403)])
        monkeypatch.setattr(Z, "_session", lambda: s)
        assert Z.member_pins("someone", limit=1) == []
        assert len(s.calls) == 1, "403 要登录态，重试也不会变"

    def test_network_error_is_retried(self, isolated, monkeypatch, no_sleep):
        """连接层抖动（断网/超时）与乱码一样值得重试。"""
        s = _ScriptedSession([
            ConnectionError("connection reset"),
            _Resp({"data": [_PIN]}),
        ])
        monkeypatch.setattr(Z, "_session", lambda: s)
        assert len(Z.member_pins("someone", limit=1)) == 1
        assert len(s.calls) == 2


class TestCookieSupport:
    def test_env_var_wins(self, tmp_path, monkeypatch):
        p = tmp_path / "cookie.txt"
        p.write_text("file_cookie=1", encoding="utf-8")
        monkeypatch.setattr(Z, "_ZHIHU_COOKIE_PATH", str(p))
        monkeypatch.setattr(Z, "_COOKIE_CACHE", {"mtime": None, "value": ""})
        monkeypatch.setenv("BIGA_ZHIHU_COOKIE", "env_cookie=1")
        assert Z.load_cookie() == "env_cookie=1"

    def test_file_with_prefix_is_stripped(self, tmp_path, monkeypatch):
        """用户会直接粘浏览器里那整行 `Cookie: xxx` ⇒ 必须容忍。"""
        p = tmp_path / "cookie.txt"
        p.write_text("Cookie: z_c0=abc; d_c0=def\n", encoding="utf-8")
        monkeypatch.setattr(Z, "_ZHIHU_COOKIE_PATH", str(p))
        monkeypatch.setattr(Z, "_COOKIE_CACHE", {"mtime": None, "value": ""})
        monkeypatch.delenv("BIGA_ZHIHU_COOKIE", raising=False)
        assert Z.load_cookie() == "z_c0=abc; d_c0=def"

    def test_absent_cookie_is_empty_not_error(self, tmp_path, monkeypatch):
        """没配 Cookie 也要能跑（当前匿名访问多数时候可用）。"""
        monkeypatch.setattr(Z, "_ZHIHU_COOKIE_PATH", str(tmp_path / "nope.txt"))
        monkeypatch.setattr(Z, "_COOKIE_CACHE", {"mtime": None, "value": ""})
        monkeypatch.delenv("BIGA_ZHIHU_COOKIE", raising=False)
        assert Z.load_cookie() == ""

    def test_cookie_lands_in_headers(self, monkeypatch):
        monkeypatch.setenv("BIGA_ZHIHU_COOKIE", "z_c0=xyz")
        h = Z._web_headers()
        assert h.get("Cookie") == "z_c0=xyz"
        assert "Authorization" not in h, "v4 不认开放平台 token，带上反而更易被拦"

    def test_header_omits_cookie_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Z, "_ZHIHU_COOKIE_PATH", str(tmp_path / "nope.txt"))
        monkeypatch.setattr(Z, "_COOKIE_CACHE", {"mtime": None, "value": ""})
        monkeypatch.delenv("BIGA_ZHIHU_COOKIE", raising=False)
        assert "Cookie" not in Z._web_headers(), "空 cookie 不能发一个空的 Cookie 头"
