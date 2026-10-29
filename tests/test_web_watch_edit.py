# -*- coding: utf-8 -*-
"""自选股写入（`POST /api/watch/watchlist`）测试。

## 为什么这个文件必须存在
在加这个端点之前，自选股**只有读通道**（`GET /api/watch/watchlist`）：
底层 `data.watchlist.add_stock/remove_stock/update_stock` 全都在，
AI 助手的 `tool_manage_watchlist` 也早就调得动，但**页面上没有任何入口**，
人只能靠跟模型说"帮我把某某加进自选"——这就是"自选股没法添加删除"的根因。
（同一个失效模式在本项目反复出现：后端齐了、前端只读。判据是
 `grep -rn <函数名> web/` 有没有非测试引用。）

## 钉住的三条语义
① **写成功必须让 `watch_list` 缓存失效**。ttl 是 15s，不失效的话紧接着的读
   会命中旧清单 ⇒ 页面显示"改了但没变"，而价格全对、接口也 `ok:true`。
   这是本项目"最难自查"的一类：不是失败，是**成功但读不回来**。
② **非 A 股代码必须挡住**（B 股 `900/200`、乱串）。放进去之后行情源永远取不到价，
   页面就长期显示"无报价"，人会以为是行情源挂了。
③ **`owner=ashen` 允许人写**，但 `source` 一律 `manual` —— 阿深自己写的行是
   `source=ashen`。禁止人改阿深的清单是过度设计（人想清理它的乱入必须能清），
   真正要保的是"谁改的"可查 —— 归因不能糊。
"""
import pytest

from tests.conftest import use_template_db


@pytest.fixture
def wl(tmp_path, monkeypatch):
    """独立库 + 强制离线（不为了补名称去外呼行情源）。

    ⚠️ 清空必须排在 `ensure_ready()` **之后**：首次迁移的判据是"库里没有行
       且没有迁移标记"，先清空再读会让 `_ensure_migrated()` 以为从没迁过，
       把默认池那 17 只**重新灌回来**——用例就会在"我明明清空了"的表上数出 17 只。
    """
    use_template_db(tmp_path, monkeypatch, "watch_edit.db")
    monkeypatch.setenv("BIGA_OFFLINE", "1")

    from data import watchlist as wld
    wld.ensure_ready()                       # 先把迁移标记打上
    wld.save_watchlist([], owner="user")
    wld.save_watchlist([], owner="ashen")

    from web.server import cache, watch
    cache.clear()
    return watch, wld


def codes(owner="user") -> list[str]:
    from data import watchlist as wld
    return [s["code"] for s in wld.load_watchlist(owner)]


# ================================================================
# 正确路径
# ================================================================

class TestAdd:

    def test_add_ok(self, wl):
        watch, _ = wl
        r = watch.edit_watchlist("add", "user", "600887", name="伊利股份")
        assert r["ok"] is True, r
        assert r["data"]["action"] == "add"
        assert codes() == ["600887"]

    def test_add_fills_name_from_quote_when_blank(self, wl, monkeypatch):
        """名称留空 ⇒ 用行情源补一次。补不到就留空，**但不能阻止写入**。"""
        watch, _ = wl
        monkeypatch.setattr(watch, "_guess_name", lambda c: "伊利股份")
        r = watch.edit_watchlist("add", "user", "600887")
        assert r["ok"] is True, r
        from data import watchlist as wld
        assert wld.load_watchlist()[0]["name"] == "伊利股份"

    def test_add_name_blank_offline_still_writes(self, wl):
        """_guess_name 拿不到（离线）时仍然写入，名字留空由行情读取侧兜底。"""
        watch, _ = wl
        r = watch.edit_watchlist("add", "user", "600887")
        assert r["ok"] is True, r
        assert codes() == ["600887"]

    def test_add_duplicate_rejected(self, wl):
        watch, _ = wl
        watch.edit_watchlist("add", "user", "600887")
        r = watch.edit_watchlist("add", "user", "600887")
        assert r["ok"] is False
        assert "已在" in r["error"]["detail"]

    def test_add_records_manual_source(self, wl):
        watch, _ = wl
        watch.edit_watchlist("add", "user", "600887")
        from data import watchlist as wld
        assert wld.load_watchlist()[0]["source"] == "manual"


class TestRemove:

    def test_remove_ok(self, wl):
        watch, _ = wl
        watch.edit_watchlist("add", "user", "600887")
        r = watch.edit_watchlist("remove", "user", "600887")
        assert r["ok"] is True, r
        assert codes() == []

    def test_remove_missing_rejected(self, wl):
        watch, _ = wl
        r = watch.edit_watchlist("remove", "user", "600887")
        assert r["ok"] is False
        assert "不在" in r["error"]["detail"]


class TestUpdate:

    def test_update_note(self, wl):
        watch, _ = wl
        watch.edit_watchlist("add", "user", "600887", name="伊利股份")
        r = watch.edit_watchlist("update", "user", "600887", note="乳业龙头")
        assert r["ok"] is True, r
        from data import watchlist as wld
        assert wld.load_watchlist()[0]["note"] == "乳业龙头"

    def test_update_note_empty_clears(self, wl):
        """`note=""` = 清空备注（不是"不改"）。这两个语义混一起会让人清不掉备注。"""
        watch, _ = wl
        watch.edit_watchlist("add", "user", "600887", name="伊利股份", note="乳业龙头")
        r = watch.edit_watchlist("update", "user", "600887", note="")
        assert r["ok"] is True, r
        from data import watchlist as wld
        assert wld.load_watchlist()[0]["note"] == ""

    def test_update_without_fields_rejected(self, wl):
        watch, _ = wl
        watch.edit_watchlist("add", "user", "600887")
        r = watch.edit_watchlist("update", "user", "600887")
        assert r["ok"] is False
        assert "name 或 note" in r["error"]["detail"]


# ================================================================
# 校验（结构性守护）
# ================================================================

class TestValidation:

    @pytest.mark.parametrize("bad", ["60088", "6008877", "abcdef", "", "  "])
    def test_bad_code_rejected(self, wl, bad):
        watch, _ = wl
        r = watch.edit_watchlist("add", "user", bad)
        assert r["ok"] is False
        assert "6 位数字" in r["error"]["detail"]

    @pytest.mark.parametrize("b_share", ["900901", "200002"])
    def test_b_share_rejected(self, wl, b_share):
        """B 股加进来永远取不到报价 ⇒ 结构性拒绝，而不是放进去再显示"无报价"。"""
        watch, _ = wl
        r = watch.edit_watchlist("add", "user", b_share)
        assert r["ok"] is False
        assert "不是 A 股" in r["error"]["detail"]

    def test_unknown_action_rejected(self, wl):
        watch, _ = wl
        r = watch.edit_watchlist("delete", "user", "600887")
        assert r["ok"] is False
        assert "未知 action" in r["error"]["detail"]

    def test_illegal_owner_falls_back_to_user(self, wl):
        """owner 走白名单：非法值退回 `user`，不报错也不写进野 owner。"""
        watch, _ = wl
        r = watch.edit_watchlist("add", "hacker", "600887")
        assert r["ok"] is True, r
        assert r["data"]["owner"] == "user"
        assert codes("user") == ["600887"]


# ================================================================
# 缓存失效（最容易漏的一条）
# ================================================================

class TestCacheInvalidation:
    """写之后必须能立刻读回新清单。

    ⚠️ 判据不能只看 `edit_watchlist` 的返回值——它永远 ok。要**先读一次**把旧值
       放进缓存，再写，再读，比对新值。只有这样才能抓住"写成功、读不回来"。
    """

    def test_add_visible_right_after_write(self, wl):
        watch, _ = wl
        first = watch.watchlist()                    # 空清单进缓存
        assert first["data"]["total"] == 0
        watch.edit_watchlist("add", "user", "600887")
        second = watch.watchlist()                  # 若没失效，这里还是 0 只
        assert second["data"]["total"] == 1, "写入后读到旧缓存 ⇒ 页面会显示'改了没变'"

    def test_remove_visible_right_after_write(self, wl):
        watch, _ = wl
        watch.edit_watchlist("add", "user", "600887")
        assert watch.watchlist()["data"]["total"] == 1      # 让"1 只"进缓存
        watch.edit_watchlist("remove", "user", "600887")
        assert watch.watchlist()["data"]["total"] == 0

    def test_update_visible_right_after_write(self, wl):
        watch, _ = wl
        watch.edit_watchlist("add", "user", "600887", name="伊利股份")
        assert watch.watchlist()["data"]["rows"][0]["note"] == ""
        watch.edit_watchlist("update", "user", "600887", note="乳业龙头")
        assert watch.watchlist()["data"]["rows"][0]["note"] == "乳业龙头"

    def test_cache_invalidate_scoped_to_namespace(self):
        """`invalidate` 只清指定 namespace，不顺手把别的页缓存一起清了。"""
        from web.server import cache
        cache.clear()
        cache.cached("a", "k", 60, lambda: "A")
        cache.cached("b", "k", 60, lambda: "B")
        assert cache.invalidate("a") == 1
        assert cache.cached("b", "k", 60, lambda: "B2") == "B"
        cache.clear()


# ================================================================
# 两轨 owner
# ================================================================

class TestOwners:

    def test_ashen_owner_writable(self, wl):
        """人可以改阿深的清单（它乱入的时候得能清）。"""
        watch, _ = wl
        r = watch.edit_watchlist("add", "ashen", "300498", name="温氏股份")
        assert r["ok"] is True, r
        assert codes("ashen") == ["300498"]
        assert codes("user") == [], "写阿深那轨不能串到我的清单"

    def test_ashen_write_marked_manual(self, wl):
        """人工写阿深清单要标 manual，才能和它研究轮自己写的区分开。"""
        watch, _ = wl
        watch.edit_watchlist("add", "ashen", "300498", name="温氏股份")
        from data import watchlist as wld
        assert wld.load_watchlist("ashen")[0]["source"] == "manual"


# ================================================================
# 端点接线（验函数 ≠ 验接线）
# ================================================================

class TestEndpoint:

    def test_post_route_registered(self):
        """端点必须真的挂在 router 上——函数写好但没人调是最常见的"好像做了"。"""
        import importlib
        import pkgutil

        import web.server.routers as R
        found = set()
        for m in pkgutil.iter_modules(R.__path__):
            rt = getattr(importlib.import_module(f"web.server.routers.{m.name}"),
                         "router", None)
            for r in (rt.routes if rt else []):
                if getattr(r, "path", "") == "/api/watch/watchlist":
                    found.update(getattr(r, "methods", set()) or set())
        assert {"GET", "POST"} <= found, f"实际方法：{found}"

    def test_post_body_forwarded(self, wl, monkeypatch):
        """路由把 body 的字段名原样透传（改名会让页面静默不生效）。"""
        watch, _ = wl
        from web.server.routers import watch as R

        seen = {}

        def fake(action, owner="user", code="", name=None, note=None):
            seen.update(action=action, owner=owner, code=code, name=name, note=note)
            return {"ok": True, "data": {}}

        monkeypatch.setattr(watch, "edit_watchlist", fake)
        R.api_watch_watchlist_edit({"action": "add", "owner": "ashen",
                                    "code": "600887", "name": "伊利", "note": "x"})
        assert seen == {"action": "add", "owner": "ashen", "code": "600887",
                        "name": "伊利", "note": "x"}

    def test_post_missing_fields_defaults(self, wl, monkeypatch):
        """缺字段不能抛 KeyError（前端一个字段没传就 500 是最糟的）。"""
        watch, _ = wl
        from web.server.routers import watch as R
        r = R.api_watch_watchlist_edit({})
        assert r["ok"] is False
        assert "未知 action" in r["error"]["detail"]
