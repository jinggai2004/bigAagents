# -*- coding: utf-8 -*-
"""阿深值班卡与手动唤醒（`/api/desk/ashen` / `/api/desk/ashen/wake`）。

这一组守的是**"权限开着≠在建仓"**那类静默失效：
· 池子空 ⇒ L2 没有标的，但页面若只显示"L2 已开启"就看不出来；
· 参数自相矛盾 ⇒ 每笔都被拒，表象同样是"没动作"；
· 手动唤醒若默认真下单 ⇒ 按一下就可能建仓（确认环节被删掉）。

⚠️ 全部用例走 `tpl_db`（模板库副本），不碰生产库。
"""
from __future__ import annotations

from datetime import datetime

import pytest

from web.server import desk


@pytest.fixture()
def tpl_db(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "desk_ashen.db")
    from db import store
    return store


@pytest.fixture()
def no_network(monkeypatch):
    """手动唤醒的行情一律打桩：用例不许真联网（离线闸门同款纪律）。"""
    import data.realtime as rt

    def _fake(codes, force_refresh=False):
        return {str(c): {"code": str(c), "name": "测试" + str(c)[-2:], "price": 10.0,
                         "prev_close": 10.0, "high": 10.2, "low": 9.9}
                for c in codes}

    monkeypatch.setattr(rt, "get_realtime_quotes", _fake)
    return _fake


class TestAshenCard:
    def test_shape_is_stable_without_account(self, tpl_db):
        """还没开过户也要有稳定结构：`exists=false`，不是 `None`、也不是 0。"""
        z = desk.ashen()
        assert z["ok"] is True
        d = z["data"]
        assert d["account"]["exists"] is False
        assert d["account"]["total_value"] is None
        assert d["account"]["valuation_complete"] is False
        for key in ("autonomy", "budget", "pools", "pool_hint", "candidates", "today"):
            assert key in d, key
        assert set(d["today"]) >= {"runs", "counts", "slots", "due_slot", "next_slot"}
        assert set(d["pools"]) == {"ashen", "user"}

    def test_level_and_params_come_from_single_source(self, tpl_db):
        """级别/参数只能来自 `ai_autonomy`，卡片不自己写 8% / 100%。"""
        from core import ai_autonomy
        d = desk.ashen()["data"]
        assert d["autonomy"]["level"] in ai_autonomy.LEVELS
        assert d["autonomy"]["params"] == ai_autonomy.resolve()
        assert d["autonomy"]["describe"] == ai_autonomy.describe(ai_autonomy.resolve())
        assert d["autonomy"]["consistent"] is True

    def test_empty_pool_is_explained_not_hidden(self, tpl_db, monkeypatch):
        """池子空必须成句说出来（"权限开了但没标的"），并给出候选数。"""
        from core import ai_routine
        monkeypatch.setattr(ai_routine, "_today_candidates",
                            lambda day, limit=8: [{"code": "600887", "name": "伊利股份"}])
        d = desk.ashen()["data"]
        assert d["pools"]["ashen"]["count"] == 0
        assert "0 只" in d["pool_hint"] and "没有标的" in d["pool_hint"]
        assert "1 只" in d["pool_hint"]

    def test_pool_read_failure_is_not_zero(self, tpl_db, monkeypatch):
        """库读挂了 ≠ 空池。二者混在一起会让人以为自选被删了。"""
        import data.watchlist as wl

        def boom(owner="user"):
            raise RuntimeError("库挂了")

        monkeypatch.setattr(wl, "snapshot", boom)
        d = desk.ashen()["data"]
        assert d["pools"]["ashen"]["available"] is False
        assert d["pools"]["ashen"]["count"] is None
        assert "读取失败" in d["pool_hint"]

    def test_inconsistent_params_are_flagged(self, tpl_db, monkeypatch):
        """参数不自洽 ⇒ 卡在明面上报出来（否则每笔都被拒且看不出原因）。"""
        from core import ai_autonomy
        bad = dict(ai_autonomy.resolve())
        bad["max_position_pct"] = 0.5
        monkeypatch.setattr(ai_autonomy, "resolve", lambda overrides=None: bad)
        d = desk.ashen()["data"]
        assert d["autonomy"]["consistent"] is False
        assert d["autonomy"]["violations"], "必须给出具体违规条目"

    def test_manual_runs_are_counted_separately(self, tpl_db):
        """排班轮与手动轮不能混成一个 total：否则无法判断系统自己有没有到岗。"""
        from db import store
        store.save_routine_run("ai_ashen", "2026-10-27", "09:40", status="ok",
                               llm_called=True)
        store.save_routine_run("ai_ashen", "2026-10-27", "手动", status="ok",
                               llm_called=True)
        c = desk.ashen(day="2026-10-27")["data"]["today"]["counts"]
        assert c["total"] == 2 and c["manual"] == 1

    def test_run_counter_excludes_manual_and_skipped(self, tpl_db):
        """页面「排班轮次 N/6」的分子只数**排班内真的调了模型**的轮次。

        分子用 `total` 会两头失真：手动演练一次就把 N 从 0 推到 1（看着像排班
        到岗了），被跳过的格子也占总数（看着像跑过 6 轮但其实一次模型都没调）。
        一天只有 6 个节点，"6/6 且全调了模型"才是这一天真的按点到岗。
        """
        from db import store
        store.save_routine_run("ai_ashen", "2026-10-27", "09:40", status="ok",
                               llm_called=True)
        store.save_routine_run("ai_ashen", "2026-10-27", "10:20", status="skipped",
                               llm_called=False)
        store.save_routine_run("ai_ashen", "2026-10-27", "手动", status="ok",
                               llm_called=True)
        c = desk.ashen(day="2026-10-27")["data"]["today"]["counts"]
        assert c["total"] == 3                     # 三行都在，历史不能隐瞒
        assert c["scheduled"] == 2                 # 排班占了两格（跑 + 跳）
        assert c["scheduled_llm"] == 1             # 真的到岗只有 1 轮


class TestManualWake:
    def _stub_run(self, monkeypatch, ret=None):
        from core import ai_routine
        seen = {}

        def fake(quotes, **kw):
            seen.update({"codes": sorted(quotes), **kw})
            return ret if ret is not None else {"llm_called": True, "slot": "手动",
                                                "kept": 0, "blocked": 0}

        monkeypatch.setattr(ai_routine, "run_due", fake)
        return seen

    def test_defaults_to_dry_run(self, tpl_db, no_network, monkeypatch):
        """默认 `dry_run=true` ⇒ `execute=False`。按一下就可能建仓是不可接受的。"""
        from db import store
        store.upsert_watchlist_item("ashen", "600887", "伊利股份", "等回踩")
        seen = self._stub_run(monkeypatch)
        z = desk.wake_ashen()
        assert z["ok"] is True and z["data"]["dry_run"] is True
        assert seen["execute"] is False
        assert z["data"]["accepted"] is True

    def test_explicit_false_executes(self, tpl_db, no_network, monkeypatch):
        from db import store
        store.upsert_watchlist_item("ashen", "600887", "伊利股份", "等回踩")
        seen = self._stub_run(monkeypatch)
        desk.wake_ashen(dry_run=False)
        assert seen["execute"] is True

    def test_quotes_follow_attention_pool(self, tpl_db, no_network, monkeypatch):
        """取行情的代码集合必须走 `attention_codes()`（自选 ∪ 今日候选 ∪ 持仓）。

        ⚠️ 实测事故：手动轮只取"自选池 ∪ 持仓"，而 `watch_add` 的主要来源是
        **今日候选** ⇒ 三条入池记录 `name` 全是空串。
        """
        import data.watchlist as wl
        monkeypatch.setattr(wl, "attention_codes",
                            lambda sources=None: ["600887", "000858"])
        seen = self._stub_run(monkeypatch)
        desk.wake_ashen()
        assert seen["codes"] == ["000858", "600887"]

    def test_no_quotes_is_business_result_not_failure(self, tpl_db, monkeypatch):
        """取不到行情 ⇒ `ok:true, accepted:false`（行情源的业务结果，不是接口坏）。"""
        import data.realtime as rt
        from db import store
        store.upsert_watchlist_item("ashen", "600887", "伊利股份", "等回踩")
        monkeypatch.setattr(rt, "get_realtime_quotes",
                            lambda codes, force_refresh=False: {})
        z = desk.wake_ashen()
        assert z["ok"] is True
        assert z["data"]["accepted"] is False
        assert "取不到实时行情" in z["data"]["reason"]

    def test_empty_pool_and_no_position_reports_why(self, tpl_db, monkeypatch):
        """两套池子都空、也无持仓 ⇒ 说清"没有可判的标的"，不去空跑一次模型。"""
        import data.watchlist as wl
        monkeypatch.setattr(wl, "attention_codes", lambda sources=None: [])
        z = desk.wake_ashen()
        assert z["ok"] is True and z["data"]["accepted"] is False
        assert "没有可判的标的" in z["data"]["reason"]


class TestDeskPageWiring:
    """页面必须真调这两个端点（"后端齐了、页面没有"是本项目反复出现的形态）。"""

    def _html(self) -> str:
        return open("web/templates/desk.html", encoding="utf-8").read()

    def test_page_calls_both_endpoints(self):
        s = self._html()
        assert "get('/api/desk/ashen')" in s
        assert "/api/desk/ashen/wake?dry_run=" in s

    def test_real_order_requires_confirm(self):
        """真下单必须先弹确认；缺了它，误点一次就产生了委托。"""
        s = self._html()
        assert "window.confirm" in s
        assert "method: 'POST'" in s

    def test_page_shows_pool_hint_and_violations(self):
        """后端的 `pool_hint` / `violations` 必须在页面上真的被渲染。"""
        s = self._html()
        assert "pool_hint" in s
        assert "violations" in s
        assert "参数自相矛盾" in s

    def test_manual_slot_is_not_dressed_as_a_time(self):
        """`手动` 槽位不许写成像时刻的字符串（否则像第 7 个班次）。"""
        from core import ai_routine
        assert ai_routine.MANUAL_SLOT == "手动"
        assert ai_routine.MANUAL_SLOT not in ai_routine.SLOTS
        assert not any(ch.isdigit() for ch in ai_routine.MANUAL_SLOT)


# ================================================================
# 情绪标记（2026-10-28）：阿深对话带表情
# ================================================================
#
# 背景：用户问"你现在能变表情吗"，阿深答"表情变不了，我这边就是个报数字的"。
# 实测房间里的角色早就有 setMood（happy/sad/neutral/think），只是对话不驱动它。
# 所以：prompt 让它输出【情绪】标记，后端剥出来单传 mood，前端驱动脸。
# 这些用例守的是：① 标记能剥干净；② 非法/缺失走代码兜底且**标 fallback**；
# ③ 兜底不能把"今天没动"配成微笑脸。

class TestMoodSplit:
    def test_strips_mark_from_answer(self):
        body, mood, src = desk._split_mood("今天没动，继续等。\n【情绪】think")
        assert mood == "think" and src == "model"
        assert "情绪" not in body, body
        assert body.strip() == "今天没动，继续等。"

    def test_accepts_bracket_form(self):
        body, mood, src = desk._split_mood("成交了 [情绪] happy")
        assert mood == "happy" and src == "model"

    def test_mark_anywhere_in_text(self):
        """标记可能在开头或中间 —— 正则要能剥，不能只在结尾才认。"""
        body, mood, src = desk._split_mood("【情绪】sad 我没做到")
        assert mood == "sad" and src == "model"
        assert body.strip() == "我没做到"

    def test_illegal_mood_is_discarded_to_fallback(self):
        """模型给了个不认识的词 ⇒ 不能透传给前端（前端只会认四种）。"""
        body, mood, src = desk._split_mood("今天还行【情绪】excited")
        assert mood != "excited"
        assert src == "fallback", (mood, src)
        assert mood in ("happy", "sad", "think", "neutral")

    def test_no_mark_uses_fallback(self):
        _, mood, src = desk._split_mood("今天没动，继续观察")
        assert src == "fallback"
        # "没动" 命中的是 sad 词表（且排在 think 之前），属设计行为：
        # 兜底只在模型没给标记时粗判，不追求语义精确。
        assert mood == "sad", mood

    def test_pure_wait_maps_to_think(self):
        """没有否定词、只有"等/观察" ⇒ 权衡类，而不是默认 neutral。"""
        _, mood, src = desk._split_mood("还在观察盘面变化")
        assert src == "fallback"
        assert mood == "think", mood

    def test_fallback_never_smiles_on_inaction(self):
        """兜底的底线：'没动/没买' 不能配 happy —— 那是比没表情更违和的错。"""
        _, mood, _ = desk._split_mood("今天没买成")
        assert mood == "sad", mood

    def test_fallback_maps_progress_to_happy(self):
        _, mood, _ = desk._split_mood("已经挂上条件了")
        assert mood == "happy", mood


class TestMoodInPrompt:
    def test_chat_rules_tell_model_it_has_a_face(self):
        """prompt 必须明确告诉它"你有脸、标记会改表情"，并纠正"变不了表情"的旧认知。"""
        assert "【情绪】" in desk._CHAT_RULES
        assert "变不了表情" in desk._CHAT_RULES, "要正面纠正它的旧自我认知"

    def test_chat_returns_mood_fields(self):
        """返回结构里有 mood 与 mood_source —— 前端和审计靠这两个字段。"""
        import inspect
        src = inspect.getsource(desk.ashen_chat)
        assert '"mood": mood' in src and '"mood_source": mood_src' in src, src


class TestEmbedMoodWiring:
    def test_ask_ashen_drives_girl_mood(self):
        """前端接线守护：回答到达后必须调 `girl.setMood`（否则后端白算）。"""
        import inspect
        src = inspect.getsource(desk)  # noqa: F841
        # embed.html 不在 python 模块里，直接读文件（前端 JS 的守护靠这个）
        from pathlib import Path
        html = (Path(__file__).resolve().parent.parent
                / "companion" / "embed.html").read_text(encoding="utf-8")
        assert "girl.setMood((z.data && z.data.mood)" in html, \
            "askAshen 没有把后端 mood 接到角色的脸上"
