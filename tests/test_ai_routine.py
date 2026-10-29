# -*- coding: utf-8 -*-
"""阿深定时值班（core/ai_routine.py）的测试。

## 这个文件在守什么

1. **`due_slot` 判间隔不判过点**：14:55 也满足 `hhmm >= '14:40'`，
   但那时补跑用的是过期信息。窗口是 30 分钟，超时不补。
2. **预筛在前、扣额度在后**：顺序反了会让上午几轮空跑吃光额度。
3. **归一不能丢三件套**：`normalize_actions` 只保留 action/code/size/reason，
   `limit_price`/`stop_loss` 会被丢掉 ⇒ 每笔买入都被判"缺限价"。
4. **首笔不被抬到 5%**：`clamp_plan` 默认最小有效仓位 5%，而首笔上限 3%
   ⇒ 不传 `min_position` 的话每笔首笔建仓都会被自己的判据拒掉。
5. **幂等靠库不靠内存**：重启后内存清零，靠内存判会重跑同一 slot。
"""

import json
from datetime import datetime

import pytest

from core import ai_routine as R


@pytest.fixture()
def tpl_db(tmp_path, monkeypatch):
    """独立空库（从模板复制）。**生产库一个字节都不许碰** —— conftest 有硬检查。"""
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "routine.db")
    return True


# ================================================================
# due_slot
# ================================================================
class TestDueSlot:
    def _at(self, hh, mm):
        return datetime(2026, 10, 27, hh, mm)

    def test_before_first_slot(self):
        assert R.due_slot(self._at(9, 30)) is None
        assert R.due_slot(self._at(9, 39)) is None

    def test_exactly_on_slot(self):
        assert R.due_slot(self._at(9, 40)) == "09:40"
        assert R.due_slot(self._at(14, 40)) == "14:40"

    def test_inside_window_returns_that_slot(self):
        assert R.due_slot(self._at(10, 5)) == "09:40"     # 25 分钟后，仍在窗口内
        assert R.due_slot(self._at(11, 29)) == "11:00"    # 29 分钟后

    def test_window_boundary_is_inclusive(self):
        assert R.due_slot(self._at(10, 10)) == "09:40"    # 正好 30 分钟

    def test_outside_window_does_not_backfill(self):
        """过期不补 —— 盘中决策 60 分钟就失效，用过期理由建仓是有害的。"""
        assert R.due_slot(self._at(10, 11)) is None        # 09:40 之后 31 分钟
        assert R.due_slot(self._at(12, 30)) is None        # 11:00 之后 90 分钟
        assert R.due_slot(self._at(15, 30)) is None

    def test_last_slot_window_covers_until_close(self):
        """14:40 之后 15 分钟仍属 S6 的补跑窗口（14:55 在收盘前，允许补）。

        ⚠️ 这与"14:55 不补"的直觉相反，但窗口的意义就是"到点后允许迟到"；
        真正的截止由 `scan_once` 的 `in_trading_session` 管（15:00 之后不跑）。
        """
        assert R.due_slot(self._at(14, 55)) == "14:40"
        assert R.due_slot(self._at(15, 11)) is None        # 30 分钟窗口已过

    def test_after_close(self):
        assert R.due_slot(self._at(15, 30)) is None


# ================================================================
# worth_llm（成本闸门）
# ================================================================
class TestWorthLLM:
    @pytest.fixture(autouse=True)
    def _no_armed(self, monkeypatch):
        """**不许耦合真实环境**：生产库里有已武装的唤醒条件，
        会让"首轮"和"静市"两条用例的结果取决于今天恰好挂了几条条件。
        """
        from core import ai_wake
        monkeypatch.setattr(ai_wake, "armed_codes", lambda *a, **kw: [])

    def _q(self, **kw):
        base = {"price": 10.0, "prev_close": 10.0, "change_pct": 0.0,
                "high": 10.1, "low": 9.9}
        base.update(kw)
        return base

    def test_first_round_always_runs(self):
        ok, why = R.worth_llm({"600519": self._q()}, last_run_at="")
        assert ok and "首轮" in why

    def test_holding_move_triggers(self):
        q = {"600519": self._q(change_pct=2.0)}
        ok, why = R.worth_llm(q, holdings=["600519"], last_run_at="")
        assert ok and "持仓" in why

    def test_watch_amplitude_triggers(self):
        q = {"000858": self._q(high=10.5, low=10.0)}      # 振幅 5%
        ok, why = R.worth_llm(q, my_watch=["000858"], last_run_at="")
        assert ok and "振幅" in why

    def test_quiet_market_skips(self):
        q = {"600519": self._q(change_pct=0.2), "000858": self._q(high=10.05, low=10.0)}
        ok, why = R.worth_llm(q, holdings=["600519"], my_watch=["000858"],
                              last_run_at="2026-10-27 09:52:00",
                              now=datetime(2026, 10, 27, 10, 5))
        assert not ok
        # 原因必须带具体数字，否则"为什么今天只跑了 2 轮"无法回答
        assert "%" in why and "分钟" in why

    def test_force_interval_triggers(self):
        q = {"600519": self._q(change_pct=0.2)}
        ok, why = R.worth_llm(q, holdings=["600519"],
                              last_run_at="2026-10-27 09:20:00",
                              now=datetime(2026, 10, 27, 10, 45))
        assert ok and "强制复盘" in why

    def test_skip_streak_triggers_even_if_interval_is_short(self):
        """连续跳过一轮就必跑，**不看**距上一轮多久。

        这条守的是一个真实的级联 bug：`last_run_at` 若取自"上一次落库的轮次"，
        跳过轮也会写 `created_at`，于是 gap 恒等于一个班距（40 分钟）< 80，
        每一格都"距上一轮 40 分钟"继续跳过 —— 09:40 跑过之后，10:20、11:00、
        14:00、14:40 全部跳过，一天只剩 2 轮（实测）。
        用秒级时间判据兜不住：一轮落在 09:44:50、下一轮 10:20:05 时 gap=75 分钟，
        "隔一班复盘"会随机退化成"隔两班"。
        """
        q = {"600519": self._q(change_pct=0.2)}
        ok, why = R.worth_llm(q, holdings=["600519"],
                              last_run_at="2026-10-27 09:52:00",
                              skips_since_llm=1,
                              now=datetime(2026, 10, 27, 10, 25))
        assert ok and "连续跳过" in why

    def test_skip_streak_zero_keeps_quiet_market_quiet(self):
        q = {"600519": self._q(change_pct=0.2)}
        ok, _ = R.worth_llm(q, holdings=["600519"],
                            last_run_at="2026-10-27 09:52:00",
                            skips_since_llm=0,
                            now=datetime(2026, 10, 27, 10, 25))
        assert not ok

    def test_missing_quote_does_not_crash(self):
        ok, _ = R.worth_llm({"600519": {}}, holdings=["600519"], last_run_at="")
        assert isinstance(ok, bool)


# ================================================================
# 归一：三件套与首笔
# ================================================================
class TestNormalize:
    def setup_method(self):
        from core import ai_personas
        ai_personas.invalidate()
        self.persona = ai_personas.get_persona("ai_ashen")

    def test_three_pieces_survive_normalization(self):
        """`normalize_actions` 会丢 limit_price/stop_loss ⇒ 这里必须回填。

        不钉这一条，线上表现是"每笔买入都缺限价"（看起来像模型不听话），
        实际是归一层的字段丢失。
        """
        out = R._normalize([{"action": "buy", "code": "600519", "position_size": 0.03,
                             "limit_price": 1600.0, "stop_loss": 1500.0,
                             "reason": "回调到支撑"}], self.persona)
        buy = [d for d in out if d["action"] == "buy"][0]
        assert buy["limit_price"] == 1600.0
        assert buy["stop_loss"] == 1500.0

    def test_first_buy_not_lifted_to_min_effective_position(self):
        """首笔 3% 不能被抬到 5%（否则被"首笔 ≤3%"判据拒掉，每笔都失败）。"""
        out = R._normalize([{"action": "buy", "code": "600519", "position_size": 0.03,
                             "limit_price": 1600.0, "stop_loss": 1500.0,
                             "reason": "x"}],
                           self.persona, min_position=0.03)
        buy = [d for d in out if d["action"] == "buy"][0]
        assert abs(buy["position_size"] - 0.03) < 1e-9, buy["position_size"]

    def test_extra_actions_kept(self):
        out = R._normalize([{"action": "watch_add", "code": "000858", "reason": "盯"},
                            {"action": "study", "code": "600519", "reason": "查"},
                            {"action": "transfer", "code": "600519"}], self.persona)
        acts = sorted(d["action"] for d in out)
        assert acts == ["study", "watch_add"], "未登记动作必须被丢掉"


# ================================================================
# 落库与幂等
# ================================================================
class TestRoutineStore:
    def test_save_and_read(self, tpl_db):
        from db import store
        assert store.routine_exists("ai_ashen", "2026-10-27", "09:40") is False
        store.save_routine_run("ai_ashen", "2026-10-27", "09:40", status="ok",
                               llm_called=True, summary="不动",
                               actions=[{"action": "pass"}],
                               blocked=[{"code": "600519", "reason": "追涨"}])
        assert store.routine_exists("ai_ashen", "2026-10-27", "09:40") is True
        rows = store.list_routine_runs("2026-10-27", persona="ai_ashen")
        assert len(rows) == 1
        assert rows[0]["llm_called"] == 1
        assert rows[0]["blocked"][0]["code"] == "600519"

    def test_upsert_overwrites_same_slot(self, tpl_db):
        """同一个 slot 重复写**覆盖**而不是插两行（手动补跑要能修正）。"""
        from db import store
        store.save_routine_run("ai_ashen", "2026-10-27", "09:40", status="skipped",
                               skip_reason="无变化")
        store.save_routine_run("ai_ashen", "2026-10-27", "09:40", status="ok",
                               llm_called=True, summary="补跑")
        rows = store.list_routine_runs("2026-10-27", persona="ai_ashen")
        assert len(rows) == 1
        assert rows[0]["status"] == "ok" and rows[0]["llm_called"] == 1

    def test_bad_status_falls_back_to_error(self, tpl_db):
        from db import store
        store.save_routine_run("ai_ashen", "2026-10-27", "09:40", status="诡异值")
        assert store.list_routine_runs("2026-10-27", persona="ai_ashen")[0]["status"] == "error"

    def test_broken_json_does_not_break_history(self, tpl_db):
        """一行坏 JSON 不能让整页历史读挂。"""
        from db import store
        store.save_routine_run("ai_ashen", "2026-10-27", "09:40", status="ok")
        conn = store.get_connection()
        with conn:
            conn.execute("UPDATE ai_routine_runs SET actions_json = '{坏'")
        conn.close()
        rows = store.list_routine_runs("2026-10-27", persona="ai_ashen")
        assert rows[0]["actions"] == []
        assert rows[0]["actions_raw"] == "{坏"


# ================================================================
# run_due 端到端（不打网络、不下真单）
# ================================================================
class _FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def call_api(self, messages, **kw):
        self.calls += 1
        return json.dumps(self.payload, ensure_ascii=False)


class TestRunDue:
    NOW = datetime(2026, 10, 27, 9, 45)
    # ⚠️ 用 10 元的票，不是 1600 元的：100 万账户 × 单票 8% = 8 万，
    #    而茅台一手 ≈ 16 万 ⇒ 8% 上限下一手都买不起（见 `test_expensive_stock_*`）。
    Q = {"600519": {"code": "600519", "name": "测试股份", "price": 10.0,
                    "prev_close": 10.0, "change_pct": -0.5,
                    "high": 10.1, "low": 9.9, "volume_hand": 200000.0,
                    "amount_wan": 80000.0}}

    def _seed(self, autonomy="L2"):
        from core import ai_personas
        from db import store
        ai_personas.invalidate()
        p = dict(ai_personas.get_persona("ai_ashen"))
        p["autonomy"] = autonomy
        store.upsert_watchlist_item("ashen", "600519", "测试股份", "等回调")
        return p

    def _patch_persona(self, monkeypatch, persona):
        from core import ai_personas
        monkeypatch.setattr(ai_personas, "get_persona",
                            lambda name: persona if name == "ai_ashen" else None)

    def _buy_reply(self):
        return {"actions": [{"action": "buy", "code": "600519", "position_size": 0.03,
                             "limit_price": 10.0, "stop_loss": 9.5,
                             "reason": "回调到支撑位，按计划建首笔"}],
                "summary": "建首笔 3%"}

    def test_not_due_slot(self, tpl_db, monkeypatch):
        p = self._seed()
        self._patch_persona(monkeypatch, p)
        out = R.run_due(self.Q, now=datetime(2026, 10, 27, 12, 30))
        assert "无到点的轮次" in out["skipped"]

    def test_idempotent(self, tpl_db, monkeypatch):
        from db import store
        p = self._seed()
        self._patch_persona(monkeypatch, p)
        store.save_routine_run("ai_ashen", "2026-10-27", "09:40", status="ok",
                               llm_called=True)
        out = R.run_due(self.Q, now=self.NOW)
        assert "已跑过" in out["skipped"]

    def test_force_outside_slot_uses_manual_slot(self, tpl_db, monkeypatch):
        """手动补跑在窗口外也要能跑，落库槽位是「手动」。

        ⚠️ 这条守的是页面按钮可用性：`due_slot` 只认"到点后 30 分钟内"，
        18:30 点「演练」会永远得到"无到点的轮次"，而这几乎总是成立
        （一天只有 6×30 分钟的窗口）。同时槽位**不能**编一个像时刻的字符串，
        否则"每天 ≤6 条"这条验收判据会被第 7 个假班次污染。
        """
        from db import store
        p = self._seed()
        self._patch_persona(monkeypatch, p)
        fake = _FakeClient(self._buy_reply())
        out = R.run_due(self.Q, now=datetime(2026, 10, 27, 18, 30),
                        force=True, execute=False, client_factory=lambda _p: fake)
        assert out.get("llm_called") is True, out
        assert out["slot"] == R.MANUAL_SLOT == "手动"
        rows = store.list_routine_runs("2026-10-27", persona="ai_ashen")
        assert [r["slot"] for r in rows] == ["手动"]

    def test_manual_runs_overwrite_same_cell(self, tpl_db, monkeypatch):
        """同一交易日手动两次 ⇒ 仍只有一格（手动轮不是排班，留多条=凭空多班次）。"""
        from db import store
        p = self._seed()
        self._patch_persona(monkeypatch, p)
        for _ in range(2):
            R.run_due(self.Q, now=datetime(2026, 10, 27, 18, 30), force=True,
                      execute=False,
                      client_factory=lambda _p: _FakeClient(self._buy_reply()))
        rows = store.list_routine_runs("2026-10-27", persona="ai_ashen")
        assert len(rows) == 1, rows
        assert R.history("2026-10-27")["counts"]["manual"] == 1

    def _set_created(self, slot, ts):
        from db import store
        conn = store.get_connection()
        with conn:
            conn.execute("UPDATE ai_routine_runs SET created_at = ? WHERE persona = 'ai_ashen'"
                         " AND trade_date = '2026-10-27' AND slot = ?", (ts, slot))
        conn.close()

    def test_skipped_round_does_not_cascade(self, tpl_db, monkeypatch):
        """被跳过的那一格不能把后面的格一起拖下水。

        改前 `last_at` 取"上一次落库的 created_at"，**跳过轮也落库**，于是 10:20
        跳过之后，11:00 看到的仍是"距上一轮 40 分钟"< 80 ⇒ 继续跳过；一天只剩
        09:40 与 13:20 两轮，"每天 6 条"永远不可能达标。而库里那批已武装的唤醒
        条件会让信号③恒真，把这个 bug 完全掩盖住 —— 只能靠这条用例守住。
        """
        from db import store
        p = self._seed()
        self._patch_persona(monkeypatch, p)
        quiet = {"600519": {**self.Q["600519"], "change_pct": 0.1,
                            "high": 10.005, "low": 10.0}}
        # 09:40 真实跑过一轮（首轮必跑，用来定当日计划）
        store.save_routine_run("ai_ashen", "2026-10-27", "09:40", status="ok",
                               llm_called=True, summary="首轮定了当日计划")
        self._set_created("09:40", "2026-10-27 09:40:10")

        fake1 = _FakeClient(self._buy_reply())
        out1 = R.run_due(quiet, now=datetime(2026, 10, 27, 10, 20, 10),
                         execute=False, client_factory=lambda _p: fake1)
        assert out1["llm_called"] is False and fake1.calls == 0, out1
        assert "无变化" in out1["skipped"]

        # 11:00 —— 必须跑，理由是"已连续跳过 1 轮"，不是"距上一轮 80 分钟"
        # 顺带守接线：`skips_since_llm` 必须真的被算出来**并传进**预筛。
        # 只验 `worth_llm(skips_since_llm=1)` 通过是不够的 —— 那验的是函数，
        # 不是它有没有被接上（本项目铁律：验函数 ≠ 验接线）。
        seen = {}
        real_worth = R.worth_llm

        def spy(quotes, **kw):
            seen.update(kw)
            return real_worth(quotes, **kw)

        monkeypatch.setattr(R, "worth_llm", spy)
        fake2 = _FakeClient(self._buy_reply())
        out2 = R.run_due(quiet, now=datetime(2026, 10, 27, 11, 0, 10),
                         execute=False, client_factory=lambda _p: fake2)
        assert seen.get("skips_since_llm") == 1, seen
        assert seen.get("last_run_at") == "2026-10-27 09:40:10", seen
        assert out2["llm_called"] is True and fake2.calls == 1, out2

    def test_prefilter_skip_does_not_consume_budget(self, tpl_db, monkeypatch):
        """预筛说"不值得"⇒ 落 skipped、llm_called=0、**不扣额度**。

        顺序反了会把额度用在"什么都没发生"的轮次上，
        14:40 那轮（最需要动手）就没得跑了。
        """
        from core import ai_budget
        from db import store
        p = self._seed()
        self._patch_persona(monkeypatch, p)
        quiet = {"600519": {**self.Q["600519"], "change_pct": 0.1,
                            "high": 10.005, "low": 10.0}}
        # 先放一条"15 分钟前跑过"的记录：否则 `worth_llm` 会判"今日首轮，必跑"
        # （首轮豁免是刻意的 —— 一天的第一轮必须跑，用来定当日计划）。
        store.save_routine_run("ai_ashen", "2026-10-27", "09:00", status="ok",
                               llm_called=True)
        conn = store.get_connection()
        with conn:
            conn.execute("UPDATE ai_routine_runs SET created_at = '2026-10-27 09:30:00'")
        conn.close()
        fake = _FakeClient(self._buy_reply())
        out = R.run_due(quiet, now=self.NOW, client_factory=lambda _p: fake)
        assert out["llm_called"] is False
        assert fake.calls == 0
        rows = store.list_routine_runs("2026-10-27", persona="ai_ashen")
        assert rows[0]["status"] == "skipped" and rows[0]["llm_called"] == 0
        assert ai_budget.view("ai_ashen", "2026-10-27")["routine"]["used"] == 0

    def test_L2_builds_first_position(self, tpl_db, monkeypatch):
        from core import ai_budget
        p = self._seed("L2")
        self._patch_persona(monkeypatch, p)
        fake = _FakeClient(self._buy_reply())
        out = R.run_due(self.Q, now=self.NOW, execute=False,
                        client_factory=lambda _p: fake)
        assert out["llm_called"] is True and out["kept"] >= 1, out
        assert out["blocked"] == 0, out["violations"]
        assert ai_budget.view("ai_ashen", "2026-10-27")["routine"]["used"] == 1

    def test_L1_cannot_build(self, tpl_db, monkeypatch):
        """L1 时同一份动作必须被拦 —— 这是建仓权的分界线。"""
        p = self._seed("L1")
        self._patch_persona(monkeypatch, p)
        fake = _FakeClient(self._buy_reply())
        out = R.run_due(self.Q, now=self.NOW, execute=False,
                        client_factory=lambda _p: fake)
        assert out["kept"] == 0
        assert any("需 L2" in v for v in out["violations"]), out["violations"]

    def test_expensive_stock_cannot_buy_one_lot(self, tpl_db, monkeypatch):
        """高价股在 8% 单票上限下**一手都买不起** —— 必须明确拒并说清原因。

        100 万 × 8% = 8 万，茅台一手 ≈ 16 万。没有这条判据时执行层只会返回
        "不足一手"，看起来像"模型仓位给太小"，实际是账户规模买不起这只票。
        说不清原因，阿深会反复提名同一只买不了的票。
        """
        p = self._seed()
        self._patch_persona(monkeypatch, p)
        expensive = {"600519": {**self.Q["600519"], "price": 1600.0, "prev_close": 1600.0,
                                "high": 1610.0, "low": 1590.0}}
        fake = _FakeClient({"actions": [{"action": "buy", "code": "600519",
                                        "position_size": 0.03, "limit_price": 1600.0,
                                        "stop_loss": 1500.0, "reason": "贵但想买"}],
                            "summary": "试"})
        out = R.run_due(expensive, now=self.NOW, execute=False,
                        client_factory=lambda _p: fake)
        assert out["kept"] == 0
        assert any("一手" in v for v in out["violations"]), out["violations"]

    def test_watch_add_really_writes_pool(self, tpl_db, monkeypatch):
        """`watch_add` 判过了还必须**真写库** —— 只判不写等于池子永远空。

        这条是"能否建仓"的前置：L2 只买自己池内的票，池子空 ⇒ 建仓权用不出来。
        """
        from db import store
        p = self._seed()
        self._patch_persona(monkeypatch, p)
        reply = {"actions": [{"action": "watch_add", "code": "600519",
                              "reason": "今日候选里评分 88.6，等回调"}],
                 "summary": "先加池"}
        out = R.run_due(self.Q, now=self.NOW, execute=True,
                        client_factory=lambda _p: _FakeClient(reply))
        assert out["kept"] == 1, out["violations"]
        rows = store.list_watchlist("ashen")
        assert any(r["code"] == "600519" and "评分" in r["note"] for r in rows), rows

    def test_watch_add_name_falls_back_to_candidates(self, tpl_db, monkeypatch):
        """入池的票**没在本次 quotes 里**时，名称要从今日候选兜底。

        ⚠️ 实测事故：手动唤醒取行情用"自选池 ∪ 持仓"，而入池的主要来源是
        **今日候选** ⇒ 三条 `watch_add` 落库 `name` 全是空串，页面上只剩裸代码。
        行情集合的口径已改为 `attention_codes()`，这条守住"即使行情缺也留得下名字"。
        """
        from db import store
        p = self._seed()
        self._patch_persona(monkeypatch, p)
        # 候选 json（只给 888888 这一条，避免依赖真实文件的当日内容）
        monkeypatch.setattr(R, "_today_candidates",
                            lambda day, limit=8: [{"code": "888888", "name": "兜底股份",
                                                   "score": 90, "board": "测试"}])
        reply = {"actions": [{"action": "watch_add", "code": "888888",
                              "reason": "候选第一名"}],
                 "summary": "加池"}
        out = R.run_due(self.Q, now=self.NOW, execute=True,
                        client_factory=lambda _p: _FakeClient(reply))
        assert out["kept"] == 1, out
        rows = {r["code"]: r for r in store.list_watchlist("ashen")}
        assert rows["888888"]["name"] == "兜底股份", rows

    def test_study_is_recorded_as_not_yet_executed(self, tpl_db, monkeypatch):
        """研究队列 P2 才有 ⇒ 必须如实写"只登记"，不能假装已排期。"""
        p = self._seed()
        self._patch_persona(monkeypatch, p)
        reply = {"actions": [{"action": "study", "code": "600519",
                              "reason": "想知道它三季报毛利率下滑的原因"}],
                 "summary": "登记研究"}
        out = R.run_due(self.Q, now=self.NOW, execute=True,
                        client_factory=lambda _p: _FakeClient(reply))
        studies = [o for o in out["orders"] if o.get("action") == "study"]
        assert studies and "尚未上线" in studies[0]["skipped"]

    def test_llm_error_is_recorded_not_raised(self, tpl_db, monkeypatch):
        from db import store
        p = self._seed()
        self._patch_persona(monkeypatch, p)

        class Boom:
            def call_api(self, *a, **kw):
                raise RuntimeError("网络断了")
        out = R.run_due(self.Q, now=self.NOW, client_factory=lambda _p: Boom())
        assert "LLM 调用失败" in out["skipped"]
        rows = store.list_routine_runs("2026-10-27", persona="ai_ashen")
        assert rows[0]["status"] == "error"

    def test_account_created_on_first_run(self, tpl_db, monkeypatch):
        """没有账户 ⇒ nav=0 ⇒ 所有买入被拒 ⇒ 建仓权永远不会生效。"""
        from db import store
        p = self._seed()
        self._patch_persona(monkeypatch, p)
        assert store.load_paper_state("ai_ashen") is None
        R.run_due(self.Q, now=self.NOW, execute=False,
                  client_factory=lambda _p: _FakeClient(self._buy_reply()))
        st = store.load_paper_state("ai_ashen")
        assert st and st["cash"] == pytest.approx(1_000_000)

    def test_execute_submits_lot_sized_order(self, tpl_db, monkeypatch):
        """执行层：整手、带限价、T+1（提交后仍是 pending，不立即成交）。"""
        from db import store
        p = self._seed()
        self._patch_persona(monkeypatch, p)
        out = R.run_due(self.Q, now=self.NOW, execute=True,
                        client_factory=lambda _p: _FakeClient(self._buy_reply()))
        orders = [o for o in out["orders"] if o.get("action") == "buy"]
        assert orders and orders[0]["status"] == "pending"
        assert orders[0]["qty"] % 100 == 0 and orders[0]["qty"] > 0
        st = store.load_paper_state("ai_ashen")
        assert st["orders"], "委托必须落库"

    # ---------------------------------------------------------------- 互动性（2026-10-28）
    def test_set_wake_really_writes_condition(self, tpl_db, monkeypatch):
        """`set_wake` 必须**真的写进条件表**，不能只判不放行。

        ⚠️ 这是本轮修的真 bug：`_execute` 原来只有 watch_add / watch_remove /
        study / buy / sell 五个分支，**没有 set_wake**。它能过授权判据（L1 就允许
        "调唤醒条件"）、能进 `actions_json`、页面"动作数"还会 +1 —— 但条件表从不写入。
        实证：阿深连续 6 轮说"等伊利 28 元下方"，而 `ai_ashen` 的条件是 **0 条**，
        价位到了不会叫醒它，用户也查不到它在等什么。

        断言必须落到**库**上，不能只断言 `out["orders"]` 里有条 ok ——那仍然
        只是在看返回值（项目铁律：验函数 ≠ 验接线）。
        """
        from db import store
        p = self._seed()
        self._patch_persona(monkeypatch, p)
        reply = {"actions": [{"action": "set_wake", "code": "600519",
                              "kind": "price_below", "threshold": 9.2,
                              "reason": "等回踩 9.2 再考虑建首笔"}],
                 "summary": "等回踩 9.2"}
        R.run_due(self.Q, now=self.NOW, execute=True,
                  client_factory=lambda _p: _FakeClient(reply))
        conds = store.list_wake_conditions(persona="ai_ashen", armed_only=False)
        hit = [c for c in conds if c["code"] == "600519" and c["kind"] == "price_below"]
        assert hit, f"条件没落库 ⇒ 它的等待不存在。实际条件: {conds}"
        assert hit[0]["threshold"] == pytest.approx(9.2)
        assert "回踩" in (hit[0]["note"] or "")

    def test_wake_without_threshold_is_rejected(self, tpl_db, monkeypatch):
        """没有阈值/时刻的条件永远不会触发 ⇒ 拒收，别给用户一个假的"我在等"。"""
        from db import store
        p = self._seed()
        self._patch_persona(monkeypatch, p)
        reply = {"actions": [{"action": "set_wake", "code": "600519",
                              "kind": "price_below", "reason": "空口承诺"}],
                 "summary": "等个好价"}
        out = R.run_due(self.Q, now=self.NOW, execute=True,
                        client_factory=lambda _p: _FakeClient(reply))
        conds = store.list_wake_conditions(persona="ai_ashen", armed_only=False)
        assert not [c for c in conds if c["code"] == "600519"], conds
        # 被拒必须留痕，不能静默消失（否则"没动作"和"被拒"看起来一样）
        trace = json.dumps(out.get("orders") or [], ensure_ascii=False) + \
            json.dumps(out.get("violations") or [], ensure_ascii=False)
        assert "threshold" in trace or "wake" in trace.lower(), trace

    def test_prompt_carries_today_trail_and_armed(self):
        """prompt 必须带上「今天的判断轨迹」与「它自己挂着的等待条件」。

        为什么这是互动性的根：原来只带 `last_summary`（**上一轮**一句），
        模型看不到自己今天已经把同一句话说了 4 遍 ⇒ 实测 6 轮两两平均相似度
        46%、第 5→6 轮 95%，用户读到的是复读机。
        """
        txt = R.build_prompt(
            now=datetime(2026, 10, 28, 11, 0), slot="11:00", level="L2",
            auton_desc="可建仓", budget_desc="单票 ≤8%",
            cash=1_000_000, nav=1_000_000, holdings=[], my_watch=[], user_watch=[],
            quotes={}, candidates=[], last_summary="等伊利 28 元下方",
            trail=[{"slot": "09:40", "summary": "跌幅小，无买点", "actions": []},
                   {"slot": "10:20", "summary": "等伊利 28 元下方",
                    "actions": [{"action": "pass"}]}],
            armed=[{"code": "600887", "kind": "price_below", "threshold": 28.0}])
        assert "今天的判断轨迹" in txt
        assert "09:40 跌幅小，无买点" in txt, txt
        assert "10:20 等伊利 28 元下方" in txt, txt
        assert "600887 price_below 28" in txt, txt
        assert "禁止复述" in txt and "必须同时输出 `set_wake`" in txt, txt

    def test_prompt_says_so_when_trail_is_empty(self):
        """空轨迹要**明说**「今天还没跑过」，不能留一段空白让模型自己猜。"""
        txt = R.build_prompt(
            now=datetime(2026, 10, 28, 9, 40), slot="09:40", level="L2",
            auton_desc="可建仓", budget_desc="单票 ≤8%",
            cash=1_000_000, nav=1_000_000, holdings=[], my_watch=[], user_watch=[],
            quotes={}, candidates=[], trail=[], armed=[])
        assert "（今天还没跑过）" in txt and "（没有）" in txt, txt


# ================================================================
# 动作闸门（2026-10-28）：把"必须动"从 prompt 的软约束改成代码的硬约束
# ================================================================
#
# 背景（实证，不是推测）：prompt 里早已写着"说在等就必须挂 set_wake"，而
# 2026-10-28 六轮的结果全是 `[('pass', None, None)]`，`ai_wake_conditions`
# 与 `ai_decisions` 里 ai_ashen 都是 **0 条**、账户成交 **0 笔**。
# ⇒ prompt 层的强制**等于没有**，模型会礼貌地忽略它且不留痕迹。
# 下面这些用例守的是：① 缺口能被判出来；② 只补正一次（不能每轮翻倍花钱）；
# ③ 两次都没动作才有兜底；④ 兜底只动池子、绝不下单。

class _SeqClient:
    """按顺序回多份答案。用来验证"首轮没动作 ⇒ 补正一次"的链路与调用次数。"""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def call_api(self, messages, **kw):
        p = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return json.dumps(p, ensure_ascii=False)


class TestActionGap:
    """`action_gap` 是纯函数：没有库、没有网络，判定必须完全确定。"""

    @pytest.mark.parametrize("summary,actions,expected", [
        ("等伊利 28 元下方", [{"action": "pass"}], "no_action"),
        ("等伊利 28 元下方", [{"action": "set_wake"}], ""),
        ("等伊利 28 元下方", [{"action": "watch_add"}], "missing_wake"),
        ("再跌 2% 再看", [{"action": "buy"}], "missing_wake"),
        ("本轮建首笔 3%", [{"action": "buy"}], ""),
        ("今天什么都没说", [{"action": "pass"}], "no_action"),
        # 有实质动作、且 summary 没有等待语义 ⇒ 不缺口（不能为了挂而挂）
        ("按计划移出温氏", [{"action": "watch_remove"}], ""),
    ])
    def test_gap(self, summary, actions, expected):
        assert R.action_gap(summary, actions) == expected

    def test_study_is_not_real_action(self):
        """`study` 只是登记问题（P2 才排期），不能算"这一天有动作"。"""
        assert R.action_gap("登记研究", [{"action": "study"}]) == "no_action"

    def test_none_actions_is_no_action(self):
        assert R.action_gap("等", None) == "no_action"


class TestAutoWatch:
    def test_skips_codes_already_in_pool(self):
        got = R._auto_watch([{"code": "600519"}, {"code": "000001"}],
                            [{"code": "600519"}])
        assert got["code"] == "000001"

    def test_returns_none_when_nothing_to_add(self):
        """没有可加的候选 ⇒ 如实返回 None，**不编一个动作出来**。"""
        assert R._auto_watch([{"code": "600519"}], [{"code": "600519"}]) is None
        assert R._auto_watch([], []) is None

    def test_reason_says_it_is_fallback(self):
        """兜底动作必须自报来历，否则复盘会把它读成"阿深的选择"。"""
        got = R._auto_watch([{"code": "600519"}], [])
        assert "系统兜底" in got["reason"]


class TestActionGate(TestRunDue):
    """端到端：假客户端控制模型回答，验证闸门、补正与兜底。"""

    PASS = {"actions": [{"action": "pass", "reason": "继续观察"}],
            "summary": "自选三票跌幅不够，等伊利 28 元下方或跌超 2% 再看"}
    WAKE = {"actions": [{"action": "set_wake", "code": "600519",
                         "kind": "price_below", "threshold": 9.5,
                         "reason": "等回踩到 9.5 再考虑"}],
            "summary": "已挂 9.5 的回踩条件"}

    def _run(self, monkeypatch, fake, execute=False):
        p = self._seed()
        self._patch_persona(monkeypatch, p)
        return R.run_due(self.Q, now=self.NOW, force=True, execute=execute,
                         client_factory=lambda _p: fake)

    def test_pass_only_triggers_exactly_one_retry(self, tpl_db, monkeypatch):
        """首轮只有 pass ⇒ 必须补正；补正给了 set_wake 就采用它，且**只补一次**。"""
        fake = _SeqClient([self.PASS, self.WAKE])
        out = self._run(monkeypatch, fake)
        assert fake.calls == 2, f"应只补正一次，实际 {fake.calls} 次"
        acts = [a.get("action") for a in out.get("orders", [])] or []
        assert out["kept"] == 1, out
        assert out["summary"] == self.WAKE["summary"], out["summary"]

    def test_good_first_reply_costs_one_call(self, tpl_db, monkeypatch):
        """首轮就有实质动作 ⇒ **绝不重试**（否则每天成本凭空翻倍）。"""
        fake = _SeqClient([self.WAKE])
        self._run(monkeypatch, fake)
        assert fake.calls == 1, f"不需要补正时不许多花钱，实际 {fake.calls} 次"

    def test_waiting_words_without_wake_also_retry(self, tpl_db, monkeypatch):
        """有动作但 summary 说在等、却没挂条件 ⇒ 同样补正（这是今天六轮的病）。"""
        first = {"actions": [{"action": "watch_add", "code": "000001", "reason": "加池"}],
                 "summary": "等伊利回到 28 元下方再看"}
        fake = _SeqClient([first, self.WAKE])
        self._run(monkeypatch, fake)
        assert fake.calls == 2

    def test_two_passes_fall_back_to_auto_watch(self, tpl_db, monkeypatch):
        """两次都只给 pass ⇒ 兜底入池，且兜底动作要标明是系统做的。"""
        monkeypatch.setattr(R, "_today_candidates",
                            lambda day, limit=8: [{"code": "000001", "name": "候选股"}])
        fake = _SeqClient([self.PASS, self.PASS])
        out = self._run(monkeypatch, fake)
        assert fake.calls == 2
        assert "auto_watch" in out.get("enforced", ""), out
        assert out["kept"] == 1, out
        from db import store
        rows = store.list_routine_runs("2026-10-27", persona="ai_ashen")
        assert "auto_watch" in (rows[0].get("enforced") or ""), rows[0]
        acts = json.loads(rows[0]["actions_raw"] or "[]") if rows[0].get("actions_raw") \
            else rows[0].get("actions") or []
        assert any(a.get("action") == "watch_add" for a in acts), acts

    def test_no_fallback_when_nothing_to_add(self, tpl_db, monkeypatch):
        """候选空 ⇒ 宁可如实记"确实没动作"，也不编一个假动作充数。"""
        monkeypatch.setattr(R, "_today_candidates", lambda day, limit=8: [])
        fake = _SeqClient([self.PASS, self.PASS])
        out = self._run(monkeypatch, fake)
        assert "auto_watch" not in out.get("enforced", ""), out
        assert out["kept"] == 0, out

    def test_fallback_never_places_order(self, tpl_db, monkeypatch):
        """红线：兜底只维护池子，**绝不下单**。

        为了"动作数不为 0"而自动买入，是拿账户纪律换页面好看。
        """
        monkeypatch.setattr(R, "_today_candidates",
                            lambda day, limit=8: [{"code": "000001", "name": "候选股"}])
        fake = _SeqClient([self.PASS, self.PASS])
        out = self._run(monkeypatch, fake, execute=True)
        for r in out.get("orders", []):
            assert r.get("action") != "buy", r
            assert "order_id" not in r, r
        from db import store
        st = store.load_paper_state("ai_ashen")
        assert (st or {}).get("fills") in (None, []), st

    def test_normal_reply_leaves_enforced_empty(self, tpl_db, monkeypatch):
        """空 `enforced` = 它自己就给了实质动作；非空 = 系统推过。复盘靠这个区分。"""
        fake = _SeqClient([self.WAKE])
        self._run(monkeypatch, fake)
        from db import store
        rows = store.list_routine_runs("2026-10-27", persona="ai_ashen")
        assert (rows[0].get("enforced") or "") == "", rows[0]

    def test_prompt_contains_three_tier_plan(self):
        """三层价位（首笔/加仓/放弃位）必须写进 prompt —— 单条"跌 2% 才买"会永不触发。"""
        txt = R.build_prompt(
            now=datetime(2026, 10, 28, 10, 20), slot="10:20", level="L2",
            auton_desc="可建仓", budget_desc="单票 ≤8%",
            cash=1_000_000, nav=1_000_000, holdings=[], my_watch=[], user_watch=[],
            quotes={}, candidates=[], trail=[], armed=[])
        assert "三层价位" in txt, txt
        assert "每轮至少要有一件实质动作" in txt, txt
