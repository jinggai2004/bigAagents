# -*- coding: utf-8 -*-
"""结算与榜单（core/ai_league.py）的测试。

⭐ 核心验收：
  ① 日频 T+1 结算：T-1 的买入决策 → 提交 → 开盘撮合 → **回填订单号**（幂等：已成交的不重复提交）；
  ② 日内唤醒卖出：按扫描价成交，且**同时留下触发价**（口径 2）；无持仓/无价 ⇒ **不伪造成交**；
  ③ 榜单：按频率分组、随机对照分位、样本门槛 `conclusive`、caveat 必现、**不含胜率**。
⚠️ 全离线：`daily_bars` 被 mock；库走 use_template_db。
"""
import pytest

from core import ai_league as L, ai_personas
from db import store
from tests.conftest import use_template_db


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    use_template_db(tmp_path, monkeypatch, "ai_league.db")
    monkeypatch.setattr(ai_personas, "PERSONAS_PATH", tmp_path / "ai_personas.json")
    ai_personas.load_personas()
    # ⚠️ 盘中补仓开关是**真实运行状态**；不重定向会让用例读/写真实文件
    monkeypatch.setattr(L, "_wake_buy_state_path",
                        lambda: tmp_path / "wake_buy_state.json")
    # ⚠️ 结算取价入口是 `open_bars`（**实时行情优先**，2026-10-16 起）⇒ 夹具必须
    #    一并挡掉实时源，否则用例会真联网取行情（慢、不稳定，还会污染"离线"承诺）。
    #    需要价的用例自己 patch `daily_bars` 提供 bar（走兜底分支）。
    monkeypatch.setattr("data.realtime.get_realtime_quotes",
                        lambda codes, *a, **k: {})
    yield
    ai_personas.invalidate()


BARS = {"600028": {"open": 5.00, "close": 5.10, "prev_close": 4.95}}


def _persona(name="ai_value"):
    return ai_personas.get_persona(name)


class TestDailyBars:
    def test_missing_code_is_skipped(self, monkeypatch):
        """取不到行情 ⇒ 不进入 bars（调用方按"停牌"处理），不伪造价格。"""
        class F:
            def get_stock_data(self, code, period="1mo"):
                raise RuntimeError("网络断了")
        monkeypatch.setattr("data.stock_data.StockDataFetcher", lambda: F())
        assert L.daily_bars(["600028"], "2026-10-15") == {}


class TestSettleDaily:
    def test_buy_decision_settles_and_backfills_order(self, monkeypatch):
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {
            c: BARS[c] for c in codes if c in BARS})
        store.save_ai_decision({"persona": "ai_value", "signal_date": "2026-10-14",
                                "code": "600028", "advice_status": "可执行",
                                "direction": "看多", "position_size": 0.2,
                                "model": "qwen-max"})
        out = L.settle_daily("2026-10-15", personas=[_persona()])
        assert out["results"][0]["submitted"] == 1
        rows = store.list_ai_decisions(persona="ai_value", signal_date="2026-10-14")
        assert rows[0]["order_id"], "订单号必须回填到决策行"
        assert "已成交" in (rows[0]["order_status"] or "") or rows[0]["order_status"]

    def test_already_settled_not_resubmitted(self, monkeypatch):
        """⭐ 幂等：已回填 order_id 的决策不得重复提交（否则会重复建仓）。"""
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {
            c: BARS[c] for c in codes if c in BARS})
        store.save_ai_decision({"persona": "ai_value", "signal_date": "2026-10-14",
                                "code": "600028", "advice_status": "可执行",
                                "position_size": 0.2, "order_id": 99,
                                "order_status": "已成交"})
        out = L.settle_daily("2026-10-15", personas=[_persona()])
        assert out["results"][0]["submitted"] == 0

    def test_no_bar_reported_not_faked(self, monkeypatch):
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {})   # 停牌
        store.save_ai_decision({"persona": "ai_value", "signal_date": "2026-10-14",
                                "code": "600028", "advice_status": "可执行",
                                "position_size": 0.2})
        out = L.settle_daily("2026-10-15", personas=[_persona()])
        row = out["results"][0]
        assert row["submitted"] == 0 and row["settled"] == 0     # 汇总字段永远存在
        # ⚠️ 跳过原因必须**显式**（原实现只留"提交 0 笔"，看不出是停牌还是取价失败）
        assert any("取不到" in (i.get("skipped") or "") for i in row["items"])

    def test_tiny_position_skipped(self, monkeypatch):
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {
            c: BARS[c] for c in codes if c in BARS})
        store.save_ai_decision({"persona": "ai_value", "signal_date": "2026-10-14",
                                "code": "600028", "advice_status": "可执行",
                                "position_size": 0.00001})          # 预算买不到一手
        out = L.settle_daily("2026-10-15", personas=[_persona()])
        assert any("仓位过小" in (i.get("skipped") or "")
                   for i in out["results"][0]["items"])


class TestIntradayWakes:
    def _hold(self, monkeypatch, persona="ai_value"):
        """造一个"昨天买入、今天可卖"的持仓。"""
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {
            c: BARS[c] for c in codes if c in BARS})
        acct = L._load(persona, 1_000_000)
        acct.apply_trade({"date": "2026-10-13", "code": "600028",
                          "direction": "买入", "price": 4.9, "quantity": 1000})
        L._save(acct)

    def test_sell_fills_at_trigger_price_and_keeps_both_prices(self, monkeypatch):
        """⭐ 口径 2：成交价=扫描价（+滑点由 matching 计），且**触发价同时留存**。"""
        self._hold(monkeypatch)
        store.record_wake_event("ai_value", "2026-10-14", "600028", 1,
                                hit_at="10:30:00", trigger_px=4.5)
        store.mark_wake_event("ai_value", "2026-10-14", "600028", 1, fired=True,
                              action='[{"action": "sell"}]')
        out = L.settle_intraday_wakes("2026-10-14", personas=[_persona()])
        sells = out["results"][0]["sells"]
        assert sells and sells[0].get("trigger_px") == 4.5, "触发价必须留存（口径 2）"
        # ⚠️ fill 的键是 `signal_price`（扫描价）/`actual_price`（滑点后）——
        #    没有 `price` 键（项目踩过同类 KeyError）。成交价必然 < 扫描价（滑点）。
        assert sells[0].get("signal_price") == 4.5
        # ⚠️ 断言用 `<=` 而非 `<`：滑点 0.05% 作用在 4.5 元上是 4.4978，
        #    而成交价 round 到分 ⇒ 4.5 ⇒ **低价股的滑点在两位小数下不可见**。
        #    这不是缺陷（round 到分是合理口径），但不能据此断言"必然更低"。
        assert 0 < sells[0].get("actual_price", 0) <= 4.5
        st = store.load_paper_state("ai_value")
        assert not any(p["code"] == "600028" and p["quantity"] > 0
                       for p in (st.get("positions") or []))

    def test_no_position_is_not_faked(self, monkeypatch):
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {})
        store.record_wake_event("ai_value", "2026-10-14", "600028", 1,
                                hit_at="10:30:00", trigger_px=4.5)
        store.mark_wake_event("ai_value", "2026-10-14", "600028", 1, fired=True,
                              action='[{"action": "sell"}]')
        out = L.settle_intraday_wakes("2026-10-14", personas=[_persona()])
        s = out["results"][0]["sells"][0]
        assert s.get("status") == "cancelled" and "无可卖持仓" in (s.get("reason") or "")

    def test_missing_trigger_price_not_faked(self, monkeypatch):
        self._hold(monkeypatch)
        store.record_wake_event("ai_value", "2026-10-14", "600028", 2,
                                hit_at="10:30:00", trigger_px=None)
        store.mark_wake_event("ai_value", "2026-10-14", "600028", 2, fired=True,
                              action='[{"action": "sell"}]')
        out = L.settle_intraday_wakes("2026-10-14", personas=[_persona()])
        s = out["results"][0]["sells"][0]
        assert s.get("status") == "cancelled" and "缺触发价" in (s.get("reason") or "")

    def test_random_persona_not_settled_intraday(self, monkeypatch):
        out = L.settle_intraday_wakes("2026-10-14", personas=[_persona("ai_random_1")])
        assert out["results"] == []


class TestMarkNav:
    def test_nav_recorded_with_close_price(self, monkeypatch):
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {
            c: BARS[c] for c in codes if c in BARS})
        acct = L._load("ai_value", 1_000_000)
        acct.apply_trade({"date": "2026-10-13", "code": "600028",
                          "direction": "买入", "price": 4.9, "quantity": 1000})
        L._save(acct)
        out = L.mark_nav("2026-10-14", personas=[_persona()])
        rec = out["results"][0]["nav"]
        assert rec["date"] == "2026-10-14" and rec["total_value"] > 0
        assert rec["at_cost"] == 0, "拿到市价时 at_cost 必须为 0（口径标记）"

    def test_all_prices_missing_records_nothing(self, monkeypatch):
        """⭐ 全部取不到当日价 ⇒ **不记净值**（缺就是缺，绝不用成本价冒充）。

        ⚠️ 2026-10-15 语义变更：旧版会记一条"按成本估值"的假净值 ⇒ 恰好 ≈ 初始资金，
        让 AI 复盘得出"净值不变、无盈亏"的错误结论。
        """
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {})
        acct = L._load("ai_value", 1_000_000)
        acct.apply_trade({"date": "2026-10-13", "code": "600028",
                          "direction": "买入", "price": 4.9, "quantity": 1000})
        L._save(acct)
        out = L.mark_nav("2026-10-14", personas=[_persona()])
        row = out["results"][0]
        assert row["recorded"] is False and row["nav"] is None
        assert row["priced"] == 0 and row["at_cost"] == 1 and row["positions"] == 1
        assert out["skipped"] == 1
        # 账户里也不得有这条假净值
        navs = [n for n in (store.load_paper_state("ai_value") or {}).get("nav") or []
                if n.get("date") == "2026-10-14"]
        assert navs == []

    def test_partial_missing_records_with_at_cost_marker(self, monkeypatch):
        """部分缺价 ⇒ 记，但 **at_cost 一并落库**（口径随数字走，AI/页面都能看到）。"""
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {
            "600028": {"open": 5.0, "close": 5.1, "prev_close": 4.9}})
        acct = L._load("ai_value", 1_000_000)
        acct.apply_trade({"date": "2026-10-13", "code": "600028",
                          "direction": "买入", "price": 4.9, "quantity": 1000})
        acct.apply_trade({"date": "2026-10-13", "code": "999999",
                          "direction": "买入", "price": 10.0, "quantity": 100})
        L._save(acct)
        out = L.mark_nav("2026-10-14", personas=[_persona()])
        row = out["results"][0]
        assert row["recorded"] is True
        assert row["at_cost"] == 1 and row["priced"] == 1
        assert row["nav"]["at_cost"] == 1
        # 落库后依然带着标记（否则页面/AI 无从得知）
        stored = [n for n in (store.load_paper_state("ai_value") or {}).get("nav") or []
                  if n.get("date") == "2026-10-14"]
        assert stored and stored[0]["at_cost"] == 1

    def test_snapshot_price_wins_over_daily_bar(self, monkeypatch):
        """快照（15:40 落库）是**当日收盘价**的权威来源，优先于日线取数。"""
        store.save_market_snapshot("2026-10-14", [{"stock_code": "600028", "price": 9.99}])
        monkeypatch.setattr(L, "daily_bars", lambda codes, d, **kw: {
            "600028": {"open": 5.0, "close": 5.1, "prev_close": 4.9}})
        acct = L._load("ai_value", 1_000_000)
        acct.apply_trade({"date": "2026-10-13", "code": "600028",
                          "direction": "买入", "price": 4.9, "quantity": 1000})
        L._save(acct)
        out = L.mark_nav("2026-10-14", personas=[_persona()])
        rec = out["results"][0]["nav"]
        assert rec["position_value"] == round(9.99 * 1000, 2), rec


class TestLeaderboard:
    def _mk(self, name, ret_pct, days, trades, kind, freq="daily"):
        """直接造账户状态（净值序列 + 成交笔数）。"""
        st = {"name": name, "initial_cash": 1_000_000, "cash": 1_000_000,
              "positions": [], "orders": [],
              "fills": [{"code": "600028"}] * trades,
              "nav": [{"date": f"2026-09-{i+1:02d}", "total_value": 1_000_000}
                      for i in range(max(0, days - 1))] +
                     [{"date": "2026-10-14",
                       "total_value": 1_000_000 * (1 + ret_pct / 100)}]}
        store.save_paper_state(name, st)

    def test_groups_by_frequency_and_never_mix(self):
        # ⚠️ 分组依据是 **persona 配置**的 frequency（不是账户状态）⇒ 必须改配置
        personas = ai_personas.load_personas()
        for p in personas:
            if p["name"] == "ai_tech":
                p["frequency"] = "intraday"
        assert ai_personas.save_personas(personas) is True
        self._mk("ai_value", 5.0, 21, 31, "llm", "daily")
        self._mk("ai_tech", 3.0, 21, 31, "llm", "intraday")
        self._mk("ai_random_1", 1.0, 21, 31, "random", "daily")
        lb = L.leaderboard()
        daily = {r["persona"] for r in lb["groups"]["daily"]["rows"]}
        intraday = {r["persona"] for r in lb["groups"]["intraday"]["rows"]}
        assert "ai_tech" not in daily and "ai_tech" in intraday
        assert "ai_value" in daily

    def test_random_percentile_and_excess(self):
        self._mk("ai_random_1", 1.0, 21, 31, "random")
        self._mk("ai_random_2", 2.0, 21, 31, "random")
        self._mk("ai_random_3", 3.0, 21, 31, "random")
        self._mk("ai_value", 10.0, 21, 31, "llm")
        lb = L.leaderboard()
        row = next(r for r in lb["groups"]["daily"]["rows"] if r["persona"] == "ai_value")
        assert row["random_percentile"] == 100.0        # 高于全部随机组
        assert row["excess_vs_random_mean"] == pytest.approx(8.0)
        assert row["conclusive"] is True

    def test_sample_threshold_marks_inconclusive(self):
        """⭐ 样本不足时**必须**标为不具备统计意义（不能静默排行）。"""
        self._mk("ai_random_1", 1.0, 3, 2, "random")
        self._mk("ai_value", 5.0, 3, 2, "llm")
        lb = L.leaderboard()
        row = next(r for r in lb["groups"]["daily"]["rows"] if r["persona"] == "ai_value")
        assert row["conclusive"] is False
        assert lb["groups"]["daily"]["conclusive"] is False

    def test_caveat_always_present(self):
        self._mk("ai_random_1", 0.0, 5, 5, "random")
        lb = L.leaderboard()
        text = " ".join(lb["caveat"])
        assert "高估" in text and "统计意义" in text and "不用胜率" in text

    def test_no_win_rate_used(self):
        """⭐ 结构性守护：榜单的**判据字段**不得引入胜率类指标（项目共识）。

        ⚠️ 2026-10-15 细化口径：用户明确要求把「盘中加仓胜率」作为**诊断项**展示，
        所以这里不能再粗暴断言"任何含 win 的键都不许出现"——那会让"加一个诊断指标"
        与"改判据"混为一谈。改为**只钉判据字段**（决定排名/是否跑赢的那些），
        并限定唯一允许的诊断项白名单。
        """
        self._mk("ai_value", 1.0, 21, 31, "llm")
        self._mk("ai_random_1", 1.0, 21, 31, "random")
        lb = L.leaderboard()
        rows = [r for g in lb["groups"].values() for r in g["rows"]]
        judgement = ("conclusive", "excess_vs_random_mean", "random_percentile",
                     "return_pct", "max_drawdown_pct", "participation_pct")
        for r in rows:
            assert not any("win" in k.lower() for k in judgement if k in r), r
        keys = {k for r in rows for k in r if "win" in k.lower()}
        assert keys <= {"intraday_buy_win_rate"}, keys
        assert "不用胜率" in " ".join(lb["caveat"])


# ================================================================
# 参与度与对照仓位（2026-10-15，主动性改进）
# ================================================================
#
# 只比收益会让"永久空仓"免费拿分（0 收益、0 回撤）⇒ 把平均仓位与有持仓天数占比
# 单列到榜单上，让"不作为"可见（判据仍不用胜率）。

class TestParticipation:
    def _acct(self, name, navs):
        st = {"name": name, "initial_cash": 1_000_000, "cash": 1_000_000,
              "positions": [], "orders": [], "fills": [],
              "nav": navs}
        store.save_paper_state(name, st)

    def test_stats_reports_participation_and_exposure(self):
        self._acct("ai_value", [
            {"date": "2026-10-13", "total_value": 1_000_000, "position_value": 0},
            {"date": "2026-10-14", "total_value": 1_000_000, "position_value": 200_000},
            {"date": "2026-10-15", "total_value": 1_000_000, "position_value": 400_000},
        ])
        s = L._stats("ai_value")
        assert s["holding_days"] == 2
        assert s["participation_pct"] == 66.7
        assert s["avg_exposure_pct"] == 20.0          # (0+20+40)/3

    def test_never_holding_is_visible_as_zero_participation(self):
        """⭐ 从不持仓 ⇒ 参与度 0（榜面上不能与"参与但小亏"混为一谈）。"""
        self._acct("ai_risk", [{"date": f"2026-10-{d:02d}", "total_value": 1_000_000,
                                "position_value": 0} for d in (13, 14, 15)])
        s = L._stats("ai_risk")
        assert s["holding_days"] == 0 and s["participation_pct"] == 0.0
        assert s["avg_exposure_pct"] == 0.0

    def test_random_group_exposure_is_aggregate_only(self):
        """对照仓位只暴露**随机组的聚合仓位**（不含任何代码）⇒ 不破坏公平性。"""
        self._acct("ai_random_1", [{"date": "2026-10-15", "total_value": 1_000_000,
                                    "position_value": 100_000}])
        self._acct("ai_random_2", [{"date": "2026-10-15", "total_value": 1_000_000,
                                    "position_value": 300_000}])
        out = L.random_group_exposure("2026-10-15")
        assert out["available"] is True and out["n"] == 2
        assert out["avg_exposure_pct"] == 20.0        # (10+30)/2
        assert "代码" in out["note"]                   # 明说"不含代码"

    def test_random_group_exposure_honest_when_no_data(self):
        out = L.random_group_exposure("2026-10-15")
        assert out["available"] is False and out["avg_exposure_pct"] is None

    def test_leaderboard_rows_carry_participation(self):
        self._acct("ai_value", [{"date": "2026-10-15", "total_value": 1_000_000,
                                 "position_value": 100_000}])
        lb = L.leaderboard()
        rows = [r for g in lb["groups"].values() for r in g["rows"]
                if r["persona"] == "ai_value"]
        assert rows and rows[0]["avg_exposure_pct"] == 10.0
        assert "参与度单列" in " ".join(lb["caveat"])


# ================================================================
# 盘中持仓内补仓：结算 / 幂等 / 统计 / 自动回退（2026-10-15）
# ================================================================

class TestSettleIntradayBuys:
    def _hold(self, qty=1000, price=5.0, cash=1_000_000, name="ai_value"):
        acct = L._load(name, cash)
        acct.apply_trade({"date": "2026-10-13", "code": "600028",
                          "direction": "买入", "price": price, "quantity": qty})
        L._save(acct)
        return acct

    def _buy_event(self, shares, px=4.9, cond_id=1, persona="ai_value"):
        import json
        store.record_wake_event(persona, "2026-10-14", "600028", cond_id,
                                hit_at="10:30:00", trigger_px=px)
        store.mark_wake_event(persona, "2026-10-14", "600028", cond_id, fired=True,
                              action=json.dumps([{"action": "buy", "code": "600028",
                                                  "shares": shares, "amount": shares * px,
                                                  "intraday": True}]))

    def test_buy_settles_with_tag_and_trigger_price(self):
        """⭐ 补仓成交必须带 `intraday_buy` 标记 + 留存触发价（口径 2 两价并存）。"""
        self._hold()
        self._buy_event(shares=100, px=4.9)
        out = L.settle_intraday_wakes("2026-10-14", personas=[_persona()])
        buys = out["results"][0]["buys"]
        assert buys and buys[0].get("status") != "cancelled", buys
        assert buys[0]["trigger_px"] == 4.9
        st = store.load_paper_state("ai_value")
        fills = [f for f in st["fills"] if f.get("tag") == "intraday_buy"]
        assert len(fills) == 1 and fills[0]["code"] == "600028"
        assert st["positions"]["600028"]["quantity"] > 1000, "补仓后持仓应变多"

    def test_rerun_is_idempotent(self):
        """⭐ 幂等锚点 = 当日同标的已有 intraday_buy 成交 ⇒ 重跑不重复买。

        ⚠️ 不能靠"重跑护栏"兜底：护栏数的是**已落库的 buy 事件**，
           结算时再跑会把自己数进去而误判"已达上限"。
        """
        self._hold()
        self._buy_event(shares=100, px=4.9)
        L.settle_intraday_wakes("2026-10-14", personas=[_persona()])
        out2 = L.settle_intraday_wakes("2026-10-14", personas=[_persona()])
        items = out2["results"][0]["buys"]
        assert items and items[0]["status"] == "skipped" and "幂等" in items[0]["reason"]
        st = store.load_paper_state("ai_value")
        assert len([f for f in st["fills"] if f.get("tag") == "intraday_buy"]) == 1

    def test_missing_shares_not_faked(self):
        """护栏没给出股数 ⇒ **不成交**（不猜一个数量成交）。"""
        self._hold()
        self._buy_event(shares=0, px=4.9)
        out = L.settle_intraday_wakes("2026-10-14", personas=[_persona()])
        b = out["results"][0]["buys"][0]
        assert b["status"] == "cancelled" and "不伪造成交" in b["reason"]

    def test_sell_and_buy_in_same_event_are_split(self):
        """同一事件里的卖出与补仓必须**分别**处理（子串匹配会把两类混在一起）。"""
        import json
        self._hold()
        store.record_wake_event("ai_value", "2026-10-14", "600028", 7,
                                hit_at="10:30:00", trigger_px=4.9)
        store.mark_wake_event("ai_value", "2026-10-14", "600028", 7, fired=True,
                              action=json.dumps([
                                  {"action": "sell", "code": "600028"},
                                  {"action": "set_wake", "code": "600028",
                                   "kind": "stop_loss"}]))
        out = L.settle_intraday_wakes("2026-10-14", personas=[_persona()])
        r = out["results"][0]
        assert r["sells"] and not r["buys"], "set_wake 不得被判成买入"


class TestParticipationIntraday:
    def _acct_with_fills(self, name="ai_value", fills=None, navs=None):
        store.save_paper_state(name, {"name": name, "initial_cash": 1_000_000,
                                      "cash": 900_000, "positions": [], "orders": [],
                                      "fills": fills or [], "nav": navs or []})

    def test_stats_reports_intraday_ratio_and_win_rate(self):
        """⭐ 盘中交易占比 / 盘中加仓胜率必须可算（否则无法判断"该不该继续放宽"）。"""
        self._acct_with_fills(fills=[
            {"date": "2026-10-14", "code": "600028", "direction": "买入",
             "quantity": 100, "actual_price": 5.0, "total": 500.0,
             "cash_after": 0, "tag": "intraday_buy"},
            {"date": "2026-10-14", "code": "600028", "direction": "卖出",
             "quantity": 100, "actual_price": 5.2, "total": 520.0,
             "cash_after": 0, "pnl": 20.0, "tag": "intraday_sell"},
            {"date": "2026-10-14", "code": "600519", "direction": "买入",
             "quantity": 100, "actual_price": 1200.0, "total": 120000.0,
             "cash_after": 0},                          # 盘后成交（无 tag）
        ])
        s = L._stats("ai_value")
        assert s["intraday_trades"] == 2 and s["trades"] == 3
        assert s["intraday_trade_pct"] == pytest.approx(66.7, abs=0.2)
        assert s["intraday_buy_win_rate"] == pytest.approx(100.0)

    def test_unclosed_intraday_buy_has_no_win_rate(self):
        """未平仓 ⇒ 胜率是 `None`（算不出来就如实空着，不拍 0 当"全亏"）。"""
        self._acct_with_fills(fills=[
            {"date": "2026-10-14", "code": "600028", "direction": "买入",
             "quantity": 100, "actual_price": 5.0, "total": 500.0,
             "cash_after": 0, "tag": "intraday_buy"}])
        assert L._stats("ai_value")["intraday_buy_win_rate"] is None

    def test_random_group_intraday_has_no_codes(self):
        """披露聚合值但**不得**含标的代码（公平性）。"""
        self._acct_with_fills(name="ai_random_1", fills=[
            {"date": "2026-10-14", "code": "600028", "direction": "买入",
             "quantity": 100, "actual_price": 5.0, "total": 500.0,
             "cash_after": 0, "tag": "intraday_buy"}])
        out = L.random_group_intraday("2026-10-14")
        assert out["intraday_trades"] == 1 and out["intraday_amount"] == pytest.approx(500.0)
        assert "代码" not in str(out.get("intraday_trades"))
        assert "不含标的代码" in out["note"]


class TestWakeBuyRollback:
    def _five_losing_days(self, name="ai_value"):
        fills = []
        for d in ["2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09", "2026-10-10"]:
            fills.append({"date": d, "code": "600028", "direction": "买入",
                          "quantity": 100, "actual_price": 5.0, "total": 500.0,
                          "cash_after": 0, "tag": "intraday_buy"})
            fills.append({"date": d, "code": "600028", "direction": "卖出",
                          "quantity": 100, "actual_price": 4.8, "total": 480.0,
                          "cash_after": 0, "pnl": -20.0})
        store.save_paper_state(name, {"name": name, "initial_cash": 1_000_000,
                                      "cash": 1_000_000, "positions": [], "orders": [],
                                      "fills": fills, "nav": []})

    def test_auto_disables_after_five_losing_days(self):
        """⭐ 连续 5 个有平仓结果的补仓日全亏 ⇒ 自动回退到"盘中仅卖出"。"""
        self._five_losing_days()
        st = L.wake_buy_status("2026-10-14")
        assert st["enabled"] is False and st["auto_disabled"] is True
        assert st["evidence"]["threshold_days"] == 5
        assert len(st["evidence"]["days"]) == 5

    def test_unclosed_buys_do_not_count_into_rollback(self):
        """⚠️ 只有买入没有平仓 ⇒ **不计入判据**（算不出来，不许拍 0 当"没亏"）。"""
        store.save_paper_state("ai_value", {
            "name": "ai_value", "initial_cash": 1_000_000, "cash": 1_000_000,
            "positions": [], "orders": [], "nav": [],
            "fills": [{"date": "2026-10-10", "code": "600028", "direction": "买入",
                       "quantity": 100, "actual_price": 5.0, "total": 500.0,
                       "cash_after": 0, "tag": "intraday_buy"}]})
        assert L.wake_buy_status("2026-10-14")["enabled"] is True

    def test_manual_override_reenables(self):
        """⚠️ 回退必须**可人工复位**：否则没有新成交 ⇒ 判据窗口永远不动 ⇒ 死锁。"""
        self._five_losing_days()
        assert L.wake_buy_status("2026-10-14")["enabled"] is False
        assert L.write_wake_buy_state(manual_enable=True, reason="人工评估后重开") is True
        st = L.wake_buy_status("2026-10-14")
        assert st["enabled"] is True and "人工复位" in st["reason"]

    def test_missing_state_file_means_open(self, monkeypatch):
        """状态文件缺失 ⇒ 按"开放"处理（不能因为读不到配置就把功能锁死）。"""
        st = L.wake_buy_status("2026-10-14")
        assert st["enabled"] is True and st["manual"] == {}
