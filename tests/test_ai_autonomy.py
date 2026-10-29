# -*- coding: utf-8 -*-
"""自营授权参数（core/ai_autonomy.py）的自洽校验测试。

## 为什么这个文件值得单独存在

原方案写了「总仓 20% / 单票 8% / 现金地板 20%」三个数字，两两看都合理，
放在一起机器一算就矛盾：总仓 ≤20% ⇒ 现金恒 ≥80% ⇒ **地板 20% 永不触发**；
8%×3 只 = 24% > 20% ⇒ **实际最多只能持 2 只**。

"再看一遍"发现不了这类错，只能算。所以这个文件的重点不是"默认参数对不对"，
而是**每一条违规都必须能被检出**——如果某条断言只能由模块自己的派生逻辑决定
（例如 `cash_floor == 1 - max_total`），那它永远是绿的，写了等于没写。
"""

import pytest

from core import ai_autonomy as A


class TestDefaults:
    def test_default_params_are_self_consistent(self):
        """默认参数必须通过校验（否则每次加载配置都抛异常）。"""
        p = A.resolve()
        assert A.validate(p) == [], A.validate(p)

    def test_cash_floor_is_derived_not_configured(self):
        """现金地板**派生**自总仓上限，传什么都会被覆盖。

        ⚠️ 允许单独配置地板，就一定会再出现"地板 > 1 − 总仓"的死条款，
        而它不报错、只是那行 `if` 永远走不到。
        """
        p = A.resolve({"max_total_pct": 0.30, "cash_floor_pct": 0.95})
        assert p["max_total_pct"] == 0.30
        assert abs(p["cash_floor_pct"] - 0.70) < 1e-9, "地板必须是 1 − 总仓上限"

    def test_lots_per_position_derived(self):
        """首笔 3% / 单票 8% ⇒ 3 笔建满（向上取整，不是四舍五入）。"""
        assert A.resolve()["lots_per_position"] == 3

    def test_cash_floor_disabled_when_total_is_full(self):
        """总仓 100% ⇒ 现金地板 0 ⇒ 必须**显式**标记为"已关闭"。

        ⚠️ 这是本次把总仓从 24% 提到 100% 带出来的副作用。地板 0 意味着
        「现金低于地板不许开仓」那条判据**永久不触发**——如果只是让它变成 0
        而不标记，它就是第二个"死条款"：纸面上有风控，实际那行 `if` 走不到。
        """
        p = A.resolve()
        assert p["cash_floor_pct"] == 0
        assert p["has_cash_floor"] is False
        # 反过来：只要留了地板，就必须能被下游看见
        p2 = A.resolve({"max_total_pct": 0.5})
        assert p2["cash_floor_pct"] == 0.5 and p2["has_cash_floor"] is True

    def test_total_cap_is_actually_usable(self):
        """总仓上限必须"用得完"—— 这是"守卫不可恒真"的**反向**应用。

        给了 100% 却只许 3 只 × 8%（= 24%），那个 100% 永远触不到顶。
        默认参数下 12 只 × 8% = 96%，留 4% 给"最后一只没建满"的常态。
        """
        p = A.resolve()
        usable = p["max_stocks"] * p["max_position_pct"]
        assert usable >= p["max_total_pct"] * A.MIN_USABLE_TOTAL_RATIO
        assert usable <= p["max_total_pct"] + 1e-9, "但不能反过来超过总仓"

    def test_total_cap_unusable_is_rejected(self):
        """把只数改回 3 ⇒ 必须报错（否则总仓 100% 是纸面权限）。"""
        errs = A.validate(A.resolve({"max_stocks": 3}))
        assert any("永远触不到顶" in e for e in errs), errs

    def test_unknown_override_keys_ignored(self):
        p = A.resolve({"不存在的参数": 1, "max_position_pct": 0.05})
        assert "不存在的参数" not in p
        assert p["max_position_pct"] == 0.05


class TestValidateCatchesViolations:
    """每一类违规都必须**真的被判出**（防恒真守卫）。"""

    def _bad(self, **over):
        p = A.resolve(over)
        return A.validate(p)

    def test_position_exceeds_total(self):
        errs = self._bad(max_position_pct=0.30, max_total_pct=0.20)
        assert any("大于总仓上限" in e for e in errs), errs

    def test_first_buy_exceeds_position(self):
        errs = self._bad(first_buy_pct=0.10, max_position_pct=0.08, max_total_pct=0.24)
        assert any("首笔" in e for e in errs), errs

    def test_max_stocks_times_position_exceeds_total(self):
        """原方案那个 bug 的机器判据：8% × 3 只 = 24% > 总仓上限。

        ⚠️ 用**显式传入**的 max_stocks 触发，而不是靠 resolve 派生——
        否则断言会跟着实现一起对，永远绿。
        """
        errs = self._bad(max_position_pct=0.08, max_total_pct=0.20, max_stocks=3)
        assert any("超过总仓上限" in e for e in errs), errs

    def test_cash_floor_above_implicit_is_rejected(self):
        """地板高于 1 − 总仓上限 ⇒ 死条款，必须报错。

        注意 resolve() 会覆盖 cash_floor，所以要**直接构造 dict** 传给
        validate()——这正是"校验接受任意来源参数"的用处。
        """
        p = A.resolve()
        p["cash_floor_pct"] = 0.95          # 手改成不可能的 95%
        errs = A.validate(p)
        assert any("永远不会触发" in e for e in errs), errs

    def test_single_lot_is_rejected(self):
        p = A.resolve({"first_buy_pct": 0.08})     # 首笔 = 单票上限 ⇒ 一笔打满
        errs = A.validate(p)
        assert any("一次打满" in e for e in errs), errs

    def test_stop_loss_too_tight_is_rejected(self):
        errs = self._bad(stop_loss_atr_mult=1.0)
        assert any("低于硬下限" in e for e in errs), errs

    def test_out_of_range_percent(self):
        errs = self._bad(max_total_pct=1.5)
        assert any("(0, 1]" in e for e in errs), errs

    def test_zero_and_negative_rejected(self):
        for bad in (0, -0.1):
            errs = self._bad(max_total_pct=bad)
            assert errs, f"max_total_pct={bad} 应被拒"

    def test_missing_key_reported(self):
        errs = A.validate({"max_total_pct": 0.24})
        assert any("缺失" in e for e in errs), errs

    def test_zero_stocks_rejected(self):
        errs = self._bad(max_stocks=0)
        assert any("必须 ≥ 1" in e for e in errs), errs

    def test_time_stop_below_one_day(self):
        errs = self._bad(time_stop_days=0)
        assert any("time_stop_days" in e for e in errs), errs


class TestCheckOrRaise:
    def test_passes_for_default(self):
        # 2026-10-27 老板要求「建仓权开 + 总仓给 100%」，默认总仓已从 24% 提到 100%。
        assert A.check_or_raise(A.resolve())["max_total_pct"] == 1.0

    def test_raises_with_all_violations_listed(self):
        p = A.resolve()
        p["max_position_pct"] = 0.50
        p["max_stocks"] = 5
        with pytest.raises(ValueError) as ei:
            A.check_or_raise(p)
        msg = str(ei.value)
        assert "自营授权参数不自洽" in msg
        assert "超过总仓上限" in msg


class TestLevels:
    def test_levels_cover_all_autonomy_tiers(self):
        assert A.LEVELS == ("L0", "L1", "L2", "L3")
        for lv in A.LEVELS:
            assert lv in A.LEVEL_DESC

    def test_normalize_falls_back_to_most_conservative(self):
        """非法级别回退 L0，**不抛异常**。

        理由：这是权限。读错时安全方向只有一种——更保守。抛异常会让整轮
        决策挂掉（连"今天不动"都记不下来），回退 L0 只是"这轮不动"。
        """
        for junk in ("", None, "L9", "l1x", 123, "L1 "):
            got = A.normalize_level(junk)
            assert got in A.LEVELS
        assert A.normalize_level("") == "L0"
        assert A.normalize_level(None) == "L0"
        assert A.normalize_level("L9") == "L0"
        assert A.normalize_level(" l2 ") == "L2", "大小写与空白应能正常识别"


# ================================================================
# 建仓权判据（autonomy_check）—— 「建仓权开」到底开在哪里，靠这些用例钉死
# ================================================================
class TestAutonomyCheck:
    Q = {"price": 10.0, "prev_close": 10.0, "change_pct": -1.0,
         "high": 10.2, "low": 9.8, "volume_hand": 120000.0,
         "amount_wan": 80000.0, "name": "测试股份"}

    def _q(self, **over):
        return {**self.Q, **over}

    def _port(self, cash=1_000_000.0, nav=1_000_000.0, holdings=None):
        return {"cash": cash, "nav": nav, "holdings": holdings or {}}

    def _buy(self, code="600519", size=0.03, limit=10.0, stop=9.5, reason="理由"):
        return {"action": "buy", "code": code, "position_size": size,
                "limit_price": limit, "stop_loss": stop, "reason": reason}

    def _run(self, actions, *, level="L2", params=None, portfolio=None,
             my_watch=None, quotes=None, user_watch=None):
        return A.autonomy_check(
            actions, level=level, params=params or A.resolve(),
            portfolio=portfolio if portfolio is not None else self._port(),
            my_watch={"600519", "000858"} if my_watch is None else my_watch,
            quotes=quotes if quotes is not None else {"600519": self._q(),
                                                      "000858": self._q()},
            user_watch=user_watch)

    # ── 级别 ────────────────────────────────────────────────
    def test_L1_has_no_build_right(self):
        """L1 是"存量经营"：**不开新仓**。这条是建仓权的分界线。"""
        out = self._run([self._buy()], level="L1")
        assert not [k for k in out["kept"] if k["action"] == "buy"]
        assert any("需 L2" in v for v in out["violations"]), out["violations"]

    def test_L2_buys_inside_pool(self):
        out = self._run([self._buy()], level="L2")
        assert [k for k in out["kept"] if k["action"] == "buy"], out["blocked"]
        assert out["blocked"] == []

    def test_illegal_level_falls_back_and_blocks(self):
        """L9 不是"更高级"，是非法值 ⇒ 回退 L0 ⇒ 连卖出都不许。"""
        out = self._run([self._buy()], level="L9")
        assert out["level"] == "L0"
        assert out["blocked"]

    # ── 池子 ────────────────────────────────────────────────
    def test_buy_outside_pool_blocked(self):
        out = self._run([self._buy(code="601899")])
        assert any("不在阿深自选池" in v for v in out["violations"]), out["violations"]

    def test_user_watch_is_read_only(self):
        """老板的自选只读 —— 但**不是因为 code 黑名单**。

        同一只票同时在两套自选里是正常状态（老板提名后大概如此），
        所以判"code 在老板池里就不许加"会误伤。真正的保证是结构性的：
        没有"写老板池"的动作类型，任何这类动作都过不了白名单。
        """
        out = self._run([{"action": "watch_user_add", "code": "600519", "reason": "x"}],
                        user_watch={"600519"})
        assert any("未登记的动作类型" in v for v in out["violations"]), out["violations"]
        # 反过来：同一只票**允许**加进阿深自己的池子
        ok = self._run([{"action": "watch_add", "code": "600519", "reason": "我也盯"}],
                       my_watch=set(), user_watch={"600519"})
        assert ok["blocked"] == [], ok["blocked"]

    # ── 三件套 ──────────────────────────────────────────────
    def test_three_pieces_required(self):
        """限价 / 止损 / 理由，缺任何一个这笔不作数（§4.13）。"""
        no_reason = self._run([self._buy(reason="")])
        assert any("缺买入理由" in v for v in no_reason["violations"])
        no_limit = self._run([self._buy(limit=None)])
        assert any("缺限价" in v for v in no_limit["violations"])
        no_stop = self._run([{**self._buy(), "stop_loss": None}])
        assert any("缺止损价" in v for v in no_stop["violations"])

    def test_stop_can_come_from_same_batch(self):
        """止损允许由同批的 `set_wake(stop_loss)` 提供（与 `ensure_stops` 同约）。"""
        acts = [{**self._buy(), "stop_loss": None},
                {"action": "set_wake", "code": "600519", "kind": "stop_loss",
                 "threshold": 9.5, "reason": "跌破就走"}]
        out = self._run(acts)
        assert [k for k in out["kept"] if k["action"] == "buy"], out["violations"]

    def test_limit_far_above_price_is_not_a_limit_order(self):
        out = self._run([self._buy(limit=10.5)])       # 现价 10.0，+5%
        assert any("不是限价单" in v for v in out["violations"]), out["violations"]

    # ── 六不买（机械判据）──────────────────────────────────
    @pytest.mark.parametrize("key,quote_kw", [
        ("chasing", {"change_pct": 5.0}),                 # 追涨
        ("near_limit_down", {"change_pct": -8.0}),        # 接近跌停
        ("one_word_board", {"high": 11.0, "low": 11.0}),  # 一字板
        ("risk_name", {"name": "ST测试"}),                 # ST
        ("risk_name", {"name": "测试退"}),                 # 退市
        ("illiquid", {"amount_wan": 1200.0}),             # 成交额太小
        ("halted", {"volume_hand": 0.0}),                 # 停牌
        ("no_price", {"price": 0.0}),                     # 无价
    ])
    def test_buy_blocks_fire(self, key, quote_kw):
        keys = [k for k, _ in A.buy_blocks(self._q(**quote_kw))]
        assert key in keys, f"{key} 未触发，实际 {keys}"

    def test_buy_blocks_clean_quote_passes(self):
        assert A.buy_blocks(self.Q) == []

    @pytest.mark.parametrize("name,blocked", [
        # 真正的风险股：ST/*ST/SST/S*ST 都是**前缀**
        ("ST测试", True), ("*ST海航", True), ("SST天海", True), ("S*ST前锋", True),
        # 退市整理股：简称里带「退」，可在首（退市中新）也可在尾（长动退）
        ("退市中新", True), ("长动退", True),
        # ⚠️ 反例：子串写法 `"ST" in name` 会把它们全误判成 ST 拒掉
        ("TEST", False), ("中ST化", False), ("伊利股份", False), ("银行ETF", False),
    ])
    def test_risk_name_is_anchored_not_substring(self, name, blocked):
        """风险股判据必须锚定，不能用子串（2026-10-27 审查反例：name="TEST"）。

        子串写法在本项目数据上误伤≈0，但"数据里恰好不出现"不是判据 ——
        审查用 `name="TEST"` 构造的探针就把它按 ST 拦下了。
        """
        keys = [k for k, _ in A.buy_blocks(self._q(name=name))]
        assert ("risk_name" in keys) is blocked, (name, keys)

    def test_missing_quote_is_a_block_not_a_pass(self):
        """"没有行情" ≠ "没有否决理由"。混淆这两者会买进一只查不到价的票。"""
        assert "no_quote" in [k for k, _ in A.buy_blocks(None)]
        out = self._run([self._buy()], quotes={})
        assert any("没有行情数据" in v for v in out["violations"]), out["violations"]

    # ── 仓位数学 ────────────────────────────────────────────
    def test_single_position_cap(self):
        out = self._run([self._buy(size=0.10)])        # 单票上限 8%
        assert any("单票" in v for v in out["violations"]), out["violations"]

    def test_first_buy_cap(self):
        out = self._run([self._buy(size=0.05)])        # 首笔上限 3%
        assert any("首笔" in v for v in out["violations"]), out["violations"]

    def test_max_stocks_cap(self):
        """12 只 × 8% = 96%，第 13 只必须被拒。"""
        held = {f"60{i:04d}": {"market_value": 80000.0} for i in range(12)}
        out = self._run([self._buy(code="000858", size=0.03)],
                        portfolio=self._port(cash=40000.0, holdings=held))
        assert any("达上限" in v for v in out["violations"]), out["violations"]

    def test_unrecognized_holding_shape_blocks_buys(self):
        """持仓项形状不认识 ⇒ **拒买**，绝不能 `continue` 跳过。

        这条是实测踩出来的：跳过不认识的行 ⇒ "12 只持仓被当成空仓" ⇒
        只数上限和总仓上限全部静默失效，而且不报错、看起来一切正常。
        """
        held = {f"60{i:04d}": 80000.0 for i in range(12)}      # 裸数字，不是 dict
        out = self._run([self._buy(code="000858", size=0.03)],
                        portfolio=self._port(cash=40000.0, holdings=held))
        assert any("账户快照不完整" in v for v in out["violations"]), out["violations"]
        assert not [k for k in out["kept"] if k["action"] == "buy"]

    def test_total_position_cap(self):
        """11 只满仓（88%）+ 第 12 只首笔 3% ⇒ 91%，**必须允许**——
        否则 100% 总仓就是纸面权限（这正是本次改动的目的）。"""
        held = {f"60{i:04d}": {"market_value": 80000.0} for i in range(11)}
        ok = self._run([self._buy(code="000858", size=0.03)],
                       portfolio=self._port(cash=120000.0, holdings=held))
        assert ok["blocked"] == [], ok["blocked"]

    def test_cash_insufficient(self):
        out = self._run([self._buy(size=0.03)],
                        portfolio=self._port(cash=1000.0))
        assert any("现金只有" in v for v in out["violations"]), out["violations"]

    def test_incomplete_portfolio_blocks_all_buys(self):
        """算不出净值就**拒买**，不是"当作有 100 万"。（缺数据＝拒绝）"""
        out = self._run([self._buy()], portfolio={"cash": None})
        assert any("账户快照不完整" in v for v in out["violations"])
        assert any("本轮所有买入被拒" in n for n in out["notes"])

    # ── 其它动作 ────────────────────────────────────────────
    def test_watch_pool_cap(self):
        """池子上限 25 只（§5.2）：超了会顶爆 `MAX_ARMED_CODES` 并静默截断。"""
        full = {f"60{i:04d}" for i in range(A.MAX_ASHEN_WATCH)}
        out = self._run([{"action": "watch_add", "code": "000858", "reason": "x"}],
                        my_watch=full)
        assert any("自选池已达上限" in v for v in out["violations"]), out["violations"]

    def test_unknown_action_rejected(self):
        """白名单而非黑名单：没登记的动作一律拒，不默认放行。"""
        out = self._run([{"action": "transfer_cash", "code": "600519", "amount": 1e6}])
        assert any("未登记的动作类型" in v for v in out["violations"])

    def test_study_cap_per_round(self):
        acts = [{"action": "study", "code": "600519", "reason": "a"},
                {"action": "study", "code": "000858", "reason": "b"}]
        out = self._run(acts)
        assert len([k for k in out["kept"] if k["action"] == "study"]) == 1
        assert any("最多登记" in v for v in out["violations"])

    def test_sell_without_holding_rejected(self):
        out = self._run([{"action": "sell", "code": "600519", "reason": "x"}])
        assert any("没有该票的可卖持仓" in v for v in out["violations"])

    def test_kept_always_has_pass(self):
        """全被拦时也要留一条 `pass`：否则这一轮在库里是空行，看不出"发生了什么"。"""
        out = self._run([self._buy(code="601899")])
        assert [k["action"] for k in out["kept"]] == ["pass"]

    def test_note_explains_disabled_cash_floor(self):
        out = self._run([self._buy()])
        assert any("现金地板已关闭" in n for n in out["notes"]), out["notes"]


class TestDescribe:
    def test_numbers_come_from_params_not_hardcoded(self):
        """prompt 里的边界必须由 `describe()` 生成。

        手写"8%、24%、3 只"的历史：改参数忘了改 prompt，模型按旧数字规划、
        代码按新数字拦 ⇒ 两边都对、结果全被拒。
        """
        txt = A.describe(A.resolve())
        assert "100%" in txt and "12 只" in txt
        assert "现金地板已关闭" in txt
        txt2 = A.describe(A.resolve({"max_total_pct": 0.5}))
        assert "50%" in txt2 and "现金地板 50%" in txt2
