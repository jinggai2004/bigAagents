"""深度分析复盘（analysis_review）单元测试：临时库 + 假 K 线，零网络/零 LLM。

覆盖四层：
1. path_facts —— 客观路径（先止损/后反弹、MFE/MAE 方向、看空对称口径）
2. mechanical_verdict / candidate_causes —— **判据**是否真的按数据成立
3. validate_llm_review —— 候选外原因、编造证据字段、越界改进项是否被拦
4. 接线 —— 回填到 done 时是否真的产出复盘（"验了函数" ≠ "验了接线"）

⚠️ 每个关键用例都配了**反向**断言（该出现的不出现 / 不该出现的出现了），
   否则"恒真"的实现也能全绿。
"""
from datetime import datetime, timedelta

import pandas as pd
import pytest

from core import analysis_review as rv
from core import analysis_history as ah
from db import store as db_store


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "test_review.db")
    yield


def _kline(points: list[tuple[str, float]], pad: float = 1.0) -> pd.DataFrame:
    """按给定 (日期, 收盘) 序列造 K 线；high/low 用 pad 撑开。"""
    return pd.DataFrame({
        "date": [p[0] for p in points],
        "close": [p[1] for p in points],
        "high": [p[1] + pad for p in points],
        "low": [p[1] - pad for p in points],
    })


def _seq(start: str, closes: list[float], pad: float = 1.0) -> pd.DataFrame:
    """从 start 起逐日（自然日）生成 K 线，仅用于行数/顺序，不查交易日历。"""
    d0 = datetime.strptime(start, "%Y-%m-%d")
    pts = [((d0 + timedelta(days=i)).strftime("%Y-%m-%d"), c)
           for i, c in enumerate(closes)]
    return _kline(pts, pad=pad)


# ================================================================
# 1. path_facts：客观路径
# ================================================================

class TestPathFacts:

    def test_base_price_is_analysis_day_close(self):
        """基准价 = 分析日当天收盘（与 forward_returns 同口径），不是次日"""
        df = _seq("2026-01-01", [100, 101, 102, 103, 104, 105, 106])
        f = rv.path_facts(df, "2026-01-01", {}, "看多")
        assert f["base_date"] == "2026-01-01"
        assert f["base_price"] == 100.0
        assert f["window_days"] == 6          # 之后 6 根

    def test_stop_then_recover_is_recorded(self):
        """先破止损、之后又回到成本之上 ⇒ days_to_stop + stop_then_recover 都必须有"""
        # 100 → 93（破 95 止损）→ 一路上到 106
        df = _seq("2026-01-01", [100, 97, 93, 96, 100, 104, 106, 105, 108, 110])
        f = rv.path_facts(df, "2026-01-01",
                          {"direction": "看多", "stop_loss": 95, "target_price": 104},
                          "看多")
        assert f["days_to_stop"] == 2
        assert f["stop_then_recover"] is True
        assert f["stop_then_target"] is True   # 扫损之后确实到了目标
        assert f["hit_stop"] == 1

    def test_no_recover_when_price_keeps_falling(self):
        """止损后继续跌 ⇒ stop_then_recover 必须是 False（反向鉴别）"""
        # 收盘 96 时 low=95 已经触及 95 止损 ⇒ 第 1 根就被扫出
        df = _seq("2026-01-01", [100, 96, 92, 88, 84, 80, 78, 76, 74, 72])
        f = rv.path_facts(df, "2026-01-01",
                          {"direction": "看多", "stop_loss": 95, "target_price": 104},
                          "看多")
        assert f["days_to_stop"] == 1
        assert f["stop_then_recover"] is False
        assert f["stop_then_target"] is False

    def test_mae_mfe_direction(self):
        """看多：mfe 取最高、mae 取最低；看空必须对称反过来"""
        df = _seq("2026-01-01", [100, 110, 90, 100])
        bull = rv.path_facts(df, "2026-01-01", {}, "看多")
        bear = rv.path_facts(df, "2026-01-01", {}, "看空")
        assert bull["mfe_pct"] == pytest.approx(0.11, abs=1e-3)   # (111-100)/100
        assert bull["mae_pct"] == pytest.approx(0.11, abs=1e-3)   # (100-89)/100
        # 看空时"有利"是下跌 ⇒ 同一个 high 变成不利、low 变成有利
        assert bear["mfe_pct"] == pytest.approx(0.11, abs=1e-3)
        assert bear["mae_pct"] == pytest.approx(0.11, abs=1e-3)

    def test_bearish_target_below(self):
        """看空的目标在下方（跌破才算触及），止损在上方 —— 口径写反必挂"""
        # 100 → 85：目标 90 被跌破，止损 105 从未涨破
        df = _seq("2026-01-01", [100, 96, 92, 88, 86, 85, 84])
        f = rv.path_facts(df, "2026-01-01",
                          {"direction": "看空", "target_price": 90, "stop_loss": 105},
                          "看空")
        assert f["hit_target"] == 1
        assert f["hit_stop"] == 0
        assert f["days_to_target"] == 3          # 第 3 根收盘 88、low 87 已破 90
        # 目标空间按**有利方向**记正数：看空跌 10% 到目标 ⇒ +0.10
        assert f["target_dist_pct"] == pytest.approx(0.10, abs=1e-3)

    def test_no_window_returns_empty_facts(self):
        """分析日之后没有 K 线 ⇒ window_days=0（调用方据此跳过，不算失败）"""
        df = _seq("2026-01-01", [100])
        f = rv.path_facts(df, "2026-01-01", {}, "看多")
        assert f["window_days"] == 0
        assert "ret_5d" not in f


# ================================================================
# 2. verdict 与候选原因：判据是否真的按数据成立
# ================================================================

class TestMechanicalVerdict:

    @pytest.mark.parametrize("label,ret,expect", [
        ("看多", -0.15, "wrong"),
        ("看多", 0.05, "correct"),
        ("看多", 0.01, "unclear"),          # 方向对但幅度太小
        ("看空", -0.10, "correct"),         # 对称口径：看空跌了才对
        ("看空", 0.10, "wrong"),
        ("看多", None, "unclear"),
    ])
    def test_verdict(self, label, ret, expect):
        facts = {} if ret is None else {"ret_20d": ret}
        assert rv.mechanical_verdict(facts, label) == expect

    def test_watch_then_surge_is_opportunity_missed(self):
        """观望后大涨不算"判错方向"，但要单独记"错失机会"（不能混进胜率）"""
        facts = {"ret_20d": 0.15}
        assert rv.mechanical_verdict(facts, "观望") == "opportunity_missed"
        assert rv.mechanical_verdict({"ret_20d": 0.02}, "观望") == "no_position"


class TestCandidateCauses:

    def test_stop_too_tight_requires_recovery(self):
        """只有"先止损 + 之后回到成本之上"才判止损过近"""
        tight = {"days_to_stop": 2, "stop_then_recover": True, "stop_dist_atr": 0.6,
                 "ret_20d": 0.05, "hit_stop": 1}
        codes = [c["code"] for c in
                 rv.candidate_causes(tight, "看多", {"stop_loss": 95}, {})]
        assert "stop_too_tight" in codes

        # 反向：同样触了止损，但之后一路跌 ⇒ 绝不能判"止损过近"
        loose = {"days_to_stop": 2, "stop_then_recover": False,
                 "ret_20d": -0.28, "hit_stop": 1}
        codes2 = [c["code"] for c in
                  rv.candidate_causes(loose, "看多", {"stop_loss": 95}, {})]
        assert "stop_too_tight" not in codes2
        assert "direction_wrong" in codes2

    def test_market_beta_when_underperform_index_but_beat_it(self):
        """绝对亏 5% 但基准亏 8%（超额 +3%）⇒ 归因为大盘，不是选股"""
        facts = {"ret_20d": -0.05, "bench_ret_20d": -0.08, "excess_20d": 0.03}
        codes = [c["code"] for c in rv.candidate_causes(facts, "看多", {}, {})]
        assert "market_beta" in codes

    def test_no_stop_only_when_no_stop_configured(self):
        """没设止损且亏 6% ⇒ no_stop；设了止损就不该出现这条"""
        facts = {"ret_20d": -0.06}
        assert "no_stop" in [c["code"] for c in
                             rv.candidate_causes(facts, "看多", {}, {})]
        assert "no_stop" not in [c["code"] for c in
                                 rv.candidate_causes(facts, "看多",
                                                     {"stop_loss": 95}, {})]

    def test_target_unrealistic_needs_mfe_evidence(self):
        """没触目标 + 最大涨幅不到目标一半 ⇒ 目标不现实"""
        facts = {"hit_target": 0, "target_dist_pct": 0.30, "mfe_pct": 0.05,
                 "ret_20d": 0.03}
        assert "target_unrealistic" in [
            c["code"] for c in rv.candidate_causes(facts, "看多",
                                                   {"target_price": 130}, {})]
        # 反向：最大涨幅接近目标（只是没到）⇒ 不能怪目标太远
        facts2 = {"hit_target": 0, "target_dist_pct": 0.30, "mfe_pct": 0.25,
                  "ret_20d": 0.20}
        assert "target_unrealistic" not in [
            c["code"] for c in rv.candidate_causes(facts2, "看多",
                                                   {"target_price": 130}, {})]

    def test_missing_data_decision(self):
        """输入有明确缺失 + 看多 + 非零仓位 ⇒ 带缺数据下结论"""
        facts = {"ret_20d": -0.10}
        payload = {"data_availability": {"financials": False, "news": True}}
        codes = [c["code"] for c in rv.candidate_causes(
            facts, "看多", {"position_size": "15%"}, payload)]
        assert "data_missing_decision" in codes
        # 反向：仓位为 0 ⇒ 没有下注，不算"带缺数据下强结论"
        codes2 = [c["code"] for c in rv.candidate_causes(
            facts, "看多", {"position_size": "0%"}, payload)]
        assert "data_missing_decision" not in codes2

    def test_unknown_label_is_treated_as_directionless(self):
        """⭐ 白名单化的意义：**新标签**不能落到看多分支（审查 2026-10-20）。

        原实现是排除法 `bull = "看空" not in label`。一旦 `_extract_decision`
        将来多返回一个标签（"暂缓"/"看不清"），它会被当成看多 —— 股价跌了
        立刻判"方向错误"，而且归因分布从此污染。

        现在只认 `startswith("看多")` / `startswith("看空")`，其余一律按
        "不预测方向"处理（no_position），最多判 opportunity_missed。
        """
        facts = {"ret_5d": -0.08}          # 明显下跌
        for label in ("暂缓", "看不清", "分析失败", "", "HOLD"):
            assert rv.mechanical_verdict(facts, label, horizon=5) == "no_position", label
            codes = [c["code"] for c in
                     rv.candidate_causes(facts, label, {}, {}, horizon=5)]
            assert "direction_wrong" not in codes, label

    def test_watch_label_never_gets_direction_wrong(self):
        """观望/驳回不预测方向 ⇒ 跌了也不能判"方向判错"（真实数据踩到的误报）

        实测：`002141 观望 5 日 -6.12%` 被判 direction_wrong —— 读起来荒谬，
        而且会把"该不该进"和"看没看反"混成一类。
        """
        facts = {"ret_5d": -0.0612, "ret_20d": -0.10}
        for label in ("观望", "驳回"):
            codes = [c["code"] for c in
                     rv.candidate_causes(facts, label, {}, {}, horizon=5)]
            assert "direction_wrong" not in codes, label
        # 观望后大涨 ⇒ 只该是"错失机会"
        up = [c["code"] for c in rv.candidate_causes(
            {"ret_5d": 0.15}, "观望", {}, {}, horizon=5)]
        assert up == ["opportunity_missed"]

    def test_target_unrealistic_only_in_full_window(self):
        """目标价是给 T+20 定的 ⇒ 5~9 天窗口不能判"目标不现实"

        实测：002129 只走了 9 天、T+5 已 +2.41%，却被判 target_unrealistic。
        """
        facts = {"hit_target": 0, "target_dist_pct": 0.30, "mfe_pct": 0.04,
                 "ret_5d": 0.024, "window_days": 9}
        assert "target_unrealistic" not in [
            c["code"] for c in rv.candidate_causes(
                facts, "看多", {"target_price": 130}, {}, horizon=5)]
        # 完整窗口（≥15 根）才允许
        facts20 = dict(facts, window_days=20, ret_20d=0.03)
        assert "target_unrealistic" in [
            c["code"] for c in rv.candidate_causes(
                facts20, "看多", {"target_price": 130}, {}, horizon=20)]

    def test_always_returns_at_least_one_code(self):
        """候选集不能为空 —— LLM 必须在有限集合里选，空集合会让它自由发挥"""
        for facts in ({}, {"ret_20d": 0.02}):
            assert rv.candidate_causes(facts, "看多", {}, {})


# ================================================================
# 3. LLM 归因校验：防幻觉的三道闸门
# ================================================================

class TestValidateLLMReview:

    def _mech(self):
        return {"facts": {"ret_20d": -0.1, "stop_dist_atr": 0.6},
                "primary_cause": "direction_wrong",
                "candidates": [{"code": "direction_wrong"}, {"code": "stop_too_tight"}]}

    def test_rejects_cause_outside_candidates(self):
        """模型编一个候选外的原因 ⇒ 降级为机械归因并留痕"""
        out, warns = rv.validate_llm_review(
            {"primary_cause": "macro_headwind", "evidence": []}, self._mech())
        assert out["primary_cause"] == "direction_wrong"
        assert any("不在候选集内" in w for w in warns)

    def test_drops_fabricated_evidence_fields(self):
        """证据字段在 facts 里不存在 ⇒ 剔除（这是"看起来有据"的典型形态）"""
        out, warns = rv.validate_llm_review({
            "primary_cause": "direction_wrong",
            "evidence": [{"field": "ret_20d", "value": -0.1},
                         {"field": "northbound_flow", "value": 1e9}]}, self._mech())
        assert [e["field"] for e in out["evidence"]] == ["ret_20d"]
        assert any("不存在的证据字段" in w for w in warns)

    def test_rejects_improvement_target_outside_whitelist(self):
        """改进项不能指向"更谨慎一些"这种不可执行的东西"""
        out, warns = rv.validate_llm_review({
            "primary_cause": "direction_wrong",
            "improvements": [{"target": "be_more_careful", "change": "下次谨慎"},
                             {"target": "stop_loss_rule", "change": "≥1.5ATR"}]},
            self._mech())
        assert [i["target"] for i in out["improvements"]] == ["stop_loss_rule"]
        assert any("白名单" in w for w in warns)

    def test_secondary_duplicates_primary_dropped(self):
        out, _ = rv.validate_llm_review({
            "primary_cause": "direction_wrong",
            "secondary_causes": ["direction_wrong", "stop_too_tight"]}, self._mech())
        assert out["secondary_causes"] == ["stop_too_tight"]


# ================================================================
# 4. store 往返 + 聚合
# ================================================================

class TestStoreAndAggregate:

    def test_upsert_review_is_idempotent_per_analysis(self, temp_store):
        """同一 analysis_id 覆盖而不是插两条（否则归因分布会被重复计数）"""
        aid = db_store.insert_analysis({
            "analysis_date": "2026-01-05", "stock_code": "600000",
            "stock_name": "浦发", "decision_label": "看多", "payload": {},
            "trade_plan": {}, "backfill_status": "done"})
        assert aid > 0
        base = {"analysis_id": aid, "stock_code": "600000", "verdict": "wrong",
                "primary_cause": "direction_wrong", "facts": {"ret_20d": -0.1},
                "candidates": [], "status": "mechanical"}
        assert db_store.upsert_analysis_review(base) > 0
        assert db_store.upsert_analysis_review(
            dict(base, verdict="correct", primary_cause="nothing_wrong")) > 0
        got = db_store.get_analysis_review(aid)
        assert got["verdict"] == "correct"
        assert got["facts"]["ret_20d"] == -0.1        # JSON 往返
        assert len(db_store.list_analysis_reviews()) == 1

    def test_pending_picks_backfilled_without_review(self, temp_store):
        """done 与 partial 都该被捞（partial 出早期结论），且 done 排在前面"""
        a1 = db_store.insert_analysis({
            "analysis_date": "2026-01-05", "stock_code": "600001",
            "decision_label": "看多", "payload": {}, "trade_plan": {},
            "backfill_status": None})
        a2 = db_store.insert_analysis({
            "analysis_date": "2026-01-06", "stock_code": "600002",
            "decision_label": "看多", "payload": {}, "trade_plan": {},
            "backfill_status": None})
        a3 = db_store.insert_analysis({
            "analysis_date": "2026-01-07", "stock_code": "600003",
            "decision_label": "看多", "payload": {}, "trade_plan": {},
            "backfill_status": None})
        # ⚠️ 没有任何 T+5 数据的不该被捞（"还没到可复盘的时候"不是"复盘失败"）
        db_store.update_analysis_backfill(a1, {"ret_5d": 0.03, "ret_20d": 0.05,
                                              "backfill_status": "done"})
        db_store.update_analysis_backfill(a2, {"ret_5d": 0.03,
                                              "backfill_status": "partial"})
        rows = db_store.pending_review_analyses()
        assert [r["id"] for r in rows] == [a1, a2]      # a3 没数据 ⇒ 不出现
        assert rows[0]["backfill_status"] == "done"     # 完整窗口优先

    def test_aggregate_filters_low_support(self):
        """出现 2 次的改进项不能升级为建议（防单样本过拟合）"""
        rows = [{"verdict": "wrong", "primary_cause": "direction_wrong",
                 "llm_review": {"improvements": [
                     {"target": "stop_loss_rule", "change": "放宽到 1.5ATR"}]}}
                for _ in range(2)]
        agg = rv.aggregate(rows, min_support=3)
        assert agg["n"] == 2
        assert agg["by_cause"][0]["code"] == "direction_wrong"
        assert agg["by_cause"][0]["count"] == 2
        assert agg["advice"] == []
        assert agg["pending"][0]["target"] == "stop_loss_rule"


# ================================================================
# 5. 接线：回填到 done 时必须真的产出复盘
# ================================================================

class TestBackfillWiring:

    def test_backfill_done_creates_review(self, temp_store):
        """T+20 到期 → analysis_reviews 自动出现一条（机械层，零 LLM）"""
        ana_date = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")
        aid = db_store.insert_analysis({
            "analysis_date": ana_date, "stock_code": "600519",
            "stock_name": "贵州茅台", "decision_label": "看多",
            "payload": {"data_availability": {}}, "trade_plan": {
                "direction": "看多", "target_price": 110, "stop_loss": 95},
            "price_at_analysis": 100})
        assert aid > 0

        # 分析日收盘 100，之后跌破 95 再反弹到 108（21 根 ⇒ T+20 到期）
        # ⚠️ 前置 7 根必须正好填满"分析日前 7 天"，否则 closes[0] 不落在分析日上，
        #    基准价就会错位（第一版就是这么错的：done=0、partial=1）。
        closes = [100, 97, 93, 96, 100, 104, 106, 108] + [107 + i * 0.1 for i in range(13)]
        df = _seq((datetime.strptime(ana_date, "%Y-%m-%d")
                   - timedelta(days=7)).strftime("%Y-%m-%d"), [99] * 7 + closes)

        stats = ah.backfill_analyses(limit=10, price_fetcher=lambda c, s, e: df,
                                     use_lock=False)
        assert stats["done"] == 1, stats
        assert stats["reviewed"] == 1, stats
        rev = db_store.get_analysis_review(aid)
        assert rev is not None
        assert rev["status"] == "mechanical"
        assert rev["facts"]["days_to_stop"] == 2
        assert rev["facts"]["stop_then_recover"] is True
        # 最终是涨的（verdict=correct）但止损被扫 —— "对了"不等于"做对了"
        assert rev["verdict"] == "correct"
        assert "stop_too_tight" in [c["code"] for c in rev["candidates"]]

    def test_partial_backfill_creates_early_review(self, temp_store):
        """只有 T+5 数据时也要产出**早期复盘**（horizon=5）

        ⚠️ 否则最早的分析要等一个月才有第一条结论 —— 而"方向判对判错"在 T+5
        就已经可判。窗口每天变长，同一条会被覆盖为更完整的 facts。
        """
        ana_date = (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")
        aid = db_store.insert_analysis({
            "analysis_date": ana_date, "stock_code": "600520",
            "decision_label": "看多", "payload": {}, "trade_plan": {},
            "price_at_analysis": 100})
        assert aid > 0
        # 分析日 100 → 之后 6 根涨到 106：T+5 收益 +5%（判对），T+20 还差得远
        df = _seq((datetime.strptime(ana_date, "%Y-%m-%d")
                   - timedelta(days=7)).strftime("%Y-%m-%d"),
                  [99] * 7 + [100 + i for i in range(7)])
        stats = ah.backfill_analyses(limit=10, price_fetcher=lambda c, s, e: df,
                                     use_lock=False)
        assert stats["partial"] == 1, stats
        assert stats["reviewed"] == 1, stats
        rev = db_store.get_analysis_review(aid)
        assert rev is not None
        assert rev["horizon"] == 5
        assert rev["verdict"] == "correct"        # 用 ret_5d 判，不是 unclear
        assert rev["facts"]["ret_5d"] == pytest.approx(0.05, abs=1e-3)

    def test_full_window_overrides_early_review(self, temp_store):
        """T+20 到期后，同一条复盘被覆盖为 horizon=20（不是新增第二条）"""
        ana_date = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")
        aid = db_store.insert_analysis({
            "analysis_date": ana_date, "stock_code": "600521",
            "decision_label": "看多", "payload": {}, "trade_plan": {},
            "price_at_analysis": 100})
        # 先人为写一条"早期复盘"
        db_store.upsert_analysis_review({
            "analysis_id": aid, "stock_code": "600521", "analysis_date": ana_date,
            "verdict": "correct", "primary_cause": "nothing_wrong", "horizon": 5,
            "facts": {"ret_5d": 0.05}, "candidates": [], "status": "mechanical"})
        closes = [100] + [100 - i for i in range(1, 21)]     # 20 天后跌到 80
        df = _seq((datetime.strptime(ana_date, "%Y-%m-%d")
                   - timedelta(days=7)).strftime("%Y-%m-%d"), [99] * 7 + closes)
        ah.backfill_analyses(limit=10, price_fetcher=lambda c, s, e: df, use_lock=False)
        rev = db_store.get_analysis_review(aid)
        assert rev["horizon"] == 20
        assert rev["verdict"] == "wrong"
        assert len(db_store.list_analysis_reviews()) == 1


# ================================================================
# 6. 里程碑自动 LLM（2026-10-29）：T+5 首写 / T+20 到期自动调模型写总结
# ================================================================

class _StubClient:
    """桩客户端：`review_one` 只会调 `call_api(messages, temperature=…, purpose=…)`。"""

    def __init__(self, reply: str | None = None, error: str | None = None):
        self.calls = 0
        self._reply = reply
        self._error = error

    def call_api(self, messages, **kw):                 # noqa: ANN001
        self.calls += 1
        if self._error:
            raise RuntimeError(self._error)
        return self._reply


_OK_REPLY = (
    '{"primary_cause": "nothing_wrong", "secondary_causes": [],'
    ' "verdict_reason": "T+5 涨 5%，方向判对",'
    ' "evidence": [{"field": "ret_5d", "value": 0.05, "means": "5 日收益 +5%"}],'
    ' "what_was_right": "方向判断",'
    ' "improvements": [{"target": "none", "change": "", "why": ""}]}'
)


class TestMilestoneAutoLLM:

    def test_env_gate_runtime_parsing(self, monkeypatch):
        """开关运行时读 env：默认开；0/false/off/no 关；1 显式开。"""
        monkeypatch.delenv(rv.ENV_ENABLE_LLM, raising=False)
        assert rv.llm_autorun_enabled() is True
        for off in ("0", "false", "OFF", "no"):
            monkeypatch.setenv(rv.ENV_ENABLE_LLM, off)
            assert rv.llm_autorun_enabled() is False
        monkeypatch.setenv(rv.ENV_ENABLE_LLM, "1")
        assert rv.llm_autorun_enabled() is True

    def test_t5_milestone_auto_llm(self, temp_store, monkeypatch):
        """T+5 首写 → 自动调 LLM → 复盘带 llm_review，主因以校验后的模型选择为准"""
        monkeypatch.setenv(rv.ENV_ENABLE_LLM, "1")
        ana_date = (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")
        aid = db_store.insert_analysis({
            "analysis_date": ana_date, "stock_code": "600530",
            "decision_label": "看多", "payload": {}, "trade_plan": {},
            "price_at_analysis": 100})
        df = _seq((datetime.strptime(ana_date, "%Y-%m-%d")
                   - timedelta(days=7)).strftime("%Y-%m-%d"),
                  [99] * 7 + [100 + i for i in range(7)])
        stub = _StubClient(reply=_OK_REPLY)
        stats = ah.backfill_analyses(limit=10, price_fetcher=lambda c, s, e: df,
                                     use_lock=False, review_client=stub)
        assert stats["reviewed"] == 1 and stats["llm_ok"] == 1, stats
        assert stub.calls == 1
        rev = db_store.get_analysis_review(aid)
        assert rev["status"] == "ok"
        assert rev["verdict"] == "correct"            # 对错仍由代码判，模型动不了
        assert rev["primary_cause"] == "nothing_wrong"
        assert rev["llm_review"]["evidence"][0]["field"] == "ret_5d"

    def test_nonmilestone_rerun_preserves_llm_review(self, temp_store, monkeypatch):
        """⚠️ 回归：日常重跑不得抹掉里程碑写的 LLM 归因（旧实现每天机械 upsert
        会把 llm_review 覆盖成 NULL —— 自动 LLM 等于没做）"""
        monkeypatch.setenv(rv.ENV_ENABLE_LLM, "1")
        ana_date = (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")
        aid = db_store.insert_analysis({
            "analysis_date": ana_date, "stock_code": "600531",
            "decision_label": "看多", "payload": {}, "trade_plan": {},
            "price_at_analysis": 100})
        df = _seq((datetime.strptime(ana_date, "%Y-%m-%d")
                   - timedelta(days=7)).strftime("%Y-%m-%d"),
                  [99] * 7 + [100 + i for i in range(7)])
        stub = _StubClient(reply=_OK_REPLY)
        ah.backfill_analyses(limit=10, price_fetcher=lambda c, s, e: df,
                             use_lock=False, review_client=stub)
        assert db_store.get_analysis_review(aid)["status"] == "ok"
        # 第二天：partial 行被再次处理，但没有新里程碑 —— 复盘必须原样保留
        stats2 = ah.backfill_analyses(limit=10, price_fetcher=lambda c, s, e: df,
                                      use_lock=False, review_client=stub)
        assert stats2["reviewed"] == 0, stats2
        assert stub.calls == 1                          # 没有第二次 LLM 调用
        rev = db_store.get_analysis_review(aid)
        assert rev["status"] == "ok"
        assert rev["llm_review"]["primary_cause"] == "nothing_wrong"

    def test_t20_milestone_auto_llm(self, temp_store, monkeypatch):
        """T+20 到期 → 覆盖为 horizon=20 且带 LLM 归因"""
        monkeypatch.setenv(rv.ENV_ENABLE_LLM, "1")
        ana_date = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")
        aid = db_store.insert_analysis({
            "analysis_date": ana_date, "stock_code": "600532",
            "decision_label": "看多", "payload": {}, "trade_plan": {},
            "price_at_analysis": 100})
        closes = [100] + [100 + i for i in range(1, 21)]   # 20 天涨到 120
        df = _seq((datetime.strptime(ana_date, "%Y-%m-%d")
                   - timedelta(days=7)).strftime("%Y-%m-%d"), [99] * 7 + closes)
        stub = _StubClient(reply=_OK_REPLY)
        stats = ah.backfill_analyses(limit=10, price_fetcher=lambda c, s, e: df,
                                     use_lock=False, review_client=stub)
        assert stats["done"] == 1 and stats["llm_ok"] == 1, stats
        rev = db_store.get_analysis_review(aid)
        assert rev["horizon"] == 20
        assert rev["status"] == "ok"

    def test_llm_failure_keeps_mechanical_conclusion(self, temp_store, monkeypatch):
        """LLM 失败 ⇒ 保留机械结论 + status=llm_error，绝不拖垮回填"""
        monkeypatch.setenv(rv.ENV_ENABLE_LLM, "1")
        ana_date = (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")
        aid = db_store.insert_analysis({
            "analysis_date": ana_date, "stock_code": "600533",
            "decision_label": "看多", "payload": {}, "trade_plan": {},
            "price_at_analysis": 100})
        df = _seq((datetime.strptime(ana_date, "%Y-%m-%d")
                   - timedelta(days=7)).strftime("%Y-%m-%d"),
                  [99] * 7 + [100 + i for i in range(7)])
        stub = _StubClient(error="模拟超时")
        stats = ah.backfill_analyses(limit=10, price_fetcher=lambda c, s, e: df,
                                     use_lock=False, review_client=stub)
        assert stats["partial"] == 1 and stats["llm_error"] == 1, stats
        rev = db_store.get_analysis_review(aid)
        assert rev["status"] == "llm_error"
        assert rev["verdict"] == "correct"             # 机械层结论完好
        assert rev["primary_cause"] == "nothing_wrong"

    def test_env_off_never_touches_client(self, temp_store):
        """conftest 默认关 ⇒ 里程碑只写机械复盘，客户端一次都不会被碰"""
        ana_date = (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")
        db_store.insert_analysis({
            "analysis_date": ana_date, "stock_code": "600534",
            "decision_label": "看多", "payload": {}, "trade_plan": {},
            "price_at_analysis": 100})
        df = _seq((datetime.strptime(ana_date, "%Y-%m-%d")
                   - timedelta(days=7)).strftime("%Y-%m-%d"),
                  [99] * 7 + [100 + i for i in range(7)])

        class _Explode:
            def call_api(self, *a, **k):               # pragma: no cover
                raise AssertionError("env 关着就不该碰客户端")

        stats = ah.backfill_analyses(limit=10, price_fetcher=lambda c, s, e: df,
                                     use_lock=False, review_client=_Explode())
        assert stats["reviewed"] == 1 and stats["llm_ok"] == 0, stats
        rows = db_store.list_analysis_reviews()
        assert rows and rows[0]["status"] == "mechanical"
        # 读者把 NULL 解析成 {}：空 dict 即"没有 LLM 归因"
        assert not rows[0]["llm_review"]
