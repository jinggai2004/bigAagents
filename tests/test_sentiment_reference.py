# -*- coding: utf-8 -*-
"""市场情绪参照层（`core/sentiment_reference.py`）的守护测试。

这里测的**不是"功能能跑"**，而是三条最容易在后续改动中被悄悄破坏的性质：
  1. 多空比的分母必须是「判出方向的条数」，不是「总条数」；
  2. 样本不足时**不给数字**，而不是给一个看起来很确定的数字；
  3. LLM 判定的置信度**由代码按样本质量强制收敛**，不听模型的。

第 3 条尤其重要：它是"小样本 + 高置信度"这种最危险输出的唯一拦截点。
"""
from __future__ import annotations

import pytest

from core import sentiment_reference as S


# ================================================================
# ① 规则层
# ================================================================

class TestClassifyStance:
    def test_explicit_actions_are_directional(self):
        assert S.classify_stance("今天满仓了")["stance"] == "bull"
        assert S.classify_stance("减仓，先跑一半")["stance"] == "bear"
        assert S.classify_stance("今天清仓")["stance"] == "bear"
        assert S.classify_stance("低吸一点")["stance"] == "bull"

    def test_watching_is_neutral_not_bull(self):
        """⚠️ 项目事故同源：`"看空" not in label` ⇒ 观望被当成看多。

        这里必须 tri-state：观望是 neutral，既不进 bull 也不进 bear 分母。
        """
        r = S.classify_stance("继续观望")
        assert r["stance"] == "neutral"
        assert r["reason"].startswith("仅观望类词")

    def test_negation_flips_direction(self):
        """"不看好"必须判成看空 —— 子串匹配会把它当"看好"。"""
        assert S.classify_stance("我不看好后市")["stance"] == "bear"
        assert S.classify_stance("看好这个位置")["stance"] == "bull"
        # 翻转过的事实要能在 hits 里查回来（可追溯）
        hits = S.classify_stance("我不看好后市")["hits"]
        assert any(h.get("negated") and h.get("flip_to") == "bear" for h in hits)

    def test_a_not_a_question_is_not_a_stance(self):
        """「要不要加仓」是提问，不是看多表态。

        ⚠️ 不拦这一类，股吧/知乎里大量"求建议"帖会被算成看多，
        多空比系统性偏多 —— 而且偏得毫无痕迹。
        """
        for text in ("要不要加仓？", "该不该抄底", "上不上车", "能不能买"):
            assert S.classify_stance(text)["stance"] == "unscored", text

    def test_no_stance_returns_unscored_not_neutral(self):
        """判不出来必须是 `unscored`，**不能**混进 neutral。

        neutral = 明确说了"观望"；unscored = 根本没表态。
        两者混一起，覆盖率就失去意义（"都判不出"会被读成"都在观望"）。
        """
        for text in ("打卡", "转发", "继续减了，我只有150%了", ""):
            assert S.classify_stance(text)["stance"] == "unscored", text

    def test_hits_are_reported_for_auditability(self):
        r = S.classify_stance("满仓，看好")
        assert {h["word"] for h in r["hits"]} >= {"满仓", "看好"}

    def test_style_phrase_does_not_count_as_stance(self):
        """⚠️ 真实误判回归：`"做高抛低吸等待下个牛市周期"` 曾判成 **bull**。

        "高抛低吸"是交易风格（网格/做T），不含多空判断；原文后半句
        "回到我熟悉的熊市"甚至偏空。若整段屏蔽失效，覆盖率越高错得越多。
        """
        r = S.classify_stance(
            "没有清，只是把个股清理了，只做宽基，做高抛低吸等待下个牛市周期")
        assert r["stance"] == "unscored", f"风格短语不应产生方向，实得 {r}"

    def test_standalone_low_suck_still_counts(self):
        """屏蔽的是整段搭配，不是词本身：单独"低吸"仍是看多。"""
        assert S.classify_stance("回调我就低吸")["stance"] == "bull"

    def test_negation_does_not_cross_punctuation(self):
        """⚠️ 真实误判回归：`"别问，已梭哈完"` 曾判成 **bear**。

        "别"与"梭哈"之间隔着"问，" —— "别问"是口语前缀，不是在否定"梭哈"。
        否定窗口一旦跨句读生效，所有"别问/不说了 + 表态"的帖子方向全反。
        """
        assert S.classify_stance("别问，已梭哈完")["stance"] == "bull"
        assert S.classify_stance("不说了，加仓")["stance"] == "bull"
        # 同一句读内的否定仍然有效
        assert S.classify_stance("不建议加仓")["stance"] == "bear"

    def test_position_percent_change_is_detected(self):
        """⚠️ 真实误判回归：`"我降了30%的仓位，满仓满融…"` 曾判 **bull**。

        词表只有"减仓/清仓"，覆盖不到"降了 N% 的仓位"这种最常见表述。
        补正则后变成 bull+bear 各一 ⇒ neutral（**不再是单边看多**）。
        它仍不是完美的（"满仓"那半属于规则层结构局限），但方向不再反。
        """
        r = S.classify_stance("我降了30%的仓位，满仓满融的心理压力太大")
        assert r["stance"] == "neutral", f"多空各一应判中性，实得 {r}"
        assert any(h.get("by") == "pattern" for h in r["hits"])


# ================================================================
# ② 聚合层
# ================================================================

def _items(*texts, votes=None):
    votes = votes or [None] * len(texts)
    return [{"text": t, "votes": v} for t, v in zip(texts, votes)]


class TestBuildSnapshot:
    def test_denominator_is_scored_not_total(self):
        """⭐ 核心断言：1 多 1 空 + 8 条没表态 ⇒ 多空比 50%，不是 10%。

        拿总条数当分母，会把"没人表态"算成"多空均衡"（可能撞巧看着合理），
        也会把"很多人观望"算成"极度分歧"，两种都是错的。
        """
        snap = S.build_snapshot(
            _items("满仓", "清仓", "打卡", "转发", "…", "…", "…", "…", "…", "…"),
            source="t")
        assert snap["n_total"] == 10
        assert snap["n_scored"] == 2
        assert snap["bull_ratio"] == 0.5
        assert snap["coverage"] == 0.2
        # 样本不足 ⇒ 必须自我标记，不能只给个数字
        assert snap["quality"] == "thin"
        assert any("门槛" in n for n in snap["notes"])

    def test_neutral_is_excluded_from_ratio(self):
        snap = S.build_snapshot(_items("满仓", "满仓", "满仓", "观望", "观望"),
                                source="t")
        assert snap["n_scored"] == 3
        assert snap["neutral"] == 2
        assert snap["bull_ratio"] == 1.0, "观望不能稀释多空比"

    def test_empty_input_is_empty_not_neutral(self):
        """⭐ 缺就是缺：没数据 ⇒ `quality=empty` 且比例是 None，**不是 0.5**。"""
        snap = S.build_snapshot([], source="t")
        assert snap["quality"] == "empty"
        assert snap["bull_ratio"] is None
        assert snap["dispersion"] is None
        assert snap["ci95"] is None

    def test_low_scored_even_with_high_total_is_thin(self):
        """100 条里只有 5 条能判方向 ⇒ 仍是 thin（总数大不等于样本好）。"""
        snap = S.build_snapshot(_items("满仓", "清仓", "打卡", *(["…"] * 97)),
                                source="t")
        assert snap["n_total"] == 100
        assert snap["n_scored"] == 2
        assert snap["quality"] == "thin"

    def test_high_total_but_low_coverage_is_flagged(self):
        """样本量够、但覆盖率低 ⇒ low_coverage（两种不足要分开报）。"""
        texts = ["满仓"] * 10 + ["清仓"] * 10 + ["打卡"] * 60
        snap = S.build_snapshot(_items(*texts), source="t")
        assert snap["n_scored"] == 20
        assert snap["quality"] == "low_coverage"
        assert any("覆盖率" in n for n in snap["notes"])

    def test_ci_width_shrinks_with_sample(self):
        small = S.build_snapshot(_items(*(["满仓"] * 5 + ["清仓"] * 5)), source="t")
        big = S.build_snapshot(_items(*(["满仓"] * 100 + ["清仓"] * 100)), source="t")
        w_small = small["ci95"][1] - small["ci95"][0]
        w_big = big["ci95"][1] - big["ci95"][0]
        assert w_small > w_big * 3, "小样本的置信区间必须显著更宽"

    def test_vote_weighted_divergence_is_surfaced(self):
        """等权 50% 但高赞全在看多 ⇒ `weight_gap` 必须暴露出来。"""
        snap = S.build_snapshot([
            {"text": "满仓", "votes": 500},
            {"text": "清仓", "votes": 1},
        ], source="t")
        assert snap["bull_ratio"] == 0.5
        assert snap["bull_ratio_vote_weighted"] > 0.9
        assert snap["weight_gap"] > 0.4
        assert any("高赞" in n for n in snap["notes"])

    def test_missing_heat_is_none_not_zero(self):
        """没有热度字段时不能按 0 算 —— 否则等于说"所有帖子一样重要"。"""
        snap = S.build_snapshot(_items("满仓", "清仓"), source="t")
        assert snap["bull_ratio_vote_weighted"] is None
        assert snap["weight_gap"] is None

    def test_time_window_uses_real_stamps(self):
        snap = S.build_snapshot([
            {"text": "满仓", "published": 1761200000},
            {"text": "清仓", "published": "2026-10-23 09:12:33"},
        ], source="t")
        assert snap["window_start"] and snap["window_end"]
        assert snap["ts_missing"] == 0

    def test_unparsable_time_is_not_now(self):
        """⚠️ 时间解析失败必须记缺失，**不能**用当前时间冒充（会把旧帖洗成新帖）。"""
        snap = S.build_snapshot([{"text": "满仓", "published": "昨天下午"}], source="t")
        assert snap["window_start"] is None
        assert snap["ts_missing"] == 1

    def test_samples_include_both_sides(self):
        """⭐ 代表样本必须**多空两边都有**。

        事故现场：原实现按顺序取前 4 条，实测取到的全是 bull ⇒ 模型看不到
        任何反方原文，只能复述一个可能有误判的比例，无从纠错。
        """
        texts = ["满仓"] * 10 + ["清仓"] * 6
        snap = S.build_snapshot(_items(*texts), source="t")
        stances = {s["stance"] for s in snap["samples"]}
        assert stances == {"bull", "bear"}, f"样本应含两侧，实得 {stances}"

    def test_samples_are_alternating_not_blocked(self):
        """交替而非分块：前两条就该是 1 多 1 空（让模型第一眼看到对立）。"""
        snap = S.build_snapshot(_items(*(["满仓"] * 6 + ["清仓"] * 6)), source="t")
        assert [s["stance"] for s in snap["samples"][:2]] == ["bull", "bear"]

    def test_sample_limit_is_capped(self):
        snap = S.build_snapshot(_items(*(["满仓"] * 30 + ["清仓"] * 30)), source="t")
        assert len(snap["samples"]) == S.SAMPLE_LIMIT

    def test_rule_fallibility_warning_is_always_present(self):
        """⚠️ 「规则可能误判」这句话必须随数字一起走 —— 否则模型会把它当测量真值。"""
        snap = S.build_snapshot(_items(*(["满仓"] * 10 + ["清仓"] * 10)), source="t")
        assert any("以原文为准" in n for n in snap["notes"])


class TestWilsonInterval:
    def test_zero_n_returns_full_range(self):
        assert S._wilson(0.0, 0) == (0.0, 1.0)

    def test_interval_stays_in_bounds(self):
        for p in (0.0, 0.01, 0.5, 0.99, 1.0):
            lo, hi = S._wilson(p, 10)
            assert 0.0 <= lo <= hi <= 1.0


# ================================================================
# ③ 相对层
# ================================================================

class TestCompareBaseline:
    def test_insufficient_history_gives_no_numbers(self):
        """⭐ 历史不足 ⇒ 所有对比字段为 None，绝不拿 2 天算"历史高位"。"""
        snap = S.build_snapshot(_items(*(["满仓"] * 10 + ["清仓"] * 10)), source="t")
        hist = [{"bull_ratio": 0.5, "n_total": 10} for _ in range(3)]
        out = S.compare_baseline(snap, hist)
        assert out["status"] == "insufficient"
        assert out["bull_ratio_ma"] is None
        assert out["bull_ratio_pct_rank"] is None
        assert out["delta"] is None
        assert "门槛" in out["note"]

    def test_ok_path_computes_rank_and_delta(self):
        snap = {"bull_ratio": 0.8, "n_total": 20}
        hist = [{"bull_ratio": v, "n_total": 20} for v in (0.2, 0.3, 0.4, 0.5, 0.6)]
        out = S.compare_baseline(snap, hist)
        assert out["status"] == "ok"
        assert out["available_days"] == 5
        assert out["bull_ratio_ma"] == pytest.approx(0.4)
        assert out["bull_ratio_pct_rank"] == 1.0, "0.8 高于全部历史 ⇒ 100 分位"
        assert out["delta"] == pytest.approx(0.4)

    def test_no_current_ratio_returns_no_data(self):
        out = S.compare_baseline({"bull_ratio": None}, [{"bull_ratio": 0.5}] * 10)
        assert out["status"] == "no_data"
        assert out["delta"] is None

    def test_volume_ratio_needs_enough_days(self):
        snap = {"bull_ratio": 0.5, "n_total": 100}
        hist = [{"bull_ratio": 0.5, "n_total": 10}] * 5
        out = S.compare_baseline(snap, hist)
        assert out["volume_ratio"] == pytest.approx(10.0)
        # 只有 2 天数据时不给倍数
        out2 = S.compare_baseline(snap, hist[:2])
        assert out2["volume_ratio"] is None


# ================================================================
# ④ 判定校验（不调 LLM，直接喂固定文本）
# ================================================================

class TestParseJudgement:
    _ok_snap = {"quality": "ok", "samples": [{"id": "S1"}, {"id": "S2"}]}

    def test_valid_reply_passes(self):
        raw = ('{"stance":"情绪偏多","summary":"看多占优","divergence":"无价量数据，不判断",'
               '"confidence":"中","evidence_ids":["S1"],"cautions":[]}')
        out = S.parse_judgement(raw, self._ok_snap)
        assert out["status"] == "ok"
        assert out["stance"] == "情绪偏多"
        assert out["evidence"] == ["S1"]

    def test_unknown_stance_falls_back(self):
        raw = '{"stance":"看多","summary":"x","confidence":"高","evidence_ids":[]}'
        out = S.parse_judgement(raw, self._ok_snap)
        assert out["stance"] == "数据不足"
        assert any("不在枚举" in w for w in out["warnings"])

    def test_fabricated_evidence_ids_are_dropped(self):
        raw = '{"stance":"情绪偏多","summary":"x","confidence":"中","evidence_ids":["S1","S99"]}'
        out = S.parse_judgement(raw, self._ok_snap)
        assert out["evidence"] == ["S1"]
        assert any("S99" in w for w in out["warnings"])

    def test_thin_sample_forces_low_confidence(self):
        """⭐ 最重要的一条：模型自称「高」，样本薄时也必须被压成「低」。

        这是"小样本 + 高置信度"唯一的结构性拦截点 —— 靠提示词要求模型自觉
        是不可靠的（模型经常不读 quality 字段）。
        """
        thin = {"quality": "thin", "samples": [{"id": "S1"}]}
        raw = '{"stance":"情绪偏多","summary":"x","confidence":"高","evidence_ids":["S1"]}'
        out = S.parse_judgement(raw, thin)
        assert out["confidence"] == "低"
        assert any("强制降为" in w for w in out["warnings"])

    def test_non_json_reply_is_reported(self):
        out = S.parse_judgement("我觉得市场情绪不错", self._ok_snap)
        assert out["status"] == "llm_error"
        assert out["confidence"] == "低"


# ================================================================
# ⑤ 参照块
# ================================================================

class TestRenderBlock:
    def test_block_is_delimited_and_carries_rules(self):
        snap = S.build_snapshot(_items(*(["满仓"] * 10 + ["清仓"] * 10)), source="t")
        block = S.render_block(snap, S.compare_baseline(snap, []))
        assert block.startswith(S.SENTIMENT_BLOCK_START)
        assert block.endswith(S.SENTIMENT_BLOCK_END)
        # 引用规则必须随块走 —— 只在系统提示里写，长对话里会被忽略
        assert "不得" in block and "买卖建议" in block

    def test_empty_snapshot_says_no_data_explicitly(self):
        """⚠️ 空数据必须写"无数据"，不能省略该行让模型以为"一切正常"。"""
        snap = S.build_snapshot([], source="t")
        block = S.render_block(snap)
        assert "无数据" in block

    def test_thin_sample_warning_is_rendered_next_to_the_number(self):
        snap = S.build_snapshot(_items("满仓", "清仓"), source="t")
        block = S.render_block(snap)
        assert "样本偏薄" in block
        assert "门槛" in block
