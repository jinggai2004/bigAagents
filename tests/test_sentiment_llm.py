# -*- coding: utf-8 -*-
"""LLM 逐条标注层：**数字由代码算、标签必须可核**。

这一层最容易被做错的地方不是"调不通模型"，而是**调通了却拿到不可信的数字**：

1. 模型自己报百分比 → 与代码算的对不上，且无法归因。
2. 模型没读原文，按标题编立场 → 看起来像标注结果，实际是幻觉。
3. 模型只标了一半 → 剩下的被"默认中性"补齐，覆盖率虚高。
4. 换标注器后历史混算 → 分位彻底失去意义。

下面每个类对应拦住上面一条。
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from core import sentiment_llm as SL
from core import sentiment_reference as SR


# ================================================================
# 假客户端（绝不联网）
# ================================================================

class FakeClient:
    """按脚本逐次返回；`Exception` 实例表示这次调用失败。"""

    model = "fake-model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def call_api(self, messages, temperature=None, purpose=None, **kw):
        self.calls.append({"messages": messages, "temperature": temperature,
                           "purpose": purpose})
        if not self.replies:
            raise RuntimeError("脚本已用完")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _reply(labels, report=None) -> str:
    return json.dumps({"labels": labels, "report": report or {}}, ensure_ascii=False)


def _row(text: str, url: str = None, votes: int = 0) -> dict:
    d = {"text": text, "votes": votes}
    if url:
        d["url"] = url
    return d


# ================================================================
# 一、引用核对：唯一的确定性校验器
# ================================================================

class TestQuoteVerify:
    def test_verbatim_quote_passes(self):
        assert SL.quote_ok("我降了30%的仓位，压力太大", "降了30%的仓位")

    def test_whitespace_difference_still_passes(self):
        """换行/空格差异不算引用不实 —— 否则会把格式问题当成编造。"""
        assert SL.quote_ok("我 降了 30% 的仓位", "降了30%的仓位")

    def test_fabricated_quote_fails(self):
        assert not SL.quote_ok("我降了30%的仓位", "我清仓了全部")

    def test_too_short_quote_fails(self):
        """⚠️ 太短的引用（'满仓'）在长文里必然命中，等于没校验。"""
        assert not SL.quote_ok("满仓了，看好后市", "看好")

    def test_short_text_allows_short_quote(self):
        """短帖放宽：正文本身只有 4 字时，不能要求 4 字以上的引用。"""
        assert SL.quote_ok("满仓", "满仓")

    def test_empty_quote_fails(self):
        assert not SL.quote_ok("满仓了", "")


# ================================================================
# 二、单块解析：脏回复必须被逐项拦下
# ================================================================

class TestParseChunk:
    ROWS = [_row("我降了30%的仓位，满仓满融压力太大"),
            _row("别问，已梭哈完"),
            _row("删帖限流两件套了，明天再杀半天会反弹")]

    def test_labels_are_mapped_by_one_based_index(self):
        raw = _reply([{"i": 1, "stance": "bear", "quote": "降了30%的仓位", "why": "减仓"},
                      {"i": 2, "stance": "bull", "quote": "已梭哈完", "why": "满仓"},
                      {"i": 3, "stance": "bear", "quote": "明天再杀半天", "why": "看跌"}])
        r = SL.parse_chunk(raw, self.ROWS, 0)
        assert r["labels"] == {0: "bear", 1: "bull", 2: "bear"}
        assert not r["rejected"]
        assert r.get("quote_hit_rate") == 1.0

    def test_index_is_offset_by_chunk_start(self):
        """第二块的 `i=1` 必须落到全局下标 `start`，否则整块标签会串位。"""
        raw = _reply([{"i": 1, "stance": "bear", "quote": "降了30%的仓位", "why": "x"}])
        r = SL.parse_chunk(raw, self.ROWS, start=10)
        assert r["labels"] == {10: "bear"}

    def test_enum_is_case_insensitive_but_unknown_becomes_unscored(self):
        raw = _reply([{"i": 1, "stance": "Bull", "quote": "降了30%的仓位"},
                      {"i": 2, "stance": "strong_buy", "quote": "已梭哈完"}])
        r = SL.parse_chunk(raw, self.ROWS, 0)
        assert r["labels"][0] == "bull"          # 'Bull' 归一化
        assert r["labels"][1] == "unscored"      # 自创枚举 ⇒ 不信任
        assert any("不在枚举内" in w for w in r["warnings"])

    def test_out_of_range_index_is_dropped(self):
        raw = _reply([{"i": 9, "stance": "bull", "quote": "已梭哈完"}])
        r = SL.parse_chunk(raw, self.ROWS, 0)
        assert 8 not in r["labels"]
        assert any("越界" in w for w in r["warnings"])

    def test_duplicate_index_keeps_first(self):
        raw = _reply([{"i": 1, "stance": "bull", "quote": "降了30%的仓位"},
                      {"i": 1, "stance": "bear", "quote": "降了30%的仓位"}])
        r = SL.parse_chunk(raw, self.ROWS, 0)
        assert r["labels"][0] == "bull"

    def test_low_quote_hit_rejects_whole_chunk(self):
        """⭐ 引用命中率低于门槛 ⇒ 整块作废（不是降级）。

        这是拦"没读原文就编立场"的唯一机制。若只降级，那些编出来的标签
        会带着 1/3 的可信度混进 `bull_ratio`。
        """
        raw = _reply([{"i": 1, "stance": "bull", "quote": "原文里没有这句"},
                      {"i": 2, "stance": "bear", "quote": "也没有这句"},
                      {"i": 3, "stance": "bull", "quote": "明天再杀半天"}])
        r = SL.parse_chunk(raw, self.ROWS, 0)
        assert r["rejected"] is True
        assert r["labels"] == {}                 # 整块清空
        assert any("整体作废" in w for w in r["warnings"])

    def test_non_json_is_rejected_not_silently_empty(self):
        r = SL.parse_chunk("<html>被风控</html>", self.ROWS, 0)
        assert r["rejected"] is True
        assert r["labels"] == {}
        assert r["warnings"]

    def test_missing_labels_key_is_rejected(self):
        r = SL.parse_chunk(json.dumps({"report": {"主线": "x"}}), self.ROWS, 0)
        assert r["rejected"] is True

    def test_report_is_extracted(self):
        raw = _reply([{"i": 1, "stance": "bull", "quote": "降了30%的仓位"}],
                     report={"主线": "仓位分歧", "多头理由": ["会反弹"]})
        r = SL.parse_chunk(raw, self.ROWS, 0)
        assert r["report"]["主线"] == "仓位分歧"

    def test_markdown_fence_is_tolerated(self):
        raw = "好的，结果如下：\n```json\n" + _reply(
            [{"i": 1, "stance": "bull", "quote": "降了30%的仓位"}]) + "\n```"
        r = SL.parse_chunk(raw, self.ROWS, 0)
        assert r["labels"] == {0: "bull"}


# ================================================================
# 三、报告里的自报统计必须被丢弃
# ================================================================

class TestStripNumbers:
    def test_arabic_percentage_is_dropped(self):
        out, w = SL._strip_numbers({"主线": "70% 的人看多", "多头理由": []})
        assert out["主线"] == ""
        assert w

    def test_chinese_numeral_is_dropped(self):
        """⚠️ 自检时发现的真实漏洞：模型更爱写「约七成看多」。

        只匹配阿拉伯数字的规则会让这条**直接进报告**，而页面旁边就是
        代码算出的"多空比 45%"，读者无从判断该信哪个。
        """
        out, w = SL._strip_numbers({"主线": "x", "异常信号": ["约七成看多"]})
        assert out["异常信号"] == []
        assert w

    def test_warning_does_not_echo_the_dropped_claim(self):
        """⚠️ 告警里也不许复述被删的内容。

        初版写成 `已丢弃：约七成看多`，结果那句话换个位置又出现在页面上，
        等于没删。只报字段名。
        """
        _, w = SL._strip_numbers({"主线": "八成看多", "多头理由": ["七成看多", "会反弹"]})
        joined = "".join(w)
        assert "八成看多" not in joined
        assert "七成看多" not in joined
        assert w

    def test_qualitative_words_are_kept(self):
        """「多数人看空」是定性叙述，不是第二套精确口径 ⇒ 保留。"""
        out, _ = SL._strip_numbers({"主线": "多数人在减仓", "空头理由": ["多数人看空"]})
        assert out["主线"] == "多数人在减仓"
        assert out["空头理由"] == ["多数人看空"]

    def test_non_dict_input_is_safe(self):
        out, w = SL._strip_numbers(None)
        assert out == {} and w == []


# ================================================================
# 四、整体标注：分批、失败隔离、缺标不补
# ================================================================

class TestLabelPosts:
    def test_chunks_by_configured_size(self, monkeypatch):
        monkeypatch.setattr(SL, "LABEL_CHUNK", 2, raising=False)
        items = [_row("满仓了，看好", f"u{i}") for i in range(5)]
        raw = _reply([{"i": 1, "stance": "bull", "quote": "满仓了，看好"}])
        c = FakeClient([raw, raw, raw])
        out = SL.label_posts(items, client=c)
        assert out["meta"]["chunks"] == 3
        assert out["meta"]["n_labeled"] == 3

    def test_failed_chunk_leaves_labels_none_not_unscored(self):
        """⭐ 失败块必须留 `None`。

        若补成 `unscored`，它会进入分母旁边的"未表态"统计，看起来像
        "模型读了但认为无方向" —— 而实际是"根本没读到"。
        """
        items = [_row("满仓了，看好", "u1"), _row("清仓了", "u2")]
        c = FakeClient([RuntimeError("boom")])
        out = SL.label_posts(items, client=c)
        assert out["labels"] == [None, None]
        assert out["meta"]["failed_chunks"] == 1
        assert out["meta"]["n_labeled"] == 0

    def test_partial_labels_keep_none_for_missing(self):
        items = [_row("满仓了，看好", "u1"), _row("清仓了，不看好", "u2")]
        c = FakeClient([_reply([{"i": 1, "stance": "bull", "quote": "满仓了，看好"}])])
        out = SL.label_posts(items, client=c)
        assert out["labels"][0] == "bull"
        assert out["labels"][1] is None

    def test_uses_registered_feature_purpose(self):
        """purpose 必须已登记，否则会落到档位默认模型（历史上 = qwen-max）。"""
        import config.model_routing as MR
        c = FakeClient([_reply([{"i": 1, "stance": "bull", "quote": "满仓了，看好"}])])
        SL.label_posts([_row("满仓了，看好")], client=c)
        assert c.calls[0]["purpose"] == "sentiment.label"
        assert "sentiment.label" in MR.FEATURES

    def test_temperature_is_zero(self):
        """标注必须可复现：同一批帖子两次跑出不同方向，历史序列就是噪声。"""
        c = FakeClient([_reply([{"i": 1, "stance": "bull", "quote": "满仓了，看好"}])])
        SL.label_posts([_row("满仓了，看好")], client=c)
        assert c.calls[0]["temperature"] == 0.0

    def test_max_posts_caps_input(self, monkeypatch):
        """⚠️ 这条同时守着"默认参数在定义时求值"这个坑。

        若把签名写成 `def label_posts(..., max_posts=MAX_POSTS)`，
        这里的 monkeypatch 就完全失效（表现为"配了但不生效"）。
        初版正是这么写的，被这条用例逮住。
        """
        monkeypatch.setattr(SL, "MAX_POSTS", 3, raising=False)
        items = [_row("满仓了，看好", f"u{i}") for i in range(10)]
        c = FakeClient([_reply([{"i": 1, "stance": "bull", "quote": "满仓了，看好"}])])
        out = SL.label_posts(items, client=c)
        assert out["meta"]["n_posts"] == 3
        assert len(out["labels"]) == 3

    def test_empty_items_does_not_call_model(self):
        c = FakeClient([])
        out = SL.label_posts([], client=c)
        assert out["meta"]["n_posts"] == 0
        assert out["meta"]["errors"]
        assert not c.calls

    def test_single_chunk_report_skips_merge(self):
        """只有一块时不该再花一次调用去"合并一份报告"。"""
        c = FakeClient([_reply([{"i": 1, "stance": "bull", "quote": "满仓了，看好"}],
                               report={"主线": "看多"})])
        out = SL.label_posts([_row("满仓了，看好")], client=c)
        assert len(c.calls) == 1
        assert out["report"]["主线"] == "看多"

    def test_withheld_labeler_when_coverage_below_half(self, monkeypatch):
        monkeypatch.setattr(SL, "LABEL_CHUNK", 10, raising=False)
        items = [_row("满仓了，看好", f"u{i}") for i in range(10)]
        c = FakeClient([_reply([{"i": 1, "stance": "bull", "quote": "满仓了，看好"},
                                {"i": 2, "stance": "bull", "quote": "满仓了，看好"}])])
        out = SL.label_posts(items, client=c)
        assert out["meta"]["withheld_labeler"] is True


# ================================================================
# 五、外部标签进指标卡：**不回落词表**
# ================================================================

class TestBuildSnapshotWithLabels:
    ITEMS = [_row("满仓了，看好"), _row("清仓了，不看好")]

    def test_rule_layer_would_say_bull_but_label_missing_stays_unscored(self):
        """⭐ 最关键的断言：缺标**不得**回落词表。

        词表会把「满仓了，看好」判成 bull。若 labels 里给 `None` 却回落，
        这份快照就混了两个标注器，`bull_ratio` 既不是 A 也不是 B。
        """
        assert SR.classify_stance("满仓了，看好")["stance"] == "bull"
        snap = SR.build_snapshot(self.ITEMS, source="t",
                                 labels=[None, {"stance": "bear"}], labeler="llm:x")
        assert snap["bull"] == 0
        assert snap["bear"] == 1
        assert snap["unscored"] == 1
        assert snap["label_missing"] == 1
        assert snap["labeler"] == "llm:x"

    def test_invalid_stance_becomes_unscored(self):
        snap = SR.build_snapshot(self.ITEMS, source="t",
                                 labels=[{"stance": "强烈看多"}, {"stance": "bear"}],
                                 labeler="llm:x")
        assert snap["unscored"] == 1 and snap["bear"] == 1

    def test_labeler_defaults(self):
        assert SR.build_snapshot(self.ITEMS, source="t")["labeler"] == "rule"
        assert SR.build_snapshot(self.ITEMS, source="t",
                                 labels=["bull", "bear"])["labeler"] == "external"

    def test_note_is_labeler_aware(self):
        snap = SR.build_snapshot(self.ITEMS, source="t",
                                 labels=["bull", "bear"], labeler="llm:x")
        assert any("模型逐条标注" in n for n in snap["notes"])
        assert not any("词表规则判定" in n for n in snap["notes"])

    def test_quote_miss_is_counted_in_notes(self):
        snap = SR.build_snapshot(self.ITEMS, source="t",
                                 labels=[{"stance": "bull", "quote": "编的", "quote_ok": False},
                                         {"stance": "bear", "quote_ok": True}],
                                 labeler="llm:x")
        assert any("引用未能逐字命中" in n for n in snap["notes"])


# ================================================================
# 六、双轨对照与报告一致性
# ================================================================

class TestAgreement:
    def test_same_rate_only_over_both_directional(self):
        """⚠️ 分母只能是"两边都判出方向"的条数。

        拿总条数当分母，会把"词表判不出、模型判出了"算成"不一致"，
        一致率被系统性压低，看起来像两个标注器严重冲突。
        """
        rows = [_row("a"), _row("b"), _row("c"), _row("d")]
        a = SL.agreement(["bull", "bear", "neutral", "bull"],
                         ["bull", "bull", "bear", None], rows)
        assert a["n_both_directional"] == 2      # 只有前两条两边都判了方向
        assert a["same_rate"] == 0.5
        assert a["rule_only"] == 1               # d：词表 bull、模型 None
        assert a["llm_only"] == 1                # c：词表 neutral、模型 bear
        assert len(a["conflicts"]) == 1
        assert a["conflicts"][0]["rule"] == "bear"

    def test_no_directional_overlap_gives_none_not_zero(self):
        a = SL.agreement(["neutral"], ["bear"], [_row("x")])
        assert a["same_rate"] is None


class TestReportConsistency:
    def test_contradiction_is_flagged(self):
        w = SL.report_consistency({"主线": "普遍看空"}, bull_ratio=0.8,
                                  n_scored=SR.MIN_SCORED_OK)
        assert w and "方向相反" in w[0]

    def test_no_warning_when_aligned(self):
        assert not SL.report_consistency({"主线": "情绪偏多"}, 0.8, SR.MIN_SCORED_OK)

    def test_thin_sample_never_flags(self):
        """样本不足时"叙述偏空、比例 0.8"不是矛盾，是噪声。"""
        assert not SL.report_consistency({"主线": "普遍看空"}, 0.8, 5)

    def test_missing_ratio_never_flags(self):
        assert not SL.report_consistency({"主线": "普遍看空"}, None, 40)


# ================================================================
# 七、持久化：口径必须能分开读
# ================================================================

@pytest.fixture
def _db(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "sentiment_llm.db")


def _snap(date, ratio, source="t", labeler="rule"):
    return {"source": source, "as_of": f"{date} 15:00:00", "labeler": labeler,
            "n_total": 40, "n_scored": 20, "bull": int(20 * ratio),
            "bear": 20 - int(20 * ratio), "neutral": 0, "unscored": 20,
            "bull_ratio": ratio, "dispersion": 0.3, "quality": "ok",
            "notes": [], "samples": []}


class TestLabelerPersistence:
    def test_history_filters_by_labeler(self, _db):
        """⭐ 跨口径不能混算：词表比例与模型比例不是同一个量。"""
        from db import store
        store.save_sentiment_snapshot(_snap("2026-10-20", 0.6, labeler="rule"))
        store.save_sentiment_snapshot(_snap("2026-10-21", 0.2, labeler="llm:m"))
        r = store.get_sentiment_history("t", days=10, labeler="rule")
        assert [h["bull_ratio"] for h in r] == [0.6]
        r2 = store.get_sentiment_history("t", days=10, labeler="llm:m")
        assert [h["bull_ratio"] for h in r2] == [0.2]

    def test_legacy_row_without_labeler_counts_as_rule(self, _db):
        """升级前的行 labeler 为 NULL ⇒ 必须算作 rule，否则历史凭空清零。"""
        from db import store
        from db.models import get_connection
        store.save_sentiment_snapshot(_snap("2026-10-20", 0.6))
        conn = get_connection()
        conn.execute("UPDATE sentiment_snapshots SET labeler = NULL")
        conn.commit()
        conn.close()
        assert len(store.get_sentiment_history("t", days=10, labeler="rule")) == 1

    def test_labels_roundtrip_and_idempotent(self, _db):
        from db import store
        rows = [{"trade_date": "2026-10-20", "source": "t", "post_key": "u:1",
                 "labeler": "llm:m", "stance": "bull", "quote": "满仓",
                 "quote_ok": True, "why": "w", "text": "满仓了"}]
        assert store.save_sentiment_labels(rows) == 1
        rows[0]["stance"] = "bear"
        assert store.save_sentiment_labels(rows) == 1
        got = store.get_sentiment_labels("2026-10-20", source="t")
        assert len(got) == 1 and got[0]["stance"] == "bear"

    def test_missing_stance_is_null_not_unscored(self, _db):
        """⚠️ "没标到" 存 NULL；"标为中性" 存 'neutral'。混了就算不出覆盖率。"""
        from db import store
        store.save_sentiment_labels([{
            "trade_date": "2026-10-20", "source": "t", "post_key": "u:2",
            "labeler": "llm:m", "stance": None, "text": "x"}])
        got = store.get_sentiment_labels("2026-10-20", source="t")
        assert got[0]["stance"] is None

    def test_invalid_rows_are_skipped(self, _db):
        from db import store
        assert store.save_sentiment_labels([{}, {"trade_date": "2026-10-20"}]) == 0


# ================================================================
# 八、迁移：老库必须补上 labeler 列
# ================================================================

class TestLabelerColumnMigration:
    def test_old_table_gets_column_and_keeps_rows(self, tmp_path, monkeypatch):
        """⚠️ `_ensure_column` 的专用测试。

        本项目已有"给旧表加列永远不生效，而版本号却显示迁移成功"的事故史。
        这里真的建一张**缺列**的表，再走 `init_store`，验证：
        ① 列被补上；② 旧行还在（不能被重建表顺手删掉）。
        """
        import db.models as M
        from db import store

        db_file = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_file))
        conn.executescript("""
        CREATE TABLE sentiment_snapshots (
            trade_date TEXT NOT NULL, source TEXT NOT NULL, as_of TEXT,
            window_start TEXT, window_end TEXT, n_total INTEGER DEFAULT 0,
            n_scored INTEGER DEFAULT 0, bull INTEGER DEFAULT 0, bear INTEGER DEFAULT 0,
            neutral INTEGER DEFAULT 0, bull_ratio REAL, dispersion REAL,
            quality TEXT, payload TEXT, updated_at TEXT,
            PRIMARY KEY (trade_date, source)
        );
        INSERT INTO sentiment_snapshots (trade_date, source, bull_ratio, quality)
        VALUES ('2026-10-19','t',0.55,'ok');
        """)
        conn.commit()
        conn.close()

        monkeypatch.setattr(M, "DB_PATH", str(db_file))
        monkeypatch.setattr(store, "_COLUMNS_CHECKED", False)
        store.init_store(force=True)

        conn = sqlite3.connect(str(db_file))
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sentiment_snapshots)")}
        rows = conn.execute("SELECT trade_date, bull_ratio FROM sentiment_snapshots").fetchall()
        conn.close()
        assert "labeler" in cols
        assert rows == [("2026-10-19", 0.55)]
