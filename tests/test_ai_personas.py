# -*- coding: utf-8 -*-
"""人格注册表（core/ai_personas.py）的测试。

⚠️ 全部走 tmp_path 隔离（PERSONAS_PATH 被 monkeypatch 到用例级文件），
   绝不读写真实的 config/ai_personas.json；ai_personas 表走 use_template_db。
"""
import json

import pytest

from tests.conftest import use_template_db
from core import ai_personas as AP


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(AP, "PERSONAS_PATH", tmp_path / "ai_personas.json")
    AP.invalidate()
    yield
    AP.invalidate()


class TestDefaults:
    def test_missing_file_materializes_defaults(self):
        """首次读取应把默认值落盘——让「可编辑」对用户可见，而不是只活在内存里。"""
        data = AP.load_personas()
        assert (AP.PERSONAS_PATH).exists()
        names = {p["name"] for p in data}
        # 用户决策（2026-10-14）：3 个 LLM 风格 + 3 个随机对照组
        assert {"ai_value", "ai_tech", "ai_risk"} <= names
        assert {"ai_random_1", "ai_random_2", "ai_random_3"} <= names

    def test_llm_vs_random_split(self):
        assert {p["name"] for p in AP.llm_personas()} == {"ai_value", "ai_tech", "ai_risk"}
        assert {p["name"] for p in AP.random_personas()} == {"ai_random_1", "ai_random_2",
                                                             "ai_random_3"}

    def test_random_personas_carry_no_model(self):
        """随机对照组不调 LLM ⇒ 绝不能带 model（否则把 LLM 成本算到对照组头上）。"""
        assert all(p["model"] is None for p in AP.random_personas())

    def test_style_prompts_declare_rule_boundaries(self):
        """风格提示词必须显式声明"不得违反交易纪律"——风格不是豁免权（C2 的红线）。"""
        for p in AP.llm_personas():
            assert "交易纪律" in p["style_prompt"], f"{p['name']} 的风格提示词缺纪律声明"

    def test_user_model_assignment_respected(self):
        """用户决策（2026-10-19 全流程切 DeepSeek）：三个 LLM persona 均为 deepseek。

        ⚠️ 这是**默认值**断言：用户改 JSON 换模型不应被本测试挡住（改的是值，不是结构）。
        但**必须挡住回退到 qwen**——qwen 免费额度已耗尽，回退会让 AI 联赛整天静默失败。
        """
        by = {p["name"]: p for p in AP.load_personas()}
        for name in ("ai_value", "ai_tech", "ai_risk"):
            assert by[name]["model"].startswith("deepseek"), (name, by[name]["model"])


class TestSanitize:
    def test_bad_entries_skipped_not_fatal(self):
        raw = [
            {"name": "", "label": "无名"},                 # 缺 name → 丢
            "not-a-dict",                                  # 非字典 → 丢
            {"name": "dup", "label": "1"},                 # 重复 → 只留第一个
            {"name": "dup", "label": "2"},
            {"name": "ok"},                                # 缺 style_prompt → **保留**（空串合法：随机对照组就没有风格）
            {"name": "good", "label": "G", "style_prompt": "s",
             "kind": "alien", "risk_bias": "wild", "frequency": "hourly",
             "model": "  "},
        ]
        out = AP._sanitize(raw)
        assert [p["name"] for p in out] == ["dup", "ok", "good"]
        g = out[-1]
        assert g["kind"] == "llm" and g["risk_bias"] == "balanced"     # 非法枚举 → 默认
        assert g["frequency"] == "daily" and g["model"] is None        # 空白 model → None

    def test_random_kind_ignores_model(self):
        out = AP._sanitize([{"name": "r", "label": "R", "kind": "random",
                             "model": "qwen-max"}])
        assert out[0]["model"] is None

    def test_quota_override_survives_sanitize(self):
        """⚠️ quotas 是 C3 的合法配置字段——被字段白名单洗掉会让配额覆盖静默失效
        （实测踩过：首版 sanitize 没带它，B.consume 的 persona 覆盖全部落空）。"""
        out = AP._sanitize([{"name": "x", "label": "X", "style_prompt": "s",
                             "quotas": {"deep_analysis": 1, "bad": "no-int"}}])
        assert out[0]["quotas"] == {"deep_analysis": 1}


class TestCorruption:
    def test_broken_json_falls_back_to_defaults(self):
        """损坏 → **降级到默认值**并留日志（宁可回到已知好的状态，不用坏数据跑交易）。"""
        AP.PERSONAS_PATH.write_text("{not json", encoding="utf-8")
        AP.invalidate()
        data = AP.load_personas()
        assert {"ai_value", "ai_risk"} <= {p["name"] for p in data}

    def test_empty_list_rejected_on_save(self):
        """拒绝把注册表写成空——宁可保留旧文件，不让调度器醒来无事可做。"""
        assert AP.save_personas([]) is False

    def test_top_level_not_list_raises(self):
        """非数组 → _sanitize 抛 ValueError（load 层捕获后降级到默认值）。"""
        with pytest.raises(ValueError):
            AP._sanitize({"name": "x"})


class TestCacheAndStore:
    def test_mtime_cache_invalidates_on_change(self, tmp_path):
        first = AP.load_personas()
        assert AP.get_persona("ai_value") is not None
        # 模拟用户编辑：改文件后必须读到新值（mtime 缓存的全部意义）
        edited = [dict(first[0], label="改名了")] + first[1:]
        assert AP.save_personas(edited) is True
        assert AP.get_persona("ai_value")["label"] == "改名了"

    def test_save_persists_sanitized_form(self, tmp_path):
        """⚠️ 存进去的就是将来读出来的（写前 sanitize）——不让脏数据过夜。"""
        AP.save_personas([{"name": "x", "label": "X", "style_prompt": "s",
                           "kind": "weird", "model": "  "}])
        on_disk = json.loads(AP.PERSONAS_PATH.read_text(encoding="utf-8"))
        assert on_disk[0]["kind"] == "llm" and on_disk[0]["model"] is None

    def test_sync_to_store(self, tmp_path, monkeypatch):
        use_template_db(tmp_path, monkeypatch, "personas.db")
        n = AP.sync_to_store()
        assert n == len(AP.load_personas())
        from db import store
        rows = store.list_ai_personas(enabled_only=False)
        assert {r["name"] for r in rows} >= {p["name"] for p in AP.load_personas()}
        # 幂等：再跑一次不炸、不重复
        assert AP.sync_to_store() == n
        assert len(store.list_ai_personas(enabled_only=False)) == n
