# -*- coding: utf-8 -*-
"""LLM 厂商路由的契约（2026-10-19 建立，2026-10-20 扩展为多厂商）。

为什么需要这组测试：
    qwen 免费额度耗尽时，一次"某个功能还是 qwen"的遗漏不会报错——
    它只会**静默失败**（AI 联赛当天少跑、复盘少一段），很难被发现。
    所以把"每个功能都解析到**可用**厂商"变成可断言的结构事实。

⚠️ 2026-10-20 起不再要求"全部 DeepSeek"：深度分析的写作型节点
（技术面/基本面/舆情/多空）改用 qwen3.7-flash（同输入对照实测输出长度比 0.99、
结构覆盖 10/10、单只成本降约 70%）。但**决策与结构化输出节点必须留在 DeepSeek**
——见 `TestEveryFeatureResolvesToLiveProvider::test_required_nodes_stay_on_deepseek`。

另外锁住思考模式开关：DeepSeek V4 / qwen3.x **默认都开启** thinking。
实测 qwen3.7-flash 同一句问话：关闭 25 输出 token / 开启 992（**34 倍**）。
不显式关闭，成本与延迟都会失控——所以 `_thinking_kwargs` 必须按厂商分别处理。
"""
import json
import os
import pathlib

import pytest

from config import model_routing as MR

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _fresh_routing(monkeypatch):
    """确保读到的是仓库里的映射文件（不受别的用例/BIGA_LLM_FEATURE_MODELS 影响）。"""
    monkeypatch.delenv(MR.ENV_ROUTING_PATH, raising=False)
    MR.invalidate()
    yield
    MR.invalidate()


class TestEveryFeatureResolvesToLiveProvider:
    """⭐ 守护的**真实意图**不是"必须是 DeepSeek"，而是"不许静默落到默认厂商"。

    2026-10-19 的版本断言"每个功能都解析到 deepseek-*"——当时确实只有 deepseek
    可用（qwen 额度耗尽）。2026-10-20 起深度分析的写作型节点改用 qwen3.7-flash，
    于是"必须 deepseek"这个断言过时了；但它要防的事故没有变：
    **配了一个未注册/未启用的模型名 ⇒ 被默认厂商接走 ⇒ 静默失败**。

    所以改为断言：每个功能解析出的模型，都必须命中一个 `enabled=True`
    且声明了 `api_key_env` 的厂商。
    """

    def test_every_feature_resolves_to_a_live_provider(self):
        from config.providers import provider_for_model
        bad = {}
        for f in MR.FEATURES:
            m = MR.resolve_model(f)
            p = provider_for_model(m) or {}
            if not p.get("id"):
                bad[f] = f"{m}（无厂商匹配 ⇒ 会被默认厂商接走）"
            elif not p.get("enabled", True):
                bad[f] = f"{m}（厂商 {p['id']} 未启用）"
            elif not p.get("api_key_env"):
                bad[f] = f"{m}（厂商 {p['id']} 没声明 api_key_env）"
        assert not bad, f"以下功能的模型没有可用的厂商（会静默失败）: {bad}"

    def test_required_nodes_stay_on_deepseek(self):
        """决策链与结构化输出节点**必须**留在 DeepSeek。

        这几处错了的代价不是"报告写得差"，而是：trade_plan 解析失败、仓位/价位
        记错、JSON 违约（都要靠 `_normalize_trade_plan` 静默兜底）。
        """
        must = ("agents.research_manager", "agents.trader",
                "agents.risk_manager", "agents.portfolio_manager",
                "nl_screener.parse", "analysis.review")
        bad = {f: MR.resolve_model(f) for f in must
               if not str(MR.resolve_model(f)).startswith("deepseek")}
        assert not bad, f"决策/结构化节点被改到了非 DeepSeek 模型: {bad}"

    def test_feature_map_covers_all_registered_features(self):
        """映射文件必须覆盖全部功能——漏一个就会落到档位默认（历史上= qwen-max）。"""
        mapped = set(MR.get_feature_models())
        missing = set(MR.FEATURES) - mapped
        assert not missing, f"映射缺失（会回落档位默认）: {sorted(missing)}"

    def test_feature_map_has_no_unknown_keys(self):
        """映射里不能有拼错/已废弃的功能键（否则是"配了但没生效"的静默坑）。"""
        unknown = set(MR.get_feature_models()) - set(MR.FEATURES)
        assert not unknown, f"未知功能键: {sorted(unknown)}"

    def test_personas_are_deepseek(self):
        raw = json.loads((ROOT / "config" / "ai_personas.json").read_text(encoding="utf-8"))
        llm = [p for p in raw if p.get("kind") == "llm" and p.get("model")]
        assert llm, "至少应有一个 LLM persona"
        bad = {p["name"]: p["model"] for p in llm
               if not str(p["model"]).startswith("deepseek")}
        assert not bad, f"persona 仍用非 DeepSeek 模型: {bad}"


class TestThinkingModeSwitch:
    """DeepSeek V4 默认 thinking=enabled；本项目默认**关闭**。"""

    def test_v4_models_get_thinking_disabled_by_default(self, monkeypatch):
        from llm.deepseek_client import _thinking_kwargs
        monkeypatch.delenv("DEEPSEEK_THINKING", raising=False)
        assert _thinking_kwargs("deepseek-v4-flash") == {
            "extra_body": {"thinking": {"type": "disabled"}}}
        assert _thinking_kwargs("deepseek-v4-pro") == {
            "extra_body": {"thinking": {"type": "disabled"}}}

    def test_env_can_enable_thinking(self, monkeypatch):
        from llm.deepseek_client import _thinking_kwargs
        monkeypatch.setenv("DEEPSEEK_THINKING", "1")
        assert _thinking_kwargs("deepseek-v4-flash")["extra_body"]["thinking"]["type"] == "enabled"
        # 显式参数优先于环境变量（单次覆盖）
        assert _thinking_kwargs("deepseek-v4-flash", False)["extra_body"]["thinking"]["type"] == "disabled"

    def test_non_v4_models_untouched(self, monkeypatch):
        """deepseek 别名（chat/reasoner）与空模型名不得注入该参数。

        ⚠️ 2026-10-20 起 qwen/glm 被视为"支持思考开关"的模型（各自的参数形态
        见 `TestQwenAndGlmThinking`），因此不再列入"完全不注入"的清单。
        """
        from llm.deepseek_client import _thinking_kwargs
        monkeypatch.delenv("DEEPSEEK_THINKING", raising=False)
        for m in ("deepseek-chat", "deepseek-reasoner", ""):
            assert _thinking_kwargs(m) == {}, m


class TestQwenAndGlmThinking:
    """qwen / glm 的关闭参数与 DeepSeek **不同形**，且服务端默认都是开启。

    实测（2026-10-20，qwen3.7-flash，同一句问话）：
        关闭 25 输出 token / 1.7s；开启 992 / 16.4s；不传 858 / 15.7s。
    按 0.2/0.8 元/百万算，单只深度分析（62k 入 / 17.8k 出）：关闭 ¥0.027，
    开着输出涨 34 倍 ⇒ 约 ¥0.50，**比留在 DeepSeek flash（¥0.17）还贵 3 倍**。
    所以这条必须锁死——"换便宜模型不关思考，等于白换"。
    """

    def test_qwen_uses_top_level_enable_thinking(self, monkeypatch):
        """qwen 的参数是请求体顶层 `enable_thinking`（不是 thinking.type）。

        ⚠️ 第一版探测把它当成 OpenAI SDK 的 `extra_body` 语义直接写进 body，
        服务端**静默忽略**，输出仍是 744 token——"配置看起来生效"最难发现。
        """
        from llm.deepseek_client import _thinking_kwargs
        monkeypatch.delenv("DEEPSEEK_THINKING", raising=False)
        for m in ("qwen3.7-flash", "qwen3.8-flash", "qwen-plus", "qwen-max"):
            assert _thinking_kwargs(m) == {"extra_body": {"enable_thinking": False}}, m
        # 单次显式开启（排查"是不是被关了"时才用）
        assert _thinking_kwargs("qwen3.7-flash", True)["extra_body"]["enable_thinking"] is True
        # 环境变量同样能开
        monkeypatch.setenv("DEEPSEEK_THINKING", "1")
        assert _thinking_kwargs("qwen3.7-flash")["extra_body"]["enable_thinking"] is True

    def test_glm_uses_thinking_type(self, monkeypatch):
        from llm.deepseek_client import _thinking_kwargs
        monkeypatch.delenv("DEEPSEEK_THINKING", raising=False)
        assert _thinking_kwargs("glm-4.7-flash") == {
            "extra_body": {"thinking": {"type": "disabled"}}}


class TestProviderRouting:
    def test_v4_models_resolve_to_deepseek_provider(self):
        """v4 模型名必须能路由到 deepseek 厂商（新环境无本地清单时也要成立）。"""
        from config.providers import provider_for_model
        for m in ("deepseek-v4-flash", "deepseek-v4-pro"):
            assert provider_for_model(m)["id"] == "deepseek", m

    def test_builtin_declares_v4_models(self):
        """内置清单要列出 v4 名字，否则 UI 下拉里选不到（只能靠通配符路由）。"""
        from config.providers import DEFAULT_PROVIDERS
        ds = next(p for p in DEFAULT_PROVIDERS if p["id"] == "deepseek")
        assert "deepseek-v4-flash" in ds["models"]
        assert "deepseek-v4-pro" in ds["models"]
