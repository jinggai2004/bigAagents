"""多厂商 LLM 接入注册表（2026-09-29）的守护测试。

改造目标：把"厂商路由"从 `llm/deepseek_client._get_client_for_model` 里的硬编码
if/else（`"qwen-"` 前缀 → DashScope，否则 → DeepSeek）外置成可配置的 JSON 注册表，
让接入新厂商（Kimi / GLM / OpenAI / 本地 Ollama / 任何 OpenAI 兼容端点）不需要改代码。

因此本文件分两组，缺一不可：
  A. **兼容性组**：注册表缺失时，路由结果必须与改造前逐条相同 —— 这是回归红线。
     改造这种"把硬编码换成配置"的重构，最典型的翻车方式就是顺手把默认行为改了，
     而用户只要没建配置文件就该感知不到任何区别。
  B. **扩展性组**：加了新厂商之后路由真的切过去 —— 否则"支持多厂商"只是句空话。
"""

import inspect
import json

import pytest

from config import providers as P


# 一个典型的第三方厂商（OpenAI 兼容端点）
KIMI = {
    "id": "kimi",
    "name": "月之暗面 Kimi",
    "base_url": "https://api.moonshot.cn/v1",
    "api_key_env": "MOONSHOT_API_KEY",
    "models": ["moonshot-v1-8k", "kimi-k2-0711-preview"],
    "match": ["moonshot-*", "kimi-*"],
}


@pytest.fixture(autouse=True)
def registry_file(tmp_path, monkeypatch):
    """每个用例一个独立注册表文件。

    同时隔离两样东西：真实 `config/llm_providers.json`（不能被测试污染）
    和模块级 mtime 缓存（否则用例之间会串味，且失败原因难以复现）。
    """
    path = tmp_path / "llm_providers.json"
    monkeypatch.setenv(P.ENV_REGISTRY_PATH, str(path))
    P.invalidate()
    yield path
    P.invalidate()


def _write(path, providers, default="deepseek"):
    """写一份注册表并让缓存失效（P.invalidate 是必须的：同一测试内连续写两次时，
    文件系统 mtime 的分辨率可能不足以区分）。"""
    path.write_text(
        json.dumps({"version": 1, "default_provider": default, "providers": providers},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    P.invalidate()


@pytest.fixture
def _with_keys(monkeypatch):
    """给内置两家厂商填上假密钥，让 provider_ready() 为真。

    用假值而非读真实 .env：测试不该依赖开发者本机配了哪几家密钥，
    否则在没配密钥的机器上会得到一堆与本次改动无关的失败。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    monkeypatch.setenv("QWEN_API_KEY", "sk-test-qwen")
    return monkeypatch


# ============================================================
# A. 兼容性：注册表不存在时，行为必须与改造前逐字一致
# ============================================================

class TestBackwardCompatibility:

    def test_no_file_falls_back_to_builtin_two(self, registry_file):
        assert not registry_file.exists()
        reg = P.get_registry()
        assert [p["id"] for p in reg["providers"]] == ["deepseek", "qwen"]
        assert reg["default_provider"] == "deepseek"

    @pytest.mark.parametrize("model,expected", [
        ("deepseek-chat", "deepseek"),
        ("deepseek-reasoner", "deepseek"),
        ("qwen-plus", "qwen"),
        ("qwen-turbo", "qwen"),
        ("qwen-max", "qwen"),
    ])
    def test_declared_models_route_to_same_provider(self, registry_file, model, expected):
        assert P.resolve_provider(model)["id"] == expected

    @pytest.mark.parametrize("model", ["qwen-next-gen", "QWEN-PLUS"])
    def test_qwen_prefix_still_routes_to_qwen(self, registry_file, model):
        """改造前的判据是"以 qwen- 开头"（大小写不敏感）。
        模型名没写在列表里（如官方新发的 qwen-xxx）也必须继续走 Qwen。"""
        assert P.resolve_provider(model)["id"] == "qwen"

    @pytest.mark.parametrize("model", ["gpt-4o", "glm-4", "claude-3", "totally-unknown"])
    def test_unknown_model_falls_back_to_deepseek(self, registry_file, model):
        """改造前：非 qwen- 前缀一律走 DeepSeek。这条必须保持——
        否则用户填错一个模型名就会得到一个莫名其妙的第三方端点报错。"""
        assert P.resolve_provider(model) is None
        assert P.provider_for_model(model)["id"] == "deepseek"
        assert P.describe_model(model) == "DeepSeek（兜底）"

    def test_client_base_url_identical_to_before(self, registry_file, _with_keys):
        """改造前 base_url 直接取 config.DEEPSEEK_BASE_URL / QWEN_BASE_URL，
        改造后取"环境变量优先、注册表兜底"，结果必须一致。"""
        from config.settings import config
        from llm.deepseek_client import _get_client_for_model, reset_client_cache

        reset_client_cache()
        try:
            got_deepseek = str(_get_client_for_model("deepseek-chat").base_url).rstrip("/")
            got_qwen = str(_get_client_for_model("qwen-plus").base_url).rstrip("/")
        finally:
            reset_client_cache()
        assert got_deepseek == config.DEEPSEEK_BASE_URL.rstrip("/")
        assert got_qwen == config.QWEN_BASE_URL.rstrip("/")

    def test_env_base_url_still_overrides_registry(self, registry_file, monkeypatch):
        """『原有的不变』的核心：.env 里已有的 DEEPSEEK_BASE_URL 必须继续生效，
        不能被注册表里的内置默认值顶掉（用户可能正指着自建代理）。"""
        _write(registry_file, [dict(P.DEFAULT_PROVIDERS[0], base_url="https://wrong.example/v1")])
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://my-proxy.example/v1")
        assert P.provider_base_url(P.get_providers()[0]) == "https://my-proxy.example/v1"

    def test_builtin_defaults_are_not_aliased(self, registry_file):
        """get_registry() 必须返回深拷贝：调用方改了返回值不能污染内置常量。"""
        reg = P.get_registry()
        reg["providers"][0]["base_url"] = "https://mutated.example"
        assert P.DEFAULT_PROVIDERS[0]["base_url"] == "https://api.deepseek.com"


# ============================================================
# B. 扩展性：新增厂商后真的路由过去
# ============================================================

class TestCustomProvider:

    def test_new_provider_takes_over_its_models(self, registry_file, monkeypatch):
        _write(registry_file, P.DEFAULT_PROVIDERS + [KIMI])
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-kimi")

        from llm.deepseek_client import _get_client_for_model, reset_client_cache
        reset_client_cache()
        try:
            client = _get_client_for_model("moonshot-v1-8k")
        finally:
            reset_client_cache()
        assert str(client.base_url).rstrip("/") == "https://api.moonshot.cn/v1"

    @pytest.mark.parametrize("model", ["moonshot-v1-128k", "kimi-latest"])
    def test_wildcard_match_handles_undeclared_models(self, registry_file, model):
        """moon 家新发的模型名不在列表里，也要能靠通配接住。"""
        _write(registry_file, P.DEFAULT_PROVIDERS + [KIMI])
        assert P.resolve_provider(model)["id"] == "kimi"

    def test_exact_match_beats_wildcard(self, registry_file):
        """顺序敏感：精确声明优先于通配。
        否则一个 `match: ["*"]` 的厂商会把其它家声明过的模型名全吃掉。"""
        catch_all = {"id": "a", "base_url": "https://a.example/v1",
                     "api_key_env": "A_KEY", "models": [], "match": ["*"]}
        specific = {"id": "b", "base_url": "https://b.example/v1",
                    "api_key_env": "B_KEY", "models": ["special-model"], "match": []}
        _write(registry_file, [catch_all, specific], default="a")
        assert P.resolve_provider("special-model")["id"] == "b"
        assert P.resolve_provider("anything-else")["id"] == "a"

    def test_disabled_provider_is_skipped(self, registry_file):
        disabled = dict(KIMI, enabled=False)
        _write(registry_file, P.DEFAULT_PROVIDERS + [disabled])
        assert P.resolve_provider("moonshot-v1-8k") is None
        assert P.provider_for_model("moonshot-v1-8k")["id"] == "deepseek"

    def test_default_provider_is_configurable(self, registry_file):
        _write(registry_file, P.DEFAULT_PROVIDERS + [KIMI], default="kimi")
        assert P.default_provider_id() == "kimi"
        # 未命中的模型改走新的默认厂商
        assert P.provider_for_model("gpt-4o")["id"] == "kimi"

    def test_available_models_flags_readiness(self, registry_file, monkeypatch):
        _write(registry_file, P.DEFAULT_PROVIDERS + [KIMI])
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
        monkeypatch.delenv("QWEN_API_KEY", raising=False)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)

        ready = {r["model"]: r["ready"] for r in P.available_models()}
        assert ready["deepseek-chat"] is True
        assert ready["qwen-plus"] is False
        assert ready["moonshot-v1-8k"] is False
        assert P.any_provider_configured() is True

    def test_any_provider_configured_covers_third_party_only(self, registry_file, monkeypatch):
        """用户只用第三家、完全没配 DeepSeek/Qwen 时，启动校验不该拦住他。"""
        _write(registry_file, [KIMI], default="kimi")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("QWEN_API_KEY", raising=False)
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-kimi")
        assert P.any_provider_configured() is True


# ============================================================
# C. 错误语义：与改造前一致（回退 + 建连前报错）
# ============================================================

class TestErrorSemantics:

    def test_missing_key_falls_back_to_default_provider(self, registry_file, monkeypatch):
        """改造前："Qwen API key 未配置，回退到 DeepSeek"。
        保留这条是因为注册表里同时存在"已配"和"还没配密钥"的厂商是常态，
        让链路继续跑 + 日志告警，比整条选股流程报错停摆好。"""
        _write(registry_file, P.DEFAULT_PROVIDERS)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
        monkeypatch.delenv("QWEN_API_KEY", raising=False)

        from llm.deepseek_client import _get_client_for_model, reset_client_cache
        reset_client_cache()
        try:
            client = _get_client_for_model("qwen-plus")
        finally:
            reset_client_cache()
        assert str(client.base_url).rstrip("/") == P.DEFAULT_PROVIDERS[0]["base_url"]

    def test_no_key_anywhere_raises_before_connecting(self, registry_file, monkeypatch):
        """一个密钥都没有时必须在**建连前**抛 ValueError，
        否则会退化成运行时的 AuthenticationError（带一堆无关栈）。"""
        _write(registry_file, [P.DEFAULT_PROVIDERS[0]], default="deepseek")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("QWEN_API_KEY", raising=False)

        from llm.deepseek_client import _get_client_for_model, reset_client_cache
        reset_client_cache()
        try:
            with pytest.raises(ValueError) as ei:
                _get_client_for_model("deepseek-chat")
        finally:
            reset_client_cache()
        # 报错必须能指路：厂商名 + 该填哪个环境变量
        assert "DeepSeek" in str(ei.value)
        assert "DEEPSEEK_API_KEY" in str(ei.value)

    def test_missing_base_url_is_reported(self, registry_file, monkeypatch):
        _write(registry_file, [{"id": "ghost", "name": "幽灵厂商", "base_url": "",
                                "api_key_env": "GHOST_KEY", "models": ["ghost-1"]}],
               default="ghost")
        monkeypatch.setenv("GHOST_KEY", "sk-ghost")

        from llm.deepseek_client import _get_client_for_model, reset_client_cache
        reset_client_cache()
        try:
            with pytest.raises(ValueError) as ei:
                _get_client_for_model("ghost-1")
        finally:
            reset_client_cache()
        assert "幽灵厂商" in str(ei.value)
        assert "Base URL" in str(ei.value)


# ============================================================
# D. 健壮性：坏配置不能把整条链路拖垮
# ============================================================

class TestRobustness:

    def test_reads_have_no_side_effects(self, registry_file):
        """只是打开配置页看看，不能凭空生成一个配置文件。"""
        P.get_registry()
        P.get_providers()
        P.available_models()
        P.any_provider_configured()
        assert not registry_file.exists()

    def test_corrupt_json_falls_back_to_builtin(self, registry_file):
        """注册表被编辑坏时必须降级到内置两家继续能跑，
        而不是让所有 LLM 调用全线瘫痪（一次手滑毁掉整个系统的代价太高）。"""
        registry_file.write_text("{ this is not json", encoding="utf-8")
        P.invalidate()
        assert [p["id"] for p in P.get_providers()] == ["deepseek", "qwen"]

    def test_non_object_json_falls_back(self, registry_file):
        registry_file.write_text("[1, 2, 3]", encoding="utf-8")
        P.invalidate()
        assert [p["id"] for p in P.get_providers()] == ["deepseek", "qwen"]

    def test_empty_providers_falls_back(self, registry_file):
        _write(registry_file, [])
        assert [p["id"] for p in P.get_providers()] == ["deepseek", "qwen"]

    def test_invalid_default_provider_is_repaired(self, registry_file):
        _write(registry_file, P.DEFAULT_PROVIDERS + [KIMI], default="nonexistent")
        assert P.default_provider_id() == "deepseek"

    def test_duplicate_and_invalid_ids_are_dropped(self, registry_file):
        _write(registry_file, P.DEFAULT_PROVIDERS + [
            dict(KIMI, id="KIMI"),          # 大小写不同 → 归一化后与前面重复，应丢弃
            dict(KIMI, id="!!bad!!"),       # 非法字符 → 丢弃
            dict(KIMI, id=""),              # 空 id → 丢弃
        ])
        assert [p["id"] for p in P.get_providers()] == ["deepseek", "qwen", "kimi"]

    def test_unknown_model_never_crashes_on_odd_input(self, registry_file):
        for model in ["", None, "   ", "a" * 200]:
            hit = P.resolve_provider(model)
            assert hit is None or isinstance(hit, dict)


# ============================================================
# E. 写入：原子、可回读、不误覆盖
# ============================================================

class TestSaveRegistry:

    def test_roundtrip(self, registry_file):
        P.save_registry({"default_provider": "kimi", "providers": P.DEFAULT_PROVIDERS + [KIMI]})
        P.invalidate()
        reg = P.get_registry()
        assert reg["default_provider"] == "kimi"
        assert [p["id"] for p in reg["providers"]] == ["deepseek", "qwen", "kimi"]
        assert reg["providers"][2]["base_url"] == "https://api.moonshot.cn/v1"
        # 落盘必须是可读 JSON（不能是 .tmp 残留）
        assert json.loads(registry_file.read_text(encoding="utf-8"))["version"] == 1

    def test_rejects_empty_providers(self, registry_file):
        with pytest.raises(ValueError):
            P.save_registry({"providers": []})

    def test_repairs_default_provider(self, registry_file):
        P.save_registry({"default_provider": "ghost", "providers": [KIMI]})
        P.invalidate()
        assert P.default_provider_id() == "kimi"

    def test_init_does_not_overwrite_existing(self, registry_file):
        _write(registry_file, [KIMI], default="kimi")
        P.init_registry_file()
        P.invalidate()
        assert [p["id"] for p in P.get_providers()] == ["kimi"]

    def test_init_creates_file_from_builtin_defaults(self, registry_file):
        assert not registry_file.exists()
        P.init_registry_file()
        assert registry_file.exists()
        P.invalidate()
        assert [p["id"] for p in P.get_providers()] == ["deepseek", "qwen"]

    def test_validate_provider_id(self):
        assert P.validate_provider_id("My-Kimi_2") == "my-kimi_2"
        for bad in ["", "  ", "!!", "-lead", "a" * 40]:
            with pytest.raises(ValueError):
                P.validate_provider_id(bad)


# ============================================================
# F. 接口稳定性：调用方零改动
# ============================================================

class TestInterfaceStability:
    """本次改造最容易犯的错是"顺手把入口函数改名/换签名" ——
    `core/agents.py` 9 处、`core/chat_agent.py`、`core/review_agent.py`、
    `core/nl_screener.py`、`data/market_opinions.py` 以及 5 个测试文件
    都按 `_get_client_for_model` 这个名字引用它。"""

    def test_signature_and_cache_semantics_preserved(self):
        from llm.deepseek_client import _get_client_for_model
        assert list(inspect.signature(_get_client_for_model).parameters) == ["model"]
        # lru_cache 的存在不是可有可无：高频调用下每次新建 OpenAI 客户端会明显变慢，
        # 而 reset_client_cache() 依赖 cache_clear 才能让新密钥生效。
        assert hasattr(_get_client_for_model, "cache_clear")
        assert hasattr(_get_client_for_model, "cache_info")
        _get_client_for_model.cache_clear()

    def test_reset_client_cache_clears_it(self):
        from llm.deepseek_client import _get_client_for_model, reset_client_cache
        reset_client_cache()
        assert _get_client_for_model.cache_info().currsize == 0

    def test_deepseek_client_still_constructible(self, registry_file, _with_keys):
        """core/agents.py 的用法：`DeepSeekClient()` 不带参数 + 显式指定模型。"""
        from llm.deepseek_client import DeepSeekClient, reset_client_cache
        reset_client_cache()
        try:
            assert DeepSeekClient().model
            assert DeepSeekClient(model="qwen-plus").model == "qwen-plus"
        finally:
            reset_client_cache()

    def test_provider_status_covers_ui_needs(self, registry_file, _with_keys):
        from llm.providers import provider_status
        rows = provider_status()
        assert {r["id"] for r in rows} == {"deepseek", "qwen"}
        first = rows[0]
        for key in ("name", "base_url", "api_key_env", "ready", "enabled",
                    "is_default", "models", "match"):
            assert key in first
        assert first["is_default"] is True
