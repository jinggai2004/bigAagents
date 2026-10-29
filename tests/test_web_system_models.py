# -*- coding: utf-8 -*-
"""配置页改模型（`POST /api/system/models`）测试。

## 为什么这个文件必须存在
加这个端点之前，配置页两张表是**纯只读**的：后端 `set_feature_model` /
`reset_feature_model` / `save_registry` 全都在、mtime 热加载也早就有，
页面却只把结果打印一遍——于是"配置界面无法配置模型"。

更隐蔽的是第二层：`models()` 原来拿 `feature_usage(hours=24)` 做主体，
**没跑过的功能根本不返回**，所以想改一个没调用过的功能，连行都看不到。
（本项目反复出现的同一类失效：后端齐了、前端只读。）

## 测试用真文件而不是 mock
两个路径都有 env 覆盖（`BIGA_LLM_PROVIDERS` / `BIGA_LLM_FEATURE_MODELS`），
所以直接写到 tmp 里跑**完整往返**：改 → 落盘 → 重新读 → 生效值变。
mock 掉 `save_registry` 只能证明"调了函数"，证明不了"写完能读回来"。
"""
import json

import pytest


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """两套临时配置文件 + 清缓存，返回 (model_routing, providers)。"""
    prov = {
        "version": 2,
        "default_provider": "deepseek",
        "providers": [
            {"id": "deepseek", "name": "DeepSeek", "base_url": "https://api.deepseek.com",
             "api_key_env": "DEEPSEEK_API_KEY",
             "models": ["deepseek-v4-flash", "deepseek-v4-pro"], "enabled": True},
            {"id": "qwen", "name": "通义千问", "base_url": "https://x.aliyuncs.com",
             "api_key_env": "QWEN_API_KEY",
             "models": ["qwen-plus", "qwen-max"], "enabled": True},
        ],
    }
    feat = {"version": 3, "models": {"chat.tool_loop": "deepseek-v4-flash"}}

    p_path = tmp_path / "providers.json"
    f_path = tmp_path / "feature_models.json"
    p_path.write_text(json.dumps(prov, ensure_ascii=False), encoding="utf-8")
    f_path.write_text(json.dumps(feat, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setenv("BIGA_LLM_PROVIDERS", str(p_path))
    monkeypatch.setenv("BIGA_LLM_FEATURE_MODELS", str(f_path))

    from config import model_routing as R
    from config import providers as P
    R.invalidate()
    P.invalidate()
    return R, P


def read_feat() -> dict:
    from config import model_routing as R
    return json.loads(R.feature_models_path().read_text(encoding="utf-8"))["models"]


def read_prov() -> dict:
    from config import providers as P
    return json.loads(P.registry_path().read_text(encoding="utf-8"))


# ================================================================
# 读：功能表必须是登记全集
# ================================================================

class TestModelsRead:

    def test_lists_all_registered_features(self, cfg):
        """**没调用过的功能也必须出现**，否则没有入口改它的模型。"""
        from web.server import system
        z = system.models()
        assert z["ok"] is True, z
        feats = {f["feature"] for f in z["data"]["features"]}
        from config.model_routing import FEATURES
        missing = set(FEATURES) - feats
        assert not missing, f"这些登记功能没出现在配置页：{missing}"

    def test_feature_row_has_override_and_effective(self, cfg):
        from web.server import system
        rows = {f["feature"]: f for f in system.models()["data"]["features"]}
        r = rows["chat.tool_loop"]
        assert r["override"] == "deepseek-v4-flash"      # 页面下拉该选这一项
        assert r["model"] == "deepseek-v4-flash"         # 实际生效
        # 未覆盖的功能：override 空（跟随默认），但 model 有值（tier 兜底算出来的）
        other = rows["agents.bull"]
        assert other["override"] == ""
        assert other["model"], "跟随默认也必须能显示实际生效的模型名"

    def test_all_models_flat_list(self, cfg):
        """下拉选项来自所有厂商的模型，且标明厂商——否则看不出 qwen-plus 属于谁。"""
        from web.server import system
        ms = system.models()["data"]["all_models"]
        names = {m["model"] for m in ms}
        assert {"deepseek-v4-flash", "qwen-max"} <= names
        assert all(m["provider"] for m in ms)


# ================================================================
# 写：功能 → 模型
# ================================================================

class TestSetFeature:

    def test_set_feature_persists(self, cfg):
        R, _ = cfg
        from web.server import system
        z = system.set_models("feature", {"feature": "agents.bull", "model": "qwen-max"})
        assert z["ok"] is True, z
        assert read_feat()["agents.bull"] == "qwen-max"
        R.invalidate()                            # 模拟"下一次调用重新读盘"
        assert R.get_feature_models()["agents.bull"] == "qwen-max"
        assert z["data"]["effective"] == "qwen-max"

    def test_empty_model_clears_override(self, cfg):
        """`model=""` = 删掉覆盖（跟随默认），**不是**"设成空模型"。"""
        R, _ = cfg
        from web.server import system
        assert "chat.tool_loop" in read_feat()
        z = system.set_models("feature", {"feature": "chat.tool_loop", "model": ""})
        assert z["ok"] is True, z
        assert "chat.tool_loop" not in read_feat()
        R.invalidate()
        # 清掉覆盖后仍然解析出一个模型（走 tier 兜底），不能变成空字符串
        assert R.resolve_model("chat.tool_loop")

    def test_missing_feature_rejected(self, cfg):
        from web.server import system
        z = system.set_models("feature", {"model": "qwen-max"})
        assert z["ok"] is False
        assert "feature" in z["error"]["detail"]

    def test_unknown_action_rejected(self, cfg):
        from web.server import system
        z = system.set_models("nope", {})
        assert z["ok"] is False
        assert "未知 action" in z["error"]["detail"]


# ================================================================
# 写：默认厂商
# ================================================================

class TestSetDefault:

    def test_switch_default_provider(self, cfg):
        _, P = cfg
        from web.server import system
        z = system.set_models("default", {"provider": "qwen"})
        assert z["ok"] is True, z
        assert read_prov()["default_provider"] == "qwen"
        P.invalidate()
        assert P.default_provider_id() == "qwen"

    def test_unknown_provider_rejected(self, cfg):
        from web.server import system
        z = system.set_models("default", {"provider": "openai"})
        assert z["ok"] is False
        assert "未知厂商" in z["error"]["detail"]

    def test_rejected_write_does_not_touch_file(self, cfg):
        _, P = cfg
        from web.server import system
        before = read_prov()
        system.set_models("default", {"provider": "openai"})
        assert read_prov() == before, "被拒的写操作不能留下半截改动"


# ================================================================
# 写：厂商模型清单 / 启停
# ================================================================

class TestProviderModels:

    def test_replace_models_dedup_and_trim(self, cfg):
        _, P = cfg
        from web.server import system
        z = system.set_models("provider_models", {
            "provider": "qwen",
            "models": ["qwen-max", " qwen-max ", "", None, "qwen-plus"],
        })
        assert z["ok"] is True, z
        assert z["data"]["models"] == ["qwen-max", "qwen-plus"]   # 去重去空
        got = next(p for p in read_prov()["providers"] if p["id"] == "qwen")
        assert got["models"] == ["qwen-max", "qwen-plus"]

    def test_models_must_be_list(self, cfg):
        from web.server import system
        z = system.set_models("provider_models", {"provider": "qwen", "models": "qwen-max"})
        assert z["ok"] is False
        assert "数组" in z["error"]["detail"]

    def test_toggle_provider(self, cfg):
        _, P = cfg
        from web.server import system
        z = system.set_models("provider_enabled", {"provider": "qwen", "enabled": False})
        assert z["ok"] is True, z
        got = next(p for p in read_prov()["providers"] if p["id"] == "qwen")
        assert got["enabled"] is False


# ================================================================
# 接线
# ================================================================

class TestEndpoint:

    def test_post_route_registered(self):
        """函数写好但没人调 = 没做。端点必须真的挂在 router 上。"""
        import importlib
        import pkgutil

        import web.server.routers as R
        found = set()
        for m in pkgutil.iter_modules(R.__path__):
            rt = getattr(importlib.import_module(f"web.server.routers.{m.name}"),
                         "router", None)
            for r in (rt.routes if rt else []):
                if getattr(r, "path", "") == "/api/system/models":
                    found.update(getattr(r, "methods", set()) or set())
        assert {"GET", "POST"} <= found, f"实际方法：{found}"

    def test_route_forwards_action(self, cfg, monkeypatch):
        """路由把 action 和整个 body 一起透传（字段名对不上会静默不生效）。"""
        from web.server import system
        from web.server.routers import system as RS

        seen = {}

        def fake(action, payload=None):
            seen["action"] = action
            seen["payload"] = payload
            return {"ok": True, "data": {}}

        monkeypatch.setattr(system, "set_models", fake)
        RS.api_system_models_set({"action": "feature", "feature": "x", "model": "y"})
        assert seen["action"] == "feature"
        assert seen["payload"]["feature"] == "x"

    def test_empty_body_survives(self, cfg):
        """前端少传字段不能 500。"""
        from web.server.routers import system as RS
        z = RS.api_system_models_set({})
        assert z["ok"] is False
        assert "未知 action" in z["error"]["detail"]
