"""功能级模型路由（config/model_routing.py）的守护测试。

本次改造的目标：把"哪个功能用哪个模型"从**代码/导入期常量**外置成一份 JSON 映射，
并且做到：

  ① **non-mutation**：不配置映射时，每个功能解析到的模型 == 改造前它实际用的模型
     （这是硬要求，也是本文件最重要的一组断言）；
  ② 解析优先级：显式传参 > 功能映射 > 调用方自带模型 > 功能默认档位 > 内置默认；
  ③ 健壮性：功能键未声明 / 值为空 / 文件损坏 → 降级到默认，**绝不抛异常**；
  ④ 热加载：mtime 缓存 + invalidate；
  ⑤ 写入失败不得静默：返回 False 且留日志；
  ⑥ **结构性守护**：FEATURES 注册表与代码里实际出现的 `purpose` 必须一一对应。

⚠️ 结构性守护是本文件的核心价值：它防的是"新加了调用点却忘了登记功能键"——
   那种情况下新调用点会**悄悄**落到默认模型，配置页上根本看不到它。
"""

import ast
import json
import logging
import os
from pathlib import Path

import pytest

from config import model_routing as MR
from config.settings import config

# 生产代码扫描范围（不含 tests/ 与 config/ —— 前者满是对照用的假 purpose，
# 后者是注册表自身）。
# `web` 也扫：`web/server/desk.py` 的"阿深面对面问答"是**唯一**不在 core 层的
# LLM 调用点。把它漏在外面时，`ai.*.chat` 成了"只在 web 出现、族键匹配不到"的
# 孤儿键（2026-10-28 实测就红了）。
# `scripts` 也扫（2026-10-28 补）：`scripts/verify_ashen_phase.py` 会真调模型来
# 实证时段剧本。漏掉它时，登记的 `dev.verify.*` 会被判成"死键"——
# 而**判成死键的直觉反应是把它从注册表删掉**，那恰好让验证脚本的 token
# 重新掉回"未登记"。扫描范围要盖住所有真会调 LLM 的目录。
_SCAN_DIRS = ("core", "llm", "data", "ui", "web", "scripts")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 已知的**遗留死 purpose**：它们出现在 DeepSeekClient 里三个**没有任何调用点**的方法
# （fund_flow_analysis / comprehensive_discussion / final_decision）中，且彼此还错位
# （资金面方法里写的是 client.comprehensive_discussion）。这些不是"功能"，不该占用
# 配置页；但静态扫描一定会看到它们，故在此显式登记为"不参与路由的遗留字面量"。
# ⚠️ 配套 test_legacy_methods_have_no_call_sites 会钉住"它们确实无人调用"——
#    一旦有人把方法接上，那条会红，逼你把 purpose 登记进 FEATURES。
_LEGACY_UNUSED_PURPOSES = frozenset({
    "client.comprehensive_discussion",
    "client.final_decision",
})
_LEGACY_METHODS = ("fund_flow_analysis", "comprehensive_discussion", "final_decision")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """每个用例：独立映射文件 + 独立 SQLite。

    映射文件用环境变量改路径（与 config/providers.py 的测试同一套做法），
    同时清空模块级 mtime 缓存，避免用例之间串味。
    独立 SQLite 是因为 call_api / feature_usage 会读写 llm_calls ——
    不隔离就会往真实 data/portfolio.db 里写假调用记录。
    """
    path = tmp_path / "llm_feature_models.json"
    monkeypatch.setenv(MR.ENV_ROUTING_PATH, str(path))
    MR.invalidate()
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "routing.db")
    yield path
    MR.invalidate()


def _write(path, mapping: dict):
    """写一份映射并让缓存失效（同一测试内连续写两次时，mtime 分辨率可能不够）。"""
    path.write_text(json_dumps(mapping), encoding="utf-8")
    MR.invalidate()


def json_dumps(mapping: dict) -> str:
    import json
    return json.dumps({"version": 1, "models": mapping}, ensure_ascii=False)


# ============================================================
# ① non-mutation：不配置时 == 改造前的模型
# ============================================================

class TestNonMutation:
    """改造前各功能实际用的模型（**写死在测试里**，不引用 FEATURES 的档位逻辑，
    否则就成了"用实现验证实现"）。"""

    # 改造前读 QWEN_TURBO_MODEL（CHAT_MODEL_FAST / market_opinions 的 config.QWEN_TURBO_MODEL）
    LEGACY_FAST = {"chat.tool_loop", "chat.tool_loop_summary", "market_opinions"}
    # 改造前读 QWEN_PLUS_MODEL（CHAT_MODEL_DEEP / REVIEW_MODEL）
    LEGACY_DEEP = {"chat.deep", "review.daily"}

    def test_fast_tier_features_match_old_constant(self):
        expected = config.QWEN_TURBO_MODEL or "qwen-max"
        for feature in self.LEGACY_FAST:
            assert MR.resolve_model(feature) == expected, feature

    def test_deep_tier_features_match_old_constant(self):
        expected = config.QWEN_PLUS_MODEL or "qwen-max"
        for feature in self.LEGACY_DEEP:
            assert MR.resolve_model(feature) == expected, feature

    def test_all_other_features_follow_global_default(self):
        others = set(MR.FEATURES) - self.LEGACY_FAST - self.LEGACY_DEEP
        assert others, "注册表为空？"
        for feature in others:
            assert MR.resolve_model(feature) == config.DEFAULT_MODEL_NAME, feature

    def test_tier_map_is_complete_and_disjoint(self):
        """档位分类必须覆盖注册表里每一个功能键（不能漏、不能重）。"""
        assert (self.LEGACY_FAST | self.LEGACY_DEEP) <= set(MR.FEATURES)
        assert not (self.LEGACY_FAST & self.LEGACY_DEEP)

    def test_unknown_and_empty_feature_fall_to_global_default(self):
        for feature in ("", None, "不存在的功能键", "unknown"):
            assert MR.resolve_model(feature) == config.DEFAULT_MODEL_NAME

    def test_empty_mapping_changes_nothing(self, _isolate):
        """空映射文件存在时，解析结果与"没有文件"完全一致。"""
        before = {k: MR.resolve_model(k) for k in MR.FEATURES}
        _write(_isolate, {})
        after = {k: MR.resolve_model(k) for k in MR.FEATURES}
        assert before == after


# ============================================================
# ② 解析优先级
# ============================================================

class TestPriority:
    def test_explicit_beats_mapping(self, _isolate):
        _write(_isolate, {"agents.bull": "qwen-turbo"})
        assert MR.resolve_model("agents.bull", explicit="qwen-max") == "qwen-max"

    def test_mapping_beats_default(self, _isolate):
        _write(_isolate, {"agents.bull": "qwen-turbo"})
        assert MR.resolve_model("agents.bull") == "qwen-turbo"

    def test_mapping_beats_client_model(self, _isolate):
        """调用方自带模型（self.model）在功能映射之后——配了就覆盖。"""
        _write(_isolate, {"agents.bull": "qwen-turbo"})
        assert MR.resolve_model("agents.bull", default="qwen-plus") == "qwen-turbo"

    def test_client_model_used_when_unmapped(self, _isolate):
        """非 mutation 的关键：没配功能映射时，调用方自带模型不能被顶掉。"""
        assert MR.resolve_model("agents.bull", default="qwen-plus") == "qwen-plus"

    def test_client_model_beats_tier(self, _isolate):
        """自带模型优先于档位默认（连通性测试传入了用户选的模型）。"""
        assert MR.resolve_model("chat.tool_loop", default="deepseek-chat") == "deepseek-chat"

    def test_empty_explicit_falls_through(self, _isolate):
        assert MR.resolve_model("agents.bull", explicit="") == config.DEFAULT_MODEL_NAME
        assert MR.resolve_model("agents.bull", explicit="   ") == config.DEFAULT_MODEL_NAME


# ============================================================
# ③ 降级：未声明 / 空值 / 损坏一律不抛
# ============================================================

class TestFallback:
    def test_undeclared_feature_uses_default(self, _isolate):
        _write(_isolate, {"agents.bull": "qwen-turbo"})
        assert MR.resolve_model("agents.bear") == config.DEFAULT_MODEL_NAME

    def test_empty_value_is_treated_as_unset(self, _isolate):
        _write(_isolate, {"agents.bull": ""})
        assert MR.feature_model("agents.bull") is None
        assert MR.resolve_model("agents.bull") == config.DEFAULT_MODEL_NAME

    def test_corrupt_json_falls_back_to_empty(self, _isolate):
        _isolate.write_text("{ 这不是 JSON", encoding="utf-8")
        MR.invalidate()
        assert MR.get_feature_models() == {}
        assert MR.resolve_model("agents.bull") == config.DEFAULT_MODEL_NAME

    def test_non_object_json_falls_back(self, _isolate):
        _isolate.write_text("[1, 2, 3]", encoding="utf-8")
        MR.invalidate()
        assert MR.get_feature_models() == {}

    def test_missing_file_is_empty(self, _isolate):
        assert not _isolate.exists()
        assert MR.get_feature_models() == {}

    def test_legacy_bare_shape_is_accepted(self, _isolate):
        """兼容裸 {purpose: model} 写法（没有外层 models 字段）。"""
        _isolate.write_text('{"agents.bull": "qwen-turbo", "version": 1}', encoding="utf-8")
        MR.invalidate()
        assert MR.feature_model("agents.bull") == "qwen-turbo"

    def test_non_string_values_are_dropped(self, _isolate):
        _isolate.write_text('{"models": {"agents.bull": 123, "agents.bear": null}}',
                            encoding="utf-8")
        MR.invalidate()
        assert MR.get_feature_models() == {}

    def test_reads_never_create_the_file(self, _isolate):
        """只是打开配置页看看，不能凭空生成配置文件。"""
        MR.get_feature_models()
        MR.resolve_model("agents.bull")
        assert not _isolate.exists()


# ============================================================
# ④ mtime 缓存与 invalidate
# ============================================================

class TestCache:
    def test_reload_without_invalidate_when_mtime_changes(self, _isolate, monkeypatch):
        _write(_isolate, {"agents.bull": "qwen-turbo"})
        assert MR.resolve_model("agents.bull") == "qwen-turbo"
        # 直接改文件、**不调用 invalidate**：靠 mtime 变化自动重读（热加载的核心）
        _isolate.write_text(json_dumps({"agents.bull": "qwen-max"}), encoding="utf-8")
        os.utime(_isolate, (1, 1))          # 强制一个明显不同的 mtime，避免同 ns
        assert MR.resolve_model("agents.bull") == "qwen-max"

    def test_second_read_uses_cache(self, _isolate, monkeypatch):
        _write(_isolate, {"agents.bull": "qwen-turbo"})
        calls = {"n": 0}
        real = MR._read_file

        def counting():
            calls["n"] += 1
            return real()

        monkeypatch.setattr(MR, "_read_file", counting)
        MR.get_feature_models()
        MR.get_feature_models()
        assert calls["n"] == 1, "同一 mtime 下第二次读取不该再解析文件"

    def test_invalidate_forces_reread(self, _isolate):
        _write(_isolate, {"agents.bull": "qwen-turbo"})
        assert MR.feature_model("agents.bull") == "qwen-turbo"
        _isolate.unlink()
        MR.invalidate()
        assert MR.get_feature_models() == {}


# ============================================================
# ⑤ 写入：原子、可回读、失败不静默
# ============================================================

class TestSave:
    def test_roundtrip(self, _isolate):
        assert MR.save_feature_models({"agents.bull": "qwen-turbo"}) is True
        MR.invalidate()
        assert MR.feature_model("agents.bull") == "qwen-turbo"
        # 落盘必须是可读 JSON，且不留 .tmp
        import json
        assert json.loads(_isolate.read_text(encoding="utf-8"))["version"] == 1
        assert not list(_isolate.parent.glob("*.tmp"))

    def test_set_and_reset(self, _isolate):
        assert MR.set_feature_model("agents.bull", "qwen-turbo") is True
        assert MR.feature_model("agents.bull") == "qwen-turbo"
        assert MR.reset_feature_model("agents.bull") is True
        assert MR.feature_model("agents.bull") is None
        # 幂等：删除不存在的项也返回 True
        assert MR.reset_feature_model("agents.bull") is True

    def test_set_empty_feature_rejected(self, _isolate):
        assert MR.set_feature_model("", "qwen-turbo") is False

    def test_write_failure_returns_false_and_logs(self, _isolate, monkeypatch, caplog):
        """⚠️ 脏写：写失败**不得静默**——必须返回 False 且留一条日志。"""
        def boom(src, dst):
            raise OSError("文件被占用")

        monkeypatch.setattr(os, "replace", boom)
        with caplog.at_level(logging.WARNING, logger="config.model_routing"):
            ok = MR.save_feature_models({"agents.bull": "qwen-turbo"})
        assert ok is False
        assert "写入失败" in caplog.text

    def test_non_dict_rejected(self, _isolate):
        assert MR.save_feature_models(["not", "a", "dict"]) is False

    def test_init_does_not_overwrite_existing(self, _isolate):
        _write(_isolate, {"agents.bull": "qwen-turbo"})
        assert MR.init_feature_models_file() is True
        MR.invalidate()
        assert MR.feature_model("agents.bull") == "qwen-turbo"

    def test_init_creates_empty_file(self, _isolate):
        assert not _isolate.exists()
        assert MR.init_feature_models_file() is True
        MR.invalidate()
        assert _isolate.exists()
        assert MR.get_feature_models() == {}


# ============================================================
# ⑥ 结构性守护：FEATURES 与代码里的 purpose 一一对应
# ============================================================

def _collect_purpose_literals() -> set:
    """AST 扫描生产代码里的 `purpose="..."` 与 `llm_purpose("...")` 字面量。

    为什么用 AST 而不是正则：`deepseek_client.llm_purpose` 的**文档字符串**里就有
    `with llm_purpose("agents.tech"):` 这样的示例（不是真实调用点），正则会把注释、
    文档、示例全算进去，产生假阳性。AST 只看真实调用节点。
    """
    found: set = set()
    for sub in _SCAN_DIRS:
        for py in (_PROJECT_ROOT / sub).rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else (
                    func.id if isinstance(func, ast.Name) else "")
                if name in ("call_api", "chat", "chat_stream"):
                    for kw in node.keywords:
                        if kw.arg == "purpose" and isinstance(kw.value, ast.Constant) \
                                and isinstance(kw.value.value, str):
                            found.add(kw.value.value)
                elif name == "llm_purpose":
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            found.add(arg.value)
    return found


def _collect_purpose_patterns() -> set:
    """AST 扫描**运行时拼接**的 purpose，把可变段换成 `*`。

    `purpose=f"ai.{name}.routine"` → `ai.*.routine`

    为什么需要它：`ai.*` 这一族的功能键是拼出来的（人设名来自
    `config/ai_personas.json`），静态字面量扫描永远抓不到。于是它们长期
    "既没登记、也无法在配置页配置"——换 `DEFAULT_MODEL_NAME` 时整族会被
    静默带走，而页面上看不到任何变化。有了这个扫描，族键登记才能被反向
    守护：**登记了族键、代码里却没有对应的 f-string 构造点 ⇒ 那条会红**。
    """
    found: set = set()
    for sub in _SCAN_DIRS:
        for py in (_PROJECT_ROOT / sub).rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for kw in node.keywords:
                    if kw.arg != "purpose" or not isinstance(kw.value, ast.JoinedStr):
                        continue
                    segs = [str(v.value) if isinstance(v, ast.Constant) else "*"
                            for v in kw.value.values]
                    pat = "".join(segs)
                    if "*" in pat:
                        found.add(pat)
    return found


def _covered(feature_key: str, literals: set, patterns: set) -> bool:
    """注册表里的这条键，在代码里有没有落点？

    三种情形都算有：
      · 键本身出现在字面量里（`config.test`）
      · 键本身出现在拼接模式里（`ai.*.routine`）
      · 它是族键，且被某个**具体**字面量命中（`ai.*.chat` ← `ai.ai_ashen.chat`）
    """
    if feature_key in literals or feature_key in patterns:
        return True
    if "*" in feature_key:
        return any(MR._glob_match(feature_key, k) for k in literals)
    return False


class TestStructuralGuard:
    def test_every_registered_feature_is_used_or_reserved(self):
        """反向：注册表里不能有"死键"（登记了却没有任何调用点）——除非显式标 reserved。"""
        used = _collect_purpose_literals()
        pats = _collect_purpose_patterns()
        dead = {k for k in MR.FEATURES
                if k not in MR.reserved_features() and not _covered(k, used, pats)}
        assert not dead, f"这些功能键登记了但代码里找不到对应 purpose：{sorted(dead)}"

    def test_every_call_site_is_registered(self):
        """正向：代码里出现的每个 purpose 都必须在 FEATURES 里登记（防漂移）。

        ⚠️ 判定走 `match_feature()`，**不能**用 `used - set(MR.FEATURES)`：
        族键登记（`dev.verify.*` / `ai.*.routine`）与代码里的具体字面量
        （`dev.verify.phase` / `ai.ai_ashen.routine`）写法天然不同，
        集合相减会把"已正确登记的族成员"误报成漂移——那会逼人把族键删掉，
        于是这些调用点又掉回"未登记"。这正是 2026-10-28 补 `dev.verify.*` 时踩的坑。
        """
        used = _collect_purpose_literals() | _collect_purpose_patterns()
        unknown = {k for k in used
                   if MR.match_feature(k) is None and k not in _LEGACY_UNUSED_PURPOSES}
        assert not unknown, (
            f"代码里出现了未登记的功能键：{sorted(unknown)}。"
            "新增 LLM 调用点时必须同步登记到 config/model_routing.py::FEATURES，"
            "否则该功能会静默落到默认模型、配置页上也看不到。")

    def test_registry_and_code_are_one_to_one(self):
        """两侧合起来必须**恰好相等**（既无死键、也无漂移）。

        逐项双向判定，而不是集合相减：族键与其成员字面量落在同一侧的不同写法上，
        `used - expected` 会得到"注册表多出 `dev.verify.*`"这种**假红**
        （实际是代码里的 `dev.verify.phase` 刚好命中它）。
        两个方向都留着，才挡得住"只增不减"和"只减不增"两种漂移。
        """
        literals = _collect_purpose_literals()
        patterns = _collect_purpose_patterns()
        used = literals | patterns
        # 方向 1：代码里的每个 purpose 都要有归属
        missing = sorted(k for k in used
                         if MR.match_feature(k) is None
                         and k not in _LEGACY_UNUSED_PURPOSES)
        # 方向 2：注册表里每个非 reserved 的键都要有落点
        dead = sorted(k for k in set(MR.FEATURES) - MR.reserved_features()
                      if not _covered(k, literals, patterns))
        assert not (missing or dead), (
            f"代码扫描{purpose_diff(set(missing), set(dead))}")

    def test_legacy_methods_have_no_call_sites(self):
        """`_LEGACY_UNUSED_PURPOSES` 的豁免必须成立：那三个方法确实没人调用。

        如果将来有人把它们接上，这条会红——那时就该把它们的 purpose 正式登记进 FEATURES。
        """
        hits = []
        for sub in _SCAN_DIRS:
            for py in (_PROJECT_ROOT / sub).rglob("*.py"):
                text = py.read_text(encoding="utf-8")
                for method in _LEGACY_METHODS:
                    if f".{method}(" in text:
                        hits.append(f"{py.name}: .{method}(")
        assert not hits, f"遗留方法已被接入，请登记其 purpose：{hits}"

    def test_reserved_features_are_documented(self):
        """无调用点的登记项必须显式标注，不能是"忘了接线"。

        当前有两类：
          · `chat.deep`            预留位（原 `CHAT_MODEL_DEEP` 死常量转来）
          · `verify.model`         已下线临时脚本的历史遗留（键形两段，族键覆盖不到；
                                   留着只为让 24h 窗口内的旧用量归组，不再产生新记录）
        集合写成字面量是刻意的：**新增一条就必须来改这里**，
        否则"登记了个没人调的东西"会悄无声息地留在配置页上。
        """
        assert MR.reserved_features() == {"chat.deep", "verify.model"}
        for key in MR.reserved_features():
            assert MR.FEATURES[key].get("desc"), f"{key} 标了 reserved 但没写清原因"

    def test_feature_metadata_complete(self):
        for key, meta in MR.FEATURES.items():
            for field in ("label", "group", "tier"):
                assert meta.get(field), f"{key} 缺少 {field}"
            assert meta["tier"] in (MR.TIER_GLOBAL, MR.TIER_FAST, MR.TIER_DEEP)


def purpose_diff(used, expected) -> str:
    missing = sorted(used - expected)
    extra = sorted(expected - used)
    return f"不一致：代码多出 {missing}｜注册表多出 {extra}"


# ============================================================
# 客户端接线：call_api / chat 真的按功能映射走
# ============================================================

class TestClientWiring:
    def _install_fake(self, monkeypatch):
        from unittest.mock import MagicMock
        seen = {}

        def fake_get(model):
            seen["model"] = model
            return MagicMock(
                chat=MagicMock(completions=MagicMock(create=MagicMock(
                    return_value=_fake_response()))))

        monkeypatch.setattr("llm.deepseek_client._get_client_for_model", fake_get)
        return seen

    def test_call_api_uses_feature_mapping(self, _isolate, monkeypatch):
        seen = self._install_fake(monkeypatch)
        _write(_isolate, {"agents.bull": "qwen-turbo"})
        from llm.deepseek_client import DeepSeekClient
        DeepSeekClient().call_api([{"role": "user", "content": "hi"}], purpose="agents.bull")
        assert seen["model"] == "qwen-turbo"

    def test_call_api_unmapped_keeps_client_model(self, _isolate, monkeypatch):
        """non-mutation 落到客户端层：没配映射时用构造时指定的模型。"""
        seen = self._install_fake(monkeypatch)
        from llm.deepseek_client import DeepSeekClient
        DeepSeekClient(model="qwen-plus").call_api([{"role": "user", "content": "hi"}])
        assert seen["model"] == "qwen-plus"

    def test_call_api_explicit_model_beats_mapping(self, _isolate, monkeypatch):
        seen = self._install_fake(monkeypatch)
        _write(_isolate, {"agents.bull": "qwen-turbo"})
        from llm.deepseek_client import DeepSeekClient
        DeepSeekClient().call_api([{"role": "user", "content": "hi"}],
                                  model="deepseek-chat", purpose="agents.bull")
        assert seen["model"] == "deepseek-chat"

    def test_chat_resolves_per_purpose_independently(self, _isolate, monkeypatch):
        """chat.tool_loop 与 chat.tool_loop_summary 必须能各自配模型。"""
        seen = self._install_fake(monkeypatch)
        _write(_isolate, {"chat.tool_loop_summary": "qwen-max"})
        from llm.deepseek_client import DeepSeekClient
        client = DeepSeekClient(model="qwen-plus")
        client.chat([{"role": "user", "content": "hi"}], purpose="chat.tool_loop_summary")
        assert seen["model"] == "qwen-max"

    def test_context_purpose_routes_model(self, _isolate, monkeypatch):
        """不传 purpose 时用 llm_purpose() 上下文里的标签路由（与 llm_calls 同源）。"""
        seen = self._install_fake(monkeypatch)
        _write(_isolate, {"agents.bear": "qwen-turbo"})
        from llm.deepseek_client import DeepSeekClient, llm_purpose
        with llm_purpose("agents.bear"):
            DeepSeekClient().call_api([{"role": "user", "content": "hi"}])
        assert seen["model"] == "qwen-turbo"


def _fake_response(content="OK"):
    from unittest.mock import MagicMock
    msg = MagicMock()
    msg.content = content
    msg.reasoning_content = None
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return resp


# ============================================================
# 用量统计
# ============================================================

class TestFeatureUsage:
    def test_counts_and_labels(self):
        from db import store
        store.insert_llm_call({"model": "qwen-max", "purpose": "agents.bull",
                               "ok": True, "total_tokens": 100})
        store.insert_llm_call({"model": "qwen-max", "purpose": "agents.bull",
                               "ok": False, "total_tokens": 50, "error": "x"})
        store.insert_llm_call({"model": "qwen-max", "purpose": "没登记的功能",
                               "ok": True, "total_tokens": 10})
        usage = {u["feature"]: u for u in MR.feature_usage(hours=24)}
        # 已登记功能一定出现（即使 0 次调用）
        assert set(MR.FEATURES) <= set(usage)
        assert usage["agents.bull"]["count"] == 2
        assert usage["agents.bull"]["failed"] == 1
        assert usage["agents.bull"]["tokens"] == 150
        assert usage["agents.news"]["count"] == 0
        assert "没登记的功能" in usage
        assert usage["没登记的功能"]["group"] == "⚠️ 未登记"

    def test_db_failure_returns_empty(self, monkeypatch):
        from db import store
        monkeypatch.setattr(store, "get_connection",
                            lambda: (_ for _ in ()).throw(RuntimeError("db down")))
        assert MR.feature_usage(hours=24) == []


# ================================================================
# non-mutation 的**真实路径**守护（2026-10-14 补，修复一处静默换模型）
# ================================================================
#
# ⚠️ 背景：首版把 `call_api` 写成 `resolve_model(label, explicit=model, default=self.model)`。
# 而 `self.model = model or config.DEFAULT_MODEL_NAME` ⇒ **永远非空**
# ⇒ `default` 层总是命中 ⇒ 功能「档位(tier)」变成**死代码**。
#
# 后果：`DEFAULT_MODEL_NAME ≠ 档位模型` 的部署里会**静默换模型** ——
# 实测复现：DEFAULT=deepseek-chat、QWEN_TURBO_MODEL=qwen-max 时，
# `chat.tool_loop` 从改造前的 qwen-max 悄悄变成 deepseek-chat。
#
# ⚠️ 为什么首版测试没抓到：它只验了**裸调** `resolve_model(...)`（不传 default），
# 而生产走的是 `call_api` 那条带 default 的路径。**"验了函数" ≠ "验了接线"。**
# 这正是本项目反复出现的失效形态（"实现了但没被调用"的同构变体）。
#
# ⇒ 本类**故意把 DEFAULT 与档位设成不同值**，并**走 `_route_default()` 真实路径**。

class TestNonMutationOnRealPath:
    @pytest.fixture
    def split_deploy(self, monkeypatch):
        """制造 DEFAULT_MODEL_NAME ≠ 档位模型 的部署（这是能暴露问题的唯一条件）。"""
        from config.settings import config
        from config import model_routing as MR
        monkeypatch.setattr(config, "DEFAULT_MODEL_NAME", "deepseek-chat", raising=False)
        monkeypatch.setattr(config, "QWEN_TURBO_MODEL", "qwen-max", raising=False)
        monkeypatch.setattr(config, "QWEN_PLUS_MODEL", "qwen-max", raising=False)
        MR.invalidate()
        yield
        MR.invalidate()

    def test_client_without_explicit_model_yields_none_default(self, split_deploy):
        """根因守护：未显式传 model 时，参与路由的 default **必须是 None**。

        若这一条失败（返回了 DEFAULT_MODEL_NAME），说明 `default` 层又会永远命中、
        档位失效 —— 静默换模型的缺陷会原样回来。
        """
        from llm.deepseek_client import DeepSeekClient
        c = DeepSeekClient()
        assert c.model == "deepseek-chat"          # 自带模型仍是全局默认（对外行为不变）
        assert c._route_default() is None, \
            "未显式传 model 时不得把 DEFAULT_MODEL_NAME 塞进 default 层（会让档位失效）"

    def test_client_with_explicit_model_is_returned(self, split_deploy):
        """显式构造的客户端必须优先 —— 系统配置页靠它"当场选模型测连通性"。"""
        from llm.deepseek_client import DeepSeekClient
        assert DeepSeekClient(model="qwen-max")._route_default() == "qwen-max"

    def test_fast_tier_features_keep_old_model(self, split_deploy):
        """`chat.*` / `market_opinions` 改造前读 `QWEN_TURBO_MODEL`（不是全局默认）。"""
        from llm.deepseek_client import DeepSeekClient
        from config.model_routing import resolve_model
        d = DeepSeekClient()._route_default()
        for feat in ("chat.tool_loop", "chat.tool_loop_summary", "market_opinions"):
            assert resolve_model(feat, explicit=None, default=d) == "qwen-max", feat

    def test_deep_tier_feature_keeps_old_model(self, split_deploy):
        """`review.daily` 改造前读 `QWEN_PLUS_MODEL`。"""
        from llm.deepseek_client import DeepSeekClient
        from config.model_routing import resolve_model
        d = DeepSeekClient()._route_default()
        assert resolve_model("review.daily", explicit=None, default=d) == "qwen-max"

    def test_global_tier_features_follow_default(self, split_deploy):
        """`global` 档位仍应跟随 `DEFAULT_MODEL_NAME`（原本就是这个行为）。"""
        from llm.deepseek_client import DeepSeekClient
        from config.model_routing import resolve_model
        d = DeepSeekClient()._route_default()
        for feat in ("client.technical_analysis", "agents.risk_manager"):
            assert resolve_model(feat, explicit=None, default=d) == "deepseek-chat", feat

    @pytest.mark.parametrize("method", ["call_api", "chat", "chat_stream"])
    def test_all_three_call_paths_use_route_default(self, method, split_deploy,
                                                    monkeypatch):
        """⚠️ 三条调用路径**都必须**走 `_route_default()`。

        只改一条会让"改了一半"静默存在（本项目已吃过"只给风控注入阈值、没给终审"的亏）。
        用源码级断言锁住：这三处的 default 参数不得再写回 `self.model`。
        """
        import inspect
        from llm import deepseek_client as DC
        src = inspect.getsource(getattr(DC.DeepSeekClient, method))
        assert "_route_default()" in src, f"{method} 未走 _route_default()"
        assert "default=self.model" not in src, \
            f"{method} 又写回了 default=self.model（会让功能档位失效）"


# ============================================================
# ⑦ 族键（`ai.*.xxx`）：运行时拼出来的功能也要能配、能看
# ============================================================

def _use_mapping(monkeypatch, tmp_path, mapping: dict):
    """把功能映射指到一个临时文件。

    ⚠️ 不能直接 `save_feature_models(...)` 去改真实的
    `config/llm_feature_models.json`：那是生产配置，测试中途失败就会把它写成
    半截（本项目已经因为"测试"改过真实库）。`BIGA_LLM_FEATURE_MODELS` 是这一层
    唯一的隔离入口，测试就该走它。
    """
    f = tmp_path / "feature_models.json"
    f.write_text(json.dumps({"version": 1, "models": mapping}, ensure_ascii=False),
                 encoding="utf-8")
    monkeypatch.setenv("BIGA_LLM_FEATURE_MODELS", str(f))
    MR.invalidate()


class TestFamilyKeys:
    """`ai.*.decision/routine/wake/chat` 这套族键的行为。

    背景（2026-10-28）：这四族每天在花 token（AI 交易员决策/唤醒、阿深值班与
    问答），但 purpose 是 `f"ai.{persona}.{kind}"` 拼出来的，静态字面量扫不到，
    于是**从来没登记过** ⇒ 配置页上看不见、换全局默认模型时被静默带走。
    """

    def test_family_key_matches_concrete_purpose(self):
        assert MR.match_feature("ai.ai_ashen.routine") == "ai.*.routine"
        assert MR.match_feature("ai.ai_tech.decision") == "ai.*.decision"
        assert MR.match_feature("ai.ai_risk.wake") == "ai.*.wake"
        assert MR.match_feature("ai.ai_ashen.chat") == "ai.*.chat"

    def test_family_key_does_not_swallow_neighbours(self):
        """段级通配，不能在字符级误吃。"""
        for k in ("ai.ai_ashen.routineX", "ai.ai_ashen", "x.ai_tech.decision",
                  "ai.ai_tech.decision.extra"):
            assert MR.match_feature(k) is None, k

    def test_label_carries_persona_name(self):
        """配置页上四个交易员不能都叫「AI 值班轮」——必须带出人设名。"""
        assert MR.feature_label("ai.ai_ashen.routine") == "AI 值班轮（ai_ashen）"
        assert MR.feature_label("ai.ai_tech.decision") == "AI 交易员决策轮（ai_tech）"

    def test_persona_binding_beats_family_key(self):
        """**persona 绑定的模型必须赢过族键**。

        `ai_personas.json` 里的 `model` 是"这个 AI 用哪个模型"（联赛设定，
        各 AI 可以不同）。若让族键赢，给 `ai.*.chat` 填一个值就会把所有
        persona 的模型一起换掉，而页面上看不出是谁改的。
        """
        model = MR.resolve_model("ai.ai_ashen.brand_new_kind", explicit=None,
                                 default="persona-model")
        assert model == "persona-model"

    def test_persona_binding_beats_registered_family(self):
        """已登记的族键同样不能顶掉 persona 绑定（用真实存在的族键验一遍）。"""
        assert MR.resolve_model("ai.ai_ashen.routine", default="persona-model") == "persona-model"

    def test_exact_key_beats_persona_binding(self, tmp_path, monkeypatch):
        """精确键是"对这一个功能的明确指定"，理应压过 persona 的粗粒度绑定。"""
        _use_mapping(monkeypatch, tmp_path,
                     {"ai.*.chat": "family-model", "ai.ai_ashen.chat": "exact-model"})
        got = MR.resolve_model("ai.ai_ashen.chat", default="persona-model")
        assert got == "exact-model", got

    def test_family_key_is_fallback_when_persona_has_no_model(self, tmp_path, monkeypatch):
        """persona 没绑模型（如随机组）时，族键要能兜住。"""
        _use_mapping(monkeypatch, tmp_path, {"ai.*.routine": "family-model"})
        assert MR.resolve_model("ai.ai_orphan.routine", default=None) == "family-model"

    def test_usage_aggregates_into_family_row(self):
        """配置页的调用次数必须聚到族键上，不能显示 0 而把真实花销藏进"未登记"。"""
        rows = {r["feature"]: r for r in MR.feature_usage(hours=24)}
        for feat in ("ai.*.decision", "ai.*.routine", "ai.*.chat", "ai.*.wake"):
            assert feat in rows, f"用量表里没有族键 {feat}"


# ============================================================
