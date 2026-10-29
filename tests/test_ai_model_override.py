# -*- coding: utf-8 -*-
"""C1 模型覆盖（`llm.deepseek_client.model_override`）的测试。

⭐ 核心验收（开发计划 S2）：**并发**跑多个 persona（各持不同 model），断言互不串号。
   ContextVar 是唯一正确的载体——全局变量/env 在并发下会把 A 的模型用到 B 头上。

⚠️ 同时守护 non-mutation：无 persona 上下文时行为必须与改造前逐字节一致
   （`_route_default()` 返回 None ⇒ 功能档位不被顶掉——2026-10-14 刚修过这个坑）。
"""
import threading

from llm.deepseek_client import DeepSeekClient, model_override


class TestOverrideBasic:
    def test_overrides_route_default(self):
        """覆盖**按调用时刻**生效（不是构造时快照）——见 TestRealProductionPath 的说明。
        因此断言必须在上下文之内；退出后还原为 None。"""
        c = DeepSeekClient()
        with model_override("deepseek-chat"):
            assert c._route_default() == "deepseek-chat"
            assert DeepSeekClient().model == "deepseek-chat"
        assert c._route_default() is None

    def test_no_override_keeps_old_behavior(self):
        """无上下文覆盖 = 与改造前完全一致（non-mutation）。"""
        c = DeepSeekClient()
        assert c._route_default() is None            # 档位不被顶掉

    def test_explicit_model_wins_over_override(self):
        """构造参数 > 上下文覆盖（都是调用方的明确意图，构造参数更近）。"""
        with model_override("deepseek-chat"):
            c = DeepSeekClient(model="qwen-max")
        assert c._route_default() == "qwen-max"

    def test_nested_override_restores_outer(self):
        with model_override("outer"):
            with model_override("inner"):
                assert DeepSeekClient()._route_default() == "inner"
            assert DeepSeekClient()._route_default() == "outer"
        assert DeepSeekClient()._route_default() is None

    def test_blank_and_none_mean_no_override(self):
        with model_override(""):
            assert DeepSeekClient()._route_default() is None
        with model_override(None):
            assert DeepSeekClient()._route_default() is None
        with model_override("   "):
            assert DeepSeekClient()._route_default() is None

    def test_exit_restores_even_on_error(self):
        try:
            with model_override("x"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert DeepSeekClient()._route_default() is None


class TestConcurrentNoBleed:
    def test_threads_do_not_cross_models(self):
        """⭐ 开发计划 S2 的核心验收：多线程各持不同 model，互不污染。
        （若实现误用全局变量，这里会随机红——正是它能存在的原因。）"""
        models = ["model-a", "model-b", "model-c"]
        results: dict[str, list] = {m: [] for m in models}
        barrier = threading.Barrier(len(models))     # 每轮 3 线程同步（不是 15）

        def worker(m: str):
            for _ in range(5):
                with model_override(m):
                    barrier.wait()                   # 强制并发窗口重叠
                    results[m].append(DeepSeekClient()._route_default())

        threads = [threading.Thread(target=worker, args=(m,)) for m in models]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for m in models:
            assert results[m] and all(got == m for got in results[m]), \
                f"{m} 的上下文被串号: {results[m]}"

    def test_main_context_unaffected_after_threads(self):
        """线程里的覆盖不许泄漏回主线程（ContextVar 按线程隔离的语义）。"""
        def worker():
            with model_override("thread-only"):
                assert DeepSeekClient()._route_default() == "thread-only"

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert DeepSeekClient()._route_default() is None


class TestRealProductionPath:
    """⭐ 2026-10-14 实测缺陷守卫：**生产路径**是「在 override 之外创建 client、
    在 override 之内调用」。

    首版 `_route_default()` 取的是**构造时刻**的覆盖值 ⇒ client 在上下文外创建时
    persona 模型全部失效（实测 3 个 AI 都跑成 qwen-max，花了 9.2 万 tokens 才发现）。
    ⚠️ 当时的测试验的是"在 override 内部构造"，与生产不一致 ⇒ 这里专门锁生产路径。
    """

    def test_client_created_outside_context_still_gets_persona_model(self):
        c = DeepSeekClient()                      # ← 上下文之外创建（生产就是这么写的）
        with model_override("deepseek-chat"):
            assert c._route_default() == "deepseek-chat"
        assert c._route_default() is None          # 退出后还原

    def test_routing_end_to_end_with_persona_model(self):
        """端到端：按 run_persona_round 的顺序，解析结果必须是 persona 的模型。"""
        from config.model_routing import resolve_model
        c = DeepSeekClient()
        with model_override("deepseek-chat"):
            got = resolve_model("ai.ai_risk.decision", explicit=None,
                                default=c._route_default())
        assert got == "deepseek-chat", "persona 绑定必须在真实调用顺序下生效"

    def test_explicit_construction_still_wins(self):
        """构造时显式指定仍是最高优先（系统配置页"当场选模型测连通性"依赖它）。"""
        with model_override("deepseek-chat"):
            assert DeepSeekClient(model="qwen-max")._route_default() == "qwen-max"


class TestPriorityWithFeatureRouting:
    def test_feature_mapping_wins_over_persona(self, tmp_path, monkeypatch):
        """优先级 = explicit > 功能映射 > persona(default) > 档位。
        功能映射是用户对单个功能的明确意图，赢过 persona 的粗粒度绑定。
        （隔离方式照抄 tests/test_model_routing.py 的 _isolate。）"""
        from config import model_routing as MR
        path = tmp_path / "routing.json"
        path.write_text('{"version":1,"models":{"agents.trader":"mapped-model"}}',
                        encoding="utf-8")
        monkeypatch.setenv(MR.ENV_ROUTING_PATH, str(path))
        MR.invalidate()
        try:
            with model_override("persona-model"):
                c = DeepSeekClient()
                assert MR.resolve_model("agents.trader",
                                        explicit=None,
                                        default=c._route_default()) == "mapped-model"
                # 未配置映射的功能落到 persona 绑定
                assert MR.resolve_model("agents.risk_manager",
                                        explicit=None,
                                        default=c._route_default()) == "persona-model"
        finally:
            MR.invalidate()
