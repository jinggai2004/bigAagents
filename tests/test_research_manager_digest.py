# -*- coding: utf-8 -*-
"""研究汇总节点的输入瘦身（2026-10-20 降本）。

**为什么改**：实测 8 只分析里 `research_manager` 一个节点占**全部成本的 40%**
（¥0.166 / 8 只 = ¥0.021/只），输入 6,960 token、输出 2,299 token。

拆开看输入：三份报告摘要约 2,400（cap 1200×3）、两轮辩论约 3,100、模板约 1,400。
所以两条瘦身路径：报告 cap 1200→900、辩论只压**非最后一轮**（最后收敛那轮保全文）。

⚠️ 必须记住的边界：**输出占该节点成本的一半**，所以砍输入最多省 1/3，
   "砍输入 60% 省 40% 成本"是错的。真正的降本只能靠减少调用次数或换模型。
"""
from core.agents import _digest_debate, _digest_for_manager, research_manager_node


def _state(bull_rounds=2, bear_rounds=2, arg_chars=2000, report_chars=3000):
    return {
        "stock_code": "600519", "stock_name": "贵州茅台",
        "stock_info": {"current_price": 1329.0},
        "bull_history": [("多" * arg_chars) for _ in range(bull_rounds)],
        "bear_history": [("空" * arg_chars) for _ in range(bear_rounds)],
        "tech_report": "技" * report_chars,
        "fundamental_report": "基" * report_chars,
        "news_report": "舆" * report_chars,
        "errors": [], "fatal_errors": [],
    }


class _Capture:
    def __init__(self):
        self.prompt = ""

    def client(self):
        outer = self

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            def call_api(self, messages, **kw):
                outer.prompt = "\n".join(str(m.get("content") or "") for m in messages)
                return "汇总结果"

        return FakeClient


# ================================================================
# 两个摘要函数
# ================================================================

class TestDigest:

    def test_report_cap_is_900_and_says_truncated(self):
        out = _digest_for_manager("字" * 5000)
        assert len(out) < 1000
        assert "已截断" in out

    def test_short_report_is_untouched(self):
        assert _digest_for_manager("很短") == "很短"
        assert _digest_for_manager("") == "（本次未产出）"

    def test_debate_digest_states_original_length(self):
        out = _digest_debate("论" * 2000)
        assert "仅保留开头论点" in out
        assert "2000" in out          # 写明原长，模型才知道自己看的是节选

    def test_debate_digest_keeps_short_text_intact(self):
        assert _digest_debate("短论点") == "短论点"


# ================================================================
# 端到端：喂给模型的 prompt 确实变短了，且没有静默丢最后一轮
# ================================================================

class TestPromptShrinks:

    def _run(self, monkeypatch, state):
        import core.agents as A
        cap = _Capture()
        monkeypatch.setattr(A, "DeepSeekClient", cap.client())
        out = research_manager_node(state)
        return cap.prompt, out

    def test_prompt_shorter_than_full_concat(self, monkeypatch):
        """4 轮×2000 字辩论 + 3 份×3000 字报告，全文拼接约 17,000 字。

        瘦身后应显著更短（三份报告 3×900 + 末轮 2×2000 + 前轮 2×500 ≈ 7,700）。
        """
        state = _state()
        full_len = (4 * 2000) + (3 * 3000)
        prompt, _ = self._run(monkeypatch, state)
        assert len(prompt) < full_len * 0.6, f"prompt={len(prompt)} 全文={full_len}"
        assert "已截断" in prompt or "仅保留开头论点" in prompt

    def test_last_round_is_kept_in_full(self, monkeypatch):
        """⭐ 反向守卫：最后一轮必须完整 —— 压掉收敛轮等于把最重要的输入删了。

        用独特标记区分轮次：最后一轮带 "FINAL_ROUND_MARK"，它必须原样出现。
        """
        state = _state(bull_rounds=3, bear_rounds=3, arg_chars=1500)
        state["bull_history"][-1] = "FINAL_ROUND_MARK" + ("尾" * 1500)
        state["bear_history"][-1] = "FINAL_ROUND_MARK" + ("尾" * 1500)
        prompt, _ = self._run(monkeypatch, state)
        assert prompt.count("FINAL_ROUND_MARK") == 2      # 多空各一次，且未被截掉

    def test_earlier_rounds_are_compressed(self, monkeypatch):
        state = _state(bull_rounds=3, bear_rounds=3, arg_chars=1500)
        state["bull_history"][0] = "EARLY_MARK" + ("早" * 1500)
        prompt, _ = self._run(monkeypatch, state)
        assert "EARLY_MARK" in prompt                      # 开头保留
        assert prompt.count("仅保留开头论点") >= 2          # 前两轮被标注截断

    def test_single_round_is_not_truncated_at_all(self, monkeypatch):
        """只有一轮时压不动，也不该出现"已截断/仅保留"字样（避免误导）。"""
        state = _state(bull_rounds=1, bear_rounds=1, arg_chars=300, report_chars=200)
        prompt, _ = self._run(monkeypatch, state)
        assert "仅保留开头论点" not in prompt
