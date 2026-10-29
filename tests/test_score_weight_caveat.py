"""评分权重的"验证状态披露"契约测试。

背景（2026-10-07）
----------------
`core/sector.py::FACTOR_WEIGHTS` 是**硬编码、从未做过 IC 验证**的权重
（数值来源：删掉北向资金后按原主观比例机械缩放 ÷0.85）。实测发现可检验的
41.2% 权重里，2 个因子（趋势=20日动量、均线排列）的 IC **方向与权重假设相反**。

问题不在于"权重拍脑袋"——早期版本都这样。问题在于**没有任何地方声明这件事**：
- UI 把「综合评分 82.1」当作最醒目的数字呈现；
- AI 助手会把它讲成"该股得到 82 分""某因子被证明有效"。

那等于把未验证的假设包装成数据结论。本文件守住"披露这件事"的三条链路，
以及**措辞边界**（不许从"未验证"升级成"无效"）。
"""

import pytest

from core.sector import FACTOR_WEIGHTS, SCORE_WEIGHT_CAVEAT


# ================================================================
# 1. 说明本身：必须说清三件事，且不许越界
# ================================================================

class TestCaveatContent:
    def test_states_unvalidated_and_direction_concern(self):
        """必须同时点明「未验证」与「方向存疑」——只写前者不够（用户仍会以为方向对）"""
        assert "未经 IC 验证" in SCORE_WEIGHT_CAVEAT
        assert "相反" in SCORE_WEIGHT_CAVEAT, "必须点明实测方向与权重假设相反"

    def test_does_not_overclaim_invalidity(self):
        """⚠️ 措辞边界：只能说"未验证 / 方向存疑"，**不许**升级成"无效"。

        样本只覆盖单一区间（2024 反转市）+ 单一池（当期成分）+ 41.2% 的权重。
        把它写成"这些因子无效"是超出数据的断言，会误导用户放弃本来可能有用的因子。
        """
        banned = ["因子无效", "权重无效", "已被证伪", "完全没有用", "因子失效"]
        hit = [w for w in banned if w in SCORE_WEIGHT_CAVEAT]
        assert not hit, f"披露越界成了断言，超出样本能支持的结论：{hit}"

    def test_points_to_evidence(self):
        """必须给出可核查的依据路径，而不是一句无从验证的免责声明"""
        assert "因子有效性分析报告" in SCORE_WEIGHT_CAVEAT

    def test_is_short_enough_to_be_read(self):
        """披露要能被读完 —— 超过一屏的免责声明等于没有"""
        assert len(SCORE_WEIGHT_CAVEAT) < 400, len(SCORE_WEIGHT_CAVEAT)

    def test_weights_still_hardcoded(self):
        """这条披露的前提是"权重仍是硬编码常量"。若哪天改成数据驱动，本文件的表述要重写。

        这里钉住前提，避免"权重已被 IC 优化过、却还在说未经检验"的反向误导。
        """
        assert sum(FACTOR_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-6)
        assert len(FACTOR_WEIGHTS) == 6


# ================================================================
# 2. 三条披露链路（UI / 系统提示词 / 工具输出）
# ================================================================

class TestCaveatReachesEverySurface:
    def test_system_prompt_references_the_single_source(self):
        """系统提示词必须**引用同一个常量**，不是另抄一份文案。

        ⚠️ 抄文案的后果：改了 `SCORE_WEIGHT_CAVEAT` 之后提示词还是旧的，
        而不报错、测试也全绿 —— 正是本项目反复踩的"文案漂移"。
        这里用 `in` 断言（而不是"包含某个词"），漂移即变红。
        """
        from core.chat_agent import SYSTEM_PROMPT
        assert SCORE_WEIGHT_CAVEAT in SYSTEM_PROMPT, (
            "系统提示词没有引用 SCORE_WEIGHT_CAVEAT（另抄了一份？）")

    def test_system_prompt_tells_model_how_to_phrase_it(self):
        """光给事实不够，还要给出**说法**：否则模型仍可能说成"得到 82 分\""""
        from core.chat_agent import SYSTEM_PROMPT
        assert "排序评分" in SYSTEM_PROMPT
        assert "不要**说成" in SYSTEM_PROMPT or "不要" in SYSTEM_PROMPT

    def test_screen_tool_output_carries_caveat(self, monkeypatch):
        """选股工具返回值里必须带「评分说明」—— 跟着数据走，不靠模型记规则。

        行为级验证（不是读源码）：把 screener 换成返回固定候选的替身，
        检查真实工具函数产出的 payload 里有没有这条。
        """
        from core import agent_tools as at
        from core.screener import StockScreener

        monkeypatch.setattr(StockScreener, "run", lambda self, **kw: {
            "candidates": [{"股票代码": "600519", "股票名称": "贵州茅台",
                            "score": 82.1, "策略": "主力资金", "factors": {}}],
        })
        out = at.tool_screen_stocks()
        assert out.get("候选数") == 1
        assert out.get("评分说明") == SCORE_WEIGHT_CAVEAT

    def test_screen_page_renders_caveat(self):
        """选股页必须渲染这条说明（源码级守护：页面渲染要在 AppTest 里跑很重，
        但"文案有没有接线"这件事用源码断言已经足够，且漂移即变红）"""
        import pathlib
        page = pathlib.Path("ui/pages/1_📊_选股.py").read_text(encoding="utf-8")
        assert "SCORE_WEIGHT_CAVEAT" in page, "选股页没有引用评分说明常量"
        assert "评分怎么读" in page, "说明没有以用户看得懂的小标题呈现"
