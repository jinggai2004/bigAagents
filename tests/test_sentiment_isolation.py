# -*- coding: utf-8 -*-
"""⚠️ 情绪参照层的**结构性隔离**：产物不许进评分/决策链。

## 为什么必须用 AST 而不是行为测试

"散户看多 ⇒ 加分"这种接线，在功能上**完全跑得通**、测试也全绿 ——
它只是错在方法论上。行为测试永远拦不住"将来有人顺手接上"，
只有源码级断言拦得住（与 `test_ai_isolation.py` 同一条纪律）。

## 允许的方向（单向只读）

    core/review_agent.py、core/agent_tools.py  →  读 sentiment_reference
    core/sentiment_reference.py                 →  不读评分模块

反向（`core/scoring.py` 等 import 本模块）就是评分污染，一律失败。

## 为什么这条纪律值得单独立一个文件

本项目已有实证：`core/replay_eval` 用 989 个交易日证明纯价量因子 IC 全负，
`FACTOR_WEIGHTS` 至今不敢据此修改。在这个前提下，把一条**无法回测**的
"散户情绪"加进评分，等于在唯一被检验过的部分之外开一个黑洞 ——
出问题时既无法归因、也无法证伪。
"""
from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# ⚠️ 本层有**两个**模块，两个都要守：
#    `sentiment_llm` 才是真正调模型的那个（产出标签与报告），
#    只守 `sentiment_reference` 会留一个可直接绕过的后门。
_MODULES = ("sentiment_reference", "sentiment_llm")

# 这些模块是**评分与交易决策**的产出方：不许引用情绪参照层
_GUARDED = [
    "core/scoring.py",
    "core/sector.py",
    "core/analysis_review.py",
    "core/screener.py",
    "core/matching.py",
    "core/risk.py",
    "core/position_risk.py",
    "core/pipeline.py",
]

# 允许读取情绪参照层的消费方（人工可见 / 报告上下文，均不改变分数）
_ALLOWED_READERS = [
    "core/review_agent.py",
    "core/agent_tools.py",
    "core/decision_digest.py",
]


def _hits_any(name: str) -> bool:
    """该导入是否指向情绪层任一模块（`core.X` / `X` / `from core import X`）。"""
    return any(name == m or name.endswith("." + m) for m in _MODULES)


def _imports_of(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            # `from core import sentiment_reference` 这种写法也要认
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


class TestSentimentDoesNotReachScoring:
    def test_scoring_modules_do_not_import_sentiment(self):
        offenders = {}
        for rel in _GUARDED:
            p = _ROOT / rel
            if not p.exists():
                continue
            hits = {n for n in _imports_of(p) if _hits_any(n)}
            if hits:
                offenders[rel] = sorted(hits)
        assert not offenders, (
            "这些评分/决策模块引用了情绪参照层（会把无法回测的散户情绪变成评分依据）：\n"
            + "\n".join(f"  {k}: {v}" for k, v in offenders.items()))

    def test_guard_has_real_targets(self):
        """⚠️ 鉴别力自检：守护名单里的文件必须**都存在**。

        否则路径写错时上面那条断言恒真 —— "看着在守，实际什么也没守"，
        本项目在 `_load_panel` 与闸门基准上都踩过这个坑。
        """
        missing = [rel for rel in _GUARDED if not (_ROOT / rel).exists()]
        assert not missing, f"守卫名单里的文件不存在（断言会变成恒真）: {missing}"

    def test_detector_can_actually_catch_a_violation(self, tmp_path):
        """用临时造出来的违规文件验证检测逻辑本身有效。"""
        bad = tmp_path / "bad_module.py"
        bad.write_text("from core import sentiment_reference as S\n", encoding="utf-8")
        hits = {n for n in _imports_of(bad) if _hits_any(n)}
        assert hits, "AST 检测必须能识别 `from core import sentiment_reference`"


class TestSentimentModuleItselfStaysPure:
    def test_module_does_not_import_scoring_modules(self):
        """反向也要守：情绪层**不许**反向依赖打分/选股，保持单向。"""
        bad = {}
        for mod in _MODULES:
            p = _ROOT / "core" / f"{mod}.py"
            assert p.exists(), f"模块不存在：{mod}"
            hits = {n for n in _imports_of(p)
                    if n.split(".")[-1] in {"scoring", "sector", "screener", "risk",
                                            "position_risk", "matching"}}
            if hits:
                bad[mod] = sorted(hits)
        assert not bad, f"情绪层反向依赖了评分模块: {bad}"

    def test_sentiment_llm_does_not_import_review_agent(self):
        """⚠️ `sentiment_llm` 只许被交互路径/调度调用，不许被复盘反向拉起 ——
        否则复盘一次触发一批模型标注，账单一夜之间翻几倍。"""
        p = _ROOT / "core" / "sentiment_llm.py"
        assert p.exists()
        hits = {n for n in _imports_of(p) if "review_agent" in n}
        assert not hits, f"标注层反向依赖了复盘: {sorted(hits)}"

    def test_allowed_readers_actually_exist(self):
        """消费方名单同样不许写错 —— 它记录了"这个模块当前接到哪里"。"""
        missing = [rel for rel in _ALLOWED_READERS if not (_ROOT / rel).exists()]
        assert not missing, f"消费方名单里的文件不存在: {missing}"
