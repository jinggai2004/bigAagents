# -*- coding: utf-8 -*-
"""⚠️ 口径 6 的结构性守护：AI 产物**绝不外流**到主系统。

AI 的交易结果若回灌到选股/分析提示词或评分，就是自我实现的循环
（与热点页知乎区块"不注入、不参与评分"是同一条纪律）。
行为测试拦不住"将来有人顺手接上"——**结构性守护**（源码级断言）才拦得住。
"""
import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# 主分析链的入口模块：**不许** import 任何 ai_* 联赛模块（反向引用是允许的：
# 联赛模块读主链的产物是"单向只读"，正是我们要的方向）。
_GUARDED = [
    "core/agents.py",
    "core/agents_rules.py",         # 2026-10-17 拆出：规则常量与纯函数同样不许回灌
    "core/screener.py",
    "core/nl_screener.py",
    "core/news_industry_review.py",
    "core/decision_digest.py",
    "core/local_pool.py",
    "llm/deepseek_client.py",       # model_override 是通用机制，不算外流；但仍不许 import ai_*
]

# 联赛模块（读方，允许单向引用主链）
_AI_MODULES = ("ai_personas", "ai_budget", "ai_agent", "ai_wake", "ai_league",
               "ai_daily_round")


def _imports_of(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class TestNoAiBleedIntoMainChain:
    def test_main_chain_does_not_import_ai_modules(self):
        offenders = {}
        for rel in _GUARDED:
            p = _ROOT / rel
            if not p.exists():
                continue
            hits = {n for n in _imports_of(p)
                    if any(n == m or n.endswith("." + m) for m in _AI_MODULES)}
            if hits:
                offenders[rel] = hits
        assert not offenders, (
            f"主分析链引用了 AI 联赛模块（违反单向只读纪律）: {offenders}")

    def test_ai_modules_do_not_write_back(self):
        """联赛模块可以**读**主链，但不许把结果写回主链的决策表
        （analyses / screen_picks 的写入函数只能由主链调用）。"""
        writers = ("insert_analysis", "save_screen_picks", "insert_screen_picks")
        for rel in ("core/ai_agent.py", "core/ai_budget.py", "core/ai_personas.py"):
            p = _ROOT / rel
            if not p.exists():
                continue
            src = p.read_text(encoding="utf-8")
            hits = [w for w in writers if w in src]
            assert not hits, f"{rel} 调用了主链写函数 {hits}（只许读）"
