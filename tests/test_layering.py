"""分层守护测试：db ← data ← core ← ui/scripts，utils 谁都能用

为什么需要它：
    分层的价值在于"改一层不会牵连另一层"，但没有测试时它只是一句口号。
    本文件用 AST 扫描真实 import（不是 grep 字符串，避免注释/字符串误报），
    把方向写反的依赖挡在提交之前。

    已知的两处历史违规都是这么被发现的：
      - core/screener.py → core.ui_tasks（业务层依赖 UI 基建）已修，见 utils/task_control.py
      - data/source_manager.py → core.logger（data 反向依赖 core）已修，见 utils/logger.py

允许方向：
    db      只能 import 同层 + utils
    data    只能 db / 同层 + utils
    core    只能 db / data / 同层 + utils
    ui      只能 db / data / core / scripts / 同层 + utils
    scripts 只能 db / data / core / 同层 + utils
    utils   只能同层
"""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 每一层允许 import 的顶层包（同层与 utils 自动允许，不在此列）
ALLOWED_UPSTREAM = {
    "db": set(),
    "data": {"db"},
    "core": {"db", "data"},
    "ui": {"db", "data", "core", "scripts"},
    "scripts": {"db", "data", "core"},
    "utils": set(),
}

# 尚未修完的反向依赖白名单。
# 每条必须写清"为什么现在还没修"，否则等于给违规开后门。
# ⚠️ 新增条目前先想想能不能下沉/上提；本列表只应减少，不应增加。
KNOWN_VIOLATIONS = {
    # core/pipeline.py 的回测阶段要复用 scripts/backtest_rule_based.py 里的
    # RuleSignalGenerator（约 450 行）。正确修法是把这个类下沉到 core/，
    # 脚本再反过来引用它——属于独立改动，不在本次清理范围内，登记待办。
    ("core/pipeline.py", "scripts.backtest_rule_based"),
}


def _iter_py_files(layer: str):
    layer_dir = PROJECT_ROOT / layer
    if not layer_dir.exists():
        return []
    return sorted(p for p in layer_dir.rglob("*.py")
                  if "__pycache__" not in p.parts and not p.name.startswith("test_"))


def _imported_modules(path: Path):
    """返回文件里所有 import 的模块名（含函数级延迟 import）"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - 语法错误由其它测试负责
        return []
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            mods.append((node.module or "", node.lineno))
    return mods


def _violations(layer: str):
    bad = []
    for path in _iter_py_files(layer):
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        for mod, lineno in _imported_modules(path):
            top = mod.split(".")[0]
            if top not in ALLOWED_UPSTREAM:
                continue          # 第三方包 / 标准库，不关心
            if top == layer or top == "utils":
                continue          # 同层与 utils 允许
            if top not in ALLOWED_UPSTREAM[layer]:
                bad.append((rel, mod, lineno))
    return bad


@pytest.mark.parametrize("layer", ["db", "data", "utils"])
def test_lower_layers_have_no_reverse_dependency(layer):
    """db / data / utils 是最底层，任何反向依赖都必须为 0（不设白名单）"""
    bad = _violations(layer)
    assert bad == [], f"{layer} 层出现反向依赖：{bad}"


def test_core_layer_only_known_violations():
    """core 层除白名单外不得 import ui / scripts"""
    # 统一成 (文件, 顶层.次层) 的形式与白名单比对
    bad = {(rel, ".".join(mod.split(".")[:2])) for rel, mod, _ in _violations("core")}
    unknown = bad - KNOWN_VIOLATIONS
    assert unknown == set(), (
        f"core 层出现新的反向依赖（不允许新增，请修代码而不是加白名单）：{sorted(unknown)}"
    )


def test_known_violations_are_still_present():
    """白名单条目若已被修掉，测试要提醒删条目——否则白名单会永久躺在代码里"""
    current = {(rel, ".".join(mod.split(".")[:2])) for rel, mod, _ in _violations("core")}
    stale = KNOWN_VIOLATIONS - current
    assert stale == set(), (
        f"以下白名单条目已不存在，请从 KNOWN_VIOLATIONS 删除：{sorted(stale)}"
    )


def test_ui_layer_has_no_reverse_dependency():
    """ui 是最上层，允许 db/data/core/scripts；这里守护它不出现别的反向依赖"""
    bad = _violations("ui")
    assert bad == [], f"ui 层出现反向依赖：{bad}"


def test_layering_scan_actually_ran():
    """守护扫描本身：防止路径写错导致"零违规"其实是什么都没扫到"""
    total = sum(len(_iter_py_files(layer)) for layer in ALLOWED_UPSTREAM)
    assert total >= 50, f"扫描到的源文件只有 {total} 个，扫描范围可能写错了"
