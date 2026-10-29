# -*- coding: utf-8 -*-
"""调度任务接线守护（2026-10-20 事故后建立）。

**事故**：`core/scheduler.py::run_evening_analysis_job` 里写了
`holder = f"scheduler:{os.getpid()}"`，但模块顶部**没有 `import os`**，
而且这一行在 `try` 之外 ⇒ 每次调度必抛 `NameError`，连 `except` 都进不去。

**为什么没人发现**：它自 2026-10-19 接进 `JOBS` 起就没被执行过一次
（10-19 的 18:10 进程没在跑，10-20 的 17:05 进程被外部终止）。已有测试覆盖了
`core.evening_analysis.run_evening_analysis()` 的业务逻辑，但**没有任何测试
真正碰 `run_*_job` 这层包装**——而接线错误恰恰只发生在这一层。

这正是「验了函数 ≠ 验了接线」的又一例。本文件用 `symtable` 做**静态**检查：
把每个函数里"引用了但模块级既没导入也没定义、也不是内置"的名字找出来。
它是结构性的（不需要真的执行任务、不需要网络/LLM），所以能覆盖全部 job。
"""
import ast
import builtins
import symtable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 模块级隐式可用的名字（Python 注入，不需要显式定义）
_IMPLICIT = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
             "__loader__", "__builtins__", "__debug__", "__annotations__"}


def _module_level_names(src: str) -> set:
    """模块级绑定过的名字（import / def / class / 赋值，含 if-try 分支里）。

    ⚠️ 刻意**不进入函数体**：那里面的是局部名，算进来会让守卫漏报
    （函数里随便一个 `x = 1` 都会把未定义的全局 `x` 洗白）。
    """
    out: set = set()

    def collect(stmts):
        for node in stmts:
            if isinstance(node, ast.Import):
                for a in node.names:
                    out.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.name != "*":
                        out.add(a.asname or a.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.add(node.name)
                # 函数体里的 `global x` + 赋值也算模块级绑定
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Global):
                        out.update(sub.names)
            elif isinstance(node, ast.ClassDef):
                out.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        out.add(t.id)
                # symtable 会把**模块级推导式**的循环变量报成"外层 global 引用"
                # （`JOB_IDS = [job_id for job_id, ... in JOBS]` 就是这种），
                # 不把它算作已绑定就会误报。
                for sub in ast.walk(node.value):
                    if isinstance(sub, ast.comprehension):
                        for n in ast.walk(sub.target):
                            if isinstance(n, ast.Name):
                                out.add(n.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                out.add(node.target.id)
            # 条件导入 / try-import：递归进语句块，但不进函数
            if isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                collect(getattr(node, "body", []) or [])
                collect(getattr(node, "orelse", []) or [])
                collect(getattr(node, "finalbody", []) or [])
                for h in getattr(node, "handlers", []) or []:
                    collect(h.body)

    collect(ast.parse(src).body)
    return out


def _referenced_globals(src: str) -> list:
    """[(所在作用域, 名字)]：函数里引用到的全局名字（symtable 会区分局部/全局）。"""
    out: list = []
    table = symtable.symtable(src, "<guard>", "exec")

    def walk(t):
        for sym in t.get_symbols():
            if sym.is_global() and sym.is_referenced():
                out.append((t.get_name(), sym.get_name()))
        for child in t.get_children():
            walk(child)

    walk(table)
    return out


def undefined_globals(src: str) -> list:
    """引用了但模块级未绑定、也不是内置的名字 —— 执行到必 `NameError`。"""
    known = _module_level_names(src) | set(dir(builtins)) | _IMPLICIT
    seen, bad = set(), []
    for scope, name in _referenced_globals(src):
        if name not in known and (scope, name) not in seen:
            seen.add((scope, name))
            bad.append(f"{scope}() → {name}")
    return bad


# ================================================================
# 1. 守卫自检（没有鉴别力的守卫等于没写）
# ================================================================

class TestGuardHasDiscriminance:

    def test_catches_missing_import(self):
        src = "def f():\n    return os.getpid()\n"
        bad = undefined_globals(src)
        assert bad and "os" in bad[0], bad

    def test_catches_missing_name_inside_try(self):
        """try 里也不行 —— 异常会被自己吞掉，反而更难发现。"""
        src = ("def f():\n"
               "    try:\n"
               "        return json.dumps({})\n"
               "    except Exception:\n"
               "        return None\n")
        assert any("json" in b for b in undefined_globals(src))

    def test_ignores_locals_and_imports(self):
        src = ("import os\n"
               "from json import dumps\n"
               "LIMIT = 3\n"
               "def f(x):\n"
               "    y = x + LIMIT\n"
               "    import sys\n"
               "    print(os.getpid(), dumps({}), y, sys.path)\n")
        assert undefined_globals(src) == []

    def test_ignores_conditional_import(self):
        """`try: import x / except: x = None` 这种也算绑定（否则守卫会误报）。"""
        src = ("try:\n"
               "    import orjson as jsonlib\n"
               "except ImportError:\n"
               "    jsonlib = None\n"
               "def f():\n"
               "    return jsonlib\n")
        assert undefined_globals(src) == []


# ================================================================
# 2. 真实模块：不容许任何未定义全局名
# ================================================================

class TestSchedulerWiring:

    def test_scheduler_has_no_undefined_globals(self):
        """⭐ 事故复现点：`run_evening_analysis_job` 用了 `os.getpid()`。

        这里一旦红，说明 `JOBS` 里某个任务执行到某行必然 NameError——
        而它可能因为"从没被执行过"或"异常被 executor 吞掉"而不留任何痕迹。
        """
        src = (ROOT / "core" / "scheduler.py").read_text(encoding="utf-8")
        bad = undefined_globals(src)
        assert not bad, ("core/scheduler.py 引用了未定义的全局名（调度执行到必崩）："
                         + "；".join(bad))

    def test_evening_job_does_not_raise_nameerror(self, tmp_path, monkeypatch):
        """⭐ 事故的**行为级**回归：真跑一次这个包装函数（不联网、不花 token）。

        结构性检查（symtable）能提前发现未定义名；这条是兜底——
        万一将来 `os` 又被人删掉，它会直接红，而不是等到 18:10 静默失败。
        """
        from tests.conftest import use_template_db
        use_template_db(tmp_path, monkeypatch, "job_wiring.db")

        import core.evening_analysis as EA
        from core import scheduler as S
        # 拦住真正的分析（否则会跑 5 只票、花真金白银）
        monkeypatch.setattr(EA, "run_evening_analysis",
                            lambda *a, **k: {"analyzed": 0, "failed": 0,
                                             "skipped_reason": "unit-test"})
        S.run_evening_analysis_job("单测")      # 不应抛 NameError / AttributeError

    def test_job_ids_match_declared_list(self):
        """`JOB_IDS` 必须与 `JOBS` 实际条目一致（健康检查靠它判"跑的是不是新代码"）。"""
        from core import scheduler as S
        assert S.JOB_IDS == [j[0] for j in S.JOBS]
