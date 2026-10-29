"""代码气味静态扫描（只读 / 不联网 / 不写文件）：给"改进空间"提供可复现证据。

为什么单独一个脚本
------------------
`except … pass`、裸 `except`、超长函数、分层反向依赖、非 CLI `print` 这些指标
只能靠扫描统计，手工数不现实；而项目已有 `scripts/system_scan.py`（表单默认值/导航/
spinner 一类），两者互补：
    - `scripts/system_scan.py`   → **UI/交互**层的气味
    - `scripts/audit_code_smells.py` → **代码结构**层的气味（本文件）
    - `scripts/audit_baseline.py`    → **"改前/改后"数字**（规模、闭环样本、面板、版本库）

用法：
    python scripts/audit_code_smells.py

⚠️ 这些数字**只用于定位**，结论仍要人工核实（项目已有教训：`spinner` 计数是弱信号，
   而 `print` 计数、静默 `except` 计数是真问题）。
"""
from __future__ import annotations

import ast
import collections
import io
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"temp", "__pycache__", ".git", ".pytest_cache", "output", "logs", "data"}


def iter_py(include_data=True):
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT)
        parts = set(rel.parts)
        if parts & SKIP_DIRS:
            continue
        if not include_data and rel.parts[0] == "data":
            continue
        yield p


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    files = list(iter_py())
    print(f"[范围] {len(files)} 个 .py（不含 temp/output/logs/.git）")

    # ---------- 1. 体积 ----------
    loc = []
    for p in files:
        n = len(read(p).splitlines())
        loc.append((n, p.relative_to(ROOT).as_posix()))
    loc.sort(reverse=True)
    print("\n=== 1. 最大的 15 个文件 ===")
    for n, f in loc[:15]:
        print(f"{n:6d}  {f}")

    # ---------- 2. 函数长度 ----------
    funcs = []
    parses = {}
    parse_fail = []
    for p in files:
        src = read(p)
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            parse_fail.append((p, e))
            continue
        parses[p] = (src, tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(node, "end_lineno", node.lineno)
                funcs.append((end - node.lineno + 1, p.relative_to(ROOT).as_posix(), node.name,
                              node.lineno, len(node.body)))
    funcs.sort(reverse=True)
    print(f"\n=== 2. 最长的 15 个函数（共 {len(funcs)} 个函数/方法）===")
    for n, f, name, ln, stmts in funcs[:15]:
        print(f"{n:5d} 行  {f}:{ln}  {name}  (顶层语句 {stmts})")
    long50 = sum(1 for n, *_ in funcs if n > 50)
    long100 = sum(1 for n, *_ in funcs if n > 100)
    print(f"  >50 行: {long50}   >100 行: {long100}")

    # ---------- 3. 异常与日志 ----------
    bare = silent = reraise_only = 0
    silent_by_file = collections.Counter()
    except_total = 0
    for p, (src, tree) in parses.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            except_total += 1
            if node.type is None:
                bare += 1
            body = node.body
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                silent += 1
                silent_by_file[p.relative_to(ROOT).as_posix()] += 1
            elif len(body) == 1 and isinstance(body[0], ast.Raise):
                reraise_only += 1
    print("\n=== 3. 异常处理 ===")
    print(f"except 总数 {except_total}；裸 except {bare}；只 pass {silent}；只 raise {reraise_only}")
    for f, n in silent_by_file.most_common(10):
        print(f"  {n:3d}  {f}")

    # ---------- 4. print ----------
    print("\n=== 4. print 调用（非 __main__/CLI）===")
    per_dir = collections.Counter()
    main_prints = 0
    for p, (src, tree) in parses.items():
        d = p.relative_to(ROOT).parts[0]
        # 找到 if __name__ == "__main__" 的行区间
        main_ranges = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                t = ast.dump(node.test)
                if "__main__" in t:
                    main_ranges.append((node.lineno, getattr(node, "end_lineno", node.lineno)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                if any(a <= node.lineno <= b for a, b in main_ranges):
                    main_prints += 1
                else:
                    per_dir[d] += 1
    for d, n in per_dir.most_common():
        print(f"  {n:4d}  {d}/")
    print(f"  ({main_prints} 处 CLI print 已排除)")

    # ---------- 5. 分层违规 ----------
    print("\n=== 5. 分层（按 import 依赖）===")
    deps = collections.defaultdict(set)
    for p, (src, tree) in parses.items():
        rel = p.relative_to(ROOT)
        if len(rel.parts) < 2:
            continue
        layer = rel.parts[0]
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for nm in names:
                top = nm.split(".")[0]
                if top in {"core", "data", "db", "ui", "llm", "utils", "config", "scripts"} and top != layer:
                    deps[layer].add(top)
    order = {"utils": 0, "config": 0, "db": 1, "data": 2, "llm": 2, "core": 3, "ui": 4, "scripts": 4}
    for layer in sorted(deps, key=lambda x: order.get(x, 9)):
        print(f"  {layer:8s} -> {sorted(deps[layer])}")
        for t in deps[layer]:
            if order.get(t, 9) > order.get(layer, 9):
                print(f"      ⚠️ 反向依赖: {layer} -> {t}")

    # ---------- 6. 全局可变状态 / 单例 / 缓存 ----------
    print("\n=== 6. 模块级可变全局与缓存 ===")
    pats = {
        "_CACHE": re.compile(r"^_[A-Z_]*CACHE[A-Z_]*\s*[:=]", re.M),
        "lru_cache": re.compile(r"@(?:functools\.)?lru_cache"),
        "global 声明": re.compile(r"^\s+global\s+\w+", re.M),
        "get_singleton": re.compile(r"^_[A-Z_]*INSTANCE[A-Z_]*\s*[:=]", re.M),
        "monkeypatch 请求": re.compile(r"requests\.(get|post)\s*=\s*|Session\.request\s*="),
    }
    for label, pat in pats.items():
        hits = []
        for p, (src, tree) in parses.items():
            m = len(pat.findall(src))
            if m:
                hits.append((m, p.relative_to(ROOT).as_posix()))
        hits.sort(reverse=True)
        print(f"  {label}: {sum(h[0] for h in hits)} 处，涉及 {len(hits)} 文件")
        for m, f in hits[:5]:
            print(f"      {m:3d}  {f}")

    # ---------- 7. 测试规模 ----------
    print("\n=== 7. 测试 ===")
    tfiles = [p for p in (ROOT / "tests").glob("test_*.py")]
    tloc = sum(len(read(p).splitlines()) for p in tfiles)
    sloc = sum(n for n, f in loc if not f.startswith("tests/"))
    print(f"  测试文件 {len(tfiles)}，测试代码 {tloc} 行；源码 {sloc} 行；测试/源码 = {tloc/sloc:.2f}")

    # ---------- 8. 文档宣称 vs 实现 ----------
    print("\n=== 8. 关键常量/契约位置（人工核对用）===")
    keys = ["MAX_SINGLE_STOCK_WEIGHT", "MIN_WEIGHT_COVERAGE", "TRADING_DAY_MODES", "JOBS =", "TOOL_RESULT_MAX_CHARS"]
    for k in keys:
        files_hit = []
        for p, (src, tree) in parses.items():
            if k in src:
                files_hit.append(p.relative_to(ROOT).as_posix())
        print(f"  {k}: {files_hit}")

    if parse_fail:
        print("\n⚠️ 解析失败:", parse_fail)


if __name__ == "__main__":
    main()
