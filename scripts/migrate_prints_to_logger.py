# -*- coding: utf-8 -*-
"""把 `print(...)` 机械替换为 `_logger.<level>(...)`（2026-10-17，P1-1 观测盲区治理）。

## 为什么

长驻调度进程的日志只收 `logging` 输出 ⇒ `print` 的失败信息**永远进不了**
`logs/scheduler_*.log`。实测事故：资金流快照全天失败，日志里只有
`采集完成: 个股 0 只 / 行业 0 个`，真因在被吞的 print 里。

## 规则（保守优先）

- 只用 **AST 定位** `print(...)` 调用，按**字节偏移**从后往前替换（避免嵌套/同行多调用错位）；
- **只处理恰有 1 个位置参数**的调用；0 参数或 2+ 参数一律**跳过并报告**（不猜怎么拼）；
  带 `*args`/`**kw`/关键字参数的也跳过；
- 级别：消息文本命中 `失败/错误/异常/警告/不可用/熔断/ERROR/WARN/FAIL` ⇒ `warning`，否则 `info`；
- 目标文件若没有 `_logger`，在**最后一个顶层 import 之后**插入
  `import logging`（若缺）+ `_logger = logging.getLogger(__name__)`；
- `--apply` 才写；默认 dry-run 打印清单。

用法：
    python scripts/migrate_prints_to_logger.py data            # 预览
    python scripts/migrate_prints_to_logger.py data --apply    # 写入
"""
from __future__ import annotations

import argparse
import ast
import io
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WARN_HINT = re.compile(r"失败|错误|异常|警告|不可用|熔断|ERROR|WARN|FAIL|✗|❌")


def _targets(root: str) -> list[str]:
    out = []
    for dp, _dn, fn in os.walk(os.path.join(_ROOT, root)):
        if "__pycache__" in dp:
            continue
        out.extend(os.path.join(dp, f) for f in fn if f.endswith(".py"))
    return sorted(out)


def _ensure_logger(src: str, lines: list[str]) -> tuple[str, bool]:
    """确保文件里有 `_logger`。Returns (新源码, 是否改动)。"""
    if re.search(r"^_logger\s*=", src, re.M):
        return src, False
    has_logging = bool(re.search(r"^import logging$", src, re.M))
    tree = ast.parse(src)
    last_import_end = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_end = node.end_lineno or node.lineno
    insert_at = last_import_end          # 1-based 行号，插到它后面
    add = []
    if not has_logging:
        add.append("import logging")
    add.append("")
    add.append("_logger = logging.getLogger(__name__)")
    new_lines = lines[:insert_at] + add + lines[insert_at:]
    return "\n".join(new_lines) + ("\n" if src.endswith("\n") else ""), True


def _line_starts(src: str) -> list[int]:
    """每行首字符在 `src` 里的**字符**偏移。"""
    starts = [0]
    for i, ch in enumerate(src):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _byte_col_to_char(line: str, byte_col: int) -> int:
    """把 AST 的 `col_offset`（**UTF-8 字节偏移**）换成字符偏移。

    ⚠️⚠️ CPython 的 `col_offset`/`end_col_offset` 是**字节**偏移，不是字符偏移。
       中文注释/字符串一旦出现在同一行前面，直接用字符切片就会**切错位置**
       （实测：把 `print(...)` 的调用切短，留下 `tries=3, ...` 变成语法错误）。
       这是本转换器踩的第二个坑。
    """
    raw = line.encode("utf-8")[:byte_col]
    return len(raw.decode("utf-8", errors="ignore"))


def _offset(starts: list[int], lines: list[str], lineno: int, col: int) -> int:
    return starts[lineno - 1] + _byte_col_to_char(lines[lineno - 1], col)


def _main_block_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    """`if __name__ == "__main__":` 的行区间。

    ⚠️ 这些块里的 `print` 是**命令行输出**，换成 logger 会让 `python data/xxx.py`
       静默无输出（取决于 logging 是否配了 stdout handler）⇒ **必须原样保留**。
    """
    out = []
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        src = ast.dump(node.test)
        if "__main__" in src and "name" in src:
            out.append((node.lineno, node.end_lineno or node.lineno))
    return out


def _convert(path: str, apply: bool) -> tuple[int, int, list[str]]:
    """Returns (转换数, 是否补了 _logger, 跳过说明)。"""
    src = io.open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:                                  # pragma: no cover
        return 0, 0, [f"跳过（语法错误）：{e}"]

    lines = src.split("\n")
    starts = _line_starts(src)
    main_ranges = _main_block_ranges(tree)
    spans: list[tuple[int, int, str]] = []
    skipped: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            continue
        ln = node.lineno
        if any(a <= ln <= b for a, b in main_ranges):
            skipped.append(f"行 {ln}：在 __main__ CLI 块内（保留 print）")
            continue
        if node.keywords or len(node.args) != 1:
            skipped.append(f"行 {ln}：参数个数={len(node.args)}"
                           f"{'/含关键字参数' if node.keywords else ''}")
            continue
        seg = ast.get_source_segment(src, node.args[0])
        if seg is None:                                       # pragma: no cover
            skipped.append(f"行 {ln}：无法取到源码片段")
            continue
        level = "warning" if WARN_HINT.search(seg) else "info"
        # ⚠️ 必须按**绝对偏移**整段替换：`print(` 可能跨多行，
        #    只按"行内列号"切片会在跨行时把代码切坏（第一版就这么挂的）。
        start = _offset(starts, lines, node.lineno, node.col_offset)
        end = _offset(starts, lines, node.end_lineno, node.end_col_offset)
        spans.append((start, end, f"_logger.{level}({seg})"))

    if not spans:
        return 0, 0, skipped

    new_src = src
    for start, end, rep in sorted(spans, key=lambda x: -x[0]):
        new_src = new_src[:start] + rep + new_src[end:]
    new_src, added = _ensure_logger(new_src, new_src.split("\n"))
    if apply:
        io.open(path, "w", encoding="utf-8").write(new_src)
    return len(spans), (1 if added else 0), skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="目录（相对项目根），如 data 或 core")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    total = added = 0
    all_skipped: list[str] = []
    for p in _targets(args.root):
        n, a, sk = _convert(p, args.apply)
        if n or a:
            rel = os.path.relpath(p, _ROOT).replace("\\", "/")
            print(f"  {n:3d} 处 print → logger  {'(并补 _logger)' if a else ''}  {rel}")
        total += n
        added += a
        all_skipped += [f"{os.path.relpath(p, _ROOT)} {s}" for s in sk]
    print(f"\n合计：{total} 处转换，{added} 个文件补了 _logger；"
          f"跳过 {len(all_skipped)} 处")
    for s in all_skipped[:20]:
        print("   跳过:", s)
    if not args.apply:
        print("\n（dry-run）加 --apply 才会写入。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
