# -*- coding: utf-8 -*-
"""系统分析扫描器（**只读**）：UX 反模式 + 数据真实性 + 可观测性风险。

用法：`python scripts/system_scan.py`（不写库、不联网）。

产出可复现的计数与定位，供优化方案引用（避免"凭印象"）。
"""
from __future__ import annotations

import io
import os
import re
import sys

# ⚠️ 本文件在 scripts/ 下 ⇒ 项目根是它的**上一级**（写错会扫不到任何文件却"看起来正常"）。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DIRS = ("ui", "core", "data", "db", "llm", "scripts")


def files():
    for d in DIRS:
        for dp, _dn, fn in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" in dp:
                continue
            for f in fn:
                if f.endswith(".py"):
                    yield os.path.join(dp, f)


def rel(p: str) -> str:
    return os.path.relpath(p, ROOT).replace("\\", "/")


def read(p: str) -> str:
    return io.open(p, encoding="utf-8").read()


def scan_silent_except():
    """`except` 块里既没有日志、也没有告知用户的 —— 真正的"静默异常"。"""
    hits = []
    for p in files():
        lines = read(p).splitlines()
        for i, ln in enumerate(lines):
            m = re.match(r"^(\s*)except\s", ln)
            if not m:
                continue
            indent = len(m.group(1))
            body = []
            for nxt in lines[i + 1:]:
                if not nxt.strip():
                    body.append(nxt)
                    continue
                cur = len(nxt) - len(nxt.lstrip())
                if cur <= indent:
                    break
                body.append(nxt)
            blob = "\n".join(body)
            if not re.search(r"(_logger|logging|st\.(warning|error|info|caption)|raise|"
                             r"warnings\.warn)", blob):
                hits.append((rel(p), i + 1, body[0].strip() if body else ""))
    return hits


def scan_or_zero():
    """把"缺失"写成 0 的高危模式（`or 0` / `if not x: x = 0`）。"""
    pats = [
        (r"\bor\s+0\.0\b", "or 0.0"),
        (r"\bor\s+0\b(?!\.)", "or 0"),
        (r"if\s+not\s+\w+:\s*\n\s*\w+\s*=\s*0", "if not x: x=0"),
    ]
    hits = []
    for p in files():
        src = read(p)
        for pat, name in pats:
            for m in re.finditer(pat, src):
                line = src[:m.start()].count("\n") + 1
                hits.append((rel(p), line, name))
    return hits


def scan_form_defaults():
    """数字输入框带"看起来像真实数据"的默认值。

    ⚠️ 必须**按括号配对**截取调用文本：否则 `min_value=0.01` 里的 `value=0.01`
       会被自己的正则匹配上（第一版扫描器就踩了这个坑，把"卖出价"报成了 0.01）。
    """
    hits = []
    for p in files():
        src = read(p)
        for m in re.finditer(r"(?:st\.)?number_input\(", src):
            i = m.end()
            depth, start = 1, i
            while i < len(src) and depth:
                if src[i] == "(":
                    depth += 1
                elif src[i] == ")":
                    depth -= 1
                i += 1
            blob = src[start:i - 1]
            lab = re.search(r"\"([^\"]+)\"", blob)
            vm = re.search(r"(?<![A-Za-z_])value\s*=\s*([^,\)]+)", blob)
            if not vm:
                continue
            val = vm.group(1).strip()
            if re.fullmatch(r"0(\.0+)?", val) or val == "None":
                continue                     # 0 / 空 = 安全的"没填"
            line = src[:m.start()].count("\n") + 1
            hits.append((rel(p), line, lab.group(1) if lab else "?", val))
    return hits


def scan_section_nav():
    """哪些页面用了区块导航（其余页面 = 长页面只能靠滚动）。"""
    used, unused = [], []
    for f in sorted(os.listdir(os.path.join(ROOT, "ui", "pages"))):
        if not f.endswith(".py") or f.startswith("__"):
            continue
        p = os.path.join(ROOT, "ui", "pages", f)
        (used if "section_nav(" in read(p) else unused).append(f)
    return used, unused


def scan_page_alerts():
    """页面是否把"无数据/取不到"如实说出来（而不是留白）。"""
    rows = []
    for f in sorted(os.listdir(os.path.join(ROOT, "ui", "pages"))):
        if not f.endswith(".py") or f.startswith("__"):
            continue
        src = read(os.path.join(ROOT, "ui", "pages", f))
        rows.append((f,
                     len(re.findall(r"st\.(warning|error|info)\(", src)),
                     len(re.findall(r"st\.spinner\(", src)),
                     len(re.findall(r"except Exception", src))))
    return rows


def main():
    print("=" * 74)
    print("A. 静默异常（except 块里既无日志也无用户提示）")
    se = scan_silent_except()
    print(f"  共 {len(se)} 处")
    by_file = {}
    for f, ln, body in se:
        by_file.setdefault(f, []).append(ln)
    for f, lns in sorted(by_file.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"   {len(lns):3d}  {f}  行 {lns[:6]}")

    print()
    print("=" * 74)
    print("B. 表单数字字段的'像真实数据'的默认值")
    for f, ln, label, val in scan_form_defaults():
        print(f"   {f}:{ln}  「{label}」= {val}")

    print()
    print("=" * 74)
    print("C. '缺失写成 0' 模式")
    oz = scan_or_zero()
    print(f"  共 {len(oz)} 处；按文件 TOP10：")
    cnt = {}
    for f, ln, name in oz:
        cnt.setdefault(f, []).append((ln, name))
    for f, v in sorted(cnt.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"   {len(v):3d}  {f}")

    print()
    print("=" * 74)
    print("D. 区块导航覆盖")
    used, unused = scan_section_nav()
    print(f"  已用（{len(used)}）：{', '.join(used)}")
    print(f"  未用（{len(unused)}）：{', '.join(unused)}")

    print()
    print("=" * 74)
    print("E. 各页面的 warning/error/info、spinner、except 计数")
    print("   %-26s %5s %7s %6s" % ("页面", "提示", "spinner", "except"))
    for f, alerts, spins, exc in scan_page_alerts():
        print("   %-26s %5d %7d %6d" % (f[:26], alerts, spins, exc))


if __name__ == "__main__":
    main()
