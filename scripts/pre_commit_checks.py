# -*- coding: utf-8 -*-
"""提交前快集闸门（2026-10-23）。

## 为什么需要它
2026-10-23 的教训：全量回归 **3245 项 / 16 分 44 秒**，它天然只能当"最后一步"，
于是出现了两次"护栏红了还没发现"：
- `test_observability_hygiene` 核心层 279 > 278、脚本层 41 > 35；
- 新任务 `sentiment_ref` 15:45 与快照 15:40 撞车。

两者的共同点是：**判据已经写好了，但没人在提交前跑它**。所以这里把四套
"结构性守护"打包成**秒级**快集，让"我记住了"变成"它自己会红"。

## 跑什么（每一项都对应一次真实事故）
| # | 检查 | 对应事故 |
|---|---|---|
| 1 | 语法（全部受管 `.py`） | 手工替换脚本把 def 与 docstring 粘成一行 |
| 2 | `config/*.json` 可解析 | 改人格表时漏逗号 ⇒ 联赛整体起不来 |
| 3 | 四件套（观测性 / 建表契约 / 排期冲突 / 任务分派） | 见上 |
| 4 | 工作区热文件（谁正在改） | 两条线并行，跑全量期间对方改源码 ⇒ 5 用例瞬时红 |
| 5 | 改动是否已生效（对照 `started_at`） | 改完没重启，长驻进程跑的还是旧代码 |

## 用法
```bash
python scripts/pre_commit_checks.py            # 完整快集（约 30~60s）
python scripts/pre_commit_checks.py --fast     # 跳过 pytest（约 2s）
python scripts/pre_commit_checks.py --quiet    # 只在失败时输出
```
退出码：0 = 可以提交；1 = 有阻断项（第 1~3 项）；2 = 仅警告（第 4~5 项）。

⚠️ 它**不能替代全量**：快集只守"已知会静默坏掉的结构性契约"，
行为正确性仍然要靠 `运行测试.bat`。
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import logging
import os
import subprocess
import sys
import time

_logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 受管源码目录：与 check_workspace 保持一致（文档/日志不算）
_SRC_DIRS = ("core", "data", "db", "ui", "scripts", "llm", "config")

# 四件套：每一项都是"结构性的、秒级的"契约守护，缺一不可。
# ⚠️ 这里**只放结构性守护**。行为测试（`test_position_risk` 之类）动辄要
#    建库/联网，放进快集会让它退化成"又一个全量"。
_CONTRACT_TESTS = (
    ("tests/test_observability_hygiene.py", "观测性：静默 except / print 不得增长"),
    ("tests/test_schema_contract.py", "建表契约：丢表 / 新表忘登记"),
    ("tests/test_schedule_conflicts.py", "排期冲突：新增任务不得引入重叠"),
    ("tests/test_job_dispatch.py", "任务分派：mode 必须真能被执行"),
)


def _rel(path: str) -> str:
    """相对项目根显示路径；**跨盘符时退回绝对路径**。

    ⚠️ Windows 上 `os.path.relpath(C:////x, D:////y)` 抛 `ValueError: path is on
    mount 'C:'`。检查器不能因为"文件在另一个盘"就崩 —— 崩了在 `main()` 里
    会被当成"该检查失败"，看起来像源码有问题，实际只是显示路径失败。
    （实测来源：`tmp_path` 在 C:，项目在 D:，自测直接红。）
    """
    try:
        return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")
    except ValueError as e:
        _logger.debug("[快集] 无法转相对路径，改用绝对路径 %s: %s", path, e)
        return str(path).replace(os.sep, "/")


def _iter_sources():
    for d in _SRC_DIRS:
        root = os.path.join(PROJECT_ROOT, d)
        if not os.path.isdir(root):
            continue
        for dp, _dn, fn in os.walk(root):
            if "__pycache__" in dp or ".pytest_cache" in dp:
                continue
            for f in fn:
                if f.endswith(".py"):
                    yield os.path.join(dp, f)


def check_syntax() -> tuple[bool, str]:
    """① 全部受管 `.py` 必须能 `ast.parse`。

    ⚠️ 为什么单独做：我用脚本批量改文件时曾把 `def _kline(...)` 和它的
        docstring 粘成一行，**编辑器不报错**，只有 import 时才炸。
    """
    t0 = time.time()
    bad = []
    n = 0
    for p in _iter_sources():
        n += 1
        try:
            with open(p, encoding="utf-8") as f:
                ast.parse(f.read())
        except SyntaxError as e:
            # 语法错误会被汇总进返回值，但"哪一行"只在日志里能直接在编辑器跳过去
            _logger.debug("[快集] 语法错误 %s:%s %s", p, e.lineno, e.msg)
            bad.append(f"{_rel(p)}:{e.lineno} {e.msg}")
        except UnicodeDecodeError as e:
            # 编码问题必须单独报：Windows 下 ANSI 混入会让 ast 抛这个而不是 SyntaxError
            _logger.debug("[快集] 编码异常 %s: %s", p, e)
            bad.append(f"{_rel(p)} 不是合法 UTF-8")
    dt = time.time() - t0
    if bad:
        return False, f"{len(bad)}/{n} 个文件语法错误：\n      " + "\n      ".join(bad[:8])
    return True, f"{n} 个 .py 全部可解析（{dt:.1f}s）"


def check_json() -> tuple[bool, str]:
    """② `config/*.json` 必须可解析 —— 配置坏了整条链路起不来，且报错很晚。"""
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "config", "*.json")))
    bad = []
    for p in files:
        try:
            with open(p, encoding="utf-8") as f:
                json.load(f)
        except (ValueError, OSError) as e:
            _logger.debug("[快集] JSON 解析失败 %s: %s", p, e)
            bad.append(f"{os.path.basename(p)}: {type(e).__name__}")
    dt = time.time() - t0
    if bad:
        return False, f"{len(bad)}/{len(files)} 个配置不可解析：" + "；".join(bad)
    return True, f"{len(files)} 个配置全部可解析（{dt:.1f}s）"


def check_contracts(runner: str) -> tuple[bool, str]:
    """③ 四件套：结构性守护。**用 exit code 判**，不靠输出尾部（会被缓冲吞掉）。"""
    t0 = time.time()
    targets = [p for p, _ in _CONTRACT_TESTS if os.path.exists(os.path.join(PROJECT_ROOT, p))]
    missing = [p for p, _ in _CONTRACT_TESTS if p not in targets]
    if not targets:
        return False, "四件套一个都找不到（路径变了？）"
    if missing:
        # 缺文件本身就是阻断项：护栏被删掉比护栏红更危险
        return False, "契约测试缺失：" + "、".join(missing)

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    # 这些用例不该动真实库：万一有漏网的，至少别让安全删除钩子拦住
    env.setdefault("CODEBUDDY_SAFE_DELETE_ENABLED", "0")
    cmd = [runner, "-m", "pytest", *targets, "-q", "-p", "no:cacheprovider",
           "--no-header", "--tb=short"]
    try:
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=900)
    except subprocess.TimeoutExpired as e:
        _logger.warning("[快集] 四件套超时（>900s），疑似有用例在联网等待: %s", e)
        return False, "四件套超时（>900s）—— 可能有用例在联网等待，先查是不是新加了外呼"
    except OSError as e:
        _logger.warning("[快集] 无法启动 pytest: %s", e)
        return False, f"无法启动 pytest：{e}"

    dt = time.time() - t0
    out = (r.stdout or "") + (r.stderr or "")
    tail = [ln for ln in out.strip().splitlines() if ln.strip()][-1:] or ["（无输出）"]
    if r.returncode != 0:
        fails = [ln for ln in out.splitlines()
                 if ln.startswith(("FAILED ", "ERROR "))]
        detail = "；".join(fails[:6]) if fails else tail[0][:200]
        return False, f"{len(fails) or '?'} 项失败（{dt:.0f}s）：{detail}"
    return True, f"四件套全部通过（{dt:.0f}s）"


def check_hotfiles(minutes: int) -> tuple[bool, str]:
    """④ 最近 N 分钟被改过的受管文件 —— **警告不阻断**。

    为什么不阻断：任何人刚改完自己的文件，它就是"热"的。这条的价值在于
    "知道别人也在动"，而不是"禁止提交"。查热文件是 `check_workspace` 的活。
    """
    cutoff = time.time() - minutes * 60
    hot = []
    for p in _iter_sources():
        try:
            m = os.path.getmtime(p)
        except OSError as e:
            _logger.debug("[快集] 读取 mtime 失败，跳过 %s: %s", p, e)
            continue
        if m >= cutoff:
            hot.append((_rel(p), m))
    if not hot:
        return True, f"最近 {minutes} 分钟无改动"
    hot.sort(key=lambda x: -x[1])
    names = [f"{n}({int((time.time() - m) / 60)}分钟前)" for n, m in hot[:6]]
    return False, (f"{len(hot)} 个文件最近 {minutes} 分钟被改："
                   + "、".join(names) + "（可能是另一条开发线）")


def check_effective() -> tuple[bool, str]:
    """⑤ 改动是否已进到长驻进程 —— **警告不阻断**（不一定需要重启，视改动而定）。"""
    hb = os.path.join(PROJECT_ROOT, "temp", "scheduler_heartbeat.json")
    try:
        with open(hb, encoding="utf-8") as f:
            started = json.load(f).get("started_at")
    except (OSError, ValueError) as e:
        _logger.debug("[快集] 读不到调度器心跳（视为未运行）: %s", e)
        return True, "读不到调度器心跳 ⇒ 未判定（进程可能没在跑）"
    if not started:
        # 旧格式心跳没有该字段：明确说"不知道"，不假装 ✓
        return True, "心跳无 started_at ⇒ 无法判定（重启一次调度器即可写入）"
    newer = []
    for p in _iter_sources():
        try:
            if os.path.getmtime(p) > float(started) + 1:
                newer.append(_rel(p))
        except (OSError, ValueError) as e:
            _logger.debug("[快集] 比较 mtime 失败 %s: %s", p, e)
            continue
    if not newer:
        return True, "长驻进程跑的就是最新代码"
    return False, (f"{len(newer)} 个文件晚于调度器启动 ⇒ 需重启才生效："
                   + "、".join(newer[:6]) + ("…" if len(newer) > 6 else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="提交前快集闸门")
    ap.add_argument("--fast", action="store_true", help="跳过 pytest（只做语法/配置）")
    ap.add_argument("--quiet", action="store_true", help="只在失败时输出")
    ap.add_argument("--minutes", type=int, default=15, help="热文件回看分钟数")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    runner = sys.executable or "python"

    steps = [("语法", check_syntax), ("配置", check_json)]
    if not args.fast:
        steps.append(("契约", lambda: check_contracts(runner)))
    steps += [("并发", lambda: check_hotfiles(args.minutes)),
              ("生效", check_effective)]

    blocking = {"语法", "配置", "契约"}
    results = []
    for i, (name, fn) in enumerate(steps, 1):
        try:
            ok, msg = fn()
        except Exception as e:                      # noqa: BLE001
            # 检查器自己崩了必须显式暴露，不能当成"通过"
            _logger.warning("[快集] %s 检查器异常: %s", name, e)
            ok, msg = False, f"检查器异常：{type(e).__name__}: {e}"
        results.append((name, ok, msg))
        if not args.quiet or not ok:
            mark = "✓" if ok else ("✗" if name in blocking else "!")
            print(f"  {mark} [{i}/{len(steps)}] {name}：{msg}")

    bad = [(n, m) for n, ok, m in results if not ok and n in blocking]
    warn = [(n, m) for n, ok, m in results if not ok and n not in blocking]
    print()
    if bad:
        print(f"✗ 有 {len(bad)} 项阻断，不能提交：" + "、".join(n for n, _ in bad))
        return 1
    if warn:
        print(f"! 通过（{len(warn)} 项警告，你确认后可提交）：" + "、".join(n for n, _ in warn))
        return 2
    print("✓ 快集全绿 —— 仍请记住它不等于全量（行为正确性走 `运行测试.bat`）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
