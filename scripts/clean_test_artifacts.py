"""清理 pytest 遗留产物（temp/pytest* 临时目录 + output/pytest_*.txt 日志）。

为什么需要它
------------
`运行测试.bat` 每次都用**带时间戳的新 basetemp**（复用旧目录会残留文件句柄，
导致莫名失败，见 MEMORY.md）。代价是每跑一次全量就留一份 ~70-90MB 的临时目录，
长期累积到 GB 级（2026-09-29 实测 62 个目录 / 1.5GB）。

本脚本按「保留最近 N 份」清理，并由 `运行测试.bat` 在每次跑测试前自动调用。

安全边界（务必保持）
--------------------
1. **只删 `temp/` 下名字以 `pytest` 开头的目录**。temp/ 里还有调度器的心跳文件
   （`scheduler_heartbeat.json`）等运行时文件，绝不能被波及。
2. **按 mtime 排序，不按名字**。历史遗留的名字五花八门（`pytest_ui1`、
   `pytest_v3full`、`pytest_full_0929`），没有统一时间戳，按名字排会误判新旧。
3. 删除失败（Windows 上目录被占用）只记录、不抛异常——清理失败不该让测试跑不起来。

用法
----
    python scripts/clean_test_artifacts.py                 # 清理，保留最近 3 份
    python scripts/clean_test_artifacts.py --keep 5        # 保留最近 5 份
    python scripts/clean_test_artifacts.py --dry-run       # 只看会删什么
    python scripts/clean_test_artifacts.py --json          # 机器可读结果
    python scripts/clean_test_artifacts.py --prefix pytest # 只清 pytest*（不碰 pt_*）
    python scripts/clean_test_artifacts.py --max-seconds 1800   # 一次性大扫除
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMP_DIR = _PROJECT_ROOT / "temp"
OUTPUT_DIR = _PROJECT_ROOT / "output"

# 删除失败只记录、不中断（见模块文档第 3 条）——但"记录"必须进日志，
# 否则 `scripts/` 的静默 except 基线守卫会把它算成"悄悄吞掉"。
_logger = logging.getLogger(__name__)

# temp/ 下只碰这些前缀，别的（心跳、锁文件……）一律不动。
# ⚠️⚠️ 2026-10-24 实测：只认 "pytest" 会**永远漏掉**手工传 `--basetemp=temp\pt_xxx`
#   留下的目录（pt_qwen / pt_review_final / pt_review_oc / pt_review_v3 / pt_v2 /
#   pt_wd 共 6 个、1.98GB，5 个月都没被清过）。故并入默认前缀。
TEMP_PREFIXES = ("pytest", "pt_")
# output/ 下的测试日志（与 .gitignore 的 output/pytest_*.txt 保持一致）
OUTPUT_GLOB = "pytest_*.txt"

DEFAULT_KEEP_DIRS = 3       # 保留最近几次运行的临时库（够复查一次失败）
DEFAULT_KEEP_LOGS = 5       # 日志很小，多留几份便于回溯
# ⚠️ 默认预算只够删 0~1 个目录（Defender 实时扫描下 rmtree 单目录可要 10~30s），
#   而每跑一次全量就新增 ~450MB ⇒ 靠默认值永远追不上。要真正清干净用 --max-seconds。
DEFAULT_MAX_SECONDS = 20.0


def _prefixes(prefixes=None) -> tuple:
    """归一化前缀参数：None ⇒ 模块默认；字符串 ⇒ 单元素；大小写不敏感比较"""
    if prefixes is None:
        return TEMP_PREFIXES
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    return tuple(p.lower() for p in prefixes if p)


def _match_prefix(name: str, prefixes: tuple) -> bool:
    return name.lower().startswith(prefixes)


def _dir_size(path: Path) -> int:
    """目录占用字节数（算不出就当 0，不影响删除决策）"""
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def plan_temp_dirs(keep: int = DEFAULT_KEEP_DIRS, temp_dir: Path = None,
                   prefixes=None) -> list:
    """→ 待删除的临时目录列表（**最老的在前**）

    只认 `temp/` 下**目录**且名字命中前缀——同前缀的文件不动
    （历史上有过 `pytest_xxx.txt` 这类日志落在 temp/ 的情况）。

    ⚠️⚠️ 返回顺序必须是"最老在前"（2026-10-24 修正）：旧实现按最新在前返回，
    而 `clean()` 遇到 `max_seconds` 预算就 `continue` 掉剩余项 ⇒ 预算永远花在
    较新的目录上，**最老的那批永远轮不到**。实测 88 个目录从 10-18 堆到 10-23
    就是这么来的。改成一刀切最老的，预算再紧也是单调收敛。
    """
    base = Path(temp_dir) if temp_dir else TEMP_DIR
    if not base.is_dir():
        return []
    pre = _prefixes(prefixes)
    cands = [p for p in base.iterdir()
             if p.is_dir() and _match_prefix(p.name, pre)]
    keep = max(keep, 0)
    if len(cands) <= keep:
        return []
    cands.sort(key=_mtime)                      # 最老在前
    return cands[:len(cands) - keep]


def plan_output_logs(keep: int = DEFAULT_KEEP_LOGS, output_dir: Path = None) -> list:
    """→ 待删除的旧测试日志（同样按 mtime，保留最近 keep 份）"""
    base = Path(output_dir) if output_dir else OUTPUT_DIR
    if not base.is_dir():
        return []
    cands = [p for p in base.glob(OUTPUT_GLOB) if p.is_file()]
    if len(cands) <= max(keep, 0):
        return []
    cands.sort(key=_mtime, reverse=True)
    return cands[max(keep, 0):]


def clean(keep_dirs: int = DEFAULT_KEEP_DIRS, keep_logs: int = DEFAULT_KEEP_LOGS,
          temp_dir: Path = None, output_dir: Path = None, dry_run: bool = False,
          max_seconds: float = DEFAULT_MAX_SECONDS, prefixes=None) -> dict:
    """执行清理，返回结构化结果（便于调用方/测试断言）。

    ⚠️⚠️ `max_seconds`（2026-10-19 实测事故）：temp/ 累积到 8000+ 文件
    （其中 2800+ 个 UI 冒烟用例留下的 .db）时，`_dir_size` + `shutil.rmtree`
    在本机 Defender 实时扫描下**跑了 30 分钟没结束**，`运行测试.bat` 卡在
    "清理"这一步、pytest 根本没启动。
    ⇒ 清理是**尽力而为**的前置步骤，必须有时间预算：超预算就停手，
      剩下的下一轮再清（绝不阻塞测试）。
    ⇒ `max_seconds<=0` 表示**不设预算**（手工大扫除用 `--max-seconds 0`）。

    `prefixes`：None ⇒ 用模块默认（pytest + pt_）；传字符串/元组可收窄。
    """
    # 必须在删除**前**数总数：删完再数就只剩"保留数 - 已删数"了（曾把 3 算成 0）
    total_temp_dirs = count_temp_dirs(temp_dir, prefixes)
    dirs = plan_temp_dirs(keep_dirs, temp_dir, prefixes)
    logs = plan_output_logs(keep_logs, output_dir)

    t0 = time.monotonic()
    budgeted = max_seconds is not None and max_seconds > 0
    removed, failed, freed = [], [], 0
    skipped_by_budget = []
    for d in dirs:
        # ⚠️ 只在 dry-run 才统计体积（真实删除时 size 统计等于把目录走两遍，
        #    是事故里"慢"的一半原因）
        size = _dir_size(d) if dry_run else 0
        if dry_run:
            removed.append(str(d))
            freed += size
            continue
        if budgeted and time.monotonic() - t0 > max_seconds:
            skipped_by_budget.append(str(d))
            continue
        try:
            shutil.rmtree(d)
            removed.append(str(d))
        except OSError as e:                    # 被占用等：记下来，不中断
            _logger.warning("[清理] %s 删除失败: %s", d, e)
            failed.append({"path": str(d), "error": str(e)})

    for f in logs:
        try:
            size = f.stat().st_size
        except OSError as e:
            _logger.warning("[清理] %s 大小读取失败（按 0 计）: %s", f, e)
            size = 0
        if dry_run:
            removed.append(str(f))
            freed += size
            continue
        try:
            f.unlink()
            removed.append(str(f))
            freed += size
        except OSError as e:
            _logger.warning("[清理] %s 删除失败: %s", f, e)
            failed.append({"path": str(f), "error": str(e)})

    return {
        "removed": removed,
        "failed": failed,
        "freed_bytes": freed,
        "freed_mb": round(freed / 1024 / 1024, 1),
        "dry_run": bool(dry_run),
        # dry-run 与真实执行都报"清理后还剩几份"（dry-run 下用总数 - 计划删除数模拟）
        "kept_dirs": max(0, total_temp_dirs - len(dirs)),
        # 超预算没删的（下一轮再清）—— 必须可见，否则"清理了"是假的
        "skipped_by_budget": skipped_by_budget,
        # 本轮结束后**实际**还剩几个：dry-run 一个没删 ⇒ 仍是总数；
        # 真实执行 = 总数 - 真删掉的（失败/超预算的都还在）。比 kept_dirs 更接近真相。
        "remaining_dirs": total_temp_dirs if dry_run
                          else max(0, total_temp_dirs - len(removed)),
    }


def count_temp_dirs(temp_dir: Path = None, prefixes=None) -> int:
    """temp/ 下命中前缀的临时目录个数"""
    base = Path(temp_dir) if temp_dir else TEMP_DIR
    if not base.is_dir():
        return 0
    pre = _prefixes(prefixes)
    return len([p for p in base.iterdir()
                if p.is_dir() and _match_prefix(p.name, pre)])



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="清理 pytest 遗留产物")
    ap.add_argument("--keep", type=int, default=DEFAULT_KEEP_DIRS,
                    help=f"保留最近 N 个临时目录（默认 {DEFAULT_KEEP_DIRS}）")
    ap.add_argument("--keep-logs", type=int, default=DEFAULT_KEEP_LOGS,
                    help=f"保留最近 N 份测试日志（默认 {DEFAULT_KEEP_LOGS}）")
    ap.add_argument("--prefix", action="append", default=None, metavar="P",
                    help="只清命中的前缀，可重复（默认 pytest + pt_）")
    ap.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS,
                    help=f"删除时间预算（默认 {DEFAULT_MAX_SECONDS:g} 秒；<=0 表示不设预算）")
    ap.add_argument("--dry-run", action="store_true", help="只列出会删什么，不实际删除")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = ap.parse_args(argv)

    result = clean(keep_dirs=args.keep, keep_logs=args.keep_logs, dry_run=args.dry_run,
                   max_seconds=args.max_seconds, prefixes=args.prefix)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        verb = "将删除" if args.dry_run else "已删除"
        if not result["removed"]:
            print(f"[清理] 无需清理（临时目录不超过 {args.keep} 个）")
        else:
            # 真实删除不统计体积（_dir_size 等于把目录走两遍，是 10-19 事故里"慢"的一半原因）
            # ⇒ 别打印"释放 0.0 MB"那种假数字，如实说明
            freed_note = (f"释放 {result['freed_mb']} MB" if result["freed_mb"]
                          else "体积未统计（真实删除不做二次遍历）")
            print(f"[清理] {verb} {len(result['removed'])} 项，{freed_note}")
            for p in result["removed"][:20]:
                print(f"  - {p}")
            if len(result["removed"]) > 20:
                print(f"  ... 其余 {len(result['removed']) - 20} 项")
        if result["failed"]:
            print(f"[清理] ⚠️ {len(result['failed'])} 项删除失败（可能被占用）：")
            for f in result["failed"][:5]:
                print(f"  ! {f['path']}: {f['error']}")
        # 必须报告"还剩多少"：只报"已删 N 项"会让人以为清干净了（2026-10-24 实测
        # 88 个目录累积 5 天，正是因为预算耗尽这件事完全不可见）
        if result["skipped_by_budget"]:
            print(f"[清理] ⏸ 超时间预算，本轮跳过 {len(result['skipped_by_budget'])} 项；"
                  f"temp/ 仍有 {result['remaining_dirs']} 个临时目录"
                  f"（手工大扫除用 --max-seconds 0）")
        elif not result["dry_run"] and result["removed"]:
            print(f"[清理] temp/ 剩余 {result['remaining_dirs']} 个临时目录")

    # 清理失败不影响测试流程：始终返回 0（调用方是 bat，不想因清理失败中断）
    return 0


if __name__ == "__main__":
    sys.exit(main())
