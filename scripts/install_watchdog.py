"""安装/卸载调度器守护的 Windows 计划任务（每 5 分钟跑一次 watchdog.py）。

⚠️ 本机（2026-10-20 实测）`schtasks.exe` 被安全策略列入程序黑名单，
   调用会直接 `PermissionError: [WinError 5]`。**遇到的部署环境若也如此**，
   改用常驻循环（随启动器一起启停，无需计划任务）：

       start "bigAagents-守护" "%PY%" "scripts\\watchdog.py" --loop

   `启动bigAagents.bat` 已经内置这一行；`停止bigAagents.bat` 会同时停掉它，
   并写 `temp\\.scheduler_disabled` 防止被守护拉回来。

为什么用 Python 调 `schtasks` 而不是写个 .bat
──────────────────────────────────────────────
`schtasks /tr "<路径>"` 要走 cmd 的 **ANSI 代码页**（中文系统是 936）。
本项目的绝对路径含中文（`D:\\桌面\\bigAagents`），在 bat 里拼这个参数
有实实在在的乱码风险——一旦乱码，计划任务会创建一个**永远跑不起来**的条目，
而 `schtasks` 本身返回成功。

Python 的 `subprocess` 走 `CreateProcessW`（Unicode），没有这个问题。

用法
────
    python scripts/install_watchdog.py            # 安装（幂等，已存在则覆盖）
    python scripts/install_watchdog.py --status   # 查看当前状态 + 最近判定
    python scripts/install_watchdog.py --remove   # 卸载
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TASK_NAME = "bigAagents-Watchdog"
WATCHDOG = ROOT / "scripts" / "watchdog.py"
LOG_FILE = ROOT / "logs" / "watchdog.log"
INTERVAL_MINUTES = 5


def pythonw() -> str:
    """优先用 pythonw.exe（无黑框）。找不到就退回当前解释器（会短暂闪窗）。"""
    exe = Path(sys.executable)
    cand = exe.with_name("pythonw.exe")
    return str(cand if cand.exists() else exe)


def _run(args: list[str]) -> tuple[int, str]:
    """调 schtasks，返回 (退出码, 合并输出)。"""
    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        _logger.debug("[watchdog] 命令执行失败（调用方会展示该消息）: %s", e)
        return 1, f"{type(e).__name__}: {e}"


def _tr_command() -> str:
    """计划任务要执行的完整命令。

    ⚠️ 含空格的路径必须各自加引号；`schtasks` 会把整个 `/tr` 值当作命令行，
       所以这里给的是"带引号的完整命令串"。
    """
    return f'"{pythonw()}" "{WATCHDOG}" --log-file "{LOG_FILE}"'


def install() -> int:
    code, out = _run(["schtasks", "/create", "/tn", TASK_NAME,
                      "/tr", _tr_command(),
                      "/sc", "minute", "/mo", str(INTERVAL_MINUTES), "/f"])
    if code != 0:
        print(f"[install] 创建计划任务失败（退出码 {code}）：\n{out}")
        print("提示：企业策略可能禁止创建计划任务。可改为手动/开机脚本调用：")
        print(f'  "{pythonw()}" "{WATCHDOG}" --log-file "{LOG_FILE}"')
        return code
    print(f"[install] 已创建计划任务「{TASK_NAME}」，每 {INTERVAL_MINUTES} 分钟执行一次。")
    print(f"[install] 命令：{_tr_command()}")
    print(f"[install] 日志：{LOG_FILE}")
    print("[install] 注意：计划任务的**起始于**目录由 bat/py 内部 cwd 决定，"
          "watchdog 自己会 cd 到项目根，不依赖它。")
    return 0


def remove() -> int:
    code, out = _run(["schtasks", "/delete", "/tn", TASK_NAME, "/f"])
    print(f"[remove] {'已删除' if code == 0 else '删除失败'}：{out.strip()[:200]}")
    return code


def status() -> int:
    code, out = _run(["schtasks", "/query", "/tn", TASK_NAME, "/v", "/fo", "list"])
    if code != 0:
        print(f"[status] 计划任务「{TASK_NAME}」不存在（未安装）。")
        return 1
    keep = ("TaskName", "Status", "Last Run Time", "Last Result",
            "Next Run Time", "Task To Run", "Scheduled Task State")
    for line in out.splitlines():
        if any(k.lower() in line.lower() for k in keep):
            print("  " + line.strip())
    # 顺带打印一次判定（不动作），让用户看到"现在该不该重启"
    print("\n[status] 当前判定：")
    import watchdog
    code2, _ = 0, None
    sys.argv = [sys.argv[0], "--status"]
    watchdog.main()
    return code2


def main() -> int:
    ap = argparse.ArgumentParser(description="安装/卸载调度器守护计划任务")
    ap.add_argument("--remove", action="store_true", help="卸载计划任务")
    ap.add_argument("--status", action="store_true", help="查看状态与当前判定")
    args = ap.parse_args()
    if args.remove:
        return remove()
    if args.status:
        return status()
    return install()


if __name__ == "__main__":
    raise SystemExit(main())
