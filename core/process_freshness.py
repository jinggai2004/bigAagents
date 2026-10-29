"""长驻进程的「代码陈旧」诊断。

**这个模块存在的唯一理由**：本项目已经**三次**栽在同一件事上——
改了代码，但跑着的进程用的还是启动时加载的旧模块，于是新逻辑**不执行、也不报错**：

1. 调度器加内置任务（要重启）；
2. 调度器**只改 `run_*_job` 的函数体**（2026-10-07：扩展 `run_snapshot_job` 后
   估值快照采集一直没生效，而日志里一个字都没提）；
3. **UI（Streamlit）**（2026-10-08：09:12 启动的进程在 22:38 仍按 21:49 之前的
   旧代码产出报告，导致"修复看起来没生效"，白排查一轮）。

前两次靠"记得重启"这种纪律来防，第三次证明纪律不管用 —— 所以把它变成**可查的事实**：
比对「进程启动时间」与「被 import 的源码文件 mtime」，一眼看出哪个进程是陈旧的。

⚠️ 只监控**被 import 的模块目录**（`core/ llm/ data/ db/ config/ scripts/`），
**不含 `ui/`**：Streamlit 每次交互都会重新执行页面脚本（`ui/pages/*.py`），
改页面不需要重启；把 `ui/` 算进来会天天误报。（`ui/components/*.py` 是被 import 的，
保守起见也不监控 —— 宁可漏报一点点，也不要误报淹没真信号。）
"""

import os
from datetime import datetime

# 被 import 的源码目录（改这些 ⇒ 必须重启长驻进程）
WATCH_DIRS = ("core", "llm", "data", "db", "config", "scripts")
_SKIP_DIRS = {"__pycache__", ".git", "temp", "tests", "output", "logs"}

# 长驻进程的识别特征（cmdline 里出现即视为本项目进程）
WATCHERS = (
    ("UI (streamlit)", ("streamlit", "ui/app.py")),
    ("调度器", ("run_scheduler.py",)),
    # ⚠️ 打包后调度器是 `bigAagents.exe --scheduler`（2026-10-27 实测漏检）：
    #    进程名不叫 python、命令行里也没有 `run_scheduler.py`（入口在 exe 内部），
    #    上面那条**匹配不到** ⇒ 报告显示"未发现长驻进程"，读起来像一切正常。
    #    这正是「未知 ≠ 正常」的反例：exe 跑着旧代码时，这一项本来就是唯一
    #    能发现它的地方，漏检等于把自检变成安慰剂。
    # ⚠️ 必须写成 `bigaagents.exe`，**不能**只写 `bigaagents`：本项目路径里处处
    #    是 `bigAagents`（如 `python -m http.server --directory <项目根目录>/
    #    companion`），只写目录名会把这类无关进程认成面板 —— 2026-10-27 用真 exe
    #    验证时，一个 http.server 被报成"面板跑的是旧代码"。
    ("调度器", ("bigaagents.exe", "--scheduler")),
    # 裸 `bigAagents.exe`（无参数）就是**面板**进程。顺序有意义：上面那条先命中，
    # 否则面板分支会把 `--scheduler` 也一起吞掉。
    ("面板 (exe)", ("bigaagents.exe",)),
)

# 这些进程名才算"本项目可能的长驻进程"。原版只放行 python/streamlit，
# 于是 `bigAagents.exe` 在**读命令行之前**就被 `continue` 掉了 —— 即便
# WATCHERS 里写了匹配规则也永远走不到。
_PROC_NAME_HINTS = ("python", "streamlit", "bigaagents")

# 「一个进程都没识别到」时的表述。必须是警告口径：识别不到 ≠ 代码已生效，
# 它更常见的原因是"没启动"或"以未登记的方式启动"（换入口脚本/换 exe 参数）。
_NO_PROC_REASON = (
    "未识别到任何长驻进程：可能没启动，也可能以未登记的方式运行"
    "（换入口脚本、或 exe 的参数不在 WATCHERS 里）。"
    "本项只比对**已识别**进程的代码新旧 —— 识别不到不代表代码已生效"
)

# 数据根必须**运行时**解析：打包后 `__file__` 位于只读包 `_internal/` 里，
# 用它推导会指向空目录 ⇒ 读到空数据（2026-10-25 实测：双击 exe 后
# 选股/复盘/研究全 missing、交易台变空账户、系统页看不到调度器）。
# 规则见 `config/paths.data_root()`；源码运行结果与原先一致。
from config.paths import data_root as _paths_data_root  # noqa: E402
_PROJECT_ROOT = _paths_data_root()


def _iter_source_files():
    for d in WATCH_DIRS:
        base = os.path.join(_PROJECT_ROOT, d)
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if x not in _SKIP_DIRS]
            for f in files:
                if f.endswith(".py"):
                    yield os.path.join(root, f)


def newest_source() -> tuple[float, str]:
    """返回 (最新源码文件的 mtime, 相对路径)；没有源码返回 (0.0, "")"""
    best_ts, best_rel = 0.0, ""
    for path in _iter_source_files():
        try:
            ts = os.path.getmtime(path)
        except OSError:
            continue
        if ts > best_ts:
            best_ts, best_rel = ts, os.path.relpath(path, _PROJECT_ROOT)
    return best_ts, best_rel


def match_kind(cmd: str) -> str | None:
    """命令行 → 进程种类（`UI (streamlit)` / `调度器` / `面板 (exe)`）；不认识返回 None。

    ⚠️ 必须先把 `\\` 归一成 `/`：Windows 的命令行里是反斜杠
    （`python -m streamlit run ui\\app.py`），直接拿 `"ui/app.py"` 去匹配
    **永远匹配不到** → UI 进程被静默漏检（实测首版只报出调度器，
    UI 明明在跑却显示"未发现进程"——这比不报更误导）。

    ⚠️ 还要**统一转小写**：真实命令行是 `D:/.../bigAagents/bigAagents.exe`，
    带大写 A。不转小写时 `"bigaagents" in cmd` 恒为 False —— 与反斜杠那个坑
    是同一类错误（拿"我以为的写法"去比"实际写出来的字符串"）。
    """
    cmd_norm = str(cmd or "").replace("\\", "/").lower()
    for kind, keys in WATCHERS:
        if all(k in cmd_norm for k in keys):
            return kind
    return None


def scan_processes() -> dict:
    """找出本项目的长驻进程及其启动时间。

    Returns:
        {"checked": bool, "reason": str|None, "processes": [{"pid","kind","cmdline",
         "started_at","started_ts"}]}

    ⚠️ psutil 不可用时返回 `checked=False` + `reason`，**不要**返回"一切正常"：
    "判不了"与"没问题"是两件事（本项目在心跳清单那件事上已经吃过一次亏）。
    """
    try:
        import psutil
    except ImportError:
        return {"checked": False, "reason": "psutil 未安装，无法读取进程启动时间",
                "processes": []}

    out = []
    for proc in psutil.process_iter(["pid", "name", "create_time", "cmdline"]):
        try:
            info = proc.info
            name = (info.get("name") or "").lower()
            # ⚠️ 这里曾只放行 python/streamlit ⇒ `bigAagents.exe` 在**读命令行之前**
            #    就被丢掉，WATCHERS 里写得再对也走不到（2026-10-27 实测：
            #    exe 调度器在跑，`check()` 返回的 processes 仍是空列表）。
            if not any(h in name for h in _PROC_NAME_HINTS):
                continue
            cmd = " ".join(info.get("cmdline") or [])
            kind = match_kind(cmd)
            if kind:
                out.append({
                    "pid": info["pid"], "kind": kind,
                    "cmdline": cmd[:110],
                    "started_ts": float(info.get("create_time") or 0),
                    "started_at": datetime.fromtimestamp(
                        info.get("create_time") or 0).strftime("%Y-%m-%d %H:%M:%S"),
                })
        except Exception:                       # noqa: BLE001 - 进程随时可能退出
            continue
    return {"checked": True, "reason": None,
            "processes": sorted(out, key=lambda p: p["started_ts"])}


def newer_sources(than_ts: float, limit: int = 3) -> list[str]:
    """比 `than_ts` 更新的源码文件（相对路径，按 mtime 倒序，最多 limit 个）"""
    hits = []
    for path in _iter_source_files():
        try:
            ts = os.path.getmtime(path)
        except OSError:
            continue
        if ts > than_ts:
            hits.append((ts, os.path.relpath(path, _PROJECT_ROOT)))
    hits.sort(reverse=True)
    return [rel for _ts, rel in hits[:limit]]


def evaluate(scan: dict, newest_ts: float = None, newest_rel: str = "",
             probe=None) -> dict:
    """把「进程清单 + 最新源码时间」判成"哪些进程跑的是旧代码"。

    抽成纯函数是为了可测：真实进程的启动时间没法在测试里造假，
    但这份判断逻辑（阈值、取哪些文件、怎么表述）必须被钉住。

    Args:
        scan: `scan_processes()` 的结果
        newest_ts: 最新源码 mtime（None 则自动扫描）
        probe: 可注入的"比 t 更新的文件"查询函数（测试用）；默认走 `newer_sources`

    Returns:
        {"checked": bool, "reason": str|None, "newest_code": {"mtime","file"},
         "processes": [{"pid","kind","started_at","stale","newer_files"}],
         "stale": [kind...]}
    """
    if probe is None:
        probe = newer_sources
    if not scan.get("checked"):
        return {"checked": False, "reason": scan.get("reason"),
                "newest_code": None, "processes": [], "stale": []}
    if newest_ts is None:
        newest_ts, newest_rel = newest_source()

    processes, stale = [], []
    for p in scan.get("processes") or []:
        files = list(probe(p.get("started_ts") or 0))
        row = {**p, "stale": bool(files), "newer_files": files}
        processes.append(row)
        if row["stale"]:
            stale.append(p.get("kind") or f"pid {p.get('pid')}")
    # ⚠️ 识别不到进程时**不能**回 `reason=None`：那读起来是"检查过了，没问题"。
    #    真实含义是"这一项没覆盖到任何东西"，必须把话说出来（未知 ≠ 正常）。
    return {"checked": True, "reason": (None if processes else _NO_PROC_REASON),
            "newest_code": {"mtime": newest_ts, "file": newest_rel},
            "processes": processes, "stale": stale}


def check() -> dict:
    """一步到位：扫描进程 + 判定陈旧（供 check_health / UI 调用）。"""
    return evaluate(scan_processes())


def describe(result: dict) -> list[str]:
    """把结果转成人类可读行（供 CLI / UI 展示）。"""
    if not result.get("checked"):
        return [f"⚠️ 无法判断长驻进程是否陈旧：{result.get('reason')}"]
    procs = result.get("processes") or []
    if not procs:
        # 由 `evaluate` 给的 reason 说明"识别不到"≠"没问题"；缺失时也要保留警告口径
        return [f"⚠️ {result.get('reason') or _NO_PROC_REASON}"]
    lines = []
    for p in procs:
        tag = "⚠️ 跑的是旧代码" if p["stale"] else "✅ 代码是新的"
        line = f"{p['kind']}（PID {p['pid']}）启动于 {p['started_at']} — {tag}"
        if p["stale"]:
            line += f"；启动之后被改过的文件：{', '.join(p['newer_files'])}"
        lines.append(line)
    if any(p["stale"] for p in procs):
        lines.append("⇒ 改动不会生效：请重启对应进程"
                     "（调度器双击 `重启调度器.bat`；UI 重跑 `streamlit run ui/app.py`）。")
    return lines
