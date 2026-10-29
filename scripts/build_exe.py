# -*- coding: utf-8 -*-
"""打包 exe（P6）—— PyInstaller **folder** 模式。

用法：
    py -3 scripts/build_exe.py                # 构建到 dist/bigAagents/
    py -3 scripts/build_exe.py --noconsole    # 不带控制台窗口（更难排查，见下）
    py -3 scripts/build_exe.py --dry-run      # 只打印命令，不真的构建

## 几个必须解释的选择

**folder 而不是 onefile**：onefile 每次启动都要把 300MB+ 解压到临时目录
（冷启 10s 起、杀软反复扫描），且 `sys._MEIPASS` 每次都是新路径 —— 任何
"按 `__file__` 定位数据"的代码都会失效。folder 模式解压一次、路径稳定。

**控制台默认保留**：第一版最需要的是"出错时能看见"。`--noconsole` 一开，
任何 import 失败都会表现为"双击没反应"——所以 `launch_exe` 已经补了两道：
① 日志落 `程序目录/logs/launch.log`（滚动 2MB×2）；② 启动失败弹系统消息框。
`--noconsole` 还要配合 `uvicorn.Config(log_config=None)`，否则 uvicorn 自建
一套写 `sys.stdout` 的 handler，在无控制台进程里直接启动失败且**零输出**。

**面板数据不进包**：`data/panel/by_code` ≈387MB，打包进去会让体积和启动都崩。
首次运行检测缺失时走正常回填流程（`data/bars.py` 已有下载/回填路径）。

**⚠️ 密钥**：`config/llm_providers.json` 会被打进包（否则 exe 里没有任何厂商
配置）。**含密钥的 exe 不要分发**。要换成外部配置，运行时设：
    BIGA_LLM_PROVIDERS=<path>  BIGA_LLM_FEATURE_MODELS=<path>
（两个模块都读这两个环境变量，见 `config/providers.py::registry_path`）

**⚠️ 重新打包要先过"批量删除闸门"**：PyInstaller 带 `--noconfirm` 时会先删掉
`dist/bigAagents`（约 2300 个文件）。在带安全删除钩子的环境里，这一步会以
`[safe-delete][SAFE_DELETE_BULK_CONFIRM_REQUIRED]` 中止（exit=1，且**旧产物还在**，
看起来像"打包无声失败"）。两种解法：`CODEBUDDY_SAFE_DELETE_ENABLED=0`，或先手工删
`dist/bigAagents` 再构建。首次构建（dist 为空）不会遇到。
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = os.path.join(ROOT, "scripts", "launch_exe.py")
DIST = os.path.join(ROOT, "dist")
BUILD = os.path.join(ROOT, "build")

# 图标：没有它 exe 是 PyInstaller 默认图标，"桌面软件"的观感全无。
# 由 `assets/make_icon.py` 生成（深底 + 递增青柱 + 上升折线，与前端设计令牌一致）。
ICON = os.path.join(ROOT, "assets", "bigaagents.ico")

# uvicorn / anyio 大量使用"运行时按名字选实现"的导入，静态分析看不出来；
# 少了这些，exe 起来会报 "Could not import module 'uvicorn.protocols...'"
HIDDEN = [
    "uvicorn.logging",
    "uvicorn.loops", "uvicorn.loops.auto", "uvicorn.loops.asyncio",
    "uvicorn.protocols", "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto", "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan", "uvicorn.lifespan.on", "uvicorn.lifespan.off",
    "websockets",
    "anyio._backends._asyncio",
    "fastapi",
    "starlette",
    "pydantic",
    "web.server.app",
    "web.server.settings",
    "web.server.warmup",
    # 面板自调用起调度器（`launch_exe --scheduler`）。这条 import 在
    # `launch_exe._run_scheduler()` 里是**函数内**延迟导入，静态分析虽能扫到，
    # 但显式列出来更稳：漏了它 ⇒ `exe --scheduler` 直接 ModuleNotFoundError，
    # 而表现为"双击面板后调度器永远没起来"（无窗口、无控制台、极难发现）。
    "scripts.run_scheduler",
]

COLLECT_SUBMODULES = ["uvicorn", "websockets", "anyio", "web.server.routers"]

# 整个包一起收（子模块 + 数据 + 二进制）。
# ⚠️ `webview`（pywebview，原生窗口）的后端是**运行时按平台名字导入**的
#    （`webview.platforms.edgechromium` 等），静态分析只能看到 `import webview`。
#    Windows 走 EdgeChromium ⇒ 依赖 `pythonnet` 的 `clr`，PyInstaller 对 pythonnet
#    一向要求显式收集，否则运行时报 "Unable to load Python.Runtime"。
#    有兜底：`launch_exe._open()` 失败会退回默认浏览器，不会让程序起不来。
COLLECT_ALL = ["webview", "pythonnet"]

# 带**数据文件**的第三方包。否则运行时会以"文件不存在"炸在 import 阶段——
# 实测（2026-10-25 exe 冒烟）：`import akshare` 要读 `akshare/file_fold/calendar.json`，
# 缺它 ⇒ `import akshare` 直接 FileNotFoundError ⇒ `/api/intel/news` 500，
# 且 `data.trading_calendar` 静默退化成"当作非交易日"（**判据全错而不报错**）。
COLLECT_DATA = ["akshare"]

# 需要"连二进制一起、且必须保持在包内相对位置"的包。
# ⚠️ `py_mini_racer` 是按 `__file__` **同级**找 `mini_racer.dll` 的（还带 `icudtl.dat`）：
#    `--collect-binaries` 会把 dll 摊到 `_internal/` 根目录 ⇒ 照样找不到。
#    akshare 的交易日历走 JS 引擎，缺它 ⇒ `data.trading_calendar` **静默降级为
#    "周一至周五"**（判据全错却不报错）。实测：exe 日志 33 条 `[交易日历] akshare 获取失败`。
PACKAGE_DATA_VERBATIM = ["py_mini_racer"]

# 明显用不到、但会被间接拉进来的重包（构建体积/时间）
EXCLUDE = ["matplotlib", "tkinter", "PyQt5", "PySide6", "notebook", "IPython"]


def _data_args() -> list[str]:
    """把"只读资源"打进包：模板 / 静态 / 桌宠 / 配置 JSON。

    ⚠️ `--add-data` 的 `DEST` 对**文件**而言是"目标**目录**"，不是"目标文件名"——
    写成 `src;companion/embed.html` 会得到 `companion/embed.html/embed.html`
    （一个目录里塞同名文件），于是 StaticFiles 请求 `/companion/embed.html` 拿到
    的是目录 ⇒ 404，而日志里只有一句"目录不存在"都没有。**一律传目录**，让
    PyInstaller 自己递归，层级才不会错。
    """
    out = []
    for rel, dest in (("web/templates", "web/templates"),
                      ("web/static", "web/static"),
                      # 阿深桌宠：3D 模型 + three vendor + 渲染模块。助手界面的
                      # iframe 与 `/companion/` 整页都靠它；`settings.companion_dir()`
                      # 取 `bundle_root()`，打包后 = `_internal/companion`。
                      ("companion", "companion")):
        src = os.path.join(ROOT, rel)
        if not os.path.isdir(src):
            raise SystemExit(f"缺少资源目录：{src}")
        out += ["--add-data", f"{src}{os.pathsep}{dest}"]
        if rel == "companion":
            n = sum(len(fns) for _dp, _dn, fns in os.walk(src))
            print(f"[build] companion: {n} 个文件进包（3D 模型 + vendor）")

    cfg = os.path.join(ROOT, "config")
    if os.path.isdir(cfg):
        tmp = os.path.join(BUILD, "config_pack")
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        n = 0
        for name in sorted(os.listdir(cfg)):
            if name.endswith(".json"):
                shutil.copy2(os.path.join(cfg, name), os.path.join(tmp, name))
                n += 1
        print(f"[build] config: {n} 个 JSON 进包")
        out += ["--add-data", f"{tmp}{os.pathsep}config"]
    return out


def _verbatim_data_args() -> list[str]:
    """把包内非 .py 文件按**原相对路径**打进包（dll/dat/so 都在此列）。"""
    import importlib.util

    out = []
    for name in PACKAGE_DATA_VERBATIM:
        spec = importlib.util.find_spec(name)
        if not spec or not spec.submodule_search_locations:
            print(f"[build] ⚠️ 未安装 {name}：其数据文件跳过（相关功能会在运行时降级）",
                  file=sys.stderr)
            continue
        root = list(spec.submodule_search_locations)[0]
        n = 0
        for dp, _dn, fns in os.walk(root):
            for fn in fns:
                if fn.endswith((".py", ".pyc")):
                    continue
                rel = os.path.relpath(dp, root)
                dest = name if rel == "." else f"{name}/{rel.replace(os.sep, '/')}"
                out += ["--add-data", f"{os.path.join(dp, fn)}{os.pathsep}{dest}"]
                n += 1
        print(f"[build] {name}: {n} 个数据/二进制文件进包")
    return out


def build_args(name: str = "bigAagents", console: bool = True,
               workpath: str | None = None) -> list[str]:
    """拼 PyInstaller 参数。

    `workpath` 可覆盖（默认 `build/work`）：中间产物 `work/<name>/<name>.exe`
    可能被外部进程（杀软扫描、残留句柄）锁住，此时 PyInstaller 会在
    `os.remove` 处抛 `PermissionError: [WinError 5]` **而不是**在 COLLECT 阶段，
    报错信息只有一行路径，极易被误判成"权限不够"。换一个中间目录即可绕开，
    不必去动那个锁 —— 它跟最终产物无关。
    """
    args = [
        "--noconfirm", "--clean", "--onedir",
        "--name", name,
        "--distpath", DIST,
        "--workpath", workpath or os.path.join(BUILD, "work"),
        "--specpath", BUILD,
        "--paths", ROOT,
    ]
    args += ["--console" if console else "--windowed"]
    if os.path.isfile(ICON):
        args += ["--icon", ICON]
    else:
        logging.getLogger("build").warning("[build] 图标缺失（%s），将用默认图标", ICON)
    for h in HIDDEN:
        args += ["--hidden-import", h]
    for m in COLLECT_SUBMODULES:
        args += ["--collect-submodules", m]
    for m in COLLECT_DATA:
        args += ["--collect-data", m]
    for m in COLLECT_ALL:
        args += ["--collect-all", m]
    for m in EXCLUDE:
        args += ["--exclude-module", m]
    args += _verbatim_data_args()
    args += _data_args()
    args.append(ENTRY)
    return args


def _exe_locked(path: str) -> bool:
    """exe 是否正被运行/占用。

    ⚠️⚠️ 这是 2026-10-25 一次真实事故的守卫：PyInstaller 的 COLLECT 阶段会先
    `shutil.rmtree(dist/<name>)`，若里面的 exe 正跑着 ⇒ 删目录时抛
    `PermissionError: [WinError 5]`，而 **rmtree 已经把其余文件删光了**
    ⇒ 留下一个"只有 exe、没有 `_internal/`"的残缺 onedir 目录。
    这个残缺目录不会被任何检查发现（`os.path.exists(exe)` 是 True），
    双击时 bootloader 找不到自己的 `_internal`，只弹一个标题为 "Error" 的系统框、
    **一行 Python 都不执行、日志为空** —— 极难定位（会误判成"打包后启动失败"）。
    所以：构建前必须确认旧 exe 没在跑，别让 rmtree 半途而废。
    """
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r+b"):          # 正在执行的 PE 映像不允许以写方式打开
            return False
    except OSError as e:
        # 打不开 ⇒ 被占用（也正是我们要拦的情况）。留一条日志，别让它变成"静默吞错"。
        logging.getLogger("build").warning("[build] %s 打不开，判定为正在运行：%s", path, e)
        return True


def _verify_artifact(exe: str) -> tuple[bool, str]:
    """校验产物是**完整的** onedir 目录，而不只是"exe 文件存在"。

    判据按内容，不按存在性：onedir 必须有 `_internal/`（PyInstaller 6.x 的布局），
    且文件数要上千级（实测 337MB / 约 2300 个文件）。只查 `os.path.exists(exe)`
    会把上面那种残缺目录当成成功 —— 这次就是被它骗了。
    """
    d = os.path.dirname(exe)
    internal = os.path.join(d, "_internal")
    if not os.path.isfile(exe):
        return False, f"没找到 {exe}"
    if not os.path.isdir(internal):
        return False, f"缺少 {internal}（onedir 的运行时都在这里，缺了双击只会弹 Error）"
    n = 0
    size = 0
    for dp, _dn, fs in os.walk(d):
        for f in fs:
            n += 1
            try:
                size += os.path.getsize(os.path.join(dp, f))
            except OSError as e:
                # 体积只是附加信息，单个文件读不到不该让校验失败 —— 但**不能静默**
                # （hygiene 守卫按"except 块里既无日志也不 raise"判吞错）。
                logging.getLogger("build").warning("[build] 体积统计跳过 %s: %s", f, e)
    if n < 500:
        return False, f"产物只有 {n} 个文件（正常应为 2000+），疑似 COLLECT 未完成"
    return True, f"{n} 个文件 / {size / 1024 ** 2:.0f} MB"


def _sync_env_beside_exe(dist_dir: str) -> None:
    """把项目根 `.env` 复制到 exe 同级目录 —— **每次重打包都会把它删掉**。

    为什么必须做成自动的：PyInstaller 带 `--noconfirm` 时会先执行
    `Removing dir dist/bigAagents`，连用户手工放在那里的 `.env` 一并删除。
    而 `config/settings.py` 在 frozen 态只从 `Path(sys.executable).parent/.env`
    兜底读一次（`__file__` 打包后在只读的 `_internal/` 里）⇒ 少了它，
    `Config.validate()` 导入即抛，表现为"助手/情报端点 500"，
    **界面上完全看不出是缺配置**。上一版只打印提示 ⇒ 重打包就得靠人记得。

    ⚠️ 复制过去的 `.env` 含密钥 ⇒ **整个 dist 目录不要分发**。
    """
    src = os.path.join(ROOT, ".env")
    if not os.path.exists(src):
        print("[build] 提示：项目根无 .env，跳过（LLM 相关功能会报\"缺配置\"，其余页面正常）")
        return
    dst = os.path.join(dist_dir, ".env")
    try:
        shutil.copyfile(src, dst)
        with open(src, encoding="utf-8", errors="replace") as f:
            n = sum(1 for line in f
                    if line.strip() and not line.lstrip().startswith("#") and "=" in line)
        print(f"[build] 已把 .env 放到 exe 同级（{n} 个配置项，含密钥，勿分发）")
    except Exception as e:                                   # noqa: BLE001
        logging.warning("[build] .env 复制失败（助手/情报端点会 500）: %s: %s",
                        type(e).__name__, e)
        print(f"[build] ⚠ .env 复制失败（助手/情报端点会 500）：{type(e).__name__}: {e}",
              file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 bigAagents exe（folder 模式）")
    ap.add_argument("--name", default="bigAagents")
    ap.add_argument("--noconsole", action="store_true", help="不显示控制台窗口")
    ap.add_argument("--workpath", default="", help="构建中间目录（默认 build/work）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令")
    a = ap.parse_args()

    workpath = a.workpath or os.path.join(BUILD, "work")
    old_exe = os.path.join(DIST, a.name, a.name + ".exe")
    if _exe_locked(old_exe):
        print(f"[build] ✗ 旧产物正在运行，必须先关掉：{old_exe}\n"
              f"        PyInstaller 要清空 {os.path.dirname(old_exe)}，"
              f"删不掉 exe 会把目录掏成残缺状态。", file=sys.stderr)
        return 3

    # ⚠️ 中间产物同样会拦构建，且报错点在 os.remove（不是 COLLECT），
    #    与上面那个"残缺目录"事故是**两种不同的失败**，别混为一谈：
    #    这里失败时 dist/ 还是完好的，只需换 --workpath 重跑。
    inter = os.path.join(workpath, a.name, a.name + ".exe")
    if _exe_locked(inter):
        alt = os.path.join(BUILD, "work_" + datetime.now().strftime("%Y%m%d_%H%M"))
        print(f"[build] ✗ 中间产物被占用：{inter}\n"
              f"        它跟最终产物无关，换一个中间目录即可绕开：\n"
              f"        python scripts/build_exe.py --workpath \"{alt}\"", file=sys.stderr)
        return 4

    args = build_args(a.name, console=not a.noconsole, workpath=workpath)
    print("[build] PyInstaller " + " ".join(args))
    if a.dry_run:
        return 0

    try:
        import PyInstaller.__main__ as P
    except ImportError:
        print("[build] 未安装 PyInstaller：py -3 -m pip install pyinstaller", file=sys.stderr)
        logging.warning("[build] 缺少 PyInstaller，无法打包（pip install pyinstaller）")
        return 2

    P.run(args)
    exe = os.path.join(DIST, a.name, a.name + ".exe")
    ok, detail = _verify_artifact(exe)
    if ok:
        print(f"[build] 完成：{exe}（{detail}）")
        print("[build] 提示：把 data/ 放到 exe 同级目录（面板数据不进包，首次运行会自动下载）")
        _sync_env_beside_exe(os.path.join(DIST, a.name))
        return 0
    print(f"[build] ✗ 产物校验失败：{detail}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
