# -*- coding: utf-8 -*-
"""桌面启动器（P6）—— exe 的入口。

做的事，按顺序：
1. **单实例**：OS 命名互斥（`core.single_instance`）。第二次双击时不再起一个
   uvicorn，而是把已有窗口/浏览器调到前台（端口从 `temp/web_port.txt` 读）。
2. **随机端口**：`bind(0)` 拿一个空闲端口 —— 8765 被别的程序占着时不该让用户
   去翻日志。
3. **同进程跑 uvicorn**（线程），**不开子进程**。
4. 开窗口：装了 `pywebview` 就用原生窗口，否则退回默认浏览器。
5. 退出清理：`.should_exit = True` → 等线程收尾 → 超时就 `os._exit`。

## 为什么不开子进程（偏离文档 §8 的原方案）
文档当初写的是"内嵌 uvicorn 子进程 + 窗口关闭杀子进程"。实测这条路要处理
Windows 上进程组、僵尸端口、任务管理器里残留的 python.exe —— 每一项都能让
"关了窗口还在后台跑"。uvicorn 在**非主线程**里不会安装信号处理器（它自己判断
`current_thread() is main_thread()`），所以线程方案是安全的，且进程退出即端口释放。
"""
from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 打包后 sys.path 已由 PyInstaller 设好；源码运行时上面这行保证能 import web/core
os.environ.setdefault("BIGA_WARMUP", "1")

_logger = logging.getLogger("launch")
# ⚠️ 端口文件路径**不在这里写死**：它必须跟 `settings.data_root()` 走
#    （打包后数据在 exe 同级，不在只读包内），见 `_port_file()`。
MUTEX_APP = "web"

# ============================================================ 调度器（2026-10-26）
# 双击面板时**顺带**把定时调度器拉起来。此前 `launch_exe` 只管 web：用户必须另外
# 双击 `启动bigAagents.bat` 才有选股/复盘/新闻任务，而那个 bat 走的是**旧版
# Streamlit（8501）**，与新面板是两条互不相干的路径 —— 实际结果是"面板能开、
# 调度器没起"，health 页显示未运行，且没人会想到还要再双击一次。
#
# ⚠️⚠️ **打包后 `sys.executable` 是 `bigaagents.exe` 自己，不是 `python.exe`**：
#    `Popen([sys.executable, "scripts/run_scheduler.py"])` 在冻结包里等于"用 exe
#    去打开一个 .py 文件" —— noconsole 下**静默失败**，一行日志都没有。
#    所以 frozen 走 `exe --scheduler` **自调用**（同包再起一个进程），
#    只有源码模式才用 python 直接跑脚本。见 `_scheduler_cmd()`。
SCHED_ARG = "--scheduler"


def _app_dir() -> str:
    """程序所在目录：打包后 = exe 同级，源码运行 = 项目根。

    ⚠️ 不能用 `os.getcwd()`：快捷方式/资源管理器的"起始位置"不可控，
       从别处启动就找错地方。日志必须落在确定的目录。
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return _ROOT


def _log_dir() -> str:
    """可写日志目录。**优先 `data_root()/logs`**，再退程序同级 `logs/`，最后临时目录。

    ⚠️ 2026-10-26：冻结后 `_app_dir()` 是 `dist/bigaagents/`，而业务日志（调度器/各任务的）
    全部写在 `data_root()/logs/`。两边不一致时同一天会出现**两个 logs 目录**，
    排查时"A 里没有就以为没记录"。我今天找 `launch.log` 就找错过一次，
    实际它在 `dist/bigaagents/logs/`，而 `logs/` 里只有调度器的。
    ⇒ 只要能写，就跟业务日志放一起。
    """
    candidates: list[str] = []
    try:
        from config.paths import data_root
        root = data_root()
        if root:
            candidates.append(os.path.join(root, "logs"))
    except Exception as e:                                   # noqa: BLE001
        # 启动极早期可能还 import 不了，属正常 ⇒ 只留 debug 痕迹再退回程序目录
        _logger.debug("[launch] data_root 不可用（%s: %s），日志退回程序目录",
                      type(e).__name__, e)
    candidates.append(os.path.join(_app_dir(), "logs"))

    for preferred in candidates:
        try:
            os.makedirs(preferred, exist_ok=True)
            probe = os.path.join(preferred, ".writable")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("1")
            os.remove(probe)
            return preferred
        except OSError as e:
            _logger.debug("[launch] 日志目录不可写: %s（%s）", preferred, e)

    import tempfile
    fallback = os.path.join(tempfile.gettempdir(), "bigAagents_logs")
    os.makedirs(fallback, exist_ok=True)
    # ⚠️ 不能用 `print(..., file=sys.stderr)`：`--noconsole` 下 sys.stderr 是 None，
    #    print 会把 None 当"默认输出"再去找 sys.stdout（也是 None）⇒ 二次崩溃。
    #    logging 的 lastResort handler 对 None 有保护，是这里唯一安全的出口。
    _logger.warning("[launch] 所有候选日志目录都不可写，改用 %s", fallback)
    return fallback


def _setup_logging() -> str:
    """装 handler 并返回日志文件路径（供启动横幅/弹窗引用）。

    ⚠️⚠️ **`--noconsole` 下 `sys.stdout` / `sys.stderr` 都是 `None`**：
       ① `StreamHandler(None)` 一 emit 就抛 `AttributeError`；
       ② 更要命的是 uvicorn 默认 `log_config` 会自己 `dictConfig` 一套指向
          `sys.stdout` 的 handler —— 在无控制台进程里直接启动失败，且**没有任何输出**
          （这就是"双击没反应"的典型成因）。所以启动 uvicorn 时显式传
          `log_config=None`，让它复用这里的 root logger（见 `main()`）。
       ③ 文件日志是**唯一**的事后排障手段，必须装；滚动 2MB×2 防止页面轮询刷爆磁盘。
    """
    handlers: list[logging.Handler] = []
    if sys.stdout is not None:                      # 有控制台才加（noconsole 下为 None）
        handlers.append(logging.StreamHandler(sys.stdout))

    log_path = os.path.join(_log_dir(), "launch.log")
    try:
        from logging.handlers import RotatingFileHandler
        handlers.append(RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024,
                                            backupCount=2, encoding="utf-8"))
    except OSError as e:                            # 磁盘满/权限
        # 同上：noconsole 下没有 stderr，只能靠 logging（lastResort 有 None 保护）
        _logger.warning("[launch] 文件日志不可用：%s", e)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers or [logging.NullHandler()],
        force=True,                                 # 重复调用（测试）不叠加 handler
    )
    return log_path


def _fatal(title: str, text: str) -> None:
    """无控制台时的最后一道可读反馈：弹系统消息框。

    有控制台就只打日志（不打断批处理/测试）；打包且无控制台才弹。
    弹窗失败绝不二次抛异常 —— 那会把"启动失败"变成"启动失败且无提示"。
    """
    _logger.error("[launch] %s: %s", title, text)
    if sys.stdout is not None:
        return
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text, f"bigAagents · {title}", 0x10)
    except Exception as e:                          # noqa: BLE001
        _logger.debug("[launch] 消息框不可用: %s: %s", type(e).__name__, e)


def pick_port() -> int:
    """让 OS 分配一个空闲端口（绑 0 再取实际端口）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _check_config() -> None:
    """启动就报"配置缺了什么"，而不是等用户点进助手页吃一个 500。

    `config/settings.py` 在"没有任何厂商 Key"时**导入即抛**；届时
    `/api/chat/*`、`/api/intel/news` 会 500。这里只是**告警**，不阻断启动：
    看盘/复盘/交易台/研究都不需要 LLM，先能看数据比整体起不来强。
    """
    try:
        import config.settings  # noqa: F401
    except Exception as e:                     # noqa: BLE001
        _logger.warning(
            "[launch] LLM 配置不可用：%s\n"
            "         把 .env 放到 exe 同级目录（%s），或配置 config/llm_providers.json。\n"
            "         在此之前：看盘/复盘/交易台/研究可用；助手与深度分析会报\"缺配置\"。",
            e, _app_dir())


def _port_file() -> str:
    from web.server import settings
    return os.path.join(settings.data_root(), "temp", "web_port.txt")


def _write_port(port: int) -> None:
    p = _port_file()
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(str(port))
    except OSError as e:
        _logger.warning("[launch] 端口文件写入失败（第二次双击将无法定位已有实例）: %s", e)


def _read_port() -> int | None:
    try:
        with open(_port_file(), encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError) as e:
        _logger.debug("[launch] 读不到端口文件（首次启动是正常的）: %s: %s",
                      type(e).__name__, e)
        return None


def _open(url: str) -> None:
    """优先 pywebview 原生窗口；缺失/失败则退回默认浏览器（不阻断启动）。"""
    try:
        import webview                                     # noqa: F401
    except Exception:                                      # noqa: BLE001
        _logger.info("[launch] 无 pywebview，使用默认浏览器")
        webbrowser.open(url)
        return
    try:
        import webview
        webview.create_window("bigAagents", url, width=1440, height=900)
        webview.start()
    except Exception as e:                                 # noqa: BLE001
        _logger.warning("[launch] pywebview 启动失败，退回浏览器: %s: %s", type(e).__name__, e)
        webbrowser.open(url)


def _data_root() -> str:
    """数据根（唯一入口 `config.paths.data_root()`，不要在本地重算）。

    ⚠️ 不能用 `_app_dir()`：打包后程序目录是 `dist/bigAagents/`，而数据在
       项目根（分发布局里两者相同，开发布局里不同）。
    """
    try:
        from config.paths import data_root
        return data_root()
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[launch] 数据根解析失败，回退程序目录: %s: %s",
                        type(e).__name__, e)
        return _app_dir()


def _is_scheduler_mode(argv: list[str] | None = None) -> bool:
    """本进程是否是"被自调用起来的调度器"。"""
    args = sys.argv[1:] if argv is None else argv
    return SCHED_ARG in args


def _ensure_streams() -> None:
    """`--noconsole` 下 `sys.stdout` / `sys.stderr` 都是 `None`，必须补成真文件。

    ⚠️ 这不是可选优化：`scripts/run_scheduler.setup_logging()` 会
       `StreamHandler(sys.stdout)` + `FileHandler`，handler 第一次 emit 时
       对 `None` 调 `write()` ⇒ `AttributeError` ⇒ 调度器起不来。
       web 分支绕过了这个问题（`log_config=None` 复用 root logger），
       调度器分支是**另一套 logging**，绕不过去，只能先把流补好。
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    path = os.path.join(_log_dir(), "scheduler_stdout.log")
    try:
        f = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
    except OSError as e:
        import io
        _logger.warning("[launch] 调度器输出日志不可写，丢弃输出: %s", e)
        f = io.StringIO()                       # 兜底：丢掉，但绝不崩
    if sys.stdout is None:
        sys.stdout = f
    if sys.stderr is None:
        sys.stderr = f


def _scheduler_alive() -> tuple[bool, dict]:
    """调度器是否在跑。返回 (alive, heartbeat)。

    ⚠️ 判据是 **`started_at` + PID**（`_process_alive`），**不是心跳 mtime**：
       心跳陈旧但进程在 ⇒ 只告警不杀（项目铁律）。这里用于"要不要再起一个"。
    """
    try:
        from core.scheduler_health import _process_alive, read_heartbeat
    except Exception as e:                     # noqa: BLE001
        _logger.debug("[launch] 心跳模块不可用: %s: %s", type(e).__name__, e)
        return False, {}
    hb = read_heartbeat()
    pid = hb.get("pid") if hb else None
    return bool(pid and _process_alive(pid)), hb


def _scheduler_cmd() -> tuple[list[str], str]:
    """(命令行, 工作目录)。frozen 与源码两条完全不同的路径，见 `SCHED_ARG` 注释。

    工作目录必须是**数据根**：`run_scheduler.setup_logging()` 用相对路径
    `os.makedirs("logs")`，cwd 不对会把日志写到别处（甚至因权限失败）。
    """
    root = _data_root()
    if getattr(sys, "frozen", False):
        return [sys.executable, SCHED_ARG], root
    script = os.path.join(_ROOT, "scripts", "run_scheduler.py")
    return [sys.executable, script], root


def _spawn_scheduler(probe_delay: float = 1.5) -> bool:
    """起一个**独立**的调度器进程；已在跑则什么都不做。返回是否新拉起。

    设计取舍：
    - **独立进程**（`DETACHED_PROCESS`）：关掉面板窗口不该停掉选股/复盘/新闻，
      调度是常驻职责，界面只是观察窗口。停止走 `停止bigAagents.bat`。
    - **双开安全**：`run_scheduler` 自带 `Global\\` OS 命名互斥，即使这里判断
      失误多起一个，第二个会自己 exit(1)，不会重复跑选股。
    - **失败不阻断面板**：调度器起不来时面板必须照样能用（看盘/复盘/交易台
      都不依赖它），只记日志。
    """
    if os.environ.get("BIGA_NO_SCHEDULER"):
        _logger.info("[launch] BIGA_NO_SCHEDULER=%s，跳过调度器",
                     os.environ.get("BIGA_NO_SCHEDULER"))
        return False

    alive, hb = _scheduler_alive()
    if alive:
        _logger.info("[launch] 调度器已在运行（pid=%s，心跳 %s），不重复启动",
                     hb.get("pid"), hb.get("ts"))
        return False

    cmd, cwd = _scheduler_cmd()
    log_path = os.path.join(_log_dir(), "scheduler_stdout.log")
    kwargs: dict = {
        "cwd": cwd,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        # DETACHED_PROCESS：不继承控制台，父进程退出后继续活；
        # CREATE_NEW_PROCESS_GROUP：Ctrl+C/CTRL_CLOSE 事件不波及它。
        kwargs["creationflags"] = (subprocess.DETACHED_PROCESS
                                   | subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8", errors="replace") as out:
            proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                    **kwargs)
    except Exception as e:                     # noqa: BLE001
        # 这里绝不能抛：面板必须能起。真实原因写日志 + 页面上 health 会显示未运行。
        _logger.warning("[launch] 调度器启动失败（面板不受影响）: %s: %s\n"
                        "         命令=%s  cwd=%s", type(e).__name__, e, cmd, cwd)
        return False

    _logger.info("[launch] 调度器已拉起 pid=%s cwd=%s 输出=%s", proc.pid, cwd, log_path)
    # 异步探测"启动后秒退"（缺依赖/互斥冲突都会这样），**不能同步 sleep**：
    # 那会把窗口打开硬生生推迟 1.5s，用户会觉得双击很慢。
    def _probe() -> None:
        try:
            time.sleep(probe_delay)
            rc = proc.poll()
            if rc is not None:
                _logger.warning("[launch] 调度器启动后立刻退出（code=%s），详见 %s",
                                rc, log_path)
        except Exception as e:                 # noqa: BLE001
            _logger.debug("[launch] 调度器探测失败: %s: %s", type(e).__name__, e)

    threading.Thread(target=_probe, name="sched-probe", daemon=True).start()
    return True


def _run_scheduler() -> int:
    """`--scheduler` 分支：把本进程变成调度器。

    ⚠️ 顺序要求：先补流（否则 run_scheduler 配日志时崩）→ 换 cwd（相对路径）
       → 裁掉自有参数（否则它的 argparse 报"unrecognized arguments"）。
    """
    argv = [a for a in sys.argv[1:] if a != SCHED_ARG]
    sys.argv = [sys.argv[0]] + argv

    _ensure_streams()
    try:
        root = _data_root()
        os.chdir(root)
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[launch] 切换工作目录失败，沿用当前目录: %s: %s",
                        type(e).__name__, e)

    _logger.info("[launch] 进入调度器模式 pid=%s cwd=%s", os.getpid(), os.getcwd())
    from scripts.run_scheduler import main as scheduler_main
    return int(scheduler_main() or 0)


def _run() -> int:
    _check_config()
    from core import single_instance

    name = single_instance.mutex_name(MUTEX_APP)
    if not single_instance.acquire_os_mutex(name):
        port = _read_port()
        if port:
            url = f"http://127.0.0.1:{port}/"
            _logger.info("[launch] 已有实例在跑，打开 %s", url)
            # 与主分支同一套判断：自动化验证时不该弹浏览器
            if os.environ.get("BIGA_NO_BROWSER"):
                _logger.info("[launch] BIGA_NO_BROWSER=1，跳过打开窗口")
            else:
                _open(url)
        else:
            _logger.warning("[launch] 已有实例在跑，但读不到端口文件：%s", _port_file())
        return 0

    port = pick_port()
    _write_port(port)
    url = f"http://127.0.0.1:{port}/"

    import uvicorn
    from web.server.app import app

    # ⚠️ `log_config=None` 是 `--noconsole` 的**必须项**：uvicorn 默认会
    #    `logging.config.dictConfig()` 一套写 `sys.stdout` 的 handler，
    #    而 noconsole 下 `sys.stdout is None` ⇒ 启动即失败且无任何输出。
    #    `access_log=False`：页面每 10s 轮询几个端点，access 日志会淹没真正的错误。
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info",
                            log_config=None, access_log=False)
    server = uvicorn.Server(config)

    # 非主线程 ⇒ uvicorn 不接管信号处理器，Ctrl+C 仍由 main() 处理
    th = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    th.start()

    deadline = time.time() + 30
    while time.time() < deadline and not getattr(server, "started", False):
        if not th.is_alive():
            _logger.error("[launch] uvicorn 线程提前退出，无法启动")
            return 1
        time.sleep(0.05)
    if not getattr(server, "started", False):
        _logger.error("[launch] 30s 内未就绪，放弃")
        return 1

    _logger.info("[launch] 服务已就绪 %s", url)
    # 顺带拉起定时调度器。放在"服务已就绪"之后而不是最前面：
    # 面板先可用优先；调度器晚十几秒没有影响（它按 cron 跑，不是即时任务）。
    _spawn_scheduler()
    # `BIGA_NO_BROWSER=1`：只起服务不开窗口（自动化冒烟测试用 —— 否则每次验证
    # 都会在桌面上弹一个浏览器，还没法判断"到底是服务没起来还是窗口没开"）
    if os.environ.get("BIGA_NO_BROWSER"):
        _logger.info("[launch] BIGA_NO_BROWSER=1，跳过打开窗口（服务继续运行）")
    else:
        try:
            _open(url)
        except KeyboardInterrupt:
            _logger.info("[launch] 打开窗口时收到中断")

    # 窗口/浏览器关掉后仍要有人守着服务：Ctrl+C 或进程被杀才退出
    try:
        while th.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        _logger.info("[launch] 收到中断，正在关闭")

    server.should_exit = True
    th.join(timeout=5)
    if th.is_alive():
        _logger.warning("[launch] 线程未在 5s 内退出，强制结束进程")
        os._exit(0)                                        # 端口随进程释放，不需要 kill 子进程
    single_instance.release_os_mutex(name)
    _logger.info("[launch] 已退出")
    return 0


def main() -> int:
    """入口：先装日志（后面所有分支都要能留痕），再跑、失败必给人看得见的反馈。"""
    # ⚠️ `--scheduler` 必须**最先**判：这条分支不能装 web 的日志 handler
    #    （会和 run_scheduler 自己那套叠加，日志重复两遍）、不能占 web 互斥、
    #    更不能起 uvicorn 或开窗口。
    if _is_scheduler_mode():
        try:
            return _run_scheduler()
        except Exception as e:                 # noqa: BLE001
            _ensure_streams()
            # 不弹窗：这是被自调用起的后台进程，弹窗只会让用户莫名其妙点确定。
            _logger.exception("[launch] 调度器模式启动失败")
            try:
                print(f"[launch] 调度器启动失败: {type(e).__name__}: {e}",
                      file=sys.stderr)
            except Exception:                  # noqa: BLE001
                # --noconsole 下 sys.stderr 是 None，print 必然失败。
                # 这是**预期状态**而非故障，但不能写成 `pass`：
                # 那只会在真正的 stderr 故障时也一声不响（守卫基线 34→35 抓到的就是这处）。
                _logger.debug("[launch] stderr 不可用（--noconsole），跳过控制台输出")
            return 1

    log_path = _setup_logging()
    _logger.info("[launch] 启动 pid=%s 程序目录=%s", os.getpid(), _app_dir())
    _logger.info("[launch] 日志文件 %s", log_path)
    try:
        rc = _run()
    except KeyboardInterrupt:
        _logger.info("[launch] 收到中断，退出")
        return 0
    except Exception as e:                     # noqa: BLE001
        # 有 traceback 才是能给开发者看的东西；`_fatal` 只负责"给人一句话 + 弹窗"
        _logger.exception("[launch] 未捕获异常，启动失败")
        _fatal("启动失败", f"{type(e).__name__}: {e}\n\n详细信息见日志：\n{log_path}")
        return 1
    if rc != 0:
        _fatal("启动失败", f"退出码 {rc}\n\n详细信息见日志：\n{log_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
