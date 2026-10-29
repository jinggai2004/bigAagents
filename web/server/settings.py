# -*- coding: utf-8 -*-
"""服务层路径与常量。

⚠️ **路径一律运行时解析**（函数而非模块常量）：项目铁律——默认参数/模块常量在
定义时求值，`monkeypatch` 会失效，测试就只能去碰真实目录。这里所有路径都做成
函数，测试用 `BIGA_DATA_ROOT` 环境变量指向 `tmp_path` 即可，不需要改源码。
"""
from __future__ import annotations

import os
from datetime import datetime

from config import paths as _paths

# 源码项目根。打包后它指向只读包内 —— **只用于"这是哪个项目的源码"这类展示/兼容**，
# 绝不能再拿它拼数据路径（要拼就用下面的 `data_root()`）。
PROJECT_ROOT = _paths.project_root()

SERVER_VERSION = "0.5.0-p5"
STARTED_AT = datetime.now().isoformat(timespec="seconds")

# 缓存 TTL（秒）—— 与文档 §4.2 一致
TTL_PICKS = 60
TTL_HEALTH = 10


# ================================================================
# 打包（P6）：一套代码既要能源码跑，也要能 exe 跑
# ================================================================
# 路径计算全部委托 `config/paths.py`（唯一来源）：`db/models.py` 的 `DB_PATH`
# 也要用同一套规则，放在这里会让 db 层反向依赖 web 层（或各写一套迟早漂移）。
# （`_paths` 已在文件顶部导入）


def is_frozen() -> bool:
    """是否跑在 PyInstaller 打出来的包里。"""
    return _paths.is_frozen()


def bundle_root() -> str:
    """**只读**资源根（模板/静态/字体）。

    源码运行 = 项目根；打包后 = `sys._MEIPASS`（folder 模式下就是 `_internal/`）。
    """
    return _paths.bundle_root()


def app_root() -> str:
    """可执行文件所在目录（打包后数据就落在它旁边，不落进只读包内）。"""
    return _paths.app_root()


def data_root() -> str:
    """可写数据根。优先级：`BIGA_DATA_ROOT` > 程序目录若自带数据布局 >
    祖先里第一个真项目根（开发布局：exe 在 `<根>/dist/<name>/`）> 程序目录。

    ⚠️ 打包后**不能**用 `bundle_root()`：`sys._MEIPASS` 是解压出来的临时树，
       程序退出即销毁（onefile）或位于 `_internal/`（folder，通常不建议写）。
       数据写进去 = 第二天打开发现库空了。

    ⚠️ 也不能只用 `app_root()`：见 `config/paths.py` 模块头记录的事故
       （双击 exe 后只有仪表盘能用）。规则与判据全在 `paths.data_root()`。
    """
    return _paths.data_root()


def picks_path() -> str:
    return os.path.join(data_root(), "output", "daily_picks", "latest.json")


def heartbeat_path() -> str:
    return os.path.join(data_root(), "temp", "scheduler_heartbeat.json")


def templates_dir() -> str:
    return os.path.join(bundle_root(), "web", "templates")


def static_dir() -> str:
    """本地字体与共享 JS（A1：零外网依赖）。"""
    return os.path.join(bundle_root(), "web", "static")


def companion_dir() -> str:
    """阿深的 3D 桌宠（独立页面，自带 three.js vendor，无构建步骤）。

    挂在 `/companion/` 而不是塞进 templates：它有自己的 importmap 与 ES module
    入口，和主站模板不是一类东西；分开挂也让它能单独打开调试。
    ⚠️ 必须在 `/`（templates 的兜底 StaticFiles）**之前**挂载，否则请求会被吞掉。
    """
    return os.path.join(bundle_root(), "companion")


def template_version() -> str:
    """模板版本 = 文件名 + mtime 的短哈希（改模板 ⇒ 版本变，前端可据此清缓存）。"""
    import hashlib
    d = templates_dir()
    try:
        names = sorted(f for f in os.listdir(d) if f.endswith(".html"))
    except OSError as e:
        import logging
        logging.getLogger(__name__).debug("[meta] 模板目录不可读: %s", e)
        return "unknown"
    h = hashlib.sha1()
    for n in names:
        try:
            h.update(f"{n}:{int(os.path.getmtime(os.path.join(d, n)))}".encode())
        except OSError as e:
            import logging
            logging.getLogger(__name__).debug("[meta] 跳过一个模板: %s", e)
    return f"{len(names)}p-{h.hexdigest()[:8]}"
