# -*- coding: utf-8 -*-
"""路径解析的**唯一来源**（无副作用、无校验）。

## 为什么单独一个模块

- `web/server/settings.py` 要用它（模板/静态/数据根/端口文件）
- `db/models.py` 要用它（`DB_PATH`）
- 但 `db/` **不能** import `config.settings`：那是带 `Config.validate()` 的配置门面，
  缺 LLM Key 时**导入即抛**，数据库层不该被"模型密钥有没有配"绑架。

⇒ 把纯路径计算抽到这里。两边都只依赖 `os`/`sys`，无循环、无副作用。

## 为什么全是函数

项目铁律：模块级常量在**定义时**求值，之后 `monkeypatch.setenv` / 运行时改环境变量
都不再影响它。路径一旦写成常量，测试就只能去碰真实目录。

⚠️ 唯一例外是 `db.models.DB_PATH`：它以 `from db.models import DB_PATH` 的形式被
多处引用，必须保持"模块属性"的形态。所以它在模块导入时由 `db_path()` 求值一次，
测试仍可用 `monkeypatch.setattr` 覆盖（`tests/conftest.py::use_template_db` 就是这么做的）。

## ⚠️⚠️ 为什么 `data_root()` 要向上探测祖先目录

2026-10-25 用户实测事故：双击 `dist/bigAagents/bigAagents.exe` 后**只有仪表盘能用**。
原因是 `app_root()` 指向 `dist/bigAagents/`，那里没有 `data/`、`output/`：

| 端点 | 结果 |
|---|---|
| `/api/picks/latest`、`/api/review/latest`、`/api/research/backtest` | `missing` |
| `/api/analysis/history` | 空列表 |
| `/api/desk/portfolio` | **新开的空账户**（100 万现金、0 持仓，看着像持仓丢了） |
| `/api/system/health` | 看不到调度器 |

而看盘/新闻/仪表盘的指数区是**联网**取的 ⇒ 唯独仪表盘"看起来能用"，
极易被误判成"只有仪表盘做好了"。所以开发布局（exe 在 `<项目根>/dist/<name>/`）
必须能自动找到项目根的数据，不能要求用户先手工写一个配置文件。
"""
from __future__ import annotations

import os
import sys


def is_frozen() -> bool:
    """是否跑在 PyInstaller 打出来的包里。"""
    return bool(getattr(sys, "frozen", False))


def project_root() -> str:
    """本文件所在项目根（源码运行用；打包后指向只读包内 `_internal/`）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bundle_root() -> str:
    """**只读**资源根（模板/静态/字体）。

    源码运行 = 项目根；打包后 = `sys._MEIPASS`（folder 模式下即 `_internal/`）。
    """
    if is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    return project_root()


def app_root() -> str:
    """**可执行文件**所在目录（不是包内目录）。"""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return project_root()


def _has_data_layout(path: str) -> bool:
    """该目录自带数据布局吗（分发布局：数据就放在 exe 旁边）。

    判据取"宽"：有 `data/` 或 `output/` 即可 —— 全新安装时两个目录都可能还是空的，
    这时**必须**认它是数据根（首次运行会在里面建库），否则永远探测不到自己。
    """
    try:
        return (os.path.isdir(os.path.join(path, "data"))
                or os.path.isdir(os.path.join(path, "output")))
    except OSError:
        return False


def _looks_like_project(path: str) -> bool:
    """祖先目录的判据取"严"：必须是**真数据布局**。

    `data/`、`output/` 这种名字在任何机器上都很常见，向上探测若也按宽判据，
    可能命中无关目录（比如桌面上的另一个项目）⇒ 读到别人的数据比读不到更糟。
    """
    try:
        return (os.path.isdir(os.path.join(path, "output", "daily_picks"))
                or os.path.isdir(os.path.join(path, "data", "panel")))
    except OSError:
        return False


def data_root() -> str:
    """可写数据根。优先级：

    1. `BIGA_DATA_ROOT` —— 显式指定（env 或 exe 同级 `.env`），永远最高
    2. 程序目录**自身**带数据布局 ⇒ 分发布局，数据就在 exe 旁边
    3. 程序目录的**祖先**里第一个真项目根 ⇒ 开发布局（`<根>/dist/<name>/`）
    4. 程序目录（兜底：可写但空，首次运行在里面建库）

    ⚠️ 绝不能只用 `app_root()`：见模块头那段事故说明。
    """
    env = os.environ.get("BIGA_DATA_ROOT")
    if env:
        return env

    base = app_root()
    if _has_data_layout(base):
        return base

    cur = base
    for _ in range(4):                      # dist/<name> → dist → 项目根
        parent = os.path.dirname(cur)
        if not parent or parent == cur:     # 到盘根
            break
        if _looks_like_project(parent):
            return parent
        cur = parent
    return base


def db_path() -> str:
    """模拟交易库路径。优先级：`BIGA_DB_PATH` > `<data_root>/data/portfolio.db`。

    ⚠️⚠️ 这是**第二个**导致"只有仪表盘能用"的坑，且比 `data_root()` 更隐蔽：
       原实现是 `DB_PATH = str(Path(__file__).parent.parent / "data" / "portfolio.db")`，
       打包后 `__file__` 在 `_internal/` ⇒ 指向 `_internal/data/portfolio.db`（空库）。
       即使 `data_root()` 修好了，交易台仍然是**新开的空账户**。
    """
    env = os.environ.get("BIGA_DB_PATH")
    if env:
        return env
    return os.path.join(data_root(), "data", "portfolio.db")
