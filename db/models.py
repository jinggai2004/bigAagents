"""数据库模型定义（SQLite + SQLAlchemy ORM）

三张核心表:
  - positions: 模拟持仓记录
  - trade_logs: 交易日志
  - daily_snapshots: 每日净值快照
"""

import os
import sqlite3
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── 数据库路径（统一走 config/paths.py，运行时解析） ──
#
# ⚠️⚠️ 不能用 `Path(__file__).resolve().parent.parent` 算死：打包成 exe 后
#   `__file__` 位于只读包 `_internal/` ⇒ `DB_PATH` 指向 `_internal/data/portfolio.db`
#   ⇒ 一个**新建的空库**（0 持仓、100 万现金）。后果不是报错，而是交易台
#   "看起来持仓全没了"（2026-10-25 用户实测："目前只有个仪表盘能用"）。
#   即使 `data_root()` 修好，这一行不改也照样是空账户 —— 两个坑独立的。
#
# 保持"模块属性"形态（`from db.models import DB_PATH` 在 store.py 等处使用），
# 由 `paths.db_path()` 在导入时求值一次；测试用 `monkeypatch.setattr` 覆盖
# （见 tests/conftest.py::use_template_db，它会同时 patch models 与 store 两处）。
from config.paths import db_path as _db_path

DB_PATH = _db_path()


# ================================================================
# 数据类（与 SQLite 表对应）
# ================================================================

@dataclass
class Position:
    """模拟持仓"""
    code: str               # 股票代码
    name: str = ""          # 股票名称
    entry_date: str = ""    # 建仓日期 YYYY-MM-DD
    entry_price: float = 0.0
    quantity: int = 0       # 持股数量（默认100股/手，整数）
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: str = "active"  # active | stopped_out | closed
    close_date: Optional[str] = None
    close_price: Optional[float] = None
    close_reason: Optional[str] = None
    id: int = 0             # 自增主键（从DB读取时填充）


@dataclass
class TradeLog:
    """交易日志"""
    date: str               # YYYY-MM-DD HH:MM
    code: str
    direction: str          # 买入 | 卖出
    price: float
    volume: int             # 股数
    reason: str = ""
    signal_source: str = ""  # 择时选股 | 止损 | 止盈 | 手动
    pnl: Optional[float] = None
    id: int = 0


@dataclass
class DailySnapshot:
    """每日净值快照"""
    date: str               # YYYY-MM-DD
    total_value: float = 0.0
    cash: float = 0.0
    position_value: float = 0.0
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    benchmark_return: float = 0.0


# ================================================================
# 数据库初始化 & 连接管理
# ================================================================

# 写锁等待时间（秒）。SQLite 默认的 delete 日志模式下，两个进程同时写会立刻抛
# "database is locked"；这里给一个等待窗口，配合 WAL 让读写互不阻塞。
_BUSY_TIMEOUT_SECONDS = 15.0


def get_connection() -> sqlite3.Connection:
    """获取 SQLite 连接（自动创建目录）

    并发说明（批次 12 后成为硬需求）：streamlit UI 进程与 scheduler 进程会
    **同时写同一个库**（analyses 表两侧都写）。默认 journal_mode=delete 在写
    冲突时会直接抛 "database is locked"，因此必须设 busy timeout 让它排队等待
    ——`timeout=N` 即等价于 PRAGMA busy_timeout（**连接级**，每次连接都要设）。

    ⚠️ 这里**刻意不使用 WAL**（2026-09-24 实测结论，勿轻易改回）：
    本项目 store 层是"每次调用开关一次连接"的短连接模式，而 WAL 会在
    **最后一个连接关闭时做 checkpoint 并删除 -wal / -shm**，另一个进程正好
    在此期间写入就会撞上 "attempt to write a readonly database"。
    对照实验（双进程 × 60 次写入，Windows）：
        WAL          → 119/120（失败 1 次）
        默认 delete   → 120/120（0 失败）
    代价只是"写时短暂阻塞读"，而本项目读写频率都很低，可接受。
    若将来改为长连接池，可重新评估 WAL。
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=_BUSY_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表（幂等操作）"""
    conn = get_connection()
    cursor = conn.cursor()

    # ── 持仓表 ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            name TEXT DEFAULT '',
            entry_date TEXT NOT NULL,
            entry_price REAL NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 100,
            stop_loss REAL,
            take_profit REAL,
            status TEXT DEFAULT 'active',
            close_date TEXT,
            close_price REAL,
            close_reason TEXT
        )
    """)

    # ── 交易日志 ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            code TEXT NOT NULL,
            direction TEXT NOT NULL,
            price REAL NOT NULL,
            volume INTEGER NOT NULL,
            reason TEXT DEFAULT '',
            signal_source TEXT DEFAULT '',
            pnl REAL
        )
    """)

    # ── 每日净值快照 ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_snapshots (
            date TEXT PRIMARY KEY,
            total_value REAL DEFAULT 0,
            cash REAL DEFAULT 0,
            position_value REAL DEFAULT 0,
            daily_return REAL DEFAULT 0,
            cumulative_return REAL DEFAULT 0,
            benchmark_return REAL DEFAULT 0
        )
    """)

    # 为常用查询建立索引
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_positions_code ON positions(code)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_trade_logs_date ON trade_logs(date)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_trade_logs_code ON trade_logs(code)
    """)

    conn.commit()
    conn.close()


def drop_all_tables():
    """⚠️ 危险操作：删除所有表。仅用于测试/重置。"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS positions")
    cursor.execute("DROP TABLE IF EXISTS trade_logs")
    cursor.execute("DROP TABLE IF EXISTS daily_snapshots")
    conn.commit()
    conn.close()
    print("⚠️ 所有表已删除")