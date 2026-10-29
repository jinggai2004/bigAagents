"""pytest 全局配置：注入项目根目录到 sys.path + 空库模板复用

为什么需要模板库
----------------
全仓 300+ 个用例都是"每个用例新建一个 SQLite 文件 + init_store() 建 20+ 张表"。
实测在 Windows 上**光建表就要 1.6s**（加上实时防护扫描，单用例 2~8s），
全量测试因此要跑 20 分钟以上。

建出来的都是**同一个空库**，没必要重复建。本模块提供：
- `template_db_path()`：整个 session 只建一次空库（models 3 张表 + store 全部表）
- `use_template_db()`：copyfile 一份给用例用 —— 语义等价于"新建文件 + init_store"，
  但省掉建表，实测快一个量级。

⚠️ 等价性由 tests/test_db_template.py 守护（表集合一致、无残留数据、
用例之间互不影响）。改这里必须先跑那个文件。
"""

import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

_TEMPLATE_DB = None


def template_db_path() -> str:
    """空库模板路径（惰性创建，整个 session 只建一次）"""
    global _TEMPLATE_DB
    if _TEMPLATE_DB is not None:
        return _TEMPLATE_DB

    import tempfile

    import db.models as models
    from db import store as db_store

    path = os.path.join(tempfile.mkdtemp(prefix="biga_db_template_"), "template.db")
    old_models, old_store = models.DB_PATH, db_store.DB_PATH
    try:
        models.DB_PATH = path
        db_store.DB_PATH = path
        models.init_db()        # positions / trade_logs / daily_snapshots
        db_store.init_store()   # store 层的全部表
    finally:
        models.DB_PATH = old_models
        db_store.DB_PATH = old_store

    _TEMPLATE_DB = path
    return path


def use_template_db(tmp_path, monkeypatch, name: str = "test.db") -> str:
    """给当前用例一个独立空库（从模板复制），并把 DB_PATH 指过去

    Args:
        tmp_path: pytest 的 tmp_path（用例级，天然隔离）
        monkeypatch: 用例级 patch，用例结束自动还原
        name: 库文件名（不同夹具可用不同名字，便于排查）

    Returns:
        库文件路径
    """
    import shutil

    import db.models as models
    from db import store as db_store

    dst = str(tmp_path / name)
    shutil.copyfile(template_db_path(), dst)
    # 两处都 patch：store 从 models 导入时绑定了同名的 DB_PATH，
    # 有些夹具只 patch 一侧，双 patch 才能保证万无一失。
    monkeypatch.setattr(models, "DB_PATH", dst)
    monkeypatch.setattr(db_store, "DB_PATH", dst)
    return dst


# ================================================================
# 生产库污染哨兵
# ================================================================
#
# 真实事故（2026-09-29）：新增的某个测试类**没有声明 DB 夹具**，于是它直接写到
# 真实的 data/portfolio.db，并在收尾时 `DROP TABLE scheduled_jobs`。
# 后果不是"报个错"那么轻——`init_store()` 用 schema 哈希存进 `PRAGMA user_version`
# 做短路，表被删掉而指纹没变 → 它判定"库已是最新"→ **那张表永久不会重建**，
# 之后所有用例一律 `no such table`，连真实运行库里也真的没了那张表。
#
# 为什么用"表集合 + schema 指纹"而不是文件 mtime 来判定：
# 调度器进程在交易时段会持续写库（日志、预警、快照），mtime 一直在变，
# 拿它当判据会天天误报。而行**级别**的写入是合法的，**结构级别**的变化不是：
# 正常测试与被调度的业务任务都不会改表结构、也不会改 user_version。
_REAL_DB = os.path.join(project_root, "data", "portfolio.db")


def _real_db_fingerprint():
    """(表名集合, user_version) —— 结构指纹，读不到返回 None（判不了就不判）"""
    import sqlite3
    try:
        conn = sqlite3.connect(f"file:{_REAL_DB}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return None
    try:
        tables = frozenset(
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"))
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        return tables, version
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def pytest_configure(config):
    config._real_db_before = _real_db_fingerprint()


@pytest.fixture(autouse=True)
def _no_real_full_market(monkeypatch, tmp_path_factory):
    """**默认禁止** `data.full_market.save_full_snapshot` 真的联网采集。

    为什么做成 autouse：`core.scheduler.save_daily_market_snapshot` 的**主路径**
    现在是全量域（新浪 `hs_a` 5,572 只），而它被十来处用例间接调用
    （快照/覆盖率/tushare 回退…）。不拦的话这些用例会：
      ① 真的打新浪接口（离线自检/CI 直接挂）；
      ② 把**真实全市场数据**写进临时库，于是"回退路径"断言全错。

    需要验证主路径的用例，在自己的用例里重新 patch 这个函数即可（用例级
    monkeypatch 在 autouse 之后生效，能覆盖本夹具）。

    ⚠️ 同时把 `stock_basic` 的**磁盘缓存**指到临时目录。踩过：某个用例顺手调到
       `stock_basic_map()`，把 mock 的 1 行清单写进真实
       `stock_basic_cache.json`，之后真跑就以为"全市场只有 1 只"。
    """
    import data.full_market as _fm

    def _blocked(*_a, **_k):
        return {"ok": False, "reason": "测试环境禁止真实全量域采集（conftest 拦截）"}

    monkeypatch.setattr(_fm, "save_full_snapshot", _blocked)

    cache = tmp_path_factory.mktemp("stock_basic") / "cache.json"
    monkeypatch.setenv("BIGA_STOCK_BASIC_CACHE", str(cache))
    yield


@pytest.fixture(autouse=True)
def _review_llm_off(monkeypatch):
    """**默认关闭**回填里程碑的自动 LLM 归因（`BIGA_ANALYSIS_REVIEW_LLM=0`）。

    为什么做成 autouse（2026-10-29）：`analysis_history._backfill_rows` 现在会在
    T+5/T+20 里程碑自动调 LLM（生产默认开，用户需求）。不拦的话，十几个
    回填相关用例会在里程碑上**真的构造 DeepSeekClient 并打外网 API** ——
    违反"测试不得耦合真实环境"的铁律，且 CI/离线直接挂。

    需要验证 LLM 路径的用例，在自己的用例里 `monkeypatch.setenv(
    "BIGA_ANALYSIS_REVIEW_LLM", "1")` + 注入 stub client（用例级 monkeypatch
    在 autouse 之后生效，能覆盖本夹具）。
    """
    monkeypatch.setenv("BIGA_ANALYSIS_REVIEW_LLM", "0")
    yield


def _real_db_diff(before, after) -> dict:
    lost = sorted(set(before[0]) - set(after[0]))
    added = sorted(set(after[0]) - set(before[0]))
    return {"lost": lost, "added": added,
            "version_changed": before[1] != after[1]}


def pytest_sessionfinish(session, exitstatus):
    """测试改坏了生产库的表结构 → 响。

    ⚠️ 必须在这个钩子里改 `exitstatus`（`pytest_terminal_summary` 在它之后才跑）。

    判据只把"**丢表**"升级为失败：
      - 丢表 = 不可逆的破坏（正是本次事故的形态），正常测试与业务任务都不会删表；
      - "多出表 / 指纹变了"只提示：开发者改了 `_SCHEMA` 后首次被迁移到真实库
        是**合法**的，拿它判失败会误伤。
    """
    before = getattr(session.config, "_real_db_before", None)
    after = _real_db_fingerprint()
    if not before or not after or before == after:
        return

    diff = _real_db_diff(before, after)
    session.config._real_db_pollution = {"before": before, "after": after, **diff}
    if diff["lost"]:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    info = getattr(config, "_real_db_pollution", None)
    if not info:
        return
    hard = bool(info["lost"])
    title = ("测试删掉了生产库的表" if hard else "生产库表结构在测试期间发生了变化")
    sep = "!" if hard else "-"
    terminalreporter.write_sep(sep, title, red=hard)
    terminalreporter.write_line(
        f"真实库 {_REAL_DB}：丢表 {info['lost'] or '无'}｜多表 {info['added'] or '无'}"
        f"｜schema 指纹 {info['before'][1]} → {info['after'][1]}", red=hard)
    terminalreporter.write_line(
        "用例必须通过 tests.conftest.use_template_db 拿临时库。"
        "直接写真实库会连带毁掉 init_store 的惰性短路——表被删而指纹没变时，"
        "它判定「库已是最新」，那张表**再也不会重建**。", red=hard)
