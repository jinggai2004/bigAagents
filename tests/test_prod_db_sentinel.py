"""生产库污染哨兵测试。

背景（2026-09-29 真实事故）：新增的测试类**没声明 DB 夹具**，直接写到真实的
`data/portfolio.db` 并在收尾 `DROP TABLE scheduled_jobs`。后果不是"报个错"：
`init_store()` 把 schema 哈希存进 `PRAGMA user_version` 做短路，表被删而指纹未变
→ 它判定"库已是最新" → **那张表再也不会重建**，之后所有用例一律 `no such table`，
真实运行库里也真的没了那张表。

所以哨兵必须满足两点，本文件各钉一条：
1. 能识别"表被删"（而不是把文件 mtime 当判据——调度器交易时段一直在写库，会天天误报）；
2. 识别到就**让退出码变 1**，不能只打一行容易被淹掉的警告
   （本项目连最终汇总行都可能落不了盘，见 MEMORY）。
"""
import os
import sqlite3

import pytest

import tests.conftest as conftest


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """本模块也绝不碰真实库（理由见文件头）。"""
    from tests.conftest import use_template_db
    return use_template_db(tmp_path, monkeypatch, "sentinel.db")


def _make_db(path, tables=("analyses", "screen_picks")):
    conn = sqlite3.connect(str(path))
    for t in tables:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {t} (id INTEGER)")
    conn.execute("PRAGMA user_version = 12345")
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture
def fake_real_db(tmp_path, monkeypatch):
    """把哨兵盯防的文件换成一个临时库（绝不动真实库）"""
    path = _make_db(tmp_path / "fake_prod.db")
    monkeypatch.setattr(conftest, "_REAL_DB", path)
    return path


class TestFingerprint:

    def test_fingerprint_uses_structure_not_mtime(self, fake_real_db):
        """判据必须是"表集合 + schema 指纹"：调度器交易时段持续写库，
        用 mtime 会天天误报。"""
        before = conftest._real_db_fingerprint()
        conn = sqlite3.connect(fake_real_db)
        conn.execute("INSERT INTO analyses (id) VALUES (1)")     # 合法写入
        conn.commit()
        conn.close()
        os.utime(fake_real_db, None)                             # 时间戳也变了
        assert conftest._real_db_fingerprint() == before, \
            "行级写入 + 时间戳变化不该被判为污染"

    def test_missing_file_is_not_judged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(conftest, "_REAL_DB", str(tmp_path / "nope.db"))
        assert conftest._real_db_fingerprint() is None           # 判不了就不判

    def test_diff_reports_lost_added_and_version(self, fake_real_db):
        before = conftest._real_db_fingerprint()
        conn = sqlite3.connect(fake_real_db)
        conn.execute("DROP TABLE screen_picks")
        conn.execute("CREATE TABLE garbage (id INTEGER)")
        conn.execute("PRAGMA user_version = 999")
        conn.commit()
        conn.close()
        diff = conftest._real_db_diff(before, conftest._real_db_fingerprint())
        assert diff["lost"] == ["screen_picks"]
        assert diff["added"] == ["garbage"]
        assert diff["version_changed"] is True


class TestSessionHooks:
    """钩子顺序很容易踩错：`pytest_terminal_summary` 在 `pytest_sessionfinish`
    **之后**才跑，所以 exitstatus 必须在 sessionfinish 里改。"""

    class _Cfg:
        pass

    class _Session:
        def __init__(self, cfg):
            self.config = cfg
            self.exitstatus = 0

    def _run(self, fake_real_db, corrupt):
        cfg = self._Cfg()
        cfg._real_db_before = conftest._real_db_fingerprint()
        if corrupt:
            conn = sqlite3.connect(fake_real_db)
            conn.execute("DROP TABLE screen_picks")
            conn.commit()
            conn.close()
        session = self._Session(cfg)
        conftest.pytest_sessionfinish(session, 0)
        return session

    def test_dropped_table_fails_the_run(self, fake_real_db):
        session = self._run(fake_real_db, corrupt=True)
        assert session.exitstatus == 1, "删掉生产库的表必须让测试失败，不能只警告"
        assert session.config._real_db_pollution["lost"] == ["screen_picks"]

    def test_clean_run_changes_nothing(self, fake_real_db):
        session = self._run(fake_real_db, corrupt=False)
        assert session.exitstatus == 0
        assert not getattr(session.config, "_real_db_pollution", None)

    def test_added_table_alone_does_not_fail(self, fake_real_db):
        """开发者改了 `_SCHEMA` 后首次被迁移到真实库是**合法**的，
        拿"多出表"判失败会误伤——只提示。"""
        cfg = self._Cfg()
        cfg._real_db_before = conftest._real_db_fingerprint()
        conn = sqlite3.connect(fake_real_db)
        conn.execute("CREATE TABLE new_feature (id INTEGER)")
        conn.commit()
        conn.close()
        session = self._Session(cfg)
        conftest.pytest_sessionfinish(session, 0)
        assert session.exitstatus == 0
        assert session.config._real_db_pollution["added"] == ["new_feature"]

    def test_terminal_summary_prints_loud_banner(self, fake_real_db):
        session = self._run(fake_real_db, corrupt=True)
        printed = []

        class TR:
            def write_sep(self, sep, title=None, **kw):
                printed.append((sep, title))

            def write_line(self, line, **kw):
                printed.append(line)

        conftest.pytest_terminal_summary(TR(), session.exitstatus, session.config)
        assert printed[0] == ("!", "测试删掉了生产库的表")   # 用 ! 分隔符 + 红色
        text = " ".join(str(x) for x in printed)
        assert "screen_picks" in text
        assert "use_template_db" in text      # 必须给出怎么修

    def test_terminal_summary_silent_when_clean(self, fake_real_db):
        session = self._run(fake_real_db, corrupt=False)
        printed = []

        class TR:
            def write_sep(self, *a, **kw):
                printed.append(a)

            def write_line(self, *a, **kw):
                printed.append(a)

        conftest.pytest_terminal_summary(TR(), session.exitstatus, session.config)
        assert printed == []


class TestThisModuleIsIsolated:
    """自证：本仓库新增涉及 DB 的测试模块必须自带隔离。"""

    def test_db_path_points_to_tmp(self, _isolated_db, tmp_path):
        import db.models as models
        assert os.path.abspath(models.DB_PATH) != os.path.abspath(conftest._REAL_DB)
        assert str(tmp_path) in os.path.abspath(models.DB_PATH)

    def test_scheduled_jobs_module_has_autouse_isolation(self):

        """`test_scheduled_jobs.py` 必须用 autouse 夹具做隔离。

        写成"每个类自己声明夹具"就可能漏掉一类——本次事故正是这样发生的，
        而后果是**生产库的表被删且不再重建**。
        """
        import inspect
        import tests.test_scheduled_jobs as mod
        src = inspect.getsource(mod)
        assert "@pytest.fixture(autouse=True)" in src, \
            "隔离必须是 autouse，否则新增测试类会静默写到真实库"
