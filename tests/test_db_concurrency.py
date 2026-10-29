"""SQLite 并发与跨进程任务锁单元测试

背景（批次 12 之后的硬需求）：streamlit UI 进程与 scheduler 进程会**同时写同一个库**
（analyses 表两侧都写），且 T+N 回填既有调度（15:35）也有 UI 手动按钮，是不同进程。

关于 WAL 的重要实测结论（2026-09-24，Windows，勿轻易改回）：
  本项目 store 层是"每次调用开关一次连接"的短连接模式，而 WAL 会在最后一个连接
  关闭时做 checkpoint 并删除 -wal/-shm，另一进程正好在此期间写入就会撞上
  "attempt to write a readonly database"。
  对照实验（双进程 × 60 次写入）：WAL → 119/120（失败 1 次）；默认 delete → 120/120。
  故采用 **delete 模式 + busy_timeout**。下面第一条断言就是把这个结论固化下来，
  防止后人"顺手优化"成 WAL 而引入偶发写入丢失。
"""
import time

import pytest

import db.models as models
from db import store as db_store


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    db_path = use_template_db(tmp_path, monkeypatch, "conc.db")
    yield


@pytest.fixture
def temp_store(temp_db):
    yield


class TestConnectionSettings:

    def test_busy_timeout_set_and_wal_disabled(self, temp_db):
        """必须设 busy_timeout（否则双进程写直接报 database is locked）；
        并且**不启用 WAL**（短连接 + 双进程下 WAL 反而会丢写入，见文件头注释）"""
        conn = models.get_connection()
        try:
            assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 15000
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        finally:
            conn.close()

    def test_concurrent_writes_from_two_connections(self, temp_db):
        """两个连接交替写入（模拟 UI 与调度交错）：不应抛 database is locked，且数据不丢"""
        for i in range(10):
            c1 = models.get_connection()
            c2 = models.get_connection()
            try:
                c1.execute(
                    "INSERT INTO analyses (analysis_date, stock_code, created_at) "
                    "VALUES (?, ?, ?)", ("2026-09-24", f"60000{i}", "2026-09-24 10:00:00"))
                c1.commit()
                c2.execute(
                    "INSERT INTO analyses (analysis_date, stock_code, created_at) "
                    "VALUES (?, ?, ?)", ("2026-09-24", f"00000{i}", "2026-09-24 10:00:00"))
                c2.commit()
            finally:
                c1.close()
                c2.close()
        assert len(db_store.list_analyses(limit=100)) == 20


class TestTaskLock:

    def test_mutual_exclusion(self, temp_store):
        assert db_store.acquire_task_lock("backfill", "holder-A") is True
        assert db_store.acquire_task_lock("backfill", "holder-B") is False

    def test_release_then_reacquire(self, temp_store):
        assert db_store.acquire_task_lock("backfill", "holder-A") is True
        db_store.release_task_lock("backfill", "holder-A")
        assert db_store.acquire_task_lock("backfill", "holder-B") is True

    def test_non_holder_cannot_release(self, temp_store):
        """释放必须带 holder，否则会误删超时后已被别人抢占的锁"""
        db_store.acquire_task_lock("backfill", "holder-A")
        db_store.release_task_lock("backfill", "intruder")
        assert db_store.acquire_task_lock("backfill", "holder-C") is False

    def test_expired_lock_can_be_preempted(self, temp_store):
        """持有者崩溃时 TTL 兜底：锁必须能自动失效，不留死锁"""
        assert db_store.acquire_task_lock("backfill", "dead-process", ttl_seconds=1) is True
        time.sleep(2.0)          # 跨过秒级时间戳边界
        assert db_store.acquire_task_lock("backfill", "new-process") is True

    def test_different_tasks_do_not_block(self, temp_store):
        db_store.acquire_task_lock("backfill", "h")
        assert db_store.acquire_task_lock("other_task", "h") is True


class TestBackfillLockIntegration:

    def test_locked_backfill_does_not_fetch(self, temp_store):
        """另一进程正在回填 → 本次直接跳过，且**一次 K 线都不拉**（避免重复消耗额度）"""
        from core.analysis_history import backfill_analyses
        db_store.insert_analysis({"analysis_date": "2026-09-01", "stock_code": "600519"})
        db_store.acquire_task_lock("backfill_analysis", "other-process")

        calls = []
        result = backfill_analyses(price_fetcher=lambda c, s, e: calls.append(c))
        assert result["locked"] is True
        assert result["checked"] == 0
        assert calls == []

    def test_lock_released_after_successful_run(self, temp_store):
        """正常跑完后锁必须释放，否则第二天调度会以为任务一直在跑"""
        from core.analysis_history import backfill_analyses
        db_store.insert_analysis({"analysis_date": "2026-09-01", "stock_code": "600519"})
        backfill_analyses(price_fetcher=lambda c, s, e: None)
        assert db_store.acquire_task_lock("backfill_analysis", "next-run") is True

    def test_lock_released_even_when_fetcher_raises(self, temp_store):
        """数据源异常时也要解锁（finally 保证），否则永久死锁"""
        from core.analysis_history import backfill_analyses
        import datetime as dt
        ana_date = (dt.datetime.now() - dt.timedelta(days=30)).strftime("%Y-%m-%d")
        db_store.insert_analysis({"analysis_date": ana_date, "stock_code": "600519"})

        def _boom(c, s, e):
            raise RuntimeError("限频")

        backfill_analyses(price_fetcher=_boom)
        assert db_store.acquire_task_lock("backfill_analysis", "next-run") is True
