# -*- coding: utf-8 -*-
"""调度器单实例互斥（2026-10-21 事故后建立）。

**事故**：三个调度器实例在同一秒启动，**全部通过**了原来的"读心跳判断 pid"检查
——因为心跳是启动成功后由后台线程写的，那一刻谁都还没写，`read_heartbeat()` 返空。

日志实证（`logs/scheduler_20261021.log`）：

```
09:19:36.875  已注册任务: 盘前选股            ← 同一毫秒三份
09:25:00.003  Running job "盘前选股"          ← 三份
09:25:00.021  Stage 1/3 选股开始 (max=5)      ← 三份
10:18:31.517  Pipeline 完成: 3211.5s, picks=5
10:18:45.980  Pipeline 完成: 3225.9s, picks=5 ← 三个不同耗时 = 三次真实执行
10:18:58.853  Pipeline 完成: 3238.8s, picks=5
```

Stage 1（选股）独占 3211 秒 ≈ 53.5 分钟，三份并发 ⇒ 互相抢网络与 SQLite，
`screen_picks` 同日 45 行 / 7 只票就是这么来的。

⇒ 换成 OS 级命名互斥（内核在进程退出时自动释放），不存在"检查-后-启动"的窗口。
"""
import os
import subprocess
import sys
import uuid

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_scheduler as R  # noqa: E402


@pytest.fixture
def mutex_name():
    """每个用例用独立的互斥名，避免与真实调度器、其他用例串扰。"""
    return f"bigAagents_test_{uuid.uuid4().hex[:12]}"


class TestSingletonMutex:

    def test_first_acquire_succeeds(self, mutex_name):
        assert R._acquire_os_mutex(mutex_name) is True

    def test_same_process_second_call_is_idempotent(self, mutex_name):
        """同进程重复调用必须 True。

        ⚠️ 否则模块被重新导入一次（或入口函数被调两次）就会把自己判成"已有实例"。
        """
        assert R._acquire_os_mutex(mutex_name) is True
        assert R._acquire_os_mutex(mutex_name) is True
        assert R._acquire_os_mutex(mutex_name) is True

    def test_second_process_is_rejected(self, mutex_name):
        """⭐ 核心用例：另一个进程抢同一个名字必须被拒。

        这是原实现（读心跳）抓不到的场景 —— 两个进程同时启动时都还没写心跳。
        """
        assert R._acquire_os_mutex(mutex_name) is True
        code = (
            "import sys; sys.path.insert(0, r'{root}/scripts');"
            "import run_scheduler as R;"
            "print(R._acquire_os_mutex(r'{name}'))"
        ).format(root=ROOT, name=mutex_name)
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, timeout=120)
        assert proc.stdout.strip() == "False", (
            f"第二个进程竟然也拿到了互斥：stdout={proc.stdout!r} stderr={proc.stderr[-400:]!r}")

    def test_different_names_do_not_conflict(self):
        """名字不同就不该互相阻塞（测试之间、多个用途之间可共存）。"""
        a = f"t_{uuid.uuid4().hex[:8]}"
        b = f"t_{uuid.uuid4().hex[:8]}"
        assert R._acquire_os_mutex(a) is True
        assert R._acquire_os_mutex(b) is True

    def test_release_on_process_exit_is_automatic(self, mutex_name):
        """子进程退出后，互斥必须自动释放（不需要人工清理）。

        这是选命名 mutex 而不是锁文件的原因：锁文件在进程被 kill 后会残留，
        下次启动就会永久起不来。
        """
        code = (
            "import sys; sys.path.insert(0, r'{root}/scripts');"
            "import run_scheduler as R;"
            "print(R._acquire_os_mutex(r'{name}'))"
        ).format(root=ROOT, name=mutex_name)
        first = subprocess.run([sys.executable, "-c", code],
                               capture_output=True, text=True, timeout=120)
        assert first.stdout.strip() == "True", first.stderr[-400:]
        # 子进程已退出 ⇒ 本进程现在应该能拿到
        assert R._acquire_os_mutex(mutex_name) is True
