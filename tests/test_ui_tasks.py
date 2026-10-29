"""core/ui_tasks 的测试：后台线程执行、进度文件、刷新存活、互斥、取消

这些都是"刷新不再打断任务"这一承诺的可验证部分：
  - 任务跑在普通线程里，进度/结果只写文件，页面只是读者；
  - 页面（乃至整个 Streamlit 会话）消失后任务继续跑完并留下结果；
  - 重复启动被心跳存活判定挡住；
  - 取消是协作式的（工作线程在下一个进度点退出）。
"""

import json
import os
import threading
import time

import pytest

from core import ui_runs, ui_tasks


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_runs, "UI_RUNS_DIR", str(tmp_path))
    ui_tasks.reset_for_tests()
    yield
    ui_tasks.reset_for_tests()


def wait_done(task, timeout=5.0):
    """等任务结束（标记被清），返回最终进度"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not ui_tasks.is_running(task):
            return ui_tasks.read_progress(task)
        time.sleep(0.02)
    raise AssertionError(f"任务 {task} 在 {timeout}s 内没有结束")


# ----------------------------------------------------------------
# 基本执行
# ----------------------------------------------------------------

class TestStart:
    def test_runs_in_background_and_saves_result(self):
        def job(progress):
            progress({"percent": 50.0, "message": "做一半"})
            ui_runs.save_result("demo", {"answer": 42})

        assert ui_tasks.start("demo", job) is True
        done = wait_done("demo")
        assert done["status"] == "done"
        assert done["percent"] == 100.0
        assert ui_runs.load_result("demo")["payload"] == {"answer": 42}

    def test_progress_gets_status_and_elapsed(self):
        def job(progress):
            progress({"percent": 10.0, "message": "第一步"})

        ui_tasks.start("demo", job)
        done = wait_done("demo")
        assert done["status"] == "done"          # 最后一次是完成
        assert isinstance(done["elapsed"], float)

    def test_result_available_without_reading_progress(self):
        """页面刷新后只知道任务名，也能拿到结果"""
        seen = {}

        def job(progress):
            progress({"percent": 100.0, "message": "结束"})
            ui_runs.save_result("demo", {"rows": [1, 2, 3]})

        ui_tasks.start("demo", job)
        wait_done("demo")
        assert ui_runs.load_result("demo")["payload"]["rows"] == [1, 2, 3]
        assert seen == {}

    def test_marker_cleared_after_finish(self):
        ui_tasks.start("demo", lambda progress: None)
        wait_done("demo")
        assert ui_runs.get_running_any("demo") is None

    def test_worker_does_not_need_streamlit(self):
        """工作线程里 streamlit 的运行时上下文不存在（也不能被使用）"""
        import streamlit.runtime as st_runtime

        captured = {}

        def job(progress):
            captured["ctx"] = st_runtime.exists()

        ui_tasks.start("demo", job)
        wait_done("demo")
        assert captured["ctx"] is False

    def test_progress_file_replaced_between_runs(self):
        ui_tasks.start("demo", lambda progress: progress({"message": "第一次"}))
        wait_done("demo")
        ui_tasks.start("demo", lambda progress: None)
        wait_done("demo")
        assert ui_tasks.read_progress("demo")["status"] == "done"


# ----------------------------------------------------------------
# 刷新存活：会话没了，任务照跑
# ----------------------------------------------------------------

class TestSurvivesRefresh:
    def test_task_keeps_running_after_ui_forgets_it(self):
        """模拟"用户刷新页面"：不再引用线程、不再回调，任务仍跑完并落盘"""
        gate = threading.Event()

        def job(progress):
            progress({"percent": 1.0, "message": "开始"})
            gate.wait(3.0)                       # 假装在慢慢跑
            ui_runs.save_result("demo", {"ok": True})

        ui_tasks.start("demo", job)
        assert ui_tasks.is_running("demo") is True
        # 页面（会话）在这里"没了"：什么都不做，等任务自己跑完
        time.sleep(0.1)
        gate.set()
        done = wait_done("demo", timeout=5.0)
        assert done["status"] == "done"
        assert ui_runs.load_result("demo")["payload"] == {"ok": True}

    def test_heartbeat_keeps_marker_alive_for_slow_task(self):
        def job(progress):
            for i in range(3):
                progress({"percent": i * 30.0, "message": f"第 {i} 步"})
                time.sleep(0.05)

        ui_tasks.start("demo", job)
        time.sleep(0.05)
        record = ui_runs.get_running_any("demo")
        assert record is not None
        # 心跳新鲜 → 判定为运行中；而 started_ts 已过去一点点
        assert ui_runs.is_alive(record, heartbeat_timeout=1) is True
        wait_done("demo")

    def test_progress_readable_while_running(self):
        release = threading.Event()

        def job(progress):
            progress({"percent": 33.0, "message": "正在处理 600519"})
            release.wait(3.0)

        ui_tasks.start("demo", job)
        deadline = time.time() + 2
        data = None
        while time.time() < deadline:
            data = ui_tasks.read_progress("demo")
            if data and data.get("percent") == 33.0:
                break
            time.sleep(0.02)
        release.set()
        assert data["message"] == "正在处理 600519"
        wait_done("demo")


# ----------------------------------------------------------------
# 心跳看门狗：存活判定不能依赖业务代码何时回调进度
# ----------------------------------------------------------------

class TestHeartbeatWatchdog:
    """真实故障回归（用户报告）：

    自然语言选股选了 15 只候选（pool = max_stocks×3），在「问财初筛」整段只有
    查询前后两次进度回调，中间的问财请求会重试/限频。只要这段超过
    `ui_runs.HEARTBEAT_TIMEOUT_SECONDS`（180 秒），页面就把**仍在后台正常跑**
    的任务判成"上一次选股没有正常结束（进程被关闭或崩溃）"，
    并且把该任务还要用的进度文件删掉 —— 用户看到的正是"提示崩溃、但后台还在跑"。

    根因：心跳原先只在 progress() 里刷新，于是"心跳停了"实际表达的是
    "业务代码 180 秒没回调"，而不是"进程没了"。看门狗把这两件事解耦。
    """

    def test_task_stays_alive_through_silent_phase(self, monkeypatch):
        """整段不调用 progress，也必须一直被判定为"在运行\""""
        # 把超时压到 0.6 秒、看门狗周期压到 0.1 秒：
        # 这样"没有看门狗就必然误判"在 1.5 秒内即可测出，不用真等 180 秒。
        monkeypatch.setattr(ui_runs, "HEARTBEAT_TIMEOUT_SECONDS", 0.6)
        monkeypatch.setattr(ui_tasks, "HEARTBEAT_INTERVAL_SECONDS", 0.1)
        release = threading.Event()

        def job(progress):
            release.wait(10)                 # 静默阶段：一次 progress 都不调

        assert ui_tasks.start("nl_screen", job) is True
        try:
            time.sleep(1.5)                  # 远超 0.6 秒的超时
            assert ui_tasks.is_running("nl_screen") is True, \
                "静默阶段（如问财初筛）不该被判成已中断"
            # 页面据此分支：仍在运行 → 显示"后台线程"而不是"没有正常结束"
            assert ui_runs.get_running_any("nl_screen") is not None
        finally:
            release.set()
            wait_done("nl_screen")

    def test_watchdog_does_not_resurrect_finished_task(self, monkeypatch):
        """任务结束后，看门狗不得把已清除的标记又写回成"运行中"（phantom running）"""
        monkeypatch.setattr(ui_tasks, "HEARTBEAT_INTERVAL_SECONDS", 0.05)

        def job(progress):
            progress({"percent": 100.0, "message": "完成"})

        assert ui_tasks.start("nl_screen", job) is True
        wait_done("nl_screen")
        time.sleep(0.3)                      # 给看门狗若干轮"作案"机会
        assert ui_runs.get_running_any("nl_screen") is None, \
            "任务已结束，标记不得被看门狗写回"
        assert ui_tasks.is_running("nl_screen") is False

    def test_heartbeat_after_clear_is_noop(self):
        """心跳在读-改-写层面就必须是幂等的：标记没了就不写"""
        ui_runs.mark_running("nl_screen")
        ui_runs.clear_running("nl_screen")
        assert ui_runs.heartbeat("nl_screen") is False
        assert ui_runs.get_running_any("nl_screen") is None

    def test_watchdog_stops_after_task_ends(self):
        """看门狗线程随任务结束而退出，不残留线程"""
        before = threading.active_count()
        ui_tasks.start("nl_screen", lambda progress: None)
        wait_done("nl_screen")
        deadline = time.time() + 2
        while threading.active_count() > before and time.time() < deadline:
            time.sleep(0.02)
        assert threading.active_count() <= before, "任务结束后不应残留看门狗线程"

    def test_running_task_heals_its_own_lost_marker(self, monkeypatch):
        """标记被误清后，仍在执行的任务应在下一个进度点把标记**自愈**回来。

        这正是用户遇到的现场：页面误判崩溃 → clear_dead_running() 删掉标记 →
        任务仍在跑却对页面永久不可见。工作线程自己就是"任务还活着"的权威，
        因此由它在下一个进度点重新登记。

        注意测试要**等第一个进度点做完再清标记**：`start()` 是同步登记标记的，
        若不等，清标记会与工作线程的第一个 progress() 并发——而那个回调本身
        就会自愈，于是"标记必须为空"的中间断言会随机失败（这不是产品缺陷，
        是断言过强）。
        """
        monkeypatch.setattr(ui_tasks, "HEARTBEAT_INTERVAL_SECONDS", 0.05)
        first_done = threading.Event()
        release = threading.Event()
        healed = threading.Event()
        hold = threading.Event()

        def job(progress):
            progress({"percent": 10.0, "message": "第一步"})
            first_done.set()                 # 第一个进度点（含心跳/自愈）已完成
            release.wait(10)                 # 停在这里，期间不会再有任何回调
            progress({"percent": 20.0, "message": "第二步"})   # 自愈发生在这里
            healed.set()
            hold.wait(10)                    # 继续运行，让断言有机会观察标记

        assert ui_tasks.start("nl_screen", job) is True
        assert first_done.wait(3), "工作线程应在 3 秒内完成第一个进度点"

        # 模拟页面误判：清掉标记（此时任务确实还在跑，且已停在 release 上）
        ui_runs.clear_running("nl_screen")
        assert ui_runs.get_running_any("nl_screen") is None
        assert ui_tasks.is_running("nl_screen") is False

        # 第二个进度点应把标记自愈回来
        release.set()
        assert healed.wait(3), "任务应已执行到第二个进度点"
        assert ui_runs.get_running_any("nl_screen") is not None, \
            "仍在执行的任务必须把丢失的标记自愈回来"
        assert ui_tasks.is_running("nl_screen") is True
        hold.set()
        wait_done("nl_screen")


# ----------------------------------------------------------------
# 互斥
# ----------------------------------------------------------------

class TestMutualExclusion:
    def test_second_start_rejected_while_running(self):
        release = threading.Event()

        def job(progress):
            release.wait(3.0)

        assert ui_tasks.start("demo", job) is True
        assert ui_tasks.start("demo", job) is False
        release.set()
        wait_done("demo")

    def test_can_start_again_after_finish(self):
        ui_tasks.start("demo", lambda progress: None)
        wait_done("demo")
        assert ui_tasks.start("demo", lambda progress: None) is True
        wait_done("demo")

    def test_stale_marker_does_not_block_forever(self, monkeypatch):
        """心跳停了的旧标记不该永远挡住新任务"""
        ui_runs.mark_running("demo")
        monkeypatch.setattr(ui_runs, "HEARTBEAT_TIMEOUT_SECONDS", 0)
        assert ui_tasks.start("demo", lambda progress: None) is True
        wait_done("demo")

    def test_stale_finally_does_not_kill_new_run(self):
        """回归：上一轮线程的收尾不能把新一轮的标记清掉"""
        release = threading.Event()

        def job(progress):
            release.wait(3.0)

        ui_tasks.start("demo", job)
        time.sleep(0.05)
        record = ui_runs.get_running_any("demo")
        assert record is not None
        ui_runs.clear_running("demo", run_id="上一轮的 run_id")
        assert ui_runs.get_running_any("demo") == record
        release.set()
        wait_done("demo")

    def test_each_start_gets_distinct_run_id(self):
        seen = []

        def job(progress):
            seen.append(ui_runs.get_running_any("demo")["run_id"])

        ui_tasks.start("demo", job)
        wait_done("demo")
        ui_tasks.start("demo", job)
        wait_done("demo")
        assert len(seen) == 2 and seen[0] != seen[1]

    def test_only_one_worker_runs(self):
        """并发点击：计数证明只跑了一个"""
        counter = {"n": 0}
        lock = threading.Lock()
        release = threading.Event()

        def job(progress):
            with lock:
                counter["n"] += 1
            release.wait(3.0)

        results = [ui_tasks.start("demo", job) for _ in range(5)]
        release.set()
        wait_done("demo")
        assert results.count(True) == 1
        assert counter["n"] == 1


# ----------------------------------------------------------------
# 取消
# ----------------------------------------------------------------

class TestCancel:
    def test_cancel_stops_task(self):
        seen = {"steps": 0}

        def job(progress):
            for i in range(200):
                seen["steps"] = i
                progress({"percent": i / 2.0, "message": f"第 {i} 步"})
                time.sleep(0.01)

        ui_tasks.start("demo", job)
        time.sleep(0.1)
        ui_tasks.request_cancel("demo")
        done = wait_done("demo", timeout=5.0)
        assert done["status"] == "cancelled"
        assert seen["steps"] < 199               # 确实提前退出了
        assert ui_tasks.is_cancelled("demo") is False   # 取消标记被清掉

    def test_cancel_clears_marker(self):
        release = threading.Event()

        def job(progress):
            progress({"percent": 1.0, "message": "跑着"})
            release.wait(3.0)

        ui_tasks.start("demo", job)
        time.sleep(0.05)
        ui_tasks.request_cancel("demo")
        release.set()
        wait_done("demo")
        assert ui_runs.get_running_any("demo") is None

    def test_cancel_before_start_is_ignored(self):
        """上一轮的取消标记不会误伤新任务"""
        ui_tasks.request_cancel("demo")
        ran = {"ok": False}

        def job(progress):
            ran["ok"] = True
            progress({"percent": 50.0, "message": "正常"})

        assert ui_tasks.start("demo", job) is True
        done = wait_done("demo")
        assert ran["ok"] is True
        assert done["status"] == "done"

    def test_cancel_is_cooperative_not_immediate(self):
        """取消只在下一个进度点生效（不打断正在进行的单步）"""
        entered = threading.Event()
        release = threading.Event()

        def job(progress):
            entered.set()
            release.wait(3.0)                    # 单步阻塞期间取消无效
            progress({"percent": 50.0, "message": "第二步"})

        ui_tasks.start("demo", job)
        assert entered.wait(2.0)
        ui_tasks.request_cancel("demo")
        time.sleep(0.05)
        assert ui_tasks.is_running("demo") is True    # 还活着
        release.set()
        assert wait_done("demo")["status"] == "cancelled"


# ----------------------------------------------------------------
# 异常
# ----------------------------------------------------------------

class TestErrors:
    def test_error_recorded_and_marker_cleared(self):
        def job(progress):
            progress({"percent": 10.0, "message": "开始"})
            raise RuntimeError("数据源炸了")

        ui_tasks.start("demo", job)
        done = wait_done("demo")
        assert done["status"] == "error"
        assert "数据源炸了" in done["message"]
        assert ui_runs.get_running_any("demo") is None

    def test_error_keeps_previous_result(self):
        """失败不该把上一次成功的结果抹掉"""
        ui_runs.save_result("demo", {"old": True})
        ui_tasks.start("demo", lambda progress: (_ for _ in ()).throw(ValueError("x")))
        wait_done("demo")
        assert ui_runs.load_result("demo")["payload"] == {"old": True}


# ----------------------------------------------------------------
# 文件细节
# ----------------------------------------------------------------

class TestProgressFile:
    def test_progress_file_is_valid_json(self, tmp_path):
        ui_tasks.start("demo", lambda progress: progress({"percent": 5.0, "message": "中"}))
        wait_done("demo")
        path = os.path.join(str(tmp_path), "progress_demo.json")
        with open(path, encoding="utf-8") as f:
            assert json.load(f)["status"] == "done"

    def test_read_progress_missing_file(self):
        assert ui_tasks.read_progress("nope") is None

    def test_clear_progress(self):
        ui_tasks.start("demo", lambda progress: None)
        wait_done("demo")
        assert ui_tasks.clear_progress("demo") is True
        assert ui_tasks.read_progress("demo") is None

    def test_extra_payload_fields_survive(self):
        """页面靠额外字段做渐进式渲染（如深度分析的分步快照）"""
        def job(progress):
            progress({"percent": 20.0, "message": "节点 A", "snapshot": {"tech": "报告"}})

        ui_tasks.start("demo", job)
        wait_done("demo")
        # done 状态会覆盖进度，但运行中的快照读取路径由下面的轮询测试覆盖
        assert ui_tasks.read_progress("demo")["status"] == "done"

    def test_snapshot_readable_while_running(self):
        release = threading.Event()

        def job(progress):
            progress({"percent": 20.0, "message": "节点 A", "snapshot": {"tech": "报告"}})
            release.wait(3.0)

        ui_tasks.start("demo", job)
        deadline = time.time() + 2
        snap = None
        while time.time() < deadline:
            data = ui_tasks.read_progress("demo") or {}
            snap = data.get("snapshot")
            if snap:
                break
            time.sleep(0.02)
        release.set()
        wait_done("demo")
        assert snap == {"tech": "报告"}

    def test_write_failure_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(ui_runs, "UI_RUNS_DIR", "Z:\\不存在的盘\\子目录")
        assert ui_tasks.write_progress("demo", {"percent": 1.0}) is False

    def test_concurrent_writes_do_not_collide(self, tmp_path):
        """回归：两个线程同时写同一个文件曾导致 os.replace 报 [WinError 5]"""
        errors = []

        def writer(i):
            try:
                for n in range(20):
                    ui_runs.write_json("shared.json", {"writer": i, "n": n})
            except Exception as e:               # noqa: BLE001
                errors.append(repr(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert errors == []
        data = ui_runs.read_json("shared.json")
        assert isinstance(data, dict) and "writer" in data
        # 临时文件没有残留
        leftovers = [n for n in os.listdir(str(tmp_path)) if n.endswith(".tmp")]
        assert leftovers == []

    def test_concurrent_heartbeat_and_progress(self, tmp_path):
        """工作线程刷心跳、页面线程读进度，互不干扰"""
        errors = []
        stop = threading.Event()

        def reader():
            try:
                while not stop.is_set():
                    ui_tasks.read_progress("demo")
                    ui_tasks.is_running("demo")
            except Exception as e:               # noqa: BLE001
                errors.append(repr(e))

        def job(progress):
            for i in range(30):
                progress({"percent": i * 3.0, "message": f"第 {i} 步"})
                time.sleep(0.005)

        ui_tasks.start("demo", job)
        readers = [threading.Thread(target=reader) for _ in range(3)]
        for t in readers:
            t.start()
        done = wait_done("demo")
        stop.set()
        for t in readers:
            t.join(5)
        assert errors == []
        assert done["status"] == "done"
