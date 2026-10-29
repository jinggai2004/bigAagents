"""UI 长任务持久化测试（P1 修复）

覆盖：结果落盘/回填、运行标记与过期判定、孤儿清理、时间格式化、
以及"写入失败不影响主流程"的容错。
"""

import json
import os
import time

import pytest

import core.ui_runs as ur


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(ur, "UI_RUNS_DIR", str(tmp_path / "ui_runs"))
    yield


class TestResultPersistence:
    def test_save_then_load(self):
        assert ur.save_result("screen", {"candidates": [{"股票代码": "600519"}]}) is True
        rec = ur.load_result("screen")
        assert rec["payload"]["candidates"][0]["股票代码"] == "600519"
        assert rec["generated_at"]
        assert ur.age_seconds(rec) < 5

    def test_load_missing_returns_none(self):
        assert ur.load_result("nope") is None

    def test_load_corrupt_file_returns_none(self):
        ur._ensure_dir()
        path = os.path.join(ur.UI_RUNS_DIR, ur.result_filename("bad"))
        with open(path, "w", encoding="utf-8") as f:
            f.write("{不是合法 JSON")
        assert ur.load_result("bad") is None

    def test_overwrite_keeps_latest(self):
        ur.save_result("screen", {"v": 1})
        time.sleep(0.01)
        ur.save_result("screen", {"v": 2})
        assert ur.load_result("screen")["payload"]["v"] == 2

    def test_non_serializable_payload_does_not_raise(self):
        """DataFrame 之类无法序列化的对象必须被容忍（default=str 兜底）"""
        import pandas as pd
        ok = ur.save_result("screen", {"df": pd.DataFrame({"a": [1]})})
        assert ok is True
        assert ur.load_result("screen") is not None

    def test_clear_result(self):
        ur.save_result("screen", {"v": 1})
        assert ur.clear_result("screen") is True
        assert ur.load_result("screen") is None


class TestRunningMarker:
    def test_mark_and_get(self):
        assert ur.mark_running("screen", {"sector": "全市场"}) is True
        rec = ur.get_running("screen")
        assert rec["meta"]["sector"] == "全市场"
        assert ur.age_seconds(rec) < 5

    def test_clear(self):
        ur.mark_running("screen")
        ur.clear_running("screen")
        assert ur.get_running("screen") is None

    def test_get_missing_returns_none(self):
        assert ur.get_running("nope") is None

    def test_stale_marker_ignored(self):
        ur.mark_running("screen")
        data = ur._load_running_raw()
        data["screen"]["started_ts"] = time.time() - 1000
        data["screen"]["heartbeat_ts"] = time.time() - 1000
        ur._atomic_write_json(ur.RUNNING_FILE, data)
        assert ur.get_running("screen", heartbeat_timeout=900) is None
        assert ur.get_running("screen", heartbeat_timeout=7200) is not None

    def test_multiple_tasks_are_independent(self):
        ur.mark_running("screen")
        ur.mark_running("analysis_000001")
        assert set(ur.running_tasks()) == {"screen", "analysis_000001"}
        ur.clear_running("screen")
        assert set(ur.running_tasks()) == {"analysis_000001"}

    def test_clear_stale_running(self):
        ur.mark_running("a")
        ur.mark_running("b")
        data = ur._load_running_raw()
        data["a"]["started_ts"] = time.time() - 5000
        ur._atomic_write_json(ur.RUNNING_FILE, data)
        assert ur.clear_stale_running(900) == 1
        assert set(ur.running_tasks()) == {"b"}


class TestHeartbeatAndLiveness:
    """回归：标记必须能区分"真的在跑"与"已经断了"。

    早期实现只记 started_ts，于是它实际退化成计时器：页面刷新会中断任务
    （Streamlit 的停止在脚本下一次调用 st.* 时生效，而进度条就是 st.* 调用），
    但标记会一直留到 15 分钟"过期"，期间页面持续谎报"仍在运行"。
    """

    def test_mark_records_heartbeat_and_pid(self):
        ur.mark_running("screen")
        rec = ur.get_running_any("screen")
        assert rec["heartbeat_ts"] and rec["pid"] == os.getpid()

    def test_heartbeat_keeps_marker_alive(self):
        ur.mark_running("screen")
        data = ur._load_running_raw()
        data["screen"]["started_ts"] = time.time() - 600     # 开始很久了
        data["screen"]["heartbeat_ts"] = time.time() - 5      # 但心跳很新
        ur._atomic_write_json(ur.RUNNING_FILE, data)
        assert ur.get_running("screen", heartbeat_timeout=90) is not None

    def test_stale_heartbeat_means_interrupted(self):
        ur.mark_running("screen")
        data = ur._load_running_raw()
        data["screen"]["heartbeat_ts"] = time.time() - 300
        ur._atomic_write_json(ur.RUNNING_FILE, data)
        assert ur.get_running("screen", heartbeat_timeout=180) is None
        assert ur.get_running_any("screen") is not None, "UI 靠它区分'已中断'"

    def test_heartbeat_noop_for_unknown_task(self):
        assert ur.heartbeat("nope") is False

    def test_clear_dead_running(self):
        ur.mark_running("a")
        ur.mark_running("b")
        data = ur._load_running_raw()
        data["a"]["heartbeat_ts"] = time.time() - 999
        ur._atomic_write_json(ur.RUNNING_FILE, data)
        assert ur.clear_dead_running() == ["a"]
        assert set(ur.running_tasks()) == {"b"}

    def test_max_age_is_a_backstop(self):
        ur.mark_running("screen")
        data = ur._load_running_raw()
        data["screen"]["started_ts"] = time.time() - ur.HEARTBEAT_MAX_AGE_SECONDS - 10
        data["screen"]["heartbeat_ts"] = time.time()
        ur._atomic_write_json(ur.RUNNING_FILE, data)
        assert ur.get_running("screen", heartbeat_timeout=10 ** 6) is None


class TestTaskGuard:
    def test_clears_marker_on_normal_exit(self):
        with ur.task_guard("screen", {"sector": "全市场"}):
            assert ur.get_running("screen") is not None
        assert ur.get_running_any("screen") is None

    def test_clears_marker_when_body_raises(self):
        """回归：脚本运行被 Streamlit 打断（抛 StopException）时，标记必须被清掉。

        早期实现把 clear_running 写在 finally 里、且排在 progress_bar.empty() 之后，
        后者同样抛异常 → 清标记被跳过 → 标记一直留到"过期"，页面持续谎报运行中。
        """
        class StopException(Exception):
            pass

        with pytest.raises(StopException):
            with ur.task_guard("screen"):
                raise StopException()
        assert ur.get_running_any("screen") is None


class TestRunId:
    """run_id：后台线程的收尾只允许清掉"自己那一轮"的标记"""

    def test_marks_own_run_id(self):
        ur.mark_running("screen", run_id="A")
        assert ur.get_running_any("screen")["run_id"] == "A"

    def test_other_run_id_does_not_clear(self):
        ur.mark_running("screen", run_id="新的一轮")
        assert ur.clear_running("screen", run_id="旧的一轮") is False
        assert ur.get_running_any("screen") is not None, "新一轮的标记不能被旧线程清掉"

    def test_matching_run_id_clears(self):
        ur.mark_running("screen", run_id="A")
        assert ur.clear_running("screen", run_id="A") is True
        assert ur.get_running_any("screen") is None

    def test_no_run_id_still_clears(self):
        ur.mark_running("screen", run_id="A")
        assert ur.clear_running("screen") is True

    def test_clear_missing_is_ok(self):
        assert ur.clear_running("nope", run_id="A") is True


class TestAgeText:
    @pytest.mark.parametrize("seconds, expected", [
        (5, "5 秒"),
        (59, "59 秒"),
        (61, "1 分 1 秒"),
        (150, "2 分 30 秒"),
        (3700, "1 小时 1 分"),
    ])
    def test_formatting(self, seconds, expected):
        assert ur.age_text(seconds) == expected

    def test_unknown_record(self):
        assert ur.age_text({}) == "未知"

    def test_running_record(self):
        ur.mark_running("screen")
        assert ur.age_text(ur.get_running("screen")).endswith("秒")
