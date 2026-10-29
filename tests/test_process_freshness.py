"""长驻进程「代码陈旧」诊断的测试（core/process_freshness.py）

为什么要测这个：本项目**三次**栽在同一件事上 —— 改完代码进程不重启，
新逻辑不执行**且不报错**（调度器加任务 / 只改 job 函数体 / UI 长驻进程）。
前两次靠"记得重启"防，第三次证明纪律不管用，所以把它变成了可查的事实。
这份测试钉住的是**判据本身**（进程怎么认、陈旧怎么判、判不了怎么表述），
真实进程的启动时间没法在测试里造假，所以把逻辑抽成纯函数来测。
"""

from core import process_freshness as pf
import pytest


class TestProcessMatching:
    def test_windows_backslash_path_is_matched(self):
        """⚠️ 首版用 `"ui/app.py"` 去匹配 Windows 命令行（反斜杠）→ UI 进程静默漏检

        漏检的后果不是"少一条信息"：报告会显示"（未发现运行中的 UI / 调度进程）"，
        让用户以为没有陈旧进程 —— 比不报更误导。
        """
        cmd = (r"C:\Users\OMEN\...\python.exe -m streamlit run "
               r"ui\app.py --server.port 8501")
        assert pf.match_kind(cmd) == "UI (streamlit)"

    def test_scheduler_is_matched(self):
        cmd = r"C:\...\python.exe scripts\run_scheduler.py"
        assert pf.match_kind(cmd) == "调度器"

    def test_packaged_scheduler_is_matched(self):
        """⚠️ 打包后是 `bigAagents.exe --scheduler`（2026-10-27 实测漏检）

        两个坑叠在一起：
        ① `scan_processes` 只放行 python/streamlit 的**进程名** ⇒ exe 在读命令行
           之前就被丢掉；
        ② 就算读了命令行，路径里是大写 `bigAagents`，而旧版 `match_kind` 不转小写
           ⇒ `"bigaagents" in cmd` 恒为 False。

        后果是最要命的那一种：exe 跑着旧代码，自检却报"未发现进程"，看起来是绿的。
        """
        cmd = r"D:\桌面\bigAagents\dist\bigAagents\bigAagents.exe --scheduler"
        assert pf.match_kind(cmd) == "调度器"

    def test_packaged_panel_is_matched_but_not_as_scheduler(self):
        """裸 exe（无参数）是面板；不能因为它也含 `bigaagents` 就当成调度器"""
        cmd = r"D:\桌面\bigAagents\dist\bigAagents\bigAagents.exe"
        assert pf.match_kind(cmd) == "面板 (exe)"

    def test_unrelated_process_is_not_matched(self):
        assert pf.match_kind(r"C:\...\python.exe -c import sys") is None
        assert pf.match_kind("") is None

    def test_project_path_alone_is_not_enough(self):
        """⚠️ 路径里带 `bigAagents` 的无关进程不能被认成长驻进程

        实测踩到：`python -m http.server 8770 --directory D:/桌面/bigAagents/companion`
        被"面板"规则命中，报告里凭空多出一条"面板跑的是旧代码"。
        判据必须是**可执行文件名** `bigaagents.exe`，不是目录名。
        """
        cmd = (r"C:\...\python.exe -m http.server 8770 "
               r"--directory D:/桌面/bigAagents/companion")
        assert pf.match_kind(cmd) is None


class TestScanFilter:
    """`scan_processes` 的**名字过滤**才是 exe 漏检的真正现场。

    只测 `match_kind` 会漏掉它：匹配规则写对了，进程在更早的一层就被
    `continue` 掉了 —— "函数对"不等于"接线对"。
    """

    def test_exe_named_process_survives_the_name_filter(self, monkeypatch):
        psutil = pytest.importorskip("psutil")

        class _P:
            def __init__(self, info):
                self.info = info

        fake = [
            _P({"pid": 1, "name": "bigAagents.exe", "create_time": 1000.0,
                "cmdline": [r"D:\...\bigAagents.exe", "--scheduler"]}),
            # 无关进程必须仍然被挡住，否则新放行的名字会把整机进程都读进来
            _P({"pid": 2, "name": "notepad.exe", "create_time": 1000.0,
                "cmdline": ["notepad.exe"]}),
        ]
        monkeypatch.setattr(psutil, "process_iter", lambda *a, **k: fake)

        out = pf.scan_processes()
        assert out["checked"] is True
        assert [p["pid"] for p in out["processes"]] == [1]
        assert out["processes"][0]["kind"] == "调度器"


class TestEvaluate:
    SCAN = {"checked": True, "reason": None, "processes": [
        {"pid": 111, "kind": "UI (streamlit)", "started_ts": 1000.0,
         "started_at": "2026-10-08 09:12:59", "cmdline": "…"},
        {"pid": 222, "kind": "调度器", "started_ts": 9000.0,
         "started_at": "2026-10-09 08:57:22", "cmdline": "…"},
    ]}

    def test_process_started_before_code_change_is_stale(self):
        # 源码在 t=5000 被改过：启动于 1000 的 UI 陈旧，启动于 9000 的调度器不陈旧
        res = pf.evaluate(self.SCAN, newest_ts=5000.0,
                          probe=lambda ts: ["core/agents.py"] if ts < 5000 else [])
        rows = {p["pid"]: p for p in res["processes"]}
        assert rows[111]["stale"] is True
        assert rows[111]["newer_files"] == ["core/agents.py"]
        assert rows[222]["stale"] is False
        assert res["stale"] == ["UI (streamlit)"]

    def test_all_fresh_reports_no_stale(self):
        res = pf.evaluate(self.SCAN, newest_ts=500.0, probe=lambda ts: [])
        assert res["stale"] == []
        assert all(p["stale"] is False for p in res["processes"])

    def test_uncheckable_is_not_reported_as_healthy(self):
        """⚠️「判不了」≠「没问题」：psutil 缺失时必须如实说判不了

        （本项目在心跳清单那件事上已吃过一次亏：把"读不到"当成"一切正常"。）
        """
        res = pf.evaluate({"checked": False, "reason": "psutil 未安装", "processes": []})
        assert res["checked"] is False
        assert "psutil" in res["reason"]
        assert res["stale"] == []
        assert "无法判断" in pf.describe(res)[0]

    def test_no_processes_is_reported_as_unknown_not_ok(self):
        """⚠️「一个进程都没识别到」≠「没问题」

        旧版这里返回 `reason=None` + "（未发现运行中的 UI / 调度进程）"，
        读起来像检查通过。但识别不到的真正常见原因是"没启动"或"以未登记的
        方式在跑"（换入口脚本 / exe 参数没登记）—— 后者正是 exe 漏检那次。
        所以必须保持**警告**口径，并说明这一项只覆盖已识别的进程。
        """
        res = pf.evaluate({"checked": True, "reason": None, "processes": []},
                          newest_ts=1.0, probe=lambda ts: [])
        assert res["checked"] is True
        assert res["reason"], "识别不到进程必须给原因，不能回 None"
        first = pf.describe(res)[0]
        assert "⚠️" in first
        assert "未识别到" in first
        assert "未登记" in first          # 说清"可能是我没认出来"，而不是"没事"
        assert "✅" not in first

    def test_describe_tells_how_to_restart(self):
        """陈旧时必须给出**可执行**的下一步，否则用户不知道该怎么办"""
        res = pf.evaluate(self.SCAN, newest_ts=5000.0,
                          probe=lambda ts: ["core/agents.py"] if ts < 5000 else [])
        text = " ".join(pf.describe(res))
        assert "重启" in text and "重启调度器.bat" in text

    def test_check_runs_on_this_machine(self):
        """真实扫描：结构必须完整（不假设结果，因为取决于当前是否在跑进程）"""
        res = pf.check()
        assert isinstance(res["checked"], bool)
        if res["checked"]:
            assert isinstance(res["processes"], list)
            for p in res["processes"]:
                assert {"pid", "kind", "started_at", "stale"} <= set(p)
        else:
            assert res["reason"], "判不了必须给原因"


class TestNewestSource:
    def test_finds_a_python_file_in_watched_dirs(self):
        ts, rel = pf.newest_source()
        assert ts > 0
        assert rel.endswith(".py")
        assert rel.replace("\\", "/").split("/")[0] in pf.WATCH_DIRS

    def test_watched_dirs_exclude_ui_and_tests(self):
        """⚠️ 监控范围必须排除 `ui/`（页面脚本每次交互都重新执行，改它不用重启）

        把 `ui/` 算进来会导致"每次改页面都报陈旧" → 误报淹没真信号。
        """
        assert "ui" not in pf.WATCH_DIRS
        assert "tests" not in pf.WATCH_DIRS
