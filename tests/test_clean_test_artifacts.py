"""pytest 遗留产物清理器 `scripts/clean_test_artifacts.py`

为什么这些用例重要：

1. 这是个**删除脚本**，它会由 `运行测试.bat` 每次跑测试时自动调用。
   删错东西的代价不可逆，所以每条"不删什么"的边界都要被钉死：
   temp/ 里有调度器的**心跳文件**（`scheduler_heartbeat.json`），
   被误删会让单实例检查以为"没有调度器在跑"，进而启动第二个调度进程。
2. 它必须**按 mtime 而不是按名字**排序。历史遗留的名字毫无规律
   （`pytest_ui1`、`pytest_v3full`、`pytest_full_0929` 混在一起，
   名字顺序与新旧顺序无关），按名字排会删掉最新的一次、留下最旧的。
3. 测试本身**绝不能碰项目真实的 temp/** —— 一律通过 temp_dir / output_dir
   参数指向 tmp_path。这正是本仓库踩过的坑：一个没做隔离的用例删掉了生产库的表。
"""

import os
import time
from pathlib import Path

import pytest

import scripts.clean_test_artifacts as mod


def _mk_dirs(base, names, *, newest_first_age_hours=None):
    """按给定顺序造目录：列表**末尾**的那个最新（mtime 递增）"""
    made = []
    n = len(names)
    for i, name in enumerate(names):
        d = base / name
        d.mkdir()
        (d / "payload.bin").write_bytes(b"0" * 512)
        age_h = (n - i) * 3600                  # 越靠后越新
        t = time.time() - age_h
        os.utime(d, (t, t))
        made.append(d)
    return made


@pytest.fixture
def sandbox(tmp_path):
    """隔离的 temp/ 与 output/，外加两个"绝不能删"的真实场景文件"""
    temp, out = tmp_path / "temp", tmp_path / "output"
    temp.mkdir()
    out.mkdir()
    heartbeat = temp / "scheduler_heartbeat.json"
    heartbeat.write_text('{"pid": 123}')
    not_a_dir = temp / "pytest_notes.txt"        # 同前缀，但是文件
    not_a_dir.write_text("x")
    other = temp / "probe.db"                    # 非 pytest 前缀
    other.write_text("x")
    return {"root": tmp_path, "temp": temp, "out": out,
            "heartbeat": heartbeat, "not_a_dir": not_a_dir, "other": other}


# ===== 保留策略：按 mtime，不按名字 =====

def test_keeps_newest_by_mtime_not_by_name(sandbox):
    """名字顺序与新旧顺序故意错开：按名字排会得出错误答案。

    这里 ui1 最旧、new 最新，但字典序上 ui1 排在 aaa 之后 —— 只要实现
    改成 sorted(name)，保留结果就会变成 aaa/full_0929/new，用例立刻变红。
    """
    names = ["pytest_ui1", "pytest_v3full", "pytest_full_0929",
             "pytest_aaa", "pytest_zzz", "pytest_new"]
    _mk_dirs(sandbox["temp"], names)

    result = mod.clean(keep_dirs=3, keep_logs=2,
                       temp_dir=sandbox["temp"], output_dir=sandbox["out"])

    kept = sorted(p.name for p in sandbox["temp"].iterdir() if p.is_dir())
    assert kept == ["pytest_aaa", "pytest_new", "pytest_zzz"], \
        f"应保留 mtime 最新的 3 个，实际保留 {kept}"
    assert len(result["removed"]) == 3
    assert result["kept_dirs"] == 3


def test_keeps_everything_when_under_the_limit(sandbox):
    _mk_dirs(sandbox["temp"], ["pytest_a", "pytest_b"])
    result = mod.clean(keep_dirs=3, temp_dir=sandbox["temp"], output_dir=sandbox["out"])
    assert result["removed"] == []
    assert result["kept_dirs"] == 2


def test_keep_zero_clears_all_temp_dirs_but_still_spares_runtime_files(sandbox):
    _mk_dirs(sandbox["temp"], ["pytest_a", "pytest_b"])
    result = mod.clean(keep_dirs=0, temp_dir=sandbox["temp"], output_dir=sandbox["out"])
    assert len(result["removed"]) == 2
    assert sandbox["heartbeat"].exists(), "清空临时目录时也不许碰心跳文件"


# ===== 安全边界：什么绝对不能删 =====

def test_never_deletes_runtime_or_non_pytest_entries(sandbox):
    """心跳、非 pytest 前缀文件、同前缀的**文件**都必须在。

    心跳被删 → 调度器单实例检查失效；probe.db 之类虽可从别处重建，
    但清理器的职责只是"清 pytest 临时目录"，越界删除就是 bug。
    """
    _mk_dirs(sandbox["temp"], ["pytest_1", "pytest_2", "pytest_3", "pytest_4"])
    mod.clean(keep_dirs=1, temp_dir=sandbox["temp"], output_dir=sandbox["out"])
    assert sandbox["heartbeat"].exists()
    assert sandbox["not_a_dir"].exists()
    assert sandbox["other"].exists()


def test_only_touches_directories_with_pytest_prefix(sandbox):
    """同前缀但**不是目录**的条目必须留下（历史上 temp/ 落过 pytest_*.txt 日志）。

    同时钉死前缀比较不分大小写：历史遗留的目录名大小写并不统一，
    只匹配小写会让 `PYTEST_xxx` 永久堆积。
    """
    (sandbox["temp"] / "pytest_not_a_dir.log").write_text("x")
    _mk_dirs(sandbox["temp"], ["pytest_x", "pytest_y", "PYTEST_upper"])   # 最后一个最新

    mod.clean(keep_dirs=1, temp_dir=sandbox["temp"], output_dir=sandbox["out"])

    assert (sandbox["temp"] / "pytest_not_a_dir.log").exists(), "同前缀文件不该被删"
    dirs = sorted(p.name for p in sandbox["temp"].iterdir() if p.is_dir())
    assert dirs == ["PYTEST_upper"], f"大小写不该影响清理范围，实际保留 {dirs}"


def test_default_paths_point_at_project_dirs_and_are_overridable():
    """守护：模块默认值指向项目 temp/ 与 output/，但必须可被参数覆盖。

    测试永远传参（见本文件所有用例）——一旦有人删掉 temp_dir 参数，
    调用方就只能操作真实目录，这里立刻变红。
    """
    assert mod.TEMP_DIR.name == "temp"
    assert mod.OUTPUT_DIR.name == "output"
    import inspect
    sig = inspect.signature(mod.clean)
    assert "temp_dir" in sig.parameters and "output_dir" in sig.parameters


# ===== dry-run / 幂等 / 容错 =====

def test_dry_run_reports_but_deletes_nothing(sandbox):
    _mk_dirs(sandbox["temp"], ["pytest_a", "pytest_b", "pytest_c", "pytest_d"])
    result = mod.clean(keep_dirs=1, temp_dir=sandbox["temp"],
                       output_dir=sandbox["out"], dry_run=True)
    assert len(result["removed"]) == 3
    assert result["dry_run"] is True
    assert len([p for p in sandbox["temp"].iterdir() if p.is_dir()]) == 4, \
        "dry-run 不得实际删除"
    # dry-run 也要如实报告"清理后还剩几份"，否则界面/日志会误导
    assert result["kept_dirs"] == 1


def test_second_run_is_noop(sandbox):
    _mk_dirs(sandbox["temp"], ["pytest_a", "pytest_b", "pytest_c", "pytest_d"])
    mod.clean(keep_dirs=2, temp_dir=sandbox["temp"], output_dir=sandbox["out"])
    again = mod.clean(keep_dirs=2, temp_dir=sandbox["temp"], output_dir=sandbox["out"])
    assert again["removed"] == []


def test_missing_directories_are_safe(tmp_path):
    """temp/ 或 output/ 不存在（首次启动、被别人删了）时不能抛异常。"""
    result = mod.clean(temp_dir=tmp_path / "nope", output_dir=tmp_path / "nope2")
    assert result["removed"] == [] and result["failed"] == []
    assert result["kept_dirs"] == 0


def test_delete_failure_is_reported_not_raised(sandbox, monkeypatch):
    """Windows 上目录被占用会 rmtree 失败——清理失败不该让测试跑不起来。"""
    _mk_dirs(sandbox["temp"], ["pytest_a", "pytest_b"])

    def boom(_path):
        raise PermissionError("WinError 5: 拒绝访问")

    monkeypatch.setattr(mod.shutil, "rmtree", boom)
    result = mod.clean(keep_dirs=0, temp_dir=sandbox["temp"], output_dir=sandbox["out"])
    assert result["removed"] == []
    assert len(result["failed"]) == 2
    assert "WinError 5" in result["failed"][0]["error"]


# ===== 日志（output/pytest_*.txt）=====

def test_logs_keep_recent_n_only(sandbox):
    for i in range(5):
        p = sandbox["out"] / f"pytest_log{i}.txt"
        p.write_text("y" * 100)
        t = time.time() - (5 - i) * 3600
        os.utime(p, (t, t))
    # 干扰项：不同前缀的日志不该被卷进来
    keep_me = sandbox["out"] / "other_log.txt"
    keep_me.write_text("z")

    mod.clean(keep_dirs=1, keep_logs=2, temp_dir=sandbox["temp"], output_dir=sandbox["out"])

    left = sorted(p.name for p in sandbox["out"].glob("pytest_*.txt"))
    assert left == ["pytest_log3.txt", "pytest_log4.txt"], left
    assert keep_me.exists()


def test_main_exit_code_is_always_zero(sandbox, monkeypatch, capsys):
    """调用方是 .bat：清理失败也不能中断测试流程，退出码恒为 0。"""
    _mk_dirs(sandbox["temp"], ["pytest_a", "pytest_b"])
    monkeypatch.setattr(mod, "TEMP_DIR", sandbox["temp"])
    monkeypatch.setattr(mod, "OUTPUT_DIR", sandbox["out"])
    monkeypatch.setattr(mod.shutil, "rmtree",
                        lambda _p: (_ for _ in ()).throw(PermissionError("busy")))
    assert mod.main(["--keep", "0"]) == 0
    out = capsys.readouterr().out
    assert "删除失败" in out


def test_main_dry_run_json_shape(sandbox, monkeypatch, capsys):
    import json
    _mk_dirs(sandbox["temp"], ["pytest_a", "pytest_b"])
    monkeypatch.setattr(mod, "TEMP_DIR", sandbox["temp"])
    monkeypatch.setattr(mod, "OUTPUT_DIR", sandbox["out"])
    assert mod.main(["--keep", "0", "--dry-run", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert len(payload["removed"]) == 2
    assert len([p for p in sandbox["temp"].iterdir() if p.is_dir()]) == 2


# ===== 2026-10-24：88 个目录累积 12.95GB 的两个根因 =====
#
# 根因① 前缀只认 "pytest"，手工 `--basetemp=temp\pt_xxx` 留下的目录永不入列
#       （实测 pt_qwen / pt_review_final / pt_review_oc / pt_review_v3 / pt_v2 /
#        pt_wd 共 6 个 / 1.98GB）。
# 根因② 待删列表按"最新在前"返回，`max_seconds` 预算一耗尽就 `continue` 掉其余，
#       于是预算永远花在较新目录上，**最老的那批永远轮不到**。

def test_pt_prefix_dirs_are_cleaned_by_default(sandbox):
    """pt_* 手工临时库必须在默认清理范围内（根因①）。"""
    _mk_dirs(sandbox["temp"], ["pytest_a", "pt_qwen", "pytest_b", "pt_v2"])
    result = mod.clean(keep_dirs=1, temp_dir=sandbox["temp"], output_dir=sandbox["out"])
    left = sorted(p.name for p in sandbox["temp"].iterdir() if p.is_dir())
    assert left == ["pt_v2"], f"pt_* 未纳入清理，实际剩 {left}"
    assert len(result["removed"]) == 3


def test_pt_prefix_is_configurable_for_narrow_runs(sandbox):
    """--prefix pytest 必须能把范围收窄回只清 pytest*。"""
    _mk_dirs(sandbox["temp"], ["pytest_a", "pt_qwen", "pytest_b", "pt_v2"])
    mod.clean(keep_dirs=1, temp_dir=sandbox["temp"], output_dir=sandbox["out"],
              prefixes=("pytest",))
    left = sorted(p.name for p in sandbox["temp"].iterdir() if p.is_dir())
    assert left == ["pt_qwen", "pt_v2", "pytest_b"], left


def test_plan_returns_oldest_first(sandbox):
    """结构性守护：待删列表**最老在前**（根因②）。

    旧实现 `sort(key=_mtime, reverse=True)` 后再切片，返回"最新在前"，
    预算一紧张就只删新目录、老目录永久腐烂。这条断言把顺序钉死。
    """
    _mk_dirs(sandbox["temp"], ["pytest_a", "pytest_b", "pytest_c", "pytest_d"])
    planned = mod.plan_temp_dirs(keep=1, temp_dir=sandbox["temp"])
    assert [p.name for p in planned] == ["pytest_a", "pytest_b", "pytest_c"], \
        "待删列表必须最老在前，否则预算永远花在新目录上"


def test_budget_exhaustion_deletes_oldest_not_newest(sandbox, monkeypatch):
    """行为守护：预算只够删 2 个时，删掉的必须是最老的两个（根因②）。"""
    _mk_dirs(sandbox["temp"], ["pytest_1", "pytest_2", "pytest_3", "pytest_4"])
    clock = {"t": 0.0}
    deleted = []
    real_rmtree = mod.shutil.rmtree
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["t"])

    def slow_rmtree(p):
        clock["t"] += 100          # 每个目录"删"掉 100 秒，模拟 Defender 扫描
        deleted.append(Path(p).name)
        real_rmtree(p)

    monkeypatch.setattr(mod.shutil, "rmtree", slow_rmtree)
    result = mod.clean(keep_dirs=1, max_seconds=150,
                       temp_dir=sandbox["temp"], output_dir=sandbox["out"])

    assert deleted == ["pytest_1", "pytest_2"], f"应优先删最老的，实际删了 {deleted}"
    assert [Path(p).name for p in result["skipped_by_budget"]] == ["pytest_3"]
    # "还剩几个"必须如实：4 个里删了 2 个 ⇒ 剩 2
    assert result["remaining_dirs"] == 2


def test_zero_budget_means_unlimited(sandbox):
    """--max-seconds 0 ⇒ 不设预算，一次删完（手工大扫除用）。"""
    _mk_dirs(sandbox["temp"], ["pytest_a", "pytest_b", "pytest_c", "pytest_d"])
    result = mod.clean(keep_dirs=1, max_seconds=0,
                       temp_dir=sandbox["temp"], output_dir=sandbox["out"])
    assert len(result["removed"]) == 3
    assert result["skipped_by_budget"] == []


def test_remaining_dirs_counts_delete_failures(sandbox, monkeypatch):
    """删失败的不算"已清"，remaining_dirs 必须把它们算回去。"""
    _mk_dirs(sandbox["temp"], ["pytest_a", "pytest_b"])
    monkeypatch.setattr(mod.shutil, "rmtree",
                        lambda _p: (_ for _ in ()).throw(PermissionError("busy")))
    result = mod.clean(keep_dirs=0, temp_dir=sandbox["temp"], output_dir=sandbox["out"])
    assert result["removed"] == []
    assert result["remaining_dirs"] == 2

