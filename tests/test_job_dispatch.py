# -*- coding: utf-8 -*-
"""任务分派的**接线守护**（2026-10-23）。

## 为什么需要它
`JOBS` 里写了 `(job_id, cron, mode, desc)`，但"注册成功"不等于"跑得起来"：

- 2026-10-19 `run_evening_analysis_job` 用了 `os.getpid()` 却没 `import os`，
  崩在 try 之外 ⇒ 从接线起**一次没跑成**；
- 2026-10-23 新增 `sentiment_ref`，注册成功、心跳里也有，但**一次没被执行过**；
- `run_job` 末尾有个**默认分支**：任何没被 `mode == "x"` 命中的 mode 都会去跑
  `DailyPipeline`。也就是说**拼错一个 mode 不会报错，只会静默跑全量流水线**。

前两条靠"跑一次"才能发现；第三条只能靠**静态分派检查**——
在这里是：`JOBS` 用到的每个 mode，必须要么被 `mode == "..."` 显式分派，
要么在 `PIPELINE_MODES` 白名单里（明确认可它走默认流水线）。
新增 mode 忘了接线 ⇒ 测试立刻红。

⚠️ 用 AST 而不是"真跑一遍"：真跑会联网/花钱，而这类错在**收集期**就该红。
"""
import ast
import os

import pytest

from core.scheduler import JOBS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCHEDULER = os.path.join(_ROOT, "core", "scheduler.py")

# 明确认可"走 DailyPipeline 默认分支"的 mode（选股/盘后全流程本身就是流水线）。
PIPELINE_MODES = {"screen", "rule"}


def _run_job_ast() -> ast.FunctionDef:
    with open(_SCHEDULER, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "run_job":
            return n
    raise AssertionError("找不到 run_job（分派检查会变成恒真的空断言）")


def _dispatched_modes() -> set:
    """`if mode == "xxx":` 里出现过的 mode 字面量。"""
    out = set()
    for n in ast.walk(_run_job_ast()):
        if (isinstance(n, ast.Compare) and isinstance(n.left, ast.Name)
                and n.left.id == "mode"):
            for c in n.comparators:
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    out.add(c.value)
    return out


class TestJobDispatch:
    def test_every_job_mode_is_dispatchable(self):
        modes = {m for _j, _k, m, _d in JOBS}
        missing = sorted(modes - _dispatched_modes() - PIPELINE_MODES)
        assert not missing, (
            f"这些 mode 既没有 `mode == \"...\"` 分支，也不在 PIPELINE_MODES 里：{missing}\n"
            f"⇒ 它们会落进 run_job 末尾的**默认分支**，静默跑 DailyPipeline（选股全流程）。\n"
            f"请补一个分支（或显式登记为流水线 mode）。")

    def test_pipeline_fallback_really_exists(self):
        """白名单的前提是"默认分支确实接得住"——否则白名单就是遮羞布。"""
        src = ast.get_source_segment(
            open(_SCHEDULER, encoding="utf-8").read(), _run_job_ast()) or ""
        assert "DailyPipeline()" in src, \
            "run_job 里没有 DailyPipeline 默认分支 ⇒ PIPELINE_MODES 白名单不成立"

    def test_dispatch_guard_has_discriminance(self, monkeypatch):
        """鉴别力自检：塞一个没接线的 mode，必须被抓出来。"""
        fake = list(JOBS) + [("zzz_fake_job", dict(hour=3, minute=0),
                              "zzz_not_wired", "故意没接线的任务")]
        monkeypatch.setattr("tests.test_job_dispatch.JOBS", fake)
        modes = {m for _j, _k, m, _d in fake}
        missing = sorted(modes - _dispatched_modes() - PIPELINE_MODES)
        assert missing == ["zzz_not_wired"]

    def test_ai_round_forwards_no_budget(self):
        """`--no-budget` 必须真的传到 run_ai_round_job（否则补跑照样吃额度）。"""
        src = ast.get_source_segment(
            open(_SCHEDULER, encoding="utf-8").read(), _run_job_ast()) or ""
        assert "consume_budget=not no_budget" in src
