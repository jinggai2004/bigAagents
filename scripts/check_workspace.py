# -*- coding: utf-8 -*-
"""工作区并发体检（2026-10-23）。

## 为什么需要它
2026-10-23 实测：同一个项目**两个会话并行开发**，各自在改文件。已经造成的实际损失：

- 17:25 `core/scheduler.py`、17:38 `tests/test_position_risk.py` 在对方跑**全量测试的中途**
  被编辑 ⇒ 5 个 `test_risk_job_*` 瞬时失败，排查半天才发现是"文件被改了"；
- 两边都用 `logs/junit_20261023{e,f,g,h,i}.xml` 这种**按天编号**的文件名，
  谁后跑谁覆盖，事后无法判断某次失败属于哪一版代码；
- `db/store.py` 的 `_SCHEMA` 是**共享契约**：两边各加一张表，靠
  `CREATE TABLE IF NOT EXISTS` 侥幸共存；一旦有人整段覆盖，**丢掉的表不会被重建**
  （`init_store` 的 schema 版本号是假安全，指纹一变就"已是最新"）。

## 用法
```bash
python scripts/check_workspace.py                 # 看最近 15 分钟被改过的受管文件
python scripts/check_workspace.py --minutes 5     # 只看最近 5 分钟
python scripts/check_workspace.py --path core/evening_analysis.py   # 改之前先探一下这个文件
```

退出码：发现"热"文件 ⇒ 1（可用在脚本/钩子里），干净 ⇒ 0。

⚠️ 它只能告诉你"有人在动"，不能阻止写入。**约定仍然是第一道闸**：
改共享契约文件前先跑一次；长跑全量测试期间不改源码。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

_logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 受管目录：只报**会被运行中的进程加载**的东西（文档/日志/临时文件不报，避免噪声）
_WATCH_DIRS = ("core", "data", "db", "ui", "scripts", "config", "tests", "llm")
_EXTS = (".py", ".json", ".bat")
_SKIP_PARTS = ("__pycache__", ".pytest_cache")

# 共享契约文件：改这些要格外小心（多会话最容易在这里互相覆盖）
_SHARED_CONTRACTS = {
    "db/store.py": "建表 schema / 存储层（两条线都会加表）",
    "core/scheduler.py": "JOBS 任务清单 / 各 job 入口",
    "core/stage_health.py": "阶段依赖表",
    "scripts/run_scheduler.py": "命令行契约",
    "config/ai_personas.json": "人格定义（联赛）",
    "config/llm_feature_models.json": "模型路由",
    "config/settings.py": "全局配置",
}


def _iter_files():
    for d in _WATCH_DIRS:
        root = os.path.join(PROJECT_ROOT, d)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [x for x in dirnames if x not in _SKIP_PARTS]
            for fn in filenames:
                if fn.endswith(_EXTS):
                    yield os.path.join(dirpath, fn)


def recent(minutes: float) -> list[tuple[float, str]]:
    """最近 `minutes` 分钟内被修改过的受管文件（按时间倒序）。"""
    cutoff = time.time() - minutes * 60
    out = []
    for p in _iter_files():
        try:
            m = os.path.getmtime(p)
        except OSError as e:            # 文件在扫描途中被删/被占用：跳过即可，但要留痕
            _logger.debug("[体检] 读取 mtime 失败，跳过 %s: %s", p, e)
            continue
        if m >= cutoff:
            out.append((m, os.path.relpath(p, PROJECT_ROOT).replace("\\", "/")))
    return sorted(out, reverse=True)


def scheduler_started_at() -> tuple[float | None, int | None]:
    """常驻调度器启动时间（取心跳文件的进程 pid；取不到时间就给 None）。"""
    hb = os.path.join(PROJECT_ROOT, "temp", "scheduler_heartbeat.json")
    try:
        with open(hb, encoding="utf-8") as f:
            data = json.load(f)
        pid = int(data.get("pid") or 0)
        # ⚠️⚠️ 必须用**进程启动时刻**，不能用心跳文件的 mtime：
        #   心跳每 60s 重写一次，mtime 永远比任何源码新 ⇒
        #   "改了但未生效"的判断会**恒为空**（2026-10-23 自己踩了一次，
        #   第一次试跑输出了"✓ 无（进程跑的就是最新代码）"而实际刚改过 scheduler）。
        started = data.get("started_at")
        if started:
            return float(started), pid
        # 旧格式心跳没有该字段 ⇒ 退回"文件创建时间不可得"，只能返回 None 让调用方明说不知道
        return None, pid
    except (OSError, ValueError) as e:
        _logger.debug("[体检] 心跳文件不可读（视为无调度器）: %s", e)
        return None, None


def _fmt_age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} 秒前"
    if seconds < 5400:
        return f"{seconds / 60:.0f} 分钟前"
    return f"{seconds / 3600:.1f} 小时前"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="工作区并发体检")
    ap.add_argument("--minutes", type=float, default=15,
                    help="时间窗口（分钟），默认 15")
    ap.add_argument("--path", help="只检查这一个文件（相对项目根）")
    args = ap.parse_args(argv)

    now = time.time()
    if args.path:
        p = os.path.join(PROJECT_ROOT, args.path)
        if not os.path.exists(p):
            print(f"✗ 文件不存在：{args.path}")
            return 2
        age = now - os.path.getmtime(p)
        hot = age <= args.minutes * 60
        note = _SHARED_CONTRACTS.get(args.path.replace("\\", "/"))
        print(f"{'⚠️ ' if hot else '✅ '}{args.path} 最后修改：{_fmt_age(age)}")
        if note:
            print(f"   共享契约文件：{note}")
        if hot:
            print("   ⇒ 另一个会话可能正在编辑它。**先确认对方改完**，"
                  "否则你的整文件写入会把对方的改动覆盖掉。")
            print("   ⇒ 若必须现在改：先 `git diff -- 该文件` 看清现状，改完立刻跑相关用例。")
            return 1
        print("   ⇒ 窗口内无人改动，可以安全编辑。")
        return 0

    rows = recent(args.minutes)
    started, pid = scheduler_started_at()
    print(f"窗口：最近 {args.minutes:g} 分钟｜受管目录：{', '.join(_WATCH_DIRS)}")
    if started:
        print(f"常驻调度器：pid={pid}，心跳 {_fmt_age(now - started)}")
    print("-" * 62)
    if not rows:
        print("✅ 没有文件在窗口内被修改。")
        return 0

    need_restart = []
    for m, rel in rows:
        mark = "🔒" if rel in _SHARED_CONTRACTS else "  "
        print(f"{mark} {_fmt_age(now - m):>10}  {rel}")
        if started and m > started and rel.split("/")[0] in ("core", "data", "db", "llm", "config"):
            need_restart.append(rel)

    hot_shared = [rel for _m, rel in rows if rel in _SHARED_CONTRACTS]
    print("-" * 62)
    if hot_shared:
        print("⚠️ 窗口内有人改过**共享契约文件**：" + "、".join(hot_shared))
        print("   ⇒ 先看 `git diff -- 这些文件`，确认没被整段覆盖（尤其 db/store.py 的建表段）。")
    # 「改了但没生效」始终单独成段（2026-10-23 审查建议）：
    # 启动时间是被监控文件里最容易被忽略的一维——"我明明改了"和"进程里还是旧的"
    # 之间只差一个重启，而这件事不该靠人记。
    print("━━ 改动是否已生效（对照调度器启动时间）━━")
    if not started:
        print("  ？ 读不到调度器心跳 ⇒ 无法判断；先确认它是否在跑。")
    elif need_restart:
        print(f"  ✗ {len(need_restart)} 个文件改了但**未生效**（需要重启调度器）：")
        print("    " + "、".join(need_restart[:10]) + ("…" if len(need_restart) > 10 else ""))
    else:
        print("  ✓ 无（进程跑的就是最新代码）")
    print("\n提示：🔒 = 共享契约文件。跑全量测试期间不要改源码；"
          "junit 文件名请带会话标识，别只用日期。")
    return 1 if hot_shared else 0


if __name__ == "__main__":
    sys.exit(main())
