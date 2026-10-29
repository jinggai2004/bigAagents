"""调度缺口检测：把"定时任务到底有没有真跑过"变成可查的事实。

## 为什么需要

项目最大的一类事故不是"任务报错"，而是**任务根本没跑、且不报错**：

| 时间 | 现象 | 根因 |
|---|---|---|
| 2026-10-12 | 13/14 时盯盘整小时 0 次、13:05/14:05 新闻采集完全缺失 | 机器进入新式待机 2 小时 |
| 2026-10-14 | `盘中异动预警` **13:34 → 15:04 缺口 90 分钟** | 同上（Kernel-Power 506 13:32:54 / 507 15:03:14） |

`misfire_grace_time=3600 + coalesce=True` 的语义是"**迟到的能补跑、超时的被静默丢弃**"，
所以漏跑时日志里**连一行都没有** —— 这是最隐蔽的失效形态，只能靠"数数量"发现。

此前 `MEMORY.md` 记的是**人工方法**："按小时统计 `Running job` 条数，缺口即被冻结时段"。
本模块把这套方法自动化：**直接从 `JOBS` 的 cron 自己枚举"今天该跑的分钟"，与日志实际比对**。

## 为什么要从 cron 反推、而不是硬编码时段

如果写成"09:00-12:00 与 13:00-16:00 应每分钟一次"，那么以后有人改了
`intraday_watch` 的 cron（或加了新任务），检测器就会**用自己的旧认知去判断**，
于是要么误报、要么漏报，而且**不会有人发现检测器已经过时**。

从 `core.scheduler.JOBS` 的 `cron_kwargs` 反推，则**调度与检测共用同一份定义**：
改 cron = 检测口径同步变，不可能漂移。这与项目"阈值唯一来源"的一贯做法一致。

## 判据

- 按当天 `day_of_week` 过滤（cron 只排周一至周五）→ 周末不报缺口。
- 只统计**调度器启动之后**的应跑时刻（启动前没跑是正常的，不是缺口）。
- **密集任务**（当天应跑 ≥ 10 次，如每分钟的盯盘）：逐分钟比对，缺的分钟聚成"缺口段"。
- **稀疏任务**（如 15:35 回填）：应跑时刻在 `[时刻, 时刻 + 宽容度]` 内有运行即算covered。
  宽容度要 ≥ `misfire_grace_time` 的补跑延迟，否则"补跑成功"会被误报成"漏跑"。
"""
from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta

_logger = logging.getLogger(__name__)

# 数据根必须**运行时**解析：打包后 `__file__` 位于只读包 `_internal/` 里，
# 用它推导会指向空目录 ⇒ 读到空数据（2026-10-25 实测：双击 exe 后
# 选股/复盘/研究全 missing、交易台变空账户、系统页看不到调度器）。
# 规则见 `config/paths.data_root()`；源码运行结果与原先一致。
from config.paths import data_root as _paths_data_root  # noqa: E402
_PROJECT_ROOT = _paths_data_root()

# 密集/稀疏任务的分界：当天应跑次数 ≥ 该值即按"逐分钟比对"处理
_DENSE_THRESHOLD = 10
# 稀疏任务的补跑宽容度（分钟）。必须 ≥ scheduler 的 misfire_grace_time(3600s) 的
# 实际补跑延迟：进程恢复后 APScheduler 会立刻补跑，实测延迟 < 5 分钟。
# 取 20 分钟足以覆盖"进程刚恢复就补跑"，又不会把"真漏了"放过。
_SPARSE_GRACE_MINUTES = 20
# 单次缺口小于该值不报（例如某分钟任务耗时略超 60s 导致偶发跳一分钟）
_MIN_GAP_REPORT = 3

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}):\d{2}")
# ⚠️ 名字要截到 " (trigger:" 之前："Running job" 后面跟的是
#    `"盘前选股 (trigger: cron[...], next run at: 2026-10-15 09:25:00 CST)"`
#    —— 直接取引号内容会把 trigger 与**下一次运行日期**一起吞进来。
_RUN_RE = re.compile(r'Running job\s+"([^"]+?)\s+\(trigger:')


def log_path_for(date: datetime | None = None) -> str:
    d = date or datetime.now()
    return os.path.join(_PROJECT_ROOT, "logs", f"scheduler_{d:%Y%m%d}.log")


def _expected_minutes(cron_kwargs: dict, day: datetime) -> list[datetime]:
    """用与调度器**同一个** CronTrigger 枚举当天的应跑时刻。

    ⚠️ **时区**：CronTrigger 必须带 tz（否则 APScheduler 报错），而它返回的是
    **带 tzinfo** 的 datetime；日志时间戳是**本地朴素**时间。两者直接比较会抛
    `TypeError: can't compare offset-naive and offset-aware datetimes`。
    所以内部统一在 `Asia/Shanghai` 上算，**出口剥掉 tzinfo**，与日志口径一致。
    """
    from apscheduler.triggers.cron import CronTrigger
    tz = _TZ()
    trig = CronTrigger(day_of_week="mon-fri", timezone=tz, **cron_kwargs)
    out: list[datetime] = []
    cur = datetime(day.year, day.month, day.day, tzinfo=tz)
    end = cur + timedelta(days=1)
    nxt = trig.get_next_fire_time(None, cur)
    guard = 0
    while nxt is not None and nxt < end:
        out.append(nxt.astimezone(tz).replace(tzinfo=None))
        guard += 1
        if guard > 5000:            # 防御：cron 写错时不至于死循环
            _logger.warning("[缺口检测] %s 枚举次数异常，已截断", cron_kwargs)
            break
        nxt = trig.get_next_fire_time(nxt, nxt + timedelta(seconds=1))
    return out


def _TZ():
    try:
        from zoneinfo import ZoneInfo
    except ImportError:                             # pragma: no cover
        from backports.zoneinfo import ZoneInfo     # type: ignore
    return ZoneInfo("Asia/Shanghai")


def _parse_log(path: str) -> dict:
    """从日志解析出：首个时间戳（≈启动时刻）与各任务的运行分钟集合。"""
    started = None
    runs: dict[str, set] = defaultdict(set)
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            m = _TS_RE.match(ln)
            if not m:
                continue
            cur = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M")
            if started is None:
                started = cur
            mr = _RUN_RE.search(ln)
            if mr:
                runs[mr.group(1)].add(cur)
    return {"started": started, "runs": runs}


def _group_gaps(missing: list[datetime]) -> list[dict]:
    """把缺失分钟聚成连续段"""
    if not missing:
        return []
    missing = sorted(missing)
    segs, start, prev = [], missing[0], missing[0]
    for t in missing[1:]:
        if (t - prev) <= timedelta(minutes=1):
            prev = t
            continue
        segs.append((start, prev))
        start = prev = t
    segs.append((start, prev))
    return [{"start": a.strftime("%H:%M"), "end": b.strftime("%H:%M"),
             "minutes": int((b - a).total_seconds() // 60) + 1} for a, b in segs]


def analyse(date: datetime | None = None, log_path: str | None = None,
            store=None, now: datetime | None = None) -> dict:
    """检测指定日期的调度缺口。

    Args:
        date: 目标日期（默认今天）
        log_path: 覆盖日志路径（测试用）
        store: 覆盖 `core.scheduler` 模块（测试用，需暴露 `JOBS`）
        now: "当前时刻"（默认系统时间）；只检查它之前的应跑时刻

    Returns:
        `{"available": bool, "reason": str, "date":…, "started":…, "jobs": {名: {...}},
          "max_gap_minutes": int, "warnings": [...]}`
        **永远不抛异常** —— 它自己失败不该拖垮健康检查。
    """
    try:
        return _analyse(date, log_path, store, now)
    except Exception as e:                          # noqa: BLE001
        _logger.warning("[缺口检测] 分析失败: %s", e)
        return {"available": False, "reason": f"{type(e).__name__}: {e}",
                "date": (date or datetime.now()).strftime("%Y-%m-%d"),
                "started": None, "jobs": {}, "max_gap_minutes": 0, "warnings": []}


def _analyse(date, log_path, store, now) -> dict:
    day = date or datetime.now()
    path = log_path or log_path_for(day)
    if not os.path.exists(path):
        return {"available": False, "reason": f"当日日志不存在: {os.path.basename(path)}",
                "date": day.strftime("%Y-%m-%d"), "jobs": {},
                "max_gap_minutes": 0, "warnings": []}

    mod = store
    if mod is None:
        from core import scheduler as mod            # noqa: PLC0415

    info = _parse_log(path)
    started = info["started"]
    runs = info["runs"]
    now = now or datetime.now()

    # 周末（或 cron 未覆盖的星期）→ 本就不该跑，不报缺口
    if day.weekday() >= 5:
        return {"available": True, "reason": "非工作日（cron 只排周一至周五），不检测缺口",
                "date": day.strftime("%Y-%m-%d"), "started": None, "jobs": {},
                "max_gap_minutes": 0, "warnings": []}

    jobs_out: dict = {}
    warnings: list[str] = []

    for _jid, cron_kwargs, _mode, desc in getattr(mod, "JOBS", ()):
        expected = _expected_minutes(cron_kwargs, day)
        if started:
            expected = [t for t in expected if t >= started]
        expected = [t for t in expected if t <= now]
        if not expected:
            continue
        actual = runs.get(desc, set())
        dense = len(expected) >= _DENSE_THRESHOLD

        if dense:
            missing = [t for t in expected if t not in actual]
            gaps = [g for g in _group_gaps(missing) if g["minutes"] >= _MIN_GAP_REPORT]
            jobs_out[desc] = {"dense": True, "expected": len(expected),
                              "actual": len(actual & set(expected)),
                              "missed": len(missing), "gaps": gaps}
        else:
            # 稀疏任务：每个应跑时刻在 [时刻, 时刻+宽容度] 内有运行即算 covered
            missed, late = [], []
            for t in expected:
                if any(t <= a <= t + timedelta(minutes=_SPARSE_GRACE_MINUTES) for a in actual):
                    if not any(a == t for a in actual):
                        late.append(t.strftime("%H:%M"))
                    continue
                missed.append(t)
            jobs_out[desc] = {"dense": False, "expected": len(expected),
                              "actual": len(expected) - len(missed),
                              "missed": len(missed),
                              "gaps": _group_gaps([m for m in missed]),
                              "late": late}
            for t in missed:
                warnings.append(f"今日「{desc}」未执行（应在 {t:%H:%M}）。")

    max_gap = 0
    for desc, j in jobs_out.items():
        if j["dense"]:
            for g in j["gaps"]:
                max_gap = max(max_gap, g["minutes"])
                warnings.append(
                    f"今日「{desc}」在 {g['start']}–{g['end']} 连续漏跑 {g['minutes']} 分钟"
                    "（进程被挂起或机器进入睡眠的典型特征）。")

    return {"available": True, "reason": "ok",
            "date": day.strftime("%Y-%m-%d"),
            "started": started.strftime("%H:%M:%S") if started else None,
            "jobs": jobs_out, "max_gap_minutes": max_gap, "warnings": warnings}


def describe(result: dict = None) -> list[str]:
    """转成可直接展示的文本行。"""
    r = result if result is not None else analyse()
    if not r.get("available"):
        return [f"调度缺口：无法检测（{r.get('reason')}）"]
    if not r.get("jobs"):
        return [f"调度缺口：{r.get('reason')}"]
    lines = [f"调度缺口（{r['date']}，调度器 {r.get('started') or '未知'} 启动）："]
    for desc, j in r["jobs"].items():
        flag = "⚠️" if j["missed"] else "✅"
        extra = ""
        if j.get("gaps"):
            extra = "　缺口 " + "、".join(f"{g['start']}–{g['end']}({g['minutes']}min)"
                                          for g in j["gaps"])
        lines.append(f"  {flag} {desc}: 应跑 {j['expected']} / 实跑 {j['actual']}{extra}")
    return lines
