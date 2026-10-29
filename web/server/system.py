# -*- coding: utf-8 -*-
"""`/api/system/*` —— 系统页三块（P5）：健康 / 任务表 / LLM 成本。

## 判活口径（项目铁律，照抄 `health.py` 的结论）
心跳文件 mtime **不是**判据；`started_at + PID` 才是。心跳陈旧但进程在 ⇒
只告警、不杀、不报"未运行"。

## 任务表为什么读 `JOBS` 常量而不是问调度器
API 进程与调度器进程是**两个进程**：这里拿不到 `BlockingScheduler` 实例，
`next_run_times()` 依赖它。假装能算下次运行时间只能是猜。
所以给的是**静态定义**（`JOBS`：18 项、cron 字段、描述）+ **实际运行态**
（`ui_runs` 里的活标记与进度文件）—— 两者都是确定的，没有一列是编的。
"""
from __future__ import annotations

import logging

from web.server import envelope, health

_logger = logging.getLogger(__name__)


def _offline() -> bool:
    try:
        from data.offline_gate import is_offline
        return bool(is_offline())
    except Exception as e:                     # noqa: BLE001
        _logger.debug("[system] 离线闸门不可用，按在线处理: %s: %s", type(e).__name__, e)
        return False


# ---------------------------------------------------------------- 健康
def status() -> dict:
    """调度器 + 进程 + 磁盘 + 离线态。`health.load_health` 是唯一心跳来源。"""
    hb = health.load_health()
    data = dict(hb.get("data") or {})
    data.update({
        "offline": _offline(),
        "data_root": _data_root(),
        "disk_free_gb": _disk_free_gb(),
        "pid_self": _pid(),
    })
    return envelope.ok(data, as_of=hb.get("as_of"))


def _data_root() -> str:
    from web.server import settings
    return settings.data_root()


def _pid():
    import os
    return os.getpid()


def _disk_free_gb():
    import shutil
    try:
        return round(shutil.disk_usage(_data_root()).free / (1024 ** 3), 1)
    except OSError as e:
        _logger.warning("[system] 磁盘空间不可读: %s", e)
        return None


# ---------------------------------------------------------------- 任务表
def tasks() -> dict:
    """18 个定时任务的定义 + 当前运行态。"""
    from core.scheduler import JOBS
    import core.ui_runs as ui_runs
    import core.ui_tasks as ui_tasks

    rows = []
    for job_id, cron, mode, desc in JOBS:
        prog = ui_tasks.read_progress(job_id)
        rec = ui_runs.get_running_any(job_id)
        alive = ui_runs.get_running(job_id) is not None
        rows.append({
            "job_id": job_id,
            "time": _cron_label(cron),
            "mode": mode,
            "desc": desc,
            "running": alive,
            "last_run_at": (rec or {}).get("started_at") if rec else None,
            "progress": _slim(prog),
        })

    # 网页触发的一次性任务也列出来（task 名带 `web_` 前缀，与定时任务不重名）
    web_rows = []
    for task, rec in (ui_runs.running_tasks() or {}).items():
        if not task.startswith("web_"):
            continue
        web_rows.append({"task": task, "started_at": (rec or {}).get("started_at"),
                         "progress": _slim(ui_tasks.read_progress(task))})

    return envelope.ok({
        "jobs": rows,
        "total": len(rows),
        "running": sum(1 for r in rows if r["running"]),
        "web_tasks": web_rows,
    })


def _slim(prog):
    if not isinstance(prog, dict):
        return None
    return {"status": prog.get("status"), "percent": prog.get("percent"),
            "message": prog.get("message"), "elapsed": prog.get("elapsed"),
            "updated_at": prog.get("updated_at")}


def _cron_label(cron: dict) -> str:
    """cron 字段 → 人能读的时间标签。

    ⚠️ 不猜具体值：`hour` 缺失 ⇒ 真的是"每小时"；`minute == "*"` ⇒ 每分钟。
       编一个"下次运行时间"只会与时区/交易日闸门合起来骗人。
    """
    hour = cron.get("hour")
    minute = cron.get("minute")
    if isinstance(minute, int) and hour is None:
        return f"每小时 :{minute:02d}"
    if minute == "*":
        return f"{hour} 每分钟" if isinstance(hour, str) else "每分钟"
    if isinstance(hour, int) and isinstance(minute, int):
        return f"{hour:02d}:{minute:02d}"
    return f"{hour or '*'}:{minute if minute is not None else '*'}"


# ---------------------------------------------------------------- 模型与厂商
def models() -> dict:
    """厂商接入表 + 功能模型分配（**不含任何密钥**、不做连通性探测）。

    ## 为什么状态列是"已配置/未配置"而不是"已连通 412ms"
    连通性要真发一次请求（每家一个、还可能 30s 超时）。系统页是每 10s 会刷的页面，
    把它做成"开页面就外呼"既慢又会把限频打满。
    `provider_ready()` 只判断"密钥有没有配"，那是**能确定**的事实；
    "连不连得通"由实际调用去暴露，不在这里假称已测过。
    """
    from config import providers as P
    from config import model_routing as R

    try:
        provs = P.get_providers()
        default_id = P.default_provider_id()
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[system] 厂商注册表读取失败: %s: %s", type(e).__name__, e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")

    rows = []
    for p in provs:
        base = p.get("base_url") or ""
        rows.append({
            "id": p["id"], "name": p["name"],
            # 只给主机名：完整 base_url 可能带内网路径，且对本机排查没额外信息
            "endpoint": _host(base),
            "models": p.get("models") or [],
            "ready": bool(P.provider_ready(p)),
            "enabled": bool(p.get("enabled", True)),
            "default": p["id"] == default_id,
        })

    try:
        usage = {u["feature"]: u for u in R.feature_usage(hours=24)}
        mapping = R.get_feature_models()
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[system] 功能模型表读取失败: %s: %s", type(e).__name__, e)
        usage, mapping = {}, {}

    # ⚠️ 功能表必须按**登记全集**列，不能只列"近 24h 有调用"的那些。
    #    原实现拿 `feature_usage()` 做主体，于是没跑过的功能在页面**根本不出现**，
    #    想改它的模型连入口都没有——这才是"配置页改不了模型"的第二层原因
    #    （第一层是页面只有只读表格）。usage 只作为附加的调用数列。
    #    `usage` 就是上面那次查询的结果，不再重查一遍（同一个请求读两次库没意义）。
    configured = dict(mapping or {}) if isinstance(mapping, dict) else {}

    users = []
    known = set()
    for feat, meta in R.FEATURES.items():
        u = usage.get(feat) or {}
        known.add(feat)
        users.append({
            "feature": feat,
            "label": meta.get("label") or feat,
            "group": meta.get("group") or "",
            "desc": meta.get("desc") or "",
            "reserved": bool(meta.get("reserved")),
            # override = 页面下拉框该显示什么：空 ⇒ 跟随默认（选项里的"跟随默认"）
            "override": configured.get(feat) or "",
            # 实际生效的模型（含 tier 兜底），和"填了什么"是两件事
            "model": R.resolve_model(feat),
            "calls": u.get("count") or 0,
            "failed": u.get("failed") or 0,
            "tokens": u.get("tokens") or 0,
        })
    # 有调用但**没登记**的功能（历史遗留/新加的临时键）：照样列出来，
    # 否则它在页面上隐形、又确实在花钱，是最难发现的一类。
    for feat, u in (usage or {}).items():
        if feat in known:
            continue
        users.append({
            "feature": feat, "label": u.get("label") or feat,
            "group": u.get("group") or "未登记", "desc": "",
            "reserved": False,
            "override": configured.get(feat) or "",
            "model": R.resolve_model(feat),
            "calls": u.get("count") or 0, "failed": u.get("failed") or 0,
            "tokens": u.get("tokens") or 0,
        })
    # 排序：有调用的在前（按调用数降），其余按分组/标签——常看的排上面
    users.sort(key=lambda r: (-(r["calls"] or 0), r["group"], r["label"]))

    # 可选模型清单：所有厂商的 models 拉平去重，给页面下拉框用。
    # 不按功能过滤（同一模型可以给任何功能用），但**标明来自哪家**，
    # 否则 `qwen-plus` 和 `deepseek-chat` 混在一列里看不出区别。
    all_models = []
    for p in rows:
        for m in (p.get("models") or []):
            all_models.append({"model": m, "provider": p["id"], "provider_name": p["name"]})

    return envelope.ok({"providers": rows, "features": users,
                        "all_models": all_models,
                        "default_provider": default_id})


# ---------------------------------------------------------------- 模型写入
def set_models(action: str, payload: dict | None = None) -> dict:
    """改模型配置（**人在页面上下拉/点击**，不经模型）。

    action:
        `feature`           {feature, model}  model 传空 = 恢复"跟随默认"
        `default`           {provider}        换全局默认厂商
        `provider_models`   {provider, models: [...]}  改某家的模型清单
        `provider_enabled`  {provider, enabled: bool}

    ## 为什么这些写操作不需要"两阶段确认"
    `tool_manage_watchlist` 那类必须 preview→commit，是因为**模型**会自作主张改东西，
    人要在对话里核对差异。这里是**人自己在配置页下拉选**，确认动作就是那一下选择。
    再套一层确认只会让人以为"选了没生效"。

    ## 生效时机
    两个配置文件都是 **mtime 热加载**（`model_routing._cached_file` /
    `providers._cached_file`），写完下一次调用就吃到新值。
    ⚠️ 但**调度器进程**里已经 import 的模块不受影响——它下次调用时会重新读文件，
       所以也不用重启；真正需要重启的只有改了 `.env` 里的密钥。
    """
    from config import model_routing as R
    from config import providers as P

    p = dict(payload or {})
    act = str(action or "").strip().lower()

    if act == "feature":
        feat = str(p.get("feature") or "").strip()
        if not feat:
            return envelope.fail("invalid", "缺少 feature")
        model = str(p.get("model") or "").strip()
        ok = R.set_feature_model(feat, model)
        if not ok:
            return envelope.fail("error", "写入功能模型失败（文件可能被占用），详见系统日志")
        return envelope.ok({
            "feature": feat, "model": model,
            "effective": R.resolve_model(feat),
            "message": ("已把「%s」设为 %s" % (R.feature_label(feat), model))
                       if model else ("「%s」已恢复跟随默认" % R.feature_label(feat)),
        })

    if act == "default":
        pid = str(p.get("provider") or "").strip().lower()
        try:
            reg = P.get_registry()
        except Exception as e:                 # noqa: BLE001
            return envelope.fail("empty", f"{type(e).__name__}: {e}")
        ids = [x.get("id") for x in (reg.get("providers") or [])]
        if pid not in ids:
            return envelope.fail("invalid", f"未知厂商：{pid}（可选 {', '.join(map(str, ids))}）")
        reg["default_provider"] = pid
        try:
            P.save_registry(reg)
        except Exception as e:                 # noqa: BLE001
            return envelope.fail("error", f"写入注册表失败：{type(e).__name__}: {e}")
        return envelope.ok({"default_provider": pid, "message": f"默认厂商已切到 {pid}"})

    if act == "provider_models":
        pid = str(p.get("provider") or "").strip().lower()
        models = p.get("models")
        if not isinstance(models, list):
            return envelope.fail("invalid", "models 必须是数组")
        # 去空、去重、去非字符串——脏值会让下拉框出现空选项，看不出是坏的
        clean, seen = [], set()
        for m in models:
            s = str(m or "").strip()
            if s and s not in seen:
                seen.add(s)
                clean.append(s)
        try:
            reg = P.get_registry()
        except Exception as e:                 # noqa: BLE001
            return envelope.fail("empty", f"{type(e).__name__}: {e}")
        target = next((x for x in (reg.get("providers") or [])
                       if x.get("id") == pid), None)
        if target is None:
            return envelope.fail("invalid", f"未知厂商：{pid}")
        target["models"] = clean
        try:
            P.save_registry(reg)
        except Exception as e:                 # noqa: BLE001
            return envelope.fail("error", f"写入注册表失败：{type(e).__name__}: {e}")
        return envelope.ok({"provider": pid, "models": clean,
                            "message": f"{pid} 现有 {len(clean)} 个模型"})

    if act == "provider_enabled":
        pid = str(p.get("provider") or "").strip().lower()
        enabled = bool(p.get("enabled"))
        try:
            reg = P.get_registry()
        except Exception as e:                 # noqa: BLE001
            return envelope.fail("empty", f"{type(e).__name__}: {e}")
        target = next((x for x in (reg.get("providers") or [])
                       if x.get("id") == pid), None)
        if target is None:
            return envelope.fail("invalid", f"未知厂商：{pid}")
        target["enabled"] = enabled
        try:
            P.save_registry(reg)
        except Exception as e:                 # noqa: BLE001
            return envelope.fail("error", f"写入注册表失败：{type(e).__name__}: {e}")
        return envelope.ok({"provider": pid, "enabled": enabled,
                            "message": f"{pid} 已{'启用' if enabled else '停用'}"})

    return envelope.fail("invalid",
                         f"未知 action：{action}（可用 feature/default/provider_models/provider_enabled）")


def _host(base_url: str) -> str:
    """`https://dashscope.aliyuncs.com/compatible-mode/v1` → `dashscope.aliyuncs.com`"""
    s = str(base_url or "")
    for pre in ("https://", "http://"):
        if s.startswith(pre):
            s = s[len(pre):]
    return s.split("/")[0] or "—"


# ---------------------------------------------------------------- LLM 成本
def cost(days: int = 7) -> dict:
    from core import llm_cost

    budget = llm_cost.budget_status()
    report = llm_cost.daily_report(days=max(1, min(30, days)))
    if report.get("error"):
        return envelope.fail("empty", f"成本报表读库失败：{report['error']}")
    try:
        by_feature = llm_cost.feature_costs(hours=24)
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[system] 按功能拆成本失败: %s: %s", type(e).__name__, e)
        by_feature = {}

    data = {
        "budget": budget,
        "days": report.get("days") or [],
        "by_purpose": report.get("by_purpose") or [],
        "by_model": report.get("by_model") or [],
        "by_feature_24h": by_feature,
        "total_cost": report.get("total_cost"),
        "total_calls": report.get("total_calls"),
        "total_tokens": report.get("total_tokens"),
        "avg_workday_cost": report.get("avg_workday_cost"),
        "workday_count": report.get("workday_count"),
        "window_days": days,
    }
    as_of = (data["days"] or [{}])[-1].get("date") if data["days"] else None
    return envelope.ok(data, as_of=as_of)


def audit(limit: int = 50) -> dict:
    """审计与留痕：**配置变更流水 / 推送流水 / 情绪标注留痕**。

    ## 为什么补（2026-10-26）
    这三张表一直有数据、web 层零引用，属于「AI 改了配置你看不见、飞书到底推没推
    你看不见、新闻情绪标注花了多少 token 你也看不见」。它们都不影响主链结果，
    但恰好是"出错时唯一能回溯"的东西：
    · `agent_config_changes` —— AI 通过工具改自选/配置时，前后快照与回滚锚点
      （`change_id`；`xxx-restore` 行是对 `xxx` 的回滚动作**本身**）。
    · `push_log` —— 飞书推送的幂等账本，同 `(day, channel, content_hash)`
      只发一次；`status<>'ok'` 就是"该推没推成功"。
    · `sentiment_labels` —— 新闻情绪打标留痕，`stance='unscored'` 是**判不出方向**，
      不是"中性"，所以单独计数、不进任何分子。

    ⚠️ 三块各自独立降级：一块读失败不许拖垮另外两块（页面上就是那一块显示原因）。
    """
    from db import store

    data: dict = {"changes": [], "pushes": [], "sentiment": None,
                  "errors": {}}
    lim = max(1, min(200, int(limit)))

    try:
        data["changes"] = store.list_agent_changes(limit=lim)
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[system] 配置变更流水读取失败: %s", e)
        data["errors"]["changes"] = f"{type(e).__name__}: {e}"

    try:
        data["pushes"] = store.list_push_log(limit=lim)
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[system] 推送流水读取失败: %s", e)
        data["errors"]["pushes"] = f"{type(e).__name__}: {e}"

    try:
        data["sentiment"] = store.sentiment_label_summary(days=30)
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[system] 情绪标注留痕读取失败: %s", e)
        data["errors"]["sentiment"] = f"{type(e).__name__}: {e}"

    push_ok = sum(1 for p in data["pushes"] if p.get("status") == "ok")
    data["counts"] = {
        "changes": len(data["changes"]),
        "pushes": len(data["pushes"]),
        "pushes_ok": push_ok,
        "pushes_failed": len(data["pushes"]) - push_ok,
        "sentiment": (data["sentiment"] or {}).get("total") or 0,
    }
    as_of = next((p.get("created_at") for p in data["pushes"] if p.get("created_at")),
                 None)
    return envelope.ok(data, as_of=as_of)
