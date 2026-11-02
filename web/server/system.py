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
    """定时任务清单（内置 + 自定义）+ 运行态 + 可编辑元数据。

    2026-11-01 起每行带 `note` / `model` / `enabled` / `mode_label`：
    「任务清单是数据不是代码」这一层的后端（`core/scheduled_jobs.py`）早就写好了，
    但页面只渲染三列只读信息 ⇒ 用户看不到也改不了备注/类型/模型（项目铁律
    "函数写好 ≠ 页面看得见"）。这里把它需要的东西一次给全。
    """
    from core.scheduler import JOBS
    from core.scheduled_jobs import JOB_MODES
    import core.ui_runs as ui_runs
    import core.ui_tasks as ui_tasks

    try:
        from db import store as _store
        overrides = _store.list_builtin_overrides()
        custom = _store.list_scheduled_jobs()
    except Exception as e:                       # noqa: BLE001
        _logger.warning("[system] 任务配置读取失败（按默认值展示）: %s", e)
        overrides, custom = {}, []

    def _mode_label(mode: str) -> str:
        return str((JOB_MODES.get(mode) or {}).get("label") or mode or "")

    rows = []
    for job_id, cron, mode, desc in JOBS:
        prog = ui_tasks.read_progress(job_id)
        rec = ui_runs.get_running_any(job_id)
        alive = ui_runs.get_running(job_id) is not None
        ov = overrides.get(job_id) or {}
        rows.append({
            "job_id": job_id,
            "kind": "builtin",
            "time": _cron_label(cron),
            "mode": mode,
            "mode_label": _mode_label(mode),
            "desc": desc,
            "note": ov.get("note") or "",
            "model": ov.get("model") or "",
            "enabled": ov.get("enabled", True) is not False,
            "can_delete": False,          # 内置任务只能停用/恢复默认，不能删
            "running": alive,
            "last_run_at": (rec or {}).get("started_at") if rec else None,
            "progress": _slim(prog),
        })

    for j in custom:
        job_id = str(j.get("job_id") or "")
        prog = ui_tasks.read_progress(job_id)
        rec = ui_runs.get_running_any(job_id)
        alive = ui_runs.get_running(job_id) is not None
        cron = j.get("cron") or {}
        rows.append({
            "job_id": job_id,
            "kind": "custom",
            "time": _cron_label(cron) if cron else "—",
            "mode": j.get("mode") or "",
            "mode_label": _mode_label(j.get("mode") or ""),
            "desc": j.get("name") or job_id,
            "note": j.get("note") or "",
            "model": j.get("model") or "",
            "enabled": bool(j.get("enabled", True)),
            "can_delete": True,
            "trading_day_only": bool(j.get("trading_day_only", True)),
            "cron": cron,
            "error": j.get("error") or "",
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
        "builtin": sum(1 for r in rows if r["kind"] == "builtin"),
        "custom": sum(1 for r in rows if r["kind"] == "custom"),
        "disabled": sum(1 for r in rows if not r["enabled"]),
        "running": sum(1 for r in rows if r["running"]),
        # 新建任务表单要用的白名单（含 label/是否调模型/是否限交易日）
        "modes": [{"mode": k, "label": v.get("label") or k,
                   "llm": bool(v.get("llm")), "desc": v.get("desc") or "",
                   "needs_trading_day": bool(v.get("needs_trading_day"))}
                  for k, v in JOB_MODES.items()],
        "web_tasks": web_rows,
    })


def set_jobs(action: str, payload: dict | None = None) -> dict:
    """定时任务的增删改查（**人在页面上点**，不经模型）。

    action:
        `save`        新增或更新自定义任务 {job_id?, name, mode, cron, note, model, enabled}
        `toggle`      启停 {job_id, enabled}（自动区分内置/自定义）
        `delete`      删除 {job_id}（自定义=真删；内置=清覆盖行恢复默认）
        `set_builtin` 内置任务的元数据覆盖 {job_id, note?, model?, enabled?}

    ## 为什么不需要"两阶段确认"
    与 `set_models` 同一条理由：这里是**人自己在配置页点**，选择动作本身就是确认。
    再套一层确认只会让人以为"点了没生效"。（模型侧的写操作才有 preview→commit。）

    ## 生效时机
    自定义任务与内置覆盖都由调度进程的 60s 热同步循环落地（`reload_custom_jobs`），
    **改完不用重启**。但最多滞后 60 秒才真正重排——返回值里明确写这一句。
    """
    from core.scheduled_jobs import JOB_MODES, find_conflicts, validate_job
    from db import store as _store

    p = dict(payload or {})
    act = str(action or "").strip().lower()
    job_id = str(p.get("job_id") or "").strip()

    if act in ("toggle", "delete", "set_builtin") and not job_id:
        return envelope.fail("invalid", "缺少 job_id")

    if act == "save":
        try:
            incoming = {
                "job_id": job_id or None,
                "name": p.get("name"),
                "mode": p.get("mode"),
                "cron": p.get("cron") or {},
                "note": p.get("note"),
                "model": p.get("model"),
                "enabled": p.get("enabled", True),
                "trading_day_only": p.get("trading_day_only", True),
            }
            if job_id:
                incoming["_original_id"] = job_id
            try:
                taken = [j["job_id"] for j in _store.list_scheduled_jobs()]
            except Exception:                  # noqa: BLE001
                taken = []
            job = validate_job(incoming, taken_ids=taken, require_id=bool(job_id))
        except ValueError as e:
            return envelope.fail("invalid", str(e))
        # 冲突提示不阻断（人和机器都需要知道会撞车，但排期是自由选择）
        try:
            conflicts = find_conflicts(job["cron"], mode=job["mode"],
                                       exclude_id=job.get("job_id"))
        except Exception:                      # noqa: BLE001
            conflicts = []
        try:
            _store.upsert_scheduled_job(job)
        except Exception as e:                 # noqa: BLE001
            _logger.warning("[system] 保存任务失败: %s", e)
            return envelope.fail("error", f"写入失败：{type(e).__name__}: {e}")
        return envelope.ok({
            "job": {**job, "time": _cron_label(job["cron"])},
            "conflicts": conflicts,
            "message": ("已保存「%s」，调度器将在 60 秒内热加载" % job["name"]),
        })

    if act == "toggle":
        want = bool(p.get("enabled", True))
        builtin_ids = {x[0] for x in _JOBS_CACHE()}
        try:
            if job_id in builtin_ids:
                rec = _store.set_builtin_override(job_id, enabled=want)
            else:
                if not _store.set_scheduled_job_enabled(job_id, want):
                    return envelope.fail("missing", f"任务不存在：{job_id}")
                rec = {"job_id": job_id, "enabled": want}
        except Exception as e:                 # noqa: BLE001
            return envelope.fail("error", f"写入失败：{type(e).__name__}: {e}")
        return envelope.ok({"job_id": job_id, "enabled": want,
                            "message": ("已%s，调度器将在 60 秒内热加载"
                                        % ("启用" if want else "停用"))})

    if act == "delete":
        builtin_ids = {x[0] for x in _JOBS_CACHE()}
        try:
            if job_id in builtin_ids:
                # 内置任务不能真删（代码里还有），删覆盖行 = 恢复默认
                _store.delete_builtin_override(job_id)
                return envelope.ok({"job_id": job_id, "restored": True,
                                    "message": "内置任务已恢复默认（启用、无备注、跟随默认模型）"})
            ok = _store.delete_scheduled_job(job_id)
        except Exception as e:                 # noqa: BLE001
            return envelope.fail("error", f"删除失败：{type(e).__name__}: {e}")
        if not ok:
            return envelope.fail("missing", f"任务不存在：{job_id}")
        return envelope.ok({"job_id": job_id, "deleted": True,
                            "message": "已删除，调度器将在 60 秒内移除"})

    if act == "set_builtin":
        builtin_ids = {x[0] for x in _JOBS_CACHE()}
        if job_id not in builtin_ids:
            return envelope.fail("invalid", f"{job_id} 不是内置任务，请用 save")
        # 局部更新：只有 payload 里出现的键才改（改开关不该抹掉备注）
        kw = {}
        if "enabled" in p:
            kw["enabled"] = bool(p.get("enabled"))
        if "note" in p:
            kw["note"] = str(p.get("note") or "")
        if "model" in p:
            m = str(p.get("model") or "").strip()[:80]
            if m:
                try:
                    from config import providers as _P
                    known = {x for pr in _P.get_providers()
                             for x in (pr.get("models") or [])}
                    if known and m not in known:
                        return envelope.fail(
                            "invalid", f"模型「{m}」不在任何厂商的模型清单里")
                except Exception as e:         # noqa: BLE001
                    _logger.warning("[system] 模型校验跳过: %s", e)
            kw["model"] = m
        try:
            rec = _store.set_builtin_override(job_id, **kw)
        except Exception as e:                 # noqa: BLE001
            return envelope.fail("error", f"写入失败：{type(e).__name__}: {e}")
        return envelope.ok({"job_id": job_id, **rec,
                            "message": "已保存，调度器将在 60 秒内热加载"})

    return envelope.fail("invalid",
                         f"未知操作「{action}」；可选 save/toggle/delete/set_builtin")


def _JOBS_CACHE() -> list:
    from core.scheduler import JOBS
    return JOBS


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

    if act == "provider_add":
        # 新增厂商（2026-11-01）：此前只能在文件里手写 llm_providers.json
        pid = str(p.get("provider") or p.get("id") or "").strip().lower()
        try:
            pid = P.validate_provider_id(pid)
        except ValueError as e:
            return envelope.fail("invalid", str(e))
        name = str(p.get("name") or pid).strip()[:40]
        base_url = str(p.get("base_url") or "").strip()
        if not base_url:
            return envelope.fail("invalid", "缺少 base_url（OpenAI 兼容端点，"
                                            "如 https://api.example.com/v1）")
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            return envelope.fail("invalid", "base_url 必须以 http:// 或 https:// 开头")
        api_key_env = str(p.get("api_key_env") or
                          f"{pid.upper().replace('-', '_')}_API_KEY").strip()
        models = p.get("models") or []
        if isinstance(models, str):
            models = [x for x in (s.strip() for s in models.split(",")) if x]
        clean, seen = [], set()
        for m in models if isinstance(models, list) else []:
            s = str(m or "").strip()
            if s and s not in seen:
                seen.add(s)
                clean.append(s)
        try:
            reg = P.get_registry()
        except Exception as e:                 # noqa: BLE001
            return envelope.fail("empty", f"{type(e).__name__}: {e}")
        exists = [x for x in (reg.get("providers") or []) if x.get("id") == pid]
        if exists:
            return envelope.fail("invalid", f"厂商 ID「{pid}」已存在，请换一个")
        reg.setdefault("providers", []).append({
            "id": pid, "name": name, "base_url": base_url,
            "api_key_env": api_key_env,
            "models": clean,
            "match": [f"{pid}-*"], "enabled": True,
        })
        try:
            P.save_registry(reg)
        except Exception as e:                 # noqa: BLE001
            return envelope.fail("error", f"写入注册表失败：{type(e).__name__}: {e}")
        return envelope.ok({
            "provider": pid,
            "message": (f"已新增厂商「{name}」。别忘了在 .env 里配置 {api_key_env}，"
                        f"然后重启进程让密钥生效"),
        })

    if act == "provider_remove":
        pid = str(p.get("provider") or "").strip().lower()
        try:
            reg = P.get_registry()
        except Exception as e:                 # noqa: BLE001
            return envelope.fail("empty", f"{type(e).__name__}: {e}")
        provs = [x for x in (reg.get("providers") or []) if x.get("id") != pid]
        if len(provs) == len(reg.get("providers") or []):
            return envelope.fail("invalid", f"未知厂商：{pid}")
        if not provs:
            return envelope.fail("invalid", "至少要保留一个厂商，否则 LLM 调用全部失败")
        # 删的是默认厂商 ⇒ 默认回落到剩下的第一个（否则注册表里默认指向不存在的 id，
        # `default_provider()` 会静默返回 None，所有"跟随默认"的功能一起失灵）
        switched = ""
        if str(reg.get("default_provider") or "").lower() == pid:
            reg["default_provider"] = provs[0]["id"]
            switched = f"，默认厂商已切到 {provs[0]['id']}"
        reg["providers"] = provs
        try:
            P.save_registry(reg)
        except Exception as e:                 # noqa: BLE001
            return envelope.fail("error", f"写入注册表失败：{type(e).__name__}: {e}")
        return envelope.ok({"provider": pid, "removed": True,
                            "message": f"已删除厂商「{pid}」{switched}"})

    return envelope.fail("invalid",
                         f"未知 action：{action}（可用 feature/default/provider_models/"
                         f"provider_enabled/provider_add/provider_remove）")


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
