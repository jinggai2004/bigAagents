"""LLM 调用成本核算（2026-10-23）。

为什么要有这一层
────────────────
`llm_calls` 从 Phase 6.2 起就记 token，但**从来没有记钱**。而本项目恰好是
"token 数与金额严重脱钩"的典型：

| 模型 | 输入单价（元/百万） | 同一 token 的相对成本 |
|---|---:|---:|
| qwen3.7-flash（≤32K） | 0.20 | 1× |
| deepseek-v4-flash（空闲） | 1.50 | 7.5× |
| qwen-max | 2.40 | 12× |
| deepseek-v4-pro（高峰） | 9.00 | **45×** |

于是"今天 token 翻倍了"这句话本身没有意义——它可能对应**更便宜**的一天
（写作节点全走 qwen3.7-flash）也可能对应**贵 5 倍**的一天（pro 撞上高峰）。
只看 token 得出的结论可以完全反向。

三个刻意的设计
──────────────
1. **单价表外置**（`config/llm_prices.json`），本模块只读不算常量。
   价格会变（DeepSeek 2026-08-17 改过峰谷，qwen 有阶梯），写死等于让历史成本
   随改价被静默重算。
2. **金额落库**（`llm_calls.cost_cny`），记的是**调用发生时**的价格——
   会计口径：已发生的费用不因后来的调价而改变。要看"按现价重算"用 `recalc()`，
   那是另一个问题，必须显式调用。
3. **峰谷按调用小时判，不按任务名判**。同一个 `evening_analysis`，18:10 跑是半价、
   15:30 跑是原价——这也是当初把它挪到 18:10 的原因，账上必须能看出这件事有没有生效。

三层 API
────────
    call_cost()     纯函数：给模型/token/时刻 → 金额（零 IO，测试主入口）
    daily_report()  读库汇总：按日 / 按用途 / 按模型
    budget_status() 今日花费与日预算的关系（给调度与页面用）
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

_logger = logging.getLogger(__name__)

# 单价表路径（环境变量覆盖为测试用；与 config/providers.py 同一套约定）
ENV_PRICES_PATH = "BIGA_LLM_PRICES"
_DEFAULT_PRICES = Path(__file__).resolve().parent.parent / "config" / "llm_prices.json"

# 日预算默认值。刻意给 5 元：实测日均约 ¥1.4（20 交易日 ≈ ¥28/月），
# 5 元等于"日均的 3.5 倍"，只有异常（批量重跑、循环重试）才会撞到，
# 正常日不会误伤。0 或负数 = 关闭闸门。
DEFAULT_DAILY_BUDGET_CNY = 5.0

_cache: dict = {"path": None, "mtime": None, "data": None}


# ------------------------------------------------------------------
# 单价表
# ------------------------------------------------------------------
def prices_path() -> Path:
    import os
    override = os.getenv(ENV_PRICES_PATH)
    return Path(override) if override else _DEFAULT_PRICES


def load_prices(force: bool = False) -> dict:
    """读单价表（按 mtime 缓存，改文件即生效，不用重启进程）。

    读不到 / 解析失败 → 返回内置最小表并记 WARNING。
    ⚠️ 不抛异常：**记账失败不能把一次成功的模型调用变成失败**（与 `_log_llm_call` 同一条纪律）。
    但也**绝不返回空表**——空表会让所有金额变 0，看起来"免费"，比报错更危险。
    """
    path = prices_path()
    try:
        mtime = path.stat().st_mtime
    except OSError as e:
        _logger.warning("单价表不可用（%s）：%s ⇒ 用内置兜底表", path, e)
        return _FALLBACK_PRICES

    if not force and _cache["data"] is not None and _cache["path"] == str(path) \
            and _cache["mtime"] == mtime:
        return _cache["data"]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "models" not in data:
            raise ValueError("缺少 models 段")
        _cache.update({"path": str(path), "mtime": mtime, "data": data})
        return data
    except Exception as e:                              # noqa: BLE001
        _logger.warning("单价表解析失败（%s）：%s ⇒ 用内置兜底表", path, e)
        return _FALLBACK_PRICES


# 兜底表：只在 JSON 读不到时用。数值与 config/llm_prices.json 一致，
# 但**不含峰谷与阶梯**（保守取原价），且故意把未知模型按最贵的 qwen-max 计。
_FALLBACK_PRICES = {
    "models": {
        "deepseek-v4-flash": {"input": 3.0, "output": 9.0},
        "deepseek-v4-pro": {"input": 9.0, "output": 27.0},
        "qwen3.7-flash": {"input": 0.6, "output": 2.4},
        "qwen-max": {"input": 2.4, "output": 9.6},
    },
    "unknown_model": {"input": 2.4, "output": 9.6},
    "peak_hours": [],
}


def _is_peak(at: datetime, table: dict) -> bool:
    return at.hour in tuple(table.get("peak_hours") or ())


def price_of(model: str, prompt_tokens: int = 0, at: datetime = None,
             table: dict = None) -> tuple:
    """返回 `(输入单价, 输出单价, 档位标签)`，单位元/百万 token。

    规则顺序（**先精确匹配，再前缀匹配，最后未知兜底**）：
      1. 精确命中模型名；
      2. `tiers` 阶梯（按**本次请求**的输入 token 判档，不是按天累计）；
      3. 峰谷（有 `peak_input` 且落在高峰小时）；
      4. 未知模型 → 按最贵的在用模型保守计（宁可高估）。
    """
    table = table if table is not None else load_prices()
    at = at or datetime.now()
    key = str(model or "").strip().lower()
    models = table.get("models") or {}

    spec = models.get(key)
    tier_label = "标准"
    if spec is None:                                    # 前缀匹配：deepseek-v4-flash-xxx
        for name, candidate in models.items():
            if key.startswith(name):
                spec, tier_label = candidate, f"~{name}"
                break

    if spec is None:
        unk = table.get("unknown_model") or {"input": 2.4, "output": 9.6}
        return float(unk["input"]), float(unk["output"]), "未知模型(保守计)"

    tiers = spec.get("tiers")
    if tiers:
        pt = int(prompt_tokens or 0)
        for t in tiers:
            limit = t.get("max_input")
            if limit is None or pt <= int(limit):
                label = "≤%dK" % (int(limit) // 1024) if limit else "长上下文"
                return float(t["input"]), float(t["output"]), f"阶梯 {label}"

    if "peak_input" in spec and _is_peak(at, table):
        return float(spec["peak_input"]), float(spec["peak_output"]), "高峰"

    return float(spec["input"]), float(spec["output"]), ("空闲" if "peak_input" in spec else "标准")


# ------------------------------------------------------------------
# 单次成本（纯函数，零 IO）
# ------------------------------------------------------------------
def call_cost(model: str, prompt_tokens: int, completion_tokens: int,
              at: datetime = None, table: dict = None) -> float:
    """一次调用的金额（元）。保留 6 位小数——单次调用常在 ¥0.000x 量级，四舍五入到分就全是 0。"""
    pin, pout, _ = price_of(model, prompt_tokens or 0, at, table)
    cost = ((prompt_tokens or 0) * pin + (completion_tokens or 0) * pout) / 1_000_000.0
    return round(cost, 6)


# ------------------------------------------------------------------
# 读库汇总
# ------------------------------------------------------------------
def _parse_ts(value) -> datetime:
    try:
        return datetime.strptime(str(value)[:19], "%Y-%m-%d %H:%M:%S")
    except Exception as e:                              # noqa: BLE001
        # ⚠️ 回落成"现在"会让**时间序列成本**整段错位，必须留痕（不能静默）。
        _logger.debug("[成本] 时间戳解析失败（按当前时间处理）: %r (%s)", value, e)
        return datetime.now()


def _row_cost(row: dict, table: dict) -> float:
    """优先用落库金额；NULL 才按当前单价估算（老行）。

    ⚠️ 失败调用（ok=0）金额按 0 计：请求被拒通常不计费，
    即使计费也拿不到 usage（prompt_tokens 为 NULL），硬算会得出假金额。
    """
    if not row.get("ok"):
        return 0.0
    stored = row.get("cost_cny")
    if stored is not None:
        return float(stored)
    return call_cost(row.get("model"), row.get("prompt_tokens") or 0,
                     row.get("completion_tokens") or 0,
                     _parse_ts(row.get("created_at")), table)


def feature_costs(hours: int = 24) -> dict:
    """近 N 小时按 purpose 汇总金额 → `{purpose: cost}`。

    为什么不塞进 `config/model_routing.feature_usage`：那个函数的职责是"路由是否
    覆盖"，让它依赖价格表会把「配置层」和「会计层」绑死——价格错了不该影响路由断言。
    页面分别取两份数据、按功能键合并即可。
    """
    from db import store
    since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        rows = store.llm_cost_rows(since=since)
    except Exception as e:                              # noqa: BLE001
        _logger.warning("按功能取成本失败：%s", e)
        return {}
    table = load_prices()
    out: dict = {}
    for r in rows:
        key = str(r.get("purpose") or "unknown")
        out[key] = out.get(key, 0.0) + _row_cost(r, table)
    return {k: round(v, 6) for k, v in out.items()}


def daily_report(days: int = 7, until: datetime = None) -> dict:
    """近 N 天成本报表：按日 / 按用途 / 按模型。

    返回结构（字段名稳定，页面与 CLI 共用）：
        {"days":[{date, cost, calls, tokens, failed}],
         "by_purpose":[{key, cost, calls, tokens}],
         "by_model":[{key, cost, calls, tokens}],
         "total_cost", "total_calls", "total_tokens",
         "avg_workday_cost", "workday_count"}
    ⚠️ `avg_workday_cost` 只统计**有实际工作量**的日子（token > 10 万）：
       周末/节假日只有零星调用，把它们平均进去会把日均压成"看起来几乎免费"。
    """
    from db import store
    until = until or datetime.now()
    since = (until - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        rows = store.llm_cost_rows(since=since)
    except Exception as e:                              # noqa: BLE001
        _logger.warning("成本报表读库失败：%s", e)
        return {"days": [], "by_purpose": [], "by_model": [], "total_cost": 0.0,
                "total_calls": 0, "total_tokens": 0, "avg_workday_cost": 0.0,
                "workday_count": 0, "error": str(e)}

    table = load_prices()
    by_day, by_purpose, by_model = {}, {}, {}
    for r in rows:
        cost = _row_cost(r, table)
        tokens = int(r.get("total_tokens") or 0) or (
            int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0))
        day = str(r.get("created_at"))[:10]
        for bucket, key in ((by_day, day), (by_purpose, r.get("purpose") or "unknown"),
                            (by_model, r.get("model") or "unknown")):
            d = bucket.setdefault(key, {"key": key, "cost": 0.0, "calls": 0,
                                        "tokens": 0, "failed": 0})
            d["cost"] += cost
            d["calls"] += 1
            d["tokens"] += tokens
            if not r.get("ok"):
                d["failed"] += 1

    days_list = [{**v, "date": v["key"]} for v in by_day.values()]
    days_list.sort(key=lambda x: x["date"], reverse=True)
    workdays = [d for d in days_list if d["tokens"] > 100_000]
    total_cost = sum(d["cost"] for d in days_list)

    def _sorted(bucket):
        return sorted(bucket.values(), key=lambda x: -x["cost"])

    return {
        "days": days_list,
        "by_purpose": _sorted(by_purpose)[:30],
        "by_model": _sorted(by_model),
        "total_cost": round(total_cost, 4),
        "total_calls": len(rows),
        "total_tokens": sum(d["tokens"] for d in days_list),
        "avg_workday_cost": round(sum(d["cost"] for d in workdays) / len(workdays), 4)
                            if workdays else 0.0,
        "workday_count": len(workdays),
    }


def today_cost(now: datetime = None) -> float:
    """今天（自然日）已花金额，给预算闸门用。"""
    from db import store
    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")
    try:
        rows = store.llm_cost_rows(since=f"{today} 00:00:00")
    except Exception as e:                              # noqa: BLE001
        _logger.warning("读取今日成本失败：%s", e)
        return 0.0
    table = load_prices()
    return round(sum(_row_cost(r, table) for r in rows
                     if str(r.get("created_at"))[:10] == today), 4)


def budget_status(daily_cap: float = None, now: datetime = None) -> dict:
    """今日花费 vs 日预算。

    `daily_cap <= 0` = 闸门关闭（`enabled=False`，永远 `exceeded=False`）。
    ⚠️ 读不到花费时返回 `exceeded=False` 但带 `degraded=True`——
       "账目查不清"不能变成"阻止今天所有任务"，也不能假装一切正常。
    """
    import config
    if daily_cap is None:
        cap = float(getattr(config, "LLM_DAILY_BUDGET_CNY", DEFAULT_DAILY_BUDGET_CNY) or 0)
    else:
        cap = float(daily_cap)
    enabled = cap > 0
    try:
        spent = today_cost(now)
        degraded = False
    except Exception as e:                              # noqa: BLE001
        _logger.warning("预算检查降级：%s", e)
        spent, degraded = 0.0, True
    return {
        "cap": round(cap, 4),
        "spent": spent,
        "remaining": round(cap - spent, 4) if enabled else None,
        "ratio": round(spent / cap, 4) if enabled and cap else 0.0,
        "exceeded": bool(enabled and spent >= cap),
        "enabled": enabled,
        "degraded": degraded,
    }


# ------------------------------------------------------------------
# 维护：按现价重算历史金额
# ------------------------------------------------------------------
def recalc(days: int = 30, dry_run: bool = True) -> dict:
    """按**当前**单价重算历史金额（显式调用，绝不自动跑）。

    为什么不做成自动：改一次单价就重算全部历史，会让"上周花了多少"这个数字
    每天都在变，与账单对不上。会计口径要的是"发生时多少钱"。
    这个函数只用于两种场景：① 单价表填错了要修；② 想回答"按今天价格，那周会花多少"。
    """
    from db import store
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    rows = store.llm_cost_rows(since=since)
    table = load_prices(force=True)
    changed, delta = 0, 0.0
    for r in rows:
        new = _row_cost({**r, "cost_cny": None}, table)     # 强制重算
        old = float(r.get("cost_cny") or 0.0)
        if abs(new - old) > 1e-9:
            changed += 1
            delta += new - old
            if not dry_run:
                store.update_llm_cost(r["id"], new)
    return {"scanned": len(rows), "changed": changed, "delta_cny": round(delta, 4),
            "dry_run": dry_run}
