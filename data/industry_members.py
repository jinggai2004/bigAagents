"""行业成分股映射（新浪行业分类，全市场）。

为什么需要（2026-10-09，热点新闻工作流）：
  新闻聚合出「利好行业」后，要落到"可能利好的个股"，就必须有 行业 → 成分股 的映射。
  库里现成的 `backtest_pool.json::industry_map` 只有 **120 只**（回测池抽样），覆盖太窄；
  这里用与回测池相同的取数路径（`stock_sector_spot` + `stock_sector_detail`，49 个行业）
  拉全市场映射。

成本与可靠性设计（铁律）：
  - 全量拉一次约 49 次请求、几分钟 → **落盘缓存**（`data/industry_members.json`，
    默认 24h 内直接读缓存），绝不每次调用都联网；
  - 每次请求带超时 + 失败**不抛给调用方**而是返回 `unavailable` 标记
    （"取数失败必须可见"，缺就是缺，不补造）；
  - 缓存过期但重新拉取失败 → 返回**陈旧缓存**并置 `degraded=True`（宁可旧，不可无，
    且必须告诉调用方这是旧数据）。
"""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

_logger = logging.getLogger(__name__)

# ⚠️⚠️ 缓存路径必须走 `data_root()`，不能用 `dirname(__file__)`：
#   frozen（exe）下 `__file__` 落在只读的 `_internal/`，缓存实际写在
#   `data_root()/data/`；用 `__file__` ⇒ exe 里永远找不到缓存 ⇒ 上游以为"没数据"，
#   推荐链路直接降级成"行业成分两个来源都不可用"（2026-10-29 exe 冒烟实测）。
#   规则与 `data/panel_store.py` 一致（唯一入口：`config/paths.data_root()`）。
from config.paths import data_root as _paths_data_root   # noqa: E402

_PROJECT_ROOT = _paths_data_root()
CACHE_PATH = os.path.join(_PROJECT_ROOT, "data", "industry_members.json")
CACHE_TTL_HOURS = 24
REQUEST_TIMEOUT = 20
# 单行业请求并发数：新浪板块接口对高频不友好，2 并发 + 49 个行业 ≈ 1 分钟内完成
MAX_WORKERS = 2


class IndustryMembersUnavailable(RuntimeError):
    """行业成分完全不可用（无缓存且拉取失败）。"""


def _fetch_all() -> dict:
    """联网拉取全市场 {行业: [代码]}。抛异常 = 彻底失败。"""
    import akshare as ak
    import socket
    socket.setdefaulttimeout(REQUEST_TIMEOUT)

    spot = ak.stock_sector_spot(indicator="新浪行业")
    industries = [(str(r["label"]), str(r["板块"])) for _, r in spot.iterrows()]
    if not industries:
        raise IndustryMembersUnavailable("stock_sector_spot 返回空表")

    members: dict[str, list] = {}

    def _one(item):
        label, name = item
        try:
            df = ak.stock_sector_detail(sector=label)
        except Exception as e:                      # noqa: BLE001
            return name, None, f"{type(e).__name__}: {e}"
        codes = []
        for _, r in df.iterrows():
            code = str(r.get("code") or "").strip()
            if code.isdigit() and len(code) == 6:
                codes.append(code)
        return name, codes, None

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for name, codes, err in ex.map(_one, industries):
            if err is not None:
                failures.append(f"{name}({err[:40]})")
                continue
            if codes:
                members[name] = codes
    # ⚠️ 少数行业失败不算失败（几十个行业全失败才判不可用），
    #    但失败清单必须留在缓存里，调用方可见
    if not members:
        raise IndustryMembersUnavailable(f"全部行业拉取失败: {failures[:3]}")
    return {"members": members, "failures": failures}


def get_industry_members(force: bool = False,
                         max_age_hours: int = CACHE_TTL_HOURS) -> dict:
    """返回 {"members": {行业: [code]}, "fetched_at": str, "degraded": bool,
    "failures": [str], "age_hours": float}。

    - `degraded=True`：数据是**陈旧缓存**（本次拉取失败）或**部分行业缺失**——
      调用方必须把这个标记带给用户。
    - 完全无数据 → 抛 `IndustryMembersUnavailable`（调用方决定降级方式）。
    """
    cached, age = _load_cache()
    if cached is not None and not force and age <= max_age_hours:
        out = dict(cached)
        out["age_hours"] = round(age, 2)
        out["degraded"] = bool(cached.get("failures"))
        return out

    try:
        fresh = _fetch_all()
    except Exception as e:                          # noqa: BLE001
        if cached is not None:
            _logger.warning("[行业成分] 拉取失败(%s)，退回 %.1f 小时前的缓存", e, age)
            out = dict(cached)
            out["age_hours"] = round(age, 2)
            out["degraded"] = True
            out["fetch_error"] = f"{type(e).__name__}: {e}"
            return out
        raise

    payload = {"members": fresh["members"], "failures": fresh["failures"],
               "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    _save_cache(payload)
    return {**payload, "age_hours": 0.0,
            "degraded": bool(fresh["failures"])}


def code_to_industry(members: dict) -> dict:
    """{行业: [code]} → {code: 行业}（后面一个行业一行代码）"""
    out: dict[str, str] = {}
    for ind, codes in (members or {}).items():
        for c in codes:
            # 先到先得：一只票理论上只属一个新浪行业，重复时保留首个
            out.setdefault(str(c), ind)
    return out


def load_cached_only() -> dict:
    """**只读磁盘缓存，绝不联网**（Web 只读端点专用）。

    为什么必须单独一条入口：全量拉一次要 49 次请求、约 1 分钟。
    `/api/*` 这类请求里联网，等于把"打开页面"变成"打一轮上游"，
    并且会把只读端点变成不可预测的长尾延迟。

    返回结构同 `get_industry_members`（`degraded` 按"缓存是否过期/有失败项"如实计算）；
    没有缓存 → 抛 `IndustryMembersUnavailable`，由调用方降级（缺就是缺，不造假）。
    """
    cached, age = _load_cache()
    if cached is None:
        raise IndustryMembersUnavailable("无磁盘缓存（只读模式不联网刷新）")
    out = dict(cached)
    out["age_hours"] = round(age, 2)
    out["read_only"] = True
    out["degraded"] = bool(cached.get("failures")) or age > CACHE_TTL_HOURS
    return out


# ------------------------------------------------------------------
# 磁盘缓存
# ------------------------------------------------------------------

def _load_cache() -> tuple[dict | None, float]:
    if not os.path.exists(CACHE_PATH):
        return None, 0.0
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        ts = datetime.strptime(data["fetched_at"], "%Y-%m-%d %H:%M:%S").timestamp()
        return data, (time.time() - ts) / 3600
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        _logger.warning("[行业成分] 缓存损坏，忽略重建: %s", e)
        return None, 0.0


def _save_cache(payload: dict) -> None:
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    # Windows：os.replace 要求目标未被打开；小文件+低频写入，退避重试兜底
    for attempt in range(3):
        try:
            os.replace(tmp, CACHE_PATH)
            return
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    _logger.warning("[行业成分] 缓存写入失败（文件被占用），本次结果未落盘")
