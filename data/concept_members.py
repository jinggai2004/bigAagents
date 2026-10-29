"""概念板块 → 成分股映射（东方财富数据中心，全市场）。

为什么需要（2026-10-28，交接报告阶段 B）
--------------------------------------
阶段 A 只把新闻关键词从 5 个大类扩到 11 个，但**大类仍是粗筐**：
新浪「有色金属」一筐装着铜、铝、黄金、稀土 —— "铜价涨"会被落到整个筐，
推荐的可能是黄金股。要做到"铜价涨 → 紫金/江铜/铜陵"，中间缺的正是
「概念/产业链」这一层。

数据源（已实测，见 `scripts/probe_concept_source.py`）
----------------------------------------------------
东财 `datacenter-web.eastmoney.com` 的 `RPT_F10_CORETHEME_BOARDTYPE`
是「股票 × 题材板块」全量映射：**99,479 行 / 1,042 个概念 / 20 页 ≈ 74s**，
字段含 `SECURITY_CODE` / `BOARD_NAME` / `NEW_BOARD_CODE`。

⚠️ 为什么不用 akshare 的 `stock_board_concept_*_em`：那条路走
`push2.eastmoney.com`，本机链路上 **TCP 能握手但 HTTP 被 RST**
（`RemoteDisconnected`），而 `datacenter-web` 同属东财却正常。
"TCP 通 ≠ 接口可用"这条教训就来自这里。

成本与可靠性设计（沿用 `data/industry_members.py` 的同一套语义）
--------------------------------------------------------------
  - 全量一次 ≈ 74s / 20 次请求 → **落盘缓存**（`data/concept_members.json`，
    默认 7 天内直接读缓存；概念归属变化远慢于行情，没必要天天拉）；
  - 超时 + 失败**不抛给调用方**而是返回 `degraded` 标记（缺就是缺，不补造）；
  - 缓存过期但重拉失败 → 返回**陈旧缓存** + `degraded=True` + `fetch_error`；
  - 完全无数据 → 抛 `ConceptMembersUnavailable`，由调用方决定降级方式。
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime

_logger = logging.getLogger(__name__)

# ⚠️⚠️ 同 `industry_members`：缓存路径必须走 `data_root()`。
#   frozen（exe）下 `__file__` 在只读的 `_internal/`，用 `dirname(__file__)`
#   ⇒ exe 里读不到这份缓存，概念优先分支静默失效（表现为"永远 industry 兜底"）。
from config.paths import data_root as _paths_data_root   # noqa: E402

_PROJECT_ROOT = _paths_data_root()
CACHE_PATH = os.path.join(_PROJECT_ROOT, "data", "concept_members.json")
# 概念归属（谁属于"铜"概念）是按季度级别变化的静态事实，7 天足够新鲜
CACHE_TTL_HOURS = 24 * 7
REQUEST_TIMEOUT = 30
PAGE_SIZE = 5000            # 实测上限：5000 行/页 → 99,479 行共 20 页（74s）

API_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REPORT_NAME = "RPT_F10_CORETHEME_BOARDTYPE"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


class ConceptMembersUnavailable(RuntimeError):
    """概念成分完全不可用（无缓存且拉取失败 / 离线）。"""


def _is_a_share(code: str) -> bool:
    """6 位 A 股代码（沪深主板/创业板/科创板/北交所），剔除 B 股。

    ⚠️ 与 `core.scheduler.is_a_share_code` 同口径，但不 import 它：
    本模块被 `core.news_industry_review` 调用，而后者又被 `core.scheduler`
    在模块级导入 —— 反向 import 会成环。**口径一致性由测试守护**
    （`tests/test_concept_members.py::test_code_filter_matches_scheduler`）。
    """
    c = str(code or "").strip()
    if len(c) != 6 or not c.isdigit():
        return False
    if c.startswith(("900", "200")):            # 沪/深 B 股
        return False
    return c.startswith(("60", "68", "000", "001", "002", "003",
                         "300", "301", "8", "4", "920"))


def _fetch_all() -> dict:
    """联网拉全量「概念 → 成分股」。抛异常 = 彻底失败。"""
    from data import offline_gate

    if offline_gate.is_offline():
        # 离线闸门：宁可不给数据，也不能悄悄外呼（`blocked()/socket_attempts()` 要能守住）
        raise ConceptMembersUnavailable("离线模式（BIGA_OFFLINE=1）不联网拉概念成分")

    import requests

    concepts: dict[str, dict] = {}
    failures: list[str] = []
    page, total_pages = 1, 1
    while page <= total_pages:
        try:
            r = requests.get(
                API_URL,
                params={"reportName": REPORT_NAME, "columns": "ALL",
                        "pageSize": PAGE_SIZE, "pageNumber": page},
                headers={"User-Agent": UA}, timeout=REQUEST_TIMEOUT)
            payload = r.json()
        except Exception as e:                  # noqa: BLE001 — 单页失败要能定位到页码
            _logger.warning("[概念成分] 第 %d 页拉取失败: %s: %s", page, type(e).__name__, e)
            failures.append(f"page{page}({type(e).__name__})")
            page += 1
            continue

        result = payload.get("result") or {}
        rows = result.get("data") or []
        if not rows and page == 1:
            raise ConceptMembersUnavailable(
                f"接口返回空表（HTTP {r.status_code}，reportName 可能改名）")
        total_pages = int(result.get("pages") or total_pages)
        for x in rows:
            name = (x.get("BOARD_NAME") or "").strip()
            code = (x.get("SECURITY_CODE") or "").strip()
            if not name or not _is_a_share(code):
                continue
            slot = concepts.setdefault(name, {"board_code": "", "codes": []})
            if not slot["board_code"]:
                slot["board_code"] = (x.get("NEW_BOARD_CODE") or "").strip()
            slot["codes"].append(code)
        page += 1

    if not concepts:
        raise ConceptMembersUnavailable(f"全部页面无有效数据: {failures[:3]}")

    # 去重（同一行可能因分页边界重复），并保持稳定顺序便于 diff
    for slot in concepts.values():
        slot["codes"] = sorted(set(slot["codes"]))
    return {"concepts": concepts, "failures": failures}


def get_concept_members(force: bool = False,
                        max_age_hours: int = CACHE_TTL_HOURS) -> dict:
    """返回 {"concepts": {概念: {"board_code": str, "codes": [code]}},
             "fetched_at": str, "degraded": bool, "failures": [str], "age_hours": float}

    - `degraded=True`：数据是**陈旧缓存**（本次拉取失败）或**部分页缺失**——
      调用方必须把这个标记带给用户（缺就是缺，不能装作完整）。
    - 完全无数据 → 抛 `ConceptMembersUnavailable`。
    """
    cached, age = _load_cache()
    if cached is not None and not force and age <= max_age_hours:
        out = dict(cached)
        out["age_hours"] = round(age, 2)
        out["degraded"] = bool(cached.get("failures"))
        return out

    try:
        fresh = _fetch_all()
    except Exception as e:                      # noqa: BLE001
        if cached is not None:
            _logger.warning("[概念成分] 拉取失败(%s)，退回 %.1f 小时前的缓存", e, age)
            out = dict(cached)
            out["age_hours"] = round(age, 2)
            out["degraded"] = True
            out["fetch_error"] = f"{type(e).__name__}: {e}"
            return out
        raise

    payload = {"concepts": fresh["concepts"], "failures": fresh["failures"],
               "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    _save_cache(payload)
    return {**payload, "age_hours": 0.0, "degraded": bool(fresh["failures"])}


def load_cached_only() -> dict:
    """**只读磁盘缓存，绝不联网**（Web 只读端点专用）。

    全量拉一次 20 页、约 52~74 秒 —— 放进 `/api/*` 请求里会变成不可预测的长尾延迟，
    并且让"打开页面"触发一轮上游。没有缓存 → 抛 `ConceptMembersUnavailable`。
    """
    cached, age = _load_cache()
    if cached is None:
        raise ConceptMembersUnavailable("无磁盘缓存（只读模式不联网刷新）")
    out = dict(cached)
    out["age_hours"] = round(age, 2)
    out["read_only"] = True
    out["degraded"] = bool(cached.get("failures")) or age > CACHE_TTL_HOURS
    return out


def search_concepts(keyword: str, concepts: dict) -> list[str]:
    """按关键词模糊找概念名（给人核对别名表用，不参与选股判定）。

    ⚠️ 不要用它直接选股：`铜` 会同时命中「铜」和「铜缆高速连接」，
    后者是通信概念、与铜价无关。选股走显式别名表（见
    `core.news_industry_review.THEME_TO_CONCEPTS`）。
    """
    kw = (keyword or "").strip()
    if not kw:
        return []
    return sorted(n for n in (concepts or {}) if kw in n)


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
        _logger.warning("[概念成分] 缓存损坏，忽略重建: %s", e)
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
        except PermissionError as e:
            # Windows 上目标被占用是常态（杀软/预览进程），退避重试即可；
            # 但要留痕，否则"写了没落盘"会变成无声故障
            _logger.debug("[概念成分] 缓存被占用，第 %d 次重试: %s", attempt + 1, e)
            time.sleep(0.5 * (attempt + 1))
    _logger.warning("[概念成分] 缓存写入失败（文件被占用），本次结果未落盘")
