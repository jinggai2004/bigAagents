# -*- coding: utf-8 -*-
"""情绪面选股：新闻 → 个股的**严匹配**（P1-5，2026-10-23）。

## 为什么需要它
`news_items.related_codes` 只关联**关注池**（自选股 ∪ 当日候选 ∪ 持仓），
所以"今天哪只票被新闻点名、但还没进池子"这件事**系统是看不见的** ——
而"从当日利好新闻发现新标的"正是用户明确要的能力。

## 为什么叫"严匹配"
宽匹配（拿关键词表去扫正文）实测每天能出 100+ 只，噪声压倒信号。
严匹配只用两条**结构性**证据：
  1. **六位代码**出现且能在当日全市场清单里查到；
  2. **完整股票简称**（≥3 字）出现在标题或正文里。
并且：
  - 长名优先、命中即"遮蔽"该片段（避免"中国石油"同时命中"石油"类短名）；
  - 剔除名称含"退"的退市股（不可交易，进池只会白跑）；
  - 名称与正文都先**去掉空白**再比（面板里存在"南 京 港"这种带空格的名字）。
实测（2026-10-23 的 332 条快讯）：命中 37~39 只/日、耗时 0.1~0.3s。

## 增量与累加
快讯每小时采一轮。这里按 `news_items.id` 增量处理（状态存 `temp/`），
命中的计数走 `store.upsert_news_stock_hits` **累加**——
否则"全天被提 5 次"会被最后一轮压成 1 次。
"""
from __future__ import annotations

import json
import logging
import os
import re

from db import store

_logger = logging.getLogger(__name__)

_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "temp", ".news_stock_hits_state.json")

_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")

# 语境线索（"严匹配"的第二道闸）：简称命中必须在**附近**出现这些词之一，
# 否则就只是"这个词刚好出现在句子里"。
# ⚠️⚠️ 为什么必须加（2026-10-23 实测）：不加时 143 只/日，榜首是
#   `300024 机器人`（17 次）——因为任何讲"人形机器人板块"的新闻都在提它；
#   `000402 金融街` 同理（地名/板块名）。这类误命中会让"情绪面选股"直接跑偏。
_CUE_WORDS = (
    "涨停", "跌停", "涨超", "跌超", "涨幅", "跌幅", "上涨", "下跌", "走强", "走弱",
    "股价", "收盘", "开盘", "盘中", "公告", "披露", "净利", "营收", "业绩", "预增",
    "中标", "签约", "回购", "增持", "减持", "停牌", "复牌", "定增", "重组", "并购",
    "净流入", "净流出", "主力", "调研", "评级", "目标价", "买入", "研报", "机构",
    "发布", "推出", "量产", "订单", "投产", "扩产", "合作", "获批", "上市",
)
_CUE_WINDOW = 18          # 线索词与简称的最大距离（字符）

# "板块/行业"后缀：紧跟在名称后面的这些词说明讲的是**板块**不是这只票。
_SECTOR_SUFFIX = ("板块", "行业", "产业", "概念", "指数", "主题", "赛道", "ETF")

# 与普通名词同形的简称：**只认代码**（名称命中必须伴随六位代码）。
# ⚠️ 实测不加这条时 `300024 机器人` 全天 15 次居首 —— "人形机器人板块涨超15%"
#    里线索词与名称只差几个字，靠距离窗口拦不住。
_AMBIGUOUS_NAMES = {"机器人", "指南针", "值得买", "三人行", "每日互动", "金融街"}
_MIN_NAME_LEN = 3          # 2 字简称与常用词撞车太狠（"上海""中国""东方"…）⇒ 一律不认
_SAMPLE_N = 3              # 每只票留几条标题样例，够人核对即可


def _norm(text: str) -> str:
    """去掉所有空白：面板里存在 "南 京 港" 这类带空格的名字。"""
    return re.sub(r"\s+", "", text or "")


_NAME_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "stock_names.json")
_NAME_TTL_DAYS = 7


def _load_name_cache() -> dict | None:
    try:
        with open(_NAME_CACHE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        # ⚠️ 缓存缺失/损坏是**正常状态**（首次运行、写到一半被截断）⇒ debug 即可，
        #    但不能静默：否则"名称表一直是空的"这种问题无从追查。
        _logger.debug("[新闻命中] 名称缓存不可用（视为未缓存）: %s: %s", type(e).__name__, e)
        return None


def ensure_name_cache(force: bool = False) -> dict:
    """确保全市场名称表可用（盘缓存 7 天，过期才联网）。

    ⚠️⚠️ 为什么**不能**用 `market_snapshot` 当名称表（2026-10-23 实测）：
      快照只有 **2902/2995** 只（新浪行业口径），创业板只覆盖 449 只 ——
      连 **宁德时代(300750)** 都不在里面。拿它做"新闻→个股"的字典，
      等于把最常被新闻点名的龙头股**系统性漏掉**，而漏掉是静默的。
      实测 `akshare.stock_info_a_code_name()` 能拿到 **5577 只**（40s，18 页）。
    ⚠️ `tushare stock_basic` 是 **5 次/天**的硬限频，不能当日常刷新手段。
    """
    import time
    cache = _load_name_cache()
    if cache and not force:
        age_days = (time.time() - float(cache.get("fetched_at") or 0)) / 86400
        if age_days <= _NAME_TTL_DAYS and cache.get("names"):
            return cache
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        names = {}
        codes = set()
        for _, r in df.iterrows():
            code = str(r.get("code") or "").strip()
            name = _norm(str(r.get("name") or ""))
            if not code:
                continue
            codes.add(code)
            if len(name) >= _MIN_NAME_LEN and "退" not in name:
                names[name] = code
        cache = {"fetched_at": time.time(), "source": "akshare.stock_info_a_code_name",
                 "count": len(codes), "names": names, "codes": sorted(codes)}
        os.makedirs(os.path.dirname(_NAME_CACHE), exist_ok=True)
        with open(_NAME_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        _logger.info("[情绪面] 名称表刷新：%d 只（%s）", len(codes), cache["source"])
        return cache
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[情绪面] 名称表联网刷新失败，退回上次缓存/快照: %s", e)
        if cache and cache.get("names"):
            return cache
        return _snapshot_fallback()


def _snapshot_fallback() -> dict:
    """最后兜底：用当日快照（覆盖不全，必须让调用方知道）。"""
    _, rows = store.load_market_snapshot(None)
    names, codes = {}, set()
    for r in rows or []:
        code = str(r.get("stock_code") or "").strip()
        name = _norm(r.get("stock_name") or "")
        if not code:
            continue
        codes.add(code)
        if len(name) >= _MIN_NAME_LEN and "退" not in name:
            names[name] = code
    return {"fetched_at": 0, "source": "market_snapshot(fallback, 覆盖不全)",
            "count": len(codes), "names": names, "codes": sorted(codes)}


def build_name_index(trade_date: str = None) -> tuple[dict, set]:
    """全市场 `{简称: 代码}` + 代码集合（盘缓存优先，见 `ensure_name_cache`）。"""
    cache = ensure_name_cache()
    return cache.get("names") or {}, set(cache.get("codes") or [])


def match_text(text: str, names: dict, codes: set) -> list[tuple[str, str, str]]:
    """返回 `[(code, name, matched_token)]`，同一只票只出现一次。

    遮蔽策略：按名称长度**降序**匹配，命中后把该片段替换成占位符，
    这样"宁德时代"不会被更短的"宁德"再命中一次（反之亦然）。
    """
    flat = _norm(text)
    if not flat:
        return []
    hits: dict[str, tuple[str, str, str]] = {}

    # ① 代码（结构性证据最强，命中即可）
    for m in _CODE_RE.finditer(flat):
        code = m.group(1)
        if code in codes and code not in hits:
            hits[code] = (code, "", code)

    # ② 完整简称（长名优先 + 遮蔽 + 语境线索）
    for name in sorted(names, key=len, reverse=True):
        pos = flat.find(name)
        if pos < 0:
            continue
        # 只看**第一次出现**的位置判断语境：已经在讲股价/公告的句子才认，
        # 纯行业讨论（"人形机器人板块持续拉升"里提到"机器人"这只票）不算。
        code = names[name]
        tail = flat[pos + len(name): pos + len(name) + 4]
        if any(tail.startswith(x) for x in _SECTOR_SUFFIX):
            flat = flat.replace(name, "\u0000")     # 讲的是板块，不是这只票
            continue
        if name in _AMBIGUOUS_NAMES and code not in hits:
            # 同名普通词：必须有六位代码佐证（`hits` 里 code 已由代码分支写入）
            flat = flat.replace(name, "\u0000")
            continue
        left = max(0, pos - _CUE_WINDOW)
        near = flat[left:pos + len(name) + _CUE_WINDOW]
        if not any(w in near for w in _CUE_WORDS):
            # 没有语境线索 ⇒ 只遮蔽不采信（避免同一只票被反复计）
            flat = flat.replace(name, "\u0000")
            continue
        if code not in hits:
            hits[code] = (code, name, name)
        flat = flat.replace(name, "\u0000")         # 遮蔽，避免子串重复命中
    return list(hits.values())


def _read_state() -> dict:
    try:
        with open(_STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        _logger.debug("[新闻命中] 状态文件不可用（按空状态处理）: %s: %s", type(e).__name__, e)
        return {}


def _write_state(d: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f)
    except OSError as e:
        _logger.warning("[情绪面] 进度写入失败（会导致重复处理）: %s", e)


def refresh(limit: int = 2000, trade_date: str = None) -> dict:
    """增量处理新入库的快讯，把命中累加进 `news_stock_hits`。

    Args:
        limit: 单次最多处理多少条（防某天一次性灌入撑爆内存/时间）

    Returns: `{processed, hits, stocks, last_id, skipped_reason}`
    """
    state = _read_state()
    last_id = int(state.get("last_id") or 0)
    conn = store.get_connection()
    try:
        rows = conn.execute(
            """SELECT id, title, content, source, sentiment, created_at
               FROM news_items WHERE id > ? ORDER BY id LIMIT ?""",
            (last_id, int(limit))).fetchall()
    finally:
        conn.close()
    if not rows:
        return {"processed": 0, "hits": 0, "stocks": 0, "last_id": last_id,
                "skipped_reason": None}

    names, codes = build_name_index(trade_date)
    if not codes:
        return {"processed": 0, "hits": 0, "stocks": 0, "last_id": last_id,
                "skipped_reason": "当日快照缺失 ⇒ 无法建全市场名称表"}

    # 按 (交易日, 代码) 聚合本轮
    agg: dict[tuple[str, str], dict] = {}
    for r in rows:
        day = str(r["created_at"] or "")[:10]
        if not day:
            continue
        text = f"{r['title'] or ''} {r['content'] or ''}"
        senti = str(r["sentiment"] or "neutral")
        for code, name, _tok in match_text(text, names, codes):
            key = (day, code)
            a = agg.setdefault(key, {"trade_date": day, "stock_code": code,
                                     "stock_name": name or None, "hits": 0,
                                     "positive": 0, "negative": 0,
                                     "first_at": r["created_at"], "last_at": r["created_at"],
                                     "_sources": set(), "samples": []})
            a["hits"] += 1
            if senti == "positive":
                a["positive"] += 1
            elif senti == "negative":
                a["negative"] += 1
            a["last_at"] = r["created_at"]
            if r["source"]:
                a["_sources"].add(str(r["source"]))
            if len(a["samples"]) < _SAMPLE_N and (r["title"] or "").strip():
                a["samples"].append(str(r["title"]).strip()[:80])

    out_rows = []
    for a in agg.values():
        a["sources"] = ",".join(sorted(a.pop("_sources"))) or None
        a["samples"] = json.dumps(a["samples"], ensure_ascii=False)
        out_rows.append(a)

    written = store.upsert_news_stock_hits(out_rows) if out_rows else 0
    new_last = max(int(r["id"]) for r in rows)
    _write_state({"last_id": new_last,
                  "updated_at": _now()})
    _logger.info("[情绪面] 处理 %d 条快讯 → 命中 %d 只（%d 组聚合），last_id=%d",
                 len(rows), len({k[1] for k in agg}), written, new_last)
    return {"processed": len(rows), "hits": sum(a["hits"] for a in agg.values()),
            "stocks": len({k[1] for k in agg}), "last_id": new_last,
            "skipped_reason": None}


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def top_hits(trade_date: str, limit: int = 20, min_hits: int = 1) -> list[dict]:
    """当日被新闻点名最多的个股（供选股/简报使用）。"""
    return store.load_news_stock_hits(trade_date, limit=limit, min_hits=min_hits)
