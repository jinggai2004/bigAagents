# -*- coding: utf-8 -*-
"""全量域（Full Universe）—— 全市场截面，2026-10-26。

## 为什么需要它
此前 `market_snapshot` 的唯一来源是**新浪 49 个行业明细**（`sina_source`），
实测覆盖率只有 **47%**：主板沪 57% / 深 75% / **创业板 31% / 科创板 5%**。
`300750 宁德时代`、`688111 金山办公` 这些主流票**根本不在选股域里** ——
这是"候选长期偏主板"的根因，也是涨跌家数常年偏小的一半原因。

根因不是新浪不给，而是**新浪的行业分类本身只覆盖 3,038 只**（`板块`表里
`公司家数` 求和即为 3,038），剩下 2,500 多只创业板/科创板票在 49 行业体系里
没有归属。所以"多抓几个行业"永远补不齐，必须换口径。

## 怎么做（三源合并，全部免权限）
| 字段 | 来源 | 覆盖 |
|---|---|---|
| 代码/名称/现价/涨跌幅/成交额/成交量 | 新浪 `Market_Center.getHQNodeData?node=hs_a` | **≈5,576（全部 A 股）** |
| PE / PB / 总市值 / 流通市值 / 换手率 | 同上（`per/pb/mktcap/nmc/turnoverratio`） | ≈5,576 |
| 行业 / 板块 | tushare `stock_basic`（**1 次/分钟** ⇒ 必须落盘缓存 7 天） | ≈5,575 |

⚠️ 为什么不用 `tushare daily_basic` 拿估值：**无权限**（120 积分档只有非复权日线），
   能力表已记「被拒 3 次、成功 0 次」；本模块因此**完全不碰它**。

⚠️ 为什么不用东财 `stock_zh_a_spot_em`：实测 `82.push2.eastmoney.com` 连接层全断
   （`eastmoney_source` 已有熔断），不能作为唯一估值源。

## 单位（本项目最经典的一类坑）
- 新浪 `amount` = **元**、`volume` = **股**、`mktcap`/`nmc` = **万元** ⇒ 落库统一为**元**。
- 与 `tushare.daily`（vol=手、amount=千元）**不是一个量级**，混用会让"成交额排序"
  整体差 1000 倍。本模块只做新浪口径，并在 `attrs["unit"]` 里写明。

## 铁律
- **`trade<=0` 的行不能写**（停牌/退市 ⇒ 没有报价，不是"0 元"）。
- 条数低于 `UNIVERSE_MIN` 视为采残 ⇒ **抛错，不落库**（半份截面比采不到更危险）。
- 覆盖率必须自报：`covered / universe` 一起返回，页面才可能诚实显示。
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

from data import sina_source

_logger = logging.getLogger(__name__)

# 新浪全市场列表节点（沪深京 A 股）
_NODE = "hs_a"
_PER_PAGE = 100                 # 服务端硬上限：给更大也只返 100
_API = ("http://vip.stock.finance.sina.com.cn/quotes_service/api/"
        "json_v2.php/Market_Center.getHQNodeData")
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "http://finance.sina.com.cn"}

# 条数下限：A 股 ~5,570 只，低于 4,000 一定是采残（限流/半途失败）
UNIVERSE_MIN = 4000

# stock_basic 磁盘缓存（1 次/分钟限频 ⇒ 靠缓存把调用降到每周 1 次）
_BASIC_TTL = 7 * 86400

_requests = requests                    # 测试注入点（与 sina_source 一致）


def _basic_cache_paths() -> tuple[str, str]:
    """(首选, 兜底) —— **不猜 `__file__`**（打包后指向只读 `_internal/`）。

    ⚠️ `BIGA_STOCK_BASIC_CACHE` 是给**测试**用的逃生口：单测若真的调用
       `stock_basic_map()`（哪怕只是顺带），会把 mock 出来的 1 行清单写进真实缓存，
       下一次真跑就以为"清单只有 1 只"。`tests/conftest.py` 已把这里指到临时目录，
       这个环境变量是第二道保险。
    """
    env = os.environ.get("BIGA_STOCK_BASIC_CACHE")
    if env:
        return env, env + ".bak"
    paths = []
    try:
        from config.paths import data_root
        paths.append(os.path.join(str(data_root()), "stock_basic_cache.json"))
    except Exception as e:                                # noqa: BLE001
        _logger.warning("[全量域] data_root 不可用（缓存退到 data/）: %s", e)
    paths.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "stock_basic_cache.json"))
    return paths[0], paths[1]


def _load_basic_cache(now: float | None = None) -> dict | None:
    import time as _t
    now = now if now is not None else _t.time()
    for p in _basic_cache_paths():
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if now - float(d.get("fetched_at") or 0) > _BASIC_TTL:
                return None
            m = d.get("map") or {}
            return m or None
        except FileNotFoundError:
            _logger.debug("[全量域] stock_basic 缓存不存在（尝试下一个路径）: %s", p)
            continue
        except Exception as e:                            # noqa: BLE001
            _logger.warning("[全量域] stock_basic 缓存读失败 %s: %s", p, e)
    return None


def _save_basic_cache(mapping: dict) -> None:
    import time as _t
    payload = {"fetched_at": _t.time(), "count": len(mapping), "map": mapping}
    for p in _basic_cache_paths():
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, p)
            return
        except Exception as e:                            # noqa: BLE001
            _logger.warning("[全量域] stock_basic 缓存落盘失败 %s: %s", p, e)


def stock_basic_map(*, force: bool = False) -> tuple[dict, str]:
    """`{code6: {"name","industry","market"}}` + 来源标记（`cache` / `tushare` / `empty`）。

    ⚠️ `stock_basic` 是 **1 次/小时**（服务端原文如此，实测连撞 3 次 40203），
       且每天还有总量上限 ⇒ 必须缓存。
       缓存过期时**先试网络，失败就用过期缓存**（行业名一年也变不了几次，
       用 8 天前的行业名 ≫ 全部标"未知"）。
    """
    if not force:
        cached = _load_basic_cache()
        if cached:
            return cached, "cache"
    try:
        from data import tushare_source as ts
        df = ts.fetch_stock_basic()
        mapping = {}
        for r in df.to_dict("records"):
            code = str(r.get("symbol") or "").strip()
            if not code:
                code = str(r.get("ts_code") or "").split(".")[0]
            if not code:
                continue
            mapping[code] = {"name": str(r.get("name") or ""),
                             "industry": str(r.get("industry") or ""),
                             "market": str(r.get("market") or "")}
        if mapping:
            _save_basic_cache(mapping)
            return mapping, "tushare"
        return {}, "empty"
    except Exception as e:                                # noqa: BLE001
        _logger.warning("[全量域] stock_basic 拉取失败（%s: %s）⇒ 用过期缓存",
                        type(e).__name__, str(e)[:120])
        stale = _load_basic_cache(now=0.0)                # 忽略 TTL
        return (stale or {}), ("cache_stale" if stale else "empty")


# ---------------------------------------------------------------- 新浪全市场
def _f(v):
    """数字字符串 → float；`''`/`None`/`'--'` ⇒ None（缺就是缺，不造 0）。"""
    try:
        if v is None or v == "" or v == "--":
            return None
        x = float(v)
        return None if math.isnan(x) or math.isinf(x) else x
    except (TypeError, ValueError):
        _logger.debug("[全量域] 数值无法解析（%r）⇒ None", v)
        return None


def _fetch_page(page: int, *, timeout: float, retries: int = 2) -> list:
    params = {"page": page, "num": _PER_PAGE, "sort": "symbol", "asc": 1,
              "node": _NODE, "symbol": "", "_s_r_a": "page"}
    for _ in range(max(1, retries)):
        try:
            r = _requests.get(_API, params=params, headers=_HEADERS, timeout=timeout)
            txt = (r.text or "").strip()
            if txt.startswith("["):
                return json.loads(txt) or []
        except Exception as e:                            # noqa: BLE001 - 单页失败重试
            _logger.debug("[全量域] 第 %d 页失败(%s: %s)，重试", page,
                          type(e).__name__, str(e)[:80])
    return []


def fetch_universe_quotes(*, timeout: float = 15.0, workers: int = 8) -> list[dict]:
    """全市场 A 股快照（新浪 `hs_a` 逐页并发）。

    Returns: `[{code,name,price,change_pct,volume,amount,pe,pb,market_cap,
               float_cap,turnover,board}]`（单位：元 / 股 / 百分数）
    """
    first = _fetch_page(1, timeout=timeout)
    if not first:
        raise RuntimeError("新浪全市场列表第 1 页无数据（接口不可用）")

    pages = 60                      # 5,576 / 100 ≈ 56；留余量
    rows: list[dict] = []
    items = [first]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for arr in ex.map(lambda p: _fetch_page(p, timeout=timeout),
                          range(2, pages + 1)):
            items.append(arr)

    seen = set()
    dropped_non_a = 0
    for arr in items:
        for it in arr or []:
            code = sina_source._clean_code(it.get("code") or it.get("symbol"))
            if not code or code in seen:
                continue
            if not _is_a_share(code):                     # 结构性剔除 B 股/基金/债券
                dropped_non_a += 1
                continue
            price = _f(it.get("trade"))
            if price is None or price <= 0:               # 停牌/退市：没有报价
                continue
            seen.add(code)
            rows.append({
                "code": code,
                "name": str(it.get("name") or "").strip(),
                "price": price,
                "change_pct": _f(it.get("changepercent")),
                "volume": _f(it.get("volume")),           # 股
                "amount": _f(it.get("amount")),           # 元
                "pe": _f(it.get("per")),
                "pb": _f(it.get("pb")),
                "market_cap": (None if _f(it.get("mktcap")) is None
                               else _f(it.get("mktcap")) * 1e4),   # 万元 → 元
                "float_cap": (None if _f(it.get("nmc")) is None
                              else _f(it.get("nmc")) * 1e4),       # 万元 → 元
                "turnover": _f(it.get("turnoverratio")),  # 百分数
                "board": sina_source._board_of(code),
            })
    if dropped_non_a:
        _logger.info("[全量域] 结构性剔除 %d 只非 A 股代码（B 股/基金/债券）", dropped_non_a)
    return rows


def _is_a_share(code: str) -> bool:
    """6 位代码是否 A 股（沪深主板/创业板/科创板/北交所）。

    ⚠️ 判据必须**结构性**（按代码段），不能"不在清单里就丢"——清单可能拿不到。
    与 `core.scheduler.is_a_share_code` 逐字一致，由
    `tests/test_full_market.py::test_is_a_share_matches_core` 守护（同 `_board_of` 的写法）。
    """
    c = str(code or "").strip()
    if len(c) != 6 or not c.isdigit():
        return False
    if c.startswith(("900", "200")):                      # B 股
        return False
    return c.startswith(("60", "68", "000", "001", "002", "003",
                         "300", "301", "8", "4", "920"))


# ---------------------------------------------------------------- 落库
def build_rows(trade_date: str | None = None, *,
               quotes: list[dict] | None = None) -> tuple[list[dict], dict]:
    """构造可直接喂 `store.save_market_snapshot` 的行 + attrs。

    ⚠️ 只保留**新浪清单内的 A 股**（结构性剔除 B 股/退市），与 `sina_source`
       的 `is_a_share_code` 口径一致；行业/板块由 `stock_basic` 补。
    """
    q = quotes if quotes is not None else fetch_universe_quotes()
    if len(q) < UNIVERSE_MIN:
        raise RuntimeError(f"全市场截面只有 {len(q)} 只（< {UNIVERSE_MIN}），"
                           f"判定为采残，拒绝落库")

    basic, basic_src = stock_basic_map()
    # ── 行业两套口径，**分列存储**（2026-10-29，铁律"行业口径不可混"）──
    #   `industry`    = 新浪 49 类（历史映射兜底，供新闻荐股匹配 `SECTOR_TO_SINA_INDUSTRY`）
    #   `industry_ts` = tushare 111 类（stock_basic，全市场 5,577 只，供选股配额）
    # ⚠️ 以前这里把 tushare 行业填进 `industry`，于是"tushare 可用时是新口径、
    #    限频时退回新浪口径"——**同一列两种口径取决于网络**，正是最隐蔽的漂移。
    fallback_ind: dict = {}
    try:
        from db import store
        fallback_ind = store.latest_industry_map()
        if fallback_ind:
            _logger.info("[全量域] 新浪行业用历史截面映射（%d 只）", len(fallback_ind))
    except Exception as e:                            # noqa: BLE001
        _logger.warning("[全量域] 新浪行业兜底映射读取失败: %s", e)

    rows, no_ind = [], 0
    for r in q:
        code = r["code"]
        meta = basic.get(code) or {}
        ind = fallback_ind.get(code) or ""            # 新浪口径（老股才有）
        ind_ts = meta.get("industry") or ""           # tushare 口径（全市场）
        if not ind_ts:
            no_ind += 1
        rows.append({
            "stock_code": code,
            "stock_name": meta.get("name") or r.get("name") or "",
            "industry": ind,
            "industry_ts": ind_ts,
            "board": r.get("board") or "",
            "price": r.get("price"),
            "change_pct": r.get("change_pct"),
            "pe": r.get("pe"),
            "pb": r.get("pb"),
            "market_cap": r.get("market_cap"),
            "float_cap": r.get("float_cap"),
            "turnover": r.get("turnover"),
            "amount": r.get("amount"),
        })

    universe = None
    try:
        codes = sina_source.market_codes()
        universe = len(codes) if codes else None
    except Exception as e:                                # noqa: BLE001
        _logger.warning("[全量域] 取 A 股清单失败（覆盖率分母缺失）: %s", e)

    covered = len(rows)
    attrs = {
        "source": "sina_full",
        "covered": covered,
        "universe": universe,
        "coverage": round(covered / universe, 4) if universe else None,
        "industry_missing": no_ind,
        "industry_covered": covered - no_ind,
        "stock_basic_source": basic_src,
        "unit": "price=元 amount=元 market_cap=元 volume=股 turnover=%",
        "pe_cov": sum(1 for r in rows if r.get("pe") is not None),
        "degraded": ([] if no_ind < covered * 0.05
                     else [f"{no_ind}/{covered} 只缺 tushare 行业"
                           f"（stock_basic={basic_src}）"]),
    }
    return rows, attrs


def save_full_snapshot(trade_date: str, *, quotes: list[dict] | None = None) -> dict:
    """采集 + 落库全市场截面。返回 `{ok, snapshot_date, rows, coverage, ...}`。"""
    from db import store

    try:
        rows, attrs = build_rows(trade_date, quotes=quotes)
    except Exception as e:                                # noqa: BLE001
        _logger.warning("[全量域] 采集失败: %s: %s", type(e).__name__, str(e)[:200])
        return {"ok": False, "snapshot_date": trade_date, "rows": 0,
                "reason": f"{type(e).__name__}: {e}"}

    written = store.save_market_snapshot(trade_date, rows, source="sina_full")
    try:
        store.save_data_coverage(
            trade_date, "market_snapshot", got=int(written or 0),
            total=attrs.get("universe"),
            status="ok" if not attrs.get("degraded") else "degraded",
            source="sina_full",
            note=f"行业缺 {attrs.get('industry_missing')} 只；"
                 f"PE 覆盖 {attrs.get('pe_cov')}")
    except Exception as e:                                # noqa: BLE001
        _logger.warning("[全量域] 覆盖率落库失败（不影响快照）: %s", e)

    _logger.info("[全量域] %s 落库 %s 行（覆盖 %s/%s，行业缺 %s，stock_basic=%s）",
                 trade_date, written, attrs.get("covered"), attrs.get("universe"),
                 attrs.get("industry_missing"), attrs.get("stock_basic_source"))
    return {"ok": bool(written), "snapshot_date": trade_date, "rows": written,
            **attrs}


# ---------------------------------------------------------------- 运维 CLI
def main(argv: list[str] | None = None) -> int:
    """`python -m data.full_market [--refresh-basic] [--snapshot [YYYYMMDD]]`

    存在的理由：`stock_basic` 是 **1 次/小时** 硬限频，撞上 40203 时
    当天全量域只能带着 `degraded` 落库。补行业不该去改代码，也不该
    等到明天 —— 给一个一行的补数入口。

    ⚠️ 输出走 `_logger`（不是 `print`）：`data/` 层禁止 `__main__` 块外的 print
       （`test_observability_hygiene` 会拦），而 CLI 也要能进日志文件、
       能在定时任务里被 `--log` 收走。`__main__` 块里配 `basicConfig` 即可。
    """
    import argparse

    ap = argparse.ArgumentParser(prog="data.full_market", description="全量域运维")
    ap.add_argument("--refresh-basic", action="store_true",
                    help="强制拉一次 stock_basic 行业表并落盘缓存（1 次/小时限频）")
    ap.add_argument("--snapshot", nargs="?", const="today", default=None,
                    metavar="YYYYMMDD",
                    help="采集并落库当日全量截面（默认今天）")
    a = ap.parse_args(argv)

    if not (a.refresh_basic or a.snapshot):
        ap.print_help()
        return 2

    rc = 0
    if a.refresh_basic:
        m, src = stock_basic_map(force=True)
        ok = src == "tushare" and bool(m)
        _logger.info("stock_basic: source=%s rows=%d%s", src, len(m),
                     "" if ok else "  ← 失败/限频，缓存未更新")
        rc = 0 if ok else 1

    if a.snapshot:
        d = a.snapshot
        if d in ("today", "", None):
            from data.trading_calendar import last_trading_day as _ltd
            d = _ltd(include_today=True) or ""
        elif len(d) == 8 and d.isdigit():
            # 跨源日期口径唯一：YYYYMMDD 混进 YYYY-MM-DD 会让按日过滤双向失效
            d = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        res = save_full_snapshot(d)
        _logger.info("snapshot %s: ok=%s rows=%s 覆盖=%s/%s 行业缺=%s degraded=%s",
                     d, res.get("ok"), res.get("rows"), res.get("covered"),
                     res.get("universe"), res.get("industry_missing"),
                     res.get("degraded"))
        rc = 0 if res.get("ok") else 1
    return rc


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(message)s")
    raise SystemExit(main())
