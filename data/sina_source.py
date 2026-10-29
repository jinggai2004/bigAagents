"""新浪行业快照 + 个股日线（选股执行层的本地数据通路）

为什么需要
----------
NL 选股执行层原先依赖问财初筛，本地拿不到 4 个指标（总市值 / 换手率 / 成交额 /
当日涨跌幅），报告里只能标"由问财初筛保证"。

2026-10-07 实测：东财全通路被代理拦截（`get_market_snapshot()` → None 连续 5 页失败、
`stock_zh_a_hist` → ProxyError `push2his.eastmoney.com`），而**新浪行业明细一次给出
全部所需标量字段**，且不依赖东财 —— 两条通路互为备份，**方案不能建立在单一上游上**。

覆盖域诚实原则（**必须随结果一起展示，不得宣称"全市场"**）
--------------------------------------------------------
实测（2026-10-07）：

    新浪行业分类覆盖 2987 只 / 全市场清单 5553 只 = **54%**
    板块严重偏斜：沪主板 64% · 深主板 83% · 创业板 34% · 科创板 5% · 北交所 21%

所以本模块**不设"对全市场覆盖率"闸门**（那样会永远返回 None），而是把覆盖域作为
数据的一部分返回（`df.attrs["coverage"]` / `["by_board"]`），并要求调用方展示：
"在覆盖域内找到 N 只" ≠ "在全市场找到 N 只"。

闸门只针对**本条通路自己的健康度**：49 个行业里成功不足 `MIN_SECTOR_SUCCESS` 时
判上游不稳，返回 None（宁可没有，也不要拿残缺样本当覆盖域）。

单位（实测值，全部集中在此换算，下游不再猜）
-------------------------------------------
行业明细原始单位：
    `per` 倍 · `pb` 倍 · `mktcap` **万元** · `nmc` 万元 ·
    `turnoverratio` **百分数** · `amount` **元** · `changepercent` 百分数
→ 本模块统一输出：`market_cap`/`float_cap` 亿元 · `amount` 亿元 ·
  `turnover`/`change_pct` 百分数 · `pe`/`pb` 倍。

⚠️ 新浪**日线**的 `turnover` 与行业快照的 `turnoverratio` **单位不同**：
    日线 `turnover` = 0.004187（**小数**，即 0.4187%）
    行业 `turnoverratio` = 2.39（**百分数**，即 2.39%）
→ 本模块统一输出 `turnover_pct`（百分数），避免"同一列名不同单位"这类静默错误
  （本项目已因单位混淆栽过三次：融资余额元/亿、同花顺百分号被吃、成交量当成交额）。
"""

import json
import logging
import os
import re
import threading
import time

import pandas as pd

_logger = logging.getLogger(__name__)

# ── 缓存 TTL ──
# ⚠️ TTL 必须显著长于任务时长，否则"有缓存 ≠ 成本受控"（本项目已踩过：
#    THS 快照 TTL 300s、选股一轮 4~5min → 中途重拉）。
SNAPSHOT_TTL = 900          # 15 分钟（盘后一轮选股 ≈ 4~5 min）
DAILY_TTL = 3600
_CODE_LIST_TTL = 7 * 86400  # 全市场代码清单一周更新一次足够
# 清单落盘路径：`stock_info_a_code_name` 实测会分页拉 18 页（网络差时 **257.7s
# 且失败**）→ 必须①限时②落盘，否则一次选股会被它拖死。
_CODE_LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "stock_code_list.json")

# 行业请求成功率下限：低于此值判上游不稳并放弃（**不是**对全市场的覆盖率闸门）
MIN_SECTOR_SUCCESS = 0.8

# 全市场清单获取的限时（秒）。超时即放弃过滤，**不阻塞主流程**。
CODE_LIST_TIMEOUT = 25.0

# 清单条数下限：低于此值判"拉到的不是全市场"（新浪/A 股 5000+，交易所清单 5500+）。
# ⚠️ 没有这条守卫时，"接口返回 200 但只给了一页/一张错误页"会被当成完整清单，
#    然后把 90% 的股票当退市剔除 —— 比不过滤严重得多。
_CODE_LIST_MIN_COUNT = 3000

_snapshot_cache = {"df": None, "ts": 0.0}
_daily_cache: dict = {}
_code_list_cache = {"codes": None, "ts": 0.0}
# 清单来源元信息（供 attrs / 排查读取）。⚠️ 必须能回答"这份清单是哪来的、几天前的"，
# 否则"用过期清单过滤"这件事对使用者完全不可见（本项目踩过：清单失效 19 天无人发现）。
_code_list_meta: dict = {"source": None, "age_days": None, "count": None, "at": None}

# 输出列（顺序固定，便于下游按名取用）
SNAPSHOT_COLUMNS = [
    "code", "name", "industry", "board",
    "price", "change_pct", "turnover", "amount", "volume",
    "pe", "pb", "market_cap", "float_cap",
]


def _clean_code(raw) -> str:
    """把各种形态的代码归一成 6 位数字串（去交易所前缀、补零）。

    实测来源形态：新浪行业明细给 `'600176'`、日线接口要 `'sh600176'`、
    全市场清单给 `'000001'`（但可能被 pandas 读成 int）。
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return ""
    for pre in ("sh", "sz", "bj"):
        if s.lower().startswith(pre):
            s = s[len(pre):]
            break
    s = re.sub(r"\D", "", s)
    if not s:
        return ""
    if len(s) == 6:
        return s
    if len(s) < 6:
        return s.zfill(6)
    return ""


# 板块名的**唯一口径来自 `core/sector.classify_by_board`**（"主板沪市"/"主板深市"）。
# 本模块为保持 data 层不反向依赖 core 而复制了判据，但返回值**逐字一致**，
# 由 `tests/test_sina_source.py::test_board_of_matches_core_sector` 守护——
# 曾因写成"沪主板"与 core 的"主板沪市"不一致而被测试抓住：这类"两处口径漂移"
# 会造成"UI 按一个键筛、覆盖率按另一个键统计"的静默错位。
_BOARD_ORDER = ("主板沪市", "主板深市", "创业板", "科创板", "北交所", "其他")


def _board_of(code: str) -> str:
    """交易所板块（口径与 `core/sector.classify_by_board` 逐字一致）

    ⚠️ `920xxx` 是 2024 起启用的**北交所**新代码段（不是"其他"）——
    此前漏判会让北交所票在"板块筛选/行业分散"里落进"其他"这个垃圾桶。
    """
    if code.startswith(("300", "301")):
        return "创业板"
    if code.startswith("688"):
        return "科创板"
    if code.startswith("60"):
        return "主板沪市"
    if code.startswith("00"):
        return "主板深市"
    if code.startswith(("8", "4", "920")):
        return "北交所"
    return "其他"


def _prefixed(code: str) -> str:
    """加交易所前缀（新浪日线接口要 `sh/sz/bj` 前缀）。

    刻意**按代码前缀直判**，不经 `_board_of`：上一版经 board 名判断，
    在把"沪主板"改成"主板沪市"（对齐 core 口径）时忘了同步改这里，
    600176 被拼成 `bj600176` —— 同一处口径漂移在同一次改动里犯了两次。
    前缀规则本身是稳定的，不依赖任何命名。
    """
    if code.startswith(("6", "9")):                   # 沪市（含科创板 688、B 股 900）
        return "sh" + code
    if code.startswith(("0", "3")):                   # 深市（含创业板 300/301）
        return "sz" + code
    return "bj" + code                                # 北交所 4xx/8xx/920


def _num(v):
    """转 float；失败返 None（**绝不返回 0** —— 缺就是缺）"""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:                                        # NaN
        return None
    return f


def _run_with_timeout(fn, timeout: float):
    """在 daemon 线程里跑 fn 并限时。

    akshare 接口没有超时参数，而 `stock_info_a_code_name` 实测分页拉 18 页、
    网络差时 **257.7s 才失败**——不能让它阻塞一次选股（本项目已有先例：
    `data/news_feed._call_with_timeout` 出于同一原因）。

    Returns: (是否按时完成, 结果)；抛出的异常原样上抛。
    """
    box: dict = {}

    def _worker():
        try:
            box["v"] = fn()
        except Exception as e:                        # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return False, None
    if "e" in box:
        raise box["e"]
    return True, box.get("v")


def _load_code_list_from_disk(now: float) -> tuple[set, float] | None:
    """读落盘清单，返回 `(codes, 龄期秒)` —— **过期也返回**。

    ⚠️ 原实现过期直接返 None ⇒ 一旦上游清单接口挂掉，就退化成"完全不过滤"，
    退市股混进快照（价格 0、涨幅 0、还占选股预取）。而**退市是单向事件**：
    30 天前的清单仍然是对的，只是可能缺几只新上市的。两害相权，用过期清单过滤
    远比不过滤安全（代价只是新股晚几天进候选），调用方按龄期决定是否告警。
    """
    try:
        with open(_CODE_LIST_FILE, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:                                 # noqa: BLE001 - 读不到就走网络
        return None
    codes = {str(c) for c in (d.get("codes") or []) if c}
    if not codes:
        return None
    fetched_at = float(d.get("fetched_at") or 0)
    age = max(0.0, now - fetched_at) if fetched_at else float("inf")
    return codes, age


def _save_code_list_to_disk(codes: set, source: str = "") -> None:
    try:
        os.makedirs(os.path.dirname(_CODE_LIST_FILE), exist_ok=True)
        with open(_CODE_LIST_FILE, "w", encoding="utf-8") as f:
            json.dump({"fetched_at": time.time(), "count": len(codes),
                       "source": source, "codes": sorted(codes)},
                      f, ensure_ascii=False)
    except Exception as e:                            # noqa: BLE001 - 存不下不影响本次
        _logger.warning("[新浪快照] 代码清单落盘失败: %s", e)


def _fetch_code_list_akshare(timeout: float) -> set | None:
    """akshare 清单（交易所口径）。⚠️ 实测 2026-10-15~26 连续报
    `Excel file format cannot be determined`（上游给的已不是 Excel），
    所以它**只能算第一档**，不能当唯一来源。"""
    ok, df = _run_with_timeout(
        lambda: __import__("akshare").stock_info_a_code_name(), timeout)
    if not ok or df is None:
        return None
    codes = {c for c in (_clean_code(x) for x in df["code"].tolist()) if c}
    return codes or None


def _fetch_code_list_sina(timeout: float = 12.0) -> set | None:
    """新浪全市场清单（`Market_Center.getHQNodeData`，`node=hs_a` 含北交所）。

    实测 2026-10-26：**5576 只 / 1.5s（8 并发）/ 退市票为 0**，且 `300750`、
    `920000` 都在。比 akshare 的交易所 Excel 可靠得多 —— 后者正在稳定报错。

    分页上限按 count 接口算；任一页失败只丢那一页，最后用总条数下限兜底判完整性。
    """
    try:
        import math
        from concurrent.futures import ThreadPoolExecutor

        import requests
    except Exception as e:                            # noqa: BLE001
        _logger.warning("[新浪快照] 新浪清单依赖不可用: %s", e)
        return None

    base = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "http://finance.sina.com.cn"}
    per_page = 100                                    # ⚠️ 服务端硬上限 100，给更大也只返 100

    try:
        r = requests.get(base + "Market_Center.getHQNodeStockCount",
                         params={"node": "hs_a"}, headers=headers, timeout=timeout)
        total = int((r.text or "").strip().strip('"'))
    except Exception as e:                            # noqa: BLE001
        _logger.warning("[新浪快照] 新浪清单条数接口失败: %s", e)
        return None
    if total < _CODE_LIST_MIN_COUNT:
        _logger.warning("[新浪快照] 新浪清单条数异常(%d)，不采用", total)
        return None

    pages = min(80, max(1, math.ceil(total / per_page)))

    def _one(page: int) -> list:
        for _ in range(2):
            try:
                rr = requests.get(base + "Market_Center.getHQNodeData",
                                  params={"page": page, "num": per_page,
                                          "sort": "symbol", "asc": 1,
                                          "node": "hs_a", "symbol": "",
                                          "_s_r_a": "page"},
                                  headers=headers, timeout=timeout)
                txt = (rr.text or "").strip()
                if txt.startswith("["):
                    return json.loads(txt) or []
            except Exception as e:                    # noqa: BLE001 - 单页失败退避重试
                _logger.debug("[新浪快照] 清单第 %d 页失败(%s)，重试", page, e)
        return []

    codes: set = set()
    with ThreadPoolExecutor(max_workers=8) as ex:
        for arr in ex.map(_one, range(1, pages + 1)):
            for it in arr:
                c = _clean_code((it or {}).get("code"))
                if c:
                    codes.add(c)
    if len(codes) < _CODE_LIST_MIN_COUNT:
        _logger.warning("[新浪快照] 新浪清单只拿到 %d 只(<%d)，判不完整不采用",
                        len(codes), _CODE_LIST_MIN_COUNT)
        return None
    return codes


def _fetch_code_list_network(timeout: float) -> tuple[set, str] | None:
    """按优先级试网络清单源，返回 `(codes, 来源名)`。

    顺序：**新浪（1.5s，实测干净）→ akshare（交易所口径，权威但当前在挂）**。
    ⚠️ 顺序不能反：akshare 那个接口 18 页 × 1.35s ≈ 24s，且当前稳定报
    `Excel file format cannot be determined` ⇒ 放前面等于每天白烧 27s 才拿到同一份清单。
    """
    for name, fn in (("sina", _fetch_code_list_sina),
                     ("akshare", _fetch_code_list_akshare)):
        try:
            codes = fn(timeout)
        except Exception as e:                        # noqa: BLE001 - 一个源挂了换下一个
            _logger.warning("[新浪快照] 代码清单源 %s 失败(%s)", name, e)
            continue
        if codes:
            return codes, name
    return None


def last_code_list_meta() -> dict:
    """清单来源/龄期（排查 & attrs 用）。⚠️ 读它之前不清缓存，只反映最近一次取值。"""
    return dict(_code_list_meta)


def market_codes(timeout: float = CODE_LIST_TIMEOUT) -> set | None:
    """全市场 A 股代码集合（用于过滤新浪分类里未清理的退市/变更标的）。

    实测 2026-10-07：新浪行业里多出 **364 只**不在 A 股清单的代码；
    2026-10-26 实测有 **186 只**价格为 0 的退市票混进快照。

    五级取数：内存缓存 → 磁盘(未过期) → 网络(akshare→新浪) → **磁盘(已过期)**
    → None。
    ⚠️ 倒数第二档是关键：上游清单接口已连续 10 天报
    `Excel file format cannot be determined`，而原实现在这一档直接放弃过滤
    ⇒ 每天把 186 只退市票写进快照。退市不可逆，用过 19 天的清单过滤仍然正确。
    返回 None 表示**连过期清单都没有**（首次部署），此时不做过滤，调用方要在
    attrs 里如实标注 `filtered=False`。
    """
    now = time.time()
    if _code_list_cache["codes"] is not None \
            and now - _code_list_cache["ts"] < _CODE_LIST_TTL:
        return _code_list_cache["codes"]

    disk = _load_code_list_from_disk(now)
    if disk and disk[1] <= _CODE_LIST_TTL:
        codes, age = disk
        _code_list_cache.update({"codes": codes, "ts": now})
        _code_list_meta.update({"source": "disk", "age_days": round(age / 86400, 1),
                                "count": len(codes), "at": now})
        return codes

    got = _fetch_code_list_network(timeout)
    if got:
        codes, src = got
        _code_list_cache.update({"codes": codes, "ts": now})
        _save_code_list_to_disk(codes, src)
        _code_list_meta.update({"source": src, "age_days": 0.0,
                                "count": len(codes), "at": now})
        _logger.info("[新浪快照] 代码清单已更新：%s / %d 只", src, len(codes))
        return codes

    if disk:
        codes, age = disk
        _code_list_cache.update({"codes": codes, "ts": now})
        _code_list_meta.update({"source": "disk_stale",
                                "age_days": round(age / 86400, 1),
                                "count": len(codes), "at": now})
        _logger.warning("[新浪快照] ⚠️ 清单源全部失败，退回 %.1f 天前的落盘清单(%d 只)"
                        "——退市不可逆，用它过滤仍安全，但新上市标的可能漏；"
                        "**不要让这个状态长期存在**", age / 86400, len(codes))
        return codes

    _code_list_meta.update({"source": None, "age_days": None, "count": None, "at": now})
    _logger.warning("[新浪快照] 代码清单获取失败且无落盘清单，本次不过滤退市标的")
    return None


def fetch_sector_snapshot() -> pd.DataFrame | None:
    """拉取全市场标量快照（走新浪 49 个行业明细，逐行业容错）。

    Returns:
        DataFrame，列见 `SNAPSHOT_COLUMNS`（单位见模块 docstring）；
        `df.attrs` 带 source / coverage / by_board / total_market / got /
        failed_sectors / filtered / elapsed / caveat。
        返回 None = 上游不稳或完全取不到（**不返回残缺数据**）。
    """
    import akshare as ak

    t0 = time.time()
    try:
        sectors = ak.stock_sector_spot(indicator="新浪行业")
    except Exception as e:                            # noqa: BLE001
        _logger.error("[新浪快照] 行业列表拉取失败: %s", e)
        return None
    if sectors is None or sectors.empty:
        _logger.error("[新浪快照] 行业列表为空")
        return None

    valid = market_codes()                            # None = 不过滤
    rows: list = []
    failed: list = []
    ok_sectors = 0

    for _, srow in sectors.iterrows():
        label = str(srow.get("label") or "").strip()
        industry = str(srow.get("板块") or "").strip()
        if not label:
            continue
        try:
            detail = ak.stock_sector_detail(sector=label)
        except Exception as e:                        # noqa: BLE001 - 单行业失败不影响其余
            failed.append(f"{industry or label}({type(e).__name__})")
            continue
        ok_sectors += 1
        if detail is None or detail.empty:
            continue
        for _, r in detail.iterrows():
            code = _clean_code(r.get("code"))
            if not code:
                continue
            if valid is not None and code not in valid:
                continue                              # 退市/变更标的（新浪未清理）
            mktcap_wan = _num(r.get("mktcap"))        # 万元
            nmc_wan = _num(r.get("nmc"))              # 万元
            amount_yuan = _num(r.get("amount"))       # 元
            rows.append({
                "code": code,
                "name": str(r.get("name") or "").strip(),
                "industry": industry,
                "board": _board_of(code),
                "price": _num(r.get("trade")),
                "change_pct": _num(r.get("changepercent")),      # 已是百分数
                "turnover": _num(r.get("turnoverratio")),        # 已是百分数
                "amount": None if amount_yuan is None else amount_yuan / 1e8,   # 元 → 亿元
                "volume": _num(r.get("volume")),
                "pe": _num(r.get("per")),
                "pb": _num(r.get("pb")),
                "market_cap": None if mktcap_wan is None else mktcap_wan / 1e4,  # 万元 → 亿元
                "float_cap": None if nmc_wan is None else nmc_wan / 1e4,         # 万元 → 亿元
            })

    total_sectors = len(sectors)
    success_rate = ok_sectors / total_sectors if total_sectors else 0.0
    if success_rate < MIN_SECTOR_SUCCESS:
        _logger.error("[新浪快照] 行业成功率 %.0f%%（%d/%d）低于下限 %.0f%%，判定上游不稳，放弃",
                      success_rate * 100, ok_sectors, total_sectors, MIN_SECTOR_SUCCESS * 100)
        return None
    if not rows:
        _logger.error("[新浪快照] 所有行业均无成分返回")
        return None

    df = pd.DataFrame(rows).drop_duplicates(subset=["code"], keep="first")
    # ⚠️「缺就是缺」：`trade<=0` 不是"价格 0 元"，是**根本没有报价**（退市/长期停牌）。
    #    原实现原样入库 ⇒ 快照里常驻 186 只价格为 0 的退市票（PT金田A、深赤湾A…），
    #    下游按低价排序/按涨幅筛都把它们当有效标的，还各白跑一次估值预取。
    #    这里丢掉并在 attrs 里报数——丢多少必须看得见，不能静默缩水。
    no_quote = int(((df["price"].isna()) | (df["price"] <= 0)).sum())
    if no_quote:
        df = df[(df["price"].notna()) & (df["price"] > 0)]
    if df.empty:
        _logger.error("[新浪快照] 全部标的无有效报价（%d 只），判上游异常", no_quote)
        return None
    df = df[SNAPSHOT_COLUMNS].reset_index(drop=True)

    # ── 覆盖域统计（**必须展示给用户**）──
    got = len(df)
    total_market = len(valid) if valid is not None else None
    by_board: dict = {}
    for board in _BOARD_ORDER:
        g = int((df["board"] == board).sum())
        t = None
        if valid is not None:
            t = sum(1 for c in valid if _board_of(c) == board)
        by_board[board] = {
            "got": g, "total": t,
            "pct": (round(g / t, 4) if t else None),
        }

    coverage = (round(got / total_market, 4) if total_market else None)
    # ⚠️「降级必须随数据走」（2026-10-17）：全市场代码清单拿不到时，本次**不过滤退市标的**，
    #    覆盖率也只能是"不知道"。原实现把这种情况也打成 `✅ …（覆盖 ?）`（INFO），
    #    读起来像成功 ⇒ 下游/值班人看不出"这一份截面可能含退市票、且覆盖未知"。
    _degraded = [] if valid is not None else ["code_list_unavailable"]
    _meta = last_code_list_meta()
    if _meta.get("source") == "disk_stale":
        _degraded.append(f"code_list_stale_{_meta.get('age_days')}d")
    df.attrs.update({
        "source": "sina_sector",
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed": round(time.time() - t0, 1),
        "got": got,
        "dropped_no_quote": no_quote,
        "code_list_source": _meta.get("source"),
        "code_list_age_days": _meta.get("age_days"),
        "total_market": total_market,
        "coverage": coverage,
        "by_board": by_board,
        "failed_sectors": failed,
        "sector_success_rate": round(success_rate, 4),
        "filtered": valid is not None,
        "degraded": _degraded,
        "caveat": _caveat(got, total_market, by_board),
    })
    if _degraded:
        _logger.warning("[新浪快照] ⚠️ %d 只（覆盖 %s，**降级** %s）/ 行业 %d/%d / %.1fs ｜ %s"
                        " ｜ 丢弃无报价 %d 只",
                        got, f"{coverage:.0%}" if coverage else "?", _degraded,
                        ok_sectors, total_sectors, df.attrs["elapsed"],
                        df.attrs["caveat"], no_quote)
    else:
        _logger.info("[新浪快照] ✅ %d 只（覆盖 %s）/ 行业 %d/%d / %.1fs ｜ 丢弃无报价 %d 只",
                     got, f"{coverage:.0%}" if coverage else "?", ok_sectors,
                     total_sectors, df.attrs["elapsed"], no_quote)
    return df


def _caveat(got: int, total_market, by_board: dict) -> str:
    """生成"候选域说明"（供 UI / 报告直接展示，避免各处自己拼）"""
    if not total_market:
        return (f"本次候选域 {got} 只（新浪行业分类）；全市场清单不可用，"
                f"**无法给出覆盖率，结论不可外推至全市场。**")
    parts = []
    for board, st in by_board.items():
        if st["total"] and st["pct"] is not None:
            parts.append(f"{board} {st['pct']:.0%}")
    detail = "、".join(parts)
    return (f"本次候选域 {got}/{total_market} 只（覆盖 {got / total_market:.0%}）——{detail}。"
            f"**未覆盖全部创业板/科创板标的，结论不可外推至全市场。**")


def get_sector_snapshot(force_refresh: bool = False) -> pd.DataFrame | None:
    """对外入口：带 TTL 缓存。"""
    now = time.time()
    cached = _snapshot_cache["df"]
    if cached is not None and not force_refresh \
            and now - _snapshot_cache["ts"] < SNAPSHOT_TTL:
        return cached
    df = fetch_sector_snapshot()
    if df is not None and not df.empty:
        _snapshot_cache["df"] = df
        _snapshot_cache["ts"] = now
    return df


def snapshot_note(df) -> str:
    """把 `df.attrs` 里的覆盖域信息整理成一句可直接展示的文案。"""
    if df is None:
        return "候选域数据不可用"
    return str(df.attrs.get("caveat") or f"候选域 {len(df)} 只")


def fetch_stock_daily(code: str, start_date: str = None, end_date: str = None,
                      adjust: str = "qfq") -> pd.DataFrame | None:
    """个股日线（新浪）。

    Args:
        code: 6 位代码
        start_date / end_date: `YYYYMMDD` 或 `YYYY-MM-DD`（两种写法都接受）
        adjust: `qfq` / `hfq` / `""`

    Returns:
        DataFrame，列为 `date open high low close volume amount
        outstanding_share turnover_pct`（`turnover_pct` 已换算成**百分数**），
        `df.attrs` 带 source / code。
        取不到返回 None（**不抛异常**）。
    """
    code = _clean_code(code)
    if len(code) != 6:
        return None

    def _fmt(d):
        return None if not d else str(d).replace("-", "")

    start, end = _fmt(start_date), _fmt(end_date)
    key = (code, start, end, adjust)
    hit = _daily_cache.get(key)
    if hit and time.time() - hit[1] < DAILY_TTL:
        return hit[0]

    try:
        import akshare as ak
        df = ak.stock_zh_a_daily(symbol=_prefixed(code),
                                 start_date=start, end_date=end, adjust=adjust or "")
    except Exception as e:                            # noqa: BLE001
        _logger.warning("[新浪日线] %s 取数失败: %s", code, e)
        return None
    if df is None or df.empty:
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "turnover" in df.columns:
        # ⚠️ 日线 turnover 是**小数**（0.004187），与行业明细的 turnoverratio（百分数）
        #    单位不同 —— 统一成百分数输出，杜绝"同列名不同单位"。
        df["turnover_pct"] = pd.to_numeric(df["turnover"], errors="coerce") * 100.0
    else:
        df["turnover_pct"] = None
    df.attrs.update({"source": "sina_daily", "code": code})
    _daily_cache[key] = (df, time.time())
    return df


def clear_cache(disk: bool = True):
    """清缓存（测试 / 强制刷新用）。

    `disk=True` 时顺带删掉落盘的代码清单——测试必须清它，否则用例之间会
    通过真实文件互相串味（本项目吃过"测试打到真实数据"的亏）。
    """
    _snapshot_cache["df"] = None
    _snapshot_cache["ts"] = 0.0
    _daily_cache.clear()
    _code_list_cache["codes"] = None
    _code_list_cache["ts"] = 0.0
    _code_list_meta.update({"source": None, "age_days": None, "count": None, "at": None})
    if disk:
        try:
            os.remove(_CODE_LIST_FILE)
        except OSError:
            pass
