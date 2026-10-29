"""
实时行情模块：自选股/指数的实时报价。

数据源降级链：腾讯 qt.gtimg.cn（批量、快，实测0.5s）→ 新浪 hq.sinajs.cn。
按代码粒度 TTL 缓存（默认15秒），双源各自带熔断器（连续失败进入冷却期）。
不依赖 akshare/东财——本环境东财域名被墙。

单位口径（v2/M0 0.4，已用真实行情实测确认，请勿凭印象改）：
    对外统一输出 **volume_hand = 成交量(手)、amount_wan = 成交额(万元)**。
    2026-09-22 以 sh600519 实测：
      腾讯 p[36]=24573(手)、p[37]=308853(万元)
          → 308853×1e4 ÷ (24573×100) = 1256.88 ≈ 现价 1253.80 ✓
      新浪 f[8]=2457294(股)、f[9]=3088526148(元)
          → f[9] ÷ f[8] = 1256.88 ≈ 现价 ✓，且 f[8]÷100 = 24573 手 与腾讯一致
    即腾讯行情是（手, 万元）、新浪行情是（股, 元）——新浪分支必须显式换算。
    原实现把新浪的股/元直接塞进 volume_hand/amount_wan，导致降级到新浪时
    看盘页"成交额(亿)"虚高 1e4 倍。

新鲜度（v2/M0 0.4）：
    返回行的 `stale` / `age_seconds` 字段标明该报价是否为"双源全失败后回退的过期缓存"。
    这类数据必须在使用处显式提示，不能当作实时价。

数据日期（v2/M0 0.5，2026-09-25 实测发现）：
    原先 `stale` **只看"缓存年龄"**——刚从数据源拉到的行年龄是 0 秒，于是
    **上一个交易日的收盘价也被标成"实时"**。实测现场（2026-09-25 为休市日）：
      腾讯返回时间戳 20260924161456 → 17.88 是 09-24 的收盘价
      而代码 `strftime("%H:%M:%S")` 把日期丢掉了 → 对外只剩 `16:14:56`
      `stale=False` → 工具输出 `"数据状态": "实时"`
      最后 AI 助手回答"**当前的实时行情**：现价 17.88 元" —— 日期被抹掉并贴上"实时"。
    修正两件事：
      1. 新增 `quote_date`(YYYY-MM-DD) / `quote_datetime`，两个源口径统一；
      2. 陈旧判据改为「**行情自带日期 < 应有的最新交易日**」(`_expected_quote_date`)，
         而不是"我们多久前拉的"。这样休市日/盘前不会误报，而真落后一个交易日会报出来。
"""

import time
import logging
from datetime import datetime

import requests

_logger = logging.getLogger(__name__)

QUOTES_TTL = 15  # 行情缓存秒数（看盘页手动刷新为主）
_SOURCE_COOLDOWN = 600  # 数据源熔断冷却秒数

TENCENT_URL = "https://qt.gtimg.cn/q={query}"
SINA_URL = "https://hq.sinajs.cn/list={query}"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://finance.sina.com.cn/",
}

# 常用指数（腾讯/新浪均为带市场前缀的完整代码）
INDEX_CODES = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
}

_quote_cache = {}  # full_code -> (row_dict, timestamp)
_circuit = {"tencent": 0.0, "sina": 0.0}  # 各源熔断恢复时间戳


def _to_full_code(code: str) -> str:
    """6位代码 → 带市场前缀代码；已带前缀则原样返回"""
    code = str(code).strip().lower()
    if code.startswith(("sh", "sz", "bj")):
        return code
    if code.startswith(("60", "68", "9")):
        return f"sh{code}"
    if code.startswith(("00", "30", "2")):
        return f"sz{code}"
    if code.startswith(("8", "4")):
        return f"bj{code}"
    return f"sz{code}"


def _f(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# 「应有的最新行情日期」缓存。日历本身是多层缓存（内存→文件→akshare），
# 但这里再缓存一层，避免每只标的每次都去问日历。
_EXPECTED_DATE_CACHE = {"date": None, "at": 0.0}
_EXPECTED_DATE_TTL = 600


def _expected_quote_date(now: datetime = None) -> str | None:
    """现在这个时刻，行情**应该**给到哪一天的数据。取不到返回 None。

    规则（这是"陈旧"的正确判据，与"缓存年龄"是两回事）：
      - 今天不是交易日（周末/节假日）→ 应为最近交易日的收盘数据；
      - 今天是交易日但还没到 09:30（盘前）→ 应为昨天的收盘数据；
      - 今天是交易日且已过 09:30（含盘后）→ 应为**今天**的数据。

    这样：休市日 09-24 的数据不会被误判成陈旧；而交易日里只拿到昨天的数据
    就会被判成陈旧（有更新的数据却没取到）。日历/pandas 不可用时返回 None，
    调用方按"不判陈旧"处理——宁可少报，不可把最新数据误标成过期。
    """
    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")
    cached = _EXPECTED_DATE_CACHE
    if cached["date"] and time.time() - cached["at"] < _EXPECTED_DATE_TTL:
        return cached["date"]
    try:
        from data import trading_calendar as cal
        if cal.is_trading_day(today) and now.hour * 60 + now.minute >= 9 * 60 + 30:
            expected = today
        else:
            expected = cal.last_trading_day(include_today=False)
    except Exception as e:                      # noqa: BLE001 - 日历不可用不能拖垮行情
        _logger.debug("[行情] 交易日历不可用，跳过数据日期陈旧判定: %s", e)
        return None
    if expected:
        cached.update({"date": expected, "at": time.time()})
    return expected


def _parse_tencent_line(line: str) -> dict | None:
    """解析腾讯行情行: v_sh600519="1~贵州茅台~600519~1297.40~...";"""
    line = line.strip().rstrip(";")
    if '="' not in line:
        return None
    full_code = line.split('="')[0].replace("v_", "")
    parts = line.split('="')[1].split("~")
    if len(parts) < 40:
        return None
    ts_raw = parts[30] if len(parts) > 30 else ""
    # 关键：日期必须保留。此前只留 "%H:%M:%S"，日期一丢，"昨天的收盘价"就无从分辨，
    # 下游只好一律当作"刚查到的实时价"（2026-09-25 实测就是这个后果）。
    quote_date, quote_time = "", ""
    if len(ts_raw) >= 14:
        try:
            dt = datetime.strptime(ts_raw[:14], "%Y%m%d%H%M%S")
            quote_date, quote_time = dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
        except ValueError:
            quote_time = ts_raw
    elif len(ts_raw) == 8:                      # 只有日期
        try:
            quote_date = datetime.strptime(ts_raw, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    elif ts_raw:
        quote_time = ts_raw
    row = {
        "full_code": full_code,
        "code": parts[2],
        "name": parts[1],
        "price": _f(parts[3]),
        "prev_close": _f(parts[4]),
        "open": _f(parts[5]),
        "change": _f(parts[31]),
        "change_pct": _f(parts[32]),
        "high": _f(parts[33]),
        "low": _f(parts[34]),
        # 腾讯行情单位（2026-09-22 sh600519 实测确认）：成交量=手、成交额=万元
        "volume_hand": _f(parts[36]) if len(parts) > 36 else None,
        "amount_wan": _f(parts[37]) if len(parts) > 37 else None,
        "turnover_rate": _f(parts[38]) if len(parts) > 38 else None,
        "pe": _f(parts[39]) if len(parts) > 39 else None,
        "amplitude": _f(parts[43]) if len(parts) > 43 else None,
        "float_mcap_yi": _f(parts[44]) if len(parts) > 44 else None,
        "total_mcap_yi": _f(parts[45]) if len(parts) > 45 else None,
        "pb": _f(parts[46]) if len(parts) > 46 else None,
        "quote_date": quote_date,
        "quote_time": quote_time,
        "quote_datetime": f"{quote_date} {quote_time}".strip(),
        "source": "tencent",
    }
    if row["price"] is None:
        return None
    return row


def _parse_sina_line(line: str, full_code: str) -> dict | None:
    """解析新浪行情行: var hq_str_sh600519="贵州茅台,开,昨收,现价,最高,最低,..." """
    if '="' not in line:
        return None
    actual_code = line.split('="')[0].replace("var hq_str_", "")
    fields = line.split('="')[1].strip().rstrip(";").strip('"').split(",")
    if len(fields) < 32 or not fields[0]:
        return None
    price, prev_close = _f(fields[3]), _f(fields[2])
    if price is None or prev_close in (None, 0):
        return None
    change = round(price - prev_close, 3)
    # 新浪行情单位（2026-09-22 sh600519 实测确认）：f[8]=成交量(股)、f[9]=成交额(元)。
    # 必须换算成与腾讯分支一致的（手, 万元），否则两个源降级切换时数量级差 1e4 倍。
    vol_share, amount_yuan = _f(fields[8]), _f(fields[9])
    # 新浪的日期/时间是两个独立字段（f[30]=2026-09-24, f[31]=15:00:00）。
    # 原先直接拼成 `quote_time`，于是"腾讯=HH:MM:SS、新浪=YYYY-MM-DD HH:MM:SS"
    # 两源口径不一致，降级切换时下游会看到两种格式。统一成 date + time + datetime。
    sina_date = fields[30].strip() if len(fields) > 30 else ""
    sina_time = fields[31].strip() if len(fields) > 31 else ""
    row = {
        "full_code": actual_code or full_code,
        "code": (actual_code or full_code)[2:],
        "name": fields[0],
        "price": price,
        "prev_close": prev_close,
        "open": _f(fields[1]),
        "change": change,
        "change_pct": round(change / prev_close * 100, 2),
        "high": _f(fields[4]),
        "low": _f(fields[5]),
        # 股 → 手
        "volume_hand": round(vol_share / 100, 0) if vol_share is not None else None,
        # 元 → 万元
        "amount_wan": round(amount_yuan / 1e4, 2) if amount_yuan is not None else None,
        "turnover_rate": None,
        "pe": None,
        "amplitude": None,
        "float_mcap_yi": None,
        "total_mcap_yi": None,
        "pb": None,
        "quote_date": sina_date,
        "quote_time": sina_time,
        "quote_datetime": f"{sina_date} {sina_time}".strip(),
        "source": "sina",
    }
    return row


def _fetch_batch(full_codes: list[str], source: str) -> dict[str, dict]:
    """从指定源批量拉取，返回 {full_code: row}"""
    url = TENCENT_URL if source == "tencent" else SINA_URL
    query = ",".join(full_codes)
    resp = requests.get(url.format(query=query), headers=_HEADERS, timeout=10)
    resp.encoding = "gbk"
    if resp.status_code != 200:
        raise ConnectionError(f"{source} HTTP {resp.status_code}")
    rows = {}
    for line in resp.text.strip().split(";"):
        line = line.strip()
        if not line:
            continue
        row = _parse_tencent_line(line) if source == "tencent" else _parse_sina_line(line, "")
        if row:
            rows[row["full_code"]] = row
    if not rows:
        raise ValueError(f"{source} 未解析到有效数据")
    return rows


def _with_freshness(row: dict, age_seconds: float, stale: bool | None = None) -> dict:
    """返回行情行的**副本**并附上新鲜度/数据日期标记。

    必须是副本：缓存里存的就是同一个 dict 对象，直接改标记会污染缓存。

    两条判据是**互相独立**的，别混为一谈：
      - `stale`/`age_seconds`：**缓存年龄**——双源都失败时回退旧缓存，显式传 True。
      - `data_lagging`：**数据日期**落后于应有的最新交易日——即"有更新的数据却没取到"。
        2026-09-25（休市）实测：数据是 09-24 收盘、本应算"最新"，但代码只看缓存年龄
        → `stale=False` → 对外写着"实时"。所以日期这一维必须单独判。
    """
    out = dict(row)
    out["age_seconds"] = round(float(age_seconds), 1)
    out["stale"] = (float(age_seconds) > QUOTES_TTL) if stale is None else stale

    qd = out.get("quote_date") or ""
    out["data_lagging"] = False
    out["staleness_reason"] = ""
    if qd:                    # 没有日期信息时不做日期判定（也避免白跑一次交易日历）
        expected = _expected_quote_date()
        if expected:
            if qd < expected:
                out["data_lagging"] = True
                out["staleness_reason"] = f"行情日期 {qd} 早于应有的 {expected}"
            elif qd > expected:
                # 集合竞价等场景可能给到"更新"的日期，不算错误
                out["staleness_reason"] = f"行情日期 {qd} 晚于应有的 {expected}"
    return out


def get_realtime_quotes(codes: list[str], force_refresh: bool = False) -> dict[str, dict]:
    """
    批量获取实时行情。

    Args:
        codes: 6位代码或带市场前缀代码（指数也用带前缀形式，如 sh000001）
        force_refresh: 跳过TTL缓存强制刷新

    Returns:
        {code: row}，键为原始入参形式；取不到的代码不在结果中。
        每个 row 附带 stale / age_seconds，标明是否为"过期缓存回退"。
    """
    now = time.time()
    results = {}
    missing_pairs = []  # (原始入参, 全码)
    for raw in codes:
        full = _to_full_code(raw)
        cached = _quote_cache.get(full)
        if cached and not force_refresh and now - cached[1] < QUOTES_TTL:
            results[raw] = _with_freshness(cached[0], now - cached[1])
        else:
            missing_pairs.append((raw, full))
    if not missing_pairs:
        return results

    missing_fulls = [full for _, full in missing_pairs]
    for source in ("tencent", "sina"):
        if now < _circuit[source]:
            continue
        try:
            rows = _fetch_batch(missing_fulls, source)
        except Exception as e:
            # 单源失败即打开该源熔断（冷却期自愈探测），改走下一源
            _logger.warning("[实时行情] %s 批量获取失败: %s", source, e)
            _circuit[source] = time.time() + _SOURCE_COOLDOWN
            continue
        _circuit[source] = 0.0  # 成功则关闭该源熔断
        for full, row in rows.items():
            _quote_cache[full] = (row, time.time())
        for raw, full in missing_pairs:
            if full in rows:
                results[raw] = _with_freshness(rows[full], 0.0)
        return results
    # 双源不可用：退回旧缓存（过期也用，聊胜于无），但必须显式标注 stale ——
    # 调用方据此提示"数据可能过期"，避免把旧价当实时价用于盘中判断。
    for raw, full in missing_pairs:
        cached = _quote_cache.get(full)
        if cached:
            _logger.warning("[实时行情] %s 双源不可用，回退 %.0f 秒前的过期缓存",
                            raw, now - cached[1])
            results[raw] = _with_freshness(cached[0], now - cached[1], stale=True)
    return results


def get_index_quotes(force_refresh: bool = False) -> dict[str, dict]:
    """三大指数实时行情，键为带前缀代码（sh000001 等）"""
    return get_realtime_quotes(list(INDEX_CODES.keys()), force_refresh=force_refresh)


def clear_quote_cache():
    """清空行情缓存（看盘页刷新按钮）"""
    _quote_cache.clear()


if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("实时行情自检...")
    idx = get_index_quotes()
    for code, row in idx.items():
        print(f"  {row['name']}: {row['price']} ({row['change_pct']:+.2f}%) [{row['source']}]")
    stocks = get_realtime_quotes(["600519", "000001"])
    for code, row in stocks.items():
        amount_yi = (row["amount_wan"] / 1e4) if row.get("amount_wan") else None
        print(f"  {row['name']}({row['code']}): {row['price']} ({row['change_pct']:+.2f}%) "
              f"换手{row['turnover_rate']}% 成交额{amount_yi:.2f}亿 "
              f"[{row['source']}{'/过期' if row.get('stale') else ''}]")
