"""腾讯行情 HTTP 直连数据源（不走 akshare 封装）

为什么要有这个模块
------------------
2026-09-25 排查"akshare 极其不稳定"，日志里的结论很集中：

    失败语句统计（logs/*.log）
      440× [Akshare-腾讯] ❌ XXXXXX 获取失败: No value to decode
      146× [Akshare-腾讯] ❌ XXXXXX 已重试 3 次，放弃
    → 586 / 587 条失败全部来自这一个接口，而它是**历史 K 线的唯一主路径**。

读 akshare 的 `stock_zh_a_hist_tx` 源码后，脆弱点是明确的三个：

1. 用 `demjson.decode` 解析响应，**响应体为空时抛的就是 `No value to decode`**
   ——错误信息只是"解码失败"，完全看不出真实原因是"上游返回了空体/被限流"；
2. 每次调用先打一次 `get_tx_start_year(symbol)` 取上市年份，**平白多一个失败点**；
3. 按年循环请求且**任何一年失败就抛异常**，整次调用作废（哪怕其余年份都拿到了）。

而同一个腾讯端点**直连 HTTP 是稳定的**（实测 9/9 成功，平均 0.6~0.7s，不需鉴权、
不封 IP）。所以本模块直接打这个端点，把那三个脆弱点去掉：

- 不带 `_var` 参数 → 返回**干净 JSON**，用标准 `json.loads` 即可，不引入 demjson；
- 不发多余请求，不依赖上市年份；
- **向后翻页**取数（见下），单页失败只影响该页，且失败原因如实记录。

关于翻页（实测得出的接口语义，容易踩坑）
----------------------------------------
该端点**忽略 param 里的起始日期**，只返回"**截至结束日期的最近 count 根**"K 线：

    请求 2024-01-01~2024-12-31 (count=640) → 实得 2022-05-18 ~ 2024-12-31 (640 根)
    请求 2020-01-01~2020-12-31 (count=640) → 实得 2018-05-21 ~ 2020-12-31 (640 根)

因此不能"传了起始日期就以为拿到了"。本模块的做法是：以目标结束日期取一页，
若这一页的最早日期仍未覆盖到目标起始日期，就以"最早日期 − 1 天"为新的结束日期
**继续向前翻页**，直到覆盖或触达页数上限，最后按 [start, end] 裁剪并按日期去重。

复权口径不混用（但有一个真实例外）
----------------------------------
优先级：`qfq` 认 `qfqday`、`hfq` 认 `hfqday`、不复权认 `day`。

**例外**：股票**没有任何复权事件**（无分红/送转）时，接口只返回 `day`，
而此时 `day` 本来就等于前复权/后复权序列。实测 `sh688981`（中芯国际）请求 `qfq`
只得到 `day`，而同为科创板的 `sh688111` 正常返回 `qfqday`——若一律判定"无数据"，
会白丢这一整类个股。

因此判据是：**请求的键缺失，但 `day` 在、且 `qfqday` 与 `hfqday` 都不在** → 用 `day`
（确属无复权事件）。反之，若存在**另一个**已复权的键（例如请求 qfq 却只有 `hfqday`），
说明确实有复权事件、只是口径不符 → **拒绝并视为无数据**，绝不回退去读别的键
（那会把"前复权"悄悄变成"后复权"，属于静默错误）。

列口径与 source_manager 现有实现保持一致（下游无需改动）：
    date( datetime64 ) / open / close / high / low / volume(股) / amount(NaN)
其中该接口第 6 个字段是**成交量(手)**而非成交额（项目内已验证），故 volume = 该值 × 100，
amount 显式置 NaN 而不是编造。
已知覆盖边界（实测，属于**上游**限制而非本模块缺陷）
----------------------------------------------------
- 沪/深主板、创业板、科创板、深中小：全覆盖（各板块实测 181/181 根，0.56~0.63s）。
- 部分**老北交所代码**上游根本没有近期数据：`bj430047`（诺思兰德）返回的最后一根
  是 **2020-06-29**，因此查 2024 年区间会如实返回"无数据"。
  注意 akshare 用的是**同一个端点**，所以这里没有变差——只是把真相显示出来了。
- 该端点单页上限 640 根，所以长区间要靠翻页（本模块已实现，见上）。
"""

import json
import logging
import random
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

_logger = logging.getLogger(__name__)

# 腾讯前复权/后复权 K 线端点（与 akshare 用的是同一个，但这里直连）
KLINE_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Referer": "https://gu.qq.com/",
    "Accept": "*/*",
}

# 单页最多 640 根（接口上限）；10 年日线约 2430 根 → 4 页足够
_PAGE_SIZE = 640
_MAX_PAGES = 12          # 安全上限：防止接口行为变化导致翻页死循环

# 复权类型 → 响应里对应的数据键。**不可互相回退**（见模块 docstring）
_ADJUST_KEY = {"qfq": "qfqday", "hfq": "hfqday", "": "day"}


def to_tx_code(symbol: str) -> str:
    """6 位代码 → 腾讯格式（sh/sz/bj 前缀）。已是前缀格式则原样返回。

    与 data.source_manager._convert_to_tx_code 同一口径；此处独立实现是为了让本模块
    可脱离 source_manager 单独使用（也避免循环导入）。
    """
    if not symbol:
        return symbol
    s = str(symbol).strip().lower()
    if s[:2] in ("sh", "sz", "bj"):
        return s
    if len(s) != 6 or not s.isdigit():
        return symbol
    if s.startswith("6"):
        return f"sh{s}"
    if s.startswith(("0", "3")):
        return f"sz{s}"
    if s.startswith(("4", "8")):
        return f"bj{s}"
    return f"sz{s}"


def _norm_date(d) -> str:
    """统一成 YYYY-MM-DD；支持 '20240101' / '2024-01-01' / date / datetime"""
    if d is None:
        return datetime.now().strftime("%Y-%m-%d")
    if isinstance(d, (datetime, date)):
        return d.strftime("%Y-%m-%d")
    s = str(d).strip().replace("/", "-")
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def _request_page(code: str, end_date: str, adjust: str, timeout: float,
                  retries: int) -> list:
    """取一页（截至 end_date 的最近 _PAGE_SIZE 根）。

    返回原始行列表（每行前 6 个字段为 date/open/close/high/low/vol_hand）。
    失败抛异常，由调用方决定是否降级——**不做静默的空结果**。
    """
    params = {
        # 起始日期给一个极早值即可：接口本就忽略它，只认 end_date 与 count
        "param": f"{code},day,1990-01-01,{end_date},{_PAGE_SIZE},{adjust}",
        "r": f"{random.random():.16f}",   # 破缓存用随机数
    }
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(KLINE_URL, params=params, headers=_HEADERS,
                                timeout=timeout)
            text = (resp.text or "").strip()
            # ⚠️ 这里正是 akshare "No value to decode" 的现场：上游返回空体。
            # 原封装把空体直接喂给 demjson，报出的错完全看不出真实原因。
            if not text:
                raise ValueError(f"上游返回空响应体 (HTTP {resp.status_code})")
            payload = json.loads(text)
            node = (payload.get("data") or {}).get(code) or {}
            key = _ADJUST_KEY.get(adjust, "day")
            rows = node.get(key)
            if rows:
                return rows
            # 例外：该股没有复权事件 → 接口只给 day，且此时 day 就等于请求的序列
            # （实测 sh688981 请求 qfq 仅得 day；若判"无数据"会白丢这类个股）
            if key != "day" and node.get("day") \
                    and not node.get("qfqday") and not node.get("hfqday"):
                _logger.debug("[腾讯直连] %s 无复权事件，day 即 %s 序列", code, adjust)
                return node["day"]
            # 结构正常但没有该复权口径的数据：可能是停牌/新股/代码不存在。
            # 显式返回空列表（"确实没有"），而不是回退到别的复权口径。
            if any(k in node for k in ("day", "qfqday", "hfqday")):
                _logger.warning(
                    "[腾讯直连] %s 请求 %s 但响应无对应复权序列（其余口径存在，"
                    "拒绝混用以免把前复权当后复权）", code, adjust)
                return []
            raise ValueError(f"响应结构异常，data[{code}] 无 K 线字段")
        except Exception as e:                       # noqa: BLE001
            last_exc = e
            if attempt < retries - 1:
                delay = 0.5 * (attempt + 1)
                _logger.warning("[腾讯直连] %s 第 %d/%d 次失败(%s)，%.1fs 后重试",
                                code, attempt + 1, retries, e, delay)
                time.sleep(delay)
    raise last_exc


def fetch_kline(symbol: str, start_date, end_date=None, adjust: str = "qfq",
                timeout: float = 10.0, retries: int = 3):
    """取腾讯日线（前复权/后复权/不复权），返回与 source_manager 口径一致的 DataFrame。

    Returns:
        DataFrame(date/open/close/high/low/volume/amount) 或 None（未取到任何数据）。
        取到部分数据时**返回已有的部分**，并在日志里说明缺了哪些页——
        少数据比没有数据有价值，但绝不假装完整（下游 date 区间可自行判断）。
    """
    code = to_tx_code(symbol)
    start = _norm_date(start_date)
    end = _norm_date(end_date)
    if pd.Timestamp(start) > pd.Timestamp(end):
        start, end = end, start

    collected: dict = {}
    errors: list = []
    cursor = end

    for page in range(_MAX_PAGES):
        try:
            rows = _request_page(code, cursor, adjust, timeout, retries)
        except Exception as e:                       # noqa: BLE001
            errors.append(f"第{page + 1}页({cursor}): {e}")
            break
        if not rows:
            break
        for r in rows:
            if r and r[0]:
                collected[str(r[0])] = r
        earliest = str(rows[0][0])
        if earliest <= start:                        # 已覆盖到目标起始日
            break
        cursor = (pd.Timestamp(earliest) - timedelta(days=1)).strftime("%Y-%m-%d")

    if not collected:
        if errors:
            _logger.error("[腾讯直连] %s 取数失败: %s", code, "; ".join(errors))
        return None
    if errors:
        _logger.warning("[腾讯直连] %s 部分页失败（返回已取得的数据）: %s",
                        code, "; ".join(errors))

    # 按日期排序并裁剪到 [start, end]
    items = sorted(collected.items(), key=lambda kv: kv[0])
    rows = [r for d, r in items if start <= d <= end]
    if not rows:
        _logger.info("[腾讯直连] %s 在 %s~%s 区间内无 K 线", code, start, end)
        return None

    df = pd.DataFrame(rows)
    # 该接口每行至少 6 列；多余的标记列（如 {} / 换手率）一律忽略
    df = df.iloc[:, :6]
    df.columns = ["date", "open", "close", "high", "low", "volume_hand"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "close", "high", "low", "volume_hand"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 第 6 个字段是**成交量(手)**不是成交额（项目内已验证），换算成股
    df["volume"] = df["volume_hand"] * 100
    # 该源不提供成交额：显式 NaN，不编造
    df["amount"] = float("nan")
    df = df.drop(columns=["volume_hand"])
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(
        subset=["date"], keep="last").reset_index(drop=True)
    return df
