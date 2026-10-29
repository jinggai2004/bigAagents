# -*- coding: utf-8 -*-
"""
模块名: data/tushare_source.py
用途: tushare HTTP 直连客户端 —— **当日全市场增量**（A 档，2026-10-17 用户拍板）。

为什么 HTTP 直连而不是 tushare SDK：
  ① 本机实测的**限频行为**必须自己掌控（`daily` 50/分钟、`adj_factor`/`trade_cal` 各 1/分钟）；
  ② 业务码必须显式校验（⚠️ **HTTP 200 ≠ 成功**，业务码 40203=限频/无权限、
     10001011=黑名单，"假凭证得同样错误码不足以证明含义"是本项目铁律）；
  ③ SDK 的内部重试是黑盒——被限频时静默等待会把 15:40 调度任务挂死。

⚠️⚠️ 2026-10-26 纠正一条错了 9 天的结论：「daily_basic 是 1 次/小时」**是假的**。
  该接口对本账号**根本没有权限**，服务端只是用"频率超限"的话术软拒绝，而且话术还会变。
  证据（logs/ 全量统计，547 次失败、**成功 0 次**）：
      339 次「频率超限(1次/小时)」 + 195 次「频率超限(1次/分钟)」 + 13 次「频率超限(5次/天)"
  **同一个接口不可能同时有三个上限** —— 三个数字互相矛盾这件事本身就说明
  它不是真的额度，而是拒绝。用户侧积分档为 120：官方口径"非复权日线数据，其他接口无法调取"。
  实测同 token 下 `daily` 正常返回 5572 行，`moneyflow` 明确报"没有接口访问权限"。
  ⇒ 判据从"按接口写死秒数节流"改为**能力表 + 语义判定**（见 `_classify_api_error`）。

⚠️⚠️ 单位口径（本项目铁律"列名相同 ≠ 单位相同"）：
  tushare `daily`:  `vol` **手**、`amount` **千元**、`pct_chg` 百分数
  tushare `daily_basic`: `total_mv`/`circ_mv` **万元**、`turnover_rate` 百分数、`pe`/`pb` 倍数
  落库口径（`market_snapshot` 表）: 金额一律**元**、`turnover`/`change_pct` 百分数
  ⇒ 换算**只在这里做一次**（与新浪路径同约定，见 core/scheduler.save_daily_market_snapshot）。
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import pandas as pd
import requests

_logger = logging.getLogger(__name__)

_API_URL = "http://api.tushare.pro"
_TIMEOUT_S = 30

# ── 实测限频（120 积分档；单位秒）——宁保守勿超限 ──
# daily 50/分钟 ⇒ 1.2s/次；stock_basic 无实测限制。
#
# ⚠️⚠️ 两类接口、两种节流语义（2026-10-17 15:18 事故）：
#   _SLEEP_INTERVAL_S —— **sleep 等待**：批量回填专用（daily），间隔 1.2s 谁都等得起；
#   _REJECT_INTERVAL_S —— **抛异常**：调度任务专用（trade_cal 1 小时、adj_factor 1 分钟），
#      15:40 任务等 1 小时等于挂死 ⇒ 立即抛，让调用方降级。
#   第一版全部用"抛异常" ⇒ 回填 2565 天全被自己的节流器打回（成功仅 53 天）。
#
# ⚠️ 2026-10-26：`daily_basic` 已从 _REJECT_INTERVAL_S **移除**。
#   它对本账号是无权限（见模块 docstring 的 547 次统计），节流秒数写得再准也没意义 ——
#   真正该做的是**根本不要发这个请求**，那由下面的能力表 `_capability()` 负责。
_SLEEP_INTERVAL_S = {"daily": 1.2, "stock_basic": 0.5}
_REJECT_INTERVAL_S = {"trade_cal": 3600.0, "adj_factor": 60.0}

# ── 接口能力表（**跨进程持久化**）──
# 为什么必须落盘：`_last_call_at` 是进程内的，进程一重启就忘干净 ⇒ 每个交易日
# 15:40/15:47 都会重新试一遍 daily_basic，一天白跑两次、一年 500 次（实测已 547 次）。
# 记录"这个接口从来没成功过"这件事，才能让下一次直接跳过。
_CAPABILITY_PATH = None            # 延迟到首次使用时按 data_root() 解析
_CAP_STATE: dict[str, dict] = {}
_CAP_LOADED = False

# 连续这么多次「频率超限」且**成功次数为 0** ⇒ 判定为软拒绝（无权限），不再逐次试探
_SOFT_REJECT_LIMIT = 3
# 判定不可用后，隔这么久允许一次"探针"（万一用户升级了积分）
_PROBE_INTERVAL_S = 7 * 24 * 3600

_last_call_at: dict[str, float] = {}          # api_name -> monotonic 时间（进程内节流）


class TushareApiError(RuntimeError):
    """tushare 业务码非 0（HTTP 200 但业务失败）。"""

    def __init__(self, code, msg: str):
        self.code = code
        self.msg = msg
        super().__init__(f"tushare 业务码 {code}: {msg}")


class TushareRateLimited(TushareApiError):
    """命中限频（本地节流或服务端 40203）。调用方应**降级**而不是重试。"""


class TushareNoPermission(TushareApiError):
    """该接口对本账号**无访问权限**（升级积分才可能变）。调用方应**永久降级**。

    ⚠️ 与 `TushareRateLimited` 严格分开（铁律"一个变量不能承载两类语义"）：
    限频是"等会儿再来"，无权限是"永远别来了"。混成一类时的后果今天已经看到 ——
    调用方以为等一会儿就好，于是每个交易日在固定时刻重试，
    实测量级 547 次失败 / 0 次成功。
    """


class TushareStaleData(TushareApiError):
    """返回数据的 trade_date 与期望不符（当日截面尚未生成）。"""


# ---------------------------------------------------------------- 能力表
def _capability_path() -> str:
    global _CAPABILITY_PATH
    if _CAPABILITY_PATH:
        return _CAPABILITY_PATH
    # ⚠️ 只认 `config.paths.data_root()`，**不用 `__file__` 兜底**：
    #    冻结后 `__file__` 指向只读包 `_internal/`，写进去要么失败、要么写到包里的临时目录
    #    （重启就没了，等于没持久化）。守卫 `TestNoFileBasedDataRoot` 正是抓这个。
    #    定位不到就返回空串 ⇒ 调用方只保留进程内状态，宁可退化也不要有个假路径。
    from config.paths import data_root
    _CAPABILITY_PATH = os.path.join(data_root(), "temp", "tushare_capability.json")
    return _CAPABILITY_PATH


def _capability_file() -> str:
    """能力表文件路径；不可用时返回空串（**不猜**）。"""
    try:
        return _capability_path()
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[tushare] data_root 不可用（%s: %s）⇒ 能力表只保留进程内状态",
                        type(e).__name__, e)
        return ""


def _load_capability() -> dict:
    global _CAP_LOADED
    if _CAP_LOADED:
        return _CAP_STATE
    _CAP_LOADED = True
    path = _capability_file()
    if not path:
        return _CAP_STATE
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            _CAP_STATE.update({k: v for k, v in d.items() if isinstance(v, dict)})
    except FileNotFoundError:
        # 首次运行没有能力表，属正常状态 —— 但**仍留 debug 痕迹**：
        # 静默 except 会让"能力表路径配错了（永远读不到）"和"第一次跑"长得一样。
        _logger.debug("[tushare] 能力表尚未创建（首次运行属正常）: %s", path)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[tushare] 能力表读取失败，按空表处理: %s: %s",
                        type(e).__name__, e)
    return _CAP_STATE


def _save_capability() -> None:
    """原子写。⚠️ 失败只记日志：能力表是优化项，坏了最多多试一次，不能炸掉主链。"""
    path = _capability_file()
    if not path:
        return                                   # 已在上游记过 WARNING，不重复刷
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_CAP_STATE, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[tushare] 能力表写入失败（不影响本次调用）: %s: %s",
                        type(e).__name__, e)


def _record(api_name: str, *, ok: bool, no_permission: bool = False,
            reason: str = "") -> None:
    ent = _load_capability().setdefault(
        api_name, {"ok_count": 0, "reject_count": 0, "no_permission_count": 0,
                   "status": "unknown", "reason": "", "updated": ""})
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    if ok:
        ent["ok_count"] = int(ent.get("ok_count", 0)) + 1
        ent["status"] = "ok"
        ent["reason"] = ""
    else:
        if no_permission:
            ent["no_permission_count"] = int(ent.get("no_permission_count", 0)) + 1
            ent["status"] = "no_permission"
        else:
            ent["reject_count"] = int(ent.get("reject_count", 0)) + 1
            # ⚠️ 判据核心：**从未成功过** 且 累计被拒 ≥ _SOFT_REJECT_LIMIT
            # ⇒ 那些"频率超限"就是无权限的软拒绝。真的限频不会永远成功不了。
            if (int(ent.get("ok_count", 0)) == 0
                    and int(ent.get("reject_count", 0)) >= _SOFT_REJECT_LIMIT
                    and int(ent.get("no_permission_count", 0)) == 0):
                ent["status"] = "no_permission"
        ent["reason"] = reason[:200]
    ent["updated"] = now
    if ent.get("status") == "no_permission":
        ent["probe_after"] = time.time() + _PROBE_INTERVAL_S
    _save_capability()


def unavailable_reason(api_name: str) -> str:
    """返回"为什么这个接口在本账号不可用"；可用时返回空串。

    ⚠️ 探针语义：标记为 no_permission 之后**不是永远不再试** ——
    隔 `_PROBE_INTERVAL_S`（7 天）放行一次。用户升级积分后无需改代码就能自愈；
    而 7 天一次的代价，远小于"每个交易日白试两次"。
    """
    ent = _load_capability().get(api_name) or {}
    if ent.get("status") != "no_permission":
        return ""
    if time.time() >= float(ent.get("probe_after") or 0):
        _logger.info("[tushare] %s 已过探针期，本次放行试一次（看积分是否升级）", api_name)
        return ""
    return (f"该接口对本账号无权限（累计被拒 {int(ent.get('reject_count', 0)) + int(ent.get('no_permission_count', 0))} 次、"
            f"成功 0 次；上次：{str(ent.get('reason') or '')[:80]}）")


def capability_snapshot() -> dict:
    """给页面/自检用：各接口的可用性现状（只读，不触发探测）。"""
    return {k: dict(v) for k, v in _load_capability().items()}


def _classify_api_error(api_name: str, code, msg: str) -> TushareApiError:
    """把服务端拒绝文案映射成**语义正确**的异常类型。

    ⚠️ 不能只看业务码：`moneyflow` 的"没有接口访问权限"和 `daily_basic` 的
    "频率超限(1次/小时)" **业务码都是 40203**。只看码会把"永久无权限"当成"限频"。
    """
    low = str(msg or "")
    no_perm_markers = ("没有接口", "无权限", "访问权限", "没有访问", "权限不足")
    if any(m in low for m in no_perm_markers):
        _record(api_name, ok=False, no_permission=True, reason=low)
        return TushareNoPermission(code, msg)
    if (code == 40203 or "频率超限" in low or "每分钟" in low
            or "每小时" in low or "超过" in low):
        _record(api_name, ok=False, reason=low)
        # 记完之后再看能力表：可能刚好越过 _SOFT_REJECT_LIMIT 被判成无权限
        why = unavailable_reason(api_name)
        if why:
            return TushareNoPermission(code, f"{why}；服务端原文：{low[:120]}")
        return TushareRateLimited(code, msg)
    _record(api_name, ok=False, reason=low)
    return TushareApiError(code, msg)


def _load_token() -> str:
    """token 只从环境变量 / 项目根 `.env` 读，**绝不硬编码**（铁律）。"""
    tok = (os.environ.get("TUSHARE_TOKEN") or "").strip()
    if tok:
        return tok
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            ".env")
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("TUSHARE_TOKEN="):
                    tok = line.split("=", 1)[1].strip()
                    break
    except OSError as e:
        # .env 不存在 ⇒ 返回空 token（call_api 会显式报 NO_TOKEN，不静默当有凭证）
        _logger.debug("[tushare] .env 不可读（%s: %s）⇒ 走环境变量/报 NO_TOKEN",
                      type(e).__name__, e)
    return tok


def _throttle(api_name: str) -> None:
    """进程内最小间隔节流（调度器是单实例 ⇒ 进程内即可）。

    ⚠️ 语义按接口分：批量回填接口（daily）**sleep 等待**；
       调度接口（daily_basic/trade_cal/adj_factor）**抛异常** ⇒ 调用方降级。
    """
    interval = _SLEEP_INTERVAL_S.get(api_name)
    if interval:
        last = _last_call_at.get(api_name)
        now = time.monotonic()
        if last is not None:
            remain = interval - (now - last)
            if remain > 0:
                time.sleep(remain)                   # 回填等 1.2s 天经地义
        _last_call_at[api_name] = time.monotonic()
        return
    interval = _REJECT_INTERVAL_S.get(api_name)
    if not interval:
        return
    last = _last_call_at.get(api_name)
    now = time.monotonic()
    if last is not None and now - last < interval:
        raise TushareRateLimited(
            "LOCAL", f"{api_name} 本地节流：{interval:.0f}s 内已调用过（限频保护，"
                     f"调用方应降级而不是等待）")
    _last_call_at[api_name] = now


def call_api(api_name: str, params: Optional[dict] = None,
             fields: str = "") -> pd.DataFrame:
    """单次 tushare 调用 → DataFrame；**业务码/限频/无权限/超时全部显式**。"""
    tok = _load_token()
    if not tok:
        raise TushareApiError("NO_TOKEN", "TUSHARE_TOKEN 未配置（环境变量或项目根 .env）")
    # ⚠️ 先查能力表，**确认无权限就不发请求**（7 天一次探针例外）。
    #    放在 _throttle 之前：无权限时不该占用节流计时，否则探针那一次会被自己挡掉。
    why = unavailable_reason(api_name)
    if why:
        raise TushareNoPermission("UNAVAILABLE", why)
    _throttle(api_name)
    t0 = time.time()
    try:
        resp = requests.post(
            _API_URL, json={"api_name": api_name, "token": tok,
                            "params": params or {}, "fields": fields},
            timeout=_TIMEOUT_S)
    except requests.RequestException as e:
        raise TushareApiError("NETWORK", f"{type(e).__name__}: {e}") from e
    if resp.status_code != 200:
        raise TushareApiError(f"HTTP{resp.status_code}", resp.text[:120])
    js = resp.json()
    code, msg = js.get("code"), str(js.get("msg") or "")
    if code != 0:
        # ⚠️ 2026-10-26：不再自己按码猜语义，交给 `_classify_api_error`。
        #    "没有接口访问权限"同样返回 40203，按码判会把它当成"限频"，
        #    于是调用方以为等一会儿就好 —— 实测等了 9 天、547 次，一次没成功。
        raise _classify_api_error(api_name, code, msg)
    _record(api_name, ok=True)
    data = js.get("data") or {}
    df = pd.DataFrame(data.get("items") or [], columns=data.get("fields") or [])
    _logger.info("[tushare] %s %s → %d 行（%.2fs）",
                 api_name, params or {}, len(df), time.time() - t0)
    return df


def _require_date(df: pd.DataFrame, expect: str, api_name: str) -> pd.DataFrame:
    """⚠️ 陈旧判据 = **数据自带日期 vs 期望日期**（铁律）：daily/daily_basic 的
    当日截面可能尚未生成（tushare 盘后才更新），拿昨日数据顶替就是静默错账。"""
    if df.empty or "trade_date" not in df.columns:
        raise TushareStaleData("EMPTY", f"{api_name} 返回空（{expect} 截面可能未生成）")
    got = {str(x) for x in df["trade_date"].dropna().unique()}
    if got != {str(expect)}:
        raise TushareStaleData(
            "DATE_MISMATCH",
            f"{api_name} 返回日期 {sorted(got)[:3]} ≠ 期望 {expect}（当日截面未生成？）")
    return df


def fetch_daily(trade_date: str) -> pd.DataFrame:
    """全市场日线（ts_code, open/high/low/close, pre_close, vol**手**, amount**千元**, pct_chg…）"""
    return _require_date(call_api("daily", {"trade_date": _compact(trade_date)}),
                         _compact(trade_date), "daily")


def fetch_daily_basic(trade_date: str) -> pd.DataFrame:
    """全市场估值（pe/pe_ttm/pb, total_mv/circ_mv**万元**, turnover_rate 百分数…）"""
    return _require_date(call_api("daily_basic", {"trade_date": _compact(trade_date)}),
                         _compact(trade_date), "daily_basic")


def fetch_stock_basic() -> pd.DataFrame:
    """全市场标的清单（ts_code/symbol/name/area/industry/market/list_date…，低频缓存友好）。"""
    return call_api("stock_basic", {"list_status": "L"},
                    "ts_code,symbol,name,industry,market,list_date")


def _compact(d: str) -> str:
    """'2026-10-17' → '20261017'（tushare 接口要紧凑日期）。"""
    return str(d).replace("-", "")


def _f(v):
    """字符串数字 → float；空串/None 保持 None（**缺就是缺，不造 0**）。"""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        # 「解析失败返回 None，绝不返回 0」是显式语义（缺失值常态，DEBUG 级不产生噪音）
        _logger.debug("[tushare] 非数值 %r ⇒ None（%s）", v, type(e).__name__)
        return None


def build_market_snapshot_rows(trade_date: str) -> tuple[list[dict], dict]:
    """tushare 版全市场估值截面（A 档核心）：daily_basic(1 次) + daily(1 次) + stock_basic(1 次)。

    Returns:
        (rows, attrs) —— rows 直接可喂 `db_store.save_market_snapshot`；
        attrs = {"coverage", "total_market", "degraded", "caveat", "source": "tushare"}

    ⚠️ 单位换算只在这里做一次：total_mv/circ_mv **万元→元**、daily.amount **千元→元**。
    ⚠️ pe 列落 **pe_ttm** 值（TTM 口径，与 BaoStock `peTTM` 一致，因子研究跨源统一）；
       新浪历史行的口径未声明——`source` 列可区分（tushare_daily_basic vs sina_sector）。
    """
    basic = fetch_daily_basic(trade_date)
    daily = fetch_daily(trade_date)
    info = fetch_stock_basic()

    # ⚠️ 单边整体缺失（如 daily 返回空）时保持空表——不能 set_index 抛 KeyError
    b = basic.set_index("ts_code") if not basic.empty else pd.DataFrame()
    dl = daily.set_index("ts_code") if not daily.empty else pd.DataFrame()
    meta = info.set_index("ts_code") if not info.empty else pd.DataFrame()

    codes = sorted(set(b.index) | set(dl.index))
    rows, missing = [], {"only_basic": 0, "only_daily": 0}
    for ts in codes:
        code6 = str(ts).split(".")[0]
        bb = b.loc[ts] if ts in b.index else None
        dd = dl.loc[ts] if ts in dl.index else None
        if bb is None:
            missing["only_daily"] += 1
        if dd is None:
            missing["only_basic"] += 1
        mm = meta.loc[ts] if ts in meta.index else None
        # close 优先 daily（价量源），缺失退 daily_basic.close
        close = _f(dd.get("close")) if dd is not None else _f(bb.get("close") if bb is not None else None)
        rows.append({
            "stock_code": code6,
            "stock_name": (str(mm.get("name")) if mm is not None else ""),
            "industry": (str(mm.get("industry")) if mm is not None else ""),
            # 行业口径隔离（2026-10-29）：tushare 的 industry 是 111 类细分口径，
            # 同时落进 `industry_ts` 专列（供选股配额）；`industry` 列留给新浪口径
            # 供新闻荐股匹配。两条口径不混（铁律）。
            "industry_ts": (str(mm.get("industry")) if mm is not None else ""),
            "board": (str(mm.get("market")) if mm is not None else ""),
            "price": close,
            "change_pct": _f(dd.get("pct_chg")) if dd is not None else None,
            "pe": _f(bb.get("pe_ttm")) if bb is not None else None,       # TTM 口径
            "pb": _f(bb.get("pb")) if bb is not None else None,
            # 万元 → 元
            "market_cap": (None if bb is None or bb.get("total_mv") in (None, "")
                           else _f(bb.get("total_mv")) * 1e4),
            "float_cap": (None if bb is None or bb.get("circ_mv") in (None, "")
                          else _f(bb.get("circ_mv")) * 1e4),
            "turnover": _f(bb.get("turnover_rate")) if bb is not None else None,
            # 千元 → 元
            "amount": (None if dd is None or dd.get("amount") in (None, "")
                       else _f(dd.get("amount")) * 1e3),
        })
    total = max(len(rows), 1)
    attrs = {"source": "tushare", "coverage": round(len(rows) / total, 4),
             "total_market": len(rows),
             "degraded": ([f"daily 缺 {missing['only_daily']} 只"
                           if missing["only_daily"] else "",
                           f"daily_basic 缺 {missing['only_basic']} 只"
                           if missing["only_basic"] else ""]),
             "caveat": "pe 列为 TTM 口径（pe_ttm）；行业来自 tushare stock_basic"}
    attrs["degraded"] = [x for x in attrs["degraded"] if x]
    return rows, attrs


_TRADE_CAL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "trade_cal_cache.json")


def fetch_trade_cal(start: str, end: str) -> list[str]:
    """交易日清单（**1/小时**限频 ⇒ 本地缓存：一次成功永久复用）。

    ⚠️ 本地 `data/trading_calendar` 的日历缓存覆盖范围有限（超出 ⇒ 工作日近似），
       长区间回填必须用这里的数据源自己的日历——节假日空行既浪费配额又会被
       `_require_date` 误判成失败。
    ⚠️ 缓存命中条件 = 已缓存区间**完全覆盖**请求区间；否则重新调用（并整段更新）。
    """
    import json
    cs, ce = _compact(start), _compact(end)
    try:
        with open(_TRADE_CAL_CACHE, encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("start") <= cs and cache.get("end") >= ce and cache.get("days"):
            return [str(d) for d in cache["days"]
                    if cs <= str(d) <= ce]
    except (OSError, ValueError, TypeError) as e:
        # 缓存损坏/不存在 ⇒ 视为无缓存重取（宁可多调一次 API，不静默当有缓存）
        _logger.debug("[tushare] trade_cal 缓存不可读（%s: %s）⇒ 重新调用",
                      type(e).__name__, e)
    df = call_api("trade_cal", {"exchange": "SSE",
                                "start_date": _compact(start),
                                "end_date": _compact(end)})
    if df.empty or "is_open" not in df.columns:
        raise TushareApiError("EMPTY", "trade_cal 返回空")
    days = sorted(str(d) for d in df.loc[df["is_open"].astype(int) == 1, "cal_date"])
    with open(_TRADE_CAL_CACHE, "w", encoding="utf-8") as f:
        json.dump({"start": _compact(start), "end": _compact(end), "days": days,
                   "at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False)
    return days
