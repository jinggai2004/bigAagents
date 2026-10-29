"""
资金流向数据获取模块（akshare版本）
数据源降级链：
  1. 东方财富 stock_individual_fund_flow（逐股历史，质量最高）
  2. Tushare moneyflow（需配置 TUSHARE_TOKEN）
  3. 同花顺 stock_fund_flow_individual（全市场快照，即时/3日/5日/10日累计；
     东财域名不可达时的兜底源，评分走"快照模式"）
"""

import pandas as pd
import sys
import io
import logging
import time
import warnings
from datetime import datetime, timedelta
import akshare as ak
from data.source_manager import data_source_manager
from data.contracts import require_columns, require_not_all_nan

# 应用请求补丁（请求头/超时）
from data.akshare_helper import patch_requests
patch_requests()

warnings.filterwarnings('ignore')

_logger = logging.getLogger(__name__)

# ── tushare moneyflow 的**权限熔断**（2026-10-24，2026-10-26 升级为跨进程）──
# 事故：本账号 `moneyflow` **永久无权限**（"抱歉，您没有接口(moneyflow)访问权限"），
# 但代码只判 `tushare_available`（token 有效）⇒ 每只票都要先付一次注定失败的
# 往返（实测 **0.47s/次**）。120 只 = 白花 ~56s，还把日志刷满。
# 与"网络抖动"不同：权限错误**不会自愈** ⇒ 一旦识别就跳过。
#
# ⚠️ 2026-10-26 补齐一个漏洞：原来的 `_MONEYFLOW_DENIED` 是**进程内**的，
#    注释里写"重启后重试"——而调度器每个交易日都会重启一次，于是每个交易日
#    15:40 又白试一轮。现在改为接 `tushare_source` 的**落盘能力表**
#    （`temp/tushare_capability.json`），一次确认、跨进程生效，7 天后才放探针。
# ⚠️ 只跳过 moneyflow，不动其它 tushare 接口（daily 是通的）。
_MONEYFLOW_DENIED = False


def _moneyflow_allowed() -> bool:
    if _MONEYFLOW_DENIED:
        return False
    # 跨进程能力表：上个进程（可能是昨天）已经确认过无权限就直接跳过
    try:
        from data import tushare_source as _ts
        if _ts.unavailable_reason("moneyflow"):
            return False
    except Exception as e:                                   # noqa: BLE001
        # 能力表不可读 ⇒ 退回进程内标记（退化为改造前的行为，但不能因此崩掉资金流）
        _logger.debug("[资金流] 能力表不可读，退回进程内判据: %s: %s",
                      type(e).__name__, e)
    return True


def _note_moneyflow_error(err) -> None:
    """识别"无权限"并熔断；其它错误（超时/限频）保持原有重试行为。"""
    global _MONEYFLOW_DENIED
    msg = str(err)
    if _MONEYFLOW_DENIED:
        return
    if ("权限" in msg) or ("没有接口" in msg) or ("permission" in msg.lower()):
        _MONEYFLOW_DENIED = True
        _logger.warning(
            "[资金流] tushare moneyflow 无权限（%s）⇒ 后续跳过该源，"
            "改走同花顺快照/东财。⚠️ 这是**权限**问题不是网络问题，重试无意义。", msg[:60])

# 设置标准输出编码为UTF-8（仅在命令行环境，避免streamlit冲突）
def _setup_stdout_encoding():
    """仅在命令行环境设置标准输出编码"""
    if sys.platform == 'win32' and not hasattr(sys.stdout, '_original_stream'):
        try:
            # 检测是否在streamlit环境中
            import streamlit
            # 在streamlit中不修改stdout
            return
        except ImportError:
            # 不在streamlit环境，可以安全修改
            try:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
            except:
                pass

_setup_stdout_encoding()


# ================================================================
# 同花顺快照降级源（东财域名不可达时的兜底）
# ================================================================

THS_SNAPSHOT_TTL = 300  # 快照缓存秒数（同花顺一次调用返回全市场，避免逐股拉取）
_THS_WINDOWS = ["即时", "3日排行", "5日排行", "10日排行"]
_ths_snapshot_cache = {}  # window -> (DataFrame, timestamp)
# window -> 最近一次失败原因（供采集侧做"失败原因可查"，见 `get_ths_snapshot`）
_ths_last_error: dict = {}

EM_CIRCUIT_COOLDOWN = 600  # 东财接口连续失败后的熔断冷却秒数
_em_circuit_open_until = 0.0  # 熔断恢复时间戳


def _parse_ths_amount(value) -> float:
    """解析同花顺金额字符串（'1.09亿'/'-1.27亿'/'5034万'/纯数字）为元"""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value) if not pd.isna(value) else 0.0
    text = str(value).strip().replace(",", "").replace("--", "")
    if not text or text in ("-", "N/A"):
        return 0.0
    try:
        if text.endswith("亿"):
            return float(text[:-1]) * 1e8
        if text.endswith("万"):
            return float(text[:-1]) * 1e4
        return float(text)
    except ValueError:
        return 0.0


def _parse_ths_number(value):
    """解析同花顺的数值字符串为 float；**解析不了返回 None**（不是 0）。

    为什么要与 `_parse_ths_amount` 分开（2026-09-25 实测的真实故障）：
    同花顺的 `涨跌幅` 是百分比字符串 `"1.25%"`，而 `_parse_ths_amount` 只认
    `亿/万/纯数字`，遇到 `%` 会 `ValueError` → **返回 0.0**。
    于是采集落库的 5210 行 `change_pct` **全是 0.0**，看数据像"今天全市场都没涨没跌"，
    不报错、也不易察觉。价格同理：停牌股返回 `"--"` 时若落成 0.0 比缺值更糟。
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if value != value else float(value)     # NaN -> None
    text = (str(value).strip().replace(",", "").replace("%", "")
            .replace("元", "").replace("--", ""))
    if not text or text in ("-", "N/A"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _is_connection_level_error(exc) -> bool:
    """是否为"连不上"类失败（域名不可达 / 被代理拒绝 / 连接被重置）。

    为什么单独判定（2026-09-25 实测）：东财 `push2his.eastmoney.com` 在本环境
    经代理不可达时，原实现对**同一个主机**退避重试 3 次（2s/4s 退避 + 每次约 25s
    的连接失败）→ 单只股票白等 **约 95s**，而结论在第一次失败时就已经确定了。

    区分两类失败：
      - **连接层**（本函数返回 True）：主机现在不通，重试同一主机没有意义 → 立即熔断降级；
      - **内容层**（返回 False）：接口通了但返回空/解析失败，可能是抖动 → 保留重试。
    """
    try:
        import requests
        if isinstance(exc, (requests.exceptions.ProxyError,
                            requests.exceptions.ConnectionError,
                            requests.exceptions.Timeout)):
            return True
    except Exception:                                # noqa: BLE001
        pass
    text = str(exc)
    return any(m in text for m in (
        "Max retries exceeded", "ProxyError", "RemoteDisconnected",
        "Connection aborted", "Unable to connect", "Connection reset",
        "Name or service not known", "getaddrinfo failed",
    ))


def get_ths_snapshot(window: str):
    """获取同花顺全市场个股资金流快照（带TTL缓存），金额统一解析为元。

    ⚠️ 失败原因记进 `_ths_last_error`（模块级）由采集侧取用 —— "失败原因必须可查"，
       不能只留下一句"采集完成: 0 只"（2026-10-16 实测事故）。
    """
    cached = _ths_snapshot_cache.get(window)
    now = time.time()
    if cached and now - cached[1] < THS_SNAPSHOT_TTL:
        return cached[0]
    try:
        df = ak.stock_fund_flow_individual(symbol=window)
        if df is None or df.empty:
            _ths_last_error[window] = "上游返回空表（akshare 未报错）"
            return None
        df = df.rename(columns={"资金流入净额": "净额"})
        # v2/M0 0.4：列名漂移时立刻失败（原实现会在下面 df["股票代码"]/["净额"]
        # 处抛 KeyError 并被 except 吞掉，只留下一条含糊的"快照获取失败"）
        require_columns(df, ["股票代码", "净额"], f"同花顺快照[{window}]")
        for col in ("流入资金", "流出资金", "净额", "成交额"):
            if col in df.columns:
                df[col] = df[col].map(_parse_ths_amount)
        require_not_all_nan(df, ["净额"], f"同花顺快照[{window}]")
        df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
        _ths_snapshot_cache[window] = (df, now)
        _ths_last_error.pop(window, None)
        return df
    except Exception as e:
        # ⚠️⚠️ 必须走 `_logger`（原为 `print`）：长驻调度进程里 print **进不了**
        #    `logs/scheduler_*.log` ⇒ 失败在日志里完全隐形（2026-10-16 实测）。
        _logger.warning("[资金流快照] 同花顺[%s]获取失败: %s: %s",
                        window, type(e).__name__, e)
        _ths_last_error[window] = f"{type(e).__name__}: {str(e)[:120]}"
        return None


def get_ths_snapshot_record(code: str):
    """从同花顺各窗口快照中抽取单只股票的资金流记录（元为单位）。

    ⚠️ **仅供盘后采集路径使用（`capture_market_flow_snapshot`）。
    选股/深度分析不要调它** —— 一次调用要拉 4 个窗口的**全市场**快照，
    实测 **75.9 秒**（即时 16.7s + 3日 20.8s + 5日 15.6s + 10日 28.6s）。
    盘中取数请用 `get_stored_snapshot_record`（读落库快照，≈0.05 秒）。
    """
    record = {"股票代码": code}
    for window in _THS_WINDOWS:
        snap = get_ths_snapshot(window)
        if snap is None:
            continue
        rows = snap[snap["股票代码"] == code]
        if rows.empty:
            continue
        row = rows.iloc[0]
        if window == "即时":
            record["净额_今日"] = float(row.get("净额", 0) or 0)
            record["成交额_今日"] = float(row.get("成交额", 0) or 0)
            record["涨跌幅"] = str(row.get("涨跌幅", "N/A"))
            record["股票简称"] = str(row.get("股票简称", ""))
            record["最新价"] = str(row.get("最新价", "N/A"))
        else:
            days = window.replace("日排行", "")
            record[f"净额_{days}日"] = float(row.get("净额", 0) or 0)
    if "净额_今日" not in record:
        return None
    record.setdefault("净额_3日", record.get("净额_今日", 0.0))
    record.setdefault("净额_5日", record.get("净额_3日", 0.0))
    record.setdefault("净额_10日", record.get("净额_5日", 0.0))
    record["快照时间"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return record


INDUSTRY_FLOW_TTL = 600  # 行业资金流缓存秒数（看盘热度榜，非高频数据）
_industry_flow_cache = None  # (DataFrame, timestamp)


def _snapshot_data_date() -> str:
    """快照数据所属交易日（YYYY-MM-DD）。

    采集在盘后 15:40 跑，所以"今天"若为交易日就是今天，否则是最近交易日。
    不写成 `datetime.now()`：休市日补采时数据属于上一交易日，混起来会让
    "数据日期"变成采集日，正是本项目反复踩过的那个坑。
    """
    try:
        from data import trading_calendar as cal
        return cal.last_trading_day()
    except Exception:                                # noqa: BLE001
        return datetime.now().strftime("%Y-%m-%d")


def capture_market_flow_snapshot(snapshot_date: str = None) -> dict:
    """【盘后一次性采集】全市场资金流（4 窗口合并）+ 行业资金流 → 落库。

    为什么必须"一次性 + 落库"（2026-09-25 实测）：
      同花顺每个窗口都要拉**全市场 5210 行** —— 即时 16.7s / 3日 20.8s /
      5日 15.6s / 10日 28.6s。于是**为了一只股票**要 75.9 秒（4 次全市场），
      而快照内存缓存只有 300 秒，选股一轮 4~5 分钟**中途必然过期再拉一轮**
      → 一次选股光在这上面就白花 ~150 秒。
    采集一次 ≈77 秒且落在盘后关键路径之外；选股改读库，≈0.05 秒/只。
    """
    from db import store as db_store
    date = snapshot_date or _snapshot_data_date()

    # ── 个股：把 4 个窗口合并成"每只一行" ──────────────────────────────
    merged: dict = {}
    window_rows: dict = {}
    for window in _THS_WINDOWS:
        _ths_snapshot_cache.pop(window, None)        # 采集必须拿最新，不吃内存缓存
        snap = get_ths_snapshot(window)
        window_rows[window] = 0 if snap is None else len(snap)
        if snap is None:
            continue
        for row in snap.to_dict("records"):
            code = str(row.get("股票代码") or "").strip()
            if not code:
                continue
            rec = merged.setdefault(code, {"stock_code": code})
            net = row.get("净额")
            if window == "即时":
                rec["stock_name"] = str(row.get("股票简称") or "")
                rec["net_today"] = net if isinstance(net, (int, float)) else None
                amount = row.get("成交额")
                rec["amount"] = amount if isinstance(amount, (int, float)) else None
                # 涨跌幅是同花顺的百分比字符串（"1.25%"）；最新价可能是 "--"（停牌）。
                # 必须用 _parse_ths_number（解析不了给 None），用 _parse_ths_amount 会一律变成 0.0。
                rec["change_pct"] = _parse_ths_number(row.get("涨跌幅"))
                rec["price"] = _parse_ths_number(row.get("最新价"))
            else:
                days = window.replace("日排行", "")
                rec[f"net_{days}d"] = net if isinstance(net, (int, float)) else None
                rec[f"window_{days}d"] = True
    # ⚠️ 不补造缺失窗口：原实现把缺失的 3/5/10 日净额 setdefault 成"今日"的值，
    # 等于凭空创造一个"三个窗口都为正"的方向分。缺就是缺，让评分如实降级。

    stocks = db_store.save_fund_flow_snapshot(date, list(merged.values()), source="ths")
    # 采集完成后立刻失效进程内缓存：同进程的调用方（如复盘）应看到刚写的数据，
    # 而不是继续吃 TTL 内的旧快照。
    _snapshot_mem_cache.update({"at": 0.0, "date": None, "by_code": {}})

    # ── 行业资金流（~90 个行业）───────────────────────────────────────
    industries = 0
    global _industry_flow_cache
    _industry_flow_cache = None                      # 采集同样要绕开内存缓存
    ind_df = get_industry_flow_snapshot()
    if ind_df is not None and not ind_df.empty:
        rows = []
        for r in ind_df.to_dict("records"):
            net = r.get("净额")
            rows.append({
                "industry": str(r.get("行业") or "").strip(),
                "net_amount": (float(net) * 1e8) if isinstance(net, (int, float)) else None,
                "change_pct": r.get("涨跌幅") if isinstance(r.get("涨跌幅"), (int, float)) else None,
                "leader": str(r.get("领涨股") or ""),
            })
        industries = db_store.save_industry_flow_snapshot(date, rows, source="ths")

    # ── 三态报告（2026-10-17 修）───────────────────────────────────────
    # ⚠️ 原实现无论采到多少行都记 INFO"采集完成" ⇒ 实测 `个股 0 只 / 行业 0 个`
    #    也被读成"成功"，而失败原因在 `print` 里、日志查不到。
    #    现在：**0 行 / 任一窗口缺失 = 失败或部分**，并带上每个窗口的原因 ⇒ 一眼看出哪一步坏的。
    #    ⚠️ 判定必须**同时看窗口**：只判"总数>0"会把"4 个窗口只成 1 个"误判成 ok
    #       （第一版就是这么写的，被自己的用例抓到）。
    _fail_windows = {w: _ths_last_error.get(w) or "未取得数据"
                     for w, n in window_rows.items() if n == 0}
    _bad = len(_fail_windows)
    if stocks == 0 and industries == 0:
        status = "failed"
    elif _bad or stocks == 0 or industries == 0:
        status = "partial"
    else:
        status = "ok"
    result = {"snapshot_date": date, "stocks": stocks, "industries": industries,
              "windows": window_rows, "failures": _fail_windows, "status": status,
              "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    if status == "ok":
        _logger.info("[资金流快照] %s 采集完成: 个股 %d 只 / 行业 %d 个",
                     date, stocks, industries)
    else:
        # ⚠️ 用 WARNING：否则"0 只"会被下游与值班人当成正常完成
        _logger.warning("[资金流快照] %s 采集**%s**: 个股 %d 只 / 行业 %d 个；"
                        "窗口明细 %s；失败原因 %s",
                        date, "部分成功" if status == "partial" else "失败",
                        stocks, industries, window_rows, _fail_windows or "—")
    return result


# 落库快照的**进程内整表缓存**：一次读 5210 行（~50ms）后按代码查 0 成本。
# 为什么要它：`get_stored_snapshot_record` 每只都开一次 SQLite 连接，
# 实测 43ms/只 —— 一轮选股 18 只就是 0.8s，纯属连接开销。
# TTL 取 120 秒：既让批量选股只在开头读一次库，又保证跨天/盘后采集后不会长期吃旧数据。
_SNAPSHOT_MEM_TTL = 120
_snapshot_mem_cache = {"at": 0.0, "date": None, "by_code": {}}


def _snapshot_mem_get() -> tuple:
    """返回 (date, {code: record})，命中缓存则零 IO"""
    now = time.time()
    c = _snapshot_mem_cache
    if c["by_code"] and now - c["at"] < _SNAPSHOT_MEM_TTL:
        return c["date"], c["by_code"]
    from db import store as db_store
    try:
        date, data = db_store.load_fund_flow_snapshot()
    except Exception as e:                           # noqa: BLE001 - 读库失败不该拖垮选股
        _logger.warning("[资金流快照] 读库失败: %s", e)
        return None, {}
    c.update({"at": now, "date": date, "by_code": data or {}})
    return date, c["by_code"]


def ensure_snapshot_captured(snapshot_date: str = None) -> dict:
    """确保某个交易日的资金流快照已落库；缺才采集（幂等）。

    给复盘任务用：用户的要求是"**复盘时获取全行业数据，而不是让选股一次次跑**"。
    所以由复盘来兜底保证当天快照存在——即使 15:40 的采集任务失败或被跳过，
    复盘也会补上；反之已存在就立刻返回（一次 SELECT），不会重复拉全市场。
    """
    from db import store as db_store
    date = snapshot_date or _snapshot_data_date()
    have_stocks = db_store.latest_fund_flow_date()
    have_industry = db_store.latest_industry_flow_date()
    if have_stocks == date and have_industry == date:
        return {"snapshot_date": date, "skipped": True,
                "reason": "当天快照已存在"}
    result = capture_market_flow_snapshot(date)
    result["skipped"] = False
    return result


def get_stored_snapshot_record(code: str):
    """从**落库快照**取单只股票的资金流记录（单位：元）。取不到返回 None。

    这是选股/深度分析的取数路径：绝不现场拉全市场。快照由盘后
    `capture_market_flow_snapshot` 生成，读库（含进程内整表缓存）≈0.04 秒/只。
    返回里带 `数据日期`，调用方必须能说出"这份资金流是哪天的"。
    """
    code = str(code).strip()
    date, by_code = _snapshot_mem_get()
    rec = by_code.get(code)
    if not rec:
        return None
    out = {"股票代码": code,
           "股票简称": rec.get("stock_name") or "",
           "净额_今日": rec.get("net_today"),
           "净额_3日": rec.get("net_3d"),
           "净额_5日": rec.get("net_5d"),
           "净额_10日": rec.get("net_10d"),
           "成交额_今日": rec.get("amount"),
           "涨跌幅": rec.get("change_pct"),
           "最新价": rec.get("price"),
           "快照时间": rec.get("captured_at") or "",
           "数据日期": date}
    return out


def get_industry_flow_snapshot():
    """
    获取同花顺行业资金流快照（90个行业）。

    Returns:
        DataFrame[行业, 净额(亿), 流入资金, 流出资金, 涨跌幅, 领涨股]，按净额降序；
        获取失败返回 None。
    """
    global _industry_flow_cache
    now = time.time()
    if _industry_flow_cache and now - _industry_flow_cache[1] < INDUSTRY_FLOW_TTL:
        return _industry_flow_cache[0]
    try:
        df = ak.stock_fund_flow_industry(symbol="即时")
        if df is None or df.empty:
            return None
        keep = {"行业": "行业", "净额": "净额", "流入资金": "流入资金",
                "流出资金": "流出资金", "行业-涨跌幅": "涨跌幅", "领涨股": "领涨股"}
        df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
        for col in ("净额", "流入资金", "流出资金"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "涨跌幅" in df.columns:
            df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
        df = df.sort_values("净额", ascending=False).reset_index(drop=True)
        _industry_flow_cache = (df, now)
        return df
    except Exception as e:
        # ⚠️ 同样必须走 logger（原因同上：print 进不了日志文件）
        _logger.warning("[资金流快照] 行业资金流获取失败: %s: %s", type(e).__name__, e)
        return None


def get_stored_industry_flow():
    """从**落库快照**读行业资金流，返回 (数据日期, DataFrame[行业, 净额(亿), 涨跌幅, 领涨股])。

    与 `get_industry_flow_snapshot` 的区别：那个是现场拉（1.1 秒/次，带 600s 内存缓存），
    这个读盘后落库的表（≈0 秒）。**市场检测的"板块轮动"每轮选股都会跑一次**，
    所以它走这里；看盘页是可交互的实时场景，仍可用现场版。
    取不到返回 (None, None)，调用方自行降级。
    """
    from db import store as db_store
    try:
        date, rows = db_store.load_industry_flow_snapshot()
    except Exception as e:                           # noqa: BLE001
        _logger.warning("[资金流快照] 行业快照读库失败: %s", e)
        return None, None
    if not rows:
        return date, None
    df = pd.DataFrame([{
        "行业": r.get("industry"),
        "净额": (r.get("net_amount") or 0) / 1e8,     # 元 → 亿（与现场版口径一致）
        "涨跌幅": r.get("change_pct"),
        "领涨股": r.get("leader"),
    } for r in rows])
    return date, df


class FundFlowAkshareDataFetcher:
    """资金流向数据获取类（使用akshare数据源）"""
    
    def __init__(self):
        self.days = 30  # 获取最近30个交易日
        self.available = True
        _logger.info("[OK] 资金流向数据获取器初始化成功（akshare数据源）")
    
    def get_fund_flow_data(self, symbol):
        """
        获取个股资金流向数据
        
        Args:
            symbol: 股票代码（6位数字）
            
        Returns:
            dict: 包含资金流向数据的字典
        """
        data = {
            "symbol": symbol,
            "fund_flow_data": None,
            "data_success": False,
            "source": "akshare"
        }
        
        # 只支持中国股票
        if not self._is_chinese_stock(symbol):
            data["error"] = "资金流向数据仅支持中国A股股票"
            return data
        
        try:
            _logger.info(f"[资金流向] 正在获取 {symbol} 的资金流向数据...")
            
            # 确定市场
            market = self._get_market(symbol)
            
            # 获取资金流向数据
            fund_flow_data = self._get_individual_fund_flow(symbol, market)
            
            if fund_flow_data:
                data["fund_flow_data"] = fund_flow_data
                data["source"] = fund_flow_data.get("source", "akshare")
                _logger.info(f"   [OK] 成功获取资金流数据（{len(fund_flow_data.get('data', []))}条, 源: {data['source']}, 模式: {fund_flow_data.get('mode', 'daily')}）")
                data["data_success"] = True
                _logger.info("[完成] 资金流向数据获取完成")
            else:
                _logger.warning("[警告] 未能获取到资金流向数据")
                
        except Exception as e:
            _logger.warning(f"[ERROR] 获取资金流向数据失败: {e}")
            data["error"] = str(e)
        
        return data
    
    def _is_chinese_stock(self, symbol):
        """判断是否为中国股票"""
        return symbol.isdigit() and len(symbol) == 6
    
    def _get_market(self, symbol):
        """
        根据股票代码判断市场
        上海证券交易所: sh (60开头, 688开头)
        深圳证券交易所: sz (00开头, 30开头)
        北京证券交易所: bj (8开头, 4开头)
        """
        if symbol.startswith('60') or symbol.startswith('688'):
            return 'sh'
        elif symbol.startswith('00') or symbol.startswith('30'):
            return 'sz'
        elif symbol.startswith('8') or symbol.startswith('4'):
            return 'bj'
        else:
            # 默认深圳
            return 'sz'
    
    def _get_individual_fund_flow(self, symbol, market):
        """获取个股资金流向数据（降级链：东财逐股历史 → tushare → 同花顺全市场快照）"""
        source = "akshare"
        mode = "daily"
        try:
            # 优先使用akshare的stock_individual_fund_flow接口（带重试；熔断冷却期内直接跳过）
            akshare_df = None
            import data.fund_flow as _mod
            em_circuit_open = time.time() < _mod._em_circuit_open_until
            em_failed = False
            if em_circuit_open:
                _logger.warning("   [Akshare] 东财接口熔断中（近期连续失败），跳过重试直接降级")
                em_failed = True
            else:
                for retry_count in range(3):
                    try:
                        _logger.info(f"   [Akshare] 正在获取资金流向 (市场: {market})..." +
                              (f" (第{retry_count+1}次)" if retry_count > 0 else ""))

                        akshare_df = ak.stock_individual_fund_flow(stock=symbol, market=market)

                        if akshare_df is not None and not akshare_df.empty:
                            _logger.info(f"   [Akshare] 获取到 {len(akshare_df)} 条资金流向数据")
                            df = akshare_df
                            break
                    except Exception as e:
                        _logger.warning(f"   [Akshare] 获取失败: {e}")
                        if _is_connection_level_error(e):
                            # 连不上就没必要再试同一主机：立即熔断并走降级链
                            # （实测这一步把单只白等的 ~95s 降到一次连接超时的时间）
                            _logger.warning("   [Akshare] 连接层失败：立即熔断并降级"
                                  "（不对同一主机做无谓重试）")
                            _mod._em_circuit_open_until = time.time() + EM_CIRCUIT_COOLDOWN
                            em_failed = True
                            break
                        if retry_count < 2:
                            delay = (retry_count + 1) * 2
                            _logger.info(f"   [Akshare] {delay}s 后重试...")
                            time.sleep(delay)
                else:
                    # 3 次都没成功（内容层失败）→ 打开熔断
                    # （冷却期内调用不续期，保证冷却结束后能自愈探测）
                    _mod._em_circuit_open_until = time.time() + EM_CIRCUIT_COOLDOWN
                    em_failed = True

            if em_failed:
                # akshare全部失败，依次尝试 tushare → 同花顺快照
                df = None
                if data_source_manager.tushare_available and _moneyflow_allowed():
                    try:
                        _logger.info(f"   [Tushare] 正在获取资金流向数据...")
                        ts_code = data_source_manager._convert_to_ts_code(symbol)
                        end_date = datetime.now().strftime('%Y%m%d')
                        start_date = (datetime.now() - timedelta(days=self.days * 2)).strftime('%Y%m%d')
                        ts_df = data_source_manager.tushare_api.moneyflow(
                            ts_code=ts_code, start_date=start_date, end_date=end_date
                        )
                        if ts_df is not None and not ts_df.empty:
                            df = ts_df.rename(columns={
                                'trade_date': '日期', 'buy_sm_amount': '小单买入',
                                'sell_sm_amount': '小单卖出', 'buy_md_amount': '中单买入',
                                'sell_md_amount': '中单卖出', 'buy_lg_amount': '大单买入',
                                'sell_lg_amount': '大单卖出', 'buy_elg_amount': '超大单买入',
                                'sell_elg_amount': '超大单卖出', 'net_mf_amount': '净额'
                            })
                            # v2/M0 0.4：tushare moneyflow 的金额单位是**万元**，而东财同名
                            # 口径是**元**。直接混用会让 compute_five_dim_score 的档位阈值
                            # （1e7 / 5e7 / 1e8）低 1e4 倍而全部失准。
                            for col in ("小单买入", "小单卖出", "中单买入", "中单卖出",
                                        "大单买入", "大单卖出", "超大单买入", "超大单卖出",
                                        "净额"):
                                if col in df.columns:
                                    df[col] = pd.to_numeric(df[col], errors="coerce") * 10000
                            # 该源不直接提供下游评分必需的"主力净流入-净额"（缺失即 0 分），
                            # 按通用口径由大单 + 超大单净额推导（单位已是元）。
                            if {"超大单买入", "超大单卖出", "大单买入", "大单卖出"}.issubset(df.columns):
                                df["超大单净流入-净额"] = df["超大单买入"] - df["超大单卖出"]
                                df["大单净流入-净额"] = df["大单买入"] - df["大单卖出"]
                                df["主力净流入-净额"] = (df["超大单净流入-净额"]
                                                         + df["大单净流入-净额"])
                            df = df.head(self.days)
                            source = "tushare"
                            _logger.info(f"   [Tushare] 成功获取 {len(df)} 条资金流向数据")
                        else:
                            _logger.info(f"   [Tushare] 未找到资金流向数据")
                    except Exception as te:
                        _logger.warning(f"   [Tushare] 获取失败: {te}")
                        _note_moneyflow_error(te)
                if df is None:
                    df = self._get_ths_fallback_df(symbol)
                    if df is None:
                        _logger.warning("   [资金流] 全部数据源不可用（东财受限/Tushare未配置/同花顺快照失败）")
                        return None
                    source, mode = "ths_snapshot", "snapshot"

            # 处理数据
            df = df.tail(self.days)
            df = df.iloc[::-1].reset_index(drop=True)

            data_list = []
            for idx, row in df.iterrows():
                item = {}
                for col in df.columns:
                    value = row.get(col)
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        continue
                    try:
                        if isinstance(value, (int, float)):
                            item[col] = value
                        else:
                            item[col] = str(value)
                    except:
                        item[col] = "N/A"
                if item:
                    data_list.append(item)

            out = {
                "data": data_list,
                "days": len(data_list),
                "columns": df.columns.tolist(),
                "market": market,
                "source": source,
                "mode": mode,
                "query_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            # 快照模式必须把"数据日期"透出去（否则下游只看到 query_time，
            # 会把上一个交易日的资金流当成"刚查到的实时数据"）。
            if mode == "snapshot" and data_list:
                snap_date = data_list[-1].get("数据日期")
                if snap_date:
                    out["data_date"] = snap_date
            return out

        except Exception as e:
            _logger.warning(f"   获取资金流向数据异常: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_ths_fallback_df(self, symbol):
        """快照兜底：从**落库快照**读单只股票（单位元），取不到返回 None。

        v3（2026-09-25）：**不再现场拉全市场**。原实现走 `get_ths_snapshot_record`，
        而一个窗口就要拉全市场 5210 行（4 个窗口合计 75.9 秒），且内存缓存只有 300 秒，
        选股一轮 4~5 分钟必然中途过期重拉 → 一次选股白花约 150 秒。
        现在只读盘后 `capture_market_flow_snapshot` 落库的表，≈0.05 秒/只。
        """
        try:
            record = get_stored_snapshot_record(symbol)
            if record is None:
                _logger.info("   [资金流快照] 库内无该标的数据（盘后 15:40 采集任务是否已跑？）")
                return None
            _logger.info(f"   [资金流快照] 命中落库快照（{record.get('股票简称', symbol)}"
                  f"，数据日期 {record.get('数据日期')}）")
            return pd.DataFrame([record])
        except Exception as e:
            _logger.warning(f"   [资金流快照] 读取失败: {e}")
            return None
    
    def _format_snapshot_for_ai(self, data, fund_flow_data):
        """同花顺快照模式的AI文本格式化"""
        records = fund_flow_data.get('data', [])
        record = records[-1] if records else {}
        result = compute_five_dim_score(pd.DataFrame(records))

        def fmt_yi(key):
            value = record.get(key)
            return f"{float(value) / 1e8:.2f}亿" if isinstance(value, (int, float)) else "N/A"

        return f"""
【个股资金流向 · 落库快照】
股票代码：{data.get('symbol', 'N/A')} {record.get('股票简称', '')}
最新价：{record.get('最新价', 'N/A')}  涨跌幅：{record.get('涨跌幅', 'N/A')}
数据日期：{fund_flow_data.get('data_date') or record.get('数据日期', 'N/A')}（盘后采集的全市场快照，非逐日序列）
⚠ 数据源说明：东财逐日接口不可达，以下为盘后采集的各窗口累计净额（全单型合计，近似主力），无单型拆分与逐日序列。

窗口累计净额：
  - 今日: {fmt_yi('净额_今日')}
  - 3日: {fmt_yi('净额_3日')}
  - 5日: {fmt_yi('净额_5日')}
  - 10日: {fmt_yi('净额_10日')}

五维评分（快照模式）：{result.get('score', 0)}/{result.get('max', 25)}
{chr(10).join(f"  - {k}: {v['score']}/{v['max']} ({v['desc']})" for k, v in result.get('breakdown', {}).items())}
"""

    def format_fund_flow_for_ai(self, data):
        """
        将资金流向数据格式化为适合AI阅读的文本
        """
        if not data or not data.get("data_success"):
            return "未能获取资金流向数据"
        
        text_parts = []
        
        fund_flow_data = data.get("fund_flow_data")
        if fund_flow_data and fund_flow_data.get("mode") == "snapshot":
            return self._format_snapshot_for_ai(data, fund_flow_data)
        if fund_flow_data:
            text_parts.append(f"""
【个股资金流向数据 - akshare数据源】
股票代码：{data.get('symbol', 'N/A')}
市场：{fund_flow_data.get('market', 'N/A').upper()}
交易日数：最近{fund_flow_data.get('days', 0)}个交易日
查询时间：{fund_flow_data.get('query_time', 'N/A')}

═══════════════════════════════════════
[资金流向详细数据]
═══════════════════════════════════════
""")
            
            # 显示每个交易日的数据
            for idx, item in enumerate(fund_flow_data.get('data', []), 1):
                date = item.get('日期', 'N/A')
                close_price = item.get('收盘价', 'N/A')
                change_pct = item.get('涨跌幅', 'N/A')
                
                text_parts.append(f"""
第 {idx} 个交易日 ({date}):
  基本信息:
    - 收盘价: {close_price}
    - 涨跌幅: {change_pct}%
  
  主力资金:
    - 主力净流入-净额: {item.get('主力净流入-净额', 'N/A')}
    - 主力净流入-净占比: {item.get('主力净流入-净占比', 'N/A')}%
  
  超大单:
    - 超大单净流入-净额: {item.get('超大单净流入-净额', 'N/A')}
    - 超大单净流入-净占比: {item.get('超大单净流入-净占比', 'N/A')}%
  
  大单:
    - 大单净流入-净额: {item.get('大单净流入-净额', 'N/A')}
    - 大单净流入-净占比: {item.get('大单净流入-净占比', 'N/A')}%
  
  中单:
    - 中单净流入-净额: {item.get('中单净流入-净额', 'N/A')}
    - 中单净流入-净占比: {item.get('中单净流入-净占比', 'N/A')}%
  
  小单:
    - 小单净流入-净额: {item.get('小单净流入-净额', 'N/A')}
    - 小单净流入-净占比: {item.get('小单净流入-净占比', 'N/A')}%
""")
            
            # 添加统计汇总
            text_parts.append("""
═══════════════════════════════════════
[统计汇总 - 最近30个交易日]
═══════════════════════════════════════
""")
            
            # 计算统计数据
            data_list = fund_flow_data.get('data', [])
            if data_list:
                # 主力净流入统计
                main_inflow_list = [item.get('主力净流入-净额', 0) for item in data_list if isinstance(item.get('主力净流入-净额'), (int, float))]
                if main_inflow_list:
                    total_main_inflow = sum(main_inflow_list)
                    avg_main_inflow = total_main_inflow / len(main_inflow_list)
                    positive_days = len([x for x in main_inflow_list if x > 0])
                    negative_days = len([x for x in main_inflow_list if x < 0])
                    
                    text_parts.append(f"""
主力资金统计:
  - 累计净流入: {total_main_inflow:.2f}
  - 平均每日净流入: {avg_main_inflow:.2f}
  - 净流入天数: {positive_days}天
  - 净流出天数: {negative_days}天
  - 净流入占比: {positive_days/len(main_inflow_list)*100:.1f}%
""")
                
                # 涨跌幅统计
                change_pct_list = [item.get('涨跌幅', 0) for item in data_list if isinstance(item.get('涨跌幅'), (int, float))]
                if change_pct_list:
                    avg_change = sum(change_pct_list) / len(change_pct_list)
                    up_days = len([x for x in change_pct_list if x > 0])
                    down_days = len([x for x in change_pct_list if x < 0])
                    
                    text_parts.append(f"""
股价统计:
  - 平均涨跌幅: {avg_change:.2f}%
  - 上涨天数: {up_days}天
  - 下跌天数: {down_days}天
  - 上涨占比: {up_days/len(change_pct_list)*100:.1f}%
""")
        
        return "\n".join(text_parts)


# ================================================================
# 五维资金流向评分模型（可被 screener 直接导入）
# ================================================================

COL_MAIN_FLOW = "主力净流入-净额"
COL_SUPER_LARGE = "超大单净流入-净额"
COL_LARGE = "大单净流入-净额"

# 同花顺快照模式专用列
SNAPSHOT_COLS = {"净额_今日", "净额_5日"}


def _compute_snapshot_score(row: dict) -> dict:
    """
    快照模式五维评分（满分25分，同花顺降级源专用）。

    同花顺只提供 即时/3日/5日/10日 累计净额，无逐日序列与单型拆分，
    各维度以窗口符号/累计值作代理，语义与逐日模式存在差异：
      维度1 方向: 今日/3日/5日 三个窗口为正的数量 × 2（0-6分）
      维度2 金额: 5日累计净额分档（0-6分，阈值与逐日模式一致）
      维度3 结构: 同花顺无单型拆分 → **不可评**（不计分、也**不计入分母**）
      维度4 趋势: 近5日 vs 前5日（≈10日累计-5日累计）（0-5分）
      维度5 持续性: 今日/3日/5日窗口连续为正的层级（0-3分）

    ⚠️ 「结构」维度为什么是 `available=False` 而不是 0 分（2026-10-10 修正）：
    原实现写 `dim3 = 0` 并计入 25 分总分 —— 等于把"**取不到数据**"当成"**该维度最差**"，
    每只票被系统性扣掉 5 分，且**与逐日模式（有单型拆分）不可比**。
    用户实测报障的入口正是它：贤丰控股"五维评分 0/25"，其中 5 分是**无数据**造成的；
    贵州茅台 20/25 若剔除无效维度其实是 20/20（满分）。
    这与市场检测 14 指标的口径一致：**缺失指标既不计分也不计入分母**。
    """
    t = float(row.get("净额_今日", 0) or 0)
    d3 = float(row.get("净额_3日", 0) or 0)
    d5 = float(row.get("净额_5日", 0) or 0)
    d10 = float(row.get("净额_10日", 0) or 0)

    pos_windows = sum(1 for x in (t, d3, d5) if x > 0)
    dim1 = min(6, pos_windows * 2)

    if d5 > 1e8:       dim2 = 6
    elif d5 > 5e7:     dim2 = 5
    elif d5 > 1e7:     dim2 = 3
    elif d5 > 0:       dim2 = 1
    else:              dim2 = 0

    older_sum = d10 - d5
    if d5 > older_sum and d5 > 0:
        dim4, trend_desc = 5, "加速流入"
    elif d5 > older_sum:
        dim4, trend_desc = 3, "流出减弱"
    elif d5 > 0 and older_sum > d5:
        dim4, trend_desc = 2, "流入减速"
    else:
        dim4, trend_desc = 0, "持续流出"

    if t > 0 and d3 > 0 and d5 > 0:   dim5 = 3
    elif t > 0 and d5 > 0:            dim5 = 2
    elif t > 0:                       dim5 = 1
    else:                             dim5 = 0

    breakdown = {
        "方向": {"score": dim1, "max": 6, "available": True,
                 "desc": f"{pos_windows}/3窗口为正(今日/3日/5日)"},
        "金额": {"score": dim2, "max": 6, "available": True,
                 "desc": f"5日累计{d5/1e8:.2f}亿"},
        "结构": {"score": None, "max": 5, "available": False,
                 "desc": "同花顺快照源无单型拆分 —— 不可评，不计入分母"},
        "趋势": {"score": dim4, "max": 5, "available": True, "desc": trend_desc},
        "持续性": {"score": dim5, "max": 3, "available": True,
                   "desc": f"今日{'流入' if t > 0 else '流出'}"},
    }
    scored = [d for d in breakdown.values() if d["available"]]
    return {
        "score": round(sum(d["score"] for d in scored), 1),
        "max": sum(d["max"] for d in scored),          # 20（结构不可评）
        "max_total": 25,
        "unavailable_dims": [k for k, d in breakdown.items() if not d["available"]],
        "mode": "snapshot",
        "breakdown": breakdown,
        "total_flow_wan": round(d5 / 1e4, 0),
        "note": ("同花顺快照降级模式：净额为全单型合计（近似主力），无逐日序列；"
                 "「结构」维度因无单型拆分**不可评**，已从总分分母剔除（20 分制）"),
    }


def compute_five_dim_score(df) -> dict:
    """
    五维资金流向评分模型（满分25分）

    维度1: 方向 (0-6分) — 近5日净流入天数
    维度2: 金额 (0-6分) — 累计净流入金额分档
    维度3: 结构 (0-5分) — 超大单占主力比
    维度4: 趋势 (0-5分) — 近5日 vs 前5日变化
    维度5: 持续性 (0-3分) — 连续净流入天数

    返回: {"score": float, "max": 25, "breakdown": {...}}
    """
    import pandas as pd
    if df is None or (hasattr(df, 'empty') and df.empty):
        return {"score": 0, "max": 25, "breakdown": {}, "error": "数据为空"}

    # 路由：逐日模式（东财/tushare） vs 快照模式（同花顺降级源）
    if hasattr(df, 'columns'):
        if SNAPSHOT_COLS.issubset(df.columns) and COL_MAIN_FLOW not in df.columns:
            row = df.iloc[-1].to_dict() if len(df) else {}
            return _compute_snapshot_score(row)
        if COL_MAIN_FLOW not in df.columns:
            return {"score": 0, "max": 25, "breakdown": {}, "error": f"缺少必要列 {COL_MAIN_FLOW}"}

    # 确保有近5日数据
    if hasattr(df, 'tail'):
        recent_5 = df.tail(5)
        earlier_5 = df.iloc[-10:-5] if len(df) >= 10 else pd.DataFrame()
        recent_flows_raw = recent_5[COL_MAIN_FLOW].values
    else:
        recent_5 = df[-5:]
        earlier_5 = df[-10:-5] if len(df) >= 10 else []
        recent_flows_raw = [d.get(COL_MAIN_FLOW, 0) for d in recent_5]

    import numpy as np
    recent_flows = [d for d in recent_flows_raw if not (isinstance(d, float) and np.isnan(d))]
    if not recent_flows:
        return {"score": 0, "max": 25, "breakdown": {}, "error": "近5日无有效数据"}

    positive_days = sum(1 for f in recent_flows if f > 0)
    dim1 = min(6, positive_days * 1.2)

    total_flow = sum(recent_flows)
    if total_flow > 1e8:       dim2 = 6
    elif total_flow > 5e7:     dim2 = 5
    elif total_flow > 1e7:     dim2 = 3
    elif total_flow > 0:       dim2 = 1
    else:                      dim2 = 0

    # ⚠️ 缺「超大单净流入-净额」列时不能让整个评分崩：结构维度退化为 0 并在 desc 里说明。
    # 原实现直接 `recent_5[COL_SUPER_LARGE].values` —— 逐日源只回部分列时抛 KeyError，
    # 结果**整个五维评分连同 UI 的资金流向渲染一起失败**（2026-09-28 实测报障：
    # `资金流向数据渲染失败: '主力净流入_亿'`，追下去其实是这里先炸）。
    if not hasattr(recent_5, 'columns'):
        super_large_raw = [d.get(COL_SUPER_LARGE, 0) for d in recent_5]
        has_sl_col = True
    elif COL_SUPER_LARGE in recent_5.columns:
        super_large_raw = recent_5[COL_SUPER_LARGE].values
        has_sl_col = True
    else:
        super_large_raw = []
        has_sl_col = False
    super_large = [d for d in super_large_raw if not (isinstance(d, float) and np.isnan(d))]
    ratios = [sl / m for sl, m in zip(super_large, recent_flows) if m > 0 and sl > 0]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    # ⚠️ 与快照模式同一条口径（2026-10-10）：**"未评分"就不能当 0 分计入分母**。
    # 原实现把"缺列/无正比"都写成 score=0，但 desc 自己写着"未评分"——
    # 说未评分却按最差分算，等于凭空扣 5 分（详见 `_compute_snapshot_score` 的说明）。
    if not has_sl_col:
        dim3, dim3_ok = None, False
        dim3_desc = "缺「超大单净流入-净额」列 —— 不可评，不计入分母"
    elif not ratios:
        dim3, dim3_ok = None, False
        dim3_desc = "近5日无「主力净流入为正」的日子，超大单占比不可算 —— 不计入分母"
    else:
        dim3_ok = True
        if avg_ratio > 0.6:   dim3 = 5
        elif avg_ratio > 0.4: dim3 = 3
        elif avg_ratio > 0.2: dim3 = 1
        else:                 dim3 = 0
        dim3_desc = f"超大单占比{avg_ratio:.0%}"

    recent_sum = sum(recent_flows)
    if len(earlier_5) > 0:
        older_flows_raw = earlier_5[COL_MAIN_FLOW].values if hasattr(earlier_5, 'columns') else [d.get(COL_MAIN_FLOW, 0) for d in earlier_5]
        older_flows = [d for d in older_flows_raw if not (isinstance(d, float) and np.isnan(d))]
        older_sum = sum(older_flows)
        if recent_sum > older_sum and recent_sum > 0:
            dim4, trend_desc = 5, "加速流入"
        elif recent_sum > older_sum:
            dim4, trend_desc = 3, "流出减弱"
        elif recent_sum > 0 and older_sum > recent_sum:
            dim4, trend_desc = 2, "流入减速"
        else:
            dim4, trend_desc = 0, "持续流出"
    else:
        dim4, trend_desc = 3, "数据不足"

    consecutive = 0
    for f in reversed(recent_flows):
        if f > 0: consecutive += 1
        else:     break
    dim5 = min(3, consecutive)

    breakdown = {
        "方向": {"score": dim1, "max": 6, "available": True,
                 "desc": f"净流入{positive_days}/5天"},
        "金额": {"score": dim2, "max": 6, "available": True,
                 "desc": f"累计{total_flow/1e8:.2f}亿"},
        "结构": {"score": dim3, "max": 5, "available": dim3_ok, "desc": dim3_desc},
        "趋势": {"score": dim4, "max": 5, "available": True, "desc": trend_desc},
        "持续性": {"score": dim5, "max": 3, "available": True,
                   "desc": f"{consecutive}天连续"},
    }
    scored = [d for d in breakdown.values() if d["available"]]
    return {
        "score": round(sum(d["score"] for d in scored), 1),
        "max": sum(d["max"] for d in scored),
        "max_total": 25,
        "unavailable_dims": [k for k, d in breakdown.items() if not d["available"]],
        "breakdown": breakdown,
        "total_flow_wan": round(total_flow / 1e4, 0),
    }


# 测试函数
if __name__ == "__main__":
    print("测试资金流向数据获取（akshare数据源）...")
    print("="*60)
    
    fetcher = FundFlowAkshareDataFetcher()
    
    if not fetcher.available:
        print("[ERROR] 资金流向数据获取器不可用")
        sys.exit(1)
    
    # 测试股票
    test_symbols = [
        ("000001", "平安银行"),
        ("600519", "贵州茅台"),
        ("000858", "五粮液")
    ]
    
    for symbol, name in test_symbols:
        print(f"\n{'='*60}")
        print(f"正在测试股票: {name} ({symbol})")
        print(f"{'='*60}\n")
        
        data = fetcher.get_fund_flow_data(symbol)
        
        if data.get("data_success"):
            print("\n" + "="*60)
            print("资金流向数据获取成功！")
            print("="*60)
            
            formatted_text = fetcher.format_fund_flow_for_ai(data)
            # 只显示前2000个字符
            preview = formatted_text[:2000] if len(formatted_text) > 2000 else formatted_text
            print(preview)
            if len(formatted_text) > 2000:
                print(f"... (共 {len(formatted_text)} 字符)")
        else:
            print(f"\n获取失败: {data.get('error', '未知错误')}")
        
        print("\n")

