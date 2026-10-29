# -*- coding: utf-8 -*-
"""
模块名: data/panel_store.py
用途: 全市场日线**面板**的落盘与读取（BaoStock 历史回填 + parquet 分片存储）

为什么需要它（2026-10-17 用户拍板：BaoStock 回填历史估值、存 `data/panel/`）：
  tushare `daily_basic` 限频 **1 次/小时** ⇒ 估值类因子（PE/PB/换手）的**历史**不可能
  从 tushare 回填；而因子 IC 研究恰恰需要"当时的截面"（否则前视）。
  BaoStock（免注册/免 token）实测 1.57s/只、**20/20 无限频**，字段含
  `turn/peTTM/pbMRQ/psTTM/isST/tradestatus/pctChg` ⇒ 全市场 10 年 ≈5 小时（一次性）。

存储形态（用户指定 `data/panel/`）：
  `data/panel/parts/daily_part_NNNN.parquet` —— **按批分片**（默认 200 只/片 ≈15MB）。
  为什么分片而不是单文件：① 回填 5 小时，中途可续跑（`progress.json` 记已完成代码）；
  ② 单文件 400MB 每次追加都要整体重写；③ 读取时 glob 合并 + 去重即可。

⚠️⚠️ **pandas≥2 必须先打 `DataFrame.append` 垫片**（`apply_baostock_pandas_shim`）：
  BaoStock 的 `ResultSet.get_data()` 在结果**超过一页**时调用 `df.append`，
  pandas 2.x 已移除该方法 ⇒ **1 年查询正常、10 年查询直接崩**（实测）。

⚠️ **BaoStock 不覆盖北交所**（实测 `bj.*` 报"股票代码未标识sh或sz"）⇒ 沪深 ~5100 只，
  覆盖率 ~91.6%。缺口必须走 `data_coverage` 表如实落库（P2-7），不许当作 100%。
"""
from __future__ import annotations

import concurrent.futures as futures
import io
import json
import logging
import os
import time
from typing import Callable, Iterable, Optional

import pandas as pd

_logger = logging.getLogger(__name__)

# ================================================================
# 路径
# ================================================================

# 数据根必须**运行时**解析：打包后 `__file__` 在只读包 `_internal/` 里，
# 用它推导会指向空目录 ⇒ 读到空数据（2026-10-25 实测）。
# 规则见 `config/paths.data_root()`；源码运行结果与原先一致。
from config.paths import data_root as _paths_data_root  # noqa: E402
_PROJECT_ROOT = _paths_data_root()
PANEL_DIR = os.path.join(_PROJECT_ROOT, "data", "panel")
PARTS_DIR = os.path.join(PANEL_DIR, "parts")
PROGRESS_PATH = os.path.join(PANEL_DIR, "progress.json")

# ================================================================
# BaoStock 适配
# ================================================================

# 字段顺序即查询串；`turn` 是**百分数**、`volume` 单位**股**、`amount` 单位**元**
FIELDS_BAOSTOCK = ("date,code,open,high,low,close,preclose,volume,amount,"
                   "turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,isST")

_NUMERIC_COLS = ("open", "high", "low", "close", "preclose", "volume",
                 "amount", "turn", "pctChg", "peTTM", "pbMRQ", "psTTM")

# 归一化后的列名（落 parquet 的 schema —— 唯一定义处）
PANEL_COLUMNS = ["stock_code", "trade_date", "open", "high", "low", "close",
                 "preclose", "volume", "amount", "turn", "pct_chg",
                 "pe_ttm", "pb_mrq", "ps_ttm", "trade_status", "is_st"]


def apply_baostock_pandas_shim() -> None:
    """给 pandas 2.x 补回 `DataFrame.append`（BaoStock 分页必需）。

    ⚠️ 幂等：重复调用无害。实测不加这个垫片：1 年查询正常、10 年查询直接崩
    （`'DataFrame' object has no attribute 'append'`）。
    """
    if hasattr(pd.DataFrame, "append"):
        return

    def _append(self: pd.DataFrame, other, ignore_index: bool = False) -> pd.DataFrame:
        # BaoStock 内部逐页 concat；等价实现且只走 pandas 公开 API
        return pd.concat([self, other], ignore_index=ignore_index)

    pd.DataFrame.append = _append  # type: ignore[attr-defined]
    _logger.info("[panel] 已打 BaoStock/pandas2 兼容垫片（DataFrame.append）")


def bs_code(code6: str) -> Optional[str]:
    """6 位代码 → BaoStock 代码（`sh.600028` / `sz.000001`）。

    ⚠️ 北交所（4/8/9 开头）BaoStock **不覆盖** ⇒ 返回 None（调用方跳过并计数，
    **不许**猜一个前缀硬查——实测 `sh.920196` 返回 0 行、`bj.*` 直接报错）。
    """
    c = str(code6 or "").strip()
    if len(c) != 6 or not c.isdigit():
        return None
    if c[0] == "6":
        return f"sh.{c}"
    if c[0] in ("0", "3"):
        return f"sz.{c}"
    return None                                    # 4/8/9 开头（北交所/退市板）


def normalize_baostock_df(raw: pd.DataFrame, code6: str) -> pd.DataFrame:
    """BaoStock 返回的**字符串表** → 归一化面板（含 dtype 与缺失处理）。

    ⚠️ BaoStock 用**空字符串**表示缺失 ⇒ `pd.to_numeric(errors="coerce")`
       恰好把 `""` 变成 NaN（**不要**写 `fillna(0)`——缺就是缺）。
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    df = raw.copy()
    if "date" not in df.columns:
        raise ValueError("BaoStock 返回缺少 date 列")
    df = df.rename(columns={"date": "trade_date", "pctChg": "pct_chg",
                            "peTTM": "pe_ttm", "pbMRQ": "pb_mrq", "psTTM": "ps_ttm",
                            "tradestatus": "trade_status", "isST": "is_st"})
    df["stock_code"] = str(code6)
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("pct_chg", "pe_ttm", "pb_mrq", "ps_ttm", "trade_status", "is_st"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("trade_status", "is_st"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")        # 可空整数（停牌日可能缺）
    keep = [c for c in PANEL_COLUMNS if c in df.columns]
    out = df[keep].sort_values("trade_date").reset_index(drop=True)
    return out


# ================================================================
# 进度（断点续跑）
# ================================================================

def _assert_not_offline(what: str) -> None:
    """离线模式（`BIGA_OFFLINE=1`）下拒绝会联网的操作（R4，2026-10-18 审查）。

    为什么必须显式拒绝：BaoStock/tushare 走裸 TCP，request 层闸门拦不到
    （`offline_gate` 只对它们**计数**）。离线模式的意义就是"一个包都不发"，
    悄悄跑起来会把这条能力变成假的。
    """
    try:
        from data import offline_gate
    except Exception as e:                                  # noqa: BLE001
        _logger.debug("[panel] offline_gate 不可用（按在线处理）: %s", e)
        return
    if offline_gate.is_offline():
        raise RuntimeError(
            f"BIGA_OFFLINE=1 下禁止{what}（会联网取数）。"
            "如需联网请先取消该环境变量。")


def _read_progress() -> dict:
    if not os.path.exists(PROGRESS_PATH):
        return {"done": {}, "failed": {}, "updated": None}
    try:
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "done" not in data:
            raise ValueError("progress.json 结构不对")
        return data
    except (json.JSONDecodeError, OSError, ValueError) as e:
        # ⚠️ 进度文件损坏 ⇒ **宁可重跑也不静默当"全部完成"**
        _logger.warning("[panel] progress.json 不可读（%s: %s）⇒ 视为无进度重跑",
                        type(e).__name__, e)
        return {"done": {}, "failed": {}, "updated": None}


def _write_progress(done: dict, failed: dict) -> None:
    """写按股进度 —— **必须保留按日键**（done_dates/failed_dates）。

    ⚠️⚠️ 2026-10-18 事故：本函数原来**整文件重写**只写 done/failed，
    把按日回填的 done_dates/failed_dates（2618 天）**静默抹掉** ——
    一次 BaoStock 按股续跑就把 tushare 的按日进度清零，下次按日回填会从头再来
    （浪费 74 分钟 + 2618 次限频调用）。统一走 `_write_progress_dates_full`。
    """
    _write_progress_full(done, failed)


# ================================================================
# 分片读写
# ================================================================

def save_part(df: pd.DataFrame) -> str:
    """写一个分片（文件名带序号与时间，避免并发覆盖）。"""
    os.makedirs(PARTS_DIR, exist_ok=True)
    n = len([f for f in os.listdir(PARTS_DIR) if f.endswith(".parquet")])
    path = os.path.join(PARTS_DIR, f"daily_part_{n:04d}.parquet")
    out = df.copy()
    out["stock_code"] = out["stock_code"].astype("category")   # 分类列省 ~60% 体积
    out.to_parquet(path, index=False)
    return path


def load_panel(start: str | None = None, end: str | None = None,
               codes: Iterable[str] | None = None) -> pd.DataFrame:
    """读全部分片 → 去重 → 按区间过滤的长表（`stock_code, trade_date, …`）。

    ⚠️ **去重规则 = `(stock_code, trade_date)` 保留最后出现的**：续跑/重跑会产生
       重复行（同一只股票可能出现在两个分片里），保留"后写的"（更新）语义。
    """
    parts = sorted(f for f in os.listdir(PARTS_DIR)
                   if f.endswith(".parquet")) if os.path.isdir(PARTS_DIR) else []
    if not parts:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    frames = [pd.read_parquet(os.path.join(PARTS_DIR, f)) for f in parts]
    df = pd.concat(frames, ignore_index=True)
    df["stock_code"] = df["stock_code"].astype(str)
    # ⚠️ 读取侧**防御性归一**：即使某个写路径漏了归一（tushare 8 位日期混入分片），
    #    过滤/去重前也先把格式对齐 —— 字符串区间过滤只在唯一格式下才可靠。
    df["trade_date"] = df["trade_date"].map(_norm_date_str)
    df = df.drop_duplicates(subset=["stock_code", "trade_date"], keep="last")
    if codes is not None:
        want = {str(c) for c in codes}
        df = df[df["stock_code"].isin(want)]
    if start:
        df = df[df["trade_date"] >= str(start)]
    if end:
        df = df[df["trade_date"] <= str(end)]
    return df.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)


def panel_status() -> dict:
    """面板概况（给 CLI/页面用）：分片数、行数、代码数、日期范围。

    ⚠️ 内部走 `load_panel()` ⇒ 11M 行下要**分钟级**（全量 dedup + sort）。
       页面/高频调用请用 `read_overview()`（读 `data/panel/overview.json`）。
    """
    df = load_panel()
    if df.empty:
        return {"parts": len([f for f in os.listdir(PARTS_DIR)
                              if f.endswith(".parquet")]) if os.path.isdir(PARTS_DIR) else 0,
                "rows": 0, "codes": 0, "date_min": None, "date_max": None}
    return {"parts": len([f for f in os.listdir(PARTS_DIR) if f.endswith(".parquet")]),
            "rows": int(len(df)), "codes": int(df["stock_code"].nunique()),
            "date_min": str(df["trade_date"].min()),
            "date_max": str(df["trade_date"].max())}


# ================================================================
# 面板概况落盘（P0-3，2026-10-18）
# ================================================================
#
# 为什么需要一份 JSON，而不是让页面调 panel_status()：
#   实测 `load_panel()` 在 11.3M 行上要 **分钟级**（全量 drop_duplicates +
#   sort_values）⇒ 页 7 冒烟直接 AppTest 超时 120s。概况是"每天回填后变一次"
#   的事实，落盘一次、页面毫秒读取，才是正确形态（也符合 P0-3 的
#   "运营元数据与事实对齐"——元数据必须是产物，而不是每次现算）。

OVERVIEW_PATH = os.path.join(PANEL_DIR, "overview.json")


def build_overview() -> dict:
    """逐分片聚合面板概况 —— **不 dedup、不排序** ⇒ 秒级（vs load_panel 分钟级）。

    ⚠️ `rows_disk` 是**落盘行数**（含续跑产生的同键重复，实测约 +0.4%）；
       真实可用行数请用 `load_panel()`（会去重）。
    """
    parts = sorted(f for f in os.listdir(PARTS_DIR)
                   if f.endswith(".parquet")) if os.path.isdir(PARTS_DIR) else []
    rows = 0
    codes: set[str] = set()
    val_codes: set[str] = set()
    dmin = dmax = None
    for name in parts:
        d = pd.read_parquet(os.path.join(PARTS_DIR, name),
                            columns=["stock_code", "trade_date", "pe_ttm"])
        rows += int(len(d))
        d["stock_code"] = d["stock_code"].astype(str)
        d["trade_date"] = d["trade_date"].map(_norm_date_str)   # 兼容历史 8 位格式
        codes |= set(d["stock_code"].unique())
        val_codes |= set(d.loc[d["pe_ttm"].notna(), "stock_code"].unique())
        if not d.empty:
            lo, hi = str(d["trade_date"].min()), str(d["trade_date"].max())
            dmin = lo if dmin is None else min(dmin, lo)
            dmax = hi if dmax is None else max(dmax, hi)
    return {"parts": len(parts), "rows_disk": rows, "codes": len(codes),
            "valued_codes": len(val_codes), "date_min": dmin, "date_max": dmax}


def write_overview() -> dict:
    """聚合 + 落盘 `data/panel/overview.json`（原子替换）。回填收尾应调用一次。"""
    ov = build_overview()
    ov["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(PANEL_DIR, exist_ok=True)
    tmp = OVERVIEW_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ov, f, ensure_ascii=False, indent=1)
    os.replace(tmp, OVERVIEW_PATH)
    _logger.info("[panel] 概况已落盘: %s", ov)
    return ov


def read_overview() -> dict | None:
    """读 `overview.json`（毫秒级）。缺失/损坏 ⇒ None（调用方自己决定回退口径）。"""
    try:
        with open(OVERVIEW_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


# ================================================================
# 回填主流程（fetch_one 可注入 ⇒ 测试不联网）
# ================================================================

def run_backfill(*, codes: list[str], start: str, end: str,
                 fetch_one: Callable[[str, str, str], pd.DataFrame],
                 batch_size: int = 200, resume: bool = True,
                 sleep: float = 0.0,
                 progress_every: int = 100) -> dict:
    # ⚠️ R4（2026-10-18 审查）：离线模式下**禁止回填**。
    #    BaoStock/tushare 都是裸 TCP，`offline_gate` 只**计数**、拦不住；
    #    与其"静默照跑"，不如明确拒绝——离线模式的全部意义就是不出网。
    _assert_not_offline("回填")
    """逐只回填并按批落 parquet。

    Args:
        codes: **6 位代码**清单（北交所会被 `bs_code` 判为 None 并跳过计数）。
        fetch_one: `(code6, start, end) → BaoStock 原始字符串表`。
                   **按调用时刻注入** ⇒ 单测可给假实现，不联网。
        resume: True 时跳过 progress.json 里已完成的代码（断点续跑）。
        sleep: 每只之间的礼貌间隔（BaoStock 实测无硬限频，默认 0）。

    Returns:
        {"codes_total", "skipped_done", "fetched", "empty", "failed",
         "rows", "parts", "elapsed_s", "elapsed_per_code"}

    ⚠️ 单只失败**不中断**整体（记录进 progress.json 的 failed，可下次重跑）；
       这与"宁可当日不成交也不用错价成交"同一思路：宁可缺一只，不造假数据。
    """
    t0 = time.time()
    prog = _read_progress()
    done: dict = dict(prog.get("done") or {})
    failed: dict = dict(prog.get("failed") or {})

    buffer: list[pd.DataFrame] = []
    # ⚠️⚠️ 与 buffer 一一对应：**落盘成功后才记 done**（2026-10-18 事故：
    #   先记 done 后落盘 ⇒ 进程在满批前被杀，50 只被标"完成"但数据从未写入分片，
    #   续跑会永久跳过它们 —— "假完成"比"失败"更危险，因为它不会被重试）。
    pending: list[tuple[str, int]] = []
    stats = {"codes_total": len(codes), "skipped_done": 0, "fetched": 0,
             "empty": 0, "failed": 0, "rows": 0, "parts": 0}
    n_parts = len([f for f in os.listdir(PARTS_DIR) if f.endswith(".parquet")]) \
        if os.path.isdir(PARTS_DIR) else 0

    def _flush(force: bool = False) -> None:
        """落盘 + **落盘成功后才写 done**（顺序与并行两版共用同一语义）。"""
        nonlocal n_parts
        if not buffer:
            return
        if not (force or len(buffer) >= max(1, batch_size)):
            return
        save_part(pd.concat(buffer, ignore_index=True))
        n_parts += 1
        stats["parts"] = n_parts
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for c, n in pending:
            done[c] = {"rows": int(n), "at": stamp}
        buffer.clear()
        pending.clear()

    for i, code6 in enumerate(codes, 1):
        bs = bs_code(code6)
        if bs is None:
            failed[code6] = "北交所/非沪深代码，BaoStock 不覆盖"
            stats["failed"] += 1
            continue
        if resume and code6 in done:
            stats["skipped_done"] += 1
            continue
        try:
            raw = fetch_one(code6, start, end)
            df = normalize_baostock_df(raw, code6)
        except Exception as e:                              # noqa: BLE001
            # 具体记录：哪只、什么异常 —— 失败原因必须可查
            failed[code6] = f"{type(e).__name__}: {str(e)[:120]}"
            stats["failed"] += 1
            _logger.warning("[panel] %s 回填失败（跳过，可续跑重试）: %s", code6, failed[code6])
            continue
        stats["fetched"] += 1
        stats["rows"] += int(len(df))
        if df.empty:
            # 空行没有数据可丢 ⇒ 可以直接记 done（新股/长期停牌，"空"也是事实）
            done[code6] = {"rows": 0, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
            stats["empty"] += 1
        else:
            buffer.append(df)
            pending.append((code6, len(df)))
            _flush()
        if progress_every and i % progress_every == 0:
            _write_progress(done, failed)                    # 中途崩了也不白跑
            _logger.info("[panel] %d/%d 已完成 %d（本批落片 %d）",
                         i, len(codes), stats["fetched"], n_parts)
        if sleep > 0:
            time.sleep(sleep)

    _flush(force=True)
    stats["parts"] = n_parts
    stats["elapsed_s"] = round(time.time() - t0, 1)
    stats["elapsed_per_code"] = round(stats["elapsed_s"] / max(1, stats["fetched"]), 2)
    _write_progress(done, failed)
    return stats


def load_stock_universe(*, listed_only: bool = True) -> tuple[list[str], dict]:
    """从 BaoStock 拿标的清单（`query_stock_basic`，需先打垫片 + login）。

    Returns:
        (6 位代码清单, 概况)。清单按 `type=="1"`（股票）过滤；
        `listed_only=True` 再过滤 `status=="1"`（在市）。

    ⚠️ **为什么默认在市、但研究建议带退市**：IC 若只用现存股票，会踩
       **幸存者偏差**（退市的差股票被剔除 ⇒ 因子显得更好）。做因子研究请
       `listed_only=False`。实测清单 9011 行（含指数/基金），股票 5563 只。
    """
    apply_baostock_pandas_shim()
    import baostock as bs                                   # 垫片之后才能导入
    rs = bs.query_stock_basic()
    df = rs.get_data()
    if df is None or df.empty:
        raise RuntimeError(f"query_stock_basic 返回空（error_code={rs.error_code}）")
    stocks = df[df["type"] == "1"].copy()
    if listed_only:
        stocks = stocks[stocks["status"] == "1"]
    codes = sorted({c.split(".")[-1] for c in stocks["code"].astype(str)
                    if bs_code(c.split(".")[-1])})           # 顺手剔北交所
    info = {"raw_rows": int(len(df)), "stock_rows": int(len(stocks)),
            "codes": len(codes), "listed_only": bool(listed_only),
            "delisted_excluded": int(len(df[df['type'] == '1']) - len(stocks))}
    return codes, info


def record_backfill_coverage(stats: dict, codes_total: int,
                             *, start: str, end: str) -> float | None:
    """把回填结果写进 `data_coverage`（P2-7 的表）——**缺口必须如实落库**。

    ⚠️ 北交所不在 BaoStock 覆盖内 ⇒ `got < total` 是**已知且必须显示的**，
       不能因为"任务成功"就记 ok/100%。
    """
    from db import store as db_store                       # 延迟导入（分层允许 data→db）
    got = int(stats.get("fetched") or 0)
    failed = int(stats.get("failed") or 0)
    if got == 0:
        status = db_store.COVERAGE_FAILED
    elif failed > 0 or got < codes_total:
        status = db_store.COVERAGE_PARTIAL
    else:
        status = db_store.COVERAGE_OK
    return db_store.save_data_coverage(
        end, "baostock_backfill", got=got, total=int(codes_total), status=status,
        source="baostock",
        note=(f"区间 {start}~{end}；失败 {failed} 只；"
              f"北交所不在 BaoStock 覆盖内；详见 data/panel/progress.json"))


# ================================================================
# 真实取数（模块级 ⇒ 可被子进程 pickle 引用）
# ================================================================

def fetch_one_baostock(code6: str, start: str, end: str) -> pd.DataFrame:
    """单只全区间真实取数 —— **按年分段**（实测决定，不是风格偏好）：

    - 一次性查 10 年：单只 **11~66s 且波动大**（结果超一页走分页通道）⇒ 外推 1~2 天；
    - 按年分段：每段 ≤~250 行、1.5~2.9s/段 ⇒ 单只 17~32s，且**每年不超一页**
      ⇒ 天然绕开 `DataFrame.append` 分页路径（垫片只作兜底）。

    ⚠️ 需已 login（顺序路径由调用方负责；并行路径由 `_mp_worker_init` 负责）。
    """
    import baostock as bs                                   # 垫片之后才能导入
    b = bs_code(code6)
    if b is None:
        raise RuntimeError("北交所/非沪深代码，BaoStock 不覆盖")
    frames = []
    y0, y1 = int(str(start)[:4]), int(str(end)[:4])
    for y in range(y0, y1 + 1):
        s = max(f"{y}-01-01", str(start))
        e = min(f"{y}-12-31", str(end))
        rs = bs.query_history_k_data_plus(
            b, FIELDS_BAOSTOCK, start_date=s, end_date=e,
            frequency="d", adjustflag="3")
        if rs.error_code != "0":
            raise RuntimeError(f"BaoStock {rs.error_code}: {rs.error_msg}（{code6} {y}）")
        frames.append(rs.get_data())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---- 子进程会话上下文（仅 worker 进程内使用；父进程不碰） ----
_MP_CTX: dict = {}


def mp_worker_init(start: str, end: str) -> None:
    """并行 worker 的初始化：**每个子进程自己 login**（BaoStock 连接是模块级
    全局 socket ⇒ 多进程共享一个连接会串包，必须进程隔离）。"""
    _MP_CTX["start"], _MP_CTX["end"] = str(start), str(end)
    apply_baostock_pandas_shim()
    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock worker 登录失败: {lg.error_msg}")


def mp_fetch_one(code6: str) -> tuple:
    """子进程任务：单只全区间 → (code6, status, rows, payload)。

    payload：ok ⇒ 归一化 records（dict 列表，体积小、pickle 安全）；
             error ⇒ 错误串。⚠️ 断线重登**一次**再试（几小时任务必然遇到断连）。
    """
    import baostock as bs
    last_err = ""
    for attempt in (1, 2):
        try:
            df = normalize_baostock_df(
                fetch_one_baostock(code6, _MP_CTX["start"], _MP_CTX["end"]), code6)
            return code6, "ok", int(len(df)), df.to_dict("records")
        except Exception as e:                              # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:150]}"
            _logger.warning("[panel] %s 取数失败（第 %d 次尝试）: %s",
                            code6, attempt, last_err)
            if "黑名单" in last_err:
                # ⚠️⚠️ 并发会话超限触发服务器保护 —— **重登只会延长拉黑**，
                #    立即把标记传回父进程熔断（见 run_backfill_parallel）。
                return code6, "blacklist", 0, last_err
            if attempt == 1:
                try:
                    bs.logout()
                except Exception as e2:                     # noqa: BLE001
                    _logger.debug("[panel] %s logout 清理失败（继续重登）: %s", code6, e2)
                lg = bs.login()
                if lg.error_code != "0":
                    return code6, "blacklist" if "黑名单" in str(lg.error_msg) else "error", \
                        0, f"重登失败: {lg.error_msg}"
                continue
    return code6, "error", 0, last_err


class _InlineExecutor:
    """测试用的**内联**执行器：submit 即在当前进程执行（不 spawn、不 pickle）
    ⇒ `run_backfill_parallel` 的编排逻辑可以完全离线测试。"""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        fut = futures.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as e:                              # noqa: BLE001
            fut.set_exception(e)
        return fut


def run_backfill_parallel(*, codes: list[str], start: str, end: str,
                          worker_fetch: Callable,
                          workers: int = 6, batch_size: int = 200,
                          resume: bool = True, progress_every: int = 25,
                          stall_timeout: int = 900,
                          executor_factory: Callable | None = None) -> dict:
    # ⚠️ R4（2026-10-18 审查）：离线模式下**禁止回填**。
    #    BaoStock/tushare 都是裸 TCP，`offline_gate` 只**计数**、拦不住；
    #    与其"静默照跑"，不如明确拒绝——离线模式的全部意义就是不出网。
    _assert_not_offline("回填")
    """并行回填编排 —— 与 `run_backfill` 同一套落盘/进度/统计语义。

    ⚠️⚠️ **父进程是唯一写者**：分片 parquet 与 progress.json 只由本进程写
       （worker 只取数）⇒ 不存在多进程写同一文件的竞态。
    ⚠️ `worker_fetch` 必须是**模块级可 pickle** 的引用（默认走
       `mp_fetch_one`）；测试可注入内联执行器 + 普通函数。
    ⚠️⚠️ **看门狗 `stall_timeout`（默认 900s）**：BaoStock 的请求**没有超时**
       ⇒ 单只取数可能永久 hang（2026-10-18 实测：14:18 之后 26 分钟零进展，
       父进程 1.6GB 内存一直挂着）。这里改成 `wait(timeout=...)`：
       超过 `stall_timeout` 秒**一只都没完成** ⇒ 熔断退出（先 `_flush` 保住
       已完成的行、再写进度），由外部重启续跑。
       为什么按"无任何完成"判：正常 25 只要 2~5 分钟，15 分钟零完成必然是卡死。
    """
    import concurrent.futures as futures_mod

    t0 = time.time()
    prog = _read_progress()
    done: dict = dict(prog.get("done") or {})
    failed: dict = dict(prog.get("failed") or {})

    stats = {"codes_total": len(codes), "skipped_done": 0, "fetched": 0,
             "empty": 0, "failed": 0, "rows": 0, "parts": 0}
    todo: list[str] = []
    for code6 in codes:
        if bs_code(code6) is None:                          # 北交所：就地判掉（同顺序版语义）
            failed[code6] = "北交所/非沪深代码，BaoStock 不覆盖"
            stats["failed"] += 1
            continue
        if resume and code6 in done:
            stats["skipped_done"] += 1
            continue
        todo.append(code6)

    n_parts = len([f for f in os.listdir(PARTS_DIR) if f.endswith(".parquet")]) \
        if os.path.isdir(PARTS_DIR) else 0
    buffer: list[pd.DataFrame] = []
    # ⚠️⚠️ 落盘成功后才记 done（2026-10-18 事故：50 只记了完成但 buffer 未 flush，
    #    进程被杀 ⇒ 数据丢失且续跑永久跳过 ⇒ "假完成"）。
    pending_parts: list[tuple[str, int]] = []

    def _flush(force: bool = False) -> None:
        nonlocal n_parts
        if not buffer:
            return
        if not (force or len(buffer) >= max(1, batch_size)):
            return
        save_part(pd.concat(buffer, ignore_index=True))
        n_parts += 1
        stats["parts"] = n_parts
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for c, n in pending_parts:
            done[c] = {"rows": int(n), "at": stamp}
        buffer.clear()
        pending_parts.clear()

    if executor_factory is None:
        def executor_factory(n_workers: int):
            from concurrent.futures import ProcessPoolExecutor
            return ProcessPoolExecutor(max_workers=max(1, n_workers),
                                       initializer=mp_worker_init,
                                       initargs=(str(start), str(end)))
    with executor_factory(workers) as ex:
        code_of = {ex.submit(worker_fetch, c): c for c in todo}
        pending = set(code_of)
        n_done = 0
        while pending:
            # ⚠️ 用 wait(timeout=) 而不是 as_completed：后者会无限阻塞，
            #    BaoStock 请求无超时 ⇒ 单只 hang 就把任务永久挂住（实测 26 分钟零进展）。
            finished, pending = futures_mod.wait(
                set(pending), timeout=max(30, int(stall_timeout)),
                return_when=futures_mod.FIRST_COMPLETED)
            if not finished:
                _logger.error(
                    "[panel] ⚠️ 已 %d 分钟无任何完成（剩 %d 只）⇒ 判定取数 hang，熔断退出；"
                    "已完成的行会先落盘，重启即续跑",
                    max(1, int(stall_timeout)) // 60, len(pending))
                for f2 in set(pending):
                    f2.cancel()
                stats["stalled"] = True
                _flush(force=True)                          # 保住已取到的行（别白跑）
                _write_progress(done, failed)
                return stats
            for fut in finished:
                code6 = code_of[fut]
                try:
                    _, status, rows, payload = fut.result()
                except Exception as e:                      # noqa: BLE001
                    # worker 抛异常（如登录失败）也算单只失败，不中断整体
                    _logger.warning("[panel] worker 异常 %s: %s", code6, e)
                    failed[code6] = f"{type(e).__name__}: {str(e)[:150]}"
                    stats["failed"] += 1
                    n_done += 1
                    continue
                if status == "blacklist":
                    # ⚠️⚠️ 熔断：多会话并发触发 BaoStock 服务器保护（实测 6 worker +
                    #    父进程 = 7 并发被拉黑；3 worker = 4 并发安全）⇒ 立即终止，
                    #    保留进度（done/failed 已写）⇒ 降并发后续跑。
                    for f2 in set(pending):
                        f2.cancel()
                    _logger.error("[panel] ⚠️ BaoStock「黑名单」——并发会话超限触发服务器保护，"
                                  "立即熔断。等 10~30 分钟后用更少 worker（≤3）续跑")
                    stats["blacklisted"] = True
                    _flush(force=True)                      # 同样先保住已取到的行
                    _write_progress(done, failed)
                    return stats
                if status == "ok":
                    stats["fetched"] += 1
                    stats["rows"] += int(rows)
                    if not payload:
                        stats["empty"] += 1                 # 0 行也是事实
                        done[code6] = {"rows": 0,
                                       "at": time.strftime("%Y-%m-%d %H:%M:%S")}
                    else:
                        buffer.append(pd.DataFrame(payload))
                        pending_parts.append((code6, int(rows)))
                        _flush()
                else:
                    failed[code6] = str(payload)
                    stats["failed"] += 1
                n_done += 1
                if progress_every and n_done % progress_every == 0:
                    _write_progress(done, failed)           # 中途崩了不白跑
                    _logger.info("[panel] 并行 %d/%d 完成（成 %d 败 %d，分片 %d）",
                                 n_done, len(todo), stats["fetched"], stats["failed"], n_parts)

    _flush(force=True)
    stats["parts"] = n_parts
    stats["elapsed_s"] = round(time.time() - t0, 1)
    stats["elapsed_per_code"] = round(stats["elapsed_s"] / max(1, stats["fetched"]), 2)
    stats["workers"] = int(workers)
    _write_progress(done, failed)
    return stats


# ================================================================
# tushare 按日回填（备选路径：BaoStock 黑名单期间的价量历史）
# ================================================================

def _norm_date_str(s) -> str:
    """日期归一：`20240517`（tushare 口径）→ `2024-05-17`（面板唯一口径）。

    ⚠️⚠️ 面板 `trade_date` 的**唯一合法格式是 `YYYY-MM-DD`**。混入 `YYYYMMDD`
    会让 `load_panel` 的字符串区间过滤**双向出错**（`'20240101' > '2024-01-02'`
    字符串比较为 True，语义上却是更早的一天），同键去重也随之失效
    （同一只股票同一天两种格式各一行 ⇒ 双行）。tushare `daily` 返回的
    `trade_date` 就是 8 位无连字符 ⇒ 写入前必须归一（2026-10-17 实测踩坑：
    按日回填 1101 万行全是不带连字符的格式，靠此函数修复）。
    """
    s = str(s).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def normalize_tushare_daily_df(raw: pd.DataFrame) -> pd.DataFrame:
    """tushare `daily(trade_date)` 长表 → 面板口径（**估值列留空，等 BaoStock 覆盖**）。

    ⚠️⚠️ 单位换算只在这里做一次：`vol` **手→股**（×100）、`amount` **千元→元**（×1000）。
    ⚠️ `trade_date` 8 位 → `YYYY-MM-DD`（面板唯一口径，见 `_norm_date_str`）。
    ⚠️ 缺失保持 NaN（**缺就是缺，不造 0**）。
    ⚠️ 退市股**没有**幸存者偏差：按 trade_date 拉全市场，退市前的交易日天然有行。
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    df = raw.copy()
    df["stock_code"] = df["ts_code"].astype(str).str.split(".").str[0]
    df["trade_date"] = df["trade_date"].map(_norm_date_str)
    # ⚠️ tushare 的成交量列叫 **vol**（手），不是 volume —— 漏了这步 rename
    #    会让 volume 列整列消失（keep 过滤后缺列）且 ×100 不生效。
    df = df.rename(columns={"trade_date": "trade_date", "pre_close": "preclose",
                            "pct_chg": "pct_chg", "vol": "volume"})
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = df["volume"] * 100.0            # 手 → 股
    if "amount" in df.columns:
        df["amount"] = df["amount"] * 1000.0           # 千元 → 元
    for col in ("turn", "pe_ttm", "pb_mrq", "ps_ttm", "trade_status", "is_st"):
        if col not in df.columns:
            df[col] = pd.NA                            # 等 BaoStock 行覆盖补齐
    keep = [c for c in PANEL_COLUMNS if c in df.columns]
    out = df[keep].sort_values("trade_date").reset_index(drop=True)
    return out


def _mark_date_done(d: str, rows: int, done_dates: dict) -> None:
    done_dates[str(d)] = {"rows": int(rows), "at": time.strftime("%Y-%m-%d %H:%M:%S")}


def run_backfill_tushare(*, trade_days: list[str],
                         fetch_one_day: Callable[[str], pd.DataFrame],
                         batch_size: int = 120, resume: bool = True,
                         progress_every: int = 20) -> dict:
    # ⚠️ R4（2026-10-18 审查）：离线模式下**禁止回填**。
    #    BaoStock/tushare 都是裸 TCP，`offline_gate` 只**计数**、拦不住；
    #    与其"静默照跑"，不如明确拒绝——离线模式的全部意义就是不出网。
    _assert_not_offline("回填")
    """按 **trade_date** 循环回填（tushare `daily` 1 次/日全市场）。

    Args:
        trade_days: **交易日清单**（'YYYYMMDD' 或 'YYYY-MM-DD'），由调用方给
          —— 生产用 `tushare_source.fetch_trade_cal`（1 次调用拿全区间，0.1s 实测）；
          本地 `data/trading_calendar` 缓存覆盖范围有限（超出 ⇒ 工作日近似），
          长区间不能用它，否则节假日全被误判成失败。

    ⚠️⚠️ 进度键是 `progress.json.done_dates`（**与按股的 done/failed 分开**）——
       一份进度文件不承载两类语义（同主键被两类产出争用 ⇒ 静默覆盖，项目踩过）。
    ⚠️ 单日失败不中断（失败立即落盘，重跑自动重试）。
    Returns: {"days_total","skipped_done","fetched","failed","rows","parts",...}
    """
    t0 = time.time()
    prog = _read_progress()
    done_dates: dict = dict(prog.get("done_dates") or {})
    failed_dates: dict = dict(prog.get("failed_dates") or {})

    days = [str(d).replace("-", "") for d in trade_days]

    stats = {"days_total": len(days), "skipped_done": 0, "fetched": 0,
             "failed": 0, "rows": 0, "parts": 0}
    n_parts = len([f for f in os.listdir(PARTS_DIR) if f.endswith(".parquet")]) \
        if os.path.isdir(PARTS_DIR) else 0
    buffer: list[pd.DataFrame] = []
    # ⚠️ 落盘成功后才记 done_dates（同一事故的另一面：先记完成 ⇒ 崩了数据丢失但永久跳过）
    pending_days: list[tuple[str, int]] = []

    def _flush(force: bool = False) -> None:
        nonlocal n_parts
        if not buffer:
            return
        if not (force or len(buffer) >= max(1, batch_size)):
            return
        save_part(pd.concat(buffer, ignore_index=True))
        n_parts += 1
        stats["parts"] = n_parts
        for d, n in pending_days:
            _mark_date_done(d, n, done_dates)
        buffer.clear()
        pending_days.clear()

    for i, d in enumerate(days, 1):
        if resume and str(d) in done_dates:
            stats["skipped_done"] += 1
            continue
        try:
            raw = fetch_one_day(d)
            df = normalize_tushare_daily_df(raw)
        except Exception as e:                              # noqa: BLE001
            failed_dates[str(d)] = f"{type(e).__name__}: {str(e)[:150]}"
            stats["failed"] += 1
            _logger.warning("[panel/tushare] %s 回填失败（跳过）: %s", d, failed_dates[str(d)])
            _write_progress_dates_full(done_dates, failed_dates)   # 失败也立即落盘（可查、可重试）
            continue
        stats["fetched"] += 1
        stats["rows"] += int(len(df))
        if df.empty:
            _mark_date_done(d, 0, done_dates)               # 无数据可丢 ⇒ 可直接记
        else:
            buffer.append(df)
            pending_days.append((d, len(df)))
            _flush()
        if progress_every and i % progress_every == 0:
            # ⚠️ 只用 `_write_progress_dates_full`（统一写出口，保留 done/failed 按股键）——
            #    旧 `_write_progress` 不带 done_dates，会把按日进度**整个覆盖丢掉**。
            _write_progress_dates_full(done_dates, failed_dates)
            _logger.info("[panel/tushare] %d/%d 日完成（%d 行，分片 %d）",
                         i, len(days), stats["fetched"], n_parts)

    _flush(force=True)
    stats["parts"] = n_parts
    stats["elapsed_s"] = round(time.time() - t0, 1)
    _write_progress_dates_full(done_dates, failed_dates)
    return stats


def _write_progress_full(done: dict, failed: dict) -> None:
    """兼容旧接口：只写按股进度（保留 done_dates/failed_dates）。

    ⚠️ `_write_progress_dates_full` 的前两个位置参数是**按日**键，必须用关键字
    传按股键（本包装原本按位置传 ⇒ 一旦被调用就 TypeError，因从未被调用而潜伏）。
    """
    prog = _read_progress()
    _write_progress_dates_full(done_dates=prog.get("done_dates") or {},
                               failed_dates=prog.get("failed_dates") or {},
                               done=done, failed=failed)


def _write_progress_dates_full(done_dates: dict, failed_dates: dict,
                               done: dict | None = None, failed: dict | None = None) -> None:
    """统一进度写出口：done/failed（按股）+ done_dates/failed_dates（按日）。"""
    os.makedirs(PANEL_DIR, exist_ok=True)
    prog = _read_progress()
    data = {"done": done if done is not None else prog.get("done") or {},
            "failed": failed if failed is not None else prog.get("failed") or {},
            "done_dates": done_dates,
            "failed_dates": failed_dates,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = PROGRESS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, PROGRESS_PATH)
