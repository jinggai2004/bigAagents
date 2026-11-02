# -*- coding: utf-8 -*-
"""回填「换手率 + 流通股本」—— 新浪日线（`ak.stock_zh_a_daily`）。

## 为什么要单独一层
面板的两代分片里，`turn` 只在 0017~0027（按股票分块）那 1,997 只有值，
且**科创板 0 只、北交所 0 只**；`outstanding_share`（流通股本）**全都没有**
（`bars.get_bars` 里那列一直是 `np.nan`）。没有这两列，换手率/异常换手/
规模因子都做不了 —— 这是 `output/文献因子落地与数据缺口_2026-10-29.md` 里
"换手率可做、成本 21 分钟"那一节的落地脚本。

## 口径（实测确认，不是照文档抄的）
- 新浪 `turnover` = `volume / outstanding_share`，是**小数**（茅台 0.003104）。
- 面板 `turn`（BaoStock）是**百分数**（0.31）。
⇒ 落盘统一成 **百分数**（×100），与 `bars.get_bars` 的 `turnover_pct` 同口径。
   ⚠️ 不换算就会差 100 倍，而且**不会报错** —— 本项目栽过三次同类坑。

- 覆盖：2016-01-04 ~ 最新交易日，单只 2,626 行、`turnover`/`outstanding_share`
  缺失 0（实测 600519 / 000001 / 600000）。北交所 `bj920187` 也支持。

## 落盘
`data/panel/turn_share/{code}.parquet`（独立目录，**不动** by_code / parts / overlay）：
    date(str) / turnover_pct(float32) / float_share(float64, 股)
`bars.get_bars` 在读时叠加（唯一的消费入口，见那边注释）。

## 用法
    python scripts/backfill_turn_share.py                # 全量增量（已新鲜的跳过）
    python scripts/backfill_turn_share.py --workers 6
    python scripts/backfill_turn_share.py --limit 50     # 先试跑
    python scripts/backfill_turn_share.py --codes 600000,000001 --force
    python scripts/backfill_turn_share.py --stale-days 0  # **日更**（收盘后补齐当天）

⚠️ `--stale-days` 默认 1（手工补历史的语义："差一天也算新鲜"，周末/节假日不反复
   打接口）。但收盘后日更时末行必然是昨天 ⇒ lag=1 会被判 skip，**当天永远补不上**。
   调度任务 `turn_share_sync` 因此显式传 `--stale-days 0`。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd                                             # noqa: E402

from config.paths import data_root                              # noqa: E402
from data import bars as bars_mod                               # noqa: E402

# ⚠️⚠️ 必须在 `import akshare` **之前**打这个补丁（2026-11-01 实测崩溃后加）。
#
# 现象：6 并发（后来降到 4 并发也一样）全量跑到第 2 秒，**整个进程** FATAL 退出：
#   `[FATAL:partition_address_space.cc(243)] Check failed: !IsConfigurablePoolInitialized()`
#   栈顶是 `py_mini_racer/mini_racer.dll` → V8 的 PartitionAlloc 全局池被并发初始化。
#   这不是 Python 异常：`except Exception` 拦不住，在调度器里跑会连带杀掉当天
#   所有剩余任务（任务台账只记到 started，看起来像"卡死"）。
#
# 根因（读 akshare 源码确认，不是猜的）：
#   `akshare/stock/stock_zh_a_sina.py::stock_zh_a_daily` 里
#       js_code = py_mini_racer.MiniRacer()   # ← **每次调用**都新建一个 V8 isolate
#       js_code.eval(hk_js_decode)
#       dict_list = js_code.call("d", ...)    # 解密新浪返回的 K 线串
#   多线程同时首次构造 isolate ⇒ 全局池重复初始化 ⇒ abort()。
#   所以"顶层 import akshare 一次"和"预热一次"都没用 —— 崩溃点在建对象，不在导入。
#
# 修法：
#   · **线程局部**：每个工作线程复用自己的 isolate（V8 isolate 有线程亲和性，
#     共享一个实例即使加锁也不安全）；
#   · **创建锁**：只把 `MiniRacer()` 的构造串行化（崩溃点是并发创建，不是并发使用）；
#   · 解密只占毫秒级，真正的耗时在网络请求 ⇒ 并发收益完全保留。
#   akshare 用的是 `import py_mini_racer` + `py_mini_racer.MiniRacer()`（模块属性
#   访问），因此替换模块属性即可生效。
import py_mini_racer                                              # noqa: E402
from py_mini_racer import MiniRacer as _RealMiniRacer             # noqa: E402

_V8_CREATE_LOCK = threading.Lock()
_V8_TLS = threading.local()


class _ThreadLocalMiniRacer:
    """`py_mini_racer.MiniRacer` 的线程局部单例包装（构造加锁）。见上方说明。"""

    def __init__(self):
        racer = getattr(_V8_TLS, "racer", None)
        if racer is None:
            with _V8_CREATE_LOCK:            # 串行化 V8 创建 —— 崩溃就发生在这里
                racer = _RealMiniRacer()
            _V8_TLS.racer = racer
        self._racer = racer

    def eval(self, *args, **kwargs):
        return self._racer.eval(*args, **kwargs)

    def call(self, *args, **kwargs):
        return self._racer.call(*args, **kwargs)

    def __getattr__(self, item):             # 其余属性透传（size/statistics 等）
        return getattr(self._racer, item)


py_mini_racer.MiniRacer = _ThreadLocalMiniRacer

# akshare 的 V8 依赖已被上面的补丁接管；它导入失败也不致命（缺包时任务报 fail
# 而不是崩）：`ak` 为 None 时 `_fetch_one` 早退。
try:
    import akshare as ak
except Exception as _e:                                        # noqa: BLE001
    ak = None
    _logger_import_error = _e

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger("backfill_turn_share")

if ak is None:                                                 # 见上方导入块的说明
    _logger.warning("[换手回填] akshare 导入失败（%s）⇒ 本次无法拉取任何数据",
                    _logger_import_error)

OUT_DIR = os.path.join(data_root(), "data", "panel", "turn_share")
MANIFEST = os.path.join(data_root(), "data", "panel", "turn_share_manifest.json")

# 起始日：与面板一致（2016-01-04 是全市场面板的第一天）
START = "20160101"

# 末行距今超过这么多交易日才重拉（周末/节假日不反复打接口）
# ⚠️ 2026-11-01：默认值 1 只适合**手工补历史**，日更场景会**永久少一天** ——
#    收盘后跑时末行必然是昨天（lag=1）⇒ `lag <= 1` 直接 skip，当天数据永远补不上。
#    日更调用方必须传 `--stale-days 0`（见 core.scheduler.run_turn_share_job）。
STALE_TRADING_DAYS = 1

_io_lock = threading.Lock()
_stat = {"ok": 0, "skip": 0, "fail": 0, "rows": 0}
_fails: list[tuple[str, str]] = []


def _sina_symbol(code: str) -> str | None:
    """6 位码 → 新浪写法（`sh600000` / `sz000001` / `bj920187`）。"""
    c = str(code).strip()
    if len(c) != 6 or not c.isdigit():
        return None
    if c[0] == "6":
        return "sh" + c
    if c[0] in ("0", "3"):
        return "sz" + c
    if c[:2] in ("43", "83", "87", "88", "92"):
        return "bj" + c
    return None


def _existing_last(code: str) -> str | None:
    path = os.path.join(OUT_DIR, f"{code}.parquet")
    if not os.path.exists(path):
        return None
    try:
        d = pd.read_parquet(path, columns=["date"])
    except (OSError, ValueError) as e:
        _logger.warning("[换手回填] %s 旧文件读不出（重拉）: %s", code, e)
        return None
    return str(d["date"].max())[:10] if len(d) else None


def _fetch_one(code: str, *, force: bool, end: str,
               stale_days: int = STALE_TRADING_DAYS) -> tuple:
    """返回 (code, status, rows)；status ∈ ok / skip / fail。"""
    path = os.path.join(OUT_DIR, f"{code}.parquet")
    if not force:
        last = _existing_last(code)
        if last and last >= end:
            return (code, "skip", 0)
        if last:
            lag = bars_mod.trading_day_lag(last, end)
            if lag is not None and lag <= stale_days:
                return (code, "skip", 0)
    sym = _sina_symbol(code)
    if sym is None:
        return (code, "fail", "非沪深/北交所代码")
    if ak is None:
        # 不在线程里重试导入：V8 初始化竞态会让进程 abort（见模块顶部说明）
        return (code, "fail", "akshare 不可用（见启动日志）")
    for attempt in (1, 2):
        try:
            raw = ak.stock_zh_a_daily(symbol=sym, start_date=START,
                                     end_date=end.replace("-", ""), adjust="")
            if raw is None or len(raw) == 0:
                return (code, "fail", "接口返回空")
            out = pd.DataFrame({
                "date": pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d"),
                # ⚠️ 小数 → 百分数（与 BaoStock `turn` 同口径）
                "turnover_pct": pd.to_numeric(raw["turnover"], errors="coerce") * 100.0,
                "float_share": pd.to_numeric(raw["outstanding_share"], errors="coerce"),
            })
            out = out.dropna(subset=["turnover_pct"]).astype(
                {"turnover_pct": "float32", "float_share": "float64"})
            if out.empty:
                return (code, "fail", "有效行为 0")
            with _io_lock:
                os.makedirs(OUT_DIR, exist_ok=True)
                tmp = path + ".tmp"
                out.to_parquet(tmp, index=False)
                os.replace(tmp, path)                 # 原子：读者不会读到半个文件
            return (code, "ok", len(out))
        except Exception as e:                        # noqa: BLE001
            _logger.debug("[换手回填] %s 第 %d 次失败（%s: %s）",
                          code, attempt, type(e).__name__, str(e)[:80])
            if attempt == 2:
                return (code, "fail", f"{type(e).__name__}: {str(e)[:120]}")
            time.sleep(1.5)                           # 新浪偶发断连，退一次再试
    return (code, "fail", "未知")


def _write_manifest(started: str, elapsed: float, end: str) -> None:
    try:
        os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
        payload = {
            "source": "akshare.stock_zh_a_daily (sina)",
            "unit": "turnover_pct = 百分数（已 ×100，与面板 turn 同口径）",
            "start": START, "end": end,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(elapsed, 1),
            "codes": len([f for f in os.listdir(OUT_DIR) if f.endswith(".parquet")])
            if os.path.isdir(OUT_DIR) else 0,
            "last_run": dict(_stat),
        }
        tmp = MANIFEST + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, MANIFEST)
    except (OSError, ValueError) as e:
        _logger.warning("[换手回填] manifest 写入失败: %s", e)


def main(argv: list | None = None) -> int:
    """入口。`argv=None` 时读 `sys.argv`（命令行）；调度器传显式列表。

    ⚠️ 必须支持显式 argv：本函数会被 `core.scheduler.run_turn_share_job` 在
    **调度器进程内**直接调用，而此时 `sys.argv` 是 exe/uvicorn 的参数
    （如 `--host 127.0.0.1`）⇒ argparse 会把它们当未知参数 `exit(2)`。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="并发数。默认 4：6 并发曾在 V8 初始化竞态下整进程 FATAL，"
                         "降到 4 与 BaoStock 回填的经验值一致（吞吐差别不大）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--codes", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stale-days", type=int, default=STALE_TRADING_DAYS,
                    help="末行距参考日超过这么多交易日才重拉；日更传 0（默认 1 给手工补历史）")
    args = ap.parse_args(argv)

    # ⚠️ 计数器是**模块级全局**：调度器同进程反复调用本函数时必须清零，
    #    否则第二轮起 ok/skip/fail 会在上一轮基础上累加，manifest 报数全错。
    _stat.update(ok=0, skip=0, fail=0, rows=0)
    _fails.clear()

    codes = [c.strip() for c in args.codes.split(",") if c.strip()] \
        or sorted(bars_mod.available_codes())
    if args.limit:
        codes = codes[:args.limit]

    # 参考日 = 今天（新浪接口自己会截到最新交易日）。
    # ⚠️ 不用"最近应收盘交易日"那套：这里只是判"要不要重拉"，
    #    判错的代价是多重拉一次，而判据写复杂了反而会漏。
    end = date.today().isoformat()
    _logger.info("[换手回填] 目标 %d 只 / 参考日 %s / %d 并发 / 起始 %s",
                 len(codes), end, args.workers, START)

    t0 = time.time()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(_fetch_one, c, force=args.force, end=end,
                          stale_days=args.stale_days): c for c in codes}
        for fut in as_completed(futs):
            code, status, info = fut.result()
            done += 1
            if status == "ok":
                _stat["ok"] += 1
                _stat["rows"] += int(info)
            elif status == "skip":
                _stat["skip"] += 1
            else:
                _stat["fail"] += 1
                _fails.append((code, str(info)[:100]))
                _logger.warning("[换手回填] %s 失败: %s", code, info)
            if done % 100 == 0 or done == len(codes):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(codes) - done) / rate / 60 if rate else 0
                print(f"  {done}/{len(codes)}  ok={_stat['ok']} skip={_stat['skip']} "
                      f"fail={_stat['fail']}  {rate:.2f} 只/s  剩余 {eta:.1f} 分钟",
                      flush=True)

    elapsed = time.time() - t0
    _write_manifest(started, elapsed, end)
    print(f"\n完成：ok={_stat['ok']} skip={_stat['skip']} fail={_stat['fail']} "
          f"行数={_stat['rows']:,} 耗时 {elapsed / 60:.1f} 分钟")
    if _fails:
        print("失败样本（最多 10 条）：")
        for c, r in _fails[:10]:
            print(f"  {c}: {r}")
    return 0 if _stat["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
