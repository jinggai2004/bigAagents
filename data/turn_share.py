# -*- coding: utf-8 -*-
"""换手率 / 流通股本数据层（`data/panel/turn_share/{code}.parquet`）。

## 这一层的两个来源（分工是**实测结论**，不是设计偏好）

1. **历史**（2016-01-04 起）：`scripts/backfill_turn_share.py` 打新浪日线回填。
   只在"重建历史 / 补大洞"时手工跑 —— 2026-11-01 实测 4 并发补 120 只花 250 秒，
   全量 5,830 只约 **4 小时**，当日常任务不可行。更要命的是 akshare 的
   `stock_zh_a_daily` 每次调用都新建一个 V8 isolate，多线程构造会让进程
   `[FATAL:partition_address_space.cc] abort()`（详见该脚本顶部说明）。

2. **日更**（2026-11-01 起）：本模块 `sync_from_snapshot()` 从 `market_snapshot`
   派生，秒级、零网络。`snapshot_flow` 每天 15:40 已经把全市场当日
   `turnover` / `float_cap` 落库，没必要为了同一份数据再打一遍接口。

## 口径（对账过，不是照文档抄的）

- `turnover_pct`：**百分数**（`1.027` = 1.027%）。与 `market_snapshot.turnover`
  同值：000001 在 2026-10-29 两侧都是 1.02728 / 1.027278。
- `float_share`：**股**，`= float_cap / price`。`market_snapshot.float_cap` 单位是
  **元**（不是万元）：000001 2026-10-29 `223,941,604,796.14 / 11.54 = 1.9406e10`，
  与历史层的 `1.940568e+10` 一致（差 0.00003%，来自价格只有两位小数）。

## 为什么要保留 `float_share` 这一列

换手率已经直接来自快照，但规模类因子（`amihud20` 的分母、流通市值）要的是
股本本身。用 `float_cap / price` 反推会在价格四舍五入上有 1e-7 级误差 ——
对因子排序无影响（实测同一只票的换手率两种算法差在第 7 位之后）。
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from config.paths import data_root

_logger = logging.getLogger("turn_share")

_COLUMNS = ["date", "turnover_pct", "float_share"]


def _as_float(v):
    """宽松转 float；不可转返回 None（**缺就是缺，绝不返 0**）。

    ⚠️ 这里必须留痕（debug 级，默认不输出）：字段类型变了（比如变成字符串
    `'N/A'`）会让"全市场换手率都缺失"，而日志里一个字都没有 —— 本项目在
    `pe_ratio` 上踩过同款坑：`is None` 判据穿透、兜底从未执行、覆盖率恒 0.82
    却没人发现。三处转换合成这一个函数，静默 except 基线才不会涨。
    """
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        _logger.debug("[换手率层] 数值转换失败（%r）: %s", v, e)
        return None


def panel_dir() -> str:
    """换手率层目录（**唯一入口**，别在别处拼路径）。"""
    return os.path.join(data_root(), "data", "panel", "turn_share")


def available_codes() -> list[str]:
    """已有 parquet 的代码（升序）。"""
    d = panel_dir()
    if not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".parquet"))


def last_date(code: str) -> str | None:
    """某只票的末行日期；文件不存在/读不出返回 None。"""
    path = os.path.join(panel_dir(), f"{code}.parquet")
    if not os.path.exists(path):
        return None
    try:
        d = pd.read_parquet(path, columns=["date"])
    except (OSError, ValueError) as e:
        _logger.warning("[换手率层] %s 旧文件读不出: %s", code, e)
        return None
    return str(d["date"].max())[:10] if len(d) else None


def sync_from_snapshot(days: int = 10, codes: list[str] | None = None) -> dict:
    """把 `market_snapshot` 中**比各票末行更新**的日期追加进换手率层。

    幂等：重复调用不会产生重复行（按 `date` 去重，保留最后写入的那条）。

    Args:
        days: 回溯几个快照日期（默认 10，够补一周+节假日）。
        codes: 只处理这些代码；None = 全部已有 parquet 的代码。

    Returns:
        dict: `{"dates": [...], "codes": n, "rows": n, "missing": n, "errors": [...]}`
        - `codes` / `rows`：真正被更新的票数与追加的行数（幂等重跑时都是 0）
        - `missing`：快照里有、但本地**没有历史 parquet** 的代码数（新股：不建文件，
          建了只有几天数据，反而会让因子在历史上大面积 NaN）
        - `errors`：单只失败的原因（不中断整体）

    Raises:
        不抛异常：单只读写失败只记进 `errors`，由调用方决定是否告警。
    """
    from db import store

    out: dict = {"dates": [], "codes": 0, "rows": 0, "missing": 0, "errors": []}
    dates = store.market_snapshot_dates(limit=max(1, int(days)))
    if not dates:
        return out

    pdir = panel_dir()
    if not os.path.isdir(pdir):
        _logger.warning("[换手率层] 目录不存在：%s（先用回填脚本建历史）", pdir)
        return out

    targets = set(codes) if codes else set(available_codes())
    if not targets:
        return out

    # ── 1) 收集待追加行：code -> [(date, turnover_pct, float_share)] ──
    pending: dict[str, list[tuple]] = {}
    missing_codes: set[str] = set()
    for d in dates:
        date_actual, rows = store.load_market_snapshot(d)
        if not date_actual or not rows:
            continue
        out["dates"].append(str(date_actual))
        for r in rows:
            code = str(r.get("stock_code") or "").strip()
            if not code:
                continue
            if code not in targets:
                if code not in missing_codes:
                    missing_codes.add(code)
                continue
            turn = r.get("turnover")
            price = r.get("price")
            fcap = r.get("float_cap")
            # 缺就是缺：任一为空都写 NaN，**绝不补 0**（0 会污染换手率均值/分位）
            tp = _as_float(turn)
            p = _as_float(price)
            c = _as_float(fcap)
            fs = (c / p) if (p is not None and c is not None and p > 0) else None
            if tp is None and fs is None:
                continue
            pending.setdefault(code, []).append((str(date_actual), tp, fs))

    out["missing"] = len(missing_codes)

    # ── 2) 逐只合并写回（原子替换：读者不会读到半个文件）──
    for code, add in pending.items():
        path = os.path.join(pdir, f"{code}.parquet")
        try:
            if os.path.exists(path):
                df = pd.read_parquet(path)
            else:
                df = pd.DataFrame(columns=_COLUMNS)
            add_df = pd.DataFrame(add, columns=_COLUMNS)
            # ⚠️ 必须先转成 float：含 None 时 DataFrame 是 object dtype，
            #    直接 concat 会触发 pandas 的 "all-NA entries" FutureWarning
            #    （未来版本会改 dtype 推断规则 ⇒ 今天是警告、明天是行为变化）。
            add_df["turnover_pct"] = pd.to_numeric(add_df["turnover_pct"], errors="coerce")
            add_df["float_share"] = pd.to_numeric(add_df["float_share"], errors="coerce")
            # 只追加严格晚于现有末行的日期：历史层（新浪全量）比快照更准，
            # 不让快照覆盖同一天的历史值。
            if len(df) and "date" in df.columns:
                last = str(df["date"].max())[:10]
                add_df = add_df[add_df["date"] > last]
            if add_df.empty:
                continue
            merged = pd.concat([df, add_df], ignore_index=True)
            merged["turnover_pct"] = pd.to_numeric(
                merged["turnover_pct"], errors="coerce").astype("float32")
            merged["float_share"] = pd.to_numeric(
                merged["float_share"], errors="coerce").astype("float64")
            merged = (merged.drop_duplicates(subset=["date"], keep="last")
                            .sort_values("date").reset_index(drop=True))
            tmp = path + ".tmp"
            merged.to_parquet(tmp, index=False)
            os.replace(tmp, path)
            out["codes"] += 1
            out["rows"] += int(len(add_df))
        except (OSError, ValueError, KeyError) as e:
            # 必须同时留痕（集合里的 errors 只在调用方汇总，扫描器/日志看不到单只现场）
            _logger.debug("[换手率层] %s 写回失败: %s", code, e)
            out["errors"].append(f"{code}: {type(e).__name__}: {str(e)[:120]}")

    if out["errors"]:
        _logger.warning("[换手率层] %d 只写入失败（前 3 条）：%s",
                        len(out["errors"]), out["errors"][:3])
    return out
