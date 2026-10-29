# -*- coding: utf-8 -*-
"""研究（`/api/research/*`）—— P3。

两个端点都**只读已有产物**，不触发计算：
- `/api/research/backtest` ← `data/backtest_records/backtest_*.json`（`core.backtest` 落盘）
- `/api/research/factors`  ← `output/ui_runs/factor_analysis.json`（因子页跑完的结果）

## 为什么不在端点里跑回测/因子
因子分析实测是分钟级（20 只 × 74 期 × 5 组），回测要读全量日线面板。
只读服务层的职责是"把算好的东西**如实**呈现"——重算属于任务队列（`core.ui_tasks`），
端点里现算只会让浏览器转圈到超时，还会和调度抢 CPU。

## 缺产物 ≠ 空结果
没有 `factor_analysis.json` ⇒ `missing`（"还没跑"）；文件在但解析失败 ⇒ `empty`（"坏了"）。
两者前端文案与排查方向不同，不许合并。
"""
from __future__ import annotations

import glob
import json
import logging
import os
from datetime import datetime

from web.server import cache, envelope, settings

_logger = logging.getLogger(__name__)

TTL = 300
_MAX_TRADES = 200


def _records_dir() -> str:
    return os.path.join(settings.data_root(), "data", "backtest_records")


def _factor_path() -> str:
    return os.path.join(settings.data_root(), "output", "ui_runs", "factor_analysis.json")


def _list_records() -> list[dict]:
    out = []
    for p in glob.glob(os.path.join(_records_dir(), "backtest_*.json")):
        try:
            st = os.stat(p)
        except OSError as e:
            _logger.debug("[research] stat 失败 %s: %s", p, e)
            continue
        out.append({"file": os.path.basename(p), "size": st.st_size, "mtime": st.st_mtime})
    out.sort(key=lambda r: r["file"], reverse=True)     # 文件名带时间戳 ⇒ 倒序即新旧
    return out


def backtest(file: str | None = None) -> dict:
    records = _list_records()
    if not records:
        return envelope.fail("missing", f"{_records_dir()} 下无 backtest_*.json")

    name = file or records[0]["file"]
    if os.path.basename(name) != name:                  # 防目录穿越：只接文件名
        return envelope.fail("invalid", "file 只能是文件名")
    path = os.path.join(_records_dir(), name)
    if not os.path.exists(path):
        return envelope.fail("missing", f"无该回测记录: {name}")

    try:
        st = os.stat(path)
        raw = cache.cached("research.backtest", (name, int(st.st_mtime), st.st_size),
                           TTL, lambda: _read_json(path))
    except OSError as e:
        return envelope.fail("empty", f"{type(e).__name__}: {e}")

    if not isinstance(raw, dict):
        return envelope.fail("empty", "回测记录顶层不是对象")
    if "__error__" in raw:                        # `_read_json` 的失败信封
        return envelope.fail("empty", raw["__error__"])

    nav = raw.get("nav_curve") or []
    trades = raw.get("trade_history") or []
    data = {
        "file": name,
        "records": records,
        "metrics": {k: raw.get(k) for k in (
            "total_return", "annual_return", "sharpe_ratio", "calmar_ratio",
            "max_drawdown", "win_rate", "profit_loss_ratio", "total_trades",
            "turnover_rate", "information_ratio", "monthly_win_rate",
            "rejected_orders", "avg_position_pct")},
        "nav_curve": nav,
        "benchmark_nav": raw.get("benchmark_nav") or [],
        "yearly_returns": raw.get("yearly_returns") or {},
        "trades": trades[-_MAX_TRADES:],
        "trades_total": len(trades),
        "start": nav[0]["date"] if nav else None,
        "end": nav[-1]["date"] if nav else None,
    }
    return envelope.ok(data, as_of=data["end"])


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        _logger.warning("[research] 解析失败 %s: %s: %s", path, type(e).__name__, e)
        return {"__error__": f"{type(e).__name__}: {e}"}


def factors() -> dict:
    path = _factor_path()
    if not os.path.exists(path):
        return envelope.fail("missing", "output/ui_runs/factor_analysis.json 不存在（因子页尚未跑过）")

    try:
        st = os.stat(path)
        raw = cache.cached("research.factors", (int(st.st_mtime), st.st_size),
                           TTL, lambda: _read_json(path))
    except OSError as e:
        return envelope.fail("empty", f"{type(e).__name__}: {e}")

    if not isinstance(raw, dict) or "__error__" in raw:
        return envelope.fail("empty", (raw or {}).get("__error__", "顶层不是对象"))

    payload = raw.get("payload") or {}
    data = {
        "generated_at": raw.get("generated_at"),
        "summary": payload.get("summary") or [],
        "layered": payload.get("layered") or {},
        "correlation": payload.get("correlation") or {},
        "periods": payload.get("periods"),
        "names": payload.get("names"),
        "panel_rows": payload.get("panel_rows"),
        "horizon": payload.get("horizon"),
        "step": payload.get("step"),
        "groups": payload.get("groups"),
        "unavailable": payload.get("unavailable") or {},
        "snapshot_status": payload.get("snapshot_status") or {},
        "note": payload.get("note") or "",
    }
    return envelope.ok(data, as_of=raw.get("generated_at"))


# ================================================================
# v4（2026-10-29）：验证与归因（回测验证设计 A/3.1/B 的呈现层）
# ================================================================
#
# 三个新端点延续同一纪律——**只读已有产物**：
# - `/api/research/backtest/validate` ← `data/backtest_records/sweep_insample_*.json`
#   （`scripts/backtest_rule_based.py --sweep-insample` 落盘；DSR/PBO 已算好）
# - `/api/research/walkforward`       ← `output/walk_forward_*.json`
#   （`scripts/run_walk_forward.py` 落盘）
# - `/api/research/attributions`      ← **唯一的例外**：交易盈亏分解是轻量在线
#   计算（~35 单 × 本地面板收盘，`data.bars.get_bars` 不联网，实测秒级），
#   且它的输入是**会变的库**（新成交随时落库）——没有"算好的产物"可读。
#   用 cache 300s 挡重复计算。风格归因（要 load_matrices，分钟级）**不在端点里跑**，
#   仍走 CLI（`scripts/run_walk_forward.py` 旁路）。

_TTL_ATTR = 300


def _sweep_files() -> list[str]:
    return glob.glob(os.path.join(_records_dir(), "sweep_insample_*.json"))


def _walkforward_files() -> list[str]:
    return glob.glob(os.path.join(settings.data_root(), "output", "walk_forward_*.json"))


def backtest_validate() -> dict:
    """参数扫描的多重检验报告（DSR / PBO，模块 A）。只读最新一份扫描产物。"""
    files = _sweep_files()
    if not files:
        return envelope.fail(
            "missing",
            "尚无 sweep_insample_*.json（先跑 scripts/backtest_rule_based.py --sweep-insample）")
    latest = max(files, key=os.path.getmtime)
    try:
        st = os.stat(latest)
        raw = cache.cached("research.validate", (latest, int(st.st_mtime), st.st_size),
                           TTL, lambda: _read_json(latest))
    except OSError as e:
        return envelope.fail("empty", f"{type(e).__name__}: {e}")
    if not isinstance(raw, dict) or "__error__" in raw:
        return envelope.fail("empty", (raw or {}).get("__error__", "顶层不是对象"))

    mt = raw.get("multiple_testing")
    if not mt:
        return envelope.fail("empty", "该扫描产物没有 multiple_testing（旧版本生成，请重跑扫描）")
    data = {
        "file": os.path.basename(latest),
        "built_at": raw.get("built_at"),
        "start": raw.get("start"), "end": raw.get("end"),
        "grid": raw.get("grid") or {},
        "n_results": len(raw.get("results") or []),
        "multiple_testing": mt,
    }
    return envelope.ok(data, as_of=mt.get("dsr", {}).get("ok") and raw.get("built_at") or None)


def walkforward() -> dict:
    """walk-forward 权重重估结果（B 模块）。只读最新一份产物。"""
    files = _walkforward_files()
    if not files:
        return envelope.fail(
            "missing",
            "尚无 walk_forward_*.json（先跑 scripts/run_walk_forward.py，手动批任务不进调度）")
    latest = max(files, key=os.path.getmtime)
    try:
        st = os.stat(latest)
        raw = cache.cached("research.walkforward", (latest, int(st.st_mtime), st.st_size),
                           TTL, lambda: _read_json(latest))
    except OSError as e:
        return envelope.fail("empty", f"{type(e).__name__}: {e}")
    if not isinstance(raw, dict) or "__error__" in raw:
        return envelope.fail("empty", (raw or {}).get("__error__", "顶层不是对象"))

    horizons = {}
    for h, res in (raw.get("horizons") or {}).items():
        summary = (res or {}).get("summary") or {}
        # 逐折只留页面要展示的字段（权重 + OOS 对比），facts 级细节留在文件里
        folds = []
        for f in (res or {}).get("folds") or []:
            folds.append({
                "test_year": f.get("test_year"),
                "train_years": f.get("train_years") or [],
                "weights_refit": f.get("weights_refit") or {},
                "oos_refit": f.get("oos_refit") or {},
                "oos_baseline": f.get("oos_baseline") or {},
            })
        horizons[str(h)] = {"summary": summary, "folds": folds}
    data = {"file": os.path.basename(latest), "built_at": raw.get("built_at"),
            "train_years": raw.get("train_years"), "horizons": horizons}
    return envelope.ok(data, as_of=raw.get("built_at"))


def _attr_identity(name: str | None) -> tuple:
    """盈亏分解的缓存键数据身份：paper_orders 一变（新成交/回填），键就变。

    ⚠️ cache 模块纪律："键必须带数据身份"。第一版用静态键 ("", "v1")，
    实测跨库串味：测试进程里真库先算出 7 回合，tmp 库的用例读到脏缓存
    （assert 7 == 0 才暴露）。
    """
    try:
        from db.models import get_connection
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS c, COALESCE(MAX(updated_at), '') AS m FROM paper_orders"
            ).fetchone()
            return (name or "", "v1", int(row["c"]), str(row["m"]))
        finally:
            conn.close()
    except Exception:                                   # noqa: BLE001
        return (name or "", "v1", "nofp")


def attributions(name: str | None = None) -> dict:
    """交易级盈亏分解（3.1）。**轻量在线计算**（见上方纪律说明），cache 300s。"""
    from core import pnl_attribution

    def _run():
        try:
            return pnl_attribution.decompose_round_trips(name=name)
        except Exception as e:                          # noqa: BLE001
            _logger.warning("[research] 盈亏分解失败: %s: %s",
                            type(e).__name__, e)
            return {"__error__": f"{type(e).__name__}: {e}"}

    res = cache.cached("research.attributions", _attr_identity(name), _TTL_ATTR, _run)
    if not isinstance(res, dict):
        return envelope.fail("empty", "分解结果不是对象")
    if "__error__" in res:
        return envelope.fail("empty", res["__error__"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {**res, "trips_shown": len(res.get("trips") or [])}
    return envelope.ok(data, as_of=now)
