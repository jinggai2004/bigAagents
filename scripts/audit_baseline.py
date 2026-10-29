#!/usr/bin/env python
"""基线体检：把「待办总览 §0 / 闭环样本 / 面板覆盖 / 版本库卫生」一次打出来。

为什么单独一个脚本
------------------
本项目所有"改进了吗"的判断都依赖一组数字（测试数、闭环样本、面板覆盖、未提交文件数）。
这些数字此前散落在各份报告里、靠手工敲命令复测，于是**报告里的数字会过期而没人发现**
（实例：`output/待办总览_2026-10-13.md` 说 `TUSHARE_TOKEN` 为空，实际早已配置；
`data/panel/progress.json` 记 done 201/failed 1436，实际 parquet 已覆盖 5830 只）。

用法
----
    python scripts/audit_baseline.py          # 人读
    python scripts/audit_baseline.py --json   # 接自动化，退出码始终 0（这是体检不是门禁）

只读、不联网、不写库；缺文件/表一律显示 `-` 而不是报错退出。
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "portfolio.db"
PANEL_PARTS = ROOT / "data" / "panel" / "parts"

SKIP_DIRS = {"temp", "__pycache__", ".git", ".pytest_cache", "output", "logs",
             ".workbuddy", "data"}


# ──────────────────────────── 1. 代码 / 测试规模 ────────────────────────────
def code_stats() -> dict:
    py_files, src_lines, test_lines, test_funcs = [], 0, 0, 0
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT)
        if set(rel.parts) & SKIP_DIRS:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        py_files.append(rel.as_posix())
        is_test = rel.parts[0] == "tests"
        if is_test:
            test_lines += len(text.splitlines())
        else:
            src_lines += len(text.splitlines())
        if is_test and rel.name.startswith("test_"):
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            test_funcs += sum(
                1 for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name.startswith("test")
            )
    tfiles = [f for f in py_files if f.startswith("tests/")]
    return {"py_files": len(py_files), "src_lines": src_lines,
            "test_lines": test_lines, "test_files": len(tfiles),
            "test_funcs": test_funcs}


# ──────────────────────────── 2. 本地面板 ────────────────────────────
def panel_stats() -> dict:
    if not PANEL_PARTS.exists():
        return {"rows": None}
    try:
        import pandas as pd
    except Exception:
        return {"rows": None, "note": "pandas 不可用"}
    parts = sorted(PANEL_PARTS.glob("*.parquet"))
    if not parts:
        return {"rows": 0}
    total, codes, dmin, dmax = 0, set(), None, None
    cols: list = []
    for p in parts:
        try:
            d = pd.read_parquet(p, columns=["stock_code", "trade_date"])
        except Exception:
            continue
        if not cols:
            try:
                cols = list(pd.read_parquet(p).columns)
            except Exception:
                cols = []
        total += len(d)
        codes |= set(d["stock_code"].unique())
        lo, hi = d["trade_date"].min(), d["trade_date"].max()
        dmin = lo if dmin is None else min(dmin, lo)
        dmax = hi if dmax is None else max(dmax, hi)
    size_mb = round(sum(p.stat().st_size for p in parts) / 1024 / 1024, 1)
    backup = PANEL_PARTS.parent / "parts_backup_datefmt"
    backup_mb = (round(sum(f.stat().st_size for f in backup.rglob("*.parquet")) / 1024 / 1024, 1)
                 if backup.exists() else 0.0)
    return {"parts": len(parts), "rows": total, "codes": len(codes),
            "date_from": str(dmin), "date_to": str(dmax),
            "size_mb": size_mb, "backup_mb": backup_mb, "cols": cols}


def panel_progress() -> dict:
    f = ROOT / "data" / "panel" / "progress.json"
    if not f.exists():
        return {}
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {"done": len(d.get("done") or {}), "failed": len(d.get("failed") or {}),
            "updated": d.get("updated")}


# ──────────────────────────── 3. 闭环样本 ────────────────────────────
def db_stats() -> dict:
    if not DB.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    except Exception:
        return {}
    cur = con.cursor()
    out: dict = {}

    def one(sql: str, default=None):
        try:
            return cur.execute(sql).fetchone()
        except Exception:
            return default

    out["tables"] = one("select count(*) from sqlite_master where type='table'")[0]
    for t in ("analyses", "screen_picks", "llm_calls", "alerts", "news_items",
              "market_snapshot", "fund_flow_snapshot"):
        row = one(f"select count(*) from {t}")
        out[f"{t}_rows"] = row[0] if row else None
    out["analyses_status"] = one(
        "select group_concat(coalesce(backfill_status,'NULL')||'='||c, ' ') from "
        "(select backfill_status, count(*) c from analyses group by 1)")
    out["analyses_ret"] = one(
        "select count(ret_5d), count(ret_20d) from analyses")
    out["screen_status"] = one(
        "select group_concat(coalesce(backfill_status,'NULL')||'='||c, ' ') from "
        "(select backfill_status, count(*) c from screen_picks group by 1)")
    out["screen_ret"] = one(
        "select count(ret_5d), count(ret_20d), count(distinct pick_date) from screen_picks")
    out["llm"] = one(
        "select count(*), sum(total_tokens), round(avg(latency_ms)), "
        "sum(case when ok=0 then 1 else 0 end) from llm_calls")
    out["daily_features"] = one(
        "select count(*), count(distinct symbol) from daily_features")
    out["coverage_rows"] = one("select count(*) from data_coverage")
    con.close()
    return out


# ──────────────────────────── 4. 版本库卫生 ────────────────────────────
BIG_PATHS = ["data/panel/market_panel.db",
             "data/panel/parts/daily_part_0000.parquet",
             "data/portfolio.db.bak_20261016_184233"]


def git_stats() -> dict:
    def run(*args) -> str:
        try:
            r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
            return r.stdout
        except Exception:
            return ""

    dirty = [ln for ln in run("status", "--porcelain").splitlines() if ln.strip()]
    last = run("log", "-1", "--format=%cs %s").strip()
    trackable = []
    for p in BIG_PATHS:
        if not (ROOT / p).exists():
            continue
        r = subprocess.run(["git", "check-ignore", "-q", p], cwd=ROOT)
        trackable.append(p) if r.returncode != 0 else None
    return {"dirty": len(dirty), "last_commit": last, "trackable_big_files": trackable}


def _flat(v):
    """sqlite 里 `group_concat(...)` 之类会被 fetchone() 包一层元组，打日志时摊平。"""
    if isinstance(v, (tuple, list)) and len(v) == 1:
        return v[0]
    return v


def print_human(c: dict, p: dict, pp: dict, d: dict, g: dict) -> None:
    print("=" * 66)
    print("bigAagents 基线体检（只读 / 不联网）")
    print("=" * 66)
    print(f"[代码] {c['py_files']} 个 py / 源码 {c['src_lines']:,} 行 / "
          f"测试 {c['test_files']} 文件 {c['test_lines']:,} 行 / "
          f"{c['test_funcs']} 个 test 函数")
    print("       （参数化后的实际用例数以 `pytest tests -q --collect-only` 为准）")
    if p.get("rows") is not None:
        print(f"[面板] {p.get('parts')} 个 part / {p['rows']:,} 行 / {p.get('codes')} 只 / "
              f"{p.get('date_from')}~{p.get('date_to')} / {p.get('size_mb')} MB")
        if p.get("backup_mb"):
            print(f"       另有 parts_backup_datefmt/ 备份 {p['backup_mb']} MB（可删）")
        if p.get("cols"):
            print(f"       列: {', '.join(p['cols'])}")
    if pp:
        print(f"[面板元数据] progress.json: done={pp.get('done')} failed={pp.get('failed')} "
              f"(updated {pp.get('updated')})  ← 与上面实际覆盖对齐了吗？")
    if d:
        print(f"[库] {d.get('tables')} 张表")
        print(f"      analyses={d.get('analyses_rows')} 状态[{_flat(d.get('analyses_status'))}] "
              f"ret_5d/20d={d.get('analyses_ret')}")
        print(f"      screen_picks={d.get('screen_picks_rows')} 状态[{_flat(d.get('screen_status'))}] "
              f"ret_5d/20d/天数={d.get('screen_ret')}")
        if d.get("llm"):
            n, tok, lat, fail = d["llm"]
            print(f"      llm_calls={n} tokens={tok} 平均={lat}ms 失败={fail}")
        print(f"      daily_features={d.get('daily_features')}  data_coverage={_flat(d.get('coverage_rows'))} 行")
    print(f"[版本库] 未提交 {g.get('dirty')} 项 / 最后提交 {g.get('last_commit')}")
    if g.get("trackable_big_files"):
        print("      ⚠️ 以下大文件**未被 .gitignore 覆盖**（一条 git add -A 就会入库）：")
        for f in g["trackable_big_files"]:
            print(f"        - {f}")
    else:
        print("      ✅ 已探测的大文件均被忽略")
    print("=" * 66)


def main() -> int:
    ap = argparse.ArgumentParser(description="基线体检（只读）")
    ap.add_argument("--json", action="store_true", help="JSON 输出")
    a = ap.parse_args()
    c, p, pp, d, g = code_stats(), panel_stats(), panel_progress(), db_stats(), git_stats()
    if a.json:
        print(json.dumps({"code": c, "panel": p, "panel_progress": pp,
                          "db": d, "git": g}, ensure_ascii=False, default=str, indent=2))
    else:
        print_human(c, p, pp, d, g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
