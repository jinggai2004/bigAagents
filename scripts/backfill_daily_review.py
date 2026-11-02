# -*- coding: utf-8 -*-
"""补生成历史交易日的每日复盘报告（DB 可复核部分）。

背景（2026-10-31）：10-30 15:50 `daily_review` 开始后因网络故障未完成
（东财快照熔断 + akshare 断连），`output/daily_reports/review_20261030.md` 缺档。

为什么不直接调 `review_agent.generate_review()`：
它会把**当前**行情喂给 LLM 并把观察核对记到当天 —— 给历史日期用会产出
"数字与叙述错位"的报告（正是"页面与 prompt 同源"铁律要防的），还会污染
预测跟踪闭环。因此本脚本只渲染**纯 DB、可按日期精确重放**的章节：
  - 数据完整性（`stage_health.data_integrity_markdown(trade_date)`）
  - 决策三章（`build_digest(as_of)` → `render_decision_sections`）
  - 阿深日报（`ashen_daily.render_markdown(day)`）
并**明确标注**叙述部分（大盘概览/LLM 点评）不可回溯、缺失原因。

用法
────
    py scripts/backfill_daily_review.py --date 2026-10-30
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
import logging

_logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REPORT_DIR = _ROOT / "output" / "daily_reports"


def build_report(date: str) -> str:
    from core import ashen_daily, stage_health
    from core.decision_digest import build_digest
    from core.review_agent import render_decision_sections

    digest = build_digest(as_of=date)
    parts = [
        f"# A股每日复盘 {date}\n",
        f"> **补生成**于 {datetime.now():%Y-%m-%d %H:%M} —— 原 15:50 任务因网络故障"
        f"（东财快照熔断 + akshare 断连）未完成。\n"
        f"> 本报告只含可按日期精确重放的 DB 章节；**叙述部分（大盘概览/当日行情/"
        f"LLM 点评/观察核对）依赖当日实时数据，不可回溯，如实缺失**。\n",
    ]
    try:
        integrity = stage_health.data_integrity_markdown(date)
        if integrity:
            parts.append(integrity)
    except Exception as e:                          # noqa: BLE001
        _logger.warning("数据完整性段渲染失败: %s", e)
        parts.append(f"> 数据完整性段渲染失败: {e}\n")
    try:
        decision_md = render_decision_sections(digest)
        if decision_md:
            parts.append(decision_md)
    except Exception as e:                          # noqa: BLE001
        _logger.warning("决策三章渲染失败: %s", e)
        parts.append(f"> 决策三章渲染失败: {e}\n")
    try:
        ashen_md = ashen_daily.render_markdown(date)
        if ashen_md:
            parts.append(ashen_md)
    except Exception as e:                          # noqa: BLE001
        _logger.warning("阿深日报渲染失败: %s", e)
        parts.append(f"> 阿深日报渲染失败: {e}\n")
    return "\n".join(p if p.endswith("\n") else p + "\n" for p in parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="补生成的交易日（YYYY-MM-DD）")
    ap.add_argument("--force", action="store_true", help="已存在也重写")
    a = ap.parse_args(argv)
    date = a.date.strip()
    out = REPORT_DIR / f"review_{date.replace('-', '')}.md"
    if out.exists() and out.stat().st_size > 0 and not a.force:
        print(f"已存在（{out}），跳过。--force 重写。")
        return 0
    text = build_report(date)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"已补生成：{out}（{len(text)} 字符）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
