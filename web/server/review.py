# -*- coding: utf-8 -*-
"""复盘（`/api/review/*`）—— P3。

数据来源：
- 正文：`output/daily_reports/review_YYYYMMDD.md`（`BIGA_REVIEW_DIR` 可覆盖，沿用 Streamlit 页约定）
- 观察点闭环：`db observations`（`store.observation_stats` / `get_unverified_observations`）
- 今日决策：`core.decision_digest.build_digest()`（实测 ≈1.1s ⇒ 单独端点 + 60s 缓存，
  不塞进 `latest`，否则正文要等它）

⚠️ 三层口径必须分开写清楚（踩过的坑）：
1. **报告日期**（`date`）= 文件名里的交易日，不是查询时间；
2. `stale` = 最新报告日期 < 最近应收盘的交易日（复用 `picks.expected_date()`，不另造）；
3. `latest.md` 只是**指针**，它不存在不代表历史报告不存在 ⇒ 正文以 `review_*.md` 为准。
"""
from __future__ import annotations

import glob
import logging
import os
import re

from web.server import cache, envelope, picks, settings

_logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"review_(\d{8})\.md$")
_MAX_CHARS = 400_000          # 单篇正文上限：超长报告截断并标注 truncated


def _report_dir() -> str:
    return os.environ.get("BIGA_REVIEW_DIR") or os.path.join(
        settings.data_root(), "output", "daily_reports")


def list_reports() -> list[dict]:
    """历史复盘清单，按日期倒序。**只读文件名**，不解析正文（列表要快）。"""
    out = []
    for p in glob.glob(os.path.join(_report_dir(), "review_*.md")):
        m = _NAME_RE.search(os.path.basename(p))
        if not m:
            continue
        d = m.group(1)
        try:
            st = os.stat(p)
        except OSError as e:
            _logger.debug("[review] stat 失败，跳过 %s: %s", p, e)
            continue
        out.append({
            "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            "file": os.path.basename(p),
            "size": st.st_size,
            "mtime": st.st_mtime,
        })
    out.sort(key=lambda r: r["date"], reverse=True)
    return out


def _read(path: str) -> tuple[str, bool]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if len(text) > _MAX_CHARS:
        return text[:_MAX_CHARS], True
    return text, False


def _find(date: str | None, reports: list[dict]) -> dict | None:
    if date:
        for r in reports:
            if r["date"] == date:
                return r
        return None
    return reports[0] if reports else None


def _valid_date(date: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", date or ""))


def latest(now=None) -> dict:
    """最新一篇复盘正文 + 观察点闭环统计。

    分区包络（C2）：正文拿不到就把 `ok` 置 false（单资源端点）；
    `stats`/`pending` 失败只影响自己，正文照常返回。
    """
    reports = list_reports()
    if not reports:
        return envelope.fail("missing", f"{_report_dir()} 下无 review_*.md")

    head = reports[0]
    path = os.path.join(_report_dir(), head["file"])
    try:
        text, truncated = _read(path)
    except OSError as e:
        return envelope.fail("empty", f"{type(e).__name__}: {e}")

    expected = picks.expected_date(now)
    data = {
        "report": envelope.ok({
            "date": head["date"],
            "file": head["file"],
            "chars": len(text),
            "truncated": truncated,
            "mtime": head["mtime"],
            "markdown": text,
            "latest_md": os.path.exists(os.path.join(_report_dir(), "latest.md")),
            "history_count": len(reports),
        }, as_of=head["date"]),
        "stale": (head["date"] < expected) if expected else None,
        "expected_date": expected,
        "stats": _stats(),
        "pending": _pending(),
    }
    return envelope.ok(data, as_of=head["date"])


def _stats(days: int = 30) -> dict:
    try:
        from db import store
        s = store.observation_stats(recent_days=days)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[review] 观察点统计失败: %s", e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")
    return envelope.ok({**s, "window_days": days})


def _pending(limit: int = 50) -> dict:
    try:
        from db import store
        rows = store.get_unverified_observations()
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[review] 待核对观察点失败: %s", e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")
    return envelope.ok({
        "count": len(rows),
        "items": [{"id": r.get("id"), "review_date": r.get("review_date"),
                   "kind": r.get("kind") or "观察点", "content": r.get("content")}
                  for r in rows[:limit]],
    })


def pending(limit: int = 50) -> dict:
    """待核对观察点（首页与复盘页共用）。"""
    return _pending(limit=limit)


def archive(date: str | None = None) -> dict:
    """无 `date` ⇒ 清单；有 `date` ⇒ 该日正文。非法日期显式 `invalid`。"""
    if date is not None and not _valid_date(date):
        return envelope.fail("invalid", f"date 需为 YYYY-MM-DD，收到 {date!r}")

    reports = list_reports()
    if date is None:
        return envelope.ok({
            "count": len(reports),
            "items": reports,
            "dir": _report_dir(),
        })

    hit = _find(date, reports)
    if hit is None:
        return envelope.fail("missing", f"无 {date} 的复盘文件")
    path = os.path.join(_report_dir(), hit["file"])
    try:
        text, truncated = _read(path)
    except OSError as e:
        return envelope.fail("empty", f"{type(e).__name__}: {e}")
    return envelope.ok({**hit, "chars": len(text), "truncated": truncated,
                        "markdown": text}, as_of=date)


def digest(now=None) -> dict:
    """今日决策三块（实时读库，不依赖复盘是否已生成）。"""
    def _load():
        from core.decision_digest import build_digest
        return build_digest()

    try:
        data = cache.cached("review.digest", (now.date() if now else "now"), 60, _load)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[review] 决策汇总失败: %s", e)
        return envelope.fail("empty", f"{type(e).__name__}: {e}")
    return envelope.ok(data, as_of=(data or {}).get("today", {}).get("date"))
