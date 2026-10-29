# -*- coding: utf-8 -*-
"""一次性修复：清理 2026-10-16 补跑残留与唤醒条件重复行。

背景（2026-10-16 排查结论）：
  ① **补跑残留（幽灵决策）**：人工补跑与当日正式盘后轮会在**同一天各产出一批决策**，
     而 `ai_decisions` 的 UPSERT 主键是 `(persona, signal_date, code)`
     —— **code 不同就成两行** ⇒ `ai_random_2` 当天出现"两条买入"
     （300199 来自 09:10 补跑、600403 来自 15:59 正式轮）。
     随机组的设计是**每天最多 1 个买入**（`ai_agent.random_decisions` 只 `r.choice` 一次）
     ⇒ 两条必然有一条是残留。留着它，等结算修好后 10-19 会**多买一只**。
  ② **唤醒条件重复**：旧 `upsert_ai_wake_condition` 按 `(persona, cond_id)` 冲突，
     而 cond_id 由函数自增、调用方不传 ⇒ **永远 INSERT、从不 UPDATE**
     ⇒ 同一 `(persona, code, kind)` 累积多行（同一只票两个止损位，风控语义被破坏）。

判定规则（保守，只动"确定有问题的"）：
  · 同一随机 persona 在同一 signal_date 出现**多笔实质动作**时，保留 `created_at` 最新的
    （正式轮产物）；**同一 code 被 UPSERT 过只会有 1 行 ⇒ 天然不会误伤**。
  · 同一 `(persona, code, kind)` 只保留 `cond_id` 最大的一条。
  · 幽灵标的 `ai_random_2/300199` 的唤醒条件一并清理（其买入决策已不存在）。

用法：
    python scripts/repair_ai_wake_dup_2026-10-16.py           # 只预览（默认 dry-run）
    python scripts/repair_ai_wake_dup_2026-10-16.py --apply    # 实际写入（自动备份）
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.models import DB_PATH, get_connection  # noqa: E402

DAY = "2026-10-16"
# 幽灵标的：其当日买入决策来自补跑（随机组每天最多一个买入），必须随决策一起清掉
GHOST = ("ai_random_2", "300199")
SUBSTANTIVE = ("可执行", "卖出")          # 实质动作（等待/'-' 不算）

# 测试残留事件：早期用例**未隔离**、直接写进真实库的产物。
# 判据用**精确靶点**而不是模糊匹配（茅台 trigger_px=110 不可能是真实行情；
# hit_at 还是整点 10:00:00 / 20:00:00）。
TEST_RESIDUE = [
    ("ai_tech", "2026-09-22", "600519", 1),
    ("ai_tech", "2026-09-22", "600519", 2),
    ("ai_tech", "2026-09-22", "600519", 3),   # 19:04 那次全量回归新增（`test_watcher.py` 未隔离）
    ("ai_tech", "2026-09-26", "600519", 1),
    ("ai_tech", "2026-09-26", "600519", 2),
    ("ai_tech", "2026-09-26", "600519", 3),
]


def _randoms() -> set[str]:
    """随机对照组的 persona 名单（唯一来源仍是人格注册表）。"""
    from core import ai_personas
    return {p["name"] for p in ai_personas.load_personas()
            if p.get("kind") == "random" and p.get("enabled")}


def plan(conn: sqlite3.Connection) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """算出要删的行：Returns (决策行, 条件行, 残留事件行)，元素为可直接 DELETE 的定位元组。"""
    randoms = _randoms()

    # ① 同日多笔实质动作的随机 persona ⇒ 只保留 created_at 最新的一笔
    dup_rows: list[tuple] = []
    for r in conn.execute(
            "SELECT persona, signal_date FROM ai_decisions "
            "WHERE signal_date=? AND advice_status IN (?,?) "
            "GROUP BY persona, signal_date HAVING COUNT(*) > 1", (DAY, *SUBSTANTIVE)):
        persona, day = r[0], r[1]
        if persona not in randoms:
            # ⚠️ LLM 组一天可以有多个买入（不同标的），**不能**按"只留一条"处理
            continue
        keep = conn.execute(
            "SELECT code FROM ai_decisions WHERE persona=? AND signal_date=? "
            "AND advice_status IN (?,?) ORDER BY created_at DESC, code DESC LIMIT 1",
            (persona, day, *SUBSTANTIVE)).fetchone()
        keep_code = keep[0] if keep else None
        for row in conn.execute(
                "SELECT code, created_at FROM ai_decisions WHERE persona=? AND signal_date=? "
                "AND advice_status IN (?,?) AND code IS NOT ?",
                (persona, day, *SUBSTANTIVE, keep_code)):
            dup_rows.append((persona, day, row[0], row[1], f"保留 {keep_code}"))

    # ② 同一 (persona, code, kind) 保留 cond_id 最大的一条
    cond_rows: list[tuple] = []
    for r in conn.execute(
            "SELECT persona, code, kind, GROUP_CONCAT(cond_id) ids, MAX(cond_id) keep "
            "FROM ai_wake_conditions GROUP BY persona, code, kind HAVING COUNT(*) > 1"):
        persona, code, kind, ids, keep = r[0], r[1], r[2], str(r[3]), r[4]
        for cid in ids.split(","):
            if int(cid) != int(keep):
                cond_rows.append((persona, code, kind, int(cid), f"保留 cond_id={keep}"))

    # ③ 幽灵标的的唤醒条件（买入决策已被清掉，条件成为孤儿）
    for r in conn.execute("SELECT persona, code, kind, cond_id FROM ai_wake_conditions "
                          "WHERE persona=? AND code=?", GHOST):
        cond_rows.append((r[0], r[1], r[2], int(r[3]), "幽灵标的（买入决策来自补跑）"))

    # ④ 测试残留事件（见 TEST_RESIDUE 注释）
    event_rows: list[tuple] = []
    for persona, day, code, cid in TEST_RESIDUE:
        row = conn.execute(
            "SELECT trigger_px FROM ai_wake_events WHERE persona=? AND trade_date=? "
            "AND code=? AND cond_id=?", (persona, day, code, cid)).fetchone()
        if row is not None:
            event_rows.append((persona, day, code, cid, f"测试残留（trigger_px={row[0]}）"))
    return dup_rows, cond_rows, event_rows


def backup() -> Path:
    src = Path(DB_PATH)
    dst = src.with_suffix(f".db.bak_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(src, dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description="清理 2026-10-16 补跑残留与唤醒条件重复行")
    ap.add_argument("--apply", action="store_true", help="实际写入（默认只预览）")
    args = ap.parse_args()

    conn = get_connection()
    try:
        decs, conds, events = plan(conn)
        print(f"=== 计划清理（{'实际写入' if args.apply else 'dry-run 预览'}）===")
        print(f"\n[决策] {len(decs)} 行")
        for persona, day, code, created, why in decs:
            print(f"  - DELETE ai_decisions ({persona}, {day}, {code}) created_at={created}｜{why}")
        print(f"\n[唤醒条件] {len(conds)} 行")
        for persona, code, kind, cid, why in conds:
            print(f"  - DELETE ai_wake_conditions ({persona}, {code}, {kind}, cond_id={cid})｜{why}")
        print(f"\n[残留事件] {len(events)} 行")
        for persona, day, code, cid, why in events:
            print(f"  - DELETE ai_wake_events ({persona}, {day}, {code}, cond_id={cid})｜{why}")

        if not (decs or conds or events):
            print("\n无需要清理的行（幂等）。")
            return 0
        if not args.apply:
            print("\n（dry-run）加 --apply 才会写入。")
            return 0

        dst = backup()
        print(f"\n已备份真实库 → {dst.name}")
        for persona, day, code, _created, _why in decs:
            conn.execute("DELETE FROM ai_decisions WHERE persona=? AND signal_date=? AND code=?",
                         (persona, day, code))
        for persona, code, kind, cid, _why in conds:
            conn.execute("DELETE FROM ai_wake_conditions WHERE persona=? AND code=? AND kind=? "
                         "AND cond_id=?", (persona, code, kind, cid))
        for persona, day, code, cid, _why in events:
            conn.execute("DELETE FROM ai_wake_events WHERE persona=? AND trade_date=? "
                         "AND code=? AND cond_id=?", (persona, day, code, cid))
        conn.commit()

        # 清完再试一次唯一索引（它是 (persona, code, kind) 的数据库级兜底）
        try:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_wake_cond_key "
                         "ON ai_wake_conditions(persona, code, kind)")
            conn.commit()
            print("已建立唯一索引 idx_ai_wake_cond_key(persona, code, kind)")
        except sqlite3.IntegrityError as e:
            print(f"⚠️ 唯一索引仍未建立（还有重复行）: {e}")
        print("清理完成。")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
