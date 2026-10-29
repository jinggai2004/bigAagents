# -*- coding: utf-8 -*-
"""一次性更正真实持仓 600028（中石化）的**录入错误价位**（2026-10-16）。

## 事故

持仓页「成交价」原有一个**假默认值 `10.0`**（`value=10.0, min_value=0.01`）：
用户不改就提交 ⇒ 成本价被**静默**写成 10.0（中石化真实价 **5.53**），
止损又填成 **10.5**。三重后果：

1. 盈亏 / 止损检查 / AI 诊断全部按**错的成本价**计算（实测 `loss_pct = -44.7%`）；
2. 止损 10.5 ≥ 现价 5.53 ⇒ **一录入就"已跌破止损"**，告警永久为真；
3. `min_value=0.01` 让 `price <= 0` 那句校验**永远不可能触发**（假校验）。

页面侧已修（去掉假默认值 + 保存前价位体检 + 人工确认）；本脚本只负责**更正这一行历史数据**。

## 用户确认的口径（2026-10-16 22:12）

- 成交价：**按今日收盘 5.53** 更正；
- 止损：**先清空（不设止损）** —— 之后用页面上的「按 ATR 建议止损」自己设。

## 纪律

- 默认 **dry-run**（只打印计划）；`--apply` 才写；
- 写前**自动备份**真库；**幂等**（重复跑只在数值不一致时动手）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

DB = os.path.join(_ROOT, "data", "portfolio.db")

CODE = "600028"
NEW_PRICE = 5.53          # 用户确认：按今日收盘价更正
NEW_STOP = None           # 用户确认：先清空（不设止损）
OLD_PRICE = 10.0          # 仅用于提示"当前值确实是错的"（不匹配也能修，只是会说明）


def backup() -> str:
    dst = f"{DB}.bak_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(DB, dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正写入（默认只预览）")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        "SELECT id, code, entry_price, stop_loss, quantity, status "
        "FROM positions WHERE code = ? AND status = 'active'", (CODE,)))
    if not rows:
        print(f"没有找到 {CODE} 的活跃持仓（幂等：无事可做）")
        return 0

    plan = []
    for r in rows:
        cur_price = float(r["entry_price"] or 0)
        cur_stop = r["stop_loss"]
        if abs(cur_price - NEW_PRICE) > 1e-9 or (cur_stop is not None and NEW_STOP is None):
            plan.append(r)

    print(f"=== 计划更正 {CODE}（{'实际写入' if args.apply else 'dry-run 预览'}）===")
    for r in rows:
        cur_stop = r["stop_loss"]
        print(f"  id={r['id']} 数量={r['quantity']}  "
              f"entry_price {r['entry_price']} → {NEW_PRICE} ｜ "
              f"stop_loss {cur_stop if cur_stop is not None else '(空)'} → "
              f"{NEW_STOP if NEW_STOP is not None else '(清空)'}")
    if not plan:
        print("\n数值已一致，无需改动（幂等）。")
        return 0
    if not args.apply:
        print("\n（dry-run）加 --apply 才会写入。")
        return 0

    dst = backup()
    print(f"\n已备份真实库 → {os.path.basename(dst)}")
    for r in plan:
        conn.execute("UPDATE positions SET entry_price = ?, stop_loss = ? WHERE id = ?",
                     (NEW_PRICE, NEW_STOP, r["id"]))
    conn.commit()
    print("已写入。")
    for r in conn.execute("SELECT id, code, entry_price, stop_loss FROM positions "
                          "WHERE code = ? AND status = 'active'", (CODE,)):
        print("  现在：", dict(r))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
