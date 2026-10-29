# -*- coding: utf-8 -*-
"""一次性数据修复（2026-10-15）：补/修 AI 联赛的历史唤醒条件。

背景（两处真实缺陷，详见 `.workbuddy/memory/2026-10-15.md`）：
  ① `ai_tech/600519` 的止损价 `threshold=5.0`（现价千元级）⇒ `ai_wake` 判其无效不触发，
     而 `ensure_stops()` 又因"已存在 stop_loss"不补兜底 ⇒ **僵尸止损、持仓裸奔**。
  ② `ai_value/600028`（10-14 买入）**完全没有止损条件** —— 那一轮跑在 `ensure_stops`
     上线之前，代码兜底从未生效。

本脚本**只改这两处**，且：
  · 阈值一律按 **持仓成本 × (1 − 8%)** 计算（与 `ai_daily_round.DEFAULT_STOP_PCT` 同源，
    不引入新口径）；
  · 原值/原因写进 `note`，**不改写历史**（可审计）；
  · 幂等：已有**有效**止损就不动；重复运行无副作用。

用法：
    python scripts/repair_ai_legacy_stops_2026-10-15.py --dry-run   # 只看要改什么
    python scripts/repair_ai_legacy_stops_2026-10-15.py             # 真的改
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")

# 需要修的目标：(persona, code, 修复原因)
TARGETS = [
    ("ai_tech", "600519", "原 threshold=5.0 对千元级标的永不触发（僵尸止损）"),
    ("ai_value", "600028", "10-14 买入时漏了代码兜底（ensure_stops 尚未上线）"),
]


def _last_snapshot_date() -> str | None:
    try:
        from db import store
        dates = store.market_snapshot_dates(limit=1)
        return dates[-1] if dates else None
    except Exception:                                        # noqa: BLE001
        return None


def main(argv):
    dry = "--dry-run" in argv
    from core import ai_agent, ai_daily_round
    from db import store

    pct = float(ai_daily_round.DEFAULT_STOP_PCT)
    day = _last_snapshot_date()
    print(f"参考日期（最近快照）：{day}｜兜底幅度：−{pct:.0%}")
    if not day:
        print("⚠️ 取不到快照日期 ⇒ 无法校验'是否有效'，中止（宁可不改）")
        return 2

    changed = 0
    for persona, code, why in TARGETS:
        st = store.load_paper_state(persona)
        pos = ((st or {}).get("positions") or {}).get(code)
        if not pos or not pos.get("quantity"):
            print(f"\n[跳过] {persona}/{code}：无持仓（不凭空造条件）")
            continue

        qty = int(pos["quantity"])
        cost = float(pos["total_cost"]) / qty
        target = round(cost * (1 - pct), 2)

        # 现价：快照优先（离线），用于判"现有阈值是否有效"
        from core import ai_league
        px = ai_league.close_prices([code], day).get(code)

        conds = [c for c in store.list_ai_wake_conditions(persona=persona, armed_only=False)
                 if str(c.get("code")) == code and c.get("kind") == "stop_loss"]
        print(f"\n{persona}/{code}：成本 {cost:.4f} × {qty}｜现价 {px}｜"
              f"兜底止损目标 {target:g}｜现有止损 {len(conds)} 条")
        for c in conds:
            ok, reason = ai_agent.wake_threshold_ok("stop_loss", c.get("threshold"), px)
            print(f"   cond_id={c['cond_id']} threshold={c.get('threshold')} "
                  f"pct={c.get('threshold_pct')} armed={c.get('armed')} "
                  f"有效={ok}{'' if ok else '（' + reason + '）'} note={c.get('note')!r}")

        valid = [c for c in conds
                 if ai_agent.wake_threshold_ok("stop_loss", c.get("threshold"), px)[0]
                 and (c.get("threshold") is not None or c.get("threshold_pct") is not None)]
        if valid:
            print("   → 已有有效止损，不改动")
            continue

        note = f"⚠️ 系统修复(2026-10-15)：{why}；按持仓成本 −{pct:.0%} 重设为 {target:g}"
        if dry:
            action = (f"更新 cond_id={conds[0]['cond_id']}" if conds else "新增")
            print(f"   → [dry-run] 将{action} stop_loss threshold={target}")
            continue
        if conds:
            cid = conds[0]["cond_id"]
            store.upsert_ai_wake_condition(persona, {
                "cond_id": cid, "code": code, "kind": "stop_loss",
                "threshold": target, "threshold_pct": pct, "armed": True, "note": note})
            print(f"   → 已更新 cond_id={cid}：threshold={target}")
        else:
            cid = store.upsert_ai_wake_condition(persona, {
                "code": code, "kind": "stop_loss",
                "threshold": target, "threshold_pct": pct, "armed": True, "note": note})
            print(f"   → 已新增 cond_id={cid}：threshold={target}")
        changed += 1

    print(f"\n{'[dry-run] ' if dry else ''}共修改 {changed} 处")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
