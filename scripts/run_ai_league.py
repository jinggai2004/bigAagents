# -*- coding: utf-8 -*-
"""多 AI 交易联赛 · 盘后决策轮手动入口（2026-10-14，S4）。

用法：
    python scripts/run_ai_league.py --dry-run              # 演练：不写库、不真调 LLM 之外的副作用
    python scripts/run_ai_league.py                       # 真跑（会调 LLM，写入 ai_decisions）
    python scripts/run_ai_league.py --date 2026-10-14 --max-rounds 4
    python scripts/run_ai_league.py --persona ai_value     # 只跑一个 persona（调试用）

⚠️ 为什么先给手动入口、暂不挂 15:58 调度：
   挂上后每个交易日会真花 token（3 persona × 最多 6 轮 ≈ 最坏 ~18 次调用）。
   这是**成本决策**，应由用户确认后再挂——但代码路径与调度完全一致，
   挂接只需在 core/scheduler.py::JOBS 加一条（见文件末尾注释）。
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="AI 交易联赛 · 盘后决策轮")
    ap.add_argument("--date", default=None, help="交易日（默认今天）")
    ap.add_argument("--max-rounds", type=int, default=6, help="每个 AI 最多几轮工具循环")
    ap.add_argument("--persona", default=None, help="只跑指定 persona（调试）")
    ap.add_argument("--dry-run", action="store_true", help="演练：不写库")
    ap.add_argument("--show", action="store_true", help="打印每个 AI 的决策详情")
    ap.add_argument("--confirm-close", action="store_true",
                    help="用当日收盘价把未成交买入确认为持仓（加速确认，不跑 LLM）")
    ap.add_argument("--no-analysis", action="store_true",
                    help="不接通现场深度分析执行器（调试用；默认接通，与调度一致）")
    args = ap.parse_args(argv)

    from core import ai_daily_round, ai_personas, ai_budget, ai_league

    # --confirm-close：只做"收盘价确认持仓 + 记净值"，完全不调 LLM（零成本）。
    if args.confirm_close:
        day = args.date
        print(f"=== AI 持仓确认（当日收盘价，{day or '今天'}）===")
        out = ai_league.settle_at_close(day, dry_run=args.dry_run)
        print(f"交易日: {out['trade_date']}｜口径: {out['mode']}")
        for r in out["results"]:
            print(f"  {r['persona']:12s} 成交 {r['filled']} 笔｜持仓 {r['positions']} 只"
                  f"｜净值 {r['nav']['total_value']:.2f}"
                  + (f"（{r['at_cost']} 只按成本估值）" if r["at_cost"] else ""))
            for it in r["items"]:
                print(f"       - {it}")
        return 0

    day = args.date
    personas = None
    if args.persona:
        p = ai_personas.get_persona(args.persona)
        if not p:
            print(f"找不到 persona: {args.persona}")
            return 2
        personas = [p]

    print(f"=== AI 决策轮 {day or '今天'}（dry_run={args.dry_run}）===")
    # ⚠️ 默认**接通**现场深度分析执行器（与 15:58 调度任务同一条代码路径）；
    #    `--no-analysis` 只用于调试（否则调试轮会因 AI 申请研究而花掉 ~4.7 万 token/次）。
    _runner = None
    if not args.no_analysis:
        from core.analysis_runner import ai_deep_analysis_runner
        _runner = ai_deep_analysis_runner(source="ai")
    out = ai_daily_round.run_daily_round(day, max_rounds=args.max_rounds,
                                        dry_run=args.dry_run, personas=personas,
                                        analysis_runner=_runner)
    s = out["summary"]
    print(f"交易日: {out['trade_date']}")
    if out["briefing_warnings"]:
        print("简报警告（如实展示，不补造）:")
        for w in out["briefing_warnings"]:
            print(f"  ⚠️ {w}")
    print(f"共 {s['total']} 个 AI：成功 {s['ok']}｜额度跳过 {s['skipped']}｜失败 {s['failed']}")
    for r in out["results"]:
        flag = {"ok": "✅", "skipped": "⏭️"}.get(r["status"], "❌")
        print(f"  {flag} {r['persona']:12s} [{r['status']}] "
              f"决策 {len(r['decisions'])} 条"
              + (f"｜{r['error']}" if r["error"] else ""))
        if args.show:
            for d in r["decisions"]:
                print(f"       - {d}")
            if r.get("summary"):
                print(f"       思路: {r['summary'][:120]}")
    if s.get("persisted"):
        print(f"落库: {s['persisted']}")
    print()
    print("额度视图（抽查 3 个 LLM persona + 1 个随机对照）:")
    for name in ("ai_value", "ai_tech", "ai_risk", "ai_random_1"):
        v = ai_budget.view(name, out["trade_date"])
        print(f"  {name:12s} deep_analysis {v['deep_analysis']['used']}/{v['deep_analysis']['quota']}"
              f"｜wake {v['wake']['used']}/{v['wake']['quota']}"
              f"｜round {v['decision_round']['used']}/{v['decision_round']['quota']}")
    return 0


if __name__ == "__main__":
    # ── 挂到调度（需用户确认，会每天花 token）──
    # 在 core/scheduler.py::JOBS 追加：
    #   {"id": "ai_daily_round", "name": "AI交易联赛·盘后决策轮", "func": run_ai_league_job,
    #    "cron": {"hour": 15, "minute": 58, "day_of_week": "mon-fri"}, ...}
    # ⚠️ 新增 JOBS 任务必须全仓搜断言（写死 len(JOBS)==N 的用例会红）。
    sys.exit(main(sys.argv[1:]))
