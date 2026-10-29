"""深度分析复盘：批量为已回填（T+20）的分析生成归因。

用法：
    # 只做机械归因（零 LLM，零成本）——补 ATR 与基准超额
    python scripts/run_analysis_review.py --limit 50

    # 再加模型归因（受限选择 + 证据/改进项校验）
    python scripts/run_analysis_review.py --limit 10 --llm

    # 只看近半年归因分布，不写库
    python scripts/run_analysis_review.py --report-only

为什么不放进 JOBS：
    机械归因已经挂在 15:35 回填里自动产出（零成本、零风险）；模型归因是**花钱**
    的一步，且它的价值依赖样本积攒（单条归因是噪声）。放进每日调度会让"每天固定
    花一笔钱"变成默认行为，而这笔钱在样本 <30 时买不到任何结论。
    等样本够了想自动化，再在 `core/scheduler.py::JOBS` 加一条即可（记得重启）。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import analysis_review  # noqa: E402
from utils.logger import get_logger  # noqa: E402

_logger = get_logger("analysis_review")


def main() -> int:
    ap = argparse.ArgumentParser(description="深度分析复盘（归因）")
    ap.add_argument("--limit", type=int, default=20, help="单次处理条数上限")
    ap.add_argument("--llm", action="store_true",
                    help="同时做模型归因（受限选择 + 校验）。不加则只做机械归因")
    ap.add_argument("--days", type=int, default=180, help="聚合统计回看天数")
    ap.add_argument("--report-only", action="store_true", help="只打印归因分布，不写库")
    args = ap.parse_args()

    if not args.report_only:
        stats = analysis_review.review_pending(limit=args.limit, with_llm=args.llm)
        print(f"复盘：处理 {stats['checked']} 条，写入 {stats['reviewed']} 条，"
              f"跳过 {stats['skipped']} 条"
              + (f"；模型归因 {stats['llm_ok']} 成功 / {stats['llm_error']} 失败"
                 if args.llm else "（仅机械归因）"))
        for e in stats["errors"][:10]:
            print(f"  ! {e}")

    # 两个窗口分开看：T+5 的"判对"和 T+20 的"判对"不是一回事，混算没有意义
    for hz in (20, 5):
        agg = analysis_review.aggregate_recent(days=args.days, horizon=hz)
        print(f"\n【T+{hz} 窗口】")
        print(agg["text"])
        if agg.get("pending"):
            print("（以下改进项出现次数不足，暂不升级为建议：）")
            for i in agg["pending"][:5]:
                print(f"  · {i['label']}：{i['count']} 次")
    _logger.info("复盘脚本完成：days=%s", args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
