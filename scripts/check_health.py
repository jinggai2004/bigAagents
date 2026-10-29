"""运行健康自检：调度器在不在跑、数据在不在长。

用法:
    python scripts/check_health.py          # 打印人类可读报告
    python scripts/check_health.py --json   # 输出 JSON（供脚本/监控消费）

退出码: 0=一切正常（无警告）；1=有需要处理的问题。
把"服务在跑"从"靠人翻日志"变成"一条命令可查"——这是批次 16 的教训：
调度器曾长期从未启动，而没有任何地方能看出来。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser(description="bigAagents 运行健康自检")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args()

    from core.scheduler_health import check_health, format_report

    health = check_health()
    if args.json:
        print(json.dumps(health, ensure_ascii=False, indent=2))
    else:
        print(format_report(health))
    return 1 if health.get("warnings") else 0


if __name__ == "__main__":
    sys.exit(main())
