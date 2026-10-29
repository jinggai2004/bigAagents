# -*- coding: utf-8 -*-
"""启动只读服务层 + 静态前端（P1）。

用法：
    python scripts/serve_web.py                 # 127.0.0.1:8765
    python scripts/serve_web.py --port 9000
    python scripts/serve_web.py --reload        # 开发时热重载

⚠️ 只读服务，**不要**把它当调度器：写操作（跑选股/采集/下单）仍走 `core/ui_tasks.py`
   任务队列，且 P1 起 Streamlit 侧写入口下掉（文档 §6.1 C5）。
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description="bigAagents 只读 API + 静态前端")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--reload", action="store_true", help="开发模式：改码自动重启")
    args = ap.parse_args()

    import uvicorn
    # 预热只在**服务进程内**生效：`--reload` 由 uvicorn 起子进程，环境变量会继承过去
    os.environ.setdefault("BIGA_WARMUP", "1")
    print(f"[serve] http://{args.host}:{args.port}/  (docs: /api/docs)")
    uvicorn.run("web.server.app:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
