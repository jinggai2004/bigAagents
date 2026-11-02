# -*- coding: utf-8 -*-
"""分析图节点耗时台账——进度条**按耗时加权**的权重来源（2026-10-31，M4 尾巴）。

为什么
──────
`web/server/jobs.py` 的进度原是 `step_idx / NODE_STEPS` **等分**，但节点实际
耗时差异巨大（timing/取数 44~260s，评分毫秒级）⇒ 进度条前 1/15 格走掉 90%
的真实时间、然后 14 格瞬间走完，用户看到的是"长期卡在 3% 然后跳 100%"。

做法
────
- 每次运行把「节点完成时刻 − 上一节点完成时刻」用 **EWMA（α=0.3）** 记到
  `data/analysis_node_timing.json`；
- 进度 = 已完成节点的历史均耗时之和 ÷（已完成 + 剩余 × 全体均值），
  剩余按 `NODE_STEPS − 已完成数` 估——不需要知道完整节点名单；
- 无台账（首次运行）⇒ 返回 None，调用方回退等分 ⇒ 冷启动无害。

边界
────
台账只是体验优化，**任何异常都不能伤主流程**：读写全部静默降级。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from config.paths import data_root

_logger = logging.getLogger("node_timing")

# ⚠️ frozen 下 `__file__` 只读 ⇒ 数据路径走 data_root()（铁律 §14）
PATH = Path(data_root()) / "data" / "analysis_node_timing.json"
ALPHA = 0.3            # EWMA：新样本权重（≈ 最近 3~4 次运行的记忆）
OUTLIER_CAP = 4.0      # 单节点权重 ≤ 全体均值 ×4，防离群拖爆分母


def _load() -> dict[str, float]:
    try:
        d = json.loads(PATH.read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in d.items() if isinstance(v, (int, float))}
    except Exception as e:                               # noqa: BLE001  缺/坏文件=冷启动
        _logger.debug("节点计时文件缺失/损坏（按冷启动处理）: %s", e)
        return {}


def _save(d: dict[str, float]) -> None:
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, PATH)                                # 原子替换（读者不会读半个文件）


def record(node: str, seconds: float) -> None:
    """EWMA 更新一个节点的耗时（秒）。异常静默。"""
    try:
        s = float(seconds)
        if not (0 < s < 3600):                           # 时钟漂移/异常值不入账
            return
        d = _load()
        old = float(d.get(str(node)) or 0)
        d[str(node)] = round(ALPHA * s + (1 - ALPHA) * old, 2) if old > 0 else round(s, 2)
        _save(d)
    except Exception as e:                               # noqa: BLE001
        _logger.warning("[节点耗时] 记录失败（不影响主流程）: %s", e)


def weighted_percent(done_nodes: list[str], total_steps: int) -> float | None:
    """按历史耗时算已完成进度（0~100）。无台账数据 ⇒ None（回退等分）。"""
    d = _load()
    if not d:
        return None
    avg = sum(d.values()) / max(len(d), 1)
    if avg <= 0:
        return None
    done_w = sum(min(d.get(n, avg), avg * OUTLIER_CAP) for n in (done_nodes or []))
    remain = max(int(total_steps) - len(done_nodes or []), 1) * avg
    return round(min(done_w / (done_w + remain), 0.99) * 100, 1)
