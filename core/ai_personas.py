# -*- coding: utf-8 -*-
"""多 AI 交易联赛 · 人格注册表（2026-10-14，S2）。

设计照抄 config/providers.py 与 config/model_routing.py 的**已验证模式**：
mtime 缓存 / 原子写 + os.replace 退避 / 读失败降级 / invalidate()。
⇒ 改 JSON 即时生效，**不用重启** Streamlit 与调度器。

分工（避免双源漂移）：
- `config/ai_personas.json` 是**唯一定义源**（用户/本模块编辑）；
- `db.store.ai_personas` 表是**同步快照**（`sync_to_store()` 幂等 upsert），
  仅供 SQL JOIN（如 ai_decisions 关联人格）与健康检查读数；调度链路一律读 JSON。

用户已定的决策（2026-10-14，详见 output/多AI交易联赛_6项决策分析_2026-10-14.md）：
- 3 个 LLM persona（价值派/技术派/风控流）+ **3 个随机对照组**（决策 1 选 B，
  kind='random'，不调 LLM —— 没有它们排行在方法论上不成立）；
- 模型分配由用户选择（决策 2/3 选 B），默认 价值派=技术派→qwen-max、风控流→deepseek-chat，
  **改 JSON 的 model 字段即可换**，不用改代码；
- 规模：3 persona + 固定候选池 + 每日按需研究额度（决策 4 选 C，见 core/ai_budget.py）。
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

_logger = logging.getLogger(__name__)

PERSONAS_PATH = Path(__file__).resolve().parent.parent / "config" / "ai_personas.json"

# ---------------- 默认人格 ----------------
#
# ⚠️ style_prompt 只提供"取舍偏好"，**必须显式声明不得违反既有铁律**
#    （TRADE_DIRECTION_RULES / 反幻觉 / 禁止做空式价位）——风格不是豁免权。
#    铁律由代码兜底（core/ai_agent.clamp_plan），提示词管不住的部分不指望提示词。

_STYLE_VALUE = (
    "你是一名价值投资者：只依据基本面与估值做决策，偏好盈利有支撑、估值合理的标的；"
    "对纯题材炒作保持怀疑；仓位分散、单票从紧；宁可错过、不做没把握的交易。"
    "⚠️ 风格只影响你的取舍偏好，你必须同样遵守系统的全部交易纪律"
    "（方向三档口径、禁止做空式价位、禁止编造数据）——风格不是豁免权。")

_STYLE_TECH = (
    "你是一名趋势交易者：依据技术面（均线相对位置、量能、动量）判断进出时机，"
    "顺应趋势、进出果断；对趋势完好但基本面平庸的标的也可参与；"
    "每笔交易必须给出明确的止损位。"
    "⚠️ 风格只影响你的取舍偏好，你必须同样遵守系统的全部交易纪律"
    "（方向三档口径、禁止做空式价位、禁止编造数据）——风格不是豁免权。")

_STYLE_RISK = (
    "你是一名风控优先的投资者：本金安全第一，只在风险收益比明确有利时出手；"
    "仓位保守、宁可空仓等待；对高波动、高估值、纯题材信号一律回避；"
    "优先保护已有盈利的持仓（及时收紧止损）。"
    "⚠️ 风格只影响你的取舍偏好，你必须同样遵守系统的全部交易纪律"
    "（方向三档口径、禁止做空式价位、禁止编造数据）——风格不是豁免权。")

# 随机对照组的 style_prompt：留空 = 不带任何风格倾向。
# ⚠️ 它们的决策由代码规则生成（core/ai_agent 的 random 分支），完全不调 LLM。

# ⚠️ `quotas` 是**成本护栏**，必须与 config/ai_personas.json 保持一致：
#    实测一次"现场深度分析" = 11 次 LLM 调用 / ≈3.5 万 token / ≈3.2~3.9 分钟，
#    约等于整个决策轮（≈1.3 万 token）的 2.7 倍 ⇒ 每日给 1 次已足够体现"主动研究"，
#    要更激进请在该 JSON 里把 persona 的 `quotas.deep_analysis` 调大。
_LLM_QUOTAS = {"deep_analysis": 1}

# 阿深（模拟自营）的交易风格。与上面三个人设的关键区别：
# 它不是"参加比赛的人设"，是**有自己账户和自选池的常驻交易员**，
# 走 `core/ai_routine.py` 的定时上班通道，而不是日频决策轮。
# ⚠️ 结尾那句"风格不是豁免权"与其他人设**逐字同款**——它是纪律声明，
#    不是客套话：所有仓位上限由 `core/ai_autonomy.autonomy_check()` 机器判定，
#    提示词写得多好听都不会放宽任何一条。
_STYLE_ASHEN = (
    "你是阿深，一名自有模拟账户的 A 股交易员，账户名 ai_ashen。"
    "你只在自己的自选池内建仓，只在下跌时接手、不追涨；每笔买入必须同时给出"
    "限价、止损价和理由，缺一个这笔就不做。你宁可空着，也不为了『有动作』而下单。"
    "你按纪律做，不靠预测：不猜涨停、不满仓赌、不因为上一笔亏了就想扳回来。"
    "做错了就在复盘里说清错在哪一步，不把责任推给『市场变化』。"
    "⚠️ 风格只影响你的取舍偏好，你必须同样遵守系统的全部交易纪律"
    "（方向三档口径、禁止做空式价位、禁止编造数据）——风格不是豁免权。")

DEFAULT_PERSONAS: list[dict] = [
    # 2026-10-19：qwen 免费额度耗尽 ⇒ 全流程切 DeepSeek（默认值也要改，
    # 否则"配置文件丢失/损坏"时降级到的默认值会把调用打回 qwen）。
    {"name": "ai_value", "label": "价值派", "kind": "llm", "model": "deepseek-v4-flash",
     "style_prompt": _STYLE_VALUE, "risk_bias": "balanced", "quotas": dict(_LLM_QUOTAS),
     "info_scope": "all", "frequency": "daily", "initial_cash": 1_000_000,
     "enabled": True},
    {"name": "ai_tech", "label": "技术派", "kind": "llm", "model": "deepseek-v4-flash",
     "style_prompt": _STYLE_TECH, "risk_bias": "aggressive", "quotas": dict(_LLM_QUOTAS),
     "info_scope": "all", "frequency": "daily", "initial_cash": 1_000_000,
     "enabled": True},
    {"name": "ai_risk", "label": "风控流", "kind": "llm", "model": "deepseek-v4-flash",
     "style_prompt": _STYLE_RISK, "risk_bias": "conservative", "quotas": dict(_LLM_QUOTAS),
     "info_scope": "all", "frequency": "daily", "initial_cash": 1_000_000,
     "enabled": True},
    # 随机对照组（决策 1 选 B：3 个，能给出经验分布/分位；零 LLM 成本）
    {"name": "ai_random_1", "label": "随机对照 1", "kind": "random", "model": None,
     "style_prompt": "", "risk_bias": "balanced",
     "info_scope": "all", "frequency": "daily", "initial_cash": 1_000_000,
     "enabled": True},
    {"name": "ai_random_2", "label": "随机对照 2", "kind": "random", "model": None,
     "style_prompt": "", "risk_bias": "balanced",
     "info_scope": "all", "frequency": "daily", "initial_cash": 1_000_000,
     "enabled": True},
    {"name": "ai_random_3", "label": "随机对照 3", "kind": "random", "model": None,
     "style_prompt": "", "risk_bias": "balanced",
     "info_scope": "all", "frequency": "daily", "initial_cash": 1_000_000,
     "enabled": True},
    # 阿深（2026-10-27）：模拟自营账户。
    #   · `league_eligible=False` —— 它**不参加联赛**：研究方法不同、账户是新加的，
    #     混进榜会让"相对随机组超额"的分母被误读。`leaderboard()` 按此字段跳过。
    #   · `frequency="intraday"` —— 它是盘中定期上班（6 轮），不是日频决策轮。
    #     ⚠️ 因此 `run_daily_round` 必须按 `league_eligible` 排除，否则它每天
    #     会被日频轮再跑一次（花两份钱、产两套互相矛盾的决策）。
    #   · `max_position=0.08` **必须**等于 `ai_autonomy.DEFAULTS["max_position_pct"]`：
    #     `clamp_plan` 用 persona 的这个值夹仓位，autonomy 用参数表判越权，
    #     两处不一致时表现为"夹到 20% 再被 8% 拒掉"（看起来像模型乱报仓位）。
    #   · `autonomy="L2"` —— 建仓权已开（老板 2026-10-27 决定）。
    #   · `quotas.routine=6` 与 6 个 slot 一致；其余额度用 `DEFAULT_QUOTA`，
    #     不在这里复制一份（复制就会漂移）。
    {"name": "ai_ashen", "label": "阿深", "kind": "llm", "model": "deepseek-v4-flash",
     "style_prompt": _STYLE_ASHEN, "risk_bias": "balanced", "max_position": 0.08,
     "quotas": {"routine": 6, "deep_analysis": 3},
     "info_scope": "all", "frequency": "intraday", "initial_cash": 1_000_000,
     "enabled": True, "autonomy": "L2", "league_eligible": False},
]

_VALID_KINDS = ("llm", "random")
_VALID_BIAS = ("conservative", "balanced", "aggressive")
_VALID_FREQ = ("daily", "intraday")


def _sanitize(raw) -> list[dict]:
    """清洗 JSON 内容：丢弃缺 name/style_prompt 的条目、补默认值、修正非法枚举。
    ⚠️ 坏条目**跳过并留日志**，不让一个手误让整个注册表不可用（降级而非崩）。"""
    if not isinstance(raw, list):
        raise ValueError("顶层必须是数组")
    out: list[dict] = []
    seen: set = set()
    for it in raw:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        kind = it.get("kind") if it.get("kind") in _VALID_KINDS else "llm"
        entry = {
            "name": name,
            "label": str(it.get("label") or name),
            "kind": kind,
            # 随机对照组**没有模型**：给了也忽略（防止误把 LLM 成本算到对照组头上）
            "model": (str(it.get("model")).strip() or None) if kind == "llm" else None,
            "style_prompt": str(it.get("style_prompt") or ""),
            "risk_bias": it.get("risk_bias") if it.get("risk_bias") in _VALID_BIAS else "balanced",
            "max_position": float(it["max_position"]) if it.get("max_position") is not None else None,
            "info_scope": str(it.get("info_scope") or "all"),
            "frequency": it.get("frequency") if it.get("frequency") in _VALID_FREQ else "daily",
            "initial_cash": float(it.get("initial_cash") or 1_000_000),
            "enabled": bool(it.get("enabled", True)),
            # ── 阿深专用（2026-10-27）──────────────────────────────
            # ⚠️ 这个 `_sanitize` 是**白名单**式的：不在这里登记的字段，
            #    写进 JSON 也读不出来。加字段只改 JSON = 没接线（这类错误
            #    在项目里出现过，表现为"配置明明写了但代码里是默认值"）。
            #
            # `autonomy`：授权级别，落配置而非代码常量 ⇒ 改一个 JSON 就能升降级。
            # 缺省 **L0**（最保守）：别人格没有这个字段，它们也不走授权判定，
            # 但万一将来有人拿它当判据，缺省必须是"什么都别做"。
            "autonomy": str(it.get("autonomy") or "L0").strip().upper(),
            # `league_eligible`：是否参加 AI 联赛。**缺省 True**——
            # 必须保持既有 6 个人设的行为不变；阿深显式写 false。
            "league_eligible": bool(it.get("league_eligible", True)),
        }
        # quotas（C3 配额覆盖，core/ai_budget.quota_for 消费）：只接受 {str: int}，
        # 非法值跳过；不认得键名也原样保留（额度种类由 ai_budget 登记把关，这里不重复校验）。
        q = it.get("quotas")
        if isinstance(q, dict):
            clean_q = {}
            for k, v in q.items():
                try:
                    clean_q[str(k)] = int(v)
                except (TypeError, ValueError):
                    continue
            if clean_q:
                entry["quotas"] = clean_q
        out.append(entry)
    return out


_CACHE = {"path": None, "mtime": None, "loaded": False, "data": None}


def invalidate() -> None:
    """清缓存（测试与"刚写完文件要立刻读到新值"的场景用）。"""
    _CACHE.update(path=None, mtime=None, loaded=False, data=None)


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return -1.0


def _default_payload() -> list[dict]:
    return [dict(p) for p in DEFAULT_PERSONAS]


def save_personas(personas: list[dict]) -> bool:
    """原子写（os.replace + 退避，照抄 model_routing.save_feature_models）。
    写前先 sanitize——**存进去的就是将来读出来的**，不让脏数据过夜。"""
    clean = _sanitize(personas)
    if not clean:
        _logger.warning("[人格注册表] 拒绝写入空注册表（保留原文件）")
        return False
    tmp = PERSONAS_PATH.with_suffix(".json.tmp")
    try:
        PERSONAS_PATH.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(clean, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                os.replace(tmp, PERSONAS_PATH)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (attempt + 1))
        invalidate()
        return True
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[人格注册表] 写入失败（不静默）: %s", e)
        return False
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def load_personas() -> list[dict]:
    """读取人格列表（mtime 缓存）。文件不存在 → 落盘默认值；损坏 → **降级到默认值**
    并留日志（宁可回到已知好的状态，不用坏数据跑交易）。"""
    if (_CACHE["loaded"] and _CACHE["path"] == str(PERSONAS_PATH)
            and _CACHE["mtime"] == _mtime(PERSONAS_PATH)):
        return _CACHE["data"]

    if not PERSONAS_PATH.exists():
        save_personas(_default_payload())          # 顺带落盘，让"可编辑"可见
        data = _sanitize(_default_payload())
        _CACHE.update(path=str(PERSONAS_PATH), mtime=_mtime(PERSONAS_PATH),
                      loaded=True, data=data)
        return data

    try:
        raw = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))
        data = _sanitize(raw)
        if not data:
            raise ValueError("清洗后为空")
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[人格注册表] 配置损坏（%s），降级使用默认值", e)
        data = _sanitize(_default_payload())

    _CACHE.update(path=str(PERSONAS_PATH), mtime=_mtime(PERSONAS_PATH),
                  loaded=True, data=data)
    return data


def get_persona(name: str) -> dict | None:
    for p in load_personas():
        if p["name"] == name:
            return p
    return None


def llm_personas() -> list[dict]:
    """只取**参加联赛**的启用 LLM persona（随机对照组不进决策轮，另走随机分支）。

    ⚠️ 必须排除 `league_eligible=False`（阿深）：它是常驻交易员，走
    `core/ai_routine.py` 的定时通道。放进这里会让它每天被日频决策轮再跑一次。
    """
    return [p for p in load_personas()
            if p["kind"] == "llm" and p["enabled"] and p.get("league_eligible", True)]


def random_personas() -> list[dict]:
    """只取启用中的随机对照组（排行的方法论基线，决策 1 选 B）。"""
    return [p for p in load_personas() if p["kind"] == "random" and p["enabled"]]


def sync_to_store() -> int:
    """把 JSON 定义幂等同步到 `ai_personas` 表（供 SQL JOIN 与健康检查读数）。
    ⚠️ 只 upsert JSON 里有的；DB 里多出的行**不删**（用户可能手工停用过）。"""
    from db import store
    n = 0
    for p in load_personas():
        if store.upsert_ai_persona(p):
            n += 1
    return n
