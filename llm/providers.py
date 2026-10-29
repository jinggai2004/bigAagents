"""按模型名构建 OpenAI 客户端（多厂商接入的执行层）。

本模块是 `llm/deepseek_client._get_client_for_model` 的实际实现 —— 后者只做转发，
以保持**函数名、签名与 lru_cache 语义完全不变**，从而让所有调用方
（`core/agents.py` 9 处、`core/chat_agent.py`、`core/review_agent.py`、
`core/nl_screener.py`、`data/market_opinions.py`）零改动。

厂商清单来自 `config/providers.py`（JSON 注册表），所以"接一家新厂商"是
改配置而不是改代码。

与改造前的语义对照（**逐条保留**）
──────────────────────────────────
    旧                                      新
    1 "qwen-" 前缀 → DashScope           →  注册表解析（内置默认下结果完全相同）
    2 其余 → DeepSeek                    →  未命中则 default_provider 兜底（默认 DeepSeek）
    3 目标厂密钥缺失 → 回退 DeepSeek      →  回退 default_provider（内置默认下即 DeepSeek）
    4 一个可用密钥都没有 → ValueError      →  同样的 ValueError（文案已含厂商名与环境变量名）

为什么要保留第 3 条而不是直接报错：注册表里可能同时存在"已配置"和"尚未配置密钥"
的厂商。用户在 UI 里把默认模型填成还没配密钥的那家时，**继续能跑**比
"整个选股链路报错停摆"重要得多；日志会明确写出回退事实。
"""

import logging
import os

import openai

from config import providers as cfg_providers

_logger = logging.getLogger(__name__)


# 请求级超时与重试（可用环境变量覆盖）。
# ⚠️ 原先**一个都没设** → 落到 openai SDK 默认的 600s、max_retries=2，
#    单次调用最坏挂 **30 分钟**。实测（2026-10-08 重跑 603127 深度分析）：
#    上游限流（日志里有 `rate limited`）之后，bull_researcher 那次调用
#    **10 分钟没有返回**，整条分析链卡死；而 llm 日志只有一行"调用 qwen-plus"，
#    从外部完全看不出是"在等"还是"死锁"。
#    显式超时至少让失败**可见**：超时 → 抛异常 → 被节点 except 记进 errors，
#    而不是无声地挂住整条链。
#    取 180s：实测正常单次调用 10~30s（max_tokens 2500），180s 足够宽松，
#    又不会让一次网络半开把整轮分析拖过 3 分钟。
LLM_TIMEOUT_SECONDS = float(os.getenv("BIGA_LLM_TIMEOUT", "180"))
LLM_MAX_RETRIES = int(os.getenv("BIGA_LLM_MAX_RETRIES", "1"))


def _openai_client(api_key: str, base_url: str) -> openai.OpenAI:
    """统一的客户端构造入口 —— 超时/重试必须**所有分支一致**。

    三个 return 点各写一遍 `openai.OpenAI(...)` 时，
    加超时很容易漏掉一两个（回退分支尤其容易忘），于是"只有主路径有超时"。
    """
    return openai.OpenAI(api_key=api_key, base_url=base_url,
                         timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES)


def get_client_for_model(model: str) -> openai.OpenAI:
    """根据模型名返回对应的 OpenAI 客户端（厂商由注册表决定）。

    调用方拿到的永远是"能用的客户端"，不保证它真的属于该模型所属厂商——
    回退发生时日志里有明确记录（见 `_fallback_client`）。
    """
    provider = cfg_providers.provider_for_model(model)
    if provider is None:
        raise ValueError(
            "LLM 注册表里没有任何可用厂商（providers 为空），无法调用 LLM。"
            "请在「系统配置 → 模型接入」里添加一个厂商。"
        )

    api_key = cfg_providers.provider_api_key(provider)
    base_url = cfg_providers.provider_base_url(provider)
    if api_key and base_url:
        return _openai_client(api_key=api_key, base_url=base_url)

    # ── 该厂商还不完整：回退到默认厂商（语义同改造前的"qwen 缺 key 回退 DeepSeek"）──
    fallback = cfg_providers.default_provider()
    if fallback and fallback["id"] != provider["id"]:
        fb_key = cfg_providers.provider_api_key(fallback)
        fb_url = cfg_providers.provider_base_url(fallback)
        if fb_key and fb_url:
            _logger.warning(
                "%s 尚未配置%s，模型 %s 的请求回退到 %s",
                provider["name"],
                "API Key" if not api_key else "Base URL",
                model, fallback["name"],
            )
            return _openai_client(api_key=fb_key, base_url=fb_url)

    # ── 全都没有：在**建连前**报错，替代运行时的 AuthenticationError / APIConnectionError ──
    missing = []
    if not api_key:
        env = provider.get("api_key_env") or f"{provider['id'].upper()}_API_KEY"
        missing.append(f"API Key（.env 里的 {env}）")
    if not base_url:
        env = provider.get("base_url_env")
        missing.append(f"Base URL（{env or 'base_url 字段'}）")
    raise ValueError(
        f"厂商「{provider['name']}」缺少 {' 和 '.join(missing)}，"
        "且没有其他可用厂商，无法调用 LLM。请在「系统配置 → 模型接入」或 .env 中补齐。"
    )


def provider_status() -> list:
    """当前各厂商的可用状态快照（UI 展示 / 连通性诊断用，不发起任何网络请求）。"""
    rows = []
    for p in cfg_providers.get_providers():
        rows.append({
            "id": p["id"],
            "name": p["name"],
            "base_url": cfg_providers.provider_base_url(p),
            "api_key_env": p.get("api_key_env") or "",
            "ready": cfg_providers.provider_ready(p),
            "enabled": p.get("enabled", True),
            "is_default": p["id"] == cfg_providers.default_provider_id(),
            "models": list(p["models"]),
            "match": list(p["match"]),
        })
    return rows
