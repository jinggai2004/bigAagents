"""统一配置管理，从 .env 读取环境变量（含启动校验）"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 始终从项目根目录加载 .env，不受运行目录影响
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# 打包成 exe 后 `__file__` 在只读包内（`_internal/config/settings.py`），
# .env **没有**（也**不该**）打进包 ⇒ 上面那行加载不到任何东西，所有 Key 为空，
# `Config.validate()` 直接抛 ValueError，凡是 import `config.settings` 的端点
# （`/api/chat/*`、`/api/intel/news`…）全部 500。
# 所以打包态**额外**从 exe 同级目录读一次 .env：密钥留在包外，用户可自行替换。
# 只在 frozen 时生效 ⇒ 源码运行/测试行为一字不变。
if getattr(sys, "frozen", False):
    _ENV_BESIDE_EXE = Path(sys.executable).resolve().parent / ".env"
    if _ENV_BESIDE_EXE.is_file():
        load_dotenv(_ENV_BESIDE_EXE, override=False)


def _any_llm_configured() -> bool:
    """是否至少有一家 LLM 厂商可用。

    2026-09-29：厂商清单外置为 `config/llm_providers.json`（见 config/providers.py）
    之后，启动校验不能再只认 DeepSeek / Qwen 两家硬编码 —— 用户完全可能
    只配 Kimi、GLM 或本地 Ollama。注册表不可用时回退到"内置两家"的判定，
    与改造前完全一致。
    """
    try:
        from config.providers import any_provider_configured
        return any_provider_configured()
    except Exception:                                   # noqa: BLE001
        return bool(Config.DEEPSEEK_API_KEY or Config.QWEN_API_KEY)


class Config:
    """全局配置 - 启动时通过 validate_config() 自动校验关键参数"""
    
    def __init__(self):
        self.validate()
    
    def validate(self):
        """A3：启动时校验关键参数（仓位、止损等），失败时快速报错"""
        errors = []
        
        if not _any_llm_configured():
            errors.append("没有任何可用的 LLM 厂商：DeepSeek / Qwen 及"
                          "config/llm_providers.json 中自定义厂商的 API Key 均为空，"
                          "LLM 调用将失败")
        
        if self.QWEN_API_KEY and not self.QWEN_BASE_URL:
            errors.append("QWEN_BASE_URL 为空但 QWEN_API_KEY 已配置")
        
        if errors:
            raise ValueError(
                "配置校验失败，请检查 .env 文件:\n  " + "\n  ".join(errors)
            )
        return self

    # DeepSeek API
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    DEFAULT_MODEL_NAME: str = os.getenv("DEFAULT_MODEL_NAME", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))

    # Qwen (DashScope) API
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    QWEN_BASE_URL: str = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    # ⚠️ 2026-10-13 实测：本账号下 qwen-plus 报 403「Free quota exhausted」、
    #    qwen-turbo 报 404「Model not exist」，均不可用；qwen-max 连通正常。
    #    变量名 PLUS/TURBO 指"模型档位"（深层/快速）而非具体型号，是历史遗留，
    #    保留以兼容既有 .env 与部署；要换模型改 .env 的值即可，不必改这里。
    QWEN_PLUS_MODEL: str = os.getenv("QWEN_PLUS_MODEL", "qwen-max")
    QWEN_TURBO_MODEL: str = os.getenv("QWEN_TURBO_MODEL", "qwen-max")

    # Tushare
    TUSHARE_TOKEN: str = os.getenv("TUSHARE_TOKEN", "")

    # 通知
    DINGTALK_WEBHOOK: str = os.getenv("DINGTALK_WEBHOOK", "")
    FEISHU_WEBHOOK: str = os.getenv("FEISHU_WEBHOOK", "")

    # LLM 日预算（元）。2026-10-23 新增。
    # 实测日均约 ¥1.4（见 output/LLM每日成本评估与护栏_2026-10-23.md），
    # 5 元 = 日均的 3.5 倍 ⇒ 正常日不会误伤，批量重跑/循环重试才会撞到。
    # **0 或负数 = 关闭闸门**（不是"预算为零就全拦"，避免部署时忘了配就全线停摆）。
    # ⚠️ 运行时用 getattr 读（不是默认参数），否则改 .env 后热重载无效。
    LLM_DAILY_BUDGET_CNY: float = float(os.getenv("LLM_DAILY_BUDGET_CNY", "5.0") or 5.0)


def validate_config() -> list[str]:
    """
    校验关键配置参数（启动时调用）。

    Returns:
        警告列表，空列表表示配置正常
    """
    warnings = []

    if not _any_llm_configured():
        warnings.append("⚠️  没有任何可用的 LLM 厂商（DeepSeek / Qwen / 自定义厂商的 Key 均为空），"
                        "LLM 调用将失败")

    if config.QWEN_API_KEY and not config.QWEN_BASE_URL:
        warnings.append("⚠️  QWEN_BASE_URL 为空但 QWEN_API_KEY 已配置，Qwen 路由可能失败")

    if not warnings:
        # C1 修复：避免模块导入时无条件 print，改用 logging
        import logging
        logging.getLogger("config").info("配置校验通过")

    return warnings


config = Config()
validate_config()


# .env 字段名 → 默认值（用于热重载）
_ENV_FIELDS = (
    ("DEEPSEEK_API_KEY", ""),
    ("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    ("DEEPSEEK_MODEL", "deepseek-chat"),
    ("QWEN_API_KEY", ""),
    ("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ("QWEN_PLUS_MODEL", "qwen-max"),
    ("QWEN_TURBO_MODEL", "qwen-max"),
    ("TUSHARE_TOKEN", ""),
    ("DINGTALK_WEBHOOK", ""),
    ("FEISHU_WEBHOOK", ""),
)


def reload_config() -> Config:
    """重新从 .env 载入配置（在线更新密钥后立即生效，无需重启进程）。

    Config 的字段在模块导入时由 os.getenv 求值并固化为类属性，编辑 .env 后不会
    自动更新（UI 上"保存后需重启"正是这个原因）。本函数以 override=True 重新
    载入 .env，并把新值写回 config 实例；调用方随后应清空 LLM 客户端缓存
    （llm.deepseek_client.reset_client_cache），否则旧的 OpenAI 客户端仍被复用。
    """
    load_dotenv(_PROJECT_ROOT / ".env", override=True)
    for field_name, default in _ENV_FIELDS:
        setattr(config, field_name, os.getenv(field_name, default))
    # DEFAULT_MODEL_NAME 依赖 DEEPSEEK_MODEL
    config.DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME") or config.DEEPSEEK_MODEL
    # 数值型配置单独处理：`_ENV_FIELDS` 一律按字符串赋值，直接塞进去会让
    # `LLM_DAILY_BUDGET_CNY` 变成 "5.0"（字符串），比较时抛 TypeError。
    # 坏值退回默认——配置写错不该让整个调度起不来。
    try:
        config.LLM_DAILY_BUDGET_CNY = float(os.getenv("LLM_DAILY_BUDGET_CNY", "5.0") or 5.0)
    except (TypeError, ValueError):
        config.LLM_DAILY_BUDGET_CNY = 5.0
    return config