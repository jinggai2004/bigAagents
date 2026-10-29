"""多厂商 LLM 接入注册表（配置层：唯一职责是解析 / 保存厂商清单）。

为什么要有这一层
────────────────
改造前，厂商路由**硬编码**在 `llm/deepseek_client._get_client_for_model` 里：

    if model.lower().startswith("qwen-"):  → 通义千问 (DashScope)
    else:                                  → DeepSeek

想接第三家（Kimi / GLM / 豆包 / OpenAI / 本地 Ollama / 任何 OpenAI 兼容端点）
只能改代码。本模块把这份清单外置成 JSON 文件，**新增厂商 = 加一条记录**，
不再动任何 Python 代码。

兼容承诺（"原有的不变"）
────────────────────────
注册表文件不存在、或内容不可解析时，返回内置的 DeepSeek + Qwen 两条默认记录，
且 base_url 仍由 `DEEPSEEK_BASE_URL` / `QWEN_BASE_URL` 环境变量覆盖 ——
与改造前的行为逐字一致。任何调用方（`core/agents.py`、`core/chat_agent.py`、
`core/review_agent.py`、`data/market_opinions.py`、`core/nl_screener.py`）
都不需要改动。

provider 字段规范
─────────────────
    id            必填。小写字母/数字/_/-，全局唯一。如 "deepseek"、"kimi"。
    name          显示名。缺省用 id。
    base_url      OpenAI 兼容端点，如 "https://api.moonshot.cn/v1"。
    base_url_env  可选。指向 .env 里的变量名，**优先级高于 base_url**。
                  默认厂商用它接住既有的 DEEPSEEK_BASE_URL / QWEN_BASE_URL。
    api_key_env   推荐。指向 .env 里的变量名，如 "MOONSHOT_API_KEY"。
    api_key       不推荐（明文落在 JSON 里，有被提交的风险）。仅作兜底。
    models        该厂商已知模型名列表（UI 下拉用；精确匹配时也用它）。
    match         通配模式列表，如 ["deepseek-*"]，用于命中列表外的模型名。
    enabled       缺省 True；置 False 后该厂商不参与路由。

路由优先级（`resolve_provider`）
────────────────────────────────
    1. `models` 精确命中（不区分大小写）
    2. `match` 通配命中，按注册表顺序取第一个
    3. 都没命中 → 由 `default_provider` 兜底
默认厂商的 `match` 只有 `["deepseek-*"]`，所以**未知模型名仍走 DeepSeek**，
与改造前一致。

⚠️ 读取函数**绝不写文件**——"只是打开配置页看看"不能变成"悄悄改了用户的配置"。
"""

import copy
import fnmatch
import json
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

# ⚠️ 本模块必须**自足**：与 settings.py 一样自己加载 .env。
# 这不是冗余——只导入 config.providers 而没导入 config.settings 的地方
# （例如 UI 里单独查厂商状态）否则会读到空密钥，把"已配置"误判成"未配置"。
# load_dotenv 默认 override=False，重复调用不会覆盖已有值，因此顺序无关。
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_logger = logging.getLogger(__name__)

# 覆盖注册表路径（测试用；生产环境不设置）。
# 用环境变量而不是函数参数，是为了让"调用方零改动"这条承诺成立。
ENV_REGISTRY_PATH = "BIGA_LLM_PROVIDERS"

# provider id 的合法形式：小写字母/数字开头，最长 32 位。
PROVIDER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

# 内置默认厂商（= 改造前硬编码的那两家）。字段变化时必须同步本文件顶部的规范文档。
DEFAULT_PROVIDERS = [
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "api_key_env": "DEEPSEEK_API_KEY",
        # 2026-10-19：deepseek-chat / deepseek-reasoner 是**废弃别名**
        #（官方文档：2026-07-24 停用），新代码一律写 v4-flash / v4-pro。
        "models": ["deepseek-v4-flash", "deepseek-v4-pro",
                   "deepseek-chat", "deepseek-reasoner"],
        "match": ["deepseek-*"],
        "enabled": True,
    },
    {
        "id": "qwen",
        "name": "通义千问 (DashScope)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "base_url_env": "QWEN_BASE_URL",
        "api_key_env": "QWEN_API_KEY",
        "models": ["qwen-plus", "qwen-turbo", "qwen-max"],
        "match": ["qwen-*"],
        "enabled": True,
    },
]

DEFAULT_REGISTRY = {
    "version": 1,
    "default_provider": "deepseek",
    "providers": DEFAULT_PROVIDERS,
}


# ============================================================
# 路径与缓存
# ============================================================

def registry_path() -> Path:
    """注册表文件路径（默认 `config/llm_providers.json`）。"""
    override = os.getenv(ENV_REGISTRY_PATH, "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "llm_providers.json"


# mtime 缓存：注册表是小文件但可能被 UI 频繁读取（每次 rerun 都读），
# 用 mtime 判断"文件变了没"即可，无需 lru_cache（要能感知外部编辑）。
_CACHE = {"path": None, "mtime": None, "loaded": False, "data": None}


def invalidate() -> None:
    """清空注册表内容缓存。外部直接改了 JSON 文件时通常不需要调用——
    mtime 变了会自动重读；本函数供 `save_registry` 与测试使用。"""
    _CACHE.update(path=None, mtime=None, loaded=False, data=None)


def registry_file_exists() -> bool:
    try:
        return registry_path().exists()
    except OSError:
        return False


def _read_file() -> dict | None:
    """读文件；不存在 / 读不动 / 不是 JSON 对象 → 返回 None（调用方回退内置默认）。"""
    path = registry_path()
    try:
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:                              # noqa: BLE001
        # 刻意只告警不抛：注册表坏掉必须降级到"内置两家"继续能跑，
        # 否则一次手滑的编辑会让整个系统的 LLM 调用全线瘫痪。
        _logger.warning("LLM 注册表读取失败，回退内置默认厂商（%s）: %s", path, e)
        return None
    return raw if isinstance(raw, dict) else None


def _cached_file() -> dict | None:
    path = registry_path()
    try:
        mtime = path.stat().st_mtime_ns if path.exists() else None
    except OSError:
        mtime = None
    if _CACHE["loaded"] and _CACHE["path"] == str(path) and _CACHE["mtime"] == mtime:
        return _CACHE["data"]
    data = _read_file()
    _CACHE.update(path=str(path), mtime=mtime, loaded=True, data=data)
    return data


# ============================================================
# 读取
# ============================================================

def _normalize(providers) -> list:
    """清洗 provider 记录：丢非法/重复 id，补字段，统一类型。"""
    out, seen = [], set()
    for p in providers or []:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "").strip().lower()
        if pid and not PROVIDER_ID_RE.match(pid):
            # 非法 id 会被用来拼 `<ID>_API_KEY` 之类的环境变量名，放进去只会
            # 制造难查的静默故障。丢弃 + 告警，比留着一个用不了的条目好。
            _logger.warning("忽略非法的 LLM 厂商 id（只允许小写字母/数字/_/-，最长 32 位）：%r", pid)
            continue
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append({
            "id": pid,
            "name": str(p.get("name") or pid).strip() or pid,
            "base_url": str(p.get("base_url") or "").strip(),
            "base_url_env": str(p.get("base_url_env") or "").strip(),
            "api_key_env": str(p.get("api_key_env") or "").strip(),
            "api_key": str(p.get("api_key") or "").strip(),
            "models": [str(m).strip() for m in (p.get("models") or []) if str(m).strip()],
            "match": [str(m).strip() for m in (p.get("match") or []) if str(m).strip()],
            "enabled": bool(p.get("enabled", True)),
        })
    return out


def get_registry() -> dict:
    """完整注册表（含 default_provider）。文件缺失/不可用时回退内置默认。"""
    raw = _cached_file()
    if not raw:
        return copy.deepcopy(DEFAULT_REGISTRY)
    providers = _normalize(raw.get("providers"))
    if not providers:
        # 空 providers 的注册表等于"没有任何厂商可用"，比回退内置默认更糟。
        return copy.deepcopy(DEFAULT_REGISTRY)
    ids = [p["id"] for p in providers]
    default_id = str(raw.get("default_provider") or "").strip().lower()
    if default_id not in ids:
        default_id = ids[0]
    return {
        "version": raw.get("version", 1),
        "default_provider": default_id,
        "providers": providers,
    }


def get_providers() -> list:
    """全部 provider（含 enabled=False 的，供 UI 展示）。"""
    return get_registry()["providers"]


def default_provider_id() -> str:
    return get_registry()["default_provider"]


def default_provider() -> dict | None:
    pid = default_provider_id()
    for p in get_providers():
        if p["id"] == pid:
            return p
    providers = get_providers()
    return providers[0] if providers else None


def resolve_provider(model: str) -> dict | None:
    """模型名 → provider（未命中返回 None，由调用方决定是否用默认厂商）。"""
    name = str(model or "").strip()
    if not name:
        return None
    active = [p for p in get_providers() if p.get("enabled", True)]
    low = name.lower()
    # 1) 精确命中已声明的模型名
    for p in active:
        if low in [m.lower() for m in p["models"]]:
            return p
    # 2) 通配命中（按注册表顺序，先到先得）
    for p in active:
        for pattern in p["match"]:
            if fnmatch.fnmatchcase(low, pattern.lower()):
                return p
    return None


def provider_for_model(model: str) -> dict | None:
    """永不返回 None（有 provider 时）：命中 → 它，未命中 → 默认厂商。"""
    return resolve_provider(model) or default_provider()


def provider_api_key(provider: dict) -> str:
    """取密钥：环境变量优先，其次 provider 里的明文（不推荐）。"""
    if not provider:
        return ""
    env_name = provider.get("api_key_env") or ""
    if env_name:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return provider.get("api_key") or ""


def provider_base_url(provider: dict) -> str:
    """取 base_url：<X>_BASE_URL 环境变量优先，其次 provider 的 base_url 字段。

    这一条是"原有的不变"的关键——用户 .env 里已有的 DEEPSEEK_BASE_URL /
    QWEN_BASE_URL 仍然生效，不会被注册表里的默认值顶掉。
    """
    if not provider:
        return ""
    env_name = provider.get("base_url_env") or ""
    if env_name:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return provider.get("base_url") or ""


def provider_ready(provider: dict) -> bool:
    """该厂商是否可用（密钥与地址都有）。"""
    return bool(provider_api_key(provider) and provider_base_url(provider))


def any_provider_configured() -> bool:
    """是否至少有一家可用 —— `config.settings.validate()` 用它放宽启动校验。"""
    return any(provider_ready(p) for p in get_providers() if p.get("enabled", True))


def available_models() -> list:
    """所有已声明模型 + 可用状态（UI 下拉 / 连通性测试用）。"""
    rows = []
    for p in get_providers():
        ready = provider_ready(p)
        for m in p["models"]:
            rows.append({
                "provider_id": p["id"],
                "provider_name": p["name"],
                "model": m,
                "ready": ready,
            })
    return rows


def describe_model(model: str) -> str:
    """'qwen-max' → '通义千问 (DashScope)'；未命中 → '默认厂商名（兜底）'。"""
    hit = resolve_provider(model)
    if hit:
        return hit["name"]
    fallback = default_provider()
    return f"{fallback['name']}（兜底）" if fallback else "无可用厂商"


# ============================================================
# 写入
# ============================================================

def save_registry(data: dict) -> Path:
    """原子写注册表（.tmp + os.replace）。

    ⚠️ Windows 上 `os.replace` 要求目标文件**没有被任何进程打开**，否则 WinError 5。
    Streamlit 热重载、编辑器预览都可能短暂持有句柄，所以退避重试而不是直接失败。
    """
    path = registry_path()
    payload = dict(data or {})
    providers = _normalize(payload.get("providers"))
    if not providers:
        raise ValueError("注册表至少要保留一个厂商，否则 LLM 调用将全部失败")
    ids = [p["id"] for p in providers]
    default_id = str(payload.get("default_provider") or "").strip().lower()
    if default_id not in ids:
        default_id = ids[0]
    payload["version"] = payload.get("version") or 1
    payload["default_provider"] = default_id
    payload["providers"] = providers

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    last_err = None
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            invalidate()
            return path
        except OSError as e:
            last_err = e
            time.sleep(0.1 * (attempt + 1))
    raise OSError(f"写入注册表失败（文件可能被占用）: {last_err}")


def init_registry_file(force: bool = False) -> Path:
    """把"当前生效的注册表"落盘成文件（UI 上的「初始化配置文件」按钮）。

    已存在且 force=False 时直接返回原路径 —— 不能覆盖用户已经改过的配置。
    """
    path = registry_path()
    if path.exists() and not force:
        return path
    return save_registry(get_registry())


def validate_provider_id(pid: str) -> str:
    """校验 provider id，返回规范化后的值；非法时抛 ValueError（UI 表单用）。"""
    value = str(pid or "").strip().lower()
    if not PROVIDER_ID_RE.match(value):
        raise ValueError("厂商 ID 只能用小写字母、数字、下划线或短横线，且以字母/数字开头（最长 32 位）")
    return value
