"""功能级模型路由（方案 C：独立功能映射 JSON）。

为什么要有这一层
────────────────
改造前，"哪个功能用哪个模型"只有三种来源，**没有一层可配置的映射**：

  1. 构造期读 `config.DEFAULT_MODEL_NAME`（agents / nl_screener 等 9 处）；
  2. **导入期固化的模块常量**（`chat_agent.CHAT_MODEL_FAST/DEEP`、`review_agent.REVIEW_MODEL`）
     —— 这才是 2026-10-13 那次"改了 .env 后必须重启进程才生效"的根因；
  3. 调用期显式传（market_opinions / 连通性测试）。

本模块把"功能 → 模型"外置成 `config/llm_feature_models.json`，**与 `llm_providers.json`
并列放在 `config/`**（用户心智一致）。改完由 mtime 缓存感知，**不用重启进程**。

功能键就是既有的 `purpose`（LLM 调用标签）
──────────────────────────────────────────
`purpose` 当初只为可观测性而加（写进 `llm_calls`），但它天然就是"功能标识"。
本层**复用它，不引入第二套键**——否则"配置里的功能"与"日志里的功能"会漂移。

解析优先级（`resolve_model`）
────────────────────────────
    1. 显式传参（调用方当次指定，如连通性测试用户选的模型 / 客户端构造时指定的模型）
    2. 功能映射（本 JSON 里为该功能单独配的模型）
    3. 调用方自带模型（`DeepSeekClient` 的 `self.model`；不传时它本身=全局默认）
    4. 功能默认档位（`FEATURES[feature]["tier"]`：全局/快/深，见下）
    5. 内置默认（`BUILTIN_DEFAULT_MODEL`）

⚠️ 第 3/4 层的存在是为了 **non-mutation（未配置时行为逐字节不变）**：
  - `client.technical_analysis` 等 14 个功能改造前读 `DEFAULT_MODEL_NAME` → tier="global"；
  - `chat.tool_loop` 改造前读 `QWEN_TURBO_MODEL or "qwen-max"` → tier="fast"；
  - `review.daily` 改造前读 `QWEN_PLUS_MODEL or "qwen-max"` → tier="deep"。
若不带这两层而一律回退 `DEFAULT_MODEL_NAME`，则 chat/复盘在"全局默认 ≠ QWEN 档位"
的部署里会**悄悄换模型**——那正是本改造最要避免的。

⚠️ 读取函数**绝不写文件**——"只是打开配置页看看"不能变成"悄悄改了用户的配置"。
"""

import json
import logging
import os
import time
from pathlib import Path

_logger = logging.getLogger(__name__)

# 覆盖映射文件路径（测试用；生产环境不设置）。
# 用环境变量而不是函数参数，与 config/providers.py 保持一致：调用方零改动。
ENV_ROUTING_PATH = "BIGA_LLM_FEATURE_MODELS"

# 兜底模型：全局默认与各档位都取不到时用它（与 settings 的默认一致）。
BUILTIN_DEFAULT_MODEL = "deepseek-chat"

# 档位 → 取哪个配置项。改造前各功能实际用的"档位"登记在此，保证 non-mutation。
#   global → config.DEFAULT_MODEL_NAME（多数节点）
#   fast   → config.QWEN_TURBO_MODEL（AI 助手主循环 / 市场观点联网搜索）
#   deep   → config.QWEN_PLUS_MODEL（每日复盘 / chat.deep 扩展位）
TIER_GLOBAL, TIER_FAST, TIER_DEEP = "global", "fast", "deep"

# ------------------------------------------------------------------
# 功能注册表：**唯一来源**
# ------------------------------------------------------------------
# 每个功能键对应一个 `purpose` 字面量（见各调用点）。
#   label    中文标签（UI 展示）
#   group    分组（UI 分组与"未声明功能"排查）
#   tier     改造前该功能实际使用的"模型档位"，参与解析第 4 层（见模块头）
#   desc     一句话说明（UI 悬浮提示）
#   reserved 预留位：**当前没有任何调用点**，登记只为保留用户可配的档位
#            （`chat.deep` 由原 `CHAT_MODEL_DEEP` 死常量转来）
FEATURES: dict[str, dict] = {
    # ── 深度分析（单票报告生成）──
    "client.technical_analysis": {
        "label": "技术面分析", "group": "深度分析", "tier": TIER_GLOBAL,
        "desc": "深度分析节点：K线/均线/指标的技术面报告"},
    "client.fundamental_analysis": {
        "label": "基本面分析", "group": "深度分析", "tier": TIER_GLOBAL,
        "desc": "深度分析节点：财务三表与三表外关键指标"},
    # 复盘归因（2026-10-20）：输入是"当时的报告 + 代码算好的事实 + 候选原因码"，
    # 输出是受限的结构化 JSON（原因码 + 证据字段 + 改进项），属于"依数据做判断"。
    "analysis.review": {
        "label": "分析复盘归因", "group": "深度分析", "tier": TIER_GLOBAL,
        "desc": "对已回填的分析做归因：在代码判定的候选原因里选主因并给改进项"},
    # ── 多智能体（辩论与决策链）──
    "agents.news": {
        "label": "舆情分析", "group": "多智能体", "tier": TIER_GLOBAL,
        "desc": "深度分析节点：新闻舆情与情绪"},
    "agents.bull": {
        "label": "多头研究员", "group": "多智能体", "tier": TIER_GLOBAL,
        "desc": "多空辩论：多头论据"},
    "agents.bear": {
        "label": "空头研究员", "group": "多智能体", "tier": TIER_GLOBAL,
        "desc": "多空辩论：空头论据"},
    "agents.research_manager": {
        "label": "研究汇总", "group": "多智能体", "tier": TIER_GLOBAL,
        "desc": "汇总三份分析报告 + 多空辩论"},
    "agents.trader": {
        "label": "交易员", "group": "多智能体", "tier": TIER_GLOBAL,
        "desc": "输出结构化 trade_plan（价位/仓位）"},
    "agents.risk_manager": {
        "label": "风控经理", "group": "多智能体", "tier": TIER_GLOBAL,
        "desc": "按风控规则审核 trade_plan"},
    "agents.portfolio_manager": {
        "label": "终审组合经理", "group": "多智能体", "tier": TIER_GLOBAL,
        "desc": "最终决策（看多可执行/看空驳回/观望等待）"},
    # ── AI 助手（工具循环对话）──
    "chat.tool_loop": {
        "label": "AI 助手主循环", "group": "AI 助手", "tier": TIER_FAST,
        "desc": "工具循环每一轮的对话调用"},
    "chat.tool_loop_summary": {
        "label": "AI 助手超限汇总", "group": "AI 助手", "tier": TIER_FAST,
        "desc": "达到最大工具轮数后的强制汇总"},
    "chat.deep": {
        "label": "AI 助手深度档位（扩展位）", "group": "AI 助手", "tier": TIER_DEEP,
        "desc": "预留位：原 CHAT_MODEL_DEEP 死常量转来，当前无调用点",
        "reserved": True},
    # ── 每日复盘 ──
    "review.daily": {
        "label": "每日复盘", "group": "每日复盘", "tier": TIER_DEEP,
        "desc": "收盘后生成 Markdown 复盘报告"},
    # ── 新闻 ──
    "news_industry.review": {
        "label": "新闻行业评审", "group": "新闻", "tier": TIER_GLOBAL,
        "desc": "24h 快讯按行业判利好/利空/中性"},
    "ui.zhihu_summary": {
        "label": "热点知乎汇总", "group": "新闻", "tier": TIER_GLOBAL,
        "desc": "热点新闻页：知乎内容要点归纳（仅展示，不参与分析）"},
    # ── 选股 ──
    "nl_screener.parse": {
        "label": "NL 选股解析", "group": "选股", "tier": TIER_GLOBAL,
        "desc": "自然语言选股条件解析（temperature=0，结果须可复现）"},
    # ── 市场观点 ──
    "market_opinions": {
        "label": "市场观点", "group": "市场观点", "tier": TIER_FAST,
        "desc": "复盘参考：Qwen 联网搜索 A股当日讨论观点"},
    # 情绪参照判定（2026-10-23）：对**真实发言样本**做情绪/立场判定，
    # 代码层强制校验 stance 枚举、evidence_ids 白名单、quality 降置信度。
    # ⚠️ tier 走 **TIER_GLOBAL**（跟随全局默认模型）：新登记的功能默认不碰快档
    #    —— 快/深档是**历史迁移**下来的特例（LEGACY_FAST/LEGACY_DEEP 白名单），
    #    新功能默认用全局模型才符合"换模型要确定性依据"的纪律。
    #    想换模型可在配置页按功能单独指定。
    "sentiment.judge": {
        "label": "情绪参照判定", "group": "市场观点", "tier": TIER_GLOBAL,
        "desc": "对发言样本判情绪立场（证据 ID 必须落在样本白名单内）"},
    # 逐条情绪标注（2026-10-23）：模型直读原文，逐条给 bull/bear/neutral/unscored
    # + 可在原文逐字命中的引用。**数字由代码从标签算，模型不报统计**。
    # ⚠️ 这里选 flash 有明确依据：本功能的失效模式是"没读原文就编立场"，
    #    而这一条由**确定性校验器**（引用子串命中率 <50% ⇒ 整块作废）拦得住，
    #    不依赖模型自评。若 `quote_hit_rate` 长期低于 80%，再按配置页升档。
    "sentiment.label": {
        "label": "情绪逐条标注", "group": "市场观点", "tier": TIER_GLOBAL,
        "desc": "逐条判立场并附原文引用（引用须逐字命中，否则整块作废）"},
    # ── 系统 ──
    "config.test": {
        "label": "连通性测试", "group": "系统", "tier": TIER_GLOBAL,
        "desc": "系统配置页「LLM 连通性测试」按钮（模型由用户当场选）"},

    # ── AI 交易员（按人设自动展开的**族键**）──
    # ⚠️ 为什么是带 `*` 的族键，而不是逐个写 `ai.ai_ashen.routine`：
    #    purpose 是运行时拼的 `f"ai.{persona}.{kind}"`，persona 来自
    #    `config/ai_personas.json`（会新增/改名），kind 有 decision/wake/routine/chat。
    #    逐个硬编码 ⇒ 每加一个人设就得改注册表，忘改的后果不是报错，而是
    #    **静默落到全局默认模型**：2026-10-28 之前 ai_tech/ai_value/ai_risk/ai_ashen
    #    四族每天在花 token，而配置页上一行都看不到（`feature_usage` 只列 FEATURES，
    #    未登记的落到"⚠️ 未登记"组里，且无法按功能指定模型）。
    #    换 `DEFAULT_MODEL_NAME` 时，这四个族会**无声地一起换模型**——没人会注意到。
    # ⚠️ 匹配规则见 `match_feature()`：精确键优先，其次按"具体段数多的族"。
    "ai.*.decision": {
        "label": "AI 交易员决策轮", "group": "AI 交易员", "tier": TIER_GLOBAL,
        "desc": "人设交易员每轮决策（选票/仓位/理由，输出受限 JSON）；"
                "当前覆盖 ai_tech / ai_value / ai_risk 三个每日决策人设"},
    # 条件命中后的反应轮：输入是"条件表达式 + 命中事实"，输出极短（一两句判断）。
    # 用快档没有风险——它不做价位计算，只决定"要不要把这条命中升级成一次决策"。
    "ai.*.wake": {
        "label": "AI 交易员唤醒轮", "group": "AI 交易员", "tier": TIER_GLOBAL,
        "desc": "挂着的等待条件被命中后的一次反应（决定是否升级为决策轮）"},
    # 阿深的值班轮：一天 6 个班次，每轮拿着自己的账本/池子/今日判断轨迹做判断。
    # 失效模式是"复读上一轮 + 空口承诺不挂条件"，靠 prompt 摊开事实解决（见
    # `ai_routine.build_prompt`），不是靠换更贵的模型。
    "ai.*.routine": {
        "label": "AI 值班轮", "group": "AI 交易员", "tier": TIER_GLOBAL,
        "desc": "按班次值守的轮次（阿深：09:40~14:40 共 6 轮，含当日判断轨迹）"},
    # 用户在场的一问一答：这一个是**唯一有人盯着看**的调用点，追问的是
    # "你为什么这么判断"。频次极低（一次 ~700pt），所以用贵一档的模型。
    "ai.*.chat": {
        "label": "AI 面对面问答", "group": "AI 交易员", "tier": TIER_GLOBAL,
        "desc": "网页房间里用户直接问交易员本人（第一人称、只读、不落动作）"},

    # ── 开发验证（不是业务功能，但**同样在花 token**）──
    # 2026-10-28：配置页上曾长期挂着三行「⚠️ 未登记功能」，全部来自我自己的
    # 验证脚本：`dev.verify.phase`（验时段剧本，仍在用）、
    # `dev.verify.m1_prompt`（验 prompt 里有没有摊开判断轨迹）、
    # `verify.model`（验模型连通性）。它们只有两个毛病：
    #   1. 不进注册表 ⇒ 配置页上算不出、也改不了档；
    #   2. 混进「⚠️ 未登记」组 ⇒ **真正的盲区（新增业务功能忘登记）会淹在里面**，
    #      而那才是这个组存在的唯一理由。
    # 所以单列一组：开发开销与业务开销在页面上分得开，"未登记"回到"0 行 = 没有盲区"。
    # ⚠️ 别把这一组当垃圾桶：它只装**只读的验证脚本**（不落单、不动账本）。
    #    任何会改数据的调用都不属于这里，必须登记到对应业务分组。
    "dev.verify.*": {
        "label": "开发验证脚本", "group": "开发验证", "tier": TIER_GLOBAL,
        "desc": "scripts/verify_*.py 跑真模型做实证（只读：不落库、不下单、不动账本）"},
    # 历史遗留：键形是 `<域>.<动作>` 两段，族键 `dev.verify.*` 是三段，覆盖不到。
    # 来源的临时脚本（temp/ 下的连通性验证）已删除，**代码里不再有任何调用点**，
    # 只有 `llm_calls` 里 24h 窗口内的旧记录还会被读到 —— 窗口滚过即自然消失。
    # 标 `reserved` 而不是删掉：删了它又会掉回「⚠️ 未登记」，那正是本次要收口的东西。
    "verify.model": {
        "label": "开发验证（历史遗留）", "group": "开发验证", "tier": TIER_GLOBAL,
        "desc": "已下线临时脚本留下的键（模型连通性验证），当前无调用点；"
                "仅用于让历史用量归组，24h 窗口滚过后不再出现",
        "reserved": True},
}


def _glob_match(pattern: str, key: str) -> bool:
    """段级通配：`ai.*.routine` 匹配 `ai.ai_ashen.routine`。

    只在 `.` 边界上通配（不做字符级 glob）——功能键是点分段标识符，
    字符级通配会让 `ai.*.chat` 意外吃掉 `ai.ai_xxx.chat_extra` 这种键。
    """
    p, k = pattern.split("."), key.split(".")
    if len(p) != len(k):
        return False
    return all(a == "*" or a == b for a, b in zip(p, k))


def match_feature(feature: str) -> str | None:
    """功能键 → 注册表里**命中的那条键**（精确优先，其次具体段最多的族键）。

    返回 `None` 表示完全未登记。调用方拿它去查 `FEATURES` 元信息、
    或去映射表里取"这条键配了什么模型"。

    为什么"具体段多者优先"：将来若要写 `ai.ai_ashen.*` 给某一个交易员特调，
    它必须压过 `ai.*.routine`。按具体段数排序即可，无需额外优先级字段。
    """
    key = str(feature or "").strip()
    if not key:
        return None
    if key in FEATURES:
        return key
    best: tuple[int, str] | None = None
    for pat in FEATURES:
        if "*" not in pat or not _glob_match(pat, key):
            continue
        spec = sum(1 for seg in pat.split(".") if seg != "*")
        if best is None or spec > best[0]:
            best = (spec, pat)
    return best[1] if best else None


def feature_meta(feature: str) -> dict:
    """功能键（可含族）的元信息；未登记返回空字典。"""
    hit = match_feature(feature)
    return (FEATURES.get(hit) or {}) if hit else {}


def reserved_features() -> frozenset:
    """预留位功能键（登记了但**当前无调用点**）。结构性守护据此放宽"反向"检查。"""
    return frozenset(k for k, v in FEATURES.items() if v.get("reserved"))


def feature_label(feature: str) -> str:
    """功能键 → 中文标签（未登记返回原键，便于"未登记功能"在 UI 上可见）。

    族键会把命中的那一段（人设名）后缀出来：`ai.ai_ashen.routine` →
    「AI 值班轮（ai_ashen）」——否则配置页上四个交易员全叫「AI 值班轮」，
    谁在花钱、谁要调档完全分不清。
    """
    key = str(feature or "")
    meta = feature_meta(key)
    if not meta:
        return key
    hit = match_feature(key) or key
    if "*" in hit and key != hit:
        pairs = list(zip(key.split("."), hit.split(".")))
        star = [k for k, h in pairs if h == "*"]
        if star:
            return f"{meta['label']}（{star[0]}）"
    return meta["label"]


# ============================================================
# 路径与缓存（照抄 config/providers.py 的 mtime 缓存模式）
# ============================================================

def feature_models_path() -> Path:
    """映射文件路径（默认 `config/llm_feature_models.json`）。"""
    override = os.getenv(ENV_ROUTING_PATH, "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "llm_feature_models.json"


# mtime 缓存：映射是小文件但会被 UI 每次 rerun 读取；用 mtime 判断"文件变了没"，
# 既能感知外部直接编辑，又不必 lru_cache（要能热加载）。
_CACHE = {"path": None, "mtime": None, "loaded": False, "data": None}


def invalidate() -> None:
    """清空映射内容缓存。外部直接改 JSON 通常不必调用（mtime 变了会自动重读）；
    本函数供 `save_feature_models` 与测试使用。"""
    _CACHE.update(path=None, mtime=None, loaded=False, data=None)


def feature_models_file_exists() -> bool:
    try:
        return feature_models_path().exists()
    except OSError:
        return False


def _sanitize(raw) -> dict:
    """只保留"键与值都是非空字符串"的映射项；其余丢弃并告警。

    空值/非字符串值必须丢弃：`{"agents.bull": ""}` 或 `{"agents.bull": 123}` 的语义是
    "配置有误"，若当成有效覆盖会让该功能解析到一个空/无意义的模型名 → 建连时报一个
    莫名其妙的错。丢弃 = 回退默认，行为可预期。
    """
    out: dict[str, str] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        k = str(key or "").strip()
        if not k:
            continue
        if not isinstance(value, str):
            _logger.warning("忽略功能模型映射中的非字符串值：%s=%r（该功能将跟随默认）",
                            k, value)
            continue
        v = value.strip()
        if not v:
            _logger.warning("忽略功能模型映射中的空值：%s（该功能将跟随默认）", k)
            continue
        out[k] = v
    return out


def _read_file() -> dict | None:
    """读文件；不存在 / 读不动 / 不是 JSON 对象 → 返回 None（调用方回退"未配置"）。"""
    path = feature_models_path()
    try:
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:                              # noqa: BLE001
        # 刻意只告警不抛：映射坏掉必须降级为"全部跟随默认"继续能跑，
        # 否则一次手滑的编辑会让所有 LLM 调用按错模型跑（或直接报错）。
        _logger.warning("功能模型映射读取失败，回退默认（%s）: %s", path, e)
        return None
    return raw if isinstance(raw, dict) else None


def _cached_file() -> dict | None:
    path = feature_models_path()
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

def get_feature_models() -> dict:
    """全部功能映射 `{purpose: model}`。文件缺失/损坏 → 空字典（= 全部跟随默认）。"""
    raw = _cached_file()
    if not raw:
        return {}
    # 兼容两种写法：{"version","models":{...}}（本模块写出的）与裸 {purpose: model}。
    body = raw.get("models")
    if not isinstance(body, dict):
        body = {k: v for k, v in raw.items() if k not in ("version", "_comment")}
    return _sanitize(body)


def feature_model(feature: str) -> str | None:
    """该功能**单独配的**模型；未配置返回 None（**不做任何回退**）。

    两级查找：先按完整键（`ai.ai_ashen.routine`）精确取，取不到再按命中的族键
    （`ai.*.routine`）取。于是"给某一个交易员特调"与"给整族定档"可以共存。
    """
    key = str(feature or "").strip()
    if not key:
        return None
    mapping = get_feature_models()
    exact = mapping.get(key)
    if exact:
        return exact
    hit = match_feature(key)
    if hit and hit != key:
        return mapping.get(hit) or None
    return None


def _tier_model(tier: str) -> str:
    """把"档位"解析成具体模型名（运行时读 config，与 reload_config 一致）。"""
    from config.settings import config
    if tier == TIER_FAST:
        return str(config.QWEN_TURBO_MODEL or "qwen-max").strip()
    if tier == TIER_DEEP:
        return str(config.QWEN_PLUS_MODEL or "qwen-max").strip()
    return str(config.DEFAULT_MODEL_NAME or BUILTIN_DEFAULT_MODEL).strip()


def resolve_model(feature: str, explicit: str = None, default: str = None) -> str:
    """解析某功能实际使用的模型。

    Args:
        feature:  功能键（= `purpose`）。
        explicit: 调用方当次显式指定的模型（优先级最高）。
        default:  调用方自带模型（如 `DeepSeekClient` 的 `self.model`）。
                  仅在前两层都没命中时使用——**这是 non-mutation 的关键**：
                  客户端构造时指定的模型不能被"功能未配置"顶掉。

    解析优先级见模块头（显式 > 功能映射 > 调用方自带 > 功能档位 > 内置默认）。
    **永不返回空串**：任何一层命中即返回；全空则回退内置默认。
    """
    exp = str(explicit or "").strip()
    if exp:
        return exp
    key = str(feature or "").strip()
    mapping = get_feature_models()
    # ① **精确键**：用户对"这一个功能"的明确指定，粒度最细 ⇒ 压过 persona 绑定。
    #    例：单独把 `ai.ai_ashen.chat` 配成 pro，就应该真的用 pro。
    exact = mapping.get(key)
    if exact:
        return exact
    # ② 调用方自带（persona 绑定 / `DeepSeekClient(model=X)`）。
    #    ⚠️ 必须排在**族键之前**：`ai_personas.json` 里的 `model` 是"这个 AI 用哪个
    #       模型"，是联赛的设定（各 AI 可以不同），不是可被功能默认值顶掉的弱约束。
    #       族键只是"没绑定时用哪个"，若让它赢，给 `ai.*.chat` 填一个值就会把
    #       所有 persona 的模型一起换掉 —— 而页面上看不出是谁改的。
    dft = str(default or "").strip()
    if dft:
        return dft
    # ③ 族键兜底：覆盖"运行时拼出来、且没绑定模型"的那些 purpose。
    hit = match_feature(key)
    if hit and hit != key:
        mapped = mapping.get(hit)
        if mapped:
            return mapped
    # ④ 功能档位。族键（`ai.*.routine`）也要能查到 tier，
    #    否则等哪天给某个族改成 fast/deep 会**静默失效**。
    tier = (FEATURES.get(hit) or {}).get("tier") if hit else None
    return _tier_model(tier or TIER_GLOBAL) or BUILTIN_DEFAULT_MODEL


# ============================================================
# 写入（原子写 + os.replace 退避重试；照抄 save_registry）
# ============================================================

def save_feature_models(mapping: dict) -> bool:
    """原子写映射文件（.tmp + os.replace）。

    ⚠️ 与 `providers.save_registry` 的差异（有意）：注册表"至少要有一家厂商"属硬约束，
    写失败必须抛；而功能映射是**可选**配置，写失败只应显式告知调用方（返回 False + 告警），
    不能把一次 UI 保存操作变成异常栈。**绝不静默**：失败一定返回 False 且留日志。

    ⚠️ Windows 上 `os.replace` 要求目标文件**没有被任何进程打开**，否则 WinError 5。
    Streamlit 热重载、编辑器预览都可能短暂持有句柄，所以退避重试而不是直接失败。
    """
    if not isinstance(mapping, dict):
        _logger.warning("功能模型映射不是字典，已忽略写入：%r", type(mapping))
        return False
    # 版本号**原样保留**（只在缺失时补 1）。
    # ⚠️ 曾经写死 `{"version": 1}`：而当前文件已经是 version 4，
    #    用户只要在配置页点一次保存就会把版本号"降级"回去——下一个人排障时
    #    看到 v1 会以为这是老格式，而 `_read_file` 其实根本不看它。
    #    版本号本身不承担任何判断（"版本号是假安全"），所以既不做迁移也不校验，
    #    但绝不能自己往回调。
    raw = _cached_file() or {}
    try:
        ver = int(raw.get("version") or 1)
    except (TypeError, ValueError):
        ver = 1
    payload = {"version": max(ver, 1), "models": _sanitize(mapping)}

    path = feature_models_path()
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        _logger.warning("功能模型映射写入临时文件失败：%s", e)
        return False

    last_err = None
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            invalidate()
            return True
        except OSError as e:
            last_err = e
            time.sleep(0.1 * (attempt + 1))
    _logger.warning("功能模型映射写入失败（文件可能被占用，已重试 5 次）：%s", last_err)
    try:
        tmp.unlink()                                   # 清理残留 .tmp，best-effort
    except OSError:
        pass
    return False


def set_feature_model(feature: str, model: str) -> bool:
    """为某功能指定模型（model 为空 = 删除该覆盖，等同跟随默认）。"""
    key = str(feature or "").strip()
    if not key:
        _logger.warning("功能键为空，忽略设置功能模型")
        return False
    current = dict(get_feature_models())
    value = str(model or "").strip()
    if value:
        current[key] = value
    else:
        current.pop(key, None)
    return save_feature_models(current)


def reset_feature_model(feature: str) -> bool:
    """清除某功能的覆盖（恢复"跟随默认"）。功能键不存在时也返回 True（幂等）。"""
    return set_feature_model(feature, "")


def init_feature_models_file(force: bool = False) -> bool:
    """把"当前生效的映射"落盘（UI 的「初始化配置文件」按钮）。

    已存在且 force=False 时**不覆盖**用户已改过的配置，直接返回 True。
    """
    path = feature_models_path()
    if path.exists() and not force:
        return True
    return save_feature_models(get_feature_models())


# ============================================================
# 用量（成本面板用；读 llm_calls，按功能汇总）
# ============================================================

def feature_usage(hours: int = 24) -> list[dict]:
    """近 N 小时按功能汇总 `llm_calls`：次数 / 失败 / tokens。

    - 已登记功能**一定出现**（无调用则计 0，便于"这个功能到底跑没跑"一眼可见）；
    - 出现了但**未登记**的 purpose 单列，标签写明"未登记功能"（R9：盲区可见而非藏起来）。
    读库失败 → 返回 []（面板显示"不可用"，不抛异常）。
    """
    from datetime import datetime, timedelta
    try:
        from db import store
        store.init_store()          # 与 llm_call_stats 同口径：先确保表存在再查询
        conn = store.get_connection()
        try:
            since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
            rows = conn.execute(
                """SELECT purpose, COUNT(*),
                          SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END),
                          SUM(COALESCE(total_tokens, 0))
                   FROM llm_calls WHERE created_at >= ?
                   GROUP BY purpose ORDER BY COUNT(*) DESC""", (since,)).fetchall()
        finally:
            conn.close()
    except Exception as e:                              # noqa: BLE001
        _logger.warning("读取 llm_calls 用量失败：%s", e)
        return []

    used = {str(r[0] or "unknown"): {"count": int(r[1] or 0),
                                     "failed": int(r[2] or 0),
                                     "tokens": int(r[3] or 0)} for r in rows}
    # 先按"命中的注册键"聚合：族键（`ai.*.routine`）要把所有具体 purpose
    # （`ai.ai_ashen.routine` 等）的用量加在一起，否则配置页上族键永远显示
    # "0 次调用"，而真实花销全藏在"⚠️ 未登记"里——那正好是过去的状态。
    agg: dict[str, dict] = {}
    unmatched: dict[str, dict] = {}
    for key, u in used.items():
        hit = match_feature(key)
        if hit is None:
            unmatched[key] = u
            continue
        slot = agg.setdefault(hit, {"count": 0, "failed": 0, "tokens": 0})
        for field in ("count", "failed", "tokens"):
            slot[field] += int(u.get(field) or 0)

    out = []
    for key, meta in FEATURES.items():
        u = agg.get(key) or {"count": 0, "failed": 0, "tokens": 0}
        out.append({"feature": key, "label": meta["label"], "group": meta["group"],
                    "count": u["count"], "failed": u["failed"], "tokens": u["tokens"]})
    for key, u in sorted(unmatched.items(), key=lambda kv: -kv[1]["count"]):
        out.append({"feature": key, "label": f"未登记功能（{key}）", "group": "⚠️ 未登记",
                    "count": u["count"], "failed": u["failed"], "tokens": u["tokens"]})
    return out
