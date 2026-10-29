import openai
import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Dict, List, Any, Optional
from config.settings import config
from config.model_routing import resolve_model
from utils.logger import get_logger
from utils.kline_text import fmt_num as _fmt_num

# 全局日志器
_llm_logger = get_logger("llm")

# ── 模型路由配置 ──
# 兼容别名：改造前路由靠这个前缀常量判断，现已改为读 config/llm_providers.json
# （见 config/providers.py）。常量保留仅为不破坏未知的外部引用，代码内不再使用。
QWEN_MODEL_PREFIXES = ("qwen-",)

# DeepSeek V4 的**思考模式默认开启**（effort=high，见官方 Thinking Mode 文档）。
# 对本项目意味着什么：实测同一句"回复：OK"——
#   非思考 ≈ 20 输出 token；思考模式 137 输出 token（约 6.8 倍），且延迟翻倍。
# 9 个分析节点产出的是"结构化报告"，不需要长链推理 ⇒ **默认显式关闭**；
# 需要更强推理时用环境变量 `DEEPSEEK_THINKING=1` 或单次传 `thinking=True`。
_THINKING_ENV = "DEEPSEEK_THINKING"


_THINKING_SEEN: set = set()


def _log_thinking_once(model: str, thinking: bool) -> None:
    """首次见到某个 (模型, 开关) 组合时打一条日志。

    为什么必须打：思考模式让**输出 token 翻 34 倍**是静默的——账单涨了、延迟涨了，
    但没有任何地方能看出"是谁把开关打开的"。这条日志是唯一的证据链。
    """
    key = (str(model), bool(thinking))
    if key in _THINKING_SEEN:
        return
    _THINKING_SEEN.add(key)
    _llm_logger.info("[LLM] 思考模式: model=%s thinking=%s（环境变量 %s 默认 off）",
                     model, "on" if thinking else "off", _THINKING_ENV)


def _thinking_kwargs(model: str, thinking: bool | None = None) -> dict:
    """给支持"思考模式"的模型显式设置开关；不支持的模型返回空字典。

    ⚠️⚠️ **三家的默认都是开启**，而参数位置各不相同。实测代价（2026-10-20，
        同一句"解释什么是移动平均线"，qwen3.7-flash）：

            关闭：25 输出 token / 1.7s
            开启：992 输出 token / 16.4s
            不传：858 输出 token / 15.7s   ← 服务端默认

        按 0.2/0.8 元/百万算，单只深度分析（62k 入 / 17.8k 出）：
        关闭 ¥0.027；开着输出涨 34 倍 ⇒ 约 **¥0.50**，比留在 DeepSeek
        flash（¥0.17）还贵 3 倍。**换便宜模型不关思考，等于白换。**

    ⚠️ 参数位置不能想当然：`enable_thinking` 是 DashScope 的**请求体顶层字段**。
        第一版探测把它当成 OpenAI SDK 的 `extra_body`（那是 SDK 参数）直接写进
        body，服务端静默忽略，输出仍是 744 token——"配置看起来生效"是最难发现的
        一类问题。本函数返回的才是 SDK 的 `extra_body`（SDK 会展开到 body 顶层）。
    """
    m = str(model or "").lower()
    if thinking is None:
        thinking = os.getenv(_THINKING_ENV, "0").strip().lower() in ("1", "true", "yes", "on")
    if m.startswith("deepseek-v4"):
        body = {"thinking": {"type": "enabled" if thinking else "disabled"}}
    elif m.startswith("qwen"):
        body = {"enable_thinking": bool(thinking)}
    elif m.startswith("glm"):
        body = {"thinking": {"type": "enabled" if thinking else "disabled"}}
    else:
        return {}
    _log_thinking_once(model, thinking)
    return {"extra_body": body}

# 调用方标签（Phase 6.2）：记录"这次调用是谁发起的"。
# 用 ContextVar 而不是全局变量：同一进程里可能有多个线程/图节点在跑，
# 全局变量会把标签串到别人的调用上。
_LLM_PURPOSE: ContextVar = ContextVar("llm_purpose", default="unknown")

# 多 AI 交易联赛（C1，2026-10-14）：在上下文内钉死"该用哪个模型"，退出即还原。
# ⚠️ 走 `_route_default()` 的 default 层而非 explicit 层 —— 这是有意的：
#    优先级 = 调用方当次 explicit > 功能映射（UI 显式配置）> **persona 绑定** > 功能档位。
#    功能映射粒度更细且是用户对单个功能的明确意图，应该赢过 persona 的粗粒度绑定；
#    而 persona 绑定与 `DeepSeekClient(model=X)` 的语义完全一致（其 non-mutation 已验证）。
#    没有 persona 上下文时它是 None ⇒ 行为与改造前逐字节一致。
# ⚠️ 不能用环境变量/全局配置代替 ContextVar —— 并发跑 N 个 AI 时会互相串模型。
_MODEL_OVERRIDE: ContextVar = ContextVar("llm_model_override", default=None)


@contextmanager
def model_override(name):
    """在本上下文内把"该用哪个模型"钉死，退出即还原。

    用途：多个 AI 跑**同一个分析图**但各自用不同模型（N 个 persona 并发）。
    name 为空/None/空白串时不覆盖（视为"跟随默认"）。
    """
    _n = str(name).strip() if name is not None else ""
    token = _MODEL_OVERRIDE.set(_n or None)
    try:
        yield
    finally:
        _MODEL_OVERRIDE.reset(token)


# ── 温度档位（2026-10-13 制定）──
# 背景：此前各调用点的 temperature 是**随手写的**，实测全仓只有 nl_screener 的
# `0.0` 有注释与文档依据（"解析结果必须可复现"），其余 17 处全无说明；而且
# 16 处连参数都没传、直接吃方法默认值（call_api 0.7 / chat 0.5），
# 于是"这个功能为什么是 0.7"根本无从回答。
#
# 定这套具名档位的目的是：**让调用点自解释**。
# 写 `temperature=TEMP_CONSERVATIVE` 比写 `temperature=0.1` 多了一层意图，
# 后来人不必再猜；要整体调整也是改这里一处。
#
# 判据是「这个任务要的是**可复现的结论**，还是**多样的表达**」——
# 投研系统绝大多数任务属于前者。
TEMP_DETERMINISTIC = 0.0    # 纯解析 / 连通性测试：结果必须逐字可复现
TEMP_CONSERVATIVE = 0.1     # 风控、最终决策：结论必须保守稳定，宁可少说不可乱说
TEMP_ANALYTICAL = 0.2       # 依给定数据做判断（指标/财报/汇总/分类）：几乎无创作空间

# 基本面分析输出上限。默认 2000 会切掉 57% 的报告（见 fundamental_analysis 注释）。
FUNDAMENTAL_MAX_TOKENS = int(os.getenv("FUNDAMENTAL_MAX_TOKENS", "3200") or 3200)
TEMP_EXPLORATORY = 0.3      # 有解读余地但受事实约束（舆情倾向、复盘叙述、综合讨论）
TEMP_CONVERSATIONAL = 0.4   # 需要生成有差异的表达（多空辩论、对话助手）

# 档位选择说明（改温度前先读）：
#   - 风控/终审/交易员这类**要落到具体价位与仓位**的，一律 ≤0.2：
#     温度越高，模型越可能给出"看起来合理但没依据"的数字，而这一层下游会直接执行。
#   - 多空辩论是唯一"需要发散"的环节（0.4）——两边温度**必须对称**，
#     否则辩论会天然偏向某一方，比不复盘更糟。
#   - 调这套值属于**未验证变更**：本项目目前没有 LLM 质量评测集，
#     真正能验证的路径是 `analyses` 表的 T+N 回填 + `decision_digest.quality_assessment`
#     的"分析结论胜率"（需 ≥30 条样本）。在那之前，低温的价值是**降低编造风险**，
#     不是"更准"。


@contextmanager
def llm_purpose(label: str):
    """给这段代码里的 LLM 调用打标签（用于 llm_calls 统计）。

    用法：
        with llm_purpose("agents.tech"):
            client.call_api(messages)
    嵌套时内层覆盖外层；退出后恢复。
    """
    token = _LLM_PURPOSE.set(label or "unknown")
    try:
        yield
    finally:
        _LLM_PURPOSE.reset(token)


def current_llm_purpose() -> str:
    return _LLM_PURPOSE.get()


def _log_llm_call(model: str, purpose: str, started: float, response=None,
                  prompt_chars: int = None, response_chars: int = None,
                  ok: bool = True, error: str = None) -> None:
    """把一次调用写进 llm_calls（best-effort）。

    刻意不抛异常：可观测性写入失败绝不能把一次成功的模型调用变成失败。
    延迟在**调用返回/抛错的两个分支里**分别测量，因此失败调用的耗时同样被记录
    （排查"是不是某个节点一直超时"时，失败耗时和成功耗时一样重要）。
    """
    try:
        from db import store
        usage = getattr(response, "usage", None) if response is not None else None
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        if total_tokens is None and (prompt_tokens or completion_tokens):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        # 金额（2026-10-23）：token 数与钱严重脱钩（同一 token，qwen3.7-flash 与
        # deepseek-v4-pro 差 45 倍），只记 token 无法回答"今天花了多少"。
        # ⚠️ 记账失败一律置 None（不是 0）：0 会被读成"免费"，报表永远偏低；
        #    None 让消费方知道要按当前单价补算。
        try:
            from core.llm_cost import call_cost
            cost = call_cost(model, prompt_tokens or 0, completion_tokens or 0) if ok else 0.0
        except Exception as _e:                 # noqa: BLE001
            cost = None
            _llm_logger.warning("成本计算失败（记 None，调用不受影响）: %s", _e)
        store.insert_llm_call({
            "model": model,
            "purpose": purpose,
            "ok": bool(ok),
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_chars": prompt_chars,
            "response_chars": response_chars,
            "error": error,
            "cost_cny": cost,
        })
    except Exception as e:                     # noqa: BLE001
        _llm_logger.warning("LLM 调用记录写入失败（不影响调用）: %s", e)


class LLMError(RuntimeError):
    """LLM 调用失败。

    v2 变更（M0）：call_api 不再把异常吞成 "API调用失败: ..." 字符串返回。
    "错误伪装成正文"会让下游把错误信息当作分析报告继续消费，最终 state.errors
    仍为空、调度日志显示"完成"，整条链路失败却看起来成功。现在统一抛出本异常，
    由 Agent 节点记录为非致命错误，或由 UI 直接展示真实失败原因。
    """


@lru_cache(maxsize=8)
def _get_client_for_model(model: str) -> openai.OpenAI:
    """根据模型名返回对应的 OpenAI 客户端（带 LRU 缓存，避免高频调用重复建连）。

    ⚠️ 厂商路由**不在这里**了（2026-09-29 改造）：
      清单外置在 `config/llm_providers.json`，解析见 `config/providers.py`，
      实际建连见 `llm/providers.py`。想接一家新厂商（Kimi / GLM / OpenAI /
      本地 Ollama / 任何 OpenAI 兼容端点）只需加一条 JSON 记录。

    本函数退化为纯转发，是为了让**函数名 / 签名 / lru_cache 语义保持不变**——
    9 处调用点与 5 个测试文件都按这个名字引用它，改名纯属无谓破坏。
    改造前的语义（"qwen-" 前缀 → DashScope、其余与缺 key 回退 → DeepSeek、
    建连前报错而不是运行时 AuthenticationError）在新实现里逐条保留。

    缓存键是**模型名**：改了某厂商的 base_url 或 API Key 后必须调用
    `reset_client_cache()`，否则旧客户端仍被复用（旧密钥一直生效）。
    """
    from llm.providers import get_client_for_model
    return get_client_for_model(model)


def reset_client_cache() -> None:
    """清空 OpenAI 客户端缓存。

    客户端按模型名缓存（lru_cache），而密钥来自 config——在线修改 .env 后若不
    清缓存，仍会沿用旧密钥直到进程重启。配置页保存密钥后调用本函数即可立即生效。
    """
    _get_client_for_model.cache_clear()


# ================================================================
# 深度分析：数据口径与反幻觉（2026-09-28）
# ================================================================
# 一次修掉三个叠加的问题：
#   1. 三个分析师提示词里**没有任何日期**（技术面尤其明显）→ 模型无时间锚点，只能从
#      训练语料里抄报告模板头，输出「数据截止：最新交易日收盘后｜分析日期：2024年X月X日」
#      （连占位符都没填）——这不是"日期算错"，而是模型在编。
#   2. `technical_analysis` 收了 `stock_data` 却**函数体里从未引用** → K 线完全没进
#      提示词，模型却要谈趋势/支撑阻力/关键位。
#   3. 提示词没有"只能用给定数据"的约束 → 输入为空时模型照样编出年报、研报、一致预期。
DATA_GUARD_RULES = """【数据使用硬约束】（违反即视为分析失败）
1. 只能使用本提示词**已经给出**的数据。未给出的字段一律写「数据缺失，无法判断」，
   不得用你的记忆、常识或推测补齐。
2. 禁止引用本提示词中未出现的内容：财务报表/定期报告、公告、券商研报、机构观点、
   一致预期、数据商（如 Wind、巨潮、Choice、同花顺），以及任何具体报告期或年份。
3. **禁止编造日期**：报告里的日期只能来自【时间口径】，不要自己推断年份。
4. 宁可篇幅短、结论保守，也不要"看起来很专业"的虚构细节。没有依据的判断就不要写。
"""


def time_section(analysis_date: str = None, data_through: str = None) -> str:
    """【时间口径】段：给模型一个明确的时间锚点。

    两个维度都要给、且名字不能混（沿用项目「数据日期 vs 查询时间」的约定）：
      - 分析日期：本次分析执行的那一天（查询时间）
      - 数据截止：输入里最后一根 K 线的日期（数据自带）
    """
    a = str(analysis_date or "").strip() or "未知"
    d = str(data_through or "").strip() or "未知"
    return ("\n【时间口径】（必须原样使用，禁止自行推断年份）\n"
            f"- 分析日期（本次分析执行日）：{a}\n"
            f"- 数据截止（输入中最后一根 K 线日期）：{d}\n"
            f"报告首行必须写成：数据截止：{d}｜分析日期：{a}\n")


# ============================================================
# 缺失字段的统一处理
# ============================================================

# 源里表示"没有值"的各种写法。注意 `0` / `0.0` **不在其中**——0 是有效值，
# 误判成缺失会把"净流入为 0"说成"数据缺失"。
_MISSING_TOKENS = {"", "n/a", "na", "none", "null", "nan", "未知", "-", "--", "暂无"}


def _is_missing(value: Any) -> bool:
    """字段是否"本次没有值"（None / NaN / 空串 / 源里的 N/A 类占位）。"""
    if value is None:
        return True
    if isinstance(value, float) and value != value:      # NaN
        return True
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_TOKENS
    return False


def field_block(pairs) -> tuple:
    """把 [(标签, 值)] 渲染成「只列真实取到的字段」的文本，并返回缺失字段名列表。

    v3（2026-09-28）：原实现写的是 `stock_info.get('sector', 'N/A')` —— 把**缺失字段
    渲染成字面量 `N/A`**，而 `get_stock_info()` 根本不返回 `sector`/`industry`。后果：

      1. 模型分不清「这个字段本次没有」和「这个字段的值就是 N/A」；
      2. 提示词又要求它分析这些字段（如「行业地位、护城河」），于是它只能老老实实
         逐项输出「数据缺失，无法判断」——用户实测看到的基本面报告就是一份"缺失清单"：
         有一整段「公司质地分析：业务模式与核心竞争力 / 行业地位 / 护城河：数据缺失，无法判断」。

    现在改成：**有值的正常列，没值的集中成一行显式声明**（配合 `missing_note`），
    模型就不必也不该再为每个缺失字段各写一段。

    Returns:
        (文本, 缺失字段名列表)
    """
    lines, missing = [], []
    for label, value in pairs:
        if _is_missing(value):
            missing.append(label)
        else:
            lines.append(f"- {label}：{value}")
    return "\n".join(lines), missing


def missing_note(missing) -> str:
    """缺失字段的统一声明段；没有缺失时返回空串。

    措辞刻意含「**逐条**」二字：不加这句，模型会为每个缺失字段各写一行
    「数据缺失，无法判断」，把报告变成缺失清单。
    """
    names = [m for m in (missing or []) if m]
    if not names:
        return ""
    return ("\n【本次未提供的字段】" + "、".join(names)
            + "。\n这些字段本次没有数据：不得猜测其取值，也不必为它们**逐条**写"
              "「数据缺失，无法判断」——如需交代，请在报告末尾用一句话统一说明。\n")


# 均线相对位置预判（v6，2026-10-13）
# ------------------------------
# 为什么必须在**输入侧**预判，而不是靠事后审计：
# 技术面提示词给的是**平铺的裸数字**（收盘价=5.46 / MA5=5.408 / MA10=5.315 /
# MA20=5.295 / MA60=5.1615），模型必须**自己逐个比较**。
# 实测（2026-10-13，600028 中国石化）：它把前三个都判对了（"高于MA5/MA10/MA20"），
# **第四个判反** —— 写成「收盘价仍低于MA60（5.1615）」，而 5.46 > 5.1615。
# 该错误随后被多头引用 2 次、空头引用 9 次，空头还以它作为核心论据。
#
# **数字全对、只错关系** ⇒ 任何"数值可追溯"类审计都拦不住
# （5.46 与 5.1615 都逐字真在输入里，`audit_numbers` 判定"可追溯"而放行）。
# ⇒ 与其等模型算错再检测，不如**不让它算**：把关系提前算好、连同差值一起写进提示词。
# 配套的检测网是 `core.report_audit.audit_relations`（一防一查，两侧都要有）。
#
# ⚠️ 加字段必须同时检查渲染层 —— 这里是渲染层本身，所以不会重演
#    "工具给了、模型看不到"那类缺陷（本项目已踩 3 次）。
_MA_POS_LINES = (
    ("MA5", "ma5"), ("MA10", "ma10"), ("MA20", "ma20"), ("MA60", "ma60"),
    ("布林带中轨", "bb_middle"),
)


def ma_position_block(indicators: Dict = None) -> str:
    """把「收盘价相对各均线/中轨的高低」**预先算好**，供提示词直接引用。

    返回空串表示算不出来（缺收盘价，或所有均线都缺失）——
    此时调用方**不应**插入该段，而不是插入一段空的（那会变成"数据缺失"噪声）。
    """
    ind = indicators or {}

    def _f(key):
        v = ind.get(key) if isinstance(ind, dict) else None
        if v is None or v != v:                 # None 或 NaN
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    price = _f("price")
    if not price:                               # 缺收盘价 → 任何"高低"都无从谈起
        return ""
    lines = []
    for label, key in _MA_POS_LINES:
        v = _f(key)
        if v is None:
            continue
        gap = (price - v) / price * 100
        rel = "高于" if gap > 0 else ("低于" if gap < 0 else "等于")
        lines.append(f"  {label} = {v:g}　→　收盘价**{rel}**{label}（{gap:+.2f}%）")
    if not lines:
        return ""

    mas = [(_f(k), k) for k in ("ma5", "ma10", "ma20", "ma60")]
    order = ""
    if all(v is not None for v, _ in mas):
        vals = [v for v, _ in mas]
        if all(vals[i] > vals[i + 1] for i in range(3)):
            order = "\n  均线排列：MA5 > MA10 > MA20 > MA60（**多头排列**）"
        elif all(vals[i] < vals[i + 1] for i in range(3)):
            order = "\n  均线排列：MA5 < MA10 < MA20 < MA60（**空头排列**）"
        else:
            order = "\n  均线排列：MA5 / MA10 / MA20 / MA60 相互纠缠（无一致方向）"
    return ("均线相对位置（**已由代码算好，请直接照抄这里的「高于/低于」，"
            "不要自己再比较一遍**）：\n"
            f"  收盘价 = {price:g}\n" + "\n".join(lines) + order)


# 执行层（交易员 / 风控）的指标块 —— **唯一实现**（v6，2026-10-13）
# ------------------------------
# 为什么需要，以及必要时要修的两个实测缺口：
#
# ① **风控审"止损是否恰当"，却拿不到 ATR**（2026-10-13 实测）。
#    风控 prompt 原先只注入 `股票 / 现价 / trade_plan / 风控规则块`，
#    而 ATR 进入风控视野的**唯一通道是 `trade_plan`**；但 `trader_node` 的实现是
#    `trade_plan = json.loads(模型输出)` —— **代码从不往 trade_plan 注入 atr**，
#    提示词要求的输出键里也没有 `atr`。
#    ⇒ 实测 **10 份产物只有 2 份**的 trade_plan 带 atr（模型自己多写的）。
#    ⚠️ **"时有时无"比"完全没给"更难发现**：时对时错，报告读起来都"有结论"，
#       只有逐份比对才看得出依据不同。ATR 是判断止损宽度的唯一标尺，必须由代码给。
#
# ② **交易员只拿到 MA20**，而它要判"均线支撑/压制"（实测报告里出现「MA60 的压制作用」，
#    只能从 tech_report 抄）。同时原 prompt 用**字面量 `N/A`** 兜底现价/MA20/RSI
#    —— 这个反模式在技术面/基本面 prompt 里**早已禁掉**（模型分不清
#    「本次没有这个字段」与「字段值就是 N/A」），这里属**漏改**。
#    ⇒ 本函数统一走 `field_block` + `missing_note`（缺就是缺，不写 N/A）。
#
# ③ 顺带复用 `ma_position_block()`：把"防"从技术面铺到执行层。
#
# ⚠️ 每段末尾都带**用途限定句**，防两种已知偏差：
#    · 风控越界（本来只审风险，喂了均线可能去重做技术分析）；
#    · 交易员被引导（本应依据研究汇总决策，可能变成"跟着均线走"）。
_EXEC_IND_LABELS = (
    ("现价", "price"),
    ("MA5", "ma5"), ("MA10", "ma10"), ("MA20", "ma20"), ("MA60", "ma60"),
    ("RSI", "rsi"),
    # ATR14 是**止损宽度的标尺**，必须显式给出（见上 ① ）
    ("ATR14(元)", "atr14"),
    ("成交量(股)", "volume"), ("5日均量", "volume_ma5"), ("20日均量", "volume_ma20"),
)
_EXEC_SCOPE_NOTE = {
    "trader": "\n⚠️ 上面这些**只用于让价位自洽、让表述可被核实**；"
              "**不要**据此推翻上游研究汇总的结论，也不要改变你的交易方向判断。",
    "risk": "\n⚠️ 上面这些**只用于核对交易计划的价位与仓位是否自洽**；"
            "**不要**据此重做技术分析或预测方向 —— 你的职责是审风险，不是找买点。",
}


def execution_indicator_block(indicators: Dict = None, *, role: str = "trader") -> str:
    """交易员 / 风控节点共用的【技术指标 + 均线相对位置 + 用途限定】块。

    `role` 取 `"trader"` / `"risk"`，只影响末尾的用途限定句（措辞不同，见上）。
    """
    ind = indicators if isinstance(indicators, dict) else {}
    text, missing = field_block(
        tuple((lbl, ind.get(key)) for lbl, key in _EXEC_IND_LABELS))
    parts = ["【技术指标与均线位置（**本节点可用的全部数值依据**）】",
             text or "（本次未取到任何技术指标）"]
    ma_pos = ma_position_block(ind)
    if ma_pos:
        parts += ["", ma_pos]
    note = missing_note(missing)
    if note:
        parts.append(note)
    parts.append(_EXEC_SCOPE_NOTE.get(role, _EXEC_SCOPE_NOTE["trader"]))
    return "\n".join(parts)


def _kline_digest(stock_data: Any, n: int = 20) -> str:
    """把 K 线压成「最近 n 根 日期/收盘/成交量」的可读摘要；拿不到就返回空串。

    ⚠️ 唯一实现在 `utils.kline_text.digest`——`core/agents.py` 取「数据截止日」用的是
    同一个模块（`last_bar_date`）。两处必须对"哪一天算数据截止""没日期怎么办"给出
    同一个答案，否则报告头部的日期与实际喂给模型的序列会互相矛盾，所以此处只做转发。
    """
    from utils.kline_text import digest
    return digest(stock_data, n)


def _render_financials(fin: Dict) -> str:
    """把 `core.agent_tools.tool_financials` 的输出渲染成给模型看的紧凑文本。

    字段顺序沿用 tool_financials 的「重要性顺序」（利润表/资产表/同比/比率在前、
    现金流量表在后），因为下游模型对超长输入同样会忽略尾部内容。
    兼容旧的 `financial_ratios` 结构。
    """
    if not fin or fin.get("error"):
        return ""

    ratios = fin.get("financial_ratios") or {}
    if ratios and not fin.get("报告期"):                # 旧结构
        lines = []
        if ratios.get("报告期"):
            lines.append(f"报告期：{ratios['报告期']}")
        for k, v in ratios.items():
            if k != "报告期":
                lines.append(f"- {k}: {_fmt_num(v)}")
        return "\n".join(lines)

    lines = []
    if fin.get("数据来源"):
        lines.append(f"数据来源：{fin['数据来源']}")
    if fin.get("数据截止"):
        lines.append(f"最新报告期：{fin['数据截止']}")
    if fin.get("报告期"):
        lines.append("报告期（新→旧）：" + " / ".join(str(x) for x in fin["报告期"]))
    if fin.get("单位"):
        lines.append(f"单位：{fin['单位']}")
    # 口径紧跟单位 —— 这是"累计数还是单季数"的唯一声明，位置必须显眼
    if fin.get("口径"):
        lines.append(f"口径：{fin['口径']}")

    # 已渲染的键先登记，末尾做"未登记字段兜底"，防止新增字段被静默丢掉。
    # ⚠️ 这里原来是**硬编码白名单**（只渲染固定的 5 个 label）：
    #    往 tool_financials 加字段时，数据会静默地到不了模型手上 ——
    #    与项目里"注册表加了别名、解析器不认"是同一类漂移缺陷。
    ordered = ("利润表(亿元)", "利润表单季(亿元)", "资产负债表(亿元)", "同比%",
               "同比基准(亿元)", "自算比率", "现金流量表(亿元)")
    rendered = {"数据来源", "数据截止", "报告期", "单位", "口径", "代码", "as_of"}

    def _emit(label: str, block: dict) -> None:
        lines.append(f"【{label}】")
        for k, v in block.items():
            if isinstance(v, list) and v:
                lines.append(f"- {k}: " + " / ".join(_fmt_num(x) for x in v))
            elif not isinstance(v, list):
                lines.append(f"- {k}: {_fmt_num(v)}")

    for label in ordered:
        block = fin.get(label)
        rendered.add(label)
        if isinstance(block, dict) and block:
            _emit(label, block)

    for key in ("未取到核心科目", "缺失字段", "三表外的关键指标", "银行口径提示", "说明"):
        val = fin.get(key)
        rendered.add(key)
        if val:
            text = " / ".join(str(x) for x in val) if isinstance(val, list) else str(val)
            lines.append(f"【{key}】{text}")

    # 兜底：任何没被上面两段渲染到的字段都要出现在文本里（新增字段不再静默丢失）
    for label, block in fin.items():
        if label in rendered:
            continue
        if isinstance(block, dict) and block:
            _emit(label, block)
        elif isinstance(block, (str, int, float)) and str(block).strip():
            lines.append(f"【{label}】{block}")
    return "\n".join(lines)


class DeepSeekClient:
    """多厂商 API 客户端（类名保留，实际支持任意家 OpenAI 兼容端点）

    模型路由规则（2026-09-29 起由 `config/llm_providers.json` 决定）:
      - 命中某厂商声明的模型名 / 匹配模式 → 该厂商
      - 未命中 → `default_provider` 兜底（内置默认即 DeepSeek，与改造前一致）
      - 目标厂商密钥未配置时自动回退到默认厂商，并记 warning
    增删厂商在「系统配置 → 模型接入」页操作，无需改代码。
    """

    def __init__(self, model=None):
        # ⚠️ 必须区分「调用方显式指定的模型」与「回落到全局默认的模型」：
        #    功能路由的 `default` 层只接受**前者**。
        #    否则 `self.model` 永远非空（回落成 `DEFAULT_MODEL_NAME`），`default` 层**总是命中**
        #    ⇒ 功能"档位(tier)"变成死代码 ⇒ 在 `DEFAULT_MODEL_NAME ≠ 档位模型` 的部署里
        #    **静默换模型**（实测可复现：DEFAULT=deepseek-chat、QWEN_TURBO_MODEL=qwen-max 时，
        #    `chat.tool_loop` 会从 qwen-max 悄悄变成 deepseek-chat）。
        #    ⚠️ 这个缺陷在"三个变量恰好同值"的默认部署下**完全看不出来** —— 属静默失效。
        #
        #    "显式指定"的来源有两个（优先级相同，都是调用方的明确意图）：
        #    ① 构造参数 model；② persona 上下文（`model_override()`，多 AI 联赛 C1）。
        #    ⚠️ ② 在 `_route_default()` 里**按调用时刻**读取，不在这里快照 ——
        #       实测踩坑（2026-10-14）：生产代码是 `client = DeepSeekClient()` 在
        #       `with model_override(...)` **之外**创建，构造时覆盖为空 ⇒ persona 模型
        #       全部失效（3 个 AI 都跑成了 qwen-max）。当时的单测验的是"在 override
        #       内部构造"，与生产路径不一致 ⇒ **验了函数 ≠ 验了接线**。
        self._given_model = model
        self.model = model or _MODEL_OVERRIDE.get() or config.DEFAULT_MODEL_NAME
        self.client = _get_client_for_model(self.model)

    def _route_default(self):
        """参与功能路由 `default` 层的值：**只有构造时显式给了模型才返回它**。

        返回 None 时，`resolve_model` 会继续落到「功能档位」→ 与改造前的表达式一致
        （旧实现：`CHAT_MODEL_FAST = QWEN_TURBO_MODEL`、`REVIEW_MODEL = QWEN_PLUS_MODEL`、
        其余走 `DEFAULT_MODEL_NAME`）。

        ⚠️ persona 覆盖（`model_override`）在这里**按调用时刻**读取，而不是构造时快照：
        生产里 client 常常在 `with model_override(...)` 之外创建（`run_persona_round` 就是），
        构造时快照会让覆盖整体失效（2026-10-14 实测：3 个 persona 全跑成 qwen-max）。
        """
        return self._given_model or _MODEL_OVERRIDE.get()

    def call_api(self, messages: List[Dict[str, str]], model: Optional[str] = None,
                 temperature: float = TEMP_ANALYTICAL, max_tokens: int = 2000,
                 purpose: Optional[str] = None,
                 thinking: bool | None = None) -> str:
        """调用 LLM API（自动路由 DeepSeek / Qwen）

        temperature 默认走 `TEMP_ANALYTICAL`（0.2）而不是常见的 0.7：
        本项目是**投研分析系统**，有两条硬约束要求低温——
          ① `_audit()` 会标记"输入里找不到出处的数字"，温度越高越容易自由发挥；
          ② 铁律"数字可追溯"，同一输入应给出基本一致的结论。
        需要多样表达的场景（多空辩论、对话助手）**必须显式传更高档位**，
        不要依赖默认值。档位定义见模块顶部 TEMP_* 常量。

        purpose: 调用方标签（不传则用 llm_purpose() 上下文里的标签，再退回 "unknown"），
                 既写入 llm_calls 供可观测性，也用作**功能级模型路由**的功能键
                 （见 config/model_routing.py：显式 model > 功能映射 > 自带模型 > 档位）。
        """
        # ⚠️ label 必须**在解析模型之前**算出：功能路由以 purpose 为键。
        label = purpose or _LLM_PURPOSE.get()
        # 功能级路由：`default` 只传「构造时**显式**指定的模型」（见 `_route_default`）。
        # ⚠️ 不能直接传 `self.model` —— 它会回落成 `DEFAULT_MODEL_NAME` 而永不为空，
        #    导致 `default` 层总是命中、功能档位失效（在 DEFAULT ≠ 档位 的部署里静默换模型）。
        model_to_use = resolve_model(label, explicit=model, default=self._route_default())
        # D1 修复：直接复用缓存的客户端，不再每次 call_api 新建实例
        client = _get_client_for_model(model_to_use)

        # 对于 reasoner 模型，自动增加 max_tokens
        if "reasoner" in model_to_use.lower() and max_tokens <= 2000:
            max_tokens = 8000

        _llm_logger.info(f"调用 {model_to_use} (temperature={temperature}, max_tokens={max_tokens})")

        started = time.perf_counter()
        prompt_chars = sum(len(str(m.get("content") or "")) for m in (messages or []))

        try:
            response = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **_thinking_kwargs(model_to_use, thinking),
            )

            message = response.choices[0].message

            result = ""
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                result += f"【推理过程】\n{message.reasoning_content}\n\n"

            if message.content:
                result += message.content

            if not result:
                # 空响应同样是失败，不能返回占位文本让下游当成报告
                raise LLMError(f"{model_to_use} 返回空内容")

            _log_llm_call(model_to_use, label, started, response=response,
                          prompt_chars=prompt_chars, response_chars=len(result), ok=True)
            return result

        except LLMError as e:
            _log_llm_call(model_to_use, label, started, prompt_chars=prompt_chars,
                          ok=False, error=str(e))
            raise
        except Exception as e:
            _llm_logger.error(f"API 调用失败: {e}")
            _log_llm_call(model_to_use, label, started, prompt_chars=prompt_chars,
                          ok=False, error=f"{type(e).__name__}: {e}")
            raise LLMError(f"{model_to_use} 调用失败: {e}") from e

    def chat(self, messages: List[Dict], model: Optional[str] = None,
             tools: Optional[List[Dict]] = None, temperature: float = TEMP_CONVERSATIONAL,
             max_tokens: int = 2000, purpose: Optional[str] = None,
             thinking: bool | None = None):
        """通用对话接口（支持 function calling），返回原始 assistant message。

        与 call_api 的区别：不吞异常、不拼装推理内容，供 Agent 循环判断
        message.tool_calls 并追加 role=tool 消息。
        同样写入 llm_calls（Phase 6.2）——AI助手这条链路的用量此前只存在会话内存里。
        模型同样走**功能级路由**（purpose 为功能键），所以 `chat.tool_loop` 与
        `chat.tool_loop_summary` 可各自配模型、互不影响。
        """
        label = purpose or _LLM_PURPOSE.get()
        model_to_use = resolve_model(label, explicit=model, default=self._route_default())
        client = _get_client_for_model(model_to_use)
        kwargs = dict(model=model_to_use, messages=messages,
                      temperature=temperature, max_tokens=max_tokens)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        _llm_logger.info(f"chat 调用 {model_to_use} (tools={len(tools) if tools else 0}, "
                         f"messages={len(messages)})")
        started = time.perf_counter()
        prompt_chars = sum(len(str(m.get("content") or "")) for m in (messages or []))
        try:
            kwargs.update(_thinking_kwargs(model_to_use, thinking))
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            _log_llm_call(model_to_use, label, started, prompt_chars=prompt_chars,
                          ok=False, error=f"{type(e).__name__}: {e}")
            raise
        usage = getattr(response, "usage", None)
        self.last_usage = None
        if usage:
            self.last_usage = {"prompt": usage.prompt_tokens,
                               "completion": usage.completion_tokens}
            _llm_logger.info(f"token 用量: prompt={usage.prompt_tokens}, "
                             f"completion={usage.completion_tokens}, total={usage.total_tokens}")
        message = response.choices[0].message
        _log_llm_call(model_to_use, label, started, response=response,
                      prompt_chars=prompt_chars,
                      response_chars=len(str(getattr(message, "content", "") or "")), ok=True)
        return message

    def chat_stream(self, messages: List[Dict], model: Optional[str] = None,
                    temperature: float = TEMP_CONVERSATIONAL, max_tokens: int = 2000,
                    purpose: Optional[str] = None):
        """流式对话，逐段 yield 文本增量（不支持 tools，用于最终回答渲染）

        流式调用同样记录耗时与用量（在流正常结束/报错时写一条）。
        注意：调用方中途放弃这个生成器时不会写记录（无从判断成败），这是有意的。
        """
        model_to_use = resolve_model(purpose or _LLM_PURPOSE.get(),
                                     explicit=model, default=self._route_default())
        client = _get_client_for_model(model_to_use)
        started = time.perf_counter()
        label = purpose or _LLM_PURPOSE.get()
        prompt_chars = sum(len(str(m.get("content") or "")) for m in (messages or []))
        chunks = 0
        try:
            stream = client.chat.completions.create(
                model=model_to_use, messages=messages,
                temperature=temperature, max_tokens=max_tokens, stream=True,
                **_thinking_kwargs(model_to_use),
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks += 1
                    yield chunk.choices[0].delta.content
        except Exception as e:
            _log_llm_call(model_to_use, label, started, prompt_chars=prompt_chars,
                          response_chars=chunks, ok=False, error=f"{type(e).__name__}: {e}")
            raise
        _log_llm_call(model_to_use, label, started, prompt_chars=prompt_chars,
                      response_chars=chunks, ok=True)

    def technical_analysis(self, stock_info: Dict, stock_data: Any, indicators: Dict,
                           analysis_date: str = None, data_through: str = None) -> str:
        """技术面分析

        v3（2026-09-28）三处修复：
          1. 注入【时间口径】（分析日期 / 数据截止）——此前提示词里**没有任何日期**，
             模型没有时间锚点，只能从训练语料里抄报告模板头，输出
             「数据截止：最新交易日收盘后｜分析日期：2024年X月X日」；
          2. `stock_data` 之前**收了却从未使用**，现压成 K 线摘要喂进提示词
             （趋势/支撑阻力/关键位都需要价格序列，只看指标值只能硬编）；
          3. 追加【数据使用硬约束】，禁止引用输入之外的年报/研报/数据商/年份。
        """
        kline = _kline_digest(stock_data)
        if kline:
            kline_section = f"\n【历史K线数据（真实输入）】\n{kline}\n"
        else:
            kline_section = ("\n【历史K线数据】未取到（数据缺失：不得描述"
                             "「放量突破 / 缩量回调 / 站上某价位」等需要价格序列的形态）\n")
        # 只列真实取到的字段；缺失的集中声明。
        # 原实现把每个字段都写成 `{...get(k, 'N/A')}`：缺字段时渲染成字面量 N/A，
        # 模型分不清「本次没有这个字段」与「字段值就是 N/A」，却仍被要求分析 MACD/KDJ/布林带，
        # 于是只能在报告里逐项写「数据缺失」。
        info_text, info_missing = field_block((
            ("股票代码", stock_info.get('symbol')),
            ("股票名称", stock_info.get('name')),
            ("当前价格", stock_info.get('current_price')),
            ("涨跌幅(%)", stock_info.get('change_percent')),
        ))
        # ⚠️ 有序字段表 + **未登记字段兜底**。
        #    这张表原先只列 13 个固定字段 —— `get_latest_indicators` 后来补出的
        #    布林中轨 / MACD柱 / 成交量 / 5日与20日均量**一个都进不了提示词**
        #    （实测 2026-10-08 重跑：技术面报告里「中轨」出现 0 次，
        #      而这一版已经把它们加进 indicators 了）。
        #    与 `_render_financials` 的硬编码白名单是同一类缺陷：
        #    **加字段必须同时检查渲染层**，否则"工具给了、模型看不到"。
        ind_labels = (
            ("收盘价", 'price'),
            ("MA5", 'ma5'), ("MA10", 'ma10'),
            ("MA20", 'ma20'), ("MA60", 'ma60'),
            ("布林带上轨", 'bb_upper'), ("布林带中轨", 'bb_middle'),
            ("布林带下轨", 'bb_lower'),
            ("RSI", 'rsi'),
            ("MACD", 'macd'), ("MACD信号线", 'macd_signal'),
            ("MACD柱", 'macd_histogram'),
            ("K值", 'k_value'), ("D值", 'd_value'),
            # 5 日与 20 日均量**分开列**：只给量比（分母是 5 日均量）时，
            # 模型会把它当成 20 日均量（实测旧报告把"20 日均量"夸大 39%）
            ("成交量(股)", 'volume'),
            ("5日均量", 'volume_ma5'), ("20日均量", 'volume_ma20'),
            ("量比(成交量/5日均量)", 'volume_ratio'),
            # ATR14 —— **止损宽度的标尺**。没有它，"止损是否恰当"这个判断只能凭感觉：
            # 实测 603127 那份计划止损距离 1.50 元，而该票 ATR14 = 2.84 元
            # （占现价 5.45%）→ 距离不到一天的平均波动，等于"必被日常噪声扫损"。
            ("ATR14(元)", 'atr14'),
        )
        _used = {key for _lbl, key in ind_labels}
        extra_fields = tuple(
            (str(k), v) for k, v in (indicators or {}).items()
            if k not in _used and isinstance(v, (int, float))
            and v == v)                                    # 排除 NaN
        ind_text, ind_missing = field_block(
            tuple((lbl, (indicators or {}).get(key)) for lbl, key in ind_labels)
            + extra_fields)
        missing_section = missing_note(info_missing + ind_missing)
        # ⚠️ 关系预判段必须有：见 `ma_position_block` 的说明 —— 实测 600028 就是
        #    因为要模型自己比较 4 个均线，才在 MA60 上把"高于"写成"低于"。
        _ma_pos = ma_position_block(indicators)
        ma_pos_section = f"{_ma_pos}\n" if _ma_pos else ""
        prompt = f"""
{time_section(analysis_date, data_through)}
你是一名资深的技术分析师。请基于以下股票数据进行专业的技术面分析：

股票信息：
{info_text or "（本次未取到基本信息）"}
{kline_section}
最新技术指标：
{ind_text or "（本次未取到任何技术指标）"}
{ma_pos_section}{missing_section}
请从以下角度进行分析（**只分析上面有数据的角度**，缺数据的角度不必单开段落）：
1. 趋势分析（均线系统、价格走势）
2. 超买超卖分析（RSI、KDJ）
3. 动量分析（MACD）
4. 支撑阻力分析（布林带）
5. 成交量分析
6. 短期、中期、长期技术判断
7. 关键技术位分析

{DATA_GUARD_RULES}
请给出技术分析报告，包含风险提示。所有结论都必须能追溯到上面的输入数据。
"""
        messages = [
            {"role": "system", "content":
                "你是一名经验丰富的股票技术分析师，具有深厚的技术分析功底。"
                "你只能使用用户消息中给出的数据；未给出的要明确写「数据缺失」，"
                "不得引用任何未出现在用户消息中的数据、报告、机构或日期。"},
            {"role": "user", "content": prompt}
        ]

        return self.call_api(messages, temperature=TEMP_ANALYTICAL,
                             purpose="client.technical_analysis")
    
    def fundamental_analysis(self, stock_info: Dict, financial_data: Dict = None,
                             quarterly_data: Dict = None,
                             analysis_date: str = None, data_through: str = None) -> str:
        """基本面分析

        v3（2026-09-28）：
          - 注入【时间口径】（分析日期 / 数据截止），此前提示词里**一个日期都没有**；
          - 财务数据改为直接渲染 `core.agent_tools.tool_financials` 的输出
            （三表科目 + 同比 + 自算比率），不再只认已废弃的 `financial_ratios` 结构；
          - 无财报时收窄分析范围（只允许写"无法判断"），并追加【数据使用硬约束】。
            调用方 core/agents.py 在完全取不到财报时**直接降级为固定文案、不调用 LLM**，
            本函数的分支只作防御。
        """
        
        # v3（2026-09-28）：直接渲染 tool_financials 的输出（三表科目+同比+自算比率）。
        # 旧实现只认 financial_data["financial_ratios"]，而调用方一直显式传 None →
        # 提示词里"详细财务指标"整段不生成，模型却仍被要求分析 ROE/毛利率/8 期季报趋势
        # → 只能编（实测编出「上交所 2021年报~2024年一季报共13份定期报告」
        #   「Wind 一致预期(2024.06.30 校验)」）。
        fin_text = _render_financials(financial_data or {})
        if fin_text:
            financial_section = "\n【财务报表（真实输入）】\n" + fin_text + "\n"
        else:
            financial_section = (
                "\n【财务报表】本次**未提供**任何财务报表数据。因此不得分析 ROE、净利率、"
                "资产负债率、现金流、季报趋势，也不得引用任何报告期、定期报告或一致预期。\n")

        # ── 派生指标表（2026-10-24）─────────────────────────────────
        # 动机：同一模型在**隔离算术题**上 8/8 全对，却在长报告里把
        #   `296.56亿元（790.1 - 293.54）`（应得 496.56，错 40%）、
        #   `28.44%（47.97/177.47）`（应得 27.03%）算错。
        # 差距来自任务形状，不是能力 ⇒ 修法是**别让它算**：程序先把单季/比率/同比
        # 连同可逐字复制的算式算好，模型只引用不推导。
        # 零额外 LLM 调用，只多约 700 输入 token（≈ ¥0.0002/次）。
        derived_items, derived_section, usage_rules = [], "", ""
        # 开关：`BIGA_DERIVED_FIN=0` 可整段关闭（应急回退 / A/B 对照都走同一条代码路径，
        # 保证两次跑的提示词只差这一段，而不是"脚本里复制了一份 prompt"）。
        _derived_on = str(os.getenv("BIGA_DERIVED_FIN", "1")).strip().lower() not in (
            "0", "false", "off", "no")
        try:
            from core.derived_fin import build as _build_derived, render as _render_derived
            from core.derived_fin import DERIVED_USAGE_RULES
            derived_items = _build_derived(financial_data or {}) if _derived_on else []
            if derived_items:
                derived_section = "\n" + _render_derived(derived_items) + "\n"
                usage_rules = DERIVED_USAGE_RULES
        except Exception as e:                  # noqa: BLE001 - 派生失败只降级，不阻断
            _llm_logger.warning("[基本面] 派生指标表构造失败，按无派生表处理: %s", e)

        quarterly_section = ""
        if quarterly_data and quarterly_data.get('data_success'):
            quarterly_section = "\n【季报数据】\n季报模块尚未集成。\n"

        # 分析维度分成两套：**没有真财报时不允许再要求分析盈利能力/财务健康度/季报趋势**
        # ——那些要求本身就是编造的来源（"必须分析 8 期季报"＋"没有季报数据"＝必然虚构）。
        if fin_text:
            scope_section = """请从以下维度进行分析（**每个维度都必须落到上面给出的财报科目上**）：

1. **公司质地（仅限财务可推断的部分）**：从已给三表能读出的公司规模、资产结构、
   资本开支/在建工程变化、应收与存货变化所反映的经营特征
   （⚠️ 业务模式、行业地位、护城河、客户结构、产能、专利、市场份额等**定性信息
   本次完全未提供**，不要为此单开段落，也不要逐项写「数据缺失」）
2. **盈利能力分析**：ROE/ROA、毛利率与净利率趋势、盈利质量与持续性
3. **财务健康度分析**：资产负债结构、偿债能力、现金流、财务风险
4. **成长性分析**：收入与利润增长趋势及驱动因素（**仅限已给报告期能支持的**）
5. **报告期趋势**：基于**上面已给出的报告期序列**分析关键科目（营收/净利/现金流）变化
6. **估值分析**：结合上面的 PE/PB 与已给财报数据判断估值水平
7. **投资价值判断**：综合评分（0-100）、投资亮点、投资风险

**分析要求：**
- 只使用上面给出的报告期数据，**不要推断未给出的季度**，也不要引用外部研报
- 每个结论都要指名依据的是哪个科目、哪一期
- 某个维度确实算不出来时，**在末尾用一句话统一说明**（如「行业与业务定性信息本次未提供，
  相关维度不评估」），**不要每个维度各写一段「数据缺失，无法判断」来凑篇幅**
"""
        else:
            scope_section = """本次**未提供财务报表数据**，只允许基于上面的基本信息与估值指标写一段简短说明：

1. 明确写出「本次未提供财务报表，盈利能力 / 偿债能力 / 现金流 / 成长性均无法判断」
2. 可以如实转述 PE/PB/PS 等估值字段的取值或缺失情况
3. **不得**出现任何报告期、定期报告、研报、机构名称、一致预期，也不得据此推断经营质量
"""

        # 字段渲染：只列真实取到的，缺失的集中声明（不要渲染成字面量 N/A）
        info_text, info_missing = field_block((
            ("股票代码", stock_info.get('symbol')),
            ("股票名称", stock_info.get('name')),
            ("当前价格", stock_info.get('current_price')),
            ("市值", stock_info.get('market_cap')),
            ("行业", stock_info.get('sector')),
            ("细分行业", stock_info.get('industry')),
        ))
        val_text, val_missing = field_block((
            ("市盈率(PE)", stock_info.get('pe_ratio')),
            ("市净率(PB)", stock_info.get('pb_ratio')),
            ("市销率(PS)", stock_info.get('ps_ratio')),
            ("Beta系数", stock_info.get('beta')),
            ("52周最高", stock_info.get('52_week_high')),
            ("52周最低", stock_info.get('52_week_low')),
        ))
        missing_section = missing_note(info_missing + val_missing)

        prompt = f"""
{time_section(analysis_date, data_through)}
你是一名资深的基本面分析师。请基于以下详细信息进行分析（只能使用已给出的数据）：

【基本信息】
{info_text or "（本次未取到任何基本信息字段）"}

【估值指标】
{val_text or "（本次未取到任何估值字段）"}
{financial_section}{derived_section}
{quarterly_section}
{missing_section}
{scope_section}
{usage_rules}
{DATA_GUARD_RULES}
请给出基本面分析报告。所有结论都必须能追溯到上面的输入数据；没有依据的维度就写「数据缺失」。
"""
        
        messages = [
            {"role": "system", "content":
                "你是一名经验丰富的股票基本面分析师，擅长公司财务分析和行业研究。"
                "你只能使用用户消息中给出的数据；未给出的要明确写「数据缺失」，"
                "不得引用任何未出现在用户消息中的报告期、定期报告、研报、机构或数据商。"},
            {"role": "user", "content": prompt}
        ]
        
        # ⚠️ 输出上限必须显式抬高：`call_api` 默认 2000，而基本面报告的实测长度分布是
        #    qwen-max 998 / qwen3.7-flash 1902（19/35 打满）/ deepseek-v4-flash 1990（8/9 打满）。
        #    打满 = 正文被**拦腰切断**（实测结尾出现「…的5.8倍** | |」这类残句，占 37/65 份）。
        #    提高上限的增量成本 ≈ 1000 输出 token ≈ ¥0.0008/次（qwen3.7-flash），
        #    而截断会让整节的结论消失 —— 这是本项目性价比最高的一处修复。
        return self.call_api(messages, temperature=TEMP_ANALYTICAL,
                             max_tokens=FUNDAMENTAL_MAX_TOKENS,
                             purpose="client.fundamental_analysis")
    
    def fund_flow_analysis(self, stock_info: Dict, indicators: Dict, fund_flow_data: Dict = None,
                           analysis_date: str = None, data_through: str = None) -> str:
        """资金面分析"""
        
        # 构建资金流向数据部分 - 使用akshare格式化数据
        fund_flow_section = ""
        if fund_flow_data and fund_flow_data.get('data_success'):
            # 使用格式化的资金流向数据
            from data.fund_flow import FundFlowAkshareDataFetcher
            fetcher = FundFlowAkshareDataFetcher()
            fund_flow_section = f"""

【近20个交易日资金流向详细数据】
{fetcher.format_fund_flow_for_ai(fund_flow_data)}

以上是通过akshare从东方财富获取的实际资金流向数据，请重点基于这些数据进行趋势分析。
"""
        else:
            fund_flow_section = "\n【资金流向数据】\n注意：未能获取到资金流向数据，将基于成交量进行分析。\n"
        
        prompt = f"""
{time_section(analysis_date, data_through)}
你是一名资深的资金面分析师，擅长从资金流向数据中洞察主力行为和市场趋势。

【基本信息】
股票代码：{stock_info.get('symbol', 'N/A')}
股票名称：{stock_info.get('name', 'N/A')}
当前价格：{stock_info.get('current_price', 'N/A')}
市值：{stock_info.get('market_cap', 'N/A')}

【技术指标】
- 量比：{indicators.get('volume_ratio', 'N/A')}
- 当前成交量与5日均量比：{indicators.get('volume_ratio', 'N/A')}
{fund_flow_section}

【分析要求】

请你**基于上述近20个交易日的完整资金流向数据**，从以下角度进行深入分析：

1. **资金流向趋势分析** ⭐ 重点
   - 分析近20个交易日主力资金的累计净流入/净流出
   - 识别资金流向的趋势性特征（持续流入、持续流出、震荡）
   - 计算主力资金净流入天数占比
   - 评估资金流向强度（累计金额、平均每日金额）

2. **主力资金行为分析** ⭐ 核心重点
   - **主力资金总体表现**：累计净流入金额、占比、趋势方向
   - **超大单分析**：机构大资金的进出动作
   - **大单分析**：主力资金的操作特征
   - **主力操作意图研判**：
     * 吸筹建仓：持续净流入 + 股价上涨/盘整
     * 派发出货：持续净流出 + 股价下跌/高位
     * 洗盘整理：震荡流入流出 + 股价调整
     * 拉升推动：集中大额流入 + 股价快速上涨

3. **散户资金行为分析**
   - **中单、小单的动向**：散户的买卖情绪
   - **主力与散户博弈**：
     * 主力流入、散户流出 → 专业资金吸筹
     * 主力流出、散户流入 → 高位接盘风险
     * 同向流动 → 趋势明确
   - 散户参与度和情绪判断

4. **量价配合分析**
   - 资金流向与股价涨跌的配合度
   - 识别量价背离：
     * 价涨量缩 + 资金流出 → 警惕顶部
     * 价跌量增 + 资金流入 → 可能见底
   - 成交活跃度变化趋势

5. **关键信号识别**
   - **买入信号**：
     * 主力持续净流入
     * 大单明显流入
     * 资金流入 + 股价上涨
   - **卖出信号**：
     * 主力持续净流出
     * 大额资金出逃
     * 资金流出 + 股价滞涨或下跌
   - **观望信号**：
     * 资金流向不明确
     * 主力与散户博弈激烈

6. **阶段性特征**
   - 早期阶段（前10个交易日）vs 近期阶段（后10个交易日）
   - 资金流向的变化趋势
   - 转折点识别

7. **投资建议**
   - 基于资金流向的操作建议
   - 关注重点和风险提示
   - 资金面对后市的指示意义
   - 未来资金流向预判

8. **投资建议**
   - 基于资金面的明确操作建议
   - 买入/持有/卖出的判断依据
   - 仓位管理建议

【分析原则】
- 主力资金持续流入 + 股价上涨 → 强势信号，主力看好
- 主力资金流出 + 股价上涨 → 警惕信号，可能是散户接盘
- 主力资金流入 + 股价下跌 → 可能是主力低位吸筹
- 主力资金流出 + 股价下跌 → 弱势信号，主力看空
- 注意区分短期波动与趋势性变化

请给出资金面分析报告。记住：要基于上面给出的实际数据进行分析，而不是假设！
{DATA_GUARD_RULES}
"""
        
        messages = [
            {"role": "system", "content":
                "你是一名经验丰富的资金面分析师，擅长市场资金流向和主力行为分析。"
                "你只能使用用户消息中给出的数据；未给出的要明确写「数据缺失」，"
                "不得引用任何未出现在用户消息中的数据、报告、机构或日期。"},
            {"role": "user", "content": prompt}
        ]
        
        return self.call_api(messages, temperature=TEMP_EXPLORATORY, max_tokens=3000,
                             purpose="client.comprehensive_discussion")
    
    def comprehensive_discussion(self, technical_report: str, fundamental_report: str, 
                               fund_flow_report: str, stock_info: Dict) -> str:
        """综合讨论"""
        prompt = f"""
现在需要进行一场投资决策会议，你作为首席分析师，需要综合各位分析师的报告进行讨论。

股票基本信息：
- 股票代码：{stock_info.get('symbol', 'N/A')}
- 股票名称：{stock_info.get('name', 'N/A')}
- 当前价格：{stock_info.get('current_price', 'N/A')}

技术面分析报告：
{technical_report}

基本面分析报告：
{fundamental_report}

资金面分析报告：
{fund_flow_report}

请作为首席分析师，综合以上三个维度的分析报告，进行深入讨论：

1. 各个分析维度的一致性和分歧点
2. 不同分析结论的权重考量
3. 当前市场环境下的投资逻辑
4. 潜在风险和机会识别
5. 不同投资周期的考量（短期、中期、长期）
6. 市场情绪和预期管理

请模拟一场专业的投资讨论会议，体现不同观点的碰撞和融合。
"""
        
        messages = [
            {"role": "system", "content": "你是一名资深的首席投资分析师，擅长综合不同维度的分析形成投资判断。"},
            {"role": "user", "content": prompt}
        ]
        
        return self.call_api(messages, temperature=TEMP_CONSERVATIVE, max_tokens=6000,
                             purpose="client.final_decision")
    
    def final_decision(self, comprehensive_discussion: str, stock_info: Dict, 
                      indicators: Dict) -> Dict[str, Any]:
        """最终投资决策"""
        prompt = f"""
基于前期的综合分析讨论，现在需要做出最终的投资决策。

股票信息：
- 股票代码：{stock_info.get('symbol', 'N/A')}
- 股票名称：{stock_info.get('name', 'N/A')}
- 当前价格：{stock_info.get('current_price', 'N/A')}

综合分析讨论结果：
{comprehensive_discussion}

当前关键技术位：
- MA20：{indicators.get('ma20', 'N/A')}
- 布林带上轨：{indicators.get('bb_upper', 'N/A')}
- 布林带下轨：{indicators.get('bb_lower', 'N/A')}

请给出最终投资决策，必须包含以下内容：

1. 投资评级：买入/持有/卖出
2. 目标价位（具体数字）
3. 操作建议（具体的买入/卖出策略）
4. 进场位置（具体价位区间）
5. 止盈位置（具体价位）
6. 止损位置（具体价位）
7. 持有周期建议
8. 风险提示
9. 仓位建议（轻仓/中等仓位/重仓）

请以JSON格式输出决策结果，格式如下：
{{
    "rating": "买入/持有/卖出",
    "target_price": "目标价位数字",
    "operation_advice": "具体操作建议",
    "entry_range": "进场价位区间",
    "take_profit": "止盈价位",
    "stop_loss": "止损价位",
    "holding_period": "持有周期",
    "position_size": "仓位建议",
    "risk_warning": "风险提示",
    "confidence_level": "信心度(1-10分)"
}}
"""
        
        messages = [
            {"role": "system", "content": "你是一名专业的投资决策专家，需要给出明确、可执行的投资建议。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.call_api(messages, temperature=TEMP_ANALYTICAL, max_tokens=4000)
        
        try:
            # 尝试解析JSON响应
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                decision_json = json.loads(json_match.group())
                return decision_json
            else:
                # 如果无法解析JSON，返回文本响应
                return {"decision_text": response}
        except:
            return {"decision_text": response}
