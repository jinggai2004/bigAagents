"""让模型**直读原文**做逐条立场标注，再由**代码**把标签算成数字。

## 为什么需要这一层（它和 `sentiment_reference.classify_stance` 的分工）

`classify_stance` 是**词表 + 正则**，零成本、可复现，但对下列结构**没有分辨能力**：

| 原文 | 词表判定 | 真实含义 |
|---|---|---|
| `我降了30%的仓位，满仓满融压力太大` | 看多（命中"满仓"） | 减仓 |
| `别问，已梭哈完` | 看空（"别"被当否定） | 满仓看多 |
| `做高抛低吸等待下个牛市周期` | 看多（命中"低吸"） | 在讲风格 |
| `删帖➕限流两件套了。明天再杀半天会反弹` | 看多（命中"反弹"） | 在讲平台与走势 |

这些不是"词表不够长"能解决的 —— 它们是**语义结构**问题（描述过去状态、
反讽、条件句、平台吐槽）。所以需要一个能读完整句子的标注器。

## 两条硬约束（本模块的全部价值都在这两条上）

**① 数字必须由代码算，不能由模型报。**
提示词明确禁止模型写百分比/条数；即使它写了，解析时也**整段丢弃**
（`_strip_numbers` 只用于告警，不用于取值）。所有计数、比例、置信区间
一律由 `sentiment_reference.build_snapshot(labels=...)` 从标签重算。
理由：模型自报的统计**无法复算、无法归因**，一旦进了报告就成了"看起来
精确的编造"。

**② 每个标签必须带可在原文中逐字命中的引用（quote）。**
`quote_ok()` 是**确定性校验器**：把 quote 与原文去空白后做子串匹配。
- 命中率 < `MIN_QUOTE_HIT` ⇒ 整块标签**作废**（不是"降级"）—— 说明模型
  在改写而非引用，那它的标签也不可信；
- 单条不符合只记 `quote_ok=False`，标签仍保留（避免格式问题吃掉覆盖）。

这条是本模块唯一能拦住"模型编造立场"的机制。没有它，`bull_ratio` 就是
一个不可核对的数字。

## 与既有铁律的关系

- **外部观点不参与评分**：本模块产出仍只走 `sentiment_reference` 的参照通道
  （复盘"市场情绪参照" / 页面展示 / `tool_market_sentiment`），
  不进入 `FACTOR_WEIGHTS`、不进选股打分。隔离由 `test_sentiment_isolation` 守。
- **缺就是缺**：模型没标到的条目记 `None` ⇒ 计入 `unscored`，**不回落词表**。
  同一份快照里混两个标注器 = 口径分叉（本项目已因同类问题吃过亏）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime

from core.sentiment_reference import MIN_SCORED_OK, STANCE_CODES, item_text

_logger = logging.getLogger(__name__)

# 提示词版本：改提示词**必须**改这个串。它随标签一起落库，
# 否则"两周后的标签和两周前的标签"混在一个历史序列里比，无人能查出差异来源。
PROMPT_VERSION = "sentillm-v1"

# 单次调用喂多少条。30 条 × 约 200 字 ≈ 4k tokens 输入，flash 模型一次返回
# 约 30 条标签 + 一段报告；再大就容易出现"后半段漏标"（实测最典型的截断形态）。
LABEL_CHUNK = 30

# 单次标注的总条数上限（防手滑把 500 条全塞进来）
MAX_POSTS = 120

# 引用至少几个字才算"可核对"。太短的引用（"满仓"）在长文里必然命中，
# 等于没校验；但太严会把短帖判死，所以对短帖放宽到 min(4, len(text))。
QUOTE_MIN_CHARS = 4

# 整块引用的命中率下限。低于它 ⇒ 整块作废。
MIN_QUOTE_HIT = 0.5

# 报告里不允许出现的统计数字（模型自己写统计 = 越权）
# ⚠️ 必须同时覆盖**中文数字**：实测模型更爱写"约七成看多"而不是"70%"。
#    只匹配阿拉伯数字是漏的（自检时 `_strip_numbers` 没拦住"约七成看多"）。
_CN_NUM = "零一二三四五六七八九十百千两几"
_NUMERIC_CLAIM = re.compile(
    rf"(?:\d+(?:\.\d+)?\s*(?:%|％|条|篇|个|倍|成))"
    rf"|(?:[{_CN_NUM}]+\s*成(?:多|左右)?)"
    rf"|(?:[{_CN_NUM}]+分之[{_CN_NUM}]+)"
    rf"|(?:[一二三四五六七八九十几]+(?:条|篇|只|成))"
)

# 叙述方向词（用于"报告与统计是否矛盾"的交叉校验）
_BULLISH_WORDS = ("偏多", "看多", "乐观", "积极", "亢奋", "追涨", "热度高", "一边倒看涨")
_BEARISH_WORDS = ("偏空", "看空", "悲观", "谨慎", "恐慌", "割肉", "离场", "冷清", "一边倒看跌")

_LABEL_PROMPT = """你在做一次**逐条立场标注**，不是写读后感。

下面是 @@N@@ 条公开发言（散户/博主），每条带编号与正文。

## 任务一：逐条判方向

每条必须给四项：
- `i`：编号（整数，**必须**与输入编号一致，不要重新编号）
- `stance`：只能取 `bull` / `bear` / `neutral` / `unscored`
  - `bull` = 看多、加仓、买入、认为要涨
  - `bear` = 看空、减仓、清仓、认为要跌
  - `neutral` = **明确表态不站边**（观望、持有不动、等回调再说）
  - `unscored` = 没有方向信息（贴图、打卡、纯提问、只报仓位不说看法、骂平台）
- `quote`：从该条正文**逐字复制**的一段（8~30 字），必须能在正文里原样找到。
  这一段就是这条标签的证据，后面会被程序逐字核对。
- `why`：不超过 20 字，说明判据（例：`降仓到30%`、`说反弹但主体是吐槽`）

## 任务二：整体观察（只要文字，不要数字）

- `主线`：这批人在讨论什么，一句话
- `多头理由`：最多 3 条
- `空头理由`：最多 3 条
- `争议焦点`：最多 3 条
- `异常信号`：与主流叙事相反的少数派声音，最多 2 条；没有就给空数组

## 铁律

1. `stance` 只能取那四个值，**不得自创**（不要 `看多`、`strong_bull`、`Bull`）。
2. `quote` 必须逐字复制，**不许改写、不许拼接、不许加省略号、不许翻译**。
   宁可短一点，也不能改字。
3. **不要在 `主线`/`理由`/`焦点` 里写任何百分比、条数、倍数** ——
   统计数字由程序计算，你写的会与实际计数不一致，等于污染。
4. 不要给操作建议，不要预测涨跌，不要推断主力/资金流向。
5. 只输出一个 JSON 对象，**不要** markdown 代码块，不要任何解释性文字。

## 输入

@@POSTS@@

## 输出格式

{"labels":[{"i":1,"stance":"bull","quote":"原文片段","why":"判据"}],
 "report":{"主线":"...","多头理由":["..."],"空头理由":["..."],
           "争议焦点":["..."],"异常信号":["..."]}}
"""

_MERGE_PROMPT = """下面是一次分批标注中各批的观察结果（同一批数据被切成多段）。请合并成一份。

要求：
1. 只合并、不新增。原文没有的理由**不得**编出来。
2. `主线` 用一句话概括各段的共同主题；若各段主题明显不同，就并列写出来。
3. **不要写任何百分比、条数、倍数**（统计数字由程序算）。
4. 不给操作建议，不预测涨跌。
5. 只输出 JSON：{"主线":"...","多头理由":["..."],"空头理由":["..."],"争议焦点":["..."],"异常信号":["..."]}

各批观察：
@@REPORTS@@
"""


# ================================================================
# 工具
# ================================================================

def _norm(s) -> str:
    """去空白。用于引用核对 —— 换行/空格差异不该算"引用不实"。"""
    return re.sub(r"\s+", "", str(s or ""))


def quote_ok(text: str, quote: str) -> bool:
    """引用是否能在原文中**逐字命中**（去空白后子串匹配）。

    ⚠️ 这是本模块唯一的确定性校验器。它拦不住"引用了原文但理解错了"，
       但能拦住"根本没读原文、照着标题编立场"这种更常见的失败。
    """
    t, q = _norm(text), _norm(quote)
    if not t or not q:
        return False
    if len(q) < min(QUOTE_MIN_CHARS, len(t)):
        return False
    return q in t


def post_key(it: dict, idx: int = 0) -> str:
    """条目的稳定键：**优先用 URL**，没有就取正文哈希。

    ⚠️ 不能用下标：同一批帖子明天重取顺序会变，下标作键会让"同一条发言"
       每天算成新记录，标签历史就变成了重复堆积。
    """
    url = str(it.get("url") or "").strip()
    if url:
        return "u:" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    body = _norm(item_text(it))
    if body:
        return "t:" + hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]
    return f"i:{idx}"


def _strip_numbers(report: dict) -> tuple[dict, list[str]]:
    """剔除报告里模型自报的统计数字（只留告警，**不留内容**）。

    ⚠️ 不是"提醒模型别写"就够了 —— 实测模型会写"约六成看多"。
       这里把含统计数字的**整条**删掉：删了只是少一句，留着就是与代码
       算出的数字并存的第二套口径。
    ⚠️⚠️ 告警里也**不能引用**被删掉的那句话。初版写成
       `已丢弃：{被删内容}`，结果页面上又原样出现了那个数字 ——
       等于换了个位置继续误导。只报**字段名**，不报内容。
    """
    warnings: list[str] = []
    if not isinstance(report, dict):
        return {}, warnings
    list_keys = ("多头理由", "空头理由", "争议焦点", "异常信号")
    out = {"主线": ""}
    line = str(report.get("主线") or "").strip()
    if line and _NUMERIC_CLAIM.search(line):
        warnings.append("报告「主线」含模型自报的统计数字，已整条丢弃（统计一律以代码计算为准）")
        line = ""
    out["主线"] = line
    for k in list_keys:
        kept = []
        for it in (report.get(k) or []):
            s = str(it or "").strip()
            if not s:
                continue
            if _NUMERIC_CLAIM.search(s):
                warnings.append(f"报告「{k}」有一条含模型自报的统计数字，已丢弃")
                continue
            kept.append(s)
        out[k] = kept[:4]
    return out, warnings


def report_consistency(report: dict, bull_ratio: float | None,
                       n_scored: int = 0) -> list[str]:
    """报告叙述方向 vs 代码算出的多空比 —— 矛盾就报警。

    ⚠️ 这是"先总结、再量化"这条链路上唯一能发现**两段互相打脸**的机制。
       没有它，页面会同时显示"模型总结：情绪偏空"和"多空比 78%"，读的人
       只会挑自己想信的那一半。有了它，至少能显式写一句"两者方向相反"。

    ⚠️ 只在**统计足够强**时才判矛盾（`n_scored >= MIN_SCORED_OK` 且
       `bull_ratio` 离 0.5 足够远）：样本薄时"叙述偏空、比例 0.45"完全正常，
       那不是矛盾，是噪声。

    Returns:
        告警列表（空表示没发现矛盾或不具备判断条件）。
    """
    if not isinstance(report, dict) or bull_ratio is None:
        return []
    if n_scored < MIN_SCORED_OK:
        return []

    parts = [str(report.get("主线") or "")]
    for k in ("多头理由", "空头理由", "争议焦点", "异常信号"):
        parts.extend(str(v or "") for v in (report.get(k) or []))
    text = " ".join(parts)
    n_bull = sum(1 for w in _BULLISH_WORDS if w in text)
    n_bear = sum(1 for w in _BEARISH_WORDS if w in text)

    out: list[str] = []
    if bull_ratio >= 0.65 and n_bear > n_bull:
        out.append(f"报告叙述偏空（{n_bear} 处偏空用词），但代码算出多空比 "
                   f"{bull_ratio:.0%} ⇒ 两者方向相反，以数字为准并核查标注")
    elif bull_ratio <= 0.35 and n_bull > n_bear:
        out.append(f"报告叙述偏多（{n_bull} 处偏多用词），但代码算出多空比 "
                   f"{bull_ratio:.0%} ⇒ 两者方向相反，以数字为准并核查标注")
    return out


def _extract_json(raw) -> dict | None:
    """从模型回复里抠出第一个 JSON 对象。

    ⚠️ 不能假定纯 JSON：实测会带 ```json 围栏或前置一句"好的"。
       `re.S` 让跨行匹配；`{.*}` 贪心到最后一个 `}`，对单对象够用。
    """
    text = str(raw or "").strip()
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        # 解析失败必须留原文，否则"为什么这批复用不上"永远查不出来
        _logger.warning("[情绪标注] 回复非合法 JSON: %s；原文前 200 字: %s", e, text[:200])
        return None
    return obj if isinstance(obj, dict) else None


# ================================================================
# 单块解析
# ================================================================

def parse_chunk(raw, rows: list, start: int = 0) -> dict:
    """一块的模型回复 → `{labels, quotes, report, warnings}`（**独立可单测**）。

    Args:
        raw: 模型原文
        rows: 这一块对应的条目（`start` 起的连续片段）
        start: 这一块在整体中的起始下标（0-based）

    Returns:
        `{"labels": {全局下标: stance}, "quotes": {全局下标: str},
          "quote_hit": {全局下标: bool}, "report": dict|None,
          "warnings": [...], "rejected": bool}`

    ⚠️ `rejected=True` 表示**整块作废**（引用命中率过低），此时 `labels` 为空。
       调用方必须把它记成"这些条目未标注"，而不是"这些条目中性"。
    """
    out = {"labels": {}, "quotes": {}, "quote_hit": {}, "whys": {},
           "report": None, "warnings": [], "rejected": False}
    obj = _extract_json(raw)
    if obj is None:
        out["warnings"].append("回复无法解析为 JSON ⇒ 本块未标注")
        out["rejected"] = True
        return out

    items = obj.get("labels")
    if not isinstance(items, list) or not items:
        out["warnings"].append("回复缺少 labels 数组 ⇒ 本块未标注")
        out["rejected"] = True
        return out

    n = len(rows)
    seen: set[int] = set()
    hits = 0
    for rec in items:
        if not isinstance(rec, dict):
            continue
        try:
            local = int(rec.get("i"))
        except (TypeError, ValueError):
            _logger.warning("[情绪标注] 编号非整数，已跳过：%r", rec.get("i"))
            out["warnings"].append(f"编号非整数，已跳过：{rec.get('i')!r}")
            continue
        # 模型偶尔从 0 开始编号；按 1-based 解释，越界直接丢
        gi = start + local - 1
        if not (start <= gi < start + n):
            out["warnings"].append(f"编号 {local} 越界（本块 {n} 条），已跳过")
            continue
        if gi in seen:
            out["warnings"].append(f"编号 {local} 重复，保留首次")
            continue
        seen.add(gi)

        stance = str(rec.get("stance") or "").strip().lower()
        if stance not in STANCE_CODES:
            out["warnings"].append(f"第 {local} 条 stance「{rec.get('stance')}」不在枚举内，记 unscored")
            stance = "unscored"
        out["labels"][gi] = stance
        out["whys"][gi] = str(rec.get("why") or "").strip()[:40]

        quote = str(rec.get("quote") or "").strip()
        ok = quote_ok(item_text(rows[gi - start]), quote)
        out["quotes"][gi] = quote
        out["quote_hit"][gi] = ok
        if ok:
            hits += 1

    out["report"] = obj.get("report") if isinstance(obj.get("report"), dict) else None

    # ── 整块信任判定 ───────────────────────────────────────────
    # ⚠️ 只在"有标签"时才判：0 条标签已经由上面的分支 rejected 了。
    if out["labels"]:
        rate = hits / len(out["labels"])
        out["quote_hit_rate"] = round(rate, 3)
        if rate < MIN_QUOTE_HIT:
            out["rejected"] = True
            out["warnings"].append(
                f"引用命中率 {rate:.0%} < {MIN_QUOTE_HIT:.0%} ⇒ 模型在改写而非引用，"
                f"本块 {len(out['labels'])} 条标签整体作废")
            out["labels"] = {}
            out["quotes"] = {}
            out["quote_hit"] = {}
            out["whys"] = {}
    return out


# ================================================================
# 报告合并
# ================================================================

def _local_merge(reports: list) -> dict:
    """不调模型的兜底合并：去重拼接。**不产生新内容**。"""
    keys = ("多头理由", "空头理由", "争议焦点", "异常信号")
    out = {"主线": "", **{k: [] for k in keys}}
    mains, seen = [], {k: set() for k in keys}
    for r in reports:
        if not isinstance(r, dict):
            continue
        m = str(r.get("主线") or "").strip()
        if m and m not in mains:
            mains.append(m)
        for k in keys:
            for v in (r.get(k) or []):
                s = str(v or "").strip()
                if s and s not in seen[k]:
                    seen[k].add(s)
                    out[k].append(s)
    out["主线"] = "；".join(mains)[:300]
    for k in keys:
        out[k] = out[k][:4]
    return out


def _merge_reports(reports: list, client) -> dict:
    """多块报告 → 一份。只有一块时直接返回，不额外花钱。"""
    reports = [r for r in reports if isinstance(r, dict)]
    if not reports:
        return {}
    if len(reports) == 1:
        return reports[0]
    prompt = _MERGE_PROMPT.replace(
        "@@REPORTS@@", json.dumps(reports, ensure_ascii=False, indent=1))
    try:
        raw = client.call_api(
            [{"role": "system", "content": "你只做信息合并，不新增内容。"},
             {"role": "user", "content": prompt}],
            temperature=0.0, purpose="sentiment.label")
        obj = _extract_json(raw)
        if isinstance(obj, dict) and obj.get("主线"):
            return obj
        _logger.warning("[情绪标注] 报告合并回复不可用，退回本地拼接")
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[情绪标注] 报告合并调用失败，退回本地拼接: %s: %s",
                        type(e).__name__, e)
    return _local_merge(reports)


# ================================================================
# 主入口
# ================================================================

def label_posts(items: list, client=None, chunk: int = None,
                max_posts: int = None, sleep: float = 0.0) -> dict:
    """对条目做**逐条 LLM 立场标注** + 一份定性报告。

    Args:
        items: 条目（字段宽容，见 `sentiment_reference.item_text`）
        client: LLM 客户端（缺省 `DeepSeekClient`）
        chunk: 单次条数，缺省 `LABEL_CHUNK`
        max_posts: 总条数上限，缺省 `MAX_POSTS`
        sleep: 块间休眠（默认 0；批量脚本可设 0.5 降触发限频的概率）

    ⚠️ `chunk` / `max_posts` 的默认值**在函数内解析**，不写成
       `def f(max_posts=MAX_POSTS)` —— 默认参数在定义时求值，
       一旦写成那样，`monkeypatch.setattr(SL, "MAX_POSTS", 3)` 就完全失效，
       表现为"配了但不生效"（本项目已被这个坑咬过一次）。

    Returns:
        ```
        {"labels": [stance|None, ...],      # 与 items[:max_posts] 逐位对齐
         "quotes": [str, ...],
         "quote_hit": [bool|None, ...],
         "report": {...} | {},
         "meta": {"model", "prompt_version", "n_posts", "n_labeled",
                  "chunks", "failed_chunks", "rejected_chunks",
                  "quote_hit_rate", "withheld_labeler": bool,
                  "errors": [...], "warnings": [...], "elapsed_s", "as_of"}}
        ```

    ⚠️ **`labels[i] is None` 表示"没标到"，不是"中性"**。消费方必须把它
       计入 `unscored`，**禁止**回落 `classify_stance` 补齐 —— 一份快照里
       混两个标注器，得到的 `bull_ratio` 既不是词表口径也不是模型口径。
    """
    rows = [it for it in (items or []) if isinstance(it, dict)][:int(max_posts or MAX_POSTS)]
    n = len(rows)
    started = time.time()
    out = {"labels": [None] * n, "quotes": [""] * n, "quote_hit": [None] * n,
           "whys": [""] * n, "report": {}, "meta": {
               "prompt_version": PROMPT_VERSION, "n_posts": n, "n_labeled": 0,
               "chunks": 0, "failed_chunks": 0, "rejected_chunks": 0,
               "quote_hit_rate": None, "withheld_labeler": False,
               "errors": [], "warnings": [],
               "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "elapsed_s": 0.0}}

    if n == 0:
        out["meta"]["errors"].append("无可标注条目")
        return out

    if client is None:
        try:
            from llm.deepseek_client import DeepSeekClient
            client = DeepSeekClient()
        except Exception as e:                  # noqa: BLE001
            _logger.warning("[情绪标注] 客户端初始化失败: %s: %s", type(e).__name__, e)
            out["meta"]["errors"].append(f"客户端初始化失败：{type(e).__name__}")
            return out

    size = max(1, int(chunk or LABEL_CHUNK))
    reports: list = []
    total_hit = total_lab = 0

    for start in range(0, n, size):
        block = rows[start:start + size]
        body = "\n\n".join(
            f"{i}. {item_text(it)[:200]}"
            for i, it in enumerate(block, start + 1))
        prompt = _LABEL_PROMPT.replace("@@N@@", str(len(block))).replace("@@POSTS@@", body)
        out["meta"]["chunks"] += 1
        try:
            raw = client.call_api(
                [{"role": "system",
                  "content": "你是严格的标注员。只按给定字段输出 JSON，不解释、不扩写。"},
                 {"role": "user", "content": prompt}],
                temperature=0.0, purpose="sentiment.label")
        except Exception as e:                  # noqa: BLE001
            _logger.warning("[情绪标注] 第 %d 块调用失败: %s: %s",
                            out["meta"]["chunks"], type(e).__name__, e)
            out["meta"]["failed_chunks"] += 1
            out["meta"]["errors"].append(
                f"第 {out['meta']['chunks']} 块调用失败：{type(e).__name__}")
            continue

        parsed = parse_chunk(raw, block, start)
        out["meta"]["warnings"].extend(parsed["warnings"])
        if parsed["rejected"]:
            out["meta"]["rejected_chunks"] += 1
            continue
        for gi, stance in parsed["labels"].items():
            out["labels"][gi] = stance
            out["quotes"][gi] = parsed["quotes"].get(gi, "")
            out["quote_hit"][gi] = parsed["quote_hit"].get(gi)
            out["whys"][gi] = parsed["whys"].get(gi, "")
        total_lab += len(parsed["labels"])
        total_hit += sum(1 for v in parsed["quote_hit"].values() if v)
        if parsed["report"]:
            reports.append(parsed["report"])
        if sleep:
            time.sleep(float(sleep))

    out["meta"]["n_labeled"] = sum(1 for s in out["labels"] if s is not None)
    if total_lab:
        out["meta"]["quote_hit_rate"] = round(total_hit / total_lab, 3)

    # ── 报告：剔除自报统计后合并 ────────────────────────────────
    cleaned = []
    for r in reports:
        c, w = _strip_numbers(r)
        out["meta"]["warnings"].extend(w)
        if c.get("主线") or any(c.get(k) for k in ("多头理由", "空头理由", "争议焦点")):
            cleaned.append(c)
    out["report"] = _merge_reports(cleaned, client)

    # ── 覆盖不足 ⇒ 拒绝给出数字 ────────────────────────────────
    # ⚠️ 标不到一半就报"多空比"，等于用 1/3 的样本代表整体。
    #    这里只**标记**不阻断（是否可用由 build_snapshot 的 quality 决定），
    #    但标记必须显式传给下游，不能静默。
    if out["meta"]["n_labeled"] and out["meta"]["n_labeled"] < n * 0.5:
        out["meta"]["withheld_labeler"] = True
        out["meta"]["warnings"].append(
            f"仅标注 {out['meta']['n_labeled']}/{n} 条（<50%）⇒ 该批数字不可代表整体")

    out["meta"]["model"] = getattr(client, "model", None) or getattr(client, "model_name", None)
    out["meta"]["elapsed_s"] = round(time.time() - started, 2)
    return out


# ================================================================
# 双轨对照：模型 vs 词表（**同一批条目**上比）
# ================================================================

def agreement(rule_labels: list, llm_labels: list, rows: list = None) -> dict:
    """词表标签 vs 模型标签的对照（同一批条目，逐位对齐）。

    ⚠️ 为什么**只在同一批条目上比**，而不是跨日比两套历史：
       跨日比会把"样本不同"混进"标注器不同"，得不出任何结论。
       （同源教训：`candidates` vs `ranked` 那次把两层混成一个榜单。）

    Returns:
        ```
        {"n": 条数,
         "n_both_directional": 两边都判出方向,
         "same_rate": 其中同向占比,          # ← 唯一有可比性的"一致率"
         "rule_only": 词表判出、模型未判,
         "llm_only": 模型判出、词表未判,      # ← 模型的"增量覆盖"
         "conflicts": [{"text","rule","llm","quote"}...]}   # 最多 5 条
        ```
    """
    rows = rows or []
    both = same = rule_only = llm_only = 0
    conflicts = []
    n = min(len(rule_labels), len(llm_labels))
    for i in range(n):
        r, l = rule_labels[i], llm_labels[i]
        r_dir = r in ("bull", "bear")
        l_dir = l in ("bull", "bear")
        if r_dir and l_dir:
            both += 1
            if r == l:
                same += 1
            elif len(conflicts) < 5:
                conflicts.append({
                    "text": (item_text(rows[i])[:110] if i < len(rows) else ""),
                    "rule": r, "llm": l})
        elif r_dir and not l_dir:
            rule_only += 1
        elif l_dir and not r_dir:
            llm_only += 1
    return {
        "n": n,
        "n_both_directional": both,
        "same_rate": round(same / both, 4) if both else None,
        "rule_only": rule_only,
        "llm_only": llm_only,
        "conflicts": conflicts,
    }


if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    # 自检：解析器对典型脏回复的处理（不打网络）
    demo_rows = [
        {"text": "我降了30%的仓位，满仓满融的压力太大"},
        {"text": "别问，已梭哈完"},
        {"text": "删帖➕限流两件套了。明天再杀半天会反弹"},
    ]
    fake = json.dumps({
        "labels": [
            {"i": 1, "stance": "bear", "quote": "降了30%的仓位", "why": "减仓"},
            {"i": 2, "stance": "Bull", "quote": "已梭哈完", "why": "满仓"},
            {"i": 3, "stance": "bull", "quote": "这句话原文里没有", "why": "反弹"},
        ],
        "report": {"主线": "仓位分歧", "多头理由": ["会反弹"], "空头理由": [],
                   "争议焦点": ["回调深度"], "异常信号": ["约七成看多"]},
    }, ensure_ascii=False)
    r = parse_chunk(fake, demo_rows, 0)
    print(json.dumps({k: v for k, v in r.items() if k != "warnings"},
                     ensure_ascii=False, indent=1))
    print("warnings:")
    for w in r["warnings"]:
        print("  -", w)
    print("\nquote_ok 自检:")
    for t, q in [("满仓了，看好", "看好"), ("满仓了，看好", "满仓满融"), ("降了30%的仓位", "降了 30% 的仓位")]:
        print(f"  {q!r:18s} in {t!r:20s} -> {quote_ok(t, q)}")
