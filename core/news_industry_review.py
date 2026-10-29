"""新闻行业利好/利空评估 + 利好个股推荐（热点新闻工作流，2026-10-09）。

工作流（用户定的设计）：
  1. 采集：`hourly_news` 已每小时入库全市场快讯（`news_items`）；
  2. 聚合：最近 24h 快讯按行业关键词归类（`data/news_feed._SECTOR_KEYWORDS` 口径）；
  3. 评估：LLM 对每个行业判定 利好/利空/中性 + 理由 + **证据**（引用库内新闻 id）；
  4. 推荐：利好行业的成分股，按最近资金流排序，给出"建议深度分析"清单。

三条铁律在本模块的落法：
  - **防幻觉（引用可追溯）**：LLM 返回的 `evidence_ids` 由**代码校验**——
    只允许出现在当日输入里的新闻 id，编造的一律剔除并记入 warnings；
    verdict 不在枚举内 → 降为中性并留痕。审计原则：我们无法阻止模型编造，
    但必须让"编造"可见、且不进库。
  - **缺就是缺**：某行业快讯不足 → 不评估（不硬判）；LLM 失败 → 返回
    `llm_error`，**不落库、不用旧结论充数**；行业成分拉不到 → 推荐降级并
    明确标 `degraded`，绝不静默用硬编码池。
  - **数据日期 ≠ 查询时间**：落库用 `review_date`（数据所属日期），
    与 `created_at` 分开。
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from datetime import datetime

from data.news_feed import _SECTOR_KEYWORDS
from db import store
# ⚠️ 温度档位必须在**模块级**导入。原先写成 `if client is None:` 分支里的局部 import，
# 而 `temperature=TEMP_ANALYTICAL` 在该分支之外使用 —— Python 一旦在函数体内出现
# import/赋值，该名字在整个函数内即为局部变量 ⇒ 调用方注入 client 时不走那个分支，
# 于是 `TEMP_ANALYTICAL` 未绑定，抛 UnboundLocalError 并被 except 吞成 "llm_error"
# （2026-10-13 实测：7 个 test_news_industry_review 用例因此全红，正是它们抓到的）。
from llm.deepseek_client import TEMP_ANALYTICAL

_logger = logging.getLogger(__name__)

VERDICTS = ("利好", "利空", "中性")
# 每行业给 LLM 的快讯条数上限：头条更能代表行业动向，且控制 token 成本
MAX_ITEMS_PER_SECTOR = 30
# 快讯少于此数的行业不评估（3 条以下的"行业动向"基本是噪声）
MIN_NEWS_PER_SECTOR = 3
# 单次评估的行业数上限（2026-10-28 阶段 A：关键词表从 5 个扩到 11 个大类，
# 原 12 已贴边 ⇒ 抬到 16 留余量。**扩词时必须重算这个数**：
# 上限一旦小于大类数，排在后面的行业会被静默截断，表现为"某些行业永远不评估"。）
MAX_SECTORS = 16


# ------------------------------------------------------------------
# ① 聚合
# ------------------------------------------------------------------

def classify_sector(text: str) -> str | None:
    """按关键词把一段文本归入行业（命中词最多的行业；无命中 → None）。

    与 `news_feed._enrich` 同一关键词表，但那里是"任一命中即算"，
    这里取**命中数最多**的行业（新闻常同时提多个行业，取最强信号）。
    """
    counts: dict[str, int] = {}
    for sector, words in _SECTOR_KEYWORDS.items():
        n = sum(1 for w in words if w in text)
        if n:
            counts[sector] = n
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def aggregate_recent_news(hours: int = 24,
                          min_news: int = MIN_NEWS_PER_SECTOR) -> list[dict]:
    """最近 N 小时快讯按行业聚合。

    Returns: [{sector, news_count, positive, negative,
               items: [{id, title, published_at, sentiment}]}, ...]
    （items 按 published_at 倒序，截到 MAX_ITEMS_PER_SECTOR 条）
    """
    rows = store.get_recent_news(limit=5000, hours=hours)
    buckets: dict[str, list] = {}
    for r in rows:
        title = (r.get("title") or "").strip()
        content = (r.get("content") or "").strip()
        text = f"{title} {content}"
        sector = classify_sector(text)
        if not sector:
            continue
        # ⚠️ 实测部分源的 title 为空（content 为主）——证据清单里空标题没有信息量，
        #    用正文前 60 字替代，否则 LLM 与用户都看不到"这条证据说了什么"。
        shown_title = title or (content[:60] + ("…" if len(content) > 60 else ""))
        buckets.setdefault(sector, []).append({
            "id": r["id"], "title": shown_title,
            "published_at": r.get("published_at") or "",
            "sentiment": r.get("sentiment") or "neutral"})
    out = []
    for sector, items in buckets.items():
        items.sort(key=lambda x: x["published_at"], reverse=True)
        out.append({
            "sector": sector,
            "news_count": len(items),
            "positive": sum(1 for x in items if x["sentiment"] == "positive"),
            "negative": sum(1 for x in items if x["sentiment"] == "negative"),
            "items": items[:MAX_ITEMS_PER_SECTOR]})
    # 条数不足的行业**不评估**——但要在结果里可见（缺就是缺）
    out = [b for b in out if b["news_count"] >= min_news]
    out.sort(key=lambda b: b["news_count"], reverse=True)
    return out[:MAX_SECTORS]


# ------------------------------------------------------------------
# ② LLM 评估
# ------------------------------------------------------------------

def _extract_json_array(text: str):
    """从 LLM 回复里抠出 JSON 数组（容忍 ```json 围栏与前后闲话）。"""
    text = str(text or "").strip()
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("回复中没有 JSON 数组")
        text = text[start:end + 1]
    return json.loads(text)


def _review_prompt(sectors: list[dict]) -> str:
    blocks = []
    for b in sectors:
        lines = [f"### {b['sector']}（{b['news_count']} 条，"
                 f"词表情绪 正{b['positive']}/负{b['negative']}）"]
        lines += [f"- [{x['id']}] {x['title']}" for x in b["items"]]
        blocks.append("\n".join(lines))
    return (
        "你是行业研究员。下面是按行业聚合的最近24小时财经快讯。"
        "对每个行业判断：这些消息对**该行业A股上市公司整体**是利好、利空还是中性。\n"
        "要求：\n"
        "1. verdict 只能取「利好」「利空」「中性」；消息不足以判断时用「中性」，不要硬判。\n"
        "2. summary 不超过 80 字，写清判断依据。\n"
        "3. evidence_ids 只能从该行业列出的新闻 id 中选（挑最支撑结论的 1~5 条），"
        "**严禁编造不存在的 id**。\n"
        "4. 只输出 JSON 数组，不要任何其他文字。\n"
        "输出格式：\n"
        '[{"sector":"行业名","verdict":"利好","summary":"...","evidence_ids":[1,2]}]\n\n'
        + "\n\n".join(blocks))


def run_industry_review(hours: int = 24, review_date: str = None,
                        client=None) -> dict:
    """聚合 → LLM 评估 → 校验 → 落库。

    Returns:
        {"status": "ok"|"no_data"|"llm_error", "review_date", "saved",
         "sectors": [...], "warnings": [...], "error"(仅 llm_error)}
    LLM 失败**不落库**（旧结论不能冒充新结论）；调用方（调度/页面）必须把
    status 原样告诉用户。
    """
    sectors = aggregate_recent_news(hours=hours)
    review_date = (review_date or datetime.now().strftime("%Y-%m-%d"))
    if not sectors:
        # 诚实返回：不是"都中性"，是"没有足够快讯"
        return {"status": "no_data", "review_date": review_date,
                "saved": 0, "sectors": [], "warnings": [],
                "error": f"最近 {hours} 小时没有聚合到 ≥{MIN_NEWS_PER_SECTOR} 条快讯的行业"}

    if client is None:
        from llm.deepseek_client import DeepSeekClient
        client = DeepSeekClient()
    model_name = getattr(client, "model", "deepseek")
    try:
        raw = client.call_api(
            [{"role": "system", "content": "你是严谨的A股行业研究员，只依据给出的新闻判断，绝不编造。"},
             {"role": "user", "content": _review_prompt(sectors)}],
            # 利好/利空/中性三分类判定，且结果要过 evidence_ids 白名单校验 → 低温
            temperature=TEMP_ANALYTICAL,
            # ⚠️ 补上 purpose：此前这次调用在 llm_calls 里落到 unknown，
            #    出问题时无法按功能定位/单独配模型。
            purpose="news_industry.review")
    except Exception as e:                          # noqa: BLE001
        _logger.warning("[行业评估] LLM 调用失败: %s", e)
        return {"status": "llm_error", "review_date": review_date,
                "saved": 0, "sectors": [], "warnings": [],
                "error": f"{type(e).__name__}: {e}"}

    try:
        parsed = _extract_json_array(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return {"status": "llm_error", "review_date": review_date,
                "saved": 0, "sectors": [], "warnings": [],
                "error": f"LLM 回复无法解析为 JSON 数组: {e}；原文前200字: {str(raw)[:200]}"}

    # ── 代码层校验：verdict 枚举 + 证据 id 白名单（防幻觉的核心）──
    valid_ids: dict[str, set] = {b["sector"]: {x["id"] for x in b["items"]}
                                 for b in sectors}
    title_of: dict[int, str] = {x["id"]: x["title"]
                                for b in sectors for x in b["items"]}
    rows, warnings = [], []
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        sector = str(item.get("sector") or "").strip()
        if sector not in valid_ids:
            warnings.append(f"LLM 返回了未给出的行业「{sector}」—— 已丢弃")
            continue
        verdict = str(item.get("verdict") or "").strip()
        if verdict not in VERDICTS:
            warnings.append(f"{sector}: verdict「{verdict}」不在枚举内 —— 降为中性")
            verdict = "中性"
        raw_ids = item.get("evidence_ids") or []
        if not isinstance(raw_ids, list):
            raw_ids = [raw_ids]
        evidence, dropped = [], []
        for nid in raw_ids:
            try:
                nid = int(nid)
            except (TypeError, ValueError):
                dropped.append(str(nid)[:20])
                continue
            if nid in valid_ids[sector]:
                evidence.append({"id": nid, "title": title_of.get(nid, "")[:60]})
            else:
                # ⚠️ 编造/串行业的 id 一律剔除——但必须留痕，否则"看起来有据"
                dropped.append(str(nid))
        if dropped:
            warnings.append(f"{sector}: 剔除了 {len(dropped)} 个无效证据 id "
                            f"（不在该行业新闻清单里）: {dropped[:5]}")
        if not evidence:
            warnings.append(f"{sector}: 没有任何有效证据 id —— 结论仅基于全部新闻整体")
        rows.append({"sector": sector, "verdict": verdict,
                     "summary": str(item.get("summary") or "").strip()[:200],
                     "evidence": evidence, "news_count": valid_ids[sector].__len__()})

    if not rows:
        return {"status": "llm_error", "review_date": review_date,
                "saved": 0, "sectors": [], "warnings": warnings,
                "error": "LLM 回复解析后没有任何有效行业"}

    saved = store.save_news_reviews(review_date, rows, model=str(model_name))
    _logger.info("[行业评估] %s：%d 个行业落库，%d 条警告",
                 review_date, saved, len(warnings))
    return {"status": "ok", "review_date": review_date, "saved": saved,
            "sectors": rows, "warnings": warnings}


# ------------------------------------------------------------------
# ③ 利好个股推荐
# ------------------------------------------------------------------

# ⚠️ 口径桥接（2026-10-09 端到端实测踩到）：评估用的行业名来自
# `_SECTOR_KEYWORDS` 的**大类**（"金融"/"新能源"），而新浪行业成分的名称是
# **细分行**（"金融行业"/"发电设备"…）——直接等值匹配利好行业会得到 0 只成分股。
# 这里显式登记 大类 → 新浪行业名 的别名（与 core/sector.py::INDUSTRY_TO_SECTOR
# 的归属口径一致：房地产归金融、公用事业归新能源）。新增关键词大类时必须同步补。
SECTOR_TO_SINA_INDUSTRY = {
    "科技": ["电子信息", "电子器件", "传媒娱乐"],
    "金融": ["金融行业", "房地产"],
    "消费": ["酿酒行业", "食品行业", "商业百货", "家电行业", "酒店旅游",
             "汽车制造", "服装鞋类", "家具行业"],
    "新能源": ["发电设备", "电力行业", "供水供气"],
    "医药": ["生物制药", "医疗器械"],
    # ── 商品 / 周期（2026-10-28 阶段 A）──
    # ⚠️ 新浪只有 49 个行业，"有色金属"一筐装着铜+铝+黄金+稀土 ⇒
    #    这一层只能解决"能进流程"，解决不了"铜价涨别推荐黄金股"。
    #    精确到个股要靠 data/concept_members.py（阶段 B）的概念成分。
    "有色": ["有色金属"],
    "能源": ["石油行业", "煤炭行业"],
    "钢铁": ["钢铁行业"],
    "农业": ["农林牧渔"],
    "化工": ["化工行业", "农药化肥", "化纤行业"],
    "建材": ["水泥行业", "玻璃行业", "建筑建材"],
}


# ------------------------------------------------------------------
# ② 主题（概念）层：把"铜价涨"落到"铜"，而不是整个有色金属筐
# ------------------------------------------------------------------
# 阶段 A 只解决"进得了流程"：铜价新闻归到「有色」，而新浪「有色金属」一筐
# 装着铜+铝+黄金+稀土 ⇒ 推荐里会混进黄金股。这一层用东财题材概念把它收敛。
#
# ⚠️ **必须显式映射，不做模糊匹配**：`铜` 还会命中「铜缆高速连接」（41 只，
#    通信/PCB 概念），模糊匹配会把铜价新闻落到 PCB 股上。宁可不匹配（回退行业），
#    也不猜——这是"缺就是缺"在选股链路上的落法。
# 概念名以 `data/concept_members.json` 实测为准（1,041 个），改名会由
# `missing_theme_concepts()` 在推荐结果里报出来。
THEME_TO_CONCEPTS = {
    "铜": ["铜"],
    "铝": ["铝"],
    "稀土": ["稀土", "稀土永磁"],
    "黄金": ["黄金", "黄金概念"],
    "焦煤": ["焦煤", "煤炭开采"],
    "动力煤": ["动力煤", "煤炭开采"],
    "钢铁": ["钢铁", "普钢", "特钢Ⅱ", "特钢Ⅲ"],
    "铁矿石": ["钢铁", "普钢"],
    "生猪": ["生猪养殖", "猪肉概念"],
    "肉鸡": ["肉鸡养殖", "鸡肉概念"],
    "水泥": ["水泥", "水泥制造", "水泥制品"],
    "纯碱": ["纯碱"],
    "锂": ["锂", "锂矿概念"],
}

# 新闻触发词 → 主题。**只用长词组**（与 `_SECTOR_KEYWORDS` 同一条纪律）：
# 裸 `金` 会命中金融/基金，裸 `钢` 会命中钢琴，裸 `煤`/`铝` 虽较安全但不留例外。
_THEME_TRIGGERS = {
    "铜": ("铜价", "铜矿", "电解铜", "铜精矿", "沪铜", "精铜"),
    "铝": ("铝价", "电解铝", "氧化铝", "沪铝"),
    "稀土": ("稀土", "镨钕", "氧化镨钕"),
    "黄金": ("金价", "黄金", "伦敦金"),
    "焦煤": ("焦煤", "炼焦煤", "焦炭"),
    "动力煤": ("动力煤", "电煤"),
    "钢铁": ("螺纹钢", "热卷", "钢价", "钢坯", "粗钢"),
    "铁矿石": ("铁矿石", "铁矿砂"),
    "生猪": ("生猪", "猪价", "能繁母猪", "猪肉价"),
    "肉鸡": ("鸡苗", "毛鸡", "白羽鸡", "鸡肉价"),
    "水泥": ("水泥",),
    "纯碱": ("纯碱",),
    "锂": ("碳酸锂", "锂矿", "锂价", "氢氧化锂"),
}


def themes_in_text(text: str) -> list:
    """文本里命中的主题（按首次出现位置排序，可多个）。

    返回多个而不是一个：一条快讯可能同时提"铜价与铝价"，调用方按需取。
    """
    t = str(text or "")
    if not t:
        return []
    hits = []
    for theme, words in _THEME_TRIGGERS.items():
        pos = min((t.find(w) for w in words if w in t), default=-1)
        if pos >= 0:
            hits.append((pos, theme))
    hits.sort()
    return [theme for _pos, theme in hits]


def classify_theme(text: str):
    """最靠前的主题（与 `classify_sector` 同一入口风格）。无命中 → None。"""
    hits = themes_in_text(text)
    return hits[0] if hits else None


def codes_for_theme(theme: str, concepts: dict) -> set:
    """主题 → 成分股代码集合（精确概念名查表，查不到就是空集）。

    `concepts` 形如 `concept_members.get_concept_members()["concepts"]`：
    `{概念名: {"board_code": "BK0999", "codes": [...]}}'。
    """
    names = THEME_TO_CONCEPTS.get(theme) or ()
    hits: set = set()
    for name in names:
        slot = (concepts or {}).get(name) or {}
        for code in slot.get("codes") or ():
            hits.add(str(code))
    return hits


def missing_theme_concepts(concepts: dict) -> list:
    """别名表里"在最新概念数据中不存在"的概念名（东财改名 / 概念合并的探针）。

    返回空列表 = 别名表与线上数据一致；非空必须让调用方看见，否则会静默退化成
    "这个主题永远查不到成分股"。
    """
    if not concepts:
        return []
    return sorted({name for names in THEME_TO_CONCEPTS.values() for name in names
                   if name not in concepts})


def _dominant_theme(evidence, summary: str):
    """从"证据新闻标题 + 行业判断理由"里选出现最多的主题（并列时取更具体的）。"""
    counter: Counter = Counter()
    for item in evidence or ():
        title = item.get("title") if isinstance(item, dict) else None
        for theme in themes_in_text(title or ""):
            counter[theme] += 1
    for theme in themes_in_text(summary or ""):
        counter[theme] += 1
    if not counter:
        return None
    return max(counter.items(), key=lambda kv: (kv[1], len(kv[0])))[0]


def codes_for_sector(sector: str, code_ind: dict) -> set:
    """大类 → 成分股代码集合。`code_ind` 是 **{code: 行业名}**（与
    `industry_members.code_to_industry` 的产出一致）。

    显式别名优先 + 双向子串兜底。兜底是给"将来新增了大类却忘了登记别名"
    留的退路（'金融' in '金融行业' 成立；'其它行业' 不会误吸任何大类名）。

    ⚠️ 实测踩过一次方向反转：本函数最初按 {行业: [code]} 写，而调用方传的是
    {code: 行业}——测试里恰巧也造了行业键的假数据，两边一起错就"绿"了
    （鉴别力探针与被替换实现行为等价的教训）。改为按真实形状断言。
    """
    alias = set(SECTOR_TO_SINA_INDUSTRY.get(sector, ()))
    hits: set = set()
    for code, ind_name in (code_ind or {}).items():
        if ind_name in alias:
            hits.add(code)
        elif sector and (sector in ind_name or ind_name in sector):
            hits.add(code)
    return hits


def recommend_stocks(review_date: str = None, top_n: int = 5,
                     allow_network: bool = True) -> dict:
    """利好行业 → 成分股 → 按最近资金流排序 → 建议深度分析清单。

    行业成分来源（按优先级降级，**每一级都标 degraded**）：
      1. `data/industry_members`（新浪行业，全市场，磁盘缓存 24h）；
      2. `backtest_pool.json::industry_map`（120 只样本池，覆盖窄，仅兜底）。
    个股反查路径（阶段 B）：`BIGA_CONCEPT_PICK=1` 时先按"证据里的具体主题"
    （铜/焦煤/生猪…）走 `data/concept_members` 的概念成分，命中才用；未命中一律
    回退行业。概念与行业是两套口径 ⇒ 结果里**另立 `concept`/`pick_basis` 字段**，
    不并进 `industry`。

    `allow_network=False`（Web 只读端点用）：成分映射**只读磁盘缓存**，绝不联网刷新
    （全量行业 49 次请求≈1 分钟、概念 20 页≈52 秒，放进 `/api/*` 就是长尾延迟）。
    无缓存时按既有降级链走（backtest_pool 兜底 + warnings 说明）——缺就是缺，不造假。
    排序依据：`fund_flow_snapshot` 最新一天的 `net_3d`（3 日主力净流入）。
    资金流也没有的行业 → 成分股按代码自然序给出并标注"无资金流排序"。
    """
    from data import industry_members as im

    date, rows = store.get_news_reviews(review_date)
    if date is None:
        return {"status": "no_reviews", "review_date": None, "picks": [],
                "degraded": True,
                "note": "还没有行业评估结果（评估任务每天 15:55 运行）"}
    bullish = [r for r in rows if r.get("verdict") == "利好"]
    if not bullish:
        return {"status": "no_bullish", "review_date": date, "picks": [],
                "degraded": False,
                "note": f"{date} 的评估中没有「利好」行业（共 {len(rows)} 个行业）"}

    warnings: list[str] = []
    try:
        # 只读模式（Web 端点）：只认磁盘缓存，宁可降级也不在请求里联网
        members = (im.get_industry_members() if allow_network
                   else im.load_cached_only())
        code_ind = im.code_to_industry(members["members"])
        if members.get("degraded"):
            warnings.append("行业成分数据不完整（部分行业拉取失败）："
                            + "；".join(members.get("failures", [])[:3]))
        if members.get("age_hours", 0) > im.CACHE_TTL_HOURS:
            warnings.append(f"行业成分是 {members.get('fetched_at')} 的旧缓存"
                            f"（本次刷新失败）")
    except im.IndustryMembersUnavailable as e:
        members, code_ind = None, None
        warnings.append(f"全市场行业成分不可用（{e}），降级为 120 只样本池 "
                        f"backtest_pool——覆盖窄，仅为参考")

    if code_ind is None:
        try:
            # ⚠️ 不能用相对路径 `open("data/backtest_pool.json")`：frozen 下 cwd 不是
            #    项目根 ⇒ 兜底池也读不到，最终表现为"两个来源都不可用"（exe 实测）。
            from config.paths import data_root as _data_root
            _pool_path = os.path.join(_data_root(), "data", "backtest_pool.json")
            pool = json.load(open(_pool_path, encoding="utf-8"))
            code_ind = pool.get("industry_map") or {}
        except (OSError, json.JSONDecodeError):
            return {"status": "no_members", "review_date": date, "picks": [],
                    "degraded": True, "warnings": warnings,
                    "note": "行业成分两个来源都不可用，无法从行业落到个股"}

    # ── 主题（概念）优先：把「有色」收敛到「铜」这类具体链条（阶段 B）──
    # 默认关（`BIGA_CONCEPT_PICK=1` 开启）。原因不是"没做完"，而是概念归属来自
    # 东财题材接口、与新浪 49 行业是两套口径，切换前要一轮灰度对照——与趋势因子
    # A+B 同一做法（默认关、可灰度、重启前行为不变）。
    concept_data: dict = {}
    if os.getenv("BIGA_CONCEPT_PICK", "0") == "1":
        from data import concept_members as cm
        try:
            got = (cm.get_concept_members() if allow_network
                   else cm.load_cached_only())
            concept_data = got.get("concepts") or {}
            if got.get("degraded"):
                warnings.append("概念成分数据不完整（部分页失败或为旧缓存）："
                                + (got.get("fetch_error")
                                   or "；".join(got.get("failures", [])[:2])))
            drift = missing_theme_concepts(concept_data)
            if drift:
                warnings.append(f"主题别名表里的 {drift[:4]} 在最新概念数据中不存在"
                                "（东财改名？请核对 THEME_TO_CONCEPTS），该主题已回退行业")
        except Exception as e:                  # noqa: BLE001 — 概念层是增强，失败不得拖垮推荐
            _logger.warning("[情报选股] 概念成分不可用，回退行业: %s: %s",
                            type(e).__name__, e)
            warnings.append(f"概念成分不可用（{type(e).__name__}: {e}），全部回退行业")

    # 最新一天的资金流（一次查询，内存分组）
    flow, flow_date = _latest_flow()
    if flow is None:
        warnings.append("没有资金流快照（15:40 任务未跑或刚清库），"
                        "成分股无法按资金流排序，按代码序给出")

    picks: list[dict] = []
    for r in bullish:
        sector = r["sector"]
        # 概念优先、行业兜底：先看这条评估的证据说了哪个具体主题（铜/焦煤/生猪…），
        # 有对应概念成分就用它；否则退回新浪行业（就是阶段 A 的行为）。
        theme, basis = None, "industry"
        codes: list = []
        if concept_data:
            theme = _dominant_theme(r.get("evidence"), r.get("summary"))
            if theme:
                by_theme = codes_for_theme(theme, concept_data)
                if by_theme:
                    codes, basis = sorted(by_theme), "concept"
        if not codes:
            theme = None
            codes = sorted(codes_for_sector(sector, code_ind))
        if not codes:
            warnings.append(f"{sector}: 行业别名与新浪行业成分都对不上"
                            "（如为新增大类，请在 SECTOR_TO_SINA_INDUSTRY 登记别名）")
            continue
        scored = []
        for c in codes:
            f = (flow or {}).get(c)
            scored.append({"stock_code": c, "stock_name": (f or {}).get("name", ""),
                           "net_3d": (f or {}).get("net_3d"),
                           "change_pct": (f or {}).get("change_pct"),
                           "flow_date": flow_date})
        scored.sort(key=lambda x: (x["net_3d"] if x["net_3d"] is not None
                                   else float("-inf")), reverse=True)
        for p in scored[:top_n]:
            p = dict(p)
            p["industry"] = sector
            # ⚠️ `concept` 另立字段，绝不并进 `industry`（两套口径混用会让
            #    SECTOR_TO_SINA_INDUSTRY 静默失效）。行业兜底时如实为 None。
            p["concept"] = theme
            p["pick_basis"] = basis
            p["sector_summary"] = r.get("summary") or ""
            p["evidence"] = r.get("evidence") or []
            picks.append(p)

    picks.sort(key=lambda x: (x["net_3d"] if x["net_3d"] is not None
                              else float("-inf")), reverse=True)
    return {"status": "ok", "review_date": date, "picks": picks,
            "degraded": bool(warnings), "warnings": warnings,
            "flow_date": flow_date,
            "note": "以上为「利好行业 × 近3日主力净流入」的排序参考，不构成交易指令；"
                    "点击后请以深度分析的三档建议（看多可执行/看空驳回/观望等待）为准"}


def _latest_flow() -> tuple[dict | None, str | None]:
    """fund_flow_snapshot 最新一天的 {code: {name, net_3d, change_pct}}。"""
    conn = store.get_connection()
    try:
        row = conn.execute(
            "SELECT MAX(snapshot_date) AS d FROM fund_flow_snapshot").fetchone()
        flow_date = row["d"] if row else None
        if not flow_date:
            return None, None
        rows = conn.execute(
            """SELECT stock_code, stock_name, net_3d, change_pct
               FROM fund_flow_snapshot WHERE snapshot_date = ?""", (flow_date,)
        ).fetchall()
        return ({r["stock_code"]: {"name": r["stock_name"],
                                   "net_3d": r["net_3d"],
                                   "change_pct": r["change_pct"]}
                 for r in rows}, flow_date)
    finally:
        conn.close()
