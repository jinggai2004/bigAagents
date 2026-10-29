"""
财经新闻快讯采集：新浪7x24 + 同花顺 + 东方财富 + 富途 四源降级链。

流程：daemon线程限时拉取 → 规范化 → hash去重 → 个股/板块关联 → 情绪标签 → SQLite落库。
每个源独立熔断（失败进冷却期，定时自愈探测）。

2026-10-26 换源：财联社(cls)退役。它的上游 `www.cls.cn/nodeapi/telegraphList`
已 404（接口迁移），而 akshare 侧既重试 10 次、又**不给请求设 timeout**，
导致每小时那趟采集必然白等满 30s 并泄漏一个阻塞线程（当日 7 次）。
接手的东财(em)/富途(futu) 实测 200 行/1.34s、50 行/1.35s，都是实时数据。
"""

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import akshare as ak
import requests

from db import store
from data.contracts import require_columns, DataContractError

_logger = logging.getLogger(__name__)

# ⚠️ 超时 12s 而不是 30s：单源实测全部 1.3s 级（2026-10-26 实测 sina/ths/em/futu
#    分别是 1.0/1.2/1.34/1.35s）。12s 已经给了 9 倍余量，再长就只是让"源真的挂了"
#    那种情况白等 —— 三源串行时 30s×3=90s，足以拖住整条每小时任务。
FETCH_TIMEOUT = 12          # 单源拉取限时（秒）
SOURCE_COOLDOWN = 1800      # 源熔断冷却（秒）
MAX_ITEMS_PER_RUN = 100     # 单次入库上限

# ── 给 akshare 的"裸 requests"补默认超时（见 `_fetch_source` 的注释）──
# thread-local 语义：**只有**在采集线程里主动设了 `_req_local.timeout` 才注入，
# 其它线程/其它模块的请求行为与改造前逐字节一致。
_req_local = threading.local()
_request_patch_installed = False
# 导入时抓一份"未被任何包装污染的"原始实现 —— 测试据此把钩子装到确定的位置，
# 不会因为"同进程里别的用例先前装过/卸过钩子"而看到不同的行为。
_ORIGINAL_SESSION_REQUEST = requests.Session.request


def _install_request_timeout(orig=None):
    """幂等地给 `requests.Session.request` 装上"按线程预算收紧超时"的钩子。

    Args:
        orig: 仅测试用 —— 显式指定被包装的底层实现。生产调用一律省略。
    Returns:
        装上去的钩子（也便于测试直接调用，不必绕 `requests.get` 全链路）。

    ⚠️⚠️ 与 `data/akshare_helper.patch_requests()` 的关系（**先查清再动手**）：
    项目里早就有一套同类补丁（`core/watcher`、`data/fund_flow`、`data/source_manager`、
    `data/stock_data` 在 import 时就调用了它），它做三件事：合并默认头 + 随机 UA +
    **注入 30s 默认超时** + 同源节流。所以"akshare 完全没超时"这个判断是错的 ——
    cls 的真实死因是「**30s 超时 × 内部 max_retries=10**」：最坏 300s，
    而外层 `join(30)` 只等 30s ⇒ 日志报 `>30s`，线程继续跑到 300s 才退（这才是泄漏）。
    ⇒ 本钩子因此采用 **min 语义**：只在"既有超时比预算更宽松"时收紧，
      不改动调用方显式设的更小值。装在 akshare_helper 的包装**外层**即可生效。
    """
    global _request_patch_installed
    if _request_patch_installed:
        return requests.Session.request
    orig = orig or requests.Session.request

    def _patched(self, method, url, **kwargs):
        budget = getattr(_req_local, "timeout", None)
        if budget:
            cur = kwargs.get("timeout")
            if cur is None:
                kwargs["timeout"] = budget
            elif isinstance(cur, (int, float)):
                kwargs["timeout"] = min(float(cur), float(budget))
            elif isinstance(cur, tuple) and len(cur) == 2:
                kwargs["timeout"] = (min(cur[0], budget), min(cur[1], budget))
            # 其它形状（如 urllib3 的 Timeout 对象）无法比较 ⇒ 原样透传
        return orig(self, method, url, **kwargs)

    requests.Session.request = _patched
    _request_patch_installed = True
    return _patched

# 源注册：解析器把 akshare 返回的 DataFrame 规范化为 [{title, content, published_at}]

def _parse_sina(df) -> list[dict]:
    """新浪7x24: 列[时间, 内容]，无标题"""
    require_columns(df, ["时间", "内容"], "新浪7x24")
    return [{"title": "", "content": str(r["内容"])[:500],
             "published_at": _norm_time(r["时间"])}
            for _, r in df.iterrows()]


def _parse_ths_generic(df) -> list[dict]:
    """同花顺/财联社: 列含[标题, 内容, 发布时间]"""
    # 这里原本全部用 r.get(...)：列名漂移时会静默产出"空标题+空内容"的记录，
    # 再去重成同一个 hash 的空条目，日志里毫无痕迹（v2/M0 0.4 加断言）
    require_columns(df, ["内容"], "同花顺/财联社")
    if not ({"标题", "发布时间"} & set(df.columns)):
        raise DataContractError(
            f"[同花顺/财联社] 既无'标题'也无'发布时间'列；实际列={list(df.columns)[:20]}")
    return [{"title": str(r.get("标题", ""))[:120], "content": str(r.get("内容", ""))[:500],
             "published_at": _norm_time(r.get("发布时间", ""))}
            for _, r in df.iterrows()]


def _parse_em(df) -> list[dict]:
    """东财全球快讯: 列[标题, 摘要, 发布时间, 链接]

    ⚠️ 东财把正文放在 **`摘要`** 而不是 `内容`（与同花顺/富途不同）。
    两个字段都试一次而不是死认一个 —— 列名漂移时宁可少拿也不要抛，
    但**必须**在 `require_columns` 里断言，否则会产出"只有标题没有正文"的空壳。
    """
    require_columns(df, ["标题", "发布时间"], "东财快讯")
    body_col = "摘要" if "摘要" in df.columns else ("内容" if "内容" in df.columns else None)
    if body_col is None:
        raise DataContractError(
            f"[东财快讯] 既无'摘要'也无'内容'列；实际列={list(df.columns)[:20]}")
    return [{"title": str(r.get("标题", ""))[:120],
             "content": str(r.get(body_col, ""))[:500],
             "published_at": _norm_time(r.get("发布时间", ""))}
            for _, r in df.iterrows()]


def _parse_futu(df) -> list[dict]:
    """富途牛牛快讯: 列[标题, 内容, 发布时间, 链接]"""
    require_columns(df, ["标题", "内容", "发布时间"], "富途快讯")
    return [{"title": str(r.get("标题", ""))[:120],
             "content": str(r.get("内容", ""))[:500],
             "published_at": _norm_time(r.get("发布时间", ""))}
            for _, r in df.iterrows()]


_SOURCES = {
    "sina": (ak.stock_info_global_sina, _parse_sina),
    "ths": (ak.stock_info_global_ths, _parse_ths_generic),
    # ── 2026-10-26 换源：cls 退役，em + futu 接手 ──
    # 财联社 `stock_info_global_cls` 实测三处致命问题，无法修复（在 akshare 内部）：
    #   ① 上游 `https://www.cls.cn/nodeapi/telegraphList` **已 404**（返回 HTML 错误页），
    #      接口迁移了，akshare 仍打旧地址；
    #   ② `make_request_with_retry_json(url, max_retries=10)` ⇒ 一个必 404 的地址重试 10 次；
    #   ③ 最要命的是它内部 `requests.get(url, params=..., headers=..., proxies=proxies)`
    #      **完全没有 timeout 参数** ⇒ 源不响应时该线程永久阻塞。
    #      外层的 `join(FETCH_TIMEOUT)` 只是"不再等"，**线程本身还在挂着** ——
    #      今天 7 次采集就是 7 个泄漏线程，且每次都白等满 30s（日志 7 条 `>30s`）。
    # 替代源实测（2026-10-26 17:09）：em 200 行/1.34s、futu 50 行/1.35s，都是实时数据。
    "em": (ak.stock_info_global_em, _parse_em),
    "futu": (ak.stock_info_global_futu, _parse_futu),
}

_circuit = {name: 0.0 for name in _SOURCES}

# 板块关键词（个股关联用，与 sector.py 六大板块对应）
#
# ⚠️ 2026-10-28（阶段 A，见 output/交接报告_股票池定性信息与情报面选股_2026-10-28.md）：
#    原先只有 5 个大行业（科技/金融/消费/新能源/医药），「铜价飞涨」这类商品/周期
#    新闻**一个词都命中不了** ⇒ sectors=[] ⇒ 根本进不了「行业→个股」反查。
#    这里补 6 个商品/周期大类，让主题先能进分类。
#
#    选词两条纪律（血泪）：
#      1) **不写裸单字**，除非该字在财经语境里几乎无歧义（铜/铝/煤 可以；金/银 不行
#         ——"金"会命中金融/基金/资金，"银"会命中银行；"钢"会命中钢琴）。
#      2) **缩写与歧义词走长词组**：不写"油"（酱油/食用油），写"原油/油价/成品油"；
#         不写"鸡"（鸡尾酒），写"毛鸡/鸡苗/白羽鸡"。
#    新增/删改词表**必须带单测**（tests/test_news_industry_review.py::TestSectorKeywords）。
_SECTOR_KEYWORDS = {
    "科技": ["半导体", "芯片", "AI", "人工智能", "算力", "软件", "消费电子", "光模块", "通信"],
    "金融": ["银行", "证券", "保险", "降准", "降息", "LPR", "央行"],
    "消费": ["白酒", "食品", "饮料", "零售", "消费", "汽车", "以旧换新"],
    "新能源": ["光伏", "锂电", "新能源", "储能", "风电", "电池"],
    "医药": ["医药", "创新药", "疫苗", "医疗器械", "CXO"],
    # ── 商品 / 周期 / 产业链（阶段 A 新增）──
    "有色": ["铜", "铝", "黄金", "白银", "稀土", "永磁", "钨", "钼", "锑", "镍",
             "锌", "锡", "钴", "铜箔", "电解铝", "氧化铝", "碳酸锂", "盐湖提锂"],
    "能源": ["原油", "油价", "石油", "天然气", "LNG", "煤", "焦煤", "动力煤",
             "柴油", "汽油", "OPEC", "页岩气"],
    "钢铁": ["钢铁", "钢材", "钢价", "螺纹钢", "铁矿石", "热卷", "冷轧", "不锈钢", "高炉"],
    "农业": ["生猪", "猪肉", "猪价", "能繁母猪", "毛鸡", "鸡苗", "白羽鸡", "鸡肉",
             "饲料", "种业", "转基因", "大豆", "玉米", "小麦", "养殖"],
    "化工": ["纯碱", "烧碱", "尿素", "磷矿", "钾肥", "乙烯", "丙烯", "MDI",
             "涤纶", "粘胶", "PVC", "氟化工", "钛白粉", "农药"],
    "建材": ["水泥", "浮法玻璃", "玻璃纤维", "石膏板", "混凝土", "建材"],
}

_POSITIVE_WORDS = ["上涨", "大涨", "涨停", "利好", "增长", "预增", "突破", "创新高", "中标", "获批", "回购", "增持"]
# ⚠️ 2026-10-21 补齐 4 个词：原先只有「下滑/预亏」，于是
#    「净利润同比下降 38.06%」「Q3 亏损 5231 万，由盈转亏」都被判成 **neutral**
#    —— 把利空当中性。对"利好选股"影响不大（漏掉的是利空），
#    但对**卖出提醒**是致命的（该报的利空报不出来）。
#
# ⚠️ 刻意**不加**「下调」：它有两个相反含义 —— "评级下调/目标价下调"是利空，
#    "下调存款准备金率/下调利率"是宽松利好。加进去会把后者误判成利空。
#    若将来要覆盖"评级下调"，应加更长的词组（"下调评级"/"下调目标价"）。
_NEGATIVE_WORDS = ["下跌", "大跌", "跌停", "利空", "下滑", "预亏", "处罚", "立案", "减持", "质押", "退市", "警告",
                   "下降", "亏损", "由盈转亏", "净流出"]

# 「良性下降」抑制（2026-10-22）：这些主语后面的"下降"是**好事**，不该算利空。
# ⚠️ 起因：补了"下降"之后，`毛利率提升，期间费用同比下降` 会被判 negative。
#    实测今日 599 条真实新闻里 24 条翻负、**0 条误伤**（都是真利空），
#    所以这是"机制上的隐患"而不是"已经发生的问题"——但成本极低，先堵住。
# ⚠️ 用正则而不是 `in`：主语和"下降"之间常隔几个字（"费用**同比**下降"）。
#    限定 6 个字以内，避免跨句误伤（"费用上升，但净利润下降"不该被抑制）。
_BENIGN_DROP = re.compile(
    r"(?:成本|费用|负债|库存|存货|应收|风险|支出|杠杆|能耗|采购价)"
    r"[^，。；！？\s]{0,6}?下降")

# 同一条新闻里"净利润下降"和"费用同比下降"可能同时出现：
# 只要还有**未被抑制**的负面词，就不该整体判中性。下面按"命中数"相减，
# 而 `_BENIGN_DROP` 最多抑制 1 次（一篇里出现多处良性下降也只算一次）。


def _neg_hits(text: str) -> int:
    """负面词命中数，扣掉"良性下降"的抑制量（最低 0）。

    ⚠️ 单独抽成函数是为了**可测**：抑制逻辑藏在 `_classify_sentiment` 里时，
       单测只能整段断言，看不出到底是哪个词生效。
    """
    neg = sum(1 for w in _NEGATIVE_WORDS if w in text)
    if neg and _BENIGN_DROP.search(text):
        neg -= 1
    return max(0, neg)


def _norm_time(value) -> str:
    """规范化时间为 '%Y-%m-%d %H:%M:%S'；解析失败返回当前时间"""
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
                "%Y年%m月%d日 %H:%M", "%m-%d %H:%M"):
        try:
            dt = datetime.strptime(text, fmt)
            if dt.year == 1900:  # 无年份格式，补当前年
                dt = dt.replace(year=datetime.now().year)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _make_hash(title: str, content: str, source: str) -> str:
    key = f"{title}|{content[:80]}|{source}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def _classify_sentiment(text: str) -> str:
    pos = sum(1 for w in _POSITIVE_WORDS if w in text)
    neg = _neg_hits(text)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _fetch_source(name: str) -> list[dict]:
    """单源拉取（daemon线程限时），返回规范化条目"""
    fn, parser = _SOURCES[name]
    result = {}

    def run():
        # ⚠️ 给 **本线程** 的 requests 注入默认超时。
        # 根因（2026-10-26）：akshare 的 `make_request_with_retry_json` 里是
        # `requests.get(url, params=..., headers=..., proxies=...)` —— **没有 timeout**。
        # 源不响应时该线程永久阻塞；外层 `join(FETCH_TIMEOUT)` 只是"不再等它"，
        # **线程本身还挂在 socket 上**。cls 今天 7 次采集 ⇒ 7 个泄漏线程。
        # 用 thread-local 而不是直接设全局：其它线程不设 `_req_local.timeout`
        # 时行为与改造前**完全一致**，不会误伤别处的长轮询。
        _install_request_timeout()
        _req_local.timeout = FETCH_TIMEOUT
        try:
            result["df"] = fn()
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)[:100]}"

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(FETCH_TIMEOUT)
    if "error" in result:
        raise ConnectionError(result["error"])
    if "df" not in result:
        raise TimeoutError(f">{FETCH_TIMEOUT}s")
    df = result["df"]
    if df is None or df.empty:
        return []
    items = parser(df)[:MAX_ITEMS_PER_RUN]
    for item in items:
        item["source"] = name
    return items


def classify_sentiment(text: str) -> str:
    """情绪标签（对外入口）。

    词表打分是 `data/news_feed` 的唯一实现：库内快讯入库时用它打标（`_enrich`），
    按个股直查的新闻没有入库环节，由工具层直接调用本函数，避免两套词表漂移。
    """
    return _classify_sentiment(text)


def _associate_stock(title: str, content: str, code: str, name: str) -> bool:
    text = f"{title} {content}"
    return bool(code and code in text) or bool(name and len(name) >= 2 and name in text)


def _enrich(items: list[dict]) -> list[dict]:
    """hash + 个股/板块关联 + 情绪标签

    个股关联用**关注池**（自选股 ∪ 今日候选 ∪ 当前持仓，见 data/watchlist.attention_pool）：
    原先只关联自选股，于是全市场候选的 related_codes 恒为空，
    "按代码查这只票的新闻"永远查不到，舆情因子只能靠简称在正文里出现来兜底。
    """
    from data import watchlist
    try:
        pool = watchlist.attention_pool()
    except Exception as e:                          # 关注池读不到不该中断采集
        _logger.warning("[新闻] 关注池读取失败，降级为自选股: %s", e)
        try:
            pool = watchlist.load_watchlist()
        except Exception:
            pool = []
    full_text = lambda item: f"{item.get('title', '')} {item.get('content', '')}"
    for item in items:
        text = full_text(item)
        item["hash"] = _make_hash(item.get("title", ""), item.get("content", ""), item.get("source", ""))
        related = [s["code"] for s in pool if _associate_stock(
            item.get("title", ""), item.get("content", ""), s.get("code", ""), s.get("name", ""))]
        sectors = [sec for sec, words in _SECTOR_KEYWORDS.items()
                   if any(w.lower() in text.lower() for w in words)]
        item["related_codes"] = related
        item["sectors"] = sectors
        item["sentiment"] = _classify_sentiment(text)
    return items


def fetch_and_store() -> dict:
    """
    拉取全部可用源并入库。

    Returns: {"sources": {源: 条数或错误}, "new": 新入库条数, "total_fetched": 拉取总数}
    """
    now = time.time()
    stats = {}
    all_items = []
    for name in _SOURCES:
        if now < _circuit[name]:
            stats[name] = f"熔断中(剩余{int((_circuit[name] - now) / 60)}分钟)"
            continue
        try:
            items = _enrich(_fetch_source(name))
            all_items.extend(items)
            stats[name] = len(items)
        except Exception as e:
            _circuit[name] = time.time() + SOURCE_COOLDOWN
            stats[name] = f"失败: {e}"
            _logger.warning("[新闻] %s 拉取失败: %s", name, e)
    new_count = store.insert_news(all_items) if all_items else 0
    try:
        from data import watchlist
        pool_size = len(watchlist.attention_pool())
    except Exception:
        pool_size = 0
    result = {"sources": stats, "new": new_count, "total_fetched": len(all_items),
              "pool_size": pool_size}
    _logger.info("[新闻] 采集完成: %s", result)
    return result


def get_latest_news(limit: int = 20, keyword: str = None, code: str = None,
                    hours: int = 72) -> list[dict]:
    """查询最近新闻（转发存储层）"""
    return store.get_recent_news(limit=limit, keyword=keyword, code=code, hours=hours)


STALE_THRESHOLD_MINUTES = 60  # 库内新闻超过该分钟数则先增量补采


def smart_latest_news(limit: int = 15, keyword: str = None, code: str = None,
                      hours: int = 72, name: str = None) -> dict:
    """
    带自动补采与关键词容错的新闻查询（AI 工具入口）。

    1. 库内最新新闻超过 STALE_THRESHOLD_MINUTES → 先增量采集一次（源熔断保护）
    2. keyword 过滤无结果 → 自动回退为返回最近新闻，并说明"未找到含该词的新闻"
    3. code 过滤无结果 → **明确说明检索范围**，而不是静默返回空表

    为什么第 3 条必要（2026-09-25 实测）：库内 242 条快讯、`related_codes` 全部为空
    （入库只关联"关注池 = 自选股∪今日候选∪持仓"），于是 `code=601166` 恒返回 0 条。
    模型拿到一个空的 news 列表、没有任何说明，就会回答"没有相关新闻"——
    把"我们没检索到"说成了"事实不存在"。传 `name` 可按正文提及简称兜底匹配。
    """
    stats = store.news_stats()
    latest = stats.get("latest")
    stale = True
    if latest:
        try:
            age_min = (datetime.now() - datetime.strptime(latest, "%Y-%m-%d %H:%M:%S")).total_seconds() / 60
            stale = age_min > STALE_THRESHOLD_MINUTES
        except ValueError:
            stale = True
    refreshed = False
    if stale:
        try:
            fetch_and_store()
            refreshed = True
        except Exception as e:
            _logger.warning("[新闻] 自动补采失败: %s", e)

    items = store.get_recent_news(limit=limit, keyword=keyword or None,
                                  code=code or None, hours=hours, name=name)
    note = None
    if not items and keyword:
        # 关键词过严（如"今天/最新"类时间词）无命中 → 回退最近新闻
        items = store.get_recent_news(limit=limit, code=code, hours=hours, name=name)
        note = f"未找到包含“{keyword}”的新闻，以下为最近快讯"
    if not items and (code or name):
        total = (stats.get("total") or 0)
        note = (f"库内 {total} 条快讯中未检索到与 {code or name} 关联的条目"
                f"（按代码关联只覆盖关注池：自选股∪今日候选∪持仓；按简称需正文出现）"
                f"。这不等于该股没有相关新闻，只是本库未命中")
    return {"news": items, "total": len(items), "auto_refreshed": refreshed, "note": note}


# ================================================================
# 按个股直查的新闻 / 公告源（v4，2026-09-28）
# ================================================================
# 为什么需要：上面的 sina / ths / cls 三源都是**全市场快讯**，按"热点/重要性"播报。
# 实测（2026-09-28）：库内 583 条快讯、今日 79 条，而按 `600211` / 「西藏药业」
# 匹配命中 **0 条** —— 冷门个股天然不会被这三条广播线提及。舆情节点于是长期报
# "本次未检索到"：虽然如实，但对分析毫无信息量。
#
# 这两个源是**按个股直查**的，实测成本极低（新闻 0.13~0.38s / 公告 1.2~1.4s），
# 而深度分析是 1~2 条/天、单标的的低频场景 → 实时抓取完全合理。
# （与批次 22「选股只读库」不冲突：那是全市场高频场景，必须盘后落库。）

EM_NEWS_SOURCE = "em_news"
EM_NOTICE_SOURCE = "em_notice"

STOCK_FETCH_TIMEOUT = 12        # 单次直查限时（秒）；akshare 自身无超时，卡住会拖死分析链
STOCK_CACHE_TTL = 300           # 进程内缓存：同一轮分析重入不重复抓
STOCK_MIN_ITEMS = 3             # 时间窗口内条数不足该值 → 放宽窗口
STOCK_WIDEN_DAYS = 30           # 一级放宽的窗口
NOTICE_LOOKBACK_DAYS = 90       # 公告检索窗口（天数）
NOTICE_MAX_ITEMS = 8

# 源级熔断（与全市场快讯源同构：失败进冷却，避免拖慢整图）
_stock_circuit: dict = {EM_NEWS_SOURCE: 0.0, EM_NOTICE_SOURCE: 0.0}
_stock_cache: dict = {}


def _em_rows_match_code(df, code: str) -> bool:
    """相关性闸门：返回的每一条都必须声明属于 `<code>`。

    ⚠️ **这是本模块最重要的一行防护**（2026-09-28 实跑抓出来的自身缺陷）：
    `ak.stock_news_em` 只认**裸 6 位代码**。带交易所后缀时它**不报错**，
    而是静默返回一条与该股完全无关的通用快讯流：

        600211        → 关键词列 == ['600211']，正文是西藏药业 ✅
        600211.SH     → 关键词列 == ['600211.SH']，正文是赛伦生物/力鼎光电 ❌

    也就是说"补 .BJ 后缀让北交所也能返回 10 条"这个直觉是**错的**——
    那 10 条只是通用流。这种"错数据"比"没数据"危险得多：模型会把
    「赛伦生物获菲律宾 GMP 证书」当成西藏药业的新闻来分析，且全程不报错。

    东财自己给的 `关键词` 列就是"这条新闻属于谁"的声明，用它做闸门，判据取**严格相等**：
    实测 9 个代码（600519/300750/002594/688981/430047/920002/600211/601166/000001）
    合法返回的 `关键词` 都恰好等于请求的裸代码。用"包含"会被 `'600211.SH'` 钻过去
    （`'600211' in '600211.SH'` 为真），等于闸门失效。

    宁可显式失败、也不要静默给错：若东财哪天改成"代码+简称"，这里会立刻报错并
    把实际收到的值写进错误信息，而不是放任混流进模型。
    """
    try:
        values = [str(v).strip() for v in df["关键词"].tolist()]
    except Exception:                               # noqa: BLE001 - 缺列由上面断言负责
        return False
    return bool(values) and all(v == code for v in values)


def _call_with_timeout(fn, timeout: int):
    """在 daemon 线程里调用（akshare 没有超时参数）"""
    result = {}

    def run():
        try:
            result["value"] = fn()
        except BaseException as e:        # noqa: BLE001 - 含 akshare 内部抛的 KeyError
            result["error"] = e

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if "error" in result:
        raise result["error"]
    if "value" not in result:
        raise TimeoutError(f"直查超过 {timeout}s 未返回")
    return result["value"]


_DT_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d")


def _parse_published(value):
    """解析发布时间；失败返回 None。

    **绝不像 `_norm_time` 那样在解析失败时回落到 `now()`** —— 那会让三条陈年新闻
    全部化身"刚刚发布"，而本函数的输出正是"是否放宽时间窗口"的判据。
    """
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _cache_get(source: str, code: str, days) -> dict | None:
    hit = _stock_cache.get((source, code, int(days or 0)))
    if not hit:
        return None
    ts, payload = hit
    if time.time() - ts > STOCK_CACHE_TTL:
        _stock_cache.pop((source, code, int(days or 0)), None)
        return None
    return json.loads(json.dumps(payload))          # 深拷贝：调用方改不动缓存


def _cache_put(source: str, code: str, days, payload: dict):
    _stock_cache[(source, code, int(days or 0))] = (
        time.time(), json.loads(json.dumps(payload)))


def clear_stock_cache():
    """清空直查缓存（测试与"强制刷新"用）"""
    _stock_cache.clear()
    for name in _stock_circuit:
        _stock_circuit[name] = 0.0


def _pick_with_window(items: list, days: int) -> tuple:
    """三级放宽：近 N 天 → 不足 STOCK_MIN_ITEMS 放宽到 30 天 → 仍不足取全量。

    为什么需要（实测 2026-09-28）：本源返回的是**近 1~1.5 个月**的 10 条，不是近几天。
    600211 的 10 条跨 08-14~09-21，若按 days=7 硬过滤只剩 1 条、601166 只剩 2 条 ——
    又会变回"近乎 0 条"的观感。放宽**必须写进 note 交给模型**，否则模型会把
    一个月前的新闻当成近期事件。
    """
    if not items:
        return [], "", None
    days = max(1, int(days or 7))
    cutoff = datetime.now() - timedelta(hours=days * 24)
    in_window = [it for it in items if it["_dt"] is None or it["_dt"] >= cutoff]
    if len(in_window) >= STOCK_MIN_ITEMS:
        picked, window = in_window, f"近{days}天"
    else:
        wide = datetime.now() - timedelta(days=STOCK_WIDEN_DAYS)
        wider = [it for it in items if it["_dt"] is None or it["_dt"] >= wide]
        if len(wider) >= STOCK_MIN_ITEMS:
            picked = wider
            window = f"近{days}天仅{len(in_window)}条，已放宽至近{STOCK_WIDEN_DAYS}天"
        else:
            picked = items
            window = f"近{days}天仅{len(in_window)}条，已取源返回的全部{len(items)}条"

    dated = [it["_dt"] for it in picked if it["_dt"]]
    span = ""
    if dated:
        newest, oldest = max(dated), min(dated)
        span = (f"；最近一条 {newest.strftime('%Y-%m-%d')}"
                f"（距今 {(datetime.now() - newest).days} 天）")
        if len(dated) > 1:
            span += f"，最早一条 {oldest.strftime('%Y-%m-%d')}"
    return picked, window, f"个股新闻源（东财）：{window}{span}"


def fetch_stock_news(code: str, days: int = 7, name: str = "",
                     limit: int = 10) -> dict:
    """**按个股直查**东财新闻（深度分析舆情节点的主源）。

    实测契约（akshare 1.18.64，2026-09-28）：
      调用 `ak.stock_news_em(symbol=...)`，**只传裸 6 位代码**；
      **0.13~0.38s / 10 条**（固定条数，非分页）；
      列 `关键词|新闻标题|新闻内容|发布时间|文章来源|新闻链接`；
      **返回的是近 1~1.5 个月**而不是近几天（见 `_pick_with_window`）。

    ⚠️ 两个实测坑（都不是"报错"，而是"静默给错东西"）：
      1. **不能带交易所后缀**：`'600211.SH'` 会静默返回与该股无关的通用快讯流
         （实测正文是赛伦生物/力鼎光电），所以有 `_em_rows_match_code` 闸门整批拒收；
      2. **北交所部分代码（871981/830799/832000）裸码直接抛 `KeyError: 'code'`**
         ——那是"该源没有这只票的新闻"的信号，不是源故障，**不进熔断**，
         否则一次北交所分析就能把新闻源关掉 30 分钟。

    Returns:
        {"items": [...], "window": str, "note": str | None,
         "source": "em_news", "error": str | None, "all_count": int, "cached": bool}
        失败时 `items=[]` 且 `error` 非空 —— **不抛异常**（降级链要靠它判断）。
    """
    code = str(code or "").strip()
    if not (code.isdigit() and len(code) == 6):
        return {"items": [], "window": "", "note": None, "source": EM_NEWS_SOURCE,
                "error": "股票代码须为6位数字", "all_count": 0}

    cached = _cache_get(EM_NEWS_SOURCE, code, days)
    if cached is not None:
        return dict(cached, cached=True)

    now = time.time()
    if now < _stock_circuit[EM_NEWS_SOURCE]:
        left = int((_stock_circuit[EM_NEWS_SOURCE] - now) / 60)
        return {"items": [], "window": "", "note": None, "source": EM_NEWS_SOURCE,
                "error": f"个股新闻源熔断中(剩余{left}分钟)", "all_count": 0}

    try:
        df = _call_with_timeout(
            lambda: ak.stock_news_em(symbol=code), STOCK_FETCH_TIMEOUT)
    except KeyError:
        # 该源没有这只票的新闻（实测 871981/830799/832000 走这里）。
        # **不进熔断**：这是"没有数据"，不是"源坏了"。
        out = {"items": [], "window": "", "source": EM_NEWS_SOURCE, "all_count": 0,
               "note": f"个股新闻源（东财）未收录 {code} 的新闻（源以 KeyError 表示无结果）",
               "error": None}
        _cache_put(EM_NEWS_SOURCE, code, days, out)
        return dict(out)
    except Exception as e:                          # noqa: BLE001
        msg = f"{type(e).__name__}: {str(e)[:80]}"
        _stock_circuit[EM_NEWS_SOURCE] = time.time() + SOURCE_COOLDOWN
        _logger.warning("[新闻] 个股直查失败 %s: %s", code, msg)
        return {"items": [], "window": "", "note": None, "source": EM_NEWS_SOURCE,
                "error": f"个股新闻源失败: {msg}", "all_count": 0}

    if df is None or getattr(df, "empty", True):
        out = {"items": [], "window": "", "source": EM_NEWS_SOURCE, "all_count": 0,
               "note": f"个股新闻源（东财）对 {code} 返回 0 条", "error": None}
        _cache_put(EM_NEWS_SOURCE, code, days, out)
        return dict(out)

    try:
        # 列名漂移要显式失败，而不是静默产出"空标题 + 空内容"的记录
        require_columns(df, ["关键词", "新闻标题", "发布时间"], "东财个股新闻")
    except DataContractError as e:
        _stock_circuit[EM_NEWS_SOURCE] = time.time() + SOURCE_COOLDOWN
        return {"items": [], "window": "", "note": None, "source": EM_NEWS_SOURCE,
                "error": str(e), "all_count": 0}

    if not _em_rows_match_code(df, code):
        # 宁可 0 条也不能把别的公司的新闻喂给模型
        kws = sorted({str(v) for v in df["关键词"].tolist()})[:3]
        _logger.warning("[新闻] 东财返回内容与请求代码不一致 %s: %s", code, kws)
        return {"items": [], "window": "", "note": None, "source": EM_NEWS_SOURCE,
                "error": (f"个股新闻源返回的条目声明为 {kws}，与请求代码 {code} 不一致"
                          "（该源只接受裸 6 位代码，带交易所后缀会返回无关通用流）"),
                "all_count": 0}

    items = []
    for _, r in df.iterrows():
        title = str(r.get("新闻标题") or "").strip()
        content = str(r.get("新闻内容") or "").strip()[:500]
        if not title and not content:
            continue
        items.append({
            "title": title[:120],
            "content": content,
            "source": str(r.get("文章来源") or "东方财富").strip() or "东方财富",
            "published_at": str(r.get("发布时间") or "").strip(),
            "url": str(r.get("新闻链接") or "").strip(),
            "_dt": _parse_published(r.get("发布时间")),
        })
    items.sort(key=lambda x: x["_dt"] or datetime.min, reverse=True)

    picked, window, note = _pick_with_window(items, days)
    for it in picked:
        it.pop("_dt", None)
    out = {"items": picked[:limit], "window": window, "source": EM_NEWS_SOURCE,
           "note": note, "error": None, "all_count": len(items)}
    _cache_put(EM_NEWS_SOURCE, code, days, out)
    return dict(out)


def fetch_stock_notices(code: str, days: int = NOTICE_LOOKBACK_DAYS,
                        limit: int = NOTICE_MAX_ITEMS) -> dict:
    """**按个股直查**东财公告。

    实测契约（akshare 1.18.64，2026-09-28）：
      `ak.stock_individual_notice_report(security=..., begin_date=..., end_date=...)`
      —— 参数名是 **`security` 不是 `symbol`**（传 symbol 直接 TypeError）；
      带日期窗口 **1.2~1.4s**，不带窗口 9.7~13.1s（拉全历史 1141~1659 条，所以必须带窗口）；
      列 `代码|名称|公告标题|公告类型|公告日期|网址`。

    ⚠️ **实测坑**：窗口内**没有任何公告**时 akshare 抛 `KeyError: '代码'`
    （600211 的 09-25~09-28 窗口复现）——那是"空结果"不是"接口坏了"。
    必须把它判为"该窗口无公告"且**不进熔断冷却**，否则每只安静的股票都会被记成
    源故障，冷�� 1800s 之后公告源就永远不可用了（经典的"降级链自己把自己关掉"）。
    """
    code = str(code or "").strip()
    if not (code.isdigit() and len(code) == 6):
        return {"items": [], "note": None, "source": EM_NOTICE_SOURCE,
                "error": "股票代码须为6位数字"}

    cached = _cache_get(EM_NOTICE_SOURCE, code, days)
    if cached is not None:
        return dict(cached, cached=True)

    now = time.time()
    if now < _stock_circuit[EM_NOTICE_SOURCE]:
        left = int((_stock_circuit[EM_NOTICE_SOURCE] - now) / 60)
        return {"items": [], "note": None, "source": EM_NOTICE_SOURCE,
                "error": f"个股公告源熔断中(剩余{left}分钟)"}

    days = max(1, int(days or NOTICE_LOOKBACK_DAYS))
    end = datetime.now()
    begin = end - timedelta(days=days)
    try:
        df = _call_with_timeout(
            lambda: ak.stock_individual_notice_report(
                security=code, begin_date=begin.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d")),
            STOCK_FETCH_TIMEOUT)
    except KeyError:
        out = {"items": [], "source": EM_NOTICE_SOURCE, "error": None,
               "note": f"个股公告源（东财）近{days}天无公告（源以 KeyError 表示空结果）"}
        _cache_put(EM_NOTICE_SOURCE, code, days, out)
        return dict(out)
    except Exception as e:                          # noqa: BLE001
        _stock_circuit[EM_NOTICE_SOURCE] = time.time() + SOURCE_COOLDOWN
        _logger.warning("[新闻] 个股公告直查失败 %s: %s", code, e)
        return {"items": [], "note": None, "source": EM_NOTICE_SOURCE,
                "error": f"个股公告源失败: {type(e).__name__}: {str(e)[:80]}"}

    if df is None or getattr(df, "empty", True):
        out = {"items": [], "source": EM_NOTICE_SOURCE, "error": None,
               "note": f"个股公告源（东财）近{days}天无公告"}
        _cache_put(EM_NOTICE_SOURCE, code, days, out)
        return dict(out)

    try:
        require_columns(df, ["代码", "公告标题", "公告日期"], "东财个股公告")
    except DataContractError as e:
        _stock_circuit[EM_NOTICE_SOURCE] = time.time() + SOURCE_COOLDOWN
        return {"items": [], "note": None, "source": EM_NOTICE_SOURCE, "error": str(e)}

    # 同源同款相关性闸门：只收声明属于本代码的条目（`代码` 列是源自己的声明）
    rows = [r for _, r in df.iterrows()
            if str(r.get("代码") or "").strip() == code]
    if not rows:
        got = sorted({str(r.get("代码") or "") for _, r in df.iterrows()})[:3]
        return {"items": [], "note": None, "source": EM_NOTICE_SOURCE,
                "error": f"个股公告源返回的条目代码为 {got}，与请求代码 {code} 不一致"}

    items = []
    for r in rows:
        title = str(r.get("公告标题") or "").strip()
        if not title:
            continue
        kind = str(r.get("公告类型") or "").strip()
        items.append({
            "title": title[:120],
            "content": kind[:60],
            "source": f"东财公告·{kind}" if kind else "东财公告",
            "published_at": str(r.get("公告日期") or "").strip(),
            "url": str(r.get("网址") or "").strip(),
            "_dt": _parse_published(r.get("公告日期")),
        })
    items.sort(key=lambda x: x["_dt"] or datetime.min, reverse=True)

    # 先算时间跨度再摘 _dt —— `picked` 与 `items` 是同一批 dict，
    # 顺序反了就是 KeyError
    dated = [it["_dt"] for it in items if it["_dt"]]
    picked = items[:limit]
    for it in picked:
        it.pop("_dt", None)
    span = ""
    if dated:
        newest = max(dated)
        span = (f"；最近一条 {newest.strftime('%Y-%m-%d')}"
                f"（距今 {(datetime.now() - newest).days} 天）")
    out = {"items": picked, "source": EM_NOTICE_SOURCE, "error": None,
           "note": f"个股公告源（东财）近{days}天共 {len(items)} 条{span}"}
    _cache_put(EM_NOTICE_SOURCE, code, days, out)
    return dict(out)


def news_sentiment_for_stock(code: str, days: int = 3, name: str = None) -> dict | None:
    """
    单股新闻情绪统计（选股舆情因子用）。
    返回 {"total", "positive", "negative", "positive_rate"}；无数据返回 None。

    v2（M0 0.2.3）：新增 name。入库时 related_codes 只关联自选股，
    因此需要按股票简称在标题/正文里再做一次匹配，否则全市场候选恒无数据。

    v3（2026-10-07）：**情绪标签在这里重算，不再读库内 `sentiment` 列**。
    ⚠️ 原实现直接 `i.get("sentiment")` —— 那是**入库当时**用词表打的标，
    词表一改就永久滞后。实测（2026-10-07）库内 2172 条快讯里有 **71 条**
    重算结果与库内标签不一致（3.3%）。后果是**同一个舆情因子有两条口径**：
    库内路径读旧标、`_news_stat` 的个股直查路径用 `classify_sentiment` 新算，
    两条路给出的"正面率"可能不同，而调用方无从分辨。
    `classify_sentiment` 的文档已声明"避免两套词表漂移"——这里才是那个漂移点。
    """
    items = store.get_recent_news(limit=50, code=code, hours=days * 24, name=name)
    if not items:
        return None
    sentiments = [classify_sentiment(f"{i.get('title') or ''} {i.get('content') or ''}")
                  for i in items]
    positive = sentiments.count("positive")
    negative = sentiments.count("negative")
    total = len(sentiments)
    return {"total": total, "positive": positive, "negative": negative,
            "positive_rate": round(positive / total, 2)}


if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = fetch_and_store()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    news = get_latest_news(limit=5)
    for n in news:
        print(f"  [{n['source']}|{n['sentiment']}] {(n['title'] or n['content'])[:60]} 关联:{n['related_codes']}")
