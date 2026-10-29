"""
市场观点聚合（每日复盘参考）：抓取"如何看待X月X日A股行情"类的市场讨论观点。

背景与诚实说明：
- 知乎直接爬取答案不可行（实测：问题页403、API强制登录、移动端风控）；
- 主路径：Qwen 联网搜索插件（DashScope enable_search，官方稳定）——
  检索知乎"如何看待{日期}A股行情"问题及相关讨论，总结高赞观点；
- 备路径：Bing 搜索解析（自控免费，但结果相关性时好时坏，仅作补充）；
- 拿到的是搜索级摘要观点（含来源），非答案全文；全文需登录cookie，不做。

结果当日缓存，落库 news_items（source=opinions）供对话检索；双路径各自熔断。
"""

import json
import logging
import re
import base64
import html as html_mod
import os
import sys
import threading
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

_logger = logging.getLogger(__name__)

OPINIONS_TTL = 7200          # 当日观点缓存秒数
SOURCE_COOLDOWN = 1800       # 路径熔断冷却秒数
_BING_TIMEOUT = 20

_opinions_cache = {}         # date_str -> (result, timestamp)
_circuit = {"llm": 0.0, "bing": 0.0}

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"}


def _default_date() -> str:
    now = datetime.now()
    # 周末问最近交易日
    weekday = now.weekday()
    if weekday == 5:
        return (now - __import__("datetime").timedelta(days=1)).strftime("%m月%d日")
    if weekday == 6:
        return (now - __import__("datetime").timedelta(days=2)).strftime("%m月%d日")
    return now.strftime("%m月%d日")


# ================================================================
# 主路径：Qwen 联网搜索
# ================================================================

_LLM_PROMPT = """联网搜索{date}的A股行情复盘讨论（当日复盘/收盘点评），来源优先级：知乎"如何看待{date}A股行情"类问题 > 雪球/财联社/新浪财经等平台的收盘点评。

输出 JSON 数组（不要其他文字），每条观点：
[{{"source": "来源与作者(如: 知乎高赞用户XXX/雪球用户/财联社)", "viewpoint": "观点摘要(60字内,要具体:点位/板块/资金面)", "stance": "看多|看空|中性"}}]

要求：3-6条；只总结真实搜索到的内容；知乎没有相关讨论时用其他平台观点；完全搜不到才输出 []。"""


def _norm_opinion(o) -> dict:
    """把 LLM 返回的条目归一到 `{"source", "viewpoint", "stance"}`。

    ⚠️ 为什么需要：提示词示例里写了 `"viewpoint": "观点摘要(60字内,要具体:...)"`，
    模型偶尔会**把括号里的说明一起当成 key 抄下来**，实测出现过 key 为
    `view点摘要(60字内,要具体:点位/板块/资金面)` 的条目。渲染层读的是
    `o.get("viewpoint","")` ⇒ 拿到空串 ⇒ 该条观点**静默消失**（条数对不上才发现）。
    ⇒ 这里做**宽容归一**：找不到标准 key 时，退而取"非 source/stance 的第一个
      字符串值"当正文。宁可字段名不标准，也不能让观点凭空丢掉。
    """
    if not isinstance(o, dict):
        return {"source": "联网搜索", "viewpoint": str(o)[:300], "stance": "中性"}
    src = o.get("source") or o.get("来源") or "联网搜索"
    body = o.get("viewpoint") or o.get("观点") or o.get("观点摘要")
    if not body:
        for k, v in o.items():
            if k in ("source", "stance", "来源", "倾向"):
                continue
            if isinstance(v, str) and v.strip():
                body = v
                break
    stance = o.get("stance") or o.get("倾向") or "中性"
    if stance not in ("看多", "看空", "中性"):
        stance = "中性"
    return {"source": str(src)[:60], "viewpoint": str(body or "").strip()[:300],
            "stance": stance}


def fetch_llm_opinions(date_str: str) -> dict:
    """Qwen 联网搜索路径：返回 {"opinions": [...], "raw"}；失败抛异常"""
    from openai import OpenAI
    from config.settings import config
    from llm.deepseek_client import (TEMP_ANALYTICAL, current_llm_purpose,
                                     llm_purpose, _log_llm_call)

    # ⚠️ 本处**绕过 DeepSeekClient** 直连 openai SDK（联网搜索插件要传
    #    extra_body={"enable_search": True}，客户端封装不支持），后果有二：
    #    ① 不吃 call_api 的温度档位，必须自己写；
    #    ② **不写 llm_calls**，所以市场观点的调用在可观测性面板里一直看不到。
    #    2026-10-14 起两处都补上：模型走**功能级路由**（功能键 market_opinions，
    #    默认=快档，与改造前的 QWEN_TURBO_MODEL 一致），并**手动记一条 llm_calls**
    #    （best-effort，写失败不影响调用），否则这个功能永远是可观测性盲区。
    #
    # ⚠️⚠️ 2026-10-23 修：**模型名必须来自 Qwen 侧**，不能直接用 `resolve_model()`。
    #    事故现场：`resolve_model("market_opinions")` 返回 `deepseek-v4-flash`
    #    （全局默认模型已于模型分级后切到 DeepSeek），却被原样发给
    #    `config.QWEN_BASE_URL`（阿里云）⇒ 404 `Model not exist` ⇒ 静默降级到 Bing
    #    ⇒ 市场观点章节长期为空，而日志只有一行 WARNING，看不出是**配错**还是**网断**。
    #    根因是「端点来源」与「模型来源」没有绑定：谁给你 key，就必须用谁家的模型名。
    #    ⇒ 这里显式绑定到 Qwen 快档（`QWEN_TURBO_MODEL`），并保留功能级覆盖
    #      （`feature_model()`）供单独指定——即便如此，也只接受非 DeepSeek 系模型名。
    from config.model_routing import feature_model
    model_to_use = feature_model("market_opinions") or str(
        config.QWEN_TURBO_MODEL or "qwen-max").strip()
    if str(model_to_use).lower().startswith(("deepseek", "gpt-")):
        # 不要把"配错端点"变成"又一次 404"：宁可用 Qwen 默认档，也要留一条可查的日志。
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[市场观点] 功能模型 %r 不属于 Qwen 端点（本函数直连 QWEN_BASE_URL）"
            "⇒ 回退 QWEN_TURBO_MODEL", model_to_use)
        model_to_use = str(config.QWEN_TURBO_MODEL or "qwen-max").strip()
    client = OpenAI(api_key=config.QWEN_API_KEY, base_url=config.QWEN_BASE_URL)
    started = time.perf_counter()
    with llm_purpose("market_opinions"):
        try:
            resp = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": _LLM_PROMPT.format(date=date_str)}],
                extra_body={"enable_search": True},
                # 收敛为 TEMP_ANALYTICAL（利好/利空/中性三分类判定 / JSON 输出）
                temperature=TEMP_ANALYTICAL, max_tokens=800,
            )
        except Exception as e:                          # noqa: BLE001
            _log_llm_call(model_to_use, current_llm_purpose(), started,
                          ok=False, error=f"{type(e).__name__}: {e}")
            raise
        _log_llm_call(model_to_use, current_llm_purpose(), started,
                      response=resp, ok=True)
    # 联网插件：观点在 content；搜索引用有时在 annotations
    text = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            opinions = json.loads(m.group(0))
            if isinstance(opinions, list):
                return {"opinions": [_norm_opinion(o) for o in opinions if o],
                        "raw": text}
        except json.JSONDecodeError:
            pass
    # JSON 解析失败：整段作为单条观点
    if text:
        return {"opinions": [{"source": "联网搜索", "viewpoint": text[:300], "stance": "中性"}],
                "raw": text}
    raise ValueError("联网搜索返回空内容")


# ================================================================
# 备路径：Bing 搜索解析（知乎结果）
# ================================================================

def _decode_bing_url(block: str) -> str:
    m = re.search(r'href="(https?://(?!www\.bing|cn\.bing)[^"]+)"', block)
    if m:
        return m.group(1)
    m = re.search(r"u=a1([A-Za-z0-9_-]{20,})", block)
    if m:
        u = m.group(1)
        u += "=" * (-len(u) % 4)
        try:
            return base64.urlsafe_b64decode(u).decode("utf-8", errors="ignore")
        except Exception:
            return ""
    return ""


def parse_bing_results(raw_html: str, max_items: int = 6) -> list[dict]:
    """解析 Bing 结果页 HTML → [{title, url, excerpt}]（宽松过滤无关项）"""
    out = []
    blocks = re.split(r'<li class="b_algo"', raw_html)[1:]
    for b in blocks:
        m_title = re.search(r"<h2[^>]*>\s*<a[^>]*>(.*?)</a>", b, re.S)
        title = html_mod.unescape(re.sub(r"<[^>]+>", "", m_title.group(1))).strip() if m_title else ""
        url = _decode_bing_url(b)
        m_snip = re.search(r"<p[^>]*>(.*?)</p>", b, re.S)
        excerpt = html_mod.unescape(re.sub(r"<[^>]+>", "", m_snip.group(1))).strip()[:200] if m_snip else ""
        if not title or not excerpt:
            continue
        # 相关性过滤：必须含行情词（实测 Bing 会篡改 site: 限定返回无关内容，仅凭域名不够）
        market_words = ("A股", "行情", "股市", "大盘", "指数", "收盘", "涨", "跌", "创业板")
        relevant = any(w in title or w in excerpt for w in market_words)
        if not relevant:
            continue
        out.append({"title": title[:120], "url": url, "excerpt": excerpt})
        if len(out) >= max_items:
            break
    return out


def fetch_bing_opinions(date_str: str) -> dict:
    """Bing 搜索路径：知乎结果摘要。返回 {"opinions": [...]}；失败/无相关结果抛异常"""
    query = f"如何看待 {date_str} A股 行情 site:zhihu.com"
    resp = requests.get("https://cn.bing.com/search",
                        params={"q": query}, headers=_HEADERS, timeout=_BING_TIMEOUT)
    resp.raise_for_status()
    items = parse_bing_results(resp.text)
    if not items:
        raise ValueError("Bing 无相关知乎结果")
    opinions = [{"source": "知乎(搜索摘要)",
                 "viewpoint": f"{i['title']} — {i['excerpt'][:120]}",
                 "stance": "中性", "url": i["url"]}
                for i in items]
    return {"opinions": opinions, "raw": ""}


# ================================================================
# 聚合入口
# ================================================================

def get_market_opinions(date_str: str = None) -> dict:
    """
    获取市场观点（主：Qwen联网搜索；备：Bing知乎结果）。当日缓存。

    Returns: {"date", "path", "opinions": [...], "as_of"}；全失败 {"error"}
    """
    date_str = date_str or _default_date()
    cache_key = f"market_opinions:{date_str}"
    cached = _opinions_cache.get(date_str)
    now = time.time()
    if cached and now - cached[1] < OPINIONS_TTL:
        return cached[0]

    # ⚠️⚠️ 跨进程缓存（2026-10-23）：模块级 `_opinions_cache` 只在本进程内有效，
    #    而项目有 3 个常驻进程（调度器 / UI / watchdog），各持一份 ⇒ 同一份观点被
    #    重复购买。实测 2026-10-23 单日 28 次调用（qwen-max，¥0.62，占当日总成本 41%），
    #    `news_items` 里同一批观点被写入 24 次，而它实际只需要 2~3 次。
    #    ⇒ 命中内存后再问一次库；写回时两边都写。
    try:
        from db import store as _store
        deep_cached = _store.get_opinion_cache(cache_key, OPINIONS_TTL)
        if deep_cached:
            _opinions_cache[date_str] = (deep_cached, now)
            return deep_cached
    except Exception as e:                              # noqa: BLE001
        _logger.warning("[市场观点] 跨进程缓存读取失败（当作未命中）: %s", e)

    result = None
    if now >= _circuit["llm"]:
        try:
            result = fetch_llm_opinions(date_str)
            result["path"] = "qwen联网搜索"
        except Exception as e:
            _logger.warning("[市场观点] LLM路径失败: %s", e)
            _circuit["llm"] = now + SOURCE_COOLDOWN
    if result is None and now >= _circuit["bing"]:
        try:
            result = fetch_bing_opinions(date_str)
            result["path"] = "bing知乎摘要"
        except Exception as e:
            _logger.warning("[市场观点] Bing路径失败: %s", e)
            _circuit["bing"] = now + SOURCE_COOLDOWN

    if result is None:
        return {"error": "市场观点双路径均不可用（联网搜索与Bing均失败），本次复盘跳过该章节"}

    result["date"] = date_str
    result["as_of"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result.setdefault("opinions", [])
    _opinions_cache[date_str] = (result, now)
    try:
        from db import store as _store
        _store.save_opinion_cache(cache_key, result)
    except Exception as e:                              # noqa: BLE001
        _logger.warning("[市场观点] 跨进程缓存写入失败（不影响本次结果）: %s", e)
    _persist(result)
    return result


def _persist(result: dict):
    """观点落库 news_items（source=opinions），供对话检索"""
    try:
        from db import store as db_store
        items = []
        for o in result.get("opinions", []):
            content = f"{o.get('viewpoint', '')}（倾向: {o.get('stance', '中性')}）"
            items.append({
                "hash": hashlib_md5(f"opinions|{result['date']}|{o.get('source', '')}|{content[:80]}"),
                "title": f"市场观点 {result['date']} — {o.get('source', '')}",
                "content": content[:500],
                "source": "opinions",
                "published_at": result.get("as_of", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "related_codes": [], "sentiment": _stance_to_sentiment(o.get("stance")),
            })
        if items:
            db_store.insert_news(items)
    except Exception as e:
        _logger.warning("[市场观点] 落库失败: %s", e)


def hashlib_md5(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _stance_to_sentiment(stance) -> str:
    stance = str(stance or "")
    if "多" in stance:
        return "positive"
    if "空" in stance:
        return "negative"
    return "neutral"


if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = get_market_opinions()
    print(json.dumps(out, ensure_ascii=False, indent=2)[:1500])
