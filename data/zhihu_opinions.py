# -*- coding: utf-8 -*-
"""知乎开放平台：话题内容雷达（**仅供参考，不注入任何提示词**）

## 能力边界（2026-10-14 / 10-23 两次实测，四条都是硬约束）

1. ⚠️ **开放平台读不了指定大V**。`/api/v1/user/contents` 在不传 `X-OAuth-Token` 时，
   官方文档明确"查询 **Access Secret 所属账号本人**"（实测返回的正是本账号的回答）。
   要读他人内容必须**对方逐个 OAuth 授权** ⇒ 原计划"采集 10 个大V"**在授权模型上不可行**。
   ⇒ 官方通道改为**按话题关键词检索**（`config/zhihu_topics.json`，同样可增删）。
2. ⚠️ **`hot_list` 对 A 股参考价值低**：实测它返回**全站热榜**共 30 条，
   首位是"雷波县教师被安排与『耻辱』二字合影…"这类社会新闻。
   ⇒ 本模块**不使用它**（也顺带省下它 100 次/日 的额度）。
3. ✅ **官方主力端点是 `/content/zhihu_search?Query=…`**，实测返回字段含
   `Title / ContentText / Url / VoteUpCount / CommentCount / AuthorName / EditTime`，
   首条即《2026.10.13 A股市场的关键翻红》——**正是本功能要的内容**。
4. ⭐⭐ **想按"某个博主"取，只能走网页端 v4 的 `/members/<url_token>/pins`**（见文件末尾
   「网页端 v4」一节）。`https://www.zhihu.com/people/shou-wo-rou-ti` 是网页地址、
   `shou-wo-rou-ti` 是 **url_token**；开放平台 v1 **没有任何**按用户查的端点
   （实测 10 个候选路径全部 404）。⚠️ 该通道是**非公开接口**，免登录可用但随时可能失效，
   与官方通道**互不消耗额度、互不依赖**，默认关闭、需在页面上显式开启。

## 三条纪律

- **隔离**：本模块结果**只做人工参考** —— 不注入任何提示词、不参与 `agents.news` /
  `news_industry_review`、不参与任何评分。（用户明确要求）
- **额度护栏**：热榜 100 次/日（已不使用）；搜索额度未明确 ⇒ 统一加**每日调用上限**，
  超限即拒绝并在调用方明说"今日额度已用尽"，**宁可少取也不打光**。
- **合规**：只走官方 API，**不新增任何平台爬虫**（附录 C/D 的边界不变）。

## 鉴权三要素（缺一即 `20001 Authorization failed`）

    1) Authorization: Bearer <AccessSecret>      ← Bearer + 一个空格后直接放 secret
    2) X-Request-Timestamp: <**秒级** Unix 时间戳>  ← 毫秒级会被统一拒（返回 20001）
    3) Content-Type: application/json

⚠️ **踩坑记录**：secret 长度 **40** 位。首次拿到的粘贴值是 **42** 位（中间多了一段 `fa`），
导致**所有端点统一返回 20001** —— "长度不对"这类粘贴错误比鉴权格式错误更难看出。
"""

import json
import logging
import os
import time

_logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 主题/成员清单是**配置**：随包分发，打包后 `__file__` 推导恰好指向 `_internal/config`
# （`--add-data` 把 config/*.json 打进去了），所以这两个保持 `_ROOT` 是对的。
TOPICS_PATH = os.path.join(_ROOT, "config", "zhihu_topics.json")
MEMBERS_PATH = os.path.join(_ROOT, "config", "zhihu_members.json")
# ⚠️ 配额是**可写状态**（每次采集都要 +1 并落盘）⇒ 必须跟数据根走。
#    用 `_ROOT` 的话打包后是只读包内的 `_internal/temp/` ⇒ 配额永远写不进去，
#    表现为"额度看着没用过，实际早超了"。
from config.paths import data_root as _paths_data_root  # noqa: E402

_QUOTA_PATH = os.path.join(_paths_data_root(), "temp", "zhihu_quota.json")

# 默认关注话题（首次运行写入配置文件；之后以文件为准，可自由增删）
DEFAULT_TOPICS = ["A股", "股市行情", "大盘走势", "A股行情分析"]

API_BASE = "https://developer.zhihu.com/api/v1"
# 每日调用上限：搜索额度未在文档中明确 ⇒ 自行设保守护栏，宁可少取
DAILY_SEARCH_LIMIT = 60
_TIMEOUT = 25
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

_topic_cache = {"mtime": None, "topics": None}

# ⚠️ **失败原因必须可查**（2026-10-15 用户报"知乎仍未返回结果"）：
#    原实现把 HTTP 异常/业务码统一 `return []` 且只写日志 ⇒ 页面只能显示
#    "本次未取到任何内容（可能额度用尽或关键词过窄）"这句**没有鉴别力的猜测**，
#    用户无法判断到底是没配 token、鉴权失败、网络不通还是真没结果。
#    ⇒ 每次失败都记下"哪一步、什么原因、服务端原话"，由页面如实展示。
_last_error: dict = {}


def last_error() -> dict:
    """最近一次检索失败的结构化原因（空 dict = 目前没有失败记录）。"""
    return dict(_last_error)


def _set_error(step: str, reason: str, detail: str = "") -> None:
    _last_error.clear()
    _last_error.update({"step": step, "reason": reason, "detail": str(detail)[:300]})


def _clear_error() -> None:
    _last_error.clear()


# ---------------------------------------------------------------- 凭证
def get_token() -> str:
    return (os.getenv("ZHIHU_API_TOKEN") or "").strip()


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {get_token()}",
        # ⚠️ **必须秒级**：实测毫秒级会被服务端统一拒为 20001
        "X-Request-Timestamp": str(int(time.time())),
        "Content-Type": "application/json",
        "User-Agent": _UA,
    }


def _session():
    import requests

    s = requests.Session()
    # 项目环境里有沙箱代理；知乎接口实测在直连下才稳定（且代理会干扰排查）。
    # ⚠️ 但**不能把"必须直连"写死**：整机若只能经代理上网，直连会全失败且现象是
    #    "点了没反应"。⇒ 留 `BIGA_ZHIHU_USE_PROXY=1` 逃生门（默认关闭＝直连）。
    if (os.getenv("BIGA_ZHIHU_USE_PROXY") or "").strip() in ("1", "true", "True"):
        return s
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    return s


# ---------------------------------------------------------------- 额度护栏
def _today() -> str:
    return time.strftime("%Y-%m-%d")


def quota_status() -> dict:
    """当日已用/上限。文件损坏时**按已用满处理**（宁可拒调，不可超支）。"""
    try:
        d = json.load(open(_QUOTA_PATH, encoding="utf-8"))
        if d.get("date") == _today():
            return {"date": d["date"], "used": int(d.get("used", 0)),
                    "limit": DAILY_SEARCH_LIMIT}
    except Exception:
        pass
    return {"date": _today(), "used": 0, "limit": DAILY_SEARCH_LIMIT}


def _bump_quota(n: int = 1) -> None:
    st = quota_status()
    try:
        os.makedirs(os.path.dirname(_QUOTA_PATH), exist_ok=True)
        tmp = _QUOTA_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"date": st["date"], "used": st["used"] + n}, f)
        os.replace(tmp, _QUOTA_PATH)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[知乎] 额度计数写入失败: %s", e)


def quota_note() -> str:
    st = quota_status()
    left = max(0, st["limit"] - st["used"])
    return f"知乎搜索额度：今日已用 {st['used']}/{st['limit']}，剩余 {left}"


# ---------------------------------------------------------------- 话题配置
def load_topics() -> list:
    """读关注话题（**mtime 缓存** —— 改配置文件即时生效，不用重启，同 `providers.py` 模式）。

    文件不存在时**创建它**并写入默认值（让"可增删"这件事对用户可见）。
    """
    try:
        if not os.path.exists(TOPICS_PATH):
            save_topics(DEFAULT_TOPICS)
            return list(DEFAULT_TOPICS)
        mtime = os.path.getmtime(TOPICS_PATH)
        if _topic_cache["mtime"] == mtime and _topic_cache["topics"] is not None:
            return list(_topic_cache["topics"])
        d = json.load(open(TOPICS_PATH, encoding="utf-8"))
        topics = d.get("topics") if isinstance(d, dict) else d
        topics = [str(t).strip() for t in (topics or []) if str(t).strip()]
        if not topics:
            topics = list(DEFAULT_TOPICS)
        _topic_cache.update({"mtime": mtime, "topics": topics})
        return list(topics)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[知乎] 话题配置读取失败，用默认值: %s", e)
        return list(DEFAULT_TOPICS)


def save_topics(topics) -> bool:
    """写话题配置（原子写）。返回是否成功 —— **失败不静默**。"""
    try:
        os.makedirs(os.path.dirname(TOPICS_PATH), exist_ok=True)
        tmp = TOPICS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"topics": [str(t).strip() for t in topics if str(t).strip()]},
                      f, ensure_ascii=False, indent=1)
        os.replace(tmp, TOPICS_PATH)
        _topic_cache["mtime"] = None                # 让下次读取重新加载
        return True
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[知乎] 话题配置写入失败: %s", e)
        return False


# ---------------------------------------------------------------- 接口
def auth_ok() -> tuple:
    """连通性探测：用 `/user/contents`（**不消耗额度**）。

    Returns: `(ok: bool, message: str)`
    """
    if not get_token():
        return False, "未配置 ZHIHU_API_TOKEN"
    try:
        r = _session().get(f"{API_BASE}/user/contents", headers=_headers(),
                           params={"ContentType": "answer", "Limit": 1, "Offset": 0},
                           timeout=_TIMEOUT)
        d = r.json()
    except Exception as e:                                   # noqa: BLE001
        return False, f"请求失败：{type(e).__name__}: {str(e)[:80]}"
    if d.get("Code") == 0:
        return True, "鉴权正常"
    return False, (f"鉴权失败（Code={d.get('Code')}）：{d.get('Message')}"
                   f"　⚠️ 请检查 secret 长度是否为 **40** 位——"
                   f"实测粘贴多出字符会统一返回 20001")

def search(keyword: str, limit: int = 10) -> list:
    """按关键词检索知乎内容。

    Returns:
        `[{title, text, url, votes, comments, author, published, content_type}]`；
        **失败返回空列表**（不抛异常、不编造），由调用方决定如何声明"取不到"。
        ⚠️ 额度用尽时**直接返回空**，并由 `quota_status()` 供页面显式提示。
    """
    kw = str(keyword or "").strip()
    if not kw:
        _set_error("参数", "关键词为空")
        return []
    if not get_token():
        _logger.warning("[知乎] 未配置 token，跳过检索")
        _set_error("凭证", "未配置 ZHIHU_API_TOKEN",
                   "在 .env 里设置 ZHIHU_API_TOKEN（Access Secret，40 位）后重启进程")
        return []
    st = quota_status()
    if st["used"] >= st["limit"]:
        _logger.warning("[知乎] 今日额度已用尽（%d/%d），不再检索", st["used"], st["limit"])
        _set_error("额度", f"今日额度已用尽（{st['used']}/{st['limit']}）")
        return []

    try:
        r = _session().get(f"{API_BASE}/content/zhihu_search", headers=_headers(),
                           params={"Query": kw, "Limit": max(1, min(int(limit), 50)),
                                   "Offset": 0},
                           timeout=_TIMEOUT)
        d = r.json()
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[知乎] 检索「%s」失败: %s: %s", kw, type(e).__name__, str(e)[:90])
        _set_error("网络", f"{type(e).__name__}: {str(e)[:160]}",
                   "直连失败（本模块已禁用系统代理）。若整机需经代理上网，"
                   "请设置 BIGA_ZHIHU_USE_PROXY=1")
        return []
    _bump_quota()                       # 只有真的发出请求才计数
    if d.get("Code") != 0:
        _logger.warning("[知乎] 检索「%s」返回 Code=%s %s", kw, d.get("Code"), d.get("Message"))
        _set_error("业务码", f"Code={d.get('Code')} {d.get('Message')}",
                   "Code=20001 表示鉴权失败——请检查 secret 长度是否为 **40** 位（多粘字符会统一返回 20001）")
        return []

    items = ((d.get("Data") or {}).get("Items")) or []
    if not items:
        _set_error("空结果", f"接口返回 Code=0 但 Items 为空（关键词「{kw}」可能过窄）")
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        url = str(it.get("Url") or "")
        out.append({
            "topic": kw,
            "title": str(it.get("Title") or "").strip(),
            "text": str(it.get("ContentText") or "").strip(),
            "url": url.split("?")[0] if url else "",
            "votes": it.get("VoteUpCount"),
            "comments": it.get("CommentCount"),
            "author": str(it.get("AuthorName") or "").strip(),
            "published": it.get("EditTime"),
            "content_type": str(it.get("ContentType") or ""),
        })
    return out


def collect(topics=None, per_topic: int = 5, sleep: float = 0.4,
            progress_cb=None) -> tuple:
    """按话题列表批量检索并**跨话题去重**。

    Returns:
        `(items, meta)`；`meta` 含 `topics / asked / got / quota / error / per_topic /
        last_error`，供页面**逐话题**声明"取到什么、没取到什么、为什么"
        （本项目"缺就是缺"的要求；含糊的"未取到"没有鉴别力）。
    """
    topics = list(topics or load_topics())
    st = quota_status()
    left = max(0, st["limit"] - st["used"])

    _clear_error()
    items, seen, errs, report = [], set(), [], []
    for i, t in enumerate(topics, 1):
        if progress_cb:
            progress_cb({"current": i, "total": len(topics), "topic": t})
        if st["used"] + i > st["limit"]:
            errs.append(f"额度不足，未检索「{t}」")
            report.append({"topic": t, "got": 0, "error": "额度不足，未检索"})
            continue
        report.append({"topic": t, "got": 0, "error": ""})
        # ⚠️ 变量名不能叫 `per_topic`：会遮蔽本函数的同名形参（每话题条数），
        #    导致 `search(limit=<list>)` —— 静默传错类型。
        got = search(t, limit=per_topic)
        report[-1]["got"] = len(got)
        if not got:
            report[-1]["error"] = dict(_last_error).get("reason", "未返回结果")
        for it in got:
            key = it["url"] or it["title"]
            if key and key not in seen:
                seen.add(key)
                items.append(it)
        if sleep:
            time.sleep(sleep)

    meta = {"topics": topics, "asked": len(topics), "got": len(items),
            "quota": quota_status(), "errors": errs, "per_topic": report,
            "last_error": last_error(), "quota_left_before": left}
    if not items and not errs:
        le = last_error()
        hint = (f"：{le.get('step')} —— {le.get('reason')}" if le else "")
        errs.append(f"所有话题均未返回结果{hint}")
    return items, meta


# ================================================================
# 网页端 v4：按博主取「想法」（**非公开接口，与上面的官方通道完全独立**）
# ================================================================
# ⚠️⚠️ 2026-10-23 实测（用户报"知乎不能返回财经博主的帖子"）：
#
#   ① 开放平台 v1 **没有任何"按用户查"的端点**。实测 10 个候选路径
#      （/user/info、/user/detail、/member/contents、/member/info、
#       /content/user_contents、/content/answer_list、/content/author_answers、
#       /search/user、/user/answers、/user/articles）**全部 404**；
#      而 `/user/contents` 对本人账号返回 `Code=0` 且 `Items=[]`（该账号没发过内容）
#      ⇒ 它确实只回"token 本人"，**不是参数写错了**。这条路无解，不要再试。
#
#   ② `https://www.zhihu.com/people/<url_token>` 是**网页**，不是 API。
#      `shou-wo-rou-ti` 是 **url_token**。对应的网页端接口是
#      `https://www.zhihu.com/api/v4/members/<url_token>/...`
#
#   ③ 实测各子端点（免登录、不带 Cookie）：
#        /pins       → **200 可用**（「手握柔荑」143 条，最新一条就在当天 15:09）
#        /answers    → 403 `code=40353 need_login`  ← 需登录 Cookie + x-zse-96 签名
#        /articles   → 403 同上
#        /activities → 200 但 `data` 恒为空（同样依赖登录态）
#      ⇒ **只有「想法」能免登录取**。回答/文章在反爬墙内，本模块**不做**
#        （要伪造 x-zse-96 签名，既不稳也超出本项目"不新增平台爬虫"的边界）。
#
#   ④ 所以本段是**非公开接口**：不消耗开放平台额度、不需要 Access Secret、
#      也不受其授权模型限制；代价是**随时可能因改版/风控失效**。
#      ⇒ 默认关闭，由页面上的「按博主采集」**显式开启**；失败必须如实报原因。

WEB_API_BASE = "https://www.zhihu.com/api/v4"
# 默认关注博主（首次运行写入配置文件；之后以文件为准，可自由增删）
DEFAULT_MEMBERS = ["shou-wo-rou-ti"]
MAX_PINS_PER_MEMBER = 49          # 实测 limit=50 只回 49 ⇒ 夹到 49，别让调用方以为要到了 50
MAX_MEMBERS_PER_RUN = 20          # 单次采集的博主数上限（防止一次点下去打太多请求）
_WEB_TIMEOUT = 20
# ⚠️ 这个 UA 必须是**浏览器**：用模块统一那个精简 UA 时，v4 更容易被风控拦。
_WEB_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# ── 反爬加固（2026-10-26）──
# 事故：15:20 那趟采集拿到 HTTP **200**，但响应体是二进制乱码：
#   `[知乎·博主] 「shou-wo-rou-ti」返回非 JSON（HTTP 200）: ��Ȉ��^��&...`
# 而同一天 16:12 / 17:0x 多次直连同一端点**全部正常**（3/3 返回 gzip JSON，
# magic=7b227061 即 `{"pa`）⇒ 这是**间歇性**反爬，不是接口坏了、更不是"API 用不了"。
# 处理办法是三条（按有效性排序）：
#   ① **带上登录 Cookie**（对稳定性提升最大，社区通用做法）；
#   ② 非 JSON 响应**显式重试**（换新 session，带退避），而不是一次失败就放弃整天；
#   ③ 失败时把 magic bytes 记进错误详情 —— 下次再出问题能一眼分清
#      "压缩流没解开"(1f8b=gzip / 28b52ffd=zstd) 和 "被加密"(其它)。
_ZHIHU_COOKIE_PATH = os.path.join(_ROOT, "config", "zhihu_cookie.txt")
_WEB_RETRIES = 3
_COOKIE_CACHE = {"mtime": None, "value": ""}

_member_cache = {"mtime": None, "members": None}


def load_cookie() -> str:
    """登录 Cookie（可选）：环境变量 `BIGA_ZHIHU_COOKIE` 优先，其次 `config/zhihu_cookie.txt`。

    格式就是浏览器 F12 → Network → 任一 zhihu.com 请求 → `Cookie` 请求头整串复制。
    ⚠️ 与"伪造签名"不同：这是**用户自己的登录态**，不是破解 x-zse-96。
      不配也能跑（当前匿名访问多数时候可用），配了只是显著降低被风控概率。
    """
    env = (os.getenv("BIGA_ZHIHU_COOKIE") or "").strip()
    if env:
        return env
    try:
        mtime = os.path.getmtime(_ZHIHU_COOKIE_PATH)
        if _COOKIE_CACHE["mtime"] == mtime:
            return _COOKIE_CACHE["value"]
        with open(_ZHIHU_COOKIE_PATH, encoding="utf-8") as f:
            raw = f.read().strip()
        # 允许用户连 "Cookie: xxx" 整行粘进来
        if raw.lower().startswith("cookie:"):
            raw = raw.split(":", 1)[1].strip()
        _COOKIE_CACHE.update({"mtime": mtime, "value": raw})
        return raw
    except FileNotFoundError:
        # 没配 Cookie 是常态（匿名访问多数时候可用）⇒ debug 级，不刷 WARNING
        _logger.debug("[知乎] 未配置 Cookie 文件（%s），按匿名访问继续", _ZHIHU_COOKIE_PATH)
        return ""
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[知乎] Cookie 文件读取失败（按未配置处理）: %s: %s",
                        type(e).__name__, e)
        return ""


def _body_kind(raw: bytes) -> str:
    """给"非 JSON 响应"定性 —— 这是排查时唯一有用的信息。

    实测编码：`{"` → 正常；`1f8b` → gzip 未解压；`28b52ffd` → zstd 未解压；
    其它一律算"被替换/加密"，与前三者有完全不同的处置方式。
    """
    if not raw:
        return "空响应体"
    head = raw[:4]
    if head[:1] in (b"{", b"["):
        return "json"
    if head[:2] == b"\x1f\x8b":
        return "gzip 未解压"
    if head == b"\x28\xb5\x2f\xfd":
        return "zstd 未解压"
    return f"非 JSON（前 8 字节 {raw[:8].hex()}）"



# ---------------------------------------------------------------- 博主配置
def load_members() -> list:
    """读关注博主（url_token 列表，**mtime 缓存** ⇒ 改文件即时生效）。

    文件不存在时创建并写入 `DEFAULT_MEMBERS`（让"可增删"对用户可见）。
    ⚠️ 与 `load_topics()` 同一个模式：坏文件退回默认值，**不让配置错误炸掉页面**。
    """
    try:
        if not os.path.exists(MEMBERS_PATH):
            save_members(DEFAULT_MEMBERS)
            return list(DEFAULT_MEMBERS)
        mtime = os.path.getmtime(MEMBERS_PATH)
        if _member_cache["mtime"] == mtime and _member_cache["members"] is not None:
            return list(_member_cache["members"])
        d = json.load(open(MEMBERS_PATH, encoding="utf-8"))
        raw = d.get("members") if isinstance(d, dict) else d
        out = []
        for m in (raw or []):
            t = _norm_member_token(m)
            if t:
                out.append(t)
        if not out:
            out = list(DEFAULT_MEMBERS)
        _member_cache.update({"mtime": mtime, "members": out})
        return list(out)
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[知乎] 博主配置读取失败，用默认值: %s", e)
        return list(DEFAULT_MEMBERS)


def save_members(members) -> bool:
    """写博主配置（原子写）。返回是否成功 —— **失败不静默**。"""
    try:
        os.makedirs(os.path.dirname(MEMBERS_PATH), exist_ok=True)
        clean, seen = [], set()
        for m in members:
            t = _norm_member_token(m)
            if t and t not in seen:
                seen.add(t)
                clean.append(t)
        tmp = MEMBERS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"members": clean}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, MEMBERS_PATH)
        _member_cache["mtime"] = None                # 让下次读取重新加载
        return True
    except Exception as e:                                   # noqa: BLE001
        _logger.warning("[知乎] 博主配置写入失败: %s", e)
        return False


def _norm_member_token(raw) -> str:
    """把用户粘贴的各种写法归一成 **url_token**。

    允许直接粘 `https://www.zhihu.com/people/shou-wo-rou-ti`、`/people/xxx`、
    带 `?` 查询串或末尾 `/` 的写法 —— 否则用户一定会粘整条 URL 然后"没反应"。
    ⚠️ 也接受 `www.zhihu.com/org/xxx`（机构号），但**不接受** `people/xxx/answers`
    这类子路径（那会把子页面当成 token，请求必 404）。
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    s = s.split("?")[0].split("#")[0].strip().strip("/")
    for marker in ("/people/", "/org/"):
        if marker in s:
            s = s.split(marker, 1)[1]
            break
    else:
        if s.startswith("people/") or s.startswith("org/"):
            s = s.split("/", 1)[1]
    s = s.split("/")[0].strip()                  # 丢掉 /answers /pins 之类子路径
    return s


# ---------------------------------------------------------------- 接口
def _web_headers() -> dict:
    h = {
        "User-Agent": _WEB_UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Referer": "https://www.zhihu.com/",
        # ⚠️ **不能带** Authorization：v4 不认开放平台 token，带上反而更易被风控
    }
    ck = load_cookie()
    if ck:
        h["Cookie"] = ck
    return h


def _web_get_json(url: str, params: dict, what: str):
    """带**响应体自检 + 重试**的 GET。返回 `(status, dict|None, err_kind)`。

    为什么不能沿用"请求一次、json() 失败就 return []"：
    2026-10-26 15:20 那次，服务端返回的是 HTTP **200**，`json()` 抛错，
    于是整趟采集直接判"未取到任何言论"，当天的情绪快照就没了 ——
    而同一端点在当天其它时刻**完全正常**。间歇性反爬不该被当成永久失败。

    ⚠️ 每次重试**新建 session**：被风控的是连接/会话特征，
       复用同一个 session 重试往往还是同样的结果。
    """
    status, kind = None, ""
    for attempt in range(1, _WEB_RETRIES + 1):
        try:
            r = _session().get(url, headers=_web_headers(), params=params,
                               timeout=_WEB_TIMEOUT)
        except Exception as e:                               # noqa: BLE001
            status, kind = None, f"{type(e).__name__}: {str(e)[:120]}"
            _logger.warning("[知乎·%s] 第 %d/%d 次请求异常: %s",
                            what, attempt, _WEB_RETRIES, kind)
        else:
            status = getattr(r, "status_code", None)
            raw = getattr(r, "content", b"") or b""
            kind = _body_kind(raw)
            if status == 200 and kind == "json":
                try:
                    js = r.json()
                except Exception as e:                       # noqa: BLE001
                    # 魔数看着像 JSON 却解析失败（截断/非法 UTF-8）⇒ 当异常响应重试
                    kind = f"json 解析失败: {type(e).__name__}"
                    _logger.warning("[知乎·%s] 第 %d/%d 次：响应体形似 JSON 但解析失败（%s）",
                                    what, attempt, _WEB_RETRIES, type(e).__name__)
                else:
                    if attempt > 1:
                        _logger.info("[知乎·%s] 第 %d 次重试成功", what, attempt)
                    return status, js, ""
            elif status in (403, 404):
                # 这两类重试也不会变（404=博主不存在；403=要登录），直接返回
                return status, None, kind or f"HTTP {status}"
            else:
                _logger.warning("[知乎·%s] 第 %d/%d 次响应异常（HTTP %s，%s）",
                                what, attempt, _WEB_RETRIES, status, kind)
        if attempt < _WEB_RETRIES:
            time.sleep(0.8 * attempt + (0.4 if not load_cookie() else 0))
    return status, None, kind


def _flatten_content(blocks) -> str:
    """把 pin 的 `content` 结构化块拼成纯文本。

    实测结构：`[{"content": "这次还是我赢了。", "own_text": "这次还是我赢了。",
                 "type": "text", "fold_type": "raw"}]`

    ⚠️ 优先 `own_text`（去掉折叠/引用标记后的正文），没有再退回 `content`；
      `content` 可能是 **dict**（图片/视频块）⇒ 只取可读字段，
      **绝不把 `str(dict)` 原样塞给用户**（那是把内部结构当正文）。
    """
    if isinstance(blocks, str):
        return blocks.strip()
    if not isinstance(blocks, list):
        return ""
    parts = []
    for b in blocks:
        if isinstance(b, str):
            t = b.strip()
        elif isinstance(b, dict):
            t = b.get("own_text") or b.get("content") or b.get("title") or ""
            if isinstance(t, dict):
                t = t.get("title") or t.get("url") or ""
            t = str(t).strip()
        else:
            t = ""
        if t:
            parts.append(t)
    return "\n".join(parts).strip()


def member_pins(url_token: str, limit: int = 20, offset: int = 0) -> list:
    """按 url_token 取某博主的「想法」。

    Returns:
        `[{topic, title, text, url, votes, comments, author, published,
           content_type}]`；字段与 `search()` **同名同义**，页面可复用同一套渲染。
        **失败返回空列表**（不抛异常、不编造），原因见 `last_error()`。

    ⚠️ 只覆盖「想法」：`/answers`、`/articles` 需登录 Cookie + x-zse-96 签名（实测 403），
      本函数**不会**去试探它们 —— 试了也只会得到 403，白白增加被风控的概率。
    """
    token = _norm_member_token(url_token)
    if not token:
        _set_error("参数", "博主 url_token 为空")
        return []
    try:
        lim = max(1, min(int(limit), MAX_PINS_PER_MEMBER))
        off = max(0, int(offset))
    except (TypeError, ValueError) as e:
        # ⚠️ 参数类型不对时退回默认值，但**必须留痕** —— 否则调用方传了字符串
        #    也只会"安静地"拿到 20 条，看不出自己写错了（也撞静默 except 基线）。
        _logger.warning("[知乎·博主] limit/offset 非法（%r/%r: %s），退回 20/0",
                        limit, offset, e)
        lim, off = 20, 0

    url = f"{WEB_API_BASE}/members/{token}/pins"
    status, d, err = _web_get_json(url, {"limit": lim, "offset": off}, "博主")
    if d is None:
        if status == 404:
            _set_error("博主不存在", f"HTTP 404：url_token「{token}」在知乎查不到",
                       "请确认粘贴的是 https://www.zhihu.com/people/<url_token> 里的那一段；"
                       "也可能是账号已注销/改名。")
        elif status == 403:
            _set_error("被拦截", "HTTP 403（need_login / 反爬）",
                       "该子端点或该账号已被要求登录态。本模块**不伪造** Cookie 与 "
                       "x-zse-96 签名；可把浏览器里的 Cookie 粘到 "
                       "config/zhihu_cookie.txt（或设 BIGA_ZHIHU_COOKIE）来提高稳定性。")
        else:
            _set_error("解析" if status == 200 else "网络",
                       f"{_WEB_RETRIES} 次重试均未拿到 JSON：{err}",
                       "间歇性反爬。若反复出现，建议配置 config/zhihu_cookie.txt"
                       "（用自己的浏览器登录态，不是破解签名）。")
        return []

    err_obj = d.get("error") if isinstance(d, dict) else None
    if err_obj:
        _set_error("业务码", f"code={err_obj.get('code')} {err_obj.get('message')}",
                   "v4 是网页端内部接口，返回结构可能随时变化")
        return []

    data = (d or {}).get("data") or []
    if not data:
        _set_error("空结果", f"「{token}」没有返回任何想法")
        return []

    out = []
    for it in data:
        if not isinstance(it, dict):
            continue
        text = _flatten_content(it.get("content"))
        pid = it.get("id")
        raw_url = str(it.get("url") or "")
        # 实测 v4 返回的是 `/pins/<id>` 相对路径 ⇒ 补全，否则页面上的链接点不开
        full = (f"https://www.zhihu.com{raw_url}" if raw_url.startswith("/")
                else (raw_url or (f"https://www.zhihu.com/pin/{pid}" if pid else "")))
        au = it.get("author") or {}
        title = str(it.get("excerpt_title") or "").strip()
        if not title:                       # 想法多数没有标题 ⇒ 用正文首句做标题
            title = (text.split("\n")[0][:40] + ("…" if len(text.split("\n")[0]) > 40 else "")) \
                if text else ""
        out.append({
            "topic": f"@{au.get('name') or token}",
            "title": title,
            "text": text,
            "url": full,
            "votes": it.get("like_count"),
            "comments": it.get("comment_count"),
            "author": str(au.get("name") or "").strip(),
            "published": it.get("created"),
            "content_type": "pin",
        })
    return out


def collect_members(members=None, per_member: int = 20, sleep: float = 0.6,
                    progress_cb=None) -> tuple:
    """按博主列表批量取想法并**跨博主去重**。

    Returns:
        `(items, meta)`；`meta` 含 `members / asked / got / by_member / errors /
        last_error / source / quota`。`quota` 恒为 `None` ——
        本通道**不消耗**开放平台额度，页面必须据此显示"本次不扣额度"，
        不能复用 `quota_status()`（那会把两套通道混成一个数字）。
    """
    members = list(members if members is not None else load_members())[:MAX_MEMBERS_PER_RUN]
    _clear_error()
    items, seen, errs, report = [], set(), [], []
    for i, m in enumerate(members, 1):
        if progress_cb:
            progress_cb({"current": i, "total": len(members), "member": m})
        report.append({"member": m, "got": 0, "error": ""})
        got = member_pins(m, limit=per_member)
        report[-1]["got"] = len(got)
        if not got:
            report[-1]["error"] = dict(_last_error).get("reason", "未返回结果")
        for it in got:
            key = it.get("url") or it.get("title")
            if key and key not in seen:
                seen.add(key)
                items.append(it)
        if sleep:
            time.sleep(sleep)

    meta = {"members": members, "asked": len(members), "got": len(items),
            "by_member": report, "errors": errs, "last_error": last_error(),
            "source": "web_v4_pins", "quota": None}
    if not items:
        le = last_error()
        hint = (f"：{le.get('step')} —— {le.get('reason')}" if le else "")
        errs.append(f"所有博主均未取到想法{hint}")
    return items, meta
