"""热点新闻页面 — 行业利好/利空评估 / 利好个股推荐 / 全市场快讯流 / 实时热榜

为什么需要这个页面（2026-10-09，用户提议的热点新闻工作流）：
库内 `news_items` 已积累数千条全市场快讯（每小时采集），且 AI 助手能通过工具读到——
**但界面上没有任何地方能看到**。本页把"采集 → 行业评估 → 利好个股 → 深度分析"
整条链路展示出来：

  1. 行业利好/利空评估：每天 15:55 调度任务（core.news_industry_review）产出，
     LLM 判定 + **证据引用（代码校验过的新闻 id）**——证据必须可点开看原文；
  2. 利好个股推荐：利好行业 × 行业成分（新浪行业，缓存 24h）× 近3日主力净流入，
     一键跳转深度分析（预填代码）；
  3. 全市场快讯流：按行业/情绪/时间筛选；
  4. 实时热榜：东财人气榜等（失败必须显示"不可用"，不装作没有）。

⚠️ 展示原则与报告一致：**推荐 ≠ 交易指令**。页面上任何个股建议都注明
"请以深度分析的三档建议为准"（看多可执行 / 看空驳回 / 观望等待）。

⚠️ 区块快速定位（2026-10-15 用户提议）：本页有 5 个区块，靠滚动找很费劲 ⇒
顶部下拉框切换。**默认「全部显示」**（与改造前行为一致）；选具体区块时只渲染那一块
（顺带省掉其他区块的取数/联网）。

⚠️ 纪律：**全局口径**（上面那行 caption）放在下拉框**之上**，任何选择下都可见；
各区块**自身的使用边界**留在区块内（选中才显示）。别把全局声明塞进区块里，
否则会重演"边界声明因区块被跳过而消失"的旧事故。
"""

import sys, os
import time                                  # ⚠️ 必须显式导入：知乎区块用了 time.strftime
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

st.set_page_config(page_title="热点新闻", page_icon="📰", layout="wide")
st.title("📰 热点新闻")
st.caption("每小时采集全市场快讯 → 每天 15:55 按行业评估利好/利空（附证据原文）→ "
           "利好行业给出成分股排序（近3日主力净流入）→ 一键深度分析。"
           "**推荐仅为排序参考，不构成交易指令**")

from db import store as db_store
from ui.components.section_nav import section_nav, visible

ANALYSIS_PAGE = "pages/2_🔍_深度分析.py"

# ================================================================
# 重依赖**延迟导入**（2026-10-15，实测驱动的改动）
# ================================================================
# ⚠️ 实测冷导入耗时（各自新进程）：`core.news_industry_review` **9.28s**
#    （其中 `akshare` 4.02s + `pandas` 1.47s）、`data.news_feed` 4.39s。
#    这两个模块原本在**页面顶部**导入 ⇒ **无论用户要看哪一块**，进入本页都要先等
#    ≈13.7s 的导入 ⇒ 表现为"点进来/切过去都很慢"。
#    ⇒ 改为**只在真正需要它的区块里导入**：
#       · 「行业评估 / 受益个股 / 快讯」需要 → 由这三个块按需触发导入；
#       · 「实时热榜 / 知乎」**完全不需要** → 因此切到这两块**不再付那 13.7s**。
#    ⚠️ 这不是"为了快而牺牲正确性"：导入是纯幂等的，只是把时机推迟到用得到的时候。
def _nir():
    """行业评估模块（唯一入口，延迟导入）。"""
    from core import news_industry_review as _m
    return _m


def _news_feed():
    """新闻采集工具模块（唯一入口，延迟导入）。"""
    from data import news_feed as _m
    return _m


# ================================================================
# 昂贵加载一律走缓存（2026-10-15，用户反馈"切换区块耗时偏久"）
# ================================================================
# ⚠️ 根因不是导航本身，而是 **Streamlit 每次交互都会重跑整个脚本**：
#    切换副模块 = 一次全页 rerun ⇒ 若不缓存，被选中区块的取数会**每次都重新联网**。
#    ⇒ 用 `st.cache_data(ttl)` 把"同一份数据"在 TTL 内复用；TTL 按数据新鲜度分档。
#    ⚠️ 缓存只影响**读取**，不改变任何口径：过期后照常重新取数。
@st.cache_data(ttl=120, show_spinner=False)
def _cached_reviews():
    return db_store.get_news_reviews()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_recommend(top_n: int):
    return _nir().recommend_stocks(top_n=top_n)


@st.cache_data(ttl=120, show_spinner=False)
def _cached_recent_news(limit: int, hours: int, version: int):
    """快讯列表。`version` 是缓存键的一部分（见 `db_store.get_news_latest`）。

    ⚠️ 2026-10-20 修：原来没有 `version` 参数，TTL 又是 120 秒，而采集每小时
       才写一次 ⇒ 用户在采集后 2 分钟内刷新，看到的仍是**旧快照**，于是报
       "新闻没采集"。加了版本号后，采集一入库缓存立刻失效；
       `ttl` 放大到 1 小时只是兜底（正常情况下 key 变化才是失效主因）。
    ⚠️ 参数名**不能**写成 `_version`：Streamlit 会把下划线前缀的参数排除在
       缓存键之外，那样这个修复会静默失效。
    """
    return db_store.get_recent_news(limit=limit, hours=hours)


@st.cache_data(ttl=180, show_spinner=False)
def _cached_hot_rank() -> dict:
    """实时热榜：**成功与失败都进缓存**（负缓存）。

    ⚠️⚠️ 为什么必须缓存失败 —— `st.cache_data` **不缓存异常**：函数抛异常时不会写入缓存，
       于是每次切回这一块都会重新发请求、重新等一遍超时。
       实测本机该接口稳定 `ProxyError`，单次 5.0s ⇒ 用户体感"切到热榜特别慢"，
       而这 5 秒**是每次都在重复浪费**（缓存命中后降到 ≈0s）。
    ⚠️ 超时从 20s 收到 **6s**：这是**交互路径**上的请求，用户等不起 20 秒；
       拿不到就如实说"暂不可用"，而不是把人卡住。
    """
    try:
        import akshare as ak
        df = _news_feed()._call_with_timeout(lambda: ak.stock_hot_rank_em(), timeout=6)
        return {"ok": True, "df": df, "error": ""}
    except Exception as e:                                   # noqa: BLE001
        return {"ok": False, "df": None,
                "error": f"{type(e).__name__}: {str(e)[:120]}"}


@st.cache_data(ttl=300, show_spinner=False)
def _cached_zhihu_auth():
    from data import zhihu_opinions as Z
    return Z.auth_ok()


# ================================================================
# 区块快速定位
# ================================================================
SEC_EVAL = "🧭 行业利好 / 利空评估"
SEC_PICKS = "⭐ 可能受益的个股"
SEC_NEWS = "🗞️ 全市场快讯"
SEC_HOT = "🔥 实时热榜"
SEC_ZHIHU = "💬 知乎话题参考"
_SECTIONS = (SEC_EVAL, SEC_PICKS, SEC_NEWS, SEC_HOT, SEC_ZHIHU)
_SEL = section_nav(_SECTIONS, key="news_section", group_label="📰 热点新闻 · 板块")


def _goto_analysis(code: str):
    """预填代码并跳转深度分析页。

    ⚠️ 必须在跳转**之前**写 `prefill_code`——switch_page 之后本页脚本即终止。
    深度分析页在创建输入框前会 pop 它（见该页注释）。
    """
    st.session_state["prefill_code"] = str(code)
    st.switch_page(ANALYSIS_PAGE)


# ================================================================
# ① 行业利好/利空评估
# ================================================================
if visible(_SEL, SEC_EVAL):
    st.subheader(SEC_EVAL)

    date, reviews = _cached_reviews()
    if date is None:
        st.info("还没有评估结果。评估任务每天 **15:55** 自动运行（聚合最近 24 小时快讯）。"
                "刚部署的话，等下一次调度或先看下面的快讯流。")
    else:
        verdict_icon = {"利好": "🟢", "利空": "🔴", "中性": "⚪"}
        st.caption(f"评估日期：**{date}**　｜　共 {len(reviews)} 个行业　｜　"
                   "证据 id 均经代码校验（只能引用当日输入的新闻）")
        for r in reviews:
            icon = verdict_icon.get(r["verdict"], "⚪")
            head = f"{icon} **{r['sector']}** —— {r['verdict']}"
            if r.get("summary"):
                head += f"　｜　{r['summary']}"
            with st.expander(head, expanded=(r["verdict"] == "利好")):
                ev = r.get("evidence") or []
                if ev:
                    st.markdown("**证据（点击 id 对应的原文标题）**")
                    for e in ev:
                        st.markdown(f"- `#{e.get('id')}` {e.get('title') or '（标题缺失）'}")
                else:
                    st.caption("（该结论没有挂具体证据——基于全部新闻整体判断）")
                st.caption(f"聚合快讯 {r.get('news_count', 0)} 条"
                           + (f"　｜　模型 {r['model']}" if r.get("model") else ""))

    # 手动触发（把调度逻辑原样跑一遍，结果与自动任务完全同源）
    with st.expander("⚙️ 立即重新评估（调用大模型，约 30~60 秒）", expanded=False):
        hours = st.selectbox("聚合窗口", (24, 48, 72), index=0, key="review_hours")
        if st.button("▶️ 运行评估", key="run_review_btn"):
            with st.spinner("评估中…"):
                result = _nir().run_industry_review(hours=int(hours))
            if result["status"] == "ok":
                st.success(f"完成：{result['saved']} 个行业落库。")
                for w in result["warnings"]:
                    st.warning(f"⚠️ {w}")
                st.rerun()
            elif result["status"] == "no_data":
                st.info(f"没有可评估的内容：{result['error']}")
            else:
                # llm_error —— 缺就是缺，不能拿旧结论充数
                st.error(f"评估失败（未落库）：{result['error']}")
                for w in result["warnings"]:
                    st.warning(f"⚠️ {w}")

# ================================================================
# ② 利好个股推荐
# ================================================================
if visible(_SEL, SEC_PICKS):
    st.subheader("⭐ 可能受益的个股（利好行业 × 近3日主力净流入）")

    rec = _cached_recommend(5)
    if rec["status"] != "ok":
        st.caption(rec.get("note") or rec["status"])
    else:
        for w in rec.get("warnings", []):
            st.warning(f"⚠️ {w}")
        st.caption(rec["note"] + (f"　｜　资金流基准日 {rec.get('flow_date')}"
                                  if rec.get("flow_date") else "　｜　无资金流数据"))
        for p in rec["picks"]:
            c1, c2 = st.columns([5, 1])
            with c1:
                net = p.get("net_3d")
                net_txt = f"{net / 1e8:.2f} 亿" if isinstance(net, (int, float)) else "无数据"
                # 概念优先时标出具体链条（有色·铜），行业兜底时只显示大类——
                # 不能把两者混成一个字段，否则看不出这条推荐是按哪套口径给的
                tag = p["industry"] + (f"·{p['concept']} 概念" if p.get("concept") else "")
                st.markdown(
                    f"**{p['stock_code']} {p['stock_name'] or '—'}**　"
                    f"({tag})　｜　近3日主力净流入 {net_txt}"
                    + (f"　｜　涨跌 {p['change_pct']:+.2f}%"
                       if isinstance(p.get("change_pct"), (int, float)) else ""))
                if p.get("sector_summary"):
                    st.caption(f"行业判断：{p['sector_summary']}")
            with c2:
                st.button("🔍 深度分析", key=f"goto_{p['stock_code']}",
                          on_click=_goto_analysis, args=(p["stock_code"],))
            st.divider()

# ================================================================
# ③ 全市场快讯流
# ================================================================
if visible(_SEL, SEC_NEWS):
    st.subheader(SEC_NEWS)

    f1, f2, f3 = st.columns(3)
    hours = f1.selectbox("时间范围", (6, 24, 72, 168), index=1, key="news_hours")
    sector = f2.selectbox("行业", ("全部",) + tuple(_news_feed()._SECTOR_KEYWORDS.keys()),
                          key="news_sector")
    sentiment = f3.selectbox("情绪", ("全部", "positive", "negative", "neutral"),
                             key="news_sentiment")

    # 缓存版本号 = 最新新闻 id（采集一入库就变），顺带拿到"数据截止"时间
    _news_ver, _news_at = db_store.get_news_latest()
    rows = _cached_recent_news(300, int(hours), _news_ver)
    shown = []
    for r in rows:
        if sentiment != "全部" and (r.get("sentiment") or "neutral") != sentiment:
            continue
        if sector != "全部":
            text = f"{r.get('title') or ''} {r.get('content') or ''}"
            if _nir().classify_sector(text) != sector:
                continue
        shown.append(r)

    if not shown:
        # ⚠️ 必须写清"是筛选清空还是库里没数据"——否则用户看到这行会以为"没采集"
        st.caption(f"该筛选条件下没有快讯（缺就是缺——不代表没有发生）。"
                   f"　｜　本次窗口内 {len(rows)} 条，库内最新入库 "
                   f"{_news_at or '（暂无）'}")
    else:
        st.caption(f"共 {len(shown)} 条（窗口 {hours}h，最多显示 300 条）"
                   f"　｜　库内最新入库 {_news_at or '（暂无）'}")
        for r in shown[:120]:
            s_icon = {"positive": "🟢", "negative": "🔴"}.get(r.get("sentiment"), "⚪")
            when = str(r.get("published_at") or "")[:16]
            # ⚠️ 部分源 title 为空（content 为主）——用正文替代，别渲染出空行
            headline = (r.get("title") or "").strip() or (r.get("content") or "").strip()[:80]
            st.markdown(f"{s_icon} **{when}**（{r.get('source')}）　{headline}")
            content = (r.get("content") or "").strip()
            if content and content != r.get("title") and len(content) > 80:
                st.caption(content[:200])

# ================================================================
# ④ 实时热榜
# ================================================================
if visible(_SEL, SEC_HOT):
    st.subheader("🔥 实时热榜（东财人气榜）")

    # ⚠️ BIGA_UI_OFFLINE：UI 冒烟测试环境跳过热榜——AppTest 会真执行页面脚本，
    #    akshare 调用在无网/沙箱环境里要么挂 20 秒、要么报代理错误，纯属浪费
    if os.environ.get("BIGA_UI_OFFLINE"):
        st.caption("（离线模式：热榜已跳过——这只影响这一节，评估与快讯都来自库内数据）")
    else:
        # ⚠️ 失败也进缓存（负缓存）：否则每次切回本块都要重新等一遍超时（实测 5.0s/次）。
        res = _cached_hot_rank()
        if not res.get("ok"):
            st.warning(f"⚠️ 热榜不可用（{res.get('error')}）：不影响上面的评估与快讯——"
                       "它们来自每小时采集的库内数据。"
                       "**该失败已缓存 180 秒，期间不会重复重试。**")
        df = res.get("df")
        if df is not None and hasattr(df, "columns") and len(df):
            cols = [c for c in ("当前排名", "股票代码", "股票名称", "最新价", "涨跌幅")
                    if c in df.columns]
            st.dataframe(df[cols].head(20), width="stretch", hide_index=True)
        elif res.get("ok"):
            st.warning("⚠️ 热榜返回了空数据（源可能临时不可用），已如实展示，不用旧数据充数。")


# ================================================================
# ⑤ 知乎话题参考（**仅供人工参考**）
# ================================================================
# ⚠️ 四条不可省略的声明（用户明确要求 + 实测能力边界）：
#   1. **完全隔离**：本节内容**不注入任何提示词**、不参与 `agents.news` /
#      `news_industry_review`、**不参与任何评分** —— 纯人工参考。
#   2. **官方通道读不了指定大V**：知乎 `/user/contents` 在不传 `X-OAuth-Token` 时只返回
#      **本人账号**内容（官方文档原文）。要读他人须对方逐个 OAuth 授权。
#      ⇒ 官方通道只做**按话题关键词检索**（话题可在此页增删）。
#   3. **额度**：搜索额度有每日上限（模块内置护栏，默认 60 次/日），
#      超限即拒绝并在页面明说 —— 宁可少取，不打光。
#   4. ⭐ **按博主取只能走网页端 v4 的 `/members/<url_token>/pins`**（2026-10-23 实测）：
#      免 token、不扣额度，但属**非公开接口**、且只覆盖「想法」
#      （`/answers`、`/articles` 实测 403 need_login）。故**单独一个采集方式**，
#      由用户显式选择；两种通道的额度/失败原因必须分开陈述，不能混成一个数字。
_ZH_KEY = "zhihu_topic_items"
_ZH_MEMBER_KEY = "zhihu_member_items"
MODE_TOPIC = "🔍 按话题检索（官方开放平台）"
MODE_MEMBER = "👤 按博主采集（网页端接口·非官方）"


LABEL_RULE = "📖 词表规则（零成本·可复现）"
LABEL_LLM = "🧠 模型逐条标注（出报告 + 逐条标签）"


def _render_llm_card(pack: dict, source: str, key: str):
    """模型逐条标注：**先出报告，再由代码把标签算成数字**。

    ⚠️ 三条不可省的边界：

    1. **按钮触发**，不随 rerun 自动跑。一批 40 条 ≈ 2 次模型调用，
       而页面每次交互都会 rerun —— 自动跑等于"每点一下就花一次钱"。
    2. **报告不进复盘提示词**。报告是模型对发言的**二手叙述**，无法逐句核对；
       进提示词就变成了"另一个模型的看法当事实"。复盘只取**数字**（统计量），
       报告只供人读（含 `争议焦点`/`异常信号`）。
    3. **数字从标签算**（`build_snapshot(labels=...)`），模型自报的百分比在
       解析阶段就被整条丢弃（`_strip_numbers`）。
    """
    from core import sentiment_llm as SL
    from core import sentiment_reference as SR

    n = min(len(pack["items"]), SL.MAX_POSTS)
    chunks = (n + SL.LABEL_CHUNK - 1) // SL.LABEL_CHUNK if n else 0
    calls = chunks + (1 if chunks > 1 else 0)

    cache_key = f"{key}_llm_{pack['ts']}"
    c1, c2 = st.columns([3, 2])
    with c1:
        st.caption(f"将对 {n} 条逐条标注（{chunks} 批 + 报告合并 = 约 {calls} 次模型调用）；"
                   f"标签与报告一并落库，刷新页面不重复调用。")
    with c2:
        st.caption("⚠️ 模型也会误判；每条标签附原文引用，可逐条核对。")

    if st.button(f"🧠 开始标注（{n} 条）", key=f"{key}_label_run"):
        with st.spinner("正在逐条标注并生成报告…"):
            try:
                res = SR.build_for_source(pack["items"], source=source, labeler="llm")
            except Exception as e:              # noqa: BLE001
                st.error(f"标注失败（{type(e).__name__}）：{e}")
                return
        st.session_state[cache_key] = res

    res = st.session_state.get(cache_key)
    if not res:
        st.info("点上方按钮开始。**未点击前不会调用任何模型。**")
        return

    snap = res["snapshot"]
    base = res["baseline"]
    report = res.get("report") or {}
    meta = res.get("label_meta") or {}
    agree = res.get("agreement") or {}

    _render_snap_metrics(snap, base)
    if base.get("status") != "ok":
        # ⚠️ 这条必须写：否则"模型口径没有分位"会被读成"这个源没历史"。
        #    实际是调度任务按**词表口径**落库，两种口径的基准不互相借用。
        st.caption("说明：每日调度（15:45）按**词表口径**落库；模型口径需在页面上"
                   "累计同口径的 5 个交易日才会出现历史分位，两者**不互相借用**。")

    # ── 报告（只展示，不注入任何提示词）────────────────────────
    if report.get("主线") or any(report.get(k) for k in ("多头理由", "空头理由", "争议焦点")):
        st.markdown("##### 📝 模型总结（仅供人工阅读，**不进复盘提示词**）")
        if report.get("主线"):
            st.markdown(f"**主线**：{report['主线']}")
        for label, k in (("多头理由", "多头理由"), ("空头理由", "空头理由"),
                         ("争议焦点", "争议焦点"), ("异常信号", "异常信号")):
            vals = report.get(k) or []
            if vals:
                st.markdown(f"- **{label}**：" + "；".join(str(v) for v in vals))
        st.caption("⚠️ 以上是模型对发言的归纳，**不是事实**；报告里若出现统计数字已被程序丢弃，"
                   "百分比一律以上方指标卡为准。")
    else:
        st.caption("本批未产出可用报告（模型未返回或全部被校验拦下）。")

    # ── 双轨对照（同一批条目上比，唯一有可比性的做法）──────────
    if agree.get("n_both_directional") is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("词表/模型同向", "—" if agree.get("same_rate") is None
                  else f"{agree['same_rate']:.0%}",
                  f"两边都判出方向 {agree.get('n_both_directional', 0)} 条")
        c2.metric("模型补判", f"{agree.get('llm_only', 0)} 条", "词表未判出方向",
                  delta_color="off")
        c3.metric("引用逐字命中", "—" if not meta.get("quote_hit_rate")
                  else f"{meta['quote_hit_rate']:.0%}", "低于 50% 则整块作废")
        if agree.get("conflicts"):
            with st.expander(f"词表与模型判定相反的 {len(agree['conflicts'])} 条（供人工核）"):
                for c in agree["conflicts"]:
                    st.caption(f"· 词表={c['rule']}／模型={c['llm']}｜{c['text']}")
        st.caption("⚠️ 两个标注器都**不是真值**。同向率高只说明「两把尺子读数接近」，"
                   "不说明读数正确；相反的那些条目才是值得人工看的样本。")

    for w in snap.get("label_warnings") or []:
        st.caption(f"⚠️ {w}")

    with st.expander("🔎 给大模型的参照块（数字，不含报告）", expanded=False):
        st.code(SR.render_block(snap, base), language="text")


def _render_snap_metrics(snap: dict, base: dict):
    """指标卡（两种标注方式共用 —— 数字口径一致，只有标注器不同）。"""
    from core import sentiment_reference as SR

    n_scored = snap.get("n_scored") or 0
    want = SR.MIN_SCORED_OK
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("判出方向", f"{n_scored} 条", f"门槛 {want}",
              delta_color="off" if n_scored >= want else "inverse")
    c2.metric("多空比", "—" if snap.get("bull_ratio") is None
              else f"{snap['bull_ratio']:.0%}", "看多 / 判出方向")
    c3.metric("覆盖率", "—" if snap.get("coverage") is None
              else f"{snap['coverage']:.0%}", f"共 {snap.get('n_total') or 0} 条")
    c4.metric("分歧度", "—" if snap.get("dispersion") is None
              else f"{snap['dispersion']:.2f}", "1=对半")

    q = snap.get("quality")
    if q and q != "ok":
        st.caption(f"⚠️ 数据质量：{SR.quality_text(q)}")
    for n in snap.get("notes") or []:
        st.caption(f"· {n}")

    if base.get("status") == "ok":
        st.caption(
            f"相对自身：近 {base['available_days']} 日均值 "
            f"{base['bull_ratio_ma']:.0%}，当日 {base['delta']:+.1%}，"
            f"处于 {base['bull_ratio_pct_rank']:.0%} 分位"
            + (f"；发言量为近 5 日均值的 {base['volume_ratio']:.1f} 倍"
               if base.get("volume_ratio") else ""))
    else:
        # ⚠️ 相对基准不可用时**要说出来**：否则"没有分位"会像"分位正常"
        st.caption(f"相对自身：{base.get('note') or '基准不可用'}")


def _render_sentiment_card(pack: dict, source: str, key: str):
    """展示「市场情绪参照」指标卡（结构化数字 + 可选 LLM 判定）。

    ⚠️ 三个刻意的设计选择：

    1. **采集即落库**（`build_for_source`），不是"看到才落库"。
       相对基准靠历史，少一天就少一天；等用户想起来再点，基准永远凑不齐。
    2. **LLM 判定做成按钮**，不自动跑。
       判定要花钱，而且同一批数据一天判一次就够；每次 rerun 自动跑
       会变成"每按一次按钮花一次钱"，而用户往往只是想看看数字。
    3. **标注方式必须显式选**。词表零成本但不是真值（实测把"降了30%仓位"
       判成看多）；模型更准但按条计费，且换标注器后**历史不可混算**
       （`labeler` 已落库，跨口径的天数会被排除而非补齐）。
    """
    from core import sentiment_reference as SR

    mode = st.radio("标注方式", [LABEL_RULE, LABEL_LLM], horizontal=True,
                    key=f"{key}_labelmode",
                    help="词表：零成本、可复现，但对「描述过去仓位 / 反讽 / 条件句」"
                         "结构上分辨不了；模型：直读原文逐条标注并出报告，"
                         "数字仍由代码从标签计算。两种口径的历史**不互相比较**。")
    if mode == LABEL_LLM:
        _render_llm_card(pack, source, key)
        return

    try:
        res = SR.build_for_source(pack["items"], source=source)
    except Exception as e:                      # noqa: BLE001
        # ⚠️ 不能静默：落库/建卡失败时"数字不显示"会被误读成"没有情绪数据"
        st.warning(f"⚠️ 情绪指标计算失败（{type(e).__name__}）：{e}")
        return
    snap, base = res["snapshot"], res["baseline"]

    _render_snap_metrics(snap, base)

    with st.expander("🔎 给大模型的参照块（原文 + 引用规则）", expanded=False):
        st.code(SR.render_block(snap, base), language="text")

    if st.button("🧠 生成情绪判定（调 1 次模型）", key=f"{key}_judge"):
        with st.spinner("正在判定…"):
            v = SR.judge(snap, base)
        st.session_state[f"{key}_verdict"] = v
    v = st.session_state.get(f"{key}_verdict")
    if v:
        if v.get("status") == "ok":
            st.info(f"**{v['stance']}**（置信度 {v['confidence']}）—— {v.get('summary')}")
            if v.get("divergence"):
                st.caption(f"与价量关系：{v['divergence']}")
            for w in v.get("warnings") or []:
                st.caption(f"⚠️ {w}")
            for c in v.get("cautions") or []:
                st.caption(f"注意：{c}")
            st.caption("⚠️ 判定仅描述情绪位置，不构成买卖建议；"
                       "规则层数字可能误判，模型已被要求以原文为准。")
        else:
            st.warning(f"情绪判定不可用（{v.get('status')}）：{v.get('summary')}")


def _render_zhihu_items(items, label="话题"):
    """条目渲染（两种采集方式共用 —— 字段同名同义，渲染也不该分叉）。"""
    for it in items:
        votes = it.get("votes")
        st.markdown(f"**{it['title'] or '（无标题）'}**")
        pub = it.get("published")
        when = ""
        if isinstance(pub, (int, float)) and pub > 0:
            when = "｜" + time.strftime("%Y-%m-%d %H:%M", time.localtime(pub))
        st.caption(f"{label}：{it['topic']}｜作者：{it['author'] or '未知'}"
                   f"｜赞同 {votes if votes is not None else '—'}"
                   f"｜评论 {it.get('comments') if it.get('comments') is not None else '—'}{when}")
        st.markdown((it["text"][:700] + ("…" if len(it["text"]) > 700 else ""))
                    or "_（正文为空）_")
        if it.get("url"):
            st.markdown(f"[查看原文]({it['url']})")
        st.divider()


def _render_zhihu_member_block():
    """按博主取「想法」——**网页端 v4，非公开接口**，与官方通道完全独立。"""
    from data import zhihu_opinions as Z

    st.warning("⚠️ 本方式走**网页端内部接口**（`www.zhihu.com/api/v4/members/<id>/pins`），"
               "**不是**官方开放平台：不消耗开放平台额度、不需要 Access Secret，"
               "但**随时可能因知乎改版/风控失效**。实测仅「想法」可免登录取；"
               "回答与文章需要登录 Cookie 与 `x-zse-96` 签名，本模块**不做**。")

    members = Z.load_members()
    edited = st.text_area("关注博主（每行一个；可直接粘 `https://www.zhihu.com/people/xxx` 整条链接）",
                          value="\n".join(members), height=110, key="zhihu_members_edit")
    b1, b2, b3 = st.columns([1, 1, 3])
    with b1:
        if st.button("💾 保存博主", key="zhihu_members_save"):
            new = [t.strip() for t in edited.splitlines() if t.strip()]
            if Z.save_members(new):
                st.success(f"已保存 {len(new)} 个博主")
            else:
                st.error("保存失败（写入异常），未改动原配置")
    with b2:
        fetch = st.button("📥 采集想法（不耗额度）", type="primary", key="zhihu_member_fetch")
    if fetch:
        with st.spinner("正在按博主取想法…"):
            items, meta = Z.collect_members(Z.load_members(), per_member=20)
        st.session_state[_ZH_MEMBER_KEY] = {"items": items, "meta": meta,
                                            "ts": time.strftime("%H:%M:%S")}

    pack = st.session_state.get(_ZH_MEMBER_KEY)
    if not pack:
        st.caption("点「📥 采集想法」按上面的博主取最新想法（结果只用于人工阅读）。")
        return

    items, meta = pack["items"], pack["meta"]
    st.caption(f"采集时间 {pack['ts']}｜博主 {meta['asked']} 个｜去重后 {meta['got']} 条"
               f"｜本通道**不消耗**开放平台额度")
    for e in meta.get("errors") or []:
        st.warning(f"⚠️ {e}")
    le = meta.get("last_error") or {}
    if le.get("reason"):
        st.error(f"**失败原因**（{le.get('step')}）：{le.get('reason')}"
                 + (f"　\n\n{le['detail']}" if le.get("detail") else ""))
    if meta.get("by_member"):
        detail = "｜".join(
            f"{r['member']}: {r['got']} 条" + (f"（{r['error']}）" if r.get("error") else "")
            for r in meta["by_member"])
        st.caption("逐博主结果 —— " + detail)
    if not items:
        st.info("本次未取到任何想法 —— 上方已列出具体原因；如实展示，不补造。")
        return

    st.divider()
    st.markdown("#### 📊 市场情绪参照")
    st.caption("由上面的想法**自动提炼**（规则层判方向 → 结构化指标 → 相对自身历史）。"
               "⚠️ 只用于描述情绪位置：**不参与评分、不决定仓位、不构成买卖建议**。")
    _render_sentiment_card(pack, "zhihu:member", "zhimember")

    st.divider()
    _render_zhihu_items(items, label="博主")


def _render_zhihu_block():
    from data import zhihu_opinions as Z

    # ⚠️ 两种采集方式**通道完全不同**：话题走官方开放平台（要 token、扣额度）；
    #    博主走网页端 v4（免 token、不扣额度、但属非公开接口）。
    #    必须让用户**显式选**，不能揉成"一键采集" —— 否则"这次到底扣没扣额度"说不清。
    mode = st.radio("采集方式", [MODE_TOPIC, MODE_MEMBER],
                    horizontal=True, key="zhihu_mode",
                    help="按话题：官方开放平台，消耗每日额度；"
                         "按博主：网页端非公开接口，不消耗额度但随时可能失效。")
    if mode == MODE_MEMBER:
        _render_zhihu_member_block()
        return

    ok, msg = _cached_zhihu_auth()              # 0 额度探测（缓存 5 分钟，避免每次 rerun 都发请求）
    c1, c2 = st.columns([3, 2])
    with c1:
        st.caption(("✅ " if ok else "⚠️ ") + msg)
    with c2:
        st.caption(Z.quota_note())

    # ---- 话题增删（写入 config/zhihu_topics.json，mtime 缓存 ⇒ 改完即生效，不用重启）
    topics = Z.load_topics()
    edited = st.text_area("关注话题（每行一个；改完点保存即生效，无需重启）",
                          value="\n".join(topics), height=110, key="zhihu_topics_edit")
    b1, b2, b3 = st.columns([1, 1, 3])
    with b1:
        if st.button("💾 保存话题", key="zhihu_topics_save"):
            new = [t.strip() for t in edited.splitlines() if t.strip()]
            if Z.save_topics(new):
                st.success(f"已保存 {len(new)} 个话题")
            else:
                st.error("保存失败（写入异常），未改动原配置")
    with b2:
        fetch = st.button("📥 采集（消耗额度）", type="primary", key="zhihu_fetch")
    if fetch:
        with st.spinner("正在按话题检索…"):
            items, meta = Z.collect(Z.load_topics(), per_topic=5)
        st.session_state[_ZH_KEY] = {"items": items, "meta": meta,
                                     "ts": time.strftime("%H:%M:%S")}

    pack = st.session_state.get(_ZH_KEY)
    if not pack:
        st.caption("点「📥 采集」按上面的话题检索知乎内容（结果只用于人工阅读）。")
        return

    items, meta = pack["items"], pack["meta"]
    st.caption(f"采集时间 {pack['ts']}｜话题 {len(meta['topics'])} 个｜"
               f"去重后 {meta['got']} 条｜"
               f"额度 {meta['quota']['used']}/{meta['quota']['limit']}")
    # ⚠️ 取不到必须**说清是哪一步失败**（2026-10-15 用户报"仍未返回结果"）：
    #    原实现只显示"可能额度用尽或关键词过窄"——这句话没有鉴别力，
    #    用户无法判断到底是没配 token、鉴权失败、网络不通还是真的没结果。
    for e in meta.get("errors") or []:
        st.warning(f"⚠️ {e}")
    le = meta.get("last_error") or {}
    if le.get("reason"):
        st.error(f"**失败原因**（{le.get('step')}）：{le.get('reason')}"
                 + (f"　\n\n{le['detail']}" if le.get("detail") else ""))
    if meta.get("per_topic"):
        detail = "｜".join(
            f"{r['topic']}: {r['got']} 条" + (f"（{r['error']}）" if r.get("error") else "")
            for r in meta["per_topic"])
        st.caption("逐话题结果 —— " + detail)
    if not items:
        st.info("本次未取到任何内容 —— 上方已列出具体原因；如实展示，不补造。")
        return

    # ---- LLM 总结（**只用于展示**，结果不回灌分析链）
    if st.button("🧠 用 LLM 汇总成本段「市场观点概览」（仅展示，不参与分析）",
                 key="zhihu_summarize"):
        with st.spinner("调用大模型汇总…"):
            try:
                from llm.deepseek_client import DeepSeekClient, TEMP_EXPLORATORY
                corpus = "\n\n".join(
                    f"【{i+1}】{it['title']}（作者：{it['author']}，赞同 {it['votes']}）\n"
                    f"{it['text'][:600]}"
                    for i, it in enumerate(items[:12]))
                client = DeepSeekClient()
                summary = client.call_api(
                    [{"role": "system", "content":
                      "你是资料整理助手。只依据给出的知乎内容做要点归纳，"
                      "**不得引入任何外部事实、数据、机构或日期**；"
                      "若内容与股市无关，明确指出。"},
                     {"role": "user", "content":
                      f"请把下面这些知乎内容归纳成 5 条以内的要点，并标注每条来自第几条：\n\n{corpus}"}],
                    temperature=TEMP_EXPLORATORY, max_tokens=900,
                    purpose="ui.zhihu_summary")
                st.session_state["zhihu_summary"] = summary
            except Exception as e:                       # noqa: BLE001
                st.error(f"汇总失败：{type(e).__name__}: {str(e)[:120]}")
    if st.session_state.get("zhihu_summary"):
        st.markdown(st.session_state["zhihu_summary"])
        st.caption("⚠️ 上面由大模型生成，**仅供人工参考**；它有可能是错的，"
                   "请以下方原文为准。本节不参与任何分析或评分。")

    st.divider()
    _render_zhihu_items(items, label="话题")


if visible(_SEL, SEC_ZHIHU):
    st.subheader(SEC_ZHIHU)
    # ⚠️ 这行声明必须放在**分支之外**（离线模式也渲染）：它是本节的使用边界，
    #    不能因为"这一节被跳过"就消失 —— 否则离线/测试环境下看不到"不参与分析"的承诺。
    st.caption("⚠️ 本节**仅供人工参考**：不注入任何提示词、不参与 `agents.news` / "
               "`news_industry_review`、**不参与任何模型分析与评分**。"
               "另：知乎**官方接口读不了指定大V**（`/user/contents` 只回 token 本人，"
               "需对方逐个 OAuth 授权）⇒ 默认按**话题关键词**检索；"
               "要按某个博主取，请切到上方「按博主采集」"
               "（网页端非公开接口，免 token、不耗额度，但仅「想法」可读、随时可能失效）。")

    # ⚠️ BIGA_UI_OFFLINE：AppTest 会**真执行页面脚本**，联网调用会拖慢/出错 ⇒ 与热榜同口径跳过
    if os.environ.get("BIGA_UI_OFFLINE"):
        st.caption("（离线模式：知乎话题参考已跳过——这只影响这一节）")
    else:
        try:
            _render_zhihu_block()
        except Exception as e:                      # noqa: BLE001
            st.warning(f"⚠️ 知乎话题参考不可用（{type(e).__name__}）："
                       "不影响上面的评估与快讯——它们来自库内数据。")
