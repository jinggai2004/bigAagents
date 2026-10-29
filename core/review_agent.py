"""
每日复盘 Agent：收盘后自动汇总当日行情 + 流水线结果，生成 Markdown 复盘报告，
并可选推送到钉钉/飞书 webhook。

数据输入（全部容忍缺失，缺什么少写什么）：
  - output/daily_picks/latest.json（当日流水线：候选/回测摘要/跳过原因）
  - 三大指数收盘行情（data/realtime）
  - 自选股涨跌 TOP（data/realtime + watchlist）
  - 行业资金热度（同花顺，data/fund_flow）
产出：output/daily_reports/review_YYYYMMDD.md 与 latest.md
"""

import os
import re
import sys
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import store

from llm.deepseek_client import DeepSeekClient, TEMP_EXPLORATORY
from config.model_routing import resolve_model

_logger = logging.getLogger(__name__)

# 复盘"大盘概览"要讲的 5 个指数（三大 + 沪深300 + 中证1000）。
# ⚠️ 不要直接用 `realtime.INDEX_CODES`（只有 3 个）——见 `_collect_market_data` 的说明：
#    少给两个宽基，模型就会自己编大小盘涨跌幅。
_REVIEW_INDEX_CODES = ("sh000001", "sz399001", "sz399006", "sh000300", "sh000852")

# 数据根必须**运行时**解析：打包后 `__file__` 位于只读包 `_internal/` 里，
# 用它推导会指向空目录 ⇒ 读到空数据（2026-10-25 实测：双击 exe 后
# 选股/复盘/研究全 missing、交易台变空账户、系统页看不到调度器）。
# 规则见 `config/paths.data_root()`；源码运行结果与原先一致。
from config.paths import data_root as _paths_data_root  # noqa: E402
_PROJECT_ROOT = _paths_data_root()
REPORT_DIR = os.path.join(_PROJECT_ROOT, "output", "daily_reports")

# ⚠️ 2026-10-14：原 `REVIEW_MODEL = config.QWEN_PLUS_MODEL or "qwen-max"` 是**导入期常量**，
#    改 .env 后不重启进程就不生效（10-13 踩过的坑）。现改为运行时解析：
#    `review.daily` 默认档=deep（QWEN_PLUS_MODEL），可在「系统配置 → 🎚️ 功能模型分配」
#    单独指定；解析优先级见 config/model_routing.py。调用点见下方 run_review。

REVIEW_SYSTEM_PROMPT = """你是一名券商策略分析师，撰写专业的 A 股每日收盘复盘（卖方晨报收盘篇风格）。
基于系统提供的当日真实数据撰写，数据包括：三大指数、市场检测14指标、自选股表现、行业资金流、
市场讨论观点（联网检索）、历史相似日统计、当日要闻、选股流水线结果，以及
**今日决策清单 / 决策质量评估 / 明日待触发**（系统程序化生成）——提供什么用什么，
缺数据的小节注明"数据暂缺"并降低该维度权重。

⚠️ 报告开头三章 **今日决策清单 / 决策质量评估 / 明日待触发** 已由系统**程序化写入**
（数字来自数据库，未经模型）。你**不要重复输出这三章**，也不要改写或重新计算其中的数字；
你可以引用它们作为判断依据。

⚠️ **阿深今日** 同样由系统程序化写入（他的值班轮次、真挂着的等待条件、账户成交）。
不要重复输出、不要替他美化。如果数据里显示他"只有结论没有实质动作"，
或者"动作全是系统补正出来的"，你可以**如实点评他这一天主不主动**——
但点评要落到"没挂条件/没动作"这类事实，不要泛泛说"表现谨慎"。

写作框架（Markdown，二级标题，从"昨日观察核对"开始；无待核对项则从"大盘概览"开始）：
1. **昨日观察核对**（仅当系统提供了"昨日观察待核对"）— 逐条核对昨日观察点，给出结论与数据依据。
2. **大盘概览** — 指数结构与风格特征，2-3句给出当日市场性质判断（如"缩量调整/放量突破/分化整理"）；可引用市场检测14指标（趋势/量能/情绪/结构四维得分）作为判断依据。
3. **资金与板块** — 行业资金净流入/流出的方向含义，与领涨领跌板块是否共振；若有市场讨论观点，归纳多空力量对比。
4. **自选股跟踪** — 领涨领跌个股及其逻辑归因（结合板块与要闻），提示需跟踪的变化。
5. **历史统计参考** — 引用相似日数据必须表述为"历史统计显示…仅供参考"，禁止说"预计会涨/跌"。
6. **消息面要点** — 当日要闻中影响市场的 2-3 条及影响方向。
7. **风险提示与明日观察** — 2-3 条**可验证**的观察点（点位/板块/数据事件），并明确当前主要风险。

硬性要求：
- 只引用提供的数据，禁止编造任何数字；数字保留一位小数。
- **单位与口径不许改写**：指标 detail 里写"成交量X亿股"是**股数**，不是金额，
  不得改写成"成交额X亿元"；detail 若注明"系上一交易日数据"，引用时必须保留该说明，
  不得当作当日数据。指标口径不确定时，宁可描述方向（放量/缩量），不要报绝对金额。
- 专业术语适度（缩量/放量、多空共振等）；逻辑链清晰：数据→含义→含义的边界。
- 全文 600-900 字；语气客观克制，不下买卖结论。
- 不要输出一级标题（#），报告头部由系统添加；从二级标题（##）开始。
- ⚠️ **观察点必须可验证**（2026-10-10 用户反馈"昨日观察核对只能写无法验证"）：每条观察点
  都要自带 **阈值 + 判定日 + 判定口径**，写法示例：
  "若 10-12 收盘价 >3825 且当日成交量 >1058 亿股，则确认短线企稳（两条同时满足才算达成）"。
  禁止"耐心等待""关注量能变化"这类无法判定的空话 —— 次日核对时它们只能被写成"无法验证"。
- 报告末尾（免责声明之前）追加一个代码块，回结构化核对与观察点数据（用于预测跟踪统计）：
  ```verification
  {"verifications": [{"id": 昨日观察id, "result": "达成|未发生|部分达成|无法验证", "basis": "用到的今日数据（如：收3813.79、量1015亿股）"}],
   "new_observations": [{"kind": "观察点|风险", "content": "含阈值+判定日+判定口径的完整表述"}]}
  ```
  new_observations 即"风险提示与明日观察"一节的条目，**content 必须自包含阈值与判定日**。
- 末尾固定：*以上为程序化数据分析，仅供参考，不构成投资建议。*
"""


def _fmt_pct(v, digits: int = 2) -> str:
    return f"{v:+.{digits}f}%" if isinstance(v, (int, float)) else "—"


def _fmt_ret(v, digits: int = 2) -> str:
    """把**小数比值**（0.0176 表示 1.76%）格式化成百分比。

    ⚠️⚠️ 与 `_fmt_pct` 的分工（2026-10-23 事故）：库里 `ret_5d` / `bench_ret_5d`
    以及 `avg_excess`、`avg_ret` 都是**比值**，而 `pnl_pct` / `total_pnl_pct` /
    `benchmark["pct"]` 已经是**百分点**。两者混用 `_fmt_pct` 的后果是
    超额收益被显示成 1/100 —— 实测「全体超额 +1.76%」在复盘里印成「+0.02%」，
    分档单调性方向仍对、量级全错，读者会以为选股毫无价值。
    """
    return _fmt_pct(v * 100 if isinstance(v, (int, float)) else v, digits)


def render_decision_sections(digest: dict) -> str:
    """把决策汇总渲染成**程序填充**的三章（不经 LLM）。

    ## 为什么要程序渲染（2026-10-10 用户反馈"复盘较简略，看不出今日操作是否正确"）
    复盘原先只讲盘面，**没有一个字讲今天选了什么、分析了什么、账户动没动** ——
    因为模型看不到这些（`_collect_market_data` 没有它们），也因为没有统一视图。
    这里由程序把 `core.decision_digest` 的结果直接写成 Markdown 插进报告：
    数字**不经过模型**，因此不会被改写（红线 2）。

    样本未到期时写明"预计 X 日"而不是省略 —— "还在等"与"坏了"是两件事。
    """
    lines: list = []
    today = digest.get("today") or {}
    quality = digest.get("quality") or {}
    actions = digest.get("actions") or {}

    # ── ① 今日决策清单 ──
    lines.append("## 今日决策清单")
    if not today.get("date"):
        lines.append("- 数据不可得（决策汇总未返回今日块）")
    screens = today.get("screens") or []
    if screens:
        head = "、".join(
            f"{s['name'] or s['code']}({s['code']}) {s['score']}"
            if s.get("score") is not None else f"{s['name'] or s['code']}({s['code']})"
            for s in screens[:5])
        strategies = sorted({s.get("strategy") or "—" for s in screens})
        degraded = [s for s in screens if s.get("degraded")]
        # ⚠️ 2026-10-29：`screens` 现在是**推荐名单**（daily_picks，已做行业分散）。
        #    以前这里读的是 `screen_picks` 扩池（110 行、纯分数排名），于是报告写
        #    "选股 110 只 → 交通/农业/工商/建设"，用户看到 4 只银行 —— 而真实名单
        #    只有 2 只（交通 + 农业），配额一直是生效的。**候选池数量单独标注**，
        #    别再把"扩池排名"当"推荐名单"。
        pool_n = len(today.get("screen_pool") or [])
        final = all(s.get("is_final") for s in screens)
        head_txt = "推荐" if final else "候选（当日无最终名单，退化为扩池前 5 名）"
        line = f"- **选股**：{len(screens)} 只{head_txt} —— {head}"
        if final and pool_n:
            line += f"｜候选池 {pool_n} 只（纯分数排名，未做行业分散）"
        line += f"｜初筛策略：{'、'.join(strategies)}"
        if degraded:
            line += "（⚠️ 含「内置精选池」降级候选，候选域被收窄）"
        lines.append(line)
    analyses = today.get("analyses") or []
    if analyses:
        detail = "、".join(
            f"{a['name'] or a['code']}={a.get('decision_label')}"
            f"（{a.get('advice_status')}）" for a in analyses[:8])
        dup = sum(1 for a in analyses if a.get("duplicate"))
        line = f"- **深度分析**：{len(analyses)} 条 —— {detail}"
        if dup:
            line += f"｜⚠️ 其中 {dup} 条为同日同票重复"
        lines.append(line)
    state_line = (f"- **模拟账户**：今日 {len(today.get('trades') or [])} 笔成交；"
                  f"持仓 {len(today.get('positions') or [])} 只")
    if today.get("positions"):
        state_line += "（" + "、".join(
            f"{p['name']} {p['quantity']}股" for p in today["positions"][:5]) + "）"
    lines.append(state_line)
    if today.get("empty_reasons"):
        lines.append(f"- 说明：{'；'.join(today['empty_reasons'])}")

    # ── ② 决策质量评估 ──
    lines.append("")
    lines.append("## 决策质量评估")
    scr = quality.get("screen") or {}
    if not scr:
        # 空字典 = 那一块没算出来（红线 3：缺数据要显式说，不能写成 "0 条"）
        lines.append("- 选股质量：数据不可得（决策汇总未返回该块）")
    elif scr.get("status") == "insufficient_sample":
        note = f"，首批预计 {scr['next_ready_date']}" if scr.get("next_ready_date") else ""
        lines.append(f"- **选股 T+{quality.get('horizon', 5)}**：已回填 "
                     f"**{scr.get('n_backfilled', 0)}/{scr.get('n_backfilled', 0) + scr.get('n_pending', 0)}**"
                     f"（样本未到期{note}）")
    else:
        top = scr.get("top5_excess")
        allx = scr.get("all_excess")
        lines.append(f"- **选股 T+{quality.get('horizon', 5)}**：已回填 {scr.get('n_backfilled')} 条｜"
                     f"TOP5 平均超额 {_fmt_ret(top)}，全体平均超额 {_fmt_ret(allx)}")
        for b in scr.get("by_score_band", []):
            if b["n"]:
                lines.append(f"  - 分数 {b['band']}：{b['n']} 只，平均超额 "
                             f"{_fmt_ret(b['avg_excess'])}，胜率 {b['win_rate']}")
    ana = quality.get("analysis") or {}
    if not ana:
        lines.append("- 分析准确率：数据不可得（决策汇总未返回该块）")
    elif ana.get("status") == "insufficient_sample":
        note = f"，首批预计 {ana['next_ready_date']}" if ana.get("next_ready_date") else ""
        lines.append(f"- **分析准确率**：已回填 **{ana.get('n_backfilled', 0)}** 条（样本未到期{note}）")
    else:
        parts = [f"{b['label']} {b['n']} 条/平均 {_fmt_ret(b['avg_ret'])}"
                 + (f"/命中率 {b['hit_rate']}" if b.get("hit_rate") is not None else "")
                 for b in ana.get("by_label", []) if b["n"]]
        lines.append("- **分析准确率**：" + "；".join(parts))
        lines.append(f"  - 期间触及目标 {ana.get('hit_target_rate')}｜触及止损 {ana.get('hit_stop_rate')}")
    pf = quality.get("portfolio") or {}
    if pf.get("status") == "ok":
        seg = [f"{c['name']} {_fmt_pct(c['pnl_pct'])}（贡献 {c['pnl']:+,.0f}）"
               for c in (pf.get("contributors") or [])]
        line = (f"- **持仓归因**：总资产 {pf.get('total_value'):,.2f}，"
                f"累计 {_fmt_pct(pf.get('total_pnl_pct'))}")
        if seg:
            line += "；" + "；".join(seg)
        bm = pf.get("benchmark") or {}
        if bm.get("pct") is not None:
            line += f"｜同期上证 {_fmt_pct(bm['pct'])}（{bm.get('start')}~{bm.get('end')}）"
        else:
            line += (f"｜同期指数不可得（{pf.get('since') or '—'} 起，"
                     f"指数特征库未覆盖到 {pf.get('price_date')}）")
        lines.append(line)
    elif not pf.get("status"):
        lines.append("- 持仓归因：数据不可得（决策汇总未返回该块）")

    # ── ③ 明日待触发 ──
    lines.append("")
    lines.append("## 明日待触发")
    pos = actions.get("positions") or []
    if not actions.get("as_of") and not pos:
        lines.append("- 数据不可得（决策汇总未返回该块）")
    elif pos:
        for p in pos:
            bits = []
            if p.get("to_stop_pct") is not None:
                bits.append(f"距止损位 {p['to_stop_pct']:+.2f}%（{p.get('stop')}）")
            if p.get("to_target_pct") is not None:
                bits.append(f"距目标位 {p['to_target_pct']:+.2f}%（{p.get('target')}）")
            if p.get("triggered") == "stop":
                bits.append("🔴 **已触及止损**")
            elif p.get("triggered") == "target":
                bits.append("🟢 **已触及目标**")
            if not bits:
                bits.append(p.get("note") or "无止损/目标位可跟踪")
            lines.append(f"- 持仓 {p['name']}（{p['code']}）：" + "；".join(bits))
    else:
        lines.append("- 持仓：无")
    watch = actions.get("watch") or []
    for w in watch[:5]:
        cond = "；".join(w.get("entry_conditions") or []) or "（未给出明确的入场条件）"
        extra = f"｜距入场 {w['to_entry_pct']:+.2f}%" if w.get("to_entry_pct") is not None else ""
        lines.append(f"- 观察 {w['name']}（{w['code']}，{w.get('advice_status')}）{extra}：需满足 {cond}")
    for r in actions.get("risks") or []:
        lines.append(f"- 风险：{r.get('text')}")
    if actions.get("empty_reasons"):
        for reason in actions["empty_reasons"]:
            lines.append(f"- 提示：{reason}")
    return "\n".join(lines)


def _collect_market_data() -> dict:
    """收集复盘所需数据，各项失败独立容忍"""
    data = {}
    try:
        from data import realtime
        # ⚠️⚠️ **必须给全 5 个指数**（2026-10-23 事故）：
        #   `realtime.INDEX_CODES` 只有上证/深成/创业板，复盘数据块里就只有这 3 行，
        #   而模型在写"大盘概览"时**天然要讲大小盘风格**（沪深300 vs 中证1000）⇒
        #   拿不到就**编**：实测 10-23 报告写出"中证1000 -1.6% vs 沪深300 -3.2%"，
        #   而当日真实值是 +2.38% / +1.71%（符号全反）。
        #   这里显式多取两个宽基（`get_realtime_quotes` 对指数同样适用），
        #   让"编不下去"变成"有据可依"。缺了就缺（下面的 except 不变）。
        idx = realtime.get_realtime_quotes(_REVIEW_INDEX_CODES)
        data["指数"] = [
            {"名称": row["name"], "收盘": row["price"], "涨跌幅%": row["change_pct"]}
            for row in idx.values() if row
        ]
    except Exception as e:
        _logger.warning("[复盘] 指数数据失败: %s", e)
    try:
        from data import realtime, watchlist
        quotes = realtime.get_realtime_quotes(watchlist.watchlist_codes())
        rows = sorted((q for q in quotes.values() if q),
                      key=lambda r: r["change_pct"] or 0, reverse=True)
        data["自选股"] = [
            {"名称": r["name"], "代码": r["code"], "收盘": r["price"], "涨跌幅%": r["change_pct"]}
            for r in rows
        ]
    except Exception as e:
        _logger.warning("[复盘] 自选股行情失败: %s", e)
    try:
        # 优先读盘后落库的行业资金流快照（盘后采集任务 15:40 已写过），
        # 现场拉只有在没有快照时才发生。
        from data.fund_flow import get_stored_industry_flow, get_industry_flow_snapshot
        date, df = get_stored_industry_flow()
        if df is None:
            df = get_industry_flow_snapshot()
            date = None
        if df is not None:
            data["行业净流入前5"] = df.head(5)[["行业", "净额"]].to_dict("records")
            data["行业净流出前5"] = df.tail(5)[["行业", "净额"]].to_dict("records")
            if date:
                data["行业资金流数据日期"] = date
    except Exception as e:
        _logger.warning("[复盘] 行业资金流失败: %s", e)
    try:
        # 隔夜外盘：只影响**开盘情绪**（宽基日度相关性约 0.10，±2% 以内无决定力）。
        # 必须带上"数据日期"——周一取到的是上周五的美股 bar，不带日期就会被当成"昨夜"。
        from data.us_market import get_overnight_us
        us = get_overnight_us()
        if us.get("unavailable"):
            data["隔夜外盘"] = f"取不到：{us.get('note', '原因未知')}"
        else:
            data["隔夜外盘"] = {
                "数据日期": us.get("data_date"),
                "查询日期(今天)": us.get("query_date"),
                "指数涨跌": {r["name"]: f"{r['change_pct']:+.2f}%" for r in us.get("indexes", [])},
                "波动等级": us.get("level_text"),
                "解读": us.get("note"),
            }
    except Exception as e:
        _logger.warning("[复盘] 外盘获取失败: %s", e)
    try:
        from core.market_detector import MarketDetector
        signal = MarketDetector().detect()
        # ⚠️ 字段名是 `scores`，不是 `dimension_scores`。
        # 原实现写错字段名 → AttributeError 被 except 吞掉 →
        # 复盘报告里"市场检测14指标"**从来没有出现过**，且不报错（静默缺失）。
        data["市场检测14指标"] = {
            "市场状态": signal.market_state, "总分": signal.total_score,
            # ⚠️⚠️ `position_limit` 是**比值**（0.20 = 20%），键名带 "%" 必须换算。
            #    原样塞进去 ⇒ 模型写"仓位上限0.2%"，读起来像"几乎不能开仓"，
            #    实际是 20%。与 `_fmt_pct` 那处是同一类错误（比值当百分点）。
            "仓位上限%": round(float(signal.position_limit) * 100, 1),
            "分项评分": {k: {"score": v.get("score"), "max": v.get("max_score"),
                            "weight": v.get("weight"), "detail": v.get("detail")}
                        for k, v in (signal.scores or {}).items()},
        }
    except Exception as e:
        _logger.warning("[复盘] 市场检测失败: %s", e)
    try:
        from data.news_feed import get_latest_news
        items = get_latest_news(limit=15, hours=24)
        if items:
            data["今日要闻"] = [
                {"时间": i["published_at"], "来源": i["source"],
                 "内容": (i["title"] or i["content"])[:100], "情绪": i["sentiment"]}
                for i in items]
    except Exception as e:
        _logger.warning("[复盘] 新闻快讯失败: %s", e)
    try:
        from data.market_opinions import get_market_opinions
        op = get_market_opinions()
        if "error" not in op:
            data["市场观点(网络讨论摘要)"] = {
                "date": op.get("date"), "opinions": op.get("opinions", []),
                "说明": "AI对搜索结果的归纳摘要，仅供参考",
            }
    except Exception as e:
        _logger.warning("[复盘] 市场观点失败: %s", e)
    # ── 市场情绪参照（2026-10-23）──────────────────────────────────
    # 与上面「市场观点」的区别：
    #   · 市场观点 = 搜索级摘要（来源不可控、无样本量、无分母）
    #   · 情绪参照 = 从**实际采集到的发言**里做结构化统计（有分母、有覆盖率、
    #     有置信区间、有相对自身的历史分位）
    # ⚠️ 刻意**只给结构化数字**，不把帖子原文塞进复盘提示词：
    #    复盘是叙述性输出，逐条原文既占 token 又容易让模型"照着抄一句当结论"。
    #    原文核对留给页面（人在看的时候才需要）。
    # ⚠️ 取不到时写"不可用"并说明原因，**不省略该字段**：
    #    省略会让模型以为"今天没有异常情绪"，而实际是"今天没取到"。
    try:
        from core import sentiment_reference as SR
        ref = SR.load_reference_block()
        data["市场情绪参照"] = ref
    except Exception as e:
        _logger.warning("[复盘] 情绪参照失败: %s", e)
        data["市场情绪参照"] = {"状态": "不可用", "原因": f"{type(e).__name__}: {e}"}
    try:
        from data.similar_days import find_similar_days
        sim = find_similar_days("sh000001", top_k=5)
        if "error" not in sim:
            data["历史相似日参考"] = {
                "当前": sim.get("target"),
                "相似日及后续表现": sim.get("similar"),
                "统计分布": sim.get("stats"),
                "说明": "统计参考非预测",
            }
    except Exception as e:
        _logger.warning("[复盘] 相似日检索失败: %s", e)
    try:
        latest_path = os.path.join(_PROJECT_ROOT, "output", "daily_picks", "latest.json")
        if os.path.exists(latest_path):
            with open(latest_path, encoding="utf-8") as f:
                latest = json.load(f)
            # ⚠️ F2（2026-10-18 审查）：还必须看 mode —— 盘前 09:25 的选股
            #    （mode="screen"）也会把 latest.json 的 date 刷成今天，
            #    只比 date 会把**上午的候选**当"今日盘后流水线"。
            if (latest.get("date") == datetime.now().strftime("%Y-%m-%d")
                    and latest.get("mode") in ("rule", "llm")):
                picks = [{"代码": p.get("股票代码"), "名称": p.get("股票名称") or "—",
                          "评分": p.get("score")} for p in latest.get("picks", [])]
                bt = latest.get("backtest") or {}
                data["流水线"] = {
                    "候选": picks, "模式": latest.get("mode"),
                    "回测摘要": {k: bt.get(k) for k in ("total_return", "sharpe_ratio",
                                                        "information_ratio", "total_trades") if bt},
                }
            else:
                if latest.get("date") == datetime.now().strftime("%Y-%m-%d"):
                    data["流水线"] = (f"今日只有 mode={latest.get('mode')!r} 的产出，"
                                      "盘后全流程（rule/llm）尚未运行")
                else:
                    data["流水线"] = "今日流水线尚未运行"
    except Exception as e:
        _logger.warning("[复盘] 流水线结果读取失败: %s", e)
    return data


def push_report(markdown_text: str, dedupe_key: str = None) -> dict:
    """按配置推送复盘到钉钉/飞书（未配置则跳过）

    Phase 1：推送实现已抽到 core/push.py，供复盘与盘中异动预警共用；
    本函数保留为复盘专用入口（标题固定为"bigAagents 复盘 + 日期"）。

    ⚠️ `dedupe_key`（2026-10-23）：同日只推一次。
    10-23 实测：13:01 手动重放 + 15:50 定时任务把同一份复盘推了两遍。
    """
    from core.push import push_markdown
    return push_markdown(markdown_text, title=f"bigAagents 复盘 {datetime.now():%Y-%m-%d}",
                          dedupe_key=dedupe_key)


# ── 推送用摘要（2026-10-23）────────────────────────────────────────────
# 为什么不让模型再写一段摘要：**决策数字不经 LLM**（本项目铁律），
# 而且复盘全文已经 4000+ 字，手机上要滚十几屏。摘要全部由**代码渲染**，
# 只从模型正文里取**第一段**（市场定性），不新增任何 LLM 调用。
def _extract_lede(markdown: str, max_chars: int = 220) -> str:
    """取模型正文的**第一段**（通常是"大盘概览"）作为结论句。

    ⚠️ 只取、不改写：模型写什么就是什么，不做二次概括。
    """
    for block in (markdown or "").split("\n\n"):
        b = block.strip()
        if not b or b.startswith("#"):
            continue
        # ⚠️ 跳过引用行（"> 生成时间 …"）——那是报告头部的元信息，不是市场结论
        if b.startswith(">"):
            continue
        b = re.sub(r"[*`>#]", "", b).strip()
        if not b:
            continue
        return b[:max_chars] + ("…" if len(b) > max_chars else "")
    return ""


def _integrity_brief(integrity_md: str) -> str:
    """从数据完整性表里抽出**缺失项**（摘要里只写缺什么，不写有的）。"""
    if not integrity_md:
        return ""
    missing = []
    for ln in integrity_md.splitlines():
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        if "⚠️" not in ln and "❌" not in ln:
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) >= 2 and cells[0] and cells[0] != "阶段":
            missing.append(f"{cells[0]}{cells[1]}")
    return "、".join(missing)


def build_review_summary(*, digest: dict, integrity_md: str, llm_markdown: str,
                         report_path: str, today: str) -> str:
    """复盘推送摘要（目标 ≤900 字）：结论 + 今日决策 + 明日待触发 + 风险 + 缺失项。

    ⚠️ 全部数字来自 `digest`（数据库程序渲染），**不经模型**。
    """
    today_blk = (digest or {}).get("today") or {}
    actions = (digest or {}).get("actions") or {}
    # ⚠️ 2026-10-23：整篇排版改由**可编辑模板**决定（`core/push_templates`），
    #    用户可在「系统配置」页改文案，改完即时生效、不用重启调度器。
    from core import push_templates as T

    lede = _extract_lede(llm_markdown)

    # ── 今日决策 ──
    screens = today_blk.get("screens") or []
    picks = "- 无记录"
    if screens:
        top = screens[:5]
        picks = (f"{len(screens)} 只 → TOP5：" + "｜".join(
            f"{s.get('name') or s.get('code')}({s.get('code')})"
            f"{(' ' + format(s.get('score'), '.1f')) if isinstance(s.get('score'), (int, float)) else ''}"
            for s in top) + "\n")
    analyses = today_blk.get("analyses") or []
    analyses_txt = "无"
    if analyses:
        uniq = {}
        for a in analyses:
            uniq.setdefault(str(a.get("code")), a)
        analyses_txt = "；".join(
            f"{a.get('name') or a.get('code')}({a.get('code')}) {a.get('advice_status')}"
            for a in list(uniq.values())[:5])
    positions = today_blk.get("positions") or []
    if positions:
        positions_txt = "；".join(
            f"{p.get('name') or p.get('code')}"
            f"（浮盈亏 {p.get('pnl_pct'):+.2f}%）" if isinstance(p.get("pnl_pct"), (int, float))
            else f"{p.get('name') or p.get('code')}" for p in positions[:5])
    else:
        positions_txt = "无"

    # ── 明日待触发 ──
    watch = actions.get("watch") or []
    watch_lines = []
    for w in watch[:5]:
        bits = []
        # ⚠️ `entry`（数值入场价）常常是 None —— 多数观察档只给了**文字条件**。
        #    此时给**第一条入场条件**的摘要（用户要知道"等什么"），不给就等于没说。
        if isinstance(w.get("to_entry_pct"), (int, float)):
            bits.append(f"距入场 {w['to_entry_pct']:+.1f}%")
        conds = w.get("entry_conditions") or []
        if conds:
            c0 = str(conds[0]).strip()
            bits.append(c0[:44] + ("…" if len(c0) > 44 else ""))
        watch_lines.append(f"- {w.get('name') or w.get('code')}({w.get('code')}，"
                           f"{w.get('advice_status')})" + ("：" + "；".join(bits) if bits else ""))
    pos_items = actions.get("positions") or []
    for p in pos_items[:3]:
        bits = []
        if isinstance(p.get("to_stop_pct"), (int, float)):
            bits.append(f"距止损 {p['to_stop_pct']:+.2f}%")
        if p.get("triggered") == "stop":
            bits.append("🔴 已触及止损")
        elif p.get("triggered") == "target":
            bits.append("🟢 已触及目标")
        if bits:
            watch_lines.append(f"- 持仓 {p.get('name') or p.get('code')}({p.get('code')})："
                               + "；".join(bits))
    watch_txt = ("\n".join(watch_lines) + "\n") if watch_lines else "- 无\n"

    # ── 风险 ──
    risks = actions.get("risks") or []
    risk_lines = []
    for r in risks[:3]:
        text = str(r.get("text") or r.get("reason") or "").strip()
        if text:
            risk_lines.append(f"- {text}")
    risks_txt = ("\n".join(risk_lines) + "\n") if risk_lines else ""

    missing = _integrity_brief(integrity_md)
    ctx = {
        "title": "A股每日复盘摘要", "date": today,
        "time": datetime.now().strftime("%H:%M"),
        "lede": lede or "（本次复盘未产出结论段）",
        "picks": picks, "analyses": analyses_txt, "positions": positions_txt,
        "watch": watch_txt, "risks": risks_txt,
        "integrity": ("缺失 " + missing) if missing else "各项齐全",
        "report_path": report_path,
        "footer": T.FOOTERS.get(T.TPL_REVIEW, ""),
    }
    return T.render_with(T.TPL_REVIEW, ctx)


def _parse_verification_block(markdown: str) -> dict | None:
    """从报告尾部解析 ```verification``` 结构化数据块"""
    m = re.search(r"```verification\s*(\{.*?\})\s*```", markdown, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as e:
        # "没有 verification 块"是正常状态，但"有块却解析不了"是**产物坏了**，
        # 两者都返回 None 会让上游无法区分 ⇒ 至少要能在日志里查到。
        _logger.debug("[复盘] verification 块解析失败（按无块处理）: %s", e)
        return None


def generate_review(push: bool = False) -> dict:
    """
    生成当日复盘报告（含预测跟踪闭环：昨日观察核对 → 落库 → 命中率统计）。

    Returns:
        {"report_path", "markdown", "push"(可选), "error"(可选)}
    """
    today = datetime.now().strftime("%Y-%m-%d")
    # 复盘负责把"全行业数据"取好（用户要求：盘后取一次，而不是让选股一次次跑）。
    # 幂等：当天快照已存在就只做一次 SELECT，不会重复拉全市场。
    try:
        from data.fund_flow import ensure_snapshot_captured
        snap = ensure_snapshot_captured()
        if snap.get("skipped"):
            _logger.info("[复盘] 资金流快照已存在（%s），跳过采集", snap.get("snapshot_date"))
        else:
            _logger.info("[复盘] 资金流快照采集完成：%s（个股 %s / 行业 %s）",
                         snap.get("snapshot_date"), snap.get("stocks"), snap.get("industries"))
    except Exception as e:
        _logger.warning("[复盘] 资金流快照采集失败（不影响复盘主流程）: %s", e)
    market = _collect_market_data()

    # ── 决策汇总（2026-10-10）：今日决策 / 质量评估 / 明日待触发 ──
    # 零联网、只读本地库；任何失败都**不影响复盘主流程**（缺三章而已，其余照常）。
    # 为什么要把原始数据喂给 LLM：它不重复输出这三章，但需要它们作为判断依据
    # （否则"风险提示与明日观察"会与当日持仓/观察档完全脱节）。
    digest: dict = {}
    try:
        from core.decision_digest import build_digest
        digest = build_digest()
        market["今日决策与质量数据"] = {
            "今日决策": digest.get("today"),
            "决策质量评估": digest.get("quality"),
            "明日待触发": digest.get("actions"),
        }
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[复盘] 决策汇总失败（报告将缺决策三章）: %s", e)

    # ── 阿深日报（2026-10-28 用户要求：「每日复盘加上阿深的每天报告」）──
    # 此前的复盘里**一个字都没有阿深**：上面那三章是主分析链的 trade_plan，
    # 他的值班记录、他挂的条件、他的账户都不在里面。
    # 两份输出分工：① `brief_for_llm` 喂给模型当依据（允许它点评）；
    # ② `render_markdown` 由**代码**写进报告（数字不经模型）。
    # 分工的原因是刚踩过的坑：他六轮都说"在等伊利 28 元下方"而条件表 0 条，
    # 若让模型凭 summary 自己写日报，它照样能把"没动"写成"持续跟踪"。
    ashen_md = ""
    try:
        from core import ashen_daily
        market["阿深今日"] = ashen_daily.brief_for_llm(today)
        ashen_md = ashen_daily.render_markdown(today)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[复盘] 阿深日报渲染失败（不影响报告主体）: %s", e)

    # 预测跟踪：昨日未核对观察点 + 历史命中率
    pending = store.get_unverified_observations()
    if pending:
        market["昨日观察待核对"] = pending
        market["预测跟踪说明"] = "请逐条核对昨日观察点（对照今日数据），并按系统要求输出 verification 块"
    obs_stats = store.observation_stats()
    if obs_stats.get("verified"):
        market["预测跟踪历史命中率"] = obs_stats

    user_prompt = (
        f"今天是 {today}。以下是今日真实数据（JSON）：\n"
        + json.dumps(market, ensure_ascii=False, default=str)
        + "\n\n请据此撰写今日复盘。")

    client = DeepSeekClient(model=resolve_model("review.daily"))
    message = client.chat(
        [{"role": "system", "content": REVIEW_SYSTEM_PROMPT},
         {"role": "user", "content": user_prompt}],
        # 叙述性但受事实约束（数字由代码渲染，模型只解读）
        temperature=TEMP_EXPLORATORY, max_tokens=2000,
        # ⚠️ 补上 purpose：此前这次调用在 llm_calls 里落到 unknown，配额耗尽时无法按功能定位。
        purpose="review.daily")
    markdown = (message.content or "").strip()
    if not markdown:
        return {"error": "LLM 返回空内容"}

    # ── 预测跟踪闭环：解析 verification 块 → 核对回填 + 今日观察落库 ──
    tracking = {}
    block = _parse_verification_block(markdown)
    if block:
        markdown = re.sub(r"```verification\s*\{.*?\}\s*```", "", markdown, flags=re.S).strip()
        verified_date = today
        for v in block.get("verifications", []) or []:
            try:
                store.update_observation_result(int(v["id"]), str(v.get("result", "无法验证")),
                                                verified_date)
            except (KeyError, TypeError, ValueError) as e:
                _logger.warning("[复盘] 核对回填失败: %s", e)
        new_obs = [{"content": o.get("content", ""), "kind": o.get("kind", "观察点")}
                   for o in block.get("new_observations", []) or [] if o.get("content")]
        if new_obs:
            store.insert_observations(today, new_obs)
        tracking = {"核对条数": len(block.get("verifications", []) or []),
                    "新增观察点": len(new_obs),
                    "历史命中率": store.observation_stats()}

    header = (f"# A股每日复盘 {today}\n\n"
              f"> 生成时间 {datetime.now():%H:%M:%S} ｜ 数据源：腾讯/新浪行情、同花顺行业资金、"
              f"联网搜索观点、历史相似日统计\n\n")

    # 决策三章由**程序**写入（数字不经过模型 → 不会被改写），插在头部之后、
    # LLM 正文之前；LLM 正文自身仍以"昨日观察核对/大盘概览"开头，顺序自然衔接。
    decision_md = ""
    if digest:
        try:
            decision_md = render_decision_sections(digest)
        except Exception as e:                  # noqa: BLE001
            _logger.warning("[复盘] 决策三章渲染失败: %s", e)
    # ── P0-5（2026-10-18）：今日数据完整性（**程序渲染，不经模型**）──
    # 模型只看到"喂给它的数据"，看不到"没喂进去的那部分" —— 缺什么只有代码知道。
    # 与"决策数字不经 LLM"同一条原则。放在头部之后、决策三章之前，读者先看到数据边界。
    integrity_md = ""
    try:
        from core import stage_health
        integrity_md = stage_health.data_integrity_markdown(today)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[复盘] 数据完整性段渲染失败（不影响报告主体）: %s", e)

    full_text = (header
                 + ((integrity_md + "\n") if integrity_md else "")
                 + ((decision_md + "\n\n") if decision_md else "")
                 + ((ashen_md + "\n\n") if ashen_md else "")
                 + markdown)

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"review_{datetime.now():%Y%m%d}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(full_text)
    with open(os.path.join(REPORT_DIR, "latest.md"), "w", encoding="utf-8") as f:
        f.write(full_text)
    _logger.info("[复盘] 报告已生成: %s (tracking=%s)", path, tracking)

    result = {"report_path": path, "markdown": full_text, "tracking": tracking}
    # 推送用摘要（2026-10-23）：全文 4000+ 字手机上要滚十几屏 ⇒ 推摘要 + 全文路径。
    # ⚠️ 摘要里的数字全部来自 digest（程序渲染），不经模型。
    try:
        summary = build_review_summary(digest=digest, integrity_md=integrity_md,
                                       llm_markdown=markdown, report_path=path, today=today)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[复盘] 摘要渲染失败（改推全文）: %s", e)
        summary = full_text
    result["summary"] = summary
    if push:
        # 同日只推一次（手动重放 + 定时任务曾把同一份复盘推两遍）
        result["push"] = push_report(summary, dedupe_key=f"review:{today}")
    return result


if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = generate_review(push=False)
    if "error" in out:
        print("失败:", out["error"])
    else:
        print(out["markdown"])
        print("\n报告路径:", out["report_path"])
