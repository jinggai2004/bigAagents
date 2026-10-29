"""
AI 对话助手的工具注册表：把现有 core/data 模块封装为 LLM function calling 工具。

每个工具返回 dict（带 as_of 时间戳）；失败返回 {"error": ...}，绝不抛异常到 Agent 循环。
注意：工具内刻意避开东财系接口（本环境被墙），实时行情走腾讯/新浪。
"""

import functools
import json
import logging
from datetime import datetime
from typing import Callable

import pandas as pd

_logger = logging.getLogger(__name__)


# ⚠️ P2-2 拆分（2026-10-17）：财务工具块已移至 `core/agent_tools_fin.py`；
#    这里 **re-export** 全部原名，保证既有 import / mock.patch 路径零改动。
from core.agent_tools_base import _tool, _now    # noqa: F401
from core.agent_tools_fin import (               # noqa: F401
    tool_financials, _FIN_FIELDS, _FIN_EQUITY_CANDIDATES,
    _FIN_NOT_IN_STATEMENTS, _FIN_CORE_LABELS, _num,
    _fin_period_key, _fin_rows, _fin_series,
    _quarter_of, _prev_quarter_key, _single_quarter,
)
# ⚠️ 2026-10-24 升级：系统级工具块（读报告/查库/看配置/模板/任务/面板/新鲜度）
#    移至 `core/agent_tools_sys.py`，同样在这里 re-export（拆分方式沿用 P2-2 先例，
#    sys 模块只依赖 agent_tools_base，无循环导入）。
from core.agent_tools_sys import (               # noqa: F401
    tool_read_output_file, tool_db_inspect, tool_config_view,
    tool_push_templates, tool_scheduled_jobs, tool_panel_bars,
    tool_data_freshness,
    # 2026-10-24 审查建议：结论类薄包装（页面同款函数，零新查询逻辑）
    tool_quality_stats, tool_llm_cost, tool_system_health,
    tool_decision_digest, tool_holdings_risk,
)
def _market_clock() -> dict:
    """今天是几号、是否交易日、最近交易日是几号。

    为什么要它：模型自身不知道"现在"，只能从工具返回里推；而 `as_of` 是**工具
    执行时间**（永远是今天），它会把"数据是上一个交易日的"掩盖成"刚刚查的"。
    2026-09-25（休市）实测：问"兴业银行当前是否值得投资"，模型答
    "**当前的实时行情**：17.88 元"——17.88 其实是 09-24 的收盘价。
    """
    today = datetime.now()
    info = {"今天": today.strftime("%Y-%m-%d"),
            "星期": "一二三四五六日"[today.weekday()],
            "是否交易日": None, "最近交易日": None}
    try:
        from data import trading_calendar as cal
        info["是否交易日"] = cal.is_trading_day(info["今天"])
        info["最近交易日"] = cal.last_trading_day()
    except Exception as e:                       # noqa: BLE001
        _logger.debug("[Agent工具] 交易日历不可用: %s", e)
    return info


def _kline_dates(df) -> list | None:
    """从 K 线 DataFrame 取日期序列，返回 ["YYYY-MM-DD", ...]；取不到返回 None。

    为什么要兼容两种形态：`get_stock_data` 返回的是**以 DatetimeIndex 为索引**、
    列名为 `Open/Close/...` 的表——**没有 `Date` 列**。而 `tool_stock_technicals`
    原先只在 `df.columns` 里找 'date/Date/日期'，找不到就退化成索引序号
    `["0","1",...,"59"]`：K 线图横轴变成数字，模型也**完全看不到数据截止日**。
    """
    col = next((c for c in ("date", "Date", "日期", "time", "trade_date")
                if c in getattr(df, "columns", [])), None)
    if col is not None:
        return [str(v)[:10] for v in df[col].tolist()]
    idx = getattr(df, "index", None)
    if idx is None:
        return None
    if isinstance(idx, pd.DatetimeIndex):
        return idx.strftime("%Y-%m-%d").tolist()
    try:
        conv = pd.to_datetime([str(v) for v in idx], errors="coerce")
        if len(conv) and not pd.isna(conv).any():
            return conv.strftime("%Y-%m-%d").tolist()
    except Exception:                            # noqa: BLE001
        pass
    return None


# ================================================================
# 工具实现
# ================================================================

@_tool
def tool_realtime_quotes(codes: list[str] = None) -> dict:
    """实时行情。codes 为6位代码列表；不传则返回自选股全部行情。
    `行情时间`/`数据日期` 是**数据自身**的时间（不是查询时间）：非交易日或盘前
    返回的是最近交易日收盘数据，`数据状态` 会如实标注，不得表述为"实时"。"""
    from data import realtime, watchlist
    if not codes:
        codes = watchlist.watchlist_codes()
    quotes = realtime.get_realtime_quotes(codes)
    if not quotes:
        return {"error": "实时行情不可用（腾讯/新浪源均失败）"}
    clock = _market_clock()
    today = clock["今天"]
    out = []
    for code, q in quotes.items():
        qd = q.get("quote_date") or ""
        if q.get("stale"):
            status = f"过期缓存（{q.get('age_seconds', 0):.0f}秒前拉取，非实时）"
        elif q.get("data_lagging"):
            status = f"数据滞后（{q.get('staleness_reason') or '落后于最近交易日'}）"
        elif qd and qd != today:
            status = f"最近交易日 {qd} 收盘数据（非当日实时）"
        else:
            status = "实时"
        out.append({
            "代码": q["code"], "名称": q["name"], "现价": q["price"],
            "涨跌幅%": q["change_pct"], "涨跌": q["change"],
            "换手率%": q["turnover_rate"],
            "成交额亿": round(q["amount_wan"] / 1e4, 2) if q["amount_wan"] else None,
            # 必须带日期：只给 "16:14:56" 时，模型无从判断这是哪一天的价格
            "行情时间": q.get("quote_datetime") or q["quote_time"],
            "数据日期": qd or "未知",
            "数据状态": status,
        })
    result = {"quotes": out,
              "今天": today,
              "今天是否交易日": clock["是否交易日"],
              "最近交易日": clock["最近交易日"]}
    if any(q.get("stale") for q in quotes.values()):
        result["警告"] = "部分或全部行情为过期缓存（数据源暂时不可用），不得当作实时价格引用"
    if any(q.get("data_lagging") for q in quotes.values()):
        warn = "部分行情日期落后于最近交易日，可能未取到最新数据"
        result["警告"] = f"{result['警告']}；{warn}" if result.get("警告") else warn
    return result


@_tool
def tool_stock_profile(symbol: str) -> dict:
    """个股基本信息与估值：名称、现价、PE、PB、总市值。
    东财接口不可达时用**腾讯实时行情**补齐估值（它会标注 `估值来源`/`估值数据日期`）；
    仍拿不到的会列进 `缺失字段`，不得据此推测数值。"""
    from data.stock_data import StockDataFetcher
    symbol = str(symbol).strip()
    if not (symbol.isdigit() and len(symbol) == 6):
        return {"error": "股票代码须为6位数字"}
    info = StockDataFetcher().get_stock_info(symbol) or {}
    keep = {k: info.get(k) for k in ("symbol", "name", "current_price", "change_percent",
                                     "pe_ratio", "pb_ratio", "market_cap", "market")}

    # 东财 push2 被墙时 get_stock_info 拿不到市值，而**腾讯实时行情本来就带**
    # 总市值/流通市值/PE/PB（实测 601166 总市值 3783.92 亿）。用它补齐，
    # 否则"市值"会长期显示 N/A，而市值是判断估值水平不可缺的一项。
    needed = ("current_price", "pe_ratio", "pb_ratio", "market_cap")
    if any(keep.get(k) in (None, "", "N/A") for k in needed):
        try:
            from data import realtime
            q = (realtime.get_realtime_quotes([symbol]) or {}).get(symbol) or {}
            filled = []
            if keep.get("current_price") in (None, "", "N/A") and q.get("price"):
                keep["current_price"] = q["price"]
                filled.append("现价")
            if keep.get("change_percent") in (None, "", "N/A") and q.get("change_pct") is not None:
                keep["change_percent"] = q["change_pct"]
                filled.append("涨跌幅")
            if keep.get("pe_ratio") in (None, "", "N/A") and q.get("pe"):
                keep["pe_ratio"] = q["pe"]
                filled.append("PE")
            if keep.get("pb_ratio") in (None, "", "N/A") and q.get("pb"):
                keep["pb_ratio"] = q["pb"]
                filled.append("PB")
            if keep.get("market_cap") in (None, "", "N/A") and q.get("total_mcap_yi"):
                keep["market_cap"] = f"{float(q['total_mcap_yi']):.2f}亿"
                filled.append("总市值")
            if filled:
                keep["估值来源"] = "腾讯实时行情（东财接口不可达时的兜底）"
                keep["估值数据日期"] = q.get("quote_date") or ""
        except Exception as e:                    # noqa: BLE001 - 兜底失败不影响基本信息
            _logger.debug("[Agent工具] %s 估值兜底失败: %s", symbol, e)

    missing = [k for k in needed if keep.get(k) in (None, "", "N/A")]
    if missing:
        keep["缺失字段"] = missing
    keep["说明"] = ("估值为最近可得数据，非实时；请结合 `估值数据日期`/tool_realtime_quotes "
                    "的 `数据日期` 判断新鲜度")
    return keep


@_tool
def tool_stock_technicals(symbol: str) -> dict:
    """个股技术面：基于**日线**的均线系统(MA5/10/20/60)、均线形态、近20日涨跌幅、
    20日年化波动率；`数据截止` 为最后一根 K 线的日期（日线数据通常到最近交易日）"""
    from data.stock_data import StockDataFetcher
    symbol = str(symbol).strip()
    if not (symbol.isdigit() and len(symbol) == 6):
        return {"error": "股票代码须为6位数字"}
    df = StockDataFetcher().get_stock_data(symbol, period="6mo")
    if df is None or not hasattr(df, "columns") or df.empty:
        return {"error": "K线数据不可用"}
    close_col = next((c for c in ("close", "Close", "收盘") if c in df.columns), None)
    if close_col is None:
        return {"error": f"K线数据缺收盘价列: {list(df.columns)[:8]}"}
    closes = df[close_col].astype(float)
    ma = {f"MA{n}": round(closes.tail(n).mean(), 2) for n in (5, 10, 20, 60) if len(closes) >= n}
    last = float(closes.iloc[-1])
    ret_20d = round((last / float(closes.iloc[-21]) - 1) * 100, 2) if len(closes) >= 21 else None
    daily_ret = closes.pct_change().dropna().tail(20)
    vol = round(float(daily_ret.std()) * (252 ** 0.5) * 100, 1) if len(daily_ret) >= 5 else None
    ma_bull = "多头排列" if len(ma) >= 3 and ma["MA5"] > ma["MA10"] > ma["MA20"] else (
        "空头排列" if len(ma) >= 3 and ma["MA5"] < ma["MA10"] < ma["MA20"] else "纠缠")
    # K线采样（对话内图表用，最多60点）。
    # 注意：不能简单用 closes.iloc[::step].tail(60) —— 当 len(closes) 为偶数时
    # 末位下标会正好被跨过，图表最后一根就比 `数据截止` 早一天。显式补上末位。
    step = max(1, len(closes) // 60)
    all_dates = _kline_dates(df)
    picks = list(range(0, len(closes), step))
    if picks and picks[-1] != len(closes) - 1:
        picks.append(len(closes) - 1)
    picks = picks[-60:]
    sampled = closes.iloc[picks]
    dates = [all_dates[i] for i in picks] if all_dates else []
    data_through = all_dates[-1] if all_dates else None
    return {"代码": symbol, "现价": round(last, 2), "均线": ma, "均线形态": ma_bull,
            "近20日涨跌幅%": ret_20d, "20日年化波动率%": vol,
            "数据截止": data_through,
            "K线根数": len(closes),
            "kline_sample": {"dates": dates, "closes": [round(float(c), 2) for c in sampled]}}


@_tool
def tool_fund_flow(symbol: str) -> dict:
    """个股资金面：五维资金评分(满分25)与各窗口净额。数据可能来自同花顺快照降级源"""
    from data.fund_flow import FundFlowAkshareDataFetcher, compute_five_dim_score
    symbol = str(symbol).strip()
    if not (symbol.isdigit() and len(symbol) == 6):
        return {"error": "股票代码须为6位数字"}
    fetcher = FundFlowAkshareDataFetcher()
    result = fetcher.get_fund_flow_data(symbol)
    if not result.get("data_success"):
        return {"error": f"资金流数据不可用: {result.get('error', '未知')}"}
    df = pd.DataFrame(result["fund_flow_data"].get("data", []))
    score = compute_five_dim_score(df)
    flow = result["fund_flow_data"]
    out = {"五维评分": f"{score.get('score')}/{score.get('max')}",
           "维度明细": score.get("breakdown", {}), "数据模式": flow.get("mode", "daily")}
    # ⚠️ 分母是**可评维度**的满分和（快照模式 20，因为「结构」无单型拆分不可评）。
    #    原先固定写成 /25，把"取不到"当成"得 0 分"，每只票被系统性扣 5 分，
    #    且与逐日模式不可比（2026-10-10 由用户报"五维评分为空"追出）。
    if score.get("unavailable_dims"):
        out["不可评维度"] = {
            "维度": score["unavailable_dims"],
            "说明": (f"这些维度数据源取不到，**已从总分分母剔除**（总分满分为 "
                     f"{score.get('max')} 而非 25）—— 不是「该维度得 0 分」。"),
        }
    if flow.get("mode") == "snapshot":
        rec = flow["data"][-1] if flow.get("data") else {}
        out["窗口净额(元)"] = {k: rec.get(k) for k in ("净额_今日", "净额_3日", "净额_5日", "净额_10日")
                              if isinstance(rec.get(k), (int, float))}
        out["降级说明"] = score.get("note", "")
        out["数据说明"] = ("同花顺即时快照（非交易日/盘前为最近交易日数据）；"
                          "结构维度无单型拆分，评分口径见 降级说明")
    return out


# 市场检测 14 指标 → 四维度（与 MarketDetector.weights 的分组一致）
_MARKET_DIMENSIONS = {
    "趋势": ("hs300_ma20_deviation", "sh_macd", "cyb_relative_strength"),
    "量能": ("volume_vs_ma20", "advance_decline_ratio", "north_flow"),
    "情绪": ("limit_up_down_ratio", "max_consecutive_board", "market_breadth", "ivix_level"),
    "结构": ("sector_rotation_speed", "size_style_bias", "margin_balance_change",
             "limit_up_upgrade_rate"),
}
# 指标 key → 中文名（给模型看，避免它把英文 key 读错）
_MARKET_INDICATOR_NAMES = {
    "hs300_ma20_deviation": "沪深300偏离20日线", "sh_macd": "沪指MACD",
    "cyb_relative_strength": "创业板相对强度", "volume_vs_ma20": "成交量/20日均量",
    "advance_decline_ratio": "涨跌家数比", "north_flow": "北向资金",
    "limit_up_down_ratio": "涨跌停比", "max_consecutive_board": "最高连板",
    "market_breadth": "市场宽度", "ivix_level": "iVIX波动率",
    "sector_rotation_speed": "板块轮动速度", "size_style_bias": "大小盘风格",
    "margin_balance_change": "融资余额变化", "limit_up_upgrade_rate": "涨停晋级率",
}


@_tool
def tool_market_signal() -> dict:
    """当前大盘环境：14指标市场检测（趋势/量能/情绪/结构四维评分与仓位建议）。耗时约30秒。
    `数据缺失指标` 列出取不到数据的指标（这些既不计分也不计入分母），
    缺失越多结论越保守，引用时必须说明。"""
    from core.market_detector import MarketDetector
    signal = MarketDetector().detect()
    # ⚠️ 字段名是 `scores`（逐指标明细），不是 `dimension_scores` ——
    # 本工具原先写的 dimension_scores 并不存在，AttributeError 被 @_tool 兜住后
    # 每次调用都返回 {"error": ...}，等于"大盘环境"这一维度从来没真正给过模型。
    scores = signal.scores or {}
    dims, missing = {}, []
    for dim, keys in _MARKET_DIMENSIONS.items():
        got = {k: scores[k] for k in keys if isinstance(scores.get(k), dict)}
        score = round(sum(v.get("weighted_score") or 0.0 for v in got.values()), 1)
        weight = round(sum(v.get("weight") or 0.0 for v in got.values()), 2)
        dims[dim] = f"{score}/{round(weight * 100)}分"
        missing.extend(_MARKET_INDICATOR_NAMES.get(k, k) for k, v in got.items()
                       if v.get("unavailable"))
    out = {
        "市场状态": signal.market_state,
        "总分": signal.total_score,
        "信号": signal.signal,
        "仓位上限": f"{signal.position_limit * 100:.0f}%",
        "策略偏向": signal.strategy_bias,
        "操作建议": signal.operation_advice,
        "维度得分": dims,
        "数据缺失指标": missing,
    }
    if signal.risk_warnings:
        out["风险提示"] = list(signal.risk_warnings)[:6]
    if missing:
        out["说明"] = (f"共 {len(missing)} 项指标无数据，已从分母中剔除（不按 0 分计），"
                      f"总分已按有效权重归一化，结论偏保守")
    return out


@_tool
def tool_market_sentiment(date: str = None) -> dict:
    """市场情绪参照（**只描述情绪位置，不作买卖依据**）。零联网、读当日已落库快照。

    ⚠️ 这个工具**故意不在返回值里给任何"看多/看空"结论**，只给数字和质量标记。
       结论由调用它的模型自己写，且必须连带 `引用规则` 一起讲给用户 ——
       否则"散户 71% 看多"会被当成系统的判断，而它只是言论统计。

    ⚠️ 没有数据时返回 `{"状态": "未采集"}` 而不是空 dict 或 `{}`：
       前者能被模型正确读懂（"今天还没采"），后者会被当成"没有异常"。
    """
    from core import sentiment_reference as SR
    ref = SR.load_reference_block(date)
    if ref.get("状态") != "可用":
        return {"状态": ref.get("状态"),
                "说明": ref.get("说明") or ref.get("原因") or "无可用数据",
                "采集日": ref.get("采集日"),
                "提示": "情绪参照缺失时**不要**据此推断情绪中性，应说明『情绪数据不可用』"}
    out = {
        "状态": "可用",
        "采集日": ref.get("采集日"),
        "来源数": ref.get("来源数"),
        "各来源": ref.get("各来源"),
        "说明": ref.get("说明"),
        "引用规则": ref.get("引用规则"),
    }
    # 提醒模型"这不是评分输入"，跟着数据走比写在系统提示里更难被忽略
    if any("不可作依据" in str(x.get("数据质量")) or
           "偏薄" in str(x.get("数据质量")) for x in ref.get("各来源") or []):
        out["质量提示"] = "存在样本偏薄的来源 ⇒ 该来源的比例与分位不可作依据，只能说明『样本不足』"
    return out


@_tool
def tool_screen_stocks(sector: str = None, max_stocks: int = 5, strategies: list = None) -> dict:
    """执行选股流程（**本地五路初筛**→六因子精筛；快照数据源不可用时才降级内置精选池）。
    sector 可选：科技/金融/消费/新能源/医药/创业板；strategies 可选初筛策略子集。
    耗时约2-4分钟，仅在用户明确要求选股时调用"""
    from core.screener import StockScreener
    strategy_list = [s for s in (strategies or []) if isinstance(s, str)] or None
    result = StockScreener().run(sector_filter=sector, max_stocks=int(max_stocks),
                                 strategies=strategy_list)
    if result.get("skipped"):
        return {"skipped": True, "原因": result.get("reason", "市场环境不佳")}
    picks = []
    for p in result.get("candidates", []):
        picks.append({
            "代码": p.get("股票代码"), "名称": p.get("股票名称"),
            "综合评分": p.get("score"), "板块": p.get("板块", ""),
            "评分状态": p.get("score_status", ""),
            "有效因子权重": p.get("score_coverage"),
            "降级": bool(p.get("degraded")),
            # v2：unknown 因子不再渲染成 "None/25"
            "因子": {k: ("数据缺失" if (v.get("status") == "unknown" or v.get("score") is None)
                        else f"{v.get('score')}/{v.get('max')}")
                    for k, v in (p.get("factors") or {}).items()},
        })
    # ⚠️ 把"权重未经检验"**跟着数据一起返回**：模型只有在读到评分的同时读到
    # 这条说明，才不会把"综合评分 82"当成已验证的结论讲给用户。
    # 只写在系统提示词里不够——长对话里模型容易忽略规则，但不会忽略工具返回值。
    from core.sector import SCORE_WEIGHT_CAVEAT
    out = {"候选数": len(picks), "候选": picks, "评分说明": SCORE_WEIGHT_CAVEAT}
    if result.get("degraded"):
        out["降级提示"] = "；".join(result.get("degrade_reasons", []))
    return out


@_tool
def tool_backtest_summary(days: int = 60) -> dict:
    """运行规则信号回测（零LLM成本）并返回绩效摘要（含对比沪深300的信息比率）。耗时约1-2分钟，
    仅在用户明确要求回测时调用"""
    from core.pipeline import DailyPipeline
    pipeline = DailyPipeline()
    bt = pipeline._stage_backtest(int(days))
    if not bt:
        return {"error": "回测未产出结果"}
    if "note" in bt:
        return bt
    return {"窗口": bt.get("window"), "总收益%": bt.get("total_return"),
            "年化%": bt.get("annual_return"), "夏普": bt.get("sharpe_ratio"),
            "最大回撤%": bt.get("max_drawdown"), "信息比率": bt.get("information_ratio"),
            "胜率%": bt.get("win_rate"), "交易次数": bt.get("total_trades"),
            "报告文件": bt.get("report_file")}


@_tool
def tool_manage_watchlist(action: str, code: str = "", name: str = "",
                          note: str = "", clear_note: bool = False,
                          change_id: str = "") -> dict:
    """管理自选股（写操作走两阶段守护：preview → 用户确认 → commit）。

    action=list    只读：列出全部自选股
    action=add     写·预览：添加（返回 change_id + 前后完整列表差异）
    action=remove  写·预览：移除
    action=update  写·预览：改名称/备注（name/note 只改传了非空值的；clear_note=true 清空备注）
    action=commit  写·执行：change_id=用户确认过的 preview 凭证（15分钟有效、一次性）
    action=restore 写·回滚：change_id=历史一次 commit 的凭证，按快照撤销

    ⚠️ 纪律：add/remove/update 只产出 preview；必须向用户复述差异、
    用户明确同意后用 commit 执行。改动在 commit 后立即生效（JSON 文件，无需重启）。
    """
    from data import watchlist as wl
    from core import agent_write_guard as guard
    action = str(action).strip().lower()

    if action == "commit":
        return guard.commit(change_id)
    if action == "restore":
        if not change_id:
            return {"error": "restore 需要 change_id（历史 commit 的凭证）"}
        return guard.restore(change_id)
    if action == "list":
        return {"自选股": wl.load_watchlist()}

    # ── 写操作：只产出 preview ──
    stocks = wl._load_for_write()
    if stocks is None:
        return {"error": "config/watchlist.json 损坏，已中止写操作——请人工修复后再试"}

    if action == "add":
        code_s = str(code or "").strip()
        if not (code_s.isdigit() and len(code_s) == 6):
            return {"error": "add 需要合法的6位 code"}
        if any(s["code"] == code_s for s in stocks):
            return {"error": f"{code_s} 已在自选中"}
        name_s = str(name or "").strip()
        note_s = str(note or "").strip()
        new_stock = {"code": code_s, "name": name_s, "note": note_s}
        new_list = stocks + [new_stock]
        return guard.begin(
            "tool_manage_watchlist", "add", f"自选股添加 {code_s}（{name_s or '未命名'}）",
            old={"自选股数": len(stocks)}, new={"新增": new_stock, "自选股数": len(new_list)},
            op={"kind": "watchlist_save", "stocks": new_list},
            revert_op={"kind": "watchlist_save", "stocks": stocks})

    if action == "remove":
        code_s = str(code or "").strip()
        target = next((s for s in stocks if s["code"] == code_s), None)
        if target is None:
            return {"error": f"{code_s} 不在自选中"}
        new_list = [s for s in stocks if s["code"] != code_s]
        return guard.begin(
            "tool_manage_watchlist", "remove", f"自选股移除 {code_s}（{target.get('name') or ''}）",
            old={"将移除": target, "自选股数": len(stocks)},
            new={"自选股数": len(new_list)},
            op={"kind": "watchlist_save", "stocks": new_list},
            revert_op={"kind": "watchlist_save", "stocks": stocks})

    if action == "update":
        code_s = str(code or "").strip()
        if not code_s:
            return {"error": "update 需要传 code"}
        target = next((s for s in stocks if s["code"] == code_s), None)
        if target is None:
            return {"error": f"{code_s} 不在自选中"}
        name_val = str(name).strip() if str(name or "").strip() else None
        note_val = ""
        if clear_note:
            note_val = ""
        elif str(note or "").strip():
            note_val = str(note).strip()
        changed = []
        if name_val and name_val != target.get("name"):
            changed.append("名称")
        if (note_val or clear_note) and note_val != (target.get("note") or ""):
            changed.append("备注")
        if not changed:
            return {"error": "没有要修改的字段（传入的值与当前一致）"}
        new_list = [dict(s) for s in stocks]
        t = next(s for s in new_list if s["code"] == code_s)
        if name_val:
            t["name"] = name_val
        if note_val or clear_note:
            t["note"] = note_val
        return guard.begin(
            "tool_manage_watchlist", "update", f"自选股 {code_s} 修改{'、'.join(changed)}",
            old={code_s: dict(target)}, new={code_s: dict(t)},
            op={"kind": "watchlist_save", "stocks": new_list},
            revert_op={"kind": "watchlist_save", "stocks": stocks})

    return {"error": f"未知 action: {action}，可用 list/add/remove/update/commit"}


@_tool
def tool_similar_days(symbol: str = "sh000001", top_k: int = 5) -> dict:
    """检索与当前最相似的历史交易日及其后续5/20日表现分布（统计参考非预测）。
    symbol 支持 sh000001(上证)/sz399001(深成)/sz399006(创业板)及自选股代码；首次使用某标的需先回填(约1-2分钟)"""
    from data import similar_days
    symbol = str(symbol).strip().lower()
    result = similar_days.find_similar_days(symbol, top_k=int(top_k))
    if "error" in result:
        return result
    result["使用说明"] = "相似度为技术特征统计匹配；后续表现为历史分布统计，仅供参考，不构成预测"
    return result


@_tool
def tool_backfill_similar_history(symbol: str = "sh000001", years: int = 10) -> dict:
    """回填某标的历史特征库（相似日检索前置步骤，约1-2分钟）。仅在检索报'特征库不足'时调用"""
    from data import similar_days
    return similar_days.backfill_history(str(symbol).strip().lower(), years=int(years))


@_tool
def tool_portfolio_positions() -> dict:
    """查询**真实持仓**（用户在「持仓」页手工录入的那份）：各持仓现价/浮盈亏/仓位占比
    与总资产/现金/风险敞口。用户问"我的持仓/我的仓位"时使用。

    ⚠️ 这是**真实持仓账本**，与「模拟账户」页的模拟盘是**两套完全独立的数据**
    （`positions` 表 vs `paper_positions` 表）。用户问"模拟账户/模拟盘"时不要用本工具
    下结论——空结果只能说明"真实持仓没录入"，不能推出"模拟账户也没持仓"。"""
    from core.portfolio import get_portfolio_manager
    from data import realtime
    pm = get_portfolio_manager()
    # ⚠️ 先批量取行情，再拿持仓（2026-10-15）：`get_active_positions()` / `load()` 会逐只联网取价，
    #    而 `_get_current_price()` 走 akshare 东财接口（取数受限时单只 ≈30s 重试）
    #    ⇒ "问一句持仓"要等几十秒。用 `active_codes()`（纯 DB）先拿代码，再一次批量取价，
    #    然后**透传**给下面所有调用 —— 这也是持仓页同款问题的同一套修法。
    codes = pm.active_codes()
    if not codes:
        return {"账本": "真实持仓（「持仓」页手工录入）", "positions": [],
                "note": ("真实持仓**未录入**——这不代表用户没有持仓。"
                         "若用户问的是「模拟账户/模拟盘」，请改用 tool_paper_account；"
                         "若确实在问真实持仓，可提示他到「持仓」页录入")}
    try:
        quotes = realtime.get_realtime_quotes(list(codes)) or {}
    except Exception:                                        # noqa: BLE001
        quotes = {}                                          # 取不到 ⇒ 回落自行取价（不造数）
    positions = pm.get_active_positions(prices=quotes)
    rows = []
    for pos in positions:
        q = quotes.get(pos.code)
        price = q["price"] if q else pos.entry_price
        pnl = round((price - pos.entry_price) * pos.quantity, 2)
        pct = round((price / pos.entry_price - 1) * 100, 2) if pos.entry_price else None
        rows.append({
            "代码": pos.code, "名称": pos.name or (q["name"] if q else ""),
            "建仓日期": pos.entry_date, "成本价": pos.entry_price, "数量": pos.quantity,
            "现价": price, "浮动盈亏": pnl, "盈亏%": pct,
            "止损": pos.stop_loss, "止盈": pos.take_profit,
        })
    portfolio = pm.load(prices=quotes)
    return {"账本": "真实持仓（「持仓」页手工录入）",
            "positions": rows,
            "总市值": round(portfolio.total_value, 2),
            "现金": round(portfolio.cash, 2)}


@_tool
def tool_paper_account() -> dict:
    """查询**模拟账户（纸上交易/模拟盘）**：每只持仓的数量/可卖/成本价/现价/浮盈亏、
    可用现金、持仓市值、总资产与累计收益率。用户问"模拟账户/模拟盘/虚拟盘怎么样、
    赚了还是亏了、要不要调仓"时使用。

    与 `tool_portfolio_positions`（真实持仓，手工录入）是**两套独立账本**，
    数据分别落在 `paper_positions` 与 `positions` 表，不要互相推断。
    """
    from db import store
    from core.paper_account import PaperAccount
    from data import realtime

    state = store.load_paper_state("default")
    if not state:
        return {"账本": "模拟账户（纸上交易）", "account": "default", "持仓": [],
                "note": "模拟账户尚未初始化或没有任何成交。可在「模拟账户」页下单开始"}

    acct = PaperAccount.from_state(state)
    # 只读查询也要顺带解冻 T+1：否则助手会照着**过期的 available** 回答
    # （周五买入的仓位在周末仍是 available=0，助手据此说"不可卖"是错的）。
    # 只在**确有解冻**时回写库，避免每次提问都写一遍。
    if acct.unlock_t1():
        store.save_paper_state(acct.name, acct.to_state())
    codes = list(acct.positions)
    quotes = realtime.get_realtime_quotes(codes) if codes else {}
    price_map = {}
    for c in codes:
        p = (quotes.get(c) or {}).get("price")
        if isinstance(p, (int, float)) and p > 0:
            price_map[c] = float(p)

    rows = []
    for code, pos in acct.positions.items():
        price = price_map.get(code)
        valued = price if price else pos.avg_cost     # 缺价退回收盘均价，不用 0 抹掉持仓
        pnl = round((valued - pos.avg_cost) * pos.quantity, 2)
        pct = round((valued / pos.avg_cost - 1) * 100, 2) if pos.avg_cost else None
        rows.append({
            "代码": code, "名称": (quotes.get(code) or {}).get("name") or "",
            "数量": pos.quantity, "可卖": pos.available,
            "成本价": round(pos.avg_cost, 4),
            "现价": round(price, 3) if price else None,
            "市值": round(valued * pos.quantity, 2),
            "浮动盈亏": pnl, "盈亏%": pct,
            "估值口径": "按最新现价" if price else "取不到现价，按成本价估值",
        })

    holdings = acct.holdings_value(price_map)
    total = acct.cash + holdings
    ret = round((total / acct.initial_cash - 1) * 100, 2) if acct.initial_cash else None
    pending = len([o for o in acct.orders if o.status == "pending"])
    quote_times = [q.get("quote_time") for q in quotes.values() if q.get("quote_time")]

    # 行情日期必须去重（多只票同一交易日 → 否则会列出 3 个一模一样的日期）
    quote_dates = sorted({(quotes.get(c) or {}).get("quote_date")
                          for c in codes if (quotes.get(c) or {}).get("quote_date")})
    stale_codes = [c for c in codes if (quotes.get(c) or {}).get("stale")]
    lag_codes = [c for c in codes if (quotes.get(c) or {}).get("data_lagging")]

    out = {
        "账本": "模拟账户（纸上交易）",
        "account": acct.name,
        "初始资金": round(acct.initial_cash, 2),
        "可用现金": round(acct.cash, 2),
        "持仓市值": round(holdings, 2),
        "总资产": round(total, 2),
        "累计收益率%": ret,
        "持仓只数": len(rows),
        "持仓": rows,
        "查询时间": datetime.now().strftime("%Y-%m-%d %H:%M"),
        # ⚠️ 这是**数据自身**的日期，不是查询时间。原先把查询日期写成"数据日期"，
        #    周末问"模拟盘怎么样"时会显示今天（休市日），与行情实为上周五相矛盾。
        "行情数据日期": "、".join(quote_dates) if quote_dates else "未取到",
        "行情时间": max(quote_times) if quote_times else None,
    }
    if pending:
        out["待成交委托"] = pending
    if not rows:
        out["note"] = "模拟账户当前**无股票持仓**（只有现金）"
    elif not price_map:
        out["note"] = ("未取到任何现价，市值/浮盈亏按**成本价**估值，仅供参照；"
                       "成交与净值以账户记录为准")
    if stale_codes:
        out["警告"] = (f"{len(stale_codes)} 只持仓行情为**过期缓存**（数据源暂时不可用），"
                       f"该部分市值不可当作最新价：{'、'.join(stale_codes)}")
    elif lag_codes:
        out["警告"] = (f"{len(lag_codes)} 只持仓行情日期落后于最近交易日，"
                       f"可能未取到最新数据：{'、'.join(lag_codes)}")
    return out


def tool_ashen_status(day: str = "") -> dict:
    """查询**阿深**（系统内置的 3D 桌宠 AI 交易员，persona=`ai_ashen`）的状态：
    今日 6 个排班轮次各自做了什么判断、它自己的模拟账户净值/持仓、它的自选池、
    以及它的建仓权级别与机械边界。`day` 缺省为今天。

    ## 为什么这是一个独立工具（2026-10-28 新增）
    阿深有自己的**第三套账本**（`ai_ashen`，100 万模拟资金）和独立的自选池，
    此前助手完全没有对应的工具，后果是实测出来的两个错误：
      · 问"阿深今天跑了几轮值班" → 它去查 `runs` 表，回答成"流水线跑了 3 轮"
        （把"值班"误解成"选股流程"）；
      · 问"阿深模拟账户净值" → 它调 `tool_paper_account`，读的是用户自己的
        `default` 盘。**数字碰巧一样（都是 100 万空仓），所以看起来没错** ——
        错误的数据源给出正确的数字，比报错更危险。

    ## 口径
    所有数字来自 `web.server.desk.ashen()`（与「交易台」页同一份实现，口径不会漂）：
    · 净值缺收盘价时是 `None` + `估值完整=False`，**不拿"只算到价那几只"的合计数冒充**；
    · 「已跑」= 排班格子内**真正调用了模型**的轮数，跳过轮不计入。
    """
    from web.server import desk as desk_api

    day_s = str(day or "").strip()
    env = desk_api.ashen(day=day_s or None)
    if not env.get("ok"):
        err = env.get("error") or {}
        return {"error": f"阿深状态读取失败: {err.get('message') or err.get('type') or err}"}

    d = env.get("data") or {}
    today = d.get("today") or {}
    counts = today.get("counts") or {}
    acct = d.get("account") or {}
    pools = d.get("pools") or {}
    au = d.get("autonomy") or {}
    persona = d.get("persona") or {}
    budget = (d.get("budget") or {}).get("routine") or {}

    runs = []
    # ⚠️ `actions` 里**永远**至少有一条兜底的 `{"action": "pass"}`（保证下游永远有动作），
    #    直接数 len(actions) 会让每一轮都显示"动作数 1"，模型于是说"它今天做了 6 个动作"，
    #    而真实情况是"6 轮都明确选择不动"。必须把 pass 剔掉再数。
    for r in sorted((today.get("runs") or []),
                    key=lambda x: str(x.get("created_at") or "")):
        raw = r.get("actions") or []
        real = [a for a in raw if str(a.get("action") or "").strip().lower() != "pass"]
        runs.append({
            "班次": r.get("slot"),
            "状态": r.get("status"),
            "调模型": bool(r.get("llm_called")),
            "结论": str(r.get("summary") or r.get("skip_reason") or "")[:180],
            "动作数": len(real),
            "动作": [{"动作": a.get("action"), "代码": a.get("code"),
                      "理由": str(a.get("reason") or "")[:90]} for a in real],
            "被拦": len(r.get("blocked") or []),
            "时间": r.get("created_at"),
        })

    pool_rows = []
    for r in ((pools.get("ashen") or {}).get("rows") or []):
        pool_rows.append({"代码": r.get("code"), "名称": r.get("name"),
                          "入池理由": str(r.get("note") or "")[:80]})

    pos_rows = []
    for p in (acct.get("positions") or []):
        pos_rows.append({"代码": p.get("code"), "股数": p.get("quantity"),
                         "成本": p.get("avg_cost"), "现价": p.get("price"),
                         "市值": p.get("market_value"), "浮盈亏%": p.get("pnl_pct")})

    out = {
        "交易员": "阿深（ai_ashen）· 系统内置的 3D 桌宠交易员，有独立模拟账户与建仓权",
        "日期": d.get("trade_date") or day_s,
        "启用": persona.get("enabled"),
        "授权级别": {
            "级别": au.get("level"),
            "说明": au.get("desc"),
            "机械边界": au.get("describe"),
            "配置合法": au.get("consistent"),
            "违规项": au.get("violations") or [],
        },
        "值班": {
            "排班": today.get("slots"),
            "已跑": counts.get("scheduled_llm"),
            "跳过": counts.get("skipped"),
            "手动补跑": counts.get("manual"),
            "当前时点该跑": today.get("due_slot"),
            "下一次": today.get("next_slot"),
            "轮次": runs,
            "口径": ("「已跑」= 排班格子内真正调用模型的轮数（跳过轮不计）；"
                     "「下一次 None」表示今日排班已结束；"
                     "「动作数 0」= 该班**明确选择不动**（模型给的是 pass），不是漏跑也不是失败"),
        },
        "账户": {
            "存在": acct.get("exists"),
            "现金": acct.get("cash"),
            "持仓只数": acct.get("position_count"),
            "持仓市值": acct.get("market_value"),
            "总资产": acct.get("total_value"),
            "估值完整": acct.get("valuation_complete"),
            "缺收盘价只数": len(acct.get("missing_codes") or []),
            "累计收益率%": acct.get("return_pct"),
            "持仓": pos_rows,
        },
        "自选池": {
            "只数": (pools.get("ashen") or {}).get("count"),
            "读取失败": not (pools.get("ashen") or {}).get("available"),
            "明细": pool_rows,
        },
        "定时额度": {"剩余": budget.get("remaining"), "配额": budget.get("quota")},
        "一句话": d.get("pool_hint"),
        # ⚠️ 这一行是给模型看的：三套账本混用会让它把用户的钱当成阿深的钱。
        "账本说明": ("这是阿深的**独立模拟账户**（ai_ashen）。它既不等于用户的模拟盘"
                     "（tool_paper_account，default），也不等于真实持仓"
                     "（tool_portfolio_positions）。被问阿深的仓位时必须用本工具。"),
    }
    if acct.get("exists") and not acct.get("valuation_complete"):
        out["账户"]["note"] = (f"{len(acct.get('missing_codes') or [])} 只持仓缺收盘价，"
                               "总资产置空而不是只算到价的那几只")
    if not acct.get("exists"):
        out["账户"]["note"] = "阿深还没有模拟账户（首次建仓时自动创建）"
    return out


def tool_market_opinions(date: str = "") -> dict:
    """聚合"如何看待X月X日A股行情"类市场讨论观点（联网检索知乎/雪球/财联社等收盘点评，含倾向标注）。
    date 格式如 '08月28日'，默认最近交易日。注意：为AI对搜索结果的摘要，来源仅供参考"""
    from data.market_opinions import get_market_opinions
    result = get_market_opinions(str(date).strip() or None)
    if "error" in result:
        return result
    return {"date": result.get("date"), "路径": result.get("path"),
            "opinions": result.get("opinions", []),
            "说明": "AI对搜索结果的归纳摘要，观点来源仅供参考，不构成投资建议"}


@_tool
def tool_latest_news(keyword: str = "", code: str = "", limit: int = 15) -> dict:
    """最近财经新闻/公告。用户问"今天/最新/近期"等时间性新闻时不要传keyword直接查询；
    keyword仅用于主题/公司名过滤。
    传 code 时**优先按个股直查**（东财个股新闻，实测 0.13~0.38s / 10 条），再补本地快讯库；
    若两者都没命中会返回 note 说明检索范围（**不等于该股无新闻**），此时不得直接断言
    "没有相关新闻"。"""
    from data.news_feed import smart_latest_news
    code = str(code).strip()
    name = ""
    if code:
        # 传简称可按"正文提及"兜底命中——库内 related_codes 只覆盖关注池，
        # 单靠代码过滤对不在池内的个股恒为 0 条（2026-09-25 实测：601166 得 0 条）。
        try:
            from data import realtime
            name = (realtime.get_realtime_quotes([code]).get(code) or {}).get("name") or ""
        except Exception as e:                    # noqa: BLE001 - 取名失败不影响新闻查询
            _logger.debug("[Agent工具] 解析 %s 简称失败: %s", code, e)

    n = min(int(limit), 30)
    out = []
    notes = []
    seen = set()

    def _push(item):
        title = (item.get("标题") or "")[:40]
        if title in seen:
            return
        seen.add(title)
        out.append(item)

    # 按个股直查优先：冷门个股只有这条路能命中（快讯库只播热门）
    if code:
        try:
            from data.news_feed import fetch_stock_news, classify_sentiment
            hit = fetch_stock_news(code, days=7, name=name) or {}
            for i in (hit.get("items") or [])[:n]:
                _push({"时间": i.get("published_at"), "来源": i.get("source"),
                       "标题": i.get("title") or (i.get("content") or "")[:60],
                       "内容": (i.get("content") or "")[:200],
                       "情绪": classify_sentiment(
                           f"{i.get('title', '')} {i.get('content', '')}"),
                       "链接": i.get("url") or "", "关联股票": [code]})
            if hit.get("note"):
                notes.append(hit["note"])
            if hit.get("error"):
                notes.append(f"个股新闻源: {hit['error']}")
        except Exception as e:                    # noqa: BLE001 - 直查失败只降级
            notes.append(f"个股新闻源不可用: {e}")

    result = smart_latest_news(limit=n, keyword=str(keyword).strip(),
                               code=code, name=name or None, hours=72)
    for i in result["news"]:
        if len(out) >= n:
            break
        _push({"时间": i["published_at"], "来源": i["source"],
               "标题": i["title"] or i["content"][:60],
               "内容": i["content"][:200], "情绪": i["sentiment"],
               "关联股票": i["related_codes"]})

    resp = {"news": out, "total": len(out)}
    if name:
        resp["查询标的"] = f"{name}({code})"
    if result.get("note"):
        notes.append(result["note"])
    if result.get("auto_refreshed"):
        notes.append("查询时已自动补采最新快讯")
    seen_notes = []
    for t in notes:
        if t and t not in seen_notes:
            seen_notes.append(t)
    if seen_notes:
        resp["note"] = "；".join(seen_notes)
    return resp


# ================================================================
# 深度分析报告读取（v1，2026-10-09）
# ================================================================
# 为什么必须有这个工具：实测（`chat_logs` id=47/48）用户问
# 「看最新的关于603127的深度分析报告，为什么舆情乐观仍短期看空」，
# 助手**只调了 `tool_latest_news`**，然后写"短期看空**可能**基于…**例如**，**若**股价已处于高位" ——
# 全是猜测口吻。原因不是态度问题：当时 15 个工具里**没有任何一个能读到报告正文**，
# 所以它只能拿行情/新闻**重新推一遍**，而这与"报告里是怎么写的"是两件事。
#
# 数据其实一直都在：`analyses` 表存了完整 payload（技术面/基本面/舆情/多空辩论/
# 研究汇总/交易计划/风控/最终决策），`db.store.list_analyses / get_analysis` 是现成的。
#
# ⚠️ 关键设计：**必须把该条报告自己的 `errors` / `audit_warnings` 一起返回**。
#    报告本身可能含幻觉（本项目 2026-10-08 就出现过整段编造的监管框架）。
#    如果只把正文交给模型，它会把这些内容当**权威结论**转述给用户 ——
#    那比"读不到报告"更糟。所以工具把"这份报告自己的可疑项"和正文**捆在一起**返回。
_REPORT_SECTIONS = (
    ("技术面", "tech_report", False),
    ("基本面", "fundamental_report", False),
    ("舆情", "news_report", False),
    ("多头辩论", "bull_history", True),
    ("空头辩论", "bear_history", True),
    ("研究汇总", "research_summary", False),
    ("交易计划", "trade_plan", False),
    ("风控审核", "risk_assessment", False),
    ("最终决策", "final_decision", False),
)
_SECTION_NAMES = tuple(name for name, _k, _m in _REPORT_SECTIONS)
_REPORT_USAGE = (
    "使用须知：① 引用报告内容时必须写明**段落名 + analysis_id**"
    "（如「技术面（analysis_id=11）」「研究汇总 ③」），不要把报告结论与你自己的推算混在一起；"
    "② 某个段落为 null/空串表示**该次分析没产出这一段**，不等于「这一段没有问题」；"
    "③ 「本报告的可核实性告警」是这份报告**自身**的可疑项（输入中找不到出处的年份/来源/数字），"
    "转述其结论时必须一并说明，不得把可疑结论当既定事实；"
    "④ 报告是**历史产物**，其数据截止日可能早于当前，引用时连带说明日期。"
)


def _report_section_text(payload: dict, key: str, is_list: bool, cap: int) -> str:
    """取一个段落的文本并按 cap 截断（截断处标明总长度，避免"看着像全文"）。"""
    val = payload.get(key)
    if is_list:
        parts = [str(x) for x in (val or [])]
        text = "\n\n".join(parts)
    elif isinstance(val, (dict, list)):
        text = json.dumps(val, ensure_ascii=False, default=str)
    else:
        text = str(val or "")
    text = text.strip()
    if not text:
        return ""
    if len(text) > cap:
        return text[:cap] + f"…（本节共 {len(text)} 字，已截断）"
    return text


@_tool
def tool_analysis_report(code: str = "", sections: list = None, limit: int = 1) -> dict:
    """读取**本地已经跑过**的深度分析报告（历史产物），可按段落取。

    用户提到「报告 / 深度分析 / 刚才那份 / 某个页面的结论」，或追问某份报告里的判断
    （如「为什么报告说舆情乐观却看空」「报告的风控结论是什么」）时，**必须用本工具**，
    不要用行情/新闻工具重新推一遍 —— 那回答的是"按现在数据我会怎么看"，
    而不是"报告里为什么这么写"。

    返回里含该报告**自身**的可核实性告警，转述其结论时必须一并说明。
    """
    from db import store

    code = str(code or "").strip()
    limit = max(1, min(int(limit or 1), 5))
    # 段落少 → 给足长度；段落多 → 每段压缩，避免把上下文塞爆（总量基本恒定在 ~9k 字内）
    chosen = [s for s in (sections or _SECTION_NAMES) if s in _SECTION_NAMES]
    if not chosen:
        chosen = list(_SECTION_NAMES)
    cap = 4000 if len(chosen) <= 2 else 1000

    try:
        rows = store.list_analyses(limit=limit, stock_code=code or None)
    except Exception as e:                                    # noqa: BLE001
        _logger.warning("[Agent工具] 读分析历史失败: %s", e)
        return {"error": f"读取本地分析历史失败：{type(e).__name__}: {e}"}

    if not rows:
        who = f"{code} " if code else ""
        # 用 error 语义：系统提示词规定「工具返回 error 时如实说明拿不到，不要猜」，
        # 这正是我们要的行为（"我读不到这份报告"），而不是让它把行情推一遍当答案。
        return {"error": f"本地分析库里没有 {who}的深度分析记录（从未跑过，或记录已被清理）",
                "提示": "可引导用户去「深度分析」页跑一次；在此期间不要凭空推断「报告里说了什么」",
                "as_of": _now()}

    reports = []
    for row in rows:
        detail = store.get_analysis(row["id"]) or {}
        payload = detail.get("payload") or {}
        body = {name: _report_section_text(payload, key, is_list, cap)
                for name, key, is_list in _REPORT_SECTIONS if name in chosen}
        audit_warns = list(payload.get("audit_warnings") or [])
        reports.append({
            "analysis_id": row["id"],
            "股票": f"{row.get('stock_name') or ''}({row.get('stock_code') or ''})",
            "分析日期": row.get("analysis_date") or payload.get("analysis_date") or "",
            "数据截止": payload.get("data_through") or "",
            "决策标签": row.get("decision_label") or "",
            "数据可用性": payload.get("data_availability") or {},
            "段落": body,
            "本报告的可核实性告警": audit_warns,
            # ⚠️ 「告警为空」不等于「报告可信」。审计只覆盖部分线索（年份/来源/依据/数值），
            #    而且**旧版本进程产出的报告压根没跑审计**（实测 id=11 就是这种情况）。
            #    不写这句，模型会把"零告警"读成"已核实"。
            "可核实性说明": ("本报告未产生任何审计告警。但这**不等于**内容已核实："
                        "审计只覆盖年份/来源/依据/数值四类线索，且该次分析若未启用审计"
                        "（例如由旧版本进程产出）也会是空 —— 请按「未经核实」对待。"
                        if not audit_warns else
                        "上面列出的告警是这份报告**自身**的可疑项，"
                        "转述其结论时必须一并说明。"),
            "本报告的其他记录": [e for e in (payload.get("errors") or [])
                                 if e not in audit_warns],
            "致命错误": list(payload.get("fatal_errors") or []),
        })

    return {"as_of": _now(), "找到份数": len(reports), "报告": reports,
            "使用须知": _REPORT_USAGE}


# ================================================================
# Schema 与分发注册表
# ================================================================

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "tool_realtime_quotes",
        "description": "获取实时行情（现价/涨跌幅/换手/成交额）。不传codes返回自选股全部行情；最快，盘面问题优先用",
        "parameters": {"type": "object", "properties": {
            "codes": {"type": "array", "items": {"type": "string"},
                      "description": "6位股票代码列表，可选；不传则返回自选股"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "tool_stock_profile",
        "description": "个股基本信息与估值（名称/现价/PE/PB/市值）",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "6位股票代码"}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "tool_stock_technicals",
        "description": "个股技术面：均线系统/均线形态/近20日涨跌幅/波动率",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "6位股票代码"}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "tool_fund_flow",
        "description": "个股资金面：五维资金评分(满分25)与各窗口净额",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "6位股票代码"}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "tool_market_signal",
        "description": "当前大盘环境检测（14指标：趋势/量能/情绪/结构与仓位建议）。耗时约30秒",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "tool_market_sentiment",
        "description": "市场情绪参照：从已采集的散户/博主言论里提炼的多空比、分歧度、覆盖率和相对自身历史分位。"
                       "⚠️ 只描述情绪位置，**不是**买卖依据，也不参与任何评分；"
                       "数据质量非「可用」时比例不可作依据。零联网、读当日已落库快照",
        "parameters": {"type": "object", "properties": {
            "date": {"type": "string",
                     "description": "采集日 YYYY-MM-DD，默认今天；只有回溯历史时才传"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "tool_screen_stocks",
        "description": "执行完整选股流程，输出TOP候选及因子拆解。耗时约2-4分钟，仅当用户明确要求选股时调用",
        "parameters": {"type": "object", "properties": {
            "sector": {"type": "string", "enum": ["科技", "金融", "消费", "新能源", "医药", "创业板"],
                       "description": "限定板块，可选"},
            "max_stocks": {"type": "integer", "description": "候选数量，默认5"},
            "strategies": {"type": "array", "items": {"type": "string",
                           "enum": ["主力资金", "技术突破", "低估值成长", "龙虎榜活跃", "行业龙头"]},
                           "description": "初筛策略子集，可选；用户指定策略/风格时传，否则不传（默认全部五路）"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "tool_backtest_summary",
        "description": "运行规则信号回测并返回绩效摘要(收益/夏普/回撤/信息比率)。耗时约1-2分钟，仅当用户明确要求回测时调用",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "回看天数，默认60"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "tool_manage_watchlist",
        "description": "管理自选股（**写操作两阶段**：add/remove/update 只产出待确认 preview 与 change_id；"
                       "必须向用户复述差异、用户明确同意后用 action=commit+change_id 执行）。list 只读",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["list", "add", "remove", "update", "commit", "restore"]},
            "code": {"type": "string", "description": "6位股票代码(add/remove/update必填)"},
            "name": {"type": "string", "description": "股票名称(add可选/update只改传了非空值的)"},
            "note": {"type": "string", "description": "备注(add可选/update时写入新备注)"},
            "clear_note": {"type": "boolean", "description": "update时清空备注，默认false"},
            "change_id": {"type": "string", "description": "action=commit时必填：用户确认过的preview凭证"}},
            "required": ["action"]}}},
    {"type": "function", "function": {
        "name": "tool_financials",
        "description": "个股财务报表：营业收入/净利润/资产总计/负债合计/权益等关键科目（亿元）+ "
                       "同比增速 + 资产负债率/净利率/ROE。问'基本面/业绩/财务怎么样/值不值得投资'时用它；"
                       "注意不良贷款率、净息差等银行专项指标不在三表内，工具会标明取不到",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "6位股票代码"},
            "periods": {"type": "integer", "description": "返回期数 1~8，默认4（最近报告期在前）"}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "tool_latest_news",
        "description": "最近财经新闻/公告（传 code 时按个股直查东财，未命中回落本地快讯库；含情绪标签与关联股票）。用户问新闻/消息面/利好利空/今天最新时使用，此时不要传keyword",
        "parameters": {"type": "object", "properties": {
            "keyword": {"type": "string", "description": "主题词/公司名过滤（如'半导体'、'茅台'），可选。勿传'今天/最新'类时间词"},
            "code": {"type": "string", "description": "6位股票代码，可选。查某只票的消息面/公告时务必传，冷门个股只有传了才查得到"},
            "limit": {"type": "integer", "description": "条数，默认15"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "tool_similar_days",
        "description": "检索与当前最相似的历史交易日及后续5/20日表现分布（历史统计参考非预测）。用户问'像历史上哪天/历史上类似走势后来怎样'时使用",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "标的代码：指数 sh000001/sz399001/sz399006 或6位股票代码，默认上证"},
            "top_k": {"type": "integer", "description": "相似日数量，默认5"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "tool_backfill_similar_history",
        "description": "回填标的的历史特征库（相似日检索报'特征库不足'时的前置步骤，约1-2分钟）",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "标的代码，默认上证"},
            "years": {"type": "integer", "description": "回填年数，默认10"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "tool_market_opinions",
        "description": "聚合'如何看待X月X日A股行情'的市场讨论观点（联网检索知乎/雪球/财联社收盘点评，含看多看空倾向）。用户问'大家怎么看今天行情/市场观点/散户情绪'时使用",
        "parameters": {"type": "object", "properties": {
            "date": {"type": "string", "description": "如'08月28日'，默认最近交易日"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "tool_portfolio_positions",
        "description": "查询**真实持仓**（用户在「持仓」页手工录入的那份）与盈亏（现价/浮盈亏/总资产/现金）。用户问'我的持仓/我的仓位'时使用。注意：这与「模拟账户」的模拟盘是两套独立账本，问模拟盘请用 tool_paper_account",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "tool_paper_account",
        "description": "查询**模拟账户（模拟盘/纸上交易）**的持仓明细、可卖数量、成本价、浮盈亏、可用现金、总资产与累计收益率。用户问'模拟账户/模拟盘/虚拟盘怎么样、赚了亏了、要不要调仓'时使用。与 tool_portfolio_positions（真实持仓）是两套独立账本，互不推断",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "tool_ashen_status",
        "description": "查询系统内置的 3D 桌宠 AI 交易员**阿深**（persona=ai_ashen）：它今天 6 个排班轮次各自做了什么判断、它**自己的**模拟账户净值/持仓、它的自选池与入池理由、建仓权级别与机械边界（不追涨不接刀等）。用户问'阿深/桌宠/那个鲸鱼女孩/值班/建仓权/L2/它今天买了吗/它的池子/它的账户'时**必须用本工具**。⚠️ 阿深是**第三套账本**：既不是用户的模拟盘（tool_paper_account），也不是真实持仓（tool_portfolio_positions），三者不可混用、不可互相推断",
        "parameters": {"type": "object", "properties": {
            "day": {"type": "string", "description": "日期 YYYY-MM-DD，留空为今天"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "tool_analysis_report",
        "description": "读取本地**已经跑过的深度分析报告**（历史产物），可按段落取（技术面/基本面/舆情/多空辩论/研究汇总/交易计划/风控/最终决策）。用户提到「报告/深度分析/刚才那份」或追问某份报告里的判断（如「为什么报告说舆情乐观仍看空」「报告的风控结论是什么」）时，**必须用本工具**；不要用行情/新闻工具重新推一遍——那回答的是「按现在的数据我会怎么看」，不是「报告里为什么这么写」。返回里含这份报告**自身**的可核实性告警，转述其结论时必须一并说明",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "6位股票代码；留空则取最新一份（不限定股票）"},
            "sections": {"type": "array", "items": {"type": "string", "enum": [
                "技术面", "基本面", "舆情", "多头辩论", "空头辩论",
                "研究汇总", "交易计划", "风控审核", "最终决策"]},
                "description": "只读指定段落（留空读全部；读全部时每段会被压缩，追问细节时请指定 1~2 段）"},
            "limit": {"type": "integer", "description": "读最近几份报告，默认 1"}},
            "required": []}}},
    # ── 2026-10-24 系统级工具：让助手答得了"项目自身"的问题 ──────────────
    {"type": "function", "function": {
        "name": "tool_read_output_file",
        "description": "读取 output/ 下的历史报告与产物（复盘/联赛/评估报告、daily_picks）。"
                       "用户问'昨天的复盘说了什么/某份报告在哪/联赛结果如何'时**必须用它读原文**，"
                       "不要凭记忆或推断复述。用法：先 action=list 找文件名（可传 pattern 过滤），"
                       "再 read 分段读（offset/limit）或 tail 看结尾结论；.json 太大时用 json_path 取子对象。"
                       "⚠️ 读到的内容是历史产物，引用时必须注明文件日期；只能读 output/ 内的文件",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["list", "read", "tail"]},
            "file": {"type": "string", "description": "output/ 内的相对路径(read/tail必填)"},
            "pattern": {"type": "string", "description": "list时的文件名子串过滤，可选"},
            "offset": {"type": "integer", "description": "read起始行(0起)，默认0"},
            "limit": {"type": "integer", "description": "单次行数，read默认150/tail默认60，最大400"},
            "json_path": {"type": "string", "description": "点分路径取JSON子对象，如 picks.0.股票代码，可选"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "tool_db_inspect",
        "description": "查询项目运行记录（白名单只读，非任意SQL）：告警/买入信号/流程运行/选股样本/"
                       "数据覆盖率/推送记录/AI联赛决策/估值与资金流快照/情绪快照/LLM调用日志(含成本)。"
                       "用户问'最近有什么告警/信号分析了没/LLM调用失败没/花了多少钱/联赛今天决策了什么'时使用。"
                       "先 action=tables 看可查的表，再 action=query 按表查最近记录",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["tables", "query"]},
            "table": {"type": "string", "description": "query时的表名（先tables看可用表）"},
            "limit": {"type": "integer", "description": "条数，默认20，最大50"},
            "code": {"type": "string", "description": "按股票代码过滤，可选"},
            "date": {"type": "string", "description": "按日期过滤(2026-10-24或20261024)，可选"},
            "only_failures": {"type": "boolean", "description": "仅llm_calls失败记录，默认false"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "tool_config_view",
        "description": "查看项目配置（只读+密钥脱敏）：llm_feature_models=功能→模型路由，"
                       "llm_providers=厂商与模型清单，llm_prices=单价，ai_personas=AI人格等。"
                       "用户问'助手接的哪个模型/某功能用谁/模型多少钱'时使用。"
                       "⚠️ 只读工具；config改动需重启长驻进程才生效，改配置请到「系统配置」页",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["list", "read"]},
            "file": {"type": "string", "description": "config/ 下的 *.json 文件名(read必填)"}},
            "required": ["action"]}}},
    {"type": "function", "function": {
        "name": "tool_push_templates",
        "description": "管理飞书推送模板（存数据库，commit后**即时生效**无需重启）。"
                       "**写操作两阶段**：save/reset 只产出待确认 preview 与 change_id（含完整前后正文）；"
                       "必须向用户复述差异、用户明确同意后用 action=commit+change_id 执行；"
                       "restore+历史change_id 可回滚。变量白名单含{% if %}变量，拼错直接拒绝",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["list", "get", "save", "reset", "commit", "restore"]},
            "key": {"type": "string", "description": "模板key(get/save/reset必填，如alert/review_summary)"},
            "body": {"type": "string", "description": "save时的完整新正文（整体替换，只产出preview）"},
            "change_id": {"type": "string", "description": "commit/restore时必填：preview凭证或历史commit凭证"}},
            "required": ["action"]}}},
    {"type": "function", "function": {
        "name": "tool_scheduled_jobs",
        "description": "定时任务全生命周期（自定义任务commit后约60秒热同步，无需重启；内置任务只读）。"
                       "**写操作两阶段**：create/edit/toggle/delete 只产出待确认 preview 与 change_id；"
                       "必须向用户复述目标与差异、用户明确同意后用 action=commit+change_id 执行；"
                       "restore+历史change_id 可回滚。create/edit 的校验走项目唯一入口 validate_job"
                       "（mode白名单/cron归一/排期冲突检测）",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["list", "get", "create", "edit", "toggle", "delete", "commit", "restore"]},
            "job_id": {"type": "string", "description": "任务ID(get/edit/toggle/delete必填；create可留空自动生成，禁止与内置重名)"},
            "enabled": {"type": "boolean", "description": "toggle时true启用/false停用（只认布尔，字符串'false'会被拒）"},
            "name": {"type": "string", "description": "create必填/edit可选：任务显示名(≤40字)"},
            "mode": {"type": "string", "description": "create/edit可选：任务类型(可选值见preview返回的可选mode)"},
            "cron": {"type": "object", "description": "create/edit可选：如{\"hour\":9,\"minute\":5}"},
            "trading_day_only": {"type": "boolean", "description": "create/edit可选：非交易日是否跳过，默认true"},
            "change_id": {"type": "string", "description": "commit/restore时必填：preview凭证或历史commit凭证"}},
            "required": ["action"]}}},
    {"type": "function", "function": {
        "name": "tool_panel_bars",
        "description": "读本地日线面板（零联网毫秒级）：多标的对比、任意日期区间、qfq/hfq/none复权。"
                       "用户要'对比几只票的走势/看某只票某段历史行情'时用它（单只近6个月看指标仍用tool_stock_technicals）。"
                       "最多5个代码；返回数据起止/最新收盘/区间涨跌幅/最多60点收盘价采样",
        "parameters": {"type": "object", "properties": {
            "codes": {"type": "array", "items": {"type": "string"},
                      "minItems": 1, "maxItems": 5,
                      "description": "6位股票代码列表(1~5个，超出会报错而不是静默丢弃)"},
            "start": {"type": "string", "description": "起始日YYYY-MM-DD或YYYYMMDD，可选"},
            "end": {"type": "string", "description": "结束日YYYY-MM-DD或YYYYMMDD，可选"},
            "adjust": {"type": "string", "enum": ["qfq", "hfq", "none"],
                       "description": "复权方式，默认qfq前复权"}},
            "required": ["codes"]}}},
    {"type": "function", "function": {
        "name": "tool_data_freshness",
        "description": "数据新鲜度总览：今天/最近交易日、本地面板覆盖多少只票、面板市场级最新日、"
                       "各数据集覆盖率。用户问'数据是不是最新的/数据到哪天了'时使用。"
                       "⚠️ 口径：个股新鲜度基准=面板市场级最新日(manifest的date_max)不是今天；"
                       "最新日之后没有数据是'缺'不是'0'",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    # ── 2026-10-24 审查建议：结论类薄包装（页面同款函数）──
    {"type": "function", "function": {
        "name": "tool_quality_stats",
        "description": "选股/分析到底有没有用：T+5胜率/超额胜率/条件区分度/因子相关性。"
                       "用户问'这套选股有没有用/胜率多少/哪个条件有效/分析准不准'时使用。"
                       "⚠️ 铁律：样本不足(n<门槛)只报样本数不报胜率——胜率必须连同样本数一起引用",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "回看天数，默认90"},
            "kind": {"type": "string", "enum": ["overview", "picks", "analysis", "conditions", "factors"],
                     "description": "overview=摘要(默认)/picks=选股表现/analysis=分析胜率/"
                                   "conditions=条件区分度/factors=因子相关性"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "tool_llm_cost",
        "description": "LLM成本与预算：近N天总花费/按日/用途Top5/按模型+今日预算状态。"
                       "用户问'今天花了多少钱/哪最贵/还剩多少预算'时使用。"
                       "⚠️ 口径：按llm_prices.json估算非账单；账目异常(degraded)≠没花钱",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "回看天数，默认3"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "tool_system_health",
        "description": "系统健康自检：调度器心跳/是否旧代码/漏跑缺口/自定义任务同步。"
                       "用户问'调度器活着吗/你跑的是新代码吗/为什么任务没生效'时使用。"
                       "⚠️ 口径：心跳过期≠进程死；'比代码旧的起始文件'清单=改了没重启的证据",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "tool_decision_digest",
        "description": "今日决策摘要：选了什么/分析了什么/待触发观察/风险清单。"
                       "用户问'今天该做什么/上次判断对不对'时使用。"
                       "⚠️ 空候选≠没选——跳过原因(如熊市跳过)会如实带出",
        "parameters": {"type": "object", "properties": {
            "as_of": {"type": "string", "description": "日期YYYY-MM-DD，默认今天"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "tool_holdings_risk",
        "description": "真实持仓风险体检：集中度/行业暴露/止损状态/不可评项。"
                       "用户问'我的持仓危险吗/止损位合适吗'时使用（账本盈亏用tool_portfolio_positions）。"
                       "⚠️ 行业/流动性/快照不可评时单列，不并入'其他'",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

TOOL_FUNCS: dict[str, Callable] = {
    "tool_realtime_quotes": tool_realtime_quotes,
    "tool_stock_profile": tool_stock_profile,
    "tool_stock_technicals": tool_stock_technicals,
    "tool_fund_flow": tool_fund_flow,
    "tool_market_signal": tool_market_signal,
    "tool_market_sentiment": tool_market_sentiment,
    "tool_screen_stocks": tool_screen_stocks,
    "tool_backtest_summary": tool_backtest_summary,
    "tool_manage_watchlist": tool_manage_watchlist,
    "tool_financials": tool_financials,
    "tool_latest_news": tool_latest_news,
    "tool_similar_days": tool_similar_days,
    "tool_backfill_similar_history": tool_backfill_similar_history,
    "tool_market_opinions": tool_market_opinions,
    "tool_portfolio_positions": tool_portfolio_positions,
    "tool_paper_account": tool_paper_account,
    "tool_ashen_status": tool_ashen_status,
    "tool_analysis_report": tool_analysis_report,
    # ── 2026-10-24 系统级工具 ──
    "tool_read_output_file": tool_read_output_file,
    "tool_db_inspect": tool_db_inspect,
    "tool_config_view": tool_config_view,
    "tool_push_templates": tool_push_templates,
    "tool_scheduled_jobs": tool_scheduled_jobs,
    "tool_panel_bars": tool_panel_bars,
    "tool_data_freshness": tool_data_freshness,
    # ── 2026-10-24 结论类薄包装 ──
    "tool_quality_stats": tool_quality_stats,
    "tool_llm_cost": tool_llm_cost,
    "tool_system_health": tool_system_health,
    "tool_decision_digest": tool_decision_digest,
    "tool_holdings_risk": tool_holdings_risk,
}


def execute_tool(name: str, arguments: str | dict) -> dict:
    """按名执行工具。arguments 为 JSON 字符串或 dict；返回结果 dict（含 error 则工具失败）"""
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return {"error": f"未知工具: {name}"}
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
    except json.JSONDecodeError as e:
        return {"error": f"参数解析失败: {e}"}
    try:
        return fn(**args)
    except Exception as e:
        # 纵深防御：注册表函数均带 @_tool 包装，此处兜底未包装的异常
        _logger.exception("[Agent工具] %s 未捕获异常", name)
        return {"error": f"{type(e).__name__}: {e}", "as_of": _now()}
