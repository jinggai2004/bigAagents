"""数据源需求体检：逐条用**真实调用路径**验证「需求是否被满足」。

为什么需要
----------
"akshare 不稳定"那次排查暴露一个问题：**没人说得清系统到底依赖哪些数据、
每条需求现在是不是真的能满足**。靠读代码只能看到"调用了哪个函数"，
看不到"这个函数现在真的返回数据吗、用的哪个源、多快"。

本脚本把需求做成一张可重复执行的检查表：每条需求都走它**在生产里的真实入口**
（不是另写一个 demo 调用），报告 成功/失败 + 实际数据源 + 条数 + 耗时。

用法：
    python scripts/check_datasources.py            # 全量体检
    python scripts/check_datasources.py --json     # 输出 JSON（给脚本/自动化消费）
    python scripts/check_datasources.py -k 指数     # 只跑名字含"指数"的检查

退出码：0=全部满足；1=有需求未满足（便于接进自动化/CI）。

注意：本脚本是**只读**的（不落库、不写业务数据），只有交易日历会刷新本地缓存
（那本来就是它的正常行为）。全市场快照与市场检测两项较重（数秒~十几秒）。
"""
import argparse
import io
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 抑制数据层大量 print（选股/数据模块习惯直接 print），只在 --verbose 时放行
_QUIET = True


def _quiet():
    return redirect_stdout(io.StringIO()) if _QUIET else _null_ctx()


from contextlib import nullcontext as _null_ctx  # noqa: E402


# ================================================================
# 各需求探针：每个返回 (ok: bool, detail: str)
# ================================================================

def check_stock_kline():
    """个股日线 K 线 —— 回测、选股评分、T+N 回填的共同底座"""
    from data.source_manager import data_source_manager
    df = data_source_manager.get_stock_hist_data("600519", "20240101", "20240331")
    if df is None or getattr(df, "empty", True):
        return False, "未取到数据"
    need = {"date", "open", "close", "high", "low", "volume"}
    missing = need - set(df.columns)
    if missing:
        return False, f"缺列 {missing}"
    return True, f"{len(df)} 根 · {df['date'].min().date()}~{df['date'].max().date()}"


def check_stock_quote():
    """个股实时行情 —— 看盘、盯盘、复盘、模拟盘、异动告警、对话工具"""
    from data import realtime
    q = realtime.get_realtime_quotes(["600519"]).get("sh600519") \
        or realtime.get_realtime_quotes(["600519"]).get("600519")
    if not q or q.get("price") is None:
        return False, "无报价"
    return True, (f"{q.get('name')} {q['price']} · 源={q.get('source')} · "
                  f"stale={q.get('stale')}")


def check_index_quote():
    """指数实时行情 —— 看盘页、市场检测"""
    from data import realtime
    rows = realtime.get_index_quotes()
    if not rows:
        return False, "无指数报价"
    ok = [k for k, v in rows.items() if v and v.get("price") is not None]
    if not ok:
        return False, "指数报价全为空"
    return True, f"{len(ok)}/{len(rows)} 个指数有价 · {ok[:3]}"


def check_index_daily():
    """指数日线 —— 市场检测 6 项指标、择时、基准、相似日"""
    from data import benchmark
    nav = benchmark.get_benchmark_nav("2024-01-01", "2024-03-31")
    if not nav:
        return False, "基准净值曲线为空"
    src = "未知"
    try:
        from data.benchmark import _fetch_index_daily
        df = _fetch_index_daily("sh000300")
        src = f"{len(df)} 根"
    except Exception as e:                       # noqa: BLE001
        src = f"明细失败: {type(e).__name__}"
    return True, f"净值 {len(nav)} 点 · 指数日线 {src}"


def check_market_snapshot():
    """全市场快照 —— 市场检测(涨跌家数/市场宽度)、情绪打分"""
    import akshare as ak
    df = ak.stock_zh_a_spot_em()
    if df is None or df.empty:
        return False, "全市场快照为空"
    return True, f"{len(df)} 只（东财）"


def check_limit_pool():
    """涨停/跌停池 —— 市场检测(涨停家数/连板)、情绪"""
    import akshare as ak
    from data import trading_calendar
    day = trading_calendar.last_trading_day().replace("-", "")
    df = ak.stock_zt_pool_em(date=day)
    n = 0 if df is None or df.empty else len(df)
    # 涨停池为空是合法结果（如极端行情全跌），因此不判失败，只如实报告
    return True, f"{day} 涨停 {n} 家"


def check_fund_flow():
    """个股资金流 —— 选股评分(五维资金分)、深度分析 Agent、对话"""
    from data.fund_flow import FundFlowAkshareDataFetcher
    r = FundFlowAkshareDataFetcher().get_fund_flow_data("600519")
    if not r or r.get("status") == "unknown":
        return False, f"无资金流数据 ({r.get('status') if r else 'None'})"
    src = r.get("source") or r.get("data_source") or "?"
    return True, f"status={r.get('status')} · 源={src}"


def check_calendar():
    """交易日历 —— 调度器与选股的交易日判定（错了会漏跑/空跑）"""
    from data import trading_calendar
    r = trading_calendar.refresh(force=True)
    if not r.get("ok", True):
        return False, f"刷新失败: {r}"
    st = trading_calendar.cache_status()
    return True, (f"{st.get('count', '?')} 个交易日 · 源={r.get('source')} · "
                  f"降级={r.get('degraded')}")


def check_news():
    """新闻快讯 —— 消息面 Agent、对话检索、情绪"""
    from data import news_feed
    items = news_feed._fetch_source("sina")       # 单源，只读，不落库
    if not items:
        return False, "新浪快讯为空"
    return True, f"{len(items)} 条（新浪）"


def check_financial():
    """财务三表 —— 深度分析(基本面)"""
    from data.stock_data import StockDataFetcher
    d = StockDataFetcher().get_financial_data("600519")
    if not d:
        return False, "无财务数据"
    keys = [k for k in ("balance_sheet", "income_statement", "cash_flow",
                        "financial_abstract", "financial_indicators") if d.get(k)]
    if not keys:
        return False, f"返回了对象但三项报表都空: {list(d.keys())[:6]}"
    return True, f"实得 {'/'.join(keys)}"


def check_basic_info():
    """个股基本信息（名称/行业/市值）—— 选股展示、深度分析"""
    from data.source_manager import data_source_manager
    info = data_source_manager.get_stock_basic_info("600519")
    if not info or info.get("name") in (None, "", "未知"):
        return False, "未取到股票名称"
    return True, f"{info.get('name')} · 行业={info.get('industry')}"


def check_market_detect():
    """汇总：市场检测 14 指标的真实可用率（决定是否跳过选股，影响面最大）

    ⚠️ 不能只看 `indicators` 里有没有键：取数失败的指标会填**默认值**（如涨跌家数
    失败给中性 4/8），键还在、值是 0——只看键会把"失败"数成"有数据"。
    真正的失败信号在 `scores[*]["detail"]`（含"失败"字样）与 `unavailable` 标记里。
    """
    from core.market_detector import MarketDetector
    sig = MarketDetector().detect()
    scores = sig.scores or {}
    failed, disabled = [], []
    for k, v in scores.items():
        if not isinstance(v, dict):
            failed.append(k)
        elif v.get("unavailable"):
            disabled.append(k)                # 数据已停更的指标（如北向），属于已知停用
        elif "失败" in str(v.get("detail", "")):
            failed.append(k)
    total = len(scores) or len(sig.indicators or {})
    covered = total - len(failed) - len(disabled)
    ok = total > 0 and covered >= max(1, int((total - len(disabled)) * 0.8))
    detail = (f"{covered}/{total} 指标可用 · {sig.market_state}"
              f"({sig.total_score:.0f}分) · 仓位上限 {sig.position_limit}")
    if disabled:
        detail += f" · 已停用: {','.join(disabled)}"
    if failed:
        detail += f" · ⚠️取数失败: {','.join(failed)}"
    return ok, detail


# ================================================================
# 检查表：需求 → 消费者 → 探针
# ================================================================

CHECKS = [
    ("个股日线 K 线", "回测/选股评分/T+N回填", check_stock_kline),
    ("个股实时行情", "看盘/盯盘/复盘/模拟盘/告警/对话", check_stock_quote),
    ("指数实时行情", "看盘页/市场检测", check_index_quote),
    ("指数日线", "市场检测6指标/择时/基准/相似日", check_index_daily),
    ("全市场快照", "市场检测(涨跌家数/宽度)/情绪", check_market_snapshot),
    ("涨跌停池", "市场检测(涨停家数/连板)/情绪", check_limit_pool),
    ("个股资金流", "选股评分(资金维度)/深度分析/对话", check_fund_flow),
    ("交易日历", "调度器/选股(交易日判定)", check_calendar),
    ("新闻快讯", "消息面Agent/对话检索/情绪", check_news),
    ("财务三表", "深度分析(基本面)", check_financial),
    ("个股基本信息", "选股展示/深度分析", check_basic_info),
    ("市场检测(汇总)", "选股前置闸门(熊市跳过)", check_market_detect),
]


def main():
    global _QUIET
    ap = argparse.ArgumentParser(description="数据源需求体检")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("-k", "--filter", default="", help="只跑名字包含该关键字的检查")
    ap.add_argument("-v", "--verbose", action="store_true", help="放行数据层 print")
    args = ap.parse_args()
    _QUIET = not args.verbose

    selected = [(n, c, f) for n, c, f in CHECKS if args.filter in n]

    results = []
    for name, consumer, fn in selected:
        t0 = time.time()
        try:
            with _quiet():
                ok, detail = fn()
        except Exception as e:                   # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {str(e)[:90]}"
            if args.verbose:
                traceback.print_exc()
        results.append({"need": name, "consumer": consumer, "ok": bool(ok),
                        "detail": detail, "seconds": round(time.time() - t0, 2)})

    if args.json:
        print(json.dumps({"results": results,
                          "satisfied": sum(r["ok"] for r in results),
                          "total": len(results)}, ensure_ascii=False, indent=2))
    else:
        print(f"=== 数据源需求体检（{len(results)} 项）===")
        w = max(len(r["need"]) for r in results) if results else 10
        for r in results:
            mark = "✅" if r["ok"] else "❌"
            print(f"{mark} {r['need']:<{w}}  {r['detail']}  [{r['seconds']}s]")
            print(f"   └ 消费者: {r['consumer']}")
        bad = [r for r in results if not r["ok"]]
        spent = sum(r["seconds"] for r in results)
        print(f"\n汇总: {len(results) - len(bad)}/{len(results)} 满足 · 耗时 {spent:.1f}s")
        if bad:
            print("未满足:")
            for r in bad:
                print(f"  ❌ {r['need']}: {r['detail']}")

    return 1 if any(not r["ok"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
