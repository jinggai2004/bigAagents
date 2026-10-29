"""
全流程编排器：选股 → (可选)LLM深度分析 → 规则信号回测 → 报告落盘。

三种运行模式：
  - "screen": 仅选股（最快，供盘前快速出候选）
  - "rule":   选股 + 规则信号回测（零 LLM 成本，定时任务默认模式）
  - "llm":    选股 + LLM 深度分析 + 规则信号回测（成本高，建议手动触发）

产出写入 output/daily_picks/（daily_YYYYmmdd_HHMM.json + latest.json），
回测报告由 BacktestEngine 存入 data/backtest_records/（回测看板可直接读取）。
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta

from core.state import merge_delta

_logger = logging.getLogger(__name__)

# 数据根必须**运行时**解析：打包后 `__file__` 位于只读包 `_internal/` 里，
# 用它推导会指向空目录 ⇒ 读到空数据（2026-10-25 实测：双击 exe 后
# 选股/复盘/研究全 missing、交易台变空账户、系统页看不到调度器）。
# 规则见 `config/paths.data_root()`；源码运行结果与原先一致。
from config.paths import data_root as _paths_data_root  # noqa: E402
_PROJECT_ROOT = _paths_data_root()
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output", "daily_picks")


def _json_default(o):
    """json.dump 的 default：dataclass 落盘为 dict，其余维持 str 兜底。

    ⚠️⚠️ 2026-10-25 修（C1）：原先 `default=str` 把 `MarketSignal`（dataclass）打成
    **repr 文本**——`MarketSignal(total_score=28.4, market_state='偏弱震荡', ...)`，
    latest.json 里 `market_signal` 是 str 不是对象。后果：
      · 新前端 `/api/picks/latest` 透传后前端拿到一整串文本，没有字段可读；
      · 每个消费方各自长一套解析（`core/ai_daily_round` 用正则取三个数，
        取不到就"解析失败 ⇒ 目标仓位不可用"）。
    ⇒ 落盘统一转 dict（`MarketSignal` 全字段是 float/str/dict/list，asdict 安全）。
      配套改动（必须成对，否则打断联赛目标仓位）：
      · `core/ai_daily_round._load_market_signal`：dict 直读 + 旧 repr 串正则回退；
      · `ui/pages/1` 的 `render_screen_result` 本就兼容 dict（`_signal_from_dict`）。
    ⚠️ 磁盘上 2026-10-25 前的 latest.json 仍是 repr 串 ⇒ 消费方必须**同时**吃两种格式。
    """
    import dataclasses
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    return str(o)


class DailyPipeline:
    """每日自动化流水线"""

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # 分阶段耗时（2026-10-21）：落进 `daily_picks/*.json` 的 `stage_timings`。
        # ⚠️ 为什么不只打日志：日志是给人看的，而"扩池前后哪一步变慢"要程序能读、
        #    能跨多次运行对比（`elapsed_seconds` 只有一个总数，看不出是选股慢还是回测慢）。
        # ⚠️ 用 perf_counter（单调）而不是 time()：后者会被系统对时/夏令时跳变影响，
        #    出现负耗时。
        self._timings: dict = {}

    # ================================================================
    # 主入口
    # ================================================================

    def run(self, mode: str = "rule", sector_filter=None, max_stocks: int = 5,
            backtest_days: int = 60, stock_codes: list[str] | None = None) -> dict:
        """
        执行流水线。

        Args:
            mode: "screen" / "rule" / "llm"
            sector_filter: 限定板块（None=全市场）
            max_stocks: 候选股数量
            backtest_days: 规则回测回看天数
            stock_codes: 直接指定股票列表（跳过选股阶段）
        """
        t0 = time.time()
        today = datetime.now().strftime("%Y-%m-%d")
        self._timings = {}          # 每次 run 重置，避免同一实例多次运行时累积
        report = {
            "date": today,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "picks": [],
            "analysis": [],
            "backtest": None,
            "market_signal": None,
            "skipped": False,
            "skip_reason": "",
            "errors": [],
            "elapsed_seconds": 0,
        }

        # ============= Stage 1: 选股 =============
        _logger.info("[Pipeline] Stage 1/3 选股开始 (sector=%s, max=%d)", sector_filter, max_stocks)
        try:
            picks, market_signal, skip_reason = self._stage_screen(sector_filter, max_stocks, stock_codes)
            report["picks"] = picks
            report["market_signal"] = market_signal
            report["skipped"] = bool(skip_reason)
            report["skip_reason"] = skip_reason
        except Exception as e:
            _logger.exception("[Pipeline] 选股阶段异常")
            report["errors"].append(f"选股失败: {e}")
            self._save(report)
            return report

        if report["skipped"]:
            _logger.info("[Pipeline] 熊市/异常市况跳过: %s", skip_reason)
            report["elapsed_seconds"] = round(time.time() - t0, 1)
            self._save(report)
            return report

        if not report["picks"]:
            report["errors"].append("选股阶段无候选结果")
            report["elapsed_seconds"] = round(time.time() - t0, 1)
            self._save(report)
            return report

        # ============= Stage 2: 分析 =============
        _logger.info("[Pipeline] Stage 2/3 分析开始 (mode=%s, %d只)", mode, len(report["picks"]))
        try:
            if mode == "llm":
                report["analysis"] = self._stage_analyze_llm(report["picks"])
            else:
                report["analysis"] = [
                    {"股票代码": p.get("股票代码", ""), "股票名称": p.get("股票名称", ""),
                     "score": p.get("score", 0), "decision": "规则模式：LLM分析未启用",
                     "factors": p.get("factors", {})}
                    for p in report["picks"]
                ]
        except Exception as e:
            _logger.exception("[Pipeline] 分析阶段异常")
            report["errors"].append(f"分析失败: {e}")

        # ============= Stage 3: 回测 =============
        if mode in ("rule", "llm"):
            _logger.info("[Pipeline] Stage 3/3 规则回测开始 (%d天窗口)", backtest_days)
            try:
                report["backtest"] = self._stage_backtest(backtest_days)
            except Exception as e:
                _logger.exception("[Pipeline] 回测阶段异常")
                report["errors"].append(f"回测失败: {e}")

        report["elapsed_seconds"] = round(time.time() - t0, 1)
        self._save(report)
        _logger.info("[Pipeline] 完成: %.1fs, picks=%d, errors=%d",
                     report["elapsed_seconds"], len(report["picks"]), len(report["errors"]))
        return report

    # ================================================================
    # Stage 实现
    # ================================================================

    def _stage_screen(self, sector_filter, max_stocks, stock_codes=None):
        """选股阶段。返回 (picks, market_signal, skip_reason)"""
        if stock_codes:
            # 用户直接指定代码时并未经过选股评分：score 置 None，不伪造 0 分
            picks = [{"股票代码": c, "股票名称": "", "score": None, "factors": {},
                      "score_status": "not_screened"} for c in stock_codes]
            return picks, None, ""

        from core.screener import StockScreener
        screener = StockScreener()
        # ⚠️ 分阶段计时（2026-10-21）：`screener.run` 是 Stage 1 的绝对大头，
        #    落库是次要但会随候选数线性增长 —— 扩池到 300 时这两步要分开看，
        #    否则"选股变慢了"无从判断是打分慢还是写库慢。
        _t = time.perf_counter()
        try:
            result = screener.run(sector_filter=sector_filter, max_stocks=max_stocks)
        finally:
            # ⚠️ 用 finally（2026-10-22）：原实现只在成功时记录，于是"选股抛异常"
            #    这个**最该看耗时**的场景反而 `stage_timings` 是空的。
            self._timings["screen.run"] = round(time.perf_counter() - _t, 2)
        market_signal = result.get("market_signal")
        if result.get("skipped"):
            return [], market_signal, result.get("reason", "市场环境不佳")
        picks = result.get("candidates", [])
        self._timings["screen.candidates"] = len(picks)
        # 选股候选入候选库（调度跑的这批是最稳定的样本来源）
        try:
            from core.screen_history import save_screen_result
            _t = time.perf_counter()
            try:
                save_screen_result(result, source="pipeline")
            finally:
                self._timings["screen.persist"] = round(time.perf_counter() - _t, 2)
        except Exception as e:                  # noqa: BLE001
            _logger.warning("[Pipeline] 选股候选落库失败: %s", e)
        _logger.info("[Pipeline] Stage 1 细分：打分 %.1fs｜落库 %.2fs｜候选 %d 只",
                     self._timings.get("screen.run", 0),
                     self._timings.get("screen.persist", 0), len(picks))
        return picks, market_signal, ""

    def _stage_analyze_llm(self, picks: list[dict]) -> list[dict]:
        """LLM 深度分析阶段：逐只跑 LangGraph 多Agent分析（成本高）"""
        from core.graph_builder import analysis_graph
        from core.analysis_history import save_analysis, save_failure

        results = []
        # 候选全集：risk_node 依赖 state.candidates 做组合级仓位计划与止损止盈，
        # 缺失时系统级风控会以空列表评估而完全空转（原实现即如此）。
        candidates = [
            {"股票代码": q.get("股票代码", ""), "股票名称": q.get("股票名称", ""),
             "score": q.get("score", 0), "板块": q.get("板块", "")}
            for q in picks
        ]
        for p in picks:
            code = p.get("股票代码", "")
            try:
                initial_state = {
                    "stock_code": code, "stock_name": p.get("股票名称", ""), "analysis_date": "",
                    "stock_info": {}, "kline_data": None, "indicators": {}, "fund_flow_data": {},
                    "tech_report": "", "fundamental_report": "", "news_report": "",
                    "bull_argument": "", "bear_argument": "", "debate_count": 0,
                    "max_debate_rounds": 2, "bull_history": [], "bear_history": [],
                    "research_summary": "", "trade_plan": {}, "risk_assessment": "",
                    "final_decision": "", "screener_score": p.get("score", 0.0),
                    "screener_factors": p.get("factors", {}), "screener_sector": p.get("策略", ""),
                    "candidates": candidates,
                    "timing_signal": {}, "position_limit": None,
                    "risk_report": {}, "risk_warnings": [],
                    "messages": [], "errors": [], "fatal_errors": [],
                }
                final_state = initial_state
                for event in analysis_graph.stream(initial_state):
                    # ⚠️ 不能用裸 update：errors / fatal_errors / data_availability 在
                    # core/state.py 里挂了 reducer（只追加 / dict 合并），图内部会自动归并，
                    # 但这里是手工累积状态，`update` 是覆盖语义 → 不按同样规则归并的话，
                    # 整条流水线只保得住最后一个节点写的内容（错误与审计告警全丢）。
                    final_state.update(merge_delta(final_state, next(iter(event.values()))))
                if final_state.get("fatal_errors"):
                    _logger.warning("[Pipeline] %s 致命错误，分析链已阻断: %s",
                                    code, final_state["fatal_errors"])
                # 分析历史落库（best-effort，内部吞异常，不影响流水线）
                save_analysis(final_state, source="pipeline")
                results.append({
                    "股票代码": code,
                    "股票名称": final_state.get("stock_name", ""),
                    "score": p.get("score", 0),
                    "decision": final_state.get("final_decision", ""),
                    "trade_plan": final_state.get("trade_plan", {}),
                    "risk_assessment": final_state.get("risk_assessment", ""),
                    "risk_report": final_state.get("risk_report", {}),
                    "timing_signal": final_state.get("timing_signal", {}),
                    "errors": final_state.get("errors", []),
                    "fatal_errors": final_state.get("fatal_errors", []),
                })
            except Exception as e:
                _logger.warning("[Pipeline] %s LLM分析失败: %s", code, e)
                # 失败样本也留痕（backfill_status=skip，避免胜率统计幸存者偏差）
                save_failure(code, str(e), source="pipeline",
                             stock_name=p.get("股票名称", ""))
                results.append({"股票代码": code, "decision": "分析失败", "error": str(e)})
        return results

    def _stage_backtest(self, backtest_days: int) -> dict | None:
        """规则信号回测阶段（零LLM），复用 backtest_rule_based 的信号生成器"""
        from scripts.backtest_rule_based import RuleSignalGenerator
        from core.backtest import BacktestEngine
        from data.benchmark import get_benchmark_nav

        end = datetime.now()
        start = end - timedelta(days=backtest_days)
        start_str, end_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        generator = RuleSignalGenerator()
        trades = generator.generate_signals(start_str, end_str)
        if not trades:
            _logger.warning("[Pipeline] 回测窗口 %s~%s 无交易信号", start_str, end_str)
            return {"window": [start_str, end_str], "note": "无交易信号"}

        engine = BacktestEngine()
        engine.inject_kline_cache(generator._data_cache)
        try:
            bench = get_benchmark_nav(start_str, end_str)
            if bench:
                engine.set_benchmark(bench)
        except Exception as e:
            _logger.warning("[Pipeline] 基准接入失败: %s", e)

        # v3：显式传入区间端点 —— 逐日估值需要知道窗口到哪天为止，
        # 否则"窗口末端仍持仓"的部分不会被计入净值（回撤又会失真）
        result = engine.run(trades, start_date=start_str, end_date=end_str)
        report_path = engine.save_report(result)
        d = result.to_dict()
        d["report_file"] = os.path.basename(report_path)
        d["window"] = [start_str, end_str]
        return d

    # ================================================================
    # 落盘
    # ================================================================

    def _save(self, report: dict):
        """写入 daily_时间戳.json 与 latest.json，并落库运行元数据"""
        # 分阶段耗时（2026-10-21）：所有出口（选股失败/跳过/正常）都会走到这里，
        # 注入一次即可，不必在每个 return 前重复写。
        if self._timings:
            report["stage_timings"] = dict(self._timings)
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            path = os.path.join(self.output_dir, f"daily_{stamp}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
            latest = os.path.join(self.output_dir, "latest.json")
            with open(latest, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
            try:
                from db import store as db_store
                bt = report.get("backtest") or {}
                db_store.insert_run("pipeline", {
                    "mode": report.get("mode"), "picks": len(report.get("picks", [])),
                    "skipped": report.get("skipped"),
                    "errors": len(report.get("errors", [])),
                    "total_return": bt.get("total_return") if bt else None,
                    "file": os.path.basename(path),
                })
            except Exception:
                pass  # 元数据落库失败不影响主流程
            _logger.info("[Pipeline] 报告已保存: %s", path)
        except Exception as e:
            _logger.exception("[Pipeline] 报告保存失败: %s", e)
