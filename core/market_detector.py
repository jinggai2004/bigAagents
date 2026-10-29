"""市场环境检测器

在选股前运行，评估大盘环境，输出仓位上限和操作建议。
14个指标覆盖趋势/量能/情绪/结构四维度。

优化项（vs原方案）：
- ⑤⑦⑧ 涨跌停相关指标：改用 ak.stock_zt_pool_em() 替代 pywencai
- ⑨ 市场宽度：改用 get_market_snapshot() 一次获取全市场行情（15s→0.5s）
- 线程池 14→8，避免 AKShare 并发过高被限流

口径约定（**必须遵守**）：任何 `_score_*` 方法在**取不到数据**时，返回的 dict 必须带
`"unavailable": True`（可以同时保留一个保守的 `score`/`max_score` 供 UI 展示）。
`detect()` 据此把它**既不计分、也不计入分母**。
2026-09-28 修正：此前除 `north_flow` 外，14 个指标取数失败时都只返回中间默认分
（5/10、4/8、3/7…）而**没打这个标志** → 默认分被当成真实得分计入总分，
等于把"数据缺失"伪装成"中等偏弱"。当日 `涨跌家数(4/8)`、`市场宽度(3/7)`
就因东财快照断连而被顶上了默认分，白送 15% 权重。
"""

import akshare as ak
import logging

_logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
from data.eastmoney_source import get_market_snapshot
from data.trading_calendar import last_trading_day
# ThreadPoolExecutor removed — mini_racer (V8 JS engine used by akshare)
# does not support multi-threaded access on Windows


@dataclass
class MarketSignal:
    """市场检测结果"""
    total_score: float              # 总分 (0-100)
    market_state: str               # 牛市/偏强震荡/中性震荡/偏弱震荡/熊市
    signal: str                     # 强烈买入/买入/观望/卖出/强烈卖出
    position_limit: float           # 仓位上限 (0.0-0.8)
    strategy_bias: str              # 策略偏向（进攻型/均衡型/防御型/清仓）
    operation_advice: str           # 操作建议文本
    scores: dict                    # 各维度得分明细
    indicators: dict                # 各指标原始值
    risk_warnings: list[str]        # 风险提示
    timestamp: str                  # 检测时间


class MarketDetector:
    """市场环境检测器"""

    def __init__(self):
        self.weights = {
            # 趋势维度 (30%)
            "hs300_ma20_deviation": 0.10,
            "sh_macd": 0.10,
            "cyb_relative_strength": 0.10,
            # 量能维度 (25%)
            "volume_vs_ma20": 0.12,
            "advance_decline_ratio": 0.08,
            "north_flow": 0.05,
            # 情绪维度 (25%)
            "limit_up_down_ratio": 0.08,
            "max_consecutive_board": 0.05,
            "market_breadth": 0.07,
            "ivix_level": 0.05,
            # 结构维度 (20%)
            "sector_rotation_speed": 0.05,
            "size_style_bias": 0.05,
            "margin_balance_change": 0.05,
            "limit_up_upgrade_rate": 0.05,
        }

    # ================================================================
    # B2②（2026-10-19）：本地优先取数层 —— 指数日线走 `index_panel`，
    # 市场宽度/涨停三指标走 `zt_calendar`（一份面板切片喂 4 个指标）。
    # 在线时本地缺失才回退 akshare；离线（BIGA_OFFLINE=1）零外呼。
    # ================================================================

    def _index(self, symbol: str, min_rows: int = 21):
        """读本地指数日线；本地缺失/过期才回退 akshare（在线），离线直接 None。

        Returns:
            DataFrame（小写列，与 akshare 同构）或 None。attrs["source"]
            ∈ {"index_panel", "akshare"} —— 消费方的 detail 应体现来源。
        """
        import data.index_bars as index_bars
        try:
            df = index_bars.get_index_bars(symbol, min_rows=min_rows)
            if df is not None:
                fr = index_bars.index_freshness(symbol)
                if fr["fresh"]:
                    df.attrs["freshness"] = fr
                    return df
                # ⚠️ P1-2（2026-10-19 审查）：新鲜度判据原来拿交易日历的**最后
                # 一个元素**当基准（日历末 10-16 < 数据 10-19 ⇒ 恒真，闸门等于没装）。
                # 现在基准是"最近应收盘的交易日"，落后超过 1 个交易日就必须
                # 判不新鲜、回落联网（离线则不可评）——否则指数停更几天，
                # 检测仍会用陈旧数据给出一个"看起来很自信"的 market_state。
                _logger.warning("[detector] %s 本地指数不新鲜，回落联网/不可评: %s",
                                symbol, fr["reason"])
        except Exception as e:                              # noqa: BLE001
            _logger.debug("[detector] 本地指数 %s 不可用: %s", symbol, e)
        from data import bars as _bars
        if _bars.offline_mode():
            return None
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df is not None and not df.empty:
                df.attrs = getattr(df, "attrs", {}) or {}
                df.attrs["source"] = "akshare"
            return df
        except Exception as e:                              # noqa: BLE001
            _logger.debug("[detector] 指数 %s 联网失败: %s", symbol, e)
            return None

    @staticmethod
    def _idx_note(df) -> str:
        """指数数据的日期后缀（`｜指数至 2026-10-19`）——**数据日期必须随数据走**。

        不用 try/except：`getattr(df, "attrs", None)` 已覆盖 `df=None`，
        无谓的 except 会被可观测性棘轮算成"静默吞异常"。
        """
        attrs = getattr(df, "attrs", None) or {}
        last = attrs.get("last_date") if isinstance(attrs, dict) else None
        return f"，指数至 {last}" if last else ""

    def _zt_stats(self) -> dict:
        """涨停/宽度统计：本地 `zt_calendar` 优先，不可用时返回 {"available": False}。"""
        try:
            from core.zt_calendar import limit_up_stats
            return limit_up_stats()
        except Exception as e:                              # noqa: BLE001
            _logger.debug("[detector] 涨停日历失败: %s", e)
            return {"available": False, "caveat": str(e)}

    # ================================================================
    # 维度一：趋势检测 (30分)
    # ================================================================

    def _score_hs300_ma20(self) -> dict:
        """沪深300指数 20日均线偏离度"""
        try:
            df = self._index("sh000300")
            if df is None:
                return {"unavailable": True, "score": 5, "max_score": 10, "raw": 0,
                        "detail": "沪深300日线不可用（本地无且离线）"}
            closes = df["close"].values
            if len(closes) < 21:
                return {"unavailable": True, "score": 5, "max_score": 10, "raw": 0, "detail": "数据不足"}

            ma20 = np.mean(closes[-20:])
            current = closes[-1]
            deviation = (current - ma20) / ma20 * 100

            if deviation > 5:    score = 10
            elif deviation > 2:  score = 7
            elif deviation > -2: score = 5
            elif deviation > -5: score = 3
            else:                score = 0

            return {
                "score": score, "max_score": 10,
                "raw": round(deviation, 2),
                "detail": (f"沪深300偏离MA20: {deviation:+.2f}%, 现价: {current:.0f}, "
                           f"MA20: {ma20:.0f}{self._idx_note(df)}"),
            }
        except Exception as e:
            return {"unavailable": True, "score": 5, "max_score": 10, "raw": 0, "detail": f"获取失败: {e}"}

    def _score_sh_macd(self) -> dict:
        """上证指数 MACD 状态"""
        try:
            df = self._index("sh000001")
            closes = df["close"].values
            if len(closes) < 26:
                return {"unavailable": True, "score": 5, "max_score": 10, "raw": {}, "detail": "数据不足"}

            ema12 = pd.Series(closes).ewm(span=12).mean().values
            ema26 = pd.Series(closes).ewm(span=26).mean().values
            dif = ema12 - ema26
            dea = pd.Series(dif).ewm(span=9).mean().values
            dif_now, dea_now = dif[-1], dea[-1]

            if dif_now > dea_now and dif_now > 0:
                score, status = 10, "多头强势"
            elif dif_now > dea_now and dif_now <= 0:
                score, status = 7, "空头反弹"
            elif dif_now <= dea_now and dif_now > 0:
                score, status = 4, "多头回调"
            else:
                score, status = 0, "空头弱势"

            return {
                "score": score, "max_score": 10,
                "raw": {"DIF": round(dif_now, 2), "DEA": round(dea_now, 2)},
                "detail": (f"上证MACD: DIF={dif_now:.2f}, DEA={dea_now:.2f} ({status})"
                           f"{self._idx_note(df)}"),
            }
        except Exception as e:
            return {"unavailable": True, "score": 5, "max_score": 10, "raw": {}, "detail": f"获取失败: {e}"}

    def _score_cyb_relative_strength(self) -> dict:
        """创业板相对强度"""
        try:
            cyb = self._index("sz399006")
            hs300 = self._index("sh000300")
            cyb_ret = (cyb["close"].values[-1] / cyb["close"].values[-20] - 1) * 100
            hs300_ret = (hs300["close"].values[-1] / hs300["close"].values[-20] - 1) * 100
            relative = cyb_ret - hs300_ret

            if relative > 5:    score = 10
            elif relative > 0:  score = 7
            elif relative > -5: score = 4
            else:               score = 0

            return {
                "score": score, "max_score": 10, "raw": round(relative, 2),
                "detail": (f"创业板({cyb_ret:+.1f}%) vs 沪深300({hs300_ret:+.1f}%), "
                           f"相对强度: {relative:+.1f}%{self._idx_note(cyb)}"),
            }
        except Exception as e:
            return {"unavailable": True, "score": 5, "max_score": 10, "raw": 0, "detail": f"获取失败: {e}"}

    # ================================================================
    # 维度二：量能检测 (25分)
    # ================================================================

    def _score_volume_vs_ma20(self) -> dict:
        """两市成交量 vs 20日均量

        ⚠️ 单位口径（2026-09-28 修正）：`ak.stock_zh_index_daily` 的 `volume` 列是
        **成交量（股）**，不是成交额（元）。原实现写 `total/1e8` 并渲染成
        "两市成交 965亿" → 复盘报告里出现"两市成交965亿元""成交额回升至1000亿元以上"，
        而真实两市成交额约 1 万亿元 —— **小了一个数量级**，模型还据此继续演绎。
        此处统一改成"成交量 X 亿股"。
        """
        try:
            sh = self._index("sh000001")
            sz = self._index("sz399001")
            # 用 volume 列（单位：股）
            sh_vol = sh["volume"].values[-1]
            sz_vol = sz["volume"].values[-1]
            total = sh_vol + sz_vol
            sh_avg = np.mean(sh["volume"].values[-21:-1])
            sz_avg = np.mean(sz["volume"].values[-21:-1])
            avg_total = sh_avg + sz_avg
            ratio = (total / avg_total - 1) * 100 if avg_total > 0 else 0

            if ratio > 50:    score = 12
            elif ratio > 20:  score = 9
            elif ratio > -20: score = 6
            elif ratio > -50: score = 3
            else:             score = 0

            # 数据日期校验：指数日线在收盘当晚才补当日 bar，直接取 [-1] 很可能
            # 拿到的是**上一个交易日**的量 → 不校验就会把旧量当今日量，
            # 与实时指数涨跌幅混在一起解读（本项目「数据日期 vs 查询时间」铁律）。
            note = ""
            try:
                bar_date = str(sh["date"].values[-1])[:10]
                expected = last_trading_day()
                if bar_date < expected:
                    note = f"（该bar为{bar_date}，最近交易日{expected}，系上一交易日数据）"
            except Exception:                       # noqa: BLE001 - 日期校验失败不影响评分
                pass

            return {
                "score": score, "max_score": 12, "raw": round(ratio, 1),
                "detail": (f"两市成交量{total/1e8:.0f}亿股 "
                           f"(20日均量{avg_total/1e8:.0f}亿股), 偏离: {ratio:+.1f}%"
                           f"{note}{self._idx_note(sh)}"),
            }
        except Exception as e:
            return {"unavailable": True, "score": 6, "max_score": 12, "raw": 0, "detail": f"获取失败: {e}"}

    def _score_advance_decline(self) -> dict:
        """涨跌家数比 (0-8分)
        
        优化：使用 get_market_snapshot() 一次获取全市场行情替代 pywencai
        """
        try:
            df = get_market_snapshot()
            if df is not None and not df.empty:
                up = (df["涨跌幅"] > 0).sum()
                down = (df["涨跌幅"] < 0).sum()
                ratio = up / max(down, 1)

                if ratio > 3:       score = 8
                elif ratio > 2:     score = 6
                elif ratio > 1:     score = 5
                elif ratio > 0.5:   score = 3
                else:               score = 0

                return {
                    "score": score, "max_score": 8, "raw": round(ratio, 2),
                    "detail": f"上涨{up}家 vs 下跌{down}家, 比值: {ratio:.2f}",
                }
        except Exception as e:
            pass
        return {"unavailable": True, "score": 4, "max_score": 8, "raw": 0, "detail": "涨跌数据获取失败，使用默认值"}

    def _score_north_flow(self) -> dict:
        """北向资金净流入 —— 已停用（v2/M0 0.4）

        沪深港通自 2024-08-19 起停止披露日频净买入/成交金额，
        ak.stock_hsgt_hist_em 只能取到历史陈旧值或直接失败。
        该指标不再计分，并由 detect() 按"不可用"处理（不计入分母），
        以免"拿不到数据"被折算成虚假的 2/5 中性分。
        """
        return {
            "score": 0, "max_score": 0, "raw": 0, "unavailable": True,
            "detail": "北向资金日频数据自 2024-08-19 起停止披露，该指标已停用（不计分）",
        }

    # ================================================================
    # 维度三：情绪检测 (25分) —— 改用 AKShare 涨停板池替代 pywencai
    # ================================================================

    def _get_zt_pool_info(self) -> dict:
        """获取涨停板池信息（B2②：**本地面板优先**，与东财池口径差写进 caveat）。

        本地版基于面板最新交易日 T（收盘口径，含尾盘炸板）：
        涨停/跌停家数、最高连板、晋级率一次算齐；盘前 09:25 检测时
        东财"今日池"本来就是空的，本地版反而有 T-1 全量数据。
        本地不可用（在线且需要实时池）才回退 akshare。
        """
        stats = self._zt_stats()
        if stats.get("available"):
            return {
                "success": True,
                "limit_up_count": stats["limit_up_count"],
                "limit_down_count": stats.get("limit_down_count", 0),
                "max_consecutive": stats["max_consecutive"],
                "upgrade_rate": stats.get("upgrade_rate"),
                "source": "panel",
                "caveat": stats.get("caveat", ""),
                "trade_date": stats.get("trade_date"),
                "df": None,
            }
        try:
            today_str = datetime.now().strftime("%Y%m%d")
            df = ak.stock_zt_pool_em(date=today_str)
            if df is not None and not df.empty:
                limit_up_count = len(df)
                consecutive_col = None
                for col in ["连板数", "连板天数", "连续涨停天数"]:
                    if col in df.columns:
                        consecutive_col = col
                        break
                max_board = int(df[consecutive_col].max()) if consecutive_col else 1
                return {
                    "success": True,
                    "limit_up_count": limit_up_count,
                    "max_consecutive": max_board,
                    "source": "eastmoney",
                    "df": df,
                }
        except Exception:
            pass
        return {"success": False, "limit_up_count": 0, "max_consecutive": 0,
                "source": "none", "df": None}

    def _score_limit_up_down(self, zt_info: dict = None) -> dict:
        """涨停/跌停家数比 (0-8分)
        
        优化：用 ak.stock_zt_pool_em 获取涨停家数，跌停家数用全市场行情估算
        """
        try:
            if zt_info is None:
                zt_info = self._get_zt_pool_info()

            limit_up = zt_info.get("limit_up_count", 0)
            # 跌停家数：本地版直接给（面板 T 日跌停判定）；东财源才用快照估算
            if zt_info.get("source") == "panel":
                limit_down = int(zt_info.get("limit_down_count", 0))
            else:
                try:
                    spot = get_market_snapshot()
                    limit_down = (spot["涨跌幅"] <= -9.9).sum()
                except Exception:
                    limit_down = max(1, limit_up // 8)  # 粗略估算
            _src = "本地面板" if zt_info.get("source") == "panel" else "东财实时池"

            if limit_down == 0:
                ratio = float('inf') if limit_up > 0 else 0
            else:
                ratio = limit_up / limit_down

            if ratio == float('inf') or ratio > 10: score = 8
            elif ratio > 5:   score = 6
            elif ratio > 2:   score = 5
            elif ratio > 1:   score = 3
            else:             score = 0

            return {
                "score": score, "max_score": 8, "raw": f"{limit_up}/{limit_down}",
                "detail": f"涨停{limit_up}家 vs 跌停{limit_down}家（{_src}）",
            }
        except Exception:
            return {"unavailable": True, "score": 4, "max_score": 8, "raw": "N/A", "detail": "涨跌停数据获取失败"}

    def _score_max_consecutive_board(self, zt_info: dict = None) -> dict:
        """连板高度 (0-5分)
        
        优化：直接从涨停板池提取连板数
        """
        try:
            if zt_info is None:
                zt_info = self._get_zt_pool_info()

            max_days = zt_info.get("max_consecutive", 0)

            if max_days >= 7:   score = 5
            elif max_days >= 5: score = 4
            elif max_days >= 3: score = 3
            elif max_days >= 1: score = 2
            else:               score = 0

            return {
                "score": score, "max_score": 5, "raw": max_days,
                "detail": f"最高连板: {max_days}板",
            }
        except Exception:
            return {"unavailable": True, "score": 2, "max_score": 5, "raw": 0, "detail": "连板数据获取失败"}

    def _score_upgrade_rate(self, zt_info: dict = None) -> dict:
        """涨停板晋级率 (0-5分)
        
        优化：利用涨停板池中的连板数列判断晋级率
        方案：将今日涨停数中连板数>=2的视为"晋级"
        """
        try:
            if zt_info is None:
                zt_info = self._get_zt_pool_info()

            if zt_info.get("source") == "panel" and zt_info.get("upgrade_rate") is not None:
                rate = float(zt_info["upgrade_rate"])
                if rate > 30:       score = 5
                elif rate > 20:     score = 4
                elif rate > 10:     score = 3
                elif rate > 5:      score = 1
                else:               score = 0
                return {
                    "score": score, "max_score": 5, "raw": rate,
                    "detail": (f"晋级率: {rate:.0f}%（本地面板 "
                               f"{zt_info.get('trade_date')}，连板≥2 占比）"),
                }
            df = zt_info.get("df")
            if df is not None and not df.empty:
                total = len(df)
                consecutive_col = None
                for col in ["连板数", "连板天数", "连续涨停天数"]:
                    if col in df.columns:
                        consecutive_col = col
                        break

                if consecutive_col and total > 0:
                    upgraded = (df[consecutive_col] >= 2).sum()
                    rate = upgraded / total * 100
                    
                    if rate > 30:       score = 5
                    elif rate > 20:     score = 4
                    elif rate > 10:     score = 3
                    elif rate > 5:      score = 1
                    else:               score = 0

                    return {
                        "score": score, "max_score": 5, "raw": round(rate, 1),
                        "detail": f"晋级率: {upgraded}/{total} ({rate:.0f}%)",
                    }
        except Exception:
            pass
        return {"unavailable": True, "score": 2, "max_score": 5, "raw": 0, "detail": "晋级率数据获取失败"}

    def _score_market_breadth(self) -> dict:
        """市场宽度：站上20日均线的个股占比 (0-7分)

        B2②（2026-10-19）：**本地面板优先** —— `zt_calendar` 一次性算全市场
        close vs MA20（5531 只，不再抽样 12 只 × 3 源重试 ⇒ 消掉 ~15 次外呼）。
        本地不可用时才回退原抽样逻辑（在线）。
        """
        try:
            stats = self._zt_stats()
            if stats.get("available") and stats.get("above_ma20_ratio") is not None:
                ratio = float(stats["above_ma20_ratio"])
                if ratio > 70:       score = 7
                elif ratio > 50:     score = 5
                elif ratio > 30:     score = 3
                else:                score = 0
                return {
                    "score": score, "max_score": 7, "raw": ratio,
                    "detail": (f"全市场{stats['above_ma20_checked']}只中站上MA20 "
                               f"{ratio:.0f}%（本地面板 {stats['trade_date']}）"),
                }

            # ── 本地不可用 ⇒ 原抽样逻辑（在线）──
            spot = get_market_snapshot()
            if spot is None or spot.empty:
                return {"unavailable": True, "score": 3, "max_score": 7, "raw": 0, "detail": "行情数据获取失败"}

            codes = spot["代码"].head(min(60, len(spot))).tolist()

            above_ma20 = 0
            checked = 0
            for code in codes[:12]:  # 12只样本（~3秒），降低API压力
                try:
                    df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
                    if len(df) >= 21:
                        ma20 = df["收盘"].tail(20).mean()
                        if df["收盘"].iloc[-1] > ma20:
                            above_ma20 += 1
                        checked += 1
                except Exception as e:                      # noqa: BLE001
                    _logger.debug("[detector] 宽度抽样 %s 失败（跳过）: %s", code, e)
                    continue

            ratio = above_ma20 / max(checked, 1) * 100

            if ratio > 70:       score = 7
            elif ratio > 50:     score = 5
            elif ratio > 30:     score = 3
            else:                score = 0

            return {
                "score": score, "max_score": 7, "raw": round(ratio, 1),
                "detail": f"样本{checked}只中{above_ma20}只站上MA20 ({ratio:.0f}%)",
            }
        except Exception as e:
            return {"unavailable": True, "score": 3, "max_score": 7, "raw": 0, "detail": f"市场宽度获取失败: {e}"}

    def _score_ivix(self) -> dict:
        """iVIX恐慌指数 / 年化波动率降级 (0-5分)

        B2②：离线模式直接走 hs300 本地波动率降级（不发起 sse 外呼）。
        """
        try:
            from data import offline_gate as _gate
            if not _gate.is_offline():
                df = ak.option_risk_indicator_sse()
            else:
                raise RuntimeError("离线模式：iVIX 未本地化，走波动率降级")
            ivix = df.iloc[-1]["ivix"]
            if ivix < 18:       score = 5
            elif ivix < 22:     score = 4
            elif ivix < 26:     score = 2
            elif ivix < 30:     score = 1
            else:               score = 0
            return {
                "score": score, "max_score": 5, "raw": round(float(ivix), 1),
                "detail": f"iVIX: {ivix:.1f}",
            }
        except Exception:
            # iVIX数据经常不可用，用年化波动率替代
            try:
                hs300 = self._index("sh000300")
                returns = hs300["close"].pct_change().dropna().tail(20)
                vol = returns.std() * np.sqrt(252) * 100
                if vol < 15:      score = 5
                elif vol < 20:    score = 4
                elif vol < 25:    score = 2
                else:             score = 0
                return {
                    "score": score, "max_score": 5, "raw": round(vol, 1),
                    "detail": f"年化波动率: {vol:.1f}% (替代iVIX)",
                }
            except Exception:
                return {"unavailable": True, "score": 3, "max_score": 5, "raw": 0, "detail": "iVIX和波动率均获取失败"}

    # ================================================================
    # 维度四：结构检测 (20分)
    # ================================================================

    def _score_sector_rotation(self) -> dict:
        """行业轮动速度 (0-5分)

        v2（2026-09-25）：**优先读盘后落库的行业资金流快照**。
        原实现每轮都调东财 `stock_sector_fund_flow_rank` —— 它在本环境不可达，
        而且市场检测**每次选股都会跑一次**，等于每轮白等一次连接超时。
        落库快照（同花顺口径，90 个行业净额）能直接算同一个离散度指标。
        """
        changes = None
        detail_src = ""
        try:
            from data.fund_flow import get_stored_industry_flow
            date, df = get_stored_industry_flow()
            if df is not None and not df.empty and "净额" in df.columns:
                changes = pd.to_numeric(df["净额"], errors="coerce").dropna()
                detail_src = f"（快照 {date}）"
        except Exception:                            # noqa: BLE001
            changes = None
        if changes is None or len(changes) <= 3:
            try:
                df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
                if df is not None and not df.empty:
                    changes = pd.to_numeric(df["主力净流入-净额"], errors="coerce").dropna()
                    detail_src = "（东财实时）"
            except Exception:
                changes = None
        if changes is not None and len(changes) > 3:
            mean_abs = changes.abs().mean()
            dispersion = changes.std() / mean_abs if mean_abs > 0 else 0
            if dispersion < 1:    score = 5
            elif dispersion < 1.5: score = 3
            else:                 score = 1
            return {
                "score": score, "max_score": 5, "raw": round(float(dispersion), 2),
                "detail": f"行业资金离散度: {dispersion:.2f}{detail_src}",
            }
        return {"unavailable": True, "score": 2, "max_score": 5, "raw": 0, "detail": "行业轮动数据获取失败"}

    def _score_size_style(self) -> dict:
        """大小盘风格偏向 (0-5分)"""
        try:
            hs300 = self._index("sh000300")
            zz1000 = self._index("sh000852")
            hs300_ret = (hs300["close"].values[-1] / hs300["close"].values[-5] - 1) * 100
            zz1000_ret = (zz1000["close"].values[-1] / zz1000["close"].values[-5] - 1) * 100
            diff = hs300_ret - zz1000_ret

            if abs(diff) > 3:       score = 5
            elif abs(diff) > 1.5:   score = 4
            elif abs(diff) > 0.5:   score = 2
            else:                   score = 0

            bias = "大盘" if diff > 0 else "小盘"
            return {
                "score": score, "max_score": 5, "raw": round(diff, 2),
                "detail": f"沪深300({hs300_ret:+.1f}%) vs 中证1000({zz1000_ret:+.1f}%), {bias}占优",
            }
        except Exception as e:
            return {"unavailable": True, "score": 2, "max_score": 5, "raw": 0, "detail": f"大小盘风格获取失败: {e}"}

    def _score_margin_balance(self) -> dict:
        """融资余额变化 (0-5分)

        v3 修正（2026-09-25 实测）。原实现有三处错误，**互相掩盖成一个"恒满分"的假指标**：

          1. 查询日用 `datetime.now()` —— 上交所融资数据**收盘后才发布**，
             盘中/当日查询必然取不到（表面只是"指标失败"，看不出是口径错）；
          2. 用 `stock_margin_detail_sse`（**逐标的明细**，实测 2000 行）却取
             `df["融资余额"].iloc[-1]` —— 那是**某一只标的**的余额，不是全市场
             （实测取到 168,474,612，即某只 ETF 的余额）；
          3. 该字段单位是**元**，而判分阈值是按"亿"设计的（>50 得 5 分）
             → 以元计恒 > 50 → **无条件得满分**，5% 权重白送，还让总分虚高。

        正确口径：`stock_margin_sse`（交易所**汇总**按日）取最近两个交易日的
        `融资余额` 之差，÷1e8 换算成亿元（实测全市场量级约 7280 亿，
        两日变化落在数十亿区间，与阈值设计一致）。

        B2②（2026-10-19）：融资余额无本地数据 ⇒ 离线模式**直接不可评**
        （不发起外呼；原实现离线时会先撞 2 次 sse 超时才失败）。
        """
        try:
            from data import offline_gate as _gate
            if _gate.is_offline():
                return {"unavailable": True, "score": 2, "max_score": 5, "raw": 0,
                        "detail": "离线模式：融资余额无本地数据（未本地化）"}
        except ImportError:
            pass
        try:
            from data import trading_calendar as _cal
            end = _cal.last_trading_day().replace("-", "")
            # 取 12 个自然日窗口，足够覆盖 ≥2 个交易日；不依赖 trading_days() 的区间参数
            start = (datetime.strptime(end, "%Y%m%d")
                     - timedelta(days=12)).strftime("%Y%m%d")
            df = ak.stock_margin_sse(start_date=start, end_date=end)
            if df is None or df.empty:
                raise ValueError("上交所融资汇总为空")
            df = df.dropna(subset=["融资余额"]).sort_values("信用交易日期")
            if len(df) < 2:
                raise ValueError(f"不足两个交易日（实得 {len(df)} 行）")
            vals = pd.to_numeric(df["融资余额"], errors="coerce")
            change_yi = (vals.iloc[-1] - vals.iloc[-2]) / 1e8

            if change_yi > 50:       score = 5
            elif change_yi > 20:     score = 4
            elif change_yi > 0:      score = 3
            elif change_yi > -20:    score = 1
            else:                    score = 0

            return {
                "score": score, "max_score": 5, "raw": round(float(change_yi), 1),
                "detail": f"融资余额变化: {change_yi:+.1f}亿"
                          f"（{df['信用交易日期'].iloc[-2]}→{df['信用交易日期'].iloc[-1]}）",
            }
        except Exception as e:                       # noqa: BLE001
            # 取不到就标 unavailable：detect() 会把它排除出「计分」与「分母」。
            # 保留保守默认分只为 UI 展示，绝不能让默认分冒充真实得分。
            return {"unavailable": True, "score": 2, "max_score": 5, "raw": 0,
                    "detail": f"融资余额数据获取失败: {str(e)[:40]}"}

    # ================================================================
    # 主检测方法
    # ================================================================

    def detect(self) -> MarketSignal:
        """执行完整市场检测，返回 MarketSignal。

        14个指标按权重计算总分，映射到5级市场状态。
        线程池并发获取加速（8 workers，避免AKShare限流）。
        """
        # 先获取涨停板池信息（后续3个指标复用）
        zt_info = self._get_zt_pool_info()

        # 定义所有评分函数
        score_funcs = {
            "hs300_ma20_deviation": self._score_hs300_ma20,
            "sh_macd": self._score_sh_macd,
            "cyb_relative_strength": self._score_cyb_relative_strength,
            "volume_vs_ma20": self._score_volume_vs_ma20,
            "advance_decline_ratio": self._score_advance_decline,
            "north_flow": self._score_north_flow,
            "limit_up_down_ratio": lambda zt=zt_info: self._score_limit_up_down(zt),
            "max_consecutive_board": lambda zt=zt_info: self._score_max_consecutive_board(zt),
            "market_breadth": self._score_market_breadth,
            "ivix_level": self._score_ivix,
            "sector_rotation_speed": self._score_sector_rotation,
            "size_style_bias": self._score_size_style,
            "margin_balance_change": self._score_margin_balance,
            "limit_up_upgrade_rate": lambda zt=zt_info: self._score_upgrade_rate(zt),
        }

        results = {}
        for key, func in score_funcs.items():
            try:
                results[key] = func()
            except Exception as e:
                results[key] = {"score": 0, "max_score": 0, "raw": 0,
                                "unavailable": True, "detail": str(e)}

        # 计算加权总分
        # v2/M0 0.4：不可用（数据停更/取数失败）的指标既不计分也不计入分母，
        # 末尾按有效权重重新归一化到 0-100。原实现会把取数失败当 0 分计入总分，
        # 相当于"行情接口挂了"被读成"市场看空"，并让总分随数据源可用性漂移。
        total_score = 0.0
        available_weight = 0.0
        score_details = {}
        risk_warnings = []

        for key, weight in self.weights.items():
            result = results.get(key, {"score": 0, "max_score": 0})
            raw_score = result.get("score", 0)
            max_score = result.get("max_score", 0)

            if result.get("unavailable") or max_score <= 0:
                score_details[key] = {
                    **result,
                    "weight": weight,
                    "weighted_score": 0.0,
                }
                continue

            available_weight += weight
            normalized = (raw_score / max_score) * weight * 100
            total_score += normalized

            score_details[key] = {
                **result,
                "weight": weight,
                "weighted_score": round(normalized, 1),
            }

            if raw_score == 0:
                risk_warnings.append(f"{key}: {result.get('detail', f'得分0/满分{max_score}')}")

        if available_weight > 0:
            # 加权项已含 weight×100，除以有效权重即可拉回 0-100 标度
            total_score = total_score / available_weight

        total_score = round(min(total_score, 100), 1)

        # ── 覆盖率闸门（2026-10-18，P0-1 断网验收暴露）──
        # ⚠️⚠️ 事故：断网跑选股，14 个指标多数"不可评"，只剩少数低分指标
        #    参与归一化 ⇒ 总分 18.2 ⇒ 判"熊市"⇒ `position_limit=0`
        #    ⇒ **自动跳过选股**。用户看到的是"市场极度弱势"，实际是"数据没取到"。
        #    这正是本项目铁律「不可评 ≠ 0 分」的翻版：不可评既不计分也不计入分母
        #    之后，**分母本身也可能失真**（只剩 2/14 个指标 ⇒ 结论没有代表性）。
        # 处理：可评权重不足 50% ⇒ 判定"数据不足"，按**中性震荡**给保守仓位上限
        #    （不阻断选股，只提示），并在 warnings 里写明缺了多少 —— 只提示不阻断。
        _nominal = sum(float(x) for x in self.weights.values()) or 1.0
        _coverage = available_weight / _nominal
        _n_avail = sum(1 for r in results.values() if not r.get("unavailable"))
        _n_total = len(results)
        if _coverage < 0.5:
            state, signal, limit, bias, advice = (
                "数据不足（中性假设）", "观望/轻仓", 0.40, "防御型",
                f"14 指标仅 {_n_avail}/{_n_total} 个可评（有效权重 {_coverage:.0%}）"
                "⇒ 不足以判定牛熊，按中性震荡处理、**不阻断选股**；请检查数据源。"
            )
            risk_warnings.append(
                f"市场检测覆盖率 {_coverage:.0%} < 50%（{_n_avail}/{_n_total} 指标可评）"
                "：市场结论降级为中性，**不代表市场真实状态**")
        # 映射到市场状态
        elif total_score >= 80:
            state, signal, limit, bias, advice = (
                "牛市", "强烈买入", 0.80, "进攻型",
                "市场环境极佳，积极做多。选股偏向科技、创业板、高Beta。"
            )
        elif total_score >= 60:
            state, signal, limit, bias, advice = (
                "偏强震荡", "买入", 0.60, "均衡型",
                "市场偏强，正常操作。选股偏向行业龙头+低估值成长。"
            )
        elif total_score >= 40:
            state, signal, limit, bias, advice = (
                "中性震荡", "观望/轻仓", 0.40, "防御型",
                "市场方向不明，降低仓位。选股偏向高股息、消费、金融。"
            )
        elif total_score >= 20:
            state, signal, limit, bias, advice = (
                "偏弱震荡", "卖出/减仓", 0.20, "极端防御",
                "市场偏弱，大幅减仓。仅保留核心仓位，优先现金管理。"
            )
        else:
            state, signal, limit, bias, advice = (
                "熊市", "强烈卖出", 0.00, "清仓",
                "市场极度弱势，建议空仓观望。系统自动跳过选股阶段。"
            )

        return MarketSignal(
            total_score=total_score,
            market_state=state,
            signal=signal,
            position_limit=limit,
            strategy_bias=bias,
            operation_advice=advice,
            scores=score_details,
            indicators={k: v.get("raw", "N/A") for k, v in results.items()},
            risk_warnings=risk_warnings,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )