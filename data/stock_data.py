import akshare as ak
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import requests
import json
import pywencai
import time
from data.source_manager import data_source_manager

# 应用 akshare 请求补丁（确保请求头/超时/重试）
from data.akshare_helper import patch_requests, retry_on_failure
import logging

_logger = logging.getLogger(__name__)
patch_requests()

# YANKED

# 东财个股详细信息接口熔断器（东财域名整体不可达时避免逐股重试白等）
_EM_INFO_COOLDOWN = 600  # 冷却秒数
_em_info_circuit_open_until = 0.0


def _period_key(value) -> str:
    """把各种"报告期"写法归一成可比较的 8 位字符串（YYYYMMDD）。

    各源写法不一致：新浪 `20260630`、同花顺 `1998`（年度）/`1998-12-31`。
    归一后才能**按时间排序**，而不是靠猜"第一行是不是最新的"。
    """
    s = str(value).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    if len(digits) == 6:                      # YYYYMM → 当月最后一天近似
        return f"{digits}31"
    if len(digits) == 4:                      # 年度
        return f"{digits}1231"
    return digits or "00000000"


def _latest_period_records(df, n: int = 8) -> list:
    """取**最近 n 个报告期**的记录（按报告期排序，不依赖源的行序）。

    为什么必须要这个（2026-09-25 实测的真实故障）：不同源的**行序方向相反**——
      - 新浪 `stock_financial_report_sina`：最新在前（倒序）
      - 同花顺 `stock_financial_abstract_ths`：最老在前（正序，1998→2025）
    原实现对同花顺的结果直接 `head(8)`，取到的是**上市以来最早的 8 期**
    （实测 600519 拿到 1998~2005），而深度分析却把它当作"最新财报"。
    这里统一排序后再取，源怎么排都不影响结果。
    """
    if df is None or getattr(df, "empty", True):
        return []
    period_col = df.columns[0]
    idx = sorted(df.index, key=lambda i: _period_key(df.loc[i, period_col]),
                 reverse=True)[:n]
    return df.loc[idx].to_dict("records")


class StockDataFetcher:
    """股票数据获取类"""
    
    def __init__(self):
        self.data = None
        self.info = None
        self.financial_data = None
        self.data_source_manager = data_source_manager
        
    def get_stock_info(self, symbol, skip_em: bool = False,
                       skip_valuation: bool = False):
        """获取股票基本信息。

        Args:
            skip_em: True 时跳过东方财富那条路径（akshare），只走"新浪（名称）
                     + 百度（PE/PB TTM）"——两者都是纯 requests。
                    **并发预取必须传 True**：akshare 里的 py_mini_racer（V8）在 Windows
                    不支持多线程（见 core/market_detector.py 的历史注释），
                    而 skip_em 把 akshare 完全排除在并发之外。
                    串行路径保持默认 False（东财可达时仍取东财数据，口径不丢）。
            skip_valuation: True 时**连百度的 PE/PB 也不取**，只留名称等基础信息。
                    ⚠️ 为什么需要（2026-10-22 实测事故）：百度的
                    `stock_zh_valuation_baidu` 是**逐票**接口，选股要一次取 100+ 只，
                    并发打过去会被限流并返回 HTML，抛 `Expecting value: line 1 column 1`
                    —— 于是每只都失败重试，**选股从 6 分钟变成 43 分钟**。
                    而 `market_snapshot` 里已经有全市场 PE（覆盖 65.5%，缺的都是亏损股）/
                    PB（91.0%），评分侧 `_snapshot_pe()` 读一次 SQL 就能拿到。
                    ⇒ **选股路径统一传 True + 走快照**；单只分析（agents/agent_tools/
                    portfolio）保持默认 False，那里一两只不会触发限流。
        """
        try:
            # 处理中国A股
            if self._is_chinese_stock(symbol):
                return self._get_chinese_stock_info(symbol, skip_em=skip_em,
                                                    skip_valuation=skip_valuation)
            # 处理港股
            elif self._is_hk_stock(symbol):
                return self._get_hk_stock_info(symbol)
            # 处理美股
            else:
                return self._get_us_stock_info(symbol)
        except Exception as e:
            return {"error": f"获取股票信息失败: {str(e)}"}
    
    def get_stock_data(self, symbol, period="1y", interval="1d"):
        """获取股票历史数据"""
        try:
            if self._is_chinese_stock(symbol):
                return self._get_chinese_stock_data(symbol, period)
            elif self._is_hk_stock(symbol):
                return self._get_hk_stock_data(symbol, period)
            else:
                return self._get_us_stock_data(symbol, period, interval)
        except Exception as e:
            return {"error": f"获取股票数据失败: {str(e)}"}
    
    def _is_chinese_stock(self, symbol):
        """判断是否为中国A股"""
        # 简单判断：包含数字且长度为6位的认为是中国A股
        return symbol.isdigit() and len(symbol) == 6
    
    def _is_hk_stock(self, symbol):
        """判断是否为港股"""
        # 港股代码通常是1-5位数字，或者前面带HK/hk前缀
        if symbol.upper().startswith('HK'):
            return True
        # 纯数字且长度在1-5位之间，认为可能是港股
        if symbol.isdigit() and 1 <= len(symbol) <= 5:
            return True
        return False
    
    def _normalize_hk_code(self, symbol):
        """规范化港股代码为5位格式（如700 -> 00700）"""
        # 移除HK前缀
        if symbol.upper().startswith('HK'):
            symbol = symbol[2:]
        # 补齐到5位
        return symbol.zfill(5)
    
    def _get_chinese_stock_info(self, symbol, skip_em: bool = False,
                                skip_valuation: bool = False):
        """获取中国股票基本信息（支持akshare和tushare数据源自动切换）

        `skip_em=True`：跳过所有走 akshare 的分支（basic_info 的东财段、
        东财个股详情重试、tushare），只保留 新浪 + 百度 两条纯 requests 路径。
        供并发预取使用（见 `data/prefetch.py` 的说明）。
        """
        global _em_info_circuit_open_until
        try:
            # 初始化基本信息
            info = {
                "symbol": symbol,
                "name": "未知",
                "current_price": "N/A",
                "change_percent": "N/A",
                "pe_ratio": "N/A",
                "pb_ratio": "N/A",
                "market_cap": "N/A",
                "market": "中国A股",
                "exchange": "上海/深圳证券交易所"
            }
            
            # 先尝试使用数据源管理器获取基本信息。
            # ⚠️ 熔断打开时**跳过**它：该方法内部同样先打东财（失败 0.6s）再降级新浪，
            #    而熔断期意味着"东财刚被证明不可达"，再试一次只是重复浪费时间
            #    （实测每次 1.88s，其中东财部分纯浪费）。
            if not skip_em and time.time() >= _em_info_circuit_open_until:
                basic_info = self.data_source_manager.get_stock_basic_info(symbol)
                if basic_info:
                    info.update(basic_info)
            
            # 方法1: 尝试获取个股详细信息（akshare，带重试 + 东财熔断器）
            # 东财接口连续失败后进入冷却期（模块级），避免逐股白等重试
            em_attempted = False
            em_success = False
            for retry_count in range(3):
                if skip_em or time.time() < _em_info_circuit_open_until:
                    continue
                em_attempted = True
                try:
                    stock_info = ak.stock_individual_info_em(symbol=symbol)
                    if stock_info is not None and not stock_info.empty:
                        for _, row in stock_info.iterrows():
                            key = row['item']
                            value = row['value']

                            if key == '股票简称':
                                info['name'] = value
                            elif key == '总市值':
                                try:
                                    if value and value != '-':
                                        info['market_cap'] = float(value)
                                except:
                                    pass
                            elif key == '市盈率-动态':
                                try:
                                    if value and value != '-':
                                        pe_value = float(value)
                                        if 0 < pe_value <= 1000:
                                            info['pe_ratio'] = pe_value
                                except:
                                    pass
                            elif key == '市净率':
                                try:
                                    if value and value != '-':
                                        pb_value = float(value)
                                        if 0 < pb_value <= 100:
                                            info['pb_ratio'] = pb_value
                                except:
                                    pass
                        em_success = True
                        break  # 成功则跳出重试循环
                except Exception as e:
                    # ⚠️ 2026-10-09 实测：东财在本机/本网环境**完全不可达**（ProxyError，
                    #    0.6s 快速失败），而原退避是 (retry+1)*2 = 2s/4s —— 单只白等 6s，
                    #    是选股单只耗时 7.7s 的主要来源（K 线只要 0.81s）。
                    #    退避缩短到 0.5s/1s：这类失败是**网络不可达**而非服务端限流，
                    #    长退避等不来成功，只会拖慢整条链路。
                    _logger.warning(f"[Akshare] 获取个股详细信息失败: {e}")
                    if retry_count < 2:
                        delay = 0.5 * (retry_count + 1)
                        _logger.info(f"[Akshare] ⏳ {delay}s 后重试...")
                        time.sleep(delay)
            if em_attempted and not em_success:
                # 本轮全部失败 → 打开熔断（冷却期内调用不续期，保证可自愈探测）
                _em_info_circuit_open_until = time.time() + _EM_INFO_COOLDOWN
            if not em_success:
                # 东财不可用（被熔断或重试全败），尝试新浪数据源获取基本信息
                _logger.info(f"[Akshare-新浪] 尝试从新浪获取基本信息...")
                try:
                    import requests as req_sina
                    tx_code = self.data_source_manager._convert_to_tx_code(symbol)
                    url = f'https://hq.sinajs.cn/list={tx_code}'
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Referer': 'https://finance.sina.com.cn/',
                    }
                    r = req_sina.get(url, headers=headers, timeout=10)
                    if r.status_code == 200 and f'hq_str_{tx_code}' in r.text:
                        start = r.text.find('"') + 1
                        end = r.text.find('"', start)
                        if start > 0 and end > start:
                            fields = r.text[start:end].split(',')
                            if len(fields) >= 1 and fields[0]:
                                if info['name'] == '未知':
                                    info['name'] = fields[0]
                                if len(fields) >= 4 and fields[3]:
                                    info['current_price'] = fields[3]
                                if len(fields) >= 2 and fields[1] and fields[2]:
                                    try:
                                        current = float(fields[3]) if fields[3] else 0
                                        prev_close = float(fields[2]) if fields[2] else 0
                                        if prev_close > 0:
                                            change = ((current - prev_close) / prev_close) * 100
                                            info['change_percent'] = round(change, 2)
                                    except:
                                        pass
                                _logger.info(f"[Akshare-新浪] ✅ 成功获取基本信息: {info['name']}")
                except Exception as e2:
                    _logger.warning(f"[Akshare-新浪] ❌ 获取失败: {e2}")
                
                # 如果新浪也失败，尝试从tushare获取
                # ⚠️ 2026-10-26：加能力表闸门。本账号 120 积分档**无权调 daily_basic**
                #    （logs 全量 547 次失败 / 0 次成功，服务端轮换用"1次/小时""1次/分钟"
                #    "5次/天"三种互相矛盾的话术软拒绝）。不加闸门的话，每只取不到基本信息的
                #    票都会先付一次注定失败的往返。
                _ts_daily_basic_ok = True
                try:
                    from data import tushare_source as _ts
                    _ts_daily_basic_ok = not _ts.unavailable_reason("daily_basic")
                except Exception as _e:                          # noqa: BLE001
                    _logger.debug("[Tushare] 能力表不可读，按可用处理: %s: %s",
                                  type(_e).__name__, _e)
                if (not skip_em and self.data_source_manager.tushare_available
                        and info['name'] == '未知' and _ts_daily_basic_ok):
                    _logger.info(f"[Tushare] 尝试获取基本信息（tushare）...")
                    try:
                        ts_code = self.data_source_manager._convert_to_ts_code(symbol)
                        df = self.data_source_manager.tushare_api.daily_basic(
                            ts_code=ts_code,
                            trade_date=datetime.now().strftime('%Y%m%d')
                        )
                        if df is not None and not df.empty:
                            row = df.iloc[0]
                            info['pe_ratio'] = row.get('pe', 'N/A')
                            info['pb_ratio'] = row.get('pb', 'N/A')
                            info['market_cap'] = row.get('total_mv', 'N/A')
                            _logger.info(f"[Tushare] ✅ 成功获取部分信息")
                    except Exception as te:
                        _logger.warning(f"[Tushare] ❌ 获取失败: {te}")
            
            # 方法2: 尝试获取历史价格和涨跌幅（如果网络允许）
            # try:
            #     # 使用更简单的接口获取实时价格
            #     real_time_data = ak.stock_zh_a_spot_em()
            #     if real_time_data is not None and not real_time_data.empty:
            #         stock_real_time = real_time_data[real_time_data['代码'] == symbol]
            #         if not stock_real_time.empty:
            #             row = stock_real_time.iloc[0]
            #             info['current_price'] = row.get('最新价', 'N/A')
            #             info['change_percent'] = row.get('涨跌幅', 'N/A')
            #             if info['name'] == '未知':
            #                 info['name'] = row.get('名称', '未知')
                        
            #             # 如果实时数据中有市盈率和市净率，优先使用
            #             if '市盈率-动态' in row and info['pe_ratio'] == 'N/A':
            #                 try:
            #                     pe_val = row['市盈率-动态']
            #                     if pe_val and pe_val != '-':
            #                         pe_val = float(pe_val)
            #                         if 0 < pe_val <= 1000:
            #                             info['pe_ratio'] = pe_val
            #                 except:
            #                     pass
                        
            #             if '市净率' in row and info['pb_ratio'] == 'N/A':
            #                 try:
            #                     pb_val = row['市净率']
            #                     if pb_val and pb_val != '-':
            #                         pb_val = float(pb_val)
            #                         if 0 < pb_val <= 100:
            #                             info['pb_ratio'] = pb_val
            #                 except:
            #                     pass
                                
            # except Exception as e:
            #     print(f"[Akshare] 获取实时数据失败: {e}")
            #     # 如果实时数据获取失败，尝试使用数据源管理器获取历史数据（支持tushare备用）
            try:
                _logger.info(f"[数据源管理器] 尝试获取最近交易数据...")
                hist_data = self.data_source_manager.get_stock_hist_data(
                    symbol=symbol,
                    start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                    end_date=datetime.now().strftime('%Y%m%d'),
                    adjust='qfq'
                )
                
                if hist_data is not None and not hist_data.empty:
                    # 标准化列名
                    if 'close' in hist_data.columns:
                        latest = hist_data.iloc[-1]
                        info['current_price'] = latest['close']
                        # 计算涨跌幅
                        if len(hist_data) > 1:
                            prev_close = hist_data.iloc[-2]['close']
                            change_pct = ((latest['close'] - prev_close) / prev_close) * 100
                            info['change_percent'] = round(change_pct, 2)
                        _logger.info(f"[数据源管理器] ✅ 成功获取价格数据")
            except Exception as e2:
                _logger.warning(f"获取历史数据也失败: {e2}")
            
            # 方法3: 使用百度估值数据获取市盈率和市净率
            # ⚠️ PE 的真实来源（2026-10-09 查明）：东财 `stock_individual_info_em`
            #    的字段映射里**没有市盈率**（akshare 现行版本只有代码/简称/股本/行业/
            #    市值/上市时间），东财不可达时这条路径必然拿不到 PE；
            #    实际生效的是下面这条 **百度股市通「市盈率(TTM)」**。
            #    ⇒ 评分里的 PE = 百度 TTM 口径（既不是东财、也不是快照的新浪 per）。
            if info['pe_ratio'] == 'N/A' and not skip_valuation:
                try:
                    pe_data = ak.stock_zh_valuation_baidu(symbol=symbol, indicator="市盈率(TTM)")
                    if pe_data is not None and not pe_data.empty:
                        latest_pe = pe_data.iloc[-1]['value']
                        if latest_pe and latest_pe != '-':
                            pe_val = float(latest_pe)
                            if 0 < pe_val <= 1000:
                                info['pe_ratio'] = pe_val
                except Exception as e:
                    _logger.warning(f"获取市盈率失败: {e}")
            
            if info['pb_ratio'] == 'N/A' and not skip_valuation:
                try:
                    pb_data = ak.stock_zh_valuation_baidu(symbol=symbol, indicator="市净率")
                    if pb_data is not None and not pb_data.empty:
                        latest_pb = pb_data.iloc[-1]['value']
                        if latest_pb and latest_pb != '-':
                            pb_val = float(latest_pb)
                            if 0 < pb_val <= 100:
                                info['pb_ratio'] = pb_val
                except Exception as e:
                    _logger.warning(f"获取市净率失败: {e}")
            
            return info
            
        except Exception as e:
            _logger.warning(f"获取中国股票信息完全失败: {e}")
            # 返回基本信息，避免完全失败
            return {
                "symbol": symbol,
                "name": f"股票{symbol}",
                "current_price": "N/A",
                "change_percent": "N/A",
                "pe_ratio": "N/A",
                "pb_ratio": "N/A",
                "market_cap": "N/A",
                "market": "中国A股",
                "exchange": "上海/深圳证券交易所"
            }
    
    def _get_hk_stock_info(self, symbol):
        """获取港股基本信息"""
        try:
            # 规范化港股代码
            hk_code = self._normalize_hk_code(symbol)
            
            # 初始化基本信息
            info = {
                "symbol": hk_code,
                "name": "未知",
                "current_price": "N/A",
                "change_percent": "N/A",
                "pe_ratio": "N/A",
                "pb_ratio": "N/A",
                "market_cap": "N/A",
                "market": "香港股市",
                "exchange": "香港交易所"
            }
            
            # 方法1: 获取港股实时行情
            try:
                # 使用akshare获取港股实时数据
                realtime_df = ak.stock_hk_spot_em()
                if realtime_df is not None and not realtime_df.empty:
                    # 查找对应股票
                    stock_data = realtime_df[realtime_df['代码'] == hk_code]
                    if not stock_data.empty:
                        row = stock_data.iloc[0]
                        info['name'] = row.get('名称', '未知')
                        info['current_price'] = row.get('最新价', 'N/A')
                        info['change_percent'] = row.get('涨跌幅', 'N/A')
                        
                        # 市值（港元）
                        market_cap = row.get('总市值', 'N/A')
                        if market_cap != 'N/A':
                            try:
                                info['market_cap'] = float(market_cap)
                            except:
                                pass
                        
                        # 市盈率
                        pe = row.get('市盈率', 'N/A')
                        if pe != 'N/A' and pe != '-':
                            try:
                                pe_val = float(pe)
                                if 0 < pe_val <= 1000:
                                    info['pe_ratio'] = pe_val
                            except:
                                pass
            except Exception as e:
                _logger.warning(f"获取港股实时数据失败: {e}")
            
            # 方法2: 尝试使用历史数据获取价格信息
            if info['current_price'] == 'N/A':
                try:
                    hist_df = ak.stock_hk_hist(symbol=hk_code, period="daily", 
                                              start_date=(datetime.now() - timedelta(days=5)).strftime('%Y%m%d'),
                                              end_date=datetime.now().strftime('%Y%m%d'), adjust="qfq")
                    if hist_df is not None and not hist_df.empty:
                        latest = hist_df.iloc[-1]
                        info['current_price'] = latest['收盘']
                        # 计算涨跌幅
                        if len(hist_df) > 1:
                            prev_close = hist_df.iloc[-2]['收盘']
                            change_pct = ((latest['收盘'] - prev_close) / prev_close) * 100
                            info['change_percent'] = round(change_pct, 2)
                except Exception as e:
                    _logger.warning(f"获取港股历史数据失败: {e}")
            
            return info
            
        except Exception as e:
            _logger.warning(f"获取港股信息完全失败: {e}")
            # 返回基本信息
            return {
                "symbol": symbol,
                "name": f"港股{symbol}",
                "current_price": "N/A",
                "change_percent": "N/A",
                "pe_ratio": "N/A",
                "pb_ratio": "N/A",
                "market_cap": "N/A",
                "market": "香港股市",
                "exchange": "香港交易所"
            }
    
    def _get_us_stock_info(self, symbol):
        """获取美股基本信息"""
        import time
        
        try:
            # 添加延迟避免频率限制
            time.sleep(1)
            
            ticker = yf.Ticker(symbol)
            
            # 先尝试获取历史数据（通常更稳定）
            try:
                hist = ticker.history(period="2d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    if len(hist) > 1:
                        prev_close = hist['Close'].iloc[-2]
                        change_percent = ((current_price - prev_close) / prev_close) * 100
                    else:
                        change_percent = 'N/A'
                else:
                    current_price = 'N/A'
                    change_percent = 'N/A'
            except:
                current_price = 'N/A'
                change_percent = 'N/A'
            
            # 获取基本信息
            try:
                info = ticker.info
                
                # 获取市盈率，优先使用trailing PE，其次forward PE
                pe_ratio = info.get('trailingPE', info.get('forwardPE', 'N/A'))
                if pe_ratio == 'N/A' or pe_ratio is None or (isinstance(pe_ratio, float) and np.isnan(pe_ratio)):
                    pe_ratio = 'N/A'
                
                # 获取市净率
                pb_ratio = info.get('priceToBook', 'N/A')
                if pb_ratio == 'N/A' or pb_ratio is None or (isinstance(pb_ratio, float) and np.isnan(pb_ratio)):
                    pb_ratio = 'N/A'
                
                # 如果历史数据没有获取到价格，尝试从info获取
                if current_price == 'N/A':
                    current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
                
                if change_percent == 'N/A':
                    change_percent = info.get('regularMarketChangePercent', 'N/A')
                    if change_percent != 'N/A' and change_percent is not None:
                        change_percent = change_percent * 100  # 转换为百分比
                
                return {
                    "symbol": symbol,
                    "name": info.get('longName', info.get('shortName', 'N/A')),
                    "current_price": current_price,
                    "change_percent": change_percent,
                    "market_cap": info.get('marketCap', 'N/A'),
                    "pe_ratio": pe_ratio,
                    "pb_ratio": pb_ratio,
                    "dividend_yield": info.get('dividendYield', 'N/A'),
                    "beta": info.get('beta', 'N/A'),
                    "52_week_high": info.get('fiftyTwoWeekHigh', 'N/A'),
                    "52_week_low": info.get('fiftyTwoWeekLow', 'N/A'),
                    "sector": info.get('sector', 'N/A'),
                    "industry": info.get('industry', 'N/A'),
                    "market": "美股",
                    "exchange": info.get('exchange', 'N/A')
                }
                
            except Exception as e:
                # 如果获取详细信息失败，返回基本价格信息
                return {
                    "symbol": symbol,
                    "name": f"美股{symbol}",
                    "current_price": current_price,
                    "change_percent": change_percent,
                    "market_cap": 'N/A',
                    "pe_ratio": 'N/A',
                    "pb_ratio": 'N/A',
                    "dividend_yield": 'N/A',
                    "beta": 'N/A',
                    "52_week_high": 'N/A',
                    "52_week_low": 'N/A',
                    "sector": 'N/A',
                    "industry": 'N/A',
                    "market": "美股",
                    "exchange": 'N/A'
                }
                
        except Exception as e:
            return {"error": f"获取美股信息失败: {str(e)}"}
    
    def _get_chinese_stock_data(self, symbol, period="1y"):
        """获取中国股票历史数据（支持akshare和tushare数据源自动切换）"""
        try:
            # 计算日期范围
            end_date = datetime.now().strftime('%Y%m%d')
            if period == "1y":
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            elif period == "6mo":
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
            elif period == "3mo":
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
            elif period.endswith("y") and period[:-1].isdigit():
                # 通用 "Ny"（如 "3y"/"5y"）：回测需要覆盖多年区间时用得上。
                # 原实现只认 1y/6mo/3mo，其余一律退到 1 年——于是"回测 2024 年"
                # 实际只取到最近一年，历史日期一条都命不中（见 core/backtest._period_covering）。
                start_date = (datetime.now() - timedelta(days=365 * int(period[:-1]))).strftime('%Y%m%d')
            elif period.endswith("mo") and period[:-2].isdigit():
                start_date = (datetime.now() - timedelta(days=30 * int(period[:-2]))).strftime('%Y%m%d')
            else:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            
            # 使用数据源管理器获取数据（自动切换akshare和tushare）
            df = self.data_source_manager.get_stock_hist_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust='qfq'
            )
            
            if df is not None and not df.empty:
                # 标准化列名为大写（与原有格式保持一致）
                df = df.rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'close': 'Close',
                    'high': 'High',
                    'low': 'Low',
                    'volume': 'Volume'
                })
                
                # 确保Date列为datetime类型
                if 'Date' not in df.columns and df.index.name == 'date':
                    df.index.name = 'Date'
                elif 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                
                _logger.info(f"✅ 成功获取 {symbol} 的历史数据，共 {len(df)} 条记录")
                return df
            else:
                return {"error": "所有数据源均无法获取历史数据"}
                
        except Exception as e:
            return {"error": f"获取中国股票数据失败: {str(e)}"}
    
    def _get_hk_stock_data(self, symbol, period="1y"):
        """获取港股历史数据"""
        try:
            # 规范化港股代码
            hk_code = self._normalize_hk_code(symbol)
            
            # 计算日期范围
            end_date = datetime.now().strftime('%Y%m%d')
            if period == "1y":
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            elif period == "6mo":
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
            elif period == "3mo":
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
            elif period == "1mo":
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            else:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            
            # 获取港股历史数据
            df = ak.stock_hk_hist(symbol=hk_code, period="daily", 
                                start_date=start_date, end_date=end_date, adjust="qfq")
            
            if df is not None and not df.empty:
                # 重命名列以匹配标准格式
                df = df.rename(columns={
                    '日期': 'Date',
                    '开盘': 'Open',
                    '收盘': 'Close',
                    '最高': 'High',
                    '最低': 'Low',
                    '成交量': 'Volume'
                })
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df
            else:
                return {"error": "无法获取港股历史数据"}
                
        except Exception as e:
            return {"error": f"获取港股数据失败: {str(e)}"}
    
    def _get_us_stock_data(self, symbol, period="1y", interval="1d"):
        """获取美股历史数据"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if not df.empty:
                return df
            else:
                return {"error": "无法获取历史数据"}
        except Exception as e:
            return {"error": f"获取美股数据失败: {str(e)}"}
    
    def calculate_technical_indicators(self, df):
        """计算技术指标"""
        try:
            if isinstance(df, dict) and "error" in df:
                return df
                
            # 移动平均线
            df['MA5'] = ta.trend.sma_indicator(df['Close'], window=5)
            df['MA10'] = ta.trend.sma_indicator(df['Close'], window=10)
            df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['MA60'] = ta.trend.sma_indicator(df['Close'], window=60)
            
            # RSI
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_histogram'] = macd.macd_diff()
            
            # 布林带
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_middle'] = bollinger.bollinger_mavg()
            df['BB_lower'] = bollinger.bollinger_lband()
            
            # KDJ指标
            df['K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
            df['D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
            
            # 成交量指标
            df['Volume_MA5'] = ta.trend.sma_indicator(df['Volume'], window=5)
            df['Volume_ratio'] = df['Volume'] / df['Volume_MA5']

            # ATR（真实波幅均值）—— 判断"止损放多宽才不会被日常噪声扫掉"的唯一依据。
            # ⚠️ 补它的原因（2026-10-09 实测 603127）：风控节点的任务是
            #    "评估止损设置是否恰当"，但提示词里**一个波动率指标都没有** ——
            #    它只能凭感觉写"止损幅度可控、设置恰当"。
            #    那只票 ATR14 = 2.84 元（占现价 5.45%，Wilder 平滑），而计划止损距离只有 1.50 元
            #    （≈0.53×ATR）——**不到一天的平均波动**，等于"必被扫损"。
            #    没有 ATR，这个结论在任何一份报告里都不可能被算出来。
            df['ATR14'] = ta.volatility.AverageTrueRange(
                df['High'], df['Low'], df['Close'], window=14).average_true_range()

            return df
            
        except Exception as e:
            return {"error": f"计算技术指标失败: {str(e)}"}
    
    def get_latest_indicators(self, df):
        """获取最新的技术指标值"""
        try:
            if isinstance(df, dict) and "error" in df:
                return df
                
            latest = df.iloc[-1]
            
            return {
                "price": latest['Close'],
                "ma5": latest['MA5'],
                "ma10": latest['MA10'], 
                "ma20": latest['MA20'],
                "ma60": latest['MA60'],
                "rsi": latest['RSI'],
                "macd": latest['MACD'],
                "macd_signal": latest['MACD_signal'],
                # v4（2026-10-08）：下面这些列**本来就算了**，只是没导出。
                # 后果实测（603127 报告）：技术面写「布林带中轨（未提供）」，
                # 而交易员又凭空给出「中轨约 45.1–45.4」——为已声明"不可得"的指标编了值。
                # 凡是算出来的、且下游会引用的，就必须一起给出。
                "macd_histogram": latest['MACD_histogram'],
                "bb_upper": latest['BB_upper'],
                "bb_middle": latest['BB_middle'],       # 布林中轨（= MA20）
                "bb_lower": latest['BB_lower'],
                "k_value": latest['K'],
                "d_value": latest['D'],
                # ⚠️ 口径必须写死：`volume_ratio` 的分母是 **5 日均量**，
                # 只给比不给均量，模型就会把它当成"20 日均量"（实测夸大 39%）。
                # 5 日均量与 20 日均量**都给**，并各自命名清楚。
                "volume": float(latest['Volume']),
                "volume_ma5": float(latest['Volume_MA5']),
                "volume_ma20": float(pd.to_numeric(df['Volume'], errors='coerce')
                                     .tail(20).mean()) if len(df) >= 20 else None,
                "volume_ratio": latest['Volume_ratio'],
                # ATR14（真实波幅均值，单位：元）—— 止损宽度的标尺。
                # 下游（风控/终审）判断"止损是否恰当"必须与它对比：
                # 距离 < 1×ATR 的止损基本等于"用日常波动换一次出局"。
                "atr14": (None if pd.isna(latest.get('ATR14')) else float(latest['ATR14'])),
            }
        except Exception as e:
            return {"error": f"获取最新指标失败: {str(e)}"}
    
    def get_financial_data(self, symbol):
        """获取详细财务数据"""
        try:
            if self._is_chinese_stock(symbol):
                return self._get_chinese_financial_data(symbol)
            elif self._is_hk_stock(symbol):
                return self._get_hk_financial_data(symbol)
            else:
                return self._get_us_financial_data(symbol)
        except Exception as e:
            return {"error": f"获取财务数据失败: {str(e)}"}
    
    def _get_chinese_financial_data(self, symbol):
        """获取中国股票财务数据

        v3 修正（2026-09-25 实测）。原实现有三处错误，合起来让深度分析把
        **1998~2005 年的数字当成"最新财报"**：

          1. `indicator` 传了**非法值**（"资产负债表"/"利润表"/"现金流量表"）——
             `stock_financial_abstract_ths` 只接受 {"按报告期","按年度","按单季度"}，
             传错值后**三张表返回的是同一份"主要指标"**（列名全是净利润/营业总收入，
             "资产负债表"里一个资产科目都没有）；
          2. 紧接着 `head(8)` —— 该接口**最老在前**（1998→2025），
             head 取到的是上市以来**最早**的 8 期（实测 600519 = 1998~2005）；
          3. 于是"基本面"输入成了二十多年前的数据，且三张表内容相同。

        现在：三张报表改用**新浪** `stock_financial_report_sina`（实测口径正确：
        列是真实报表科目），再用 `_latest_period_records` 按报告期排序取最近 8 期。
        同花顺的"主要指标"退为兜底（用合法 indicator），避免单源不可用时全空。
        """
        N = 8
        financial_data = {
            "symbol": symbol,
            "balance_sheet": None,  # 资产负债表
            "income_statement": None,  # 利润表
            "cash_flow": None,  # 现金流量表
            "financial_ratios": {},  # 财务比率
            "quarter_data": None,  # 季度数据
        }

        def _ths(name: str):
            """同花顺主要指标兜底（indicator 必须用它的合法取值）"""
            try:
                df = ak.stock_financial_abstract_ths(symbol=symbol, indicator=name)
                if df is not None and not df.empty:
                    return df
            except Exception as e:               # noqa: BLE001
                _logger.warning(f"同花顺{name}获取失败: {e}")
            return None

        # 1. 三张报表：走新浪（科目正确、最新在前），失败再退回同花顺主要指标
        for key, rep in (("balance_sheet", "资产负债表"),
                         ("income_statement", "利润表"),
                         ("cash_flow", "现金流量表")):
            rows = []
            try:
                df = ak.stock_financial_report_sina(stock=symbol, symbol=rep)
                rows = _latest_period_records(df, N)
            except Exception as e:               # noqa: BLE001
                _logger.warning(f"获取{rep}失败: {e}")
            if not rows:
                # 兜底：同花顺主要指标（合法 indicator），并注明来源不是完整报表
                ths = _ths("按报告期")
                if ths is not None:
                    rows = _latest_period_records(ths, N)
                    for r in rows:
                        r["_source"] = "ths主要指标(非完整报表)"
                    _logger.info(f"⚠️ {rep} 仅取到同花顺主要指标兜底数据（非完整报表科目）")
            financial_data[key] = rows or None

        # 2. 主要财务指标（东财）
        try:
            financial_abstract = ak.stock_financial_abstract(symbol=symbol)
            if financial_abstract is not None and not financial_abstract.empty:
                key_indicators = [
                    '净资产收益率(ROE)', '总资产报酬率(ROA)', '销售毛利率', '销售净利率',
                    '资产负债率', '流动比率', '速动比率', '存货周转率', '应收账款周转率',
                    '总资产周转率', '营业收入同比增长', '净利润同比增长'
                ]

                indicator_rows = financial_abstract[financial_abstract['指标'].isin(key_indicators)]

                if not indicator_rows.empty:
                    # ⚠️ 不要假设第 0 列就是最新：不同源列序不同，统一按日期排序后再取。
                    # （原实现写死 date_columns[0]，同样是"猜方向"的写法。）
                    date_columns = sorted(
                        (c for c in financial_abstract.columns if c not in ('选项', '指标')),
                        key=_period_key, reverse=True)
                    if date_columns:
                        latest_date = date_columns[0]

                        # 构建财务比率字典
                        financial_ratios = {"报告期": latest_date}

                        # 提取每个指标的最新值
                        for _, row in indicator_rows.iterrows():
                            indicator_name = row['指标']
                            value = row.get(latest_date, 'N/A')
                            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                                try:
                                    financial_ratios[indicator_name] = str(value)
                                except Exception:            # noqa: BLE001
                                    financial_ratios[indicator_name] = "N/A"
                            else:
                                financial_ratios[indicator_name] = "N/A"

                        financial_data["financial_ratios"] = financial_ratios
        except Exception as e:                       # noqa: BLE001
            _logger.warning(f"获取财务指标失败: {e}")

        # 注意：季报数据现在由 quarterly_report_data.py 模块使用 akshare 获取（8期完整季报）
        # 不再使用问财获取季报，避免重复

        return financial_data
    
    # 已删除 _get_quarter_data_from_wencai 方法
    # 季报数据现在统一由 quarterly_report_data.py 模块使用 akshare 获取
    # 获取最近8期完整季报（利润表、资产负债表、现金流量表）
    # 避免重复获取，提高效率
    
    def _get_hk_financial_data(self, symbol):
        """获取港股财务数据"""
        hk_code = self._normalize_hk_code(symbol)
        
        financial_data = {
            "symbol": hk_code,
            "balance_sheet": None,
            "income_statement": None,
            "cash_flow": None,
            "financial_ratios": {},
            "quarter_data": None,
            "data_source": "eastmoney",
            "note": "港股财务数据来自东方财富"
        }
        
        try:
            # 使用akshare获取港股财务指标（东方财富数据源）
            _logger.info(f"正在获取港股 {hk_code} 的财务指标...")
            try:
                financial_indicator = ak.stock_hk_financial_indicator_em(symbol=hk_code)
                
                if financial_indicator is not None and not financial_indicator.empty:
                    # 将财务指标数据转换为字典
                    indicator_dict = financial_indicator.iloc[0].to_dict()
                    
                    # 整理财务比率数据
                    financial_data["financial_ratios"] = {
                        "基本每股收益": self._safe_convert(indicator_dict.get('基本每股收益(元)', 'N/A')),
                        "每股净资产": self._safe_convert(indicator_dict.get('每股净资产(元)', 'N/A')),
                        "每股股息TTM": self._safe_convert(indicator_dict.get('每股股息TTM(港元)', 'N/A')),
                        "派息比率": self._safe_convert(indicator_dict.get('派息比率(%)', 'N/A')),
                        "每股经营现金流": self._safe_convert(indicator_dict.get('每股经营现金流(元)', 'N/A')),
                        "股息率TTM": self._safe_convert(indicator_dict.get('股息率TTM(%)', 'N/A')),
                        "总市值": self._safe_convert(indicator_dict.get('总市值(港元)', 'N/A')),
                        "港股市值": self._safe_convert(indicator_dict.get('港股市值(港元)', 'N/A')),
                        "营业总收入": self._safe_convert(indicator_dict.get('营业总收入', 'N/A')),
                        "营业收入环比增长": self._safe_convert(indicator_dict.get('营业总收入滚动环比增长(%)', 'N/A')),
                        "销售净利率": self._safe_convert(indicator_dict.get('销售净利率(%)', 'N/A')),
                        "净利润": self._safe_convert(indicator_dict.get('净利润', 'N/A')),
                        "净利润环比增长": self._safe_convert(indicator_dict.get('净利润滚动环比增长(%)', 'N/A')),
                        "ROE股东权益回报率": self._safe_convert(indicator_dict.get('股东权益回报率(%)', 'N/A')),
                        "市盈率": self._safe_convert(indicator_dict.get('市盈率', 'N/A')),
                        "市净率": self._safe_convert(indicator_dict.get('市净率', 'N/A')),
                        "ROA总资产回报率": self._safe_convert(indicator_dict.get('总资产回报率(%)', 'N/A')),
                        "法定股本": self._safe_convert(indicator_dict.get('法定股本(股)', 'N/A')),
                        "已发行股本": self._safe_convert(indicator_dict.get('已发行股本(股)', 'N/A')),
                        "每手股": self._safe_convert(indicator_dict.get('每手股', 'N/A')),
                    }
                    
                    _logger.info(f"✅ 成功获取港股 {hk_code} 的财务指标")
                    _logger.info(f"   ROE: {financial_data['financial_ratios']['ROE股东权益回报率']}")
                    _logger.info(f"   市盈率: {financial_data['financial_ratios']['市盈率']}")
                    _logger.info(f"   市净率: {financial_data['financial_ratios']['市净率']}")
                else:
                    _logger.info(f"⚠️ 未获取到港股 {hk_code} 的财务指标数据")
                    financial_data["note"] = "未获取到财务数据"
                    
            except Exception as e:
                _logger.warning(f"⚠️ 获取港股财务指标失败: {e}")
                financial_data["note"] = f"获取财务数据失败: {str(e)}"
            
            return financial_data
            
        except Exception as e:
            _logger.warning(f"获取港股财务数据异常: {e}")
            financial_data["note"] = f"获取失败: {str(e)}"
            return financial_data
    
    def _get_us_financial_data(self, symbol):
        """获取美股财务数据"""
        financial_data = {
            "symbol": symbol,
            "balance_sheet": None,
            "income_statement": None,
            "cash_flow": None,
            "financial_ratios": {},
            "quarter_data": None,
        }
        
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # 1. 资产负债表
            try:
                balance_sheet = stock.balance_sheet
                if balance_sheet is not None and not balance_sheet.empty:
                    financial_data["balance_sheet"] = balance_sheet.iloc[:, :4].to_dict('index')
            except Exception as e:
                _logger.warning(f"获取资产负债表失败: {e}")
            
            # 2. 利润表
            try:
                income_stmt = stock.income_stmt
                if income_stmt is not None and not income_stmt.empty:
                    financial_data["income_statement"] = income_stmt.iloc[:, :4].to_dict('index')
            except Exception as e:
                _logger.warning(f"获取利润表失败: {e}")
            
            # 3. 现金流量表
            try:
                cash_flow = stock.cashflow
                if cash_flow is not None and not cash_flow.empty:
                    financial_data["cash_flow"] = cash_flow.iloc[:, :4].to_dict('index')
            except Exception as e:
                _logger.warning(f"获取现金流量表失败: {e}")
            
            # 4. 财务比率（从info中提取）
            financial_data["financial_ratios"] = {
                "ROE": info.get('returnOnEquity', 'N/A'),
                "ROA": info.get('returnOnAssets', 'N/A'),
                "毛利率": info.get('grossMargins', 'N/A'),
                "营业利润率": info.get('operatingMargins', 'N/A'),
                "净利率": info.get('profitMargins', 'N/A'),
                "资产负债率": info.get('debtToEquity', 'N/A'),
                "流动比率": info.get('currentRatio', 'N/A'),
                "速动比率": info.get('quickRatio', 'N/A'),
                "EPS": info.get('trailingEps', 'N/A'),
                "每股账面价值": info.get('bookValue', 'N/A'),
                "股息率": info.get('dividendYield', 'N/A'),
                "派息率": info.get('payoutRatio', 'N/A'),
                "收入增长": info.get('revenueGrowth', 'N/A'),
                "盈利增长": info.get('earningsGrowth', 'N/A'),
            }
            
            return financial_data
            
        except Exception as e:
            _logger.warning(f"获取美股财务数据失败: {e}")
            return financial_data
    
    # 已删除 get_fund_flow_data 方法（使用问财）
    # 资金流向数据现在统一由 fund_flow_akshare.py 模块使用 akshare 获取
    # 获取近20个交易日的详细资金流向数据（主力、超大单、大单、中单、小单）
    # 避免重复获取，提高效率和数据质量
    # 
    # 删除说明：
    # - 删除了约160行代码
    # - 删除原因：重复获取，数据格式不规整，日期范围不准确
    # - 新方案：使用 akshare 的 stock_individual_fund_flow 接口
    # - 新方案优势：数据标准化、准确获取最近20个交易日、6类资金详细分类
    
    def get_risk_data(self, symbol):
        """
        获取股票风险数据（限售解禁、大股东减持、重要事件）
        只支持中国A股
        """
        try:
            # 只有中国A股才支持风险数据查询
            if not self._is_chinese_stock(symbol):
                return {
                    'symbol': symbol,
                    'data_success': False,
                    'error': '仅支持中国A股风险数据查询'
                }
            
            # 使用风险数据获取器
            from risk_data_fetcher import RiskDataFetcher
            fetcher = RiskDataFetcher()
            risk_data = fetcher.get_risk_data(symbol)
            
            return risk_data
            
        except Exception as e:
            return {
                'symbol': symbol,
                'data_success': False,
                'error': f'获取风险数据失败: {str(e)}'
            }
    
    def _safe_convert(self, value):
        """安全地转换数值"""
        if value is None or value == '' or (isinstance(value, float) and np.isnan(value)):
            return 'N/A'
        try:
            if isinstance(value, str):
                # 移除百分号和逗号
                value = value.replace('%', '').replace(',', '')
                return float(value)
            return value
        except:
            return value
    
    def _calculate_main_fund_ratio(self, main_fund, total_fund):
        """计算主力资金占比"""
        try:
            if main_fund != 'N/A' and total_fund != 'N/A' and total_fund != 0:
                ratio = (main_fund / total_fund) * 100
                return f"{ratio:.2f}%"
        except:
            pass
        return 'N/A'

