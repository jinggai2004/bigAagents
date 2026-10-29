"""
数据源管理器
实现多源自动降级：腾讯 HTTP 直连 → akshare → tushare（含 TTL 缓存）

历史 K 线的降级链（2026-09-25 调整）：
  腾讯 HTTP 直连（data/tencent_source）→ akshare 的腾讯封装 → tushare
为什么把直连提到 akshare 前面：日志实测 586/587 条取数失败全部来自 akshare 的
`stock_zh_a_hist_tx`（空响应体被 demjson 解码报 "No value to decode"），
而同一个腾讯端点直连 HTTP 是稳定的。详见 data/tencent_source.py 的模块说明。
"""

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ============================================================
# 关键修复: 在引入 akshare 之前打上请求补丁
# 为所有 requests 请求注入 User-Agent 等请求头，
# 解决东方财富服务器 RemoteDisconnected 问题
# ============================================================
from data.akshare_helper import patch_requests, retry_on_failure
patch_requests()

# ── A2: 引入日志器记录缓存/重试事件 ──
from utils.logger import get_logger
_src_logger = get_logger("data.source_manager")

# ── M0 0.4: 数据契约断言（格式漂移时显式失败，不再静默产出坏数据）──
from data.contracts import require_columns, require_not_all_nan
import logging

_logger = logging.getLogger(__name__)


class DataSourceManager:
    """数据源管理器 - 实现akshare与tushare自动切换（含 TTL 缓存）"""

    def __init__(self, cache_ttl: int = 300):
        self.tushare_token = os.getenv('TUSHARE_TOKEN', '')
        self.tushare_available = False
        self.tushare_api = None

        # ── A2: TTL 缓存参数 ──
        self.cache_ttl = cache_ttl  # 缓存有效期（秒），默认 300s
        self._hist_cache: dict = {}  # {(symbol, start, end, adjust): (df, timestamp)}

        # 初始化tushare
        if self.tushare_token:
            try:
                import tushare as ts
                ts.set_token(self.tushare_token)
                self.tushare_api = ts.pro_api()
                self.tushare_available = True
                _logger.info("✅ Tushare数据源初始化成功")
            except Exception as e:
                _logger.warning(f"⚠️ Tushare数据源初始化失败: {e}")
                self.tushare_available = False
        else:
            _logger.info("ℹ️ 未配置Tushare Token，将仅使用Akshare数据源")

    def get_stock_hist_data(self, symbol, start_date=None, end_date=None, adjust='qfq'):
        """
        获取股票历史数据（优先akshare，失败时使用tushare）

        Args:
            symbol: 股票代码（6位数字）
            start_date: 开始日期（格式：'20240101'或'2024-01-01'）
            end_date: 结束日期
            adjust: 复权类型（'qfq'前复权, 'hfq'后复权, ''不复权）

        Returns:
            DataFrame: 包含日期、开盘、收盘、最高、最低、成交量等列
        """
        # 标准化日期格式
        if start_date:
            start_date = start_date.replace('-', '')
        if end_date:
            end_date = end_date.replace('-', '')
        else:
            end_date = datetime.now().strftime('%Y%m%d')

        # ── A2: 检查 TTL 缓存 ──
        cache_key = (symbol, start_date, end_date, adjust)
        if cache_key in self._hist_cache:
            cached_df, cached_time = self._hist_cache[cache_key]
            age = time.time() - cached_time
            if age < self.cache_ttl:
                _src_logger.debug(f"[Cache] ✅ {symbol} 缓存命中 ({age:.0f}s 前)")
                return cached_df.copy()
            else:
                _src_logger.debug(f"[Cache] ⏰ {symbol} 缓存已过期 ({age:.0f}s > {self.cache_ttl}s)，重新拉取")
                del self._hist_cache[cache_key]

        # ── 第一层：腾讯 HTTP 直连（不经 akshare）──
        # 为什么把直连放在 akshare **前面**：akshare 的 stock_zh_a_hist_tx 本质只是这个
        # 端点的封装，却引入三个脆弱点：
        #   1) 用 demjson 解码，**响应体为空时报 "No value to decode"**（看不出真实原因）；
        #   2) 每次先打一次 get_tx_start_year() 取上市年份 —— 平白多一个失败点；
        #   3) 按年循环且任一 年失败即整体抛错，其余年份已拿到的数据也一起丢掉。
        # 日志实测（2026-09-25）：**586 / 587 条失败全部来自这一个接口**，
        # 而它是历史 K 线的唯一主路径。直连实测 9/9 成功、平均 0.65s，且与 akshare
        # 输出逐日等价（600519 全 2024 年 242 个交易日，close/volume 零差异）。
        try:
            from data.tencent_source import fetch_kline
            # 预算收紧（timeout=8 / retries=2，最坏约 16s）：这一层只是降级链的**第一站**，
            # 后面还有 akshare 与 tushare。离线或上游不可达时，不能让它把时间耗光
            # （旧实现里 akshare 走的是默认 30s 超时 × 3 次重试 + 6s 退避，最坏近百秒）。
            df = fetch_kline(symbol, start_date, end_date, adjust=adjust,
                             timeout=8.0, retries=2)
            if df is not None and not df.empty:
                require_columns(df, ['date', 'open', 'close', 'high', 'low', 'volume'],
                                '腾讯直连')
                require_not_all_nan(df, ['open', 'close'], '腾讯直连')
                _src_logger.info(f"[腾讯直连] ✅ {symbol} 获取成功 ({len(df)} 条)")
                self._hist_cache[cache_key] = (df.copy(), time.time())
                return df
            _src_logger.warning(f"[腾讯直连] ⚠️ {symbol} 未取到数据，降级到 akshare")
        except Exception as e:
            _src_logger.warning(f"[腾讯直连] ❌ {symbol} 失败: {e}，降级到 akshare")

        # 第二层：akshare 封装（保留为兜底）
        for retry_count in range(3):
            try:
                import akshare as ak
                _src_logger.info(f"[Akshare-腾讯] 获取 {symbol} 历史数据…" +
                      (f" (第{retry_count+1}/3次重试)" if retry_count > 0 else ""))

                # 使用腾讯数据源（proxy.finance.qq.com），避免东方财富API屏蔽
                tx_symbol = self._convert_to_tx_code(symbol)
                df = ak.stock_zh_a_hist_tx(
                    symbol=tx_symbol,
                    start_date=f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}",
                    end_date=f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}",
                    adjust=adjust
                )

                if df is not None and not df.empty:
                    # akshare 的 stock_zh_a_hist_tx 把腾讯 K 线的第 6 个字段命名为 amount，
                    # 但**该列其实是成交量（手），不是成交额**。实测证据（2026-09-22）：
                    #   sz000001 2024-01-02~01-05 四天，该列取值 1158366 / 733610 /
                    #   864194 / 1991622，与东财 stock_zh_a_hist 的"成交量"列**逐日一位不差**；
                    #   且 1,158,366 手 ÷ 0.60% 换手率 ≈ 193 亿股，与平安银行总股本一致。
                    # 原实现按"万元"再 ×10000，得到的既不是成交量也不是成交额。
                    df = df.rename(columns={"amount": "volume_hand"})
                    df['date'] = pd.to_datetime(df['date'])
                    # 手 → 股
                    df['volume'] = pd.to_numeric(df['volume_hand'], errors='coerce') * 100
                    # 该源不提供成交额：显式置 NaN，而不是编造一个数字
                    df['amount'] = float('nan')
                    df = df.drop(columns=['volume_hand'])
                    # v2/M0 0.4：格式漂移（改列名/全 NaN）时立刻失败并走降级，
                    # 而不是把缺列的表交回给回测/指标计算
                    require_columns(df, ['date', 'open', 'close', 'high', 'low', 'volume'],
                                    'Akshare-腾讯')
                    require_not_all_nan(df, ['open', 'close'], 'Akshare-腾讯')

                    _src_logger.info(f"[Akshare-腾讯] ✅ {symbol} 获取成功 ({len(df)} 条)")

                    # ── A2: 写入 TTL 缓存 ──
                    self._hist_cache[cache_key] = (df.copy(), time.time())
                    return df
                else:
                    _src_logger.warning(f"[Akshare-腾讯] ⚠️ {symbol} 返回空数据，重试中…")
                    time.sleep(1)
                    continue
            except Exception as e:
                _src_logger.error(f"[Akshare-腾讯] ❌ {symbol} 获取失败: {e}")
                if retry_count < 2:
                    delay = (retry_count + 1) * 2
                    _src_logger.info(f"[Akshare-腾讯] ⏳ {delay}s 后重试…")
                    time.sleep(delay)
                else:
                    _src_logger.error(f"[Akshare-腾讯] ❌ {symbol} 已重试 3 次，放弃")

        # akshare失败，尝试tushare
        if self.tushare_available:
            try:
                _src_logger.info(f"[Tushare] 获取 {symbol} 历史数据（备用数据源）…")

                # 转换股票代码格式（添加市场后缀）
                ts_code = self._convert_to_ts_code(symbol)

                # 转换复权类型
                adj_dict = {'qfq': 'qfq', 'hfq': 'hfq', '': None}
                adj = adj_dict.get(adjust, 'qfq')

                # 格式化日期
                start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}" if start_date else None
                end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}" if end_date else None

                # 获取数据
                df = self.tushare_api.daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    adj=adj
                )

                if df is not None and not df.empty:
                    # 标准化列名和数据格式
                    df = df.rename(columns={
                        'trade_date': 'date',
                        'vol': 'volume',
                        'amount': 'amount'
                    })
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')

                    # 转换成交量单位（tushare单位是手，转换为股）
                    df['volume'] = df['volume'] * 100
                    # 转换成交额单位（tushare单位是千元，转换为元）
                    df['amount'] = df['amount'] * 1000

                    _src_logger.info(f"[Tushare] ✅ {symbol} 获取成功 ({len(df)} 条)")
                    # ── A2: 写入 TTL 缓存 ──
                    self._hist_cache[cache_key] = (df.copy(), time.time())
                    return df
            except Exception as e:
                _src_logger.error(f"[Tushare] ❌ {symbol} 获取失败: {e}")

        # 两个数据源都失败
        _src_logger.error(f"❌ {symbol} 所有数据源均获取失败")
        return None

    def get_stock_basic_info(self, symbol):
        """
        获取股票基本信息（优先akshare，失败时使用tushare）

        Args:
            symbol: 股票代码

        Returns:
            dict: 股票基本信息
        """
        info = {
            "symbol": symbol,
            "name": "未知",
            "industry": "未知",
            "market": "未知"
        }

        # 优先使用akshare（仅尝试1次，失败快速降级到新浪）
        for retry_count in range(1):
            try:
                import akshare as ak
                _logger.info(f"[Akshare] 正在获取 {symbol} 的基本信息...")

                stock_info = ak.stock_individual_info_em(symbol=symbol)
                if stock_info is not None and not stock_info.empty:
                    for _, row in stock_info.iterrows():
                        key = row['item']
                        value = row['value']

                        if key == '股票简称':
                            info['name'] = value
                        elif key == '所处行业':
                            info['industry'] = value
                        elif key == '上市时间':
                            info['list_date'] = value
                        elif key == '总市值':
                            info['market_cap'] = value
                        elif key == '流通市值':
                            info['circulating_market_cap'] = value

                    _logger.info(f"[Akshare] ✅ 成功获取基本信息")
                    return info
            except Exception as e:
                _logger.warning(f"[Akshare] ❌ 获取失败: {e}")

        # akshare（东方财富）失败，尝试新浪单只股票接口获取名称
        try:
            import requests as req
            tx_code = self._convert_to_tx_code(symbol)
            url = f'https://hq.sinajs.cn/list={tx_code}'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://finance.sina.com.cn/',
            }
            r = req.get(url, headers=headers, timeout=10)
            # 返回格式: var hq_str_sh603212="名称,open,pre_close,current,...";
            if r.status_code == 200 and f'hq_str_{tx_code}' in r.text:
                # 提取引号内的数据
                start = r.text.find('"') + 1
                end = r.text.find('"', start)
                if start > 0 and end > start:
                    fields = r.text[start:end].split(',')
                    if len(fields) >= 1 and fields[0]:
                        info['name'] = fields[0]
                        info['market'] = '中国A股'
                        _logger.info(f"[新浪个股] ✅ 成功获取基本信息: {info['name']}")
                        return info
        except Exception as e:
            _logger.warning(f"[新浪个股] ❌ 获取失败: {e}")

        # akshare失败，尝试tushare（备用数据源）
        if self.tushare_available:
            try:
                _logger.info(f"[Tushare] 正在获取 {symbol} 的基本信息（备用数据源）...")

                ts_code = self._convert_to_ts_code(symbol)
                df = self.tushare_api.stock_basic(
                    ts_code=ts_code,
                    fields='ts_code,name,area,industry,market,list_date'
                )

                if df is not None and not df.empty:
                    info['name'] = df.iloc[0]['name']
                    info['industry'] = df.iloc[0]['industry']
                    info['market'] = df.iloc[0]['market']
                    info['list_date'] = df.iloc[0]['list_date']

                    _logger.info(f"[Tushare] ✅ 成功获取基本信息")
                    return info
            except Exception as e:
                _logger.warning(f"[Tushare] ❌ 获取失败: {e}")

        return info

    def get_realtime_quotes(self, symbol):
        """
        获取实时行情数据（优先akshare，失败时使用tushare）

        Args:
            symbol: 股票代码

        Returns:
            dict: 实时行情数据
        """
        quotes = {}

        # 优先使用新浪个股接口（比全市场遍历快得多）
        for retry_count in range(3):
            try:
                import requests as req_sina
                tx_code = self._convert_to_tx_code(symbol)
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
                        if len(fields) >= 32:
                            def safe_float(val, default=0):
                                try: return float(val) if val else default
                                except: return default

                            quotes = {
                                'symbol': symbol,
                                'name': fields[0] if fields[0] else '',
                                'open': safe_float(fields[1]),
                                'pre_close': safe_float(fields[2]),
                                'price': safe_float(fields[3]),
                                'high': safe_float(fields[4]),
                                'low': safe_float(fields[5]),
                                'volume': safe_float(fields[8]),
                                'amount': safe_float(fields[9]),
                            }
                            # 计算涨跌幅
                            if quotes['pre_close'] > 0:
                                quotes['change'] = round(quotes['price'] - quotes['pre_close'], 3)
                                quotes['change_percent'] = round(
                                    (quotes['change'] / quotes['pre_close']) * 100, 2
                                )
                            _logger.info(f"[新浪个股] ✅ 成功获取实时行情: {quotes['name']} {quotes['price']}")
                            return quotes
            except Exception as e:
                _logger.warning(f"[新浪个股] ❌ 获取失败: {e}")
                if retry_count < 2:
                    delay = (retry_count + 1) * 2
                    _logger.info(f"[新浪个股] ⏳ {delay}s 后重试...")
                    time.sleep(delay)

        # akshare失败，尝试tushare
        if self.tushare_available:
            try:
                _logger.info(f"[Tushare] 正在获取 {symbol} 的实时行情（备用数据源）...")

                ts_code = self._convert_to_ts_code(symbol)
                df = self.tushare_api.daily(
                    ts_code=ts_code,
                    start_date=datetime.now().strftime('%Y%m%d'),
                    end_date=datetime.now().strftime('%Y%m%d')
                )

                if df is not None and not df.empty:
                    row = df.iloc[0]
                    quotes = {
                        'symbol': symbol,
                        'price': row['close'],
                        'change_percent': row['pct_chg'],
                        'volume': row['vol'] * 100,
                        'amount': row['amount'] * 1000,
                        'high': row['high'],
                        'low': row['low'],
                        'open': row['open'],
                        'pre_close': row['pre_close']
                    }
                    _logger.info(f"[Tushare] ✅ 成功获取实时行情")
                    return quotes
            except Exception as e:
                _logger.warning(f"[Tushare] ❌ 获取失败: {e}")

        return quotes

    def get_financial_data(self, symbol, report_type='income'):
        """
        获取财务数据（优先akshare，失败时使用tushare）

        Args:
            symbol: 股票代码
            report_type: 报表类型（'income'利润表, 'balance'资产负债表, 'cashflow'现金流量表）

        Returns:
            DataFrame: 财务数据
        """
        # 优先使用akshare（带重试机制）
        for retry_count in range(3):
            try:
                import akshare as ak
                _logger.info(f"[Akshare] 正在获取 {symbol} 的财务数据..." +
                      (f" (第{retry_count+1}次)" if retry_count > 0 else ""))

                if report_type == 'income':
                    df = ak.stock_financial_report_sina(stock=symbol, symbol="利润表")
                elif report_type == 'balance':
                    df = ak.stock_financial_report_sina(stock=symbol, symbol="资产负债表")
                elif report_type == 'cashflow':
                    df = ak.stock_financial_report_sina(stock=symbol, symbol="现金流量表")
                else:
                    df = None

                if df is not None and not df.empty:
                    _logger.info(f"[Akshare] ✅ 成功获取财务数据")
                    return df
            except Exception as e:
                _logger.warning(f"[Akshare] ❌ 获取失败: {e}")
                if retry_count < 2:
                    delay = (retry_count + 1) * 2
                    _logger.info(f"[Akshare] ⏳ {delay}s 后重试...")
                    time.sleep(delay)

        # akshare失败，尝试tushare
        if self.tushare_available:
            try:
                _logger.info(f"[Tushare] 正在获取 {symbol} 的财务数据（备用数据源）...")

                ts_code = self._convert_to_ts_code(symbol)

                if report_type == 'income':
                    df = self.tushare_api.income(ts_code=ts_code)
                elif report_type == 'balance':
                    df = self.tushare_api.balancesheet(ts_code=ts_code)
                elif report_type == 'cashflow':
                    df = self.tushare_api.cashflow(ts_code=ts_code)
                else:
                    df = None

                if df is not None and not df.empty:
                    _logger.info(f"[Tushare] ✅ 成功获取财务数据")
                    return df
            except Exception as e:
                _logger.warning(f"[Tushare] ❌ 获取失败: {e}")

        return None

    def _convert_to_ts_code(self, symbol):
        """
        将6位股票代码转换为tushare格式（带市场后缀）

        Args:
            symbol: 6位股票代码

        Returns:
            str: tushare格式代码（如：000001.SZ）
        """
        if not symbol or len(symbol) != 6:
            return symbol

        # 根据代码判断市场
        if symbol.startswith('6'):
            # 上海主板
            return f"{symbol}.SH"
        elif symbol.startswith('0') or symbol.startswith('3'):
            # 深圳主板和创业板
            return f"{symbol}.SZ"
        elif symbol.startswith('8') or symbol.startswith('4'):
            # 北交所
            return f"{symbol}.BJ"
        else:
            # 默认深圳
            return f"{symbol}.SZ"

    def _convert_from_ts_code(self, ts_code):
        """
        将tushare格式代码转换为6位代码

        Args:
            ts_code: tushare格式代码（如：000001.SZ）

        Returns:
            str: 6位股票代码
        """
        if '.' in ts_code:
            return ts_code.split('.')[0]
        return ts_code

    def _convert_to_tx_code(self, symbol):
        """
        将6位股票代码转换为腾讯数据源格式（带 sh/sz 前缀）

        Args:
            symbol: 6位股票代码

        Returns:
            str: 腾讯格式代码（如：sh603212, sz000001）
        """
        if not symbol or len(symbol) != 6:
            return symbol

        if symbol.startswith('6'):
            return f"sh{symbol}"
        elif symbol.startswith('0') or symbol.startswith('3'):
            return f"sz{symbol}"
        elif symbol.startswith('8') or symbol.startswith('4'):
            return f"bj{symbol}"
        else:
            return f"sz{symbol}"


# 全局数据源管理器实例
data_source_manager = DataSourceManager()