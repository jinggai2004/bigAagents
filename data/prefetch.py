"""并发预取个股行情与基本信息（2026-10-09，P0：扩展选股候选域）

## 为什么需要
因子精筛是"逐票串行取数"，实测 **≈9.6s/只**（其中基本信息 7.7s、K 线 0.87s），
40 只候选要 6.4 分钟 → **取数产能直接锁死了候选域宽度**（`local_screen.TOTAL_LIMIT`），
也就锁死了"选股结果会不会又是同几只"。

而这两类取数**天然可并发**——瓶颈不是上游慢，是我们在一只一只地等。

## 并发安全的两条依据（已核实，不是假设）
1. `data/tencent_source.fetch_kline` 用 `requests`（每次调用内部建连），**线程安全**；
2. `ak.stock_zh_valuation_baidu` 用 `requests` + 正则解析**不用 py_mini_racer**
   （已读 akshare 源码确认）——而 mini_racer（V8）是 akshare 里唯一不能多线程的部分
   （见 `core/market_detector.py` 的历史注释：ThreadPoolExecutor 曾因此被移除）。
⇒ 因此本模块**强制 `skip_em=True`**：预取的信息取数只走"新浪 + 百度"（都是 requests），
   **完全不进 akshare**。东财那条路径留给后续串行兜底（那里东财可达时仍能拿到数据），
   这样"并发"与"口径完整"不冲突。

## 另外两条纪律
- **限速**：腾讯/新浪/百度都是非正式公开接口，全局限速（默认 5 req/s），
  参数集中在本模块常量里，便于按上游反馈调整；
- **缺就是缺**：失败的股票进 `failed` 字典并带原因，**不**返回空 DataFrame 冒充"没数据"；
  调用方（screener）对失败项退回串行取数，不会静默丢股票。
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)

# 并发度与限速：8 并发配合 5 req/s 的全局闸门 ≈ 有效吞吐 2.5 只/s（每只 2 次请求）
MAX_WORKERS = 8
RATE_PER_SEC = 5.0
# K 线取数区间：与 `_get_kline_data` 的 `period="3mo"` 等价（MA20/波动率/20日动量够用）
KLINE_PERIOD = "3mo"


class _RateLimiter:
    """全局令牌桶（简单版）：保证"所有线程合计"不超过 rate_per_sec。

    注意是**全局**而非每线程——限速的目的是对上游友好，不是限制单线程。
    """

    def __init__(self, rate_per_sec: float):
        self._interval = 1.0 / max(rate_per_sec, 0.1)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            sleep_for = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self._interval
        if sleep_for > 0:
            time.sleep(sleep_for)


@dataclass
class PrefetchResult:
    klines: dict = field(default_factory=dict)      # code -> DataFrame
    infos: dict = field(default_factory=dict)       # code -> info dict
    failed: dict = field(default_factory=dict)      # code -> 原因（缺就是缺，带原因）
    elapsed: float = 0.0
    warnings: list = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return len(self.klines)

    def summary(self) -> str:
        return (f"并发预取 {len(self.klines)} 只成功"
                + (f"，{len(self.failed)} 只失败" if self.failed else "")
                + f"（{self.elapsed:.1f}s）")


def prefetch_market_data(codes, *, fetcher=None, with_info: bool = True,
                         max_workers: int = MAX_WORKERS,
                         rate_per_sec: float = RATE_PER_SEC,
                         progress_cb=None) -> PrefetchResult:
    """并发预取 K 线（+ 可选基本信息）。

    Args:
        codes: 股票代码列表（会去重、保持顺序无关）
        fetcher: `StockDataFetcher`；None 则内部创建（调用方通常传入共享实例，
                 以复用其内部缓存）
        with_info: 是否同时预取基本信息（PE/名称）。预取走 `skip_em=True`，
                 **不进 akshare**（线程安全前提，见模块文档）
        progress_cb: 进度回调，签名 `cb(done, total)`

    Returns:
        PrefetchResult；失败的股票在 `.failed` 里带原因，调用方需自行兜底。
    """
    # ⚠️ 顺序不能反：必须**先 strip 再去重**。首版写成
    # `dict.fromkeys(codes)` 后再 strip，于是 "600519" 与 " 600519 " 被当成两只，
    # 同一只股票白取两次（测试 test_duplicate_codes_are_fetched_once 抓到）。
    codes = [c for c in dict.fromkeys(str(x).strip() for x in (codes or [])) if c]
    res = PrefetchResult()
    if not codes:
        return res

    if fetcher is None:
        from data.stock_data import StockDataFetcher
        fetcher = StockDataFetcher()

    limiter = _RateLimiter(rate_per_sec)
    t0 = time.time()
    done = 0

    def _one(code: str):
        """返回 (code, kline|None, info|None, error|None)。

        每个请求前都过全局限速闸门——这是对上游唯一的礼貌保证。
        """
        kline, info = None, None
        # ① K 线（走 source_manager：直连优先、失败降级）
        try:
            limiter.wait()
            df = fetcher.get_stock_data(code, period=KLINE_PERIOD)
            if isinstance(df, dict) and "error" in df:
                return code, None, None, f"K线失败: {df['error']}"
            if hasattr(df, "columns") and len(df):
                kline = df
        except Exception as e:                      # noqa: BLE001
            return code, None, None, f"K线异常: {type(e).__name__}: {e}"
        # ② 基本信息（skip_em=True → 只用新浪+百度，避开 akshare 的线程不安全性）
        if with_info:
            try:
                limiter.wait()
                # ② skip_valuation=True ⇒ 不逐票打百度估值接口（2026-10-22）：
                #    116 只并发打百度会被限流并返回 HTML（`Expecting value: line 1
                #    column 1`），每只失败重试把选股从 6 分钟拖到 43 分钟。
                #    PE/PB 改由评分侧读 `market_snapshot`（PE 覆盖 65.5% / PB 91%）。
                got = fetcher.get_stock_info(code, skip_em=True, skip_valuation=True)
                if isinstance(got, dict) and not got.get("error"):
                    info = got
            except Exception as e:                  # noqa: BLE001
                # 信息失败不致命：PE 因子会标 unknown（已有语义），K 线仍可用
                res.warnings.append(f"{code} 基本信息失败: {type(e).__name__}")
        return code, kline, info, None

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        futures = {ex.submit(_one, c): c for c in codes}
        for fu in as_completed(futures):
            code = futures[fu]
            try:
                c, kline, info, err = fu.result()
            except Exception as e:                  # noqa: BLE001
                res.failed[code] = f"任务异常: {type(e).__name__}: {e}"
            else:
                if err:
                    res.failed[code] = err
                if kline is not None:
                    res.klines[c] = kline
                if info is not None:
                    res.infos[c] = info
            done += 1
            if progress_cb:
                try:
                    progress_cb(done, len(codes))
                except Exception:                   # noqa: BLE001
                    pass

    res.elapsed = time.time() - t0
    _logger.info("[并发预取] %s；失败 %d 只", res.summary(), len(res.failed))
    return res
