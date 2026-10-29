"""
资金流数据源可达性探测（方案B前置步骤）
逐个测试候选 akshare 接口，每个接口独立线程+超时，互不阻塞。
用法: python scripts/probe_fund_flow.py
"""
import sys
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")

from data.akshare_helper import patch_requests
patch_requests()

import akshare as ak


def probe(name, fn, timeout=25):
    """在独立线程中调用接口，超时不阻塞后续探测"""
    def run():
        t0 = time.time()
        df = fn()
        elapsed = time.time() - t0
        return df, elapsed

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(run)
        try:
            df, elapsed = fut.result(timeout=timeout)
            if df is None or (hasattr(df, "empty") and df.empty):
                return name, "EMPTY", "返回空数据", 0
            cols = list(df.columns)
            return name, "OK", f"{len(df)}行, {elapsed:.1f}s", cols
        except FutTimeout:
            return name, "TIMEOUT", f">{timeout}s 无响应", []
        except Exception as e:
            return name, "FAIL", f"{type(e).__name__}: {str(e)[:120]}", []


def main():
    candidates = [
        ("原接口 stock_individual_fund_flow(000001)",
         lambda: ak.stock_individual_fund_flow(stock="000001", market="sz")),
        ("东财批量 stock_individual_fund_flow_rank(今日)",
         lambda: ak.stock_individual_fund_flow_rank(indicator="今日")),
        ("东财批量 stock_individual_fund_flow_rank(5日排行)",
         lambda: ak.stock_individual_fund_flow_rank(indicator="5日排行")),
        ("东财批量 stock_individual_fund_flow_rank(10日排行)",
         lambda: ak.stock_individual_fund_flow_rank(indicator="10日排行")),
        ("东财行业 stock_sector_fund_flow_rank(今日)",
         lambda: ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流向")),
        ("同花顺 stock_fund_flow_individual(即时)",
         lambda: ak.stock_fund_flow_individual(symbol="即时")),
    ]

    print("=" * 70)
    print("资金流候选数据源探测")
    print("=" * 70)
    results = []
    for name, fn in candidates:
        r_name, status, detail, cols = probe(name, fn)
        results.append((r_name, status))
        print(f"\n[{status}] {r_name}\n    {detail}")
        if cols:
            print(f"    列: {cols[:18]}")

    print("\n" + "=" * 70)
    ok = [n for n, s in results if s == "OK"]
    print(f"结论: {len(ok)}/{len(results)} 个接口可用 -> {ok if ok else '全部不可用'}")


if __name__ == "__main__":
    main()
