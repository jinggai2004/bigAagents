#!/usr/bin/env python3
"""数据源可用性验证脚本 v2

测试项目已有和可用的数据源，而非仅测试 AKShare 直接调用。

运行: py check_data_sources.py
"""

import sys
import time

# ★ 必须先应用请求补丁
from data.akshare_helper import patch_requests, retry_on_failure
patch_requests()

import akshare as ak
import pandas as pd
from data.source_manager import DataSourceManager

# 关键验证日期
KEY_DATES = ["20240117", "20240206", "20240924", "20240930"]

def check_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def fail(reason):
    print(f"  ❌ 失败: {reason}")
    return False

def ok(msg=""):
    suffix = f" → {msg}" if msg else ""
    print(f"  ✅ 通过{suffix}")
    return True

def retry_call(fn, name, max_attempts=3):
    """带重试的函数调用"""
    for attempt in range(max_attempts):
        try:
            print(f"  {name} [尝试 {attempt+1}/{max_attempts}]...")
            result = fn()
            if result is not None:
                return result
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(2 + attempt)
                continue
            raise e
    return None

# ──────────────────────────────────────────────
# 检查 1: 三大指数日线 — 测试多种数据源
# ──────────────────────────────────────────────
check_section("检查1: 三大指数日线")

index_map = {
    "上证指数": "sh000001",
    "深证成指": "sz399001",
    "创业板指": "sz399006",
}
all_index_ok = True
index_source = ""

# 方案A: 东方财富 (股票行情接口，可能被屏蔽)
for name, symbol in index_map.items():
    try:
        df = retry_call(lambda s=symbol: ak.stock_zh_index_daily_em(symbol=s), f"{name} (东方财富)")
        if df is not None and not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
            dates = set(df["date"].values)
            missing = [d for d in KEY_DATES if d not in dates]
            if not missing:
                index_source = "东方财富"
                ok(f"{name} — 东方财富: {len(df)} 条，关键日期全在")
                continue
    except:
        pass

    # 方案B: 新浪财经指数接口
    try:
        if symbol == "sh000001":
            df = retry_call(lambda: ak.stock_zh_index_daily(symbol="sh000001"), f"{name} (新浪)")
        elif symbol == "sz399001":
            df = retry_call(lambda: ak.stock_zh_index_daily(symbol="sz399001"), f"{name} (新浪)")
        else:
            df = retry_call(lambda: ak.stock_zh_index_daily(symbol="sz399006"), f"{name} (新浪)")
        
        if df is not None and not df.empty:
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y%m%d")
            dates = set(df[date_col].values)
            missing = [d for d in KEY_DATES if d not in dates]
            if not missing:
                if not index_source:
                    index_source = "新浪财经"
                ok(f"{name} — 新浪: {len(df)} 条，关键日期全在")
                continue
    except:
        pass

    all_index_ok = False
    fail(f"{name} — 所有数据源均失败")

if all_index_ok:
    print(f"\n  ✅ 指数日线数据可用 (数据源: {index_source})，择时引擎可支撑")
else:
    print("\n  ❌ 指数日线不可用，择时引擎需要替代方案")

# ──────────────────────────────────────────────
# 检查 2: 全市场涨跌分布
# ────────────────────────⏎──────────────────────
check_section("检查2: 全市场行情")

breadth_ok = False
breadth_source = ""

# 方案A: 东方财富实时行情
try:
    df_spot = retry_call(lambda: ak.stock_zh_a_spot_em(), "东方财富全市场行情")
    if df_spot is not None and not df_spot.empty:
        if "涨跌幅" in df_spot.columns and len(df_spot) > 1000:
            up = (df_spot["涨跌幅"] > 0).sum()
            down = (df_spot["涨跌幅"] < 0).sum()
            breadth_ok = True
            breadth_source = "东方财富"
            ok(f"涨{up}/跌{down}/共{len(df_spot)} — {breadth_source}")
except:
    pass

# 方案B: 如果东方财富不通，用新浪实时行情取样前500只估算市场宽度
if not breadth_ok:
    try:
        # 使用已有的 SourceManager，它能回退到新浪
        dm = DataSourceManager()
        codes_to_test = ["000001", "600519", "300750", "000858", "002230",
                         "600036", "601318", "000063", "300760", "600276"]
        up_count = down_count = 0
        valid_count = 0
        for code in codes_to_test:
            try:
                q = dm.get_realtime_quotes(code)
                if q and q.get("change_percent", 0) > 0:
                    up_count += 1
                elif q:
                    down_count += 1
                valid_count += 1
            except:
                pass
        if valid_count >= 5:
            breadth_ok = True
            breadth_source = "新浪个股采样"
            ok(f"涨{up_count}/跌{down_count}/采样{valid_count} — {breadth_source}")
    except:
        pass

if not breadth_ok:
    print("\n  ⚠️  涨跌分布不可用，市场宽度评分将降级为仅用指数涨跌判断")

# ──────────────────────────────────────────────
# 检查 3: 个股K线 — 测试腾讯数据源
# ─────────────────────────────⏎───────────────
check_section("检查3: 个股K线")

kline_ok = True
kline_source = ""
dm = DataSourceManager()

test_codes = ["000001", "600519", "300750"]

for code in test_codes:
    try:
        df_k = retry_call(
            lambda c=code: dm.get_stock_hist_data(
                c, start_date="20240101", end_date="20260725", adjust="qfq"
            ),
            f"{code} (腾讯数据源)"
        )
        
        if df_k is not None and not df_k.empty:
            if not kline_source:
                kline_source = "腾讯金融(sz/sh)"
            ok(f"{code} — {len(df_k)} 条 (腾讯)")
            continue
        
        # 腾讯数据源失败，回退测试
        try:
            df_k2 = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            if df_k2 is not None and not df_k2.empty:
                if not kline_source:
                    kline_source = "东方财富"
                ok(f"{code} — {len(df_k2)} 条 (东方财富)")
                continue
        except:
            pass
        
        kline_ok = False
        fail(f"{code} — 所有数据源失败")
    except Exception as e:
        kline_ok = False
        fail(f"{code} — {e}")

if kline_ok:
    print(f"\n  ✅ 个股K线可用 (数据源: {kline_source})")

# ──────────────────────────────────────────────
# 检查 4: 北向资金 (已验证过，快速重测)
# ──────────────────────────────────────────────
check_section("检查4: 北向资金")

north_ok = False
try:
    df_north = retry_call(lambda: ak.stock_hsgt_hist_em(symbol="北向资金"), "北向资金")
    if df_north is not None and not df_north.empty:
        north_ok = True
        ok(f"{len(df_north)} 条")
except Exception as e:
    fail(str(e))

# ──────────────────────────────────────────────
# 检查 5: 资金流数据源（降级链: 东财→同花顺→tushare）
# ──────────────────────────────────────────────
check_section("检查5: 资金流数据源（降级链）")

flow_ok = False
flow_source = ""
try:
    df_flow = retry_call(lambda: ak.stock_individual_fund_flow(stock="000001", market="sz"), "东财个股资金流")
    if df_flow is not None and not df_flow.empty:
        flow_ok, flow_source = True, "东方财富逐股历史"
        ok(f"东财逐股历史 {len(df_flow)} 条")
except Exception as e:
    fail(str(e))
if not flow_ok:
    print("  ↳ 东财不可达，尝试同花顺快照源...")
    try:
        df_ths = retry_call(lambda: ak.stock_fund_flow_individual(symbol="即时"), "同花顺个股资金流快照")
        if df_ths is not None and not df_ths.empty:
            flow_ok, flow_source = True, "同花顺全市场快照"
            ok(f"同花顺快照 {len(df_ths)} 条（评分走快照模式）")
    except Exception as e:
        fail(str(e))
if not flow_ok:
    print("  ↳ 全部不可达。可配置 TUSHARE_TOKEN 启用 tushare moneyflow 降级链")

# ──────────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  数据源可用性汇总")
print(f"{'='*60}")
checks = {
    "指数日线 (择时趋势+量能+动量)": (all_index_ok, index_source),
    "全市场行情 (择时宽度评分)": (breadth_ok, breadth_source),
    "个股K线 (均线+波动率)": (kline_ok, kline_source),
    "北向资金 (北向因子)": (north_ok, "东方财富/沪深港通"),
    "资金流 (资金因子)": (flow_ok, flow_source or "全部不可用"),
}
for name, (status, source) in checks.items():
    icon = "✅" if status else "❌"
    src_info = f" [{source}]" if source else ""
    print(f"  {icon}  {name}{src_info}")

passed = sum(1 for v in checks.values() if v[0])
total = len(checks)
print(f"\n  通过: {passed}/{total}")

if passed == total:
    print("\n  🎉 所有数据源可用！择时引擎 + 风控引擎可以正常开发！")
elif passed >= 3:
    missing_items = [name for name, (s, _) in checks.items() if not s]
    print(f"\n  ⚠️  不可用项: {missing_items}")
    print(f"  将使用降级方案: 实时行情用新浪采样替代，指数用个股替代")
else:
    print(f"\n  🔴 多项数据源不可用，建议检查网络/AKShare版本")

sys.exit(0 if passed == total else 1)