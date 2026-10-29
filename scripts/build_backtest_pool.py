#!/usr/bin/env python3
"""构建回测股票池（v3 / P0-2）——把手工挑的 15 只白马换成 100+ 只行业分层抽样。

## 为什么必须换（`早期文档/回测模块修改建议报告.md` P0-2）

原池是 15 只手挑标的（茅台/宁德/迈瑞/平安…），而它们恰恰**因为** 2024–2025 表现好
才会在 2026 年进入自选名单——这是**前视选样 + 幸存者偏差**的教科书案例：
用它们回测那段时间等于开卷考试，收益/夏普被系统性高估，且偏差量级无法从回测内部察觉。

报告的三个方案里有两条实测**在本机不可行**：

  - "最优：历史指数成分股"——akshare 只有 `index_stock_cons_csindex`（**当期**成分），
    没有历史成分接口；
  - "次优：已落库 snapshot 取当日全市场"——`daily_snapshots` 0 行，
    `screen_picks` / `fund_flow_snapshot` 只覆盖 2026-09-25~09-30，够不到 2024–2025。

因此采用报告里的"最低限度"方案：**扩大到 100+ 只 + 行业均衡 + 随机对照**。

## 数据来源与残留偏差（必须一起公开）

行业分类取新浪行业（`stock_sector_spot` + `stock_sector_detail`，49 个行业；
东方财富的板块接口在本机被代理拦截）。**分类与成分都是"当前"的**，
所以：

  - ✅ 消除了"15 只手挑"这一层最严重的选样偏差；
  - ✅ 选择规则客观（按行业分层、代码排序确定抽样），任何人重跑得到同一份池；
  - ❌ **幸存者偏差未完全消除**：2024 年之前退市的票不在今天的行业成分里。
    报告与回测输出都必须写明这一条，且结论应与随机对照（`--baseline`）一起读。

用法:
  py scripts/build_backtest_pool.py                 # 生成 data/backtest_pool.json（约 120 只）
  py scripts/build_backtest_pool.py --size 150      # 指定规模
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

OUT_FILE = _PROJECT_ROOT / "data" / "backtest_pool.json"

# 市值下限（单位：万元，与新浪 `mktcap` 字段一致）——剔除微盘，避免"1 万手就能拉涨停"
MIN_MKTCAP_WAN = 200_000          # 20 亿元
# 单只标的的市值上限：不设。池子本就该有大小盘之分，否则又是另一种选样。

EXCLUDE_PREFIX = ("2", "4", "8", "9")   # B 股 / 北交所 / 老三板：流动性与交易规则差异大


def _log(msg: str):
    print(msg, flush=True)


def fetch_industry_map() -> dict:
    """返回 {code: 行业名}（新浪行业分类，全市场）"""
    import akshare as ak

    t0 = time.time()
    spot = ak.stock_sector_spot(indicator="新浪行业")
    industries = [(str(r["label"]), str(r["板块"])) for _, r in spot.iterrows()]
    _log(f"  行业板块: {len(industries)} 个（{time.time() - t0:.1f}s）")

    mapping: dict[str, str] = {}
    names: dict[str, str] = {}
    caps: dict[str, float] = {}
    for i, (label, ind_name) in enumerate(industries, 1):
        try:
            detail = ak.stock_sector_detail(sector=label)
        except Exception as e:                          # noqa: BLE001 - 单个行业失败不影响整体
            _log(f"    [{i}/{len(industries)}] {ind_name}: 拉取失败({e})，跳过")
            continue
        for _, r in detail.iterrows():
            code = str(r.get("code") or "").strip()
            name = str(r.get("name") or "").strip()
            if not (code.isdigit() and len(code) == 6):
                continue
            if code[0] in EXCLUDE_PREFIX:
                continue
            if "ST" in name.upper() or "退" in name:
                continue
            try:
                cap = float(r.get("mktcap") or 0)
            except (TypeError, ValueError):
                cap = 0.0
            if cap < MIN_MKTCAP_WAN:
                continue
            mapping[code] = ind_name
            names[code] = name
            caps[code] = cap
        if i % 10 == 0:
            _log(f"    ... {i}/{len(industries)} 个行业，累计 {len(mapping)} 只")
    _log(f"  全市场筛后: {len(mapping)} 只 / {len(set(mapping.values()))} 个行业"
         f"（{time.time() - t0:.1f}s）")
    return mapping, names, caps


def sample_pool(mapping: dict, target: int) -> list:
    """确定性行业分层抽样：每个行业先保 1 只，其余按全池等距补齐。

    不用 `random`——同一个市场状态重跑必须得到同一份池，否则回测不可复现。
    """
    by_ind: dict[str, list] = {}
    for code, ind in mapping.items():
        by_ind.setdefault(ind, []).append(code)
    for codes in by_ind.values():
        codes.sort()

    picked = {sorted(codes)[0] for codes in by_ind.values()}      # 每行业保底 1 只
    rest = sorted(set(mapping) - picked)
    need = max(0, target - len(picked))
    if need and rest:
        step = len(rest) / need
        for i in range(need):
            picked.add(rest[int(i * step)])
    elif need and not rest:
        _log(f"  ⚠️  可补样本不足，池规模停在 {len(picked)}（目标 {target}）")
    return sorted(picked)


def main():
    parser = argparse.ArgumentParser(description="构建回测股票池（P0-2）")
    parser.add_argument("--size", type=int, default=120, help="目标池规模（默认 120）")
    parser.add_argument("--out", default=str(OUT_FILE))
    args = parser.parse_args()

    _log("构建回测股票池（P0-2）")
    mapping, names, caps = fetch_industry_map()
    if not mapping:
        _log("❌ 行业数据为空，无法建池（检查网络/代理）")
        return 1
    if len(mapping) < 100:
        _log(f"❌ 筛后仅 {len(mapping)} 只，不足以构成 100+ 池；请检查是否被代理拦截")
        return 1

    pool = sample_pool(mapping, args.size)
    industries = sorted({mapping[c] for c in pool})
    payload = {
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "akshare stock_sector_spot / stock_sector_detail（新浪行业分类）",
        "pool_size": len(pool),
        "industry_count": len(industries),
        "min_mktcap_wan": MIN_MKTCAP_WAN,
        "caveat": ("行业分类与成分均为**当期**数据；已消除'手挑 15 只'这一层选样偏差，"
                   "但**幸存者偏差仍在**（2024 年前退市的票不在今天的成分里）。"
                   "结论需与随机对照（--baseline）一起解读。"),
        "pool": pool,
        "industry_map": {c: mapping[c] for c in pool},
        "names": {c: names[c] for c in pool},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(f"\n✅ 已写入 {out}")
    _log(f"   池规模 {len(pool)} 只 / 行业 {len(industries)} 个")
    _log(f"   样例: " + ", ".join(f"{c} {names[c]}({mapping[c]})" for c in pool[:6]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
