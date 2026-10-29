# -*- coding: utf-8 -*-
"""申万 L1 行业成分（B2 补充，2026-10-29，回测验证设计 3.2 的前置数据工程）。

为什么需要：面板 16 列没有 industry（`full_market` 的 20 个键里也没有），
唯一市值/行业源 `market_snapshot` 只覆盖十几个交易日 —— Brinson 归因需要
"每只票属于哪个行业"，只能一次联网回填落本地。

⚠️⚠️ 前视偏差（设计文档 §3.2 评审提醒）：这里落的是**今天的成分快照**。
用今天的行业归属解释 2016–2026 的收益 = 假设"行业归属十年不变"，
行业中性/配置类结论会偏乐观。`load` 返回值带 `snapshot_date`，
消费方（`core/attribution.py`）必须把它标进结果（degraded 语义），
**不当成精确归因**。`join_date`（计入日期）保留为部分 point-in-time 线索，
但它只反映最后一次计入，历史进出无法还原。

口径纪律：只用申万 L1（31 类，2021 修订版）。**禁止混入**新浪 49 类 /
tushare 细分 —— 行业口径混进同一个映射会让下游静默失效（老坑）。

设计（与 `index_bars` 同风格）：
- `backfill_industry_swa()`：全量拉 31 个 L1 成分 → 覆盖写 parquet（幂等，量小）；
- `load_industry_swa()`：**不联网**；本地没有 ⇒ None（缺就是缺）。
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

import pandas as pd

_logger = logging.getLogger(__name__)

# 数据根必须**运行时**解析：打包后 `__file__` 位于只读包 `_internal/` 里，
# 用它推导会指向空目录 ⇒ 读到空数据（2026-10-25 实测）。规则同 index_bars。
from config.paths import data_root as _paths_data_root  # noqa: E402

PROJECT_ROOT = _paths_data_root()
INDUSTRY_PATH = os.path.join(PROJECT_ROOT, "data", "panel", "industry_swa_l1.parquet")

# 申万 L1（2021 修订版，31 个）。代码与名称写死——成分集合稳定，
# 动态发现接口（index_realtime_sw）反而引入另一条失败路径。
SW_L1 = {
    "801010": "农林牧渔", "801030": "基础化工", "801040": "钢铁",
    "801050": "有色金属", "801080": "电子", "801110": "家用电器",
    "801120": "食品饮料", "801130": "纺织服饰", "801140": "轻工制造",
    "801150": "医药生物", "801160": "公用事业", "801170": "交通运输",
    "801180": "房地产", "801200": "商贸零售", "801210": "社会服务",
    "801230": "综合", "801710": "建筑材料", "801720": "建筑装饰",
    "801730": "电力设备", "801740": "国防军工", "801750": "计算机",
    "801760": "传媒", "801770": "通信", "801780": "银行",
    "801790": "非银金融", "801880": "汽车", "801890": "机械设备",
    "801950": "煤炭", "801960": "石油石化", "801970": "环保",
    "801980": "美容护理",
}


def _fetch_one(symbol: str):
    """单行业成分（akshare）。独立成函数供测试注入。"""
    import akshare as ak
    return ak.index_component_sw(symbol=symbol)


def backfill_industry_swa(retry: int = 1, sleep_secs: float = 0.5) -> dict:
    """全量回填 31 个申万 L1 成分 → 覆盖写 `industry_swa_l1.parquet`。

    Returns: {industries_ok, industries_failed, stocks, failed: [...],
             snapshot_date, path}
    失败的行业跳过并留痕（不抛异常——部分成功也比零强，下次重跑补齐）。
    """
    rows: list[dict] = []
    failed: list[str] = []
    for i, (code, name) in enumerate(SW_L1.items()):
        df = None
        for attempt in range(retry + 1):
            try:
                df = _fetch_one(code)
                break
            except Exception as e:                      # noqa: BLE001
                if attempt < retry:
                    time.sleep(sleep_secs)
                else:
                    failed.append(f"{code}({name}): {type(e).__name__}: {e}")
                    _logger.warning("[行业回填] %s(%s) 失败：%s", code, name, e)
        if df is None or getattr(df, "empty", True):
            continue
        code_col = "证券代码" if "证券代码" in df.columns else df.columns[1]
        join_col = "计入日期" if "计入日期" in df.columns else None
        for _, r in df.iterrows():
            stock = str(r[code_col]).zfill(6)
            rows.append({"stock_code": stock,
                         "industry_code": code,
                         "industry_name": name,
                         "join_date": (str(r[join_col])[:10] if join_col and
                                       pd.notna(r.get(join_col)) else None)})
        if i < len(SW_L1) - 1:
            time.sleep(sleep_secs)
    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    out = {"industries_ok": len(SW_L1) - len(failed), "industries_failed": len(failed),
           "stocks": len({r["stock_code"] for r in rows}), "failed": failed,
           "snapshot_date": snapshot_date, "path": INDUSTRY_PATH}
    if rows:
        df_out = pd.DataFrame(rows)
        df_out["snapshot_date"] = snapshot_date
        os.makedirs(os.path.dirname(INDUSTRY_PATH), exist_ok=True)
        df_out.to_parquet(INDUSTRY_PATH, index=False)
        _logger.info("[行业回填] %s 行业 / %s 只票 → %s",
                     out["industries_ok"], out["stocks"], INDUSTRY_PATH)
    return out


def load_industry_swa() -> dict | None:
    """读本地行业映射（**不联网**）。

    Returns:
        {"industry": {stock_code: 行业名}, "snapshot_date": str, "rows": int}
        文件不存在/损坏 ⇒ None（缺就是缺——消费方据此标 degraded）。
    """
    if not os.path.exists(INDUSTRY_PATH):
        return None
    try:
        df = pd.read_parquet(INDUSTRY_PATH)
    except Exception as e:                              # noqa: BLE001
        _logger.warning("[行业回填] 本地行业文件读取失败（按无数据处理）：%s", e)
        return None
    if df.empty or "stock_code" not in df.columns:
        return None
    snapshot = str(df["snapshot_date"].iloc[0]) if "snapshot_date" in df.columns else ""
    industry = {str(r.stock_code): str(r.industry_name) for r in df.itertuples()}
    return {"industry": industry, "snapshot_date": snapshot, "rows": int(len(df))}
