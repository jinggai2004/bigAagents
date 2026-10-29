# -*- coding: utf-8 -*-
"""投资者关注度（attention）数据源 —— 东财人气榜**逐日历史**。

## 为什么值得单独建模块（2026-10-13 实测更正）

项目此前把"舆情"整体归入**无法验证** —— `core/factor_analysis.UNAVAILABLE` 里写着
`news_positive: 历史舆情无稳定免费源`，因子报告也写"58.8% 的权重（主力资金/PE/舆情）
当前根本无法验证"。

**该判断对「文本情绪」成立，但对「关注度 / 行为情绪」不成立。** 实测
`ak.stock_hot_rank_detail_em(symbol)`：

    366 行（整整一年），列 = 时间 / 排名 / 证券代码 / 新晋粉丝 / 铁杆粉丝
    2025-10-13   1828  SZ000665  0.2536  0.7464
    2025-10-14   1808  SZ000665  0.2229  0.7771

⇒ **逐日、无需登录态、可直接回填一年历史**（不是"从今天起慢慢攒"）。
所以"情绪因子无法验证"这句话要改成：**情绪因子从今天起可以验证了**。

## 两条必须随数据一起声明的口径（不可省略）

1. ⚠️ **`新晋粉丝 / 铁杆粉丝` 的定义未核实**：官方接口文档**没有**给出定义。
   实测值域 0~1（**看似占比**），但本模块**只原样存储，不在代码里假设其含义**。
   在向数据源确认之前，任何"这是占比"的说法都不能当作事实。
2. ⚠️ **覆盖不均**：人气榜只覆盖"有点击/关注"的股票；冷门股、次新股可能无数据。
   ⇒ **取不到就是取不到**（返回空），**绝不用 0 填充** —— 0 会被下游读成"关注度极低"，
   而真实情况是"没有数据"，两者含义完全相反。
"""

import json
import logging
import os
import time

import pandas as pd

_logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp", "attention")

# 接口返回的中文列 → 本模块标准列
_COLMAP = {"时间": "date", "排名": "rank",
           "新晋粉丝": "new_fans", "铁杆粉丝": "loyal_fans"}
OUT_COLUMNS = ("date", "rank", "new_fans", "loyal_fans")


def _cache_path(code: str) -> str:
    return os.path.join(_CACHE_DIR, f"{code}.json")


def to_symbol(code: str) -> str:
    """把 6 位裸码规范成该接口要求的**带交易所前缀**写法。

    ⚠️ 实测（2026-10-13 23:40）：`stock_hot_rank_detail_em` 只接受
    `SZ000665` 这种带前缀的形式；传裸码 `000665` **不报错但返回结构异常**
    （1 列），随后在 `rename` 处炸出 `Length mismatch` —— 属于"接口不报错、
    数据不对"的静默形态，所以必须在入口就规范化，而不是靠调用方自觉。
    """
    s = str(code).strip().upper()
    if s[:2] in ("SH", "SZ", "BJ"):
        return s
    s = s.zfill(6)
    if s[0] == "6" or s[0] == "5" or s.startswith("9"):
        return "SH" + s
    if s[0] in ("0", "3", "1", "2"):
        return "SZ" + s
    if s[0] in ("4", "8"):
        return "BJ" + s
    return "SZ" + s


def fetch_attention_history(code: str, retries: int = 2) -> pd.DataFrame | None:
    """取单只股票的**人气榜逐日历史**（约 366 行）。

    Returns:
        列 = `date / rank / new_fans / loyal_fans` 的 DataFrame；
        **取不到返回 None**（不返回空 DataFrame、更不返回 0 行填充的结果），
        由调用方决定如何声明"缺失"。
    """
    import akshare as ak

    sym = to_symbol(code)
    last_err = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            df = ak.stock_hot_rank_detail_em(symbol=sym)
            if df is None or len(df) == 0:
                return None
            df = df.rename(columns=_COLMAP)
            keep = [c for c in OUT_COLUMNS if c in df.columns]
            if "date" not in keep:
                _logger.warning("[关注度] %s 返回列不符: %s", sym, list(df.columns))
                return None
            out = df[keep].copy()
            out["date"] = out["date"].astype(str).str.slice(0, 10)
            for c in ("rank", "new_fans", "loyal_fans"):
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce")
            return out.sort_values("date").reset_index(drop=True)
        except Exception as e:                      # noqa: BLE001
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * attempt)
    _logger.warning("[关注度] %s 取数失败（%d 次）: %s", sym, retries, last_err)
    return None


def load_cached(code: str, max_age_days: float = 1.0) -> pd.DataFrame | None:
    """读本地缓存；过期或不存在返回 None。"""
    p = _cache_path(code)
    if not os.path.exists(p):
        return None
    try:
        age_days = (time.time() - os.path.getmtime(p)) / 86400.0
        if age_days > max_age_days:
            return None
        d = json.load(open(p, encoding="utf-8"))
        return pd.DataFrame(d) if d else None
    except Exception as e:                           # noqa: BLE001
        _logger.warning("[关注度] %s 缓存读取失败: %s", code, e)
        return None


def save_cache(code: str, df: pd.DataFrame) -> bool:
    """写本地缓存（原子写，避免半截文件被当成有效数据）。"""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        p = _cache_path(code)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=False)
        os.replace(tmp, p)
        return True
    except Exception as e:                           # noqa: BLE001
        _logger.warning("[关注度] %s 缓存写入失败: %s", code, e)
        return False


def fetch_with_cache(code: str, max_age_days: float = 1.0,
                     force: bool = False) -> pd.DataFrame | None:
    """优先读缓存，缺失/过期再取数并回写。"""
    if not force:
        cached = load_cached(code, max_age_days)
        if cached is not None:
            return cached
    df = fetch_attention_history(code)
    if df is not None:
        save_cache(code, df)
    return df


def load_map(codes, start: str = None, end: str = None, *, force: bool = False,
             sleep: float = 0.12, progress_cb=None) -> dict:
    """批量取多只股票的关注度历史 → `{code: DataFrame}`。

    ⚠️ **取不到的股票不会出现在返回字典里**（不是空 DataFrame）——
    调用方据此区分"没有数据"与"关注度为 0"，这正是本项目"缺就是缺"的要求。

    Args:
        codes: 股票代码集合
        start / end: 可选区间过滤（`YYYY-MM-DD`）
        force: 忽略缓存重新取数
        sleep: 每次请求间隔（东财公开接口，主动限速，避免被判异常流量）
    """
    codes = [str(c).strip() for c in codes if str(c).strip()]
    total = max(1, len(codes))
    out = {}
    miss = []
    for i, code in enumerate(codes, 1):
        if progress_cb:
            progress_cb({"current": i, "total": total, "code": code,
                         "percent": i / total * 100})
        df = fetch_with_cache(code, force=force)
        if df is None or df.empty:
            miss.append(code)
            continue
        if start:
            df = df[df["date"] >= start]
        if end:
            df = df[df["date"] <= end]
        if not df.empty:
            out[code] = df.reset_index(drop=True)
        else:
            miss.append(code)
        if sleep:
            time.sleep(sleep)
    _logger.info("[关注度] 取到 %d 只，缺失 %d 只（缺失样本：%s）",
                 len(out), len(miss), miss[:8])
    return out


def coverage_note(got: int, asked: int) -> str:
    """给报告用的覆盖率声明（**必须随结果展示**，同 `market_snapshot` 的 caveat 口径）。"""
    if asked <= 0:
        return ""
    pct = got / asked * 100
    return (f"关注度数据覆盖 {got}/{asked} 只（{pct:.0f}%）。"
            f"⚠️ 人气榜只覆盖有点击/关注的股票，**未覆盖的票是「无数据」而非「关注度为 0」**；"
            f"另：`新晋粉丝/铁杆粉丝` 的官方定义未核实（实测值域 0~1，看似占比）。")
