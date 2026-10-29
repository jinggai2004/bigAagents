"""外盘（美股）隔夜行情 —— 供 A 股早盘参考

为什么单独一个模块，而不是塞进 `core/market_detector.py`：

  1. **取数口径要能独立验证**。"数据日期"在这里是核心字段 —— 周一早上取到的是
     上周五的美股，这是正常的（周末无交易），但如果不把它显式返回，下游就会
     把 3 天前的行情当成"昨夜"来解读。
  2. **不该悄悄改变选股评分**。`MarketDetector` 的 14 个指标各有权重，新增一个
     指标必须重新归一化，否则等于变相放大其他指标 —— 那是需要单独评审的改动，
     不能用"顺手加一个"的方式做掉。

实证依据（决定"按阈值分级"而不是"线性给分"）：

  - 沪深300 vs 标普500 日度 Pearson 仅 **0.10**（雪球 DCC-GARCH，2024–2026）；
    中银证券管涛周度均值 0.24；国联民生 2000 年至今 21%→27%。
  - 近 10 年统计（2016–2026，纳指100 vs 上证）：纳指单日跌 ≥4% 的 13 次里，
    A 股次日**收红仍占 53.8%**；涨 ≥4% 的 14 次里，A 股次日大涨仅 14.3%；
    **±2% 以内，A 股次日涨跌几乎对半分，外盘没有决定力**。
  - 分层联动：半导体 0.72 / AI算力 0.68 / 光伏 0.65 高联动；
    白酒、银行、地产 0.12–0.21 近乎独立。
  - 危机期才急剧增强：美股单周跌超 7% 的 9 次中 A 股跟跌 6 次（相关 0.77）；
    VIX 飙升期 6 个月滚动相关跳升至 0.5+。

→ 结论：外盘是**开盘情绪输入**，且只有大幅波动时才有实际影响。
   所以本模块的输出是「涨跌幅 + 分级 + 数据日期」，而不是一个想当然的分数。
"""

import logging
import time
from datetime import datetime

_logger = logging.getLogger(__name__)

# 三大指数（新浪美股接口）。纳指对应 A 股科技/成长，道指对应传统价值
US_INDEXES = (
    ("纳斯达克", ".IXIC"),
    ("标普500", ".INX"),
    ("道琼斯", ".DJI"),
)

# 分级阈值：(绝对涨跌幅下限%, level, 中文)
# 依据：±2% 以内对 A 股次日无决定力 → 只有 ≥2% 才算"值得提示"
LEVELS = (
    (4.0, "extreme", "剧烈波动"),
    (2.0, "notable", "明显波动"),
    (1.0, "mild", "小幅波动"),
    (0.0, "neutral", "无异动"),
)

_CACHE: dict = {"ts": 0.0, "data": None}
CACHE_TTL_OK = 1800        # 成功结果缓存 30 分钟（美股一天只更新一次）
CACHE_TTL_FAIL = 300       # 失败结果缓存 5 分钟，避免网络故障时反复长时间重试


def _classify(change_pct: float) -> tuple[str, str]:
    """把涨跌幅分级。传入绝对值。"""
    for threshold, level, text in LEVELS:
        if change_pct >= threshold:
            return level, text
    return "neutral", "无异动"


def _fetch_one(name: str, symbol: str) -> dict | None:
    """取单个指数最近两根日线，算涨跌幅。数据不足返回 None。"""
    import akshare as ak

    df = ak.index_us_stock_sina(symbol=symbol)
    if df is None or len(df) < 2:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    try:
        close = float(last["close"])
        prev_close = float(prev["close"])
    except (KeyError, TypeError, ValueError):
        return None
    if prev_close <= 0 or close <= 0:
        return None
    return {
        "name": name,
        "symbol": symbol,
        "data_date": str(last.get("date"))[:10],
        "close": round(close, 2),
        "change_pct": round((close / prev_close - 1) * 100, 2),
    }


def get_overnight_us(force: bool = False) -> dict:
    """获取最近一个美股交易日的三大指数表现。

    Returns:
        成功:
            {
              "data_date": "2026-09-25",     # **美股**那一根 bar 的日期
              "query_date": "2026-09-28",    # 本次查询日期（两者正交，都要给）
              "days_ago": 3,                 # 相隔天数（周末/长假会>1，属正常）
              "indexes": [{name, symbol, data_date, close, change_pct, level, level_text}],
              "max_abs_change_pct": 0.48,
              "level": "neutral", "level_text": "无异动",
              "note": "...",
              "fetch_time": "2026-09-28 17:10:00",
            }
        取不到（**必须显式标记，不能返回空 dict 让下游猜**）:
            {"unavailable": True, "query_date": ..., "note": "失败原因", ...}
    """
    now = time.time()
    cached = _CACHE.get("data")
    if not force and cached:
        ttl = CACHE_TTL_OK if not cached.get("unavailable") else CACHE_TTL_FAIL
        if now - _CACHE["ts"] < ttl:
            return cached

    today = datetime.now().strftime("%Y-%m-%d")
    indexes, errors = [], []
    for name, symbol in US_INDEXES:
        try:
            row = _fetch_one(name, symbol)
        except Exception as e:                              # noqa: BLE001 - 逐指数降级
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:60]}")
            continue
        if row is None:
            errors.append(f"{name}: 返回数据不足 2 根")
            continue
        level, level_text = _classify(abs(row["change_pct"]))
        row["level"], row["level_text"] = level, level_text
        indexes.append(row)

    if not indexes:
        result = {
            "unavailable": True,
            "query_date": today,
            "note": "外盘数据获取失败（" + "；".join(errors) + "）",
            "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _CACHE.update(ts=now, data=result)
        _logger.warning("[外盘] 获取失败: %s", errors)
        return result

    # 用绝对值最大的那个指数决定整体等级（最坏情况优先，避免"纳指暴跌但道指平稳"被平均掉）
    driver = max(indexes, key=lambda r: abs(r["change_pct"]))
    max_abs = abs(driver["change_pct"])
    level, level_text = _classify(max_abs)

    # data_date 取各指数中最新的那根
    data_date = max(r["data_date"] for r in indexes)
    days_ago = None
    try:
        days_ago = (datetime.strptime(today, "%Y-%m-%d")
                    - datetime.strptime(data_date, "%Y-%m-%d")).days
    except ValueError:
        pass

    if level == "neutral":
        note = (f"上一美股交易日为 {data_date}，三大指数波动均在 1% 以内。"
                "历史统计：外盘 ±2% 以内对 A 股次日涨跌几乎无决定力，"
                "不必据此调整仓位。")
    elif level == "mild":
        note = (f"上一美股交易日为 {data_date}，最大波动 {max_abs:.2f}%（{driver['name']}）。"
                "影响通常限于 A 股开盘情绪，10:00 后由内资主导。")
    elif level == "notable":
        note = (f"上一美股交易日为 {data_date}，{driver['name']} 波动 {max_abs:.2f}%。"
                "近 10 年统计：美股跌 2–4% 时 A 股次日下跌概率约 62%（多为小跌）；"
                "涨 2–4% 时上涨概率约 51%，**高开回落是常态**。"
                "关注半导体/AI算力/消费电子等 0.68–0.72 高联动板块，"
                "银行、白酒、中药等内需板块通常走独立行情。")
    else:
        note = (f"⚠️ 上一美股交易日为 {data_date}，{driver['name']} 波动 {max_abs:.2f}%，属剧烈波动。"
                "历史上此类情形 A 股**几乎必然低开**；但近 10 年统计显示，"
                "纳指单日跌 ≥4% 后 A 股次日收红仍占 53.8%，**低开不等于全天走弱**。"
                "若同时伴随 VIX 飙升或全球系统性事件，联动会显著增强（跟跌概率大幅上升）。")

    if days_ago is not None and days_ago > 4:
        note += f"　⚠️ 该 bar 距今 {days_ago} 天（长假？），请确认是否为最新交易日数据。"

    result = {
        "data_date": data_date,
        "query_date": today,
        "days_ago": days_ago,
        "indexes": indexes,
        "max_abs_change_pct": round(max_abs, 2),
        "driver": driver["name"],
        "level": level,
        "level_text": level_text,
        "note": note,
        "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _CACHE.update(ts=now, data=result)
    _logger.info("[外盘] %s %s，最大波动 %.2f%%（%s）",
                 data_date, level_text, max_abs, driver["name"])
    return result


def reset_cache():
    """清缓存（测试用）。"""
    _CACHE.update(ts=0.0, data=None)
