"""本地候选池：用全市场快照做粗筛，替代问财初筛（D1 决策）

三层结构（见 `output/NL选股改进方案v2_数据层与使用边界.md` 第 3.4 节 / 图 1）：

    L0 全市场清单        sina_source.market_codes()
    L1 粗筛（本地快照）   只用 source == "snapshot" 的条件
                         （PE / 总市值 / 换手率 / 成交额 / 当日涨跌幅）
    L2 逐票精算          kline / flow / news / score 类条件

为什么要分层
-----------
全市场 5553 只，逐票精算每只 1~3s → 不可接受。
而快照类条件可以**在毫秒级过滤掉绝大部分**（2623 只 → 几十只），
剩下的才送去做 K 线 / 资金流 / 舆情精算。这正是"信息面 Agent 属于第二层"
这条架构约束的落地：它按个股深挖的产能只能覆盖几十只，不能覆盖全市场。

两条不可动摇的规则
----------------
1. **快照不可用 → 明确失败**（`PoolUnavailable`），绝不降级到硬编码精选池。
   拿几十只硬编码股票冒充"按条件选出"比报错更糟。
2. **截断必须如实记录**：L1 后仍可能剩几百只，按成交额降序截断到 `max_precision`，
   并记录 `truncated` 只数，由 UI 展示为"另有 N 只因精算上限未逐一核对"。
   静默丢弃会让用户以为"符合条件的就这么多"。

关于"板块"
--------
`nl_screener.SECTORS` 里的 `科技/金融/消费/新能源/医药` 是**概念板块**，
而快照的 `industry` 列是新浪的 49 个**行业名**（玻璃行业、酿酒行业…）——两套口径。
本模块用 `SECTOR_INDUSTRIES` 做映射；映射不到时**明确报错**，不静默不过滤。

注意 `"创业板"` 同时出现在 `SECTORS` 与 `BOARDS` 里（历史原因，测试也依赖），
所以这里先判断"sector 是否其实是板块名"，是则按 board 过滤——
否则 `创业板` 会当成行业名去匹配，恒为 0 只。
"""

from dataclasses import dataclass, field

import pandas as pd

from data import sina_source
from core.indicator_registry import INDICATORS, SOURCE_SNAPSHOT

# 逐票精算的**上限**：快照粗筛后仍可能剩几百只，而精算每只 1~3s。
# 60 只 ≈ 1~3 分钟，是可接受的量级。
MAX_PRECISION = 60

# 概念板块 → 新浪行业名（实测 49 个行业名的子集）。
# ⚠️ 这是一张**人工映射表**：新浪没有"科技/消费"这种概念分类，只有行业分类。
#    映射不到行业时本模块会明确报错，而不是放行全部——"板块没生效但结果照出"
#    会让用户以为筛过了。
SECTOR_INDUSTRIES = {
    "科技": ("电子信息", "电子器件", "传媒娱乐", "仪器仪表"),
    "金融": ("金融行业",),
    "消费": ("食品行业", "酿酒行业", "商业百货", "酒店旅游", "服装鞋类",
             "家电行业", "家具行业", "纺织行业"),
    "新能源": ("发电设备", "汽车制造", "电力行业"),
    "医药": ("生物制药", "医疗器械"),
}

# 这些值出现在 SECTORS 里，但语义上是**交易所板块**而不是行业
_BOARD_LIKE = {
    "创业板": ("创业板",),
    "科创板": ("科创板",),
    "北交所": ("北交所",),
    "主板": ("主板沪市", "主板深市"),
    "沪市": ("主板沪市",),
    "深市": ("主板深市",),
}


class PoolUnavailable(RuntimeError):
    """候选池不可用（全市场快照取不到）→ 明确失败，**不降级**"""


@dataclass
class PoolResult:
    codes: list = field(default_factory=list)
    snapshot: object = None                # DataFrame（去重后的覆盖域）
    conditions_applied: list = field(default_factory=list)   # 粗筛层已执行
    conditions_deferred: list = field(default_factory=list)  # 需逐票精算
    stage_counts: dict = field(default_factory=dict)
    truncated: int = 0
    sort_note: str = ""
    caveat: str = ""
    notes: list = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.codes

    def to_dict(self) -> dict:
        return {
            "codes": list(self.codes),
            "count": len(self.codes),
            "conditions_applied": list(self.conditions_applied),
            "conditions_deferred": list(self.conditions_deferred),
            "stage_counts": dict(self.stage_counts),
            "truncated": self.truncated,
            "sort_note": self.sort_note,
            "caveat": self.caveat,
            "notes": list(self.notes),
        }


def _apply_condition(df, cond):
    """把一个条件作用到快照列 → 布尔掩码；该指标不在快照里则返回 None。

    ⚠️ 缺数据（NaN）的比较在 pandas 里一律为 False → **缺失自动算"未通过"**，
    与评分卡"缺数据不给分"、核对环节"数据缺失按未通过"保持同一立场。
    """
    if cond.field not in df.columns:
        return None
    col = df[cond.field]
    if cond.op == "is_true":
        return col.fillna(False).astype(bool)
    if cond.op == "between":
        return col.between(cond.value, cond.value2)
    return {
        "lt": col < cond.value,
        "le": col <= cond.value,
        "gt": col > cond.value,
        "ge": col >= cond.value,
        "eq": col == cond.value,
    }[cond.op]


def _mask_for_sector(df, sector: str):
    """概念板块 / 板块名 → 掩码。

    Returns: (掩码, 说明)；无法映射时抛 `PoolUnavailable`（**不静默放行**）。
    """
    if sector in _BOARD_LIKE:
        boards = _BOARD_LIKE[sector]
        return df["board"].isin(boards), f"按交易所板块「{sector}」过滤"

    industries = SECTOR_INDUSTRIES.get(sector)
    if not industries:
        raise PoolUnavailable(
            f"板块「{sector}」暂无行业映射，无法在本地执行。"
            f"可用概念板块：{'、'.join(SECTOR_INDUSTRIES)}；"
            f"或改用交易所板块：{'、'.join(_BOARD_LIKE)}")
    return (df["industry"].isin(industries),
            f"按概念板块「{sector}」过滤（对应行业：{'、'.join(industries)}）")


def build_pool(query, max_precision: int = MAX_PRECISION,
               snapshot=None) -> PoolResult:
    """按条件构建候选池。

    Args:
        query: `nl_screener.NLQuery`
        max_precision: 送入逐票精算的上限
        snapshot: 可注入快照（测试用）；None 则取 `sina_source.get_sector_snapshot()`

    Raises:
        PoolUnavailable: 快照取不到（**明确失败，绝不降级到硬编码池**）
    """
    snap = snapshot if snapshot is not None else sina_source.get_sector_snapshot()
    if snap is None or len(snap) == 0:
        raise PoolUnavailable(
            "全市场快照不可用（新浪行业明细取数失败），无法进行本地选股。"
            "本次**不降级到内置精选池**——那样得到的是硬编码股票而非按条件选出的结果。"
            "请稍后重试；若持续失败，请到「系统配置 → 运行健康」查看数据源状态。")

    df = snap.copy()
    counts = {"覆盖域": len(df)}
    notes = []
    applied, deferred = [], []

    # ── L1：快照类条件（毫秒级）──
    for cond in query.conditions:
        if INDICATORS[cond.field].source != SOURCE_SNAPSHOT:
            deferred.append(cond.scaled())
            continue
        mask = _apply_condition(df, cond)
        if mask is None:                              # 理论上不会发生（已对齐列名）
            deferred.append(cond.scaled())
            continue
        before = len(df)
        df = df[mask]
        counts[f"粗筛·{cond.field}"] = len(df)
        applied.append(f"{cond.scaled()}（{before} → {len(df)} 只）")

    # ── 板块 / ST 过滤 ──
    if query.sector:
        mask, desc = _mask_for_sector(df, query.sector)
        df = df[mask]
        counts["板块过滤后"] = len(df)
        notes.append(desc)
    if query.board and query.board != query.sector:
        boards = _BOARD_LIKE.get(query.board)
        if boards:
            df = df[df["board"].isin(boards)]
            counts["交易所板块过滤后"] = len(df)
            notes.append(f"按交易所板块「{query.board}」过滤")
    if query.exclude_st:
        st = df["name"].astype(str).str.upper().str.contains("ST", na=False)
        removed = int(st.sum())
        df = df[~st]
        counts["剔除 ST 后"] = len(df)
        if removed:
            notes.append(f"已剔除 {removed} 只 ST/*ST 股")

    counts["粗筛后候选"] = len(df)

    # ── 截断（**必须如实记录**）──
    truncated = 0
    sort_note = ""
    if len(df) > max_precision:
        df = df.sort_values("amount", ascending=False, na_position="last")
        truncated = len(df) - max_precision
        df = df.head(max_precision).reset_index(drop=True)
        sort_note = f"粗筛命中 {truncated + max_precision} 只，按成交额降序取前 {max_precision} 只"

    codes = [str(c) for c in df["code"].tolist()]
    counts["送入精算"] = len(codes)

    caveat = sina_source.snapshot_note(snap)
    if truncated:
        notes.append(
            f"⚠️ 另有 {truncated} 只满足粗筛条件，但因逐票精算上限"
            f"（{max_precision} 只）未逐一核对——不是「不符合」，是「还没算」。")

    return PoolResult(codes=codes, snapshot=df,
                      conditions_applied=applied, conditions_deferred=deferred,
                      stage_counts=counts, truncated=truncated,
                      sort_note=sort_note, caveat=caveat, notes=notes)


def snapshot_row_for(result: PoolResult, code: str):
    """在候选池快照里取某只股票的行（找不到返回 None）"""
    try:
        hit = result.snapshot[result.snapshot["code"] == str(code)]
        return None if hit.empty else hit.iloc[0]
    except Exception:                                 # noqa: BLE001
        return None


def pool_note(result: PoolResult) -> str:
    """给 UI 的一句候选域说明（含覆盖域与截断）"""
    parts = [result.caveat]
    if result.sort_note:
        parts.append(result.sort_note)
    return "；".join(p for p in parts if p)
