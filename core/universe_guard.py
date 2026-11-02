# -*- coding: utf-8 -*-
"""回测宇宙有效性守卫：上市满一年 + 北交所只在开市后纳入。

为什么必须有这一层（2026-10-31 实测踩到）
──────────────────────────────────────────
61 因子 A/B 实验里，"低流动性"类因子在 2019~2023 年的 top20 里**北交所（920/8x/4x）
占 21%~55%**，而 2024~2026 只占 7%~16%。再往下查，2020-11 的 top5 全是 920xxx，
那几只票当期只有 **14~19 根 bar 且处于停牌**，`fwd20` 报 **+142%/+187%**——
这是**复牌跳空**造成的伪收益，不是策略能力。

北交所 **2021-11-15** 才开市。在此之前 920xxx 的 K 线是原新三板历史映射过来的，
口径与连续竞价市场不同，**不能和主板混在一个截面里排序**。

两条规则
────────
1. `min_history_days`（默认 365 自然日）：上市不满一年的票不进候选——
   次新股前 20 日没有涨跌停约束（首日无限制），波动与换手不可比；
2. 北交所代码（`920/83/87/88/43/82` 开头）在 `BSE_START`（2021-11-15）之前**不纳入**。

⚠️ 这条守卫只能用在**回测/研究**侧。线上选股用的是当期实时快照，
   本来就只看当下可交易的票，不需要也不应该套"上市满一年"（会误杀正常次新）。
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path

import pandas as pd

_logger = logging.getLogger("universe_guard")
# ⚠️⚠️ 数据/可写路径**必须**走 config.paths（铁律）：frozen 后 `__file__` 在只读
#    的 `_internal/` 里，用 `__file__` 拼 data/ 或 temp/ 会静默读到空目录
#    （本项目 2026-10-25 有过 12 个模块同时踩的事故，`TestNoFileBasedDataRoot` 就是这么来的）。
from config.paths import data_root as _data_root  # noqa: E402

BY_CODE_DIR = Path(_data_root()) / "data" / "panel" / "by_code"
CACHE = Path(_data_root()) / "temp" / "listing_table.parquet"

_empty_warned = False      # 清单为空只告警一次，避免逐行刷屏

BSE_START = pd.Timestamp("2021-11-15")      # 北交所开市日（此前 920xxx 数据不可比）
BSE_PREFIXES = ("920", "83", "87", "88", "43", "82")
DEFAULT_MIN_HISTORY_DAYS = 365


def is_bse(code: str) -> bool:
    return str(code).startswith(BSE_PREFIXES)


def build_listing_table(force: bool = False) -> pd.DataFrame:
    """每只票的首末交易日 + 行数。

    只读 parquet **footer 元数据**（不解码数据），5,830 只约 40 秒；
    落盘到 `temp/listing_table.parquet` 后复用。
    """
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)
    import pyarrow.parquet as pq

    rows = []
    failed = 0
    for f in sorted(glob.glob(str(BY_CODE_DIR / "*.parquet"))):
        code = Path(f).stem
        try:
            md = pq.ParquetFile(f).metadata
        except Exception as e:                  # noqa: BLE001  单个 parquet 损坏不该炸整张表
            # 但**必须留痕**：静默跳过会让"这只票没进清单"变成不可解释的空洞
            _logger.warning("[universe_guard] 跳过无法读取的 %s: %s: %s",
                            Path(f).name, type(e).__name__, e)
            failed += 1
            continue
        if md.num_rows == 0 or md.num_row_groups == 0:
            continue
        st = md.row_group(0).column(0).statistics    # 第 0 列是 date
        if st is None:
            continue
        rows.append({"code": code, "first_date": pd.Timestamp(st.min),
                     "last_date": pd.Timestamp(st.max), "rows": int(md.num_rows)})
    df = pd.DataFrame(rows)
    if failed:
        _logger.warning("[universe_guard] 共 %d 个 parquet 读不出元数据（已跳过）", failed)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def comparable_since(code: str, *, listing: pd.DataFrame | None = None,
                     min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
                     exclude_bse: bool = True) -> pd.Timestamp | None:
    """该票**从哪天起**才与同期截面可比。`None` = 永远不可比（清单里没有）。

    两个约束取较晚者：
      1. `first_date + min_history_days`（上市满一年）；
      2. 北交所代码的 `BSE_START`（2021-11-15 开市前数据是新三板映射）。

    ⚠️ 这是**唯一一处**把"可比"翻译成时间点的地方：月度截面（`eligible`）
       与日频回测（`trim`）都走它，避免"月度按 365 天、日频按 250 个交易日"
       这类口径分裂——那种分裂不会报错，只会让两边结论对不上。
    """
    if listing is None:
        listing = build_listing_table()
    row = listing[listing["code"].astype(str) == str(code)]
    if row.empty:
        return None
    first = pd.Timestamp(row["first_date"].iloc[0])
    since = first + pd.Timedelta(days=int(min_history_days))
    if exclude_bse and is_bse(code) and since < BSE_START:
        since = BSE_START
    return since


def comparable_mask(df: pd.DataFrame, *, date_col: str = "date",
                    code_col: str = "code", code: str | None = None,
                    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
                    listing: pd.DataFrame | None = None,
                    exclude_bse: bool = True) -> pd.Series:
    """逐行判定"这一行（这只票在这一天）是否可比"。

    Args:
        code: 单票表（日线）直接传代码，避免调用方为了满足 `code_col` 而造列。
        code_col: 多票长表（月度截面）用这一列。

    Returns:
        与 df 同索引的 bool。**没有清单信息的票一律 False**——宁可少算，
        不拿来源不明的历史去排名。
    """
    global _empty_warned

    if listing is None:
        listing = build_listing_table()
    if listing is None or len(listing) == 0:
        # ⚠️⚠️ 清单整体为空 = **守卫的依据不存在**（`data/panel/by_code` 读不到，
        #    例如用户把 exe 单独拷到没有祖先 data/ 的目录）。
        #    此时若照常判"不可比"，会把**全部**历史判为不可比 ⇒ 研究面板 IC、
        #    条件回测、盘后回测集体返回空，而界面上看不出原因。
        #    "缺就是缺"说的是数据缺失不许假装有；这里是**判据缺失**，
        #    正确表现是明确告警 + 放行（fail-open），而不是假装一切不可比。
        if not _empty_warned:
            _logger.warning("[universe_guard] 上市清单为空（找不到 %s）⇒ "
                            "**宇宙守卫本次未生效**，历史未做可比性裁剪。"
                            "请确认数据目录（data/panel/by_code）存在。", BY_CODE_DIR)
            _empty_warned = True
        return pd.Series(True, index=df.index)

    first = listing.set_index(listing["code"].astype(str))["first_date"]

    codes = (pd.Series(str(code), index=df.index) if code is not None
             else df[code_col].astype(str))
    dates = pd.to_datetime(df[date_col])

    since = codes.map(first) + pd.Timedelta(days=int(min_history_days))
    if exclude_bse:
        bse = codes.map(is_bse).fillna(False).astype(bool)
        # 只抬升"确实是北交所且算出来早于开市日"的行；NaT 不能被顺手改成 BSE_START
        # （那等于给清单里没有的票发通行证）。
        adjust = bse & since.notna() & (since < BSE_START)
        since = since.mask(adjust, BSE_START)
    return (since.notna() & (dates >= since)).fillna(False).astype(bool)


def eligible(df: pd.DataFrame, *, month_col: str = "month_ts",
             code_col: str = "code",
             min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
             listing: pd.DataFrame | None = None,
             exclude_bse: bool = True) -> pd.Series:
    """月度截面版：该行在**月末那天**是否属于可比宇宙。

    实现委托给 `comparable_mask`——月度与日频共用一套判定（见其 docstring）。
    """
    return comparable_mask(df, date_col=month_col, code_col=code_col,
                           min_history_days=min_history_days,
                           listing=listing, exclude_bse=exclude_bse)


def trim(df: pd.DataFrame, *, date_col: str = "date", code_col: str = "code",
         code: str | None = None,
         min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
         listing: pd.DataFrame | None = None,
         exclude_bse: bool = True) -> pd.DataFrame:
    """**按行剪裁**历史日线：只留可比区间的行（日频回测/因子统计的入口）。

    与 `eligible`（按月判断）是同一判据的两种用法：
      - 月度截面：整月只有一行，判"那天可不可比"；
      - 日频回测：一只票一整段 K 线，**要剪掉上市初期的行**，而不是整只票丢弃
        （2020-06 上市的票，2019 年那段不可比、2021 年之后可比，不能一刀切）。

    Returns:
        裁剪后的副本（原 df 不动）。全部不可比时返回**空表**（保持列结构），
        调用方的"无数据"分支照常走。
    """
    if df is None or len(df) == 0:
        return df
    mask = comparable_mask(df, date_col=date_col, code_col=code_col, code=code,
                           min_history_days=min_history_days, listing=listing,
                           exclude_bse=exclude_bse)
    return df[mask.to_numpy()].copy()


def guard_enabled() -> bool:
    """研究侧**默认开启**；`BIGA_NO_UNIVERSE_GUARD=1` 可关。

    为什么要留这个开关：守卫会**剪掉**一部分历史行（上市不满一年、北交所开市前），
    关掉才能复现"没有守卫时看到的是什么"——做前后对比必须有这条退路，
    否则只能靠改代码。
    ⚠️ 关掉得到的数字**不可用于决策**（这正是 2026-10-31 那次错误的成因）。
    """
    import os

    return os.environ.get("BIGA_NO_UNIVERSE_GUARD", "").strip().lower() \
        not in ("1", "true", "yes")


def trim_with_stats(df: pd.DataFrame, code: str, **kw) -> tuple[pd.DataFrame, int]:
    """`trim` 的带统计版：返回 (裁剪后 df, 被剪掉的行数)。

    `code` 必须是证券代码字符串（调用方都在按票循环，不存在"不知道是谁"）。
    统计值必须往外报：**静默剪掉 80% 的历史**与"本来就没数据"看起来一样，
    没有这个数字就分不清是守卫在起作用还是数据缺失。
    """
    if df is None or len(df) == 0:
        return df, 0
    out = trim(df, code=str(code), **kw)
    return out, max(0, len(df) - len(out))


def trim_matrix(mat: pd.DataFrame, *, min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
                exclude_bse: bool = True,
                listing: pd.DataFrame | None = None) -> pd.DataFrame:
    """**宽表版**裁剪：`date × code` 的矩阵，把不可比格子置 NaN。

    为什么需要它：`core.replay_eval.load_matrices()` 返回的是宽矩阵（逐列一只票），
    逐列 `trim` 要跑 5,000 次索引对齐；这里用一次广播造掩码，快两个数量级。

    ⚠️ 置 **NaN 而不是 0**：0 会被当成"当天涨跌 0%"参与排序和均值，
       等于把不可比的票当成"表现平平的票"——那比直接剔除更误导。
    """
    if mat is None or mat.empty:
        return mat
    if listing is None:
        listing = build_listing_table()
    if listing is None or len(listing) == 0:
        return mat          # 判据缺失 ⇒ 放行（理由见 `comparable_mask`）
    first = listing.set_index(listing["code"].astype(str))["first_date"]
    codes = [str(c) for c in mat.columns]
    since = pd.Series(codes, index=codes).map(first) + pd.Timedelta(days=int(min_history_days))
    if exclude_bse:
        bse = pd.Series([is_bse(c) for c in codes], index=codes)
        adjust = bse & since.notna() & (since < BSE_START)
        since = since.mask(adjust, BSE_START)
    dates = pd.to_datetime(mat.index)
    # (T 行 × N 列) 的布尔掩码：True = 可比
    ok = dates.to_numpy()[:, None] >= since.to_numpy()[None, :]
    # NaT（清单里没有的票）对应 since 是 NaT ⇒ 比较恒 False，这里显式保证
    ok &= since.notna().to_numpy()[None, :]
    return mat.where(ok)
