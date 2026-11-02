# -*- coding: utf-8 -*-
"""PIT 财务快照（唯一来源）：财报按公告日贴到月度截面。

为什么必须在 core
──────────────────
`scripts/financial_factor_ic.py`（IC 验证）与 `scripts/portfolio_ab_test.py`
（组合实验）都要"月末 d 只能用 pub_date <= d 的财报"这段逻辑。
各写一遍必然走样——项目铁律「页面与 prompt 说同一件事必须同源」。

PIT 对齐（唯一正确做法，不要改）
────────────────────────────────
每月末 `d` 时，只能用 `pub_date <= d` 的财报，取其中 `pub_date` 最大的
那一期（同日取 `stat_date` 最大）。用 `stat_date` 对齐等于提前用上未来
才公告的报表——A 股回测最常见的前视偏差。

数据源
──────
`data/panel/financials/<code>.parquet`（2026-10-31 起为东财
`RPT_LICO_FN_CPD` 全市场版，见 `scripts/backfill_financials_em.py`）：
- `pub_date` = 首次公告日（PIT 锚点；数值是最新重述版，BaoStock 同语义）
- `stat_date` = 报告期；`net_profit` 是**归母**口径；`eps` 是**累计**非 TTM
- `ocf_to_np` = ocfps/eps（盈利质量代理，东财源独有；IC +0.015、t=+3.5、
  过 DSR p=0.015、分年 7/8 正 —— 本项目第一个过多重检验的基本面因子）
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path

import pandas as pd

from config.paths import data_root

_logger = logging.getLogger("financial_pit")

# ⚠️ frozen 下 `__file__` 在只读 _internal/ ⇒ 数据路径必须走 data_root()
FIN_DIR = Path(data_root()) / "data" / "panel" / "financials"


def load_financials() -> pd.DataFrame:
    """读规范层 per-code 文件并合入可选的 cashflow 表（BaoStock 遗留）。"""
    files = sorted(glob.glob(str(FIN_DIR / "*.parquet")))
    if not files:
        raise SystemExit(f"没有回填数据：{FIN_DIR}\n"
                         f"先跑 `py scripts/backfill_financials_em.py`")
    frames = []
    bad = 0
    for f in files:
        try:
            d = pd.read_parquet(f)
        except Exception as e:                       # noqa: BLE001  半个文件/损坏
            _logger.warning("跳过损坏文件 %s: %s", f, e)
            bad += 1
            continue
        if len(d):
            frames.append(d)
    _logger.info("回填文件 %s 个（损坏跳过 %s），合入 %s 只", len(files), bad, len(frames))
    df = pd.concat(frames, ignore_index=True)
    df["code"] = df["code"].astype(str)
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    df["stat_date"] = pd.to_datetime(df["stat_date"])

    # ── 并入现金流表（BaoStock 遗留，仅 46 只；cfo_to_np 盈利质量直给）──
    cf_dir = FIN_DIR / "cashflow"
    cf_files = sorted(glob.glob(str(cf_dir / "*.parquet")))
    if cf_files:
        cf = pd.concat([pd.read_parquet(f) for f in cf_files], ignore_index=True)
        cf["code"] = cf["code"].astype(str)
        cf["pub_date"] = pd.to_datetime(cf["pub_date"])
        cf["stat_date"] = pd.to_datetime(cf["stat_date"])
        keep = [c for c in ("cfo_to_np", "cfo_to_or") if c in cf.columns]
        df = df.merge(cf[["code", "stat_date", "pub_date"] + keep],
                      on=["code", "stat_date", "pub_date"], how="left")
        _logger.info("并入现金流 %s 期（%s 只）", f"{len(cf):,}", f"{cf['code'].nunique():,}")
    else:
        _logger.info("无现金流表 %s —— 东财源已提供 ocf_to_np 代理", cf_dir)
    return df.sort_values(["code", "stat_date"])


def add_derived(fin: pd.DataFrame) -> pd.DataFrame:
    """同股按报告期排序后算同比（t vs t−4 季）+ 盈利质量代理。"""
    df = fin.sort_values(["code", "stat_date"]).copy()
    g = df.groupby("code", sort=False)
    for col, new in (("roe", "roe_yoy"), ("net_profit", "profit_yoy"),
                     ("cfo_to_np", "cfo_np_yoy")):
        if col not in df.columns:
            continue
        prev = g[col].shift(4)
        if col in ("roe", "cfo_to_np"):
            df[new] = df[col] - prev                       # 比率用差值
        else:
            df[new] = df[col] / prev.where(prev > 0) - 1.0  # 金额用增速（基数为负则 NaN）
    # 盈利质量代理（东财源）：ocfps/eps —— 同为每股、同为报告期累计口径，
    # 相除消掉股本 ⇒ ≈ 经营现金流/归母净利润。eps=0/NaN 时如实 NaN。
    if {"ocfps", "eps"} <= set(df.columns):
        df["ocf_to_np"] = df["ocfps"] / df["eps"].where(df["eps"] != 0)
    return df


# pit_snapshot 的默认输出列（ic 脚本口径；存在才输出，东财/现金流列可选）
DEFAULT_SNAPSHOT_COLS = ("roe", "gp_margin", "np_margin", "roe_yoy", "profit_yoy",
                         "cfo_to_np", "cfo_np_yoy", "ystz", "sjltz", "ocf_to_np")


def pit_snapshot(fin: pd.DataFrame, months: pd.DatetimeIndex,
                 value_cols: list[str] | None = None) -> pd.DataFrame:
    """把财报按 pub_date 贴到每个月末（backward），返回长表。

    Args:
        fin: `load_financials()`/`add_derived()` 的输出（长表，含 pub_date）。
        months: 月末时间戳集合（与量价面板的 `month_ts` 同源）。
        value_cols: 输出的因子列；缺省用 `DEFAULT_SNAPSHOT_COLS` 中存在的列。

    Returns:
        长表 [month_ts, code, stat_date, pub_date, *value_cols]。
        每只票每月**恰好一行**（该时点可见的最新一期）。
    """
    if value_cols is None:
        value_cols = [c for c in DEFAULT_SNAPSHOT_COLS if c in fin.columns]
    out = []
    for d in months:
        sub = fin[fin["pub_date"] <= d]
        if sub.empty:
            continue
        sub = sub.sort_values("pub_date")
        # 每只票取最后一期（pub_date 最大；同一天取 stat_date 最大）
        sub = sub.drop_duplicates(subset=["code"], keep="last")
        sub = sub.assign(month_ts=d)
        out.append(sub[["month_ts", "code", "stat_date", "pub_date", *value_cols]])
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def merge_into_monthly_panel(scan: pd.DataFrame, fin: pd.DataFrame,
                             value_cols: list[str]) -> pd.DataFrame:
    """把 PIT 快照 merge 进量价月度面板（month_ts + code 内连接补列）。

    scan 需含 `month_ts`（datetime）与 `code`（str）；财报不可见的行
    （新上市/未披露）新列保持 NaN —— 由调用方决定 skipna 还是置落选。
    """
    months = pd.DatetimeIndex(sorted(pd.to_datetime(scan["month_ts"]).unique()))
    snap = pit_snapshot(fin, months, value_cols)
    return scan.merge(snap[["month_ts", "code", *value_cols]],
                      on=["month_ts", "code"], how="left")


# ── 盈利质量闸门（2026-10-31）─────────────────────────────────────────
# 依据（A/B 实验 `output/组合AB实验_*_fin.json`，top=20、单边 10bp）：
# - **AF 臂（混入等权合成）被否决**：押注段超额 −9.47% vs 现状 −0.12% ——
#   单因子 IC 0.015 进等权合成是稀释（A3 混趋势因子同一教训）；
# - **AFG 臂（只剔最差 10%、排序不变）通过**：押注段超额 +2.26%、
#   净年化 19.10% vs 16.22%、回撤 −22.1% vs −26.3%，开发段也更好。
# ⇒ 与 `core/crowding.py` 同一条纪律：**做闸门不做分数**。
QUALITY_GATE_PCT = 0.10       # 剔除 ocf_to_np 最差 10%（A/B 同参）
QUALITY_GATE_MIN_SEEN = 20    # 可见值 < 20 只 ⇒ 分位是噪声，不生效
QUALITY_GATE_MIN_COV = 0.60   # 候选池覆盖率 < 60% ⇒ 数据不可信，不生效


def latest_visible_multi(codes, cols, asof=None) -> dict[str, dict]:
    """每只票在 `asof` 前**已公告**的最新一期里，多个字段的值。

    为什么要多字段版：初筛"低估值成长"要同时卡 `roe>10%` 与 `净利同比>20%`，
    逐字段调 `latest_visible` 等于把每个 parquet 读两遍（140 只 ×2 次 IO）。
    这里一次读完、一次排序取最后一期，字段口径与 `latest_visible` 完全一致
    （后者现在是本函数的薄封装，避免两处各写一遍 PIT 规则）。

    Returns: `{code: {col: value}}`；缺文件/无可见值/列不存在 ⇒ 该字段不出现在 dict 里
    （缺 ≠ 0）。
    """
    asof_ts = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now().normalize()
    want = [str(c) for c in (cols or [])]
    # ⚠️⚠️ 派生列**不落盘**，不只 `ocf_to_np` —— `roe_yoy`/`profit_yoy`/
    #    `cfo_np_yoy` 都是 `add_derived()` 现算的。把任何一个直接放进
    #    `read_parquet(columns=...)` 都会抛 "列不存在"，被 except 吞成
    #    "该票缺数据"⇒ **整个候选池的财务判定静默失效**（leftside_monitor
    #    首版实测：637 只全部返回空 dict，就是 `roe_yoy` 这一列害的）。
    # 派生列 → 现算所需的原始列
    DERIVED_INPUTS = {
        "ocf_to_np": ("ocfps", "eps"),
        "roe_yoy": ("roe",),            # roe 本身落盘；roe_yoy = roe − roe.shift(4)
        "profit_yoy": ("net_profit",),
        "cfo_np_yoy": ("cfo_to_np",),   # 现金流表本就缺 ⇒ 基本拿不到，如实缺
    }
    derived = {c for c in want if c in DERIVED_INPUTS}
    # ⚠️ 但 roe_yoy 需要**最近 5 期**的 roe 才能算差值 —— 单读最后一期算不出来。
    #    旧行为（把 roe_yoy 当落盘列读）是彻底失败；退而求其次：直接读全表
    #    现算。为控制 IO，只对含"需要历史期"的派生列走全表路径。
    read_cols = ["pub_date", *[c for c in want if c not in derived]]
    for c in derived:
        for src in DERIVED_INPUTS[c]:
            if src not in read_cols:
                read_cols.append(src)
    # ⚠️ `stat_date` **不许**进 read_cols：它是"可能不存在"的列（不同批次落盘的
    #    列集不同），`read_parquet(columns=[...])` 遇缺列直接抛异常 ⇒ 被下面的
    #    except 吞成"缺数据"⇒ 闸门静默失效（quality_gate 夹具实测抓红）。
    #    roe_yoy 按期序差分，用 pub_date 排序即可 —— 公告顺序与报告期顺序单调
    #    对应（同一份报告先有报告期后有公告日），差分结果等价。
    out: dict[str, dict] = {}
    for c in {str(x) for x in (codes or [])}:
        p = FIN_DIR / f"{c}.parquet"
        if not p.exists():
            continue
        try:
            d = pd.read_parquet(p, columns=read_cols)
        except Exception as e:                       # noqa: BLE001  损坏文件按缺处理
            _logger.warning("[PIT] 读 %s 失败: %s", p.name, e)
            continue
        # 现算派生列（与 add_derived 同口径：比率差值 / 金额增速）
        num = lambda s: pd.to_numeric(s, errors="coerce")   # noqa: E731
        if "ocf_to_np" in derived and {"ocfps", "eps"} <= set(d.columns):
            eps = num(d["eps"])
            d = d.assign(ocf_to_np=num(d["ocfps"]) / eps.where(eps != 0))
        if "roe_yoy" in derived and "roe" in d.columns:
            # 按期序差分（pub_date 与报告期单调对应，见上）；需要 ≥5 期才有值
            d = d.sort_values("pub_date")
            d = d.assign(roe_yoy=num(d["roe"]) - num(d["roe"]).shift(4))
        if "profit_yoy" in derived and "net_profit" in d.columns:
            d = d.sort_values("pub_date")
            npf = num(d["net_profit"])
            d = d.assign(profit_yoy=npf / npf.shift(4).where(npf.shift(4) > 0) - 1.0)
        d = d[pd.to_datetime(d["pub_date"]) <= asof_ts]
        if d.empty:
            continue
        d = d.sort_values("pub_date")
        last = d.iloc[-1]
        row = {}
        for col in want:
            if col in d.columns:
                v = pd.to_numeric(pd.Series([last[col]]), errors="coerce").iloc[0]
                if pd.notna(v):
                    row[col] = float(v)
        if row:
            out[c] = row
    return out


def latest_visible(codes, col: str = "ocf_to_np", asof=None) -> dict[str, float]:
    """每只票在 `asof` 前**已公告**的最新一期 `col` 值（PIT 纪律）。

    薄封装 `latest_visible_multi` —— PIT 规则只写一遍，两处口径不会分叉。
    """
    return {c: v[col] for c, v in
            latest_visible_multi(codes, [col], asof=asof).items() if col in v}


def apply_quality_gate(codes, pct: float = QUALITY_GATE_PCT,
                       asof=None) -> tuple:
    """对一批候选剔除盈利质量最差 `pct` 分位 → `(保留, 被剔清单, meta)`。

    边界纪律（与 `core.crowding.apply_gate` 同一套）：
      1. 财务数据取不到 ⇒ **不过滤**，meta 写明"未生效"；
      2. 可见值太少（< `QUALITY_GATE_MIN_SEEN`）或覆盖率过低 ⇒ 不生效；
      3. 单只缺 ocf_to_np（未披露/银行口径差异/文件缺失）⇒ **不剔除**
         （缺 ≠ 差，铁律"缺就是缺"——被剔除必须是因为"质量已知且差"）；
      4. 分位在**候选池内**算。⚠️ 口径注记：A/B 实验是在流动性下限后的
         全月截面算分位；线上候选池更小（五路初筛后），被剔集合可能略
         不同 —— meta 里带 `pool` 字段供复核。
    """
    asof_ts = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now().normalize()
    codes = [str(c) for c in (codes or [])]
    meta = {"enabled": True, "applied": False, "asof": None, "pct": pct,
            "threshold": None, "dropped": [], "note": "",
            "pool": len(codes), "coverage": None}
    if not codes:
        meta["note"] = "空候选池"
        return codes, [], meta
    asof_ts = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now().normalize()
    meta["asof"] = str(asof_ts.date())

    vals = latest_visible(codes, col="ocf_to_np", asof=asof_ts)
    meta["coverage"] = round(len(vals) / max(len(codes), 1), 3)
    if len(vals) < QUALITY_GATE_MIN_SEEN:
        meta["note"] = (f"可见盈利质量值仅 {len(vals)} 只（< {QUALITY_GATE_MIN_SEEN}）"
                        "⇒ 闸门未生效（不剔除任何候选）")
        return codes, [], meta
    if meta["coverage"] < QUALITY_GATE_MIN_COV:
        meta["note"] = (f"候选池覆盖率 {meta['coverage']:.0%}（< {QUALITY_GATE_MIN_COV:.0%}）"
                        "⇒ 闸门未生效（不剔除任何候选）")
        return codes, [], meta

    arr = pd.Series(vals)
    thresh = float(arr.quantile(pct))
    meta["threshold"] = round(thresh, 4)
    keep, dropped = [], []
    for c in codes:
        v = vals.get(c)
        if v is not None and v < thresh:             # 严格小于：恰在阈值上的保留
            dropped.append({"股票代码": c, "ocf_to_np": round(v, 4)})
        else:
            keep.append(c)                           # 缺数据的保留（边界 3）
    if not dropped:
        meta["note"] = "没有候选落在剔除分位内（正常）"
        return codes, [], meta
    meta["applied"] = True
    _logger.info("[选股] 盈利质量闸门：阈值 %.4f（池内 %.0f%% 分位），剔 %d 只",
                 thresh, pct * 100, len(dropped))
    return keep, dropped, meta
