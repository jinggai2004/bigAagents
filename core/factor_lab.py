# -*- coding: utf-8 -*-
"""因子实验室：用户自定义因子的回测与应用。

三个层次（缺一层就会出现"看起来能加因子，其实结论不可信"）
────────────────────────────────────────────────────────────
1. **表达式**（`core/factor_dsl.py`）：用户写 `amihud20 / lndvol20` 这样的一行，
   不是写代码。AST 白名单校验，`fwd20` 等目标列结构性不可引用。
2. **回测**（本模块）：月度截面 IC / t / ICIR / 分年同号 / Q5−Q1 + **DSR/PBO**。
3. **应用**（本模块末）：只有两条合法身份——**闸门**（默认）或**分数**（需 A/B）。

⚠️ 本模块最关键的一条：**试验次数必须累计**
──────────────────────────────────────────
DSR 的 hurdle 来自「全部试验的 |SR| 分布」。如果每次只把用户这一个因子当试验，
`n_trials=1` ⇒ 任何因子都会"显著"——三重检验等于没做，只是换了个名字显得严谨。
所以试验集恒为 **30 个内置价量因子 + 全部已登记用户因子**，用户每回测一次就 +1，
hurdle 只升不降。这是防"反复改表达式试到显著"的唯一结构性办法。

⚠️ 为什么默认是闸门而不是分数
──────────────────────────────
A/B 实证（2026-10-31）：把弱 IC 因子（0.015）混进等权合成，押注段超额从 −0.12%
恶化到 **−9.47%**——弱信号会稀释排序结构。同一因子改成"只剔最差 10%、排序不动"
则押注段 **+2.26%**、净年化 19.10% vs 16.22%。所以本模块对"分数"那条路只提供
送审接口，不直接改 `FACTOR_WEIGHTS`。

数据与口径
──────────
- 量价：`temp/factor_scan_scan.parquet`（月度截面 30 因子 + `fwd20`，约 51.6 万行）
- 财务：`core/financial_pit.pit_snapshot`（PIT 对齐，公告日之后才可见）
- 合并用 **left join 到量价**：财务缺失的票仍参与价量因子回测（缺≠0，NaN 不参与该因子）
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.paths import data_root
from core import factor_dsl
from core.factor_multiple_testing import adjust_by_multiple_testing
from core.factor_stats import MIN_MONTHS, ic_table, ls_series, year_consistency

_logger = logging.getLogger("factor_lab")

ROOT = Path(data_root())
SCAN_CACHE = ROOT / "temp" / "factor_scan_scan.parquet"
TRIAL_CACHE = ROOT / "temp" / "factor_lab_trials.parquet"

# 只要过了这条线，才允许"应用"（闸门或送审分数）
MIN_ABS_IC = 0.010          # |IC| 下限：低于此值的因子在 5,000 只截面里几乎没有区分度
MIN_ABS_T = 2.0             # t 下限
MIN_YEAR_RATIO = 0.7        # 分年同号率下限（7/10 年）
MIN_COVERAGE = 0.60         # 覆盖低于 60% 的因子在实盘会大部分票"无法判定"

# ── 面板 ────────────────────────────────────────────────────────────────
# 面板分两档，这是**性能上唯一重要的一件事**：
#   量价档：只读 `temp/factor_scan_scan.parquet` —— **0.4 秒**
#   财务档：再加 `load_financials()` → 要读 **5,582 个** per-code parquet —— **21.5 秒**
# 实测拆解（2026-11-01）：读扫描缓存 0.4s / load_financials 21.5s / pit_snapshot 3.9s /
# merge 0.3s。也就是说 48 秒首载里，26 秒是在读那 5,582 个小文件，跟"回测"没关系。
# ⇒ 表达式只用量价字段时**根本不碰财务**；两档各自落 parquet 缓存，第二次开进程 1~2 秒。
PANEL_CACHE = {"price": ROOT / "temp" / "factor_lab_panel_price.parquet",
               "fin": ROOT / "temp" / "factor_lab_panel_fin.parquet"}
PANEL_META = {"price": ROOT / "temp" / "factor_lab_panel_price.meta.json",
              "fin": ROOT / "temp" / "factor_lab_panel_fin.meta.json"}

_MEMO: dict[str, pd.DataFrame | None] = {"price": None, "fin": None}
_MEMO_STAMP: dict[str, str | None] = {"price": None, "fin": None}
_FIN_DIGEST: str | None = None


def _fin_digest() -> str:
    """财务目录指纹（文件数 + 最新 mtime）。

    只算一次并留在进程里：5,582 次 stat 约 0.2 秒，比重读 21 秒便宜得多，
    但没必要每次调用都算。回填新数据会改 mtime ⇒ 指纹变 ⇒ 缓存失效。
    """
    global _FIN_DIGEST
    if _FIN_DIGEST is None:
        try:
            from core.financial_pit import FIN_DIR as _FD
            files = list(Path(_FD).glob("*.parquet"))
            # 用 ns 不用 s：秒级粒度扛不住"同一秒内重建"（本项目在时间判据上踩过）
            newest = max((f.stat().st_mtime_ns for f in files), default=0)
            _FIN_DIGEST = f"{len(files)}:{newest}"
        except Exception as e:                        # noqa: BLE001
            _logger.debug("现金流目录指纹计算失败（按空指纹处理）: %s", e)
            _FIN_DIGEST = "0:0"
    return _FIN_DIGEST


def _stamp(kind: str) -> str:
    mt = SCAN_CACHE.stat().st_mtime_ns
    return f"{mt}" if kind == "price" else f"{mt}|{_fin_digest()}"


def _cache_ok(kind: str) -> bool:
    """落盘缓存是否可用（存在 + 指纹一致）。只读 json，不读 parquet。"""
    try:
        if not PANEL_CACHE[kind].exists() or not PANEL_META[kind].exists():
            return False
        meta = json.loads(PANEL_META[kind].read_text(encoding="utf-8"))
        return meta.get("stamp") == _stamp(kind)
    except Exception as e:                            # noqa: BLE001  缓存坏了就当没有
        _logger.debug("面板缓存元数据不可用（按未缓存处理）: %s", e)
        return False


def _build(kind: str) -> pd.DataFrame:
    if not SCAN_CACHE.exists():
        raise FileNotFoundError(
            f"缺少量价扫描缓存 {SCAN_CACHE}（先用 scripts/factor_paper_scan.py 生成）")
    scan = pd.read_parquet(SCAN_CACHE)
    if "month_ts" not in scan.columns:
        scan = scan.rename(columns={"date": "month_ts"})
    scan["month_ts"] = pd.to_datetime(scan["month_ts"])
    scan["code"] = scan["code"].astype(str)
    if kind == "price":
        return scan

    # 财务 PIT：best-effort —— 财务缺失不能让因子实验室整体不可用（价量因子仍要能测）
    try:
        from core.financial_pit import add_derived, load_financials, pit_snapshot
        fin = add_derived(load_financials())
        months = pd.DatetimeIndex(sorted(scan["month_ts"].unique()))
        # 显式给全部财务字段（默认只给 DEFAULT_SNAPSHOT_COLS，`eps`/`bps`/`ocfps`
        # 不在里面 ⇒ 字段表有、面板却没有 ⇒ 用户写 log(revenue) 得到空结果）
        want = [c for c in factor_dsl.FIN_FIELDS if c in fin.columns]
        snap = pit_snapshot(fin, months, value_cols=want)
        cols = [c for c in snap.columns if c not in ("month_ts", "code",
                                                     "stat_date", "pub_date")]
        panel = scan.merge(snap[["month_ts", "code", *cols]],
                           on=["month_ts", "code"], how="left")
        _logger.info("[因子实验室] 财务档含 %d 列（%s）", len(cols), ", ".join(cols))
    except Exception as e:                            # noqa: BLE001
        _logger.warning("[因子实验室] 财务 PIT 不可用（只做量价因子）: %s", e)
        panel = scan
    return panel


def _persist(kind: str, panel: pd.DataFrame) -> None:
    """落盘缓存。⚠️ 先写 .tmp 再 `os.replace` —— 直接 `to_parquet` 会 truncate 原文件，
    这期间别的进程读到的是半个文件（本项目在面板重建上踩过同一个坑）。"""
    import os
    try:
        PANEL_CACHE[kind].parent.mkdir(parents=True, exist_ok=True)
        tmp = PANEL_CACHE[kind].with_suffix(".tmp.parquet")
        panel.to_parquet(tmp, index=False)
        os.replace(tmp, PANEL_CACHE[kind])
        PANEL_META[kind].write_text(
            json.dumps({"stamp": _stamp(kind), "rows": int(len(panel)),
                        "cols": int(len(panel.columns))}, ensure_ascii=False),
            encoding="utf-8")
    except Exception as e:                            # noqa: BLE001
        _logger.warning("[因子实验室] 面板缓存写入失败（不影响本次回测）: %s", e)


def load_panel(need_fin: bool = False, force: bool = False) -> pd.DataFrame:
    """量价月度截面（可选并上财务 PIT 列）。

    Args:
        need_fin: 表达式用到财务字段才传 True —— 差 21 秒（5,582 个文件）。
        force: 忽略缓存重建。

    `need_fin=True` 时若财务不可用，返回的仍是量价档（财务列缺失=NaN），
    由 `field_coverage()` 如实标出，不让整个功能因为一份数据挂掉。
    """
    kind = "fin" if need_fin else "price"
    if not SCAN_CACHE.exists():
        raise FileNotFoundError(
            f"缺少量价扫描缓存 {SCAN_CACHE}（先用 scripts/factor_paper_scan.py 生成）")
    stamp = _stamp(kind)
    if not force and _MEMO[kind] is not None and _MEMO_STAMP[kind] == stamp:
        return _MEMO[kind]
    if not force and _cache_ok(kind):
        try:
            p = pd.read_parquet(PANEL_CACHE[kind])
            _MEMO[kind], _MEMO_STAMP[kind] = p, stamp
            _logger.info("[因子实验室] 面板命中缓存（%s，%d 行）", kind, len(p))
            return p
        except Exception as e:                        # noqa: BLE001
            _logger.warning("[因子实验室] 缓存读取失败，重建: %s", e)
    t0 = time.perf_counter()
    p = _build(kind)
    _MEMO[kind], _MEMO_STAMP[kind] = p, stamp
    _persist(kind, p)
    _logger.info("[因子实验室] 面板构建完成（%s，%d 行）耗时 %.1fs",
                 kind, len(p), time.perf_counter() - t0)
    return p


def panel_ready(need_fin: bool = False) -> bool:
    """不重建就能拿到面板吗（进程内有 或 落盘缓存有效）。**不触发生成。**"""
    kind = "fin" if need_fin else "price"
    if _MEMO[kind] is not None:
        return True
    return SCAN_CACHE.exists() and _cache_ok(kind)


def panel_info(need_fin: bool | None = None, allow_load: bool = True) -> dict | None:
    """面板概况。`allow_load=False` 时**绝不触发生成**（页面首屏用）。

    Returns None = 还没有可用的面板（调用方应显示"首次回测时构建"）。
    """
    if need_fin is None:
        need_fin = panel_ready(True)
    if not panel_ready(need_fin):
        # 财务档还没建，但量价档可用 ⇒ 退化到量价档（仍是真实数据，只是没财务列），
        # 别让页面显示成"面板未构建"——那会让人以为功能坏了。
        if panel_ready(False):
            need_fin = False
        elif not allow_load:
            return None
        else:
            need_fin = False
    p = load_panel(need_fin)
    cov = p.groupby("month_ts")["code"].nunique()
    return {"rows": int(len(p)), "stocks": int(p["code"].nunique()),
            "months": int(p["month_ts"].nunique()),
            "start": str(p["month_ts"].min().date()),
            "end": str(p["month_ts"].max().date()),
            "coverage_median": int(cov.median()),
            "fields": len(factor_dsl.ALL_FIELDS),
            "price_fields": len(factor_dsl.PRICE_FIELDS),
            "fin_fields": len(factor_dsl.FIN_FIELDS),
            "has_fin": need_fin,
            "cached": bool(panel_ready(need_fin))}


# 字段覆盖率（进程内缓存；面板换了就重算）
_FIELD_COV: dict[str, float | None] = {}
_FIELD_COV_MTIME: str | None = None

# 覆盖率低于这个值的字段在页面上标"空"，并在回测报错时点出来。
# 25% 是"过半月份连一格都没有"的量级，已无统计意义（不是嫌少，是没有）。
FIELD_EMPTY_BELOW = 0.25


def field_coverage(full: bool = False) -> dict[str, float | None]:
    """每个字段的非空比例（0~1）。缺列记 0；**财务档没加载时财务字段给 None**。

    为什么不一律返 0：NaN 覆盖率与"没加载"在页面上要区分得开——前者是数据问题
    （`ivol20`/`dvol20` 就是列存在但 100% NaN），后者只是还没算（点一次回测就有）。

    ⚠️ 量价覆盖率只读扫描缓存（0.4 秒），**绝不为了给页面标个百分比去读 5,582 个
    财务文件**（21.5 秒）。财务覆盖率只在财务档已经可用（缓存命中/已加载）时才给。
    """
    global _FIELD_COV, _FIELD_COV_MTIME
    kind = "fin" if (full and panel_ready(True)) else "price"
    stamp = _stamp(kind)
    if _FIELD_COV and _FIELD_COV_MTIME == stamp:
        return _FIELD_COV
    p = load_panel(kind == "fin")
    cov: dict[str, float | None] = {}
    for c in factor_dsl.ALL_FIELDS:
        if c in factor_dsl.FIN_FIELDS and kind != "fin":
            cov[c] = None                     # 没加载 ≠ 空
        else:
            cov[c] = float(p[c].notna().mean()) if c in p.columns else 0.0
    _FIELD_COV, _FIELD_COV_MTIME = cov, stamp
    return cov


def catalog(full: bool = False) -> list[dict]:
    """字段清单（页面用）：带中文名、分组、覆盖率与"是否可用"。

    `coverage=None`（财务字段未加载）时 `available` 也是 None —— 页面据此显示
    "待回测"而不是"不可用"。
    """
    cov = field_coverage(full=full)
    out = []
    for g, d in factor_dsl.FIELD_GROUPS.items():
        for name, label in d.items():
            c = cov.get(name)
            out.append({"name": name, "label": label, "group": g,
                        "coverage": None if c is None else round(c, 3),
                        "available": None if c is None else c >= FIELD_EMPTY_BELOW})
    return out


def needs_fin(fields) -> bool:
    """表达式是否引用财务字段（决定要不要付那 21 秒）。"""
    return any(f in factor_dsl.FIN_FIELDS for f in (fields or []))


# ── 试验集（DSR 的 hurdle 来源）──────────────────────────────────────────
def _builtin_trials(panel: pd.DataFrame) -> pd.DataFrame:
    """30 个内置价量因子的月频多空序列矩阵（缓存到 parquet，按面板 mtime 失效）。"""
    mt = SCAN_CACHE.stat().st_mtime
    if TRIAL_CACHE.exists():
        try:
            cached = pd.read_parquet(TRIAL_CACHE)
            if abs(float(cached.attrs.get("src_mtime", -1)) - mt) < 1:
                return cached
        except Exception as e:                        # noqa: BLE001  缓存坏了就重建
            _logger.debug("试验集缓存不可用（重建）: %s", e)
    cols = [c for c in factor_dsl.PRICE_FIELDS if c in panel.columns]
    data = {}
    for c in cols:
        s = ls_series(panel, c)
        if s is not None:
            data[c] = s
    mat = pd.DataFrame(data).sort_index()
    mat.attrs["src_mtime"] = mt
    try:
        TRIAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        mat.to_parquet(TRIAL_CACHE)               # 不存 attrs，回读时靠 mtime 比对失效
    except Exception as e:                        # noqa: BLE001
        _logger.warning("[因子实验室] 试验集缓存写入失败（不影响回测）: %s", e)
    return mat


def _user_trials(exclude_id: str | None = None) -> dict[str, pd.Series]:
    """已登记用户因子贡献的试验（用登记时缓存的月频多空序列，不重算）。"""
    from db import store
    out: dict[str, pd.Series] = {}
    for row in store.list_user_factors():
        if exclude_id and row.get("factor_id") == exclude_id:
            continue
        m = row.get("metrics") or {}
        sp = (m.get("spread") or {}) if isinstance(m, dict) else {}
        if not sp:
            continue
        s = pd.Series({pd.Timestamp(k): v for k, v in sp.items()}).sort_index()
        if len(s) >= 20:
            out[f"uf:{row['factor_id']}"] = s
    return out


def _spread_dict(s: pd.Series | None) -> dict:
    if s is None:
        return {}
    return {d.strftime("%Y-%m-%d"): round(float(v), 8) for d, v in s.dropna().items()}


# 冗余判据（两条线，含义不同）
#   0.95 = "基本是同一个变量"：同一个量的单调变换（`amihud20/lndvol20` 与 `amihud20`
#          实测 rho 1.00），这种复制品必须挡掉——不带来分散，只让闸门把同一个赌注下两次。
#   0.85 = "有实质重叠"：只提示。两个流动性因子相关 0.9 很常见，一刀切会误杀
#          （MAX 与 IVOL 相关高，但在 9/11 年仍有残余信息，靠正交才看清）。
DUPLICATE_RHO = 0.95
REDUNDANT_RHO = 0.85


def redundancy(panel: pd.DataFrame, col: str, exclude=(),
               sample_months: int = 12) -> dict:
    """候选因子与既有价量因子的截面相关（平均 Spearman）。

    为什么必须有这一步：本项目最贵的一课是 **MAX 被 IVOL 吸收**——
    max20 单看 t=−6 很漂亮，但正交掉 IVOL 后只剩 −0.0146（9/11 年）。
    相关性高意味着它可能只是已有因子的变形，而不是新信号。

    ⚠️ `exclude` 必须传**表达式自己引用的字段**：拿 `lndvol20` 跟 `lndvol20` 比，
    rho 恒 1.00 ⇒ 任何单字段因子都会被判"自我重复"（实测踩过这个坑）。

    只抽样若干个月（默认 12）算——90 个月的 30×相关矩阵没有必要，抽样足够看出
    0.9 这种量级的冗余，而成本从分钟级降到秒级。
    """
    skip = {col}
    cands = [c for c in factor_dsl.PRICE_FIELDS
             if c in panel.columns and c not in skip]
    if not cands:
        return {"max_rho": None, "top": [], "duplicate": False,
                "redundant": False, "note": "无对照因子"}
    own = set(exclude)
    months = sorted(panel["month_ts"].unique())
    step = max(1, len(months) // max(1, sample_months))
    sub = panel[panel["month_ts"].isin(months[::step])]
    out = []
    for c in cands:
        t = sub[["month_ts", col, c]].dropna()
        if len(t) < 500:
            continue
        per = t.groupby("month_ts")[[col, c]].apply(
            lambda g: g[col].corr(g[c], method="spearman") if len(g) >= 50 else np.nan)
        per = per.dropna()
        if len(per) < 3:
            continue
        # `own=True`：这个字段就在表达式里，和它高相关是**必然**的（不是重复，
        # 是构成关系）。只作展示，不参与 duplicate 判定——否则 `lndvol20` 会与
        # 自己 rho=1.00 被自己挡掉。
        out.append({"factor": c, "rho": round(float(per.mean()), 3),
                    "own": c in own})
    out.sort(key=lambda x: -abs(x["rho"]))
    ext = [x for x in out if not x["own"]]
    return {"max_rho": abs(ext[0]["rho"]) if ext else None,
            "top": out[:3],
            "months": int(sub["month_ts"].nunique()),
            "duplicate": bool(ext and abs(ext[0]["rho"]) >= DUPLICATE_RHO),
            "redundant": bool(ext and abs(ext[0]["rho"]) >= REDUNDANT_RHO)}


# ── 回测 ────────────────────────────────────────────────────────────────
def backtest(expr: str, name: str | None = None) -> dict:
    """对一个表达式跑完整回测。返回可直接落库的 metrics。

    不写库、不改任何配置——回测是**纯读**操作（页面可以随便点）。
    """
    from db import store
    t0 = time.perf_counter()
    validated = factor_dsl.validate(expr)
    # 只用量价字段就**不加载财务档**：差 21.5 秒（5,582 个 parquet 文件）
    panel = load_panel(need_fin=needs_fin(validated["fields"]))
    vals = factor_dsl.evaluate(expr, panel, validated=validated)
    panel = panel.assign(_uf=vals)
    n_valid = int(panel["_uf"].notna().sum())
    if n_valid < 1000:
        # 说清楚是哪个字段空了——只报"有效值太少"会让人去查语法，实际是数据没落盘
        cov = field_coverage()
        empty = [f for f in validated["fields"] if cov.get(f, 0.0) < 0.01]
        hint = (f"；这些字段在面板里几乎是空的：{', '.join(empty)}（覆盖率 <1%）"
                if empty else "；先换个字段，或套 cs_rank/去极值")
        raise ValueError(
            f"该表达式只有 {n_valid} 个有效值（面板 {len(panel):,} 行）{hint}")

    row = ic_table(panel, name or "user", "_uf")
    if "t_stat" not in row:
        return {"ok": False, "reason": row.get("verdict", "无法计算 IC"),
                "coverage": round(n_valid / max(len(panel), 1), 3),
                "fields": validated["fields"]}

    direction = 1 if (row.get("ic_mean") or 0) >= 0 else -1
    yc = year_consistency(row.get("by_year") or {}, direction)

    # 试验集 = 30 内置 + 全部已登记用户因子（含自己）。**必须含自己**：它确实是一次试验。
    # ⚠️ 两个坑都踩过：
    #    ① `adjust_by_multiple_testing` 要 `{名: Series}`，传 DataFrame 会在它的
    #       `series_map or {}` 处抛 "truth value is ambiguous"；
    #    ② 它按 `row["name"]` 去 `series_map` 里找序列 ⇒ 候选者的键名必须**与行同名**，
    #       否则拿到 None、dsr 落成 {"ok": false}，每个因子都被判"未过多重检验"。
    builtin = _builtin_trials(panel)
    trials: dict[str, pd.Series] = {c: builtin[c] for c in builtin.columns}
    spread = ls_series(panel, "_uf")
    key = f"candidate:{name or 'user'}"
    for k, s in _user_trials().items():
        trials[k] = s
    if spread is not None:
        trials[key] = spread

    res_rows = [dict(row, name=key)]
    mt = adjust_by_multiple_testing(res_rows, trials)
    dsr = res_rows[0].get("dsr") or {}
    row = res_rows[0]

    cov_month = (panel.dropna(subset=["_uf"]).groupby("month_ts")["code"].nunique()
                 / panel.groupby("month_ts")["code"].nunique())
    coverage = float(n_valid / max(len(panel), 1))
    latest = panel["month_ts"].max()
    latest_rows = panel[panel["month_ts"] == latest]
    latest_cov = float(latest_rows["_uf"].notna().mean()) if len(latest_rows) else 0.0

    # 判定：先看数据够不够，再看统计，最后才看能不能用
    red = redundancy(panel, "_uf", exclude=validated["fields"])
    reasons: list[str] = []
    if coverage < MIN_COVERAGE:
        reasons.append(f"覆盖率 {coverage:.0%} < {MIN_COVERAGE:.0%}（实盘大量票无法判定）")
    if abs(row["ic_mean"]) < MIN_ABS_IC:
        reasons.append(f"|IC| {abs(row['ic_mean']):.3f} < {MIN_ABS_IC}")
    if abs(row["t_stat"]) < MIN_ABS_T:
        reasons.append(f"|t| {abs(row['t_stat']):.1f} < {MIN_ABS_T}")
    if not dsr.get("significant"):
        p = dsr.get("p_value")
        reasons.append("未过 DSR 多重检验" + (f"（p={p:.3f}）" if isinstance(p, float) else ""))
    if yc.get("consistent") is False:
        reasons.append(f"分年同号 {yc['same_sign']}/{yc['years']} < {MIN_YEAR_RATIO:.0%}")
    if red.get("duplicate"):
        top = red["top"][0]
        reasons.append(f"与 `{top['factor']}` 截面相关 {top['rho']:+.2f}"
                       f"（≥{DUPLICATE_RHO}）⇒ 基本是它的单调变换，不是新信号")
    usable = not reasons

    return {
        "ok": True,
        "expr": expr,
        "name": name or "user",
        "fields": validated["fields"],
        "funcs": validated["funcs"],
        "coverage": round(coverage, 3),
        "latest_coverage": round(latest_cov, 3),
        "n_valid": n_valid,
        "ic_mean": row.get("ic_mean"),
        "ic_ir": row.get("ic_ir"),
        "t_stat": row.get("t_stat"),
        "n_months": row.get("n_months"),
        "pos_pct": row.get("pos_pct"),
        "q5_q1_spread": row.get("q5_q1_spread"),
        "by_year": row.get("by_year"),
        "year_consistency": yc,
        "direction": direction,
        "direction_label": "越高越好" if direction > 0 else "越低越好（反向）",
        "dsr": {k: dsr.get(k) for k in ("ok", "significant", "p_value", "sr_annual")},
        "n_trials": mt.get("n_trials"),
        "hurdle_sr_monthly": mt.get("sr_star_monthly"),
        "pbo": (mt.get("pbo") or {}).get("pbo"),
        "redundancy": red,
        "usable": usable,
        "blockers": reasons,
        "suggested_kind": "gate" if usable else None,
        "verdict": ("可用于闸门" if usable else "不可用"),
        "spread": _spread_dict(spread),
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "ran_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def preview(expr: str, limit: int = 12) -> dict:
    """最近一个月的截面取值（让用户在回测前先看量纲对不对）。"""
    validated = factor_dsl.validate(expr)
    panel = load_panel(need_fin=needs_fin(validated["fields"]))
    vals = factor_dsl.evaluate(expr, panel, validated=validated)
    p = panel.assign(_uf=vals)
    latest = p["month_ts"].max()
    last = p[p["month_ts"] == latest].dropna(subset=["_uf"])
    rows = (last.sort_values("_uf", ascending=False)
            .head(max(1, min(limit, 50)))[["code", "_uf"]])
    return {
        "as_of": str(latest.date()),
        "universe": int(len(p[p["month_ts"] == latest])),
        "valid": int(len(last)),
        "stats": {"min": float(last["_uf"].min()) if len(last) else None,
                  "median": float(last["_uf"].median()) if len(last) else None,
                  "max": float(last["_uf"].max()) if len(last) else None},
        "top": [{"code": r["code"], "value": round(float(r["_uf"]), 6)}
                for _, r in rows.iterrows()],
        "fields": validated["fields"],
    }


# ── 登记 / 应用 ──────────────────────────────────────────────────────────
def register(expr: str, name: str, *, kind: str = "gate", gate_pct: float = 0.1,
             note: str = "", factor_id: str | None = None) -> dict:
    """回测并落库（draft → validated / rejected 由 `usable` 决定）。"""
    from db import store
    name = (name or "").strip()
    if not name:
        raise ValueError("给因子起个名字（页面列表里要靠它认）")
    if kind not in ("gate", "score"):
        raise ValueError("kind 只能是 gate（闸门）或 score（分数，需 A/B）")
    if not (0.01 <= gate_pct <= 0.5):
        raise ValueError("gate_pct 取 1%~50%（剔太多会把候选池打空）")

    m = backtest(expr, name)
    if not m.get("ok"):
        raise ValueError(m.get("reason") or "回测失败")
    status = "validated" if m["usable"] else "rejected"
    row = store.upsert_user_factor(
        {"factor_id": factor_id, "name": name, "expr": expr, "kind": kind,
         "gate_pct": float(gate_pct), "note": note,
         "status": status, "direction": m["direction"], "metrics": m})
    _logger.info("[因子实验室] %s %s（%s）：IC %s / t %s / DSR %s ⇒ %s",
                 "更新" if factor_id else "登记", name, expr,
                 m["ic_mean"], m["t_stat"], (m["dsr"] or {}).get("significant"), status)
    return {"factor": row, "metrics": m}


def set_status(factor_id: str, status: str) -> dict:
    """改状态。`applied` 才会真正进入选股闸门。"""
    from db import store
    if status not in ("draft", "validated", "rejected", "applied", "disabled"):
        raise ValueError("未知状态")
    row = store.get_user_factor(factor_id)
    if not row:
        raise ValueError("因子不存在")
    if status == "applied":
        m = row.get("metrics") or {}
        if not m.get("usable"):
            raise ValueError("没通过检验的因子不许应用（先看 blockers 差在哪）")
        if row.get("kind") == "score":
            raise ValueError("分数类因子必须先过 A/B（scripts/portfolio_ab_test.py），"
                             "不能从页面直接启用")
    return store.set_user_factor_status(factor_id, status)


def applied_gates() -> list[dict]:
    from db import store
    return [r for r in store.list_user_factors() if r.get("status") == "applied"]


def apply_gates(codes, asof=None) -> tuple[list[str], list[dict], dict]:
    """把 `applied` 的用户因子当**闸门**用：剔掉因子最差 `gate_pct` 的票。

    边界纪律与 crowding / 盈利质量闸门一致（三条都是踩过坑换来的）：
    1. 因子缺值的票**不剔**——"不知道"不等于"差"；
    2. 有效值少于 20 只 ⇒ 分位是噪声，闸门不生效；
    3. 覆盖率低于 60% ⇒ 数据不可信，闸门不生效。
    """
    codes = [str(c) for c in (codes or [])]
    meta = {"applied": False, "gates": [], "dropped": [], "pool": len(codes),
            "note": ""}
    rows = applied_gates()
    if not codes or not rows:
        meta["note"] = "无已应用的因子闸门" if not rows else "空候选池"
        return codes, [], meta

    # 任一已应用因子用到财务字段才加载财务档（否则只读量价，省 21 秒）
    _need = any(needs_fin(factor_dsl.validate(r["expr"])["fields"])
                for r in rows if r.get("expr"))
    panel = load_panel(need_fin=_need)
    # 判据时点：默认用面板最后一期（实盘就是"最近一个月末的快照"）
    asof_ts = pd.Timestamp(asof) if asof is not None else panel["month_ts"].max()
    sub = panel[panel["month_ts"] <= asof_ts]
    if sub.empty:
        meta["note"] = f"{asof_ts.date()} 之前无面板数据"
        return codes, [], meta
    latest = sub["month_ts"].max()
    cur = sub[sub["month_ts"] == latest]
    meta["asof"] = str(latest.date())

    keep = list(codes)
    hit_any = False
    for row in rows:
        try:
            v = factor_dsl.evaluate(row["expr"], cur)
        except Exception as e:                        # noqa: BLE001  坏表达式不许炸选股
            _logger.warning("[因子闸门] %s 求值失败，跳过: %s", row.get("name"), e)
            continue
        # ⚠️ 分位必须在**候选池内**算，不能用全市场。
        #    实测踩过：用全市场 20% 分位去切一个 16 只的大盘股池，16 只全被剔光
        #    （大盘股在全市场的该因子分位都在低端），覆盖率还算出 350%。
        #    这与"流动性下限用一条全市场门槛会静默取消北交所选股权"是同一个坑：
        #    池子的构成与全市场不同时，全市场分位就是错的尺子。
        s = pd.Series(v.to_numpy(), index=cur["code"].astype(str).to_numpy())
        s = s[~s.index.duplicated(keep="last")]
        s = s[s.index.isin(set(codes))].dropna()
        g = {"factor_id": row["factor_id"], "name": row["name"],
             "expr": row["expr"], "pct": row["gate_pct"], "scope": "pool",
             "coverage": round(len(s) / max(len(codes), 1), 3)}
        if len(s) < 20 or g["coverage"] < MIN_COVERAGE:
            g["note"] = f"未生效（池内有效 {len(s)} 只 / 覆盖 {g['coverage']:.0%}）"
            meta["gates"].append(g)
            continue
        direction = int(row.get("direction") or 1)
        # 反向因子（direction=-1）里"值越大越差"⇒ 剔的是高分位。
        # ⚠️ 这两个 quantile 参数反过一次：写成 `quantile(1-pct)` 去剔低分位，
        #    会把**九成**候选剔掉（40 只剔 36 只），而且看上去"闸门生效了"。
        if direction > 0:
            thr = s.quantile(row["gate_pct"])
            bad = set(s[s < thr].index)
        else:
            thr = s.quantile(1 - row["gate_pct"])
            bad = set(s[s > thr].index)
        dropped = [c for c in keep if c in bad]
        g.update({"threshold": round(float(thr), 6), "dropped": len(dropped),
                  "dropped_codes": dropped[:20], "note": ""})
        meta["gates"].append(g)
        if dropped:
            hit_any = True
            meta["dropped"].extend(dropped)
            keep = [c for c in keep if c not in bad]
    meta["applied"] = hit_any
    meta["kept"] = len(keep)
    return keep, meta["dropped"], meta
