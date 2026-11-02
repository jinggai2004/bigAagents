# -*- coding: utf-8 -*-
"""交易拥挤度闸门的守护测试（2026-10-30）。

## 为什么单独一个文件

这条改动有**三处"算术上等价、行为上不等价"**的陷阱，出错时页面照样是
5~10 只看着正常的候选，只有排序悄悄变了：

1. 分位方向反了（拥挤度越高分位越低）⇒ 闸门剔掉的正好是该买的那一端；
2. 行业分位退回"全池分位"时不报告 ⇒ 测的变成"这是哪个行业"，不是"谁更拥挤"；
3. 快照缺失时静默放行 ⇒ "今天没有拥挤的票"与"闸门根本没跑"在页面上无法区分。

所以这里钉住的是**口径一致**（与 `StockScreener` 的分位逐值相等）、
**降级可见**、以及**方向不可反转**。
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import crowding
from core.screener import StockScreener

_ROOT = Path(__file__).resolve().parent.parent


# ================================================================
# 1. 分位：与选股链**逐值相等**
# ================================================================

def test_pct_rank_matches_screener():
    """与 `StockScreener._percentile_map(higher_is_better=True)` 逐值相等。

    ⚠️ 这是"同一把尺子"的机械保证：两套分位若各写一遍，会在同一页面上
       给出互相矛盾的排名，而这种矛盾没有人看得出来。
    """
    rng = np.random.default_rng(20261030)
    raw = {f"{i:06d}": float(v) for i, v in
           enumerate(np.round(rng.normal(0, 1, 300), 2))}   # 四舍五入 ⇒ 制造并列
    mine = crowding.pct_rank(raw)
    theirs = StockScreener._percentile_map(raw, higher_is_better=True)
    assert mine == theirs


def test_pct_rank_direction_is_crowded_is_high():
    """方向钉死：值最大 ⇒ 1.0（最拥挤）。"""
    got = crowding.pct_rank({"a": 1.0, "b": 2.0, "c": 3.0})
    assert got["c"] == 1.0 and got["a"] == 0.0 and got["b"] == 0.5


def test_pct_rank_ties_average_and_edges():
    v = crowding.pct_rank({"a": 1, "b": 3, "c": 3, "d": 3})
    assert v["b"] == v["c"] == v["d"]          # 并列必须同分
    assert v["a"] == 0.0
    # 唯一元素 ⇒ 0.0（"唯一一只 = 不拥挤"，保守方向：不会误剔）
    assert crowding.pct_rank({"x": 5.0}) == {"x": 0.0}
    assert crowding.pct_rank({}) == {}


def test_pct_rank_drops_non_finite():
    """NaN/inf/布尔不参与分位 —— 当成 0 会把整条分布往下压。"""
    got = crowding.pct_rank({"a": 1.0, "b": float("nan"), "c": float("inf"),
                             "d": True, "e": 3.0})
    assert set(got) == {"a", "e"}
    assert got["e"] == 1.0


# ================================================================
# 2. 行业内分位
# ================================================================

def test_industry_pct_rank_grouping_and_pool():
    """两组各 >= min_group 时各排各的；小组合并与缺行业合并进"其余"池。"""
    values, ind = {}, {}
    for i in range(30):                       # 银行：值很小
        c = f"B{i:03d}"
        values[c], ind[c] = 0.1 + i * 0.001, "银行"
    for i in range(30):                       # 电子：值很大
        c = f"E{i:03d}"
        values[c], ind[c] = 5.0 + i * 0.001, "电子"
    for i in range(3):                        # 小组（<20）⇒ 进其余池
        c = f"S{i:03d}"
        # ⚠️ 值必须与缺行业组**不重叠**：首版取 9.0/10.0/11.0，正好与
        #    `8.0 + i` 的 11.0 撞上 ⇒ 并列平均名次让断言假失败（用例 bug，不是实现 bug）。
        values[c], ind[c] = 20.0 + i, "小行业"
    for i in range(5):                        # 缺行业 ⇒ 进其余池
        c = f"N{i:03d}"
        values[c] = 8.0 + i
    pct, stats = crowding.industry_pct_rank(values, ind, min_group=20)
    assert stats == {"industry": 60, "pool": 8, "groups": 2, "missing": 5}
    # 行业内：银行组内最大的那只仍然是 1.0（不会被电子组的高值压到底部）
    assert pct["B029"] == 1.0 and pct["B000"] == 0.0
    assert pct["E029"] == 1.0 and pct["E000"] == 0.0
    # 其余池：自身池内排序（9.0/9.1/9.2 vs 8.0~8.4）
    assert pct["S002"] == 1.0 and pct["N000"] == 0.0


def test_industry_pct_rank_matches_screener():
    """与 `StockScreener._percentile_by_industry` 逐值相等（同一把尺子）。"""
    s = StockScreener()
    rng = np.random.default_rng(7)
    values, ind = {}, {}
    for g in ("银行", "电子", "医药"):
        for i in range(25):
            c = f"{g}{i:02d}"
            values[c] = float(np.round(rng.normal(0, 1), 2))
            ind[c] = g
    for i in range(4):
        values[f"缺{i}"] = float(np.round(rng.normal(0, 1), 2))
    s._industry_cache = ind               # 直接注入，避免读当日快照
    mine, _ = crowding.industry_pct_rank(values, ind)
    theirs, _ = s._percentile_by_industry(values, higher_is_better=True)
    assert mine == theirs


def test_industry_pct_rank_without_industry_reports_pool():
    """行业表不可用 ⇒ 全部进"其余"池，且**必须报告**（不是静默等同旧行为）。"""
    values = {f"C{i:03d}": float(i) for i in range(50)}
    pct, stats = crowding.industry_pct_rank(values, {})
    assert stats["industry"] == 0 and stats["groups"] == 0
    assert stats["pool"] == 50 and stats["missing"] == 50
    assert len(pct) == 50


# ================================================================
# 3. 合成与判据
# ================================================================

def test_crowd_trade_of():
    assert crowding.crowd_trade_of(0.2, 0.8) == pytest.approx(0.5)
    # 只缺一个 ⇒ 用另一个（不是当 0）
    assert crowding.crowd_trade_of(None, 0.9) == pytest.approx(0.9)
    assert crowding.crowd_trade_of(0.3, None) == pytest.approx(0.3)
    assert crowding.crowd_trade_of(None, None) is None


def test_is_gated_boundaries_and_and_semantics():
    assert crowding.is_gated(0.90, 0.80)                    # 边界含
    assert not crowding.is_gated(0.8999, 0.80)
    assert not crowding.is_gated(0.95, 0.7999)
    # 必须是 and 不是 or：只拥堵不涨（或只涨不拥挤）都不该剔
    assert not crowding.is_gated(0.99, 0.10)
    assert not crowding.is_gated(0.10, 0.99)
    # 缺数据 ⇒ 不判（减法机制里，误杀比漏判更糟）
    assert not crowding.is_gated(None, 0.99)
    assert not crowding.is_gated(0.99, None)


# ================================================================
# 4. 闸门套用：边界必须显式
# ================================================================

def _snap(rows: dict, date: str = None) -> dict:
    """构造快照。⚠️ 默认日必须是**今天**：写死某一天，等真实日期走过去之后
    滞后就会超过 `MAX_SNAPSHOT_LAG_DAYS`，这些用例会以"闸门未生效"的方式假失败
    （本项目踩过：测试夹具的日期要落在业务窗口内）。"""
    import time as _t
    date = date or _t.strftime("%Y-%m-%d")
    return {"date": date, "rows": rows, "meta": {}, "available": [date], "note": ""}


def test_apply_gate_no_snapshot_does_not_filter(monkeypatch):
    """"缺数据" ≠ "都是拥挤股" —— 不过滤，且必须在 note 里说明。"""
    monkeypatch.setattr(crowding, "load", lambda as_of=None: {
        "date": None, "rows": {}, "meta": {}, "available": [], "note": "无快照"})
    codes = [f"{i:06d}" for i in range(30)]
    keep, dropped, meta = crowding.apply_gate(codes)
    assert keep == codes and dropped == []
    assert meta["applied"] is False and meta["note"]


def test_apply_gate_drops_and_reports():
    codes = [f"{i:06d}" for i in range(20)]
    rows = {c: {"crowd_trade": 0.1, "mom20": 0.1, "industry": "银行"} for c in codes}
    rows[codes[0]] = {"crowd_trade": 0.95, "mom20": 0.85, "industry": "电子"}
    keep, dropped, meta = crowding.apply_gate(codes, snapshot=_snap(rows))
    assert codes[0] not in keep and len(keep) == 19
    assert dropped[0]["股票代码"] == codes[0]
    assert meta["applied"] is True and meta["date"] == _snap({})["date"]
    assert "剔除 1 只" in meta["note"]


def test_apply_gate_stale_snapshot_does_not_filter():
    """快照过期（> MAX_SNAPSHOT_LAG_DAYS 个交易日）⇒ 不过滤，并说明原因。

    "用一周前的拥挤度砍今天的候选"比不砍更糟 —— 而且必须说出来，
    不能表现为"今天恰好没有拥挤的票"。
    """
    codes = [f"{i:06d}" for i in range(20)]
    rows = {c: {"crowd_trade": 0.99, "mom20": 0.99, "industry": "x"} for c in codes}
    keep, dropped, meta = crowding.apply_gate(codes, snapshot=_snap(rows, "2020-01-02"))
    assert keep == codes and dropped == [] and meta["applied"] is False
    assert meta["lag_days"] is not None and meta["lag_days"] > crowding.MAX_SNAPSHOT_LAG_DAYS
    assert "滞后" in meta["note"]


def test_apply_gate_missing_code_is_kept():
    """快照里没有这只 ⇒ 保留（不因为"查不到"就剔）。"""
    codes = [f"{i:06d}" for i in range(15)]
    rows = {c: {"crowd_trade": 0.9, "mom20": 0.9, "industry": "x"} for c in codes[:1]}
    keep, dropped, meta = crowding.apply_gate(codes, snapshot=_snap(rows))
    # 只有 codes[0] 命中 ⇒ 14 只保留（>= min_keep 10）⇒ 过滤生效
    assert codes[0] not in keep and meta["applied"] is True


def test_apply_gate_gives_up_when_pool_too_small():
    """剩余不足 min_keep ⇒ 放弃过滤（池子已很小，再砍只剩噪声）。"""
    codes = [f"{i:06d}" for i in range(12)]
    rows = {c: {"crowd_trade": 0.95, "mom20": 0.95, "industry": "x"} for c in codes}
    keep, dropped, meta = crowding.apply_gate(codes, snapshot=_snap(rows), min_keep=10)
    assert keep == codes and dropped == [] and meta["applied"] is False
    assert "放弃过滤" in meta["note"]


def test_apply_gate_env_switch(monkeypatch):
    monkeypatch.setenv("BIGA_CROWD_GATE", "0")
    codes = [f"{i:06d}" for i in range(20)]
    rows = {c: {"crowd_trade": 0.99, "mom20": 0.99, "industry": "x"} for c in codes}
    keep, _dropped, meta = crowding.apply_gate(codes, snapshot=_snap(rows))
    assert keep == codes and meta["applied"] is False
    assert "已关闭" in meta["note"]


# ================================================================
# 5. 快照读写
# ================================================================

def test_snapshot_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(crowding, "snapshot_dir", lambda: str(tmp_path))
    res = {"date": "2026-10-29", "rows": [
        {"code": "000001", "crowd_trade": 0.5, "mom20": 0.4, "turn": 0.9,
         "amount": 2.3e9, "ret20": 0.05, "industry": "银行"},
        {"code": "600000", "crowd_trade": None, "mom20": 0.9, "turn": 1.1,
         "amount": 1.0e9, "ret20": 0.31, "industry": "银行"},
    ], "meta": {"date": "2026-10-29", "gated": 0}}
    p = crowding.save(res)
    assert p and Path(p).exists()
    got = crowding.load()
    assert got["date"] == "2026-10-29"
    assert got["rows"]["000001"]["crowd_trade"] == pytest.approx(0.5)
    assert got["rows"]["600000"]["crowd_trade"] is None        # None 必须原样往返
    assert got["meta"]["gated"] == 0


def test_snapshot_load_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(crowding, "snapshot_dir", lambda: str(tmp_path))
    got = crowding.load()
    assert got["date"] is None and got["rows"] == {}
    assert "crowding" in got["note"]           # 提示怎么生成，而不是空话


# ================================================================
# 6. compute：口径与降级可见
# ================================================================

def _fake_rows():
    rows = []
    for i in range(30):                        # 银行：换手低但逐只递增
        rows.append({"code": f"B{i:03d}", "turn": 0.5 + i * 0.01,
                     "amount": 1e8 + i * 1e6, "industry_ts": "银行"})
    for i in range(30):                        # 电子：换手高
        rows.append({"code": f"E{i:03d}", "turn": 8.0 + i * 0.1,
                     "amount": 1e9 + i * 1e7, "industry_ts": "电子"})
    return pd.DataFrame(rows)


def test_compute_is_industry_neutral(monkeypatch):
    """银行组内最高换手的那只，其分位应接近 1 —— 与电子组无关。

    这是"必须行业内"的机械证明：若走全池分位，银行全部会落在最低档
    （换手 0.5 vs 8.0），拥挤度闸门对银行永不生效，而页面看不出。
    """
    frame = _fake_rows()
    monkeypatch.setattr(crowding, "read_snapshot_rows", lambda d: frame)
    monkeypatch.setattr(crowding, "sw_industry_map",
                        lambda path=None: {**{f"B{i:03d}": "银行" for i in range(30)},
                                           **{f"E{i:03d}": "电子" for i in range(30)}})
    res = crowding.compute(as_of="2026-10-29",
                           ret20={c: 0.01 * i for i, c in enumerate(frame["code"])})
    rows = {r["code"]: r for r in res["rows"]}
    assert res["meta"]["industry_groups"] == 2
    assert rows["B029"]["crowd_trade"] > 0.9      # 银行组内最拥挤（换手最高）
    assert rows["B000"]["crowd_trade"] < 0.1
    assert all(r["industry"] for r in res["rows"])
    # 涨幅可用数与注入的 ret20 对齐
    assert res["meta"]["mom_n"] == len(frame)


def test_compute_reports_degraded_without_industry(monkeypatch):
    """行业表缺失 ⇒ 必须写进 meta["degraded"]，否则"全池分位"会被当成"行业内分位"。"""
    frame = _fake_rows()
    monkeypatch.setattr(crowding, "read_snapshot_rows", lambda d: frame)
    monkeypatch.setattr(crowding, "sw_industry_map", lambda path=None: {})
    res = crowding.compute(as_of="2026-10-29", ret20={c: 0.0 for c in frame["code"]})
    assert res["meta"]["degraded"]
    assert "退化为全池分位" in res["meta"]["degraded"][0]


# ================================================================
# 7. 结构性守护：它是闸门，不是因子
# ================================================================

def test_not_in_factor_weights():
    """拥挤度**不得**进评分权重：价量因子 IC 全负（11/11 年），
    进了评分就会变成"越拥挤分数越高"，方向正好反了。"""
    from core.sector import FACTOR_WEIGHTS
    assert not any("拥挤" in k or "crowd" in k.lower() for k in FACTOR_WEIGHTS)


def test_score_single_does_not_use_crowding():
    """`_score_single` 里不得出现拥挤度 —— 评分链与闸门是两条路，不许接线。"""
    src = (_ROOT / "core" / "screener.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_score_single")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert "crowding" not in names
    assert "crowd_trade" not in names


def test_run_reports_crowding_gate_in_result():
    """`run()` 的结果字典必须带 `crowding_gate`。

    ⚠️ 用 AST 找 dict 字面量里的字符串键，而不是 grep 源码 ——
       注释里也写着 `crowding_gate`，文本断言会被自己的注释命中（本项目踩过）。
    """
    src = (_ROOT / "core" / "screener.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    keys = {n.value for n in ast.walk(fn)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "crowding_gate" in keys
    calls = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert "_apply_crowding_gate" in calls


def test_gate_is_wired_into_run_before_scoring():
    """闸门必须在打分之**前**（否则被剔的票已经进了 ranked/候选池）。"""
    src = (_ROOT / "core" / "screener.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    order = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr in ("_apply_crowding_gate", "_score_candidates")]
    by = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr in ("_apply_crowding_gate", "_score_candidates"):
            by[n.func.attr] = n.lineno
    assert by["_apply_crowding_gate"] < by["_score_candidates"]
    assert order
