# -*- coding: utf-8 -*-
"""因子实验室（2026-11-01）：用户自定义因子的导入 / 回测 / 应用。

用户的原话是"目前因子都是你导入的，怎么做到用户进行因子的导入回测应用"。
这件事的真难点不是"让用户填一行公式"，而是**填完之后凭什么敢用**——所以本文件
钉的是那些"看起来跑通了、其实结论不可信"的地方：

1. **表达式不能用代码的理由**：`fwd20`（未来收益）必须结构性不可引用。放行一句
   `fwd20`，IC 就是 1.0、DSR 必过——这不是刁难，是回测里最常见的自欺。
2. **试验次数必须累计**：只把候选自己当试验 ⇒ `n_trials=1` ⇒ 任何因子都"显著"。
   所以 `backtest` 的试验集恒为「内置 30 个 + 已登记用户因子 + 自己」。
3. **DSR 结果会读成 None**：`adjust_by_multiple_testing` 是就地改行，传副本就拿到
   `significant=None`，于是每个因子都被判"未过多重检验"（首版实测踩到）。
4. **冗余不能被自己的输入判死**：`lndvol20` 与 `lndvol20` 相关恒 1.00，
   不排除自身会误杀所有单字段因子。
5. **闸门的边界纪律**：缺值的票不剔、有效值太少不生效、覆盖不足不生效。
6. **应用要有落地分支**：`set_status('applied')` 只是门票，选股链没有对应阶段
   就等于"动作静默消失"（阿深 set_wake 的教训）。这里直接断言 screener 里存在
   该阶段，并断言"没有已应用因子时零开销返回"。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import factor_dsl as D                   # noqa: E402
from core import factor_lab as FL                  # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    import db.models as M
    monkeypatch.setattr(M, "DB_PATH", str(tmp_path / "t.db"))
    from db import store as S
    S.init_store()
    return S


# ---------------------------------------------------------------- 表达式白名单
@pytest.mark.parametrize("expr", [
    "amihud20 / lndvol20",
    "cs_rank(ivol20) - cs_rank(turn20)",
    "log(amihud20 + 1)",
    "clip(turn20, 0, 0.5)",
    "turn20 > 0.02",
    "(vol20 > 0.3) and (turn20 > 0.01)",
    "ocf_to_np",
    "maximum(vol20, dvol20)",
])
def test_valid_expressions(expr):
    v = D.validate(expr)
    assert v["fields"], "至少引用一个字段"
    assert isinstance(v["ast"].body, object)


@pytest.mark.parametrize("expr,needle", [
    ("fwd20", "未知字段"),                       # 目标列必须不可引用
    ("code", "未知字段"),
    ("month_ts", "未知字段"),
    ("__import__('os')", "双下划线"),
    ("turn20.__class__", "双下划线"),
    ("turn20[0]", "Subscript"),
    ("lambda: turn20", "Lambda"),
    ("[x for x in [1]]", "ListComp"),
    ("{'a': 1}", "Dict"),
    ("turn20 if 1 else 2", "IfExp"),             # 条件表达式不给放行，用 where/clip
    ("abs", "未知字段"),                          # 裸函数名=当成字段，必然未知
    ("1 + 2", "没有任何字段"),
    ("True", "只允许数字常量"),
    ("os.system('x')", "方法调用"),
])
def test_rejected_expressions(expr, needle):
    with pytest.raises(D.FactorExprError) as ei:
        D.validate(expr)
    assert needle in str(ei.value)


def test_unknown_field_suggests_typo():
    """拼错字段要给出近似名——只报"未知字段"会让人对着清单一个个数。"""
    with pytest.raises(D.FactorExprError) as ei:
        D.validate("ivvol20")
    assert "vol20" in str(ei.value)


def test_expr_len_limit():
    with pytest.raises(D.FactorExprError) as ei:
        D.validate("turn20 + " * 80 + "turn20")
    assert "过长" in str(ei.value)


# ---------------------------------------------------------------- 求值
def _panel():
    """两个月 × 4 只的合成面板（不碰真实数据：测试不得耦合真实环境）。"""
    return pd.DataFrame({
        "month_ts": pd.to_datetime(["2024-01-31"] * 4 + ["2024-02-29"] * 4),
        "code": ["000001", "000002", "000003", "000004"] * 2,
        "turn20": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        "vol20": [1.0, 0.0, 2.0, 4.0, 1.0, 0.0, 2.0, 4.0],
    })


def test_evaluate_arith_and_div_zero():
    p = _panel()
    out = D.evaluate("turn20 / vol20", p)
    assert out.notna().sum() == 6                  # 除零的两格是 NaN
    assert not np.isinf(out).any(), "除零不能产生 inf（会让 rank 把它排到最前）"


def test_cs_rank_is_within_month():
    p = _panel()
    out = D.evaluate("cs_rank(turn20)", p)
    # 两个月各 4 只 ⇒ 每月百分位都是 0.25/0.5/0.75/1.0，与绝对量级无关
    assert sorted(out.round(3).unique().tolist())[:4] == [0.25, 0.5, 0.75, 1.0]
    assert (out.groupby(p["month_ts"]).max() == 1.0).all()


def test_missing_column_is_nan_not_zero():
    """字段在清单里但面板没这列 ⇒ 如实 NaN（缺就是缺，不返 0）。"""
    p = _panel()
    out = D.evaluate("revenue", p)          # 合成面板里没有 revenue
    assert out.isna().all()


def test_compare_and_boolop_return_numeric():
    p = _panel()
    out = D.evaluate("(turn20 > 2) and (vol20 > 0)", p)
    assert set(out.dropna().unique()) <= {0.0, 1.0}


def test_result_must_be_series():
    """`evaluate` 拿到常数结果必须报错，不能静默返回标量（下游会 `.notna()` 炸）。"""
    import ast
    p = _panel()
    v2 = {"ast": ast.parse("1", mode="eval"), "fields": ["turn20"], "funcs": []}
    with pytest.raises(D.FactorExprError):
        D.evaluate("turn20", p, validated=v2)


# ---------------------------------------------------------------- 冗余判据
def _redundancy_panel(months=6, n=120):
    """合成面板：`_uf` 与 `lndvol20` 完全同秩，`amihud20` 是独立噪音。"""
    rng = np.random.default_rng(7)
    rows = []
    for i, m in enumerate(pd.date_range("2023-01-31", periods=months, freq="ME")):
        order = rng.permutation(n)
        for k in range(n):
            v = float(order[k])
            rows.append({"month_ts": m, "code": f"{k:06d}", "_uf": v,
                         "lndvol20": v, "amihud20": float(rng.normal())})
    return pd.DataFrame(rows)


def test_redundancy_excludes_own_inputs():
    """与自身输入的相关是**构成关系**不是重复；不排除会误杀所有单字段因子。"""
    p = _redundancy_panel()
    # 不排除：与 lndvol20（就是它自己）rho=1.0 ⇒ 被判重复（这是当初的 bug）
    r_all = FL.redundancy(p, "_uf", exclude=[])
    assert r_all["duplicate"] is True
    assert r_all["top"][0]["factor"] == "lndvol20"
    # 排除自身输入：判定只看向非输入因子 ⇒ 不再被判重复
    r_ext = FL.redundancy(p, "_uf", exclude=["lndvol20"])
    assert any(x["own"] for x in r_ext["top"]), "输入仍应展示（供人判断）"
    assert r_ext["duplicate"] is False
    assert abs(r_ext["max_rho"]) < FL.REDUNDANT_RHO


# ---------------------------------------------------------------- 落库与状态
def test_register_requires_name_and_rejects_bad_gate(tmp_db, monkeypatch):
    monkeypatch.setattr(FL, "backtest", lambda e, n=None: {
        "ok": True, "usable": True, "ic_mean": 0.05, "t_stat": 3.0,
        "direction": 1, "dsr": {"significant": True}, "verdict": "可用于闸门"})
    with pytest.raises(ValueError):
        FL.register("turn20", "")
    with pytest.raises(ValueError):
        FL.register("turn20", "x", kind="nope")
    with pytest.raises(ValueError):
        FL.register("turn20", "x", gate_pct=0.9)


def test_rename_keeps_factor_id(tmp_db, monkeypatch):
    """改名不许换 ID：换了 ID，试验集里这条因子的历史贡献就断了（等于重置 hurdle）。"""
    monkeypatch.setattr(FL, "backtest", lambda e, n=None: {
        "ok": True, "usable": False, "ic_mean": 0.001, "t_stat": 0.3,
        "direction": 1, "dsr": {"significant": False}, "verdict": "不可用"})
    r1 = FL.register("turn20", "原名")["factor"]
    r2 = FL.register("turn20 * 2", "改后的名", factor_id=r1["factor_id"])["factor"]
    assert r2["factor_id"] == r1["factor_id"]
    assert r2["name"] == "改后的名"
    assert r2["expr"] == "turn20 * 2"


def test_applied_blocked_when_not_usable(tmp_db, monkeypatch):
    monkeypatch.setattr(FL, "backtest", lambda e, n=None: {
        "ok": True, "usable": False, "ic_mean": 0.001, "t_stat": 0.3,
        "direction": 1, "dsr": {"significant": False}, "verdict": "不可用"})
    fid = FL.register("turn20", "差的")["factor"]["factor_id"]
    with pytest.raises(ValueError) as ei:
        FL.set_status(fid, "applied")
    assert "没通过检验" in str(ei.value)


def test_score_kind_must_pass_ab(tmp_db, monkeypatch):
    """分数类因子不许从页面直接启用——AF 臂押注段 −9.47% 的教训。"""
    monkeypatch.setattr(FL, "backtest", lambda e, n=None: {
        "ok": True, "usable": True, "ic_mean": 0.06, "t_stat": 4.0,
        "direction": 1, "dsr": {"significant": True}, "verdict": "可用于闸门"})
    fid = FL.register("turn20", "想进分数", kind="score")["factor"]["factor_id"]
    with pytest.raises(ValueError) as ei:
        FL.set_status(fid, "applied")
    assert "A/B" in str(ei.value)


def test_gate_kind_can_be_applied(tmp_db, monkeypatch):
    monkeypatch.setattr(FL, "backtest", lambda e, n=None: {
        "ok": True, "usable": True, "ic_mean": 0.06, "t_stat": 4.0,
        "direction": 1, "dsr": {"significant": True}, "verdict": "可用于闸门"})
    fid = FL.register("turn20", "好因子")["factor"]["factor_id"]
    assert FL.set_status(fid, "applied")["status"] == "applied"
    assert [g["factor_id"] for g in FL.applied_gates()] == [fid]


# ---------------------------------------------------------------- 闸门边界
def _gate_panel(n=40, months=2):
    months_ts = pd.date_range("2024-01-31", periods=months, freq="ME")
    rows = []
    for m in months_ts:
        for k in range(n):
            rows.append({"month_ts": m, "code": f"{k:06d}", "turn20": float(k)})
    return pd.DataFrame(rows)


def _setup_gate(monkeypatch, db, pct=0.1, direction=1, n=40):
    """造一个已 applied 的因子 + 合成面板（不打真实库/真面板）。"""
    panel = _gate_panel(n=n)
    monkeypatch.setattr(FL, "load_panel",
                        lambda need_fin=False, force=False: panel)
    row = db.upsert_user_factor({
        "name": "测试闸门", "expr": "turn20", "kind": "gate", "status": "applied",
        "direction": direction, "gate_pct": pct,
        "metrics": {"usable": True}})
    return row, panel


def test_gate_drops_worst_decile(tmp_db, monkeypatch):
    row, panel = _setup_gate(monkeypatch, tmp_db)
    codes = [f"{k:06d}" for k in range(40)]
    keep, _d, meta = FL.apply_gates(codes)
    assert meta["applied"] is True
    assert len(keep) == 36, "40 只剔 10% ⇒ 剔 4 只"
    assert "000000" not in keep, "最低分位必须被剔"
    assert "000039" in keep


def test_gate_reverse_direction(tmp_db, monkeypatch):
    """方向为 −1 时剔的是**高分位**（值越大越差）。"""
    _setup_gate(monkeypatch, tmp_db, direction=-1)
    codes = [f"{k:06d}" for k in range(40)]
    keep, _d, _m = FL.apply_gates(codes)
    assert "000039" not in keep and "000000" in keep


def test_gate_noop_when_coverage_low(tmp_db, monkeypatch):
    """因子只覆盖极少数票 ⇒ 分位没意义，闸门必须不生效（不许瞎剔）。"""
    _setup_gate(monkeypatch, tmp_db)
    codes = [f"{k:06d}" for k in range(10)] + [f"9{k:05d}" for k in range(200)]
    keep, _d, meta = FL.apply_gates(codes)
    assert meta["applied"] is False
    assert len(keep) == len(codes)
    assert "未生效" in meta["gates"][0]["note"]


def test_gate_keeps_unknown_values(tmp_db, monkeypatch):
    """缺值的票不剔——"不知道"不等于"差"。"""
    row, panel = _setup_gate(monkeypatch, tmp_db, n=40)
    codes = [f"{k:06d}" for k in range(40)] + ["888888"]   # 888888 不在面板里
    keep, _d, _m = FL.apply_gates(codes)
    assert "888888" in keep


def test_no_applied_gates_is_noop(tmp_db, monkeypatch):
    called = {"load": 0}
    monkeypatch.setattr(FL, "load_panel",
                        lambda need_fin=False, force=False: called.__setitem__("load", called["load"] + 1))
    keep, _d, meta = FL.apply_gates(["000001"])
    assert called["load"] == 0, "没有已应用因子时不许加载面板（50s 的面板不能白付）"
    assert meta["applied"] is False


# ---------------------------------------------------------------- 选股链接线
def test_screener_has_user_gate_stage():
    """结构性断言：选股链里必须有这一阶段（判据放行≠落地分支）。"""
    src = (_ROOT / "core" / "screener.py").read_text(encoding="utf-8")
    assert "_apply_user_gates" in src
    assert '"user_gate"' in src
    # 进度区间必须登记，否则 percent 回退到 (0,100) 兜底破坏单调性
    assert '"user_gate":' in src


def test_screener_user_gate_short_circuits_without_applied(tmp_db, monkeypatch):
    """没有已应用因子时 `_apply_user_gates` 必须原样返回（不能碰面板）。"""
    from core.screener import StockScreener
    s = StockScreener.__new__(StockScreener)          # 不跑 __init__（会联网/建缓存）
    monkeypatch.setattr(FL, "applied_gates", lambda: [])
    cands = [{"股票代码": "000001"}, {"股票代码": "000002"}]
    keep, meta = StockScreener._apply_user_gates(s, cands)
    assert keep is cands
    assert meta["applied"] is False


# ---------------------------------------------------------------- 面板两档 / 缓存
def test_needs_fin_classification():
    """量价表达式不许付财务那 21 秒；含财务字段才加载财务档。"""
    assert FL.needs_fin(D.validate("amihud20 / lndvol20")["fields"]) is False
    assert FL.needs_fin(D.validate("cs_rank(turn20)")["fields"]) is False
    assert FL.needs_fin(D.validate("ocf_to_np")["fields"]) is True
    assert FL.needs_fin(D.validate("turn20 - roe")["fields"]) is True
    assert FL.needs_fin([]) is False


def test_price_panel_does_not_touch_financials(tmp_path, monkeypatch):
    """结构性守护：`need_fin=False` 时**一次都不许**调 `load_financials`。

    为什么值得一个测试：财务档要读 5,582 个 per-code parquet（21.5 秒），
    而 90% 的因子是纯量价的。哪天有人图省事把财务合并写回无条件的路径上，
    页面的"打开就转圈"会原样回来，而且没人会觉得是 bug。
    """
    import core.financial_pit as fp
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("纯量价因子不该读财务文件")

    monkeypatch.setattr(fp, "load_financials", _boom)
    monkeypatch.setattr(FL, "SCAN_CACHE", tmp_path / "scan.parquet")
    monkeypatch.setattr(FL, "PANEL_CACHE", {"price": tmp_path / "p.parquet",
                                            "fin": tmp_path / "f.parquet"})
    monkeypatch.setattr(FL, "PANEL_META", {"price": tmp_path / "p.json",
                                           "fin": tmp_path / "f.json"})
    monkeypatch.setitem(FL._MEMO, "price", None)
    monkeypatch.setitem(FL._MEMO_STAMP, "price", None)

    scan = pd.DataFrame({
        "month_ts": pd.to_datetime(["2024-01-31"] * 3 + ["2024-02-29"] * 3),
        "code": [f"{i:06d}" for i in range(3)] * 2,
        "turn20": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "fwd20": [0.0] * 6,
    })
    scan.to_parquet(FL.SCAN_CACHE, index=False)

    p = FL.load_panel(need_fin=False)
    assert len(p) == 6
    assert called["n"] == 0, "量价档绝不能调 load_financials"
    assert FL.PANEL_CACHE["price"].exists(), "应落盘缓存"


def test_panel_cache_roundtrip(tmp_path, monkeypatch):
    """落盘缓存要能命中：清掉内存 memo 后，第二次读的是 parquet 而不是重建。"""
    monkeypatch.setattr(FL, "SCAN_CACHE", tmp_path / "scan.parquet")
    monkeypatch.setattr(FL, "PANEL_CACHE", {"price": tmp_path / "p.parquet",
                                            "fin": tmp_path / "f.parquet"})
    monkeypatch.setattr(FL, "PANEL_META", {"price": tmp_path / "p.json",
                                           "fin": tmp_path / "f.json"})
    pd.DataFrame({"month_ts": pd.to_datetime(["2024-01-31"] * 2),
                  "code": ["000001", "000002"],
                  "turn20": [0.01, 0.02], "fwd20": [0.0, 0.0]}).to_parquet(
        FL.SCAN_CACHE, index=False)
    monkeypatch.setitem(FL._MEMO, "price", None)
    monkeypatch.setitem(FL._MEMO_STAMP, "price", None)
    FL.load_panel(need_fin=False)
    assert FL.panel_ready(False) is True

    # 模拟"下一个进程"：内存没了，但缓存文件在
    monkeypatch.setitem(FL._MEMO, "price", None)
    monkeypatch.setitem(FL._MEMO_STAMP, "price", None)
    monkeypatch.setattr(FL, "_build",
                        lambda kind: (_ for _ in ()).throw(
                            AssertionError("缓存命中时不该重建")))
    p = FL.load_panel(need_fin=False)
    assert len(p) == 2


def test_panel_cache_invalidated_by_scan_change(tmp_path, monkeypatch):
    """扫描缓存变了 ⇒ 指纹变 ⇒ 旧缓存必须失效（否则回测悄悄用过期面板）。"""
    monkeypatch.setattr(FL, "SCAN_CACHE", tmp_path / "scan.parquet")
    monkeypatch.setattr(FL, "PANEL_CACHE", {"price": tmp_path / "p.parquet",
                                            "fin": tmp_path / "f.parquet"})
    monkeypatch.setattr(FL, "PANEL_META", {"price": tmp_path / "p.json",
                                           "fin": tmp_path / "f.json"})
    pd.DataFrame({"month_ts": pd.to_datetime(["2024-01-31"]), "code": ["000001"],
                  "turn20": [0.01], "fwd20": [0.0]}).to_parquet(FL.SCAN_CACHE, index=False)
    monkeypatch.setitem(FL._MEMO, "price", None)
    monkeypatch.setitem(FL._MEMO_STAMP, "price", None)
    FL.load_panel(need_fin=False)
    assert FL.panel_ready(False) is True

    import os
    st = FL.SCAN_CACHE.stat()
    os.utime(FL.SCAN_CACHE, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # 明确推后 1 秒
    monkeypatch.setitem(FL._MEMO, "price", None)
    monkeypatch.setitem(FL._MEMO_STAMP, "price", None)
    assert FL.panel_ready(False) is False, "指纹变了就应判缓存失效"
