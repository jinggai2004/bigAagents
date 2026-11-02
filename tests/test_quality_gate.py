# -*- coding: utf-8 -*-
"""盈利质量闸门（core.financial_pit.apply_quality_gate）测试。

口径来源：2026-10-31 A/B 实验 —— AFG 臂（剔 ocf_to_np 最差 10%）押注段
超额 +2.26% vs 现状 −0.12%；AF 臂（混入等权合成）−9.47% 被否决。
闸门边界纪律与 core.crowding.apply_gate 同一套，这里逐条钉住。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import financial_pit as fp           # noqa: E402


@pytest.fixture()
def fin_files(tmp_path, monkeypatch):
    """临时 FIN_DIR：25 只有财报（值已知），2 只完全无文件。

    ⚠️ 按真实落盘口径写（原始列 ocfps/eps，**没有**派生列 ocf_to_np ——
    那是 add_derived 现算的，latest_visible 必须自己算）。
    """
    d = tmp_path / "financials"
    d.mkdir()
    # 25 只等距 ocf_to_np = ocfps/eps ∈ [0.05, 0.41]
    vals = {f"{600000+i}": round(0.05 + i * 0.015, 4) for i in range(25)}
    for code, v in vals.items():
        pd.DataFrame({
            "pub_date": pd.to_datetime(["2025-04-20"]),
            "ocfps": [v * 1.0],          # eps=1 ⇒ ocf_to_np = v
            "eps": [1.0],
        }).to_parquet(d / f"{code}.parquet", index=False)
    monkeypatch.setattr(fp, "FIN_DIR", d)
    return d, vals


ASOF = "2025-06-30"


def test_gate_drops_bottom_decile(fin_files):
    """池内 10% 分位：27 只（25 可见）⇒ 剔除最差 ~10%。"""
    codes = [f"{600000+i}" for i in range(25)] + ["700001", "700002"]  # 700x 无文件
    keep, dropped, meta = fp.apply_quality_gate(codes, asof=ASOF)
    assert meta["applied"] is True
    assert "600000" in [x["股票代码"] for x in dropped]      # 最差 0.05 必被剔
    assert "600000" not in keep
    assert "600024" in keep                                  # 最好 0.40 附近保留
    # 线性插值 p10 of 25 个等距值 ⇒ 严格小于阈值的恰是最小 2~3 只
    assert 1 <= len(dropped) <= 4
    # 闸门不改池子规模之外的东西：保留 + 剔除 = 全部
    assert len(keep) + len(dropped) == len(codes)


def test_missing_file_is_kept_not_dropped(fin_files):
    """无财报文件的票**保留**（缺 ≠ 差，铁律"缺就是缺"）。"""
    keep, dropped, meta = fp.apply_quality_gate(["700001"], asof=ASOF)
    # 可见值 0 只 < 20 ⇒ 闸门未生效，全部保留
    assert meta["applied"] is False
    assert keep == ["700001"]


def test_too_few_visible_values_disables_gate(fin_files):
    """可见值 < 20 只 ⇒ 分位是噪声，闸门不生效。"""
    codes = [f"{600000+i}" for i in range(10)]               # 10 只有值
    keep, dropped, meta = fp.apply_quality_gate(codes, asof=ASOF)
    assert meta["applied"] is False
    assert "未生效" in meta["note"]
    assert dropped == []
    assert keep == codes                                     # 一个都没剔


def test_low_coverage_disables_gate(fin_files, tmp_path, monkeypatch):
    """覆盖率 < 60% ⇒ 数据不可信，闸门不生效。"""
    codes = [f"{600000+i}" for i in range(25)] + [f"{800000+i}" for i in range(30)]
    # 55 只里 25 只有值 = 45% < 60%
    keep, dropped, meta = fp.apply_quality_gate(codes, asof=ASOF)
    assert meta["applied"] is False
    assert meta["coverage"] < fp.QUALITY_GATE_MIN_COV


def test_pit_boundary_asof_excludes_future_filings(fin_files, tmp_path):
    """asof 之前未公告的财报**不可见**（PIT 纪律）。"""
    d, _ = fin_files
    pd.DataFrame({
        "pub_date": pd.to_datetime(["2025-07-10"]),          # 晚于 asof
        "ocfps": [0.99],
        "eps": [1.0],                                        # ⇒ ocf_to_np 0.99
    }).to_parquet(d / "900001.parquet", index=False)
    vals = fp.latest_visible(["900001"], asof="2025-06-30")
    assert "900001" not in vals                              # 未来公告看不到
    vals2 = fp.latest_visible(["900001"], asof="2025-08-01")
    assert vals2.get("900001") == pytest.approx(0.99)        # 公告后可见


def test_gate_never_drops_when_all_equal(fin_files):
    """全部相同值 ⇒ 没有严格小于阈值的票 ⇒ 不剔除、不报错。"""
    d, _ = fin_files
    for i in range(25):
        pd.DataFrame({
            "pub_date": pd.to_datetime(["2025-04-20"]),
            "ocfps": [0.20],
            "eps": [1.0],
        }).to_parquet(d / f"{810000+i}.parquet", index=False)
    keep, dropped, meta = fp.apply_quality_gate([f"{810000+i}" for i in range(25)],
                                                asof=ASOF)
    assert not dropped                                       # 恒不剔（阈值=值本身）


def test_empty_codes_pool(fin_files):
    keep, dropped, meta = fp.apply_quality_gate([], asof=ASOF)
    assert meta["applied"] is False and keep == []
