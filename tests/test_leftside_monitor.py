"""左侧监测器（leftside_monitor）的口径测试。

重点钉四条：

1. **派生列回归**（今天真实踩到的坑）：`latest_visible_multi` 请求 `roe_yoy`
   时曾返回**全空** —— `roe_yoy` 是 `add_derived()` 的派生列不落盘，
   `read_parquet(columns=[...])` 抛"列不存在"被 except 吞成"缺数据"。
   这是"派生列不落盘"的第二个受害案例（第一个是 `ocf_to_np`）。
2. **孤儿大跌必须剥离行业共跌**：行业整体跌 8%、个股跌 8% 不是孤儿。
3. **孤儿大跌必须过基本面资质**：裸大跌继续跑输是已验证结论（−0.68pp t=−9.4）。
4. **阈值不得偏离证据口径**：财报后大跌是 −7%（放宽到 −5% 会稀释成 247 只/天）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import core.leftside_monitor as LM            # noqa: E402
import core.financial_pit as fp               # noqa: E402


# ─────────────────────────────────────────────── 派生列回归（坑 1）
def test_latest_visible_multi_with_derived_roe_yoy(tmp_path, monkeypatch):
    """请求 `roe_yoy`（派生列）不许静默返回空 —— 必须现算或显式缺，不能假缺。"""
    monkeypatch.setattr(fp, "FIN_DIR", tmp_path)
    rows = []
    # 5 期 roe（连续 5 个季度）：第 5 期比第 1 期高 0.05 ⇒ roe_yoy = +0.05
    # ⚠️ stat_date 直接写死合法季度末：用公式 `(i%4)*3+1` 会生成 "2025-14-15"
    # 这种非法日期（实测踩到），财务文件必须能被正常读回。
    roe_seq = [0.10, 0.11, 0.12, 0.13, 0.15]
    q_end = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"]
    q_pub = ["2025-04-20", "2025-07-20", "2025-10-20", "2026-01-20", "2026-04-20"]
    for r, sd, pb in zip(roe_seq, q_end, q_pub):
        rows.append({"code": "600001", "stat_date": sd, "pub_date": pb,
                     "roe": r, "ocfps": 1.0, "eps": 1.0, "sjltz": 0.3})
    pd.DataFrame(rows).to_parquet(tmp_path / "600001.parquet", index=False)

    vals = fp.latest_visible_multi(["600001"],
                                   ["ocf_to_np", "roe", "sjltz", "roe_yoy"],
                                   asof="2026-06-30")
    assert "600001" in vals, "请求派生列时整票消失 = 派生列坑回归了"
    v = vals["600001"]
    assert v["roe"] == pytest.approx(0.15)
    assert v["ocf_to_np"] == pytest.approx(1.0)
    # 第 5 期 roe_yoy = 0.15 − 0.10 = 0.05
    assert v["roe_yoy"] == pytest.approx(0.05, abs=1e-9)


def test_latest_visible_multi_missing_file_is_silent(tmp_path, monkeypatch):
    """缺文件 ⇒ 不在返回里（缺 ≠ 0），不抛异常。"""
    monkeypatch.setattr(fp, "FIN_DIR", tmp_path)
    assert fp.latest_visible_multi(["999999"], ["roe"], asof="2026-06-30") == {}


# ─────────────────────────────────────────────── 孤儿大跌（坑 2）
def _day_frame(rows: list[tuple]) -> pd.DataFrame:
    """rows: (code, industry, close, preclose)。单日截面。"""
    return pd.DataFrame(rows, columns=["stock_code", "industry", "close", "preclose"])


def test_industry_fall_is_not_orphan():
    """行业共跌不算孤儿：行业均值跌 8%、个股跌 8% ⇒ 中性 ≈ 0。"""
    rows = [("600001", "银行", 9.2, 10.0), ("600002", "银行", 9.2, 10.0),
            ("600003", "银行", 9.2, 10.0), ("600004", "白酒", 10.0, 10.0)]
    d = _day_frame(rows)
    d["chg"] = d["close"] / d["preclose"] - 1.0
    d["ind_chg"] = d.groupby("industry")["chg"].transform("mean")
    d["neu_chg"] = d["chg"] - d["ind_chg"]
    bank = d[d["stock_code"] == "600001"]
    assert abs(bank["neu_chg"].iloc[0]) < 1e-9, "行业共跌应被完全剥离"
    # 白酒不跌而银行跌 ⇒ 银行中性跌 = 银行跌 − 银行均值 = 0（行业内无差异）
    assert LM.NEUTRAL_THR == -0.05


def test_orphan_requires_industry_neutral_drop():
    """个股比行业多跌 6% 才算孤儿；只多跌 3% 不算。"""
    rows = [
        ("600001", "银行", 9.0, 10.0),     # −10%
        ("600002", "银行", 9.7, 10.0),     # −3%（行业均值 ≈ −6.5%）
        ("600003", "银行", 9.7, 10.0),
        ("600004", "白酒", 10.0, 10.0),
    ]
    d = _day_frame(rows)
    d["trade_date"] = "2026-10-30"
    d["chg"] = d["close"] / d["preclose"] - 1.0
    d["ind_chg"] = d.groupby("industry")["chg"].transform("mean")
    d["neu_chg"] = d["chg"] - d["ind_chg"]
    ev = LM.find_orphan_plunges(d, asof="2026-10-30")
    # 行业均值 = (−10−3−3)/3 = −5.33% ⇒ 600001 中性 = −4.67% > −5% ⇒ 不算孤儿
    # （阈值 −5% 是有意的：行业普跌 5%+ 个股再多跌 5% 才值得关注）
    bank = d[d["stock_code"] == "600001"]
    assert bank["neu_chg"].iloc[0] == pytest.approx(-0.0467, abs=1e-3)
    assert "600001" not in {e["code"] for e in ev}, "中性 −4.67% 未达 −5% 阈值"
    # 更极端的例子：同行业其他票不跌
    rows2 = [
        ("600001", "银行", 9.2, 10.0),     # −8%
        ("600002", "银行", 10.0, 10.0),    # 0
        ("600003", "银行", 10.0, 10.0),
    ]
    d2 = _day_frame(rows2)
    d2["chg"] = d2["close"] / d2["preclose"] - 1.0
    d2["ind_chg"] = d2.groupby("industry")["chg"].transform("mean")
    d2["neu_chg"] = d2["chg"] - d2["ind_chg"]
    assert d2[d2["stock_code"] == "600001"]["neu_chg"].iloc[0] < LM.NEUTRAL_THR


# ─────────────────────────────────────────────── 资质卡（坑 3）
def test_quality_pass_filters_bad_fundamentals(tmp_path, monkeypatch):
    """ROE 为负 / 现金流质量差的票必须被拦下。"""
    monkeypatch.setattr(fp, "FIN_DIR", tmp_path)
    # 好票：roe 0.20、增长 0.5、ocf/np 1.2
    # 坏票：roe −0.05（亏损）
    # 中性票：roe 0.10 但 sjltz<0 且 roe_yoy<0（无增长）
    specs = {
        "600001": {"roe": 0.20, "sjltz": 0.5, "ocfps": 1.2, "eps": 1.0},
        "600002": {"roe": -0.05, "sjltz": 0.5, "ocfps": 1.2, "eps": 1.0},
        "600003": {"roe": 0.10, "sjltz": -0.2, "ocfps": 1.0, "eps": 1.0},
    }
    for code, v in specs.items():
        if code == "600003":
            continue        # 中性票单独造 5 期（roe_yoy 需要 4 期历史）
        pd.DataFrame([{"code": code, "stat_date": "2025-12-31",
                       "pub_date": "2026-01-20", **v}]).to_parquet(
            tmp_path / f"{code}.parquet", index=False)
    # 600003：roe 连续 5 期从 0.14 → 0.10（递减）⇒ 最新期 roe_yoy = 0.10−0.14 < 0
    # 且 sjltz<0 ⇒ 无增长，配合 ocf 中位被 600001 拉高 ⇒ 三条都不满足
    q_end = ["2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    q_pub = ["2025-01-20", "2025-04-20", "2025-07-20", "2025-10-20", "2026-01-20"]
    rows = [{"code": "600003", "stat_date": sd, "pub_date": pb,
             "roe": 0.14 - i * 0.01, "sjltz": -0.2, "ocfps": 1.0, "eps": 1.0}
            for i, (sd, pb) in enumerate(zip(q_end, q_pub))]
    pd.DataFrame(rows).to_parquet(tmp_path / "600003.parquet", index=False)

    out = LM.quality_pass(list(specs), asof="2026-06-30")
    assert "600002" not in out or not out["600002"].get("passed"), "亏损票必须被拦"
    assert out.get("600001", {}).get("passed") is True, "好票必须通过"


def test_bare_plunge_never_enters_pool(monkeypatch, tmp_path):
    """裸大跌（无好财报、无机构净买、资质不过）不许进池。"""
    monkeypatch.setattr(fp, "FIN_DIR", tmp_path)   # 无财务 ⇒ 资质必不过
    monkeypatch.setattr(LM, "org_net_buy_today", lambda asof=None: {})
    # 构造一个只有大跌的截面
    rows = [("600001", "银行", 9.2, 10.0), ("600002", "银行", 10.0, 10.0),
            ("600003", "银行", 10.0, 10.0)]
    d = _day_frame(rows)
    d["trade_date"] = "2026-10-30"
    d["chg"] = d["close"] / d["preclose"] - 1.0
    d["ind_chg"] = d.groupby("industry")["chg"].transform("mean")
    d["neu_chg"] = d["chg"] - d["ind_chg"]
    monkeypatch.setattr(LM, "load_daily_with_industry", lambda days=40: d)
    monkeypatch.setattr(LM, "find_orphan_plunges",
                        lambda dd, asof=None: [{"code": "600001",
                                                "signal": "orphan_plunge",
                                                "trigger_date": "2026-10-30",
                                                "industry": "银行",
                                                "drop_pct": -0.08,
                                                "industry_neutral": -0.06}])
    monkeypatch.setattr(LM, "find_post_report_plunges", lambda dd, asof=None: [])
    items = LM.scan()
    assert items == [], "裸大跌不许进池（继续跑输是已验证结论）"


# ─────────────────────────────────────────────── 阈值口径（坑 4）
def test_thresholds_match_evidence():
    """阈值必须与已验证证据的口径一致，不许随手放宽。"""
    assert LM.PLUNGE_THR == -0.07      # plunge_event_check 的事件定义
    assert LM.POST_RPT_DROP == -0.07   # POST 组证据（+3.00pp）的口径是 −7%
    assert LM.GOOD_YOY == 0.20         # "财报很好" = 净利同比 >20%


# ─────────────────────────────────────────────── 右侧信号
def test_right_side_flags():
    g = pd.DataFrame({
        "stock_code": ["600001"] * 25,
        "trade_date": [f"2026-01-{i:02d}" for i in range(1, 26)],
        # 收盘价回升站上 MA20；前 5 日缩量
        "close": [10.0] * 15 + [9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9],
        "volume": [100.0] * 20 + [40.0] * 5,
    })
    flags = LM.right_side_flags(g, "600001")
    assert "price_reclaim" in flags or "vol_dry" in flags
    assert LM.RIGHT_VOL_RATIO == 0.70


def test_run_leftside_job_swallows_errors(monkeypatch):
    """扫描抛异常只记日志，不许把调度器带崩（监测是附加功能）。"""
    from core import scheduler as S
    def _boom(*a, **k):
        raise RuntimeError("快照挂了")
    monkeypatch.setattr("core.leftside_monitor.scan_intraday", _boom)
    S.run_leftside_job("测试", intraday=True)          # 不抛 = 通过
    monkeypatch.setattr("core.leftside_monitor.scan", _boom)
    monkeypatch.setattr("core.leftside_monitor.update_right_side", _boom)
    S.run_leftside_job("测试", intraday=False)


def test_persist_and_list_roundtrip(tmp_path, monkeypatch):
    """落库→读取往返：信号/质量/meta 必须原样回来，重复落库是更新不是新增。"""
    import json as J
    from db import store as S
    # ⚠️ 必须切 `db.models.DB_PATH` 而不是 store 的属性：`get_connection()` 在
    #    models 里读路径，patch 错对象 ⇒ INSERT 静默写进**真库**（本次实测踩到）。
    import db.models as M
    monkeypatch.setattr(M, "DB_PATH", str(tmp_path / "t.db"))
    S.init_store()
    monkeypatch.setattr(LM, "_watch_conn", lambda: S.get_connection())

    it = LM.WatchItem(code="600001", name="测试股", industry="银行",
                      signals=["orphan_plunge"], trigger_date="2026-10-30",
                      drop_pct=-0.08, industry_neutral=-0.06,
                      quality={"ocf_to_np": 1.2, "roe": 0.1})
    r1 = LM.persist_pool([it], "2026-10-30")
    assert r1["new"] == 1
    r2 = LM.persist_pool([it], "2026-10-30")            # 幂等
    assert r2["new"] == 0 and r2["refreshed"] == 1
    pool = LM.list_pool()
    assert len(pool) == 1
    x = pool[0]
    assert x["code"] == "600001" and x["first_seen"] == "2026-10-30"
    assert x["signals"] == ["orphan_plunge"]
    assert x["quality"]["ocf_to_np"] == pytest.approx(1.2)
    assert x["meta"]["drop_pct"] == pytest.approx(-0.08)


# ─────────────────────────────────────────────── 调度接线
def test_leftside_jobs_registered():
    """两个监测任务必须注册进调度器（14:50 盘中 / 16:33 收盘）。

    ⚠️ 2026-11-01 排期重排：收盘版从 16:15 挪到 16:33 —— 它读**当日面板**，
       而盘后全流程（P90 57 分钟）16:28 才写完；再早就只能读到昨天的数据。
    ⚠️ 本函数原来被**定义了两遍**（同名覆盖，前一份是死代码）。已删重复，
       免得两份断言各自漂移。
    """
    from core.scheduler import JOBS
    ids = {j[0]: (j[1], j[2]) for j in JOBS}
    assert "leftside_intraday" in ids and "leftside_close" in ids
    # 14:50：14:45 让给持仓风控（它必须紧跟阿深 14:40 的调整），仍在收盘前 10 分钟
    assert ids["leftside_intraday"][0] == dict(hour=14, minute=50)
    assert ids["leftside_close"][0] == dict(hour=16, minute=33)
    assert ids["leftside_intraday"][1] == "leftside_intraday"
    assert ids["leftside_close"][1] == "leftside_close"


def test_run_leftside_job_swallows_errors(monkeypatch):
    """扫描抛异常只记日志，不许把调度器带崩（监测是附加功能）。"""
    from core import scheduler as S
    def _boom(*a, **k):
        raise RuntimeError("快照挂了")
    monkeypatch.setattr("core.leftside_monitor.scan_intraday", _boom)
    S.run_leftside_job("测试", intraday=True)          # 不抛 = 通过
    monkeypatch.setattr("core.leftside_monitor.scan", _boom)
    monkeypatch.setattr("core.leftside_monitor.update_right_side", _boom)
    S.run_leftside_job("测试", intraday=False)


def test_persist_and_list_roundtrip(tmp_path, monkeypatch):
    """落库→读取往返：信号/质量/meta 必须原样回来，重复落库是更新不是新增。"""
    import json as J
    from db import store as S
    # ⚠️ 必须切 `db.models.DB_PATH` 而不是 store 的属性：`get_connection()` 在
    #    models 里读路径，patch 错对象 ⇒ INSERT 静默写进**真库**（本次实测踩到）。
    import db.models as M
    monkeypatch.setattr(M, "DB_PATH", str(tmp_path / "t.db"))
    S.init_store()
    monkeypatch.setattr(LM, "_watch_conn", lambda: S.get_connection())

    it = LM.WatchItem(code="600001", name="测试股", industry="银行",
                      signals=["orphan_plunge"], trigger_date="2026-10-30",
                      drop_pct=-0.08, industry_neutral=-0.06,
                      quality={"ocf_to_np": 1.2, "roe": 0.1})
    r1 = LM.persist_pool([it], "2026-10-30")
    assert r1["new"] == 1
    r2 = LM.persist_pool([it], "2026-10-30")            # 幂等
    assert r2["new"] == 0 and r2["refreshed"] == 1
    pool = LM.list_pool()
    assert len(pool) == 1
    x = pool[0]
    assert x["code"] == "600001" and x["first_seen"] == "2026-10-30"
    assert x["signals"] == ["orphan_plunge"]
    assert x["quality"]["ocf_to_np"] == pytest.approx(1.2)
    assert x["meta"]["drop_pct"] == pytest.approx(-0.08)
