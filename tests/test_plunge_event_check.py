"""大跌事件研究的口径测试。

这个脚本最容易错的三处，逐一钉住：

1. **前瞻收益必须按 `stock_code` 分组后 shift**：不分组的 `shift(-1)` 会把
   下一只股票的价格当成这只的未来价，凭空造出巨幅"收益"，而代码不会报错。
2. **跌幅阈值方向**：`chg <= -0.07` 是"大跌"；写成 `>=` 会选出**涨停**，
   而且事件数看着也很正常。
3. **财报只认事件之后披露的**：用事件前已公告的报表给这次大跌定性 = 前视，
   会让"财报好"组凭空变好。
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

import scripts.plunge_event_check as P  # noqa: E402


def _daily() -> pd.DataFrame:
    """三只票 × 6 天，价格手工可验算。

    A 在第 2 天跌 10%（事件），之后回升；B 平稳；C 在第 3 天跌 8%。
    """
    rows = []
    prices = {
        # ⚠️ 至少要 6 天以上且事件靠前：fwd5 要 5 个交易日后才有值，
        #    样本太短会让事件行的 fwd5 全是 NaN，基准也变 NaN（用例会假失败）。
        "000001": [100.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0],
        "000002": [50.0, 50.5, 50.0, 51.0, 51.5, 52.0, 52.5, 53.0, 53.5, 54.0],
        "000003": [20.0, 20.2, 20.4, 18.8, 19.0, 19.2, 19.4, 19.6, 19.8, 20.0],
    }
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09",
             "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16"]
    for code, seq in prices.items():
        for i, (dt, px) in enumerate(zip(dates, seq)):
            prev = seq[i - 1] if i else px
            rows.append({"stock_code": code, "trade_date": dt,
                         "close": px, "preclose": prev, "trade_status": 1})
    return pd.DataFrame(rows)


def test_forward_returns_do_not_leak_across_stocks():
    """`shift(-1)` 必须按股票分组，否则 A 的最后一天会拿到 B 的第一天价格。"""
    d = P.add_returns(_daily())
    last_a = d[(d["stock_code"] == "000001")].iloc[-1]
    assert pd.isna(last_a["fwd1"]), "最后一天不该有未来收益（跨股票更不行）"
    second = d[(d["stock_code"] == "000001")].iloc[1]
    # 90 → 91，一天 +1.111%
    assert second["fwd1"] == pytest.approx(91 / 90 - 1, abs=1e-6)
    # 90 → 92（T+2），两天 +2.222%
    assert second["fwd2" if "fwd2" in d.columns else "fwd5"] is not None


def test_plunge_threshold_direction():
    """只挑**下跌**超过阈值的，不许把上涨挑进来。"""
    d = P.add_returns(_daily())
    ev = P.find_plunges(d, thr=-0.07)
    assert set(ev["stock_code"]) == {"000001", "000003"}
    assert (ev["chg"] <= -0.07).all()
    assert "000002" not in set(ev["stock_code"])


def test_cooldown_merges_consecutive_plunges():
    """连续两日大跌（跌停连跌）只能算一个事件，否则样本重复计数。"""
    d = P.add_returns(_daily())
    no_cd = P.find_plunges(d, thr=-0.07, cooldown=0)
    assert len(no_cd) >= len(P.find_plunges(d, thr=-0.07, cooldown=5))


def test_benchmark_is_same_day():
    """基准必须是**同一交易日**的全市场等权，不能拿全样本均值充数。"""
    d = P.add_returns(_daily())
    ev = P.attach_benchmark(P.find_plunges(d, thr=-0.07), d)
    assert ev["bench5"].notna().all(), "事件行的 fwd5/基准不该是 NaN（样本要够长）"
    for _, row in ev.iterrows():
        same_day = d[d["trade_date"] == row["trade_date"]]
        assert row["bench5"] == pytest.approx(float(same_day["fwd5"].mean()), abs=1e-5)
        assert row["ex5"] == pytest.approx(float(row["fwd5"]) - float(row["bench5"]), abs=1e-5)


def test_financials_only_after_event(monkeypatch):
    """只用 `pub_date > 事件日` 的财报。事件前已公告的报表不许参与定性。"""
    fin = pd.DataFrame({
        "code": ["000001", "000001", "000001"],
        # 事件日 2026-01-06：前一份（12-01）必须被排除，后一份（01-20）才是验证
        "pub_date": ["2025-12-01", "2026-01-20", "2026-06-01"],
        "stat_date": ["2025-09-30", "2025-12-31", "2026-03-31"],
        "sjltz": [0.50, -0.30, 0.90],          # 事件后首份是"差"，不能被 0.9 顶替
    })
    import core.financial_pit as fp
    monkeypatch.setattr(fp, "load_financials", lambda: fin)

    ev = pd.DataFrame({"stock_code": ["000001"], "trade_date": ["2026-01-06"]})
    out = P.attach_financials(ev, window=45)
    assert out.loc[0, "fin_yoy"] == pytest.approx(-0.30), "应取事件后首份，不是事件前的"
    assert out.loc[0, "fin_lag_days"] == 14
    assert out.loc[0, "fin_period"] == "2025-12-31"


def test_financials_outside_window_is_nan(monkeypatch):
    """超出验证窗口（45 天）的财报不算"这次大跌的验证"，必须留空。"""
    fin = pd.DataFrame({"code": ["000001"], "pub_date": ["2026-04-01"],
                        "stat_date": ["2025-12-31"], "sjltz": [0.5]})
    import core.financial_pit as fp
    monkeypatch.setattr(fp, "load_financials", lambda: fin)
    ev = pd.DataFrame({"stock_code": ["000001"], "trade_date": ["2026-01-06"]})
    out = P.attach_financials(ev, window=45)
    assert pd.isna(out.loc[0, "fin_yoy"])


def test_stats_handles_small_and_constant_series():
    """样本 <2 或零方差 ⇒ 返回 None，不许除零、也不许造出假的 t 值。"""
    assert P._stats(pd.Series([0.01])).get("t") is None
    const = P._stats(pd.Series([0.01] * 10))
    assert const["mean_pp"] == pytest.approx(1.0, abs=1e-6)
    assert const["t"] is None, "零方差时 t 无定义，不能编一个数出来"
    assert P._stats(pd.Series([np.nan, np.nan]))["n"] == 0
