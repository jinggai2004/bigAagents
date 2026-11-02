# -*- coding: utf-8 -*-
"""换手率层「快照派生日更」的守护（`data/turn_share.py::sync_from_snapshot`）。

## 为什么这几条是必要的

日更链路是 2026-11-01 补的，起因是"turn_share 层停在 10-28，第 4 个交易日
（11-03）起量价因子开始出 NaN"。这条链路有三个**静默失效**的方式：

1. **幂等坏了**：每次跑都重复追加 → parquet 长出重复行，`ffill`/`rolling` 全错，
   而且不会报错（只是因子值变了）；
2. **缺值写成 0**：换手率 0 会污染均值与分位（本项目铁律：缺就是缺，不返 0）；
3. **覆盖了更准的历史**：历史层是新浪全量回填、快照是当日截面，若用快照覆盖同一天，
   等于用精度低的源改写高精度的值 —— 而且看不出差别。

这三条都只能靠测试守住（跑一次看出不来的错）。
"""
import os

import pandas as pd
import pytest

from data import turn_share as TS


def _write(path, rows):
    pd.DataFrame(rows, columns=["date", "turnover_pct", "float_share"]).to_parquet(
        path, index=False)


def _read(code_dir, code):
    return pd.read_parquet(os.path.join(code_dir, f"{code}.parquet"))


@pytest.fixture
def panel(tmp_path, monkeypatch):
    """把换手率层指到临时目录（绝不写真库/真数据）。"""
    d = tmp_path / "turn_share"
    d.mkdir()
    monkeypatch.setattr(TS, "panel_dir", lambda: str(d))
    return d


@pytest.fixture
def fake_snapshot(monkeypatch):
    """伪造快照：返回 {日期: [行]}，由用例自己决定内容。"""
    def _install(by_date):
        import db.store as store
        monkeypatch.setattr(store, "market_snapshot_dates",
                            lambda limit=400: sorted(by_date)[-limit:])
        monkeypatch.setattr(store, "load_market_snapshot",
                            lambda date=None, codes=None: (date, by_date.get(date, [])))
    return _install


def _snap_row(code, price, turnover, float_cap):
    return {"stock_code": code, "price": price, "turnover": turnover,
            "float_cap": float_cap}


class TestSyncFromSnapshot:
    def test_appends_newer_date_with_correct_units(self, panel, fake_snapshot):
        """追加当日行，且 `float_share = float_cap / price`（单位是元/股，不是万元）。"""
        _write(panel / "600000.parquet",
               [("2026-10-28", 0.194343, 3.330584e10)])
        fake_snapshot({
            "2026-10-28": [_snap_row("600000", 9.99, 0.226370, 332725324617.0)],
            # ↑ 同一天：**不该**被覆盖（历史层更准）
            "2026-10-29": [_snap_row("600000", 9.99, 0.226370, 332725324617.0)],
        })

        res = TS.sync_from_snapshot(days=10)

        assert res["codes"] == 1 and res["rows"] == 1
        df = _read(panel, "600000")
        assert df["date"].tolist() == ["2026-10-28", "2026-10-29"]
        row = df.iloc[-1]
        assert float(row["turnover_pct"]) == pytest.approx(0.226370, abs=1e-6)
        assert float(row["float_share"]) == pytest.approx(332725324617.0 / 9.99, rel=1e-9)
        # 历史那天必须原样保留
        assert float(df.iloc[0]["turnover_pct"]) == pytest.approx(0.194343, abs=1e-6)

    def test_idempotent_second_run_appends_nothing(self, panel, fake_snapshot):
        """幂等：同一份快照跑两次，第二次必须是 0 行（否则 parquet 会长重复行）。"""
        _write(panel / "000001.parquet", [("2026-10-28", 0.869338, 1.940568e10)])
        fake_snapshot({"2026-10-29": [_snap_row("000001", 11.54, 1.02728, 223941604796.14)]})

        first = TS.sync_from_snapshot(days=10)
        second = TS.sync_from_snapshot(days=10)

        assert (first["codes"], first["rows"]) == (1, 1)
        assert (second["codes"], second["rows"]) == (0, 0)
        assert len(_read(panel, "000001")) == 2

    def test_missing_value_is_nan_not_zero(self, panel, fake_snapshot):
        """`turnover` 缺失时写 NaN —— 写 0 会污染均值/分位，而且完全不报错。"""
        _write(panel / "000002.parquet", [("2026-10-28", 1.869055, 9.715113154e9)])
        fake_snapshot({"2026-10-29": [_snap_row("000002", 8.5, None, 9.7e9)]})

        TS.sync_from_snapshot(days=10)

        row = _read(panel, "000002").iloc[-1]
        assert pd.isna(row["turnover_pct"])
        # float_share 仍应算出（它只依赖 price/float_cap）
        assert float(row["float_share"]) == pytest.approx(9.7e9 / 8.5, rel=1e-9)

    def test_new_code_without_history_is_skipped_not_created(self, panel, fake_snapshot):
        """快照里没有历史 parquet 的新股：跳过（建了只有几天数据，历史会全 NaN）。"""
        _write(panel / "600000.parquet", [("2026-10-28", 0.19, 3.33e10)])
        fake_snapshot({"2026-10-29": [
            _snap_row("600000", 9.99, 0.22, 3.33e11),
            _snap_row("301999", 20.0, 5.0, 8.0e9),      # 新股，本地无 parquet
        ]})

        res = TS.sync_from_snapshot(days=10)

        assert res["missing"] == 1
        assert not os.path.exists(panel / "301999.parquet")

    def test_does_not_rewrite_when_history_is_newer(self, panel, fake_snapshot):
        """历史层已到 10-30、快照只到 10-29 ⇒ 一行都不该动（不倒退、不重写）。"""
        _write(panel / "000001.parquet", [("2026-10-30", 0.960670, 1.940568e10)])
        fake_snapshot({"2026-10-29": [_snap_row("000001", 11.54, 1.02728, 2.2e11)]})

        res = TS.sync_from_snapshot(days=10)

        assert (res["codes"], res["rows"]) == (0, 0)
        assert _read(panel, "000001")["date"].tolist() == ["2026-10-30"]

    def test_zero_price_does_not_divide_by_zero(self, panel, fake_snapshot):
        """停牌/异常价 0：不许 ZeroDivisionError，float_share 落 NaN。"""
        _write(panel / "600000.parquet", [("2026-10-28", 0.19, 3.33e10)])
        fake_snapshot({"2026-10-29": [_snap_row("600000", 0, 0.22, 3.33e11)]})

        res = TS.sync_from_snapshot(days=10)

        assert res["errors"] == []
        row = _read(panel, "600000").iloc[-1]
        assert pd.isna(row["float_share"])
        assert float(row["turnover_pct"]) == pytest.approx(0.22, abs=1e-6)
