# -*- coding: utf-8 -*-
"""面板日增量（`data.panel_daily` + `bars` 覆盖层）：写一个文件就能让 K 线"到今天"。

存在理由（一句话）：`data/panel/by_code` 是 2026-10-18 一次性建的，之后**无人维护**，
实测 10-26 时 `date_max` 还停在 10-16 ⇒ 交易台持仓估值、MA60、因子截面全部滞后 7 个
交易日。修法不是"每晚重写 5,830 个 parquet"（167s + 读者可能读到半个文件），
而是每天写一个小覆盖层、读时叠加。

本文件守三件事：
1. **叠加正确**：`get_bars()` 能读到覆盖层里比 by_code 更新的交易日，且同一天不重复；
2. **幂等**：同一天重跑覆盖（不是追加）—— 否则行数会随重跑次数增长；
3. **`panel_date_max()` 认覆盖层**：不改它的话新鲜度闸门天天误报"滞后"。
"""
import os

import pandas as pd
import pytest

from data import bars, panel_daily, panel_store


@pytest.fixture
def panel_dirs(tmp_path, monkeypatch):
    """把 by_code 与覆盖层都指向 tmp（**绝不碰真实 5,830 个文件**）"""
    by_code = tmp_path / "by_code"
    overlay = tmp_path / "daily_overlay"
    by_code.mkdir()
    overlay.mkdir()
    monkeypatch.setattr(bars, "BY_CODE_DIR", str(by_code))
    monkeypatch.setattr(bars, "OVERLAY_DIR", str(overlay))
    monkeypatch.setattr(bars, "MANIFEST_PATH", str(tmp_path / "manifest.json"))
    monkeypatch.setattr(panel_store, "PANEL_DIR", str(tmp_path))
    bars.clear_cache()
    yield {"by_code": by_code, "overlay": overlay, "root": tmp_path}
    bars.clear_cache()


def _write_base(dirs, code="600519", days=("2026-10-15", "2026-10-16"), close=100.0):
    df = pd.DataFrame([{"trade_date": d, "open": close, "high": close, "low": close,
                        "close": close, "preclose": close, "volume": 1e6,
                        "amount": 1e9, "pct_chg": 0.5, "turn": float("nan"),
                        "pe_ttm": float("nan"), "pb_mrq": float("nan"),
                        "ps_ttm": float("nan"), "trade_status": None, "is_st": None}
                       for d in days])
    for c in ("open", "high", "low", "close", "preclose", "volume", "amount",
              "pct_chg", "turn", "pe_ttm", "pb_mrq", "ps_ttm"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df.to_parquet(os.path.join(str(dirs["by_code"]), f"{code}.parquet"), index=False)


def _frames(code="600519", dates=("2026-10-19",), close=110.0):
    rows = []
    for d in dates:
        rows.append({"stock_code": code, "trade_date": d, "open": close,
                     "high": close, "low": close, "close": close, "preclose": close,
                     "volume": 2e6, "amount": 2e9, "pct_chg": 1.1, "turn": None,
                     "pe_ttm": None, "pb_mrq": None, "ps_ttm": None,
                     "trade_status": None, "is_st": None})
    return pd.DataFrame(rows)


def test_overlay_extends_bars(panel_dirs):
    """by_code 到 10-16，覆盖层到 10-19 ⇒ get_bars 必须能看到 10-19"""
    _write_base(panel_dirs)
    r = panel_daily.append_trade_day("2026-10-19", frames=_frames())
    assert r["ok"] and r["codes"] == 1

    df = bars.get_bars("600519", adjust="none")
    assert df is not None
    last = str(df["date"].iloc[-1])[:10]
    assert last == "2026-10-19", f"覆盖层没叠加进来（last={last}）"
    assert len(df) == 3
    assert float(df["close"].iloc[-1]) == pytest.approx(110.0)


def test_overlay_is_idempotent(panel_dirs):
    """同一天重跑 ⇒ 覆盖，不追加（否则行数随重跑次数增长）"""
    _write_base(panel_dirs)
    panel_daily.append_trade_day("2026-10-19", frames=_frames())
    panel_daily.append_trade_day("2026-10-19", frames=_frames(close=111.0))
    df = bars.get_bars("600519", adjust="none")
    assert len(df) == 3, "同一天被写了两行"
    assert float(df["close"].iloc[-1]) == pytest.approx(111.0), "新值没覆盖旧值"


def test_panel_date_max_follows_overlay(panel_dirs):
    """`panel_date_max()` 必须认覆盖层（新鲜度闸门的基准日）"""
    _write_base(panel_dirs)
    with open(str(panel_dirs["root"] / "overview.json"), "w", encoding="utf-8") as f:
        f.write('{"date_max": "2026-10-16"}')
    bars.clear_cache()
    assert bars.panel_date_max() == "2026-10-16"
    panel_daily.append_trade_day("2026-10-23", frames=_frames(dates=("2026-10-23",)))
    assert bars.panel_date_max() == "2026-10-23", "覆盖层日期没被 join 进 panel_date_max"


def test_empty_frames_reported_not_written(panel_dirs):
    """取回车但里面没有目标日 ⇒ `ok:False` + 原因（不写空文件，不静默成功）"""
    r = panel_daily.append_trade_day("2026-10-26", frames=_frames(dates=("2026-10-19",)))
    assert r["ok"] is False and "2026-10-26" in r["reason"]
    assert panel_daily.overlay_dates() == []


def test_new_code_not_in_by_code_still_readable(panel_dirs):
    """新股（by_code 里没有）也要能从覆盖层读到 —— 否则新上市票永远选不到"""
    r = panel_daily.append_trade_day("2026-10-19", frames=_frames(code="301999"))
    assert r["ok"]
    df = bars.get_bars("301999", adjust="none")
    assert df is not None and len(df) == 1
