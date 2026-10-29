# -*- coding: utf-8 -*-
"""本地日线单一入口 `data/bars.py`（P0-1）的语义边界。

锁住的不是"能读文件"，而是几条**错了会让全项目数字失真的**契约：

  1. **schema 与 `sina_source.fetch_stock_daily` 对齐** —— 消费方改调用方时不改下游；
  2. **前复权靠 `pct_chg` 累乘重建** —— 面板只存未复权价，若直接返回未复权，
     除权日会出现假跳空（MA/动量/波动率全错）；
  3. **`as_of` 必须截断** —— 回测取价的前视防护唯一出口；
  4. **本地无数据 ⇒ None，不隐式联网** —— "缺就是缺"，且断网可跑；
  5. **代码写法归一**（`600028` / `sh.600028` / `600028.SH`）。
"""
import os

import numpy as np
import pandas as pd
import pytest

import data.bars as B

# 构造一段含"除权跳空"的行情：第 11 天 10 送 10（价格腰斩），pct_chg 仍 +0%
DAYS = 30
_BASE = 10.0


def _raw_for_code(code: str = "600028") -> pd.DataFrame:
    dates = pd.date_range("2026-01-05", periods=DAYS, freq="B")
    rows = []
    price = _BASE
    for i, d in enumerate(dates):
        if i == 11:                       # 除权日：价格腰斩但涨跌幅 0
            price = price / 2.0
            pct = 0.0
        else:
            pct = 1.0                      # 每天 +1%
            prev = price
            price = price * (1 + pct / 100)   # 不取整：本用例验证算法，不是精度
        pre = price if i == 11 else price / (1 + pct / 100)
        rows.append({
            "trade_date": d.strftime("%Y-%m-%d"), "open": price, "high": price * 1.01,
            "low": price * 0.99, "close": price, "preclose": pre,
            "volume": 1e6, "amount": 1e7, "pct_chg": pct, "turn": 1.5,
            "pe_ttm": 12.0, "pb_mrq": 1.2, "ps_ttm": 2.0,
            "trade_status": 1, "is_st": 0, "stock_code": code,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def bars_env(tmp_path, monkeypatch):
    """隔离的 by_code 目录（不碰真实面板）。

    ⚠️ 2026-10-26：**必须同时隔离 `OVERLAY_DIR`**。日增量覆盖层是"读时叠加"的，
       只 patch `BY_CODE_DIR` 的话，真实 `data/panel/daily_overlay/*.parquet`
       仍会叠到本夹具的合成行情上（实测：`600028` 的末根从 2026-02-13 变成
       2026-10-26，一批新鲜度/复权用例整片转红）。
    """
    import data.panel_store as _ps

    monkeypatch.setattr(B, "BY_CODE_DIR", str(tmp_path / "by_code"))
    monkeypatch.setattr(B, "MANIFEST_PATH", str(tmp_path / "by_code_manifest.json"))
    monkeypatch.setattr(B, "OVERLAY_DIR", str(tmp_path / "overlay"))
    # ⚠️ `TURN_SHARE_DIR` 是**导入时**算出来的常量（`PANEL_DIR` 拼出来的），
    #    只 patch `panel_store.PANEL_DIR` 挡不住它 —— 不隔离的话用例会读到真实的
    #    `data/panel/turn_share/*.parquet`（600028 真实存在，回填完就有值）。
    monkeypatch.setattr(B, "TURN_SHARE_DIR", str(tmp_path / "turn_share"))
    monkeypatch.setattr(_ps, "PANEL_DIR", str(tmp_path))
    os.makedirs(B.BY_CODE_DIR, exist_ok=True)
    B.clear_cache()
    yield tmp_path
    # ⚠️ `_load_raw` 是**进程级 lru_cache**：只在 setup 清是不够的 ——
    #    临时面板的 DataFrame 会活到后续用例里（实测：test_screener 读到
    #    本夹具的 2026-02-13 数据 ⇒ 被判"过期" ⇒ 用例莫名其妙红）。
    B.clear_cache()


def _write(code: str, df: pd.DataFrame) -> None:
    out = df.drop(columns=["stock_code"], errors="ignore").copy()
    out.to_parquet(os.path.join(B.BY_CODE_DIR, f"{code}.parquet"), index=False)


# ================================================================
# 1. 代码归一
# ================================================================

class TestCleanCode:
    @pytest.mark.parametrize("raw,expect", [
        ("600028", "600028"), ("sh.600028", "600028"), ("600028.SH", "600028"),
        ("sz.000001", "000001"), ("1", "000001"),
    ])
    def test_variants(self, raw, expect):
        assert B._clean(raw) == expect


# ================================================================
# 2. 读取与 schema
# ================================================================

class TestGetBars:
    def test_schema_matches_sina_fetch(self, bars_env):
        """⭐ 列名必须与 `fetch_stock_daily` 对齐（消费方零改动）。"""
        _write("600028", _raw_for_code())
        df = B.get_bars("600028")
        assert df is not None
        assert list(df.columns) == B._OUT_COLS
        assert df["date"].dtype.kind == "M", "date 必须是 datetime"
        assert df.attrs["source"] == "panel" and df.attrs["code"] == "600028"

    def test_range_filter_inclusive(self, bars_env):
        _write("600028", _raw_for_code())
        full = B.get_bars("600028")
        lo, hi = full["date"].iloc[5], full["date"].iloc[10]
        df = B.get_bars("600028", lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"))
        assert len(df) == 6 and df["date"].min() == lo and df["date"].max() == hi

    def test_as_of_truncates_future(self, bars_env):
        """⭐⭐ 前视防护：as_of 之后的行**绝不能**出现（回测取价唯一出口）。"""
        _write("600028", _raw_for_code())
        full = B.get_bars("600028")
        cut = full["date"].iloc[14]
        df = B.get_bars("600028", as_of=cut.strftime("%Y-%m-%d"))
        assert df["date"].max() == cut, "as_of 当天含、之后必须全部剔除"
        assert len(df) == 15

    def test_missing_code_returns_none_not_exception(self, bars_env):
        assert B.get_bars("999999") is None
        assert B.has_bars("999999") is False

    def test_min_rows_gate(self, bars_env):
        _write("600028", _raw_for_code())
        assert B.get_bars("600028", min_rows=100) is None
        assert B.get_bars("600028", min_rows=20) is not None

    def test_no_hidden_network(self, bars_env, monkeypatch):
        """⭐ 本地命中时**不许**碰任何联网入口（断网可跑的前提）。"""
        _write("600028", _raw_for_code())
        import data.sina_source as ss

        def _boom(*a, **k):
            raise AssertionError("get_bars 不该联网")

        monkeypatch.setattr(ss, "fetch_stock_daily", _boom)
        monkeypatch.setattr("data.stock_data.StockDataFetcher.get_stock_data", _boom)
        assert B.get_bars("600028") is not None


# ================================================================
# 3. 复权（核心口径）
# ================================================================

class TestQfq:
    def test_qfq_returns_reconstruct_daily_returns(self, bars_env):
        """⭐⭐ 前复权序列的日收益必须等于 `pct_chg` —— 这是"除权日无假跳空"的判据。

        夹具第 11 天价格腰斩（10 送 10）、pct_chg=0：
        - 若返回未复权 ⇒ 当日收益 −50%（假暴跌，MA/动量全错）；
        - 正确复权 ⇒ 当日收益 0%。
        """
        _write("600028", _raw_for_code())
        df = B.get_bars("600028", adjust="qfq")
        ret = df["close"].pct_change().dropna()
        raw = _raw_for_code()
        expect = raw["pct_chg"].to_numpy()[1:] / 100.0
        # 夹具保持精确浮点（不 round）⇒ 这里验证的是**算法**而非价格精度；
        # 真实数据只有 2~4 位小数，误差量级见 test_qfq_tolerant_to_price_rounding。
        np.testing.assert_allclose(ret.to_numpy(), expect, rtol=1e-6, atol=1e-9)

    def test_qfq_anchor_is_last_close(self, bars_env):
        """基准 = 最后一根收盘（与联网 qfq 同锚点，便于逐值对照）。"""
        _write("600028", _raw_for_code())
        qfq = B.get_bars("600028")
        raw = _raw_for_code()
        assert qfq["close"].iloc[-1] == pytest.approx(float(raw["close"].iloc[-1]), rel=1e-9)

    def test_adjust_none_returns_raw(self, bars_env):
        """`adjust="none"` ⇒ 原样未复权（用于"复权与否"对照实验）。"""
        _write("600028", _raw_for_code())
        df = B.get_bars("600028", adjust="none")
        raw = _raw_for_code()
        assert df["close"].iloc[11] == pytest.approx(float(raw["close"].iloc[11]), rel=1e-6)
        # 未复权在除权日的收益是 −50%（正是 qfq 要修正的那个假信号）
        assert df["close"].pct_change().iloc[11] == pytest.approx(-0.5, abs=1e-6)

    def test_hfq_anchors_first_bar_not_last(self, bars_env):
        """⭐⭐ D3（2026-10-18 审查）：`hfq` 必须锚**第一根**，不许静默返回 qfq。

        原实现把 qfq/hfq 一起走锚定末日的公式、`attrs` 却标 `hfq` —— 元数据说谎。
        判据：hfq[0] == 未复权 close[0]；且 hfq 与 qfq 在除权日之后**绝对值不同**。
        """
        _write("600028", _raw_for_code())
        raw = _raw_for_code()
        hfq = B.get_bars("600028", adjust="hfq")
        qfq = B.get_bars("600028", adjust="qfq")
        assert hfq["close"].iloc[0] == pytest.approx(float(raw["close"].iloc[0]), rel=1e-6), \
            "后复权首日必须等于未复权首日"
        assert hfq.attrs["adjust"] == "hfq"
        # 除权日（第 11 天）之后的绝对值：qfq 锚末日、hfq 锚首日 ⇒ 必然不同
        assert hfq["close"].iloc[-1] != pytest.approx(qfq["close"].iloc[-1], rel=1e-6)

    def test_hfq_returns_still_equal_pct_chg(self, bars_env):
        """后复权同样要满足"日收益 == pct_chg"（换锚点不改收益率）。"""
        _write("600028", _raw_for_code())
        hfq = B.get_bars("600028", adjust="hfq")
        ret = hfq["close"].pct_change().dropna().to_numpy()
        expect = _raw_for_code()["pct_chg"].to_numpy()[1:] / 100.0
        np.testing.assert_allclose(ret, expect, rtol=2e-4, atol=1e-9)

    def test_unknown_adjust_raises(self, bars_env):
        """D3：未知复权口径**不许静默当"不复权"**。"""
        _write("600028", _raw_for_code())
        with pytest.raises(ValueError, match="adjust"):
            B.get_bars("600028", adjust="adj")

    def test_missing_preclose_derived_from_pct_chg(self, bars_env):
        """preclose 缺失 ⇒ 用 `close/(1+pct_chg/100)` 反推（两条数据路径互备）。

        面板里 tushare 行的 preclose 来自 `pre_close`、BaoStock 行来自 `preclose`；
        任一侧缺失都要能重建复权，否则除权日会出现假跳空。
        """
        df = _raw_for_code()
        df["preclose"] = np.nan                    # 整列缺失
        _write("600028", df)
        got = B.get_bars("600028")
        ret = got["close"].pct_change().dropna().to_numpy()
        expect = _raw_for_code()["pct_chg"].to_numpy()[1:] / 100.0
        np.testing.assert_allclose(ret, expect, rtol=2e-4, atol=1e-9)


# ================================================================
# 4. 批量与覆盖
# ================================================================

class TestBatchAndCoverage:
    def test_get_bars_many_omits_missing(self, bars_env):
        _write("600028", _raw_for_code("600028"))
        _write("000001", _raw_for_code("000001"))
        got = B.get_bars_many(["600028", "000001", "600519"])
        assert set(got) == {"600028", "000001"}, "缺失代码不许出现在结果里（调用方须报告缺口）"

    def test_available_codes_and_coverage(self, bars_env):
        _write("600028", _raw_for_code("600028"))
        assert B.available_codes() == {"600028"}
        cov = B.coverage()
        assert cov["codes"] == 1 and cov["mb"] >= 0


# ================================================================
# 5. 新鲜度（D5，2026-10-18）
# ================================================================
#
# 为什么必须锁：面板里有 2.2% 的标的最后一根 K 线早于 2026-09-01（退市/长期停牌），
# `min_rows=20` 只看"够不够长"，不看"够不够新" ⇒ 停牌股几个月前的形态会被当作
# 当前信号参与排名。判据必须是**交易日**间隔，基准必须是**面板市场级最新日**
# （用"今天"当基准会让每个周末全体判滞后）。

class TestFreshness:
    def _cal(self, monkeypatch, days):
        monkeypatch.setattr(B, "_trade_days_cache", list(days))

    def test_trading_day_lag_counts_trading_days_only(self, monkeypatch):
        self._cal(monkeypatch, ["2026-10-12", "2026-10-13", "2026-10-14",
                                "2026-10-15", "2026-10-16"])
        # 10-14 → 10-16：中间只隔 2 个交易日（自然日是 2 天，巧合相等）
        assert B.trading_day_lag("2026-10-14", "2026-10-16") == 2
        # 跨周末：10-09(五) → 10-12(一) 是 1 个交易日，自然日是 3 天
        self._cal(monkeypatch, ["2026-10-09", "2026-10-12"])
        assert B.trading_day_lag("2026-10-09", "2026-10-12") == 1
        assert B.trading_day_lag("2026-10-12", "2026-10-12") == 0

    def test_lag_unknown_when_calendar_absent(self, monkeypatch):
        """日历缺失 ⇒ None（**不是 0**）：不知道就别假装"没问题"。"""
        self._cal(monkeypatch, [])
        assert B.trading_day_lag("2026-09-01", "2026-10-16") is None

    def test_attrs_carry_lag_and_flags(self, bars_env, monkeypatch):
        _write("600028", _raw_for_code())
        last = _raw_for_code()["trade_date"].iloc[-1]
        monkeypatch.setattr(B, "panel_date_max", lambda: last)
        self._cal(monkeypatch, list(_raw_for_code()["trade_date"]))
        df = B.get_bars("600028")
        assert df.attrs["last_date"] == last and df.attrs["lag_days"] == 0
        assert df.attrs["data_lagging"] is False

    def test_lagging_flag_and_reason(self, bars_env, monkeypatch):
        """落后超过默认阈值（10 个交易日）⇒ `data_lagging` 为真并给出原因文本。

        ⚠️ R6（2026-10-18 审查）：**不能用 `end=` 造"过期"** —— `end` 是调用方
        明确要求的区间末日，参考日就该是它（用它当基准会把"按你要的区间取到的数据"
        判成过期）。真正的过期场景是"面板覆盖已经往前走，这只票没跟上"：
        所以这里把 `panel_date_max` 调后，而不是把 `end` 调前。
        """
        _write("600028", _raw_for_code())
        days = list(_raw_for_code()["trade_date"])
        cal = days + ["2026-03-02", "2026-03-03"]
        self._cal(monkeypatch, cal)
        monkeypatch.setattr(B, "panel_date_max", lambda: cal[-1])
        df = B.get_bars("600028")
        assert df.attrs["ref_date"] == cal[-1]
        assert df.attrs["lag_days"] == 2
        assert df.attrs["data_lagging"] is False, "2 个交易日还没到 10 日阈值"

        monkeypatch.setattr(B, "panel_date_max", lambda: days[-1])
        df2 = B.get_bars("600028")
        assert df2.attrs["lag_days"] == 0

    def test_lagging_true_when_far_behind(self, bars_env, monkeypatch):
        """落后 > 阈值 ⇒ True + 原因文本（页面/结果里要能读到它）。"""
        _write("600028", _raw_for_code())
        days = list(_raw_for_code()["trade_date"])
        cal = days + [f"2026-04-{d:02d}" for d in range(1, 20)]
        self._cal(monkeypatch, cal)
        monkeypatch.setattr(B, "panel_date_max", lambda: cal[-1])
        df = B.get_bars("600028")
        assert df.attrs["lag_days"] == len(cal) - len(days)
        assert df.attrs["data_lagging"] is True
        assert "落后参考日" in df.attrs["staleness_reason"]

    def test_end_sets_the_reference_date(self, bars_env, monkeypatch):
        """⭐ R6：`end=` 是调用方要的末日 ⇒ 它就是参考日，不算过期。

        这条防的是"把按区间取到的历史数据判成过期"——回测/因子研究大量传 `end`。
        """
        _write("600028", _raw_for_code())
        days = list(_raw_for_code()["trade_date"])
        self._cal(monkeypatch, days)
        monkeypatch.setattr(B, "panel_date_max", lambda: days[-1])
        df = B.get_bars("600028", end=days[4])
        assert df.attrs["ref_date"] == days[4]
        assert df.attrs["lag_days"] == 0
        assert df.attrs["data_lagging"] is False

    def test_max_lag_returns_none(self, bars_env, monkeypatch):
        """`max_lag` 是**消费方可选**的硬闸门（默认不拦截，交给调用方决策）。"""
        _write("600028", _raw_for_code())
        days = list(_raw_for_code()["trade_date"])
        cal = days + [f"2026-05-{d:02d}" for d in range(1, 10)]
        self._cal(monkeypatch, cal)
        monkeypatch.setattr(B, "panel_date_max", lambda: cal[-1])
        assert B.get_bars("600028", max_lag=100) is not None
        assert B.get_bars("600028", max_lag=3) is None

    def test_data_lagging_follows_effective_threshold(self, bars_env, monkeypatch):
        """⭐ R6：`data_lagging` 必须按**生效阈值**（`max_lag`）判，不能写死 10。

        否则传 `max_lag=3` 时 lag=5 的票"被闸门挡了，但标记说它新鲜"——自相矛盾。
        """
        _write("600028", _raw_for_code())
        days = list(_raw_for_code()["trade_date"])
        cal = days + [f"2026-06-{d:02d}" for d in range(1, 10)]
        self._cal(monkeypatch, cal)
        monkeypatch.setattr(B, "panel_date_max", lambda: cal[-1])   # lag = 9
        df = B.get_bars("600028", max_lag=20)                       # 阈值 20 ⇒ 新鲜
        assert df.attrs["lag_threshold"] == 20 and df.attrs["data_lagging"] is False
        assert B.get_bars("600028", max_lag=3) is None               # 阈值 3 ⇒ 被拦

    def test_lag_unknown_when_ref_beyond_calendar(self, bars_env, monkeypatch):
        """⭐ R6：参考日超出日历覆盖 ⇒ **未知（None）**，不能算成 0（低估成最新）。"""
        _write("600028", _raw_for_code())
        days = list(_raw_for_code()["trade_date"])
        self._cal(monkeypatch, days)                     # 日历只到 days[-1]
        monkeypatch.setattr(B, "panel_date_max", lambda: "2026-12-31")
        df = B.get_bars("600028")
        assert df.attrs["lag_days"] is None
        assert df.attrs["data_lagging"] is None, "未知不能落到 False（那是'不知道却报平安'）"

    def test_ref_never_beyond_panel_max(self, bars_env, monkeypatch):
        """`as_of` 超过面板覆盖 ⇒ 参考日收敛到面板最新日（否则全市场都判滞后）。"""
        _write("600028", _raw_for_code())
        days = list(_raw_for_code()["trade_date"])
        self._cal(monkeypatch, days)
        monkeypatch.setattr(B, "panel_date_max", lambda: days[-1])
        df = B.get_bars("600028", as_of="2099-12-31")
        assert df.attrs["ref_date"] == days[-1]
        assert df.attrs["lag_days"] == 0

    def test_staleness_ignores_window(self, bars_env, monkeypatch):
        """⭐ R1：`staleness()` 必须按**全历史末根**判，不受 90 天窗口影响。

        长期停牌/退市股在近 90 天里 0 行 ⇒ 窗口内取数返回 None，
        但"它已经过期"这件事仍然必须能被判定出来（否则会掉进联网兜底）。
        """
        _write("600028", _raw_for_code())
        days = list(_raw_for_code()["trade_date"])
        cal = days + [f"2026-07-{d:02d}" for d in range(1, 25)]
        self._cal(monkeypatch, cal)
        monkeypatch.setattr(B, "panel_date_max", lambda: cal[-1])
        assert B.get_bars("600028", start="2026-06-01", end=cal[-1]) is None
        st = B.staleness("600028")
        assert st["last_date"] == days[-1]
        assert st["data_lagging"] is True and st["lag_days"] > 10

    def test_staleness_unknown_for_missing_code(self, bars_env):
        st = B.staleness("999999")
        assert st["last_date"] is None and st["data_lagging"] is None


# ================================================================
# 换手率 / 流通股本回填层（2026-10-29）
# ================================================================

class TestTurnShareLayer:
    """`get_bars` 叠加 `turn_share/{code}.parquet` 的语义边界。

    这层的意义：面板 `turn` 只覆盖 1,997 只（**科创板 0、北交所 0**），
    `outstanding_share` 根本没有。回填后消费方（因子/研究脚本）才能拿到
    换手率与流通市值 —— 而**这一切只通过 `get_bars` 生效**。
    """

    def _write_ts(self, code, rows, tmp_path):
        d = tmp_path / "turn_share"
        os.makedirs(d, exist_ok=True)
        pd.DataFrame(rows).to_parquet(os.path.join(str(d), f"{code}.parquet"),
                                      index=False)
        B.clear_cache()

    def test_fills_missing_turnover_and_float_share(self, bars_env):
        _write("600028", _raw_for_code("600028"))
        self._write_ts("600028", [
            {"date": "2026-01-05", "turnover_pct": 0.31, "float_share": 1.0e10},
            {"date": "2026-01-06", "turnover_pct": 0.42, "float_share": 1.0e10},
        ], bars_env)
        df = B.get_bars("600028", "2026-01-05", "2026-01-06")
        # 面板里 turn=1.5（合成值）⇒ **不许被回填层覆盖**
        assert df["turnover_pct"].tolist() == [1.5, 1.5]
        # outstanding_share 面板恒 NaN ⇒ 被回填层填上
        assert df["outstanding_share"].notna().all()
        assert float(df["outstanding_share"].iloc[0]) == pytest.approx(1.0e10)

    def test_fills_when_panel_turn_is_nan(self, bars_env):
        raw = _raw_for_code("600028")
        raw["turn"] = np.nan                     # 模拟科创板/北交所那种"turn 全空"
        _write("600028", raw)
        self._write_ts("600028", [
            {"date": "2026-01-05", "turnover_pct": 2.5, "float_share": 3.0e9},
        ], bars_env)
        df = B.get_bars("600028", "2026-01-05", "2026-01-05")
        assert float(df["turnover_pct"].iloc[0]) == pytest.approx(2.5)
        assert float(df["outstanding_share"].iloc[0]) == pytest.approx(3.0e9)

    def test_absent_layer_leaves_nan(self, bars_env):
        """没有回填文件 ⇒ 保持 NaN（缺就是缺，绝不用 0 填充 ——
        0 会被下游读成"换手率极低"，与"没有数据"方向完全相反）。"""
        _write("600028", _raw_for_code("600028"))
        df = B.get_bars("600028", "2026-01-05", "2026-01-05")
        assert pd.isna(df["outstanding_share"].iloc[0])
        assert float(df["turnover_pct"].iloc[0]) == pytest.approx(1.5)

    def test_date_mismatch_does_not_shift_values(self, bars_env):
        """回填层与面板日期不对齐 ⇒ 只填空位，**不许错行**。

        `map` 是按日期键的；若实现写成按位置赋值，这里会得到 0.31 之后的错位值。
        """
        _write("600028", _raw_for_code("600028"))
        self._write_ts("600028", [
            {"date": "2025-12-31", "turnover_pct": 9.9, "float_share": 1.0e9},
        ], bars_env)
        df = B.get_bars("600028", "2026-01-06", "2026-01-06")
        assert pd.isna(df["outstanding_share"].iloc[0]), "无对应日期的行不该被填"
