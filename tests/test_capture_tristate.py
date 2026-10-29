# -*- coding: utf-8 -*-
"""采集任务的**三态报告**与"失败原因可查"（2026-10-17 修）。

## 事故

实测 `2026-10-16 15:40` 日志：

```
[资金流快照] 2026-10-16 采集完成: 个股 0 只 / 行业 0 个        ← INFO，像成功
```

而真实原因是 `data/fund_flow.py` 里一句 `print(f"   [同花顺] {window} 快照获取失败: {e}")`
—— **长驻调度进程里 `print` 进不了 `logs/scheduler_*.log`** ⇒ 全天失败在日志里完全隐形。

## 三条必须钉住的语义

1. **0 行不得记"完成"**：必须按 `status` 分三态（ok / partial / failed），非 ok 记 **WARNING**；
2. **失败原因随结果走**：`failures` 要带每个窗口的原因（"失败原因必须可查"，不能只说"0 只"）；
3. **成功路径不得被降级**：全部采到时仍是 INFO"完成"（新规则不许自己引入误报）。
"""
import pandas as pd
import pytest

from core import scheduler
from data import fund_flow as F
from db import store
from tests.conftest import use_template_db

DAY = "2026-10-16"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    use_template_db(tmp_path, monkeypatch, "capture_tristate.db")
    # 落库换成计数器（本用例只验"三态报告"，不验表结构）
    monkeypatch.setattr(store, "save_fund_flow_snapshot",
                        lambda d, rows, source=None: len(rows))
    monkeypatch.setattr(store, "save_industry_flow_snapshot",
                        lambda d, rows, source=None: len(rows))
    F._ths_last_error.clear()
    yield
    F._ths_last_error.clear()


def _stub_windows(monkeypatch, ok_windows: set):
    """让指定窗口成功、其余失败（并在 `_ths_last_error` 里留下原因）。"""
    def _fake(window):
        if window in ok_windows:
            return pd.DataFrame({"股票代码": ["600028"], "净额": [1.0],
                                 "股票简称": ["中国石化"], "成交额": [1.0],
                                 "涨跌幅": ["1.0%"], "最新价": ["5.53"]})
        F._ths_last_error[window] = "模拟：上游返回空表"
        return None
    monkeypatch.setattr(F, "get_ths_snapshot", _fake)


def _industry(monkeypatch, ok: bool):
    def _fake():
        if not ok:
            return None
        return pd.DataFrame({"行业": ["银行"], "净额": [1.0], "涨跌幅": [0.5],
                             "领涨股": ["x"]})
    monkeypatch.setattr(F, "get_industry_flow_snapshot", _fake)


class TestTriStateReport:
    def test_all_failed_is_failed_with_reasons(self, monkeypatch, caplog):
        """⭐ 全失败 ⇒ status='failed' + 每个窗口都有原因 + **WARNING**（不是"完成"）。"""
        _stub_windows(monkeypatch, ok_windows=set())
        _industry(monkeypatch, ok=False)
        with caplog.at_level("WARNING"):
            r = F.capture_market_flow_snapshot(DAY)
        assert r["status"] == "failed", r
        assert r["stocks"] == 0 and r["industries"] == 0
        assert set(r["windows"]) == set(F._THS_WINDOWS)
        assert all(n == 0 for n in r["windows"].values())
        assert len(r["failures"]) == len(F._THS_WINDOWS)
        assert all("上游返回空表" in v for v in r["failures"].values()), r["failures"]
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "失败" in joined and "窗口明细" in joined, joined

    def test_partial_is_partial_not_ok(self, monkeypatch):
        """⭐ 只采到一个窗口 ⇒ 'partial'（不许当成成功，也不许当成全败）。"""
        _stub_windows(monkeypatch, ok_windows={"即时"})
        _industry(monkeypatch, ok=True)
        r = F.capture_market_flow_snapshot(DAY)
        assert r["status"] == "partial", r
        assert r["stocks"] == 1 and r["industries"] == 1
        assert set(r["failures"]) == set(F._THS_WINDOWS) - {"即时"}

    def test_all_ok_stays_ok_and_info(self, monkeypatch, caplog):
        """⚠️ 鉴别力：全部采到时必须仍是 ok（**不许**让新规则把正常路径也报成异常）。"""
        _stub_windows(monkeypatch, ok_windows=set(F._THS_WINDOWS))
        _industry(monkeypatch, ok=True)
        with caplog.at_level("INFO"):
            r = F.capture_market_flow_snapshot(DAY)
        assert r["status"] == "ok", r
        assert r["failures"] == {}
        assert not [x for x in caplog.records if x.levelname == "WARNING"], \
            "正常采集不该产生 WARNING"


class TestSchedulerLogsTriState:
    def test_failed_capture_logged_as_warning(self, monkeypatch, caplog):
        """⭐ 调度侧也必须按三态记 —— 否则"0 只"仍会被读成完成。"""
        monkeypatch.setattr(F, "capture_market_flow_snapshot",
                            lambda *a, **k: {"snapshot_date": DAY, "stocks": 0,
                                             "industries": 0,
                                             "windows": {"即时": 0}, "status": "failed",
                                             "failures": {"即时": "模拟失败"}})
        # 估值截面是另一个上游：本用例只需它别联网
        monkeypatch.setattr(scheduler, "save_daily_market_snapshot",
                            lambda *a, **k: {"ok": False, "reason": "用例跳过"})
        with caplog.at_level("WARNING"):
            scheduler.run_snapshot_job(job_desc="测试采集")
        msgs = "\n".join(r.getMessage() for r in caplog.records)
        assert "失败" in msgs and "模拟失败" in msgs, msgs
