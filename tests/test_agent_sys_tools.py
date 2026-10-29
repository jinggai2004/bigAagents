# -*- coding: utf-8 -*-
"""系统级工具测试（2026-10-24）：读报告 / 查库 / 看配置 / 推送模板 / 定时任务 / 面板K线 / 数据新鲜度。

原则沿用 `tests/test_tool_contract.py`：**只 mock 叶子依赖，不 mock 被测工具自身**。
- DB 用例一律走 `use_template_db`（绝不碰真实 data/portfolio.db）；
- 文件系统用例 monkeypatch 模块常量 `OUTPUT_ROOT` / `CONFIG_ROOT` 到 tmp（不耦合真实 output/）；
- 安全边界（路径穿越 / .env / 未知表）与"短内容不被截断"都要有断言——
  这类工具一旦放宽，就等于把项目内部暴露给模型，必须有回归测试钉住。
"""

import json
import os

import pandas as pd
import pytest

from core import agent_tools_sys as SYS
from core.agent_tools import TOOL_FUNCS, TOOL_SCHEMAS


# ================================================================
# 夹具
# ================================================================

@pytest.fixture()
def fake_output(tmp_path, monkeypatch):
    """临时 output 目录：一层报告 + 一层 daily_picks JSON。"""
    root = tmp_path / "output"
    (root / "daily_picks").mkdir(parents=True)
    (root / "复盘_20261023.md").write_text(
        "".join(f"第{i}行：复盘内容\n" for i in range(1, 301)), encoding="utf-8")
    (root / "daily_picks" / "latest.json").write_text(json.dumps(
        {"date": "2026-10-23", "picks": [{"股票代码": "600519", "股票名称": "贵州茅台"}]},
        ensure_ascii=False), encoding="utf-8")
    (root / "big.json").write_text(json.dumps(
        {"picks": [{"代码": "600519", "备注": "x" * 8000}]}, ensure_ascii=False),
        encoding="utf-8")
    monkeypatch.setattr(SYS, "OUTPUT_ROOT", str(root))
    return root


@pytest.fixture()
def fake_config(tmp_path, monkeypatch):
    """临时 config 目录：含假密钥的 provider 配置。"""
    root = tmp_path / "config"
    root.mkdir()
    (root / "llm_providers.json").write_text(json.dumps({
        "default_provider": "deepseek",
        "providers": [{"id": "deepseek", "api_key": "sk-1234567890",
                       "api_key_env": "DEEPSEEK_API_KEY", "models": ["deepseek-v4-flash"]}],
    }, ensure_ascii=False), encoding="utf-8")
    (root / "settings.py").write_text("SECRET = 'x'", encoding="utf-8")
    monkeypatch.setattr(SYS, "CONFIG_ROOT", str(root))
    return root


@pytest.fixture()
def tpl_db(tmp_path, monkeypatch):
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "sys_tools.db")
    from db import store
    return store


# ================================================================
# 1. tool_read_output_file
# ================================================================

def test_output_list_filters_by_pattern(fake_output):
    r = TOOL_FUNCS["tool_read_output_file"](action="list")
    assert r["文件数"] == 3
    r2 = TOOL_FUNCS["tool_read_output_file"](action="list", pattern="复盘")
    assert r2["文件数"] == 1 and r2["文件"][0]["文件"] == "复盘_20261023.md"
    r3 = TOOL_FUNCS["tool_read_output_file"](action="list", pattern="daily_picks")
    assert any("latest.json" in f["文件"] for f in r3["文件"])


def test_output_read_offset_and_tail(fake_output):
    r = TOOL_FUNCS["tool_read_output_file"](action="read", file="复盘_20261023.md",
                                            offset=10, limit=5)
    assert "第 11 ~ 15 行" in r["行范围"]
    assert r["内容"].splitlines()[0].startswith("第11行")
    t = TOOL_FUNCS["tool_read_output_file"](action="tail", file="复盘_20261023.md", limit=3)
    assert "第300行" in t["内容"] and "298" in t["行范围"]
    assert r["修改时间"], "必须给文件修改时间（历史产物提醒靠它）"


def test_output_read_json_and_json_path(fake_output):
    r = TOOL_FUNCS["tool_read_output_file"](action="read", file="daily_picks/latest.json")
    assert r["类型"] == "json" and r["数据"]["date"] == "2026-10-23"
    sub = TOOL_FUNCS["tool_read_output_file"](action="read", file="daily_picks/latest.json",
                                              json_path="picks.0.股票代码")
    assert sub["数据"] == "600519"
    bad = TOOL_FUNCS["tool_read_output_file"](action="read", file="daily_picks/latest.json",
                                              json_path="picks.99")
    assert "error" in bad and "顶层键" in bad


def test_output_big_json_suggests_json_path(fake_output):
    r = TOOL_FUNCS["tool_read_output_file"](action="read", file="big.json")
    assert "error" in r and "json_path" in r["提示"]
    sub = TOOL_FUNCS["tool_read_output_file"](action="read", file="big.json",
                                              json_path="picks.0.代码")
    assert sub.get("数据") == "600519"


def test_output_rejects_path_escape(fake_output):
    """路径穿越必须被 realpath 挡住（三种写法）。"""
    for bad in ("../config/settings.py", "a/../../core/chat_agent.py",
                "..\\config\\settings.py", "/etc/passwd"):
        r = TOOL_FUNCS["tool_read_output_file"](action="read", file=bad)
        assert "error" in r, f"未拦截: {bad}"


# ================================================================
# 2. tool_db_inspect
# ================================================================

def _seed_alerts(store, rows):
    conn = store.get_connection()
    try:
        for r in rows:
            conn.execute(
                "INSERT INTO alerts (trade_date, code, name, rule, level, message, price,"
                " change_pct, quote_time, source) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (r["date"], r["code"], r.get("name", ""), "测试", "info", "消息",
                 10.0, 1.0, "14:30:00", "test"))
        conn.commit()
    finally:
        conn.close()


def test_db_inspect_lists_tables(tpl_db):
    r = TOOL_FUNCS["tool_db_inspect"](action="tables")
    assert "alerts" in r["可查表"] and "llm_calls" in r["可查表"]
    assert r["可查表"]["alerts"]["行数"] == 0
    assert "code" in r["可查表"]["alerts"]["可过滤"]


def test_db_inspect_query_code_and_date_filters(tpl_db):
    _seed_alerts(tpl_db, [
        {"date": "2026-10-23", "code": "600519", "name": "贵州茅台"},
        {"date": "2026-10-23", "code": "000001", "name": "平安银行"},
        {"date": "2026-10-22", "code": "600519", "name": "贵州茅台"},
    ])
    r = TOOL_FUNCS["tool_db_inspect"](action="query", table="alerts", limit=10)
    assert r["条数"] == 3
    r2 = TOOL_FUNCS["tool_db_inspect"](action="query", table="alerts", code="600519")
    assert r2["条数"] == 2 and r2["过滤条件"] == "code=600519"
    # 8 位日期写法必须被归一化后才能命中
    r3 = TOOL_FUNCS["tool_db_inspect"](action="query", table="alerts", date="20261022")
    assert r3["条数"] == 1 and r3["记录"][0]["code"] == "600519"
    assert r3["记录"][0]["trade_date"] == "2026-10-22"


def test_db_inspect_unknown_table_and_ignored_filter(tpl_db):
    bad = TOOL_FUNCS["tool_db_inspect"](action="query", table="users")
    assert "error" in bad and "alerts" in bad["可用表"]
    r = TOOL_FUNCS["tool_db_inspect"](action="query", table="runs", code="600519")
    assert "提示" in r and "没有对应的过滤列" in r["提示"]


def test_db_inspect_llm_failures_only(tpl_db):
    conn = tpl_db.get_connection()
    try:
        for ok in (1, 0, 0):
            conn.execute(
                "INSERT INTO llm_calls (created_at, model, purpose, ok, latency_ms,"
                " total_tokens, cost_cny, error) VALUES (?,?,?,?,?,?,?,?)",
                ("2026-10-24 10:00:00", "deepseek-v4-flash", "chat.tool_loop",
                 ok, 1000, 500, 0.001, "" if ok else "超时"))
        conn.commit()
    finally:
        conn.close()
    r = TOOL_FUNCS["tool_db_inspect"](action="query", table="llm_calls", only_failures=True)
    assert r["条数"] == 2 and all(x["ok"] == 0 for x in r["记录"])
    assert "仅失败" in r["过滤条件"]


# ================================================================
# 3. tool_config_view
# ================================================================

def test_config_list_and_masking(fake_config):
    r = TOOL_FUNCS["tool_config_view"](action="list")
    assert "llm_providers.json" in [f["文件"] for f in r["配置文件"]]
    assert "settings.py" not in [f["文件"] for f in r["配置文件"]], "只开放 JSON，源码不列"

    got = TOOL_FUNCS["tool_config_view"](action="read", file="llm_providers.json")
    prov = got["配置"]["providers"][0]
    assert prov["api_key"] == "***已脱敏***", "真实密钥必须掩码"
    assert prov["api_key_env"] == "DEEPSEEK_API_KEY", "环境变量名要保留（它不是密钥）"
    assert prov["models"] == ["deepseek-v4-flash"]


def test_config_masks_secret_subtrees(fake_config):
    """质量复核 P1-1：列表/嵌套容器里的密钥也必须掩码，不能递归到无敏感键名后原样返回。"""
    (fake_config / "nested.json").write_text(json.dumps({
        "api_keys": ["sk-list-secret"],
        "api_key": {"value": "sk-nested-secret", "note": "x"},
        "normal": {"plain_list": ["a", "b"]},
    }, ensure_ascii=False), encoding="utf-8")
    got = TOOL_FUNCS["tool_config_view"](action="read", file="nested.json")
    cfg = got["配置"]
    # 整个子树掩成一个标量（不留结构线索），list/dict 都不再递归原样返回
    assert cfg["api_keys"] == "***已脱敏***", "敏感键的子树整体掩码"
    assert cfg["api_key"] == "***已脱敏***", "敏感键的嵌套值整体掩码"
    assert cfg["normal"] == {"plain_list": ["a", "b"]}, "普通键不受影响"


def test_config_rejects_env_and_py_and_escape(fake_config):
    env = TOOL_FUNCS["tool_config_view"](action="read", file=".env")
    assert "error" in env and "安全边界" in env["error"]
    # 子目录伪装的 .env 也要按 basename 拦住（质量复核 P1-1）
    sub = TOOL_FUNCS["tool_config_view"](action="read", file="nested/.env")
    assert "error" in sub
    py = TOOL_FUNCS["tool_config_view"](action="read", file="settings.py")
    assert "error" in py
    esc = TOOL_FUNCS["tool_config_view"](action="read", file="../core/chat_agent.py")
    assert "error" in esc
    # 非 JSON 扩展名拒绝（nested/secret.txt 曾可读）
    txt = TOOL_FUNCS["tool_config_view"](action="read", file="nested/secret.txt")
    assert "error" in txt


# ================================================================
# 4. tool_manage_watchlist · update
# ================================================================

def test_watchlist_update_and_clear_note(tmp_path, monkeypatch, tpl_db):
    """自选股 name/note 的更新与清空语义。

    ⚠️ 2026-10-27：自选股落 SQLite 后，这个用例**必须**带 `tpl_db` 并先打迁移标记。
    缺了它们，`add_stock` 会触发首次迁移 —— 迁移会把默认池写进**当时连接的那个库**，
    而那个库就是生产 `data/portfolio.db`（实测发生了：`app_flags` 被写成
    `default:15`、`watchlist_items` 多出 16 行测试数据）。
    """
    from data import watchlist as wl
    monkeypatch.setattr(wl, "WATCHLIST_PATH", str(tmp_path / "watchlist.json"))
    tpl_db.set_flag(wl._MIGRATED_FLAG, "test")        # 阻止首次迁移
    # ⚠️ 601127 不在 DEFAULT_WATCHLIST 里（600519 在，add 会返回"已在自选中"）
    assert wl.add_stock("601127", name="赛力斯")[0]
    ok = wl.update_stock("601127", note="核心持仓")
    assert ok[0] and "备注" in ok[1]
    item = next(s for s in wl.load_watchlist() if s["code"] == "601127")
    assert item["note"] == "核心持仓"
    # 清空备注（note="" 且非 None）
    assert wl.update_stock("601127", note="")[0]
    assert next(s for s in wl.load_watchlist() if s["code"] == "601127")["note"] == ""
    assert not wl.update_stock("000000", note="x")[0]
    assert not wl.update_stock("601127")[0], "一个字段都不传必须报错，不能假装成功"


def test_watchlist_tool_update_path(tmp_path, monkeypatch, tpl_db):
    """update 走两阶段守护（preview → commit）；commit 会写审计，必须挂临时库。"""
    from data import watchlist as wl
    monkeypatch.setattr(wl, "WATCHLIST_PATH", str(tmp_path / "watchlist.json"))
    p = TOOL_FUNCS["tool_manage_watchlist"](action="add", code="601127", name="赛力斯")
    assert p.get("待确认") is True, "add 只产出 preview"
    ok = TOOL_FUNCS["tool_manage_watchlist"](action="commit", change_id=p["change_id"])
    assert ok.get("success") is True
    assert any(s["code"] == "601127" for s in wl.load_watchlist())
    p2 = TOOL_FUNCS["tool_manage_watchlist"](action="update", code="601127", note="整车底仓")
    assert p2.get("待确认") is True
    r2 = TOOL_FUNCS["tool_manage_watchlist"](action="commit", change_id=p2["change_id"])
    assert r2.get("success") is True
    assert any(s["note"] == "整车底仓" for s in wl.load_watchlist())
    bad = TOOL_FUNCS["tool_manage_watchlist"](action="update", code="")
    assert "error" in bad


# ================================================================
# 5. tool_push_templates
# ================================================================

def test_push_templates_list_and_get(tpl_db):
    r = TOOL_FUNCS["tool_push_templates"](action="list")
    keys = [x["key"] for x in r["模板"]]
    assert "alert" in keys and "review_summary" in keys
    got = TOOL_FUNCS["tool_push_templates"](action="get", key="alert")
    assert got["是否自定义"] is False
    assert "title" in got["可用变量"]
    assert got["当前生效正文"] == got["内置默认正文"]
    assert "error" in TOOL_FUNCS["tool_push_templates"](action="get", key="nope")


def test_push_templates_save_validates_and_persists(tpl_db):
    """校验失败被拒；合法正文走 preview → commit（两阶段守护）。"""
    bad = TOOL_FUNCS["tool_push_templates"](
        action="save", key="alert", body="# {{title}}\n{{typo_var}}")
    assert "error" in bad and "typo_var" in bad["error"]

    # {% if %} 的变量也要查（质量复核 P1-2）：拼错的条件变量渲染时整块静默消失
    bad_if = TOOL_FUNCS["tool_push_templates"](
        action="save", key="alert", body="{% if typo_cond %}{{title}}{% endif %}")
    assert "error" in bad_if and "typo_cond" in bad_if["error"]

    # 不支持的控制符（else）拒绝
    bad_else = TOOL_FUNCS["tool_push_templates"](
        action="save", key="alert", body="{{title}}{% else %}")
    assert "error" in bad_else

    unbalanced = TOOL_FUNCS["tool_push_templates"](
        action="save", key="alert", body="# {{title}}{% if count %}x")
    assert "error" in unbalanced and "不配对" in unbalanced["error"]

    empty = TOOL_FUNCS["tool_push_templates"](action="save", key="alert", body="  ")
    assert "error" in empty and "reset" in empty["error"]

    body = "# {{title}} · 共 {{count}} 条\n\n{{items}}\n\n{{footer}}"
    p = TOOL_FUNCS["tool_push_templates"](action="save", key="alert", body=body)
    assert p.get("待确认") is True and p["change_id"]
    assert "alert" not in tpl_db.list_push_templates(), "preview 不得写库"
    ok = TOOL_FUNCS["tool_push_templates"](action="commit", change_id=p["change_id"])
    assert ok.get("success") is True
    assert tpl_db.list_push_templates()["alert"].startswith("# {{title}}")
    got = TOOL_FUNCS["tool_push_templates"](action="get", key="alert")
    assert got["是否自定义"] is True

    pr = TOOL_FUNCS["tool_push_templates"](action="reset", key="alert")
    assert pr.get("待确认") is True
    rr = TOOL_FUNCS["tool_push_templates"](action="commit", change_id=pr["change_id"])
    assert rr.get("success") is True
    assert "alert" not in tpl_db.list_push_templates()


# ================================================================
# 6. tool_scheduled_jobs
# ================================================================

def test_scheduled_jobs_list_has_builtin_and_custom(tpl_db):
    tpl_db.upsert_scheduled_job({"job_id": "custom_test", "name": "测试任务",
                                 "mode": "screen",
                                 "cron": {"hour": 9, "minute": 5}})
    r = TOOL_FUNCS["tool_scheduled_jobs"](action="list")
    assert any(j.get("job_id") == "morning_screen" for j in r["内置任务"])
    assert [j["job_id"] for j in r["自定义任务"]] == ["custom_test"]
    assert "60秒" in r["生效说明"]


def test_scheduled_jobs_cannot_toggle_builtin(tpl_db):
    r = TOOL_FUNCS["tool_scheduled_jobs"](action="toggle", job_id="daily_review", enabled=False)
    assert "error" in r and "内置任务" in r["error"]


def test_scheduled_jobs_toggle_get_delete_custom(tpl_db):
    """toggle/delete 走两阶段守护；生命周期与 restore 详见 test_agent_write_guard.py。"""
    tpl_db.upsert_scheduled_job({"job_id": "custom_x", "name": "X", "mode": "screen",
                                 "cron": {"hour": 10, "minute": 0}})
    off = TOOL_FUNCS["tool_scheduled_jobs"](action="toggle", job_id="custom_x", enabled=False)
    assert off.get("待确认") is True
    got0 = TOOL_FUNCS["tool_scheduled_jobs"](action="get", job_id="custom_x")
    assert got0["任务"]["enabled"] is True, "preview 不得写库"
    ok = TOOL_FUNCS["tool_scheduled_jobs"](action="commit", change_id=off["change_id"])
    assert ok.get("success") is True
    got = TOOL_FUNCS["tool_scheduled_jobs"](action="get", job_id="custom_x")
    assert got["任务"]["enabled"] is False
    assert "error" in TOOL_FUNCS["tool_scheduled_jobs"](action="toggle", job_id="自定义不存在")
    dp = TOOL_FUNCS["tool_scheduled_jobs"](action="delete", job_id="custom_x")
    assert dp.get("待确认") is True
    d_ok = TOOL_FUNCS["tool_scheduled_jobs"](action="commit", change_id=dp["change_id"])
    assert d_ok.get("success") is True
    assert not tpl_db.get_scheduled_job("custom_x")
    assert "error" in TOOL_FUNCS["tool_scheduled_jobs"](action="delete", job_id="daily_review")


# ================================================================
# 7. tool_panel_bars（叶子 mock：只替换取数与最新日）
# ================================================================

def _fake_bar_frame(n=30, base=100.0):
    idx = pd.date_range("2026-09-01", periods=n, freq="D")
    return pd.DataFrame({"close": [base + i for i in range(n)]}, index=idx)


def test_panel_bars_multi_and_missing(monkeypatch):
    def fake_many(codes, start=None, end=None, **kw):
        return {"600519": _fake_bar_frame(30, 100.0)}   # 000001 刻意缺失
    monkeypatch.setattr("data.bars.get_bars_many", fake_many)
    monkeypatch.setattr("data.bars.panel_date_max", lambda: "2026-10-23")

    r = TOOL_FUNCS["tool_panel_bars"](codes=["600519", "000001"], start="2026-09-01")
    one = r["标的"]["600519"]
    assert one["行数"] == 30
    assert one["数据起止"] == ["2026-09-01", "2026-09-30"]
    assert one["最新收盘"] == 129.0
    assert one["区间涨跌幅%"] == 29.0
    assert len(one["kline_sample"]["closes"]) <= 60
    assert one["kline_sample"]["dates"][-1] == "2026-09-30", "采样末位必须补齐"
    assert r["缺失"] == ["000001"], "缺失要单列，不能当成 0"
    assert r["面板市场级最新日"] == "2026-10-23"


def test_panel_bars_rejects_bad_args(monkeypatch):
    assert "error" in TOOL_FUNCS["tool_panel_bars"](codes=[])
    assert "error" in TOOL_FUNCS["tool_panel_bars"](codes=["60051"])       # 5位
    assert "error" in TOOL_FUNCS["tool_panel_bars"](codes=["600519"], adjust="xx")
    # 6 只必须报错而非静默丢弃第 6 只（质量复核 P2-1）
    six = TOOL_FUNCS["tool_panel_bars"](
        codes=["600519", "000001", "600036", "601318", "300750", "002594"])
    assert "error" in six and "5" in six["error"]


# ================================================================
# 8. tool_data_freshness
# ================================================================

def test_data_freshness_uses_panel_date_max(tpl_db, monkeypatch):
    monkeypatch.setattr("data.bars.coverage",
                        lambda: {"codes": 5830, "mb": 387.4,
                                 "manifest": {"codes": 5830, "rows": 11257063}})
    monkeypatch.setattr("data.bars.panel_date_max", lambda: "2026-10-16")
    monkeypatch.setattr("data.trading_calendar.is_trading_day", lambda d: False)
    monkeypatch.setattr("data.trading_calendar.last_trading_day", lambda: "2026-10-23")

    r = TOOL_FUNCS["tool_data_freshness"]()
    assert r["本地K线面板"]["市场级最新日"] == "2026-10-16"
    assert r["本地K线面板"]["代码数"] == 5830
    assert r["最近交易日"] == "2026-10-23"
    assert "manifest" in r["本地K线面板"], "原始 manifest 要一并给出（排障要看）"
    assert "不是今天" in r["口径说明"]
    assert r["数据集覆盖(data_coverage)"] == "（暂无记录）"


# ================================================================
# 9. 注册表与截断配额
# ================================================================

def test_all_new_tools_registered():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    for n in ("tool_read_output_file", "tool_db_inspect", "tool_config_view",
              "tool_push_templates", "tool_scheduled_jobs", "tool_panel_bars",
              "tool_data_freshness"):
        assert n in names and n in TOOL_FUNCS, f"{n} 未注册"


def test_watchlist_schema_exposes_update():
    schema = next(s for s in TOOL_SCHEMAS
                  if s["function"]["name"] == "tool_manage_watchlist")
    assert "update" in schema["function"]["parameters"]["properties"]["action"]["enum"]
    assert "clear_note" in schema["function"]["parameters"]["properties"]


def test_summarize_result_per_tool_limits():
    """系统级读工具配额更大，其余工具维持 1500（成本护栏不被整体放宽）。"""
    from core.chat_agent import ChatAgent
    big = {"content": "a" * 9000}
    read = ChatAgent._summarize_result(big, "tool_read_output_file")
    default = ChatAgent._summarize_result(big)
    assert len(read) < 7200 and len(read) > 7000
    assert len(default) < 1600 and "已截断" in default
    assert "已截断" in read
    # 未登记的旧工具不受影响
    assert len(ChatAgent._summarize_result(big, "tool_market_signal")) < 1600
