# -*- coding: utf-8 -*-
"""「阿深今日」日报（core/ashen_daily.py）的守护测试。

## 这个文件在守什么

日报的价值不在于"多一节文字"，而在于它把**从表里取的事实**摆出来。
2026-10-28 的实证是：阿深六轮都在说"等伊利 28 元下方"，而条件表 0 条、
决策表 0 条、成交 0 笔 —— 只要日报允许模型凭 summary 自己写，这件事就永远
不会被看见。所以这里守四件事：

1. 数字来自表（值班记录/等待条件/账户），不是模型的话；
2. `enforced`（被系统补正/兜底）**必须**出现在报告里，不能藏起来；
3. 数据不可得要说"数据不可得"，不能渲染成"空仓/0 条"这种看起来像结论的话；
4. 六轮零动作时要有那句判语，且"被补正才有动作"与"主动有动作"不能混。
"""

from datetime import datetime

import pytest

from core import ashen_daily as AD


@pytest.fixture()
def tpl_db(tmp_path, monkeypatch):
    """独立空库（从模板复制）。**生产库一个字节都不许碰**。"""
    from tests.conftest import use_template_db
    use_template_db(tmp_path, monkeypatch, "ashen_daily.db")
    return True


DAY = "2026-10-28"


def _seed(*, with_actions=True, enforced=""):
    from db import store
    acts = [{"action": "watch_add", "code": "000001", "reason": "加池"}] if with_actions \
        else [{"action": "pass", "reason": "等回调"}]
    store.save_routine_run("ai_ashen", DAY, "09:40", status="ok", llm_called=True,
                           summary="开盘先看，等伊利 28 元下方", actions=acts,
                           enforced=enforced)
    store.save_routine_run("ai_ashen", DAY, "10:20", status="skipped",
                           llm_called=False, skip_reason="行情平静")


def test_collect_reads_from_tables(tpl_db):
    _seed()
    d = AD.collect(DAY)
    assert d["stats"]["llm_rounds"] == 1
    assert d["stats"]["actions"] == 1
    assert d["stats"]["slots"] == 2, "被跳过的格也要算进来（否则答不出为何只有 2 轮）"


def test_pass_is_not_an_action(tpl_db):
    """`pass` 不算实质动作 —— 六轮全 pass 的这一天，actions 必须是 0。"""
    _seed(with_actions=False)
    d = AD.collect(DAY)
    assert d["stats"]["llm_rounds"] == 1
    assert d["stats"]["actions"] == 0
    md = AD.render_markdown(DAY)
    assert "一件能落到系统里的事都没做" in md, md


def test_enforced_is_shown_not_hidden(tpl_db):
    """被系统补正/兜底的动作必须标出来。

    藏起来等于帮它遮羞：用户看到"今天有 1 件动作"会以为阿深终于主动了。
    """
    _seed(enforced="no_action→retry_useless+auto_watch")
    d = AD.collect(DAY)
    assert d["stats"]["enforced_rounds"] == 1
    md = AD.render_markdown(DAY)
    assert "补正" in md and "auto_watch" in md, md
    assert "没有一件是他主动提出的" in md, md


def test_normal_day_does_not_blame(tpl_db):
    """主动有动作的日子**不能**被扣上"系统推的"这顶帽子。"""
    _seed(enforced="")
    md = AD.render_markdown(DAY)
    assert "没有一件是他主动提出的" not in md, md


def test_armed_conditions_come_from_table_not_summary(tpl_db):
    """等待条件取自 `ai_wake_conditions`，不是 summary 里说的那句"在等"。"""
    _seed()
    from db import store
    store.upsert_ai_wake_condition("ai_ashen", {"code": "600887", "kind": "price_below",
                                                "threshold": 28.0, "note": "等回踩"})
    md = AD.render_markdown(DAY)
    assert "600887" in md and "price_below" in md and "28" in md, md


def test_no_armed_conditions_is_stated_explicitly(tpl_db):
    """0 条条件要明说，并指出"他说过的等待在系统里不存在"。"""
    _seed()
    md = AD.render_markdown(DAY)
    assert "0 条" in md and "系统里并不存在" in md, md


def test_account_read_failure_is_not_empty_position(tpl_db, monkeypatch):
    """读库失败 ≠ 空仓。写成"空仓"是把故障伪装成结论。"""
    from db import store
    monkeypatch.setattr(store, "load_paper_state",
                        lambda _n: (_ for _ in ()).throw(RuntimeError("db down")))
    d = AD.collect(DAY)
    assert d["account_available"] is False
    md = AD.render_markdown(DAY)
    assert "数据不可得（不是空仓）" in md, md


def test_armed_read_failure_is_not_zero(tpl_db, monkeypatch):
    from db import store
    monkeypatch.setattr(store, "list_wake_conditions",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("db down")))
    md = AD.render_markdown(DAY)
    assert "当前挂着的等待条件：数据不可得" in md, md


def test_brief_for_llm_flags_enforced(tpl_db):
    """喂给复盘模型的摘要也要带"是否被系统补正"，两边口径必须一致。"""
    _seed(enforced="no_action→retry_ok")
    b = AD.brief_for_llm(DAY)
    row = b["逐轮结论"][0]
    assert row["是否被系统补正"] is True
    assert row["实质动作数"] == 1


def test_collect_defaults_to_today(tpl_db):
    """不传日期时默认今天（页面/复盘都是这么调的）。"""
    d = AD.collect()
    assert d["day"] == datetime.now().strftime("%Y-%m-%d")


def test_full_report_has_ashen_section(tpl_db, monkeypatch):
    """接线守护：日报必须真的进 `review_agent.generate_review` 的产物。

    函数写好了不等于页面看得见 —— 这一层只查"章节被拼进报告"。
    完整 LLM 链路在 review 的集成测试里，这里不重复花钱。
    """
    _seed()
    from core import review_agent
    md = AD.render_markdown(DAY)
    assert md.startswith("\n## 阿深今日"), md[:40]
    # 渲染入口确实被 generate_review 调用（源码级断言，防"写了没人调"）
    import inspect
    src = inspect.getsource(review_agent.generate_review)
    assert "ashen_daily.render_markdown" in src, "报告里没有接阿深日报"
    assert "ashen_daily.brief_for_llm" in src, "模型上下文里没有阿深摘要"
