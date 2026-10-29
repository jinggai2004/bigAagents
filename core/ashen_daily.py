# -*- coding: utf-8 -*-
"""「阿深今日」日报（程序取数 + 程序渲染，**零 LLM**）。

## 为什么单独一个模块

用户 2026-10-28 的原话是「每日复盘加上阿深的每天报告」。此前的复盘报告里
**一个字都没有阿深**：`render_decision_sections(digest)` 渲染的是主分析链的
trade_plan（今日决策清单 / 决策质量评估 / 明日待触发），阿深的值班记录、
他挂的条件、他的账户都不在那个 digest 里。

## 为什么数字必须由代码渲染而不是交给模型

同一轮改造里我们刚踩过"模型说什么就算什么"的坑：阿深六轮都说"在等伊利
28 元下方"，而条件表里一条都没有。如果日报也让模型凭 summary 自己写，
它照样能把"今天一直在等"写成"今天持续跟踪"，读者无从分辨。
所以本模块从**表**里取事实，模型只被允许在别处做点评。

## 一个必须保留的区分：`enforced`

`ai_routine_runs.enforced` 非空 = 这一轮是**被系统补正/兜底**才有动作的。
日报必须把它标出来，否则用户看到"今天有 3 件动作"会以为阿深终于主动了。
"""

from __future__ import annotations

import logging
from datetime import datetime

_logger = logging.getLogger(__name__)

PERSONA = "ai_ashen"
# 动作闸门定义的"实质动作"（与 ai_routine.REAL_ACTIONS 同源，不抄字面量）。
# `pass` 不算动作；`study` 只是登记问题。
_REAL = ("buy", "sell", "watch_add", "watch_remove", "set_wake")


def _real_actions(row: dict) -> list[dict]:
    return [a for a in (row.get("actions") or [])
            if isinstance(a, dict) and str(a.get("action") or "").lower() in _REAL]


def collect(day: str | None = None) -> dict:
    """取阿深当日事实。各块独立容错：一块读不到就标 unavailable，**不编**。"""
    from db import store

    day = day or datetime.now().strftime("%Y-%m-%d")
    out: dict = {"day": day, "persona": PERSONA}

    # ① 值班轮次（含被跳过的格 —— "今天为什么只跑了 2 轮"也在这一层回答）
    try:
        runs = store.list_routine_runs(trade_date=day, persona=PERSONA, limit=50)
        runs = sorted(runs or [], key=lambda r: str(r.get("created_at") or ""))
        out["runs"] = runs
        out["runs_available"] = True
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[阿深日报] 值班记录读取失败: %s", e)
        out["runs"], out["runs_available"] = [], False

    # ② 他**真的**挂着的等待条件（不是嘴上说的）
    try:
        armed = store.list_wake_conditions(persona=PERSONA, armed_only=True, limit=50)
        out["armed"] = [r for r in (armed or []) if r.get("code")]
        out["armed_available"] = True
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[阿深日报] 等待条件读取失败: %s", e)
        out["armed"], out["armed_available"] = [], False

    # ③ 账户（现金/持仓/今日成交）
    try:
        st = store.load_paper_state(PERSONA) or {}
        fills = st.get("fills") or []
        today_fills = [f for f in fills if str(f.get("date") or "")[:10] == day]
        out["account"] = {
            "cash": st.get("cash"),
            "positions": [{"code": c, "qty": d.get("quantity"),
                           "avg_cost": d.get("avg_cost")}
                          for c, d in (st.get("positions") or {}).items()],
            "fills_today": today_fills,
            "exists": bool(st),
        }
        out["account_available"] = True
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[阿深日报] 账户读取失败: %s", e)
        out["account"], out["account_available"] = {}, False

    out["stats"] = _stats(out)
    return out


def _stats(d: dict) -> dict:
    """汇总指标 —— 复盘页与 LLM 点评共用，两边口径必须一致。"""
    runs = d.get("runs") or []
    llm = [r for r in runs if int(r.get("llm_called") or 0)]
    acted = [r for r in llm if _real_actions(r)]
    enforced = [r for r in llm if str(r.get("enforced") or "")]
    wakes = [a for r in llm for a in _real_actions(r)
             if str(a.get("action")) == "set_wake"]
    return {
        "slots": len(runs),
        "llm_rounds": len(llm),
        "acted_rounds": len(acted),
        "enforced_rounds": len(enforced),
        "actions": sum(len(_real_actions(r)) for r in llm),
        "wake_actions": len(wakes),
        "armed_now": len(d.get("armed") or []),
        "fills": len((d.get("account") or {}).get("fills_today") or []),
        "positions": len((d.get("account") or {}).get("positions") or []),
    }


def _cond_text(c: dict) -> str:
    if c.get("wake_at"):
        v = f"{c['wake_at']} 定时"
    elif c.get("threshold") is not None:
        v = f"{float(c['threshold']):g}"
    elif c.get("threshold_pct") is not None:
        v = f"{float(c['threshold_pct']) * 100:g}%"
    else:
        v = "（无阈值，不会触发）"
    name = c.get("name") or ""
    return f"{c.get('code')}{(' ' + name) if name else ''} {c.get('kind')} {v}"


def render_markdown(day: str | None = None) -> str:
    """渲染「阿深今日」章节。数据不可得时**明说**，不渲染一段看似正常的话。"""
    d = collect(day)
    s = d["stats"]
    lines = ["", "## 阿深今日", ""]

    # ── 账户一行（空仓就是空仓，不美化）──
    acct = d.get("account") or {}
    if d.get("account_available"):
        cash = acct.get("cash")
        cash_txt = f"{float(cash):,.0f}" if isinstance(cash, (int, float)) else "读不到"
        pos = acct.get("positions") or []
        pos_txt = "、".join(f"{p['code']} {p.get('qty')}股" for p in pos) if pos else "**空仓**"
        lines.append(f"- 账户：现金 {cash_txt} ｜ 持仓 {pos_txt} ｜ 今日成交 {s['fills']} 笔")
    else:
        lines.append("- 账户：数据不可得（不是空仓）")

    # ── 一行总结（把"今天到底动没动"说死）──
    if d.get("runs_available"):
        if not s["llm_rounds"]:
            lines.append(f"- 值班：今天没有调用模型的轮次（{s['slots']} 格被跳过）")
        else:
            line = (f"- 值班：{s['llm_rounds']} 轮调用模型，"
                    f"{s['actions']} 件实质动作，其中 {s['wake_actions']} 件是挂等待条件")
            if s["enforced_rounds"]:
                # 这一行是给老板看的"它到底主不主动"。藏起来等于帮它遮羞。
                line += f"；**{s['enforced_rounds']} 轮是被系统补正/兜底才有动作的**"
            lines.append(line)
    else:
        lines.append("- 值班：数据不可得")

    # ── 他真挂着的条件（嘴上说在等什么 ≠ 系统里挂着）──
    if not d.get("armed_available"):
        lines.append("- 当前挂着的等待条件：数据不可得")
    elif d.get("armed"):
        lines.append(f"- 当前挂着的等待条件（{s['armed_now']} 条，价位到了真会叫他）：")
        for c in d["armed"][:8]:
            lines.append(f"    - {_cond_text(c)}")
    else:
        lines.append("- 当前挂着的等待条件：**0 条** —— 如果他今天说过「在等某某价」，"
                     "那句等待在系统里并不存在")

    # ── 逐轮轨迹 ──
    if d.get("runs_available") and d.get("runs"):
        lines.append("")
        lines.append("**今日轮次**：")
        for r in d["runs"]:
            slot = r.get("slot") or "?"
            if not int(r.get("llm_called") or 0):
                reason = str(r.get("skip_reason") or "未调用模型")[:60]
                lines.append(f"- {slot} 跳过（{reason}）")
                continue
            summary = str(r.get("summary") or "").strip()[:90] or "（无结论）"
            acts = _real_actions(r)
            if acts:
                act_txt = "、".join(
                    f"{a.get('action')} {a.get('code') or ''}".strip() for a in acts)
            else:
                act_txt = "**无实质动作**"
            enforced = str(r.get("enforced") or "")
            mark = f"｜补正:{enforced}" if enforced else ""
            lines.append(f"- {slot} {summary}｜动作 {act_txt}{mark}")

    # ── 一句判语（代码算，不是模型感想）──
    if d.get("runs_available") and s["llm_rounds"] and not s["actions"]:
        lines.append("")
        lines.append("> 今天他每轮都在说话，但**一件能落到系统里的事都没做**。")
    elif s["enforced_rounds"] and s["enforced_rounds"] == s["acted_rounds"]:
        lines.append("")
        lines.append("> 今天所有动作都出自系统补正或兜底，没有一件是他主动提出的。")

    return "\n".join(lines)


def brief_for_llm(day: str | None = None) -> dict:
    """给复盘 LLM 的摘要（它只能点评，数字由上面的渲染段负责）。"""
    d = collect(day)
    return {
        "日期": d["day"],
        "统计": d["stats"],
        "挂着的等待条件": [_cond_text(c) for c in (d.get("armed") or [])[:6]],
        "逐轮结论": [{"轮次": r.get("slot"),
                      "结论": str(r.get("summary") or "")[:120],
                      "实质动作数": len(_real_actions(r)),
                      "是否被系统补正": bool(str(r.get("enforced") or ""))}
                     for r in (d.get("runs") or [])
                     if int(r.get("llm_called") or 0)],
        "今日成交笔数": d["stats"]["fills"],
        "持仓数": d["stats"]["positions"],
    }


if __name__ == "__main__":                       # 手动预览：python -m core.ashen_daily
    print(render_markdown())
