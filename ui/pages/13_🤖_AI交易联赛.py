"""AI 交易联赛页 — 每日复盘 + 设置（P1，2026-10-14）

给用户两件事：
  1. **每日复盘**：每个 AI 对前一交易日的复盘（自述 + 程序化事实核对）与今日操作理由；
  2. **便捷设置**：直接编辑各 AI 的模型/风格/风险偏好/上限/启停，写回 `config/ai_personas.json`
     （mtime 缓存 ⇒ **改完即时生效，不用重启**）。

设计纪律：
  - 复盘区**只读、零联网、零 LLM**（离线安全）；数字全部来自账本，不经 LLM；
  - "AI 自述复盘"与"程序化事实"**分开呈现**，用户能一眼看出两者是否一致；
  - 设置写库前先 sanitize（`ai_personas.save_personas` 内部做），非法输入被丢弃并提示；
  - 口径声明用 `st.caption`/`st.markdown`（`_texts` 冒烟只收这两类，写 subheader 会离线消失）。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from datetime import datetime

from db import store

st.set_page_config(page_title="AI 交易联赛", page_icon="🤖", layout="wide")
st.title("🤖 AI 交易联赛")

from core import ai_personas, ai_review, ai_league  # noqa: E402

_OFFLINE = os.environ.get("BIGA_UI_OFFLINE") == "1"

tab_review, tab_settings, tab_board = st.tabs(["📋 每日复盘", "⚙️ 设置", "🏆 榜单"])


# ================================================================
# 📋 每日复盘
# ================================================================
def _order_tag(d: dict) -> str:
    """决策行末尾的**订单状态**后缀（成交 / 未结算 / 作废 / 错过窗口）。

    ⚠️ 必须显示。本项目反复出现的失败形态就是「AI 说了买卖，但根本没成交」：
    结算任务取不到价静默跳过、卖出决策没有结算路径、积压单错过 T+1 窗口……
    不显示状态，用户就会以为"操作过了"，实际账户一动不动。
    """
    status = (d.get("order_status") or "").strip()
    if status:
        return f"　·　{status}"
    if d.get("advice_status") in ("可执行", "卖出"):
        return "　·　⚠️ 未结算"
    return ""


def _render_review():
    st.markdown("**每日复盘：AI 对前一交易日的复盘 + 今日操作理由**")
    st.caption("⚠️ 每行分两块：**AI 自述复盘**（模型原话，可能不准）与"
               "**账本事实**（程序化、不经 LLM）——请对照看，别只信自述。")

    # 交易日选择：默认今天；只读（不触发任何网络）。
    default_day = datetime.now().strftime("%Y-%m-%d")
    try:
        days = sorted({str(r.get("signal_date")) for r in store.list_ai_decisions()},
                      reverse=True)
    except Exception:                                        # noqa: BLE001
        days = []
    options = days if default_day in days else ([default_day] + days)
    day = st.selectbox("复盘基准日（决策日）", options, index=0, key="ai_review_day")

    rv = ai_review.daily_review(day)
    for w in rv["warnings"]:
        st.warning(w)
    st.caption(f"决策日 **{rv['as_of']}**｜前一交易日 **{rv['prev_trading_day'] or '—'}**"
               + (f"｜随机对照组当日平均净值变动 **{rv['random_mean_return_pct']}%**"
                  if rv["random_mean_return_pct"] is not None else ""))

    if not rv["rows"]:
        st.info("还没有任何 AI 账户/决策数据。")
        return

    # ⚠️ 盘前/盘中打开时"今天"通常还没跑决策轮（每交易日 15:58 才跑）⇒
    #    默认落在今天就会**整屏空白**（"今日无决策记录" + "未留下自述复盘"），
    #    用户很容易误判成功能坏了。这里明确告知**什么时候会跑**，而不是留白。
    if day == default_day and not any(r["today_decisions"] for r in rv["rows"]):
        st.info("⏳ **今天的决策轮还没运行** —— 每个交易日 **15:58** 自动执行"
                "（复盘前一交易日 + 给出当日决策）。\n\n"
                "想先看往期记录，把上面「复盘基准日」切到有记录的日期即可。")

    for r in rv["rows"]:
        _kind = {"llm": "AI", "random": "随机对照"}.get(r.get("kind"), r.get("kind"))
        _title = f"{r['label']}（{_kind}"
        if r.get("model"):
            _title += f"｜{r['model']}"
        _title += "）"
        with st.expander(_title, expanded=(r.get("kind") == "llm")):
            # —— 今日操作 + 理由 ——
            st.markdown("**今日交易操作与理由**")
            if r["today_decisions"]:
                for d in r["today_decisions"]:
                    st.markdown(f"- `{d.get('code')}` **{d.get('advice_status') or '—'}**"
                                f"（仓位 {d.get('position_size')}）{_order_tag(d)}"
                                f"｜{d.get('reason') or '（无理由）'}")
            else:
                st.caption("今日无决策记录。")
            if r.get("today_summary"):
                st.caption(f"本轮总体思路：{r['today_summary']}")

            # —— 前一日复盘 ——
            st.markdown(f"**对前一交易日（{r['prev_date'] or '—'}）的复盘**")
            st.markdown(f"· AI 自述：{r.get('ai_review_text') or '（该轮未留下自述复盘）'}")
            if r["prev_decisions"]:
                _pl = "；".join(
                    f"`{d.get('code')}` {d.get('advice_status') or '—'}"
                    f"（{d.get('position_size')}）{_order_tag(d)}" for d in r["prev_decisions"])
                st.caption(f"账本事实 · 昨日决策：{_pl}")
            else:
                st.caption("账本事实 · 昨日无决策记录。")

            # —— 账户事实 ——
            _bits = []
            if r["nav_today"]:
                _bits.append(f"净值 {r['nav_today']['total_value']:,.2f}")
            if r["nav_change_pct"] is not None:
                _bits.append(f"较前一日 {r['nav_change_pct']:+.2f}%")
            if r["fills_today"]:
                _fs = "；".join(
                    f"`{f.get('code')}` {f.get('direction')} {f.get('quantity')}股"
                    f" @{f.get('actual_price')}" for f in r["fills_today"])
                _bits.append(f"当日成交：{_fs}")
            else:
                _bits.append("当日无成交")
            if r["positions"]:
                _ps = "；".join(
                    f"`{p['code']}` {p['qty']}股（可卖 {p['available']}）"
                    for p in r["positions"])
                _bits.append(f"持仓：{_ps}")
            st.caption("账本事实 · " + "｜".join(_bits))
            # ⚠️ 口径提示必须展示：净值含"按成本估值"的部分时，用户不能拿它当盈亏。
            if r.get("nav_note"):
                st.caption("⚠️ " + r["nav_note"])


# ================================================================
# ⚙️ 设置
# ================================================================
_BIAS_CN = {"conservative": "保守", "balanced": "均衡", "aggressive": "激进"}
_BIAS_OPTS = list(_BIAS_CN.keys())
_MODEL_HINT = "如 qwen-max / deepseek-chat（留空 = 用系统默认模型）"


def _render_settings():
    st.markdown("**AI 设置**")
    st.caption("改完点「保存全部」即写入 `config/ai_personas.json`，**即时生效、不用重启**"
               "（人格注册表是 mtime 缓存）。模型/风格/风险偏好/单票上限都会影响下一次决策轮。"
               "⚠️ 随机对照组（kind=random）**不调 LLM**：没有模型与风格可设。")

    personas = ai_personas.load_personas()

    with st.form("ai_persona_form"):
        edited = []
        for i, p in enumerate(personas):
            is_llm = p.get("kind") == "llm"
            st.markdown(f"#### {p.get('label') or p['name']}"
                        f"（`{p['name']}`｜{'AI' if is_llm else '随机对照'}）")
            c1, c2, c3 = st.columns([2, 2, 1])
            label = c1.text_input("显示名", value=p.get("label") or p["name"],
                                  key=f"label_{i}")
            model = c2.text_input("模型", value=p.get("model") or "",
                                  key=f"model_{i}", help=_MODEL_HINT,
                                  disabled=not is_llm)
            enabled = c3.checkbox("启用", value=bool(p.get("enabled", True)),
                                  key=f"enabled_{i}")
            d1, d2, d3 = st.columns([2, 2, 2])
            bias = d1.selectbox("风险偏好", _BIAS_OPTS,
                                index=_BIAS_OPTS.index(p["risk_bias"])
                                if p.get("risk_bias") in _BIAS_OPTS else 1,
                                format_func=lambda x: _BIAS_CN.get(x, x),
                                key=f"bias_{i}")
            maxpos = d2.number_input("单票上限（0 = 用系统默认 25%）",
                                     min_value=0.0, max_value=1.0,
                                     value=float(p.get("max_position") or 0.0),
                                     step=0.05, key=f"maxpos_{i}")
            cash = d3.number_input("初始资金", min_value=0.0,
                                   value=float(p.get("initial_cash") or 1_000_000),
                                   step=100_000.0, key=f"cash_{i}")
            style = ""
            if is_llm:
                style = st.text_area("风格提示词", value=p.get("style_prompt") or "",
                                     height=120, key=f"style_{i}",
                                     help="只影响取舍偏好；系统交易纪律由代码兜底，风格不是豁免权。")
            edited.append({**p, "label": label, "model": (model or None),
                           "enabled": enabled, "risk_bias": bias,
                           "max_position": (maxpos or None),
                           "initial_cash": cash, "style_prompt": style})
            st.divider()

        saved = st.form_submit_button("💾 保存全部", type="primary")

    if saved:
        ok = ai_personas.save_personas(edited)
        if ok:
            n = ai_personas.sync_to_store()
            st.success(f"已保存 {len(edited)} 个人格（同步快照 {n} 条）。下一次决策轮即生效。")
        else:
            st.error("保存失败（内容为空或被拒绝）——已保留原配置，请检查后重试。")

    b1, b2 = st.columns(2)
    if b1.button("♻️ 恢复默认人格", key="ai_reset_personas",
                 help="会用内置默认（3 AI + 3 随机对照）**覆盖**当前配置。"):
        ai_personas.save_personas(ai_personas.DEFAULT_PERSONAS)
        ai_personas.sync_to_store()
        st.success("已恢复默认人格配置。请刷新页面查看。")
    if b2.button("🔄 同步到数据库快照", key="ai_sync_personas",
                 help="把 JSON 定义同步到 ai_personas 表（供 SQL 关联/健康检查读数）。"):
        n = ai_personas.sync_to_store()
        st.success(f"已同步 {n} 条。")

    # —— 盘中补仓开关（含自动回退状态 + 人工复位）——
    st.divider()
    st.markdown("**盘中补仓（持仓内小额度）**")
    try:
        from core import ai_league as _al
        _wb = _al.wake_buy_status()
        _ev = _wb.get("evidence") or {}
        if _wb.get("enabled"):
            st.caption(f"✅ 当前**开放**：{_wb.get('reason')}")
        else:
            st.error(f"⛔ 当前**关闭**：{_wb.get('reason')}")
        if _ev.get("days"):
            st.caption("回退判据（按补仓日的**已实现**盈亏，未平仓不计入）："
                       + "｜".join(f"{d}: {_ev['pnl'].get(d)}" for d in _ev["days"]))
        else:
            st.caption("回退判据：目前还没有「已平仓的盘中补仓」样本。")
        st.caption("硬边界（代码强制，AI 无法绕过）：仅**已有持仓**标的、仅 `price_below` 触发；"
                   "单次 ≤ 持仓市值 3% 且 ≤ 权益 1%；单票 1 次/日、全日 ≤2 次且 ≤ 权益 2%。")
        _c1, _c2 = st.columns([1, 3])
        if _c1.button("🔓 人工复位（重新开放）", key="ai_wake_buy_enable"):
            if _al.write_wake_buy_state(manual_enable=True, reason="用户在设置页手动复位"):
                st.success("已重新开放盘中补仓。")
            else:
                st.error("写入失败（状态文件不可写）。")
        if _c2.button("🔒 人工关闭", key="ai_wake_buy_disable"):
            if _al.write_wake_buy_state(manual_enable=False, reason="用户在设置页手动关闭"):
                st.success("已关闭盘中补仓（自动回退判据仍会继续统计）。")
            else:
                st.error("写入失败（状态文件不可写）。")
    except Exception as e:                                   # noqa: BLE001
        st.warning(f"⚠️ 盘中补仓状态不可用（{type(e).__name__}）：{e}")

    # —— 手动"收盘价确认持仓"（零 LLM 成本的便捷入口）——
    st.divider()
    st.markdown("**手动确认持仓（收盘价口径）**")
    st.caption("⚠️ 口径说明：这是**模拟联赛的加速确认**——按**当日收盘价**把未成交的买入"
               "撮合成持仓，供你盘后立刻看到持仓/净值。正式结算仍走次日 09:32 的 **T+1 开盘价**"
               "（真实可行）。已确认过的不会重复成交（幂等）。")
    if _OFFLINE:
        st.caption("（离线模式：已跳过手动确认入口——它会请求行情。）")
    else:
        _cd = st.date_input("确认交易日", value=datetime.now().date(), key="ai_confirm_day")
        if st.button("✅ 按当日收盘价确认持仓", key="ai_confirm_btn"):
            try:
                out = ai_league.settle_at_close(str(_cd))
                _lines = [f"- {r['persona']}：成交 {r['filled']} 笔，"
                          f"持仓 {r['positions']} 只，净值 {r['nav']['total_value']:,.2f}"
                          for r in out["results"]]
                st.success(f"已按收盘价确认（{out['trade_date']}）：\n" + "\n".join(_lines))
            except Exception as e:                            # noqa: BLE001
                st.error(f"确认失败（{type(e).__name__}）：{e}")


# ================================================================
# 🏆 榜单
# ================================================================
def _render_board():
    st.markdown("**榜单（分频分组，随机对照组为方法论基线）**")
    try:
        lb = ai_league.leaderboard()
    except Exception as e:                                   # noqa: BLE001
        st.warning(f"榜单不可用（{type(e).__name__}）：{e}")
        return
    for c in lb["caveat"]:
        st.caption(c)
    if not lb["groups"]:
        st.info("还没有榜单数据：跑一次决策轮 + 确认持仓后即可看到。")
        return
    import pandas as pd
    for freq, g in lb["groups"].items():
        st.markdown(f"**{'日频（T+1 开盘成交）' if freq == 'daily' else '日内（扫描价成交）'} 组**")
        rm = g.get("random_mean_return_pct")
        st.caption(f"随机对照组均值 {('—' if rm is None else f'{rm}%')}（{g['random_count']} 个）｜"
                   f"本组样本是否充分：{'是' if g['conclusive'] else '**否（无统计意义）**'}")
        df = pd.DataFrame([{
            "AI": r.get("label") or r["persona"],
            "类型": {"llm": "AI", "random": "随机对照"}.get(r["kind"], r["kind"]),
            "模型": r.get("model") or "—（规则）",
            "累计收益%": r["return_pct"],
            "最大回撤%": r["max_drawdown_pct"],
            "参与度%（有持仓天数占比）": r.get("participation_pct"),
            "平均仓位%": r.get("avg_exposure_pct"),
            "盘中交易占比%": r.get("intraday_trade_pct"),
            "盘中补仓胜率%": r.get("intraday_buy_win_rate"),
            "相对随机超额%": r.get("excess_vs_random_mean"),
            "随机分位%": r.get("random_percentile"),
            "交易日": r["nav_days"], "成交笔数": r["trades"],
            "样本充分": "是" if r["conclusive"] else "否",
        } for r in g["rows"]])
        st.dataframe(df, width="stretch", hide_index=True)


with tab_review:
    try:
        _render_review()
    except Exception as e:                                   # noqa: BLE001
        st.warning(f"⚠️ 复盘区不可用（{type(e).__name__}）：{e}")

with tab_settings:
    try:
        _render_settings()
    except Exception as e:                                   # noqa: BLE001
        st.warning(f"⚠️ 设置区不可用（{type(e).__name__}）：{e}")

with tab_board:
    try:
        _render_board()
    except Exception as e:                                   # noqa: BLE001
        st.warning(f"⚠️ 榜单区不可用（{type(e).__name__}）：{e}")
