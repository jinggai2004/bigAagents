"""模拟账户页 — 虚拟资金、按现价成交、持仓与净值（Phase 2）

设计取舍：
  - **刷新安全**：账户状态在 SQLite 里（`paper_accounts/positions/orders/fills/nav`），
    刷新页面直接从库里重建，不像选股/深度分析那样依赖会话状态；
  - **现价按需获取**：只有点「刷新现价」或下单时才请求行情。未刷新时持仓按成本价估值，
    并在页面上写明——避免每次渲染都联网，也避免把过期缓存当现价；
  - **过期缓存拒绝成交**：行情 `stale=True` 时不允许成交，否则会按旧价记账；
  - T+1 与涨跌停口径全部来自 `core.matching` / `core.paper_account`（与回测同规则）。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from db import store
from core import matching
from core.paper_account import PaperAccount
from data import realtime
from ui.components.section_nav import section_nav, visible

st.set_page_config(page_title="模拟账户", page_icon="💰", layout="wide")
st.title("💰 模拟账户")

DEFAULT_CASH = 1_000_000.0


def _account_choices() -> list:
    """可选账户：默认账户 + 所有 AI 联赛人格 + 库里已有的其它账户。

    ⚠️ 不是"只列 AI"——手工账户（`default` 与用户自建）必须仍可选，
    否则改造会**悄悄剥夺**原有功能（把单账户页变成只看联赛的页）。
    """
    from core import ai_personas
    ordered = ["default"]
    for p in sorted(ai_personas.load_personas(), key=lambda x: x["name"]):
        if p["name"] not in ordered:
            ordered.append(p["name"])
    for n in store.list_paper_accounts():
        if n not in ordered:
            ordered.append(n)
    return ordered


_ACCOUNT_CHOICES = _account_choices()
# ⚠️ widget 的 session_state 必须在**该 widget 实例化之前**写（项目铁律）；
#    这里直接用返回值，不预先写 state。
ACCOUNT_NAME = st.selectbox(
    "账户", _ACCOUNT_CHOICES, index=0, key="paper_account_select",
    help="`default` 为手工账户；`ai_*` 是 AI 交易联赛的模拟账户（各 AI 独立账本）")


def load_or_create() -> PaperAccount:
    state = store.load_paper_state(ACCOUNT_NAME)
    if state is None:
        acct = PaperAccount(DEFAULT_CASH, name=ACCOUNT_NAME)
        store.save_paper_state(ACCOUNT_NAME, acct.to_state())
        return acct
    return PaperAccount.from_state(state)


def persist(acct: PaperAccount):
    store.save_paper_state(acct.name, acct.to_state())


def fetch_prices(codes: list) -> dict:
    codes = [c for c in dict.fromkeys(codes) if c]
    if not codes:
        return {}
    return {c: q for c, q in realtime.get_realtime_quotes(codes).items() if q}


@st.cache_data(ttl=realtime.QUOTES_TTL, show_spinner=False)
def _cached_quotes(codes: tuple) -> dict:
    """页面级行情缓存（TTL 与 `realtime.QUOTES_TTL` 对齐 ⇒ 口径不变）。

    ⚠️ 入参必须是 **tuple**（list 不可哈希，`st.cache_data` 无法建键）。
    ⚠️ **失败也返回 `{}`**：`st.cache_data` **不缓存异常** —— 若让异常穿透，
       每次 rerun / 每次切账户都会把同一个超时重等一遍（看盘页与热点页都踩过）。
    """
    try:
        return fetch_prices(list(codes))
    except Exception:                                        # noqa: BLE001
        return {}


acct = load_or_create()

# ⚠️ 进入页面就**主动解冻 T+1**（2026-10-10 用户实际踩到）：
# 解冻原先只发生在"有成交的调用"里（`apply_trade`/`settle_open`），
# 于是**光刷新页面不会解冻** —— 周五买入的仓位到了周末仍是 `available=0`，
# 用户点卖出只会看到 "T+1 未解锁或无可用持仓"，而真实原因只是"没人帮你解冻过"。
# 这里在渲染持仓表之前调用一次，保证页面上的「可卖」列是**按今天**算出来的。
# 有实际解冻才回写库（避免每次刷新都写一遍）。
_unlocked = acct.unlock_t1()
if _unlocked:
    persist(acct)

# 行情：**按账户缓存 + 缺价就自动补取**（2026-10-16 修复，用户报"资金要加载才正确"）
#
# 原实现两处缺陷，合起来导致"必须点一次「刷新现价」才正确"：
#   ① 缓存键是**全局**的 `"paper_quotes"`，不区分账户 ⇒ 切到 `ai_value` 时复用的是
#      `default`（无持仓）留下的空 dict，**新账户的持仓一只价都没有**；
#   ② 守卫写成 `if "paper_quotes" not in st.session_state` ⇒ 只要首次进页面时账户空仓，
#      键就被写成了 `{}`，此后**任何账户都不会再自动取价**。
# ⇒ 持仓只能按**成本价**估值，于是「总资产 = 现金 + 成本 ≈ 初始资金」，
#   用户看到 "1,000,000（+0.00%）" 会以为账户被重置（正是 10-09 加自动取价要避免的误读）。
_QUOTES_KEY = f"paper_quotes::{ACCOUNT_NAME}"
quotes = st.session_state.get(_QUOTES_KEY) or {}
_priceless = [c for c in acct.positions
              if not (isinstance((quotes.get(c) or {}).get("price"), (int, float))
                      and (quotes.get(c) or {})["price"] > 0)]
if _priceless:
    _fetched = _cached_quotes(tuple(sorted(_priceless)))
    if _fetched:
        quotes = {**quotes, **_fetched}
        st.session_state[_QUOTES_KEY] = quotes
price_map = {c: (q or {}).get("price") for c, q in quotes.items()}
_priced = [c for c in acct.positions
           if isinstance(price_map.get(c), (int, float)) and price_map[c] > 0]
_valuation = "现价" if len(_priced) == len(acct.positions) and acct.positions else "成本价"

# ================= 概览 =================
total = acct.total_value(price_map)
holdings_value = acct.holdings_value(price_map)
ret_pct = (total / acct.initial_cash - 1) * 100 if acct.initial_cash else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"总资产（{_valuation}口径）", f"{total:,.2f}", f"{ret_pct:+.2f}%")
c2.metric("可用现金", f"{acct.cash:,.2f}")
c3.metric("持仓市值", f"{holdings_value:,.2f}")
c4.metric("持仓只数", len(acct.positions))
st.caption(f"初始资金 {acct.initial_cash:,.0f} 元 ｜ 账户 `{acct.name}` ｜ "
           f"成交 {len(acct.fills)} 笔 ｜ 数据存于 SQLite，刷新页面不丢")
if _valuation == "成本价" and acct.positions:
    st.info("⚠️ 部分持仓暂无现价，上表按**持仓成本价**估值——**不是账户被重置**。"
            "点下方「🔄 刷新现价」即按最新行情估值。")

# ================= 现价 =================
pcol1, pcol2 = st.columns([1, 4])
with pcol1:
    if st.button("🔄 刷新现价", width="stretch"):
        # ⚠️ 两处都要清：`realtime` 的**内部缓存** + 页面级 `st.cache_data`。
        #    只清前者会被页面缓存整个短路 ⇒ 函数压根不被调用 ⇒ "点了刷新价格不动"
        #    （看盘页已踩过同一个坑，这里必须同款）。
        realtime.clear_quote_cache()
        _cached_quotes.clear()
        st.session_state[_QUOTES_KEY] = _cached_quotes(tuple(sorted(acct.positions)))
        st.rerun()
with pcol2:
    if quotes:
        stale = [c for c, q in quotes.items() if (q or {}).get("stale")]
        st.caption(f"现价已更新（{len(quotes)} 只）" + (f"；其中 {len(stale)} 只为过期缓存，仅展示不用于成交" if stale else ""))
    else:
        st.caption("未刷新现价：下表按**持仓成本价**估值，不联网、不使用过期缓存")


# ================= 区块快速定位 =================
# 区块快速定位（2026-10-15）：本页 5 个区块纵向很长（下单/持仓/挂单成交/净值/AI联赛）
# ⇒ 顶部下拉框切换。⚠️ 默认「全部显示」保持改造前行为；
#   账户/现价等**共用前置**仍留在各区块之外，任何选择下都先算好（否则区块之间会互相依赖）。
SEC_ORDER = "下单"
SEC_HOLD = "持仓"
SEC_PENDING = "挂单与成交"
SEC_NAV = "净值"
SEC_LEAGUE = "🏆 AI 交易联赛"
_SEL = section_nav((SEC_ORDER, SEC_HOLD, SEC_PENDING, SEC_NAV, SEC_LEAGUE),
                   key="paper_section", group_label="💰 模拟账户 · 板块")

if visible(_SEL, SEC_ORDER):
    # ================= 下单 =================
    st.subheader(SEC_ORDER)

    FRACTIONS = {"1/4 仓": 0.25, "1/3 仓": 1 / 3, "1/2 仓": 0.5,
                 "2/3 仓": 2 / 3, "全仓": 1.0}


    def _round_lot(qty: float) -> int:
        """向下取整到 100 股（A股一手）；不足一手返回 0。"""
        return int(qty // 100) * 100


    o1, o2 = st.columns([2, 2])
    with o1:
        code = st.text_input("股票代码", value="", placeholder="6位代码",
                             max_chars=6, key="order_code")
    with o2:
        direction = st.selectbox("方向", ["买入", "卖出"], key="order_direction")

    _valid_code = bool(code) and len(code) == 6 and code.isdigit()
    pos = acct.positions.get(code) if _valid_code else None

    # 代码输入完成且 session 里还没有它的现价 → 自动拉一次（之后不再重复拉）
    quote = None
    if _valid_code:
        if code in quotes:
            quote = quotes[code] or None
        else:
            fetched = fetch_prices([code]).get(code)
            if fetched:
                quotes = {**quotes, code: fetched}
                st.session_state[_QUOTES_KEY] = quotes
                quote = fetched
    price = (quote or {}).get("price") if isinstance((quote or {}).get("price"), (int, float)) else None

    # ---- 数量来源：按仓位比例（默认，贴合操作习惯）或按数量 ----
    mode = st.radio("下单方式", ["按仓位比例", "按数量"], horizontal=True, key="order_mode")

    qty: int | None = None          # None = 还算不出来（缺现价/无持仓等，禁止提交）
    _qty_note = ""
    if mode == "按仓位比例":
        fcol1, fcol2 = st.columns([1, 2])
        with fcol1:
            frac_label = st.selectbox("仓位比例", list(FRACTIONS), key="order_frac")
        frac = FRACTIONS[frac_label]
        if direction == "买入":
            if price:
                # 预算按可用现金的比例，费用上浮 0.3% 保守估（佣金 0.025% + 余量），
                # 避免按比例算出的数量在扣费后现金不足被拒
                budget = acct.cash * frac / 1.003
                qty = _round_lot(budget / price)
                _qty_note = f"≈ 用 {frac_label}现金（{acct.cash * frac:,.0f} 元）按现价 {price} 折股"
            else:
                _qty_note = "⚠️ 暂无该股现价，无法按比例折股——请稍后重试或改用「按数量」"
        else:                       # 卖出：比例作用于**可卖数量**
            if pos is None:
                _qty_note = "⚠️ 无该股持仓，不能卖出"
            else:
                base = pos.available if frac_label != "全仓" else pos.quantity
                qty = _round_lot(base * frac) if frac_label != "全仓" else base
                _qty_note = f"≈ 可卖 {pos.available} 股的 {frac_label}"
                if frac_label == "全仓" and pos.available < pos.quantity:
                    _qty_note += f"（当日买入的 {pos.quantity - pos.available} 股受 T+1 限制，不可卖）"
    else:
        # 按数量：上限动态钳制——卖出 ≤ 可卖，买入 ≤ 现金可买上限（有现价时）
        max_qty = None
        if direction == "卖出":
            max_qty = pos.available if pos else 0
        elif price:
            max_qty = _round_lot(acct.cash / (price * 1.003))
        if direction == "卖出" and (pos is None or pos.available <= 0):
            st.caption("⚠️ 该股无可卖持仓（未持有或 T+1 未解锁）")
        qcol1, qcol2 = st.columns([1, 2])
        with qcol1:
            quantity = st.number_input(
                "数量（股，100 的整数倍）", min_value=0,
                max_value=max_qty if max_qty is not None else None,
                step=100, value=0, key="order_qty")
        with qcol2:
            if max_qty is not None:
                st.caption(f"本次可输入上限：**{max_qty:,} 股**（超出部分会被拒绝）")
        qty = int(quantity) if quantity > 0 else None
        if qty is not None and qty % 100 != 0:
            st.caption("⚠️ 数量需为 100 的整数倍（一手）")

    if _qty_note:
        st.caption(_qty_note)
    if qty:
        st.caption(f"待下单：**{direction} {code} {qty:,} 股**"
                   + (f" @ 现价 {price}" if price else "（无现价，提交时再取）"))

    # ---- 确认弹窗：成交前必须人工确认（防手滑） ----
    @st.dialog("⚠️ 确认下单")
    def _confirm_dialog():
        st.markdown(f"### {direction} {name_display}（{code}）")
        st.markdown(f"**数量：{qty:,} 股**　｜　参考现价：{price}")
        st.divider()
        # 用真实撮合口径预估（含滑点与费用），而不是自己再算一套
        # ⚠️ `matching.buy_fill/sell_fill` 返回的是 `actual_price`（已含滑点），
        #    **没有** `price` 键（2026-10-10 实测：KeyError: 'price'）。
        #    用 `est["actual_price"]`；直接访问 `["price"]` 会让确认弹窗必崩。
        if direction == "买入":
            est = matching.buy_fill(price, qty)
            st.markdown(f"预估成交价 {est['actual_price']}　｜　预估金额（含佣金）**{est['total']:,.2f} 元**")
            st.caption(f"成交后可用现金约 {acct.cash - est['total']:,.2f} 元")
        else:
            est = matching.sell_fill(price, qty)
            st.markdown(f"预估成交价 {est['actual_price']}　｜　预估金额（扣印花税后）**{est['total']:,.2f} 元**")
            if qty > (pos.available if pos else 0):
                st.error(f"⚠️ 数量超过可卖（{(pos.available if pos else 0):,} 股），将被拒绝")
            else:
                st.caption(f"预计盈亏：{(est['actual_price'] - pos.avg_cost) * qty:+,.2f} 元"
                           f"（成本 {pos.avg_cost:.3f}）")
        st.caption("确认后将重新取一次最新行情成交（过期缓存会被拒绝）。")
        b1, b2 = st.columns(2)
        if b1.button("✅ 确认成交", type="primary", width="stretch"):
            fresh = fetch_prices([code]).get(code)
            if not fresh:
                st.error("取不到该股票行情，未成交")
            elif fresh.get("stale"):
                st.error("行情为过期缓存（数据源暂时不可用），已拒绝成交——避免按旧价记账")
            else:
                out = acct.apply_trade({
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "code": code, "direction": direction,
                    "price": fresh["price"], "quantity": int(qty)})
                if out and out.get("status") == "rejected":
                    st.error(f"未成交：{out['reason']}")
                elif out is None:
                    st.error("未成交：指令无效（数量/价格不合法）")
                else:
                    persist(acct)
                    st.session_state[_QUOTES_KEY] = {**quotes, code: fresh}
                    st.rerun()
        if b2.button("❌ 取消", width="stretch"):
            st.rerun()


    name_display = (quote or {}).get("name") or code
    # ⚠️ `price is not None` 是**外部依赖**（联网取行情），不是"用户没填完"。
    #    原先两者共用一句提示"填写完整（代码/数量或比例/现价）后即可提交"，
    #    用户取不到行情时会以为是自己漏填，反复改数量也点不动按钮（2026-10-10 复盘）。
    #    这里把禁用原因**分流**成可区分的三种，并各自给出下一步动作。
    _partial = dict(
        code_ok=_valid_code,
        qty_ok=bool(qty and qty > 0 and qty % 100 == 0),
        sell_ok=not (direction == "卖出" and pos is None),
    )
    _has_quote = price is not None
    can_submit = (_partial["code_ok"] and _partial["qty_ok"]
                  and _partial["sell_ok"] and _has_quote)
    if st.button("🧾 提交下单（弹窗确认）", type="primary",
                 disabled=not can_submit, key="order_submit"):
        _confirm_dialog()

    # 未就绪时，明确"卡在哪一步"，而不是一句笼统的"填写完整"
    if not can_submit and (_valid_code or mode):
        if not _partial["code_ok"]:
            st.caption("ℹ️ 请输入 6 位股票代码。")
        elif not _has_quote:
            st.caption(f"⚠️ **取不到 {code} 的现价**，无法按比例折股/校验金额，提交按钮已禁用。"
                       "这不等于你没填完 —— 请稍后重试，或点上方「🔄 刷新现价」；"
                       "若长期取不到，可能是该代码不存在或行情源暂时不可用。")
        elif not _partial["sell_ok"]:
            st.caption(f"ℹ️ 当前未持有 {code}（或 T+1 未解锁），无法卖出。")
        elif not _partial["qty_ok"]:
            st.caption("ℹ️ 数量需为 **100 的整数倍**（一手），且不能为 0。")

    st.caption("成交价 = 现价叠加滑点，并计佣金（0.025%，最低 5 元）与卖出印花税 0.1%；"
               "买入当日不可卖（T+1）。按信号自动下单 + 次日开盘撮合留给调度任务。")

if visible(_SEL, SEC_HOLD):
    # ================= 持仓 =================
    st.subheader(SEC_HOLD)
    if not acct.positions:
        st.info("当前空仓")
    else:
        rows = []
        for c, pos in sorted(acct.positions.items()):
            q = quotes.get(c) or {}
            price = q.get("price")
            has_price = isinstance(price, (int, float)) and price > 0
            valued = price if has_price else pos.avg_cost
            pnl = (valued - pos.avg_cost) * pos.quantity
            rows.append({
                "代码": c, "名称": q.get("name") or "—",
                "数量": pos.quantity, "可卖": pos.available,
                "成本价": round(pos.avg_cost, 3),
                "现价": round(valued, 3),
                "估值来源": "现价" if has_price else "成本价",
                "市值": round(valued * pos.quantity, 2),
                "浮动盈亏": round(pnl, 2),
                "盈亏%": round(pnl / (pos.avg_cost * pos.quantity) * 100, 2) if pos.avg_cost else 0.0,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("「可卖」小于「数量」说明有当日买入的仓位受 T+1 限制")

if visible(_SEL, SEC_PENDING):
    # ================= 挂单与成交 =================
    pending = [o for o in acct.orders if o.status == "pending"]
    if pending:
        st.subheader("待撮合挂单（下一个交易日开盘执行）")
        st.dataframe(pd.DataFrame([o.to_dict() for o in pending]), width="stretch", hide_index=True)

    if acct.fills:
        with st.expander(f"成交明细（最近 20 笔，共 {len(acct.fills)} 笔）", expanded=False):
            recent = list(reversed(acct.fills[-20:]))
            st.dataframe(pd.DataFrame(recent), width="stretch", hide_index=True)

if visible(_SEL, SEC_NAV):
    # ================= 净值 =================
    st.subheader(SEC_NAV)
    n1, n2 = st.columns([1, 4])
    with n1:
        if st.button("📌 记录今日净值", width="stretch"):
            rec = acct.mark_to_market(datetime.now().strftime("%Y-%m-%d"), price_map)
            persist(acct)
            st.success(f"已记录：总资产 {rec['total_value']:,.2f}（{rec['return_pct']:+.2f}%）")
    with n2:
        st.caption("按当前估值记一条净值（回测看板的净值曲线口径与之一致）")

    if acct.nav:
        nav_df = pd.DataFrame(acct.nav)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["total_value"],
                                 mode="lines+markers", name="总资产",
                                 line=dict(color="#3498db", width=2)))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10),
                          yaxis_title="总资产 (元)")
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("还没有净值记录：点「记录今日净值」开始累计")

    # ================= 重置 =================
    with st.expander("⚙️ 账户操作", expanded=False):
        st.warning("重置会**删除**该账户的全部持仓、成交与净值记录，不可恢复。")
        if st.button("🗑️ 重置模拟账户"):
            removed = store.delete_paper_account(ACCOUNT_NAME)
            st.success(f"已重置（清理 {removed} 行），请刷新页面")
            st.session_state.pop(_QUOTES_KEY, None)
            st.rerun()

    # ================= 🏆 AI 交易联赛 =================
    # ⚠️ 本区块**只读**（不触发任何 LLM、不联网）⇒ 离线模式下也安全。
    # ⚠️ 口径纪律：日频与日内**不可混榜**；样本不足时必须明说；不用"胜率"当判据。

if visible(_SEL, SEC_LEAGUE):
    st.divider()
    # ⚠️ 用 markdown 而非 st.header：`_texts`（冒烟断言）只收 markdown/caption，
    #    header/subheader 会被整段忽略 ⇒ 关键说明写在那里等于"离线不可见"。
    st.markdown("### " + SEC_LEAGUE + "（三风格 × 随机对照）")

    try:
        from core import ai_budget, ai_league, ai_personas

        lb = ai_league.leaderboard()
        # ⚠️ caveat 用 caption（markdown 类）展示：`_texts` 只收 markdown/caption，
        #    写在 subheader 里离线会整段消失（项目已踩过）。
        for _c in lb["caveat"]:
            st.caption(_c)

        if not lb["groups"]:
            st.info("还没有任何 AI 账户数据：先跑一次盘后决策轮"
                    "（`python scripts/run_ai_league.py`）并等结算任务执行。")
        for _freq, _g in lb["groups"].items():
            _freq_cn = {"daily": "日频（T+1 开盘成交）",
                        "intraday": "日内（扫描价成交）"}.get(_freq, _freq)
            st.markdown(f"**{_freq_cn} 组**")
            _rm = _g.get("random_mean_return_pct")
            st.caption(f"随机对照组均值 {('—' if _rm is None else f'{_rm}%')}"
                       f"（{_g['random_count']} 个）｜"
                       f"本组样本是否充分：{'是' if _g['conclusive'] else '**否（无统计意义）**'}")
            import pandas as _pd
            _df = _pd.DataFrame([{
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
                "交易日": r["nav_days"],
                "成交笔数": r["trades"],
                "样本充分": "是" if r["conclusive"] else "否",
            } for r in _g["rows"]])
            st.dataframe(_df, width="stretch", hide_index=True)

        # ---- 净值对比（各 AI 账本）----
        _nav_rows = {}
        for _p in ai_personas.load_personas():
            _st = store.load_paper_state(_p["name"])
            if not _st:
                continue
            for _n in (_st.get("nav") or []):
                _nav_rows.setdefault(_n["date"], {})[_p["label"] or _p["name"]] = _n["total_value"]
        if _nav_rows:
            import pandas as _pd
            _navdf = _pd.DataFrame.from_dict(_nav_rows, orient="index").sort_index()
            st.markdown("**净值走势（各 AI 独立账本）**")
            st.line_chart(_navdf)

        # ---- 额度视图（今天还剩多少可用的研究/唤醒额度）----
        with st.expander("今日额度（每个 AI 的 deep_analysis / wake / decision_round）",
                         expanded=False):
            _today = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
            _rows = []
            for _p in ai_personas.load_personas():
                _v = ai_budget.view(_p["name"], _today)
                _rows.append({"AI": _p["label"] or _p["name"],
                              **{f"{k}（用/额）": f"{v['used']}/{v['quota']}"
                                 for k, v in _v.items()}})
            if _rows:
                import pandas as _pd
                st.dataframe(_pd.DataFrame(_rows), width="stretch", hide_index=True)
    except Exception as _e:                                  # noqa: BLE001
        st.warning(f"⚠️ 联赛信息不可用（{type(_e).__name__}）：不影响上方的账户功能。")
