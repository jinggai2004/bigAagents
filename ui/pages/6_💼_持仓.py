"""持仓管理页面 — 真实持仓录入/平仓、实时盈亏、风险敞口"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st

from core.portfolio import (get_portfolio_manager, level_sanity,
                            level_sanity_blocking, SEVERITY_WARN)
from data import realtime

st.set_page_config(page_title="持仓管理", page_icon="💼", layout="wide")
st.title("💼 持仓管理")
st.caption("录入真实持仓后，AI 助手即可回答'我的持仓怎么样'；盈亏按实时行情计算")


# ================================================================
# 行情：**批量取一次**，之后透传给所有需要现价的调用（2026-10-15）
# ================================================================
# ⚠️ 为什么这是本页最大的性能开关：`PortfolioManager` 的
#    `load()` / `check_stop_loss()` / `check_trailing_stops()` / `get_exposure()`
#    **各自都会逐只联网取价**，走的是 akshare 东财接口 → 取数受限时要重试 3 次 + 指数退避
#    （实测单只 ≈30s 才失败）。本页一次渲染实测触发 **≈9 次取价** ⇒ 1 只持仓 63 秒。
#    而本页**本来就已经批量取过行情**（腾讯直连、批量、快）⇒ 透传即可全部省掉。
# ⚠️ TTL 15 秒：与 `data.realtime.QUOTES_TTL` 对齐，不改变数据新鲜度语义。
@st.cache_data(ttl=15, show_spinner=False)
def _cached_quotes(codes: tuple):
    if not codes:
        return {}
    # ⚠️ 入参必须是 tuple（list 不可哈希，`st.cache_data` 无法建键）
    return realtime.get_realtime_quotes(list(codes))


def _flash(msg: str) -> None:
    """登记一条"操作成功"提示，**下一轮**再显示。

    ⚠️ 为什么不能直接 `st.success(...)`：它后面紧跟 `st.rerun()`，
       Streamlit 会把本轮已输出的元素**整个丢弃** ⇒ 用户**永远看不到**那句"开仓成功"。
       操作类反馈必须跨 rerun 存活（本项目反复的失败形态就是"看着像做了、其实没提示"）。
    """
    st.session_state["_flash_msg"] = msg


def _render_flash() -> None:
    msg = st.session_state.pop("_flash_msg", None)
    if msg:
        st.success(msg)


def _position_codes() -> tuple:
    """活跃持仓代码（**不取价**）。

    ⚠️ 不能用 `get_active_positions()`：那条路会走 `load()` → 逐只取价，
       而我们现在正处在"为了少取价才要先拿代码"的位置上（鸡生蛋）。
    """
    return tuple(get_portfolio_manager().active_codes())


def color_pnl(val):
    if isinstance(val, (int, float)):
        color = "#e4574c" if val > 0 else ("#3fa66a" if val < 0 else "#888")
        return f"color: {color}; font-weight: bold"


# 行业 → ATR 倍数（与 core/risk.py::RiskEngine.ATR_MULTIPLIER 保持一致）。
# 波动越大的板块给越宽的止损带，否则正常回撤就会被打掉。
_SECTOR_OPTS = {
    "默认(2.5×)": "",
    "科技(3.0×)": "科技",
    "新能源(3.0×)": "新能源",
    "医药(2.5×)": "医药",
    "金融(2.0×)": "金融",
    "消费(2.0×)": "消费",
}


def _apply_atr_suggestion(code: str, price: float, sector: str):
    """按 ATR 算建议止损/止盈，暂存到中间态（不直接落库，用户可再改）。

    ⚠️ **不能直接写 `st.session_state["open_sl"]`**：那两个 key 是开仓表单里
    `st.number_input(..., key="open_sl"/"open_tp")` 的 widget key，而本函数是被
    同一轮里**创建 widget 之后**的按钮触发的。Streamlit 禁止在 widget 实例化之后
    修改它的 key，会抛：

        StreamlitAPIException: `st.session_state.open_sl` cannot be modified
        after the widget with key `open_sl` is instantiated.

    这正是「点按钮就报错」的原因。正确做法：先存中间态，下一轮在**创建 widget
    之前**（`_drain_pending_levels`）落到 widget key 上。
    """
    if not (code.isdigit() and len(code) == 6):
        st.warning("请先填写 6 位股票代码")
        return
    with st.spinner("读取K线并计算 ATR…"):
        try:
            from core.risk import get_risk_engine
            info = get_risk_engine().suggest_levels(code, entry_price=price, sector=sector)
        except Exception as e:                              # noqa: BLE001
            st.error(f"计算失败：{type(e).__name__}: {e}")
            return
    if info.get("unavailable"):
        # 取不到就明说，不用"看起来合理"的固定百分比硬凑
        st.warning(f"未能给出建议：{info.get('note')}")
        return
    st.session_state["_pending_levels"] = {
        "code": code,
        "stop_loss": float(info["stop_loss"]),
        "take_profit": float(info["take_profit"]),
        "info": info,
    }
    st.rerun()


def _drain_pending_levels():
    """把上一轮算好的建议写进 widget key。

    **必须在创建 open_sl / open_tp 这两个 widget 之前调用**（见 `_apply_atr_suggestion`
    的说明）。写成"创建后再写"就会抛 StreamlitAPIException。
    """
    pending = st.session_state.pop("_pending_levels", None)
    if not pending:
        return
    st.session_state["open_sl"] = pending["stop_loss"]
    st.session_state["open_tp"] = pending["take_profit"]
    st.session_state["_atr_info"] = pending["info"]


def _level(v) -> str:
    """价格位显示：有值给数字（去掉多余的 .0），无值给「—」（明确"未设置"）。

    ⚠️ 必须**统一成字符串**（2026-10-15 实测发现）：这一列原先混用数字与 "—"，
       会让 `st.dataframe` 的 Arrow 序列化直接失败：
           `ArrowInvalid: Could not convert '—' ... tried to convert to double`
       Streamlit 会走"自动修复"兜底（页面仍能显示），但**控制台会刷一大段 traceback**，
       用户会以为页面坏了 —— 而且这条路每渲染一次就走一次。
    """
    return f"{float(v):g}" if isinstance(v, (int, float)) and v else "—"


def render_positions(quotes: dict | None = None):
    pm = get_portfolio_manager()
    prices = quotes or {}
    positions = pm.get_active_positions(prices=prices)
    if not positions:
        st.info("当前无持仓，请在下方录入开仓记录")
        return
    if quotes is None:                      # 兼容直调（未透传行情时保持原行为）
        quotes = realtime.get_realtime_quotes([p.code for p in positions])
    rows = []
    for pos in positions:
        q = quotes.get(pos.code)
        price = q["price"] if q and q.get("price") else pos.entry_price
        market_value = price * pos.quantity
        pnl = round((price - pos.entry_price) * pos.quantity, 2)
        pnl_pct = round((price / pos.entry_price - 1) * 100, 2) if pos.entry_price else None
        rows.append({
            "代码": pos.code, "名称": pos.name or (q["name"] if q else "") or "—",
            "建仓日期": pos.entry_date, "成本价": pos.entry_price,
            "数量": pos.quantity, "现价": price, "市值": round(market_value, 2),
            "浮动盈亏": pnl, "盈亏%": pnl_pct,
            "止损": _level(pos.stop_loss), "止盈": _level(pos.take_profit),
        })
    df = pd.DataFrame(rows)
    styler = df.style.map(color_pnl, subset=["浮动盈亏", "盈亏%"])
    st.dataframe(styler, width="stretch", hide_index=True, height=300)

    # 汇总
    portfolio = pm.load(prices=prices)
    total_pnl = sum(r["浮动盈亏"] for r in rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总资产(元)", f"{portfolio.total_value:,.0f}")
    c2.metric("现金(元)", f"{portfolio.cash:,.0f}")
    c3.metric("持仓市值(元)", f"{portfolio.total_value - portfolio.cash:,.0f}")
    c4.metric("合计浮动盈亏", f"{total_pnl:+,.0f}",
              delta="盈利" if total_pnl > 0 else ("亏损" if total_pnl < 0 else "持平"),
              delta_color="off")
    return positions


def render_risk_panel(quotes: dict | None = None):
    """风险巡检：止损/止盈触发 + 追踪止损可上移 + 行业敞口。

    这三块能力（`check_stop_loss` / `check_trailing_stops` / `get_exposure`）
    在 core 层早就实现了，但此前**没有任何地方调用** —— 持仓录了止损价，
    系统却从不检查它有没有被打穿。这里把它接上。
    """
    pm = get_portfolio_manager()
    prices = quotes or {}
    positions = pm.get_active_positions(prices=prices)
    if not positions:
        return

    st.subheader("🛡️ 风险巡检")

    # ── 1) 止损/止盈触发 ──
    try:
        alerts = pm.check_stop_loss(prices=prices)
    except Exception as e:                                  # noqa: BLE001
        st.warning(f"止损检查失败：{type(e).__name__}: {e}")
        alerts = []
    if alerts:
        for a in alerts:
            tag = "🔴" if a.get("type") == "止损" else "🟢"
            pct = a.get("loss_pct") if a.get("type") == "止损" else a.get("profit_pct")
            st.error(f"{tag} **{a.get('type')}触发** {a.get('code')} {a.get('name') or ''}："
                     f"现价 {a.get('current_price')} vs 设定 {a.get('trigger_price')}"
                     + (f"（浮动 {pct:+.2f}%）" if isinstance(pct, (int, float)) else ""))
    else:
        st.caption("✅ 无持仓触及已设定的止损/止盈价。")

    # ── 2) 追踪止损（只给建议，改库要你确认）──
    try:
        trails = pm.check_trailing_stops(prices=prices)
    except Exception as e:                                  # noqa: BLE001
        st.warning(f"追踪止损检查失败：{type(e).__name__}: {e}")
        trails = []
    if trails:
        st.markdown("**止损调整建议** —— 浮盈超 2×ATR 时按追踪止损上移（锁利润）；"
                    "浮亏时只是建议把偏宽的止损收紧到 ATR 标准位。"
                    "两种都**只朝有利方向移动**，绝不会把止损改松。")
        rows = [{"代码": t["code"], "名称": t["name"] or "—",
                 "成本价": t["entry_price"], "现价": t["current_price"],
                 "浮动%": t["profit_pct"],
                 "当前止损": t["current_stop"] if t["current_stop"] else "未设",
                 "建议止损": t["suggested_stop"], "类型": t["kind"],
                 "说明": t["note"]} for t in trails]
        st.dataframe(rows, width="stretch", hide_index=True)
        opt = {f"{t['code']} {t['name'] or ''} → {t['suggested_stop']}（{t['kind']}）": t
               for t in trails}
        pick = st.selectbox("选择要应用的建议", list(opt.keys()), key="apply_trail")
        if st.button("✅ 应用该止损价", key="apply_trail_btn"):
            t = opt[pick]
            if pm.update_stop_loss(t["code"], t["suggested_stop"]):
                _flash(f"已将 {t['code']} 的止损更新为 {t['suggested_stop']}")
                st.rerun()
            else:
                st.error("更新失败（持仓可能已平仓）")
    else:
        st.caption("ℹ️ 暂无建议上移的止损（浮盈未达 2×ATR，或 ATR/价格取不到）。"
                   "追踪止损不会把止损改松，未达标就不动。")

    # ── 3) 持仓结构体检（集中度 / 流动性）──
    render_position_structure(prices)


def render_position_structure(quotes: dict | None = None):
    """持仓结构体检：单票/行业集中度 + 流动性（退出所需天数）。

    计算唯一实现 `core/position_risk.py`（纯函数；阈值只认 `RiskEngine` 一处）。

    ⚠️ 为什么换掉原来的"行业敞口"（2026-10-17）：那版是按**代码前缀**粗分的
       （60→"金融/公用"、00→"金融/地产"…），与真实行业毫无关系 ——
       把中石化和招商银行算成同一个"行业"，集中度结论是错的。
       现在用 `market_snapshot` 里的**真实行业**，并顺带补上流动性。
    """
    from core.position_risk import evaluate_positions
    try:
        rep = evaluate_positions(prices=quotes or {})
    except Exception as e:                                  # noqa: BLE001
        st.warning(f"结构体检失败：{type(e).__name__}: {e}")
        return
    if not rep.get("positions"):
        return

    st.markdown("**持仓结构体检**（集中度 / 流动性）")
    if rep.get("data_unavailable") or not rep.get("evaluated"):
        st.warning(f"结构体检不可用：{rep.get('data_unavailable') or '取不到现价或组合为空'} —— "
                   "未用**成本价**顶替计算占比。")
        return

    lim = rep["limits"]
    cols = st.columns(4)
    cols[0].metric("总仓位", f"{rep['position_ratio']:.1%}")
    cols[1].metric("现金", f"{rep['cash_ratio']:.1%}")
    top = rep.get("top_position")
    cols[2].metric("最大单票", f"{top['weight']:.1%}" if top else "—",
                   delta=f"上限 {lim['single']:.0%}", delta_color="off")
    sec = rep.get("sector_exposure") or {}
    if sec:
        k = next(iter(sec))
        cols[3].metric(f"最大行业·{k[:6]}", f"{sec[k]:.1%}",
                       delta=f"上限 {lim['sector']:.0%}", delta_color="off")
    else:
        cols[3].metric("最大行业", "不可评",
                       delta=f"上限 {lim['sector']:.0%}", delta_color="off")

    for i in rep["issues"]:
        if i["severity"] == "high":
            st.error(i["message"])
        else:
            st.warning(i["message"])
    if not rep["issues"]:
        st.caption(f"✅ 结构体检通过：单票 ≤{lim['single']:.0%}、行业 ≤{lim['sector']:.0%}、"
                   f"退出所需 <{lim['liquidity_days']:g} 个交易日成交额。")

    rows = [{
        "代码": p["code"], "名称": p["name"] or "—",
        "市值": f"{p['market_value']:,.0f}",
        "占净值": f"{p['weight']:.1%}" if p["weight"] is not None else "—",
        "行业": p["industry"] or "不可评",
        "日成交额(亿)": f"{p['amount'] / 1e8:,.2f}" if p.get("amount") else "不可评",
        "退出需(天)": f"{p['liquidity_days']:.2f}" if p["liquidity_days"] is not None else "不可评",
    } for p in rep["positions"]]
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)

    # ⚠️ 不可评的项必须**如实列出**：只报"通过"而不说"有几只没算进去"，
    #    等于默示它们没问题（本项目铁律：缺就是缺，不补造、也不隐藏）。
    miss = rep.get("missing_industry") or []
    if miss:
        st.caption(f"⚠️ 行业**不可评**：{'、'.join(miss)} —— 不在 "
                   f"{rep.get('snapshot_date') or '—'} 的估值截面快照里，"
                   f"**未**并进任何行业统计（不猜、不并进「其他」）。")
    un = rep.get("unavailable_liquidity") or []
    if un:
        st.caption(f"⚠️ 流动性**不可评**：{'、'.join(un)} —— 取不到当日成交额。"
                   f"⚠️ **不等于**流动性没问题。")
    st.caption("口径：占比 = **当前市值 ÷ 总资产**（总资产 = 现金 + 持仓市值）；"
               f"流动性 = **持仓市值 ÷ 该股当日成交额**（天）。"
               f"行业与成交额取自 **{rep.get('snapshot_date') or '—'}** 的估值截面快照。"
               "止盈止损由 ATR 与行业倍数算出，仅供参考，不构成投资建议。")


def render_trade_forms(quotes: dict | None = None):
    pm = get_portfolio_manager()
    prices = quotes or {}
    # 必须在创建 open_sl / open_tp 两个 widget **之前**（见 _apply_atr_suggestion）
    _drain_pending_levels()
    tab_open, tab_close = st.tabs(["📥 开仓（买入）", "📤 平仓（卖出）"])
    with tab_open:
        c1, c2 = st.columns(2)
        with c1:
            code = st.text_input("股票代码", placeholder="6位代码", max_chars=6)
            # ⚠️⚠️ **不得给"成交价"设一个看起来像真实价格的默认值**（2026-10-16 真实事故）：
            #    原为 `value=10.0, min_value=0.01` ⇒ 用户不改就提交，成本价被**静默**写成 10.0
            #    （中石化真实价 5.53）⇒ 盈亏、止损检查、AI 诊断全部跟着错；
            #    更糟的是 `min_value=0.01` 让下面那句 `price <= 0` 校验**永远不可能触发**（假校验）。
            #    现在留空（`value=None`）并要求必填 —— 占位符只是提示，不会被当成数据。
            price = st.number_input("成交价", min_value=0.0, value=None, step=0.01,
                                    placeholder="必填：你的实际成交价")
            stop_loss = st.number_input("止损价（可选，0=不设）", min_value=0.0, step=0.01,
                                        key="open_sl")
        with c2:
            name = st.text_input("股票名称（可选）", placeholder="如 平安银行")
            # ⚠️ 同样**不给假默认值**：原为 `value=100` ⇒ 忘记改就少记仓位（100 股看起来"像真的"）。
            quantity = st.number_input("数量（股，100 的整数倍）", min_value=0, value=None,
                                       step=100, placeholder="必填：你的实际股数")
            take_profit = st.number_input("止盈价（可选，0=不设）", min_value=0.0, step=0.01,
                                          key="open_tp")

        # 新代码不在持仓里 ⇒ 上面那份批量行情没有它。单独补取一次
        # （否则"与现价对比"的体检对**新开仓**永远不生效，而新开仓正是默认值事故的场景）。
        if code.isdigit() and len(code) == 6 and code not in prices:
            _one = _cached_quotes((code,))
            if _one:
                prices = {**prices, **_one}
        _cur = (prices.get(code) or {}).get("price") if isinstance(prices.get(code), dict) else None
        if isinstance(_cur, (int, float)) and _cur > 0:
            _q = prices[code]
            st.caption(f"现价 **{_cur:g}**（{_q.get('name') or code}，行情日期 "
                       f"{_q.get('quote_date') or '—'}）")

        # 止损/止盈不用自己拍脑袋：按 ATR 算（与深度分析链同一套口径）
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            sector_label = st.selectbox("行业（决定 ATR 倍数）", list(_SECTOR_OPTS.keys()),
                                        key="open_sector")
        with sc2:
            st.caption("按 ATR（14日真实波幅）计算，盈亏比固定 1:2。"
                       "波动大的板块用更宽的止损带，避免正常回撤被打掉。")
            if st.button("📐 按 ATR 建议止损/止盈", key="open_atr_btn"):
                if not price:
                    st.warning("请**先填成交价**再点 ATR 建议 —— 否则会按一个占位价算出"
                               "与实际无关的止损位（历史事故：默认价 10.0 让中石化的止损算成了 9.8）。")
                else:
                    _apply_atr_suggestion(code, float(price), _SECTOR_OPTS[sector_label])
            _info = st.session_state.get("_atr_info")
            if _info and _info.get("code") == code:
                st.info(f"ATR {_info['atr']}（占成本 {_info['atr_pct']}%），倍数 "
                        f"{_info['multiplier']}× → 止损 **{_info['stop_loss']}**"
                        f"（风险 {_info['risk_pct']}%）、止盈 **{_info['take_profit']}**")
                if _info.get("warning"):
                    st.warning(_info["warning"])

        # ---- 保存前的"自洽性体检"（唯一实现 `core.portfolio.level_sanity`）----
        # ⚠️ 只**如实展示矛盾**、不硬拦：`止损 > 成本价` 在"浮盈后上移止损"时是合法的。
        #    但"止损不低于现价 ⇒ 一录入就已触发"这类自相矛盾必须让人**看见**再决定。
        _issues = level_sanity(entry_price=float(price or 0), stop_loss=stop_loss,
                               take_profit=take_profit, current_price=_cur)
        for _i in _issues:
            _msg = "⚠️ 价位体检：" + _i["message"]
            st.warning(_msg) if _i["severity"] == SEVERITY_WARN else st.info(_msg)
        _ack = True
        if level_sanity_blocking(_issues):
            _ack = st.checkbox("我已核对上面的价位矛盾，确认仍要按这些价位保存",
                               key="open_ack")

        reason = st.text_input("开仓理由（可选）", placeholder="如 技术突破")
        if st.button("📥 确认开仓", type="primary"):
            if not (code.isdigit() and len(code) == 6):
                st.error("股票代码须为6位数字")
            elif not price or float(price) <= 0:
                st.error("请填写**你的实际成交价** —— 这里刻意不给默认值："
                         "默认值会被当成真实成本价写进持仓，之后盈亏与止损检查全错。")
            elif not quantity or int(quantity) <= 0 or int(quantity) % 100 != 0:
                st.error("请填写**实际股数**，且为 100 的整数倍（一手）"
                         "—— 这里刻意不给默认值：忘记改就会把仓位记少。")
            elif not _ack:
                st.error("上方价位体检有未确认的矛盾，请先勾选确认框（或改掉价位）")
            else:
                pos_id = pm.open_position(code=code, price=float(price), quantity=int(quantity),
                                          stop_loss=float(stop_loss) or None,
                                          take_profit=float(take_profit) or None,
                                          name=name, reason=reason or "手动录入")
                _flash(f"开仓成功（持仓ID {pos_id}）：{code} {quantity} 股 @ {price}")
                st.rerun()
    with tab_close:
        # ⚠️ 必须传 prices：不传的话这里会为**每只持仓**自行取价（实测单只 ≈30s 的失败重试路径）。
        positions = pm.get_active_positions(prices=prices)
        if not positions:
            st.caption("当前无持仓可平")
        else:
            options = {f"{p.code} {p.name or ''} {p.quantity}股": p for p in positions}
            choice = st.selectbox("选择持仓", list(options.keys()))
            pos = options[choice]
            _qc = prices.get(pos.code) or {}
            _live = _qc.get("price") if isinstance(_qc.get("price"), (int, float)) else None
            _live = _live if (_live and _live > 0) else None
            c1, c2 = st.columns(2)
            with c1:
                # ⚠️⚠️ **不得默认用成本价**（2026-10-17 修）：
                #    原为 `value=float(pos.entry_price)` ⇒ 用户不改就提交，
                #    平仓价被静默记成**成本价** ⇒ 盈亏恒为 **0**，且看起来"很合理"。
                #    现改为**默认当日现价**（取不到则留空必填），且 key 带代码 ——
                #    否则切换持仓时会复用上一只的输入值（Streamlit widget state 按 key 存）。
                close_price = st.number_input(
                    "卖出价（必填）", min_value=0.0,
                    value=float(_live) if _live else None, step=0.01,
                    key=f"close_price_{pos.code}",
                    placeholder="必填：你的实际卖出价",
                    help="默认取当日现价。**不是**成本价——默认成本价会把盈亏记成 0。")
            with c2:
                close_reason = st.text_input("平仓原因", value="手动平仓")
            if close_price and float(close_price) > 0:
                est_pnl = round((float(close_price) - pos.entry_price) * pos.quantity, 2)
                _pct = ((float(close_price) / pos.entry_price - 1) * 100
                        if pos.entry_price else 0.0)
                st.caption(f"成本价 **{pos.entry_price:g}** ｜ 现价 "
                           f"{_live if _live else '（未取到）'} ｜ 卖出价 **{close_price:g}**"
                           f" ⇒ 预计盈亏 **{est_pnl:+,.0f} 元（{_pct:+.2f}%）**")
                if abs(float(close_price) - pos.entry_price) < 1e-9:
                    st.warning("⚠️ 卖出价与**成本价相同** ⇒ 盈亏会被记成 0。"
                               "若这不是你的实际成交价，请改掉再提交。")
            else:
                est_pnl = 0.0
                st.caption("请填写**实际卖出价**（默认取现价；取不到现价时需手填）。")
            if st.button("📤 确认平仓", type="primary",
                         disabled=not (close_price and float(close_price) > 0)):
                ok = pm.close_position(code=pos.code, price=float(close_price),
                                       reason=close_reason)
                if ok:
                    _flash(f"已平仓 {pos.code}，盈亏 {est_pnl:+,.0f} 元")
                    st.rerun()
                else:
                    st.error("平仓失败")


def main():
    _render_flash()
    st.subheader("当前持仓")
    # ⚠️ 行情**只批量取一次**，然后把它传给所有需要现价的调用（2026-10-15）。
    #    实测背景：`PortfolioManager.load()` / `check_stop_loss()` / `check_trailing_stops()` /
    #    `get_exposure()` **各自都会逐只联网取价**，而 `_get_current_price()` 走 akshare 东财接口，
    #    取数受限时要重试 3 次 + 指数退避（单只 ≈30s）⇒ 本页一次渲染实测触发 **≈9 次取价**，
    #    1 只持仓就要 **63 秒**（随持仓数线性放大）。
    #    而本页其实**本来就已经批量取过行情**（`realtime.get_realtime_quotes` 走腾讯直连、快）
    #    ⇒ 把这份行情透传下去，那 9 次全部省掉。
    quotes = _cached_quotes(_position_codes())
    render_positions(quotes)
    st.divider()
    render_risk_panel(quotes)
    st.divider()
    render_trade_forms(quotes)
    st.divider()
    st.caption("💡 录入持仓后，到「AI 助手」问'我的持仓怎么样'即可获得 AI 诊断；"
               "数据存储于本地 SQLite（data/portfolio.db）")
    st.caption("⚠️ 本页仅为持仓记录与盈亏跟踪，不连接券商账户，不构成投资建议。")


main()
