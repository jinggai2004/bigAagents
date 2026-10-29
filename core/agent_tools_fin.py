# -*- coding: utf-8 -*-
"""财务报表工具（从 `core/agent_tools.py` 拆出，P2-2，2026-10-17）。

包含：三表候选科目静态表（`_FIN_FIELDS` 等）+ 科目提取/单季派生 helpers +
`tool_financials`。拆分动机：agent_tools.py 1292 行难以维护；本模块自成一体
（只依赖 `agent_tools_base` 与 `data.stock_data`）。

⚠️ 对外契约不变：`core/agent_tools.py` **re-export** 全部原名 ⇒
既有 `patch("core.agent_tools.tool_financials")`、`_at._FIN_FIELDS` 等测试零改动。
"""
import logging

from core.agent_tools_base import _tool, _now   # noqa: F401

_logger = logging.getLogger(__name__)

# ================================================================
# 财务报表工具用到的静态表
# ================================================================

# 「展示名 → 候选科目名」。**必须用候选表而不是固定科目名**：不同行业的科目名差别很大。
# 实测（601166 兴业银行 2026-06-30，单位亿元）：
#   一般股口径        →  银行口径
#   营业总收入           营业收入 1101.77
#   流动资产/存货        →（银行没有这些）资产总计 114609.40、负债合计 105001.17
#   归属于母公司所有者的净利润 → 归属于母公司的净利润 411.31，且多出「净利息收入」728.04
# 若写死"营业总收入/流动资产"，银行股会整行取空。
# 归母权益的候选科目。提成常量是因为它被两处用（资产表展示段 + ROE 计算），
# 原先第二处写成 `_FIN_FIELDS["资产负债表"][2][1]` —— 靠**下标**取值，
# 一旦有人重排候选表，ROE 会静默用错科目。
_FIN_EQUITY_CANDIDATES = ("归属于母公司股东的权益", "归属于母公司股东权益合计",
                          "归属于母公司所有者权益合计", "所有者权益(或股东权益)合计",
                          "股东权益合计")

_FIN_FIELDS = {
    "利润表": (
        ("营业收入", ("营业收入", "营业总收入")),
        ("净利润", ("净利润",)),
        ("归母净利润", ("归属于母公司股东的净利润", "归属于母公司的净利润",
                       "归属于母公司所有者的净利润")),
        ("营业利润", ("营业利润",)),
        ("净利息收入", ("净利息收入",)),
        ("手续费及佣金净收入", ("手续费及佣金净收入",)),
        ("研发费用", ("研发费用",)),
        ("销售毛利率", ("销售毛利率",)),
    ),
    "资产负债表": (
        ("资产总计", ("资产总计", "资产")),
        ("负债合计", ("负债合计", "负债")),
        ("归母权益", _FIN_EQUITY_CANDIDATES),
        # 工商股项在前、银行项在后：科目按"行业是否存在"自然过滤（缺失的不占位），
        # 于是工商股拿到 货币资金/存货，银行拿到 发放贷款及垫款/贷款损失准备。
        # 若把银行项前置，工商股会被"发放贷款及垫款"占位而丢掉"存货"——
        # 实测 600519 也有小额"发放贷款及垫款"（其财务公司放款），会挤掉存货。
        ("货币资金", ("货币资金",)),
        ("存货", ("存货",)),
        ("应收账款", ("应收账款",)),
        ("发放贷款及垫款", ("发放贷款及垫款",)),
        ("贷款损失准备", ("减:贷款损失准备", "贷款损失准备")),
    ),
    "现金流量表": (
        ("经营活动净额", ("经营活动产生的现金流量净额", "经营活动产生的现金流量")),
        ("投资活动净额", ("投资活动产生的现金流量净额", "投资活动产生的现金流量")),
        ("筹资活动净额", ("筹资活动产生的现金流量净额", "筹资活动产生的现金流量")),
        ("期末现金及等价物", ("期末现金及现金等价物余额",)),
    ),
}

# 财报三表里**没有**、但对某些行业最关键的指标 —— 必须如实说明取不到，
# 否则问"银行股值不值得投"时模型会缺了 ROE/不良率还答得像很完整。
_FIN_NOT_IN_STATEMENTS = ("不良贷款率", "净息差", "拨备覆盖率", "资本充足率")

# 任何行业都该有的科目。只有这些缺失才值得报"未取到"——
# 「货币资金/存货/应收账款」对银行、券商本来就不存在，报出来只会误导模型。
_FIN_CORE_LABELS = ("营业收入", "净利润", "资产总计", "负债合计", "归母权益", "经营活动净额")


def _num(value):
    """把财报里的值转成 float（单位：元）；取不到返回 None。

    源里同一列可能是 float、也可能带千分位/单位/占位符（"--"、"N/A"），
    且 DataFrame 里的空值是 **NaN**（`NaN != NaN` 为真，不能直接用 `== None` 判）。
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if value != value else float(value)      # NaN -> None
    s = str(value).strip().replace(",", "")
    if not s or s.lower() in ("--", "-", "n/a", "nan", "none", "null"):
        return None
    mult = 1.0
    for suffix, m in (("万亿", 1e12), ("亿元", 1e8), ("亿", 1e8), ("万元", 1e4), ("万", 1e4)):
        if s.endswith(suffix):
            s, mult = s[: -len(suffix)], m
            break
    try:
        return float(s) * mult
    except ValueError:
        return None


def _fin_period_key(value) -> str:
    """报告期归一成 8 位 YYYYMMDD。

    直接复用 `data.stock_data._period_key`：那里已经处理了各源写法
    （新浪 `20260630`、同花顺 `1998`/`1998-12-31`），**不允许再写第二套口径**。
    """
    from data.stock_data import _period_key
    return _period_key(value)


def _fin_rows(fin: dict, key: str) -> list:
    """取某张表的 (报告期, 记录)，**按报告期倒序**（最新在前）。

    不能依赖源的行序：新浪最新在前、同花顺最老在前（见 `_latest_period_records` 的说明）。
    """
    out = []
    for r in (fin.get(key) or []):
        pk = _fin_period_key(r.get("报告日") or r.get("报告期"))
        if pk != "00000000":
            out.append((pk, r))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def _fin_series(rows: list, candidates: tuple, n: int):
    """在已倒序的 rows 里找**第一个数据够用**的候选科目，返回 (列名, [亿元...])。

    判据是"过半期数有值"而不是"第一行有值"：财报里同一科目常有断档
    （如银行「资产」这一行在部分期次为空，但「资产总计」有值）。
    """
    limit = rows[:n]
    if not limit:
        return None, None
    need = max(1, (len(limit) + 1) // 2)
    fallback = None
    for name in candidates:
        vals = [_num(r.get(name)) for _, r in limit]
        if len([v for v in vals if v is not None]) < need:
            continue
        rounded = [None if v is None else round(v / 1e8, 2) for v in vals]
        if vals[0] is not None:
            return name, rounded          # 最新报告期有值 → 直接采用
        if fallback is None:
            fallback = (name, rounded)    # 最新期断档但其余期够用 → 备选
    return fallback or (None, None)


_QUARTER_ENDS = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}


def _quarter_of(pk: str):
    """报告期键 → 季度序号（1~4）；不是季末返回 None"""
    for q, end in _QUARTER_ENDS.items():
        if pk[4:8] == end:
            return q
    return None


def _prev_quarter_key(pk: str) -> str:
    """上一个季末的报告期键（20260630 → 20260331；Q1 回退到上年 1231）"""
    q = _quarter_of(pk)
    if q is None:
        return ""
    if q == 1:
        return f"{int(pk[:4]) - 1}1231"
    return f"{pk[:4]}{_QUARTER_ENDS[q - 1]}"


def _single_quarter(rows: list, candidates: tuple, n: int):
    """把**累计口径**序列转成**单季**序列，返回 (科目名, [单季亿元...])。

    为什么必须给这一列（2026-10-08 实测 603127）：
      财报接口返回的是**累计数**（年初至今）。那一份报告把同一组累计数按"单季"解读，
      推出「单季净利率 106%」「环比 +122.8%」「利润是 2025 全年 2 倍」——
      三个结论全部错位，且让整场多空辩论的主战场建立在一个错误前提上。
      只在提示词里写"注意是累计口径"不够（模型会漏读），**派生列要直接算好给它**。

    规则：单季 = 本期累计 − 上一季累计；Q1 的累计数本身就是单季。
    上一季取**原始行**（不受 n 限制），所以 n=4 时也能算出 Q3/Q4 的单季。
    跨年不回退（Q1 用自身），因此不存在"减出上一个年度 Q4"的语义错误。
    """
    by_pk = dict(rows)
    col, vals = _fin_series(rows, candidates, n)
    if not col:
        return None, None
    out_vals = []
    for pk, v in zip([r[0] for r in rows[:n]], vals):
        q = _quarter_of(pk)
        if v is None or q is None:
            out_vals.append(None)
            continue
        if q == 1:
            out_vals.append(v)                       # Q1 累计即单季
            continue
        prev_row = by_pk.get(_prev_quarter_key(pk))
        pv = _num(prev_row.get(col)) if prev_row else None
        if pv is None:
            out_vals.append(None)                    # 缺上一季 → 如实留空，不猜
            continue
        out_vals.append(round(v - round(pv / 1e8, 2), 2))
    return col, out_vals



@_tool
def tool_financials(symbol: str, periods: int = 4) -> dict:
    """个股财务报表：利润表/资产负债表/现金流量表的关键科目（单位亿元）+ 同比增速 + 自算比率。

    periods 为返回期数（1~8，默认 4）。**字段顺序即重要性顺序**：工具结果传给模型前
    有长度上限，超出会从尾部截断，所以把最该看的放前面。
    比率由三表自算（同日可比）；不良率/净息差等银行专项指标**不在三表内**，见 `三表外的关键指标`。
    """
    from data.stock_data import StockDataFetcher
    symbol = str(symbol).strip()
    if not (symbol.isdigit() and len(symbol) == 6):
        return {"error": "股票代码须为6位数字"}
    n = max(1, min(int(periods or 4), 8))
    fin = StockDataFetcher().get_financial_data(symbol) or {}
    if fin.get("error"):
        return {"error": f"财务数据不可用: {fin['error']}"}

    keys = ("income_statement", "balance_sheet", "cash_flow")
    rows_by: dict = {k: _fin_rows(fin, k) for k in keys}
    if not rows_by["income_statement"] and not rows_by["balance_sheet"]:
        return {"error": "未取到财务报表（新浪/同花顺均不可用）"}

    # 以"有数据的那张表"的期为基准，避免只取到部分报表时报告期错位
    base = max((rows_by[k] for k in keys), key=len)
    periods_all = [pk for pk, _ in base]
    fmt = lambda pk: f"{pk[:4]}-{pk[4:6]}-{pk[6:]}"       # noqa: E731

    # ⚠️ 不能用 `fin.get("balance_sheet", [{}])[0]`：`.get` 的默认值只在**键不存在**时生效，
    # 而 `_get_chinese_financial_data` 取不到时会显式存 None（键存在、值为 None），
    # 那样会 `None[0]` 抛 TypeError，整个工具返回 error。
    # 来源要**三张表一起看**：兜底是按表各自发生的，只读资产负债表会漏报。
    sources = {r.get("_source") for k in keys for _, r in rows_by[k] if r.get("_source")}
    source = "；".join(sorted(sources)) if sources else None
    out = {"代码": symbol, "数据截止": fmt(periods_all[0]),
           "报告期": [fmt(pk) for pk in periods_all[:n]],
           "单位": "亿元",
           # ⚠️ 口径必须显式声明。财报接口返回的是**累计数**（年初至今），
           # 不声明就会发生 2026-10-08 那份 603127 报告的事：
           # 模型把累计当单季，推出「单季净利率106%」「环比+122.8%」等一串错结论。
           "口径": ("累计（年初至今）：同一自然年内相邻报告期相减才是单季；"
                    "跨报告期不可算环比，同比须与去年同报告期比"),
           "数据来源": source or "新浪财报三表（stock_financial_report_sina）"}

    # ── 三表关键科目 ───────────────────────────────────────────────
    missing_labels: list = []
    for label, key in (("利润表", "income_statement"),
                       ("资产负债表", "balance_sheet")):
        block = {}
        for disp, cands in _FIN_FIELDS[label]:
            col, vals = _fin_series(rows_by[key], cands, n)
            if col:
                block[disp] = vals
            else:
                missing_labels.append(disp)
            # 上限 = `_FIN_FIELDS[label]` 的字段数（利润表 8 / 资产表 8），
            # 即"有多少给多少"。原先是 5 —— 直接导致两处后果：
            #   ① 提示词要求分析"资本开支/在建工程、应收与存货变化"，而
            #      在建工程/应收账款在资产表第 6/7 位，**永远到不了模型手上**；
            #   ② 模型据此报"关键科目缺失"，而数据其实就在库里。
            if len(block) >= len(_FIN_FIELDS[label]):
                break
        if block:
            out[f"{label}(亿元)"] = block

    # ── 同比增速（与去年同报告期比，累计口径可比；财报为累计数，勿用环比）──
    def _val_of(r: dict, cands: tuple):
        for c in cands:
            v = _num(r.get(c))
            if v is not None:
                return v
        return None

    def _yoy(cands: tuple):
        """返回 (同比%, 基准报告期键, 基准值/亿元)；算不出返回 None。

        ⚠️ **必须把基准值一起给出去**。原来只给一个百分比，
        模型无法核对它从哪来的：实测（2026-10-08 修复后重跑 603127）
        基本面报告写到「2025H1 为 0.65 亿元，由 2025-12-31 累计 2.98 −
        2025-09-30 累计 0.81 = 2.17 亿元？矛盾」—— 它看不到 2025-06-30
        （periods=4 的窗口里没有），只能自己倒推，推出矛盾后还专门写了一段
        "存在严重数据校验问题"。工具算出的数字**必须自带可核对的依据**。
        """
        rows = rows_by["income_statement"] or rows_by["balance_sheet"]
        if not rows:
            return None
        pk0, r0 = rows[0]
        cur = _val_of(r0, cands)
        prev_pk, prev = None, None
        for pk, r in rows[1:]:
            if pk[4:] == pk0[4:] and pk < pk0:      # 同月日、更早一年
                prev_pk, prev = pk, _val_of(r, cands)
                break
        if cur and prev:
            return round((cur / prev - 1) * 100, 2), prev_pk, round(prev / 1e8, 2)
        return None

    yoy, yoy_base = {}, {}
    for disp, cands in (("营业收入", ("营业收入", "营业总收入")),
                        ("净利润", ("净利润",)),
                        ("归母净利润", ("归属于母公司股东的净利润",
                                    "归属于母公司的净利润",
                                    "归属于母公司所有者的净利润"))):
        r = _yoy(cands)
        if r is not None:
            pct, base_pk, base_val = r
            yoy[disp] = pct
            yoy_base[disp] = f"{fmt(base_pk)} = {base_val}"
    if yoy:
        out["同比%"] = yoy
    if yoy_base:
        out["同比基准(亿元)"] = yoy_base

    # ── 利润表单季（累计 → 单季）──
    # 放在"同比"之前、紧挨利润表，便于模型两边对照；键名带"单季"以示区别。
    sq_col, sq_vals = _single_quarter(rows_by["income_statement"],
                                      ("营业收入", "营业总收入"), n)
    if sq_col and any(v is not None for v in sq_vals):
        out["利润表单季(亿元)"] = {"营业收入": sq_vals}
        out["利润表单季说明"] = (
            f"上一行是按【{sq_col}】自算的**单季**（本期累计 − 上一季累计）；"
            f"与上面【利润表(亿元)】的**累计**值不可混用，也不可直接对累计值算环比")

    # ── 自算比率（只用三表数据，不依赖被墙的东财指标接口）────────────
    def _latest(key: str, cands: tuple):
        rows = rows_by.get(key) or []
        return _val_of(rows[0][1], cands) if rows else None

    assets = _latest("balance_sheet", ("资产总计", "资产"))
    liab = _latest("balance_sheet", ("负债合计", "负债"))
    equity = _latest("balance_sheet", _FIN_EQUITY_CANDIDATES)
    rev = _latest("income_statement", ("营业收入", "营业总收入"))
    npf = _latest("income_statement", ("净利润",))
    loans = _latest("balance_sheet", ("发放贷款及垫款",))
    prov = _latest("balance_sheet", ("减:贷款损失准备", "贷款损失准备"))
    ratios = {}
    if assets and liab:
        ratios["资产负债率%"] = round(liab / assets * 100, 2)
    if rev and npf:
        ratios["净利率%"] = round(npf / rev * 100, 2)
    if equity and npf:
        ratios["ROE%(报告期累计,未年化)"] = round(npf / equity * 100, 2)
    if loans and prov:
        ratios["贷款损失准备/贷款%"] = round(prov / loans * 100, 2)
    if ratios:
        out["自算比率"] = ratios

    # ── 现金流量表放后面（重要性低于上面几项，被截断也不心疼）─────────
    cf = {}
    for disp, cands in _FIN_FIELDS["现金流量表"]:
        col, vals = _fin_series(rows_by["cash_flow"], cands, n)
        if col:
            cf[disp] = vals
        if len(cf) >= 3:
            break
    if cf:
        out["现金流量表(亿元)"] = cf

    if missing_labels:
        core_missing = [x for x in missing_labels if x in _FIN_CORE_LABELS]
        if core_missing:
            out["未取到核心科目"] = core_missing[:6]
    # 是否为银行/券商类：判据用**利润表的净利息收入**。
    # 不要用"有没有发放贷款及垫款"——实测 600519（茅台）也有小额该科目
    # （其财务公司放款 4.16 亿），会把它误判成银行，进而凭空声明一堆"取不到"。
    net_interest = (out.get("利润表(亿元)") or {}).get("净利息收入") or []
    if any(isinstance(v, (int, float)) and v > 0 for v in net_interest):
        out["三表外的关键指标"] = list(_FIN_NOT_IN_STATEMENTS)
        # 必须提示口径，否则模型会按工商企业的直觉把银行的高杠杆读成"高风险"。
        # 实测（2026-09-25）：助手看到 91.62% 就写"资产负债率偏高，需关注其风险控制能力"——
        # 对银行而言这是存款形成的负债，是**正常业务模式**，不是风险信号。
        out["银行口径提示"] = ("银行以客户存款为主要负债，资产负债率通常 90% 上下属正常业务模式，"
                          "不可与工商企业横向比较；其净利率亦偏高（利息支出计入营业成本）。"
                          "评价银行应看 ROE、不良率、净息差、拨备，而非资产负债率")
        out["说明"] = ("该标的为银行/券商类科目结构，上述指标（尤其不良贷款率、净息差、"
                      "拨备覆盖率）不在财务报表三表中，需专项披露，本工具取不到；"
                      "回答时不得据此推测")
    return out


