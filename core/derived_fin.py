# -*- coding: utf-8 -*-
"""财务派生指标表：把「模型最容易自己算错」的指标**由程序算好**再喂给模型。

动机（2026-10-23 实测，见 `output/qwen3.7错误谱系与模型重评估_2026-10-23.md`）：
同一个模型在**隔离算术题**上 8/8 全对，却在长报告里把

    `296.56亿元（790.1 - 293.54）`   应得 496.56   —— 错 40%
    `28.44%（47.97/177.47）`         应得 27.03%   —— 错 1.4 个百分点，并被下游
                                                    推成「26 倍 PE 合理偏低」

算错。差距来自**任务形状**，不是能力：算术正确率随"同时要维持的数值上下文"增多而下降。

所以修法不是换模型，而是**别让它算**：

  · 程序算好 → 表里给 `值` + 可逐字复制的 `算式` + 用的是哪几个数
  · 模型只做「引用 + 解释」，不做「推导」
  · 缺科目 / 缺上一季 ⇒ **一律不产出**，绝不填 0、绝不用相邻期顶替

⚠️ 算式必须只用**表内已展示的两位小数**拼接。模型抄的是这些字面量，
   `report_audit.audit_arithmetic` 核对的也是它们；用 4 位精度会造成
   「程序对、模型抄完反被审计判错」的假告警。

⚠️ 本模块**不取数**：只接受 `core.agent_tools_fin.tool_financials` 的输出，
   所以它可以在任何有 `fin` dict 的地方复用（含离线测试）。
"""
import logging

_logger = logging.getLogger(__name__)

_QUARTER_ENDS = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}

# 单季派生：累计 → 单季。
# ⚠️ 覆盖必须**成片**。首版只给利润表三项 + 经营现金流，把投资/筹资净额留空，
#    实测模型立刻自己外推那两项，并在正文里留下
#    「Q3单季流出107.45亿元（算式：-107.45 - (-645.9) ? 此处需注意：…）」
#    「*再次核对*」「*修正*」这类**草稿痕迹** —— 部分覆盖会诱使它按同一套路补全。
#    现金流三表净额本来就在原输入里，补齐是零成本，所以一并派生。
_SQ_ITEMS = (
    ("净利润", "利润表"),
    ("归母净利润", "利润表"),
    ("营业利润", "利润表"),
    ("经营活动净额", "现金流量表"),
    ("投资活动净额", "现金流量表"),
    ("筹资活动净额", "现金流量表"),
)

# 比率派生：(展示名, 分子块, 分子键, 分母块, 分母键, 单位, 备注)
# ⚠️ ROE 的分子必须与分母**同口径**（都用"归母"）。`tool_financials` 的
#    `自算比率` 里那条 `ROE%` 是「净利润 ÷ 归母权益」，分子含少数股东损益 ——
#    两者数值会略有差异，所以这里明确标出口径，避免模型把两条当成同一个数。
_RATIO_ITEMS = (
    ("净利率", "利润表", "净利润", "利润表", "营业收入", "%", "报告期累计，未年化"),
    ("归母净利率", "利润表", "归母净利润", "利润表", "营业收入", "%", "报告期累计，未年化"),
    ("营业利润率", "利润表", "营业利润", "利润表", "营业收入", "%", "报告期累计，未年化"),
    ("资产负债率", "资产负债表", "负债合计", "资产负债表", "资产总计", "%", "时点数"),
    ("ROE(归母净利/归母权益)", "利润表", "归母净利润", "资产负债表", "归母权益", "%",
     "报告期累计，未年化；与自算比率中的 ROE% 分子口径不同"),
    ("ROA", "利润表", "净利润", "资产负债表", "资产总计", "%", "报告期累计，未年化"),
    ("研发费用率", "利润表", "研发费用", "利润表", "营业收入", "%", "报告期累计"),
    ("经营现金流/净利润", "现金流量表", "经营活动净额", "利润表", "净利润", "倍", "报告期累计"),
)

# 同比派生：(展示名, 块, 键)。只在"自算结果与工具给的同比一致"时才带算式，
# 否则说明源口径不同（如扣非/调整后），给算式反成误导。
_YOY_ITEMS = (("营业收入", "利润表", "营业收入"),
              ("净利润", "利润表", "净利润"),
              ("归母净利润", "利润表", "归母净利润"))

_MAX_ITEMS = 26


def _pk8(value) -> str:
    """报告期归一成 8 位 YYYYMMDD。

    ⚠️ `tool_financials` 输出的 `报告期` 是**带横线**的（`2026-09-30`），
    而季度判断与"上一季"拼装都按 8 位做。首版直接拿 `pk[4:8]` 去比 `"0930"`，
    得到的是 `"-09-"` ⇒ **一个单季都算不出来**，且静默返回空表。
    所有跨期比较前必须先过这里。
    """
    return str(value or "").replace("-", "").replace("/", "").strip()


def _disp(pk8: str) -> str:
    """8 位键 → 展示用的 `YYYY-MM-DD`（不是 8 位就原样返回）。"""
    s = str(pk8 or "")
    return f"{s[:4]}-{s[4:6]}-{s[6:]}" if len(s) == 8 and s.isdigit() else s


def _q(pk: str):
    """报告期键 → 季度序号（1~4）；不是季末返回 None。"""
    for q, end in _QUARTER_ENDS.items():
        if str(pk)[4:8] == end:
            return q
    return None


def _prev_q(pk: str) -> str:
    """上一个季末的报告期键（20260630 → 20260331；Q1 回退到上年 1231）。"""
    q = _q(pk)
    if q is None:
        return ""
    if q == 1:
        return f"{int(str(pk)[:4]) - 1}1231"
    return f"{str(pk)[:4]}{_QUARTER_ENDS[q - 1]}"


def _num(v):
    """数值化；None/NaN/非数值返回 None（**绝不返 0**）。"""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return None if v != v else float(v)
    return None


def _block(fin: dict, name: str) -> dict:
    b = fin.get(f"{name}(亿元)")
    return b if isinstance(b, dict) else {}


def _series(fin: dict, block: str, key: str):
    """返回 (报告期列表[8 位], 值列表)，两者均已按「新→旧」对齐。

    NaN 与占位符一律折成 None（`tool_financials` 的序列里可能有断档），
    绝不折成 0 —— 0 会被下游当成真实数值继续参与比值计算。
    """
    periods = [p for p in (_pk8(x) for x in (fin.get("报告期") or [])) if len(p) == 8]
    vals = _block(fin, block).get(key)
    if not isinstance(vals, list) or not periods:
        return [], []
    n = min(len(periods), len(vals))
    clean = [None if (not isinstance(x, (int, float)) or x != x) else round(float(x), 2)
             for x in vals[:n]]
    return periods[:n], clean


def _lit(v: float) -> str:
    """算式里的数字字面量；负数加括号。

    不加括号会写出 `-895.05 - -645.90` 这种句子：既是病句，也绕开了
    `audit_arithmetic` 的 `a - b` 正则（`b` 不认负号）—— 于是"抄了也对不上"。
    写成 `(-895.05) - (-645.90)` 后，该正则因内层 `(` 而不匹配，走安全分支。
    """
    return f"({v:.2f})" if v < 0 else f"{v:.2f}"


def _item(name: str, value: float, unit: str, formula: str | None,
          basis: str, note: str = "") -> dict:
    return {"name": name, "value": round(value, 2), "unit": unit,
            "formula": formula, "basis": _disp(basis), "note": note}


def _single_quarter_items(fin: dict, out: list) -> None:
    """累计 → 单季。**上一季必须真的出现在窗口里**，否则跳过（不猜）。

    每个科目最多给**最近两期**：最新一期是模型必用的，前一期供它写"环比/连续两季"
    这类对比；再往前只会把表撑长、把最新项挤到截断之外。
    """
    for key, block in _SQ_ITEMS:
        periods, vals = _series(fin, block, key)
        if not periods:
            continue
        by_pk = dict(zip(periods, vals))
        emitted = 0
        for i, pk in enumerate(periods):
            cur = vals[i]
            if cur is None:
                continue
            q = _q(pk)
            if q is None:
                continue
            if q == 1:
                # Q1 的累计本身就是单季 —— 这是**定义**，不是减法，故不给算式
                out.append(_item(f"{key}(单季, {_disp(pk)})", cur, "亿元", None, pk,
                                 "Q1 累计即单季"))
                emitted += 1
            else:
                prev = by_pk.get(_prev_q(pk))
                if prev is None:
                    continue
                out.append(_item(f"{key}(单季, {_disp(pk)})", cur - prev, "亿元",
                                 f"{_lit(cur)} - {_lit(prev)}", pk,
                                 "单季 = 本期累计 − 上一季累计"))
                emitted += 1
            if emitted >= 2:
                break


def _ratio_items(fin: dict, out: list) -> None:
    periods, _ = _series(fin, "利润表", "营业收入")
    if not periods:
        periods, _ = _series(fin, "资产负债表", "资产总计")
    pk = periods[0] if periods else ""
    for name, nb, nk, db, dk, unit, note in _RATIO_ITEMS:
        _, nvals = _series(fin, nb, nk)
        _, dvals = _series(fin, db, dk)
        if not nvals or not dvals:
            continue
        a, b = nvals[0], dvals[0]
        if a is None or b is None or abs(b) < 1e-12:
            continue
        if unit == "%":
            out.append(_item(f"{name}({_disp(pk)})", a / b * 100, "%",
                             f"{_lit(a)} / {_lit(b)} × 100", pk, note))
        else:
            out.append(_item(f"{name}({_disp(pk)})", a / b, "倍",
                             f"{_lit(a)} / {_lit(b)}", pk, note))


def _diff_items(fin: dict, out: list) -> None:
    """净资产 = 资产总计 − 负债合计（模型常写"净资产 X 亿"却不给依据）。"""
    _, av = _series(fin, "资产负债表", "资产总计")
    _, lv = _series(fin, "资产负债表", "负债合计")
    periods, _ = _series(fin, "资产负债表", "资产总计")
    pk = periods[0] if periods else ""
    if av and lv and av[0] is not None and lv[0] is not None:
        out.append(_item(f"净资产({_disp(pk)})", av[0] - lv[0], "亿元",
                         f"{_lit(av[0])} - {_lit(lv[0])}", pk, "时点数"))


def _yoy_items(fin: dict, out: list) -> None:
    """同比算式版。**只在自算值与工具给的同比一致时**才附算式。"""
    given = fin.get("同比%") or {}
    if not isinstance(given, dict):
        return
    for disp, block, key in _YOY_ITEMS:
        want = _num(given.get(disp))
        if want is None:
            continue
        periods, vals = _series(fin, block, key)
        if not periods or vals[0] is None:
            continue
        pk0 = periods[0]
        base = None
        for i, pk in enumerate(periods[1:], start=1):
            if pk[4:] == pk0[4:] and pk < pk0:
                base = vals[i]
                break
        if base is None or abs(base) < 1e-12:
            # 值本身是有效输入（源自带同比），只是窗口里凑不出去年同报告期 ⇒
            # 仍然要给出，只是**不附算式**。首版这里直接丢弃，导致"同比"整类消失，
            # 而同比恰恰是报告里被引用最多的一类数字。
            out.append(_item(f"{disp}同比", want, "%", None, pk0,
                             "源给出；报告期窗口内无去年同报告期，无法附算式"))
            continue
        calc = round((vals[0] / base - 1) * 100, 2)
        if abs(calc - want) > 0.1:      # 口径不同（扣非/追溯调整）⇒ 不给算式
            out.append(_item(f"{disp}同比", want, "%", None, pk0,
                             "源给出；与按累计直算的结果不同，勿自行改写"))
            continue
        # ⚠️ 同比算式**不能出现 `/` 与裸 `=`**，也不能写成 `（a - b）`：
        #  `audit_arithmetic` 的两条正则分别把 `x%（a / b）` 判成 a/b*100、
        #  把 `x单位（a - b）` 判成 a−b，而它对"括号内后续还有内容"是看不见的
        #  （`_ar_skippable` 拿的是整段匹配，截到 b 就结束）。
        #  用 `÷` + 内层括号可以同时避开两条正则；数学上仍逐字可复现：
        #  (790.10 − 700.00) ÷ 700.00 × 100 = 12.87。
        out.append(_item(f"{disp}同比", want, "%",
                         f"({_lit(vals[0])} - {_lit(base)}) ÷ {_lit(base)} × 100", pk0,
                         "同比 = (本期累计 − 去年同报告期) ÷ 去年同报告期"))


def build(fin: dict, stock_info: dict = None, *, limit: int = _MAX_ITEMS) -> list:
    """由 `tool_financials` 的输出构造派生指标列表（算不出的一律不产出）。

    Args:
        fin: `core.agent_tools_fin.tool_financials` 的返回值
        stock_info: 预留（当前不派生估值项：市值÷净利润 ≠ PE(TTM)，派生它只会
            造出"看起来像 PE 的错数"，已改为在提示词里禁止自推 PE）
        limit: 条数上限（默认 22，控制提示词增量）

    Returns:
        [{"name", "value", "unit", "formula", "basis", "note"}, ...]；无可用数据返回 []。
    """
    if not isinstance(fin, dict) or fin.get("error"):
        return []
    if not fin.get("报告期"):
        return []
    out: list = []
    try:
        # 顺序 = 截断时的保留优先级：单季 → 比率 → 同比 → 差值。
        # 同比必须排在"净资产"之前：同比几乎每份报告都会引用，净资产只是偶尔提到。
        _single_quarter_items(fin, out)
        _ratio_items(fin, out)
        _yoy_items(fin, out)
        _diff_items(fin, out)
    except Exception as e:                       # noqa: BLE001 - 派生失败只降级，不阻断分析
        _logger.warning("[派生指标] 计算失败，按无派生表处理: %s", e)
        return []
    # 去重（同名同报告期只留第一条），并保留插入顺序（单季 → 比率 → 差值 → 同比）
    seen, uniq = set(), []
    for it in out:
        k = (it["name"], it["basis"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(it)
    return uniq[:max(1, int(limit or _MAX_ITEMS))]


def render(items: list) -> str:
    """渲染成给模型的紧凑文本；空列表返回空串（调用方据此不加整段）。"""
    if not items:
        return ""
    lines = ["【派生指标表（由程序按上面的科目算好，可直接引用）】"]
    for it in items:
        unit = it.get("unit") or ""
        # ⚠️ 不能用 `:g`：它只保留 6 位有效数字，`114609.40` 会被写成 `114609`，
        #    而模型抄走后审计按字面量核 ⇒ 程序给了错数、模型承担误报。
        val = f"{it['value']:.2f}".rstrip("0").rstrip(".") or "0"
        # `%` 紧贴数值（"10%" 而不是 "10 %"）：模型会照抄这个写法，
        # 而 `_AR_DIV` 的 `claim%` 分支要求百分号**紧跟**在数字后面。
        line = (f"- {it['name']} = {val}{unit}" if unit == "%"
                else f"- {it['name']} = {val} {unit}".rstrip())
        if it.get("formula"):
            line += f" ｜ 算式 = {it['formula']}"
        if it.get("note"):
            line += f" ｜ {it['note']}"
        lines.append(line)
    return "\n".join(lines)


# 引用规则：**必须跟在派生表后面**。第 2/3 条对应两次实测事故：
#   ① 300760 报告自己算出「PE 26 倍」并据此判"估值合理偏低"，而输入的 PE 是另一个数；
#   ② 600941 报告在表外自己减累计数，错 40%。
DERIVED_USAGE_RULES = """【派生指标使用规则（硬约束）】
1. 上面【派生指标表】里的每一项都已由程序算好：**直接引用它的"值"**；
   需要写出推导过程时，**逐字复制该行的"算式"**，不要自己重算、不要改动小数位。
2. 表里没有的比率或增速，若你决定自己算，必须在同一句的括号内写出完整算式，
   且算式里的每个数字都必须能在【估值指标】【财务报表】【派生指标表】中逐字找到。
3. PE / PB / PS **只能引用【估值指标】里给出的值**；不得用「市值 ÷ 净利润」等
   方式自行推算 PE，也不得把"报告期累计净利润"当作 TTM 净利润使用。
4. 某个指标在表里没有出现，说明**输入不足以计算**（缺科目或缺上一季），
   此时写「数据缺失」，不要用相邻报告期顶替、也不要假设为零。
5. 正文只写结论与依据，**不要写推导草稿**：不得出现「再次核对」「修正」「此处需注意」
   「?」这类边算边疑的痕迹。算不出来就不写，或写「数据缺失」。
"""
