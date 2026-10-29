"""报告可核实性审计测试（core/report_audit.py）

背景（2026-09-28）：深度分析的三份报告在输入为空时照样输出极专业的内容——
技术面编出「分析日期：2024年X月X日」的模板头，基本面编出「上交所 13 份定期报告」
「Wind 一致预期(2024.06.30 校验)」，舆情在 0 条新闻下编出「交叉核查沪深交易所/
巨潮资讯网」「Wind 统计 3 家券商点评」。审计要把这些线索标出来。
"""

from core.report_audit import (audit_citations, audit_numbers,
                                      audit_report, audit_reports)


def test_empty_report_is_clean():
    assert audit_report("", "任意输入") == []
    assert audit_report(None, "任意输入") == []


def test_year_absent_from_input_is_flagged():
    warns = audit_report("截至2024年6月28日收盘后，公司无新增公告。", allowed_text="2026-09-25")
    assert any("2024" in w for w in warns), warns


# ---- 年份识别的误报防线（2026-09-28 首次实跑技术面报告时真的误报过）----

def test_decimal_fraction_is_not_read_as_a_year():
    """`0.2095` / `-1.2095` 的尾数是小数，不是年份。

    首次实跑技术面报告时报了「提及年份「2095」」，就是被这类小数的尾数触发的。
    加上前导小数点排除后必须不再误报。
    """
    report = ("数据截止：2026-09-24｜分析日期：2026-09-28\n"
              "换手率 0.2095，MACD 柱 -1.2095，20 日年化波动率 0.2031。")
    # ⚠️ 关掉数值检查：本用例是**钉年份正则**的，夹具故意只给日期，
    #    于是报告里的指标值（0.2095 等）在 allowed 里查不到 —— 那是数值规则
    #    的范畴（真实链路上这些指标一定在 allowed 里），会盖住本用例要验的东西。
    warns = audit_report(report, allowed_text="2026-09-24 2026-09-28",
                         check_numbers=False)
    assert warns == [], warns


def test_quantity_number_is_not_read_as_a_year():
    """`2095万元` / `2000点` 这类数量不是年份"""
    report = "成交额 2095万元，指数收于 2000点，换手 2085万股。"
    # 同上：只验年份规则，故关掉数值检查
    warns = audit_report(report, allowed_text="2026-09-28", check_numbers=False)
    assert warns == [], warns


def test_dotted_date_still_detected():
    """收紧正则不能把 `2019.08.30` 这种写法一起放过"""
    warns = audit_report("报告期截至 2019.08.30。", allowed_text="2026-09-28")
    assert any("2019" in w for w in warns), warns


def test_year_warning_carries_context_for_triage():
    """告警必须带原文片段：否则"提及年份「X」"无法定位，只能靠重跑复现（而模型输出随机）。"""
    warns = audit_report("据上交所披露，公司2021年报已于当年发布。", allowed_text="2026-09-28")
    hit = [w for w in warns if "2021" in w]
    assert hit, warns
    assert "上交所" in hit[0], hit[0]


def test_year_present_in_input_is_not_flagged():
    """财报里给了 2024 报告期 → 报告提到 2024 是合法的，不能误报"""
    allowed = "报告期（新→旧）：2025-06-30 / 2024-06-30，同比 11.1"
    # 只验"年份在输入里出现就不该报"；9.0 不在这个最小夹具里属数值规则范畴
    assert audit_report("2024年同期营业收入为 9.0 亿元", allowed_text=allowed,
                        check_numbers=False) == []


def test_external_source_and_institution_are_flagged():
    warns = audit_report("据 Wind 统计近3个月有3家券商点评，其中中信证券维持增持。",
                         allowed_text="现价 36.48 MA20 36.69")
    joined = " ".join(warns)
    assert "Wind" in joined
    assert "中信证券" in joined


def test_source_mentioned_in_input_is_allowed():
    """新闻里本来就引用了来源（如财联社）→ 报告转述不算幻觉"""
    warns = audit_report("财联社报道公司获得批文。", allowed_text="(cls | 财联社) 公司获得批文")
    assert warns == []


def test_disclosure_terms_flagged_only_without_financials_or_news():
    report = "公司2023年度报告显示营收增长，季报趋势向好。"
    no_data = audit_report(report, allowed_text="现价 36.48", has_financials=False, has_news=False)
    assert any("披露材料" in w for w in no_data), no_data

    has_fin = audit_report(report, allowed_text="现价 36.48，2023年度报告：营收 24.97 亿",
                           has_financials=True)
    assert not any("披露材料" in w for w in has_fin), has_fin


def test_warnings_are_capped():
    """一次罗列几十条会让 errors 变成噪音，必须封顶"""
    report = " ".join(f"{y}年" for y in range(1990, 1999)) + " Wind 巨潮 中信证券"
    warns = audit_report(report, allowed_text="2026")
    assert 0 < len(warns) <= 6


def test_audit_reports_prefixes_label():
    out = audit_reports({"技术面": "分析日期：2024年X月X日"},
                        allowed_text="2026-09-25")
    assert out and out[0].startswith("[技术面] 可核实性:")


def test_clean_report_produces_no_warning():
    report = ("数据截止：2026-09-24｜分析日期：2026-09-25\n"
              "现价 36.48 元，MA20 为 36.69，RSI 46.51，处于中性区间。")
    allowed = "2026-09-24 2026-09-25 36.48 36.69 46.51 ma20 rsi"
    assert audit_report(report, allowed_text=allowed) == []


# ================================================================
# 数值可追溯（v4，2026-10-08）
# ================================================================
#
# 背景：前三类规则（年份/来源/披露材料）抓不到最严重的编造 ——
# 实测 603127 报告里凭空出现的"市值 588 亿""股本 11.3 亿股""波动率 38.2%"
# "换手率 0.87%""卖一档挂单不足 800 手""过去 N 年出现 4 次平均涨幅 +18.3%"，
# 它们既不是年份也不是来源名，是**具体数字**。

def test_fabricated_numbers_are_flagged():
    """凭空出现的具体数字必须被标记（这是最影响结论的一类编造）"""
    allowed = "现价 52.06 MA5 47.602 资产总计 108.26 营业收入 7.04"
    report = ("当前股价52.06元对应市值约588亿元（按验算股本11.3亿股）。"
              "60日年化波动率38.2%。卖一档挂单量不足800手。"
              "历史回测显示后续20日平均涨幅+18.3%。")
    warns = audit_numbers(report, allowed, max_warn=8)
    joined = " ".join(warns)
    assert "588亿" in joined, warns
    assert "11.3亿" in joined, warns
    assert "800手" in joined, warns

def test_numbers_present_in_input_are_not_flagged():
    """输入里有的数字（含四舍五入后的写法）不得误报"""
    allowed = "现价 52.06 MA20 44.9565 MA60 46.68 量比 1.74 资产总计 108.26"
    report = "现价52.06元，MA20 为 44.96，MA60 46.68，量比约 1.74 倍，资产总计 108.3 亿元。"
    assert audit_numbers(report, allowed) == []

def test_derived_numbers_are_not_flagged():
    """模型合法地做一步四则运算（差额/比率）不算编造

    例：MACD 柱 = 0.618 − (−0.178) = 0.796；净利率 = 7.48 / 7.04 = 106.21
    """
    allowed = "macd 0.618 macd_signal -0.178 净利润 7.48 营业收入 7.04"
    report = "MACD 柱为 0.796，净利率 106.21%。"
    assert audit_numbers(report, allowed) == []

def test_small_unitless_integers_are_ignored():
    """1~30 的无单位整数是行文用词（"3-5 个理由""分 2 轮"），追它们只会淹没真信号"""
    report = "给出3个理由，分2轮辩论，共5条证据。"
    assert audit_numbers(report, allowed_text="") == []

def test_number_warning_carries_context():
    """告警要带原文片段，否则"数字 X 无出处"无法定位"""
    warns = audit_numbers("公司市值约588亿元。", "现价 52.06", max_warn=3)
    assert warns and "588" in warns[0] and "市值" in warns[0], warns

def test_numeric_check_respects_priority_over_years():
    """数值告警必须能进最终结果 —— 不能被年份把配额吃光

    首版实现是"每类往同一个 list 追加、满 6 条就 return"，
    实测年份先占满了配额，数值型编造一条都出不来（规则等于没接）。
    """
    report = ("2021年、2022年、2023年、2024年均有类似结构，"
              "市值约588亿元，股本11.3亿股。")
    warns = audit_report(report, allowed_text="2026-10-08")
    joined = " ".join(warns)
    assert "588亿" in joined, warns

def test_check_numbers_can_be_disabled():
    """需要只验其他规则时可以关掉数值检查"""
    assert audit_report("市值约588亿元。", "现价 52.06", check_numbers=False) == []


# ================================================================
# 伪造引用（v5）：声称"输入中列出…"但内容是假的
# ================================================================
#
# 比"凭空出现一个数字"更恶劣：把编造包装成**引用**，读者会以为"输入里就是这么写的"。
# 实测（2026-10-08 修复后重跑 603127）：空头为驳斥多头写下
#   「但输入中：\n - 仅列出"MA5=48.12, MA10=46.25, MA20=44.01, MA60=41.89"」
# 真值是 47.602 / 46.852 / 44.9565 / 46.6823 —— 一个都不对。
# 普通数值规则抓不到：48.12 能被输入里两个数凑出来（52.06 − 3.88 ≈ 48.18，落在容差内）。

def test_fabricated_citation_is_flagged():
    allowed = "现价 52.06 MA5 47.602 MA10 46.852 MA20 44.9565 MA60 46.6823 营收 3.88"
    report = ('多头称"站稳全部四条均线"，但输入中：\n'
              '   - 仅列出"MA5=48.12, MA10=46.25, MA20=44.01, MA60=41.89"，'
              '未提供 MA60 的计算起始日；')
    warns = audit_citations(report, allowed, max_warn=6)
    joined = " ".join(warns)
    assert "48.12" in joined and "41.89" in joined, warns

def test_citation_rule_survives_arithmetic_traceability():
    """这是它存在的理由：48.12 能被 allowed 里的数字凑出来，普通数值规则会放行"""
    allowed = "现价 52.06 营收 3.88"
    report = '输入中仅列出"MA5=48.12"，与多头所称不符。'
    assert audit_numbers(report, allowed) == [], "前提：普通数值规则确实抓不到它"
    assert audit_citations(report, allowed), "伪造引用规则必须抓到"

def test_genuine_citation_is_not_flagged():
    """真的引用（数字逐字出现在输入里）不得误报"""
    allowed = "现价 52.06 MA60 46.6823 MA5 47.602"
    report = '技术面报告原文明确声明现价 52.06、MA60 46.6823。'
    assert audit_citations(report, allowed) == []

def test_citation_claim_may_span_lines():
    """⚠️ 实测报告里是「输入中：\\n - 仅列出…」—— 正则不吃换行时规则会静默失效"""
    allowed = "现价 52.06"
    report = '但输入中：\n   - 仅列出"MA5=48.12"'
    assert audit_citations(report, allowed), "跨行的引用声明必须能匹配"

def test_citation_ignores_rhetorical_counts():
    """"输入中给出 3 条线索"这类无单位小整数不算被引用的数字"""
    assert audit_citations("输入中给出 3 条线索", "现价 52.06") == []


# ================================================================
# v5 误报防线（都是重跑 603127 时真实踩到的）
# ================================================================

def test_indicator_glued_to_degree_is_not_a_year():
    """`MA2045°斜率` 不是年份 2045 —— 是 "MA20" 与 "45°" 粘连。

    重跑 603127 时研究汇总真报过一次「提及年份「2045」」。
    指标名+数字粘连在技术面报告里很常见，前导排字母后不再误报。
    """
    assert audit_report("四均线多头排列+MA2045°斜率放量", "2026-10-08") == []

def test_arithmetic_involving_percent_times_weight_is_not_flagged():
    """风控自己算的 "3.05% × 15%仓位 = 0.4575%" 不是编造

    首版白名单没有 `a*b/100`，把 0.4575% 当成了"找不到出处的数字"。
    """
    report = "理论亏损为 3.05% × 15%仓位 = 0.4575% 组合净值回撤"
    assert audit_numbers(report, "止损 3.05% 仓位 15%") == []

def test_generic_above_given_phrase_is_not_a_citation_claim():
    """「上面给出的数字」是正常推理用语，不该触发引用检查

    首版把「给出」「显示」这类泛化动词也当成引用声明，误报了风控的心算结果。
    """
    report = "按上面给出的止损位，理论亏损为 3.05% × 15% = 0.4575%。"
    assert audit_citations(report, "止损 3.05% 仓位 15%") == []

def test_real_year_fabrication_still_caught():
    """收紧误报之后，真正的编造年份必须仍然抓得到（否则是"修误报修掉了真报"）"""
    warns = audit_report("参考2021年CXO板块多次K>90后继续上行", "2026-10-08")
    assert any("2021" in w for w in warns), warns


# ================================================================
# 依据类编造（v6，2026-10-09）
# ================================================================
# 实测缺口：旧的三张词表（数据商 / 券商 / 披露材料）**一个都抓不到"制度类"编造**。
# 一份 603127 报告里凭空出现了《证券公司风险控制指标管理办法》《权益类交易风控指引V3.2》
# 《单一股票持仓限额制度》第4.1条、《波动率超额仓位扣减规则》、《企业会计准则第14号》第25条、
# CDE《2026生物医药研发白皮书》P23、"内部审计简报[16]"、"销售部门周报"、
# "天眼查/启信宝穿透显示"、"蒙特卡洛模拟 10000 次路径"、"证监会《会计监管风险提示第11号》"
# —— 它们既不是年份、也不是数据商名、也不是披露材料，于是全部通过，
# 最终把一个合规的交易计划判成"超限 3.5 倍、强制否决"。
#
# ⚠️ 这类编造比"多写一个数字"危害更大：它给结论安了一个**不存在的依据**。

def test_fabricated_document_name_is_flagged():
    """书名号里的文件/制度名，输入里没有就是编造 —— 一条规则覆盖大多数制度类幻觉"""
    report = ("依据《证券公司风险控制指标管理办法》，并参照《权益类交易风控指引V3.2》，"
              "单票静态仓位上限为 8%。")
    warns = audit_report(report, allowed_text="现价 52.06 计划仓位 15%")
    joined = " ".join(warns)
    assert "《证券公司风险控制指标管理办法》" in joined, warns
    assert "《权益类交易风控指引V3.2》" in joined, warns


def test_document_name_present_in_input_is_not_flagged():
    """上游真的提到过某文件（如新闻正文）时不得误报"""
    allowed = "新闻[3]：公司发布《2026年限制性股票激励计划（草案）》的公告"
    report = "舆情提到公司发布了《2026年限制性股票激励计划（草案）》。"
    assert audit_report(report, allowed_text=allowed) == []


def test_fabricated_article_and_doc_number_are_flagged():
    """条款号与监管文号：编造合规依据时几乎必然出现"""
    warns = audit_report(
        "按《单一股票持仓限额制度》第4.1条，并依据苏药监受理〔2026〕198号，"
        "实际动态仓位上限应下调至 5.3%。", "现价 52.06")
    joined = " ".join(warns)
    assert "第4.1条" in joined, warns
    assert "〔2026〕198号" in joined, warns


def test_article_number_in_input_is_not_flagged():
    assert audit_report("按第3.2.1条执行。", "条款 第3.2.1条 已收录") == []


def test_authority_term_not_in_input_is_flagged():
    """制度类名词（蒙特卡洛 / 附注 / 内部简报…）也必须在输入里查得到"""
    warns = audit_report("经蒙特卡洛模拟 10000 次路径，6 个月内跌破概率 68.3%；"
                         "见 2026 半年报附注五（22）。", "现价 52.06 营收 7.04")
    joined = " ".join(warns)
    assert "蒙特卡洛" in joined, warns


def test_authority_term_in_input_is_not_flagged():
    allowed = "公告[11]：发布 H 股 2026 中期报告；新闻：公司披露计提减值附注"
    assert audit_report("公告提到中期报告的附注内容。", allowed_text=allowed) == []


def test_third_party_business_data_is_external_source():
    """天眼查 / 启信宝 / CVSource 属第三方数据源，本项目通路里没有"""
    warns = audit_report("启信宝穿透显示平均成立 3.2 年、无产品上市。", "现价 52.06")
    assert any("启信宝" in w for w in warns), warns


def test_negated_source_mention_is_not_flagged():
    """⚠️ 实测误报：提示词里的 EVIDENCE_CONSTRAINT 列举了 Wind/同花顺/Choice/iFinD，
    模型一旦在正文复述"本项目禁止引用 Wind 等"，5 条来源告警全亮 —— 会淹没真信号。"""
    report = ("本项目禁止引用 Wind / 万得 / 同花顺 / Choice / iFinD 等外部终端数据，"
              "本报告未使用任何研报。")
    assert audit_report(report, allowed_text="现价 52.06") == []


def test_source_mentioned_positively_is_still_flagged():
    """收紧误报之后，真引用必须仍然抓得到（修误报不能把真报一起修掉）"""
    warns = audit_report("据 Wind 统计，近 3 个月仅 3 家券商覆盖。", "现价 52.06")
    assert any("Wind" in w for w in warns), warns


def test_negation_does_not_shadow_a_later_real_mention():
    """同一来源既被声明"不使用"、又被真正引用时，仍须报出来"""
    report = "本报告未使用 Wind 数据。但 Wind 显示其一致预期为 45x。"
    warns = audit_report(report, allowed_text="现价 52.06")
    assert any("Wind" in w for w in warns), warns


def test_disclosure_citation_checked_even_with_financials():
    """⚠️ 具体引用物（研报/公告编号/招股说明书）**任何节点都要查**。

    旧版把"报告期泛称"和"具体引用物"放在同一个 `has_financials` 闸门后面，
    而下游节点（辩论/汇总/风控/终审）一律传 has_financials=True
    ⇒ 这条规则对它们**永久失效**，"公告[15]附件…"就是这样通过的。
    """
    warns = audit_report("据研报与调研纪要，公司订单饱满。", "现价 52.06",
                         has_financials=True, has_news=True)
    assert any("研报" in w for w in warns), warns


def test_disclosure_periods_still_gated_by_financials():
    """报告期泛称反过来**不能**误报：拿到财报后模型正常写"半年报"是合法的

    （allowed 里只有 `2025-06-30` 这种日期，没有"半年报"这三个字。）
    """
    assert audit_report("根据半年报，营业收入 7.04 亿。", "2025-06-30 营业收入 7.04",
                        has_financials=True) == []


def test_authorities_outrank_years_in_warning_budget():
    """依据类告警必须能进最终结果 —— 不能被年份把 6 条配额吃光

    ⚠️ 与数值规则同一类风险（首版就是被年份吃光配额而静默失效）。
    """
    report = ("2021年、2022年、2023年均有类似结构，依据《证券公司风险控制指标管理办法》，"
              "仓位上限为 3.3%。")
    warns = audit_report(report, allowed_text="2026-10-08 现价 52.06")
    assert any("《证券公司风险控制指标管理办法》" in w for w in warns), warns


# ================================================================
# 「第N条」的歧义消解（2026-10-09 实测误报）
# ================================================================
# 新代码第一次端到端重跑就踩到：模型写「基本面报告**第1条**中确凿列出」
# 「**第4条**载明…」——指的是**输入报告里的第 N 个条目**，不是法规条款。
# 首版直接按"第X条"判成编造依据，一次报了 3 条假告警，
# 而这恰恰是"修误报时别把真报一起修掉"的另一面：**新规则也会自己引入误报**。

def test_report_item_number_is_not_a_legal_article():
    report = ("→ 归母权益 89.75 亿元在基本面报告第1条中确凿列出；"
              "第4条载明“2026-06-30 累计营收 7.04 亿元”。")
    warns = audit_report(report, allowed_text="现价 52.06")
    assert not any("条款号" in w for w in warns), warns


def test_article_with_document_context_is_flagged():
    """真条款号必须仍然抓得到（附近有文档类词，或本来就是多层编号）"""
    warns = audit_report("按《单一股票持仓限额制度》第4.1条，上限为 8%。", "现价 52.06")
    assert any("第4.1条" in w for w in warns), warns
    warns2 = audit_report("按第3.2.1条执行。", "现价 52.06")
    assert any("第3.2.1条" in w for w in warns2), warns2


def test_bare_single_level_article_is_not_flagged():
    """孤零零的「第1条」太弱（口语里到处是）→ 不报，避免把告警配额浪费在噪声上"""
    assert audit_report("请见第1条。", allowed_text="现价 52.06") == []


def test_authority_term_inside_doc_name_is_not_double_reported():
    """《企业会计准则第30号》已经由书名号规则报了 → 里面的"准则"不该再报一遍

    同一处被两条规则重复计数只会浪费 6 条告警配额。
    """
    warns = audit_report("依据《企业会计准则第30号——财务报表列报》处理。", "现价 52.06")
    assert any("《企业会计准则第30号——财务报表列报》" in w for w in warns), warns
    assert not any("依据「准则」" in w for w in warns), warns


def test_authority_term_outside_doc_name_still_flagged():
    """书名号之外的制度类名词仍要报（不能因为去重把真报也去掉）"""
    warns = audit_report("经蒙特卡洛模拟 10000 次路径测算。", "现价 52.06")
    assert any("蒙特卡洛" in w for w in warns), warns


def test_generic_accounting_language_is_not_an_authority_claim():
    """⚠️ 实测（2026-10-09 端到端重跑）："会计准则下归母净利润的法定构成"
    "会计准则允许高毛利模式" 这类**通用表述**不是编造依据。

    首版把"准则"收进 AUTHORITY_TERMS，一次报了 2 条噪声。
    真正有害的是**引用具体标准号**（《企业会计准则第30号》），由书名号规则覆盖。
    """
    report = "这混淆了会计准则下“归母净利润”的法定构成，也非审计结论。"
    assert audit_report(report, allowed_text="现价 52.06 归母净利润 7.48") == []


def test_specific_standard_number_is_still_flagged():
    """去噪不能去掉真报：引用具体标准号仍要报"""
    warns = audit_report("根据《企业会计准则第30号——财务报表列报》，归母净利润本就包含…",
                         allowed_text="现价 52.06")
    assert any("第30号" in w for w in warns), warns


# ---- 算式自洽（2026-10-23，v8）----
# 背景：全量回扫 65 份历史报告，4 条"括号里写了算式、括号外结果算错"，全部由
# qwen3.7-flash 撰写。旧的「数值可追溯」对这四类**完全静默**——两个被减数都能在
# 输入里逐字找到，错的是它们相减的结果。

from core.report_audit import audit_arithmetic, audit_band_position  # noqa: E402


def test_arithmetic_catches_real_2026_10_23_errors():
    """三条来自真实报告的原句，必须全部报出。"""
    cases = [
        ("Q2单季净利润296.56亿元（790.1 - 293.54），环比下滑。", "496.56"),
        ("Q3单季经营净额1,104.36亿元（1,852.9 - 1,148.54）。", "704.36"),
        ("净利率为28.44%（47.97/177.47），盈利能力极强。", "27.03"),
    ]
    for text, expect in cases:
        warns = audit_arithmetic(text)
        assert warns, f"未报出：{text}"
        assert expect in warns[0], warns


def test_arithmetic_allows_correct_subtraction():
    assert audit_arithmetic("Q3单季营收2,422.6亿元（7,802.95 - 5,380.35）。") == []


def test_arithmetic_ignores_base_value_lists():
    """括号里是并列基准值，不是除法（实测误报 3 次）。"""
    text = "归母净利润同比+4.04%（基准为2025-06-30的1,333.68/464.55/460.16亿元）。"
    assert audit_arithmetic(text) == []


def test_arithmetic_ignores_date_ranges():
    assert audit_arithmetic("近12个月涨幅+82.3%（2023.10–2024.10），属高波动标的。") == []


def test_arithmetic_allows_explicit_derivation():
    """模型把推导写出来并说明口径差异 ⇒ 是诚实，不是算错。"""
    text = "净利率23.79%（归母净利润/营业收入=1.02/5.35≈19.1%，但自算比率给出23.79%）。"
    assert audit_arithmetic(text) == []


def test_arithmetic_keeps_negative_result():
    """`-1.07亿元（0.18 - 1.25）` 是**算对**的，负号在括号外。"""
    assert audit_arithmetic("Q2 2026归母净利润：-1.07亿元（0.18 - 1.25）。") == []


def test_arithmetic_empty_report():
    assert audit_arithmetic("") == []
    assert audit_arithmetic(None) == []


# ---- 布林带方位断言（v8）----

_IND = {"price": 99.14, "bb_upper": 103.49, "bb_middle": 97.66, "bb_lower": 91.83}


def test_band_position_catches_self_contradiction():
    """600941 原句：99.14 明明高于下轨 91.83，却写「下轨（91.83）之下」。"""
    text = "收盘价 99.14 位于布林带中轨（97.66）之上，下轨（91.83）之下，上轨（103.49）之下。"
    warns = audit_band_position(text, _IND)
    assert warns and "下轨" in warns[0], warns


def test_band_position_passes_correct_statement():
    text = "收盘价 99.14 位于布林带中轨（97.66）之上，上轨（103.49）之下。"
    assert audit_band_position(text, _IND) == []


def test_band_position_ignores_fuzzy_wording():
    """「为更下方的极端支撑参考」不是方位断言 —— 这是刻意不收的一族。"""
    assert audit_band_position("下轨 91.3102 为更下方的极端支撑参考。", _IND) == []
    assert audit_band_position("下轨（91.31）为下方强支撑参考。", _IND) == []


def test_band_position_skips_when_value_mismatches():
    """报告写的轨道值与真值不符 ⇒ 这条断言不可判，不报。"""
    assert audit_band_position("下轨（80.00）之下。", _IND) == []


def test_band_position_requires_indicators():
    assert audit_band_position("下轨（91.83）之下。", None) == []
