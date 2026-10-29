"""报告可核实性审计（反幻觉闸门）

背景（2026-09-28 定位）：深度分析的三份报告在**输入为空**的情况下照样输出极专业的
内容——
  - 技术面：输入里没有任何日期，报告头部却写「数据截止：最新交易日收盘后｜分析日期：2024年X月X日」
    （连占位符都没填，是模型从训练语料里抄的模板头）；
  - 基本面：调用方显式传 `financial_data=None`，报告却写「基于上交所 2021年报~2024年一季报
    共13份定期报告」「Wind 一致预期（2024.06.30 校验）」；
  - 舆情：实际新闻 0 条，报告却写「经交叉核查沪深交易所公告平台及巨潮资讯网」「据 Wind 统计
    近3个月仅3家券商点评（中信证券 2024-05-08）」。

本模块只做一件事：把报告里**无法在上游输入中定位的外部事实**挑出来，交给调用方记进
`errors` / 在 UI 上高亮。**刻意不修改报告正文**——改写会掩盖问题，标记才会暴露问题；
而且自动删改会引入新的不可核实内容（"我们删了什么"本身也不可见）。

判据刻意保守（宁可少报，不可误伤）：只查"一眼假"的痕迹——未提供的年份、
未出现过的数据商/券商/工商数据源名、输入里根本没有的披露材料、
**依据类编造**（文档名《…》/ 条款号 / 监管文号 / 制度类名词）、
以及**数值不可追溯**（含"伪造引用"这一最隐蔽的子类）。
"""

import logging
import re

# 数据商 / 行情终端：出现在报告里就说明模型在引用输入之外的信息源
EXTERNAL_SOURCES = (
    "Wind", "万得", "巨潮", "Choice", "iFinD", "同花顺", "东方财富",
    "Bloomberg", "彭博", "路透", "Reuters",
)

# 券商 / 机构：本项目输入里**从不包含**研报，出现即为编造
INSTITUTIONS = (
    "中信证券", "中金公司", "华泰证券", "国泰君安", "招商证券", "广发证券",
    "海通证券", "申万宏源", "国盛证券", "平安证券", "天风证券", "东吴证券",
    "兴业证券", "浙商证券", "民生证券", "东方证券", "光大证券", "方正证券",
    "华创证券", "国信证券",
)

# 披露类材料，分两组（判据不同，见各自注释）
#
# ① 报告期泛称：只有输入确实带了财报/公告/新闻时才允许出现。
#    这类词**本身不指向具体文件**（"半年报""季报"），模型在拿到财报时完全可能
#    正常地写出来，而 allowed 里只有"2025-06-30"这种日期，查词表会**误报** ——
#    所以保留"本次有没有给财报/新闻"这个闸门。
DISCLOSURE_PERIODS = (
    "定期报告", "年度报告", "年报", "一季报", "半年报", "二季报", "三季报",
    "季报", "业绩预告",
)
# ② 具体引用物：**任何节点都要查**（原先被 `has_financials` 一并跳过 → 对下游永久失效）。
#    它们指向的是"某个具体文件/某个外部结论"，输入里没有就是编造，与本次是否给了财报无关。
DISCLOSURE_CITATIONS = (
    "公告编号", "招股说明书", "一致预期", "研报", "调研纪要", "电话会议",
)

# ── 依据类名词（v6，2026-10-09）────────────────────────────────────────
# 实测缺口：旧三张词表（数据商/券商/披露材料）**一个都抓不到**"制度类"编造 ——
# 一份报告里凭空出现了《证券公司风险控制指标管理办法》《权益类交易风控指引V3.2》
# 《单一股票持仓限额制度》第4.1条、《波动率超额仓位扣减规则》、《企业会计准则第14号》第25条、
# CDE《2026生物医药研发白皮书》P23、"内部审计简报[16]"、"销售部门周报"、
# "天眼查/启信宝穿透显示"、"蒙特卡洛模拟 10000 次路径"、"证监会《会计监管风险提示第11号》"
# —— 它们既不是年份、也不是数据商名、也不是披露材料，于是**全部通过**，
# 最终把一个合规的交易计划判成"超限 3.5 倍、强制否决"。
#
# 判据与来源名一致：**必须在 allowed（上游真实输入）里查得到**。
# 只收"文档/制度形状"的词，不收"依据""办法"这类日常词（"没有办法""解决办法"会误报）。
# ⚠️ **不收"准则"**（2026-10-09 实测）：它太常出现在通用表述里
# （"会计准则下归母净利润的法定构成""会计准则允许高毛利模式"），
# 这类话不是"编造依据"。**真正有害的形式是引用了具体标准号**（《企业会计准则第30号》），
# 那由 `_DOC_NAME_RE` 的书名号规则覆盖，不需要靠这个词兜。
AUTHORITY_TERMS = (
    "指引", "指导原则", "征求意见稿", "白皮书", "蓝皮书",
    "简报", "附注", "附件", "风控手册", "内控编号",
    "蒙特卡洛", "压力测试", "监管风险提示", "审计意见",
    "公示数据库", "穿透式",
)

# 第三方工商/商业数据源：本项目的数据通路里没有它们（出现过一次是**新闻正文**里
# 被提及，那种情况 allowed 里会有这个词，不会误报）。
THIRD_PARTY_DATA = (
    "天眼查", "启信宝", "企查查", "CVSource", "投中", "药智", "米内网", "Wind资讯",
)

# 《…》里的文档名：**任何**在正文里用书名号标出的文件，都必须在输入里出现过。
# 这是最省事也最精确的一条 —— 本项目自己的报告很少用书名号，
# 而编造制度时几乎必然写成《XX办法》《XX指引》。
_DOC_NAME_RE = re.compile(r"《([^》\n]{2,40})》")
# 条款号：第4.1条 / 第 3.2.1 条
_ARTICLE_RE = re.compile(r"第\s*\d+(?:\.\d+)*\s*条")
# 监管文号：〔2026〕198号 / 【2026】198号
_DOC_NO_RE = re.compile(r"[〔【\[]\s*(?:19|20)\d{2}\s*[〕】\]]\s*\d+\s*号")

# 「第N条」的歧义消解（2026-10-09 实测踩到）：
# 模型写「基本面报告**第1条**中确凿列出」「**第4条**载明…」，指的是**输入报告里的第 N 个条目**，
# 不是法规条款 —— 首版直接按"第X条"判成编造依据，一次报了 3 条假告警。
# 消解规则（两条，任一满足才报）：
#   ① 前文是"报告/原文/输入/列表…"→ 明确是**引用报告的条目**，不报；
#   ② 要么是**多层编号**（第4.2条 —— 口语里几乎不会这么指代条目），
#      要么附近 30 字内出现了文档类词（办法/准则/指引/制度/规则…）。
_REPORT_REF_WORDS = ("报告", "原文", "上文", "前文", "输入", "列表", "表格",
                     "条目", "清单", "摘要", "正文", "附注表")
_DOC_CONTEXT_WORDS = ("办法", "准则", "指引", "条例", "制度", "规则", "规范",
                      "预案", "实施细则", "的规定")

# 否定语境：报告中**主动声明"不引用/未使用"某来源**时，不该被当成引用它。
# 实测误报（2026-10-08）：把风控提示词自身复述出来就会触发 ——
#   提示词里的 `EVIDENCE_CONSTRAINT` 列举了 Wind/万得/同花顺/Choice/iFinD，
#   模型一旦在正文里写"本项目禁止引用 Wind 等外部终端"，5 条来源告警全亮。
_NEG_MARKERS = ("禁止", "不得", "不应", "不可", "不能", "未使用", "未引用", "未出现",
                "未提供", "未采用", "不使用", "没有", "从未", "无需")
# 分句边界：逗号/顿号**不算**（否定句里的列举正是被它们分隔的）
_CLAUSE_BOUNDARY = "。；！？\n\r"

# 年份识别：`(19|20)` 开头的 4 位数字。
# 两处排除都是为了压误报，各自都有实测依据：
#   - `(?<![\d.])`：**前导不能是小数点**。2026-09-28 首次实跑技术面报告时就误报过一次
#     「提及年份「2095」」——那是从某个小数的尾数里切出来的（如 `0.2095`），不是年份。
#     年份几乎不会写成 `.2026`，所以这个排除没有代价。
#   - `(?![\d万亿手股点倍])`：**后面不能跟数量单位**。`2095万元`、`2085万手`、`2000点`
#     都是数量，不是年份；而真正的年份后面跟的是「年 / - / . / 空格」，不受影响。
#   - `(?<![A-Za-z])`（2026-10-08 补）：**前导不能是字母**。实测重跑 603127 时误报过
#     「提及年份「2045」」——原文是「四均线多头排列+MA2045°斜率+10.8日放量突破」，
#     即 "MA20" 与 "45°" 粘连成了 "MA2045"。指标名+数字粘连在技术面报告里很常见。
_YEAR_RE = re.compile(r"(?<![\d.A-Za-z])((?:19|20)\d{2})(?![\d万亿手股点倍])")

MAX_WARNINGS = 6

# 告警里带上原文片段：没有上下文的话，"提及年份「2095」"这种误报根本无法定位，
# 只能靠重跑复现（模型输出还是随机的）。
_CONTEXT_CHARS = 14


_logger = logging.getLogger(__name__)


def _years_in(text: str) -> list[tuple[str, str]]:
    """按出现顺序去重，返回 (年份, 原文片段) 列表；片段用于人工复核。"""
    seen, out = set(), []
    for m in _YEAR_RE.finditer(text or ""):
        year = m.group(1)
        if year in seen:
            continue
        seen.add(year)
        start = max(0, m.start() - _CONTEXT_CHARS)
        end = min(len(text), m.end() + _CONTEXT_CHARS)
        snippet = text[start:end].replace("\n", " ").strip()
        out.append((year, snippet))
    return out


def _in_negation(text: str, pos: int, lookback: int = 120) -> bool:
    """`pos` 是否处于否定语境的**同一分句**里（"禁止引用 Wind / 同花顺"里那两个词都不算引用）。

    ⚠️ 判据是"**分句内**有没有否定词"，不是"前面 N 个字符内有没有"：
    否定句常写成**列举**（"禁止引用 Wind、万得、同花顺、Choice、iFinD 等外部终端"），
    固定字符窗只覆盖得到前两个词，后面的照旧被误报（实测首版就是这样，
    "Choice""iFinD" 仍被判为"引用了输入中不存在的来源"）。
    分句边界取 `。；！？` 与换行 —— 逗号、顿号不算，因为它们正在列举之中。
    """
    seg = text[max(0, pos - lookback):pos]
    cuts = [seg.rfind(ch) for ch in _CLAUSE_BOUNDARY]
    cut = max(cuts) if cuts else -1
    if cut >= 0:
        seg = seg[cut + 1:]
    return any(mk in seg for mk in _NEG_MARKERS)


def _token_is_fabricated(text: str, token: str) -> bool:
    """token 是否**至少有一次**出现在非否定语境里。

    只看"词在不在正文里"会把"本报告未使用 Wind 数据"也算成编造 ——
    必须逐个出现位置判断，全部处于否定语境才算"没引用"。
    """
    start = 0
    while True:
        i = text.find(token, start)
        if i < 0:
            return False
        if not _in_negation(text, i):
            return True
        start = i + len(token)


def _doc_name_spans(text: str) -> list[tuple[int, int]]:
    """《…》 的字符区间（用于避免同一处被两条规则重复报告）"""
    return [(m.start(), m.end()) for m in _DOC_NAME_RE.finditer(text or "")]


def _inside_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)


def audit_authorities(report: str, allowed_text: str = "", *,
                      max_warn: int = 4) -> list[str]:
    """检查「依据类」编造：文档名 / 条款号 / 监管文号 / 制度类名词。

    这一类编造危害最大：它不是"多写了个数字"，而是**给结论安了一个不存在的依据**，
    让一个本可核实的判断（仓位 8% vs 上限 25%）看起来像有权威背书。
    """
    report = str(report or "")
    allowed = str(allowed_text or "")
    out: list[str] = []
    doc_spans = _doc_name_spans(report)

    for m in _DOC_NAME_RE.finditer(report):
        name = m.group(1).strip()
        # 书名号里 2~3 个字多半是强调用法（《多头逻辑》），4 字以上才是文件/制度名
        if len(name) < 4 or name in allowed or _in_negation(report, m.start()):
            continue
        out.append(f"引用了输入中不存在的文件/制度《{name}》")
        if len(out) >= max_warn:
            return out

    for m in _ARTICLE_RE.finditer(report):
        raw = m.group(0)
        if raw.replace(" ", "") in allowed.replace(" ", ""):
            continue
        if _in_negation(report, m.start()):
            continue
        # 「…报告第1条」= 引用输入的条目，不是法规条款（见 _REPORT_REF_WORDS 说明）
        before = report[max(0, m.start() - 10):m.start()]
        if any(w in before for w in _REPORT_REF_WORDS):
            continue
        # 单层数字（第1条）没有文档上下文时太弱 → 不报；多层（第4.2条）或附近有文档类词才报
        near = report[max(0, m.start() - 30):m.start()]
        if "." not in raw and not any(w in near for w in _DOC_CONTEXT_WORDS):
            continue
        out.append(f"引用了输入中不存在的条款号「{raw}」")
        if len(out) >= max_warn:
            return out

    for m in _DOC_NO_RE.finditer(report):
        raw = m.group(0)
        if raw.replace(" ", "") in allowed.replace(" ", "") or _in_negation(report, m.start()):
            continue
        out.append(f"引用了输入中不存在的监管文号「{raw}」")
        if len(out) >= max_warn:
            return out

    for term in AUTHORITY_TERMS:
        if term in allowed:
            continue
        # 逐出现位置判断：落在《…》里的（如《企业会计准则第30号》里的"准则"）跳过 ——
        # 同一处被两条规则重复计数只会浪费告警配额。
        start, flagged = 0, False
        while not flagged:
            i = report.find(term, start)
            if i < 0:
                break
            if not _inside_spans(i, doc_spans) and not _in_negation(report, i):
                flagged = True
            start = i + len(term)
        if not flagged:
            continue
        out.append(f"引用了输入中不存在的依据「{term}」")
        if len(out) >= max_warn:
            return out

    return out


def audit_report(report: str, allowed_text: str = "", *,
                 has_financials: bool = False, has_news: bool = False,
                 check_numbers: bool = True) -> list[str]:
    """扫描报告里无法在输入中定位的外部事实。

    Args:
        report: 模型产出的报告正文
        allowed_text: 上游真实输入的可搜索文本（行情/指标/K线摘要/财报/新闻/日期）。
            报告里出现的东西只要在这里能找到，就不算幻觉。
        has_financials: 本次是否真的提供了财务报表
        has_news: 本次是否真的提供了新闻/公告
        check_numbers: 是否做「数值可追溯」检查（见 `audit_numbers`）。
            对辩论/汇总/风控类报告建议开启；对纯定性报告可关。

    Returns:
        告警列表（人类可读，可直接塞进 state["errors"]）；无问题返回空列表。
    """
    report = str(report or "")
    allowed = str(allowed_text or "")
    if not report.strip():
        return []

    # 五类线索**各自收集、最后按优先级合并取前 MAX_WARNINGS 条**。
    # ⚠️ 不能像首版那样"每类往同一个 list 里追加、满了就 return"：
    #    年份与来源名很容易先把 6 条配额吃满（实测 603127 报告每段都有 2~3 个
    #    不存在的年份），于是**最该看的数值型编造一条都出不来**。
    # 优先级：伪造引用（最欺骗）> 来源名 / 依据类（最确定的造假）> 数值 > 年份 > 报告期泛称。
    citations: list[str] = []
    nums: list[str] = []
    if check_numbers:
        citations = audit_citations(report, allowed, max_warn=MAX_WARNINGS)
        nums = audit_numbers(report, allowed, max_warn=MAX_WARNINGS)

    sources: list[str] = []
    for token in EXTERNAL_SOURCES + THIRD_PARTY_DATA + INSTITUTIONS:
        # 逐出现位置判否定语境：见 `_token_is_fabricated` 的说明
        if token not in allowed and _token_is_fabricated(report, token):
            sources.append(f"引用了输入中不存在的来源「{token}」")

    authorities = audit_authorities(report, allowed, max_warn=MAX_WARNINGS)

    years: list[str] = []
    for year, snippet in _years_in(report):
        if year not in allowed:
            years.append(f"提及年份「{year}」，但输入数据中无此年份（原文：…{snippet}…）")

    # 报告期泛称：仅在**本次确实没给财报与新闻**时才是编造（见 DISCLOSURE_PERIODS 注释）。
    # 具体引用物（研报/公告编号/招股说明书…）**任何节点都查**——旧版把两者放在同一个
    # 闸门后面，导致下游节点（一律传 has_financials=True）整条规则永久失效。
    disclosure: list[str] = []
    if not (has_financials or has_news):
        for term in DISCLOSURE_PERIODS:
            if term in report and term not in allowed:
                disclosure.append(f"提及「{term}」等披露材料，但本次未提供财报与新闻数据")
    for term in DISCLOSURE_CITATIONS:
        if term in allowed or not _token_is_fabricated(report, term):
            continue
        disclosure.append(f"提及「{term}」，但输入中没有任何被引用物")

    merged = citations + sources + authorities + nums + years + disclosure
    return merged[:MAX_WARNINGS]


# 数值可追溯（v4，2026-10-08）
# --------------------------
# 前三类规则（年份 / 来源名 / 披露材料）抓不到最严重的一类编造：**凭空出现的具体数字**。
# 实测 603127 报告里被漏掉的有：市值 588 亿（真实 390 亿）、股本 11.3 亿股（真实 7.49）、
# 60 日年化波动率 38.2%（真实 77.0%）、换手率 0.87%、"卖一档挂单不足 800 手"
# （项目根本没有 L2 数据）、"过去 N 年出现 4 次、平均涨幅 +18.3%"、"回撤概率 76.3%"。
#
# 判据：报告里的数字必须能在 allowed（上游真实输入）里找到，或能由其中的数字
# 通过**简单四则运算**得到（模型合法地算个净利率、算个差额是正常的）。
#
# ⚠️ 这是**筛查工具，不是判定器**：
#   - 会有漏报（allowed 文本里数字很多，凑巧能"算出来"的编造值不会报）；
#   - 会有误报（模型做多步运算、或引用了输入里等值的另一种写法时会被标记）。
#   所以只报不改（与 `audit_report` 的一贯立场一致），由人/后续节点决定如何处理。
_NUM_RE = re.compile(
    # ⚠️ 前导断言只能用 [A-Za-z0-9._]，**不能用 \w**：Python 的正则里 `\w` 匹配
    #    Unicode 词字符，**包括中文**，于是"约588亿"这类写法（数字前面是汉字）
    #    会被整片挡掉 —— 实测首版就是这样，一个数字都没抽出来，规则静默失效。
    r"(?<![A-Za-z0-9._])"              # 前导不是字母/数字/点（避免从 0.2095 里切出 2095）
    r"(\d[\d,]*(?:\.\d+)?)"            # 数字（允许千分位）
    r"\s*(%|％|亿|万|倍|手|元|股|个点|个百分点|bp|次|天|日|周|月|年)?")
# "具体的数字"才值得追：带小数（38.2 / 0.87 / 11.3）或 ≥31 且带度量词的整数（588亿 / 800手）。
# 1~30 的无单位整数多为行文用词（"3-5 个理由""分 2 轮""近 20 日"），追它们只会淹没真信号。
_NUM_MIN_INT_WITH_UNIT = 31
_NUM_TOL = 0.02                        # 相对容差（吸收四舍五入与单位换写）


def _to_float(raw: str):
    try:
        return float(str(raw).replace(",", ""))
    except (TypeError, ValueError):
        _logger.debug("[审计] 数值/类型不符，按缺省处理")
        return None


def _num_values(text: str, cap: int = 4000) -> set:
    """文本里出现的全部数值（集合）。cap 防止超长文本把审计拖慢。"""
    out = set()
    for i, m in enumerate(_NUM_RE.finditer(text or "")):
        if i >= cap:
            break
        v = _to_float(m.group(1))
        if v is not None:
            out.add(v)
    return out


def _traceable(v: float, allowed_nums: set) -> bool:
    """v 能否由 allowed 里的数字"直接找到或一步算出"。"""
    def close(a: float, b: float) -> bool:
        return abs(a - b) <= max(0.01, _NUM_TOL * abs(b))

    for a in allowed_nums:
        if close(v, a) or close(v, a * 100) or close(v, a / 100):
            return True            # 等值，或百分数/小数的写法差异
    # 一步运算：只取量级较大的若干个数两两组合（防组合爆炸）。
    # 候选覆盖模型**合法**会做的几类算术，否则会把它们误报成编造：
    #   差额（累计→单季）、比率（净利率/资产负债率 ×100）、增速（(a-b)/b ×100）
    big = sorted(allowed_nums, key=abs, reverse=True)[:80]
    for i, a in enumerate(big):
        for b in big[i + 1:]:
            # 覆盖模型**合法**会做的几类算术（漏了就会把它们误报成编造）：
            #   差额（累计→单季）、比率（净利率/资产负债率 ×100）、
            #   增速((a-b)/b ×100)、以及**百分比乘权重**（"亏损 3.05% × 15%仓位
            #   = 0.4575%"——实测风控段就是这么算的，首版白名单里没有 a*b/100）
            cands = [a + b, abs(a - b), a * b, a * b / 100]
            if b:
                cands += [a / b, a / b * 100, (a - b) / b, (a - b) / b * 100]
            if a:
                cands += [b / a, b / a * 100, (b - a) / a * 100]
            if any(close(v, c) for c in cands):
                return True
    return False


# 「伪造引用」检查（v5，2026-10-08）
# ------------------------------
# 比"凭空出现一个数字"更恶劣的一种编造：**声称自己在引用输入**，但内容是假的。
# 实测（2026-10-08 修复后重跑 603127）：空头为驳斥多头，写下
#   「但输入中：仅列出"MA5=48.12, MA10=46.25, MA20=44.01, MA60=41.89"」
# 而输入里的真值是 47.602 / 46.852 / 44.9565 / 46.6823 —— 一个都不对。
# 这种写法极具欺骗性：读者会以为"输入里就是这么写的"。
#
# 为什么 `audit_numbers` 抓不到：48.12 恰好能被输入里的两个数凑出来
# （52.06 − 3.88 ≈ 48.18，落在 2% 容差内），于是被判为"可推导"。
# 所以这里用**更严的判据**：只要报告声称"输入中列出/明确声明…"，
# 紧随其后的数字就必须在输入里**逐字出现**，不做任何容差与算术豁免。
_CITATION_CLAIM = re.compile(
    # ⚠️ 间隔必须允许**换行**：实测报告里是
    #    「但输入中：\n   - 仅列出"MA5=48.12, …"」，
    #    首版用 `[^\n。；]{0,16}?`（不含换行）→ 规则一条都没命中，静默失效。
    #    这里用 `[\s\S]{0,24}?`（限长以免跨段乱匹配）。
    #
    # ⚠️ 措辞必须**收紧到枚举型**（仅列出/明确声明/原文为…）。首版把「给出」「显示」
    #    这类泛化动词也收进来了，而「上面给出的数字」是正常推理里的常用说法 ——
    #    实测重跑 603127 时误报了两条：风控写「理论亏损为 3.05% × 15%仓位 = 0.4575%」，
    #    3.05%/0.4575% 是**它自己算的**，不是引用输入。
    r"(?:输入中|输入里|数据中|原文|报告中)[\s\S]{0,24}?"
    r"(?:仅列出|仅给出|明确声明|原文为|仅有)")

# 算式里的数字不算"被引用的输入值"：紧邻算术运算符的跳过
# （运算符字面量：× * ÷ / + - = 与中英文括号）
_CITATION_WINDOW = 160          # 声明之后多大范围内算"被引用的数字"


def audit_citations(report: str, allowed_text: str = "", *, max_warn: int = 3) -> list[str]:
    """检查"声称引用输入"处的数字是否真的出现在输入里。

    只对**引用声明之后**的一小段窗口取数字，且要求**逐字命中**：
    这类断言的可信度完全建立在"输入里就是这么写的"上，容差会让它失去意义。
    """
    allowed = str(allowed_text or "")
    report = str(report or "")
    out: list[str] = []
    for m in _CITATION_CLAIM.finditer(report):
        window = report[m.end():m.end() + _CITATION_WINDOW]
        for nm in _NUM_RE.finditer(window):
            raw = nm.group(1)
            v = _to_float(raw)
            if v is None:
                continue
            # 与数值规则同一套"具体性"判据，避免把"3 条""5 个"之类算进来
            unit = nm.group(2) or ""
            if not (("." in raw) or (abs(v) >= _NUM_MIN_INT_WITH_UNIT and unit)):
                continue
            if raw in allowed:
                continue            # 逐字命中 → 确实是引用
            # ⚠️ 这里**不做**"算式豁免"。首版加过一条"紧邻算术运算符就跳过"，
            #    结果把主要用例也跳过了 —— "MA5=48.12" 里数字前面正好是 `=`，
            #    于是规则重新变哑。误报源头其实在**措辞**（"上面给出的数字"
            #    是正常推理用语），已从 claim 正则里剔除，无需再做数字豁免。
            snippet = window[max(0, nm.start() - 20):nm.end() + 12].replace("\n", " ")
            out.append(f"以「{m.group()[:12]}…」的口径引用了输入中**不存在**的数字"
                       f"「{raw}{unit}」（原文：…{snippet}…）")
            if len(out) >= max_warn:
                return out
    return out


def audit_numbers(report: str, allowed_text: str = "", *, max_warn: int = 5) -> list[str]:
    """报告里"找不到出处"的具体数字（数值可追溯检查）。

    排序：带度量词或百分号的优先、量级大的优先 —— 编造的"588 亿""38.2%"比
    "11.3"更值得先看，因为它们更影响结论。
    """
    allowed_nums = _num_values(allowed_text)
    flagged: list[tuple] = []
    seen: set = set()
    for m in _NUM_RE.finditer(report or ""):
        raw, unit = m.group(1), m.group(2) or ""
        if raw in seen:
            continue
        seen.add(raw)
        v = _to_float(raw)
        if v is None:
            continue
        specific = ("." in raw) or (abs(v) >= _NUM_MIN_INT_WITH_UNIT and bool(unit))
        if not specific or _traceable(v, allowed_nums):
            continue
        start, end = max(0, m.start() - 14), min(len(report), m.end() + 14)
        flagged.append((bool(unit), abs(v), raw, unit, report[start:end]))

    flagged.sort(key=lambda x: (not x[0], -x[1]))
    out = []
    for _has_unit, _mag, raw, unit, snippet in flagged[:max(0, max_warn)]:
        out.append(f"数字「{raw}{unit}」在输入中找不到出处（原文：…{snippet}…）")
    return out


def audit_reports(reports: dict, allowed_text: str = "", *,
                  has_financials: bool = False, has_news: bool = False) -> list[str]:
    """批量审计多份报告：{名称: 正文} → 带前缀的告警列表。"""
    out: list[str] = []
    for label, text in (reports or {}).items():
        for w in audit_report(text, allowed_text,
                              has_financials=has_financials, has_news=has_news):
            out.append(f"[{label}] 可核实性: {w}")
    return out


# 算式自洽（v8，2026-10-23）
# ------------------------------
# 定位：`audit_numbers` 只管"这个数在输入里有没有出处"，**管不了"这个数算得对不对"**。
# 模型自己写出 `（790.1 - 293.54）` 时，两个被减数都能在输入里逐字找到 ⇒ 旧三道审计
# 全部放行，但 790.1 - 293.54 = 496.56，报告写的是 296.56（差 40%）。
#
# 实测（2026-10-23 全量回扫 65 份历史报告）：65 份里可自核算式 55 条，
#   4 条算错，**全部由 qwen3.7-flash 撰写**：
#     · 600941 中国移动 2026-10-23：Q2 单季净利 296.56（实为 496.56）
#     · 600941 中国移动 2026-10-23：Q3 单季经营净额 1104.36（实为 704.36）
#     · 300760 迈瑞医疗 2026-10-22：净利率 28.44%（47.97/177.47 = 27.03%）
#       ⇒ 这句又喂给估值结论「净利率接近30% ⇒ 26倍PE合理偏低」，**错误会往下传**
#
# 判定只看**一个句子内部**：括号里写了算式、括号外写了结果 ⇒ 结果必须等于算式。
# 不推断、不补全，算式与结果不同句就不判（宁可漏报）。
#
# 【刻意不判的三种写法】（每条都有实测误报）
#   · 括号内是一组并列基准值：`归母净利同比+4.04%（基准为1,333.68/464.55/460.16亿元）`
#   · 括号内是日期：`涨+82.3%（2023.10–2024.10）`
#   · 括号内已含 `≈` / `=`：模型把推导过程写出来了，按"结果相等"判会误伤
#     （实测 002040：它自己写了 `1.02/5.35≈19.1%` 并说明与给定比率口径不同 —— 这是诚实）
_AR_NUM = r"\d[\d,]*(?:\.\d+)?"
_AR_UNIT = r"(?:亿元|亿|万元|万|元|%|％|倍|个百分点)?"
_AR_SUB = re.compile(
    rf"(?P<claim>{_AR_NUM})\s*{_AR_UNIT}\s*[（(]\s*"
    rf"(?P<a>{_AR_NUM})\s*[-−–]\s*(?P<b>{_AR_NUM})\s*[)）]")
_AR_DIV = re.compile(
    rf"(?P<claim>{_AR_NUM})\s*[%％]\s*[（(][^（()）]{{0,40}}?"
    rf"(?P<a>{_AR_NUM})\s*/\s*(?P<b>{_AR_NUM})")

_AR_MAX_WARN = 2


def _ar_float(s: str) -> float:
    return float(str(s).replace(",", ""))


def _ar_skippable(inner: str) -> bool:
    """括号内不是"一个算式"的三种情况。"""
    if len(re.findall(r"[-−–+/]", inner)) >= 2:      # 并列基准值
        return True
    if re.search(r"\d{4}[.\-/]\d{1,2}\b", inner):    # 日期分量
        return True
    return ("≈" in inner) or ("=" in inner)


def audit_arithmetic(report: str, *, max_warn: int = _AR_MAX_WARN) -> list[str]:
    """括号内写出了算式时，括号外的结果必须成立（算式自洽）。

    Args:
        report: 任意节点正文
        max_warn: 上限（默认 2，理由同 `audit_relations`：同一处错误常被反复引用）

    Returns:
        人类可读告警；无问题返回空列表。**只报不改**。
    """
    text = str(report or "")
    if not text.strip():
        return []
    out: list[str] = []
    seen: set = set()
    for pat in (_AR_SUB, _AR_DIV):
        for m in pat.finditer(text):
            inner = m.group(0)
            if _ar_skippable(inner):
                continue
            try:
                claim = _ar_float(m.group("claim"))
                a = _ar_float(m.group("a"))
                b = _ar_float(m.group("b"))
            except (TypeError, ValueError):
                _logger.debug("[审计] 数值/类型不符，按缺省处理")
                continue
            if pat is _AR_SUB:
                want = a - b
                # ⚠️ 负号在括号外（`-1.07亿元（0.18 - 1.25）`）：正则只捕到 1.07，
                #    必须按符号还原，否则把**算对的**判成算错。
                before = text[max(0, m.start("claim") - 1):m.start("claim")]
                got = -claim if before in "-−" else claim
            else:
                if abs(b) < 1e-12:
                    continue
                want = a / b * 100
                got = claim
                if abs(got - want) <= 0.15:          # 0.15 个百分点以内视为四舍五入
                    continue
                key = ("div", round(got, 2), round(want, 2))
                if key in seen:
                    continue
                seen.add(key)
                out.append(f"括号算式不成立：「{m.group('a')} / {m.group('b')}」"
                           f"应得 {want:.2f}%，报告写 {got:.2f}%")
                continue
            if abs(want) < 1e-12:
                continue
            if abs(got - want) <= max(0.02, abs(want) * 0.005):
                continue
            key = ("sub", round(got, 2), round(want, 2))
            if key in seen:
                continue
            seen.add(key)
            out.append(f"括号算式不成立：「{m.group('a')} - {m.group('b')}」"
                       f"应得 {want:.2f}，报告写 {got:.2f}")
            if len(out) >= max(0, max_warn):
                return out
    return out[:max(0, max_warn)]


def audit_band_position(report: str, indicators: dict = None, *,
                        max_warn: int = 1) -> list[str]:
    """布林带**方位断言**：括号里给了轨道数值、紧跟方位词时，方向必须成立。

    ⚠️ 为什么另开一个函数而不并进 `audit_relations`：那里**刻意不收**布林上/下轨
    （见上方【刻意不收的族】），理由是"股价接近上轨"这类描述关系的阈值是主观的。
    这里只收**可判定的写法** —— 括号给出该轨数值、紧跟"之上/之下/上方/下方"，
    此时既没有阈值、也没有解读空间。实测反例（600941 2026-10-23）：
        「收盘价 99.14 位于布林带中轨（97.66）之上，下轨（91.83）之下，上轨（103.49）之下」
    99.14 > 91.83，`下轨之下` 与句子自身矛盾。
    反向的**合法**写法必须继续放行：
        「下轨 91.3102 为更下方的极端支撑参考」—— 没有括号、不是方位断言。
        「若跌破下轨（91.31）则……」—— 条件句，`_REL_COND` 排除。
    """
    text = str(report or "")
    ind = indicators if isinstance(indicators, dict) else {}
    if not text.strip() or not ind:
        return []
    try:
        price = float(ind.get("price"))
    except (TypeError, ValueError):
        _logger.debug("[审计] 数值/类型不符，按缺省处理")
        return []
    truth = {"上轨": ind.get("bb_upper"), "中轨": ind.get("bb_middle"),
             "下轨": ind.get("bb_lower")}
    out: list[str] = []
    for m in re.finditer(r"(上轨|中轨|下轨)\s*[（(]\s*(" + _AR_NUM
                         + r")\s*[)）]\s*(之上|之下|上方|下方)", text):
        name, raw, word = m.group(1), m.group(2), m.group(3)
        val = truth.get(name)
        try:
            val = float(val)
            written = _ar_float(raw)
        except (TypeError, ValueError):
            _logger.debug("[审计] 数值/类型不符，按缺省处理")
            continue
        if abs(written - val) > max(0.02, abs(val) * 0.005):
            continue                      # 报告写的数与本轨真值不符 ⇒ 不是这条断言，不判
        said_above = word in ("之上", "上方")
        truth_above = price > val
        if said_above == truth_above:
            continue
        head = text[max(0, m.start() - 20):m.start()]
        if any(w in head for w in _REL_COND):
            continue
        snippet = text[max(0, m.start() - 18):m.end() + 10].replace("\n", " ").strip()
        out.append(f"「…{snippet}…」方位写反：现价 {price:g} "
                   f"{'高于' if truth_above else '低于'} {name} {val:g}")
        if len(out) >= max(0, max_warn):
            break
    return out


# 数值**关系**一致性（v7，2026-10-13）
# ------------------------------
# 【为什么需要】上面三类规则（来源名 / 数值可追溯 / 伪造引用）有一个**共同的盲区**：
# 它们都只检查"每个数字有没有出处"，**不检查"数字之间的关系是否成立"**。
# 实测（600028 中国石化）报告写「收盘价5.46元…然而，收盘价仍低于MA60（5.1615）」，
# 而 5.46 > 5.1615：两个数字**都逐字真在输入里** ⇒ `audit_numbers` 判"可追溯"、
# `audit_citations` 不触发、来源名/年份/文档名全不命中 ⇒ **三审全过**。
# 该错关系被多头引用 2 次、空头引用 9 次，空头还以它为核心论据。
#
# 【独有价值区】⚠️ 这是决定"该不该加族"的判据（2026-10-13 实测得出）：
#   只有当**两侧真值都已在 state 里**时，加关系规则才有意义。三分类：
#     ① 两侧数字**不在**输入（编造的历史统计 / 同业对比 / 宏观）→ **已由数值·来源名审计抓**
#        （实测 6/6 全命中）⇒ 再补是**重复覆盖**，收益为零、误报为正；
#     ② 关系**模糊主观**（"接近上轨"、"逼近警戒线"、"处于中性偏强"）→ 无可判定真值 ⇒ **做不到**；
#     ③ 两侧真值在、关系判错 → **唯一需要关系审计**（本函数只管这一类）。
#   ⇒ 实测 10 份报告约 **550 处**关系断言，绝大多数属 ①② ⇒ **不要按族铺开**。
#
# 【本版实现的 3 个族】（按用户决定）
#   族1 价格 ↔ MA5/10/20/60·布林带中轨（**含反向主语**，如「MA60 远低于现价」）
#   族2 指标 ↔ 指标（MA 之间、K值/D值、MACD 与信号线、MACD 与零轴）
#   族3 成交量 ↔ 5日/20日均量
#
# 【刻意不收的族】（每条都有实测理由，改动前请先读）
#   · **布林带上/下轨**：实测 8× 多为「股价接近布林带上轨」这类**模糊关系**（阈值主观），
#     收了收益低、误报面大 ⇒ 维持不收。
#   · **财务科目间**（"货币资金0.29亿低于应收账款3.32亿"）：真值在，但需
#     **科目别名表 + 单位对齐 + 派生量推导**三层，而实测这类**几乎不出错**
#     （模型自己在正文写了"（0.29/3.32=8.7%）"在复核）⇒ 投入大、收益小。
#   · **价格 ↔ 支撑/阻力位**：⚠️ **循环论证** —— 支撑阻力位是模型自己定的、不在 state，
#     用它校验它自己的表述没有意义。
#
# 【误报防护】五条，每条都是"宁可漏报不可误伤"，且**必须同时验证真报仍在**
#   （回归基准用 600028 原句 + 合法表述集，见 `tests/test_ma_relation_guard.py`）：
#     ① 关系词白名单不含 突破/跌破/站上 —— 它们天然多用于条件与历史语境；
#     ② 条件句排除（若/如果/一旦/…）；③ 历史句排除（历史/此前/曾/…）；
#     ④ 否定排除（不/未/免，"不会低于"）；⑤ 近似相等容差（方向本身无意义）。
#
# ⚠️ 与 `audit_numbers` 同一定位：**筛查工具，不是判定器**，只报不改。
# ⚠️ 上限独立且很小（`_REL_MAX_WARN = 2`）：关系告警在 `_audit` 里**排在最前**，
#    不限量会把数值/年份类告警整批挤出 `MAX_WARNINGS`。同一个错关系常被引用十几次，
#    报一条比报十条有用。

_GAP = r"[^。；！？\n]{0,14}?"          # 两词之间允许的间隔（限长以免跨段乱匹配）
_REL_A = r"(?P<rel>高于|低于|大于|小于)"
_REL_B = r"(?P<rel>之上|之下|上方|下方)"
_REL_SYM = r"(?P<rel>>|＞|<|＜)"
_REL_ABOVE = ("高于", "大于", "之上", "上方", ">", "＞")
_REL_COND = ("若", "如果", "一旦", "假如", "假设", "倘若", "除非")
_REL_PAST = ("历史", "此前", "过去", "曾", "以往", "当时", "前期")
_REL_NEG = ("不", "未", "免")
_REL_LOOKBACK = 14
_REL_TOL = 0.001                       # 相对容差：几乎相等时"高于/低于"本身无意义
_REL_MAX_WARN = 2                      # ⚠️ 独立上限，见上


def _rel_is_above(word: str) -> bool:
    return word in _REL_ABOVE


_PARA_NUM = re.compile(r"[（(][^）)]*[）)]")


def _gap_clean(seg: str) -> bool:
    """间隔区是否"干净"：**剔除括号内的数字后**不应再残留裸数字。

    ⚠️ 这是族2（指标↔指标）专属的一道守卫，来自实测暴露的**误配机制**：
      原文「MACD柱0.0009，三者均大于0且MACD略高于信号线」——
      "大于" 的真实主语是**三者**（指的是三个值都 >0），
      但 `_P2_FWD`（左指标 + 关系词 + 右指标）会跨越 "0.0009，三者均" 把
      "MACD柱 大于 MACD" 配成一对，从而**报出一个不存在的矛盾**。
      规律：**指标值一旦裸写出来，就说明这个比较已经结束了**，
      后面的关系词属于另一个主语。所以裸数字出现在间隔区 ⇒ 判为误配。
    为什么允许括号内的数字：实测合法写法是
      「MACD线（0.0568）位于信号线（0.0374）之上」——数值在括号里，这类必须保留。
    ⚠️ 只用于族2。族1/族3 的值本来**就裸写**（"收盘价5.46元"/"成交量228,443,300股"），
      套这道守卫会把合法断言全部漏掉。
    """
    return not re.search(r"\d", _PARA_NUM.sub("", seg))


# ---------- 词素（不含分组，便于在多个模式里复用而不会撞组名）----------
_PRICE_T = r"收盘价|股价|价格|现价|当前价|最新价"
_IND_T = r"布林带中轨|MA60|MA20|MA10|MA5|中轨"
# ⚠️ 长名必须在前：否则 "MACD信号线" 会被 "MACD" 前缀截断、"MA60" 会被 "MA6" 截断
_I2_T = (r"MACD信号线|MACD柱|MACD线|MACD值|MACD|信号线|DIF|DEA"
         r"|K值|D值|MA60|MA20|MA10|MA5")
_VOL_T = r"成交量|量能"
_VMA_T = r"5日均量|20日均量"

_IND_KEY = {"MA5": "ma5", "MA10": "ma10", "MA20": "ma20", "MA60": "ma60",
            "布林带中轨": "bb_middle", "中轨": "bb_middle"}
_IND_LABEL = {"ma5": "MA5", "ma10": "MA10", "ma20": "MA20", "ma60": "MA60",
              "bb_middle": "布林带中轨"}
_I2_KEY = {"MA5": "ma5", "MA10": "ma10", "MA20": "ma20", "MA60": "ma60",
           "K值": "k_value", "D值": "d_value", "MACD柱": "macd_histogram",
           "MACD信号线": "macd_signal", "信号线": "macd_signal", "DEA": "macd_signal",
           "MACD线": "macd", "MACD值": "macd", "MACD": "macd", "DIF": "macd"}
_I2_LABEL = {"ma5": "MA5", "ma10": "MA10", "ma20": "MA20", "ma60": "MA60",
             "k_value": "K值", "d_value": "D值", "macd": "MACD",
             "macd_signal": "MACD信号线", "macd_histogram": "MACD柱"}
_VMA_KEY = {"5日均量": "volume_ma5", "20日均量": "volume_ma20"}
_VMA_LABEL = {"volume_ma5": "5日均量", "volume_ma20": "20日均量"}

# ---------- 族1：价格 ↔ 指标 ----------
# 价格在左 → 关系词方向**就是**"价格相对指标"
_P1_PRICE_FWD = re.compile(r"(?:" + _PRICE_T + r")" + _GAP + _REL_A + _GAP
                           + r"(?P<tgt>" + _IND_T + r")")
_P1_PRICE_BACK = re.compile(r"(?:" + _PRICE_T + r")" + _GAP
                            + r"(?P<tgt>" + _IND_T + r")" + _GAP + _REL_B)
# 指标在左、价格在右 → 关系词方向是"**指标**相对价格" ⇒ 解读成"价格相对指标"时要**取反**
#   实测例：「MA60（39.406）持续上行且远低于现价」→ 价格高于 MA60
_P1_IND_FWD = re.compile(r"(?P<tgt>" + _IND_T + r")" + _GAP + _REL_A + _GAP
                         + r"(?:" + _PRICE_T + r")")
_P1_IND_BACK = re.compile(r"(?P<tgt>" + _IND_T + r")" + _GAP
                          + r"(?:" + _PRICE_T + r")" + _GAP + _REL_B)

# ---------- 族2：指标 ↔ 指标（含与零轴）----------
_P2_FWD = re.compile(r"(?P<a>" + _I2_T + r")" + _GAP + _REL_A + _GAP
                     + r"(?P<b>" + _I2_T + r")")
_P2_BACK = re.compile(r"(?P<a>" + _I2_T + r")" + _GAP
                      + r"(?P<b>" + _I2_T + r")" + _GAP + _REL_B)
# 符号写法（"MA5 > MA10"）。间隔收紧到 6 字，符号两侧不该有长内容
_P2_SYM = re.compile(r"(?P<a>" + _I2_T + r")[^。；！？\n]{0,6}?" + _REL_SYM
                     + r"[^。；！？\n]{0,6}?(?P<b>" + _I2_T + r")")
# 与零轴比较（零轴真值恒为 0）。间隔放宽到 20：实测有
#   「MACD值（0.0568），但绝对值仍处零轴下方」这类中间隔了较长修饰语的写法。
#   `[^。；！？\n]` 已阻止跨句匹配，所以放宽是安全的。
_P2_ZERO = re.compile(r"(?P<a>MACD线|MACD值|MACD|DIF)[^。；！？\n]{0,20}?"
                      r"(?:零轴|0轴)[^。；！？\n]{0,4}?(?P<rel>上方|之上|下方|之下)")

# ---------- 族3：成交量 ↔ 均量 ----------
_P3_FWD = re.compile(r"(?P<a>" + _VOL_T + r")" + _GAP + _REL_A + _GAP
                     + r"(?P<b>" + _VMA_T + r")")
_P3_BACK = re.compile(r"(?P<a>" + _VOL_T + r")" + _GAP
                      + r"(?P<b>" + _VMA_T + r")" + _GAP + _REL_B)
_P3_REV = re.compile(r"(?P<c>" + _VMA_T + r")" + _GAP + _REL_A + _GAP
                     + r"(?P<d>" + _VOL_T + r")")


def audit_relations(report: str, indicators: dict = None, *,
                    max_warn: int = _REL_MAX_WARN) -> list[str]:
    """检查报告里「数值之间的关系」是否与真值矛盾（数值**关系**可核实性）。

    Args:
        report: 任意节点的正文（技术面 / 多空辩论 / 汇总 / 交易 / 风控 / 终审）
        indicators: 真值（`state["indicators"]`）。缺某字段则该字段参与的断言**不判**
            （缺就是缺，不补造）
        max_warn: 上限（默认 `_REL_MAX_WARN=2`，刻意很小 —— 见模块注释）

    Returns:
        裸消息列表（由调用方加 `[节点] 关系矛盾: ` 前缀）；无矛盾返回空列表。

    覆盖三个族：价格↔指标（含反向主语）/ 指标↔指标（含零轴）/ 成交量↔均量。
    同一 (族, 左名, 右名, 错误方向) **只报一条**。
    """
    report = str(report or "")
    ind = indicators
    if not report.strip() or not isinstance(ind, dict):
        return []

    def _v(key):
        if not key:
            return None
        x = ind.get(key)
        if x is None or x != x:                      # None 或 NaN
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            _logger.debug("[审计] 数值/类型不符，按缺省处理")
            return None

    out: list[str] = []
    seen: set = set()

    def _emit(fam: str, lv, rv, said_above: bool, pos: int, word_start: int,
              lname: str, rname: str) -> None:
        if lv is None or rv is None or lname is None or rname is None:
            return
        if lv == rv:
            return
        denom = abs(lv) if abs(lv) > 1e-12 else 1.0
        rel = (lv - rv) / denom
        if abs(rel) <= _REL_TOL:
            return                                   # ⑤ 近似相等 → 方向无意义
        truth_above = lv > rv
        if said_above == truth_above:
            return                                   # 方向一致 → 正常
        if any(w in report[max(0, word_start - 3):word_start] for w in _REL_NEG):
            return                                   # ④ 否定式
        head = report[max(0, pos - _REL_LOOKBACK):pos]
        if any(w in head for w in _REL_COND):
            return                                   # ② 条件句
        if any(w in head for w in _REL_PAST):
            return                                   # ③ 历史句
        key = (fam, lname, rname, said_above)
        if key in seen:
            return
        seen.add(key)
        snippet = report[max(0, pos - 20):pos + 26].replace("\n", " ").strip()
        out.append(
            f"「…{snippet}…」方向写反：真值为 {lname}={lv:g} "
            f"{'高于' if truth_above else '低于'} {rname}={rv:g}（{rel * 100:+.2f}%）"
        )

    price = _v("price")

    # ── 族1：价格 ↔ 指标（价格在左）──
    if price is not None:
        for m in _P1_PRICE_FWD.finditer(report):
            k = _IND_KEY.get(m.group("tgt"))
            _emit("price", price, _v(k), _rel_is_above(m.group("rel")),
                  m.start(), m.start("rel"), "收盘价", _IND_LABEL.get(k, m.group("tgt")))
        for m in _P1_PRICE_BACK.finditer(report):
            k = _IND_KEY.get(m.group("tgt"))
            _emit("price", price, _v(k), _rel_is_above(m.group("rel")),
                  m.start(), m.start("rel"), "收盘价", _IND_LABEL.get(k, m.group("tgt")))
        # ── 族1：反向主语（指标在左 ⇒ 关系词方向取反才是"价格相对指标"）──
        for m in _P1_IND_FWD.finditer(report):
            k = _IND_KEY.get(m.group("tgt"))
            _emit("price", price, _v(k), not _rel_is_above(m.group("rel")),
                  m.start(), m.start("rel"), "收盘价", _IND_LABEL.get(k, m.group("tgt")))
        for m in _P1_IND_BACK.finditer(report):
            k = _IND_KEY.get(m.group("tgt"))
            _emit("price", price, _v(k), not _rel_is_above(m.group("rel")),
                  m.start(), m.start("rel"), "收盘价", _IND_LABEL.get(k, m.group("tgt")))

    # ── 族2：指标 ↔ 指标 ──
    for pat in (_P2_FWD, _P2_BACK, _P2_SYM):
        for m in pat.finditer(report):
            if not (_gap_clean(report[m.end("a"):m.start("rel")])
                    and _gap_clean(report[m.end("rel"):m.start("b")])):
                continue                     # ⚠️ 误配守卫，见 _gap_clean
            ka, kb = _I2_KEY.get(m.group("a")), _I2_KEY.get(m.group("b"))
            _emit("ind2", _v(ka), _v(kb), _rel_is_above(m.group("rel")),
                  m.start(), m.start("rel"),
                  _I2_LABEL.get(ka, m.group("a")), _I2_LABEL.get(kb, m.group("b")))
    # ── 族2b：MACD ↔ 零轴（零轴真值恒为 0）──
    for m in _P2_ZERO.finditer(report):
        if not _gap_clean(report[m.end("a"):m.start("rel")]):
            continue                         # 同上
        ka = _I2_KEY.get(m.group("a"))
        _emit("zero", _v(ka), 0.0, _rel_is_above(m.group("rel")),
              m.start(), m.start("rel"), _I2_LABEL.get(ka, m.group("a")), "零轴")

    # ── 族3：成交量 ↔ 均量 ──
    vol = _v("volume")
    if vol is not None:
        for m in _P3_FWD.finditer(report):
            k = _VMA_KEY.get(m.group("b"))
            _emit("vol", vol, _v(k), _rel_is_above(m.group("rel")),
                  m.start(), m.start("rel"), "成交量", _VMA_LABEL.get(k, m.group("b")))
        for m in _P3_BACK.finditer(report):
            k = _VMA_KEY.get(m.group("b"))
            _emit("vol", vol, _v(k), _rel_is_above(m.group("rel")),
                  m.start(), m.start("rel"), "成交量", _VMA_LABEL.get(k, m.group("b")))
        for m in _P3_REV.finditer(report):
            k = _VMA_KEY.get(m.group("c"))
            _emit("vol", vol, _v(k), not _rel_is_above(m.group("rel")),
                  m.start(), m.start("rel"), "成交量", _VMA_LABEL.get(k, m.group("c")))

    return out[:max(1, max_warn)]


# 风控阈值 与「仓位 vs 上限」一致性（v8，2026-10-13）
# ------------------------------
# 这一族与 `audit_relations` 的**真值来源不同**：两侧不在 `indicators` 里，
# 而在 `RiskEngine`（阈值）与 `trade_plan`（计划仓位）。所以单独成函数。
#
# 为什么值得单列（项目最严重的两次风控事故都是**阈值被编造**）：
#   · 2026-10-08 风控编出「单票上限 8%、关注类 5%」，把本在限额内的 15% 判为
#     "超限 200%、触发熔断、驳回"——而项目真实上限是 25%；
#   · 同日终审又自己造了「合规上限 3.3%」把方案驳回，与风控用的不是同一套规则。
#   `audit_numbers` **抓不到**：8、3.3 都能由输入里的数字凑出来 ⇒ 必须按**语义位置**检。
#
# 两类检查：
#   ① **阈值本身**：报告写的"单票…上限 X%"必须**等于**真值 25%。
#      项目原话：编造合规依据比不给依据更危险。
#   ② **比较结论**：写「计划 X% 未超过上限 Y%」时 X ≤ Y 必须成立；写「超限」同理
#      必须 X > Y。**结论与数值矛盾即报**（这正是风控/终审之间最会出错的地方）。
#
# ⚠️ 保守设计（每条都在压误报）：
#   · 四类阈值**各自限定单位词**（单票/行业 只认「上限」；回撤 只认「熔断线|上限」；
#     每笔 只认「最大风险|风险|上限」）—— 否则「最大回撤预估值为 0%」会被误当成
#     「回撤熔断线 0%」（实测报告里真有这句）。
#   · 比较结论只在**明确矛盾**时报；「计划 0%」（观望档）不会触发（0 ≤ 25 成立）。
#   · 上限值与真值比用**相对容差**，避免 25 与 25.0 的写法差异。
_RISK_LIMIT_PATTERNS = (
    ("single", re.compile(
        r"(?:单票|单只|单一标的)[^。；！？\n]{0,12}?上限[^。；！？\n]{0,8}?(\d+(?:\.\d+)?)\s*%")),
    ("sector", re.compile(
        r"(?:单行业|行业仓位)[^。；！？\n]{0,10}?上限[^。；！？\n]{0,8}?(\d+(?:\.\d+)?)\s*%")),
    ("drawdown", re.compile(
        r"(?:最大回撤|回撤)[^。；！？\n]{0,10}?(?:熔断线|上限)[^。；！？\n]{0,8}?(\d+(?:\.\d+)?)\s*%")),
    ("per_trade", re.compile(
        r"(?:每笔|单笔)[^。；！？\n]{0,10}?(?:最大风险|风险|上限)[^。；！？\n]{0,8}?(\d+(?:\.\d+)?)\s*%")),
)
_RISK_LIMIT_LABEL = {"single": "单票最大仓位", "sector": "单行业最大仓位",
                     "drawdown": "组合最大回撤熔断线", "per_trade": "每笔交易最大风险"}
# 「计划 X% … 上限 Y%」+ 前后语境（用于判断它声称的是"未超限"还是"超限"）
_RISK_CMP_RE = re.compile(
    r"计划[^。；！？\n]{0,8}?(?P<plan>\d+(?:\.\d+)?)\s*%"
    r"(?P<mid>[^。；！？\n]{0,20}?)上限[^。；！？\n]{0,8}?(?P<lim>\d+(?:\.\d+)?)\s*%"
    r"(?P<tail>[^。；！？\n]{0,24})")
_RISK_NEG_CLAIM = ("未超过", "不超过", "未超限", "未超出", "未超", "符合")
_RISK_POS_CLAIM = ("超限", "超出", "超过")
# 「计划（仓位）X%」——⚠️ 间隔收紧到 4 字：宽松版会把
# 「计划采用分批建仓，首笔不超过 15%」的首笔比例误当成计划仓位。
_RISK_PLAN_RE = re.compile(
    r"计划(?:仓位|持仓|的仓位)?[^。；！？\n]{0,4}?(?P<plan>\d+(?:\.\d+)?)\s*%")


def audit_risk_limits(report: str, *, position_size=None, limits: dict = None,
                      max_warn: int = 2) -> list[str]:
    """检查报告里引用的**风控阈值**与「仓位 vs 上限」结论是否与真值一致。

    Args:
        report: 风控 / 终审节点的正文
        position_size: 交易计划的仓位（`0.15` 或 `"15%"` 都可；未知传 None）
        limits: 真值阈值，**分数形式**（`{"single": 0.25, ...}`），取自 `RiskEngine`
        max_warn: 上限（小，理由同 `audit_relations`）

    Returns:
        裸消息列表（调用方加 `[节点] 风控阈值: ` 前缀）；无问题返回空列表。
    """
    report = str(report or "")
    if not report.strip() or not isinstance(limits, dict) or not limits:
        return []

    real = {}
    for k, v in limits.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            _logger.debug("[审计] 数值/类型不符，按缺省处理")
            continue
        if fv > 1.0:                       # 容忍调用方直接传百分数
            fv = fv / 100.0
        real[k] = fv
    if not real:
        return []

    out: list[str] = []
    seen: set = set()

    def _add(msg: str) -> None:
        if msg not in seen:
            seen.add(msg)
            out.append(msg)

    # ── ① 阈值本身是否与真值一致 ──
    for kind, pat in _RISK_LIMIT_PATTERNS:
        want = real.get(kind)
        if want is None:
            continue
        for m in pat.finditer(report):
            got = float(m.group(1)) / 100.0
            if abs(got - want) <= 0.001:
                continue
            label = _RISK_LIMIT_LABEL.get(kind, kind)
            snippet = report[max(0, m.start() - 16):m.end() + 12].replace("\n", " ").strip()
            _add(f"「…{snippet}…」写的 {label} 是 {got * 100:g}%，"
                 f"但项目真实阈值是 {want * 100:g}%（编造合规阈值比不给依据更危险）")

    # ── ② 「计划 X% vs 上限 Y%」的结论是否与数值矛盾 ──
    pos = None
    if position_size is not None:
        try:
            s = str(position_size).strip().rstrip("%")
            pos = float(s)
            if pos > 1.0 or "%" in str(position_size):
                pos = pos / 100.0
        except (TypeError, ValueError):
            _logger.debug("[审计] 数值/类型不符，按缺省处理")
            pos = None

    for m in _RISK_CMP_RE.finditer(report):
        plan_pct = float(m.group("plan")) / 100.0
        lim_pct = float(m.group("lim")) / 100.0
        ctx = m.group("mid") + m.group("tail")
        # 上限值不在任何真实阈值里 → 这是编造的上限
        if not any(abs(lim_pct - v) <= 0.001 for v in real.values()):
            allowed = "、".join(f"{v * 100:g}%" for v in real.values())
            _add(f"「…{m.group(0).strip()[:60]}…」里的上限 {lim_pct * 100:g}% "
                 f"不在项目风控规则内（真实阈值为 {allowed}）")
            continue
        is_neg = any(w in ctx for w in _RISK_NEG_CLAIM)
        is_pos = (not is_neg) and any(w in ctx for w in _RISK_POS_CLAIM)
        if is_neg and plan_pct > lim_pct:
            _add(f"「…{m.group(0).strip()[:60]}…」写的是「未超过」，"
                 f"但 {plan_pct * 100:g}% > {lim_pct * 100:g}% —— 结论与数值矛盾")
        elif is_pos and plan_pct <= lim_pct:
            _add(f"「…{m.group(0).strip()[:60]}…」写的是「超限」，"
                 f"但 {plan_pct * 100:g}% ≤ {lim_pct * 100:g}% —— 结论与数值矛盾")

    # ── ③ 报告是否把**交易计划里的仓位**引错 ──
    # ⚠️ 单独一条更严的规则：必须紧贴「计划」二字（≤4 字间隔），
    #    否则「计划采用分批建仓，首笔不超过 15%」会把首笔比例误当计划仓位。
    if pos is not None:
        for m in _RISK_PLAN_RE.finditer(report):
            plan_pct = float(m.group("plan")) / 100.0
            if abs(plan_pct - pos) <= 0.001:
                continue
            _add(f"「…{m.group(0).strip()[:50]}…」把计划仓位写成 {plan_pct * 100:g}%，"
                 f"但交易计划里的仓位是 {pos * 100:g}%")
    return out[:max(1, max_warn)]
