"""
统一存储层：新闻条目 / 对话日志 / 相似历史日特征向量（SQLite，与 portfolio.db 同库）。

设计原则：
- 结构化累积型数据入库（查询/去重/增量/保留策略）；报告 MD 与回测 JSON 保留文件形态
- 原生 sqlite3（与 db/models.py 一致，不引 ORM）
"""

import hashlib
import json
import sqlite3
import logging
from datetime import datetime, timedelta

from db.models import DB_PATH, get_connection

_logger = logging.getLogger(__name__)

# 数据保留策略（天）
NEWS_RETENTION_DAYS = 180
CHAT_RETENTION_DAYS = 90
LLM_CALL_RETENTION_DAYS = 30
# 分析历史要覆盖 T+20 回看窗口，保留期必须 > 20 个交易日（≈30 自然日），
# 取 400 天（约 270 个交易日）兼顾存储成本与年度复盘需求
ANALYSES_RETENTION_DAYS = 400
# 选股候选同样要覆盖 T+20 回看窗口并留年度复盘空间（每天最多几十行，量不大）
SCREEN_PICKS_RETENTION_DAYS = 400
# 全市场资金流快照（约 5200 行/天）。资金流是"当天状态"，很少回溯一个月以上，
# 但保留 120 天便于与 T+N 收益做交叉验证（"当日主力净流入高的后来涨了没"）。
FUND_FLOW_SNAPSHOT_RETENTION_DAYS = 120
# 估值截面快照（PE / PB / 市值 / 换手率，约 3000 行/天）。
# ⚠️ 这张表**不能照抄资金流的 120 天**：资金流是"当天状态"，而这批列的**唯一用途**
# 就是算 PE / 市值 / 换手率的 **IC** —— 而 IC 要求"某日截面 vs 之后收益"，样本期
# 至少要几十个交易日、且越长越稳（本项目回测口径用 2024-01~2025-06 共约 360 个交易日）。
# 按 120 天保留 = 永远只有半年样本，且被清理得比积累还快，等于白采。
# 取 800 天（≈ 550 个交易日），覆盖一年半的 IC 分析窗口；每天 3000 行 ≈ 0.6MB，
# 存储成本可接受。
MARKET_SNAPSHOT_RETENTION_DAYS = 800
# 新闻行业评估：结论只在当天有操作意义（次日盘前看的是最新一期），
# 但保留 90 天用于回看"当时的判断对不对"（与 analyses 的 T+N 回填思路一致）
NEWS_REVIEW_RETENTION_DAYS = 90

_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hash TEXT UNIQUE,
    title TEXT,
    content TEXT,
    source TEXT,
    published_at TEXT,
    related_codes TEXT,
    sentiment TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_published ON news_items(published_at);
CREATE INDEX IF NOT EXISTS idx_news_hash ON news_items(hash);

-- 情绪面选股（P1-5，2026-10-23）：新闻 → 个股的**严匹配**命中聚合。
-- 为什么单独建表而不是复用 news_items.related_codes：那列只关联**关注池**
-- （自选股∪候选∪持仓），"从新闻里发现新票"恰恰要求对接**全市场**名称表。
-- 一行 = (交易日, 个股)：多轮快讯命中累加，情绪分正/负计数。
CREATE TABLE IF NOT EXISTS news_stock_hits (
    trade_date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    hits INTEGER DEFAULT 0,
    positive INTEGER DEFAULT 0,
    negative INTEGER DEFAULT 0,
    first_at TEXT,
    last_at TEXT,
    sources TEXT,
    samples TEXT,
    updated_at TEXT,
    PRIMARY KEY (trade_date, stock_code)
);
CREATE INDEX IF NOT EXISTS idx_nsh_date ON news_stock_hits(trade_date);
CREATE INDEX IF NOT EXISTS idx_nsh_hits ON news_stock_hits(trade_date, hits DESC);

CREATE TABLE IF NOT EXISTS chat_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT,
    content TEXT,
    tools_used TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_chat_created ON chat_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_logs(session_id);

-- ===== 对话会话维表（P1：让对话历史"读得出来"）=====
-- 背景：chat_logs 只有消息维，session_id 此前只活在 st.session_state 里，
-- 浏览器一刷新就换号 → 历史"写了也读不出来"。会话维表补上会话粒度：
-- 标题/人格/模型/消息数/token 累计，UI 靠它列会话、切会话、回放。
CREATE TABLE IF NOT EXISTS chat_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE,
    title TEXT,
    persona TEXT,
    model TEXT,
    source TEXT,               -- ui / api / legacy（旧数据补登记）
    message_count INTEGER DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_csess_updated ON chat_sessions(updated_at);

-- ===== LLM 调用链（Phase 6.2）=====
-- 目的：LLM 是成本与不可复现性的主要来源。此前只能靠 llm 日志文件翻，
-- 无法回答"这周花了多少 token""哪一步最慢""失败集中在哪个节点"。
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    model TEXT,
    purpose TEXT,               -- 调用方标签（agents.tech / nl_screener.parse / chat …）
    ok INTEGER,                 -- 1=成功 0=失败
    latency_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    prompt_chars INTEGER,
    response_chars INTEGER,
    error TEXT,
    cost_cny REAL                -- 本次调用的金额（元），按**调用发生时**的单价算；NULL=未知/未计
);
CREATE INDEX IF NOT EXISTS idx_llm_created ON llm_calls(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_purpose ON llm_calls(purpose, created_at);

CREATE TABLE IF NOT EXISTS daily_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    date TEXT,
    vector TEXT,
    close REAL,
    future_ret_5d REAL,
    future_ret_20d REAL,
    updated_at TEXT,
    UNIQUE(symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_features_symbol ON daily_features(symbol, date);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT,
    summary TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_type ON runs(run_type, created_at);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_date TEXT,
    content TEXT,
    kind TEXT,
    verified_date TEXT,
    result TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_review ON observations(review_date);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT,
    created_at TEXT,
    code TEXT,
    name TEXT,
    rule TEXT,
    level TEXT,
    message TEXT,
    price REAL,
    change_pct REAL,
    quote_time TEXT,
    source TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_date ON alerts(trade_date, created_at);

-- ===== 模拟交易账户（Phase 2）=====
CREATE TABLE IF NOT EXISTS paper_accounts (
    name TEXT PRIMARY KEY,
    initial_cash REAL NOT NULL,
    cash REAL NOT NULL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS paper_positions (
    name TEXT NOT NULL,
    code TEXT NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 0,
    available INTEGER NOT NULL DEFAULT 0,
    total_cost REAL NOT NULL DEFAULT 0,
    last_buy_date TEXT,
    updated_at TEXT,
    PRIMARY KEY (name, code)
);

CREATE TABLE IF NOT EXISTS paper_orders (
    name TEXT NOT NULL,
    order_id INTEGER NOT NULL,
    code TEXT,
    direction TEXT,
    quantity INTEGER,
    submit_date TEXT,
    status TEXT,
    reason TEXT,
    filled_quantity INTEGER DEFAULT 0,
    filled_price REAL DEFAULT 0,
    filled_date TEXT,
    -- 限价意图（2026-10-27）：此前只有"次日开盘市价成交"，"跌到 X 才买"表达不出来，
    -- 写进去也会退化成"次日开盘无条件买"——恰好是想避免的追价。
    -- 0/NULL = 市价单（旧数据语义，行为与改动前完全一致）。
    limit_price REAL,
    decision_price REAL,   -- 决策时看到的参考价，算滑点的分母
    slippage REAL,         -- filled_price - decision_price：买入为正=追高吃亏，卖出为负=割肉吃亏
    updated_at TEXT,
    PRIMARY KEY (name, order_id)
);

CREATE TABLE IF NOT EXISTS paper_fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date TEXT,
    code TEXT,
    direction TEXT,
    quantity INTEGER,
    actual_price REAL,
    cash_after REAL,
    payload TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_paper_fills ON paper_fills(name, date);
-- 同一状态重复落库时保持幂等（否则每次保存都会重复插入历史成交）
CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_fills_uniq
    ON paper_fills(name, date, code, direction, quantity, actual_price, cash_after);

CREATE TABLE IF NOT EXISTS paper_nav (
    name TEXT NOT NULL,
    date TEXT NOT NULL,
    cash REAL,
    position_value REAL,
    total_value REAL,
    return_pct REAL,
    -- ⚠️ 当日估值中"因取不到收盘价而**按成本价**兜底"的持仓只数（2026-10-15 加）。
    --    口径必须**随数字一起落库**：没有它，一条"成本价兜底的假净值"看起来与真实净值
    --    完全一样（实测 AI 据此得出"净值不变、无盈亏"的错误复盘）。
    --    0 / NULL = 全部按真实市价估值。
    at_cost INTEGER,
    updated_at TEXT,
    PRIMARY KEY (name, date)
);

-- ===== 深度分析历史（含 T+N 回看闭环）=====
-- 目的：此前深度分析结果只存 output/ui_runs/analysis_{code}.json，
-- 每只股票仅保留最后一份且会覆盖，无时间维度、无历史回放、无收益核对。
-- 本表复制 observations 的"预测 → 核对"范式到分析决策层面：
-- 每次分析落一行（payload 存完整快照 JSON），T+N 收盘价回填 ret_5d/ret_20d，
-- 累积出"模型实际胜率"数据，供后续对照实验与反思注入使用。
-- backfill_status 状态机：
--   NULL   = 新分析，尚未回填（每日调度会重试）
--   partial= 已有部分数据（如 ret_5d 已算出但 ret_20d 未到期）
--   done   = ret_20d 已回填（终态）
--   failed = 超过回填时限（分析日 +60 自然日仍不满足）或数据源持续失败
--   skip   = 分析本身失败/无有效决策，不参与胜率统计（但保留样本避免幸存者偏差）
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_date TEXT,          -- 分析日（YYYY-MM-DD，回填对齐基准）
    stock_code TEXT,
    stock_name TEXT,
    price_at_analysis REAL,      -- 分析时股价（T+0 基准价）
    decision_label TEXT,         -- 看多|看空|观望|驳回|分析失败
    model TEXT,                  -- 分析所用 LLM
    source TEXT,                 -- ui|pipeline（分析发起入口）
    payload TEXT,                -- 完整分析快照 JSON（bull/bear 辩论、各 agent 输出等）
    trade_plan TEXT,             -- 交易计划 JSON（direction/target_price/stop_loss…）
    timing_regime TEXT,          -- 择时状态（bull|bear|range…）
    market_score REAL,           -- 择时得分
    market_score_source TEXT,    -- 分数来自哪个引擎：market_detector|timing_engine|fallback
                                 -- ⚠️ 2026-10-20 起分析图改用 MarketDetector（与选股同口径）；
                                 -- 此列之前的历史行是 timing_engine，两批分数不可直接比大小。
    ret_5d REAL,                 -- T+5 收益率（回填）
    ret_20d REAL,                -- T+20 收益率（回填）
    hit_target INTEGER,          -- 期间是否触及目标价（1/0/NULL 未回填）
    hit_stop INTEGER,            -- 期间是否触及止损价（1/0/NULL 未回填）
    backfill_status TEXT,
    backfilled_at TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_ana_code_date ON analyses(stock_code, analysis_date);
CREATE INDEX IF NOT EXISTS idx_ana_date ON analyses(analysis_date);
CREATE INDEX IF NOT EXISTS idx_ana_status ON analyses(backfill_status);

-- ===== 深度分析复盘（2026-10-20）=====
-- 为什么单独一张表而不是往 analyses 加列：
--   ① 复盘是**一次分析的派生结论**，与"那次分析本身"是一对一但语义不同
--      （analyses 记"说了什么"，本表记"事后看对不对、为什么"）；
--   ② 复盘要能重算（数据修正/归因阈值调整后重跑），旧结论必须留着对比——
--      若与 analyses 同行，重算只能覆盖，两侧结论的差异就永久丢失了。
-- facts 必须落库：它是"归因结论"的唯一依据，不留下来就无法回答
-- "当时凭什么说止损过近"（与回放必须带数据指纹同理）。
-- status: mechanical=只有机械归因 / ok=LLM 归因通过校验 / ok_with_warnings /
--         llm_error（保留机械结论）/ rejected（LLM 选了候选外的原因且重试仍失败）
CREATE TABLE IF NOT EXISTS analysis_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL UNIQUE,   -- 1:1 指向 analyses.id
    stock_code TEXT,
    stock_name TEXT,
    analysis_date TEXT,
    review_date TEXT,                      -- 复盘执行日（≠ 分析日）
    decision_label TEXT,
    model TEXT,                            -- 分析时所用模型（换模型对照要靠它）
    verdict TEXT,                          -- correct|partially_correct|wrong|unclear|opportunity_missed|no_position
    primary_cause TEXT,
    secondary_causes TEXT,                 -- JSON 数组
    horizon INTEGER,                       -- 本结论基于哪个窗口：5=早期（partial）/ 20=完整（done）
    facts TEXT,                            -- JSON：代码算出的客观价格路径
    candidates TEXT,                       -- JSON：候选原因码 + 每条的实测证据
    llm_review TEXT,                       -- JSON：模型归因（校验后）
    status TEXT,
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_rev_cause ON analysis_reviews(primary_cause);
CREATE INDEX IF NOT EXISTS idx_rev_verdict ON analysis_reviews(verdict);
CREATE INDEX IF NOT EXISTS idx_rev_date ON analysis_reviews(analysis_date);

-- ===== 跨进程任务互斥锁 =====
-- ===== 选股候选（P2：条件命中率回测的样本来源）=====
-- 为什么要有这张表：深度分析一天最多 1~2 次，攒够"胜率对照实验"需要的 50+ 样本要两个月；
-- 而一次选股就产出 5~10 只候选，且天然带"条件命中明细 + 因子得分"，
-- 是回答"这些条件到底有没有区分度"唯一能快速攒够样本的地方。
-- bench_ret_* 是同期上证指数收益（来自 daily_features），用于算超额——
-- 选股赚不赚钱必须跟大盘比，否则牛市里随便选都"赢"。
CREATE TABLE IF NOT EXISTS screen_picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pick_date TEXT,
    stock_code TEXT,
    stock_name TEXT,
    source TEXT,                -- screen / nl_screen / pipeline
    strategy TEXT,              -- 初筛策略（主力资金 / 放量突破 / …）
    score REAL,
    score_coverage REAL,        -- 因子覆盖率（数据不全的候选靠它排除）
    price_at_pick REAL,
    rank_in_run INTEGER,        -- 本次选股中的名次（1=第一名）
    market_state TEXT,
    market_score REAL,
    factors TEXT,               -- JSON {因子: {score, max, status, detail}}
    conditions TEXT,            -- JSON {hits: [], misses: [], unverified: []}
    bench_ret_5d REAL,
    bench_ret_20d REAL,
    ret_5d REAL,
    ret_20d REAL,
    backfill_status TEXT,       -- NULL→partial→done；超龄 failed；无候选 skip
    backfilled_at TEXT,
    pe_source TEXT,             -- B2③：本次评分的 PE 口径（eastmoney/snapshot_sina）
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_pick_date ON screen_picks(pick_date);
CREATE INDEX IF NOT EXISTS idx_pick_code ON screen_picks(stock_code, pick_date);
CREATE INDEX IF NOT EXISTS idx_pick_status ON screen_picks(backfill_status);

-- 场景：T+N 回填既有调度（15:35）也有 UI 手动按钮，两者是**不同进程**。
-- 同时跑会重复拉同一批 K 线，白白消耗数据源额度并触发限频。
-- 用数据库做锁而非文件锁：SQLite 在 WAL 下写事务串行化，INSERT OR IGNORE
-- 的原子性由数据库保证；expires_at 让持有者崩溃后锁能自动释放（不留死锁）。
CREATE TABLE IF NOT EXISTS task_locks (
    task TEXT PRIMARY KEY,
    holder TEXT,
    acquired_at TEXT,
    expires_at TEXT
);

-- 全市场资金流快照（盘后一次性落库，选股只读这里）。
--
-- 为什么必须落库（2026-09-25 实测）：同花顺的快照接口一次返回**全市场 5210 行**，
-- 而"按窗口"要分别请求 —— 即时 16.7s / 3日 20.8s / 5日 15.6s / 10日 28.6s。
-- 于是为了拿到**一只**股票的资金流要 75.9 秒（4 次全市场），而快照内存缓存只有 300 秒，
-- 选股一轮 4~5 分钟**中途必然过期再拉一轮** → 一次选股在这上面白花 150 秒左右。
-- 改成盘后（15:40）采集一次、全天复用：采集 ≈77s（一次），选股读库 ≈0.05s/只。
CREATE TABLE IF NOT EXISTS fund_flow_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT,         -- 数据所属交易日（YYYY-MM-DD），≠ 采集时间
    stock_code TEXT,
    stock_name TEXT,
    net_today REAL,             -- 净额（元）；'即时' 窗口
    net_3d REAL, net_5d REAL, net_10d REAL,
    amount REAL,                -- 成交额（元）
    change_pct REAL,
    price REAL,
    source TEXT,                -- ths / eastmoney
    captured_at TEXT,           -- 采集时刻（便于区分"数据日期"与"采集时间"）
    created_at TEXT,
    UNIQUE(snapshot_date, stock_code)
);
CREATE INDEX IF NOT EXISTS idx_ffsnap_date ON fund_flow_snapshot(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_ffsnap_code ON fund_flow_snapshot(stock_code, snapshot_date);

-- 行业资金流快照（全行业 ~90 个）。与上一张表同理落库：
-- 市场检测的"板块轮动"原走东财 stock_sector_fund_flow_rank（本环境不可达），
-- 现在优先读这张表（同花顺的行业净额可直接算同样的离散度）。
CREATE TABLE IF NOT EXISTS industry_flow_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT,
    industry TEXT,
    net_amount REAL,            -- 净额（元）
    change_pct REAL,
    leader TEXT,                -- 领涨股
    source TEXT,
    captured_at TEXT,
    created_at TEXT,
    UNIQUE(snapshot_date, industry)
);
CREATE INDEX IF NOT EXISTS idx_indflow_date ON industry_flow_snapshot(snapshot_date);

-- 估值/规模截面快照（PE / PB / 市值 / 换手率，新浪 49 行业明细，约 3000 行/天）。
-- 为什么必须**逐日**落库、不能"用当期快照顶替"：
--   IC 的定义是「某日 T 的截面因子排名 vs T 之后的收益排名」，它要求能重建
--   "T 那一天的全市场截面"。K 线类因子天然满足（任何历史交易日的日线都能重算），
--   而 PE / 市值 / 换手率的**绝对水平**只存在于当时的快照里 —— 用今天的 PE 去算
--   去年的 IC 就是**前视**，得到的数字毫无意义。
--   ⇒ 这正是"六因子权重里 PE 那一项无法验证"的根因；本表是那个前置条件。
-- 单位：amount / market_cap / float_cap 一律 **元**（与 fund_flow_snapshot 对齐，
--   避免跨表比大小的时候踩"同名列不同量级"）；pe / pb 为倍数，turnover / change_pct
--   是**百分数**（不是小数）。
CREATE TABLE IF NOT EXISTS market_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT,         -- 数据所属交易日（YYYY-MM-DD），≠ 采集时间
    stock_code TEXT,
    stock_name TEXT,
    industry TEXT,              -- 新浪行业名（当期口径，见 caveat）
    board TEXT,                 -- 主板沪市/主板深市/创业板/科创板/北交所/其他
    price REAL,
    change_pct REAL,            -- 涨跌幅，百分数
    pe REAL,
    pb REAL,
    market_cap REAL,            -- 总市值（元）
    float_cap REAL,             -- 流通市值（元）
    turnover REAL,              -- 换手率，百分数
    amount REAL,                -- 成交额（元）
    source TEXT,
    captured_at TEXT,           -- 采集时刻（区分"数据日期"与"采集时间"）
    created_at TEXT,
    UNIQUE(snapshot_date, stock_code)
);
CREATE INDEX IF NOT EXISTS idx_mksnap_date ON market_snapshot(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_mksnap_code ON market_snapshot(stock_code, snapshot_date);

-- ===== 采集覆盖率（P2-7，2026-10-17）=====
-- 为什么单独一张表：`market_snapshot` 只能回答"落了哪些股票"，**回答不了**
--   "本该有多少只"。实测 10-16 只落了 2991 只（全市场 ~5459）⇒ **覆盖率 54.8%**，
--   而页面/日志都只显示"2991 只"，看不出来它其实只覆盖了一半。
--   ⇒ 把「实取 / 全市场 / 覆盖率 / 降级原因」逐日落库，才能看出趋势与断崖。
-- ⚠️ `total` 允许为 NULL（来源不给全市场数时）⇒ 此时 ratio 也是 NULL，**不许写 0**。
--   （0 会把"不知道"伪装成"覆盖率为零"，正是本项目反复踩的坑。）
CREATE TABLE IF NOT EXISTS data_coverage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data_date TEXT NOT NULL,    -- 数据所属交易日（YYYY-MM-DD），≠ 采集时刻
    dataset TEXT NOT NULL,      -- market_snapshot / fund_flow_stock / fund_flow_industry …
    source TEXT,                -- sina_sector / ths / eastmoney / tushare
    got INTEGER,                -- 实取条数
    total INTEGER,              -- 全市场应有条数；NULL = 来源未提供（不是 0）
    ratio REAL,                 -- got / total；total 为 NULL 时也为 NULL
    status TEXT,                -- ok / partial / degraded / failed
    note TEXT,                  -- 降级/失败原因（必须可查）
    captured_at TEXT,
    created_at TEXT,
    UNIQUE(data_date, dataset)
);
CREATE INDEX IF NOT EXISTS idx_datacov_date ON data_coverage(data_date);

-- ===== 新闻行业利好/利空评估（热点新闻工作流，2026-10-09）=====
-- 每天 15:55 由 core/news_industry_review.py 生成：把最近 24h 快讯按行业聚合，
-- LLM 评估每个行业「利好/利空/中性」+ 理由 + 证据（引用库内 news_items.id）。
-- ⚠️ evidence 里的 id 由**代码校验**（必须出现在当日输入里）——防 LLM 编造引用。
-- 保留策略：按 review_date（数据所属日期）清理，与 created_at 分开
-- （补跑历史日期的评估时 created_at 是"今天"，按创建时间清理会漏删）。
CREATE TABLE IF NOT EXISTS news_industry_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_date TEXT NOT NULL,      -- 评估的数据日期（YYYY-MM-DD），≠ 生成时间
    sector TEXT NOT NULL,           -- 行业（news_feed._SECTOR_KEYWORDS 口径）
    verdict TEXT NOT NULL,          -- 利好 / 利空 / 中性
    summary TEXT,                   -- 一句话理由
    evidence TEXT,                  -- JSON [{"id": news_id, "title": ...}]
    news_count INTEGER,             -- 该行业当日聚合到的快讯条数
    model TEXT,                     -- 评估用模型（可追溯）
    created_at TEXT,
    UNIQUE(review_date, sector)
);
CREATE INDEX IF NOT EXISTS idx_news_review_date ON news_industry_reviews(review_date);

-- ===== 自定义定时任务（L2 扩展层）=====
-- 内置任务硬编码在 core/scheduler.py::JOBS 里，改时间/加任务都要改代码再重启。
-- 而"忘了重启"这个坑本项目已经真实踩过两次：
--   1) 新增 risk_check 没重启 → 健康检查的 code 段报"跑的是旧代码"；
--   2) snapshot_flow(15:40) 与 daily_review(15:45) 只隔 5 分钟而快照跑 9 分钟
--      → 复盘重复采集一次（9-28 实测）。
-- 本表把"什么时候跑 / 开不开"变成数据：调度进程每 60s 热加载一次，改完不用重启。
--
-- 刻意**只存"跑哪个已有任务(mode) + 什么时候跑"**，不支持"任意 prompt 定时执行"：
-- LLM 是真实成本，自由文本的定时任务等于给账单开一个无人看守的阀门。
-- mode 必须落在 core/scheduled_jobs.py::JOB_MODES 白名单内（深度分析不在其中）。
--
-- 保留策略：本表是**配置**而非时间序列（与 task_locks 同类），不参与 cleanup()。
-- 自定义任务不在 JOB_IDS 里，因此不会触发健康检查的"旧代码"告警，
-- 心跳另用 custom_job_ids 字段单独报告（见 core/scheduler_health.py）。
CREATE TABLE IF NOT EXISTS scheduled_jobs (
    job_id TEXT PRIMARY KEY,        -- custom_xxx；禁止与内置 job_id 重名
    name TEXT NOT NULL,             -- 展示名（同时作为日志与推送标题）
    mode TEXT NOT NULL,             -- 执行逻辑（白名单见 core/scheduled_jobs.py）
    cron TEXT NOT NULL,             -- JSON: {"hour":15,"minute":5,"day_of_week":"mon-fri"}
    enabled INTEGER DEFAULT 1,      -- 0=停用（改开关也是一次热加载，不用重启）
    trading_day_only INTEGER DEFAULT 1,  -- 非交易日是否跳过（依赖行情数据的 mode 会被强制为 1）
    note TEXT DEFAULT '',
    created_at TEXT,
    updated_at TEXT
);

-- ================================================================
-- AI 交易联赛（2026-10-14，见 output/多AI交易联赛_开发计划_2026-10-14.md）
-- ⚠️ 全部是**新表**（CREATE TABLE IF NOT EXISTS 够用）；给**既有表**加列
--    必须走 _ensure_column()——init_store 的哈希版本号对旧表加列是假安全。
-- ================================================================

-- ① 人格（AI 定义）。name 同时是 paper_accounts.name，直接复用账户体系。
--    ⚠️ kind='random' 的随机对照组**不调 LLM**（口径 1：没有它排行不成立）；
--    ⚠️ frequency 是口径 3 的载体：daily（T+1 开盘成交）与 intraday（扫描价成交）
--    **不可混榜**，结算与排行必须按此分组。
CREATE TABLE IF NOT EXISTS ai_personas (
    name           TEXT PRIMARY KEY,   -- 账户名兼 AI 标识，如 ai_value
    label          TEXT NOT NULL,      -- 中文显示名
    kind           TEXT NOT NULL DEFAULT 'llm',   -- llm / random（随机对照组）
    model          TEXT,               -- C1：绑定的模型；空 = 跟随全局默认
    style_prompt   TEXT NOT NULL,      -- C2：风格提示词（写入交易员/风控节点）
    risk_bias      TEXT,               -- C2 代码兜底：conservative/balanced/aggressive
    max_position   REAL,               -- 单票仓位上限（代码夹紧；空 = RiskEngine 值）
    info_scope     TEXT,               -- all / tech / fundamental / news（信息集裁剪）
    frequency      TEXT NOT NULL DEFAULT 'daily', -- daily / intraday（不可混榜）
    initial_cash   REAL NOT NULL DEFAULT 1000000,
    enabled        INTEGER NOT NULL DEFAULT 1,
    created_at TEXT, updated_at TEXT
);

-- ② 研究预算台账（C3）。**唯一实现**：所有资源扣减都经 consume_ai_budget()。
--    配额是"当日快照"：改配额只影响当天与以后，历史日期的行永不被 UPDATE 碰到。
CREATE TABLE IF NOT EXISTS ai_budget (
    persona     TEXT NOT NULL,
    budget_date TEXT NOT NULL,         -- 交易日
    kind        TEXT NOT NULL,         -- deep_analysis / wake / decision_round
    used        INTEGER NOT NULL DEFAULT 0,
    quota       INTEGER NOT NULL,
    updated_at  TEXT,
    PRIMARY KEY (persona, budget_date, kind)
);

-- ③ 唤醒条件（C4）。stop_loss/take_profit 仅持仓有效；price_* 观察价；time 定时。
CREATE TABLE IF NOT EXISTS ai_wake_conditions (
    persona    TEXT NOT NULL,
    cond_id    INTEGER NOT NULL,
    code       TEXT NOT NULL,
    kind       TEXT NOT NULL,          -- stop_loss/take_profit/price_above/price_below/time
    threshold  REAL,                   -- 价格阈值（绝对价）
    threshold_pct REAL,                -- 百分比阈值（相对持仓成本；随机对照组用）
    wake_at    TEXT,                   -- kind=time 时用（HH:MM）
    once       INTEGER NOT NULL DEFAULT 1,
    armed      INTEGER NOT NULL DEFAULT 1,
    expires_on TEXT,                   -- 到期日；过期自动失活
    note       TEXT,
    created_at TEXT, updated_at TEXT,
    PRIMARY KEY (persona, cond_id)
);

-- ④ 唤醒事件（口径 4 的幂等锚点）。同 (persona, code, cond_id, trade_date) 只触发一次
--    —— 进程重启后重复扫描不会重复触发，全靠这个主键。
CREATE TABLE IF NOT EXISTS ai_wake_events (
    persona    TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    code       TEXT NOT NULL,
    cond_id    INTEGER NOT NULL,
    hit_at     TEXT,                   -- 命中时刻 HH:MM:SS
    trigger_px REAL,                   -- 命中时的行情价（口径 2：成交另记，触发价≠成交价）
    action     TEXT,                   -- 决策轮产出的动作摘要
    fired      INTEGER NOT NULL DEFAULT 0,   -- 1=真的跑了决策轮（受 wake 配额限制）
    skip_reason TEXT,
    PRIMARY KEY (persona, trade_date, code, cond_id)
);

-- ⑤ 决策记录（幂等锚点 + 归因链）。⚠️ 主键 (persona, signal_date, code)：
--    重跑同一天同一只票不会重复下单；order_id 只在成交后回填。
CREATE TABLE IF NOT EXISTS ai_decisions (
    persona       TEXT NOT NULL,
    signal_date   TEXT NOT NULL,        -- 出决策的交易日 D
    code          TEXT NOT NULL,
    advice_status TEXT,                 -- 可执行/驳回/等待
    direction     TEXT,
    position_size REAL,
    entry REAL, target REAL, stop REAL,
    model         TEXT,                 -- 出决策时实际用的模型（归因）
    analysis_id   INTEGER,              -- 指向 analyses.id
    order_id      INTEGER,              -- 成交后在 account 里的单号
    order_status  TEXT, order_reason TEXT,
    created_at    TEXT,
    PRIMARY KEY (persona, signal_date, code)
);
CREATE INDEX IF NOT EXISTS idx_ai_decisions_date ON ai_decisions(signal_date, persona);

-- ⑥ AI 自述复盘（P1，2026-10-14）。每天每个 AI 一段"对前一交易日的复盘"文本。
--    ⚠️ 与 ai_decisions 分表：那张表承载"交易决策"、这张承载"复盘叙述"，
--    避免同一主键被两类语义争用（本项目已因主键争用踩过一次静默覆盖）。
CREATE TABLE IF NOT EXISTS ai_reviews (
    persona     TEXT NOT NULL,
    trade_date  TEXT NOT NULL,          -- 做出这段复盘的交易日（= 决策轮的 signal_date）
    review_text TEXT,                   -- AI 对**前一交易日**交易的复盘（自述）
    summary     TEXT,                   -- 该轮的总体思路（final.summary）
    model       TEXT,
    created_at  TEXT,
    PRIMARY KEY (persona, trade_date)
);

-- ⑦ 推送留痕（2026-10-23）。为什么需要它：此前**推送完全没有留痕** ⇒
--    无法回答"某天到底推了几条、分别是什么、为什么同一份内容推了两遍"
--    （10-23 实测：手动重放 + 定时任务把同一份复盘推了两次）。
--    content_hash 用于幂等：同一天同一内容只推一次。
-- ⚠️ 唯一索引是 **partial**（只约束 status='ok'）：失败的记录**不能**挡住重试，
--    否则一次网络抖动就会让这条通知永远发不出去。
CREATE TABLE IF NOT EXISTS push_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day TEXT,                 -- 推送日 YYYY-MM-DD（幂等按它判）
    channel TEXT,             -- 钉钉 / 飞书
    title TEXT,
    content_hash TEXT,        -- 正文 SHA1 前 12 位
    content_len INTEGER,
    status TEXT,              -- ok / failed
    detail TEXT,              -- 失败原因
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_push_log_day ON push_log(day, channel);
CREATE UNIQUE INDEX IF NOT EXISTS uniq_push_log_ok
    ON push_log(day, channel, content_hash) WHERE status = 'ok';

-- ⑧ 买入信号队列（2026-10-23）。用途：盘中**确定性判据**命中"买入条件/信号"
--    ⇒ 落库 ⇒ 收盘后拿去做深度分析 ⇒ **只有结论可行（advice_status='可执行'）
--    才推飞书**。用户要的是"达到买入条件后深度分析，可行再通知"，
--    而不是把每条涨幅 5% 都推一遍。
-- ⚠️ 唯一键 (day, code, kind)：同一只票同一天同类信号只记一次
--    （盘中每分钟扫描，重复命中会刷屏）。
CREATE TABLE IF NOT EXISTS signal_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day TEXT NOT NULL,
    code TEXT NOT NULL,
    name TEXT,
    kind TEXT NOT NULL,          -- ma20_breakout 等信号类型
    detail TEXT,                -- 人类可读的触发说明
    price REAL,
    ma20 REAL,
    change_pct REAL,
    relation TEXT,              -- 自选 / 今日候选 / AI武装 / 持仓
    analyzed INTEGER DEFAULT 0, -- 1=已被深度分析覆盖
    created_at TEXT,
    UNIQUE (day, code, kind)
);
CREATE INDEX IF NOT EXISTS idx_signal_day ON signal_events(day, analyzed);

-- ⑨ 推送消息模板（2026-10-23）。用户要"发送消息应可编辑" ⇒ 把**外壳文案**
--    从代码里拿出来，放进库里、在配置页编辑。
--    为什么存**库**而不是 config/*.json：① 与 `scheduled_jobs`（自定义定时任务）
--    一致 —— 用户可编辑的配置本来就该在库里；② 调度器长驻进程与 Web 页面
--    **天然读到同一份**（文件方案要额外处理 mtime 缓存与多实例一致性）；
--    ③ 不脏 git（用户改文案不该产生代码库的 diff）。
--    ⚠️ 表里**只存被改过的**模板；没改过的走代码里的内置默认
--    （⇒ 默认文案升级后，未被用户改过的模板自动跟随新默认）。
CREATE TABLE IF NOT EXISTS push_templates (
    key TEXT PRIMARY KEY,      -- alert / review_summary / position_risk / signal_conclusion
    body TEXT NOT NULL,        -- 模板正文（{{var}} 占位符）
    updated_at TEXT
);

-- ⑩½ AI 助手配置写审计（2026-10-24，P0 整改）：两阶段写守护的落库痕迹。
--    每次 commit 一行，含前后**完整**快照与可回滚描述符——误改/误删可追责、可恢复。
--    保留策略：审计数据**只增不删**（cleanup 不碰），与 push_templates 同类属配置痕迹。
CREATE TABLE IF NOT EXISTS agent_config_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    change_id TEXT NOT NULL,   -- 与写守护的一次 preview 对应；restore 也用它索引
    tool TEXT NOT NULL,        -- tool_push_templates / tool_scheduled_jobs / tool_manage_watchlist
    action TEXT NOT NULL,      -- save/reset/toggle/delete/create/edit/restore:...
    summary TEXT DEFAULT '',
    ok INTEGER DEFAULT 1,
    message TEXT DEFAULT '',
    old_json TEXT,             -- 写之前完整快照（JSON 序列化，可反序列化恢复）
    new_json TEXT,             -- 写之后完整快照
    revert_json TEXT,          -- 可序列化回滚描述符（agent_write_guard._execute 的 op）
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_changes_cid ON agent_config_changes(change_id);

-- ⑩ 市场情绪参照快照（2026-10-23）。一行 = (采集日, 来源) 的**结构化指标卡**。
--    为什么必须落库：情绪比的**绝对值没有意义**，只有"相对自身近 20 日"才有信息量
--    （绝对阈值已被飞书告警证伪：成交额 ≥10 亿对工行恒真、对小盘恒假）。
--    ⇒ 没有历史就没有基准，没有基准这一层就退化成噪声。
--    为什么存 JSON 之外还单列几个数值：相对基准要按 20 日做 SQL 聚合，
--    反序列化全表 20×N 行再算没必要。
--    ⚠️ `quality` 必须进表：低质量样本的"极端分位"不该被当成信号。
CREATE TABLE IF NOT EXISTS sentiment_snapshots (
    trade_date TEXT NOT NULL,
    source TEXT NOT NULL,
    as_of TEXT,
    labeler TEXT,              -- rule / llm:<model>：**口径标识**，见下
    window_start TEXT,
    window_end TEXT,
    n_total INTEGER DEFAULT 0,
    n_scored INTEGER DEFAULT 0,
    bull INTEGER DEFAULT 0,
    bear INTEGER DEFAULT 0,
    neutral INTEGER DEFAULT 0,
    bull_ratio REAL,
    dispersion REAL,
    quality TEXT,
    payload TEXT,
    updated_at TEXT,
    PRIMARY KEY (trade_date, source)
);
CREATE INDEX IF NOT EXISTS idx_sent_date ON sentiment_snapshots(source, trade_date);

-- 逐条标签（2026-10-23）：**这一层真正的原始资产**。
-- ⚠️ 日聚合可以从标签重算，反过来不行。两个标注器（rule / llm）都存，
--    双轨对照才能在任何时点重算，而不必等两天都跑到。
-- ⚠️ `post_key` 用 URL 或正文哈希，**不用下标**：同一批帖子隔天重取顺序会变，
--    下标作键会把"同一条发言"每天算成新记录。
CREATE TABLE IF NOT EXISTS sentiment_labels (
    trade_date TEXT NOT NULL,
    source TEXT NOT NULL,
    post_key TEXT NOT NULL,
    labeler TEXT NOT NULL,
    stance TEXT,
    quote TEXT,
    quote_ok INTEGER,
    why TEXT,
    text TEXT,
    created_at TEXT,
    PRIMARY KEY (trade_date, source, post_key, labeler)
);
CREATE INDEX IF NOT EXISTS idx_sent_label_date ON sentiment_labels(trade_date, source);

-- ===== 外部观点缓存（2026-10-23）=====
-- 为什么不能只用进程内 dict：`data/market_opinions` 原用模块级 `_opinions_cache`，
-- 但常驻进程有 3 个（调度器 / UI / watchdog），**各持一份缓存** ⇒ 同一份观点被
-- 重复购买。实测 2026-10-23 单日 28 次调用（同一份内容买了 24 次，¥0.62），
-- 而它实际只需要 2~3 次。缓存落库才能跨进程共享。
CREATE TABLE IF NOT EXISTS opinion_cache (
    cache_key TEXT PRIMARY KEY,   -- 例：market_opinions:10月23日
    payload TEXT NOT NULL,        -- JSON 原文
    created_at TEXT NOT NULL
);

-- ===== 自选股（双轨：user / ashen，2026-10-27）=====
-- 为什么从 JSON 迁到这里：两套自选必然同时存在（用户一套、阿深一套），
-- 而 JSON 是**整文件替换**语义 ⇒ 两个写入方会互相覆盖对方的清单，并发写
-- 只会留下最后一份（且丢的那一份不会有任何报错）。
-- 主键 `(owner, code)` 天然防重复添加、`INSERT OR REPLACE` 天然幂等。
-- ⚠️ 同一只票可以同时在两套里（用户提名给阿深后大概率如此），这是**正常状态**，
--    不是数据冲突 ⇒ 不能用"全局 code 唯一"来建模。
CREATE TABLE IF NOT EXISTS watchlist_items (
    owner       TEXT NOT NULL,          -- 'user' | 'ashen'
    code        TEXT NOT NULL,
    name        TEXT DEFAULT '',
    note        TEXT DEFAULT '',        -- 用户备注 / 阿深加的"理由"
    source      TEXT DEFAULT 'manual',  -- manual|nominate|research|screen|migrated
    added_at    TEXT NOT NULL,
    updated_at  TEXT,
    PRIMARY KEY (owner, code)
);
CREATE INDEX IF NOT EXISTS idx_wl_owner ON watchlist_items(owner, added_at DESC);

-- ===== 应用级一次性标记（2026-10-27）=====
-- 为什么需要它：数据迁移"只做一次"这件事必须跟**它迁移的那份数据**同生共死。
-- 早期方案把标记写成库旁边的 JSON 文件，结果库被替换/重建时标记还在 ⇒
-- 新库直接跳过迁移，用户的自选股原地消失（且不报错）。
-- 放进库里 ⇒ 删库即重来；单测换临时库时标记自动隔离，不会污染生产。
-- ⚠️ 只存"做没做过"，**不存业务数据**（业务数据各有自己的表）。
CREATE TABLE IF NOT EXISTS app_flags (
    key         TEXT PRIMARY KEY,
    value       TEXT DEFAULT '',
    updated_at  TEXT
);

-- ===== 阿深定时值班轮次（P1，2026-10-27）=====
-- 用途：记录"这一轮该不该跑、跑没跑、花没花 LLM"，是幂等与成本审计的唯一依据。
--
-- ⚠️ 主键 `(persona, trade_date, slot)` 就是**幂等锚点**：
--    盘中的 `scan_once` 每分钟跑一次，到点后的 30 分钟内会被打到很多次；
--    没有这个主键 ⇒ 同一个 slot 一轮跑 N 次 ⇒ 同一只票被买 N 遍。
--    判断"跑过没有"必须查库，不能查内存（重启后内存清零，会重复跑）。
--
-- ⚠️ `status` 与 `llm_called` **必须分开存**：
--    `status='skipped'` 的轮次是常态（6 轮里只有 2~3 轮真调模型），
--    而这些轮次**不占额度**。若把"跳过"和"没跑过"混成一个 NULL，
--    就再也算不清"今天到底叫了几次模型"（成本对账的唯一来源）。
--
-- ⚠️ `data_through` 与 `trade_date` 分开：盘中那几轮的 K 线只到昨收，
--    把"执行日"当"数据截止日"会让事后复盘误以为用的是当日收盘数据。
CREATE TABLE IF NOT EXISTS ai_routine_runs (
    persona       TEXT NOT NULL,
    trade_date    TEXT NOT NULL,          -- 执行日 YYYY-MM-DD
    slot          TEXT NOT NULL,          -- '09:40' / '10:20' / ...
    status        TEXT NOT NULL,          -- ok | skipped | error
    llm_called    INTEGER DEFAULT 0,      -- 是否真的调了模型（1/0，跳过轮为 0）
    skip_reason   TEXT DEFAULT '',        -- 跳过的具体原因（不能写"无"）
    summary       TEXT DEFAULT '',        -- 本轮结论（≤80 字，模型原话）
    actions_json  TEXT,                   -- 通过授权判据后的动作（JSON 数组）
    blocked_json  TEXT,                   -- 被拦下的动作与理由（供审计）
    autonomy      TEXT,                   -- 本轮适用的授权级别（升降级可追溯）
    data_through  TEXT,                   -- K 线最后一根的日期（不是执行日）
    model         TEXT,
    -- 动作闸门的处置结果（2026-10-28）。为什么必须落库而不是只写日志：
    --   "这一轮是被系统补正才动的"与"它自己想到要动"是**两件完全不同的事**，
    --   复盘要能分清。取值形如 `missing_wake→retry_ok` / `no_action→retry_useless
    --   +auto_watch`；空串 = 本轮它自己就给了实质动作（正常）。
    --   没有这一列时，页面只能看到 summary 正常、动作 1 条，看不出是兜底出来的。
    enforced      TEXT DEFAULT '',
    created_at    TEXT,
    PRIMARY KEY (persona, trade_date, slot)
);
CREATE INDEX IF NOT EXISTS idx_routine_day ON ai_routine_runs(trade_date, persona);
"""

# schema 指纹：由建表语句文本派生，改了 schema 就自动变（见 init_store 的说明）。
# 取模是为了落在 SQLite user_version 的 32 位有符号整数范围内。
_SCHEMA_VERSION = int(hashlib.sha1(_SCHEMA.encode("utf-8")).hexdigest(), 16) % (2**31)

# 列迁移只做一次/进程（F1）：`_ensure_column` 要 PRAGMA + ALTER，50+ 处 init_store
# 调用里只有第一处需要真正检查。⚠️ 它**必须在版本短路之前**执行 —— 见 init_store 内注释。
_COLUMNS_CHECKED = False


def _migrate_columns(conn) -> None:
    """列迁移（幂等、**表不存在则跳过**）—— F1 双保险的执行体。

    ⚠️ 为什么表不存在要跳过：全新空库还没有 `screen_picks`，
    `ALTER TABLE ... ADD COLUMN` 会直接抛 `no such table`；
    空库由 `init_store` 里的 `executescript(_SCHEMA)` 建表，**天生带新列**。
    """
    if _table_exists(conn, "ai_wake_conditions"):
        _ensure_column(conn, "ai_wake_conditions", "threshold_pct", "REAL")
    if _table_exists(conn, "paper_nav"):
        _ensure_column(conn, "paper_nav", "at_cost", "INTEGER")
    # B2③（2026-10-18 审查）：PE 允许双口径（东财实时 / 本地快照新浪口径），
    # `screen_picks` 必须记录本次评分用的是哪一个 —— 否则 IC/质量评估混口径。
    if _table_exists(conn, "screen_picks"):
        _ensure_column(conn, "screen_picks", "pe_source", "TEXT")
    # 行业口径隔离（2026-10-29）：`industry` 一直是**新浪 49 类**（供新闻荐股匹配，
    #   `SECTOR_TO_SINA_INDUSTRY` 依赖它），但只覆盖老股；`industry_ts` 存 **tushare
    #   111 类**（全市场 5,577 只，粒度更细，"银行/证券/保险"各自独立），供选股
    #   的行业分散配额用。**两列绝不可混**——同一列里混两套口径，配额会时而用新浪
    #   时而用 tushare，`SECTOR_TO_SINA_INDUSTRY` 也会静默失效（铁律"行业口径不可混"）。
    if _table_exists(conn, "market_snapshot"):
        _ensure_column(conn, "market_snapshot", "industry_ts", "TEXT")
    # 情绪口径（2026-10-23）：同一来源同一天，词表标注与模型标注算出的
    # `bull_ratio` **不是同一个量**。缺这一列，历史表里两种行完全一样，
    # 近 20 日基准会把两把尺子混着平均。旧行补成 'rule'（此前只有词表）。
    if _table_exists(conn, "sentiment_snapshots"):
        _ensure_column(conn, "sentiment_snapshots", "labeler", "TEXT")
    # 复盘窗口（2026-10-20）：同一张表先承载 T+5 早期结论、T+20 到期后覆盖为完整版。
    # 没有它就无法区分"5 天窗口的判对"与"20 天窗口的判对"——两类结论混在一起，
    # 归因分布会被早期样本污染（与 PE 双口径同一个坑）。
    if _table_exists(conn, "analysis_reviews"):
        _ensure_column(conn, "analysis_reviews", "horizon", "INTEGER")
    # 择时口径（2026-10-20）：分析与选股统一到 MarketDetector 后，`market_score`
    # 的含义变了（之前是 TimingEngine 的 16.9 量级，之后是 MarketDetector 的 37.9 量级）。
    # 加一列记录来源，否则"按分数筛历史样本"会把两把尺子混着用。
    if _table_exists(conn, "analyses"):
        _ensure_column(conn, "analyses", "market_score_source", "TEXT")
    # 金额（2026-10-23）：`llm_calls` 一直只有 token，没有钱。
    # token 不能代替金额——qwen3.7-flash 与 deepseek-v4-pro 同一 token 差 22 倍，
    # 只看 token 会得出"今天用量翻倍"而实际更便宜的反向结论。
    # 落库值是**调用发生时的价格**，改价不重算历史（会计口径），需要重算用 recalc。
    if _table_exists(conn, "llm_calls"):
        _ensure_column(conn, "llm_calls", "cost_cny", "REAL")
    if _table_exists(conn, "analyses"):
        _ensure_analyses_unique(conn)
    # 限价单与滑点归因（2026-10-27）：`paper_orders` 只有"次日开盘市价成交"，
    # 于是"跌到 15.80 才买"这个意图无法表达，实际成交价变成开盘价——想避免的追价
    # 反而被强制执行。加 `limit_price` 让开盘价高于限价即撤单，同时落
    # `decision_price/slippage`：**滑点是判断一个策略到底有没有判断力的唯一硬数据**，
    # 比任何复盘文字都可信（长期为负说明挂单阈值设得太紧）。
    if _table_exists(conn, "paper_orders"):
        _ensure_column(conn, "paper_orders", "limit_price", "REAL")
        _ensure_column(conn, "paper_orders", "decision_price", "REAL")
        _ensure_column(conn, "paper_orders", "slippage", "REAL")
    # 动作闸门（2026-10-28）：prompt 里"说在等就必须挂 set_wake"是软约束，
    # 实测六轮全部 pass、0 条条件 ⇒ 判定搬进代码后需要记录"补正有没有生效"。
    # 旧行为空串 = 当年没有闸门（不是"它自己动的"）—— 这个区别在复盘里要说清。
    if _table_exists(conn, "ai_routine_runs"):
        _ensure_column(conn, "ai_routine_runs", "enforced", "TEXT")


def _ensure_analyses_unique(conn) -> None:
    """`analyses` 加 `(analysis_date, stock_code)` 唯一索引（**先归档去重再建**）。

    为什么（2026-10-19）：同一天同一只票被重复分析会变成多个"样本"，
    把胜率统计放大 —— 实测 `2026-10-08 603127` 有 **4 行**（其中两对完全相同），
    于是 `_analysis_quality` 把 1 只票算成 4 个样本，得出"看空 4/4 全对"。

    ⚠️ 不能把唯一索引写进 `_SCHEMA`：`executescript` 遇到已存在的重复行会**整体失败**，
    连带建表语句一起中断（长期运行的库必然有历史重复）。
    顺序必须是：先归档到备份表 → 再去重 → 最后建索引。
    """
    try:
        dup_groups = conn.execute(
            """SELECT COUNT(*) FROM (SELECT analysis_date, stock_code
                 FROM analyses GROUP BY analysis_date, stock_code
                 HAVING COUNT(*) > 1)""").fetchone()[0]
        if dup_groups:
            # 留一份原件（可回溯"当时到底重复了什么"），再按 keep=latest 去重
            conn.execute("CREATE TABLE IF NOT EXISTS analyses_dup_backup_20261019 AS "
                         "SELECT * FROM analyses")
            conn.execute(
                """DELETE FROM analyses WHERE id NOT IN (
                     SELECT MAX(id) FROM analyses GROUP BY analysis_date, stock_code)""")
            conn.commit()
            _logger.warning("[存储] analyses 去重：%d 组重复已合并（原件见 "
                            "analyses_dup_backup_20261019）", dup_groups)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uniq_ana_date_code "
                     "ON analyses(analysis_date, stock_code)")
        conn.commit()
    except Exception as e:                                  # noqa: BLE001
        # 索引建不上不能让 init_store 失败（否则整个应用起不来）；
        # 但必须响亮地记下来 —— 没有它，同日重复分析会继续污染统计。
        _logger.warning("[存储] analyses 唯一索引创建失败（重复样本可能继续存在）: %s", e)


def _table_exists(conn, table: str) -> bool:
    r = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                     (table,)).fetchone()
    return r is not None


def init_store(force: bool = False):
    """初始化存储表（幂等）

    force=True 时跳过短路、无条件重放建表语句。用于"怀疑表结构损坏/被误删"
    的人工修复场景——正常情况下不需要。

    短路：库已经建过且 schema 没变时直接返回。
    本项目是"每次调用开关一次连接"的短连接模式，而全仓有 50+ 处调用本函数
    （每个写函数开头都调一次）。在新建库上 executescript 建 20+ 张表实测
    1.6s 量级（Windows + 实时防护），重复执行虽只有毫秒级，但对连续调用场景
    （批量导入、测试夹具）是纯浪费。

    版本号由 schema 文本哈希派生：**改了 schema 自动失效**，杜绝"改表忘了 bump
    版本号"这类典型事故。哈希存在 SQLite 文件头的 user_version 里，
    判断只需读文件头，不做任何建表 IO。
    """
    conn = get_connection()
    try:
        # ⚠️⚠️ F1（2026-10-18 审查）：列迁移必须**先于版本短路**跑一次。
        #    `pe_source` 当天只加在 `_ensure_column`、没改 `_SCHEMA` ⇒ 哈希不变 ⇒
        #    长期运行的旧库在"版本号相等"处早退，迁移**永远不会执行**，
        #    而 `insert_screen_picks` 的 INSERT 已带新列 ⇒ 明天开盘第一次选股
        #    就会把全部样本静默丢光（异常被自己的 except 吞掉，只剩一条 warning）。
        #    双保险：① 本块（短路前）+ executescript 后各跑一次 `_migrate_columns`
        #    （幂等、表不存在则跳过——全新空库由下面的建表语句建，天生带新列）；
        #    ② 新列同步写进 `_SCHEMA`（哈希变化 ⇒ 旧库必然完整走一遍迁移）。
        global _COLUMNS_CHECKED
        if not _COLUMNS_CHECKED or force:
            _migrate_columns(conn)
            _COLUMNS_CHECKED = True
        if not force:
            row = conn.execute("PRAGMA user_version").fetchone()
            if row and row[0] == _SCHEMA_VERSION:
                return
        conn.executescript(_SCHEMA)
        # ⚠️ 迁移守卫的**唯一正确用法**（2026-10-14 实战）：
        #    CREATE TABLE IF NOT EXISTS 只对新表生效；给**已存在**的表加列必须显式 ALTER。
        #    不加这一步，长期运行的库（含真实库）会永远缺这一列，
        #    而哈希版本号却已"迁移成功"——静默缺列是最难发现的一类。
        _migrate_columns(conn)
        # ⚠️ 唤醒条件的**业务幂等键** = (persona, code, kind)（2026-10-16 修订）。
        #    单独执行 + 容错：历史库可能残留重复行（旧实现"永远 INSERT"的产物），
        #    那时建唯一索引会抛 IntegrityError —— **绝不能因为一条索引让整个建表脚本中断**
        #    （应用层已改为"同键复用同一 cond_id"的 UPDATE 语义，功能不依赖该索引）。
        try:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_wake_cond_key "
                         "ON ai_wake_conditions(persona, code, kind)")
        except sqlite3.IntegrityError as e:                  # noqa: PERF203
            _logger.warning("[store] ai_wake_conditions 有重复的 (persona, code, kind) ⇒ "
                            "唯一索引未建（需先清理重复行）: %s", e)
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        conn.commit()
    finally:
        conn.close()


# ================================================================
# 新闻条目
# ================================================================

def insert_news(items: list[dict]) -> int:
    """
    批量插入新闻（hash 去重）。

    items: [{hash, title, content, source, published_at, related_codes(list), sentiment}]
    Returns: 新插入条数（重复忽略）
    """
    if not items:
        return 0
    init_store()
    conn = get_connection()
    new_count = 0
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for item in items:
            try:
                conn.execute(
                    """INSERT INTO news_items
                       (hash, title, content, source, published_at, related_codes, sentiment, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (item["hash"], item.get("title", ""), item.get("content", ""),
                     item.get("source", ""), item.get("published_at", ""),
                     json.dumps(item.get("related_codes", []), ensure_ascii=False),
                     item.get("sentiment", "neutral"), now))
                new_count += 1
            except sqlite3.IntegrityError:
                continue  # hash 重复
        conn.commit()
    finally:
        conn.close()
    return new_count


def get_last_run(run_type: str) -> dict | None:
    """最近一次某类型的运行记录（`summary` 已解析成 dict）；没有则 None。

    用途：把"任务跑了但内部有失败"暴露到页面上。例如
    `job:backfill_analysis` 的 `ok=False`（复盘全废）原先只写日志，
    页面上完全看不出来（2026-10-20 审查 B3）。
    """
    init_store()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT run_type, summary, created_at FROM runs "
            "WHERE run_type = ? ORDER BY id DESC LIMIT 1", (run_type,)).fetchone()
        if not row:
            return None
        try:
            summary = json.loads(row["summary"] or "{}")
        except (ValueError, TypeError) as e:       # noqa: BLE001
            _logger.warning("运行记录 summary 不可解析（%s）：%s", run_type, e)
            summary = {}
        return {"run_type": row["run_type"], "created_at": row["created_at"],
                "summary": summary if isinstance(summary, dict) else {}}
    except Exception as e:                          # noqa: BLE001
        _logger.warning("读取最近运行记录失败（%s）：%s", run_type, e)
        return None
    finally:
        conn.close()


def get_news_latest() -> tuple[int, str]:
    """最新一条新闻的 `(id, created_at)`；表空时返回 `(0, "")`。

    两个用途共用一次查询：
    1. **页面缓存的版本号**（`st.cache_data` 的 key 组成部分）——`news_items`
       只增不删、id 单调递增，所以"id 变了"⇔"有新新闻入库"。这样采集进程
       一写入，页面缓存立刻失效，不需要等 TTL 过期。
       ⚠️ 为什么不用 `max(created_at)` 当版本号：那列**没有索引**，要扫全表
       （6956 行）；id 是主键，O(1)。这个函数每次页面渲染都会调，必须够便宜。
    2. **页面上的「数据截止」展示**——用户看到 23:05 就知道不是"没采集"，
       而不是刷新一次才发现"哦有了"（2026-10-20 用户反馈"新闻没采集"，
       实际是 120 秒 TTL 内看到的是旧快照）。
    """
    init_store()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, created_at FROM news_items ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return 0, ""
        return int(row["id"] or 0), str(row["created_at"] or "")
    except Exception as e:                      # noqa: BLE001
        _logger.warning("读取新闻最新记录失败（页面缓存将退化为 ttl 行为）：%s", e)
        return 0, ""
    finally:
        conn.close()


def get_recent_news(limit: int = 20, keyword: str = None, code: str = None,
                    hours: int = 72, name: str = None) -> list[dict]:
    """最近新闻。可按关键词（标题/内容 LIKE）或关联股票过滤，按发布时间倒序

    v2（M0 0.2.3）：新增 name 参数。入库时的 related_codes 只覆盖自选股
    （见 data/news_feed._enrich），因此全市场候选必须能按股票简称在标题/正文里
    匹配，否则选股的"新闻正面率"因子对非自选股永远是 unknown。
    code 与 name 之间是 **OR** 关系：命中关联代码或正文提及简称都算该股新闻。
    """
    init_store()
    conn = get_connection()
    try:
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        sql = "SELECT * FROM news_items WHERE published_at >= ?"
        params: list = [since]
        if keyword:
            sql += " AND (title LIKE ? OR content LIKE ?)"
            params += [f"%{keyword}%", f"%{keyword}%"]
        stock_clauses: list = []
        if code:
            stock_clauses.append("related_codes LIKE ?")
            params.append(f'%"{code}"%')
        clean_name = str(name).strip() if name else ""
        if len(clean_name) >= 2:
            stock_clauses.append("(title LIKE ? OR content LIKE ?)")
            params += [f"%{clean_name}%", f"%{clean_name}%"]
        if stock_clauses:
            sql += " AND (" + " OR ".join(stock_clauses) + ")"
        sql += " ORDER BY published_at DESC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["related_codes"] = json.loads(d.get("related_codes") or "[]")
            except (TypeError, json.JSONDecodeError):
                d["related_codes"] = []
            result.append(d)
        return result
    finally:
        conn.close()


def news_stats() -> dict:
    """新闻表统计（配置页数据卡）"""
    init_store()
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM news_items").fetchone()[0]
        latest = conn.execute(
            "SELECT MAX(published_at) FROM news_items").fetchone()[0]
        today = datetime.now().strftime("%Y-%m-%d")
        today_count = conn.execute(
            "SELECT COUNT(*) FROM news_items WHERE published_at >= ?",
            (today,)).fetchone()[0]
        return {"total": total, "today": today_count, "latest": latest}
    finally:
        conn.close()


# ================================================================
# 对话日志
# ================================================================

def insert_chat_log(session_id: str, role: str, content: str,
                    tools_used: list[str] = None,
                    prompt_tokens: int = None, completion_tokens: int = None):
    """写入一条对话消息，并同步维护 chat_sessions 会话维表。

    同时维护两维是"历史可读"的前提：以前只有消息维（chat_logs），
    没有会话维，列表无从下手；这里保证任何调用方写消息都会顺带登记会话。
    """
    init_store()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO chat_logs (session_id, role, content, tools_used,
               prompt_tokens, completion_tokens, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, role, content,
             json.dumps(tools_used or [], ensure_ascii=False),
             prompt_tokens, completion_tokens, now))
        if session_id:
            conn.execute(
                """INSERT OR IGNORE INTO chat_sessions
                   (session_id, title, source, message_count,
                    prompt_tokens, completion_tokens, created_at, updated_at)
                   VALUES (?, '', 'ui', 0, 0, 0, ?, ?)""", (session_id, now, now))
            conn.execute(
                """UPDATE chat_sessions SET updated_at = ?,
                   message_count = COALESCE(message_count, 0) + 1,
                   prompt_tokens = COALESCE(prompt_tokens, 0) + COALESCE(?, 0),
                   completion_tokens = COALESCE(completion_tokens, 0) + COALESCE(?, 0)
                   WHERE session_id = ?""",
                (now, prompt_tokens, completion_tokens, session_id))
        conn.commit()
    finally:
        conn.close()


# ================================================================
# 对话会话维表（P1）
# ================================================================

def create_chat_session(session_id: str, title: str = "", persona: str = None,
                        model: str = None, source: str = "ui") -> int:
    """登记会话（已存在则跳过）。返回 1=新建 / 0=已存在或 session_id 为空"""
    if not session_id:
        return 0
    init_store()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT OR IGNORE INTO chat_sessions
               (session_id, title, persona, model, source, message_count,
                prompt_tokens, completion_tokens, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 0, 0, 0, ?, ?)""",
            (session_id, title or "", persona, model, source or "ui", now, now))
        conn.commit()
        return 1 if cur.rowcount else 0
    finally:
        conn.close()


def update_chat_session(session_id: str, fields: dict):
    """更新会话元信息（白名单：title/persona/model/source）"""
    if not session_id or not fields:
        return
    allowed = {"title", "persona", "model", "source"}
    init_store()
    conn = get_connection()
    try:
        sets, params = ["updated_at = ?"], [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k} = ?")
                params.append(v)
        if len(sets) == 1:
            return
        params.append(session_id)
        conn.execute(f"UPDATE chat_sessions SET {', '.join(sets)} WHERE session_id = ?", params)
        conn.commit()
    finally:
        conn.close()


def list_chat_sessions(limit: int = 30) -> list[dict]:
    """会话列表（最近更新在前）。

    消息数与首条提问都实时从 chat_logs 聚合，不迷信维表里的计数器——
    历史行可能由旧版本写入（计数器为 0）。
    顺带把"只有 chat_logs、没有会话维表"的老会话补登记（source='legacy'），
    否则升级前的历史永远列不出来。
    """
    init_store()
    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR IGNORE INTO chat_sessions
               (session_id, title, source, message_count, created_at, updated_at)
               SELECT l.session_id,
                      COALESCE((SELECT substr(l2.content, 1, 18) FROM chat_logs l2
                                WHERE l2.session_id = l.session_id
                                ORDER BY l2.id LIMIT 1), '历史会话'),
                      'legacy', 0, MIN(l.created_at), MAX(l.created_at)
               FROM chat_logs l
               WHERE l.session_id IS NOT NULL AND l.session_id <> ''
               GROUP BY l.session_id""")
        conn.commit()
        rows = conn.execute(
            """SELECT s.session_id, s.title, s.persona, s.model, s.source,
                      s.created_at, s.updated_at,
                      s.prompt_tokens, s.completion_tokens,
                      (SELECT COUNT(*) FROM chat_logs l
                        WHERE l.session_id = s.session_id) AS message_count,
                      (SELECT substr(l.content, 1, 18) FROM chat_logs l
                        WHERE l.session_id = s.session_id AND l.role = 'user'
                        ORDER BY l.id LIMIT 1) AS first_msg
               FROM chat_sessions s
               ORDER BY s.updated_at DESC, s.id DESC LIMIT ?""",
            (int(limit),)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_chat_session(session_id: str) -> dict | None:
    """取单个会话元信息（含实时消息数）"""
    if not session_id:
        return None
    init_store()
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT s.*, (SELECT COUNT(*) FROM chat_logs l
                            WHERE l.session_id = s.session_id) AS message_count
               FROM chat_sessions s WHERE s.session_id = ?""",
            (session_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_chat_messages(session_id: str, limit: int = 200) -> list[dict]:
    """取会话消息（正序）。limit 限制的是"最近 N 条"，不是最早 N 条"""
    if not session_id:
        return []
    init_store()
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT id, role, content, tools_used, prompt_tokens,
                      completion_tokens, created_at
               FROM chat_logs WHERE session_id = ?
               ORDER BY id DESC LIMIT ?""", (session_id, int(limit))).fetchall()
        out = []
        for r in rows[::-1]:
            d = dict(r)
            try:
                d["tools_used"] = json.loads(d.get("tools_used") or "[]")
            except (TypeError, json.JSONDecodeError):
                d["tools_used"] = []
            out.append(d)
        return out
    finally:
        conn.close()


def clear_chat_messages(session_id: str) -> int:
    """清空会话消息（保留会话本身），返回删除条数"""
    if not session_id:
        return 0
    init_store()
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM chat_logs WHERE session_id = ?", (session_id,))
        conn.execute(
            """UPDATE chat_sessions SET message_count = 0, prompt_tokens = 0,
               completion_tokens = 0, updated_at = ? WHERE session_id = ?""",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session_id))
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()


def delete_chat_session(session_id: str) -> int:
    """删除会话及其全部消息，返回被删消息条数"""
    if not session_id:
        return 0
    init_store()
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM chat_logs WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()


def chat_stats(recent_days: int = 30) -> dict:
    """对话维度统计（配置页用）：会话数 / 消息数 / token 累计"""
    init_store()
    conn = get_connection()
    try:
        cutoff = (datetime.now() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
        sessions = conn.execute("SELECT COUNT(*) FROM chat_sessions").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM chat_logs").fetchone()[0]
        tok = conn.execute(
            "SELECT COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0) "
            "FROM chat_logs").fetchone()
        recent = conn.execute(
            """SELECT COUNT(*) FROM chat_sessions
               WHERE updated_at >= ?""", (cutoff,)).fetchone()[0]
        return {
            "recent_days": recent_days,
            "sessions": int(sessions or 0),
            "messages": int(messages or 0),
            "prompt_tokens": int(tok[0] or 0),
            "completion_tokens": int(tok[1] or 0),
            "recent_sessions": int(recent or 0),
        }
    finally:
        conn.close()


# ================================================================
# 相似历史日特征向量
# ================================================================

def upsert_daily_feature(symbol: str, date: str, vector: dict, close: float,
                         future_ret_5d: float = None, future_ret_20d: float = None):
    """写入/更新某标的某日的特征向量（JSON 存储）"""
    init_store()
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO daily_features (symbol, date, vector, close,
               future_ret_5d, future_ret_20d, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(symbol, date) DO UPDATE SET
                 vector=excluded.vector, close=excluded.close,
                 future_ret_5d=excluded.future_ret_5d,
                 future_ret_20d=excluded.future_ret_20d,
                 updated_at=excluded.updated_at""",
            (symbol, date, json.dumps(vector, ensure_ascii=False), close,
             future_ret_5d, future_ret_20d,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
    finally:
        conn.close()


def load_features(symbol: str) -> list[dict]:
    """读取标的全历史特征（按日期升序）。返回 [{date, vector(dict), close, future_ret_5d, future_ret_20d}]"""
    init_store()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT date, vector, close, future_ret_5d, future_ret_20d "
            "FROM daily_features WHERE symbol = ? ORDER BY date", (symbol,)).fetchall()
        out = []
        for r in rows:
            try:
                vector = json.loads(r["vector"])
            except (TypeError, json.JSONDecodeError):
                continue
            out.append({"date": r["date"], "vector": vector, "close": r["close"],
                        "future_ret_5d": r["future_ret_5d"], "future_ret_20d": r["future_ret_20d"]})
        return out
    finally:
        conn.close()


def feature_stats() -> dict:
    init_store()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT symbol, COUNT(*), MIN(date), MAX(date) FROM daily_features "
            "GROUP BY symbol").fetchall()
        return {r[0]: {"days": r[1], "from": r[2], "to": r[3]} for r in rows}
    finally:
        conn.close()


# ================================================================
# 运行记录（回测/流水线元数据）
# ================================================================

def insert_run(run_type: str, summary: dict):
    """记录一次运行（回测/流水线）的摘要元数据；payload 为 dict"""
    init_store()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO runs (run_type, summary, created_at) VALUES (?, ?, ?)",
            (run_type, json.dumps(summary, ensure_ascii=False, default=str),
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
    except Exception as e:
        _logger.warning("[存储] insert_run 失败: %s", e)
    finally:
        conn.close()


# ================================================================
# 预测跟踪：复盘观察点（"明日观察"落库 → 次日核对 → 命中率）
# ================================================================

def insert_observations(review_date: str, items: list[dict]) -> int:
    """保存一期复盘的观察点。items: [{content, kind}]，kind=观察点|风险"""
    init_store()
    conn = get_connection()
    inserted = 0
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for it in items:
            # 同一复盘日期同一内容不重复入库
            exists = conn.execute(
                "SELECT 1 FROM observations WHERE review_date = ? AND content = ?",
                (review_date, it["content"])).fetchone()
            if exists:
                continue
            conn.execute(
                "INSERT INTO observations (review_date, content, kind, created_at) VALUES (?, ?, ?, ?)",
                (review_date, it["content"], it.get("kind", "观察点"), now))
            inserted += 1
        conn.commit()
    finally:
        conn.close()
    return inserted


def get_unverified_observations(limit_days: int = 3) -> list[dict]:
    """取此前数日内未核对的观察点（次日复盘核对用；排除当日自己，避免自我核对）"""
    init_store()
    conn = get_connection()
    try:
        cutoff = (datetime.now() - timedelta(days=limit_days)).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")
        rows = conn.execute(
            "SELECT id, review_date, content, kind FROM observations "
            "WHERE result IS NULL AND review_date >= ? AND review_date < ? "
            "ORDER BY review_date, id",
            (cutoff, today)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_observation_result(obs_id: int, result: str, verified_date: str):
    """回填核对结果。result: 达成|未发生|部分达成|无法验证"""
    init_store()
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE observations SET result = ?, verified_date = ? WHERE id = ?",
            (result, verified_date, obs_id))
        conn.commit()
    finally:
        conn.close()


def observation_stats(recent_days: int = 30) -> dict:
    """观察点命中统计（近 N 天核对的）"""
    init_store()
    conn = get_connection()
    try:
        cutoff = (datetime.now() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
        rows = conn.execute(
            "SELECT result, COUNT(*) FROM observations "
            "WHERE verified_date >= ? GROUP BY result", (cutoff,)).fetchall()
        counts = {r[0]: r[1] for r in rows}
        verified = sum(counts.values())
        hit = counts.get("达成", 0) + counts.get("部分达成", 0)
        return {"verified": verified, "达成": counts.get("达成", 0),
                "部分达成": counts.get("部分达成", 0), "未发生": counts.get("未发生", 0),
                "无法验证": counts.get("无法验证", 0),
                "命中率": round(hit / verified, 2) if verified else None}
    finally:
        conn.close()


# ================================================================
# 盘中异动预警（Phase 1）
# ================================================================

def insert_signal(day: str, code: str, name: str, kind: str, detail: str,
                  price: float = None, ma20: float = None, change_pct: float = None,
                  relation: str = "") -> bool:
    """记录一条买入信号。**幂等**（同一天同票同类型只记一次）。

    Returns: True = 新记录；False = 已存在（不是错误）。
    """
    if not code or not kind:
        return False
    init_store()
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO signal_events (day, code, name, kind, detail, price, "
            "ma20, change_pct, relation, analyzed, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
            (str(day)[:10], str(code), str(name or ""), str(kind), str(detail or ""),
             price, ma20, change_pct, str(relation or ""),
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        return bool(cur.rowcount)
    finally:
        conn.close()


def list_signals(day: str, analyzed_only: bool = False) -> list[dict]:
    """当日买入信号（按时间升序）。`analyzed_only=True` 只取尚未被深度分析覆盖的。"""
    init_store()
    conn = get_connection()
    try:
        sql = ("SELECT id, day, code, name, kind, detail, price, ma20, change_pct, "
               "relation, analyzed, created_at FROM signal_events WHERE day = ?")
        if analyzed_only:
            sql += " AND analyzed = 0"
        sql += " ORDER BY id"
        rows = conn.execute(sql, (str(day)[:10],)).fetchall()
    finally:
        conn.close()
    keys = ("id", "day", "code", "name", "kind", "detail", "price", "ma20",
            "change_pct", "relation", "analyzed", "created_at")
    return [dict(zip(keys, r)) for r in rows]


def mark_signals_analyzed(day: str, codes: list) -> int:
    """把已纳入深度分析的信号标记为 analyzed=1（幂等）。"""
    codes = [str(c) for c in (codes or []) if c]
    if not codes:
        return 0
    init_store()
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE signal_events SET analyzed = 1 WHERE day = ? AND code IN "
            f"({','.join('?' * len(codes))})", [str(day)[:10]] + codes)
        conn.commit()
        return int(cur.rowcount or 0)
    finally:
        conn.close()


def list_push_templates() -> dict:
    """用户改过的推送模板：`{key: body}`。没有记录返回 {}（⇒ 全部走内置默认）。

    ⚠️ 只返回**被改过的** —— 默认文案的升级要能自动跟随代码里的新默认。
    """
    init_store()
    conn = get_connection()
    try:
        rows = conn.execute("SELECT key, body FROM push_templates").fetchall()
    finally:
        conn.close()
    return {str(r["key"]): str(r["body"]) for r in rows if r["key"]}


def set_push_template(key: str, body: str) -> bool:
    """保存一条用户自定义模板（UPSERT）。返回是否写入成功。"""
    if not key:
        return False
    init_store()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO push_templates (key, body, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET body=excluded.body, updated_at=excluded.updated_at",
            (str(key), str(body or ""), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        return True
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[存储] 保存推送模板失败: %s", e)
        return False
    finally:
        conn.close()


def delete_push_template(key: str) -> bool:
    """删除用户覆盖 ⇒ 恢复内置默认。"""
    if not key:
        return False
    init_store()
    conn = get_connection()
    try:
        conn.execute("DELETE FROM push_templates WHERE key = ?", (str(key),))
        conn.commit()
        return True
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[存储] 删除推送模板失败: %s", e)
        return False
    finally:
        conn.close()


# ================================================================
# AI 助手配置写审计（2026-10-24，core/agent_write_guard 的唯一存储入口）
# ================================================================

def save_agent_change(rec: dict) -> int | None:
    """落一行写审计。返回自增 id；失败返回 None（调用方记日志，不抛）。"""
    init_store()
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO agent_config_changes (change_id, tool, action, summary, ok,"
            " message, old_json, new_json, revert_json, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (str(rec.get("change_id") or ""), str(rec.get("tool") or ""),
             str(rec.get("action") or ""), str(rec.get("summary") or ""),
             int(rec.get("ok", 1)), str(rec.get("message") or ""),
             rec.get("old_json"), rec.get("new_json"), rec.get("revert_json"),
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        return int(cur.lastrowid)
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[存储] 写审计落库失败: %s", e)
        return None
    finally:
        conn.close()


def get_agent_change(change_id: str) -> dict | None:
    """按 change_id 取最近一次审计行（含前后快照与 revert 描述符）。"""
    init_store()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM agent_config_changes WHERE change_id = ? ORDER BY id DESC LIMIT 1",
            (str(change_id),)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_agent_changes(*, limit: int = 50) -> list[dict]:
    """配置变更审计流水（`agent_config_changes`，倒序）。

    `change_id` 是回滚锚点：`xxx-restore` 行表示对 `xxx` 的回滚动作本身。
    **不要在这里做"是否已回滚"的推断** —— 同一 change_id 可能被回滚多次，
    真实状态得看 `revert_json` 与后续同 change_id 的行。
    """
    init_store()
    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT id, change_id, tool, action, summary, ok, message, created_at "
            "FROM agent_config_changes ORDER BY id DESC LIMIT ?", (int(limit),))]
    finally:
        conn.close()


def list_push_log(*, limit: int = 50) -> list[dict]:
    """推送流水（`push_log`）。`status=ok` 是幂等锚点：同 (day, channel, hash) 只发一次。"""
    init_store()
    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT day, channel, title, content_hash, content_len, status, "
            "detail, created_at FROM push_log ORDER BY id DESC LIMIT ?",
            (int(limit),))]
    finally:
        conn.close()


def sentiment_label_summary(*, days: int = 30) -> dict:
    """情绪标注留痕的**按日聚合**（不逐条列，50 条原文会把页面撑爆）。

    返回每日 `{date, total, by_stance, by_labeler}`，外加全量 `by_stance`。
    ⚠️ `stance='unscored'` 是"判不出方向"，不是"中性" —— 单独计数、不进任何分子。
    """
    init_store()
    conn = get_connection()
    try:
        per_day = [dict(r) for r in conn.execute(
            "SELECT trade_date AS date, COUNT(*) AS total, "
            "SUM(CASE WHEN stance='bull' THEN 1 ELSE 0 END) AS bull, "
            "SUM(CASE WHEN stance='bear' THEN 1 ELSE 0 END) AS bear, "
            "SUM(CASE WHEN stance='neutral' THEN 1 ELSE 0 END) AS neutral, "
            "SUM(CASE WHEN stance='unscored' THEN 1 ELSE 0 END) AS unscored "
            "FROM sentiment_labels GROUP BY trade_date "
            "ORDER BY trade_date DESC LIMIT ?", (int(days),))]
        by_labeler = [dict(r) for r in conn.execute(
            "SELECT labeler, COUNT(*) AS n FROM sentiment_labels "
            "GROUP BY labeler ORDER BY n DESC")]
        total = conn.execute("SELECT COUNT(*) FROM sentiment_labels").fetchone()[0]
        return {"total": total, "per_day": per_day, "by_labeler": by_labeler}
    finally:
        conn.close()


def list_active_positions() -> list[dict]:
    """活跃持仓摘要（**纯 SQL 读表，不取价**）：code / name / quantity / stop_loss / take_profit。

    ⚠️ 为什么不走 `PortfolioManager.get_active_positions()`：那条路径是
    `load()` → `_price_of(code, None)` → **逐只联网取价**（实测单只 ≈30s，
    且取数受限时还要重试 + 指数退避）。盯盘是**每分钟**调一次的 ⇒ 每轮白等几十秒。
    持仓的风控位（止损/止盈）是**库里存的数**，与现价无关 ⇒ 读表即可。

    ⚠️ 不返回市值/占比：占比要总资产（需要现金 + 逐只市值），由
    `core.position_risk.build_holdings` 负责（它显式传 prices 避免联网）。
    """
    init_store()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT code, name, quantity, stop_loss, take_profit, entry_price "
            "FROM positions WHERE status = 'active'").fetchall()
    finally:
        conn.close()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        out.append({
            "code": str(d.get("code") or "").strip(),
            "name": d.get("name") or "",
            "quantity": d.get("quantity"),
            "stop_loss": d.get("stop_loss"),
            "take_profit": d.get("take_profit"),
            "entry_price": d.get("entry_price"),
        })
    return [d for d in out if d["code"]]


def push_already_sent(day: str, channel: str, content_hash: str) -> bool:
    """该内容今天是否已经**成功**推过（幂等判据）。

    ⚠️ 只认 `status='ok'`：失败记录不算"已推"，否则一次网络抖动会让这条通知
    永远发不出去（唯一索引也只约束 ok 行，与此口径一致）。
    """
    if not content_hash:
        return False
    init_store()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM push_log WHERE day = ? AND channel = ? AND content_hash = ? "
            "AND status = 'ok' LIMIT 1",
            (str(day)[:10], str(channel or ""), str(content_hash))).fetchone()
    finally:
        conn.close()
    return row is not None


def log_push(day: str, channel: str, title: str, content_hash: str,
             content_len: int, status: str, detail: str = "") -> None:
    """写一条推送留痕。**失败不能影响推送主流程**（记日志即可）。"""
    try:
        init_store()
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO push_log (day, channel, title, content_hash, content_len, "
                "status, detail, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (str(day)[:10], str(channel or ""), str(title or ""), str(content_hash or ""),
                 int(content_len or 0), str(status or ""), str(detail or ""),
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[存储] 写推送留痕失败（不影响推送）: %s", e)


def insert_alerts(alerts: list, trade_date: str = None) -> int:
    """保存盘中异动预警。alerts: [{code,name,rule,level,message,price,change_pct,quote_time,source}]

    同一交易日、同标的、同规则、同消息只记一次——轮询是每分钟一次，
    否则盘中会刷出一堆重复行。
    """
    if not alerts:
        return 0
    init_store()
    conn = get_connection()
    inserted = 0
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        day = trade_date or now[:10]
        for a in alerts:
            exists = conn.execute(
                "SELECT 1 FROM alerts WHERE trade_date = ? AND code = ? AND rule = ? "
                "AND message = ?", (day, a.get("code"), a.get("rule"), a.get("message"))
            ).fetchone()
            if exists:
                continue
            conn.execute(
                "INSERT INTO alerts (trade_date, created_at, code, name, rule, level, "
                "message, price, change_pct, quote_time, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (day, now, a.get("code"), a.get("name"), a.get("rule"), a.get("level"),
                 a.get("message"), a.get("price"), a.get("change_pct"),
                 a.get("quote_time"), a.get("source")))
            inserted += 1
        conn.commit()
    finally:
        conn.close()
    return inserted


def get_recent_alerts(limit: int = 50, trade_date: str = None) -> list:
    """最近预警（默认当日），按时间倒序"""
    init_store()
    conn = get_connection()
    try:
        day = trade_date or datetime.now().strftime("%Y-%m-%d")
        rows = conn.execute(
            "SELECT * FROM alerts WHERE trade_date = ? ORDER BY created_at DESC, id DESC "
            "LIMIT ?", (day, int(limit))).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ================================================================
# 模拟交易账户（Phase 2）
# ================================================================

_PAPER_TABLES = ("paper_accounts", "paper_positions", "paper_orders",
                 "paper_fills", "paper_nav")


def save_paper_state(name: str, state: dict) -> dict:
    """把一个模拟账户的完整状态写库（单事务）。

    state 由 `core.paper_account.PaperAccount.to_state()` 生成。
    写入策略：positions 用"快照覆盖"（先删后插）；orders/nav 按主键 upsert；
    fills 依赖唯一索引 + INSERT OR IGNORE 保持幂等——重复保存同一状态不会累积重复记录。
    """
    if not state:
        return {"saved": False, "reason": "空状态"}
    init_store()
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    counts = {"positions": 0, "orders": 0, "fills": 0, "nav": 0}
    try:
        conn.execute(
            "INSERT INTO paper_accounts (name, initial_cash, cash, updated_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET "
            "initial_cash=excluded.initial_cash, cash=excluded.cash, "
            "updated_at=excluded.updated_at",
            (name, float(state.get("initial_cash") or 0.0),
             float(state.get("cash") or 0.0), now))

        conn.execute("DELETE FROM paper_positions WHERE name = ?", (name,))
        for code, p in (state.get("positions") or {}).items():
            conn.execute(
                "INSERT INTO paper_positions (name, code, quantity, available, total_cost, "
                "last_buy_date, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, code, int(p.get("quantity") or 0), int(p.get("available") or 0),
                 float(p.get("total_cost") or 0.0), str(p.get("last_buy_date") or ""), now))
            counts["positions"] += 1

        # 限价/滑点：`0` 与 `None` 都表示"市价单"，一律落 NULL。
        # ⚠️ 不能让 0.0 和 NULL 同时存在：下游判"是否限价单"要写两遍，
        #    而漏写一遍就会把市价单当成"限价 0 元"从而永远撤单。
        def _pos_or_none(v):
            return float(v) if isinstance(v, (int, float)) and v > 0 else None

        for o in (state.get("orders") or []):
            conn.execute(
                "INSERT INTO paper_orders (name, order_id, code, direction, quantity, "
                "submit_date, status, reason, filled_quantity, filled_price, filled_date, "
                "limit_price, decision_price, slippage, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(name, order_id) DO UPDATE SET code=excluded.code, "
                "direction=excluded.direction, quantity=excluded.quantity, "
                "submit_date=excluded.submit_date, status=excluded.status, "
                "reason=excluded.reason, filled_quantity=excluded.filled_quantity, "
                "filled_price=excluded.filled_price, filled_date=excluded.filled_date, "
                "limit_price=excluded.limit_price, decision_price=excluded.decision_price, "
                "slippage=excluded.slippage, "
                "updated_at=excluded.updated_at",
                (name, int(o.get("order_id") or 0), o.get("code"), o.get("direction"),
                 int(o.get("quantity") or 0), o.get("submit_date"), o.get("status"),
                 o.get("reason"), int(o.get("filled_quantity") or 0),
                 float(o.get("filled_price") or 0.0), o.get("filled_date"),
                 _pos_or_none(o.get("limit_price")), _pos_or_none(o.get("decision_price")),
                 (float(o["slippage"]) if isinstance(o.get("slippage"), (int, float)) else None),
                 now))
            counts["orders"] += 1

        for f in (state.get("fills") or []):
            cur = conn.execute(
                "INSERT OR IGNORE INTO paper_fills (name, date, code, direction, quantity, "
                "actual_price, cash_after, payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, f.get("date"), f.get("code"), f.get("direction"),
                 int(f.get("quantity") or 0), float(f.get("actual_price") or 0.0),
                 float(f.get("cash_after") or 0.0),
                 json.dumps(f, ensure_ascii=False, default=str), now))
            counts["fills"] += max(0, cur.rowcount)

        for n in (state.get("nav") or []):
            conn.execute(
                "INSERT INTO paper_nav (name, date, cash, position_value, total_value, "
                "return_pct, at_cost, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(name, date) DO UPDATE SET cash=excluded.cash, "
                "position_value=excluded.position_value, total_value=excluded.total_value, "
                "return_pct=excluded.return_pct, at_cost=excluded.at_cost, "
                "updated_at=excluded.updated_at",
                (name, str(n.get("date")), float(n.get("cash") or 0.0),
                 float(n.get("position_value") or 0.0), float(n.get("total_value") or 0.0),
                 float(n.get("return_pct") or 0.0),
                 (int(n["at_cost"]) if n.get("at_cost") is not None else None), now))
            counts["nav"] += 1

        conn.commit()
        return {"saved": True, **counts}
    finally:
        conn.close()


def load_paper_state(name: str = "default") -> dict | None:
    """读取模拟账户状态（可直接喂给 PaperAccount.from_state）；不存在返回 None"""
    init_store()
    conn = get_connection()
    try:
        acc = conn.execute("SELECT * FROM paper_accounts WHERE name = ?", (name,)).fetchone()
        if acc is None:
            return None

        positions = {}
        for r in conn.execute("SELECT * FROM paper_positions WHERE name = ?", (name,)).fetchall():
            d = dict(r)
            positions[d["code"]] = {
                "code": d["code"], "quantity": d["quantity"], "available": d["available"],
                "total_cost": d["total_cost"], "last_buy_date": d.get("last_buy_date") or "",
            }

        orders = []
        for r in conn.execute("SELECT * FROM paper_orders WHERE name = ? ORDER BY order_id",
                              (name,)).fetchall():
            d = dict(r)
            d.pop("name", None)
            d.pop("updated_at", None)
            orders.append(d)

        fills = []
        for r in conn.execute("SELECT payload FROM paper_fills WHERE name = ? ORDER BY id",
                              (name,)).fetchall():
            try:
                fills.append(json.loads(r["payload"]))
            except (TypeError, json.JSONDecodeError):
                continue

        nav = []
        for r in conn.execute("SELECT * FROM paper_nav WHERE name = ? ORDER BY date",
                              (name,)).fetchall():
            d = dict(r)
            d.pop("name", None)
            d.pop("updated_at", None)
            nav.append(d)

        return {"name": acc["name"], "initial_cash": acc["initial_cash"], "cash": acc["cash"],
                "positions": positions, "orders": orders, "fills": fills, "nav": nav}
    finally:
        conn.close()


def list_paper_accounts() -> list:
    init_store()
    conn = get_connection()
    try:
        return [r["name"] for r in conn.execute(
            "SELECT name FROM paper_accounts ORDER BY name").fetchall()]
    finally:
        conn.close()


def list_paper_fills(date: str = None, name: str = None) -> list[dict]:
    """成交明细（可按日/按账户过滤）。

    交易台「联赛每日决策」要展示"决策 → 实际成交"：只给决策不给成交，
    用户无法判断"说了买"和"真买了"是不是一回事（T+1 开盘撮合次日才成交）。
    """
    init_store()
    conn = get_connection()
    try:
        sql = ("SELECT id, name, date, code, direction, quantity, actual_price, "
               "cash_after, created_at FROM paper_fills WHERE 1=1")
        args: list = []
        if date:
            sql += " AND date = ?"; args.append(date)
        if name:
            sql += " AND name = ?"; args.append(name)
        sql += " ORDER BY date DESC, name, id"
        return [dict(r) for r in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()


def delete_paper_account(name: str) -> int:
    """删除账户及其全部记录（"重置模拟盘"用），返回受影响行数"""
    init_store()
    conn = get_connection()
    removed = 0
    try:
        for table in _PAPER_TABLES:
            removed += conn.execute(f"DELETE FROM {table} WHERE name = ?", (name,)).rowcount
        conn.commit()
        return removed
    finally:
        conn.close()


# ================================================================
# LLM 调用记录（Phase 6.2：调用链可观测性）
# ================================================================

# ================================================================
# 外部观点缓存（跨进程）
# ================================================================

def get_opinion_cache(cache_key: str, max_age_sec: int = 0) -> dict | None:
    """读跨进程观点缓存；过期/不存在/解析失败 ⇒ None。

    ⚠️ 解析失败返回 None（当作没有缓存）而不是抛异常：缓存坏了只能让调用方
    多花一次钱，不该让整条复盘链路失败。
    """
    try:
        init_store()
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT payload, created_at FROM opinion_cache WHERE cache_key = ?",
                (cache_key,)).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        if max_age_sec:
            created = datetime.strptime(str(row["created_at"])[:19], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - created).total_seconds() > max_age_sec:
                return None
        return json.loads(row["payload"])
    except Exception as e:                              # noqa: BLE001
        _logger.warning("[观点缓存] 读取失败（当作未命中）: %s", e)
        return None


def save_opinion_cache(cache_key: str, payload: dict) -> None:
    """写跨进程观点缓存（best-effort，失败只记日志）。"""
    try:
        init_store()
        conn = get_connection()
        try:
            conn.execute(
                """INSERT INTO opinion_cache (cache_key, payload, created_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                       payload = excluded.payload, created_at = excluded.created_at""",
                (cache_key, json.dumps(payload, ensure_ascii=False),
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:                              # noqa: BLE001
        _logger.warning("[观点缓存] 写入失败（不影响调用）: %s", e)


def insert_llm_call(record: dict) -> int:
    """记录一次 LLM 调用；返回自增 id，失败返回 0。

    **绝不能因为可观测性写入失败而影响调用本身**：写入异常在这里吞掉并记日志，
    调用方（llm/deepseek_client）只做 best-effort 记录。
    """
    try:
        init_store()
        conn = get_connection()
        try:
            cur = conn.execute(
                """INSERT INTO llm_calls
                   (created_at, model, purpose, ok, latency_ms, prompt_tokens,
                    completion_tokens, total_tokens, prompt_chars, response_chars, error,
                    cost_cny)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 str(record.get("model") or ""),
                 str(record.get("purpose") or "unknown"),
                 1 if record.get("ok") else 0,
                 int(record.get("latency_ms") or 0),
                 record.get("prompt_tokens"), record.get("completion_tokens"),
                 record.get("total_tokens"), record.get("prompt_chars"),
                 record.get("response_chars"), str(record.get("error") or "")[:300],
                 record.get("cost_cny")))
            conn.commit()
            return cur.lastrowid or 0
        finally:
            conn.close()
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[LLM记录] 写入失败: %s", e)
        return 0


def llm_cost_rows(since: str = None, until: str = None) -> list[dict]:
    """取原始调用行（含金额）供成本核算，不在 SQL 里做价格计算。

    ⚠️ 金额**不在 SQL 里算**：单价表是 JSON、还带峰谷与阶梯，SQLite 里做不了。
    SQL 只负责按时间切片，价格规则全部留给 `core/llm_cost`（纯函数，可测）。

    `cost_cny` 为 NULL 表示"这次没记账"（老行、或写入时价格表读不到）——
    调用方应据此**回退到按当前单价估算**，而不是当 0 处理。
    """
    init_store()
    where, params = [], []
    if since:
        where.append("created_at >= ?")
        params.append(since)
    if until:
        where.append("created_at < ?")
        params.append(until)
    sql = ("SELECT id, created_at, model, purpose, ok, prompt_tokens, completion_tokens, "
           "total_tokens, cost_cny FROM llm_calls")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at"
    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def update_llm_cost(call_id: int, cost_cny: float) -> None:
    """回填单条金额（价格表变更后重算用）。"""
    init_store()
    conn = get_connection()
    try:
        conn.execute("UPDATE llm_calls SET cost_cny = ? WHERE id = ?", (cost_cny, call_id))
        conn.commit()
    finally:
        conn.close()


def llm_call_stats(recent_days: int = 7) -> dict:
    """近 N 天的调用统计（配置页面板 + 成本核对）"""
    init_store()
    since = (datetime.now() - timedelta(days=recent_days)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT COUNT(*) AS n,
                      SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS failed,
                      AVG(CASE WHEN ok = 1 THEN latency_ms END) AS avg_ms,
                      SUM(COALESCE(prompt_tokens, 0)),
                      SUM(COALESCE(completion_tokens, 0))
               FROM llm_calls WHERE created_at >= ?""", (since,)).fetchone()
        total = int(row[0] or 0)
        failed = int(row[1] or 0)

        def _group(column: str) -> list:
            rows = conn.execute(
                f"""SELECT {column}, COUNT(*), SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END),
                           AVG(latency_ms), SUM(COALESCE(total_tokens, 0))
                    FROM llm_calls WHERE created_at >= ?
                    GROUP BY {column} ORDER BY COUNT(*) DESC""", (since,)).fetchall()
            return [{"key": r[0] or "?", "count": int(r[1] or 0), "failed": int(r[2] or 0),
                     "avg_latency_ms": int(r[3] or 0), "total_tokens": int(r[4] or 0)}
                    for r in rows]

        return {
            "recent_days": recent_days,
            "total": total,
            "failed": failed,
            "failure_rate": round(failed / total, 3) if total else 0.0,
            "avg_latency_ms": int(row[2]) if row[2] else 0,
            "prompt_tokens": int(row[3] or 0),
            "completion_tokens": int(row[4] or 0),
            "by_model": _group("model"),
            "by_purpose": _group("purpose"),
        }
    finally:
        conn.close()


def get_recent_llm_calls(limit: int = 50, only_failures: bool = False,
                         purpose: str = None) -> list:
    """最近的调用明细（默认全部；only_failures=True 只看失败）"""
    init_store()
    conn = get_connection()
    try:
        sql = "SELECT * FROM llm_calls WHERE 1=1"
        params: list = []
        if only_failures:
            sql += " AND ok = 0"
        if purpose:
            sql += " AND purpose = ?"
            params.append(purpose)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


# ================================================================
# 深度分析历史（含 T+N 回看闭环）
# ================================================================

def insert_analysis(record: dict) -> int:
    """保存一次深度分析记录，返回自增 id（失败返回 0）。

    与 insert_llm_call 同理：历史落库绝不能影响分析主流程，
    异常吞掉并记日志，调用方只做 best-effort。
    record 关键字段：analysis_date/stock_code/stock_name/price_at_analysis/
    decision_label/model/source/payload(dict)/trade_plan(dict)/timing_regime/market_score
    """
    try:
        init_store()
        conn = get_connection()
        try:
            # ⚠️ UPSERT（依赖 `uniq_ana_date_code`）：同一天同一只票只保留**最新**一次分析，
            #    重跑要作废旧的回填结果（目标/止损可能变了），让 15:35 的任务重算。
            cur = conn.execute(
                """INSERT INTO analyses
                   (analysis_date, stock_code, stock_name, price_at_analysis,
                    decision_label, model, source, payload, trade_plan,
                    timing_regime, market_score, market_score_source,
                    backfill_status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(analysis_date, stock_code) DO UPDATE SET
                     stock_name=excluded.stock_name,
                     price_at_analysis=excluded.price_at_analysis,
                     decision_label=excluded.decision_label,
                     model=excluded.model,
                     source=excluded.source,
                     payload=excluded.payload,
                     trade_plan=excluded.trade_plan,
                     timing_regime=excluded.timing_regime,
                     market_score=excluded.market_score,
                     market_score_source=excluded.market_score_source,
                     backfill_status=excluded.backfill_status,
                     created_at=excluded.created_at,
                     ret_5d=NULL, ret_20d=NULL, hit_target=NULL, hit_stop=NULL,
                     backfilled_at=NULL""",
                (record.get("analysis_date"), record.get("stock_code"),
                 record.get("stock_name"), record.get("price_at_analysis"),
                 record.get("decision_label"), record.get("model"),
                 record.get("source", "ui"),
                 json.dumps(record.get("payload") or {}, ensure_ascii=False, default=str),
                 json.dumps(record.get("trade_plan") or {}, ensure_ascii=False, default=str),
                 record.get("timing_regime"), record.get("market_score"),
                 record.get("market_score_source"),
                 record.get("backfill_status"),
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            return cur.lastrowid or 0
        finally:
            conn.close()
    except Exception as e:                     # noqa: BLE001
        _logger.warning("[分析历史] 写入失败: %s", e)
        return 0


def list_analyses(limit: int = 50, stock_code: str = None,
                  only_backfillable: bool = False,
                  backfill_status: str = None) -> list[dict]:
    """分析历史列表（时间倒序）。列表查询不解析 payload JSON，避免大 JSON 拖慢 UI。

    only_backfillable=True 只取待回填行（backfill_status 为 NULL/partial），
    供回填任务与手动回填按钮使用。
    backfill_status 精确匹配指定状态（如手动重试 failed 行）。
    """
    init_store()
    conn = get_connection()
    try:
        sql = ("SELECT id, analysis_date, stock_code, stock_name, price_at_analysis, "
               "decision_label, model, source, timing_regime, market_score, "
               "ret_5d, ret_20d, hit_target, hit_stop, backfill_status, backfilled_at, "
               "created_at FROM analyses WHERE 1=1")
        params: list = []
        if stock_code:
            sql += " AND stock_code = ?"
            params.append(stock_code)
        if only_backfillable:
            sql += " AND (backfill_status IS NULL OR backfill_status = 'partial')"
        if backfill_status:
            sql += " AND backfill_status = ?"
            params.append(backfill_status)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def list_analyses_since(cutoff_date: str, limit: int = 200,
                        with_plan: bool = True) -> list[dict]:
    """取 `cutoff_date`（含）之后的分析记录（时间倒序）。

    与 `list_analyses` 的区别：这里**解析 trade_plan**（`advice_status` / `entry` /
    `stop_loss` / `target_price` 都在其中），而 `list_analyses` 刻意不解析它
    （避免大 payload 拖慢 UI 列表）。只取 trade_plan 一列，不取 payload。

    `with_plan=False` 时不返回 trade_plan 字段（调用方只要元信息时省一次解析）。
    用途：每日触发核对（`analysis_history.check_triggers_daily`）与决策汇总
    （`core.decision_digest`）。
    """
    init_store()
    conn = get_connection()
    try:
        cols = ("id, analysis_date, stock_code, stock_name, price_at_analysis, "
                "decision_label, source, ret_5d, ret_20d, hit_target, hit_stop, "
                "backfill_status, created_at")
        if with_plan:
            cols += ", trade_plan"
        rows = conn.execute(
            f"SELECT {cols} FROM analyses WHERE analysis_date >= ? "
            "ORDER BY id DESC LIMIT ?",
            (str(cutoff_date)[:10], int(limit))).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if with_plan:
                try:
                    d["trade_plan"] = json.loads(d.get("trade_plan") or "{}")
                except (TypeError, json.JSONDecodeError):
                    d["trade_plan"] = {}
            out.append(d)
        return out
    finally:
        conn.close()


def get_analysis(ana_id: int) -> dict | None:
    """取单条完整记录（含 payload/trade_plan，已反序列化为 dict）"""
    init_store()
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (ana_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        for key in ("payload", "trade_plan"):
            try:
                d[key] = json.loads(d.get(key) or "{}")
            except (TypeError, json.JSONDecodeError):
                d[key] = {}
        return d
    finally:
        conn.close()


def update_analysis_backfill(ana_id: int, fields: dict):
    """回填 T+N 结果。fields 可含：ret_5d/ret_20d/hit_target/hit_stop/
    backfill_status/backfilled_at/price_at_analysis"""
    if not fields:
        return
    init_store()
    conn = get_connection()
    try:
        allowed = {"ret_5d", "ret_20d", "hit_target", "hit_stop",
                   "backfill_status", "backfilled_at", "price_at_analysis"}
        sets, params = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k} = ?")
                params.append(v)
        if not sets:
            return
        params.append(ana_id)
        conn.execute(f"UPDATE analyses SET {', '.join(sets)} WHERE id = ?", params)
        conn.commit()
    finally:
        conn.close()


# ================================================================
# 深度分析复盘（2026-10-20）
# ================================================================

_REVIEW_JSON_FIELDS = ("secondary_causes", "facts", "candidates", "llm_review")


def _decode_review(row) -> dict:
    d = dict(row)
    for key in _REVIEW_JSON_FIELDS:
        raw = d.get(key)
        if raw in (None, ""):
            d[key] = ([] if key == "secondary_causes" else {})
            continue
        try:
            d[key] = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as e:
            # 坏 JSON 不能静默当成空——它是"归因结论丢了"，必须留下原值供排查
            _logger.warning("复盘记录 %s 的 %s 不是合法 JSON（%s），保留原文前 200 字",
                            d.get("analysis_id"), key, e)
            d[key] = {"_corrupt": str(raw)[:200]} if key != "secondary_causes" else []
    return d


def upsert_analysis_review(record: dict) -> int:
    """写入/覆盖一条复盘（按 `analysis_id` 唯一）。返回行 id，失败返回 0。

    ⚠️ 覆盖是**有意**的：同一条分析重算复盘时，新结论应该替换旧结论，
       但 `facts` 会一起替换——所以"归因阈值变了"这类历史差异靠 `updated_at`
       + 备份判断，不靠保留两份。
    """
    if not record or not record.get("analysis_id"):
        return 0
    try:
        init_store()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_connection()
        try:
            cur = conn.execute(
                """INSERT INTO analysis_reviews
                   (analysis_id, stock_code, stock_name, analysis_date, review_date,
                    decision_label, model, verdict, primary_cause, secondary_causes,
                    horizon, facts, candidates, llm_review, status, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(analysis_id) DO UPDATE SET
                     verdict=excluded.verdict, primary_cause=excluded.primary_cause,
                     secondary_causes=excluded.secondary_causes, facts=excluded.facts,
                     candidates=excluded.candidates, llm_review=excluded.llm_review,
                     status=excluded.status, review_date=excluded.review_date,
                     horizon=excluded.horizon, model=excluded.model,
                     updated_at=excluded.updated_at""",
                (record.get("analysis_id"), record.get("stock_code"),
                 record.get("stock_name"), record.get("analysis_date"),
                 record.get("review_date"), record.get("decision_label"),
                 record.get("model"), record.get("verdict"),
                 record.get("primary_cause"),
                 json.dumps(record.get("secondary_causes") or [], ensure_ascii=False),
                 record.get("horizon") or 20,
                 json.dumps(record.get("facts") or {}, ensure_ascii=False),
                 json.dumps(record.get("candidates") or [], ensure_ascii=False),
                 json.dumps(record.get("llm_review") or {}, ensure_ascii=False)
                 if record.get("llm_review") else None,
                 record.get("status") or "mechanical", now, now))
            conn.commit()
            if cur.lastrowid:
                return int(cur.lastrowid)
            got = conn.execute("SELECT id FROM analysis_reviews WHERE analysis_id = ?",
                               (record.get("analysis_id"),)).fetchone()
            return int(got[0]) if got else 0
        finally:
            conn.close()
    except Exception as e:                      # noqa: BLE001
        _logger.warning("upsert_analysis_review 失败: %s", e)
        return 0


def get_analysis_review(analysis_id: int) -> dict | None:
    init_store()
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM analysis_reviews WHERE analysis_id = ?",
                           (analysis_id,)).fetchone()
        return _decode_review(row) if row else None
    finally:
        conn.close()


def list_analysis_reviews(limit: int = 200, since: str = None,
                          verdict: str = None, cause: str = None,
                          only_llm: bool = False,
                          horizon: int = None) -> list[dict]:
    """按时间倒序列出复盘。`since` 为分析日起点（含），用于"近 N 天归因分布"。

    `horizon`：5=早期结论 / 20=完整窗口。**统计时必须显式指定或分组**——
    两者混在一起算出来的"判错率"没有意义（5 天窗口的判错和 20 天窗口的判错
    是两件不同的事）。
    """
    init_store()
    sql = "SELECT * FROM analysis_reviews WHERE 1=1"
    params: list = []
    if since:
        sql += " AND analysis_date >= ?"
        params.append(str(since)[:10])
    if verdict:
        sql += " AND verdict = ?"
        params.append(verdict)
    if cause:
        sql += " AND primary_cause = ?"
        params.append(cause)
    if horizon:
        sql += " AND horizon = ?"
        params.append(int(horizon))
    if only_llm:
        sql += " AND llm_review IS NOT NULL"
    sql += " ORDER BY analysis_date DESC, id DESC LIMIT ?"
    params.append(int(limit))
    conn = get_connection()
    try:
        return [_decode_review(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def pending_review_analyses(limit: int = 50,
                            include_mechanical: bool = False,
                            allow_partial: bool = True) -> list[dict]:
    """待复盘的分析 —— 复盘批任务的输入。

    两个筛选维度：
    - **窗口**：`done`（T+20 完整）优先；`allow_partial=True` 时把 `partial`
      （只有 T+5）也纳入 —— 否则最早的分析也要等一个月才有第一条复盘，
      而"方向判对判错"在 T+5 就已经可判（只是扫损后反弹的判据不可靠）。
    - **已有记录**：`include_mechanical=True` 时把"只有机械归因"的也捞出来 ——
      回填阶段为了不重复取数，机械复盘是**不含 ATR** 的（回填只拉了分析日前 7 天，
      不够 14 根）。批任务会重取长历史补上 `atr14` / `stop_dist_atr`，
      那是"止损是否过近"唯一可量化的依据。
      ⚠️ 2026-10-29 起 `llm_error` 的也一并捞出（里程碑 LLM 失败后，手动批量补跑
      是唯一的重试入口 —— done 行回填任务不再碰、partial 行非里程碑也不碰）。

    排序：`done` 先于 `partial`、无记录的先于有机械记录的，同组内按分析日倒序。
    """
    init_store()
    status_sql = "a.backfill_status IN ('done','partial')" if allow_partial \
        else "a.backfill_status = 'done'"
    where = ("(r.id IS NULL"
             + (" OR r.status IN ('mechanical','llm_error')"
                if include_mechanical else "") + ")")
    conn = get_connection()
    try:
        rows = conn.execute(
            f"""SELECT a.id, a.stock_code, a.stock_name, a.analysis_date,
                       a.decision_label, a.model, a.ret_5d, a.ret_20d,
                       a.hit_target, a.hit_stop, a.backfill_status
                FROM analyses a LEFT JOIN analysis_reviews r ON r.analysis_id = a.id
                WHERE {status_sql} AND a.ret_5d IS NOT NULL AND {where}
                ORDER BY (r.id IS NULL) DESC, (a.backfill_status = 'done') DESC,
                         a.analysis_date DESC LIMIT ?""",
            (int(limit),)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def analysis_stats(recent_days: int = 30) -> dict:
    """分析决策胜率统计（近 N 天、已回填样本）。
    胜率口径：decision_label='看多' 且 ret_5d > 0 记为胜；
    '看空' 且 ret_5d < 0 记为胜（对称口径）；'观望'/失败样本不参与。
    """
    init_store()
    conn = get_connection()
    try:
        cutoff = (datetime.now() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
        rows = conn.execute(
            "SELECT decision_label, backfill_status, COUNT(*) FROM analyses "
            "WHERE analysis_date >= ? GROUP BY decision_label, backfill_status",
            (cutoff,)).fetchall()
        by_label = {}
        total = 0
        for label, status, n in rows:
            total += n
            d = by_label.setdefault(label or "未知", {})
            d[status or "待回填"] = n

        # 看多样本的 5 日胜率（对标"预测跟踪闭环"的命中率）
        bull = conn.execute(
            """SELECT COUNT(*), SUM(CASE WHEN ret_5d > 0 THEN 1 ELSE 0 END)
               FROM analyses
               WHERE analysis_date >= ? AND decision_label = '看多'
                 AND ret_5d IS NOT NULL""", (cutoff,)).fetchone()
        bull_n, bull_win = int(bull[0] or 0), int(bull[1] or 0)

        bear = conn.execute(
            """SELECT COUNT(*), SUM(CASE WHEN ret_5d < 0 THEN 1 ELSE 0 END)
               FROM analyses
               WHERE analysis_date >= ? AND decision_label = '看空'
                 AND ret_5d IS NOT NULL""", (cutoff,)).fetchone()
        bear_n, bear_win = int(bear[0] or 0), int(bear[1] or 0)

        pending = conn.execute(
            """SELECT COUNT(*) FROM analyses WHERE analysis_date >= ?
               AND (backfill_status IS NULL OR backfill_status = 'partial')""",
            (cutoff,)).fetchone()[0]

        return {
            "recent_days": recent_days, "total": total, "by_label": by_label,
            "bull_samples": bull_n, "bull_win_rate": round(bull_win / bull_n, 2) if bull_n else None,
            "bear_samples": bear_n, "bear_win_rate": round(bear_win / bear_n, 2) if bear_n else None,
            "pending_backfill": int(pending or 0),
        }
    finally:
        conn.close()


# ================================================================
# 选股候选（P2：条件命中率回测的样本来源）
# ================================================================

def insert_screen_picks(picks: list[dict]) -> int:
    """批量写入一次选股的候选，返回写入条数（失败返回 0）。

    与 insert_analysis 同理：落库绝不能影响选股主流程。
    空列表直接返回 0（熊市跳过选股时就是没有候选）。
    """
    if not picks:
        return 0
    try:
        init_store()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_connection()
        try:
            conn.executemany(
                """INSERT INTO screen_picks
                   (pick_date, stock_code, stock_name, source, strategy, score,
                    score_coverage, price_at_pick, rank_in_run, market_state,
                    market_score, factors, conditions, pe_source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(p.get("pick_date"), p.get("stock_code"), p.get("stock_name"),
                  p.get("source", "ui"), p.get("strategy"), p.get("score"),
                  p.get("score_coverage"), p.get("price_at_pick"),
                  p.get("rank_in_run"), p.get("market_state"), p.get("market_score"),
                  json.dumps(p.get("factors") or {}, ensure_ascii=False, default=str),
                  json.dumps(p.get("conditions") or {}, ensure_ascii=False, default=str),
                  p.get("pe_source"),
                  now) for p in picks])
            conn.commit()
            return len(picks)
        finally:
            conn.close()
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[选股历史] 写入失败: %s", e)
        return 0


def list_screen_picks(limit: int = 100, stock_code: str = None, source: str = None,
                      only_backfillable: bool = False,
                      backfill_status: str = None,
                      oldest_first: bool = False,
                      max_pick_date: str = None) -> list[dict]:
    """选股候选列表（默认时间倒序）。列表查询不解析 factors/conditions JSON。

    Args:
        oldest_first: 按 id **升序**取，供 T+N 回填使用（见下）。
        max_pick_date: 只要 `pick_date <= 该日期` 的行（"已到期"过滤）。

    ⚠️ 为什么回填必须 `oldest_first=True`
    --------------------------------
    原来固定 `ORDER BY id DESC LIMIT 200`，而扩池后**每天落库 200~440 行**
    （2026-10-28 实测 438 行）。于是"最新 200 条"永远只覆盖**当天**的新候选，
    而 T+5/T+20 要等 5~20 个交易日 —— 等它们到期时，早已被后面的新行挤出窗口。
    实测后果：最老的待回填是 `id=43`（`pick_date=2026-09-25`），而窗口下界已经是
    `id=1781`；全表 `ret_20d` 覆盖 **0 / 2074**，即"选股质量闭环"从未拿到过 20 日收益。

    这是典型的「分页窗口把待办挤出」：待办是**老的**，窗口却取**最新的**。
    """
    init_store()
    conn = get_connection()
    try:
        sql = ("SELECT id, pick_date, stock_code, stock_name, source, strategy, "
               "score, score_coverage, price_at_pick, rank_in_run, market_state, "
               "market_score, bench_ret_5d, bench_ret_20d, ret_5d, ret_20d, "
               "backfill_status, backfilled_at, pe_source, created_at "
               "FROM screen_picks WHERE 1=1")
        params: list = []
        if stock_code:
            sql += " AND stock_code = ?"
            params.append(stock_code)
        if source:
            sql += " AND source = ?"
            params.append(source)
        if only_backfillable:
            sql += " AND (backfill_status IS NULL OR backfill_status = 'partial')"
        if backfill_status:
            sql += " AND backfill_status = ?"
            params.append(backfill_status)
        if max_pick_date:
            # 只取「已经到期」的候选：T+5 需要起点之后 6 根 K 线，约 7 个自然日。
            # 不过滤的话窗口里全是当天刚选出的新行，它们必然算出空收益被 skip
            # —— 白占窗口、白拉行情（实测每天白跑 17 分钟、0 条有效回填）。
            sql += " AND pick_date <= ?"
            params.append(max_pick_date)
        # 默认倒序（页面/接口看"最新的"）；回填走升序（待办是老的那批）
        sql += (" ORDER BY id ASC LIMIT ?" if oldest_first
                else " ORDER BY id DESC LIMIT ?")
        params.append(int(limit))
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def update_screen_pick_backfill(pick_id: int, fields: dict):
    """回填 T+N 收益与基准收益（白名单字段）"""
    if not fields:
        return
    init_store()
    conn = get_connection()
    try:
        allowed = {"ret_5d", "ret_20d", "bench_ret_5d", "bench_ret_20d",
                   "backfill_status", "backfilled_at", "price_at_pick"}
        sets, params = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k} = ?")
                params.append(v)
        if not sets:
            return
        params.append(pick_id)
        conn.execute(f"UPDATE screen_picks SET {', '.join(sets)} WHERE id = ?", params)
        conn.commit()
    finally:
        conn.close()


def benchmark_returns_batch(dates: list[str], symbol: str = "sh000001") -> dict:
    """批量取同期指数收益，返回 {date: {"bench_ret_5d":…, "bench_ret_20d":…}}。

    daily_features 存的是**百分比**，这里统一换算成小数，避免与 ret_5d 混用。
    查不到的日期**不出现在结果里**（缺基准就不算超额，绝不用 0 冒充）。

    为什么批量：回填一批候选时，候选往往集中在少数几个选股日（一天 5~10 只），
    逐条查会把同一天的指数特征反复取 N 遍。批次 16 的预演里 20 条候选触发了
    5 次重复查询与 5 次重复的指数 K 线请求——一次查询搞定。
    """
    if not dates:
        return {}
    uniq = sorted({d for d in dates if d})
    if not uniq:
        return {}
    init_store()
    conn = get_connection()
    try:
        placeholders = ",".join("?" * len(uniq))
        rows = conn.execute(
            f"SELECT date, future_ret_5d, future_ret_20d FROM daily_features "
            f"WHERE symbol = ? AND date IN ({placeholders})",
            [symbol] + uniq).fetchall()
    finally:
        conn.close()
    out = {}
    for r in rows:
        d = {}
        if r["future_ret_5d"] is not None:
            d["bench_ret_5d"] = round(r["future_ret_5d"] / 100.0, 6)
        if r["future_ret_20d"] is not None:
            d["bench_ret_20d"] = round(r["future_ret_20d"] / 100.0, 6)
        if d:
            out[r["date"]] = d
    return out


def benchmark_returns(date: str, symbol: str = "sh000001") -> dict:
    """单日同期指数收益（批量版的便捷包装）"""
    if not date:
        return {}
    return benchmark_returns_batch([date], symbol).get(date, {})


# ================================================================
# screen_picks 的**去重口径**（唯一实现，2026-10-24）
# ================================================================
# ⚠️⚠️ 为什么必须先"去重"再统计：
#   同一 `(pick_date, stock_code)` 会被**多次写入** —— `pipeline` 在 09:35 / 11:50 /
#   15:39 回填三次、`screen` 收盘再写一次，同一只票同一天最多 **9 行**。实测
#   2026-10-24：`screen_picks` **857 行只有 401 条唯一记录（53% 是重复行）**。
#   后果不是"多几条"：任何按行求均值/胜率/计数都会被**重复行加权**。
#   实测事故：某次分析里 "ret20>=15% 共 21 条、5 日超额 -5.20%"，去重后
#   只有 **6 条 / 2 只票**（万科A、保利发展）⇒ 量级差 3.5 倍、结论强度虚高。
#
# ⚠️ 为什么不加唯一索引（那是最直觉的修法）：`screen_history.recent_top_overlap`
#   依赖按 `created_at` 分批比较"最近几次选股的重合度"，唯一键会把它废掉。
#   ⇒ 写入继续自由，**读取侧统一走下面两个函数**。
#
# ⚠️ 保留哪一行 = `id` 最大的那行（"后写覆盖"），与 `panel_store.load_panel`
#   的去重约定一致；这样回填过的行优先于未回填的行。
SCREEN_PICKS_DEDUP_KEY = "pick_date, stock_code"


def _dedup_screen_picks_sql(cols: str, where: str = "", tail: str = "") -> str:
    """包一层"每个 (pick_date, stock_code) 只留 id 最大的一行"的 SQL。

    用窗口函数而不是 `GROUP BY`：调用方要的是**整行**（score/ret/payload…），
    而 `GROUP BY` 只能配聚合函数，取 `MAX(score)` 之类会把不同列的行混在一起。
    """
    return (f"SELECT {cols} FROM ("
            f"  SELECT s.*, ROW_NUMBER() OVER ("
            f"    PARTITION BY s.pick_date, s.stock_code ORDER BY s.id DESC) AS _rn"
            f"  FROM screen_picks s {where}) WHERE _rn = 1 {tail}")


def load_screen_picks_dedup(cols: str = "*", where: str = "",
                            params: tuple = (), tail: str = "") -> list:
    """按 (pick_date, stock_code) 去重后读取 `screen_picks`（原始 sqlite3.Row 列表）。

    ⚠️ 任何"求平均/胜率/计数"的消费方都必须走这里或 `count_screen_picks_dedup`，
    否则会被重复行加权（见本节顶部事故）。
    """
    init_store()
    conn = get_connection()
    try:
        return conn.execute(_dedup_screen_picks_sql(cols, where, tail),
                            tuple(params or ())).fetchall()
    finally:
        conn.close()


def count_screen_picks_dedup(where: str = "", params: tuple = ()) -> int:
    """按 (pick_date, stock_code) 去重的条数（用于"积压/待回填"这类计数）。"""
    rows = load_screen_picks_dedup("COUNT(*) AS n", where, params)
    return int(rows[0]["n"] if rows else 0)


def _picks_with_payload(recent_days: int = 180) -> list[dict]:
    """取近 N 天**已回填**的候选（含解析后的 factors/conditions），供统计用"""
    cutoff = (datetime.now() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
    init_store()
    conn = get_connection()
    try:
        # ⚠️ 必须去重：同一只票同一天会被写多行（见本节顶部说明）
        rows = conn.execute(
            _dedup_screen_picks_sql(
                "id, pick_date, stock_code, stock_name, source, score, factors, "
                "conditions, ret_5d, ret_20d, bench_ret_5d, bench_ret_20d",
                "WHERE s.pick_date >= ? AND s.ret_5d IS NOT NULL", "ORDER BY id"),
            (cutoff,)).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        for key in ("factors", "conditions"):
            try:
                d[key] = json.loads(d.get(key) or "{}")
            except (TypeError, json.JSONDecodeError):
                d[key] = {}
        out.append(d)
    return out


def _avg(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def screen_pick_stats(recent_days: int = 180) -> dict:
    """选股整体表现：样本数、T+5 平均收益/胜率、**超额**收益与超额胜率。

    超额 = 个股收益 - 同期上证收益。牛市里闭眼选都"赢"，只有超额才能说明选股有效。
    缺基准的样本不计入超额（分母单独统计），不拿 0 顶替。
    """
    rows = _picks_with_payload(recent_days)
    if not rows:
        return {"recent_days": recent_days, "samples": 0, "avg_ret_5d": None,
                "win_rate_5d": None, "excess_samples": 0, "avg_excess_5d": None,
                "excess_win_rate": None, "pending_backfill": 0}
    rets = [r["ret_5d"] for r in rows]
    excess = [r["ret_5d"] - r["bench_ret_5d"] for r in rows
              if r.get("bench_ret_5d") is not None]
    init_store()
    conn = get_connection()
    try:
        # ⚠️ 待回填也按 (pick_date, stock_code) 去重：否则"积压"会被放大 ~2.1 倍
        pending = conn.execute(
            _dedup_screen_picks_sql(
                "COUNT(*) AS n",
                "WHERE s.pick_date >= ? AND (s.backfill_status IS NULL "
                "OR s.backfill_status = 'partial')"),
            ((datetime.now() - timedelta(days=recent_days)).strftime("%Y-%m-%d"),)
        ).fetchone()[0]
    finally:
        conn.close()
    return {
        "recent_days": recent_days,
        "samples": len(rows),
        "avg_ret_5d": _avg(rets),
        "win_rate_5d": round(sum(1 for r in rets if r > 0) / len(rets), 2),
        "excess_samples": len(excess),
        "avg_excess_5d": _avg(excess),
        "excess_win_rate": (round(sum(1 for e in excess if e > 0) / len(excess), 2)
                            if excess else None),
        "pending_backfill": int(pending or 0),
    }


def condition_stats(recent_days: int = 180, min_samples: int = 3) -> dict:
    """按**条件**分组看区分度（P2 的核心问题：自然语言条件到底有没有用）。

    hits 里的条目形如 "市盈率(PE) 小于 20倍：实际 15.3倍"，取冒号前作为条件名。
    每个条件给：样本数、T+5 平均收益、胜率、平均超额；
    与全样本基线对比，差值就是这条条件的"区分度"。
    样本 < min_samples 的条件不输出（小样本噪声大于信号）。
    """
    rows = _picks_with_payload(recent_days)
    if not rows:
        return {"recent_days": recent_days, "samples": 0, "baseline": None,
                "conditions": [], "min_samples": min_samples}
    base_rets = [r["ret_5d"] for r in rows]
    base_excess = [r["ret_5d"] - r["bench_ret_5d"] for r in rows
                   if r.get("bench_ret_5d") is not None]
    baseline = {
        "samples": len(rows),
        "avg_ret_5d": _avg(base_rets),
        "win_rate_5d": round(sum(1 for r in base_rets if r > 0) / len(base_rets), 2),
        "avg_excess_5d": _avg(base_excess),
    }
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        for hit in (r.get("conditions") or {}).get("hits") or []:
            name = str(hit).split("：")[0].strip()
            if name:
                buckets.setdefault(name, []).append(r)
    out = []
    for name, group in buckets.items():
        if len(group) < min_samples:
            continue
        rets = [g["ret_5d"] for g in group]
        excess = [g["ret_5d"] - g["bench_ret_5d"] for g in group
                  if g.get("bench_ret_5d") is not None]
        avg_ret = _avg(rets)
        out.append({
            "condition": name,
            "samples": len(group),
            "avg_ret_5d": avg_ret,
            "win_rate_5d": round(sum(1 for r in rets if r > 0) / len(rets), 2),
            "avg_excess_5d": _avg(excess),
            # 区分度 = 该条件组平均收益 - 全样本平均收益（正=这条条件有正向区分度）
            "edge_vs_base": (round(avg_ret - baseline["avg_ret_5d"], 4)
                             if avg_ret is not None and baseline["avg_ret_5d"] is not None
                             else None),
        })
    out.sort(key=lambda x: -x["samples"])
    return {"recent_days": recent_days, "samples": len(rows), "baseline": baseline,
            "conditions": out, "min_samples": min_samples}


def factor_stats(recent_days: int = 180, min_samples: int = 5) -> dict:
    """按**因子得分**看相关性：因子分高是否真的对应更高的 T+5 收益（近似 IC）。

    用 Pearson 相关系数而非分组均值——样本少时更敏感，且不需要拍分组阈值。
    因子状态非 ok（数据缺失）的样本不参与，避免"缺失被当成 0 分"。
    """
    rows = _picks_with_payload(recent_days)
    if not rows:
        return {"recent_days": recent_days, "samples": 0, "factors": [],
                "min_samples": min_samples}
    pairs: dict[str, list[tuple]] = {}
    for r in rows:
        for name, meta in (r.get("factors") or {}).items():
            if not isinstance(meta, dict):
                continue
            if str(meta.get("status") or "ok") != "ok":
                continue            # 数据缺失的因子不进相关性（缺失≠0 分）
            score = meta.get("score")
            if score is None or r.get("ret_5d") is None:
                continue
            pairs.setdefault(name, []).append((float(score), float(r["ret_5d"])))

    def _pearson(xs: list) -> float | None:
        n = len(xs)
        if n < 3:
            return None
        mx = sum(x for x, _ in xs) / n
        my = sum(y for _, y in xs) / n
        sxy = sum((x - mx) * (y - my) for x, y in xs)
        sxx = sum((x - mx) ** 2 for x, _ in xs)
        syy = sum((y - my) ** 2 for _, y in xs)
        if sxx <= 0 or syy <= 0:
            return None             # 取值无差异 → 相关系数无定义
        return round(sxy / (sxx ** 0.5 * syy ** 0.5), 3)

    out = []
    for name, xs in pairs.items():
        if len(xs) < min_samples:
            continue
        out.append({"factor": name, "samples": len(xs),
                    "avg_score": round(sum(x for x, _ in xs) / len(xs), 2),
                    "ic_ret_5d": _pearson(xs)})
    out.sort(key=lambda x: -(abs(x["ic_ret_5d"] or 0)))
    return {"recent_days": recent_days, "samples": len(rows), "factors": out,
            "min_samples": min_samples}


# ================================================================
# 跨进程任务互斥锁
# ================================================================

def acquire_task_lock(task: str, holder: str, ttl_seconds: int = 600) -> bool:
    """获取跨进程任务锁（拿不到返回 False，调用方应直接跳过本次执行）。

    Args:
        task: 任务名（如 "backfill_analysis"）
        holder: 持有者标识（建议 pid+时间戳，用于安全释放）
        ttl_seconds: 持锁上限；超时后别的进程可抢占（防止持有者崩溃留下死锁）
    """
    init_store()
    conn = get_connection()
    try:
        now = datetime.now()
        now_s = now.strftime("%Y-%m-%d %H:%M:%S")
        expires_s = (now + timedelta(seconds=ttl_seconds)).strftime("%Y-%m-%d %H:%M:%S")
        # 先清过期锁：持有者被杀/崩溃时的兜底。
        # 用 <= 而非 <：时间戳只到秒，若用 <，恰好同一秒到期的锁会漏删。
        conn.execute("DELETE FROM task_locks WHERE expires_at <= ?", (now_s,))
        cur = conn.execute(
            "INSERT OR IGNORE INTO task_locks (task, holder, acquired_at, expires_at) "
            "VALUES (?, ?, ?, ?)", (task, holder, now_s, expires_s))
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def release_task_lock(task: str, holder: str = None):
    """释放锁。传 holder 时只释放自己持有的（避免误删超时后已被抢占的锁）"""
    init_store()
    conn = get_connection()
    try:
        if holder:
            conn.execute("DELETE FROM task_locks WHERE task = ? AND holder = ?",
                         (task, holder))
        else:
            conn.execute("DELETE FROM task_locks WHERE task = ?", (task,))
        conn.commit()
    finally:
        conn.close()


# ================================================================
# 全市场资金流 / 行业资金流快照（盘后采集，盘中只读）
# ================================================================

def save_fund_flow_snapshot(snapshot_date: str, rows: list[dict],
                            source: str = "ths") -> int:
    """写入某交易日的全市场资金流快照（同 date+code 覆盖）。

    rows: [{stock_code, stock_name, net_today, net_3d, net_5d, net_10d,
            amount, change_pct, price}, ...]
    `snapshot_date` 是**数据所属交易日**，与 `captured_at`（采集时刻）分开存——
    盘后 15:40 采集时两者同一天，但若补采历史数据就会不同，混在一起会误导。
    """
    if not snapshot_date or not rows:
        return 0
    init_store()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    try:
        payload = []
        for r in rows:
            code = str(r.get("stock_code") or "").strip()
            if not code:
                continue
            payload.append((
                snapshot_date, code, r.get("stock_name") or "",
                r.get("net_today"), r.get("net_3d"), r.get("net_5d"), r.get("net_10d"),
                r.get("amount"), r.get("change_pct"), r.get("price"),
                source, now, now))
        if not payload:
            return 0
        conn.executemany(
            """INSERT INTO fund_flow_snapshot
               (snapshot_date, stock_code, stock_name, net_today, net_3d, net_5d,
                net_10d, amount, change_pct, price, source, captured_at, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(snapshot_date, stock_code) DO UPDATE SET
                 stock_name=excluded.stock_name, net_today=excluded.net_today,
                 net_3d=excluded.net_3d, net_5d=excluded.net_5d, net_10d=excluded.net_10d,
                 amount=excluded.amount, change_pct=excluded.change_pct,
                 price=excluded.price, source=excluded.source,
                 captured_at=excluded.captured_at""",
            payload)
        conn.commit()
        return len(payload)
    finally:
        conn.close()


def save_industry_flow_snapshot(snapshot_date: str, rows: list[dict],
                                source: str = "ths") -> int:
    """写入某交易日的行业资金流快照（同 date+industry 覆盖）"""
    if not snapshot_date or not rows:
        return 0
    init_store()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    try:
        payload = []
        for r in rows:
            ind = str(r.get("industry") or "").strip()
            if not ind:
                continue
            payload.append((snapshot_date, ind, r.get("net_amount"),
                            r.get("change_pct"), r.get("leader") or "",
                            source, now, now))
        if not payload:
            return 0
        conn.executemany(
            """INSERT INTO industry_flow_snapshot
               (snapshot_date, industry, net_amount, change_pct, leader,
                source, captured_at, created_at)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(snapshot_date, industry) DO UPDATE SET
                 net_amount=excluded.net_amount, change_pct=excluded.change_pct,
                 leader=excluded.leader, source=excluded.source,
                 captured_at=excluded.captured_at""",
            payload)
        conn.commit()
        return len(payload)
    finally:
        conn.close()


def latest_fund_flow_date() -> str | None:
    """最近一次已落库的资金流快照日期；没有则 None。

    选股（9:25/11:30）应当用**最近一个已完整采集的交易日**的资金流——
    盘前本来就没有当日数据，用昨天收盘的完整数据是正确口径，而不是"陈旧"。
    """
    init_store()
    conn = get_connection()
    try:
        row = conn.execute("SELECT MAX(snapshot_date) AS d FROM fund_flow_snapshot").fetchone()
        return row["d"] if row and row["d"] else None
    finally:
        conn.close()


def load_fund_flow_snapshot(snapshot_date: str = None,
                            codes: list[str] = None) -> tuple[str | None, dict]:
    """读取资金流快照，返回 (实际日期, {code: record})。

    `snapshot_date=None` 时用最近一次落库的日期（见 `latest_fund_flow_date`），
    并把实际使用的日期一并返回——调用方必须能说出"这份资金流是哪天的"。
    """
    date = snapshot_date or latest_fund_flow_date()
    if not date:
        return None, {}
    init_store()
    conn = get_connection()
    try:
        sql = "SELECT * FROM fund_flow_snapshot WHERE snapshot_date = ?"
        params: list = [date]
        if codes:
            clean = [str(c).strip() for c in codes if str(c).strip()]
            if not clean:
                return date, {}
            sql += f" AND stock_code IN ({','.join('?' * len(clean))})"
            params += clean
        rows = conn.execute(sql, params).fetchall()
        return date, {r["stock_code"]: dict(r) for r in rows}
    finally:
        conn.close()


def latest_industry_flow_date() -> str | None:
    """最近一次已落库的**行业**资金流快照日期。

    必须单独查这张表：行业快照与个股快照是两次写入，若从个股表推日期，
    遇到"只有行业数据"（或个股窗口全失败）时会误判成"没有快照"而整段读空。
    """
    init_store()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT MAX(snapshot_date) AS d FROM industry_flow_snapshot").fetchone()
        return row["d"] if row and row["d"] else None
    finally:
        conn.close()


def load_industry_flow_snapshot(snapshot_date: str = None) -> tuple[str | None, list[dict]]:
    """读取行业资金流快照，返回 (实际日期, rows[按净额降序])"""
    date = snapshot_date or latest_industry_flow_date()
    if not date:
        return None, []
    init_store()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM industry_flow_snapshot WHERE snapshot_date = ? "
            "ORDER BY net_amount DESC", (date,)).fetchall()
        return date, [dict(r) for r in rows]
    finally:
        conn.close()


# ── 估值/规模截面快照（PE / 市值 / 换手率 IC 的前置数据）──

def save_market_snapshot(snapshot_date: str, rows: list[dict],
                         source: str = "sina_sector") -> int:
    """写入某交易日的全市场估值截面（同 date+code 覆盖）。

    rows: [{code/stock_code, name/stock_name, industry, board, price, change_pct,
            pe, pb, market_cap, float_cap, turnover, amount}, ...]
    `snapshot_date` 是**数据所属交易日**，与 `captured_at` 分开——
    盘后采集时两者同一天，补采历史时会不同，混在一起会误导。

    ⚠️ 单位为**元**（金额列）与**百分数**（turnover / change_pct）：
    `data.sina_source` 的 DataFrame 给的是"亿元"，转换由调用方
    （`core.scheduler.save_daily_market_snapshot`）负责，见那里的注释。
    """
    if not snapshot_date or not rows:
        return 0
    init_store()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = []
    skipped_no_quote = 0
    for r in rows:
        code = str(r.get("stock_code") or r.get("code") or "").strip()
        if not code:
            continue
        # ⚠️ 结构性拦截（"缺就是缺"）：`price` **显式给成 0/负数**不是"0 元"，是没有报价
        #    ⇒ 不入库。上游（新浪行业明细）对退市票长期返回 `trade=0`，此前直通落库，
        #    导致 market_snapshot 里常驻 186 只 `price=0/change_pct=0` 的退市票
        #    （2026-10-26 实测 389 只零涨幅中 186 只属此类），下游按价格/涨幅筛全被污染。
        #    ⚠️ 只在**显式给了数值**时拦：`price=None` 是"没提供"（PE/PB 补采只写估值列），
        #    不能当坏值丢掉，否则会误删正常行。
        _p = r.get("price")
        if isinstance(_p, (int, float)) and not isinstance(_p, bool) and _p <= 0:
            skipped_no_quote += 1
            continue
        payload.append((
            snapshot_date, code,
            r.get("stock_name") or r.get("name") or "",
            r.get("industry") or "", r.get("board") or "",
            r.get("price"), r.get("change_pct"), r.get("pe"), r.get("pb"),
            r.get("market_cap"), r.get("float_cap"), r.get("turnover"),
            r.get("amount"), r.get("industry_ts") or "", source, now, now))
    if skipped_no_quote:
        _logger.warning("[market_snapshot] %s 丢弃 %d 只无有效报价(price<=0)的标的"
                       "（退市/停牌无报价，不是 0 元）", snapshot_date, skipped_no_quote)
    if not payload:
        return 0
    conn = get_connection()
    try:
        conn.executemany(
            """INSERT INTO market_snapshot
               (snapshot_date, stock_code, stock_name, industry, board, price,
                change_pct, pe, pb, market_cap, float_cap, turnover, amount,
                industry_ts, source, captured_at, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(snapshot_date, stock_code) DO UPDATE SET
                 stock_name=excluded.stock_name, industry=excluded.industry,
                 board=excluded.board, price=excluded.price,
                 change_pct=excluded.change_pct, pe=excluded.pe, pb=excluded.pb,
                 market_cap=excluded.market_cap, float_cap=excluded.float_cap,
                 turnover=excluded.turnover, amount=excluded.amount,
                 industry_ts=excluded.industry_ts,
                 source=excluded.source, captured_at=excluded.captured_at""",
            payload)
        conn.commit()
        return len(payload)
    finally:
        conn.close()


def save_news_reviews(review_date: str, rows: list[dict],
                      model: str = "") -> int:
    """写入某日期的行业评估（同 date+sector 覆盖）。

    rows: [{sector, verdict, summary, evidence, news_count}, ...]
    `review_date` 是**数据所属日期**，与 `created_at` 分开——
    补跑历史日期的评估时两者不同（与 market_snapshot 同一原则）。
    """
    if not review_date or not rows:
        return 0
    init_store()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = []
    for r in rows:
        sector = str(r.get("sector") or "").strip()
        verdict = str(r.get("verdict") or "").strip()
        if not sector or not verdict:
            continue
        evidence = r.get("evidence")
        if not isinstance(evidence, str):
            evidence = json.dumps(evidence or [], ensure_ascii=False)
        payload.append((
            review_date, sector, verdict, r.get("summary") or "",
            evidence, int(r.get("news_count") or 0), model, now))
    if not payload:
        return 0
    conn = get_connection()
    try:
        conn.executemany(
            """INSERT INTO news_industry_reviews
               (review_date, sector, verdict, summary, evidence,
                news_count, model, created_at)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(review_date, sector) DO UPDATE SET
                 verdict=excluded.verdict, summary=excluded.summary,
                 evidence=excluded.evidence, news_count=excluded.news_count,
                 model=excluded.model, created_at=excluded.created_at""",
            payload)
        conn.commit()
        return len(payload)
    finally:
        conn.close()


def get_news_reviews(review_date: str = None) -> tuple[str | None, list[dict]]:
    """读行业评估。`review_date=None` → 返回**最新一期**。

    Returns: (review_date, rows)；没有任何数据 → (None, [])。
    evidence 反序列化回列表（损坏时降级为空列表并保留原文在 evidence_raw）。
    """
    init_store()
    conn = get_connection()
    try:
        if review_date is None:
            row = conn.execute(
                "SELECT MAX(review_date) AS d FROM news_industry_reviews").fetchone()
            review_date = row["d"] if row else None
            if not review_date:
                return None, []
        rows = conn.execute(
            """SELECT * FROM news_industry_reviews
               WHERE review_date = ? ORDER BY id""", (review_date,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["evidence"] = json.loads(d.get("evidence") or "[]")
                if not isinstance(d["evidence"], list):
                    raise ValueError
            except (ValueError, json.JSONDecodeError):
                d["evidence"] = []
                d["evidence_raw"] = r["evidence"]
            out.append(d)
        return review_date, out
    finally:
        conn.close()


def news_review_dates(limit: int = 30) -> list[str]:
    """已有评估的日期（升序）。用于页面展示历史与"攒够样本没有"。"""
    init_store()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT review_date FROM news_industry_reviews "
            "WHERE review_date IS NOT NULL ORDER BY review_date").fetchall()
        dates = [r["review_date"] for r in rows]
        return dates[-limit:] if limit and len(dates) > limit else dates
    finally:
        conn.close()


def latest_industry_map(date: str = None) -> dict:
    """`{code: industry}` —— 每只股票**最近一次已知**的行业归属。

    用途：全量域（`data.full_market`）从新浪拿不到行业时用历史映射补，
    并且**只补非空值**（空串不是行业，是"没采到"）。

    ⚠️ 默认是**历史并集**（每只取 `MAX(snapshot_date)` 那天，SQLite 的
       bare-column + MAX 语义保证取到的是同一行），**不是"最新那一天"**：
       旧实现只读最新一天的截面，一旦当天行业采集降级就会把前面
       几百只的已知行业一起丢掉（实测并集 2,809 只 vs 单日 2,624 只）。
       行业名一年也变不了几次，用旧值 ≫ 标"未知"。
    """
    init_store()
    conn = get_connection()
    try:
        if date:
            return {r["stock_code"]: r["industry"] for r in conn.execute(
                "SELECT stock_code, industry FROM market_snapshot "
                "WHERE snapshot_date = ? AND industry IS NOT NULL AND industry <> ''",
                (date,))}
        return {r["stock_code"]: r["industry"] for r in conn.execute(
            "SELECT stock_code, industry, MAX(snapshot_date) AS d FROM market_snapshot "
            "WHERE industry IS NOT NULL AND industry <> '' "
            "GROUP BY stock_code")}
    finally:
        conn.close()


def market_snapshot_dates(limit: int = 400) -> list[str]:
    """已落库的截面日期（升序）。用于判断"攒够样本没有"。"""
    init_store()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT snapshot_date FROM market_snapshot "
            "WHERE snapshot_date IS NOT NULL ORDER BY snapshot_date").fetchall()
        dates = [r["snapshot_date"] for r in rows]
        return dates[-limit:] if limit and len(dates) > limit else dates
    finally:
        conn.close()


def load_market_snapshot(snapshot_date: str = None,
                         codes: list[str] = None) -> tuple[str | None, list[dict]]:
    """读取某日估值截面，返回 (实际日期, rows)。

    ⚠️ `snapshot_date=None` 时严禁"顺手取最近一天"当默认——调用方必须显式
    使用返回的 `date`，否则会把不同日期的截面混进同一次计算（前视/后视都可能）。
    """
    date = snapshot_date
    if not date:
        dates = market_snapshot_dates(limit=1)
        date = dates[-1] if dates else None
    if not date:
        return None, []
    init_store()
    conn = get_connection()
    try:
        sql = "SELECT * FROM market_snapshot WHERE snapshot_date = ?"
        params: list = [date]
        if codes:
            clean = [str(c).strip() for c in codes if str(c).strip()]
            if not clean:
                return date, []
            sql += f" AND stock_code IN ({','.join('?' * len(clean))})"
            params += clean
        rows = conn.execute(sql, params).fetchall()
        return date, [dict(r) for r in rows]
    finally:
        conn.close()


# ================================================================
# 采集覆盖率（P2-7）
# ================================================================

COVERAGE_OK = "ok"
COVERAGE_PARTIAL = "partial"
COVERAGE_DEGRADED = "degraded"
COVERAGE_FAILED = "failed"
_COVERAGE_STATUSES = (COVERAGE_OK, COVERAGE_PARTIAL, COVERAGE_DEGRADED, COVERAGE_FAILED)


def save_data_coverage(data_date: str, dataset: str, *, got: int,
                       total: int | None = None, status: str = COVERAGE_OK,
                       source: str = "", note: str = "") -> float | None:
    """记录「某日某数据集采到多少 / 全市场多少」。同 (data_date, dataset) 覆盖。

    Args:
        total: 全市场应有条数。**None = 来源没给**（如新浪板块接口不返回总数）⇒
               ratio 记 NULL，**绝不写 0** —— 0 会把"不知道"伪装成"覆盖率为零"。
        status: ok / partial / degraded / failed。非法值会被夹到 `degraded` 并写进 note，
                不静默接受（否则健康页会显示一个不存在的状态）。

    Returns:
        覆盖率（0~1）或 None。

    ⚠️ 为什么 `got` 允许为 0：**失败也要落库**。"0 只"本身是一条重要事实
       （`market_snapshot` 只有 0 行时，光看那张表只能得出"今天没数据"，
       看不出是"任务没跑"还是"跑了但源挂了"）。
    """
    d = str(data_date or "").strip()
    ds = str(dataset or "").strip()
    if not d or not ds:
        raise ValueError("save_data_coverage 需要 data_date 与 dataset")
    g = int(got or 0)
    if g < 0:
        raise ValueError("got 不能为负")
    tot = None if total is None else int(total)
    if tot is not None and tot <= 0:
        tot = None                                   # 0/负数当"未知"，不是"全市场 0 只"
    st = str(status or "").strip()
    extra = []
    if st not in _COVERAGE_STATUSES:
        extra.append(f"未知状态 {st!r} 已按 degraded 记录")
        st = COVERAGE_DEGRADED
    ratio = round(g / tot, 4) if tot else None
    # 状态与数据的一致性：0 行却报 ok 是自相矛盾的（本项目踩过"0 只记完成"）
    if g == 0 and st == COVERAGE_OK:
        st = COVERAGE_FAILED
        extra.append("0 行不得记 ok，已改记 failed")
    note_txt = "；".join([x for x in (note, *extra) if x])[:500]

    init_store()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO data_coverage
               (data_date, dataset, source, got, total, ratio, status, note,
                captured_at, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(data_date, dataset) DO UPDATE SET
                 source=excluded.source, got=excluded.got, total=excluded.total,
                 ratio=excluded.ratio, status=excluded.status, note=excluded.note,
                 captured_at=excluded.captured_at""",
            (d, ds, source, g, tot, ratio, st, note_txt, now, now))
        conn.commit()
        return ratio
    finally:
        conn.close()


def load_data_coverage(dataset: str = None, limit: int = 30) -> list[dict]:
    """最近 N 条覆盖率记录（按数据日期倒序）。`dataset=None` = 全部。"""
    init_store()
    conn = get_connection()
    try:
        sql = "SELECT * FROM data_coverage"
        params: list = []
        if dataset:
            sql += " WHERE dataset = ?"
            params.append(dataset)
        sql += " ORDER BY data_date DESC, dataset ASC LIMIT ?"
        params.append(max(1, int(limit)))
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def data_coverage_summary(dataset: str = "market_snapshot") -> dict:
    """该数据集的覆盖概况 —— 给健康检查用。

    Returns: {"latest": 最新一条 | None, "days": 有记录的天数,
              "low_ratio_days": 覆盖率 <90% 的天数, "missing_days": ratio 为 NULL 的天数}
    ⚠️ `missing_days` 与 `low_ratio_days` 分开统计：前者是"不知道"，后者是"确实低"，
       混在一起会把"没数据"读成"质量差"。
    """
    init_store()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM data_coverage WHERE dataset = ? ORDER BY data_date DESC",
            (dataset,)).fetchall()
        rows = [dict(r) for r in rows]
    finally:
        conn.close()
    low = sum(1 for r in rows if r.get("ratio") is not None and r["ratio"] < 0.9)
    missing = sum(1 for r in rows if r.get("ratio") is None)
    return {"latest": rows[0] if rows else None, "days": len(rows),
            "low_ratio_days": low, "missing_days": missing}



# ================================================================
# 自定义定时任务（L2 扩展层）
# ================================================================
#
# 与建表语句里的说明对应：这张表是**配置**，不是累积型数据，所以
#   1) 不参与 `cleanup()` 的保留策略（同 task_locks）；
#   2) 行数极少（个位数），无需索引以外的优化。
#
# ⚠️ cron 解析失败时**不返回 0/默认值**，而是把 `cron` 置为 None 并把原因放进
#    `error`：调度器据此跳过该任务，UI 据此标红。若悄悄塞一个默认时间，
#    用户会看到一个"自己没设置过"的执行时刻（"缺就是缺，不补造"）。

_JOB_FIELDS = ("job_id", "name", "mode", "cron", "enabled",
               "trading_day_only", "note", "created_at", "updated_at")


def _row_to_scheduled_job(row) -> dict:
    """SQLite 行 → dict（cron JSON 反序列化失败不吞错）"""
    job = {k: row[k] for k in _JOB_FIELDS if k in row.keys()}
    raw = job.get("cron")
    job["cron_raw"] = raw
    job["enabled"] = bool(job.get("enabled", 1))
    job["trading_day_only"] = bool(job.get("trading_day_only", 1))
    try:
        parsed = json.loads(raw) if raw else None
        if not isinstance(parsed, dict):
            raise ValueError("cron 不是 JSON 对象")
        job["cron"] = parsed
        job["error"] = ""
    except Exception as e:                      # noqa: BLE001
        job["cron"] = None
        job["error"] = f"cron 无法解析：{e}"
        _logger.warning("[存储] 自定义任务 %s 的 cron 无法解析: %s", job.get("job_id"), e)
    return job


def list_scheduled_jobs(only_enabled: bool = False) -> list[dict]:
    """列出自定义定时任务（按 job_id 排序，便于稳定对比/展示）"""
    init_store()
    conn = get_connection()
    try:
        sql = "SELECT * FROM scheduled_jobs"
        if only_enabled:
            sql += " WHERE enabled = 1"
        sql += " ORDER BY job_id"
        return [_row_to_scheduled_job(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def get_scheduled_job(job_id: str) -> dict:
    """取单条，不存在返回 {}"""
    init_store()
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM scheduled_jobs WHERE job_id = ?",
                           (job_id,)).fetchone()
        return _row_to_scheduled_job(row) if row else {}
    finally:
        conn.close()


def upsert_scheduled_job(job: dict) -> str:
    """新增或更新（按 job_id）。cron 若为 dict 会自动序列化。返回 job_id。"""
    job_id = str(job.get("job_id") or "").strip()
    if not job_id:
        raise ValueError("job_id 不能为空")
    init_store()
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cron = job.get("cron")
    cron_text = cron if isinstance(cron, str) else json.dumps(cron or {}, ensure_ascii=False)
    try:
        exists = conn.execute("SELECT 1 FROM scheduled_jobs WHERE job_id = ?",
                              (job_id,)).fetchone()
        if exists:
            conn.execute(
                """UPDATE scheduled_jobs SET name=?, mode=?, cron=?, enabled=?,
                       trading_day_only=?, note=?, updated_at=? WHERE job_id=?""",
                (job.get("name") or job_id, job.get("mode") or "", cron_text,
                 1 if job.get("enabled", True) else 0,
                 1 if job.get("trading_day_only", True) else 0,
                 job.get("note") or "", now, job_id))
        else:
            conn.execute(
                """INSERT INTO scheduled_jobs
                       (job_id, name, mode, cron, enabled, trading_day_only,
                        note, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (job_id, job.get("name") or job_id, job.get("mode") or "", cron_text,
                 1 if job.get("enabled", True) else 0,
                 1 if job.get("trading_day_only", True) else 0,
                 job.get("note") or "", now, now))
        conn.commit()
    finally:
        conn.close()
    return job_id


def set_scheduled_job_enabled(job_id: str, enabled: bool) -> bool:
    """只改开关。返回是否命中该行（False=任务不存在）。"""
    init_store()
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE scheduled_jobs SET enabled=?, updated_at=? WHERE job_id=?",
            (1 if enabled else 0, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), job_id))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_scheduled_job(job_id: str) -> bool:
    """删除。返回是否真的删掉了（False=本来就不存在）。"""
    init_store()
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM scheduled_jobs WHERE job_id = ?", (job_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ================================================================
# 保留策略与统计
# ================================================================

def cleanup(retention_days: dict = None):
    """按保留策略清理旧数据（可由调度周期性调用）

    新表必须在这里登记策略——**除非它是配置表**（`task_locks`、`scheduled_jobs`
    这类"人工维护的少量行"），配置表被按日期清理掉会直接丢用户设置。
    """
    retention = {"news_items": NEWS_RETENTION_DAYS,
                 "chat_logs": CHAT_RETENTION_DAYS,
                 "llm_calls": LLM_CALL_RETENTION_DAYS,
                 "analyses": ANALYSES_RETENTION_DAYS,
                 "screen_picks": SCREEN_PICKS_RETENTION_DAYS,
                 "fund_flow_snapshot": FUND_FLOW_SNAPSHOT_RETENTION_DAYS,
                 "industry_flow_snapshot": FUND_FLOW_SNAPSHOT_RETENTION_DAYS,
                 "market_snapshot": MARKET_SNAPSHOT_RETENTION_DAYS,
                 "news_industry_reviews": NEWS_REVIEW_RETENTION_DAYS}
    if retention_days:
        retention.update(retention_days)
    init_store()
    conn = get_connection()
    deleted = {}
    # 过期判据按表选列：绝大多数表用 `created_at`（"什么时候写进来的"），
    # 但**快照类表要用 `snapshot_date`（"数据属于哪一天"）**——
    # 补采历史数据时 created_at 是"今天"，按创建时间清理会让 2000 年的数据永远留在库里。
    date_col = {"fund_flow_snapshot": "snapshot_date",
                "industry_flow_snapshot": "snapshot_date",
                "market_snapshot": "snapshot_date",
                "news_industry_reviews": "review_date"}
    try:
        for table, days in retention.items():
            col = date_col.get(table, "created_at")
            if col == "snapshot_date":
                cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            else:
                cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
            cur = conn.execute(f"DELETE FROM {table} WHERE {col} < ?", (cutoff,))
            deleted[table] = cur.rowcount
        # 会话维表不按 created_at 过期（老会话可能一直在用），
        # 只在消息被清理后回收"已无消息"的空壳会话，否则会越积越多
        sess_cutoff = (datetime.now() - timedelta(days=retention["chat_logs"])
                       ).strftime("%Y-%m-%d %H:%M:%S")
        cur = conn.execute(
            """DELETE FROM chat_sessions WHERE updated_at < ?
               AND NOT EXISTS (SELECT 1 FROM chat_logs l
                               WHERE l.session_id = chat_sessions.session_id)""",
            (sess_cutoff,))
        deleted["chat_sessions"] = cur.rowcount
        conn.commit()
    finally:
        conn.close()
    if any(deleted.values()):
        _logger.info("[存储] 保留策略清理: %s", deleted)
    return deleted


def store_stats() -> dict:
    """全库统计（配置页数据卡）。各表时间列名不同：news/chat 用 created_at，features 用 updated_at"""
    init_store()
    conn = get_connection()
    try:
        time_col = {"news_items": "created_at", "chat_logs": "created_at",
                    "daily_features": "updated_at", "runs": "created_at",
                    "llm_calls": "created_at", "alerts": "created_at",
                    "analyses": "created_at", "chat_sessions": "updated_at",
                    "screen_picks": "created_at",
                    # 资金流快照看 **snapshot_date**（数据所属交易日）而不是采集时间：
                    # 健康检查要判断"今天的快照有没有生成"，采集时间串了日期就看不出来。
                    "fund_flow_snapshot": "snapshot_date",
                    "industry_flow_snapshot": "snapshot_date"}
        out = {}
        for table, col in time_col.items():
            total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            latest = conn.execute(
                f"SELECT MAX({col}) FROM {table}").fetchone()[0]
            out[table] = {"rows": total, "latest": latest}
        return out
    finally:
        conn.close()


# ================================================================
# AI 交易联赛（S1 数据底座，2026-10-14）
# 设计与口径见 output/多AI交易联赛_开发计划_2026-10-14.md
# ================================================================

def _ensure_column(conn, table: str, column: str, ddl: str) -> None:
    """给**已存在**的表补列（迁移守卫）。

    ⚠️ 为什么必须有它：init_store 的版本号是 `_SCHEMA` 文本的哈希，改了 schema
    会重放建表语句，但语句全是 `CREATE TABLE IF NOT EXISTS` ⇒ 只对**新表**生效；
    **给既有表加列永远不会生效**，而版本号却已"迁移成功"——这就是假安全
    （2026-10-14 发现，当时全仓 ALTER TABLE 为 0 处）。
    本次 5 张 AI 表都是新表用不上它；但以后给 paper_* 等旧表加列**必须**走这里。
    """
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _ai_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------- ① 人格 ----------------

def upsert_ai_persona(p: dict) -> bool:
    """新增/更新一个 AI 人格。name 同时是 paper_accounts.name（复用账户体系）。

    ⚠️ kind='random' 是随机对照组（不调 LLM）；frequency='intraday' 的 AI
    与 daily 的**不可混榜**（口径 3）——这两个字段是排行榜分组与结算口径的依据。
    """
    init_store()
    conn = get_connection()
    try:
        now = _ai_now()
        conn.execute(
            """INSERT INTO ai_personas (name, label, kind, model, style_prompt,
                    risk_bias, max_position, info_scope, frequency, initial_cash,
                    enabled, created_at, updated_at)
               VALUES (:name, :label, :kind, :model, :style_prompt, :risk_bias,
                    :max_position, :info_scope, :frequency, :initial_cash,
                    :enabled, :created_at, :updated_at)
               ON CONFLICT(name) DO UPDATE SET
                    label=excluded.label, kind=excluded.kind, model=excluded.model,
                    style_prompt=excluded.style_prompt, risk_bias=excluded.risk_bias,
                    max_position=excluded.max_position, info_scope=excluded.info_scope,
                    frequency=excluded.frequency,
                    -- ⚠️ initial_cash 只在首次建账时生效：改初始资金不能追溯重置
                    --    已有账户的账本（否则历史净值曲线全部作废）
                    initial_cash=ai_personas.initial_cash,
                    enabled=excluded.enabled, updated_at=excluded.updated_at""",
            {"name": p.get("name"), "label": p.get("label") or p.get("name"),
             "kind": p.get("kind") or "llm", "model": p.get("model"),
             "style_prompt": p.get("style_prompt") or "",
             "risk_bias": p.get("risk_bias") or "balanced",
             "max_position": p.get("max_position"),
             "info_scope": p.get("info_scope") or "all",
             "frequency": p.get("frequency") or "daily",
             "initial_cash": float(p.get("initial_cash") or 1000000),
             "enabled": 1 if p.get("enabled", True) else 0,
             "created_at": now, "updated_at": now})
        conn.commit()
        return True
    finally:
        conn.close()


def _ai_persona_row_to_dict(r) -> dict:
    keys = ("name", "label", "kind", "model", "style_prompt", "risk_bias",
            "max_position", "info_scope", "frequency", "initial_cash", "enabled")
    return dict(zip(keys, r))


def list_ai_personas(*, enabled_only: bool = True, kind: str = None) -> list[dict]:
    init_store()
    conn = get_connection()
    try:
        sql = ("SELECT name, label, kind, model, style_prompt, risk_bias, "
               "max_position, info_scope, frequency, initial_cash, enabled "
               "FROM ai_personas WHERE 1=1")
        args: list = []
        if enabled_only:
            sql += " AND enabled=1"
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        sql += " ORDER BY name"
        return [_ai_persona_row_to_dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def get_ai_persona(name: str) -> dict | None:
    init_store()
    conn = get_connection()
    try:
        r = conn.execute(
            "SELECT name, label, kind, model, style_prompt, risk_bias, "
            "max_position, info_scope, frequency, initial_cash, enabled "
            "FROM ai_personas WHERE name=?", (name,)).fetchone()
        return _ai_persona_row_to_dict(r) if r else None
    finally:
        conn.close()


def delete_ai_persona(name: str) -> bool:
    """删除人格。⚠️ 只删人格定义，**不删账户账本**（paper_* 表由账户页自己管理，
    避免"误删 AI 把交易历史一起带走"）。返回是否真的删了行。"""
    init_store()
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM ai_personas WHERE name=?", (name,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ---------------- ② 研究预算（C3 唯一实现） ----------------

def consume_ai_budget(persona: str, kind: str, *, n: int = 1,
                      trade_date: str, quota: int) -> tuple[bool, dict]:
    """申请并扣减当日额度。**所有资源扣减必须经此函数**（唯一实现）。

    ⚠️ 判定与扣减在**同一条 UPDATE** 里完成（`used + n <= quota` 作 WHERE 条件）
    —— SQLite 单条 UPDATE 原子，先跑完再记账会出现"并发时超额"。
    ⚠️ quota 传**当前配额**：行已存在时同步更新它 ⇒ 当天改配额当天生效，
    而历史日期的行永不被碰（配额快照语义：改配额不影响已过日期）。

    Returns:
        (是否获批, 当前额度视图 {"used","quota","remaining"})
    """
    if n <= 0:
        return False, {"used": 0, "quota": int(quota), "remaining": int(quota)}
    init_store()
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO ai_budget (persona, budget_date, kind, used, quota, updated_at)
               VALUES (?, ?, ?, 0, ?, ?)
               ON CONFLICT(persona, budget_date, kind) DO NOTHING""",
            (persona, trade_date, kind, int(quota), _ai_now()))
        cur = conn.execute(
            """UPDATE ai_budget SET used = used + ?, quota = ?, updated_at = ?
               WHERE persona = ? AND budget_date = ? AND kind = ?
                 AND used + ? <= quota""",
            (int(n), int(quota), _ai_now(), persona, trade_date, kind, int(n)))
        conn.commit()
        granted = cur.rowcount == 1
        row = conn.execute(
            "SELECT used, quota FROM ai_budget "
            "WHERE persona=? AND budget_date=? AND kind=?",
            (persona, trade_date, kind)).fetchone()
        used, q = (row[0], row[1]) if row else (0, int(quota))
        return granted, {"used": used, "quota": q, "remaining": max(0, q - used)}
    finally:
        conn.close()


def ai_budget_view(persona: str, trade_date: str) -> dict:
    """当日各 kind 的额度视图（给 AI 的 get_my_budget() 工具与 UI 用）。
    没用过的 kind 也**必须出现**（计 0）——否则 AI 看不到"还有多少额度"。"""
    init_store()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT kind, used, quota FROM ai_budget "
            "WHERE persona=? AND budget_date=?", (persona, trade_date)).fetchall()
        return {k: {"used": u, "quota": q, "remaining": max(0, q - u)}
                for k, u, q in rows}
    finally:
        conn.close()


# ---------------- ③④ 唤醒条件与事件（C4） ----------------

def upsert_ai_wake_condition(persona: str, cond: dict) -> int:
    """新增/更新唤醒条件。**幂等键 = `(persona, code, kind)`**（2026-10-16 修订）。

    ⚠️ 原实现的冲突键是 `(persona, cond_id)`，而 `cond_id` 由本函数自增、
    `persist_round` 传进来的 cond **不带 cond_id** ⇒ 每次调用都分配新号
    ⇒ **永远 INSERT、从不 UPDATE** ⇒ 同一标的的条件**无限累积**
    （实测同一只票出现两个止损位：600519 的 1184.01 与 1250.0、600028 的 5.2 与 5.0
     ⇒ 风控语义被破坏：到底该按哪个价位止损？）。

    Returns: cond_id
    """
    init_store()
    conn = get_connection()
    try:
        now = _ai_now()
        code, kind = cond.get("code"), cond.get("kind")
        cond_id = cond.get("cond_id")
        if cond_id is None and code is not None and kind is not None:
            # 同一 (persona, code, kind) **复用同一行** ⇒ 走 UPDATE 而不是不断新增
            row = conn.execute(
                "SELECT cond_id FROM ai_wake_conditions "
                "WHERE persona=? AND code=? AND kind=? ORDER BY cond_id LIMIT 1",
                (persona, code, kind)).fetchone()
            if row is not None:
                cond_id = int(row[0])
        if cond_id is None:
            nxt = conn.execute(
                "SELECT COALESCE(MAX(cond_id), 0) + 1 FROM ai_wake_conditions "
                "WHERE persona=?", (persona,)).fetchone()[0]
            cond_id = int(nxt)
        cond = {**cond, "cond_id": int(cond_id)}
        conn.execute(
            """INSERT INTO ai_wake_conditions (persona, cond_id, code, kind,
                    threshold, threshold_pct, wake_at, once, armed, expires_on, note,
                    created_at, updated_at)
               VALUES (:persona, :cond_id, :code, :kind, :threshold, :threshold_pct,
                    :wake_at, :once, :armed, :expires_on, :note,
                    :created_at, :updated_at)
               ON CONFLICT(persona, cond_id) DO UPDATE SET
                    code=excluded.code, kind=excluded.kind,
                    threshold=excluded.threshold, threshold_pct=excluded.threshold_pct,
                    wake_at=excluded.wake_at,
                    once=excluded.once, armed=excluded.armed,
                    expires_on=excluded.expires_on, note=excluded.note,
                    updated_at=excluded.updated_at""",
            {"persona": persona, "cond_id": int(cond["cond_id"]),
             "code": cond.get("code"), "kind": cond.get("kind"),
             "threshold": cond.get("threshold"),
             "threshold_pct": cond.get("threshold_pct"),
             "wake_at": cond.get("wake_at"),
             "once": 1 if cond.get("once", True) else 0,
             "armed": 1 if cond.get("armed", True) else 0,
             "expires_on": cond.get("expires_on"), "note": cond.get("note"),
             "created_at": now, "updated_at": now})
        conn.commit()
        return int(cond["cond_id"])
    finally:
        conn.close()


def list_ai_wake_conditions(*, persona: str = None, armed_only: bool = True,
                            trade_date: str = None) -> list[dict]:
    """列出唤醒条件。trade_date 给了就顺带过滤掉已过期（expires_on < trade_date）的条目
    —— 判定侧不判过期条件，但**不在这里删行**（保留台账供复盘）。"""
    init_store()
    conn = get_connection()
    try:
        sql = ("SELECT persona, cond_id, code, kind, threshold, threshold_pct, "
               "wake_at, once, armed, expires_on, note "
               "FROM ai_wake_conditions WHERE 1=1")
        args: list = []
        if persona:
            sql += " AND persona=?"
            args.append(persona)
        if armed_only:
            sql += " AND armed=1"
        if trade_date:
            sql += " AND (expires_on IS NULL OR expires_on >= ?)"
            args.append(trade_date)
        sql += " ORDER BY persona, cond_id"
        keys = ("persona", "cond_id", "code", "kind", "threshold", "threshold_pct",
                "wake_at", "once", "armed", "expires_on", "note")
        return [dict(zip(keys, r)) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def deactivate_expired_wake_conditions(trade_date: str) -> int:
    """把 expires_on < trade_date 的条件标记失活（armed=0）。返回失活条数。
    ⚠️ 只标记不删除——台账要留着复盘"AI 定的条件后来怎么样了"。"""
    init_store()
    conn = get_connection()
    try:
        cur = conn.execute(
            """UPDATE ai_wake_conditions SET armed=0, updated_at=?
               WHERE armed=1 AND expires_on IS NOT NULL AND expires_on < ?""",
            (_ai_now(), trade_date))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def record_wake_event(persona: str, trade_date: str, code: str, cond_id: int,
                      *, hit_at: str, trigger_px: float = None) -> bool:
    """记录唤醒事件（幂等）。Returns: 是否**新触发**（False=当天已触发过）。

    口径 4 的锚点：主键 (persona, trade_date, code, cond_id) ⇒ 进程重启后
    重复扫描同一天不会重复触发。
    """
    init_store()
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT OR IGNORE INTO ai_wake_events
               (persona, trade_date, code, cond_id, hit_at, trigger_px, fired)
               VALUES (?, ?, ?, ?, ?, ?, 0)""",
            (persona, trade_date, code, int(cond_id), hit_at, trigger_px))
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def mark_wake_event(persona: str, trade_date: str, code: str, cond_id: int,
                    *, fired: bool = None, action: str = None,
                    skip_reason: str = None) -> bool:
    """回填唤醒事件的结果（是否真跑了决策轮 / 动作摘要 / 跳过原因）。"""
    init_store()
    conn = get_connection()
    try:
        sets, args = [], []
        if fired is not None:
            sets.append("fired=?"); args.append(1 if fired else 0)
        if action is not None:
            sets.append("action=?"); args.append(action)
        if skip_reason is not None:
            sets.append("skip_reason=?"); args.append(skip_reason)
        if not sets:
            return False
        args += [persona, trade_date, code, int(cond_id)]
        cur = conn.execute(
            f"UPDATE ai_wake_events SET {', '.join(sets)} "
            "WHERE persona=? AND trade_date=? AND code=? AND cond_id=?", args)
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def list_wake_events(*, persona: str = None, trade_date: str = None) -> list[dict]:
    init_store()
    conn = get_connection()
    try:
        sql = ("SELECT persona, trade_date, code, cond_id, hit_at, trigger_px, "
               "action, fired, skip_reason FROM ai_wake_events WHERE 1=1")
        args: list = []
        if persona:
            sql += " AND persona=?"; args.append(persona)
        if trade_date:
            sql += " AND trade_date=?"; args.append(trade_date)
        sql += " ORDER BY trade_date DESC, hit_at DESC"
        keys = ("persona", "trade_date", "code", "cond_id", "hit_at",
                "trigger_px", "action", "fired", "skip_reason")
        return [dict(zip(keys, r)) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def list_wake_conditions(*, persona: str = None, armed_only: bool = True,
                         limit: int = 300) -> list[dict]:
    """`ai_wake_conditions` —— AI 埋的盘中触发条件（价格阈值/时点）。

    `armed_only=True` 只取 `armed=1`（当前仍生效的）。**不要把它当成"待办清单"**：
    未触发的条件可能只是价位没到，不代表系统有问题。
    """
    init_store()
    conn = get_connection()
    try:
        sql = ("SELECT persona, cond_id, code, kind, threshold, threshold_pct, "
               "wake_at, once, armed, expires_on, note, updated_at "
               "FROM ai_wake_conditions WHERE 1=1")
        args: list = []
        if persona:
            sql += " AND persona=?"; args.append(persona)
        if armed_only:
            sql += " AND armed=1"
        sql += " ORDER BY persona, code, cond_id LIMIT ?"
        args.append(int(limit))
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def wake_event_stats(*, days: int = 30, limit: int = 60) -> dict:
    """唤醒事件聚合：`fired` 计数 + 跳过原因分布 + 最近 N 条。

    ⚠️ 分母是**命中过的条件数**，不是"应该命中的条件数" —— 条件没被触发时
       不会有事件行，所以这里算不出"未命中率"，不要编。
    """
    init_store()
    conn = get_connection()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT persona, trade_date, code, cond_id, hit_at, trigger_px, "
            "action, fired, skip_reason FROM ai_wake_events "
            "ORDER BY trade_date DESC, hit_at DESC LIMIT ?", (int(limit),))]
        agg = [dict(r) for r in conn.execute(
            "SELECT persona, COUNT(*) AS hits, SUM(fired) AS fired, "
            "SUM(CASE WHEN fired=0 THEN 1 ELSE 0 END) AS skipped "
            "FROM ai_wake_events GROUP BY persona ORDER BY hits DESC")]
        why = [dict(r) for r in conn.execute(
            "SELECT skip_reason, COUNT(*) AS n FROM ai_wake_events "
            "WHERE fired=0 AND skip_reason IS NOT NULL AND skip_reason<>'' "
            "GROUP BY skip_reason ORDER BY n DESC LIMIT 10")]
        total = conn.execute("SELECT COUNT(*) FROM ai_wake_events").fetchone()[0]
        last = conn.execute(
            "SELECT MAX(trade_date) FROM ai_wake_events").fetchone()[0]
        return {"total": total, "last_date": last, "by_persona": agg,
                "skip_reasons": why, "events": rows}
    finally:
        conn.close()


# ---------------- ⑤ 决策记录（幂等锚点 + 归因链） ----------------

_DECISION_COLS = ("persona", "signal_date", "code", "advice_status", "direction",
                  "position_size", "entry", "target", "stop", "model",
                  "analysis_id", "order_id", "order_status", "order_reason",
                  "created_at")


def save_ai_decision(d: dict) -> bool:
    """保存/更新一条决策（幂等）。主键 (persona, signal_date, code)。

    ⚠️ UPSERT 里 order_id / order_status 用 COALESCE 保留**已回填**的成交信息：
    决策重跑（比如当天补跑分析后再出决策）不能把已成交的订单抹掉。
    """
    init_store()
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO ai_decisions (persona, signal_date, code, advice_status,
                    direction, position_size, entry, target, stop, model,
                    analysis_id, order_id, order_status, order_reason, created_at)
               VALUES (:persona, :signal_date, :code, :advice_status, :direction,
                    :position_size, :entry, :target, :stop, :model,
                    :analysis_id, :order_id, :order_status, :order_reason, :created_at)
               ON CONFLICT(persona, signal_date, code) DO UPDATE SET
                    advice_status=excluded.advice_status,
                    direction=excluded.direction,
                    position_size=excluded.position_size,
                    entry=excluded.entry, target=excluded.target, stop=excluded.stop,
                    model=excluded.model, analysis_id=excluded.analysis_id,
                    order_reason=excluded.order_reason,
                    order_id=COALESCE(ai_decisions.order_id, excluded.order_id),
                    order_status=COALESCE(ai_decisions.order_status, excluded.order_status)""",
            {"persona": d.get("persona"), "signal_date": d.get("signal_date"),
             "code": d.get("code"), "advice_status": d.get("advice_status"),
             "direction": d.get("direction"),
             "position_size": d.get("position_size"),
             "entry": d.get("entry"), "target": d.get("target"),
             "stop": d.get("stop"), "model": d.get("model"),
             "analysis_id": d.get("analysis_id"), "order_id": d.get("order_id"),
             "order_status": d.get("order_status"),
             "order_reason": d.get("order_reason"),
             "created_at": d.get("created_at") or _ai_now()})
        conn.commit()
        return True
    finally:
        conn.close()


def list_ai_decisions(*, persona: str = None, signal_date: str = None) -> list[dict]:
    init_store()
    conn = get_connection()
    try:
        sql = (f"SELECT {', '.join(_DECISION_COLS)} FROM ai_decisions WHERE 1=1")
        args: list = []
        if persona:
            sql += " AND persona=?"; args.append(persona)
        if signal_date:
            sql += " AND signal_date=?"; args.append(signal_date)
        sql += " ORDER BY signal_date DESC, persona, code"
        return [dict(zip(_DECISION_COLS, r)) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def list_ai_decision_dates(limit: int = 60) -> list[str]:
    """已有决策的交易日（**降序**，最新在前）—— 交易台日期选择器的数据源。

    为什么单独给一个函数而不是让前端拿全量决策自己 distinct：决策表会一直长，
    页面只需要"有哪些日"，不该为下拉框把 6 个月的行全搬进内存。
    """
    init_store()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT signal_date FROM ai_decisions "
            "WHERE signal_date IS NOT NULL ORDER BY signal_date DESC LIMIT ?",
            (int(limit or 60),)).fetchall()
        return [r["signal_date"] for r in rows]
    finally:
        conn.close()


# ---------------- ⑥ AI 自述复盘 ----------------

_REVIEW_COLS = ("persona", "trade_date", "review_text", "summary", "model", "created_at")


def save_ai_review(persona: str, trade_date: str, *, review_text: str = "",
                   summary: str = "", model: str = None) -> bool:
    """保存/更新某 AI 在 trade_date 的自述复盘（幂等，主键 persona+trade_date）。"""
    init_store()
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO ai_reviews (persona, trade_date, review_text, summary, model, created_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(persona, trade_date) DO UPDATE SET
                    review_text=excluded.review_text, summary=excluded.summary,
                    model=excluded.model, created_at=excluded.created_at""",
            (persona, trade_date, review_text or "", summary or "", model, _ai_now()))
        conn.commit()
        return True
    finally:
        conn.close()


def get_ai_review(persona: str, trade_date: str) -> dict | None:
    init_store()
    conn = get_connection()
    try:
        row = conn.execute(
            f"SELECT {', '.join(_REVIEW_COLS)} FROM ai_reviews WHERE persona=? AND trade_date=?",
            (persona, trade_date)).fetchone()
        return dict(zip(_REVIEW_COLS, row)) if row else None
    finally:
        conn.close()


def list_ai_reviews(*, persona: str = None, trade_date: str = None) -> list[dict]:
    init_store()
    conn = get_connection()
    try:
        sql = f"SELECT {', '.join(_REVIEW_COLS)} FROM ai_reviews WHERE 1=1"
        args: list = []
        if persona:
            sql += " AND persona=?"; args.append(persona)
        if trade_date:
            sql += " AND trade_date=?"; args.append(trade_date)
        sql += " ORDER BY trade_date DESC, persona"
        return [dict(zip(_REVIEW_COLS, r)) for r in conn.execute(sql, args)]
    finally:
        conn.close()


# ================================================================
# 情绪面选股：新闻 → 个股命中（P1-5，2026-10-23）
# ================================================================

def upsert_news_stock_hits(rows: list[dict]) -> int:
    """累加写入（同一 (交易日, 个股) 多轮快讯命中要**相加**，不是覆盖）。

    为什么不覆盖：快讯每小时采一轮，若后一轮覆盖前一轮，
    "全天被提及 5 次"会被压成"最后一小时 1 次" —— 热度直接失真。
    ⚠️ 计数用 SQL 侧 `hits = hits + ?`，读-改-写会在并发下丢计数。
    """
    if not rows:
        return 0
    init_store()
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = 0
    try:
        for r in rows:
            conn.execute(
                """INSERT INTO news_stock_hits
                   (trade_date, stock_code, stock_name, hits, positive, negative,
                    first_at, last_at, sources, samples, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(trade_date, stock_code) DO UPDATE SET
                     hits = news_stock_hits.hits + excluded.hits,
                     positive = news_stock_hits.positive + excluded.positive,
                     negative = news_stock_hits.negative + excluded.negative,
                     last_at = excluded.last_at,
                     sources = excluded.sources,
                     samples = excluded.samples,
                     stock_name = COALESCE(excluded.stock_name, news_stock_hits.stock_name),
                     updated_at = excluded.updated_at""",
                (r["trade_date"], r["stock_code"], r.get("stock_name"),
                 int(r.get("hits") or 0), int(r.get("positive") or 0),
                 int(r.get("negative") or 0), r.get("first_at"), r.get("last_at"),
                 r.get("sources"), r.get("samples"), now))
            n += 1
        conn.commit()
        return n
    finally:
        conn.close()


def load_news_stock_hits(trade_date: str, limit: int = 100,
                         min_hits: int = 1) -> list[dict]:
    """当日命中最多的个股（按 hits 降序，含正负计数与样例标题）。"""
    init_store()
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT * FROM news_stock_hits
               WHERE trade_date = ? AND hits >= ?
               ORDER BY hits DESC, positive DESC, stock_code
               LIMIT ?""", (trade_date, int(min_hits), int(limit))).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["samples"] = json.loads(d.get("samples") or "[]")
            except (ValueError, json.JSONDecodeError):
                d["samples"] = []
            out.append(d)
        return out
    finally:
        conn.close()


# ================================================================
# 市场情绪参照快照（2026-10-23）
# ================================================================

def save_sentiment_snapshot(snapshot: dict) -> bool:
    """写入/覆盖某来源某日的情绪指标卡。

    ⚠️ 用 `trade_date = as_of[:10]`（**采集日**）而不是"发言日期"：
       发言可能跨好几天（股吧一页混着旧帖），按发言日期归档会让同一份快照
       散到多天，基准就永远凑不齐。

    ⚠️ 覆盖而不是累加：快照是"当日那一刻的状态"，同一天跑两次应被后一次取代
       （与 `push_log` 的幂等不同 —— 那里是事件流，这里是状态）。
    """
    if not isinstance(snapshot, dict):
        return False
    as_of = str(snapshot.get("as_of") or "")
    trade_date = as_of[:10] if len(as_of) >= 10 else datetime.now().strftime("%Y-%m-%d")
    source = str(snapshot.get("source") or "").strip()
    if not source:
        return False
    init_store()
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn.execute(
            """INSERT INTO sentiment_snapshots
               (trade_date, source, as_of, labeler, window_start, window_end, n_total,
                n_scored, bull, bear, neutral, bull_ratio, dispersion, quality,
                payload, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(trade_date, source) DO UPDATE SET
                 as_of = excluded.as_of, labeler = excluded.labeler,
                 window_start = excluded.window_start,
                 window_end = excluded.window_end, n_total = excluded.n_total,
                 n_scored = excluded.n_scored, bull = excluded.bull,
                 bear = excluded.bear, neutral = excluded.neutral,
                 bull_ratio = excluded.bull_ratio, dispersion = excluded.dispersion,
                 quality = excluded.quality, payload = excluded.payload,
                 updated_at = excluded.updated_at""",
            (trade_date, source, as_of, str(snapshot.get("labeler") or "rule"),
             snapshot.get("window_start"),
             snapshot.get("window_end"), int(snapshot.get("n_total") or 0),
             int(snapshot.get("n_scored") or 0), int(snapshot.get("bull") or 0),
             int(snapshot.get("bear") or 0), int(snapshot.get("neutral") or 0),
             snapshot.get("bull_ratio"), snapshot.get("dispersion"),
             snapshot.get("quality"),
             json.dumps(snapshot, ensure_ascii=False, default=str), now))
        conn.commit()
        return True
    finally:
        conn.close()


def get_sentiment_day(trade_date: str) -> list[dict]:
    """取某**采集日**全部来源的指标卡（含 quality=empty 的行）。

    ⚠️ 这里**不过滤 empty**：消费方（复盘/页面）需要知道"这个源今天跑了但没取到"，
       与"这个源今天根本没跑"是两件事。过滤掉的只有 NULL 比例的聚合场景。
    """
    init_store()
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT payload, source, quality FROM sentiment_snapshots
               WHERE trade_date = ? ORDER BY source""", (trade_date,)).fetchall()
        out = []
        for r in rows:
            try:
                d = json.loads(r["payload"] or "{}")
            except (ValueError, json.JSONDecodeError) as e:
                _logger.warning("[情绪快照] %s/%s payload 解析失败: %s",
                                trade_date, r["source"], e)
                continue
            if isinstance(d, dict):
                out.append(d)
        return out
    finally:
        conn.close()


def get_sentiment_history(source: str, days: int = 20,
                          before_date: str = None,
                          labeler: str = None) -> list[dict]:
    """取某来源最近 `days` 个采集日的历史指标卡（**不含 `before_date` 当天**）。

    ⚠️ 必须剔除当天：把当日自己算进"近 20 日均值"，会把偏离度稀释成 1/20，
       当日再怎么异常都看不出来（自比较陷阱）。
    ⚠️ 只取 `quality != 'empty'` 的行：空样本日的 `bull_ratio` 是 NULL，
       留着会让"可用天数"虚高，分位的分母就错了。
    ⚠️⚠️ `labeler` **给了就只取同一口径**：词表标注与模型标注算出的
       `bull_ratio` 不是同一个量，混着平均会得到一个既不是 A 也不是 B 的数列。
       历史里口径不一致的天数会体现在 `available_days` 上（变少），
       由 `compare_baseline` 报"历史不足"，**不会**被悄悄补齐。
    """
    init_store()
    conn = get_connection()
    try:
        sql = ("SELECT payload, trade_date, bull_ratio, quality FROM sentiment_snapshots "
               "WHERE source = ? AND quality != 'empty' AND bull_ratio IS NOT NULL")
        args: list = [source]
        if before_date:
            sql += " AND trade_date < ?"
            args.append(before_date)
        if labeler:
            # 旧行（labeler 为 NULL）在词表时代产生 ⇒ 归到 'rule'，
            # 否则升级后历史会突然全部"口径不符"而凭空清空。
            sql += " AND COALESCE(labeler, 'rule') = ?"
            args.append(str(labeler))
        sql += " ORDER BY trade_date DESC LIMIT ?"
        args.append(int(days))
        out = []
        for r in conn.execute(sql, args).fetchall():
            try:
                d = json.loads(r["payload"] or "{}")
            except (ValueError, json.JSONDecodeError) as e:
                # 坏行不能让整个基准计算失败，但必须留痕（否则"历史凭空少一天"查不出）
                _logger.warning("[情绪快照] payload 解析失败 %s: %s", r["trade_date"], e)
                continue
            if not isinstance(d, dict) or d.get("bull_ratio") is None:
                continue
            out.append(d)
        return out
    finally:
        conn.close()


def save_sentiment_labels(rows: list) -> int:
    """写入逐条标签（**幂等覆盖**：同 `(日期,来源,条文键,标注器)` 只留最新）。

    Args:
        rows: `[{"trade_date","source","post_key","labeler","stance",
                 "quote","quote_ok","why","text"}]`

    Returns:
        实际写入条数（非法行被跳过，不计入）。

    ⚠️ `stance` 允许为 `None`（模型没标到）——**不能**落成 'unscored'：
       "没标到"和"标为中性的"是两件事，混了就算不出真实覆盖率。
    """
    if not rows:
        return 0
    init_store()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    good = []
    for d in rows:
        if not isinstance(d, dict):
            continue
        if not (d.get("trade_date") and d.get("source")
                and d.get("post_key") and d.get("labeler")):
            continue
        good.append((
            str(d["trade_date"]), str(d["source"]), str(d["post_key"]),
            str(d["labeler"]), d.get("stance"), str(d.get("quote") or "")[:120],
            None if d.get("quote_ok") is None else int(bool(d.get("quote_ok"))),
            str(d.get("why") or "")[:80], str(d.get("text") or "")[:300], now))
    if not good:
        return 0
    conn = get_connection()
    try:
        conn.executemany(
            """INSERT INTO sentiment_labels
               (trade_date, source, post_key, labeler, stance, quote, quote_ok,
                why, text, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(trade_date, source, post_key, labeler) DO UPDATE SET
                 stance = excluded.stance, quote = excluded.quote,
                 quote_ok = excluded.quote_ok, why = excluded.why,
                 text = excluded.text""", good)
        conn.commit()
        return len(good)
    finally:
        conn.close()


def get_sentiment_labels(trade_date: str, source: str = None,
                         labeler: str = None, limit: int = 500) -> list[dict]:
    """取某日逐条标签（按 `post_key, labeler` 排序，便于两标注器逐位比对）。"""
    init_store()
    conn = get_connection()
    try:
        sql = "SELECT * FROM sentiment_labels WHERE trade_date = ?"
        args: list = [trade_date]
        if source:
            sql += " AND source = ?"
            args.append(source)
        if labeler:
            sql += " AND labeler = ?"
            args.append(labeler)
        sql += " ORDER BY post_key, labeler LIMIT ?"
        args.append(int(limit))
        return [dict(r) for r in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()


# ================================================================
# 应用级一次性标记
# ================================================================
def get_flag(key: str, default: str = "") -> str:
    """读一个一次性标记；不存在返回 `default`（**不是 None**，调用方少一层判断）。"""
    init_store()
    conn = get_connection()
    try:
        row = conn.execute("SELECT value FROM app_flags WHERE key = ?", (key,)).fetchone()
        return row[0] if row and row[0] is not None else default
    finally:
        conn.close()


def set_flag(key: str, value: str = "1") -> None:
    """写/覆盖一个标记（幂等）。"""
    init_store()
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO app_flags(key, value, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (key, str(value), _wl_ts()))
    finally:
        conn.close()


# ================================================================
# 自选股（双轨：user / ashen，2026-10-27）
# ================================================================
# 为什么数据层在这里而策略在 `data/watchlist.py`：与其它表一致
# （store 只做 SQL，不做业务判断），且 `data/watchlist.py` 需要保留
# 旧函数签名给 `core/agent_tools.py`、`core/review_agent.py` 用。

WATCHLIST_OWNERS = ("user", "ashen")


def _wl_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def list_watchlist(owner: str = "user") -> list[dict]:
    """某 owner 的自选清单，按加入时间升序（与 JSON 时代 append 的顺序一致）。

    返回 `[{code,name,note,source,added_at,updated_at}]`；空清单返回 `[]`
    （**不是 None**：调用方要能区分"读失败"和"本来就是空"）。
    """
    init_store()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT code, name, note, source, added_at, updated_at "
            "FROM watchlist_items WHERE owner = ? ORDER BY added_at, rowid",
            (owner,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def count_watchlist(owner: str) -> int:
    init_store()
    conn = get_connection()
    try:
        return int(conn.execute("SELECT COUNT(*) FROM watchlist_items WHERE owner = ?",
                                (owner,)).fetchone()[0])
    finally:
        conn.close()


def replace_watchlist(owner: str, stocks: list[dict]) -> int:
    """**全量替换**某 owner 的清单（事务内 DELETE + INSERT），返回最终条数。

    ⚠️ 这个"全量替换"语义必须保留：`core/agent_write_guard` 的
    `watchlist_save`（AI 助手的 add/remove/update 及其**回滚**）传的就是
    "变更后的完整列表"，靠替换落地。改成增量语义会让回滚变成空操作。
    清空再插入必须同事务——中途失败会留下半份清单，而空清单在下游
    等于"今天没有自选"，静默且难查。
    """
    init_store()
    rows = []
    for s in stocks or []:
        code = str((s or {}).get("code") or "").strip()
        if not (code.isdigit() and len(code) == 6):
            continue                                  # 脏数据不入库，但不报错中断整批
        rows.append((
            owner, code,
            str((s or {}).get("name") or ""),
            str((s or {}).get("note") or ""),
            str((s or {}).get("source") or "manual"),
            str((s or {}).get("added_at") or _wl_ts()),
            _wl_ts(),
        ))
    conn = get_connection()
    try:
        with conn:                                    # 上下文管理器 = 单事务
            conn.execute("DELETE FROM watchlist_items WHERE owner = ?", (owner,))
            if rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO watchlist_items"
                    "(owner, code, name, note, source, added_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?)", rows)
        return len(rows)
    finally:
        conn.close()


def upsert_watchlist_item(owner: str, code: str, name: str = "", note: str = "",
                          source: str = "manual") -> bool:
    """单只加入（已存在则更新名称/备注/来源，**不重置 added_at**）。

    返回 False ⇒ code 不合法（6 位数字），**没写库**。校验放在这里而不是只放
    调用方：`watchlist_items` 的 code 会被下游当股票代码去取行情，脏值进去
    只会在别处变成"无报价"，查起来绕一大圈。
    """
    code = str(code or "").strip()
    if not (code.isdigit() and len(code) == 6):
        return False
    init_store()
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO watchlist_items(owner, code, name, note, source, added_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(owner, code) DO UPDATE SET "
                "  name = CASE WHEN excluded.name <> '' THEN excluded.name ELSE watchlist_items.name END,"
                "  note = CASE WHEN excluded.note <> '' THEN excluded.note ELSE watchlist_items.note END,"
                "  source = excluded.source,"
                "  updated_at = excluded.updated_at",
                (owner, code, name or "", note or "", source or "manual", _wl_ts(), _wl_ts()))
        return True
    finally:
        conn.close()


def delete_watchlist_item(owner: str, code: str) -> bool:
    """移除单只；返回是否真的删掉了（False ⇒ 本来就没有）。"""
    init_store()
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute("DELETE FROM watchlist_items WHERE owner = ? AND code = ?",
                               (owner, code))
        return cur.rowcount > 0
    finally:
        conn.close()


def update_watchlist_item(owner: str, code: str, name: str = None,
                          note: str = None) -> bool:
    """改名称/备注。`None` = 不改该字段；`note=""` = 清空备注。返回是否命中。"""
    init_store()
    sets, args = [], []
    if name is not None:
        sets.append("name = ?")
        args.append(str(name))
    if note is not None:
        sets.append("note = ?")
        args.append(str(note))
    if not sets:
        return False
    sets.append("updated_at = ?")
    args.extend([_wl_ts(), owner, code])
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                f"UPDATE watchlist_items SET {', '.join(sets)} WHERE owner = ? AND code = ?",
                args)
        return cur.rowcount > 0
    finally:
        conn.close()


# ================================================================
# 阿深定时值班轮次（P1，2026-10-27）
# ================================================================
# 这是"盘中每分钟扫描"与"每 40 分钟才跑一轮"之间的那道闸的**唯一依据**。
# 判"跑过没有"必须查库：内存标记在重启后清零，会让同一个 slot 重跑。

ROUTINE_STATUSES = ("ok", "skipped", "error")


def routine_exists(persona: str, trade_date: str, slot: str) -> bool:
    """该 persona 在 (交易日, slot) 是否已经跑过（ok / skipped / error **都算跑过**）。

    ⚠️ `error` 也算跑过，是刻意的：否则一个持续性错误会在每个 10 秒的扫描里
    重试一遍，把日志和额度烧光。要重试请走**手动补跑**（`force=True`），
    那是有人在看着的时候才会发生的事。
    """
    init_store()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM ai_routine_runs WHERE persona = ? AND trade_date = ? "
            "AND slot = ?", (persona, trade_date, slot)).fetchone()
        return row is not None
    finally:
        conn.close()


def save_routine_run(persona: str, trade_date: str, slot: str, *,
                     status: str, llm_called: bool = False,
                     skip_reason: str = "", summary: str = "",
                     actions: list | None = None, blocked: list | None = None,
                     autonomy: str = "", data_through: str = "",
                     model: str = "", enforced: str = "") -> bool:
    """落一轮的结果（按主键 UPSERT，重复调用**覆盖**而不是插两行）。

    覆盖而非报错，是为了让"手动补跑"能修正同一格：`force=True` 重跑后
    这一格应该反映最新一次的真实结果，而不是留着两条让人猜哪条算数。

    `enforced` 记录动作闸门的处置（见建表注释）：空串 = 本轮它自己就给了
    实质动作；非空 = 经过补正或兜底。复盘靠这一列区分"它想到的"和"系统推的"。
    """
    init_store()
    st = str(status or "").strip().lower()
    if st not in ROUTINE_STATUSES:
        st = "error"
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO ai_routine_runs"
                "(persona, trade_date, slot, status, llm_called, skip_reason, summary,"
                " actions_json, blocked_json, autonomy, data_through, model, enforced,"
                " created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(persona, trade_date, slot) DO UPDATE SET "
                " status=excluded.status, llm_called=excluded.llm_called,"
                " skip_reason=excluded.skip_reason, summary=excluded.summary,"
                " actions_json=excluded.actions_json, blocked_json=excluded.blocked_json,"
                " autonomy=excluded.autonomy, data_through=excluded.data_through,"
                " model=excluded.model, enforced=excluded.enforced,"
                " created_at=excluded.created_at",
                (persona, trade_date, slot, st, 1 if llm_called else 0,
                 str(skip_reason or ""), str(summary or ""),
                 json.dumps(actions, ensure_ascii=False) if actions is not None else None,
                 json.dumps(blocked, ensure_ascii=False) if blocked is not None else None,
                 str(autonomy or ""), str(data_through or ""), str(model or ""),
                 str(enforced or ""),
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        return True
    finally:
        conn.close()


def list_routine_runs(trade_date: str = None, persona: str = None,
                      limit: int = 50) -> list[dict]:
    """读值班轮次（默认当天、按 slot 升序）。`actions_json` 解析成 list 再返回。

    `actions_json` 解析失败时给 `[]` 并保留原文到 `actions_raw` ——
    不能让一行坏 JSON 把整个页面的历史读挂（页面只是展示，不该抛异常）。
    """
    init_store()
    where, args = [], []
    if trade_date:
        where.append("trade_date = ?")
        args.append(trade_date)
    if persona:
        where.append("persona = ?")
        args.append(persona)
    sql = ("SELECT persona, trade_date, slot, status, llm_called, skip_reason, summary,"
           " actions_json, blocked_json, autonomy, data_through, model, enforced,"
           " created_at "
           "FROM ai_routine_runs")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY trade_date DESC, slot DESC LIMIT ?"
    args.append(int(limit))
    conn = get_connection()
    try:
        rows = [dict(r) for r in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()
    for r in rows:
        for key in ("actions_json", "blocked_json"):
            raw = r.pop(key, None)
            out_key = key.replace("_json", "")
            try:
                r[out_key] = json.loads(raw) if raw else []
            except (TypeError, ValueError) as e:
                # 坏 JSON 不能把整页历史读挂，但也不能静默：
                # 一旦静默，页面上"这一轮没有动作"与"这行数据坏了"长得一模一样。
                _logger.warning("[值班记录] %s 的 %s 不是合法 JSON: %s", r.get("slot"), key, e)
                r[out_key] = []
                r[out_key + "_raw"] = raw
    return rows
