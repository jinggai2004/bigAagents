# bigAagents · A股多 Agent 智能研究系统

**简体中文** | [English](#english--english)

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-blue">
  <img alt="Tests" src="https://img.shields.io/badge/tests-3500%2B-green">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20Docker-lightgrey">
</p>

> **定位：研究与复盘系统，不是交易系统。**
> 它回答的问题不是"明天买什么"，而是——**「AI 的判断到底准不准？错在哪？下次怎么改？」**
> 不连接券商、不下真实订单，这是设计决定，不是缺陷。

---

## 目录

- [这是什么](#这是什么)
- [核心机制详解](#核心机制详解)
  - [1. 预测跟踪闭环](#1-预测跟踪闭环)
  - [2. A 股制度工程化](#2-a-股制度工程化)
  - [3. 防自欺机制](#3-防自欺机制)
  - [4. 归因体系](#4-归因体系)
- [功能页面总览](#功能页面总览web-界面-10-页--48-端点)
- [每日自动化](#每日自动化)
- [快速开始](#快速开始)
- [配置参考](#配置参考)
- [测试与质量保障](#测试与质量保障)
- [目录结构](#目录结构)
- [数据源与合规声明](#数据源与合规声明)
- [FAQ](#faq)
- [Roadmap](#roadmap)
- [免责声明](#免责声明--disclaimer)
- [English](#english--english)

---

## 这是什么

bigAagents 是一套 A 股多 Agent 智能分析系统，覆盖完整闭环：

```
数据采集(多源降级) → 全市场快照 → 情报/新闻面选股 → 多Agent深度分析 → 交易计划
      ↑                                                          ↓
改进白名单 ← 归因(LLM+机械) ← T+5/T+20 机械复盘 ← 模拟撮合(T+1/涨跌停) ← 3×3随机对照联赛
```

大多数 LLM 投资项目止步于"生成观点"：模型写得头头是道，但没有人回头验证它对不对、
错的时候是运气不好还是方法错了。本系统的核心是对每个观点做**闭环问责**：

- 每条深度分析在 **T+5、T+20 两个时点被机械复盘**——对错由代码判，不由模型自评；
- 复盘产生的"改进项"沉淀为白名单，出现 ≥3 次自动升级为系统改进建议；
- 所有"战绩"都要过**随机对照**与**多重检验校正**——赢过随机组才算数，试了 100 组
  参数后最好的那个夏普要过 DSR 检验才算数。

### 设计哲学

1. **缺就是缺**：数据不足显式标 `unavailable` / `degraded`，绝不静默补 0——
   一个假装存在的数字比一个诚实的空位危险得多；
2. **结构性守护 > 行为守护**：防幻觉靠校验闸门（引用命中率 <50% 整块作废），
   不靠"请在 prompt 里不要编造"；
3. **AI 结果绝不回灌主链**：联赛的 AI 决策只在对照层比较，永远不污染数据管道。

---

## 核心机制详解

### 1. 预测跟踪闭环

这是全系统最有价值的部分，也是与同类项目差异最大的部分。

**触发**：每日收盘后的回填任务里，每条深度分析到达两个里程碑时自动触发复盘——

| 里程碑 | 触发条件 | 产出 |
|---|---|---|
| **T+5** | 该分析第一次凑齐 5 个交易日的前瞻数据 | 早期结论（方向在 T+5 已可判） |
| **T+20** | 凑齐 20 个交易日（完整窗口） | 覆盖早期结论（`horizon=20`），不是新增一条 |

**机械层**（代码算，不经模型）：区间收益 `ret_5d/ret_20d`、最大浮盈/浮亏
（MFE/MAE）、止损距离 ÷ ATR14、相对沪深300本地指数的超额、是否先扫损后反弹……
由此产出**方向判决**（`correct / partially_correct / wrong / unclear / no_position`）
与**候选原因集**（如 `stop_too_tight` 止损过近、`direction_wrong`、`market_beta`）。

**LLM 层**（被三道闸约束）：模型归因只允许从机械层给的**候选原因集**里选主因，
判语必须引用 `facts` 里的真实数字；**引用命中率 <50% 整块作废**；
改进项必须在白名单内。LLM 失败自动降级为纯机械结论并留痕（`llm_error` 可重试），
绝不拖垮回填主流程。运行时开关：环境变量 `BIGA_ANALYSIS_REVIEW_LLM`（默认开）。

**回流**：`stop_too_tight` 这类原因码出现 ≥3 次，自动升级为页面上的「值得改的」
改进建议；交易级盈亏分解进一步把原因码**金额化**（"止损过近的 5 笔合计损失 X 元"，
见 [归因体系](#4-归因体系)）。

### 2. A 股制度工程化

通用回测框架直接套 A 股会犯制度性错误，本项目把这些制度细节做成了数据驱动的模块：

**涨跌停幅度按「板块 + 日期」历法**（`core/zt_calendar.py`），不是简单一个 10%：

| 板块 | 非 ST | ST | 制度分界日 |
|---|---|---|---|
| 北交所 | 30% | 30% | — |
| 科创板 | 20% | 20% | — |
| 创业板 | 2020-08-24 前 10%、之后 20% | 前 5%、之后**也是 20%** | 2020-08-24 |
| 主板 | 10% | **2026-07-06 前 5%、之后 10%** | 2026-07-06 |

两个分界日均由面板数据反推（前一日涨停幅度分布 vs 当日的突变），不是硬编码日期。
旧实现"先判 ST 一律 5%"曾导致某日跌停家数报 14 实为 2，全历史 2,609/2,619 日结果有差异。

**其他制度细节**：T+1 撮合（当日买不可卖）、模拟成交走涨跌停/停牌约束、
`is_a_share_code()` 结构性剔除非 A 股代码（688/300/920/8x/4x 各归其位）、
流动性下限按板块分位（一条全市场门槛会静默取消北交所的选股权）。

**数据工程纪律**：多源降级链（新浪 → 东财 → BaoStock → Tushare，按能力表自动切换）、
连接层失败立即熔断、HTTP 200 ≠ 成功（必须解析业务码）、限频探针
（如 tushare `stock_basic` 1 次/小时，撞了自动冷却 7 天）、跨源日期口径唯一、
单位口径显式换算（新浪换手是小数、BaoStock 是百分数，差 100 倍且不报错）。

### 3. 防自欺机制

**3×3 随机对照联赛**：3 个 LLM 人设各管一个模拟账户，与 3 个**随机选股对照组**
同场竞技。判据用"相对随机组的超额 + 分位"，而非绝对收益——牛市里随机组也在涨。
AI 结果只在对照层比较，绝不回灌主链。

**多重检验校正**（`core/backtest_validate.py`，零 scipy 依赖）：

- **DSR（Deflated Sharpe Ratio）**：回答"试了 N 组参数后，最好的那个夏普还显著吗"。
  Hurdle = 各次试验夏普的期望最大值（Euler–Mascheroni 常数进公式，偏度只在
  PSR 分母——这两处写反是文献复现最常见错误，本项目用数值回归测试锁死）。
  强制日频口径（年化 SR 配 T=2500 会差 √252 倍）。
- **PBO（CSCV）**：把 T×C 收益矩阵切 16 块、遍历 12,870 种训练/测试对分，
  度量"样本内最优配置在样本外落到后一半"的概率。阈值 0.3 是**有意的保守选择**
  （文献常用 0.5），由测试锁死并注明。

**Walk-forward 权重重估闸门**（`core/replay_eval.py`）：滚动"训 4 年 → 测 1 年"，
训练窗内逐因子 IC → ICIR 加权（负 ICIR 归零而非反向押注），与固定实盘权重基线
做逐折样本外对比。`FACTOR_WEIGHTS` 的任何改动必须附 OOS 对比
（模板：`docs/WEIGHT_CHANGES.md`）——"不能用 1 年样本改权重"从纪律变成机制。

> 这个系统对自己不留情面：真实面板首跑时，训练窗 2022–2025 三个可回放因子的
> ICIR 全部 ≤0，重估侧直接判"无可配置权重"；基线 2026 年样本外超额 −0.67%。
> 系统的结论是"没有任何加码这些因子的样本外证据"——并如实显示在研究页上。

### 4. 归因体系

回答"钱是哪里赚的、亏是哪里亏的"，三层由细到粗：

| 层 | 模块 | 回答的问题 | 关键设计 |
|---|---|---|---|
| **交易级** | `core/pnl_attribution.py` | 每笔钱的入场择时/持有/执行摩擦各占多少 | FIFO 配对成回合；决策价从面板收盘重建（35/35 笔 NULL 自愈）；滑点假零修复；反事实对照"若按计划价成交"；与复盘闭环咬合——原因码升级为金额化证据 |
| **行业级** | `core/attribution.py` | 超额来自选对股还是押对行业 | Brinson 三效应（配置/选股/交互）+ Carino 多期链接（恒等式 Σ=R−B 由测试锁死）；缺行业映射的票进 `__未分类__` 桶参与分解并留痕；快照时点前视偏差显式标记 |
| **风格级** | `core/style_attribution.py` | 组合暴露在哪些风格上 | 因子模仿组合 + Newey-West HAC t 值；市场因子=全市场等权（唯一来源）；规模/换手/估值因子**照实声明 unavailable**（面板无市值历史、覆盖率不足），绝不硬算 |

---

## 功能页面总览（Web 界面 10 页 / 48 端点）

| 页面 | 功能 |
|------|------|
| 🏠 工作台 | 全局驾驶舱：市场概览、任务进度条（自证活着：`elapsed_live`+`idle_seconds`）、AI 状态 |
| 🔍 深度分析 | LangGraph 多 Agent：技术/基本面/舆情分析师 → 多空辩论 → 交易计划；每条分析带**数据可用性审计**与关系审计告警；**T+5/T+20 自动复盘**时间线 |
| 🧠 选股 | 全市场快照选股（`UNIVERSE_MIN=4000` 采残拒落库）、自然语言条件筛选、东财概念成分、行业内分位与流动性闸门 |
| 📰 情报 | 新闻快讯、行业评估（LLM+证据引用）、**情报荐股**、知乎博主想法采集（可选）、实时热榜 |
| 📈 研究 | 规则回测（净值/回撤/信息比率）、因子 IC/分层/五年一致性、**🧪 过拟合体检（DSR/PBO/试验数）**、**🧬 Walk-forward 逐折表**、**💰 交易盈亏分解** |
| 🗣️ AI 交易联赛 | 3 LLM × 3 随机对照模拟交易联赛：榜单、净值、逐轮复盘、T+1 结算 |
| 💼 自选与持仓 | 自选股管理、真实持仓录入与风险巡检、模拟账户（含联赛各独立账本） |
| 📋 复盘 | 每日复盘报告与决策摘要（数字程序化生成，不经 LLM） |
| 💬 AI 助手 | 对话式投研：查行情/资金面/技术面/新闻/相似历史日/选股/回测/管理自选 |
| ⚙️ 系统 | 多厂商 LLM 接入与功能级模型分配、调度任务管理、健康检查、数据新鲜度面板 |

前端为 FastAPI + 原生 JS（无构建步骤），WebSocket 推任务进度；
端点契约与页面数字严格同源（页面与 prompt 共用同一个 core 函数，防"两套口径"）。

## 每日自动化

**15 个内置调度任务**（时刻与含义以 `core/scheduler.py::JOBS` 为唯一权威），
交易日按闸门自动跳过非交易日。关键时点举例：

```
09:25  盘前选股
09:32  联赛 T+1 开盘结算
15:05  联赛日内结算
15:30  盘后全流程（深度分析/模拟下单）
15:35  分析历史回填 + T+5/T+20 里程碑复盘（+ 自动 LLM 归因）
15:40  资金流/估值截面快照
15:45  每日复盘
15:58  联赛盘后决策轮（唯一花钱的环节，≈1.3 万 tokens/日）
```

> ⚠️ 调度器**必须用户前台启动**（双击 `启动调度器.bat`），后台会话会被回收。

## 快速开始

> 本项目**按 Windows 优先开发**（bat 脚本、Win32 单实例互斥），跨平台在 Roadmap。
> 也提供 Dockerfile（Linux 下服务层可用，调度器行为需自行验证）。

### 1. 环境与安装

```bash
git clone https://github.com/jinggai2004/bigAagents.git
cd bigAagents
pip install -r requirements.txt        # Python 3.12+
```

### 2. 配置 LLM

```bash
# 多厂商注册表（OpenAI 兼容）：复制模板后填入你的 api_key
copy config\llm_providers.example.json config\llm_providers.json
# 可选环境变量见 .env.example（DeepSeek/Qwen 内置，Kimi/GLM/豆包/OpenAI/Ollama 可加）
copy .env.example .env
```

### 3. 启动 Web 界面

```bash
python scripts/serve_web.py            # 默认 127.0.0.1:8765
python scripts/serve_web.py --port 9000 --reload
```

### 4. 重建数据（可选但推荐）

无数据时系统以 `degraded` 状态运行。全市场快照（约 2 分钟）：

```bash
python -m data.full_market --refresh-basic --snapshot
```

日线面板/换手率/申万行业/回测扫描产物等完整重建命令与耗时，
见 **[docs/DATA_SETUP.md](docs/DATA_SETUP.md)**。

## 配置参考

### `.env` 关键变量

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` / `QWEN_API_KEY` | 内置两厂商密钥（OpenAI 兼容端点均可接） |
| `DEFAULT_MODEL_NAME` | 深度分析/AI 助手的默认模型 |
| `TUSHARE_TOKEN` | 可选备用源（个股资金流降级链） |
| `DINGTALK_WEBHOOK` / `FEISHU_WEBHOOK` | 可选推送通知 |
| `BIGA_ZHIHU_COOKIE` | 可选，知乎博主想法采集（匿名多数可用，遇反爬需配） |

### `config/llm_providers.json`（多厂商注册表）

任何 **OpenAI 兼容端点**都能接入（Kimi / 智谱 / 豆包 / OpenAI / 本地 Ollama…）。
要点：`api_key_env` 指向 .env 变量名（**清单里不存明文密钥**）；
路由优先级 = `models` 精确命中 → `match` 通配 → `default_provider` 兜底；
该文件已被 .gitignore 排除。

### 功能级模型分配（可选）

"哪个功能用哪个模型"外置在 `config/llm_feature_models.json`，
界面在「系统配置 → 🎚️ 功能模型分配」，**改完即时生效**（按 mtime 热加载）。
例如让辩论环节用强模型、摘要环节用快模型，以控制成本（盘后深度分析单只 ≈ ¥0.17）。

## 测试与质量保障

```bash
运行测试.bat                            # 全量（Windows，约 6 分钟，junit xml 报告）
python -m pytest tests/test_backtest_validate.py -q    # 单文件
git config core.hooksPath .githooks    # 启用 pre-push 快速闸门
```

- **3,500+ 用例**；conftest autouse 夹具保证**全套件零 LLM 调用、零真实网络**
  （断网模拟用 `socket.connect` 补丁，判据 = `blocked()==0 且 socket_attempts()==0`）
- 端点数量由 **`TestDocEndpointCount` 硬闸门**看守：前端文档（48 路径/53 操作）与
  实际路由必须同步，改端点不改文档直接红
- 数值公式类模块（DSR/PBO/Carino/逆正态）全部带**数值锚点回归测试**——公式写反
  （如 γ 写成偏度）会在已知锚点值上精确失败
- CI（GitHub Actions）：语法全检（compileall）+ 纯逻辑测试子集（无需本地数据）

## 目录结构

```
core/     # 领域层（全部业务逻辑在此，Web 层只做薄包装）
  ├─ agents*.py          # LangGraph 多Agent：分析师/辩论/交易计划
  ├─ analysis_review.py  # 复盘：机械归因 + LLM 白名单校验（enrich_with_llm 唯一实现）
  ├─ analysis_history.py # T+N 回填 + 里程碑触发
  ├─ backtest.py         # 撮合引擎（T+1/涨跌停/费用）
  ├─ backtest_validate.py# DSR / PBO / Acklam 逆正态（零 scipy）
  ├─ replay_eval.py      # 因子回放 + walk-forward 重估
  ├─ attribution.py      # Brinson + Carino
  ├─ pnl_attribution.py  # 交易级盈亏分解（FIFO 回合）
  ├─ style_attribution.py# 风格回归（NW-HAC）
  ├─ zt_calendar.py      # 涨跌停「板块+日期」历法
  ├─ market_factor.py    # 全市场等权因子（唯一来源）
  └─ scheduler.py        # 15 个调度任务（JOBS 唯一权威）
data/     # 数据层：多源(新浪/东财/BaoStock/Tushare/akshare) + 降级链 + 全市场面板
web/      # FastAPI 服务 + 原生 JS 前端（10 页模板 / 48 端点 / 1 WebSocket）
db/       # SQLite 持久化（timeout=15，无 WAL 是刻意选择）
llm/      # 多厂商客户端与路由（含成本护栏）
scripts/  # 数据回填 / 回测扫描 / walk-forward / 打包 / 看门狗
companion/ # 桌面伴生 UI 资产（three.js 房间/角色模型）
docs/     # 权重变更闸门模板 / 数据重建指南
output/   # 前端开发文档（端点契约，测试闸门消费）
ui/       # 早期 Streamlit 页面（已被 web/ 取代，保留供参考）
```

## 数据源与合规声明

- 数据来自公开接口：新浪财经、东方财富、BaoStock、Tushare（社区版）、akshare 聚合
- **仅限个人研究学习用途**，请遵守各数据源服务条款；无实时行情分发、无商业转售
- 部分源有限频（如 tushare `stock_basic` 1 次/小时），代码内置熔断、退避与能力表探针
- 请自行申请并保管 API Key；任何密钥文件均已在 `.gitignore` 排除，注册表只存变量名引用

## FAQ

**Q: 为什么不做实盘交易？**
A: 设计定位就是研究系统。ai-hedge-fund 的 slogan 是 "the LLM never touches the
trade"——LLM 的判断要先进过验证闭环，才有资格谈执行；本系统把全部精力放在
"验证闭环"这一层。

**Q: 和 qlib / TradingAgents 等项目什么关系？**
A: 不重叠。qlib 是量化研究平台（因子/模型/回测），TradingAgents 是 LLM 分析
pipeline；本系统补的是它们都没有的一层——**AI 判断的长期问责**：判对判错的
机械度量、错误原因的受约束归因、以及把结论回流成改进项的制度。

**Q: 每天要花多少钱？**
A: 默认配置下唯一花钱的环节是联赛盘后决策轮（≈1.3 万 tokens/日）；
盘后深度分析按需触发（单只 ≈¥0.17）；T+5/T+20 复盘的 LLM 归因是轻量调用
（~2-3K tokens/次，每天 10-20 次）。全部可用 `BIGA_ANALYSIS_REVIEW_LLM=0` 等开关关闭。

**Q: 为什么我的面板数据覆盖率不满？**
A: 有意为之。各数据源能力不同（BaoStock 无北交所估值、新浪换手是小数口径……），
系统按"缺就是缺"显示 `degraded` 并标注口径，而不是拿部分数据冒充完整结论。

**Q: Linux / macOS 能跑吗？**
A: 服务层可以（有 Dockerfile）；调度器与部分脚本依赖 Windows 细节
（bat、Win32 单实例互斥）。跨平台在 Roadmap 首位。

## Roadmap

- [ ] 跨平台解耦（Linux/WSL 调度器、bat → python -m 入口全备）
- [ ] 分钟级数据与日内回测
- [ ] 绩效归因页面化（Brinson/风格归因当前走 CLI 产物）
- [ ] 参数扫描调度化（sweep → 研究页"过拟合体检"全自动）
- [ ] 更多市场（港股/ETF 轮动）

## 免责声明 / Disclaimer

本项目仅供技术研究与学习，**不构成任何投资建议**。AI 生成的分析可能完全错误——
系统内复盘模块存在的意义，就是持续量化"它到底错得多不多"。股市有风险，入市需谨慎。

<a name="english--english"></a>
## English

**bigAagents** is a multi-agent A-share (China stock market) *research & post-mortem*
system. It does not place real trades by design; instead it measures **how wrong AI
judgments are, why, and what to change next**.

### The prediction-tracking loop (what makes it different)

Every deep analysis is mechanically reviewed at **T+5 and T+20**:

- **Mechanical layer** (code, not model): forward returns, MFE/MAE, stop distance in
  ATR multiples, excess return vs CSI 300, stop-then-recover flags → produces the
  direction verdict and a **candidate-cause set** (`stop_too_tight`, `direction_wrong`,
  `market_beta`, …).
- **LLM layer** (constrained): the model may only pick a primary cause *from that set*
  and must cite real numbers from `facts`; **citation hit-rate < 50% discards the whole
  block** (hallucination gate). Improvements must come from a whitelist; a cause
  appearing ≥3 times is auto-promoted to a system-improvement suggestion.

### A-share market microstructure

Limit-up/down magnitudes are modeled **per board and per date** — including the
ChiNext 2020-08-24 regime break (10%→20%, ST included) and the main-board ST
2026-07-06 break (5%→10%), both *derived from data*, not hardcoded. T+1 matching,
board-aware liquidity floors, structural non-A-share filtering, multi-source fallback
chains with circuit breakers and rate-limit probes.

### Anti-self-deception

- **3×3 randomized-control LLM league**: three LLM personas vs three random-control
  portfolios; judged by *relative excess vs the random group*.
- **DSR / PBO** (multiple-testing correction, zero scipy): the Deflated Sharpe Ratio
  (Euler–Mascheroni constant in the hurdle, skew only in the PSR denominator — pinned
  by numeric-anchor tests) and CSCV-based PBO (16 blocks, 12,870 splits).
- **Walk-forward refit gate**: 4-year train / 1-year test rolling refit with ICIR
  weights (negative ICIR → weight 0, never a directional flip); any `FACTOR_WEIGHTS`
  change must ship with an OOS comparison (`docs/WEIGHT_CHANGES.md`).

### Attribution stack

Trade-level P&L decomposition (FIFO round trips, entry/exit friction, counterfactual
"if filled at planned price"), Brinson/Carino sector attribution (identity Σ=R−B pinned
by tests), and factor style regression with Newey-West HAC t-stats — factors without
full panel coverage are declared `unavailable` instead of being faked.

### Quick start

```bash
git clone https://github.com/jinggai2004/bigAagents.git && cd bigAagents
pip install -r requirements.txt                       # Python 3.12+
copy config\llm_providers.example.json config\llm_providers.json   # fill your api_key
python scripts/serve_web.py                           # http://127.0.0.1:8765
```

Windows-first (bat scripts); a Dockerfile is provided for the service layer.
Data rebuild guide: `docs/DATA_SETUP.md`. 3,500+ tests; CI runs a pure-logic subset.

## License

[Apache-2.0](LICENSE) © 2026 jinggai2004
