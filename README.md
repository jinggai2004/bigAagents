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

## 这是什么

bigAagents 是一套 A 股多 Agent 智能分析系统，覆盖完整闭环：

```
数据采集(多源降级) → 全市场快照 → 情报/新闻面选股 → 多Agent深度分析 → 交易计划
      ↑                                                          ↓
改进白名单 ← 归因(LLM+机械) ← T+5/T+20 机械复盘 ← 模拟撮合(T+1/涨跌停) ← 3×3随机对照联赛
```

大多数 LLM 投资项目止步于"生成观点"；本系统的核心是对每个观点做**闭环问责**：
每条深度分析在 T+5、T+20 两个时点被机械复盘（方向对错代码判、候选原因集、
LLM 归因过白名单校验、金额化盈亏分解），复盘结论再回流为系统改进项。

## 核心差异化

| # | 能力 | 说明 |
|---|------|------|
| 1 | **预测跟踪闭环** | 对/错由代码判（`mechanical_verdict`），模型只从**候选原因集**里选并必须引用真实数据；引用命中率 <50% 整块作废（幻觉审计闸门） |
| 2 | **A 股制度工程化** | 涨跌停幅度按「板块+日期」历法（创业板 2020-08-24、主板 ST 2026-07-06 两个制度分界日由数据反推）；T+1 撮合；多源降级链 + 连接层熔断 + 限频探针 |
| 3 | **防自欺机制** | 3×3 随机对照联赛（同模型 vs 随机组）；DSR/PBO 多重检验校正；walk-forward 权重重估闸门——`FACTOR_WEIGHTS` 改动必须附样本外对比（`docs/WEIGHT_CHANGES.md`） |
| 4 | **归因体系** | 交易级盈亏分解（入场/出场摩擦、反事实对照、金额化）→ Brinson+Carino 多期链接 → 风格回归（Newey-West HAC）；"缺就是缺"，覆盖不足显式标 `degraded` 而非静默补 0 |

## 功能总览（Web 界面 10 页 / 48 端点）

| 页面 | 功能 |
|------|------|
| 🏠 工作台 | 全局驾驶舱：市场概览、任务进度、AI 状态 |
| 🔍 深度分析 | LangGraph 多 Agent：技术/基本面/舆情 → 多空辩论 → 交易计划 → **T+5/T+20 自动复盘** |
| 🧠 选股 | 全市场快照选股、自然语言条件筛选、概念成分 |
| 📰 情报 | 新闻快讯、行业评估、情报荐股、舆情参考 |
| 📈 研究 | 回测报告、因子 IC/分层、**过拟合体检(DSR/PBO)**、**Walk-forward 逐折表**、**交易盈亏分解** |
| 🗣️ AI 交易联赛 | 3 LLM × 3 随机对照模拟交易联赛（榜单/净值/复盘） |
| 💼 自选与持仓 | 自选股、持仓风险巡检、模拟账户（独立账本） |
| 📋 复盘 | 每日复盘报告与决策摘要（数字不经 LLM） |
| 💬 AI 助手 | 对话式投研：行情/资金面/技术面/新闻/选股/回测 |
| ⚙️ 系统 | LLM 配置与路由、调度任务、健康检查、数据新鲜度 |

**每日自动化**：15 个内置调度任务（`core/scheduler.py::JOBS` 为唯一权威），
盘前选股 → 联赛结算 → 盘后全流程 → 行业评估 → 复盘归因，全程闸门跳过非交易日。

## 快速开始

> ⚠️ 本项目**按 Windows 优先开发**（bat 脚本、Win32 单实例互斥），跨平台在 Roadmap。
> 也提供 Dockerfile（Linux 下仅服务层可用，调度器需自行验证）。

### 1. 环境与安装

```bash
git clone https://github.com/jinggai2004/bigAagents.git
cd bigAagents
pip install -r requirements.txt        # Python 3.12+
```

### 2. 配置 LLM

```bash
# 多厂商注册表（OpenAI 兼容）：填入你的 api_key
copy config\llm_providers.example.json config\llm_providers.json
# 可选环境变量见 .env.example
copy .env.example .env
```

### 3. 启动 Web 界面

```bash
python scripts/serve_web.py            # 默认 127.0.0.1:8765
python scripts/serve_web.py --port 9000 --reload
```

### 4. 重建数据（可选但推荐）

无数据时系统以 `degraded` 状态运行（缺就是缺，不会假装是 0）。
全市场面板/行业/基准等数据的重建命令与耗时见 **[docs/DATA_SETUP.md](docs/DATA_SETUP.md)**。

## 测试

```bash
运行测试.bat                            # 全量（Windows，约 6 分钟，junit xml 报告）
python -m pytest tests/test_backtest_validate.py -q    # 单文件
git config core.hooksPath .githooks    # 启用 pre-push 快速闸门
```

3,500+ 用例；CI（GitHub Actions）跑语法全检 + 纯逻辑子集（无需本地数据）。

## 目录结构

```
core/     # 领域层：agents / analysis_review(复盘) / backtest_validate(DSR·PBO)
          #   replay_eval(回放·walk-forward) / attribution(Brinson) / pnl_attribution(盈亏分解)
          #   style_attribution / zt_calendar(涨跌停历法) / scheduler(调度)
data/     # 数据层：多源(新浪/东财/BaoStock/Tushare) + 降级链 + 全市场面板
web/      # FastAPI 服务 + 原生 JS 前端（10 页模板 / 48 端点）
db/       # SQLite 持久化（timeout=15，无 WAL 是刻意选择）
llm/      # 多厂商 LLM 客户端与路由（含成本护栏）
scripts/  # 数据回填 / 回测扫描 / walk-forward / 打包
companion/ # 桌面伴生 UI 资产（three.js）
docs/     # 权重变更闸门模板等
```

## 数据源与合规声明

- 数据来自公开接口：新浪财经、东方财富、BaoStock、Tushare（社区版）、akshare 聚合
- **仅限个人研究学习用途**，请遵守各数据源服务条款；无实时行情分发、无商业转售
- 部分源有限频（如 tushare `stock_basic` 1 次/小时），代码内置熔断、退避与能力表探针
- 请自行申请并保管 API Key；任何密钥文件均已在 `.gitignore` 排除

## 免责声明 / Disclaimer

本项目仅供技术研究与学习，**不构成任何投资建议**。AI 生成的分析可能完全错误——
系统内复盘模块存在的意义，就是持续量化"它到底错得多不多"。股市有风险，入市需谨慎。

<a name="english--english"></a>
## English

**bigAagents** is a multi-agent A-share (China stock market) *research & post-mortem* system.
It does not place real trades by design; instead it measures **how wrong AI judgments are,
why, and what to change**.

- **Prediction tracking loop** — every deep analysis is mechanically reviewed at T+5 and T+20
  (direction verdict by code, candidate-cause attribution validated against real data, LLM output
  discarded when citation hit-rate < 50%).
- **A-share market microstructure** — limit-up/down magnitudes modeled per board *and per date*
  (incl. ChiNext 2020-08-24 and main-board-ST 2026-07-06 regime breaks), T+1 matching, multi-source
  fallback chains with circuit breakers.
- **Anti-self-deception** — 3×3 randomized-control LLM league; Deflated Sharpe Ratio & PBO (CSCV)
  for multiple-testing correction; walk-forward refit gate for factor weights.
- **Attribution stack** — trade-level P&L decomposition, Brinson/Carino sector attribution,
  factor style regression with Newey-West HAC t-stats.

Windows-first (bat scripts); `python scripts/serve_web.py` starts the web UI.
Data rebuild guide: `docs/DATA_SETUP.md`. Apache-2.0 licensed; research use only, not investment advice.

## License

[Apache-2.0](LICENSE) © 2026 jinggai2004
