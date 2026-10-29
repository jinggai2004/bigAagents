# 数据重建指南（DATA SETUP）

本仓库**不含任何行情数据**（面板 parquet、SQLite 库、缓存均不入库）。
首次克隆后系统可以启动，但多数面板会显示 `degraded`（缺就是缺，不假装是 0）。
按需执行下面的命令逐步重建。

所有命令在项目根目录、Windows PowerShell / Git Bash 下执行；
先完成 `pip install -r requirements.txt` 与 LLM 配置（见 README）。

## 0. 自检

```bash
python check_data_sources.py     # 各数据源连通性与能力表
python scripts/check_workspace.py # 共享契约检查（改代码后跑）
```

## 1. 全市场快照（必做，约 2 分钟）

新浪 `hs_a` 一次拿全 5,500+ 只的价量 + PE/PB/市值/换手：

```bash
python -m data.full_market --refresh-basic --snapshot
```

> ⚠️ `stock_basic` 走 tushare，**1 次/小时**硬限频；失败会自动降级标 `degraded`，稍后重试即可。

## 2. 日线历史面板（可选，数小时级）

BaoStock 全历史日线 + 估值（`pe_ttm` 约 13.8 秒/只，无北交所）：

```bash
python scripts/backfill_panel_baostock.py
```

产出 `data/panel/`（约 300~600MB parquet，已被 .gitignore 排除）。

## 3. 换手率/流通股本回填（可选，约 72 分钟）

新浪日线自带 `turnover`+`outstanding_share`，覆盖全部板块（含北交所）：

```bash
python scripts/backfill_turn_share.py
```

> ⚠️ 单位口径：新浪是小数（0.0031）、BaoStock 是百分数（0.31），合并时已自动换算。

## 4. 申万一级行业（推荐，约 2 分钟）

Brinson 归因与行业分位依赖；31 个 L1 行业、约 5,200 只：

```bash
python -c "from data.industry_swa import backfill_industry_swa; backfill_industry_swa()"
```

> ⚠️ 方法论注意：今天的成分快照解释历史收益存在**前视偏差**
> （行业归属十年不变假设），归因输出会带 `industry_snapshot` 标记。

## 5. 验证与归因产物（研究页新区块的数据源）

| 页面区块 | 产物生成命令 | 产物位置 |
|---|---|---|
| 🧪 过拟合体检 | `python scripts/backtest_rule_based.py --sweep-insample` | `data/backtest_records/sweep_insample_*.json` |
| 🧬 Walk-forward | `python scripts/run_walk_forward.py --days 1500` | `output/walk_forward_*.json` |
| 💰 交易盈亏分解 | 无需预生成（端点在线轻算，读模拟账户成交） | — |
| 深度分析复盘 | `python scripts/run_analysis_review.py --limit 100 --llm` | SQLite `analysis_reviews` |

## 6. 调度器（自动日更）

```bash
启动调度器.bat     # 或 python -m core.scheduler（以 JOBS 注释为准）
```

调度器**只能用户前台双击/手动启动**（后台会话会被回收）；
判活看心跳 `started_at`+PID，不要看文件 mtime。
