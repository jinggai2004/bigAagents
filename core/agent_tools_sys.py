# -*- coding: utf-8 -*-
"""AI 助手系统级工具（2026-10-24 升级）——让助手"看得见项目自身"。

为什么需要这个模块（见 output/AI助手工具能力评估与同类Agent调研_2026-10-24.md）：
    此前 17 个工具全部面向**投研数据**（行情/财务/资金/持仓），而用户对项目的
    问题远不止这些："昨天的复盘报告说了什么""最近有什么告警""现在助手接的哪个
    模型""数据是不是最新的"——这些全都要 AI 读 output/ 报告、查 DB 运行表、看
    config 才能答，旧工具一个都够不着。

设计红线（评估报告第五章，逐条落实）：
    ① 只开放**白名单**文件（output/ 内）与**白名单**表（固定查询形状），
       绝不给任意路径 / 任意 SQL —— 路径穿越与注入都在源头堵死；
    ② 密钥只读+脱敏：.env 永不可读，config JSON 里疑似密钥的值一律掩码；
       `api_key_env` 是环境变量**名**不是密钥，保留展示；
    ③ 写操作只碰"运行时可热生效"的配置：推送模板（DB，即时生效）、
       自定义任务开关（调度器每约60秒同步）、自选股备注（下次读取即生效）；
       **内置 JOBS / llm_providers.json / settings.py 不提供写**——
       它们要重启长驻进程才生效，让 AI 改等于制造"改了没生效"的幻觉
       （QWEN_TURBO_MODEL 静默降级事故的同款土壤）；
    ④ 读到的报告/记录是**历史产物**：返回值必须带文件修改时间与提醒，
       防止模型把上月报告的结论说成"当前状态"。

工具清单（7 个，全部注册进 agent_tools.TOOL_SCHEMAS / TOOL_FUNCS）：
    tool_read_output_file   读 output/ 下的报告与 JSON（分段/尾部/json_path 取子对象）
    tool_db_inspect         白名单表只读查询（告警/信号/运行记录/覆盖率/LLM成本…）
    tool_config_view        只读+脱敏看 config/*.json（模型路由/价格表/人格…）
    tool_push_templates     飞书推送模板 list/get/save/reset（即时生效）
    tool_scheduled_jobs     内置任务只读列出 + 自定义任务开关（约60秒热同步）
    tool_panel_bars         读本地日线面板（bars.get_bars，多标的对比，零联网）
    tool_data_freshness     数据新鲜度总览（口径：manifest.date_max，不是今天）

拆分方式沿用 P2-2 的先例（agent_tools_fin）：本模块只依赖 agent_tools_base
（`_tool` 包装），由 agent_tools.py **re-export**，避免循环导入。
"""

import json
import logging
import os
import re
import sqlite3
from datetime import datetime

from core.agent_tools_base import _tool

_logger = logging.getLogger(__name__)

# 根目录常量做成**模块属性**（不是函数内字面量）：测试用 monkeypatch 替换到
# 临时目录，避免耦合真实环境（铁律：测试不得写真实库/真实 output）。
#
# ⚠️ 但"模块属性" ≠ "可以用 `__file__` 推导"：打包后 `__file__` 在只读包
#    `_internal/` 里，推算出来是空目录（2026-10-25 实测 exe 读不到数据）。
#    ⇒ 数据根与配置根**分开取**，二者在 exe 里不是同一个目录：
#      · OUTPUT_ROOT 跟**数据根**走（可写，exe 同级或项目根）
#      · CONFIG_ROOT 跟**只读包**走（config/*.json 是 `--add-data` 打进去的）
from config.paths import bundle_root as _paths_bundle_root  # noqa: E402
from config.paths import data_root as _paths_data_root  # noqa: E402

_PROJECT_ROOT = _paths_data_root()
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "output")
CONFIG_ROOT = os.path.join(_paths_bundle_root(), "config")


# ================================================================
# 通用小工具
# ================================================================

def _norm_date(s: str) -> str:
    """`20261024` → `2026-10-24`；已是带杠形式原样返回（跨源日期口径唯一：读侧也防御归一）。"""
    s = str(s or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def _safe_resolve(root: str, rel: str) -> str | None:
    """把相对路径安全解析到 root **内部**；越界（路径穿越）/绝对路径/不存在返回 None。

    必须用 realpath：`output/a/../../core/scheduler.py` 这类绕法在字符串前缀
    判断下能蒙混，realpath 展开后才能保证真落在 root 里。
    """
    rel = str(rel or "").strip().replace("\\", "/")
    if not rel or rel.startswith(("/", "~")) or ":" in rel:
        return None
    root_real = os.path.realpath(root)
    full = os.path.realpath(os.path.join(root_real, rel))
    if full != root_real and not full.startswith(root_real + os.sep):
        return None
    if not os.path.isfile(full):
        return None
    return full


def _mtime_str(path: str) -> str:
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
    except OSError as e:                               # noqa: BLE001 - 判不了就给空，别炸
        _logger.debug("[Agent工具] 修改时间读取失败 %s: %s", path, e)
        return ""


def _dig_json(obj, path: str):
    """按点分路径取 JSON 子对象：`picks.0.股票代码`。返回 (找到没有, 值)。"""
    cur = obj
    for part in [p for p in str(path).split(".") if p.strip()]:
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError) as e:
                _logger.debug("[Agent工具] json_path 下标无效 %r: %s", part, e)
                return False, None
        elif isinstance(cur, dict):
            if part not in cur:
                return False, None
            cur = cur[part]
        else:
            return False, None
    return True, cur


# ================================================================
# ① 读 output/ 报告与产物
# ================================================================

_READ_MAX_FILE_BYTES = 20 * 1024 * 1024      # 超过 20MB 的文件拒绝整读（防大 JSON 撑爆）
_READ_MAX_LINES = 400                        # 单次最多返回的行数
_READ_CHAR_CAP = 6000                        # 单次返回正文的字符上限（chat 侧配额 7000）
_HISTORICAL_NOTE = ("历史产物：内容反映其修改时间时点的状态，不代表当前最新数据；"
                    "引用时必须注明文件日期")


@_tool
def tool_read_output_file(action: str = "list", file: str = "", pattern: str = "",
                          offset: int = 0, limit: int = 150, json_path: str = "") -> dict:
    """读取 output/ 目录下的**历史报告与产物**（复盘/联赛/评估报告、daily_picks 等）。

    action=list  列文件（可传 pattern 子串过滤，按修改时间倒序，最多80个）
    action=read  按行分段读文本（.md/.txt/.csv）；.json 会解析后返回，太大时用
                 json_path 取子对象（点分路径，如 picks.0.股票代码）
    action=tail  读最后 limit 行（快速看结论段）

    ⚠️ 只能读 output/ **内部**的文件；.env / 源码 / data/ 一律不可读（安全边界）。
    """
    action = str(action or "list").strip().lower()

    if action == "list":
        pat = str(pattern or "").strip().lower()
        entries = []
        for dirpath, _dirs, files in os.walk(OUTPUT_ROOT):
            for fname in files:
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, OUTPUT_ROOT).replace("\\", "/")
                if pat and pat not in rel.lower():
                    continue
                try:
                    size_kb = round(os.path.getsize(full) / 1024, 1)
                except OSError as e:                   # noqa: BLE001
                    _logger.debug("[Agent工具] 跳过读不到大小的文件 %s: %s", full, e)
                    continue
                entries.append({"文件": rel, "大小KB": size_kb,
                                "修改时间": _mtime_str(full)})
        entries.sort(key=lambda e: e["修改时间"], reverse=True)
        total = len(entries)
        return {"文件数": total, "文件": entries[:80],
                "说明": "按修改时间倒序，最多列80个；用 pattern 缩小范围（如 '复盘'/'daily_picks'）"}

    if action in ("read", "tail"):
        full = _safe_resolve(OUTPUT_ROOT, file)
        if full is None:
            return {"error": f"文件不存在或不在 output/ 目录内: {file!r}",
                    "提示": "先用 action=list 找到确切文件名；只能读 output/ 内的相对路径"}
        if os.path.getsize(full) > _READ_MAX_FILE_BYTES:
            return {"error": f"文件超过 {_READ_MAX_FILE_BYTES // (1024*1024)}MB，拒绝整读"}

        rel = os.path.relpath(full, OUTPUT_ROOT).replace("\\", "/")
        meta = {"file": rel, "修改时间": _mtime_str(full)}

        # JSON：解析后返回（结构化，模型好引用）；太大则引导 json_path 下钻
        if full.lower().endswith(".json"):
            try:
                with open(full, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError) as e:
                _logger.warning("[Agent工具] 报告 JSON 解析失败 %s: %s", rel, e)
                return {"error": f"JSON 解析失败: {e}", **meta}
            if json_path:
                ok, sub = _dig_json(data, json_path)
                if not ok:
                    top = list(data)[:20] if isinstance(data, dict) else (
                        f"list[{len(data)}]" if isinstance(data, list) else type(data).__name__)
                    return {"error": f"json_path 取不到: {json_path!r}", "顶层键": top, **meta}
                data = sub
            text = json.dumps(data, ensure_ascii=False, default=str)
            if len(text) > _READ_CHAR_CAP:
                top = list(data)[:20] if isinstance(data, dict) else (
                    f"list[{len(data)}]" if isinstance(data, list) else type(data).__name__)
                return {"error": f"内容 {len(text)} 字符超出单次上限 {_READ_CHAR_CAP}",
                        "提示": "用 json_path 按点分路径取子对象（如 picks / picks.0 / 报告.技术面）",
                        "顶层键": top, **meta}
            return {"类型": "json", "数据": data, "说明": _HISTORICAL_NOTE, **meta}

        # 文本：按行分段
        try:
            with open(full, encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError as e:
            _logger.warning("[Agent工具] 报告读取失败 %s: %s", rel, e)
            return {"error": f"读取失败: {e}", **meta}
        total = len(lines)
        if action == "tail":
            n = max(1, min(int(limit) if limit else 60, _READ_MAX_LINES))
            seg = lines[-n:]
            rng = f"第 {max(1, total - n + 1)} ~ {total} 行（共 {total} 行）"
        else:
            off = max(0, int(offset) if offset else 0)
            n = max(1, min(int(limit) if limit else 150, _READ_MAX_LINES))
            seg = lines[off:off + n]
            rng = f"第 {off + 1} ~ {off + len(seg)} 行（共 {total} 行）"
        text = "\n".join(seg)
        out = {"行范围": rng, "内容": text, **meta}
        if len(text) > _READ_CHAR_CAP:
            out["内容"] = text[:_READ_CHAR_CAP]
            out["截断提示"] = (f"正文超长已截断到 {_READ_CHAR_CAP} 字符；"
                              "把 limit 调小或用 offset 分段读")
        if not seg:
            out["内容"] = ""
            out["说明"] = "该行范围没有内容（offset 超出文件末尾）"
        else:
            out["说明"] = _HISTORICAL_NOTE
        return out

    return {"error": f"未知 action: {action}，可用 list/read/tail"}


# ================================================================
# ② 白名单表只读查询
# ================================================================

# 每张表：desc（给模型看的说明）/ cols（只取这些列，长文本列刻意不带或靠截断防护）
#          / code_col / date_col（可选过滤列）/ order_col（倒序基准）。
# ⚠️ cols 是**白名单**而不是 SELECT *：防止未来加列（如超长 payload）悄悄撑爆
#    工具返回；要查新列必须显式来这里登记（改契约全仓搜断言）。
_DB_TABLES = {
    "alerts": {
        "desc": "盘中异动/风控告警", "order_col": "id",
        "cols": ["trade_date", "code", "name", "rule", "level", "message",
                 "price", "change_pct", "quote_time", "source"],
        "code_col": "code", "date_col": "trade_date"},
    "signal_events": {
        "desc": "买入信号事件（待分析/已分析）", "order_col": "id",
        "cols": ["day", "code", "name", "kind", "detail", "price", "change_pct",
                 "relation", "analyzed"],
        "code_col": "code", "date_col": "day"},
    "runs": {
        "desc": "流程运行记录（run_type + 摘要）", "order_col": "id",
        "cols": ["run_type", "summary", "created_at"]},
    "observations": {
        "desc": "复盘观察与验证结果", "order_col": "id",
        "cols": ["review_date", "kind", "content", "verified_date", "result"]},
    "screen_picks": {
        "desc": "选股候选历史样本（评分/T+N收益回填用）", "order_col": "id",
        "cols": ["pick_date", "stock_code", "stock_name", "source", "strategy",
                 "score", "score_coverage", "rank_in_run", "market_state", "market_score"],
        "code_col": "stock_code", "date_col": "pick_date"},
    "data_coverage": {
        "desc": "各数据集每日覆盖率", "order_col": "id",
        "cols": ["data_date", "dataset", "source", "got", "total", "ratio", "status", "note"],
        "date_col": "data_date"},
    "push_log": {
        "desc": "飞书推送记录", "order_col": "id",
        "cols": ["day", "channel", "title", "status", "detail"], "date_col": "day"},
    "news_items": {
        "desc": "本地新闻快讯库（标题/来源/情绪）", "order_col": "id",
        "cols": ["title", "source", "published_at", "related_codes", "sentiment"]},
    "daily_features": {
        "desc": "相似日特征库（每标的每日一行）", "order_col": "id",
        "cols": ["symbol", "date", "close", "future_ret_5d", "future_ret_20d"],
        "code_col": "symbol", "date_col": "date"},
    "ai_decisions": {
        "desc": "AI交易联赛每日决策（persona/方向/订单状态）", "order_col": "signal_date",
        "cols": ["persona", "signal_date", "code", "advice_status", "direction",
                 "position_size", "entry", "target", "stop", "model",
                 "order_status", "order_reason"],
        "code_col": "code", "date_col": "signal_date"},
    "market_snapshot": {
        "desc": "全市场估值截面快照（PE/PB/市值/换手，逐日积累）", "order_col": "id",
        "cols": ["snapshot_date", "stock_code", "stock_name", "industry", "price",
                 "change_pct", "pe", "pb", "market_cap", "turnover", "amount"],
        "code_col": "stock_code", "date_col": "snapshot_date"},
    "fund_flow_snapshot": {
        "desc": "全市场资金流快照（今日/3/5/10日净额）", "order_col": "id",
        "cols": ["snapshot_date", "stock_code", "stock_name", "net_today", "net_3d",
                 "net_5d", "net_10d", "amount", "change_pct", "price"],
        "code_col": "stock_code", "date_col": "snapshot_date"},
    "sentiment_snapshots": {
        "desc": "市场情绪快照（多空比/分歧度/质量）", "order_col": "trade_date",
        "cols": ["trade_date", "source", "as_of", "n_total", "n_scored",
                 "bull_ratio", "dispersion", "quality"],
        "date_col": "trade_date"},
    "analysis_reviews": {
        "desc": "深度分析的事后复核（对错判定/原因归因）", "order_col": "id",
        "cols": ["analysis_id", "stock_code", "stock_name", "analysis_date",
                 "review_date", "decision_label", "model", "verdict", "primary_cause"],
        "code_col": "stock_code"},
    "llm_calls": {
        "desc": "LLM调用日志（模型/场景/延迟/token/成本/失败原因）", "order_col": "id",
        "cols": ["created_at", "model", "purpose", "ok", "latency_ms",
                 "total_tokens", "cost_cny", "error"]},
}

_VALUE_TRUNC = 200      # 单元格超长截断（news 标题/observation 内容等防撑爆）


def _trunc_row(row) -> dict:
    out = {}
    for k in row.keys():
        v = row[k]
        if isinstance(v, str) and len(v) > _VALUE_TRUNC:
            v = v[:_VALUE_TRUNC] + "…(截断)"
        out[k] = v
    return out


@_tool
def tool_db_inspect(action: str = "tables", table: str = "", limit: int = 20,
                     code: str = "", date: str = "", only_failures: bool = False) -> dict:
    """查询项目数据库里的**运行与记录类表**（白名单，只读，固定查询形状——不是任意SQL）。

    action=tables 列出可查的表（含说明与行数）
    action=query 按表查询最近记录，可带过滤：code（按股票代码）、date（按日期，
                 支持 2026-10-24 或 20261024）、only_failures（仅 llm_calls 失败记录）

    用户问"最近有什么告警/信号有没有分析/LLM调用失败没/数据覆盖率/联赛今天决策了什么"
    这类**系统运行状态**问题时使用。
    """
    action = str(action or "tables").strip().lower()

    from db.models import get_connection
    conn = get_connection()
    try:
        if action == "tables":
            out = {}
            for t, spec in _DB_TABLES.items():
                try:
                    n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                except sqlite3.Error as e:               # noqa: BLE001 - 单表判不了不影响其它
                    _logger.debug("[Agent工具] 表 %s 计数失败: %s", t, e)
                    n = None
                filt = []
                if spec.get("code_col"):
                    filt.append("code")
                if spec.get("date_col"):
                    filt.append("date")
                out[t] = {"说明": spec["desc"], "行数": n,
                          "可过滤": "/".join(filt) or "无"}
            return {"可查表": out,
                    "用法": "action=query + 表名，如查最近告警：table=alerts, limit=20"}

        if action == "query":
            spec = _DB_TABLES.get(str(table or "").strip())
            if spec is None:
                return {"error": f"未知表: {table!r}（不在白名单）",
                        "可用表": sorted(_DB_TABLES),
                        "提示": "先 action=tables 看可查的表"}
            where, params = [], []
            code_s = str(code or "").strip()
            date_s = _norm_date(date)
            if code_s and spec.get("code_col"):
                where.append(f"{spec['code_col']} = ?")
                params.append(code_s)
            if date_s and spec.get("date_col"):
                where.append(f"{spec['date_col']} = ?")
                params.append(date_s)
            if only_failures and table == "llm_calls":
                where.append("ok = 0")
            n = min(max(int(limit) if limit else 20, 1), 50)
            sql = (f"SELECT {', '.join(spec['cols'])} FROM {table}"
                   + (" WHERE " + " AND ".join(where) if where else "")
                   + f" ORDER BY {spec['order_col']} DESC LIMIT ?")
            rows = [_trunc_row(r) for r in conn.execute(sql, (*params, n)).fetchall()]
            out = {"表": table, "说明": spec["desc"], "条数": len(rows), "记录": rows}
            applied = [f"code={code_s}"] if code_s and spec.get("code_col") else []
            if date_s and spec.get("date_col"):
                applied.append(f"date={date_s}")
            if only_failures and table == "llm_calls":
                applied.append("仅失败")
            if applied:
                out["过滤条件"] = "，".join(applied)
            elif code_s or date or only_failures:
                out["提示"] = "该表没有对应的过滤列，过滤参数被忽略"
            out["说明2"] = "按 %s 倒序取最近 %d 条" % (spec["order_col"], n)
            return out

        return {"error": f"未知 action: {action}，可用 tables/query"}
    finally:
        conn.close()


# ================================================================
# ③ config 只读+脱敏
# ================================================================

_SECRET_KEY_RE = re.compile(r"(api_key|apikey|secret|token|password|passwd)", re.I)
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")       # 形如 DEEPSEEK_API_KEY 的环境变量名
_KEY_LIKE_RE = re.compile(r"^(sk-|ak-|ghp_)")


def _mask_secrets(obj):
    """递归脱敏：键名含 key/secret/token/password ⇒ **整个子树**掩码（list/dict/标量都遮）。

    初版只遮"敏感键直接对应字符串"，`{"api_keys":["sk-…"]}` / `{"api_key":{"value":…}}`
    会漏（质量复核 P1-1）。规则：
    · 值是形如 `DEEPSEEK_API_KEY` 的环境变量名（api_key_env 这类"去哪个变量取"的名字）
      ⇒ 保留展示——模型要靠它回答"密钥在哪个环境变量"；
    · 其余一律 "***已脱敏***"，不再递归进子树。
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _SECRET_KEY_RE.search(str(k)):
                if isinstance(v, str) and _ENV_NAME_RE.match(v) and not _KEY_LIKE_RE.match(v):
                    out[k] = v
                else:
                    out[k] = "***已脱敏***"
            else:
                out[k] = _mask_secrets(v)
        return out
    if isinstance(obj, list):
        return [_mask_secrets(x) for x in obj]
    return obj


@_tool
def tool_config_view(action: str, file: str = "") -> dict:
    """查看项目配置（**只读+脱敏**）：模型路由、LLM价格表、AI人格、板块映射等 config/*.json。

    action=list 列出 config 下的 JSON 配置文件
    action=read 读取指定文件（密钥值自动掩码；.env 与 .py 源码不可读）

    用户问"助手接的哪个模型/某功能路由到谁/模型单价多少/有哪些AI人格"时使用。
    改配置请到「系统配置」页——本工具刻意**不提供写**（config 改动要重启长驻进程
    才生效，对话里改等于制造"改了没生效"的错觉）。
    """
    action = str(action or "list").strip().lower()

    if action == "list":
        entries = []
        for fname in sorted(os.listdir(CONFIG_ROOT)):
            if not fname.endswith(".json"):
                continue
            full = os.path.join(CONFIG_ROOT, fname)
            try:
                entries.append({"文件": fname,
                                "大小KB": round(os.path.getsize(full) / 1024, 1),
                                "修改时间": _mtime_str(full)})
            except OSError as e:                        # noqa: BLE001
                _logger.debug("[Agent工具] 跳过读不到大小的配置 %s: %s", full, e)
                continue
        return {"配置文件": entries,
                "说明": "只开放 *.json；.env（密钥）与 *.py（源码）不可读。"
                        "常用：llm_feature_models.json=功能→模型路由，"
                        "llm_providers.json=厂商与模型清单，llm_prices.json=单价"}

    if action == "read":
        fname = str(file or "").strip().replace("\\", "/")
        base = fname.rsplit("/", 1)[-1]
        # ⚠️ 按 **basename** 拒 .env（初版只比完整相对路径，`nested/.env` 能绕过）；
        #    且只允许 *.json（初版 `nested/secret.txt` 也能读）。
        if base.lower() == ".env" or base.lower().endswith((".py", ".pyc")):
            return {"error": f"安全边界：{base} 不可读",
                    "提示": "密钥在 .env（永不开放给对话）；源码请直接看代码"}
        if not base.lower().endswith(".json"):
            return {"error": "只允许读取 config/ 下的 *.json", "收到": fname}
        full = _safe_resolve(CONFIG_ROOT, fname)
        if full is None:
            return {"error": f"文件不存在或不在 config/ 内: {fname!r}",
                    "提示": "先 action=list 看文件名；只支持 config/ 下的 *.json"}
        try:
            with open(full, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            _logger.warning("[Agent工具] 配置读取/解析失败 %s: %s", fname, e)
            return {"error": f"读取/解析失败: {e}"}
        return {"file": fname, "修改时间": _mtime_str(full), "配置": _mask_secrets(data),
                "说明": "疑似密钥的值已脱敏为 ***；api_key_env 是环境变量名（不是密钥本身）。"
                        "注意：config 改动需重启长驻进程才生效（本工具只读）"}

    return {"error": f"未知 action: {action}，可用 list/read"}


# ================================================================
# ④ 飞书推送模板（DB 存储，改完即时生效）
# ================================================================

_TPL_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_TPL_IF_RE = re.compile(r"\{%\s*if\s+([A-Za-z_][A-Za-z0-9_]*)\s*%\}")
_TPL_BAD_TAG_RE = re.compile(r"\{%(?!\s*(?:if\b|endif\b))")


def _validate_template_body(key: str, body: str, known: set) -> dict | None:
    """校验模板正文；不合法返回 error dict，合法返回 None。

    ⚠️ 初版只查 {{变量}}，不查 {% if 变量 %}——`{% if typo_var %}关键{% endif %}`
    能过保存、渲染时整块静默消失（质量复核 P1-2）。这里与渲染语法对齐：
    条件变量也必须在 KNOWN_VARS 里；不支持的标签（else/for 等）直接拒。
    """
    unknown = sorted(set(_TPL_VAR_RE.findall(body)) - known)
    if unknown:
        return {"error": f"正文里有该模板不支持的变量: {unknown}",
                "该模板可用变量": sorted(known),
                "提示": "拼错的变量会原样出现在推送里，所以保存前直接拒绝"}
    unknown_if = sorted(set(_TPL_IF_RE.findall(body)) - known)
    if unknown_if:
        return {"error": f"条件块 {{{{% if %}}}} 使用了不支持的变量: {unknown_if}",
                "该模板可用变量": sorted(known),
                "提示": "条件变量为空时整块会被静默丢弃——拼错比报错更危险"}
    if body.count("{% if") != body.count("{% endif %}"):
        return {"error": "{% if %} 与 {% endif %} 数量不配对",
                "提示": "条件块必须成对出现，否则渲染结果不可预期"}
    if _TPL_BAD_TAG_RE.search(body):
        return {"error": "正文含不支持的模板控制符（只支持 {% if %}…{% endif %}）"}
    return None


@_tool
def tool_push_templates(action: str, key: str = "", body: str = "",
                        change_id: str = "") -> dict:
    """管理飞书推送模板（外壳与条目格式，存数据库，**commit 后即时生效**）。

    action=list/get  只读：列出 / 看单个（当前生效正文 + 内置默认 + 可用变量）
    action=save      写·预览：校验正文（变量白名单含 {% if %} 变量、配对、控制符），
                     返回 change_id + **完整**前后正文，不落任何写
    action=reset     写·预览：恢复内置默认（返回 change_id）
    action=commit    写·执行：change_id=用户确认过的 preview 凭证（15分钟有效、一次性）
    action=restore   写·回滚：change_id=历史一次 commit 的凭证，按快照撤销

    ⚠️ 纪律：save/reset 只产出 preview；必须向用户复述"改动前/改动后"完整差异、
       用户明确同意后用 commit 执行。数字不经模板改写，模板只管排版。
    """
    from core import push_templates as T
    from core import agent_write_guard as guard
    action = str(action or "list").strip().lower()
    key = str(key or "").strip()

    if action == "commit":
        return guard.commit(change_id)
    if action == "restore":
        if not change_id:
            return {"error": "restore 需要 change_id（历史 commit 的凭证）"}
        return guard.restore(change_id)

    if action == "list":
        overrides = T.user_overrides()
        items = []
        for k in T.KNOWN_VARS:
            label, varnote = T.TEMPLATE_META[k]
            body_now = T.get_template(k)
            items.append({"key": k, "说明": label,
                          "可用变量": sorted(T.KNOWN_VARS[k]),
                          "是否自定义": k in overrides,
                          "当前正文(截断)": body_now[:300]})
        return {"模板": items,
                "提示": "改模板：先 get 看原文 → action=save 拿 change_id → 用户确认后 commit"}

    if action == "get":
        if key not in T.KNOWN_VARS:
            return {"error": f"未知模板 key: {key!r}", "可用key": sorted(T.KNOWN_VARS)}
        label, _varnote = T.TEMPLATE_META[key]
        return {"key": key, "说明": label, "可用变量": sorted(T.KNOWN_VARS[key]),
                "是否自定义": key in T.user_overrides(),
                "当前生效正文": T.get_template(key),
                "内置默认正文": T.DEFAULT_TEMPLATES.get(key, "")}

    if action == "save":
        if key not in T.KNOWN_VARS:
            return {"error": f"未知模板 key: {key!r}", "可用key": sorted(T.KNOWN_VARS)}
        body_s = str(body or "")
        if not body_s.strip():
            return {"error": "body 为空（整体替换不能存空正文；想恢复默认请用 action=reset）"}
        bad = _validate_template_body(key, body_s, T.KNOWN_VARS[key])
        if bad:
            return bad
        old = T.get_template(key)
        if old == body_s:
            return {"error": "新正文与当前生效正文完全一致，无需提交"}
        return guard.begin(
            "tool_push_templates", "save", f"推送模板 {key}（{T.TEMPLATE_META[key][0]}）整体替换",
            old={"当前生效正文": old}, new={"新正文": body_s},
            op={"kind": "template_save", "key": key, "body": body_s},
            revert_op={"kind": "template_save", "key": key, "body": old})

    if action == "reset":
        if key not in T.KNOWN_VARS:
            return {"error": f"未知模板 key: {key!r}", "可用key": sorted(T.KNOWN_VARS)}
        old = T.get_template(key)
        if key not in T.user_overrides():
            return {"error": f"模板 {key} 当前就是内置默认，无需 reset"}
        return guard.begin(
            "tool_push_templates", "reset", f"推送模板 {key} 恢复内置默认",
            old={"当前生效正文": old}, new={"恢复为内置默认": T.DEFAULT_TEMPLATES.get(key, "")},
            op={"kind": "template_reset", "key": key},
            revert_op={"kind": "template_save", "key": key, "body": old})

    return {"error": f"未知 action: {action}，可用 list/get/save/reset/commit/restore"}


# ================================================================
# ⑤ 调度任务（内置只读 + 自定义任务全生命周期，写操作走两阶段守护）
# ================================================================

def _builtin_job_ids(strict: bool = False) -> set:
    """内置任务 id 集合。

    ⚠️ fail-closed（质量复核 P2-2）：写动作（strict=True）在取不到内置清单时
    **必须报错停止**——否则保护恰好在模块故障时失效，DB 里同名记录可被
    toggle/delete 顶掉/删除。只读 list 用 strict=False（列不出来只影响展示）。
    """
    try:
        from core.scheduler import JOBS
        return {j[0] for j in JOBS}
    except Exception as e:                              # noqa: BLE001
        _logger.warning("[Agent工具] 读取内置 JOBS 失败: %s", e)
        if strict:
            raise RuntimeError(f"内置任务清单不可用，写操作已中止（fail-closed）: {e}")
        return set()


def _parse_enabled(v) -> bool | None:
    """严格布尔解析：**只认 JSON true/false**。字符串一律拒绝——
    `bool("false")==True` 是实测坑（质量复核），宁可让模型重发真布尔，不做猜测。
    """
    return v if isinstance(v, bool) else None


def _job_snapshot(job: dict) -> dict:
    """从 get_scheduled_job 的解析结果还原可入库的干净任务 dict（用于审计/revert）。"""
    return {"job_id": job.get("job_id"), "name": job.get("name"),
            "mode": job.get("mode"), "cron": job.get("cron_raw"),
            "enabled": bool(job.get("enabled", 1)),
            "trading_day_only": bool(job.get("trading_day_only", 1)),
            "note": job.get("note") or ""}


@_tool
def tool_scheduled_jobs(action: str, job_id: str = "", enabled: bool = True,
                        name: str = "", mode: str = "", cron: dict = None,
                        trading_day_only: bool = True, change_id: str = "") -> dict:
    """查看/新增/修改/开关/删除/回滚定时任务（自定义任务全生命周期）。

    action=list      只读：内置任务（代码定义）+ 自定义任务
    action=get       只读：单个自定义任务详情
    action=create    写·预览：新增任务（校验走 core/scheduled_jobs.validate_job 唯一入口），
                     返回 change_id + 冲突提示，不落写
    action=edit      写·预览：改已有任务的 name/mode/cron/trading_day_only（job_id 不变）
    action=toggle    写·预览：启用/停用（enabled 只认 true/false，传 \"false\" 字符串会被拒）
    action=delete    写·预览：删除任务
    action=commit    写·执行：change_id=用户确认过的 preview 凭证（15分钟有效、一次性）
    action=restore   写·回滚：change_id=历史 commit 凭证，按快照撤销

    生效方式：commit 后调度器每约60秒自动同步（无需重启）；内置任务只能看
    （改它需改代码+重启调度器）。create/edit 前必须向用户复述完整任务定义，
    toggle/delete/restore 前必须向用户复述目标 job_id 与操作；用户明确同意后才 commit。
    """
    from db import store
    from core import agent_write_guard as guard
    from core import scheduled_jobs as SJ
    action = str(action or "list").strip().lower()
    job_id = str(job_id or "").strip()

    if action == "commit":
        return guard.commit(change_id)
    if action == "restore":
        if not change_id:
            return {"error": "restore 需要 change_id（历史 commit 的凭证）"}
        return guard.restore(change_id)

    if action == "list":
        builtin = []
        try:
            from core.scheduler import JOBS
            for jid, cron, mode, desc in JOBS:
                builtin.append({"job_id": jid,
                                "时刻": f"{cron.get('hour', 0):02d}:{cron.get('minute', 0):02d}",
                                "类型": mode, "描述": desc,
                                "可修改": "内置任务：改代码+重启调度器才生效"})
        except Exception as e:                          # noqa: BLE001
            _logger.warning("[Agent工具] 内置任务清单读取失败: %s", e)
            builtin.append({"error": f"内置任务清单读取失败: {e}"})
        custom = [{"job_id": j.get("job_id"), "cron": j.get("cron"),
                   "enabled": j.get("enabled"), "trading_day_only": j.get("trading_day_only"),
                   "name": j.get("name"), "mode": j.get("mode"),
                   "error": j.get("error") or ""}
                  for j in store.list_scheduled_jobs()]
        return {"内置任务": builtin, "自定义任务": custom,
                "生效说明": "自定义任务 commit 后由调度器每约60秒自动同步（无需重启）；"
                            "内置任务改动需改代码并重启调度器",
                "边界": "自定义任务支持 create/edit/toggle/delete；内置任务只读"}

    if action == "get":
        if not job_id:
            return {"error": "job_id 不能为空"}
        job = store.get_scheduled_job(job_id)
        if not job:
            if job_id in _builtin_job_ids():
                return {"error": f"{job_id} 是内置任务（代码定义，无数据库记录）",
                        "提示": "内置任务时刻见 action=list；改它需要改代码并重启调度器"}
            return {"error": f"任务不存在: {job_id}"}
        return {"任务": job}

    # ── 以下是写操作：全部只产出 preview（change_id），真正的写在 commit ──
    try:
        builtin_ids = _builtin_job_ids(strict=True)     # fail-closed
    except RuntimeError as e:
        _logger.warning("[Agent工具] 内置任务清单不可用，写操作中止(fail-closed): %s", e)
        return {"error": str(e)}

    if action == "create":
        if job_id in builtin_ids:
            return {"error": f"'{job_id}' 是内置任务 id，不能用；请换名或留空自动生成"}
        try:
            taken = builtin_ids | {j["job_id"] for j in store.list_scheduled_jobs()}
            validated = SJ.validate_job(
                {"job_id": job_id, "name": name, "mode": mode, "cron": cron or {},
                 "trading_day_only": trading_day_only}, taken_ids=taken)
        except ValueError as e:
            _logger.debug("[Agent工具] create 任务定义不合法: %s", e)
            return {"error": f"任务定义不合法: {e}",
                    "可选mode": {k: v.get("label") for k, v in SJ.JOB_MODES.items()}}
        conflicts = SJ.find_conflicts(validated["cron"], validated["mode"],
                                      custom_jobs=store.list_scheduled_jobs())
        preview = guard.begin(
            "tool_scheduled_jobs", "create", f"新增定时任务 {validated['job_id']}（{validated['name']}）",
            old={"任务": "不存在"}, new={"任务": validated},
            op={"kind": "job_upsert", "job": validated},
            revert_op={"kind": "job_delete", "job_id": validated["job_id"]})
        if conflicts:
            preview["排期冲突提示"] = [
                f"{c.get('name')}（{c.get('job_id')}）与其时间窗口重叠" for c in conflicts[:5]]
            preview["提示"] = (preview["下一步"] + "；⚠️ 存在时间窗口重叠，向用户说明后再决定")
        return preview

    if action == "edit":
        if not job_id:
            return {"error": "edit 需要传 job_id"}
        if job_id in builtin_ids:
            return {"error": f"{job_id} 是内置任务，不能由对话修改",
                    "提示": "内置任务的排期要改代码（core/scheduler.py JOBS）并重启调度器"}
        old = store.get_scheduled_job(job_id)
        if not old:
            return {"error": f"任务不存在: {job_id}", "提示": "先 action=list 看自定义任务的 job_id"}
        merged = _job_snapshot(old)
        if name := str(name or "").strip():
            merged["name"] = name
        if mode := str(mode or "").strip():
            merged["mode"] = mode
        if cron:
            merged["cron"] = cron
        merged["trading_day_only"] = trading_day_only
        merged["_original_id"] = job_id
        try:
            taken = builtin_ids | {j["job_id"] for j in store.list_scheduled_jobs()
                                   if j["job_id"] != job_id}
            validated = SJ.validate_job(merged, taken_ids=taken)
        except ValueError as e:
            _logger.debug("[Agent工具] edit 任务定义不合法: %s", e)
            return {"error": f"任务定义不合法: {e}",
                    "可选mode": {k: v.get("label") for k, v in SJ.JOB_MODES.items()}}
        old_snap = _job_snapshot(old)
        conflicts = SJ.find_conflicts(validated["cron"], validated["mode"],
                                      exclude_id=job_id,
                                      custom_jobs=store.list_scheduled_jobs())
        preview = guard.begin(
            "tool_scheduled_jobs", "edit", f"修改定时任务 {job_id}",
            old={"任务(修改前)": old_snap}, new={"任务(修改后)": validated},
            op={"kind": "job_upsert", "job": validated},
            revert_op={"kind": "job_upsert", "job": old_snap})
        if conflicts:
            preview["排期冲突提示"] = [
                f"{c.get('name')}（{c.get('job_id')}）与其时间窗口重叠" for c in conflicts[:5]]
        return preview

    if action == "toggle":
        if not job_id:
            return {"error": "toggle 需要传 job_id"}
        if job_id in builtin_ids:
            return {"error": f"{job_id} 是内置任务，不能由对话开关",
                    "提示": "内置任务的启停要改代码（core/scheduler.py JOBS）并重启调度器"}
        parsed = _parse_enabled(enabled)
        if parsed is None:
            return {"error": f"enabled 只接受 true/false（收到 {enabled!r}）——"
                            "字符串 'false' 会被当 true，是已知坑，直接拒绝"}
        old = store.get_scheduled_job(job_id)
        if not old:
            return {"error": f"任务不存在: {job_id}", "提示": "先 action=list 看自定义任务的 job_id"}
        if bool(old.get("enabled", 1)) == parsed:
            return {"error": f"任务 {job_id} 当前已是 {'启用' if parsed else '停用'} 状态，无需变更"}
        return guard.begin(
            "tool_scheduled_jobs", "toggle", f"{'启用' if parsed else '停用'}任务 {job_id}",
            old={"enabled": bool(old.get("enabled", 1))}, new={"enabled": parsed},
            op={"kind": "job_toggle", "job_id": job_id, "enabled": parsed},
            revert_op={"kind": "job_toggle", "job_id": job_id, "enabled": not parsed})

    if action == "delete":
        if not job_id:
            return {"error": "delete 需要传 job_id"}
        if job_id in builtin_ids:
            return {"error": f"{job_id} 是内置任务，不可删除"}
        old = store.get_scheduled_job(job_id)
        if not old:
            return {"error": f"任务不存在: {job_id}"}
        old_snap = _job_snapshot(old)
        return guard.begin(
            "tool_scheduled_jobs", "delete", f"删除定时任务 {job_id}（{old.get('name')}）",
            old={"任务(将被删除)": old_snap}, new={"任务": "删除"},
            op={"kind": "job_delete", "job_id": job_id},
            revert_op={"kind": "job_upsert", "job": old_snap})

    return {"error": f"未知 action: {action}，可用 list/get/create/edit/toggle/delete/commit/restore"}


# ================================================================
# ⑥ 本地日线面板读取（零联网）
# ================================================================

def _panel_dates(df) -> list | None:
    """面板 DataFrame 的日期序列（兼容 Date 列与 DatetimeIndex 两种形态）。"""
    import pandas as pd
    col = next((c for c in ("date", "Date", "trade_date") if c in getattr(df, "columns", [])),
               None)
    if col is not None:
        return [str(v)[:10] for v in df[col].tolist()]
    idx = getattr(df, "index", None)
    if isinstance(idx, pd.DatetimeIndex):
        return idx.strftime("%Y-%m-%d").tolist()
    return None


@_tool
def tool_panel_bars(codes: list, start: str = "", end: str = "",
                    adjust: str = "qfq") -> dict:
    """读取**本地日线面板**（多标的对比/任意日期区间/复权方式可选；零联网，毫秒级）。

    codes 最多5个6位代码；start/end 支持 2026-10-24 或 20261024（含端点）；
    adjust: qfq前复权(默认)/hfq后复权/none不复权。
    每个标的返回：数据起止/行数/最新收盘/区间涨跌幅/最多60点的收盘价采样（含K线图数据）。

    用户问"对比某几只股票某段区间的走势/看看某只票2024年的行情"时使用——
    比 tool_stock_technicals（单只、固定6个月、另算指标）更适合**多标的与长区间**。
    """
    from data import bars
    if not codes:
        return {"error": "codes 不能为空（最多5个6位代码）"}
    if isinstance(codes, str):                 # 模型偶尔会传字符串，容错成单元素
        codes = [codes]
    if len(codes) > 5:
        return {"error": f"最多支持5个代码（收到 {len(codes)} 个）——"
                        "静默丢弃会让对比结果看似完整，请先缩小范围", "收到代码数": len(codes)}
    clean = []
    for c in codes:
        c = str(c).strip()
        if not (c.isdigit() and len(c) == 6):
            return {"error": f"股票代码须为6位数字: {c!r}"}
        if c not in clean:
            clean.append(c)
    adjust_s = str(adjust or "qfq").strip().lower()
    if adjust_s not in ("qfq", "hfq", "none", ""):
        return {"error": f"adjust 须为 qfq/hfq/none: {adjust!r}"}
    start_s = _norm_date(start) if str(start or "").strip() else None
    end_s = _norm_date(end) if str(end or "").strip() else None

    frames = bars.get_bars_many(clean, start_s, end_s, adjust=adjust_s)
    # ⚠️ manifest 里没有 date_max 键（实测只有 codes/rows/at 等）——"面板市场级最新日"
    #    的权威来源是 bars.panel_date_max()（先读 panel_store overview，再回落 manifest）。
    try:
        panel_max = bars.panel_date_max()
    except Exception as e:                              # noqa: BLE001 - 判不了就留空，不阻断
        _logger.debug("[Agent工具] panel_date_max 不可用: %s", e)
        panel_max = None
    out, missing = {}, []
    for c in clean:
        df = frames.get(c)
        if df is None or getattr(df, "empty", True):
            missing.append(c)
            continue
        close_col = next((k for k in ("close", "Close", "收盘") if k in df.columns), None)
        if close_col is None:
            missing.append(c)
            continue
        closes = df[close_col].astype(float)
        dates = _panel_dates(df) or []
        # 采样最多60点（与 tool_stock_technicals 同款"末位补齐"，防偶数步长跨过末根）
        step = max(1, len(closes) // 60)
        picks = list(range(0, len(closes), step))
        if picks and picks[-1] != len(closes) - 1:
            picks.append(len(closes) - 1)
        picks = picks[-60:]
        sampled = closes.iloc[picks]
        first, last = float(closes.iloc[0]), float(closes.iloc[-1])
        ret = round((last / first - 1) * 100, 2) if first else None
        entry = {"数据起止": [dates[0], dates[-1]] if dates else None,
                 "行数": int(len(closes)),
                 "最新收盘": round(last, 2),
                 "区间涨跌幅%": ret,
                 "kline_sample": {"dates": [dates[i] for i in picks] if dates else [],
                                  "closes": [round(float(x), 2) for x in sampled]}}
        if df.attrs.get("data_lagging"):
            entry["数据滞后"] = f"末根落后参考交易日 {df.attrs.get('lag_days')} 个交易日"
        out[c] = entry
    result = {"标的": out, "复权": adjust_s or "none"}
    if missing:
        result["缺失"] = missing
        result["缺失说明"] = "本地面板没有这些代码的日线（或区间内无数据）——这是'没有'，不是0"
    if panel_max:
        result["面板市场级最新日"] = panel_max
    result["说明"] = ("本地日线（不联网）；个股新鲜度基准=面板市场级最新日"
                      "（bars.panel_date_max），不是今天")
    return result


# ================================================================
# ⑦ 数据新鲜度总览
# ================================================================

@_tool
def tool_data_freshness() -> dict:
    """数据新鲜度总览：今天/最近交易日、本地K线面板覆盖与市场级最新日、各数据集覆盖率。

    回答"数据是不是最新的/数据到哪天了/面板覆盖多少只票"这类问题。
    ⚠️ 口径（项目铁律）：个股新鲜度基准 = **面板市场级最新日（manifest.date_max）**，
    不是今天——收盘前/未回填时最新日落后于今天是正常现象，不代表数据坏了。
    """
    from data import bars
    today = datetime.now()
    info = {"今天": today.strftime("%Y-%m-%d"),
            "星期": "一二三四五六日"[today.weekday()]}
    try:
        from data import trading_calendar as cal
        info["是否交易日"] = cal.is_trading_day(info["今天"])
        info["最近交易日"] = cal.last_trading_day()
    except Exception as e:                              # noqa: BLE001 - 日历不可用不阻断
        _logger.debug("[Agent工具] 交易日历不可用: %s", e)
        info["日历"] = f"不可用（{e}），以工具返回的数据日期为准"

    try:
        cov = bars.coverage()
    except Exception as e:                              # noqa: BLE001
        _logger.warning("[Agent工具] 面板概况读取失败: %s", e)
        cov = {"error": f"面板概况读取失败: {e}"}
    manifest = (cov or {}).get("manifest") or {}
    # ⚠️ manifest 没有 date_max 键（实测只有 codes/rows/at）⇒ 最新日必须问
    #    bars.panel_date_max()（先读 panel_store overview.json，再回落 manifest）。
    try:
        panel_max = bars.panel_date_max()
    except Exception as e:                              # noqa: BLE001
        _logger.debug("[Agent工具] panel_date_max 不可用: %s", e)
        panel_max = None
    panel = {"代码数": cov.get("codes"), "磁盘MB": cov.get("mb"),
             "市场级最新日": panel_max}
    if "error" in cov:
        panel["错误"] = cov["error"]
    if manifest:
        panel["manifest"] = manifest

    coverage_rows = []
    try:
        from db.models import get_connection
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT data_date, dataset, source, got, total, ratio, status "
                "FROM data_coverage ORDER BY id DESC LIMIT 15").fetchall()
            coverage_rows = [_trunc_row(r) for r in rows]
        finally:
            conn.close()
    except sqlite3.Error as e:                          # noqa: BLE001 - 单一来源缺不阻断
        _logger.warning("[Agent工具] data_coverage 读取失败: %s", e)
        coverage_rows = [{"error": f"data_coverage 读取失败: {e}"}]

    return {**info,
            "本地K线面板": panel,
            "数据集覆盖(data_coverage)": coverage_rows or "（暂无记录）",
            "口径说明": ("个股新鲜度基准 = 面板市场级最新日（bars.panel_date_max()），"
                        "不是今天。最新日之后没有某只票的数据，可能是尚未收盘、"
                        "停牌或未回填——是'缺'不是'0'，不得当作 0 值参与计算")}


# ================================================================
# ⑧ 结论类薄包装（2026-10-24 审查建议：把"已算好的层"接给助手）
#    全部是已存在、页面已在用的函数的薄包装，不写新查询逻辑。
#    验收线（审查模板）：as_of+口径 / 没有≠0 / n<门槛不报结论 / 字符配额 / 正反用例。
# ================================================================

# 胜率类结论的最低样本门槛（审查实测 bull_samples=4 就报 100% ⇒ 必须拦在包装层）
_MIN_ANALYSIS_SAMPLES = 10
_MIN_PICK_SAMPLES = 20
_LIST_CAP = 10          # 明细列表最多返回条数（防截断成半个 JSON）


def _gate_win_rate(samples: int, win_rate, min_samples: int, label: str) -> dict:
    """n < 门槛 ⇒ 只报样本数、不报胜率结论。绝不能把 100%（n=4）当结论给模型。"""
    if win_rate is None:
        return {"样本数": samples, "胜率": "不可评（无已回填样本）"}
    if samples < min_samples:
        return {"样本数": samples, "胜率": None,
                "结论门槛": f"n<{min_samples} 不作结论",
                "说明": f"{label}样本仅 {samples} 个，胜率 {win_rate} 是噪声不是结论"}
    return {"样本数": samples, "胜率": f"{win_rate * 100:.0f}%"}


@_tool
def tool_quality_stats(days: int = 90, kind: str = "overview") -> dict:
    """选股/分析到底有没有用的结论统计（页面同款函数，薄包装）。

    kind=overview  选股表现 + 分析胜率的摘要（默认；带样本门槛，n 不足不报结论）
    kind=picks     选股整体表现（T+5 收益/胜率/**超额**，缺基准样本不计入超额）
    kind=analysis  深度分析决策胜率（看多/看空对称口径，只统计已回填样本）
    kind=conditions 按选股条件的区分度（store 自带 min_samples=3 门槛）
    kind=factors   因子得分与收益相关性（Pearson，store 自带 min_samples=5 门槛）

    ⚠️ 口径纪律：样本不足时**只报样本数、不报胜率**；超额样本单独计，不拿 0 顶替；
       待回填样本列在" pending_backfill "，是'还没到期'不是'失败'。
    """
    from db import store
    kind = str(kind or "overview").strip().lower()
    d = max(1, min(int(days) if days else 90, 365))

    if kind == "picks":
        s = store.screen_pick_stats(d)
        return {"口径": f"近 {s['recent_days']} 天、按(日期,代码)去重、T+5",
                "样本数": s["samples"], "T+5平均收益": s["avg_ret_5d"],
                "T+5胜率": _gate_win_rate(s["samples"], s["win_rate_5d"],
                                         _MIN_PICK_SAMPLES, "选股"),
                "超额样本数": s["excess_samples"], "平均超额": s["avg_excess_5d"],
                "超额胜率": _gate_win_rate(s["excess_samples"], s["excess_win_rate"],
                                          _MIN_PICK_SAMPLES, "超额"),
                "待回填": s["pending_backfill"],
                "说明": "超额=个股收益-同期上证；牛市里只有超额能说明选股有效"}

    if kind == "analysis":
        a = store.analysis_stats(min(d, 60))
        out = {"口径": f"近 {a['recent_days']} 天、已回填样本、看多/看空对称口径",
               "分析总数": a["total"], "标签分布": a["by_label"],
               "看多": _gate_win_rate(a["bull_samples"], a["bull_win_rate"],
                                     _MIN_ANALYSIS_SAMPLES, "看多"),
               "看空": _gate_win_rate(a["bear_samples"], a["bear_win_rate"],
                                     _MIN_ANALYSIS_SAMPLES, "看空"),
               "待回填": a["pending_backfill"]}
        if a["pending_backfill"]:
            out["说明"] = ("胜率只统计已回填（T+5 已到期）样本；待回填是'还没到期'"
                          "不是失败，别当成亏损")
        return out

    if kind == "conditions":
        c = store.condition_stats(d)
        return {"口径": f"近 {c['recent_days']} 天、单条件样本≥{c['min_samples']} 才列出",
                "总样本": c["samples"], "基线": c["baseline"],
                "条件": (c["conditions"] or "（没有任何条件达到样本门槛——是'样本不足'不是'条件无效'）")[:12]}

    if kind == "factors":
        f = store.factor_stats(d)
        return {"口径": f"近 {f['recent_days']} 天、Pearson 相关、单因子样本≥{f['min_samples']}",
                "总样本": f["samples"],
                "因子": (f["factors"] or "（没有任何因子达到样本门槛）")[:12]}

    if kind == "overview":
        s = store.screen_pick_stats(d)
        a = store.analysis_stats(min(d, 60))
        return {"选股": {"样本数": s["samples"],
                         "T+5胜率": _gate_win_rate(s["samples"], s["win_rate_5d"],
                                                  _MIN_PICK_SAMPLES, "选股"),
                         "超额胜率": _gate_win_rate(s["excess_samples"],
                                                   s["excess_win_rate"], _MIN_PICK_SAMPLES, "超额"),
                         "平均超额": s["avg_excess_5d"], "待回填": s["pending_backfill"]},
                "深度分析": {"总数": a["total"],
                            "看多胜率": _gate_win_rate(a["bull_samples"], a["bull_win_rate"],
                                                     _MIN_ANALYSIS_SAMPLES, "看多"),
                            "看空胜率": _gate_win_rate(a["bear_samples"], a["bear_win_rate"],
                                                     _MIN_ANALYSIS_SAMPLES, "看空"),
                            "待回填": a["pending_backfill"]},
                "提醒": ("样本不足的维度只列样本数、不报胜率——胜率必须连同样本数一起引用，"
                        "禁止把'胜率100%'单独说成结论")}

    return {"error": f"未知 kind: {kind}，可用 overview/picks/analysis/conditions/factors"}


@_tool
def tool_llm_cost(days: int = 3) -> dict:
    """LLM 成本与预算（估算口径，页面同款函数）。

    返回近 N 天总花费/按日/按用途 Top5/按模型 + 今日预算状态（cap/已花/剩余/是否超限）。
    ⚠️ 口径：金额按 config/llm_prices.json 单价表**估算**（非账单）；
       失败调用也计 cost；degraded=True 表示账目查不清，不是"没花钱"。
    """
    from core import llm_cost
    d = max(1, min(int(days) if days else 3, 30))
    rep = llm_cost.daily_report(d)
    budget = llm_cost.budget_status()
    return {"口径": f"近 {d} 天、按 llm_prices.json 估算（非账单）",
            "总花费元": rep["total_cost"], "总调用": rep["total_calls"],
            "总tokens": rep["total_tokens"],
            "按日": rep["days"][:d],
            "用途Top5": rep["by_purpose"][:5],
            "按模型": rep["by_model"][:6],
            "工作日均花费元": rep["avg_workday_cost"], "有效工作日数": rep["workday_count"],
            "预算": {"日上限元": budget.get("cap"), "今日已花元": budget.get("spent"),
                     "剩余元": budget.get("remaining"),
                     "已超限": budget.get("exceeded"), "闸门启用": budget.get("enabled"),
                     "账目异常": budget.get("degraded", False)}}


@_tool
def tool_system_health() -> dict:
    """系统健康自检（调度器心跳/旧代码/漏跑缺口/自定义任务同步）。

    回答"你跑的是新代码吗""调度器活着吗""哪个任务没生效"。
    ⚠️ 口径：心跳过期 ≠ 进程死（可能在跑长任务）；`code.stale` 的 missing 清单就是
    "改了代码还没重启"的文件列表——这正是"改了配置没生效"的可查证据。
    """
    from core import scheduler_health
    h = scheduler_health.check_health(check_processes=True, check_schedule_gap=True)
    code = h.get("code") or {}
    sched = h.get("scheduler") or {}
    return {"调度器": {"运行中": sched.get("running"), "原因": sched.get("reason"),
                     "心跳时间": sched.get("ts"), "空闲秒": sched.get("idle_seconds")},
            "代码新鲜度": {"跑的是旧代码": bool(code.get("stale")),
                        "比代码旧的起始文件": code.get("missing") or [],
                        "未预期的额外文件": code.get("extra") or []},
            "自定义任务": h.get("custom_jobs"),
            "调度缺口": h.get("schedule_gap"),
            "进程检查": h.get("long_running"),
            "需要处理": (h.get("warnings") or [])[:8]}


@_tool
def tool_decision_digest(as_of: str = "") -> dict:
    """今日决策摘要：选了什么/分析了什么/持仓与待触发/风险清单。

    回答"今天该做什么""上次判断对不对"。空候选 ≠ 没选——`skipped` 原因会如实带出
    （熊市跳过选股是设计行为）。数据源是 decision_digest 模块（页面同款）。
    """
    from core import decision_digest
    d_str = _norm_date(as_of) if str(as_of or "").strip() else None
    td = decision_digest.today_decisions(d_str)
    al = decision_digest.action_list(d_str)
    out = {"日期": td.get("date"),
           # ⚠️ 2026-10-29：`screens` 已改为**推荐名单**（daily_picks，做过行业分散），
           #    扩池候选单独在 `screen_pool`。阿深问"今天筛出多大池子"要看 pool，
           #    问"今天推荐了哪几只"要看 screens —— 两者混用会出现
           #    "候选数 5"（本该 110）或"名单 4 只银行"（本该 2 只）这种错。
           "选股": {"推荐数": len(td.get("screens") or []),
                    "候选数": len(td.get("screen_pool") or []),
                    "跳过原因": td.get("empty_reasons") or [],
                    "推荐": (td.get("screens") or [])[:_LIST_CAP],
                    "候选": (td.get("screen_pool") or [])[:_LIST_CAP]},
           "深度分析": (td.get("analyses") or [])[:_LIST_CAP],
           "持仓变动": (td.get("trades") or [])[:_LIST_CAP],
           "待触发观察": (al.get("watch") or [])[:_LIST_CAP],
           "风险清单": (al.get("risks") or [])[:_LIST_CAP],
           "当前持仓数": len(al.get("positions") or [])}
    if not td.get("screens") and not td.get("analyses"):
        out["说明"] = ("今日没有选股/分析产物——先看跳过原因：熊市跳过是设计行为，"
                      "不是系统故障")
    return out


def _cap_list(v, cap: int = _LIST_CAP) -> list:
    """把"可能是 list 也可能是 dict"的明细统一裁成 ≤cap 的 list（防截断成半个 JSON）。"""
    if isinstance(v, dict):
        return [{"key": k, **(x if isinstance(x, dict) else {"值": x})}
                for k, x in list(v.items())[:cap]]
    if isinstance(v, list):
        return v[:cap]
    return []


@_tool
def tool_holdings_risk() -> dict:
    """真实持仓风险体检：集中度/行业暴露/止损状态/不可评项。

    回答"我的持仓危险吗""止损位合适吗"。与 tool_portfolio_positions（账本+盈亏）不同，
    本工具给的是**风险结论**。行业/流动性/快照不可评时单列"不可评"，不并入"其他"。
    """
    from core import position_risk
    from data import realtime
    built = None
    try:
        codes = position_risk.build_holdings()["holdings"]
        codes = [h.get("code") for h in codes if h.get("code")]
        quotes = realtime.get_realtime_quotes(codes) if codes else {}
        built = position_risk.evaluate_positions(prices=quotes)
    except Exception as e:                              # noqa: BLE001 - 体检失败给 error，不编
        _logger.warning("[Agent工具] 持仓风险体检失败: %s", e)
        return {"error": f"持仓风险体检不可用: {e}"}
    if not built:
        return {"error": "持仓风险体检无结果"}
    return {"快照日": built.get("snapshot_date"),
            "总资产": built.get("total_equity"), "现金": built.get("cash"),
            "现金占比": built.get("cash_ratio"),
            "仓位占比": built.get("position_ratio"),
            "最大单一持仓": built.get("top_position"),
            "行业暴露": (built.get("sector_exposure") or {})[:12] if isinstance(
                built.get("sector_exposure"), list) else built.get("sector_exposure"),
            "已评估持仓": _cap_list(built.get("positions")),
            "风险问题": _cap_list(built.get("issues")),
            "止损/止盈建议": _cap_list(built.get("limits")),
            "不可评说明": {"数据缺失": built.get("data_unavailable") or built.get("unavailable") or [],
                          "缺行业": built.get("missing_industry") or [],
                          "缺快照": built.get("missing_snapshot") or [],
                          "流动性不可评": built.get("unavailable_liquidity") or []}}
