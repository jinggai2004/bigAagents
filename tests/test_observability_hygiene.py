# -*- coding: utf-8 -*-
"""可观测性卫生守护（2026-10-17，P1-1 收尾）。

## 为什么要有它

`logs/scheduler_*.log` **只收 `logging` 输出**。所以长驻进程里的 `print` = 观测盲区：
实测事故是资金流快照全天失败，日志里却只有 `采集完成: 个股 0 只 / 行业 0 个`，
真因在被吞的 `print` 里。同类的还有"`except` 里什么都不做"——失败彻底隐形。

## 两条守护 + 一条自检

1. **`core/` 与 `data/` 不得有非 CLI 的 `print`**（`__main__` 块里的 print 是命令行输出，保留）；
2. **静默 `except` 不得增长**（棘轮 ratchet）—— 存量 300 处没法一次改完，
   但**不许再增加**；每修掉一处就把基线往下调。
3. ⚠️ **鉴别力自检**：用临时造出的违规样本验证上面两条**真的能抓到** ——
   否则"全绿"可能只是因为检查写成了恒真（本项目已踩过：守护退化成假安全）。
"""
import ast
import io
import os
import re
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# 打印迁移的范围：core/ 与 data/ 是**取数与计算层**（长驻进程里跑，必须进日志）。
# scripts/ 是命令行工具，其 print 是**刻意的控制台输出**，不在此列。
LOG_SCOPE = ("core", "data")

# ⚠️ 棘轮基线（**2026-10-18 分层实测**）。**这不是"允许这么多"**，而是"不许再增加"：
#    修掉任意一处后请把它调小。
#
# 为什么分两层（D8，2026-10-18 审查）：
#   原来只有一个 300 的总基线，于是 `scripts/audit_*.py`（只读审计脚本，"跳过某维度"
#   是设计语义）新增 10 处，也能"借"核心层的额度 —— **核心链路新增 13 处仍然合法**。
#   现在分开：
#     · 核心层 `core/ data/ db/ llm/ ui/`（长驻进程跑的计算/取数/存储层）——严格，新增必须补日志；
#     · 脚本层 `scripts/`（CLI 工具）——宽松，但仍不许暴涨。
# 2026-10-23 按实测**收紧**（不是放宽）：补完 5 处留痕后
#   核心层 279 → 277，脚本层 37 → 34。
# 2026-10-24 再收紧到实测值：`core/report_audit.py` 7 处数值解析兜底补了留痕
#   ⇒ 核心层 277 → **273**（同批还补了 `scripts/` 3 处，脚本层维持 34）。
# ⚠️ 基线只能往下调；要往上调必须先在 PR/提交说明里解释为什么那个 except 不能记日志。
CORE_SILENT_EXCEPT_BASELINE = 273
SCRIPT_SILENT_EXCEPT_BASELINE = 34

_SILENT_ABSENT = re.compile(
    r"(_logger|logging|st\.(warning|error|info|caption)|raise|warnings\.warn"
    r"|\.set_exception\()")   # set_exception 是**显式异常传递**（fut.result() 会重抛）⇒ 与 raise 等价


def _iter_py(roots):
    for root in roots:
        for dp, _dn, fn in os.walk(root):
            if "__pycache__" in dp:
                continue
            for f in fn:
                if f.endswith(".py"):
                    yield os.path.join(dp, f)


def _main_block_ranges(tree):
    out = []
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.dump(node.test):
            out.append((node.lineno, node.end_lineno or node.lineno))
    return out


def find_noncli_prints(roots):
    """返回 [(文件, 行号)]，只算 `__main__` 块**之外**的 print。"""
    hits = []
    for p in _iter_py(roots):
        try:
            tree = ast.parse(io.open(p, encoding="utf-8").read())
        except SyntaxError:
            continue
        ranges = _main_block_ranges(tree)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                    and not any(a <= node.lineno <= b for a, b in ranges)):
                hits.append((p, node.lineno))
    return hits


def _except_types(ln: str) -> list:
    """解析 `except X as e:` / `except (A, B):` 的类型名；裸 `except:` 返回 []。

    ⚠️ 只认**单行** except 子句。多行写法（`except (\n A,\n B\n):`）在这里
    解析不出来，会退化成"有类型但识别不到白名单"⇒ 按未豁免处理（偏保守）。
    """
    m = re.match(r"^\s*except\s*(.*?)\s*:\s*(?:#.*)?$", ln)
    if not m:
        return []
    body = m.group(1).strip()
    if not body:
        return []
    body = re.sub(r"\s+as\s+\w+\s*$", "", body)     # 去掉 `as e`
    body = body.strip().strip("()")
    return [t.strip() for t in body.split(",") if t.strip()]


# 「正常缺失」类型：`except FileNotFoundError: return {}` 表达的是**显式语义**
# （"文件不存在"），不是"吞掉未知错误"。让它们打日志会把常态当故障——
# `watchdog.py` 每 5 分钟读一次心跳，"心跳文件不存在"是它正常的启动状态。
#
# ⚠️ 白名单必须按**类型**，不能按**文件/路径**：整文件豁免会把真正的吞错
#    一起放行（审查 2026-10-20 明确提醒过这一点）。
# ⚠️ 只豁免**纯**缺失类型：`except (FileNotFoundError, OSError)` 里 OSError
#    范围更宽，仍应打日志 —— 见 `_except_types` 的 all() 判断。
_MISSING_TYPES = {"FileNotFoundError", "IsADirectoryError", "NotADirectoryError"}


def find_silent_excepts(roots, ignore_missing: bool = False):
    """返回 [(文件, 行号)]：`except` 块里**既无日志、也无用户提示、也不 raise**。

    `ignore_missing=True` 时，`except` 类型**全部**属于 `_MISSING_TYPES` 的块
    视为"正常缺失"而不计入。默认 False ⇒ `core/`/`data/`/`db/`/`llm/` 的既有口径
    一字不变（那里从来没有这类豁免，也不该有）。
    """
    hits = []
    for p in _iter_py(roots):
        lines = io.open(p, encoding="utf-8").read().splitlines()
        for i, ln in enumerate(lines):
            m = re.match(r"^(\s*)except\s", ln)
            if not m:
                continue
            if ignore_missing:
                types = _except_types(ln)
                if types and all(t in _MISSING_TYPES for t in types):
                    continue
            indent = len(m.group(1))
            body = []
            for nxt in lines[i + 1:]:
                if not nxt.strip():
                    body.append(nxt)
                    continue
                if len(nxt) - len(nxt.lstrip()) <= indent:
                    break
                body.append(nxt)
            if not _SILENT_ABSENT.search("\n".join(body)):
                hits.append((p, i + 1))
    return hits


class TestPrintIsNotAnObservabilityBlindSpot:
    def test_no_noncli_print_in_data_and_core(self):
        """⭐ `core/`、`data/` 里不得再有非 CLI 的 `print`（失败必须进日志文件）。"""
        hits = find_noncli_prints([os.path.join(_ROOT, d) for d in LOG_SCOPE])
        assert hits == [], (
            "这些 print 在长驻进程里进不了日志文件（应改用 `_logger`）：\n  "
            + "\n  ".join(f"{os.path.relpath(p, _ROOT)}:{ln}" for p, ln in hits[:20]))
        assert len(list(_iter_py([os.path.join(_ROOT, "data")]))) > 10, \
            "扫描范围内文件数异常 ⇒ 路径写错了（守卫会变成恒真的空断言）"


class TestSilentExceptDoesNotGrow:
    def _assert_within(self, scope, baseline, label, ignore_missing=False):
        hits = find_silent_excepts([os.path.join(_ROOT, d) for d in scope],
                                   ignore_missing=ignore_missing)
        worst = {}
        for p, _ln in hits:
            worst[p] = worst.get(p, 0) + 1
        top = sorted(worst.items(), key=lambda kv: -kv[1])[:5]
        assert len(hits) <= baseline, (
            f"{label}静默 except 从 {baseline} 涨到了 {len(hits)}。"
            "新增的 except 请至少记一条日志（`_logger.warning(...)`）。"
            "当前最多的是：\n  "
            + "\n  ".join(f"{os.path.relpath(p, _ROOT)}: {n} 处" for p, n in top))

    def test_core_layer_within_baseline(self):
        """⭐⭐ 核心层（长驻进程）**独立基线** —— 不许借脚本层的额度（D8）。"""
        self._assert_within(("core", "data", "db", "llm", "ui"),
                            CORE_SILENT_EXCEPT_BASELINE, "核心层")

    def test_script_layer_within_baseline(self):
        """脚本层（CLI 工具）单独基线：审计脚本"跳过某维度"是设计语义。

        ⚠️ 2026-10-20：`scripts/` 层允许**纯缺失类型**豁免（见 `_MISSING_TYPES`）。
        起因是 `watchdog.py` 新增 15 处 `except`，其中多数是
        `except FileNotFoundError: return {}`（心跳不存在是守护的正常启动状态），
        给它们补日志只会把常态当故障。核心层**不开这个口子**。
        """
        self._assert_within(("scripts",), SCRIPT_SILENT_EXCEPT_BASELINE, "scripts/",
                            ignore_missing=True)


class TestGuardHasDiscriminance:
    """⚠️ 守卫的**鉴别力自检**：违规样本必须被抓住。

    没有这一条，"检查通过"可能只是因为检查写错了（恒真）—— 本项目已有的教训。
    """

    def test_detects_noncli_print(self, tmp_path):
        bad = tmp_path / "sample.py"
        bad.write_text('def f():\n    print("失败")\n', encoding="utf-8")
        assert find_noncli_prints([str(tmp_path)]), "非 CLI print 没被抓住 ⇒ 检查无效"

    def test_ignores_cli_print(self, tmp_path):
        ok = tmp_path / "cli.py"
        ok.write_text('if __name__ == "__main__":\n    print("命令行输出")\n',
                      encoding="utf-8")
        assert find_noncli_prints([str(tmp_path)]) == [], \
            "__main__ 块的 print 是 CLI 输出，不该被误报"

    def test_detects_silent_except(self, tmp_path):
        bad = tmp_path / "sample.py"
        bad.write_text("def f():\n    try:\n        pass\n    except Exception:\n        return None\n",
                       encoding="utf-8")
        assert find_silent_excepts([str(tmp_path)]), "静默 except 没被抓住 ⇒ 检查无效"

    def test_accepts_logged_except(self, tmp_path):
        ok = tmp_path / "sample.py"
        ok.write_text("def f():\n    try:\n        pass\n"
                      "    except Exception as e:\n        _logger.warning('x: %s', e)\n",
                      encoding="utf-8")
        assert find_silent_excepts([str(tmp_path)]) == [], "记了日志还被误报"


class TestCaptureReportsTriState:
    """P0-1 的结构性守护：采集必须报三态，调度必须看 `status`。

    ⚠️ "验了函数 ≠ 验了接线"：只断言 `capture_*` 返回值含 `status` 不够，
       还要断言**调度侧真的读了它** —— 否则又回到"0 只却记完成"。
    """

    def test_capture_returns_status_and_failures(self):
        from data import fund_flow as F
        src = io.open(os.path.join(_ROOT, "data", "fund_flow.py"), encoding="utf-8").read()
        assert '"status": status' in src, "采集返回值缺 status（无法三态报告）"
        assert '"failures"' in src, "采集返回值缺 failures（失败原因不可查）"
        assert hasattr(F, "_ths_last_error"), "_ths_last_error 是失败原因的唯一来源"

    def test_scheduler_reads_status_and_warns(self):
        src = io.open(os.path.join(_ROOT, "core", "scheduler.py"),
                      encoding="utf-8").read()
        assert 'result.get("status")' in src, "调度侧没有读 status ⇒ 三态报告等于没接线"
        assert "_logger.warning" in src


class TestDegradationTravelsWithData:
    """「降级必须随数据走」（P1-2）：来源不可用时**结果本身**要带得出来。

    实测：新浪全市场代码清单拿不到 ⇒ 本次**不过滤退市标的**，覆盖率只能是 `?`，
    而原实现照样打成 `✅ …（覆盖 ?）`（INFO）⇒ 下游完全看不出"这份截面有问题"。
    """

    def test_sina_marks_degraded_in_attrs(self):
        src = io.open(os.path.join(_ROOT, "data", "sina_source.py"),
                      encoding="utf-8").read()
        assert '"degraded": _degraded' in src, "快照 attrs 未记录降级"
        assert "code_list_unavailable" in src, "缺少具体的降级原因标识"
        assert '_logger.warning("[新浪快照] ⚠️' in src, \
            "降级时必须以 WARNING 记（原来的 ✅ INFO 会被读成成功）"

    def test_scheduler_passes_degraded_through_and_warns(self):
        src = io.open(os.path.join(_ROOT, "core", "scheduler.py"),
                      encoding="utf-8").read()
        # 2026-10-17 起改为先取 `_degraded = df.attrs.get("degraded") or []`（P2-7
        # 还要用它决定覆盖率的 status），return 里是 `"degraded": _degraded`。
        # ⚠️ 断言别绑定具体变量名 —— 绑定"return 里带出了 degraded"这个**契约**。
        assert '"degraded": _degraded' in src or '"degraded": df.attrs.get("degraded")' in src, \
            "调度层没有把来源层的 degraded 带出去 ⇒ 页面/健康检查看不到"
        assert 'm.get("degraded")' in src, "调度没有按 degraded 分流日志级别"


class TestWatchJobLogLevel:
    """P2-1：`提醒 0 条` 是**正常结果**，但不该记 INFO —— 实测每天 183 行噪音把真事件埋了。

    ⚠️ 注意本用例**不是**在说"提醒 0 条 是 bug"：它是冷却期 30 分钟 + 每分钟扫描的必然结果。
       要钉的是**日志级别**，不是逻辑。
    """

    class _StubWatcher:
        def __init__(self, fired, saved):
            self.fired, self.saved = fired, saved

        def scan_once(self):
            return {"skipped": False, "checked": 28, "alerts": 20,
                    "fired": self.fired, "saved": self.saved}

    def _run(self, monkeypatch, caplog, fired, saved):
        from core import watcher as W
        monkeypatch.setattr(W, "get_watcher",
                            lambda: TestWatchJobLogLevel._StubWatcher(fired, saved))
        from core import scheduler
        with caplog.at_level("DEBUG"):
            scheduler.run_watch_job("测试盯盘")
        return [r for r in caplog.records if "命中" in r.getMessage()]

    def test_no_event_logged_at_debug(self, monkeypatch, caplog):
        recs = self._run(monkeypatch, caplog, fired=0, saved=0)
        assert recs and all(r.levelname == "DEBUG" for r in recs), \
            "「提醒 0 条」不该占 INFO（每天上百行噪音）"

    def test_real_event_logged_at_info(self, monkeypatch, caplog):
        """⚠️ 鉴别力：真触发时必须回到 INFO，否则修复会把真事件也埋掉。"""
        recs = self._run(monkeypatch, caplog, fired=3, saved=3)
        assert recs and all(r.levelname == "INFO" for r in recs), \
            "真触发必须 INFO（不能为了降噪把事件也降级）"
