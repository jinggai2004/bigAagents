# -*- coding: utf-8 -*-
"""用户自定义因子的安全表达式层（DSL）。

为什么不是"让用户写 Python"
──────────────────────────
用户能在页面上填的只有一行表达式，它会在**服务进程里**对 50 万行面板求值。
`eval()` + 黑名单挡不住 `__class__`/`__subclasses__` 这类逃逸，而白名单正则
又容易被绕过。这里的做法是**结构性守护**：自己遍历 AST，只接受列出的节点类型，
根本不调 `eval()`——不是"过滤危险写法"，而是"不认识的写法直接拒绝"。

已被拒的（写进测试钉住）
────────────────────────
属性访问 `a.__class__` / 下标 `a[0]` / lambda / 赋值 / import / 字典与集合
字面量 / 海象 / 推导式 / f-string / await / 调用非白名单函数。

允许的
──────
- 字段名（量价面板列 + 财务 PIT 列，见 `FIELD_GROUPS`）
- 数字常量、四则运算、幂、取负/取正
- 比较与 `and/or`（结果转 0/1，供 `where` 用）
- 白名单函数（见 `FUNCS`），含两个**截面**算子 `cs_rank`/`cs_z`

⚠️ 目标列被结构性排除
────────────────────
`fwd20`（未来 20 日收益）与 `month_ts`/`code` 等主键绝不出现在可用字段里。
否则用户一句 `fwd20` 就能让 IC 变成 1.0 并"通过"所有检验——
这不是刁难，是回测里最常见的自欺。
"""
from __future__ import annotations

import ast
import math
import re

import numpy as np
import pandas as pd

MAX_EXPR_LEN = 400          # 一行表达式放得下；也是防 DoS 的第一道
MAX_DEPTH = 24              # AST 深度上限（嵌套过深时递归求值会爆栈）

# ── 可用字段 ────────────────────────────────────────────────────────────
# 量价面板列（`temp/factor_scan_scan.parquet` 的 30 个因子列）
PRICE_FIELDS = {
    "mom12_1": "12个月动量（剔除最近1月）",
    "mom1m": "1个月动量/反转",
    "mom1w": "1周动量",
    "mom6m": "6个月动量",
    "high52": "距52周高点",
    "ret_max5": "近5日最大单日涨幅",
    "up_streak": "连涨天数",
    "ivol20": "20日残差波动率（IVOL）",
    "vol20": "20日总波动率",
    "max20": "近20日最大单日涨幅（MAX）",
    "dvol20": "下行波动率",
    "skew20": "收益偏度",
    "kurt20": "收益峰度",
    "vol_ratio": "量比",
    "turn20": "20日平均换手率",
    "turn_std20": "换手率20日标准差",
    "amihud20": "Amihud非流动性",
    "lndvol20": "log成交额20日",
    "turn_trend": "换手率趋势",
    "turn_shock": "换手率冲击",
    "lnfloat": "log流通市值",
    "pv_corr20": "量价相关20日",
    "obv_slope": "OBV斜率",
    "vr5_20": "5日/20日量比",
    "neg_rsi14": "RSI超买度（取负）",
    "boll_pos": "布林带位置",
    "upshd20": "20日上影占比",
    "macd_hist": "MACD柱",
    "atr_pct": "ATR占价比",
    "gapfreq20": "跳空频率",
}

# 财务 PIT 列（`core/financial_pit.pit_snapshot` 的输出 + 原始每股列）
FIN_FIELDS = {
    "roe": "ROE（加权，小数）",
    "gp_margin": "毛利率（银行无=NaN）",
    "np_margin": "净利率",
    "roe_yoy": "ROE同比变化（差值）",
    "profit_yoy": "净利润同比增速",
    "ocf_to_np": "经营现金流/净利润（盈利质量）",
    "cfo_to_np": "现金流/净利润（旧源，覆盖低）",
    "cfo_np_yoy": "现金流/净利同比变化",
    "ystz": "营收同比（东财直给）",
    "sjltz": "净利同比（东财直给）",
    "eps": "每股收益（累计）",
    "bps": "每股净资产",
    "ocfps": "每股经营现金流（累计）",
    "revenue": "营业收入（元，量级大，建议套 log）",
    "net_profit": "归母净利润（元，量级大，建议套 log）",
}

FIELD_GROUPS = {"量价": PRICE_FIELDS, "财务": FIN_FIELDS}
ALL_FIELDS = {**PRICE_FIELDS, **FIN_FIELDS}

# ── 白名单函数 ──────────────────────────────────────────────────────────
_UNARY = {"abs": np.abs, "log": np.log, "log1p": np.log1p, "log10": np.log10,
          "sqrt": np.sqrt, "sign": np.sign, "exp": np.exp}
_BINARY = {"minimum": np.minimum, "maximum": np.maximum}
_SCALAR_ARG = {"clip"}          # clip(x, lo, hi)
_CS = {"cs_rank", "cs_z"}       # 截面算子：按月分组
FUNCS = sorted(set(_UNARY) | set(_BINARY) | _SCALAR_ARG | _CS)


class FactorExprError(ValueError):
    """表达式不合法（信息直接给用户看，要具体到哪个节点不行）。"""


def _node_name(n: ast.AST) -> str:
    return type(n).__name__


def validate(expr: str) -> dict:
    """静态校验表达式，返回 `{ast, fields, funcs, depth}`。不合法抛 FactorExprError。"""
    raw = (expr or "").strip()
    if not raw:
        raise FactorExprError("表达式为空")
    if len(raw) > MAX_EXPR_LEN:
        raise FactorExprError(f"表达式过长（{len(raw)} 字符，上限 {MAX_EXPR_LEN}）")
    if "__" in raw:
        raise FactorExprError("表达式里不许出现双下划线")
    try:
        tree = ast.parse(raw, mode="eval")
    except SyntaxError as e:
        raise FactorExprError(f"语法错误：{e.msg}（第 {e.offset} 字符）") from e

    fields: set[str] = set()
    funcs: set[str] = set()

    def walk(n: ast.AST, depth: int) -> None:
        if depth > MAX_DEPTH:
            raise FactorExprError(f"嵌套过深（>{MAX_DEPTH} 层）")
        if isinstance(n, ast.Expression):
            return walk(n.body, depth + 1)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
                raise FactorExprError(f"只允许数字常量，收到 {n.value!r}")
            return None
        if isinstance(n, ast.Name):
            if n.id not in ALL_FIELDS:
                hint = _suggest(n.id)
                raise FactorExprError(f"未知字段 `{n.id}`{hint}")
            fields.add(n.id)
            return None
        if isinstance(n, ast.BinOp):
            if not isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
                raise FactorExprError(f"不支持的运算符 {type(n.op).__name__}")
            walk(n.left, depth + 1)
            return walk(n.right, depth + 1)
        if isinstance(n, ast.UnaryOp):
            if not isinstance(n.op, (ast.UAdd, ast.USub)):
                raise FactorExprError(f"不支持的一元运算符 {type(n.op).__name__}")
            return walk(n.operand, depth + 1)
        if isinstance(n, ast.Compare):
            if len(n.ops) != 1:
                raise FactorExprError("比较只允许两个操作数（不要链式 a<b<c）")
            if not isinstance(n.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE,
                                         ast.Eq, ast.NotEq)):
                raise FactorExprError(f"不支持的比较 {type(n.ops[0]).__name__}")
            walk(n.left, depth + 1)
            return walk(n.comparators[0], depth + 1)
        if isinstance(n, ast.BoolOp):
            if not isinstance(n.op, (ast.And, ast.Or)):
                raise FactorExprError(f"不支持的逻辑运算 {type(n.op).__name__}")
            for v in n.values:
                walk(v, depth + 1)
            return None
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise FactorExprError("只允许调用白名单函数，不允许方法调用")
            fn = n.func.id
            if fn not in FUNCS:
                raise FactorExprError(f"未知函数 `{fn}`（可用：{', '.join(FUNCS)}）")
            if n.keywords:
                raise FactorExprError(f"`{fn}` 不接受关键字参数")
            if fn in _UNARY and len(n.args) != 1:
                raise FactorExprError(f"`{fn}` 需要 1 个参数")
            if fn in _BINARY and len(n.args) != 2:
                raise FactorExprError(f"`{fn}` 需要 2 个参数")
            if fn == "clip" and len(n.args) != 3:
                raise FactorExprError("`clip(x, lo, hi)` 需要 3 个参数")
            if fn in _CS and len(n.args) != 1:
                raise FactorExprError(f"`{fn}` 需要 1 个参数")
            funcs.add(fn)
            for a in n.args:
                walk(a, depth + 1)
            return None
        raise FactorExprError(f"不允许的写法：{_node_name(n)}")

    walk(tree, 0)
    if not fields:
        raise FactorExprError("表达式里没有任何字段（纯数字常量没有因子含义）")
    return {"ast": tree, "fields": sorted(fields), "funcs": sorted(funcs)}


def _suggest(name: str) -> str:
    """拼错的字段名给个近似提示（大小写/前后缀差异最容易发生）。"""
    low = name.lower()
    cand = [f for f in ALL_FIELDS if low in f or f in low]
    if not cand:
        cand = [f for f in ALL_FIELDS
                if re.sub(r"[^a-z]", "", f) == re.sub(r"[^a-z]", "", low)]
    return f"，是不是 `{cand[0]}`？" if cand else ""


def describe(expr: str) -> dict:
    """校验并把用到的字段翻成中文说明（页面/审计用）。不跑数据，纯静态。"""
    v = validate(expr)
    return {"fields": [{"name": f, "label": ALL_FIELDS[f],
                        "group": "量价" if f in PRICE_FIELDS else "财务"} for f in v["fields"]],
            "funcs": v["funcs"]}


class _Ctx:
    """求值上下文：字段 → Series，另带月度分组供截面算子使用。"""

    def __init__(self, panel: pd.DataFrame):
        self.panel = panel
        self.month = panel["month_ts"]
        self._cols: dict[str, pd.Series] = {}

    def col(self, name: str) -> pd.Series:
        s = self._cols.get(name)
        if s is None:
            if name not in self.panel.columns:
                # 字段表里有、但当前面板没这列（如旧源现金流列）⇒ 如实 NaN，不报错
                s = pd.Series(np.nan, index=self.panel.index, dtype="float64")
            else:
                s = pd.to_numeric(self.panel[name], errors="coerce")
            self._cols[name] = s
        return s

    def cs(self, name: str, s: pd.Series) -> pd.Series:
        """按月分组做截面变换。NaN 保持 NaN（缺就是缺，不填 0）。"""
        s = pd.to_numeric(s, errors="coerce")
        g = s.groupby(self.month)
        if name == "cs_rank":
            return g.rank(pct=True)
        mu, sd = g.transform("mean"), g.transform("std")
        return (s - mu) / sd.where(sd > 0)


def _eval(node: ast.AST, ctx: _Ctx) -> pd.Series | float:
    if isinstance(node, ast.Expression):
        return _eval(node.body, ctx)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return ctx.col(node.id)
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, ctx), _eval(node.right, ctx)
        op = node.op
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, ast.Div):
            # 除零 → NaN（不产生 inf，否则后续 rank 会把 inf 排到最前）
            if isinstance(b, pd.Series):
                return pd.Series(a) / b.replace(0, np.nan)
            return pd.Series(a) / b if b != 0 else pd.Series(np.nan, index=ctx.panel.index)
        if isinstance(op, ast.Pow):
            return np.power(a, b)
        if isinstance(op, ast.Mod):
            return pd.Series(a) % b
        raise FactorExprError(f"不支持的运算符 {type(op).__name__}（内部错误）")
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, ctx)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.Compare):
        a, b = _eval(node.left, ctx), _eval(node.comparators[0], ctx)
        op = node.ops[0]
        if isinstance(op, ast.Lt):
            r = a < b
        elif isinstance(op, ast.LtE):
            r = a <= b
        elif isinstance(op, ast.Gt):
            r = a > b
        elif isinstance(op, ast.GtE):
            r = a >= b
        elif isinstance(op, ast.Eq):
            r = a == b
        else:
            r = a != b
        return pd.Series(r, index=ctx.panel.index).astype("float64")
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, ctx) for v in node.values]
        out = pd.Series(vals[0], index=ctx.panel.index).astype("float64")
        for v in vals[1:]:
            o = pd.Series(v, index=ctx.panel.index).astype("float64")
            out = np.minimum(out, o) if isinstance(node.op, ast.And) else np.maximum(out, o)
        return out
    if isinstance(node, ast.Call):
        fn = node.func.id
        args = [_eval(a, ctx) for a in node.args]
        if fn in _UNARY:
            return _UNARY[fn](pd.Series(args[0], index=ctx.panel.index)
                              if not isinstance(args[0], pd.Series) else args[0])
        if fn in _BINARY:
            return _BINARY[fn](args[0], args[1])
        if fn == "clip":
            return np.clip(args[0], args[1], args[2])
        if fn in _CS:
            s = args[0] if isinstance(args[0], pd.Series) else pd.Series(
                args[0], index=ctx.panel.index)
            return ctx.cs(fn, s)
        raise FactorExprError(f"未知函数 {fn}（内部错误）")
    raise FactorExprError(f"不允许的写法：{_node_name(node)}")


def evaluate(expr: str, panel: pd.DataFrame, *, validated: dict | None = None) -> pd.Series:
    """在面板上求值，返回 float Series（inf 归为 NaN）。

    缺列（如旧源现金流）按 NaN 处理——**不是 0**。页面上会显示覆盖率，用户自己
    判断这个因子是不是"数据不够"而不是"没信号"。
    """
    v = validated or validate(expr)
    ctx = _Ctx(panel)
    out = _eval(v["ast"], ctx)
    if not isinstance(out, pd.Series):
        raise FactorExprError("表达式结果不是一个序列（是不是只写了常数？）")
    out = pd.to_numeric(out, errors="coerce")
    return out.replace([math.inf, -math.inf], np.nan)


def field_catalog() -> list[dict]:
    """页面用的字段清单（分组 + 中文名）。"""
    return [{"name": k, "label": lab, "group": g}
            for g, d in FIELD_GROUPS.items() for k, lab in d.items()]
