"""对账：把报告里的"算式"逐条重算。

只查**自证式**断言——即同一个句子里既有算式又有结果，不依赖任何外部数据：
  1) `2,422.6亿元（7,802.95 - 5,380.35）`  → 括号内算式必须等于括号外的数
  2) `13.63%（1,063.25 / 7,802.95）`       → 百分比必须等于除法的结果
  3) `下降1.81%` + 两个数                  → 同比必须对得上

不查"目标价是否合理""趋势判断对不对"这类需要行情/常识的判断，
因为那会把"观点分歧"和"事实错误"混在一起。
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import sys

_logger = logging.getLogger(__name__)

# ---------- 数字解析 ----------
_NUM = r"\d[\d,]*(?:\.\d+)?"


def _f(s: str) -> float:
    return float(s.replace(",", ""))


# ---------- 规则 1：括号内的减法 ----------
# 形如  2,422.6亿元（7,802.95 - 5,380.35）   或   273.15（1,063.25-790.1）
_SUB = re.compile(
    rf"({_NUM})\s*(?:亿元|亿|万元|万|元|%|倍|个百分点)?\s*[（(]\s*"
    rf"({_NUM})\s*[-−–]\s*({_NUM})\s*[)）]"
)

# ---------- 规则 2：括号内的除法（百分比） ----------
_DIV = re.compile(
    rf"({_NUM})\s*%\s*[（(][^（()）]{{0,40}}?"
    rf"({_NUM})\s*/\s*({_NUM})"
)

# ---------- 规则 3：括号内加法 ----------
_ADD = re.compile(
    rf"({_NUM})\s*(?:亿元|亿|万元|万|元|%|倍)?\s*[（(]\s*"
    rf"({_NUM})\s*\+\s*({_NUM})\s*[)）]"
)


def _is_base_list(inner: str) -> bool:
    """括号里是"一组基准值"而不是"一个算式"。

    实测误报：`归母净利润同比+4.04%（基准为2025-06-30的1,333.68/464.55/460.16亿元）`
    —— 括号内是三个基准数的并列，不是除法。判据：出现 ≥2 个运算符即视为并列。
    """
    ops = re.findall(r"[-−–+/]", inner)
    return len(ops) >= 2


def _is_date_expr(inner: str) -> bool:
    """含 `2023.10` / `2025-06-30` 这类日期分量 ⇒ 不是算式。"""
    return bool(re.search(r"\d{4}[.\-/]\d{1,2}\b", inner))


def _is_self_derivation(inner: str) -> bool:
    """含 `≈` 或 `=` ⇒ 模型自己把推导过程写出来了，不按"结果必须相等"判。"""
    return ("≈" in inner) or ("=" in inner)


def sub_errors(text: str, tol: float = 0.02, rel: float = 0.005) -> list[dict]:
    """括号内 `a - b` 的结果必须等于括号外的数（相对误差 <0.5% 视为排版误差）。"""
    out = []
    for m in _SUB.finditer(text):
        inner = m.group(0)
        if _is_base_list(inner):
            continue
        if _is_date_expr(inner) or _is_self_derivation(inner):
            continue
        got, a, b = _f(m.group(1)), _f(m.group(2)), _f(m.group(3))
        want = a - b
        if abs(want) < 1e-9:
            continue
        # ⚠️ 负号在括号外（`-1.07亿元（0.18 - 1.25）`）：正则只捕到 `1.07`，
        #    必须按符号还原，否则把**算对**的写成算错。
        before = text[max(0, m.start(1) - 1):m.start(1)]
        signed = -got if before in "-−" else got
        if abs(signed - want) <= max(tol, abs(want) * rel):
            continue
        out.append({"kind": "sub", "claim": signed, "calc": round(want, 4),
                    "expr": f"{m.group(2)} - {m.group(3)}", "raw": m.group(0)[:60]})
    return out


def div_errors(text: str, tol: float = 0.15) -> list[dict]:
    """括号内 `a / b` 的结果必须等于 `x%`（容忍 0.15 个百分点）。"""
    out = []
    for m in _DIV.finditer(text):
        inner = m.group(0)
        if _is_base_list(inner) or _is_date_expr(inner) or _is_self_derivation(inner):
            continue
        got, a, b = _f(m.group(1)), _f(m.group(2)), _f(m.group(3))
        if abs(b) < 1e-9:
            continue
        want = a / b * 100
        if abs(got - want) <= tol:
            continue
        out.append({"kind": "div", "claim": got, "calc": round(want, 4),
                    "expr": f"{m.group(2)} / {m.group(3)}", "raw": m.group(0)[:60]})
    return out


def add_errors(text: str, tol: float = 0.02, rel: float = 0.005) -> list[dict]:
    out = []
    for m in _ADD.finditer(text):
        inner = m.group(0)
        if _is_base_list(inner) or _is_date_expr(inner) or _is_self_derivation(inner):
            continue
        got, a, b = _f(m.group(1)), _f(m.group(2)), _f(m.group(3))
        want = a + b
        if abs(want) < 1e-9:
            continue
        if abs(got - want) <= max(tol, abs(want) * rel):
            continue
        out.append({"kind": "add", "claim": got, "calc": round(want, 4),
                    "expr": f"{m.group(2)} + {m.group(3)}", "raw": m.group(0)[:60]})
    return out


def audit(text: str) -> list[dict]:
    if not text:
        return []
    return sub_errors(text) + div_errors(text) + add_errors(text)


def n_claims(text: str) -> int:
    """可自证的断言条数（分母）。"""
    if not text:
        return 0
    return len(list(_SUB.finditer(text))) + len(list(_DIV.finditer(text))) + len(list(_ADD.finditer(text)))


def boll_contradictions(text: str) -> list[dict]:
    """同一句里"收盘价 vs 布林带三轨"的方位断言必须成立。

    实测样本（qwen3.7-flash，600941 / 2026-10-23）：
      「收盘价 99.14 位于布林带中轨（97.66）之上，下轨（91.83）之下，上轨（103.49）之下」
    99.14 显然高于下轨 91.83，`下轨之下` 自相矛盾。这类错误**不需要外部数据**，
    句子自身就构成反例 —— 所以可以全量自动判。
    """
    out = []
    band_re = re.compile(r"(上轨|中轨|下轨)[^\d（(]{0,8}[（(]?\s*(\d+(?:\.\d+)?)")
    close_re = re.compile(r"(?:收盘价|现价|股价|当前价)[^\d]{0,4}(\d+(?:\.\d+)?)")
    for sent in re.split(r"[。；;\n]", text):
        cm = close_re.search(sent)
        if not cm:
            continue
        close = _f(cm.group(1))
        for bm in band_re.finditer(sent):
            name, val = bm.group(1), _f(bm.group(2))
            tail = sent[bm.end():bm.end() + 12]
            if re.search(r"(?:之上|上方)", tail) and close <= val:
                out.append({"band": name, "close": close, "value": val,
                            "said": "之上", "sent": sent.strip()[:60]})
            elif re.search(r"(?:之下|下方)", tail) and close >= val:
                out.append({"band": name, "close": close, "value": val,
                            "said": "之下", "sent": sent.strip()[:60]})
    return out


# 各模型实际负责的段落
SECTIONS_QWEN37 = ("tech_report", "fundamental_report", "news_report", "bull_history", "bear_history")
SECTIONS_DECISION = ("research_summary", "risk_assessment", "final_decision")


def load_rows(db="data/portfolio.db"):
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    return list(c.execute("SELECT id, analysis_date, stock_code, stock_name, model, payload "
                          "FROM analyses ORDER BY analysis_date, id"))


def run(db="data/portfolio.db"):
    rows = load_rows(db)
    per_model: dict[str, dict] = {}
    samples: list[dict] = []
    for r in rows:
        try:
            p = json.loads(r["payload"] or "{}")
        except json.JSONDecodeError as e:
            # payload 坏了要能查：整行被跳过会让"样本数变少"看不出原因
            _logger.debug("[对账] payload 非 JSON，跳过该行: %s", e)
            continue
        model = str(r["model"] or "?")
        # 归因：mixed 里写作段落归 qwen3.7-flash，决策段落归 deepseek-v4-flash
        if model.startswith("mixed:"):
            writer = "qwen3.7-flash"
            secs = SECTIONS_QWEN37
        else:
            writer = model
            secs = tuple(k for k in p if isinstance(p.get(k), str))
        text = "\n".join(p.get(k, "") if isinstance(p.get(k), str) else "" for k in secs)
        if not text.strip():
            continue
        errs = audit(text)
        bolls = boll_contradictions(text) if "tech_report" in secs else []
        d = per_model.setdefault(writer, {"reports": 0, "claims": 0, "errors": 0,
                                          "boll": 0, "boll_reports": 0})
        d["reports"] += 1
        d["claims"] += n_claims(text)
        d["errors"] += len(errs)
        d["boll"] += len(bolls)
        if bolls:
            d["boll_reports"] += 1
        for b in bolls:
            samples.append({**b, "kind": "boll", "raw": b["sent"],
                            "date": str(r["analysis_date"]), "code": str(r["stock_code"]),
                            "name": str(r["stock_name"]), "writer": writer,
                            "claim": f"{b['band']}{b['said']}", "calc": b["value"],
                            "expr": ""})
        for e in errs:
            e.update(date=str(r["analysis_date"]), code=str(r["stock_code"]),
                     name=str(r["stock_name"]), writer=writer)
            samples.append(e)
    return per_model, samples


if __name__ == "__main__":
    per_model, samples = run(sys.argv[1] if len(sys.argv) > 1 else "data/portfolio.db")
    print(f"{'模型(负责段落)':<22}{'报告':>5}{'可核算式':>9}{'算错':>6}{'错误率':>8}{'布林矛盾':>9}{'涉事报告':>9}")
    print("-" * 72)
    for m, d in sorted(per_model.items(), key=lambda x: -x[1]["claims"]):
        rate = d["errors"] / d["claims"] * 100 if d["claims"] else 0.0
        print(f"{m:<22}{d['reports']:>5}{d['claims']:>9}{d['errors']:>6}{rate:>7.1f}%"
              f"{d['boll']:>9}{d['boll_reports']:>9}")
    print(f"\n共 {len(samples)} 处算错，逐条：")
    for e in samples:
        print(f"  {e['date']} {e['code']} {e['name']} [{e['writer']}] "
              f"写成 {e['claim']} 实为 {e['calc']}  ({e['expr']})")
