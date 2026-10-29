# -*- coding: utf-8 -*-
"""A/B 对照：同一输入、同一模型，只切换【派生指标表】。

对照组怎么做到"只差这一段"：不复制提示词，而是给 `fundamental_analysis` 加了一个
运行时开关 `BIGA_DERIVED_FIN`（0/1），两组都走**生产同一条代码路径** ——
否则"脚本里手抄一份 prompt"必然与线上漂移，测出来的差异不可信。

判据（都在本地算，不靠模型自评）：
  · `audit_arithmetic` 告警数 —— 括号算式与结果不匹配的条数
  · `audit_numbers` 告警数    —— 数字在输入里找不到出处的条数
  · 正文长度 / 是否疑似截断     —— 防止"少写就少错"的假优势

⚠️ 小样本：默认每格 2 轮。结论只能作**方向性**证据，不能拿来报"降低了 X%"。

用法：
    python scripts/bench_derived_fin.py
    python scripts/bench_derived_fin.py --rounds 3 --codes 600941
"""
import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent_tools import tool_financials                     # noqa: E402
from core.report_audit import audit_arithmetic, audit_numbers    # noqa: E402
from llm.deepseek_client import DeepSeekClient                   # noqa: E402

DB = os.path.join("data", "portfolio.db")


def _last_analysis(code: str) -> dict:
    """从库里取这只股票最近一次分析的**真实输入**（价格/日期），不另造数据。"""
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        r = con.execute(
            "select stock_code, stock_name, price_at_analysis, analysis_date, payload "
            "from analyses where stock_code=? order by analysis_date desc limit 1",
            (code,)).fetchone()
    finally:
        con.close()
    if not r:
        return {}
    payload = json.loads(r["payload"] or "{}")
    return {"symbol": r["stock_code"], "name": r["stock_name"],
            "current_price": r["price_at_analysis"],
            "analysis_date": r["analysis_date"],
            "data_through": payload.get("data_through") or r["analysis_date"]}


def _run_one(client, stock_info: dict, fin: dict) -> dict:
    text = client.fundamental_analysis(
        {k: v for k, v in stock_info.items() if k not in ("analysis_date", "data_through")},
        financial_data=fin,
        analysis_date=stock_info.get("analysis_date"),
        data_through=stock_info.get("data_through"))
    text = text or ""
    allowed = json.dumps(fin, ensure_ascii=False, default=str)
    arith = audit_arithmetic(text)
    nums = audit_numbers(text, allowed)
    # 粗判截断：结尾既没有句末标点，也没有 markdown 强调符收尾。
    # ⚠️ 首版漏了 `*` / `_`，把 `*注：…不评估。*` 这种**完整**的结尾判成截断，
    #    表格里出现两个假阳性 —— 交付物宁可少一列，也不能给假结论。
    tail = text.rstrip()[-30:] if text.strip() else ""
    truncated = bool(tail) and not re.search(r"[。！？；:：)）】\]】*_`]$", tail)
    return {"len": len(text), "arith": arith, "nums": nums,
            "truncated": truncated, "tail": tail, "text": text}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="600941,300760")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    client = DeepSeekClient()
    rows, detail = [], {}
    for code in codes:
        info = _last_analysis(code)
        if not info:
            print(f"[跳过] {code}: 库里没有分析记录")
            continue
        fin = tool_financials(symbol=code, periods=4)
        if fin.get("error"):
            print(f"[跳过] {code}: 财务数据不可用 {fin['error']}")
            continue
        detail[code] = {"stock_info": info, "rounds": {}}
        for flag in ("0", "1"):
            os.environ["BIGA_DERIVED_FIN"] = flag
            got = []
            for i in range(max(1, args.rounds)):
                print(f"[跑] {code} 派生表={'关' if flag == '0' else '开'} 第{i + 1}轮 ...",
                      flush=True)
                got.append(_run_one(client, info, fin))
            detail[code]["rounds"][flag] = got
            rows.append({
                "code": code, "flag": flag,
                "n": len(got),
                "arith": sum(len(g["arith"]) for g in got),
                "nums": sum(len(g["nums"]) for g in got),
                "avg_len": int(sum(g["len"] for g in got) / len(got)),
                "trunc": sum(1 for g in got if g["truncated"]),
            })
    os.environ.pop("BIGA_DERIVED_FIN", None)

    lines = ["# 派生指标表 A/B 对照（同一输入 × 同一模型）", "",
             f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}　"
             f"每格 {args.rounds} 轮　小样本，仅作方向性证据", "",
             "| 代码 | 派生表 | 轮数 | 算式错误 | 无出处数字 | 平均长度 | 疑似截断 |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['code']} | {'关' if r['flag'] == '0' else '**开**'} | "
                     f"{r['n']} | {r['arith']} | {r['nums']} | {r['avg_len']} | {r['trunc']} |")
    off = sum(r["arith"] for r in rows if r["flag"] == "0")
    on = sum(r["arith"] for r in rows if r["flag"] == "1")
    lines += ["", f"**合计算式错误：关 = {off}，开 = {on}**"]

    # 告警明细（原文片段）—— 只看计数会掩盖"错在哪一类"
    lines += ["", "## 告警明细", ""]
    for code, d in detail.items():
        for flag in ("0", "1"):
            for i, g in enumerate(d["rounds"].get(flag, []), start=1):
                for w in g["arith"]:
                    lines.append(f"- `{code}` 派生表{'关' if flag == '0' else '开'} "
                                 f"第{i}轮 · {w}")
    md = "\n".join(lines) + "\n"
    out = args.out or os.path.join("output", f"派生指标表AB对照_{datetime.now():%Y%m%d_%H%M}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    # 原文单独落一份，便于人工复核"它到底抄没抄"
    raw = os.path.splitext(out)[0] + "_原文.json"
    with open(raw, "w", encoding="utf-8") as f:
        json.dump({c: {"stock_info": d["stock_info"],
                       "rounds": {k: [{"text": g["text"], "arith": g["arith"]} for g in v]
                                  for k, v in d["rounds"].items()}}
                   for c, d in detail.items()}, f, ensure_ascii=False, indent=2)
    print(md)
    print(f"已写入 {out}\n原文 {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
