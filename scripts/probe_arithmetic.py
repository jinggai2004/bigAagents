"""针对性算术测试：同一份累计财报，三个模型各自推单季值。

为什么单独做这个：技术面对照实验（`bench_tech_models.py`）证明**复述数字不是差别**——
三个模型都是 100%。历史语料里 qwen3.7-flash 的 4 处错误全都出在
"由累计值相减推单季值"和"两个数相除得比率"上，所以直接把这两种动作拆出来测。

输入是 600941 中国移动 2026 年三季报的**真实累计值**（营收/净利/经营现金流 三组），
真值由脚本自己算，模型只负责"照算式复述"。这样判分不依赖任何外部判断。
"""
from __future__ import annotations

import json
import logging
import re
import time

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import OpenAI  # noqa: E402

from config.settings import config  # noqa: E402
from llm.deepseek_client import _thinking_kwargs  # noqa: E402

# 2026 年三季报累计值（单位：亿元）—— 与 600941 报告中的数据一致
CUM = {
    "营业收入": {"Q1": 2664.78, "H1": 5380.35, "9M": 7802.95},
    "归母净利润": {"Q1": 293.54, "H1": 790.10, "9M": 1063.25},
    "经营活动现金流量净额": {"Q1": 714.47, "H1": 1148.54, "9M": 1852.90},
}
FY = {"营业收入": 11032.0, "归母净利润": 1288.0}          # 2025 全年，用于同比

PROMPT = f"""下面是某公司 2026 年**累计口径**财报数据（单位：亿元）：

营业收入：一季度 2664.78，上半年 5380.35，前三季度 7802.95
归母净利润：一季度 293.54，上半年 790.10，前三季度 1063.25
经营活动现金流量净额：一季度 714.47，上半年 1148.54，前三季度 1852.90

2025 年全年：营业收入 11032.0，归母净利润 1288.0

请计算并逐条列出（**只输出数字与算式，不要分析、不要给投资建议**）：
1. Q2 单季营业收入、Q3 单季营业收入
2. Q2 单季归母净利润、Q3 单季归母净利润
3. Q2 单季经营活动现金流量净额、Q3 单季经营活动现金流量净额
4. 前三季度归母净利率（= 前三季度归母净利润 / 前三季度营业收入）
5. 前三季度营业收入同比增速（= 前三季度营业收入 / 2025全年营业收入 - 1）

格式：每行 `名称: 值亿元（算式）`。数值保留两位小数。"""

# 真值
T = {
    "Q2营收": round(CUM["营业收入"]["H1"] - CUM["营业收入"]["Q1"], 2),
    "Q3营收": round(CUM["营业收入"]["9M"] - CUM["营业收入"]["H1"], 2),
    "Q2净利": round(CUM["归母净利润"]["H1"] - CUM["归母净利润"]["Q1"], 2),
    "Q3净利": round(CUM["归母净利润"]["9M"] - CUM["归母净利润"]["H1"], 2),
    "Q2现金流": round(CUM["经营活动现金流量净额"]["H1"] - CUM["经营活动现金流量净额"]["Q1"], 2),
    "Q3现金流": round(CUM["经营活动现金流量净额"]["9M"] - CUM["经营活动现金流量净额"]["H1"], 2),
    "净利率": round(CUM["归母净利润"]["9M"] / CUM["营业收入"]["9M"] * 100, 2),
    "营收同比": round((CUM["营业收入"]["9M"] / FY["营业收入"] - 1) * 100, 2),
}

# 文本里找数值的锚点（顺序敏感：长锚点在前）
ANCHOR = [
    ("Q2营收", [r"Q2[^\n]{0,12}营业收入", r"Q2[^\n]{0,10}营收"]),
    ("Q3营收", [r"Q3[^\n]{0,12}营业收入", r"Q3[^\n]{0,10}营收"]),
    ("Q2净利", [r"Q2[^\n]{0,12}归母净利润", r"Q2[^\n]{0,10}净利润"]),
    ("Q3净利", [r"Q3[^\n]{0,12}归母净利润", r"Q3[^\n]{0,10}净利润"]),
    ("Q2现金流", [r"Q2[^\n]{0,16}现金流量净额"]),
    ("Q3现金流", [r"Q3[^\n]{0,16}现金流量净额"]),
    ("净利率", [r"净利率"]),
    ("营收同比", [r"营业收入同比", r"营收同比"]),
]


def parse(text: str) -> dict:
    out = {}
    for key, pats in ANCHOR:
        for pat in pats:
            m = re.search(pat + r"[^\n]{0,20}?(-?\d+(?:\.\d+)?)", text)
            if m:
                out[key] = float(m.group(1))
                break
    return out


def main() -> None:
    models = ["qwen3.7-flash", "deepseek-v4-flash", "qwen-max"]
    print("真值：", {k: v for k, v in T.items()})
    print("-" * 78)
    summary = {}
    for model in models:
        key, base = ((config.DEEPSEEK_API_KEY, config.DEEPSEEK_BASE_URL)
                     if model.startswith("deepseek")
                     else (config.QWEN_API_KEY, config.QWEN_BASE_URL))
        cli = OpenAI(api_key=key, base_url=base)
        t0 = time.perf_counter()
        try:
            r = cli.chat.completions.create(
                model=model, messages=[{"role": "user", "content": PROMPT}],
                temperature=0.2, max_tokens=900, **_thinking_kwargs(model, thinking=False))
            text = r.choices[0].message.content or ""
            el = time.perf_counter() - t0
        except Exception as e:                    # noqa: BLE001
            # 不静默：模型被限频/端点变更时，不记日志会让它**从结果表里无声消失**，
            # 看起来像"该模型没被测到"，实际是调用失败。
            print(f"{model}: FAIL {type(e).__name__} {str(e)[:120]}")
            logging.getLogger(__name__).warning(
                "[算术测试] %s 调用失败: %s: %s", model, type(e).__name__, e)
            continue
        got = parse(text)
        ok = [k for k in T if k in got and abs(got[k] - T[k]) <= max(0.02, abs(T[k]) * 0.005)]
        miss = [k for k in T if k not in got]
        wrong = [k for k in T if k in got and k not in ok]
        print(f"\n{model}  {el:.1f}s  in={r.usage.prompt_tokens} out={r.usage.completion_tokens}")
        for k in T:
            mark = "✓" if k in ok else ("✗" if k in wrong else "—")
            print(f"   {mark} {k:8} 真值 {T[k]:>10}  报告 {got.get(k, '未解析'):>10}")
        summary[model] = {"ok": len(ok), "wrong": len(wrong), "miss": len(miss),
                          "wrong_keys": wrong, "elapsed": round(el, 1),
                          "raw": text}
    os.makedirs("logs/bench", exist_ok=True)
    io_path = "logs/bench/arithmetic_probe.json"
    with open(io_path, "w", encoding="utf-8") as f:
        json.dump({"truth": T, "summary": summary}, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 78)
    for m, s in summary.items():
        print(f"{m:<20} 正确 {s['ok']}/{len(T)}  算错 {s['wrong']}  未答 {s['miss']}  "
              f"{s['elapsed']}s   {s['wrong_keys']}")
    print(f"明细: {io_path}")


if __name__ == "__main__":
    main()
