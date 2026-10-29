"""同一输入 × 多模型：技术面分析对照实验。

为什么不用语料审计：历史报告只有 35 份可归因到 qwen3.7-flash，其中"可自证的算式"
仅 34 条 —— 样本量不足以支撑"某模型更差"的结论，且不同日期、不同股票、不同输入长度
互相污染。**唯一能给出因果结论的做法是同一输入对照**。

实验设计：
  · 输入固定：本地日线（截至指定交易日）+ 生产同款指标计算 + **生产原版提示词**
    （直接调 `DeepSeekClient.technical_analysis`，把 `call_api` 换成捕获器，
      保证提示词与线上逐字一致，而不是我照抄一份可能已经走样的模板）
  · 答案固定：指标值由同一份日线算出，模型的任务只是"正确复述 + 正确比较"
  · 判分固定：① 指标复述是否与真值一致 ② 布林带位置是否自相矛盾
              ③ 引用了输入里没有的数字（编造）
  · 每只股票 × 每个模型一次，温度与线上一致（TEMP_ANALYTICAL）

用法：
    python scripts/bench_tech_models.py                # 跑全部
    python scripts/bench_tech_models.py --dry          # 只构题不调用（验证提示词）
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from datetime import datetime

_logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from data.bars import get_bars  # noqa: E402

OUT_DIR = "logs/bench"
DEFAULT_STOCKS = ("600028", "000651", "600941")
MODELS = ("qwen3.7-flash", "deepseek-v4-flash", "qwen-max")


# ---------------------------------------------------------------- 建题
def _indicators_from_bars(code: str, as_of: str) -> tuple[dict, dict, pd.DataFrame]:
    """用本地日线造与生产同口径的指标字典（`data/stock_data.py:674` 的公式）。"""
    import ta

    df = get_bars(code, as_of=as_of, adjust="qfq")
    if df is None or len(df) < 70:
        raise SystemExit(f"{code} 本地日线不足（{0 if df is None else len(df)} 行）")
    d = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                           "close": "Close", "volume": "Volume"}).copy()
    for c in ("Open", "High", "Low", "Close", "Volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Close", "High", "Low"])

    d["MA5"] = ta.trend.sma_indicator(d["Close"], window=5)
    d["MA10"] = ta.trend.sma_indicator(d["Close"], window=10)
    d["MA20"] = ta.trend.sma_indicator(d["Close"], window=20)
    d["MA60"] = ta.trend.sma_indicator(d["Close"], window=60)
    d["RSI"] = ta.momentum.rsi(d["Close"], window=14)
    macd = ta.trend.MACD(d["Close"])
    d["MACD"] = macd.macd()
    d["MACD_signal"] = macd.macd_signal()
    d["MACD_histogram"] = macd.macd_diff()
    boll = ta.volatility.BollingerBands(d["Close"])
    d["BB_upper"] = boll.bollinger_hband()
    d["BB_middle"] = boll.bollinger_mavg()
    d["BB_lower"] = boll.bollinger_lband()
    d["K"] = ta.momentum.stoch(d["High"], d["Low"], d["Close"])
    d["D"] = ta.momentum.stoch_signal(d["High"], d["Low"], d["Close"])
    d["Volume_MA5"] = ta.trend.sma_indicator(d["Volume"], window=5)
    d["Volume_ratio"] = d["Volume"] / d["Volume_MA5"]
    d["ATR14"] = ta.volatility.AverageTrueRange(
        d["High"], d["Low"], d["Close"], window=14).average_true_range()

    lt = d.iloc[-1]
    prev = d.iloc[-2] if len(d) > 1 else lt
    ind = {
        "price": float(lt["Close"]),
        "ma5": float(lt["MA5"]), "ma10": float(lt["MA10"]),
        "ma20": float(lt["MA20"]), "ma60": float(lt["MA60"]),
        "rsi": float(lt["RSI"]),
        "macd": float(lt["MACD"]), "macd_signal": float(lt["MACD_signal"]),
        "macd_histogram": float(lt["MACD_histogram"]),
        "bb_upper": float(lt["BB_upper"]), "bb_middle": float(lt["BB_middle"]),
        "bb_lower": float(lt["BB_lower"]),
        "k_value": float(lt["K"]), "d_value": float(lt["D"]),
        "volume": float(lt["Volume"]),
        "volume_ma5": float(lt["Volume_MA5"]),
        "volume_ma20": float(d["Volume"].tail(20).mean()),
        "volume_ratio": float(lt["Volume_ratio"]),
        "atr14": float(lt["ATR14"]),
    }
    info = {
        "symbol": code, "name": code,
        "current_price": round(float(lt["Close"]), 2),
        "change_percent": round((float(lt["Close"]) / float(prev["Close"]) - 1) * 100, 2),
    }
    return ind, info, d


def build_prompt(code: str, as_of: str) -> tuple[list[dict], dict, dict]:
    """返回 (messages, indicators, info)；messages 是**生产原版提示词**。"""
    from llm.deepseek_client import DeepSeekClient

    ind, info, d = _indicators_from_bars(code, as_of)
    captured: dict = {}

    class _NoCall(DeepSeekClient):
        def __init__(self):                      # noqa: D107  — 跳过网络/密钥初始化
            self.model = "capture"
            self.client = None

        def call_api(self, messages, **kw):      # noqa: D102
            captured["messages"] = messages
            captured["kwargs"] = kw
            return ""

    kline = d.rename(columns={"Close": "收盘", "Volume": "成交量"})
    if "date" in kline.columns:
        kline = kline[["date", "收盘", "成交量"]]
    _NoCall().technical_analysis(info, kline, ind,
                                 analysis_date=as_of, data_through=as_of)
    return captured["messages"], ind, info


# ---------------------------------------------------------------- 调用
def _route(model: str):
    from config.settings import config
    if model.lower().startswith("deepseek"):
        return config.DEEPSEEK_API_KEY, config.DEEPSEEK_BASE_URL
    return config.QWEN_API_KEY, config.QWEN_BASE_URL


def _thinking_kwargs(model: str) -> dict:
    """**必须用生产同一个函数**，且显式关思考。

    ⚠️ 第一版这里只给 qwen3 关思考、其余返回 `{}` —— DeepSeek 服务端默认**开启**，
    实测 `deepseek-v4-flash` 三次调用各烧满 2000 输出 token、正文**全为空**
    （推理内容进了 `reasoning_content`，`content` 是空串）。不统一开关，
    "哪个模型更能写"这个对比就变成"谁恰好没在推理上耗尽预算"。
    """
    from llm.deepseek_client import _thinking_kwargs as prod
    return prod(model, thinking=False)


def call_model(model: str, messages: list[dict]) -> dict:
    from openai import OpenAI
    from llm.deepseek_client import TEMP_ANALYTICAL

    key, base = _route(model)
    cli = OpenAI(api_key=key, base_url=base)
    t0 = time.perf_counter()
    try:
        r = cli.chat.completions.create(
            model=model, messages=messages, temperature=TEMP_ANALYTICAL,
            max_tokens=2000, **_thinking_kwargs(model))
        text = (r.choices[0].message.content or "")
        u = r.usage
        u_out = getattr(u, "completion_tokens", 0) or 0
        # 推理内容单独记：正文为空但 output 打满，说明预算全花在思考上
        reason = getattr(r.choices[0].message, "reasoning_content", None) or ""
        return {"ok": True, "text": text,
                "elapsed_s": round(time.perf_counter() - t0, 2),
                "in": getattr(u, "prompt_tokens", 0) or 0,
                "out": u_out, "reason_chars": len(str(reason)),
                "truncated": u_out >= 1999 and not text.strip()}
    except Exception as e:                       # noqa: BLE001
        # 失败已进返回值，但"哪个模型/哪只票失败"只在日志里能串起来
        _logger.debug("[对照] 调用失败（另见返回值 error）: %s: %s", type(e).__name__, e)
        return {"ok": False, "text": "", "error": f"{type(e).__name__}: {str(e)[:200]}",
                "elapsed_s": round(time.perf_counter() - t0, 2), "in": 0, "out": 0}


# ---------------------------------------------------------------- 判分
_PAT = {
    "ma5": r"MA\s*5[^0-9\n]{0,10}(\d+(?:\.\d+)?)",
    "ma10": r"MA\s*10[^0-9\n]{0,10}(\d+(?:\.\d+)?)",
    "ma20": r"MA\s*20[^0-9\n]{0,10}(\d+(?:\.\d+)?)",
    "ma60": r"MA\s*60[^0-9\n]{0,10}(\d+(?:\.\d+)?)",
    "bb_upper": r"(?:布林带上轨|布林上轨|上轨)[^0-9\n]{0,12}(\d+(?:\.\d+)?)",
    "bb_middle": r"(?:布林带中轨|布林中轨|中轨)[^0-9\n]{0,12}(\d+(?:\.\d+)?)",
    "bb_lower": r"(?:布林带下轨|布林下轨|下轨)[^0-9\n]{0,12}(\d+(?:\.\d+)?)",
    "rsi": r"RSI[^0-9\n]{0,8}(\d+(?:\.\d+)?)",
}


_NOISE = ("差距", "超过", "大于", "小于", "以上", "以下", "附近", "若", "如果",
          "跌破", "突破", "接近", "偏离", "上方", "下方", "之上", "之下")


def _candidates(pat: str, text: str):
    """标签后紧邻的候选数字，剔除四类"不是指标值"的情况。

    ⚠️ 第一版直接取第一个匹配，制造了 6 条假错误：
      `MA5 > MA10 > MA20 > MA60`  → 读成 MA5=10、MA10=20、MA20=60
      `MA10（+1.23%）`            → 读成 MA10=1.23（那是偏离度）
      `RSI 超过 70 即视为超买`      → 读成 RSI=70（那是阈值）
      `上轨与中轨差距约为 1.19元`   → 读成下轨=1.19（那是带宽）
    指标值必须是**裸数**：不带正负号、不带百分号、不是 MA{nn} 的一部分、
    与标签之间没有比较/阈值/差值类词。
    """
    for m in re.finditer(pat, text):
        num = m.group(1)
        start = m.start(1)
        before = text[max(0, start - 1):start]
        after = text[m.end(1):m.end(1) + 1]
        if before in "+-−" or after == "%":
            continue
        if re.search(r"MA\s*$", text[max(0, start - 3):start]):
            continue
        gap = text[m.end(0) - len(num) - 16:m.end(0) - len(num)]
        if any(w in gap for w in _NOISE):
            continue
        yield float(num)


def check_indicators(text: str, ind: dict) -> dict:
    """逐个指标取报告里"紧跟标签的裸数"，与真值比对；未提及记为"未复述"。"""
    hits, total, mentioned = [], 0, 0
    for key, pat in _PAT.items():
        truth = ind.get(key)
        if truth is None:
            continue
        got = next(_candidates(pat, text), None)
        if got is None:
            continue
        mentioned += 1
        total += 1
        tol = max(0.02, abs(truth) * 0.002)
        hits.append({"key": key, "got": got, "truth": round(truth, 4),
                     "ok": abs(got - truth) <= tol})
    return {"n": total, "ok": sum(1 for h in hits if h["ok"]), "detail": hits,
            "restated": mentioned}


def check_boll_position(text: str, ind: dict) -> list[str]:
    """布林带位置断言：把 close 与三条带比对，判定报告里的"之上/之下"是否成立。"""
    close, up, mid, low = ind["price"], ind["bb_upper"], ind["bb_middle"], ind["bb_lower"]
    band = {"上轨": up, "中轨": mid, "下轨": low}
    bad = []
    for sent in re.split(r"[。；;\n]", text):
        for name, val in band.items():
            if name not in sent:
                continue
            if re.search(rf"{name}[^。；;，,]{{0,14}}(?:之上|上方|之上方)", sent):
                if not close > val:
                    bad.append(f"称「{name}之上」但 {close:.2f} ≤ {val:.2f}｜{sent.strip()[:50]}")
            if re.search(rf"{name}[^。；;，,]{{0,14}}(?:之下|下方)", sent):
                if not close < val:
                    bad.append(f"称「{name}之下」但 {close:.2f} ≥ {val:.2f}｜{sent.strip()[:50]}")
    return bad


def check_untraceable(text: str, ind: dict, prompt: str) -> list[str]:
    """报告里出现、但输入里查不到、也不是输入中两数简单换算的数字。

    只抓 ≥2 位小数的数（整数多为"20 根 K 线""2026 年"这类计数/日期，噪声太大）。
    """
    allowed = set()
    for v in ind.values():
        if isinstance(v, (int, float)):
            allowed.add(round(float(v), 2))
    for m in re.finditer(r"\d+(?:\.\d+)?", prompt):
        try:
            allowed.add(round(float(m.group(0)), 2))
        except ValueError as e:
            # 正则抓出来的数字理论上都能转 float；转不了说明正则退化，值得查
            _logger.debug("[对照] 提示词里的数字无法解析: %r (%s)", m.group(0), e)
            pass
    out = []
    for m in re.finditer(r"(?<![\d.])(\d{1,6}\.\d{2,4})(?![\d.])", text):
        v = round(float(m.group(1)), 2)
        if any(abs(v - a) <= max(0.02, a * 0.004) for a in allowed):
            continue
        # 百分比/比值不在此判据内（模型有权做派生换算）
        tail = text[m.end():m.end() + 3]
        if tail.startswith("%") or "百分" in tail:
            continue
        out.append(v)
    return out


def score(text: str, ind: dict, prompt: str) -> dict:
    ic = check_indicators(text, ind)
    boll = check_boll_position(text, ind)
    untrace = check_untraceable(text, ind, prompt)
    return {"ind_total": ic["n"], "ind_ok": ic["ok"], "ind_detail": ic["detail"],
            "restated": ic["restated"],
            "boll_conflicts": boll, "untraceable": untrace,
            "chars": len(text)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--stocks", default=",".join(DEFAULT_STOCKS))
    ap.add_argument("--as-of", default="2026-10-16")
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    stocks = [s.strip() for s in args.stocks.split(",") if s.strip()]
    models = [s.strip() for s in args.models.split(",") if s.strip()]

    report = {"as_of": args.as_of, "cases": [], "summary": {}}
    agg = {m: {"n": 0, "ind_total": 0, "ind_ok": 0, "restated": 0, "boll": 0,
               "untrace": 0, "in": 0, "out": 0, "elapsed": 0.0, "fail": 0,
               "empty": 0} for m in models}

    for code in stocks:
        messages, ind, info = build_prompt(code, args.as_of)
        prompt = messages[-1]["content"]
        if args.dry:
            print(f"=== {code} 提示词 {len(prompt)} 字，指标 {len(ind)} 项 ===")
            print(prompt[:700])
            continue
        for model in models:
            res = call_model(model, messages)
            rec = {"code": code, "model": model, "prompt_chars": len(prompt),
                   "truth": {k: round(v, 4) if isinstance(v, float) else v for k, v in ind.items()}}
            rec.update({k: v for k, v in res.items() if k != "text"})
            if res["ok"]:
                rec["score"] = score(res["text"], ind, prompt)
                fn = f"{OUT_DIR}/{code}_{model.replace('/', '_')}.md"
                io.open(fn, "w", encoding="utf-8").write(res["text"])
                rec["file"] = fn
                agg[model]["n"] += 1
                agg[model]["ind_total"] += rec["score"]["ind_total"]
                agg[model]["ind_ok"] += rec["score"]["ind_ok"]
                agg[model]["restated"] += rec["score"]["restated"]
                agg[model]["boll"] += len(rec["score"]["boll_conflicts"])
                agg[model]["untrace"] += len(rec["score"]["untraceable"])
                agg[model]["in"] += res["in"]
                agg[model]["out"] += res["out"]
                agg[model]["elapsed"] += res["elapsed_s"]
                if rec.get("truncated"):
                    agg[model]["empty"] += 1
            else:
                agg[model]["fail"] += 1
            report["cases"].append(rec)
            st = rec.get("score", {})
            print(f"{code} {model:20} {rec['elapsed_s']:>6}s in={rec['in']:>5} out={rec['out']:>5} "
                  f"复述 {st.get('restated',0)}/8 "
                  f"准确 {st.get('ind_ok',0)}/{st.get('ind_total',0)} "
                  f"布林矛盾 {len(st.get('boll_conflicts',[]))} "
                  f"无出处 {len(st.get('untraceable',[]))}"
                  + ("  ⚠正文为空" if rec.get("truncated") else "")
                  + ("" if res["ok"] else f"  FAIL {res.get('error','')[:60]}"))

    report["summary"] = agg
    io.open(f"{OUT_DIR}/report.json", "w", encoding="utf-8").write(
        json.dumps(report, ensure_ascii=False, indent=2))
    if not args.dry:
        print("\n" + "=" * 78)
        print(f"{'模型':<20}{'成功':>5}{'复述指标':>9}{'复述准确率':>14}{'布林矛盾':>10}{'无出处':>8}{'平均耗时':>10}")
        for m in models:
            a = agg[m]
            acc = f"{a['ind_ok']}/{a['ind_total']}" if a["ind_total"] else "-"
            pct = f" ({a['ind_ok']/a['ind_total']*100:.0f}%)" if a["ind_total"] else ""
            avg = a["elapsed"] / a["n"] if a["n"] else 0
            print(f"{m:<20}{a['n']:>5}{a['restated']:>9}{acc + pct:>14}{a['boll']:>10}"
                  f"{a['untrace']:>8}{avg:>9.1f}s")
        print(f"\n明细报告: {OUT_DIR}/report.json")


if __name__ == "__main__":
    main()
