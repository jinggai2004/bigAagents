# -*- coding: utf-8 -*-
"""概念板块数据源探针：判定「概念 → 成分股」当前能不能取、粒度够不够。

为什么单独写它（2026-10-28，交接报告阶段 B）
-------------------------------------------
`output/交接报告_股票池定性信息与情报面选股_2026-10-28.md` 明确要求
"所有接口函数名与字段以**运行时验证**为准"。**七条**候选路逐个试过，
判定必须可复跑，否则下一个人（或换台机器/换代理后）还得重走一遍。

判定的层级（**层层递进，早失败早返回**，避免"HTTP 200 就算通"）：

    接口不存在 → 网络不可达 → 返回空表 → 字段不全 → 粒度不够 → ok

⚠️ 两个真实踩过的坑，探针必须能复现：
1) **TCP 能连通 ≠ 接口可用**。东财 push2 的 4 个子域 TCP 443 全部握手成功，
   但 HTTP 请求被 RST（RemoteDisconnected）——只看 connect() 会误判成"通"。
2) **有名字 ≠ 有成分**。同花顺概念列表 375 个（name+code 都在），
   但 akshare 没有成分接口，详情页翻页要 JS token（401 + chameleon）。

用法
----
    python scripts/probe_concept_source.py            # 快速探测（约 20~30s）
    python scripts/probe_concept_source.py --full     # 额外拉全量东财题材（+75s）
    python scripts/probe_concept_source.py --json     # 机器可读

退出码：0 = 至少一条 ok；2 = 全部不可用（便于 CI / 交接脚本判定）。
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import sys

_logger = logging.getLogger(__name__)

TIMEOUT = 15
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# 粒度判据：情报面选股真正需要的是"铜价涨 → 铜矿股"这种细主题，
# 只有"有色金属"这种大筐等于没解决问题（那正是阶段 A 已有的能力）。
THEME_PROBES = ("铜", "稀土", "黄金", "铝", "煤", "钢")

# 东财数据中心：股票 × 题材板块 全量映射（字段见 _EM_COLS 的实测结果）
EM_DATACENTER = "https://datacenter-web.eastmoney.com/api/data/v1/get"
EM_REPORT = "RPT_F10_CORETHEME_BOARDTYPE"


def _result(source, verdict, detail, **extra):
    """统一结果结构。

    verdict ∈ {ok, thin, no_func, blocked, empty, schema, error}
      ok      = 可用且粒度足够
      thin    = 接口通但粒度不够（有概念、无细分主题）
    """
    return {"source": source, "verdict": verdict, "detail": detail, **extra}


def _get(url, params=None, sess=None, referer=None, timeout=TIMEOUT):
    import requests

    h = {"User-Agent": UA}
    if referer:
        h["Referer"] = referer
    getter = sess.get if sess is not None else requests.get
    return getter(url, params=params, headers=h, timeout=timeout)


# ------------------------------------------------------------------
# ① 东方财富 push2：接口最规整，但本机链路在 TLS 之后被 RST
# ------------------------------------------------------------------
def probe_eastmoney_push2() -> dict:
    import akshare as ak

    if not (hasattr(ak, "stock_board_concept_name_em")
            and hasattr(ak, "stock_board_concept_cons_em")):
        return _result("东财 push2(akshare)", "no_func",
                       "akshare 缺 stock_board_concept_*_em")

    try:
        df = ak.stock_board_concept_name_em()
    except Exception as e:                      # noqa: BLE001 — 探针要如实分类，不中断
        _logger.warning("[探针] 东财 push2 概念列表失败: %s: %s", type(e).__name__, e)
        # 同一台机器上用新浪做对照，证明"不是断网"
        control = "对照(新浪 hq.sinajs.cn) 未测"
        try:
            r = _get("https://hq.sinajs.cn/list=sh601899",
                     referer="https://finance.sina.com.cn")
            control = f"对照(新浪) HTTP {r.status_code}"
        except Exception as e2:                 # noqa: BLE001
            _logger.warning("[探针] 对照源也失败: %s: %s", type(e2).__name__, e2)
            control = f"对照(新浪)也失败: {type(e2).__name__}"
        return _result("东财 push2(akshare)", "blocked",
                       f"{type(e).__name__}: {str(e)[:100]}；{control}"
                       + ("（故非断网，是链路/反爬）" if "200" in control else ""),
                       control=control)
    if df is None or len(df) == 0:
        return _result("东财 push2(akshare)", "empty", "概念列表返回空表")
    return _result("东财 push2(akshare)", "ok", f"概念 {len(df)} 个",
                   concepts=len(df))


# ------------------------------------------------------------------
# ② 东财数据中心：datacenter-web（与 push2 不同域名/链路，本机可用）
# ------------------------------------------------------------------
def probe_eastmoney_datacenter(full: bool = False) -> dict:
    """全量「股票 × 题材板块」映射，反查即"概念 → 成分股"。

    实测（2026-10-28）：99,479 行 / 1,042 个概念 / pageSize=5000 → 20 页 / 74s。
    探针默认只拉第 1 页 + 一次精确过滤，约 5s，足以判定"通不通、粒度够不够"。
    """
    base = {"reportName": EM_REPORT, "columns": "ALL"}
    try:
        r = _get(EM_DATACENTER, {**base, "pageSize": 5000, "pageNumber": 1})
        payload = r.json()
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[探针] 东财数据中心失败: %s: %s", type(e).__name__, e)
        return _result("东财数据中心", "blocked", f"{type(e).__name__}: {str(e)[:100]}")

    res = payload.get("result") or {}
    rows = res.get("data") or []
    if not rows:
        return _result("东财数据中心", "empty",
                       f"HTTP {r.status_code} 但 data 为空（reportName 可能改名）")
    first = rows[0]
    need = {"SECURITY_CODE", "BOARD_NAME"}
    if not need.issubset(first):
        return _result("东财数据中心", "schema",
                       f"缺字段 {sorted(need - set(first))}；实际 {sorted(first)[:8]}")
    if not first.get("NEW_BOARD_CODE"):
        return _result("东财数据中心", "schema", "缺 NEW_BOARD_CODE（BK 代码）")

    # 精确过滤：能按概念名直接取，说明不必每次拉全量
    filter_ok, filter_n = False, 0
    try:
        rf = _get(EM_DATACENTER, {**base, "filter": '(BOARD_NAME="铜")',
                                  "pageSize": 200, "pageNumber": 1})
        rf_res = rf.json().get("result") or {}
        filter_n = rf_res.get("count") or 0
        filter_ok = filter_n > 0
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[探针] 东财数据中心 filter 失败: %s: %s",
                        type(e).__name__, e)

    concepts = {(x.get("BOARD_NAME") or "") for x in rows}
    # ⚠️ 第 1 页只有 68/1041 个概念，**不能**拿它判定"铜/稀土 有没有"——
    #    第一版就这么写了，结果快速模式把「铜」判成缺失（假阴性）。
    #    粒度核对只在 `--full`（拉全量）时做；做不到就如实说"未核对"。
    sampled = {t: sum(1 for c in concepts if t in c) for t in THEME_PROBES}

    extra = {"rows_total": res.get("count"), "page_rows": len(rows),
             "concepts_on_page": len(concepts), "filter_count": filter_n,
             "themes_sampled": sampled}
    missing = None
    if full:
        try:
            allc, page, total_pages = set(), 1, res.get("pages") or 1
            while page <= total_pages:
                rp = _get(EM_DATACENTER, {**base, "pageSize": 5000, "pageNumber": page})
                rp_res = rp.json().get("result") or {}
                allc |= {(x.get("BOARD_NAME") or "") for x in (rp_res.get("data") or [])}
                total_pages = rp_res.get("pages") or total_pages
                page += 1
            extra["concepts_total"] = len(allc)
            extra["themes_total"] = {t: sum(1 for c in allc if t in c) for t in THEME_PROBES}
            missing = [t for t, n in extra["themes_total"].items() if n == 0]
        except Exception as e:                  # noqa: BLE001
            _logger.warning("[探针] 东财数据中心全量失败: %s: %s", type(e).__name__, e)
            extra["full_error"] = f"{type(e).__name__}: {str(e)[:80]}"

    # 接口可用性判据：拿到行 + 字段齐 + **精确过滤真的能落到概念上**
    if not filter_ok:
        return _result("东财数据中心", "schema",
                       f"映射 {res.get('count')} 行，但 filter=(BOARD_NAME=\"铜\") 无结果"
                       "（过滤语法变了，不能按概念名取数）", **extra)
    if missing:
        return _result("东财数据中心", "thin",
                       f"映射 {res.get('count')} 行，但缺主题 {missing}", **extra)

    scope = (f"全量概念 {extra['concepts_total']} 个" if "concepts_total" in extra
             else f"本页概念 {len(concepts)} 个（粒度未核对，加 --full）")
    return _result("东财数据中心", "ok",
                   f"映射 {res.get('count')} 行；{scope}；"
                   f"精确过滤「铜」{filter_n} 只", **extra)


# ------------------------------------------------------------------
# ③ 新浪概念：接口通，但只有 175 个粗概念，无铜/稀土/铝/煤/钢
# ------------------------------------------------------------------
def probe_sina_concept() -> dict:
    import re

    try:
        r = _get("https://vip.stock.finance.sina.com.cn/q/view/newFLJK.php",
                 {"param": "class"},
                 referer="https://vip.stock.finance.sina.com.cn/")
        r.encoding = "gbk"
        m = re.search(r"=\s*(\{.*\})", r.text, re.S)
        if not m:
            return _result("新浪概念", "schema",
                           f"HTTP {r.status_code} 但未匹配 `var ... = {{...}}`，"
                           f"前 60 字: {r.text[:60]!r}")
        table = json.loads(m.group(1))
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[探针] 新浪概念列表失败: %s: %s", type(e).__name__, e)
        return _result("新浪概念", "blocked", f"{type(e).__name__}: {str(e)[:100]}")

    names, nodes = [], {}
    for _k, v in table.items():
        parts = str(v).split(",")
        if len(parts) >= 3:
            names.append(parts[1])
            nodes[parts[1]] = parts[0]

    # 成分接口：用第一个真实 node 验证
    cons_n, probe_node = -1, next(iter(nodes.values()), None)
    if probe_node:
        try:
            rc = _get("https://vip.stock.finance.sina.com.cn/quotes_service/api/"
                      "json_v2.php/Market_Center.getHQNodeData",
                      {"page": 1, "num": 5, "sort": "symbol", "asc": 1,
                       "node": probe_node, "symbol": "", "_s_r_a": "page"},
                      referer="https://vip.stock.finance.sina.com.cn/")
            data = json.loads(rc.text or "[]")
            cons_n = len(data)
        except Exception as e:                  # noqa: BLE001
            _logger.warning("[探针] 新浪概念成分失败: %s: %s", type(e).__name__, e)

    themes = {t: sum(1 for n in names if t in n) for t in THEME_PROBES}
    missing = [t for t, n in themes.items() if n == 0]
    verdict = "thin" if missing or cons_n <= 0 else "ok"
    return _result("新浪概念", verdict,
                   f"概念 {len(names)} 个；成分接口返回 {cons_n} 行；"
                   f"缺主题 {missing or '无'}",
                   concepts=len(names), cons_rows=cons_n, themes=themes)


# ------------------------------------------------------------------
# ④ 同花顺：有概念名（含 code），但 akshare 无成分接口；详情页分页需 JS 反爬
# ------------------------------------------------------------------
def probe_ths() -> dict:
    import akshare as ak

    if not hasattr(ak, "stock_board_concept_name_ths"):
        return _result("同花顺(akshare)", "no_func", "akshare 缺 stock_board_concept_name_ths")
    try:
        names = ak.stock_board_concept_name_ths()
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[探针] 同花顺概念列表失败: %s: %s", type(e).__name__, e)
        return _result("同花顺(akshare)", "blocked", f"{type(e).__name__}: {str(e)[:120]}")
    if names is None or len(names) == 0:
        return _result("同花顺(akshare)", "empty", "概念列表返回空表")

    has_cons = hasattr(ak, "stock_board_concept_cons_ths")
    page_state = "未测"
    try:
        import io

        import pandas as pd
        import requests
        sess = requests.Session()
        code = str(names.iloc[0]["code"])
        r = _get(f"http://q.10jqka.com.cn/gn/detail/code/{code}/", sess=sess,
                 referer="http://q.10jqka.com.cn/gn/")
        tables = pd.read_html(io.StringIO(r.text))
        first = tables[0] if tables else None
        page_state = (f"首页 {r.status_code}，表 {first.shape}，" if first is not None
                      else f"首页 {r.status_code} 无表")
        r2 = _get(f"http://q.10jqka.com.cn/gn/detail/field/264648/order/desc/"
                  f"page/2/ajax/1/code/{code}/", sess=sess,
                  referer="http://q.10jqka.com.cn/gn/")
        page_state += f"翻页 {r2.status_code}"
        if r2.status_code != 200:
            page_state += "（需 JS 反爬 token）"
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[探针] 同花顺详情页失败: %s: %s", type(e).__name__, e)
        page_state = f"详情页失败 {type(e).__name__}: {str(e)[:80]}"

    return _result("同花顺(akshare)", "no_func" if not has_cons else "ok",
                   f"概念 {len(names)} 个（有 name+code）；成分接口 "
                   f"{'有' if has_cons else '缺失'}；详情页 {page_state}",
                   concepts=len(names), has_cons=has_cons)


# ------------------------------------------------------------------
# ⑤ 富途：有概念成分接口，但是**美股**主题，不适用 A 股
# ------------------------------------------------------------------
def probe_futu() -> dict:
    import inspect

    import akshare as ak

    fn = getattr(ak, "stock_concept_cons_futu", None)
    if fn is None:
        return _result("富途(akshare)", "no_func", "akshare 缺 stock_concept_cons_futu")
    doc = (fn.__doc__ or "").replace("\n", " ")
    try:
        sig = str(inspect.signature(fn))
    except (TypeError, ValueError) as e:
        _logger.debug("[探针] 富途接口无签名: %s", e)
        sig = "(无签名)"
    us_market = "美股" in doc or "futunn.com/quote/sparks-us" in doc
    return _result("富途(akshare)", "schema" if us_market else "ok",
                   f"签名 {sig}；{'是美股概念（us 站点），A 股不可用' if us_market else '疑似 A 股可用'}"
                   f"；doc: {doc[:90]}")


# ------------------------------------------------------------------
# ⑥ 雪球：screener 需要登录 token（cookie 只有 acw_tc，无 xq_a_token）
# ------------------------------------------------------------------
def probe_xueqiu() -> dict:
    try:
        import requests
        sess = requests.Session()
        sess.headers.update({"User-Agent": UA})
        sess.get("https://xueqiu.com/", timeout=TIMEOUT)
        r = sess.get("https://stock.xueqiu.com/v5/stock/screener/quote/list.json",
                     params={"page": 1, "size": 5, "order": "desc",
                             "order_by": "percent", "market": "CN", "type": "bk_1"},
                     timeout=TIMEOUT)
    except Exception as e:                      # noqa: BLE001
        _logger.warning("[探针] 雪球失败: %s: %s", type(e).__name__, e)
        return _result("雪球", "blocked", f"{type(e).__name__}: {str(e)[:100]}")
    body = (r.text or "")[:120]
    ok = r.status_code == 200 and '"data"' in body
    return _result("雪球", "ok" if ok else "blocked",
                   f"HTTP {r.status_code}；cookies={sorted(sess.cookies.keys())}；"
                   f"{'' if ok else '需登录 token（xq_a_token）'} {body}")


# ------------------------------------------------------------------
# ⑦ 腾讯：板块排行接口 rpc 名失效 / 返回空列表
# ------------------------------------------------------------------
def probe_tencent() -> dict:
    urls = [
        ("getBoardRank", "https://proxy.finance.qq.com/cgi/cgi-bin/rank/hs/getBoardRank",
         {"board_type": "concept", "sort_type": "price", "direct": "down",
          "offset": 0, "count": 5}),
        ("getRank", "https://proxy.finance.qq.com/cgi/cgi-bin/rank/pt/getRank",
         {"board_type": "concept", "sort_type": "price", "direct": "down",
          "offset": 0, "count": 5}),
    ]
    states, ok = [], False
    for name, url, params in urls:
        try:
            r = _get(url, params)
            body = (r.text or "")[:90]
            states.append(f"{name} {r.status_code} {body}")
            if r.status_code == 200:
                d = r.json().get("data") or {}
                if d.get("rank_list"):
                    ok = True
        except Exception as e:                  # noqa: BLE001
            _logger.warning("[探针] 腾讯 %s 失败: %s: %s", name, type(e).__name__, e)
            states.append(f"{name} FAIL {type(e).__name__}")
    return _result("腾讯", "ok" if ok else "no_func", "；".join(states))


PROBES = (probe_eastmoney_datacenter, probe_eastmoney_push2, probe_sina_concept,
          probe_ths, probe_futu, probe_xueqiu, probe_tencent)


def run(full: bool = False):
    rows = []
    for fn in PROBES:
        try:
            rows.append(fn(full=full) if fn is probe_eastmoney_datacenter else fn())
        except Exception as e:                  # noqa: BLE001 — 单源异常不得拖垮整体探测
            _logger.warning("[探针] %s 未预期异常: %s", fn.__name__, e)
            rows.append(_result(fn.__name__, "error",
                                f"{type(e).__name__}: {str(e)[:120]}"))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="概念板块数据源探针")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--full", action="store_true",
                    help="额外拉全量东财题材映射（+约 75s，用于核对概念总数）")
    args = ap.parse_args(argv)

    socket.setdefaulttimeout(TIMEOUT)
    rows = run(full=args.full)
    usable = [r for r in rows if r["verdict"] == "ok"]

    if args.json:
        print(json.dumps({"rows": rows, "usable": len(usable)},
                         ensure_ascii=False, indent=2))
    else:
        print(f"{'源':<20}{'判定':<9}明细")
        print("-" * 100)
        for r in rows:
            print(f"{r['source']:<20}{r['verdict']:<9}{r['detail']}")
        print()
        if usable:
            print(f"结论：{len(usable)} 条可用 → {[r['source'] for r in usable]}")
        else:
            print("结论：没有可用的「概念→成分股」数据源。"
                  "阶段 B 需换环境或改源，**不要凭猜写解析器**。")
    return 0 if usable else 2


if __name__ == "__main__":
    sys.exit(main())
