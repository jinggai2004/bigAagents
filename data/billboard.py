# -*- coding: utf-8 -*-
"""龙虎榜「机构买卖」数据源（东财 datacenter-web）。

为什么补这个
────────────
五路初筛里「龙虎榜活跃」自 2026-10-09 本地化后一直 `unavailable`——问财失效后没有
本地数据源，一路覆盖率直接归零。而它恰好是五路里**唯一不依赖价量形态**的一路：
主力资金看资金流、技术突破看涨幅换手、低估值看成 PE、行业龙头看规模，
全是"价格/规模"的导数；机构席位净买入是**另一个主体的行为**，信息维度不同。

数据源（2026-11-01 实测）
─────────────────────────
- `RPT_ORGANIZATION_TRADE_DETAILS`：机构专用席位买卖汇总。**59 行/交易日、1.2 秒**，
  字段直接给 `NET_BUY_AMT`（机构净买额）、`BUY_AMT`/`SELL_AMT`、`BUY_COUNT`/`SELL_COUNT`（机构家数）。
  这就是原条件「机构净买入大于 1000 万」需要的那个数，不用自己从营业部明细里拼。
- `RPT_DAILYBILLBOARD_DETAILSNEW`：全席位龙虎榜（337 行/3 天），带 `D1/D5/D20_CLOSE_ADJCHRATE`
  （榜后涨跌幅，已复权）⇒ **可以先验证信号有没有预测力，再决定接不接**。
  本项目不接受"有数据就等于有信息"，所以 `evidence()` 就是干这个的。

口径
────
- 缓存按**交易日**一个 parquet（`data/panel/billboard/YYYY-MM-DD.parquet`），单日 60 行左右，
  断点续跑天然成立；失败不写文件 ⇒ 下次自动重试。
- 必须 filter `SECURITY_TYPE_CODE="058001001"`（剔 B 股），否则会混进 900xxx。
- 金额单位是**元**（`NET_BUY_AMT`），页面/条件里的"1000 万"要乘 1e7 比较。
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from config.paths import data_root
from data.codes import is_a_share_code

_logger = logging.getLogger("data.billboard")

BASE = "https://datacenter-web.eastmoney.com/api/data/v1/get"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
TIMEOUT = 12
PAGE_SIZE = 500          # 单日最多 ~60 行，一页够
A_SHARE_TYPE = "058001001"

# ⚠️ frozen 后 `__file__` 在只读 _internal/ ⇒ 缓存必须走 data_root()（项目铁律）
CACHE_DIR = Path(data_root()) / "data" / "panel" / "billboard"

# ⚠️ 这张报表只给 D1/D2/D3/D5/D10，**没有 D20**（同族的龙虎榜明细才有）。
#    漏掉 D3/D10 的后果是"想验证更长的衰减期时发现列不存在，得重抓历史"（踩过）。
ORG_COLS = ["SECURITY_CODE", "SECURITY_NAME_ABBR", "TRADE_DATE", "NET_BUY_AMT",
            "BUY_AMT", "SELL_AMT", "BUY_COUNT", "SELL_COUNT", "EXPLANATION",
            "D1_CLOSE_ADJCHRATE", "D2_CLOSE_ADJCHRATE", "D3_CLOSE_ADJCHRATE",
            "D5_CLOSE_ADJCHRATE", "D10_CLOSE_ADJCHRATE"]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def _get(sess: requests.Session, report: str, flt: str, page: int = 1,
         sort: str | None = None, asc: bool = False) -> list[dict]:
    params = {"reportName": report, "columns": "ALL", "filter": flt,
              "pageSize": PAGE_SIZE, "pageNumber": page}
    if sort:
        params["sortColumns"] = sort
        params["sortTypes"] = "1" if asc else "-1"
    r = sess.get(BASE, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    # HTTP 200 ≠ 成功：必须解析业务码（项目铁律）
    if not payload.get("success"):
        raise RuntimeError(f"东财返回 success=False: {payload.get('message')}")
    return (payload.get("result") or {}).get("data") or []


def fetch_org_day(sess: requests.Session, day: str) -> pd.DataFrame:
    """拉某个交易日的机构席位汇总。无数据（非交易日/接口未更新）返回空表。

    ⚠️ 这张报表**没有 `SECURITY_TYPE_CODE` 列**（同族的龙虎榜明细才有），
    所以服务端过滤 B 股做不到——加了报 `success=False: 列不存在`。
    B 股只能在 Python 侧用 `is_a_share_code()` 剔（结构性剔除，不靠名称）。
    """
    rows = _get(sess, "RPT_ORGANIZATION_TRADE_DETAILS", f"(TRADE_DATE='{day}')")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = [c for c in ORG_COLS if c in df.columns]
    df = df[keep].copy()
    df["code"] = df["SECURITY_CODE"].astype(str).str.zfill(6)
    df = df[df["code"].map(is_a_share_code)]
    df["trade_date"] = day
    return df.reset_index(drop=True)


def _path(day: str) -> Path:
    return CACHE_DIR / f"{day}.parquet"


def _weekdays_back(asof: str, n_calendar: int = 20) -> list[str]:
    """往回数自然日并只保留工作日（节假日交给"取不到就跳过"处理）。"""
    d = datetime.strptime(asof, "%Y-%m-%d").date()
    out = []
    for i in range(n_calendar):
        cur = d - timedelta(days=i)
        if cur.weekday() < 5:
            out.append(cur.isoformat())
    return out


def ensure_days(days: list[str], *, force: bool = False) -> dict:
    """确保这些日期的缓存存在。返回 `{day: rows}`（取不到的记 0，不抛）。

    设计取舍：缺失/非交易日**不写空文件**——空文件会被当成"这天没有机构上榜"，
    与"这天还没抓"无法区分；而"没抓到"是要重试的。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    got: dict[str, int] = {}
    todo = [d for d in days if force or not _path(d).exists()]
    if not todo:
        return {d: int(len(pd.read_parquet(_path(d)))) for d in days if _path(d).exists()}
    sess = _session()
    for d in todo:
        try:
            df = fetch_org_day(sess, d)
        except Exception as e:                        # noqa: BLE001
            _logger.warning("[龙虎榜] %s 抓取失败（不写缓存，下次重试）: %s", d, e)
            got[d] = 0
            continue
        if df.empty:
            got[d] = 0
            continue
        df.to_parquet(_path(d), index=False)
        got[d] = len(df)
        time.sleep(0.3)                               # 别把东财打急
    return got


def load_range(start: str, end: str) -> pd.DataFrame:
    """读缓存区间（**不联网**；要补数据先调 `ensure_days`）。"""
    if not CACHE_DIR.exists():
        return pd.DataFrame()
    files = sorted(CACHE_DIR.glob("*.parquet"))
    parts = []
    for f in files:
        d = f.stem
        if start <= d <= end:
            try:
                parts.append(pd.read_parquet(f))
            except Exception as e:                    # noqa: BLE001  坏文件跳过
                _logger.warning("[龙虎榜] 读 %s 失败: %s", f.name, e)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def org_net_buy(codes=None, *, asof: str | None = None, days: int = 5,
                min_net: float = 1e7) -> dict[str, dict]:
    """近 `days` 个交易日各股的**机构净买额累计**。

    Args:
        codes: 只算这些（None=全部）。
        asof: 截止日（含）；默认今天。回测/复盘要传，别用 today()。
        days: 回看多少个“有数据的交易日”。
        min_net: 少于这个累计净买额的不返回（元）。默认 1000 万 = 原问财条件。

    Returns:
        `{code: {"net": 元, "times": 上榜次数, "last_date": 最后上榜日}}`
    """
    asof = asof or date.today().isoformat()
    cand = _weekdays_back(asof, n_calendar=max(10, days * 3))
    with_data = [d for d in cand if _path(d).exists()]
    if not with_data:
        return {}
    span = with_data[:days]                            # 已按倒序
    if not span:
        return {}
    df = load_range(span[-1], span[0])
    if df.empty:
        return {}
    # 缓存可能早于"抓取时剔 B 股"那次改动 ⇒ 这里再结构性过滤一次。
    # 不能只在 fetch 侧过滤：读侧不设防的话，历史缓存会把 900xxx 带进候选池。
    df = df[df["code"].map(is_a_share_code)]
    if df.empty:
        return {}
    if codes is not None:
        want = {str(c).zfill(6) for c in codes}
        df = df[df["code"].isin(want)]
    if df.empty:
        return {}
    agg = (df.groupby("code")
             .agg(net=("NET_BUY_AMT", "sum"),
                  times=("trade_date", "nunique"),
                  last_date=("trade_date", "max"),
                  name=("SECURITY_NAME_ABBR", "last"))
             .reset_index())
    agg = agg[agg["net"] >= min_net]
    return {r["code"]: {"net": float(r["net"]), "times": int(r["times"]),
                        "last_date": str(r["last_date"]),
                        "name": str(r["name"] or "")}
            for _, r in agg.iterrows()}


def evidence(start: str, end: str, horizon: str = "D5_CLOSE_ADJCHRATE") -> dict:
    """用榜后涨跌幅（接口自带、已复权）验证"机构净买入"有没有预测力。

    为什么必须先跑这个：本项目最贵的教训是**有数据 ≠ 有信息**。龙虎榜是典型的
    "讲得通但可能是噪音"的因子——机构买了之后涨不涨，得看数，不能凭印象接进去。

    Returns: 分组的均值/胜率 + 全样本基准，以及 t 值（组间差 / 标准误）。
    """
    df = load_range(start, end)
    if df.empty or horizon not in df.columns:
        return {"ok": False, "reason": f"缓存无数据或缺 {horizon}（先 ensure_days 回填）"}
    d = df[[horizon, "NET_BUY_AMT", "code", "trade_date"]].copy()
    d[horizon] = pd.to_numeric(d[horizon], errors="coerce")
    d = d.dropna(subset=[horizon])
    if len(d) < 100:
        return {"ok": False, "reason": f"样本太少（{len(d)}），榜后收益还没跑出来"}

    base = float(d[horizon].mean())
    buy = d[d["NET_BUY_AMT"] >= 1e7]
    sell = d[d["NET_BUY_AMT"] <= -1e7]
    def _stat(x: pd.DataFrame) -> dict:
        if x.empty:
            return {"n": 0}
        v = x[horizon]
        return {"n": int(len(v)), "mean": round(float(v.mean()), 3),
                "win": round(float((v > 0).mean()), 3),
                "excess": round(float(v.mean()) - base, 3)}
    # Welch t：两组均值差是否显著（样本量差异大，不用合并方差）
    t = None
    if len(buy) >= 30 and len(sell) >= 30:
        a, b = buy[horizon], sell[horizon]
        se = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 0.5
        if se > 0:
            t = round(float((a.mean() - b.mean()) / se), 2)
    return {"ok": True, "horizon": horizon, "rows": int(len(d)),
            "sample_start": start, "sample_end": end,
            "baseline_mean": round(base, 3), "baseline_win": round(float((d[horizon] > 0).mean()), 3),
            "net_buy_gt_1000w": _stat(buy), "net_sell_gt_1000w": _stat(sell),
            "t_buy_minus_sell": t}


def cache_status() -> dict:
    if not CACHE_DIR.exists():
        return {"days": 0, "first": None, "last": None, "rows": 0}
    files = sorted(CACHE_DIR.glob("*.parquet"))
    if not files:
        return {"days": 0, "first": None, "last": None, "rows": 0}
    rows = 0
    for f in files:
        try:
            rows += int(len(pd.read_parquet(f, columns=["code"])))
        except Exception as e:                        # noqa: BLE001
            _logger.debug("缓存分片行数读取失败，跳过 %s: %s", f, e)
    return {"days": len(files), "first": files[0].stem, "last": files[-1].stem,
            "rows": rows}


def main(argv=None) -> int:
    """运维口：`py -m data.billboard --backfill [--days N]` / `--evidence` / `--status`。"""
    import argparse
    import json
    ap = argparse.ArgumentParser(description="龙虎榜（机构席位）数据源")
    ap.add_argument("--backfill", action="store_true", help="回填最近 N 个自然日")
    ap.add_argument("--days", type=int, default=200, help="回填窗口（自然日）")
    ap.add_argument("--evidence", action="store_true", help="验证机构净买入的预测力")
    ap.add_argument("--status", action="store_true", help="看缓存状态")
    ap.add_argument("--asof", default=date.today().isoformat())
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if a.backfill:
        days = _weekdays_back(a.asof, n_calendar=a.days)
        got = ensure_days(days)
        ok = {d: n for d, n in got.items() if n}
        _logger.info("回填完成：%s/%s 个工作日有数据", len(ok), len(days))
        _logger.info("缓存：%s", cache_status())
    if a.evidence:
        start = (datetime.strptime(a.asof, "%Y-%m-%d") - timedelta(days=a.days)).date().isoformat()
        for h in ("D1", "D3", "D5", "D10"):
            r = evidence(start, a.asof, f"{h}_CLOSE_ADJCHRATE")
            if not r.get("ok"):
                _logger.info("%s: %s", h, r.get("reason"))
                continue
            b, s = r["net_buy_gt_1000w"], r["net_sell_gt_1000w"]
            _logger.info(
                "%3s | 全榜基准 %+.2f%% | 机构净买 %+.2f%%(n=%s, 胜率 %.0f%%) "
                "超额 %+.2fpp | 机构净卖 %+.2f%% | t=%s",
                h, r["baseline_mean"], b["mean"], b["n"], b["win"] * 100,
                b["excess"], s["mean"], r["t_buy_minus_sell"])
    if a.status or not (a.backfill or a.evidence):
        _logger.info("%s", json.dumps(cache_status(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
