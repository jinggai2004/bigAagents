# -*- coding: utf-8 -*-
"""从同花顺导出文件导入自选股与持仓。

## 支持什么（导出方式以 PC 同花顺为准，2026-10-30 核实）

**自选股**（自选列表 → Ctrl+A → 右键「数据导出-导出选中股票」）：
   xlsx / csv / txt，含股票代码+名称；多分组要**切换分组分别导出**后一起传进来。

**持仓**，两种文件都支持，自动识别：
  1. 「查询 → 资金股份」右键「输出到Excel」——当前持仓**快照**，
     有 股票余额/参考成本价；⚠️ 快照**没有建仓日期**，导入时 entry_date 记为导入当天。
  2. 「查询 → 交割单」导出——按 买入/卖出 逐笔**重建**持仓（移动平均成本），
     可还原建仓日期与成本价；PC 端单次最多查 3 个月，**分多次导出的多个文件
     一起传**会自动合并。⚠️ 导出区间没覆盖完整历史的，成本与日期是区间内近似。

不支持「对账单」重建持仓：它只有发生金额、没有成交数量。

## 用法

    python scripts/import_from_ths.py watchlist 导出.csv [更多文件...] [--owner user|ashen] [--commit]
    python scripts/import_from_ths.py positions  交割单1.xlsx 交割单2.xlsx [--commit]
    python scripts/import_from_ths.py detect     某个文件     # 只识别格式，什么都不写

默认 **dry-run**：只打印将要发生什么；确认无误后加 `--commit` 才真正写库。

## 写库语义（刻意选择，勿随意改）

- 自选股：走 `data.watchlist.add_stock`（fail-closed，库读失败即中止），
  只**增量新增**、绝不清空现有清单；source 记为 `ths_import` 便于溯源。
- 持仓：只新增「当前无活跃持仓」的代码；已有活跃持仓的**跳过并报告**
  （两边账本成本口径不同，绝不做对账式覆盖）。止损/止盈留空——
  `check_stop_loss` 对 NULL 是安全跳过（2026-10-30 已核），由人工后续设置。
- 只收 A 股（`data.codes.is_a_share_code`：沪深主板/创业/科创/北交所）；
  基金、B 股、债券行会被过滤并在报告里点名。
- 全程零联网：名称只取文件自带的，缺失就留空。

## 为什么不解析 blk 板块文件

`T0002\\blocknew\\*.blk` 是同花顺内部格式（版本间不保证兼容，官方只建议用它
做软件内恢复，不建议第三方解析）；官方给出的可移植路径就是上面的导出功能。
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 允许 `python scripts/import_from_ths.py` 直跑

from data.codes import is_a_share_code  # noqa: E402

_logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# 读文件：xlsx/xls + csv/txt（编码回退 utf-8-sig → gb18030，同花顺中文导出常是 GBK）
# ----------------------------------------------------------------
_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")
_SEP_CANDIDATES = (",", "\t", ";")


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError as e:
            _logger.debug("按 %s 解码失败（尝试下一编码）: %s", enc, e)
            continue
    return raw.decode("gb18030", errors="replace")  # 最后兜底：不抛，替换非法字符


def read_table(path: str | Path) -> pd.DataFrame:
    """读同花顺导出文件为 DataFrame（表头原样保留，列名去空白）。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"文件不存在: {p}")
    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(p, dtype=str)
    elif suffix in (".csv", ".txt"):
        text = _read_text(p)
        # 分隔符探测：取能切出最多列的分隔符
        best, best_cols = None, 1
        for sep in _SEP_CANDIDATES:
            cols = len(text.splitlines()[0].split(sep)) if text.splitlines() else 1
            if cols > best_cols:
                best, best_cols = sep, cols
        if best is None:  # 无分隔符 ⇒ 每行一个代码（可能带名称）的裸文本
            rows = [ln.split() for ln in text.splitlines() if ln.strip()]
            width = max((len(r) for r in rows), default=0)
            data = [r + [""] * (width - len(r)) for r in rows]
            df = pd.DataFrame(data)
        else:
            from io import StringIO
            df = pd.read_csv(StringIO(text), sep=best, dtype=str)
    else:
        raise ValueError(f"不支持的文件类型 {suffix}（支持 xlsx/xls/csv/txt）: {p}")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


# ----------------------------------------------------------------
# 列定位
# ----------------------------------------------------------------
def _find_col(cols, keywords: tuple[str, ...]) -> str | None:
    """列名包含任一关键词即命中（'代码' 会命中 '股票代码'/'证券代码'）。"""
    for kw in keywords:
        for c in cols:
            if kw.lower() in str(c).lower():
                return c
    return None


def _num(series: pd.Series) -> pd.Series:
    """文本数字清洗：去千分位逗号/空格/¥，转 float。"""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[,，\s¥￥]", "", regex=True),
        errors="coerce")


_WL_CODE_KW = ("股票代码", "证券代码", "代码", "code")
_WL_NAME_KW = ("股票名称", "证券名称", "名称", "简称", "name")
_PS_QTY_SNAP_KW = ("股票余额", "证券余额", "持仓数量", "持仓余额", "当前持仓", "余额")
_PS_COST_KW = ("参考成本价", "成本价", "买入成本", "摊薄成本", "成本")
_PS_DATE_KW = ("成交日期", "发生日期", "日期", "time")
_PS_DIR_KW = ("买卖方向", "操作", "业务名称", "交易类型", "摘要", "方向")
_PS_QTY_LED_KW = ("成交数量", "成交股数", "数量")
_PS_PRICE_KW = ("成交价格", "成交价", "价格")
_BUY_WORDS = {"买入", "买", "b", "buy", "BUY", "申购"}
_SELL_WORDS = {"卖出", "卖", "s", "sell", "SELL", "赎回"}


# ----------------------------------------------------------------
# 自选股
# ----------------------------------------------------------------
def parse_watchlist(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """解析自选股导出 → (rows [{"code","name"}], dropped [原始值])。

    兼容「数据导出」的带表头表格，也兼容无表头的裸代码文本（第一列代码、
    第二列若有则当名称）。
    """
    cols = list(df.columns)
    code_col = _find_col(cols, _WL_CODE_KW)
    name_col = _find_col([c for c in cols if c != code_col], _WL_NAME_KW)
    if code_col is None:
        first = cols[0]
        vals = df[first].astype(str).str.strip()
        if (vals.str.fullmatch(r"\d{6}").mean() or 0) >= 0.6:  # 无表头裸文本
            code_col, name_col = first, (cols[1] if len(cols) > 1 else None)
        else:
            raise ValueError(f"找不到代码列（现有列: {cols}）")

    rows: dict[str, dict] = {}
    dropped: list[str] = []
    for _, r in df.iterrows():
        code = str(r.get(code_col, "")).strip().split(".")[0]  # '000001.SZ' / 数字格
        if code.endswith(".0"):  # Excel 把代码读成数字的兜底
            code = code[:-2]
        name = str(r.get(name_col, "")).strip() if name_col else ""
        if not re.fullmatch(r"\d{6}", code):
            if code and code.lower() != "nan":
                dropped.append(code)
            continue
        if not is_a_share_code(code):
            dropped.append(code)
            continue
        if code not in rows:
            rows[code] = {"code": code, "name": name}
        elif name and not rows[code]["name"]:
            rows[code]["name"] = name
    return list(rows.values()), dropped


# ----------------------------------------------------------------
# 持仓：两种模式
# ----------------------------------------------------------------
def detect_mode(df: pd.DataFrame) -> str | None:
    """'snapshot'（资金股份）/ 'ledger'（交割单）/ None（识别不了）。"""
    cols = tuple(df.columns)
    if _find_col(cols, _PS_QTY_SNAP_KW) and _find_col(cols, _PS_COST_KW):
        return "snapshot"
    if (_find_col(cols, _PS_DIR_KW) and _find_col(cols, _PS_QTY_LED_KW)
            and _find_col(cols, _PS_PRICE_KW)):
        return "ledger"
    return None


def parse_snapshot(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """资金股份快照 → (rows, dropped)。entry_date 由调用方补（快照无日期）。"""
    cols = list(df.columns)
    code_col = _find_col(cols, _WL_CODE_KW)
    name_col = _find_col([c for c in cols if c != code_col], _WL_NAME_KW)
    qty_col = _find_col(cols, _PS_QTY_SNAP_KW)
    cost_col = _find_col(cols, _PS_COST_KW)
    if not code_col or not qty_col or not cost_col:
        raise ValueError(f"快照列不完整（现有列: {cols}）")

    rows, dropped = [], []
    qty_s, cost_s = _num(df[qty_col]), _num(df[cost_col])
    for i in range(len(df)):
        code = str(df[code_col].iloc[i]).strip().split(".")[0]
        if code.endswith(".0"):
            code = code[:-2]
        qty = qty_s.iloc[i]
        if not re.fullmatch(r"\d{6}", code):
            if code and code.lower() != "nan":
                dropped.append(code)
            continue
        if not is_a_share_code(code):
            dropped.append(code)
            continue
        if not qty or qty <= 0 or pd.isna(qty):
            continue  # 已清仓/零余额，正常状态，不进报告
        cost = float(cost_s.iloc[i]) if pd.notna(cost_s.iloc[i]) else 0.0
        if cost <= 0:
            cost = 0.0  # 成本缺失 ⇒ 记 0 并在报告点名（由人工补）
        rows.append({"code": code,
                     "name": str(df[name_col].iloc[i]).strip() if name_col else "",
                     "quantity": int(qty), "entry_price": cost, "entry_date": None,
                     "warnings": ["成本价缺失"] if cost <= 0 else []})
    return rows, dropped


def parse_ledger(dfs: list[pd.DataFrame]) -> tuple[list[dict], list[str]]:
    """交割单（可多个分段文件合并）逐笔重建持仓，移动平均成本。

    ⚠️ 导出区间不完整的（先卖后买出现负持仓）会逐笔记 warning，
    但**仍按可得数据导入**——半成品成本比空着强，且 warning 会呈现在报告里。
    """
    frames = []
    for df in dfs:
        cols = list(df.columns)
        code_col = _find_col(cols, _WL_CODE_KW)
        date_col = _find_col(cols, _PS_DATE_KW)
        dir_col = _find_col(cols, _PS_DIR_KW)
        qty_col = _find_col(cols, _PS_QTY_LED_KW)
        price_col = _find_col(cols, _PS_PRICE_KW)
        name_col = _find_col([c for c in cols if c != code_col], _WL_NAME_KW)
        if not all([code_col, dir_col, qty_col, price_col]):
            raise ValueError(f"交割单列不完整（现有列: {cols}）")
        frames.append(pd.DataFrame({
            "code": df[code_col].astype(str).str.strip().str.split(".").str[0],
            "name": df[name_col].astype(str).str.strip() if name_col else "",
            "date": pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT,
            "dir": df[dir_col].astype(str).str.strip(),
            "qty": _num(df[qty_col]),
            "price": _num(df[price_col]),
        }))
    if not frames:
        return [], []
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows[all_rows["code"].str.fullmatch(r"\d{6}", na=False)]
    all_rows = all_rows[all_rows["code"].map(is_a_share_code)]
    all_rows = all_rows.dropna(subset=["qty", "price"])
    all_rows = all_rows[all_rows["qty"] > 0]

    out, dropped = [], []
    for code, g in all_rows.groupby("code"):
        g = g.sort_values("date", na_position="last")
        name = next((n for n in g["name"] if n and n.lower() != "nan"), "")
        qty_on_hand, total_cost = 0.0, 0.0
        first_buy, warnings = None, []
        for _, r in g.iterrows():
            d = r["dir"]
            if d in _BUY_WORDS:
                total_cost += r["price"] * r["qty"]
                qty_on_hand += r["qty"]
                if first_buy is None:
                    first_buy = r["date"]
            elif d in _SELL_WORDS:
                if qty_on_hand <= 0:
                    warnings.append(f"{r['date']:%Y-%m-%d} 卖出{r['qty']:.0f}股时区间内无持仓"
                                    "（导出区间可能不完整，成本被低估）")
                    continue
                avg = total_cost / qty_on_hand
                total_cost -= avg * min(r["qty"], qty_on_hand)
                qty_on_hand -= r["qty"]
                if qty_on_hand <= 0:
                    qty_on_hand, total_cost = 0.0, 0.0
            else:
                warnings.append(f"未识别的业务类型「{d}」已跳过（{r['date']:%Y-%m-%d}）")
        if qty_on_hand > 0:
            entry_date = (first_buy.strftime("%Y-%m-%d")
                          if first_buy is not None and pd.notna(first_buy)
                          else None)
            out.append({"code": code, "name": name,
                        "quantity": int(qty_on_hand),
                        "entry_price": round(total_cost / qty_on_hand, 4),
                        "entry_date": entry_date,
                        "warnings": warnings})
    return out, dropped


# ----------------------------------------------------------------
# 写库
# ----------------------------------------------------------------
def commit_watchlist(rows: list[dict], owner: str = "user") -> dict:
    """增量写入自选股。走 add_stock 的 fail-closed 路径；已存在的跳过。"""
    from data import watchlist as wl
    owner = wl.normalize_owner(owner)
    existing = {s["code"] for s in wl.load_watchlist(owner)}
    added, skipped = [], []
    for r in rows:
        if r["code"] in existing:
            skipped.append(r["code"])
            continue
        ok, msg = wl.add_stock(r["code"], r.get("name", ""), owner=owner,
                               source="ths_import")
        (added if ok else skipped).append(r["code"])
        if not ok:
            _logger.warning("[导入] %s: %s", r["code"], msg)
    return {"owner": owner, "added": added, "skipped": skipped}


def commit_positions(rows: list[dict]) -> dict:
    """新增活跃持仓：已有活跃持仓的代码跳过（绝不做对账式覆盖）。"""
    from db import store
    from db.models import get_connection

    existing = {p["code"] for p in store.list_active_positions()}
    added, skipped, failed = [], [], []
    today = date.today().isoformat()
    for r in rows:
        if r["code"] in existing:
            skipped.append(r["code"])
            continue
        entry_date = r.get("entry_date") or today  # 快照模式无建仓日 ⇒ 记导入日
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO positions
                   (code, name, entry_date, entry_price, quantity,
                    stop_loss, take_profit, status)
                   VALUES (?, ?, ?, ?, ?, NULL, NULL, 'active')""",
                (r["code"], r.get("name", ""), entry_date,
                 r["entry_price"], r["quantity"]))
            cur.execute(
                """INSERT INTO trade_logs
                   (date, code, direction, price, volume, reason, signal_source, pnl)
                   VALUES (?, ?, '买入', ?, ?, '同花顺导入', '外部导入', NULL)""",
                (f"{entry_date} 00:00", r["code"],
                 r["entry_price"], r["quantity"]))
            conn.commit()
            conn.close()
            existing.add(r["code"])
            added.append(r["code"])
        except Exception as e:  # noqa: BLE001
            _logger.warning("[导入] 写入持仓 %s 失败: %s", r["code"], e)
            failed.append(r["code"])
    return {"added": added, "skipped_existing": skipped, "failed": failed}


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
def _fmt_rows(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        warn = f"  ⚠ {'; '.join(r['warnings'])}" if r.get("warnings") else ""
        lines.append(f"  {r['code']}  {str(r.get('name') or '-'):<10}"
                     f"{r['quantity']:>8d} 股  成本 {r['entry_price']:<10}"
                     f"建仓 {r.get('entry_date') or '导入日'}{warn}")
    return "\n".join(lines) or "  （无）"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(description="从同花顺导出文件导入自选股与持仓")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_wl = sub.add_parser("watchlist", help="导入自选股导出文件（xlsx/csv/txt，可多文件）")
    ap_wl.add_argument("files", nargs="+")
    ap_wl.add_argument("--owner", default="user", choices=("user", "ashen"))
    ap_wl.add_argument("--commit", action="store_true", help="真正写库（默认 dry-run）")

    ap_ps = sub.add_parser("positions", help="导入持仓（资金股份快照 或 交割单，可多文件合并）")
    ap_ps.add_argument("files", nargs="+")
    ap_ps.add_argument("--commit", action="store_true", help="真正写库（默认 dry-run）")

    ap_dt = sub.add_parser("detect", help="只识别文件格式，什么都不写")
    ap_dt.add_argument("file")
    args = ap.parse_args(argv)

    if args.cmd == "detect":
        df = read_table(args.file)
        print(f"列: {list(df.columns)}")
        print(f"识别结果: {detect_mode(df) or '非持仓文件（按自选股解析如下）'}")
        if detect_mode(df) is None:
            rows, dropped = parse_watchlist(df)
            print(f"自选股 {len(rows)} 只，过滤 {len(dropped)}: {dropped[:10]}")
        return 0

    if args.cmd == "watchlist":
        merged: dict[str, dict] = {}
        dropped_all: list[str] = []
        for f in args.files:
            r, d = parse_watchlist(read_table(f))
            for x in r:
                merged.setdefault(x["code"], x)   # 多文件按 code 去重，先到先得
            dropped_all += d
        rows = list(merged.values())
        existing = set()
        if args.commit:
            from data import watchlist as wl
            existing = {s["code"] for s in wl.load_watchlist(args.owner)}
        print(f"解析出 A 股 {len(rows)} 只；非A股/非法过滤 {len(dropped_all)}: "
              f"{sorted(set(dropped_all))[:10]}")
        for r in rows[:50]:
            tag = "（已存在，将跳过）" if r["code"] in existing else ""
            print(f"  {r['code']}  {r.get('name') or '-'}{tag}")
        if len(rows) > 50:
            print(f"  … 共 {len(rows)} 只")
        if not args.commit:
            print("\n[dry-run] 未写库。确认无误后加 --commit 执行。")
            return 0
        res = commit_watchlist(rows, owner=args.owner)
        print(f"\n已写入 owner={res['owner']}：新增 {len(res['added'])}，"
              f"已存在/失败跳过 {len(res['skipped'])}")
        print(f"SUMMARY {json.dumps(res, ensure_ascii=False)}")
        return 0

    if args.cmd == "positions":
        snapshots, ledgers = [], []
        for f in args.files:
            df = read_table(f)
            mode = detect_mode(df)
            if mode == "snapshot":
                snapshots.append(df)
            elif mode == "ledger":
                ledgers.append(df)
            else:
                print(f"[跳过] 无法识别 {f} 的格式（列: {list(df.columns)[:12]}…）")
        rows, dropped_all = [], []
        for df in snapshots:
            r, d = parse_snapshot(df)
            rows += r
            dropped_all += d
        if ledgers:
            r, d = parse_ledger(ledgers)
            rows += [x for x in r if x["code"] not in {y["code"] for y in rows}]
            dropped_all += d
        if not rows:
            print("没有解析出任何持仓。请用 detect 子命令检查文件格式。")
            return 1
        print(f"解析出持仓 {len(rows)} 只；非A股过滤 {len(set(dropped_all))}: "
              f"{sorted(set(dropped_all))[:10]}")
        print(_fmt_rows(rows))
        if not args.commit:
            print("\n[dry-run] 未写库。确认无误后加 --commit 执行。")
            return 0
        res = commit_positions(rows)
        print(f"\n已写入：新增 {len(res['added'])} {res['added']}，"
              f"已有活跃持仓跳过 {len(res['skipped_existing'])} {res['skipped_existing']}，"
              f"失败 {len(res['failed'])} {res['failed']}")
        print("⚠ 导入的持仓止损/止盈为空，请在页面补设；"
              "快照模式的建仓日期记为导入当天。")
        print(f"SUMMARY {json.dumps(res, ensure_ascii=False)}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
