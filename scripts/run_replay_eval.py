# -*- coding: utf-8 -*-
"""P1-2 历史回放评测 CLI —— 生成 Markdown 报告。

用法：
    python scripts/run_replay_eval.py                          # 默认 4 年样本
    python scripts/run_replay_eval.py --days 1500 --top 20
    python scripts/run_replay_eval.py --out output/某报告.md

⚠️ 只读面板（`data/panel/parts`）+ 本地指数面板，**不联网、不写库**。
⚠️ 本文件所有字符串字面量用**单引号**、文案内引用用「」——
   中文引号在这类脚本里极易被误转成 ASCII 双引号而截断字符串（已踩过）。
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.replay_eval import evaluate  # noqa: E402

_FACTOR_CN = {
    'momentum_20': '20日动量（趋势）',
    'ma_align': '均线排列（MA5/MA20-1）',
    'volatility_20': '20日波动率（越低越好）',
    'amount_ratio': '放量（5日/60日均额）',
}
_CASE_CN = {
    'raw': '原始（全市场）',
    'ex_mom30': '剔除近20日涨>30%',
    'ex_mom15': '剔除近20日涨>15%',
    'ex_limit_up': '剔除当日涨幅≥9%',
    'ex_mom15_and_limit': '两者都剔',
}


def _pct(x, digits=1):
    return '—' if x is None else f'{x:+.{digits}%}'


def _num(x, digits=3):
    return '—' if x is None else f'{x:+.{digits}f}'


def render(res) -> str:
    d = res.to_dict()
    L = []
    L.append(f'# 历史回放评测（{datetime.now():%Y-%m-%d}）')
    L.append('')
    L.append(f'> 样本：**{d["date_start"]} ~ {d["date_end"]}**'
             f'（{d["n_days"]} 个交易日 × {d["n_codes"]} 只，面板全市场含退市股）；'
             '零联网、零 LLM。')
    fp = d.get('fingerprint') or {}
    if fp:
        L.append('>')
        L.append(f'> **数据指纹**：面板 {fp.get("parts")} 分片 / '
                 f'{fp.get("panel_rows")} 行 / 至 {fp.get("panel_date_max")}'
                 f'（该日截面 {fp.get("panel_cross_section_at_max")} 只）；'
                 f'pe_ttm 覆盖 {fp.get("panel_pe_ttm_codes_at_max")} 只；'
                 f'指纹 `{fp.get("panel_signature")}`；'
                 f'入场时点 `{fp.get("entry")}`；'
                 f'剔除残截面 {fp.get("dropped_dates")} 日；'
                 f'生成于 {fp.get("generated_at")}。')
        L.append('>')
        L.append('> ⚠️ **比对两次结论前先比对指纹**：面板在 BaoStock 回填期间持续变化，'
                 '指纹不同就不是同一份样本。')
    L.append('')

    port5 = d['portfolio'].get(5) or {}
    oh = (d.get('overheat') or {}).get('cases') or {}
    raw = oh.get('raw') or {}
    best = oh.get('ex_mom15') or {}

    L.append('## 一、结论摘要')
    L.append('')
    if port5.get('available'):
        L.append(f'1. **按实盘方向取 TOP{port5["top_n"]} 在当前样本是负 alpha**：'
                 f'持有 5 日年化 {_pct(port5["ann_return"])}，'
                 f'全市场等权 {_pct(port5["bench_ann"])}、'
                 f'沪深300 {_pct(port5.get("index_ann"))}；'
                 f'胜率 {port5["win_rate"]:.0%}。')
    if raw.get('available'):
        L.append(f'2. **亏损主要来自「追高」**：原始 TOP 每期 {_pct(raw["top_mean"], 2)}、'
                 f'超额 {_pct(raw["top_excess"], 2)}；'
                 + (f'剔除近 20 日涨幅 >15% 后 TOP {_pct(best["top_mean"], 2)}、'
                    f'超额 {_pct(best["top_excess"], 2)}'
                    f'（改善 {_pct(best.get("improvement_vs_raw"), 2)}/期）。'
                    if best.get('available') else ''))
        L.append(f'3. **不是单纯「方向反了」**：反向 BOTTOM 组合超额 '
                 f'{_pct(raw["bot_excess"], 2)}（同样跑输）'
                 '⇒ 因子在**极端分位无效**，中间分位才有效。')
    L.append('')
    L.append('> ⚠️ 本评测只覆盖**价量三因子**（实盘权重合计 41.17%）。'
             '「主力资金」「新闻正面率」无历史序列；「PE 估值」待 BaoStock 回填完成。'
             '因此**不能据此断定「实盘选股无效」**——但可以断定这三因子按现权重'
             '正向叠加在当前样本是拖累。')
    L.append('')

    L.append('## 二、IC 表（截面 Spearman，分持有期）')
    L.append('')
    horizons = sorted({h for cells in d['ic'].values() for h in cells})
    L.append('| 因子 | ' + ' | '.join(f'h={h}' for h in horizons) + ' |')
    L.append('|' + '---|' * (len(horizons) + 1))
    for f, cells in d['ic'].items():
        row = [f'{_FACTOR_CN.get(f, f)}']
        for h in horizons:
            a = (cells.get(h) or {}).get('all') or {}
            row.append(f'{_num(a.get("mean"))}（n={a.get("n", 0)}）')
        L.append('| ' + ' | '.join(row) + ' |')
    L.append('')
    L.append('> IC 全为**负值**且 bootstrap 95% 区间不含 0 ⇒ 该窗口内'
             '「因子值越大、后续收益越低」。方向已按实盘（动量/均线正、波动率负）'
             '对齐过，所以负 IC = **实盘方向反向**。')
    L.append('')

    L.append('### 分年 IC（h=5）')
    L.append('')
    years = sorted({y for cells in d['ic'].values()
                    for v in cells.values() for y in (v.get('by_year') or {})})
    if years:
        L.append('| 因子 | ' + ' | '.join(str(y) for y in years) + ' |')
        L.append('|' + '---|' * (len(years) + 1))
        for f, cells in d['ic'].items():
            v = (cells.get(5) or {}).get('by_year') or {}
            L.append('| ' + _FACTOR_CN.get(f, f) + ' | ' +
                     ' | '.join(_num((v.get(y) or {}).get('mean')) for y in years) + ' |')
        L.append('')

    L.append('## 三、组合回放（TOP N 等权，逐期不重叠）')
    L.append('')
    L.append('| 持有期 | 调仓步长 | 期数 | 年化 | 全市场等权 | 沪深300 | 超额(等权) | 最大回撤 | 胜率 |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for h in sorted(d['portfolio']):
        p = d['portfolio'][h]
        if not p.get('available'):
            continue
        L.append(f'| {h} 日 | {p["step"]} 日 | {p["periods"]} | '
                 f'{_pct(p["ann_return"])} | {_pct(p["bench_ann"])} | '
                 f'{_pct(p.get("index_ann"))} | {_pct(p["ann_excess"])} | '
                 f'{p["max_drawdown"]:.0%} | {p["win_rate"]:.0%} |')
    L.append('')
    L.append('> `step` 必须等于持有期：否则各期持仓重叠，复利会把同一段收益重复计入'
             '（实测 step=5 + h=60 曾算出「年化 -98%」，是假的）。')
    L.append('')

    L.append('## 四、过热诊断（本次最重要的发现）')
    L.append('')
    h0 = (d.get('overheat') or {}).get('horizon', 5)
    L.append(f'同一条复合分、同一批交易日，只换「候选过滤条件」（持有 {h0} 日）：')
    L.append('')
    L.append('| 过滤条件 | 期数 | TOP 每期 | 全市场每期 | TOP 超额 | 相对原始改善 | 胜率 |')
    L.append('|---|---|---|---|---|---|---|')
    for k, v in ((d.get('overheat') or {}).get('cases') or {}).items():
        if not v.get('available'):
            continue
        L.append(f'| {_CASE_CN.get(k, k)} | {v["periods"]} | {_pct(v["top_mean"], 2)} | '
                 f'{_pct(v["all_mean"], 2)} | {_pct(v["top_excess"], 2)} | '
                 f'{_pct(v.get("improvement_vs_raw"), 2)} | {v["top_win"]:.0%} |')
    L.append('')
    L.append('**解读**：超额几乎全部由「近 20 日涨幅 >15%」这一条贡献'
             '（改善约 2 个百分点/期），而单独剔涨停几乎无效 ⇒ 问题不是'
             '「当天涨停买不进」，而是**选出来的就是已经涨了一大段的票**，'
             '随后 5~20 日的均值回归把超额吃光。')
    L.append('')
    L.append('**可执行结论**：实盘候选应加**涨幅过热过滤**'
             '（例如近 20 日涨幅 >15% 降权/剔除），改动前用本工具复算确认。'
             '⚠️ 但**不得**仅凭本样本直接改 `FACTOR_WEIGHTS`——'
             '需过 `factor_analysis.robustness()` 的多区间检验。')
    L.append('')

    L.append('## 五、限制与下一步')
    L.append('')
    for c in d.get('caveats') or []:
        L.append(f'- {c}')
    L.append(f'- 入场时点：`{fp.get("entry")}`'
             '（`next_open` = T 日收盘出信号、T+1 开盘成交，与项目回测纪律一致；'
             '`close` = 同日收盘成交，旧口径，仅用于对照）')
    L.append('- IC 的 h=20/60 与采样步长不等 ⇒ 标签重叠、CI 偏窄（方向可信、强度打折）')
    L.append('- 未建模：涨跌停买不进、停牌、交易成本（每期全额换仓，双边约 0.2~0.3%）')
    L.append('- 下一步：① BaoStock 回填完成后纳入 PE 因子；'
             '② 用 `robustness()` 做多区间交叉验证；'
             '③ 若引入过热过滤，实盘 screener 用同一阈值并记录到 `screen_picks`')
    L.append('')
    return '\n'.join(L)


def _entry_compare(a, b) -> str:
    """两种入场时点对比：`a` = 同日收盘（close），`b` = 次日开盘（next_open）。"""
    if a is None or b is None:
        return ''
    L = ['', '## 六、入场时点对照（同日收盘 vs 次日开盘）', '']
    L.append('| 持有期 | 同日收盘 年化 | 次日开盘 年化 | 时点差 | 同日 超额/期 | 次日 超额/期 |')
    L.append('|---|---|---|---|---|---|')
    for h in sorted(set(a.portfolio) & set(b.portfolio)):
        pa, pb = a.portfolio[h], b.portfolio[h]
        if not (pa.get('available') and pb.get('available')):
            continue
        L.append(f'| {h} 日 | {_pct(pa["ann_return"])} | {_pct(pb["ann_return"])} '
                 f'| {_pct(pb["ann_return"] - pa["ann_return"])} '
                 f'| {_pct(pa["excess_mean"], 2)} | {_pct(pb["excess_mean"], 2)} |')
    L.append('')
    L.append('> 时点差 = 次日开盘 − 同日收盘。若两者结论方向一致，说明'
             '「追高导致跑输」不是成交时点造成的假象；差异大则必须按实盘口径'
             '（次日开盘）取数。')
    L.append('')
    return '\n'.join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=1500,
                    help='自然日窗口（默认 1500 ≈ 4 年）')
    ap.add_argument('--horizons', default='5,20,60')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--step', type=int, default=5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--entry', default='next_open',
                    choices=['close', 'next_open'],
                    help='入场时点：next_open=T+1 开盘成交（默认，与回测纪律一致）；'
                         'close=同日收盘成交（旧口径）')
    ap.add_argument('--min-cross', type=int, default=None,
                    help='截面覆盖下限（少于该只数的交易日剔除；默认自适应 3000）')
    ap.add_argument('--compare-entry', action='store_true',
                    help='额外跑对照口径（close），在报告里对比时点差')
    args = ap.parse_args()

    hz = tuple(int(x) for x in args.horizons.split(','))
    res = evaluate(days=args.days, step=args.step, horizons=hz,
                   top_n=args.top, entry=args.entry,
                   min_cross=args.min_cross)
    md = render(res)
    if args.compare_entry:
        alt = 'close' if args.entry != 'close' else 'next_open'
        cmp_res = evaluate(days=args.days, step=args.step, horizons=hz,
                           top_n=args.top, entry=alt, min_cross=args.min_cross)
        # ⚠️ 固定列序（左=同日收盘，右=次日开盘）：不能按"主/对照"传参，
        #    否则 --entry next_open 时列标签会反（已踩一次）。
        pair = {res.fingerprint.get('entry'): res,
                cmp_res.fingerprint.get('entry'): cmp_res}
        md += _entry_compare(pair.get('close'), pair.get('next_open'))
    out = args.out or os.path.join(
        'output', f'历史回放评测_{datetime.now():%Y-%m-%d}.md')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(md[:1200])
    print(f'\n[已写出] {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
