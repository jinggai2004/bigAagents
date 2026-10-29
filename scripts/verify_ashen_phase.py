"""实证「不同时间不同回答」：同一份账户/池子/行情，只改时刻，跑两轮真模型。

判据不是"看起来不一样"，而是：
  1. 两轮 summary 的差异必须落在**本时段的必答题**上（尾盘要谈过夜、开盘不谈）；
  2. 两轮 summary 的字面相似度必须显著低于改造前（46%）；
  3. prompt 里的【这个时段】必须是不同的阶段名。

⚠️ 只读：不落 routine_runs、不下单、不动账本。cost 约 ¥0.01。
"""
import json
import sys
from datetime import datetime

sys.path.insert(0, '.')
from core.ai_routine import build_prompt, phase_of

# 同一份事实：空仓、三只自选、都没跌够
KW = dict(
    level='L2', auton_desc='可在自己自选池内建仓，单票≤8%、总仓≤24%',
    budget_desc='单票≤8%，总仓≤24%，同时最多 3 只',
    cash=1_000_000.0, nav=1_000_000.0, holdings=[],
    my_watch=[
        {'code': '600887', 'name': '伊利股份', 'price': 28.92, 'change_pct': -0.8},
        {'code': '601166', 'name': '兴业银行', 'price': 18.71, 'change_pct': -0.6},
        {'code': '300498', 'name': '温氏股份', 'price': 17.23, 'change_pct': -0.9},
    ],
    user_watch=[], quotes={}, candidates=[],
    trail=[{'slot': '09:40', 'summary': '开盘低开但很快拉回，没给回踩机会，先看', 'actions': []}],
    armed=[],
)

CASES = [
    (datetime(2026, 10, 28, 9, 40), '09:40'),
    (datetime(2026, 10, 28, 14, 40), '14:40'),
]


def main():
    from llm.deepseek_client import DeepSeekClient, TEMP_ANALYTICAL

    client = DeepSeekClient()
    out = []
    for now, slot in CASES:
        pr = build_prompt(now=now, slot=slot, **KW)
        ph = phase_of(now)
        assert f'【这个时段】{ph[1]}' in pr, f'{slot} 的 prompt 没带上时段'
        raw = client.call_api([{'role': 'user', 'content': pr}],
                              temperature=TEMP_ANALYTICAL, max_tokens=600,
                              purpose='dev.verify.phase')
        txt = str(raw or '').strip()
        # 模型可能包 ```json ```；抠出第一个 { 到最后一个 }
        a, b = txt.find('{'), txt.rfind('}')
        obj = json.loads(txt[a:b + 1]) if a >= 0 and b > a else {}
        out.append({'slot': slot, 'phase': ph[1], 'summary': obj.get('summary') or '',
                    'actions': obj.get('actions') or []})
        print(f"--- {slot}（{ph[1]}）")
        print('  必答题:', ph[3])
        print('  结论  :', out[-1]['summary'])
        print('  动作  :', [x.get('action') for x in out[-1]['actions']])

    import difflib
    s1, s2 = out[0]['summary'], out[1]['summary']
    sim = difflib.SequenceMatcher(None, s1, s2).ratio()
    print('\n=== 判据 ===')
    print(f'两轮 summary 相似度: {sim:.0%}（改造前实测 46%~95%）')
    bad = []
    if not s1 or not s2:
        bad.append('有一轮没给出 summary')
    if sim > 0.62:
        bad.append(f'相似度 {sim:.0%} 偏高，时段剧本没有起作用')
    if out[0]['phase'] == out[1]['phase']:
        bad.append('两轮落在同一时段')
    # 尾盘那轮必须触及"过夜/空仓/明天"，开盘那轮不该抢这个话
    tail = out[1]['summary']
    if not any(k in tail for k in ('过夜', '空仓', '明天', '尾盘', '留仓', '收盘')):
        bad.append(f'尾盘轮的结论没有回答"过夜还是空仓"：{tail}')
    print('时间差异化:', 'OK' if not bad else 'FAIL')
    for x in bad:
        print('  ✗', x)
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
