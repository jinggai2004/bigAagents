// 对话层：**能查到的走真接口，查不到的如实说查不到**，绝不编造行情。
//
// 数据来源（全部只读，2026-10-27 接入）：
//   GET /api/picks/latest            今日选股（output/daily_picks/latest.json）
//   GET /api/watch/indices           四指数实时/收盘
//   GET /api/watch/watchlist?owner=  两套自选（user 你的 / ashen 我自己的）
//
// ⚠️ 历史教训：这个文件原来全是硬编码 mock —— 问"今天选到什么股"永远回答
//    `300750 宁德时代、601899 紫金矿业`，而 601899 根本不在当天的 latest.json 里。
//    看起来像"能查"，实际是在编，比明说"查不到"危险得多。
//
// 本文件**不依赖 three.js**（气泡已拆到 bubble.js），所以能在 node 里直接
// 跑真实 HTTP 做验收 —— 见 temp/domcheck/dialogue_test.mjs。

/** 取一个只读接口的 payload；任何失败都返回 null（调用方据此如实回答）。 */
async function api(path) {
  // companion 挂在 /companion/，接口在同源的 /api/* ⇒ 用根相对路径
  try {
    const r = await fetch('/api' + path, { cache: 'no-store' });
    if (!r.ok) return null;
    const b = await r.json();
    return b && b.ok ? b.data : null;
  } catch (e) {
    return null;                     // file:// 打开、断网、服务没起 都走这里
  }
}

const isFile = () => typeof location !== 'undefined' && location.protocol === 'file:';

function noApi(what) {
  return isFile()
    ? `我这边是直接打开的文件，连不上 API，看不到${what}。（从服务里打开 companion 我才能查。）`
    : `${what}现在取不到，我不瞎报。`;
}

export class Dialogue {
  constructor(getCtx) { this.getCtx = getCtx; }

  async respond(raw) {
    const text = raw.trim();
    const ctx = this.getCtx();
    const has = (...ws) => ws.some(w => text.includes(w));

    // ── 需要真数据的三类 ──
    if (has('选股', '选到', '推荐', '买什么', '有什么票')) return await this._picks();
    if (has('大盘', '行情', '市场怎么样', '沪指', '上证', '指数')) return await this._market();
    if (has('自选', '关注', '盯什么', '看什么票')) return await this._watch();

    // ── 纯本地（不需要数据，别去打扰接口）──
    if (has('你好', 'hi', '在吗', '嗨')) return pick(['你好呀，我是阿深，你的专属交易员。', '在的在的，盯盘呢（真的在盯）。']);
    if (has('在干嘛', '在做什么', '干什么呢')) return `我在「${ctx.stateLabel}」。${pick(ctx.lines)}`;
    if (has('复盘')) return '整篇复盘在「复盘」页。我这边只读得到当天的选股、行情和自选，读不到那篇文字。';
    if (has('几点', '时间', '现在几点')) return `现在是 ${ctx.time}，我正处于「${ctx.stateLabel}」状态。`;
    if (has('睡觉', '去睡')) return '好困……那我去眯一会儿，有事叫我。';
    if (has('跳舞', '才艺', '表演')) return '跳舞我不会，转圈给你看——（转椅.gif）';
    if (has('咖啡', '喝')) return '咕嘟咕嘟……这杯敬市场。';
    if (has('名字', '你是谁', '你叫什么', '介绍一下自己')) return '我叫阿深，DeepSeek 的蓝鲸化身，住在这间 12 平的小房间里打工。';
    if (has('谢谢', '辛苦')) return '不客气。敬畏市场，敬畏 K 线。';
    return '这个我还不会。可以问我「今天选到什么股」「大盘怎么样」「你自选里有什么」。';
  }

  async _picks() {
    const d = await api('/picks/latest');
    const picks = (d && d.picks) || [];
    if (!picks.length) {
      return d === null ? noApi('选股结果') : '今天还没有选股结果——要么还没跑，要么跑出来是空的。';
    }
    const list = picks.slice(0, 6).map(p => {
      const code = p['股票代码'] ?? p.code ?? '';
      const name = p['股票名称'] ?? p.name ?? '';
      return `${code} ${name}`.trim();
    }).join('、');
    return `今天选出 ${picks.length} 只：${list}${picks.length > 6 ? ' 等' : ''}。数据日期 ${d.date || '—'}。`;
  }

  async _market() {
    const d = await api('/watch/indices');
    const rows = (d && d.indices) || [];
    if (!rows.length) return noApi('行情');
    const parts = rows.map(x => {
      const px = x.price == null ? '—' : Number(x.price).toFixed(2);
      const pct = x.change_pct == null ? '—' : (x.change_pct >= 0 ? '+' : '') + Number(x.change_pct).toFixed(2) + '%';
      return `${x.name} ${px} ${pct}`;
    });
    const sh = rows.find(x => x.name === '上证指数');
    let mood = '';
    if (sh && sh.change_pct != null) {
      mood = sh.change_pct > 1 ? '红彤彤的，心情好。'
        : sh.change_pct > 0 ? '小涨，还行。'
        : sh.change_pct > -1 ? '小幅回调，不慌。' : '绿了……我去泡杯浓茶。';
    }
    // `live=false` 或 basis=close ⇒ 必须自报口径，否则会被当成实时价
    const real = d.live && rows.every(x => x.basis !== 'close');
    return `${parts.join('；')}。${mood}${real ? '' : '（这是收盘/本地价，不是实时报价）'}`;
  }

  async _watch() {
    const mine = await api('/watch/watchlist?owner=ashen');
    const yours = await api('/watch/watchlist?owner=user');
    if (!mine || !yours) return noApi('自选清单');
    const fmt = (d) => {
      const rows = (d.rows || []).slice(0, 5).map(r => `${r.code} ${r.name}`.trim());
      return rows.length ? rows.join('、') + (d.total > 5 ? ' 等' : '') : '（空）';
    };
    return `我自己 ${mine.total} 只：${fmt(mine)}。你那边 ${yours.total} 只：${fmt(yours)}。` +
      '你的清单我随时能看到，但我不会去改它。';
  }
}

function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }
