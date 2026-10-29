// 迷你 K 线画布：给三块显示器当屏幕内容，走势受"模拟大盘涨跌"影响（红涨绿跌）
import * as THREE from 'three';

export class MarketScreen {
  constructor(width = 256, height = 160, candleCount = 24) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = width; this.canvas.height = height;
    this.ctx = this.canvas.getContext('2d');
    this.count = candleCount;
    this.price = 100;
    this.trend = 0;      // -1 ~ 1，外部根据情绪设定
    this.candles = [];
    for (let i = 0; i < candleCount; i++) this._next();
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this._acc = 0;
    this.active = true;
    this.redraw();
  }

  _next() {
    const drift = this.trend * 0.9;
    const open = this.price;
    const close = open + (Math.random() - 0.5 + drift) * 2.2;
    const high = Math.max(open, close) + Math.random() * 0.9;
    const low = Math.min(open, close) - Math.random() * 0.9;
    this.candles.push({ open, close, high, low });
    if (this.candles.length > this.count) this.candles.shift();
    this.price = Math.max(20, close);
  }

  update(dt) {
    if (!this.active) return;
    this._acc += dt;
    if (this._acc > 1.6) { this._acc = 0; this._next(); this.redraw(); }
  }

  setTrend(t) { this.trend = t; }

  redraw() {
    const { ctx, canvas, candles } = this;
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = '#101418'; ctx.fillRect(0, 0, W, H);
    let lo = Infinity, hi = -Infinity;
    for (const c of candles) { lo = Math.min(lo, c.low); hi = Math.max(hi, c.high); }
    const pad = (hi - lo) * 0.1 || 1; lo -= pad; hi += pad;
    const y = v => H - ((v - lo) / (hi - lo)) * (H - 16) - 8;
    const cw = W / this.count;
    candles.forEach((c, i) => {
      const up = c.close >= c.open;                     // 中国惯例：涨红跌绿
      const col = up ? '#E24B4A' : '#1D9E75';
      const x = i * cw + cw / 2;
      ctx.strokeStyle = col; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, y(c.high)); ctx.lineTo(x, y(c.low)); ctx.stroke();
      const top = y(Math.max(c.open, c.close)), bot = y(Math.min(c.open, c.close));
      ctx.fillStyle = col;
      ctx.fillRect(x - cw * 0.32, top, cw * 0.64, Math.max(2, bot - top));
    });
    this.texture.needsUpdate = true;
  }
}

// 终端风格屏幕（绿字黑底，滚动的伪代码/日志）
export class TerminalScreen {
  constructor(width = 256, height = 160) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = width; this.canvas.height = height;
    this.ctx = this.canvas.getContext('2d');
    this.lines = ['> SELECT * FROM market WHERE mood = calm', '> backfill: 300750 ok (0.8s)', '> factor ic = -0.03 (skip)', '> 敬畏市场', '> watch: 601313 zombie? skip'];
    this.offset = 0;
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this._acc = 0;
    this.active = true;
    this.redraw();
  }

  update(dt) {
    if (!this.active) return;
    this._acc += dt;
    if (this._acc > 0.9) { this._acc = 0; this.offset++; this.redraw(); }
  }

  redraw() {
    const { ctx, canvas, lines } = this;
    ctx.fillStyle = '#0A0F0C'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = '11px monospace'; ctx.fillStyle = '#5DCAA5';
    const visible = 9;
    for (let i = 0; i < visible; i++) {
      const idx = (this.offset + i) % lines.length;
      ctx.fillText(lines[idx], 8, 20 + i * 15);
    }
    this.texture.needsUpdate = true;
  }
}

// 游戏屏：小鲸鱼吃豆豆（下班打游戏）
export class GameScreen {
  constructor(width = 256, height = 160) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = width; this.canvas.height = height;
    this.ctx = this.canvas.getContext('2d');
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.wx = 40; this.dir = 1;
    this.dots = Array.from({ length: 6 }, (_, i) => ({ x: 30 + i * 36, y: 60 + (i % 3) * 22, eaten: false }));
    this.score = 9950;
    this._acc = 0;
    this.active = true;
    this.redraw();
  }

  update(dt) {
    if (!this.active) return;
    this._acc += dt;
    if (this._acc > 0.12) { this._acc = 0; this._tick(); this.redraw(); }
  }

  _tick() {
    this.wx += this.dir * 6;
    if (this.wx > 216 || this.wx < 40) { this.dir *= -1; this.score += 10; }
    for (const d of this.dots) {
      if (!d.eaten && Math.abs(d.x - this.wx) < 12) { d.eaten = true; this.score += 50; }
    }
    if (this.dots.every(d => d.eaten)) this.dots.forEach((d, i) => { d.eaten = false; d.x = 30 + i * 36; });
  }

  redraw() {
    const { ctx, canvas } = this;
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = '#101A2E'; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#1D2C4A'; ctx.fillRect(0, H - 26, W, 26);
    ctx.fillStyle = '#F7D154'; ctx.font = 'bold 12px monospace';
    ctx.fillText('SCORE ' + this.score, 8, 18);
    ctx.fillStyle = '#5F5E5A'; ctx.font = '10px monospace';
    ctx.fillText('鲸 鱼 大 作 战', 160, 18);
    for (const d of this.dots) {
      if (d.eaten) continue;
      ctx.fillStyle = '#F7D154';
      ctx.beginPath(); ctx.arc(d.x, d.y, 4, 0, Math.PI * 2); ctx.fill();
    }
    // 小鲸鱼（吃豆朝向）
    ctx.fillStyle = '#4A8FD9';
    ctx.beginPath(); ctx.ellipse(this.wx, 118, 16, 11, 0, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath();
    const tx = this.wx - this.dir * 16;
    ctx.moveTo(tx, 118); ctx.lineTo(tx - this.dir * 12, 108); ctx.lineTo(tx - this.dir * 12, 128);
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(this.wx + this.dir * 7, 114, 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#101A2E';
    ctx.beginPath(); ctx.arc(this.wx + this.dir * 8, 114, 1.5, 0, Math.PI * 2); ctx.fill();
    this.texture.needsUpdate = true;
  }
}

// 视频屏：模拟播放器（片名 + 色块画面 + 进度条）
export class VideoScreen {
  constructor(width = 256, height = 160) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = width; this.canvas.height = height;
    this.ctx = this.canvas.getContext('2d');
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.progress = 0.35;
    this.hue = 200;
    this._acc = 0;
    this.active = true;
    this.redraw();
  }

  update(dt) {
    if (!this.active) return;
    this._acc += dt;
    if (this._acc > 0.5) {
      this._acc = 0;
      this.progress += 0.004; if (this.progress > 0.97) this.progress = 0.05;
      this.hue = (this.hue + 8) % 360;
      this.redraw();
    }
  }

  redraw() {
    const { ctx, canvas } = this;
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = '#0D0D10'; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#1A1A20'; ctx.fillRect(0, 0, W, 18);
    ctx.fillStyle = '#E8E6E0'; ctx.font = '11px sans-serif';
    ctx.fillText('《鲸鱼也要上班》EP12', 8, 13);
    // 内容区：缓慢变色的"画面"
    ctx.fillStyle = `hsl(${this.hue}, 45%, 38%)`; ctx.fillRect(8, 24, W - 16, 96);
    ctx.fillStyle = `hsl(${(this.hue + 40) % 360}, 50%, 55%)`;
    ctx.beginPath(); ctx.arc(60 + (this.hue % 80), 66, 20, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.beginPath();
    ctx.moveTo(W / 2 - 8, 58); ctx.lineTo(W / 2 + 12, 72); ctx.lineTo(W / 2 - 8, 86);
    ctx.closePath(); ctx.fill();
    // 进度条
    ctx.fillStyle = '#2A2A32'; ctx.fillRect(8, H - 22, W - 16, 5);
    ctx.fillStyle = '#E24B4A'; ctx.fillRect(8, H - 22, (W - 16) * this.progress, 5);
    ctx.fillStyle = '#E8E6E0'; ctx.font = '9px monospace';
    ctx.fillText('19:02 / 45:00', 8, H - 6);
    this.texture.needsUpdate = true;
  }
}
