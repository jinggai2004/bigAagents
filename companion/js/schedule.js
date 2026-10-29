// 作息状态机：真实时钟驱动 + 模拟时钟（时间快进调试）
// P0 交易历为 mock：周一~周五 = trading，周末 = nontrading，可手动切换

export class Schedule {
  constructor(config) {
    this.cfg = config;
    this.state = null;
    this.listeners = [];
    // 模拟时钟
    this.simDate = new Date();          // 当前模拟时间
    this.speed = 1;                     // 倍速（1 = 真实）
    this.paused = false;
    this.forceDayType = null;           // null | 'trading' | 'nontrading'
    this._lastReal = performance.now();
  }

  dayType(date) {
    if (this.forceDayType) return this.forceDayType;
    const d = date.getDay();
    return (d >= 1 && d <= 5) ? 'trading' : 'nontrading';
  }

  minutesOf(date) {
    return date.getHours() * 60 + date.getMinutes() + date.getSeconds() / 60;
  }

  resolve(date) {
    const m = this.minutesOf(date);
    const dt = this.dayType(date);
    for (const row of this.cfg.timetable) {
      if (row.days !== 'all' && row.days !== dt) continue;
      const [sh, sm] = row.start.split(':').map(Number);
      const [eh, em] = row.end.split(':').map(Number);
      const s = sh * 60 + sm, e = eh * 60 + em;
      // 支持跨午夜（如 22:00 ~ 次日 08:30 写作 22:00~29:59）
      if (s <= m && m < e) return this.cfg.states[row.state];
      if (e > 24 * 60 && m + 24 * 60 < e && s <= m + 24 * 60) return this.cfg.states[row.state];
    }
    return this.cfg.states.offduty;
  }

  onChange(fn) { this.listeners.push(fn); }

  update() {
    const now = performance.now();
    if (!this.paused && this.speed > 0) {
      this.simDate = new Date(this.simDate.getTime() + (now - this._lastReal) * this.speed);
    }
    this._lastReal = now;
    const next = this.resolve(this.simDate);
    if (next !== this.state) {
      const prev = this.state;
      this.state = next;
      for (const fn of this.listeners) fn(next, prev);
    }
  }

  setTime(h, m) {
    const d = new Date(this.simDate);
    d.setHours(h, m, 0, 0);
    this.simDate = d;
    this.state = null; // 强制下次 update 重新解析并触发
  }

  hhmm() {
    const p = n => String(n).padStart(2, '0');
    return `${p(this.simDate.getHours())}:${p(this.simDate.getMinutes())}`;
  }
}
