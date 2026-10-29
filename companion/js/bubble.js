// 头顶气泡：跟随角色头顶的屏幕坐标。
//
// 从 `dialogue.js` 拆出来（2026-10-27）：Dialogue 只是"文本进、文本出"，
// 不该被 three.js 绑架 —— 拆开后它能在 node 里直接跑真实 HTTP 做验收，
// 而不必为了测一句话去搭 WebGL 环境。
import * as THREE from 'three';

export class Bubble {
  constructor(el, camera) {
    this.el = el; this.camera = camera;
    this._timer = null; this._v = null;
  }
  say(text, anchorFn, ms = 5200) {
    this.el.textContent = text;
    this.el.style.display = 'block';
    this._anchorFn = anchorFn;
    clearTimeout(this._timer);
    this._timer = setTimeout(() => { this.el.style.display = 'none'; }, ms);
  }
  update() {
    if (this.el.style.display === 'none' || !this._anchorFn) return;
    if (!this._v) this._v = new THREE.Vector3();
    const p = this._anchorFn(this._v).project(this.camera);
    const x = (p.x * 0.5 + 0.5) * window.innerWidth;
    const y = (-p.y * 0.5 + 0.5) * window.innerHeight;
    this.el.style.left = `${x}px`; this.el.style.top = `${y - 46}px`;
  }
}
