// 阿深正式模型：GLB 单网格 + GPU 伪骨骼（头/尾权重旋转）+ 企鹅步态 + 表情系统
// 与 WhaleGirl 占位体接口一致；加载失败时 main.js 回退到占位体
import * as THREE from 'three';
import { GLTFLoader } from '../vendor/GLTFLoader.js';

const TARGET_HEIGHT = 1.35;
const WALK_SPEED = 0.85;        // 单位/秒
const STRIDE = 0.22;            // 步幅
const TURN_RATE = 5.5;          // 原地转向角速度 rad/s

// 模型局部系（原始 GLB 单位）实测锚点
const NECK = new THREE.Vector3(0, 0.03, -0.10);
const TAIL_BASE = new THREE.Vector3(0, -0.12, -0.60);
const HEAD_Y = 0.146;           // 头心高度（原始单位）
const FACE_FRONT_Z = 0.0;       // 面部最前平面

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const sstep = (t) => t * t * (3 - 2 * t);

// ⚠️ 脸部叠加层默认关闭（2026-10-28）。
//    glb 的贴图**自带完整的眼睛、嘴巴和腮红**（`mesh.material.map` 为真；
//    可开 `companion/model_preview.html?layers=0` 看素模）。原来那两层是在
//    已经画好的脸上**再画一遍**：
//      · `facePlane` 用 #5A3A38 画一条嘴 —— 位置对不上模型的粉嘴，正面看像
//        "下巴上悬着一条深棕色的线"；眉毛更是画在刘海后面，根本看不见；
//      · `blushPlane` 是一整块 0.46×0.18 的粉色径向渐变，把鼻子和发际线
//        一起染粉 —— 用户反馈的"脸上有粉色的表情、很诡异"就是它。
//    现在改成"用模型自己的脸"，情绪靠姿态（点头/低头/歪头）+ 头顶符号表达。
//    要恢复旧行为把这里改成 true，但**先跑 model_preview 对照素模**，
//    确认线条真的对齐了五官再打开，别再盲改。
const FACE_OVERLAY = false;

// ---------------- 表情贴图画布 ----------------
function makeFaceTexture(draw) {
  const cv = document.createElement('canvas');
  cv.width = 128; cv.height = 96;
  const c = cv.getContext('2d');
  c.strokeStyle = '#5A3A38'; c.fillStyle = '#5A3A38';
  c.lineWidth = 5; c.lineCap = 'round';
  draw(c);
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

const FACES = {
  neutral: c => {                       // 浅微笑
    c.beginPath(); c.arc(64, 34, 16, Math.PI * 0.18, Math.PI * 0.82); c.stroke();
  },
  happy: c => {                          // 大笑（开口）
    c.beginPath(); c.arc(64, 30, 18, Math.PI * 0.12, Math.PI * 0.88);
    c.quadraticCurveTo(64, 68, 46, 36); c.fill();
    c.fillStyle = '#E8756D';
    c.beginPath(); c.ellipse(64, 44, 9, 6, 0, 0, Math.PI); c.fill();
  },
  sad: c => {                            // 难过（下撇嘴 + 八字眉）
    c.beginPath(); c.arc(64, 58, 14, Math.PI * 1.18, Math.PI * 1.82); c.stroke();
    c.lineWidth = 4;
    c.beginPath(); c.moveTo(22, 22); c.lineTo(44, 30); c.stroke();
    c.beginPath(); c.moveTo(106, 22); c.lineTo(84, 30); c.stroke();
  },
  focused: c => {                        // 认真（一字嘴 + 平眉）
    c.beginPath(); c.moveTo(48, 40); c.lineTo(80, 40); c.stroke();
    c.lineWidth = 4;
    c.beginPath(); c.moveTo(24, 24); c.lineTo(46, 24); c.stroke();
    c.beginPath(); c.moveTo(104, 24); c.lineTo(82, 24); c.stroke();
  },
  sleepy: c => {                         // 困（波浪嘴）
    c.lineWidth = 4;
    c.beginPath(); c.moveTo(48, 42);
    for (let i = 0; i <= 8; i++) c.lineTo(48 + i * 4, 42 + Math.sin(i * Math.PI / 4) * 3);
    c.stroke();
  },
  surprised: c => {                      // 惊讶（O 嘴 + 挑眉）
    c.beginPath(); c.ellipse(64, 42, 8, 11, 0, 0, Math.PI * 2); c.stroke();
    c.lineWidth = 4;
    c.beginPath(); c.moveTo(24, 20); c.quadraticCurveTo(35, 14, 46, 22); c.stroke();
    c.beginPath(); c.moveTo(104, 20); c.quadraticCurveTo(93, 14, 82, 22); c.stroke();
  },
};

// 头顶表情符号精灵
const EMO_TEXTURES = {};
function emoTexture(kind) {
  if (EMO_TEXTURES[kind]) return EMO_TEXTURES[kind];
  const cv = document.createElement('canvas');
  cv.width = cv.height = 64;
  const c = cv.getContext('2d');
  c.font = 'bold 44px sans-serif'; c.textAlign = 'center'; c.textBaseline = 'middle';
  const col = { note: '#F0B429', angry: '#F0483E', spark: '#2DD4BF', zzz: '#8B94A7', excl: '#4C9AFF' }[kind] || '#fff';
  c.fillStyle = col;
  const ch = { note: '♪', angry: '✸', spark: '✦', zzz: 'Z', excl: '!' }[kind] || '?';
  c.fillText(ch, 32, 34);
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  EMO_TEXTURES[kind] = t;
  return t;
}

export class ModelGirl {
  static async create(scene, url) {
    const gltf = await new GLTFLoader().loadAsync(url);
    return new ModelGirl(scene, gltf.scene);
  }

  constructor(scene, model) {
    this.root = new THREE.Group();      // 位置 + 朝向（状态机控制）
    this.inner = new THREE.Group();     // 身体层动画（沉浮/摇摆/呼吸）
    this.root.add(this.inner);
    this._rig(model);
    this._buildFace();
    this._buildEmoSprites();

    scene.add(this.root);
    this.mode = 'stand';
    this.mood = 'neutral';
    this.emotion = 'neutral';
    this._emotionTimer = 0;
    this.away = false;
    this._t = 0;

    // 步态状态
    this._walk = null;      // {from,to,faceFrom,dr,dist,dur,steps,t,phase}
    this._faceTarget = null;

    // 姿态（表情驱动，每帧向目标缓动）
    this._pose = { pitch: 0, yaw: 0, sink: 0, roll: 0 };
    this._poseTarget = { pitch: 0, yaw: 0, sink: 0, roll: 0 };
  }

  // ---------- 伪骨骼装配：权重属性 + shader 注入 ----------
  _rig(model) {
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const s = TARGET_HEIGHT / size.y;
    model.scale.setScalar(s);
    const box2 = new THREE.Box3().setFromObject(model);
    const c = box2.getCenter(new THREE.Vector3());
    model.position.set(-c.x, -box2.min.y, -c.z);
    this.inner.add(model);
    this.model = model;

    let mesh = null;
    model.traverse(o => { if (o.isMesh && !mesh) mesh = o; });
    this.mesh = mesh;
    const geo = mesh.geometry;
    const pos = geo.attributes.position;
    const n = pos.count;
    const headW = new Float32Array(n);
    const tailW = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const y = pos.getY(i), z = pos.getZ(i);
      // 头：颈部(0.02)以下不转，0.10 以上全转
      headW[i] = sstep(clamp((y - 0.02) / 0.08, 0, 1));
      // 尾：-0.60 起渐变，-0.82 全转
      tailW[i] = sstep(clamp((-0.60 - z) / 0.22, 0, 1));
    }
    geo.setAttribute('aHeadW', new THREE.BufferAttribute(headW, 1));
    geo.setAttribute('aTailW', new THREE.BufferAttribute(tailW, 1));

    const mat = mesh.material.clone();
    mesh.material = mat;
    this._tintables = mat.color ? [{ mat, base: mat.color.getHex() }] : [];
    const self = this;
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uHeadPitch = self._uHeadPitch = { value: 0 };
      shader.uniforms.uHeadYaw = self._uHeadYaw = { value: 0 };
      shader.uniforms.uTailYaw = self._uTailYaw = { value: 0 };
      shader.uniforms.uNeck = { value: NECK };
      shader.uniforms.uTailBase = { value: TAIL_BASE };
      shader.vertexShader = `
        attribute float aHeadW; attribute float aTailW;
        uniform float uHeadPitch, uHeadYaw, uTailYaw;
        uniform vec3 uNeck, uTailBase;
        vec3 asRotX(vec3 p, float a){ float c=cos(a), s=sin(a); return vec3(p.x, c*p.y-s*p.z, s*p.y+c*p.z); }
        vec3 asRotY(vec3 p, float a){ float c=cos(a), s=sin(a); return vec3(c*p.x+s*p.z, p.y, -s*p.x+c*p.z); }
      ` + shader.vertexShader.replace('#include <begin_vertex>', `
        #include <begin_vertex>
        if (aHeadW > 0.001) {
          vec3 p = transformed - uNeck;
          p = asRotY(p, uHeadYaw * aHeadW);
          p = asRotX(p, uHeadPitch * aHeadW);
          transformed = p + uNeck;
        }
        if (aTailW > 0.001) {
          vec3 p = transformed - uTailBase;
          p = asRotY(p, uTailYaw * aTailW);
          transformed = p + uTailBase;
        }
      `);
    };
    mat.customProgramCacheKey = () => 'ashen-rig1';
  }

  // ---------- 脸部表情贴片（嘴/眉画布，跟随头部旋转） ----------
  _buildFace() {
    const bb = new THREE.Box3().setFromObject(this.model);
    // 面部锚点：微调项，实测嘴部约在身高 66~70% 处
    const faceY = bb.min.y + (bb.max.y - bb.min.y) * 0.68;
    const faceZ = bb.max.z - 0.008;

    this.faceAnchor = new THREE.Group();
    this.faceAnchor.position.set(0, faceY, faceZ);
    this.inner.add(this.faceAnchor);

    this.facePlane = new THREE.Mesh(
      new THREE.PlaneGeometry(0.44, 0.32),
      new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false })
    );
    this.facePlane.renderOrder = 5;
    this.facePlane.position.z = 0.008;
    this.facePlane.visible = FACE_OVERLAY;      // 见文件头 FACE_OVERLAY 的说明
    this.faceAnchor.add(this.facePlane);
    this._faceTextures = {};
    this._setFace('neutral');

    // 脸红 overlay
    const blushTex = (() => {
      const cv = document.createElement('canvas'); cv.width = 128; cv.height = 48;
      const c = cv.getContext('2d');
      for (const x of [28, 100]) {
        const g = c.createRadialGradient(x, 24, 2, x, 24, 18);
        g.addColorStop(0, 'rgba(245,140,130,0.85)');
        g.addColorStop(1, 'rgba(245,140,130,0)');
        c.fillStyle = g; c.fillRect(0, 0, 128, 48);
      }
      const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t;
    })();
    this.blushPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(0.46, 0.18),
      new THREE.MeshBasicMaterial({ map: blushTex, transparent: true, depthWrite: false, opacity: 0.35 })
    );
    this.blushPlane.renderOrder = 4;
    this.blushPlane.position.set(0, 0.02, 0.006);
    this.blushPlane.visible = FACE_OVERLAY;     // 模型的贴图自带脸颊红，不必再糊一层
    this.faceAnchor.add(this.blushPlane);
  }

  _setFace(name) {
    // 表情名**始终记录**：姿态与头顶符号要靠它选型（即使不画脸）。
    const changed = this._currentFace !== name;
    this._currentFace = name;
    if (!changed) return;
    // 叠层关掉时不生成贴图（每张 128×96 的 canvas，6 个表情没必要白算）
    if (!FACE_OVERLAY || !this.facePlane) return;
    if (!this._faceTextures[name]) this._faceTextures[name] = makeFaceTexture(FACES[name] || FACES.neutral);
    this.facePlane.material.map = this._faceTextures[name];
    this.facePlane.material.needsUpdate = true;
  }

  // ---------- 头顶表情符号 ----------
  _buildEmoSprites() {
    this._emos = [];
    for (let i = 0; i < 3; i++) {
      const spr = new THREE.Sprite(new THREE.SpriteMaterial({
        map: emoTexture('note'), transparent: true, opacity: 0, depthWrite: false,
      }));
      spr.scale.setScalar(0.18);
      spr.position.set(0.15 + i * 0.12, TARGET_HEIGHT + 0.15, 0);
      this.inner.add(spr);
      this._emos.push({ spr, life: 0, vy: 0 });
    }
  }

  showEmo(kind, dur = 2.2) {
    const slot = this._emos.find(e => e.life <= 0) || this._emos[0];
    slot.spr.material.map = emoTexture(kind);
    slot.spr.material.needsUpdate = true;
    slot.life = dur; slot.vy = 0;
  }

  // ---------- 表情状态机 ----------
  setEmotion(e, dur = 0) {
    this.emotion = e;
    this._emotionTimer = dur;
    const P = this._poseTarget;
    switch (e) {
      case 'happy': P.pitch = -0.06; P.sink = 0; this._setFace('happy'); this.showEmo('note', 2.5); break;
      case 'laugh': P.pitch = -0.12; P.sink = 0; this._setFace('happy'); this.showEmo('note', 3); break;
      case 'sad': P.pitch = 0.22; P.sink = -0.06; this._setFace('sad'); this.showEmo('angry', 2.5); break;
      case 'focused': P.pitch = 0.10; P.sink = 0; this._setFace('focused'); break;
      case 'sleepy': P.pitch = 0.28; P.sink = -0.03; this._setFace('sleepy'); this.showEmo('zzz', 3); break;
      case 'surprised': P.pitch = -0.14; P.sink = 0; this._setFace('surprised'); this.showEmo('excl', 1.6); break;
      default: P.pitch = 0; P.sink = 0; this._setFace('neutral');
    }
  }

  // 由外部（心情/模式）推导基础表情
  syncEmotion() {
    if (this._emotionTimer > 0) return;
    if (this.mode === 'sleep') { this.setEmotion('sleepy'); return; }
    if (this.mode === 'type') { this.setEmotion('focused'); return; }
    this.setEmotion(this.mood === 'happy' ? 'happy' : this.mood === 'sad' ? 'sad' : 'neutral');
  }

  // ---------- 移动 ----------
  goTo(spot) {
    const [x, y = 0, z] = spot.pos;
    const from = this.root.position.clone();
    const to = new THREE.Vector3(x, y, z);
    const dist = from.distanceTo(to);
    let dr = spot.face - this.root.rotation.y;
    while (dr > Math.PI) dr -= Math.PI * 2; while (dr < -Math.PI) dr += Math.PI * 2;
    this._walk = {
      from, to, faceFrom: this.root.rotation.y, dr,
      dist, dur: Math.max(0.25, dist / WALK_SPEED),
      steps: Math.max(2, Math.round(dist / STRIDE) * 2),
      t: 0, phase: Math.abs(dr) > 0.5 ? 'turn' : 'walk',
    };
  }

  disappear() { this.away = true; this.root.visible = false; }
  stop() { this._walk = null; this.inner.rotation.x = 0; this.inner.rotation.z = 0; }   // 打断当前行走
  appearAt(spot) {
    this.away = false;
    this.root.visible = true;
    this.root.position.set(spot.pos[0], spot.pos[1] || 0, spot.pos[2]);
    this.root.rotation.y = spot.face;
    this._walk = null;
  }

  setMode(mode) {
    if (this.mode !== mode) { this.mode = mode; this.syncEmotion(); }
  }
  setMood(mood) {
    this.mood = mood;
    this.syncEmotion();
    for (const t of this._tintables) {
      if (mood === 'sad') t.mat.color.setHex(t.base).lerp(new THREE.Color(0x6E7688), 0.35);
      else t.mat.color.setHex(t.base);
    }
  }

  faceCamera(cameraPos) {
    const d = cameraPos.clone().sub(this.root.position);
    this._faceTarget = Math.atan2(d.x, d.z);
  }
  clearFace() { this._faceTarget = null; }

  headWorld(v) {
    return v.set(this.root.position.x, this.root.position.y + TARGET_HEIGHT * 0.86, this.root.position.z);
  }

  // ---------- 主更新 ----------
  update(dt) {
    this._t += dt;
    const t = this._t;

    if (this._emotionTimer > 0) {
      this._emotionTimer -= dt;
      if (this._emotionTimer <= 0) this.syncEmotion();
    }

    // 姿态向目标缓动
    const k = Math.min(1, dt * 6);
    for (const key of ['pitch', 'yaw', 'sink', 'roll']) {
      this._pose[key] += (this._poseTarget[key] - this._pose[key]) * k;
    }
    if (this._uHeadPitch) {
      this._uHeadPitch.value = this._pose.pitch + Math.sin(t * 1.3) * 0.02;
      this._uHeadYaw.value = this._pose.yaw + Math.sin(t * 0.9) * 0.03;
      this._uTailYaw.value = Math.sin(t * 2.1) * 0.16 + this._pose.roll * 0.5;
    }
    // 表情贴片跟随头部俯仰/偏航
    if (this.faceAnchor) {
      this.faceAnchor.rotation.x = this._pose.pitch * 0.9;
      this.faceAnchor.rotation.y = this._pose.yaw * 0.9;
    }
    // 脸红随表情（叠层关闭时整块不显示，不必每帧算插值）
    if (this.blushPlane.visible) {
      const bo = this.blushPlane.material;
      const bt = (this.emotion === 'happy' || this.emotion === 'laugh') ? 0.9 : this.emotion === 'neutral' ? 0.35 : 0;
      bo.opacity += (bt - bo.opacity) * k;
    }

    // 头顶符号上浮
    for (const e of this._emos) {
      if (e.life > 0) {
        e.life -= dt; e.vy += dt * 0.12;
        e.spr.position.y = TARGET_HEIGHT + 0.15 + e.vy;
        e.spr.material.opacity = Math.min(1, e.life);
      } else {
        e.spr.material.opacity = 0;
      }
    }

    // ---------- 步态 ----------
    if (this._walk) {
      const w = this._walk;
      if (w.phase === 'turn') {
        // 先原地转身，再起步（消除边滑边转）
        const remaining = w.dr - (this.root.rotation.y - w.faceFrom);
        const step = TURN_RATE * dt * Math.sign(w.dr || 1);
        if (Math.abs(remaining) <= Math.abs(step)) {
          this.root.rotation.y = w.faceFrom + w.dr;
          w.phase = 'walk'; w.t = 0;
        } else {
          this.root.rotation.y += step;
        }
      } else if (w.phase === 'walk') {
        w.t += dt / w.dur;
        const k2 = Math.min(1, w.t);
        const e = sstep(k2);   // 起步/停步速度缓冲
        this.root.position.lerpVectors(w.from, w.to, e);
        // 企鹅步：摇摆 + 颠簸 + 前倾，频率与步数锁定
        const ph = k2 * w.steps * Math.PI;
        this.root.position.y = w.to.y + Math.abs(Math.sin(ph)) * 0.035;
        this.inner.rotation.z = Math.sin(ph) * 0.075;
        this.inner.rotation.x = 0.07;
        if (this._uTailYaw) this._uTailYaw.value += Math.sin(ph) * 0.1;
        if (k2 >= 1) { w.phase = 'settle'; w.t = 0; w.dur = 0.18; }
      } else { // settle：停步回正
        w.t += dt / w.dur;
        const k3 = Math.min(1, w.t);
        this.inner.rotation.z *= (1 - k3);
        this.inner.rotation.x *= (1 - k3);
        this.root.position.y = w.to.y;
        if (k3 >= 1) { this._walk = null; this.inner.rotation.z = 0; this.inner.rotation.x = 0; }
      }
    } else if (this._faceTarget !== null) {
      let dr = this._faceTarget - this.root.rotation.y;
      while (dr > Math.PI) dr -= Math.PI * 2; while (dr < -Math.PI) dr += Math.PI * 2;
      this.root.rotation.y += dr * Math.min(1, dt * 8);
    }

    // ---------- 模式姿态（无步态时） ----------
    if (!this._walk) {
      this.inner.rotation.x = 0;
      if (this.mode === 'sleep') {
        this.root.rotation.x = -Math.PI / 2 * 0.92;
        this.root.position.y = 0.42 + this._pose.sink;
        const br = 1 + Math.sin(t * 1.2) * 0.02;
        this.inner.scale.set(br, br, br);
        this.inner.position.y = 0;
      } else {
        this.root.rotation.x = 0;
        this.inner.scale.set(1, 1, 1);
        if (this.mode === 'type') {
          this.inner.position.y = -0.12 + this._pose.sink + Math.sin(t * 9) * 0.012;
        } else if (this.mode === 'sit') {
          this.inner.position.y = -0.04 + this._pose.sink;
        } else if (this.mode === 'wave') {
          this.inner.position.y = this._pose.sink + Math.abs(Math.sin(t * 6)) * 0.06;
          this.inner.rotation.z = Math.sin(t * 6) * 0.14;
        } else {
          this.inner.position.y = Math.sin(t * 2) * 0.02 + this._pose.sink;
        }
      }
    }

    // 开心待机时偶尔小跳
    if (!this._walk && this.mode === 'stand' && (this.emotion === 'happy' || this.emotion === 'laugh')) {
      if (Math.sin(t * 3.2) > 0.985) this.inner.position.y += 0.05;
    }
  }
}
