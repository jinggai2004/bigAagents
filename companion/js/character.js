// 阿深占位体：Three.js 基元拼装的 Q 版鲸娘（P0 验证状态机与动画接线，P1 换正式模型）
import * as THREE from 'three';

const HAIR = 0x4A6FA5, DRESS = 0x2B3A67, APRON = 0xF5F2EC, SKIN = 0xFFEFE6;

function lam(color) { return new THREE.MeshLambertMaterial({ color }); }

// 腮红贴图：径向渐变，中心柔粉、边缘 alpha=0。
// ⚠️ 用贴图而不是"压扁的球"：球有硬轮廓，正面看是一块粉色圆饼；
//    渐变的边缘 alpha 归零后和肤色自然过渡，才像"脸上透出来的一点血色"。
//    只建一次、所有角色共用（每次 new 一个 64×64 canvas 是没必要的开销）。
let _BLUSH_TEX = null;
function _blushTexture() {
  if (_BLUSH_TEX) return _BLUSH_TEX;
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const rg = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  rg.addColorStop(0.00, 'rgba(246,142,132,0.62)');
  rg.addColorStop(0.45, 'rgba(246,150,140,0.30)');
  rg.addColorStop(0.75, 'rgba(246,158,148,0.09)');
  rg.addColorStop(1.00, 'rgba(246,160,150,0)');
  g.fillStyle = rg;
  g.fillRect(0, 0, 64, 64);
  _BLUSH_TEX = new THREE.CanvasTexture(c);
  return _BLUSH_TEX;
}

export class WhaleGirl {
  constructor(scene) {
    this.root = new THREE.Group();
    this.body = new THREE.Group();
    this.root.add(this.body);
    // 注意：_tintables 必须在 _build() 之前初始化（_build 里的 _tint 依赖它）
    this._tintables = [];
    // 五官引用：眨眼/表情在 update 里要按 mood 改，不能每帧重建几何
    this.brows = [];        // [{mesh, side, base}] 眉（表情主要靠它）
    this.blushes = [];      // 腮红贴片（Canvas 渐变，边缘透明）
    this.lids = [];         // 上眼睑弧线（不随眨眼压扁，闭眼时留眼线）
    this._eyeOpen = 1;      // 当前表情决定的睁眼度（0~1），乘到眨眼里
    this._walk = null;
    this._t = 0;
    this._blink = 0;
    this._faceTarget = null;
    this.mode = 'stand';
    this.mood = 'neutral';
    this.away = false;          // 外出中（不在房间）
    this._build();
    scene.add(this.root);
  }

  _tint(mat, base) { this._tintables.push({ mat, base }); }

  _build() {
    // 女仆裙：上身为胶囊，下摆为展开的裙摆圆台 + 白色裙边
    const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.24, 0.18, 6, 14), lam(DRESS));
    torso.position.y = 0.62; this.body.add(torso); this._tint(torso.material, DRESS);
    const skirt = new THREE.Mesh(new THREE.CylinderGeometry(0.23, 0.42, 0.4, 18), lam(DRESS));
    skirt.position.y = 0.32; this.body.add(skirt); this._tint(skirt.material, DRESS);
    const hem = new THREE.Mesh(new THREE.TorusGeometry(0.41, 0.035, 8, 22), lam(APRON));
    hem.rotation.x = Math.PI / 2; hem.position.y = 0.14; this.body.add(hem);
    // 白色围裙 + 领口 + 纽扣
    const apron = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.26, 0.05), lam(APRON));
    apron.position.set(0, 0.34, 0.3); this.body.add(apron);
    const collar = new THREE.Mesh(new THREE.TorusGeometry(0.16, 0.04, 8, 16), lam(APRON));
    collar.rotation.x = Math.PI / 2 - 0.15; collar.position.y = 0.8; this.body.add(collar);
    for (let i = 0; i < 3; i++) {
      const btn = new THREE.Mesh(new THREE.SphereGeometry(0.025, 8, 6), lam(APRON));
      btn.position.set(0, 0.68 - i * 0.11, 0.245); this.body.add(btn);
    }
    // 白袜 + 小鞋
    for (const s of [-1, 1]) {
      const sock = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.065, 0.1, 10), lam(APRON));
      sock.position.set(0.13 * s, 0.12, 0.04); this.body.add(sock);
      const shoe = new THREE.Mesh(new THREE.SphereGeometry(0.09, 10, 8), lam(0x22304F));
      shoe.scale.set(1, 0.6, 1.3); shoe.position.set(0.13 * s, 0.06, 0.06);
      this.body.add(shoe);
    }
    // 手臂（肩部枢轴）
    this.arms = [];
    for (const s of [-1, 1]) {
      const pivot = new THREE.Group();
      pivot.position.set(0.30 * s, 0.62, 0);
      const arm = new THREE.Mesh(new THREE.CapsuleGeometry(0.07, 0.2, 4, 8), lam(APRON));
      arm.position.y = -0.14; pivot.add(arm);
      this.body.add(pivot); this.arms.push(pivot);
    }
    // 鲸鱼尾（两片尾鳍 + 浅色腹面）
    this.tail = new THREE.Group();
    this.tail.position.set(0, 0.3, -0.3);
    for (const s of [-1, 1]) {
      const fluke = new THREE.Mesh(new THREE.SphereGeometry(0.19, 10, 8), lam(HAIR));
      fluke.scale.set(1.5, 0.32, 0.75);
      fluke.position.set(0.17 * s, 0.05, -0.06);
      fluke.rotation.z = 0.5 * s;
      this.tail.add(fluke); this._tint(fluke.material, HAIR);
    }
    const belly = new THREE.Mesh(new THREE.SphereGeometry(0.13, 10, 8), lam(0xA8C4E8));
    belly.scale.set(1.3, 0.5, 1); belly.position.set(0, -0.03, 0.02);
    this.tail.add(belly);
    this.body.add(this.tail);

    // 头
    const head = new THREE.Group(); head.position.y = 1.0; this.body.add(head); this.head = head;
    const face = new THREE.Mesh(new THREE.SphereGeometry(0.40, 24, 18), lam(SKIN));
    face.position.z = 0.06; head.add(face);
    const hair = new THREE.Mesh(new THREE.SphereGeometry(0.44, 24, 18), lam(HAIR));
    hair.position.set(0, 0.03, -0.06); head.add(hair); this._tint(hair.material, HAIR);
    // 后发绺（两侧垂下）
    for (const s of [-1, 1]) {
      const lock = new THREE.Mesh(new THREE.CapsuleGeometry(0.10, 0.34, 4, 8), lam(HAIR));
      lock.position.set(0.38 * s, -0.28, -0.08); lock.rotation.z = 0.18 * s;
      head.add(lock); this._tint(lock.material, HAIR);
    }
    // 刘海（三绺，遮住额头）
    for (const [bx, by, bz, r] of [[-0.2, 0.3, 0.33, 0.14], [0, 0.26, 0.37, 0.15], [0.2, 0.3, 0.33, 0.14]]) {
      const bang = new THREE.Mesh(new THREE.SphereGeometry(r, 10, 8), lam(HAIR));
      bang.scale.set(1, 1.25, 0.7); bang.position.set(bx, by, bz);
      head.add(bang); this._tint(bang.material, HAIR);
    }
    // 五官：眼睛 / 眉毛 / 腮红 / 嘴（见 _buildFace 里的注释，别再退回"贴片球"）
    this.eyes = [];
    this._buildFace(head);
    // 鲸鳍耳
    for (const s of [-1, 1]) {
      const fin = new THREE.Mesh(new THREE.ConeGeometry(0.13, 0.3, 4), lam(HAIR));
      fin.scale.z = 0.45; fin.position.set(0.44 * s, 0.06, -0.02);
      fin.rotation.z = s * -1.9; head.add(fin); this._tint(fin.material, HAIR);
    }
    // 女仆发箍（白环 + 褶球）
    const band = new THREE.Mesh(new THREE.TorusGeometry(0.3, 0.045, 8, 20), lam(APRON));
    band.rotation.x = Math.PI / 2 - 0.25; band.position.set(0, 0.32, 0.02); head.add(band);
    for (let i = 0; i < 6; i++) {
      const a = (i / 5 - 0.5) * 1.8;
      const puff = new THREE.Mesh(new THREE.SphereGeometry(0.06, 8, 6), lam(APRON));
      puff.position.set(Math.sin(a) * 0.3, 0.32 + Math.cos(a) * 0.1, 0.02 + Math.cos(a) * 0.03);
      head.add(puff);
    }
    // 蝴蝶结
    const bowL = new THREE.Mesh(new THREE.ConeGeometry(0.07, 0.14, 4), lam(0x7FA8D9));
    bowL.rotation.z = Math.PI / 2; bowL.position.set(0.36, 0.34, 0.08); head.add(bowL);
    const bowR = bowL.clone(); bowR.rotation.z = -Math.PI / 2; bowR.position.x = 0.50; head.add(bowR);
    // 呆毛
    const curve = new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(0, 0.44, 0), new THREE.Vector3(0.05, 0.62, 0), new THREE.Vector3(0.2, 0.66, -0.05));
    this.ahoge = new THREE.Mesh(new THREE.TubeGeometry(curve, 8, 0.018, 6), lam(HAIR));
    head.add(this.ahoge);
  }

  // ── 五官 ───────────────────────────────────────────────────────
  // ⚠️ 为什么重做（2026-10-28）：原来是「0.065 实心球当眼睛 + 0.045 球压扁当腮红
  //    贴在 z=0.38」。而脸球心在 z=0.06、半径 0.40 ⇒ 脸面在 z≈0.4286，
  //    于是**腮红浮在脸外**，正面看是一块硬边的粉色圆饼（用户反馈"脸上有粉色的
  //    表情，很诡异"）；眼球也前伸到 0.485，像两颗珠子粘上去。
  //    这里改成按球面法线摆放 + 分层：
  //      眼白（压扁球，嵌进脸）/ 虹膜 / 瞳孔 / 双高光 —— 有神靠的是双高光
  //      上眼睑（独立弧线，眨眼压扁时不跟着变，闭眼还有一条眼线）
  //      眉毛（表情主要靠它，mood 时改倾角）
  //      腮红（Canvas 径向渐变贴片，边缘 alpha 归零 ⇒ 没有硬边）
  //      嘴（半圆环，rotation.z 翻出笑/哭，neutral 时压平成浅弧）
  _buildFace(head) {
    const FC = new THREE.Vector3(0, 0, 0.06);   // 脸球心（head 局部）
    const R = 0.40;
    // 脸上某点：dx/dy 是相对脸心的偏移，返回球面坐标（沿 +z 内嵌 inset 防 z-fighting）
    const onFace = (dx, dy, inset = 0.005) => {
      const r2 = Math.max(0.0009, R * R - dx * dx - dy * dy);
      return new THREE.Vector3(FC.x + dx, FC.y + dy, FC.z + Math.sqrt(r2) - inset);
    };
    const normalAt = (p) => p.clone().sub(FC).normalize();

    // 眼位下移（y 0.048→0.012）：原来眼睛压在脸上半部，下面留出一大片空白，
    // 嘴离得老远 —— 比例不对是"诡异"的主要来源，不是贴图精度。
    // 眉毛跟着下来到 0.095：刘海（三绺球，下沿到 y≈0.11）原本把眉整个盖住，
    // 等于表情少了最关键的器官。
    for (const s of [-1, 1]) {
      const p = onFace(0.140 * s, 0.012, 0.004);
      const n = normalAt(p);
      const eye = new THREE.Group();
      eye.position.copy(p);
      // ⚠️ 必须朝法线：脸在这个位置的切线偏了约 20°，不转的话虹膜会整体
      //    歪向脸颊外侧（实测截图：两只眼的瞳孔都往外飘），看着像斜视。
      eye.lookAt(p.clone().add(n));

      const white = new THREE.Mesh(new THREE.SphereGeometry(0.062, 18, 14),
        new THREE.MeshLambertMaterial({ color: 0xFBFAF7 }));
      white.scale.set(1, 1.08, 0.38);            // 压扁 ⇒ 只露前面，不像珠子
      eye.add(white);

      const iris = new THREE.Mesh(new THREE.CircleGeometry(0.046, 20),
        new THREE.MeshBasicMaterial({ color: 0x2B4671 }));
      iris.position.z = 0.0235; eye.add(iris);

      const pupil = new THREE.Mesh(new THREE.CircleGeometry(0.022, 14),
        new THREE.MeshBasicMaterial({ color: 0x0E1729 }));
      pupil.position.z = 0.0248; eye.add(pupil);

      // 双高光：一大一小、位置错开。单点高光看起来像"塑料珠子"，
      // 两点（左上大 + 右下小）才让人读出"这双眼睛在看东西"。
      const hl1 = new THREE.Mesh(new THREE.CircleGeometry(0.0130, 10),
        new THREE.MeshBasicMaterial({ color: 0xFFFFFF }));
      hl1.position.set(-0.013, 0.014, 0.0262); eye.add(hl1);
      const hl2 = new THREE.Mesh(new THREE.CircleGeometry(0.0062, 8),
        new THREE.MeshBasicMaterial({ color: 0xFFFFFF, transparent: true, opacity: 0.8 }));
      hl2.position.set(0.011, -0.011, 0.0262); eye.add(hl2);

      head.add(eye);
      this.eyes.push(eye);

      // 上眼睑：单独一个 wrapper 做朝向，**不放进 eye 组**——眼组眨眼时会
      // scale.y 压扁，眼睑跟着压就变成"整只眼消失"，闭眼时该留下一条眼线。
      const lidWrap = new THREE.Group();
      lidWrap.position.copy(p);
      lidWrap.lookAt(p.clone().add(n));
      lidWrap.userData.baseY = p.y;              // 表情改"眼睑下压"时要回到原位
      const lid = new THREE.Mesh(new THREE.TorusGeometry(0.060, 0.0068, 6, 18, Math.PI * 0.92),
        new THREE.MeshBasicMaterial({ color: 0x2A2F44 }));
      lid.position.z = 0.004;
      // Torus 弧从 +X 逆时针扫 0.92π（0°→165.6°，中心 82.8°）。
      // 只需再转 +0.04π 让弧心落在正上方 —— 曾误写成 0.54π（多转半圈），
      // 弧跑到下半圈变成两边的 C，把眼睛夹成一条缝。
      lid.rotation.z = Math.PI * 0.04;
      lidWrap.add(lid);
      head.add(lidWrap);
      this.lids.push(lidWrap);

      // 眉毛：外层 wrapper 定朝向，内层 rotation.z 表达情绪。
      // ⚠️ 不能直接对眉毛 lookAt —— 那会把 rotation.z 覆盖掉，眉形和情绪都没了。
      const bp = onFace(0.118 * s, 0.086, 0.002);
      const bn = normalAt(bp);
      const browWrap = new THREE.Group();
      browWrap.position.copy(bp);
      browWrap.lookAt(bp.clone().add(bn));
      const brow = new THREE.Mesh(new THREE.TorusGeometry(0.056, 0.0080, 5, 12, Math.PI * 0.62),
        new THREE.MeshBasicMaterial({ color: 0x3B4260 }));
      // ⚠️ 两侧必须**镜像**，不能给同一个 rotation.z。
      //    `browWrap.lookAt()` 对左右两点给出的是**同一个世界朝向**（不是镜像），
      //    因此同一条弧在两侧的"内/外端"映射是反的：写 `π*0.07 - 0.10*s`
      //    会让左眉朝外斜、右眉朝内斜 —— 视觉上就是"单侧挑眉"，四种心情都一样。
      //    实测（世界坐标下"内端比外端高多少"）：
      //      基线 左 -0.0247 / 右 +0.0415（异号）→ 左侧 scale.x=-1 + rz 取反后
      //      两侧同为 +0.0333，差 0。**只看 local rotation.z 是看不出来的。**
      const BROW_BASE = Math.PI * 0.07;
      brow.rotation.z = s < 0 ? -BROW_BASE : BROW_BASE;
      brow.scale.x = s < 0 ? -1 : 1;
      browWrap.add(brow);
      head.add(browWrap);
      this.brows.push({ mesh: brow, side: s, base: brow.rotation.z });

      // 腮红：Canvas 径向渐变，边缘 alpha 归零 ⇒ 和脸自然融合，没有圆边
      const cp = onFace(0.216 * s, -0.072, 0.001);
      const blush = new THREE.Mesh(new THREE.PlaneGeometry(0.15, 0.095),
        new THREE.MeshBasicMaterial({ map: _blushTexture(), transparent: true,
                                      depthWrite: false, opacity: 0.85 }));
      blush.position.copy(cp);
      blush.lookAt(cp.clone().add(normalAt(cp)));
      head.add(blush);
      this.blushes.push(blush);
    }

    // 嘴：半圆环。rotation.z=0 ⇒ 弧在上（∩ 笑）；=π ⇒ 弧在下（∪ 委屈）。
    // neutral 用 scale.y 压成一条浅弧——不要用另一套几何，那会让三种嘴型
    // 在不同光照下宽度不一致（贴图/几何不一致是"表情跳变"的常见来源）。
    const mp = onFace(0, -0.132, 0.004);
    const mouth = new THREE.Mesh(new THREE.TorusGeometry(0.048, 0.0105, 6, 20, Math.PI),
      new THREE.MeshBasicMaterial({ color: 0xA5534A }));
    mouth.scale.set(1, 0.32, 1);
    const mouthWrap = new THREE.Group();
    mouthWrap.position.copy(mp);
    mouthWrap.lookAt(mp.clone().add(normalAt(mp)));
    mouthWrap.add(mouth);
    head.add(mouthWrap);
    this.mouth = mouth;
  }

  // 表情：改眉毛倾角 + 嘴型 + 睁眼度。几何不变，只动 transform ⇒ 无分配、可每帧调。
  _applyMoodFace() {
    const m = this.mood;
    // 眉：sad 内侧上抬（八字眉）、happy 略平、think 一高一低
    const tilt = m === 'sad' ? 0.20 : m === 'happy' ? -0.06 : m === 'think' ? 0.14 : 0;
    for (const b of this.brows) {
      // `base` 已带镜像符号（左眉 scale.x=-1 + rz 取反），但**旋转量的世界方向
      // 也被负缩放镜像了** ⇒ 情绪倾斜仍要乘 `side` 才能两侧同向。
      // 实测（世界坐标"内端比外端高多少"）：
      //   `base - tilt`（不乘 side）      → sad 左 0.016 / 右 0.049（反向，是错的）
      //   `base - tilt * side`（本行）    → sad 左 0.049 / 右 0.049（对称）
      // 也就是说：**base 要镜像、tilt 不能镜像**，两者缺一都会歪。
      b.mesh.rotation.z = b.base - tilt * b.side;
      b.mesh.position.y = (m === 'sad' ? 0.004 : 0) + (b.side > 0 && m === 'think' ? 0.012 : 0);
    }
    const mo = this.mouth;
    // 弧向：Torus 的 arc 从 +X 逆时针扫过 +Y ⇒ rotation.z=0 是「⌢」（中间高、
    // 两端下垂 = 不悦），=π 才是「⌣」（两端上翘 = 笑）。
    // ⚠️ 这里曾经写反：happy 显示成皱眉嘴、sad 显示成笑脸。代码读起来完全正常，
    //    是把四种表情拍成图并排比对才发现的——**表情这种事必须看图验证**。
    if (m === 'happy') { mo.rotation.z = Math.PI; mo.scale.set(1.15, 1, 1); }
    else if (m === 'sad') { mo.rotation.z = 0; mo.scale.set(0.95, 0.8, 1); }
    else if (m === 'think') { mo.rotation.z = Math.PI * 0.88; mo.scale.set(0.5, 0.55, 1); }
    // 默认：极浅的微上扬（笑意的底子），压扁 0.3 ⇒ 几乎是一条平线
    else { mo.rotation.z = Math.PI; mo.scale.set(1, 0.3, 1); }

    this._eyeOpen = m === 'happy' ? 0.66 : m === 'sad' ? 0.86 : m === 'think' ? 0.92 : 1;
    // 眯眼靠"上眼睑下压"，不是把整只眼球 scale.y 压扁 ——
    // 压扁会把虹膜和高光一起压成横条，看着像眼睛没了（第一版就是这样）。
    const lidDrop = m === 'happy' ? -0.011 : m === 'sad' ? 0.005 : 0;
    for (const w of this.lids) w.position.y = (w.userData.baseY || 0) + lidDrop;
    // 腮红：开心时红一点（把"情绪"落到脸上，而不是只有嘴在动）
    const op = m === 'happy' ? 1 : m === 'sad' ? 0.5 : 0.85;
    for (const b of this.blushes) b.material.opacity = op;
  }

  // 走到某个点并面向某方向（直线 + 小跳，P0 简化；spot.pos[1] 为座位高度）
  goTo(spot) {
    const [x, y = 0, z] = spot.pos;
    this._walk = {
      from: this.root.position.clone(),
      to: new THREE.Vector3(x, y, z),
      faceFrom: this.root.rotation.y,
      faceTo: spot.face,
      t: 0,
    };
  }

  // 出门：走到门口后由 main 调用 disappear
  disappear() { this.away = true; this.root.visible = false; }
  stop() { this._walk = null; }   // 打断当前行走
  appearAt(spot) {
    this.away = false; this._pendingSpot = null;
    this.root.visible = true;
    this.root.position.set(spot.pos[0], 0, spot.pos[2]);
    this.root.rotation.y = spot.face;
  }

  setMode(mode) { this.mode = mode; }
  setMood(mood) {
    this.mood = mood;
    for (const t of this._tintables) {
      if (mood === 'sad') t.mat.color.setHex(t.base).lerp(new THREE.Color(0x6E7688), 0.55);
      else t.mat.color.setHex(t.base);
    }
    // 表情落在五官上（眉/嘴/睁眼度/腮红浓度），不只是改衣服颜色。
    // ⚠️ 原来 setMood 只做调色 ⇒ 「开心」和「中性」在脸上完全一样，
    //    用户点一下角色只会说台词，看不出它高兴。
    this._applyMoodFace();
  }

  faceCamera(cameraPos) {
    const d = cameraPos.clone().sub(this.root.position);
    this._faceTarget = Math.atan2(d.x, d.z);
  }
  clearFace() { this._faceTarget = null; }

  headWorld(v) { return this.head.getWorldPosition(v); }

  update(dt) {
    this._t += dt;
    const t = this._t;

    // 移动插值
    if (this._walk) {
      const w = this._walk; w.t += dt / 1.4;
      const k = Math.min(1, w.t), e = k * k * (3 - 2 * k);
      this.root.position.lerpVectors(w.from, w.to, e);
      this.root.position.y = Math.abs(Math.sin(k * Math.PI * 3)) * 0.07;
      let dr = w.faceTo - w.faceFrom;
      while (dr > Math.PI) dr -= Math.PI * 2; while (dr < -Math.PI) dr += Math.PI * 2;
      this.root.rotation.y = w.faceFrom + dr * e;
      if (k >= 1) { this.root.position.y = w.to.y; this._walk = null; }
    }

    // 面向目标（对话时看镜头）
    if (this._faceTarget !== null && !this._walk) {
      let dr = this._faceTarget - this.root.rotation.y;
      while (dr > Math.PI) dr -= Math.PI * 2; while (dr < -Math.PI) dr += Math.PI * 2;
      this.root.rotation.y += dr * Math.min(1, dt * 8);
    }

    const speedMul = this.mood === 'happy' ? 1.6 : this.mood === 'sad' ? 0.55 : 1;

    // 待机浮动 + 尾巴摆动 + 呆毛摇晃
    if (this.mode !== 'sleep' && !this._walk) {
      this.body.position.y = Math.sin(t * 2 * speedMul) * 0.025;
    }
    this.tail.rotation.y = Math.sin(t * 2.2 * speedMul) * 0.35;
    this.ahoge.rotation.z = Math.sin(t * 1.7) * 0.12;

    // 眨眼
    this._blink -= dt;
    if (this._blink < 0) this._blink = 2 + Math.random() * 3;
    const bl = this._blink < 0.12 ? 0.15 : 1;
    // 睁眼度由 mood 决定（见 _applyMoodFace），再乘眨眼 ⇒ 笑眼在眨眼时也能闭合
    const open = bl * (this._eyeOpen == null ? 1 : this._eyeOpen);
    for (const e of this.eyes) e.scale.y = this.mode === 'sleep' ? 0.1 : open;

    // 模式动作
    if (this.mode === 'type') {
      this.arms[0].rotation.x = -1.1 + Math.sin(t * 14) * 0.12;
      this.arms[1].rotation.x = -1.1 + Math.cos(t * 14) * 0.12;
      this.body.position.y = -0.16;
      this.head.rotation.x = 0.12;
    } else if (this.mode === 'sit') {
      this.body.position.y = -0.18;
      this.arms[0].rotation.x = -0.5; this.arms[1].rotation.x = -0.5;
      this.head.rotation.x = -0.05;
    } else if (this.mode === 'sleep') {
      this.root.rotation.x = -Math.PI / 2 * 0.9;
      this.root.position.y = 0.42;
      this.body.position.y = 0;
      this.arms[0].rotation.set(0.4, 0, 0.5); this.arms[1].rotation.set(0.4, 0, -0.5);
      this.body.scale.y = 1 + Math.sin(t * 1.2) * 0.03;   // 呼吸
    } else if (this.mode === 'wave') {
      this.arms[0].rotation.z = 2.4 + Math.sin(t * 10) * 0.35;
      this.arms[1].rotation.z = -2.4 - Math.cos(t * 10) * 0.35;
    } else {
      this.arms[0].rotation.set(Math.sin(t * 1.8) * 0.08, 0, 0.15);
      this.arms[1].rotation.set(Math.cos(t * 1.8) * 0.08, 0, -0.15);
      this.head.rotation.x = 0;
    }
    if (this.mode !== 'sleep') {
      this.root.rotation.x = 0;
      if (this.mode !== 'sit' && this.mode !== 'type') this.body.position.y = Math.sin(t * 2 * speedMul) * 0.025;
    }
  }
}
