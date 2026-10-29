// 房间：程序化低模小屋（v0.4 精装修版）
import * as THREE from 'three';
import { MarketScreen, TerminalScreen, GameScreen, VideoScreen } from './market.js';

const C = {
  wall: 0xEAE4D8, wood: 0x8A6A48, darkwood: 0x5E4630,
  navy: 0x2B3A67, white: 0xF5F2EC, metal: 0x666E78,
  green: 0x4C8C4A, bedding: 0xBCD0E8,
};

function box(w, h, d, color, x = 0, y = 0, z = 0, ry = 0) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshLambertMaterial({ color }));
  m.position.set(x, y, z); m.rotation.y = ry;
  return m;
}
function cyl(rt, rb, h, color, x = 0, y = 0, z = 0, seg = 14) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, h, seg), new THREE.MeshLambertMaterial({ color }));
  m.position.set(x, y, z);
  return m;
}
function sph(r, color, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, 14, 10), new THREE.MeshLambertMaterial({ color }));
  m.position.set(x, y, z);
  return m;
}

// 木地板贴图（板条 + 色差）
function floorTexture() {
  const cv = document.createElement('canvas'); cv.width = cv.height = 512;
  const c = cv.getContext('2d');
  const tones = ['#C9B391', '#C2AA85', '#CFBB98', '#BFA87F'];
  for (let row = 0; row < 8; row++) {
    c.fillStyle = tones[row % 4];
    c.fillRect(0, row * 64, 512, 64);
    c.strokeStyle = 'rgba(90,70,48,0.55)'; c.lineWidth = 2;
    c.strokeRect(-2, row * 64, 516, 64);
    const off = (row % 2) * 170;
    for (let x = off; x < 512; x += 256) {
      c.beginPath(); c.moveTo(x, row * 64); c.lineTo(x, row * 64 + 64); c.stroke();
    }
    c.strokeStyle = 'rgba(90,70,48,0.15)'; c.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const gy = row * 64 + 8 + i * 12;
      c.beginPath(); c.moveTo(0, gy);
      for (let x = 0; x <= 512; x += 64) c.lineTo(x, gy + Math.sin(x * 0.05 + i) * 2);
      c.stroke();
    }
  }
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export class Room {
  constructor(scene) {
    this.group = new THREE.Group();
    this.screens = [];
    this.clockHands = {};
    this._monitorPlanes = [];
    this.workScr = [];
    this.playScr = [];
    this._build();
    scene.add(this.group);
    this.setScreenMode('work');
  }

  // 屏幕模式：work=行情/终端 · play=游戏/视频 · off=熄屏（PC 灯条同灭）
  setScreenMode(mode) {
    for (const { work, play } of this._monitorPlanes) {
      work.visible = mode === 'work';
      play.visible = mode === 'play';
    }
    for (const s of this.workScr) s.active = mode === 'work';
    for (const s of this.playScr) s.active = mode === 'play';
    if (this.pcRgb) this.pcRgb.visible = mode !== 'off';
    this.screenMode = mode;
  }

  _build() {
    const g = this.group;
    // 地板（木纹贴图）+ 圆形地毯
    const floor = new THREE.Mesh(new THREE.BoxGeometry(6.4, 0.1, 5.4),
      new THREE.MeshLambertMaterial({ map: floorTexture() }));
    floor.position.y = -0.05; g.add(floor);
    const rug = new THREE.Mesh(new THREE.CylinderGeometry(1.1, 1.1, 0.02, 28), new THREE.MeshLambertMaterial({ color: 0xD8CBB2 }));
    rug.position.set(0.4, 0.012, 0.2); g.add(rug);
    const rugRing = new THREE.Mesh(new THREE.TorusGeometry(0.95, 0.035, 6, 28), new THREE.MeshLambertMaterial({ color: 0xB56A4C }));
    rugRing.rotation.x = Math.PI / 2; rugRing.position.set(0.4, 0.025, 0.2); g.add(rugRing);

    // 天花板 + 吊灯
    g.add(box(6.4, 0.1, 5.4, C.wall, 0, 3.05, 0));
    const lampGrp = new THREE.Group();
    lampGrp.add(cyl(0.02, 0.02, 0.5, C.metal, 0, 0.25, 0));
    lampGrp.add(cyl(0.16, 0.22, 0.16, 0xD9CBB2, 0, -0.05, 0));
    this.ceilingBulb = new THREE.Mesh(new THREE.SphereGeometry(0.1, 12, 10),
      new THREE.MeshBasicMaterial({ color: 0x8A8468 }));
    this.ceilingBulb.position.y = -0.14; lampGrp.add(this.ceilingBulb);
    lampGrp.position.set(0.4, 2.8, 0.2); g.add(lampGrp);

    // 墙：后墙 + 左墙 + 右墙（右墙留窗洞）+ 踢脚线
    g.add(box(6.4, 3.0, 0.12, C.wall, 0, 1.5, -2.7));
    g.add(box(0.12, 3.0, 5.4, C.wall, -3.2, 1.5, 0));
    g.add(box(0.12, 3.0, 1.6, C.wall, 3.2, 1.5, -1.9));
    g.add(box(0.12, 3.0, 2.4, C.wall, 3.2, 1.5, 1.5));
    g.add(box(0.12, 0.9, 1.4, C.wall, 3.2, 0.45, -0.5));
    g.add(box(0.12, 0.6, 1.4, C.wall, 3.2, 2.7, -0.5));
    g.add(box(6.4, 0.12, 0.05, C.white, 0, 0.06, -2.63));
    g.add(box(0.05, 0.12, 5.4, C.white, -3.13, 0.06, 0));
    g.add(box(0.05, 0.12, 5.4, C.white, 3.13, 0.06, 0));

    // 空调（后墙右上）+ 出风口
    const ac = new THREE.Group();
    ac.add(box(1.0, 0.3, 0.2, 0xF2F0EA, 0, 0, 0));
    ac.add(box(0.9, 0.05, 0.06, 0xD8D4CC, 0, -0.12, 0.08));
    const acLed = new THREE.Mesh(new THREE.PlaneGeometry(0.05, 0.02), new THREE.MeshBasicMaterial({ color: 0x3FE0A8 }));
    acLed.position.set(0.35, 0.05, 0.105); ac.add(acLed);
    ac.position.set(2.5, 2.45, -2.6); g.add(ac);

    // 窗户：玻璃 + 天光 + 窗框 + 窗帘 + 窗台
    const sky = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 1.5), new THREE.MeshBasicMaterial({ color: 0x87B7E8 }));
    sky.position.set(3.34, 1.65, -0.5); sky.rotation.y = -Math.PI / 2;
    g.add(sky); this.sky = sky;
    const winFrame = new THREE.Group();
    winFrame.add(box(0.08, 1.56, 0.08, C.white, 0, 0, -0.7));
    winFrame.add(box(0.08, 1.56, 0.08, C.white, 0, 0, 0.7));
    winFrame.add(box(0.08, 0.08, 1.48, C.white, 0, 0.76, 0));
    winFrame.add(box(0.08, 0.08, 1.48, C.white, 0, -0.76, 0));
    winFrame.add(box(0.06, 1.5, 0.05, C.white, 0, 0, 0));
    winFrame.add(box(0.06, 0.06, 1.4, C.white, 0, 0, 0));
    winFrame.position.set(3.2, 1.65, -0.5); g.add(winFrame);
    for (const s of [-1, 1]) g.add(box(0.1, 1.8, 0.3, 0xD9CBB2, 3.12, 1.6, -0.5 + s * 0.85));
    g.add(box(0.2, 0.06, 1.9, C.white, 3.2, 0.93, -0.5));
    g.add(box(0.16, 0.05, 1.7, C.darkwood, 3.2, 2.48, -0.5));
    // 窗台小物：仙人掌 + 翻开的书
    g.add(cyl(0.05, 0.04, 0.07, 0xB0563A, 3.08, 1.0, 0.1));
    const cac = sph(0.05, C.green, 3.08, 1.09, 0.1); cac.scale.y = 1.5; g.add(cac);
    const sillBook = box(0.14, 0.015, 0.2, 0x2B6CB0, 3.06, 0.97, -0.9, 0.3);
    g.add(sillBook);

    // L 型书桌
    g.add(box(3.4, 0.08, 0.8, C.wood, 0.3, 0.76, -2.25));
    g.add(box(0.08, 0.72, 0.7, C.darkwood, -1.3, 0.38, -2.25));
    g.add(box(0.08, 0.72, 0.7, C.darkwood, 1.9, 0.38, -2.25));
    g.add(box(0.8, 0.08, 0.6, C.wood, 2.0, 0.76, -1.95));
    // 抽屉柜（桌下左侧）+ PC 主机（桌下右侧，带灯条）
    const drawer = new THREE.Group();
    drawer.add(box(0.5, 0.72, 0.6, C.wood, 0, 0.36, 0));
    for (let i = 0; i < 3; i++) {
      drawer.add(box(0.44, 0.03, 0.02, C.darkwood, 0, 0.18 + i * 0.2, 0.3));
      drawer.add(box(0.12, 0.025, 0.03, C.metal, 0, 0.26 + i * 0.2, 0.31));
    }
    drawer.position.set(-0.55, 0, -2.25); g.add(drawer);
    const pc = new THREE.Group();
    pc.add(box(0.24, 0.5, 0.5, 0x22262B, 0, 0.25, 0));
    const rgb = new THREE.Mesh(new THREE.PlaneGeometry(0.03, 0.36), new THREE.MeshBasicMaterial({ color: 0x35D0C2 }));
    rgb.position.set(0.125, 0.28, 0); rgb.rotation.y = Math.PI / 2; pc.add(rgb);
    this.pcRgb = rgb;
    pc.position.set(1.45, 0, -2.3); g.add(pc);

    // 三块显示器（每块两层画面：工作层 + 娱乐层，熄屏时全隐藏）
    const mkMonitor = (workTex, playTex, w, h, x, z, ry = 0) => {
      const grp = new THREE.Group();
      grp.add(box(w + 0.06, h + 0.06, 0.04, 0x1A1D22, 0, 0, -0.01));
      const work = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: workTex }));
      work.position.z = 0.012; grp.add(work);
      const play = new THREE.Mesh(new THREE.PlaneGeometry(w, h),
        new THREE.MeshBasicMaterial({ map: playTex || null, color: playTex ? 0xFFFFFF : 0x0A0A0C }));
      play.position.z = 0.014; play.visible = false; grp.add(play);
      grp.add(box(0.05, 0.22, 0.05, C.metal, 0, -h / 2 - 0.11, 0));
      grp.position.set(x, 1.15, z); grp.rotation.y = ry;
      g.add(grp);
      this._monitorPlanes.push({ work, play });
      return grp;
    };
    const kline = new MarketScreen(); this.screens.push(kline);
    const term = new TerminalScreen(); this.screens.push(term);
    const kline2 = new MarketScreen(256, 160, 16); this.screens.push(kline2);
    this.workScr = [kline, term, kline2];
    this.gameScr = new GameScreen(); this.screens.push(this.gameScr);
    this.videoScr = new VideoScreen(); this.screens.push(this.videoScr);
    this.playScr = [this.gameScr, this.videoScr];
    this.gameScr.active = false; this.videoScr.active = false;
    const monMain = mkMonitor(kline.texture, this.gameScr.texture, 0.78, 0.5, -0.15, -2.28);
    mkMonitor(term.texture, this.videoScr.texture, 0.56, 0.38, 0.78, -2.3, -0.25);
    mkMonitor(kline2.texture, null, 0.5, 0.34, -0.95, -2.3, 0.22);
    monMain.add(box(0.1, 0.04, 0.04, 0x1A1D22, 0, 0.3, 0));   // 摄像头
    const hp = new THREE.Group();                              // 耳机挂主屏侧面
    hp.add(new THREE.Mesh(new THREE.TorusGeometry(0.07, 0.015, 6, 14, Math.PI), new THREE.MeshLambertMaterial({ color: 0x2B3A67 })));
    hp.add(cyl(0.035, 0.035, 0.03, 0x2B3A67, -0.07, 0, 0).rotateZ(Math.PI / 2));
    hp.add(cyl(0.035, 0.035, 0.03, 0x2B3A67, 0.07, 0, 0).rotateZ(Math.PI / 2));
    hp.position.set(-0.48, 0.05, 0.02); monMain.add(hp);

    // 桌面：键盘鼠标 + 手机 + 文件纸张 + 笔 + 咖啡杯
    g.add(box(0.5, 0.03, 0.18, 0xD8D4CC, 0.35, 0.81, -2.0));
    g.add(box(0.1, 0.02, 0.14, 0xD8D4CC, 0.75, 0.8, -2.0));
    g.add(box(0.08, 0.012, 0.16, 0x1A1D22, 0.95, 0.805, -1.95, 0.4));   // 手机
    for (let i = 0; i < 3; i++) g.add(box(0.24, 0.006, 0.32, 0xFBFAF6, -0.5 + i * 0.02, 0.805 + i * 0.006, -2.0 + i * 0.015, 0.15 * i));
    const pen = cyl(0.008, 0.008, 0.18, 0xC0392B, -0.32, 0.81, -1.9, 8).rotateZ(Math.PI / 2 - 0.2);
    g.add(pen);
    const mug = new THREE.Group();
    mug.add(cyl(0.05, 0.045, 0.1, 0xC0392B, 0, 0.05, 0));
    const handle = new THREE.Mesh(new THREE.TorusGeometry(0.032, 0.01, 6, 12), new THREE.MeshLambertMaterial({ color: 0xC0392B }));
    handle.position.x = 0.055; mug.add(handle);
    mug.position.set(1.35, 0.8, -2.15); g.add(mug);
    g.add(box(0.24, 0.035, 0.17, 0x2B6CB0, 1.6, 0.815, -2.1));
    g.add(box(0.2, 0.03, 0.15, 0xB7791F, 1.61, 0.85, -2.11));
    // 台灯（可见灯体，点光源在 main 里对准这里）
    const deskLamp = new THREE.Group();
    deskLamp.add(cyl(0.07, 0.09, 0.03, 0x3A3F46, 0, 0.015, 0));
    const arm1 = box(0.03, 0.34, 0.03, C.metal, 0, 0.19, 0); arm1.rotation.z = 0.35; deskLamp.add(arm1);
    deskLamp.add(cyl(0.05, 0.09, 0.12, 0x3A3F46, -0.12, 0.38, 0));
    deskLamp.position.set(-0.75, 0.8, -2.15); g.add(deskLamp);
    this.lampHead = new THREE.Vector3(-0.87, 1.18, -2.15);

    // 办公椅
    const chair = new THREE.Group();
    chair.add(new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.45), new THREE.MeshLambertMaterial({ color: C.metal })));
    chair.add(box(0.5, 0.07, 0.48, C.navy, 0, 0.26, 0));
    chair.add(box(0.5, 0.55, 0.07, C.navy, 0, 0.55, -0.22));
    chair.position.set(0.55, 0.02, -1.55); g.add(chair);

    // 咖啡机（侧台）
    const cm = new THREE.Group();
    cm.add(box(0.26, 0.34, 0.26, 0x3A3F46, 0, 0.17, 0));
    cm.add(box(0.1, 0.08, 0.1, C.metal, 0, 0.06, 0.14));
    cm.add(box(0.12, 0.1, 0.12, C.white, 0, 0.05, 0.15));
    cm.position.set(2.0, 0.8, -2.0); g.add(cm);
    this.coffeeMachine = cm;

    // 垃圾桶
    g.add(cyl(0.09, 0.075, 0.24, 0x9AA0A6, 0.15, 0.12, -1.85));

    // 巴菲特照片（木相框 + 生成肖像，失败则手绘兜底）
    const poster = new THREE.Group();
    poster.add(box(1.0, 1.16, 0.06, C.darkwood, 0, 0, -0.02));
    poster.add(box(0.9, 1.06, 0.02, 0xF5F2EC, 0, 0, 0.012));
    const posterTex = new THREE.TextureLoader().load('./assets/buffett.jpg', undefined, undefined, () => {
      const cv = document.createElement('canvas'); cv.width = cv.height = 256;
      const c = cv.getContext('2d');
      c.fillStyle = '#D9C9A8'; c.fillRect(0, 0, 256, 256);
      c.fillStyle = '#8C6E4A'; c.beginPath(); c.arc(128, 108, 56, 0, Math.PI * 2); c.fill();
      c.fillStyle = '#3A3F46'; c.fillRect(76, 168, 104, 66);
      c.fillStyle = '#5A4A38'; c.font = 'bold 20px serif'; c.textAlign = 'center';
      c.fillText('Buffett', 128, 40);
      const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace;
      photo.material.map = t; photo.material.needsUpdate = true;
    });
    posterTex.colorSpace = THREE.SRGBColorSpace;
    const photo = new THREE.Mesh(new THREE.PlaneGeometry(0.86, 1.0), new THREE.MeshBasicMaterial({ map: posterTex }));
    photo.position.z = 0.026;
    poster.add(photo);
    poster.position.set(-1.85, 1.6, -2.62); g.add(poster);

    // K 线艺术画（显示器右侧小挂画）
    const artCv = document.createElement('canvas'); artCv.width = 128; artCv.height = 96;
    {
      const c = artCv.getContext('2d');
      c.fillStyle = '#F5F2EC'; c.fillRect(0, 0, 128, 96);
      for (let i = 0; i < 12; i++) {
        const x = 10 + i * 10, up = Math.random() > 0.45;
        c.strokeStyle = c.fillStyle = up ? '#E24B4A' : '#1D9E75';
        c.beginPath(); c.moveTo(x, 20 + Math.random() * 50); c.lineTo(x, 30 + Math.random() * 50); c.stroke();
        c.fillRect(x - 3, 35 + Math.random() * 30, 6, 12);
      }
      const t = new THREE.CanvasTexture(artCv); t.colorSpace = THREE.SRGBColorSpace;
      const art = new THREE.Group();
      art.add(box(0.5, 0.4, 0.04, C.darkwood, 0, 0, -0.01));
      const artFace = new THREE.Mesh(new THREE.PlaneGeometry(0.44, 0.34), new THREE.MeshBasicMaterial({ map: t }));
      artFace.position.z = 0.012; art.add(artFace);
      art.position.set(2.35, 1.7, -2.62); g.add(art);
    }

    // 置物架（书桌上方）：书 + 小盆栽
    g.add(box(1.4, 0.05, 0.25, C.wood, 0.3, 2.05, -2.55));
    const shelfCols = [0xC0392B, 0x2B6CB0, 0x2F855A, 0xB7791F];
    for (let i = 0; i < 5; i++) g.add(box(0.07, 0.26, 0.16, shelfCols[i % 4], -0.2 + i * 0.11, 2.21, -2.55));
    g.add(cyl(0.05, 0.04, 0.07, 0xB0563A, 0.75, 2.11, -2.55));
    const shelfPlant = sph(0.06, C.green, 0.75, 2.2, -2.55); shelfPlant.scale.y = 1.3; g.add(shelfPlant);

    // 便利贴墙（桌右上方）
    const noteCols = [0xF7D154, 0xF5A9A0, 0x9FE1CB, 0xB5D4F4];
    noteCols.forEach((col, i) => {
      const note = new THREE.Mesh(new THREE.PlaneGeometry(0.09, 0.09), new THREE.MeshBasicMaterial({ color: col }));
      note.position.set(1.15 + (i % 2) * 0.13, 1.75 - Math.floor(i / 2) * 0.13, -2.63);
      note.rotation.z = (i - 1.5) * 0.12;
      g.add(note);
    });

    // 门（左墙）+ 门口地垫 + 挂钩与帽子 + 插座开关
    const door = new THREE.Group();
    door.add(box(0.1, 2.1, 0.98, C.darkwood, 0, 1.05, 0));
    door.add(box(0.06, 1.94, 0.84, C.wood, 0.04, 0.97, 0));
    door.add(box(0.02, 0.5, 0.34, C.darkwood, 0.08, 1.35, 0));
    door.add(box(0.02, 0.5, 0.34, C.darkwood, 0.08, 0.55, 0));
    const knob = sph(0.045, 0xC9A84C, 0.09, 0.95, 0.32);
    door.add(knob);
    door.position.set(-3.18, 0, -0.55); g.add(door);
    g.add(box(0.5, 0.02, 0.9, 0xB0563A, -2.9, 0.012, -0.55));
    const hookRail = new THREE.Group();
    hookRail.add(box(0.04, 0.06, 0.5, C.wood, 0, 0, 0));
    for (let i = 0; i < 3; i++) hookRail.add(cyl(0.012, 0.012, 0.08, C.metal, 0.03, -0.06, -0.15 + i * 0.15, 8));
    hookRail.position.set(-3.13, 1.75, -1.5); g.add(hookRail);
    const hat = new THREE.Group();
    hat.add(sph(0.09, 0x8C6E4A, 0, 0.03, 0));
    const brim = cyl(0.14, 0.14, 0.015, 0x8C6E4A, 0, 0, 0, 16); hat.add(brim);
    hat.position.set(-3.1, 1.62, -1.65); g.add(hat);
    const socket = new THREE.Group();
    socket.add(box(0.02, 0.12, 0.09, C.white, 0, 0, 0));
    socket.add(sph(0.012, 0x888780, 0.012, 0.02, -0.02));
    socket.add(sph(0.012, 0x888780, 0.012, 0.02, 0.02));
    socket.position.set(-3.13, 1.1, -1.1); g.add(socket);
    const sw = box(0.02, 0.08, 0.06, C.white, -3.13, 1.25, -0.2); g.add(sw);

    // 挂历（左墙）
    const calCv = document.createElement('canvas'); calCv.width = 128; calCv.height = 160;
    {
      const c = calCv.getContext('2d');
      c.fillStyle = '#FBFAF6'; c.fillRect(0, 0, 128, 160);
      c.fillStyle = '#C0392B'; c.fillRect(0, 0, 128, 30);
      c.fillStyle = '#fff'; c.font = 'bold 16px sans-serif'; c.textAlign = 'center';
      c.fillText('2026 · 10', 64, 21);
      c.fillStyle = '#2C2C2A'; c.font = '9px sans-serif';
      let d = 1;
      for (let r = 0; r < 5 && d <= 31; r++) for (let col = 0; col < 7 && d <= 31; col++) {
        c.fillText(String(d), 12 + col * 18, 52 + r * 22); d++;
      }
      c.strokeStyle = '#C0392B'; c.lineWidth = 2;
      c.strokeRect(12 + 0 * 18 - 7, 43 + 3 * 22, 16, 14);   // 圈出今天（26日≈第4行第1列）
      const t = new THREE.CanvasTexture(calCv); t.colorSpace = THREE.SRGBColorSpace;
      const cal = new THREE.Group();
      cal.add(box(0.36, 0.46, 0.02, C.white, 0, 0, -0.01));
      const calFace = new THREE.Mesh(new THREE.PlaneGeometry(0.32, 0.42), new THREE.MeshBasicMaterial({ map: t }));
      cal.add(calFace);
      cal.position.set(-3.13, 1.7, 0.6); cal.rotation.y = Math.PI / 2; g.add(cal);
    }

    // 床（左前角）：床头柜 + 闹钟 + 鲸鱼玩偶 + 拖鞋
    const bed = new THREE.Group();
    bed.add(box(1.0, 0.25, 2.0, C.darkwood, 0, 0.13, 0));
    bed.add(box(0.95, 0.14, 1.95, C.bedding, 0, 0.32, 0));
    bed.add(box(0.96, 0.06, 1.1, 0x8FA8CC, 0, 0.38, 0.4));
    bed.add(box(0.6, 0.12, 0.35, C.white, 0, 0.4, -0.75));
    bed.position.set(-2.55, 0, 1.0); g.add(bed);
    const nstand = new THREE.Group();
    nstand.add(box(0.4, 0.42, 0.35, C.wood, 0, 0.21, 0));
    nstand.add(box(0.3, 0.02, 0.25, C.darkwood, 0, 0.33, 0.18));
    const alarm = new THREE.Group();
    alarm.add(cyl(0.05, 0.05, 0.03, 0xD94F3D, 0, 0, 0).rotateX(Math.PI / 2));
    alarm.add(sph(0.018, 0xC9A84C, -0.035, 0.05, 0));
    alarm.add(sph(0.018, 0xC9A84C, 0.035, 0.05, 0));
    alarm.position.set(0, 0.45, 0); alarm.rotation.y = 0.5; nstand.add(alarm);
    nstand.position.set(-2.55, 0, -0.25); g.add(nstand);
    const plush = new THREE.Group();
    const pb = sph(0.13, 0x7FA8D9, 0, 0, 0); pb.scale.set(1, 0.85, 1.35); plush.add(pb);
    const pbelly = sph(0.09, 0xF5F2EC, 0, -0.02, 0.06); pbelly.scale.set(1, 0.8, 1.2); plush.add(pbelly);
    for (const s of [-1, 1]) {
      const fl = sph(0.07, 0x7FA8D9, 0.13 * s, 0.03, -0.14); fl.scale.set(1.4, 0.4, 0.9); plush.add(fl);
      plush.add(sph(0.014, 0x1B2A4A, 0.05 * s, 0.04, 0.16));
    }
    plush.position.set(-2.3, 0.5, 1.3); plush.rotation.y = 0.7; g.add(plush);
    for (const s of [-1, 1]) g.add(box(0.1, 0.04, 0.24, 0xD9A8A0, -1.85 + s * 0.12, 0.02, 0.15, s * 0.15));

    // 沙发（右侧）+ 靠枕 + 落地灯 + 茶几
    const sofa = new THREE.Group();
    sofa.add(box(1.3, 0.3, 0.6, 0x7A8C5C, 0, 0.15, 0));
    sofa.add(box(1.3, 0.5, 0.18, 0x7A8C5C, 0, 0.5, -0.21));
    sofa.add(box(0.18, 0.45, 0.6, 0x6E7F52, -0.62, 0.42, 0));
    sofa.add(box(0.18, 0.45, 0.6, 0x6E7F52, 0.62, 0.42, 0));
    sofa.add(box(0.3, 0.12, 0.28, 0xD9CBB2, -0.3, 0.5, -0.05));
    sofa.add(box(0.3, 0.12, 0.28, 0xBCD0E8, 0.32, 0.5, -0.02));
    sofa.position.set(1.95, 0, 0.95); sofa.rotation.y = -Math.PI / 4; g.add(sofa);
    const floorLamp = new THREE.Group();
    floorLamp.add(cyl(0.14, 0.16, 0.03, C.metal, 0, 0.015, 0));
    floorLamp.add(cyl(0.02, 0.02, 1.3, C.metal, 0, 0.68, 0));
    floorLamp.add(cyl(0.14, 0.2, 0.22, 0xF2E3C4, 0, 1.4, 0));
    floorLamp.position.set(2.75, 0, 1.7); g.add(floorLamp);
    const stbl = new THREE.Group();
    stbl.add(cyl(0.3, 0.3, 0.04, C.wood, 0, 0.42, 0, 18));
    stbl.add(cyl(0.03, 0.03, 0.42, C.darkwood, 0, 0.21, 0));
    stbl.add(cyl(0.16, 0.16, 0.03, C.darkwood, 0, 0.015, 0, 18));
    stbl.add(cyl(0.04, 0.035, 0.07, C.white, 0.08, 0.475, 0.05));
    stbl.add(box(0.2, 0.03, 0.15, 0x6B46C1, -0.05, 0.455, -0.06, 0.4));
    stbl.position.set(2.7, 0, 0.5); g.add(stbl);

    // 墙角健身角：瑜伽垫 + 哑铃
    const mat1 = cyl(0.09, 0.09, 0.7, 0x5DCAA5, -2.75, 0.09, -1.95, 12).rotateZ(Math.PI / 2);
    mat1.rotation.y = 0.5; g.add(mat1);
    const db = new THREE.Group();
    db.add(cyl(0.02, 0.02, 0.22, C.metal, 0, 0, 0, 8).rotateZ(Math.PI / 2));
    db.add(cyl(0.07, 0.07, 0.05, 0x3A3F46, -0.12, 0, 0, 12).rotateZ(Math.PI / 2));
    db.add(cyl(0.07, 0.07, 0.05, 0x3A3F46, 0.12, 0, 0, 12).rotateZ(Math.PI / 2));
    db.position.set(-2.5, 0.07, -2.2); db.rotation.y = 0.6; g.add(db);

    // 绿萝（窗台）
    const plant = new THREE.Group();
    plant.add(new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.07, 0.14, 10), new THREE.MeshLambertMaterial({ color: 0xB0563A })));
    const leaves = new THREE.Mesh(new THREE.IcosahedronGeometry(0.16, 0), new THREE.MeshLambertMaterial({ color: C.green }));
    leaves.position.y = 0.2; plant.add(leaves);
    plant.position.set(3.1, 1.0, -0.15); g.add(plant);

    // 挂钟（后墙）
    const clock = new THREE.Group();
    const face = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.04, 24), new THREE.MeshLambertMaterial({ color: C.white }));
    face.rotation.x = Math.PI / 2; clock.add(face);
    const mkHand = (len, wdt) => {
      const geo = new THREE.BoxGeometry(wdt, len, 0.01);
      geo.translate(0, len / 2, 0);
      const h = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x2C2C2A }));
      h.position.z = 0.03; clock.add(h); return h;
    };
    this.clockHands.hour = mkHand(0.1, 0.016);
    this.clockHands.min = mkHand(0.16, 0.012);
    this.clockHands.sec = mkHand(0.18, 0.005);
    clock.position.set(1.6, 2.35, -2.62); g.add(clock);
  }

  // 天空颜色 + 灯光随模拟时间变化
  updateSky(hour, sunLight, ambient, lampLight) {
    const skyCol = (() => {
      if (hour >= 6 && hour < 7.5) return 0xF2B27A;
      if (hour >= 7.5 && hour < 17) return 0x87B7E8;
      if (hour >= 17 && hour < 19) return 0xC98AB0;
      return 0x1B2A4A;
    })();
    this.sky.material.color.setHex(skyCol);
    const day = hour >= 7 && hour < 18;
    sunLight.intensity = day ? 0.9 : 0.05;
    ambient.intensity = day ? 0.75 : 0.32;
    lampLight.intensity = (hour >= 18 || hour < 7) ? 0.8 : 0.0;
    this.ceilingBulb.material.color.setHex(day ? 0x8A8468 : 0xFFE9B8);
  }

  updateClock(date) {
    const s = date.getSeconds(), m = date.getMinutes(), h = date.getHours() % 12;
    this.clockHands.sec.rotation.z = -(s / 60) * Math.PI * 2;
    this.clockHands.min.rotation.z = -((m + s / 60) / 60) * Math.PI * 2;
    this.clockHands.hour.rotation.z = -((h + m / 60) / 12) * Math.PI * 2;
  }

  updateScreens(dt) { for (const s of this.screens) s.update(dt); }
  setMood(t) { for (const s of this.screens) if (s.setTrend) s.setTrend(t); }
}
