// 主入口：场景 + 状态机 + 角色 + 对话 的接线
import * as THREE from 'three';
import { Room } from './room.js';
import { WhaleGirl } from './character.js';
import { ModelGirl } from './modelgirl.js';
import { Schedule } from './schedule.js';
import { Dialogue } from './dialogue.js';
import { Bubble } from './bubble.js';

// fetch file:// 失败时的兜底配置（与 schedule.json 同构，P1 改为打包内 JS 模块）
const FALLBACK_CFG = {
  version: 0, name: '阿深',
  states: {
    premarket: { label: '盘前准备', spot: 'coffee', anim: 'stand', lines: ['泡杯咖啡，看看隔夜外盘……'] },
    meeting: { label: '晨会', spot: 'whiteboard', anim: 'stand', lines: ['今日计划：控制仓位。'] },
    watch_am: { label: '盯盘（上午）', spot: 'desk', anim: 'type', lines: ['开盘了，精神起来！'] },
    lunch: { label: '午休', spot: 'sofa', anim: 'sit', lines: ['午饭吃什么呢……'] },
    watch_pm: { label: '盯盘（下午）', spot: 'desk', anim: 'type', lines: ['别打瞌睡。'] },
    review: { label: '复盘', spot: 'desk', anim: 'type', lines: ['写今天的复盘……'] },
    offduty: { label: '自由活动', spot: 'random', anim: 'wander', lines: ['下班！'] },
    sleep: { label: '睡觉', spot: 'bed', anim: 'sleep', lines: ['zzz……'] },
  },
  timetable: [
    { days: 'trading', start: '08:30', end: '09:15', state: 'premarket' },
    { days: 'trading', start: '09:15', end: '09:30', state: 'meeting' },
    { days: 'trading', start: '09:30', end: '11:30', state: 'watch_am' },
    { days: 'trading', start: '11:30', end: '13:00', state: 'lunch' },
    { days: 'trading', start: '13:00', end: '15:00', state: 'watch_pm' },
    { days: 'trading', start: '15:00', end: '16:30', state: 'review' },
    { days: 'trading', start: '16:30', end: '22:00', state: 'offduty' },
    { days: 'all', start: '22:00', end: '29:59', state: 'sleep' },
    { days: 'all', start: '00:00', end: '08:30', state: 'sleep' },
    { days: 'nontrading', start: '08:30', end: '22:00', state: 'offduty' },
  ],
  spots: {
    desk: { pos: [0.55, 0, -1.55], face: 3.14159 },
    whiteboard: { pos: [-1.55, 0, -1.95], face: 3.14159 },
    coffee: { pos: [1.75, 0, -2.05], face: 3.14159 },
    bed: { pos: [-2.05, 0, 1.05], face: 1.5708 },
    sofa: { pos: [1.78, 0.32, 0.72], face: -0.7854 },
    window: { pos: [2.35, 0, -0.55], face: 1.5708 },
    door: { pos: [-2.85, 0, -0.55], face: -1.5708 },
  },
};

const OFFDUTY_SPOTS = ['sofa', 'window', 'sofa', 'desk'];

async function boot() {
  const cfg = await fetch('./schedule.json').then(r => r.json()).catch(() => FALLBACK_CFG);

  // --- 渲染基础 ---
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  document.getElementById('app').appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x181C24);
  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
  const camBase = new THREE.Vector3(0.35, 2.15, 5.3);
  camera.position.copy(camBase);
  camera.lookAt(0.1, 0.85, -0.6);

  const ambient = new THREE.AmbientLight(0xFFFFFF, 0.75);
  const sun = new THREE.DirectionalLight(0xFFF2DD, 0.9);
  sun.position.set(4, 6, 3);
  const lamp = new THREE.PointLight(0xFFCA7A, 0, 6);
  lamp.position.copy(new THREE.Vector3(-0.87, 1.3, -2.05));   // 对准桌面台灯
  scene.add(ambient, sun, lamp);

  // --- 模块 ---
  const room = new Room(scene);
  let girl;
  try {
    girl = await ModelGirl.create(scene, './assets/ashen.glb');
  } catch (e) {
    console.warn('正式模型加载失败，回退到占位体:', e);
    girl = new WhaleGirl(scene);
  }
  const schedule = new Schedule(cfg);
  const marketPct = { value: 0.32 };  // P0 mock 大盘涨跌幅 %
  const dialogue = new Dialogue(() => ({
    stateLabel: schedule.state?.label ?? '加载中',
    lines: schedule.state?.lines ?? ['……'],
    time: schedule.hhmm(),
    marketPct: marketPct.value,
  }));
  const bubble = new Bubble(document.getElementById('bubble'), camera);

  // --- 状态切换：换位置 + 换动作 ---
  let wanderTimer = 0, awayTimer = 0, pendingOut = false;
  const randomSpot = () => OFFDUTY_SPOTS[Math.floor(Math.random() * OFFDUTY_SPOTS.length)];
  // 屏幕跟着状态走：工作=行情终端 · 午休/自由活动=游戏视频 · 睡觉/出门=熄屏
  function screenModeFor(state) {
    if (state.anim === 'type' || state.anim === 'stand') return 'work';
    if (state.anim === 'sleep') return 'off';
    return 'play';
  }
  function setStatus(label, chip) {
    document.getElementById('stateLabel').textContent = label;
    const chipEl = document.getElementById('stateChip');
    chipEl.textContent = chip;
    const colors = {
      '工作中': ['#E1F5EE', '#085041'], '休息中': ['#EEEDFE', '#3C3489'],
      '外出中': ['#FAEEDA', '#854F0B'], '待机': ['#F1EFE8', '#444441'],
    };
    const [bg, fg] = colors[chip] || colors['待机'];
    chipEl.style.background = bg; chipEl.style.color = fg;
  }
  function applyState(state) {
    let spotName = state.spot;
    if (spotName === 'random') spotName = randomSpot();
    const spot = cfg.spots[spotName];
    if (girl.away) girl.appearAt(cfg.spots.door);   // 到点了人还在外面 → 从门口回来
    girl.goTo(spot);
    girl.setMode(state.anim === 'wander' ? 'stand' : state.anim);
    wanderTimer = 0; awayTimer = 0; pendingOut = false;
    room.setScreenMode(screenModeFor(state));
    setStatus(state.label, state.anim === 'type' ? '工作中' : state.anim === 'sleep' ? '休息中' : '待机');
  }
  schedule.onChange(applyState);

  // --- 对话：打断当前动作 → 面向镜头回应 → 恢复（外出时先喊回来） ---
  let resume = null, busy = 0;
  async function onCommand(text) {
    let prefix = '';
    if (girl.away) {
      girl.appearAt(cfg.spots.door);
      girl.goTo(cfg.spots.sofa);
      awayTimer = 0;
      room.setScreenMode(screenModeFor(schedule.state));
      setStatus(schedule.state.label, '待机');
      prefix = '我回来啦！刚才出去逛了一圈。';
    }
    // 被叫到的**动作先发生**（转身、惊讶），不能等接口回来才动——那样会有 1 秒僵硬
    if (busy <= 0) resume = girl.mode;
    girl.stop();                    // 走路时被叫到：立刻站定转身
    girl.setMode('wave');
    girl.faceCamera(camera.position);
    if (girl.setEmotion) girl.setEmotion('surprised', 1.5);   // 被叫到：惊讶 → 随后自动回落
    busy = 2.6;
    // 数据回来再开口。dialogue 里查不到会**如实说查不到**，不编行情。
    let reply;
    try {
      reply = await dialogue.respond(text);
    } catch (e) {
      reply = '我这边出错了，这条没查到。';
    }
    // 气泡时长按字数算：真实回答比原来的模板长（会列出具体代码），5.2s 读不完
    const full = prefix + reply;
    const ms = Math.max(5200, Math.min(15000, full.length * 170));
    bubble.say(full, v => girl.headWorld(v).add(new THREE.Vector3(0, 0.55, 0)), ms);
  }
  const input = document.getElementById('cmdInput');
  document.getElementById('cmdForm').addEventListener('submit', e => {
    e.preventDefault();
    const t = input.value;
    if (!t.trim()) return;
    input.value = '';
    onCommand(t);
  });
  for (const chip of document.querySelectorAll('.chip')) {
    chip.addEventListener('click', () => onCommand(chip.dataset.cmd));
  }

  // --- 调试面板 ---
  const timeSlider = document.getElementById('timeSlider');
  const speedSel = document.getElementById('speedSel');
  const moodSlider = document.getElementById('moodSlider');
  const dayToggle = document.getElementById('dayToggle');
  timeSlider.addEventListener('input', () => {
    schedule.speed = 0;
    schedule.setTime(Math.floor(timeSlider.value / 60), timeSlider.value % 60);
  });
  speedSel.addEventListener('change', () => { schedule.speed = Number(speedSel.value); });
  dayToggle.addEventListener('change', () => {
    schedule.forceDayType = dayToggle.checked ? 'nontrading' : null;
    schedule.state = null;
  });
  moodSlider.addEventListener('input', () => {
    marketPct.value = Number(moodSlider.value);
    room.setMood(marketPct.value / 3);
    const m = marketPct.value > 1 ? 'happy' : marketPct.value < -1 ? 'sad' : 'neutral';
    girl.setMood(m);
    document.getElementById('moodLabel').textContent =
      marketPct.value > 1 ? '开心' : marketPct.value < -1 ? '低落' : '平静';
  });

  // --- 鼠标视差 ---
  let mx = 0, my = 0;
  window.addEventListener('mousemove', e => {
    mx = (e.clientX / window.innerWidth - 0.5) * 2;
    my = (e.clientY / window.innerHeight - 0.5) * 2;
  });
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // --- 调试钩子（截图冒烟/排查用） ---
  window.__girl = girl; window.__sched = schedule; window.__cfg = cfg; window.__room = room;

  // --- 主循环 ---
  let last = performance.now();
  function loop(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;

    schedule.update();
    const hour = schedule.simDate.getHours() + schedule.simDate.getMinutes() / 60;

    if (busy > 0) {
      busy -= dt;
      if (busy <= 0 && resume) { girl.setMode(resume); girl.clearFace(); resume = null; }
    }

    // 自由活动：溜达，或出门玩（一段时间不在房间）
    if (schedule.state?.anim === 'wander' && busy <= 0) {
      if (girl.away) {
        awayTimer -= dt;
        if (awayTimer <= 0) {   // 玩够了，回家
          girl.appearAt(cfg.spots.door);
          girl.goTo(cfg.spots[randomSpot()]);
          room.setScreenMode(screenModeFor(schedule.state));
          setStatus(schedule.state.label, '待机');
        }
      } else if (!girl._walk) {
        wanderTimer += dt;
        if (wanderTimer > 7) {
          wanderTimer = 0;
          if (Math.random() < 0.45) {   // 出门
            girl.goTo(cfg.spots.door);
            pendingOut = true;
          } else {
            girl.goTo(cfg.spots[randomSpot()]);
          }
        }
      }
    }
    // 走到门口 → 出门消失（人走灯灭，电脑关掉）
    if (pendingOut && !girl._walk) {
      pendingOut = false;
      girl.disappear();
      awayTimer = 12 + Math.random() * 25;
      room.setScreenMode('off');
      setStatus(schedule.state.label, '外出中');
    }

    room.updateSky(hour, sun, ambient, lamp);
    room.updateClock(schedule.simDate);
    room.updateScreens(dt);
    girl.update(dt);
    bubble.update();

    camera.position.x = camBase.x + mx * 0.22;
    camera.position.y = camBase.y - my * 0.12;
    camera.lookAt(0.1, 0.85, -0.6);

    document.getElementById('clockLabel').textContent = schedule.hhmm();
    if (schedule.speed > 0) timeSlider.value = Math.floor(hour * 60) % 1440;

    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

boot();
