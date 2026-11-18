// Photo booth add-on (loaded by local_addons.py): hold an open hand (all 5 fingers) for 5 s -> 3-2-1 countdown -> snapshot of the
// annotated live canvas plus hand skeletons -> saved on the server + uploaded -> QR in the bottom-right
// corner for 10 s. Hands are tracked and drawn on every frame in every phase, so the live view never pauses.
import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import qrcode from 'https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/+esm';

const HOLD_MS = 5000, COUNT_FROM = 3, SHOW_MS = 10000, GAP_MS = 400;  // GAP: brief detection dropouts allowed while holding
const DETECT_MS = 33;  // ~30 hand detections per second; drawing runs every animation frame
const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const CONN = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const TIPS = [4, 8, 12, 16, 20], PALM = [0, 5, 9, 13, 17];

const vid = document.getElementById('vid'), cv = document.getElementById('cv'), stage = document.getElementById('stage');

document.head.insertAdjacentHTML('beforeend', `<style>
.pb-hands{pointer-events:none;z-index:2}
.pb-hold{position:absolute;left:50%;bottom:16px;transform:translateX(-50%);z-index:5;display:flex;align-items:center;gap:10px;padding:8px 16px 8px 8px;border-radius:999px;background:rgba(255,255,255,.94);color:#13202b;font-weight:600;box-shadow:0 8px 24px -10px rgba(0,0,0,.5)}
.pb-hold svg{transform:rotate(-90deg)}
.pb-count{position:absolute;inset:0;z-index:6;display:grid;place-items:center;pointer-events:none;font:700 min(34vh,220px)/1 "IBM Plex Sans Thai",system-ui,sans-serif;color:#fff;text-shadow:0 6px 40px rgba(0,0,0,.55)}
.pb-count span{display:grid;place-items:center;width:1.35em;height:1.35em;border-radius:50%;background:rgba(15,27,36,.5);animation:pb-pop 1s ease-out}
@keyframes pb-pop{from{transform:scale(1.5);opacity:0}30%{opacity:1}to{transform:scale(1)}}
.pb-flash{position:absolute;inset:0;z-index:7;background:#fff;pointer-events:none;animation:pb-flash .5s ease-out forwards}
@keyframes pb-flash{to{opacity:0}}
.pb-card{position:absolute;right:16px;bottom:16px;z-index:8;width:min(220px,40%);background:#fff;color:#13202b;border-radius:16px;padding:10px;box-shadow:0 16px 40px -12px rgba(0,0,0,.55)}
.pb-card .qr{width:100%;aspect-ratio:1;display:block;image-rendering:pixelated}
.pb-card .foot{display:flex;gap:8px;align-items:center;margin-top:6px}
.pb-card img.shot{width:56px;aspect-ratio:4/3;object-fit:cover;border-radius:6px;flex:none}
.pb-card p{margin:0;font-size:12px;line-height:1.35;color:#5b6b7a}
.pb-card b{color:#13202b;font-size:14px}
.pb-card .bar{height:4px;background:#e3f2f3;border-radius:3px;overflow:hidden;margin-top:8px}
.pb-card .bar i{display:block;height:100%;background:#0e7c86;transform-origin:left;animation:pb-shrink linear forwards}
@keyframes pb-shrink{to{transform:scaleX(0)}}
@media(prefers-reduced-motion:reduce){.pb-count span,.pb-flash{animation:none}.pb-flash{display:none}}
</style>`);

const el = (cls, html = '') => { const d = document.createElement('div'); d.className = cls; d.innerHTML = html; return d; };
const R = 15, C = 2 * Math.PI * R;
const holdUi = el('pb-hold hide', `<svg width="36" height="36" viewBox="0 0 36 36"><circle cx="18" cy="18" r="${R}" fill="none" stroke="#e3f2f3" stroke-width="4"/><circle id="pbRing" cx="18" cy="18" r="${R}" fill="none" stroke="#0e7c86" stroke-width="4" stroke-linecap="round" stroke-dasharray="${C}" stroke-dashoffset="${C}"/></svg><span id="pbHoldText"></span>`);
const countUi = el('pb-count hide');
// hand layer: same box and object-fit as #cv (both take the video's aspect), so normalised landmarks line up
const hcv = document.createElement('canvas'); hcv.className = 'pb-hands'; cv.after(hcv);
const hx = hcv.getContext('2d');
stage.append(holdUi, countUi);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------- hand detection ----------
let hands = null, loading = null;
function loadHands() {
  loading ??= (async () => {
    const files = await FilesetResolver.forVisionTasks(WASM);
    const opts = d => ({ baseOptions: { modelAssetPath: MODEL, delegate: d }, runningMode: 'VIDEO', numHands: 2 });
    try { hands = await HandLandmarker.createFromOptions(files, opts('GPU')); }
    catch { hands = await HandLandmarker.createFromOptions(files, opts('CPU')); }
  })().catch(e => { console.warn('hand model failed to load', e); setTimeout(() => loading = null, 10000); });
  return loading;
}

// all five fingers extended; landmarks are normalised to the frame, so x is rescaled to keep distances true
function openPalm(lm, aspect) {
  const p = lm.map(q => [q.x * aspect, q.y]);
  const d = (a, b) => Math.hypot(p[a][0] - p[b][0], p[a][1] - p[b][1]);
  const palm = d(0, 9);
  const fingers = [[8, 6], [12, 10], [16, 14], [20, 18]].every(([tip, pip]) => d(tip, 0) > d(pip, 0) * 1.1);
  const thumb = d(4, 17) > d(3, 17) && d(4, 5) > 0.45 * palm;
  return fingers && thumb;
}

let raw = [], smooth = [];  // latest landmarks per hand, and an eased copy for drawing (less jitter)
function detect(now) {
  raw = hands.detectForVideo(vid, now).landmarks;
  const same = smooth.length === raw.length;
  smooth = raw.map((lm, i) => same ? lm.map((q, j) => ({ x: smooth[i][j].x + (q.x - smooth[i][j].x) * .6, y: smooth[i][j].y + (q.y - smooth[i][j].y) * .6 })) : lm.map(q => ({ x: q.x, y: q.y })));
}

// ---------- state machine: idle -> holding -> counting -> busy -> showing -> idle ----------
let phase = 'idle', holdStart = null, lastOpen = 0, focus = -1, lastDet = 0;
const live = () => !!vid.srcObject && !vid.paused && vid.readyState >= 2;

function setHold(ms) {
  if (ms === null) { holdUi.classList.add('hide'); return; }
  holdUi.classList.remove('hide');
  document.getElementById('pbRing').style.strokeDashoffset = C * (1 - Math.min(ms / HOLD_MS, 1));
  document.getElementById('pbHoldText').textContent = `ชูมือค้างไว้อีก ${Math.max(0, Math.ceil((HOLD_MS - ms) / 1000))} วิ เพื่อถ่ายรูป`;
}

function update(now) {
  if (phase !== 'idle' && phase !== 'holding') return;  // counting/busy/showing: keep drawing, never re-trigger
  const open = raw.findIndex(lm => openPalm(lm, vid.videoWidth / vid.videoHeight));
  if (open >= 0) { lastOpen = now; holdStart ??= now; focus = open; }
  else if (now - lastOpen > GAP_MS) holdStart = null;
  if (holdStart === null) { phase = 'idle'; focus = -1; setHold(null); return; }
  phase = 'holding';
  const held = now - holdStart;
  setHold(held > 300 ? held : null);
  if (held >= HOLD_MS) { burst(); countdown(); }
}

function countdown() {
  phase = 'counting'; holdStart = null; setHold(null);
  let n = COUNT_FROM;
  const step = () => {
    if (!live()) { countUi.classList.add('hide'); phase = 'idle'; return; }
    if (n === 0) { countUi.classList.add('hide'); shoot(); return; }
    countUi.innerHTML = `<span>${n}</span>`; countUi.classList.remove('hide');
    n--; setTimeout(step, 1000);
  };
  step();
}

// ---------- drawing ----------
const toPx = (lm, W, H) => lm.map(q => [q.x * W, q.y * H]);
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const palmCenter = pts => PALM.reduce((c, i) => [c[0] + pts[i][0] / PALM.length, c[1] + pts[i][1] / PALM.length], [0, 0]);

function drawHand(ctx, pts, W, glow) {
  const lw = Math.max(2, W / 300);
  ctx.save();
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.shadowColor = '#22d3ee'; ctx.shadowBlur = 6 + 18 * glow;
  ctx.strokeStyle = glow > 0 ? `rgba(${Math.round(94 + 140 * glow)},${Math.round(234 + 10 * glow)},${Math.round(212 + 30 * glow)},1)` : '#5eead4';
  ctx.lineWidth = lw * (1 + .5 * glow);
  ctx.beginPath();
  for (const [a, b] of CONN) { ctx.moveTo(...pts[a]); ctx.lineTo(...pts[b]); }
  ctx.stroke();
  ctx.fillStyle = '#fff';
  pts.forEach((p, i) => { ctx.beginPath(); ctx.arc(p[0], p[1], lw * (TIPS.includes(i) ? 1.7 : 1.1), 0, Math.PI * 2); ctx.fill(); });
  ctx.restore();
}

// charging effect around the palm while the 5 s hold runs (p = 0..1)
function drawCharge(ctx, pts, p, now) {
  const [cx, cy] = palmCenter(pts), s = dist(pts[0], pts[9]), Rr = s * 1.45;
  ctx.save();
  ctx.translate(cx, cy); ctx.lineCap = 'round';
  ctx.shadowColor = '#22d3ee'; ctx.shadowBlur = 20;
  ctx.strokeStyle = 'rgba(34,211,238,.22)'; ctx.lineWidth = s * .07;
  ctx.beginPath(); ctx.arc(0, 0, Rr, 0, Math.PI * 2); ctx.stroke();
  ctx.strokeStyle = `hsl(${188 - 40 * p} 95% ${58 + 14 * p}%)`; ctx.lineWidth = s * .1;
  ctx.beginPath(); ctx.arc(0, 0, Rr, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * p); ctx.stroke();
  ctx.rotate(now / 650); ctx.setLineDash([s * .14, s * .2]);
  ctx.strokeStyle = `rgba(165,243,252,${.45 + .45 * p})`; ctx.lineWidth = s * .03;
  ctx.beginPath(); ctx.arc(0, 0, Rr * 1.2, 0, Math.PI * 2); ctx.stroke();
  ctx.rotate(-now / 325); ctx.setLineDash([s * .04, s * .1]);
  ctx.beginPath(); ctx.arc(0, 0, Rr * .78, 0, Math.PI * 2); ctx.stroke();
  ctx.restore();
  // glowing fingertips
  for (const t of TIPS) {
    const r = s * (.1 + .1 * p) * (1 + .18 * Math.sin(now / 110 + t));
    const g = ctx.createRadialGradient(pts[t][0], pts[t][1], 0, pts[t][0], pts[t][1], r);
    g.addColorStop(0, 'rgba(255,255,255,.95)'); g.addColorStop(.35, 'rgba(103,232,249,.75)'); g.addColorStop(1, 'rgba(34,211,238,0)');
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(pts[t][0], pts[t][1], r, 0, Math.PI * 2); ctx.fill();
  }
  // particles drawn in from the ring towards the palm, faster as the hold fills up
  if (!reduceMotion) for (let k = 0; k < 1 + 4 * p; k++) {
    const a = Math.random() * Math.PI * 2, r0 = Rr * (1.25 + Math.random() * .4);
    parts.push({ x: cx + Math.cos(a) * r0, y: cy + Math.sin(a) * r0, tx: cx, ty: cy, life: 0, max: 450 + Math.random() * 250, size: s * (.025 + Math.random() * .03), mode: 'in' });
  }
  lastCharge = { cx, cy, s };
}

let parts = [], lastCharge = null, lastFrame = 0;
function burst() {
  if (!lastCharge || reduceMotion) return;
  const { cx, cy, s } = lastCharge;
  for (let k = 0; k < 90; k++) {
    const a = Math.random() * Math.PI * 2, v = s * (2 + Math.random() * 5) / 1000;
    parts.push({ x: cx, y: cy, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 0, max: 600 + Math.random() * 500, size: s * (.03 + Math.random() * .05), mode: 'out' });
  }
}
function drawParticles(ctx, dt) {
  ctx.save(); ctx.globalCompositeOperation = 'lighter';
  parts = parts.filter(q => (q.life += dt) < q.max);
  for (const q of parts) {
    const t = q.life / q.max;
    if (q.mode === 'in') { q.x += (q.tx - q.x) * Math.min(1, dt / 160); q.y += (q.ty - q.y) * Math.min(1, dt / 160); }
    else { q.x += q.vx * dt; q.y += q.vy * dt; q.vx *= .985; q.vy *= .985; }
    ctx.fillStyle = `rgba(${q.mode === 'in' ? '125,249,255' : '186,255,240'},${(1 - t) * .9})`;
    ctx.beginPath(); ctx.arc(q.x, q.y, q.size * (1 - t * .5), 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
}

function render(now) {
  const W = vid.videoWidth, H = vid.videoHeight;
  if (hcv.width !== W || hcv.height !== H) { hcv.width = W; hcv.height = H; }
  const dt = Math.min(now - lastFrame, 100); lastFrame = now;
  hx.clearRect(0, 0, W, H);
  const p = phase === 'holding' ? Math.min((now - holdStart) / HOLD_MS, 1) : phase === 'counting' ? 1 : 0;
  smooth.forEach((lm, i) => {
    const pts = toPx(lm, W, H), charged = p > 0 && i === focus;
    if (charged) drawCharge(hx, pts, p, now);
    drawHand(hx, pts, W, charged ? p : 0);
  });
  drawParticles(hx, dt);
}

function frame(now) {
  requestAnimationFrame(frame);
  if (!live()) {
    if (phase === 'idle' || phase === 'holding') { phase = 'idle'; holdStart = null; setHold(null); }
    if (raw.length || parts.length) { raw = []; smooth = []; parts = []; hx.clearRect(0, 0, hcv.width, hcv.height); }
    return;
  }
  if (!hands) { loadHands(); return; }
  if (now - lastDet >= DETECT_MS) {
    lastDet = now;
    try { detect(now); update(now); } catch (e) { console.warn(e); }
  }
  render(now);
}

// ---------- capture ----------
async function shoot() {
  phase = 'busy';
  // the live canvas already carries boxes, emotions and body skeletons; add the hand skeletons on top
  const shot = document.createElement('canvas'); shot.width = cv.width; shot.height = cv.height;
  const sx = shot.getContext('2d'); sx.drawImage(cv, 0, 0);
  smooth.forEach(lm => drawHand(sx, toPx(lm, shot.width, shot.height), shot.width, .5));
  const blob = await new Promise(r => shot.toBlob(r, 'image/jpeg', 0.92));
  const flash = el('pb-flash'); stage.append(flash); setTimeout(() => flash.remove(), 600);
  const thumb = URL.createObjectURL(blob);
  const foot = (title, sub) => `<div class="foot"><img class="shot" src="${thumb}" alt="รูปที่ถ่าย"><p><b>${title}</b><br>${sub}</p></div>`;
  const card = el('pb-card', foot('กำลังอัพโหลด...', 'รอสักครู่'));
  stage.append(card);
  let html;
  try {
    const r = await fetch('/api/capture', { method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.status);
    if (j.url) {
      const q = qrcode(0, 'M'); q.addData(j.url); q.make();
      html = `<img class="qr" src="${q.createDataURL(8, 2)}" alt="QR สำหรับโหลดรูป">` + foot('สแกนเพื่อโหลดรูป', 'ลิงก์หมดอายุใน 1 ชม.');
    } else html = foot('บันทึกรูปในเครื่องแล้ว', j.error);
  } catch (e) {
    html = foot('บันทึกรูปไม่สำเร็จ', e.message);
  }
  card.innerHTML = html + `<div class="bar"><i style="animation-duration:${SHOW_MS}ms"></i></div>`;
  phase = 'showing';
  setTimeout(() => { card.remove(); URL.revokeObjectURL(thumb); phase = 'idle'; }, SHOW_MS);
}

requestAnimationFrame(frame);
