// Main thread: camera, overlay drawing, UI. Inference lives in detector.worker.js,
// the ESP32 link in serial.js, the decision rule in fusion.js.
//
// Three independent loops, so one being slow never stalls the others:
//   video      - native <video> playback at the camera's own frame rate
//   detection  - pump(): one frame in flight to the worker at a time
//   panels     - setInterval at 10 Hz for cards, sparkline, banner
import {
  ALARM, VETOED, CLEAR, UNVERIFIED, WARM, COLD, NEUTRAL, OFFLINE, CLASS_NAMES,
  ThermalBuffer, FusionEngine,
} from "./fusion.js";
import { ThermalSerial, serialSupported } from "./serial.js";
import { ServoController, PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX } from "./servo.js";

const $ = id => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), octx = overlay.getContext("2d");

const thermal = new ThermalBuffer();
const engine = new FusionEngine(thermal);
const settings = { conf: 0.45, iou: 0.45, size: 416, sound: true, notify: false };

let stream = null, worker = null, workerReady = false, busy = false, busySince = 0;
let boxes = [], detMs = 0, detFps = 0, lastResultAt = 0, backend = "", forceWasm = false, workerErrors = 0;
let log = [], simTimer = null, audio = null, beepTimer = null;

// ------------------------------------------------------------------ helpers
function toast(msg, ms = 5000) {
  const t = $("toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), ms);
}
function chip(id, cls, text) {
  const c = $(id); c.className = "chip " + cls; c.lastChild.textContent = text;
}
function jetColor(t, lo = 18, hi = 38) {
  // Same breakpoints as OpenCV COLORMAP_JET, which physics_thermal_synth.py uses.
  const v = Math.max(0, Math.min(1, (t - lo) / (hi - lo)));
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 3)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 1)));
  return `rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`;
}
const pct = t => Math.max(0, Math.min(100, (t - 18) / 20 * 100));

// ------------------------------------------------------------------ detector worker
function startWorker() {
  worker?.terminate();
  workerReady = false; busy = false;
  chip("st-ai", "warn", "AI loading…");
  worker = new Worker(new URL("./detector.worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "ready") {
      workerReady = true; workerErrors = 0; backend = data.backend;
      chip("st-ai", "on", `AI ${backend.toUpperCase()}`);
      pump();
    } else if (data.type === "result") {
      busy = false;
      boxes = data.boxes;
      detMs = data.ms;
      const now = performance.now();
      if (lastResultAt) detFps = 0.85 * detFps + 0.15 * (1000 / Math.max(1, now - lastResultAt));
      lastResultAt = now;
      workerErrors = 0;
      if (stream) engine.update(boxes);
      pump();
    } else if (data.type === "error") {
      busy = false;
      workerErrors++;
      console.error("detector:", data.message);
      if (data.during === "load" && !forceWasm) {
        forceWasm = true; toast("GPU backend failed, switching to CPU (WASM)."); startWorker();
      } else if (data.during === "load") {
        chip("st-ai", "bad", "AI failed"); toast("Could not load the model: " + data.message, 10000);
      } else if (workerErrors >= 3 && !forceWasm) {
        forceWasm = true; toast("GPU inference unstable, switching to CPU (WASM)."); startWorker();
      } else pump();
    }
  };
  worker.onerror = e => { console.error(e); busy = false; chip("st-ai", "bad", "AI crashed"); setTimeout(startWorker, 1000); };
  worker.postMessage({ type: "load", modelUrl: new URL(`../model/weapon_rgb_${settings.size}.onnx`, import.meta.url).href,
                       size: settings.size, forceWasm });
}

async function pump() {
  if (!workerReady || busy || !stream || document.hidden || video.readyState < 2) return;
  busy = true; busySince = performance.now();
  try {
    const bitmap = await createImageBitmap(video);
    worker.postMessage({ type: "frame", bitmap, conf: settings.conf, iou: settings.iou }, [bitmap]);
  } catch (e) {
    busy = false;
    setTimeout(pump, 50);
  }
}

// Watchdog: a frame stuck in the worker for 4 s means it hung (GPU device
// lost, driver reset). Restart it instead of freezing the demo.
setInterval(() => {
  if (busy && performance.now() - busySince > 4000) {
    toast("Detector stalled, restarting it.");
    startWorker();
  } else if (!busy) pump();
}, 1000);
document.addEventListener("visibilitychange", () => pump());

// ------------------------------------------------------------------ camera
async function listCameras() {
  const sel = $("cam-select"), cur = sel.value;
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput");
  sel.innerHTML = cams.map((c, i) => `<option value="${c.deviceId}">${c.label || "Camera " + (i + 1)}</option>`).join("")
    || '<option value="">No camera found</option>';
  if (cams.some(c => c.deviceId === cur)) sel.value = cur;
}

async function startCamera() {
  stopCamera();
  const deviceId = $("cam-select").value;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
               width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
    });
  } catch (e) {
    stream = null;
    const why = {
      NotAllowedError: "Camera permission was blocked. Click the camera icon in the address bar and allow it.",
      NotReadableError: "The camera is in use by another app (Zoom, Teams, Camera, or the old Streamlit demo). Close it and try again.",
      NotFoundError: "No camera found. Plug in the webcam and try again.",
      OverconstrainedError: "That camera is no longer available. Pick another from the list.",
    }[e.name] || `Camera error: ${e.message}`;
    toast(why, 9000); chip("st-cam", "bad", "Camera"); return;
  }
  video.srcObject = stream;
  await video.play();
  const track = stream.getVideoTracks()[0];
  track.addEventListener("ended", () => { toast("The camera was disconnected."); stopCamera(); });
  const s = track.getSettings();
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;
  $("stage").style.setProperty("--ar", (video.videoWidth / video.videoHeight).toFixed(4));
  $("stage-empty").hidden = true;
  chip("st-cam", "on", `Camera ${video.videoWidth}×${video.videoHeight} @${Math.round(s.frameRate || 30)}`);
  $("btn-cam").textContent = "Stop camera"; $("btn-cam").className = "btn danger";
  ensureAudio();
  await listCameras();
  if (deviceId) $("cam-select").value = deviceId;
  engine.reset();
  pump();
}

function stopCamera() {
  stream?.getTracks().forEach(t => t.stop());
  stream = null; boxes = [];
  video.srcObject = null;
  $("stage-empty").hidden = false;
  chip("st-cam", "", "Camera");
  $("btn-cam").textContent = "Start camera"; $("btn-cam").className = "btn primary";
  engine.reset();
}

// ------------------------------------------------------------------ overlay (every display frame)
function drawOverlay() {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (stream) {
    const st = engine.state;
    const color = st === ALARM ? "#ff4d47" : st === VETOED ? "#3fcf7f" : st === UNVERIFIED ? "#f5b041" : "#9aa6b8";
    const lw = Math.max(2, overlay.width / 220);
    const fs = Math.round(Math.max(13, overlay.width / 42));
    octx.font = `600 ${fs}px "IBM Plex Mono", monospace`;
    for (const b of boxes) {
      octx.strokeStyle = color; octx.lineWidth = lw;
      octx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
      let label = `${CLASS_NAMES[b.cls]} ${(b.conf * 100).toFixed(0)}%`;
      if (st === VETOED) label += " · body heat";
      const tw = octx.measureText(label).width + 10, th = fs + 8;
      const y = b.y1 > th ? b.y1 - th : b.y1;
      octx.fillStyle = color; octx.fillRect(b.x1, y, tw, th);
      octx.fillStyle = st === ALARM ? "#fff" : "#06110a";
      octx.fillText(label, b.x1 + 5, y + th - 7);
    }
  }
  requestAnimationFrame(drawOverlay);
}
requestAnimationFrame(drawOverlay);

// ------------------------------------------------------------------ thermal sensor
const sensor = new ThermalSerial(
  (amb, obj) => thermal.push(amb, obj),
  ({ connected, message }) => {
    chip("st-sensor", connected ? "warn" : "", connected ? "Sensor…" : "Sensor");
    $("btn-sensor").textContent = connected ? "Disconnect sensor" : "Connect sensor";
    if (!connected) { thermal.clear(); servoSeen = null; servo.armed = false; if (message !== "Disconnected") toast(message); }
  },
  s => (servoSeen = { ...s, at: performance.now() }),
);

// ------------------------------------------------------------------ servo gimbal (10 Hz)
const servo = new ServoController();
let servoSeen = null;   // last "S,pan,tilt,mode" from the firmware

setInterval(() => {
  if (!sensor.port) return;
  const cmd = servo.tick(stream ? boxes : [], video.videoWidth, video.videoHeight);
  if (cmd) sensor.send(cmd);
}, 100);

function renderServo() {
  const fresh = servoSeen && performance.now() - servoSeen.at < 1500;
  const pill = $("servo-pill"), dot = $("gimbal-dot");
  if (!fresh) {
    pill.className = "pill grey";
    pill.textContent = sensor.port ? "NO SERVO DATA" : "OFFLINE";
    $("servo-pos").textContent = sensor.port ? "Flash the servo firmware" : "Connect the ESP32";
    dot.className = "gimbal-dot off";
    return;
  }
  if (servo.state === "WAIT") {
    pill.className = "pill amber";
    pill.textContent = "WAITING";
    $("servo-pos").textContent = "Press Calibrate to start the servos";
    dot.className = "gimbal-dot off";
    return;
  }
  const label = servo.state === "TRACKING" ? "LOCKED ON" : servo.state;
  pill.className = "pill " + (servo.state === "TRACKING" ? "red" : servo.state === "PATROL" ? "blue" : "grey");
  pill.textContent = label;
  $("servo-pos").textContent = `pan ${servoSeen.pan}° · tilt ${servoSeen.tilt}°`;
  // Pan 120° is the gimbal's left; draw it on the left.
  dot.style.left = `${(PAN_MAX - servoSeen.pan) / (PAN_MAX - PAN_MIN) * 100}%`;
  dot.style.top = `${(servoSeen.tilt - TILT_MIN) / (TILT_MAX - TILT_MIN) * 100}%`;
  dot.className = "gimbal-dot" + (servo.state === "TRACKING" ? " track" : "");
}
setInterval(renderServo, 100);

async function toggleSensor() {
  if (sensor.port) { await sensor.disconnect(); return; }
  if (!serialSupported()) { toast("This browser has no Web Serial. Use Chrome or Edge on a laptop.", 9000); return; }
  if ($("sim").checked) { $("sim").checked = false; setSim(false); }
  try { await sensor.connect(); thermal.clear(); }
  catch (e) {
    if (e.name === "NotFoundError") return;   // user closed the picker
    toast(e.name === "InvalidStateError" || /open/i.test(e.message)
      ? "The port is busy. Close Arduino's Serial Monitor (and the old Python demo), then try again."
      : `Could not open the port: ${e.message}`, 9000);
  }
}

function setSim(on) {
  $("sim-row").hidden = !on;
  clearInterval(simTimer);
  thermal.clear();
  if (on) {
    if (sensor.port) sensor.disconnect();
    simTimer = setInterval(() => thermal.push(31.0, parseFloat($("sim-t").value)), 100);
  }
}

// ------------------------------------------------------------------ alarm output
function ensureAudio() {
  if (!audio) try { audio = new AudioContext(); } catch {}
  audio?.resume?.();
}
function beep() {
  if (!audio || !settings.sound) return;
  const o = audio.createOscillator(), g = audio.createGain(), t = audio.currentTime;
  o.type = "square"; o.frequency.value = 1750;
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.25, t + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.22);
  o.connect(g).connect(audio.destination); o.start(t); o.stop(t + 0.24);
}
function syncBeeping() {
  if (engine.state === ALARM && settings.sound) {
    if (!beepTimer) { beep(); beepTimer = setInterval(beep, 420); }
  } else if (beepTimer) { clearInterval(beepTimer); beepTimer = null; }
}

const EVENT_TEXT = { [ALARM]: "WEAPON ALARM", [VETOED]: "Rejected · body heat", [UNVERIFIED]: "Shape only · no sensor" };
engine.onEvent = ev => {
  log.unshift(ev); log.length = Math.min(log.length, 300);
  renderLog();
  if (ev.state === ALARM && settings.notify && Notification.permission === "granted") {
    new Notification("Weapon detected", { body: `${ev.shape}${ev.object != null ? ` · ${ev.object.toFixed(2)} °C` : ""}`, tag: "weapon" });
  }
};
function renderLog() {
  const tb = $("log");
  if (!log.length) { tb.innerHTML = '<tr class="empty"><td colspan="5">No events yet. Alarms and rejected false alarms appear here.</td></tr>'; return; }
  tb.innerHTML = log.map(e => `<tr><td class="num">${e.time}</td><td class="ev-${e.state}">${EVENT_TEXT[e.state]}</td>
    <td>${e.shape}</td><td class="num">${e.object == null ? "-" : e.object.toFixed(2)}</td>
    <td class="num">${e.delta == null ? "-" : (e.delta >= 0 ? "+" : "") + e.delta.toFixed(2)}</td></tr>`).join("");
}
function exportCsv() {
  const rows = [["time", "event", "shape", "object_c", "delta_c"],
    ...log.slice().reverse().map(e => [e.time, EVENT_TEXT[e.state], e.shape, e.object?.toFixed(2) ?? "", e.delta?.toFixed(2) ?? ""])];
  const blob = new Blob([rows.map(r => r.join(",")).join("\n")], { type: "text/csv" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "fusion_event_log.csv" });
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ------------------------------------------------------------------ panels (10 Hz)
const BANNER = {
  [ALARM]: ["b-alarm", "WEAPON DETECTED", "Weapon shape seen and the infrared reading is not body heat"],
  [VETOED]: ["b-veto", "FALSE ALARM REJECTED", "Gun-like shape, but the sensor reads skin temperature: a finger or hand"],
  [UNVERIFIED]: ["b-unv", "WEAPON SHAPE · UNVERIFIED", "No thermal data. Connect the sensor to confirm"],
  [CLEAR]: ["b-clear", "AREA CLEAR", "No weapon-shaped object in view"],
};
const TH_PILL = { [WARM]: ["red", "BODY HEAT"], [COLD]: ["blue", "COLD OBJECT"], [NEUTRAL]: ["grey", "BACKGROUND"], [OFFLINE]: ["amber", "OFFLINE"] };

function drawSpark() {
  const c = $("spark"), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== w * dpr) { c.width = w * dpr; c.height = h * dpr; }
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const now = performance.now(), span = 30000;
  const pts = thermal.history.filter(r => now - r[0] <= span);
  if (pts.length < 2) return;
  const vals = pts.map(p => p[2]).concat(thermal.baseline ?? []);
  let lo = Math.min(...vals) - 0.5, hi = Math.max(...vals) + 0.5;
  const X = t => w - (now - t) / span * w, Y = v => h - 14 - (v - lo) / (hi - lo) * (h - 22);
  g.strokeStyle = "#223046"; g.lineWidth = 1;
  for (let i = 0; i <= 2; i++) { const y = 4 + i * (h - 18) / 2; g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); }
  if (thermal.baseline != null) {
    g.setLineDash([4, 4]); g.strokeStyle = "#8592a6";
    g.beginPath(); g.moveTo(0, Y(thermal.baseline)); g.lineTo(w, Y(thermal.baseline)); g.stroke(); g.setLineDash([]);
  }
  g.strokeStyle = "#ff7a4d"; g.lineWidth = 2; g.beginPath();
  pts.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[2])) : g.moveTo(X(p[0]), Y(p[2]))));
  g.stroke();
  const last = pts[pts.length - 1];
  g.fillStyle = "#ff7a4d"; g.beginPath(); g.arc(X(last[0]), Y(last[2]), 3.5, 0, 7); g.fill();
  g.fillStyle = "#8592a6"; g.font = '10px "IBM Plex Mono", monospace';
  g.fillText(`${hi.toFixed(1)}°`, 2, 12); g.fillText(`${lo.toFixed(1)}°`, 2, h - 16); g.fillText("30 s", 2, h - 2);
  g.fillText("now", w - 22, h - 2);
}

function renderPanels() {
  const th = engine.thermalSnapshot();
  const sim = $("sim").checked;
  const st = stream ? engine.state : null;

  // banner
  const bn = st ? BANNER[st] : ["b-clear", "SYSTEM IDLE", "Start the camera to begin"];
  const b = $("banner");
  b.className = "banner " + bn[0];
  b.firstElementChild.textContent = bn[1];
  b.lastElementChild.textContent = bn[2] + (sim && st ? " · SIMULATED SENSOR" : "");
  $("stage").classList.toggle("alarm", st === ALARM);
  syncBeeping();

  // shape
  if (!stream) { $("shape-val").innerHTML = '<span class="pill grey">OFF</span>'; $("shape-det").textContent = "Camera not running"; }
  else if (boxes.length) {
    $("shape-val").innerHTML = '<span class="pill red">WEAPON SHAPE</span>';
    $("shape-det").textContent = boxes.map(x => `${CLASS_NAMES[x.cls]} ${(x.conf * 100).toFixed(0)}%`).join(", ");
  } else { $("shape-val").innerHTML = '<span class="pill green">NONE</span>'; $("shape-det").textContent = "No gun or L-shape in view"; }

  // thermal
  const [pc, pt] = TH_PILL[th.state];
  $("th-pill").className = "pill " + pc; $("th-pill").textContent = pt;
  $("th-lbl").textContent = "Infrared · temperature" + (sim ? " · SIMULATED" : th.ambient != null ? ` · room ${th.ambient.toFixed(1)} °C` : "");
  if (th.object != null) {
    $("th-temp").textContent = th.object.toFixed(2);
    $("th-temp").style.color = jetColor(th.object);
    const base = thermal.baseline;
    const d = th.object - base;
    $("th-det").textContent = `ΔT ${d >= 0 ? "+" : ""}${d.toFixed(2)} °C vs background ${base.toFixed(2)} °C`;
    $("mk-obj").hidden = false; $("mk-obj").style.left = pct(th.object) + "%"; $("mk-obj").textContent = th.object.toFixed(1);
    $("mk-base").hidden = false; $("mk-base").style.left = pct(base) + "%";
    if (!sim) chip("st-sensor", "on", `Sensor ${th.object.toFixed(1)}°`);
  } else {
    $("th-temp").textContent = "--.-"; $("th-temp").style.color = "";
    $("th-det").textContent = sensor.port ? "Waiting for readings… press EN on the ESP32 if this stays" : "Connect the ESP32 or turn on Simulate";
    $("mk-obj").hidden = $("mk-base").hidden = true;
    if (sensor.port) chip("st-sensor", "warn", "Sensor…");
  }
  drawSpark();

  // decision
  const dv = { [ALARM]: ["red", "ALARM"], [VETOED]: ["green", "REJECTED"], [UNVERIFIED]: ["amber", "UNVERIFIED"], [CLEAR]: ["green", "CLEAR"] };
  const [dc, dt] = st ? dv[st] : ["grey", "IDLE"];
  $("dec-val").innerHTML = `<span class="pill ${dc}">${dt}</span>`;
  $("dec-rule").textContent = engine.settings.strict ? "shape AND cold metal" : "shape AND NOT body-heat";

  $("hud").textContent = stream && workerReady
    ? `${backend.toUpperCase()} · ${detFps.toFixed(1)} det/s · ${detMs.toFixed(0)} ms · ${settings.size}px` : "";
}
setInterval(renderPanels, 100);

// ------------------------------------------------------------------ wiring
function bindRange(id, fmt, apply) {
  const el = $(id), out = $(id + "-v");
  const f = () => { const v = parseFloat(el.value); out.textContent = fmt(v); apply(v); };
  el.addEventListener("input", f); f();
}
bindRange("conf", v => `${Math.round(v * 100)}%`, v => (settings.conf = v));
bindRange("warm", v => `${v.toFixed(1)} °C`, v => (engine.settings.warmDelta = v));
bindRange("cold", v => `${v.toFixed(1)} °C`, v => (engine.settings.coldDelta = v));
bindRange("pan", v => `${v}°`, v => (servo.manual.pan = v));
bindRange("tilt", v => `${v}°`, v => (servo.manual.tilt = v));
$("servo-mode").addEventListener("change", e => { servo.mode = e.target.value; $("servo-manual").hidden = servo.mode !== "manual"; });
$("rev-pan").addEventListener("change", e => (servo.reversePan = e.target.checked));
$("rev-tilt").addEventListener("change", e => (servo.reverseTilt = e.target.checked));
$("sim-t").addEventListener("input", () => ($("sim-v").textContent = `${parseFloat($("sim-t").value).toFixed(1)} °C`));
$("mode").addEventListener("change", e => (engine.settings.strict = e.target.value === "strict"));
$("size").addEventListener("change", e => { settings.size = +e.target.value; startWorker(); });
$("sound").addEventListener("change", e => { settings.sound = e.target.checked; ensureAudio(); });
$("notify").addEventListener("change", async e => {
  settings.notify = e.target.checked;
  if (settings.notify && "Notification" in window && Notification.permission !== "granted") {
    if ((await Notification.requestPermission()) !== "granted") { e.target.checked = settings.notify = false; toast("Notifications were blocked."); }
  }
});
$("btn-cam").addEventListener("click", () => (stream ? stopCamera() : startCamera()));
$("btn-cam-big").addEventListener("click", startCamera);
$("cam-select").addEventListener("change", () => stream && startCamera());
$("btn-sensor").addEventListener("click", toggleSensor);
$("btn-cal").addEventListener("click", () => {
  const b = thermal.calibrate();
  if (b != null && sensor.port) servo.armed = true;
  toast(b == null ? "No sensor reading yet." : `Background set to ${b.toFixed(2)} °C${servo.armed ? " · servos started" : ""}`, 2500);
});
$("sim").addEventListener("change", e => setSim(e.target.checked));
$("btn-csv").addEventListener("click", exportCsv);
$("btn-clear").addEventListener("click", () => { log = []; renderLog(); });
$("btn-fs").addEventListener("click", () => document.fullscreenElement ? document.exitFullscreen() : $("stage").requestFullscreen?.());
document.addEventListener("keydown", e => {
  if (e.target.matches("input, select, textarea")) return;
  if (e.key === "c" || e.key === "C") $("btn-cal").click();
  else if (e.key === "m" || e.key === "M") { $("sound").checked = !$("sound").checked; $("sound").dispatchEvent(new Event("change")); }
  else if (e.key === "f" || e.key === "F") $("btn-fs").click();
});

if (!window.isSecureContext) toast("Open this page over https:// or http://localhost, otherwise the camera and sensor are blocked.", 12000);
if (!serialSupported()) $("note").textContent = "Web Serial is not available in this browser, so only Simulate works for the sensor. Use Chrome or Edge on a laptop.";
navigator.mediaDevices?.enumerateDevices && listCameras().catch(() => {});
startWorker();
