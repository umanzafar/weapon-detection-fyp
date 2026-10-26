// node --test 12_WebApp/tests/      (run make_fixtures.py first)
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { letterboxParams, decode } from "../js/postprocess.js";
import {
  parseThermalLine, classifyThermal, decide, ThermalBuffer, FusionEngine,
  WARM, COLD, NEUTRAL, OFFLINE, ALARM, VETOED, CLEAR, UNVERIFIED,
} from "../js/fusion.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIX = path.join(HERE, "fixtures");
const INDEX = path.join(FIX, "index.json");
const fixtures = existsSync(INDEX) ? JSON.parse(readFileSync(INDEX)) : [];
const f32 = p => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };

function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = ix * iy;
  return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i);
}

test("letterbox matches ultralytics for a 640x480 webcam frame", () => {
  assert.deepEqual(letterboxParams(640, 480, 416), { scale: 0.65, newW: 416, newH: 312, padX: 0, padY: 52 });
});

test("fixtures exist", () => assert.ok(fixtures.length > 0, "run py 12_WebApp/tests/make_fixtures.py"));

for (const name of fixtures) {
  test(`JS decoder reproduces Ultralytics boxes: ${name}`, () => {
    const meta = JSON.parse(readFileSync(path.join(FIX, name + ".json")));
    const raw = f32(path.join(FIX, name + ".output.f32"));
    const lb = letterboxParams(meta.width, meta.height, meta.size);
    const got = decode(raw, meta.dims, lb, meta.width, meta.height, meta.conf, meta.iou);
    assert.equal(got.length, meta.expected.length, "box count");
    for (const e of meta.expected) {
      const m = got.find(g => iou([g.x1, g.y1, g.x2, g.y2], e.box) > 0.9 && g.cls === e.cls);
      assert.ok(m, `no match for ${JSON.stringify(e)} in ${JSON.stringify(got)}`);
      assert.ok(Math.abs(m.conf - e.conf) < 0.03, `conf ${m.conf} vs ${e.conf}`);
    }
  });
}

test("onnxruntime-web (the browser runtime) loads the model and matches Python ORT", async () => {
  const ort = await import("onnxruntime-web").catch(() => null);
  if (!ort || !fixtures.length) return;   // optional: npm i onnxruntime-web@1.20.1 in tests/
  ort.env.wasm.numThreads = 1;
  const model = readFileSync(path.join(HERE, "..", "model", "weapon_rgb_416.onnx"));
  const s = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
  const name = fixtures[0];
  const meta = JSON.parse(readFileSync(path.join(FIX, name + ".json")));
  const x = f32(path.join(FIX, name + ".input.f32"));
  const out = (await s.run({ [s.inputNames[0]]: new ort.Tensor("float32", Float32Array.from(x), [1, 3, 416, 416]) }))[s.outputNames[0]];
  assert.deepEqual(Array.from(out.dims), meta.dims);
  const ref = f32(path.join(FIX, name + ".output.f32"));
  let maxDiff = 0;
  for (let i = 0; i < ref.length; i++) maxDiff = Math.max(maxDiff, Math.abs(ref[i] - out.data[i]));
  assert.ok(maxDiff < 1e-2, `max abs diff ${maxDiff}`);
});

// ---- fusion rule: same cases as 05_Code/tests/test_live_sensors.py
test("parses all three sketch formats and ignores noise", () => {
  assert.deepEqual(parseThermalLine("T,31.49,29.84\r"), [31.49, 29.84]);
  assert.deepEqual(parseThermalLine("amb  31.49 | obj  29.84 | d  -1.65 | avg 29.90"), [31.49, 29.84]);
  assert.deepEqual(parseThermalLine("ambient 31.49 C | object 29.84 C"), [31.49, 29.84]);
  for (const l of ["", "=== BOOT OK - mlx90614_stream ===", "ERR,read failed (NaN)"]) assert.equal(parseThermalLine(l), null);
});

test("thermal state and decision table", () => {
  assert.equal(classifyThermal(34.0, 31.5), WARM);
  assert.equal(classifyThermal(29.84, 31.5), COLD);
  assert.equal(classifyThermal(31.7, 31.5), NEUTRAL);
  assert.equal(classifyThermal(null, 31.5), OFFLINE);
  assert.equal(decide(true, WARM), VETOED);
  assert.equal(decide(true, COLD), ALARM);
  assert.equal(decide(true, NEUTRAL), ALARM);
  assert.equal(decide(true, NEUTRAL, true), CLEAR);
  assert.equal(decide(true, OFFLINE), UNVERIFIED);
  for (const s of [WARM, COLD, NEUTRAL, OFFLINE]) assert.equal(decide(false, s), CLEAR);
});

// ---- servo gimbal
test("servo: status lines parse and are not mistaken for temperature", async () => {
  const { parseServoLine } = await import("../js/serial.js");
  assert.deepEqual(parseServoLine("S,84,90,PATROL\r"), { pan: 84, tilt: 90, mode: "PATROL" });
  assert.equal(parseServoLine("T,31.4,31.2"), null);
  assert.equal(parseThermalLine("S,84,90,PATROL"), null);
});

test("servo: lock mode patrols, tracks a box, holds, then patrols again", async () => {
  const { ServoController } = await import("../js/servo.js");
  const s = new ServoController();
  let t = 0;
  const step = boxes => { t += 100; return s.tick(boxes, 640, 480, t); };
  assert.equal(step([]), "P");
  assert.equal(s.state, "PATROL");
  // Object on the left of the frame -> pan above 90 (gimbal's left).
  const left = [{ x1: 40, y1: 200, x2: 120, y2: 280, conf: 0.9, cls: 3 }];
  const cmd = step(left);
  assert.match(cmd, /^A\d+,\d+$/);
  assert.ok(+cmd.slice(1).split(",")[0] > 100, cmd);
  assert.equal(s.state, "TRACKING");
  assert.equal(step([]), "H", "brief loss holds position");
  for (let i = 0; i < 25; i++) step([]);
  assert.equal(s.state, "PATROL");
  s.reversePan = true;
  assert.ok(s.boxToAngles(left[0], 640, 480).pan < 80, "reverse flips direction");
});

test("servo: keepalive is re-sent at least once a second, angles stay in range", async () => {
  const { ServoController, PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX } = await import("../js/servo.js");
  const s = new ServoController();
  s.mode = "center";
  let sent = 0;
  for (let t = 0; t < 5000; t += 100) if (s.tick([], 640, 480, t)) sent++;
  assert.ok(sent >= 5, `sent ${sent} in 5 s`);
  const corner = s.boxToAngles({ x1: 0, y1: 0, x2: 1, y2: 1 }, 640, 480);
  assert.ok(corner.pan >= PAN_MIN && corner.pan <= PAN_MAX && corner.tilt >= TILT_MIN && corner.tilt <= TILT_MAX);
});

// Detector and sensor both at 10 Hz, as in the real demo.
function rig() {
  const th = new ThermalBuffer();
  const eng = new FusionEngine(th);
  const events = [];
  eng.onEvent = e => events.push(e.state);
  let t = 1000;
  th.push(31.5, 31.5, t);                         // background
  const feed = (obj, shape) => {
    t += 100; th.push(31.5, obj, t);
    return eng.update(shape ? [{ x1: 0, y1: 0, x2: 10, y2: 10, conf: 0.8, cls: 3 }] : [], t);
  };
  return { eng, events, feed };
}

test("engine: room-temperature gun alarms within ~0.6 s, latches, then clears", () => {
  const { eng, events, feed } = rig();
  const states = [];
  for (let i = 0; i < 8; i++) states.push(feed(31.4, true));
  assert.equal(states[0], CLEAR, "one frame is not enough");
  const first = states.indexOf(ALARM);
  assert.ok(first > 0 && first <= 6, `alarm at frame ${first}`);
  assert.equal(feed(31.4, false), ALARM, "latched through a dropped frame");
  for (let i = 0; i < 25; i++) feed(31.4, false);
  assert.equal(eng.state, CLEAR, "latch expires");
  assert.deepEqual(events, [ALARM]);
});

test("engine: finger approaching never alarms, even while its temperature is still rising", () => {
  const { eng, events, feed } = rig();
  // Camera sees the finger-gun from the first frame; the IR reading climbs over ~0.4 s.
  for (const obj of [31.6, 32.0, 32.8, 33.6, 34.1, 34.2, 34.2, 34.2, 34.2, 34.2]) feed(obj, true);
  assert.equal(eng.state, VETOED);
  assert.ok(!events.includes(ALARM), `events: ${events}`);
  // Finger pulled away: shape and heat fade together, still no alarm.
  for (const obj of [33.0, 32.0, 31.6, 31.5, 31.5, 31.5]) feed(obj, false);
  assert.ok(!events.includes(ALARM), `events: ${events}`);
});

test("engine: body heat cancels a running alarm immediately", () => {
  const { eng, feed } = rig();
  for (let i = 0; i < 8; i++) feed(31.4, true);
  assert.equal(eng.state, ALARM);
  feed(34.2, true);
  assert.equal(eng.state, VETOED);
});
