// RGB + thermal decision-level fusion. Port of 05_Code/live_sensors.py, kept
// rule-for-rule identical so the Python tests describe this file too.
//
//   shape   - the YOLO detector sees a weapon-shaped object (any class)
//   thermal - IR object temperature relative to a calibrated background:
//               WARM (skin), COLD (colder than background: metal), NEUTRAL
//   ALARM  = shape AND NOT WARM     (standard)
//   ALARM  = shape AND COLD         (strict)
//   VETOED = shape AND WARM         (finger-gun: camera fooled, thermal not)

export const WARM = "WARM", COLD = "COLD", NEUTRAL = "NEUTRAL", OFFLINE = "OFFLINE";
export const ALARM = "ALARM", VETOED = "VETOED", CLEAR = "CLEAR", UNVERIFIED = "UNVERIFIED";
export const CLASS_NAMES = ["hidden", "partial", "half", "full"];

const PATTERNS = [
  /^T,(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)/,                          // mlx90614_stream
  /amb\s+(-?\d+(?:\.\d+)?)\s*\|\s*obj\s+(-?\d+(?:\.\d+)?)/,          // mlx90614_explore
  /ambient\s+(-?\d+(?:\.\d+)?)\s*C\s*\|\s*object\s+(-?\d+(?:\.\d+)?)/, // mlx90614_test
];

export function parseThermalLine(line) {
  const s = line.trim();
  for (const re of PATTERNS) {
    const m = s.match(re);
    if (m) return [parseFloat(m[1]), parseFloat(m[2])];
  }
  return null;
}

export function classifyThermal(objectC, baselineC, warmDelta = 1.0, coldDelta = 1.0) {
  if (objectC == null || baselineC == null) return OFFLINE;
  const d = objectC - baselineC;
  if (d >= warmDelta) return WARM;
  if (d <= -coldDelta) return COLD;
  return NEUTRAL;
}

export function decide(shapeSeen, thermalState, strict = false) {
  if (!shapeSeen) return CLEAR;
  if (thermalState === OFFLINE) return UNVERIFIED;
  if (thermalState === WARM) return VETOED;
  if (strict && thermalState !== COLD) return CLEAR;
  return ALARM;
}

const STALE_MS = 2000, SMOOTH_MS = 500;

export class ThermalBuffer {
  constructor() { this.history = []; this.baseline = null; }

  push(ambient, object, t = performance.now()) {
    this.history.push([t, ambient, object]);
    const cutoff = t - 60000;
    while (this.history.length && this.history[0][0] < cutoff) this.history.shift();
    // First reading becomes the background; Calibrate retakes it.
    if (this.baseline == null) this.baseline = object;
  }

  // [ambient, median object over the last SMOOTH_MS], or null if stale.
  latest(t = performance.now()) {
    const h = this.history;
    if (!h.length || t - h[h.length - 1][0] > STALE_MS) return null;
    const recent = h.filter(r => t - r[0] <= SMOOTH_MS);
    const rows = recent.length ? recent : [h[h.length - 1]];
    const objs = rows.map(r => r[2]).sort((a, b) => a - b);
    return [rows[rows.length - 1][1], objs[Math.floor(objs.length / 2)]];
  }

  // Highest object reading in the last `ms`. A finger moving in shows up as
  // one or two warm samples before the median catches up.
  peak(ms, t = performance.now()) {
    let hi = null;
    for (const r of this.history) if (t - r[0] <= ms && (hi == null || r[2] > hi)) hi = r[2];
    return hi;
  }

  calibrate() {
    const r = this.latest();
    if (r) this.baseline = r[1];
    return this.baseline;
  }

  clear() { this.history = []; this.baseline = null; }
}

export class FusionEngine {
  constructor(thermal) {
    this.thermal = thermal;
    // confirmMs: how long a shape must persist before it can alarm, so the IR
    // sensor (~0.3 s response) has time to report a finger. The veto is never delayed.
    this.settings = { warmDelta: 1.0, coldDelta: 1.0, strict: false, holdMs: 1500, confirmMs: 500, warmLookbackMs: 800 };
    this.hits = [];          // last 4 detector passes: did any box appear?
    this.shapeSince = null;
    this.state = CLEAR;
    this.thermalState = OFFLINE;
    this.lastAlarmAt = -Infinity;
    this.onEvent = null;     // (eventObject) => void, fired on state transitions
  }

  thermalSnapshot(t = performance.now()) {
    const r = this.thermal.latest(t);
    const obj = r ? r[1] : null;
    const { warmDelta, coldDelta, warmLookbackMs } = this.settings;
    let state = classifyThermal(obj, this.thermal.baseline, warmDelta, coldDelta);
    const peak = r ? this.thermal.peak(warmLookbackMs, t) : null;
    if (state !== OFFLINE && peak != null && peak - this.thermal.baseline >= warmDelta) state = WARM;
    // evidence: the reading the decision rests on (the warm peak for a veto).
    return { ambient: r ? r[0] : null, object: obj, state, evidence: state === WARM ? peak : obj };
  }

  // Call once per detector pass. 2-of-4 debouncing: one stray frame can't
  // trigger; a real object alarms after confirmMs (about half a second).
  update(boxes, t = performance.now()) {
    this.hits.push(boxes.length > 0);
    if (this.hits.length > 4) this.hits.shift();
    const shapeSeen = this.hits.filter(Boolean).length >= 2;
    if (!shapeSeen) this.shapeSince = null;
    else if (this.shapeSince == null) this.shapeSince = t;
    const th = this.thermalSnapshot(t);
    let next = decide(shapeSeen, th.state, this.settings.strict);

    const prev = this.state;
    const confirmed = shapeSeen && t - this.shapeSince >= this.settings.confirmMs;
    if ((next === ALARM || next === UNVERIFIED) && !confirmed && prev !== next) next = CLEAR;
    if (next === ALARM || next === UNVERIFIED) this.lastAlarmAt = t;
    // Latch only against CLEAR: body heat must still be able to cancel an alarm at once.
    else if (next === CLEAR && (prev === ALARM || prev === UNVERIFIED) && t - this.lastAlarmAt < this.settings.holdMs) next = prev;

    this.state = next;
    this.thermalState = th.state;
    if (next !== prev && next !== CLEAR && this.onEvent) {
      const best = boxes.reduce((a, b) => (!a || b.conf > a.conf ? b : a), null);
      const base = this.thermal.baseline;
      this.onEvent({
        time: new Date().toLocaleTimeString(),
        state: next,
        shape: best ? `${CLASS_NAMES[best.cls]} ${(best.conf * 100).toFixed(0)}%` : "-",
        object: th.evidence,
        delta: th.evidence == null || base == null ? null : th.evidence - base,
      });
    }
    return next;
  }

  reset() { this.hits = []; this.shapeSince = null; this.state = CLEAR; }
}
