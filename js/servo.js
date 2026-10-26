// Decides what the pan/tilt gimbal should do, and turns that into firmware
// commands. Pure logic: tick() returns the command line to send (or null), so
// it is unit-tested without hardware (tests/webapp.test.mjs).
//
// Modes:
//   lock    - patrol while clear; when a weapon shape is in view, turn to point
//             at it and follow it; back to patrol LOST_MS after it disappears
//   patrol  - always sweep
//   center  - hold straight ahead
//   manual  - hold the slider angles
//
// The firmware falls back to patrol after 3 s without a command, so every
// non-patrol state is re-sent at least every KEEPALIVE_MS.

export const PAN_MIN = 60, PAN_MAX = 120, TILT_MIN = 70, TILT_MAX = 110;
const KEEPALIVE_MS = 1000, AIM_MIN_INTERVAL_MS = 100, LOST_MS = 2000, SMOOTH = 0.35;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export class ServoController {
  constructor() {
    this.mode = "lock";
    this.reversePan = false;
    this.reverseTilt = false;
    this.manual = { pan: 90, tilt: 90 };
    this.aim = null;            // smoothed {pan, tilt} while tracking
    this.lastSeenAt = -Infinity;
    this.lastCmd = null;
    this.lastSentAt = -Infinity;
    this.state = "PATROL";      // what the console shows: PATROL / TRACKING / HOLD / CENTER / MANUAL
  }

  // Camera's ~60° horizontal view maps onto the 60° pan window, so the
  // gimbal points roughly where the object is.
  boxToAngles(box, w, h) {
    const cx = (box.x1 + box.x2) / 2 / w, cy = (box.y1 + box.y2) / 2 / h;
    const panDir = this.reversePan ? -1 : 1, tiltDir = this.reverseTilt ? -1 : 1;
    return {
      pan: clamp(90 + (0.5 - cx) * (PAN_MAX - PAN_MIN) * panDir, PAN_MIN, PAN_MAX),
      tilt: clamp(90 + (cy - 0.5) * (TILT_MAX - TILT_MIN) * tiltDir, TILT_MIN, TILT_MAX),
    };
  }

  _emit(cmd, t) {
    const changed = cmd !== this.lastCmd;
    const due = t - this.lastSentAt >= KEEPALIVE_MS;
    const rateOk = t - this.lastSentAt >= AIM_MIN_INTERVAL_MS;
    if ((changed && rateOk) || due) {
      this.lastCmd = cmd; this.lastSentAt = t;
      return cmd;
    }
    return null;
  }

  // boxes: detector boxes in frame pixels; w/h: frame size. Call ~10x per second.
  tick(boxes, w, h, t = performance.now()) {
    if (this.mode === "patrol") { this.state = "PATROL"; this.aim = null; return this._emit("P", t); }
    if (this.mode === "center") { this.state = "CENTER"; this.aim = null; return this._emit("C", t); }
    if (this.mode === "manual") {
      this.state = "MANUAL"; this.aim = null;
      return this._emit(`A${Math.round(this.manual.pan)},${Math.round(this.manual.tilt)}`, t);
    }
    // lock
    if (boxes && boxes.length && w && h) {
      const best = boxes.reduce((a, b) => (b.conf > a.conf ? b : a));
      const target = this.boxToAngles(best, w, h);
      this.aim = this.aim
        ? { pan: this.aim.pan + SMOOTH * (target.pan - this.aim.pan), tilt: this.aim.tilt + SMOOTH * (target.tilt - this.aim.tilt) }
        : target;
      this.lastSeenAt = t;
      this.state = "TRACKING";
      return this._emit(`A${Math.round(this.aim.pan)},${Math.round(this.aim.tilt)}`, t);
    }
    if (t - this.lastSeenAt < LOST_MS) { this.state = "HOLD"; return this._emit("H", t); }
    this.aim = null;
    this.state = "PATROL";
    return this._emit("P", t);
  }
}
