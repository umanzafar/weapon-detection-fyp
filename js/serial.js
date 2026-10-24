// ESP32 + MLX90614 over the Web Serial API (Chrome / Edge desktop only).
// Firmware: 10_Hardware/mlx90614_stream, prints "T,<ambient>,<object>" at 10 Hz.
import { parseThermalLine } from "./fusion.js";

export const serialSupported = () => "serial" in navigator;

export class ThermalSerial {
  constructor(onReading, onStatus) {
    this.onReading = onReading;   // (ambient, object) => void
    this.onStatus = onStatus;     // ({connected, message}) => void
    this.port = null;
    this.reader = null;
    this.keepReading = false;
    if (serialSupported()) {
      navigator.serial.addEventListener("disconnect", e => {
        if (e.target === this.port) this._closed("ESP32 unplugged");
      });
    }
  }

  // Must be called from a click: the browser shows its own port picker.
  async connect() {
    const port = await navigator.serial.requestPort();
    await port.open({ baudRate: 115200 });
    // Opening the port toggles DTR/RTS, which resets the ESP32 on most
    // DevKits; it reboots and starts streaming in about 1.5 s.
    this.port = port;
    this.keepReading = true;
    this.onStatus({ connected: true, message: "Connected, waiting for readings…" });
    this._loop();
  }

  async _loop() {
    const decoder = new TextDecoder();
    let buf = "";
    while (this.port && this.port.readable && this.keepReading) {
      this.reader = this.port.readable.getReader();
      try {
        for (;;) {
          const { value, done } = await this.reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          let nl;
          while ((nl = buf.indexOf("\n")) >= 0) {
            const r = parseThermalLine(buf.slice(0, nl));
            buf = buf.slice(nl + 1);
            if (r) this.onReading(r[0], r[1]);
          }
          if (buf.length > 4096) buf = "";   // garbage with no newline (wrong baud)
        }
      } catch (e) {
        // Framing/buffer errors are recoverable: loop round and take a new reader.
        if (!this.keepReading) break;
      } finally {
        try { this.reader.releaseLock(); } catch {}
      }
    }
  }

  async disconnect() {
    this.keepReading = false;
    try { await this.reader?.cancel(); } catch {}
    try { await this.port?.close(); } catch {}
    this._closed("Disconnected");
  }

  _closed(message) {
    this.port = null;
    this.keepReading = false;
    this.onStatus({ connected: false, message });
  }
}
