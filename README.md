# Concealed Weapon Detection Console

Live RGB + thermal sensor fusion for concealed weapon detection, running entirely in the browser.
Final Year Project, Department of Computer Systems Engineering, MUST (2022–2026).

**Live site:** https://umanzafar.github.io/weapon-detection-fyp/ · **Console:** https://umanzafar.github.io/weapon-detection-fyp/console.html

A webcam finds weapon-shaped objects with a YOLO11n detector; an MLX90614 infrared sensor on an
ESP32 reads their temperature. The alarm fires only when both agree, so a finger shaped like a gun
(warm skin) is rejected instead of raising a false alarm.

| Camera sees | Infrared reads | Result |
|---|---|---|
| Weapon shape | Not body heat (room temperature or colder) | **Weapon alarm** |
| Weapon shape | Body heat (finger / hand) | **False alarm rejected** |
| Nothing | Anything | Clear |
| Weapon shape | No sensor | Unverified |

## Run it

Needs **Chrome or Edge on a laptop** (Web Serial is not available in Firefox, Safari or on phones).

1. Flash `firmware/mlx90614_stream/mlx90614_stream.ino` to the ESP32 (libraries: Adafruit MLX90614,
   ESP32Servo). Wiring: sensor VIN→3V3, GND→GND, SDA→D21, SCL→D22; pan servo signal→GPIO18,
   tilt servo signal→GPIO19, servo power from a **separate 5 V supply** whose GND is tied to an
   ESP32 GND. Close Arduino's Serial Monitor afterwards.
2. Open the GitHub Pages link, or locally: `python -m http.server 8000` in this folder, then
   `http://localhost:8000`.
3. **Start camera** → **Connect sensor** (pick the CP210x port) → point the sensor at empty space →
   **Calibrate**. Hold objects 1–3 cm from the sensor.

No ESP32? Tick **Simulate** to drive the temperature with a slider (the screen is labelled SIMULATED).

Shortcuts: `C` calibrate, `M` mute, `F` fullscreen.

**Servo gimbal.** The servos start patrolling (slow left–right sweep) as soon as the ESP32 powers up.
Once connected, the console's *Servo gimbal* card picks the mode: **Patrol + lock on target**
(default: turns to the detected weapon and follows it, back to patrol 2 s after it leaves view),
**Patrol only**, **Hold center**, or **Manual**. If the gimbal turns the wrong way, tick *Reverse pan*
/ *Reverse tilt*. If the infrared sensor is mounted on the gimbal, use *Hold center* for the finger
and gun tests, because the sensor must stay 1–3 cm from the object.

## How it works

```
 webcam ──► <video> (native 30 fps) ──► createImageBitmap ──► Web Worker
                                                               │ letterbox 416², ONNX Runtime Web
                                                               │ WebGPU (WASM fallback), NMS
 ESP32 ──USB──► Web Serial ──► ThermalBuffer                   ▼
   "T,amb,obj" @10 Hz            │ median, peak, baseline    boxes
                                 └──────────► FusionEngine ◄──┘ ──► banner, alarm tone, log
```

- **Video never waits for the AI.** The camera plays natively; only one frame is ever in flight to
  the worker, and newer frames are dropped rather than queued.
- **Watchdog.** A detector pass stuck for 4 s restarts the worker; repeated GPU failures fall back to WASM.
- **Fusion rule** (`js/fusion.js`): shape must persist 0.5 s before it can alarm (2-of-4 detector
  passes plus a confirmation window); any skin-warm reading in the last 0.8 s vetoes at once, so a
  finger moving in never beeps even while the sensor is still warming up.
- Detector: YOLO11n, 4 visibility classes (hidden / partial / half / full), trained on our own photos
  plus 302 vetted pistol images, exported to ONNX at 416 px and 320 px.

## Tests

```
pip install ultralytics onnxruntime opencv-python
python tests/make_fixtures.py      # needs the project's training data and weights
npm install && npm test
```

The tests check the JavaScript decoder against Ultralytics' own output on real validation images,
check that ONNX Runtime Web reproduces Python ONNX Runtime, and cover the fusion rule, including a
finger approaching with a still-rising temperature.

## Limits

- The MLX90614 is a single-point, wide-angle (90°) sensor: it averages everything in its view, so
  objects must be within a few centimetres. It cannot see shape; an imaging thermal array (e.g. MLX90640)
  is the next step.
- At high room temperature the contrast shrinks: we measured a concealed weapon 3.19 °C colder than the
  same shirt without it in a 31.5 °C room.
- A weapon taken apart and hidden in pieces is out of scope; that needs metal detection or X-ray.
