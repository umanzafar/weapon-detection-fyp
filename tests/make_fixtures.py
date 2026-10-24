"""
Builds fixtures for the Node tests: for a few RGB val images, the letterboxed
input tensor, the raw ONNX output, and Ultralytics' own final boxes (from the
.pt model) as ground truth for the JS decoder.

    py 12_WebApp/tests/make_fixtures.py
    node --test 12_WebApp/tests/
"""
import glob
import json
import os

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PT = os.path.join(ROOT, "runs", "detect", "weapon_detector_rgb", "weights", "best.pt")
ONNX = os.path.join(ROOT, "12_WebApp", "model", "weapon_rgb_416.onnx")
OUT = os.path.join(ROOT, "12_WebApp", "tests", "fixtures")
SIZE, CONF, IOU, N_IMAGES = 416, 0.25, 0.45, 4


def letterbox(img, size):
    # Mirrors js/postprocess.js letterboxParams + the worker's drawImage.
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    px, py = round((size - nw) / 2 - 0.1), round((size - nh) / 2 - 0.1)
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas


def main():
    os.makedirs(OUT, exist_ok=True)
    # Reference from Ultralytics running the same ONNX file: a .pt model would
    # letterbox to a rectangular (minimal-padding) input and score differently.
    pt = YOLO(ONNX, task="detect")
    sess = ort.InferenceSession(ONNX)
    images = sorted(glob.glob(os.path.join(ROOT, "04_Dataset_RGB", "images", "val", "*.jpg")))
    written = []
    for path in images:
        bgr = cv2.imread(path)
        ref = pt.predict(bgr, imgsz=SIZE, conf=CONF, iou=IOU, agnostic_nms=True, verbose=False)[0].boxes
        if len(ref) == 0:
            continue
        rgb = cv2.cvtColor(letterbox(bgr, SIZE), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        raw = sess.run(None, {"images": x})[0]
        name = os.path.splitext(os.path.basename(path))[0]
        x.astype("<f4").tofile(os.path.join(OUT, f"{name}.input.f32"))
        raw.astype("<f4").tofile(os.path.join(OUT, f"{name}.output.f32"))
        with open(os.path.join(OUT, f"{name}.json"), "w") as f:
            json.dump({
                "width": bgr.shape[1], "height": bgr.shape[0], "size": SIZE, "conf": CONF, "iou": IOU,
                "dims": list(raw.shape),
                "expected": [{"box": b.xyxy[0].tolist(), "conf": float(b.conf), "cls": int(b.cls)} for b in ref],
            }, f, indent=1)
        written.append(name)
        if len(written) == N_IMAGES:
            break
    # The tests read this list, so leftovers from older runs are ignored.
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(written, f)
    print(f"wrote {len(written)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
