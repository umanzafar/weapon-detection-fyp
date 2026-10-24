// YOLO11n ONNX pre/post-processing. Pure functions, no DOM, so the same file
// runs in the detector worker and in the Node tests (tests/postprocess.test.mjs).
//
// Export: Ultralytics 8.4.37, format=onnx, opset 12, static imgsz.
// Output "output0" is [1, 4 + nc, N]: rows 0-3 are cx, cy, w, h in input
// pixels, rows 4.. are per-class scores (already sigmoid). No NMS in-graph.

// Matches ultralytics LetterBox(auto=False, center=True): uniform scale to fit,
// centred, padded with grey 114.
export function letterboxParams(srcW, srcH, size) {
  const scale = Math.min(size / srcW, size / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = Math.max(0, Math.round((size - newW) / 2 - 0.1));
  const padY = Math.max(0, Math.round((size - newH) / 2 - 0.1));
  return { scale, newW, newH, padX, padY };
}

// RGBA bytes (canvas ImageData) -> planar RGB float32 in [0, 1], NCHW.
export function rgbaToChw(rgba, size, out = new Float32Array(3 * size * size)) {
  const plane = size * size;
  for (let i = 0, p = 0; p < plane; i += 4, p++) {
    out[p] = rgba[i] / 255;
    out[p + plane] = rgba[i + 1] / 255;
    out[p + 2 * plane] = rgba[i + 2] / 255;
  }
  return out;
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union > 0 ? inter / union : 0;
}

// Class-agnostic: for the alarm any weapon class counts, and two boxes of
// different visibility classes on one object would just be clutter.
export function nms(boxes, iouThres = 0.45, maxDet = 20) {
  boxes.sort((a, b) => b.conf - a.conf);
  const kept = [];
  for (const b of boxes) {
    if (kept.length >= maxDet) break;
    if (kept.every(k => iou(k, b) < iouThres)) kept.push(b);
  }
  return kept;
}

// data: Float32Array of output0, dims: [1, C, N]. Returns boxes in source-image
// pixels: {x1, y1, x2, y2, conf, cls}.
export function decode(data, dims, lb, srcW, srcH, confThres = 0.45, iouThres = 0.45) {
  const C = dims[1], N = dims[2], nc = C - 4;
  const cand = [];
  for (let i = 0; i < N; i++) {
    let best = 0, cls = -1;
    for (let k = 0; k < nc; k++) {
      const s = data[(4 + k) * N + i];
      if (s > best) { best = s; cls = k; }
    }
    if (best < confThres) continue;
    const cx = data[i], cy = data[N + i], w = data[2 * N + i], h = data[3 * N + i];
    const clamp = (v, hi) => Math.max(0, Math.min(hi, v));
    cand.push({
      x1: clamp((cx - w / 2 - lb.padX) / lb.scale, srcW),
      y1: clamp((cy - h / 2 - lb.padY) / lb.scale, srcH),
      x2: clamp((cx + w / 2 - lb.padX) / lb.scale, srcW),
      y2: clamp((cy + h / 2 - lb.padY) / lb.scale, srcH),
      conf: best, cls,
    });
  }
  return nms(cand, iouThres);
}
