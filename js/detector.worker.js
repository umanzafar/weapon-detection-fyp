// Runs YOLO off the main thread so the video never waits for inference.
// Protocol (all messages are plain objects):
//   in:  {type:"load", modelUrl, size}       out: {type:"ready", backend, warmupMs} | {type:"error", message}
//   in:  {type:"frame", bitmap, conf, iou}   out: {type:"result", boxes, ms}
// The main thread sends a frame only after the previous result arrives, so
// frames are dropped at the source and nothing can ever queue up.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.mjs";
import { letterboxParams, rgbaToChw, decode } from "./postprocess.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
// Threads need cross-origin isolation, which GitHub Pages can't enable.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

let session = null, size = 416, canvas = null, ctx = null, input = null, inputName = "images";

async function createSession(url, forceWasm) {
  const attempts = [];
  if ("gpu" in navigator && !forceWasm) attempts.push("webgpu");
  attempts.push("wasm");
  let lastErr;
  for (const ep of attempts) {
    try {
      const s = await ort.InferenceSession.create(url, { executionProviders: [ep], graphOptimizationLevel: "all" });
      return { s, backend: ep };
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

async function load({ modelUrl, size: sz, forceWasm }) {
  size = sz;
  canvas = new OffscreenCanvas(size, size);
  ctx = canvas.getContext("2d", { willReadFrequently: true });
  input = new Float32Array(3 * size * size);
  const { s, backend } = await createSession(modelUrl, forceWasm);
  session = s;
  inputName = s.inputNames[0];
  // The first run compiles kernels (seconds on WebGPU); do it now, not on the first real frame.
  const t0 = performance.now();
  await session.run({ [inputName]: new ort.Tensor("float32", input, [1, 3, size, size]) });
  postMessage({ type: "ready", backend, warmupMs: performance.now() - t0 });
}

async function detect({ bitmap, conf, iou }) {
  const t0 = performance.now();
  const W = bitmap.width, H = bitmap.height;
  const lb = letterboxParams(W, H, size);
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(bitmap, lb.padX, lb.padY, lb.newW, lb.newH);
  bitmap.close();
  rgbaToChw(ctx.getImageData(0, 0, size, size).data, size, input);
  const out = await session.run({ [inputName]: new ort.Tensor("float32", input, [1, 3, size, size]) });
  const t = out[session.outputNames[0]];
  const boxes = decode(t.data, t.dims, lb, W, H, conf, iou);
  postMessage({ type: "result", boxes, ms: performance.now() - t0, width: W, height: H });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") await load(data);
    else if (data.type === "frame") {
      if (!session) { data.bitmap.close(); postMessage({ type: "result", boxes: [], ms: 0 }); return; }
      await detect(data);
    }
  } catch (e) {
    if (data.bitmap) try { data.bitmap.close(); } catch {}
    postMessage({ type: "error", message: String(e && e.message || e), during: data.type });
  }
};
