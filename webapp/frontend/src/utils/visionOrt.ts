import * as ort from "onnxruntime-web/wasm";
import type { MacroTriage } from "../types/arborgrid";

const IMGSZ = 1024;
const WORK_SZ = 640; // Grade 640x640 idêntica ao pipeline V3 (run_v3_inference_and_update_map.py)

const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
const IMAGENET_STD = [0.229, 0.224, 0.225] as const;

// Sigmoid^{-1}(0.50)=0, Sigmoid^{-1}(0.40)=ln(0.4/0.6)
const LOGIT_THRESH_COPA = 0.0;
const LOGIT_THRESH_WIRE = Math.log(0.4 / 0.6);

// Paleta semântica harmônica:
// - Copa isolada: Verde | Fiação isolada: Amarelo
// - Margem de vento: Laranja suave (zona contínua sem furo central)
// - Interseção real (Copa ∩ Fiação): Vermelho (mistura natural quando ambas estão ativas)
const RGB_COPA: readonly [number, number, number] = [45, 195, 60];
const RGB_WIRE: readonly [number, number, number] = [250, 204, 21];
const RGB_INTERSECTION: readonly [number, number, number] = [220, 38, 38];
const RGB_WIND: readonly [number, number, number] = [249, 115, 22];

export interface VisionLayerVisibility {
  copa: boolean;
  fiacao: boolean;
  vento?: boolean;
  opacity: number;
  borderThickness?: number;
}

export interface VisionInferenceResult {
  inferenceMs: number;
  postprocessMs: number;
  threadsUsed: number;
  crossOriginIsolated: boolean;
  triage: MacroTriage;
  copaPx: number;
  wirePx: number;
  trueOverlapPx: number;
  contactDirectPx: number;
  windBufferPx: number;
  masks: {
    width: number;
    height: number;
    copa: Uint8Array;
    wire: Uint8Array;
    windZone: Uint8Array;
  };
}

const MODEL_CACHE_NAME = "arborgrid-vision-onnx-v1";

function buildModelCacheKey(modelUrl: string, cacheVersion?: string): string {
  return cacheVersion
    ? `${modelUrl}${modelUrl.includes("?") ? "&" : "?"}v=${encodeURIComponent(cacheVersion)}`
    : modelUrl;
}

export async function isModelCached(
  modelUrl: string,
  cacheVersion?: string,
): Promise<boolean> {
  if (typeof caches === "undefined") return false;
  try {
    const cache = await caches.open(MODEL_CACHE_NAME);
    const cacheKey = buildModelCacheKey(modelUrl, cacheVersion);
    const matched = await cache.match(cacheKey);
    return Boolean(matched && matched.ok);
  } catch {
    return false;
  }
}

export type OnnxProgressListener = (loadedMb: number, totalMb: number, statusText: string) => void;

let cachedSession: ort.InferenceSession | null = null;
let sessionPromise: Promise<ort.InferenceSession> | null = null;
let lastProgress: { loadedMb: number; totalMb: number; statusText: string } | null = null;
const progressListeners = new Set<OnnxProgressListener>();

export function isOnnxSessionReady(): boolean {
  return cachedSession !== null;
}

export function subscribeOnnxProgress(listener: OnnxProgressListener): () => void {
  progressListeners.add(listener);
  if (lastProgress) {
    listener(lastProgress.loadedMb, lastProgress.totalMb, lastProgress.statusText);
  }
  return () => {
    progressListeners.delete(listener);
  };
}

function getRuntimeThreads(): { threads: number; isolated: boolean } {
  const isolated = typeof self !== "undefined" && Boolean(self.crossOriginIsolated);
  if (!isolated) {
    return { threads: 1, isolated: false };
  }
  const hw = typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 8 : 4;
  return { threads: Math.max(2, Math.min(hw, 12)), isolated: true };
}

function configureOrtEnv(): { threads: number; isolated: boolean } {
  const info = getRuntimeThreads();
  // URL absoluta: no servidor de desenvolvimento, o Vite reescreve import("/ort/...") com
  // "?import" e recusa servir arquivos de public/ assim; com a origem completa ele não mexe.
  ort.env.wasm.wasmPaths = new URL("/ort/", self.location.origin).href;
  ort.env.wasm.simd = true;
  ort.env.wasm.proxy = true;
  ort.env.wasm.numThreads = info.threads;
  return info;
}

export async function ensureOnnxSession(
  modelUrl: string,
  onProgress?: OnnxProgressListener,
  cacheVersion?: string,
): Promise<ort.InferenceSession> {
  if (cachedSession) return cachedSession;

  let unsub: (() => void) | null = null;
  if (onProgress) {
    unsub = subscribeOnnxProgress(onProgress);
  }

  const emitProgress = (loadedMb: number, totalMb: number, statusText: string) => {
    lastProgress = { loadedMb, totalMb, statusText };
    for (const listener of progressListeners) {
      listener(loadedMb, totalMb, statusText);
    }
  };

  if (!sessionPromise) {
    sessionPromise = (async () => {
      configureOrtEnv();
      emitProgress(0, 51.8, "Preparando ferramenta de análise visual...");

      const cacheKey = buildModelCacheKey(modelUrl, cacheVersion);

      let buffer: Uint8Array | null = null;

      if (typeof caches !== "undefined") {
        try {
          const cache = await caches.open(MODEL_CACHE_NAME);
          const cachedResp = await cache.match(cacheKey);
          if (cachedResp && cachedResp.ok) {
            const ab = await cachedResp.arrayBuffer();
            if (ab.byteLength > 0) {
              buffer = new Uint8Array(ab);
              const mb = buffer.byteLength / (1024 * 1024);
              emitProgress(mb, mb, "Inicializando análise visual...");
            }
          }
        } catch {
          buffer = null;
        }
      }

      if (!buffer) {
        const response = await fetch(modelUrl);
        if (!response.ok) {
          throw new Error("Não foi possível preparar a análise visual. Tente novamente.");
        }

        const totalBytes = Number(response.headers.get("content-length") || 54289646);
        const totalMb = totalBytes / (1024 * 1024);

        if (response.body) {
          const reader = response.body.getReader();
          const chunks: Uint8Array[] = [];
          let received = 0;
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            if (value) {
              chunks.push(value);
              received += value.length;
              emitProgress(
                received / (1024 * 1024),
                totalMb,
                "Preparando ferramenta de análise visual...",
              );
            }
          }
          buffer = new Uint8Array(received);
          let offset = 0;
          for (const chunk of chunks) {
            buffer.set(chunk, offset);
            offset += chunk.length;
          }
        } else {
          buffer = new Uint8Array(await response.arrayBuffer());
        }

        if (typeof caches !== "undefined") {
          try {
            const cache = await caches.open(MODEL_CACHE_NAME);
            const copy = new ArrayBuffer(buffer.byteLength);
            new Uint8Array(copy).set(buffer);
            await cache.put(
              cacheKey,
              new Response(copy, {
                status: 200,
                headers: {
                  "Content-Type": "application/octet-stream",
                  "Content-Length": String(buffer.byteLength),
                },
              }),
            );
          } catch {
            // Ignora erro de Cache Storage (cota ou modo privado)
          }
        }

        emitProgress(totalMb, totalMb, "Inicializando análise visual...");
      }

      const session = await ort.InferenceSession.create(buffer, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      cachedSession = session;
      lastProgress = null;
      return session;
    })();
  }

  try {
    return await sessionPromise;
  } catch (err) {
    sessionPromise = null;
    lastProgress = null;
    throw err;
  } finally {
    unsub?.();
  }
}

function filterSmallComponents(
  mask: Uint8Array,
  width: number,
  height: number,
  minArea: number,
  keepDominantRatio = 0,
): Uint8Array {
  const n = width * height;
  const labels = new Int32Array(n);
  const queue = new Int32Array(n);
  const areas: number[] = [0];
  let currentLabel = 0;

  for (let i = 0; i < n; i++) {
    if (mask[i] === 0 || labels[i] !== 0) continue;
    currentLabel += 1;
    let head = 0;
    let tail = 0;
    queue[tail++] = i;
    labels[i] = currentLabel;

    while (head < tail) {
      const idx = queue[head++];
      const y = (idx / width) | 0;
      const x = idx - y * width;

      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          const nidx = ny * width + nx;
          if (mask[nidx] === 1 && labels[nidx] === 0) {
            labels[nidx] = currentLabel;
            queue[tail++] = nidx;
          }
        }
      }
    }
    areas.push(tail);
  }

  let maxArea = 0;
  for (let k = 1; k < areas.length; k++) {
    if (areas[k] > maxArea) maxArea = areas[k];
  }
  const threshold =
    keepDominantRatio > 0 ? Math.max(minArea, maxArea * keepDominantRatio) : minArea;

  const clean = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const lbl = labels[i];
    if (lbl > 0 && areas[lbl] >= threshold) {
      clean[i] = 1;
    }
  }
  return clean;
}

function exactSquaredDistanceTransform(
  mask: Uint8Array,
  width: number,
  height: number,
): Float64Array {
  // Felzenszwalb & Huttenlocher (2012) exact 2D squared Euclidean Distance Transform in O(W*H)
  const inf = 1e6;
  const g = new Float64Array(width * height);

  for (let x = 0; x < width; x++) {
    g[x] = mask[x] === 1 ? 0 : inf;
    for (let y = 1; y < height; y++) {
      const idx = y * width + x;
      const prev = g[(y - 1) * width + x];
      g[idx] = mask[idx] === 1 ? 0 : Math.min(inf, prev + 1);
    }
    for (let y = height - 2; y >= 0; y--) {
      const idx = y * width + x;
      const down = g[(y + 1) * width + x] + 1;
      if (down < g[idx]) g[idx] = down;
    }
  }

  const dt = new Float64Array(width * height);
  const v = new Int32Array(width);
  const z = new Float64Array(width + 1);

  for (let y = 0; y < height; y++) {
    const rowOff = y * width;
    let k = 0;
    v[0] = 0;
    z[0] = -Infinity;
    z[1] = Infinity;

    const fCol = (u: number) => {
      const val = g[rowOff + u];
      return val * val;
    };

    for (let q = 1; q < width; q++) {
      const fq = fCol(q);
      let s = 0;
      while (true) {
        const vk = v[k];
        s = (fq + q * q - (fCol(vk) + vk * vk)) / (2 * q - 2 * vk);
        if (s <= z[k] && k > 0) {
          k -= 1;
        } else {
          break;
        }
      }
      if (s <= z[k] && k === 0) {
        // Defensive guard (unreachable with z[0] = -Infinity)
        v[0] = q;
      } else {
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = Infinity;
      }
    }

    k = 0;
    for (let q = 0; q < width; q++) {
      while (z[k + 1] < q) {
        k += 1;
      }
      const vk = v[k];
      const dx = q - vk;
      dt[rowOff + q] = dx * dx + fCol(vk);
    }
  }

  return dt;
}

export async function runOnnxInference(
  session: ort.InferenceSession,
  imageEl: HTMLImageElement,
): Promise<VisionInferenceResult> {
  const { threads, isolated } = getRuntimeThreads();

  // 1. Pré-processamento 1024x1024 RGB ImageNet (NCHW Float32), equivalente ao cv2.resize(1024, 1024)
  const offscreen = document.createElement("canvas");
  offscreen.width = IMGSZ;
  offscreen.height = IMGSZ;
  const ctx = offscreen.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Seu navegador não conseguiu processar a imagem.");
  ctx.drawImage(imageEl, 0, 0, IMGSZ, IMGSZ);
  const rgba = ctx.getImageData(0, 0, IMGSZ, IMGSZ).data;

  const pixelsPerChannel = IMGSZ * IMGSZ;
  const inputData = new Float32Array(3 * pixelsPerChannel);
  for (let i = 0; i < pixelsPerChannel; i++) {
    const p = i * 4;
    inputData[i] = (rgba[p] / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    inputData[pixelsPerChannel + i] = (rgba[p + 1] / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    inputData[2 * pixelsPerChannel + i] =
      (rgba[p + 2] / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }

  const inputName = session.inputNames[0] || "pixel_values";
  const tensor = new ort.Tensor("float32", inputData, [1, 3, IMGSZ, IMGSZ]);

  const t0 = performance.now();
  const outputs = await session.run({ [inputName]: tensor });
  const t1 = performance.now();

  const outputName = session.outputNames[0] || "logits";
  const outTensor = outputs[outputName];
  const logits = outTensor.data as Float32Array;
  const [, , outH, outW] = outTensor.dims; // esperado [1, 3, 256, 256]

  // 2. Interpolação bilinear align_corners=False (256 -> 640) + limiares exatos V3
  const tPost0 = performance.now();
  const workN = WORK_SZ * WORK_SZ;
  const rawCopa = new Uint8Array(workN);
  const rawWire = new Uint8Array(workN);

  const chStride = outH * outW;
  const ratioY = outH / WORK_SZ;
  const ratioX = outW / WORK_SZ;

  const x0Arr = new Int32Array(WORK_SZ);
  const x1Arr = new Int32Array(WORK_SZ);
  const dxArr = new Float32Array(WORK_SZ);
  for (let x = 0; x < WORK_SZ; x++) {
    const sx = Math.min(Math.max((x + 0.5) * ratioX - 0.5, 0), outW - 1);
    const x0 = Math.floor(sx);
    x0Arr[x] = x0;
    x1Arr[x] = Math.min(x0 + 1, outW - 1);
    dxArr[x] = sx - x0;
  }

  for (let y = 0; y < WORK_SZ; y++) {
    const sy = Math.min(Math.max((y + 0.5) * ratioY - 0.5, 0), outH - 1);
    const y0 = Math.floor(sy);
    const y1 = Math.min(y0 + 1, outH - 1);
    const dy = sy - y0;
    const rowOff = y * WORK_SZ;
    const y0W = y0 * outW;
    const y1W = y1 * outW;

    for (let x = 0; x < WORK_SZ; x++) {
      const x0 = x0Arr[x];
      const x1 = x1Arr[x];
      const dx = dxArr[x];

      const idx00 = y0W + x0;
      const idx01 = y0W + x1;
      const idx10 = y1W + x0;
      const idx11 = y1W + x1;
      const w00 = (1 - dx) * (1 - dy);
      const w01 = dx * (1 - dy);
      const w10 = (1 - dx) * dy;
      const w11 = dx * dy;

      const lCopa =
        logits[idx00] * w00 + logits[idx01] * w01 + logits[idx10] * w10 + logits[idx11] * w11;
      const lWire =
        logits[chStride + idx00] * w00 +
        logits[chStride + idx01] * w01 +
        logits[chStride + idx10] * w10 +
        logits[chStride + idx11] * w11;

      const wi = rowOff + x;
      if (lCopa >= LOGIT_THRESH_COPA) rawCopa[wi] = 1;
      if (lWire >= LOGIT_THRESH_WIRE) rawWire[wi] = 1;
    }
  }

  // 3. Filtros morfológicos + distância euclidiana exata V3 em 640x640 (r1=6px, r2=24px)
  const cleanWire = filterSmallComponents(rawWire, WORK_SZ, WORK_SZ, 80);
  const cleanCopa = filterSmallComponents(rawCopa, WORK_SZ, WORK_SZ, 800, 0.15);

  const windZone = new Uint8Array(workN);

  let copaPx = 0;
  let wirePx = 0;
  let trueOverlapPx = 0;
  for (let i = 0; i < workN; i++) {
    if (cleanCopa[i]) copaPx += 1;
    if (cleanWire[i]) wirePx += 1;
    if (cleanCopa[i] && cleanWire[i]) trueOverlapPx += 1;
  }

  let contactDirectPx = 0;
  let windBufferPx = 0;
  if (wirePx > 0 && copaPx > 0) {
    const wireDistSq = exactSquaredDistanceTransform(cleanWire, WORK_SZ, WORK_SZ);
    const rDirectSq = 6.0 * 6.0; // r1 = 6 px @ 640px
    const rWindSq = 24.0 * 24.0; // r2 = 24 px @ 640px
    for (let i = 0; i < workN; i++) {
      if (cleanCopa[i] === 1) {
        const d2 = wireDistSq[i];
        if (d2 <= rWindSq) {
          // Zona visual preenchida por inteiro (0 <= d <= 24px), sem anel oco no centro
          windZone[i] = 1;
        }
        if (d2 <= rDirectSq) {
          contactDirectPx += 1;
        } else if (d2 <= rWindSq) {
          windBufferPx += 1;
        }
      }
    }
  }

  let triage: MacroTriage = "SEGURO";
  if (contactDirectPx >= 3000 && wirePx >= 800) {
    triage = "CRITICO";
  } else if ((contactDirectPx >= 250 || windBufferPx >= 1200) && wirePx >= 200) {
    triage = "PROXIMIDADE";
  }

  const tPost1 = performance.now();

  return {
    inferenceMs: Math.round(t1 - t0),
    postprocessMs: Math.round(tPost1 - tPost0),
    threadsUsed: threads,
    crossOriginIsolated: isolated,
    triage,
    copaPx,
    wirePx,
    trueOverlapPx,
    contactDirectPx,
    windBufferPx,
    masks: {
      width: WORK_SZ,
      height: WORK_SZ,
      copa: cleanCopa,
      wire: cleanWire,
      windZone,
    },
  };
}

function resampleNearestMask(
  src: Uint8Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): Uint8Array {
  const dst = new Uint8Array(dstW * dstH);
  const mapX = new Int32Array(dstW);
  for (let x = 0; x < dstW; x++) {
    mapX[x] = Math.min(srcW - 1, Math.floor(((x + 0.5) * srcW) / dstW));
  }
  for (let y = 0; y < dstH; y++) {
    const syOff = Math.min(srcH - 1, Math.floor(((y + 0.5) * srcH) / dstH)) * srcW;
    const dyOff = y * dstW;
    for (let x = 0; x < dstW; x++) {
      dst[dyOff + x] = src[syOff + mapX[x]];
    }
  }
  return dst;
}

function getContourOffsets(thickness: number): ReadonlyArray<readonly [number, number]> {
  const level = Math.max(0, Math.min(5, Math.round(thickness)));
  if (level <= 0) return [];
  if (level === 1) return [[0, 0]];
  if (level === 2) {
    return [
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ];
  }
  const offsets: Array<readonly [number, number]> = [];
  if (level === 3) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        offsets.push([dx, dy]);
      }
    }
    return offsets;
  }
  if (level === 4) {
    for (let dy = -1; dy <= 2; dy++) {
      for (let dx = -1; dx <= 2; dx++) {
        const cx = dx - 0.5;
        const cy = dy - 0.5;
        if (cx * cx + cy * cy <= 3.0) offsets.push([dx, dy]);
      }
    }
    return offsets;
  }
  // level === 5 (disco 5x5 sem os 4 cantos extremos -> 21 px)
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      if (dx * dx + dy * dy <= 5) offsets.push([dx, dy]);
    }
  }
  return offsets;
}

function paintContour(
  pixels: Uint8ClampedArray,
  mask: Uint8Array,
  w: number,
  h: number,
  rgb: readonly [number, number, number],
  thickness: number,
): void {
  const brush = getContourOffsets(thickness);
  if (brush.length === 0) return;

  const [cr, cg, cb] = rgb;
  for (let y = 0; y < h; y++) {
    const yOff = y * w;
    for (let x = 0; x < w; x++) {
      const idx = yOff + x;
      if (mask[idx] === 0) continue;

      const onBorder =
        x === 0 ||
        y === 0 ||
        x === w - 1 ||
        y === h - 1 ||
        mask[idx - 1] === 0 ||
        mask[idx + 1] === 0 ||
        mask[idx - w] === 0 ||
        mask[idx + w] === 0;

      if (!onBorder) continue;

      if (brush.length === 1) {
        const p = idx * 4;
        pixels[p] = cr;
        pixels[p + 1] = cg;
        pixels[p + 2] = cb;
        pixels[p + 3] = 255;
      } else {
        for (let k = 0; k < brush.length; k++) {
          const nx = x + brush[k][0];
          const ny = y + brush[k][1];
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          const p = (ny * w + nx) * 4;
          pixels[p] = cr;
          pixels[p + 1] = cg;
          pixels[p + 2] = cb;
          pixels[p + 3] = 255;
        }
      }
    }
  }
}

export function renderResultToCanvas(
  canvas: HTMLCanvasElement,
  imageEl: HTMLImageElement,
  result: VisionInferenceResult | null,
  layers: VisionLayerVisibility,
): void {
  // Preserva integralmente o aspect ratio da fotografia enviada
  const natW = imageEl.naturalWidth || IMGSZ;
  const natH = imageEl.naturalHeight || IMGSZ;
  const scale = IMGSZ / Math.max(natW, natH);
  const cw = Math.max(1, Math.round(natW * scale));
  const ch = Math.max(1, Math.round(natH * scale));

  canvas.width = cw;
  canvas.height = ch;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return;

  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(imageEl, 0, 0, cw, ch);

  if (!result) return;

  const { width, height, copa, wire, windZone } = result.masks;
  const mCopa = layers.copa ? resampleNearestMask(copa, width, height, cw, ch) : null;
  const mWire = layers.fiacao ? resampleNearestMask(wire, width, height, cw, ch) : null;
  const mWind = layers.vento ? resampleNearestMask(windZone, width, height, cw, ch) : null;

  // Interseção verdadeira = pixels em que Copa E Fiação estão ambas ativas
  const totalPixels = cw * ch;
  let mIntersection: Uint8Array | null = null;
  if (mCopa && mWire) {
    mIntersection = new Uint8Array(totalPixels);
    for (let i = 0; i < totalPixels; i++) {
      if (mCopa[i] && mWire[i]) {
        mIntersection[i] = 1;
      }
    }
  }

  const imgData = ctx.getImageData(0, 0, cw, ch);
  const pixels = imgData.data;
  const alpha = Math.min(1, Math.max(0, layers.opacity));
  const invAlpha = 1 - alpha;

  // 1. Mescla translúcida: Copa (verde), Margem de vento (laranja), Fiação (amarelo) e Interseção (vermelho)
  for (let i = 0; i < totalPixels; i++) {
    const isInter = Boolean(mIntersection && mIntersection[i]);
    let rgb: readonly [number, number, number] | null = null;
    if (isInter) {
      rgb = RGB_INTERSECTION;
    } else if (mWire && mWire[i]) {
      rgb = RGB_WIRE;
    } else if (mWind && mWind[i]) {
      rgb = RGB_WIND;
    } else if (mCopa && mCopa[i]) {
      rgb = RGB_COPA;
    }

    if (rgb) {
      const p = i * 4;
      pixels[p] = Math.round(pixels[p] * invAlpha + rgb[0] * alpha);
      pixels[p + 1] = Math.round(pixels[p + 1] * invAlpha + rgb[1] * alpha);
      pixels[p + 2] = Math.round(pixels[p + 2] * invAlpha + rgb[2] * alpha);
    }
  }

  // 2. Contornos configuráveis (0 = sem borda, 1..5 = espessura crescente uniforme)
  const borderLevel = Math.max(0, Math.min(5, Math.round(layers.borderThickness ?? 2)));
  if (borderLevel > 0) {
    if (mCopa) paintContour(pixels, mCopa, cw, ch, RGB_COPA, borderLevel);
    if (mWind) paintContour(pixels, mWind, cw, ch, RGB_WIND, borderLevel);
    if (mWire) paintContour(pixels, mWire, cw, ch, RGB_WIRE, borderLevel);
    if (mIntersection) paintContour(pixels, mIntersection, cw, ch, RGB_INTERSECTION, borderLevel);
  }

  ctx.putImageData(imgData, 0, 0);
}
